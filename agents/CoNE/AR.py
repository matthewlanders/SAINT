import os, random, time
from collections import namedtuple
from dataclasses import dataclass

import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
import tyro
from torch.distributions import Categorical
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.tensorboard import SummaryWriter

from envs.CoNE import CoNE


@dataclass
class Args:
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    runs_dir: str = "runs"
    """directory into which run data will be stored"""
    wandb_project_name: str = "CoNE-A2C-GAE"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    save_model: bool = False
    """whether to save model"""
    randomize_initial_state: bool = False
    """if toggled, agent will start in random (non-terminal) grid location"""
    data_load_path: str = "MUST_PROVIDE_PATH"
    """file path for offline data to be loaded (used to infer env params)"""
    small_state: bool = True

    total_timesteps: int = 50000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for Generalized Advantage Estimation"""
    rollout_steps: int = 128
    """number of steps to collect per policy update"""
    ent_coef: float = 0.01
    """Entropy coefficient for exploration bonus in the actor loss."""
    vf_coef: float = 0.5
    """How much to scale the value loss relative to the policy gradient loss."""
    max_grad_norm: float = 0.5
    """Clip gradients above this norm for both actor and value networks."""
    lr_decay_rate: float = 0.99995
    max_steps_per_episode: int = 100


Transition = namedtuple("Transition", ["state", "action", "reward", "terminated", "value"])
FullTransition = namedtuple("FullTransition", Transition._fields + ("return_", "advantage"))


class Actor(nn.Module):
    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_size: int = 256,
        lstm_hidden_size: int = 256,
        action_emb_dim: int = 32,
        lstm_hidden_layers=2,
    ):
        super().__init__()
        self.d = action_size
        self.start_tok = 2
        self.lstm_num_layers = lstm_hidden_layers
        self.lstm_hidden_size = lstm_hidden_size

        self.state_emb = nn.Sequential(
            nn.Linear(state_size, hidden_size), nn.ReLU()
        )
        self.action_emb = nn.Embedding(self.start_tok + 1, action_emb_dim)
        self.lstm = nn.LSTM(
            hidden_size + action_emb_dim, lstm_hidden_size, batch_first=True, num_layers=lstm_hidden_layers
        )
        self.heads = nn.ModuleList(
            [nn.Linear(lstm_hidden_size, 2) for _ in range(self.d)]
        )
        self.lstm_h = lstm_hidden_size

    def _core(self, state, given_actions=None):
        if state.dim() == 1:
            state = state.unsqueeze(0)
        B, dev = state.size(0), state.device
        s_emb = self.state_emb(state)

        h = torch.zeros(self.lstm_num_layers, B, self.lstm_hidden_size, device=device)
        c = torch.zeros(self.lstm_num_layers, B, self.lstm_hidden_size, device=device)
        hidden = (h, c)

        prev = self.action_emb(
            torch.full((B, 1), self.start_tok, device=dev)
        ).squeeze(1)

        tot_lp = torch.zeros(B, device=dev)
        tot_ent = torch.zeros_like(tot_lp)
        actions, prefix_lps, per_ent = [], [], []

        provided = (
            list(torch.split(given_actions.long(), 1, dim=1))
            if given_actions is not None
            else None
        )

        for i, head in enumerate(self.heads):
            prefix_lps.append(tot_lp)

            out, hidden = self.lstm(
                torch.cat([s_emb, prev], 1).unsqueeze(1), hidden
            )
            logits = head(out.squeeze(1))
            dist = Categorical(logits=logits)

            a_i = provided[i].squeeze(1) if provided else dist.sample()
            lp_i = dist.log_prob(a_i)
            ent_i = dist.entropy()

            tot_lp += lp_i
            tot_ent += ent_i
            per_ent.append(ent_i.detach())
            prev = self.action_emb(a_i)

            if not provided:
                actions.append(a_i.unsqueeze(1))

        act_out = torch.cat(actions, 1) if provided is None else given_actions
        return act_out, tot_lp, tot_ent, torch.stack(prefix_lps, 1), torch.stack(per_ent, 1)

    @torch.no_grad()
    def act(self, state):
        actions, logp, *_ = self._core(state)
        return actions, logp

    def evaluate_actions(self, states, actions):

        _, lp, ent, prefix_lp, per_ent = self._core(states, given_actions=actions)
        return lp, ent, prefix_lp, per_ent


class Value(nn.Module):
    def __init__(self, state_size):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.network(x)


def compute_returns_and_advantages_gae(
    transitions: list[Transition],
    next_value: float,
    gamma: float,
    gae_lambda: float
) -> list[FullTransition]:

    advantages = []
    last_gae_lam = 0.0

    for t in reversed(range(len(transitions))):
        reward = transitions[t].reward
        current_value = transitions[t].value
        is_terminal = transitions[t].terminated
        delta = reward + gamma * next_value * (1.0 - float(is_terminal)) - current_value
        last_gae_lam = delta + gamma * gae_lambda * (1.0 - float(is_terminal)) * last_gae_lam
        advantages.insert(0, last_gae_lam)
        next_value = current_value

    returns = [adv + trans.value for adv, trans in zip(advantages, transitions)]
    full_transitions = [FullTransition(*trans, return_=ret, advantage=adv)
                        for trans, ret, adv in zip(transitions, returns, advantages)]

    return full_transitions


def train(env, actor, value_net, actor_optimizer, value_optimizer, a_scheduler, v_scheduler, args, device, writer):
    global_step = 0
    gamma = args.gamma
    gae_lambda = args.gae_lambda
    rollout_steps = args.rollout_steps
    ent_coef = args.ent_coef
    vf_coef = args.vf_coef
    max_grad_norm = args.max_grad_norm
    total_timesteps = args.total_timesteps

    state, info = env.reset()

    while global_step < total_timesteps:
        transitions = []

        for step in range(rollout_steps):
            state_t = torch.as_tensor(state, dtype=torch.float32).to(device)
            with torch.no_grad():
                actions_binary_t, _ = actor.act(state_t)
                value = value_net(state_t).item()

            actions_np = actions_binary_t.squeeze(0).cpu().numpy()
            next_state, reward, terminated, truncated, info = env.step(actions_np)

            transitions.append(
                Transition(
                    state=state,
                    action=actions_np,
                    reward=float(reward),
                    terminated=terminated,
                    value=value
                )
            )

            state = next_state
            global_step += 1

            done = terminated or truncated
            if done:
                if "episode" in info:
                    ep_info = info['episode']
                    print(f"global_step={global_step}, episodic_return={ep_info['r']:.2f}, length={ep_info['l']}")
                    writer.add_scalar("learning/episodic_return", ep_info["r"], global_step)
                    writer.add_scalar("learning/episodic_length", ep_info["l"], global_step)
                elif "final_info" in info and isinstance(info["final_info"], dict) and 'episode' in info["final_info"]:
                    ep_info = info['final_info']['episode']
                    print(f"global_step={global_step}, episodic_return={ep_info['r']:.2f}, length={ep_info['l']}")
                    writer.add_scalar("learning/episodic_return", ep_info["r"], global_step)
                    writer.add_scalar("learning/episodic_length", ep_info["l"], global_step)

                state, info = env.reset()

            if global_step >= total_timesteps:
                break

        if not transitions: continue

        with torch.no_grad():
             last_terminated = transitions[-1].terminated
             if not last_terminated:
                 next_state_t = torch.as_tensor(state, dtype=torch.float32).to(device)
                 next_value = value_net(next_state_t).item()
             else:
                 next_value = 0.0

        full_transitions = compute_returns_and_advantages_gae(
            transitions, next_value, gamma, gae_lambda
        )
        if not full_transitions: continue

        all_states = torch.as_tensor(np.array([tr.state for tr in full_transitions]), dtype=torch.float32).to(device)
        all_actions = torch.as_tensor(np.array([tr.action for tr in full_transitions]), dtype=torch.float32).to(device)
        all_advantages = torch.as_tensor([tr.advantage for tr in full_transitions], dtype=torch.float32).to(device)
        all_returns = torch.as_tensor([tr.return_ for tr in full_transitions], dtype=torch.float32).to(device)

        if len(all_advantages) > 1:
            advantages_norm = (all_advantages - all_advantages.mean()) / (all_advantages.std() + 1e-8)
        else:
            advantages_norm = all_advantages

        actor_optimizer.zero_grad()
        new_log_probs, entropy_b, prefix_lp, per_ent = actor.evaluate_actions(all_states, all_actions)

        pg_loss = -(new_log_probs * advantages_norm).mean()

        ent_corr = (per_ent * prefix_lp).sum(1).mean()
        total_entropy = entropy_b.mean() + ent_corr

        actor_loss = pg_loss - ent_coef * total_entropy
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
        actor_optimizer.step()
        if a_scheduler: a_scheduler.step()

        value_optimizer.zero_grad()
        predicted_values = value_net(all_states).squeeze(-1)
        value_loss_unscaled = F.mse_loss(predicted_values, all_returns)
        value_loss = vf_coef * value_loss_unscaled
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(value_net.parameters(), max_grad_norm)
        value_optimizer.step()
        if v_scheduler: v_scheduler.step()

        writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/value_loss", value_loss_unscaled.item(), global_step)
        writer.add_scalar("charts/entropy", total_entropy.mean().item(), global_step)
        writer.add_scalar("charts/advantage_mean", all_advantages.mean().item(), global_step)
        writer.add_scalar("charts/return_mean", all_returns.mean().item(), global_step)
        if a_scheduler:
            writer.add_scalar("learning_rate/actor_lr", actor_optimizer.param_groups[0]['lr'], global_step)
        if v_scheduler:
            writer.add_scalar("learning_rate/value_lr", value_optimizer.param_groups[0]['lr'], global_step)

        if global_step >= total_timesteps:
            break

    writer.close()


if __name__ == "__main__":
    args = tyro.cli(Args)

    if not args.data_load_path or args.data_load_path == "MUST_PROVIDE_PATH":
        raise ValueError("Please provide a valid --data-load-path argument.")
    try:
        parts = args.data_load_path.split('/')[-1].split('-', 4)
        grid_dimension, grid_size, num_pits = map(int, parts[:3])
        suboptimal_rate = parts[3]
    except Exception as e:
        raise ValueError(f"Bad --data-load-path format: {e}")

    exp_name = f"{grid_dimension}-{grid_size}-{num_pits}-{suboptimal_rate}-AR"
    run_name = f"{exp_name}__{args.seed}__{int(time.time())}"

    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"{args.runs_dir}/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    env = CoNE(
        grid_dimension=grid_dimension,
        grid_size=grid_size,
        num_total_pits=num_pits,
        num_clusters=1,
        distribute_pits_evenly=True,
        small_state=args.small_state,
        max_steps_per_episode=args.max_steps_per_episode,
        randomize_initial_state=args.randomize_initial_state,
        load_terminal_states=True,
    )

    state_size = env.grid_dimension if args.small_state else len(env.terminal_states) + env.grid_dimension
    binary_action_size = 2 * grid_dimension

    actor = Actor(state_size, binary_action_size).to(device)
    value_net = Value(state_size).to(device)

    actor_opt = torch.optim.Adam(actor.parameters(), lr=args.learning_rate, eps=1e-5)
    value_opt = torch.optim.Adam(value_net.parameters(), lr=args.learning_rate, eps=1e-5)
    a_sched = ExponentialLR(actor_opt, gamma=args.lr_decay_rate) if args.lr_decay_rate < 1.0 else None
    v_sched = ExponentialLR(value_opt, gamma=args.lr_decay_rate) if args.lr_decay_rate < 1.0 else None

    train(
        env=env,
        actor=actor,
        value_net=value_net,
        actor_optimizer=actor_opt,
        value_optimizer=value_opt,
        a_scheduler=a_sched,
        v_scheduler=v_sched,
        args=args,
        device=device,
        writer=writer
    )

    writer.close()
