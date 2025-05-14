import os
import random
import time
from collections import namedtuple
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
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

    agent_hidden_size: int = 256
    """hidden size of the agent networks"""
    num_heads: int = 4
    num_attention_blocks: int = 1
    num_attention_ffn_layers: int = 2
    num_decision_layers: int = 4
    ln: bool = True
    film_generator_num_layers: int = 2
    generator_hidden_dim: int = 128


Transition = namedtuple("Transition", ["state", "action", "reward", "terminated", "value"])
FullTransition = namedtuple("FullTransition", Transition._fields + ("return_", "advantage"))


class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, num_hidden_layers=2, ln=False):
        super(MAB, self).__init__()
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)

        if dim_V % num_heads != 0:
             raise ValueError(f"dim_V ({dim_V}) must be divisible by num_heads ({num_heads})")
        self.dim_split = dim_V // num_heads

        fc_o_layers = []
        hidden_dim = dim_V * 4
        current_dim = dim_V
        if num_hidden_layers > 0:
            fc_o_layers.append(nn.Linear(current_dim, hidden_dim))
            fc_o_layers.append(nn.ReLU())
            current_dim = hidden_dim

            for _ in range(num_hidden_layers - 1):
                fc_o_layers.append(nn.Linear(current_dim, hidden_dim))
                fc_o_layers.append(nn.ReLU())
            fc_o_layers.append(nn.Linear(current_dim, dim_V))
        else:
            fc_o_layers.append(nn.Linear(dim_V, dim_V))

        self.fc_o = nn.Sequential(*fc_o_layers)

        self.ln0 = nn.LayerNorm(dim_V) if ln else nn.Identity()
        self.ln1 = nn.LayerNorm(dim_V) if ln else nn.Identity()

    def forward(self, Q, K):
        B, n, _ = Q.size()
        _, m, _ = K.size()

        Q_ = self.fc_q(Q)
        K_ = self.fc_k(K)
        V_ = self.fc_v(K)
        Q_orig = Q_

        Q_ = torch.cat(Q_.split(self.dim_split, 2), 0)
        K_ = torch.cat(K_.split(self.dim_split, 2), 0)
        V_ = torch.cat(V_.split(self.dim_split, 2), 0)

        A = torch.matmul(Q_, K_.transpose(1, 2)) / (self.dim_split ** 0.5)
        A = torch.softmax(A, -1)

        O = torch.matmul(A, V_)
        O = torch.cat(O.split(B, 0), -1)
        O = self.ln0(O + Q_orig)
        O_ff = self.fc_o(O)
        O = O + O_ff
        O = self.ln1(O)

        return O


class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_attention_ffn_layers=2, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, num_hidden_layers=num_attention_ffn_layers, ln=ln)

    def forward(self, X):
        return self.mab(X, X)


def build_mlp(in_features, out_features, num_layers=2, hidden_dim=256):
    layers = []
    current_dim = in_features
    if num_layers == 1:
         layers.append(nn.Linear(current_dim, out_features))
    elif num_layers > 1:
        layers.append(nn.Linear(current_dim, hidden_dim))
        layers.append(nn.ReLU())
        current_dim = hidden_dim
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(current_dim, out_features))
    else:
         raise ValueError("num_decision_layers must be >= 1")

    return nn.Sequential(*layers)


class FiLMLayer(nn.Module):
    def __init__(self,
                 context_dim,
                 target_feature_dim,
                 generator_num_layers=2,
                 generator_hidden_dim=256):
        super().__init__()
        self.target_feature_dim = target_feature_dim

        self.generator = build_mlp(
            in_features=context_dim,
            out_features=2 * target_feature_dim,  # Output gamma and beta
            num_layers=generator_num_layers,
            hidden_dim=generator_hidden_dim
        )

    def forward(self, target_features, context):
        params = self.generator(context)

        gamma = params[:, :self.target_feature_dim]
        beta = params[:, self.target_feature_dim:]

        num_extra_dims = target_features.dim() - 2

        if num_extra_dims < 0:
             num_extra_dims = 0

        reshape_dims = (context.shape[0],) + (1,) * num_extra_dims + (self.target_feature_dim,)

        gamma = gamma.view(*reshape_dims)
        beta = beta.view(*reshape_dims)

        conditioned_features = gamma * target_features + beta
        return conditioned_features


class Agent(nn.Module):
    def __init__(
        self,
        state_size,
        action_dims,
        critic_hidden_size=256,
        d_model=128,
        num_heads=4,
        film_generator_num_layers=2,
        film_generator_hidden_dim=256,
        num_attention_blocks=2,
        num_attention_ffn_layers=2,
        ln=False,
        decision_mlp_layers=2,
        decision_mlp_hidden_size=256
    ):
        super().__init__()
        self.state_size = state_size
        self.action_dims = action_dims
        self.action_size = len(action_dims)
        self.d_model = d_model

        self.critic = nn.Sequential(
            nn.Linear(state_size, critic_hidden_size),
            nn.ReLU(),
            nn.Linear(critic_hidden_size, 1)
        )

        self.action_slot_embedding = nn.Embedding(self.action_size, d_model)

        self.film_layer = FiLMLayer(
            context_dim=state_size,
            target_feature_dim=d_model,
            generator_num_layers=film_generator_num_layers,
            generator_hidden_dim=film_generator_hidden_dim
        )

        sab_layers = []
        current_dim = d_model
        for _ in range(num_attention_blocks):
            sab_layers.append(SAB(
                dim_in=current_dim,
                dim_out=d_model,
                num_heads=num_heads,
                num_attention_ffn_layers=num_attention_ffn_layers,
                ln=ln
                ))
            current_dim = d_model
        self.set_attention = nn.Sequential(*sab_layers)

        decision_in_features = d_model
        self.decision_layers = nn.ModuleList()
        for ad in action_dims:
            self.decision_layers.append(
                build_mlp(
                    decision_in_features,
                    ad,
                    num_layers=decision_mlp_layers,
                    hidden_dim=decision_mlp_hidden_size
                )
            )

    def get_value(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.critic(x)

    def get_action_and_value(self, x, multi_action=None):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        B = x.shape[0]

        slot_indices = torch.arange(self.action_size, device=x.device).unsqueeze(0).expand(B, -1)
        slot_tokens = self.action_slot_embedding(slot_indices)

        conditioned_slot_tokens = self.film_layer(slot_tokens, x)
        processed_slot_tokens = self.set_attention(conditioned_slot_tokens)

        slot_representation = processed_slot_tokens

        decision_mlp_input = slot_representation

        actions_list = []
        log_probs_list = []
        entropies_list = []

        for i in range(self.action_size):
            slot_input = decision_mlp_input[:, i, :]
            logits_slot = self.decision_layers[i](slot_input)

            dist = Categorical(logits=logits_slot)
            if multi_action is None:
                action_slot = dist.sample()
            else:
                action_slot = multi_action[:, i].long().to(logits_slot.device)

            log_prob_slot = dist.log_prob(action_slot)
            entropy_slot = dist.entropy()

            actions_list.append(action_slot)
            log_probs_list.append(log_prob_slot)
            entropies_list.append(entropy_slot)

        final_actions = torch.stack(actions_list, dim=1)
        summed_log_prob = torch.stack(log_probs_list, dim=1).sum(dim=1)
        summed_entropy = torch.stack(entropies_list, dim=1).sum(dim=1)

        value = self.critic(x)

        return final_actions, summed_log_prob, summed_entropy, value


class Actor(Agent):
    def __init__(self, state_size: int, action_size: int, d_model, num_heads, critic_hidden_size,
                 film_generator_num_layers, film_generator_hidden_dim, num_attention_blocks, num_attention_ffn_layers,
                 ln, decision_mlp_layers, decision_mlp_hidden_size):
        super().__init__(
            state_size=state_size,
            action_dims=[2] * action_size,
            d_model=d_model,
            num_heads=num_heads,
            critic_hidden_size=critic_hidden_size,
            film_generator_num_layers=film_generator_num_layers,
            film_generator_hidden_dim=film_generator_hidden_dim,
            num_attention_blocks=num_attention_blocks,
            num_attention_ffn_layers=num_attention_ffn_layers,
            ln=ln,
            decision_mlp_layers=decision_mlp_layers,
            decision_mlp_hidden_size=decision_mlp_hidden_size
        )

    def forward(self, state: torch.Tensor):
        actions, logp, _, _ = self.get_action_and_value(state)
        return actions.float(), logp

    @torch.no_grad()
    def get_action(self, state: torch.Tensor):
        actions, _ = self.forward(state)
        return actions

    def get_log_prob_of_action(self, state, actions):
        _, logp, _, _ = self.get_action_and_value(state, multi_action=actions.long())
        return logp

    def get_entropy(self, state):
        _, _, entropy, _ = self.get_action_and_value(state)
        return entropy


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
                actions_binary_t, log_prob_t = actor.forward(state_t)
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
        new_log_probs = actor.get_log_prob_of_action(all_states, all_actions)
        entropy_batch = actor.get_entropy(all_states)

        pg_loss = -(new_log_probs * advantages_norm).mean()
        entropy_loss = -entropy_batch.mean()
        actor_loss = pg_loss + ent_coef * entropy_loss
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
        writer.add_scalar("losses/entropy_loss", entropy_loss.item(), global_step)
        writer.add_scalar("losses/value_loss", value_loss_unscaled.item(), global_step)
        writer.add_scalar("charts/entropy", entropy_batch.mean().item(), global_step)
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

    actor = Actor(
        state_size=state_size,
        action_size=binary_action_size,
        d_model=128,
        num_heads=args.num_heads,
        critic_hidden_size=args.agent_hidden_size,
        film_generator_num_layers=args.film_generator_num_layers,
        film_generator_hidden_dim=args.generator_hidden_dim,
        num_attention_blocks=args.num_attention_blocks,
        num_attention_ffn_layers=args.num_attention_ffn_layers,
        ln=args.ln,
        decision_mlp_layers=args.num_decision_layers,
        decision_mlp_hidden_size=args.agent_hidden_size
    )

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

