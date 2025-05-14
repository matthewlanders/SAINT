import random
import time
from dataclasses import dataclass, field
import os

import numpy as np
import torch
import torch.nn as nn
import tyro
from torch.distributions import Categorical
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.tensorboard import SummaryWriter

import gymnasium as gym
from envs.DiscretizedActionEnv import DiscretizedActionEnv


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "ppo_cityflow_cleanrl"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    runs_dir: str = "runs"

    gym_env: str = "HalfCheetah-v4"
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    num_steps: int = 4000
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 2
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = False
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = False
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    lr_decay_rate: float = 0.99995

    agent_hidden_size: int = 256
    """hidden size of the agent networks"""

    cityflow_config: str = "agents/cityflow/configs/Irregular/config.json"
    """Path to the CityFlow config file"""
    max_episode_steps: int = 1000
    """Maximum steps per episode in the CityFlow environment"""

    # Calculated arguments
    batch_size: int = field(init=False)
    minibatch_size: int = field(init=False)

    def __post_init__(self):
        self.batch_size = int(self.num_envs * self.num_steps)
        self.minibatch_size = int(self.batch_size // self.num_minibatches)
        if self.batch_size % self.num_minibatches != 0:
             raise ValueError(f"batch_size ({self.batch_size}) must be divisible by num_minibatches ({self.num_minibatches})")


class Agent(nn.Module):
    def __init__(self, state_size, action_dims, hidden_size=256):
        super().__init__()
        self.action_dims = action_dims
        self.num_intersections = len(action_dims)
        self.total_logit_dim = sum(action_dims)

        self.critic = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        self.actor_base = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

        self.actor_head = nn.Linear(hidden_size, self.total_logit_dim)

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, multi_action=None):
        actor_hidden = self.actor_base(x)
        all_logits = self.actor_head(actor_hidden)

        split_logits = torch.split(all_logits, self.action_dims, dim=1)

        actions_list = []
        log_probs_list = []
        entropies_list = []

        for i, logits_slot in enumerate(split_logits):
            dist = Categorical(logits=logits_slot)
            if multi_action is None:
                action_slot = dist.sample()
            else:
                action_slot = multi_action[:, i].long()

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


def train():
    obs = torch.zeros((args.num_steps, args.num_envs) + state_shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs, num_agents), dtype=torch.long).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    global_step = 0
    start_time = time.time()

    next_obs = torch.Tensor(envs.reset(seed=args.seed)).to(device).unsqueeze(0)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size

    print(f"Starting training for {args.total_timesteps} timesteps...")
    print(f"Batch size: {args.batch_size}, Mini-batch size: {args.minibatch_size}")

    for update in range(num_updates):
        current_episode_return = 0.0
        current_episode_length = 0

        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()

            actions[step] = action
            logprobs[step] = logprob

            action_np = action.squeeze(0).cpu().numpy()
            next_state_np, reward, done, info = envs.step(action_np)

            rewards[step] = torch.tensor(reward).to(device).view(-1)

            next_obs = torch.Tensor(next_state_np).to(device).unsqueeze(0)
            next_done = torch.Tensor([done]).to(device)

            current_episode_return += reward
            current_episode_length += 1

            if done:
                print(
                    f"global_step={global_step}, episodic_return={current_episode_return:.2f}, episodic_length={current_episode_length}")
                writer.add_scalar("learning/episodic_return", current_episode_return, global_step)
                writer.add_scalar("learning/episodic_length", current_episode_length, global_step)

                current_episode_return = 0.0
                current_episode_length = 0

                next_obs = torch.Tensor(envs.reset()).to(device).unsqueeze(0)

        with torch.no_grad():
            next_value = agent.get_value(next_obs.squeeze(0) if args.num_envs == 1 else next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        b_obs = obs.reshape((-1,) + state_shape)
        b_actions = actions.reshape((-1, num_agents))
        b_logprobs = logprobs.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # Calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

        if args.anneal_lr:
            scheduler.step()

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        sps = int(global_step / (time.time() - start_time))
        writer.add_scalar("charts/SPS", sps, global_step)
        print(f"SPS: {sps}")


if __name__ == "__main__":
    args = tyro.cli(Args)
    size = args.cityflow_config.split("/")[-2]
    run_name = f"Factored-{size}__{args.seed}__{int(time.time())}"

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

    base_env = gym.make(args.gym_env)
    envs = DiscretizedActionEnv(base_env, 11)

    state_shape = envs.observation_space.shape
    action_dims = list(envs.action_space.nvec)
    num_agents = len(action_dims)

    agent = Agent(state_shape[0], action_dims, args.agent_hidden_size).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    scheduler = ExponentialLR(optimizer, gamma=args.lr_decay_rate)

    train()

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.pt"
        torch.save(agent.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    writer.close()
