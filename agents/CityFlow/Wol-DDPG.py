import math
import os
import random
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import tyro
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from gymnasium import spaces
import wandb
import pynndescent

from envs.CityFlowEnv import CityFlowEnv


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    runs_dir: str = "runs"
    wandb_project_name: str = "wolpertinger_cityflow"
    wandb_entity: str = None
    save_model: bool = False

    cityflow_config: str = "agents/cityflow/configs/1x4/config.json"
    max_episode_steps: int = 1000

    total_timesteps: int = 500000
    actor_lr: float = 1e-4
    critic_lr: float = 1e-3
    buffer_size: int = 50000
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 128
    learning_starts: int = 1000
    train_frequency: int = 1

    action_embedding_dim: int = 32
    k_neighbors: int = 10
    exploration_noise_std: float = 0.1

    pynn_metric: str = 'euclidean'
    pynn_n_neighbors: int = 50

    actor_hidden_layers: int = 2
    critic_hidden_layers: int = 2
    hidden_size: int = 256

    action_dims: list = field(default_factory=list)
    num_intersections: int = 0
    joint_action_dim: int = 0
    state_dim: int = 0


_FACTORS_CACHE = {}


def _calculate_factors(action_dims: list):
    global _FACTORS_CACHE
    action_dims_tuple = tuple(action_dims)
    if action_dims_tuple not in _FACTORS_CACHE:
        factors = []
        prod = 1
        for ad in reversed(action_dims):
            factors.insert(0, prod)
            prod *= ad
        _FACTORS_CACHE[action_dims_tuple] = torch.tensor(factors, dtype=torch.long)
    return _FACTORS_CACHE[action_dims_tuple]


def decode_joint_action(joint_action_indices: torch.Tensor, action_dims: list, device: torch.device) -> torch.Tensor:
    if joint_action_indices.dim() == 0:
        joint_action_indices = joint_action_indices.unsqueeze(0)

    num_intersections = len(action_dims)
    factors = _calculate_factors(action_dims).to(device)

    batch_size = joint_action_indices.shape[0]
    multi_actions = torch.zeros((batch_size, num_intersections), dtype=torch.long, device=device)

    temp_indices = joint_action_indices.clone()
    for i in range(num_intersections):
        multi_actions[:, i] = temp_indices // factors[i]
        temp_indices = temp_indices % factors[i]

    return multi_actions


def encode_multi_action(multi_actions: torch.Tensor, action_dims: list, device: torch.device) -> torch.Tensor:
    if multi_actions.dim() == 1:
        multi_actions = multi_actions.unsqueeze(0)

    factors = _calculate_factors(action_dims).to(device)
    joint_indices = torch.sum(multi_actions * factors, dim=1)
    return joint_indices


def compute_cityflow_embeddings(action_dims: list, embedding_dim: int) -> np.ndarray:
    num_intersections = len(action_dims)
    joint_action_dim = math.prod(action_dims)
    print(f"Computing embeddings for {joint_action_dim} joint actions ({num_intersections} intersections)...")

    action_embeddings_np = np.zeros((joint_action_dim, embedding_dim), dtype=np.float32)
    joint_indices_tensor = torch.arange(joint_action_dim, dtype=torch.long)
    multi_actions_decoded = decode_joint_action(joint_indices_tensor, action_dims, torch.device('cpu'))

    for action_idx in range(joint_action_dim):
        multi_action_vec = multi_actions_decoded[action_idx].numpy().astype(np.float32)
        if embedding_dim >= num_intersections:
            action_embeddings_np[action_idx, :num_intersections] = multi_action_vec
        else:
            action_embeddings_np[action_idx, :] = multi_action_vec[:embedding_dim]
            if action_idx == 0:
                print(f"Warning: embedding_dim ({embedding_dim}) < num_intersections ({num_intersections}). Truncating multi-action vector.")

    print(f"CityFlow joint action embeddings computed (Shape: {action_embeddings_np.shape}).")
    return action_embeddings_np


class ActorNetwork(nn.Module):
    def __init__(self, state_size, action_embedding_dim, hidden_size=256, num_hidden_layers=2):
        super().__init__()
        layers = [nn.Linear(state_size, hidden_size), nn.ReLU()]
        for _ in range(num_hidden_layers - 1): layers.extend([nn.Linear(hidden_size, hidden_size), nn.ReLU()])
        layers.append(nn.Linear(hidden_size, action_embedding_dim))
        layers.append(nn.Tanh())
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        model_device = next(self.parameters()).device
        if not isinstance(x, torch.Tensor): x = torch.tensor(x, dtype=torch.float32, device=model_device)
        elif x.dtype != torch.float32: x = x.float()
        if x.device != model_device: x = x.to(model_device)
        if x.dim() > 2: x = x.squeeze(0)
        return self.network(x)


class CriticNetwork(nn.Module):
    def __init__(self, state_size, action_embedding_dim, hidden_size=256, num_hidden_layers=2):
        super().__init__()
        layers = [nn.Linear(state_size + action_embedding_dim, hidden_size), nn.ReLU()]
        for _ in range(num_hidden_layers - 1):
            layers.extend([nn.Linear(hidden_size, hidden_size), nn.ReLU()])
        layers.append(nn.Linear(hidden_size, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, state, action_embedding):
        state = state.float()
        action_embedding = action_embedding.float()
        x = torch.cat([state, action_embedding], dim=1)
        return self.network(x)


def learn(args: Args, env: CityFlowEnv, rb: ReplayBuffer, actor: ActorNetwork, critic: CriticNetwork,
          target_actor: ActorNetwork, target_critic: CriticNetwork, actor_optimizer: optim.Optimizer,
          critic_optimizer: optim.Optimizer, action_embeddings_tensor: torch.Tensor,
          pynndescent_index: pynndescent.NNDescent, device: torch.device, writer: SummaryWriter):

    obs = env.reset()
    noise_dist = torch.distributions.Normal(0, args.exploration_noise_std)
    episode_returns = []
    episode_lengths = []
    current_episode_return = 0.0
    current_episode_length = 0

    print(f"Starting training loop for {args.total_timesteps} timesteps...")

    for global_step in range(args.total_timesteps):
        if global_step < args.learning_starts:
            joint_action_idx_selected = np.random.randint(args.joint_action_dim)
        else:
            with torch.no_grad():
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)

                if obs_tensor.dim() == 1:
                    obs_tensor = obs_tensor.unsqueeze(0)
                elif obs_tensor.dim() > 2:
                    obs_tensor = obs_tensor.view(1, -1)

                proto_action = actor(obs_tensor).squeeze(0)
                noise = noise_dist.sample((args.action_embedding_dim,)).to(device)
                proto_action_noisy = torch.clamp(proto_action + noise, -1, 1)
                proto_action_noisy_np = proto_action_noisy.unsqueeze(0).cpu().numpy().astype(np.float32)
                neighbor_indices_np, _ = pynndescent_index.query(proto_action_noisy_np, k=args.k_neighbors)
                neighbor_joint_indices = torch.from_numpy(neighbor_indices_np[0]).long().to(device)

                num_found_neighbors = neighbor_joint_indices.shape[0]
                if num_found_neighbors > 0:
                    neighbor_embeddings = action_embeddings_tensor[neighbor_joint_indices]

                    state_repeated = obs_tensor.repeat(num_found_neighbors, 1)
                    q_values_neighbors = critic(state_repeated,neighbor_embeddings)
                    best_neighbor_idx_in_k = torch.argmax(q_values_neighbors).item()
                    joint_action_idx_selected = neighbor_joint_indices[best_neighbor_idx_in_k].item()
                else:
                    print(f"Warning: PyNNDescent found 0 neighbors (k={args.k_neighbors}). Using random action.")
                    joint_action_idx_selected = np.random.randint(args.joint_action_dim)

        joint_action_tensor = torch.tensor([joint_action_idx_selected], dtype=torch.long, device=device)
        multi_action_tensor = decode_joint_action(joint_action_tensor, args.action_dims, device)
        env_action = multi_action_tensor.squeeze(0).cpu().numpy()

        next_obs, reward, done, info = env.step(env_action)

        current_episode_return += reward
        current_episode_length += 1
        if done:
            print(
                f"global_step={global_step}, episodic_return={current_episode_return:.2f}, "
                f"episodic_length={current_episode_length}")
            writer.add_scalar("learning/episodic_return", current_episode_return, global_step)
            writer.add_scalar("learning/episodic_length", current_episode_length, global_step)

            episode_returns.append(current_episode_return)
            episode_lengths.append(current_episode_length)
            current_episode_return = 0.0
            current_episode_length = 0
            next_obs = env.reset()

        action_to_store = np.array([joint_action_idx_selected], dtype=np.int64)
        rb.add(obs, next_obs, action_to_store, reward, done, [info])
        obs = next_obs

        if global_step >= args.learning_starts and global_step % args.train_frequency == 0:
            data = rb.sample(args.batch_size)
            observations = data.observations.float().to(device)
            next_observations = data.next_observations.float().to(device)
            batch_rewards = data.rewards.float().to(device)
            batch_dones = data.dones.float().to(device)
            stored_joint_action_indices = data.actions.long().squeeze(1).to(device)

            with torch.no_grad():
                next_proto_actions_target = target_actor(next_observations)
                next_proto_actions_target_np = next_proto_actions_target.cpu().numpy().astype(np.float32)
                neighbor_indices_batch_np, _ = pynndescent_index.query(next_proto_actions_target_np, k=args.k_neighbors)
                neighbor_joint_indices_batch = torch.from_numpy(neighbor_indices_batch_np).long().to(device)
                next_obs_repeated = next_observations.unsqueeze(1).repeat(1, args.k_neighbors, 1).view(-1, args.state_dim)
                flat_neighbor_joint_indices = neighbor_joint_indices_batch.reshape(-1)
                all_neighbor_embeddings_flat = action_embeddings_tensor[flat_neighbor_joint_indices]
                q_values_neighbors_batch_flat = target_critic(next_obs_repeated, all_neighbor_embeddings_flat)
                q_values_neighbors_batch = q_values_neighbors_batch_flat.view(args.batch_size, args.k_neighbors)
                max_q_next_target, _ = torch.max(q_values_neighbors_batch, dim=1, keepdim=True)
                td_target = batch_rewards + args.gamma * max_q_next_target * (1 - batch_dones)

            action_embeddings_batch = action_embeddings_tensor[stored_joint_action_indices]
            q_current = critic(observations, action_embeddings_batch)
            critic_loss = F.mse_loss(q_current, td_target)

            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            for param in critic.parameters():
                param.requires_grad = False
            proto_actions_pred = actor(observations)
            actor_q_values = critic(observations, proto_actions_pred)
            actor_loss = -actor_q_values.mean()
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()
            for param in critic.parameters():
                param.requires_grad = True

            for target_param, local_param in zip(target_critic.parameters(), critic.parameters()):
                target_param.data.copy_(args.tau * local_param.data + (1.0 - args.tau) * target_param.data)
            for target_param, local_param in zip(target_actor.parameters(), actor.parameters()):
                target_param.data.copy_(args.tau * local_param.data + (1.0 - args.tau) * target_param.data)

            if global_step % 100 == 0:
                writer.add_scalar("losses/critic_loss", critic_loss.item(), global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/mean_q_current", q_current.mean().item(), global_step)


if __name__ == "__main__":
    args = tyro.cli(Args)

    size_info = args.cityflow_config.split('/')[-2] if '/' in args.cityflow_config else "unknown_size"
    run_name = f"Wolpertinger_{size_info}__{args.seed}__{int(time.time())}"
    args.exp_name = f"Wolpertinger_{size_info}"

    if args.track:
        wandb.init(project=args.wandb_project_name, entity=args.wandb_entity, sync_tensorboard=True,
                   config=vars(args), name=run_name, monitor_gym=True, save_code=True)
    writer = SummaryWriter(f"{args.runs_dir}/{run_name}")
    writer.add_text("hyperparameters", "|param|value|\n|-|-|\n%s"
                 % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Using device: {device}")

    env = CityFlowEnv(configPath=args.cityflow_config, episodeSteps=args.max_episode_steps)
    args.state_dim = env.observation_space.shape[0]
    args.action_dims = list(env.action_space.nvec)
    args.num_intersections = len(args.action_dims)
    args.joint_action_dim = math.prod(args.action_dims)
    buffer_action_space = spaces.Discrete(args.joint_action_dim)
    print(f"Environment: State Dim={args.state_dim}, Num Intersections={args.num_intersections}, Joint Actions={args.joint_action_dim}")

    actor = ActorNetwork(args.state_dim, args.action_embedding_dim, args.hidden_size, args.actor_hidden_layers).to(device)
    critic = CriticNetwork(args.state_dim, args.action_embedding_dim, args.hidden_size, args.critic_hidden_layers).to(device)
    target_actor = ActorNetwork(args.state_dim, args.action_embedding_dim, args.hidden_size, args.actor_hidden_layers).to(device)
    target_critic = CriticNetwork(args.state_dim, args.action_embedding_dim, args.hidden_size, args.critic_hidden_layers).to(device)
    target_actor.load_state_dict(actor.state_dict())
    target_critic.load_state_dict(critic.state_dict())
    target_actor.eval()
    target_critic.eval()
    actor_optimizer = optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic_optimizer = optim.Adam(critic.parameters(), lr=args.critic_lr)

    rb = ReplayBuffer(
        args.buffer_size,
        env.observation_space,
        buffer_action_space,
        device=device,
        handle_timeout_termination=False,
        n_envs=1)

    action_embeddings_np = compute_cityflow_embeddings(args.action_dims, args.action_embedding_dim)
    action_embeddings_tensor = torch.tensor(action_embeddings_np, dtype=torch.float32).to(device)
    print(f"Building PyNNDescent index (metric={args.pynn_metric}, n_neighbors={args.pynn_n_neighbors})...")
    pynndescent_index = pynndescent.NNDescent(
        data=action_embeddings_np, metric=args.pynn_metric, n_neighbors=args.pynn_n_neighbors,
        random_state=args.seed, verbose=True)
    pynndescent_index.prepare()
    print("PyNNDescent index built and prepared.")

    learn(args, env, rb, actor, critic, target_actor, target_critic, actor_optimizer, critic_optimizer,
          action_embeddings_tensor, pynndescent_index, device, writer)

    env.close()
    writer.close()
    if args.track:
        wandb.finish()
