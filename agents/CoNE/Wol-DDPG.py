import random
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import tyro
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from gymnasium import spaces
import faiss

from envs.CoNE import CoNE


@dataclass
class Args:
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    runs_dir: str = "runs"
    wandb_project_name: str = "faa"
    wandb_entity: str = None
    save_model: bool = False
    randomize_initial_state: bool = False
    data_load_path: str = None 
    small_state: bool = False
    suboptimal_rate: float = 0.9
    max_steps_per_episode: int = 100

    total_timesteps: int = 50000
    actor_lr: float = 1e-4
    critic_lr: float = 1e-3
    buffer_size: int = 10000
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 128
    learning_starts: int = 1000
    train_frequency: int = 10

    action_embedding_dim: int = 16
    k_neighbors: int = 10
    exploration_noise_std: float = 0.1

    pynn_metric: str = 'euclidean'
    pynn_n_neighbors: int = 50

    actor_hidden_layers: int = 2
    critic_hidden_layers: int = 2
    hidden_size: int = 256


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


def compute_displacement_embeddings(num_actions: int, grid_dimension: int, embedding_dim: int) -> np.ndarray:
    """
    (unchanged)
    """
    num_basic_moves = 2 * grid_dimension
    if num_actions != (2 ** num_basic_moves):
        raise ValueError(f"num_actions ({num_actions}) does not match 2**(2*grid_dimension) "
                         f"where grid_dimension={grid_dimension} (expected {2 ** num_basic_moves})")

    print(
        f"Computing displacement embeddings for {num_actions} actions, {grid_dimension}D grid, target dim {embedding_dim}.")

    action_embeddings_np = np.zeros((num_actions, embedding_dim), dtype=np.float32)
    displacement = np.zeros(grid_dimension, dtype=np.float32)

    for action_idx in range(num_actions):
        displacement.fill(0)
        temp_idx = action_idx
        for basic_move_idx in range(num_basic_moves):
            if (temp_idx % 2) == 1:
                dim = basic_move_idx // 2
                sign = 1 - 2 * (basic_move_idx % 2)
                displacement[dim] += sign
            temp_idx //= 2
        if embedding_dim >= grid_dimension:
            action_embeddings_np[action_idx, :grid_dimension] = displacement
        else:
            action_embeddings_np[action_idx, :] = displacement[:embedding_dim]
            if action_idx == 0:
                print(f"Warning: embedding_dim ({embedding_dim}) < grid_dimension ({grid_dimension}). "
                      f"Truncating displacement vector. Consider projection.")

    print("Displacement embeddings computed.")
    return action_embeddings_np


def gpu_search_in_batches(index, queries_np, k, batch_size=8):
    all_D, all_I = [], []
    for i in range(0, len(queries_np), batch_size):
        sub = queries_np[i : i+batch_size]
        D, I = index.search(sub, k)
        all_D.append(D); all_I.append(I)
    return np.vstack(all_D), np.vstack(all_I)


def train(args, env, rb, actor, critic, target_actor, target_critic, actor_optimizer, critic_optimizer,
          action_embeddings_tensor, faiss_index, state_size, device, writer):

    obs = env.reset()
    num_actions_total = action_embeddings_tensor.shape[0]
    k = min(args.k_neighbors, num_actions_total)
    noise_dist = torch.distributions.Normal(0, args.exploration_noise_std)

    for global_step in range(args.total_timesteps):
        if global_step < args.learning_starts:
            action_idx_rb = np.random.randint(num_actions_total)
        else:
            with torch.no_grad():
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                proto_action = actor(obs_tensor).squeeze(0)
                noise = noise_dist.sample((args.action_embedding_dim,)).to(device)
                proto_noisy = torch.clamp(proto_action + noise, -1, 1)
                proto_noisy_np = proto_noisy.unsqueeze(0).cpu().numpy().astype(np.float32)

                
                D, I = faiss_index.search(proto_noisy_np, k)
                neighbor_original_indices = torch.from_numpy(I[0]).long().to(device)
                

                num_found = neighbor_original_indices.shape[0]
                neighbor_embeds = action_embeddings_tensor[neighbor_original_indices]
                state_rep = obs_tensor.repeat(num_found, 1)
                q_vals = critic(state_rep, neighbor_embeds)

                if num_found > 0:
                    best_in_k = torch.argmax(q_vals).item()
                    action_idx_rb = neighbor_original_indices[best_in_k].item()
                else:
                    action_idx_rb = np.random.randint(num_actions_total)

        env_action = env.compute_action_from_index(action_idx_rb)
        next_state, reward, terminated, truncated, info = env.step(env_action)
        real_next_obs = next_state.copy()
        obs = real_next_obs

        action_to_store = np.array([action_idx_rb], dtype=np.int64)
        rb.add(obs, real_next_obs, action_to_store, reward, terminated, info)

        done = terminated or truncated
        if done:
            if "episode" in info:
                ep = info['episode']
                print(f"global_step={global_step}, episodic_return={ep['r']:.2f}, length={ep['l']}")
                writer.add_scalar("learning/episodic_return", ep["r"], global_step)
                writer.add_scalar("learning/episodic_length", ep["l"], global_step)
            elif "final_info" in info and isinstance(info["final_info"], dict) and 'episode' in info["final_info"]:
                ep = info["final_info"]['episode']
                print(f"global_step={global_step}, episodic_return={ep['r']:.2f}, length={ep['l']}")
                writer.add_scalar("learning/episodic_return", ep["r"], global_step)
                writer.add_scalar("learning/episodic_length", ep["l"], global_step)
            obs, info = env.reset()

        if global_step > args.learning_starts and global_step % args.train_frequency == 0:
            data = rb.sample(args.batch_size)
            observations = data.observations.float().to(device)
            next_observations = data.next_observations.float().to(device)
            batch_rewards = data.rewards.float().to(device)
            batch_dones = data.dones.float().to(device)
            stored_action_indices = data.actions.long().squeeze(1).to(device)

            with torch.no_grad():
                next_proto = target_actor(next_observations)
                next_proto_np = next_proto.cpu().numpy().astype(np.float32)
                D_batch, I_batch = gpu_search_in_batches(faiss_index, next_proto_np, k, batch_size=8)
                neighbor_indices_batch = torch.from_numpy(I_batch).long().to(device)

                flat_idx = neighbor_indices_batch.reshape(-1)
                all_nei_embeds = action_embeddings_tensor[flat_idx]
                next_obs_rep = next_observations.unsqueeze(1).repeat(1, args.k_neighbors, 1).view(-1, state_size)
                q_flat = target_critic(next_obs_rep, all_nei_embeds)
                q_batch = q_flat.view(args.batch_size, args.k_neighbors)
                max_q_next, _ = torch.max(q_batch, dim=1, keepdim=True)
                td_target = batch_rewards + args.gamma * max_q_next * (1 - batch_dones)

            action_emb_batch = action_embeddings_tensor[stored_action_indices]
            q_current = critic(observations, action_emb_batch)
            critic_loss = F.mse_loss(q_current, td_target)

            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            for param in critic.parameters():
                param.requires_grad = False
            proto_pred = actor(observations)
            actor_q_values = critic(observations, proto_pred)
            actor_loss = -actor_q_values.mean()
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()
            for param in critic.parameters():
                param.requires_grad = True

            for tp, lp in zip(target_critic.parameters(), critic.parameters()):
                tp.data.copy_(args.tau * lp.data + (1.0 - args.tau) * tp.data)
            for tp, lp in zip(target_actor.parameters(), actor.parameters()):
                tp.data.copy_(args.tau * lp.data + (1.0 - args.tau) * tp.data)

            if global_step % 100 == 0:
                writer.add_scalar("losses/td_loss", critic_loss.item(), global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/q_values", q_current.mean().item(), global_step)


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
    num_possible_actions = 2 ** (2 * grid_dimension)

    d = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    actor = ActorNetwork(state_size, args.action_embedding_dim, args.hidden_size, args.actor_hidden_layers).to(d)
    critic = CriticNetwork(state_size, args.action_embedding_dim, args.hidden_size, args.critic_hidden_layers).to(d)
    actor_optimizer = optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic_optimizer = optim.Adam(critic.parameters(), lr=args.critic_lr)
    target_actor = ActorNetwork(state_size, args.action_embedding_dim, args.hidden_size, args.actor_hidden_layers).to(d)
    target_critic = CriticNetwork(state_size, args.action_embedding_dim, args.hidden_size, args.critic_hidden_layers).to(d)
    target_actor.load_state_dict(actor.state_dict())
    target_critic.load_state_dict(critic.state_dict())

    action_embeddings_np = compute_displacement_embeddings(
        num_actions=num_possible_actions,
        grid_dimension=grid_dimension,
        embedding_dim=args.action_embedding_dim
    )

    state_size = env.grid_dimension if args.small_state else len(env.terminal_states) + env.grid_dimension
    binary_action_size = 2 * grid_dimension

    gpu_id = 0
    res = faiss.StandardGpuResources()
    cpu_index = faiss.IndexFlatL2(action_embeddings_np.shape[1])
    cpu_index.add(action_embeddings_np)
    faiss_index = faiss.index_cpu_to_gpu(res, gpu_id, cpu_index)
    print(f"FAISS GPU index built and ready on GPU")

    action_embeddings_tensor = torch.tensor(action_embeddings_np).to(d)
    action_space = spaces.Discrete(num_possible_actions)
    rb = ReplayBuffer(
        args.buffer_size,
        observation_space=spaces.MultiDiscrete([grid_size] * state_size),
        action_space=action_space,
        device=d,
        handle_timeout_termination=False,
        n_envs=1
    )

    train(args, env, rb, actor, critic, target_actor, target_critic,
          actor_optimizer, critic_optimizer, action_embeddings_tensor, faiss_index,
          state_size, d, writer)

    writer.close()
