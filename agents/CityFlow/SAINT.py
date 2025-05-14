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

from envs.CityFlowEnv import CityFlowEnv


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

    env_id: str = "CityFlow"
    total_timesteps: int = 500000
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
    lr_decay_rate: float = 0.999

    agent_hidden_size: int = 256
    """hidden size of the agent networks"""
    num_heads: int = 1
    num_attention_blocks: int = 3
    num_attention_ffn_layers: int = 2
    num_decision_layers: int = 2
    ln: bool = True
    film_generator_num_layers: int = 2
    generator_hidden_dim: int = 128

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
            nn.Linear(critic_hidden_size, critic_hidden_size),
            nn.ReLU(),
            nn.Linear(critic_hidden_size, critic_hidden_size),
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


def train():
    obs = torch.zeros((args.num_steps, args.num_envs) + state_shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs, num_agents), dtype=torch.long).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    global_step = 0
    start_time = time.time()

    next_obs = torch.Tensor(envs.reset()).to(device).unsqueeze(0)
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
    run_name = f"SAINT-{size}__{args.seed}__{int(time.time())}"

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

    envs = CityFlowEnv(configPath=args.cityflow_config, episodeSteps=args.max_episode_steps)
    assert args.num_envs == 1, "This implementation currently supports num_envs=1"

    state_shape = envs.observation_space.shape
    action_dims = list(envs.action_space.nvec)
    num_agents = len(action_dims)

    agent = Agent(state_size=state_shape[0],
                  action_dims=action_dims,
                  num_heads=args.num_heads,
                  num_attention_blocks=args.num_attention_blocks,
                  num_attention_ffn_layers=args.num_attention_ffn_layers,
                  decision_mlp_layers=args.num_decision_layers,
                  film_generator_num_layers=args.film_generator_num_layers,
                  film_generator_hidden_dim=args.generator_hidden_dim,
                  ln=args.ln).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    scheduler = ExponentialLR(optimizer, gamma=args.lr_decay_rate)

    train()

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.pt"
        torch.save(agent.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    writer.close()
