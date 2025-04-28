import dataclasses as dc
from copy import deepcopy
import gymnasium as gym
from itertools import count
import os

import torch
from torch import nn, Tensor

from .utils import (
    get_discounted_returns, 
    Buffer,
)

class Value(nn.Module):
    def __init__(self, state_dim:int, hidden_dim:int):
        super().__init__()
        # easier to see output of each layer, so no nn.Sequential
        self.lin1 = nn.Linear(state_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.lin3 = nn.Linear(hidden_dim, 1)

    def common_forward(self, x:Tensor):
        x = self.lin1(x);        x = self.relu1(x)
        x = self.lin2(x);        x = self.relu2(x)
        return x

    def forward(self, x:Tensor):
        x = self.common_forward(x)
        x = self.lin3(x)
        return x


class Policy(Value):
    def __init__(self, state_dim:int, action_dim:int, hidden_dim:int):
        super().__init__(state_dim, hidden_dim)
        self.lin3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x:Tensor):
        x = self.common_forward(x)
        x = self.lin3(x)
        return x
    

class ActorCritic(nn.Module):
    def __init__(self, state_dim:int, action_dim:int, hidden_dim:int):
        super().__init__()
        self.policy = Policy(state_dim, action_dim, hidden_dim)
        self.value = Value(state_dim, hidden_dim)

    def forward(self, x:Tensor) -> tuple[Tensor, Tensor]:
        return self.policy(x), self.value(x)

    def sample_action(self, state:Tensor):
        with torch.no_grad():
            action_logits, state_vals = self(state)
            dist = torch.distributions.Categorical(logits=action_logits)
            action = dist.sample()
            action_logprobs = dist.log_prob(action)
        return action, action_logprobs, state_vals


# default config to test on CartPole
@dc.dataclass
class PPOConfig:
    # PPO config
    clip_range:float = 0.2
    clip_max:float = 1 + clip_range
    clip_min:float = 1 - clip_range
    target_kl:float = 0.05 # Usually 0.01 or 0.05

    ## Training config
    log_losses:bool = False
    lr_actor:float = 3e-4
    lr_critic:float = 1e-3
    K:int = 80
    batch_size:int = 32
    weight_decay:float = 0.0
    update_timestep:int = 500
    hidden_dim:int = 64

    # General RL config
    gamma:float = 0.99
    max_steps:int = int(1e7)
    num_episodes:int = 1000

    device:torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype:torch.dtype = torch.float32 #if "cpu" in device.type else torch.bfloat16


class PPO:
    def __init__(self, env_name:str, ppo_config:PPOConfig):
        self.env = gym.make(env_name)
        self.rollout_buffer = Buffer()
        self.config = ppo_config

        self.config.action_dim = self.env.action_space.n
        self.config.state_dim = self.env.observation_space.shape[0]

        self.actor_critic = ActorCritic(
            state_dim=self.config.state_dim,
            action_dim=self.config.action_dim,
            hidden_dim=self.config.hidden_dim
        ).to(self.config.device) #; self.actor_critic.compile()
        print(f"Actor-Critic model: \n{self.actor_critic}")
        print("Actor-Critic Model: Number of parameters: ", sum(p.numel() for p in self.actor_critic.parameters()))
        self.actor_critic_old = deepcopy(self.actor_critic)
        self.actor_critic_old.eval()
        self.actor_critic_old.requires_grad_(False)

        self.optimizer = torch.optim.AdamW([
            {"params": self.actor_critic.policy.parameters(), "lr": self.config.lr_actor},
            {"params": self.actor_critic.value.parameters(), "lr": self.config.lr_critic}
        ], weight_decay=self.config.weight_decay)


    def update(self, normalize_returns:bool=True, normalize_advantages:bool=True):
        avg = lambda x: sum(x)/len(x)
        # Compute discounted returns and Normalize
        returns = torch.tensor(get_discounted_returns(self.rollout_buffer.rewards, self.rollout_buffer.terminals, self.config.gamma)).to(self.config.device) # (num_timesteps,)
        if normalize_returns:
            returns = ((returns - returns.mean()) / (returns.std() + 1e-8)).detach().unsqueeze(-1) # (num_timesteps, 1)
        buf_size = len(returns)

        # Preprocess buffer data
        buffer_states = torch.stack(self.rollout_buffer.states).detach().to(self.config.device) # (num_timesteps, state_dim)
        buffer_actions = torch.stack(self.rollout_buffer.actions).detach().to(self.config.device) # (num_timesteps, action_dim)
        buffer_action_logprobs = torch.stack(self.rollout_buffer.action_logprobs).detach().to(self.config.device) # (num_timesteps, 1)
        buffer_state_vals = torch.stack(self.rollout_buffer.state_vals).detach().to(self.config.device) # (num_timesteps, 1)

        # Compute advantage: detached
        advantages = returns - buffer_state_vals # (num_timesteps, 1)

        # Normlaize advantages
        if normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7) # (num_timesteps, 1)

        # K Epochs
        losses = {"policy": [], "value": []}
        kldivs_list = []
        for _ in range(self.config.K):
            for i in range(0, buf_size - self.config.batch_size + 1, self.config.batch_size):
                batch_idx = slice(i, min(i + self.config.batch_size, buf_size))

                batch_states = buffer_states[batch_idx]          # (B, state_dim)
                batch_actions = buffer_actions[batch_idx]        # (B, action_dim)
                batch_action_logprobs = buffer_action_logprobs[batch_idx]  # (B, 1)
                batch_returns = returns[batch_idx]               # (B, 1)
                batch_advantages = advantages[batch_idx]         # (B, 1)

                # Compute advantage
                action_logits, state_vals = self.actor_critic(batch_states) # (B, action_dim), (B, 1)
                dist = torch.distributions.Categorical(logits=action_logits)
                action_logprobs = dist.log_prob(batch_actions) # (B, 1)
                
                # Value function loss
                value_loss = nn.functional.mse_loss(state_vals, batch_returns)

                # Policy function loss
                log_ratios = action_logprobs - batch_action_logprobs # (B, 1)
                ratios = torch.exp(log_ratios) # (B, 1)
                clipped_objective = torch.clip(ratios, self.config.clip_min, self.config.clip_max) * batch_advantages # (B, 1)
                unclipped_objective = ratios * batch_advantages # (B, 1)
                policy_loss = -torch.min(clipped_objective, unclipped_objective).mean() # (,)

                # KL Divergence
                with torch.no_grad():
                    # https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/ppo/ppo.py#L262-L265
                    # http://joschu.net/blog/kl-approx.html
                    log_ratios = log_ratios.detach()
                    approx_kl_div = ((log_ratios.exp() - 1) - log_ratios).mean().cpu().item()
                    kldivs_list.append(approx_kl_div)
                if approx_kl_div > self.config.target_kl * 1.5:
                    break
                
                # Optimize
                value_loss.backward()
                policy_loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                # Store losses
                losses["policy"].append(policy_loss.item())
                losses["value"].append(value_loss.item())

        # Update old policy
        self.actor_critic_old.load_state_dict(self.actor_critic.state_dict())
        self.rollout_buffer.clear()
        avg_policy_loss, avg_val_loss, avg_kl_div = avg(losses["policy"]), avg(losses["value"]), avg(kldivs_list)
        return (avg_policy_loss, avg_val_loss, avg_kl_div)
    

    def train(self):
        sum_rewards_list = []; num_steps = int(0); avg_kl_div_list = []; episode_length_list = []
        iterator = count(1) if self.config.num_episodes == -1 else range(self.config.num_episodes)
        for episode_num in iterator:
            state, info = self.env.reset()
            state = torch.as_tensor(state, device=self.config.device)
            sum_rewards = int(0)
            for tstep in count(1):
                num_steps += 1

                # Sample action from old policy
                action, action_logprobs, state_vals = self.actor_critic_old.sample_action(state)

                # Feed action to environment
                next_state, reward, terminal, truncated, info = self.env.step(action.item())
                sum_rewards += reward
                
                # Store to buffer
                self.rollout_buffer.store(state, action, action_logprobs, state_vals, reward, terminal)

                if num_steps % self.config.update_timestep == 0:
                    (avg_policy_loss, avg_val_loss, avg_kl_div) = self.update() ; avg_kl_div_list.append(avg_kl_div)
                    if self.config.log_losses:
                        print(f"|| Episode {episode_num} || Policy loss Avg: {avg_policy_loss:.3f} || Value loss Avg: {avg_val_loss:.3f} || KL Div Avg: {avg_kl_div:.4f} ||")

                if terminal or truncated:
                    break

                if num_steps > self.config.max_steps:
                    return sum_rewards_list, avg_kl_div_list
                
                # Update state
                state = torch.as_tensor(next_state, device=self.config.device)

            sum_rewards_list.append(sum_rewards)
            episode_length_list.append(tstep)

            # Logging
            tab_char = "\t" if self.config.log_losses else ""
            print(f"{tab_char}|| Episode Number: {episode_num} || Sum rewards: {sum_rewards:.2f} || Episode Length: {tstep} ||")
        self.actor_critic_old.load_state_dict(self.actor_critic.state_dict())
        return sum_rewards_list, episode_length_list, avg_kl_div_list


    def save(self, path:str):
        torch.save(self.actor_critic.state_dict(), path)


    def load(self, path:str):
        self.actor_critic.load_state_dict(torch.load(path))


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    ppo_agent = PPO("CartPole-v1", PPOConfig())
    sum_rewards_list, episode_length_list, avg_kl_div_list = ppo_agent.train()
    ppo_agent.save("models/ppo_cartpole.pth")

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(10, 15))

    axes[0].plot(sum_rewards_list, label="Sum of Rewards")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Sum of Rewards")
    axes[0].set_title("Sum of Rewards vs Episode")
    axes[0].legend()
    axes[0].grid()

    axes[1].plot(episode_length_list, label="Episode Length", color="orange")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Episode Length")
    axes[1].set_title("Episode Length vs Episode")
    axes[1].legend()
    axes[1].grid()

    axes[2].plot(avg_kl_div_list, label="KL Divergence", color="green")
    axes[2].set_xlabel("Steps")
    axes[2].set_ylabel("KL Divergence")
    axes[2].set_title("KL Divergence")
    axes[2].legend()
    axes[2].grid()

    plt.tight_layout()
    plt.savefig(f"logs/ppo_cartpole.png")
    plt.show()
    plt.close()

