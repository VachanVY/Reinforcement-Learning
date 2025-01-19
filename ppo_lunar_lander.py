import gymnasium as gym
import numpy as np
import typing as tp
import random
from dataclasses import dataclass
from itertools import count
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import time

import torch
from torch import nn, Tensor

ENV_NAME = "LunarLander-v3"


def update_scene(num, frames, patch):
    patch.set_data(frames[num])
    return patch,

def plot_animation(frames:list, save_path:tp.Optional[str]=None, repeat=False, interval=40):
    fig = plt.figure()
    patch = plt.imshow(frames[0])
    plt.axis('off')
    animation = anim.FuncAnimation(
        fig, update_scene, fargs=(frames, patch),
        frames=len(frames), repeat=repeat, interval=interval)
    if save_path is not None:
        animation.save(save_path, writer="pillow", fps=20)
    return animation

def show_one_episode(action_sampler:tp.Callable, save_path:tp.Optional[str]=None, repeat=False):
    frames = []
    env = gym.make(ENV_NAME, render_mode="rgb_array")
    obs, info = env.reset()
    sum_rewards = int(0)
    with torch.no_grad():
        for step in count(0):
            frames.append(env.render())
            action = action_sampler(obs)
            obs, reward, done, truncated, info = env.step(action)
            sum_rewards += reward
            if done or truncated:
                print("Sum of Rewards:", sum_rewards)
                print("done at step", step+1)
                break
    env.close()
    return plot_animation(frames, repeat=repeat, save_path=save_path)


@dataclass
class config:
    # PPO config
    clip_range:float = 0.2
    clip_max:float = 1 + clip_range
    clip_min:float = 1 - clip_range
    target_kl:float = 0.01 # Usually 0.01 or 0.05

    ## Training config
    log_losses:bool = False
    lr_actor:float = 3e-4
    lr_critic:float = 1e-3
    K:int = 80
    batch_size:int = 32
    weight_decay:float = 0.0
    update_timestep:int = 100

    # General RL config
    gamma:float = 0.99
    max_steps:int = int(1e7)

    device:torch.device = torch.device("cuda" if False else "cpu") # cpu good for very very small models
    dtype:torch.dtype = torch.float32 if "cpu" in device.type else torch.bfloat16


class Buffer:
    def __init__(self): 
        self.clear()

    def clear(self):
        self.states = []
        self.actions = []
        self.action_logprobs = []
        self.state_vals = []

        self.rewards = []
        self.terminals = []

    def store(self, state, action, action_logprob, state_val, reward, terminal):
        self.states.append(state)                   # state: (state_dim,) -> states: (num_timesteps, state_dim) 
        self.actions.append(action)                 # action: (2,) -> actions: (num_timesteps, 2)
        self.action_logprobs.append(action_logprob) # action_logprob: (1,) -> action_logprobs: (num_timesteps, 1)
        self.state_vals.append(state_val)           # state_val: (1,) -> state_vals: (num_timesteps, 1)
        self.rewards.append(reward)                 # reward: (1,) -> rewards: (num_timesteps, 1)
        self.terminals.append(terminal)             # terminal: (1,) -> terminals: (num_timesteps, 1)


def discounted_returns(rewards:tp.Sequence, is_terminals:tp.Sequence, discount_factor:float):
    discounted_rewards = [];    discounted_reward = 0
    for reward, is_terminal in zip(reversed(rewards), reversed(is_terminals)):
        if is_terminal: discounted_reward = 0
        discounted_reward = reward + (discount_factor * discounted_reward)
        discounted_rewards.insert(0, float(discounted_reward))
    return torch.tensor(discounted_rewards, device=config.device)


class Value(nn.Module):
    def __init__(self, state_dim:int, hidden_dim:int=64):
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
    def __init__(self, state_dim:int, action_dim:int, hidden_dim:int=64):
        super().__init__(state_dim, hidden_dim)
        self.lin3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x:Tensor):
        x = self.common_forward(x)
        x = self.lin3(x)
        return x


class ActorCritic(nn.Module):
    def __init__(self, state_dim:int, action_dim:int, hidden_dim:int=64):
        super().__init__()
        self.policy = Policy(state_dim, action_dim, hidden_dim)
        self.value = Value(state_dim, hidden_dim)

    def forward(self, x:Tensor) -> tuple[Tensor, Tensor]:
        return self.policy(x), self.value(x)


def sample_action(state:Tensor):
    with torch.no_grad():
        action_logits, state_vals = actor_critic_old(state)
        dist = torch.distributions.Categorical(logits=action_logits)
        action = dist.sample()
        action_logprobs = dist.log_prob(action)
    return action, action_logprobs, state_vals


def update():
    avg = lambda x: sum(x)/len(x)
    # Compute discounted returns and Normalize
    returns = discounted_returns(replay_buffer.rewards, replay_buffer.terminals, config.gamma).to(config.device) # (num_timesteps,)
    returns = ((returns - returns.mean()) / (returns.std() + 1e-8)).detach().unsqueeze(-1) # (num_timesteps, 1)
    buf_size = len(returns)

    # Preprocess buffer data
    buffer_states = torch.stack(replay_buffer.states).detach().to(config.device) # (num_timesteps, state_dim)
    buffer_actions = torch.stack(replay_buffer.actions).detach().to(config.device) # (num_timesteps, action_dim)
    buffer_action_logprobs = torch.stack(replay_buffer.action_logprobs).detach().to(config.device) # (num_timesteps, 1)
    buffer_state_vals = torch.stack(replay_buffer.state_vals).detach().to(config.device) # (num_timesteps, 1)

    # Compute advantage: detached
    advantages = returns - buffer_state_vals # (num_timesteps, 1)

    # Normlaize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7) # (num_timesteps, 1)

    # K Epochs
    losses = {"policy": [], "value": []}
    kldivs_list = []
    for _ in range(config.K):
        rand_idx = torch.randperm(config.batch_size if buf_size > config.batch_size else buf_size)
        batch_returns = returns[rand_idx] # (B, 1)
        batch_advantages = advantages[rand_idx] # (B, 1)
        batch_states = buffer_states[rand_idx] # (B, state_dim)
        batch_actions = buffer_actions[rand_idx] # (B, action_dim)
        batch_action_logprobs = buffer_action_logprobs[rand_idx] # (B, 1)

        # Compute advantage
        action_logits, state_vals = actor_critic(batch_states) # (B, action_dim), (B, 1)
        dist = torch.distributions.Categorical(logits=action_logits)
        action_logprobs = dist.log_prob(batch_actions) # (B, 1)
        
        # Value function loss
        value_loss = nn.functional.mse_loss(state_vals, batch_returns)

        # Policy function loss
        log_ratios = action_logprobs - batch_action_logprobs # (B, 1)
        ratios = torch.exp(log_ratios) # (B, 1)
        clipped_objective = torch.clip(ratios, config.clip_min, config.clip_max) * batch_advantages # (B, 1)
        unclipped_objective = ratios * batch_advantages # (B, 1)
        policy_loss = -torch.min(clipped_objective, unclipped_objective).mean() # (,)

        # KL Divergence
        with torch.no_grad(): # https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/ppo/ppo.py#L262-L265
            log_ratios = log_ratios.detach()
            approx_kl_div = ((log_ratios.exp() - 1) - log_ratios).mean().cpu().item()
            kldivs_list.append(approx_kl_div)
        if approx_kl_div > config.target_kl * 1.5:
            break
        
        # Optimize
        value_loss.backward()
        policy_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Store losses
        losses["policy"].append(policy_loss.item())
        losses["value"].append(value_loss.item())

    # Update old policy
    actor_critic_old.load_state_dict(actor_critic.state_dict())
    replay_buffer.clear()
    avg_policy_loss, avg_val_loss, avg_kl_div  = avg(losses["policy"]), avg(losses["value"]), avg(kldivs_list)
    return (avg_policy_loss, avg_val_loss, avg_kl_div)


def train():
    try:
        sum_rewards_list = []; num_steps = int(0); avg_kl_div_list = []; episode_length_list = []
        for episode_num in count(1):
            state, info = env.reset()
            state = torch.as_tensor(state, device=config.device)
            sum_rewards = int(0)
            for tstep in count(1):
                num_steps += 1

                # Sample action from old policy
                action, action_logprobs, state_vals = sample_action(state)

                # Feed action to environment
                next_state, reward, terminal, truncated, info = env.step(action.item())
                sum_rewards += reward
                
                # Store to buffer
                replay_buffer.store(state, action, action_logprobs, state_vals, reward, terminal)

                if num_steps % config.update_timestep == 0:
                    (avg_policy_loss, avg_val_loss, avg_kl_div) = update() ; avg_kl_div_list.append(avg_kl_div)
                    if config.log_losses:
                        print(f"|| Episode {episode_num} || Policy loss Avg: {avg_policy_loss:.3f} || Value loss Avg: {avg_val_loss:.3f} || KL Div Avg: {avg_kl_div:.4f} ||")

                if terminal or truncated:
                    break

                if num_steps > config.max_steps:
                    return sum_rewards_list, avg_kl_div_list
                
                # Update state
                state = torch.as_tensor(next_state, device=config.device)

            sum_rewards_list.append(sum_rewards)
            episode_length_list.append(tstep)

            # Logging
            tab_char = "\t" if config.log_losses else ""
            print(f"{tab_char}|| Episode Number: {episode_num} || Sum rewards: {sum_rewards:.2f} || Episode Length: {tstep} ||")
    except KeyboardInterrupt:
        print("Training interrupted.")
    actor_critic_old.load_state_dict(actor_critic.state_dict())
    return sum_rewards_list, avg_kl_div_list, episode_length_list

def action_sampler(state:np.ndarray):
    return sample_action(torch.as_tensor(state, device=config.device).unsqueeze(0))[0].item()

if __name__ == "__main__":
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED+1)
    torch.manual_seed(SEED+2)
    torch.use_deterministic_algorithms(mode=True, warn_only=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    env = gym.make(ENV_NAME)
    print("States:", env.observation_space.shape[0])
    print("Actions:", env.action_space.n)

    print("CONFIG:", config(), sep="\n")

    NUM_ACTIONS = env.action_space.n
    NUM_STATES = env.observation_space.shape[0]

    actor_critic = ActorCritic(NUM_STATES, NUM_ACTIONS)
    actor_critic.to(config.device); # actor_critic.compile()
    optimizer = torch.optim.AdamW([
        {"params": actor_critic.policy.parameters(), "lr": config.lr_actor},
        {"params": actor_critic.value.parameters(), "lr": config.lr_critic}
    ], weight_decay=config.weight_decay)
    optimizer.zero_grad()

    actor_critic_old = ActorCritic(NUM_STATES, NUM_ACTIONS)
    actor_critic_old.to(config.device); # actor_critic_old.compile()
    # actor_critic_old.requires_grad_(False)
    actor_critic_old.load_state_dict(actor_critic.state_dict())

    replay_buffer = Buffer()
    print(actor_critic)
    print("Number of parameters:", sum(p.numel() for p in actor_critic.parameters()))
    time.sleep(3)

    sum_rewards_list, avg_kl_div_list, episode_length_list = train()

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
    plt.savefig(f"images/{ENV_NAME}_combined_plots.png")
    plt.show()
    plt.close()

    torch.save(actor_critic_old, ckpt_path:=f"ckpt/ppo_{ENV_NAME}.pth")
    print("Model saved to", ckpt_path)

    show_one_episode(action_sampler, save_path=f"images/{ENV_NAME}.gif", repeat=False)
    plt.close()