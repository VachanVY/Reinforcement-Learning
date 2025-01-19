import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation as anim
from dataclasses import dataclass
from itertools import count
import random
import typing as tp
import time
import math

import torch
from torch import nn, Tensor


SEED:int = 42
ENV_NAME:str = "CartPole-v1"

@dataclass
class config:
    num_steps_per_episode:int = 500
    num_episodes:int = 1000 # 1000
    gamma:float = 0.99
    
    maxlr:float = 1e-3
    minlr:float = maxlr*0.1
    warmup_steps:int = 1
    weight_decay:float = 0.0

    device:torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype:torch.dtype = torch.float32 # if "cpu" in device.type else torch.bfloat16


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

def show_one_episode(action_sampler:tp.Callable, save_path:tp.Optional[str]=None, n_max_steps=500, repeat=False):
    frames = []
    env = gym.make(ENV_NAME, render_mode="rgb_array")
    obs, info = env.reset()
    with torch.no_grad():
        for step in range(n_max_steps):
            frames.append(env.render())
            action = action_sampler(obs)
            obs, reward, done, truncated, info = env.step(action)
            if done or truncated:
                print("done at step", step+1)
                break
    env.close()
    return plot_animation(frames, repeat=repeat, save_path=save_path)


class CosineDecayWithWarmup:
    def __init__(
        self,
        warmup_steps:int,
        max_learning_rate:float,
        decay_steps:int,
        min_learning_rate:float
    ):
        self.warmup_steps = warmup_steps
        self.max_learning_rate = max_learning_rate
        self.decay_steps = decay_steps
        self.min_learning_rate = min_learning_rate

    def __call__(self, step):
        if step < self.warmup_steps:
            return self.max_learning_rate * step / self.warmup_steps
        if step > self.decay_steps:
            return self.min_learning_rate
        decay_ratio = (step - self.warmup_steps) / (self.decay_steps - self.warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.min_learning_rate + coeff * (self.max_learning_rate - self.min_learning_rate)


def smooth_rewards(sum_rewards_list, smoothing_factor=0.9):
    smoothed_rewards = []
    running_average = 0  # Initialize the running average
    for reward in sum_rewards_list:
        running_average = smoothing_factor * running_average + (1 - smoothing_factor) * reward
        smoothed_rewards.append(running_average)
    
    return smoothed_rewards


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim:int, action_dim:int):
        super().__init__()
        assert action_dim > 1
        self.fc1 = nn.Linear(state_dim, 16)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(16, 16)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(16, action_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, state):
        x = self.relu1(self.fc1(state))
        x = self.relu2(self.fc2(x))
        logits = self.fc3(x)
        return self.softmax(logits)

# Define the Value Network
class ValueNetwork(nn.Module):
    def __init__(self, state_dim:int):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 16)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(16, 16)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(16, 1)

    def forward(self, state):
        x = self.relu1(self.fc1(state))
        x = self.relu2(self.fc2(x))
        value = self.fc3(x)
        return value # (B, 1)
    

def sample_prob_action_from_pi(pi:PolicyNetwork, state:Tensor):
    probas:Tensor = pi(state).squeeze(0)
    dist = torch.distributions.Categorical(probas)
    action = dist.sample()
    return action, dist


def main():
    print("Training Starts...")
    num_steps_over = 0; sum_rewards_list = []
    for episode_num in range(config.num_episodes):
        state, info = env.reset()
        state = torch.as_tensor(state, dtype=config.dtype, device=config.device).unsqueeze(0)
        sum_rewards = 0; t0 = time.time()
        I = 1.0

        lr = get_lr(episode_num)
        for param_group1, param_group2  in zip(vopt.param_groups, popt.param_groups):
            param_group1["lr"] = lr; param_group2["lr"] = lr
        for tstep in count(0):
            num_steps_over += 1

            # Sample Action from Policy
            action, dist = sample_prob_action_from_pi(pi_fn, state)
            next_state, reward, done, truncated, info = env.step(int(action))
            next_state = torch.as_tensor(next_state, dtype=config.dtype, device=config.device).unsqueeze(0)
            sum_rewards += reward

            # Actor-Critic Algorithm
            ## Compute the Value Loss and Update the Value Network
            current_state_val:Tensor = value_fn(state)
            with torch.no_grad():
                next_state_val:Tensor = value_fn(next_state)
                target:Tensor = reward + config.gamma*next_state_val*(1-int(done))
            td_error = target - current_state_val
            value_loss = td_error.pow(2).sum()
            value_loss.backward()
            vopt.step()
            vopt.zero_grad()

            ## Compute the Policy Loss and Update the Policy Network
            td_error = td_error.detach()
            policy_loss:Tensor = -dist.log_prob(action).mul(td_error).mul(I)
            I *= config.gamma
            policy_loss.backward()
            popt.step()
            popt.zero_grad()    

            if done or truncated:
                break

            # Update the state
            state = next_state

        print(f"|| Episode: {episode_num+1} || Reward: {sum_rewards} || lr: {lr:<12e} || dt: {(time.time()-t0):.4f} ||")
        sum_rewards_list.append(sum_rewards)

        if sum_rewards[-50:].mean() > 490:
                break
        
    print("Training Ends...")
    return sum_rewards_list

if __name__ == "__main__":
    random.seed(SEED)
    np.random.seed(SEED+1)
    torch.manual_seed(SEED+2)
    torch.use_deterministic_algorithms(mode=True, warn_only=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    env = gym.make(ENV_NAME, render_mode="rgb_array")
    pi_fn = PolicyNetwork(env.observation_space.shape[0], env.action_space.n)
    pi_fn.to(config.device, dtype=torch.float32)
    pi_fn.compile()
    print(pi_fn, end=f"| Number of parameters: {sum(p.numel() for p in pi_fn.parameters())}\n\n")

    value_fn = ValueNetwork(env.observation_space.shape[0])
    value_fn.to(config.device, dtype=torch.float32)
    value_fn.compile()
    print(value_fn, end=f"| Number of parameters: {sum(p.numel() for p in value_fn.parameters())}\n\n")

    vopt = torch.optim.AdamW(value_fn.parameters(), lr=config.maxlr, weight_decay=config.weight_decay, fused=True)
    popt = torch.optim.AdamW(pi_fn.parameters(), lr=config.maxlr, weight_decay=config.weight_decay, fused=True)
    vopt.zero_grad(); popt.zero_grad()

    get_lr = CosineDecayWithWarmup(
        warmup_steps=config.warmup_steps,
        max_learning_rate=config.maxlr,
        decay_steps=config.num_episodes,
        min_learning_rate=config.minlr
    )

    sum_rewards_list = main()

    @torch.no_grad()
    def action_sampler(state):
        return sample_prob_action_from_pi(pi_fn, torch.as_tensor(state, dtype=torch.float32, device=config.device))[0].item()
    
    plt.plot(sum_rewards_list, label="Original rewards")
    plt.plot(smooth_rewards(sum_rewards_list, smoothing_factor=0.99), label="Smoothed rewards")
    plt.legend()
    plt.yticks(np.arange(0, 501, 50))
    plt.xlabel("Episode")
    plt.ylabel("Sum of rewards")
    plt.title("Sum of rewards per episode")
    plt.savefig("actor_critic_cartpole_rewards.png")
    plt.close()

    print("Making GIF...")
    show_one_episode(action_sampler, repeat=False, n_max_steps=500, save_path="actor_critic_cartpole.gif")
    plt.close()
    print("GIF Created Successfully!")

