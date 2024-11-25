import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation as anim
from dataclasses import dataclass
from itertools import count
from collections import deque
import random
import typing as tp

import torch
from torch import nn, Tensor


SEED:int = 42


@dataclass
class config:
    num_steps:int = 500_000
    num_steps_per_episode:int = 500
    num_episodes:int = num_steps//num_steps_per_episode # 1000
    num_warmup_steps:int = num_steps_per_episode*7 # 3500
    gamma:float = 0.99
    
    batch_size:int = 32
    lr:float = 1e-4
    weight_decay:float = 0.0

    device:torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype:torch.dtype = torch.float32 # if "cpu" in device.type else torch.bfloat16

    generator:torch.Generator = torch.Generator(device=device)
    generator.manual_seed(SEED+3)


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim:int, action_dim:int):
        super().__init__()
        assert action_dim > 1
        last_dim = 1 if action_dim == 2 else action_dim
        self.fc1 = nn.Linear(state_dim, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, last_dim)
        self.softmax_or_sigmoid = nn.Sigmoid() if last_dim == 1 else nn.Softmax(dim=-1)

    def forward(self, state):
        x = self.relu1(self.fc1(state))
        x = self.relu2(self.fc2(x))
        logits = self.fc3(x)
        return self.softmax_or_sigmoid(logits)


# Define the Value Network
class ValueNetwork(nn.Module):
    def __init__(self, state_dim:int):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 1)

    def forward(self, state):
        x = self.relu1(self.fc1(state))
        x = self.relu2(self.fc2(x))
        value = self.fc3(x)
        return value # (B, 1)
    

@torch.no_grad()
def sample_prob_action_from_pi(pi:PolicyNetwork, state:Tensor):
    left_proba:Tensor = pi(state)
    # If `left_proba` is high, then `action` will most likely be `False` or 0, which means left
    action = (torch.rand(size=(1, 1), device=config.device, generator=config.generator) > left_proba).int().item()
    return int(action)


@torch.compiler.disable(recursive=True)
def sample_from_buffer(replay_buffer:deque):
    batched_samples = random.sample(replay_buffer, config.batch_size) # Frames stored in uint8 [0, 255]
    instances = list(zip(*batched_samples))
    current_states, actions, rewards, next_states, dones = [
        torch.as_tensor(np.asarray(inst), device=config.device, dtype=torch.float32) for inst in instances
    ]
    return current_states, actions, rewards, next_states, dones


@torch.compile
def train_step():
    # Sample from replay buffer
    current_states, actions, rewards, next_states, dones = sample_from_buffer(replay_buffer)
    
    # Value Loss and Update weights
    zero_if_terminal_else_one = 1.0 - dones
    td_error:Tensor = (
        (rewards + config.gamma*value_fn(next_states).squeeze(1)*zero_if_terminal_else_one) -
        value_fn(current_states).squeeze(1)
    ) # (B,)
    value_loss = 0.5 * td_error.pow(2).mean() # (,)
    value_loss.backward()
    vopt.step()
    vopt.zero_grad()

    # Policy Loss and Update weights
    # CHATGPT changes
    td_error = ((td_error - td_error.mean()) / (td_error.std() + 1e-8)).detach() # (B,)
    y_target:Tensor = 1.0 - actions # (B,)
    left_probas:Tensor = pi_fn(current_states).squeeze(1) # (B,)
    pi_loss = -torch.mean(
        (torch.log(left_probas) * y_target + torch.log(1.0 - left_probas) * (1.0 - y_target))*td_error,
        dim=0
    )
    pi_loss.backward()
    popt.step()
    popt.zero_grad()


def main():
    print(f"Training Starts...\nWARMING UP TILL ~{config.num_warmup_steps//config.num_steps_per_episode} episodes...")
    num_steps_over = 0; sum_rewards_list = []
    for episode_num in range(config.num_episodes):
        state, info = env.reset()
        sum_rewards = 0.0
        for tstep in count(0):
            num_steps_over += 1
            
            # Sample action from policy
            if num_steps_over < config.num_warmup_steps:
                action = env.action_space.sample()
            else:
                action = sample_prob_action_from_pi(pi_fn, torch.as_tensor(state, device=config.device, dtype=torch.float32).unsqueeze(0))
            next_state, reward, done, truncated, info = env.step(action)
            replay_buffer.append((state, action, reward, next_state, done))

            # Train the networks
            if num_steps_over >= config.num_warmup_steps:
                train_step()

            sum_rewards += reward
            if done or truncated:
                break
            
            # Update state
            state = next_state

        # LOGGING
        print(f"Episode {episode_num+1}/{config.num_episodes} | Sum of rewards: {sum_rewards:.2f}")
        sum_rewards_list.append(sum_rewards)

    print("Training is over after", num_steps_over)
    return sum_rewards_list


if __name__ == "__main__":
    random.seed(SEED)
    np.random.seed(SEED+1)
    torch.manual_seed(SEED+2)
    torch.use_deterministic_algorithms(mode=True, warn_only=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    env = gym.make("CartPole-v1", render_mode="rgb_array")

    pi_fn = PolicyNetwork(env.observation_space.shape[0], env.action_space.n)
    pi_fn.to(config.device)
    print(pi_fn, end=f"| Number of parameters: {sum(p.numel() for p in pi_fn.parameters())}\n\n")

    value_fn = ValueNetwork(env.observation_space.shape[0])
    value_fn.to(config.device)
    print(value_fn, end=f"| Number of parameters: {sum(p.numel() for p in value_fn.parameters())}\n\n")

    vopt = torch.optim.AdamW(value_fn.parameters(), lr=config.lr, weight_decay=config.weight_decay, fused=True)
    popt = torch.optim.AdamW(pi_fn.parameters(), lr=config.lr, weight_decay=config.weight_decay, fused=True)
    vopt.zero_grad(), popt.zero_grad()

    replay_buffer = deque(maxlen=5000)

    sum_rewards_list = main()

    plt.plot(sum_rewards_list)
    plt.yticks(np.arange(0, 501, 50))
    plt.xlabel("Episode")
    plt.ylabel("Sum of rewards")
    plt.title("Sum of rewards per episode")
    plt.show()
    