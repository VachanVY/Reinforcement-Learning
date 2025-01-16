import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import typing as tp
import time
import random
import math

import torch
from torch import nn, Tensor

SEED = 42
ENV_NAME = "CartPole-v1"
random.seed(SEED)
np.random.seed(SEED+1)
torch.manual_seed(SEED+2)
torch.use_deterministic_algorithms(mode=True, warn_only=False)


def plot_environment(env:gym.Env, figsize:tuple[int, int]=(5, 4)):
    plt.figure(figsize=figsize)
    img = env.render()
    plt.imshow(img)
    plt.axis("off")
    return img

def update_scene(num, frames, patch):
    patch.set_data(frames[num])
    return patch,

def plot_animation(frames:list, save_path:str, repeat=False, interval=40):
    fig = plt.figure()
    patch = plt.imshow(frames[0])
    plt.axis('off')
    animation = anim.FuncAnimation(
        fig, update_scene, fargs=(frames, patch),
        frames=len(frames), repeat=repeat, interval=interval)
    animation.save(save_path, writer="pillow", fps=20)
    return animation

def show_one_episode(action_sampler:tp.Callable, save_path:str, n_max_steps=500, repeat=False):
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
    return plot_animation(frames, save_path=save_path, repeat=repeat)


class config:
    num_episodes:int = 800 # number of episodes
    batch_size:int = 10
    max_steps_per_episode:int = 1000
    
    gamma:float = 0.95
    
    lr:float = 0.01
    weight_decay:float = 0.0

    device:torch.device = torch.device("cuda" if False else "cpu") # cpu good for very very small models
    dtype:torch.dtype = torch.float32 if "cpu" in device.type else torch.bfloat16

    generator:torch.Generator = torch.Generator(device=device)
    generator.manual_seed(SEED+3)


class Policy(nn.Module):
    def __init__(self, num_inputs:int, num_hidden:int, num_outputs:int):
        super().__init__()
        self.linear1 = nn.Linear(num_inputs, num_hidden)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(num_hidden, num_outputs)
        self.last_act_func = nn.Sigmoid() if num_outputs == 1 else nn.Softmax(dim=-1)

    def forward(self, x:Tensor): # (B, num_inputs=4)
        x = self.relu(self.linear1(x)) # (B, num_hidden=8)
        x = self.last_act_func(self.linear2(x)) # (B, num_outputs=1)
        return x
    

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
    

@torch.compile
def loss_fn(pi:Policy, obs:Tensor) -> tuple[Tensor, Tensor]:
    left_proba = pi(obs.unsqueeze(0)) # (1, 1)
    action = torch.rand(size=(1, 1), device=config.device, generator=config.generator) > left_proba # If `left_proba` is high, then `action` will most likely be `False` or 0, which means left
    y_target = 1.0 if action==False else 0.0  # If `action` is left, then `y_target` will be 1.0 
    loss = torch.log(left_proba) * y_target + torch.log(1.0 - left_proba) * (1.0 - y_target)
    return loss.mean(), action


def play_one_episode(
    env:gym.Env,
    pi:Policy,
    obs:np.ndarray,
):
    obs = torch.as_tensor(obs, device=config.device)
    with autocast:
        loss, action_taken = loss_fn(pi, obs)
    loss.backward()
    grads = [p.grad for p in pi.parameters()]; pi.zero_grad()
    action_taken = int(action_taken.item())
    obs, reward, done, truncated, info = env.step(action_taken)
    return (obs, reward, done, truncated), (loss, grads)


def discount_rewards(rewards:list, discount_factor:float):
    discounted = np.array(rewards)
    for step in range(len(rewards) - 2, -1, -1):
        discounted[step] += discounted[step + 1] * discount_factor
    return discounted


def apply_grads(grads:list[list[Tensor]], rewards:list[list[Tensor]]):
    param_list_len = len([p for p in pi.parameters()])
    to_be_applied_grads:list[list[Tensor]] = [[] for _ in range(param_list_len)] # initialize list

    for n_ep_per_iter, grads_in_ep in enumerate(grads):
        for step, grads_in_step in enumerate(grads_in_ep):
            for i, grad in enumerate(grads_in_step):
                to_be_applied_grads[i].append(grad*rewards[n_ep_per_iter][step])

    grads:list[Tensor] = [torch.stack(grad_list).mean(dim=0) for grad_list in to_be_applied_grads]
    
    for p, grad in zip(pi.parameters(), grads):
        p.grad = grad

    optimizer.step()
    optimizer.zero_grad()


def play():
    episode_length_avgs = []
    for iter_num in range(1, config.num_episodes+1):
        t0 = time.time()
        all_rewards, all_grads = [], []
        lr = get_lr(iter_num)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        for episode in range(config.batch_size):
            current_grads, current_rewards = [], []
            obs, info = env.reset()
            for step in range(config.max_steps_per_episode):
                (
                    (obs, reward, done, truncated), (loss, grads)
                ) = play_one_episode(env, pi, obs)
                current_grads.append(grads); current_rewards.append(reward)
                if done or truncated:
                    break
            
            all_rewards.append(current_rewards); all_grads.append(current_grads)
        t1 = time.time(); dt = t1-t0
        episode_len_avg = sum(map(len, all_rewards))/config.batch_size; episode_length_avgs.append(episode_len_avg)
        print(f"| Step: {iter_num}/{config.num_episodes} || Average Episode Length {episode_len_avg:.2f} || lr: {lr:e} || dt: {dt:.2f}s |")

        all_rewards = list(map(lambda r:discount_rewards(r, config.gamma), all_rewards))

        flattened_rewards = np.concatenate(all_rewards)
        all_rewards = [r - flattened_rewards.mean()/ flattened_rewards.std() for r in all_rewards]
        apply_grads(all_grads, all_rewards)
    return episode_length_avgs


if __name__ == "__main__":
    env = gym.make(ENV_NAME)
    pi = Policy(
        num_inputs=env.observation_space.shape[0], # 4
        num_hidden=env.observation_space.shape[0]*2, # 8
        num_outputs=1 if env.action_space.n == 2 else env.action_space.n, #  (left, right)
    )
    pi.to(config.device)
    optimizer = torch.optim.NAdam(
        pi.parameters(),
        lr=config.lr, 
        weight_decay=config.weight_decay,
        maximize=True # grad ascent
    )
    autocast = torch.autocast(device_type=config.device.type, dtype=config.dtype)
    get_lr = CosineDecayWithWarmup(
        warmup_steps=1,
        max_learning_rate=config.lr,
        decay_steps=config.num_episodes,
        min_learning_rate=config.lr*0.1
    )
    print(pi, sum(p.numel() for p in pi.parameters()), sep="\nNumber of parameters: ")

    avg_episode_lengths = play()

    plt.plot(avg_episode_lengths)
    plt.xlabel("Number of Iterations")
    plt.ylabel("Average Episode Length")
    plt.grid()
    plt.savefig("images/monte_policy_gradient_cartpole.png")
    plt.show()
    plt.close()

    @torch.no_grad()
    def sample_action(obs):
        left_proba = pi(torch.as_tensor(obs[np.newaxis], device=config.device)).squeeze().item()
        return int(random.random() > left_proba)
    
    show_one_episode(sample_action, save_path="images/monte_policy_gradient_cartpole.gif", repeat=False, n_max_steps=500)
    plt.close()
    env.close()