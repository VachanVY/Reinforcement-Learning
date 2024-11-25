import gymnasium as gym; import ale_py; gym.register_envs(ale_py)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import typing as tp
import time
import random
import math
from collections import deque
from dataclasses import dataclass
import os
from itertools import count

import torch
from torch import nn, Tensor
from torch.utils.tensorboard import SummaryWriter


SEED = 42
ENV_NAME = "ALE/Pong-v5"


@dataclass
class config:
    num_steps:int = 25_000_000 # 25 million steps!!!
    num_warmup_steps:int = 50_000
    gamma:float = 0.99

    buffer_size:int = 1_000_000
    
    lr:float = 1e-4
    weight_decay:float = 0.0000
    batch_size:int = 32
    clip_norm:float = 5.0

    init_eps:float = 1.0
    final_eps:float = 0.1
    eps_decay_steps:int = 1_000_000

    device:torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype:torch.dtype = torch.float32 if "cpu" in device.type else torch.bfloat16

    autocast:torch.autocast = torch.autocast(
        device_type=device.type, dtype=dtype, enabled="cuda" in device.type
    )
    logging_every_n_episode:int = 1
    save_every_n_steps:int = 50_000

    generator:torch.Generator = torch.Generator(device=device).manual_seed(SEED+343434)


def prepro(I:np.ndarray|Tensor, device:torch.device=config.device):
    """
    Input: (210, 160, 3)

    Output: (1, 80, 80)

    https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5
    """
    assert len(I.shape) == 3, "must be (H, W, C)"
    if isinstance(I, np.ndarray):
        I = torch.as_tensor(I)
    I = I.clone().float().to(device)
    I = I[35:195] # (160, 160, 3)
    I = I[::2,::2, 0] # (80, 80)
    I[I == 144] = 0 # erase background (background type 1)
    I[I == 109] = 0 # erase background (background type 2)
    I[I != 0] = 1 # everything else (paddles, ball) just set to 1
    return I[None]/255.0 # (1, 80, 80)


class DQN(nn.Module):
    def __init__(self, fan_in:int, fan_out:int):
        super().__init__()
        self.in_channels = fan_in
        self.num_actions = fan_out

        self.conv1 = nn.Conv2d(
            in_channels=self.in_channels, out_channels=16, kernel_size=8, stride=4
        ) # (B, 16, 19, 19)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=4, stride=2
        ) # (B, 32, 8, 8)
        self.relu2 = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.LazyLinear(out_features=256) # (B, 256)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=256, out_features=self.num_actions) # (B, num_actions)

    def forward(self, x:Tensor): # (B, 4, 80, 80)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x
    
def get_model(*args, log:bool=False, **kwargs):
    model = DQN(**kwargs)
    model.to(config.device)
    shape = model(torch.randn(*args, device=config.device)).shape
    if log:
        print("\n\nModel input shape:", args)
        print("Model output shape", shape, end="\n\n")
    return model


@torch.no_grad()
def sample_action(dqn:DQN, obs:np.ndarray, epsilon:float) -> Tensor:
    if random.random() <= epsilon: return torch.randint(low=0, high=dqn.num_actions, size=(1,), device=config.device, generator=config.generator)
    else: return dqn(torch.as_tensor(obs, device=config.device)).squeeze(0).argmax()


def train_step(dqn:DQN, replay_buffer:deque, optimizer:torch.optim.Optimizer):
    # sample instances
    batched_samples = random.sample(replay_buffer, config.batch_size) # Frames stored in uint8 [0, 255]
    instances = list(zip(*batched_samples))
    next_states, actions, rewards, current_states, dones = [
        torch.as_tensor(np.asarray(inst), device=config.device, dtype=torch.float32) for inst in instances
    ]
    current_states, next_states = current_states.squeeze(1).to(config.device)/255., next_states.squeeze(1).to(config.device)/255. # (0.0, 1.0)
    # input model
    with torch.no_grad():
        with config.autocast:
            next_Q_val:Tensor = dqn(next_states) # (B, num_actions)

        next_Q_val = next_Q_val.max(dim=1).values
        zero_if_terminal_else_one = 1 - dones # 0 if done==True else 1

        Qtarget:Tensor = (rewards + config.gamma * next_Q_val * zero_if_terminal_else_one) # (B,)
    
    with config.autocast:
        Qpred:Tensor = dqn(current_states) # (B, num_actions)
        # Select Q values of actions that were taken
        Qpred = Qpred.gather(1, actions.unsqueeze(1).long()).squeeze(-1) # (B,)
        loss = nn.functional.smooth_l1_loss(Qpred, Qtarget, beta=1.0)
    loss.backward()
    norm:tp.Optional[Tensor] = None
    try:
        norm = torch.nn.utils.clip_grad_norm_(dqn.parameters(), config.clip_norm, error_if_nonfinite=True)
        optimizer.step() # will not step if norm is inf or NaN
    except RuntimeError as e:
        print(e)
    optimizer.zero_grad()
    return loss.item(), norm


def get_epsilon(step:int, start_eps:float=config.init_eps, end_eps:float=config.final_eps, annealing_steps:int=config.eps_decay_steps) -> float:
    if step < annealing_steps:
        return start_eps + (step / annealing_steps) * (end_eps - start_eps)
    return end_eps


def handle_buffer_to_store(buffer:tuple[Tensor, int, float, Tensor, bool]):
    """
    * assuming elements in the buffer is in CPU
    * convert frames to uint8
    """
    (phi_next, action, reward, phi_prev, done) = buffer
    to_uint8:tp.Callable[[Tensor], Tensor] = lambda phi: (phi*255).to(torch.uint8)
    return (to_uint8(phi_next), action, reward, to_uint8(phi_prev), done)


def main():
    random.seed(SEED)
    np.random.seed(SEED+1)
    torch.manual_seed(SEED+2)
    torch.use_deterministic_algorithms(mode=True, warn_only=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    env = gym.make(ENV_NAME, difficulty=0, obs_type="rgb", full_action_space=False)
    model = get_model(1, 4, 80, 80, log=True, fan_in=4, fan_out=int(env.action_space.n))
    print(model, "\nNumber of parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    model.compile()

    replay_buffer = deque(maxlen=config.buffer_size)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
        fused=True
    )
    optimizer.zero_grad()
    writer = SummaryWriter()

    try:
        print("Training Starting..."); os.makedirs("ckpt", exist_ok=True)
        sum_rewards = []; num_steps_over = 0
        for episode_num in count(1):
            state, info = env.reset()
            phi_prev = torch.cat([
                prepro_state:=prepro(state),
                torch.zeros_like(prepro_state),
                torch.zeros_like(prepro_state),
                torch.zeros_like(prepro_state)
            ]).unsqueeze(0) # (1, 4, 80, 80)
            reward_sum = 0; t0 = time.time()
            for tstep in count(0):
                epsilon = get_epsilon(num_steps_over)
                # agent selects action every 4th frame, it's last action is repeated on the skipped frames # THIS IS ALREADY IMPLEMENTED IN THE ENVIRONMENT
                action = int(sample_action(model, phi_prev, epsilon=epsilon).item())
                assert 0 <= action <= 5, f"Action: {action} is not in the range [0, 5]"
                state, reward, done, truncated, info = env.step(action)
                num_steps_over = info["frame_number"]
                if reward > 0:
                    print(f"\t|| Reward {reward} ||", end="")

                phi_next = torch.cat([prepro(state).unsqueeze(0), phi_prev[:, :-1]], dim=1)
                # convert to cpu to store: CPU memory is cheaper
                replay_buffer.append(handle_buffer_to_store((phi_next.cpu(), action, reward, phi_prev.cpu(), done)))
                phi_prev = phi_next

                if num_steps_over > config.num_warmup_steps:
                    loss, norm = train_step(model, replay_buffer, optimizer)
                    # TENSORBOARD LOGGING
                    writer.add_scalar("Loss", loss, num_steps_over)
                    writer.add_scalar("Gradient Norm", norm, num_steps_over)

                reward_sum += reward
                if done or truncated:
                    break

                # SAVE MODEL
                if num_steps_over % config.save_every_n_steps == 0:
                    print("Saving Model Checkpoint...")
                    torch.save(model.state_dict(), f"ckpt/model{num_steps_over}.pth")
                    
            dt = time.time() - t0
            sum_rewards.append(reward_sum)
            if episode_num % config.logging_every_n_episode == 0:
                print(f"\n| Step: {num_steps_over} | Episode: {episode_num} || Σ Rewards: {reward_sum:<6.2f} |"
                    f"| lr: {config.lr:<12e} || dt: {dt:<5.2f}s || Eps: {epsilon:3f} || INFO: {info} |")
            
            # TENSORBOARD LOGGING
            writer.add_scalar("Sum of Reward", reward_sum, num_steps_over)
            writer.add_scalar("Steps per Episode", tstep, num_steps_over)

            if num_steps_over >= config.num_steps:
                print(f"Training Completed {num_steps_over}..."); break
    except KeyboardInterrupt:
        print("\nTraining Interrupted...")

    writer.close(); env.close()
    torch.save(model.state_dict(), f"ckpt/model_final{num_steps_over}.pth")

    plt.plot(sum_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Sum of Rewards")
    plt.title("Sum of Rewards vs Episode")
    plt.yticks(np.arange(-21, 22))
    plt.savefig(os.path.join("ckpt", "sum_rewards.png"))
    plt.show()

def check_enough_ram(crash_if_no_mem):
    """https://github.com/gordicaleksa/pytorch-learn-reinforcement-learning/blob/main/utils/replay_buffer.py#L169-L185"""
    import psutil
    def to_GBs(memory_in_bytes):
        return f'{memory_in_bytes / 2 ** 30:.2f} GBs'

    available_memory = psutil.virtual_memory().available
    frames = torch.zeros([config.buffer_size] + [4, 80, 80], dtype=torch.uint8)
    actions = torch.zeros([config.buffer_size, 1], dtype=torch.uint8) # [0, 1, 2, 3, 4, 5]
    rewards = torch.zeros([config.buffer_size, 1], dtype=torch.float32)  # [-1, 0, 1]
    dones = torch.zeros([config.buffer_size, 1], dtype=torch.bool) # [True, False]

    required_memory = frames.nbytes + actions.nbytes + rewards.nbytes + dones.nbytes
    print(f'required memory = {to_GBs(required_memory)}, available memory = {to_GBs(available_memory)}')

    if required_memory > available_memory:
        message = f"Not enough memory to store the complete replay buffer! \n" \
                    f"required: {to_GBs(required_memory)} > available: {to_GBs(available_memory)} \n" \
                    f"Page swapping will make your training super slow once you hit your RAM limit." \
                    f"You can either modify replay_buffer_size argument or set crash_if_no_mem to False to ignore it."
        if crash_if_no_mem:
            raise MemoryError(message)
        else:
            print(message)


if __name__ == "__main__":
    check_enough_ram(crash_if_no_mem=True)
    main()
