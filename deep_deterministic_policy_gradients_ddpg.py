import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import typing as tp
from collections import OrderedDict, deque
import matplotlib.animation as anim
import random

import torch
from torch import (
    nn, Tensor
)


ENV_NAME = "Pendulum-v1"
env = gym.make(ENV_NAME)


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
    obs, info = env.reset(); sum_rewards = 0
    with torch.no_grad():
        for step in range(n_max_steps):
            frames.append(env.render())
            action = action_sampler(obs)
            obs, reward, done, truncated, info = env.step(action)
            sum_rewards += reward
            if done or truncated:
                print("done at step", step+1)
                print("sum of rewards", sum_rewards)
                break
    env.close()
    return plot_animation(frames, repeat=repeat, save_path=save_path)


class xonfig:
    action_range:tuple = (float(env.action_space.low[0]), float(env.action_space.high[0]))

    tau:float = 0.002

    num_episodes:int = 2000
    gamma:float = 0.99
    buffer_size:int = 50_000

    device:torch.device = torch.device("cuda" if False else "cpu") # cpu good for small models
    
    batch_size:int = 64
    hidden_dim:int = 64
    actor_lr:float = 5e-4
    dqn_lr:float = 5e-4
    weight_decay:float = 0.0


class ModTanh(nn.Tanh):
    def __init__(self, min_range:int, max_range:int):
        super().__init__()
        assert abs(min_range) == max_range
        self.min = min_range
        self.max = max_range

    def forward(self, x:Tensor):
        # return super().forward(x) * (self.max - self.min) + self.min # rescale tanh output to [min, max] range
        return super().forward(x) * self.max # [-1, 1]*max => [-max, max]


class Actor(nn.Module): # POLICY
    def __init__(self, state_dim:int, action_dim:int, hidden_dim:int):
        super().__init__()
        self.layer1 = nn.Linear(state_dim, hidden_dim); self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim, hidden_dim); self.relu2 = nn.ReLU()
        self.layer3 = nn.Linear(hidden_dim, action_dim); self.modtanh = ModTanh(*xonfig.action_range)

    def forward(self, state:Tensor): # (B, state_dim)
        x = self.relu1(self.layer1(state))
        x = self.relu2(self.layer2(x))
        return self.modtanh(self.layer3(x)) # (B, action_dim=1)
    

class DQN(nn.Module): 
    def __init__(self, state_dim:int, action_dim:int, hidden_dim:int):
        super().__init__()
        self.layer1 = nn.Linear(state_dim + action_dim, hidden_dim); self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim, hidden_dim); self.relu2 = nn.ReLU()
        self.layer3 = nn.Linear(hidden_dim, 1)
        
    def forward(self, state:Tensor, action:Tensor): # (B, state_dim), (B, action_dim=1)
        state_action = torch.cat([state, action], dim=-1) # (B, dim = state_dim + action_dim)
        x = self.relu1(self.layer1(state_action)) # (B, hidden_dim)
        x = self.relu2(self.layer2(x)) # (B, hidden_dim)
        return self.layer3(x) # Q(state, action) # (B, 1)
    

@torch.no_grad()
def update_ema(ema_model:nn.Module, model:nn.Module, decay:float):
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # ema = decay * ema + (1 - decay) * param
        ema_params[name].mul_(decay).add_(param.data, alpha=1-decay)


# adds noise to action, encouraging exploration # TODO: understand the math behind `OrnsteinUhlenbeckActionNoise`
class OrnsteinUhlenbeckActionNoise:
    """ from https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py#L48-L67 """
    def __init__(
        self, mu:np.ndarray, sigma:np.ndarray, theta:float=0.15, dt:float=1e-2, x0:tp.Optional[np.ndarray]=None
    ):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return torch.from_numpy(x).float().to(xonfig.device)

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)


@torch.no_grad()
def sample_action(state:Tensor, noiser:OrnsteinUhlenbeckActionNoise): # (1, state_dim)
    action = torch.clip(
        actor_net(state).cpu().squeeze(0) + noiser(),
        *xonfig.action_range
    ) # (action_dim,)
    return action


def ddpg_train_step(
    states:Tensor,
    actions:Tensor,
    next_states:Tensor,
    rewards:Tensor,
    is_terminal:Tensor
):
    """
    * `states`: `(n, state_dim)`
    * `actions`: `(n, action_dim)`
    * `next_states`: `(n, state_dim)`
    * `rewards`: `(n,)`
    * `is_terminal`: `(n,)`
    """
    rewards, is_terminal = rewards.unsqueeze(-1), is_terminal.unsqueeze(-1) # (n,) -> (n, 1)

    # Optimize DQN/Critic
    with torch.no_grad(): # anyway models in this block are not trainable
        q_next_state = dqn_ema_net(next_states, actor_ema_net(next_states)) # (n, 1)
        q_target = rewards + xonfig.gamma * q_next_state * (1 - is_terminal) # (n, 1)
    q_pred = dqn_net(states, actions) # (n, 1)
    qloss = nn.functional.mse_loss(q_pred, q_target, reduction="sum") # (,)
    qloss.backward()
    dqn_optimizer.step()
    dqn_optimizer.zero_grad()
    
    # Optimize Actor
    dqn_net.requires_grad_(False)
    ## Assuming that the critic Q is a trained model, 
    ## if Q(s, a) is high, then the action a is good, else bad if Q(s, a) is low action a is bad.
    ## we want the actor_net to tweak it's actions such that the Q(s, actor_net(s)) is high (Q is freezed, so Q won't tweak it's weights to make Q(s, actor_net(s)) high)
    ## so we want to maximize Q(s, actor_net(s)) -> minimize -Q(s, actor_net(s))
    actor_loss:Tensor = -dqn_net(states, actor_net(states)).sum()
    actor_loss.backward()
    actor_optimizer.step()
    actor_optimizer.zero_grad()
    dqn_net.requires_grad_(True)


if __name__ == "__main__":
    STATE_DIM = env.observation_space.shape[0]
    ACTION_DIM = env.action_space.shape[0]

    actor_net = Actor(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        hidden_dim=xonfig.hidden_dim
    ); print(actor_net)
    actor_optimizer = torch.optim.AdamW(actor_net.parameters(), lr=xonfig.actor_lr, weight_decay=xonfig.weight_decay)
    actor_ema_net = deepcopy(actor_net).requires_grad_(False)

    dqn_net = DQN(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        hidden_dim=xonfig.hidden_dim
    ); print(dqn_net)
    dqn_optimizer = torch.optim.AdamW(dqn_net.parameters(), lr=xonfig.dqn_lr, weight_decay=xonfig.weight_decay)
    dqn_ema_net = deepcopy(dqn_net).requires_grad_(False)

    replay_buffer = deque(maxlen=xonfig.buffer_size)
    action_noiser = OrnsteinUhlenbeckActionNoise(mu=np.zeros(1), sigma=np.ones(1) * 0.05)


    sum_rewards_list = []
    for episode in range(xonfig.num_episodes):
        state, info = env.reset()
        state = torch.as_tensor(state, dtype=torch.float32, device=xonfig.device) # (state_dim,)
        sum_rewards = float(0)
        while True:
            # sample action
            action = sample_action(state.unsqueeze(0), noiser=action_noiser) # (action_dim,)

            # input action into environment to get rewards, next_state,
            next_state, reward, done, truncated, info = env.step(action.numpy())
            next_state = torch.as_tensor(next_state, dtype=torch.float32, device=xonfig.device)
            sum_rewards += reward

            # add to buffer
            replay_buffer.append((
                next_state.cpu(), action.cpu(), torch.as_tensor(reward).cpu(), 
                state.cpu(), torch.as_tensor(done).cpu()
            ))
            
            # train step
            if len(replay_buffer) > xonfig.batch_size*5:
                batched_samples = random.sample(replay_buffer, xonfig.batch_size)
                instances = list(zip(*batched_samples))
                next_states, actions, rewards, states, dones = [
                    torch.as_tensor(np.asarray(inst), device=xonfig.device, dtype=torch.float32) for inst in instances
                ]
                ddpg_train_step(states, actions, next_states, rewards, dones)
                update_ema(actor_ema_net, actor_net, decay=1-xonfig.tau)
                update_ema(dqn_ema_net, dqn_net, decay=1-xonfig.tau)
            
            if done or truncated:
                break

            state = next_state

        # logging
        print(f"|| Episode: {episode} || Sum of Rewards: {sum_rewards:.4f} ||")
        sum_rewards_list.append(sum_rewards)

    plt.plot(sum_rewards_list)
    plt.xlabel("Episodes")
    plt.ylabel("Sum of Rewards")
    plt.yticks(np.arange(min(sum_rewards_list)+1, 0+1, 100).tolist()+[0])
    plt.title(f"DDPG Algorithm on {env.spec.id}")
    plt.savefig(f"images/ddpg_on_{ENV_NAME}.png")
    plt.grid()
    plt.show()

    def action_sampler(state:np.ndarray):
        action = torch.clip(
            actor_net(torch.as_tensor(state, dtype=torch.float, device=xonfig.device)).cpu(),
            *xonfig.action_range
        ).numpy()
        return action
    
    show_one_episode(action_sampler, save_path="images/ddpg_pendulum.gif")

