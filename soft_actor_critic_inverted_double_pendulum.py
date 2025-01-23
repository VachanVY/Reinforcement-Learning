import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import typing as tp
from collections import OrderedDict, deque
import matplotlib.animation as anim
import random
from dataclasses import dataclass
from itertools import count

import torch
from torch import nn, Tensor


ENV_NAME = "InvertedDoublePendulum-v5"


def update_scene(num, frames, patch):
    patch.set_data(frames[num])
    return patch,

def plot_animation(frames:list, save_path:tp.Optional[str]=None, repeat=False, interval=40, title:tp.Optional[str]=None):
    fig = plt.figure()
    patch = plt.imshow(frames[0])
    plt.axis('off')
    if title:
        plt.title(title)
    animation = anim.FuncAnimation(
        fig, update_scene, fargs=(frames, patch),
        frames=len(frames), repeat=repeat, interval=interval)
    if save_path is not None:
        animation.save(save_path, writer="pillow", fps=20)
    return animation

def show_one_episode(action_sampler:tp.Callable, save_path:tp.Optional[str]=None, repeat=False, title:tp.Optional[str]=None):
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
    return plot_animation(frames, repeat=repeat, save_path=save_path, title=title)


@dataclass
class xonfig:
    num_episodes:int = 1000
    gamma:float = 0.99
    device:torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_updates:int = 1 # idk why but for 10 updates, result's weren't so great...

    adaptive_alpha:bool = True
    alpha:float = 0.12 # initial value
    tau: float = 0.005

    buffer_size:int = 50_000
    batch_size:int = 64
    dqn_lr:float = 5e-4
    actor_lr:float = 5e-4
    alpha_lr:float = 5e-4
    weight_decay:float = 0.0
    
    hidden_dim:int = 64


class ActorPolicy(nn.Module):
    def __init__(
        self, 
        state_dims:int, 
        hidden_dim:int,
        action_dims:int,
    ):
        super().__init__()
        self.l1 = nn.Linear(state_dims, hidden_dim); self.relu1 = nn.ReLU()
        self.l2 = nn.Linear(hidden_dim, hidden_dim); self.relu2 = nn.ReLU()

        self.mu_mean = nn.Linear(hidden_dim, action_dims)
        self.sigma_log_std = nn.Linear(hidden_dim, action_dims)


    def forward(self, state:Tensor):
        x = self.relu1(self.l1(state))
        x = self.relu2(self.l2(x))

        mu = self.mu_mean(x)
        # If log_std is too small (e.g. log_std << 20, the standard deviation becomes extremely close to zero, leading to highly peaked distributions. 
        # This can cause numerical issues like exploding gradients or division by near-zero values during backpropagation

        # if log_std is too large (e.g., log_std > 2), the standard deviation becomes excessively large, leading to very high-variance policies, 
        # high exploration, and poor convergence.

        # [-20, 2] has been found to work well across a variety of continuous control tasks in reinforcement learning
        std = torch.clip(self.sigma_log_std(x), -20, 2).exp() # after exp bounds => (2.06e-09, 7.3890)
        return mu, std
    

class CriticActionValue(nn.Module):
    def __init__(self, state_dims:int, hidden_dim:int, action_dims:int):
        super().__init__()
        self.l1 = nn.Linear(state_dims + action_dims, hidden_dim); self.relu1 = nn.ReLU()
        self.l2 = nn.Linear(hidden_dim, hidden_dim); self.relu2 = nn.ReLU()
        self.l3 = nn.Linear(hidden_dim, 1)

    def forward(
        self, 
        state:Tensor, # (B, state_dims)
        action:Tensor # (B, action_dims)
    ):
        x = torch.cat([state, action], dim=-1) # (B, dim = state_dims + action_dims)
        x = self.relu1(self.l1(x))
        x = self.relu2(self.l2(x))
        q_value = self.l3(x)
        return q_value # (B, 1)
    

@torch.no_grad()
def update_ema(ema_model:nn.Module, model:nn.Module, decay:float):
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # ema = decay * ema + (1 - decay) * param
        ema_params[name].mul_(decay).add_(param.data, alpha=1-decay)


def sample_actions(state:Tensor, actions_max_bound:float):
    mu, std = policy_net(state)
    dist = torch.distributions.Normal(mu, std)
    unbound_action = dist.rsample() # (B, actions_dim) # dist.sample() is torch.no_grad() mode
    action = torch.tanh(unbound_action)*actions_max_bound # [-1, 1] * max => [-max, max]

    # CHATGPTed the tanh correction: TODO: Understand this
    log_prob = dist.log_prob(unbound_action) - torch.log(1 - action.pow(2) + 1e-6)  # Tanh correction
    log_prob:Tensor = log_prob.sum(dim=-1, keepdim=True)  # Sum over action dimensions
    return action, log_prob
    # return action, dist


@torch.compile()
def sac_train_step(
    states:Tensor,
    actions:Tensor,
    next_states:Tensor,
    rewards:Tensor,
    is_terminal:Tensor
):
    """
    * `states`: `(B, state_dim)`
    * `actions`: `(B, action_dim)`
    * `next_states`: `(B, state_dim)`
    * `rewards`: `(B,)`
    * `is_terminal`: `(B,)`
    """
    rewards, is_terminal = rewards.unsqueeze(-1), is_terminal.unsqueeze(-1) # (B,) => (B, 1)    

    # Optimize DQNs
    ## a_next ~ π(s_next)
    ## get target Q values: y = r + γ * ( Q_target(s_next, a_next) - α * log(π(a_next|s_next)) ) * (1 - is_terminal)
    ## L1 = MSE(Q1(s, a), y) ## L2 = MSE(Q2(s, a), y) ## optimize loss (L1, L2)
    with torch.no_grad():
        actions_next, log_prob = sample_actions(next_states, ACTION_BOUNDS)
        q_next1, q_next2 = dqn_target1(next_states, actions_next), dqn_target2(next_states, actions_next) # (B, 1), (B, 1)
        # why sum on log prob? because dist.log_prob(a_next) is (B, action_dim) and we want to sum over action_dim to get (B, 1)
        # why min of the 2 q values? To avoid maximization bias, see https://arxiv.org/abs/1812.05905
        q_next:Tensor = torch.min(q_next1, q_next2) - xonfig.alpha * log_prob # (B, 1)
        q_next_target:Tensor = rewards + xonfig.gamma * q_next * (1 - is_terminal) # (B, 1)
    
    dqn1_loss = nn.functional.mse_loss(dqn1(states, actions), q_next_target, reduction="mean")
    dqn2_loss = nn.functional.mse_loss(dqn2(states, actions), q_next_target, reduction="mean")
    (dqn1_loss + dqn2_loss).backward()
    # dqn1_loss.backward(); dqn2_loss.backward()
    dqn1_optimizer.step(); dqn2_optimizer.step()
    dqn1_optimizer.zero_grad(); dqn2_optimizer.zero_grad()

    # Optimize Policy
    dqn1.requires_grad_(False); dqn2.requires_grad_(False)
    actions, log_probs = sample_actions(states, ACTION_BOUNDS)
    ## maximize entropy, minimize negative entropy
    ## maximize q value by minimizing -q value, tweaks the policy weights through the actions to maximize q value, doesn't tweak the q network itself as they are freezed
    pi_loss:Tensor = (xonfig.alpha * log_probs - torch.min(dqn1(states, actions), dqn2(states, actions))).mean()
    pi_loss.backward()
    policy_optimizer.step()
    policy_optimizer.zero_grad()
    dqn1.requires_grad_(True); dqn2.requires_grad_(True)

    # Optimize Alpha
    if xonfig.adaptive_alpha:
        alpha_loss = -log_alpha * (log_prob + target_entropy).mean()
        alpha_loss.backward()
        alpha_optimizer.step()
        alpha_optimizer.zero_grad()
        xonfig.alpha = log_alpha.exp().item()


if __name__ == "__main__":
    env = gym.make(ENV_NAME)
    STATE_DIMS = env.observation_space.shape[0] # 24
    ACTION_DIMS = env.action_space.shape[0] # 4
    ACTION_BOUNDS = [env.action_space.low, env.action_space.high] # [array([-1., -1., -1., -1.], dtype=float32), array([1., 1., 1., 1.], dtype=float32)]
    ACTION_BOUNDS = ACTION_BOUNDS[1][0] # 1.0

    if xonfig.adaptive_alpha:
        log_alpha = torch.nn.Parameter(
            torch.tensor(np.log(xonfig.alpha)),
            requires_grad=True
        )
        target_entropy = torch.tensor(-ACTION_DIMS, device=xonfig.device, dtype=torch.float32)
        alpha_optimizer = torch.optim.AdamW([log_alpha], lr=xonfig.alpha_lr, weight_decay=0.0)

    dqn1 = CriticActionValue(STATE_DIMS, xonfig.hidden_dim, ACTION_DIMS).to(xonfig.device); dqn1.compile()
    dqn2 = deepcopy(dqn1)
    dqn_target1 = deepcopy(dqn1).requires_grad_(False)
    dqn_target2 = deepcopy(dqn2).requires_grad_(False)
    dqn1_optimizer = torch.optim.AdamW(dqn1.parameters(), lr=xonfig.dqn_lr, weight_decay=xonfig.weight_decay)
    dqn2_optimizer = torch.optim.AdamW(dqn2.parameters(), lr=xonfig.dqn_lr, weight_decay=xonfig.weight_decay)

    policy_net = ActorPolicy(STATE_DIMS, xonfig.hidden_dim, ACTION_DIMS).to(xonfig.device); policy_net.compile()
    policy_optimizer = torch.optim.AdamW(policy_net.parameters(), lr=xonfig.actor_lr, weight_decay=xonfig.weight_decay)

    replay_buffer = deque(maxlen=xonfig.buffer_size)

    sum_rewards_list = []; num_timesteps_list = []
    try:
        for episode in range(1, xonfig.num_episodes+1):
            state, info = env.reset()
            state = torch.as_tensor(state, device=xonfig.device, dtype=torch.float32)
            sum_rewards = float(0)
            for tstep in count(1):
                # sample action from policy
                with torch.no_grad():
                    action, _log_prob = sample_actions(state.unsqueeze(0), ACTION_BOUNDS) # (1, actions_dims)
                    action = action.squeeze(0) # (action_dims,)

                # action into the environment and get the next state and reward
                next_state, reward, done, truncated, info = env.step(action.cpu().detach().numpy())
                next_state = torch.as_tensor(next_state, dtype=torch.float32, device=xonfig.device)
                sum_rewards += reward

                # store the transition in the replay buffer
                replay_buffer.append((
                    next_state.cpu(), action.cpu(), torch.as_tensor(reward).cpu(),
                    state.cpu(), torch.as_tensor(done).cpu()
                ))

                # optimize networks
                if len(replay_buffer) >= xonfig.batch_size*5:
                    for _ in range(xonfig.num_updates):
                        batched_samples = random.sample(replay_buffer, xonfig.batch_size)
                        next_states, actions, rewards, states, dones = [
                            torch.as_tensor(np.asarray(inst), device=xonfig.device, dtype=torch.float32) for inst in list(zip(*batched_samples))
                        ] # (B, state_dim), (B, action_dim), (B,), (B, state_dim), (B,)
                        sac_train_step(states, actions, next_states, rewards, dones)
                        update_ema(dqn_target1, dqn1, decay=1 - xonfig.tau)
                        update_ema(dqn_target2, dqn2, decay=1 - xonfig.tau)

                if done or truncated:
                    break

                state = next_state

            sum_rewards_list.append(sum_rewards)
            num_timesteps_list.append(tstep)
            print(f"|| Episode: {episode} || Sum of Rewards: {sum_rewards:.4f} || Timesteps: {tstep} ||")
    except KeyboardInterrupt:
        print("Training Interrupted")

    adaptive_str = 'adaptive_alpha' if xonfig.adaptive_alpha else ''
    show_one_episode(
        lambda x: sample_actions(torch.as_tensor(x, dtype=torch.float32, device=xonfig.device).unsqueeze(0), ACTION_BOUNDS)[0].squeeze(0).cpu().numpy(),
        save_path=f"images/sac_{ENV_NAME.lower()}_.gif",
        title=f"SAC Trained Agent {f'{adaptive_str} ' if adaptive_str else ''}"
    ); plt.close()


    plt.plot(sum_rewards_list, label="Sum of Rewards")
    plt.plot(num_timesteps_list, label="Timesteps")
    plt.yticks(np.arange(0, mx:=max(sum_rewards_list), mx//10).tolist())
    plt.xlabel("Episodes")
    plt.grid(True)
    plt.legend()
    plt.ylabel("Sum of Rewards")
    plt.title("Sum of Rewards per Episode")
    plt.savefig(f"images/sac_rewards_{adaptive_str}_{ENV_NAME}.png")
    plt.show()
    plt.close()