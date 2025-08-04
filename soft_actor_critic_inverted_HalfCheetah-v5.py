import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import typing as tp
from collections import OrderedDict, deque
import matplotlib.animation as anim
import random
import dataclasses as dc
from itertools import count
import time
import os

DETERMINISTIC = False
if DETERMINISTIC:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch
from torch import nn, Tensor

from torch.utils.tensorboard import SummaryWriter


ENV_NAME = "HalfCheetah-v5"


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

@torch.no_grad()
def show_one_episode(
    env:gym.Env, action_sampler: tp.Callable,
    save_path: tp.Optional[str] = None,
    repeat: bool=False
):
    frames = []
    
    obs, info = env.reset(); sum_rewards = 0

    for step in count(0):
        frames.append(env.render())
        action = action_sampler(obs)
        obs, reward, done, truncated, info = env.step(action)
        sum_rewards += info["episode"]["r"] if "episode" in info else reward
        if done or truncated:
            print("done at step", step+1)
            print("sum of rewards", sum_rewards)
            break

    env.close()
    return plot_animation(frames, repeat=repeat, save_path=save_path)


@dc.dataclass
class xonfig:
    num_episodes:int = 100
    gamma:float = 0.99

    device:torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype:torch.dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32

    # num_updates:int = 1
    # update_every_n_steps:int = 1
    start_learning:int = 5000

    adaptive_alpha:bool = True
    alpha:float = 0.2 # initial value
    tau: float = 0.005

    buffer_size:int = 1_000_000
    batch_size:int = 256
    dqn_lr:float = 1e-3
    actor_lr:float = 3e-4
    alpha_lr:float = 1e-3
    weight_decay:float = 0.0
    hidden_dim:int = 256

    actor_update_freq:int = 2
    target_network_update_freq:int = 1

    logg_tb:bool = True
    run_name:str = f"runs_sac_{ENV_NAME}"


def make_env(env:gym.Env, seed:tp.Optional[int], normalize_rewards:bool, normalize_obs:bool, capture_video:bool):
    if capture_video:
        env = gym.wrappers.RecordVideo(env, f"{time.time()}_{ENV_NAME}")
    if normalize_rewards:
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.NormalizeReward(env)
    if normalize_obs:
        env = gym.wrappers.NormalizeObservation(env)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


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
        # NOTE: experiment with (-5, 2) clip bounds 
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


def sample_actions(state:Tensor, actions_max_bound:float): # (B, state_dims)
    mu, std = policy_net(state) # (B, action_dims), (B, action_dims)
    dist = torch.distributions.Normal(mu, std)
    unbound_action = dist.rsample() # (B, actions_dim) # dist.sample() is torch.no_grad() mode
    action = torch.tanh(unbound_action)*actions_max_bound # [-1, 1] * max => [-max, max]

    # Tanh correction: TODO add intuition
    log_prob = dist.log_prob(unbound_action) - torch.log(1 - action.pow(2) + 1e-6) # (B, action_dim)
    
    ## why sum on log prob? because dist.log_prob(a_next) is (B, action_dim) and we want to sum over action_dim to get (B, 1)
    ## Each action dimension is independent, so the total log-prob of a joint action vector
    ### Imagine you have two independent clocks, each ticking with its own probability. 
    ### The chance of clock 1 showing x and clock 2 showing y is the product of their individual chances:
    ###     p(x, y) = p1(x) * p2(y)
    ### In log-space, that product becomes a sum:
    ###     log p(x, y) = log p1(x) + log p2(y)
    ### Now generalize this to a d-dimensional action vector. 
    ### If each component of the action is sampled independently from a Gaussian distribution, 
    ### then the joint probability of the entire action is the product of all d marginal probabilities:
    ###     p(a) = p1(a1) * p2(a2) * ... * pd(ad)
    ### Taking logs:
    ###     log p(a) = log p1(a1) + log p2(a2) + ... + log pd(ad)
    log_prob:Tensor = log_prob.sum(dim=-1, keepdim=True)  # Sum over action dimensions
    return action, log_prob


def sac_train_step(
    states:Tensor,
    actions:Tensor,
    next_states:Tensor,
    rewards:Tensor,
    is_terminal:Tensor,
    global_step:int
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
        q_next1:Tensor; q_next2:Tensor
        q_next1, q_next2 = dqn_target1(next_states, actions_next), dqn_target2(next_states, actions_next) # (B, 1), (B, 1)
        
        # why min of the two q values? To avoid maximization bias, see https://arxiv.org/abs/1812.05905
        q_next:Tensor = torch.min(q_next1, q_next2) - xonfig.alpha * log_prob # (B, 1)
        q_next_target:Tensor = rewards + xonfig.gamma * q_next * (1 - is_terminal) # (B, 1)
    
    dqn1_loss = nn.functional.mse_loss(dqn1(states, actions), q_next_target, reduction="mean")
    dqn2_loss = nn.functional.mse_loss(dqn2(states, actions), q_next_target, reduction="mean")
    (dqn1_loss + dqn2_loss).backward() # dqn1_loss.backward(); dqn2_loss.backward()
    dqn1_optimizer.step(); dqn2_optimizer.step()
    dqn1_optimizer.zero_grad(); dqn2_optimizer.zero_grad()

    # Optimize Policy
    if global_step % xonfig.actor_update_freq == 0:
        dqn1.requires_grad_(False); dqn2.requires_grad_(False)
        for _ in range(xonfig.actor_update_freq):
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

    if xonfig.logg_tb:
        writer.add_scalar("sac_update/Q_next1", q_next1.mean().item(), global_step=global_step)
        writer.add_scalar("sac_update/Q_next2", q_next2.mean().item(), global_step=global_step)
        writer.add_scalar("sac_update/Q_next_target", q_next_target.mean().item(), global_step=global_step)
        writer.add_scalar("sac_update/Q_next", q_next.mean().item(), global_step=global_step)
        writer.add_scalar("sac_update/dqn1_loss", dqn1_loss.item(), global_step=global_step)
        writer.add_scalar("sac_update/dqn2_loss", dqn2_loss.item(), global_step=global_step)
        if global_step % xonfig.actor_update_freq == 0:
            writer.add_scalar("sac_update/pi_loss", pi_loss.item(), global_step=global_step)
            writer.add_scalar("sac_update/log_probs", log_probs.mean().item(), global_step=global_step)
        if xonfig.adaptive_alpha:
            writer.add_scalar("sac_update/alpha", xonfig.alpha, global_step=global_step)
        

if __name__ == "__main__":
    SEED = 0
    random.seed(SEED)
    np.random.seed(SEED+1)
    torch.manual_seed(SEED+2)
    if DETERMINISTIC:
        torch.use_deterministic_algorithms(mode=True, warn_only=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if xonfig.logg_tb:
        writer = SummaryWriter(
            log_dir=f"runs_SAC_{ENV_NAME}",
            filename_suffix=f"{xonfig.run_name}"
        )
        writer.add_text(
            "Training Configuration",
            str(dc.asdict(xonfig()))
        )

    raw_env = gym.make(ENV_NAME, render_mode="rgb_array")
    NORMALIZE_REW = True
    env = make_env(
        raw_env,
        seed=SEED+3,
        normalize_rewards=NORMALIZE_REW,
        normalize_obs=True, ######################################## experiment with this
        capture_video=False
    )
    STATE_DIMS = raw_env.observation_space.shape[0]
    ACTION_DIMS = raw_env.action_space.shape[0]
    ACTION_BOUNDS = [raw_env.action_space.low, raw_env.action_space.high]
    ACTION_BOUNDS = ACTION_BOUNDS[1][0]

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
    autocast = torch.autocast(
        device_type=xonfig.device.type,
        dtype=xonfig.dtype,
        enabled=(xonfig.device.type == "cuda" and xonfig.dtype == torch.bfloat16),
    )

    sum_rewards_list = []; episode_length_list = []; num_steps_over = int(1)
    try:
        for episode in range(1, xonfig.num_episodes+1):
            state, info = env.reset()
            state = torch.as_tensor(state, device=xonfig.device, dtype=torch.float32)
            sum_rewards = float(0)
            for tstep in count(1):
                # sample action from policy
                with torch.no_grad():
                    if num_steps_over > xonfig.start_learning:
                        action, _log_prob = sample_actions(state.unsqueeze(0), ACTION_BOUNDS) # (1, actions_dims)
                        action = action.squeeze(0) # (action_dims,)
                    else:
                        action = torch.as_tensor(env.action_space.sample(), device=xonfig.device, dtype=torch.float32) # (action_dims,)

                # action into the environment and get the next state and reward
                next_state, reward, done, truncated, info = env.step(action.cpu().detach().numpy())
                next_state = torch.as_tensor(next_state, dtype=torch.float32, device=xonfig.device)
                if NORMALIZE_REW:
                    stats:dict = info.get("episode", {})
                    unnormalized_rewards = stats.get("r", 0)
                    sum_rewards += unnormalized_rewards
                    if xonfig.logg_tb: writer.add_scalar("rewards", unnormalized_rewards, global_step=num_steps_over)
                else: 
                    sum_rewards += reward
                    if xonfig.logg_tb: writer.add_scalar("rewards", reward, global_step=num_steps_over)


                # store the transition in the replay buffer
                replay_buffer.append((
                    next_state.cpu(), action.cpu(), torch.as_tensor(reward).cpu(),
                    state.cpu(), torch.as_tensor(done).cpu()
                ))

                # optimize networks
                if num_steps_over > xonfig.start_learning:
                    batched_samples = random.sample(replay_buffer, xonfig.batch_size)
                    next_states, actions, rewards, states, dones = [
                        torch.as_tensor(np.asarray(inst), device=xonfig.device, dtype=torch.float32) for inst in list(zip(*batched_samples))
                    ] # (B, state_dim), (B, action_dim), (B,), (B, state_dim), (B,)
                    sac_train_step(states, actions, next_states, rewards, dones, global_step=num_steps_over)
                    if num_steps_over % xonfig.target_network_update_freq == 0:
                        update_ema(dqn_target1, dqn1, decay=1 - xonfig.tau)
                        update_ema(dqn_target2, dqn2, decay=1 - xonfig.tau)
                
                num_steps_over += 1
                if done or truncated:
                    break

                state = next_state

            sum_rewards_list.append(sum_rewards)
            episode_length_list.append(tstep)
            if xonfig.logg_tb:
                writer.add_scalar("episode_length", tstep, global_step=episode)
                writer.add_scalar("sum_rewards", sum_rewards, global_step=episode)
            print(
                f"|| Episode: {episode} || Sum of Rewards: {sum_rewards:.4f} || Timesteps: {tstep} |"
                f"| Num Steps: {num_steps_over} || Alpha: {xonfig.alpha} ||"
            )
    except KeyboardInterrupt:
        print("Training Interrupted")
    finally:
        torch.save(
            obj={
                "policy_net": policy_net.state_dict(),
                "dqn1": dqn1.state_dict(),
                "dqn2": dqn2.state_dict(),
                "dqn_target1": dqn_target1.state_dict(),
                "dqn_target2": dqn_target2.state_dict(),
                "env": env,
            },
            f=f"ckpt/SAC_{ENV_NAME}_final.pt",
        )

    adaptive_str = 'adaptive_alpha' if xonfig.adaptive_alpha else ''

    def get_deterministic_actions(state:Tensor, action_bounds=ACTION_BOUNDS):
        state = state.unsqueeze(0) # (1, state_dims)
        mu, _ = policy_net(state)
        action = torch.tanh(mu).mul(action_bounds)
        action = action.squeeze(0).cpu().numpy()
        return action

    show_one_episode(
        env,
        lambda x: get_deterministic_actions(torch.as_tensor(x, dtype=torch.float32, device=xonfig.device), ACTION_BOUNDS),
        save_path=f"images/SAC_{ENV_NAME}_deterministic.gif",
    ); plt.close()
    show_one_episode(
        env,
        lambda x: sample_actions(torch.as_tensor(x, dtype=torch.float32, device=xonfig.device), ACTION_BOUNDS)[0].cpu().numpy(),
        save_path=f"images/SAC_{ENV_NAME}.gif",
    ); plt.close()

    def moving_average(x, w):
        return np.convolve(x, np.ones(w)/w, mode='valid')

    window = 20
    episodes = range(len(sum_rewards_list))
    smoothed_rewards = moving_average(sum_rewards_list, window)
    smoothed_lengths = moving_average(episode_length_list, window)

    axes:list[plt.Axes]
    fig, axes = plt.subplots(2, 1, figsize=(10, 15))

    axes[0].plot(sum_rewards_list, label="Sum of Rewards")
    axes[0].plot(episodes[window-1:], smoothed_rewards, label=f"Smoothed Sum of Rewards (window={window})", color="red")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Sum of Rewards")
    axes[0].set_title("Sum of Rewards vs Episode")
    axes[0].legend()
    axes[0].grid()

    axes[1].plot(episode_length_list, label="Episode Length", color="orange")
    axes[1].plot(episodes[window-1:], smoothed_lengths, label=f"Smoothed Episode Length (window={window})", color="purple")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Episode Length")
    axes[1].set_title("Episode Length vs Episode")
    axes[1].legend()
    axes[1].grid()

    plt.tight_layout()
    plt.show()
    plt.savefig(f"images/SAC_Cont_Actions_{ENV_NAME}{xonfig.run_name}_training_curves.png")
    plt.close(fig)
    env.close()
    if xonfig.logg_tb:
        writer.close()
    print("Training Finished")