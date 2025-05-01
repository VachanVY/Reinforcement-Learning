import gymnasium as gym
from itertools import count
from collections import deque
import typing as tp
from tqdm import trange
import dataclasses as dc
import time
import random
import numpy as np
import os
from copy import deepcopy
import matplotlib.pyplot as plt

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch
from torch import nn, Tensor
from torch.utils.tensorboard import SummaryWriter


def make_env(env, seed, capture_video, run_name):
    env = gym.wrappers.RecordEpisodeStatistics(env)
    if capture_video:
        env = gym.wrappers.RecordVideo(env, f"{run_name}_{ENV_NAME}")
    env = gym.wrappers.ClipAction(env)
    env = gym.wrappers.NormalizeObservation(env)
    env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, *ACTION_BOUNDS), None)
    env = gym.wrappers.NormalizeReward(env)
    env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


from matplotlib import animation as anim

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

def show_one_episode(env, action_sampler: tp.Callable, save_path: tp.Optional[str] = None,
                     repeat=False):
    frames = []
        
    obs, info = env.reset(); sum_rewards = 0

    with torch.no_grad():
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
    max_steps:int = int(6e5)
    save_model_every:int = int(2e5)

    # General RL hyperparameters
    num_episodes:int = 0
    gamma:float = 0.99
    update_every:int = 2000
    # PPO hyperparameters
    K:int = 10 # num ppo iterations
    clip_range:float = 0.2
    target_kl:tp.Optional[float] = None

    # Action noise hyperparameters
    init_action_std:float = 0.5
    action_std_decay_rate:float = 0.0
    min_action_std:float = 0.1
    decay_std_every:int = int(max_steps/12)

    # General Training hyperparameters
    device:torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype:torch.dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32
    batch_size:int = 64
    weight_decay:float = 0e-4
    actor_lr:float = 3e-4
    critic_lr:float = 3e-4
    clip_norm:float = 0.0
    hidden_dim:int = 64
    beta1:float = 0.9
    beta2:float = 0.99

    logg_losses:bool = False
    logg_tb:bool = False # tensorboard logging


class Buffer:
    def __init__(self):
        self.clear()

    def clear(self):
        self.states       = []
        self.actions      = []
        self.rewards      = []
        self.dones        = []
        self.log_probs    = []
        self.state_values = []
    
    def store(
        self, 
        state:Tensor, 
        action:Tensor, 
        action_logprob:Tensor, 
        state_val:Tensor, 
        reward:int|float, 
        done:bool 
    ):
        self.states.append(state.cpu())
        self.actions.append(action.cpu())
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(action_logprob.cpu())
        self.state_values.append(state_val.cpu())


def get_discounted_returns(rewards:tp.Sequence, is_terminals:tp.Sequence, gamma:float):
    discounted_returns = deque()
    discounted_reward = 0
    for reward, is_terminal in zip(reversed(rewards), reversed(is_terminals)):
        if is_terminal: discounted_reward = 0
        discounted_reward = reward + gamma * discounted_reward
        discounted_returns.appendleft(float(discounted_reward))
    return list(discounted_returns)


class ValueNet(nn.Module):
    def __init__(self, state_dim:int, hidden_dim:int):
        super().__init__()
        self.linear1 = nn.Linear(state_dim, hidden_dim); self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, hidden_dim); self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(hidden_dim, 1)

    def forward(self, x:Tensor): # state: (B, state_dim)
        x = self.linear1(x); x = self.relu1(x)
        x = self.linear2(x); x = self.relu2(x)
        x = self.linear3(x); 
        return x


class PolicyNet(nn.Module):
    def __init__(self, state_dim:int, hidden_dim:int, num_actions:int, init_action_std:float):
        super().__init__()
        self.linear1 = nn.Linear(state_dim, hidden_dim); self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, hidden_dim); self.relu2 = nn.ReLU()

        self.dist_mean = nn.Linear(hidden_dim, num_actions)
        import math
        dist_log_std = nn.Parameter(
            torch.full((num_actions,), fill_value=math.log(init_action_std)), requires_grad=False ############# experimenting with True/False
        )
        self.dist_log_std:Tensor
        if not dist_log_std.requires_grad:
            self.register_buffer("dist_log_std", dist_log_std)
        else:
            self.dist_log_std = dist_log_std

    def forward(self, x:Tensor): # state: (B, state_dim)
        x = self.linear1(x); x = self.relu1(x)
        x = self.linear2(x); x = self.relu2(x)

        mean = self.dist_mean(x)
        std = torch.clamp(self.dist_log_std, min=-20, max=2).exp()
        std = std.expand_as(mean) # (B, num_actions)
        return mean, std # (B, num_actions), (B, num_actions)
    
    def decay_std(self, action_std_decay_rate:float, min_action_std:float):
        # https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/PPO.py#L159-L173
        min_std = torch.tensor(min_action_std, device=self.dist_log_std.device)
        new_std = self.dist_log_std.exp() - action_std_decay_rate
        self.dist_log_std.copy_(new_std.clip(min=min_std).log())


class ActorCritic(nn.Module):
    def __init__(self, state_dim:int, hidden_dim:int, num_actions:int, init_action_std:float):
        super().__init__()
        self.policy_net = PolicyNet(state_dim, hidden_dim, num_actions, init_action_std=init_action_std)
        self.value_net = ValueNet(state_dim, hidden_dim)

    def forward(self, state:Tensor): # state: (B, state_dim)
        mean:Tensor; std:Tensor
        state_value:Tensor
        mean, std = self.policy_net(state) # (B, num_actions), (B, num_actions)
        state_value = self.value_net(state) # (B, 1)
        return (mean, std), state_value # (B, num_actions), (B, num_actions), (B, 1)
    
    def sample_action(self, state:Tensor, deterministic:bool=False) -> tuple[Tensor, Tensor, Tensor]:
        mean:Tensor; std:Tensor; state_val:Tensor
        (mean, std), state_val = self(state) # (B=1, num_actions), (B=1, num_actions), (B=1, 1)
        if deterministic:
            return mean, state_val, None
        covariance_matrix = torch.diag_embed(std.pow(2)) # (B=1, num_actions, num_actions)
        dist = torch.distributions.MultivariateNormal(
            loc=mean, covariance_matrix=covariance_matrix
        )
        action = dist.rsample() # (B=1, num_actions)
        action_logprobas = dist.log_prob(action) # (B=1,) # log(P(a1, a2, ...)) # already summed
        return action, state_val, action_logprobas
    
    def decay_std(self, action_std_decay_rate:float, min_action_std:float):
        self.policy_net.decay_std(action_std_decay_rate, min_action_std)


def update(total_steps_done:int):
    buf_states = torch.stack(buffer.states).to(xonfig.device).detach() # (B, state_dim)
    buf_actions = torch.stack(buffer.actions).to(xonfig.device).detach() # (B, num_actions)
    buf_returns = torch.tensor(
        get_discounted_returns(buffer.rewards, buffer.dones, xonfig.gamma),
        device=xonfig.device
    ).unsqueeze(-1).detach(); buf_size = len(buf_returns)

    if buf_size > 1:
        buf_returns = ((buf_returns - buf_returns.mean()) / (buf_returns.std() + 1e-6)).detach()
    
    buf_state_vals = torch.stack(buffer.state_values).to(xonfig.device).detach() # (B, 1)
    buf_action_logproba = torch.stack(buffer.log_probs).to(xonfig.device).detach() # (B, 1)

    advantages = (buf_returns - buf_state_vals).detach()

    # K Epochs
    losses = {"policy": [], "value": []}
    kldivs_list = []
    for _ in range(xonfig.K):
        for i in range(0, buf_size - xonfig.batch_size + 1, xonfig.batch_size):
            batch_idx = slice(i, min(i + xonfig.batch_size, buf_size))
            batch_returns = buf_returns[batch_idx] # (B, 1)
            batch_advantages = advantages[batch_idx] # (B, 1)
            batch_states = buf_states[batch_idx] # (B, state_dim)
            batch_actions = buf_actions[batch_idx] # (B, action_dim)
            batch_action_logprobs = buf_action_logproba[batch_idx] # (B, 1)

            # Compute advantage
            with autocast:  # Ensure all forward passes and loss computations are inside this block
                mean:Tensor; std:Tensor; state_value:Tensor
                (mean, std), state_value = actor_critic(batch_states) # ((B, num_actions), (B, num_actions)), (B, 1)

                # compute new action logprobas
                covariance_matrix = torch.diag_embed(std.pow(2)) # (B, num_actions, num_actions)
                dist = torch.distributions.MultivariateNormal(
                    loc=mean, covariance_matrix=covariance_matrix
                )
                new_action_logprobas:Tensor = dist.log_prob(batch_actions).unsqueeze(-1) # (B, 1)

                # value loss
                value_loss = nn.functional.mse_loss(state_value, batch_returns)

                # policy loss
                log_ratios = new_action_logprobas - batch_action_logprobs # (B, 1)
                r = log_ratios.exp()
                unclipped_obj = r * batch_advantages
                clipped_obj = r.clip(1-xonfig.clip_range, 1+xonfig.clip_range) * batch_advantages
                policy_loss = -torch.min(unclipped_obj, clipped_obj).mean()

                # KL divergence
                with torch.no_grad():
                    # https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/ppo/ppo.py#L262-L265
                    # http://joschu.net/blog/kl-approx.html
                    log_ratios = log_ratios.detach()
                    approx_kl_div = ((log_ratios.exp() - 1) - log_ratios).mean().cpu().item()
                    kldivs_list.append(approx_kl_div)

                norm:tp.Optional[Tensor] = None
                if xonfig.target_kl is not None and approx_kl_div > xonfig.target_kl:
                    break

                entropy_loss:Tensor = -dist.entropy().mean()
                (policy_loss + 1.0 * value_loss + 0.0 * entropy_loss).backward()

                if xonfig.clip_norm > 0:
                    norm = nn.utils.clip_grad_norm_(
                        actor_critic.parameters(), xonfig.clip_norm
                    )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                
                losses["policy"].append(policy_loss.cpu().item())
                losses["value"].append(value_loss.cpu().item())
            
    if xonfig.logg_tb:
        writer.add_scalar(
            "losses/entropy_loss", entropy_loss.cpu().item(), total_steps_done
        )
        if norm is not None:
            writer.add_scalar("losses/grad_norm", norm.cpu().item(), total_steps_done)
    
    buffer.clear()
    actor_critic_old.load_state_dict(actor_critic.state_dict())
    avg = lambda x: sum(x)/max(len(x), 1)
    return avg(losses["policy"]), avg(losses["value"]), avg(kldivs_list)


def train():
    sum_rewards_list = []; num_steps = int(1); avg_kl_div_list = []; episode_length_list = []
    try:
        iterator = range(xonfig.num_episodes) if xonfig.num_episodes > 0 else count(1)
        for episode_num in iterator:
            state, info = env.reset()
            state = torch.as_tensor(state, device=xonfig.device, dtype=torch.float32).unsqueeze(0) # (B=1, state_dim)
            sum_rewards = float(0); t0 = time.time()
            for tstep in count(1):
                # sample actions
                with torch.no_grad():
                    action, state_value, action_logproba = actor_critic_old.sample_action(state) # (B=1, state_dim) => (B=1, action_dim)
                    action = action.squeeze(0) # (B=1, action_dim) => (action_dim,)
                    state_value = state_value.squeeze(0)

                # take action get reward and next state
                next_state, reward, done, truncated, info = env.step(
                    action.clip(
                        int(ACTION_BOUNDS[0].item()), int(ACTION_BOUNDS[1].item())
                    ).cpu().detach().numpy()
                )
                # print(f"{info=}") # info={'episode': {'r': np.float64(-1256.1483533844719), 'l': 200, 't': 0.175791}}
                next_state = torch.as_tensor(next_state, device=xonfig.device, dtype=torch.float32).unsqueeze(0)
                stats:dict = info.get("episode", {})
                unnormalized_rewards = stats.get("r", 0)
                sum_rewards += unnormalized_rewards

                # store transitions in buffer
                buffer.store(
                    state=state.squeeze(0),
                    action=action,
                    action_logprob=action_logproba,
                    state_val=state_value,
                    reward=reward,
                    done=done
                )

                # ppo update
                if num_steps % xonfig.update_every == 0:
                    avg_policy_loss, avg_value_loss, avg_kl_div = update(num_steps)
                    if xonfig.logg_tb:
                        writer.add_scalar("losses/policy Loss", avg_policy_loss, num_steps)
                        writer.add_scalar("losses/value Loss", avg_value_loss, num_steps)
                        writer.add_scalar("kl divergence", avg_kl_div, num_steps)

                    avg_kl_div_list.append(avg_kl_div)
                    if xonfig.logg_losses:
                        print(f"\t|| Policy loss Avg: {avg_policy_loss:.3f} |"
                              f"| Value loss Avg: {avg_value_loss:.3f} || KL Div Avg: {avg_kl_div:.4f} ||")
                        
                if num_steps % xonfig.save_model_every == 0 or num_steps == xonfig.max_steps:
                    torch.save(
                        obj=actor_critic.state_dict(),
                        f=f"ckpt/PPO_continuous_{ENV_NAME}_{num_steps}.ptc"
                    )
                
                # decay action std
                if xonfig.action_std_decay_rate > 0 and num_steps % xonfig.decay_std_every == 0:
                    actor_critic.decay_std(
                        action_std_decay_rate=xonfig.action_std_decay_rate,
                        min_action_std=xonfig.min_action_std
                    )
                    if xonfig.logg_tb:
                        writer.add_scalar("action std", actor_critic.action_std, num_steps)

                # terminate episode
                if done or truncated:
                    if num_steps >= xonfig.max_steps:
                        print(f"Training Completed {num_steps}...")
                        if xonfig.logg_tb:
                            writer.close()
                        torch.save(
                            obj=actor_critic.state_dict(),
                            f=f"ckpt/PPO_continuous_{ENV_NAME}_final.ptc"
                        )
                        return sum_rewards_list, episode_length_list, avg_kl_div_list
                    break

                # update state
                state = next_state  # (B=1, state_dim)
                num_steps += 1
            
            # log episode results
            sum_rewards_list.append(sum_rewards)
            episode_length_list.append(tstep)
            if xonfig.logg_tb:
                writer.add_scalar("Episode Length", tstep, num_steps)
                writer.add_scalar("Episode Reward", sum_rewards, num_steps)

            dt = time.time() - t0
            print(f"|| Episode: {episode_num} || Σ Rewards: {sum_rewards:<6.2f} || TimeSteps: {tstep} |"
                  f"| dt: {dt:<5.2f}s || INFO: {info} || Total Steps: {num_steps}  || Action Std: {actor_critic.policy_net.dist_log_std.exp().mean().item():.4f} ||")
    except KeyboardInterrupt:
        print("Training interrupted by user.")
    
    if xonfig.logg_tb:
        writer.close()
    return sum_rewards_list, episode_length_list, avg_kl_div_list


if __name__ == "__main__":
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED+1)
    torch.manual_seed(SEED+2)
    torch.use_deterministic_algorithms(mode=True, warn_only=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    ENV_NAME = "Pendulum-v1"

    raw_env = gym.make(ENV_NAME, render_mode="rgb_array")

    NUM_ACTIONS = raw_env.action_space.shape[0]
    NUM_STATES = raw_env.observation_space.shape[0]

    print(f"Number of actions: {NUM_ACTIONS}")
    print(f"Number of states: {NUM_STATES}")

    ACTION_BOUNDS = [raw_env.action_space.low, raw_env.action_space.high]
    print(f"Action bounds: {ACTION_BOUNDS}")
    ACTION_BOUND = max(ACTION_BOUNDS[0].min(), ACTION_BOUNDS[1].max())
    print(f"Action bound: {ACTION_BOUND}")

    env = make_env(raw_env, SEED, False, "test")

    print(xonfig())

    actor_critic = ActorCritic(NUM_STATES, xonfig.hidden_dim, NUM_ACTIONS, init_action_std=xonfig.init_action_std)
    actor_critic.to(xonfig.device)
    actor_critic.compile()
    print("Number of parameters in critic: ", sum(p.numel() for p in actor_critic.value_net.parameters() if p.requires_grad))
    print("Number of parameters in actor: ", sum(p.numel() for p in actor_critic.policy_net.parameters() if p.requires_grad))

    actor_critic_old = deepcopy(actor_critic)
    actor_critic_old.eval()
    actor_critic_old.requires_grad_(False)

    buffer = Buffer()
    optimizer = torch.optim.AdamW([
        {"params": actor_critic.policy_net.parameters(), "lr": xonfig.actor_lr},
        {"params": actor_critic.value_net.parameters(), "lr": xonfig.critic_lr},
    ], weight_decay=xonfig.weight_decay, betas=(xonfig.beta1, xonfig.beta2)
    )

    autocast = torch.autocast(
        device_type=xonfig.device.type,
        dtype=xonfig.dtype,
        enabled=(xonfig.device.type == "cuda" and xonfig.dtype == torch.bfloat16),
    )

    if xonfig.logg_tb:
        writer = SummaryWriter(
            log_dir=f"runs/PPO_{ENV_NAME}",
            filename_suffix=f"_{int(time.time())}"
        )

    sum_rewards_list, episode_length_list, avg_kl_div_list = train()

    with torch.no_grad():
        show_one_episode(
            env,
            lambda x: \
                actor_critic_old.sample_action(torch.tensor(x, device=xonfig.device).float(), deterministic=False)[0].cpu().numpy(),
            save_path="PPO_pendulumv1_learnable_logstd_normalized_rew_obs1.gif"
        )

    # Plot raw rewards (transparent)
    plt.plot(sum_rewards_list, label="Raw Sum Rewards", color="red", alpha=0.3)

    def moving_average(x, w):
        return np.convolve(x, np.ones(w)/w, mode='valid')

    # Plot smoothed rewards (bold)
    window = 20
    plt.plot(range(window - 1, len(sum_rewards_list)), moving_average(sum_rewards_list, window), label=f"Smoothed (window={window})", color="red", linewidth=2)
    plt.ylabel("Sum Rewards")
    plt.xlabel("Episode")
    plt.title("PPO on Pendulum-v1 — Episode Rewards")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("PPO_pendulumv1_learnable_logstd_normalized_rew_obs.png")
    plt.show()

