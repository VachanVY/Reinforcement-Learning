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
from matplotlib import animation as anim
import math
import argparse

# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch
from torch import nn, Tensor
from torch.utils.tensorboard import SummaryWriter

from utils.lr_schedulers import CosineDecayWithWarmup


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


def strtobool(s:str) -> bool:
    if s.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif s.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError(f"Boolean value expected, got {s}.")


@dc.dataclass
class Xonfig:
    max_steps:int = 525_000*4

    # General RL hyperparameters
    num_episodes:int = 0
    gamma:float = 0.99
    update_every:int = 2048*4

    # PPO hyperparameters
    K:int = 10 # num ppo iterations
    clip_range:float = 0.2
    target_kl:tp.Optional[float] = None
    val_coeff:float = 0.5
    entropy_coeff:float = 0.000
    trace_decay:float = 0.95
    clip_val_loss:bool = False

    save_model_every:int = int(num_episodes // 4) if num_episodes > 0 else int(max_steps // 4)

    # Action noise hyperparameters
    init_action_std:float = 0.5
    action_std_decay_rate:float = 0.0
    min_action_std:float = 0.1
    decay_std_every:int = int(max_steps / 12) if max_steps > 0 else int(num_episodes / 24)
    learnable_std:bool = True
    if learnable_std: assert action_std_decay_rate == 0
    if learnable_std: init_action_std += 0.1

    # General Training hyperparameters
    device:torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype:torch.dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32
    batch_size:int = 64
    weight_decay:float = 0e-4
    actor_lr:float = 3e-4
    critic_lr:float = 3e-4
    clip_norm:float = 0.5
    hidden_dim:int = 256
    beta1:float = 0.9
    beta2:float = 0.999

    lr_decay:bool = False
    warmup_steps:int = 5000

    logg_losses:bool = False
    logg_tb:bool = True # tensorboard logging

    # Add fields for things previously defined outside or conditionally
    env_name: str = "Ant-v5"
    seed: int = 0
    capture_video: bool = False
    run_name: str = "_GAE"
    assert run_name[0] == "_"

    def parse_args(self):
        parser = argparse.ArgumentParser(description="PPO Continuous Action Space")
        parser.add_argument("--seed", type=int, default=self.seed, help="Random seed")
        parser.add_argument("--capture_video", type=strtobool, default=self.capture_video, help="Capture video of the agent")

        parser.add_argument("--clip_norm", type=float, default=self.clip_norm, help="Gradient clipping norm")
        parser.add_argument("--val_coeff", type=float, default=self.val_coeff, help="Value function coefficient")
        parser.add_argument("--clip_val_loss", type=strtobool, default=self.clip_val_loss, help="Clip value loss")
        parser.add_argument("--entropy_coeff", type=float, default=self.entropy_coeff, help="Entropy coefficient")
        parser.add_argument("--actor_lr", type=float, default=self.actor_lr, help="Actor learning rate")
        parser.add_argument("--critic_lr", type=float, default=self.critic_lr, help="Critic learning rate")
        parser.add_argument("--K", type=int, default=self.K, help="Number of PPO iterations")
        parser.add_argument("--action_std_decay_rate", type=float, default=self.action_std_decay_rate, help="Action std decay rate")
        parser.add_argument("--learnable_std", type=strtobool, default=self.learnable_std, help="Learnable action std")

        parser.add_argument("--num_episodes", type=int, default=self.num_episodes, help="Number of episodes to train")
        parser.add_argument("--max_steps", type=int, default=self.max_steps, help="Maximum number of steps to train")

        args = parser.parse_args()
        for key, value in vars(args).items():
            if hasattr(self, key):
                if self.__dict__[key] != value:
                    print(f"Overriding {key} from {self.__dict__[key]} to {value}")
                    self.run_name += f"_{key}_{value}"
                setattr(self, key, value)
            else:
                print(f"Warning: {key} is not a valid argument for Xonfig. Ignoring it.")
        # if self.num_episodes > 0:
        #     self.max_steps = int(100000000e10)


class RolloutBuffer:
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
        self.linear1 = nn.Linear(state_dim, hidden_dim); self.act1 = torch.nn.Tanh()
        self.linear2 = nn.Linear(hidden_dim, hidden_dim); self.act2 = torch.nn.Tanh()
        self.linear3 = nn.Linear(hidden_dim, 1)

    def forward(self, x:Tensor): # state: (B, state_dim)
        x = self.linear1(x); x = self.act1(x)
        x = self.linear2(x); x = self.act2(x)
        x = self.linear3(x); 
        return x


class PolicyNet(nn.Module):
    def __init__(self, state_dim:int, hidden_dim:int, num_actions:int, init_action_std:float, learnable_std:bool):
        super().__init__()
        self.linear1 = nn.Linear(state_dim, hidden_dim); self.act1 = torch.nn.Tanh()
        self.linear2 = nn.Linear(hidden_dim, hidden_dim); self.act2 = torch.nn.Tanh()

        self.dist_mean = nn.Linear(hidden_dim, num_actions)
        dist_log_std = nn.Parameter(
            torch.full((num_actions,), fill_value=math.log(init_action_std)), requires_grad=learnable_std
        )
        self.dist_log_std:Tensor
        if not dist_log_std.requires_grad:
            self.register_buffer("dist_log_std", dist_log_std)
        else:
            self.dist_log_std = dist_log_std

    def forward(self, x:Tensor): # state: (B, state_dim)
        x = self.linear1(x); x = self.act1(x)
        x = self.linear2(x); x = self.act2(x)

        mean = self.dist_mean(x)
        std = torch.clamp(self.dist_log_std, min=-20, max=2).exp() # (num_actions,)
        std = std.expand_as(mean) # (B, num_actions)
        return mean, std # (B, num_actions), (B, num_actions)
    
    def decay_std(self, action_std_decay_rate:float, min_action_std:float):
        # https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/PPO.py#L159-L173
        min_std = torch.tensor(min_action_std, device=self.dist_log_std.device)
        new_std = torch.clamp(self.dist_log_std, min=-20, max=2).exp() - action_std_decay_rate
        self.dist_log_std.copy_(new_std.clip(min=min_std).log())


class ActorCritic(nn.Module):
    def __init__(self, state_dim:int, hidden_dim:int, num_actions:int, init_action_std:float, learnable_std:bool=True):
        super().__init__()
        self.policy_net = PolicyNet(state_dim, hidden_dim, num_actions, init_action_std=init_action_std, learnable_std=learnable_std)
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


@torch.no_grad()
def calculate_advantages_gae(
    rewards_list:tp.Sequence[float],
    state_values_list:tp.Sequence[float],
    gamma:float,
    trace_decay:float
):
    GAEs = deque()
    A_GAE = 0
    next_state_value = 0.0

    for reward, state_value in zip(reversed(rewards_list), reversed(state_values_list)):
        td_error = reward + gamma * next_state_value - state_value
        A_GAE = td_error + gamma * trace_decay * A_GAE
        GAEs.appendleft(A_GAE)
        next_state_value = state_value
    return list(GAEs)


num_updates:int = 0
def update(total_steps_done:int):
    global num_updates
    buf_size = len(buffer.rewards)

    buf_states = torch.stack(buffer.states).to(xonfig.device).detach() # (B, state_dim)
    buf_actions = torch.stack(buffer.actions).to(xonfig.device).detach() # (B, num_actions)
   
    buf_action_logproba = torch.stack(buffer.log_probs).to(xonfig.device).detach() # (B, 1)

    advantages = torch.tensor(
        calculate_advantages_gae(
            rewards_list=buffer.rewards,
            state_values_list=buffer.state_values,
            gamma=xonfig.gamma,
            trace_decay=xonfig.trace_decay
        ), device=xonfig.device
    ).unsqueeze(-1).detach() # (B, 1)

    buf_state_values = torch.as_tensor(buffer.state_values, device=xonfig.device).unsqueeze(-1)
    buf_returns = (advantages + buf_state_values).detach() 

    if buf_size > 1:
        advantages = ((advantages - advantages.mean()) / (advantages.std() + 1e-6)).detach() # (B, 1)

    # K Epochs
    losses = {"policy": [], "value": []}
    kldivs_list = []; norm:tp.Optional[Tensor] = None
    for _ in range(xonfig.K):
        for start_idx in range(0, buf_size - xonfig.batch_size + 1, xonfig.batch_size):
            num_updates += 1
            batch_idx = slice(start_idx, min(start_idx + xonfig.batch_size, buf_size))
            batch_returns = buf_returns[batch_idx] # (B, 1)
            batch_advantages = advantages[batch_idx] # (B, 1)
            batch_states = buf_states[batch_idx] # (B, state_dim)
            batch_actions = buf_actions[batch_idx] # (B, action_dim)
            batch_action_logprobs = buf_action_logproba[batch_idx] # (B, 1)
            batch_state_values = buf_state_values[batch_idx] # (B, 1)

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
                if xonfig.clip_val_loss:
                    unclipped_value_loss = nn.functional.mse_loss(state_value, batch_returns, reduction="none") # (B, 1)
                    
                    clipped_state_value = batch_state_values + (state_value - batch_state_values).clip(-xonfig.clip_range, xonfig.clip_range) # (B, 1)
                    clipped_value_loss = nn.functional.mse_loss(clipped_state_value, batch_returns, reduction="none") # (B, 1)
                    
                    value_loss = torch.max(unclipped_value_loss, clipped_value_loss).mean()
                else:
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

                if xonfig.target_kl is not None and approx_kl_div > xonfig.target_kl * 1.5:
                    break

                entropy_loss:Tensor = -dist.entropy().mean()

            (policy_loss + xonfig.val_coeff * value_loss + xonfig.entropy_coeff * entropy_loss).backward()

            if xonfig.clip_norm > 0:
                norm = nn.utils.clip_grad_norm_(
                    actor_critic.parameters(), xonfig.clip_norm
                )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            
            losses["policy"].append(policy_loss.cpu().item())
            losses["value"].append(value_loss.cpu().item())
            
    if xonfig.logg_tb:
        try:
            writer.add_scalar("see/log_ratios", log_ratios.mean().item(), total_steps_done)
            writer.add_scalar("see/advantages", advantages.mean().item(), total_steps_done)
            writer.add_scalar("see/clipped_loss", -clipped_obj.mean().item(), total_steps_done)
            writer.add_scalar("see/unclipped_loss", -unclipped_obj.mean().item(), total_steps_done)
            writer.add_scalar("see/entropy", -entropy_loss.cpu().item(), total_steps_done)
            if norm is not None:
                writer.add_scalar("see/grad_norm", norm.cpu().item(), total_steps_done)
        except Exception as e:
            print(e)
    
    buffer.clear()
    actor_critic_old.load_state_dict(actor_critic.state_dict())
    avg = lambda x: sum(x)/max(len(x), 1)
    return avg(losses["policy"]), avg(losses["value"]), avg(kldivs_list)


def train():
    sum_rewards_list = []; num_steps = int(1); avg_kl_div_list = []; episode_length_list = []; best_mean_sum_rewards = float("-inf")
    try:
        iterator = range(xonfig.num_episodes) if xonfig.num_episodes > 0 else count(1)
        for episode_num in iterator:
            state, info = env.reset()
            state = torch.as_tensor(state, device=xonfig.device, dtype=torch.float32).unsqueeze(0) # (B=1, state_dim)
            sum_rewards = float(0); t0 = time.time()
            for tstep in count(1):
                lr = get_lr(num_steps)
                actor_param_group = optimizer.param_groups[0]
                actor_param_group["lr"] = lr
                
                # sample actions
                with torch.no_grad():
                    action, state_value, action_logproba = actor_critic_old.sample_action(state) # (B=1, state_dim) => (B=1, action_dim)
                    action = action.squeeze(0) # (B=1, action_dim) => (action_dim,)
                    state_value = state_value.squeeze(0)

                # take action get reward and next state
                next_state, reward, done, truncated, info = env.step(
                    action.clip(
                        -int(ACTION_BOUND), int(ACTION_BOUND)
                    ).cpu().detach().numpy()
                )
                # print(f"{info=}") # info={'episode': {'r': np.float64(-1256.1483533844719), 'l': 200, 't': 0.175791}}
                next_state = torch.as_tensor(next_state, device=xonfig.device, dtype=torch.float32).unsqueeze(0)
                if NORMALIZE_REW:
                    stats:dict = info.get("episode", {})
                    unnormalized_rewards = stats.get("r", 0)
                    sum_rewards += unnormalized_rewards
                else: sum_rewards += reward

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
                        writer.add_scalar("losses/kl divergence", avg_kl_div, num_steps)

                    avg_kl_div_list.append(avg_kl_div)
                    if xonfig.logg_losses:
                        print(f"\t|| Policy loss Avg: {avg_policy_loss:.3f} |"
                              f"| Value loss Avg: {avg_value_loss:.3f} || KL Div Avg: {avg_kl_div:.4f} ||")
                
                # decay action std
                if xonfig.action_std_decay_rate > 0 and num_steps % xonfig.decay_std_every == 0:
                    actor_critic.decay_std(
                        action_std_decay_rate=xonfig.action_std_decay_rate,
                        min_action_std=xonfig.min_action_std
                    )
                    if xonfig.logg_tb:
                        writer.add_scalar("action std", actor_critic.policy_net.dist_log_std.exp().mean(), num_steps)

                # terminate episode
                if done or truncated:
                    if num_steps >= xonfig.max_steps and xonfig.num_episodes <= 0:
                        print(f"Training Completed {num_steps}...")
                        if xonfig.logg_tb:
                            writer.close()
                        raise KeyboardInterrupt
                    break

                # update state
                state = next_state  # (B=1, state_dim)
                num_steps += 1
            
            # log episode results
            if episode_num % xonfig.save_model_every == 0:
                torch.save(
                    obj=actor_critic.state_dict(),
                    f=f"ckpt/PPO_Continuous_Actions_{ENV_NAME}_{episode_num}{xonfig.run_name}.ptc"
                )
            sum_rewards_list.append(sum_rewards)
            episode_length_list.append(tstep)
            if xonfig.logg_tb:
                writer.add_scalar("Episode Length", tstep, num_steps)
                writer.add_scalar("Episode Reward", sum_rewards, num_steps)
                writer.add_scalar("see/state_value", state_value.mean().item(), num_steps)

            dt = time.time() - t0
            print(f"|| Episode: {episode_num} || Σ Rewards: {sum_rewards:<6.2f} || TimeSteps: {tstep} |"
                  f"| dt: {dt:<5.2f}s || Total Steps: {num_steps}  || Action Std: {actor_critic.policy_net.dist_log_std.exp().mean().item():.4f} || lr: {lr:e} ||") 
    except KeyboardInterrupt:
        print("Training done.")
        if xonfig.logg_tb:
            writer.close()
        torch.save(
            obj=actor_critic.state_dict(),
            f=f"ckpt/PPO_Continuous_Actions_{ENV_NAME}_final{xonfig.run_name}.ptc"
        )
    return sum_rewards_list, episode_length_list, avg_kl_div_list


if __name__ == "__main__":
    SEED = None
    random.seed(SEED)
    np.random.seed(SEED)
    # torch.manual_seed(SEED)
    # torch.use_deterministic_algorithms(mode=True, warn_only=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    xonfig = Xonfig()
    print("Default Args:", xonfig, sep="\n")
    xonfig.parse_args()
    print("Parsed Args:", xonfig, sep="\n")

    ENV_NAME = xonfig.env_name
    if xonfig.capture_video:
        os.makedirs(f"videos/{ENV_NAME}{xonfig.run_name}", exist_ok=True)

    raw_env = gym.make(ENV_NAME, render_mode="rgb_array")

    NUM_ACTIONS = raw_env.action_space.shape[0]
    NUM_STATES = raw_env.observation_space.shape[0]

    print(f"Number of actions: {NUM_ACTIONS}")
    print(f"State Dim: {NUM_STATES}")

    ACTION_BOUNDS = [raw_env.action_space.low, raw_env.action_space.high]
    print(f"Action bounds: {ACTION_BOUNDS}")
    ACTION_BOUND = max(ACTION_BOUNDS[0].min(), ACTION_BOUNDS[1].max())
    print(f"Action bound: {ACTION_BOUND}")

    NORMALIZE_REW = True
    env = make_env(
        raw_env, SEED,
        normalize_obs=True,
        normalize_rewards=NORMALIZE_REW,
        capture_video=xonfig.capture_video
    )

    actor_critic = ActorCritic(NUM_STATES, xonfig.hidden_dim, NUM_ACTIONS, init_action_std=xonfig.init_action_std, learnable_std=xonfig.learnable_std)
    actor_critic.to(xonfig.device)
    actor_critic.compile()
    print("Number of parameters in critic: ", sum(p.numel() for p in actor_critic.value_net.parameters() if p.requires_grad))
    print("Number of parameters in actor: ", sum(p.numel() for p in actor_critic.policy_net.parameters() if p.requires_grad))

    actor_critic_old = deepcopy(actor_critic)
    actor_critic_old.eval()
    actor_critic_old.requires_grad_(False)

    buffer = RolloutBuffer()
    optimizer = torch.optim.AdamW([
        {"params": actor_critic.policy_net.parameters(), "lr": xonfig.actor_lr},
        {"params": actor_critic.value_net.parameters(), "lr": xonfig.critic_lr},
    ], weight_decay=xonfig.weight_decay, betas=(xonfig.beta1, xonfig.beta2)
    )
    get_lr = CosineDecayWithWarmup(
        warmup_steps=xonfig.warmup_steps, 
        max_learning_rate=xonfig.actor_lr,
        decay_steps=xonfig.max_steps,
        min_learning_rate=xonfig.actor_lr * 0.1
    ) if not xonfig.lr_decay else lambda _: xonfig.actor_lr

    autocast = torch.autocast(
        device_type=xonfig.device.type,
        dtype=xonfig.dtype,
        enabled=(xonfig.device.type == "cuda" and xonfig.dtype == torch.bfloat16),
    )

    if xonfig.logg_tb:
        writer = SummaryWriter(
            log_dir=f"runs_PPO_{ENV_NAME}",
            filename_suffix=f"{xonfig.run_name}"
        )
        writer.add_text(
            "Training Configuration",
            str(dc.asdict(xonfig))
        )

    sum_rewards_list, episode_length_list, avg_kl_div_list = train()
    print("\n\nTotal number of updates done:", num_updates)
    
    show_one_episode(
        env,
        lambda x: \
            actor_critic_old.sample_action(torch.tensor(x, device=xonfig.device).float(), deterministic=False)[0].cpu().numpy(),
        save_path=f"images/PPO_Continuous_Actions_{ENV_NAME}{xonfig.run_name}.gif"
    ); plt.close()

    show_one_episode(
        env,
        lambda x: \
            actor_critic_old.sample_action(torch.tensor(x, device=xonfig.device).float(), deterministic=True)[0].cpu().numpy(),
        save_path=f"images/PPO_Continuous_Actions_{ENV_NAME}{xonfig.run_name}_deterministic.gif"
    ); plt.close()

    def moving_average(x, w):
        return np.convolve(x, np.ones(w)/w, mode='valid')

    window = 20
    episodes = range(len(sum_rewards_list))
    smoothed_rewards = moving_average(sum_rewards_list, window)
    smoothed_lengths = moving_average(episode_length_list, window)

    fig, axes = plt.subplots(3, 1, figsize=(10, 15))

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

    axes[2].plot(avg_kl_div_list, label="KL Divergence", color="green")
    axes[2].set_xlabel("Num Update Steps")
    axes[2].set_ylabel("KL Divergence")
    axes[2].set_title("KL Divergence")
    axes[2].legend()
    axes[2].grid()

    plt.tight_layout()
    plt.show()
    plt.savefig(f"images/PPO_Cont_Actions_{ENV_NAME}{xonfig.run_name}_training_curves.png")
    plt.close(fig)