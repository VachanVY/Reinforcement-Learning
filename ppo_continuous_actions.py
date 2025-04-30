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

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch
from torch import (
    nn,
    Tensor
)
from torch.utils.tensorboard import SummaryWriter

SEED = 42
random.seed(SEED)
np.random.seed(SEED+1)
torch.manual_seed(SEED+2)
torch.use_deterministic_algorithms(mode=True, warn_only=True)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# The goal is to make hops that move in the forward (right) direction by applying torque to the three hinges that connect the four body parts.
# The default duration of an episode is 1000 timesteps.
ENV_NAME = "MountainCarContinuous-v0"

env = gym.make(ENV_NAME)

NUM_ACTIONS = env.action_space.shape[0]
NUM_STATES = env.observation_space.shape[0]

print(f"Number of actions: {NUM_ACTIONS}")
print(f"Number of states: {NUM_STATES}")

ACTION_BOUNDS = [env.action_space.low, env.action_space.high]
print(f"Action bounds: {ACTION_BOUNDS}")
ACTION_BOUND = max(ACTION_BOUNDS[0].max(), ACTION_BOUNDS[1].max())
print(f"Action bound: {ACTION_BOUND}")


@dc.dataclass
class xonfig:
    gamma:float = 0.99
    max_steps:int = int(3e6)
    save_model_every:int = int(2e5)

    update_every:int = 100 * 1
    num_ppo_iter:int = 2
    clip_range:float = 0.2
    target_kl:float = 0.01; target_kl *= 1.5 # why 1.5? so as to not be too strict nor too lenient

    device:torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype:torch.dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32
    batch_size:int = 32
    weight_decay:float = 1e-4
    actor_lr:float = 3e-4
    critic_lr:float = 1e-3 #7e-4
    clip_norm:float = 10.0

    hidden_dim:int = 128

    logg_losses:bool = True
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
        self.linear3 = nn.Linear(hidden_dim, hidden_dim); self.relu3 = nn.ReLU()
        self.linear4 = nn.Linear(hidden_dim, 1)

    def forward(self, x:Tensor): # state: (B, state_dim)
        x = self.linear1(x); x = self.relu1(x)
        x = self.linear2(x); x = self.relu2(x)
        x = self.linear3(x); x = self.relu3(x)
        x = self.linear4(x); 
        return x
    

class PolicyNet(nn.Module):
    def __init__(self, state_dim:int, hidden_dim:int, num_actions:int):
        super().__init__()
        self.linear1 = nn.Linear(state_dim, hidden_dim); self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, hidden_dim); self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(hidden_dim, hidden_dim); self.relu3 = nn.ReLU()


        self.dist_log_std = nn.Linear(hidden_dim, num_actions)
        self.dist_mean = nn.Linear(hidden_dim, num_actions)

    def forward(self, x:Tensor): # state: (B, state_dim)
        x = self.linear1(x); x = self.relu1(x)
        x = self.linear2(x); x = self.relu2(x)
        x = self.linear3(x); x = self.relu3(x)

        mean = self.dist_mean(x)
        std = torch.clip(self.dist_log_std(x), -20, 2).exp() # (B, num_actions)
        return mean, std # (B, num_actions), (B, num_actions)


class ActorCritic(nn.Module):
    def __init__(self, state_dim:int, hidden_dim:int, num_actions:int):
        super().__init__()
        self.policy_net = PolicyNet(state_dim, hidden_dim, num_actions)
        self.value_net = ValueNet(state_dim, hidden_dim)

    def forward(self, state:Tensor): # state: (B, state_dim)
        mean, std = self.policy_net(state) # (B, num_actions), (B, num_actions)
        state_value = self.value_net(state) # (B, 1)
        return (mean, std), state_value # (B, num_actions), (B, num_actions), (B, 1)
    
    def sample_action(self, state:Tensor, max_action_bound:int, deterministic:bool=False) -> tuple[Tensor, Tensor, Tensor]:
        assert not deterministic, "for now..."
        mean:Tensor; std:Tensor; state_val:Tensor
        (mean, std), state_val = self(state) # (B=1, num_actions), (B=1, num_actions), (B=1, 1)
        mean = mean.squeeze(0) # (num_actions,)
        std = std.squeeze(0) # (num_actions,)
        state_val = state_val.squeeze(0) # (1,)

        if not deterministic:
            dist = torch.distributions.Normal(
                loc=mean, scale=std # (num_actions,) # (num_actions,)
            )
            unbound_action = dist.rsample() # (num_actions,)
        else:
            # TODO: unbound_action = mean; ...
            ...

        unbound_action_log_prob:Tensor = dist.log_prob(unbound_action) # (num_actions,)

        action = torch.tanh(unbound_action) * max_action_bound # [-1, 1] * max_action_bound => [-max_action_bound, max_action_bound]
        action_log_prob = (unbound_action_log_prob - torch.log(1 - action.pow(2) + 1e-6)).sum(-1) # (,) # log(P(a1, a2, ...)) = log(P(a1)) + log(P(a2)) + ... 

        return action, action_log_prob, state_val # (num_actions,), (,), (1,)
    

actor_critic = ActorCritic(NUM_STATES, xonfig.hidden_dim, NUM_ACTIONS)
actor_critic.to(xonfig.device)
print("Number of parameters in critic: ", sum(p.numel() for p in actor_critic.value_net.parameters() if p.requires_grad))
print("Number of parameters in actor: ", sum(p.numel() for p in actor_critic.policy_net.parameters() if p.requires_grad))
# actor_critic.compile()

buffer = Buffer()
optimizer = torch.optim.AdamW([
        {"params": actor_critic.policy_net.parameters(), "lr": xonfig.actor_lr},
        {"params": actor_critic.value_net.parameters(), "lr": xonfig.critic_lr}
    ], weight_decay=xonfig.weight_decay
)
if xonfig.logg_tb:
    writer = SummaryWriter(
        log_dir=f"runs/PPO_{ENV_NAME}/{dc.asdict(xonfig())}"
    )


def ppo_update(buffer:Buffer, normalize_advantages:bool=True, normalize_returns:bool=True):
    buf_states:Tensor = torch.stack(buffer.states).to(device=xonfig.device).detach() # (n, state_dim)
    buf_actions:Tensor = torch.stack(buffer.actions).to(device=xonfig.device).detach() # (n, num_actions)
    buf_returns:Tensor = torch.as_tensor(get_discounted_returns(buffer.rewards, buffer.dones, xonfig.gamma), device=xonfig.device).detach() # (n,)
    buf_returns = (buf_returns - buf_returns.mean()) / (buf_returns.std() + 1e-6) # (n,)
    
    buf_state_vals:Tensor = torch.stack(buffer.state_values).to(device=xonfig.device).detach() # (n, 1)
    buf_action_logprobas:Tensor = torch.stack(buffer.log_probs).to(device=xonfig.device).detach() # (n, 1)

    advantages = (buf_returns - buf_state_vals.squeeze(1)).detach() # (n,)

    losses = {"policy": [], "value": []}
    kldivs_list = []
    for ppo_iter in range(xonfig.num_ppo_iter):
        rand_idx = torch.randperm(min(len(buffer.states), xonfig.batch_size))

        batch_states = buf_states[rand_idx] # (B, state_dim)
        batch_actions = buf_actions[rand_idx] # (B, num_actions)
        batch_returns = buf_returns[rand_idx].unsqueeze(-1) # (B, 1)
        batch_old_action_logprobas = buf_action_logprobas[rand_idx] # (B,)
        batch_advantages = advantages[rand_idx] # (B,)

        # compute new action logprobas
        mean:Tensor; std:Tensor; state_vals:Tensor
        (mean, std), state_vals = actor_critic(batch_states) # (B, num_actions), (B, num_actions), (B, 1) <= (B, state_dim)
        batch_unbound_actions = torch.atanh(batch_actions/ACTION_BOUND) # batch_actions = torch.tanh(batch_unbound_actions)*action_bound
        unbound_action_logprobas:Tensor = torch.distributions.Normal( # (B, num_actions)
            loc=mean, scale=std
        ).log_prob(batch_unbound_actions)
        new_action_logprobas = (unbound_action_logprobas - torch.log(1 - batch_actions.pow(2) + 1e-6)).sum(-1) # (B,) <= (B, num_actions,)

        # value loss
        value_loss = nn.functional.mse_loss(state_vals, batch_returns)

        # policy loss
        log_ratios = new_action_logprobas - batch_old_action_logprobas
        r = log_ratios.exp() # (B,)
        unclipped_obj = r * batch_advantages # (B,)
        clipped_obj = r.clip(1 - xonfig.clip_range, 1 + xonfig.clip_range) * batch_advantages # (B,)
        policy_loss = -1 * torch.min(unclipped_obj, clipped_obj).mean()

        with torch.no_grad():
            # https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/ppo/ppo.py#L262-L265
            # http://joschu.net/blog/kl-approx.html
            log_ratios = log_ratios.detach()
            # print(f"{log_ratios.mean()=}, {log_ratios.std()=}")
            approx_kl_div = ((log_ratios.exp() - 1) - log_ratios).mean().cpu().item()
            kldivs_list.append(approx_kl_div)

        norm = None
        if approx_kl_div >= xonfig.target_kl:
            break

        (policy_loss + value_loss).backward()
        
        try:
            if xonfig.clip_norm > 0:
                norm = torch.nn.utils.clip_grad_norm_(actor_critic.parameters(), xonfig.clip_norm, error_if_nonfinite=True)
            optimizer.step()
        except RuntimeError as e:
            print(f"\tClipNorm Error:", norm, sep="\t")
        finally:
            optimizer.zero_grad()

        # Store losses
        losses["policy"].append(policy_loss.item())
        losses["value"].append(value_loss.item())

    buffer.clear()
    avg = lambda x: sum(x)/max(len(x), 1)
    avg_policy_loss, avg_val_loss, avg_kl_div = avg(losses["policy"]), avg(losses["value"]), avg(kldivs_list)
    return avg_policy_loss, avg_val_loss, avg_kl_div, norm


def train():
    sum_rewards_list = []; num_steps = int(1); avg_kl_div_list = []; episode_length_list = []
    try:
        for episode_num in count(1):
            state, info = env.reset()
            state = torch.as_tensor(state, device=xonfig.device, dtype=torch.float32).unsqueeze(0) # (B=1, state_dim)
            sum_rewards = float(0); t0 = time.time()
            for tstep in count(1):
                # sample actions
                with torch.no_grad():
                    action, action_logproba, state_value = actor_critic.sample_action(state, ACTION_BOUND) # (B=1, state_dim) => (B=1, action_dim)

                # take action get reward and next state
                next_state, reward, done, truncated, info = env.step(action.cpu().detach().numpy())
                sum_rewards += reward

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
                    avg_policy_loss, avg_val_loss, avg_kl_div, norm = ppo_update(
                        buffer=buffer,
                        normalize_advantages=True, # check results with and without advantage normalization
                        normalize_returns=True
                    )
                    if xonfig.logg_tb:
                        writer.add_scalar("Avg Policy Loss per Update", avg_policy_loss, num_steps)
                        writer.add_scalar("Avg Value Loss per Update", avg_val_loss, num_steps)
                        writer.add_scalar("Avg KL Divergence per Update", avg_kl_div, num_steps)
                        if norm is not None:
                            writer.add_scalar("Gradient Norm", norm, num_steps)

                    avg_kl_div_list.append(avg_kl_div)
                    if xonfig.logg_losses:
                        print(f"\t|| Policy loss Avg: {avg_policy_loss:.3f} |"
                              f"| Value loss Avg: {avg_val_loss:.3f} || KL Div Avg: {avg_kl_div:.4f} ||")
                        
                if num_steps % xonfig.save_model_every == 0:
                    torch.save(
                        obj=actor_critic.state_dict(),
                        f=f"ckpt/ppo_{ENV_NAME}_{num_steps}.ptc"
                    )

                # terminate episode
                if done or truncated:
                    if num_steps >= xonfig.max_steps:
                        print(f"Training Completed {num_steps}...")
                        if xonfig.logg_tb:
                            writer.close()
                        return sum_rewards_list, episode_length_list, avg_kl_div_list
                    break

                # update state
                state = torch.as_tensor(next_state, device=xonfig.device, dtype=torch.float32).unsqueeze(0) # (B=1, state_dim)

                num_steps += 1
                
            # log episode results
            sum_rewards_list.append(sum_rewards)
            episode_length_list.append(tstep)
            if xonfig.logg_tb:
                writer.add_scalar("Episode Length", tstep, num_steps)
                writer.add_scalar("Episode Reward", sum_rewards, num_steps)

            dt = time.time() - t0
            print(f"Episode: {episode_num} || Total Steps: {num_steps}  || Σ Rewards: {sum_rewards:<6.2f} || Tsteps: {tstep} || dt: {dt:<5.2f}s || INFO: {info} ||")
    except KeyboardInterrupt:
        print("Training interrupted by user.")
    
    if xonfig.logg_tb: writer.close()
    return sum_rewards_list, episode_length_list, avg_kl_div_list


sum_rewards_list, episode_length_list, avg_kl_div_list = train()