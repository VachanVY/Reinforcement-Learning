import dataclasses as dc
import gymnasium as gym
import numpy as np
from itertools import count
from copy import deepcopy
import random
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import typing as tp
import pickle

import torch
from torch import Tensor, nn
from torch.utils.tensorboard import SummaryWriter


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
        with torch.no_grad():
            action = action_sampler(obs)
        obs, reward, done, truncated, info = env.step(action)
        sum_rewards += info["episode"]["r"] if "episode" in info else reward
        if done or truncated:
            print("done at step", step+1)
            print("sum of rewards", sum_rewards)
            break

    return plot_animation(frames, repeat=repeat, save_path=save_path)


ENV_NAME = "HalfCheetah-v5"
# USING SB3 CODE INSTEAD OF MY CUSTOM CODE SINCE SB3 CODE IS COMPACT
def train_expert_policy():
    from stable_baselines3 import SAC
    from stable_baselines3.common.monitor import Monitor

    env = gym.make(ENV_NAME, render_mode="rgb_array")
    env = Monitor(env)

    model = SAC("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=200_000)

    save_path = "ckpt/expert_policy_imitation_learning"
    model.save(save_path)
    print(f"Model saved to {save_path}")
    return model, env

def sample_actions(policy:nn.Module, state: Tensor) -> Tensor: # (B, state_dims)
    action = policy(state, deterministic=True)
    return action


@dc.dataclass
class xonfig:
    device:torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    learning_rate:float = 1e-4
    weight_decay:float = 1e-4
    batch_size:int = 256

    # init_expert_rollouts = 25
    num_iterations = 10 # 10
    num_rollouts = 15
    num_epochs = 4
    dagger_init_beta:float = 0.9


def get_beta_niter(niter, num_iterations):
    return max(1 - niter / num_iterations, 0)


def get_rollout_dataset(policy:nn.Module, iter_num:int, logg=True):
    rollout_dataset = [] # (B, state_dims + action_dims)
    rollout_rewards = []
    rollout_lengths = []
    for rollout_idx in range(1, xonfig.num_rollouts + 1):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=xonfig.device).unsqueeze(0)
        episode_reward = 0; episode_length = 0
        
        for i in count(1):
            actions = sample_actions(policy, state)
            (
                next_state, # (state_dims,)
                reward,     # (,)
                terminated, # (,)
                truncated,  # (,)
                info
            ) = env.step(actions.squeeze(0).cpu().detach().numpy())
            next_state = torch.tensor(next_state, dtype=torch.float32, device=xonfig.device).unsqueeze(0)
            rollout_dataset.append(torch.cat((state, actions), dim=-1).cpu().detach())

            episode_reward += reward; episode_length += 1
            state = next_state
            
            if terminated or truncated:
                rollout_rewards.append(episode_reward)
                rollout_lengths.append(episode_length)
                break
        global_step = iter_num * xonfig.num_rollouts + rollout_idx
        print(f"\t| Rollout {rollout_idx}/{xonfig.num_rollouts} | Episode reward: {episode_reward:.2f} | Episode length: {episode_length} |")
        if logg:
            writer.add_scalar("rollout/Sum of rewards", episode_reward, global_step)
            writer.add_scalar("rollout/Episode length", episode_length, global_step)

    rollout_dataset = torch.cat(rollout_dataset, dim=0).cpu() # (nT, state_dims + action_dims)
    return rollout_dataset, np.mean(rollout_rewards), np.mean(rollout_lengths)


def train():
    dataset = torch.empty((0, STATE_DIMS + ACTION_DIMS), dtype=torch.float32).cpu() # (B, state_dims + action_dims)
    print_eq = lambda: print("=" * 50)
    INFERENCE_BATCH_SIZE = 8192

    for niter in range(1, xonfig.num_iterations + 1):
        # beta policy
        beta_policy = expert_policy if random.random() < (beta:=get_beta_niter(niter, xonfig.num_iterations)) else policy

        # collect rollouts from beta policy
        print_eq()
        print(f"Collecting rollouts from beta policy with beta: {beta} ... Expert: {beta_policy == expert_policy} or learner: {beta_policy == policy}")
        rollout_dataset, mean_rewards, mean_episode_length = get_rollout_dataset(beta_policy, iter_num=niter) # (nT, state_dims + action_dims)

        # add to dataset, state visited by beta policy and actions taken by expert policy
        print(f"Adding {rollout_dataset.shape[0]} rollouts to dataset ...")
        expert_dataset = []
        with torch.no_grad():
            for i in range(0, rollout_dataset.shape[0], INFERENCE_BATCH_SIZE):
                idx = slice(i, min(i + INFERENCE_BATCH_SIZE, rollout_dataset.shape[0]))
                states = rollout_dataset[idx, :STATE_DIMS].to(xonfig.device) # (B, state_dims) - only extract states
                expert_actions = sample_actions(expert_policy, states) # (B, action_dims)
                expert_dataset.append(torch.cat((states, expert_actions), dim=-1).cpu().detach())

        expert_dataset = torch.cat(expert_dataset, dim=0)
        dataset = torch.cat((dataset, expert_dataset), dim=0) # (nT, state_dims + action_dims)

        # train policy
        print(f"Training policy for {xonfig.num_epochs} epochs ...")
        print_eq()
        epoch_losses = []
        for epoch in range(xonfig.num_epochs):
            num_batches = (dataset.shape[0] + xonfig.batch_size - 1) // xonfig.batch_size
            batch_losses = []
            
            for batch_idx in range(num_batches):
                # Sample random indices with replacement to ensure exact batch size
                idx = torch.randint(0, dataset.shape[0], (xonfig.batch_size,))
                data_batch = dataset[idx] # (B, state_dims + action_dims)
                state, actions = data_batch[:, :STATE_DIMS].to(xonfig.device), data_batch[:, STATE_DIMS:].to(xonfig.device) # (B, state_dims), (B, action_dims)

                # train
                with autocast:
                    pred_action = sample_actions(policy, state) # (B, action_dims)
                    loss = nn.functional.mse_loss(pred_action, target=actions.detach())
                
                batch_losses.append(loss.detach().cpu().item())
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            epoch_loss = np.mean(batch_losses)
            epoch_losses.append(epoch_loss)
            print(f"\t| Epoch {epoch + 1}/{xonfig.num_epochs} | Loss: {epoch_loss:.4f} |")
            writer.add_scalar("loss/epoch", epoch_loss, (niter - 1) * xonfig.num_epochs + epoch)

        iteration_loss = np.mean(epoch_losses)
        print(f"|| Iteration: {niter} || Mean Sum of Rewards: {mean_rewards} || Mean Episode Length: {mean_episode_length} || Loss: {iteration_loss} || Beta: {beta} ||")
    
    torch.save({
        "policy_net": policy.state_dict(),
        "optimizer": optimizer.state_dict(),
        "env": env,
        "dataset": dataset
    }, f"ckpt/SAC_HalfCheetah-v5_final.pt")

if __name__ == "__main__":
    model, env_ = train_expert_policy()

    expert_policy = model.policy.actor
    policy:nn.Module = deepcopy(expert_policy).apply(lambda x: expert_policy.init_weights(x))
    env = gym.wrappers.RecordVideo(env_, f"imitation_learning_videos")

    STATE_DIMS = env.observation_space.shape[0]
    ACTION_DIMS = env.action_space.shape[0]
    ACTION_BOUNDS = [env.action_space.low, env.action_space.high]
    ACTION_BOUNDS = ACTION_BOUNDS[1][0]
    print(f"|| State dims: {STATE_DIMS} || Action dims: {ACTION_DIMS} || Action bounds: {ACTION_BOUNDS} ||")

    expert_policy.to(xonfig.device)
    policy.to(xonfig.device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=xonfig.learning_rate, weight_decay=xonfig.weight_decay)

    writer = SummaryWriter(
        log_dir="runs_imitation_learning"
    )
    print(expert_policy)

    autocast = torch.autocast(
        xonfig.device.type,
        dtype=torch.bfloat16 if xonfig.device.type == "cuda" else torch.float32,
        enabled=xonfig.device.type == "cuda" and torch.cuda.is_bf16_supported()
    )
    train()

    show_one_episode(
        env=env,
        action_sampler=lambda obs: sample_actions(policy, torch.tensor(obs, dtype=torch.float32, device=xonfig.device).unsqueeze(0)).squeeze(0).cpu().numpy(),
        save_path="images/learner_policy_imitation_learning_animation.gif"
    )

    show_one_episode(
        env=env,
        action_sampler=lambda obs: sample_actions(expert_policy, torch.tensor(obs, dtype=torch.float32, device=xonfig.device).unsqueeze(0)).squeeze(0).cpu().numpy(),
        save_path="images/expert_policy_imitation_learning_animation.gif"
    )
    pickle.dump(env, open(f"ckpt/CheetahEnv_{__file__}.pkl", 'wb'))
    env.close()