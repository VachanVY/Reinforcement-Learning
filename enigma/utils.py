from collections import deque
import typing as tp


def get_discounted_returns(rewards:tp.Sequence, is_terminals:tp.Sequence, gamma:float):
    discounted_returns = deque()
    discounted_reward = 0
    for reward, is_terminal in zip(reversed(rewards), reversed(is_terminals)):
        if is_terminal: discounted_reward = 0
        discounted_reward = reward + gamma * discounted_reward
        discounted_returns.appendleft(float(discounted_reward))
    return list(discounted_returns)


class Buffer:
    def __init__(self): 
        self.clear()

    def clear(self):
        self.states = []
        self.actions = []
        self.action_logprobs = []
        self.state_vals = []

        self.rewards = []
        self.terminals = []

    def store(self, state, action, action_logprob, state_val, reward, terminal):
        self.states.append(state)                   # state: (state_dim,) -> states: (num_timesteps, state_dim) 
        self.actions.append(action)                 # action: (2,) -> actions: (num_timesteps, 2)
        self.action_logprobs.append(action_logprob) # action_logprob: (1,) -> action_logprobs: (num_timesteps, 1)
        self.state_vals.append(state_val)           # state_val: (1,) -> state_vals: (num_timesteps, 1)
        self.rewards.append(reward)                 # reward: (1,) -> rewards: (num_timesteps, 1)
        self.terminals.append(terminal)             # terminal: (1,) -> terminals: (num_timesteps, 1)


def update_scene(num, frames, patch):
    patch.set_data(frames[num])
    return patch,

def plot_animation(frames:list, save_path:tp.Optional[str]=None, repeat=False, interval=40):
    import matplotlib.pyplot as plt
    from matplotlib import animation as anim

    fig = plt.figure()
    patch = plt.imshow(frames[0])
    plt.axis('off')
    animation = anim.FuncAnimation(
        fig, update_scene, fargs=(frames, patch),
        frames=len(frames), repeat=repeat, interval=interval)
    if save_path is not None:
        animation.save(save_path, writer="pillow", fps=20)
    return animation

def show_one_episode(env_name:str, action_sampler:tp.Callable, save_path:tp.Optional[str]=None, repeat=False):
    import gymnasium as gym
    from torch import no_grad
    from itertools import count

    frames = []
    env = gym.make(env_name, render_mode="rgb_array")
    obs, info = env.reset()
    sum_rewards = int(0)
    with no_grad():
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
    return plot_animation(frames, repeat=repeat, save_path=save_path)
