import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import typing as tp
import gymnasium as gym
from itertools import count
from queue import PriorityQueue
from typing import Optional, Callable

from custom_envs import ShortcutMazeEnv


env = ShortcutMazeEnv(render_mode=None, layout_change_step=3000, max_episode_steps=500)
NUM_ACTIONS = env.action_space.n
NUM_STATES = env.observation_space.n
NUM_EPISODES = 1000
EPSILON = 0.1
ALPHA = 0.1
GAMMA = 0.99
OptionalInt: tp.TypeAlias = tp.Optional[int]


def update_scene(num, frames, patch):
    patch.set_data(frames[num])
    return patch,

def plot_animation(frames:list, save_path:str, title:Optional[str]=None, repeat=False, interval=500):
    fig = plt.figure()
    patch = plt.imshow(frames[0])
    plt.axis('off')
    if title is None:
        title = save_path
    plt.title(title, fontsize=16)
    animation = anim.FuncAnimation(
        fig, update_scene, fargs=(frames, patch),
        frames=len(frames), repeat=repeat, interval=interval)
    animation.save(save_path, writer="pillow", fps=20)
    return animation

def show_one_episode(env:gym.Env, action_sampler:Callable, save_path:str, title:Optional[str], repeat=False):
    frames = []
    state, info = env.reset()
    sum_rewards = 0
    for step in count(0):
        frames.append(env.render())
        try: action = action_sampler(state) 
        except: action = action_sampler() # for env.action_space.sample
        state, reward, done, truncated, info = env.step(action)
        sum_rewards += reward
        if done or truncated:
            print(f"|| done at step: {step+1} ||")
            print(f"|| sum_rewards: {sum_rewards} ||")
            break
    frames.append(env.render())
    return plot_animation([f for f in frames], save_path, title=title, repeat=repeat)


def init_q_vals(num_states:int, num_actions:int, init_rand:bool=True):
    q_vals: dict[int, list[float]] = dict()
    for state in range(num_states):
        q_vals[state] = [(random.random() if init_rand else 0.0) for _ in range(num_actions)]
    return q_vals


def init_env_model(num_states:int, num_actions:int):
    #              state  # index of list = action            reward, next_state
    env_model: dict[int, list[tuple[OptionalInt, OptionalInt]]] = dict()
    for state in range(num_states):
        env_model[state] = [(None, None) for _ in range(num_actions)] # (reward, next_state)
    return env_model


def sample_action(state_qvalues:list[int], epsilon:float):
    if random.random() < epsilon:
        return random.randint(0, NUM_ACTIONS - 1)
    return state_qvalues.index(max(state_qvalues)) # get action which has the max q value


def see_shortcut_maze(Q_vals:dict[int, list[float]], title:str, unblock:bool=False):
    # see the learned policy
    env = ShortcutMazeEnv(render_mode="rgb_array", layout_change_step=3000, max_episode_steps=500, unblock=unblock)
    show_one_episode(env, lambda state: sample_action(Q_vals[state], epsilon=0), f"images/shortcut_maze_{title}.gif", title=title)
    env.close()
    del env


def prioritized_sweeping(log:bool=True):
    q_values = init_q_vals(NUM_STATES, NUM_ACTIONS)
    env_model = init_env_model(NUM_STATES, NUM_ACTIONS)

    pque = PriorityQueue()

    sum_rewards_episodes = []; timestep_episodes = []
    total_step = 0
    for episode in range(1, NUM_EPISODES+1):
        state, info = env.reset(); sum_rewards = float(0)
        for tstep in count(1):
            total_step += 1
            action = sample_action(q_values[state], EPSILON)
            next_state, reward, done, truncated, info = env.step(action); sum_rewards += reward
            diff = reward + GAMMA * max(q_values[next_state]) - q_values[state][action]
            if abs(diff) > 1e-5:
                pque.put((-abs(diff), (state, action))) # lower values have higher priority # so here higher abs(diff) have higher priority
            
            env_model[state][action] = (reward, next_state) # (reward, next_state)
            if done or truncated:   
                break
            state = next_state
        sum_rewards_episodes.append(sum_rewards)
        timestep_episodes.append(tstep)
        if log:
            print(f"|| Epsisode: {episode} || Sum of Reward: {sum_rewards} || Total Timesteps: {tstep} ||")

        num_iter = 0
        while not pque.empty():
            num_iter += 1
            p, (state, action) = pque.get()
            reward, next_state = env_model[state][action]
            q_values[state][action] += ALPHA * (reward + GAMMA * max(q_values[next_state]) - q_values[state][action])

            # loop for all previous states and actions that lead to the current state
            for prev_state in range(NUM_STATES):
                for prev_action in range(NUM_ACTIONS):
                    try: prev_reward, prev_next_state = env_model[prev_state][prev_action]
                    except:
                        print("state-action fault. 😂 =>", prev_state, prev_action)
                        continue
                    # Add state-action pairs that lead to the current state to the priority queue
                    if prev_next_state == state:
                        diff = (prev_reward + GAMMA * max(q_values[state])) - q_values[prev_state][prev_action]
                        if abs(diff) > 1e-5: 
                            pque.put((-abs(diff), (prev_state, prev_action)))
            if num_iter > 1000:
                break

    return q_values, sum_rewards_episodes, timestep_episodes


if __name__ == "__main__":
    q_values, sum_rewards_episodes, timestep_episodes = prioritized_sweeping(log=True)
    see_shortcut_maze(q_values, "prioritized_sweeping_maze_env"); plt.close()

    plt.plot(sum_rewards_episodes[:], label="Sum of Rewards per Episode", color="red")
    plt.xlabel("Episodes")
    plt.ylabel("Smoothed Sum of Rewards")
    plt.grid(True)
    plt.title("|| Sum of Rewards per Episode ||")
    plt.legend()
    plt.savefig("images/prioritized_sweeping_maze_env_sum_rewards.png")
    plt.show()
    plt.close()

    plt.plot(timestep_episodes[:], label="Time Steps per Episode", color="blue")
    plt.xlabel("Episodes")
    plt.ylabel("Time Steps")
    plt.grid(True)
    plt.title("|| Time Steps per Episode ||")
    plt.legend()
    plt.savefig("images/prioritized_sweeping_maze_env_num_training_curves.png")
    plt.show()
    plt.close()

    print("SO MUCH BETTER THAN JUST DYNA-Q!!!")