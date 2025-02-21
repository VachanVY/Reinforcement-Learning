import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import typing as tp
import gymnasium as gym
from gymnasium import spaces
from itertools import count
import math
from copy import deepcopy

from shortcutmaze import ShortcutMazeEnv
from typing import Optional, Callable


env = ShortcutMazeEnv(render_mode=None, layout_change_step=3000, max_episode_steps=500)
NUM_ACTIONS = env.action_space.n
NUM_STATES = env.observation_space.n
NUM_EPISODES = 1000
NUM_PLANNING_STEPS = [0, 5, 50]
EPSILON = 0.1
ALPHA = 0.1
GAMMA = 0.99
KAPPA = 0.001
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

def init_last_visited_times(num_states:int, num_actions:int):
    last_visited_time_step: dict[int, list[OptionalInt]] = dict()
    for state in range(num_states):
        last_visited_time_step[state] = [None for _ in range(num_actions)]
    return last_visited_time_step


def sample_action(state_qvalues:list[int], epsilon:float):
    if random.random() < epsilon:
        return random.randint(0, NUM_ACTIONS - 1)
    return state_qvalues.index(max(state_qvalues)) # get action which has the max q value

def random_prev_observed_state(last_visited_time_step: dict[int, list[OptionalInt]]):
    prev_observed_states = []
    for state in last_visited_time_step.keys():
        if any(last_visited_time_step[state]): # any val in list not None? if yes then the state was observed
            prev_observed_states.append(state)
    return random.choices(prev_observed_states, k=1)[0]

def random_planning_action_for_state(env_model_state:list[tuple[OptionalInt, OptionalInt]]):
    possible_actions = []
    for action, (reward, next_state) in enumerate(env_model_state):
        # check if that action was taken, if it was taken, reward and next_state wouldn't be None
        if (reward is not None) and (next_state is not None):
            possible_actions.append(action)
    return random.choices(possible_actions, k=1)[0]


def see_shortcut_maze(Q_vals:dict[int, list[float]], title:str, unblock:bool=False):
    # see the learned policy
    env = ShortcutMazeEnv(render_mode="rgb_array", layout_change_step=3000, max_episode_steps=500, unblock=unblock)
    show_one_episode(env, lambda state: sample_action(Q_vals[state], epsilon=0), f"images/shortcut_maze_{title}.gif", title=title)
    env.close()
    del env


def dynaQ_dynaQplus(num_planning_steps:int , dyna_q_plus:bool=False, log:bool=False, q_values=None, epsilon=EPSILON):
    plan = True if num_planning_steps>0 else False
    title = f"Dyna-Q {'+' if dyna_q_plus else ''} with {num_planning_steps} planning steps" if plan else "No planning"
    if not plan: assert not dyna_q_plus
    q_values = init_q_vals(NUM_STATES, NUM_ACTIONS) if q_values is None else q_values
    env_model = init_env_model(NUM_STATES, NUM_ACTIONS) if plan else None
    last_visited_time_step = init_last_visited_times(NUM_STATES, NUM_ACTIONS)

    sum_rewards_episodes = []; timestep_episodes = []
    total_step = 0
    for episode in range(1, NUM_EPISODES+1):
        state, info = env.reset(); sum_rewards = float(0)
        for tstep in count(1):
            total_step += 1
            action = sample_action(q_values[state], EPSILON)
            next_state, reward, done, truncated, info = env.step(action); sum_rewards += reward
            q_values[state][action] += ALPHA * (reward + GAMMA * max(q_values[next_state]) - q_values[state][action])
            last_visited_time_step[state][action] = total_step
            if env_model is not None:
                env_model[state][action] = (reward, next_state) # (reward, next_state)
            if done or truncated:
                break
            # if total_step == 2999:
            #     print("|| layout change || Taking snapshot of maze before layout change ||")
            #     see_shortcut_maze(q_values, "|| before change layout || " + title)
            # if total_step == 10000:
            #     print("|| Taking snapshot of maze after layout change ||")
            #     see_shortcut_maze(q_values, "|| after change layout || " + title)
            state = next_state
        sum_rewards_episodes.append(sum_rewards)
        timestep_episodes.append(tstep)
        if log:
            print(f"Epsisode: {episode} || Sum of Reward: {sum_rewards} || Total Timesteps: {tstep}")

        # Planning
        if plan:
            for planning_step in range(num_planning_steps):
                planning_state = random_prev_observed_state(last_visited_time_step) # randomly prev observed state for planning
                planning_action = random_planning_action_for_state(env_model[planning_state]) # randomly select a action that previously occurred in this state
                planning_reward, planning_next_state = env_model[planning_state][planning_action]
                
                if dyna_q_plus:
                    # To encourage behavior that tests
                    # long-untried actions, a special “bonus reward” is given on simulated experiences involving
                    # these actions. In particular, if the modeled reward for a transition is r, and the transition
                    # has not been tried in τ time steps, then **planning updates** are done as if that transition
                    # produced a reward of r + κ*(τ)^0.5, for some small  κ. This encourages the agent to keep
                    # testing all accessible state transitions and even to find long sequences of actions in order
                    # to carry out such tests.
                    #                                       current step - last visited
                    planning_reward += KAPPA * math.sqrt(total_step - last_visited_time_step[planning_state][planning_action])

                q_values[planning_state][planning_action] += ALPHA * (
                    planning_reward + GAMMA * max(q_values[planning_next_state]) - q_values[planning_state][planning_action]
                )
    print("Total Steps: ", total_step)
    # if dyna_q_plus:
    #     print(last_visited_time_step)
    return q_values, sum_rewards_episodes, timestep_episodes

if __name__ == "__main__":
    q1_values, sum1_rewards_episodes, timestep1_episodes = dynaQ_dynaQplus(num_planning_steps=0, dyna_q_plus=False)
    q2_values, sum2_rewards_episodes, timestep2_episodes = dynaQ_dynaQplus(num_planning_steps=5, dyna_q_plus=False)
    q3_values, sum3_rewards_episodes, timestep3_episodes = dynaQ_dynaQplus(num_planning_steps=50, dyna_q_plus=False)
    q4_values, sum4_rewards_episodes, timestep4_episodes = dynaQ_dynaQplus(num_planning_steps=10, dyna_q_plus=False)
    q5_values, sum5_rewards_episodes, timestep5_episodes = dynaQ_dynaQplus(num_planning_steps=25, dyna_q_plus=False)

    def moving_average(data, window_size): return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

    window_size = 50  # You can adjust the window size as needed

    smoothed_sum1_rewards = moving_average(sum1_rewards_episodes, window_size)
    smoothed_sum2_rewards = moving_average(sum2_rewards_episodes, window_size)
    smoothed_sum3_rewards = moving_average(sum3_rewards_episodes, window_size)
    smoothed_sum4_rewards = moving_average(sum4_rewards_episodes, window_size)
    smoothed_sum5_rewards = moving_average(sum5_rewards_episodes, window_size)
    
    till = 400
    plt.plot(smoothed_sum1_rewards[:till], label="Dyna-Q with 0 planning steps")
    plt.plot(smoothed_sum2_rewards[:till], label="Dyna-Q with 5 planning steps")
    plt.plot(smoothed_sum4_rewards[:till], label="Dyna-Q with 10 planning steps")
    plt.plot(smoothed_sum5_rewards[:till], label="Dyna-Q with 25 planning steps")
    plt.plot(smoothed_sum3_rewards[:till], label="Dyna-Q with 50 planning steps")
    plt.xlabel("Episodes")
    plt.ylabel("Smoothed Sum of Rewards")
    plt.grid(True)
    plt.title("|| Smoothed Sum of Rewards per Episode ||")
    plt.legend()
    plt.savefig("images/dyna_q_num_planning_steps_zoomed.png")
    plt.show()
    plt.close()

    till = None
    plt.plot(smoothed_sum1_rewards[:till], label="Dyna-Q with 0 planning steps")
    plt.plot(smoothed_sum2_rewards[:till], label="Dyna-Q with 5 planning steps")
    plt.plot(smoothed_sum4_rewards[:till], label="Dyna-Q with 10 planning steps")
    plt.plot(smoothed_sum5_rewards[:till], label="Dyna-Q with 25 planning steps")
    plt.plot(smoothed_sum3_rewards[:till], label="Dyna-Q with 50 planning steps")
    plt.xlabel("Episodes")
    plt.ylabel("Smoothed Sum of Rewards")
    plt.grid(True)
    plt.title("|| Smoothed Sum of Rewards per Episode ||")
    plt.legend()
    plt.savefig("images/dyna_q_num_planning_steps.png")
    plt.show()
    plt.close()

    see_shortcut_maze(q5_values, "Dyna-Q_with_25_planning_steps.gif")
    plt.close()

    print("\nWARNING: GOTTA DEBUG THE BELOW PART, CAN'T SEE THE IMPROVEMENT IN DYNA-Q+ OVER DYNA-Q\n")

    # Now unblock is true, env is changed to make the path to goal shorter. To see the difference between Dyna-Q and Dyna-Q+.
    # Initial Q values are set to Q values Dyna-Q, and we compare with train again with Dyna-Q+ and Dyna-Q on the changed env to see difference.
    env = ShortcutMazeEnv(render_mode="rgb_array", layout_change_step=3000, max_episode_steps=500, unblock=True)
    q_values_q, sum_rewards_episodes_q, timestep_episodes_q = dynaQ_dynaQplus(
        num_planning_steps=25, dyna_q_plus=False, q_values=q5_values, log=False
    )

    env = ShortcutMazeEnv(render_mode="rgb_array", layout_change_step=3000, max_episode_steps=500, unblock=True)
    q_values_qplus, sum_rewards_episodes_qplus, timestep_episodes_qplus = dynaQ_dynaQplus(
        num_planning_steps=25, dyna_q_plus=True, q_values=q5_values, log=False
    )

    plt.plot(moving_average(sum_rewards_episodes_q, 100)[:], label="Dyna-Q with 25 planning steps")
    plt.plot(moving_average(sum_rewards_episodes_qplus, 100)[:], label="Dyna-Q+ with 25 planning steps")
    plt.xlabel("Episodes")
    plt.ylabel("Smoothed Sum of Rewards")
    plt.grid(True)
    plt.title("|| Dyna-Q vs Dyna-Q+ ||")
    plt.legend()
    plt.savefig("images/dyna_q_vs_dyna_qplus.png")
    plt.show()
    plt.close()
    
    see_shortcut_maze(q_values_q, "Dyna-Q_with_25_planning_steps", unblock=True)
    see_shortcut_maze(q_values_qplus, "Dyna-Q+_with_25_planning_steps", unblock=True)  
    
