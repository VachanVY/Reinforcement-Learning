import gymnasium as gym
from itertools import count
from typing import Optional
import matplotlib.pyplot as plt
import matplotlib.animation as anim


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
    animation.save(save_path, writer="ffmpeg", fps=20)
    return animation


def init_v_pi_vals(num_states:int):
    V = {state_val:0.0 for state_val in range(num_states)}
    pi = {state_val:0 for state_val in range(num_states)}
    return V, pi # V(s) = v # pi(s) = a # init values arbitrarily given state s


def value_iteration(env:gym.Env, gamma:float, theta:float):
    V, _ = init_v_pi_vals(env.observation_space.n)
    while True:
        delta = 0
        for state in range(env.observation_space.n):
            v = V[state]
            V[state] = max(
                [sum([p*(r + gamma*V[s_]) for p, s_, r, _ in env.unwrapped.P[state][action]]) for action in range(env.action_space.n)]
            )
            delta = max(delta, abs(v - V[state]))
        if delta < theta:
            break
    return V


def extract_policy(V:dict[int, float], env:gym.Env, gamma:float):
    pi = {state_val:0 for state_val in range(env.observation_space.n)}
    for state in range(env.observation_space.n):
        pi[state] = max(
            range(env.action_space.n),
            key=lambda action: sum([p*(r + gamma*V[s_]) for p, s_, r, _ in env.unwrapped.P[state][action]])
        )
    return pi


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_postfix", type=str, default="")
    args = parser.parse_args()

    ENV_NAME = "Taxi-v3"
    env = gym.make(ENV_NAME)
    gamma = 0.99
    theta = 1e-6
    V = value_iteration(env, gamma, theta)
    pi_vals = extract_policy(V, env, gamma)
    env.close()

    # see optimal policy
    env = gym.make(ENV_NAME, render_mode="rgb_array")

    state, info = env.reset()
    frames = [env.render()]
    for i in count():
        action = pi_vals[state]
        state, reward, done, truncated, info = env.step(action)
        print("|| rewards:",  reward, "||")
        frames.append(env.render())
        if done or truncated:
            break

    plot_animation(
        frames, save_path=f"images/{ENV_NAME}_value_iteration{args.path_postfix}.gif", 
        title=f"Value Iteration on {ENV_NAME} Environment", repeat=False, interval=2000
    ); plt.close()

    print("Done!")
    env.close()

# CLEAR EXPLANATION OF THE TAXI-V3 ENVIRONMENT FROM CHATGPT:
"""
> Environment: Taxi-v3

>> Initial State:

The taxi starts at a random location within the grid.
The passenger starts at one of the designated pick-up locations.
The passenger also has a randomly assigned destination (one of the four designated locations).

>> Objective:

The agent (taxi) must:
Navigate to the passenger's current location.
Pick up the passenger.
Navigate to the passenger's desired destination.
Drop off the passenger at the correct destination.
The episode ends after a successful drop-off.
Actions: The environment allows six discrete actions:

0 (South): Move the taxi one cell down.
1 (North): Move the taxi one cell up.
2 (East): Move the taxi one cell right.
3 (West): Move the taxi one cell left.
4 (Pickup): Attempt to pick up the passenger.
5 (Drop-off): Attempt to drop off the passenger.

>> State Space: The state is a combination of:

The taxi's position (25 possible positions in a 5x5 grid).
The passenger's location (5 possible states: Red, Green, Yellow, Blue, or in the taxi).
The passenger's destination (4 possible locations: Red, Green, Yellow, Blue).
Total state space: 
25*5*4 = 500 possible states.

>> Rewards:

+20 for successfully dropping off the passenger at the correct location.
-10 for attempting to pick up or drop off the passenger at the wrong location.
-1 for each step taken, including valid moves and unsuccessful attempts to pick up or drop off the passenger. This incentivizes the agent to solve the task as quickly as possible.
"""