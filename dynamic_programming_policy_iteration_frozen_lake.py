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


def policy_evaluation(V:dict[int, float], pi:dict[int, int], env:gym.Env, gamma:float, theta:float):
    while True:
        delta = 0
        for state in range(env.observation_space.n):
            v = V[state]
            V[state] = sum([p*(r + gamma*V[s_]) for p, s_, r, _ in env.unwrapped.P[state][pi[state]]])
            delta = max(delta, abs(v - V[state]))
        if delta < theta:
            break


def policy_improvement(V:dict[int, float], pi:dict[int, int], env:gym.Env, gamma:float):
    policy_stable = True
    for state in range(env.observation_space.n):
        old_action = pi[state]
        # find the largest element which maximizes the returned value from the below lambda function
        pi[state] = max(
            range(env.action_space.n), 
            key=lambda action: sum([p*(r + gamma*V[s_]) for p, s_, r, _ in env.unwrapped.P[state][action]])
        )
        if old_action != pi[state]:
            policy_stable = False
    return policy_stable


def policy_iteration(env:gym.Env, gamma:float, theta:float):
    V, pi = init_v_pi_vals(env.observation_space.n)
    while True:
        policy_evaluation(V, pi, env, gamma, theta)
        if policy_improvement(V, pi, env, gamma):
            break
    return V, pi

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--is_slippery", type=str)
    args = parser.parse_args()

    SLIPPERY = True if args.is_slippery.lower()=="y" else False

    env = gym.make("FrozenLake-v1", is_slippery=SLIPPERY)
    print(f"Environment is {'slippery' if SLIPPERY else 'not slippery'}")
    v_vals, pi_vals = policy_iteration(env, gamma=0.9, theta=1e-8)
    env.close()
    del env

    # see the optimal policy
    env = gym.make("FrozenLake-v1", is_slippery=SLIPPERY, render_mode="rgb_array")

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
        frames, save_path=f"images/frozen{'_slippery' if SLIPPERY else ''}_lake_policy_iteration.gif", 
        title=f"Policy Iteration on Frozen {'STOCHASTIC' if SLIPPERY else 'DETERMINISTIC'} Lake Environment", repeat=False, interval=2000
    )

    print("Done!")
    env.close()
