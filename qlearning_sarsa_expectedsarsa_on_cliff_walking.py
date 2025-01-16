"""
Experimenting Q-Learning, SARSA and Expected SARSA on Cliff Walking Environment
Page 132, Figure 6.6 from the book "Reinforcement Learning: An Introduction" by Richard S. Sutton and Andrew G. Barto
Author: Vachan V Y
"""

import gymnasium as gym
import random
from itertools import count
import matplotlib.pyplot as plt
from typing import Callable, Optional
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation as anim


env = gym.make("CliffWalking-v0")
NUM_EPISODES = 1000
EPSILON = 0.1
GAMMA = 0.99
NUM_ACTIONS = env.action_space.n
NUM_STATES = env.observation_space.n
ALPHA = 0.1
TERMINAL_STATE = 47


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
        action = action_sampler(state)
        state, reward, done, truncated, info = env.step(action)
        sum_rewards += reward
        if done or truncated:
            print(f"|| done at step: {step+1} ||")
            print(f"|| sum_rewards: {sum_rewards} ||")
            break
    frames.append(env.render())
    return plot_animation(frames, save_path, title=title, repeat=repeat)

def see_cliff_walking(Q_vals:dict[int, list[float]], title:str):
    # see the learned policy
    env = gym.make("CliffWalking-v0", render_mode="rgb_array")
    show_one_episode(env, lambda state: sample_action(Q_vals[state], epsilon=0), f"images/cliff_walking_{title}.gif", title=title)
    env.close()
    del env


def init_q_values(num_states:int, num_actions:int):
    q_values:dict[int, list[float]] = {}
    for state in range(num_states):
        q_values[state] = [
            random.random() if state != TERMINAL_STATE else 0 for _ in range(num_actions) 
        ] # [up, right, down, left]
    return q_values


def sample_action(qvals_of_state:list[float], epsilon:float):
    if random.random() < epsilon:
        return random.randint(0, NUM_ACTIONS - 1)
    else:
        return qvals_of_state.index(max(qvals_of_state))
    

def q_learning_update(
    alpha:float, gamma:float, reward:float, 
    next_state_qvals:list[float], qvals_of_state:list[float], 
    action:int, **kwargs
):
    qvals_of_state[action] += alpha * (reward + gamma * max(next_state_qvals) - qvals_of_state[action])

def sarsa_update(
    alpha:float, gamma:float, reward:float, 
    next_state_qvals:list[float], qvals_of_state:list[float], 
    action:int, next_action:int, **kwargs
):
    qvals_of_state[action] += alpha * (reward + gamma * next_state_qvals[next_action] - qvals_of_state[action])

def expected_sarsa_update(
    alpha:float, gamma:float, reward:float, 
    next_state_qvals:list[float], qvals_of_state:list[float], 
    action:int, **kwargs
):
    expected_value = sum(next_state_qvals) / len(next_state_qvals)
    qvals_of_state[action] += alpha * (reward + gamma * expected_value - qvals_of_state[action])

def double_q_learning_update(
    alpha:float, gamma:float, reward:float, 
    next_state_qvals_1:list[float], next_state_qvals_2:list[float], 
    qvals_of_state_1:list[float], qvals_of_state_2:list[float],
    action:int, **kwargs
):
    """
    >> Why Double Q-Learning? To avoid maximization bias

    >>> consider a single state s where there are many actions a whose true values, q(s, a),
    are all zero but whose estimated values, Q(s, a), are uncertain and thus distributed 
    some above and some below zero. The maximum of the true values is zero, but the maximum
    of the estimates is positive, a positive bias. We call this maximization bias.
    """
    if random.random() < 0.5:
        best_next_action = next_state_qvals_1.index(max(next_state_qvals_1)) # take action from Q1 but take Q value estimate from Q2 <= for Q1 update
        qvals_of_state_1[action] += alpha * (reward + gamma * next_state_qvals_2[best_next_action] - qvals_of_state_1[action])
    else:
        best_next_action = next_state_qvals_2.index(max(next_state_qvals_2)) # take action from Q2 but take Q value estimate from Q1 <= for Q2 update
        qvals_of_state_2[action] += alpha * (reward + gamma * next_state_qvals_1[best_next_action] - qvals_of_state_2[action])


def cliff_walking_experiment(alpha:float, gamma:float, epsilon:float, update_fn:Callable, seed:int, log:bool=False):
    random.seed(seed)
    Q_vals_A = init_q_values(NUM_STATES, NUM_ACTIONS)
    Q_vals_B = init_q_values(NUM_STATES, NUM_ACTIONS) if update_fn == double_q_learning_update else None
    sum_rewards_list = []
    for episode in range(1, NUM_EPISODES+1):
        state, info = env.reset()
        sum_rewards = 0
        action = sample_action(Q_vals_A[state], epsilon=epsilon)
        for tstep in count(0):
            next_state, reward, done, truncated, info = env.step(action)
            next_action = sample_action(Q_vals_A[next_state], epsilon=epsilon)
            sum_rewards += reward

            if update_fn == double_q_learning_update:
                update_fn(
                    alpha=alpha,
                    gamma=gamma,
                    reward=reward,
                    next_state_qvals_1=Q_vals_A[next_state],
                    next_state_qvals_2=Q_vals_B[next_state],
                    qvals_of_state_1=Q_vals_A[state],
                    qvals_of_state_2=Q_vals_B[state],
                    action=action
                )
            else:
                update_fn(
                    alpha=alpha,
                    gamma=gamma,
                    reward=reward,
                    next_state_qvals=Q_vals_A[next_state],
                    qvals_of_state=Q_vals_A[state],
                    action=action,
                    next_action=next_action
                )

            if done or truncated:
                break

            state = next_state
            action = next_action
            
        sum_rewards_list.append(sum_rewards)
        if log:
            print(
                f"|| Episode: {episode} || Sum of Reward: {sum_rewards} || info: {info} ||"
            )
    return sum_rewards_list, Q_vals_A, Q_vals_B


# average experiment over 100 runs
def experiment(alpha:float, gamma:float, epsilon:float):
    print("Running experiment with alpha:", alpha, "gamma:", gamma, "epsilon:", epsilon)
    print("Running Q-Learning")
    avg_sum_rewards_list_q_learning =  np.mean([
        cliff_walking_experiment(
            alpha=alpha, gamma=gamma, 
            epsilon=epsilon, update_fn=q_learning_update,
            seed=seed*42
        )[0] for seed in range(100)
    ], axis=0)

    print("Running SARSA")
    avg_sum_rewards_list_sarsa = np.mean([
        cliff_walking_experiment(
            alpha=alpha, gamma=gamma, 
            epsilon=epsilon, update_fn=sarsa_update,
            seed=seed*42
        )[0] for seed in range(100)
    ], axis=0)

    print("Running Expected SARSA")
    avg_sum_rewards_list_expected_sarsa = np.mean([
        cliff_walking_experiment(
            alpha=alpha, gamma=gamma, 
            epsilon=epsilon, update_fn=expected_sarsa_update,
            seed=seed*42
        )[0] for seed in range(100)
    ], axis=0)

    print("Running Double Q-Learning")
    avg_sum_rewards_list_double_q_learning = np.mean([
        cliff_walking_experiment(
            alpha=alpha, gamma=gamma, 
            epsilon=epsilon, update_fn=double_q_learning_update,
            seed=seed*42
        )[0] for seed in range(100)
    ], axis=0)

    print("Done!!!")
    plt.plot(avg_sum_rewards_list_q_learning, label="Q-Learning")
    plt.plot(avg_sum_rewards_list_sarsa, label="SARSA")
    plt.plot(avg_sum_rewards_list_expected_sarsa, label="Expected SARSA")
    plt.plot(avg_sum_rewards_list_double_q_learning, label="Double Q-Learning")
    plt.ylim(-200, 0)
    plt.yticks(np.arange(-200, 0, 20).tolist() + [-13])
    plt.xlabel("Episodes")
    plt.ylabel("Sum of Rewards")
    plt.grid()
    plt.text(0, -20, f"alpha: {alpha}, gamma: {gamma}, epsilon: {epsilon}", fontsize=12)
    plt.savefig(f"images/cliff_walking_gamma{gamma}_alpha{alpha}_epsilon{epsilon}.png")
    plt.legend()
    plt.show()
    plt.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    def t_or_f(arg):
        ua = str(arg).upper()
        if 'TRUE'.startswith(ua):
            return True
        elif 'FALSE'.startswith(ua):
            return False
        else:
            raise ValueError("Expected True or False")

    parser.add_argument("--experiments", type=t_or_f, default=False)
    parser.add_argument("--animation", type=t_or_f, default=True)
    args = parser.parse_args()

    if args.animation:
        qlearning_sum_rewards_list, qlearning_q_vals, _ = cliff_walking_experiment(
            alpha=ALPHA, gamma=GAMMA, epsilon=EPSILON, update_fn=q_learning_update, seed=42
        )

        sarsa_sum_rewards_list, sarsa_q_vals, _ = cliff_walking_experiment(
            alpha=ALPHA, gamma=GAMMA, epsilon=EPSILON, update_fn=sarsa_update, seed=42
        )

        expected_sarsa_sum_rewards_list, expected_sarsa_q_vals, _ = cliff_walking_experiment(
            alpha=ALPHA, gamma=GAMMA, epsilon=EPSILON, update_fn=expected_sarsa_update, seed=42
        )

        double_qlearning_sum_rewards_list, double_qlearning_q_vals_A, double_qlearning_q_vals_B = cliff_walking_experiment(
            alpha=ALPHA, gamma=GAMMA, epsilon=EPSILON, update_fn=double_q_learning_update, seed=42
        )

        plt.plot(qlearning_sum_rewards_list, label="Q-Learning")
        plt.plot(sarsa_sum_rewards_list, label="Sarsa")
        plt.plot(expected_sarsa_sum_rewards_list, label="Expected Sarsa")
        plt.plot(double_qlearning_sum_rewards_list, label="Double Q-Learning")
        plt.ylim(-500, 0)
        plt.yticks(np.arange(-500, 0, 20).tolist() + [-13])
        plt.xlabel("Episodes")
        plt.ylabel("Sum of Rewards")
        plt.title("Cliff Walking Experiment")
        plt.legend()
        plt.grid()
        plt.close()

        print("making gif for qlearning")
        see_cliff_walking(qlearning_q_vals, "qlearning"); plt.close()

        print("making gif for sarsa")
        see_cliff_walking(sarsa_q_vals, "sarsa"); plt.close()

        print("making gif for expected sarsa")
        see_cliff_walking(expected_sarsa_q_vals, "expected_sarsa"); plt.close()

        print("making gif for double qlearning")
        see_cliff_walking(double_qlearning_q_vals_A, "double_qlearning"); plt.close()

        print("Done!!! See in the images folder for the gifs")

    if args.experiments:
        experiment(alpha=ALPHA, gamma=GAMMA, epsilon=EPSILON)
        experiment(alpha=ALPHA, gamma=0.95, epsilon=EPSILON)
        experiment(alpha=ALPHA, gamma=0.9, epsilon=EPSILON)
        experiment(alpha=ALPHA, gamma=0.75, epsilon=EPSILON)
        experiment(alpha=ALPHA, gamma=0.75, epsilon=0)
        experiment(alpha=ALPHA, gamma=0.5, epsilon=EPSILON)
        experiment(alpha=0.01, gamma=GAMMA, epsilon=EPSILON)
        experiment(alpha=0.01, gamma=GAMMA, epsilon=0)
        print("Done!!!")

