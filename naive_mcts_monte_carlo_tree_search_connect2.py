import gymnasium as gym
import numpy as np
from itertools import count
from custom_envs import Connect2Env
import typing as tp
import math
import random
from copy import deepcopy


class MCTSNode:
    def __init__(self, env:Connect2Env, parent=None):
        self.env = env # env's current state should be same as `self.state`
        self.state = env.board.copy()
        # possible actions from this state, can be an empty list if the state is terminal
        self.possible_actions = [i for i in range(len(self.state)) if self.state[i] == 0]
        self.children:list[MCTSNode|None] = [] # possible states from this state
        self.parent:tp.Optional[MCTSNode] = parent
        self.num_visits = 0
        self.sum_backed = 0

        self._get_children()

    def _get_children(self):
        """Get the children of the node"""
        if not self.children:
            for action in self.possible_actions:
                env = deepcopy(self.env)
                state, reward, done, truncated, info = env.step(action)
                self.children.append(MCTSNode(env, parent=self))
    
    def expand_by_selecting_action(self, weighted_rand:bool=True):
        """Expand the tree given action"""
        if not self.children:
            return None
        
        children_ucbs = [child._ucb(parent_visits=self.num_visits) for child in self.children]
        if not weighted_rand:
            return self.possible_actions[children_ucbs.index(max(children_ucbs))]
        return random.choices(self.possible_actions, weights=children_ucbs, k=1)[0]

    def _ucb(self, parent_visits:int, c:float=2.0):
        """Upper Confidence Bound to select the best action, Exploration vs Exploitation"""
        value = self.sum_backed/self.num_visits
        exploration_term = math.sqrt(math.log(parent_visits)/self.num_visits) if self.num_visits > 0 else 1e9
        return value + c*exploration_term
           
    def rollout(self):
        """Node not visited, simulate the game till the end to get value for that state"""
        env = deepcopy(self.env)
        state = env.board
        while True:
            if np.all(state != 0) or env._check_win(): # terminal state
                return env.get_value_of_state(state)
            action = random.choice([i for i in range(len(state)) if state[i] == 0])
            state, reward, done, truncated, info = env.step(action)
    
    def backup(self, value:float):
        """Update the value of the nodes in the path"""
        self.num_visits += 1
        self.sum_backed += value
        if self.parent is not None:
            self.parent.backup(value)


class MCTSRootNode(MCTSNode):
    ...


class config:
    num_episodes = 1000
    num_iterations = 10


def run():
    env = Connect2Env()
    sum_rewards_list = []
    for episode in range(1, config.num_episodes+1):
        state, info = env.reset()
        sum_rewards = 0
        for tstep in count(1):
            root = MCTSRootNode(env, parent=None) # current state => root node

            for _ in range(config.num_iterations): # number of iterations
                # Selection
                selected_node = root
                while selected_node.children: # while does not have a child
                    action:int = selected_node.expand_by_selecting_action()
                    selected_node._get_children()
                    selected_node = selected_node.children[action]
                
                # Expansion
                if selected_node.num_visits > 0:
                    action = selected_node.expand_by_selecting_action()
                    selected_node = selected_node.children[action]

                # Simulation
                value = selected_node.rollout()
                
                # Backup
                selected_node.backup(value)

            # Select the best action
            action = root.expand_by_selecting_action(weighted_rand=False)
            state, reward, done, truncated, info = env.step(action)
            sum_rewards += reward

            if done or truncated:
                break

        sum_rewards_list.append(sum_rewards)
        print(f"|| Episode: {episode} || Sum of reward: {sum_rewards} ||")
    return sum_rewards_list
