"""Author: CHATGPT"""

import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np

# Actions: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
ACTION_DELTAS = {
    0: (-1, 0),
    1: (0, 1),
    2: (1, 0),
    3: (0, -1),
}

class ShortcutMazeEnv(gym.Env):
    """
    A "shortcut maze" environment inspired by Sutton & Barto Example 8.3.
    
    - Maze is a 6-row x 9-column grid (rows=6, cols=9).
    - Start (S) at bottom-left corner => (row=5, col=3).
    - Goal (G) at top-right corner => (row=0, col=8).
    - There is a horizontal "block row" that initially leaves
      only the leftmost column open. After 'layout_change_step'
      steps, one block on the right side is removed, creating
      a shortcut to the goal.
    """

    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(self, layout_change_step, max_episode_steps, unblock:bool=False, render_mode="human", **kwargs):
        super().__init__()
        self.render_mode = render_mode
        self.layout_change_step = layout_change_step
        self.max_episode_steps = max_episode_steps
        self.unblock = unblock  # Store the unblock parameter

        # TOTAL Number of steps taken since environment was created
        self.total_steps = 0

        # Maze dimensions
        self.nrows = 6
        self.ncols = 9

        # Define the action and observation spaces
        # We'll treat each grid cell as a discrete state index.
        self.action_space = spaces.Discrete(4)  # up, right, down, left
        self.observation_space = spaces.Discrete(self.nrows * self.ncols)

        # Starting position (row=5, col=3)
        self.start_pos = (5, 3)
        # Goal position (row=0, col=8)
        self.goal_pos = (0, 8)

        # Build initial grid layout
        # 0 => free cell, 1 => blocked cell
        self.grid = self._build_initial_grid()
        if unblock:
            self.grid[3, 7] = 0

        # Track the agent state
        self.agent_pos = None
        self.num_steps_taken = 0

        # Pygame rendering stuff
        self.window = None
        self.clock = None
        self.cell_size = 60

    def _build_initial_grid(self):
        """
        Build the initial layout.
        The middle row (row=3 from the top if 0=top) is blocked except col=0.
        """
        grid = np.zeros((self.nrows, self.ncols), dtype=int)

        # Block row = 3 (0-based from the top). 
        # That means row index 3 is blocked except for col=0.
        # So row=3, col=1..8 are blocked.
        # (Remember row=0 is top, row=5 is bottom.)
        for c in range(1, self.ncols):
            grid[3, c] = 1  # blocked

        return grid

    def _pos_to_obs(self, row, col):
        """Convert (row, col) into a single integer for the observation space."""
        return row * self.ncols + col

    def _is_valid(self, row, col):
        """Check if (row, col) is within bounds and not blocked."""
        if 0 <= row < self.nrows and 0 <= col < self.ncols:
            return self.grid[row, col] == 0  # 0 => free
        return False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = list(self.start_pos)
        self.num_steps_taken = 0

        # Restore initial grid and consider unblock parameter
        self.grid = self._build_initial_grid()
        if self.unblock:  # Apply unblock if specified
            self.grid[3, 7] = 0

        obs = self._pos_to_obs(*self.agent_pos)
        info = {}
        return obs, info

    def step(self, action):
        # Count step
        self.num_steps_taken += 1
        self.total_steps += 1

        # If it's time to open the shortcut, do so
        if self.total_steps == self.layout_change_step:
            # Remove one block on the right side (row=3, col=7 for instance)
            self.grid[3, 7] = 0  # open this cell

        # Move agent according to the action if valid
        dr, dc = ACTION_DELTAS[action]
        new_r = self.agent_pos[0] + dr
        new_c = self.agent_pos[1] + dc

        if self._is_valid(new_r, new_c):
            self.agent_pos = [new_r, new_c]  # update agent position

        # Check if reached goal
        done = (self.agent_pos[0] == self.goal_pos[0] and 
                self.agent_pos[1] == self.goal_pos[1])

        # You can tweak the reward function to match your preference:
        # e.g., -1 per step, +10 if goal is reached.
        reward = -1
        if done:
            reward = 0  # or +10, or +1, etc.

        # Check for time truncation
        truncated = self.num_steps_taken >= self.max_episode_steps

        obs = self._pos_to_obs(*self.agent_pos)
        info = {}

        return obs, reward, done, truncated, info

    def render(self):
        """
        Renders the grid using pygame.
        - White squares for free cells
        - Gray squares for blocked cells
        - Blue circle for the agent
        - Green square for the goal
        
        Returns a NumPy array if render_mode is 'rgb_array'.
        """
        if self.render_mode not in ["human", "rgb_array"]:
            return

        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.set_caption("Shortcut Maze Env")
            width = self.ncols * self.cell_size
            height = self.nrows * self.cell_size
            self.window = pygame.display.set_mode((width, height))
            self.clock = pygame.time.Clock()
        
        canvas = pygame.Surface((self.ncols * self.cell_size, self.nrows * self.cell_size))
        canvas.fill((255, 255, 255))

        # Draw grid
        for r in range(self.nrows):
            for c in range(self.ncols):
                rect = pygame.Rect(
                    c * self.cell_size, 
                    r * self.cell_size, 
                    self.cell_size, 
                    self.cell_size
                )

                if self.grid[r, c] == 1:
                    pygame.draw.rect(canvas, (128, 128, 128), rect)
                else:
                    pygame.draw.rect(canvas, (255, 255, 255), rect)
                
                pygame.draw.rect(canvas, (0, 0, 0), rect, 1)

        # Draw goal (G) as a green rectangle
        goal_rect = pygame.Rect(
            self.goal_pos[1] * self.cell_size,
            self.goal_pos[0] * self.cell_size,
            self.cell_size,
            self.cell_size
        )
        pygame.draw.rect(canvas, (0, 200, 0), goal_rect)

        # Draw agent as a blue circle
        agent_center = (
            self.agent_pos[1] * self.cell_size + self.cell_size // 2,
            self.agent_pos[0] * self.cell_size + self.cell_size // 2
        )
        pygame.draw.circle(canvas, (0, 0, 200), agent_center, self.cell_size // 3)

        if self.render_mode == "human":
            self.window.blit(canvas, (0, 0))
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), (1, 0, 2)
            )


    def close(self):
        if self.window is not None:
            pygame.quit()
            self.window = None
        self.total_steps = 0

# test
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib import animation as anim
    from itertools import count
    from typing import Optional, Callable

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
            except: action = action_sampler() # env.action_space.sample
            state, reward, done, truncated, info = env.step(action)
            sum_rewards += reward
            if done or truncated:
                print(f"|| done at step: {step+1} ||")
                print(f"|| sum_rewards: {sum_rewards} ||")
                break
        frames.append(env.render())
        return plot_animation(frames, save_path, title=title, repeat=repeat)
    
    # see the learned policy
    env = ShortcutMazeEnv(render_mode="rgb_array", layout_change_step=300, max_episode_steps=500)
    show_one_episode(env, env.action_space.sample, f"dsds{3}.gif", title="sdsda")
    env.close()
    del env
