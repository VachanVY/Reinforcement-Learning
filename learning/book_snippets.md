## Some snippets important snippets from books
* In reinforcement learning the only guidance the agent gets is through rewards, and rewards are typically sparse and delayed. For example, if the agent manages to balance the pole for 100 steps, how can it know which of the 100 actions it took were good, and which were bad?
* All it knows is that the pole fell after the last action, but surely this previous action is not entirely responsible. This is called the
credit assignment problem: when the agent gets a reward, it is hard for it to know which actions should get credited (or blamed) for it
* To tackle this problem, a common strategy is to evaluate an action based on the sum of all the rewards that come after it, usually applying a
discount factor, γ (gamma), at each step
* Of course, a good action may be followed by several bad actions that cause the pole to fall quickly, resulting in the good action getting a low return
* However, if we play the game enough times, on average good actions will get a higher return than bad ones. We want to estimate how much better or worse an action is, compared to the other possible actions, on average.
* This is called the action advantage. For this, we must run many episodes and normalize all the action returns, by subtracting the mean and dividing by the standard deviation. After that, we can reasonably assume that actions with a negative advantage were bad while actions with a positive advantage were good