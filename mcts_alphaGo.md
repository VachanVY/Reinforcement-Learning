# Monte Carlo Tree Search (MCTS) Algorithm
* 

---

# Mastering the Game of Go with Deep Neural Networks and Tree Search
![image](https://github.com/user-attachments/assets/5c8e404c-eed3-4896-8223-143fe17057d2)

* All games of perfect information have an optimal value function, $v^*(s)$,
which determines the outcome of the game, from every board position
or state $s$, under perfect play by all players.
* These games may be solved by recursively computing the optimal value function in a search tree
  $b^d$
  containing approximately
  possible sequences of moves, where $b$ is
  the game’s breadth (number of legal moves per position) and $d$ is its
  depth (game length).
* Extensive search is infeasible, but two general principles can reduce the effectiveness of the search space:
   * First, the depth of the search may be reduced by position evaluation:
     truncating the search tree at state $s$ and replacing the subtree below $s$ by an approximate value function $v(s) \approx v^*(s)$ that predicts the outcome
     from state $s$.
   * Second, the breadth of the search
     may be reduced by sampling actions from a policy $p(a|s)$ that is a probability
     distribution over possible moves $a$ in position $s$.
* Monte Carlo tree search (MCTS) uses Monte Carlo rollouts to estimate the value of each state in a search tree.
* As more simulations are executed, the search tree grows larger and the relevant values become more accurate.
* The policy used to select actions during search is also improved over time, by selecting children with higher values.

* We pass in the board
  position as a $19 \times 19$ image and use convolutional layers to construct a
  representation of the position.
* We use these neural networks to reduce
  the effective depth and breadth of the search tree: evaluating positions
  using a value network, and sampling actions using a policy network.
* We begin by training a supervised
  learning (SL) policy network $p_\sigma$ directly from expert human moves.
* We also train a
  fast policy $p_\pi$ that can rapidly sample actions during rollouts.
* Next, we
  train a reinforcement learning (RL) policy network $p_\rho$ that improves
  the SL policy network by optimizing the final outcome of games of self-play.
* This adjusts the policy towards the correct goal of winning games,
  rather than maximizing predictive accuracy.
* Finally, we train a value
  network $v_\theta$ that predicts the winner of games played by the RL policy
  network against itself.

## Supervised Learning of Policy Networks
* For the first stage of the training pipeline, we build on prior work
  on predicting expert moves in the game of Go using supervised
  learning.
* The SL policy network $p_\sigma(a | s)$ alternates between convolutional
  layers with weights $\sigma$, and rectifier nonlinearities. A final softmax layer outputs
  a probability distribution over all legal moves $a$.
* The policy network is trained on randomly sampled state-action pairs $(s, a)$, using stochastic gradient ascent to
  maximize the likelihood of the human move $a$ selected in state $s$.
* The network predicted expert moves on a held-out test set with an accuracy of
  57.0% using all input features, and 55.7% using only raw board position and move history as inputs, compared to the state-of-the-art from
  other research groups of 44.4%.
* We also
  trained a faster but less accurate rollout policy $p_\pi(a|s)$, using a linear
  softmax of small pattern features (see Extended Data Table 4) with
  weights $\theta$; this achieved an accuracy of 24.2%, using just $2 \mu s$ to select
  an action, rather than 3 ms for the policy network.

## Reinforcement Learning of Policy Networks
![image](https://github.com/user-attachments/assets/363314f8-b814-47fc-b37e-0d636bdd6308)

* The second stage of the training pipeline aims at improving the policy network by policy gradient reinforcement learning (RL).
* We play games between the current policy network $p_\rho$ and a randomly selected previous iteration of the policy network.
* Randomizing from a pool of opponents in this way stabilizes training by preventing overfitting to the current policy.
* ![image](https://github.com/user-attachments/assets/76f2d6c2-6674-4ade-8e24-c7cacee79627)
* We evaluated the performance of the RL policy network in-game
  play, sampling each move $a_t \sim p_\rho (.|s_t)$ from its output probability
  distribution over actions.

## Reinforcement Learning of Value Networks
![image](https://github.com/user-attachments/assets/951bfaab-1b15-494f-8067-7fbd60d83409)
![image](https://github.com/user-attachments/assets/a7b56249-c7dc-471f-8243-3db30e26f47e)

---
* We also assessed variants of AlphaGo that evaluated positions
  using just the value network ($\lambda = 0$) or just rollouts ($\lambda = 1$) (see
  Fig. 4b). Even without rollouts, AlphaGo exceeded the performance
  of all other Go programs, demonstrating that value networks provide
  a viable alternative to Monte Carlo evaluation in Go. However, the
  mixed evaluation ($\lambda = 0.5$) performed best, winning 
  95% of games
  against other variants.
