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

---

# AlphaGo Zero: Mastering the game of Go without human knowledge

## Introduction

A long-standing goal of artificial intelligence is an algorithm that learns, *tabula rasa*, superhuman proficiency in challenging domains. **Recently, AlphaGo became the first program to defeat a world champion in the game of Go.** The tree search in AlphaGo evaluated positions and selected moves using deep neural networks. These neural networks were trained by supervised learning from human expert moves, and by reinforcement learning from self-play. 

Here we introduce *AlphaGo Zero*, an algorithm based solely on reinforcement learning, without human data, guidance or domain knowledge beyond game rules. AlphaGo becomes its own teacher: a neural network is trained to predict AlphaGo Zero’s own move selections and the winner of AlphaGo’s games. This neural network improves the strength of the tree search, resulting in highly accurate move selection and stronger self-play in the next iteration. **Starting *tabula rasa*, our new program AlphaGo Zero achieved superhuman performance, winning 100–0 against the previously published, champion-defeating AlphaGo.**

## Self-Play Reinforcement Learning in AlphaGo Zero

The program plays a game $s_1, ..., s_T$ against itself. In each position $s$, an MCTS $\pi_s$ is executed using the latest neural network $f_\theta$. Moves are selected according to the search probabilities computed by the MCTS $a \sim \pi_s$. The terminal position $s_T$ is scored according to the rules of the game to compute the game winner $z$.

The neural network takes the raw board position $s$ as its input, passes it through many convolutional layers with parameters $\theta$, and outputs both a vector $p$, representing a probability distribution over moves, and a scalar value $v$, representing the probability of the current player winning in position $s$. The neural network parameters $\theta$ are updated to maximize the similarity of the policy network $p$ to the search probabilities $\pi$, and to minimize the error between the predicted winner $v$ and the game winner $z$:

$$(p,v) = f_\theta(s)$$

The new parameters are used in the next iteration of self-play.

## Reinforcement Learning in AlphaGo Zero

Our new method uses a deep neural network $f_\theta$ with parameters $\theta$. This neural network takes as an input the raw board representation of the position and its history, and outputs both move probabilities and a value, $p = f_\theta(s)$. The vector of move probabilities represents the probability of selecting each move $a$ (including pass), $p_a = P(a|s)$. The value is a scalar evaluation, estimating the probability of the current player winning from position $s$.

This neural network combines the roles of both policy network and value network into a single architecture. The neural network consists of many *residual blocks* of convolutional layers with batch normalization and rectifier nonlinearities.

The neural network in AlphaGo Zero is trained from games of self-play by a novel reinforcement learning algorithm. In each position $s$, an MCTS search is executed, guided by the neural network. The MCTS search outputs probabilities $\pi$ that are proportional to the probabilities of visiting each move. MCTS may therefore be viewed as a powerful policy evaluation or *expert* $\pi_{exp}$. The main idea of our reinforcement learning algorithm is to use these search operators.

MCTS may be viewed as a self-play algorithm that, given neural network parameters $\theta$ and a root position $s$, computes a vector of search probabilities recommending moves to play: $\pi = \alpha(s)$, proportional to the exponentiated visit count for each move, $\pi_a \propto N(s,a)^{1/\tau}$, where $\tau$ is a temperature parameter.

The neural network is trained by a self-play reinforcement learning algorithm that uses MCTS to play each move. First, the neural network is initialized to random weights $\theta$. At each subsequent iteration $i \geq 1$, games of self-play are generated. At each step, an MCTS search is executed using the previous iteration of neural network $f_{\theta_{i-1}}$ and a move is played by sampling the search probabilities $\pi$. A game terminates at step $T$ when both players pass, when the search value drops below a resignation threshold, or when the game reaches a maximum length. The game is then scored to give a final reward of $r \in \{-1, 1\}$. The data for each time-step $t$ is stored as $(s_t, \pi_t, z_t)$, where $z_t = r$ is the game winner from the perspective of the current player at step $t$ in parallel.

## Monte Carlo Tree Search (MCTS) in AlphaGo Zero

Each simulation traverses the MCTS search tree by selecting moves that maximize an upper confidence bound $Q(s,a) + U(s,a)$, where $U(s,a) \propto \frac{P(s,a)}{1+N(s,a)}$ until a leaf node $s_L$ is encountered. This leaf position is expanded and evaluated only once by the network to generate visit counts, prior probabilities, and evaluation function $(P(s_L),V(s_L)) = f_\theta(s_L)$. Each edge $(s,a)$ traversed in the simulation is updated to increment visit count $N(s,a)$ and to update its action value to the mean evaluation over these simulations:

$$Q(s,a) = \frac{1}{N(s,a)} \sum_{i=1}^{N(s,a)} v_i$$

where $s, a \rightarrow s'$ indicates that a simulation eventually reached $s'$ after taking move $a$ from position $s$.

### Empirical Analysis of AlphaGo Zero Training

We applied our reinforcement learning pipeline to train our program AlphaGo Zero. Training started from completely random behavior and continued without human intervention for approximately three days.

Over the course of training, 4.9 million games of self-play were generated, using 1,600 simulations for each MCTS, which corresponds to approximately 0.4s thinking time per move. Parameters were updated approximately.

![image](https://github.com/user-attachments/assets/a3b1c17d-3dae-49bb-af4c-e5af760c078b)

![image](https://github.com/user-attachments/assets/50f358b2-9a65-4c34-8f77-411605eb7f2d)

Notably, although supervised learning achieved higher
move prediction accuracy, the self-learned player performed much
better overall, defeating the human-trained player within the first 24 h
of training. This suggests that AlphaGo Zero may be learning a strategy
that is qualitatively different to human play
