# Reinforcement-Learning
## Playing Atari with Deep Reinforcement Learning

![Alt Text](images/dqn_pong.gif)

![image](https://github.com/user-attachments/assets/c778e6b2-bf32-4638-a835-e7d6d3490844)

## (Important Clips from the paper)
* The
model is a convolutional neural network, trained with a variant of Q-learning,
whose input is raw pixels and whose output is a value function estimating future
rewards. We apply our method to seven Atari 2600 games from the Arcade Learning
Environment, with no adjustment of the architecture or learning algorithm

### Introduction
* Most successful RL applications
that operate on these domains have relied on hand-crafted features combined with linear value
functions or policy representations. Clearly, the performance of such systems heavily relies on the
quality of the feature representation.
* Firstly, most successful deep learning applications to date have required large amounts of hand labelled
training data
* RL algorithms, on the other hand, must be able to learn from a scalar reward
signal that is frequently sparse, noisy and delayed. The delay between actions and resulting rewards,
which can be thousands of timesteps long, seems particularly daunting when compared to the direct
association between inputs and targets found in supervised learning
* Another issue is that most deep
learning algorithms assume the data samples to be independent, while in reinforcement learning one
typically encounters sequences of highly correlated states
* Furthermore, in RL the data distribution
changes as the algorithm learns new behaviours, which can be problematic for deep learning
methods that assume a fixed underlying distribution

* The network is
trained with a variant of the Q-learning [26] algorithm, with stochastic gradient descent to update
the weights. To alleviate the problems of correlated data and non-stationary distributions, we use
an experience replay mechanism [13] which randomly samples previous transitions, and thereby
smooths the training distribution over many past behaviors
* The network was not provided
with any game-specific information or hand-designed visual features, and was not privy to the
internal state of the emulator; it learned from nothing but the video input, the reward and terminal
signals, and the set of possible actions—just as a human player would. Furthermore the network architecture
and all hyperparameters used for training were kept constant across the games

### Background
* ![image](https://github.com/user-attachments/assets/20f449cc-9595-4f4e-9128-59599b55549d)
* ![image](https://github.com/user-attachments/assets/0e714b40-8993-481d-a57c-b077a099d837)

### Deep Reinforcement Learning
* Experience replay where we store the agent’s experiences at each time-step, et = (st; at; rt; st+1)
in a data-set D = e1; :::; eN, pooled over many episodes into a replay memory. During the inner
loop of the algorithm, we apply Q-learning updates, or minibatch updates, to samples of experience,
e  D, drawn at random from the pool of stored samples. After performing experience replay,
the agent selects and executes an action according to an eps-greedy policy
* ![image](https://github.com/user-attachments/assets/8434e1f9-70e8-4b5d-9d3d-d093a742b9e2)
* Advantages:
  * First, each step of experience is potentially used in many weight updates, which allows for greater data efficiency.
  * Second, learning directly from consecutive samples is inefficient, due to the strong correlations
between the samples; randomizing the samples breaks these correlations and therefore reduces the
variance of the updates
  * Third, when learning on-policy the current parameters determine the next
data sample that the parameters are trained on. For example, if the maximizing action is to move left
then the training samples will be dominated by samples from the left-hand side; if the maximizing
action then switches to the right then the training distribution will also switch. It is easy to see how
unwanted feedback loops may arise and the parameters could get stuck in a poor local minimum, or
even diverge catastrophically. Note that when learning by experience replay, it is necessary to learn off-policy
(because our current parameters are different to those used to generate the sample), which motivates
the choice of Q-learning

* In practice, our algorithm only stores the last N experience tuples in the replay memory, and samples
uniformly at random from D when performing updates. This approach is in some respects limited
since the memory buffer does not differentiate important transitions and always overwrites with
recent transitions due to the finite memory size N. Similarly, the uniform sampling gives equal
importance to all transitions in the replay memory. A more sophisticated sampling strategy might
emphasize transitions from which we can learn the most, similar to prioritized sweeping

![image](https://github.com/user-attachments/assets/4ace2761-7e8c-4fcd-9691-ac6cdfce0124)
![image](https://github.com/user-attachments/assets/8a03abfc-1bd0-4ed9-ae5e-1e12e8d44cb7)

### Experiments
* ![image](https://github.com/user-attachments/assets/56ec5a1b-8976-4de8-9a48-904af341c71d)
