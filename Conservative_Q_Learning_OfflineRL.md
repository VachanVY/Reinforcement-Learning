# Conservative Q-Learning for Offline Reinforcement Learning
* Directly utilizing existing value-based off-policy RL algorithms in an offline setting generally results
in poor performance, due to issues with bootstrapping from out-of-distribution actions and
overfitting. This typically manifests as erroneously optimistic value function estimates. If
we can instead learn a conservative estimate of the value function, which provides a lower bound on
the true values, this overestimation problem could be addressed
* In fact, because policy evaluation and improvement typically only use the value of the policy, we can learn a less conservative lower
bound Q-function, such that only the expected value of Q-function under the policy is lower-bounded,
as opposed to a point-wise lower bound
* The primary contribution is an algorithmic framework, which we call conservative Q-learning (CQL),
for learning conservative, lower-bound estimates of the value function, by regularizing the Q-values
during training. Our theoretical analysis of CQL shows that only the expected value of this Q-function
under the policy lower-bounds the true policy value, preventing extra under-estimation that can arise
with point-wise lower-bounded Q-functions
* <img width="1086" height="470" alt="image" src="https://github.com/user-attachments/assets/cd1accef-b9c2-4733-8437-215b038ea2af" />

