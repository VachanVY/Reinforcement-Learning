# Reinforcement Learning: an Introduction

## Introduction
* 4 main elements:
  * Policy: maps state to action
  * Reward Signal: Goal of a Reinforcement Learning Problem, tells what is good in an immediate sense. Agent's sole objective is to maximize the total reward it recieves over the long run
  * Value Function (State Value):
     * Tells how good that state is in the long run i.e the total amount of reward an agent can expect to accumulate in the future, starting from that state
     * Action choices are based on value judgements, we seek states which bring highest value, not reward
  * Model of the environment (Optional): Mimics the environment, given state and next action, tries to predicts the next state. They are used for planning which we will see later.
    Methods which use models and planning are called model-based methods, else model-free methods

## Multi-Armed Bandits
* To roughly assess the relative effectiveness of the greedy and eps-greedy (select random action with probability eps) action-value methods, we compared them numerically on a suite of test problems
* ![image](https://github.com/user-attachments/assets/13df279c-e336-4e78-a3eb-5189cc43dd51)
* The greedy method improved slightly faster than the
other methods at the very beginning, but then leveled of at a lower level
* ![image](https://github.com/user-attachments/assets/78b015be-aed9-412a-8a3a-5665cc7a7ca5)
* The greedy method performed significantly worse in the long run because it, often got stuck performing suboptimal actions

### Incremental Implementation
* ![image](https://github.com/user-attachments/assets/b3e8824c-8180-4f5c-b0c9-83af210798d6)
* ![image](https://github.com/user-attachments/assets/94a5cd90-03b9-41f0-9ec4-e593ae7c6def) 1/n can be replaced with any step-size alpha

### Optimistic Initial Values
* Initial action values can also be used as a simple way to encourage exploration
* Whichever actions are initially selected, the reward is less than the starting estimates; the learner switches to
other actions, being “disappointed” with the rewards it is receiving. The result is that all
actions are tried several times before the value estimates converge
![image](https://github.com/user-attachments/assets/2e4af548-e2ac-42f4-9916-c966b42082ba)

### Upper-Confidence-Bound Action Selection
* ![image](https://github.com/user-attachments/assets/351a2b04-b532-493a-abba-b02f4929aca5)
* Nt(a) denotes the number of times that action a has
been selected prior to time t
* Each time a is selected the uncertainty is presumably
reduced: Nt(a) increments, and, as it appears in the denominator, the uncertainty term
decreases. On the other hand, each time an action other than a is selected, t increases but
Nt(a) does not

## Finite Markov Decision Processes
* ![image](https://github.com/user-attachments/assets/d5bf0c05-8fc2-4197-9836-faae6c5cff09)
* ![image](https://github.com/user-attachments/assets/8ceeb2ab-0af5-4fde-a2df-0d07a274c9cd)
* ![image](https://github.com/user-attachments/assets/61d37bc1-7404-4b71-a514-efc92d38b869)
  Also reduces the variance of the state value which is the estimate of the return
* ![image](https://github.com/user-attachments/assets/fbceef0a-8cf8-4ed7-80c3-b477b6aabb38)
* ![image](https://github.com/user-attachments/assets/b6e49429-b75b-4ad7-b553-078219a66a40)
* ![image](https://github.com/user-attachments/assets/0e668e8e-2190-44cd-b22b-d23155f07e55)
  Bellman equation
* ![image](https://github.com/user-attachments/assets/e6ac7732-cad8-4bd3-9166-347612c33521)
* ![image](https://github.com/user-attachments/assets/4af9a41b-b6e4-4658-a4a6-cd34a9121727)
* ![image](https://github.com/user-attachments/assets/27a913a4-f9cb-4d9c-93a8-0ed07b9c46c1)
* ![image](https://github.com/user-attachments/assets/d731311e-fe57-469a-a561-a66ea1fafff1)

## Dynamic Programming
* Policy Iteration:
  ![image](https://github.com/user-attachments/assets/4d22fb89-f625-49cb-ad1d-e885ffec098b)
  * Policy Evaluation: Improve the accuracy of State-Values
    * Keep running until the maximum change in state values for all states is less than the threshold
    * **Actions are sampled by the policy itself for the iteration, which might not be the best action, so it takes time and iterates many times**
  * Policy Iteration: Improve policy by 'argmaxing'
    * ![image](https://github.com/user-attachments/assets/724f9a12-e801-43d6-9301-df755b02be2a)
    * So far we have seen how, given a policy and its value function, we can easily evaluate a change in the policy at a single state to a particular action. It is a natural extension to consider changes at all states and to all possible actions, selecting at each state the action that appears best according to q_pi(s, a).
     ![image](https://github.com/user-attachments/assets/75000134-3b08-4c35-80d6-99aa1098874d)
* Value Iteration:
  ![image](https://github.com/user-attachments/assets/c20cb8f6-3116-4f0f-95d8-b8a11cf35650)
  * The policy evaluation step of policy iteration can be truncated in several ways without losing the convergence guarantees of policy iteration. One important special case is when policy evaluation is stopped after just one sweep (one update of each state)
  * ![image](https://github.com/user-attachments/assets/ca50eb74-dad5-437a-b92f-4baca6ddb013)
    **Here actions are not sampled by the policy in the policy evaluation step, instead we take the largest state-value given all possible actions from that state**
* A major drawback to the DP methods that we have discussed so far is that they involve
operations over the entire state set of the MDP, that is, they require sweeps of the state
set. If the state set is very large, then even a single sweep can be prohibitively expensive.
For example, the game of backgammon has over 10^20 states. Even if we could perform
the value iteration update on a million states per second, it would take over a thousand
years to complete a single sweep.
* Asynchronous Dynamic Programming: Asynchronous DP algorithms are in-place iterative DP algorithms that are not organized
in terms of systematic sweeps of the state set. These algorithms update the values of
states in any order whatsoever, using whatever values of other states happen to be
available. The values of some states may be updated several times before the values of
others are updated once. To converge correctly, however, an asynchronous algorithm
must continue to update the values of all the states: it can’t ignore any state after some
point in the computation. Asynchronous DP algorithms allow great flexibility in selecting
states to update.
