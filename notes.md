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
