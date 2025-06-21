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

## Monte Carlo Methods
* ![image](https://github.com/user-attachments/assets/7e36863b-6f06-4cd9-94c3-acd09d5815c2)
* ![image](https://github.com/user-attachments/assets/113bd28a-9eea-451e-968c-c8e6d5eaeff4)
* Monte Carlo Estimation of Action Value: If a model is not available, then it is particularly useful to estimate action values (the values of state–action pairs) rather than state values
* For policy evaluation to work for action
values, we must assure continual exploration. One way to do this is by specifying that
the episodes start in a state–action pair, and that every pair has a nonzero probability of
being selected as the start. This guarantees that all state–action pairs will be visited an
infinite number of times in the limit of an infinite number of episodes. We call this the
assumption of exploring starts. Keep non-zero probs policy for good exploration
### On-Policy
* ![image](https://github.com/user-attachments/assets/d5d395a0-1b3e-4342-9d72-771b9f11eb4a)
* ![image](https://github.com/user-attachments/assets/84bb5af0-af1c-4c6e-99c8-3b55d725c7f3)
## Off-Policy via Importance sampling
* ![image](https://github.com/user-attachments/assets/b7711099-aad3-48fe-b043-81441fb5ed07)
* ![image](https://github.com/user-attachments/assets/870d839c-bab7-418c-a913-c377a0c99d2f)
* ![WhatsApp Image 2025-06-16 at 09 04 28_31f5895e](https://github.com/user-attachments/assets/effa2b11-426a-41c1-a89a-5e8f954817db)
* ![WhatsApp Image 2025-06-16 at 09 05 00_00a84827](https://github.com/user-attachments/assets/c29a875e-85b1-4cd0-ae67-4d4d269666ee)
* ![image](https://github.com/user-attachments/assets/92879797-52b2-4425-a389-38f0cdde28d0)
* ![image](https://github.com/user-attachments/assets/f4c3fbd4-f7f6-4c9b-b8b2-96836c7406fb)

## Temporal-Difference Learning
* ![image](https://github.com/user-attachments/assets/1dc6b572-cbda-4c7d-b507-fa1a4d1c4534)
* ![image](https://github.com/user-attachments/assets/4a7596aa-7316-40cc-84f8-743d3808ba32)
* ![image](https://github.com/user-attachments/assets/82aeb3ca-a69a-4e5c-8c16-d125a47d5f67)
* ![image](https://github.com/user-attachments/assets/c9fdb18f-6b2f-4253-b30c-8a59984f5543)
* ![image](https://github.com/user-attachments/assets/5525a554-e503-4b93-b636-b24c47bbe874)
* ![image](https://github.com/user-attachments/assets/325b8c8f-e2f7-4d4f-a127-682a558cdb4d)
* ![image](https://github.com/user-attachments/assets/d0efbabe-13e7-4aa8-8b1a-8f3cfb0ebf2c)
* ![image](https://github.com/user-attachments/assets/ee107f91-07e1-4a64-92a9-717de451672e)
```python
  def double_q_learning_update(
    alpha:float, gamma:float, reward:float, 
    next_state_qvals_1:list[float], next_state_qvals_2:list[float], 
    qvals_of_state_1:list[float], qvals_of_state_2:list[float],
    action:int, **kwargs
):

    # >> Why Double Q-Learning? To avoid maximization bias
    # >>> consider a single state s where there are many actions a whose true values, q(s, a),
    # are all zero but whose estimated values, Q(s, a), are uncertain and thus distributed 
    # some above and some below zero. The maximum of the true values is zero, but the maximum
    # of the estimates is positive, a positive bias. We call this maximization bias.
    if random.random() < 0.5:
        best_next_action = next_state_qvals_1.index(max(next_state_qvals_1)) # take action from Q1 but take Q value estimate from Q2 <= for Q1 update
        qvals_of_state_1[action] += alpha * (reward + gamma * next_state_qvals_2[best_next_action] - qvals_of_state_1[action])
    else:
        best_next_action = next_state_qvals_2.index(max(next_state_qvals_2)) # take action from Q2 but take Q value estimate from Q1 <= for Q2 update
        qvals_of_state_2[action] += alpha * (reward + gamma * next_state_qvals_1[best_next_action] - qvals_of_state_2[action])
  ```
* While sampling actions for Double Q-learning, remember to sample it from Q1 + Q2 function

## Policy Gradient Methods
<!--
* ![image](https://github.com/user-attachments/assets/878f53ae-ffc6-42c2-bd76-9c0bb23b7810)
* ![image](https://github.com/user-attachments/assets/31dcdbeb-628c-4573-8882-88058d0cff12)
* ![image](https://github.com/user-attachments/assets/6796b6c8-5ca8-4307-bb48-4a6fd0e61c2b)
-->
* ![WhatsApp Image 2025-06-17 at 13 44 46_5f41ba88](https://github.com/user-attachments/assets/e8acb350-be46-43f1-8851-40617c1d7299)
* ![WhatsApp Image 2025-06-17 at 13 45 19_e75d0b84](https://github.com/user-attachments/assets/f6a6bc29-9c45-401f-a602-93fa728ca6ad)
* ![WhatsApp Image 2025-06-17 at 13 45 37_e764d135](https://github.com/user-attachments/assets/2cd1e53b-e51f-4c80-ade1-90d34f52e8de)
* ![WhatsApp Image 2025-06-17 at 13 45 58_e65623ab](https://github.com/user-attachments/assets/4c2231b1-2992-47ea-8e0b-94cba086a9d5)
* ![image](https://github.com/user-attachments/assets/d4514450-ee68-4659-a665-1a7acf626043)
* ![image](https://github.com/user-attachments/assets/d0e8fd80-297b-41bc-a981-c1e67725a54a)
* ![image](https://github.com/user-attachments/assets/411ad142-8147-49d5-91f0-b1b1a86b6475)
* ![image](https://github.com/user-attachments/assets/4f66b085-a334-4ac0-9ff3-64335a59966b)
  Although the REINFORCE-with-baseline method learns both a policy and a state-value
function, we do not consider it to be an actor–critic method because its state-value function
is used only as a baseline, not as a critic. That is, it is not used for bootstrapping (updating
the value estimate for a state from the estimated values of subsequent states), but only
as a baseline for the state whose estimate is being updated. This is a useful distinction,
for only through bootstrapping do we introduce bias and an asymptotic dependence
on the quality of the function approximation.
* ![image](https://github.com/user-attachments/assets/3af07b8b-ed34-466f-a0ff-a073f39e0686)
* ![image](https://github.com/user-attachments/assets/88880cab-29f0-4b00-bd44-71e99f3f2d23)
* ![image](https://github.com/user-attachments/assets/02bba5b0-e081-43a4-8c3f-d5af5ac31186)

# Deep Reinforcement Learning
## Proximal Policy Optimization (PPO)
In Reinforcement Learning, the term is want to maximize is:

![image](https://github.com/user-attachments/assets/ce821355-ce73-4873-bc49-c732d382df8e)

PPO Loss explanation:

$$L^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[ \min \left( r_t(\theta) \hat{A}_t,\ \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_t \right) \right]$$

Where:

$$r_t = \frac{\pi_{\theta}(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$$



If **$r_t > 0$**: $\pi_\theta$ is more confident about action $a_t$ than $\pi_{\text{old}}$.
We want this confidence to increase — but not so much that it shoots off.

If **$r_t < 0$**: $\pi_\theta$ is more confident about action $a_t$ than $\pi_{\text{old}}$.
We want this confidence to decrease — not so much that it shoots off.


Expanded form:

$$
L^{\text{CLIP}}(\theta) =
\begin{cases}
r_t * \hat{A}_t & \text{if } r_t < 1 - \epsilon \\
r_t * \hat{A}_t & \text{if } 1 - \epsilon \le r_t \le 1 + \epsilon \\
\text{clip}(r_t, 1 - \epsilon, 1 + \epsilon) * \hat{A}_t & \text{if } r_t > 1 + \epsilon
\end{cases}
\quad \text{for } \hat{A}_t > 0
$$

$$
L^{\text{CLIP}}(\theta) =
\begin{cases}
\text{clip}(r_t, 1 - \epsilon, 1 + \epsilon) * \hat{A}_t & \text{if } r_t < 1 - \epsilon \\
r_t * \hat{A}_t & \text{if } 1 - \epsilon \le r_t \le 1 + \epsilon \\
r_t * \hat{A}_t & \text{if } r_t > 1 + \epsilon
\end{cases}
\quad \text{for } \hat{A}_t < 0
$$

---

**When $\hat{A}_t > 0$:**  
Action is selected from $\pi_{\text{old}}$. **$\hat{A}_t$** validates that the action is good.  
We want to incorporate that more into $\pi_\theta$, but not let it rise too much — so we clip the **upper bound**.

**When $\hat{A}_t < 0$:**  
Action is selected from $\pi_{\text{old}}$. **$\hat{A}_t$** tells us that the action is not good.  
We want to incorporate that into $\pi_\theta$, but not let it decrease too much — so we clip the **lower bound**.

```python
def ppo_update(total_steps_done:int):
    buf_size = len(buffer.rewards)

    buf_states = torch.stack(buffer.states).to(xonfig.device).detach() # (B, state_dim)
    buf_actions = torch.stack(buffer.actions).to(xonfig.device).detach() # (B, num_actions)

    # buf_returns = torch.tensor(
    #     get_discounted_returns(buffer.rewards, buffer.dones, xonfig.gamma),
    #     device=xonfig.device
    # ).unsqueeze(-1).detach()
    # if buf_size > 1:
    #     buf_returns = ((buf_returns - buf_returns.mean()) / (buf_returns.std() + 1e-6)).detach()
    
    buf_action_logproba = torch.stack(buffer.log_probs).to(xonfig.device).detach() # (B, 1)

    advantages = torch.tensor(
        calculate_advantages_gae(
            rewards_list=buffer.rewards,
            state_values_list=buffer.state_values,
            gamma=xonfig.gamma,
            trace_decay=xonfig.trace_decay
        ), device=xonfig.device
    ).unsqueeze(-1).detach() # (B, 1)

    buf_returns = (advantages + torch.as_tensor(buffer.state_values, device=xonfig.device).unsqueeze(-1)).detach() 

    if buf_size > 1:
        advantages = ((advantages - advantages.mean()) / (advantages.std() + 1e-6)).detach() # (B, 1)

    # K Epochs
    losses = {"policy": [], "value": []}
    kldivs_list = []; norm:tp.Optional[Tensor] = None
    for _ in range(xonfig.K):
        for start_idx in range(0, buf_size - xonfig.batch_size + 1, xonfig.batch_size):
            batch_idx = slice(start_idx, min(start_idx + xonfig.batch_size, buf_size))
            batch_returns = buf_returns[batch_idx] # (B, 1)
            batch_advantages = advantages[batch_idx] # (B, 1)
            batch_states = buf_states[batch_idx] # (B, state_dim)
            batch_actions = buf_actions[batch_idx] # (B, action_dim)
            batch_action_logprobs = buf_action_logproba[batch_idx] # (B, 1)

            # Compute advantage
            with autocast:  # Ensure all forward passes and loss computations are inside this block
                mean:Tensor; std:Tensor; state_value:Tensor
                (mean, std), state_value = actor_critic(batch_states) # ((B, num_actions), (B, num_actions)), (B, 1)

                # compute new action logprobas
                covariance_matrix = torch.diag_embed(std.pow(2)) # (B, num_actions, num_actions)
                dist = torch.distributions.MultivariateNormal(
                    loc=mean, covariance_matrix=covariance_matrix
                )
                new_action_logprobas:Tensor = dist.log_prob(batch_actions).unsqueeze(-1) # (B, 1)

                # value loss
                value_loss = nn.functional.mse_loss(state_value, batch_returns)

                # policy loss
                log_ratios = new_action_logprobas - batch_action_logprobs # (B, 1)
                r = log_ratios.exp()
                unclipped_obj = r * batch_advantages
                clipped_obj = r.clip(1-xonfig.clip_range, 1+xonfig.clip_range) * batch_advantages
                policy_loss = -torch.min(unclipped_obj, clipped_obj).mean()

                # KL divergence
                with torch.no_grad():
                    # https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/ppo/ppo.py#L262-L265
                    # http://joschu.net/blog/kl-approx.html
                    log_ratios = log_ratios.detach()
                    approx_kl_div = ((log_ratios.exp() - 1) - log_ratios).mean().cpu().item()
                    kldivs_list.append(approx_kl_div)

                if xonfig.target_kl is not None and approx_kl_div > xonfig.target_kl * 1.5:
                    break

                entropy_loss:Tensor = -dist.entropy().mean()

            (policy_loss + xonfig.val_coeff * value_loss + xonfig.entropy_coeff * entropy_loss).backward()

            if xonfig.clip_norm > 0:
                norm = nn.utils.clip_grad_norm_(
                    actor_critic.parameters(), xonfig.clip_norm
                )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            
            losses["policy"].append(policy_loss.cpu().item())
            losses["value"].append(value_loss.cpu().item())
            
    if xonfig.logg_tb:
        if norm is not None:
            writer.add_scalar("losses/grad_norm", norm.cpu().item(), total_steps_done)
    
    buffer.clear()
    actor_critic_old.load_state_dict(actor_critic.state_dict())
    avg = lambda x: sum(x)/max(len(x), 1)
    return avg(losses["policy"]), avg(losses["value"]), avg(kldivs_list)
```

## Deep Deterministic Policy Gradient (DDPG)
![WhatsApp Image 2025-06-20 at 23 27 53_46a142bb](https://github.com/user-attachments/assets/e1f0c4a0-c0aa-4ddb-82c1-afdd15a1b8b0)
```python
def ddpg_train_step(
    states:Tensor,
    actions:Tensor,
    next_states:Tensor,
    rewards:Tensor,
    is_terminal:Tensor
):
    """
    * `states`: `(n, state_dim)`
    * `actions`: `(n, action_dim)`
    * `next_states`: `(n, state_dim)`
    * `rewards`: `(n,)`
    * `is_terminal`: `(n,)`
    """
    rewards, is_terminal = rewards.unsqueeze(-1), is_terminal.unsqueeze(-1) # (n,) -> (n, 1)

    # Optimize DQN/Critic
    with torch.no_grad(): # anyway models in this block are not trainable
        q_next_state = dqn_ema_net(next_states, actor_ema_net(next_states)) # (n, 1)
        q_target = rewards + xonfig.gamma * q_next_state * (1 - is_terminal) # (n, 1)
    q_pred = dqn_net(states, actions) # (n, 1)
    qloss = nn.functional.mse_loss(q_pred, q_target, reduction="sum") # (,)
    qloss.backward()
    dqn_optimizer.step()
    dqn_optimizer.zero_grad()
    
    # Optimize Actor
    dqn_net.requires_grad_(False)
    ## Assuming that the critic Q is a trained model, 
    ## if Q(s, a) is high, then the action a is good, else bad if Q(s, a) is low action a is bad.
    ## we want the actor_net to tweak it's actions such that the Q(s, actor_net(s)) is high (Q is freezed, so Q won't tweak it's weights to make Q(s, actor_net(s)) high)
    ## so we want to maximize Q(s, actor_net(s)) -> minimize -Q(s, actor_net(s))
    actor_loss:Tensor = -dqn_net(states, actor_net(states)).sum()
    actor_loss.backward()
    actor_optimizer.step()
    actor_optimizer.zero_grad()
    dqn_net.requires_grad_(True)
```

## Soft Actor Critic (SAC)
* ![image](https://github.com/user-attachments/assets/b2cd1d8f-7a90-4c32-b0da-23657b9ff106)
* ![WhatsApp Image 2025-06-21 at 12 27 40_8afb54b5](https://github.com/user-attachments/assets/8946218f-9875-4244-abb0-69c882d54e2b)
* ![image](https://github.com/user-attachments/assets/fac3a628-52dc-442a-b276-cfedca96b332)
* ![WhatsApp Image 2025-06-21 at 12 34 34_fb3a259f](https://github.com/user-attachments/assets/a6f32302-ba55-4eb3-a45b-ff46d23945ad)
* ![WhatsApp Image 2025-06-21 at 12 36 31_5196858a](https://github.com/user-attachments/assets/b484d993-9780-4d9e-8467-7d0f78afa6a2)
* ![WhatsApp Image 2025-06-21 at 17 12 00_22e42f24](https://github.com/user-attachments/assets/a888279f-3fd5-4bb6-b57f-e9351f4b70fe)
* ![WhatsApp Image 2025-06-21 at 17 14 19_3e0ed50d](https://github.com/user-attachments/assets/dd8fef5f-9276-47fd-89c0-934476b1d2ea)
* ![WhatsApp Image 2025-06-21 at 17 14 44_5e370b18](https://github.com/user-attachments/assets/6dd63870-7d99-4304-ae3f-5df82d0ac101)

```python
@torch.compile()
def sac_train_step(
    states:Tensor,
    actions:Tensor,
    next_states:Tensor,
    rewards:Tensor,
    is_terminal:Tensor
):
    """
    * `states`: `(B, state_dim)`
    * `actions`: `(B, action_dim)`
    * `next_states`: `(B, state_dim)`
    * `rewards`: `(B,)`
    * `is_terminal`: `(B,)`
    """
    rewards, is_terminal = rewards.unsqueeze(-1), is_terminal.unsqueeze(-1) # (B,) => (B, 1)    

    # Optimize DQNs
    ## a_next ~ π(s_next)
    ## get target Q values: y = r + γ * ( Q_target(s_next, a_next) - α * log(π(a_next|s_next)) ) * (1 - is_terminal)
    ## L1 = MSE(Q1(s, a), y) ## L2 = MSE(Q2(s, a), y) ## optimize loss (L1, L2)
    with torch.no_grad():
        actions_next, log_prob = sample_actions(next_states, ACTION_BOUNDS)
        q_next1, q_next2 = dqn_target1(next_states, actions_next), dqn_target2(next_states, actions_next) # (B, 1), (B, 1)
        # why min of the two q values? To avoid maximization bias, see https://arxiv.org/abs/1812.05905
        q_next:Tensor = torch.min(q_next1, q_next2) - xonfig.alpha * log_prob # (B, 1)
        q_next_target:Tensor = rewards + xonfig.gamma * q_next * (1 - is_terminal) # (B, 1)
    
    dqn1_loss = nn.functional.mse_loss(dqn1(states, actions), q_next_target, reduction="mean")
    dqn2_loss = nn.functional.mse_loss(dqn2(states, actions), q_next_target, reduction="mean")
    (dqn1_loss + dqn2_loss).backward() # dqn1_loss.backward(); dqn2_loss.backward()
    dqn1_optimizer.step(); dqn2_optimizer.step()
    dqn1_optimizer.zero_grad(); dqn2_optimizer.zero_grad()

    # Optimize Policy
    dqn1.requires_grad_(False); dqn2.requires_grad_(False)
    actions, log_probs = sample_actions(states, ACTION_BOUNDS)
    ## maximize entropy, minimize negative entropy
    ## maximize q value by minimizing -q value, tweaks the policy weights through the actions to maximize q value, doesn't tweak the q network itself as they are freezed
    pi_loss:Tensor = (xonfig.alpha * log_probs - torch.min(dqn1(states, actions), dqn2(states, actions))).mean()
    pi_loss.backward()
    policy_optimizer.step()
    policy_optimizer.zero_grad()
    dqn1.requires_grad_(True); dqn2.requires_grad_(True)

    # Optimize Alpha
    if xonfig.adaptive_alpha:
        alpha_loss = -log_alpha * (log_prob + target_entropy).mean()
        alpha_loss.backward()
        alpha_optimizer.step()
        alpha_optimizer.zero_grad()
        xonfig.alpha = log_alpha.exp().item()
```
