# Delayed Q-learning
* It’s a model-free, off-policy reinforcement learning algorithm designed to have probably approximately correct (PAC) guarantees.
* Unlike standard Q-learning, it doesn’t update after every single step. Instead, it delays updates until it has enough evidence that a new estimate is significantly better than the old one.
* This makes it more sample-efficient and statistically robust, avoiding noisy updates.

# Opposition Learning
* Opposition learning is the idea of simultaneously considering a guess and its “opposite” in the search space, to accelerate convergence toward the optimal solution
* <img width="622" height="559" alt="image" src="https://github.com/user-attachments/assets/e1a2dddf-b9e9-4a00-bc22-56c2556f0968" />

# N-step Algorithm
<img width="656" height="591" alt="image" src="https://github.com/user-attachments/assets/222919d1-36bb-412b-85ed-d8b12428181f" />
<img width="716" height="836" alt="image" src="https://github.com/user-attachments/assets/03846758-9de1-4062-bc9b-d5310a8bad0d" />
<img width="656" height="62" alt="image" src="https://github.com/user-attachments/assets/17bd58b6-80ae-4384-b3ad-23fc0ac376d6" />

# Eligibility Traces
## Watkins's Q(lambda)
## Accumulating vs Replacing Eligibility Traces
