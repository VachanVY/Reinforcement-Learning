import optuna
import numpy as np
import matplotlib.pyplot as plt
import json
import logging

from ..ppo_discrete import PPO, PPOConfig

logging.basicConfig(
    filename='ppo_discrete_lunar_lander.log',   # File to write logs to
    filemode='w',          # 'w' for overwrite, 'a' for append
    level=logging.DEBUG,   # Log level
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def objective(trial: optuna.Trial) -> float:
    try:
        # Suggest hyperparameters with type hints
        lr_actor: float = trial.suggest_float('lr', 1e-4, 1e-3, log=True)
        lr_critic: float = trial.suggest_float('lr', 1e-4, 1e-3, log=True)
        batch_size: int = trial.suggest_categorical('batch_size', [32, 64, 128])
        update_timestep: int = trial.suggest_int('update_timesteps', 200, 800, step=100)
        K = trial.suggest_int('K', 1, 100, step=10)
        target_kl = trial.suggest_float('target_kl', 0.01, 0.1, log=True)

        xonfig = PPOConfig()
        
        # Modify your config
        xonfig.lr_actor = lr_actor
        xonfig.lr_critic = lr_critic
        xonfig.batch_size = batch_size
        xonfig.update_timestep = update_timestep
        xonfig.K = K
        xonfig.target_kl = target_kl
        xonfig.num_episodes = 3000

        print(str(xonfig))
        
        # Train and evaluate
        ppo_agent = PPO("LunarLander-v3", xonfig)
        sum_rewards_list, _, _ = ppo_agent.train()

        plt.plot(sum_rewards_list)
        plt.title("Training Rewards")
        plt.xlabel("Episode")
        plt.ylabel("Sum of Rewards")
        plt.savefig(f"ppo_discrete_lunar_lander_{trial.number}.png")
        plt.close()
        with open(f"ppo_discrete_lunar_lander_trial_{trial.number}_config.json", "w") as f:
            json.dump(xonfig.__dict__, f)
    except:
        
        print(str(xonfig))
    avg_sum_rewards = float(np.mean(sum_rewards_list))
    print(str(avg_sum_rewards))
    return avg_sum_rewards # Explicit float return


if __name__ == "__main__":
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20, n_jobs=1)
