# =============================================================================
# train_ppo_shor_wandb.py
# PPO training with Weights & Biass logging
# =============================================================================

import numpy as np
import wandb

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

from src.environments.shor_env import ShorCodeEnv


# =============================================================================
# CONFIG
# =============================================================================

TOTAL_TIMESTEPS = 200_000
NOISE_RATE = 0.05
PROJECT_NAME = "shor-ppo"


# =============================================================================
# INIT WANDB
# =============================================================================

run = wandb.init(
    project=PROJECT_NAME,
    config={
        "algo": "PPO",
        "env": "ShorCodeEnv",
        "noise_rate": NOISE_RATE,
        "timesteps": TOTAL_TIMESTEPS,
        "policy": "MlpPolicy",
    },
    sync_tensorboard=True,  # optional if using TB
    monitor_gym=True,
    save_code=True,
)


# =============================================================================
# CUSTOM CALLBACK (LOG METRICS)
# =============================================================================

class WandbCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_correct = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])

        for info in infos:
            if "reward" in info:
                reward = info["reward"]
                self.episode_rewards.append(reward)

                correct = 1 if reward > 0 else 0
                self.episode_correct.append(correct)

        # Log every 100 steps
        if len(self.episode_rewards) >= 100:
            wandb.log({
                "mean_reward": np.mean(self.episode_rewards),
                "accuracy": np.mean(self.episode_correct),
            })

            self.episode_rewards.clear()
            self.episode_correct.clear()

        return True


# =============================================================================
# ENV
# =============================================================================

env = Monitor(ShorCodeEnv(noise_rate=NOISE_RATE))


# =============================================================================
# MODEL
# =============================================================================

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    gamma=0.99,
    batch_size=256,
    n_steps=2048,
    ent_coef=0.01,
)


# =============================================================================
# TRAIN
# =============================================================================

print("\n🚀 Training PPO with wandb...")
model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=WandbCallback()
)


# =============================================================================
# SAVE MODEL
# =============================================================================

model.save("ppo_shor_wandb")
wandb.save("ppo_shor_wandb.zip")

print("✅ Model saved")


# =============================================================================
# FINAL EVAL
# =============================================================================

eval_env = ShorCodeEnv(noise_rate=NOISE_RATE)

correct = 0
rewards = []

for _ in range(5000):
    obs, _ = eval_env.reset()
    action, _ = model.predict(obs, deterministic=True)
    _, reward, _, _, _ = eval_env.step(action)

    rewards.append(reward)
    if reward > 0:
        correct += 1

accuracy = correct / 5000
mean_reward = np.mean(rewards)

wandb.log({
    "final_accuracy": accuracy,
    "final_mean_reward": mean_reward,
})

print("\n📊 Final Results")
print(f"Accuracy     : {accuracy:.4f}")
print(f"Mean reward  : {mean_reward:.4f}")


# =============================================================================
# FINISH
# =============================================================================

wandb.finish()