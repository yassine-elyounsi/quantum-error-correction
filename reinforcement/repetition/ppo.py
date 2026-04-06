# train_ppo.py
import wandb
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback

from src.environments.repetition_env import RepetitionCodeEnv

# ── 1. Initialize W&B ──────────────────────────────────────
wandb.init(
    project="quantum_rl",
    name="ppo_repetition_5percent_noise",
    config={
        "algorithm": "PPO",
        "env": "RepetitionCodeEnv",
        "noise_rate": 0.05,
        "n_steps": 256,
        "learning_rate": 3e-4,
        "batch_size": 64,
    }
)

# ── 2. Define vectorized environment ───────────────────────
env = make_vec_env(lambda: RepetitionCodeEnv(noise_rate=0.05), n_envs=8)

# ── 3. Callback to log score/accuracy to W&B ───────────────
class WandbLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_count = 0

    def _on_step(self) -> bool:
        # SB3 gives 'infos' for each environment
        infos = self.locals.get("infos", [])
        for info in infos:
            reward = info.get("reward", None)
            if reward is not None:
                self.episode_rewards.append(reward)
                self.episode_count += 1

                # Log every 100 episodes
                if self.episode_count % 100 == 0:
                    mean_reward = np.mean(self.episode_rewards[-100:])
                    accuracy = np.mean([r>0 for r in self.episode_rewards[-100:]])
                    wandb.log({
                        "mean_reward_100ep": mean_reward,
                        "accuracy_100ep": accuracy,
                        "episode_count": self.episode_count
                    })
        return True

callback = WandbLoggingCallback()

# ── 4. Initialize PPO ───────────────────────────────────────
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=256,
    batch_size=64,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    verbose=1,
)

# ── 5. Train ───────────────────────────────────────────────
model.learn(
    total_timesteps=50_000, 
    callback=callback
)

# ── 6. Save model ─────────────────────────────────────────
model.save("ppo_repetition_env")

# ── 7. Evaluate trained agent ──────────────────────────────
env_test = RepetitionCodeEnv(noise_rate=0.05)
episodes = 5000
correct = 0

for _ in range(episodes):
    obs, _ = env_test.reset()
    action, _ = model.predict(obs, deterministic=True)
    _, reward, _, _, _ = env_test.step(action)
    if reward > 0:
        correct += 1

accuracy = correct / episodes
print(f"PPO Agent Accuracy over {episodes} episodes: {accuracy:.2%}")

# Log final accuracy to W&B
wandb.log({"final_accuracy": accuracy})
wandb.finish()