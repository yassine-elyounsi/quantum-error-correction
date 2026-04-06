# train_dqn.py
import wandb
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback

from src.environments.repetition_env import RepetitionCodeEnv

# ── 1. Initialize W&B ──────────────────────────────────────
wandb.init(
    project="quantum_rl",
    name="dqn_repetition_5percent_noise",
    config={
        "algorithm": "DQN",
        "env": "RepetitionCodeEnv",
        "noise_rate": 0.05,
        "learning_rate": 3e-4,
        "batch_size": 64,
        "buffer_size": 10000,
        "target_update_interval": 500,
        "train_freq": 4,
    }
)

# ── 2. Environment (DQN works better with non-vectorized envs) ─
env = RepetitionCodeEnv(noise_rate=0.05)

# ── 3. Callback to log score/accuracy to W&B ───────────────
class WandbLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_count = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            reward = info.get("reward", None)
            if reward is not None:
                self.episode_rewards.append(reward)
                self.episode_count += 1

                # Log every 100 episodes
                if self.episode_count % 100 == 0:
                    mean_reward = np.mean(self.episode_rewards[-100:])
                    accuracy = np.mean([r > 0 for r in self.episode_rewards[-100:]])
                    wandb.log({
                        "mean_reward_100ep": mean_reward,
                        "accuracy_100ep": accuracy,
                        "episode_count": self.episode_count
                    })
        return True

callback = WandbLoggingCallback()

# ── 4. Initialize DQN ───────────────────────────────────────
model = DQN(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    buffer_size=10_000,
    learning_starts=1000,
    batch_size=64,
    tau=1.0,
    gamma=0.99,
    target_update_interval=500,
    train_freq=4,
    exploration_fraction=0.1,
    exploration_final_eps=0.05,
    verbose=1
)

# ── 5. Train ───────────────────────────────────────────────
model.learn(total_timesteps=70_000, callback=callback)

# ── 6. Save model ─────────────────────────────────────────
model.save("dqn_repetition_env")

# ── 7. Evaluate trained agent ──────────────────────────────
episodes = 5000
correct = 0

for _ in range(episodes):
    obs, _ = env.reset()
    action, _ = model.predict(obs, deterministic=True)
    _, reward, _, _, _ = env.step(action)
    if reward > 0:
        correct += 1

accuracy = correct / episodes
print(f"DQN Agent Accuracy over {episodes} episodes: {accuracy:.2%}")

# Log final accuracy to W&B
wandb.log({"final_accuracy": accuracy})
wandb.finish()