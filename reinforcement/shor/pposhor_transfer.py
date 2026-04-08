# =============================================================================
# transfer_ppo_rep_to_shor.py
#
# Transfer learning: RepetitionCode PPO → ShorCode PPO
#
# Fixed for SB3 versions that use policy_net / value_net
# instead of the older shared_net.
#
# Structure in your SB3 version (net_arch=[64,64]):
#   mlp_extractor.policy_net = Sequential(
#       Linear(obs → 64),   ← index 0  input layer
#       Tanh,
#       Linear(64 → 64),    ← index 2  brain layer  ← we transfer this
#       Tanh,
#   )
#   mlp_extractor.value_net  = Sequential(same structure)
#
# Transfer strategy:
#   - Copy policy_net brain layers (64→64) from RepCode → Shor
#   - Input layer (2→64 becomes 8→64) is rebuilt randomly
#   - Output heads are rebuilt randomly
#   - Phase 1: freeze brain, train input + output heads only
#   - Phase 2: unfreeze all, fine-tune at lower LR
# =============================================================================

import numpy as np
import torch.nn as nn
import wandb
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

from src.environments.shor_env import ShorCodeEnv


# =============================================================================
# CONFIG
# =============================================================================

REP_MODEL_PATH     = "ppo_repetition_env"
SCRATCH_MODEL_PATH = "ppo_shor_wandb"
TRANSFER_OUT_PATH  = "ppo_shor_transfer"

NOISE_RATE         = 0.05
PHASE1_TIMESTEPS   = 100_000
PHASE2_TIMESTEPS   = 100_000
EVAL_EPISODES      = 5_000

NET_ARCH           = [64, 64]
PHASE1_LR          = 3e-4
PHASE2_LR          = 1e-4


# =============================================================================
# WANDB CALLBACK
# =============================================================================

class WandbCallback(BaseCallback):
    def __init__(self, label, verbose=0):
        super().__init__(verbose)
        self.label   = label
        self.rewards = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.rewards.append(info["episode"]["r"])

        if len(self.rewards) >= 100:
            mean_r   = float(np.mean(self.rewards))
            accuracy = float(np.mean([1 if r > 0 else 0
                                      for r in self.rewards]))
            wandb.log({
                f"{self.label}/mean_reward": mean_r,
                f"{self.label}/accuracy":    accuracy,
                "global_step": self.num_timesteps,
            })
            self.rewards.clear()

        return True


# =============================================================================
# HELPERS — work on policy_net (your SB3 version has no shared_net)
# =============================================================================

def get_brain_layers(model):
    """
    Return the Linear layers AFTER the input layer from policy_net.
    These are the 64→64 layers we transfer.

    policy_net layout (net_arch=[64,64]):
        Sequential(
            [0] Linear(obs → 64)   ← input layer, skip this
            [1] Tanh
            [2] Linear(64 → 64)    ← brain layer, transfer this
            [3] Tanh
        )
    """
    policy_net  = model.policy.mlp_extractor.policy_net
    all_linears = [m for m in policy_net if isinstance(m, nn.Linear)]
    # Skip index 0 (input layer), return the rest
    return all_linears[1:]


def extract_hidden_weights(rep_model):
    """Extract 64→64 brain layer weights from the RepCode model."""
    brain_layers   = get_brain_layers(rep_model)
    hidden_weights = [
        {'weight': layer.weight.data.clone(),
         'bias':   layer.bias.data.clone()}
        for layer in brain_layers
    ]

    print(f'\n  Extracted {len(hidden_weights)} brain layer(s) from RepCode:')
    for i, w in enumerate(hidden_weights):
        print(f'    Brain layer {i}: {list(w["weight"].shape)}')

    return hidden_weights


def build_transfer_model(shor_env, hidden_weights):
    """
    Build fresh Shor PPO and inject RepCode brain weights into policy_net.
    Input layer (8→64) and all output heads stay randomly initialised.
    """
    model = PPO(
        "MlpPolicy",
        shor_env,
        verbose=0,
        learning_rate=PHASE1_LR,
        gamma=0.99,
        batch_size=256,
        n_steps=2048,
        ent_coef=0.01,
        policy_kwargs={"net_arch": NET_ARCH},
    )

    brain_layers = get_brain_layers(model)

    if len(brain_layers) != len(hidden_weights):
        raise ValueError(
            f'Layer mismatch: RepCode has {len(hidden_weights)} brain layers '
            f'but Shor has {len(brain_layers)}. '
            f'Check NET_ARCH={NET_ARCH} matches your RepCode training config.'
        )

    print('\n  Injecting brain weights into Shor policy_net:')
    for i, (layer, w) in enumerate(zip(brain_layers, hidden_weights)):
        layer.weight.data.copy_(w['weight'])
        layer.bias.data.copy_(w['bias'])
        print(f'    Brain layer {i}: {list(w["weight"].shape)}  ✓')

    return model


def freeze_brain(model):
    """Freeze brain layers so only input layer + output heads train."""
    for layer in get_brain_layers(model):
        for param in layer.parameters():
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.policy.parameters()
                    if p.requires_grad)
    frozen    = sum(p.numel() for p in model.policy.parameters()
                    if not p.requires_grad)
    print(f'\n  Phase 1 — brain FROZEN')
    print(f'    Trainable : {trainable:,}  |  Frozen : {frozen:,}')


def unfreeze_all(model):
    """Unfreeze everything for Phase 2."""
    for param in model.policy.parameters():
        param.requires_grad = True
    total = sum(p.numel() for p in model.policy.parameters())
    print(f'\n  Phase 2 — all params UNFROZEN ({total:,} total)')


def set_lr(model, lr):
    for pg in model.policy.optimizer.param_groups:
        pg['lr'] = lr


def evaluate(model, label, n_episodes=EVAL_EPISODES):
    env     = ShorCodeEnv(noise_rate=NOISE_RATE)
    correct = 0
    rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        action, _ = model.predict(obs, deterministic=True)
        _, reward, _, _, _ = env.step(int(action))
        rewards.append(reward)
        if reward > 0:
            correct += 1
    acc  = correct / n_episodes
    mean = float(np.mean(rewards))
    print(f'  [{label}]  accuracy={acc:.4f}  mean_reward={mean:+.4f}')
    return acc, mean


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':

    print('=' * 60)
    print('  TRANSFER LEARNING: RepCode PPO → Shor PPO')
    print('=' * 60)

    wandb.init(
        project="shor-ppo",
        name="transfer_rep_to_shor",
        config={
            "type":             "transfer",
            "source":           "repetition_code",
            "noise_rate":       NOISE_RATE,
            "phase1_steps":     PHASE1_TIMESTEPS,
            "phase2_steps":     PHASE2_TIMESTEPS,
            "net_arch":         NET_ARCH,
            "phase1_lr":        PHASE1_LR,
            "phase2_lr":        PHASE2_LR,
        },
    )

    # ── 1. Load RepCode model ─────────────────────────────────────────────────
    print(f'\n[1/5] Loading RepCode model: {REP_MODEL_PATH}')
    rep_model = PPO.load(REP_MODEL_PATH)

    print('  policy_net layers:')
    for name, param in rep_model.policy.mlp_extractor.policy_net.named_parameters():
        print(f'    {name:40s}  {list(param.shape)}')

    # ── 2. Extract brain weights ──────────────────────────────────────────────
    print('\n[2/5] Extracting brain weights...')
    hidden_weights = extract_hidden_weights(rep_model)

    # ── 3. Build transfer model ───────────────────────────────────────────────
    print('\n[3/5] Building Shor transfer model...')
    shor_env       = Monitor(ShorCodeEnv(noise_rate=NOISE_RATE))
    transfer_model = build_transfer_model(shor_env, hidden_weights)

    # ── 4. Phase 1: freeze brain, train heads only ────────────────────────────
    print(f'\n[4/5] Phase 1 — {PHASE1_TIMESTEPS:,} steps (heads only)...')
    freeze_brain(transfer_model)

    transfer_model.learn(
        total_timesteps=PHASE1_TIMESTEPS,
        callback=WandbCallback(label='transfer_phase1'),
        reset_num_timesteps=True,
    )

    acc1, rew1 = evaluate(transfer_model, 'Phase 1')
    wandb.log({"transfer_phase1/final_accuracy": acc1,
               "transfer_phase1/final_reward":   rew1})

    # ── 5. Phase 2: unfreeze, full fine-tune ─────────────────────────────────
    print(f'\n[5/5] Phase 2 — {PHASE2_TIMESTEPS:,} steps (full fine-tune)...')
    unfreeze_all(transfer_model)
    set_lr(transfer_model, PHASE2_LR)

    transfer_model.learn(
        total_timesteps=PHASE2_TIMESTEPS,
        callback=WandbCallback(label='transfer_phase2'),
        reset_num_timesteps=False,
    )

    acc2, rew2 = evaluate(transfer_model, 'Phase 2')
    wandb.log({"transfer_phase2/final_accuracy": acc2,
               "transfer_phase2/final_reward":   rew2})

    transfer_model.save(TRANSFER_OUT_PATH)
    print(f'\n  Transfer model saved → {TRANSFER_OUT_PATH}')

    # ── Load scratch model for comparison ─────────────────────────────────────
    print(f'\n  Loading scratch model: {SCRATCH_MODEL_PATH}')
    scratch_model = PPO.load(SCRATCH_MODEL_PATH)
    acc_s, rew_s  = evaluate(scratch_model, 'Scratch')

    # ── Summary ───────────────────────────────────────────────────────────────
    print('\n' + '=' * 60)
    print('  FINAL COMPARISON')
    print('=' * 60)
    print(f'  {"Model":<20}  {"Accuracy":>10}  {"Mean Reward":>12}')
    print(f'  {"─"*46}')
    print(f'  {"Scratch":<20}  {acc_s:>10.4f}  {rew_s:>+12.4f}')
    print(f'  {"Transfer Phase 1":<20}  {acc1:>10.4f}  {rew1:>+12.4f}')
    print(f'  {"Transfer Phase 2":<20}  {acc2:>10.4f}  {rew2:>+12.4f}')

    gain = acc2 - acc_s
    print(f'\n  Transfer gain: {gain:+.4f}  '
          f'{"✓ transfer helps" if gain > 0 else "✗ scratch wins"}')

    # ── Bar chart ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    labels  = ['Scratch', 'Transfer\nPhase 1', 'Transfer\nPhase 2']
    accs    = [acc_s, acc1, acc2]
    colors  = ['#E8602C', '#7BAFD4', '#378ADD']

    bars = ax.bar(labels, accs, color=colors, width=0.45, edgecolor='white')
    for bar, val in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                val + 0.01, f'{val:.3f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.axhline(1/28, color='gray', ls='--', lw=1.2,
               label='Random baseline (~3.6%)')
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Accuracy  (reward > 0)')
    ax.set_title('Shor Code PPO — Scratch vs Transfer Learning')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('transfer_summary.png', dpi=150)
    wandb.log({"summary_chart": wandb.Image('transfer_summary.png')})
    print('  Chart saved → transfer_summary.png')
    plt.show()

    wandb.finish()