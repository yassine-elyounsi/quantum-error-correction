# save this as inspect_checkpoints.py in your quantum_rl root
import os
import torch
import glob

CKPT_DIR = "checkpoints"

print(f"\n{'═'*65}")
print(f"  CHECKPOINT INSPECTOR")
print(f"{'═'*65}")

ckpts = sorted(glob.glob(os.path.join(CKPT_DIR, "*.pt")))

if not ckpts:
    print("  No checkpoints found in checkpoints/")
else:
    for path in ckpts:
        try:
            ckpt = torch.load(path, map_location="cpu")
            
            epsilon    = ckpt.get("epsilon",    "?")
            steps_done = ckpt.get("steps_done", "?")
            
            # Infer distance from network architecture
            # Conv1 weight shape: (32, in_channels, 3, 3)
            # in_channels = 2k = 2*d
            # so d = in_channels / 2
            conv1 = ckpt["online_net"].get("conv.0.weight", None)
            if conv1 is not None:
                in_ch     = conv1.shape[1]   # number of input channels
                d         = in_ch // 2       # distance = k = in_ch / 2
                grid_size = int(conv1.shape[0])  # not grid, this is out_channels
                
                # Grid size from advantage stream
                # advantage[0] = Linear(flat_size, 256)
                # flat_size = 64 * grid_size²
                adv = ckpt["online_net"].get("advantage_stream.0.weight", None)
                if adv is not None:
                    flat_size = adv.shape[1]        # 64 * grid²
                    grid_sq   = flat_size // 64
                    import math
                    grid_size = int(math.sqrt(grid_sq))
                    
                    # n_actions from advantage stream output
                    adv_out   = ckpt["online_net"].get("advantage_stream.2.weight")
                    n_actions = adv_out.shape[0]
                    
                    # Verify: n_actions = 2*d² + 1
                    # d² = (n_actions - 1) / 2
                    d_from_actions = int(((n_actions - 1) / 2) ** 0.5)
                    
                    fname = os.path.basename(path)
                    size  = os.path.getsize(path) / 1024 / 1024
                    
                    print(f"\n  📁 {fname}  ({size:.1f} MB)")
                    print(f"     distance    : d = {d_from_actions}")
                    print(f"     grid size   : {grid_size}×{grid_size}")
                    print(f"     state shape : ({in_ch}, {grid_size}, {grid_size})")
                    print(f"     n_actions   : {n_actions}")
                    print(f"     epsilon     : {epsilon:.4f}" if isinstance(epsilon, float) else f"     epsilon     : {epsilon}")
                    print(f"     steps_done  : {steps_done:,}" if isinstance(steps_done, int) else f"     steps_done  : {steps_done}")
                    
                    # Estimate episode based on steps_done
                    # Assuming ~15 steps per episode on average
                    if isinstance(steps_done, int) and steps_done > 0:
                        est_episodes = steps_done // 15
                        print(f"     est episodes: ~{est_episodes:,}")
                    
                    # Epsilon tells us training progress
                    if isinstance(epsilon, float):
                        if epsilon > 0.9:
                            status = "⚠️  Very early training (ε ≈ 1.0)"
                        elif epsilon > 0.5:
                            status = "🔄  Early-mid training"
                        elif epsilon > 0.1:
                            status = "📈  Mid-late training"
                        else:
                            status = "✅  Late training (ε ≈ min)"
                        print(f"     status      : {status}")
                        
        except Exception as e:
            fname = os.path.basename(path)
            print(f"\n  ❌ {fname}  — could not read: {e}")

print(f"\n{'═'*65}\n")