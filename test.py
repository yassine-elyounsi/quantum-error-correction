# fix_meta_with_runid.py
import json

meta = {
    "episode":       25000,
    "wandb_run_id":  "s482hwh8",   # ← your actual run ID
    "best_survival": 0.0,
    "global_step":   109200
}

with open(r"checkpoints_d5_continuous_new\d5_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print("Done:", json.dumps(meta, indent=2))