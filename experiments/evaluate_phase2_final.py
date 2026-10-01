"""
Final evaluation — uses the BEST-BY-HELD-OUT checkpoint from
train_phase2_longer.py.

Runs the full 500-sample evaluation on all three test sets and writes a
clean summary table.

Usage:
    python evaluate_phase2_final.py
"""
import sys, json
sys.path.insert(0, "/home/claude/exp")

import numpy as np
import torch

from knot_transformer import KnotTransformer, device
from generalization_experiment import (
    gen_random_points, evaluate_on_test_set,
)

SEED = 20260525
CKPT = "/home/claude/exp/phase2_best_holdout.pt"
np.random.seed(SEED)
torch.manual_seed(SEED)

# Rebuild the same three 500-sample test sets used elsewhere
rng = np.random.default_rng(SEED)
_ = gen_random_points(2000, 4, rng)
_ = gen_random_points(2000, 5, rng)
test_2dof = gen_random_points(500, 4, rng)
test_3dof = gen_random_points(500, 5, rng)
test_4dof = gen_random_points(500, 6, rng)

model = KnotTransformer(
    d_model=64, nhead=4,
    num_encoder_layers=3, num_decoder_layers=3,
    dim_feedforward=256, dropout=0.1,
    max_points=10, max_knots=6,
)
model.load_state_dict(torch.load(CKPT, map_location=device))
print(f"Loaded: {CKPT}\n")

results = {}
for n_dof, test_set, split in [
    (2, test_2dof, "TRAIN-N"),
    (3, test_3dof, "TRAIN-N"),
    (4, test_4dof, "HELD-OUT"),
]:
    r = evaluate_on_test_set(model, test_set, n_dof=n_dof,
                             name=f"final {n_dof}dof")
    n_better = int(round(r["acc"] * r["n_valid"]))
    ratio = r["median_model"] / r["median_chordal"] if r["median_chordal"] else float("nan")
    print(f"[{split:>8}] N={n_dof+2} ({n_dof} dof) | "
          f"win {r['acc']:.3f} ({n_better}/{r['n_valid']}) | "
          f"med κ model {r['median_model']:>7.2f} chord {r['median_chordal']:>7.2f} | "
          f"ratio {ratio:.4f} | knot-L1 {r['mean_knot_l1']:.4f}")
    results[f"dof{n_dof}"] = r

with open("/home/claude/exp/phase2_final_results.json", "w") as f:
    json.dump(results, f, indent=2, default=float)
print(f"\nWrote /home/claude/exp/phase2_final_results.json")
