"""
Generalization experiment: does the KnotTransformer transfer to unseen N?

Setup:
  Train on mixed N=4 (2 dof) + N=5 (3 dof) sequences.
  Evaluate on a held-out N=6 (4 dof) test set.

Metrics (matching the thesis's reporting in Tables 6.7 / 6.9):
  - "Acc": fraction of test samples where the transformer beats the chordal
    heuristic on total curvature.
  - Median curvature for transformer vs chordal.

Output is a small JSON summary plus a results table printed to stdout.

Caveats baked into the experiment design:
  - This is Phase 1 only (heuristic-approximation, no energy fine-tuning).
    The thesis's headline numbers come from Phase 2.
  - CPU-only training, so we keep epochs modest (100) and dataset modest
    (~2000 per dof) to finish in reasonable time. Bigger runs would refine
    the numbers but not flip the qualitative finding.
  - The thesis evaluated against "hard" test samples (kappa > 1000). Here we
    use random uniform points and report on whatever distribution comes out
    of that; we'll see the distribution in the output.
"""

import json
import time
import numpy as np
import torch
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, "/home/claude/exp")

from knot_transformer import (
    KnotTransformer,
    BSplineDataset,
    Phase1Trainer,
    collate_variable_length,
    build_full_knot_vector,
    device,
)
import curvature_layer_nd as cl


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 20260525
np.random.seed(SEED)
torch.manual_seed(SEED)


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def gen_random_points(n_samples: int, n_points: int, rng: np.random.Generator):
    """Random 2D point sequences uniformly in [-0.5, 0.5]^2.

    This roughly matches the thesis's data style (random configurations
    bounded in a unit square; the thesis used [0, 0.75] with perturbations).
    """
    return [rng.uniform(-0.5, 0.5, (n_points, 2)) for _ in range(n_samples)]


def chordal_interior_knots(points: np.ndarray) -> np.ndarray:
    """Compute chordal heuristic interior knots for a single (N, 2) sequence."""
    diffs = np.linalg.norm(np.diff(points, axis=0), axis=1)
    total = diffs.sum() + 1e-12
    cum   = np.cumsum(diffs) / total
    # interior knots are the cumulative fractions excluding the last (which is 1.0)
    return cum[:-1]


def build_dataset(n_per_dof: int, rng: np.random.Generator, dofs=(2, 3)):
    """Build a mixed-dof training dataset.

    dof k = N-2 interior knots → N = k+2 points.
    """
    pts_list, heur_list = [], []
    for k in dofs:
        N = k + 2
        pts = gen_random_points(n_per_dof, N, rng)
        for p in pts:
            pts_list.append(p)
            heur_list.append(chordal_interior_knots(p))
    return BSplineDataset(pts_list, heur_list, task="curvature")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

EPS_BOUNDARY = 0.02   # interior knots clamped into [EPS, 1-EPS]


def rescale_knots_to_interior(knots: np.ndarray, eps: float = EPS_BOUNDARY) -> np.ndarray:
    """Linearly compress knots from [0,1] into [eps, 1-eps] to keep them
    strictly interior. Needed because the KnotTransformer's normalisation
    (cumsum / cumsum[-1]) pins the last predicted knot to 1.0 by construction,
    which produces a degenerate knot vector. The rescaling preserves ordering
    and relative spacing — it only compresses the range.
    """
    return eps + (1.0 - 2.0 * eps) * knots


def compute_curvature_batch(points_np: np.ndarray, knots_interior: np.ndarray):
    """Compute total curvature for a single sample.

    points_np: (N, 2) numpy
    knots_interior: (K,) numpy with K = N-2 interior knots in (0, 1)
    Returns: scalar curvature, or np.nan if degenerate.
    """
    # curvatureLayer expects (B, N, 2, 1)
    pts_t = torch.tensor(points_np, dtype=torch.double).unsqueeze(0).unsqueeze(-1)
    # Full knot vector: 4 zeros, K interior, 4 ones
    full_kv = np.concatenate([np.zeros(4), knots_interior, np.ones(4)])
    knots_t = torch.tensor(full_kv, dtype=torch.double).unsqueeze(0)

    # Check non-degenerate
    sorted_kv = np.sort(knots_interior)
    if np.any(np.diff(sorted_kv) < 1e-6):
        return float("nan")
    if knots_interior.min() <= 1e-6 or knots_interior.max() >= 1.0 - 1e-6:
        return float("nan")

    try:
        with torch.no_grad():
            C = cl.solve(pts_t, knots_t)
            eng = cl.global_curvature(C, knots_t, 3, 100)
        val = float(eng.item())
        if not np.isfinite(val):
            return float("nan")
        return val
    except Exception:
        return float("nan")


def evaluate_on_test_set(
    model: KnotTransformer,
    points_list,
    n_dof: int,
    name: str = "test",
):
    """Evaluate model on a list of (N, 2) sequences.

    Returns dict with:
      acc:           fraction where model beats chordal
      n_valid:       number of samples where both energies computed cleanly
      kappa_model:   list of model curvatures
      kappa_chordal: list of chordal curvatures
      median_model, median_chordal
      knot_l1:       mean L1 distance model_knots vs chordal_knots
    """
    model.eval()
    N = n_dof + 2

    kappa_model   = []
    kappa_chordal = []
    knot_l1       = []
    better        = 0
    valid         = 0
    pred_failed   = 0
    chord_failed  = 0

    for points_np in points_list:
        chord_knots = chordal_interior_knots(points_np)

        # Model prediction
        pts_t = torch.tensor(points_np, dtype=torch.double).to(device)
        with torch.no_grad():
            pred_knots = model.predict(pts_t, num_knots=n_dof)
        pred_knots_np = pred_knots.cpu().numpy()
        # The KnotTransformer's normalisation pins the last predicted knot
        # to 1.0 by construction (cumsum / cumsum[-1]), so we compress
        # predictions slightly into the interior before feeding to the solver.
        pred_knots_eval = rescale_knots_to_interior(pred_knots_np)

        # Both energies
        k_m = compute_curvature_batch(points_np, pred_knots_eval)
        k_c = compute_curvature_batch(points_np, chord_knots)

        if np.isnan(k_m):
            pred_failed += 1
        if np.isnan(k_c):
            chord_failed += 1

        if np.isnan(k_m) or np.isnan(k_c):
            continue

        kappa_model.append(k_m)
        kappa_chordal.append(k_c)
        knot_l1.append(float(np.mean(np.abs(pred_knots_eval - chord_knots))))
        if k_m <= k_c:
            better += 1
        valid += 1

    return {
        "name":            name,
        "n_dof":           n_dof,
        "n_points":        N,
        "n_total":         len(points_list),
        "n_valid":         valid,
        "n_pred_failed":   pred_failed,
        "n_chord_failed":  chord_failed,
        "acc":             better / valid if valid else float("nan"),
        "median_model":    float(np.median(kappa_model)) if kappa_model else float("nan"),
        "median_chordal":  float(np.median(kappa_chordal)) if kappa_chordal else float("nan"),
        "mean_knot_l1":    float(np.mean(knot_l1)) if knot_l1 else float("nan"),
    }


def print_eval_block(r: dict):
    print(f"\n── Eval on N={r['n_points']} ({r['n_dof']} dof, name='{r['name']}') ──")
    print(f"  Samples           : {r['n_total']} total, {r['n_valid']} valid "
          f"(pred-fail {r['n_pred_failed']}, chord-fail {r['n_chord_failed']})")
    print(f"  Model beats chord : {r['acc']:.4f}  "
          f"({int(r['acc'] * r['n_valid']) if not np.isnan(r['acc']) else 0}/{r['n_valid']})")
    print(f"  Median kappa      : model {r['median_model']:.3f}  |  chordal {r['median_chordal']:.3f}")
    print(f"  Mean |knot diff|  : {r['mean_knot_l1']:.5f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    print("Generalization experiment — KnotTransformer trained on N∈{4,5},")
    print("                            tested on held-out N=6.")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Seed:   {SEED}")

    rng = np.random.default_rng(SEED)

    # Datasets
    N_TRAIN_PER_DOF = 2000
    N_TEST_PER_DOF  = 500

    print(f"\nBuilding datasets ({N_TRAIN_PER_DOF}/dof train, "
          f"{N_TEST_PER_DOF}/dof test) ...")
    train_ds = build_dataset(N_TRAIN_PER_DOF, rng, dofs=(2, 3))
    print(f"  Train: {len(train_ds)} samples (mixed 2 + 3 dof)")

    # In-distribution test sets (N=4, N=5)
    test_2dof = gen_random_points(N_TEST_PER_DOF, 4, rng)
    test_3dof = gen_random_points(N_TEST_PER_DOF, 5, rng)
    # Out-of-distribution test set (N=6 — never seen during training)
    test_4dof = gen_random_points(N_TEST_PER_DOF, 6, rng)
    print(f"  Test : {N_TEST_PER_DOF} each at 2 dof, 3 dof, 4 dof (held out)")

    # Build model — same architecture as smoke test
    print("\nBuilding model ...")
    model = KnotTransformer(
        d_model=64, nhead=4,
        num_encoder_layers=3, num_decoder_layers=3,
        dim_feedforward=256, dropout=0.1,
        max_points=10, max_knots=6,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    # Phase 1 training
    print("\nPhase 1 training ...")
    p1 = Phase1Trainer(model, lr=1e-3, batch_size=64)
    history = p1.run(
        train_ds, epochs=100, log_every=10,
        save_path="/home/claude/exp/phase1_generalization.pt",
    )
    print(f"  Final L1 loss: {history[-1]:.6f}")

    # Load best
    model.load_state_dict(torch.load(
        "/home/claude/exp/phase1_generalization.pt",
        map_location=device,
    ))

    # Evaluate
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)

    r_2 = evaluate_on_test_set(model, test_2dof, n_dof=2, name="in-dist 2dof")
    r_3 = evaluate_on_test_set(model, test_3dof, n_dof=3, name="in-dist 3dof")
    r_4 = evaluate_on_test_set(model, test_4dof, n_dof=4, name="HELD-OUT 4dof")

    print_eval_block(r_2)
    print_eval_block(r_3)
    print_eval_block(r_4)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'N':>3} {'dof':>4} {'split':>15} {'acc':>8} {'med model':>12} "
          f"{'med chord':>12} {'ratio':>8}")
    for r in (r_2, r_3, r_4):
        ratio = (r['median_model'] / r['median_chordal']
                 if r['median_chordal'] and not np.isnan(r['median_chordal'])
                 else float("nan"))
        split = "train-N" if r['n_dof'] in (2, 3) else "UNSEEN"
        print(f"{r['n_points']:>3} {r['n_dof']:>4} {split:>15} "
              f"{r['acc']:>8.4f} {r['median_model']:>12.3f} "
              f"{r['median_chordal']:>12.3f} {ratio:>8.3f}")

    summary = {
        "seed":            SEED,
        "n_train_per_dof": N_TRAIN_PER_DOF,
        "n_test_per_dof":  N_TEST_PER_DOF,
        "epochs":          100,
        "final_loss":      history[-1],
        "loss_history":    history,
        "results": {
            "in_dist_2dof": r_2,
            "in_dist_3dof": r_3,
            "heldout_4dof": r_4,
        },
    }

    out_path = "/home/claude/exp/generalization_results.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\nResults written to: {out_path}")

    elapsed = time.time() - t0
    print(f"Wall time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
