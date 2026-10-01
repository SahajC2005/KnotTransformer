"""
Phase 2 training — resumable, with held-out validation.

Key differences from the previous Phase 2 trainer:
  1. Periodically evaluates on the held-out N=6 test set during training
     and saves the BEST checkpoint by held-out median curvature ratio
     (not by training loss). Lesson learned the hard way: training loss
     and held-out generalization diverge after ~30 epochs.
  2. Saves a "trajectory" log of (epoch, train_loss, held_out_ratio)
     so you can plot and see exactly when generalization peaked.
  3. Resumable across runs — each invocation does CHUNK epochs and
     saves state. Configurable target via the TARGET env var.
  4. Energy cap + low learning rate preset for stability.
  5. Uses the chord-filtered "1k" training set.

Usage:
    # First run — starts from Phase 1 checkpoint:
    python train_phase2_longer.py

    # Continue (each call does CHUNK epochs):
    python train_phase2_longer.py

    # Push to a higher target (default 200):
    TARGET=300 python train_phase2_longer.py

    # Change chunk size (default 3 epochs ~ 2 min):
    CHUNK=5 python train_phase2_longer.py

Outputs:
    /home/claude/exp/phase2_state_v2.pt          — resumable training state
    /home/claude/exp/phase2_best_holdout.pt      — best checkpoint by held-out
    /home/claude/exp/phase2_best_trainloss.pt    — best by training loss
    /home/claude/exp/phase2_trajectory.json      — full training log
"""

import sys, os, time, json
sys.path.insert(0, "/home/claude/exp")

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from knot_transformer import (
    KnotTransformer, BSplineDataset, collate_variable_length, device,
)
import curvature_layer_nd as cl

# Re-use helpers from earlier scripts
from generalization_experiment import (
    gen_random_points,
    evaluate_on_test_set,
    chordal_interior_knots,
    compute_curvature_batch,
)


# ─── Config ──────────────────────────────────────────────────────────────────
SEED          = 20260525
TARGET_EPOCHS = int(os.environ.get("TARGET", 200))
CHUNK_EPOCHS  = int(os.environ.get("CHUNK", 3))
HOLDOUT_EVERY = int(os.environ.get("HOLDOUT_EVERY", 5))   # eval every N epochs
EPS_BOUNDARY  = 0.02
BATCH_SIZE    = 64
LR            = 1e-5
GRAD_CLIP     = 1.0
ENERGY_CAP    = 1000.0
N_HELDOUT     = 200   # smaller than full 500 for speed during training

# Paths — _v2 suffix to keep separate from earlier (epoch-60) state
DATA_PT     = "/home/claude/exp/dataset_1k.pt"
PHASE1_PT   = "/home/claude/exp/phase1_generalization.pt"
STATE_PT    = "/home/claude/exp/phase2_state_v2.pt"
BEST_HOLDOUT_PT  = "/home/claude/exp/phase2_best_holdout.pt"
BEST_TRAIN_PT    = "/home/claude/exp/phase2_best_trainloss.pt"
TRAJ_PT          = "/home/claude/exp/phase2_trajectory.json"


# ─── Differentiable helpers ──────────────────────────────────────────────────
def rescale_interior(knots: torch.Tensor, eps: float = EPS_BOUNDARY) -> torch.Tensor:
    """Compress (0,1) into [eps, 1-eps], differentiable."""
    return eps + (1.0 - 2.0 * eps) * knots


# ─── Training loop (N-grouped, batched solver) ──────────────────────────────
def train_one_epoch(model, loader, optim):
    model.train()
    epoch_capped_energy = 0.0
    epoch_valid = 0
    nan_skips = 0
    raw_energies = []

    for batch in loader:
        points_2d  = batch["points"].to(device)
        point_mask = batch["point_mask"].to(device)
        num_knots  = batch["num_knots"].to(device)

        optim.zero_grad()
        predicted = model(points_2d, point_mask, num_knots)
        B = points_2d.shape[0]

        # Group by point count for batched solver calls
        n_pts_per = (point_mask == False).sum(dim=1)
        groups = {}
        for b in range(B):
            n = int(n_pts_per[b].item())
            groups.setdefault(n, []).append(b)

        batch_energy_sum = []
        batch_n_valid = 0
        for n_pts, idxs in groups.items():
            K = n_pts - 2
            pts_g  = points_2d[idxs, :n_pts]
            pred_g = predicted[idxs, :K]
            interior_g = rescale_interior(pred_g)

            G = len(idxs)
            zeros = torch.zeros(G, 4, dtype=torch.double, device=device)
            ones  = torch.ones( G, 4, dtype=torch.double, device=device)
            full_kv = torch.cat([zeros, interior_g, ones], dim=1)
            pts_cl = pts_g.unsqueeze(-1)

            try:
                C   = cl.solve(pts_cl, full_kv)
                eng = cl.global_curvature(C, full_kv, 3, 100)
            except Exception:
                nan_skips += G
                continue

            finite = torch.isfinite(eng)
            n_bad = int((~finite).sum().item())
            if n_bad:
                nan_skips += n_bad
            n_ok = int(finite.sum().item())
            if n_ok == 0:
                continue

            eng_ok = eng[finite]
            eng_capped = torch.clamp(eng_ok, max=ENERGY_CAP)
            batch_energy_sum.append(eng_capped.sum())
            batch_n_valid += n_ok
            raw_energies.extend(eng_ok.detach().tolist())

        if not batch_energy_sum:
            continue

        loss = torch.stack(batch_energy_sum).sum() / batch_n_valid
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optim.step()

        epoch_capped_energy += float(loss.detach()) * batch_n_valid
        epoch_valid += batch_n_valid

    capped_mean = epoch_capped_energy / max(epoch_valid, 1)
    raw_median  = float(np.median(raw_energies)) if raw_energies else float("nan")
    raw_p99     = float(np.percentile(raw_energies, 99)) if raw_energies else float("nan")
    return capped_mean, raw_median, raw_p99, epoch_valid, nan_skips


# ─── Held-out evaluation (N=6, never seen during training) ──────────────────
def evaluate_holdout(model, holdout_pts):
    """Returns (win_rate, median_ratio) on the held-out N=6 set."""
    r = evaluate_on_test_set(model, holdout_pts, n_dof=4, name="holdout")
    if r["median_chordal"] and not np.isnan(r["median_chordal"]):
        ratio = r["median_model"] / r["median_chordal"]
    else:
        ratio = float("nan")
    return r["acc"], ratio, r


# ─── Build / load the held-out test set ──────────────────────────────────────
def build_holdout():
    """Reproduces the same N=6 test set used in earlier evaluations,
    truncated to N_HELDOUT samples for speed during training."""
    rng = np.random.default_rng(SEED)
    # Consume the same RNG draws as generalization_experiment.py
    _ = gen_random_points(2000, 4, rng)
    _ = gen_random_points(2000, 5, rng)
    _ = gen_random_points(500, 4, rng)
    _ = gen_random_points(500, 5, rng)
    holdout_full = gen_random_points(500, 6, rng)
    return holdout_full[:N_HELDOUT]


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    payload = torch.load(DATA_PT, weights_only=False)
    ds = BSplineDataset(payload["pts"], payload["heur"], task="curvature")
    print(f"Training set: {len(ds)} samples (chord-κ-filtered 1k subset)")

    holdout = build_holdout()
    print(f"Held-out set: {len(holdout)} samples at N=6 (4 dof, UNSEEN)")

    model = KnotTransformer(
        d_model=64, nhead=4,
        num_encoder_layers=3, num_decoder_layers=3,
        dim_feedforward=256, dropout=0.1,
        max_points=10, max_knots=6,
    )
    optim = torch.optim.Adam(model.parameters(), lr=LR)

    # Resume or initialise
    completed = 0
    trajectory = []
    best_holdout_ratio = float("inf")
    best_train_loss    = float("inf")

    if os.path.exists(STATE_PT):
        state = torch.load(STATE_PT, weights_only=False, map_location=device)
        model.load_state_dict(state["model"])
        optim.load_state_dict(state["opt"])
        completed = state["completed"]
        trajectory = state["trajectory"]
        best_holdout_ratio = state.get("best_holdout_ratio", float("inf"))
        best_train_loss    = state.get("best_train_loss", float("inf"))
        print(f"Resumed at epoch {completed}/{TARGET_EPOCHS}")
        print(f"  Best held-out ratio so far: {best_holdout_ratio:.4f}")
        print(f"  Best train loss so far:     {best_train_loss:.4f}")
    else:
        print("Starting fresh from Phase 1 checkpoint.")
        model.load_state_dict(torch.load(PHASE1_PT, map_location=device))

    if completed >= TARGET_EPOCHS:
        print(f"Already at {completed}/{TARGET_EPOCHS}. Done.")
        return

    epochs_this_run = min(CHUNK_EPOCHS, TARGET_EPOCHS - completed)
    print(f"\nRunning {epochs_this_run} epochs "
          f"({completed+1}..{completed+epochs_this_run}) | "
          f"batch={BATCH_SIZE}, lr={LR}, ε={EPS_BOUNDARY}, cap={ENERGY_CAP}")
    print(f"Held-out eval every {HOLDOUT_EVERY} epochs")
    print("─" * 76)

    loader = DataLoader(
        ds, batch_size=BATCH_SIZE,
        shuffle=True, collate_fn=collate_variable_length,
    )

    t0 = time.time()
    for e in range(1, epochs_this_run + 1):
        capped, raw_med, raw_p99, nv, nskip = train_one_epoch(model, loader, optim)
        epoch_no = completed + e

        # Save best by training loss
        train_improved = ""
        if capped < best_train_loss:
            best_train_loss = capped
            torch.save(model.state_dict(), BEST_TRAIN_PT)
            train_improved = " [↓train]"

        # Periodic held-out evaluation
        holdout_improved = ""
        hr_win, hr_ratio = float("nan"), float("nan")
        if epoch_no % HOLDOUT_EVERY == 0 or epoch_no == completed + epochs_this_run:
            t_eval = time.time()
            hr_win, hr_ratio, _ = evaluate_holdout(model, holdout)
            eval_s = time.time() - t_eval
            if hr_ratio < best_holdout_ratio:
                best_holdout_ratio = hr_ratio
                torch.save(model.state_dict(), BEST_HOLDOUT_PT)
                holdout_improved = " [↓holdout]"

        trajectory.append({
            "epoch":      epoch_no,
            "train_loss": capped,
            "raw_median": raw_med,
            "raw_p99":    raw_p99,
            "holdout_win":   hr_win,
            "holdout_ratio": hr_ratio,
        })

        if np.isnan(hr_ratio):
            print(f"Epoch {epoch_no:>4d}/{TARGET_EPOCHS} | "
                  f"train κ: {capped:>7.2f}  med {raw_med:>6.1f}{train_improved}")
        else:
            print(f"Epoch {epoch_no:>4d}/{TARGET_EPOCHS} | "
                  f"train κ: {capped:>7.2f}  med {raw_med:>6.1f} | "
                  f"holdout: win {hr_win:.3f}  ratio {hr_ratio:.4f}"
                  f"{train_improved}{holdout_improved}")

    completed += epochs_this_run
    torch.save({
        "model":               model.state_dict(),
        "opt":                 optim.state_dict(),
        "completed":           completed,
        "trajectory":          trajectory,
        "best_holdout_ratio":  best_holdout_ratio,
        "best_train_loss":     best_train_loss,
    }, STATE_PT)
    with open(TRAJ_PT, "w") as f:
        json.dump(trajectory, f, indent=2, default=float)

    elapsed = time.time() - t0
    print("─" * 76)
    print(f"Chunk done in {elapsed:.1f}s ({elapsed/epochs_this_run:.2f}s/epoch)")
    print(f"Total epochs:       {completed}/{TARGET_EPOCHS}")
    print(f"Best held-out:      ratio {best_holdout_ratio:.4f} → {BEST_HOLDOUT_PT}")
    print(f"Best train loss:    {best_train_loss:.4f}    → {BEST_TRAIN_PT}")


if __name__ == "__main__":
    main()
