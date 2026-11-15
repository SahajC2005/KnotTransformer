"""
quadrature_sweep.py — does the reported curvature depend on `num_pts`?

Background
----------
`cl.global_curvature` integrates kappa^2 * speed on a UNIFORM grid of
`num_pts` samples. The repository uses:

    num_pts = 100   in every evaluation path
    num_pts =  50   as N_QUAD during Phase 2 training (Phase2_BufferBox)

Nobody checked that either resolution had converged. On the hard N=6 sample
stored in cell 42, the integrand peaks near 6.3e5 while its median is about
1 — three orders of magnitude of dynamic range inside a spike narrower than
the grid spacing (at num_pts=100, dt = 0.0101). Which samples happen to land
near the peak then determines the answer, so the reported number is an
aliasing artifact rather than an approximation.

This script measures how much that matters, in two modes.

    single      Sweep num_pts on one sample and show the convergence curve
                against the converged value. Fast (seconds). Use this first.

    population  Hold the knot vectors fixed and re-measure a whole test set
                at each resolution, reporting win rate and median ratio.
                This is the question that decides whether the repository's
                headline numbers need restating: a per-sample error that is
                roughly symmetric across a population partly cancels in a
                median, and only a sweep shows whether it does here.

Usage
-----
    # regression fixture for the solver first
    python experiments/quad_common.py

    # one sample, quick
    python experiments/quadrature_sweep.py --mode single

    # the stored N=6 evaluation set, chordal vs centripetal (no checkpoint)
    python experiments/quadrature_sweep.py --mode population --n-samples 100

    # the real question, if you have a Phase 2 checkpoint
    python experiments/quadrature_sweep.py --mode population \
        --method model --ckpt data/phase2_best_holdout.pt --n-samples 500

Notes
-----
* Knot vectors are computed ONCE per sample (at --knot-pts, only relevant
  for --method optimised) and then evaluated at every sweep resolution. That
  isolates the measurement effect from the optimisation effect, which is the
  comparison the repository's tables actually depend on.
* `--set repo` replays the exact `default_rng` draw sequence used by cell 29
  and `evaluate_phase2_final.py`, so the samples are the ones the stored
  results came from.
* Outputs a JSON file and, unless --no-plot, a PNG.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

import quad_common as qc

DEFAULT_SWEEP = [50, 100, 200, 400, 800, 1600, 3200]
REPO_SEED = 20260525


# ---------------------------------------------------------------------------
# test sets
# ---------------------------------------------------------------------------
def repo_test_set(dof: int, n_samples: int, seed: int = REPO_SEED):
    """Replay the draw sequence used by cell 29 / evaluate_phase2_final.py.

        rng = default_rng(SEED)
        _      = gen_random_points(2000, 4, rng)
        _      = gen_random_points(2000, 5, rng)
        test_2 = gen_random_points(500, 4, rng)   # dof 2
        test_3 = gen_random_points(500, 5, rng)   # dof 3
        test_4 = gen_random_points(500, 6, rng)   # dof 4

    Drawing in the same order matters: the generator is stateful, so the
    dof-4 set is only reproducible if the earlier draws happen too.
    """
    from generalization_experiment import gen_random_points
    rng = np.random.default_rng(seed)
    gen_random_points(2000, 4, rng)
    gen_random_points(2000, 5, rng)
    sets = {
        2: gen_random_points(500, 4, rng),
        3: gen_random_points(500, 5, rng),
        4: gen_random_points(500, 6, rng),
    }
    if dof not in sets:
        raise ValueError(f"--dof must be 2, 3 or 4 for --set repo (got {dof})")
    return sets[dof][:n_samples]


def fresh_test_set(dof: int, n_samples: int, seed: int):
    rng = np.random.default_rng(seed)
    return [rng.uniform(-0.5, 0.5, (dof + 2, 2)) for _ in range(n_samples)]


# ---------------------------------------------------------------------------
# single-sample convergence
# ---------------------------------------------------------------------------
def run_single(args) -> dict:
    pts = qc.NOTEBOOK_SAMPLE
    pairs = [
        ("chordal", qc.NOTEBOOK_CHORDAL_KNOTS),
        ("model", qc.NOTEBOOK_MODEL_KNOTS),
    ]
    print("Single-sample convergence — the N=6 point set from cell 42")
    print(f"Device: {args.device}")
    print()

    conv = {}
    for label, knots in pairs:
        e, m = qc.converged_energy(pts, knots, args.device)
        conv[label] = {"energy": e, "num_pts": m}
        print(f"  {label:<8} converged kappa = {e:12.4f}  (settled at num_pts={m})")

    rows = []
    print()
    print(f"{'num_pts':>8} {'chordal':>12} {'model':>12} {'ratio':>9} "
          f"{'chord err':>10} {'model err':>10}  used by")
    print("-" * 82)
    for m in args.sweep:
        ec = qc.energy(pts, qc.NOTEBOOK_CHORDAL_KNOTS, m, args.device)
        em = qc.energy(pts, qc.NOTEBOOK_MODEL_KNOTS, m, args.device)
        rc = ec / conv["chordal"]["energy"]
        rm = em / conv["model"]["energy"]
        used = ("N_QUAD, training" if m == 50
                else "every eval path" if m == 100 else "")
        rows.append({"num_pts": m, "chordal": ec, "model": em,
                     "ratio": em / ec if ec else float("nan"),
                     "chordal_rel": rc, "model_rel": rm, "used_by": used})
        print(f"{m:>8} {ec:>12.2f} {em:>12.2f} {em / ec:>9.4f} "
              f"{rc:>9.2f}x {rm:>9.2f}x  {used}")

    print("-" * 82)
    true_ratio = conv["model"]["energy"] / conv["chordal"]["energy"]
    print(f"{'converged':>8} {conv['chordal']['energy']:>12.2f} "
          f"{conv['model']['energy']:>12.2f} {true_ratio:>9.4f}"
          f"{'':>11}{'':>11}  the actual integral")

    print()
    print("Reference: cell 42 of the notebook recorded "
          f"chordal {qc.NOTEBOOK_KAPPA_AT_100['chordal']:.2f} and "
          f"model {qc.NOTEBOOK_KAPPA_AT_100['model']:.2f} at num_pts=100.")

    result = {"mode": "single", "sweep": rows, "converged": conv,
              "converged_ratio": true_ratio,
              "notebook_at_100": qc.NOTEBOOK_KAPPA_AT_100}

    if not args.no_plot:
        _plot_single(rows, conv, args.out_dir)
    return result


def _plot_single(rows, conv, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:                                  # pragma: no cover
        print(f"  (skipping plot: {exc})")
        return
    ms = [r["num_pts"] for r in rows]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for label, colour in (("chordal", "tab:blue"), ("model", "tab:orange")):
        ax.plot(ms, [r[label] for r in rows], "o-", color=colour, label=label)
        ax.axhline(conv[label]["energy"], color=colour, ls="--", alpha=0.55,
                   label=f"{label} converged = {conv[label]['energy']:.1f}")
    for m, name in ((50, "N_QUAD (training)"), (100, "eval")):
        ax.axvline(m, color="crimson", ls=":", alpha=0.8)
        ax.annotate(name, (m, ax.get_ylim()[1]), rotation=90,
                    va="top", ha="right", fontsize=8, color="crimson")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("num_pts")
    ax.set_ylabel(r"computed $\kappa$")
    ax.set_title("Bending energy vs quadrature resolution (N=6, cell-42 sample)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "quadrature_convergence.png")
    fig.savefig(path, dpi=130)
    print(f"  wrote {path}")


# ---------------------------------------------------------------------------
# population sweep
# ---------------------------------------------------------------------------
def run_population(args) -> dict:
    if args.set == "repo":
        samples = repo_test_set(args.dof, args.n_samples)
        set_desc = f"repo draw sequence, seed {REPO_SEED}, dof {args.dof}"
    else:
        samples = fresh_test_set(args.dof, args.n_samples, args.seed)
        set_desc = f"fresh uniform, seed {args.seed}, dof {args.dof}"

    model = qc.load_model(args.ckpt, args.device) if args.ckpt else None
    if args.method == "model" and model is None:
        raise SystemExit(
            f"--method model needs a checkpoint; {args.ckpt or '(none given)'} not found.\n"
            "Checkpoints live on Google Drive (see README). Use --method centripetal "
            "to run without one.")

    print(f"Population sweep — {len(samples)} samples, {set_desc}")
    print(f"Baseline: chordal     Method: {args.method}     Device: {args.device}")
    print(f"Knots fixed per sample, then re-measured at each resolution.")
    print()

    # Knot vectors, computed once per sample.
    knots = []
    t0 = time.time()
    for i, pts in enumerate(samples):
        ku = qc.chordal_knots(pts)
        try:
            km = qc.method_knots(args.method, pts, model, args.knot_pts, args.device)
        except Exception:
            km = None
        knots.append((ku, km))
        if args.method == "optimised" and (i + 1) % 20 == 0:
            el = time.time() - t0
            print(f"    knots {i + 1}/{len(samples)}  "
                  f"({el:.0f}s elapsed, ~{el / (i + 1) * (len(samples) - i - 1):.0f}s left)")

    rows = []
    print(f"{'num_pts':>8} {'valid':>7} {'win rate':>10} {'med base':>11} "
          f"{'med method':>11} {'ratio':>9}  used by")
    print("-" * 78)

    sweep = list(args.sweep)
    if args.converged_too:
        sweep = sweep + [args.converged_pts]

    for m in sweep:
        kb, km_, wins, n_valid = [], [], 0, 0
        for pts, (ku, kmth) in zip(samples, knots):
            if kmth is None:
                continue
            eb = qc.energy(pts, ku, m, args.device)
            em = qc.energy(pts, kmth, m, args.device)
            if not (np.isfinite(eb) and np.isfinite(em)):
                continue
            kb.append(eb)
            km_.append(em)
            wins += int(em <= eb)
            n_valid += 1
        if n_valid == 0:
            print(f"{m:>8} {0:>7}   — all samples failed —")
            continue
        med_b = float(np.median(kb))
        med_m = float(np.median(km_))
        used = ("N_QUAD, training" if m == 50
                else "every eval path" if m == 100
                else "converged" if m == args.converged_pts else "")
        rows.append({"num_pts": m, "n_valid": n_valid,
                     "win_rate": wins / n_valid,
                     "median_baseline": med_b, "median_method": med_m,
                     "ratio": med_m / med_b if med_b else float("nan"),
                     "used_by": used})
        print(f"{m:>8} {n_valid:>7} {wins / n_valid:>9.3f} {med_b:>11.2f} "
              f"{med_m:>11.2f} {med_m / med_b:>9.4f}  {used}")

    print("-" * 78)
    if len(rows) >= 2:
        wr = [r["win_rate"] for r in rows]
        rt = [r["ratio"] for r in rows]
        print(f"win rate spread across resolutions : {min(wr):.3f} .. {max(wr):.3f} "
              f"({(max(wr) - min(wr)) * 100:.1f} points)")
        print(f"ratio spread across resolutions    : {min(rt):.4f} .. {max(rt):.4f} "
              f"({max(rt) / max(min(rt), 1e-12):.2f}x)")
        print()
        print("Read it this way: if the win rate and ratio are flat across "
              "resolutions, the\nper-sample aliasing cancels in the median and "
              "the published tables stand as\ncomparisons. If they move, every "
              "ratio in the repository is resolution-\ndependent and needs "
              "restating at a converged setting.")

    result = {"mode": "population", "set": set_desc, "method": args.method,
              "n_samples": len(samples), "rows": rows}

    if not args.no_plot:
        _plot_population(rows, args.method, args.out_dir)
    return result


def _plot_population(rows, method, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:                                  # pragma: no cover
        print(f"  (skipping plot: {exc})")
        return
    if not rows:
        return
    ms = [r["num_pts"] for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.2))
    ax1.plot(ms, [r["win_rate"] for r in rows], "o-", color="tab:blue")
    ax1.set_xscale("log")
    ax1.set_xlabel("num_pts")
    ax1.set_ylabel(f"win rate, {method} vs chordal")
    ax1.set_title("Win rate vs quadrature resolution")
    ax1.grid(alpha=0.3, which="both")
    ax2.plot(ms, [r["ratio"] for r in rows], "o-", color="tab:orange")
    ax2.set_xscale("log")
    ax2.set_xlabel("num_pts")
    ax2.set_ylabel("median ratio")
    ax2.set_title("Median curvature ratio vs resolution")
    ax2.grid(alpha=0.3, which="both")
    for ax in (ax1, ax2):
        for m, name in ((50, "training"), (100, "eval")):
            if m >= min(ms):
                ax.axvline(m, color="crimson", ls=":", alpha=0.8)
    fig.suptitle(f"Does the metric move with resolution?  ({method} vs chordal)")
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"quadrature_population_{method}.png")
    fig.savefig(path, dpi=130)
    print(f"  wrote {path}")


# ---------------------------------------------------------------------------
def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(
        description="Measure how reported curvature depends on quadrature resolution.")
    ap.add_argument("--mode", choices=("single", "population"), default="single")
    ap.add_argument("--sweep", type=int, nargs="+", default=DEFAULT_SWEEP,
                    help="num_pts values to test")
    ap.add_argument("--device", default="cpu", help="cpu or cuda")
    ap.add_argument("--set", choices=("repo", "fresh"), default="repo",
                    help="'repo' replays the stored evaluation draw sequence")
    ap.add_argument("--dof", type=int, default=4,
                    help="interior knot count; 4 means N=6, the held-out length")
    ap.add_argument("--n-samples", type=int, default=100)
    ap.add_argument("--seed", type=int, default=99999, help="for --set fresh")
    ap.add_argument("--method", choices=qc.METHODS, default="centripetal",
                    help="what to compare against chordal")
    ap.add_argument("--ckpt", default="", help="Phase 2 checkpoint for --method model")
    ap.add_argument("--knot-pts", type=int, default=2000,
                    help="resolution used when DERIVING knots (--method optimised)")
    ap.add_argument("--converged-pts", type=int, default=6400)
    ap.add_argument("--no-converged", dest="converged_too", action="store_false",
                    help="skip the extra measurement at --converged-pts, "
                         "which is the slowest row by far")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--out-dir", default=os.path.join(here, "quadrature"))
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    print("=" * 78)
    print("QUADRATURE RESOLUTION SWEEP")
    print("=" * 78)
    t0 = time.time()
    result = run_single(args) if args.mode == "single" else run_population(args)
    result["elapsed_sec"] = round(time.time() - t0, 1)
    result["sweep_requested"] = args.sweep

    out = args.json or os.path.join(
        args.out_dir, f"quadrature_sweep_{args.mode}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(result, fh, indent=2, default=float)
    print(f"\nWrote {out}   ({result['elapsed_sec']}s)")


if __name__ == "__main__":
    main()
