"""
energy_landscape.py — the curvature landscape figure, with the marker bug fixed.

What this replaces
------------------
Cells 42 and 43 of `notebooks/Phase2_BufferBox.ipynb` produce
`curvature_landscape.png` and `curvature_landscape_v2.png`. Both are
unreliable, for two separate reasons:

1.  `find_local_min_robust` (cell 43) returns an improved kappa paired with
    the WRONG knot vector:

        best_k_this = init.copy()
        ...
            if k_eval < best_kappa_this:
                best_kappa_this = k_eval
                best_k = knots.detach().numpy().copy()   # writes best_k
        if best_kappa_this < best_overall_kappa:
            best_overall_kappa = best_kappa_this
            best_overall_k     = best_k_this             # reads best_k_this

    `best_k` is a dead store and `best_k_this` is never updated after
    initialisation, so the returned knots are the initial RANDOM start. Those
    knots then set `k1_fixed` and `k4_fixed`, which define the slice the whole
    heatmap is computed on — so the published figure is a slice through an
    arbitrary point with a mismarked optimum.

2.  Cell 42 sets `cl.device = DEVICE` (CUDA) and then builds CPU tensors.
    `construct` allocates its temporaries on `cl.device`, they meet CPU
    inputs, a RuntimeError is raised, and `except Exception: break` swallows
    it on the first iteration — which is why that cell's stored output reports
    `curvature : inf` with the local-min knots identical to the model's.

Here the optimiser keeps each (knots, energy) pair together, every reported
pair is re-derived through `qc.assert_pair` before it is plotted, and
`qc.energy` pins the solver's device global on every call.

Usage
-----
    python experiments/energy_landscape.py
    python experiments/energy_landscape.py --grid 96 --num-pts 2000
    python experiments/energy_landscape.py --slice-at chordal
    python experiments/energy_landscape.py --ckpt data/phase2_best_holdout.pt
    python experiments/energy_landscape.py --sample-seed 20260601 --min-chordal 500

Outputs
-------
    experiments/quadrature/energy_landscape.png
    experiments/quadrature/energy_landscape.json
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

import quad_common as qc


# ---------------------------------------------------------------------------
# sample selection
# ---------------------------------------------------------------------------
def pick_sample(args):
    """Return (points, description)."""
    if args.sample_seed is None:
        return qc.NOTEBOOK_SAMPLE, "cell-42 sample (N=6)"

    rng = np.random.default_rng(args.sample_seed)
    n = args.dof + 2
    for attempt in range(args.max_tries):
        pts = rng.uniform(-0.5, 0.5, (n, 2))
        kc = qc.chordal_knots(pts)
        e = qc.energy(pts, kc, args.num_pts)
        if np.isfinite(e) and e > args.min_chordal:
            return pts, (f"seed {args.sample_seed}, draw {attempt + 1}, "
                         f"N={n}, chordal kappa {e:.1f}")
    raise SystemExit(
        f"no sample with chordal kappa > {args.min_chordal} in "
        f"{args.max_tries} draws; lower --min-chordal")


# ---------------------------------------------------------------------------
# the grid
# ---------------------------------------------------------------------------
def sweep_slice(points, u1, u4, grid, num_pts, device, verbose=True):
    """kappa over (u2, u3) with u1 and u4 held fixed.

    Returns (values, axis) where values[j, i] corresponds to
    u2 = axis[i], u3 = axis[j], and NaN marks the invalid half-plane
    u3 <= u2 + MIN_GAP where no ordered knot vector exists.
    """
    lo, hi = u1 + qc.MIN_GAP, u4 - qc.MIN_GAP
    if hi - lo < 0.02:
        raise SystemExit(
            f"slice is too thin: u1={u1:.4f}, u4={u4:.4f} leave only "
            f"{hi - lo:.4f} between them. Pick another --slice-at.")

    axis = lo + (hi - lo) * (np.arange(grid) + 0.5) / grid
    vals = np.full((grid, grid), np.nan)
    t0 = time.time()
    for j, u3 in enumerate(axis):
        for i, u2 in enumerate(axis):
            if u3 <= u2 + qc.MIN_GAP:
                continue
            e = qc.energy(points, [u1, u2, u3, u4], num_pts, device)
            if np.isfinite(e):
                vals[j, i] = e
        if verbose and (j + 1) % max(1, grid // 8) == 0:
            el = time.time() - t0
            print(f"    row {j + 1}/{grid}  "
                  f"({el:.0f}s elapsed, ~{el / (j + 1) * (grid - j - 1):.0f}s left)")
    return vals, axis


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------
def plot(vals, axis, marks, points, meta, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
    except Exception as exc:                                  # pragma: no cover
        print(f"  (skipping plot: {exc})")
        return None

    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        print("  (nothing finite to plot)")
        return None
    vmin = max(float(finite.min()), 1e-3)
    vmax = float(np.percentile(finite, 98))

    fig, (ax, bx) = plt.subplots(
        1, 2, figsize=(13.2, 5.6), gridspec_kw={"width_ratios": [1.25, 1]})

    masked = np.ma.masked_invalid(vals)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#2e2e34")
    im = ax.imshow(masked, origin="lower", aspect="auto", cmap=cmap,
                   extent=[axis[0], axis[-1], axis[0], axis[-1]],
                   norm=LogNorm(vmin=vmin, vmax=vmax))
    fig.colorbar(im, ax=ax, label=r"total curvature $\kappa$ (log)")

    styles = {
        "chordal":  dict(marker="o", color="cyan",  size=170),
        "model":    dict(marker="D", color="red",   size=150),
        "optimum":  dict(marker="*", color="lime",  size=330),
    }
    for name, m in marks.items():
        u = m["knots"]
        st = styles.get(name, dict(marker="s", color="white", size=120))
        inside = (axis[0] <= u[1] <= axis[-1]) and (axis[0] <= u[2] <= axis[-1])
        ax.scatter([u[1]], [u[2]], marker=st["marker"], s=st["size"],
                   facecolor=st["color"], edgecolor="black", linewidth=1.8,
                   zorder=6,
                   label=f"{name} ($\\kappa$={m['kappa']:.1f})"
                         + ("" if inside else "  [off slice]"))
    ax.plot(axis, axis + qc.MIN_GAP, color="white", lw=0.8, ls=":", alpha=0.6)
    ax.set_xlabel(r"knot $u_2$")
    ax.set_ylabel(r"knot $u_3$")
    ax.set_title(f"Curvature landscape on the slice "
                 f"$u_1$={meta['u1']:.3f}, $u_4$={meta['u4']:.3f}\n"
                 f"{meta['sample']}  ·  num_pts={meta['num_pts']}  ·  "
                 f"{meta['grid']}x{meta['grid']} grid")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.85)

    # right panel: the curves themselves, so the markers mean something
    for name, m in marks.items():
        st = styles.get(name, dict(color="white"))
        xy = _curve(points, m["knots"])
        if xy is not None:
            bx.plot(xy[:, 0], xy[:, 1], color=st["color"], lw=1.7,
                    label=f"{name}  $\\kappa$={m['kappa']:.1f}")
    bx.scatter(points[:, 0], points[:, 1], c="black", s=46, zorder=6,
               label="waypoints")
    bx.set_aspect("equal", adjustable="datalim")
    bx.grid(alpha=0.3)
    bx.set_title("The interpolants those knots produce")
    bx.legend(fontsize=8)

    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "energy_landscape.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    print(f"  wrote {path}")
    return path


def _curve(points, interior, n_eval=320):
    """Sample the interpolant, for the right-hand panel."""
    ctrl = qc.control_points(points, interior)
    if ctrl is None:
        return None
    import torch
    from quad_common import cl, DEGREE
    kv = torch.tensor(qc.full_knot_vector(interior),
                      dtype=torch.double).unsqueeze(0)
    cp = torch.tensor(ctrl, dtype=torch.double).unsqueeze(0).unsqueeze(-1)
    cl.device = torch.device("cpu")
    out = []
    for t in np.linspace(0.0, 1.0 - 1e-9, n_eval):
        tt = torch.full((1,), float(t), dtype=torch.double)
        try:
            with torch.no_grad():
                p = cl.de_boor(kv, tt, cp, DEGREE)
            out.append([float(p[0, 0, 0]), float(p[0, 1, 0])])
        except Exception:
            return None
    return np.asarray(out)


# ---------------------------------------------------------------------------
def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(
        description="Corrected curvature-landscape figure (replaces cells 42/43).")
    ap.add_argument("--grid", type=int, default=64)
    ap.add_argument("--num-pts", type=int, default=2000,
                    help="quadrature resolution for the grid; 100 reproduces "
                         "the notebook's (aliased) surface for comparison")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--slice-at", choices=("optimum", "chordal", "model"),
                    default="optimum",
                    help="whose u1 and u4 to hold fixed")
    ap.add_argument("--restarts", type=int, default=12)
    ap.add_argument("--opt-steps", type=int, default=200)
    ap.add_argument("--ckpt", default="",
                    help="Phase 2 checkpoint; without one the model marker "
                         "uses the knot vector recorded in cell 42")
    ap.add_argument("--sample-seed", type=int, default=None,
                    help="draw a fresh sample instead of using cell 42's")
    ap.add_argument("--dof", type=int, default=4)
    ap.add_argument("--min-chordal", type=float, default=500.0)
    ap.add_argument("--max-tries", type=int, default=400)
    ap.add_argument("--out-dir", default=os.path.join(here, "quadrature"))
    args = ap.parse_args()

    if args.dof != 4 and args.sample_seed is not None:
        raise SystemExit("this slice plot fixes u1 and u4, so it needs dof 4 (N=6)")

    print("=" * 78)
    print("CURVATURE LANDSCAPE  (corrected)")
    print("=" * 78)

    points, desc = pick_sample(args)
    print(f"Sample  : {desc}")
    print(f"Device  : {args.device}     grid {args.grid}x{args.grid}     "
          f"num_pts {args.num_pts}")
    print()

    marks = {}

    kc = qc.chordal_knots(points)
    ec = qc.energy(points, kc, args.num_pts, args.device)
    qc.assert_pair(points, kc, ec, args.num_pts, args.device, "chordal")
    marks["chordal"] = {"knots": kc, "kappa": ec}
    print(f"chordal  knots {np.round(kc, 5)}   kappa {ec:.2f}")

    model = qc.load_model(args.ckpt, args.device) if args.ckpt else None
    if model is not None:
        km = qc.model_knots(model, points, args.device)
        src = os.path.basename(args.ckpt)
    elif args.sample_seed is None:
        km = qc.project_knots(qc.NOTEBOOK_MODEL_KNOTS)
        src = "cell-42 record"
    else:
        km = None
        src = "unavailable (no checkpoint, fresh sample)"
    if km is not None:
        em = qc.energy(points, km, args.num_pts, args.device)
        qc.assert_pair(points, km, em, args.num_pts, args.device, "model")
        marks["model"] = {"knots": km, "kappa": em}
        print(f"model    knots {np.round(km, 5)}   kappa {em:.2f}   [{src}]")
    else:
        print(f"model    skipped — {src}")

    print(f"\nsearching for the best local minimum "
          f"({args.restarts} restarts + chordal/model starts) ...")
    seeds = [kc] + ([km] if km is not None else [])
    ko, eo = qc.multi_restart(points, args.num_pts, args.device,
                              n_restarts=args.restarts, steps=args.opt_steps,
                              include=seeds)
    # This is the assertion the notebook needed and did not have.
    qc.assert_pair(points, ko, eo, args.num_pts, args.device, "optimum")
    marks["optimum"] = {"knots": ko, "kappa": eo}
    print(f"optimum  knots {np.round(ko, 5)}   kappa {eo:.2f}")
    print("  (knots and kappa re-derived from each other — the pairing cannot "
          "silently drift)")

    anchor = marks.get(args.slice_at) or marks["optimum"]
    u1, u4 = float(anchor["knots"][0]), float(anchor["knots"][3])
    print(f"\nslicing at u1={u1:.4f}, u4={u4:.4f}  (from '{args.slice_at}')")

    vals, axis = sweep_slice(points, u1, u4, args.grid, args.num_pts, args.device)
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        raise SystemExit(
            "every grid point was invalid or singular — widen the slice "
            "(try --slice-at chordal) or lower --grid")
    print(f"\n{finite.size}/{vals.size} grid points valid; "
          f"kappa {finite.min():.2f} .. {finite.max():.2f}")

    jbest, ibest = np.unravel_index(np.nanargmin(vals), vals.shape)
    print(f"grid minimum on this slice: kappa {vals[jbest, ibest]:.2f} at "
          f"u2={axis[ibest]:.4f}, u3={axis[jbest]:.4f}")
    if eo < vals[jbest, ibest]:
        print("  the descent found a lower point than the grid resolves — "
              "expected, the grid is coarse")

    meta = {"sample": desc, "u1": u1, "u4": u4, "grid": args.grid,
            "num_pts": args.num_pts, "slice_at": args.slice_at,
            "device": args.device}
    plot(vals, axis, marks, points, meta, args.out_dir)

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, "energy_landscape.json")
    with open(out, "w") as fh:
        json.dump({
            "meta": meta,
            "points": points.tolist(),
            "marks": {k: {"knots": np.asarray(v["knots"]).tolist(),
                          "kappa": v["kappa"]} for k, v in marks.items()},
            "axis": axis.tolist(),
            "grid_min": {"kappa": float(vals[jbest, ibest]),
                         "u2": float(axis[ibest]), "u3": float(axis[jbest])},
        }, fh, indent=2, default=float)
    print(f"  wrote {out}")

    print("\nTo see the surface the notebook was actually optimising against, "
          "re-run with\n  --num-pts 100   (the aliased evaluation setting).")


if __name__ == "__main__":
    main()
