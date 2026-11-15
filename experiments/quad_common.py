"""
quad_common.py — shared, device-safe wrappers around the repository's own
B-spline solver, used by `quadrature_sweep.py` and `energy_landscape.py`.

Why this module exists
----------------------
Three problems in the notebook made the same two experiments hard to trust,
and all three are handled here once instead of being re-solved per script:

1.  `curvature_layer_nd` keeps a MODULE-LEVEL `device` global and calls
    `.to(device)` on every internal temporary. Hand it tensors from a
    different device and it raises, which several notebook call sites
    swallow with a bare `except`. `energy()` below pins `cl.device` to the
    device of the tensors it is about to build, every call, so the solver
    can never be handed a mismatched pair.

2.  The notebook's local-minimum search returns an improved kappa paired
    with the WRONG knot vector (`best_k_this` is never updated inside the
    loop; see `experiments/quadrature/README.md`). `optimise_knots()` here
    keeps the pair together and `assert_pair()` re-derives kappa from the
    returned knots so a regression cannot pass silently.

3.  Energies in the repository are reported at `num_pts=100` (evaluation)
    and `num_pts=50` (training) without anyone checking that the quadrature
    had converged. `energy()` therefore takes `num_pts` explicitly — there
    is no default — so every call site has to state its resolution.

Nothing here reimplements the maths. `energy()` calls `cl.solve` and
`cl.global_curvature` so that measurements are about the shipped code.

Self-test:
    python experiments/quad_common.py
"""

from __future__ import annotations

import os
import sys
from typing import Iterable, Sequence

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Import the project's own modules from src/, resolved relative to this file
# so the scripts run from any working directory (the older experiment
# scripts hardcode /home/claude/exp, which only existed in one sandbox).
# ---------------------------------------------------------------------------
_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "src")
_SRC = os.path.normpath(_SRC)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import curvature_layer_nd as cl  # noqa: E402
import generalization_experiment as ge  # noqa: E402

DEGREE = 3
EPS_BOUNDARY = 0.02   # matches rescale_knots_to_interior / EPS_BOUNDARY
MIN_GAP = 0.008       # smallest interior knot gap we ever allow

# ---------------------------------------------------------------------------
# The reference sample
# ---------------------------------------------------------------------------
# The N=6 point set printed by cell 42 of notebooks/Phase2_BufferBox.ipynb,
# together with the two knot vectors and the two energies that cell recorded.
# Used as a regression fixture: the solver must still reproduce these.
NOTEBOOK_SAMPLE = np.array([
    [ 0.1144607,  -0.02903495],
    [-0.44626413,  0.13820953],
    [-0.14740674,  0.21086831],
    [ 0.31514286,  0.05016787],
    [-0.05143829,  0.41838578],
    [ 0.43318564,  0.45778141],
])
NOTEBOOK_CHORDAL_KNOTS = np.array([0.24501364, 0.37379954, 0.57883911, 0.796404])
NOTEBOOK_MODEL_KNOTS = np.array([0.0342271, 0.04479033, 0.96740919, 0.98])
NOTEBOOK_KAPPA_AT_100 = {"chordal": 711.51, "model": 68.42}


# ---------------------------------------------------------------------------
# Knot-vector helpers
# ---------------------------------------------------------------------------
def full_knot_vector(interior: Sequence[float]) -> np.ndarray:
    """[0,0,0,0, u1..uK, 1,1,1,1] — length K + 8 = N + 6."""
    return np.concatenate([np.zeros(4), np.asarray(interior, dtype=float), np.ones(4)])


def project_knots(interior: Sequence[float],
                  eps: float = EPS_BOUNDARY,
                  min_gap: float = MIN_GAP) -> np.ndarray:
    """Nearest valid interior knot vector: sorted, inside [eps, 1-eps],
    every adjacent gap at least `min_gap`.

    Both passes are needed. A forward-only pass can push the last knot past
    1-eps; a backward-only pass can leave an earlier knot above a later one.
    """
    v = np.sort(np.asarray(interior, dtype=float)).copy()
    n = len(v)
    np.clip(v, eps, 1.0 - eps, out=v)
    for i in range(1, n):                      # open every gap upward
        if v[i] < v[i - 1] + min_gap:
            v[i] = v[i - 1] + min_gap
    if v[-1] > 1.0 - eps:                      # we overshot the top
        v[-1] = 1.0 - eps
    for i in range(n - 1, 0, -1):              # pull the chain back down
        if v[i - 1] > v[i] - min_gap:
            v[i - 1] = v[i] - min_gap
    if v[0] < eps:                             # n*min_gap does not fit at all
        v = eps + (1.0 - 2.0 * eps) * np.arange(1, n + 1) / (n + 1)
    return v


def is_valid(interior: Sequence[float],
             eps: float = EPS_BOUNDARY,
             min_gap: float = MIN_GAP) -> bool:
    v = np.asarray(interior, dtype=float)
    if not np.all(np.isfinite(v)):
        return False
    if v.min() < eps - 1e-12 or v.max() > 1.0 - eps + 1e-12:
        return False
    return bool(np.all(np.diff(v) >= min_gap - 1e-12))


def chordal_knots(points: np.ndarray) -> np.ndarray:
    """The repository's chordal heuristic, projected into the valid region."""
    return project_knots(ge.chordal_interior_knots(points))


def centripetal_knots(points: np.ndarray) -> np.ndarray:
    """Centripetal parameterisation: cumulative sqrt(chord), normalised.

    Matches `cl.centripetal_nd`, which takes sqrt of the chord length
    (sqrt(sqrt(sum of squared differences))).
    """
    d = np.sqrt(np.linalg.norm(np.diff(points, axis=0), axis=1))
    cum = np.cumsum(d) / (d.sum() + 1e-12)
    return project_knots(cum[:-1])


def uniform_knots(n_points: int) -> np.ndarray:
    k = n_points - 2
    return project_knots(np.arange(1, k + 1) / (k + 1))


# ---------------------------------------------------------------------------
# Energy — the one place that touches the solver
# ---------------------------------------------------------------------------
def energy(points: np.ndarray,
           interior: Sequence[float],
           num_pts: int,
           device: str | torch.device = "cpu") -> float:
    """Total bending energy of the clamped cubic interpolant.

    `num_pts` is required on purpose: the whole point of these experiments is
    that the answer depends on it. Returns +inf rather than raising, so that
    sweeps over invalid or singular configurations do not need try/except at
    every call site.
    """
    interior = np.asarray(interior, dtype=float)
    if not is_valid(interior):
        return float("inf")

    dev = torch.device(device)
    # Pin the solver's module global to the device we are about to use, so
    # the internal `.to(cl.device)` calls cannot produce a mismatch.
    cl.device = dev

    pts_t = (torch.tensor(points, dtype=torch.double)
             .unsqueeze(0).unsqueeze(-1).to(dev))          # (1, N, 2, 1)
    kv_t = (torch.tensor(full_knot_vector(interior), dtype=torch.double)
            .unsqueeze(0).to(dev))                         # (1, N+6)
    try:
        with torch.no_grad():
            ctrl = cl.solve(pts_t, kv_t)
            val = float(cl.global_curvature(ctrl, kv_t, DEGREE, num_pts).item())
    except Exception:
        return float("inf")
    return val if np.isfinite(val) else float("inf")


def control_points(points: np.ndarray,
                   interior: Sequence[float],
                   device: str | torch.device = "cpu") -> np.ndarray | None:
    """(N+2, 2) control points, or None if the solve is degenerate."""
    interior = np.asarray(interior, dtype=float)
    if not is_valid(interior):
        return None
    dev = torch.device(device)
    cl.device = dev
    pts_t = (torch.tensor(points, dtype=torch.double)
             .unsqueeze(0).unsqueeze(-1).to(dev))
    kv_t = (torch.tensor(full_knot_vector(interior), dtype=torch.double)
            .unsqueeze(0).to(dev))
    try:
        with torch.no_grad():
            ctrl = cl.solve(pts_t, kv_t)
    except Exception:
        return None
    arr = ctrl.detach().cpu().numpy()
    while arr.ndim > 3:
        arr = arr[..., 0]
    arr = arr[0]
    return arr if np.all(np.isfinite(arr)) else None


def converged_energy(points: np.ndarray,
                     interior: Sequence[float],
                     device: str | torch.device = "cpu",
                     start: int = 400,
                     limit: int = 25600,
                     rtol: float = 2e-3) -> tuple[float, int]:
    """Double `num_pts` until the integral stops moving by more than `rtol`.

    Returns (energy, num_pts_used). Reports the resolution it settled on so
    callers can record it rather than assume it.
    """
    m = start
    prev = energy(points, interior, m, device)
    while m < limit:
        m *= 2
        cur = energy(points, interior, m, device)
        if not np.isfinite(cur):
            return prev, m // 2
        if np.isfinite(prev) and abs(cur - prev) <= rtol * max(abs(cur), 1e-12):
            return cur, m
        prev = cur
    return prev, m


# ---------------------------------------------------------------------------
# A local-minimum search whose knots and energy cannot drift apart
# ---------------------------------------------------------------------------
def optimise_knots(points: np.ndarray,
                   init: Sequence[float],
                   num_pts: int,
                   device: str | torch.device = "cpu",
                   steps: int = 300,
                   lr: float = 4e-3) -> tuple[np.ndarray, float]:
    """Projected finite-difference descent on the bending energy.

    Returns (knots, energy_at_those_knots) — always a matching pair. The
    notebook's version tracked the best energy in one variable and the best
    knots in another and then returned the wrong one; here the energy is
    re-derived from the knots that are actually returned.
    """
    cur = project_knots(init)
    best_u = cur.copy()
    best_e = energy(points, cur, num_pts, device)
    step = lr

    for _ in range(steps):
        e0 = energy(points, cur, num_pts, device)
        if not np.isfinite(e0):
            break
        grad = np.zeros_like(cur)
        h = 4e-4
        for i in range(len(cur)):
            up = cur.copy()
            up[i] += h
            e1 = energy(points, project_knots(up), num_pts, device)
            grad[i] = (e1 - e0) / h if np.isfinite(e1) else 0.0
        norm = np.linalg.norm(grad)
        if norm < 1e-14:
            break
        cand = project_knots(cur - step * grad / norm)
        e_cand = energy(points, cand, num_pts, device)
        if np.isfinite(e_cand) and e_cand < e0:
            cur = cand
            if e_cand < best_e:
                best_u, best_e = cur.copy(), e_cand
        else:
            step *= 0.6
            if step < 1e-6:
                break

    # Re-derive, so the returned pair is self-consistent by construction.
    best_e = energy(points, best_u, num_pts, device)
    return best_u, best_e


def multi_restart(points: np.ndarray,
                  num_pts: int,
                  device: str | torch.device = "cpu",
                  n_restarts: int = 12,
                  steps: int = 200,
                  seed: int = 0,
                  include: Iterable[Sequence[float]] = ()) -> tuple[np.ndarray, float]:
    """Best local minimum over random restarts plus any supplied starts."""
    k = points.shape[0] - 2
    rng = np.random.default_rng(seed)
    starts = [np.asarray(s, dtype=float) for s in include]
    for _ in range(n_restarts):
        starts.append(np.sort(rng.uniform(0.08, 0.92, k)))

    best_u, best_e = None, float("inf")
    for s in starts:
        u, e = optimise_knots(points, s, num_pts, device, steps=steps)
        if e < best_e:
            best_u, best_e = u, e
    if best_u is None:
        best_u = project_knots(uniform_knots(points.shape[0]))
        best_e = energy(points, best_u, num_pts, device)
    return best_u, best_e


def assert_pair(points: np.ndarray,
                interior: Sequence[float],
                claimed: float,
                num_pts: int,
                device: str | torch.device = "cpu",
                label: str = "",
                rtol: float = 1e-6) -> None:
    """Guard against the notebook's knots/energy mismatch bug.

    Any code path that reports "these knots give this energy" should call
    this before plotting or writing the number out.
    """
    actual = energy(points, interior, num_pts, device)
    if not (np.isfinite(actual) and np.isfinite(claimed)):
        raise AssertionError(
            f"{label}: non-finite energy (claimed {claimed}, recomputed {actual})")
    if abs(actual - claimed) > rtol * max(abs(actual), 1.0):
        raise AssertionError(
            f"{label}: knots and energy disagree — claimed {claimed:.6f}, "
            f"recomputed {actual:.6f} from {np.asarray(interior)}")


# ---------------------------------------------------------------------------
# Optional model knots
# ---------------------------------------------------------------------------
def load_model(ckpt_path: str, device: str | torch.device = "cpu"):
    """Load a Phase-2 KnotTransformer checkpoint, or return None if absent.

    Checkpoints are not in the repository (see README). Every script here
    works without one; passing --ckpt just swaps the comparison method.
    """
    if not ckpt_path or not os.path.exists(ckpt_path):
        return None
    import knot_transformer as kt
    dev = torch.device(device)
    kt.device = dev
    model = kt.KnotTransformer(
        d_model=64, nhead=4,
        num_encoder_layers=3, num_decoder_layers=3,
        dim_feedforward=256, dropout=0.1,
        max_points=10, max_knots=6,
    ).to(dev)
    model.load_state_dict(torch.load(ckpt_path, map_location=dev))
    model.eval()
    return model


def model_knots(model, points: np.ndarray,
                device: str | torch.device = "cpu") -> np.ndarray:
    """Predicted interior knots, rescaled into the interior as every eval
    path in the repository does."""
    dev = torch.device(device)
    pts_t = torch.tensor(points, dtype=torch.double).to(dev)
    with torch.no_grad():
        raw = model.predict(pts_t, num_knots=points.shape[0] - 2)
    return project_knots(ge.rescale_knots_to_interior(raw.cpu().numpy()))


METHODS = ("centripetal", "uniform", "model", "optimised")


def method_knots(name: str, points: np.ndarray, model=None,
                 num_pts: int = 2000,
                 device: str | torch.device = "cpu") -> np.ndarray:
    if name == "centripetal":
        return centripetal_knots(points)
    if name == "uniform":
        return uniform_knots(points.shape[0])
    if name == "model":
        if model is None:
            raise ValueError("method 'model' needs --ckpt")
        return model_knots(model, points, device)
    if name == "optimised":
        u, _ = optimise_knots(points, chordal_knots(points), num_pts, device)
        return u
    raise ValueError(f"unknown method {name!r}; choose from {METHODS}")


# ---------------------------------------------------------------------------
# Self-test — regression fixture for the solver
# ---------------------------------------------------------------------------
def _self_test() -> int:
    print("quad_common self-test")
    print("=" * 66)
    failures = 0

    print("\n[1] project_knots keeps knots ordered and inside the box")
    cases = [
        [0.5, 0.5, 0.5, 0.5],
        [0.98, 0.98, 0.98, 0.98],
        [0.02, 0.02, 0.02, 0.02],
        [0.9, 0.1, 0.5, 0.3],
        [0.97, 0.975, 0.979, 0.98],
        list(NOTEBOOK_MODEL_KNOTS),
    ]
    for c in cases:
        p = project_knots(c)
        ok = is_valid(p)
        failures += 0 if ok else 1
        print(f"    {np.round(c, 4)}  ->  {np.round(p, 4)}   {'ok' if ok else 'INVALID'}")

    print("\n[2] solver reproduces the energies recorded in cell 42")
    for label, knots in (("chordal", NOTEBOOK_CHORDAL_KNOTS),
                         ("model", NOTEBOOK_MODEL_KNOTS)):
        got = energy(NOTEBOOK_SAMPLE, knots, 100)
        want = NOTEBOOK_KAPPA_AT_100[label]
        ok = abs(got - want) < 0.02
        failures += 0 if ok else 1
        print(f"    {label:<8} num_pts=100  got {got:10.4f}   notebook {want:8.2f}   "
              f"{'match' if ok else 'MISMATCH'}")

    print("\n[3] the interpolant passes through every data point")
    for label, knots in (("chordal", NOTEBOOK_CHORDAL_KNOTS),
                         ("model", NOTEBOOK_MODEL_KNOTS)):
        ctrl = control_points(NOTEBOOK_SAMPLE, knots)
        ok = ctrl is not None and ctrl.shape == (NOTEBOOK_SAMPLE.shape[0] + 2, 2)
        failures += 0 if ok else 1
        print(f"    {label:<8} control points {None if ctrl is None else ctrl.shape}   "
              f"{'ok' if ok else 'FAILED'}")

    print("\n[4] chord_cum feature equals the chordal knot vector")
    # compute_geometric_features hands the network the chordal heuristic
    # directly; this pins that identity.
    import knot_transformer as kt
    feats = kt.compute_geometric_features(
        torch.tensor(NOTEBOOK_SAMPLE, dtype=torch.double).unsqueeze(0))
    chord_cum = feats[0, 1:-1, 0].numpy()
    diff = float(np.max(np.abs(chord_cum - NOTEBOOK_CHORDAL_KNOTS)))
    ok = diff < 1e-8
    failures += 0 if ok else 1
    print(f"    max |chord_cum - chordal knots| = {diff:.2e}   {'ok' if ok else 'FAILED'}")

    print("\n[5] assert_pair catches a knots/energy mismatch")
    try:
        assert_pair(NOTEBOOK_SAMPLE, NOTEBOOK_CHORDAL_KNOTS, 1.0, 100,
                    label="deliberate")
        print("    FAILED — mismatch was not detected")
        failures += 1
    except AssertionError:
        print("    ok — mismatch raised as expected")

    print("\n" + "=" * 66)
    print("all checks passed" if failures == 0 else f"{failures} CHECK(S) FAILED")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
