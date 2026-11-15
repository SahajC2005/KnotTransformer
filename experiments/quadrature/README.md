# Quadrature resolution — does `num_pts` change the answer?

`cl.global_curvature` integrates `κ²·speed` on a **uniform** grid of `num_pts` samples.
The repository uses two settings and never checked that either had converged:

| Setting | Where |
|---|---|
| `num_pts = 100` | every evaluation path — `compute_curvature_batch`, `full_eval_gpu`, `evaluate_holdout_gpu`, the L-BFGS refiner |
| `N_QUAD = 50` | Phase 2 training in `notebooks/Phase2_BufferBox.ipynb` |

## Scripts

```bash
# regression fixture for the solver — run this first
python experiments/quad_common.py

# one sample, seconds
python experiments/quadrature_sweep.py --mode single

# the stored N=6 evaluation set, chordal vs centripetal, no checkpoint needed
python experiments/quadrature_sweep.py --mode population --n-samples 100

# the question that matters, if you have a Phase 2 checkpoint
python experiments/quadrature_sweep.py --mode population \
    --method model --ckpt data/phase2_best_holdout.pt --n-samples 500

# corrected landscape figure (replaces notebook cells 42 and 43)
python experiments/energy_landscape.py --grid 64 --num-pts 2000
python experiments/energy_landscape.py --grid 64 --num-pts 100   # the aliased surface
```

`experiments/quad_common.py` holds the shared solver wrappers. It calls `cl.solve` and
`cl.global_curvature` directly, so these measurements are about the shipped code rather
than a reimplementation.

## What has been measured so far

**Status:** the single-sample numbers below are measured and reproducible. The
**population sweep has not been run** — there is no Python interpreter on the machine
where these scripts were written, so `quadrature_sweep.py --mode population` is
untested and its result is the open question, not a finding.

The numbers come from an independent JavaScript port of `curvature_layer_nd.py`,
validated against the notebook's own stored output:

| Knot vector | Port, `num_pts=100` | Notebook (cell 42) | Interp. residual |
|---|---|---|---|
| Chordal | 711.5194 | 711.51 | 3.7e-9 |
| Model | 68.4229 | 68.42 | 4.1e-8 |

Matching to five figures means the natural end-condition rows, the half-open-interval
case at `t = 1` and both derivative recursions are reproduced correctly.

### Convergence on the cell-42 sample (N = 6, chordal κ > 500)

Same two knot vectors, resolution swept, using the repository's own quadrature formula:

| `num_pts` | Chordal κ | Model κ | Ratio | Used by |
|---:|---:|---:|---:|---|
| **50** | 4936.79 | 16.09 | **0.0033** | `N_QUAD`, training |
| **100** | 711.52 | 68.42 | **0.0962** | every eval path |
| 200 | 2607.56 | 72.44 | 0.0278 | |
| 400 | 2959.49 | 161.68 | 0.0546 | |
| 800 | 2862.09 | 137.17 | 0.0479 | |
| 1600 | 2855.38 | 139.18 | 0.0487 | |
| 3200 | 2855.37 | 139.18 | 0.0487 | |
| **converged** | **2855.37** | **139.18** | **0.0487** | the actual integral |

### Why

The integrand peaks at **6.3 × 10⁵** at `t = 0.570` on the chordal curve while its
median is about **1** — three orders of magnitude of dynamic range inside a spike
narrower than the grid spacing. At `num_pts = 100`, `Δt = 0.0101`. Which samples land
near the peak determines the answer, so the number is an aliasing artifact rather than
an approximation. The model's integrand peaks at 3.2 × 10⁴ at `t = 0.0275`.

### Consequences on this sample

1. **Absolute κ is wrong by 2–12×.** The evaluation setting reports 711.5 against a
   true 2855.4 for chordal, and 68.4 against 139.2 for the model.
2. **The ratio is wrong by about 2×.** 0.0962 at `num_pts = 100` against a converged
   0.0487. The ratio is the project's headline metric.
3. **The training objective is a different function from the evaluation metric.** At
   `N_QUAD = 50` the two curves read 4936.8 and 16.1 — ratio 0.0033, off by a factor
   of 15, and wrong in the *opposite direction* for each curve. Phase 2 was not
   minimising a noisy version of the eval metric; it was minimising a different
   quantity.

Separately, the endpoint weights in `(h/3)·(4·S1 + 2·S2)` give the two end samples
weight 2 instead of 1. That error is about 1% and is the least of the problems here.

## Scope — read before quoting any of this

* This is **one** sample, and cell 42 selected it *because* it was hard
  (`chord κ > 500`). It is the worst case, not the typical one.
* Medians over 500 samples are more robust than any single value. Aliasing error that
  is roughly symmetric across a population partly cancels in a median. **Whether it
  does here is exactly what `--mode population` measures, and that has not been run.**

Read the population output this way:

* win rate and ratio **flat** across resolutions → the per-sample aliasing cancels in
  the median, and the published tables stand as *comparisons* even though the absolute
  κ values are wrong;
* win rate or ratio **moves** → every ratio in the repository is resolution-dependent
  and needs restating at a converged setting before the paper quotes any of them.

## The marker bug in cells 42 and 43

`find_local_min_robust` returns an improved κ paired with the wrong knot vector:

```python
best_k_this = init.copy()
...
    if k_eval < best_kappa_this:
        best_kappa_this = k_eval
        best_k = knots.detach().numpy().copy()   # writes best_k
if best_kappa_this < best_overall_kappa:
    best_overall_kappa = best_kappa_this
    best_overall_k     = best_k_this             # reads best_k_this
```

`best_k` is a dead store and `best_k_this` is never updated after initialisation, so
the returned knots are the initial **random** start. Those knots then set `k1_fixed`
and `k4_fixed`, which define the slice the heatmap is computed on — so
`curvature_landscape_v2.png` is a slice through an arbitrary point with a mismarked
optimum. One-line fix: `best_k_this = knots.detach().numpy().copy()`.

`energy_landscape.py` keeps each (knots, κ) pair together and runs every reported pair
through `qc.assert_pair`, which re-derives κ from the knots it is about to plot. A
regression cannot pass silently.

## The device global

Cell 42 sets `cl.device = DEVICE` (CUDA) and then builds CPU tensors. `construct`
allocates its temporaries on `cl.device`, they meet CPU inputs, a `RuntimeError` is
raised, and `except Exception: break` swallows it on the first iteration — which is why
that cell's stored output reports `curvature : inf` with local-min knots identical to
the model's. Cell 45's trace *does* show real L-BFGS improvements, so it ran in a
session where `cl.device` was CPU: **the validity of that experiment depends on which
cells were run first.**

`qc.energy` pins `cl.device` to the device of the tensors it is about to build, on
every call, so the solver cannot be handed a mismatched pair. The root fix is still
worth doing in `src/`: derive the device from the inputs (`dev = knots.device`)
throughout `construct`, `de_boor`, `solve_nd` and the derivative functions, which would
also delete the cell-18 monkey-patch.

## Suggested fix for the solver

The integrand is smooth between spikes, so ~2,000 uniform points converges and costs
about 20× one solver pass — nothing next to training. For the training loop, where cost
does matter, integrate on a grid refined near the knots, since that is where the
`3/Δu` factor in the derivative recursion puts the spikes.
