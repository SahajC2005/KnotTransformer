# Airfoil benchmark — UIUC Airfoil Coordinates Database

Notebook: `notebooks/Airfoil_UIUC_Eval.ipynb`

> **Important:** these numbers compare the chordal heuristic against a **per-instance
> Nelder-Mead optimizer** of the bending energy. They are an *upper bound* for what a
> knot-prediction model could achieve on this data. They are **not** KnotTransformer
> results. To benchmark the trained model, replace the body of `optimize_knots` (the
> `MODEL PREDICTION HOOK`) with a call to the model.

## Setup

- Data: 1,433 UIUC airfoils, resampled to 201 points each (Selig ordering), from the
  `npuljc/Airfoil_preprocessing` mirror. The notebook clones this repo; it is gitignored.
- Sample: 30 airfoils (`random.seed(42)`, first 30 of a 40-sample, sorted).
- Sequences per airfoil: lengths L ∈ {6, 7, 8} × two tiers:
  - **smooth**: evenly spaced around the whole contour
  - **hard**: unevenly spaced, concentrated around the leading edge (sharp curvature)
- Fit: interpolating cubic B-spline (`scipy.make_interp_spline`, k=3)
- Metric: bending energy E = ∫ κ² |C'| du; ratio = E_optimized / E_chordal (lower is better)
- Total: 180 sequence evaluations

## Results

| L | Tier | n | Win % | Median ratio | Mean ratio |
|---|---|---|---|---|---|
| 6 | smooth | 30 | 100.0 | 0.714 | 0.721 |
| 6 | hard | 30 | 100.0 | 0.382 | 0.370 |
| 7 | smooth | 30 | 100.0 | 0.198 | 0.235 |
| 7 | hard | 30 | 96.7 | 0.512 | 0.498 |
| 8 | smooth | 30 | 100.0 | 0.939 | 0.942 |
| 8 | hard | 30 | 96.7 | 0.710 | 0.701 |
| all | hard only | 90 | 97.8 | 0.536 | — |
| all | overall | 180 | 98.9 | 0.676 | — |

## Figures

- `curvature_ratio_by_length.png`: median ratio by length and tier
- `example_spline_comparison_combs.png`: hard-tier, L=7 example with curvature combs
  on a shared scale (chordal vs optimized)

The per-sequence CSVs (`airfoil_bspline_results.csv`, `airfoil_bspline_summary.csv`)
are written by the notebook. Add them here after the next run.
