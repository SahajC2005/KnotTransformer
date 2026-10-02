# Local-minimum analysis — L-BFGS refinement

Notebook: `notebooks/Phase2_BufferBox.ipynb`, "Local-minimum analysis (L-BFGS refinement)" section
Checkpoint: seed 20260527 `phase2_best_holdout.pt`
Samples: 100 held-out N=6, of which 97 were valid (3 skipped). Runtime: 32.8 min (20.3 s/sample)

## Results (median curvature)

| Method | Win rate | Median κ | Ratio vs chordal | Ratio vs model |
|---|---|---|---|---|
| Chordal (baseline) | — | 396.3 | 1.000 | — |
| Transformer (1 forward pass) | 79.4% | 89.6 | 0.226 | 1.000 |
| Transformer + L-BFGS | 80.4% | 86.2 | 0.217 | 0.962 |
| Best random restart | 90.7% | 130.6 | 0.329 | 1.457 |

L-BFGS improvement over the raw model: median 0.0%, mean 6.1%, max 99.1%. For 81.4% of
samples the improvement is under 5%.

## Interpretation

- L-BFGS barely moves the model's knots (refined/model = 0.96). The model's output already
  sits at, or very near, a local minimum.
- Random restarts win more often (90.7%) but have a *higher* median κ than the model.
  Restarts avoid the model's rare catastrophic cases (for example, sample 70: model
  κ = 3703 vs chordal κ = 234.5) but usually land in worse basins.

## ⚠ Inconsistent with the README table

The top-level README reports "Transformer + 200 steps refinement: ratio 0.140, 96% win",
and "best of 15 random restarts: ratio 0.114, 100% win". That table used Adam refinement,
a different sample set (median chordal κ 133.8 vs 396.3 here) and a different restart
budget. These numbers are not comparable. Pick one protocol before the paper reports either.

Output JSON (on Drive): `knot_transformer_ckpts/local_min_analysis_lbfgs.json`. Copy it
into this folder.
