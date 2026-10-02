# Real-world transfer — road-like fine-tuning and real GPS turns

The base model was trained on uniform random point clouds. On real road geometry it
predicts near-coincident (pathological) knots. These experiments try to close that gap.

Data generator: `src/road_generator.py`. Each sample is a straight approach, then a
circular bend (20°–150°, radius 0.15–0.6), then a straight exit, with GPS-style jitter.
Waypoints are sampled by arc length and normalized to the model's box.

Common setup: warm-start from seed 20260527 `phase2_best_holdout.pt`; 3,000 road samples
over N ∈ {6, 7, 8}; 40 epochs; lr 3e-5.

## Run 1 — road fine-tune (`notebooks/Phase2_RoadFinetune.ipynb`)

| | Win vs chordal (real turns) | Median ratio |
|---|---|---|
| Before fine-tune | 13.3% | 9.637 |
| After fine-tune (best-on-real ckpt) | 26.7% | 1.661 |

The model is still worse than chordal on real turns.

**Caveats:**
- Only **15** real turns.
- The checkpoint was selected *and* reported on the same 15 turns (winner's curse).
- Dropout was active during evaluation (missing `model.eval()`). This is why the logged
  epoch-30 ratio (1.226) differs from the reloaded best (1.661).

## Run 2 — road fine-tune + buffer-box penalty (`notebooks/Phase2_BufferBox.ipynb`)

Same setup, plus `losses.buffer_box_penalty` (BOX_MARGIN 0.25, LAMBDA_BOX 200, N_QUAD 50).

| | Win vs chordal (real turns) | Median ratio |
|---|---|---|
| Before fine-tune | 13.3% | 9.637 |
| After fine-tune | **0.0%** | 7.421 |

**Negative result.** With λ=200, the penalty dominates the loss: road train κ stalls
around 590–630, vs 254 in Run 1. Real-turn performance does not recover. If you revisit
this, try a smaller λ or ramp λ up over training.

## Run 3 — trip-split real turns (`notebooks/Phase2_RealTurns_TripSplit.ipynb`) — **not yet run**

This run fixes the problems in Runs 1–2:

1. Calls `model.eval()` during evaluation (dropout bug fix)
2. Extracts many turns per trip (~560 available), not one
3. Splits **by trip** into train/val/test, so no trip appears in more than one split
4. Mixes real training turns into the road pool (oversampled 3×)
5. Early-stops on val (patience 5); reports on test with bootstrap CIs

The fine-tune cell has no outputs yet. Fill in this section after running it.
