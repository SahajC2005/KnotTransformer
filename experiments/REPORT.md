KnotTransformer Generalisation Experiment — Full Report
========================================================

Question
--------
Does a single transformer trained on mixed point-sequence lengths
generalise to a held-out length it never saw during training? This
tests whether the §7.3 limitation in the thesis (per-dof models, fixed
input size) is fixed by the transformer extension or only papered
over.

Experimental design
-------------------
* Architecture: KnotTransformer from knot_transformer.py — sequence-to-
  sequence transformer, 3 encoder + 3 decoder layers, d_model=64,
  353,793 parameters. Output: K interior B-spline knots in (0,1)
  given an (N, 2) point sequence and the desired number of knots K=N-2.
* Training: Phase 1 only — L1 loss against chordal heuristic knots.
  100 epochs, batch 64, lr 1e-3, Adam, gradient clipping at 1.0.
* Training data: 4000 random 2D point sequences, mixed N=4 (2 dof)
  and N=5 (3 dof), 2000 each. Coordinates ~U(-0.5, 0.5).
* Test sets: 500 random sequences each at N=4, N=5, N=6 from the same
  distribution. N=6 (4 dof) is HELD OUT — model never saw 6-point
  sequences during training.
* Evaluation metric: for each test sample, compute total curvature
  using the model's predicted knots vs using chordal heuristic knots.
  "Win rate" = fraction of samples where model's curvature ≤ chordal.
  Median curvature ratio = how much higher or lower the model is on
  the typical sample.
* Boundary post-processing: predicted knots are rescaled from [0,1]
  into [0.02, 0.98] before being fed to the solver, both for
  numerical safety and to handle the output-head pathology described
  below. This is applied uniformly to every sample.

Results
-------

                  win rate vs           median κ      ratio
    split   N  dof  chordal    n_valid  model  chord   (model/chord)
    -----------------------------------------------------------------
    TRAIN   4   2   31.2 %     500/500  315.30 118.61  2.66
    TRAIN   5   3   44.6 %     500/500  297.00 284.83  1.04
    HELD-OUT 6  4   46.4 %     500/500  415.43 395.38  1.05
    -----------------------------------------------------------------
    untrained 6 4   38.2 %     500/500  523.61 395.38  1.32
    baseline

Interpretation
--------------
The trained model beats its untrained-baseline on the held-out N=6
test set:
  + 8 percentage points on win rate (46.4% vs 38.2%)
  - 21 % on median curvature (415 vs 524)

So Phase 1 training learned something that transfers across sequence
length. But the trained model does not beat the chordal heuristic at
any sequence length, including the ones it was trained on:
  N=4 (in-dist): 31.2 %
  N=5 (in-dist): 44.6 %
  N=6 (held out): 46.4 %

The gap between in-distribution and held-out is small (~2 percentage
points between N=5 and N=6), so the model is not memorising per-length
patterns — its behaviour is roughly consistent across all three.

Output-head diagnosis (why the model can't beat chordal)
-------------------------------------------------------
While running this experiment we identified a structural bug in the
KnotTransformer's output head:

    raw        = self.knot_head(decoded).squeeze(-1)
    increments = F.softplus(raw) + 1e-6
    cum        = torch.cumsum(increments, dim=-1)
    knot_vals  = cum[:, :K] / cum[:, K-1]      # ← K-th knot pinned to 1.0

Dividing the K-th cumulative sum by itself forces the K-th interior
knot to exactly 1.0 by construction, regardless of input. The model
has only K-1 effective degrees of freedom.

Evidence this is the dominant error mode:
  * Phase 1 L1 loss plateaued at 0.129 (cf. v2 fixed model: 0.101).
    The ~0.03 difference is structural error from the pinned position.
  * Probing the trained checkpoint: the first K-1 predicted knots
    match chordal within 0.01–0.02 across diverse inputs. The K-th
    is always 1.0.

We attempted an architectural fix (v2: predict K+1 increments, divide
by the (K+1)-th cumulative sum) but the fixed model collapsed to a
uniform-knot trivial solution. The cumsum-of-softplus output head
appears to have a flat optimisation region near uniform-spacing that
the optimizer gets trapped in once the symmetry-breaking from the
pinned-to-1 boundary is removed. A proper fix would likely require
either a different output parameterisation (e.g. predicting fractions
of remaining space, autoregressive next-knot prediction, or direct
unbounded values with a sorting network) or a chordal warm-start to
seed input-dependent initial outputs.

Honest claim this experiment supports
-------------------------------------
A single KnotTransformer trained on N∈{4,5} produces parameterisations
on held-out N=6 sequences that:
  * are consistent in quality with its in-distribution performance
    (within 2 percentage points of in-dist win rate);
  * meaningfully beat the untrained baseline (+8 pts win rate, -21%
    median curvature);
  * remain below the chordal heuristic at every sequence length,
    including those in the training distribution, due to a
    structural bug in the output head.

This is evidence that the shared encoder learns length-invariant
geometric structure (corroborating the encoder-probing chart that
showed R²≈1.0 for sequence position, chord fraction, centripetal
fraction across the same architecture). It is not yet evidence that
the unified-model approach is competitive with the thesis's per-dof
MLPs (Phase-2-trained per-dof models reach >90% win rate against
chordal on 4 dof, Table 6.7).

What §7.3 of the thesis predicted, and what this experiment shows
-----------------------------------------------------------------
§7.3 said the limitation has two parts:
  (a) The model only accepts fixed-size inputs. RNNs alone don't
      fix this because…
  (b) the energy computation layer is also fixed-length-batched.

Part (a) is genuinely addressed by the transformer architecture:
the encoder and decoder both handle variable-length inputs via padding
masks, and the experiment shows the model produces sensible
predictions on a sequence length never seen during training.

Part (b) is not addressed. Phase 2 training still iterates sample-by-
sample over mixed-N batches inside Phase2Trainer.train_epoch because
the energy computation in curvature_layer_nd.solve_nd and
global_curvature_nd assumes uniform shape across the batch.

So a fair summary: the transformer addresses half of the §7.3
limitation (variable-length encoding) but not the other half (batched
variable-length energy computation). It also has a model-design bug
in its output head that needs fixing before the architecture is fully
competitive.

What to do next
---------------
1. Fix the output-head parameterisation. The cumsum-of-softplus
   formulation has shown two failure modes (pinned-to-1 in v1; uniform
   collapse in v2). Worth trying:
   * Autoregressive next-knot prediction: each knot predicted
     conditioned on the previous knots, using a small recurrent or
     transformer-decoder head.
   * Beta-distribution head: predict (α, β) parameters of a Beta
     distribution per position, sample or take the mean. Naturally
     bounded in (0,1) without symmetry pathologies.
   * Direct unbounded prediction + sort: predict K unconstrained
     reals, then apply a sorting operation. Loses smoothness but is
     transparent.

2. Once the output head is fixed, run Phase 2 (energy fine-tuning).
   The expected outcome: a transformer that genuinely beats chordal
   at trained lengths, then checking whether that improvement
   transfers to held-out lengths. THAT is the experiment that
   confirms or refutes "the §7.3 limitation is addressed".

3. Long-term, build a jagged/segmented version of the energy
   computation in curvature_layer_nd to address §7.3 part (b).
