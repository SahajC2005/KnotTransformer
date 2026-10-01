# KnotTransformer

**Length-Generalizing Neural Parameterization for B-Spline Curve Interpolation**

A single encoder–decoder Transformer that predicts curvature-minimizing knot vectors for B-spline interpolation, trained on sequence lengths N∈{4,5} and generalizing to unseen lengths N=6,7,8 — addressing the fixed-input-size limitation of prior per-length MLP approaches.

---

## The Problem in One Sentence

Given a handful of ordered points, infinitely many smooth B-spline curves pass through them. The only thing we choose is the curve's internal pacing — the **knot vector** — and we want the pacing that gives the **least-bendy curve**. A neural network learns to pick that pacing instantly, grading its own curves with no answer key.

---

## Key Results

| Method | Win rate vs chordal | Median κ ratio |
|---|---|---|
| Chordal heuristic (baseline) | — | 1.000 |
| **KnotTransformer** (1 forward pass) | **75.7% ± 4.0%** | **0.295** |
| KnotTransformer + refinement (200 steps) | 96.0% | 0.140 |
| Best random restart (15 starts) | 100.0% | 0.114 |

**Hard-test results** (chordal κ > 1000, multi-seed mean ± std):

| N | Split | Win rate | Curvature ratio |
|---|---|---|---|
| 4 | Train | 0.934 ± 0.008 | 0.021 ± 0.008 |
| 5 | Train | 0.915 ± 0.013 | 0.056 ± 0.003 |
| 6 | **Held-out** | **0.904 ± 0.016** | **0.054 ± 0.002** |

On hard cases, the model produces curvature **2–6% of chordal's**, winning ~90% of the time.

**Length extrapolation** (trained on N∈{4,5} only):

| N | Split | Win rate | Ratio |
|---|---|---|---|
| 6 | Held-out (1 step) | 68.0% | 0.306 |
| 7 | Far held-out (2 steps) | 67.8% | 0.385 |
| 8 | Far held-out (3 steps) | 67.4% | 0.423 |

Performance degrades gracefully — never collapses. N=8 uses decoder query slots never exercised during training.

---

## Architecture

A DETR-style encoder–decoder Transformer with **353,793 parameters**:

```
Points (N×2)
  → Feature extraction (6 features per point)
  → Linear projection (6→64) + positional embedding
  → Encoder (3 layers self-attention, d_model=64, 4 heads)
  → Decoder (3 layers cross-attention, 7 learned knot queries)
  → Knot head (softplus → cumsum → normalize → ordered knots in (0,1))
  → Differentiable B-spline solve → Curvature energy (the loss)
```

The entire pipeline from points to curvature energy is differentiable, including implicit differentiation through the linear solve (∂P/∂u = −A⁻¹(∂A/∂u)P), enabling end-to-end self-supervised training.

---

## Training

**Phase 1 (supervised pretraining):** L1 loss against the chordal heuristic. 100 epochs, lr=1e-3. Provides a sensible initialization.

**Phase 2 (self-supervised fine-tuning):** Direct curvature minimization — no labels needed. The loss is the total bending energy of the curve the model produces. 300 epochs, lr=1e-5, energy cap=1000, batch=128.

**Key stabilization techniques:**
- Energy capping at 1000 (prevents outlier-dominated gradients)
- Chord-filtered training set (samples with chordal κ < 1000)
- Gradient clipping at 1.0
- N-grouped batched solver (partial solution to variable-length batched energy computation)

---

## Repository Structure

```
KnotTransformer/
├── README.md                           ← you are here
├── requirements.txt                    ← Python dependencies
│
├── src/                                ← core source code
│   ├── knot_transformer.py             ← 2D KnotTransformer model + trainers
│   ├── knot_transformer_nd.py          ← nD extension (2D, 3D, any dimension)
│   ├── curvature_layer_nd.py           ← differentiable B-spline solver (nD)
│   ├── curvatureLayer.py               ← backward-compat wrapper for thesis code
│   ├── generalization_experiment.py    ← Phase 1 generalization experiment
│   └── interpretability.py             ← 5 interpretability tools (attention, probing, etc.)
│
├── notebooks/                          ← Colab/Jupyter notebooks
│   ├── KnotTransformer_Phase2_v2.ipynb ← hardened GPU training notebook (recommended)
│   └── RRRReeeaaaalllll_B_spline.ipynb ← full experiment notebook
│
├── experiments/                        ← training & evaluation scripts
│   ├── train_phase2_longer.py          ← resumable CPU/local training script
│   ├── evaluate_phase2_final.py        ← final 500-sample evaluation
│   ├── REPORT.md                       ← experiment report
│   └── results/                        ← saved evaluation JSON files
│       ├── eval_dof2.json
│       ├── eval_dof3.json
│       ├── eval_dof4.json
│       └── final_summary.txt
│
├── paper/                              ← LaTeX paper source
│   └── main.tex                        ← complete paper with TikZ diagrams
│
├── web/                                ← interactive visualizations
│   ├── transformer_deep_dive.html      ← full transformer mechanics (matrices, attention)
│   ├── knot_transformer_pipeline.html  ← 5-step pipeline walkthrough
│   └── knot_transformer_explainer.html ← B-spline + curvature interactive demo
│
├── docs/                               ← documentation
│   └── SETUP.md                        ← environment setup guide
│
└── data/                               ← data files (see note below)
    └── .gitkeep
```

> **Note on large files:** Checkpoint files (`.pt`, ~3–8 MB each) and `dataset_1k.pt` (~1 MB) are not included in the repository. They are stored on Google Drive. See [Checkpoints](#checkpoints) below for download links.

---

## Quick Start

### Option A: Google Colab (recommended for GPU training)

1. Open `notebooks/KnotTransformer_Phase2_v2.ipynb` in [Google Colab](https://colab.research.google.com)
2. Runtime → Change runtime type → GPU (L4 recommended)
3. Upload `src/` files when prompted (or use the colab_bundle.zip)
4. Run cells in order. Training takes ~1.5–3 hours on L4 GPU.

### Option B: Local setup

```bash
# Clone the repo
git clone https://github.com/[username]/KnotTransformer.git
cd KnotTransformer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Mac/Linux
# venv\Scripts\activate   # Windows

# Install dependencies
pip install torch  # add --index-url for CUDA version
pip install -r requirements.txt

# Smoke test
python src/knot_transformer.py      # 2D model test
python src/curvature_layer_nd.py    # B-spline solver test

# Run Phase 1 generalization experiment
python src/generalization_experiment.py

# Train Phase 2 (CPU, ~3+ hours for 300 epochs)
python experiments/train_phase2_longer.py
```

---

## Interactive Visualizations

Open any of these HTML files directly in your browser — no server needed:

| File | What it shows |
|---|---|
| `web/transformer_deep_dive.html` | **Every matrix multiply** in the attention mechanism — Q, K, V projections, scores, softmax, weighted sum. Clickable cells show exact dot-product computations. Interactive multi-head and layer-stacking diagrams. |
| `web/knot_transformer_pipeline.html` | **5-step pipeline walkthrough** — input tokenization, encoder self-attention, decoder cross-attention, knot head output, full gradient chain. Click tokens to see attention patterns. |
| `web/knot_transformer_explainer.html` | **B-spline fundamentals** — drag knot sliders to reshape the curve through fixed points. See control points move, curvature change. Presets for uniform/chordal/optimized. |

---

## Mathematical Background

### The interpolation constraint (fit is free)

For N data points and a knot vector **u**, the clamped cubic B-spline's control points are determined by a linear solve:

**A(u) P = D̃**

where A is the (N+2)×(N+2) collocation matrix built from Cox–de Boor basis functions. For *any* valid knot vector, this system has a unique solution — every choice of knots produces a curve that passes exactly through all data points. There is no fit-vs-smoothness trade-off.

### The energy objective

The total bending energy is:

**E(u) = ∫₀¹ κ(t)² ‖C'(t)‖ dt**

where κ(t) = |x'y'' − y'x''| / (x'² + y'²)^(3/2) is the curvature at parameter t.

### The self-supervised loss

**L(θ) = (1/N) Σᵢ E(f_θ(Dᵢ))**

The network predicts knots, the solver builds a curve, and the curve's own curvature is the loss. No labels needed.

### Gradient through the solver

**∂P/∂uₖ = −A⁻¹ (∂A/∂uₖ) P**

Implicit differentiation through the linear solve enables end-to-end gradient flow from curvature back to the transformer weights.

---

## Checkpoints

Trained checkpoints are stored on Google Drive (too large for GitHub without LFS):

| File | Description | Size |
|---|---|---|
| `phase1_generalization.pt` | Phase 1 checkpoint (chordal pretraining) | 2.8 MB |
| `phase2_best_holdout.pt` | Best Phase 2 checkpoint by held-out metric (seed 25) | 2.7 MB |
| `dataset_1k.pt` | Chord-filtered training set (3,187 samples) | 0.9 MB |
| `seed_20260526/phase2_best_holdout.pt` | Seed 26 checkpoint | 2.7 MB |
| `seed_20260527/phase2_best_holdout.pt` | Seed 27 checkpoint | 2.7 MB |

To use locally, place checkpoint files in the `data/` directory.

---

## Reproducibility

Three independent seeds (20260525, 20260526, 20260527), each trained 300 epochs from the same Phase 1 checkpoint:

| Seed | Win rate (N=6) | Best ratio | Best epoch |
|---|---|---|---|
| 20260525 | 72.0% | 0.275 | 285 |
| 20260526 | 75.0% | ~0.228 | 285 |
| 20260527 | 80.0% | ~0.228 | 285 |
| **Mean ± std** | **75.7% ± 4.0%** | | |

---

## Local Minimum Analysis

The model performs amortized optimization — one forward pass that approximates the solution of the curvature minimization problem. Analysis of 100 held-out samples:

| Method | Median κ | vs chordal | vs model |
|---|---|---|---|
| Chordal (baseline) | 133.8 | 1.000 | — |
| Transformer (1 forward pass, ~1ms) | 86.7 | 0.648 | 1.000 |
| Transformer + 200 steps refinement (~2s) | 18.7 | 0.140 | 0.216 |
| Best of 15 random restarts (~15s) | 15.3 | 0.114 | 0.176 |

The transformer lands in high-quality basins (refined/best = 0.82) but not at the basin floors (refined/model = 0.22). The **hybrid approach** (model + gradient refinement) achieves 96% win rate with curvature 14% of baseline — the recommended production method.

---

## Known Limitations

- **Output-head constraint:** The cumsum normalization reduces effective degrees of freedom by ~1. Mitigated by boundary rescaling; architectural fix deferred to future work.
- **2D experiments only:** The solver supports nD; 3D experiments are in progress.
- **Random uniform data:** No real-world geometric data tested.
- **Single baseline:** Compared against chordal only; centripetal and optimal baselines pending.

---

## Extending This Work

Several promising directions:

1. **Transformer + refinement hybrid** — use the model as a fast initializer, polish with gradient descent. Already demonstrated: 96% win rate at 14% of baseline curvature.
2. **3D extension** — the `curvature_layer_nd.py` module supports arbitrary dimension.
3. **Real-world data** — CAD outlines, handwriting strokes, robot trajectories.
4. **Output-head fix** — autoregressive or beta-distribution parameterization.
5. **Multi-energy training** — simultaneously minimize curvature and arc length.

---

## Citation

If you use this work, please cite:

```bibtex
@misc{knottransformer2026,
  title={Length-Generalizing Neural Parameterization for B-Spline Curve Interpolation},
  author={Chhetri, Sahaj and Chhetri, Vinai},
  year={2026},
  note={Extends the self-supervised B-spline parameterization framework of Chhetri (2024)}
}
```

---

## Acknowledgments

This work extends the self-supervised B-spline parameterization framework developed by Vinai Chhetri in their Master's thesis. The differentiable energy layer, two-phase training pipeline, and curvature/arc-length objectives are from that work. Our contribution is the variable-length Transformer architecture and the length-generalization experiments.

---

## License

MIT License. See [LICENSE](LICENSE) for details.
