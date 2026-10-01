# KnotTransformer — Setup Guide

## Files in this project

| File | Purpose |
|---|---|
| `knot_transformer.py` | Original 2D transformer model + Phase 1 & 2 trainers |
| `knot_transformer_nd.py` | Extended nD transformer (2D, 3D, any spatial dim) |
| `curvature_layer_nd.py` | Differentiable B-spline solver for any spatial dimension |
| `interpretability.py` | Five interpretability tools (attention, probing, ablation, sensitivity, latent) |
| `requirements.txt` | All Python dependencies |

---

## 1. Python version

Python **3.9, 3.10, or 3.11** recommended. Check yours:

```bash
python --version
```

---

## 2. Create a virtual environment (recommended)

```bash
# In your project folder
python -m venv venv

# Activate — Windows
venv\Scripts\activate

# Activate — Mac / Linux
source venv/bin/activate
```

In VSCode: open the Command Palette (`Ctrl+Shift+P`), type
`Python: Select Interpreter`, and choose the `venv` interpreter.

---

## 3. Install PyTorch

**Check if you have a GPU first:**

```bash
# Windows / Linux
nvidia-smi

# Mac (Apple Silicon — use CPU install below)
```

**GPU (CUDA 11.8) — fastest Phase 2 training:**

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

**GPU (CUDA 12.1):**

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

**CPU only (no GPU):**

```bash
pip install torch
```

---

## 4. Install all other dependencies

```bash
pip install -r requirements.txt
```

---

## 5. Verify installation

```bash
python -c "
import torch, numpy, matplotlib, sklearn, scipy
print('torch      ', torch.__version__)
print('numpy      ', numpy.__version__)
print('matplotlib ', matplotlib.__version__)
print('sklearn    ', sklearn.__version__)
print('scipy      ', scipy.__version__)
print('GPU available:', torch.cuda.is_available())
"
```

---

## 6. Quick smoke test

```bash
# Test 2D transformer
python knot_transformer.py

# Test nD transformer (2D, 3D, 5D)
python knot_transformer_nd.py

# Test nD B-spline solver
python curvature_layer_nd.py

# Test interpretability tools
python interpretability.py
```

---

## 7. Connecting to your existing thesis notebooks

Your existing `UNSUP_2D.ipynb`, `UNSUP-3D.ipynb`, and `Copy_of_UNSUP-4D.ipynb`
use `curvatureLayer.py` and `npfile.py`. Place those files in the same
folder as the new files. The new `curvature_layer_nd.py` is fully backward
compatible — `solve`, `global_curvature`, and `arc` all work identically.

```python
# Drop-in replacement in your existing notebooks:
import curvature_layer_nd as curvatureLayer   # same API, adds nD support
```

---

## 8. Common issues

**`ModuleNotFoundError: No module named 'umap'`**
UMAP is optional. Tool 5 falls back to PCA automatically if umap-learn
is not installed. To install it: `pip install umap-learn`

**`CUDA out of memory` during Phase 2**
Reduce batch size in `Phase2TrainerND(batch_size=...)`. The thesis
found large batches (512+) are needed for stability but you can start
smaller and increase.

**`torch.linalg.inv_ex` returns non-zero `info`**
Degenerate knot vector (duplicate interior knots). The `dup()` function
in `curvature_layer_nd.py` filters these out before training — make
sure you run it after Phase 1.

**`umap-learn` install fails on Windows**
Try: `pip install umap-learn --no-build-isolation`
Or use the PCA fallback (Tool 5 handles this automatically).
