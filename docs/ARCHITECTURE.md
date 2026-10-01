# KnotTransformer Architecture

Detailed component-by-component description of the model.

---

## Overview

| Component | Parameters | Purpose |
|---|---|---|
| Input projection | 448 | Lift 6-d features to 64-d working space |
| Positional embedding | 640 | Encode point ordering (max 10 positions) |
| Encoder (3 layers) | 149,184 | Self-attention: points learn about each other |
| Knot queries | 448 | 7 learned vectors, one per possible knot |
| Decoder (3 layers) | 198,720 | Cross-attention: queries extract knots from encoded points |
| Knot head MLP | 2,145 | Convert 64-d vectors to raw scalar scores |
| Other (biases, norms) | ~2,208 | LayerNorm parameters across all layers |
| **Total** | **353,793** | |

---

## Hyperparameters

```python
d_model            = 64      # working dimension
nhead              = 4       # attention heads (d_k = 16 per head)
num_encoder_layers = 3
num_decoder_layers = 3
dim_feedforward    = 256     # FFN hidden dimension (4× d_model)
dropout            = 0.1
max_points         = 10      # maximum input sequence length
max_knots          = 6       # maximum interior knots (K_max)
```

---

## 1. Input Features (6 per point)

Each data point Dₖ = (x, y) is augmented with local geometric context:

| # | Feature | Description |
|---|---|---|
| 1 | x | x-coordinate |
| 2 | y | y-coordinate |
| 3 | d_prev | Distance to previous point (0 for first) |
| 4 | d_next | Distance to next point (0 for last) |
| 5 | θ | Turning angle at this point (radians) |
| 6 | k/N | Normalized position in sequence |

---

## 2. Encoder Layer (×3)

Each layer:
1. **Multi-head self-attention** (4 heads, d_k=16)
   - Q, K, V projections: 3 × (64×64) = 12,288 params
   - Output projection W_O: 64×64 = 4,096 params
2. **Residual + LayerNorm** (128 params)
3. **Feed-forward network** (64→256→64, GELU activation, 33,088 params)
4. **Residual + LayerNorm** (128 params)

Total per layer: 49,728 parameters.

**Padding masks** ensure shorter sequences ignore padded positions in the attention computation.

---

## 3. Decoder Layer (×3)

Each layer:
1. **Self-attention among queries** (queries coordinate with each other)
   - Same structure as encoder self-attention: 16,512 params
2. **Cross-attention to encoder memory**
   - Q from queries, K and V from encoder output: 16,512 params
3. **Feed-forward network** (64→256→64): 33,216 params

Total per layer: 66,240 parameters.

---

## 4. Knot Queries

7 learnable vectors of dimension 64 (448 parameters). Each query learns to represent "I am knot position k."

For a sample with K = N-2 interior knots, only the first K+1 queries are active (K knots + 1 slack). The slack query absorbs the remaining parameter range between the last knot and 1.0.

---

## 5. Monotonic Output Head

```
decoded query (64-d)
  → Linear(64, 32) + GELU
  → Linear(32, 1)
  → raw score rᵢ
  → sᵢ = softplus(rᵢ) + ε     (positive)
  → cᵢ = cumsum(s)              (increasing)
  → uᵢ = cᵢ / c_{K+1}          (normalized to (0,1))
```

At evaluation, boundary rescaling compresses knots to [0.02, 0.98].

---

## 6. Variable-Length Handling

**Input:** Padding masks in the encoder handle N = 4 to N = 10.

**Output:** The num_knots vector tells the decoder how many of its 7 queries to use per sample.

**Batching:** Within each mini-batch, samples are grouped by N. Each group goes through the differentiable solver as a single batched call.
