# Mathematical Background

A self-contained reference for the mathematics behind the KnotTransformer.

---

## 1. B-Spline Curves

### Definition

A clamped cubic B-spline curve with knot vector U = {0,0,0,0, u₁,...,uₖ, 1,1,1,1} and control points P₀,...,Pₙ is:

```
C(t) = Σᵢ Nᵢ,₃(t) Pᵢ,    t ∈ [0, 1]
```

### Cox–de Boor Recursion

The basis functions are built recursively:

**Degree 0** (indicator functions):
```
Nᵢ,₀(t) = 1  if uᵢ ≤ t < uᵢ₊₁
           0  otherwise
```

**Degree p ≥ 1** (linear blending):
```
Nᵢ,ₚ(t) = (t - uᵢ)/(uᵢ₊ₚ - uᵢ) · Nᵢ,ₚ₋₁(t) + (uᵢ₊ₚ₊₁ - t)/(uᵢ₊ₚ₊₁ - uᵢ₊₁) · Nᵢ₊₁,ₚ₋₁(t)
```

with 0/0 ≡ 0.

### Dimensions

For N data points:
- Knot vector length: |U| = N + 6
- Control points: N + 2
- Interior knots (free variables): K = N − 2

### Properties

- **Non-negativity and partition of unity:** Nᵢ,ₚ(t) ≥ 0 and Σᵢ Nᵢ,ₚ(t) = 1
- **Local support:** Nᵢ,ₚ(t) = 0 for t ∉ [uᵢ, uᵢ₊ₚ₊₁)
- **Clamping:** Multiplicity-4 end knots force C(0) = P₀ and C(1) = Pₙ

---

## 2. Interpolation: The Linear Solve

Given N ordered data points D₀,...,D_{N-1} ∈ ℝ² and parameter values determined by the knots, the interpolation conditions are:

```
C(t̄ₖ) = Σᵢ Nᵢ,₃(t̄ₖ) Pᵢ = Dₖ,    k = 0,...,N-1
```

With natural end conditions C''(0) = C''(1) = 0, this gives a square (N+2)×(N+2) linear system:

```
A(u) P = D̃
```

where:
- **A(u)** is the collocation matrix (basis function values at data parameters)
- **P** is the (N+2)×2 control point matrix
- **D̃** stacks the data points with two zero rows for end conditions

**The fit is free:** For any valid knot vector u, this system has a unique solution P(u) = A(u)⁻¹D̃. Every resulting curve passes exactly through all data points.

---

## 3. Derivatives

The derivative of a degree-p B-spline is a degree-(p-1) B-spline with first-difference control points:

```
C'(t) = Σᵢ Nᵢ,ₚ₋₁(t) Qᵢ

where Qᵢ = p/(uᵢ₊ₚ₊₁ - uᵢ₊₁) · (Pᵢ₊₁ - Pᵢ)
```

The second derivative:
```
C''(t) = Σᵢ Nᵢ,ₚ₋₂(t) Rᵢ

where Rᵢ = (p-1)/(uᵢ₊ₚ - uᵢ₊₁) · (Qᵢ₊₁ - Qᵢ)
```

**Key insight:** The knot differences uᵢ₊ₚ₊₁ - uᵢ₊₁ appear in the denominator. Moving two knots close together increases |Qᵢ|, increasing the derivative magnitude and potentially the curvature. This is the mechanism by which knot spacing controls curvature.

---

## 4. Curvature

For a parametric planar curve C(t) = (x(t), y(t)):

```
κ(t) = (x'y'' - y'x'') / (x'² + y'²)^(3/2)
```

**Components:**
- x'y'' - y'x'' : cross product of velocity and acceleration (measures turning)
- (x'² + y'²)^(3/2) : speed cubed (normalizes for parameterization speed)
- κ = 1/R : reciprocal of osculating circle radius

**Total bending energy:**
```
E_curv(u) = ∫₀¹ κ(t)² ‖C'(t)‖ dt ≈ (1/M) Σⱼ κ(tⱼ)² ‖C'(tⱼ)‖
```

Approximated by Newton–Cotes quadrature with M = 100 samples.

---

## 5. The Optimization Problem

The full chain:

```
u → A⁻¹ → P(u) → derivatives + quadrature → E(u) ∈ ℝ
```

This landscape is non-convex with multiple local minima.

---

## 6. The Self-Supervised Loss

```
L(θ) = (1/N) Σᵢ min(E(f_θ(Dᵢ)), E_cap)
```

where f_θ is the KnotTransformer, E_cap = 1000 prevents outlier-dominated gradients.

---

## 7. Gradient Through the Pipeline

```
∂L/∂θ = (∂E/∂u) · (∂u/∂θ)
```

**∂u/∂θ:** Standard backpropagation through the transformer.

**∂E/∂u:** Requires implicit differentiation through the linear solve:

```
A(u)P = D̃
→ (∂A/∂uₖ)P + A(∂P/∂uₖ) = 0
→ ∂P/∂uₖ = -A⁻¹ (∂A/∂uₖ) P
```

This is the special ingredient — it tells us how control points change when a knot moves, enabling gradient flow from the curvature loss all the way back to the transformer weights.

---

## 8. Classical Baselines

**Chordal parameterization:**
```
t̄ₖ = Σⱼ₌₁ᵏ ‖Dⱼ - Dⱼ₋₁‖ / Σⱼ₌₁ᴺ⁻¹ ‖Dⱼ - Dⱼ₋₁‖
```

Spaces knots proportional to inter-point distances. Ignores turning angles and curvature.

**Centripetal parameterization:** Uses square roots of chord lengths instead, damping the effect of large gaps.

**Uniform parameterization:** Equal spacing, ignoring point geometry entirely.

---

## 9. Transformer Attention Mechanism

**Multi-head self-attention:**
```
Q = H·W_Q,  K = H·W_K,  V = H·W_V     (three learned projections)
scores = Q·Kᵀ / √dₖ                     (pairwise similarity)
weights = softmax(scores)                 (normalize to probabilities)
output = weights · V                      (weighted sum of values)
```

With h = 4 heads, dₖ = 64/4 = 16 per head. Each head computes its own N×N attention pattern on a 16-d slice of the representation.

**Monotonic output head:**
```
sᵢ = softplus(rᵢ) + ε     (positive increments)
cᵢ = Σⱼ≤ᵢ sⱼ              (cumulative sum → increasing)
uᵢ = cᵢ / c_{K+1}          (normalize → (0,1))
```

Guarantees valid ordered knots by construction.
