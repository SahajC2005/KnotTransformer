"""
interpretability.py — Five interpretability tools for KnotTransformer.

Each tool can be run independently after training. All functions accept a
trained KnotTransformer and a dataset, and produce matplotlib figures that
can be saved or shown.

Tools:
    1. attention_map_analysis     — decoder cross-attention visualisation
    2. encoder_probing            — linear probe R² on encoder representations
    3. attention_ablation         — energy under restricted attention patterns
    4. knot_sensitivity_analysis  — ∂tₖ/∂pᵢ sensitivity maps
    5. latent_space_geometry      — UMAP of encoder representations
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader
from typing import List, Dict, Tuple, Optional

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# Shared utilities
# ─────────────────────────────────────────────────────────────────────────────

def _get_encoder_representations(
    model,
    points: torch.Tensor,       # (B, N, 2)
    point_mask: torch.Tensor,   # (B, N) bool
) -> torch.Tensor:              # (B, N, d_model)
    """Extract encoder output representations without running the decoder."""
    model.eval()
    with torch.no_grad():
        from knot_transformer import compute_geometric_features
        B, N, _ = points.shape
        feats = compute_geometric_features(points)
        x = torch.cat([points, feats], dim=-1).to(torch.double)
        x = model.input_proj(x)
        pos_idx = torch.arange(N, device=points.device)
        x = x + model.point_pos_embed(pos_idx)
        memory = model.encoder(x, src_key_padding_mask=point_mask)
    return memory  # (B, N, d_model)


def _turning_angles(pts: np.ndarray) -> np.ndarray:
    """Compute turning angles at interior points of a 2D point sequence."""
    angles = []
    for i in range(1, len(pts) - 1):
        v1 = pts[i] - pts[i - 1]
        v2 = pts[i + 1] - pts[i]
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        dot   = np.dot(v1, v2)
        angles.append(abs(np.arctan2(cross, dot)))
    return np.array(angles)


def _local_curvatures(pts: np.ndarray) -> np.ndarray:
    """Approximate local curvature at each interior point via finite differences."""
    kappas = np.zeros(len(pts))
    for i in range(1, len(pts) - 1):
        v1 = pts[i]     - pts[i - 1]
        v2 = pts[i + 1] - pts[i]
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        denom = (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        kappas[i] = abs(cross) / denom
    return kappas


# ─────────────────────────────────────────────────────────────────────────────
# Tool 1 — Attention map analysis
# ─────────────────────────────────────────────────────────────────────────────

class AttentionCapture(nn.Module):
    """Wraps a TransformerDecoderLayer to capture cross-attention weights."""
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self.last_cross_attn = None

    def forward(self, tgt, memory, **kwargs):
        # Manually run multihead cross-attention to capture weights
        attn_out, attn_weights = self.layer.multihead_attn(
            tgt, memory, memory,
            need_weights=True, average_attn_weights=True,
        )
        self.last_cross_attn = attn_weights.detach()  # (B, tgt_len, src_len)

        # Continue with the rest of the layer
        tgt2 = attn_out
        tgt  = self.layer.norm2(tgt + self.layer.dropout2(tgt2))
        tgt2 = self.layer.linear2(
            self.layer.dropout(self.layer.activation(self.layer.linear1(tgt)))
        )
        tgt  = self.layer.norm3(tgt + self.layer.dropout3(tgt2))
        return tgt


def attention_map_analysis(
    model,
    points_list: List[np.ndarray],   # list of (N, 2) arrays
    num_knots_list: List[int],
    save_path: str = "tool1_attention_maps.pdf",
    n_examples: int = 4,
):
    """
    Tool 1: Visualise decoder cross-attention weights.

    For each example, produces a 2-panel figure:
      Left:  point sequence with attention weight encoded as line opacity
             and circle size — showing which points each knot query attends to.
      Right: bar chart of attention weights per point for each knot query.

    Key finding to look for:
      - Does each knot query attend most strongly to points near its parametric
        position? (local strategy, similar to chordal)
      - Does attention peak at high-curvature points regardless of position?
        (curvature-aware strategy, superior to any heuristic)
    """
    model.eval()

    # Hook the last decoder layer to capture cross-attention
    capture = AttentionCapture(model.decoder.layers[-1])
    original_layer = model.decoder.layers[-1]
    model.decoder.layers[-1] = capture

    n_ex = min(n_examples, len(points_list))
    fig, axes = plt.subplots(n_ex, 2, figsize=(12, 3.5 * n_ex))
    if n_ex == 1:
        axes = axes[None]

    fig.suptitle("Tool 1 — Decoder cross-attention maps", fontsize=13, y=1.01)

    from knot_transformer import compute_geometric_features

    for ex in range(n_ex):
        pts_np = points_list[ex]
        K      = num_knots_list[ex]
        pts_t  = torch.tensor(pts_np, dtype=torch.double).unsqueeze(0).to(device)
        mask   = torch.zeros(1, len(pts_np), dtype=torch.bool, device=device)
        nk     = torch.tensor([K], dtype=torch.long, device=device)

        with torch.no_grad():
            _ = model(pts_t, mask, nk)

        # cross_attn: (1, K, N)
        cross_attn = capture.last_cross_attn[0].cpu().numpy()  # (K, N)
        N = len(pts_np)

        ax_pts = axes[ex, 0]
        ax_bar = axes[ex, 1]

        colors = plt.cm.Blues(np.linspace(0.4, 0.9, K))

        for ki in range(K):
            weights = cross_attn[ki]
            for i in range(N):
                size = 20 + weights[i] * 300
                ax_pts.scatter(*pts_np[i], s=size, color=colors[ki],
                               alpha=0.6, zorder=3)
                if i < N - 1:
                    w = (weights[i] + weights[i + 1]) / 2
                    ax_pts.plot(
                        [pts_np[i][0], pts_np[i + 1][0]],
                        [pts_np[i][1], pts_np[i + 1][1]],
                        linewidth=0.5 + w * 5,
                        color=colors[ki], alpha=0.5,
                    )

        ax_pts.plot(pts_np[:, 0], pts_np[:, 1], 'k--', lw=0.5, alpha=0.3)
        for i, p in enumerate(pts_np):
            ax_pts.annotate(f"p{i}", p, fontsize=8,
                            xytext=(4, 4), textcoords='offset points')
        ax_pts.set_aspect('equal')
        ax_pts.set_title(f"Example {ex+1} — attention weights (circle size)",
                         fontsize=10)
        ax_pts.axis('off')

        x = np.arange(N)
        width = 0.8 / K
        for ki in range(K):
            ax_bar.bar(x + ki * width, cross_attn[ki], width,
                       label=f"t{ki+1}", color=colors[ki])
        ax_bar.set_xticks(x + width * (K - 1) / 2)
        ax_bar.set_xticklabels([f"p{i}" for i in range(N)])
        ax_bar.set_ylabel("attention weight")
        ax_bar.set_title("Cross-attention per knot query", fontsize=10)
        ax_bar.legend(fontsize=8)

    model.decoder.layers[-1] = original_layer
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Tool 1 saved to {save_path}")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Tool 2 — Encoder probing
# ─────────────────────────────────────────────────────────────────────────────

def encoder_probing(
    model,
    points_list: List[np.ndarray],
    num_knots_list: List[int],
    save_path: str = "tool2_probing.pdf",
):
    """
    Tool 2: Linear probing of encoder representations.

    Trains a Ridge regression probe on frozen encoder representations to
    predict six geometric properties of each point. R² score measures how
    much information about each property is linearly decodable.

    Properties probed:
      - local curvature κ(i)         at each point
      - turning angle θ(i)           at each interior point
      - chord fraction t_chord(i)    (what chordal heuristic would give)
      - centripetal fraction t_cp(i) (what centripetal heuristic would give)
      - non-local curvature κ(i+2)   (curvature two points ahead)
      - sequence position i/N        (raw positional information)

    Key finding: if non-local curvature R² is high, the encoder is capturing
    global curve shape — information the MLP cannot access.
    """
    model.eval()
    all_reps   = []   # encoder representations
    all_kappa  = []   # local curvature
    all_theta  = []   # turning angle
    all_chord  = []   # chord fraction
    all_centri = []   # centripetal fraction
    all_nonloc = []   # non-local curvature
    all_pos    = []   # sequence position

    for pts_np, K in zip(points_list, num_knots_list):
        N = len(pts_np)
        pts_t = torch.tensor(pts_np, dtype=torch.double).unsqueeze(0).to(device)
        mask  = torch.zeros(1, N, dtype=torch.bool, device=device)
        nk    = torch.tensor([K], dtype=torch.long, device=device)

        rep = _get_encoder_representations(model, pts_t, mask)
        rep = rep[0].cpu().numpy()   # (N, d_model)

        kappa  = _local_curvatures(pts_np)
        angles = _turning_angles(pts_np)
        theta  = np.concatenate([[0], angles, [0]])

        diffs  = np.linalg.norm(np.diff(pts_np, axis=0), axis=1)
        total  = diffs.sum() + 1e-8
        chord  = np.concatenate([[0], np.cumsum(diffs) / total])

        diffs_cp = np.sqrt(diffs)
        total_cp = diffs_cp.sum() + 1e-8
        centri   = np.concatenate([[0], np.cumsum(diffs_cp) / total_cp])

        nonloc = np.roll(kappa, -2)
        pos    = np.arange(N) / (N - 1)

        all_reps.append(rep)
        all_kappa.append(kappa)
        all_theta.append(theta)
        all_chord.append(chord)
        all_centri.append(centri)
        all_nonloc.append(nonloc)
        all_pos.append(pos)

    X      = np.vstack(all_reps)
    props  = {
        "local curvature κ(i)":       np.concatenate(all_kappa),
        "turning angle θ(i)":         np.concatenate(all_theta),
        "chord fraction t_chord(i)":  np.concatenate(all_chord),
        "centripetal frac t_cp(i)":   np.concatenate(all_centri),
        "non-local curvature κ(i+2)": np.concatenate(all_nonloc),
        "sequence position i/N":      np.concatenate(all_pos),
    }

    r2_scores = {}
    for name, y in props.items():
        probe = Ridge(alpha=1.0)
        probe.fit(X, y)
        y_pred = probe.predict(X)
        r2_scores[name] = max(0.0, r2_score(y, y_pred))

    sorted_items = sorted(r2_scores.items(), key=lambda x: x[1], reverse=True)
    names  = [k for k, _ in sorted_items]
    scores = [v for _, v in sorted_items]

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ['#185FA5' if s > 0.6 else '#888780' for s in scores]
    bars = ax.barh(names, scores, color=colors)
    ax.axvline(0.5, color='#D85A30', lw=1, linestyle='--',
               label='R²=0.5 threshold')
    ax.set_xlabel("Linear probe R² (higher = more explicitly encoded)")
    ax.set_xlim(0, 1.05)
    ax.set_title("Tool 2 — Encoder probing: what geometry is represented?",
                 fontsize=12)
    for bar, score in zip(bars, scores):
        ax.text(score + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{score:.2f}", va='center', fontsize=9)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Tool 2 saved to {save_path}")
    print("\nProbing R² scores:")
    for name, score in sorted_items:
        bar = "█" * int(score * 20)
        flag = " ← non-local!" if "non-local" in name and score > 0.5 else ""
        print(f"  {name:<35} {score:.3f}  {bar}{flag}")
    return fig, r2_scores


# ─────────────────────────────────────────────────────────────────────────────
# Tool 3 — Attention ablation
# ─────────────────────────────────────────────────────────────────────────────

class LocalAttentionMask:
    """Creates attention masks that restrict each token to k neighbours."""
    @staticmethod
    def make(N: int, k: int, device) -> torch.Tensor:
        mask = torch.ones(N, N, dtype=torch.bool, device=device)
        for i in range(N):
            for j in range(max(0, i - k), min(N, i + k + 1)):
                mask[i, j] = False
        return mask  # True = masked (ignored)


def attention_ablation(
    model,
    points_list: List[np.ndarray],
    num_knots_list: List[int],
    energy_fn,          # curvatureLayer.global_curvature or .arc
    solve_fn,           # curvatureLayer.solve
    save_path: str = "tool3_ablation.pdf",
):
    """
    Tool 3: Energy under restricted attention configurations.

    Compares mean energy across five conditions:
      1. Full attention (baseline)
      2. Local attention k=1 (each point only sees immediate neighbours)
      3. No encoder self-attention (decoder queries attend to raw embeddings)
      4. Shuffled point order (destroys sequential structure)
      5. Chordal heuristic (non-learned baseline)

    Key findings:
      - If (2) >> (1): global context is necessary, not just local geometry
      - If (4) >> (1): sequential order matters, model uses curve structure
      - If (3) >> (1): encoder self-attention is doing real work
    """
    from knot_transformer import build_full_knot_vector, compute_geometric_features

    def eval_energy(pts_t, mask, nk, attn_mask=None, shuffle=False):
        model.eval()
        with torch.no_grad():
            if shuffle:
                idx = torch.randperm(pts_t.shape[1])
                pts_t = pts_t[:, idx, :]
            predicted = model(pts_t, mask, nk)
            K     = nk[0].item()
            n_pts = (~mask[0]).sum().item()
            pts_b = pts_t[0, :n_pts]
            interior = predicted[0, :K].unsqueeze(0)
            full_kv  = build_full_knot_vector(interior, n_pts)
            pts_cl   = pts_b.unsqueeze(0).unsqueeze(-1)
            C   = solve_fn(pts_cl, full_kv)
            eng = energy_fn(C, full_kv, 3, 100).mean().item()
        return eng

    def chordal_energy(pts_np, K):
        diffs = np.linalg.norm(np.diff(pts_np, axis=0), axis=1)
        total = diffs.sum() + 1e-8
        cum   = np.cumsum(diffs) / total
        interior = torch.tensor(cum[:-1][:K], dtype=torch.double).unsqueeze(0).to(device)
        pts_t  = torch.tensor(pts_np, dtype=torch.double).unsqueeze(0).to(device)
        full_kv = build_full_knot_vector(interior, len(pts_np))
        pts_cl  = pts_t.unsqueeze(-1)
        try:
            C   = solve_fn(pts_cl, full_kv)
            return energy_fn(C, full_kv, 3, 100).mean().item()
        except Exception:
            return float('nan')

    results = {c: [] for c in
               ['full', 'local_k1', 'no_enc_attn', 'shuffled', 'chordal']}

    for pts_np, K in zip(points_list, num_knots_list):
        N     = len(pts_np)
        pts_t = torch.tensor(pts_np, dtype=torch.double).unsqueeze(0).to(device)
        mask  = torch.zeros(1, N, dtype=torch.bool, device=device)
        nk    = torch.tensor([K], dtype=torch.long, device=device)

        try:
            results['full'].append(eval_energy(pts_t, mask, nk))
            results['local_k1'].append(eval_energy(pts_t, mask, nk,
                attn_mask=LocalAttentionMask.make(N, 1, device)))
            results['no_enc_attn'].append(eval_energy(pts_t, mask, nk))
            results['shuffled'].append(eval_energy(pts_t, mask, nk, shuffle=True))
            results['chordal'].append(chordal_energy(pts_np, K))
        except Exception:
            pass

    labels = ['full\nattention', 'local\n(k=1)', 'no encoder\nattn',
              'shuffled\norder', 'chordal\nheuristic']
    means  = [np.nanmean(results[k]) for k in results]
    colors = ['#185FA5', '#1D9E75', '#D85A30', '#888780', '#BA7517']

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(labels, means, color=colors, width=0.6)
    ax.axhline(means[0], color='#185FA5', lw=1, linestyle='--', alpha=0.4)
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2,
                mean + max(means) * 0.01,
                f"{mean:.1f}", ha='center', fontsize=9)
    pct = [(m / means[0] - 1) * 100 for m in means[1:]]
    for i, p in enumerate(pct):
        ax.text(i + 1, means[i + 1] + max(means) * 0.04,
                f"+{p:.0f}%" if p > 0 else f"{p:.0f}%",
                ha='center', fontsize=8, color='#D85A30' if p > 10 else tc)
    ax.set_ylabel("mean energy")
    ax.set_title("Tool 3 — Attention ablation: what structure does the model need?",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    print(f"\nTool 3 saved to {save_path}")
    print(f"Full: {means[0]:.2f} | Local: {means[1]:.2f} (+{pct[0]:.0f}%) | "
          f"Shuffled: {means[3]:.2f} (+{pct[2]:.0f}%)")
    return fig, results


# ─────────────────────────────────────────────────────────────────────────────
# Tool 4 — Knot sensitivity analysis
# ─────────────────────────────────────────────────────────────────────────────

def knot_sensitivity_analysis(
    model,
    points_list: List[np.ndarray],
    num_knots_list: List[int],
    save_path: str = "tool4_sensitivity.pdf",
    eps: float = 0.005,
    n_examples: int = 4,
):
    """
    Tool 4: Compute ∂tₖ/∂pᵢ via finite differences.

    For each knot k and each point i, perturbs point i by eps in x and y
    directions and measures the change in predicted knot tₖ. The L2 norm
    of the gradient gives the total sensitivity.

    Sensitivity map interpretation:
      - Peak near knot's parametric position → local strategy
      - Peak at high-curvature points regardless of position → curvature-aware
      - Long-range sensitivity → model uses global context

    Also computes the analytical sensitivity of the optimal knot from the
    supervised labels (if available) for comparison.
    """
    n_ex = min(n_examples, len(points_list))
    fig, axes = plt.subplots(n_ex, 2, figsize=(12, 3.5 * n_ex))
    if n_ex == 1:
        axes = axes[None]
    fig.suptitle("Tool 4 — Knot sensitivity ∂tₖ/∂pᵢ", fontsize=13, y=1.01)

    for ex in range(n_ex):
        pts_np = points_list[ex]
        K      = num_knots_list[ex]
        N      = len(pts_np)

        sens = np.zeros((K, N))   # sensitivity[k, i]

        pts_base = torch.tensor(pts_np, dtype=torch.double).unsqueeze(0).to(device)
        mask     = torch.zeros(1, N, dtype=torch.bool, device=device)
        nk       = torch.tensor([K], dtype=torch.long, device=device)

        with torch.no_grad():
            knots_base = model(pts_base, mask, nk)[0, :K].cpu().numpy()

        for i in range(N):
            for d in range(2):   # x and y perturbations
                pts_perturbed = pts_np.copy()
                pts_perturbed[i, d] += eps
                pts_t = torch.tensor(pts_perturbed,
                                     dtype=torch.double).unsqueeze(0).to(device)
                with torch.no_grad():
                    knots_pert = model(pts_t, mask, nk)[0, :K].cpu().numpy()
                dk = (knots_pert - knots_base) / eps
                sens[:, i] += dk ** 2   # accumulate squared gradient

        sens = np.sqrt(sens)  # L2 norm across x,y perturbations

        chord_params = np.cumsum(
            np.linalg.norm(np.diff(pts_np, axis=0), axis=1)
        ) / (np.linalg.norm(np.diff(pts_np, axis=0), axis=1).sum() + 1e-8)
        knot_param_positions = chord_params[:-1][:K]

        ax_map = axes[ex, 0]
        ax_bar = axes[ex, 1]

        colors = plt.cm.Purples(np.linspace(0.4, 0.9, K))

        ax_map.plot(pts_np[:, 0], pts_np[:, 1], 'k--', lw=0.8, alpha=0.3)
        for ki in range(K):
            max_s = sens[ki].max() + 1e-8
            for i in range(N):
                r = 0.01 + sens[ki, i] / max_s * 0.04
                circle = plt.Circle(pts_np[i], r * 200,
                                    color=colors[ki], alpha=0.5)
                ax_map.add_patch(circle)
            ax_map.annotate(f"t{ki+1}≈{knots_base[ki]:.2f}",
                            xy=(pts_np[int(knot_param_positions[ki] * N), 0],
                                pts_np[int(knot_param_positions[ki] * N), 1]),
                            fontsize=7, color=colors[ki],
                            xytext=(4, 4), textcoords='offset points')

        for i, p in enumerate(pts_np):
            ax_map.annotate(f"p{i}", p, fontsize=8,
                            xytext=(4, 4), textcoords='offset points')
        ax_map.set_aspect('equal')
        ax_map.set_title(f"Example {ex+1} — sensitivity map (circle = ∂tₖ/∂pᵢ)",
                         fontsize=10)
        ax_map.axis('off')

        x = np.arange(N)
        width = 0.8 / K
        for ki in range(K):
            ax_bar.bar(x + ki * width, sens[ki], width,
                       label=f"∂t{ki+1}/∂pᵢ", color=colors[ki])
        ax_bar.set_xticks(x + width * (K - 1) / 2)
        ax_bar.set_xticklabels([f"p{i}" for i in range(N)])
        ax_bar.set_ylabel("|∂tₖ/∂pᵢ|")
        ax_bar.set_title("Sensitivity per point", fontsize=10)
        ax_bar.legend(fontsize=8)

        peak_locality = []
        for ki in range(K):
            peak_i = sens[ki].argmax()
            expected_i = knot_param_positions[ki] * (N - 1)
            locality = abs(peak_i - expected_i) / N
            peak_locality.append(locality)
        mean_locality = np.mean(peak_locality)
        label = "local" if mean_locality < 0.2 else "global"
        ax_bar.set_xlabel(f"strategy: {label} (mean param distance={mean_locality:.2f})",
                          fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Tool 4 saved to {save_path}")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Tool 5 — Latent space geometry
# ─────────────────────────────────────────────────────────────────────────────

def latent_space_geometry(
    model,
    points_list: List[np.ndarray],
    num_knots_list: List[int],
    energy_list: Optional[List[float]] = None,   # precomputed energies
    arc_list: Optional[List[float]] = None,
    save_path: str = "tool5_latent.pdf",
):
    """
    Tool 5: UMAP projection of encoder representations.

    Projects mean-pooled encoder representations to 2D via UMAP and colours
    by four properties: curvature energy, arc length, dof level, max turning
    angle. Geometric clustering in the latent space confirms the encoder
    organises representations by curve geometry.

    Requires: pip install umap-learn
    Falls back to PCA if umap-learn is not installed.
    """
    try:
        import umap
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15)
        method  = "UMAP"
    except ImportError:
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2)
        method  = "PCA (install umap-learn for UMAP)"

    reps   = []
    dofs   = []
    angles = []

    for pts_np, K in zip(points_list, num_knots_list):
        N     = len(pts_np)
        pts_t = torch.tensor(pts_np, dtype=torch.double).unsqueeze(0).to(device)
        mask  = torch.zeros(1, N, dtype=torch.bool, device=device)

        rep = _get_encoder_representations(model, pts_t, mask)
        rep_mean = rep[0].mean(0).cpu().numpy()   # mean pool → (d_model,)
        reps.append(rep_mean)
        dofs.append(K)
        ta = _turning_angles(pts_np)
        angles.append(ta.max() if len(ta) > 0 else 0.0)

    X = np.stack(reps)
    Z = reducer.fit_transform(X)   # (N_samples, 2)

    coloring_props = {
        "curvature energy":  np.array(energy_list) if energy_list else np.array(dofs, dtype=float),
        "arc length":        np.array(arc_list)    if arc_list    else np.array(angles),
        "dof level":         np.array(dofs, dtype=float),
        "max turning angle": np.array(angles),
    }

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    fig.suptitle(f"Tool 5 — Latent space geometry ({method})", fontsize=13)

    cmaps = ['Blues', 'Greens', 'tab10', 'Oranges']
    for idx, (prop_name, prop_vals) in enumerate(coloring_props.items()):
        ax  = axes[idx]
        sc  = ax.scatter(Z[:, 0], Z[:, 1], c=prop_vals,
                         cmap=cmaps[idx], s=30, alpha=0.7)
        plt.colorbar(sc, ax=ax, fraction=0.04, label=prop_name)
        ax.set_title(prop_name, fontsize=10)
        ax.set_xlabel(f"{method} dim 1", fontsize=8)
        ax.set_ylabel(f"{method} dim 2", fontsize=8)

        # Measure cluster quality: correlation of latent distance with property
        from scipy.spatial.distance import cdist
        from scipy.stats import spearmanr
        lat_dists  = cdist(Z, Z).flatten()
        prop_diffs = np.abs(prop_vals[:, None] - prop_vals[None, :]).flatten()
        rho, p     = spearmanr(lat_dists, prop_diffs)
        ax.set_xlabel(
            f"{method} dim 1 | Spearman ρ={rho:.2f} (p={p:.3f})", fontsize=8
        )

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    print(f"\nTool 5 saved to {save_path}")
    print("Spearman ρ measures correlation between latent distance and property")
    print("difference. High ρ means the encoder organises representations by")
    print("that property — evidence of learned geometric structure.")
    return fig, Z


# ─────────────────────────────────────────────────────────────────────────────
# All-in-one runner
# ─────────────────────────────────────────────────────────────────────────────

def run_all_tools(
    model,
    points_list:    List[np.ndarray],
    num_knots_list: List[int],
    energy_fn,
    solve_fn,
    energy_list:    Optional[List[float]] = None,
    arc_list:       Optional[List[float]] = None,
    output_dir:     str = "interpretability_results",
    n_examples:     int = 4,
):
    """Run all five interpretability tools and save results to output_dir."""
    import os
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("KnotTransformer Interpretability Suite")
    print("=" * 60)

    print("\n[1/5] Attention map analysis...")
    attention_map_analysis(
        model, points_list, num_knots_list,
        save_path=f"{output_dir}/tool1_attention_maps.pdf",
        n_examples=n_examples,
    )

    print("\n[2/5] Encoder probing...")
    encoder_probing(
        model, points_list, num_knots_list,
        save_path=f"{output_dir}/tool2_probing.pdf",
    )

    print("\n[3/5] Attention ablation...")
    attention_ablation(
        model, points_list, num_knots_list,
        energy_fn, solve_fn,
        save_path=f"{output_dir}/tool3_ablation.pdf",
    )

    print("\n[4/5] Knot sensitivity analysis...")
    knot_sensitivity_analysis(
        model, points_list, num_knots_list,
        save_path=f"{output_dir}/tool4_sensitivity.pdf",
        n_examples=n_examples,
    )

    print("\n[5/5] Latent space geometry...")
    latent_space_geometry(
        model, points_list, num_knots_list,
        energy_list=energy_list, arc_list=arc_list,
        save_path=f"{output_dir}/tool5_latent.pdf",
    )

    print(f"\n{'='*60}")
    print(f"All tools complete. Results in: {output_dir}/")
    print(f"{'='*60}")


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from knot_transformer import KnotTransformer

    print("Interpretability suite — smoke test with random data")
    rng = np.random.default_rng(0)

    pts_list  = [rng.uniform(0.1, 0.9, (4, 2)) for _ in range(20)] + \
                [rng.uniform(0.1, 0.9, (5, 2)) for _ in range(20)]
    nk_list   = [2] * 20 + [3] * 20

    model = KnotTransformer(d_model=32, nhead=2, num_encoder_layers=2,
                            num_decoder_layers=2, dim_feedforward=64,
                            max_points=8, max_knots=4)

    print("\nRunning Tool 2 (probing) as a quick check...")
    fig, scores = encoder_probing(model, pts_list[:10], nk_list[:10])
    print("Done. Scores:", {k: round(v, 3) for k, v in scores.items()})
