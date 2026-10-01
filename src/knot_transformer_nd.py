"""
knot_transformer_nd.py — Unified transformer for B-spline knot parameterisation
in arbitrary spatial dimensions (2D, 3D, nD).

Extends knot_transformer.py with:
  - spatial_dim argument: works for 2D, 3D, or any nD point sequences
  - nD geometric features: generalised turning angles, chord/centripetal fracs
  - nD curvature via curvature_layer_nd.py
  - same two-phase training pipeline (Phase1Trainer, Phase2Trainer)
  - fully backward compatible: spatial_dim=2 gives identical behaviour

Usage:
    model = KnotTransformerND(spatial_dim=3)   # 3D curves
    model = KnotTransformerND(spatial_dim=2)   # 2D — same as before
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Feature Engineering — nD
# ---------------------------------------------------------------------------

def compute_geometric_features_nd(points: torch.Tensor) -> torch.Tensor:
    """
    Compute rotation- and scale-invariant geometric features for point
    sequences in arbitrary spatial dimension d.

    Args:
        points: (B, N, d)  — batch of N points in R^d

    Returns:
        features: (B, N, 4) — same four features regardless of d:
            [0] cumulative chordal fraction
            [1] local segment length fraction
            [2] turning angle at point i (normalised to [0,1])
                In 2D: signed cross product angle
                In nD: unsigned angle between consecutive edge vectors
            [3] cumulative centripetal fraction
    """
    B, N, d = points.shape

    # Segment vectors and lengths
    diffs      = points[:, 1:] - points[:, :-1]                  # (B, N-1, d)
    chord_lens = torch.norm(diffs, dim=-1)                        # (B, N-1)
    centri_lens = torch.sqrt(chord_lens + 1e-8)                  # (B, N-1)

    chord_total  = chord_lens.sum(dim=1,  keepdim=True) + 1e-8   # (B, 1)
    centri_total = centri_lens.sum(dim=1, keepdim=True) + 1e-8   # (B, 1)

    chord_frac  = chord_lens  / chord_total                       # (B, N-1)
    centri_frac = centri_lens / centri_total                      # (B, N-1)

    chord_cum  = torch.cat([
        torch.zeros(B, 1, device=points.device),
        torch.cumsum(chord_frac, dim=1)
    ], dim=1)                                                      # (B, N)

    centri_cum = torch.cat([
        torch.zeros(B, 1, device=points.device),
        torch.cumsum(centri_frac, dim=1)
    ], dim=1)                                                      # (B, N)

    local_frac = torch.cat([
        torch.zeros(B, 1, device=points.device),
        chord_frac,
        torch.zeros(B, 1, device=points.device)
    ], dim=1)[:, :N]                                              # (B, N)

    # Turning angles — generalised to nD
    angles = torch.zeros(B, N, device=points.device)
    if N >= 3:
        v1 = diffs[:, :-1]    # (B, N-2, d)
        v2 = diffs[:, 1:]     # (B, N-2, d)

        if d == 2:
            # Signed 2D angle
            cross = v1[:, :, 0] * v2[:, :, 1] - v1[:, :, 1] * v2[:, :, 0]
            dot   = (v1 * v2).sum(dim=-1)
            theta = torch.atan2(cross, dot)                        # (B, N-2)
            theta_norm = (theta + torch.pi) / (2 * torch.pi)
        else:
            # Unsigned nD angle via dot product
            dot       = (v1 * v2).sum(dim=-1)                     # (B, N-2)
            norm1     = torch.norm(v1, dim=-1) + 1e-8
            norm2     = torch.norm(v2, dim=-1) + 1e-8
            cos_theta = (dot / (norm1 * norm2)).clamp(-1 + 1e-6, 1 - 1e-6)
            theta     = torch.acos(cos_theta)                      # (B, N-2)
            theta_norm = theta / torch.pi                          # → [0,1]

        angles[:, 1:N - 1] = theta_norm

    features = torch.stack([chord_cum, local_frac, angles, centri_cum], dim=-1)
    return features   # (B, N, 4)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class BSplineDatasetND(Dataset):
    """
    Dataset for point sequences in arbitrary spatial dimension d.

    Each sample:
        points    — (N, d)  tensor
        heuristic — (K,)    interior knot values in (0,1)
        num_knots — int
        spatial_dim — int   d
    """

    def __init__(
        self,
        points_list:    List[np.ndarray],    # list of (N, d) arrays
        heuristic_list: List[np.ndarray],    # list of (K,) interior knots
        task: str = "curvature",
    ):
        assert task in ("curvature", "arc")
        self.task    = task
        self.samples = []
        for pts, heur in zip(points_list, heuristic_list):
            self.samples.append({
                "points":      torch.tensor(pts,  dtype=torch.double),
                "heuristic":   torch.tensor(heur, dtype=torch.double),
                "num_knots":   len(heur),
                "spatial_dim": pts.shape[1],
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_nd(batch):
    """
    Collate samples with variable N (point count), K (knot count), and d
    (spatial dimension). Pads point sequences to the longest in the batch.
    Samples with different spatial dims are handled by zero-padding to max d.
    """
    max_N = max(s["points"].shape[0] for s in batch)
    max_K = max(s["num_knots"]       for s in batch)
    max_d = max(s["spatial_dim"]     for s in batch)
    B     = len(batch)

    points_pad  = torch.zeros(B, max_N, max_d, dtype=torch.double)
    heur_pad    = torch.zeros(B, max_K,         dtype=torch.double)
    num_knots   = torch.zeros(B,                dtype=torch.long)
    spatial_dim = torch.zeros(B,                dtype=torch.long)
    point_mask  = torch.ones(B, max_N,          dtype=torch.bool)   # True=ignore

    for i, s in enumerate(batch):
        N = s["points"].shape[0]
        K = s["num_knots"]
        d = s["spatial_dim"]
        points_pad[i, :N, :d] = s["points"]
        heur_pad[i, :K]       = s["heuristic"]
        num_knots[i]           = K
        spatial_dim[i]         = d
        point_mask[i, :N]      = False

    return {
        "points":      points_pad,
        "heuristic":   heur_pad,
        "num_knots":   num_knots,
        "spatial_dim": spatial_dim,
        "point_mask":  point_mask,
    }


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class KnotTransformerND(nn.Module):
    """
    Sequence-to-sequence transformer for B-spline knot parameterisation
    in arbitrary spatial dimension d.

    The model is spatial-dimension-agnostic: the same weights handle 2D,
    3D, and nD point sequences. Spatial dimension only affects the input
    projection layer.

    Architecture (unchanged from KnotTransformer):
        Input:   (B, N, d) points + (B, N, 4) geometric features
                 → projected to (B, N, d_model)
        Encoder: transformer encoder over point sequence
        Decoder: DETR-style parallel knot queries × encoder memory
        Head:    log-increment → softplus → cumsum → normalise → (0,1)

    Args:
        spatial_dim:        int   spatial dimension of input points (default 2)
        d_model:            int   transformer hidden dimension
        nhead:              int   number of attention heads
        num_encoder_layers: int
        num_decoder_layers: int
        dim_feedforward:    int
        dropout:            float
        max_points:         int   maximum sequence length
        max_knots:          int   maximum interior knots
    """

    def __init__(
        self,
        spatial_dim:        int   = 2,
        d_model:            int   = 64,
        nhead:              int   = 4,
        num_encoder_layers: int   = 3,
        num_decoder_layers: int   = 3,
        dim_feedforward:    int   = 256,
        dropout:            float = 0.1,
        max_points:         int   = 20,
        max_knots:          int   = 10,
    ):
        super().__init__()
        self.spatial_dim = spatial_dim
        self.d_model     = d_model
        self.max_knots   = max_knots
        self.max_points  = max_points

        # Input: d spatial coords + 4 geometric features
        self.input_proj = nn.Linear(spatial_dim + 4, d_model)

        # Learned positional encoding for point sequence
        self.point_pos_embed = nn.Embedding(max_points, d_model)

        # Transformer encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout,
            batch_first=True, dtype=torch.double,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_encoder_layers)

        # Learnable knot queries
        self.knot_queries = nn.Embedding(max_knots, d_model)

        # Transformer decoder
        dec_layer = nn.TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout,
            batch_first=True, dtype=torch.double,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_decoder_layers)

        # Knot prediction head: log-increments → ordered knots in (0,1)
        self.knot_head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

        self.double()

    def forward(
        self,
        points:     torch.Tensor,   # (B, N, d)
        point_mask: torch.Tensor,   # (B, N) bool
        num_knots:  torch.Tensor,   # (B,)   int
    ) -> torch.Tensor:              # (B, max_knots)
        B, N, d = points.shape

        # Geometric features (works for any d via compute_geometric_features_nd)
        feats = compute_geometric_features_nd(points)             # (B, N, 4)

        # If spatial_dim < d (zero-padded batch), slice to model's spatial_dim
        pts_in = points[:, :, :self.spatial_dim]
        x = torch.cat([pts_in, feats], dim=-1).to(torch.double)  # (B, N, d+4)
        x = self.input_proj(x)                                    # (B, N, d_model)

        # Positional encoding
        pos_idx = torch.arange(N, device=points.device)
        x = x + self.point_pos_embed(pos_idx)

        # Encoder
        memory = self.encoder(x, src_key_padding_mask=point_mask)

        # Decoder with learnable knot queries
        q_idx   = torch.arange(self.max_knots, device=points.device)
        queries = self.knot_queries(q_idx).unsqueeze(0).expand(B, -1, -1)
        decoded = self.decoder(
            queries, memory,
            memory_key_padding_mask=point_mask,
        )                                                          # (B, max_K, d_model)

        # Log-increment → strictly positive → cumsum → normalise to (0,1)
        raw        = self.knot_head(decoded).squeeze(-1)           # (B, max_K)
        increments = F.softplus(raw) + 1e-6
        cum        = torch.cumsum(increments, dim=-1)

        knot_vals  = torch.zeros_like(cum)
        for b in range(B):
            K     = num_knots[b].item()
            total = cum[b, K - 1]
            knot_vals[b, :K] = cum[b, :K] / (total + 1e-8)

        return knot_vals   # (B, max_knots)

    def predict(
        self,
        points:    torch.Tensor,   # (N, d) single sample
        num_knots: int,
    ) -> torch.Tensor:             # (num_knots,)
        """Single-sample inference."""
        pts  = points.unsqueeze(0).to(device)
        mask = torch.zeros(1, pts.shape[1], dtype=torch.bool, device=device)
        nk   = torch.tensor([num_knots], dtype=torch.long, device=device)
        with torch.no_grad():
            out = self.forward(pts, mask, nk)
        return out[0, :num_knots]


# ---------------------------------------------------------------------------
# Full knot vector builder
# ---------------------------------------------------------------------------

def build_full_knot_vector(interior: torch.Tensor, N_points: int) -> torch.Tensor:
    """
    Build full cubic B-spline knot vector from interior knots.
    [0,0,0,0, t1,...,tK, 1,1,1,1]
    """
    B   = interior.shape[0]
    dev = interior.device
    zeros = torch.zeros(B, 4, dtype=torch.double, device=dev)
    ones  = torch.ones( B, 4, dtype=torch.double, device=dev)
    return torch.cat([zeros, interior, ones], dim=1)


# ---------------------------------------------------------------------------
# Phase 1 Trainer
# ---------------------------------------------------------------------------

class Phase1TrainerND:
    """
    Pretrains the transformer to approximate heuristic knots (chordal for
    curvature, centripetal for arc length) using L1 loss.
    Works for any spatial dimension — the heuristic is computed the same
    way in nD (chordal = Euclidean distances, centripetal = sqrt of distances).
    """

    def __init__(self, model: KnotTransformerND, lr: float = 1e-3,
                 batch_size: int = 64):
        self.model     = model.to(device)
        self.optimiser = torch.optim.Adam(model.parameters(), lr=lr)
        self.loss_fn   = nn.L1Loss()
        self.bs        = batch_size

    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total = 0.0
        n     = 0
        for batch in loader:
            pts       = batch["points"].to(device)
            heuristic = batch["heuristic"].to(device)
            mask      = batch["point_mask"].to(device)
            nk        = batch["num_knots"].to(device)

            self.optimiser.zero_grad()
            pred = self.model(pts, mask, nk)

            B    = pts.shape[0]
            loss = torch.tensor(0.0, dtype=torch.double, device=device)
            for b in range(B):
                K    = nk[b].item()
                loss = loss + self.loss_fn(pred[b, :K], heuristic[b, :K])
            loss = loss / B
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimiser.step()
            total += loss.item()
            n     += 1
        return total / max(n, 1)

    def run(self, dataset, epochs=200, save_path="phase1_nd.pt", log_every=20):
        loader    = DataLoader(dataset, batch_size=self.bs, shuffle=True,
                               collate_fn=collate_nd)
        best_loss = float("inf")
        history   = []
        print(f"Phase 1 (nD) — {epochs} epochs | device: {device}")
        print("-" * 55)
        for epoch in range(1, epochs + 1):
            loss = self.train_epoch(loader)
            history.append(loss)
            if loss < best_loss:
                best_loss = loss
                torch.save(self.model.state_dict(), save_path)
            if epoch % log_every == 0 or epoch == 1:
                print(f"  Epoch {epoch:>4d}/{epochs} | L1: {loss:.6f} "
                      f"| best: {best_loss:.6f}")
        print(f"\nPhase 1 done. Saved to '{save_path}'")
        return history


# ---------------------------------------------------------------------------
# Phase 2 Trainer
# ---------------------------------------------------------------------------

class Phase2TrainerND:
    """
    Fine-tunes the pretrained transformer to minimise B-spline energy
    (curvature or arc length) directly.

    Uses curvature_layer_nd.global_curvature_nd or arc_nd as the loss,
    which are differentiable for any spatial dimension.
    """

    def __init__(
        self,
        model:      KnotTransformerND,
        energy_fn,                       # global_curvature_nd or arc_nd
        solve_fn,                        # solve_nd
        task:       str   = "curvature",
        lr:         float = 1e-4,
        batch_size: int   = 512,
    ):
        self.model      = model.to(device)
        self.energy_fn  = energy_fn
        self.solve_fn   = solve_fn
        self.task       = task
        self.optimiser  = torch.optim.Adam(model.parameters(), lr=lr)
        self.bs         = batch_size

    def _energy(self, pts_nd, knots):
        """
        pts_nd: (B, N, d)  — raw points
        knots:  (B, N+6)
        """
        B, N, d = pts_nd.shape
        P_list  = [pts_nd[:, i, :] for i in range(N)]   # list of (B, d)
        C       = self.solve_fn(P_list, knots)
        return self.energy_fn(C, knots, 3, 100)

    def filter_good_samples(self, dataset, heuristic_energy_list):
        """Keep only samples where model already beats the heuristic energy."""
        self.model.eval()
        good = []
        loader = DataLoader(dataset, batch_size=1, collate_fn=collate_nd)
        with torch.no_grad():
            for idx, batch in enumerate(loader):
                pts  = batch["points"].to(device)
                mask = batch["point_mask"].to(device)
                nk   = batch["num_knots"].to(device)
                pred = self.model(pts, mask, nk)
                K    = nk[0].item()
                n_pts = (~mask[0]).sum().item()
                pts_b = pts[0, :n_pts, :self.model.spatial_dim].unsqueeze(0)
                interior = pred[0, :K].unsqueeze(0)
                full_kv  = build_full_knot_vector(interior, n_pts)
                try:
                    eng = self._energy(pts_b, full_kv).mean().item()
                    if eng <= heuristic_energy_list[idx]:
                        good.append(idx)
                except Exception:
                    pass
        print(f"  Filter: {len(good)}/{len(dataset)} samples pass gate.")
        filtered         = BSplineDatasetND.__new__(BSplineDatasetND)
        filtered.task    = dataset.task
        filtered.samples = [dataset.samples[i] for i in good]
        return filtered

    def train_epoch(self, loader):
        self.model.train()
        total = 0.0
        n     = 0
        for batch in loader:
            pts  = batch["points"].to(device)
            mask = batch["point_mask"].to(device)
            nk   = batch["num_knots"].to(device)
            self.optimiser.zero_grad()
            pred = self.model(pts, mask, nk)

            B    = pts.shape[0]
            loss = torch.tensor(0.0, dtype=torch.double, device=device)
            n_valid = 0
            for b in range(B):
                K     = nk[b].item()
                n_pts = (~mask[b]).sum().item()
                pts_b = pts[b, :n_pts, :self.model.spatial_dim].unsqueeze(0)
                interior = pred[b, :K].unsqueeze(0)
                full_kv  = build_full_knot_vector(interior, n_pts)
                try:
                    eng  = self._energy(pts_b, full_kv).mean()
                    loss = loss + eng
                    n_valid += 1
                except Exception:
                    pass
            if n_valid == 0:
                continue
            loss = loss / n_valid
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimiser.step()
            total += loss.item()
            n     += 1
        return total / max(n, 1)

    def run(self, dataset, heuristic_energy_list, epochs=200,
            phase1_path="phase1_nd.pt", save_path="phase2_nd.pt",
            log_every=20, filter_first=True):
        self.model.load_state_dict(
            torch.load(phase1_path, map_location=device)
        )
        if filter_first:
            dataset = self.filter_good_samples(dataset, heuristic_energy_list)

        loader    = DataLoader(dataset, batch_size=self.bs, shuffle=True,
                               collate_fn=collate_nd)
        best_eng  = float("inf")
        history   = []
        print(f"\nPhase 2 (nD, {self.task}) — {epochs} epochs")
        print("-" * 55)
        for epoch in range(1, epochs + 1):
            eng = self.train_epoch(loader)
            history.append(eng)
            if eng < best_eng:
                best_eng = eng
                torch.save(self.model.state_dict(), save_path)
            if epoch % log_every == 0 or epoch == 1:
                print(f"  Epoch {epoch:>4d}/{epochs} | energy: {eng:.4f} "
                      f"| best: {best_eng:.4f}")
        print(f"\nPhase 2 done. Saved to '{save_path}'")
        return history


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from curvature_layer_nd import (
        solve_nd, global_curvature_nd, arc_nd,
        chordal_nd, centripetal_nd,
    )

    print("KnotTransformerND — smoke test")
    print("=" * 55)
    rng = np.random.default_rng(42)

    for d_space, label in [(2, "2D"), (3, "3D"), (5, "5D")]:
        print(f"\n{label} (spatial_dim={d_space}):")

        model = KnotTransformerND(
            spatial_dim=d_space, d_model=32, nhead=2,
            num_encoder_layers=2, num_decoder_layers=2,
            dim_feedforward=64, max_points=10, max_knots=6,
        )
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params:,}")

        # Build tiny mixed-dof dataset
        pts_list  = []
        heur_list = []
        for _ in range(30):
            N   = rng.integers(4, 7)     # 4–6 points → 2–4 dof
            pts = rng.uniform(0.1, 0.9, (N, d_space))
            diffs = np.linalg.norm(np.diff(pts, axis=0), axis=1)
            total = diffs.sum() + 1e-8
            cum   = np.cumsum(diffs) / total
            heur  = cum[:-1]             # interior knots
            pts_list.append(pts)
            heur_list.append(heur)

        dataset = BSplineDatasetND(pts_list, heur_list, task="curvature")
        print(f"  Dataset: {len(dataset)} samples, mixed dof")

        # Quick phase 1
        trainer = Phase1TrainerND(model, lr=1e-3, batch_size=8)
        hist    = trainer.run(dataset, epochs=3, log_every=1,
                              save_path=f"/tmp/phase1_{d_space}d.pt")

        # Single inference
        sample_pts = torch.tensor(pts_list[0], dtype=torch.double)
        K_pred     = len(heur_list[0])
        knots_pred = model.predict(sample_pts, K_pred)
        print(f"  Heuristic knots: {heur_list[0].round(3)}")
        print(f"  Predicted knots: {knots_pred.cpu().numpy().round(3)}")

    print("\nAll spatial dimensions working.")
