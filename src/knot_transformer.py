"""
KnotTransformer — A unified transformer model for B-spline knot parameterisation.

Replaces separate MLP models for 2/3/4 dof with a single sequence-to-sequence
transformer that handles variable-length point sequences and variable knot counts.

Training is two-phase (matching the existing self-supervised pipeline):
  Phase 1: Pretrain against heuristic knots (chordal for curvature, centripetal
            for arc length) using L1 loss.
  Phase 2: Fine-tune directly against the B-spline energy (curvature or arc
            length) using the differentiable curvatureLayer functions.

Usage:
    from knot_transformer import KnotTransformer, Phase1Trainer, Phase2Trainer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------

def compute_geometric_features(points: torch.Tensor) -> torch.Tensor:
    """
    Compute rotation- and scale-invariant geometric features for a batch of
    point sequences. Mirrors the hand-crafted features used in the existing
    MLP notebooks.

    Args:
        points: (B, N, 2)  — batch of N 2-D points each

    Returns:
        features: (B, N, 4) per-point features:
            [0] cumulative arc fraction up to point i  (chordal)
            [1] local segment length fraction
            [2] turning angle at point i (normalised to [0,1])
            [3] cumulative centripetal fraction up to point i
    """
    B, N, _ = points.shape

    # Segment lengths — chordal and centripetal
    diffs = points[:, 1:] - points[:, :-1]                     # (B, N-1, 2)
    chord_lens = torch.norm(diffs, dim=-1)                      # (B, N-1)
    centri_lens = torch.sqrt(chord_lens + 1e-8)                 # (B, N-1)

    chord_total  = chord_lens.sum(dim=1, keepdim=True) + 1e-8   # (B, 1)
    centri_total = centri_lens.sum(dim=1, keepdim=True) + 1e-8  # (B, 1)

    chord_frac  = chord_lens  / chord_total   # (B, N-1)
    centri_frac = centri_lens / centri_total  # (B, N-1)

    # Cumulative fractions — pad a leading zero for the first point
    chord_cum  = torch.cat([torch.zeros(B, 1, device=points.device),
                            torch.cumsum(chord_frac,  dim=1)], dim=1)  # (B, N)
    centri_cum = torch.cat([torch.zeros(B, 1, device=points.device),
                            torch.cumsum(centri_frac, dim=1)], dim=1)  # (B, N)

    # Local fraction — pad zeros at both ends so every point has a value
    local_frac = torch.cat([torch.zeros(B, 1, device=points.device),
                            chord_frac,
                            torch.zeros(B, 1, device=points.device)], dim=1)  # (B, N+1)
    local_frac = local_frac[:, :N]  # trim to (B, N)

    # Turning angles via cross-product and dot-product of consecutive edges
    # For interior points only; boundary points get angle = 0
    angles = torch.zeros(B, N, device=points.device)
    if N >= 3:
        v1 = diffs[:, :-1]   # (B, N-2, 2)
        v2 = diffs[:, 1:]    # (B, N-2, 2)
        cross = v1[:, :, 0] * v2[:, :, 1] - v1[:, :, 1] * v2[:, :, 0]
        dot   = (v1 * v2).sum(dim=-1)
        theta = torch.atan2(cross, dot)                   # (B, N-2)
        theta_norm = (theta + torch.pi) / (2 * torch.pi)  # → [0,1]
        angles[:, 1:N-1] = theta_norm

    features = torch.stack([chord_cum, local_frac, angles, centri_cum], dim=-1)
    return features  # (B, N, 4)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class BSplineDataset(Dataset):
    """
    Wraps point sequences of mixed dof (2, 3, 4 interior knots).

    Each sample contains:
        points    — (N, 2) tensor
        heuristic — (K,)   tensor of interior heuristic knot values ∈ (0,1)
        num_knots — int
        task      — 'curvature' | 'arc'
    """

    def __init__(
        self,
        points_list: List[np.ndarray],      # list of (N, 2) arrays
        heuristic_list: List[np.ndarray],   # list of (K,) interior knots
        task: str = "curvature",
    ):
        assert task in ("curvature", "arc")
        self.task = task
        self.samples = []
        for pts, heur in zip(points_list, heuristic_list):
            self.samples.append({
                "points":    torch.tensor(pts,  dtype=torch.double),
                "heuristic": torch.tensor(heur, dtype=torch.double),
                "num_knots": len(heur),
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_variable_length(batch):
    """
    Collate samples that may have different N (point count) and K (knot count).
    Pads point sequences to the longest in the batch; pads knots likewise.
    Returns a padding mask so attention ignores padded positions.
    """
    max_N = max(s["points"].shape[0] for s in batch)
    max_K = max(s["num_knots"] for s in batch)
    B = len(batch)

    points_pad    = torch.zeros(B, max_N, 2, dtype=torch.double)
    heuristic_pad = torch.zeros(B, max_K,    dtype=torch.double)
    num_knots     = torch.zeros(B,            dtype=torch.long)
    point_mask    = torch.ones(B, max_N,     dtype=torch.bool)   # True = ignore
    knot_mask     = torch.ones(B, max_K,     dtype=torch.bool)

    for i, s in enumerate(batch):
        N = s["points"].shape[0]
        K = s["num_knots"]
        points_pad[i, :N]    = s["points"]
        heuristic_pad[i, :K] = s["heuristic"]
        num_knots[i]          = K
        point_mask[i, :N]     = False   # False = attend
        knot_mask[i, :K]      = False

    return {
        "points":       points_pad,
        "heuristic":    heuristic_pad,
        "num_knots":    num_knots,
        "point_mask":   point_mask,
        "knot_mask":    knot_mask,
    }


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class KnotTransformer(nn.Module):
    """
    Sequence-to-sequence transformer that maps a variable-length 2-D point
    sequence to a variable-length sorted interior knot sequence ∈ (0,1).

    Architecture
    ────────────
    Encoder  : point coords + geometric features  →  context memory
    Decoder  : learned knot queries × memory       →  knot values

    The decoder uses DETR-style parallel decoding: N_max learnable query
    vectors each attend to the encoded point memory and independently predict
    one interior knot value. At inference time only the first `num_knots`
    outputs are used.

    Ordering is enforced by predicting log-increments and exponentiating
    (strictly positive increments → strictly increasing knots) followed by
    normalisation to (0,1). This is differentiable and avoids duplicate knots.
    """

    def __init__(
        self,
        d_model:           int = 64,
        nhead:             int = 4,
        num_encoder_layers:int = 3,
        num_decoder_layers:int = 3,
        dim_feedforward:   int = 256,
        dropout:           float = 0.1,
        max_points:        int = 10,   # maximum N in any sequence
        max_knots:         int = 6,    # maximum interior knots (4 dof + margin)
    ):
        super().__init__()
        self.d_model    = d_model
        self.max_knots  = max_knots
        self.max_points = max_points

        # Input: 2 coords + 4 geometric features = 6
        self.input_proj = nn.Linear(6, d_model)

        # Learned positional encoding for point sequence positions
        self.point_pos_embed = nn.Embedding(max_points, d_model)

        # Transformer encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout,
            batch_first=True, dtype=torch.double,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_encoder_layers)

        # Learned knot queries — one per possible interior knot position
        self.knot_queries = nn.Embedding(max_knots, d_model)

        # Transformer decoder
        dec_layer = nn.TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout,
            batch_first=True, dtype=torch.double,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_decoder_layers)

        # Predict log-increments: softplus ensures positivity, cumsum gives ordering
        self.knot_head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

        # Convert parameters to double precision throughout
        self.double()

    def forward(
        self,
        points:      torch.Tensor,   # (B, N, 2)
        point_mask:  torch.Tensor,   # (B, N) bool — True = padded position
        num_knots:   torch.Tensor,   # (B,)   int  — how many knots each sample needs
    ) -> torch.Tensor:               # (B, max_knots) — knot values, padded
        """
        Forward pass.

        Returns knot values in (0,1) sorted ascending.
        Positions beyond num_knots[b] are meaningless for sample b.
        """
        B, N, _ = points.shape

        # ── Geometric features ───────────────────────────────────────────────
        feats = compute_geometric_features(points)            # (B, N, 4)
        x = torch.cat([points, feats], dim=-1).to(torch.double)  # (B, N, 6)
        x = self.input_proj(x)                               # (B, N, d_model)

        # ── Positional encoding ──────────────────────────────────────────────
        pos_idx = torch.arange(N, device=points.device)
        x = x + self.point_pos_embed(pos_idx)                # (B, N, d_model)

        # ── Encoder ─────────────────────────────────────────────────────────
        memory = self.encoder(x, src_key_padding_mask=point_mask)  # (B, N, d_model)

        # ── Decoder with learnable knot queries ──────────────────────────────
        q_idx   = torch.arange(self.max_knots, device=points.device)
        queries = self.knot_queries(q_idx)                   # (max_knots, d_model)
        queries = queries.unsqueeze(0).expand(B, -1, -1)     # (B, max_knots, d_model)

        decoded = self.decoder(
            queries, memory,
            memory_key_padding_mask=point_mask,
        )                                                    # (B, max_knots, d_model)

        # ── Knot prediction ──────────────────────────────────────────────────
        # Raw log-increments → strictly positive via softplus → cumsum → normalise
        raw        = self.knot_head(decoded).squeeze(-1)     # (B, max_knots)
        increments = F.softplus(raw) + 1e-6                  # strictly positive
        cum        = torch.cumsum(increments, dim=-1)        # (B, max_knots)

        # Normalise each sample independently using its own num_knots
        knot_vals = torch.zeros_like(cum)
        for b in range(B):
            K = num_knots[b].item()
            total = cum[b, K - 1]
            knot_vals[b, :K] = cum[b, :K] / (total + 1e-8)  # → (0, 1)

        return knot_vals  # (B, max_knots)

    def predict(
        self,
        points:    torch.Tensor,   # (N, 2) single sample, no batch dim
        num_knots: int,
    ) -> torch.Tensor:             # (num_knots,)
        """Convenience wrapper for single-sample inference."""
        pts   = points.unsqueeze(0).to(device)               # (1, N, 2)
        mask  = torch.zeros(1, pts.shape[1], dtype=torch.bool, device=device)
        nk    = torch.tensor([num_knots], dtype=torch.long,  device=device)
        with torch.no_grad():
            out = self.forward(pts, mask, nk)
        return out[0, :num_knots]


# ---------------------------------------------------------------------------
# Full knot vector builder
# (adds the clamped boundary knots around the predicted interior knots)
# ---------------------------------------------------------------------------

def build_full_knot_vector(interior: torch.Tensor, N_points: int) -> torch.Tensor:
    """
    Build a full cubic B-spline knot vector from predicted interior knots.

    For N interpolation points the knot vector has N + 6 entries:
        [0,0,0,0, t1,...,tK, 1,1,1,1]   where K = N - 2
    """
    B  = interior.shape[0]
    dev = interior.device
    zeros = torch.zeros(B, 4, dtype=torch.double, device=dev)
    ones  = torch.ones( B, 4, dtype=torch.double, device=dev)
    return torch.cat([zeros, interior, ones], dim=1)  # (B, N+6)


# ---------------------------------------------------------------------------
# Phase 1 — Heuristic pretraining
# ---------------------------------------------------------------------------

class Phase1Trainer:
    """
    Trains the transformer to approximate heuristic knots (chordal for
    curvature minimisation, centripetal for arc-length minimisation).

    Loss: L1 between predicted interior knots and heuristic interior knots.
    """

    def __init__(
        self,
        model:      KnotTransformer,
        lr:         float = 1e-3,
        batch_size: int   = 64,
    ):
        self.model      = model.to(device)
        self.optimiser  = torch.optim.Adam(model.parameters(), lr=lr)
        self.loss_fn    = nn.L1Loss()
        self.batch_size = batch_size

    def train_epoch(self, dataloader: DataLoader) -> Tuple[float, float]:
        """
        One training epoch.
        Returns (mean_loss, accuracy) where accuracy = fraction of samples
        whose predicted knots give lower energy than the heuristic.
        (Energy check is skipped in phase 1 — accuracy here is knot-closeness.)
        """
        self.model.train()
        total_loss = 0.0
        n_batches  = 0

        for batch in dataloader:
            points     = batch["points"].to(device)
            heuristic  = batch["heuristic"].to(device)
            point_mask = batch["point_mask"].to(device)
            num_knots  = batch["num_knots"].to(device)

            self.optimiser.zero_grad()
            predicted = self.model(points, point_mask, num_knots)  # (B, max_K)

            # Only supervise on the valid (non-padded) knot positions
            B   = points.shape[0]
            loss = torch.tensor(0.0, dtype=torch.double, device=device)
            for b in range(B):
                K = num_knots[b].item()
                loss = loss + self.loss_fn(predicted[b, :K], heuristic[b, :K])
            loss = loss / B

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimiser.step()

            total_loss += loss.item()
            n_batches  += 1

        return total_loss / max(n_batches, 1)

    def run(
        self,
        dataset:    BSplineDataset,
        epochs:     int = 200,
        save_path:  str = "phase1_knot_transformer.pt",
        log_every:  int = 10,
    ):
        loader = DataLoader(
            dataset, batch_size=self.batch_size,
            shuffle=True, collate_fn=collate_variable_length,
        )
        best_loss = float("inf")
        history   = []

        print(f"Phase 1 — Heuristic pretraining  |  {epochs} epochs  |  "
              f"device: {device}")
        print("─" * 60)

        for epoch in range(1, epochs + 1):
            loss = self.train_epoch(loader)
            history.append(loss)

            if loss < best_loss:
                best_loss = loss
                torch.save(self.model.state_dict(), save_path)

            if epoch % log_every == 0 or epoch == 1:
                print(f"Epoch {epoch:>4d}/{epochs}  |  L1 loss: {loss:.6f}"
                      f"  |  best: {best_loss:.6f}")

        print(f"\nPhase 1 complete. Best model saved to '{save_path}'")
        return history


# ---------------------------------------------------------------------------
# Phase 2 — Energy fine-tuning (self-supervised)
# ---------------------------------------------------------------------------

class Phase2Trainer:
    """
    Fine-tunes the pretrained transformer to minimise B-spline energy
    (curvature or arc length) directly.

    The energy is computed via the differentiable functions in curvatureLayer
    so gradients flow back through the B-spline solver to the transformer
    weights — exactly as in the existing MLP self-supervised pipeline.

    Key differences from Phase 1:
      - Loss = mean energy over the batch (not L1 against knots)
      - Uses large batch sizes for stability (same finding as MLP experiments)
      - Filters out samples where phase 1 model already beats the heuristic
        (the 'good' list from the existing pipeline)
    """

    def __init__(
        self,
        model:       KnotTransformer,
        energy_fn,   # curvatureLayer.global_curvature or curvatureLayer.arc
        solve_fn,    # curvatureLayer.solve
        task:        str   = "curvature",   # 'curvature' | 'arc'
        lr:          float = 1e-4,
        batch_size:  int   = 512,
    ):
        assert task in ("curvature", "arc")
        self.model      = model.to(device)
        self.energy_fn  = energy_fn
        self.solve_fn   = solve_fn
        self.task       = task
        self.optimiser  = torch.optim.Adam(model.parameters(), lr=lr)
        self.batch_size = batch_size

    def _compute_energy(
        self,
        points:   torch.Tensor,   # (B, N, 2, 1)  — format for curvatureLayer
        knots:    torch.Tensor,   # (B, N+6)
    ) -> torch.Tensor:            # (B,)
        C   = self.solve_fn(points, knots)
        eng = self.energy_fn(C, knots, 3, 100)   # 100 pts for speed in training
        return eng

    def train_epoch(self, dataloader: DataLoader) -> Tuple[float, float]:
        self.model.train()
        total_energy  = 0.0
        total_better  = 0
        total_samples = 0
        n_batches     = 0

        for batch in dataloader:
            points_2d  = batch["points"].to(device)       # (B, N, 2)
            heuristic  = batch["heuristic"].to(device)    # (B, K)
            point_mask = batch["point_mask"].to(device)
            num_knots  = batch["num_knots"].to(device)

            self.optimiser.zero_grad()

            # Predict interior knots
            predicted = self.model(points_2d, point_mask, num_knots)  # (B, max_K)

            B, N, _ = points_2d.shape

            # Build full knot vectors for energy computation
            # For a mixed batch we process each unique (N, K) pair together
            # Here we process sample by sample for correctness (can be batched
            # if all samples in the loader share the same N)
            energy_sum = torch.tensor(0.0, dtype=torch.double, device=device)
            n_valid    = 0

            for b in range(B):
                K      = num_knots[b].item()
                n_pts  = (point_mask[b] == False).sum().item()
                pts_b  = points_2d[b, :n_pts]              # (n_pts, 2)

                # curvatureLayer expects (B, N, 2, 1) format
                pts_cl = pts_b.unsqueeze(0).unsqueeze(-1)  # (1, n_pts, 2, 1)

                # Build knot vector
                interior = predicted[b, :K].unsqueeze(0)   # (1, K)
                full_kv  = build_full_knot_vector(interior, n_pts)  # (1, n_pts+6)

                try:
                    eng = self._compute_energy(pts_cl, full_kv)
                    energy_sum = energy_sum + eng.mean()
                    n_valid   += 1
                except Exception:
                    # Degenerate knot vector — skip this sample
                    pass

            if n_valid == 0:
                continue

            loss = energy_sum / n_valid
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimiser.step()

            total_energy  += loss.item()
            total_samples += B
            n_batches     += 1

        mean_energy = total_energy / max(n_batches, 1)
        return mean_energy

    def filter_good_samples(
        self,
        dataset:    BSplineDataset,
        heuristic_energy_list: List[float],
    ) -> BSplineDataset:
        """
        Filters the dataset to keep only samples where the current model
        already achieves lower energy than the heuristic — the 'good' list
        from the existing pipeline.
        """
        self.model.eval()
        good_indices = []

        loader = DataLoader(
            dataset, batch_size=1,
            collate_fn=collate_variable_length,
        )

        with torch.no_grad():
            for idx, batch in enumerate(loader):
                points_2d  = batch["points"].to(device)
                point_mask = batch["point_mask"].to(device)
                num_knots  = batch["num_knots"].to(device)

                predicted  = self.model(points_2d, point_mask, num_knots)
                K          = num_knots[0].item()
                n_pts      = (point_mask[0] == False).sum().item()
                pts_b      = points_2d[0, :n_pts]
                pts_cl     = pts_b.unsqueeze(0).unsqueeze(-1)
                interior   = predicted[0, :K].unsqueeze(0)
                full_kv    = build_full_knot_vector(interior, n_pts)

                try:
                    eng = self._compute_energy(pts_cl, full_kv)
                    if eng.mean().item() <= heuristic_energy_list[idx]:
                        good_indices.append(idx)
                except Exception:
                    pass

        print(f"  Filtering: {len(good_indices)}/{len(dataset)} samples "
              f"pass phase-1 energy gate.")

        good_samples = [dataset.samples[i] for i in good_indices]
        filtered     = BSplineDataset.__new__(BSplineDataset)
        filtered.task    = dataset.task
        filtered.samples = good_samples
        return filtered

    def run(
        self,
        dataset:               BSplineDataset,
        heuristic_energy_list: List[float],
        epochs:                int   = 200,
        phase1_path:           str   = "phase1_knot_transformer.pt",
        save_path:             str   = "phase2_knot_transformer.pt",
        log_every:             int   = 10,
        filter_first:          bool  = True,
    ):
        # Load best phase 1 weights
        self.model.load_state_dict(torch.load(phase1_path, map_location=device))

        # Filter to samples where model already beats heuristic
        if filter_first:
            dataset = self.filter_good_samples(dataset, heuristic_energy_list)

        loader = DataLoader(
            dataset, batch_size=self.batch_size,
            shuffle=True, collate_fn=collate_variable_length,
        )

        best_energy = float("inf")
        history     = []

        print(f"\nPhase 2 — Energy fine-tuning ({self.task})  |  {epochs} epochs")
        print("─" * 60)

        for epoch in range(1, epochs + 1):
            mean_energy = self.train_epoch(loader)
            history.append(mean_energy)

            if mean_energy < best_energy:
                best_energy = mean_energy
                torch.save(self.model.state_dict(), save_path)

            if epoch % log_every == 0 or epoch == 1:
                print(f"Epoch {epoch:>4d}/{epochs}  |  "
                      f"mean energy: {mean_energy:.4f}  |  "
                      f"best: {best_energy:.4f}")

        print(f"\nPhase 2 complete. Best model saved to '{save_path}'")
        return history


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_vs_heuristic(
    model:               KnotTransformer,
    dataset:             BSplineDataset,
    heuristic_energy_list: List[float],
    energy_fn,
    solve_fn,
) -> dict:
    """
    Evaluates model predictions against heuristic energies.
    Returns accuracy (fraction where model beats heuristic), median energies,
    and mean L1 distance between predicted and heuristic knots.
    """
    model.eval()
    predicted_energies = []
    knot_distances     = []
    better_count       = 0

    loader = DataLoader(
        dataset, batch_size=1,
        collate_fn=collate_variable_length,
    )

    with torch.no_grad():
        for idx, batch in enumerate(loader):
            points_2d  = batch["points"].to(device)
            heuristic  = batch["heuristic"].to(device)
            point_mask = batch["point_mask"].to(device)
            num_knots  = batch["num_knots"].to(device)

            predicted  = model(points_2d, point_mask, num_knots)
            K          = num_knots[0].item()
            n_pts      = (point_mask[0] == False).sum().item()
            pts_b      = points_2d[0, :n_pts]
            pts_cl     = pts_b.unsqueeze(0).unsqueeze(-1)
            interior   = predicted[0, :K].unsqueeze(0)
            full_kv    = build_full_knot_vector(interior, n_pts)

            try:
                C   = solve_fn(pts_cl, full_kv)
                eng = energy_fn(C, full_kv, 3, 100).mean().item()
                predicted_energies.append(eng)

                kd = torch.abs(predicted[0, :K] - heuristic[0, :K]).mean().item()
                knot_distances.append(kd)

                if eng <= heuristic_energy_list[idx]:
                    better_count += 1
            except Exception:
                predicted_energies.append(float("nan"))
                knot_distances.append(float("nan"))

    valid_energies = [e for e in predicted_energies if not np.isnan(e)]
    valid_dists    = [d for d in knot_distances    if not np.isnan(d)]

    return {
        "accuracy":       better_count / len(dataset),
        "median_energy":  float(np.median(valid_energies)) if valid_energies else np.nan,
        "mean_knot_dist": float(np.mean(valid_dists))      if valid_dists    else np.nan,
        "n_valid":        len(valid_energies),
        "n_total":        len(dataset),
    }


# ---------------------------------------------------------------------------
# Quick-start example
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import curvatureLayer as cl

    print("KnotTransformer — quick smoke test")
    print("=" * 60)

    # Build a tiny synthetic dataset (replace with real data)
    rng = np.random.default_rng(42)
    samples_pts  = []
    samples_heur = []

    for _ in range(200):
        # Random 4-point sequence (2 dof → 2 interior knots)
        pts  = rng.uniform(0.1, 0.9, (4, 2))
        # Chordal heuristic interior knots
        diffs = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        total = diffs.sum() + 1e-8
        cum   = np.cumsum(diffs) / total
        heur  = cum[:-1]   # interior knots only
        samples_pts.append(pts)
        samples_heur.append(heur)

    for _ in range(200):
        # Random 5-point sequence (3 dof → 3 interior knots)
        pts   = rng.uniform(0.1, 0.9, (5, 2))
        diffs = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        total = diffs.sum() + 1e-8
        cum   = np.cumsum(diffs) / total
        heur  = cum[:-1]
        samples_pts.append(pts)
        samples_heur.append(heur)

    dataset = BSplineDataset(samples_pts, samples_heur, task="curvature")
    print(f"Dataset: {len(dataset)} samples (mixed 2 dof + 3 dof)")

    # Build model
    model = KnotTransformer(
        d_model=64, nhead=4,
        num_encoder_layers=3, num_decoder_layers=3,
        dim_feedforward=256, dropout=0.1,
        max_points=10, max_knots=6,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Phase 1
    p1 = Phase1Trainer(model, lr=1e-3, batch_size=32)
    p1.run(dataset, epochs=5, log_every=1, save_path="phase1_test.pt")

    # Single-sample inference demo
    model.load_state_dict(torch.load("phase1_test.pt", map_location=device))
    sample_pts   = torch.tensor(samples_pts[0], dtype=torch.double)
    predicted_kv = model.predict(sample_pts, num_knots=2)
    print(f"\nSample inference (2 dof):")
    print(f"  Heuristic knots : {samples_heur[0]}")
    print(f"  Predicted knots : {predicted_kv.cpu().numpy()}")
