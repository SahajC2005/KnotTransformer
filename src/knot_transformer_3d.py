"""
knot_transformer_3d.py — KnotTransformer adapted for d-dimensional points
(works for d=2 or d=3). The ONLY change from the 2D model is the input:
the geometric-feature extractor consumes d coordinates instead of 2.
Encoder, decoder, knot-queries and knot-head are dimension-independent
because knots live in 1D parameter space.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def geometric_features(points, mask=None):
    """
    points : (B, N, d)   padded point sequences
    mask   : (B, N) bool, True = real point (optional)
    returns (B, N, d+3): [coords(d), dist_prev, dist_next, turning_angle]
    All terms are Euclidean / dot-product based, so they work for any d.
    """
    B, N, d = points.shape
    prev = torch.zeros_like(points); prev[:, 1:] = points[:, :-1]
    nxt  = torch.zeros_like(points); nxt[:, :-1] = points[:, 1:]
    dist_prev = torch.linalg.norm(points - prev, dim=-1, keepdim=True)
    dist_next = torch.linalg.norm(nxt - points, dim=-1, keepdim=True)
    v_in  = points - prev
    v_out = nxt - points
    cos = (v_in * v_out).sum(-1) / (
        torch.linalg.norm(v_in, dim=-1) * torch.linalg.norm(v_out, dim=-1) + 1e-9)
    angle = torch.acos(cos.clamp(-1 + 1e-6, 1 - 1e-6)).unsqueeze(-1)
    feats = torch.cat([points, dist_prev, dist_next, angle], dim=-1)
    return torch.nan_to_num(feats, 0.0)


class KnotTransformer3D(nn.Module):
    def __init__(self, dim=3, d_model=64, nhead=4,
                 num_encoder_layers=3, num_decoder_layers=3,
                 dim_feedforward=256, dropout=0.1,
                 max_points=12, max_knots=8):
        super().__init__()
        self.dim = dim
        self.max_points = max_points
        self.max_knots = max_knots
        in_feat = dim + 3                                  # coords + 3 geom features
        self.input_proj = nn.Linear(in_feat, d_model)
        self.pos_embed  = nn.Parameter(torch.randn(max_points, d_model) * 0.02)
        enc = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                         dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_encoder_layers)
        dec = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                         dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(dec, num_decoder_layers)
        self.knot_queries = nn.Parameter(torch.randn(max_knots, d_model) * 0.02)
        self.knot_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(),
            nn.Linear(d_model // 2, 1))

    def forward(self, points, point_mask, num_knots):
        """
        points     : (B, N, dim) padded
        point_mask : (B, N) bool, True = PAD (Transformer convention)
        num_knots  : (B,) int, how many interior knots each sample needs
        returns (B, max_knots) knot values in (0,1); use first num_knots[b] per sample.
        """
        B, N, _ = points.shape
        real = ~point_mask
        feats = geometric_features(points, real)
        x = self.input_proj(feats) + self.pos_embed[:N].unsqueeze(0)
        memory = self.encoder(x, src_key_padding_mask=point_mask)
        q = self.knot_queries.unsqueeze(0).expand(B, -1, -1)
        dec = self.decoder(q, memory, memory_key_padding_mask=point_mask)
        raw = self.knot_head(dec).squeeze(-1)              # (B, max_knots)
        inc = F.softplus(raw) + 1e-6
        cum = torch.cumsum(inc, dim=-1)
        knots = torch.zeros_like(cum)
        for b in range(B):
            K = int(num_knots[b])
            knots[b, :K] = cum[b, :K] / (cum[b, K - 1] + 1e-8)   # last -> 1.0 (known head limitation)
        return knots

    @torch.no_grad()
    def predict(self, points, num_knots):
        """points:(N,dim) single sample -> (K,) interior knots in (0,1)."""
        self.eval()
        pts = points.unsqueeze(0)
        mask = torch.zeros(1, pts.shape[1], dtype=torch.bool, device=pts.device)
        nk = torch.tensor([num_knots], device=pts.device)
        return self.forward(pts, mask, nk)[0, :num_knots]


def collate_3d(batch):
    """batch: list of (points(N,dim), num_knots). Pads to max N."""
    dim = batch[0][0].shape[1]
    maxN = max(p.shape[0] for p, _ in batch)
    B = len(batch)
    pts = torch.zeros(B, maxN, dim, dtype=torch.float64)
    mask = torch.ones(B, maxN, dtype=torch.bool)            # True = PAD
    nk = torch.zeros(B, dtype=torch.long)
    for i, (p, k) in enumerate(batch):
        n = p.shape[0]
        pts[i, :n] = p
        mask[i, :n] = False
        nk[i] = k
    return pts, mask, nk


# ----------------------------------------------------------------------
# Warm-start: transfer weights from a 2D checkpoint (skip the input layer)
# ----------------------------------------------------------------------
def warm_start_from(model, state_dict, verbose=True):
    """
    Copy weights from a (2D) checkpoint into this KnotTransformer3D where the key
    NAME matches and the SHAPE is compatible.

      * input_proj.*  -> always reinitialized (2D and 3D read different coords, and
                         the feature columns are not aligned: 3D inserts a z column).
      * pos_embed / knot_queries -> copied on the overlapping rows if sizes differ
                         (so different max_points / max_knots still transfer).
      * everything else (encoder, decoder, knot_head) -> copied only on exact shape.

    Returns (n_loaded, n_total, skipped[list of (name, reason)]).
    """
    tgt = model.state_dict()
    partial = {'pos_embed', 'knot_queries'}
    loaded, skipped = 0, []
    for k, v in tgt.items():
        if k.startswith('input_proj'):
            skipped.append((k, 'input layer (reinitialized by design)')); continue
        if k not in state_dict:
            skipped.append((k, 'absent in source')); continue
        s = state_dict[k]
        if tuple(s.shape) == tuple(v.shape):
            tgt[k] = s.clone(); loaded += 1
        elif k in partial and s.dim() == v.dim() and all(a <= b for a, b in zip(s.shape, v.shape)):
            sl = tuple(slice(0, a) for a in s.shape)
            vv = v.clone(); vv[sl] = s; tgt[k] = vv; loaded += 1
        elif k in partial and s.dim() == v.dim():
            sl = tuple(slice(0, min(a, b)) for a, b in zip(s.shape, v.shape))
            vv = v.clone(); vv[sl] = s[sl]; tgt[k] = vv; loaded += 1
        else:
            skipped.append((k, f'shape {tuple(s.shape)} != {tuple(v.shape)}'))
    # Safe fallback: if exact-name matching caught less than half the network, the
    # source likely uses different parent attribute names. Match by key-suffix + shape,
    # but ONLY when that (suffix, shape) is UNIQUE in the source (ambiguous matches such
    # as encoder-vs-decoder self_attn are skipped, never mis-mapped).
    if loaded < 0.5 * len(tgt):
        from collections import Counter
        def suf(k): return '.'.join(k.split('.')[1:])   # drop only the first token
        keys = [(suf(k), tuple(v.shape)) for k, v in state_dict.items()]
        counts = Counter(keys)
        lookup = {(suf(k), tuple(v.shape)): v for k, v in state_dict.items()}
        loaded, skipped = 0, []
        for k, v in tgt.items():
            if k.startswith('input_proj'):
                skipped.append((k, 'input layer (reinitialized by design)')); continue
            key = (suf(k), tuple(v.shape))
            if k in state_dict and tuple(state_dict[k].shape) == tuple(v.shape):
                tgt[k] = state_dict[k].clone(); loaded += 1
            elif counts.get(key, 0) == 1:
                tgt[k] = lookup[key].clone(); loaded += 1
            else:
                skipped.append((k, 'no unambiguous match'))
        if verbose:
            print(f"(suffix fallback, uniqueness-safe: {loaded}/{len(tgt)})")
    model.load_state_dict(tgt)
    if verbose:
        print(f"warm-start: transferred {loaded}/{len(tgt)} tensors from checkpoint")
        if loaded == 0:
            print("  WARNING: nothing transferred — the checkpoint's layer names do not")
            print("  match this model. Train from scratch, or share knot_transformer.py to align.")
        for k, why in skipped:
            print(f"    skip {k}  [{why}]")
    return loaded, len(tgt), skipped
