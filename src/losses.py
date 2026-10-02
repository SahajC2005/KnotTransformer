"""
losses.py — auxiliary training losses for KnotTransformer Phase 2.

buffer_box_penalty
    Soft containment penalty on B-spline control points. A B-spline lies inside
    the convex hull of its control points, so keeping the control points inside
    a margin-expanded bounding box of the waypoints keeps the whole curve near
    the data. Intended to stop the model from "inflating" curves (huge loops have
    low curvature) to game the bending-energy loss.

    Used in notebooks/Phase2_BufferBox.ipynb as
        loss = capped_energy + LAMBDA_BOX * buffer_box_penalty(C, pts_g)
    with BOX_MARGIN = 0.25, LAMBDA_BOX = 200.0.

    Status: negative result on real-turn fine-tuning (see
    experiments/road_finetune/README.md). Kept for reproducibility.
"""
import torch

BOX_MARGIN = 0.25   # allow control points up to 25% beyond the data extent
LAMBDA_BOX = 200.0  # weight of the penalty relative to curvature energy


def buffer_box_penalty(C, pts_g, margin=BOX_MARGIN):
    """
    C      : control points from curvature_layer_nd.solve, [G, n_ctrl, 2]
             (trailing singleton dims are squeezed away)
    pts_g  : waypoints, [G, n_pts, 2]
    margin : fractional expansion of the waypoint bounding box
    returns: scalar — sum of squared distances by which control points
             fall outside the expanded box (0 if all are inside)
    """
    Cc = C
    while Cc.dim() > 3:
        Cc = Cc.squeeze(-1)
    lo = pts_g.min(dim=1).values                              # [G, 2]
    hi = pts_g.max(dim=1).values                              # [G, 2]
    center = (0.5 * (lo + hi)).unsqueeze(1)                   # [G, 1, 2]
    half = (0.5 * (hi - lo) * (1.0 + margin)).unsqueeze(1)    # [G, 1, 2]
    excess = torch.clamp((Cc - center).abs() - half, min=0.0)
    return (excess ** 2).sum()
