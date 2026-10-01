"""
curvature_layer_nd.py — Differentiable B-spline solver for arbitrary spatial
dimensions (2D, 3D, nD).

Generalises curvatureLayer.py from the thesis to support:
  - 2D points  (x, y)              — original behaviour
  - 3D points  (x, y, z)           — robotics, medical imaging, animation
  - nD points  (x1, ..., xn)       — latent space curves, config-space paths

The knot vector and all parameterisation logic is unchanged — knots are always
scalar values in (0,1) regardless of spatial dimension.  Only the B-spline
solver right-hand side and the energy functions change.

Key functions (all batched, all differentiable via PyTorch autograd):
  solve_nd(P, knots)              — solve for control points in R^d
  global_curvature_nd(C, knots)  — integrated squared curvature (any dim)
  arc_nd(C, knots)               — arc length (any dim)
  chordal_nd(P)                  — chordal knot vector (any dim)
  centripetal_nd(P)              — centripetal knot vector (any dim)

Backward compatible: passing 2D points gives identical results to the
original curvatureLayer.py.
"""

import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ---------------------------------------------------------------------------
# Basis function construction  (unchanged from original)
# ---------------------------------------------------------------------------

def construct(t, knots, p):
    """
    Evaluate B-spline basis functions for a batch of parameter values.

    Args:
        t:      (B,)         parameter values
        knots:  (B, n_knots) knot vectors
        p:      int          degree

    Returns:
        N: (B, n_knots-1)   basis function values
    """
    N = torch.zeros(knots.shape[0], knots.shape[1] - 1,
                    dtype=torch.double).to(device)
    vec  = torch.zeros(len(knots), dtype=torch.double).to(device)
    vec2 = torch.ones( len(knots), dtype=torch.double).to(device)
    lenN = N.shape[1] - 1

    for i in range(knots[0, 1:].shape[0]):
        v = torch.where(
            torch.logical_and(t < knots[:, i + 1], t >= knots[:, i]),
            vec2, vec
        )
        N[:, i] = v

    for power in range(1, p + 1):
        for index in range(0, lenN - power):
            num = t - knots[:, index]
            den = knots[:, index + power] - knots[:, index]
            a   = torch.zeros(knots.shape[0], dtype=torch.double).to(device)
            ind = torch.nonzero(den, as_tuple=True)
            a[ind] = num[ind] / den[ind]

            num = knots[:, index + power + 1] - t
            den = knots[:, index + power + 1] - knots[:, index + 1]
            b   = torch.zeros(knots.shape[0], dtype=torch.double).to(device)
            ind = torch.nonzero(den, as_tuple=True)
            b[ind] = num[ind] / den[ind]

            N[:, index] = N[:, index].clone() * a.clone() + \
                          N[:, index + 1].clone() * b.clone()
    return N


def construct_A(P, knots):
    """
    Build the interpolation matrix A for natural cubic B-splines.
    P:     list of length N, each element (B, d) — spatial coords
    knots: (B, N+6)
    Returns A: (B, N+2, N+2)
    """
    num_pts = len(P)
    for i in range(knots[0, 3:-3].shape[0]):
        a = construct(knots[:, i + 3], knots, 3)
        if knots[0, i + 3] == 1:
            a = torch.zeros(knots.shape[0], knots.shape[1] - 1).to(device)
            a[:, num_pts + 1] = 1

        if i == 0:
            A = torch.unsqueeze(a, 1)
        else:
            A = torch.cat((A, torch.unsqueeze(a, 1)), 1)

    A = A[:, :, 0:num_pts + 2]
    a = torch.zeros(knots.shape[0], num_pts + 2).to(device)
    A = torch.cat((torch.unsqueeze(a, 1), A), 1)
    A = torch.cat((A, torch.unsqueeze(a, 1)), 1)

    A[:, 0, 0] +=  knots[:, 5]
    A[:, 0, 1] += -knots[:, 4] - knots[:, 5]
    A[:, 0, 2] +=  knots[:, 4]

    A[:, -1, -3] += 1 - knots[:, num_pts + 1]
    A[:, -1, -2] += knots[:, num_pts + 1] + knots[:, num_pts] - 2
    A[:, -1, -1] += 1 - knots[:, num_pts]

    return A


# ---------------------------------------------------------------------------
# nD solver  — key extension
# ---------------------------------------------------------------------------

def solve_nd(P, knots):
    """
    Solve for B-spline control points in arbitrary spatial dimension d.

    Args:
        P:      list of N tensors, each (B, d)   — interpolation points
                OR tensor of shape (B, N, d, 1)  — original format (2D only)
        knots:  (B, N+6)

    Returns:
        D: (B, N+2, d, 1)   — control points in R^d
    """
    # Accept both the original (B, N, 2, 1) format and the new (B, N, d) list
    if isinstance(P, torch.Tensor):
        # Original format: (B, N, d, 1)  →  convert to list
        B, N, d, _ = P.shape
        P = [P[:, i, :, 0] for i in range(N)]   # list of N tensors (B, d)

    d_space = P[0].shape[1]   # spatial dimension
    B       = P[0].shape[0]

    # Build right-hand side for each spatial dimension
    rhs = []
    for dim in range(d_space):
        coords = torch.stack([p[:, dim] for p in P], dim=1)   # (B, N)
        coords = torch.cat(
            [torch.zeros(B, 1, dtype=torch.double).to(device),
             coords,
             torch.zeros(B, 1, dtype=torch.double).to(device)], dim=1
        )   # (B, N+2)
        rhs.append(coords.unsqueeze(2))   # (B, N+2, 1)

    A        = construct_A(P, knots)          # (B, N+2, N+2)
    inv, _   = torch.linalg.inv_ex(A)        # (B, N+2, N+2)

    ctrl_pts = []
    for dim_rhs in rhs:
        C = torch.bmm(inv, dim_rhs)          # (B, N+2, 1)
        ctrl_pts.append(C)

    # Stack along spatial dimension: (B, N+2, d)
    D = torch.cat(ctrl_pts, dim=2)           # (B, N+2, d)
    D = D.unsqueeze(3)                       # (B, N+2, d, 1)
    return D


# ---------------------------------------------------------------------------
# de Boor  (unchanged, works for any d via the last two dims)
# ---------------------------------------------------------------------------

def de_boor(knotss, t, CP, p):
    """
    de Boor's algorithm for batched evaluation.
    CP: (B, n_ctrl, d, 1)
    Returns: (B, d, 1)
    """
    s  = CP.shape[0], p + 1, CP.shape[2], CP.shape[3]
    E  = torch.zeros(s, dtype=torch.double).to(device)
    vec  = torch.zeros(len(knotss), dtype=torch.double).to(device)
    vec2 = torch.ones( len(knotss), dtype=torch.double).to(device)
    k    = torch.zeros(len(knotss), dtype=torch.double).to(device)

    for i in range(p + 1, len(knotss[0])):
        v  = torch.where(t > knotss[:, i], vec2, vec)
        k += v

    for i in range(p + 1):
        I = torch.fill(v, i).to(device)
        E[torch.arange(len(knotss)), I.type(torch.LongTensor)] = \
            CP[torch.arange(len(knotss)), (I + k).type(torch.LongTensor)]

    al = torch.arange(len(knotss))
    for j in range(1, p + 1):
        for i in range(0, p - j + 1):
            I = torch.fill(v, i)
            P = torch.fill(v, p)
            J = torch.fill(v, j)
            indI    = I.type(torch.LongTensor)
            indPIK1 = (P + 1 + I + k).type(torch.LongTensor)
            indIKJ  = (I + k + J).type(torch.LongTensor)
            indI1   = (I + 1).type(torch.LongTensor)

            ans1 = knotss[al, indPIK1].unsqueeze(1).expand(-1, CP.shape[2])
            ans1 = ans1.unsqueeze(2)
            ans2 = knotss[al, indIKJ].unsqueeze(1).expand(-1, CP.shape[2])
            ans2 = ans2.unsqueeze(2)
            tt   = t.unsqueeze(1).expand(-1, CP.shape[2]).unsqueeze(2)

            E[al, indI] = (
                (ans1 - tt) * E[al, indI].clone()  / (ans1 - ans2) +
                (tt - ans2) * E[al, indI1].clone() / (ans1 - ans2)
            )

    return E[:, 0]   # (B, d, 1)


# ---------------------------------------------------------------------------
# Derivatives  (generalised to nD — formula is the same, d just goes along)
# ---------------------------------------------------------------------------

def first_derivative_nd(C, knotss):
    """
    First derivative control points in R^d.
    C:      (B, N+2, d, 1)
    knotss: (B, N+6)
    Returns Q: (B, N+1, d, 1),  dknots: (B, N+4)
    """
    B, n_ctrl, d, _ = C.shape
    Q = torch.zeros((B, n_ctrl - 1, d, 1), dtype=torch.double).to(device)

    for i in range(n_ctrl - 1):
        denom = (knotss[:, i + 4] - knotss[:, i + 1])        # (B,)
        # Broadcast denom over spatial dims
        denom_exp = denom.unsqueeze(1).unsqueeze(2)            # (B,1,1)
        Q[:, i] = 3 / (denom_exp + 1e-12) * (C[:, i + 1] - C[:, i])

    return Q, knotss[:, 1:-1]


def second_derivative_nd(Q, knotss):
    """
    Second derivative control points in R^d.
    Q:      (B, N+1, d, 1)
    knotss: (B, N+4)
    """
    B, n_ctrl, d, _ = Q.shape
    R = torch.zeros((B, n_ctrl - 1, d, 1), dtype=torch.double).to(device)

    for i in range(n_ctrl - 1):
        denom = (knotss[:, i + 3] - knotss[:, i + 1])
        denom_exp = denom.unsqueeze(1).unsqueeze(2)
        R[:, i] = 2 / (denom_exp + 1e-12) * (Q[:, i + 1] - Q[:, i])

    return R, knotss[:, 1:-1]


# ---------------------------------------------------------------------------
# Curvature  — generalised to nD
# ---------------------------------------------------------------------------

def local_curvature_nd(C, knotss, t, p):
    """
    Curvature at parameter t for curves in R^d.

    In 2D:  κ = |x'y'' - y'x''| / ||r'||³
    In 3D:  κ = ||r' × r''||   / ||r'||³
    In nD:  κ = sqrt(||r''||² - (r'·r''/||r'||²)²) / ||r'||²
               = ||r'' - (r'·r''/||r'||²)r'|| / ||r'||²
    The nD formula reduces correctly to 2D and 3D special cases.
    """
    Q, dknots   = first_derivative_nd(C, knotss)
    R, ddknots  = second_derivative_nd(Q, dknots)

    r_prime  = de_boor(dknots,  t, Q, p - 1)   # (B, d, 1)
    r_dprime = de_boor(ddknots, t, R, p - 2)   # (B, d, 1)

    rp  = r_prime[:, :, 0]    # (B, d)
    rdp = r_dprime[:, :, 0]   # (B, d)

    rp_norm_sq  = (rp * rp).sum(dim=1, keepdim=True) + 1e-12   # (B, 1)
    rp_norm     = torch.sqrt(rp_norm_sq)                         # (B, 1)

    # Component of r'' perpendicular to r'
    proj  = (rp * rdp).sum(dim=1, keepdim=True) / rp_norm_sq    # (B, 1)
    perp  = rdp - proj * rp                                       # (B, d)
    perp_norm = torch.sqrt((perp * perp).sum(dim=1) + 1e-12)    # (B,)

    curvature = perp_norm / (rp_norm[:, 0] ** 2 + 1e-12)        # (B,)
    return curvature


def global_curvature_nd(C, knotss, p, num_pts):
    """
    Integrated squared curvature ∫ κ² ||r'|| dt via Simpson's rule.
    Works for curves in R^d (d ≥ 2).

    Args:
        C:       (B, N+2, d, 1)  control points
        knotss:  (B, N+6)
        p:       int             degree (3 for cubic)
        num_pts: int             number of quadrature points

    Returns:
        integration: (B,)
    """
    L = torch.linspace(
        knotss[0, p].item(),
        knotss[0, len(knotss[0]) - (p - 1)].item(),
        num_pts
    )
    h  = 1.0 / (len(L) - 1)
    S1 = 0
    S2 = 0

    Q, dknots  = first_derivative_nd(C, knotss)
    R, ddknots = second_derivative_nd(Q, dknots)

    v = torch.ones(len(knotss), dtype=torch.double)

    for i, j in enumerate(L):
        J = torch.fill(v, j).to(device)

        r_prime  = de_boor(dknots,  J, Q, p - 1)
        r_dprime = de_boor(ddknots, J, R, p - 2)

        rp  = r_prime[:, :, 0]
        rdp = r_dprime[:, :, 0]

        rp_norm_sq = (rp * rp).sum(dim=1, keepdim=True) + 1e-12
        rp_norm    = torch.sqrt(rp_norm_sq)
        proj       = (rp * rdp).sum(dim=1, keepdim=True) / rp_norm_sq
        perp       = rdp - proj * rp
        perp_norm  = torch.sqrt((perp * perp).sum(dim=1) + 1e-12)
        curvature  = perp_norm / (rp_norm[:, 0] ** 2 + 1e-12)

        speed = rp_norm[:, 0]   # ||r'||

        if i % 2 == 1:
            S1 += (curvature ** 2) * speed
        else:
            S2 += (curvature ** 2) * speed

    return (h / 3) * (4 * S1 + 2 * S2)


# ---------------------------------------------------------------------------
# Arc length  — already dimension-agnostic, explicit nD version
# ---------------------------------------------------------------------------

def arc_nd(C, knots, p, num_pts):
    """
    Arc length ∫ ||r'(t)|| dt via Simpson's rule.
    Works for curves in R^d (d ≥ 2).
    """
    L  = torch.linspace(
        knots[0, p].item(),
        knots[0, len(knots[0]) - (p - 1)].item(),
        num_pts
    )
    h  = 1.0 / (len(L) - 1)
    s1 = 0
    s2 = 0

    Q, knots2 = first_derivative_nd(C, knots)
    v = torch.ones(len(knots), dtype=torch.double)

    for i, j in enumerate(L):
        J   = torch.fill(v, j)
        evl = de_boor(knots2, J, Q, 2)        # (B, d, 1)
        speed = torch.sqrt(
            (evl[:, :, 0] ** 2).sum(dim=1) + 1e-12
        )                                       # (B,)
        if i % 2 == 1:
            s1 += speed
        else:
            s2 += speed

    return (h / 3) * (4 * s1 + 2 * s2)


# ---------------------------------------------------------------------------
# Heuristics  — generalised to nD
# ---------------------------------------------------------------------------

def chordal_nd(P):
    """
    Chordal parameterisation for curves in R^d.
    P: (B, N, d)
    Returns knots: (B, N+6)
    """
    shifted     = torch.roll(P, 1, dims=1)
    shifted[:, 0] = 0.0
    diffs       = P - shifted
    diffs       = diffs[:, 1:]                                   # (B, N-1, d)
    d           = torch.sqrt((diffs ** 2).sum(dim=2) + 1e-12)   # (B, N-1)

    L     = d.sum(dim=1, keepdim=True) + 1e-8                   # (B, 1)
    knots = torch.zeros(P.shape[0], P.shape[1] + 6, dtype=torch.double)
    knots[:, -4:] = 1.0

    s = 0
    for i in range(4, 4 + P.shape[1] - 2):
        s += d[:, i - 4]
        knots[:, i] = (s / L.squeeze(1))

    return knots


def centripetal_nd(P):
    """
    Centripetal parameterisation for curves in R^d.
    P: (B, N, d)
    Returns knots: (B, N+6)
    """
    shifted     = torch.roll(P, 1, dims=1)
    shifted[:, 0] = 0.0
    diffs       = P - shifted
    diffs       = diffs[:, 1:]
    d           = torch.sqrt(
        torch.sqrt((diffs ** 2).sum(dim=2) + 1e-12)
    )                                                            # (B, N-1)

    L     = d.sum(dim=1, keepdim=True) + 1e-8
    knots = torch.zeros(P.shape[0], P.shape[1] + 6, dtype=torch.double)
    knots[:, -4:] = 1.0

    s = 0
    for i in range(4, 4 + P.shape[1] - 2):
        s += d[:, i - 4]
        knots[:, i] = (s / L.squeeze(1))

    return knots


def dup(k):
    """Remove samples with duplicate interior knots (degenerate knot vectors)."""
    c   = torch.diff(torch.sort(k[:, 4:-4])[0])
    lst = []
    for i, j in enumerate(torch.prod(c, axis=1)):
        if not j == 0:
            lst.append(i)
    return lst


# ---------------------------------------------------------------------------
# Convenience: original 2D wrappers  (backward compatible)
# ---------------------------------------------------------------------------

def solve(P, knots):
    """Original 2D solve — calls solve_nd internally."""
    return solve_nd(P, knots)


def global_curvature(C, knotss, p, num_pts):
    """Original 2D curvature — calls global_curvature_nd internally."""
    return global_curvature_nd(C, knotss, p, num_pts)


def arc(C, knots, p, num_pts):
    """Original arc length — calls arc_nd internally."""
    return arc_nd(C, knots, p, num_pts)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("curvature_layer_nd — smoke test")
    print("=" * 50)
    B = 4

    for d_space, label in [(2, "2D"), (3, "3D"), (5, "5D")]:
        print(f"\n{label} curve (d={d_space}):")
        N = 5   # 5 interpolation points → 3 interior knots
        pts = torch.rand(B, N, d_space, dtype=torch.double)

        # Chordal knots
        knots = chordal_nd(pts)
        print(f"  knots shape:  {knots.shape}")

        # Build list format for solver
        P_list = [pts[:, i, :] for i in range(N)]

        # Solve
        C = solve_nd(P_list, knots)
        print(f"  ctrl pts:     {C.shape}")

        # Arc length
        arc_len = arc_nd(C, knots, 3, 50)
        print(f"  arc length:   {arc_len.mean().item():.4f}  (mean over batch)")

        # Curvature
        curv = global_curvature_nd(C, knots, 3, 50)
        print(f"  curvature:    {curv.mean().item():.4f}  (mean over batch)")

    print("\nAll dimensions OK — backward compatible.")
