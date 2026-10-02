"""
bspline3d.py  —  dimension-general (2D/3D/nD) differentiable B-spline
interpolation + curvature bending energy, in pure PyTorch.

The ONLY mathematics that differs from the 2D code is the curvature:
a space curve uses the full vector cross product  kappa = ||C' x C''|| / ||C'||^3 ,
which reduces exactly to the 2D scalar formula when z = 0.

Everything is differentiable w.r.t. the interior knots, so the bending energy
can be used directly as a self-supervised Phase-2 loss.
"""
import torch

P_DEG = 3
_EPS = 1e-12


# ----------------------------------------------------------------------
# Knot / parameter construction
# ----------------------------------------------------------------------
def clamped_knots(u_int):
    """u_int: (..., K) interior knots in (0,1) -> full clamped cubic vector (..., K+8)."""
    *lead, K = u_int.shape
    z = u_int.new_zeros(*lead, P_DEG + 1)
    o = u_int.new_ones(*lead, P_DEG + 1)
    return torch.cat([z, u_int, o], dim=-1)


def data_params(u_int):
    """Data points are interpolated at t = [0, u_1, ..., u_K, 1]."""
    *lead, K = u_int.shape
    z = u_int.new_zeros(*lead, 1)
    o = u_int.new_ones(*lead, 1)
    return torch.cat([z, u_int, o], dim=-1)


# ----------------------------------------------------------------------
# Differentiable Cox-de Boor basis (single knot vector, batch of params)
# ----------------------------------------------------------------------
def _safe_div(num, den):
    return num / torch.where(den.abs() < _EPS, torch.ones_like(den), den) \
        * (den.abs() >= _EPS)


def basis_matrix(U, t, p=P_DEG, deriv=0):
    """
    U : (M,) knot vector (1D)
    t : (T,) parameters
    returns (T, n_ctrl) matrix of N_{i,p}^{(deriv)}(t_j), differentiable w.r.t. U and t.
    Ordinary recursion for degrees 1..(p-deriv); derivative recursion for the top `deriv` degrees.
    """
    M = U.shape[0]
    n_ctrl = M - p - 1
    Ul = U[:-1].unsqueeze(0)
    Ur = U[1:].unsqueeze(0)
    tt = t.unsqueeze(1)
    end = U[-1]
    ge = tt >= Ul
    lt = tt < Ur
    at_end = (tt >= end) & (Ur >= end)
    N = (ge & (lt | at_end)).to(U.dtype)            # degree 0

    for q in range(1, p + 1):
        ncol = M - q - 1
        Ni   = N[:, :ncol]
        Ni1  = N[:, 1:ncol + 1]
        Ui   = U[:ncol].unsqueeze(0)
        Uiq  = U[q:q + ncol].unsqueeze(0)
        Ui1  = U[1:1 + ncol].unsqueeze(0)
        Uiq1 = U[q + 1:q + 1 + ncol].unsqueeze(0)
        d1 = Uiq - Ui
        d2 = Uiq1 - Ui1
        if q <= p - deriv:
            term1 = _safe_div(tt - Ui, d1) * Ni
            term2 = _safe_div(Uiq1 - tt, d2) * Ni1
        else:
            qf = float(q)
            term1 = _safe_div(qf * Ni, d1)
            term2 = _safe_div(qf * Ni1, d2)
            N = term1 - term2
            continue
        N = term1 + term2
    return N


# ----------------------------------------------------------------------
# Collocation matrix and control-point solve (any dimension d)
# ----------------------------------------------------------------------
def build_collocation(u_int):
    """u_int: (K,) -> A (N+2, N+2), full knot vector U."""
    U = clamped_knots(u_int)
    tbar = data_params(u_int)                  # (N,)
    N = tbar.shape[0]
    rows_interp = basis_matrix(U, tbar, P_DEG, deriv=0)    # (N, N+2)
    ends = torch.stack([u_int.new_zeros(1).squeeze(0),
                        u_int.new_ones(1).squeeze(0)])      # t=0, t=1
    rows_end = basis_matrix(U, ends, P_DEG, deriv=2)        # (2, N+2)  C''=0 rows
    A = torch.cat([rows_interp, rows_end], dim=0)           # (N+2, N+2)
    return A, U


def solve_control_points(D, u_int):
    """
    D     : (N, d) data points (d = 2 or 3 or anything)
    u_int : (K,) interior knots, K = N-2
    returns control points P : (N+2, d)
    """
    A, U = build_collocation(u_int)
    N, d = D.shape
    rhs = torch.cat([D, D.new_zeros(2, d)], dim=0)          # (N+2, d)
    P = torch.linalg.solve(A, rhs)                          # differentiable solve
    return P, U


# ----------------------------------------------------------------------
# Curve derivatives and curvature bending energy
# ----------------------------------------------------------------------
def eval_derivs(U, P, t):
    """Return C, C', C'' at parameters t. P:(n_ctrl,d). Each output (T,d)."""
    B0 = basis_matrix(U, t, P_DEG, deriv=0)
    B1 = basis_matrix(U, t, P_DEG, deriv=1)
    B2 = basis_matrix(U, t, P_DEG, deriv=2)
    return B0 @ P, B1 @ P, B2 @ P


def _cross_norm(Cp, Cpp):
    """||C' x C''|| that works for d=2 (scalar cross) and d=3 (vector cross)."""
    d = Cp.shape[-1]
    if d == 2:
        return (Cp[..., 0] * Cpp[..., 1] - Cp[..., 1] * Cpp[..., 0]).abs()
    elif d == 3:
        return torch.linalg.norm(torch.cross(Cp, Cpp, dim=-1), dim=-1)
    else:
        # general nD: ||C'||^2||C''||^2 - (C'.C'')^2  (Gram determinant)
        a = (Cp * Cp).sum(-1)
        b = (Cpp * Cpp).sum(-1)
        c = (Cp * Cpp).sum(-1)
        return torch.sqrt(torch.clamp(a * b - c * c, min=0.0))


def bending_energy(D, u_int, n_quad=100):
    """
    Total squared-curvature bending energy   E = int_0^1 kappa^2 ||C'|| dt
    D:(N,d), u_int:(K,). Differentiable w.r.t. u_int. Works for d=2 and d=3.
    """
    P, U = solve_control_points(D, u_int)
    # midpoint quadrature on (0,1)
    t = (torch.arange(n_quad, dtype=D.dtype, device=D.device) + 0.5) / n_quad
    C0, Cp, Cpp = eval_derivs(U, P, t)
    speed = torch.linalg.norm(Cp, dim=-1)                  # ||C'||
    kappa = _cross_norm(Cp, Cpp) / (speed ** 3 + _EPS)     # curvature
    integrand = kappa ** 2 * speed                         # kappa^2 ||C'||
    return integrand.mean()                                 # (b-a)=1 -> mean == integral


# ----------------------------------------------------------------------
# Chordal heuristic (any dimension)
# ----------------------------------------------------------------------
def chordal_knots(D):
    """D:(N,d) -> interior knots (K,) by cumulative chord length."""
    diffs = torch.linalg.norm(D[1:] - D[:-1], dim=-1)
    cum = torch.cumsum(diffs, 0)
    total = cum[-1]
    frac = cum / (total + _EPS)
    return frac[:-1].clone()


def safe_curvature(D, u_int, n_quad=100):
    """Non-differentiable scalar energy with degeneracy guards (for eval)."""
    u = torch.as_tensor(u_int, dtype=D.dtype, device=D.device)
    su, _ = torch.sort(u)
    spacing_ok = (su.numel() < 2) or ((su[1:] - su[:-1]).min() >= 1e-4)
    if (not spacing_ok) or su.min() <= 1e-4 or su.max() >= 1 - 1e-4:
        return float('inf')
    try:
        with torch.no_grad():
            e = bending_energy(D, su, n_quad)
        v = float(e)
        return v if v == v and v != float('inf') else float('inf')
    except Exception:
        return float('inf')


# ----------------------------------------------------------------------
# Batched versions (all samples in a group share the same N) — for GPU training
# ----------------------------------------------------------------------
def basis_matrix_batch(U, t, p=P_DEG, deriv=0):
    """U:(G,M) batch of knot vectors, t:(T,) shared params -> (G,T,n_ctrl)."""
    G, M = U.shape
    Ul = U[:, :-1].unsqueeze(1)          # (G,1,M-1)
    Ur = U[:, 1:].unsqueeze(1)
    tt = t.view(1, -1, 1)                 # (1,T,1)
    end = U[:, -1].view(G, 1, 1)
    ge = tt >= Ul
    lt = tt < Ur
    at_end = (tt >= end) & (Ur >= end)
    N = (ge & (lt | at_end)).to(U.dtype)  # (G,T,M-1)
    for q in range(1, p + 1):
        ncol = M - q - 1
        Ni  = N[:, :, :ncol]
        Ni1 = N[:, :, 1:ncol + 1]
        Ui   = U[:, :ncol].unsqueeze(1)
        Uiq  = U[:, q:q + ncol].unsqueeze(1)
        Ui1  = U[:, 1:1 + ncol].unsqueeze(1)
        Uiq1 = U[:, q + 1:q + 1 + ncol].unsqueeze(1)
        d1 = Uiq - Ui
        d2 = Uiq1 - Ui1
        if q <= p - deriv:
            t1 = _safe_div(tt - Ui, d1) * Ni
            t2 = _safe_div(Uiq1 - tt, d2) * Ni1
            N = t1 + t2
        else:
            qf = float(q)
            N = _safe_div(qf * Ni, d1) - _safe_div(qf * Ni1, d2)
    return N



def basis_diag(U, tvals, p=P_DEG, deriv=0):
    """U:(G,M), tvals:(G,) per-sample scalar params -> (G,n_ctrl)."""
    G, M = U.shape
    Ul = U[:, :-1]; Ur = U[:, 1:]
    tt = tvals.unsqueeze(1)
    end = U[:, -1:].clone()
    ge = tt >= Ul; lt = tt < Ur
    at_end = (tt >= end) & (Ur >= end)
    N = (ge & (lt | at_end)).to(U.dtype)
    for q in range(1, p + 1):
        ncol = M - q - 1
        Ni  = N[:, :ncol]; Ni1 = N[:, 1:ncol + 1]
        Ui   = U[:, :ncol]; Uiq  = U[:, q:q + ncol]
        Ui1  = U[:, 1:1 + ncol]; Uiq1 = U[:, q + 1:q + 1 + ncol]
        d1 = Uiq - Ui; d2 = Uiq1 - Ui1
        if q <= p - deriv:
            N = _safe_div(tt - Ui, d1) * Ni + _safe_div(Uiq1 - tt, d2) * Ni1
        else:
            qf = float(q); N = _safe_div(qf * Ni, d1) - _safe_div(qf * Ni1, d2)
    return N


def bending_energy_batch(D, u_int, n_quad=100):
    """
    D:(G,N,d) same N across the group, u_int:(G,K) -> (G,) energies. Differentiable.
    """
    G, Nn, d = D.shape
    K = u_int.shape[1]
    z = u_int.new_zeros(G, P_DEG + 1); o = u_int.new_ones(G, P_DEG + 1)
    U = torch.cat([z, u_int, o], dim=1)                       # (G, K+8)
    tbar = torch.cat([u_int.new_zeros(G, 1), u_int, u_int.new_ones(G, 1)], dim=1)  # (G,N)
    # collocation rows must be evaluated at per-sample params -> loop only over the
    # N interpolation rows is avoided by building A per sample via batched basis at tbar.
    # tbar differs per sample, so evaluate basis at each sample's own params:
    A_rows = []
    # interpolation rows: for each data-parameter position j, basis at tbar[:,j]
    # do it vectorised over j using a (G,N,n_ctrl) gather
    n_ctrl = Nn + 2
    Aint = torch.stack([basis_diag(U, tbar[:, j], P_DEG, 0) for j in range(Nn)], dim=1)  # (G,N,n_ctrl)
    z_end = basis_diag(U, u_int.new_zeros(G), P_DEG, deriv=2)  # (G,n_ctrl)
    o_end = basis_diag(U, u_int.new_ones(G),  P_DEG, deriv=2)
    Aend = torch.stack([z_end, o_end], dim=1)                 # (G,2,n_ctrl)
    A = torch.cat([Aint, Aend], dim=1)                        # (G,N+2,N+2)
    rhs = torch.cat([D, D.new_zeros(G, 2, d)], dim=1)          # (G,N+2,d)
    P = torch.linalg.solve(A, rhs)                            # (G,N+2,d)
    tq = (torch.arange(n_quad, dtype=D.dtype, device=D.device) + 0.5) / n_quad
    B0 = basis_matrix_batch(U, tq, P_DEG, 0)                  # (G,T,n_ctrl)
    B1 = basis_matrix_batch(U, tq, P_DEG, 1)
    B2 = basis_matrix_batch(U, tq, P_DEG, 2)
    Cp  = torch.bmm(B1, P)                                     # (G,T,d)
    Cpp = torch.bmm(B2, P)
    speed = torch.linalg.norm(Cp, dim=-1)
    if d == 2:
        cn = (Cp[..., 0]*Cpp[..., 1] - Cp[..., 1]*Cpp[..., 0]).abs()
    else:
        cn = torch.linalg.norm(torch.cross(Cp, Cpp, dim=-1), dim=-1)
    kappa = cn / (speed**3 + _EPS)
    return (kappa**2 * speed).mean(dim=1)                     # (G,)
