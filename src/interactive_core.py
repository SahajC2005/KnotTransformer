"""Core compute for the interactive B-spline tool (widget-independent, testable)."""
import numpy as np, torch, matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
import bspline3d as b3
torch.set_default_dtype(torch.float64)

def curve_xyz(D, u, n=200):
    u = torch.as_tensor(u, dtype=torch.float64)
    su = torch.sort(u)[0].clamp(0.02, 0.98)
    P, U = b3.solve_control_points(D, su)
    t = torch.linspace(0, 1, n, dtype=torch.float64)
    C, _, _ = b3.eval_derivs(U, P, t)
    return C.detach().numpy()

QUAD = 100
def optimize_knots(D, K, steps=150, restarts=2, n_quad=QUAD):
    """Gradient-descend the bending energy. ALWAYS includes the chordal knots as a
    starting point, so the result can only match or beat chordal."""
    chord = b3.chordal_knots(D).numpy()
    inits = [chord] + [np.sort(np.random.uniform(0.15, 0.85, K)) for _ in range(restarts)]
    best_u = chord.copy()
    best_e = b3.safe_curvature(D, torch.tensor(chord).clamp(0.02, 0.98), n_quad)
    for init in inits:
        u = torch.tensor(np.sort(init).clip(0.03, 0.97), requires_grad=True)
        opt = torch.optim.Adam([u], lr=2e-2)
        for _ in range(steps):
            su = torch.sort(u)[0].clamp(0.02, 0.98)
            try: e = b3.bending_energy(D, su, n_quad=n_quad)
            except Exception: break
            if not torch.isfinite(e): break
            opt.zero_grad(); e.backward()
            torch.nn.utils.clip_grad_norm_([u], 1.0); opt.step()
        uu = torch.sort(u.detach())[0].clamp(0.02, 0.98)
        ev = b3.safe_curvature(D, uu, n_quad)
        if np.isfinite(ev) and ev < best_e:
            best_e, best_u = ev, uu.numpy()
    # hard guarantee: never return something worse than chordal
    ec = b3.safe_curvature(D, torch.tensor(chord).clamp(0.02, 0.98), n_quad)
    if not (best_e < ec):
        best_u = chord
    return best_u

def compute_figure(points, dim, K, source="Energy-optimized (target)", model=None, DEVICE="cpu"):
    """points: (N,dim) array. Returns (fig, kappa_chordal, kappa_cmp)."""
    D = torch.tensor(np.asarray(points, dtype=float))
    assert D.shape[0] == K + 2, f"need {K+2} points for {K} interior knots, got {D.shape[0]}"
    # left: chordal
    ck = b3.chordal_knots(D)
    kc = b3.safe_curvature(D, ck, QUAD)
    # right: comparison
    if source.startswith("Trained") and model is not None:
        pk = model.predict(D.to(DEVICE), num_knots=K).cpu().numpy()
        pk = 0.02 + 0.96 * pk
        cmp_label = "Trained model"
    else:
        pk = optimize_knots(D, K)
        cmp_label = "Energy-optimized (target)"
    km = b3.safe_curvature(D, pk, QUAD)

    Cc, Cm = curve_xyz(D, ck), curve_xyz(D, pk)
    Dn = D.numpy()
    fig = plt.figure(figsize=(12, 5.2))
    proj = "3d" if dim == 3 else None
    for pos, C, ttl, col, kap in [
        (121, Cc, "Chordal heuristic", "tab:blue", kc),
        (122, Cm, cmp_label, "tab:red", km)]:
        ax = fig.add_subplot(pos, projection=proj)
        if dim == 3:
            ax.plot(C[:,0], C[:,1], C[:,2], color=col, lw=2.5)
            ax.scatter(Dn[:,0], Dn[:,1], Dn[:,2], color="black", s=45, zorder=5)
            ax.set_zlabel("z")
        else:
            ax.plot(C[:,0], C[:,1], color=col, lw=2.5)
            ax.scatter(Dn[:,0], Dn[:,1], color="black", s=55, zorder=5)
            ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(f"{ttl}\n\u03ba = {kap:.2f}", fontsize=11)
        ax.set_xlabel("x"); ax.set_ylabel("y")
    ratio = km / kc if kc else float("nan")
    fig.suptitle(f"{dim}D  |  {K} interior knots ({K+2} points)   "
                 f"chordal \u03ba={kc:.1f}   comparison \u03ba={km:.1f}   ratio={ratio:.3f}",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    return fig, kc, km
