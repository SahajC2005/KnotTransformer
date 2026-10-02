"""
road_generator.py — procedural road-like point sequences for training a
knot-placement model that transfers to real GPS turns.

Each sample is: straight-ish approach -> curved bend (random angle & radius)
-> straight-ish exit, with mild perpendicular GPS-style jitter. N waypoints
are sampled by arc length along the path, then the whole thing is centred and
uniformly scaled into ~[-0.5, 0.5]^2 (the model's box). This mirrors exactly
what the real-turn evaluation wrapper does, so train and test see the same
kind of input.
"""
import numpy as np


def _unit(theta):
    return np.array([np.cos(theta), np.sin(theta)])


def make_road_path(rng, n_dense=400):
    """Return a dense (n_dense, 2) polyline: approach -> bend -> exit."""
    # --- random geometry ---
    approach_len = rng.uniform(0.4, 1.4)
    exit_len     = rng.uniform(0.4, 1.4)
    turn_angle   = rng.uniform(np.deg2rad(20), np.deg2rad(150)) * rng.choice([-1, 1])
    radius       = rng.uniform(0.15, 0.6)          # bend tightness
    heading0     = rng.uniform(0, 2 * np.pi)       # initial direction

    # arc length of the bend
    arc_len = abs(turn_angle) * radius
    total = approach_len + arc_len + exit_len
    n_a = max(int(n_dense * approach_len / total), 2)
    n_c = max(int(n_dense * arc_len     / total), 2)
    n_e = n_dense - n_a - n_c

    pts = []
    # approach: straight line
    p = np.zeros(2); d = _unit(heading0)
    ts = np.linspace(0, approach_len, n_a, endpoint=False)
    pts.extend(p + np.outer(ts, d))
    p = p + approach_len * d

    # bend: circular arc
    # centre is perpendicular to heading, on the turn side
    side = np.sign(turn_angle) if turn_angle != 0 else 1
    perp = np.array([-d[1], d[0]]) * side
    centre = p + radius * perp
    a0 = np.arctan2(*(p - centre)[::-1])
    angs = a0 + np.linspace(0, turn_angle, n_c, endpoint=False)
    arc = centre + radius * np.stack([np.cos(angs), np.sin(angs)], axis=1)
    pts.extend(arc)
    p = arc[-1]
    # new heading after the bend
    heading1 = heading0 + turn_angle
    d = _unit(heading1)

    # exit: straight line
    ts = np.linspace(0, exit_len, max(n_e, 2))
    pts.extend(p + np.outer(ts, d))

    path = np.array(pts)
    # perpendicular GPS jitter (small, proportional to scale)
    jitter = rng.normal(0, 0.01, size=len(path))
    tang = np.gradient(path, axis=0)
    tang /= (np.linalg.norm(tang, axis=1, keepdims=True) + 1e-9)
    nrm = np.stack([-tang[:, 1], tang[:, 0]], axis=1)
    path = path + nrm * jitter[:, None]
    return path


def sample_waypoints(path, N, rng, jitter_idx=True):
    """Sample N points by arc length; optionally jitter the sample positions."""
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    total = s[-1]
    fracs = np.linspace(0, 1, N)
    if jitter_idx and N > 2:
        # nudge interior sample positions so spacing isn't perfectly uniform
        fracs[1:-1] += rng.normal(0, 0.5 / N, size=N - 2)
        fracs = np.clip(np.sort(fracs), 0, 1)
    targets = fracs * total
    idx = np.searchsorted(s, targets).clip(1, len(path) - 1)
    # linear interp between path vertices
    out = []
    for t, i in zip(targets, idx):
        s0, s1 = s[i - 1], s[i]
        w = 0.0 if s1 == s0 else (t - s0) / (s1 - s0)
        out.append(path[i - 1] * (1 - w) + path[i] * w)
    return np.array(out)


def normalize_box(seq):
    """Centre + uniform scale into ~[-0.5,0.5]^2 (matches the eval wrapper)."""
    c = seq.mean(axis=0)
    span = np.ptp(seq, axis=0).max()
    return (seq - c) / (span if span > 0 else 1.0)


def gen_road_points(n_samples, N, rng):
    """List of (N,2) normalized road-like waypoint sequences."""
    out = []
    tries = 0
    while len(out) < n_samples and tries < n_samples * 10:
        tries += 1
        path = make_road_path(rng)
        seq = sample_waypoints(path, N, rng)
        seqn = normalize_box(seq)
        # reject near-coincident waypoints (measured in the normalized box)
        d = np.linalg.norm(np.diff(seqn, axis=0), axis=1)
        if d.min() < 0.03:
            continue
        out.append(seqn)
    return out


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(3, 4, figsize=(14, 10))
    for ax in axes.flat:
        path = make_road_path(rng)
        N = rng.integers(6, 9)
        seq = sample_waypoints(path, N, rng)
        seqn = normalize_box(seq)
        pathn = normalize_box(path)
        ax.plot(pathn[:, 0], pathn[:, 1], color="0.7", lw=1, label="road")
        ax.scatter(seqn[:, 0], seqn[:, 1], c="crimson", s=40, zorder=5,
                   label=f"{N} waypoints")
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("Road-like synthetic training samples (approach -> bend -> exit)")
    fig.tight_layout()
    fig.savefig("road_samples.png", dpi=110)
    print("saved road_samples.png")
    # quick stats
    rng = np.random.default_rng(1)
    data = gen_road_points(500, 8, rng)
    print(f"generated {len(data)} samples, shape {data[0].shape}")
    dd = np.concatenate([np.linalg.norm(np.diff(s,axis=0),axis=1) for s in data])
    print(f"neighbor-distance range in normalized box: {dd.min():.3f}..{dd.max():.3f}")
