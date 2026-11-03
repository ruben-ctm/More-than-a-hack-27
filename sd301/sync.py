"""Align the broadcast ATD to the tactical timeline using player positions only.

Both trackers report players in the same normalised pitch coordinates.  A broadcast frame shows a
subset of the 22 players (the camera follows the ball), so the matching cost of broadcast frame b
against tactical frame t is the median over the visible broadcast players of the distance to the
nearest tactical player.  The right t gives ~0.2-0.5 m, any other t > 1.7 m.

1. split the broadcast timeline into live runs (>= MIN_PLAYERS tracked players);
2. for 3 anchor frames per run, search the whole tactical timeline -> a per-run offset t - b
   (kept only if the anchors agree);
3. refine every live frame in a +/-REFINE window around the run offset.

Nothing here touches the possession ground truth or any video.
"""
import os
import numpy as np

from .io import PITCH_L, PITCH_W, N_FRAMES

MIN_PLAYERS = 8
REFINE = 40
ACCEPT_M = 1.0


def _cost(P, tx, ty):
    """median_k min_j |P_k - T_j| for each tactical frame row (tx, ty: (m, slots))."""
    dx = P[None, :, 0, None] - np.nan_to_num(tx[:, None, :], nan=1e6)
    dy = P[None, :, 1, None] - np.nan_to_num(ty[:, None, :], nan=1e6)
    d = np.sqrt(dx * dx + dy * dy).min(2)
    return np.median(d, 1)


def full_search(P, tpx, tpy, chunk=20000):
    out = np.empty(len(tpx))
    for s in range(0, len(tpx), chunk):
        out[s:s + chunk] = _cost(P, tpx[s:s + chunk], tpy[s:s + chunk])
    return out


def align(bpx, bpy, tpx, tpy, log=print):
    """Returns map_t (n_b,) int32 (-1 = unmatched) and cost (n_b,) float32."""
    nb = len(bpx)
    n = np.isfinite(bpx).sum(1)
    live = n >= MIN_PLAYERS
    # live runs, bridging holes shorter than 25 frames
    d = np.diff(np.r_[0, live.astype(np.int8), 0])
    st, en = np.where(d == 1)[0], np.where(d == -1)[0]
    runs = []
    for s, e in zip(st, en):
        if runs and s - runs[-1][1] < 25:
            runs[-1][1] = e
        else:
            runs.append([s, e])
    runs = [r for r in runs if r[1] - r[0] >= 50]
    log("live runs: %d covering %d frames" % (len(runs), sum(r[1] - r[0] for r in runs)))

    map_t = np.full(nb, -1, np.int32)
    cost = np.full(nb, np.inf, np.float32)
    for i, (s, e) in enumerate(runs):
        offs = []
        for frac in (0.15, 0.5, 0.85):
            b = int(s + frac * (e - s))
            while b < e and not live[b]:
                b += 1
            if not live[b]:
                continue
            m = np.isfinite(bpx[b])
            P = np.stack([bpx[b][m] * PITCH_L, bpy[b][m] * PITCH_W], 1)
            c = full_search(P, tpx, tpy)
            t = int(np.argmin(c))
            far = np.sort(c[np.abs(np.arange(len(c)) - t) > 200])[0]
            if c[t] < ACCEPT_M and far > 1.5 * c[t] + 0.3:
                offs.append(t - b)
        if not offs:
            continue
        off = int(np.median(offs))
        if len(offs) >= 2 and np.ptp(offs) > 30:
            # the run mixes timelines (e.g. a cut inside): handled by the refine window below
            pass
        # refine each live frame around the run offset (follow slow drift / jumps)
        cur = off
        for b in range(s, e):
            if not live[b]:
                continue
            lo, hi = max(b + cur - REFINE, 0), min(b + cur + REFINE + 1, len(tpx))
            m = np.isfinite(bpx[b])
            P = np.stack([bpx[b][m] * PITCH_L, bpy[b][m] * PITCH_W], 1)
            c = _cost(P, tpx[lo:hi], tpy[lo:hi])
            j = int(np.argmin(c))
            if c[j] < ACCEPT_M:
                map_t[b] = lo + j
                cost[b] = c[j]
                cur = lo + j - b
        if i % 20 == 0:
            log("run %d/%d  off %d  anchors %d" % (i, len(runs), off, len(offs)))
    return map_t, cost


def load_or_build(trk, bro, cache, log=print):
    if os.path.exists(cache):
        z = np.load(cache)
        return z["map_t"], z["cost"]
    map_t, cost = align(bro["px"], bro["py"], trk["px"], trk["py"], log)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.savez(cache, map_t=map_t, cost=cost)
    return map_t, cost


MAX_MATCH_COST_M = 0.6


def broadcast_ball_on_tactical(trk, cache_dir=None, log=print):
    """(bx, by) metres on the tactical timeline from the broadcast tracker, NaN elsewhere.

    Only frames whose player-position match is tight (< MAX_MATCH_COST_M) are used; when several
    broadcast frames map to the same tactical frame the best match wins.  The caller uses this
    only to fill gaps of the tactical ball.
    """
    from .io import load_broadcast
    bro = load_broadcast(cache_dir)
    from games import asset_paths
    cache_dir = cache_dir or os.path.join(os.path.dirname(asset_paths(source="broadcast")["atd_raw"]), "cache")
    map_t, cost = load_or_build(trk, bro, os.path.join(cache_dir, "broadcast_sync.npz"), log)
    bx, by = bro["bx"] * PITCH_L, bro["by"] * PITCH_W
    fx = np.full(N_FRAMES, np.nan)
    fy = np.full(N_FRAMES, np.nan)
    fc = np.full(N_FRAMES, np.inf)
    ok = (map_t >= 0) & np.isfinite(bx) & (cost < MAX_MATCH_COST_M)
    for b in np.where(ok)[0]:
        t = map_t[b]
        if cost[b] < fc[t]:
            fx[t], fy[t], fc[t] = bx[b], by[b], cost[b]
    return fx, fy
