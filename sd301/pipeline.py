"""SD-301 possession pipeline (tracking-only, no training, no use of the ground truth).

Idea
----
1. Clean the ball track: short gaps are bridged, long gaps stay unknown.
2. Per frame, compute (a) *team evidence*: which side controls the ball (soft, distance weighted,
   so crowds do not flip a hard nearest-player decision) and (b) *dead-ball evidence*: ball outside
   the pitch, ball stationary, ball missing for a long time, players walking.
3. Decode the three states H / A / D with a Viterbi pass whose switching costs encode football
   knowledge: a hand-over needs ~1-2 s of sustained opposing control, a stoppage needs ~1 s of
   dead-ball evidence.  Missing ball = no evidence = the previous state persists.

All constants live in `Params` and are fixed a priori from football reasoning (distances in
metres, durations in seconds).  None is fitted on the ground-truth possession file or the video.
"""
from dataclasses import dataclass, asdict

import numpy as np

from .io import FPS, HALVES, N_FRAMES, PITCH_L, PITCH_W

H, A, D = 0, 1, 2
LABELS = np.array(["H", "A", "D"])


@dataclass
class Params:
    # --- ball cleaning
    bridge_s: float = 0.6          # linearly bridge ball gaps up to this long
    # --- team evidence
    sigma_m: float = 2.5           # distance kernel of the control weight
    ctrl_max_m: float = 6.0        # players further than this from the ball carry no weight
    evid_floor: float = 0.3        # denominator floor: weak evidence stays weak
    # --- dead-ball evidence (log-odds per frame, + means dead)
    out_margin_m: float = 0.3      # ball this far outside the lines counts as out of play
    out_logit: float = 1.5
    still_window_s: float = 2.0    # displacement window for "stationary"
    still_m: float = 1.5           # moved less than this over the window => stationary
    still_logit: float = 0.5
    moving_m: float = 5.0          # moved more than this => clearly in play
    moving_logit: float = -0.5
    gap_dead_s: float = 4.0        # a ball unseen this long is more likely dead than hidden
    gap_logit: float = 0.25
    alive_bias: float = -0.1       # prior: the ball is usually in play
    slow_players_ms: float = 1.2   # mean speed under this over the window => stoppage-like
    slow_logit: float = 0.15
    # --- sequence decoding (costs in nats; 1 frame of full evidence ~ 1 nat)
    team_gain: float = 1.0
    switch_s: float = 1.5          # H<->A needs about this much sustained evidence
    dead_switch_s: float = 1.0     # entering / leaving D


def bridge_gaps(v, max_gap):
    """Linear interpolation across NaN runs no longer than max_gap frames."""
    v = v.astype(np.float64).copy()
    ok = np.isfinite(v)
    if ok.sum() < 2:
        return v
    idx = np.arange(len(v))
    interp = np.interp(idx, idx[ok], v[ok])
    # run length of the missing run each frame belongs to
    run = missing_run_length(~ok)
    fill = (~ok) & (run <= max_gap)
    # do not extrapolate beyond the first / last observation
    fill &= (idx > idx[ok][0]) & (idx < idx[ok][-1])
    v[fill] = interp[fill]
    return v


def missing_run_length(missing):
    """For each frame, the length of the True-run it is in (0 where False)."""
    n = len(missing)
    out = np.zeros(n, np.int32)
    m = missing.astype(np.int8)
    d = np.diff(np.concatenate(([0], m, [0])))
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
    for s, e in zip(starts, ends):
        out[s:e] = e - s
    return out


def team_evidence(px, py, team, bx, by, p):
    """(n,) in [-1, 1]: +1 = home controls the ball, -1 = away, 0 = nobody / no ball."""
    n = len(bx)
    out = np.zeros(n, np.float64)
    known = np.isfinite(bx)
    dx = px[known] - bx[known, None]
    dy = py[known] - by[known, None]
    d2 = dx * dx + dy * dy
    w = np.exp(-d2 / (2 * p.sigma_m ** 2))
    w[~np.isfinite(w)] = 0.0
    w[d2 > p.ctrl_max_m ** 2] = 0.0
    sign = np.where(team == 0, 1.0, -1.0)[None, :]
    num = (w * sign).sum(1)
    den = w.sum(1)
    out[known] = num / np.maximum(den, p.evid_floor)
    # a controlling player pulls the ball: the closest one dominates, so scale by its weight
    out[known] *= np.minimum(1.0, w.max(1) / 0.5) if w.size else 1.0
    return np.clip(out, -1, 1)


def dead_logit(bx, by, bxi, byi, ps, p):
    """(n,) per-frame log-odds of the ball being dead (positive) vs in play (negative)."""
    n = len(bx)
    L = np.full(n, p.alive_bias)
    known = np.isfinite(bxi)

    out = known & ((bxi < -p.out_margin_m) | (bxi > PITCH_L + p.out_margin_m) |
                   (byi < -p.out_margin_m) | (byi > PITCH_W + p.out_margin_m))
    L[out] += p.out_logit

    w = int(p.still_window_s * FPS / 2)
    ahead = np.roll(np.where(known, bxi, np.nan), -w), np.roll(np.where(known, byi, np.nan), -w)
    back = np.roll(np.where(known, bxi, np.nan), w), np.roll(np.where(known, byi, np.nan), w)
    disp = np.hypot(ahead[0] - back[0], ahead[1] - back[1])
    still = np.isfinite(disp) & (disp < p.still_m)
    moving = np.isfinite(disp) & (disp > p.moving_m)
    L[still] += p.still_logit
    L[moving] += p.moving_logit

    gap = missing_run_length(~np.isfinite(bx))
    L[gap > p.gap_dead_s * FPS] += p.gap_logit

    # mean player speed over a +/-1 s window: stoppages have walking players
    spd = np.nanmean(ps, axis=1)
    spd = np.where(np.isfinite(spd), spd, np.nan)
    k = FPS
    kernel = np.ones(2 * k + 1)
    num = np.convolve(np.nan_to_num(spd), kernel, "same")
    cnt = np.convolve(np.isfinite(spd).astype(float), kernel, "same")
    mean_spd = np.where(cnt > 0, num / np.maximum(cnt, 1), np.nan)
    L[np.isfinite(mean_spd) & (mean_spd < p.slow_players_ms)] += p.slow_logit
    return L


def viterbi(team_ev, dlogit, p):
    """Decode H/A/D for one contiguous segment.  Costs are negative log-likelihoods."""
    n = len(team_ev)
    c_sw = p.switch_s * FPS * 0.5          # evidence is in [-1, 1]; 0.5 avg => scale
    c_d = p.dead_switch_s * FPS * 0.5
    # per-frame emission costs
    cost = np.zeros((n, 3))
    # dlogit > 0 means "dead": being alive is costly then, being dead is rewarded
    cost[:, H] = -p.team_gain * team_ev + dlogit
    cost[:, A] = +p.team_gain * team_ev + dlogit
    cost[:, D] = -dlogit
    # transitions
    T = np.array([[0, c_sw, c_d], [c_sw, 0, c_d], [c_d, c_d, 0]], dtype=np.float64)
    back = np.zeros((n, 3), np.int8)
    acc = cost[0].copy()
    for t in range(1, n):
        m = acc[:, None] + T                 # m[i, j]: from i to j
        back[t] = m.argmin(0)
        acc = m[back[t], np.arange(3)] + cost[t]
    path = np.zeros(n, np.int8)
    path[-1] = acc.argmin()
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def run(trk, p=None):
    p = p or Params()
    labels = np.full(N_FRAMES, "X", dtype="<U1")
    bridge = int(p.bridge_s * FPS)
    for half, (a, b) in HALVES.items():
        s = slice(a, b + 1)
        bx, by = trk["bx"][s], trk["by"][s]
        bxi, byi = bridge_gaps(bx, bridge), bridge_gaps(by, bridge)
        ev = team_evidence(trk["px"][s], trk["py"][s], trk["team"], bxi, byi, p)
        dl = dead_logit(bx, by, bxi, byi, trk["ps"][s], p)
        path = viterbi(ev, dl, p)
        labels[s] = LABELS[path]
    return labels


def write_csv(path, labels):
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        fh.write("frame,label\n")
        fh.writelines("{},{}\n".format(i, l) for i, l in enumerate(labels))


def params_dict(p):
    return asdict(p)
