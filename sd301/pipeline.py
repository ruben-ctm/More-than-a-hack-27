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
    # --- broadcast tracker as a second source for the ball (aligned on player positions)
    use_broadcast_ball: bool = False   # tried and dropped, see technical document
    # --- gaps bounded by the touchlines (ball went out / came back from outside)
    bnd_dist_m: float = 3.0
    bnd_logit: float = 0.3
    min_gap_s: float = 1.0
    # --- self-supervised player-only models (fitted on pseudo-labels from this match's tracking)
    use_player_models: bool = True
    player_dead_gain: float = 0.5   # weight of the player-only dead-ball log-odds
    player_team_gain: float = 0.4   # weight of the player-only possession evidence
    pseudo_conf: float = 0.5        # |smoothed team evidence| needed for a possession pseudo-label
    feature_window_s: float = 3.0   # +/- window of the player-only features (chosen on held-out
                                    # pseudo-label AUC: dead 0.875->0.89, possession 0.851->0.855)
    # --- restarts
    same_team_after_dead_nats: float = 1.7   # ln(0.85/0.15): restarts mostly go to the opponent
    # --- Smart Tagging set pieces (team tag = side taking the restart)
    use_smart_tagging: bool = True
    tag_gain: float = 0.3
    tag_window_s: float = 12.0
    use_phase_tags: bool = True
    phase_gain: float = 0.25        # Metrica phases are automatic (~70-80 % reliable): keep weak
    phase_cap: float = 0.5          # overlapping phases never add more than this per frame
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
    L += boundary_gap_logit(bx, by, p)
    return L


def _line_distance(x, y):
    return np.minimum.reduce([x, PITCH_L - x, y, PITCH_W - y])


def boundary_gap_logit(bx, by, p):
    """+logit across ball gaps that start / end next to a touchline: the ball went out."""
    n = len(bx)
    out = np.zeros(n)
    missing = ~np.isfinite(bx)
    d = np.diff(np.concatenate(([0], missing.astype(np.int8), [0])))
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
    for s0, e0 in zip(starts, ends):
        if e0 - s0 < p.min_gap_s * FPS or s0 == 0 or e0 >= n:
            continue
        near_start = _line_distance(bx[s0 - 1], by[s0 - 1]) < p.bnd_dist_m
        near_end = _line_distance(bx[e0], by[e0]) < p.bnd_dist_m
        out[s0:e0] += p.bnd_logit * (int(near_start) + int(near_end))
    return out


def viterbi(team_ev, dlogit, p):
    """Decode H / A / D(after H) / D(after A) for one contiguous segment.

    Two dead states remember who had the ball before the stoppage, because restarts mostly go
    to the *other* side (throw-in, goal kick, corner and free kick are awarded against the last
    toucher) -- resuming with the same side costs `same_team_after_dead_nats`.
    Costs are negative log-likelihoods; being in play is costly where dead-ball evidence is high.
    """
    n = len(team_ev)
    c_sw = p.switch_s * FPS * 0.5          # evidence is in [-1, 1]; 0.5 avg => scale
    c_d = p.dead_switch_s * FPS * 0.5
    k = p.same_team_after_dead_nats
    HS, AS, DH, DA = 0, 1, 2, 3
    cost = np.zeros((n, 4))
    cost[:, HS] = -p.team_gain * team_ev + dlogit
    cost[:, AS] = +p.team_gain * team_ev + dlogit
    cost[:, DH] = cost[:, DA] = -dlogit
    T = np.full((4, 4), np.inf)
    np.fill_diagonal(T, 0.0)
    T[HS, AS] = T[AS, HS] = c_sw
    T[HS, DH] = c_d                      # H stops -> dead after H
    T[AS, DA] = c_d
    T[DH, AS] = c_d                      # restart by the opponent
    T[DA, HS] = c_d
    T[DH, HS] = c_d + k                  # restart by the same side
    T[DA, AS] = c_d + k
    T[DH, DA] = T[DA, DH] = k            # owner unknown while dead: pay the same prior
    # forbid the "impossible" shortcuts H->DA / A->DH (a stoppage keeps the last owner)
    T[HS, DA] = T[AS, DH] = c_d + k
    back = np.zeros((n, 4), np.int8)
    acc = cost[0].copy()
    ar = np.arange(4)
    for t in range(1, n):
        m = acc[:, None] + T
        back[t] = m.argmin(0)
        acc = m[back[t], ar] + cost[t]
    path = np.zeros(n, np.int8)
    path[-1] = acc.argmin()
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return np.where(path >= 2, D, path)


def _moving(x, k):
    from .features import moving_mean
    return moving_mean(x, k)


def fit_player_models(trk, bridged, ev_ball, p):
    """Self-supervised models for frames the ball tracker cannot see.

    dead-ball model : one model for the whole match, targets from ball-visible frames
                      (outside the lines or stationary = dead, travelling fast inside = alive).
    possession model: one model per half (sides swap), targets from frames where the ball is
                      tracked and one side clearly controls it.
    """
    from .features import dead_features, possession_features, fit_logistic
    Xd, yd = [], []
    pos = {}
    for half, (a, b) in HALVES.items():
        s = slice(a, b + 1)
        bxi, byi = bridged[half]
        k = np.isfinite(bxi)
        out = k & ((bxi < -p.out_margin_m) | (bxi > PITCH_L + p.out_margin_m) |
                   (byi < -p.out_margin_m) | (byi > PITCH_W + p.out_margin_m))
        w = FPS
        disp = np.hypot(np.roll(bxi, -w) - np.roll(bxi, w), np.roll(byi, -w) - np.roll(byi, w))
        dead = out | (k & (disp < 1.0))
        alive = k & ~out & (disp > 8.0)
        m = dead | alive
        X = dead_features(trk["px"][s], trk["py"][s], trk["ps"][s], trk["team"], k=int(p.feature_window_s * FPS))
        Xd.append(X[m])
        yd.append(dead[m].astype(float))
        # possession
        evs = _moving(ev_ball[half], FPS)
        mp = k & (np.abs(evs) > p.pseudo_conf)
        XP = possession_features(trk["px"][s], trk["py"][s], trk["ps"][s], trk["team"], k=int(p.feature_window_s * FPS))
        pos[half] = (fit_logistic(XP[mp], (evs[mp] > 0).astype(float)), XP)
    dead_model = fit_logistic(np.vstack(Xd), np.concatenate(yd))
    return dead_model, pos


HOLDER_CODES = {"BUILD UP", "PROGRESSION", "BALL IN FINAL THIRD", "BALL IN THE BOX",
                "PLAYERS IN THE BOX", "CREATING CHANCES", "LONG BALL", "ATTACKING TRANSITION",
                "GOALS"}
OTHER_CODES = {"DEFENDING IN MIDDLE THIRD", "DEFENDING IN ATTACKING THIRD",
               "DEFENDING IN DEFENSIVE THIRD", "DEFENSIVE TRANSITION"}


def tag_bias(n, a, tags, p):
    """Soft team evidence from Smart Tagging.

    * SET PIECES with a named side: that side restarts, so it holds the ball just after.
    * Returns (set-piece bias, phase bias); the caller decides where each may act.
    * Attacking phases name the side holding the ball; defending phases name the side that is
      defending, so the holder is the other one.  Both are automatic tags: small weight.
    """
    bias = np.zeros(n)
    phase = np.zeros(n)
    if p.use_phase_tags:
        for t in tags:
            team = t.get("Team")
            if team in (None, "N/A"):
                continue
            home = team.lower().startswith("barcelona")
            if t["code"] in HOLDER_CODES:
                sign = 1.0 if home else -1.0
            elif t["code"] in OTHER_CODES:
                sign = -1.0 if home else 1.0
            else:
                continue
            lo, hi = max(t["start"] - a, 0), min(t["end"] - a + 1, n)
            if hi > lo:
                phase[lo:hi] += sign * p.phase_gain
        phase = np.clip(phase, -p.phase_cap, p.phase_cap)
    for t in tags:
        team = t.get("Team")
        if t["code"] != "SET PIECES" or team in (None, "N/A"):
            continue
        sign = 1.0 if team.lower().startswith("barcelona") else -1.0
        lo = t["start"] - a
        hi = lo + int(p.tag_window_s * FPS)
        lo, hi = max(lo, 0), min(hi, n)
        if hi > lo:
            bias[lo:hi] += sign * p.tag_gain
    return np.clip(bias, -1, 1), phase


def run(trk, p=None, tags=None, return_debug=False, bro_ball=None):
    from .features import predict_logit, dead_features
    p = p or Params()
    if p.use_broadcast_ball and bro_ball is not None:
        trk = dict(trk)
        fx, fy = bro_ball
        gap = ~np.isfinite(trk["bx"]) & np.isfinite(fx)
        trk["bx"] = np.where(gap, fx, trk["bx"])
        trk["by"] = np.where(gap, fy, trk["by"])
    labels = np.full(N_FRAMES, "X", dtype="<U1")
    bridge = int(p.bridge_s * FPS)
    bridged, ev_ball = {}, {}
    for half, (a, b) in HALVES.items():
        s = slice(a, b + 1)
        bxi, byi = bridge_gaps(trk["bx"][s], bridge), bridge_gaps(trk["by"][s], bridge)
        bridged[half] = (bxi, byi)
        ev_ball[half] = team_evidence(trk["px"][s], trk["py"][s], trk["team"], bxi, byi, p)
    models = fit_player_models(trk, bridged, ev_ball, p) if p.use_player_models else None
    debug = {}
    for half, (a, b) in HALVES.items():
        s = slice(a, b + 1)
        bx, by = trk["bx"][s], trk["by"][s]
        bxi, byi = bridged[half]
        ev = ev_ball[half].copy()
        dl = dead_logit(bx, by, bxi, byi, trk["ps"][s], p)
        if models is not None:
            dead_model, pos = models
            xd = dead_features(trk["px"][s], trk["py"][s], trk["ps"][s], trk["team"], k=int(p.feature_window_s * FPS))
            dl = dl + p.player_dead_gain * np.clip(predict_logit(dead_model, xd), -4, 4)
            mdl, XP = pos[half]
            ev += p.player_team_gain * np.tanh(predict_logit(mdl, XP) / 2.0)
        if p.use_smart_tagging and tags:
            sp_bias, phase = tag_bias(len(ev), a, tags, p)
            ev += sp_bias
            # Phases last tens of seconds and span several possessions (DATA.md): they must not
            # override measured control, so they only speak where the ball is not tracked.
            ev += phase * (~np.isfinite(bxi))
        ev = np.clip(ev, -1.0, 1.0)
        path = viterbi(ev, dl, p)
        labels[s] = LABELS[path]
        debug[half] = dict(ev=ev, dl=dl)
    return (labels, debug) if return_debug else labels


def write_csv(path, labels):
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        fh.write("frame,label\n")
        fh.writelines("{},{}\n".format(i, l) for i, l in enumerate(labels))


def params_dict(p):
    return asdict(p)
