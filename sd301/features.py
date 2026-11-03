"""Player-only features (no ball needed) and a tiny numpy logistic regression.

These let the pipeline say something about frames where the ball is *not* tracked (41% of the
graded frames), using self-supervision: the training targets are pseudo-labels derived from the
tracking data itself on frames where the ball IS tracked.  No ground-truth possession data, no
video, no human labels are involved anywhere.
"""
import numpy as np

from .io import FPS, PITCH_L, PITCH_W


def moving_mean(x, k):
    """Centered moving mean ignoring NaN, window 2k+1."""
    x = np.asarray(x, np.float64)
    ok = np.isfinite(x)
    kern = np.ones(2 * k + 1)
    num = np.apply_along_axis(lambda c: np.convolve(c, kern, "same"), 0, np.where(ok, x, 0.0))
    cnt = np.apply_along_axis(lambda c: np.convolve(c, kern, "same"), 0, ok.astype(float))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt > 0, num / cnt, np.nan)


def dead_features(px, py, ps, team, k=FPS):
    """(n, f) match-state features: how the 22 players move collectively over +/-1 s."""
    spd = ps
    f = []
    f.append(moving_mean(np.nanmean(spd, 1), k))                                  # mean speed
    f.append(moving_mean(np.nanmean(np.where(spd > 3.0, 1.0, 0.0), 1), k))        # share running
    f.append(moving_mean(np.nanmean(np.where(spd < 0.8, 1.0, 0.0), 1), k))        # share standing
    cx, cy = np.nanmean(px, 1), np.nanmean(py, 1)
    ncx = np.hypot(np.roll(cx, -k) - np.roll(cx, k), np.roll(cy, -k) - np.roll(cy, k))
    f.append(ncx)                                                                  # centroid drift
    f.append(moving_mean(np.nanstd(px, 1), k))                                     # length of block
    f.append(moving_mean(np.nanstd(py, 1), k))                                     # width of block
    return np.column_stack(f)


def possession_features(px, py, ps, team, k=FPS):
    """(n, f) features describing which side is on the front foot, from players only."""
    home, away = team == 0, team == 1
    hx, ax = px[:, home], px[:, away]
    hy, ay = py[:, home], py[:, away]
    f = []
    # signed by the learner per half (it fits one logistic model per half, so direction is free)
    f.append(moving_mean(np.nanmean(hx, 1) - np.nanmean(ax, 1), k))               # who is further up
    f.append(moving_mean(np.nanstd(hx, 1) - np.nanstd(ax, 1), k))                 # block length diff
    f.append(moving_mean(np.nanstd(hy, 1) - np.nanstd(ay, 1), k))                 # width diff
    f.append(moving_mean(np.nanmean(ps[:, home], 1) - np.nanmean(ps[:, away], 1), k))
    # deepest outfield line: attacking side pushes it up (use 10th / 90th percentile of x)
    for q in (10, 90):
        f.append(moving_mean(np.nanpercentile(hx, q, 1) - np.nanpercentile(ax, q, 1), k))
    return np.column_stack(f)


def standardize(X, mu=None, sd=None):
    if mu is None:
        mu = np.nanmean(X, 0)
        sd = np.nanstd(X, 0) + 1e-9
    Z = (X - mu) / sd
    return np.nan_to_num(Z), mu, sd


def fit_logistic(X, y, l2=1.0, iters=50, w=None):
    """IRLS logistic regression with intercept, class-balanced.  Returns (beta, mu, sd)."""
    Z, mu, sd = standardize(X)
    Z1 = np.column_stack([np.ones(len(Z)), Z])
    sw = np.ones(len(y)) if w is None else w.copy()
    pos = y == 1
    sw = np.where(pos, sw * 0.5 / max(pos.mean(), 1e-6), sw * 0.5 / max(1 - pos.mean(), 1e-6))
    beta = np.zeros(Z1.shape[1])
    R = l2 * np.eye(Z1.shape[1])
    R[0, 0] = 0
    for _ in range(iters):
        eta = Z1 @ beta
        mu_ = 1 / (1 + np.exp(-np.clip(eta, -30, 30)))
        W = sw * mu_ * (1 - mu_) + 1e-9
        g = Z1.T @ (sw * (y - mu_)) - R @ beta
        Hs = Z1.T @ (Z1 * W[:, None]) + R
        step = np.linalg.solve(Hs, g)
        beta = beta + step
        if np.abs(step).max() < 1e-6:
            break
    return beta, mu, sd


def predict_logit(model, X):
    beta, mu, sd = model
    Z, _, _ = standardize(X, mu, sd)
    return beta[0] + Z @ beta[1:]


def auc(score, y):
    order = np.argsort(score)
    r = np.empty(len(score))
    r[order] = np.arange(1, len(score) + 1)
    n1 = y.sum()
    n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
