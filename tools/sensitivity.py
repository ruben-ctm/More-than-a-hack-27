"""Ground-truth-free robustness check: perturb each hand-set constant by +/-30 % and report how much
the decoded labels move (frame agreement with the reference run, and possession count).  Writes
a markdown table to stdout."""
import os, sys, warnings
from dataclasses import fields, replace
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sd301.io import load_tracking, load_smart_tagging, HALVES
from sd301.pipeline import Params, run

NUMERIC = ["sigma_m", "ctrl_max_m", "switch_s", "dead_switch_s", "still_m", "gap_dead_s",
           "gap_logit", "player_dead_gain", "player_team_gain", "same_team_after_dead_nats",
           "tag_gain", "phase_gain", "alive_bias", "out_logit", "still_logit"]
trk, tags = load_tracking(), load_smart_tagging()
base = run(trk, Params(), tags)
mask = np.zeros(len(base), bool)
for a, b in HALVES.values():
    mask[a:b + 1] = True

def n_poss(l):
    n = 0
    for a, b in HALVES.values():
        x = l[a:b + 1]; c = (x != "D")
        n += int(np.sum(c & ~np.r_[False, c[:-1]] | (np.r_[False, x[1:] != x[:-1]] & c)))
    return n

print("| constant | -30 % agree | +30 % agree | -30 % poss | +30 % poss |")
print("|---|---|---|---|---|")
print("| (reference) | - | - | %d | %d |" % (n_poss(base), n_poss(base)))
for name in NUMERIC:
    v = getattr(Params(), name)
    row = []
    for f in (0.7, 1.3):
        l = run(trk, replace(Params(), **{name: v * f}), tags)
        row.append((np.mean(l[mask] == base[mask]), n_poss(l)))
    print("| %s | %.3f | %.3f | %d | %d |" % (name, row[0][0], row[1][0], row[0][1], row[1][1]))
