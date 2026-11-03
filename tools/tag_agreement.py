"""Ground-truth-free full-match check against Metrica's automatic Smart Tagging phases.

For phases whose Team tag implies who holds the ball (attacking phases: the tagged side; defending
phases and defensive transitions: the *other* side), measure how often a submission's alive labels
agree.  Smart Tagging is an automatic, noisy input -- so this is a consistency indicator, not a
score, and it is never used to fit anything.

    python tools/tag_agreement.py outputs/SD-301.csv evaluation/results/baseline_smooth50.csv
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sd301.io import load_smart_tagging, HALVES

HOLDER = {"BUILD UP", "PROGRESSION", "BALL IN FINAL THIRD", "BALL IN THE BOX", "PLAYERS IN THE BOX",
          "CREATING CHANCES", "LONG BALL", "ATTACKING TRANSITION", "GOALS"}
OTHER = {"DEFENDING IN MIDDLE THIRD", "DEFENDING IN ATTACKING THIRD", "DEFENDING IN DEFENSIVE THIRD",
         "DEFENSIVE TRANSITION"}


def holder_label(t):
    team = t.get("Team")
    if team in (None, "N/A"):
        return None
    home = team.lower().startswith("barcelona")
    if t["code"] in HOLDER:
        return "H" if home else "A"
    if t["code"] in OTHER:
        return "A" if home else "H"
    return None


def agreement(lab, tags):
    per, tot = {}, [0, 0]
    for t in tags:
        h = holder_label(t)
        if h is None:
            continue
        for a, b in HALVES.values():
            lo, hi = max(t["start"], a), min(t["end"], b)
            if hi <= lo:
                continue
            seg = lab[lo:hi + 1]
            alive = seg != "D"
            if alive.sum() == 0:
                continue
            ok = int((seg[alive] == h).sum()); n = int(alive.sum())
            c = per.setdefault(t["code"], [0, 0]); c[0] += ok; c[1] += n
            tot[0] += ok; tot[1] += n
    return tot, per

if __name__ == "__main__":
    tags = load_smart_tagging()
    for path in sys.argv[1:]:
        lab = pd.read_csv(path).label.values
        tot, per = agreement(lab, tags)
        print("%-34s overall %.3f  (%d alive frames)" % (os.path.basename(path), tot[0] / tot[1], tot[1]))
        if "-v" in os.environ.get("TAGV", ""):
            for k, (o, n) in sorted(per.items()):
                print("    %-32s %.3f" % (k, o / n))
