"""Ground-truth-free sanity statistics of a submission (possession counts, dead share, ...)."""
import sys, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sd301.io import HALVES

def segs(l, a, b):
    out, s = [], a
    for i in range(a + 1, b + 2):
        if i > b or l[i] != l[s]:
            out.append((l[s], s, i - 1)); s = i
    return out

lab = pd.read_csv(sys.argv[1]).label.values
for h, (a, b) in HALVES.items():
    sg = segs(lab, a, b)
    p = [x for x in sg if x[0] in "HA"]; d = [x for x in sg if x[0] == "D"]
    print("half %d | possessions %d (mean %.1fs, median %.1fs) | dead stretches %d (mean %.1fs, median %.1fs) | D share %.2f | H share of alive %.2f"
          % (h, len(p), np.mean([(e-s+1)/25 for _, s, e in p]), np.median([(e-s+1)/25 for _, s, e in p]),
             len(d), np.mean([(e-s+1)/25 for _, s, e in d]), np.median([(e-s+1)/25 for _, s, e in d]),
             np.mean(lab[a:b+1] == "D"), np.mean(lab[a:b+1][lab[a:b+1] != "D"] == "H")))
