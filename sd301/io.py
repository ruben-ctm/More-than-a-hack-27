"""Loading the tactical ATD, homography and Smart Tagging into plain numpy arrays.

Everything is indexed by *video frame* (0..199318), the same index as a submission row.
Coordinates are returned in metres on a 105.3 x 68.0 pitch, origin at the pitch corner
(ATD is normalised 0-1, we multiply by the pitch size).
"""
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import GAME, GAMES, asset_paths  # noqa: E402

FPS = 25
N_FRAMES = GAMES[GAME]["n_video_frames"]
PITCH_L, PITCH_W = GAMES[GAME]["pitch"]
# graded spans (video frames, inclusive): published in DATA.md, boundaries only
HALVES = {1: (11185, 90651), 2: (110808, 190197)}


def slot_teams(meta_path):
    """0 = home (Barcelona), 1 = away, per track slot."""
    root = ET.parse(meta_path).getroot()
    team_ids = [t.get("id") for t in root.iter("Team")]
    return np.array([0 if p.get("teamId") == team_ids[0] else 1
                     for p in root.iter("Player")], dtype=np.int8)


def _parse_group(g):
    return g.split(",")


def parse_atd(raw_path, n_slots, n_frames=N_FRAMES):
    """Return px, py, ps (n_frames, n_slots) and bx, by, bs (n_frames,) in normalised units."""
    px = np.full((n_frames, n_slots), np.nan, np.float32)
    py = np.full((n_frames, n_slots), np.nan, np.float32)
    ps = np.full((n_frames, n_slots), np.nan, np.float32)
    bx = np.full(n_frames, np.nan, np.float32)
    by = np.full(n_frames, np.nan, np.float32)
    bs = np.full(n_frames, np.nan, np.float32)
    with open(raw_path) as fh:
        for line in fh:
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                continue
            f = int(parts[0]) - 1
            groups = parts[1].split(";")
            flat = ",".join(groups).split(",")
            arr = np.array(flat[:3 * n_slots], dtype=np.float32).reshape(-1, 3)
            k = arr.shape[0]
            px[f, :k], py[f, :k], ps[f, :k] = arr[:, 0], arr[:, 1], arr[:, 2]
            b = np.array(parts[2].split(","), dtype=np.float32)
            bx[f], by[f], bs[f] = b[0], b[1], b[2] if len(b) > 2 else np.nan
    return px, py, ps, bx, by, bs


def load_tracking(cache_dir=None, force=False):
    """Parsed tactical ATD, cached as .npz (cache lives outside git: data/ is ignored)."""
    paths = asset_paths()
    cache_dir = cache_dir or os.path.join(os.path.dirname(paths["atd_raw"]), "cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, "tactical_atd.npz")
    team = slot_teams(paths["atd_meta"])
    if os.path.exists(cache) and not force:
        z = np.load(cache)
        out = {k: z[k] for k in z.files}
    else:
        px, py, ps, bx, by, bs = parse_atd(paths["atd_raw"], len(team))
        out = dict(px=px, py=py, ps=ps, bx=bx, by=by, bs=bs)
        np.savez(cache, **out)
    out["team"] = team
    out["px"] = out["px"] * PITCH_L
    out["py"] = out["py"] * PITCH_W
    out["bx"] = out["bx"] * PITCH_L
    out["by"] = out["by"] * PITCH_W
    return out


def load_smart_tagging():
    """List of dicts: start/end frame, code, labels."""
    path = asset_paths()["smart_tagging"]
    root = ET.parse(path).getroot()
    out = []
    for inst in root.iter("instance"):
        d = {"id": int(inst.findtext("ID")),
             "start": int(round(float(inst.findtext("start")) * FPS)),
             "end": int(round(float(inst.findtext("end")) * FPS)),
             "code": inst.findtext("code")}
        for lab in inst.findall("label"):
            d[lab.findtext("group")] = lab.findtext("text")
        out.append(d)
    return out


def load_homography():
    import pandas as pd
    df = pd.read_csv(asset_paths()["homography"])
    return df
