"""Regenerate the SD-301 submission from the tactical ATD.

    python run_sd301.py                       # writes outputs/SD-301.csv
    python run_sd301.py --out some/path.csv
    python evaluation/validate_submission.py outputs/SD-301.csv

Needs data/tactical/BAR-ATH_tactical_atd.{xml,txt} (see DATA.md). Only numpy is required.
"""
import argparse
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sd301.io import load_tracking  # noqa: E402
from sd301.pipeline import Params, run, write_csv, params_dict  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=os.path.join("outputs", "SD-301.csv"))
    args = ap.parse_args()
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    params = Params()
    labels = run(load_tracking(), params)
    write_csv(args.out, labels)
    print("wrote", args.out)
    print("params:", params_dict(params))
