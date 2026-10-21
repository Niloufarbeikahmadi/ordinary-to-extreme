"""
Verification: our GEV estimators vs the reference implementation mevpy.

mevpy (E. Zorzetto): https://github.com/EnricoZorzetto/mevpy
Clone it anywhere and set MEVPY_DIR below (or pass the path as argument):

    git clone https://github.com/EnricoZorzetto/mevpy
    python scripts/verify_gev_against_mevpy.py path/to/mevpy

The check fits every 30-year Padova window with both implementations and
reports the largest differences in parameters and 200-year return levels.
"""
from __future__ import annotations

import importlib.util
import sys
import warnings
from pathlib import Path

try:
    _ROOT = Path(__file__).resolve().parents[1]
except NameError:
    _ROOT = Path.cwd()
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from rainfall_extremes import datasets, gev  # noqa: E402
from rainfall_extremes.moving_gev import window_maxima  # noqa: E402
from rainfall_extremes.windows import build_windows  # noqa: E402

MEVPY_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else _ROOT.parent / "mevpy"


def load_mevpy_gev(mevpy_dir: Path):
    path = mevpy_dir / "mevpy" / "gev_fun.py"
    if not path.exists():
        raise FileNotFoundError(f"mevpy not found at {path}. Clone it and "
                                f"pass its folder as argument.")
    spec = importlib.util.spec_from_file_location("mevpy_gev_fun", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    mg = load_mevpy_gev(MEVPY_DIR)
    annual = datasets.load_annual("padova", "asis")
    rows = []
    for w in build_windows(annual, "calendar"):
        x = window_maxima(annual, w)
        ours_lm, ours_ml = gev.fit_lmom(x), gev.fit_ml(x)
        ref_lm = mg.gev_fit_lmom(x)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ref_ml = mg.gev_fit_ml(x)
        rows.append({
            "window_end": w.end,
            "lmom_max_abs_param_diff": np.max(np.abs(
                np.array(ours_lm.as_tuple()) - np.array(ref_lm))),
            "ml_xi_diff": ours_ml.xi - ref_ml[0],
            "ml_rl200_diff_mm": float(gev.return_level(200, ours_ml)
                                      - gev.return_level(200, tuple(ref_ml))),
            "ml_nll_ours_minus_ref": ours_ml.nll - gev.negloglik(tuple(ref_ml), x),
        })
    df = pd.DataFrame(rows)
    print(f"{len(df)} Padova windows compared with mevpy")
    print(f"L-moments : max |parameter difference| = "
          f"{df['lmom_max_abs_param_diff'].max():.2e}")
    print(f"ML        : max |xi difference|        = "
          f"{df['ml_xi_diff'].abs().max():.2e}")
    print(f"ML        : max |200-yr level diff|    = "
          f"{df['ml_rl200_diff_mm'].abs().max():.3f} mm")
    print(f"ML        : our likelihood is never worse: "
          f"{bool((df['ml_nll_ours_minus_ref'] <= 1e-6).all())}")
    return df


if __name__ == "__main__":
    RESULT = main()
