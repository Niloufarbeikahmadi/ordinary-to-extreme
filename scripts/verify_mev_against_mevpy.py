"""
Verification: our MEVD-Weibull implementation vs the reference mevpy.

    git clone https://github.com/EnricoZorzetto/mevpy
    python scripts/verify_mev_against_mevpy.py path/to/mevpy

For every 30-year window of both stations, mevpy's ``mev_fit`` (ws=1,
PWM, threshold=1 mm, potmode=True) estimates the yearly (N, C, W). The
200-year level is then obtained from mevpy's own MEV equation ``mev_fun``.

Note: mevpy's ``mev_quant`` fails with current NumPy ("setting an array
element with a sequence"), because ``fsolve`` now returns a length-1
array. We therefore call ``scipy.optimize.fsolve`` on ``mev_fun``
directly, with the same starting point mevpy uses (the mean annual
maximum), and add the threshold back as ``mev_quant`` does.
"""
from __future__ import annotations

import importlib
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
from scipy.optimize import fsolve  # noqa: E402

from rainfall_extremes import config, datasets, mev  # noqa: E402
from rainfall_extremes.windows import build_windows  # noqa: E402

MEVPY_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else _ROOT.parent / "mevpy"


def main():
    if not (MEVPY_DIR / "mevpy" / "mev_fun.py").exists():
        raise FileNotFoundError(f"mevpy not found in {MEVPY_DIR}.")
    sys.path.insert(0, str(MEVPY_DIR))
    warnings.simplefilter("ignore")
    importlib.import_module("mevpy.mev_fun")
    mf = sys.modules["mevpy.mev_fun"]
    thr, F = config.WET_THRESHOLD_MM, 1 - 1 / config.RETURN_PERIOD
    out = []
    for st in ("bologna", "padova"):
        daily, annual = datasets.load_daily(st), datasets.load_annual(st, "asis")
        yearly = mev.yearly_parameters(daily).set_index("year")
        amax = annual.set_index("year")["annual_max_mm"]
        for w in build_windows(annual, "calendar"):
            sub = (daily[daily["year"].isin(w.years)][["year", "precip_mm"]]
                   .rename(columns={"year": "YEAR", "precip_mm": "PRCP"}))
            N, C, W = mf.mev_fit(sub, ws=1, how="pwm", threshold=thr,
                                 potmode=True)
            x0 = amax.loc[list(w.years)].mean()
            ref = fsolve(lambda y: mf.mev_fun(y, F, N, C, W), x0)[0] + thr
            p = yearly.loc[list(w.years)]
            ours = mev.mev_return_level(config.RETURN_PERIOD, p["N"], p["C"],
                                        p["w"])
            out.append({"station": st, "window_end": w.end, "ours": ours,
                        "mevpy": ref,
                        "max_dC": np.max(np.abs(p["C"].to_numpy() - C)),
                        "max_dw": np.max(np.abs(p["w"].to_numpy() - W)),
                        "max_dN": np.max(np.abs(p["N"].to_numpy() - N))})
    df = pd.DataFrame(out)
    df["diff_mm"] = df["ours"] - df["mevpy"]
    for st, g in df.groupby("station"):
        print(f"{st}: {len(g)} windows | max |200-yr diff| = "
              f"{g['diff_mm'].abs().max():.2e} mm | max |dC| = "
              f"{g['max_dC'].max():.1e} | max |dw| = {g['max_dw'].max():.1e}"
              f" | max |dN| = {g['max_dN'].max():.0f}")
    return df


if __name__ == "__main__":
    RESULT = main()
