# Raw data (not tracked by git)

Copy the two assignment files here **without renaming them**:

| File | Station | Format | Units |
|---|---|---|---|
| `Bologna Daily Rainfall_assignment_19 october 2026.csv` | Bologna, GHCN-D ITE00100550 | CSV: `STATION, STATION_NAME, ELEVATION, LATITUDE, LONGITUDE, DATE (YYYYMMDD), PRCP` | **tenths of mm**: the reader multiplies by 0.1 |
| `polenidata_Assignment_19 october 2026.txt` | Padova (Poleni–La Specola series) | A line holding the year (e.g. `1725`), then one daily value per line, 365/366 lines per year | mm |

The file names are set in `src/rainfall_extremes/config.py` (`RAW_FILES`).

**Do not commit these files.** The Padova series was provided for this exercise and is not ours to redistribute. GHCN-Daily data are public domain, but they can be downloaded directly from NOAA NCEI.

Raw files are never modified by the code. All transformations happen in memory and are written to `data/processed/`.
