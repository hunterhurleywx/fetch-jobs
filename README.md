# fetch-jobs

Cloud jobs for a severe-weather research project: for a chosen convective day, they download the
GFS, GEFS, ECMWF HRES and ECMWF EPS model runs that cover it (SPC Day 8 through Day 1), decode only the
fields needed onto a 0.5° grid, and score the severe-weather ingredients.

- `.github/workflows/fetch.yml`: the GitHub Actions job (run per case with `workflow_dispatch`).
- `download_case.py`, `fetch.py`: byte-range downloads from NOAA (AWS) and ECMWF open data (Google Cloud).
- `fetch_spc.py`: SPC outlooks, discussions and storm reports for the same day.
- `run_case.py`, `scoring.py`, `ci.py`, `verify_case.py`, `calibrate.py`, `plots.py`: scoring,
  convective-initiation layer, verification against reports, provisional calibration and maps.
- `remote_fetch.py`: starts the cloud job from a local machine and collects the results.

Research prototype. Probabilities are provisional and not an official forecast.
Model data: NOAA NCEP (GFS, GEFS) and ECMWF open data (CC BY 4.0).
