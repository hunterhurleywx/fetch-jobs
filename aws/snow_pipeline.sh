#!/bin/bash
# Snowfall-total test run on the AWS machine (experimental). Started by aws/pipeline.sh when CAM_RUNS=snow.
# LEADS carries the window: "<start YYYYMMDDHH> <end> <last NCEP init> <last ECMWF init>".
set -uo pipefail
D="$DATE"
O="cases/$D/output"
L="cases/$D/logs"
mkdir -p "$O" "$L"
read -r S E LN LE <<< "$LEADS"
echo "machine: $(nproc) cores, $(free -g | awk '/Mem/{print $2}') GB RAM; snow window $S-$E, last NCEP $LN, last EC $LE"
step() {
  local name=$1; shift
  local t0=$(date +%s)
  echo "$(date -u +%H:%M:%SZ) >> $name"
  "$@"; local s=$?
  echo "$(date -u +%H:%M:%SZ) << $name: exit $s after $(( $(date +%s) - t0 )) s"
  return $s
}
step "snow fields, SLR methods, NBM, NOHRSC" python -u snow/snow_case.py fetch --out "$O" --start "$S" --end "$E" \
  --last-ncep "$LN" --last-ec "$LE" > "$L/snow_fetch.log" 2>&1
s=$?; tail -n 25 "$L/snow_fetch.log"
[ $s -ne 0 ] && exit $s

rf() {  # Pletcher 2026 RF / LR: public repo, own Python 3.11 venv (pickles need scikit-learn 1.2.2)
  dnf install -y -q python3.11 python3.11-pip git-lfs >/dev/null 2>&1 || dnf install -y -q python3.11 python3.11-pip
  git clone -q --depth 1 https://github.com/mdpletcher/utahrfslr.git /opt/utahrfslr || return 1
  (cd /opt/utahrfslr && (git lfs pull 2>/dev/null || true); ls -la models/rf models/lr)
  python3.11 -m venv /opt/venv311 && /opt/venv311/bin/pip install -q "numpy<2" "pandas<2.2" "scikit-learn==1.2.2" joblib || return 1
  /opt/venv311/bin/python -u snow/slr_rf.py --repo /opt/utahrfslr --out "$O"
}
step "Pletcher SLR (RF + LR)" rf > "$L/snow_rf.log" 2>&1; tail -n 15 "$L/snow_rf.log"
step "apply Pletcher SLR" python -u snow/snow_case.py finish --out "$O" > "$L/snow_finish.log" 2>&1; tail -n 5 "$L/snow_finish.log"
rm -f "$O/slr_features.npz"   # large; not needed locally
ls -la "$O"
exit 0
