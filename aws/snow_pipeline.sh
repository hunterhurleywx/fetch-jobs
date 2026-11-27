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
step "snow fields, SLR methods, NBM, NOHRSC" timeout 2400 python -u snow/snow_case.py fetch --out "$O" --start "$S" --end "$E" \
  --last-ncep "$LN" --last-ec "$LE" > "$L/snow_fetch.log" 2>&1
s=$?; tail -n 25 "$L/snow_fetch.log"
[ $s -ne 0 ] && exit $s

rf() {  # Pletcher 2026 RF / LR: public repo, own venv. The RF pickle needs a newer scikit-learn than the repo's
  # environment file says (1.2.2 refused its tree nodes; its test pickle needs numpy 2): try versions until it loads.
  git clone -q --depth 1 https://github.com/mdpletcher/utahrfslr.git /opt/utahrfslr || return 1
  ls -la /opt/utahrfslr/models/rf /opt/utahrfslr/models/lr
  python3.12 -m venv /opt/venvrf || return 1
  for v in 1.5.2 1.4.2 1.6.1 1.3.2; do
    /opt/venvrf/bin/pip install -q "scikit-learn==$v" pandas joblib || continue
    if /opt/venvrf/bin/python -c "import pickle; pickle.load(open('/opt/utahrfslr/models/rf/rf_slr_model.pkl','rb'))" 2>/dev/null; then
      echo "RF loads with scikit-learn $v"; break
    fi
    echo "RF does not load with scikit-learn $v"
  done
  /opt/venvrf/bin/python -u snow/slr_rf.py --repo /opt/utahrfslr --out "$O"
}
step "Pletcher SLR (RF + LR)" timeout 1800 bash -c "$(declare -f rf); O=$O; rf" > "$L/snow_rf.log" 2>&1; tail -n 15 "$L/snow_rf.log"
step "apply Pletcher SLR" timeout 900 python -u snow/snow_case.py finish --out "$O" > "$L/snow_finish.log" 2>&1; tail -n 5 "$L/snow_finish.log"
rm -f "$O/slr_features.npz"   # large; not needed locally
ls -la "$O"
exit 0
