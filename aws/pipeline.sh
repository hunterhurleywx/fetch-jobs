#!/bin/bash
# One convective day, end to end, on a big AWS machine: the same steps as .github/workflows/case.yml, with the
# download passes run side by side and every decode / parcel step sized to the machine (nproc).
# Env: DATE, LEADS ("8 7 ... 1"), CAM_RUNS, EC_CIN (1/0), ADDONS (1/0), R2_* (cache), POGS_CACHE, POGS_CASES.
# Run from the repo root. Each step logs to cases/$DATE/logs/; the step list and timings go to stdout.
set -uo pipefail
N=$(nproc)
D="$DATE"
L="cases/$D/logs"
mkdir -p "$L"
MAIN_MODELS="gfs gefs hres eps hrrr rap rrfs refs eps_z"  # NAM 3 km / NAM 12 km retired 3 Nov 2026
X_MODELS="gfs_x gefs_x hres_x eps_x hrrr_ci rap_ci rrfs_ci refs_ci"
UA_MODELS="gfs_snd hres_snd gefs_pl eps_pl"
ADD_MODELS="gfs_add gefs_add hres_add hrrr_add rap_add hrrr_sfc"
EPS_ADD="eps_add"
PCL_MODELS="hres_pcl eps_pcl"
export POGS_CAM_RUNS="${CAM_RUNS:-}" POGS_EC_CIN="${EC_CIN:-1}"
ADD=""; [ "${ADDONS:-1}" != "0" ] && ADD="$ADD_MODELS $EPS_ADD"
FAILED=0

step() {  # step <name> <required 1/0> <command...>: time it, keep going on optional failures
  local name=$1 req=$2; shift 2
  local t0=$(date +%s)
  echo "$(date -u +%H:%M:%SZ) >> $name"
  "$@"; local s=$?
  echo "$(date -u +%H:%M:%SZ) << $name: exit $s after $(( $(date +%s) - t0 )) s"
  if [ $s -ne 0 ] && [ "$req" = 1 ]; then echo "!!! required step failed: $name"; exit $s; fi
  [ $s -ne 0 ] && FAILED=1
  return 0
}

dl() {  # dl <log name> <decode procs> <ncep workers> <ec workers> <models...>
  local log=$1 dp=$2 nw=$3 ew=$4; shift 4
  python -u download_case.py "$D" --leads $LEADS --models "$@" --ncep-workers $nw --ec-workers $ew --decode-procs $dp \
    > "$L/download_$log.log" 2>&1
  local s=$?; tail -n 3 "$L/download_$log.log"; return $s
}

# decode processes per pass, as shares of the machine (64 cores: main 32, x 16; then UA 24, add 12, EPS add 24)
q() { local v=$(( N * $1 / 64 )); [ $v -lt 1 ] && v=1; echo $v; }

echo "machine: $N cores, $(free -g | awk '/Mem/{print $2}') GB RAM, $(df -h /tmp | tail -1 | awk '{print $4}') free on /tmp"
step "SPC outlooks, discussions and reports" 1 python -u fetch_spc.py "$D"
step "pull decoded fields from R2" 1 python -u r2_cache.py pull "$D" --leads $LEADS --models $MAIN_MODELS $X_MODELS $UA_MODELS $ADD $PCL_MODELS

main_and_x() {
  dl main $(q 32) 128 32 $MAIN_MODELS & local p1=$!
  dl x $(q 16) 64 24 $X_MODELS & local p2=$!
  wait $p1; local s1=$?; wait $p2; local s2=$?
  return $(( s1 || s2 ))
}
step "main fields + CI extras (side by side)" 1 main_and_x
step "push main/x to R2" 1 python -u r2_cache.py push "$D" --leads $LEADS --models $MAIN_MODELS $X_MODELS

# upper air, add-ons and EPS add-ons are independent cache sets: run all three at once
extras() {
  dl ua $(q 24) 96 24 $UA_MODELS & local p1=$!
  local p2="" p3=""
  if [ "${ADDONS:-1}" != "0" ]; then
    dl add $(q 12) 96 16 $ADD_MODELS & p2=$!
    dl eps_add $(q 24) 16 32 $EPS_ADD & p3=$!
  fi
  local s=0
  wait $p1 || s=1
  [ -n "$p2" ] && { wait $p2 || s=1; }
  [ -n "$p3" ] && { wait $p3 || s=1; }
  return $s
}
step "upper air + add-ons + EPS add-ons (side by side)" 0 extras
step "push upper air / add-ons to R2" 0 python -u r2_cache.py push "$D" --leads $LEADS --models $UA_MODELS $ADD
step "ECMWF CIN / surface-based share (derive_parcel)" 0 bash -c "python -u derive_parcel.py '$D' --leads $LEADS --procs $N > '$L/derive_parcel.log' 2>&1"
step "push ECMWF parcel fields to R2" 0 python -u r2_cache.py push "$D" --leads $LEADS --models $PCL_MODELS
step "CAPE diagnostics (cape_diag)" 0 bash -c "python -u cape_diag.py '$D' --leads $LEADS > '$L/cape_diag.log' 2>&1"
if [ -n "${POGS_CAM_RUNS:-}" ]; then
  step "native 3 km storm snapshots (cam_snap)" 0 bash -c "python -u cam_snap.py '$D' > '$L/cam_snap.log' 2>&1"
  step "NAM signal vs forcing (nam_signal)" 0 bash -c "python -u nam_signal.py '$D' > '$L/nam_signal.log' 2>&1"
  step "surface fields per model (sfc_fields)" 0 bash -c "python -u sfc_fields.py '$D' > '$L/sfc_fields.log' 2>&1"
fi
step "score (run_case)" 1 bash -c "python -u run_case.py '$D' --leads $LEADS --no-plots > '$L/run_case.log' 2>&1"
step "verify" 0 bash -c "python -u verify_case.py '$D' > '$L/verify.log' 2>&1"
step "CI layer" 1 bash -c "python -u ci.py '$D' --leads $LEADS > '$L/ci.log' 2>&1 && python -u ci.py '$D' --leads $LEADS --env v3 --out ci_layers_v3.npz > '$L/ci_v3.log' 2>&1"
step "upper-air diagnostics" 0 bash -c "python -u diag_upper.py '$D' --leads $LEADS > '$L/diag_upper.log' 2>&1"
AI_RUNS=$(echo "${POGS_CAM_RUNS:-}" | tr ';' '\n' | grep -E '^(aifs|aifs_ens|aigfs):' | tr '\n' ' ')
if [ -n "$AI_RUNS" ]; then  # AI models (experimental, display only): ECMWF AIFS / AIFS ENS, scored per run + meso-low track
  pip install -q matplotlib cartopy scikit-learn shapely pandas >/dev/null 2>&1 || true
  cp calib/*.json "$POGS_CASES/" 2>/dev/null || true
  step "AI models (AIFS / AIFS ENS)" 0 bash -c "python -u experimental/ai_models.py '$D' --day $LEADS --spc-lead $LEADS --runs $AI_RUNS > '$L/ai_models.log' 2>&1"
fi
echo "pipeline finished; optional-step failures: $FAILED"
exit 0
