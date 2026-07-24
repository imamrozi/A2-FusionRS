#!/bin/bash
# scripts/rerun_cbf_nosentiment_full.sh
#
# Re-run SELURUH matriks eksperimen A2-IRM (main branch, pra-Fase 2) dengan
# CBF yang TIDAK memasukkan sentiment (CBFConfig.include_sentiment=False),
# baik untuk sentimen global (Darraz reimpl) maupun keempat mode ABSA.
#
# 3 domain x 5 seed x 5 varian model = 75 run. Semua cache ABSA/SA sudah ada
# (diverifikasi sebelum run) -- TIDAK ada inferensi BERT baru, cuma DeepMF+
# CBF+fusion di-refit per run (~5 menit/run berdasarkan smoke test).
#
# Hasil TERPISAH dari varian asli (prefix *_cbf_nosentiment, lihat
# run_baseline.py/run_baseline_absa.py --no-cbf-sentiment) -- tidak menimpa
# ledger yang sudah ada.
#
# Satu run gagal TIDAK menghentikan seluruh matriks -- dicatat ke
# FAILED_RUNS.txt, sisanya tetap lanjut (job 6+ jam, kegagalan 1 run tidak
# boleh membuang progress run lainnya).

set -uo pipefail
cd "$(dirname "$0")/.."

PY="./venv/Scripts/python.exe"
SEEDS=(42 123 456 789 1011)
LOG_DIR="checkpoints/results/logs/cbf_nosentiment_full"
mkdir -p "$LOG_DIR"
FAILED_LOG="$LOG_DIR/FAILED_RUNS.txt"
: > "$FAILED_LOG"

# domain_label:darraz_config
DARRAZ_CONFIGS=(
  "amazon_electronics:configs/amazon_electronics_config.yaml"
  "restaurant:configs/yelp_config.yaml"
  "tripadvisor_hotel:configs/tripadvisor_hotel_config.yaml"
)

# domain_label:absa_config_prefix (4 mode file per domain, nama tetap)
ABSA_DOMAINS=(
  "amazon_electronics:configs/amazon_electronics_config_absa"
  "restaurant:configs/yelp_config_absa"
  "tripadvisor_hotel:configs/tripadvisor_hotel_config_absa"
)
# suffix file:nama mode (utk log saja)
ABSA_MODES=(
  ":mean"
  "_concat:concat"
  "_concat_confidence:concat_confidence"
  "_confidence:confidence_mean"
)

TOTAL=75
DONE=0
T0=$(date +%s)

run_one() {
  local label="$1"; shift
  local logfile="$LOG_DIR/${label}.log"
  echo "[$(date '+%H:%M:%S')] MULAI  ($((DONE+1))/$TOTAL): $label"
  if "$PY" "$@" > "$logfile" 2>&1; then
    echo "[$(date '+%H:%M:%S')] SELESAI ($((DONE+1))/$TOTAL): $label"
  else
    echo "[$(date '+%H:%M:%S')] GAGAL   ($((DONE+1))/$TOTAL): $label -- lihat $logfile" | tee -a "$FAILED_LOG"
  fi
  DONE=$((DONE+1))
}

# ---- 1. Darraz reimpl (SA global) x 3 domain x 5 seed = 15 run ----
for entry in "${DARRAZ_CONFIGS[@]}"; do
  domain="${entry%%:*}"; cfg="${entry#*:}"
  for seed in "${SEEDS[@]}"; do
    run_one "darraz_reimpl_${domain}_seed${seed}" \
      run_baseline.py --config "$cfg" --seed "$seed" --no-cbf-sentiment
  done
done

# ---- 2. 4 mode ABSA x 3 domain x 5 seed = 60 run ----
for entry in "${ABSA_DOMAINS[@]}"; do
  domain="${entry%%:*}"; cfg_prefix="${entry#*:}"
  for mode_entry in "${ABSA_MODES[@]}"; do
    suffix="${mode_entry%%:*}"; mode_name="${mode_entry#*:}"
    cfg="${cfg_prefix}${suffix}.yaml"
    for seed in "${SEEDS[@]}"; do
      run_one "absa_${mode_name}_${domain}_seed${seed}" \
        run_baseline_absa.py --config "$cfg" --seed "$seed" --no-cbf-sentiment
    done
  done
done

ELAPSED=$(( $(date +%s) - T0 ))
echo ""
echo "=== SELESAI SEMUA: $DONE/$TOTAL run, $((ELAPSED/60)) menit ==="
N_FAILED=$(wc -l < "$FAILED_LOG")
if [ "$N_FAILED" -gt 0 ]; then
  echo "PERINGATAN: $N_FAILED run GAGAL -- lihat $FAILED_LOG"
else
  echo "Semua run SUKSES, tidak ada kegagalan."
fi
