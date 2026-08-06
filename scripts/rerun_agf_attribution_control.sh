#!/bin/bash
# scripts/rerun_agf_attribution_control.sh
#
# Stage G lanjutan (plan pure-painting-wilkes.md): 15 run tambahan utk
# skenario `static_keyword_pyabsa` -- kontrol atribusi yang BENAR (info
# sama dgn agf_keyword_oof_perseq: keyword ABSA concat+confidence HSTACK
# PyABSA-rich, tapi fusi TREE statis, bukan AGF) -- yang TIDAK dimasukkan
# ke scripts/rerun_agf_triage.sh (45-run) semula.
#
# Kenapa perlu: perbandingan atribusi awal (agf_keyword_oof_perseq vs
# static_pyabsa) TIDAK adil -- static_pyabsa cuma dapat PyABSA 5-dim
# summary (TANPA keyword ABSA), sedangkan agf_keyword_oof_perseq dapat
# keyword ABSA + PyABSA aspect-sequence + base OOF sekaligus. Selisih
# -14% s/d -28% yang terlihat sebagian besar krn info lebih banyak, BUKAN
# murni krn mekanisme AGF/attention. static_keyword_pyabsa menyamakan
# info (keyword ABSA + PyABSA-rich, SAMA persis dgn agf_keyword_oof_
# perseq's dua sumber sentimen) sehingga selisih AGF vs static_keyword_
# pyabsa baru benar2 mengisolasi nilai tambah mekanisme fusi (attention+
# gating) di atas info yang identik -- pertanyaan yang sebenarnya ingin
# dijawab verdict lama (memori phase2-agf-final-verdict).
#
# 3 domain x 5 seed = 15 run. Resumable, pola sama scripts/rerun_agf_
# triage.sh -- aman dijalankan terpisah tanpa mengganggu 45 run yg sudah
# selesai (skenario/nama file beda, tidak overlap).

set -uo pipefail
cd "$(dirname "$0")/.."

PY="python"  # Colab: python3 di PATH sistem, bukan venv/Scripts/python.exe (Windows lokal)
RESULTS_DIR="checkpoints/results"
SEEDS=(42 123 456 789 1011)
LOG_DIR="checkpoints/results/logs/agf_triage"
mkdir -p "$LOG_DIR"
FAILED_LOG="$LOG_DIR/FAILED_RUNS.txt"
touch "$FAILED_LOG"

DOMAIN_CONFIGS=(
  "amazon_electronics:configs/amazon_electronics_config_agf_colab.yaml"
  "restaurant:configs/yelp_config_agf_colab.yaml"
  "tripadvisor_hotel:configs/tripadvisor_hotel_config_agf_colab.yaml"
)

TOTAL=15
DONE=0
T0=$(date +%s)

run_one() {
  local label="$1"; local expected_out="$2"; shift 2
  local logfile="$LOG_DIR/${label}.log"

  if [ -f "$RESULTS_DIR/$expected_out" ]; then
    echo "[$(date '+%H:%M:%S')] LEWATI  ($((DONE+1))/$TOTAL): $label (hasil sudah ada -- resume)"
    DONE=$((DONE+1))
    return
  fi

  echo "[$(date '+%H:%M:%S')] MULAI   ($((DONE+1))/$TOTAL): $label"
  if "$PY" "$@" > "$logfile" 2>&1; then
    echo "[$(date '+%H:%M:%S')] SELESAI ($((DONE+1))/$TOTAL): $label"
  else
    echo "[$(date '+%H:%M:%S')] GAGAL   ($((DONE+1))/$TOTAL): $label -- lihat $logfile" | tee -a "$FAILED_LOG"
  fi
  DONE=$((DONE+1))
}

for entry in "${DOMAIN_CONFIGS[@]}"; do
  domain="${entry%%:*}"; cfg="${entry#*:}"
  for seed in "${SEEDS[@]}"; do
    out="agf_static_keyword_pyabsa_${domain}_seed${seed}.yaml"
    run_one "static_keyword_pyabsa_${domain}_seed${seed}" "$out" \
      run_attention_gated_fusion.py --config "$cfg" --scenario static_keyword_pyabsa --seed "$seed"
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
