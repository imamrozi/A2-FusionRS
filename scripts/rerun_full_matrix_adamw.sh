#!/bin/bash
# scripts/rerun_full_matrix_adamw.sh
#
# Re-run SELURUH matriks eksperimen A2-IRM (main branch) setelah perbaikan
# bug kolaps DeepMF (SGD polos -> AdamW lr=0,002, reports/methodology_
# audit_2026-07-26.md Temuan A3/23). SEMUA hasil lama (checkpoints/results/)
# sudah diarsipkan ke checkpoints/results_pre_adamw_fix_2026-07-26/ oleh
# user -- checkpoints/results/ sekarang kosong, run ini mengisi ulang dari
# nol dgn DeepMF yang sudah benar.
#
# Tujuan (ditegaskan user): fokus paper pada perbandingan hybrid model +
# sentimen GLOBAL vs hybrid model + ABSA (4 skenario) -- tingkat error
# (RMSE/MAE) dan uji signifikansi Wilcoxon, BUKAN dekomposisi leakage P2/P3.
# Baris `no_sentiment_ablation` disertakan sbg REFERENSI JUJUR (floor
# DeepMF+CBF tanpa sentimen sama sekali) -- murah (15 run, bukan 45),
# supaya angka utama tetap punya konteks tanpa perlu mesin P2/P3 penuh.
#
# 3 domain x 5 seed x (5 model target_review + 1 floor no_sentiment) = 90 run.
# Protokol default run_baseline.py/run_baseline_absa.py = target_review (P1),
# TIDAK perlu flag apa pun -- itu satu2nya protokol yg dipakai matriks utama.
#
# PRASYARAT: checkpoint model SA (sentiment_bert/) per domain HARUS ada dari
# run_baseline.py sebelumnya (dilatih ulang otomatis di run pertama tiap
# domain kalau belum ada -- lihat urutan di bawah, SA-global dijalankan
# LEBIH DULU per domain sebelum ABSA, supaya checkpoint-nya tersedia).
#
# Satu run gagal TIDAK menghentikan seluruh matriks -- dicatat ke
# FAILED_RUNS.txt, sisanya tetap lanjut.
#
# RESUMABLE: sebelum menjalankan tiap run, dicek dulu apakah file hasil
# YAML-nya sudah ada -- kalau ada, DILEWATI (bukan diulang).

set -uo pipefail
cd "$(dirname "$0")/.."

PY="python"  # Colab: python3 di PATH sistem, bukan venv/Scripts/python.exe (Windows lokal)
RESULTS_DIR="checkpoints/results"
SEEDS=(42 123 456 789 1011)
LOG_DIR="checkpoints/results/logs/full_matrix_adamw"
mkdir -p "$LOG_DIR"
FAILED_LOG="$LOG_DIR/FAILED_RUNS.txt"
touch "$FAILED_LOG"  # TIDAK ditruncate -- resumable, riwayat kegagalan sesi sebelumnya dipertahankan

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
# domain_label:config utk vehicle no_sentiment_ablation (concat_confidence
# dipakai sembarang -- mode ABSA tidak relevan lg krn sentimen di-nolkan)
FLOOR_CONFIGS=(
  "amazon_electronics:configs/amazon_electronics_config_absa_concat_confidence.yaml"
  "restaurant:configs/yelp_config_absa_concat_confidence.yaml"
  "tripadvisor_hotel:configs/tripadvisor_hotel_config_absa_concat_confidence.yaml"
)

TOTAL=90
DONE=0
T0=$(date +%s)

# run_one <label> <expected_output_yaml> <python_args...>
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

# ---- 1. Darraz reimpl (SA global) x 3 domain x 5 seed = 15 run ----
# WAJIB duluan per domain -- ABSA (tahap 2) butuh checkpoint SA dari sini.
for entry in "${DARRAZ_CONFIGS[@]}"; do
  domain="${entry%%:*}"; cfg="${entry#*:}"
  for seed in "${SEEDS[@]}"; do
    out="baseline_reimpl_cbf_nosentiment_${domain}_seed${seed}.yaml"
    run_one "darraz_reimpl_${domain}_seed${seed}" "$out" \
      run_baseline.py --config "$cfg" --seed "$seed"
  done
done

# ---- 2. 4 mode ABSA x 3 domain x 5 seed = 60 run ----
declare -A MODE_PREFIX=(
  [mean]="absa_ablation"
  [concat]="absa_ablation_concat"
  [concat_confidence]="absa_ablation_concat_confidence"
  [confidence_mean]="absa_ablation_confidence_mean"
)

for entry in "${ABSA_DOMAINS[@]}"; do
  domain="${entry%%:*}"; cfg_prefix="${entry#*:}"
  for mode_entry in "${ABSA_MODES[@]}"; do
    suffix="${mode_entry%%:*}"; mode_name="${mode_entry#*:}"
    cfg="${cfg_prefix}${suffix}.yaml"
    for seed in "${SEEDS[@]}"; do
      out="${MODE_PREFIX[$mode_name]}_cbf_nosentiment_${domain}_seed${seed}.yaml"
      run_one "absa_${mode_name}_${domain}_seed${seed}" "$out" \
        run_baseline_absa.py --config "$cfg" --seed "$seed"
    done
  done
done

# ---- 3. Floor no_sentiment_ablation x 3 domain x 5 seed = 15 run ----
# Vehicle: config concat_confidence (arbitrer, ABSA mode tdk relevan krn
# sentimen di-nolkan sepenuhnya oleh protokol ini).
for entry in "${FLOOR_CONFIGS[@]}"; do
  domain="${entry%%:*}"; cfg="${entry#*:}"
  for seed in "${SEEDS[@]}"; do
    out="absa_ablation_concat_confidence_cbf_nosentiment_no_sentiment_${domain}_seed${seed}.yaml"
    run_one "floor_no_sentiment_${domain}_seed${seed}" "$out" \
      run_baseline_absa.py --config "$cfg" --seed "$seed" --sentiment-protocol no_sentiment_ablation
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
