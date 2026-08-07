#!/bin/bash
# scripts/run_agf_clean_selection.sh
#
# Tahap 6 (plan pure-painting-wilkes.md): SELEKSI JANGKAR arsitektur
# A2-FusionRS "clean" -- dijalankan di DEV split, TEST TIDAK PERNAH
# DISENTUH.
#
# Arsitektur yang diseleksi (konstan di semua sel):
#   --scenario a2fusionrs_clean   -> DeepMF + CBF + PyABSA, TANPA keyword-
#                                    ABSA, TANPA NMF+DecisionTree
#   --extra-pyabsa perseq_rich    -> sequence per-aspek + ringkasan level-
#                                    review, dari SATU pass skoring PyABSA
#   --input-standardize           -> ditetapkan a priori (disparitas skala
#                                    antar-modalitas), BUKAN axis seleksi
#
# GRID SELEKSI (4 varian): residual_base {none, user_item_bias}
#                        x representation {vector, asymmetric}
# 3 domain x 3 seed = 9 kombinasi stream, 36 run AGF total.
#
# ATURAN KEPUTUSAN -- DIPRA-REGISTRASI SEBELUM RUN (jangan diubah setelah
# melihat hasil; ini yang membuat seleksi bisa dipertanggungjawabkan):
#   1. Per domain: rata-rata dev-RMSE lintas 3 seed untuk tiap varian.
#   2. Ranking 4 varian per domain (1=terbaik).
#   3. Varian dgn RATA-RATA RANKING terendah lintas 3 domain = PEMENANG.
#   4. Tie-break: mean dev-RMSE ternormalisasi per domain.
#   5. SATU konfigurasi dipakai utk ketiga domain -- TIDAK boleh memilih
#      per-domain (itu overfitting ke dev).
# Analisis dijalankan oleh scripts/analyze_agf_selection.py (mengimplemen-
# tasikan aturan di atas apa adanya).
#
# CATATAN BIAYA: --stream-cache membuat DeepMF OOF + CBF LOO dihitung SEKALI
# per (domain, seed), lalu dipakai ulang oleh 4 varian. Varian pertama tiap
# kombinasi ~8 menit, tiga sisanya ~1 menit. Estimasi total ~1,5-2 jam.
#
# RESUMABLE: run yang hasilnya sudah ada di-skip.

set -uo pipefail
cd "$(dirname "$0")/.."

PY="python"  # Colab: python3 di PATH sistem
RESULTS_DIR="checkpoints/results_phase2_clean/dev"
SEEDS=(42 123 456)
LOG_DIR="checkpoints/results_phase2_clean/logs/selection"
mkdir -p "$LOG_DIR"
FAILED_LOG="$LOG_DIR/FAILED_RUNS.txt"
touch "$FAILED_LOG"

DOMAIN_CONFIGS=(
  "amazon_electronics:configs/amazon_electronics_config_agf_colab.yaml"
  "restaurant:configs/yelp_config_agf_colab.yaml"
  "tripadvisor_hotel:configs/tripadvisor_hotel_config_agf_colab.yaml"
)
RESIDUAL_BASES=(none user_item_bias)
REPRESENTATIONS=(vector asymmetric)

TOTAL=36
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

# Urutan loop SENGAJA: domain -> seed -> varian. Dgn begitu 4 varian pada
# (domain, seed) yang sama dijalankan BERURUTAN sehingga cache stream baru
# dibangun sekali lalu langsung dipakai 3x, bukan dibangun ulang.
for entry in "${DOMAIN_CONFIGS[@]}"; do
  domain="${entry%%:*}"; cfg="${entry#*:}"
  for seed in "${SEEDS[@]}"; do
    for res in "${RESIDUAL_BASES[@]}"; do
      for rep in "${REPRESENTATIONS[@]}"; do
        tag="${res}_${rep}"
        out="agf_a2fusionrs_clean_${tag}_${domain}_seed${seed}.yaml"
        run_one "sel_${domain}_seed${seed}_${tag}" "$out" \
          run_attention_gated_fusion.py \
          --config "$cfg" \
          --scenario a2fusionrs_clean \
          --extra-pyabsa perseq_rich \
          --input-standardize \
          --residual-base "$res" \
          --representation "$rep" \
          --stage select \
          --stream-cache \
          --run-tag "$tag" \
          --seed "$seed"
      done
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
echo ""
echo "LANGKAH BERIKUTNYA: python scripts/analyze_agf_selection.py"
echo "(menerapkan aturan keputusan yang sudah dipra-registrasi di header script ini)"
