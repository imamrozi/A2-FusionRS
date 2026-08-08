#!/bin/bash
# scripts/run_agf_v2_selection.sh
#
# SELEKSI KAPASITAS AGF di SELECTION_DEV -- arsitektur v2 (pasca-diagnosis).
#
# ===================================================================
#  Script ini TIDAK menyentuh test set (--stage select). Hasil ditulis
#  ke checkpoints/results_phase2_clean_v2/dev/.
# ===================================================================
#
# LATAR (reports/gates_1_3_summary.md):
#   Gerbang-1: dgn scorer disetarakan, ekstraksi PyABSA >= leksikon keyword
#              di 3/3 domain -> pakai PyABSA utk EKSTRAKSI, SA-BERT
#              per-domain utk SKORING (scorer yang sama dgn A2-IRM).
#   Gerbang-3: selisih yang dikira "nilai struktur" sebenarnya adalah akses
#              ke skor review global -> tambahkan token sentimen global.
#
# ARSITEKTUR YANG DISELEKSI (konstan di semua sel):
#   --scenario a2fusionrs_clean
#   --extra-pyabsa sabert_perseq_rich   (ekstraksi PyABSA + skoring SA-BERT)
#   --global-sentiment-token
#   --residual-base user_item_bias --representation asymmetric
#   --input-standardize
# Jangkar & representasi TIDAK diseleksi ulang: keduanya sudah dikunci di
# seleksi sebelumnya (reports/agf_clean_selection/README.md, commit c71d7b8)
# dan merupakan pilihan yang tidak spesifik-arsitektur. Menyeleksi ulang
# semuanya akan meledakkan grid tanpa menjawab pertanyaan baru.
#
# AXIS YANG DISELEKSI -- kapasitas AGF, BELUM PERNAH di-tune sama sekali:
#   d            {64, 128}          dimensi embedding bersama
#   weight_decay {0, 1e-4, 1e-3}    L2 Adam
# = 6 varian x 3 domain x 3 seed = 54 run.
#
# Kenapa weight_decay ikut: ia langsung relevan untuk target ROBUSTNESS --
# menekan koreksi AGF ke nol sehingga model belajar koreksi yang robust
# alih-alih menghafal noise train. Sel B faktorial lama menunjukkan AGF
# memang sudah lebih stabil dari tree (SD e-commerce 0,0481 vs 0,1023);
# axis ini menguji apakah stabilitas itu bisa ditingkatkan lagi.
#
# ================== ATURAN KEPUTUSAN (PRA-REGISTRASI) ==================
# Ditetapkan SEBELUM script dijalankan; jangan diubah setelah melihat hasil.
#
#  1. Per domain: rata-rata dev-RMSE lintas 3 seed -> ranking 6 varian.
#  2. Pemenang = rata-rata rank TERENDAH lintas 3 domain.
#  3. Tie-break: mean RMSE ternormalisasi (dibagi RMSE terbaik per domain).
#  4. SATU konfigurasi untuk ketiga domain -- TIDAK boleh pilih per-domain
#     (itu overfitting ke dev).
#  5. Tabel seleksi WAJIB di-commit SEBELUM tahap konfirmasi dijalankan.
#
# Analisis: python scripts/analyze_agf_v2_selection.py
# =======================================================================
#
# PRASYARAT: cache skor SA-BERT atas aspek PyABSA harus sudah ada.
#   python scripts/precompute_pyabsa_sabert_scores.py     (butuh GPU)

set -uo pipefail
cd "$(dirname "$0")/.."

PY="python"
RESULTS_SUBDIR="results_phase2_clean_v2"
RESULTS_DIR="checkpoints/${RESULTS_SUBDIR}/dev"
SEEDS=(42 123 456)
LOG_DIR="checkpoints/${RESULTS_SUBDIR}/logs/selection"
mkdir -p "$LOG_DIR"
FAILED_LOG="$LOG_DIR/FAILED_RUNS.txt"
touch "$FAILED_LOG"

DOMAIN_CONFIGS=(
  "amazon_electronics:configs/amazon_electronics_config_agf_colab.yaml"
  "restaurant:configs/yelp_config_agf_colab.yaml"
  "tripadvisor_hotel:configs/tripadvisor_hotel_config_agf_colab.yaml"
)

# --- Prasyarat: gagal CEPAT kalau cache precompute belum ada ---
for entry in "${DOMAIN_CONFIGS[@]}"; do
  domain="${entry%%:*}"
  case "$domain" in
    restaurant)         ckpt="yelp_restaurant"; label="restaurant" ;;
    amazon_electronics) ckpt="amazon_electronics"; label="amazon_electronics" ;;
    tripadvisor_hotel)  ckpt="tripadvisor_hotel"; label="tripadvisor_hotel" ;;
  esac
  f="checkpoints/${ckpt}/pyabsa/sabert_aspect_scores_${label}.csv"
  if [ ! -f "$f" ]; then
    echo "BERHENTI: cache belum ada -> $f"
    echo "Jalankan dulu: $PY scripts/precompute_pyabsa_sabert_scores.py --domain $domain"
    exit 1
  fi
done
echo "Prasyarat OK: cache SA-BERT-atas-aspek tersedia utk 3 domain."
echo ""

AGF_D=(64 128)
AGF_WD=(0.0 0.0001 0.001)

TOTAL=$(( ${#AGF_D[@]} * ${#AGF_WD[@]} * 3 * ${#SEEDS[@]} ))
DONE=0
T0=$(date +%s)

run_one() {
  local label="$1"; local expected_out="$2"; shift 2
  local logfile="$LOG_DIR/${label}.log"

  if [ -f "$RESULTS_DIR/$expected_out" ]; then
    echo "[$(date '+%H:%M:%S')] LEWATI  ($((DONE+1))/$TOTAL): $label (sudah ada -- resume)"
    DONE=$((DONE+1)); return
  fi

  echo "[$(date '+%H:%M:%S')] MULAI   ($((DONE+1))/$TOTAL): $label"
  if "$PY" "$@" > "$logfile" 2>&1; then
    echo "[$(date '+%H:%M:%S')] SELESAI ($((DONE+1))/$TOTAL): $label"
  else
    echo "[$(date '+%H:%M:%S')] GAGAL   ($((DONE+1))/$TOTAL): $label -- lihat $logfile" | tee -a "$FAILED_LOG"
  fi
  DONE=$((DONE+1))
}

# Urutan domain -> seed -> varian: 6 varian pada (domain,seed) yang sama
# berurutan supaya stream DeepMF/CBF dibangun SEKALI lalu dipakai 5x lagi.
for entry in "${DOMAIN_CONFIGS[@]}"; do
  domain="${entry%%:*}"; cfg="${entry#*:}"
  for seed in "${SEEDS[@]}"; do
    for d in "${AGF_D[@]}"; do
      for wd in "${AGF_WD[@]}"; do
        wd_tag=$(echo "$wd" | tr -d '.-')
        tag="d${d}wd${wd_tag}"
        # Nama file mengikuti run_attention_gated_fusion.py:1430-1432 --
        # agf_{scenario}_{run_tag}_{domain}_seed{seed}.yaml (TANPA suffix
        # stage; pemisahan stage dilakukan lewat subfolder dev/ vs test/).
        run_one "sel_${tag}_${domain}_seed${seed}" \
          "agf_a2fusionrs_clean_${tag}_${domain}_seed${seed}.yaml" \
          run_attention_gated_fusion.py --config "$cfg" \
          --scenario a2fusionrs_clean --extra-pyabsa sabert_perseq_rich \
          --global-sentiment-token \
          --residual-base user_item_bias --representation asymmetric \
          --input-standardize \
          --agf-d "$d" --agf-weight-decay "$wd" \
          --stage select --stream-cache --results-subdir "$RESULTS_SUBDIR" \
          --run-tag "$tag" --seed "$seed"
      done
    done
  done
done

ELAPSED=$(( $(date +%s) - T0 ))
echo ""
echo "=== SELESAI: $DONE/$TOTAL run, $((ELAPSED/60)) menit ==="
N_FAILED=$(wc -l < "$FAILED_LOG")
if [ "$N_FAILED" -gt 0 ]; then
  echo "PERINGATAN: $N_FAILED run GAGAL -- lihat $FAILED_LOG"
else
  echo "Semua run SUKSES."
fi
echo ""
echo "LANGKAH BERIKUTNYA: python scripts/analyze_agf_v2_selection.py"
echo "  (lalu COMMIT tabel seleksinya SEBELUM menjalankan konfirmasi di test)"
