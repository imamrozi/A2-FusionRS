#!/bin/bash
# scripts/run_agf_v2_factorial.sh
#
# KONFIRMASI di TEST SET -- arsitektur v2 (pasca-diagnosis Gerbang 1-3).
#
# ===================================================================
#  PERINGATAN: script ini MENYENTUH TEST SET (--stage confirm).
#  Jalankan HANYA SETELAH pemenang seleksi dikunci & di-COMMIT
#  (reports/agf_v2_selection/). save_results_yaml(overwrite=False)
#  akan GAGAL KERAS bila hasil sudah ada -- itu disengaja.
# ===================================================================
#
# DESAIN SEL -- dipilih supaya TIGA efek bisa dipisahkan, dan supaya klaim
# keunggulan memakai pembanding yang ADIL (bukan pembanding naif):
#
#  | Sel | Ekstraksi aspek | Scorer  | Fusi | Token global | Sequence |
#  |-----|-----------------|---------|------|--------------|----------|
#  | A   | leksikon keyword| SA-BERT | tree | implisit     | -        |  SUDAH ADA
#  | B'  | leksikon keyword| SA-BERT | AGF  | YA           | -        |  <- PEMBANDING ADIL
#  | E   | PyABSA          | SA-BERT | AGF  | YA           | -        |
#  | F   | PyABSA          | SA-BERT | AGF  | YA           | YA       |  <- TARGET
#  | F-  | PyABSA          | SA-BERT | AGF  | TIDAK        | YA       |  ablasi token global
#
# Estimasi efek (Wilcoxon per-seed + Fisher-combined):
#   efek EKSTRAKSI PyABSA (adil)  : E  - B'   <- klaim utama, scorer & token
#                                               global dikonstankan
#   efek representasi sequence    : F  - E
#   efek token sentimen global    : F  - F-
#   total vs A2-IRM               : F  - A    (mencakup SEMUA perubahan)
#
# KENAPA B' WAJIB ADA (reports/gates_1_3_summary.md Bagian 4): token
# sentimen global menolong KEDUA cabang. Ia BUKAN keunggulan PyABSA,
# melainkan fitur yang hilang dari representasi gaya A2-IRM. Pada probe
# linier, selisih thd pembanding NAIF (-2,5%/-2,4%/+0,1%) kira-kira DUA KALI
# selisih thd pembanding ADIL (-0,8%/-1,2%/+0,6%). Melaporkan angka naif
# sebagai kontribusi PyABSA = melebih-lebihkan ~2x, dan tidak akan bertahan
# di review. Sel A tetap dilaporkan sebagai konteks, TAPI klaim kontribusi
# PyABSA HARUS memakai E - B'.
#
# 4 sel baru x 3 domain x 5 seed = 60 run.
#
# PRASYARAT:
#   1. cache precompute: scripts/precompute_pyabsa_sabert_scores.py
#   2. pemenang seleksi sudah dikunci di bawah & reports/agf_v2_selection/
#      sudah di-commit.

set -uo pipefail
cd "$(dirname "$0")/.."

PY="python"
RESULTS_SUBDIR="results_phase2_clean_v2"
RESULTS_DIR="checkpoints/${RESULTS_SUBDIR}/test"
SEEDS=(42 123 456 789 1011)
LOG_DIR="checkpoints/${RESULTS_SUBDIR}/logs/factorial"
mkdir -p "$LOG_DIR"
FAILED_LOG="$LOG_DIR/FAILED_RUNS.txt"
touch "$FAILED_LOG"

# ============ PEMENANG SELEKSI DEV -- ISI SETELAH SELEKSI ============
# JANGAN menebak nilai ini. Mengubahnya tanpa menjalankan ulang seleksi =
# memilih arsitektur di test set.
WINNER_D=""
WINNER_WD=""
# =====================================================================

if [ -z "$WINNER_D" ] || [ -z "$WINNER_WD" ]; then
  cat <<'MSG'
BERHENTI: pemenang seleksi belum dikunci.

Urutan yang WAJIB diikuti:
  1. bash scripts/run_agf_v2_selection.sh
  2. python scripts/analyze_agf_v2_selection.py
  3. commit reports/agf_v2_selection/   <- jejak audit SEBELUM test disentuh
  4. isi WINNER_D & WINNER_WD di script ini, lalu jalankan.

Menebak nilainya = memilih arsitektur berdasarkan test set (p-hacking).
MSG
  exit 1
fi

DOMAIN_CONFIGS=(
  "amazon_electronics:configs/amazon_electronics_config_agf_colab.yaml"
  "restaurant:configs/yelp_config_agf_colab.yaml"
  "tripadvisor_hotel:configs/tripadvisor_hotel_config_agf_colab.yaml"
)

# --- Prasyarat: cache precompute ---
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
    echo "Jalankan: $PY scripts/precompute_pyabsa_sabert_scores.py --domain $domain"
    exit 1
  fi
done
echo "Prasyarat OK. Pemenang seleksi: d=$WINNER_D weight_decay=$WINNER_WD"
echo ""

TOTAL=60
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

# Urutan domain -> seed -> sel supaya stream DeepMF/CBF dibangun SEKALI
# per (domain,seed) lalu dipakai 3x berikutnya.
for entry in "${DOMAIN_CONFIGS[@]}"; do
  domain="${entry%%:*}"; cfg="${entry#*:}"
  for seed in "${SEEDS[@]}"; do

    # ---- SEL B': keyword + AGF + token global (PEMBANDING ADIL) ----
    run_one "Bfair_keyword_agf_global_${domain}_seed${seed}" \
      "agf_agf_keyword_cellBfair_${domain}_seed${seed}.yaml" \
      run_attention_gated_fusion.py --config "$cfg" \
      --scenario agf_keyword --extra-pyabsa none --global-sentiment-token \
      --residual-base user_item_bias --representation asymmetric \
      --input-standardize --agf-d "$WINNER_D" --agf-weight-decay "$WINNER_WD" \
      --stage confirm --stream-cache --results-subdir "$RESULTS_SUBDIR" \
      --run-tag cellBfair --seed "$seed"

    # ---- SEL E: PyABSA-ekstraksi + SA-BERT + AGF + global (tanpa sequence) ----
    run_one "E_sabert_rich_${domain}_seed${seed}" \
      "agf_a2fusionrs_clean_cellE_${domain}_seed${seed}.yaml" \
      run_attention_gated_fusion.py --config "$cfg" \
      --scenario a2fusionrs_clean --extra-pyabsa sabert_rich --global-sentiment-token \
      --residual-base user_item_bias --representation asymmetric \
      --input-standardize --agf-d "$WINNER_D" --agf-weight-decay "$WINNER_WD" \
      --stage confirm --stream-cache --results-subdir "$RESULTS_SUBDIR" \
      --run-tag cellE --seed "$seed"

    # ---- SEL F: + sequence identitas aspek (ARSITEKTUR TARGET) ----
    run_one "F_sabert_perseq_rich_${domain}_seed${seed}" \
      "agf_a2fusionrs_clean_cellF_${domain}_seed${seed}.yaml" \
      run_attention_gated_fusion.py --config "$cfg" \
      --scenario a2fusionrs_clean --extra-pyabsa sabert_perseq_rich --global-sentiment-token \
      --residual-base user_item_bias --representation asymmetric \
      --input-standardize --agf-d "$WINNER_D" --agf-weight-decay "$WINNER_WD" \
      --stage confirm --stream-cache --results-subdir "$RESULTS_SUBDIR" \
      --run-tag cellF --seed "$seed"

    # ---- SEL F-: ablasi token sentimen global ----
    run_one "Fminus_no_global_${domain}_seed${seed}" \
      "agf_a2fusionrs_clean_cellFminus_${domain}_seed${seed}.yaml" \
      run_attention_gated_fusion.py --config "$cfg" \
      --scenario a2fusionrs_clean --extra-pyabsa sabert_perseq_rich \
      --residual-base user_item_bias --representation asymmetric \
      --input-standardize --agf-d "$WINNER_D" --agf-weight-decay "$WINNER_WD" \
      --stage confirm --stream-cache --results-subdir "$RESULTS_SUBDIR" \
      --run-tag cellFminus --seed "$seed"

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
echo "LANGKAH BERIKUTNYA:"
echo "  python scripts/analyze_agf_v2_factorial.py   (efek + signifikansi)"
echo "  python scripts/analyze_agf_robustness.py     (SD, worst-case, cold-start)"
