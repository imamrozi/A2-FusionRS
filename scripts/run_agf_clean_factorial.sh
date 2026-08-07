#!/bin/bash
# scripts/run_agf_clean_factorial.sh
#
# Tahap 7 (plan pure-painting-wilkes.md): DESAIN FAKTORIAL 2x2 di TEST SET.
#
# ===================================================================
#  PERINGATAN: script ini MENYENTUH TEST SET (--stage confirm).
#  Jalankan HANYA SETELAH pemenang seleksi dikunci & di-commit
#  (lihat reports/agf_clean_selection/README.md, commit c71d7b8).
#  save_results_yaml(overwrite=False) akan GAGAL KERAS bila hasil
#  sudah ada -- itu disengaja: run konfirmasi tidak boleh diulang
#  diam-diam.
# ===================================================================
#
# TUJUAN: memisahkan pengaruh DUA penggantian yang diusulkan, yang TIDAK
# bisa dijawab oleh perbandingan A2-IRM vs arsitektur-bersih saja (di situ
# dua faktor berubah sekaligus):
#
#   | Sel | Sumber ABSA          | Fusi          | Status               |
#   |-----|----------------------|---------------|----------------------|
#   |  A  | keyword concat+conf  | statis NMF+DT | = A2-IRM, SUDAH ADA  |
#   |  B  | keyword concat+conf  | dinamis AGF   | di-run di sini       |
#   |  C  | PyABSA rich 9-dim    | statis NMF+DT | di-run di sini       |
#   |  D0 | PyABSA rich 9-dim    | dinamis AGF   | di-run di sini       |
#   |  D  | PyABSA seq + rich    | dinamis AGF   | di-run (ARSITEKTUR   |
#   |     |                      |               |  TARGET)             |
#
# Estimasi efek:
#   efek PyABSA (fusi konstan)     : C - A  (statis),  D0 - B  (AGF)
#   efek fusi dinamis (info konstan): B - A  (keyword), D0 - C  (PyABSA)
#   interaksi ABSA x fusi           : (D0 - C) - (B - A)
#   efek representasi sequence      : D - D0
#
# BATASAN YANG WAJIB DINYATAKAN DI MANUSKRIP: sel C & D0 memakai PyABSA
# rich 9-dim (agregasi), BUKAN sequence, karena tree NMF+DT secara
# STRUKTURAL tidak bisa mengkonsumsi sequence panjang-variabel. Jadi
# D - A (arsitektur bersih vs A2-IRM) mencakup TIGA perubahan sekaligus;
# dekomposisinya justru itulah gunanya sel B/C/D0. Ketidakmampuan tree
# mengkonsumsi sequence bukan cacat desain eksperimen -- itu temuan
# struktural yang mendukung tesis fusi dinamis.
#
# JANGKAR: sel B, D0, D memakai konfigurasi PEMENANG SELEKSI DEV
# (--residual-base user_item_bias --representation asymmetric), SAMA utk
# ketiganya supaya perbandingan apple-to-apple. Sel A & C fusi statis
# (tree tidak butuh jangkar).
#
# 4 sel baru x 3 domain x 5 seed = 60 run. Dgn --stream-cache, stream
# DeepMF/CBF dibangun sekali per (domain, seed) lalu dipakai 4 sel.

set -uo pipefail
cd "$(dirname "$0")/.."

PY="python"
RESULTS_DIR="checkpoints/results_phase2_clean/test"
SEEDS=(42 123 456 789 1011)
LOG_DIR="checkpoints/results_phase2_clean/logs/factorial"
mkdir -p "$LOG_DIR"
FAILED_LOG="$LOG_DIR/FAILED_RUNS.txt"
touch "$FAILED_LOG"

DOMAIN_CONFIGS=(
  "amazon_electronics:configs/amazon_electronics_config_agf_colab.yaml"
  "restaurant:configs/yelp_config_agf_colab.yaml"
  "tripadvisor_hotel:configs/tripadvisor_hotel_config_agf_colab.yaml"
)

# Konfigurasi pemenang seleksi DEV -- JANGAN diubah tanpa menjalankan
# ulang seleksi (mengubahnya di sini = memilih arsitektur di test).
WINNER_RESIDUAL="user_item_bias"
WINNER_REPR="asymmetric"

TOTAL=60
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

# Urutan domain -> seed -> sel: 4 sel pada (domain,seed) yang sama berurutan
# supaya cache stream dibangun sekali lalu dipakai 3x berikutnya.
for entry in "${DOMAIN_CONFIGS[@]}"; do
  domain="${entry%%:*}"; cfg="${entry#*:}"
  for seed in "${SEEDS[@]}"; do

    # ---- SEL B: keyword ABSA + AGF (jangkar pemenang) ----
    run_one "B_keyword_agf_${domain}_seed${seed}" \
      "agf_agf_keyword_cellB_${domain}_seed${seed}.yaml" \
      run_attention_gated_fusion.py --config "$cfg" \
      --scenario agf_keyword --extra-pyabsa none \
      --residual-base "$WINNER_RESIDUAL" --representation "$WINNER_REPR" \
      --input-standardize --stage confirm --stream-cache \
      --run-tag cellB --seed "$seed"

    # ---- SEL C: PyABSA-rich + fusi statis NMF+DT ----
    run_one "C_pyabsa_tree_${domain}_seed${seed}" \
      "agf_static_pyabsa_rich_cellC_${domain}_seed${seed}.yaml" \
      run_attention_gated_fusion.py --config "$cfg" \
      --scenario static_pyabsa_rich \
      --stage confirm --stream-cache \
      --run-tag cellC --seed "$seed"

    # ---- SEL D0: PyABSA-rich + AGF (tanpa sequence) ----
    run_one "D0_pyabsa_agf_${domain}_seed${seed}" \
      "agf_a2fusionrs_clean_cellD0_${domain}_seed${seed}.yaml" \
      run_attention_gated_fusion.py --config "$cfg" \
      --scenario a2fusionrs_clean --extra-pyabsa rich \
      --residual-base "$WINNER_RESIDUAL" --representation "$WINNER_REPR" \
      --input-standardize --stage confirm --stream-cache \
      --run-tag cellD0 --seed "$seed"

    # ---- SEL D: PyABSA sequence + rich + AGF (ARSITEKTUR TARGET) ----
    run_one "D_pyabsa_seq_agf_${domain}_seed${seed}" \
      "agf_a2fusionrs_clean_cellD_${domain}_seed${seed}.yaml" \
      run_attention_gated_fusion.py --config "$cfg" \
      --scenario a2fusionrs_clean --extra-pyabsa perseq_rich \
      --residual-base "$WINNER_RESIDUAL" --representation "$WINNER_REPR" \
      --input-standardize --stage confirm --stream-cache \
      --run-tag cellD --seed "$seed"

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
echo "LANGKAH BERIKUTNYA: python scripts/analyze_agf_factorial.py"
