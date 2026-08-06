#!/bin/bash
# scripts/rerun_agf_triage.sh
#
# Stage F (re-validasi A2-FusionRS di atas A2-IRM terkoreksi, plan
# pure-painting-wilkes.md): triage 3 skenario kunci x 3 domain x 5 seed =
# 45 run baru, DIJALANKAN MANUAL OLEH USER DI COLAB (branch
# phase2-a2-fusionrs-v2, PyABSA cache 3 domain SUDAH ada lokal/Drive --
# TIDAK perlu inferensi PyABSA baru).
#
# Kenapa 45 run (bukan 60 spt draft plan awal): skenario ke-4 ("static +
# keyword") = A2-IRM concat_confidence, SUDAH ADA hasilnya di
# checkpoints/results/absa_ablation_concat_confidence_cbf_nosentiment_
# {domain}_seed{seed}.yaml (90-run matrix AdamW yang sudah diperbaiki) --
# tidak perlu dihitung ulang, cukup dipakai langsung sbg baseline
# perbandingan di Stage G (build_manuscript_table.py-style).
#
# 3 skenario yg DIJALANKAN di sini:
#   static_pyabsa      : kontrol atribusi (tree NMF+DT + sentimen PyABSA)
#   agf_keyword         : AGF + keyword ABSA POLOS (tanpa redesign) --
#                          titik banding "apakah redesign Jalur X sepadan?"
#   agf_keyword_oof_perseq : "A2-FusionRS penuh" -- konfigurasi pemenang
#                          verdict lama (memori phase2-agf-final-verdict):
#                          --representation asymmetric --residual-base
#                          static_fusion_oof --extra-pyabsa perseq
#
# Tujuan (Stage G): uji ulang KEDUA klaim verdict lama di atas DeepMF/CBF
# yang SUDAH benar (AdamW, OOF/LOO) -- verdict lama (2026-07-18) dihitung
# SELURUHNYA di atas DeepMF SGD kolaps (lihat commit message Stage
# A-D & memori phase2-agf-final-verdict utk bukti lengkap).
#
# Satu run gagal TIDAK menghentikan seluruh matriks -- dicatat ke
# FAILED_RUNS.txt, sisanya tetap lanjut. RESUMABLE: run yg hasilnya sudah
# ada di-skip.

set -uo pipefail
cd "$(dirname "$0")/.."

PY="python"  # Colab: python3 di PATH sistem, bukan venv/Scripts/python.exe (Windows lokal)
RESULTS_DIR="checkpoints/results"
SEEDS=(42 123 456 789 1011)
LOG_DIR="checkpoints/results/logs/agf_triage"
mkdir -p "$LOG_DIR"
FAILED_LOG="$LOG_DIR/FAILED_RUNS.txt"
touch "$FAILED_LOG"

# domain_label:config_path (varian _colab, GPU Colab)
DOMAIN_CONFIGS=(
  "amazon_electronics:configs/amazon_electronics_config_agf_colab.yaml"
  "restaurant:configs/yelp_config_agf_colab.yaml"
  "tripadvisor_hotel:configs/tripadvisor_hotel_config_agf_colab.yaml"
)

TOTAL=45
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

# ---- 1. static_pyabsa: kontrol atribusi (15 run) ----
for entry in "${DOMAIN_CONFIGS[@]}"; do
  domain="${entry%%:*}"; cfg="${entry#*:}"
  for seed in "${SEEDS[@]}"; do
    out="agf_static_pyabsa_${domain}_seed${seed}.yaml"
    run_one "static_pyabsa_${domain}_seed${seed}" "$out" \
      run_attention_gated_fusion.py --config "$cfg" --scenario static_pyabsa --seed "$seed"
  done
done

# ---- 2. agf_keyword polos: titik banding redesign (15 run) ----
for entry in "${DOMAIN_CONFIGS[@]}"; do
  domain="${entry%%:*}"; cfg="${entry#*:}"
  for seed in "${SEEDS[@]}"; do
    out="agf_agf_keyword_${domain}_seed${seed}.yaml"
    run_one "agf_keyword_${domain}_seed${seed}" "$out" \
      run_attention_gated_fusion.py --config "$cfg" --scenario agf_keyword --seed "$seed"
  done
done

# ---- 3. agf_keyword + asymmetric + residual OOF + perseq: A2-FusionRS
#         PENUH (konfigurasi pemenang verdict lama) (15 run) ----
for entry in "${DOMAIN_CONFIGS[@]}"; do
  domain="${entry%%:*}"; cfg="${entry#*:}"
  for seed in "${SEEDS[@]}"; do
    out="agf_agf_keyword_oof_perseq_${domain}_seed${seed}.yaml"
    run_one "agf_keyword_oof_perseq_${domain}_seed${seed}" "$out" \
      run_attention_gated_fusion.py --config "$cfg" --scenario agf_keyword \
      --representation asymmetric --residual-base static_fusion_oof \
      --extra-pyabsa perseq --run-tag oof_perseq --seed "$seed"
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
