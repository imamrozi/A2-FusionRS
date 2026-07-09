# A2-FusionRS: Pipeline Eksperimen Fase 1

Skeleton pipeline untuk reimplementasi baseline (Darraz et al., ESWA 2025)
dan persiapan fair comparison dengan model A2-FusionRS.

## Status Implementasi

| Modul | Status | Catatan |
|---|---|---|
| `src/data_loader.py` | Fungsional, belum diuji pada data riil | Perlu file dataset diunduh manual |
| `src/split_generator.py` | Fungsional, ada sanity check anti-leakage | Siap dipakai |
| `src/preprocessing.py` | Fungsional | Siap dipakai |
| `src/baseline/sentiment_bert.py` | Fungsional, asumsi labeling perlu divalidasi | Lihat docstring `derive_sentiment_label()` |
| `src/baseline/deepmf.py` | Fungsional + method `predict()` untuk fusion | Belum smoke-test (butuh torch) |
| `src/baseline/cbf_clustering.py` | **Lengkap**: `build_item_dataframe()` + `CBFPredictor` | Lolos smoke test dengan data dummy |
| `src/baseline/fusion_nmf_dt.py` | Fungsional | Lolos smoke test dengan data dummy |
| `src/evaluation/metrics.py` | Fungsional, termasuk `sanity_check_rmse()` | Lolos smoke test dengan data dummy |
| `run_baseline.py` | **Lengkap tahap 1-8** | Perlu full run pada data riil untuk validasi end-to-end |

**Simplifikasi yang perlu diperhatikan (didokumentasikan di kode):**
- Prediksi CBF memakai reduksi cosine-similarity-ke-preference-cluster (lihat docstring `CBFPredictor`) -- bukan klaim identik dengan detail eksak baseline paper
- Ranking metrics (Precision/Recall/NDCG@K) memakai candidate set terbatas pada item test set (bukan full-catalog ranking) -- **wajib dinyatakan sebagai batasan di bagian metodologi manuskrip**
- Fitur item (`review_count`, `avg_rating`) dihitung dari TRAIN SET SAJA, bukan dari kolom `business_review_count`/`business_stars` bawaan dataset (yang berpotensi sudah mengagregasi data test)

## Langkah Setup

### Laptop Lokal (i5/24GB, tanpa GPU)

```bash
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

Opsional (disarankan): instal torch versi CPU-only terlebih dahulu agar
lebih ringan (lihat catatan di dalam `requirements.txt`):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### Google Colab (Pro/Pro+)

Jangan pakai venv di Colab. Jalankan langsung di cell notebook:
```python
!pip install -q -r requirements-colab.txt

import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
```
`torch`, `pandas`, `numpy`, `scipy` sudah tersedia default di Colab -- tidak
disertakan di `requirements-colab.txt` agar tidak menimpa build CUDA bawaan.

## Langkah Sebelum Menjalankan Eksperimen

1. **Unduh dataset** (lihat diskusi sumber data -- data.world/brianray sudah retired,
   gunakan Kaggle "Yelp Recruiting Competition"/`yelp-recsys-2013`, atau
   Yelp Open Dataset resmi sebagai fallback)
2. **Inspeksi skema** sebelum apa pun:
   ```bash
   python scripts/inspect_raw_schema.py --path data/raw/<file_dataset_anda>
   ```
   Script ini otomatis mendeteksi format (CSV/JSON) dan membandingkan kolom
   yang tersedia dengan `REQUIRED_COLUMNS` di `src/data_loader.py`. Jika ada
   kolom hilang/nama berbeda, sesuaikan `data_loader.py` dulu sebelum lanjut.
3. **Buat subset kecil untuk validasi cepat** (~5.000 baris, menit bukan jam):
   ```bash
   python scripts/make_subset.py \
     --input data/raw/<file_dataset_anda> \
     --output data/raw/yelp_subset_5k.csv \
     --domain restaurant \
     --n-rows 5000
   ```
4. **Jalankan smoke test pipeline penuh pada subset:**
   ```bash
   python run_baseline.py --config configs/yelp_config_quicktest.yaml
   ```
   Config ini memakai hyperparameter yang diperkecil (epoch=1, dimensi
   embedding kecil, dsb) khusus untuk verifikasi bahwa seluruh tahap 1-8
   berjalan tanpa error -- **BUKAN untuk angka RMSE yang dilaporkan di
   manuskrip**.
5. **Konfirmasi strategi labeling sentiment** di `sentiment_bert.py` terhadap
   detail metodologi lengkap baseline paper sebelum full run
6. **Setelah smoke test lolos**, jalankan versi penuh:
   ```bash
   python run_baseline.py --config configs/yelp_config.yaml
   ```
   (sebaiknya di Colab Pro/Pro+ untuk tahap fine-tuning BERT)

## Menjalankan Pipeline

```bash
python run_baseline.py --config configs/yelp_config.yaml
```

## Catatan Penting (Metodologis)

- **Split identik untuk semua model**: Jangan generate split baru untuk tiap
  model yang dibandingkan. Gunakan `data/splits/yelp_restaurant/` yang sama
  untuk baseline, SVD, NCF, DeepFM, A2-FusionRS, dan semua varian ablasi.
- **Anomali RMSE baseline paper (0.01-0.02)**: Sebelum menyandingkan hasil
  reimplementasi dengan angka yang dilaporkan paper asli, jalankan
  `sanity_check_rmse()` untuk mendeteksi dini jika pipeline kita sendiri
  mengalami masalah serupa (leakage, evaluasi pada train set, dsb).
- **Checkpoint ke Google Drive** jika dijalankan di Colab -- set
  `logging.checkpoint_dir` di config ke path yang di-mount ke Drive, karena
  runtime Colab bisa disconnect sewaktu-waktu.

## Struktur Selanjutnya (Fase 2 & A2-FusionRS)

Modul-modul `data_loader.py`, `split_generator.py`, `preprocessing.py`, dan
`evaluation/metrics.py` dirancang untuk dipakai ulang tanpa modifikasi saat
membangun:
- Dataset Amazon & TripAdvisor (Fase 2, generalisasi multi-domain)
- Arsitektur A2-FusionRS penuh (stream ABSA-BERT, Attention-Gated Fusion) --
  akan ditempatkan di `src/a2fusionrs/` sejajar dengan `src/baseline/`
- Varian ablasi A1-A5
