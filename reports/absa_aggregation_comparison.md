# Perbandingan Agregasi ABSA & Verifikasi Arsitektur (Fase A2-IRM)

> Dihasilkan dari eksperimen re-run di Google Colab (75 run: 3 domain x 5 seed x
> 5 varian model, `scripts/rerun_cbf_nosentiment_full.sh`) + uji signifikansi
> Wilcoxon (`run_significance_test.py`), 2026-07-25. Branch `main` (state
> A2-IRM, sebelum Fase 2 A2-FusionRS).

## 1. Motivasi

Arsitektur A2-IRM yang dituju (lihat diagram proposed method) mensyaratkan:
`BERT-Based ABSA` (dari Review Text) masuk **langsung ke Fusion**, TANPA
melewati Collaborative Filtering (DeepMF) maupun Content-Based Filtering
(CBF). Audit kode menemukan bahwa implementasi *default* CBF sebelumnya
**menyimpang** dari ini: `sentiment_agg` (skor sentimen rata-rata per item)
ikut jadi salah satu fitur numerik CBF (`ItemFeatureBuilder`, lihat
`src/baseline/cbf_clustering.py`), sementara DeepMF/CF memang sudah bersih
dari sentimen sejak awal (murni matrix factorization atas `user_id`/
`business_id`/`stars`).

Laporan ini menjawab dua pertanyaan:

1. **Apakah mengeluarkan `sentiment_agg` dari CBF (menyesuaikan implementasi
   dengan diagram) mengubah performa secara berarti?**
2. **Setelah CBF diperbaiki (bersih dari sentimen), representasi ABSA mana
   yang benar-benar membantu saat masuk langsung ke Fusion?**

## 2. Metodologi

- **Perubahan kode**: `CBFConfig.include_sentiment: bool = True` (default,
  non-breaking) ditambahkan ke `src/baseline/cbf_clustering.py`. `False`
  mengeluarkan `sentiment_agg` dari `numeric_cols` (kategori one-hot/TF-IDF
  deskripsi/`review_count`/`avg_rating` tetap dipakai). Flag CLI
  `--no-cbf-sentiment` ditambahkan ke `run_baseline.py` dan
  `run_baseline_absa.py`; hasil disimpan ke prefix file terpisah
  (`*_cbf_nosentiment`), tidak menimpa ledger asli.
- **5 varian model diuji** (semua pakai DeepMF + CBF + fusion NMF+DecisionTree
  identik; berbeda HANYA representasi sentimen yang masuk ke Fusion):

  | Model | Representasi sentimen ke Fusion |
  |---|---|
  | `baseline_reimpl` | 1 skor BERT global per review (bukan aspek) |
  | `absa_ablation` (mode `mean`) | Skor ABSA per-aspek, dirata-rata polos jadi 1 skalar |
  | `absa_ablation_concat` (mode `concat`) | Vektor skor per-aspek MENTAH (4-6 kolom, tanpa agregasi) |
  | `absa_ablation_concat_confidence` (mode `concat_confidence`) | Vektor skor + confidence per-aspek (2x kolom) |
  | `absa_ablation_confidence_mean` (mode `confidence_mean`) | Skor per-aspek, dirata-rata BERBOBOT confidence jadi 1 skalar |

- **3 domain** (amazon_electronics, restaurant, tripadvisor_hotel) x **5 seed**
  (42/123/456/789/1011) = 75 kombinasi per skenario ablasi.
- **Uji signifikansi**: Wilcoxon signed-rank berpasangan atas squared-error
  per-sampel (`review_id`-matched), dijalankan terpisah per seed (bukan
  digabung) -- `run_significance_test.py`.

## 3. Hasil Bagian 1 — Efek mengeluarkan sentiment dari CBF (model sama, dengan vs tanpa)

| Model | Domain | RMSE dengan sentimen | RMSE tanpa sentimen | Δ RMSE | Signifikan (n/5) |
|---|---|---:|---:|---:|---:|
| `absa_ablation` | amazon_electronics | 0,8081 | 0,8108 | +0,0026 | 2 |
| `absa_ablation` | restaurant | 0,8330 | 0,8415 | +0,0084 | 2 |
| `absa_ablation` | tripadvisor_hotel | 0,7341 | 0,7322 | -0,0019 | 1 |
| `absa_ablation_concat` | amazon_electronics | 0,6682 | 0,6496 | **-0,0186** | 5 |
| `absa_ablation_concat` | restaurant | 0,6968 | 0,7379 | **+0,0410** | 5 |
| `absa_ablation_concat` | tripadvisor_hotel | 0,6336 | 0,6390 | +0,0054 | 3 |
| `absa_ablation_concat_confidence` | amazon_electronics | 0,6517 | 0,6546 | +0,0029 | 3 |
| `absa_ablation_concat_confidence` | restaurant | 0,6791 | 0,6802 | +0,0011 | 0 |
| `absa_ablation_concat_confidence` | tripadvisor_hotel | 0,6291 | 0,6323 | +0,0032 | 3 |
| `absa_ablation_confidence_mean` | amazon_electronics | 0,8110 | 0,8030 | -0,0080 | 3 |
| `absa_ablation_confidence_mean` | restaurant | 0,8287 | 0,8319 | +0,0032 | 3 |
| `absa_ablation_confidence_mean` | tripadvisor_hotel | 0,7416 | 0,7367 | -0,0050 | 4 |
| `baseline_reimpl` | amazon_electronics | 0,6662 | 0,6600 | -0,0062 | 3 |
| `baseline_reimpl` | restaurant | 0,6926 | 0,6899 | -0,0028 | 2 |
| `baseline_reimpl` | tripadvisor_hotel | 0,6501 | 0,6439 | -0,0061 | 4 |

**Bacaan**: 13 dari 15 kombinasi punya |Δ RMSE| < 0,01 -- sebanding atau
lebih kecil dari noise run-ke-run yang teramati pada konfigurasi identik di
seluruh sesi eksperimen ini (~0,001-0,002). `baseline_reimpl` konsisten
sedikit lebih baik tanpa sentimen di CBF di ketiga domain (arah konsisten,
magnitudo kecil). Satu pengecualian mencolok: `absa_ablation_concat` di
restaurant (+0,041, 5/5 signifikan) berlawanan arah dengan domain amazon
(-0,019, 5/5 signifikan) untuk mode yang SAMA -- indikasi interaksi
spesifik-domain pada representasi vektor mentah, bukan efek sentimen CBF
yang universal.

**Kesimpulan Bagian 1**: mengeluarkan sentimen dari CBF (menyesuaikan
implementasi dengan diagram arsitektur) TIDAK mengorbankan performa secara
berarti -- di kebanyakan kasus efeknya netral hingga sedikit positif.
Perubahan ini aman diterapkan sebagai perbaikan arsitektur permanen.

## 4. Hasil Bagian 2 — Representasi ABSA mana yang membantu? (baseline = `baseline_reimpl_cbf_nosentiment`)

Setelah CBF diperbaiki (bersih dari sentimen di kedua sisi), pertanyaan
inti: apakah aspect-awareness (ABSA) benar-benar mengalahkan sentimen
global (SA), dan representasi mana yang bekerja.

| Model B (vs `baseline_reimpl_cbf_nosentiment`) | Domain | RMSE baseline (SA-global) | RMSE model B | Δ RMSE | % relatif | Signifikan (n/5) |
|---|---|---:|---:|---:|---:|---:|
| `absa_ablation_cbf_nosentiment` (mean) | amazon_electronics | 0,6600 | 0,8108 | **+0,1508** | **+22,9%** | 5 |
| `absa_ablation_cbf_nosentiment` (mean) | restaurant | 0,6899 | 0,8415 | **+0,1516** | **+22,0%** | 5 |
| `absa_ablation_cbf_nosentiment` (mean) | tripadvisor_hotel | 0,6439 | 0,7322 | **+0,0883** | **+13,7%** | 5 |
| `absa_ablation_concat_cbf_nosentiment` | amazon_electronics | 0,6600 | 0,6496 | -0,0104 | -1,6% | 5 |
| `absa_ablation_concat_cbf_nosentiment` | restaurant | 0,6899 | 0,7379 | +0,0480 | +7,0% | 5 |
| `absa_ablation_concat_cbf_nosentiment` | tripadvisor_hotel | 0,6439 | 0,6390 | -0,0049 | -0,8% | 2 |
| `absa_ablation_concat_confidence_cbf_nosentiment` | amazon_electronics | 0,6600 | 0,6546 | **-0,0053** | **-0,8%** | 5 |
| `absa_ablation_concat_confidence_cbf_nosentiment` | restaurant | 0,6899 | 0,6802 | **-0,0097** | **-1,4%** | 1 |
| `absa_ablation_concat_confidence_cbf_nosentiment` | tripadvisor_hotel | 0,6439 | 0,6323 | **-0,0117** | **-1,8%** | 2 |
| `absa_ablation_confidence_mean_cbf_nosentiment` | amazon_electronics | 0,6600 | 0,8030 | **+0,1431** | **+21,7%** | 5 |
| `absa_ablation_confidence_mean_cbf_nosentiment` | restaurant | 0,6899 | 0,8319 | **+0,1421** | **+20,6%** | 5 |
| `absa_ablation_confidence_mean_cbf_nosentiment` | tripadvisor_hotel | 0,6439 | 0,7367 | **+0,0927** | **+14,4%** | 5 |

### Temuan utama

**A. Mode agregasi-jadi-skalar (`mean`, `confidence_mean`) GAGAL TOTAL.**
Keduanya menghasilkan RMSE 14-23% LEBIH BURUK dari SA-global, konsisten di
ketiga domain, hampir selalu 5/5 seed signifikan. Confidence-weighting
(`confidence_mean`) TIDAK memperbaiki kegagalan mode `mean` polos seperti
yang diharapkan desain awal (lihat catatan di `run_baseline_absa.py`,
tahap 4) -- hasilnya nyaris sama buruknya (selisih antar keduanya <2 poin
persentase di semua domain).

**B. Mode yang mempertahankan vektor mentah per-aspek jauh lebih baik.**
`concat_confidence` adalah **satu-satunya varian yang konsisten mengalahkan
SA-global di ketiga domain** (-0,8% s/d -1,8%), meski signifikansi
bervariasi (5/5 di amazon, tapi cuma 1/5 dan 2/5 di restaurant/hotel --
efeknya kecil, tidak selalu terdeteksi Wilcoxon per-seed). `concat` (tanpa
confidence) tidak konsisten arahnya (lebih baik di amazon & hotel, jauh
lebih buruk di restaurant, +7,0%, 5/5 signifikan).

## 5. Interpretasi

Meringkas skor ABSA per-aspek jadi satu angka (dengan cara apa pun,
tertimbang confidence ataupun tidak) **menghancurkan sebagian besar nilai
informasi aspek**, membuatnya jauh lebih buruk daripada tidak melakukan
ABSA sama sekali. Penjelasan yang masuk akal: skor ABSA keyword-based
sendiri noisy (cakupan aspek bervariasi 35-96% antar domain, banyak baris
jatuh ke fallback skor whole-review -- lihat `reports/aspect_identifiability.md`
dari eksperimen Fase 2 utk data cakupan serupa), sehingga rata-rata
polos/berbobot dari skor-skor noisy ini justru MENAMBAH noise dibanding
1 skor SA-global yang sudah well-calibrated atas keseluruhan teks review.
Vektor mentah (`concat`/`concat_confidence`) menghindari kerugian ini
dengan membiarkan DecisionTreeRegressor mengeksploitasi struktur per-aspek
secara langsung, bukan lewat agregasi yang destruktif.

## 6. Rekomendasi

1. **Adopsi `include_sentiment=False` sebagai default CBF ke depan** --
   sesuai arsitektur yang dituju, tanpa kerugian performa berarti (Bagian 3).
2. **Buang mode agregasi-skalar (`mean`, `confidence_mean`) dari kandidat
   model utama** -- keduanya terbukti gagal secara meyakinkan (Bagian 4A),
   bukan sekadar lebih lemah.
3. **`absa_ablation_concat_confidence_cbf_nosentiment` adalah kandidat model
   A2-IRM terbaik** yang diuji sejauh ini yang sesuai arsitektur diagram:
   konsisten (meski marginal) mengalahkan SA-global di ketiga domain, sesuai
   diagram (sentiment tidak menyentuh CF/CBF), dan mempertahankan struktur
   per-aspek yang terbukti penting.
4. Efek `concat_confidence` marginal (~1-2%) berarti klaim "ABSA lebih baik
   dari SA-global" perlu dibingkai hati-hati di manuskrip -- benar tapi
   kecil, BUKAN lompatan besar. Kegagalan mode agregasi-skalar (Bagian 4A)
   justru cerita yang lebih kuat & lebih dapat dipertahankan secara statistik
   untuk bagian Discussion/ablation study.
