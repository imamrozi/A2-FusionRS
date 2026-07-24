# Leakage Audit -- Fase 1 Step 1

> Dihasilkan oleh `scripts/audit_leakage.py`. MURNI-BACA artefak yang sudah
> ada di `checkpoints/results/` dan `checkpoints/*/sentiment_bert/` --
> tidak ada model yang dilatih ulang untuk laporan ini.

## Cakupan & Keterbatasan (baca sebelum tabel di bawah)

Spec Fase 1 Step 1 meminta diagnostik atas EMPAT stream:
`deepmf_preds`, `cbf_preds`, `sentiment_score`, `base_preds`. Investigasi
langsung terhadap `run_baseline_absa.py` (baris 397-435) mengonfirmasi
`deepmf_preds` dan `cbf_preds` **tidak pernah dipersist ke disk oleh run
manapun** -- baik lokal maupun Colab. Keduanya dihitung in-memory lalu
langsung dikonsumsi `NMFDecisionTreeFusion.fit()/predict()`
(`fusion_nmf_dt.py`), tanpa pernah lewat `save_predictions()`. Ini gap
arsitektural pipeline sejak awal, bukan keterbatasan lingkungan run kali
ini.

Dua opsi diajukan ke user: (a) fit ulang DeepMF+CBF sekali secara
deterministik (split beku, kode `src/legacy/` tak diubah) untuk melengkapi
4 stream penuh, atau (b) laporkan hanya yang tersedia murni-baca. **User
memilih (b).** Konsekuensinya:

- Item 1 & 2 (RMSE sentimen-saja, RMSE global mean): dihitung penuh --
  keduanya inheren melibatkan fit model trivial (regresi linear 1-fitur,
  rata-rata konstanta) yang memang menjadi definisi item itu sendiri di
  spec, bukan "retraining" yang dihindari.
- Item 3 (korelasi + MI): hanya pasangan `sentiment_score` vs `base_preds`
  -- satu-satunya pasangan dengan kedua kolom tersedia sbg artefak.
- Item 4 (VIF) & 5 (rank efektif SVD): **tidak dihitung** -- kedua metrik
  ini hanya bermakna dengan >=3-4 stream independen; dipaksakan dengan 2
  kolom akan menghasilkan angka trivial dan menyesatkan.

Sebagai kompensasi kekuatan bukti (item 1-2-3 saja), tiga kontrol tambahan
ditambahkan atas permintaan user: DecisionTreeRegressor(max_depth=10)
(kapasitas sama persis dgn regressor fusion asli) dan isotonic regression
univariat -- keduanya atas sentiment_score saja -- plus baris referensi
RMSE fusion penuh (`base_preds`), supaya jelas seberapa dekat sentiment
sendirian terhadap model 3-stream.

Representasi domain: run `baseline_reimpl` (reimplementasi fusion NMF+DT
Darraz) seed 42 -- link pertama dalam lineage Darraz -> A2-IRM ->
A2-FusionRS, satu-satunya model dengan artefak lengkap di ketiga domain
utk audit ini.

## Hasil per domain

### Domain: `amazon_electronics`

Baris dipakai: 98400 train (98400 bertaut sentiment_score),
16580 test (16580 bertaut sentiment_score) --
artefak `predictions_baseline_reimpl_amazon_electronics_seed42.csv`,
`checkpoints/.../sentiment_bert/sentiment_scores.csv`.

**1-2. RMSE sentimen-saja (3 kontrol kapasitas) vs model konstanta vs fusion penuh**

| Model | RMSE (test) |
|---|---:|
| `stars ~ sentiment_score` (regresi linear, fit di train) | 0.7112 |
| `stars ~ sentiment_score` (DecisionTreeRegressor max_depth=10, fit di train) | 0.6339 |
| `stars ~ sentiment_score` (isotonic regression univariat, fit di train) | 0.6316 |
| Global mean (konstanta = 4.3754, dari train) | 1.2143 |
| *(referensi)* `base_preds` -- fusion NMF+DT penuh (3 stream: DeepMF+CBF+sentiment) | 0.6554 |

DT dgn max_depth=10 SENGAJA disamakan dgn kapasitas
regressor fusion asli (`FusionConfig.dt_max_depth`) -- kalau RMSE-nya
mendekati baris referensi `base_preds` walau HANYA pakai sentiment_score
(tanpa DeepMF/CBF), itu bukti kapasitas model bukan penjelas performa;
sentiment_score sendirian yang menjelaskan.

**3. Korelasi & mutual information (test set) -- HANYA pasangan tersedia**

| Pasangan stream | Pearson r | p-value | Mutual information |
|---|---:|---:|---:|
| sentiment_score vs base_preds | 0.9404 | 0.00e+00 | 2.5418 |

deepmf_preds dan cbf_preds **tidak tersedia** sbg artefak tersimpan (lihat
§ Cakupan & Keterbatasan) -- matriks 4x4 penuh tidak dapat dihitung.

**4. VIF** -- tidak dapat dihitung. VIF mengukur multikolinearitas antar
>=2 prediktor independen; dengan hanya sentiment_score yang berdiri
sendiri sbg fitur tersedia (base_preds adalah OUTPUT fusion, bukan
prediktor independen sejajar), tidak ada basis perhitungan yang bermakna.

**5. Rank efektif (SVD, ambang 99% varians)** -- tidak dihitung dgn alasan
sama seperti VIF: matriks fitur fusi penuh (4 kolom) tidak tersedia; rank
dari 2 kolom yang ada trivial (<=2) dan tidak informatif untuk pertanyaan
"apakah cross-attention Fase 2 punya ruang kerja".

### Domain: `restaurant`

Baris dipakai: 95181 train (95181 bertaut sentiment_score),
13233 test (13233 bertaut sentiment_score) --
artefak `predictions_baseline_reimpl_restaurant_seed42.csv`,
`checkpoints/.../sentiment_bert/sentiment_scores.csv`.

**1-2. RMSE sentimen-saja (3 kontrol kapasitas) vs model konstanta vs fusion penuh**

| Model | RMSE (test) |
|---|---:|
| `stars ~ sentiment_score` (regresi linear, fit di train) | 0.7492 |
| `stars ~ sentiment_score` (DecisionTreeRegressor max_depth=10, fit di train) | 0.6576 |
| `stars ~ sentiment_score` (isotonic regression univariat, fit di train) | 0.6539 |
| Global mean (konstanta = 3.7577, dari train) | 1.1516 |
| *(referensi)* `base_preds` -- fusion NMF+DT penuh (3 stream: DeepMF+CBF+sentiment) | 0.6988 |

DT dgn max_depth=10 SENGAJA disamakan dgn kapasitas
regressor fusion asli (`FusionConfig.dt_max_depth`) -- kalau RMSE-nya
mendekati baris referensi `base_preds` walau HANYA pakai sentiment_score
(tanpa DeepMF/CBF), itu bukti kapasitas model bukan penjelas performa;
sentiment_score sendirian yang menjelaskan.

**3. Korelasi & mutual information (test set) -- HANYA pasangan tersedia**

| Pasangan stream | Pearson r | p-value | Mutual information |
|---|---:|---:|---:|
| sentiment_score vs base_preds | 0.9010 | 0.00e+00 | 3.0487 |

deepmf_preds dan cbf_preds **tidak tersedia** sbg artefak tersimpan (lihat
§ Cakupan & Keterbatasan) -- matriks 4x4 penuh tidak dapat dihitung.

**4. VIF** -- tidak dapat dihitung. VIF mengukur multikolinearitas antar
>=2 prediktor independen; dengan hanya sentiment_score yang berdiri
sendiri sbg fitur tersedia (base_preds adalah OUTPUT fusion, bukan
prediktor independen sejajar), tidak ada basis perhitungan yang bermakna.

**5. Rank efektif (SVD, ambang 99% varians)** -- tidak dihitung dgn alasan
sama seperti VIF: matriks fitur fusi penuh (4 kolom) tidak tersedia; rank
dari 2 kolom yang ada trivial (<=2) dan tidak informatif untuk pertanyaan
"apakah cross-attention Fase 2 punya ruang kerja".

### Domain: `tripadvisor_hotel`

Baris dipakai: 64280 train (64280 bertaut sentiment_score),
11795 test (11795 bertaut sentiment_score) --
artefak `predictions_baseline_reimpl_tripadvisor_hotel_seed42.csv`,
`checkpoints/.../sentiment_bert/sentiment_scores.csv`.

**1-2. RMSE sentimen-saja (3 kontrol kapasitas) vs model konstanta vs fusion penuh**

| Model | RMSE (test) |
|---|---:|
| `stars ~ sentiment_score` (regresi linear, fit di train) | 0.7128 |
| `stars ~ sentiment_score` (DecisionTreeRegressor max_depth=10, fit di train) | 0.6122 |
| `stars ~ sentiment_score` (isotonic regression univariat, fit di train) | 0.6066 |
| Global mean (konstanta = 3.9316, dari train) | 0.9163 |
| *(referensi)* `base_preds` -- fusion NMF+DT penuh (3 stream: DeepMF+CBF+sentiment) | 0.6396 |

DT dgn max_depth=10 SENGAJA disamakan dgn kapasitas
regressor fusion asli (`FusionConfig.dt_max_depth`) -- kalau RMSE-nya
mendekati baris referensi `base_preds` walau HANYA pakai sentiment_score
(tanpa DeepMF/CBF), itu bukti kapasitas model bukan penjelas performa;
sentiment_score sendirian yang menjelaskan.

**3. Korelasi & mutual information (test set) -- HANYA pasangan tersedia**

| Pasangan stream | Pearson r | p-value | Mutual information |
|---|---:|---:|---:|
| sentiment_score vs base_preds | 0.8175 | 0.00e+00 | 2.9369 |

deepmf_preds dan cbf_preds **tidak tersedia** sbg artefak tersimpan (lihat
§ Cakupan & Keterbatasan) -- matriks 4x4 penuh tidak dapat dihitung.

**4. VIF** -- tidak dapat dihitung. VIF mengukur multikolinearitas antar
>=2 prediktor independen; dengan hanya sentiment_score yang berdiri
sendiri sbg fitur tersedia (base_preds adalah OUTPUT fusion, bukan
prediktor independen sejajar), tidak ada basis perhitungan yang bermakna.

**5. Rank efektif (SVD, ambang 99% varians)** -- tidak dihitung dgn alasan
sama seperti VIF: matriks fitur fusi penuh (4 kolom) tidak tersedia; rank
dari 2 kolom yang ada trivial (<=2) dan tidak informatif untuk pertanyaan
"apakah cross-attention Fase 2 punya ruang kerja".


## Interpretasi (acuan tabel §"Interpretasi yang diharapkan" di phase1_spec.md)

Baca §"Cakupan & Keterbatasan" di atas sebelum menafsirkan -- korelasi
antar-stream **tidak** bisa dinilai lengkap (hanya 1 dari 6 pasangan
mungkin di matriks 4x4 yang tersedia), dan rank efektif/VIF sama sekali
tidak dihitung. Baris RMSE(`stars ~ sentiment_score`) tetap dapat
dibandingkan langsung terhadap perkiraan 0,65 di spec -- ini bagian paling
kuat berdiri sendiri dari audit ini.

**Keputusan lanjutan (Step 2, sesi web, di luar cakupan sesi ini):**
apakah kekuatan bukti pada item 1-2-3 (tanpa item 4-5) sudah cukup untuk
Step 2, atau apakah opsi (a) -- fit ulang DeepMF+CBF -- perlu dijalankan
belakangan untuk melengkapi item 4-5.
