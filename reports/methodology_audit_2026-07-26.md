# Audit Metodologi Menyeluruh — branch `main` (A2-IRM)

**Tanggal:** 2026-07-26
**Pemicu:** ditemukannya ketidakkonsistenan RNG antara script tuning dan script
verifikasi; user meminta pengecekan menyeluruh agar eksperimen "benar-benar bisa
dipertanggungjawabkan, akademis dan bersih".
**Cakupan:** `src/baseline/*`, `src/split_generator.py`, `src/evaluation/metrics.py`,
`run_baseline.py`, `run_baseline_absa.py`, `scripts/tune_deepmf_*.py`.

Temuan diurutkan berdasarkan tingkat keparahan terhadap validitas klaim ilmiah.

---

## A. TEMUAN KRITIS (mempengaruhi angka yang akan dilaporkan)

### A1. Metrik ranking (Precision/Recall/NDCG@K) praktis TIDAK BERMAKNA

**Status: PASTI (terverifikasi dengan data)**

Candidate set untuk ranking dibangun HANYA dari baris test milik user itu sendiri
(`run_baseline_absa.py`, tahap 8: `for user_id, group in test_df_eval.groupby("user_id")`).
Fakta pada domain hotel:

| Statistik | Nilai |
|---|---:|
| Baris test | 11.795 |
| User unik di test | 11.236 |
| Rata-rata item test per user | **1,05** |
| User dengan ≤5 item test | **100,0%** |
| User dengan tepat 1 item test | **96,1%** |

Artinya untuk 96% user, "me-ranking top-5 dari 1 kandidat" — Recall@5 otomatis 1,0
dan NDCG@5 otomatis ~1,0 tanpa peduli kualitas model. Ini menjelaskan kenapa SEMUA
run melaporkan Recall@K ≈ 0,9999 dan NDCG@K ≈ 0,996 — angka itu **bukan** indikator
kualitas ranking, melainkan artefak desain evaluasi.

**Risiko:** melaporkan NDCG@20 = 0,9965 di manuskrip akan langsung dipertanyakan
reviewer. Ini termasuk kategori klaim yang tidak bisa dipertahankan.

**Rekomendasi:** (a) HAPUS metrik ranking dari manuskrip dan laporkan RMSE/MAE saja,
ATAU (b) implementasi protokol ranking yang benar (sampled negatives, mis. 100
negatif per positif, atau full-catalog ranking). Opsi (b) sudah tercatat sebagai
"Fase 4" di `docs/phase1_spec.md` branch lain. Catatan kode saat ini sudah menyebut
keterbatasan ini, tapi angkanya tetap dilaporkan tanpa peringatan menonjol.

---

### A2. Val set dipakai GANDA: early-stopping DAN seleksi kandidat tuning

**Status: PASTI (by design, terbaca jelas di kode)**

`DeepMFTrainer.fit(train, val)` melakukan early stopping: menyimpan bobot dengan
val RMSE terbaik lalu me-restore-nya (`src/baseline/deepmf.py`, "Restore bobot model
DeepMF dari epoch dengan val RMSE terbaik"). Di `scripts/tune_deepmf_oof_val.py`,
model yang SUDAH dioptimasi terhadap val itu kemudian **dievaluasi lagi di val** untuk
menghitung `val_fusion_rmse` — metrik yang dipakai memilih kandidat.

Efek berantai (compounding): kelima fold OOF JUGA di-early-stop pada val yang sama,
jadi `train_deepmf_preds` pun ikut terkontaminasi preferensi terhadap val.

**Konsekuensi:** `val_fusion_rmse` **bias optimistik secara sistematis**, dan besar
biasnya BERBEDA antar konfigurasi (config yang lebih mudah "menempel" ke val dapat
untung lebih besar). Test set tidak mendapat keuntungan itu.

**Ini adalah penjelasan utama pola val↔test yang selama ini membingungkan:**

| Config | val_fusion_rmse | test RMSE | Arah |
|---|---:|---:|---|
| SGD lr=0,003 | 0,9317 (terbaik) | 1,3065 | terbalik |
| AdamW lr=0,002 ep=20 | 0,9803 | 1,1309 | terbalik |
| AdamW lr=0,002 ep=5 | 1,3583 (buruk) | 1,1135 (terbaik) | terbalik |

Bukan kebetulan/noise semata — ada mekanisme bias yang konkret.

**Rekomendasi:** butuh nested/3-way split (train / dev-early-stopping / val-seleksi)
atau k-fold di dalam train untuk early stopping, sehingga set yang dipakai memilih
kandidat TIDAK sama dengan yang dipakai early stopping.

---

### A3. Konfigurasi DEFAULT sendiri tidak stabil lintas seed (RMSE 3,14 di seed 456)

**Status: PASTI (baru muncul di run multi-seed yang sedang berjalan)**

| Config | seed 42 | seed 123 | seed 456 |
|---|---:|---:|---:|
| Default (SGD lr=0,001) | 1,1183 | 1,1339 | **3,1419** |

Angka 3,14 adalah pola "kolaps ke prediktor konstan" yang sama seperti yang terlihat
sebelumnya pada SGD lr=0,005, embedding_dim=32, dan epochs=8 (semua ≈3,06). Jadi
**bukan hanya kandidat tuning yang kolaps — baseline default pun kolaps di seed
tertentu.**

**Konsekuensi berat:** seluruh perbandingan "default 1,1183 vs kandidat X" selama
sesi ini berdiri di atas SATU seed dari konfigurasi yang ternyata punya variansi
ekstrem. Selisih 0,43% (kemenangan `epochs=5`) sama sekali tidak berarti dibanding
rentang variansi seed yang mencapai ~2,0 RMSE.

**Rekomendasi:** (a) semua klaim perbandingan WAJIB multi-seed dengan mean±SD, tidak
boleh single-seed; (b) investigasi akar penyebab kolaps (SGD tanpa momentum + sigmoid
output + inisialisasi std=0,01 adalah kombinasi rawan saturasi/vanishing gradient);
(c) pertimbangkan gradient clipping / inisialisasi lebih baik sebagai perbaikan
struktural, bukan sekadar mencari hyperparameter yang "kebetulan tidak kolaps".

---

## B. TEMUAN SEDANG (mempengaruhi validitas proses tuning, bukan angka final)

### B1. Ketidakkonsistenan RNG antara script tuning dan script verifikasi

**Status: PASTI**

- `scripts/tune_deepmf_oof_val.py`: `torch.manual_seed(seed)` dipanggil **tepat sebelum**
  membangun tiap model DeepMF (baris 187, 205).
- `run_baseline*.py`: `torch.manual_seed()` hanya dipanggil **sekali di awal pipeline**
  (baris 77/68), jauh sebelum DeepMF dilatih — dengan preprocessing, ABSA, dan CBF
  terjadi di antaranya.

Akibat: state RNG saat inisialisasi bobot DeepMF **berbeda** antara kedua script
meskipun `seed` nominal sama. Config yang sama menghasilkan model yang berbeda.

**Rekomendasi:** tambahkan `torch.manual_seed(exp_cfg["seed"])` tepat sebelum
pembuatan `DeepMFTrainer` di `run_baseline*.py` (dan sebelum `compute_oof_predictions`).

### B2. Universe item CBF berbeda antara tuning dan pipeline

**Status: PASTI**

- `run_baseline*.py`: `full_df_for_items = pd.concat([train_df, val_df, test_df])`
- `scripts/tune_deepmf_oof_val.py`: `pd.concat([train_df, val_df])`

Clustering CBF di-fit pada himpunan item yang berbeda → jumlah/isi cluster berbeda →
`cbf_preds` tidak sebanding. Menambah ketidaksetaraan antara angka tuning dan verifikasi.

**Catatan:** versi tuning justru lebih "bersih" (tidak menyentuh test). Versi pipeline
bersifat transduktif (lihat B3).

### B3. CBF bersifat transduktif — item test ikut membentuk ruang PCA/cluster

**Status: PASTI, tapi bisa dipertahankan jika dinyatakan eksplisit**

`build_item_dataframe(full_df, train_df)` mengambil `all_items` dari `full_df` yang
mencakup test. Item khusus-test masuk sebagai baris cold-start (deskripsi kosong,
rating global) tapi **tetap ikut di-fit PCA dan clustering**.

Teks/rating-nya sendiri tidak bocor (hanya dari train), jadi ini bukan label leakage.
Tapi asumsi "semua item diketahui di muka" adalah setting transduktif yang harus
dinyatakan eksplisit di manuskrip, bukan diasumsikan diam-diam.

### B4. Bug logika coordinate search — pemenang stage tidak dibandingkan ke baseline

**Status: PASTI (sudah teridentifikasi sebelumnya, belum diperbaiki)**

`min(stage_results, ...)` di `scripts/tune_deepmf_oof_val.py` hanya membandingkan
sesama kandidat DALAM stage yang sama, tidak pernah membandingkan ke `current_best`
yang dibawa dari stage sebelumnya. Log "config terbaik sejauh ini" karenanya bisa
menyesatkan — persis yang terjadi pada `stage_adamw_epochs` (melaporkan `epochs=5`
padahal `epochs=20` lebih baik di val).

---

## C. TEMUAN RINGAN / KEBERSIHAN KODE

### C1. `InteractionDataset` mengubah indeks integer jadi float saat `negative_ratio=0`

`np.concatenate([pos_users, np.array([])])` — array kosong bertipe float64 memaksa
seluruh array jadi float. Sumber `DeprecationWarning` yang muncul di semua run.
Tidak mengubah hasil (dikonversi balik ke `torch.long`), tapi rapuh dan berisik.

### C2. Field config mati (sudah diperbaiki hari ini)

`deepmf.optimizer` dan `deepmf.epochs` sebelumnya tidak pernah dibaca dari YAML —
override diam-diam diabaikan. Sudah diperbaiki di commit `d84fc57` dan `67000a7`.
**Pelajaran:** perlu validasi bahwa setiap key config benar-benar terpakai, agar
kasus serupa tidak terulang tanpa terdeteksi.

---

## D. YANG DIPERIKSA DAN TERBUKTI BERSIH

| Komponen | Status |
|---|---|
| `UserBasedSplitGenerator` | Bersih — split per-user time-aware, ada `_validate_no_leakage()` yang menolak review_id muncul di >1 split; split tidak pernah diregenerate (load-only) |
| `compute_oof_predictions()` | Logika OOF benar — tiap baris train diprediksi model yang tidak pernah melihatnya; test tetap pakai model penuh (genuinely out-of-sample) |
| `CBFPredictor.predict_train_loo()` | Aritmetika LOO benar (terverifikasi test unit); pendekatan nearest-centroid didokumentasikan eksplisit sebagai aproksimasi |
| `compute_p3_features()` | Benar — LOO untuk train, agregat penuh untuk eval, fallback global eksplisit; 6 test unit lulus |
| `build_item_dataframe()` | `description_text`/`avg_rating`/`sentiment_agg` dihitung HANYA dari train (tidak ada leakage nilai dari test) |
| `NMFDecisionTreeFusion` | Fit hanya di train, `_feature_min` dari train dipakai saat predict (tidak refit di test) |
| `compute_rmse_mae`, `precision_recall_ndcg_at_k` | Implementasi metrik itu sendiri benar (masalahnya di candidate set, lihat A1) |
| `config_utils.load_config` | Merge `_base` rekursif benar, backward compatible |

---

## E. REKOMENDASI PRIORITAS

1. **HENTIKAN pencarian hyperparameter lanjutan** sampai A2 (val ganda) dan A3
   (instabilitas seed) diselesaikan — mencari kombinasi di atas metrik yang bias dan
   baseline yang tidak stabil hanya menghasilkan temuan semu (sudah terjadi 3 kali).
2. **Selesaikan A3 dulu**: investigasi kolaps training. Ini akar masalah yang membuat
   semua perbandingan tidak reliable.
3. **Perbaiki B1 & B2** (murah, beberapa baris) agar tuning dan verifikasi setara.
4. **Putuskan A1**: hapus metrik ranking dari manuskrip, atau implementasi protokol
   yang benar.
5. **Semua klaim di manuskrip wajib multi-seed** (mean ± SD, minimal 5 seed sesuai
   protokol proyek), tidak ada pengecualian.

---

## F. CATATAN TENTANG TEMUAN LEAKAGE SENTIMEN (konteks lebih besar)

Audit ini TIDAK mengubah temuan utama sesi sebelumnya, yang tetap berdiri:
`no_sentiment` (≈1,11) → `p3_historical_agg` (0,9565) → `target_review` (0,6345),
yang menunjukkan ~2/3 perbaikan RMSE pada protokol P1 berasal dari akses ke review
target (leakage) dan ~1/3 dari sinyal historis yang sah.

Namun temuan A3 berarti angka-angka itu pun perlu diulang multi-seed sebelum
dilaporkan, karena semuanya diukur pada seed 42 saja.
