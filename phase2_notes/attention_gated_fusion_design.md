# Desain Attention-Gated Fusion (A2-FusionRS, Fase 2)

Dokumen ini menuangkan hasil diskusi desain arsitektur Attention-Gated Fusion
untuk A2-FusionRS -- pengembangan lanjutan dari A2-IRM (Fase 1, sudah
selesai & termuat di `manuscript/A2-IRM_manuscript_draft.md`). Ditulis
SEBELUM implementasi kode dimulai, sebagai acuan bersama supaya keputusan
arsitektur & cakupan eksperimen tidak berubah-ubah di tengah jalan.

## 1. Kondisi kode Fase 1 yang jadi pijakan (diverifikasi langsung, bukan asumsi)

- **DeepMF** (`src/baseline/deepmf.py`): `DeepMFModel.forward(..., return_latent=True)`
  SUDAH mendukung pengembalian representasi laten 32-dim (output `deep_layers`,
  hidden layer terakhir dari `[256,128,64,32]`, sebelum `output_layer`) --
  tapi `DeepMFTrainer.predict()` yang dipakai pipeline saat ini TIDAK
  memanfaatkannya, hanya mengambil skalar rating akhir. Perlu plumbing baru
  (bukan perubahan arsitektur DeepMF itu sendiri) untuk mengekspos latent
  ini per baris user-item.
- **CBF** (`src/baseline/cbf_clustering.py`): BUKAN neural net -- pipeline
  klasik (KMeans/Agglomerative + rumus blend manual di `CBFPredictor.predict()`)
  yang saat ini hanya mengeluarkan skalar. Representasi vektor yang sudah
  dihitung di tengah jalan dan bisa diekspos TANPA mengubah clustering:
  vektor fitur item hasil PCA (`ItemFeatureBuilder`, ~50 dim) + vektor
  preferensi user atas cluster (`user_cluster_pref`, dim = jumlah cluster
  terpilih, bervariasi 2-20 tergantung domain).
- **ABSA** (`src/a2fusionrs/absa_bert.py`, varian Concat+Confidence -- performa
  terbaik di A2-IRM): sudah alami berbentuk vektor (2K dim, K=4/5/6 tergantung
  domain, lihat Table II manuskrip A2-IRM), siap dipakai langsung sebagai
  input attention tanpa modifikasi.
- **Fusi Fase 1** (`src/baseline/fusion_nmf_dt.py`): hanya menerima 3 SKALAR
  (prediksi DeepMF, prediksi CBF, skor sentimen/vektor ABSA) -> NMF -> concat
  fitur asli -> DecisionTreeRegressor. Ini jadi baseline pembanding utama.

**Keputusan desain kunci**: ubah dari late-fusion skalar (Fase 1) ke
feature-level fusion vektor -- attention butuh representasi kaya per
modalitas, bukan angka tunggal. DeepMF & ABSA sudah/mudah menyediakan ini;
CBF butuh sedikit plumbing tambahan (bukan re-desain clustering).

## 2. Arsitektur yang disepakati

```
DeepMF latent (32d) ----> Linear proj ----\
CBF (PCA item + cluster pref user) -> Linear proj --> [3 token, dim d bersama]
ABSA Concat+Confidence (2K d) --------> Linear proj ----/
                                                |
                                    Multi-head Self-Attention
                                    (2-4 head, residual + LayerNorm)
                                                |
                                    Gated Fusion (MLP kecil + softmax/
                                    sigmoid per modalitas -> weighted sum)
                                                |
                                    Prediction head (MLP d -> d/2 -> 1)
                                                |
                                        Prediksi rating
```

Dimensi bersama default: d=64 (akan divariasikan di studi sensitivitas,
lihat Tier 2 di bawah).

### Skema training (2 tahap, BUKAN end-to-end penuh)

DeepMF trainable, CBF non-differentiable (klasik), ABSA frozen pretrained --
end-to-end joint training tidak realistis/tidak perlu.

1. **Tahap 1**: latih DeepMF, fit CBF, skor ABSA persis seperti Fase 1
   (kode sebagian besar reuse langsung) -- ketiganya jadi "expert" beku.
2. **Tahap 2**: freeze ketiganya, ekstrak fitur vektor tiap baris, latih
   HANYA modul attention+gate+prediction-head (jaringan kecil, ribuan
   parameter) dengan MSE loss + Adam. Murah secara komputasi -- lihat
   estimasi biaya di Bagian 4.

## 3. Cakupan eksperimen (diperluas utk target jurnal Q1, halaman lebih
   longgar dari conference paper)

### Tier 1 -- Wajib (menjawab pertanyaan reviewer utama: dari mana gain-nya?)

1. **Desain faktorial 2x2 (Fusion x ABSA-extraction)** -- paling penting,
   memisahkan kontribusi "fusi baru" vs "ABSA baru" (dua hal yang berubah
   sekaligus di Fase 2):
   - (a) Static (NMF+DT) + Keyword ABSA = baseline A2-IRM, SUDAH ADA.
   - (b) Static (NMF+DT) + Model-based ABSA (PyABSA) = baru.
   - (c) AGF + Keyword ABSA = baru.
   - (d) AGF + Model-based ABSA (PyABSA) = **A2-FusionRS penuh**, baru.
2. **Leave-one-modality-out**: AGF dengan hanya 2 dari 3 modalitas aktif
   (DeepMF+CBF, DeepMF+ABSA, CBF+ABSA) vs ketiganya -- kontribusi marjinal
   tiap modalitas + validasi bahwa gating belajar bobot masuk akal.
3. **Arsitektur internal AGF**: Attention-only (tanpa gate, mis. mean-pool
   token hasil attention) vs Gating-only (tanpa cross-attention, gate
   langsung dari proyeksi awal) vs Full AGF -- mengisolasi kontribusi
   attention vs gating secara terpisah.
4. Semua di atas dijalankan di 3 domain (restoran/Amazon Electronics/
   TripAdvisor Hotel) x 5 seed x uji signifikansi (Wilcoxon per seed +
   Fisher combined p-value), konsisten metodologi A2-IRM.

### Tier 2 -- Sangat menguatkan, biaya moderat

5. **Baseline eksternal** (bukan cuma versi lama sendiri): Concat+MLP
   (deep, tanpa attention/gating) dan Weighted-average tetap (gating naif,
   tanpa attention) -- pembanding adil untuk klaim "attention DAN gating
   penting".
6. **Sensitivitas hyperparameter**: variasi dimensi embedding bersama (d)
   dan jumlah attention head. Dijalankan TERBATAS (1 domain x 2 seed x
   ~5 kombinasi) -- studi pendukung, bukan hasil utama, tidak perlu
   dikali penuh 3 domain x 5 seed.
7. **Formalisasi perbandingan checkpoint PyABSA** (english vs multilingual)
   -- data sudah ada di `phase2_notes/pyabsa_investigation.md`, tinggal
   dirapikan jadi bagian metodologi/hasil paper.

### Tier 3 -- Menambah kedalaman, TIDAK butuh training baru (reuse hasil Tier 1/2)

8. **Analisis efisiensi/trade-off**: tabel jumlah parameter, waktu
   training, waktu inferensi per-sampel -- fusi statis vs varian-varian
   AGF. Menjawab "berapa biaya tambahan utk akurasi yang didapat?" --
   dicatat otomatis selama run Tier 1/2 berjalan (logging waktu), bukan
   eksperimen terpisah.
9. **Studi kasus interpretability**: contoh konkret bobot gate per-modalitas
   pada pasangan user-item spesifik (mis. review dgn sinyal ABSA kuat ->
   gate ABSA tinggi, vs review generik -> gate ABSA rendah) -- bukti
   kualitatif gating bukan black box.
10. **Stratifikasi sparsity**: pecah test set berdasarkan jumlah interaksi
    user/item (jarang vs sering), bandingkan RMSE static vs AGF per
    stratum -- relevan utk isu cold-start yang sering jadi concern
    reviewer RecSys. Analisis ulang atas prediksi yang sudah ada, bukan
    run baru.

## 4. Estimasi biaya komputasi (lingkungan: Google Colab, GPU T4)

### Biaya dominan: skoring PyABSA (checkpoint "english", direkomendasikan
di `pyabsa_investigation.md`) -- SEKALI per domain, BUKAN dikali seed/skenario

Skor ABSA per-review tidak bergantung split train/val/test atau seed
(beda dgn DeepMF yang bergantung split) -- cukup di-skor sekali per domain,
di-cache per `review_id` (replikasi pola caching yang sudah ada di
`run_baseline_absa.py` utk ABSA keyword), lalu dipakai ulang utk semua
5 seed dan semua skenario ablasi.

| Domain | Jumlah review | Estimasi waktu (GPU T4, 0,165 detik/review) |
|---|---|---|
| TripAdvisor Hotel | 79.562 | ~3,6 jam (terdokumentasi di benchmark) |
| Amazon Electronics | 122.068 | ~5,6 jam (terdokumentasi di benchmark) |
| Restaurant | 118.695 | ~5,4 jam (ekstrapolasi metodologi sama) |
| **Total** | | **~14,6 jam GPU** |

### Biaya training modul AGF (per skenario x domain x seed)

Jaringan kecil (proyeksi + attention 3-token + gate + MLP head -- ribuan
parameter, bukan jutaan), dilatih di atas fitur yang SUDAH diekstrak
(tidak perlu forward-pass BERT lagi). Estimasi kasar (belum dibenchmark
langsung -- akan dikonfirmasi di run pertama): 2-10 menit/run di GPU Colab.

- Tier 1+2 skenario baru: Static+Model-ABSA, AGF+Keyword, Full AGF,
  Attention-only, Gating-only, 3x leave-one-out, Concat+MLP,
  Weighted-average = **10 skenario** x 3 domain x 5 seed = **150 run**
  -> ~12,5 jam GPU.
- Studi sensitivitas hyperparameter: 1 domain x 2 seed x ~5 kombinasi
  -> ~1 jam GPU.

### Total: ~14,6 + 12,5 + 1 ~= **28 jam GPU Colab**

Perlu dicicil per domain (simpan cache skor ABSA ke Google Drive setelah
tiap domain selesai, pola yang sama seperti checkpoint PyABSA yang sudah
terbukti bekerja di investigasi sebelumnya) -- bukan harus 1 sesi panjang.
Colab Pro kemungkinan cukup kalau dicicil beberapa sesi; Pro+ lebih nyaman
kalau ingin sesi lebih panjang sekali jalan.

## 5. Belum diputuskan / perlu dikonfirmasi saat implementasi

- Nilai eksak d (dimensi bersama) dan jumlah attention head default sebelum
  studi sensitivitas -- usulan awal d=64, 2 head, akan divalidasi empiris.
  perlu dipastikan.
- Jumlah epoch training modul AGF (belum ada observasi val-loss curve
  sungguhan seperti DeepMF punya).
- Detail arsitektur gate untuk leave-one-out (2 modalitas) -- apakah pakai
  modul gate yang sama (disesuaikan ukurannya) atau modul terpisah.
- Representasi CBF final: perlu diputuskan apakah PCA item vector + user
  cluster preference vector digabung (concat) sebelum proyeksi linear, atau
  diproyeksikan terpisah lalu dijumlah -- pilih setelah lihat performa awal.

## 6. Rencana perbaikan (setelah Stage 7): representasi PyABSA per-aspek via attention

**Masalah yang teridentifikasi** (diskusi pasca-150-run, sebelum Stage 7):
`vectorize_absa_features()` (`src/a2fusionrs/pyabsa_scorer.py`) meringkas
output PyABSA yang SEBENARNYA per-aspek (jumlah aspek variabel, nama aspek
open-vocabulary) menjadi vektor 5-dim via RATA-RATA lintas aspek
(`mean_positive_prob`, `mean_negative_prob`, `mean_confidence`). Ini secara
konseptual OPERASI YANG SAMA dengan varian "Mean" A2-IRM Fase 1 yang
terbukti empiris merusak sinyal ("averaging destroys exactly the polarity
contrast", Section V manuskrip A2-IRM) -- `std_positive_prob` cuma mitigasi
parsial (tahu "ada variasi", tidak tahu aspek mana yang bermasalah).

**Alasan desain saat ini** (bukan kelalaian, tapi keterbatasan format
tetap): Concat+Confidence Fase 1 bisa jadi vektor fixed-size karena K
(jumlah aspek) TETAP per domain (taksonomi keyword). PyABSA open-vocabulary
-- jumlah & nama aspek beda tiap review -- tidak punya "slot kolom tetap"
utk di-concat langsung dengan cara yang sama.

**Rencana perbaikan**: ganti agregasi rata-rata dengan sub-layer ATTENTION
yang memproses sequence per-aspek PyABSA (panjang variabel: skor+confidence
tiap aspek yang ditemukan) SEBELUM masuk ke token modalitas 'absa' utama --
bobot gabungan antar aspek DIPELAJARI (bukan rata-rata polos), lebih
konsisten dgn filosofi "Attention-Gated" arsitektur ini, dan berpotensi
menghindari jebakan averaging yang sudah terbukti di Fase 1.

**Urutan kerja yang disepakati**: JANGAN redesign dulu sebelum ada bukti
empiris. Selesaikan Stage 7 (analisis 150 run yang sudah ada) dulu --
terutama bandingkan skenario `agf_keyword` (arsitektur AGF + representasi
ABSA Concat+Confidence Fase 1 yang terbukti terbaik) vs `full_agf`/skenario
PyABSA lain (arsitektur AGF sama + representasi rata-rata 5-dim baru). Kalau
`agf_keyword` secara konsisten mengungguli skenario PyABSA lintas
domain/seed -- itu bukti kuat averaging memang merugikan di sini juga, dan
investasi redesign+re-run (~5,5 jam GPU tambahan) sepadan. Kalau selisihnya
kecil/tidak konsisten, redesign bisa ditunda jadi catatan future work di
manuskrip alih-alih dikerjakan sekarang.
