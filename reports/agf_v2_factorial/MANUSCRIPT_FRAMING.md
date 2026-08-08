# Kerangka manuskrip A2-FusionRS — berbasis data yang SUDAH ADA

**Tanggal:** 2026-08-08
**Status:** tidak memerlukan run baru. Seluruh angka berasal dari
`checkpoints/results` (A2-IRM), `checkpoints/results_phase2_clean/test` (v1),
dan `checkpoints/results_phase2_clean_v2/test` (v2).

## Posisi dalam program riset

Darraz 2025 (baseline arsitektur) → **A2-IRM** (pengaruh ABSA pada hybrid) →
**A2-FusionRS** (pengaruh fusi adaptif). A2-IRM adalah tahapan penulis
sendiri, bukan baseline pihak ketiga; membandingkan terhadapnya adalah
kemajuan program riset, bukan pelemahan baseline.

## Definisi model usulan (paling defensible)

**A2-FusionRS = komponen sentimen A2-IRM (keyword-ABSA + SA-BERT yang SAMA)
+ Attention-Gated Fusion dengan token sentimen global, menggantikan
NMF+DecisionTree.**

Kenapa definisi ini yang dipilih:
- **Satu faktor berubah** (mekanisme fusi) → kontribusinya terisolasi bersih
- Scorer identik di kedua sisi → tidak ada celah tuduhan baseline tidak adil
- PyABSA menjadi temuan NEGATIF di ablasi, yang menaikkan kredibilitas

## Hasil utama (mean 5 seed, TEST)

| Domain | A2-IRM | AGF saja | **AGF+token (usulan)** | usulan vs A2-IRM |
|---|---|---|---|---|
| Restaurant | 0,6821 | 0,7114 | 0,7054 | **+3,41%** (kalah, 5/5 sig) |
| E-commerce | 0,7141 | 0,6905 | **0,6856** | **−3,99%** (menang, 4/5) |
| Hotel | 0,6279 | 0,6422 | **0,6197** | **−1,29%** (menang, 3/5) |

Sel `AGF saja` = v1 `cellB`; `AGF+token` = v2 `cellBfair`. Konfigurasi
keduanya **diverifikasi identik** (`d=64, n_heads=2, epochs=30,
weight_decay=0`, `residual_base=user_item_bias`, `representation=asymmetric`,
`input_standardize=True`) — berbeda HANYA pada token sentimen global.
Jadi B→B′ mengisolasi efek token secara sempurna.

## Temuan mekanistik utama: efek token global berbeda tajam antar cabang

| Cabang | efek token global (R / E / H) |
|---|---|
| keyword (B→B′) | −0,84% / −0,71% / −3,50% |
| PyABSA (F−→F) | −8,78% / −15,50% / −5,93% |

Ini konfirmasi kuantitatif Gerbang-3 di DUA cabang berbeda:
keyword-concat sudah memperoleh sinyal review-level secara implisit lewat
pengisian aspek tak-match, sehingga menambahkannya eksplisit hanya membantu
sedikit; agregasi PyABSA menghancurkan akses itu, sehingga memulihkannya
membantu besar. Prediksi teoretis yang terbukti kuantitatif — bukti kuat.

**Catatan spesifik hotel:** AGF sendirian KALAH (0,6422 vs 0,6279), tapi
AGF+token MENANG (0,6197). Token itulah yang membuat AGF layak di domain itu.

## Kalibrasi klaim (koreksi terhadap draf sebelumnya)

Menyebut "tokenisasi multi-granularitas" sebagai kontribusi arsitektural
utama TERLALU KUAT untuk cabang keyword: kontribusinya hanya 0,7–3,5%, dan
sebagian besar kemenangan e-commerce sebenarnya dari AGF itu sendiri
(−3,3%). Klaim harus dinyatakan per-domain, bukan digeneralisasi.

Posisi jujur: **AGF menang 2/3 domain; token global menentukan di hotel dan
kecil di dua domain lain; restaurant kalah dengan sebab teridentifikasi.**

## Restaurant kalah — laporkan dengan penjelasan mekanistik

45% defisit berasal dari rating bintang 1 (hanya 5,7% data): MSE tree 1,2395
vs AGF 1,4949. AGF sistematis memprediksi terlalu tinggi di rating rendah
(+0,126 dibanding tree) — perilaku khas jaringan ber-loss MSE yang menyusut
ke mean, sedangkan tree membuat partisi tajam. Restaurant paling terdampak
karena mean rating-nya terendah (3,785), jadi porsi rating rendahnya terbesar.

## Ablasi temuan negatif (memperkuat kredibilitas)

- Ekstraksi PyABSA (pembanding adil E−B′): +0,48% / −0,45% / +1,47% —
  tidak terbukti menguntungkan meski supervisi scorer sudah disetarakan
- Sequence identitas aspek (F−E): −0,10% / +1,37% / +0,25% — praktis nol,
  bertahan walau vocab dinaikkan 500→2000
- Tuning kapasitas AGF (54 run dev): pemenang = konfigurasi DEFAULT →
  bottleneck bukan kapasitas model
- Sinyal residual: temporal/aktivitas/panjang teks |r|<0,05; struktur
  residual per-user & per-item ±0,5% → residual pada dasarnya derau

## Batasan yang WAJIB dinyatakan

1. Restaurant kalah signifikan (5/5 seed) — jangan disamarkan.
2. Klaim robustness rentan: audit kalibrasi menunjukkan kolaps A2-IRM di
   seed 1011 sebagian besar kegagalan kalibrasi, bukan instabilitas
   struktural (lihat `ADDENDUM_calibration_audit.md`). Kalau robustness jadi
   kontribusi utama, perlu 10–15 seed, bukan 5.
3. Skor SA-BERT tersaturasi biner (81,6% di luar [0,01, 0,99]); bintang 1
   vs 2 dan 4 vs 5 praktis tak terbedakan; bintang 3 bimodal karena rating
   netral dibuang dari pelatihan SA. Ini batas informasi, bukan batas fusi.
