# Mengapa PyABSA kalah dari keyword-ABSA? — diagnosis kanal sentimen

**Tanggal:** 2026-08-07
**Pemicu:** hasil faktorial Tahap 7 (60 run, TEST SET) menunjukkan penggantian
keyword-ABSA → PyABSA MERUGIKAN di 3/3 domain.
**Pertanyaan yang dijawab:** *"apakah PyABSA yang kemudian di-cluster secara
akademik lebih relevan dan lebih baik (justified) daripada ABSA fixed-taxonomy?"*
**Jawaban singkat: TIDAK.** Clustering tidak menolong — secara empiris justru
memperburuk — karena defisitnya ada pada **kualitas skor sentimen**, bukan pada
bentuk representasi. Rinciannya di bawah.

Reproduksi: `venv/Scripts/python.exe scripts/diagnose_sentiment_signal_quality.py`
→ `reports/sentiment_signal_quality.csv`

---

## 1. Dua hipotesis bersaing

| | Klaim | Prediksi bila benar |
|---|---|---|
| **H-struktur** | PyABSA open-vocabulary kehilangan struktur posisi-tetap milik taksonomi keyword; agregasi/pooling merusak sinyal | memetakan PyABSA ke taksonomi tetap (clustering) **menutup gap** |
| **H-supervisi** | skor keyword-ABSA berasal dari `GlobalSentimentBERT` yang **di-fine-tune pada label turunan `stars` domain itu sendiri**; PyABSA memakai checkpoint pretrained generik tanpa supervisi rating | clustering **tidak menolong**; defisit ada di kualitas skor |

Basis kode untuk H-supervisi:
[sentiment_bert.py:47](src/baseline/sentiment_bert.py#L47) `derive_sentiment_label()`
(`stars >= 4` → positif, `stars <= 2` → negatif), dilatih di
[run_baseline.py:116](run_baseline.py#L116) atas train+val. Keyword-ABSA memanggil
skorer yang sama lewat `self.sentiment_model.predict_proba()`
([absa_bert.py:203](src/a2fusionrs/absa_bert.py#L203)).

> **Tidak ada kebocoran test.** SA-BERT dilatih hanya pada train+val
> ([run_baseline.py:113](run_baseline.py#L113)). A2-IRM tetap sah secara
> metodologis. Yang ditunjukkan di sini adalah bahwa perbandingan
> "keyword-ABSA vs PyABSA" **terkonfound oleh supervisi domain**, sehingga
> tidak boleh dibaca sebagai perbandingan metode ABSA an sich.

---

## 2. Desain uji

Daya prediksi rating dari **kanal sentimen saja** — tanpa DeepMF, tanpa CBF,
tanpa fusi. Regressor sama (Ridge α=1, fitur distandarkan), split sama, hanya
**representasi** yang berubah. Kosakata aspek & clustering dibangun **hanya dari
review train**.

## 3. Hasil (RMSE test, kanal sentimen saja — makin kecil makin baik)

| # | Representasi | dim | Restaurant | E-commerce | Hotel |
|---|---|---|---|---|---|
| 1 | keyword_concat_conf (A2-IRM, SA-BERT tersupervisi) | 8/10/12 | **0,7416** | **0,7071** | **0,6856** |
| 2 | sabert_global_1dim (satu skalar, tersupervisi) | 1 | 0,7492 | 0,7112 | 0,7128 |
| 2b | pyabsa_meanpos_1dim (satu skalar, tanpa supervisi) | 1 | 0,8228 | 0,9795 | 0,7178 |
| 3 | pyabsa_rich9 (dipakai sel C & D₀) | 9 | 0,7931 | 0,8916 | 0,6953 |
| 3b | pyabsa_clustered (taksonomi terinduksi, K=\|keyword\|) | 8/10/12 | 0,9763 | 1,0831 | 0,8030 |
| 4 | pyabsa_bag_top200 (identitas aspek penuh, posisi tetap) | 400 | 0,9720 | 1,0766 | 0,7985 |

### 3.1 Uji terbersih: 1-dim vs 1-dim (baris 2 vs 2b)

Struktur **dikonstankan sepenuhnya** — keduanya satu skalar per review. Satu-satunya
perbedaan adalah scorer-nya. Ini mengisolasi efek supervisi domain:

| Domain | SA-BERT (tersupervisi) | PyABSA (generik) | selisih |
|---|---|---|---|
| Restaurant | 0,7492 | 0,8228 | **+9,8%** |
| E-commerce | 0,7112 | 0,9795 | **+37,7%** |
| Hotel | 0,7128 | 0,7178 | **+0,7%** |

**Urutan besar-kecilnya cocok sempurna (3/3) dengan urutan gap faktorial**
(D−A: +12,2% / +21,1% / +3,9%): E-commerce terbesar, Restaurant menengah,
Hotel terkecil. Korespondensi peringkat lintas tiga domain ini sulit dijelaskan
oleh sebab lain selain kualitas scorer.

### 3.2 Jawaban langsung soal clustering (baris 3b)

Clustering **benar-benar dijalankan** (induksi kategori aspek gaya LSA:
matriks kejadian istilah×review train → TruncatedSVD → KMeans, K disamakan
dengan jumlah aspek taksonomi keyword), bukan sekadar diargumentasikan.

Hasilnya **lebih buruk daripada agregasi rich9 biasa di 3/3 domain**
(0,9763 vs 0,7931 · 1,0831 vs 0,8916 · 0,8030 vs 0,6953), dan jauh di bawah
keyword. Memulihkan struktur posisi-tetap **tidak** memulihkan performa.

### 3.3 Mengapa H-struktur tertolak

- Baris 2 < baris 3 di 2/3 domain: **satu skalar** tersupervisi mengalahkan
  **sembilan dimensi** PyABSA → defisit bukan soal dimensi atau struktur.
- Baris 4 mempertahankan identitas aspek **penuh** pada posisi tetap — persis
  keunggulan struktural yang dihipotesiskan — dan justru performa **terburuk**.
  Karena setiap skema clustering adalah penggabungan **linier** kolom-kolom ini,
  baris 4 membatasi **informasi** yang tersedia bagi skema clustering mana pun.
- Sel D (sequence, identitas aspek dipertahankan via embedding) hanya
  memperbaiki D₀ sebesar 0,5%/0,8%/0,2% — konsisten: memulihkan struktur
  memberi keuntungan marginal, sementara gap-nya 12–35%.

**Caveat jujur:** baris 4 adalah batas atas dalam hal *informasi*, bukan dalam hal
*varians estimasi* — clustering punya parameter lebih sedikit sehingga bisa saja
mengungguli baris 4. Karena itu argumennya tidak bersandar pada baris 4 saja,
melainkan pada baris 3b (clustering nyata, tetap kalah) dan baris 2-vs-2b
(struktur dikonstankan, gap tetap muncul).

---

## 4. Konsekuensi

1. **Untuk pertanyaan user:** PyABSA + clustering **tidak** lebih baik. Dari sisi
   *konstruksi taksonomi* clustering memang lebih *justified* daripada leksikon
   manual (dan menjawab keterbatasan "manually curated" yang sudah diakui di
   manuskrip A2-IRM), tapi keunggulan metodologis itu **tidak berubah menjadi
   keunggulan akurasi**, karena bottleneck-nya ada di tempat lain.

2. **Perbandingan Tahap 7 harus dibingkai ulang.** Sel A/B vs C/D₀/D bukan
   "keyword-ABSA vs PyABSA" melainkan **"skorer tersupervisi-domain vs skorer
   pretrained generik"**, dengan metode ekstraksi aspek ikut berubah. Ini
   **wajib** dinyatakan di manuskrip; tanpa itu, klaim "PyABSA lebih buruk dari
   keyword-ABSA" menyesatkan.

3. **Uji yang adil belum dilakukan** dan sekarang jelas bentuknya: fine-tune
   scorer PyABSA (atau kalibrasi skor per-aspeknya) pada label turunan rating
   domain, sehingga kedua cabang punya tingkat supervisi yang sama. Baru setelah
   itu selisihnya bisa dibaca sebagai perbandingan metode ABSA.

4. **Catatan untuk manuskrip A2-IRM:** naskah menyebut BERT "fine-tuned per domain
   on the training split" ([draft:82](manuscript/A2-IRM_manuscript_draft.md#L82))
   tapi **tidak menyatakan bahwa labelnya diturunkan dari `stars`**. Sebaiknya
   diungkapkan eksplisit — ini memengaruhi cara pembaca menafsirkan kontribusi
   kanal ABSA.
