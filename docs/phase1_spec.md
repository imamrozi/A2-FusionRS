# Fase 1 — Spesifikasi Kerja: Reproduksi Dua Protokol

> **Status:** aktif · **Durasi target:** 2–3 minggu · **Berkas ini adalah kontrak kerja Fase 1.**
> Simpan di `docs/phase1_spec.md`. Claude Code wajib membacanya sebelum mengeksekusi step mana pun.

---

## 1. Tujuan

Menghasilkan **satu tabel** yang menunjukkan berapa banyak dari peningkatan akurasi yang
dilaporkan lineage Darraz et al. (2025) → A2-IRM → A2-FusionRS bertahan ketika review target
$d_{ui}$ dihapus dari fitur saat inferensi.

Fase 1 **bukan** fase pengembangan model. Tidak ada arsitektur baru yang dibangun di sini.
Satu-satunya keluaran adalah bukti kuantitatif yang menentukan apakah Fase 2 layak dijalankan.

### Definisi: review target

Untuk pasangan evaluasi $(u,i)$ dengan rating $r_{ui}$ dan teks review $d_{ui}$, **review target**
adalah $d_{ui}$ itu sendiri. Saat deployment, $d_{ui}$ belum ada — user belum mencoba item.
Setiap fitur yang diturunkan darinya tidak tersedia pada saat sistem perlu bekerja.

| Review | Boleh dipakai memprediksi $(u,i)$? | Alasan |
|---|:--:|---|
| $d_{ui}$ | tidak | Review target |
| $d_{uj}$, $j \neq i$ | ya | Riwayat user |
| $d_{vi}$, $v \neq u$ | ya | Reputasi item |

---

## 2. Empat arm

| Arm | Split | Review target | Fitur aspek untuk baris evaluasi | Status |
|---|---|---|---|---|
| **P0** | Tidak ada (train = predict, mengikuti Algorithm 1 Darraz) | Terlihat | Dari review target | perlu dijalankan |
| **P1** | Held-out user-based (yang sudah ada) | Terlihat | Dari review target | **sudah ada di ledger** |
| **P2** | Held-out, identik dengan P1 | Dihapus | Nilai netral (rata-rata train global) | perlu dijalankan |
| **P3** | Held-out, identik dengan P1 | Dihapus | Profil aspek item dari review train (agregat naif) | perlu dijalankan |

**Aturan tak boleh dilanggar:** P1, P2, P3 memakai **split yang sama persis**. Jangan pernah
me-regenerate split. Komparabilitas antar-arm adalah inti klaim ilmiahnya; P1 vs P2 harus berbeda
hanya pada satu faktor.

Split temporal masuk di Step 6 sebagai pemeriksaan robustness, **bukan** sebagai variabel utama.

---

## 3. Step-by-step

### Step 0 — Bekukan kondisi sekarang

**Tempat: Claude Code · Waktu: ½ hari**

1. `git tag -a v1.0-legacy-protocol -m "State producing ledger numbers (target-review protocol)"`, push ke origin.
2. Pindahkan pipeline protokol lama ke `src/legacy/`. **Hanya path dan import yang boleh berubah** — logika tidak boleh disentuh.
3. Tulis `tests/test_legacy_reproduction.py`: jalankan pipeline lama untuk domain `amazon` seed 42, assert RMSE = 0.6418 ± 5e-4 (acuan: `A2-FusionRS_results_ledger.md`).
4. Jalankan test, pastikan lulus.

Setelah step ini, `src/legacy/` diperlakukan **read-only**: hanya dipanggil, tidak pernah di-refactor.

**Selesai bila:** test lulus dan tag ada di remote.

---

### Step 1 — Diagnostik leakage

**Tempat: Claude Code · Waktu: 2–3 jam · Rasio nilai/usaha tertinggi di seluruh Fase 1**

Buat `scripts/audit_leakage.py` yang **membaca artefak run yang sudah ada** di
`checkpoints/results/` (tidak melatih ulang apa pun) dan menghasilkan
`reports/leakage_audit.md` berisi, per domain:

1. RMSE regresi linear `stars ~ sentiment_score` pada test set — mengukur berapa banyak rating yang dapat diprediksi dari sentimen review target **saja**.
2. RMSE model konstanta (global mean) sebagai pembanding.
3. Matriks korelasi Pearson **dan** mutual information antar `deepmf_preds`, `cbf_preds`, `sentiment_score`, `base_preds` pada test set.
4. VIF tiap stream.
5. Rank efektif matriks fitur fusi (SVD, ambang 99% varians terjelaskan).

Format tabel markdown, satu blok per domain, sertakan jumlah baris yang dipakai.

**Interpretasi yang diharapkan:**

| Temuan | Artinya |
|---|---|
| RMSE(`stars ~ sentiment_score`) mendekati 0,65 | Sentimen review target sendirian menjelaskan hampir seluruh performa |
| Korelasi antar-stream > 0,7 | Redundansi terkonfirmasi; matriks kovarians error mendekati singular |
| Rank efektif < 4 | Cross-attention tidak punya ruang kerja |

**Selesai bila:** `reports/leakage_audit.md` ada dan terbaca.

---

### Step 2 — Tinjau hasil diagnostik

**Tempat: sesi web · Waktu: 1 jam · JANGAN DILEWATI**

Bawa isi `reports/leakage_audit.md` ke sesi web. Yang diputuskan:

- Apakah angkanya cukup kuat menjadi bukti utama manuskrip.
- Bagaimana membingkainya secara **deskriptif dan netral** (deskripsikan prosedur dan konsekuensi terukurnya; jangan berteori tentang motif penulis asli).

Kalau RMSE(`stars ~ sentiment_score`) ternyata jauh dari 0,65, sebagian analisis perlu direvisi
**sebelum** menginvestasikan dua minggu berikutnya.

---

### Step 3 — Bangun `ReviewScope`

**Tempat: Claude Code · Waktu: 2–3 hari**

1. **`src/data/scope.py`** — kelas `ReviewScope` dengan method
   `visible_review_ids(user_id, item_id, timestamp) -> set[str]`.
   Dua implementasi: `TargetVisibleScope` (menyertakan review $(u,i)$) dan
   `HistoricalScope` (mengecualikannya).
   Ini **satu-satunya** tempat aturan visibilitas didefinisikan. Semua feature builder wajib
   melewatinya — alasan desain: aturan no-target-review terlalu mudah dilanggar tanpa sengaja
   kalau tersebar ke banyak modul.

2. **`src/features/provenance.py`** — setiap feature builder mencatat
   `dict[row_id -> set[review_id]]` yang dipakai membangun fitur baris itu.

3. **`src/eval/guards.py`** — `assert_no_target_leakage(provenance, eval_index)` yang gagal keras
   bila sebuah `row_id` muncul di dalam provenance-nya sendiri.

4. Panggil guard di akhir setiap pipeline, **sebagai bagian dari run**, bukan sebagai test terpisah.

5. **`configs/protocol/legacy_target_review.yaml`** dan **`configs/protocol/deployment_valid.yaml`** —
   protokol dipilih lewat config, **tidak pernah** lewat edit kode.

> **Jebakan implementasi — baca dua kali.**
> Untuk baris **training** $(u,i)$, profil aspek item $i$ juga harus mengecualikan review $(u,i)$.
> Kalau tidak, model belajar shortcut saat training yang tidak tersedia saat test, dan hasilnya
> menyesatkan. Terapkan leave-one-out di dalam train juga.

**Selesai bila:** run dengan `deployment_valid` lolos guard; run dengan `legacy` memicu guard, dan
bypass-nya eksplisit serta terbatas pada arm legacy saja.

---

### Step 4 — Jalankan P2 dan P3

**Tempat: Claude Code · Waktu: 4–6 hari (mayoritas waktu tunggu run)**

Konfigurasi: 3 domain × 5 seed (42, 123, 456, 789, 1011), **split identik dengan yang tersimpan**.

**P2 (neutral).** Untuk baris evaluasi, ganti `sentiment_score`, vektor keyword-ABSA, dan sekuens
aspek PyABSA dengan nilai rata-rata train global. CBF tetap memakai `sentiment_agg` level-item dari
train — ini agregat historis, sah.

**P3 (agregat naif).** Ganti fitur aspek baris evaluasi dengan profil aspek item $i$ yang dihitung
dari review train item tersebut: mean per-aspek, **tanpa** shrinkage, **tanpa** bobot importance
user. Sengaja naif — inilah yang harus dikalahkan Fase 2.

Model per arm: A2-IRM, A2-FusionRS, dan baseline reimpl Darraz.
Baseline CF murni (Item-KNN, SVD, NeuMF, DeepFM) **tidak perlu dijalankan ulang** karena tidak
menyentuh review — pakai angka ledger.

Keluaran ke `checkpoints/results/` dengan prefix `protocol_p2_*` dan `protocol_p3_*`, format YAML
sama seperti run sebelumnya.

---

### Step 5 — Reproduksi P0 (Darraz setia)

**Tempat: Claude Code · Waktu: 2–3 hari**

Buat `scripts/run_darraz_original_protocol.py` yang mengikuti Algorithm 1 paper sedekat mungkin:

- Tanpa train/test split pada tahap fusi.
- Fit NMF + DecisionTreeRegressor pada seluruh dataset.
- Prediksi pada seluruh dataset yang sama.
- Sentimen global BERT masuk ke CF, CBF, **dan** fusion (topologi asli — lihat §3.4 dan §3.4.2 paper).
- Laporkan RMSE, MAE, NDCG.

Dokumentasikan setiap deviasi yang terpaksa dilakukan di `reports/darraz_reproduction.md`.

**Kenapa wajib:** tanpa P0, reviewer akan menduga reimplementasi Anda yang lemah, bukan protokolnya
yang bermasalah. P0 menutup pintu itu. Tujuannya bukan performa, melainkan menunjukkan bahwa
reimplementasi mampu mereproduksi **orde angka** yang dilaporkan (RMSE 0,10–0,55).

---

### Step 6 — Robustness temporal

**Tempat: Claude Code · Waktu: 2 hari**

Satu run P3 dengan split leave-one-out temporal, domain Amazon saja, 3 seed. Cukup untuk menyatakan
di manuskrip bahwa temuan tidak bergantung pada skema split.

---

### Step 7 — Gerbang keputusan

**Tempat: sesi web**

Bawa tabel lengkap. Yang dinilai:

1. Apakah penurunan P1 → P2 cukup besar untuk menopang klaim protokol.
2. Apakah P3 menyisakan headroom yang layak dikejar Fase 2.
3. Rumusan presisi untuk kontribusi 1 manuskrip.

---

## 4. Tabel keluaran Fase 1

Deliverable tunggal Fase 1. Kandidat kuat untuk Tabel 1 atau 2 manuskrip.

**RMSE — domain Amazon** (ulangi untuk Restaurant dan Hotel):

| Model | P0 (protokol asli) | P1 (held-out, review target) | P2 (tanpa review target) | P3 (agregat historis naif) |
|---|---:|---:|---:|---:|
| Darraz reimpl | | | | |
| A2-IRM | — | 0,6517 | | |
| A2-FusionRS | — | 0,6418 | | |
| NeuMF (referensi) | — | 1,1528 | 1,1528 | 1,1528 |
| Global Mean | — | 1,2143 | 1,2143 | 1,2143 |

Sertakan SD lintas 5 seed dan uji Wilcoxon berpasangan P1 vs P2 dan P2 vs P3.
Selalu laporkan Δ-RMSE bersama p-value — jangan pernah menyimpulkan dari p saja.

---

## 5. Prediksi yang dapat difalsifikasi

Angka di bawah adalah **perkiraan, bukan target**. Gunakan sebagai alat diagnosis.

| Kolom | Perkiraan | Kalau meleset jauh |
|---|---|---|
| P0 | 0,10–0,55 | Reimplementasi belum setia — perbaiki sebelum lanjut |
| P2 | 0,95–1,10 | Kalau tetap ≈0,65, ada jalur leakage lain yang belum ditemukan |
| P3 | 0,90–1,05 | Kalau ≈ P2, agregat naif tak berguna — justru memperkuat motivasi Fase 2 |

---

## 6. Invarian yang berlaku sepanjang Fase 1

Salin ke `CLAUDE.md`.

1. Fitur untuk memprediksi $(u,i)$ tidak boleh diturunkan dari review $(u,i)$, kecuali di bawah `protocol: legacy`.
2. `src/legacy/` tidak boleh diubah.
3. Split tidak boleh di-regenerate. P1, P2, P3 memakai split yang identik.
4. Setiap angka baru masuk ke results ledger dengan prefix berkas sumbernya.
5. Protokol selalu dipilih lewat config, tidak pernah lewat edit kode.
6. Leave-one-out berlaku juga di dalam split train, bukan hanya di test.
7. Tidak ada angka yang ditulis ke manuskrip yang tidak ada di results ledger.
8. Setiap run wajib mem-persist prediksi semua stream per-baris (bukan hanya prediksi fusion akhir) ke `checkpoints/results/` — termasuk stream yang sebelumnya hanya hidup in-memory (mis. `deepmf_preds`, `cbf_preds`). Dipicu oleh temuan `reports/leakage_audit.md`: diagnostik leakage Step 1 tidak bisa menghitung korelasi/VIF/rank efektif penuh karena kedua stream itu tak pernah dipersist oleh run manapun, lokal maupun Colab.
9. Setiap kebijakan fallback/default (nilai pengganti saat data tidak cukup, atau protokol menonaktifkan suatu fitur — mis. rata-rata global di P2, profil item di P3 saat item tanpa aspek terdeteksi, toleransi numerik di test regresi) harus dinyatakan **eksplisit** di kode/komentar berikut alasannya — tidak pernah implisit lewat default parameter yang tak dijelaskan. Kolom diagnostik (mis. `aspect_fallback`/`n_shared_aspects`) juga harus dipisah tegas dari nilai fitur yang sesungguhnya dipakai model, tidak boleh diam-diam ikut mengubah fitur kecuali didokumentasikan sebagai keputusan sadar.

---

## 7. Pembagian tempat kerja

| Step | Tempat |
|---|---|
| 0, 1, 3, 4, 5, 6 | **Claude Code** |
| 2, 7 | **Sesi web** |

Antarmuka antara keduanya adalah `A2-FusionRS_results_ledger.md`: Claude Code **menulis angka**,
sesi web **membaca ledger dan memutuskan klaim**. Arah ini tidak pernah dibalik.

---

## 8. Di luar cakupan Fase 1

Supaya tidak ada scope creep:

- Model aspect-bridge dengan dekomposisi importance × quality → **Fase 2**
- Gate berbasis reliabilitas, DeepMF/CBF sebagai vektor laten → **Fase 2**
- Baseline review-aware (DeepCoNN, NARRE, ANR, DAML) → **Fase 3**
- Evaluasi ter-slice dan eksperimen redundansi terkontrol → **Fase 4**
- Perbaikan protokol ranking (100-negatif / full-catalog) → **Fase 4**
- Penulisan manuskrip → **Fase 5**

Satu pengecualian yang boleh dikerjakan kapan saja karena murah dan berdiri sendiri:
ablasi DecisionTree dengan vs tanpa komponen NMF (`nmf_components: 3` atas 3 fitur mentah).
