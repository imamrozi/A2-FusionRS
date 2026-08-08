# Faktorial v2 di TEST — hasil & pembacaan jujur

**Tanggal:** 2026-08-08
**Run:** 60 run (`checkpoints/results_phase2_clean_v2/test/`), 4 sel × 3 domain × 5 seed.
**Konfigurasi:** pemenang seleksi DEV `d=64, weight_decay=0.0`
(`reports/agf_v2_selection/`, commit `7581c1e` — dikunci SEBELUM test disentuh).

**Integritas terverifikasi:** 60/60 YAML, 60/60 predictions, `FAILED_RUNS.txt`
kosong, semua `stage=confirm` + `eval_split=test` + `agf_d=64` +
`weight_decay=0.0`, 5 seed unik per sel×domain. Nol pelanggaran.

Reproduksi: `python scripts/analyze_agf_v2_factorial.py`,
`python scripts/analyze_agf_robustness.py --version v2`

---

## 1. RMSE per sel (mean 5 seed, TEST)

| Sel | Deskripsi | Restaurant | E-commerce | Hotel |
|---|---|---|---|---|
| A  | keyword + tree (A2-IRM) | **0,6821** | 0,7141 | 0,6279 |
| B′ | keyword + AGF + token global | 0,7054 | 0,6856 | **0,6197** |
| E  | PyABSA + SA-BERT + AGF + global | 0,7088 | **0,6825** | 0,6288 |
| F  | + sequence identitas aspek (TARGET) | 0,7080 | 0,6919 | 0,6304 |
| F− | F tanpa token global (ablasi) | 0,7762 | 0,8188 | 0,6701 |

## 2. Estimasi efek

| Efek | Restaurant | E-commerce | Hotel |
|---|---|---|---|
| **EKSTRAKSI PyABSA (adil, E−B′)** | +0,48% (3/5) | −0,45% (4/5) | +1,47% (5/5) |
| representasi sequence (F−E) | −0,10% (4/5) | +1,37% (4/5) | +0,25% (2/5) |
| **token sentimen global (F−F−)** | **−8,78%** (5/5) | **−15,50%** (5/5) | **−5,93%** (5/5) |
| TOTAL vs A2-IRM (F−A) | +3,80% (5/5) | −3,12% (4/5) | +0,41% (**0/5**, p=0,987) |

---

## 3. Pembacaan jujur — hipotesis utama TIDAK terbukti

**Arsitektur target (F) tidak mengalahkan A2-IRM secara konsisten.** Menang di
e-commerce (−3,12%), kalah signifikan di restaurant (+3,80%, 5/5), dan di hotel
benar-benar seri — p=0,987, 0/5 seed signifikan, bukan sekadar "kalah tipis".

**Ekstraksi PyABSA tidak berkontribusi positif.** Pada pembanding ADIL (E−B′,
scorer & token global dikonstankan), PyABSA hanya menang di e-commerce
(−0,45%) dan kalah di restaurant (+0,48%) serta hotel (+1,47%, 5/5 signifikan).
Probe linier Gerbang-1 memprediksi menang 2/3; pada pipeline penuh hasilnya
berbalik.

**Representasi sequence praktis nol.** −0,10% / +1,37% / +0,25% — konsisten
dengan temuan v1 (D vs D₀ hanya 0,5%/0,8%/0,2%). Menambah identitas aspek
lewat AspectSequencePooling tidak menghasilkan nilai yang terukur, bahkan
setelah vocab dinaikkan 500→2000 dan scorer diperbaiki.

### Temuan terpenting: B′ mengungguli F di KETIGA domain

| Domain | B′ vs A2-IRM | F vs A2-IRM |
|---|---|---|
| Restaurant | −3,41% (kalah) | −3,80% (kalah) |
| E-commerce | **−3,99% (menang)** | −3,12% (menang) |
| Hotel | **−1,29% (menang, 3/5)** | +0,41% (seri, 0/5) |

`B′` = keyword-ABSA + AGF + token global, **tanpa PyABSA sama sekali**.
Menambahkan PyABSA-ekstraksi dan sequence pooling di atasnya bukan hanya tidak
menolong — ia **menurunkan** performa di ketiga domain. Konfigurasi terbaik yang
ditemukan eksperimen ini justru yang TIDAK memakai PyABSA.

### Driver sebenarnya: token sentimen global

−5,9% s/d −15,5%, semua 5/5 signifikan — jauh melampaui efek ekstraksi maupun
sequence. Ini mengonfirmasi Gerbang-3: yang selama ini dikira "nilai struktur
posisi-tetap" sebenarnya akses ke skor review global, dan mengembalikannya
sebagai token eksplisit adalah satu-satunya perubahan yang benar-benar bekerja.

---

## 4. Robustness — bagian yang justru paling kuat

E-commerce (satu-satunya domain dengan instabilitas nyata):

| Sel | SD | worst-case | z_LOO seed 1011 |
|---|---|---|---|
| A (A2-IRM) | 0,1023 | 0,8961 | **17,8** |
| B′ | 0,0438 | 0,7626 | 10,4 |
| E | 0,0395 | 0,7525 | 14,1 |
| F | **0,0357** | **0,7491** | 3,9 |

Semua varian AGF memangkas SD 2–3× dan worst-case ~13–16% dibanding A2-IRM.
`F` sedikit lebih stabil dari `B′` — jadi PyABSA+sequence memberi kontribusi
kecil pada stabilitas meski merugikan akurasi.

Di hotel & restaurant SD semua model kecil (0,0015–0,0042): **tidak ada klaim
robustness yang bisa dibuat di sana**. Keunggulan robustness bersifat
spesifik-domain, hanya di mana baseline memang tidak stabil.

**Batasan:** 5 seed terlalu tipis untuk estimasi varians. Kalau robustness jadi
kontribusi utama manuskrip, tambah ke 10–15 seed sebelum submit.

---

## 5. Implikasi untuk manuskrip

Yang bisa diklaim jujur:
1. **Token sentimen global** adalah perbaikan nyata, besar, konsisten 3/3
   domain — dan itu temuan diagnostik yang bisa dipertanggungjawabkan.
2. **AGF memberi robustness** yang nyata di domain tidak stabil (worst-case
   e-commerce turun 13–16%), meski bukan akurasi rata-rata yang lebih baik.
3. **Penggantian keyword-ABSA dengan PyABSA tidak terbukti menguntungkan**
   untuk tugas prediksi rating, bahkan setelah supervisi scorer disetarakan.
   Ini temuan negatif yang informatif, bukan kegagalan eksperimen.

Yang TIDAK boleh diklaim:
- "Arsitektur usulan mengalahkan A2-IRM" — tidak benar di 2/3 domain.
- "PyABSA meningkatkan akurasi" — pembanding adil menunjukkan sebaliknya.
- `F − A` sebagai kontribusi PyABSA — mayoritas selisihnya berasal dari token
  global yang menolong kedua cabang.
