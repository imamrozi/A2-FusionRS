# Gerbang 1–3: jalur konkret agar arsitektur usulan mengalahkan A2-IRM

**Tanggal:** 2026-08-08
**Konteks:** faktorial Tahap 7 (60 run, test) menunjukkan arsitektur bersih kalah
3/3 dari A2-IRM. Tiga gerbang lokal berikut mendiagnosis sebabnya dan menguji
perbaikan **sebelum** membakar jam Colab.

Semua angka di bawah adalah **probe linier (Ridge) pada kanal sentimen saja**
(tanpa DeepMF/CBF/fusi), subsample 5000 train / 3000 test per domain. Yang sah
disimpulkan hanya **peringkat antar-representasi**, bukan RMSE end-to-end.

Reproduksi:
- `scripts/gate_pyabsa_extraction_sabert_scoring.py` → `reports/gate_pyabsaext_sabert.csv`
- `scripts/retest_structure_recovery.py` → `reports/structure_recovery_retest.csv`
- `scripts/gate3_global_sentiment_token.py` → `reports/gate3_global_token.csv`

---

## Ringkasan tiga gerbang

| Gerbang | Pertanyaan | Putusan |
|---|---|---|
| 1 | Setelah scorer disetarakan, apakah ekstraksi PyABSA ≥ leksikon keyword? | **LOLOS 3/3** (−4,3% / −0,3% / −0,4%), tapi tipis di 2 domain |
| 2 | Bisakah struktur posisi-tetap dipulihkan dari aspek open-vocabulary? | **TIDAK** — clustering & bag-of-aspects tidak memberi keuntungan |
| 3 | Dari mana sebenarnya "keunggulan struktur" itu? | **Bukan struktur** — melainkan akses ke skor review global |

---

## 1. Perbaikan arsitektur: kunci scorer, ganti ekstraksinya

Diagnosis `pyabsa_vs_keyword_diagnosis.md` menunjukkan gap PyABSA-vs-keyword
berasal dari **supervisi scorer**: keyword-ABSA memakai `GlobalSentimentBERT`
yang di-fine-tune pada label turunan `stars` domain sendiri, PyABSA memakai
checkpoint generik. Perbaikannya: PyABSA untuk **ekstraksi** (cakupan 77–80%
restaurant/e-commerce, 70% hotel), SA-BERT per-domain untuk **skoring** —
scorer yang sama persis dengan A2-IRM. Tetap 1× ABSA; leksikon statis dan
NMF+DT tetap dihapus.

Gerbang 1 (kedua sisi order-statistics 9-dim, struktur dibuang dari keduanya):

| Domain | usulan | keyword_rich9 | selisih |
|---|---|---|---|
| Restaurant | 0,8043 | 0,8408 | −4,3% |
| E-commerce | 0,8084 | 0,8107 | −0,3% |
| Hotel | 0,7207 | 0,7239 | −0,4% |

Ekstraksi PyABSA tidak kalah. Tapi hanya restaurant yang marginnya nyata.

## 2. Cacat encoding pada gerbang 2 — dikoreksi

Gerbang 2 awalnya menyimpulkan clustering "katastrofik" (0,9720 / 1,0157 /
0,8131). **Itu artefak cacat encoding saya sendiri**: cluster tanpa aspek diisi
`0.0`, padahal pada skala sentimen [0,1] nilai itu berarti "sangat negatif",
bukan "tidak ada data". Keyword tidak terkena karena aspek tak-match diberi skor
fallback review.

Setelah imputasi yang benar (`retest_structure_recovery.py`, skema `v1_impute`):

| Representasi | Restaurant | E-commerce | Hotel |
|---|---|---|---|
| clustered [v0_zero] (angka lama) | 0,9720 | 1,0157 | 0,8131 |
| clustered [v1_impute] (dikoreksi) | 0,8082 | 0,8098 | 0,7328 |
| usulan_rich9 (tanpa struktur) | 0,8043 | 0,8084 | 0,7207 |

`v0_zero` mereproduksi angka lama persis (kontrol kebenaran lulus). Kesimpulan
yang bertahan: clustering **netral**, bukan katastrofik — tapi tetap tidak
memberi keuntungan atas order-statistics.

## 3. Temuan utama: "keunggulan struktur" itu bukan struktur

`keyword_concat_conf` unggul 5,7–11,6% atas bentuk order-statistics. Label
"nilai struktur posisi-tetap" untuk selisih itu **salah**.

Mekanisme sebenarnya: pada keyword-ABSA, aspek yang tidak match diisi **skor
sentimen seluruh review**. Karena banyak review hanya mencocokkan sebagian
aspek, mayoritas kolom `keyword_concat_conf` sebenarnya memuat skor review
global — model mendapatkannya cuma-cuma. Order-statistics menghancurkan akses
itu.

Bukti pendukung: kolom confidence hampir tidak berkontribusi (0,0% / 0,1% /
0,6%), jadi keunggulannya murni dari nilai skor, bukan bobot bukti.

Tambahkan skor global sebagai fitur eksplisit dan selisihnya bukan hanya
hilang — ia berbalik:

| Representasi | Restaurant | E-commerce | Hotel |
|---|---|---|---|
| kw_concat (A2-IRM apa adanya) | 0,7468 | 0,7164 | 0,6828 |
| kw_rich9 (order-stats) | 0,8408 | 0,8107 | 0,7239 |
| kw_rich9 **+ global** | 0,7332 | 0,7073 | 0,6777 |
| kw_concat **+ global** | 0,7343 | 0,7080 | 0,6794 |
| **usulan_rich9 + global** | **0,7282** | **0,6996** | 0,6832 |
| [batas atas] usulan + kw_concat + global | 0,7192 | 0,6970 | 0,6674 |

---

## 4. Pembacaan yang jujur

**Perbandingan yang sah adalah `usulan + global` vs `kw_concat + global`**, bukan
vs `kw_concat` polos:

| Domain | usulan+global | pembanding ADIL | selisih adil | (vs naif) |
|---|---|---|---|---|
| Restaurant | 0,7282 | 0,7343 | **−0,8%** | (−2,5%) |
| E-commerce | 0,6996 | 0,7080 | **−1,2%** | (−2,4%) |
| Hotel | 0,6832 | 0,6794 | +0,6% | (+0,1%) |

**Token global menolong KEDUA cabang.** Ia bukan keunggulan PyABSA, melainkan
fitur yang hilang dari representasi gaya A2-IRM. Melaporkan selisih terhadap
pembanding naif sebagai kontribusi PyABSA akan **melebih-lebihkan ~2×**, dan
manuskrip wajib memakai pembanding adil.

Setelah disetarakan, kontribusi ekstraksi PyABSA adalah **~1% di 2/3 domain,
dengan hotel kalah tipis 0,6%**. Nyata, konsisten arahnya, tapi kecil.

Varian `usulan + kw_concat + global` adalah yang terbaik di 3/3, tapi itu
memakai **dua sistem ABSA sekaligus** — arsitektur yang sudah ditolak. Dicatat
sebagai batas atas informasi saja, tidak boleh diklaim sebagai model usulan.

## 5. Yang belum diuji

Semua di atas adalah probe **linier**. `AspectSequencePooling` di AGF dapat
mempelajari bobot per-identitas-aspek dan interaksi yang Ridge atas rata-rata
cluster tidak bisa nyatakan — itu justru mekanisme yang diklaim paper. Bukti
sebelumnya tidak menguntungkan (sel D vs D₀ hanya 0,5% / 0,8% / 0,2%), **tapi
itu diukur di atas skor PyABSA yang buruk**; dengan skor SA-BERT bahannya jauh
lebih baik. Ini pertanyaan terbuka yang sah.

## 6. Rekomendasi

Lanjut ke Colab, dengan **dua perubahan wajib** pada rancangan:

1. **Tambahkan token sentimen global** ke arsitektur AGF (skor SA-BERT
   level-review). Ini satu scorer pada dua granularitas — tetap 1× ABSA, dan
   secara arsitektural wajar sebagai token tersendiri.
2. **Tambahkan sel faktorial `kw_concat + global`** sebagai pembanding adil.
   Tanpa sel ini, klaim keunggulan tidak dapat dipertanggungjawabkan.

Ekspektasi yang realistis: keunggulan ~1% pada kanal sentimen bisa saja **tidak
bertahan** setelah digabung DeepMF+CBF yang mendominasi varians. Kalau target
mutlaknya adalah mengalahkan A2-IRM pada RMSE rata-rata, peluangnya ada tapi
tipis. Jalur yang lebih kuat mungkin **robustness** — sel B menunjukkan AGF
memangkas SD lintas seed dari 0,1023 → 0,0481 di e-commerce dan tidak kolaps
pada split sulit (seed 1011: A2-IRM 0,8961 vs AGF 0,7758).
