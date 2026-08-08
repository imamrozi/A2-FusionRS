# Addendum: audit kalibrasi — klaim mana yang bertahan?

**Tanggal:** 2026-08-08
**Pemicu:** pertanyaan metodologis apakah menambah kalibrasi isotonic punya
landasan akademik yang cukup untuk level doktoral.

Jawabannya membalikkan sebagian kesimpulan di `README.md`. Addendum ini
**wajib dibaca bersama** dokumen itu.

Reproduksi: seluruh angka di bawah dihitung dari `predictions_*.csv` yang
sudah ada (tanpa run baru), dengan protokol fit-di-separuh / nilai-di-separuh
(5 seed × 10 pengulangan split acak) untuk mengestimasi transfer kalibrasi
secara realistis, bukan batas atas optimistik.

---

## 1. Kenapa audit ini perlu

Kalibrasi isotonic sempat diusulkan sebagai komponen "sesudah AGF" karena
memberi perbaikan terukur pada model usulan. Dua keberatan mematahkannya:

**Landasan literatur lemah.** Kalibrasi isotonic mapan untuk **probabilitas
klasifikasi** (Zadrozny & Elkan 2002; Niculescu-Mizil & Caruana 2005; Guo et
al. 2017) — memastikan skor 0,8 berarti 80% benar. Untuk **regresi rating**
ia bukan komponen standar. Menyitir literatur kalibrasi klasifikasi sebagai
pembenaran adalah lompatan yang tidak akan bertahan di sidang.

**Keadilan perbandingan.** Kalau model usulan dikalibrasi, baseline WAJIB
dikalibrasi juga. Kalau tidak, perbandingannya curang.

## 2. Hasil uji keadilan

| Domain | selisih AGF−A2IRM sebelum | setelah KEDUANYA dikalibrasi |
|---|---|---|
| Restaurant | +3,41% (kalah) | **+2,61% (kalah)** |
| E-commerce | −3,99% (menang) | **−0,23% (praktis seri)** |
| Hotel | −1,29% (menang) | **−0,97% (menang tipis)** |

A2-IRM justru mendapat manfaat LEBIH BESAR dari kalibrasi di e-commerce
(−7,27% vs AGF −3,64%). Kalibrasi bukan kontribusi — ia perbaikan seragam
yang, bila diterapkan jujur, **menghapus sebagian besar keunggulan
arsitektur usulan**.

## 3. Temuan yang lebih serius: klaim robustness juga runtuh

Kolaps A2-IRM di seed 1011 e-commerce (RMSE 0,8961) hampir sepenuhnya
diselamatkan kalibrasi → 0,6854.

| Metrik (e-commerce) | sebelum kalibrasi | setelah kalibrasi adil |
|---|---|---|
| SD lintas seed | A2-IRM 0,1023 vs AGF 0,0438 (2,3×) | 0,0140 vs 0,0110 (**1,27×**) |
| worst-case | 0,8961 vs 0,7626 (−14,9%) | 0,6854 vs 0,6791 (**−0,9%**) |

**Kolaps itu kegagalan KALIBRASI, bukan instabilitas fundamental.**
Keunggulan robustness AGF sebagian besar adalah "AGF kebetulan lebih
terkalibrasi", bukan "AGF lebih kokoh secara struktural".

Ini kritik yang hampir pasti muncul dari penguji: *"keunggulan robustness
Anda hilang oleh perbaikan post-hoc sepele pada baseline, jadi itu bukan
keunggulan arsitektural."* Lebih baik ditemukan sekarang.

## 4. Klaim yang BERTAHAN: token sentimen global

| Domain | efek F−F− sebelum | setelah kedua sisi dikalibrasi |
|---|---|---|
| Restaurant | −8,78% | **−9,10%** |
| E-commerce | −15,50% | **−16,33%** |
| Hotel | −5,93% | **−6,40%** |

Efeknya **tidak berkurang, bahkan menguat**. Ini membuktikan token global
adalah **perolehan INFORMASI**, bukan artefak kalibrasi — berbeda kategori
dari semua klaim lain di atas.

---

## 5. Posisi manuskrip setelah audit

**Bertahan uji:**
- Token sentimen global: −6% s/d −16%, konsisten 3/3 domain, 5/5 seed
  signifikan, tahan terhadap kalibrasi adil. Disertai penjelasan mekanistik
  (Gerbang-3): keyword-concat memperoleh sinyal ini secara implisit lewat
  pengisian aspek tak-match; representasi berbasis agregasi kehilangannya
  dan harus memulihkannya secara eksplisit.

**TIDAK bertahan:**
- Keunggulan akurasi AGF atas A2-IRM (sebagian besar artefak kalibrasi)
- Keunggulan robustness AGF (idem — kolaps baseline adalah kegagalan
  kalibrasi)
- Ekstraksi PyABSA (sudah negatif sebelum audit)
- Sequence pooling identitas aspek (sudah ~nol sebelum audit)

**Implikasi:** kontribusi menyempit dari "arsitektur baru yang lebih baik"
menjadi "temuan diagnostik tentang dari mana keunggulan representasi ABSA
sebenarnya berasal". Itu kontribusi ANALISIS, bukan ARSITEKTUR — lebih
sempit, tapi jauh lebih kokoh karena setiap penjelasan alternatif sudah
diuji dan disingkirkan satu per satu.

**Kalibrasi TIDAK direkomendasikan** untuk dimasukkan sebagai komponen:
landasan literaturnya lemah untuk regresi, dan bila diterapkan adil ia
justru merugikan posisi model usulan.
