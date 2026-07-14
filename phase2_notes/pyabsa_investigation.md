# Investigasi PyABSA — Catatan Fase 2 (A2-FusionRS)

**Tanggal**: 2026-07-14
**Konteks**: Langkah pertama menuju A2-FusionRS (per diskusi roadmap) adalah mengganti ABSA
keyword-matching (`src/a2fusionrs/absa_bert.py::KeywordAspectSentimentScorer`, dipakai penuh
di A2-IRM) dengan ekstraksi aspek berbasis model (PyABSA), sebelum masuk ke desain
Attention-Gated Fusion. Dokumen ini merangkum hasil investigasi awal — apa yang terbukti
bekerja, apa yang tidak, dan rekomendasi konkret untuk saat integrasi sungguhan dimulai.

## Ringkasan eksekutif

PyABSA **terbukti feasible secara teknis** dan **kualitas outputnya menjanjikan** — divalidasi
lewat uji CPU lokal yang berhasil penuh. Tapi environment-nya (checkpoint pretrained,
dependency lama) **terbukti rapuh** di lingkungan lain (Colab GPU), dengan rangkaian kegagalan
yang eskalasi dari mudah-ditambal ke fundamental. Rekomendasi: **jangan pakai checkpoint
pretrained PyABSA off-the-shelf saat integrasi sungguhan** — fine-tune model ABSA sendiri
langsung lewat `transformers` (sejalan dengan desain proposal: modul Aspect Discovery +
Aspect-Sentiment Co-Attention yang di-fine-tune dgn combined loss, bukan library pihak ketiga).

## 1. Validasi lokal (Windows, Python 3.9, CPU) — BERHASIL

### Setup environment
Instalasi `pyabsa==2.4.2` butuh beberapa workaround (semua low-risk, **tidak mengubah**
`torch`/`transformers`/`numpy`/`pandas`/`scikit-learn` yang sudah divalidasi Fase 1):
- `spacy<3.8` (spacy versi terbaru butuh Python>=3.10, venv ini 3.9.13) -> resolve ke 3.7.5
- `findfile==2.0.1` (versi terbaru pakai sintaks union-type `X | None` yg crash di Python 3.9
  tanpa `from __future__ import annotations`)
- `protobuf` + `sentencepiece` (dibutuhkan tokenizer DeBERTa-v2 checkpoint)
- `en_core_web_sm` (model spacy; auto-downloader PyABSA salah panggil Python sistem, perlu
  `python -m spacy download en_core_web_sm` manual di venv yg benar)

### Hasil uji (25 sample review, TripAdvisor Hotel domain, checkpoint "english")
- Checkpoint dimuat & inferensi jalan tanpa masalah, first try.
- **Kecepatan CPU**: 2,78 detik/review -> ekstrapolasi TripAdvisor penuh (79.562 review)
  ~61 jam CPU sekuensial -- **tidak praktis tanpa GPU**, konsisten dgn pola SA-BERT sebelumnya.
- **Cakupan**: 20/25 (80%) review dapat >=1 aspek. **Catatan penting**: keyword-matching pada
  sample yg SAMA justru dapat 24/25 (96%) -- PyABSA TIDAK otomatis menang di cakupan mentah,
  masih butuh strategi fallback sama seperti pipeline sekarang.
- **Kualitas** (nilai tambah nyata): PyABSA menangkap nuansa **di dalam** satu kategori aspek yg
  sama. Contoh nyata: 1 review menghasilkan `TVs`(+), `bathroom light`(−), `front desk`(−),
  `stay`(+), `bed`(+) -- lima istilah spesifik dgn polaritas individual. Keyword-matching
  memetakan SEMUA ini jadi 1 kategori "rooms" (5 kalimat digabung/dirata-rata), kehilangan
  kontras positif/negatif yg justru jadi temuan inti A2-IRM (averaging = destruktif).
- **Confidence per-aspek** tersedia langsung, format sama persis dgn yg dibutuhkan mekanisme
  "concat+confidence" yg terbukti menang di A2-IRM.
- **Temuan data tak terduga**: 1 dari 25 sample ternyata berbahasa Italia. Keyword-matching
  (benar) hasilkan 0 aspek (keyword semua Inggris). PyABSA (checkpoint "English") tetap
  hasilkan 7 istilah yg terlihat masuk akal, TAPI keandalannya meragukan (model tdk dilatih
  utk Italia). **Belum diukur skala isu multi-bahasa di korpus TripAdvisor secara keseluruhan**
  -- perlu dicek terpisah sebelum keputusan bahasa/checkpoint final.

### Isu desain terbuka (BUKAN soal implementasi, tapi soal arsitektur)
PyABSA menghasilkan istilah aspek **open-vocabulary** (beda-beda tiap review), bukan dari
taksonomi tetap 4-6 kategori seperti keyword-matching sekarang. Representasi fitur "concat"
(vektor panjang tetap) yg jadi pemenang di A2-IRM **tidak langsung compatible** tanpa strategi
pemetaan. **Keputusan yang sudah diambil**: JANGAN paksa pemetaan taksonomi tetap sekarang
(itu akan mengulang masalah averaging yg sudah terbukti merusak sinyal) -- proposal sendiri
memasangkan modul "Aspect Discovery" dgn "Aspect-Sentiment Co-Attention Layer" sbg SATU unit,
jadi input panjang-variabel PyABSA memang didesain utk attention, bukan fusion statis
NMF+DecisionTree yg dipakai sekarang. **Integrasi PyABSA ditunda sampai desain
Attention-Gated Fusion dimulai.**

## 2. Uji kecepatan GPU (Google Colab) — TIDAK SELESAI, dihentikan

Upaya benchmark 500 review di GPU (Tesla T4) menemui rangkaian kegagalan lingkungan yg
eskalatif, didokumentasikan di sini supaya tidak perlu diulang dari nol nanti:

| # | Masalah | Penyebab | Status |
|---|---|---|---|
| 1 | `TypeError: UpdateChecker.check() takes 1 positional argument but 3 were given` | Versi `update-checker` yg ter-resolve di Colab tidak cocok dgn API yg diharapkan `metric-visualizer` (dependency pyabsa) | Fixed: pin `update-checker==0.18.0` |
| 2 | Checkpoint zip ter-download tapi cuma "1MB" & gagal unzip | Downloader internal PyABSA sendiri rusak di Colab (kemungkinan gagal handle redirect Xet storage HuggingFace) | Worked around: `wget` manual + taruh di lokasi yg PyABSA harapkan -- **berhasil sekali** |
| 3 | `wget`/`hf_hub_download` ulang dapat `403 Forbidden` / `SignatureError: invalid key pair id` di CDN `us.gcp.cdn.hf.co` | Bug infrastruktur Xet storage HuggingFace sendiri (CDN edge tertentu key-pair-nya tidak valid) -- **di luar kendali kita**, murni soal CDN mana yg kebagian scr acak | Retry beberapa kali; kadang berhasil (edge `cas-bridge.xethub.hf.co`), kadang gagal |
| 4 | Checkpoint `multilingual` berhasil download+extract, tapi `ModuleNotFoundError: No module named 'transformers.models.deberta_v2.tokenization_deberta_v2_fast'` saat `pickle.load()` | Checkpoint pretrained PyABSA (era ~2023, warning resmi minta `transformers<=4.29.0`) di-pickle dgn struktur modul internal `transformers` versi lama yg sudah direorganisasi total di versi modern (Colab pakai versi jauh lebih baru) | **Belum dicoba fix (downgrade transformers)** -- diputuskan berhenti di sini |

**Keputusan berhenti**: pola kegagalan makin ke fundamental (bug pip trivial -> CDN pihak
ketiga -> inkompatibilitas pickle/library), dan kemungkinan besar TIDAK spesifik ke satu
checkpoint (semua checkpoint PyABSA dari author yg sama, era yg sama). Waktu tambahan menambal
lingkungan Colab ini tidak sebanding dgn nilainya, mengingat angka throughput GPU **tidak
menghalangi keputusan apa pun yg sudah diambil** (integrasi sudah ditunda ke fase
Attention-Gated Fusion terlepas dari angka ini).

## 3. Rekomendasi untuk fase integrasi sungguhan (nanti)

1. **Jangan bergantung pada checkpoint pretrained PyABSA off-the-shelf.** Riwayat kegagalan
   di atas (checkpoint usang, hosting rapuh, versi `transformers` yg diasumsikan tetap
   `<=4.29.0`) adalah sinyal kuat package ini kurang terawat utk proyek riset multi-tahun.
2. **Fine-tune model ABSA sendiri langsung lewat `transformers`** (DeBERTa-v2/v3 atau BERT-base
   biasa) pada data domain kita sendiri -- ini justru LEBIH sejalan dgn deskripsi arsitektur
   proposal ("modul Aspect Discovery... di-fine-tune dgn combined loss") drpd menggunakan
   library pretrained pihak ketiga. Hindari dependency `pyabsa`/`spacy`/`findfile` sama sekali.
3. **Ukur skala isu multi-bahasa** di korpus TripAdvisor (dan domain lain) sebelum finalisasi
   strategi -- temuan 1 review Italia dari 25 sample menunjukkan ini nyata, bukan kasus langka.
4. **Desain representasi ABSA baru dibarengi dgn Attention-Gated Fusion**, bukan dipaksa masuk
   fusion statis NMF+DecisionTree yg ada sekarang -- output open-vocabulary/panjang-variabel
   memang cocoknya dgn cross-attention, sesuai desain proposal sendiri.
5. Kalau tetap ingin uji kecepatan GPU PyABSA lagi suatu saat, mulai dari **Cell 1 = downgrade
   `transformers<=4.29.0` di awal sesi** (bukan ditambal belakangan) -- tapi mengingat rekomendasi
   #1-2 di atas, kemungkinan besar tidak perlu lagi dicoba.

## Lampiran: environment lokal yg berhasil (utk referensi cepat)

```
Python 3.9.13 (Windows, venv proyek ini)
pyabsa==2.4.2
spacy==3.7.5 (dipin dari default terbaru)
findfile==2.0.1 (dipin dari default terbaru)
protobuf==6.33.6
sentencepiece==0.2.2
en_core_web_sm (via `python -m spacy download en_core_web_sm`)
# torch/transformers/numpy/pandas/scikit-learn: TIDAK BERUBAH dari Fase 1
```
