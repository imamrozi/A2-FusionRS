# Investigasi PyABSA — Catatan Fase 2 (A2-FusionRS)

**Tanggal**: 2026-07-14
**Konteks**: Langkah pertama menuju A2-FusionRS (per diskusi roadmap) adalah mengganti ABSA
keyword-matching (`src/a2fusionrs/absa_bert.py::KeywordAspectSentimentScorer`, dipakai penuh
di A2-IRM) dengan ekstraksi aspek berbasis model (PyABSA), sebelum masuk ke desain
Attention-Gated Fusion. Dokumen ini merangkum hasil investigasi awal — apa yang terbukti
bekerja, apa yang tidak, dan rekomendasi konkret untuk saat integrasi sungguhan dimulai.

## Ringkasan eksekutif

PyABSA **terbukti feasible penuh secara teknis** di CPU (lokal) maupun GPU (Colab, setelah
serangkaian perbaikan environment yang didokumentasikan lengkap di bawah). **Pemilihan
checkpoint sangat menentukan kualitas**: checkpoint `english` memberi cakupan aspek jauh lebih
baik (70%) dibanding `multilingual` (17,2%) pada domain kita yang mayoritas berbahasa Inggris,
dengan kecepatan GPU yang tetap praktis (~3,6-5,6 jam/domain skala penuh). Kualitas outputnya
menjanjikan (menangkap nuansa dalam-kategori yang keyword-matching lewatkan). **Keputusan
desain tetap**: integrasi ke pipeline ditunda sampai fase Attention-Gated Fusion (output
open-vocabulary PyABSA cocoknya dengan cross-attention, bukan fusion statis NMF+DecisionTree
yang ada sekarang). Rekomendasi tambahan: pertimbangkan fine-tune model ABSA sendiri
ketimbang bergantung checkpoint pretrained off-the-shelf yang terbukti sudah usang (era 2023).

## 1. Validasi lokal (Windows, Python 3.9, CPU)

### Setup environment
Instalasi `pyabsa==2.4.2` butuh beberapa workaround (semua low-risk, **tidak mengubah**
`torch`/`transformers`/`numpy`/`pandas`/`scikit-learn` yang sudah divalidasi Fase 1):
- `spacy<3.8` (spacy versi terbaru butuh Python>=3.10, venv ini 3.9.13) -> resolve ke 3.7.5
- `findfile==2.0.1` (versi terbaru pakai sintaks union-type `X | None` yg crash di Python 3.9
  tanpa `from __future__ import annotations`)
- `protobuf` + `sentencepiece` (dibutuhkan tokenizer DeBERTa-v2 checkpoint)
- `en_core_web_sm` (model spacy; auto-downloader PyABSA salah panggil Python sistem, perlu
  `python -m spacy download en_core_web_sm` manual di venv yg benar)

### Hasil uji CPU (25 sample review, TripAdvisor Hotel domain, checkpoint "english")
- Checkpoint dimuat & inferensi jalan tanpa masalah, first try (lewat downloader internal
  PyABSA sendiri, tanpa workaround download manual).
- **Kecepatan CPU**: 2,78 detik/review -> tidak praktis utk skala penuh tanpa GPU.
- **Cakupan**: 20/25 (80%) review dapat >=1 aspek. Keyword-matching pada sample yg SAMA
  dapat 24/25 (96%) -- di sample kecil ini keyword-matching sedikit lebih tinggi cakupan
  mentahnya, tapi lihat hasil GPU sample besar (500 review) di bawah utk gambaran lebih solid.
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

## 2. Uji GPU (Google Colab, Tesla T4) — BERHASIL setelah 6 kegagalan environment

Perjalanan panjang, semua penyebab & fix didokumentasikan lengkap di sini supaya tidak perlu
diulang dari nol nanti:

| # | Masalah | Penyebab | Fix |
|---|---|---|---|
| 1 | `TypeError: UpdateChecker.check() takes 1 positional argument but 3 were given` | Versi `update-checker` yg ter-resolve di Colab tidak cocok dgn API yg diharapkan `metric-visualizer` (dependency pyabsa) | Pin `update-checker==0.18.0` (install SETELAH pyabsa, supaya tidak ketiban upgrade ulang) |
| 2 | Checkpoint zip ter-download tapi cuma "1MB" & gagal unzip | Downloader internal PyABSA sendiri rusak di Colab (gagal handle redirect Xet storage HuggingFace) | `wget`/`curl` manual |
| 3 | `wget` ulang / `hf_hub_download` dapat `403 Forbidden` / `SignatureError: invalid key pair id` di CDN `us.gcp.cdn.hf.co` | Bug infrastruktur Xet storage HuggingFace -- traffic dari Colab (jalan di GCP) tampaknya konsisten diarahkan ke CDN edge GCP yg rusak, sedangkan traffic dari jaringan non-GCP (mis. komputer lokal biasa) diarahkan ke edge lain yg sehat (`cas-bridge.xethub.hf.co`) | **Download checkpoint dari mesin LOKAL (bukan Colab)**, lalu user upload manual hasil zip-nya ke Google Drive; Colab tinggal extract dari Drive, tidak pernah hubungi HuggingFace utk file checkpoint ini lagi |
| 4 | `ModuleNotFoundError: No module named 'transformers.models.deberta_v2.tokenization_deberta_v2_fast'` saat `pickle.load()` config checkpoint | Checkpoint di-pickle era ~2023 dgn struktur modul internal `transformers` versi lama; versi `transformers` default Colab sudah mereorganisasi modul itu total | **Pin `transformers==4.57.6`** (versi PERSIS yg terbukti bekerja di venv lokal -- BUKAN cuma `<=4.29.0` sesuai saran generik pyabsa) + wajib restart session setelah pin |
| 5 | Download base model backbone (`microsoft/mdeberta-v3-base`, ~1,33GB) macet berulang, sesi Colab terus "Connecting.../Resuming execution" | Kombinasi: (a) download besar di jaringan Colab yg kadang tidak stabil, (b) `~/.cache/huggingface/` di Colab ephemeral -- hilang tiap restart, jadi tiap sesi baru mengulang download dari nol | Set `HF_HOME` ke path Google Drive SEBELUM download apa pun -- backbone ter-cache permanen, cuma perlu berhasil download SEKALI selamanya |
| 6 | (Insiden sampingan, bukan bug PyABSA) Disk `C:` di mesin lokal sempat penuh (0 byte bebas) gara-gara cache HuggingFace lokal 2,4GB dari uji verifikasi | Cache `huggingface_hub` default ke `C:\Users\...\.cache\`, drive itu memang sudah sangat penuh sepanjang sesi ini | Cache dihapus (`rm -rf ~/.cache/huggingface`) segera setelah terpakai; ke depan pertimbangkan set `HF_HOME` ke `D:` juga utk kerja lokal |

### Hasil benchmark final (500 review TripAdvisor Hotel, sample identik `random_state=42`, GPU Tesla T4)

| Checkpoint | Kecepatan | Cakupan (>=1 aspek) | Rata-rata aspek/review | Estimasi TripAdvisor penuh | Estimasi Amazon Electronics |
|---|---|---|---|---|---|
| **english** | 0,165 detik/review | **350/500 (70,0%)** | **2,77** | ~3,6 jam | ~5,6 jam |
| multilingual | 0,037 detik/review | 86/500 (17,2%) | 0,60 | ~0,8 jam | ~1,3 jam |

**Kesimpulan checkpoint**: `english` jelas lebih unggul kualitasnya (4x cakupan, 4,6x lebih
banyak aspek per review) dibanding `multilingual`, konsisten dgn hasil CPU sample kecil
sebelumnya (bukan kebetulan). `multilingual` jauh lebih cepat tapi cakupannya terlalu rendah
utk domain kita yg mayoritas berbahasa Inggris -- kemungkinan besar model multilingual
mengorbankan recall per-bahasa demi dukungan lintas-bahasa (trade-off umum di NLP). Kedua
checkpoint tetap praktis secara waktu di GPU (hitungan jam, bukan puluhan jam spt di CPU).

**Rekomendasi checkpoint**: kalau tetap pakai PyABSA off-the-shelf nanti, pakai **`english`**
utk domain yg mayoritas berbahasa Inggris (Amazon Electronics, TripAdvisor Hotel, Restoran) --
terima trade-off review non-Inggris (spt sample Italia yg ditemukan) tidak akan tertangkap
dgn baik.

## 3. Isu desain terbuka (BUKAN soal implementasi, tapi soal arsitektur)

PyABSA menghasilkan istilah aspek **open-vocabulary** (beda-beda tiap review), bukan dari
taksonomi tetap 4-6 kategori seperti keyword-matching sekarang. Representasi fitur "concat"
(vektor panjang tetap) yg jadi pemenang di A2-IRM **tidak langsung compatible** tanpa strategi
pemetaan. **Keputusan yang sudah diambil**: JANGAN paksa pemetaan taksonomi tetap sekarang
(itu akan mengulang masalah averaging yg sudah terbukti merusak sinyal) -- proposal sendiri
memasangkan modul "Aspect Discovery" dgn "Aspect-Sentiment Co-Attention Layer" sbg SATU unit,
jadi input panjang-variabel PyABSA memang didesain utk attention, bukan fusion statis
NMF+DecisionTree yg dipakai sekarang. **Integrasi PyABSA ditunda sampai desain
Attention-Gated Fusion dimulai.**

## 4. Rekomendasi untuk fase integrasi sungguhan (nanti)

1. **Pakai checkpoint `english`** (bukan `multilingual`) kalau tetap memilih jalur PyABSA
   off-the-shelf -- bukti empiris di atas jelas mendukung ini utk domain kita.
2. **Pertimbangkan tetap fine-tune model ABSA sendiri** langsung lewat `transformers`
   (DeBERTa-v2/v3 atau BERT-base) pada data domain kita sendiri -- checkpoint PyABSA terbukti
   berfungsi (setelah banyak perbaikan), tapi usianya (2023) dan rangkaian masalah environment
   di atas tetap sinyal bahwa dependency ini butuh effort ekstra utk dipertahankan jangka
   panjang. Fine-tune sendiri juga LEBIH sejalan dgn deskripsi arsitektur proposal ("modul
   Aspect Discovery... di-fine-tune dgn combined loss").
3. **Ukur skala isu multi-bahasa** di korpus TripAdvisor (dan domain lain) sebelum finalisasi
   strategi -- temuan 1 review Italia dari sample menunjukkan ini nyata, bukan kasus langka,
   dan langsung berkorelasi dgn cakupan checkpoint `english` yg "cuma" 70% (bukan mendekati
   100%) -- sebagian gap ini kemungkinan direview non-Inggris yg memang di luar kapasitas
   checkpoint `english`.
4. **Desain representasi ABSA baru dibarengi dgn Attention-Gated Fusion**, bukan dipaksa masuk
   fusion statis NMF+DecisionTree yg ada sekarang.
5. **Kalau pakai PyABSA lagi di Colab**, replikasi urutan setup yg SUDAH terbukti bekerja
   (lihat Lampiran B) -- tidak perlu re-diagnosis dari nol.

## Lampiran A: environment lokal yang berhasil (CPU)

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

## Lampiran B: urutan setup Colab yang terbukti bekerja (GPU)

Checkpoint zip (`english` 606MB, `multilingual` 811MB) didownload dari mesin lokal (BUKAN
Colab, krn CDN Xet HuggingFace bermasalah khusus utk traffic dari GCP), lalu diupload manual
oleh user ke Google Drive folder `pyabsa_checkpoints/`.

```python
# Sesi baru (VM benar-benar baru) -- Cell 1 & 2
from google.colab import drive
drive.mount('/content/drive')
import os
os.environ["HF_HOME"] = "/content/drive/MyDrive/PHD-STUDENT/Code/A2-FusionRS/hf_cache"
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.makedirs(os.environ["HF_HOME"], exist_ok=True)

!pip install -q pyabsa
!pip install -q "update-checker==0.18.0"
!pip install -q "transformers==4.57.6"
!python -m spacy download en_core_web_sm -q
# --> WAJIB Restart session di sini <--
```

```python
# Setelah restart session (paket ter-install tetap ada) -- Cell 3-5
from google.colab import drive
drive.mount('/content/drive')
import os
os.environ["HF_HOME"] = "/content/drive/MyDrive/PHD-STUDENT/Code/A2-FusionRS/hf_cache"
os.environ["HF_HUB_DISABLE_XET"] = "1"

import zipfile
DRIVE_ZIP = "/content/drive/MyDrive/PHD-STUDENT/Code/A2-FusionRS/pyabsa_checkpoints/fast_lcf_atepc_English_cdw_apcacc_82.36_apcf1_81.89_atef1_75.43.zip"
LOCAL_DIR = "./checkpoints/ATEPC_ENGLISH_CHECKPOINT/fast_lcf_atepc_English_cdw_apcacc_82.36_apcf1_81.89_atef1_75.43"
os.makedirs(LOCAL_DIR, exist_ok=True)
with zipfile.ZipFile(DRIVE_ZIP) as z:
    z.extractall(LOCAL_DIR)

from pyabsa import AspectTermExtraction as ATEPC
aspect_extractor = ATEPC.AspectExtractor(checkpoint=LOCAL_DIR, auto_device=True)
```

Sesi Colab berikutnya (VM baru lagi, tapi Drive persisten): cukup ulangi blok kedua di atas
(mount + extract + load) -- tidak perlu install/download apa pun dari internet lagi selamanya,
kecuali `pip install` paket (yg hilang tiap VM baru, beda dgn restart session biasa).
