# A2-FusionRS — Kerangka Beranotasi Manuskrip (target: Expert Systems with Applications, Q1)

> Format ESWA: single-column, ~17+ halaman. Judul bagian = bahasa Inggris (final);
> anotasi (poin isi, tabel/gambar, klaim, referensi) = bahasa Indonesia untuk review.
> Semua angka merujuk `A2-FusionRS_results_ledger.md`. Tag `[REF: ...]` = butuh sitasi
> nyata (dicari via web, diverifikasi Anda). Target ~40+ referensi.

---

## Judul (kandidat)
1. **"A2-FusionRS: Attention-Gated Fusion of Model-Based Aspect Sentiment for Review-Aware Recommendation under Extreme Sparsity"**
2. "Complementing Keyword Aspects with Model-Based ABSA: An Attention-Gated Fusion Recommender"
3. "When Do Fine-Grained Aspect Sentiments Help Recommendation? An Attribution Study with Attention-Gated Fusion"

> Rekomendasi: #1 (menonjolkan mekanisme + kontribusi + konteks sparsity). Hindari kata "novel/superior".

## Abstract (~250 kata, terstruktur)
- Konteks: RS berbasis review; sparsity ekstrem melumpuhkan CF murni; ABSA memberi sinyal preferensi fine-grained.
- Gap: ABSA berbasis kata-kunci (keyword) terbatas taksonomi tetap & cakupan; peran ABSA berbasis model + cara memfusikannya belum dipilah rigor.
- Metode: A2-FusionRS — attention-gated fusion atas DeepMF + CBF + ABSA, dengan modalitas PyABSA per-aspek (aspect-sequence pooling).
- Hasil kunci (dari ledger): unggul signifikan atas A2-IRM & 4 baseline (NeuMF, DeepFM, SVD, Item-KNN) di 3 domain × 5 seed (Wilcoxon p<0,001); RMSE Amazon 0,642 / Restaurant 0,667 / Hotel 0,620.
- Temuan: peningkatan bersumber dari ABSA berbasis model sebagai modalitas komplementer; manfaatnya terbesar saat cakupan keyword-ABSA rendah; attention-gated fusion menyamai fusi statis pada akurasi sambil menyediakan interpretability.
- Kata kunci: recommender systems; aspect-based sentiment analysis; attention mechanism; data sparsity; multimodal fusion; explainability.

---

## 1. Introduction (~2–2,5 hal)
**Isi:**
- Ledakan review daring → sumber preferensi kaya; RS berbasis rating murni menderita sparsity & cold-start. [REF: RS survey, sparsity, cold-start]
- Sentimen tingkat-aspek (ABSA) menangkap "apa yang disuka/tak disuka" per aspek, lebih informatif dari sentimen dokumen. [REF: ABSA survey; sentiment-aware RS]
- Dua celah konkret: (a) ABSA keyword-based terbatas taksonomi tetap & cakupan rendah di domain tertentu; (b) ketika sinyal ABSA ditambahkan, *dari mana* peningkatan berasal & *mekanisme fusi mana* yang berperan jarang dipilah — banyak paper mengklaim arsitektur canggih tanpa kontrol atribusi. [REF: hybrid RS; attention fusion RS]
- Penelitian ini membangun di atas A2-IRM (pendahulu, fusi statis + keyword-ABSA) dan bertanya: apakah ABSA berbasis model (open-vocabulary) + fusi attention memperbaikinya, dan *mengapa*.

**Kontribusi (eksplisit, 4 butir):**
1. A2-FusionRS: kerangka attention-gated fusion yang mengintegrasikan ABSA berbasis model per-aspek (aspect-sequence pooling) sebagai modalitas komplementer, unggul signifikan atas baseline internal & eksternal di 3 domain.
2. Kerangka **atribusi rigor** (ablasi komponen + kontrol fusi-statis-vs-attention + analisis plafon informasi via OOF residual) yang memisahkan sumber peningkatan — jarang dilakukan di literatur RS+ABSA.
3. Temuan empiris **ketergantungan-cakupan**: manfaat ABSA berbasis model berbanding terbalik dengan cakupan keyword-ABSA domain (paling menolong saat keyword lemah).
4. Evaluasi menyeluruh: 5 seed × 3 domain, uji Wilcoxon, baseline klasik+neural+hybrid, analisis interpretability & efisiensi; kode & konfigurasi terbuka (reproducibility).

**Research Questions (opsional, memperjelas):** RQ1 apakah A2-FusionRS > baseline? RQ2 dari mana peningkatannya (ABSA model vs fusi attention)? RQ3 kapan ABSA berbasis model paling menolong?

> Gambar: Fig. 1 (konsep tingkat-tinggi: review → aspek+sentimen → fusi → rekomendasi) opsional.

## 2. Related Work (~2,5–3 hal)
**2.1 Collaborative filtering & neural recommendation** — MF, NCF/NeuMF, DeepFM; keterbatasan pada sparsity. [REF: MF Koren; NeuMF He 2017; DeepFM Guo 2017; 2–3 terbaru]
**2.2 Review-aware & content-based recommendation** — memanfaatkan teks review; embedding review; keterbatasan sentimen dokumen. [REF: review-based RS ×3–4]
**2.3 Aspect-based sentiment analysis for recommendation** — keyword/lexicon vs model-based (BERT/PyABSA); bagaimana aspek dipakai di RS. [REF: ABSA survey; ATEPC/PyABSA; sentiment-aware RS ×3]
**2.4 Attention & multimodal fusion in RS** — attention/gating untuk menggabungkan sinyal heterogen; interpretability. [REF: attention RS ×3; multimodal fusion ×2]
**2.5 Positioning** — sintesis gap: mayoritas mengklaim keunggulan arsitektur tanpa atribusi; cakupan/keterbatasan keyword-ABSA jarang dianalisis lintas-domain. Paragraf penutup memosisikan A2-FusionRS.

> WAJIB: sintesis (tabel perbandingan pendekatan?), bukan daftar. Tabel 1 kandidat: matriks metode (CF? review? ABSA type? fusion? attribution?).

## 3. Preliminaries and Problem Formulation (~1 hal)
- Notasi: user u∈U, item i∈I, rating r_ui∈[1,5], review d_ui.
- Definisi masalah: prediksi rating (regresi) untuk pasangan (u,i) teramati, dievaluasi RMSE/MAE + ranking (P/R/NDCG@K).
- Definisi modalitas: DeepMF (sinyal kolaboratif), CBF (konten item + preferensi cluster user), ABSA (polaritas aspek dari review). [Persamaan notasi]

## 4. Proposed Methodology: A2-FusionRS (~3,5–4 hal) — INTI
**Gambar utama: Fig. 2 — Arsitektur A2-FusionRS** (alur lengkap; dibuat sebagai artifact/SVG).

**4.1 Overview** — pipeline: ekstraksi 3 modalitas → tokenisasi modalitas → cross-attention + gated pooling → residual prediction head. Filosofi: asimetris (DeepMF/CBF→prediktor skalar; ABSA→representasi kaya).

**4.2 Modality encoders**
- DeepMF: embedding user/item → interaksi → prediksi rating ternormalisasi (skalar). [Eq: DeepMF]
- CBF: fitur item (PCA) + preferensi cluster user → skalar. [Eq: CBF]
- ABSA: **PyABSA per-aspek** — untuk tiap review, ekstrak himpunan aspek + probabilitas polaritas (pos/neu/neg) + confidence. [Eq: PyABSA aspect set]

**4.3 Aspect-sequence pooling** (komponen teknis kunci)
- Masalah: PyABSA open-vocabulary → jumlah aspek variabel, tak bisa di-concat kolom tetap; rata-rata menghancurkan kontras antar-aspek (pelajaran dari A2-IRM Fase 1).
- Solusi: sequence per-aspek [embedding identitas aspek ⊕ (P_neg,P_neu,P_pos,conf)] → query-attention masked → 1 token modalitas ABSA. [Eq: aspect embedding, attention pooling]
- Klaim: identitas aspek = informasi yang tree tak bisa konsumsi (motivasi struktural). Dilaporkan JUJUR: keunggulan atas tree ternyata tidak robust (lihat §6).

**4.4 Attention-gated fusion**
- Token modalitas {deepmf, cbf, absa} → multi-head cross-attention → gated pooling (bobot gate per-modalitas, dinormalisasi). [Eq: attention, gate]
- Interpretability: bobot gate = kontribusi tiap modalitas per prediksi.

**4.5 Residual prediction head & training**
- Residual: prediksi = base (fusi statis NMF+DT, via OOF 5-fold utk cegah kebocoran) + koreksi head(fused). [Eq: residual]
- Loss MSE pada rating ternormalisasi; Adam + weight decay; restore epoch val-terbaik. [Eq: loss]

**4.6 Complexity & efficiency** — jumlah parameter, kompleksitas fusi; disiapkan utk Tabel efisiensi §6.

## 5. Experimental Setup (~1,5–2 hal)
**5.1 Datasets** — Amazon Electronics, Yelp Restaurant, TripAdvisor Hotel; statistik (Tabel 2: #user, #item, #review, sparsity, cakupan keyword-ABSA & PyABSA, rata-rata aspek/review). Proses subsample (5-core dll) dijelaskan. [REF: dataset sumber Amazon/Yelp/TripAdvisor]
**5.2 Split & protocol** — user-based split identik semua model; train/val/test; seed {42,123,456,789,1011}.
**5.3 Baselines** — tier: heuristik (Global Mean); CF klasik (Item-KNN, SVD) [REF: surprise]; CF neural (NeuMF [REF He 2017], DeepFM [REF Guo 2017]); hybrid (A2-IRM, pendahulu). Semua diadaptasi ke regresi rating, protokol identik.
**5.4 Metrics** — RMSE, MAE (utama); Precision/Recall/NDCG@K (ranking); uji Wilcoxon signed-rank per seed atas squared-error + pelaporan Δ-RMSE. [REF: evaluasi RS; significance testing]
**5.5 Implementation details** — PyABSA checkpoint "english" (ATEPC) [REF: PyABSA/BERT]; hyperparameter (Tabel 3); lingkungan (Colab GPU); ketersediaan kode.

## 6. Results and Discussion (~3,5–4 hal) — grounded penuh ke ledger
**6.1 Overall performance (RQ1)** — **Tabel 4** (mean±SD semua model × 3 domain, dari ledger §2). **Fig. 3** bar chart RMSE + error bar. A2-FusionRS terbaik di semua domain; unggul signifikan (Tabel 5: ringkasan Wilcoxon 5/5). Narasi konvergensi pure-CF ~1,1–1,2 = plafon CF pada sparsity ekstrem; sinyal review membawa ke ~0,65.
**6.2 Ablation study (RQ2)** — **Tabel 6** (ablasi komponen dari ledger §3). Bukti: AGF-tanpa-PyABSA ≈ A2-IRM (plafon informasi); tree+PyABSA & AGF+PyABSA sama-sama menembus → **atribusi ke PyABSA**. **Fig. 4** ablation ladder.
**6.3 Attribution: does the attention fusion help beyond PyABSA? (kejujuran)** — kontrol tree+PyABSA vs AGF: TIDAK unggul robust (Amazon tree menang, Restaurant AGF 4/5, Hotel seri). Disampaikan sebagai temuan, bukan disembunyikan. AGF diposisikan = fusi interpretable yang menyamai fusi statis. Catatan artefak Wilcoxon N-besar.
**6.4 When does model-based ABSA help? (RQ3)** — **Fig. 5**: manfaat PyABSA vs cakupan keyword-ABSA (tren terbalik: −0,014 Amazon → −0,009 Hotel). Insight mekanistik.
**6.5 Interpretability** — **Fig. 6** distribusi bobot gate per modalitas per domain; studi kasus 1–2 prediksi (aspek mana mendorong). 
**6.6 Efficiency** — **Tabel 7**: parameter, waktu latih/inferensi vs baseline.
**6.7 Threats to validity** — 3 domain, bahasa Inggris, 1 checkpoint PyABSA; artefak N-besar; keunggulan-fusi tak robust; subsample.

## 7. Conclusion and Future Work (~0,75–1 hal)
- Rangkuman kontribusi & jawaban RQ (tanpa overclaim).
- Future work: attention pooling lebih kaya; multi-bahasa; checkpoint ABSA domain-spesifik; baseline review-aware terbaru; uji pada domain cakupan-menengah lain.

## Declarations
- CRediT authorship; Data availability (repo GitHub, split, seed); Funding (UM 2026); Conflict of interest; (opsional) penggunaan alat bantu.

---

## Daftar Gambar (dibuat terpisah)
- Fig. 1 (opsional) konsep tingkat-tinggi.
- **Fig. 2 Arsitektur A2-FusionRS** (utama, artifact/SVG).
- Fig. 3 Bar RMSE + error bar (3 domain).
- Fig. 4 Ablation ladder.
- Fig. 5 Manfaat PyABSA vs cakupan aspek.
- Fig. 6 Distribusi bobot gate (interpretability).

## Daftar Tabel
- Tabel 1 Perbandingan pendekatan (related work).
- Tabel 2 Statistik dataset.
- Tabel 3 Hyperparameter.
- Tabel 4 Hasil utama mean±SD (ledger §2).
- Tabel 5 Ringkasan signifikansi Wilcoxon (ledger §4).
- Tabel 6 Ablasi komponen (ledger §3).
- Tabel 7 Efisiensi.

## Rencana Referensi (≥40; mayoritas Q1/Q2 + IEEE/WoS; utamakan DOI) — dicari via web, diverifikasi Anda
- CF & neural RS: ~8 (MF, NeuMF, DeepFM, NCF-terbaru, sparsity/cold-start).
- Review-aware & content-based RS: ~7.
- ABSA (survey, model-based/PyABSA/ATEPC/BERT): ~8.
- Attention/gating & multimodal fusion (RS & umum): ~7.
- Sentiment-aware RS: ~5.
- Evaluasi/metodologi (Wilcoxon, protokol RS, reproducibility): ~4.
- Dataset & domain (Amazon/Yelp/TripAdvisor): ~4.

## Anggaran halaman (≈17+)
Intro 2,3 · Related 2,8 · Prelim 1 · Method 3,8 · Setup 1,8 · Results 3,8 · Conclusion 0,9 · Refs 1,8.
