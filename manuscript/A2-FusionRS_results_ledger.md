# A2-FusionRS — Results Ledger (sumber kebenaran angka manuskrip)

> Semua angka di bawah SUDAH terverifikasi dari run multi-seed (5 seed: 42, 123,
> 456, 789, 1011) di Colab, tersimpan di `checkpoints/results/` pada Google Drive.
> **JANGAN menulis angka di manuskrip yang tidak ada di berkas ini.** Setiap angka
> punya prefix file sumbernya untuk audit.
>
> Protokol: split user-based identik untuk SEMUA model; test set Amazon=16.580,
> Restaurant=13.233, Hotel=11.795 sampel. Metrik utama RMSE (skala rating 1–5,
> lebih kecil lebih baik). Signifikansi: Wilcoxon signed-rank berpasangan per
> seed atas squared-error per-sampel.

## 1. Karakteristik domain (cakupan ABSA)

| Domain | Cakupan keyword-ABSA | Cakupan PyABSA | Rata-rata aspek/review (PyABSA) | Sparsity |
|---|---:|---:|---:|---:|
| Amazon Electronics | 45,1% | 80,4% | 1,81 | 99,91% |
| Restaurant (Yelp) | 87,7% | (dari coverage summary) | (dari coverage summary) | 5-core |
| TripAdvisor Hotel | 95,9% | 70,2% | 2,77 | — |

> CATATAN: cakupan keyword-ABSA vs PyABSA adalah metrik BERBEDA (keyword =
> taksonomi tetap; PyABSA = open-vocabulary). Angka Restaurant PyABSA perlu
> dikutip dari `pyabsa_coverage_summary` di Drive sebelum final.

## 2. Tabel utama — RMSE mean±SD (5 seed)

Prefix file: `<prefix>_<domain>_seed<seed>.yaml`

| Model | Prefix file | Amazon | Restaurant | Hotel |
|---|---|---:|---:|---:|
| Item-KNN | `classical_cf_item_knn` | 1,2240±,000 | 1,2019±,000 | 0,9163±,000 |
| SVD | `classical_cf_svd` | 1,1420±,001 | 1,0753±,001 | 0,8953±,000 |
| NeuMF | `neural_cf_neumf` | 1,1528±,001 | 1,0740±,001 | 0,8399±,001 |
| DeepFM | `neural_cf_deepfm` | 1,1529±,002 | 1,0746±,002 | 0,8393±,001 |
| A2-IRM (concat+conf) | `absa_ablation_concat_confidence` | 0,6517±,003 | 0,6791±,001 | 0,6291±,003 |
| **A2-FusionRS** | `agf_agf_keyword_oof_perseq` | **0,6418±,002** | **0,6665±,001** | **0,6196±,003** |
| Global Mean Predictor | (referensi) | 1,2143 | 1,1516 | 0,9163 |
| _rating_std(test)_ | (identitas) | 1,2120 | 1,1513 | 0,9133 |

> Global Mean Predictor ≈ rating_std (identitas matematis) → konfirmasi angka konsisten.
> TEMUAN PENTING: Item-KNN Hotel (0,9163) = PERSIS Global-Mean Hotel (0,9163) →
> Item-KNN berdegenerasi ke prediksi rata-rata (tak ada tetangga co-rating di data
> sparse); di Amazon/Restaurant malah > mean (tetangga bising). CF murni yang
> berfungsi (SVD/NeuMF/DeepFM) hanya −2…−8% dari mean; sinyal review menekan
> −31…−47%. Ini pembelaan reviewer-proof bahwa baseline BUKAN strawman.

## 3. Ablasi komponen A2-FusionRS — RMSE mean±SD (5 seed)

| Varian | Prefix file | Amazon | Restaurant | Hotel |
|---|---|---:|---:|---:|
| AGF tanpa PyABSA (plafon) | `agf_agf_keyword_oof` | 0,6520±,002 | 0,6773±,001 | 0,6286±,003 |
| tree(NMF+DT) + PyABSA | `agf_static_keyword_pyabsa_ctrl` | 0,6384±,001 | 0,6676±,001 | 0,6201±,003 |
| AGF + PyABSA order-stats | `agf_agf_keyword_oof_pyrich` | 0,6407±,002 | 0,6674±,001 | 0,6216±,003 |
| AGF + PyABSA identitas aspek (final) | `agf_agf_keyword_oof_perseq` | 0,6418±,002 | 0,6665±,001 | 0,6196±,003 |

## 4. Uji signifikansi (Wilcoxon, ringkasan n-seed signifikan)

| Perbandingan | Amazon | Restaurant | Hotel | Interpretasi |
|---|---|---|---|---|
| A2-FusionRS vs A2-IRM | 5/5 (p 10⁻¹¹²…10⁻¹⁹⁸) | 5/5 | 5/5 | HEADLINE menang telak |
| A2-FusionRS vs NeuMF | 5/5 (p≈0) | 5/5 | 5/5 | menang atas SOTA neural CF |
| A2-FusionRS vs DeepFM | 5/5 | 5/5 | 5/5 | idem |
| A2-FusionRS vs SVD | 5/5 | 5/5 | 5/5 | idem |
| A2-FusionRS vs Item-KNN | 5/5 | 5/5 | 5/5 | idem |
| tree+PyABSA vs A2-IRM | 5/5 | 5/5 | 5/5 | PyABSA menembus plafon |
| AGF-tanpa-PyABSA vs A2-IRM | seri* | seri (1/5) | seri* | arsitektur sendiri = plafon |
| **AGF vs tree (atribusi)** | tree menang | AGF 4/5 | seri (2/5) | AGF TIDAK unggul robust atas tree |

> \* "5/5 signifikan" utk AGF-noPy vs A2-IRM adalah artefak Wilcoxon N-besar
> (12–16k pasang mendeteksi geser distribusi mikro walau RMSE praktis seri).
> WAJIB lapor Δ-RMSE bersama p-value; JANGAN klaim peningkatan dari p saja.

## 5. Temuan mekanistik (untuk Discussion)

1. **Manfaat PyABSA ∝ 1/cakupan-keyword**: tree+PyABSA vs A2-IRM turun −0,0143
   (Amazon 45%), −0,0115 (Restaurant 88%), −0,0090 (Hotel 96%). PyABSA menambal
   paling banyak di domain yang keyword-ABSA-nya paling lemah.
2. **Keunggulan AGF atas tree TIDAK robust** (multi-seed): Amazon tree menang
   (+0,0034), Restaurant AGF marginal (−0,0011, 4/5), Hotel seri (−0,0005, 2/5).
   Tren monotonik single-seed tidak bertahan. → AGF diposisikan sebagai fusi
   INTERPRETABLE yang MENYAMAI fusi statis, BUKAN unggul akurasi.
3. **Pure-CF konvergen ~1,1–1,2** (KNN/SVD/NeuMF/DeepFM) = plafon CF sejati pada
   review ultra-sparse; sinyal review (konten+sentimen) yang membawa ke ~0,65.

## 5b. Interpretability (§6.5) — terverifikasi (seed 42; Exp-A lintas 5 seed)

**Exp-A — bobot gate rata-rata per modalitas (5 seed):**

| Domain (cakupan) | deepmf | cbf | absa(keyword) | pyabsa_aspect |
|---|---:|---:|---:|---:|
| Amazon (45,1%) | 0,236 | 0,234 | 0,251 | 0,279 |
| Restaurant (87,7%) | 0,239 | 0,224 | 0,248 | 0,290 |
| Hotel (95,9%) | 0,201 | 0,302 | 0,286 | 0,211 |

> Korelasi cakupan-keyword vs gate pyabsa_aspek: **r = −0,52** → modalitas
> PyABSA-aspek diberi bobot LEBIH BESAR di domain cakupan-RENDAH → INDEPENDEN
> mengonfirmasi temuan akurasi (manfaat PyABSA ∝ 1/cakupan). Triangulasi.
> CAVEAT: n=3 domain, Hotel outlier utama (cakupan tertinggi & bobot aspek
> terendah 0,211) → indikatif, bukan konklusif. Catat juga: di Hotel, CBF
> dominan (0,302) — ketergantungan modalitas bersifat domain-adaptif.

**Exp-C — faithfulness (buang aspek top-atensi vs acak, |Δpred|):**

| Domain | \|Δ\|top | \|Δ\|acak | top>acak | Wilcoxon p |
|---|---:|---:|---:|---:|
| Amazon | 0,0511 | 0,0160 | 71,2% | ≈0 |
| Restaurant | 0,0603 | 0,0319 | 69,9% | ≈0 |
| Hotel | 0,0475 | 0,0214 | 71,3% | ≈0 |

> Aspek top-atensi berdampak 2–3× lebih besar dari aspek acak (~70% baris, p≈0)
> → atensi FAITHFUL (mencerminkan pengaruh nyata pada koreksi), bukan dekoratif.

**Exp-B — studi kasus**: ilustratif; diseleksi aspek-bernama + koheren (sentimen
aspek-top searah prediksi). JUJUR: tak semua kasus koheren (Exp-C: ~70%, bukan
100%); banyak aspek top = <UNK> di luar vocab top-500 (terutama Amazon).

## 5c. Efisiensi (Tabel 10) — seed 42 (A2-FusionRS rata-rata 5 seed), GPU sama

| Model | params | train(s) A/R/H | inferensi(ms) A/R/H |
|---|---|---|---|
| Item-KNN | — (memory-based) | 3,0/0,9/0,3 | 180/392/66 |
| SVD | 2,42M/1,10M/1,34M | 1,1/0,9/0,6 | 112/85/58 |
| NeuMF | 3,09M/1,42M/1,73M | 49,9/41,8/29,5 | 32/23/122 |
| DeepFM | 1,58M/0,74M/0,89M | 42,3/40,7/28,2 | 35/24/82 |
| A2-FusionRS (fusion head) | ~0,063M | 40,6/38,9/26,2 | 14,5/11,5/10,6 |

> JUJUR: params A2-FusionRS = fusion head SAJA (encoder DeepMF/CBF/PyABSA beku,
> dikecualikan) → BUKAN "lebih ringan end-to-end". Yang sah dibandingkan langsung
> = latensi inferensi (A2-FusionRS TERCEPAT, 11–15ms). Trade-off: pra-komputasi
> offline (DeepMF + skoring PyABSA) yang baseline pure-CF tak butuh.

## 6. Klaim yang BOLEH & TIDAK BOLEH ditulis

BOLEH (didukung data):
- A2-FusionRS mengungguli A2-IRM & 4 baseline eksternal, signifikan, 3 domain × 5 seed.
- Peningkatan bersumber dari ABSA berbasis model per-aspek (PyABSA) sebagai modalitas komplementer (atribusi via ablasi + kontrol tree+PyABSA).
- Manfaat PyABSA terbesar pada domain cakupan-keyword terlemah.

TIDAK BOLEH (overclaim):
- "Attention-gated fusion mengungguli fusi statis pada akurasi" (kontrol tree+PyABSA menyamainya).
- Menyimpulkan keunggulan dari p-value saat RMSE seri (artefak N-besar).
- Generalisasi di luar 3 domain / bahasa Inggris / 1 checkpoint PyABSA.

---
_Terakhir diperbarui: setelah run baseline eksternal multi-seed (commit 76fcb0c)._
