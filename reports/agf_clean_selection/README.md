# Seleksi Jangkar A2-FusionRS "clean" — hasil DEV (2026-08-07)

**Status: PEMENANG DIKUNCI. Test set BELUM disentuh saat dokumen ini dibuat.**
Dokumen + CSV di folder ini adalah **jejak audit** bahwa konfigurasi dipilih
sebelum evaluasi test dijalankan (Tahap 7).

## Protokol

- 4 varian × 3 domain × 3 seed (42/123/456) = **36 run**, semua `--stage select`
  → dievaluasi pada `selection_dev` (15% potongan dari TRAIN).
  **Test set tidak pernah dimuat**, bahkan tidak ikut membentuk universe item CBF.
- Konstan di semua sel (ditetapkan a priori, **bukan** axis seleksi):
  `--scenario a2fusionrs_clean` (DeepMF + CBF + PyABSA; tanpa keyword-ABSA,
  tanpa NMF+DecisionTree), `--extra-pyabsa perseq_rich`, `--input-standardize`.
- Axis yang diseleksi: `residual_base {none, user_item_bias}` ×
  `representation {vector, asymmetric}`.

Aturan keputusan **dipra-registrasi** di header `scripts/run_agf_clean_selection.sh`
sebelum run dijalankan, diimplementasikan apa adanya oleh
`scripts/analyze_agf_selection.py`: mean dev-RMSE per (domain, varian) → ranking
per domain → varian dengan **mean-rank terendah lintas domain** menang;
tie-break = mean dev-RMSE ternormalisasi. Satu konfigurasi untuk ketiga domain
(pemilihan per-domain dilarang — itu overfitting ke dev).

## Hasil

Mean dev-RMSE (± SD lintas 3 seed), rank per domain dalam kurung:

| Varian | Restaurant | E-commerce | Hotel | mean rank |
|---|---|---|---|---|
| jangkar bias + skalar | 0,7417 ± 0,0052 (2) | **0,7659 ± 0,0066 (1)** | 0,6711 ± 0,0017 (3) | **2,00** |
| jangkar bias + vektor | 0,7450 ± 0,0125 (3) | 0,7717 ± 0,0235 (2) | **0,6672 ± 0,0043 (1)** | **2,00** |
| tanpa jangkar + vektor | **0,7384 ± 0,0057 (1)** | 0,7876 ± 0,0169 (4) | 0,6699 ± 0,0066 (2) | 2,33 |
| tanpa jangkar + skalar | 0,7462 ± 0,0041 (4) | 0,7838 ± 0,0034 (3) | 0,6773 ± 0,0049 (4) | 3,67 |

**PEMENANG: `--residual-base user_item_bias --representation asymmetric`**
(mean rank 2,00; mean dev-RMSE ternormalisasi 1,0035)

## Catatan kejujuran — WAJIB dibaca sebelum mengutip hasil ini

1. **Pemenang ditetapkan lewat TIE-BREAK, bukan kemenangan telak.** Dua varian
   berjangkar sama-sama mean rank 2,00; pemisahnya hanya RMSE ternormalisasi
   1,0035 vs 1,0055 — selisih **0,2%**. Menyebut konfigurasi ini "terbaik"
   tanpa kualifikasi akan menyesatkan.

2. **Tidak ada varian yang menang di semua domain.** Setiap dari tiga varian
   teratas menempati rank 1 di domain berbeda (restaurant → tanpa jangkar +
   vektor; e-commerce → jangkar bias + skalar; hotel → jangkar bias + vektor).
   Ini konsisten dengan tema berulang proyek ini: manfaat komponen bersifat
   **bergantung domain**, bukan universal.

3. **Efek jangkar tetap terlihat pada agregat**: dua varian berjangkar menempati
   dua posisi teratas (mean rank 2,00 keduanya) vs tanpa jangkar (2,33 dan 3,67).
   Tapi selisih absolutnya kecil (~1%), jadi klaim yang bisa dipertahankan
   adalah "jangkar bias sedikit membantu secara rata-rata", bukan "jangkar
   bias penting".

4. **Dev ≠ test.** `selection_dev` diambil dari TRAIN, sehingga user/item-nya
   sudah pernah dilihat model — berbeda dari test yang punya cold-start holdout.
   Ranking di dev **tidak dijamin** bertahan di test. Ini konsekuensi yang
   diterima secara sadar sebagai harga dari menolak seleksi-di-test.

5. **Satu run pernah gagal lalu diulang sukses**: `amazon_electronics seed42
   none_vector` gagal pada percobaan 2026-08-07 09:08 dan berhasil pada
   pengulangan 10:05 (RMSE 0,7821). Log kegagalan **tertimpa** oleh log run
   ulang (nama file sama), sehingga penyebab pastinya tidak terdokumentasi —
   kemungkinan gangguan sesi Colab, tapi ini tidak bisa dipastikan. 35 run
   lainnya tidak ada yang gagal. Determinisme pipeline sendiri sudah
   diverifikasi terpisah (bit-identical antara cache-hit/cache-miss/tanpa-cache,
   commit `d323cf6`).

## Berkas

- `selection_table.csv` — mean/SD/rank per (domain, varian)
- `selection_ranking.csv` — agregasi lintas domain + tie-break
- `raw_per_run_dev.csv` — 36 baris mentah (dev_rmse, val_rmse, n_param, waktu)

Hasil mentah lengkap (YAML + predictions + gates) ada di
`checkpoints/results_phase2_clean/dev/` (tidak di-track git karena besar).
