# Invarian Fase 1 — A2-FusionRS

Berlaku sepanjang Fase 1 (lihat `docs/phase1_spec.md`). Kontrak kerja utama
ada di file itu; daftar ini adalah salinan operasional untuk Claude Code.

1. Fitur untuk memprediksi $(u,i)$ tidak boleh diturunkan dari review $(u,i)$, kecuali di bawah `protocol: legacy`.
2. `src/legacy/` tidak boleh diubah.
3. Split tidak boleh di-regenerate. P1, P2, P3 memakai split yang identik.
4. Setiap angka baru masuk ke results ledger dengan prefix berkas sumbernya.
5. Protokol selalu dipilih lewat config, tidak pernah lewat edit kode.
6. Leave-one-out berlaku juga di dalam split train, bukan hanya di test.
7. Tidak ada angka yang ditulis ke manuskrip yang tidak ada di results ledger.
8. Setiap run wajib mem-persist prediksi semua stream per-baris (bukan hanya prediksi fusion akhir) ke `checkpoints/results/`, termasuk stream yang sebelumnya hanya hidup in-memory (mis. `deepmf_preds`, `cbf_preds`). Lihat `reports/leakage_audit.md` § Cakupan & Keterbatasan untuk insiden yang memotivasi invarian ini: diagnostik leakage tidak bisa menghitung korelasi/VIF/rank efektif penuh karena kedua stream itu tak pernah dipersist oleh run manapun, lokal maupun Colab.
9. Setiap kebijakan fallback/default (nilai pengganti saat data tidak cukup, atau protokol menonaktifkan suatu fitur — mis. rata-rata global di P2, profil item di P3 saat item tanpa aspek terdeteksi, toleransi numerik di test regresi) harus dinyatakan **eksplisit** di kode/komentar berikut alasannya — tidak pernah implisit lewat default parameter yang tak dijelaskan. Kolom diagnostik (mis. `aspect_fallback`/`n_shared_aspects`) juga harus dipisah tegas dari nilai fitur yang sesungguhnya dipakai model, tidak boleh diam-diam ikut mengubah fitur kecuali didokumentasikan sebagai keputusan sadar.
