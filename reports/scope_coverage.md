# Scope Coverage -- Fase 1 Step 3 (pra-Step 4)

> Dihasilkan oleh `tests/test_scope_guard.py::test_coverage_report_written_for_all_domains`.
> Di bawah `HistoricalScope` (protokol deployment-valid): review historis = review
> TRAIN milik user/item tsb (baris eval diambil dari TEST, jadi review test itu
> sendiri tidak pernah termasuk -- tidak perlu LOO eksplisit di sini, beda dgn Test 3
> yang khusus memeriksa baris TRAIN). 500 baris test/domain (`random_state=42`).
> "Aspek beririsan" = irisan set aspek yang DISEBUT (keyword match, bukan skor
> sentimen) di seluruh review historis user U dan seluruh review historis item I;
> baris dgn 0 review historis (user ATAU item) otomatis dihitung sbg 0 aspek
> beririsan (tidak ada basis apa pun).

| Domain | n eval | % 0 review historis user | % 0 review historis item | % 0 aspek beririsan |
|---|---:|---:|---:|---:|
| `amazon_electronics` | 500 | 0.0% | 0.6% | 25.4% |
| `restaurant` | 500 | 0.0% | 0.0% | 0.4% |
| `tripadvisor_hotel` | 500 | 0.0% | 0.2% | 1.2% |

Entitas unik yang benar-benar diproses (keyword matching, bukan disimulasikan):

| Domain | user unik tersentuh | item unik tersentuh |
|---|---:|---:|
| `amazon_electronics` | 492 | 462 |
| `restaurant` | 472 | 420 |
| `tripadvisor_hotel` | 497 | 387 |

**Implikasi utk Step 4.** % 0 aspek beririsan tinggi berarti fitur aspek-personal
(irisan preferensi aspek user x profil aspek item) tidak akan punya sinyal untuk
porsi baris eval sebesar itu di bawah protokol deployment-valid -- P2/P3 perlu
fallback (mis. rata-rata global) utk baris-baris tsb, konsisten dgn desain P2 di
phase1_spec.md ("Untuk baris evaluasi, ganti ... dengan nilai rata-rata train global").
