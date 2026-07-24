# Aspect Diversity -- Diagnostik Pra-Fase 2 (kandidat variabel moderator)

> Dihasilkan oleh `scripts/audit_aspect_diversity.py`. Sumber: cache PyABSA
> open-vocabulary (`aspects_json`), BUKAN taksonomi keyword 4-6-kategori
> tetap yang dipakai `aspect_identifiability.md`. Basis: TRAIN saja.
> Normalisasi vocab RINGAN (case/whitespace/lemma), BUKAN clustering
> sinonim penuh -- lihat docstring modul untuk detail & keterbatasan.

## Hasil per domain

### Domain: `amazon_electronics`

Review train tercakup cache PyABSA: 98400. Total mention
aspek (setelah normalisasi, sebelum dedup lintas-review): 170302.

| Metrik | Nilai |
|---|---:|
| Jumlah aspek unik (setelah normalisasi vocab) | 18223 |
| Entropi Shannon distribusi aspek | 9.86 bit (maks teoritis 14.15 bit = 69.7% dari maks) |
| Gini coefficient | 0.855 |
| Rata-rata aspek unik per review | 1.73 |
| Median aspek unik per review | 1.0 |

**Top 20 aspek terbanyak:**

| # | Aspek (dinormalisasi) | Mention | % dari total |
|---:|---|---:|---:|
| 1 | work | 8130 | 4.77% |
| 2 | price | 7854 | 4.61% |
| 3 | cable | 4138 | 2.43% |
| 4 | sound | 3843 | 2.26% |
| 5 | fit | 2379 | 1.40% |
| 6 | battery | 2153 | 1.26% |
| 7 | case | 2074 | 1.22% |
| 8 | quality | 1961 | 1.15% |
| 9 | drive | 1789 | 1.05% |
| 10 | use | 1598 | 0.94% |
| 11 | install | 1470 | 0.86% |
| 12 | cord | 1279 | 0.75% |
| 13 | lens | 1207 | 0.71% |
| 14 | screen | 1165 | 0.68% |
| 15 | sound quality | 1151 | 0.68% |
| 16 | picture | 1147 | 0.67% |
| 17 | keyboard | 1142 | 0.67% |
| 18 | speed | 1103 | 0.65% |
| 19 | size | 1070 | 0.63% |
| 20 | performance | 963 | 0.57% |

### Domain: `restaurant`

Review train tercakup cache PyABSA: 95181. Total mention
aspek (setelah normalisasi, sebelum dedup lintas-review): 248977.

| Metrik | Nilai |
|---|---:|
| Jumlah aspek unik (setelah normalisasi vocab) | 14759 |
| Entropi Shannon distribusi aspek | 8.49 bit (maks teoritis 13.85 bit = 61.3% dari maks) |
| Gini coefficient | 0.912 |
| Rata-rata aspek unik per review | 2.62 |
| Median aspek unik per review | 3.0 |

**Top 20 aspek terbanyak:**

| # | Aspek (dinormalisasi) | Mention | % dari total |
|---:|---|---:|---:|
| 1 | food | 21895 | 8.79% |
| 2 | service | 17922 | 7.20% |
| 3 | price | 8580 | 3.45% |
| 4 | place | 6750 | 2.71% |
| 5 | staff | 6434 | 2.58% |
| 6 | atmosphere | 6232 | 2.50% |
| 7 | pizza | 3870 | 1.55% |
| 8 | they | 3780 | 1.52% |
| 9 | server | 3294 | 1.32% |
| 10 | it | 2984 | 1.20% |
| 11 | drink | 2955 | 1.19% |
| 12 | portion | 2421 | 0.97% |
| 13 | burger | 2369 | 0.95% |
| 14 | wait | 2299 | 0.92% |
| 15 | salad | 2042 | 0.82% |
| 16 | waitress | 1999 | 0.80% |
| 17 | menu | 1907 | 0.77% |
| 18 | fry | 1892 | 0.76% |
| 19 | $ | 1857 | 0.75% |
| 20 | beer | 1845 | 0.74% |

### Domain: `tripadvisor_hotel`

Review train tercakup cache PyABSA: 64280. Total mention
aspek (setelah normalisasi, sebelum dedup lintas-review): 171900.

| Metrik | Nilai |
|---|---:|
| Jumlah aspek unik (setelah normalisasi vocab) | 6275 |
| Entropi Shannon distribusi aspek | 6.90 bit (maks teoritis 12.62 bit = 54.7% dari maks) |
| Gini coefficient | 0.938 |
| Rata-rata aspek unik per review | 2.67 |
| Median aspek unik per review | 3.0 |

**Top 20 aspek terbanyak:**

| # | Aspek (dinormalisasi) | Mention | % dari total |
|---:|---|---:|---:|
| 1 | room | 27651 | 16.09% |
| 2 | staff | 16830 | 9.79% |
| 3 | service | 9814 | 5.71% |
| 4 | location | 7113 | 4.14% |
| 5 | bed | 4483 | 2.61% |
| 6 | price | 4443 | 2.58% |
| 7 | breakfast | 4246 | 2.47% |
| 8 | bathroom | 3711 | 2.16% |
| 9 | food | 3057 | 1.78% |
| 10 | lobby | 2668 | 1.55% |
| 11 | rate | 2450 | 1.43% |
| 12 | pool | 2401 | 1.40% |
| 13 | view | 2280 | 1.33% |
| 14 | area | 2072 | 1.21% |
| 15 | stay | 1864 | 1.08% |
| 16 | they | 1802 | 1.05% |
| 17 | place | 1736 | 1.01% |
| 18 | decor | 1253 | 0.73% |
| 19 | bar | 1134 | 0.66% |
| 20 | hotel | 1117 | 0.65% |


## Ringkasan lintas domain

| Domain | Aspek unik | Entropi (bit) | Gini | Rata-rata aspek/review |
|---|---:|---:|---:|---:|
| `amazon_electronics` | 18223 | 9.86 | 0.855 | 1.73 |
| `restaurant` | 14759 | 8.49 | 0.912 | 2.62 |
| `tripadvisor_hotel` | 6275 | 6.90 | 0.938 | 2.67 |

**Catatan pembacaan.** Entropi tinggi + Gini rendah = kosakata aspek dipakai
merata (kandidat baik utk moderator "domain aspek-beragam"). Gini tinggi
berarti segelintir aspek mendominasi mention walau kosakata NOMINAL luas --
lebih dekat ke taksonomi 4-6-kategori `aspect_identifiability.md` secara
efektif, meski secara nominal kosakata PyABSA jauh lebih besar.

**Catatan kualitas data (diobservasi, tidak difilter diam-diam).** Beberapa
entri top-20 adalah noise ekstraksi ATEPC, bukan aspek sungguhan -- mis.
`they`/`it` (restaurant, hotel) dan simbol `$` (restaurant). Angka
laporan ini APA ADANYA dari cache PyABSA (tidak ada post-filtering manual)
supaya tidak menyembunyikan karakteristik nyata pipeline ekstraksi; kalau
metrik ini dipakai lebih jauh (mis. jadi fitur moderator), pembersihan
stopword/pronoun eksplisit perlu ditambahkan sbg langkah terpisah yang
didokumentasikan.
