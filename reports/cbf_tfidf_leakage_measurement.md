# Pengukuran Leakage TF-IDF CBF (leave-one-out-dalam-train)

> Dihasilkan oleh `scripts/measure_cbf_tfidf_leakage.py`. Mengukur seberapa
> besar profil TF-IDF item (`description_text`, `src/baseline/cbf_clustering.py::
> build_item_dataframe`) berubah kalau review milik baris train yang sedang
> dievaluasi DIKECUALIKAN dari agregat item itu sendiri -- proxy langsung
> utk seberapa besar CBF "mengintip" review targetnya sendiri saat training.
> Metodologi: lihat docstring modul. Basis: TRAIN saja, 500
> baris sampel/domain.

## Hasil per domain

### Domain: `amazon_electronics`

n_train=98400, n_item unik=9200, baris disampel=500
(dari total train, `random_state=42`).

**Shift keseluruhan** (1 - cosine similarity antara profil TF-IDF item NAIF
vs LEAVE-ONE-OUT, dirata-rata semua baris sampel): mean=0.0280,
median=0.0045, p90=0.0587.

52.2% baris sampel berasal dari item dengan
review_count <=15 (rentang di mana efek 1 review paling terasa).

**Shift per rentang review_count item:**

| review_count | n baris sampel | mean shift | median shift | p90 shift |
|---|---:|---:|---:|---:|
| 1-5 | 67 | 0.1067 | 0.0306 | 0.2995 |
| 6-15 | 194 | 0.0317 | 0.0144 | 0.0753 |
| 16-50 | 166 | 0.0040 | 0.0014 | 0.0115 |
| 51+ | 73 | 0.0007 | 0.0001 | 0.0020 |

### Domain: `restaurant`

n_train=95181, n_item unik=3746, baris disampel=500
(dari total train, `random_state=42`).

**Shift keseluruhan** (1 - cosine similarity antara profil TF-IDF item NAIF
vs LEAVE-ONE-OUT, dirata-rata semua baris sampel): mean=0.0069,
median=0.0007, p90=0.0143.

18.0% baris sampel berasal dari item dengan
review_count <=15 (rentang di mana efek 1 review paling terasa).

**Shift per rentang review_count item:**

| review_count | n baris sampel | mean shift | median shift | p90 shift |
|---|---:|---:|---:|---:|
| 1-5 | 13 | 0.0806 | 0.0668 | 0.1965 |
| 6-15 | 77 | 0.0226 | 0.0116 | 0.0517 |
| 16-50 | 182 | 0.0033 | 0.0017 | 0.0072 |
| 51+ | 228 | 0.0003 | 0.0002 | 0.0007 |

### Domain: `tripadvisor_hotel`

n_train=64280, n_item unik=2055, baris disampel=500
(dari total train, `random_state=42`).

**Shift keseluruhan** (1 - cosine similarity antara profil TF-IDF item NAIF
vs LEAVE-ONE-OUT, dirata-rata semua baris sampel): mean=0.0043,
median=0.0005, p90=0.0091.

12.6% baris sampel berasal dari item dengan
review_count <=15 (rentang di mana efek 1 review paling terasa).

**Shift per rentang review_count item:**

| review_count | n baris sampel | mean shift | median shift | p90 shift |
|---|---:|---:|---:|---:|
| 1-5 | 11 | 0.0673 | 0.0351 | 0.1652 |
| 6-15 | 52 | 0.0160 | 0.0113 | 0.0301 |
| 16-50 | 169 | 0.0029 | 0.0015 | 0.0067 |
| 51+ | 268 | 0.0002 | 0.0001 | 0.0006 |


## Ringkasan lintas domain

| Domain | Mean shift (1-cosine) | Median shift | % baris dari item ber-review-count <=15 |
|---|---:|---:|---:|
| `amazon_electronics` | 0.0280 | 0.0045 | 52.2% |
| `restaurant` | 0.0069 | 0.0007 | 18.0% |
| `tripadvisor_hotel` | 0.0043 | 0.0005 | 12.6% |

## Interpretasi

Shift mendekati 0 = profil TF-IDF item PRAKTIS TIDAK BERUBAH kalau review
target dikecualikan (leakage yang terukur dapat diabaikan). Shift mendekati
1 = profil berubah drastis (leakage besar). Bandingkan dgn strata
review_count: kalau shift terkonsentrasi HANYA di item ber-review-count
rendah (efek dilusi -- 1 dari sedikit review = proporsi besar) dan p90/mean
keseluruhan tetap kecil, itu artinya sebagian besar baris (item populer)
PRAKTIS aman, dan risiko nyata terbatas pada ekor sparse (item baru/jarang
direview) -- bukan masalah sistemik di seluruh dataset.
