# Aspect Identifiability -- Diagnostik Pra-Fase 2

> Dihasilkan oleh `scripts/audit_aspect_identifiability.py`. Mengukur apakah
> data TRAIN yang ada (k-core sudah difilter, lihat laporan sebelumnya) punya
> cukup sinyal untuk mengidentifikasi preferensi aspek per-user (w_u) --
> SEBELUM waktu diinvestasikan membangun arsitektur yang mengasumsikannya.
> Definisi "mention" dan metodologi tiap section: lihat docstring modul.

## Hasil per domain

### Domain: `amazon_electronics`

n_train = 98400 review, 14749 user unik, 9200 item unik.
Taksonomi aspek nominal (5): quality_durability, price_value, shipping_packaging, ease_of_use, customer_service.

**1. Distribusi total mention aspek per user (train)**

| p10 | p25 | p50 | p75 | p90 |
|---:|---:|---:|---:|---:|
| 0.00 | 1.00 | 3.00 | 6.00 | 10.00 |

**2. Distribusi jumlah aspek UNIK per user (train)**

| p10 | p25 | p50 | p75 | p90 |
|---:|---:|---:|---:|---:|
| 0.00 | 1.00 | 2.00 | 3.00 | 4.00 |

(maksimum mungkin = 5, sama dgn ukuran vocab nominal)

**3. Ukuran kosakata aspek EFEKTIF (bukan nominal)**

Total mention di seluruh train: 66173. Aspek yang menutupi:
- 80% total mention: **4** dari 5 aspek nominal
- 95% total mention: **5** dari 5 aspek nominal

Distribusi share per aspek (diurut menurun):

| Aspek | Total mention | % dari total |
|---|---:|---:|
| price_value | 23300 | 35.2% |
| quality_durability | 15216 | 23.0% |
| ease_of_use | 11107 | 16.8% |
| shipping_packaging | 8358 | 12.6% |
| customer_service | 8192 | 12.4% |

**4. Kunci: user dgn >=3 mention utk >=3 aspek berbeda**

**4.5%** (661/14749) user train memenuhi ambang
ini -- kandidat kasar "punya harapan w_u teridentifikasi per-user". Sisanya
(95.5%) TIDAK cukup data personal utk mengestimasi bobot
5 dimensi aspek secara andal per-user.

**5. Sisi item (pembanding)**

| | p10 | p25 | p50 | p75 | p90 |
|---|---:|---:|---:|---:|---:|
| Total mention per item | 1.00 | 2.00 | 5.00 | 8.00 | 15.00 |
| Aspek unik per item | 1.00 | 2.00 | 3.00 | 4.00 | 5.00 |

**6. Split-half reliability (stabilitas profil frekuensi-aspek user)**

User dgn >=4 review train dibagi acak jadi 2 belahan (`seed=42`), profil =
vektor hitungan mention per aspek per belahan, korelasi Pearson antar-belahan
per user, dirata-rata.

- User diperiksa (>=4 review): 14749
- Dilewati (review train <4): 0
- Dilewati (salah satu belahan varians nol -- korelasi tak terdefinisi): 5831
- **Korelasi rata-rata antar-belahan: 0.364** (median: 0.535, n=8918)

### Domain: `restaurant`

n_train = 95181 review, 7152 user unik, 3746 item unik.
Taksonomi aspek nominal (4): food, service, price, ambiance.

**1. Distribusi total mention aspek per user (train)**

| p10 | p25 | p50 | p75 | p90 |
|---:|---:|---:|---:|---:|
| 6.00 | 8.00 | 13.00 | 23.00 | 50.00 |

**2. Distribusi jumlah aspek UNIK per user (train)**

| p10 | p25 | p50 | p75 | p90 |
|---:|---:|---:|---:|---:|
| 3.00 | 3.00 | 4.00 | 4.00 | 4.00 |

(maksimum mungkin = 4, sama dgn ukuran vocab nominal)

**3. Ukuran kosakata aspek EFEKTIF (bukan nominal)**

Total mention di seluruh train: 168731. Aspek yang menutupi:
- 80% total mention: **3** dari 4 aspek nominal
- 95% total mention: **4** dari 4 aspek nominal

Distribusi share per aspek (diurut menurun):

| Aspek | Total mention | % dari total |
|---|---:|---:|
| food | 68931 | 40.9% |
| service | 44920 | 26.6% |
| price | 30432 | 18.0% |
| ambiance | 24448 | 14.5% |

**4. Kunci: user dgn >=3 mention utk >=3 aspek berbeda**

**48.3%** (3454/7152) user train memenuhi ambang
ini -- kandidat kasar "punya harapan w_u teridentifikasi per-user". Sisanya
(51.7%) TIDAK cukup data personal utk mengestimasi bobot
4 dimensi aspek secara andal per-user.

**5. Sisi item (pembanding)**

| | p10 | p25 | p50 | p75 | p90 |
|---|---:|---:|---:|---:|---:|
| Total mention per item | 5.50 | 10.00 | 22.00 | 52.00 | 108.00 |
| Aspek unik per item | 3.00 | 3.00 | 4.00 | 4.00 | 4.00 |

**6. Split-half reliability (stabilitas profil frekuensi-aspek user)**

User dgn >=4 review train dibagi acak jadi 2 belahan (`seed=42`), profil =
vektor hitungan mention per aspek per belahan, korelasi Pearson antar-belahan
per user, dirata-rata.

- User diperiksa (>=4 review): 7152
- Dilewati (review train <4): 0
- Dilewati (salah satu belahan varians nol -- korelasi tak terdefinisi): 328
- **Korelasi rata-rata antar-belahan: 0.583** (median: 0.739, n=6824)

### Domain: `tripadvisor_hotel`

n_train = 64280 review, 11236 user unik, 2055 item unik.
Taksonomi aspek nominal (6): cleanliness, service, value, location, rooms, sleep_quality.

**1. Distribusi total mention aspek per user (train)**

| p10 | p25 | p50 | p75 | p90 |
|---:|---:|---:|---:|---:|
| 9.00 | 13.00 | 17.00 | 22.00 | 30.00 |

**2. Distribusi jumlah aspek UNIK per user (train)**

| p10 | p25 | p50 | p75 | p90 |
|---:|---:|---:|---:|---:|
| 4.00 | 5.00 | 6.00 | 6.00 | 6.00 |

(maksimum mungkin = 6, sama dgn ukuran vocab nominal)

**3. Ukuran kosakata aspek EFEKTIF (bukan nominal)**

Total mention di seluruh train: 210899. Aspek yang menutupi:
- 80% total mention: **5** dari 6 aspek nominal
- 95% total mention: **6** dari 6 aspek nominal

Distribusi share per aspek (diurut menurun):

| Aspek | Total mention | % dari total |
|---|---:|---:|
| rooms | 57180 | 27.1% |
| service | 43873 | 20.8% |
| location | 37447 | 17.8% |
| cleanliness | 25612 | 12.1% |
| value | 24840 | 11.8% |
| sleep_quality | 21947 | 10.4% |

**4. Kunci: user dgn >=3 mention utk >=3 aspek berbeda**

**71.8%** (8068/11236) user train memenuhi ambang
ini -- kandidat kasar "punya harapan w_u teridentifikasi per-user". Sisanya
(28.2%) TIDAK cukup data personal utk mengestimasi bobot
6 dimensi aspek secara andal per-user.

**5. Sisi item (pembanding)**

| | p10 | p25 | p50 | p75 | p90 |
|---|---:|---:|---:|---:|---:|
| Total mention per item | 15.00 | 26.00 | 57.00 | 137.00 | 247.00 |
| Aspek unik per item | 6.00 | 6.00 | 6.00 | 6.00 | 6.00 |

**6. Split-half reliability (stabilitas profil frekuensi-aspek user)**

User dgn >=4 review train dibagi acak jadi 2 belahan (`seed=42`), profil =
vektor hitungan mention per aspek per belahan, korelasi Pearson antar-belahan
per user, dirata-rata.

- User diperiksa (>=4 review): 11236
- Dilewati (review train <4): 0
- Dilewati (salah satu belahan varians nol -- korelasi tak terdefinisi): 478
- **Korelasi rata-rata antar-belahan: 0.513** (median: 0.586, n=10758)


## Ringkasan lintas domain

| Domain | Vocab efektif @80% | Vocab efektif @95% | % user >=3 mention x >=3 aspek | Korelasi split-half rata-rata |
|---|---:|---:|---:|---:|
| `amazon_electronics` | 4/5 | 5/5 | 4.5% | 0.364 (n=8918) |
| `restaurant` | 3/4 | 4/4 | 48.3% | 0.583 (n=6824) |
| `tripadvisor_hotel` | 5/6 | 6/6 | 71.8% | 0.513 (n=10758) |

**Catatan pembacaan.** Section 4 dan 6 adalah dua sudut pandang atas
pertanyaan yang sama: Section 4 mengukur KECUKUPAN data (cross-sectional,
sekali potong) per user; Section 6 mengukur STABILITAS sinyal itu sendiri
(kalau dibagi dua secara acak, apakah "aspek favorit" user tetap konsisten).
%% qualify tinggi tapi korelasi split-half rendah berarti masalahnya BUKAN
kekurangan data mentah, melainkan preferensi aspek itu sendiri TIDAK stabil/
noise-dominated pada level individual -- dua kesimpulan yang punya implikasi
desain berbeda utk Fase 2.
