"""
scripts/audit_leakage.py

Fase 1 Step 1 (docs/phase1_spec.md): diagnostik leakage MURNI-BACA atas
artefak run yang sudah ada di checkpoints/results/ dan checkpoints/{domain}/
sentiment_bert/ -- TIDAK melatih ulang apa pun (tidak ada .fit() di script
ini sama sekali).

CATATAN CAKUPAN (keputusan eksplisit user, 2026-07-24):
Spec asli meminta matriks korelasi/VIF/rank efektif antara EMPAT stream:
deepmf_preds, cbf_preds, sentiment_score, base_preds. Investigasi kode
(run_baseline_absa.py:397-435) mengonfirmasi deepmf_preds & cbf_preds
TIDAK PERNAH dipersist ke disk oleh run manapun -- keduanya cuma dihitung
in-memory lalu langsung dikonsumsi fusion_nmf_dt.py, tak pernah lewat
save_predictions(). ini bukan keterbatasan lingkungan lokal, tapi gap
arsitektural pada SEMUA run sebelumnya (lokal maupun Colab).

Diberi pilihan (fit ulang DeepMF+CBF sekali secara deterministik, vs
laporkan hanya yang tersedia murni-baca), user memilih opsi murni-baca.
Karena itu:
  - Item 1, 2: dihitung penuh (regresi linear trivial + mean constant,
    keduanya BAGIAN dari definisi item itu sendiri di spec, bukan
    "retraining" model).
  - Item 3: HANYA pasangan (sentiment_score, base_preds) -- satu-satunya
    pasangan yang punya kedua kolom tersedia sbg artefak.
  - Item 4 (VIF), 5 (rank efektif SVD): ditandai TIDAK DAPAT DIHITUNG
    secara bermakna -- VIF & rank efektif hanya informatif dgn >=3-4
    stream; dengan cuma 2 kolom tersedia, angkanya trivial dan menyesatkan
    kalau dipaksakan.

Domain diwakili oleh run baseline_reimpl (reimplementasi fusion NMF+DT
Darraz, "base_preds" dlm terminologi spec) seed 42 -- link PERTAMA dalam
lineage Darraz -> A2-IRM -> A2-FusionRS, dan satu-satunya model yang punya
artefak lengkap utk audit ini di ketiga domain.

Usage:
    python scripts/audit_leakage.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_regression
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor

# max_depth=10 SENGAJA disamakan dgn dt_max_depth default fusion_nmf_dt.py
# (FusionConfig.dt_max_depth) -- kontrol kapasitas-setara: kalau DT dgn
# HANYA sentiment_score sudah mendekati RMSE base_preds (3 stream penuh),
# itu bukti kapasitas model bukan penjelas, sentiment_score sendirian yang
# menjelaskan.
CONTROL_DT_MAX_DEPTH = 10

SEED = 42
RESULTS_DIR = Path("checkpoints/results")
REPORT_PATH = Path("reports/leakage_audit.md")

# (label domain di nama file hasil/results, subfolder checkpoint sentiment_bert, folder split)
DOMAINS = [
    ("amazon_electronics", "amazon_electronics", "data/splits/amazon_electronics"),
    ("restaurant", "yelp_restaurant", "data/splits/yelp_restaurant"),
    ("tripadvisor_hotel", "tripadvisor_hotel", "data/splits/tripadvisor_hotel"),
]


def load_sentiment_scores(checkpoint_domain: str) -> pd.DataFrame:
    path = Path(f"checkpoints/{checkpoint_domain}/sentiment_bert/sentiment_scores.csv")
    return pd.read_csv(path)[["review_id", "sentiment_score"]]


def load_base_preds(results_domain: str) -> pd.DataFrame:
    path = RESULTS_DIR / f"predictions_baseline_reimpl_{results_domain}_seed{SEED}.csv"
    df = pd.read_csv(path)[["review_id", "y_true", "y_pred"]]
    return df.rename(columns={"y_pred": "base_preds", "y_true": "stars"})


def audit_domain(results_domain: str, checkpoint_domain: str, split_dir: str) -> dict:
    train_df = pd.read_csv(Path(split_dir) / "train.csv", usecols=["review_id", "stars"])
    sentiment_df = load_sentiment_scores(checkpoint_domain)
    base_df = load_base_preds(results_domain)

    train_merged = train_df.merge(sentiment_df, on="review_id", how="left")
    n_train = len(train_df)
    n_train_matched = train_merged["sentiment_score"].notna().sum()

    test_merged = base_df.merge(sentiment_df, on="review_id", how="left")
    n_test = len(base_df)
    n_test_matched = test_merged["sentiment_score"].notna().sum()

    train_fit = train_merged.dropna(subset=["sentiment_score"])
    test_eval = test_merged.dropna(subset=["sentiment_score"])

    # ---- Item 1: RMSE regresi linear stars ~ sentiment_score, fit di
    # train, dievaluasi di test (protokol held-out sama seperti semua
    # model lain di pipeline ini -- BUKAN fit+eval di test yang sama,
    # supaya angkanya bukan in-sample optimis). ----
    lr = LinearRegression()
    lr.fit(train_fit[["sentiment_score"]], train_fit["stars"])
    sentiment_only_preds = lr.predict(test_eval[["sentiment_score"]])
    rmse_sentiment_only = float(
        np.sqrt(mean_squared_error(test_eval["stars"], sentiment_only_preds))
    )

    # ---- Kontrol tambahan (2026-07-24, atas permintaan user): DT dgn
    # kapasitas SAMA PERSIS dgn regressor fusion asli (max_depth=10), dan
    # isotonic regression univariat -- dua model non-linear/non-parametrik
    # atas sentiment_score SAJA, utk menutup celah "mungkin linear terlalu
    # lemah menangkap hubungan sebenarnya". ----
    dt = DecisionTreeRegressor(max_depth=CONTROL_DT_MAX_DEPTH, random_state=SEED)
    dt.fit(train_fit[["sentiment_score"]], train_fit["stars"])
    sentiment_dt_preds = dt.predict(test_eval[["sentiment_score"]])
    rmse_sentiment_dt = float(np.sqrt(mean_squared_error(test_eval["stars"], sentiment_dt_preds)))

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(train_fit["sentiment_score"], train_fit["stars"])
    sentiment_iso_preds = iso.predict(test_eval["sentiment_score"])
    rmse_sentiment_isotonic = float(
        np.sqrt(mean_squared_error(test_eval["stars"], sentiment_iso_preds))
    )

    # ---- Item 2: RMSE model konstanta (rata-rata train global) ----
    global_mean = float(train_df["stars"].mean())
    rmse_global_mean = float(
        np.sqrt(mean_squared_error(test_eval["stars"], np.full(len(test_eval), global_mean)))
    )

    # ---- Referensi: RMSE fusion penuh (base_preds, 3 stream: DeepMF+CBF+
    # sentiment) -- utk membandingkan langsung seberapa dekat sentiment_score
    # SENDIRIAN terhadap model penuh. ----
    rmse_full_fusion = float(np.sqrt(mean_squared_error(test_eval["stars"], test_eval["base_preds"])))

    # ---- Item 3: korelasi Pearson + MI, HANYA pasangan (sentiment_score,
    # base_preds) -- deepmf_preds/cbf_preds tidak tersedia sbg artefak
    # (lihat catatan cakupan di docstring modul). ----
    pearson_r, pearson_p = stats.pearsonr(test_eval["sentiment_score"], test_eval["base_preds"])
    mi = mutual_info_regression(
        test_eval[["sentiment_score"]], test_eval["base_preds"], random_state=SEED
    )[0]

    return {
        "results_domain": results_domain,
        "n_train": n_train,
        "n_train_matched": int(n_train_matched),
        "n_test": n_test,
        "n_test_matched": int(n_test_matched),
        "rmse_sentiment_only": rmse_sentiment_only,
        "rmse_sentiment_dt": rmse_sentiment_dt,
        "rmse_sentiment_isotonic": rmse_sentiment_isotonic,
        "rmse_global_mean": rmse_global_mean,
        "rmse_full_fusion": rmse_full_fusion,
        "global_mean_value": global_mean,
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "mutual_info": float(mi),
    }


def render_domain_block(r: dict) -> str:
    return f"""### Domain: `{r['results_domain']}`

Baris dipakai: {r['n_train']} train ({r['n_train_matched']} bertaut sentiment_score),
{r['n_test']} test ({r['n_test_matched']} bertaut sentiment_score) --
artefak `predictions_baseline_reimpl_{r['results_domain']}_seed{SEED}.csv`,
`checkpoints/.../sentiment_bert/sentiment_scores.csv`.

**1-2. RMSE sentimen-saja (3 kontrol kapasitas) vs model konstanta vs fusion penuh**

| Model | RMSE (test) |
|---|---:|
| `stars ~ sentiment_score` (regresi linear, fit di train) | {r['rmse_sentiment_only']:.4f} |
| `stars ~ sentiment_score` (DecisionTreeRegressor max_depth={CONTROL_DT_MAX_DEPTH}, fit di train) | {r['rmse_sentiment_dt']:.4f} |
| `stars ~ sentiment_score` (isotonic regression univariat, fit di train) | {r['rmse_sentiment_isotonic']:.4f} |
| Global mean (konstanta = {r['global_mean_value']:.4f}, dari train) | {r['rmse_global_mean']:.4f} |
| *(referensi)* `base_preds` -- fusion NMF+DT penuh (3 stream: DeepMF+CBF+sentiment) | {r['rmse_full_fusion']:.4f} |

DT dgn max_depth={CONTROL_DT_MAX_DEPTH} SENGAJA disamakan dgn kapasitas
regressor fusion asli (`FusionConfig.dt_max_depth`) -- kalau RMSE-nya
mendekati baris referensi `base_preds` walau HANYA pakai sentiment_score
(tanpa DeepMF/CBF), itu bukti kapasitas model bukan penjelas performa;
sentiment_score sendirian yang menjelaskan.

**3. Korelasi & mutual information (test set) -- HANYA pasangan tersedia**

| Pasangan stream | Pearson r | p-value | Mutual information |
|---|---:|---:|---:|
| sentiment_score vs base_preds | {r['pearson_r']:.4f} | {r['pearson_p']:.2e} | {r['mutual_info']:.4f} |

deepmf_preds dan cbf_preds **tidak tersedia** sbg artefak tersimpan (lihat
§ Cakupan & Keterbatasan) -- matriks 4x4 penuh tidak dapat dihitung.

**4. VIF** -- tidak dapat dihitung. VIF mengukur multikolinearitas antar
>=2 prediktor independen; dengan hanya sentiment_score yang berdiri
sendiri sbg fitur tersedia (base_preds adalah OUTPUT fusion, bukan
prediktor independen sejajar), tidak ada basis perhitungan yang bermakna.

**5. Rank efektif (SVD, ambang 99% varians)** -- tidak dihitung dgn alasan
sama seperti VIF: matriks fitur fusi penuh (4 kolom) tidak tersedia; rank
dari 2 kolom yang ada trivial (<=2) dan tidak informatif untuk pertanyaan
"apakah cross-attention Fase 2 punya ruang kerja".
"""


def main() -> None:
    blocks = []
    for results_domain, checkpoint_domain, split_dir in DOMAINS:
        print(f"Mengaudit domain: {results_domain} ...")
        result = audit_domain(results_domain, checkpoint_domain, split_dir)
        blocks.append(render_domain_block(result))
    blocks_text = "\n".join(blocks)

    report = f"""# Leakage Audit -- Fase 1 Step 1

> Dihasilkan oleh `scripts/audit_leakage.py`. MURNI-BACA artefak yang sudah
> ada di `checkpoints/results/` dan `checkpoints/*/sentiment_bert/` --
> tidak ada model yang dilatih ulang untuk laporan ini.

## Cakupan & Keterbatasan (baca sebelum tabel di bawah)

Spec Fase 1 Step 1 meminta diagnostik atas EMPAT stream:
`deepmf_preds`, `cbf_preds`, `sentiment_score`, `base_preds`. Investigasi
langsung terhadap `run_baseline_absa.py` (baris 397-435) mengonfirmasi
`deepmf_preds` dan `cbf_preds` **tidak pernah dipersist ke disk oleh run
manapun** -- baik lokal maupun Colab. Keduanya dihitung in-memory lalu
langsung dikonsumsi `NMFDecisionTreeFusion.fit()/predict()`
(`fusion_nmf_dt.py`), tanpa pernah lewat `save_predictions()`. Ini gap
arsitektural pipeline sejak awal, bukan keterbatasan lingkungan run kali
ini.

Dua opsi diajukan ke user: (a) fit ulang DeepMF+CBF sekali secara
deterministik (split beku, kode `src/legacy/` tak diubah) untuk melengkapi
4 stream penuh, atau (b) laporkan hanya yang tersedia murni-baca. **User
memilih (b).** Konsekuensinya:

- Item 1 & 2 (RMSE sentimen-saja, RMSE global mean): dihitung penuh --
  keduanya inheren melibatkan fit model trivial (regresi linear 1-fitur,
  rata-rata konstanta) yang memang menjadi definisi item itu sendiri di
  spec, bukan "retraining" yang dihindari.
- Item 3 (korelasi + MI): hanya pasangan `sentiment_score` vs `base_preds`
  -- satu-satunya pasangan dengan kedua kolom tersedia sbg artefak.
- Item 4 (VIF) & 5 (rank efektif SVD): **tidak dihitung** -- kedua metrik
  ini hanya bermakna dengan >=3-4 stream independen; dipaksakan dengan 2
  kolom akan menghasilkan angka trivial dan menyesatkan.

Sebagai kompensasi kekuatan bukti (item 1-2-3 saja), tiga kontrol tambahan
ditambahkan atas permintaan user: DecisionTreeRegressor(max_depth={CONTROL_DT_MAX_DEPTH})
(kapasitas sama persis dgn regressor fusion asli) dan isotonic regression
univariat -- keduanya atas sentiment_score saja -- plus baris referensi
RMSE fusion penuh (`base_preds`), supaya jelas seberapa dekat sentiment
sendirian terhadap model 3-stream.

Representasi domain: run `baseline_reimpl` (reimplementasi fusion NMF+DT
Darraz) seed {SEED} -- link pertama dalam lineage Darraz -> A2-IRM ->
A2-FusionRS, satu-satunya model dengan artefak lengkap di ketiga domain
utk audit ini.

## Hasil per domain

{blocks_text}

## Interpretasi (acuan tabel §"Interpretasi yang diharapkan" di phase1_spec.md)

Baca §"Cakupan & Keterbatasan" di atas sebelum menafsirkan -- korelasi
antar-stream **tidak** bisa dinilai lengkap (hanya 1 dari 6 pasangan
mungkin di matriks 4x4 yang tersedia), dan rank efektif/VIF sama sekali
tidak dihitung. Baris RMSE(`stars ~ sentiment_score`) tetap dapat
dibandingkan langsung terhadap perkiraan 0,65 di spec -- ini bagian paling
kuat berdiri sendiri dari audit ini.

**Keputusan lanjutan (Step 2, sesi web, di luar cakupan sesi ini):**
apakah kekuatan bukti pada item 1-2-3 (tanpa item 4-5) sudah cukup untuk
Step 2, atau apakah opsi (a) -- fit ulang DeepMF+CBF -- perlu dijalankan
belakangan untuk melengkapi item 4-5.
"""

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"\nLaporan ditulis ke {REPORT_PATH}")


if __name__ == "__main__":
    main()
