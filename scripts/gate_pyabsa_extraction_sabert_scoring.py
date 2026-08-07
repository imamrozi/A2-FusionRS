"""
scripts/gate_pyabsa_extraction_sabert_scoring.py

GERBANG TAHAP 1 (rencana "mengalahkan A2-IRM"): menguji arsitektur usulan
yang DIREVISI -- PyABSA dipakai untuk EKSTRAKSI aspek, SA-BERT (yang sudah
di-fine-tune per domain) dipakai untuk SKORING aspek.

LATAR: `reports/pyabsa_vs_keyword_diagnosis.md` membuktikan gap PyABSA vs
keyword-ABSA berasal dari SUPERVISI scorer, bukan struktur representasi.
Uji 1-dim vs 1-dim (struktur dikonstankan) menunjukkan skor SA-BERT
tersupervisi mengungguli skor PyABSA generik +9,8%/+37,7%/+0,7%, dan
urutannya cocok 3/3 dgn urutan gap faktorial.

PERBAIKAN YANG DIUJI DI SINI: kunci scorer-nya, ganti hanya ekstraksinya.
- Ekstraksi aspek : PyABSA open-vocabulary (cakupan 78/80/70%) -- inilah
                    nilai asli PyABSA, terutama di e-commerce yg leksikon
                    keyword-nya hanya menutup 45,1%.
- Skoring aspek   : GlobalSentimentBERT per-domain -- scorer yang SAMA
                    PERSIS dgn A2-IRM, sehingga perbandingannya adil.
Leksikon keyword statis tetap dihapus; NMF+DT tetap dihapus. Tetap 1x ABSA
(satu ekstraksi, satu scorer).

ATURAN GERBANG
- PRIMER (uji ekstraksi yang ADIL): `pyabsaext_sabert_rich9` <=
  `keyword_rich9` di >= 2 dari 3 domain. Kedua representasi ini
  order-statistics 9-dim berbentuk IDENTIK, sehingga struktur posisi-tetap
  dibuang dari kedua cabang dan yang tersisa murni kualitas EKSTRAKSI.
- SEKUNDER (dilaporkan, bukan penentu): posisi terhadap
  `keyword_concat_conf`, yaitu palang absolut yang harus dilewati
  arsitektur akhir. Cabang PyABSA tidak diharapkan melewatinya DI SINI,
  karena justru AspectSequencePooling di AGF -- yang tidak dipakai di
  gerbang ini -- yang bertugas memulihkan keunggulan struktur.

CATATAN INTEGRITAS: aturan primer di atas adalah REVISI dari aturan awal
(yang membandingkan langsung ke `keyword_concat_conf`). Revisi dibuat
SEBELUM angka pembanding `keyword_rich9` pernah dihitung atau dilihat --
alasannya murni desain: `keyword_concat_conf` punya kolom per-aspek
berposisi tetap sedangkan usulan berbentuk order-statistics, sehingga
perbandingan langsung mencampur perbedaan EKSTRAKSI dgn perbedaan
STRUKTUR dan tidak dapat ditafsirkan. Kedua angka tetap dilaporkan
keduanya, jadi pembaca bisa menilai sendiri.

Kalau gerbang primer gagal: ekstraksi open-vocabulary memang tidak lebih
informatif drpd leksikon terkurasi untuk tugas ini -- temuan yg tetap
layak dilaporkan, tapi bukan kemenangan, dan jam Colab tidak dibakar.

CATATAN VALIDITAS
- Ini diagnostik SUBSAMPLE pada kanal sentimen saja (tanpa DeepMF/CBF/
  fusi). Angkanya TIDAK sebanding dgn RMSE rekomendasi end-to-end; yang
  valid hanya PERBANDINGAN ANTAR-REPRESENTASI, karena semuanya memakai
  subsample, regressor, dan split yang sama.
- `max_length=256` (bukan 512 default) hanya untuk kecepatan gerbang.
  Tokenizer memakai padding="max_length", jadi untuk teks di bawah 256
  token skornya IDENTIK dgn 512 -- kalimat aspek jauh di bawah batas itu.
  Fallback teks review penuh BISA terpotong; itu berlaku sama untuk semua
  representasi berbasis SA-BERT di sini, jadi perbandingannya tetap adil.
- Tidak ada test yang dipakai untuk memilih apa pun: gerbang ini hanya
  memutuskan LANJUT/BERHENTI, dan pemilihan konfigurasi tetap di
  selection_dev sesuai protokol.

Usage:
    venv/Scripts/python.exe scripts/gate_pyabsa_extraction_sabert_scoring.py
    venv/Scripts/python.exe scripts/gate_pyabsa_extraction_sabert_scoring.py --domain tripadvisor_hotel
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.a2fusionrs.pyabsa_scorer import (  # noqa: E402
    load_cached_scores,
    vectorize_absa_features_rich,
)
from src.baseline.sentiment_bert import GlobalSentimentBERT, SentimentBertConfig  # noqa: E402
from src.preprocessing import TextPreprocessor  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

RATING_COL = "stars"
N_TRAIN = 5000
N_TEST = 3000
SAMPLE_SEED = 42
RIDGE_ALPHA = 1.0
GATE_MAX_LENGTH = 256

DOMAINS = {
    "restaurant": dict(split="yelp_restaurant", ckpt="yelp_restaurant", pyabsa="restaurant"),
    "amazon_electronics": dict(
        split="amazon_electronics", ckpt="amazon_electronics", pyabsa="amazon_electronics"
    ),
    "tripadvisor_hotel": dict(
        split="tripadvisor_hotel", ckpt="tripadvisor_hotel", pyabsa="tripadvisor_hotel"
    ),
}
DOMAIN_LABELS = {
    "restaurant": "Restaurant",
    "amazon_electronics": "E-commerce",
    "tripadvisor_hotel": "Hotel",
}

SABERT_RICH_NAMES = [
    "n_aspects_norm", "mean_pos", "min_pos", "max_pos", "range_pos",
    "max_neg", "mean_confidence", "frac_negative", "frac_positive",
]


@__import__("torch").no_grad()
def predict_proba_dynamic(sa: GlobalSentimentBERT, texts: list[str], batch_size: int = 64,
                          max_length: int = GATE_MAX_LENGTH) -> np.ndarray:
    """Inference batched dgn DYNAMIC PADDING (padding='longest' per batch),
    pengganti `GlobalSentimentBERT.predict_proba` yang memakai
    padding='max_length' sehingga tiap kalimat aspek pendek dipaksa jadi
    `max_length` token.

    HASILNYA IDENTIK secara numerik (dalam toleransi float): attention_mask
    menutup posisi padding, jadi output pada token nyata tidak dipengaruhi
    banyaknya padding. Yang berubah hanya jumlah komputasi sia-sia.
    Ekuivalensinya DIVERIFIKASI di `--verify-fast` sebelum dipakai.

    Teks diurutkan berdasar panjang lalu dikembalikan ke urutan asli, supaya
    tiap batch berisi teks berpanjangan serupa (padding minimum).
    """
    import torch

    sa.model.eval()
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    probs = np.empty(len(texts), dtype=np.float32)

    for start in range(0, len(order), batch_size):
        idx = order[start : start + batch_size]
        enc = sa.tokenizer(
            [texts[i] for i in idx], truncation=True, padding="longest",
            max_length=max_length, return_tensors="pt",
        )
        enc = {k: v.to(sa.config.device) for k, v in enc.items()}
        logits = sa.model(**enc).logits.float()
        probs[idx] = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()

    return probs


def _sabert_rich(per_review: dict[int, list[float]], review_ids: list[int],
                 fallback: dict[int, float], evidence_cap: int = 3) -> np.ndarray:
    """Order-statistics 9-dim atas skor SA-BERT per aspek -- struktur kolom
    sengaja DISAMAKAN dgn `vectorize_absa_features_rich` supaya selisih
    terhadap `pyabsa_rich9` murni berasal dari SCORER, bukan bentuk fitur."""
    out = np.zeros((len(review_ids), len(SABERT_RICH_NAMES)), dtype=np.float32)
    for r, rid in enumerate(review_ids):
        scores = per_review.get(rid, [])
        if not scores:
            fb = fallback.get(rid, 0.5)
            out[r] = [0.0, fb, fb, fb, 0.0, 1.0 - fb, 0.0, 0.0, 0.0]
            continue
        s = np.asarray(scores, dtype=np.float32)
        out[r] = [
            min(len(s) / evidence_cap, 1.0),
            float(s.mean()), float(s.min()), float(s.max()),
            float(s.max() - s.min()), float((1.0 - s).max()),
            float(np.abs(s - 0.5).mean() * 2.0),   # confidence = jarak dari ambang
            float((s < 0.5).mean()), float((s >= 0.5).mean()),
        ]
    return out


def _keyword_rich(kw: pd.DataFrame) -> pd.DataFrame:
    """KONTROL LIKE-FOR-LIKE: ringkas keyword-ABSA jadi order-statistics 9-dim
    yang SAMA PERSIS bentuknya dgn `pyabsaext_sabert_rich9`, sehingga struktur
    posisi-tetap dibuang dari KEDUA cabang dan selisihnya murni mencerminkan
    kualitas EKSTRAKSI (leksikon keyword vs PyABSA open-vocabulary).

    Tanpa kontrol ini, `keyword_concat_conf` (punya kolom per-aspek berposisi
    tetap) vs `pyabsaext_sabert_rich9` (order-statistics) mencampur dua
    perbedaan sekaligus dan gerbangnya tidak dapat ditafsirkan.

    Aspek yang MATCH diidentifikasi dari kolom confidence: `absa_bert.py:346`
    mendefinisikan confidence = (sentiment_conf + evidence_conf)/2 dengan
    evidence_conf = min(n_kalimat/3, 1). Untuk aspek TANPA kalimat yang match,
    evidence_conf = 0 sehingga confidence = |skor - 0,5| tepat; aspek yang
    match selalu punya confidence lebih besar dari itu.
    """
    aspects = [c for c in kw.columns if c != "review_id" and not c.endswith("_confidence")]
    rows = np.zeros((len(kw), len(SABERT_RICH_NAMES)), dtype=np.float32)

    scores_all = kw[aspects].to_numpy(np.float32)
    conf_all = kw[[f"{a}_confidence" for a in aspects]].to_numpy(np.float32)
    matched_mask = conf_all > (np.abs(scores_all - 0.5) + 1e-6)

    for r in range(len(kw)):
        s = scores_all[r][matched_mask[r]]
        if s.size == 0:
            fb = float(scores_all[r][0])  # aspek tak-match memuat skor fallback review
            rows[r] = [0.0, fb, fb, fb, 0.0, 1.0 - fb, 0.0, 0.0, 0.0]
            continue
        rows[r] = [
            min(s.size / 3.0, 1.0),
            float(s.mean()), float(s.min()), float(s.max()),
            float(s.max() - s.min()), float((1.0 - s).max()),
            float(np.abs(s - 0.5).mean() * 2.0),
            float((s < 0.5).mean()), float((s >= 0.5).mean()),
        ]
    out = pd.DataFrame(rows, columns=SABERT_RICH_NAMES)
    out.insert(0, "review_id", kw["review_id"].values)
    return out


def _fit_eval(x_tr, y_tr, x_te, y_te) -> dict:
    scaler = StandardScaler().fit(x_tr)
    model = Ridge(alpha=RIDGE_ALPHA).fit(scaler.transform(x_tr), y_tr)
    pred = model.predict(scaler.transform(x_te))
    return {
        "rmse": float(np.sqrt(np.mean((pred - y_te) ** 2))),
        "pearson_r": float(np.corrcoef(pred, y_te)[0, 1]) if np.std(pred) > 1e-12 else 0.0,
        "n_features": x_tr.shape[1],
    }


def run_domain(domain: str, cfg: dict, fast: bool = True) -> list[dict]:
    logger.info("=" * 70)
    logger.info("DOMAIN: %s", DOMAIN_LABELS[domain])
    logger.info("=" * 70)

    base = _REPO_ROOT / "data" / "splits" / cfg["split"]
    train = pd.read_csv(base / "train.csv", usecols=["review_id", RATING_COL, "text"])
    test = pd.read_csv(base / "test.csv", usecols=["review_id", RATING_COL, "text"])
    train = train.sample(n=min(N_TRAIN, len(train)), random_state=SAMPLE_SEED)
    test = test.sample(n=min(N_TEST, len(test)), random_state=SAMPLE_SEED)
    sample = pd.concat([train, test], ignore_index=True)
    logger.info("Subsample: train=%d, test=%d", len(train), len(test))

    pdir = _REPO_ROOT / "checkpoints" / cfg["ckpt"] / "pyabsa"
    scored = load_cached_scores(str(pdir / f"pyabsa_scores_{cfg['pyabsa']}.csv"))
    scored = scored[scored["review_id"].isin(set(sample["review_id"]))].reset_index(drop=True)
    fb_path = pdir / f"sa_fallback_scores_{cfg['pyabsa']}.csv"
    pyabsa_fallback = {}
    if fb_path.exists():
        fb = pd.read_csv(fb_path)
        pyabsa_fallback = dict(zip(fb["review_id"], fb["fallback_score"]))

    # ---- Ekstraksi PyABSA -> kalimat aspek -> skoring SA-BERT ----
    pre = TextPreprocessor()
    aspects_by_review = dict(zip(scored["review_id"], scored["aspects"]))

    flat_texts: list[str] = []
    flat_rid: list[int] = []
    fallback_texts: dict[int, str] = {}

    for row in sample.itertuples(index=False):
        text_bert = pre.clean_for_bert(row.text)
        terms = aspects_by_review.get(row.review_id, [])
        sentences = pre_split(text_bert)
        matched_any = False
        for term in terms:
            t = str(term).lower().strip()
            if not t:
                continue
            hits = [s for s in sentences if t in s.lower()]
            if hits:
                flat_texts.append(" ".join(hits))
                flat_rid.append(row.review_id)
                matched_any = True
        if not matched_any:
            fallback_texts[row.review_id] = text_bert

    logger.info(
        "Ekstraksi PyABSA: %d kalimat-aspek utk %d review; %d review jatuh ke fallback (%.1f%%)",
        len(flat_texts), len(sample), len(fallback_texts),
        100.0 * len(fallback_texts) / len(sample),
    )

    sa_cfg = SentimentBertConfig(max_length=GATE_MAX_LENGTH, batch_size=32, num_workers=0)
    sa = GlobalSentimentBERT.load(
        str(_REPO_ROOT / "checkpoints" / cfg["ckpt"] / "sentiment_bert"), sa_cfg
    )

    score = predict_proba_dynamic if fast else sa.predict_proba

    per_review: dict[int, list[float]] = {}
    if flat_texts:
        probs = score(sa, flat_texts) if fast else score(flat_texts)
        for rid, p in zip(flat_rid, probs):
            per_review.setdefault(rid, []).append(float(p))

    sabert_fallback: dict[int, float] = {}
    if fallback_texts:
        rids = list(fallback_texts)
        fb_texts = [fallback_texts[r] for r in rids]
        fb_probs = score(sa, fb_texts) if fast else score(fb_texts)
        sabert_fallback = {r: float(p) for r, p in zip(rids, fb_probs)}

    # ---- Rakit representasi (semua pada subsample yang sama) ----
    ids = sample["review_id"].tolist()
    new_rich = pd.DataFrame(
        _sabert_rich(per_review, ids, sabert_fallback), columns=SABERT_RICH_NAMES
    )
    new_rich.insert(0, "review_id", ids)

    new_mean = pd.DataFrame(
        {"review_id": ids,
         "mean": [float(np.mean(per_review[r])) if per_review.get(r)
                  else sabert_fallback.get(r, 0.5) for r in ids]}
    )

    sdir = _REPO_ROOT / "checkpoints" / cfg["ckpt"] / "sentiment_bert"
    kw = pd.read_csv(sdir / "absa_concat_confidence_scores.csv")
    glob = pd.read_csv(sdir / "sentiment_scores.csv")

    old_rich = pd.DataFrame(
        vectorize_absa_features_rich(scored, fallback_scores=pyabsa_fallback),
        columns=[f"rich_{i}" for i in range(9)],
    )
    old_rich.insert(0, "review_id", scored["review_id"].values)

    reps = {
        "keyword_concat_conf (A2-IRM)": kw,
        "keyword_rich9 (KONTROL like-for-like)": _keyword_rich(kw),
        "sabert_global_1dim": glob,
        "pyabsa_rich9 (sel C/D0 lama)": old_rich,
        "USULAN pyabsaext_sabert_mean1": new_mean,
        "USULAN pyabsaext_sabert_rich9": new_rich,
    }

    rows = []
    for label, feat in reps.items():
        fcols = [c for c in feat.columns if c != "review_id"]
        tr = train.merge(feat, on="review_id", how="inner")
        te = test.merge(feat, on="review_id", how="inner")
        if tr.empty or te.empty:
            logger.warning("%s: gagal join -- DILEWATI", label)
            continue
        res = _fit_eval(
            tr[fcols].to_numpy(np.float64), tr[RATING_COL].to_numpy(np.float64),
            te[fcols].to_numpy(np.float64), te[RATING_COL].to_numpy(np.float64),
        )
        rows.append({"domain": DOMAIN_LABELS[domain], "representation": label,
                     "n_train": len(tr), "n_test": len(te), **res})
        logger.info("  %-32s dim=%3d  RMSE=%.4f  r=%.4f",
                    label, res["n_features"], res["rmse"], res["pearson_r"])
    return rows


def pre_split(text: str) -> list[str]:
    """Sentence-split identik dgn AspectBasedSentimentBERT._split_sentences
    (nltk, min 10 karakter) supaya jalur ekstraksi sebanding dgn keyword."""
    import nltk

    if not isinstance(text, str) or not text.strip():
        return []
    return [s for s in nltk.sent_tokenize(text) if len(s.strip()) >= 10]


def verify_fast_path(domain: str = "tripadvisor_hotel", n: int = 256) -> None:
    """GERBANG KEBENARAN utk `predict_proba_dynamic`: skor jalur cepat harus
    cocok dgn `GlobalSentimentBERT.predict_proba` (padding='max_length').
    Dijalankan atas campuran kalimat pendek & teks review panjang."""
    cfg = DOMAINS[domain]
    base = _REPO_ROOT / "data" / "splits" / cfg["split"]
    df = pd.read_csv(base / "test.csv", usecols=["text"]).sample(n=n, random_state=0)
    pre = TextPreprocessor()
    texts: list[str] = []
    for t in df["text"]:
        clean = pre.clean_for_bert(t)
        texts.append(clean)                       # teks panjang
        texts.extend(pre_split(clean)[:2])        # kalimat pendek
    texts = [t for t in texts if t.strip()][: n * 2]

    sa_cfg = SentimentBertConfig(max_length=GATE_MAX_LENGTH, batch_size=32, num_workers=0)
    sa = GlobalSentimentBERT.load(
        str(_REPO_ROOT / "checkpoints" / cfg["ckpt"] / "sentiment_bert"), sa_cfg
    )
    ref = sa.predict_proba(texts)
    fast = predict_proba_dynamic(sa, texts)
    max_abs = float(np.max(np.abs(ref - fast)))
    print(f"\nVERIFIKASI jalur cepat pada {len(texts)} teks ({domain}):")
    print(f"  selisih absolut maksimum = {max_abs:.3e}")
    tol = 1e-4
    if max_abs > tol:
        raise SystemExit(
            f"GAGAL: selisih {max_abs:.3e} > toleransi {tol:.0e}. Jalur cepat TIDAK "
            "boleh dipakai -- jalankan dgn --no-fast."
        )
    print(f"  LOLOS (<= {tol:.0e}) -- jalur cepat aman dipakai.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=list(DOMAINS), help="jalankan satu domain saja")
    ap.add_argument("--verify-fast", action="store_true",
                    help="hanya verifikasi ekuivalensi jalur cepat, lalu keluar")
    ap.add_argument("--no-fast", action="store_true",
                    help="pakai predict_proba asli (padding='max_length'), lebih lambat")
    args = ap.parse_args()

    if args.verify_fast:
        verify_fast_path()
        return

    targets = {args.domain: DOMAINS[args.domain]} if args.domain else DOMAINS

    all_rows = []
    for domain, cfg in targets.items():
        all_rows.extend(run_domain(domain, cfg, fast=not args.no_fast))

    out = pd.DataFrame(all_rows)
    dest = _REPO_ROOT / "reports" / "gate_pyabsaext_sabert.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if args.domain and dest.exists():
        prev = pd.read_csv(dest)
        out = pd.concat([prev[prev["domain"] != DOMAIN_LABELS[args.domain]], out], ignore_index=True)
    out.to_csv(dest, index=False)

    print(f"\n{'=' * 70}\nGERBANG: RMSE kanal sentimen saja (subsample)\n{'=' * 70}")
    print(out.pivot_table(index="representation", columns="domain", values="rmse").to_string())

    usul_key = "USULAN pyabsaext_sabert_rich9"
    ctrl_key = "keyword_rich9 (KONTROL like-for-like)"
    abs_key = "keyword_concat_conf (A2-IRM)"

    print(f"\n{'=' * 70}\nPUTUSAN GERBANG -- PRIMER: usulan vs kontrol like-for-like\n{'=' * 70}")
    print("(kedua sisi order-statistics 9-dim; struktur dibuang dari KEDUANYA,\n"
          " jadi selisihnya murni kualitas EKSTRAKSI)\n")
    n_pass, n_total = 0, 0
    for dom in out["domain"].unique():
        d = out[out["domain"] == dom].set_index("representation")["rmse"]
        if usul_key not in d or ctrl_key not in d:
            continue
        usul, ctrl = d[usul_key], d[ctrl_key]
        ok = usul <= ctrl
        n_total += 1
        n_pass += int(ok)
        print(f"  {dom:12} usulan={usul:.4f} vs keyword_rich9={ctrl:.4f}  "
              f"({(usul - ctrl) / ctrl * 100:+.1f}%)  -> {'LOLOS' if ok else 'TIDAK'}")
    print(f"\n  {n_pass}/{n_total} domain lolos. Aturan: >= 2/3 -> LANJUT ke Colab.")
    print(f"  PUTUSAN: {'LANJUT' if n_pass >= 2 else 'BERHENTI -- jangan bakar jam Colab'}")

    print(f"\n{'-' * 70}\nSEKUNDER (dilaporkan, bukan penentu): palang absolut\n{'-' * 70}")
    for dom in out["domain"].unique():
        d = out[out["domain"] == dom].set_index("representation")["rmse"]
        if usul_key not in d or abs_key not in d:
            continue
        print(f"  {dom:12} usulan={d[usul_key]:.4f} vs keyword_concat_conf={d[abs_key]:.4f}  "
              f"({(d[usul_key] - d[abs_key]) / d[abs_key] * 100:+.1f}%)")
    print("  -> selisih di sini adalah beban yang harus ditutup AspectSequencePooling di AGF.")
    print(f"\nDisimpan ke {dest}")


if __name__ == "__main__":
    main()
