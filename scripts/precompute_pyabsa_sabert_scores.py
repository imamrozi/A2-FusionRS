"""
scripts/precompute_pyabsa_sabert_scores.py

PRECOMPUTE: skor SA-BERT untuk aspek yang DIEKSTRAK PyABSA, pada dataset
PENUH (train+val+test), di-cache ke disk supaya runner AGF tidak perlu
memuat/menjalankan BERT sama sekali.

LATAR (reports/gates_1_3_summary.md): diagnosis menunjukkan gap PyABSA vs
keyword-ABSA berasal dari SUPERVISI scorer, bukan dari ekstraksi. PyABSA
memakai checkpoint generik; keyword-ABSA memakai `GlobalSentimentBERT`
yang di-fine-tune pada label turunan `stars` domain sendiri. Perbaikannya:
pakai PyABSA untuk EKSTRAKSI aspek (cakupan 77-80% restaurant/e-commerce,
70% hotel) dan SA-BERT per-domain untuk SKORING -- scorer yang SAMA PERSIS
dengan A2-IRM, sehingga perbandingannya adil. Tetap 1x ABSA.

Gerbang-1 memvalidasi pendekatan ini pada subsample 5000/3000: ekstraksi
PyABSA >= leksikon keyword di 3/3 domain saat scorer disetarakan. Skrip ini
memperluasnya ke dataset penuh agar bisa dipakai pipeline sesungguhnya.

OUTPUT (per domain, di checkpoints/{ckpt}/pyabsa/):
  sabert_aspect_scores_{label}.csv   review_id, aspect_term, sabert_score
  sabert_fallback_{label}.csv        review_id, fallback_score
                                     (review tanpa aspek PyABSA sama sekali
                                      -> skor SA-BERT seluruh review)

CATATAN KONSISTENSI: sentence-split & pencocokan istilah aspek meniru
`AspectBasedSentimentBERT._split_sentences` (nltk, minimum 10 karakter)
supaya jalur ekstraksi sebanding dengan keyword-ABSA. Teks dibersihkan
dengan `TextPreprocessor.clean_for_bert`, sama seperti pipeline lain.

BIAYA: ~2,7 aspek/review -> restaurant ~320rb teks. Di GPU Colab dgn
dynamic padding: puluhan menit. Di CPU: berjam-jam -- JALANKAN DI COLAB.

Usage (Colab):
    python scripts/precompute_pyabsa_sabert_scores.py --domain restaurant
    python scripts/precompute_pyabsa_sabert_scores.py          # ketiga domain
    python scripts/precompute_pyabsa_sabert_scores.py --force  # timpa cache
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.a2fusionrs.pyabsa_scorer import load_cached_scores  # noqa: E402
from src.baseline.sentiment_bert import GlobalSentimentBERT, SentimentBertConfig  # noqa: E402
from src.preprocessing import TextPreprocessor, ensure_nltk_resources  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MAX_LENGTH = 256
BATCH_SIZE = 128 if torch.cuda.is_available() else 32
MIN_SENTENCE_CHARS = 10

DOMAINS = {
    "restaurant": dict(split="yelp_restaurant", ckpt="yelp_restaurant", label="restaurant"),
    "amazon_electronics": dict(
        split="amazon_electronics", ckpt="amazon_electronics", label="amazon_electronics"
    ),
    "tripadvisor_hotel": dict(
        split="tripadvisor_hotel", ckpt="tripadvisor_hotel", label="tripadvisor_hotel"
    ),
}


def split_sentences(text: str) -> list[str]:
    import nltk

    if not isinstance(text, str) or not text.strip():
        return []
    return [s for s in nltk.sent_tokenize(text) if len(s.strip()) >= MIN_SENTENCE_CHARS]


@torch.no_grad()
def predict_proba_dynamic(sa: GlobalSentimentBERT, texts: list[str]) -> np.ndarray:
    """Inference batched dgn DYNAMIC PADDING, pengganti
    `GlobalSentimentBERT.predict_proba` yang memakai padding='max_length'
    (memaksa tiap kalimat aspek pendek jadi MAX_LENGTH token).

    Ekuivalensi numerik sudah DIVERIFIKASI di
    `gate_pyabsa_extraction_sabert_scoring.py --verify-fast`: selisih
    absolut maksimum 9,5e-07 pada 512 teks campuran kalimat pendek & review
    panjang. attention_mask menutup posisi padding, jadi output pada token
    nyata tidak dipengaruhi banyaknya padding.

    Teks diurutkan menurut panjang lalu dikembalikan ke urutan asli supaya
    tiap batch berisi teks berpanjangan serupa (padding minimum).
    """
    sa.model.eval()
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    probs = np.empty(len(texts), dtype=np.float32)
    n_batches = (len(order) + BATCH_SIZE - 1) // BATCH_SIZE

    for bi, start in enumerate(range(0, len(order), BATCH_SIZE)):
        idx = order[start : start + BATCH_SIZE]
        enc = sa.tokenizer(
            [texts[i] for i in idx], truncation=True, padding="longest",
            max_length=MAX_LENGTH, return_tensors="pt",
        )
        enc = {k: v.to(sa.config.device) for k, v in enc.items()}
        with torch.autocast(device_type="cuda", enabled=sa.config.use_amp):
            logits = sa.model(**enc).logits
        probs[idx] = torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy()

        if bi % 200 == 0 or bi == n_batches - 1:
            logger.info("  inference batch %d/%d (%.0f%%)", bi + 1, n_batches,
                        100.0 * (bi + 1) / n_batches)
    return probs


def run_domain(domain: str, cfg: dict, force: bool) -> None:
    pdir = _REPO_ROOT / "checkpoints" / cfg["ckpt"] / "pyabsa"
    out_scores = pdir / f"sabert_aspect_scores_{cfg['label']}.csv"
    out_fb = pdir / f"sabert_fallback_{cfg['label']}.csv"

    if out_scores.exists() and out_fb.exists() and not force:
        logger.info("[%s] cache sudah ada -> DILEWATI (pakai --force utk menimpa): %s",
                    domain, out_scores.name)
        return

    logger.info("=" * 70)
    logger.info("DOMAIN: %s", domain)
    logger.info("=" * 70)

    base = _REPO_ROOT / "data" / "splits" / cfg["split"]
    parts = [pd.read_csv(base / f"{s}.csv", usecols=["review_id", "text"])
             for s in ("train", "val", "test")]
    reviews = pd.concat(parts, ignore_index=True).drop_duplicates("review_id")
    logger.info("Total review (train+val+test): %d", len(reviews))

    cache_path = pdir / f"pyabsa_scores_{cfg['label']}.csv"
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Cache PyABSA tidak ditemukan: {cache_path} -- jalankan run_pyabsa_scoring.py dulu."
        )
    scored = load_cached_scores(str(cache_path))
    aspects_by_review = dict(zip(scored["review_id"], scored["aspects"]))
    logger.info("Cache PyABSA dimuat: %d review", len(scored))

    pre = TextPreprocessor()
    flat_texts: list[str] = []
    flat_rid: list = []
    flat_terms: list[str] = []
    fallback_texts: dict = {}

    for row in reviews.itertuples(index=False):
        text_bert = pre.clean_for_bert(row.text)
        terms = aspects_by_review.get(row.review_id, [])
        sentences = split_sentences(text_bert)
        matched_any = False
        for term in terms:
            t = str(term).lower().strip()
            if not t:
                continue
            hits = [s for s in sentences if t in s.lower()]
            if hits:
                flat_texts.append(" ".join(hits))
                flat_rid.append(row.review_id)
                flat_terms.append(t)
                matched_any = True
        if not matched_any:
            fallback_texts[row.review_id] = text_bert

    n_cov = len(reviews) - len(fallback_texts)
    logger.info(
        "Ekstraksi: %d kalimat-aspek dari %d review; cakupan %.1f%% "
        "(%d review jatuh ke fallback seluruh-review)",
        len(flat_texts), len(reviews), 100.0 * n_cov / len(reviews), len(fallback_texts),
    )

    sa_cfg = SentimentBertConfig(max_length=MAX_LENGTH, batch_size=BATCH_SIZE)
    sa = GlobalSentimentBERT.load(
        str(_REPO_ROOT / "checkpoints" / cfg["ckpt"] / "sentiment_bert"), sa_cfg
    )
    logger.info("Device: %s | batch=%d | AMP=%s",
                sa.config.device, BATCH_SIZE, sa.config.use_amp)

    logger.info("Skoring %d kalimat-aspek...", len(flat_texts))
    probs = predict_proba_dynamic(sa, flat_texts) if flat_texts else np.empty(0, np.float32)

    logger.info("Skoring %d teks fallback...", len(fallback_texts))
    fb_rids = list(fallback_texts)
    fb_probs = (
        predict_proba_dynamic(sa, [fallback_texts[r] for r in fb_rids])
        if fb_rids else np.empty(0, np.float32)
    )

    pdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"review_id": flat_rid, "aspect_term": flat_terms, "sabert_score": probs}
    ).to_csv(out_scores, index=False)
    pd.DataFrame(
        {"review_id": fb_rids, "fallback_score": fb_probs}
    ).to_csv(out_fb, index=False)

    logger.info("Disimpan: %s (%d baris)", out_scores.name, len(flat_rid))
    logger.info("Disimpan: %s (%d baris)", out_fb.name, len(fb_rids))

    covered = set(flat_rid) | set(fb_rids)
    missing = set(reviews["review_id"]) - covered
    if missing:
        raise SystemExit(
            f"BERHENTI: {len(missing)} review tidak punya skor aspek MAUPUN fallback. "
            "Setiap review harus menghasilkan minimal satu sinyal."
        )
    logger.info("Verifikasi: 100%% review punya minimal satu sinyal. OK.\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=list(DOMAINS), help="jalankan satu domain saja")
    ap.add_argument("--force", action="store_true", help="timpa cache yang sudah ada")
    ap.add_argument(
        "--quicktest", action="store_true",
        help="pakai split & checkpoint *_quicktest -- untuk verifikasi jalur end-to-end "
        "di CPU lokal SEBELUM menjalankan yang penuh di Colab. Hasilnya TIDAK boleh "
        "dipakai untuk klaim apa pun (dataset mainan).",
    )
    args = ap.parse_args()

    # EKSPLISIT, bukan menumpang efek samping konstruktor TextPreprocessor:
    # `split_sentences()` memanggil nltk.sent_tokenize yang butuh resource
    # 'punkt_tab' di NLTK >= 3.9. Tanpa ini, kegagalan baru muncul di tengah
    # loop ekstraksi (LookupError), setelah dataset selesai dimuat.
    ensure_nltk_resources()

    targets = {args.domain: DOMAINS[args.domain]} if args.domain else DOMAINS
    if args.quicktest:
        targets = {
            d: {**c, "split": f"{c['split']}_quicktest", "ckpt": f"{c['ckpt']}_quicktest"}
            for d, c in targets.items()
        }
        logger.warning(
            "MODE QUICKTEST: memakai split/checkpoint mainan. Hasil HANYA untuk "
            "verifikasi jalur kode, TIDAK sah untuk klaim apa pun."
        )
    for domain, cfg in targets.items():
        run_domain(domain, cfg, args.force)

    logger.info("SELESAI. Cache siap dipakai runner AGF (--extra-pyabsa sabert_*).")


if __name__ == "__main__":
    main()
