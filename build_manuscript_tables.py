"""
build_manuscript_tables.py

Ekstrak nilai numerik Tabel 2, 4, 9, 10 manuskrip A2-FusionRS dari file hasil
yang sudah ada (checkpoints/results/ + data/splits/). Tidak menjalankan model;
hanya membaca YAML/CSV/split. Jalankan lokal:

    python build_manuscript_tables.py
"""
from __future__ import annotations
import yaml, numpy as np, pandas as pd
from pathlib import Path

RD = Path("checkpoints/results")
SPLITS = Path("data/splits")
SEEDS = [42, 123, 456, 789, 1011]
DOMAINS = ["amazon_electronics", "restaurant", "tripadvisor_hotel"]
SPLIT_DIR = {"amazon_electronics": "amazon_electronics", "restaurant": "yelp_restaurant",
             "tripadvisor_hotel": "tripadvisor_hotel"}


def _cov_summary():
    for p in [RD / "pyabsa_coverage_summary.csv", RD / "agf" / "pyabsa_coverage_summary.csv"]:
        if p.exists():
            return pd.read_csv(p).set_index("domain")
    return None


def table2():
    from src.split_generator import UserBasedSplitGenerator
    cov = _cov_summary()
    print("\n### Table 2. Dataset statistics")
    print(f"{'Domain':20s}{'#users':>9s}{'#items':>9s}{'#inter':>10s}{'sparsity':>10s}"
          f"{'kw_cov':>9s}{'py_cov':>9s}{'asp/rev':>9s}")
    for dom in DOMAINS:
        sp = UserBasedSplitGenerator.load(SPLITS / SPLIT_DIR[dom])
        df = pd.concat([sp["train"], sp["val"], sp["test"]], ignore_index=True)
        nu, ni, n = df["user_id"].nunique(), df["business_id"].nunique(), len(df)
        spars = 1 - n / (nu * ni)
        # keyword coverage dari YAML A2-IRM
        kw = None
        y = RD / f"absa_ablation_concat_confidence_{dom}_seed42.yaml"
        if y.exists():
            kw = yaml.safe_load(open(y)).get("aspect_coverage", {}).get("pct_with_any_aspect_match")
        pc = ac = None
        if cov is not None and dom in cov.index:
            pc = cov.loc[dom, "pct_with_any_aspect"]; ac = cov.loc[dom, "avg_aspects_per_review"]
        print(f"{dom:20s}{nu:>9,d}{ni:>9,d}{n:>10,d}{spars*100:>9.2f}%"
              f"{(kw*100 if kw else float('nan')):>8.1f}%{(pc if pc else float('nan')):>8.1f}%{ac if ac else float('nan'):>9.2f}")


def _agg_metrics(prefix):
    out = {}
    for dom in DOMAINS:
        m = {"rmse": [], "mae": [], "p5": [], "r5": [], "n5": []}
        for s in SEEDS:
            p = RD / f"{prefix}_{dom}_seed{s}.yaml"
            if not p.exists():
                continue
            y = yaml.safe_load(open(p))
            m["rmse"].append(y.get("rmse")); m["mae"].append(y.get("mae"))
            for key, dst in [("precision_at_k", "p5"), ("recall_at_k", "r5"), ("ndcg_at_k", "n5")]:
                d = y.get(key, {}) or {}
                m[dst].append(d.get(5, d.get("5")))
        out[dom] = {k: (np.nanmean([x for x in v if x is not None]) if any(x is not None for x in v) else None)
                    for k, v in m.items()}
    return out


def table4():
    models = [("Item-KNN", "classical_cf_item_knn"), ("SVD", "classical_cf_svd"),
              ("NeuMF", "neural_cf_neumf"), ("DeepFM", "neural_cf_deepfm"),
              ("A2-IRM", "absa_ablation_concat_confidence"),
              ("A2-FusionRS", "agf_agf_keyword_oof_perseq")]
    print("\n### Table 4. RMSE / MAE / P@5 / R@5 / NDCG@5 (mean over 5 seeds)")
    for name, pfx in models:
        r = _agg_metrics(pfx)
        print(f"\n{name}:")
        for dom in DOMAINS:
            x = r[dom]; f = lambda v: f"{v:.4f}" if v is not None else "  -  "
            print(f"  {dom:20s} RMSE={f(x['rmse'])} MAE={f(x['mae'])} "
                  f"P@5={f(x['p5'])} R@5={f(x['r5'])} NDCG@5={f(x['n5'])}")


def table10():
    print("\n### Table 10. Efficiency (baselines seed 42; A2-FusionRS mean over 5 seeds)")
    print(f"{'Model':14s}{'Domain':20s}{'params':>13s}{'train(s)':>10s}{'predict(ms)':>13s}")

    def row(name, pfx, seeds):
        for dom in DOMAINS:
            np_, tt, pt = [], [], []
            for s in seeds:
                p = RD / f"{pfx}_{dom}_seed{s}.yaml"
                if not p.exists():
                    continue
                y = yaml.safe_load(open(p))
                np_.append(y.get("n_parameters")); tt.append(y.get("train_time_seconds")); pt.append(y.get("predict_time_seconds"))
            f = lambda v: (np.nanmean([x for x in v if x is not None]) if any(x is not None for x in v) else None)
            npar, t, p_ = f(np_), f(tt), f(pt)
            fs = lambda x, fm: (fm.format(x) if x is not None else "   —   ")
            print(f"{name:14s}{dom:20s}{fs(npar,'{:,.0f}'):>13s}{fs(t,'{:.1f}'):>10s}"
                  f"{fs(p_*1000 if p_ is not None else None,'{:.1f}'):>13s}")

    for name, pfx in [("Item-KNN", "classical_cf_item_knn"), ("SVD", "classical_cf_svd"),
                      ("NeuMF", "neural_cf_neumf"), ("DeepFM", "neural_cf_deepfm")]:
        row(name, pfx, [42])
    row("A2-FusionRS", "agf_agf_keyword_oof_perseq", SEEDS)


if __name__ == "__main__":
    table2()
    table4()
    table10()
    print("\n(Table 9 case studies via analyze_interpretability.py --results-dir checkpoints/results)")
