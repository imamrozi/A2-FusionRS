"""
generate_manuscript_figures.py

Regenerasi 3 figure manuskrip (fig1 hybrid vs classical CF, fig2 RMSE
utama, fig3 dotplot variance per-seed) dari hasil rerun 90-run AdamW
(checkpoints/results/*.yaml) + classical CF pra-fix yang diarsipkan
(checkpoints/results_pre_adamw_fix_2026-07-26/*.yaml -- item-KNN/SVD tidak
memakai DeepMF sama sekali, jadi tidak terpengaruh bug kolaps).

Usage:
    python scripts/generate_manuscript_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

RESULTS_DIR = Path("checkpoints/results")
ARCHIVE_DIR = Path("checkpoints/results_pre_adamw_fix_2026-07-26")
OUT_DIR = Path("manuscript/figures")
SEEDS = [42, 123, 456, 789, 1011]
DOMAINS = ["restaurant", "amazon_electronics", "tripadvisor_hotel"]
DOMAIN_LABELS = {"restaurant": "Restaurant", "amazon_electronics": "E-commerce", "tripadvisor_hotel": "Hotel"}

# Colorblind-safe categorical palette (Okabe-Ito), fixed hue order.
C_BLUE = "#0072B2"
C_ORANGE = "#E69F00"
C_GREEN = "#009E73"
C_RED = "#D55E00"
C_PURPLE = "#CC79A7"
C_GRAY = "#999999"


def load_rmse(prefix: str, domain: str, results_dir: Path = RESULTS_DIR) -> list[float]:
    vals = []
    for seed in SEEDS:
        path = results_dir / f"{prefix}_{domain}_seed{seed}.yaml"
        with open(path) as f:
            d = yaml.safe_load(f)
        vals.append(d["rmse"])
    return vals


def fig1_hybrid_vs_classical_cf() -> None:
    methods = ["item_knn", "svd", "hybrid"]
    method_labels = ["Item-KNN", "SVD", "Adapted hybrid\n(baseline, global SA)"]
    colors = [C_GRAY, C_ORANGE, C_BLUE]

    means, stds = {}, {}
    for domain in DOMAINS:
        means[domain] = []
        stds[domain] = []
        for m in methods:
            if m == "hybrid":
                vals = load_rmse("baseline_reimpl_cbf_nosentiment", domain, RESULTS_DIR)
            else:
                vals = load_rmse(f"classical_cf_{m}", domain, ARCHIVE_DIR)
            means[domain].append(float(np.mean(vals)))
            stds[domain].append(float(np.std(vals, ddof=1)))

    x = np.arange(len(DOMAINS))
    width = 0.25
    fig, ax = plt.subplots(figsize=(6.5, 4.0), dpi=200)
    for i, (m_label, color) in enumerate(zip(method_labels, colors)):
        vals = [means[d][i] for d in DOMAINS]
        errs = [stds[d][i] for d in DOMAINS]
        ax.bar(x + (i - 1) * width, vals, width, yerr=errs, capsize=3,
               label=m_label, color=color, edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([DOMAIN_LABELS[d] for d in DOMAINS])
    ax.set_ylabel("RMSE (lower is better)")
    ax.set_title("Adapted hybrid baseline vs. non-hybrid classical CF\n(mean ± SD over 5 seeds)")
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    out = OUT_DIR / "fig1_hybrid_vs_classical_cf.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"[fig1] saved to {out}")


def fig2_rmse_main_result() -> None:
    variants = [
        ("absa_ablation_concat_confidence_cbf_nosentiment_no_sentiment_fixed_v2", "No-sentiment\nfloor", C_GRAY),
        ("baseline_reimpl_cbf_nosentiment", "Baseline\n(global SA)", C_BLUE),
        ("absa_ablation_cbf_nosentiment", "ABSA\nmean", C_RED),
        ("absa_ablation_confidence_mean_cbf_nosentiment", "ABSA\nconf-mean", C_ORANGE),
        ("absa_ablation_concat_cbf_nosentiment", "ABSA\nconcat", C_PURPLE),
        ("absa_ablation_concat_confidence_cbf_nosentiment", "ABSA\nconcat+conf", C_GREEN),
    ]
    baseline_offset = 1  # index of "Baseline" in variants -- best-variant search starts after it

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.2), dpi=200, sharey=False)
    for ax, domain in zip(axes, DOMAINS):
        means, stds = [], []
        for prefix, _, _ in variants:
            vals = load_rmse(prefix, domain)
            means.append(float(np.mean(vals)))
            stds.append(float(np.std(vals, ddof=1)))
        x = np.arange(len(variants))
        colors = [c for _, _, c in variants]
        bars = ax.bar(x, means, yerr=stds, capsize=3, color=colors, edgecolor="white", linewidth=0.5)
        # highlight the best (lowest-mean) bar with a black outline, if it beats baseline
        best_idx = int(np.argmin(means[baseline_offset + 1:])) + baseline_offset + 1
        if means[best_idx] < means[baseline_offset]:
            bars[best_idx].set_edgecolor("black")
            bars[best_idx].set_linewidth(1.6)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [lbl.replace("\n", " ") for _, lbl, _ in variants],
            fontsize=7.5, rotation=25, ha="right",
        )
        ax.set_title(DOMAIN_LABELS[domain], fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.25, linewidth=0.5)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("RMSE (lower is better)")
    fig.suptitle(
        "RMSE of the adapted hybrid baseline vs. four ABSA sentiment-fusion variants, plus a\n"
        "no-sentiment floor (mean ± SD over 5 seeds; black outline = best variant, where it beats baseline)",
        fontsize=9.5,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    out = OUT_DIR / "fig2_rmse_main_result.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"[fig2] saved to {out}")


def fig3_variance_dotplot() -> None:
    variants = [
        ("baseline_reimpl_cbf_nosentiment", "Baseline", C_BLUE),
        ("absa_ablation_concat_cbf_nosentiment", "Concat", C_PURPLE),
        ("absa_ablation_concat_confidence_cbf_nosentiment", "Concat+conf", C_GREEN),
    ]
    # Extreme seed per domain+model, computed in the manuscript's Table IV analysis.
    extreme_seed = {
        ("restaurant", "baseline_reimpl_cbf_nosentiment"): 789,
        ("restaurant", "absa_ablation_concat_confidence_cbf_nosentiment"): 1011,
        ("amazon_electronics", "baseline_reimpl_cbf_nosentiment"): 1011,
        ("amazon_electronics", "absa_ablation_concat_confidence_cbf_nosentiment"): 1011,
        ("tripadvisor_hotel", "baseline_reimpl_cbf_nosentiment"): 789,
        ("tripadvisor_hotel", "absa_ablation_concat_confidence_cbf_nosentiment"): 123,
    }

    fig, axes = plt.subplots(1, 3, figsize=(11, 4.2), dpi=200, sharey=False)
    for ax, domain in zip(axes, DOMAINS):
        for i, (prefix, label, color) in enumerate(variants):
            vals = load_rmse(prefix, domain)
            xs = np.full(len(vals), i, dtype=float)
            rng = np.random.default_rng(abs(hash((domain, prefix))) % (2**32))
            jitter = rng.uniform(-0.08, 0.08, size=len(vals))
            ax.scatter(xs + jitter, vals, color=color, s=28, zorder=3, alpha=0.85)
            ax.hlines(np.mean(vals), i - 0.18, i + 0.18, color=color, linewidth=2, zorder=4)
            ext_seed = extreme_seed.get((domain, prefix))
            if ext_seed is not None:
                ext_idx = SEEDS.index(ext_seed)
                ax.scatter([xs[ext_idx] + jitter[ext_idx]], [vals[ext_idx]],
                           facecolors="none", edgecolors="black", s=90, linewidths=1.4, zorder=5)
        ax.set_xticks(range(len(variants)))
        ax.set_xticklabels([lbl for _, lbl, _ in variants], fontsize=8.5)
        ax.set_title(DOMAIN_LABELS[domain], fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.25, linewidth=0.5)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("RMSE per seed")
    fig.suptitle(
        "Raw per-seed RMSE (n=5 seeds; horizontal bar = mean) — ringed point marks each\n"
        "domain/model's single most extreme seed (Table IV)",
        fontsize=9.5,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    out = OUT_DIR / "fig3_variance_dotplot.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"[fig3] saved to {out}")


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig1_hybrid_vs_classical_cf()
    fig2_rmse_main_result()
    fig3_variance_dotplot()
