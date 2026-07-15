# A2-IRM: An Aspect-Aware Integrated Representation Model for Cross-Domain Hybrid Recommender Systems

**Authors:** Imam Fahrur Rozi¹, Triyanna Widiyaningtyas¹, Didik Dwi Prasetya¹, Andriana Kusuma Dewi¹, Rahmawati Febrifyaning Tias¹, Deshinta Arrova Dewi²

¹ Department of Electrical Engineering and Informatics, Universitas Negeri Malang, Malang, Indonesia
² Faculty of Data Science and Information Technology, INTI International University, Nilai, Malaysia

*Draft prepared for internal review — author order, affiliations, and funding acknowledgment (UM Dana Internal Penelitian Desentralisasi 2026) to be confirmed before submission. Condensed 6-page conference version of `A2-IRM_manuscript_draft.md` — see that file for the full-length version with the complete stability/variance deep-dive (Table IV, Fig. 3) and extended Discussion.*

---

## Abstract

Hybrid recommender systems that integrate collaborative filtering, content-based filtering, and sentiment analysis have consistently outperformed single-technique baselines, but the sentiment signal in most such systems remains a single global polarity score per review — a representation that discards exactly the aspect-level nuance (food versus service, price versus durability) that makes review text informative in the first place. We reimplement the hybrid architecture of Darraz et al. (deep matrix factorization, K-means/agglomerative content-based clustering, and NMF–DecisionTreeRegressor feature fusion) as a faithful baseline, then replace its global BERT sentiment score with an aspect-based sentiment analysis (ABSA) representation and evaluate four alternative fusion strategies for injecting it into the pipeline: naive mean aggregation across matched aspects, confidence-weighted mean aggregation, raw per-aspect score concatenation, and per-aspect score concatenation augmented with explicit per-aspect confidence features. All five configurations are evaluated on three structurally distinct domains — Yelp restaurant reviews, Amazon Electronics reviews, and TripAdvisor hotel reviews — under an identical protocol (5 seeds, paired Wilcoxon significance testing per seed with Fisher-combined p-values). The two aggregation-based variants degrade RMSE substantially and consistently across all three domains (12.9–21.7% relative increase, 5/5 seeds significant), confirming that flattening multi-aspect sentiment into a single scalar destroys predictive signal rather than merely diluting it. Concatenating raw per-aspect scores without aggregation restores near-baseline parity. Critically, adding per-aspect confidence as an explicit auxiliary feature — rather than using it to weight an aggregate — yields the only configuration that significantly *improves* on the baseline in every domain (RMSE reductions of 2.0–3.2%, 4/5 to 5/5 seeds significant), while simultaneously reducing run-to-run variance by 4–10×. This pattern replicates with consistent direction and comparable effect size across three domains with markedly different aspect-keyword coverage (45.1% to 95.9% of reviews matching at least one aspect keyword), which we argue is evidence that the mechanism — not a dataset-specific artifact — is responsible for the improvement. We term this validated tri-modal representation A2-IRM (Aspect-Aware Integrated Representation Model) and position it as the empirical foundation for a subsequent attention-gated fusion architecture (A2-FusionRS).

**Keywords:** aspect-based sentiment analysis; hybrid recommender system; BERT; deep matrix factorization; content-based filtering; cross-domain evaluation; feature fusion

---

## I. Introduction

Recommender systems that rely solely on the numeric rating in a user–item interaction discard a large fraction of the information a review actually contains. A one-star hotel review complaining exclusively about noisy air conditioning and a one-star review complaining about rude staff carry the same numeric signal but very different implications for what the system should recommend next. This observation has motivated a substantial body of work on integrating sentiment analysis into collaborative and content-based filtering [1]–[3], most recently through pretrained transformer language models such as BERT [23], which substantially outperform lexicon-based sentiment tools on review text [4], [5].

Darraz et al. [6] recently proposed a hybrid architecture that integrates a fine-tuned BERT sentiment classifier, deep matrix factorization (DeepMF) for collaborative filtering, and K-means/agglomerative clustering for content-based filtering, combining the three signals through non-negative matrix factorization (NMF) followed by a DecisionTreeRegressor. This design is representative of a broader pattern in the literature [7]–[12]: sentiment analysis is computed once per review as a single global polarity score, then fused with collaborative and content-based signals as one additional feature. We argue this design carries a structural cost independent of how well the underlying sentiment classifier performs: a review can express positive sentiment toward one aspect of an item and negative sentiment toward another, and collapsing this into one scalar necessarily discards information. As we show empirically, the specific manner in which that collapse happens determines whether the resulting feature helps or actively harms downstream rating prediction — naive aggregation does not just fail to add value, it degrades RMSE substantially and consistently across three domains.

This study is part of a longer-running research program on similarity- and factorization-based recommendation [18]–[21] and aspect-based sentiment analysis under noisy, imbalanced review data [22]. The wider program targets a fusion mechanism — Attention-Gated Fusion, combining cross-attention over the three modality streams with a learned, per-user gating mechanism — intended to replace the static NMF–DecisionTreeRegressor fusion with an adaptive, explainable one (A2-FusionRS, targeted at a Q1 journal venue). Before committing to that architecture, it is necessary to establish which representation of aspect-level sentiment is worth fusing in the first place; an attention mechanism cannot recover information a poorly designed upstream aggregation step has already destroyed. This paper addresses that prior question, calling the validated tri-modal representation A2-IRM (Aspect-Aware Integrated Representation Model).

The contributions of this paper are threefold: (1) we reimplement the Darraz et al. hybrid baseline end-to-end and extend the original two-domain evaluation to a third, structurally distinct domain (e-commerce electronics), generalizing the pipeline to be domain-agnostic; (2) we design and evaluate four aspect-based sentiment fusion strategies under an identical, statistically rigorous protocol (5 seeds, paired Wilcoxon tests, Fisher-combined p-values), and show that per-aspect score concatenation with confidence as an explicit auxiliary feature is the only variant that consistently and significantly improves on the baseline; (3) we show this result replicates across three domains with markedly different aspect-keyword coverage (45.1–95.9%), evidence for a domain-general mechanism rather than a dataset-specific artifact.

---

## II. Related Work

The case for incorporating sentiment into recommender systems rests on review sentiment and numeric rating being correlated but not redundant [13]. Elahi et al. [7] combined BERT embeddings with collaborative filtering on Amazon data, finding sentiment and rating are not always strongly correlated. Karabila et al. [4], [5] fine-tuned BERT for e-commerce sentiment fused with SVD-based CF. Li et al. [8] and Duan et al. [9] integrate review-derived sentiment directly into matrix factorization objectives — across this line of work, sentiment is computed once per review as a single scalar, the design this paper directly interrogates. A smaller body of work moves below the whole-review level: Kim et al. [10] propose an aspect-based CF model (AXCF) for explainability rather than accuracy; Yang et al. [14] apply attention to content representations alone, not jointly over collaborative, content-based, and sentiment streams; Ray, Garain, and Sarkar [15] combine BERT sentiment with fuzzy-logic aspect categorization for TripAdvisor reviews to structure retrieval rather than as a numeric fusion feature. None directly compares alternative strategies for turning a per-aspect sentiment vector into a fusable feature — the concatenation-versus-aggregation, score-versus-confidence design space this paper evaluates empirically. Relative to Darraz et al. [6], our direct baseline, we retain the same feature-combination architecture but replace the global sentiment stream with an aspect-based one, isolating sentiment granularity from fusion-mechanism design. To our knowledge, this is the first study to compare naive aggregation, confidence-weighted aggregation, and non-aggregated concatenation of ABSA output within an otherwise fixed hybrid architecture across three domains under a shared, seed-controlled, statistically tested protocol.

---

## III. Materials and Methods

### A. Datasets

We evaluate on three publicly sourced review datasets: Yelp restaurant reviews, Amazon Electronics reviews (e-commerce, no native category metadata), and TripAdvisor hotel reviews (with a native aspect taxonomy exploited directly in Section III-C). Each dataset was filtered to users/items with ≥5 interactions and split 80/10/10 (train/validation/test), user-based with cold-start holdout. Table I summarizes the resulting datasets.

**Table I. Dataset characteristics after minimum-interaction filtering (≥5 reviews per user and per item).**

| Domain | Reviews | Users | Items | Sparsity | Mean rating | Test set size |
|---|---:|---:|---:|---:|---:|---:|
| Restaurant (Yelp) | 118,695 | 7,152 | 3,757 | 99.56% | 3.76 | 13,233 |
| E-commerce (Amazon Electronics) | 122,068 | 14,750 | 9,226 | 99.91% | 4.37 | 16,580 |
| Hotel (TripAdvisor) | 79,562 | 11,236 | 2,056 | 99.66% | 3.94 | 11,795 |

Neither Amazon Electronics nor TripAdvisor Hotel carries business-attribute metadata analogous to Yelp's business categories; both loaders degrade gracefully when such metadata is absent (Section III-B).

### B. Reimplemented Baseline Architecture

We reimplement the Darraz et al. [6] hybrid architecture as our baseline: three parallel feature-extraction streams and a static fusion stage. **Collaborative filtering (DeepMF)** uses a 128-dimensional user/item embedding, an element-wise interaction layer, and a [256, 128, 64, 32]-unit feed-forward stack (ReLU, 0.3 dropout), trained with MSE loss via mini-batch SGD (batch 512, lr 0.001), without negative sampling (an earlier 1:4 negative-sampling configuration overfit within 1–2 epochs empirically). **Content-based filtering (clustering)** builds item features from up to four sources — one-hot category (where available), TF-IDF review text (500 terms), aggregated per-item sentiment, and popularity metrics — reduced to 50 principal components; K-means (elbow-selected K) is used for restaurant/e-commerce, agglomerative clustering for hotel, following Darraz et al.'s domain-specific choice. **Sentiment analysis** in the baseline is a single BERT-base-uncased classifier (AdamW, lr 1×10⁻⁵, 3 epochs, max length 128) fine-tuned per domain, producing one polarity score per review. **Fusion** combines the three per-(user, item) feature values via NMF (3 components) followed by a DecisionTreeRegressor (max depth 10), held identical across the baseline and all four ABSA variants so that any RMSE difference is attributable to the sentiment representation alone. We additionally report two non-hybrid classical CF baselines — item-KNN (cosine, k=40) and SVD (100 factors, 20 epochs, lr 0.005, reg 0.02) — to establish the accuracy gain from hybridization itself.

### C. Aspect-Based Sentiment Fusion Variants

We replace the global sentiment stream with an ABSA module reusing the same fine-tuned BERT [23] classifier — no additional model training — restructuring how it is applied. For each review, sentences are matched against a domain-specific keyword lexicon (Table II); a review with no match falls back to a whole-review score, so every review contributes at least one signal.

**Notation.** Let $\mathcal{A} = \{a_1, \ldots, a_K\}$ denote a domain's fixed aspect set ($K \in \{4,5,6\}$, Table II), and $A(r) \subseteq \mathcal{A}$ the aspects matched in review $r$. For $a \in A(r)$, $\hat{s}(a,r) \in [0,1]$ is the BERT score on sentences matched to aspect $a$, and $n(a,r)$ the number of matched sentences. $s_0(r) \in [0,1]$ is the whole-review fallback score.

For $a \in A(r)$, a per-aspect confidence combines a margin term and an evidence-count term:

$$c(a,r) = \max\left(\frac{|2\hat{s}(a,r) - 1| + \min(n(a,r)/3,\ 1)}{2},\ 0.05\right) \tag{1}$$

The margin term is largest when $\hat{s}(a,r)$ is decisive (near 0 or 1); the evidence term grows with matched sentences, capped at 3; the 0.05 floor keeps every aspect contributing a non-zero weight. This heuristic is specific to this pipeline, not drawn from prior work.

**Table II. Aspect keyword taxonomies used for ABSA sentence matching.**

| Domain | Aspects (n) | Aspect list | Source |
|---|---|---|---|
| Restaurant | 4 | food, service, price, ambiance | Manually curated |
| E-commerce | 5 | quality/durability, price/value, shipping/packaging, ease of use, customer service | Manually curated |
| Hotel | 6 | cleanliness, service, value, location, rooms, sleep quality | Native TripAdvisor taxonomy |

We evaluate four strategies for turning this per-aspect representation into a fusable feature, holding the fusion mechanism (Section III-B) fixed across all four:

1. **Mean.** Per-aspect scores are averaged over matched aspects only:

$$s_{\text{mean}}(r) = \begin{cases} \dfrac{1}{|A(r)|} \displaystyle\sum_{a \in A(r)} \hat{s}(a,r) & A(r) \neq \emptyset \\[6pt] s_0(r) & A(r) = \emptyset \end{cases} \tag{2}$$

2. **Confidence-weighted mean.** As above, weighted by $c(a,r)$ (Eq. 1):

$$s_{\text{conf}}(r) = \begin{cases} \dfrac{\sum_{a \in A(r)} c(a,r)\, \hat{s}(a,r)}{\sum_{a \in A(r)} c(a,r)} & A(r) \neq \emptyset \\[6pt] s_0(r) & A(r) = \emptyset \end{cases} \tag{3}$$

3. **Concat.** Preserves a value for every aspect, substituting the fallback for unmatched aspects, passed to the fusion stage without aggregation:

$$\tilde{s}(a,r) = \begin{cases} \hat{s}(a,r) & a \in A(r) \\ s_0(r) & a \notin A(r) \end{cases}, \qquad \mathbf{v}_{\text{concat}}(r) = [\tilde{s}(a_1,r), \ldots, \tilde{s}(a_K,r)] \in [0,1]^{K} \tag{4}$$

This yields one feature per aspect (4–6 depending on domain). The content-based stream instead receives $\bar{s}(r) = \frac{1}{K}\sum_{a \in \mathcal{A}} \tilde{s}(a,r)$ — the mean over the *entire* concat vector, which coincides with $s_{\text{mean}}(r)$ only when $A(r) = \mathcal{A}$ or $A(r) = \emptyset$; this does not affect Table III, which uses $\mathbf{v}_{\text{concat}}(r)$ directly as fusion input.

4. **Concat + confidence.** As Concat, with each aspect's confidence appended as an additional feature (doubling the feature count to 8–12) rather than used as an aggregation weight:

$$\tilde{c}(a,r) = \frac{|2\tilde{s}(a,r) - 1| + \min(n(a,r)/3,\ 1)}{2} \tag{5}$$

Unlike Eq. 1, $\tilde{c}(a,r)$ is not floored at 0.05 in the implementation that produced the results below — the floor is applied only in the confidence-weighted-mean aggregation (Eq. 3); we report this transparently rather than silently re-deriving Table III.

$$\mathbf{v}_{\text{concat+conf}}(r) = [\tilde{s}(a_1,r), \ldots, \tilde{s}(a_K,r),\ \tilde{c}(a_1,r), \ldots, \tilde{c}(a_K,r)] \in [0,1]^{2K} \tag{6}$$

This is the only variant in which confidence acts as a signal to the regressor rather than an aggregation weight.

### D. Experimental Setup and Evaluation Protocol

All five configurations and the two classical CF baselines are evaluated on each domain under 5 random seeds (42, 123, 456, 789, 1011); the split and BERT checkpoint are held fixed across seeds within a domain, isolating model-training variance. RMSE and MAE on held-out test ratings are the primary metrics; Precision/Recall/NDCG@K are reported for completeness but use a candidate-set-limited protocol that produces near-ceiling values with little discriminative power, so this paper's claims rest on RMSE/MAE. Significance between the baseline and each variant is assessed via paired Wilcoxon signed-rank tests on per-sample squared errors per seed; we report both the count of seeds reaching p < 0.05 and a Fisher-combined p-value across the 5 seeds (read as a conventional cross-seed summary, since the 5 tests share an identical test set).

---

## IV. Results

### A. Hybrid Baseline versus Classical Collaborative Filtering

The reimplemented hybrid baseline substantially outperforms non-hybrid classical CF on all three domains: RMSE reductions of 29–46% relative to item-KNN and 27–42% relative to SVD (restaurant: 0.693 vs. 1.202/1.075; e-commerce: 0.666 vs. 1.224/1.142; hotel: 0.650 vs. 0.916/0.895; all 5/5 seeds significant, p < 0.001 Fisher-combined), confirming the architecture is a meaningful reference point rather than a strawman.

### B. Effect of ABSA Sentiment-Fusion Strategy

Table III and Fig. 1 report RMSE for the baseline and four ABSA variants. The pattern is consistent in direction across all three domains despite substantially different rating distributions, sparsity, and aspect coverage.

**Mean and confidence-weighted mean aggregation both degrade RMSE substantially and significantly in every domain** — 12.9–21.3% relative increase for Mean (restaurant +20.3%, e-commerce +21.3%, hotel +12.9%) and 14.1–21.7% for Confidence-weighted mean (restaurant +19.6%, e-commerce +21.7%, hotel +14.1%), both 5/5 seeds significant (Fisher-combined p < 10⁻⁶). Confidence-weighting does not recover accuracy; in the hotel domain it shows the highest run-to-run variance of any configuration (SD = 0.038, >10× the best variant), suggesting instability rather than merely bias.

**Concat restores near-baseline parity** (+0.6% restaurant, +0.3% e-commerce, −2.5% hotel), the only variant with inconsistent significance across domains (2/5 restaurant, 4/5 e-commerce and hotel), with hotel showing Concat alone already improving on baseline.

**Concat + confidence is the only variant that significantly improves on the baseline in every domain** — RMSE reductions of 2.0% (restaurant, 0.693→0.679), 2.2% (e-commerce, 0.666→0.652), 3.2% (hotel, 0.650→0.629), reaching 4/5 seeds significant in restaurant and 5/5 elsewhere. It is also the lowest-variance configuration in every domain (SD 0.0010–0.0035), a 4–10× reduction relative to the baseline's own seed-to-seed SD (0.0100–0.0217). Inspecting raw per-seed RMSE, this variance advantage is genuinely uniform across seeds in every domain; the baseline's own variance in the hotel domain, by contrast, is disproportionately driven by a single outlier seed (RMSE 0.6886 at seed 123, excluding which reduces its SD 6.8×) rather than being uniformly spread — Concat + confidence shows no comparable single-seed sensitivity anywhere.

**Table III. RMSE (mean ± SD over 5 seeds) and significance vs. baseline (Wilcoxon per seed / Fisher-combined).**

| Variant | Restaurant | E-commerce | Hotel |
|---|---|---|---|
| Baseline (global SA) | 0.6926 ± 0.0100 | 0.6662 ± 0.0129 | 0.6501 ± 0.0217 |
| ABSA mean | 0.8330 ± 0.0103 (5/5)\*\*\* | 0.8081 ± 0.0068 (5/5)\*\*\* | 0.7341 ± 0.0073 (5/5)\*\*\* |
| ABSA confidence-mean | 0.8287 ± 0.0074 (5/5)\*\*\* | 0.8110 ± 0.0083 (5/5)\*\*\* | 0.7416 ± 0.0380 (5/5)\*\*\* |
| ABSA concat | 0.6968 ± 0.0020 (2/5)\*\*\* | 0.6682 ± 0.0063 (4/5)\*\*\* | 0.6336 ± 0.0047 (4/5)\*\*\* |
| ABSA concat + confidence | **0.6791 ± 0.0010** (4/5)\*\*\* | **0.6517 ± 0.0032** (5/5)\*\*\* | **0.6291 ± 0.0035** (5/5)\*\*\* |

\*\*\* Fisher-combined p < 0.001. Bold denotes the best-performing (lowest RMSE) variant per domain.

![Fig. 1. RMSE of the reimplemented hybrid baseline vs. four ABSA sentiment-fusion variants, across three domains (mean ± SD over 5 seeds; significance vs. baseline, Wilcoxon per seed + Fisher-combined p)](figures/fig2_rmse_main_result.png)

### C. Cross-Domain Consistency and Aspect Coverage

We measured the fraction of reviews matching at least one aspect keyword before BERT scoring: 87.7% restaurant, 45.1% e-commerce, 95.9% hotel — a 51-point range. Despite this, the direction of every pairwise comparison in Table III is identical across all three domains. The relative *magnitude* of the Concat + confidence improvement does not track coverage monotonically, however: hotel has both the highest coverage and largest effect (3.2%), directionally consistent with richer coverage supplying more material to exploit, but restaurant has the second-highest coverage (87.7%) yet the *smallest* effect (1.95%), while e-commerce has the lowest coverage (45.1%) but a larger effect (2.17%) than restaurant. We do not read this as a clean dose-response relationship; distinguishing an aspect-coverage effect from other domain-level confounds would require a larger, more systematically varied set of domains than the three evaluated here.

---

## V. Discussion

**Why does naive aggregation actively hurt?** We suggest two mechanisms. First, the whole-review fallback for unmatched reviews means the "mean" score for a substantial share of reviews — more than half in e-commerce — is computed over a single matched sentence or the fallback rather than a genuine multi-aspect average, injecting noise relative to the baseline's dedicated global classifier. Second, reviews with multiple matched aspects of opposing polarity (e.g., "food was excellent, service was slow") see a plain or confidence-weighted mean collapse toward an uninformative neutral score, destroying exactly the polarity contrast the aspect decomposition was meant to preserve. Concatenation avoids both failure modes by preserving the full vector and letting the downstream DecisionTreeRegressor determine how to use it, rather than pre-committing to an aggregation rule.

**Why does confidence help as a feature but not as a weight?** Confidence-weighting an aggregate can only redistribute mass among already-averaged aspects — it cannot recover the polarity contrast averaging discards. Supplying confidence as an explicit feature instead gives the regressor a signal about *how much to trust* each aspect score independently of its value, which a tree-based regressor can condition on directly. This also plausibly explains Concat + confidence's substantially reduced variance: aspect-confidence features may act as implicit regularization against noisy, low-evidence aspect scores across seeds.

**Cross-domain generalization.** The consistency of direction and approximate effect size across three domains with markedly different rating distributions, sparsity, and a 51-point aspect-coverage range is, in our view, the strongest evidence that the Concat + confidence result reflects a genuine property of the representation rather than a Yelp-specific artifact — though three domains constrain what can be claimed as a general guarantee.

**Limitations.** The restaurant and e-commerce aspect lexicons (Table II) were manually curated rather than empirically validated, unlike the hotel domain's native taxonomy, likely contributing to e-commerce's lower coverage (45.1%); whether a refined lexicon would widen the effect is an open question, since coverage does not predict effect size monotonically (Section IV-C). The ranking metrics use a candidate-set-limited protocol producing near-ceiling values with limited discriminative power; this study's claims rest on RMSE/MAE. Content-based clustering falls back to text/popularity features only for e-commerce and hotel, which lack category metadata — we did not isolate this stream's marginal contribution. Seeds vary only stochastic components downstream of a fixed split and BERT checkpoint, not split-choice variance itself.

---

## VI. Conclusion and Future Work

We reimplemented a published hybrid recommender architecture and used it as a controlled testbed to evaluate four strategies for converting ABSA output into a fusable feature. Across three structurally different domains, naive aggregation — with or without confidence weighting — degrades rating-prediction accuracy relative to a well-tuned global-sentiment baseline, while preserving per-aspect scores as a raw feature vector supplemented with explicit per-aspect confidence as an auxiliary (not aggregating) feature is the only strategy that significantly and consistently improves on it, with a 4–10× reduction in cross-seed variance. This argues that *how* aspect-level sentiment is represented before fusion is at least as consequential as *whether* it is used at all — a question orthogonal to, and prior to, the choice of fusion mechanism itself.

This validated representation, A2-IRM, is the direct empirical input to the next stage of this research program: replacing the static NMF–DecisionTreeRegressor fusion evaluated here with an Attention-Gated Fusion Network — cross-attention over the three modality streams followed by a learned, per-user gating mechanism — intended to adapt each stream's relative contribution dynamically and support aspect- and modality-level explainability that a static fusion cannot provide by construction. Concat + confidence is the representation we carry forward as the ABSA input stream to that architecture (A2-FusionRS), rather than re-opening the aggregation-strategy question at that stage.

---

## Acknowledgment

*[To be completed — this research is part of a decentralized internal research grant (Dana Internal Penelitian Desentralisasi FT-Matching Fund) at Universitas Negeri Malang, 2026.]*

---

## References

*[Numbered per first citation order, IEEE style — same reference list as the full-length manuscript version; see `A2-IRM_manuscript_draft.md` for the complete list with all 23 entries and the flagged renumbering/metadata-verification note.]*

[1] T. Chang, Z. Zhang, and X. Cai, "Explainable recommender system directed by reconstructed explanatory factors and multi-modal matrix factorization," *Concurrency and Computation*, vol. 36, no. 21, p. e8208, Sep. 2024, doi: 10.1002/cpe.8208.

[2] N. Darraz, I. Karabila, A. El-Ansari, N. Alami, and M. El Mallahi, "Enhancing recommendation systems with collaborative filtering and sentiment analysis: dimensionality reduction for improved content-based approaches," *Knowl Inf Syst*, vol. 67, no. 8, pp. 7157–7191, Aug. 2025, doi: 10.1007/s10115-025-02452-z.

[3] N. Liu and J. Zhao, "Recommendation System Based on Deep Sentiment Analysis and Matrix Factorization," *IEEE Access*, vol. 11, pp. 16994–17001, 2023, doi: 10.1109/ACCESS.2023.3246060.

[4] I. Karabila, N. Darraz, A. EL-Ansari, N. Alami, and M. EL Mallahi, "BERT-enhanced sentiment analysis for personalized e-commerce recommendations," *Multimed Tools Appl*, vol. 83, no. 19, pp. 56463–56488, Dec. 2023, doi: 10.1007/s11042-023-17689-5.

[5] I. Karabila, N. Darraz, A. El-Ansari, N. Alami, and M. E. Mallahi, "A hybrid approach combining sentiment analysis and deep learning to mitigate data sparsity in recommender systems," *Neurocomputing*, vol. 636, p. 129886, Jul. 2025, doi: 10.1016/j.neucom.2025.129886.

[6] N. Darraz, I. Karabila, A. El-Ansari, N. Alami, and M. El Mallahi, "Integrated sentiment analysis with BERT for enhanced hybrid recommendation systems," *Expert Systems with Applications*, vol. 261, p. 125533, Feb. 2025, doi: 10.1016/j.eswa.2024.125533.

[7] M. Elahi, et al., "[Hybrid recommender system incorporating sentiment analysis on Amazon Digital Music and Video Games datasets]," 2023.

[8] X. J. Li, G. S. Deng, X. Z. Wang, X. L. Wu, and Q. W. Zeng, "A hybrid recommendation algorithm based on user comment sentiment and matrix decomposition," *Information Systems*, vol. 117, p. 102244, Jul. 2023, doi: 10.1016/j.is.2023.102244.

[9] R. Duan, C. Jiang, and H. K. Jain, "Combining review-based collaborative filtering and matrix factorization: A solution to rating's sparsity problem," *Decision Support Systems*, vol. 156, p. 113748, May 2022, doi: 10.1016/j.dss.2022.113748.

[10] D. Kim, Q. Li, D. Jang, and J. Kim, "AXCF: Aspect-based collaborative filtering for explainable recommendations," *Expert Systems*, vol. 41, no. 8, p. e13594, Aug. 2024, doi: 10.1111/exsy.13594.

[11] M. Ibrahim, I. S. Bajwa, N. Sarwar, F. Hajjej, and H. A. Sakr, "An Intelligent Hybrid Neural Collaborative Filtering Approach for True Recommendations," *IEEE Access*, vol. 11, pp. 64831–64849, 2023, doi: 10.1109/ACCESS.2023.3289751.

[12] T.-D. Dang, N.-T. Moreno-García, and F. De la Prieta, "[Sentiment analysis and genre-based similarity in collaborative filtering for movie recommendation]," 2021.

[13] S. Al-Ghuribi and S. A. Noah, "[A survey on sentiment-aware recommender systems]," 2019.

[14] S. Yang, Q. Li, H. Lim, and J. Kim, "An Attentive Aspect-Based Recommendation Model With Deep Neural Network," *IEEE Access*, vol. 12, pp. 5781–5791, 2024, doi: 10.1109/ACCESS.2023.3349291.

[15] A. Ray, A. Garain, and R. Sarkar, "[Hotel recommendation system combining sentiment analysis and aspect-based review categorization for TripAdvisor reviews]," 2021.

[16] R. Bhatt, K. Patel, and P. Gaudani, "[A survey on recommendation system hybridization strategies]," 2014.

[17] H. Fayyaz, S. Ebrahimian, D. Nawara, R. Ibrahim, and R. Kashef, "[A review of recommender system hybridization techniques]," 2020.

[18] T. Widiyaningtyas, I. Hidayah, and T. B. Adji, "User profile correlation-based similarity (UPCSim) algorithm in movie recommendation system," *J Big Data*, vol. 8, no. 1, p. 52, Dec. 2021, doi: 10.1186/s40537-021-00425-x.

[19] T. Widiyaningtyas, I. Hidayah, and T. B. Adji, "Recommendation Algorithm Using Clustering-Based UPCSim (CB-UPCSim)," *Computers*, vol. 10, no. 10, p. 123, Oct. 2021, doi: 10.3390/computers10100123.

[20] T. Widiyaningtyas, M. I. Ardiansyah, and T. B. Adji, "Recommendation Algorithm Using SVD and Weight Point Rank (SVD-WPR)," *BDCC*, vol. 6, no. 4, p. 121, Oct. 2022, doi: 10.3390/bdcc6040121.

[21] T. Widiyaningtyas, A. P. Wibawa, U. Pujianto, and W. Caesarendra, "MF-NCG: Recommendation Algorithm Using Matrix Factorization-based Normalized Cumulative Genre," *IJIES*, vol. 17, no. 2, pp. 180–189, Apr. 2024, doi: 10.22266/ijies2024.0430.16.

[22] I. F. Rozi, R. Arianto, D. R. Yunianto, A. Y. Ananta, S. Rahmawati, and Krismawati, "Enhancing Aspect-Based Sentiment Analysis for Radio Station Public Opinion: Evaluating Preprocessing Strategies and Imbalanced Data Handling," in *2024 International Conference on Electrical and Information Technology (IEIT)*, Malang, Indonesia: IEEE, Sep. 2024, pp. 103–108, doi: 10.1109/IEIT64341.2024.10763129.

[23] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, "BERT: Pre-training of deep bidirectional transformers for language understanding," in *Proc. 2019 Conf. North American Chapter Assoc. Comput. Linguistics: Human Language Technologies (NAACL-HLT)*, Minneapolis, MN, USA, Jun. 2019, pp. 4171–4186, doi: 10.18653/v1/N19-1423.

*[Same numbering caveat as the full-length version: [18]–[22] are cited earlier in text (Introduction) than [7]–[17] (Related Work) — a final renumbering pass is needed before submission; reference content is unaffected.]*
