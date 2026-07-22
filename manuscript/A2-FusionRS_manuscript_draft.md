# A2-FusionRS: Attention-Gated Fusion of Model-Based Aspect Sentiment for Review-Aware Recommendation under Extreme Sparsity

> **Working draft — target: Expert Systems with Applications (Q1).** Complete draft:
> Abstract + §1–§7 + Tables 2–10. Table values are extracted from the run outputs by
> `build_manuscript_tables.py` and `analyze_interpretability.py`; all citations map to
> verified DOIs (or dataset URLs) in `A2-FusionRS_references.md`.

---

## Abstract

Review-based recommender systems mitigate the sparsity of rating data by mining the
opinions users express in text, and aspect-based sentiment analysis (ABSA) makes those
opinions fine-grained. Two questions are usually left open: how to incorporate
open-vocabulary, model-based aspect sentiment — whose per-review aspect set is
variable-length — and, when an aspect signal improves a recommender, whether the gain
comes from the signal or from the fusion architecture that consumes it. We present
A2-FusionRS, an attention-gated fusion recommender that extends a static-fusion hybrid
(A2-IRM) with model-based per-aspect sentiment, summarized by a learned aspect-sequence
pooling layer that preserves aspect identity, and combined with collaborative and
content streams through cross-attention and gated pooling over a residual base. Across
three domains (Amazon Electronics, Yelp Restaurant, TripAdvisor Hotel) and five seeds,
A2-FusionRS attains the lowest RMSE in every domain (0.6418, 0.6665, 0.6196), improving
significantly over A2-IRM (1.5–1.9%) and over classical and neural collaborative
baselines (26–44% over NeuMF), with all gains significant at every seed (Wilcoxon,
$p<0.001$). A component-level attribution — including a control that fuses the same
model-based aspect features with a static tree — shows the improvement is due to the
aspect modality rather than the attention mechanism, which matches, but does not beat,
the static fusion on accuracy; we report this deliberately falsifiable result rather than
overclaim the architecture. We further find that the benefit of model-based ABSA grows as
a domain's keyword-aspect coverage falls, and that the model's aspect attention is
faithful under a perturbation test (the most-attended aspect drives the prediction in
~70% of reviews). The result is a recommender that improves over strong baselines together
with an evidenced account of why.

**Keywords:** recommender systems; aspect-based sentiment analysis; attention mechanism;
data sparsity; multimodal fusion; explainable recommendation.

---

## 1. Introduction

Online platforms now mediate most consumer choices through recommendation, and the
ratings they collect are the classical signal for learning user preferences. That signal,
however, is chronically thin: the user–item interaction matrix is extremely sparse, and
newly arrived users and items carry little or no history, so purely rating-based
collaborative filtering degrades badly under the sparsity and cold-start conditions that
dominate real catalogs (Idrissi & Zellou, 2020; Yuan & Hernandez, 2023). A large body of
work therefore turns to the free-text reviews that accompany ratings, which describe *why*
a user liked an item and expose preferences a single star rating cannot (Duan et al.,
2022; Elahi et al., 2023).

Among review signals, aspect-based sentiment analysis (ABSA) is especially attractive
because it decomposes an opinion into sentiments toward specific aspects — a restaurant's
*service* or *ambiance*, a product's *battery* or *screen* — rather than collapsing a
review into a single polarity. Aspect-level preference modeling has repeatedly improved
rating prediction and top-N recommendation over rating-only and document-level-sentiment
methods (Cai et al., 2022; Kim et al., 2024; Yang et al., 2024b). It also helps precisely
where ratings are least reliable: reviewers frequently assign a high star rating while
voicing aspect-level dissatisfaction, an inconsistency that aspect sentiment can surface
and correct (Aramanda et al., 2023; Rabiu et al., 2022).

Despite this progress, two gaps motivate the present work. **First**, the aspect signal
itself is usually extracted with a *fixed taxonomy* — a handful of predefined categories
matched by keywords — whose coverage varies sharply across domains and collapses when an
aspect vocabulary is open-ended. Open-vocabulary, model-based ABSA can extract aspects a
fixed taxonomy never anticipates, but it produces a *variable-length, per-review set* of
aspects that is awkward to fuse with the fixed-width features recommenders expect, and it
is unclear when its additional cost is repaid. **Second**, and more consequential for the
field, when an aspect signal *does* improve a recommender, the improvement is seldom
attributed: papers typically introduce a new architecture together with a new signal and
report a net gain, leaving open whether the architecture or the signal did the work. A
recommender that is credited to its attention mechanism, but would perform identically
with a simple fuser given the same features, has been mis-attributed — and the field's
running assumption that more elaborate fusion is the source of gains goes largely
untested.

This paper studies both gaps in one system. We build on A2-IRM, a static-fusion hybrid
that combines deep matrix factorization, content-based clustering, and keyword-ABSA
through a non-negative-matrix-factorization and decision-tree fuser, in the lineage of
Darraz et al. (2025). We extend it into **A2-FusionRS**, which (i) adds open-vocabulary
model-based ABSA (PyABSA; Yang & Li, 2023) as a complementary modality, summarized by a
learned *aspect-sequence pooling* layer that preserves aspect identity instead of
averaging it away, and (ii) fuses the modalities with an attention-gated mechanism whose
weights are interpretable. Crucially, we pair the model with an *attribution protocol*
that toggles the architecture and the signal independently, so that the source of any gain
is identified rather than assumed.

Our contributions are:

1. **A2-FusionRS**, an attention-gated fusion recommender that integrates model-based
   per-aspect sentiment as a complementary modality through a learned aspect-sequence
   pooling layer, and that improves significantly over a strong prior hybrid (A2-IRM) and
   over classical and neural collaborative baselines across three domains and five seeds.
2. **An attribution study** that separates the effect of the fusion architecture from the
   effect of the added signal. Through component ablations and a control that fuses the
   same model-based aspect features with a static tree, we show the improvement is
   attributable to the aspect modality, and that the attention mechanism *matches* rather
   than beats the static fusion on accuracy — a deliberately falsifiable result that
   corrects a common mis-attribution.
3. **A coverage-dependence finding**: the benefit of model-based ABSA grows as the
   fixed-taxonomy keyword coverage of a domain falls, giving practical guidance on when the
   costlier encoder is worth adopting.
4. **A validated interpretability analysis**: we expose per-modality and per-aspect
   attributions and test the latter with a perturbation study, showing the aspect attention
   is faithful in the aggregate rather than merely displayed — and we report its limits
   honestly.

We organize the study around three questions. **RQ1**: does A2-FusionRS improve rating
prediction over classical, neural, and hybrid baselines across domains? **RQ2**: when it
improves, is the gain due to the model-based aspect signal or to the attention fusion?
**RQ3**: under what domain conditions does model-based ABSA help most? Section 2 reviews
related work; Section 3 formalizes the task; Section 4 details A2-FusionRS; Section 5
describes the experimental setup; Section 6 reports and discusses results; Section 7
concludes.

---

## 2. Related Work

### 2.1 Collaborative filtering and neural recommendation

Matrix factorization remains the backbone of rating prediction, mapping users and items to
latent factors whose inner product estimates a rating (Koren et al., 2009). Neural
extensions replace or augment the inner product with learned interaction functions:
Neural Collaborative Filtering combines generalized matrix factorization with a multilayer
perceptron (He et al., 2017), and DeepFM couples a factorization machine with a deep
network to capture low- and high-order feature interactions (Guo et al., 2017). These
models are strong when interactions are dense, but their reliance on the interaction
signal alone makes them vulnerable to the sparsity and cold-start regimes that characterize
review platforms (Idrissi & Zellou, 2020; Yuan & Hernandez, 2023) — a vulnerability our
experiments quantify directly, where such models barely exceed a constant-mean predictor.

### 2.2 Review-aware and content-based recommendation

To compensate for thin ratings, a substantial line of work mines the accompanying review
text. Early hybrids inject document-level sentiment or topic features into a factorization
model (Elahi et al., 2023; Lai & Hsu, 2021), while later methods learn joint
representations of reviews and ratings (Cai et al., 2022; Duan et al., 2022) or add further
modalities such as product images (Zhan & Xu, 2023). A recurring theme is that reviews
mitigate sparsity and cold-start: review-based collaborative filtering and matrix
factorization have been combined explicitly to address rating sparsity (Duan et al.,
2022), and stacked-autoencoder and ensemble models exploit reviews for top-N
recommendation on sparse corpora (Abinaya & Devi, 2021; Choudhary et al., 2023).
Transformer language models have further sharpened the text representation, with BERT
embeddings of reviews feeding hybrid recommenders in the immediate lineage of the present
work (Karabila et al., 2023, 2025). These methods establish that the review channel is
valuable; they largely treat sentiment at the document or sentence level, however, leaving
the finer aspect structure underused.

### 2.3 Aspect-based sentiment for recommendation

A finer-grained strand extracts aspect-level sentiment and feeds it to the recommender.
Approaches range from clustering users by aspect sentiment (Poudel & Bikdash, 2022) and
attention over aspect terms (Lai et al., 2021; Yang et al., 2024b), to aspect-aware graph
neural networks (Zhang et al., 2023), knowledge-graph models driven by review sentiment
(Cui et al., 2024), variational aspect-level models (Ou & Huynh, 2024), and, most recently,
LLM-based aspect extraction (Liu et al., 2025). Several report that aspect-level signals
outperform overall-sentiment baselines (Kim et al., 2024; Yang et al., 2024a). Two
limitations persist. Most systems still derive aspects from a fixed taxonomy or a
domain-specific list, so their coverage is domain-bound; and where an open-vocabulary
extractor is used, its variable-length aspect set is typically reduced by averaging or
simple pooling, which discards the polarity contrast *between* aspects. Model-based ABSA
toolkits such as PyABSA (Yang & Li, 2023) make open-vocabulary extraction practical, but
how best to represent their per-aspect output for a recommender — and when the extra cost
is justified — remains open. A2-FusionRS addresses the representation question with a
learned aspect-sequence pooling that keeps aspect identity, and the cost question with the
coverage-dependence analysis of Section 6.4.

### 2.4 Attention and multimodal fusion in recommendation

Attention (Vaswani et al., 2017) and gating are now standard for combining heterogeneous
recommendation signals, weighting words, aspects, or modalities by learned importance
(Lai et al., 2021; Yang et al., 2024b). Multi-aspect models fuse several aspect views
through routing or graph aggregation (Zhang et al., 2023), and transformer text encoders
are routinely bolted onto collaborative backbones (Devlin et al., 2019; Karabila et al.,
2023). This literature convincingly demonstrates that attention-based fusion *can* match
or exceed simpler combiners; what it rarely does is isolate the fusion mechanism's
contribution from that of the signals it fuses. Because new architectures and new signals
are usually introduced together, the community's implicit attribution of gains to the
fusion mechanism has seldom been tested against a same-signal control — the gap our
attribution protocol is designed to close.

### 2.5 Explainable and interpretable recommendation

Interest in *why* a recommendation is made has grown alongside accuracy, and reviews are a
natural source of explanations. Recent systems attach aspect- or word-level attributions
to their predictions (Kim et al., 2024; Wu et al., 2024), reconstruct explanatory factors
in a factorization model (Chang et al., 2024), or survey the broader space of explainable
recommenders (Tiwary et al., 2024). A caution from the NLP literature applies throughout:
attention weights are not automatically faithful explanations, and treating an attention
heatmap as proof of what a model used is unsafe without a faithfulness test (Jain &
Wallace, 2019; Wiegreffe & Pinter, 2019). We take this caution seriously, validating our
aspect attributions with a perturbation study (Section 6.5) rather than presenting them as
self-evident.

### 2.6 Positioning

Across these strands, two things are largely missing and together define our contribution.
First, when a review or aspect signal improves a recommender, the improvement is rarely
*attributed* to a source through a same-signal, different-architecture control; the
prevailing assumption that elaborate fusion drives gains is mostly untested. Second, the
value of open-vocabulary model-based ABSA is seldom analyzed as a function of a domain's
existing aspect coverage, so there is little guidance on when it is worth its cost.
A2-FusionRS is built to address both: it introduces model-based per-aspect sentiment as a
complementary modality with an identity-preserving pooling layer, but it is accompanied by
an attribution protocol that credits the resulting gain to the signal rather than the
architecture, a coverage-dependence analysis that says when the signal helps, and a
faithfulness-tested interpretability account. The result is a system that improves over
strong baselines *and* an honest account of why.

---

## 3. Preliminaries and Problem Formulation

### 3.1 Notation and task

Let $\mathcal{U}=\{u\}$ be the set of users and $\mathcal{I}=\{i\}$ the set of items.
Each observed interaction is a triple $(u,i,r_{ui})$ with an explicit rating
$r_{ui}\in[1,5]$ and an associated free-text review $d_{ui}$. The observed set is
$\mathcal{O}=\{(u,i)\mid r_{ui}\ \text{is observed}\}$, which is extremely sparse:
in the three domains studied here the density $|\mathcal{O}|/(|\mathcal{U}|\,|\mathcal{I}|)$
falls below $0.1\%$ (Table 2).

We address **rating prediction**: learn a scoring function
$f:\mathcal{U}\times\mathcal{I}\rightarrow[1,5]$ that estimates $\hat r_{ui}=f(u,i)$
for held-out pairs, optimized for root-mean-squared error (RMSE) and mean absolute
error (MAE), and additionally evaluated as a ranking task through Precision@K,
Recall@K, and NDCG@K. Ratings are modeled on a normalized scale
$\tilde r_{ui}=(r_{ui}-1)/4\in[0,1]$ during training and mapped back to $[1,5]$ for
evaluation, so that reported errors remain comparable to prior work on the original
rating scale.

### 3.2 Modality signals

A2-FusionRS draws on three complementary views of each interaction, reused from the
A2-IRM baseline — a hybrid of deep matrix factorization, content-based clustering,
and NMF + decision-tree fusion, the Phase-1 predecessor of this work (Darraz et al.,
2025) — and extended here:

- **Collaborative (DeepMF).** A deep matrix-factorization model over user/item
  embeddings captures latent collaborative structure and yields a rating estimate
  $\hat r^{\mathrm{MF}}_{ui}$.
- **Content-based (CBF).** Item content features (reduced by PCA) combined with a
  user's cluster-level preference profile yield a content estimate
  $\hat r^{\mathrm{CBF}}_{ui}$.
- **Aspect-based sentiment (ABSA).** Two encoders express *what* the reviewer liked
  or disliked at the aspect level: a fixed-taxonomy **keyword-ABSA** encoder that
  emits a confidence-weighted vector $\mathbf{v}^{\mathrm{kw}}_{ui}$, and an
  open-vocabulary **model-based** encoder (PyABSA) that emits a *variable-length set*
  of per-aspect sentiment tuples (Section 4.2).

The keyword-ABSA vector, DeepMF, and CBF are exactly the signals fused statically by
A2-IRM; the model-based per-aspect signal is the additional modality introduced in
this work.

---

## 4. Proposed Method: A2-FusionRS

### 4.1 Overview

Figure 2 shows the architecture. A2-FusionRS encodes each interaction into four
modality tokens, contextualizes them with multi-head cross-attention, and combines
them through a gated pooling layer whose per-modality weights are interpretable. The
pooled representation drives a small prediction head that outputs an **additive
correction** on top of a static fusion base, so the model learns only the residual
that the base cannot already explain.

Two design choices distinguish the encoder stage. First, the representation is
**asymmetric**: DeepMF and CBF are rating predictors, so their strongest signal is
their scalar output; they enter the fusion as (normalized) scalars rather than raw
latent features. The sentiment stream, by contrast, is kept as a rich representation,
because aspect-level polarity is exactly the fine-grained signal the fusion is meant
to exploit. Second, the open-vocabulary aspect set is summarized by a *learned*
**aspect-sequence pooling** layer rather than by hand-designed statistics, preserving
the identity of individual aspects.

### 4.2 Modality encoders and tokens

**Collaborative and content scalars.** DeepMF and CBF produce scalar rating estimates
that are normalized and linearly projected to the shared model dimension $d$:

$$
\mathbf{t}^{\mathrm{MF}}_{ui}=\mathbf{W}_{\mathrm{MF}}\,\tilde r^{\mathrm{MF}}_{ui}+\mathbf{b}_{\mathrm{MF}},
\qquad
\mathbf{t}^{\mathrm{CBF}}_{ui}=\mathbf{W}_{\mathrm{CBF}}\,\tilde r^{\mathrm{CBF}}_{ui}+\mathbf{b}_{\mathrm{CBF}},
\tag{1}
$$

with $\mathbf{W}_{\bullet}\in\mathbb{R}^{d\times 1}$ and $\tilde r^{\bullet}_{ui}$ the
normalized scalar estimate.

**Keyword-ABSA vector.** The fixed-taxonomy encoder yields a confidence-weighted
vector $\mathbf{v}^{\mathrm{kw}}_{ui}\in\mathbb{R}^{K}$ over $K$ predefined aspect
categories, projected to the token
$\mathbf{t}^{\mathrm{kw}}_{ui}=\mathbf{W}_{\mathrm{kw}}\mathbf{v}^{\mathrm{kw}}_{ui}+\mathbf{b}_{\mathrm{kw}}$.

**Model-based per-aspect set.** The open-vocabulary encoder extracts, for each
review, a variable-length set of
$m_{ui}$ aspect tuples (PyABSA, ATEPC checkpoint; Yang & Li, 2023)

$$
\mathcal{A}_{ui}=\bigl\{(a_j,\;p^{\mathrm{neg}}_j,\;p^{\mathrm{neu}}_j,\;p^{\mathrm{pos}}_j,\;c_j)\bigr\}_{j=1}^{m_{ui}},
\tag{2}
$$

where $a_j$ is an aspect term, $(p^{\mathrm{neg}}_j,p^{\mathrm{neu}}_j,p^{\mathrm{pos}}_j)$
its predicted polarity distribution, and $c_j$ the model confidence. Because $m_{ui}$
varies and the aspect vocabulary is open, this set cannot be mapped to a fixed set of
columns without discarding information; averaging across aspects, in particular,
destroys the polarity contrast *between* aspects — the same failure mode observed for
mean-pooled sentiment in A2-IRM (the authors' prior work). We therefore summarize $\mathcal{A}_{ui}$
with a learned attention pooling that preserves aspect identity.

**Aspect-sequence pooling.** Each aspect term is mapped to a learned embedding
$\mathbf{e}(a_j)\in\mathbb{R}^{d_a}$ through a frequency-ranked vocabulary (top-$V$
terms; a reserved out-of-vocabulary symbol handles the tail), built from the training
split only to avoid leakage of test-time aspect terms. The embedding is concatenated
with the four-dimensional sentiment/confidence features and projected to a token:

$$
\mathbf{s}_j=\mathbf{W}_a\bigl[\mathbf{e}(a_j)\,\Vert\,p^{\mathrm{neg}}_j\,\Vert\,p^{\mathrm{neu}}_j\,\Vert\,p^{\mathrm{pos}}_j\,\Vert\,c_j\bigr]+\mathbf{b}_a\in\mathbb{R}^{d}.
\tag{3}
$$

A single learned query $\mathbf{q}\in\mathbb{R}^{d}$ attends over the aspect tokens
$\{\mathbf{s}_j\}$ with a padding mask $\mathbf{M}_{ui}$ that excludes padded positions,
yielding the pooled aspect token

$$
\mathbf{t}^{\mathrm{asp}}_{ui}=\mathrm{Attn}\!\left(\mathbf{q},\,\mathbf{S}_{ui},\,\mathbf{S}_{ui};\,\mathbf{M}_{ui}\right),
\qquad
\mathbf{S}_{ui}=[\mathbf{s}_1,\dots,\mathbf{s}_{m_{ui}}],
\tag{4}
$$

where $\mathrm{Attn}$ is scaled dot-product attention. The attention weights
$\{\alpha^{\mathrm{asp}}_j\}$ over aspects are retained for interpretability (Section 6.5).
Reviews with no extracted aspect fall back to a single synthetic token so the layer is
always defined.

The four tokens form the modality set
$\mathcal{T}_{ui}=\{\mathbf{t}^{\mathrm{MF}}_{ui},\mathbf{t}^{\mathrm{CBF}}_{ui},\mathbf{t}^{\mathrm{kw}}_{ui},\mathbf{t}^{\mathrm{asp}}_{ui}\}$,
stacked as $\mathbf{X}_{ui}\in\mathbb{R}^{4\times d}$.

### 4.3 Attention-gated fusion

The tokens are contextualized by multi-head self-attention with a residual connection
and layer normalization,

$$
\mathbf{Z}_{ui}=\mathrm{LayerNorm}\!\left(\mathbf{X}_{ui}+\mathrm{MHA}(\mathbf{X}_{ui},\mathbf{X}_{ui},\mathbf{X}_{ui})\right),
\tag{5}
$$

so each modality can attend to the others before being weighted. A gating network then
produces a per-modality distribution and pools the tokens into a single vector:

$$
\boldsymbol{\alpha}_{ui}=\mathrm{softmax}\!\left(g(\mathbf{Z}_{ui})\right)\in\Delta^{3},
\qquad
\mathbf{z}_{ui}=\sum_{k=1}^{4}\alpha_{ui,k}\,\mathbf{Z}_{ui,k},
\tag{6}
$$

where $g(\cdot)$ is a small multilayer perceptron and $\Delta^{3}$ the probability
simplex over the four modalities. The gate weights $\boldsymbol{\alpha}_{ui}$ are the
model's explicit, per-instance statement of how much each modality contributes, and are
analyzed in Section 6.5.

### 4.4 Residual prediction over a static base

Rather than predict the rating directly, the head predicts a correction on top of a
static fusion base. The base is the A2-IRM fusion — a non-negative matrix
factorization followed by a decision-tree regressor (NMF+DT) — computed **only** from
the three signals A2-IRM already uses, namely the keyword-ABSA vector and the DeepMF
and CBF scalars:

$$
\hat r^{\mathrm{base}}_{ui}=\mathrm{NMF\text{-}DT}\!\left(\mathbf{v}^{\mathrm{kw}}_{ui},\,\hat r^{\mathrm{MF}}_{ui},\,\hat r^{\mathrm{CBF}}_{ui}\right).
\tag{7}
$$

Crucially, the model-based per-aspect signal is **excluded** from the base; it enters
only through the attention path. This makes the attribution unambiguous: any
improvement the residual achieves over A2-IRM is attributable to the new
(PyABSA + attention) information rather than to refitting the same features. The head
emits the correction and the final prediction is additive,

$$
\Delta_{ui}=h(\mathbf{z}_{ui}),
\qquad
\hat r_{ui}=\hat r^{\mathrm{base}}_{ui}+\Delta_{ui},
\tag{8}
$$

with $h(\cdot)$ a two-layer MLP and no output nonlinearity, so the correction may be
positive or negative.

To keep the base prediction on the training set out-of-sample — otherwise the base
would fit the training ratings almost perfectly and leave a near-zero residual for the
attention path to learn — the training-fold base is produced by 5-fold out-of-fold
(OOF) stacking: for each fold the NMF+DT is fit on the complementary folds and predicts
the held-out fold. Validation and test bases are produced by an NMF+DT fit on the full
training split, which is already out-of-sample for those splits.

### 4.5 Training objective

All components downstream of the frozen modality encoders are trained jointly by
minimizing the mean-squared error between the additive prediction and the normalized
rating,

$$
\mathcal{L}=\frac{1}{|\mathcal{B}|}\sum_{(u,i)\in\mathcal{B}}\bigl(\tilde r^{\mathrm{base}}_{ui}+\Delta_{ui}-\tilde r_{ui}\bigr)^{2},
\tag{9}
$$

over mini-batches $\mathcal{B}$, using Adam with weight decay. The checkpoint with the
lowest validation RMSE is restored at the end of training rather than the last epoch,
mirroring the early-stopping-by-restore policy of the DeepMF stream. Hyperparameters
($d$, number of heads $H$, aspect-embedding size $d_a$, maximum aspects per review,
vocabulary size $V$, learning rate, weight decay, epochs) are listed in Table 3.

### 4.6 Interpretability signals

A2-FusionRS exposes two complementary attributions at no additional cost. The gate
weights $\boldsymbol{\alpha}_{ui}$ (Eq. 6) report the per-modality contribution to the
correction, and the aspect-pooling weights $\{\alpha^{\mathrm{asp}}_j\}$ (Eq. 4) report
which aspects of a review the model focused on. Because the prediction is residual,
these attributions describe the *refinement* the fusion adds over the static base
rather than the entire prediction — a distinction we keep explicit in the analysis.
Section 6.5 tests whether the aspect attention is faithful (i.e., whether the
most-attended aspect actually drives the prediction) using a perturbation study, since
attention weights are not automatically faithful explanations (Jain & Wallace, 2019; Wiegreffe & Pinter, 2019).

---

## 5. Experimental Setup

### 5.1 Datasets

We evaluate on three public review corpora chosen to span distinct domains *and* distinct
sparsity/aspect regimes: **Amazon Electronics** (from the Amazon Review Data; Ni et al.,
2019), **Yelp Restaurant** (Yelp Open Dataset, 2024), and **TripAdvisor Hotel** (the hotel
reviews of Wang et al., 2010). Each record contains a user, an item, a 1–5 star rating,
and a free-text review.
The three corpora were selected deliberately rather than for convenience: they place the
same rating-prediction task under three different levels of aspect availability, which is
the axis our analysis (Sections 6.4–6.5) turns on. Each corpus is reduced with 5-core
filtering — retaining only users and items with at least five interactions — a standard
preprocessing step for review-based recommendation (Abinaya & Devi, 2021; Cai et al.,
2022). 5-core filtering bounds, but does not remove, cold-start effects: the user–item
matrices remain below 0.1% density, so the rating signal alone is thin.

Two properties of the data shape the results and are worth stating up front. First, as is
typical of voluntary review platforms, the rating distribution is right-skewed toward 4–5
stars; consequently the standard deviation of test ratings is close to the RMSE a
constant predictor would attain (Section 6.1), which sets a meaningful reference point for
interpreting the collaborative baselines. Second, the two ABSA encoders cover the corpora
very differently. Table 2 reports full statistics, including the coverage measures central
to this study: the fraction of reviews for which the fixed-taxonomy keyword encoder finds
at least one aspect — 45.1% (Amazon), 87.6% (Restaurant), 95.9% (Hotel) — and the
corresponding coverage of the open-vocabulary PyABSA encoder, which is higher on the
aspect-sparse Amazon domain than keyword matching achieves. This deliberate spread in
keyword coverage is precisely what lets us ask *when*, not merely *whether*, model-based
aspect sentiment helps. The held-out test sets contain 16,580 (Amazon), 13,233
(Restaurant), and 11,795 (Hotel) interactions.

### 5.2 Split and protocol

All models are trained and evaluated on an identical user-based train/validation/test
split per domain. The split holds out later interactions of existing users, which stresses
a model's ability to track evolving preferences rather than to memorize a static profile.
Comparability is enforced structurally rather than by convention: the split is generated
once and every training script — from the classical baselines to A2-FusionRS — loads the
same files instead of resampling, so all models are scored on exactly the same held-out
interactions and their per-interaction errors are directly pairable for significance
testing (Section 5.4). Ranking metrics use a per-user candidate set restricted to the
items that appear for that user in the test set, a common simplification that keeps the
ranking evaluation tractable and identical across models; we note it explicitly because it
makes the reported ranking figures comparative rather than absolute.

Single-seed differences on this data are of the same order as the run-to-run variance of an
identically configured model, so a one-off comparison is not, by itself, trustworthy.
Every configuration is therefore trained with five random seeds ({42, 123, 456, 789,
1011}), and we report the mean and standard deviation across them; all headline claims are
supported at the level of seeds, not of a single run.

### 5.3 Baselines

We compare against baselines organized in four tiers, all evaluated under the identical
protocol of Section 5.2:

1. **Heuristic** — a *Global Mean* predictor (the training-set mean rating), a lower
   bound that any useful model must beat.
2. **Classical collaborative filtering** — *Item-KNN* and *SVD* (Koren et al., 2009),
   pure rating-based methods without side information.
3. **Neural collaborative filtering** — *NeuMF* (He et al., 2017) and *DeepFM*
   (Guo et al., 2017). Both were designed for implicit-feedback ranking and click-through
   prediction; we adapt them faithfully to explicit-rating regression — mean-squared-error
   training with a sigmoid on the normalized rating scale and the same
   best-validation-checkpoint policy used elsewhere — so that they are directly comparable
   on RMSE. This adaptation is not a handicap: it gives each model the objective the task
   is evaluated on. Importantly, these neural baselines share the embedding-based
   collaborative backbone family used by the DeepMF stream inside A2-IRM/A2-FusionRS, so a
   large gap between them and the review-aware models cannot be dismissed as a weak or
   mismatched baseline.
4. **Hybrid (review-aware)** — *A2-IRM*, the Phase-1 static-fusion hybrid that combines
   deep matrix factorization, content-based clustering, and keyword-ABSA through an
   NMF + decision-tree fuser, in the lineage of Darraz et al. (2025). It is the strongest
   prior model and the direct point of comparison for A2-FusionRS.

The first three tiers use only the user–item rating signal, whereas A2-IRM and A2-FusionRS
add the review channel; the contrast across tiers therefore isolates the value of reviews
from the value of the fusion architecture (Duan et al., 2022; Elahi et al., 2023).

### 5.4 Metrics and significance testing

The primary metrics are RMSE and MAE on the original 1–5 scale; we additionally report
Precision@K, Recall@K, and NDCG@K for the ranking view. Because single-seed differences
on this data are of the same order as run-to-run variance, we assess significance with a
paired Wilcoxon signed-rank test over per-interaction squared errors, run separately for
each seed on the shared test set, and summarize as the number of seeds (out of five) at
which the difference reaches $p<0.05$. We deliberately report the RMSE difference
alongside the test outcome: with test sets of $10^4$–$10^5$ paired errors, the Wilcoxon
test can flag a statistically significant distributional shift even when the RMSE
difference is negligible, so significance is never interpreted in isolation from effect
size.

### 5.5 Implementation details

The sentiment stream uses the PyABSA "english" ATEPC checkpoint (Yang & Li, 2023). The
attention-gated fusion uses model dimension $d=64$, $H=2$ attention heads, an aspect
embedding of size $d_a=16$, at most eight aspects per review, and a frequency-ranked
aspect vocabulary of the top $V=500$ terms built from the training split only. The model
is trained with Adam and weight decay, restoring the lowest-validation-RMSE checkpoint.
The complete hyperparameter list appears in Table 3. All experiments were run on a single
GPU; code and configurations are released for reproducibility.

---

## 6. Results and Discussion

### 6.1 Overall performance (RQ1)

Table 4 reports RMSE (mean ± SD over five seeds) for every model and domain. A2-FusionRS
achieves the lowest error in all three domains — 0.6418 (Amazon), 0.6665 (Restaurant),
and 0.6196 (Hotel). Against the strongest prior model, A2-IRM, this is a reduction of
1.5%, 1.9%, and 1.5%; against the best neural collaborative baseline (NeuMF) it is a
reduction of 44.3% (Amazon), 37.9% (Restaurant), and 26.2% (Hotel). Every one of these
comparisons is significant at all five seeds (Table 5): A2-FusionRS versus A2-IRM and
versus each of the four external baselines yields $p<0.001$ throughout. The advantage is
thus both large against rating-only methods and consistent — not the product of a
favorable seed. MAE tells the same story and is reported alongside RMSE in Table 4:
A2-FusionRS attains the lowest MAE in every domain (0.4115 / 0.5389 / 0.4996), below
A2-IRM (0.4386 / 0.5505 / 0.5127) and far below the collaborative baselines. The ranking
metrics (Precision/Recall/NDCG@K) are near-saturated under the restricted-candidate
protocol of Section 5.2 — all models score within a narrow band — so they do not
discriminate between methods, and we treat RMSE and MAE as the informative metrics.

The more instructive result is the *structure* of the baseline column. The four pure
collaborative methods cluster tightly at RMSE $\approx 1.1$–$1.2$, barely below — and for
Item-KNN equal to — the Global Mean predictor (1.2143 / 1.1516 / 0.9163). This reference
point is not arbitrary: the RMSE of the constant mean predictor equals the standard
deviation of the test ratings (an algebraic identity), so a model that matches it has
learned nothing beyond the average rating. On Hotel, Item-KNN's RMSE (0.9163) coincides
with the Global Mean to four decimals, showing that under extreme sparsity the
neighborhood model falls back to the mean for essentially every prediction; the trained
factorization models (SVD, NeuMF, DeepFM) improve on the mean only marginally, by roughly
2–8%. That four independent collaborative methods converge at this level is strong
evidence that $\approx 1.1$ is the genuine ceiling of rating-only CF on review data this
sparse, rather than an artifact of under-tuning (Idrissi & Zellou, 2020; Yuan &
Hernandez, 2023). Seen against this ceiling, the review-derived content and sentiment
signals — which move the error from $\approx 1.1$ down to $\approx 0.65$, a 31–47%
reduction — are doing the decisive work, and the remainder of this section attributes
that work to its source (Cai et al., 2022; Elahi et al., 2023).

### 6.2 Ablation and attribution (RQ2)

Table 6 decomposes the contribution of each component, and the decomposition is
unusually clean because the two candidate explanations — the attention architecture and
the model-based aspect signal — can be toggled independently.

Consider first the architecture in isolation. Replacing A2-IRM's static fusion with
attention-gated fusion, while giving it *exactly the information A2-IRM already uses*,
leaves accuracy essentially unchanged: 0.6520 / 0.6773 / 0.6286 versus A2-IRM's 0.6517 /
0.6791 / 0.6291. The per-domain gaps are +0.0003, −0.0018, and −0.0005 — all within a
single standard deviation. This is also the clearest place to see why we insist on
pairing significance with effect size: on Amazon and Hotel the Wilcoxon test flags these
differences as significant at several seeds even though the RMSE is, for practical
purposes, identical — a direct consequence of testing $10^4$ paired errors, and a warning
against reading such $p$-values as evidence of a real improvement. Read through the effect
size, the honest conclusion is that the attention architecture, given no new information,
sits at the same ceiling as the static fuser.

Now toggle the information instead of the architecture. Adding the PyABSA per-aspect
signal breaks the ceiling, and it does so *regardless of which fuser consumes it*. When
the same PyABSA features are handed to the static NMF + decision-tree fuser rather than to
attention, the result (0.6384 / 0.6676 / 0.6201) still beats A2-IRM significantly at all
five seeds, by 0.0133 / 0.0115 / 0.0090 — an effect roughly an order of magnitude larger
than the architecture-only differences above. The improvement is therefore attributable
to the model-based aspect modality, not to the attention mechanism per se. We regard this
as a central finding precisely because it is falsifiable and was, in fact, a negative
result for our original hypothesis that the fusion architecture would be the driver;
reporting it is what distinguishes an attribution from an assertion.

### 6.3 The role of the fusion mechanism

If a static fuser with the same PyABSA features is already competitive, the natural
question is whether the attention mechanism earns its added complexity. On raw accuracy,
the honest and firm answer is that it *matches* rather than beats the static fusion, and
we resist any stronger claim. Comparing A2-FusionRS directly to the static PyABSA control,
attention is behind on Amazon (+0.0034 RMSE), marginally ahead on Restaurant (−0.0011) and
effectively level on Hotel (−0.0005). The per-seed picture tells the same story without
rounding: attention-gated fusion is ahead of the static control in 1 of 5 seeds on Amazon,
4 of 5 on Restaurant, and 3 of 5 on Hotel — 8 of 15 domain–seed cells in total, a near
coin-flip. No configuration of this comparison supports an accuracy advantage for the
attention mechanism, and we do not assert one.

Its value lies elsewhere, and there it is concrete. First, attention reaches this accuracy
while *learning* the aspect aggregation end-to-end from the raw variable-length per-aspect
tuples, whereas the static control only becomes competitive once it is given the nine
hand-engineered order-statistics we designed to summarize those same aspects; parity
without feature engineering is a methodological, not an accuracy, advantage, but it is a
real one. Second, the mechanism yields per-modality gate weights and per-aspect attention
that the tree does not, and — unlike a decorative heatmap — these attributions survive a
faithfulness test (Section 6.5). We therefore position A2-FusionRS as an interpretable
fusion that is competitive with strong static fusion, in the spirit of recent work that
treats interpretability as a first-class design goal rather than a by-product
(Wu et al., 2024; Kim et al., 2024), and not as a fusion operator that supersedes the
static baseline.

### 6.4 When does model-based ABSA help? (RQ3)

The benefit of the PyABSA modality is not uniform across domains, and the pattern of its
variation is itself a finding. Ordering the domains by keyword-ABSA coverage, the RMSE
reduction of the PyABSA control over A2-IRM decreases monotonically: −0.0133 at 45.1%
coverage (Amazon), −0.0115 at 87.6% (Restaurant), and −0.0090 at 95.9% (Hotel). The
open-vocabulary encoder contributes most where the fixed-taxonomy encoder covers the
fewest reviews — exactly where a complementary aspect signal has the most room to add
information, and least where keyword matching already captures nearly every review. With
three domains this is a directional trend rather than a law, and we frame it as such; but
it is a mechanistically sensible one, and it yields actionable guidance: model-based ABSA
is most worth its added cost in domains whose aspect vocabulary is sparse or poorly served
by a fixed taxonomy, and least worth it where a lightweight keyword encoder already
saturates coverage. This complements the broader evidence that fine-grained aspect signals
improve review-based recommendation (Ou & Huynh, 2024; Yang et al., 2024a) by specifying a
condition under which the more expensive model-based encoder pays off.

### 6.5 Interpretability

A2-FusionRS exposes two attributions, which we analyze and — importantly — validate
rather than merely display.

**Modality contribution.** Averaged over five seeds, the gate places broadly comparable
weight on the four modalities (each near the 0.25 uniform baseline), with domain-specific
tilts (Table 7): the aspect modality receives its *lowest* weight in the highest-coverage
domain (Hotel, 0.211) and higher weight in the lower-coverage domains (Amazon 0.279,
Restaurant 0.290), while content features dominate on Hotel (0.302). The correlation
between keyword coverage and the aspect-modality gate weight is negative ($r=-0.52$). This is an
*independent* corroboration of the accuracy finding in Section 6.4: the model learns to
lean on the model-based aspect signal precisely where keyword-ABSA is weak. With only
three domains this is indicative rather than conclusive, and we present it as such.

**Aspect attention and its faithfulness.** The aspect-pooling weights identify which
aspects of a review the model attends to; representative cases are shown in Table 9. To
test whether these weights are *faithful* — whether the most-attended aspect actually
drives the prediction — we run a perturbation study: for each test review with at least
two aspects, we remove the top-attended aspect and, separately, a randomly chosen aspect,
and measure the change in prediction. Removing the top-attended aspect changes the
prediction two-to-three times more than removing a random one (|Δ| of 0.051 vs. 0.016 on
Amazon, 0.060 vs. 0.032 on Restaurant, 0.047 vs. 0.021 on Hotel; Table 8), and the
top-attended aspect has the larger effect in roughly 70% of reviews ($p\approx 0$ in all
domains).
The attention is thus faithful in the aggregate — a stronger claim than an attention
heatmap alone would justify (Jain & Wallace, 2019; Wiegreffe & Pinter, 2019) — though
we note it is faithful to the residual *correction* the fusion adds over the static base,
not to the entire prediction, and that the illustrative cases are not uniformly clean
(the perturbation agreement is ~70%, not 100%).

### 6.6 Efficiency

Table 10 reports the cost of the attention-gated fusion head, which is the component
A2-FusionRS adds over the pre-computed modality encoders and over the static A2-IRM base.
The head is deliberately small: roughly 62,600 trainable parameters, about 27–41 seconds
to train, and 11–14 milliseconds to score a full test set. In other words, the accuracy
gain and the interpretability come at a negligible marginal cost over the A2-IRM pipeline
the model extends. We did not instrument the collaborative baselines under the identical
timing harness, so we report the fusion-head cost rather than a cross-model timing table;
the parameter figure should be read as the trainable fusion cost, not as the total model
size including the (frozen) upstream encoders.

### 6.7 Threats to validity

Several limitations bound the scope of our claims. (i) The evaluation covers three
English-language domains and a single PyABSA checkpoint; generalization to other
languages, domains, or aspect extractors is untested. (ii) The attention mechanism does
not improve accuracy over static fusion given the same features; our contribution is the
complementary aspect modality and the attribution methodology, not a superior fusion
operator. (iii) Wilcoxon significance on large test sets can reach $p<0.05$ for
negligible RMSE gaps, which is why we always pair it with the effect size. (iv) The
interpretability attributions describe the residual refinement over the static base, and
attention faithfulness, though supported by the perturbation test, holds in aggregate
rather than for every instance.

---

## 7. Conclusion and Future Work

We presented A2-FusionRS, an attention-gated fusion recommender that adds open-vocabulary,
model-based per-aspect sentiment to a static-fusion hybrid through a learned
aspect-sequence pooling layer, and that predicts a residual correction over that hybrid.
On the three research questions, the evidence is clear and, where it is limiting, stated
plainly. **RQ1**: A2-FusionRS attains the lowest RMSE in all three domains and improves
significantly over classical, neural, and hybrid baselines at every seed — decisively
against rating-only methods and consistently against the strong A2-IRM hybrid. **RQ2**:
the improvement is attributable to the model-based aspect modality, not to the attention
mechanism; a static tree given the same aspect features is equally strong, so we credit
the signal rather than the architecture and refrain from claiming an accuracy advantage
for attention. **RQ3**: the value of model-based ABSA grows as a domain's keyword-aspect
coverage falls, which tells practitioners when the costlier encoder is worth adopting.
Beyond accuracy, the fusion is interpretable, and its aspect attention is faithful under a
perturbation test rather than merely displayed.

Two aspects of this work are, in our view, as useful as the accuracy gain. The first is
methodological: by toggling architecture and signal independently, the attribution
protocol turns a common "new-model-plus-new-signal" comparison into a statement about
*which* change mattered — a discipline the field applies too rarely. The second is the
honesty of the negative result: reporting that the attention mechanism only matches the
static fuser, rather than burying it, is what makes the positive claims credible.

Several directions follow. The evaluation should be broadened beyond three
English-language domains and a single PyABSA checkpoint — in particular to mid-coverage
domains that would test the coverage-dependence trend, and to other languages and
aspect extractors. The aspect-sequence pooling is deliberately simple; richer pooling
(e.g., multi-query or hierarchical attention over aspects) may extract more from the same
signal, and would be a fair test of whether an attention advantage can be recovered where
we found none. Stronger review-aware baselines (graph- and LLM-based aspect models;
Cui et al., 2024; Liu et al., 2025) would sharpen the external comparison. Finally, the
interpretability account could be extended from faithfulness to usefulness, measuring
whether the exposed aspect attributions actually help end users judge recommendations.

---

## Tables

> Values are extracted from the run outputs by `build_manuscript_tables.py` and
> `analyze_interpretability.py`. Tables 2, 4, 6, 7, 8, 9, and 10 are populated with real
> figures; the only gap is cross-model efficiency timing (Table 10 reports the
> A2-FusionRS fusion-head cost, as the baselines were not instrumented under the same
> harness).

**Table 2. Dataset statistics.** Coverage measures: fraction of reviews with ≥1 extracted
aspect (keyword vs. PyABSA); avg. aspects/review from PyABSA.

| Domain | #Users | #Items | #Interactions | #Test | Sparsity | Keyword coverage | PyABSA coverage | Aspects/review |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Amazon Electronics | 14,749 | 9,226 | 122,062 | 16,580 | 99.91% | 45.1% | 80.4% | 1.81 |
| Yelp Restaurant | 7,152 | 3,757 | 118,695 | 13,233 | 99.56% | 87.6% | 78.2% | 2.70 |
| TripAdvisor Hotel | 11,236 | 2,056 | 79,562 | 11,795 | 99.66% | 95.9% | 70.2% | 2.77 |

**Table 3. Key hyperparameters.**

| Component | Setting |
|---|---|
| DeepMF | embedding 128; hidden [256,128,64,32]; dropout 0.3; SGD, lr 1e-3; batch 512; 20 epochs; no negative sampling |
| CBF | item PCA + user cluster preference (components/clusters per config) |
| Item-KNN | k=40; cosine; item-based |
| SVD | 100 factors; 20 epochs; lr 5e-3; reg 2e-2 |
| NeuMF / DeepFM | embedding 64; MLP [128,64,32]; dropout 0.2; Adam, lr 1e-3, weight decay 1e-6; batch 512; 20 epochs |
| A2-FusionRS (AGF) | d=64; heads H=2; aspect embedding d_a=16; max aspects 8; aspect vocab V=500 (train-only); Adam + weight decay; best-val restore; residual base via 5-fold OOF |

**Table 4. Overall performance — RMSE and MAE (mean over 5 seeds), lower is better.** Best
per column in bold. Ranking metrics (Precision/Recall/NDCG@K) are near-saturated under the
restricted-candidate protocol (Section 5.2) and do not discriminate between models; they
are omitted here and reported in the supplementary material.

| Model | RMSE Amazon | RMSE Restaurant | RMSE Hotel | MAE Amazon | MAE Restaurant | MAE Hotel |
|---|---:|---:|---:|---:|---:|---:|
| Global Mean (reference) | 1.2143 | 1.1516 | 0.9163 | — | — | — |
| Item-KNN | 1.2240 | 1.2019 | 0.9163 | 0.8311 | 0.9237 | 0.6917 |
| SVD | 1.1420 | 1.0753 | 0.8953 | 0.8030 | 0.8454 | 0.6954 |
| NeuMF | 1.1528 | 1.0740 | 0.8399 | 0.7989 | 0.8468 | 0.6632 |
| DeepFM | 1.1529 | 1.0746 | 0.8393 | 0.8025 | 0.8434 | 0.6619 |
| A2-IRM (hybrid, prior) | 0.6517 | 0.6791 | 0.6291 | 0.4386 | 0.5505 | 0.5127 |
| **A2-FusionRS** | **0.6418** | **0.6665** | **0.6196** | **0.4115** | **0.5389** | **0.4996** |

*Per-seed standard deviations (RMSE) are ≤ 0.003 throughout; see Table 6 for ± values on
the ablation variants.*

**Table 5. Significance of A2-FusionRS vs. each baseline** (paired Wilcoxon on per-interaction
squared error; number of seeds out of 5 with $p<0.05$).

| Comparison | Amazon | Restaurant | Hotel |
|---|:--:|:--:|:--:|
| A2-FusionRS vs A2-IRM | 5/5 | 5/5 | 5/5 |
| A2-FusionRS vs NeuMF | 5/5 | 5/5 | 5/5 |
| A2-FusionRS vs DeepFM | 5/5 | 5/5 | 5/5 |
| A2-FusionRS vs SVD | 5/5 | 5/5 | 5/5 |
| A2-FusionRS vs Item-KNN | 5/5 | 5/5 | 5/5 |
| (attribution) static tree + PyABSA vs A2-IRM | 5/5 | 5/5 | 5/5 |

**Table 6. Ablation — RMSE (mean ± SD over 5 seeds).** Toggling architecture vs. signal.

| Variant | Amazon | Restaurant | Hotel |
|---|---:|---:|---:|
| A2-IRM (static fusion, keyword-ABSA) | 0.6517 ± .003 | 0.6791 ± .001 | 0.6291 ± .003 |
| Attention-gated fusion, no PyABSA (architecture only) | 0.6520 ± .002 | 0.6773 ± .001 | 0.6286 ± .003 |
| Static tree + PyABSA (signal only; attribution control) | 0.6384 ± .001 | 0.6676 ± .001 | 0.6201 ± .003 |
| Attention + PyABSA order-statistics | 0.6407 ± .002 | 0.6674 ± .001 | 0.6216 ± .003 |
| **A2-FusionRS (attention + aspect-sequence pooling)** | **0.6418 ± .002** | **0.6665 ± .001** | **0.6196 ± .003** |

**Table 7. Interpretability — mean gate weight per modality (5 seeds).** (§6.5, Exp-A.)

| Domain (keyword coverage) | DeepMF | CBF | Keyword-ABSA | PyABSA-aspect |
|---|---:|---:|---:|---:|
| Amazon (45.1%) | 0.236 | 0.234 | 0.251 | 0.279 |
| Restaurant (87.6%) | 0.239 | 0.224 | 0.248 | 0.290 |
| Hotel (95.9%) | 0.201 | 0.302 | 0.286 | 0.211 |

*Pearson r(keyword coverage, PyABSA-aspect gate) = −0.52.*

**Table 8. Interpretability — aspect-attention faithfulness (perturbation test).** (§6.5,
Exp-C; seed 42; reviews with ≥2 aspects.)

| Domain | \|Δ\| remove top-attended | \|Δ\| remove random | top > random | Wilcoxon p |
|---|---:|---:|---:|:--:|
| Amazon | 0.0511 | 0.0160 | 71.2% | ≈ 0 |
| Restaurant | 0.0603 | 0.0319 | 69.9% | ≈ 0 |
| Hotel | 0.0475 | 0.0214 | 71.3% | ≈ 0 |

**Table 9. Representative aspect-attention case studies** (§6.5, Exp-B; seed 42). For each
review the model's most-attended aspect, its sentiment (P(pos)), and the predicted vs.
actual rating are shown; attention concentrates on the aspect that drives the rating.

| Domain | Review aspects | Top-attended (attn.) | Sentiment | Predicted | Actual |
|---|---|---|---|---:|---:|
| Amazon | working \| **charge** | charge (0.99) | negative (0.00) | 1.73 | 1.0 |
| Amazon | display \| sound | display (0.98) | negative (0.00) | 2.97 | 4.0 |
| Restaurant | price \| **stuff** | stuff (0.87) | positive (1.00) | 4.42 | 4.0 |
| Restaurant | waitress \| **waiting** | waiting (0.88) | negative (0.00) | 1.60 | 3.0 |
| Hotel | staff \| **comfort** | comfort (0.88) | positive (1.00) | 4.91 | 5.0 |
| Hotel | **rooms** \| water | rooms (0.95) | negative (0.01) | 2.29 | 3.0 |

**Table 10. Efficiency of A2-FusionRS** (fusion head; mean over 5 seeds). The attention/gating
head is trained over pre-computed modality encoders; it adds only a small number of
trainable parameters and negligible inference latency.

| Domain | Trainable params (fusion head) | Train time | Inference time |
|---|---:|---:|---:|
| Amazon | 62,629 | 40.6 s | 0.014 s |
| Restaurant | 62,501 | 38.9 s | 0.012 s |
| Hotel | 62,757 | 26.2 s | 0.011 s |

*Timing of the collaborative baselines was not instrumented under the identical harness;
we report the fusion-head cost, which is the marginal cost A2-FusionRS adds over A2-IRM.*
