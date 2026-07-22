# A2-FusionRS: Attention-Gated Fusion of Model-Based Aspect Sentiment for Review-Aware Recommendation under Extreme Sparsity

> **Working draft — target: Expert Systems with Applications (Q1).** Numbers cite
> `A2-FusionRS_results_ledger.md`; `[REF: ...]` marks a citation to be sourced and
> verified. Sections are drafted in the order Method → Results → Introduction/Related
> Work (most code-grounded first). This file currently contains §3 Preliminaries and
> §4 Proposed Method.

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
and NMF + decision-tree fusion in the lineage of Darraz et al. (2025) [CONFIRM: this
is the Phase-1 predecessor citation] — and extended here:

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

We evaluate on three public review corpora that span distinct domains and sparsity
regimes: **Amazon Electronics**, **Yelp Restaurant**, and **TripAdvisor Hotel**. Each
record contains a user, an item, a 1–5 star rating, and a free-text review. Following
common practice for review-based recommendation (Cai et al., 2022; Yang et al., 2024),
the corpora are reduced with 5-core filtering so that every retained user and item has
at least five interactions, which bounds — but does not eliminate — cold-start effects.
Table 2 reports the resulting statistics, including the two ABSA coverage measures that
matter for this study: the fraction of reviews for which the fixed-taxonomy keyword
encoder finds at least one aspect, and the corresponding coverage of the open-vocabulary
PyABSA encoder. Keyword coverage differs markedly across domains — 45.1% (Amazon),
87.7% (Restaurant), and 95.9% (Hotel) — a spread we exploit in the analysis of
Section 6.4. The held-out test sets contain 16,580 (Amazon), 13,233 (Restaurant), and
11,795 (Hotel) interactions. [REF: dataset provenance — Amazon/McAuley, Yelp Open
Dataset, TripAdvisor — to be added.]

### 5.2 Split and protocol

All models are trained and evaluated on an identical user-based train/validation/test
split per domain, generated once and reused, so that every method is compared on exactly
the same held-out interactions. This invariant is enforced structurally: the split is
produced once and every training script loads it rather than resampling. To separate
genuine effects from run-to-run stochasticity, each configuration is trained with five
random seeds ({42, 123, 456, 789, 1011}); we report the mean and standard deviation
across seeds.

### 5.3 Baselines

We compare against baselines organized in four tiers, all evaluated under the identical
protocol of Section 5.2:

1. **Heuristic** — a *Global Mean* predictor (the training-set mean rating), a lower
   bound that any useful model must beat.
2. **Classical collaborative filtering** — *Item-KNN* and *SVD* (Koren et al., 2009),
   pure rating-based methods without side information.
3. **Neural collaborative filtering** — *NeuMF* (He et al., 2017) and *DeepFM*
   (Guo et al., 2017), re-implemented as explicit-rating predictors (mean-squared-error
   training, sigmoid on the normalized scale) so that they are directly comparable on
   RMSE.
4. **Hybrid (review-aware)** — *A2-IRM*, the Phase-1 static-fusion hybrid that combines
   deep matrix factorization, content-based clustering, and keyword-ABSA through an
   NMF + decision-tree fuser, in the lineage of Darraz et al. (2025).

The classical and neural collaborative baselines use only the user–item rating signal;
the contrast between them and the review-aware methods isolates the value of the review
channel (Duan et al., 2022; Elahi et al., 2023).

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
and 0.6196 (Hotel) — improving over the strongest prior model, A2-IRM, by 1.5%, 1.9%,
and 1.5% respectively. The improvement over A2-IRM is significant at all five seeds in
every domain (Table 5), as is the improvement over each of the four external baselines
($p<0.001$ throughout).

A striking pattern organizes the baselines. The four pure collaborative methods cluster
tightly at RMSE $\approx 1.1$–$1.2$, i.e., barely better than — and for Item-KNN
essentially equal to — the Global Mean predictor (1.2143 / 1.1516 / 0.9163). On Hotel,
Item-KNN's RMSE (0.9163) coincides with the Global Mean exactly, indicating that under
extreme sparsity the neighborhood model degenerates to predicting the mean. That four
independent collaborative methods converge at this level is strong evidence it reflects
the genuine ceiling of rating-only CF on such sparse review data, not an under-tuned
baseline (Idrissi & Zellou, 2020; Yuan & Hernandez, 2023). The review-derived content and
sentiment signals are what move the error from $\approx 1.1$ down to $\approx 0.65$ — a
31–47% reduction — which is precisely the contribution of review-aware modeling
(Cai et al., 2022; Elahi et al., 2023).

### 6.2 Ablation and attribution (RQ2)

Table 6 decomposes the contribution of each component. Removing the model-based aspect
signal from A2-FusionRS (leaving attention-gated fusion over the same information A2-IRM
uses) yields RMSE statistically indistinguishable from A2-IRM (0.6520 / 0.6773 / 0.6286
vs. 0.6517 / 0.6791 / 0.6291): the attention architecture alone, with no new information,
sits at the same ceiling. Conversely, adding the PyABSA per-aspect signal breaks that
ceiling — and it does so *regardless of the fusion mechanism*. A control in which the
same PyABSA features are fused by the static NMF + decision-tree fuser (rather than by
attention) also beats A2-IRM significantly at all seeds (0.6384 / 0.6676 / 0.6201). The
gain is therefore attributable to the model-based aspect modality, not to the attention
fusion per se — an attribution we consider a core, and deliberately falsifiable, finding.

### 6.3 The role of the fusion mechanism

Given that a static fuser with the same PyABSA features is competitive, does the
attention mechanism add anything? On raw accuracy, the honest answer is: it matches,
rather than beats, the static fusion. Across seeds, attention-gated fusion is behind the
static control on Amazon (+0.0034 RMSE), marginally ahead on Restaurant (−0.0011, four of
five seeds) and level on Hotel (−0.0005, two of five seeds) — a wash overall. We
therefore do **not** claim an accuracy advantage for the attention mechanism. Its value
is elsewhere and is real: it reaches this accuracy while *learning* the aspect
aggregation end-to-end, whereas the static control requires the hand-engineered
order-statistics we designed for it; and it exposes per-modality and per-aspect
attributions that the tree does not (Section 6.5). This positions A2-FusionRS as an
interpretable fusion that is competitive with strong static fusion, not as a
mechanism that supersedes it — a distinction that recent interpretable-recommendation
work also foregrounds (Wu et al., 2024; Kim et al., 2024).

### 6.4 When does model-based ABSA help? (RQ3)

The benefit of the PyABSA modality is not uniform across domains, and its variation is
mechanistically informative. The RMSE reduction of the PyABSA control over A2-IRM is
largest where keyword-ABSA coverage is lowest — −0.0133 on Amazon (45.1% coverage),
−0.0115 on Restaurant (87.7%), and −0.0090 on Hotel (95.9%). In other words, the
open-vocabulary encoder contributes most where the fixed-taxonomy encoder is weakest,
which is exactly where a complementary aspect signal should matter most. This
coverage-dependent behavior offers practical guidance: model-based ABSA is most worth its
cost in domains with sparse or ill-fitting aspect taxonomies (Ou et al., 2024;
Yang et al., 2024).

### 6.5 Interpretability

A2-FusionRS exposes two attributions, which we analyze and — importantly — validate
rather than merely display.

**Modality contribution.** Averaged over five seeds, the gate places broadly comparable
weight on the four modalities (each near the 0.25 uniform baseline), with domain-specific
tilts: the aspect modality receives its *lowest* weight in the highest-coverage domain
(Hotel, 0.211) and higher weight in the lower-coverage domains (Amazon 0.279, Restaurant
0.290), while content features dominate on Hotel (0.302). The correlation between keyword
coverage and the aspect-modality gate weight is negative ($r=-0.52$). This is an
*independent* corroboration of the accuracy finding in Section 6.4: the model learns to
lean on the model-based aspect signal precisely where keyword-ABSA is weak. With only
three domains this is indicative rather than conclusive, and we present it as such.

**Aspect attention and its faithfulness.** The aspect-pooling weights identify which
aspects of a review the model attends to; representative cases are shown in Table 8. To
test whether these weights are *faithful* — whether the most-attended aspect actually
drives the prediction — we run a perturbation study: for each test review with at least
two aspects, we remove the top-attended aspect and, separately, a randomly chosen aspect,
and measure the change in prediction. Removing the top-attended aspect changes the
prediction two-to-three times more than removing a random one (|Δ| of 0.051 vs. 0.016 on
Amazon, 0.060 vs. 0.032 on Restaurant, 0.047 vs. 0.021 on Hotel), and the top-attended
aspect has the larger effect in roughly 70% of reviews ($p\approx 0$ in all domains).
The attention is thus faithful in the aggregate — a stronger claim than an attention
heatmap alone would justify (Jain & Wallace, 2019; Wiegreffe & Pinter, 2019) — though
we note it is faithful to the residual *correction* the fusion adds over the static base,
not to the entire prediction, and that the illustrative cases are not uniformly clean
(the perturbation agreement is ~70%, not 100%).

### 6.6 Efficiency

[Table 7: parameter counts and train/inference times per model. A2-FusionRS adds a small
attention/gating head over the pre-computed modality encoders; its parameter count and
inference latency are modest relative to the neural CF baselines — numbers to be filled
from the recorded instrumentation.]

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
