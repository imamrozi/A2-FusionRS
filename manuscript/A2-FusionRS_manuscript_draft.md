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

We evaluate on three public review corpora chosen to span distinct domains *and* distinct
sparsity/aspect regimes: **Amazon Electronics**, **Yelp Restaurant**, and **TripAdvisor
Hotel**. Each record contains a user, an item, a 1–5 star rating, and a free-text review.
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
at least one aspect — 45.1% (Amazon), 87.7% (Restaurant), 95.9% (Hotel) — and the
corresponding coverage of the open-vocabulary PyABSA encoder, which is higher on the
aspect-sparse Amazon domain than keyword matching achieves. This deliberate spread in
keyword coverage is precisely what lets us ask *when*, not merely *whether*, model-based
aspect sentiment helps. The held-out test sets contain 16,580 (Amazon), 13,233
(Restaurant), and 11,795 (Hotel) interactions. [REF: dataset provenance — Amazon/McAuley,
Yelp Open Dataset, TripAdvisor — to be added.]

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
favorable seed. The MAE and ranking metrics (Precision/Recall/NDCG@K) preserve the same
ordering and are reported in Table 4.

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
coverage (Amazon), −0.0115 at 87.7% (Restaurant), and −0.0090 at 95.9% (Hotel). The
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
