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
A2-IRM baseline [REF: A2-IRM / prior work] and extended here:

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

**Model-based per-aspect set.** The open-vocabulary encoder (PyABSA, ATEPC
checkpoint [REF: PyABSA/ATEPC]) extracts, for each review, a variable-length set of
$m_{ui}$ aspect tuples

$$
\mathcal{A}_{ui}=\bigl\{(a_j,\;p^{\mathrm{neg}}_j,\;p^{\mathrm{neu}}_j,\;p^{\mathrm{pos}}_j,\;c_j)\bigr\}_{j=1}^{m_{ui}},
\tag{2}
$$

where $a_j$ is an aspect term, $(p^{\mathrm{neg}}_j,p^{\mathrm{neu}}_j,p^{\mathrm{pos}}_j)$
its predicted polarity distribution, and $c_j$ the model confidence. Because $m_{ui}$
varies and the aspect vocabulary is open, this set cannot be mapped to a fixed set of
columns without discarding information; averaging across aspects, in particular,
destroys the polarity contrast *between* aspects — the same failure mode observed for
mean-pooled sentiment in A2-IRM [REF: A2-IRM]. We therefore summarize $\mathcal{A}_{ui}$
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
attention weights are not automatically faithful explanations [REF: Jain & Wallace 2019; Wiegreffe & Pinter 2019].
