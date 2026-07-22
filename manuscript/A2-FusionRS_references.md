# A2-FusionRS — Reference Tracker (verified, with DOI + quartile status)

> **Honesty policy.** Only references confirmed to exist via a web/DOI lookup are
> listed. DOIs shown were seen in search results — **verify each against
> doi.org before submission.** Scopus quartile (SJR) is **NOT machine-verifiable
> here**; the "Quartile" column is my best estimate from the venue's standing and
> **MUST be confirmed on scimagojr.com / Scopus Sources by the author.**
>
> Target for an ESWA (Q1) paper: ~45–60 references, **majority Scopus Q1–Q3
> journals**; a minority of foundational top-tier conferences (WWW, IJCAI,
> NeurIPS, NAACL/EMNLP, CIKM) is standard and accepted for canonical methods.
> Q4 minimized. Prefer DOI; a highly-relevant Q1–Q3 item without DOI is allowed.

## A. Verified — foundational methods (mostly top conferences + 1 IEEE journal)

| Key | Citation | Venue | Type | DOI (verify) | Quartile (verify) |
|---|---|---|---|---|---|
| koren2009 | Koren Y., Bell R., Volinsky C. Matrix factorization techniques for recommender systems. *IEEE Computer*, 42(8):30–37, 2009. | IEEE Computer | Journal | 10.1109/MC.2009.263 | Q1 (est.) |
| he2017ncf | He X., Liao L., Zhang H., Nie L., Hu X., Chua T.-S. Neural collaborative filtering. *Proc. WWW '17*, 173–182, 2017. | WWW | Conf | 10.1145/3038912.3052569 | Core A* conf |
| guo2017deepfm | Guo H., Tang R., Ye Y., Li Z., He X. DeepFM: A factorization-machine based neural network for CTR prediction. *Proc. IJCAI 2017*, 1725–1731. | IJCAI | Conf | 10.24963/ijcai.2017/239 | Core A* conf |
| vaswani2017 | Vaswani A., et al. Attention is all you need. *Proc. NeurIPS 2017*, 5998–6008. | NeurIPS | Conf | (arXiv:1706.03762; no DOI) | Core A* conf |
| devlin2019bert | Devlin J., Chang M.-W., Lee K., Toutanova K. BERT: Pre-training of deep bidirectional transformers for language understanding. *Proc. NAACL-HLT 2019*, 4171–4186. | NAACL | Conf | 10.18653/v1/N19-1423 | Core A conf |
| yang2023pyabsa | Yang H., Li K. PyABSA: A modularized framework for reproducible aspect-based sentiment analysis. *Proc. CIKM '23*, 5117–5122, 2023. | CIKM (ACM) | Conf | 10.1145/3583780.3614752 | Core A conf |
| jain2019attn | Jain S., Wallace B.C. Attention is not explanation. *Proc. NAACL-HLT 2019*, 3543–3556. | NAACL | Conf | 10.18653/v1/N19-1357 | Core A conf |
| wiegreffe2019 | Wiegreffe S., Pinter Y. Attention is not not explanation. *Proc. EMNLP-IJCNLP 2019*, 11–20. | EMNLP | Conf | 10.18653/v1/D19-1002 | Core A conf |

## B. Verified — domain journals (Q1–Q3 targets for Related Work)

| Key | Citation | Venue | Type | DOI/PII (verify) | Quartile (verify) |
|---|---|---|---|---|---|
| absaSurvey2020 | Zhou J., et al. Aspect-based sentiment analysis: A survey of deep learning methods. *IEEE Trans. Computational Social Systems*, 7(6):1358–1375, 2020. | IEEE T-CSS | Journal | (verify DOI) | Q1 (est.) |
| absaSurvey2023 | (Authors) Aspect-based sentiment analysis using deep learning approaches: A survey. *Computer Science Review*, 2023. | Computer Science Review | Journal | PII S1574013723000436 | Q1 (est.) |
| fedAspectGNN2025 | (Authors) FedAspect-GNN: Integrating aspect-level sentiment analysis and graph neural networks for federated recommendation. *Expert Systems with Applications*, 2025. | ESWA | Journal | PII S0957417425041442 | Q1 (est.) |

## Still to source (Q1–Q3 journals preferred) — for Intro/Related Work
- Data sparsity & cold-start in recommendation (survey / method) — ~2
- Review-based / content-based recommendation (deep) — ~4
- Sentiment-aware recommendation (recent, Q1–Q3 journal) — ~4
- ABSA methods (ATEPC/LCF, open-vocab aspect extraction) — ~3
- Attention / multimodal & gated fusion in RS — ~4
- Hybrid recommender systems — ~3
- Evaluation methodology (Wilcoxon significance in ML; RS evaluation protocol) — ~2
- Domain datasets (Amazon, Yelp, TripAdvisor) provenance — ~3
- Explainable recommendation — ~3

> Workflow: these are sourced via WebSearch as each Related-Work paragraph is
> written, then the author confirms quartile + DOI on Scopus/SJR.
