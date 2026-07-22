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

## B. Verified — from the author's SLR collection (Zotero export; real DOIs)

> Selected as most relevant to A2-FusionRS. All DOIs are from the author's own
> Zotero/Crossref export → low fabrication risk. Quartile still to confirm on SJR,
> but venues below are predominantly recognized Q1–Q2 (ESWA, Information Sciences,
> IPM, Neurocomputing, Neural Networks, ACM TOIS/TIST, DSS, KBS-family).

**Phase-1 predecessor / closest hybrids (DeepMF + sentiment + NMF-DT lineage)**
| Key | Citation | Venue | DOI |
|---|---|---|---|
| darraz2025irm | Darraz N., Karabila I., El-Ansari A., Alami N., El Mallahi M. Integrated sentiment analysis with BERT for enhanced hybrid recommendation systems. *ESWA*, 261:125533, 2025. | ESWA (Q1) | 10.1016/j.eswa.2024.125533 |
| darraz2025deepmf | Darraz N., et al. Advancing recommendation systems with DeepMF and hybrid sentiment analysis. *ESWA*, 279:127432, 2025. | ESWA (Q1) | 10.1016/j.eswa.2025.127432 |
| karabila2025 | Karabila I., et al. A hybrid approach combining sentiment analysis and deep learning to mitigate data sparsity in recommender systems. *Neurocomputing*, 636:129886, 2025. | Neurocomputing (Q1) | 10.1016/j.neucom.2025.129886 |
| karabila2023 | Karabila I., et al. BERT-enhanced sentiment analysis for personalized e-commerce recommendations. *Multimedia Tools Appl.*, 83:56463–56488, 2023. | MTAP (Q2) | 10.1007/s11042-023-17689-5 |

**Aspect-based sentiment for recommendation**
| Key | Citation | Venue | DOI |
|---|---|---|---|
| cai2022 | Cai Y., Ke W., Cui E., Yu F. A deep recommendation model of cross-grained sentiments of user reviews and ratings (DeepCGSR). *Information Processing & Management*, 59(2):102842, 2022. | IPM (Q1) | 10.1016/j.ipm.2021.102842 |
| ou2024 | Ou W., Huynh V.-N. Aspect-level item recommendation based on user reviews with variational autoencoders. *Information Sciences*, 671:120655, 2024. | Inf. Sci. (Q1) | 10.1016/j.ins.2024.120655 |
| kim2024axcf | Kim D., Li Q., Jang D., Kim J. AXCF: Aspect-based collaborative filtering for explainable recommendations. *Expert Systems*, 41(8):e13594, 2024. | Expert Systems (Q2) | 10.1111/exsy.13594 |
| yang2024a_ijhm | Yang S., Li Q., Jang D., Kim J. (**2024a**) Developing personalized restaurant recommendation model with ABSA. *Int. J. Hospitality Management*, 121:103803, 2024. | IJHM (Q1) | 10.1016/j.ijhm.2024.103803 |
| yang2024b_aarn | Yang S., Li Q., Lim H., Kim J. (**2024b**) An attentive aspect-based recommendation model with deep neural network (AARN). *IEEE Access*, 12:5781–5791, 2024. | IEEE Access (Q1/Q2) | 10.1109/ACCESS.2023.3349291 |
| poudel2022 | Poudel S., Bikdash M. Collaborative filtering based on multi-level user clustering and aspect sentiment. *Data & Information Management*, 6(4):100021, 2022. | DIM | 10.1016/j.dim.2022.100021 |
| zhang2023magnn | Zhang C., et al. Multi-aspect enhanced graph neural networks for recommendation. *Neural Networks*, 157:90–102, 2023. | Neural Networks (Q1) | 10.1016/j.neunet.2022.10.001 |
| cui2024rakcr | Cui Y., et al. RAKCR: Reviews sentiment-aware knowledge graph convolutional networks. *ESWA*, 248:123403, 2024. | ESWA (Q1) | 10.1016/j.eswa.2024.123403 |
| lai2021xgb | Lai C.-H., Liu D.-R., Lien K.-S. A hybrid of XGBoost and aspect-based review mining with attention neural network. *Int. J. Machine Learning & Cybernetics*, 12:1203–1217, 2021. | IJMLC (Q1/Q2) | 10.1007/s13042-020-01229-w |

**Sentiment/review-based recommendation & rating-review inconsistency**
| Key | Citation | Venue | DOI |
|---|---|---|---|
| abinaya2021 | Abinaya S., Devi M.K.K. Enhancing Top-N recommendation using stacked autoencoder in context-aware recommender system. *Neural Processing Letters*, 53:1865–1888, 2021. (uses Amazon 5-core) | Neural Process. Lett. (Q2) | 10.1007/s11063-021-10475-0 |
| rabiu2022 | Rabiu I., Salim N., Da'u A., Nasser M. Modeling sentimental bias and temporal dynamics for adaptive deep recommendation. *ESWA*, 191:116262, 2022. | ESWA (Q1) | 10.1016/j.eswa.2021.116262 |
| elahi2023 | Elahi M., et al. Hybrid recommendation by incorporating the sentiment of product reviews. *Information Sciences*, 625:738–756, 2023. | Inf. Sci. (Q1) | 10.1016/j.ins.2023.01.051 |
| aramanda2023 | Aramanda A., Md.Abdul S., Vedala R. enemos-p: enhanced emotion specific prediction (rating-review inconsistency). *ESWA*, 227:120190, 2023. | ESWA (Q1) | 10.1016/j.eswa.2023.120190 |
| duan2022 | Duan R., Jiang C., Jain H.K. Combining review-based CF and matrix factorization: a solution to rating sparsity. *Decision Support Systems*, 156:113748, 2022. | DSS (Q1) | 10.1016/j.dss.2022.113748 |
| zhan2023 | Zhan Z., Xu B. Analyzing review sentiments and product images by parallel deep nets. *IPM*, 60(1):103166, 2023. | IPM (Q1) | 10.1016/j.ipm.2022.103166 |
| yangc2021 | Yang C., Chen X., Liu L., Sweetser P. Leveraging semantic features for recommendation: sentence-level emotion analysis. *IPM*, 58(3):102543, 2021. | IPM (Q1) | 10.1016/j.ipm.2021.102543 |
| lai2021is | Lai C.-H., Hsu C.-Y. Rating prediction based on combination of review mining and user preference analysis. *Information Systems*, 99:101742, 2021. | Inf. Systems (Q1) | 10.1016/j.is.2021.101742 |
| choudhary2023 | Choudhary C., Singh I., Kumar M. SARWAS: deep ensemble learning for sentiment-based recommendation. *ESWA*, 216:119420, 2023. | ESWA (Q1) | 10.1016/j.eswa.2022.119420 |

**Interpretable / explainable recommendation**
| Key | Citation | Venue | DOI |
|---|---|---|---|
| wu2024direct | Wu X., Wan H., Tan Q., Yao W., Liu N. DIRECT: dual interpretable recommendation with multi-aspect word attribution. *ACM Trans. Intelligent Systems and Technology*, 15(5):1–21, 2024. | ACM TIST (Q1) | 10.1145/3663483 |
| liu2025sagcn | Liu F., et al. Understanding before recommendation: semantic aspect-aware review exploitation via LLMs (SAGCN). *ACM Trans. Information Systems*, 43(2):1–26, 2025. | ACM TOIS (Q1) | 10.1145/3704999 |
| chang2024ers | Chang T., Zhang Z., Cai X. Explainable recommender directed by reconstructed factors and multi-modal MF. *Concurrency Computat. Pract. Exper.*, 36(21):e8208, 2024. | CCPE (Q3) | 10.1002/cpe.8208 |

**Surveys — sparsity, cold-start, RS**
| Key | Citation | Venue | DOI |
|---|---|---|---|
| idrissi2020 | Idrissi N., Zellou A. A systematic literature review of sparsity issues in recommender systems. *Social Network Analysis and Mining*, 10:15, 2020. | SNAM (Q1/Q2) | 10.1007/s13278-020-0626-2 |
| yuan2023 | Yuan H., Hernandez A.A. User cold start problem in recommendation systems: a systematic review. *IEEE Access*, 11:136958–136977, 2023. | IEEE Access | 10.1109/ACCESS.2023.3338705 |
| saifudin2024 | Saifudin I., Widiyaningtyas T. Systematic literature review on recommender system. *IEEE Access*, 12:19827–19847, 2024. | IEEE Access | 10.1109/ACCESS.2024.3359274 |
| tiwary2024 | Tiwary N., et al. A review of explainable recommender systems utilizing knowledge graphs and RL. *IEEE Access*, 12:91999–92019, 2024. | IEEE Access | 10.1109/ACCESS.2024.3422416 |

## Notes / to confirm
- **darraz2025irm** is the presumed Phase-1 (A2-IRM) predecessor — CONFIRM with author.
- Dataset provenance (Amazon/McAuley, Yelp Open Dataset, TripAdvisor) still needs
  a source citation each — not in the SLR set; source separately.
- Statistical-comparison methodology (Wilcoxon signed-rank for ML) — optional
  Demšar (2006) *JMLR* citation; not in the SLR set.
- Full SLR CSV retained by author (~90 items) — remainder available if more
  Related-Work breadth is needed.
