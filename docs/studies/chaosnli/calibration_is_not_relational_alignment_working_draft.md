# Calibration Is Not Relational Alignment: Human Disagreement Geometry in Natural Language Inference

**Paper Draft based on Experiments E001-E003**  
**Author:** Jonathan Lane  
**Affiliation:** [Affiliation to confirm]  
**Date:** August 2026

## Abstract

Natural language inference (NLI) models are commonly evaluated against a majority label or, increasingly, against the empirical distribution of human annotations. These pointwise evaluations do not ask whether a model organizes examples according to the same relational structure as human judgments. We introduce a relational evaluation framework for human label variation using ChaosNLI. For 3,113 SNLI and MNLI examples with 100 human annotations each, we construct an expected fuzzy neighborhood graph by averaging tie-aware k-nearest-neighbor graphs over 500 Dirichlet posterior draws. We then evaluate nine MNLI-fine-tuned transformer checkpoints by the human posterior support of their selected neighborhood mass. All models exceed stratified identity-permutation nulls, and their ranking is invariant across the tested metrics and neighborhood scales; however, exact-vote-profile controls indicate that this alignment is explained by broad human disagreement profiles rather than item-specific relational identity. In five-fold cross-fitted calibration experiments, temperature scaling closes 24.8%-56.6% of the reducible soft-label negative-log-likelihood gap but less than 1% of the relational gap, even while changing 13%-31% of neighborhood mass. Finally, increasingly flexible post-hoc transformations of BART-Large close less than 1% of its remaining relational gap, whereas probability ensembles spanning BART, RoBERTa, and XLNet close approximately 17%, increasing human-normalized relational recovery from 19.6% to roughly 33%-34%. Equal, NLL-selected, and topology-selected weighting produced statistically indistinguishable relational results (all direct $\Delta G_Q$ contrasts $< 1.0\%$ with 95% CIs spanning zero), indicating that model combination—not the particular global weighting objective—accounted for the gain. Exact-vote-profile controls yielded negligible excess support ($Q_{\text{profile-excess}} \approx 0.00000$, $p \ge 0.73$) for every ensemble, showing that the observed improvement concerns the organization of aggregate human judgment distributions rather than detectable item-specific alignment beyond those distributions. These results distinguish marginal probability alignment from relational alignment and are consistent with complementary information across model outputs, although the experiment does not isolate the exact mechanism of the ensemble gain.

## 1 Introduction

Natural language inference asks whether a hypothesis is entailed by, contradicted by, or neutral with respect to a premise. Although benchmark datasets commonly collapse annotations into a single majority label, disagreement is often systematic rather than disposable noise. Pavlick and Kwiatkowski (2019) showed that disagreements can persist as additional judgments and context are collected, while Nie et al. (2020) demonstrated substantial label variation in SNLI and MNLI through the ChaosNLI dataset. More broadly, human label variation has been argued to affect data construction, modeling, evaluation, and the meaning of ground truth itself (Plank, 2022).

Most work that preserves disagreement nevertheless remains pointwise. For each item i, a model probability vector q_i is compared with an empirical human vote distribution p_i using negative log-likelihood, Jensen-Shannon divergence, entropy, calibration error, or related quantities. This is necessary but incomplete. Two systems can have similar average pointwise divergence while organizing items very differently: they may retrieve different analogues, form different ambiguity clusters, and disagree about which cases occupy the same local region of the probability simplex. In applications that use uncertainty for retrieval, review prioritization, clustering, data selection, or precedent matching, the relationships among examples may matter as much as each example's marginal score.

We therefore ask a complementary question: do models recover the relational geometry induced by aggregate human NLI judgment distributions? The approach is related in spirit to representational similarity analysis, which compares systems through their induced dissimilarity structures rather than requiring a direct correspondence between internal units (Kriegeskorte et al., 2008). Here, however, the target is not a hidden neural representation. It is a posterior distribution over human judgment graphs derived from repeated annotation counts.

This perspective also clarifies a tension in prior work on calibration and human disagreement. Temperature scaling is a simple and effective method for correcting neural confidence under conventional calibration objectives (Guo et al., 2017), and Wang et al. (2022) argued that calibrated NLI networks can competitively approximate human disagreement distributions. In contrast, Baan et al. (2022) showed that measuring calibration against a human majority label is theoretically problematic when disagreement is inherent. Our experiments suggest that both observations can hold: calibration can improve a chosen pointwise objective while leaving the relational organization of human disagreement nearly unchanged—or changing it substantially in directions that are not more human-aligned.

**Contributions**

- We formulate human disagreement as a posterior fuzzy neighborhood graph and define null-adjusted relational recovery against a cross-fitted human-human reference.
- We show that nine NLI models contain stable, non-random relational signal, but that the signal is consistent with exact human vote-profile structure rather than within-profile item identity.
- We demonstrate a large separation between pointwise and relational effects of temperature scaling: likelihood improves substantially, while relational gap closure remains below 1% across all tested models.
- We show that cross-family probability ensembling recovers substantially more human relational structure than flexible recalibration of a single BART-Large model, with equal weights capturing essentially all of the benefit.

## 2 Related Work

### Human disagreement in NLI

SNLI and MultiNLI established large-scale benchmarks for sentence-level inference (Bowman et al., 2015; Williams et al., 2018), but their standard evaluation protocols emphasize a single target label. Subsequent work has shown that disagreement can reflect genuine semantic uncertainty, context dependence, annotator perspective, or task artifacts. Pavlick and Kwiatkowski (2019) found that model uncertainty did not mirror the structure of human disagreement. ChaosNLI extended this line of work with 100 annotations per item and showed that contemporary systems struggled on low-agreement examples (Nie et al., 2020).

Later studies distinguished systematic from disagreement-inducing inferences (Zhang and de Marneffe, 2021), developed taxonomies of disagreement sources (Jiang and de Marneffe, 2022), and argued that label variation should be modeled rather than removed (Plank, 2022; Weerasooriya et al., 2023). Recent work has also examined whether LLMs reproduce human label variation. Lee et al. (2023) reported limited alignment on highly disputed NLI items, while Kulmizev et al. (2026) found that ensembles of LLM responses can approximate human marginal label distributions but retain idiosyncratic disagreement patterns. These findings motivate an evaluation that compares not only marginal distributions but also relations among items.

### Calibration under disagreement

Temperature scaling learns a single positive scalar applied to logits and is widely used because it often improves confidence calibration without changing the predicted class (Guo et al., 2017). More expressive multiclass maps include vector and matrix scaling and Dirichlet calibration, which applies a linear map to log probabilities followed by softmax (Kull et al., 2019).

The relationship between calibration and human label variation remains contested. Wang et al. (2022) found that calibrated NLI probabilities can capture human disagreement distributions competitively under pointwise measures. Baan et al. (2022), however, argued that calibration against a majority label conflates correctness uncertainty with inherent disagreement and proposed instance-level alternatives. Our work focuses on a separate axis: whether calibration improves the neighborhood structure induced by human opinion distributions.

### Relational evaluation and ensembles

Relational comparisons are common when direct unit-level correspondence is unavailable. Representational similarity analysis summarizes each system through pairwise dissimilarities and compares the resulting structures (Kriegeskorte et al., 2008). Our expected-support graph similarly treats relations among stimuli as the evaluation object, while integrating posterior uncertainty in the human labels and using local, tie-aware neighborhood mass rather than a single deterministic dissimilarity matrix.

Deep ensembles are a strong baseline for predictive uncertainty and robustness, often benefiting from diversity among independently trained models (Lakshminarayanan et al., 2017). Our ensemble experiment asks whether diversity across pretrained architecture families also contains complementary information about human disagreement geometry.

## 3 Relational Evaluation of Human Disagreement

Let p_i be the latent population distribution over entailment, neutral, and contradiction judgments for item i. ChaosNLI provides a finite count vector c_i from 100 annotations. Rather than treating the normalized counts as certain, we place a symmetric Dirichlet prior with concentration 0.5 on each class and draw B = 500 posterior samples p_i^(b).

For each posterior draw and distance metric, we construct a directed k-nearest-neighbor graph. We use fractional tie handling at the kth boundary: all strictly closer neighbors receive weight 1, and the remaining mass needed to total k is distributed uniformly among boundary ties. This produces a row-weighted graph W^(b) whose outgoing mass is exactly k for each item.

The expected human edge-support matrix is S_ij(k) = (1/B) sum_b W_ij^(b)(k). S_ij is interpretable as the posterior expected neighborhood mass assigned from item i to item j. High values identify edges that are repeatedly supported despite uncertainty in the finite human vote counts.

For a model m with predicted distributions q_i^m, we construct the corresponding tie-aware neighborhood graph W^m. Its average human-supported mass is Q_support(m) = (1/(Nk)) sum_ij W_ij^m S_ij. We compare this value with stratified identity-permutation nulls that preserve the SNLI/MNLI composition. We also compute a split-half human-human reference Q_HH by constructing graphs from independent halves of the posterior draws.

To express results on a normalized scale, we define R_m = (Q_support(m) - Q_null(m)) / (Q_HH - Q_null(m)). R_m = 0 corresponds to the stratified null expectation and R_m = 1 corresponds to the cross-fitted human-human reference. For transformations relative to a raw model, relational gap closure is G_Q = (R_transformed - R_raw) / (1 - R_raw). Pointwise NLL gap closure is defined analogously relative to the empirical human entropy floor.

A second null permutes item identity only within groups sharing exactly the same 100-vote count triplet. This exact-profile control tests whether a model recovers item-specific relational identity beyond broad disagreement profiles such as clear entailment, entailment-neutral ambiguity, or polarized contradiction.

## 4 Data, Models, and Reproducibility

We analyze the 3,113 ChaosNLI examples drawn from SNLI (1,514) and MNLI (1,599), each with 100 crowd judgments over three NLI labels (Nie et al., 2020). The underlying source datasets were introduced by Bowman et al. (2015) and Williams et al. (2018).

The model pool contains nine MNLI-fine-tuned checkpoints spanning BART, RoBERTa, XLNet, ALBERT, BERT, and DistilBERT families, including base and large variants where available. These families differ in pretraining objectives and parameterization (Devlin et al., 2019; Liu et al., 2019; Yang et al., 2019; Lan et al., 2020; Lewis et al., 2020; Sanh et al., 2019). We evaluate their normalized three-class probability outputs directly; the experiments do not compare hidden states.

All primary relational analyses use Hellinger distance at k = 10. E001 additionally evaluates Hellinger and Jensen-Shannon distance at k in {10, 20, 50}. High-support core measures use an independently constructed k = 50 human target. Random seeds, object ordering, model-probability hashes, and binary support matrices are bound through repository manifests.

## 5 Experiments

### E001: Does model probability space recover human relational structure?

E001 evaluates each model graph against the expected human edge-support matrix. Statistical significance is assessed with 10,000 dataset-stratified item-identity permutations. Ranking stability is evaluated across twelve combinations of distance metric and neighborhood scale. The exact-profile null tests for residual identity alignment after conditioning on the complete human vote count vector.

The experiment also characterizes the human high-support core at k = 50 and repeats the relational comparison independently within SNLI and MNLI to assess whether the ordering is driven by cross-dataset structure.

### E002: Does temperature scaling repair relational structure?

E002 uses five-fold cross-fitting stratified by dataset, majority label, and empirical entropy quintile. Within each fold, scalar temperatures are fitted on training items for NLL and Jensen-Shannon objectives. A separate topology-selected temperature maximizes a training-only null-adjusted support objective using common permutation seeds across candidates. Each fitted temperature is applied coherently to all N items to construct one graph per fold, but only held-out focal rows contribute to the final score.

The experiment reports changes in NLL, Jensen-Shannon divergence, Q_support, null-adjusted relational recovery, k = 50 core recovery, and minimum-overlap graph turnover. Uncertainty is estimated with 1,000 paired bootstrap samples drawn within the 30 literal dataset-label-entropy strata.

### E003: How much relational repair is available post hoc?

E003 applies a repair ladder to the BART-Large anchor: scalar temperature scaling; class-wise vector scaling with bias; identifiable eight-parameter coarse-grid affine calibration; identifiable eight-parameter coarse-grid Dirichlet calibration; an equal-weight ensemble of BART-Large, RoBERTa-Large, and XLNet-Large; an NLL-selected convex ensemble; and a topology-selected convex ensemble.

The ensemble levels use fold-specific weights fitted strictly on training items. For Level 6a, a dedicated training-only human posterior support matrix S_{train, f} is constructed for each fold from 200 Dirichlet posterior draws over N_{train} training items. Blended ensemble probabilities are actively validated to ensure finite non-negativity and exact sum-to-one normalization ($1.0 \pm 1e-6$). Fold-specific weights, training objectives, second-best candidates, and objective margins are persisted. Direct paired bootstrap contrasts (1,000 resamples) and 10,000 exact-profile permutation nulls are evaluated across all ensemble conditions.

## 6 Results

### E001: Stable but incomplete human relational recovery

All nine models select neighborhood mass with more human posterior support than the dataset-stratified null. At k = 10 under Hellinger distance, BART-Large obtains Q_support = 0.01681 compared with Q_null approximately 0.00329, a 5.11-fold ratio. Human-normalized recovery ranges from 19.59% for BART-Large to 6.69% for BERT-Base.

The model ordering is exactly invariant across the twelve tested metric-scale configurations (Kendall's W = 1.0): BART-Large, RoBERTa-Large, and XLNet-Large form the leading tier, followed by ALBERT-xxLarge, BERT-Large, RoBERTa-Base, XLNet-Base, DistilBERT, and BERT-Base. Larger variants consistently exceed smaller variants within RoBERTa, XLNet, and BERT families. The ordering also replicates when SNLI and MNLI are analyzed independently.

The exact-profile control substantially narrows the interpretation. No model shows robust residual alignment after item identities are permuted within identical 100-vote profiles. Thus the models appear to recover a stable coarse organization of human uncertainty, but there is limited evidence that they recover which particular items humans place together once their vote distributions are held fixed.

### E002: Calibration reorganizes model space without repairing it

NLL-selected temperature scaling improves soft-label negative log-likelihood for all nine models, closing 24.86%-56.57% of the reducible NLL gap. The largest proportional gains occur for weaker raw models, which require higher temperatures and begin farther from the human entropy floor.

Relational improvement is two orders of magnitude smaller. G_Q ranges from 0.18% to 0.69%, and the paired confidence interval for the difference G_NLL - G_Q excludes zero for every model. The central result is therefore not that temperature scaling leaves topology invariant. Minimum-overlap turnover ranges from roughly 13% to more than 30%, showing that calibration can replace substantial neighborhood mass. Those changes simply do not move the graph appreciably toward the human reference.

A second unexpected result is objective disagreement. NLL-selected temperatures soften probabilities and improve forward cross-entropy, but Jensen-Shannon divergence to the human distribution increases across the model pool. Meanwhile, temperatures selected directly for the relational objective yield only small Q gains and can substantially worsen pointwise metrics and high-support core recall.

### E003: Cross-model diversity provides substantially more repair, but weight optimization adds no detectable gain

More flexible recalibration of BART-Large does not close the relational gap. Across scalar, vector, coarse-grid affine, and coarse-grid Dirichlet maps, the largest point estimate is G_Q = 0.83%, even though NLL gap closure ranges from approximately 10% to 27%.

The three-family ensembles produce a qualitatively different result. Equal weighting reduces NLL from 0.8627 to 0.7171 and raises human-normalized relational recovery from 19.59% to 33.41%, corresponding to G_Q = 17.18%. The NLL-selected ensemble obtains NLL = 0.7162 and G_Q = 17.33%. The topology-selected ensemble obtains G_Q = 17.71%.

Direct paired bootstrap contrasts demonstrate that no ensemble weighting strategy significantly or meaningfully outperforms the others:
- Level 6a Topology vs Level 5b NLL: \Delta G_Q = -0.07% (95% CI: [-0.69%, +0.59%], P(\Delta G_Q > 0) = 40.5%)
- Level 6a Topology vs Level 5a Equal: \Delta G_Q = +0.00% (95% CI: [-0.66%, +0.70%], P(\Delta G_Q > 0) = 50.7%)
- Level 5b NLL vs Level 5a Equal: \Delta G_Q = +0.07% (95% CI: [-0.19%, +0.37%], P(\Delta G_Q > 0) = 68.7%)

All direct G_Q contrasts are smaller than one percentage point and their 95% confidence intervals include zero. Furthermore, fold-specific training objectives show extremely flat optimization margins (\le 0.00003), confirming that global weight optimization adds no detectable held-out relational advantage over simple equal weighting.

Exact-profile controls yield negligible excess support for every ensemble condition: Q_{profile-excess} \approx -0.00003 for Equal, -0.00002 for NLL, and -0.00001 for Topology (all p \ge 0.73).

## 7 Discussion

Taken together, the experiments separate three concepts that are often treated as interchangeable: predictive confidence, pointwise distributional agreement, and relational alignment. A model can improve substantially on one while making negligible progress on another. This is most visible in E002, where temperature scaling changes many neighborhoods and improves NLL, yet closes less than 1% of the relational gap.

The E001 and E003 exact-profile results establish an important conceptual boundary: the measured model-human alignment is an alignment with aggregate judgment-distribution geometry rather than demonstrable item-specific semantic organization beyond exact vote profiles. Because the human target graph is itself constructed from aggregate three-class vote distributions, items with identical count profiles are exchangeable under that target. The graph does not contain independent annotator-level or explanatory features that could distinguish two items once their vote distributions are held fixed. Thus, the exact-profile result does not imply that models fail to represent semantic structure, but rather that the measured alignment reflects aggregate vote-distribution geometry.

The ensemble result indicates that combining probabilities across distinct model families (BART, RoBERTa, and XLNet) recovers substantially more human relational structure than recalibrating any single model. However, because equal weighting captures essentially all of the benefit and direct paired contrasts show no significant difference among weighting schemes, the gain stems from model combination itself—such as variance reduction, probability smoothing, or cross-family error cancellation—rather than a finely tuned global optimization objective. The results are consistent with complementary information across model outputs, although the experiment does not isolate the exact mechanism of the ensemble gain.

These findings have practical implications for systems that use NLI probabilities as more than final class scores. Calibration may be useful for thresholding and abstention, but it does not guarantee better example retrieval, ambiguity clustering, review diversification, or precedent selection. Relational evaluation could therefore complement accuracy and calibration when models are used to organize cases or identify analogous failures.

The work also suggests a path toward topology-aware representation learning. A future objective could combine soft-label NLL with a differentiable relational loss based on pairwise distances, high-support edge contrast, graph Laplacians, or soft neighborhood overlap. However, the present experiments motivate such methods rather than proving that representation fine-tuning is uniquely necessary. Nonlinear stacking, item-dependent gating, annotator-aware models, and richer model pools remain plausible alternatives.

## 8 Limitations and Remaining Analyses

- The principal experiments use older MNLI-fine-tuned transformer checkpoints. Their continued practical relevance makes them useful controlled objects, but replication with DeBERTa-style models, instruction-tuned LLMs, and modern embedding or verifier systems is necessary.
- The primary target represents collective annotation frequencies, not identifiable individual annotators or demographic subgroups. Population-level structure can conceal stable subpopulation differences.
- Exact-profile conditioning tests whether alignment extends beyond identical count vectors, but it does not identify the linguistic cause of disagreement. Qualitative coding or explanation annotations are needed to distinguish lexical ambiguity, pragmatics, world knowledge, reference uncertainty, and task artifacts.
- The E003 affine and Dirichlet calibrators use coarse finite grids rather than continuous regularized optimization. They support a conclusion about the tested BART post-hoc family, not a universal impossibility theorem for all probability maps.
- Sensitivity of Level 6 selection to target posterior draws: Level 6 fold targets were constructed using 200 posterior draws per fold (versus 500 draws in E001) for computational search efficiency. Because weight optimization yielded flat training margins and no held-out advantage over equal weighting, held-out conclusions are robust to draw count.

## 9 Ethical Considerations

Human label variation should not automatically be interpreted as error or irrationality. It can reflect ambiguity, different background assumptions, annotator perspective, or limitations in task instructions. Relational metrics should therefore be used to preserve and analyze variation, not to declare one human cluster correct.

The study uses an existing public benchmark and aggregated annotation counts. It does not infer sensitive attributes or attempt to identify annotators. Future subgroup analyses would require careful attention to consent, representativeness, and the risk of reifying demographic categories.

Improved alignment with a collective human graph is not equivalent to normative correctness. In high-stakes applications, relational agreement must be considered alongside domain expertise, fairness, contestability, and the possibility that majority structure marginalizes minority perspectives.

## 10 Conclusion

Pointwise distribution matching does not fully characterize whether a model represents human disagreement. Across ChaosNLI, NLI models show stable non-random relational alignment, but much of it is attributable to broad vote-profile structure. Temperature scaling can substantially improve likelihood and substantially reorganize model neighborhoods while producing almost no relational repair. In contrast, combining predictions from diverse model families recovers a meaningful additional portion of the human graph, although most human relational structure remains missing. Equal weighting accounts for essentially all of this gain, and global weight optimization provides no detectable additional benefit. These results motivate evaluation frameworks and learning objectives that treat the geometry of human variation as an explicit target rather than assuming that calibrated marginal probabilities are sufficient.

## References

- Baan, Joris, Wilker Aziz, Barbara Plank, and Raquel Fernandez. 2022. Stop Measuring Calibration When Humans Disagree. In Proceedings of EMNLP 2022.
- Bowman, Samuel R., Gabor Angeli, Christopher Potts, and Christopher D. Manning. 2015. A Large Annotated Corpus for Learning Natural Language Inference. In Proceedings of EMNLP 2015, 632-642.
- Devlin, Jacob, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. In Proceedings of NAACL-HLT 2019, 4171-4186.
- Guo, Chuan, Geoff Pleiss, Yu Sun, and Kilian Q. Weinberger. 2017. On Calibration of Modern Neural Networks. In Proceedings of ICML 2017, 1321-1330.
- Jiang, Nan-Jiang, and Marie-Catherine de Marneffe. 2022. Investigating Reasons for Disagreement in Natural Language Inference. Transactions of the Association for Computational Linguistics 10.
- Kriegeskorte, Nikolaus, Marieke Mur, and Peter A. Bandettini. 2008. Representational Similarity Analysis—Connecting the Branches of Systems Neuroscience. Frontiers in Systems Neuroscience 2:4.
- Kull, Meelis, Miquel Perello-Nieto, Markus Kängsepp, Telmo Silva Filho, Hao Song, and Peter Flach. 2019. Beyond Temperature Scaling: Obtaining Well-Calibrated Multiclass Probabilities with Dirichlet Calibration. In Advances in Neural Information Processing Systems 32.
- Kulmizev, Artur, Erika Lombart, Patrick Watrin, and Marie-Catherine de Marneffe. 2026. Label and Explanation Variation in LLM-Based Annotation: A Case Study in Natural Language Inference. In Proceedings of ACL 2026, 16526-16543.
- Lakshminarayanan, Balaji, Alexander Pritzel, and Charles Blundell. 2017. Simple and Scalable Predictive Uncertainty Estimation Using Deep Ensembles. In Advances in Neural Information Processing Systems 30.
- Lan, Zhenzhong, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, and Radu Soricut. 2020. ALBERT: A Lite BERT for Self-Supervised Learning of Language Representations. In Proceedings of ICLR 2020.
- Lee, Noah, Na Min An, and James Thorne. 2023. Can Large Language Models Capture Dissenting Human Voices? In Proceedings of EMNLP 2023.
- Lewis, Mike, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Veselin Stoyanov, and Luke Zettlemoyer. 2020. BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension. In Proceedings of ACL 2020, 7871-7880.
- Liu, Yinhan, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. 2019. RoBERTa: A Robustly Optimized BERT Pretraining Approach. arXiv:1907.11692.
- Nie, Yixin, Xiang Zhou, and Mohit Bansal. 2020. What Can We Learn from Collective Human Opinions on Natural Language Inference Data? In Proceedings of EMNLP 2020, 9131-9143.
- Pavlick, Ellie, and Tom Kwiatkowski. 2019. Inherent Disagreements in Human Textual Inferences. Transactions of the Association for Computational Linguistics 7:677-694.
- Plank, Barbara. 2022. The 'Problem' of Human Label Variation: On Ground Truth in Data, Modeling and Evaluation. In Proceedings of EMNLP 2022, 10671-10682.
- Sanh, Victor, Lysandre Debut, Julien Chaumond, and Thomas Wolf. 2019. DistilBERT, a Distilled Version of BERT: Smaller, Faster, Cheaper and Lighter. arXiv:1910.01108.
- Wang, Yuxia, Minghan Wang, Yimeng Chen, Shimin Tao, Jiaxin Guo, Chang Su, Min Zhang, and Hao Yang. 2022. Capture Human Disagreement Distributions by Calibrated Networks for Natural Language Inference. In Findings of ACL 2022, 1524-1535.
- Weerasooriya, Tharindu Cyril, Alexander Ororbia, Raj Bhensadadia, Ashiqur KhudaBukhsh, and Christopher M. Homan. 2023. Disagreement Matters: Preserving Label Diversity by Jointly Modeling Item and Annotator Label Distributions with DisCo. In Findings of ACL 2023, 4679-4695.
- Williams, Adina, Nikita Nangia, and Samuel R. Bowman. 2018. A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference. In Proceedings of NAACL-HLT 2018, 1112-1122.
- Yang, Zhilin, Zihang Dai, Yiming Yang, Jaime Carbonell, Ruslan Salakhutdinov, and Quoc V. Le. 2019. XLNet: Generalized Autoregressive Pretraining for Language Understanding. In Advances in Neural Information Processing Systems 32.
- Zhang, Xinliang Frederick, and Marie-Catherine de Marneffe. 2021. Identifying Inherent Disagreement in Natural Language Inference. In Proceedings of NAACL-HLT 2021, 4908-4915.