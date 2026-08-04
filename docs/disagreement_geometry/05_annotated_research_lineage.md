# Annotated Research Lineage

This is a curated map, not an exhaustive systematic review. Each entry includes:

- **What it established**
- **What to borrow**
- **What not to assume**

## 1. Reliability and measurement

### Cronbach, Gleser, Nanda, and Rajaratnam (1972)  
*The Dependability of Behavioral Measurements: Theory of Generalizability for Scores and Profiles.*

**Established:** Reliability can be decomposed across multiple facets rather than reduced to one agreement coefficient.

**Borrow:** Treat items, votes, annotators, folds, and posterior draws as distinct uncertainty sources.

**Experiment:** Variance decomposition for relational support and prototype complexity.

**Do not assume:** High reliability implies correct or normatively valid judgments.

### Brennan (2001)  
*Generalizability Theory.*

**Established:** A mature framework for estimating dependability under different measurement designs.

**Borrow:** Decision studies for choosing annotation counts and sampling designs.

### Dawid and Skene (1979)  
“Maximum Likelihood Estimation of Observer Error-Rates Using the EM Algorithm.” *Applied Statistics*, 28(1), 20–28. DOI: 10.2307/2346806.

**Established:** Latent truth and annotator confusion matrices can be jointly estimated without observed gold labels.

**Borrow:** Annotator-specific error structure when identities exist.

**Do not assume:** Every disagreement is error around one latent truth.

## 2. Cultural consensus and shared knowledge

### Romney, Weller, and Batchelder (1986)  
“Culture as Consensus: A Theory of Culture and Informant Accuracy.” *American Anthropologist*, 88(2), 313–338. DOI: 10.1525/aa.1986.88.2.02a00020.

**Established:** Agreement patterns can reveal a shared answer key and individual competence.

**Borrow:** Eigenstructure of respondent agreement; sample-size reasoning; latent shared culture.

**Experiment:** Compare one-consensus versus multi-consensus mixture models.

**Do not assume:** One culture or one correct shared answer exists.

## 3. Structured subjectivity and viewpoint discovery

### Brown (1980)  
*Political Subjectivity: Applications of Q Methodology in Political Science.*

**Established:** Subjectivity has discoverable structure.

### Brown (1993)  
“A Primer on Q Methodology.” *Operant Subjectivity*, 16(3/4).

**Established:** Q sorting, person-by-person correlation, factor extraction, and factor interpretation can identify coherent viewpoints.

**Borrow:** Treat people as the objects to cluster based on their patterns across statements.

**Experiment:** On repeated-identity data, compare Q factors with graph-derived respondent clusters.

**Do not assume:** Factor labels explain causal reasons without qualitative interpretation.

## 4. Psychometrics and ideal points

### Poole and Rosenthal (1997)  
*Congress: A Political-Economic History of Roll Call Voting.*

### Poole, Lewis, Lo, and Carroll (2011)  
“Scaling Roll Call Votes with W-NOMINATE in R.” *Journal of Statistical Software*, 42(14). DOI: 10.18637/jss.v042.i14.

**Established:** Voting behavior can often be represented in a low-dimensional spatial model.

**Borrow:** Joint respondent/item coordinates; issue-dependent discrimination; model comparison by latent dimension.

**Experiment:** Embed annotators and items jointly when repeated judgments are available.

**Do not assume:** Every domain is adequately one-dimensional.

## 5. Social choice and judgment aggregation

### Arrow (1951/1963)  
*Social Choice and Individual Values.*

**Established:** No rank-order aggregation rule satisfies all desirable conditions over unrestricted preferences.

### List and Pettit (2002)  
“Aggregating Sets of Judgments: An Impossibility Result.” *Economics & Philosophy*, 18(1), 89–110. DOI: 10.1017/S0266267102001098.

**Established:** Aggregating logically connected judgments can create collective inconsistency under mild requirements.

**Borrow:** Treat majority-label compression as a structural transformation with possible impossibility and coherence costs.

**Experiment:** Measure relational and logical structures lost under majority aggregation.

**Do not assume:** A better aggregation rule removes every trade-off.

## 6. Polarization measurement

### Esteban and Ray (1994)  
“On the Measurement of Polarization.” *Econometrica*, 62(4), 819–851. DOI: 10.2307/2951734.

**Established:** Polarization is distinct from inequality or variance and combines group identification with inter-group alienation.

**Borrow:** Separate within-cluster cohesion from between-cluster distance.

**Experiment:** Group-aware polarization in respondent or prototype space.

**Do not assume:** High entropy is polarization.

## 7. Opinion dynamics

### DeGroot (1974)  
“Reaching a Consensus.” *Journal of the American Statistical Association*, 69(345), 118–121.

**Established:** Repeated weighted averaging can produce consensus under graph conditions.

### Hegselmann and Krause (2002)  
“Opinion Dynamics and Bounded Confidence: Models, Analysis and Simulation.” *Journal of Artificial Societies and Social Simulation*, 5(3).

**Established:** Local interaction within confidence bounds can produce consensus, polarization, or fragmentation.

**Borrow:** Confidence thresholds, cluster transitions, and multi-scale dynamics.

### Lorenz (2007)  
“Continuous Opinion Dynamics under Bounded Confidence: A Survey.” Preprint arXiv:0707.1762.

**Established:** A broad mathematical taxonomy of bounded-confidence systems, including multidimensional opinions.

### Dandekar, Goel, and Lee (2013)  
“Biased Assimilation, Homophily, and the Dynamics of Polarization.” *PNAS*, 110(15), 5791–5796.

**Established:** Homophily alone under DeGroot averaging does not necessarily polarize; biased assimilation changes the result.

**Borrow:** Distinguish network segregation from belief-update mechanisms.

**Do not assume:** Static judgment geometry reveals causal dynamics.

## 8. Compositional data and probability geometry

### Aitchison (1982)  
“The Statistical Analysis of Compositional Data.” *Journal of the Royal Statistical Society, Series B*, 44(2), 139–160. DOI: 10.1111/j.2517-6161.1982.tb01195.x.

**Established:** Proportions live in a simplex and require geometry respecting the fixed-sum constraint.

**Borrow:** Ratio-aware geometry, log-ratio coordinates, and explicit sample-space thinking.

**Experiment:** Compare Hellinger and Aitchison graph stability.

**Do not assume:** Aitchison distance is automatically best for label probabilities, especially with zeros.

### Amari and Nagaoka (2000)  
*Methods of Information Geometry.*

**Established:** Probability distributions form manifolds with Fisher information geometry and dual connections.

**Borrow:** Geodesics, projections, and principled probability-space transformations.

### Pistone (2019)  
*Information Geometry of the Probability Simplex: A Short Course.* arXiv:1911.01876.

**Borrow:** Accessible bridge between full-simplex geometry and statistical physics.

## 9. Optimal transport and structural comparison

### Panaretos and Zemel (2019)  
“Statistical Aspects of Wasserstein Distances.” *Annual Review of Statistics and Its Application*, 6, 405–431.

**Established:** Wasserstein metrics compare distributions through the effort required to transport mass.

**Borrow:** Distribution perturbation, barycenters, and geometry-aware comparison.

### Xu, Luo, Zha, and Carin (2019)  
“Gromov-Wasserstein Learning for Graph Matching and Node Embedding.” *ICML 2019*, PMLR 97.

**Established:** GW compares graph structure and can infer correspondence across spaces.

**Borrow:** Compare collective geometries without fixed item alignment.

**Do not assume:** A low structural distance implies semantic equivalence.

### Vayer, Courty, Tavenard, Chapel, and Flamary (2019)  
“Optimal Transport for Structured Data with Application on Graphs.” *ICML 2019*, PMLR 97.

**Established:** Fused GW combines observed features and internal structure.

**Borrow:** Align groups or datasets using both text/item features and judgment geometry.

## 10. Crowd disagreement as signal

### Aroyo and Welty (2015)  
“Truth Is a Lie: Crowd Truth and the Seven Myths of Human Annotation.” *AI Magazine*, 36(1), 15–24. DOI: 10.1609/aimag.v36i1.2564.

**Established:** Semantic disagreement can represent legitimate ambiguity rather than bad annotation.

**Borrow:** Measure ambiguity across input, worker, and annotation dimensions.

**Do not assume:** All disagreement is equally valid or informative.

### Dumitrache, Inel, Aroyo, Timmermans, and Welty (2018)  
“CrowdTruth 2.0: Quality Metrics for Crowdsourcing with Disagreement.” arXiv:1808.06080.

**Borrow:** Joint quality metrics and worker–item interdependence.

## 11. Perspectivist machine learning

### Basile, Cabitza, Campagner, and Fell (2021/2023)  
“Toward a Perspectivist Turn in Ground Truthing for Predictive Computing.” *AAAI 2023*. DOI: 10.1609/aaai.v37i6.25840.

**Established:** Majority-vote ground truth can erase meaningful perspectives; data pipelines should preserve disagreement.

**Borrow:** Treat aggregation as a documented design choice.

### Fleisig, Blodgett, Klein, and Talat (2024)  
“The Perspectivist Paradigm Shift: Assumptions and Challenges of Capturing Human Labels.” arXiv:2405.05860.

**Established:** Perspectivism introduces practical and normative challenges, not just technical benefits.

**Borrow:** Explicitly document whose perspectives, sampling, causes of disagreement, and downstream use.

## 12. Human disagreement in NLI

### Nie, Zhou, and Bansal (2020)  
“What Can We Learn from Collective Human Opinions on Natural Language Inference Data?” *EMNLP 2020*. DOI: 10.18653/v1/2020.emnlp-main.734.

**Established:** ChaosNLI provides 100 judgments per example and shows that models struggle badly on high-disagreement items and fail to recover collective label distributions.

**Borrow:** Dense human vote distributions and agreement-stratified evaluation.

### Lee, An, and Thorne (2023)  
“Can Large Language Models Capture Dissenting Human Voices?” *EMNLP 2023*. DOI: 10.18653/v1/2023.emnlp-main.278.

**Established:** LLM distributions estimated with log probabilities or sampling show limited alignment with human disagreement, especially on high-disagreement items.

**Borrow:** LPE/MCE comparison and modern-model bridge.

**Current-project extension:** Move from pointwise distribution fit to relational geometry and effective resolution.

## 13. Demographic and pluralistic model evaluation

### Santurkar et al. (2023)  
“Whose Opinions Do Language Models Reflect?” *ICML 2023*, PMLR 202.

**Established:** OpinionQA evaluates model alignment with 60 US demographic groups and finds substantial group misalignment.

**Borrow:** Group-level distribution comparison and steering evaluation.

### Kirk et al. (2024)  
“The PRISM Alignment Dataset.” *NeurIPS 2024*.

**Established:** PRISM links 1,500 participants from 75 countries to fine-grained feedback across 8,011 live conversations with 21 LLMs.

**Borrow:** Individual, demographic, contextual, and multicultural geometry.

### Jakobsen, Cabello, and Søgaard (2023)  
“Being Right for Whose Right Reasons?” *ACL 2023*. DOI: 10.18653/v1/2023.acl-long.59.

**Established:** Human rationales vary across demographic groups, and model rationale alignment is not group-neutral.

**Borrow:** Compare label geometry with rationale geometry.

## 14. Pluralistic alignment methods

### Feng et al. (2024)  
“Modular Pluralism: Pluralistic Alignment via Multi-LLM Collaboration.” *EMNLP 2024*. DOI: 10.18653/v1/2024.emnlp-main.240.

**Established:** Specialized community models can be composed to support Overton, steerable, and distributional pluralism.

**Borrow:** Modular model pools and community-specific patching.

**Current-project extension:** Select modules by measured relational contribution and required effective resolution.

### Poole-Dayan et al. (2025)  
“Benchmarking Overton Pluralism in LLMs.” NeurIPS 2025 evaluation workshop / arXiv:2512.01351.

**Established:** Overton pluralism can be framed as viewpoint coverage and approximated through scalable evaluation.

**Borrow:** Separate range coverage from population-distribution matching.

### Nie et al. (2026)  
“PERSPECTRA: A Scalable and Configurable Pluralist Benchmark of Perspectives from Arguments.” arXiv:2602.08716.

**Established:** Structured debate graphs and naturalistic arguments enable viewpoint counting, matching, and polarity evaluation.

**Borrow:** Argument-graph geometry and configurable perspective sets.

## 15. Priority reading sequence

Read these first:

1. Romney, Weller, and Batchelder (1986)
2. Brown (1993)
3. List and Pettit (2002)
4. Hegselmann and Krause (2002)
5. Aitchison (1982)
6. Esteban and Ray (1994)
7. Aroyo and Welty (2015)
8. Nie, Zhou, and Bansal (2020)
9. Basile et al. (2023)
10. Santurkar et al. (2023)
11. Kirk et al. (2024)
12. Feng et al. (2024)

Then add:

13. Xu et al. (2019) for cross-space geometry
14. Lee et al. (2023) for LLM disagreement distributions
15. Fleisig et al. (2024) for normative and data-pipeline cautions

## 16. Most important synthesis

Older fields already established that:

- agreement patterns contain latent structure;
- viewpoints can be mapped;
- aggregation can destroy coherence;
- polarization is not variance;
- interaction rules change collective shape;
- probability vectors require a proper geometry.

The current opportunity is to combine those lessons into an AI evaluation and optimization framework:

> Measure which resolution and regions of collective judgment an AI system preserves, then design calibration, ensembles, routing, and distillation around that structure.
