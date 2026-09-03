# Neurocognitive Outcome Measurement in Oncology Trials — Classifier Development

Does an oncology trial actually measure cognition? This repository holds the
hand-labelled dataset, the model development, and the comparison that answers
that question at registry scale.

The finding that started it: **1.6%** of glioblastoma immunotherapy trials
register a neurocognitive outcome, against **20.6%** for quality of life. This
work asks whether that gap is specific to glioblastoma or general to oncology —
and builds the tool needed to measure it across tens of thousands of trials.

The trained CNS model is deployed and live:
**[cognitive-outcome-classifier-api](https://github.com/Mubashir-zz/cognitive-outcome-classifier-api)** ·
[live endpoint](https://cognitive-outcome-classifier-api.onrender.com/about)

## The labelling problem

1,888 trials across CNS, breast, lung and head & neck were pulled from
ClinicalTrials.gov and international registries (ChiCTR, EU-CTR, JPRN, CTRI,
ANZCTR and others). Every one was read and classified by hand rather than
keyword-matched, because the distinctions that matter are not lexical:

| Counts as cognition | Does not |
|---|---|
| MMSE, MoCA, HVLT-R, TMT, COWAT, FACT-Cog | NANO neurologic exam |
| Named cognitive domain with an instrument | Karnofsky / ECOG performance status |
| | A QoL questionnaire's cognitive *subscale* |

That last row is where automated approaches fail. "Cognitive functioning" as
one of fifteen EORTC QLQ-C30 subscales is not a neurocognitive endpoint, and no
amount of keyword matching separates it from one that is.

## What was tested

**A TF-IDF + LASSO logistic regression baseline** (R, glmnet), with the keyword
flag included as a feature.

**A fine-tuned Bio_ClinicalBERT model** (Python, PyTorch), trained on raw
outcome text with no keyword feature at all — a strictly harder version of the
task.

Both were measured against the same yardstick: **lift over a naive
keyword-presence rule**, per cancer type. Raw accuracy is not a fair comparison
when one model gets the keyword flag and the other does not.

## Results

The TF-IDF model's headline numbers look good — AUROC 0.971, F1 0.935. They are
also, in the way that counts, meaningless:

| Cancer type | n | Model accuracy | Keyword-only baseline | Lift |
|---|---|---|---|---|
| Breast | 152 | 99.3% | 99.3% | **0.000** |
| CNS | 85 | 71.8% | 71.8% | **0.000** |
| Head & Neck | 40 | 100.0% | 100.0% | **0.000** |
| Lung | 77 | 100.0% | 100.0% | **0.000** |

Zero lift across all four types. The model learned to read the keyword flag and
little else. Its top-weighted terms — `metformin`, `medulloblastoma`, `proton`,
`idh`, `rano` — are CNS clinical vocabulary with no relationship to cognition,
which is what confounding with sample composition looks like.

Bag-of-words cannot do negation-sensitive, contextual reasoning, and that is
exactly what separating "NANO scale, not cognitive" from "MMSE administered"
requires.

**Bio_ClinicalBERT changed the picture only for CNS.** Sensitivity there rose
from 52% to 75–77%, replicated across two independent configurations. For
breast, lung and head & neck it did not beat the keyword rule — those types
have standardised instrument vocabulary and the simple rule is already at the
ceiling.

So the deployed system is a hybrid, and the routing is an empirical result
rather than a design preference: **keyword rule for breast/lung/head & neck,
fine-tuned BERT for CNS.**

Full metrics, every model iteration, and the limitations —
including a disclosed test-set-reuse caveat — are in
[`results/PHASE2_RESULTS.md`](results/PHASE2_RESULTS.md).

## Reproducing

**R baseline** — every package is declared in the script and installed on first
run:

```bash
Rscript analysis/phase2_train_classifier.R
```

Writes `results/`, `figures/` and `models/` from `data/CLASSIFIER_training_population.csv`.
`set.seed(2026)` makes the split deterministic; a clean run reproduces the
confusion matrix and per-type table above exactly.

**BERT fine-tuning** — `analysis/Phase2b_BERT_finetune.ipynb`, built for Google
Colab on a T4. Point it at the same training file.

## Layout

```
data/
  MASTER_gold_standard_1888.csv         1,888 hand-labelled trials, all 4 types
  CLASSIFIER_training_population.csv    1,804-trial training subset
analysis/
  phase2_train_classifier.R             TF-IDF + LASSO baseline
  Phase2b_BERT_finetune.ipynb           Bio_ClinicalBERT fine-tuning
results/
  PHASE2_RESULTS.md                     full write-up, all runs, limitations
  phase2_performance_overall.txt        held-out test metrics
  phase2_performance_by_cancer_type.csv per-type metrics with baseline lift
  phase2_top_predictive_terms.csv       LASSO coefficients
figures/
  phase2_roc_curve.png
  phase2_performance_by_type.png
models/
  cognitive_classifier_v1.rds           fitted glmnet model + vocabulary
```

The fine-tuned Bio_ClinicalBERT weights (~433MB) are not in the repository —
they are hosted on Hugging Face Hub and pulled at serving time by the API. Two
Phase 2 checkpoints exist; **Run 2** (class-weighted, 4 epochs) is the one
referenced in the results and deployed. Run 1 is a development record only.

## Limitations

- The same 357-trial test set was evaluated three times across model iterations.
  That is a soft form of test-set overfitting and the threshold-tuned numbers
  should be read with that in mind.
- R and Python splits are not identical — matched stratification logic, but
  different RNG implementations (R: 1430/354, Colab: 1427/357).
- Head & neck has 40 test trials and 17% positives in training. A handful of
  flipped predictions moves those percentages a lot.
- BERT truncates at 512 tokens, which cuts later outcomes from trials with very
  long multi-outcome lists — mostly paediatric CNS.

## Author

Mubashir Ahmad Khan
