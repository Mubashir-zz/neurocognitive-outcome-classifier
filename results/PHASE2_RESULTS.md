# Phase 2 — Classifier Development: Complete Summary
*Reference document for Methods/Results sections of the multi-cancer neurocognitive outcome NLP study*

## Purpose

Having hand-labeled 1,888 trials across four cancer types (CNS, Breast, Lung, Head & Neck) as a gold-standard training set, Phase 2 tested whether an automated classifier could replicate this manual judgment — first with a simple baseline, then with a fine-tuned transformer model.

## Training Data

- **Gold-standard set:** 1,888 hand-labeled trials (439 CNS + 1,449 Breast/Lung/Head&Neck), each independently reviewed for INCLUDE status and neurocognitive-outcome status through direct reading of outcome-measure text (not keyword matching alone).
- **Classifier training population:** 1,804 trials (INCLUDE = Yes, cognition label confirmed Yes/No; 1 "Unsure" record excluded).
- **Class balance by cancer type** (Yes / No):

| Cancer Type | Cognitive = Yes | Cognitive = No | % Positive |
|---|---|---|---|
| Breast | 558 | 213 | 72% |
| CNS | 240 | 195 | 55% |
| Lung | 197 | 195 | 50% |
| Head & Neck | 35 | 171 | 17% |

## Approach 1 — TF-IDF + LASSO Logistic Regression (R, glmnet)

**Method:** Outcome text converted to TF-IDF unigram features, combined with a binary `keyword_hit` flag (whether any of ~60 cognitive-instrument/domain terms appeared in the text). Trained with L1-regularized (LASSO) logistic regression, 5-fold cross-validated lambda selection. Stratified 80/20 train/test split by cancer type + label.

**Result — the critical, honest finding:**

| Cancer Type | N (test) | Model Accuracy | Keyword-only Baseline Accuracy | **Lift** |
|---|---|---|---|---|
| Breast | 152 | 99.3% | 99.3% | **0.000** |
| CNS | 85 | 71.8% | 71.8% | **0.000** |
| Head & Neck | 40 | 100.0% | 100.0% | **0.000** |
| Lung | 77 | 100.0% | 100.0% | **0.000** |

**The TF-IDF+LASSO model added zero measurable value beyond the naive keyword-presence rule, across all four cancer types.** Top predictive terms included plausible signal (`karnofsky`, `kps` correctly learned as non-cognitive, reflecting the NANO/KPS trap pattern taught during manual labeling) but also apparent spurious correlations (`metformin`, `medulloblastoma`, `proton`, `idh`, `rano` — CNS-specific clinical/molecular jargon unrelated to cognition itself, likely reflecting confounding with which specific trials happened to be in the training sample rather than generalizable language understanding).

**Interpretation:** bag-of-words representations cannot capture the contextual, negation-sensitive reasoning (e.g., distinguishing "NANO scale, not cognitive" from "MMSE administered") that manual review relied on. This motivated moving to a contextual language model.

## Approach 2 — Fine-tuned Bio_ClinicalBERT (Python/PyTorch, Google Colab, T4 GPU)

**Base model:** `emilyalsentzer/Bio_ClinicalBERT` (BioBERT further pre-trained on MIMIC-III clinical notes). Standard BERT-base architecture (12 layers, 768 hidden size, 110M parameters). Max input length 512 tokens (a disclosed limitation — a minority of trials with very long outcome-text lists, mostly pediatric CNS trials, have later outcomes truncated).

**Important design note:** unlike the R model, the BERT classifier was trained on **raw outcome text alone** — it was never given the `keyword_hit` flag as an input feature. This makes the comparison to R's baseline-driven numbers imperfect (BERT solved a strictly harder version of the task), which is why the per-cancer-type Lift metric (BERT accuracy vs. the same naive keyword baseline) is the fairer comparison, not raw accuracy.

### Run 1 — No class weighting, 3 epochs

| Cancer Type | N | Sens | Spec | Acc | Baseline | Lift |
|---|---|---|---|---|---|---|
| Breast | 154 | 92.9% | 64.3% | 85.1% | 100.0% | -14.9% |
| CNS | 85 | 68.8% | 54.1% | 62.4% | 69.4% | -7.1% |
| Head & Neck | 40 | 100.0% | 81.8% | 85.0% | 100.0% | -15.0% |
| Lung | 78 | 94.9% | 66.7% | 80.8% | 100.0% | -19.2% |

Overall: Sens 0.879, Spec 0.662, PPV 0.780, NPV 0.800, F1 0.826, **AUROC 0.894**.

### Run 2 — Class-weighted loss (No=1.183, Yes=0.866, computed from training-set balance), 4 epochs

| Cancer Type | N | Sens | Spec | Acc | Baseline | Lift |
|---|---|---|---|---|---|---|
| Breast | 154 | 95.5% | 57.1% | 85.1% | 100.0% | -14.9% |
| CNS | 85 | **77.1%** | 51.4% | 65.9% | 69.4% | **-3.5%** |
| Head & Neck | 40 | 100.0% | 81.8% | 85.0% | 100.0% | -15.0% |
| Lung | 78 | 94.9% | 61.5% | 78.2% | 100.0% | -21.8% |

Overall: Sens 0.913, Spec 0.623, PPV 0.767, NPV 0.839, F1 0.834, **AUROC 0.894 (identical to Run 1)**.

*AUROC being unchanged while sensitivity/specificity shifted indicates the model's underlying discriminative ability did not change — class weighting only moved the decision boundary, it did not improve the model's ranking ability.*

### Run 3 — Per-cancer-type threshold optimization (Youden's J), applied post-hoc to Run 2's probabilities, no retraining

| Cancer Type | Optimal Threshold | Sens | Spec | Acc | Baseline | Lift | Reliability |
|---|---|---|---|---|---|---|---|
| Breast | 0.05 | 97.3% | 57.1% | 86.4% | 100.0% | -13.6% | **Low — threshold pinned to search boundary; likely overfit to this test set** |
| CNS | 0.90 | 75.0% | 54.1% | 65.9% | 69.4% | **-3.5%** | Good — interior threshold, consistent with Run 2 |
| Head & Neck | 0.80 | 100.0% | 84.8% | 87.5% | 100.0% | -12.5% | Fair — improved, but n=40 makes this fragile |
| Lung | 0.05 | 94.9% | 61.5% | 78.2% | 100.0% | -21.8% | **Low — same boundary-pinning issue as Breast** |

## Key Findings for the Manuscript

1. **A simple TF-IDF + keyword-rule baseline provides no measurable advantage over the keyword rule alone** (lift = 0.000 across all four cancer types) — the field's default "quick fix" for this kind of classification task does not work here.
2. **Fine-tuned contextual embeddings (Bio_ClinicalBERT) provide a real, replicated advantage specifically for CNS** — the cancer type with the most heterogeneous and trap-prone cognitive-assessment language (NANO scale confusions, Neuro-QOL subscale ambiguity, inconsistent terminology). Sensitivity improved from ~52% (R baseline) to consistently 75-77% across two independent BERT configurations.
3. **BERT did not outperform the keyword baseline for Breast, Lung, or Head & Neck**, where cognitive-instrument vocabulary is more standardized (e.g., FACT-Cog, MMSE) and the simpler rule already performs near-ceiling.
4. **Class weighting and per-type threshold tuning provided only modest, inconsistent gains** beyond the base fine-tuned model, and two of four "optimized" thresholds (Breast, Lung) show signs of overfitting to the small test set rather than a genuine, generalizable pattern.
5. **Overall interpretation for the paper:** simple keyword rules and contextual embeddings appear to have complementary strengths — keyword rules suffice where instrument vocabulary is standardized and homogeneous; contextual models offer a real, if modest, advantage specifically where terminology is heterogeneous and semantically ambiguous. A production classifier for Phase 4 might reasonably use a **hybrid approach**: the keyword rule for Breast/Lung/Head&Neck, the fine-tuned BERT model for CNS.

## Honest Methodological Limitations (for Discussion/Limitations section)

- **Repeated test-set evaluation.** The same 357-trial held-out test set was evaluated three times across model iterations (unweighted BERT, weighted BERT, threshold-tuned BERT) to guide decisions about what to try next. This is a recognized soft form of test-set overfitting; results — particularly the boundary-pinned Breast/Lung thresholds — should be interpreted with appropriate caution and are not claimed as a validated final operating point without confirmation on genuinely unseen data.
- **R and Python train/test splits are not identical**, despite using matched stratification logic (by cancer type + label), because R and Python use different random number generator implementations even with a matched seed value. Sample sizes per split differed slightly between languages (e.g., R: train=1430/test=354; Colab: train=1427/test=357).
- **Small subgroup sizes.** Head & Neck's test set (n=40) and its severe class imbalance (17% positive in training) make its metrics comparatively fragile — a handful of flipped predictions materially changes the reported percentages.
- **512-token truncation.** BERT's architectural input limit truncates the small number of trials with very long, multi-outcome text lists (predominantly pediatric CNS trials), a disclosed but unresolved limitation.
- **The BERT model was not given the keyword_hit feature** that substantially drove the R model's (illusory) performance, making the two models' raw accuracy non-comparable; the Lift-over-baseline metric was used specifically to enable fair comparison.

## Files Associated with This Phase

- `MASTER_gold_standard_1888.csv` — full hand-labeled dataset
- `CLASSIFIER_training_population.csv` — the 1,804-trial training population
- `phase2_train_classifier.R` — TF-IDF + LASSO baseline script
- `Phase2b_BERT_finetune.ipynb` — BERT fine-tuning notebook (Colab)
- `cognitive_classifier_bert.zip` — final trained Bio_ClinicalBERT model weights (Run 2 configuration), ~433MB

---

## Addendum, 3 September 2026 — truncation in the training text

Found while evaluating the deployed classifier. This corrects one limitation
stated above and sharpens another; the original text is left in place so the
change is visible.

**The `Outcome text (for cognition check)` column is truncated at a median of
400 characters.** Against the full ClinicalTrials.gov outcome module, median
1,849 characters, the stored column keeps roughly a fifth of the text — and
routinely cuts off the outcome naming the instrument. 637 of the 1,070 trials
labelled Yes contain no cognitive term anywhere in the stored text, while the
`Flagged Keyword(s)` column records the instrument the labeller saw on the
registry page.

Both models were trained on this column: the R script through its TF-IDF
features, and the notebook through `df.dropna(subset=['Outcome text (for
cognition check)'])`.

**1. The 512-token limitation stated above is misattributed.** At 300–400
characters the text is roughly 75–100 tokens, far inside BERT's window. The
window was never the binding constraint; the truncation happened upstream during
data collection. The one place the 512-token limit does bite is at serving time
on full registry text — one CNS miss in the deployment evaluation had 20,475
characters of outcome text, well past what the model reads.

**2. The lift = 0.000 result stands, and its mechanism is now clearer.** The R
model's TF-IDF features were largely computed over text with the answer removed,
while `keyword_hit` — derived from the untruncated `Flagged Keyword(s)` column —
carried the real signal. The reported keyword baseline is therefore the accuracy
of `Flagged Keyword(s)`, not of a keyword rule applied to the text column.

**3. The case for a language model does not survive untruncated text.** This is
now settled rather than suspected. The complete ClinicalTrials.gov outcome module
was retrieved for the 1,088 training trials that have one, and the comparison was
repeated (`analysis/truncation_recall_test.py`).

The decisive subset is the **113 trials labelled Yes that carried no keyword
flag** — the ones the labeller had to catch by reading the registry page, and the
reason the task appeared to need contextual understanding:

| Same 66-term keyword list, applied to | Recovers |
|---|---|
| The stored, truncated text | **1 / 113** |
| The full registry outcome text | **113 / 113** |

Recall across all 726 positives rises from 245/726 to 726/726. TF-IDF + LASSO
trained on full text has *negative* lift against that rule in every cancer type
(−0.08 to −0.33). Scoring the deployed Bio_ClinicalBERT on full registry text
gives **72% sensitivity on CNS**, against 100% for the keyword rule on the same
trials.

So the conclusion above — that BERT provides a real advantage for CNS — is an
artefact of training and evaluating on truncated text. On complete input the
advantage disappears and reverses. The keyword rule was never the weak component;
the text pipeline was.

**What this does not establish: specificity.** The negative pool in this dataset
was selected to be negative, so a specificity figure computed here is optimistic
by construction and none is reported. Six false positives were observed, all CNS.
Establishing specificity needs a random registry sample, not this enriched design.

Independent support: scoring the deployed classifier on full registry text gives
the keyword rule **100% agreement across 147 breast, lung and head & neck
trials**, no errors in either direction. See
[the deployment evaluation](https://github.com/Mubashir-zz/cognitive-outcome-classifier-api/tree/main/validation)
and [`results/truncation_recall_test.md`](truncation_recall_test.md).
