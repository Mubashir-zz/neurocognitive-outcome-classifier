###############################################################################
# Sensitivity analysis — does the Phase 2 result survive on untruncated text?
#
# The `Outcome text (for cognition check)` column in the labelled dataset is
# truncated at a median of 400 characters. The full ClinicalTrials.gov outcome
# module has a median of roughly 1,850. For most trials labelled Yes, the stored
# text no longer contains the instrument the labeller saw.
#
# Phase 2 concluded that a TF-IDF + LASSO model adds no lift over a keyword rule.
# That conclusion was reached on the truncated text. This script repeats the
# comparison on complete registry text for the ClinicalTrials.gov subset, so the
# conclusion can be stated on inputs that actually contain the signal.
#
# Run analysis/fetch_full_outcome_text.R's Python counterpart first:
#   python analysis/fetch_full_outcome_text.py
#   Rscript analysis/phase2_full_text_sensitivity.R
###############################################################################

suppressPackageStartupMessages({
  library(tidyverse); library(tidytext); library(glmnet); library(Matrix); library(jsonlite)
})
set.seed(2026)

labelled <- read_csv("data/CLASSIFIER_training_population.csv", show_col_types = FALSE) %>%
  transmute(
    trial_id    = `NCT/TrialID`,
    cancer_type = `Cancer Type`,
    stored_text = `Outcome text (for cognition check)`,
    keyword_hit = as.integer(!is.na(`Flagged Keyword(s)`) & `Flagged Keyword(s)` != ""),
    label       = if_else(`Measures cognition? (Y/N)` == "Yes", 1L, 0L)
  ) %>%
  filter(!is.na(stored_text), stored_text != "")

full <- read_csv("data/derived/ctgov_full_outcome_text.csv", show_col_types = FALSE)

dat <- labelled %>%
  inner_join(full, by = "trial_id") %>%
  distinct(trial_id, .keep_all = TRUE) %>%
  mutate(row_id = row_number())

message("Labelled trials: ", nrow(labelled))
message("With retrievable ClinicalTrials.gov text: ", nrow(dat))
message("Median stored chars:   ", median(nchar(dat$stored_text)))
message("Median registry chars: ", median(dat$n_chars))

# ---- Keyword rule applied to text, not to the Flagged Keyword(s) column ------
kw <- fromJSON("data/derived/cognitive_keywords.json")$keywords
kw_rule <- function(x) as.integer(str_detect(str_to_lower(x), fixed(paste(kw, collapse = "|")) %>% {NULL} %||% ""))
# str_detect with a fixed alternation is not valid; use a straightforward loop
kw_rule <- function(x) {
  lx <- str_to_lower(x)
  hit <- rep(FALSE, length(lx))
  for (k in kw) hit <- hit | str_detect(lx, fixed(k))
  as.integer(hit)
}

dat <- dat %>% mutate(
  kw_stored = kw_rule(stored_text),
  kw_full   = kw_rule(full_outcome_text)
)

acc <- function(pred, truth) mean(pred == truth)

cat("\n================ KEYWORD RULE, APPLIED TO THE TEXT ================\n")
kw_tab <- dat %>%
  group_by(cancer_type) %>%
  summarise(
    n = n(),
    acc_stored_text = round(acc(kw_stored, label), 4),
    acc_full_text   = round(acc(kw_full,   label), 4),
    sens_stored     = round(sum(kw_stored == 1 & label == 1) / max(sum(label == 1), 1), 4),
    sens_full       = round(sum(kw_full   == 1 & label == 1) / max(sum(label == 1), 1), 4),
    .groups = "drop"
  )
print(kw_tab)
cat("\nOverall: stored-text accuracy ", round(acc(dat$kw_stored, dat$label), 4),
    " | full-text accuracy ", round(acc(dat$kw_full, dat$label), 4), "\n", sep = "")
cat("Overall sensitivity: stored ",
    round(sum(dat$kw_stored == 1 & dat$label == 1) / sum(dat$label == 1), 4),
    " -> full ",
    round(sum(dat$kw_full == 1 & dat$label == 1) / sum(dat$label == 1), 4), "\n", sep = "")

# ---- TF-IDF + LASSO on each text source -------------------------------------
dat <- dat %>% mutate(strata = paste(cancer_type, label))
test_ids <- dat %>% group_by(strata) %>% slice_sample(prop = 0.2) %>% pull(row_id)
train <- dat %>% filter(!row_id %in% test_ids)
test  <- dat %>% filter(row_id %in% test_ids)
message("\nTrain ", nrow(train), " | Test ", nrow(test))

fit_and_eval <- function(text_col, label_name) {
  tok <- function(df) {
    df %>% select(row_id, text = all_of(text_col)) %>%
      unnest_tokens(word, text) %>%
      anti_join(stop_words, by = "word") %>%
      filter(str_detect(word, "[a-z]")) %>%
      count(row_id, word) %>% bind_tf_idf(word, row_id, n)
  }
  tr_tok <- tok(train)
  vocab  <- tr_tok %>% count(word, sort = TRUE) %>% filter(n >= 3) %>% pull(word)
  to_sparse <- function(tk, ids) {
    tk <- tk %>% filter(word %in% vocab)
    tk$word   <- factor(tk$word, levels = vocab)
    tk$row_id <- factor(tk$row_id, levels = ids)
    sparseMatrix(i = as.integer(tk$row_id), j = as.integer(tk$word), x = tk$tf_idf,
                 dims = c(length(ids), length(vocab)), dimnames = list(ids, vocab))
  }
  Xtr <- to_sparse(tr_tok, train$row_id)
  Xte <- to_sparse(tok(test), test$row_id)
  ytr <- train$label[match(rownames(Xtr), train$row_id)]
  yte <- test$label[match(rownames(Xte), test$row_id)]

  cvfit <- cv.glmnet(Xtr, ytr, family = "binomial", alpha = 1, nfolds = 5, type.measure = "auc")
  pred  <- as.integer(as.numeric(predict(cvfit, Xte, s = "lambda.min", type = "response")) >= 0.5)

  ct <- test$cancer_type[match(rownames(Xte), test$row_id)]
  base_col <- if (text_col == "stored_text") test$kw_stored else test$kw_full
  base <- base_col[match(rownames(Xte), test$row_id)]

  tibble(source = label_name, cancer_type = ct, actual = yte, model = pred, keyword = base) %>%
    group_by(source, cancer_type) %>%
    summarise(n = n(),
              model_acc    = round(mean(model == actual), 4),
              keyword_acc  = round(mean(keyword == actual), 4),
              lift         = round(mean(model == actual) - mean(keyword == actual), 4),
              .groups = "drop")
}

cat("\n================ TF-IDF + LASSO vs KEYWORD RULE ON THE SAME TEXT ================\n")
res <- bind_rows(
  fit_and_eval("stored_text",       "truncated (as in Phase 2)"),
  fit_and_eval("full_outcome_text", "full registry text")
)
print(res, n = 40)

dir.create("results", showWarnings = FALSE)
write_csv(kw_tab, "results/full_text_keyword_comparison.csv")
write_csv(res,    "results/full_text_lift_comparison.csv")
cat("\nWrote results/full_text_keyword_comparison.csv and results/full_text_lift_comparison.csv\n")
