###############################################################################
# PHASE 2 — Building and validating the neurocognitive-outcome classifier
#
# WHAT THIS DOES
#   1. Loads your master gold-standard training file (1,804 hand-labeled trials
#      across CNS, Breast, Lung, Head&Neck)
#   2. Converts each trial's outcome text into numeric features (TF-IDF)
#   3. Trains a LASSO logistic regression classifier (glmnet) to predict
#      whether a trial measures a neurocognitive outcome
#   4. Evaluates it on a held-out test set it never saw during training
#   5. Reports performance overall AND broken out by cancer type, so you can
#      see where it generalizes well and where it doesn't
#   6. Saves the trained model + vectorizer so Phase 4 (scoring the full
#      ~50,000-trial landscape) can reuse it without retraining
#
# HOW TO RUN
#   From the repo root:  Rscript analysis/phase2_train_classifier.R
#   Or with an explicit input and output directory:
#     Rscript analysis/phase2_train_classifier.R data/CLASSIFIER_training_population.csv .
#   In RStudio, Source (Cmd/Ctrl+Shift+S) also works -- it falls back to a
#   file picker if the default input path is not found. First run installs packages.
###############################################################################

# ---- 0. Setup ----
.need <- c("tidyverse", "tidytext", "glmnet", "Matrix", "pROC", "scales")
.miss <- .need[!vapply(.need, requireNamespace, logical(1), quietly = TRUE)]
if (length(.miss)) install.packages(.miss, repos = "https://cloud.r-project.org")
library(tidyverse); library(tidytext); library(glmnet); library(Matrix); library(pROC); library(scales)

set.seed(2026)  # reproducibility

# Input resolution, in order of preference:
#   1. a path given on the command line   -> Rscript analysis/phase2_train_classifier.R data/...csv
#   2. the copy that ships with this repo -> works from the repo root with no arguments
#   3. a file picker, interactive sessions only
.args <- commandArgs(trailingOnly = TRUE)
.default_input <- "data/CLASSIFIER_training_population.csv"
INPUT <- if (length(.args) >= 1) {
  .args[[1]]
} else if (file.exists(.default_input)) {
  .default_input
} else if (interactive()) {
  message("\n>>> A file-picker window is opening. Select CLASSIFIER_training_population.csv <<<\n")
  file.choose()
} else {
  stop("No input file. Pass one as an argument, or run from the repo root where ",
       .default_input, " exists.", call. = FALSE)
}
if (!file.exists(INPUT)) stop("Input file not found: ", INPUT, call. = FALSE)
message("Input: ", INPUT)

OUT <- if (length(.args) >= 2) .args[[2]] else "."
fig_dir <- file.path(OUT, "figures"); res_dir <- file.path(OUT, "results"); model_dir <- file.path(OUT, "models")
for (d in c(fig_dir, res_dir, model_dir)) dir.create(d, showWarnings = FALSE, recursive = TRUE)

pub_theme <- theme_minimal(base_size = 12) +
  theme(panel.grid.minor = element_blank(), axis.title = element_text(face = "bold"),
        plot.title = element_text(face = "bold", size = 13), legend.position = "top")

# ---- 1. Load and prepare data ----
dat <- read_csv(INPUT, show_col_types = FALSE) %>%
  mutate(row_id = row_number()) %>%   # FIX: guaranteed-unique key -- NCT/TrialID can repeat
  transmute(                          #      (basket trials legitimately match >1 cancer type)
    row_id,
    trial_id = `NCT/TrialID`,
    cancer_type = `Cancer Type`,
    text = `Outcome text (for cognition check)`,
    keyword_hit = as.integer(!is.na(`Flagged Keyword(s)`) & `Flagged Keyword(s)` != ""),
    label = if_else(`Measures cognition? (Y/N)` == "Yes", 1L, 0L)
  ) %>%
  filter(!is.na(text), text != "")

message("Loaded ", nrow(dat), " labeled trials.")
print(table(dat$cancer_type, dat$label))

# ---- 2. Stratified train/test split (80/20), stratified by cancer type AND label ----
dat <- dat %>% mutate(strata = paste(cancer_type, label))
test_idx <- dat %>% group_by(strata) %>% slice_sample(prop = 0.2) %>% pull(row_id)
train <- dat %>% filter(!row_id %in% test_idx)
test  <- dat %>% filter(row_id %in% test_idx)
message("Train: ", nrow(train), "  |  Test (held out): ", nrow(test))

# ---- 3. Text features: TF-IDF on unigrams + bigrams ----
tokenize_tfidf <- function(df) {
  df %>%
    unnest_tokens(word, text, token = "ngrams", n = 1) %>%
    anti_join(stop_words, by = "word") %>%
    filter(str_detect(word, "[a-z]")) %>%
    count(row_id, word) %>%
    bind_tf_idf(word, row_id, n)
}

train_tokens <- tokenize_tfidf(train)
vocab <- train_tokens %>% count(word, sort = TRUE) %>% filter(n >= 3) %>% pull(word)  # drop ultra-rare words
message("Vocabulary size (words appearing >=3 times in training set): ", length(vocab))

to_sparse <- function(tokens, vocab, ids) {
  tokens <- tokens %>% filter(word %in% vocab)
  tokens$word <- factor(tokens$word, levels = vocab)
  tokens$row_id <- factor(tokens$row_id, levels = ids)
  sparseMatrix(i = as.integer(tokens$row_id), j = as.integer(tokens$word), x = tokens$tf_idf,
               dims = c(length(ids), length(vocab)), dimnames = list(ids, vocab))
}

X_train_text <- to_sparse(train_tokens, vocab, train$row_id)
test_tokens  <- tokenize_tfidf(test)
X_test_text  <- to_sparse(test_tokens, vocab, test$row_id)

# add the keyword-hit flag as one more feature alongside the TF-IDF matrix
X_train <- cbind(X_train_text, keyword_hit = train$keyword_hit[match(rownames(X_train_text), train$row_id)])
X_test  <- cbind(X_test_text,  keyword_hit = test$keyword_hit[match(rownames(X_test_text), test$row_id)])
y_train <- train$label[match(rownames(X_train_text), train$row_id)]
y_test  <- test$label[match(rownames(X_test_text), test$row_id)]

# ---- 4. Train LASSO logistic regression with cross-validated lambda ----
cvfit <- cv.glmnet(X_train, y_train, family = "binomial", alpha = 1, nfolds = 5, type.measure = "auc")
message(sprintf("\nBest lambda (cross-validated): %.5f | CV AUC: %.3f",
                 cvfit$lambda.min, max(cvfit$cvm)))

# ---- 5. Evaluate on held-out test set ----
pred_prob <- as.numeric(predict(cvfit, X_test, s = "lambda.min", type = "response"))
pred_class <- if_else(pred_prob >= 0.5, 1L, 0L)

confmat <- table(Predicted = pred_class, Actual = y_test)
TP <- confmat["1","1"]; TN <- confmat["0","0"]; FP <- confmat["1","0"]; FN <- confmat["0","1"]
sens <- TP/(TP+FN); spec <- TN/(TN+FP); ppv <- TP/(TP+FP); npv <- TN/(TN+FN)
f1 <- 2*ppv*sens/(ppv+sens)
roc_obj <- roc(y_test, pred_prob, quiet = TRUE)
auc_val <- auc(roc_obj)

sink(file.path(res_dir, "phase2_performance_overall.txt"))
cat("PHASE 2 CLASSIFIER — HELD-OUT TEST SET PERFORMANCE\n")
cat("Test set size:", length(y_test), "\n\n")
print(confmat)
cat(sprintf("\nSensitivity (Recall): %.3f\n", sens))
cat(sprintf("Specificity:          %.3f\n", spec))
cat(sprintf("PPV (Precision):      %.3f\n", ppv))
cat(sprintf("NPV:                  %.3f\n", npv))
cat(sprintf("F1 score:             %.3f\n", f1))
cat(sprintf("AUROC:                %.3f\n", auc_val))
sink()
cat(readLines(file.path(res_dir, "phase2_performance_overall.txt")), sep = "\n")

# ---- 5b. HONESTY CHECK: how much does the model add beyond the simple keyword flag alone? ----
# This matters: if keyword_hit alone predicts the label almost perfectly, the model
# isn't learning much beyond "was a cognitive word present" -- which we already knew
# how to do. The real test is whether it beats that naive baseline, especially on CNS,
# where manual review caught the most nuanced traps (NANO, QoL-subscale, etc.)
baseline_acc <- test %>%
  mutate(kw_pred = keyword_hit) %>%
  group_by(cancer_type) %>%
  summarise(baseline_accuracy = mean(kw_pred == label), n = n(), .groups = "drop")
cat("\n\n--- Naive baseline: keyword_hit alone as the prediction, by cancer type ---\n")
print(baseline_acc)
cat("\nCompare this to the model's real accuracy per type below. Where the model barely\n")
cat("beats this baseline, it has not learned much beyond simple keyword presence.\n")

by_type <- tibble(row_id = as.integer(rownames(X_test)), cancer_type = test$cancer_type[match(rownames(X_test), test$row_id)],
                   actual = y_test, predicted = pred_class, prob = pred_prob)

by_type_summary <- by_type %>%
  group_by(cancer_type) %>%
  summarise(n = n(),
            sens = sum(predicted==1 & actual==1) / max(sum(actual==1),1),
            spec = sum(predicted==0 & actual==0) / max(sum(actual==0),1),
            acc  = mean(predicted==actual),
            .groups = "drop")
by_type_summary <- by_type_summary %>% left_join(baseline_acc %>% select(cancer_type, baseline_accuracy), by = "cancer_type") %>%
  mutate(lift_over_baseline = acc - baseline_accuracy)
write_csv(by_type_summary, file.path(res_dir, "phase2_performance_by_cancer_type.csv"))
cat("\n\nPerformance by cancer type (held-out test set), with baseline comparison:\n")
cat("  acc = model accuracy | baseline_accuracy = keyword-hit-alone accuracy | lift = the difference\n")
print(by_type_summary)

# ---- 7. Which words drive the model (interpretability) ----
coefs <- coef(cvfit, s = "lambda.min")
coef_df <- tibble(term = rownames(coefs), weight = as.numeric(coefs)) %>%
  filter(term != "(Intercept)", weight != 0) %>% arrange(desc(weight))
write_csv(coef_df, file.path(res_dir, "phase2_top_predictive_terms.csv"))
cat("\nTop 15 words pushing toward COGNITIVE:\n"); print(head(coef_df, 15))
cat("\nTop 15 words pushing toward NOT cognitive:\n"); print(tail(coef_df, 15))

# ---- 8. Figures ----
f1_roc <- ggplot(data.frame(fpr = 1-roc_obj$specificities, tpr = roc_obj$sensitivities),
                  aes(fpr, tpr)) +
  geom_line(color = "#1F4E5F", linewidth = 1.1) + geom_abline(linetype = "dashed", color = "grey60") +
  labs(title = "Classifier ROC curve (held-out test set)",
       subtitle = sprintf("AUROC = %.3f", auc_val), x = "False Positive Rate", y = "True Positive Rate") +
  pub_theme
ggsave(file.path(fig_dir, "phase2_roc_curve.png"), f1_roc, width = 6, height = 5, dpi = 300)

f2_bytype <- by_type_summary %>% pivot_longer(c(sens, spec), names_to = "metric", values_to = "value") %>%
  mutate(metric = recode(metric, sens = "Sensitivity", spec = "Specificity"))
f2 <- ggplot(f2_bytype, aes(cancer_type, value, fill = metric)) +
  geom_col(position = "dodge") + scale_y_continuous(labels = percent_format(), limits = c(0,1)) +
  scale_fill_manual(values = c("Sensitivity"="#C0392B","Specificity"="#1F4E5F")) +
  labs(title = "Classifier performance by cancer type", x = NULL, y = NULL, fill = NULL) + pub_theme
ggsave(file.path(fig_dir, "phase2_performance_by_type.png"), f2, width = 7, height = 4.5, dpi = 300)

# ---- 9. Save model + vocabulary for Phase 4 deployment ----
saveRDS(list(model = cvfit, vocab = vocab, lambda = cvfit$lambda.min),
        file.path(model_dir, "cognitive_classifier_v1.rds"))

message("\nDone. Model  -> ", file.path(model_dir, "cognitive_classifier_v1.rds"))
message("Figures -> ", fig_dir, "   Results -> ", res_dir)
