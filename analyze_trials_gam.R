#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(stringr)
  library(readr)
  library(ggplot2)
  library(mgcv)
  library(gratia)
})

# Config
# Set the path to the Optuna trials.csv and the metric to analyze
trials_csv <- "optuna_results/unified_standard_v3_unified_mode_standard/trials.csv"
metric_of_interest <- "saliency_snr"  # alternatives: "saliency_auc", "wiou", "val_acc"
out_prefix <- sub("\\.csv$", "", trials_csv)

message(sprintf("Reading: %s", trials_csv))
df <- readr::read_csv(trials_csv, show_col_types = FALSE)

# Ensure state is character for robust filtering
df <- df %>% mutate(state = as.character(state))

# Pivot user_attrs_*_gc*_cons* columns to long format and parse metric/GC/Cons from names
name_regex <- "^user_attrs_([A-Za-z0-9_]+)_gc([0-9]+\\.[0-9]+)_cons([0-9]+\\.[0-9]+)$"

long_df <- df %>%
  # Keep all columns; only the selected 'user_attrs_*' columns are pivoted
  pivot_longer(
    cols = starts_with("user_attrs_"),
    names_to = "attr",
    values_to = "score",
    values_drop_na = FALSE
  ) %>%
  filter(str_detect(attr, "_gc") & str_detect(attr, "_cons")) %>%
  tidyr::extract(
    col = attr,
    into = c("metric", "gc", "cons"),
    regex = name_regex,
    remove = TRUE
  ) %>%
  mutate(
    gc = suppressWarnings(as.numeric(gc)),
    cons = suppressWarnings(as.numeric(cons)),
    score = suppressWarnings(as.numeric(score)),
    metric = as.character(metric)
  )

# Filter: keep selected metric, drop NA scores, drop PRUNED trials
dat <- long_df %>%
  filter(
    metric == .env$metric_of_interest,
    !is.na(score),
    !is.na(gc), !is.na(cons),
    !grepl("PRUNED", toupper(state))
  )

if (nrow(dat) < 10) {
  stop(sprintf("Not enough rows after filtering for metric '%s' (n=%d).", metric_of_interest, nrow(dat)), call. = FALSE)
}

message(sprintf("Fitting GAM for metric='%s' on %d rows from %d trials...",
                metric_of_interest, nrow(dat), dplyr::n_distinct(dat$number)))

# Choose spline basis sizes based on unique values to avoid mgcv k>unique errors
n_gc <- dplyr::n_distinct(dat$gc)
n_cons <- dplyr::n_distinct(dat$cons)
k_gc <- max(3, min(10, n_gc - 1))
k_cons <- max(3, min(10, n_cons - 1))

# Fall back to linear terms if unique values are too few for a smooth
term_gc <- if (n_gc >= 4) sprintf("s(gc, k=%d)", k_gc) else "gc"
term_cons <- if (n_cons >= 4) sprintf("s(cons, k=%d)", k_cons) else "cons"
form_str <- sprintf("score ~ %s + %s", term_gc, term_cons)
message(sprintf("GAM formula: %s (unique gc=%d, cons=%d)", form_str, n_gc, n_cons))

gam_fit <- mgcv::gam(as.formula(form_str), data = dat, method = "REML")

print(summary(gam_fit))

# Visualize the splines and model diagnostic plots
## ========= Partial-effect plots (no saving; print to RStudio plots pane) ======
# Detect which terms are smooths
sm_labels <- mgcv::smooths(gam_fit)               # e.g., "s(gc)", "s(cons)"
is_smooth_gc   <- any(grepl("^s\\(gc\\)$",   sm_labels))
is_smooth_cons <- any(grepl("^s\\(cons\\)$", sm_labels))

term_name <- function(var, is_smooth) if (is_smooth) sprintf("s(%s)", var) else var

get_term_stats <- function(fit, var, is_smooth) {
  ss <- summary(fit)
  if (is_smooth) {
    row <- match(sprintf("s(%s)", var), rownames(ss$s.table))
    if (!is.na(row)) {
      sprintf("edf=%.2f, ref.df=%.2f, p=%g",
              ss$s.table[row, "edf"], ss$s.table[row, "Ref.df"], ss$s.table[row, "p-value"])
    } else NULL
  } else {
    row <- match(var, rownames(ss$p.table))
    if (!is.na(row)) {
      est <- ss$p.table[row, "Estimate"]; se <- ss$p.table[row, "Std. Error"]; p <- ss$p.table[row, "Pr(>|t|)"]
      sprintf("β=%.3f (SE=%.3f), p=%g", est, se, p)
    } else NULL
  }
}

plot_partial <- function(fit, var, is_smooth, n = 200) {
  tn   <- term_name(var, is_smooth)
  rng  <- range(dat[[var]], na.rm = TRUE)
  grid <- data.frame(
    gc   = median(dat$gc,   na.rm = TRUE),
    cons = median(dat$cons, na.rm = TRUE)
  )
  grid[[var]] <- seq(rng[1], rng[2], length.out = n)

  pr <- predict(fit, newdata = grid, type = "terms", se.fit = TRUE, terms = tn)
  dfp <- data.frame(
    x   = grid[[var]],
    fit = as.numeric(pr$fit),
    se  = as.numeric(pr$se.fit)
  )
  stats_lab <- get_term_stats(fit, var, is_smooth)

  p <- ggplot2::ggplot(dfp, ggplot2::aes(x = x, y = fit)) +
    ggplot2::geom_ribbon(ggplot2::aes(ymin = fit - 1.96 * se, ymax = fit + 1.96 * se), alpha = 0.2) +
    ggplot2::geom_line(linewidth = 0.8) +
    ggplot2::geom_hline(yintercept = 0, linetype = 2) +
    ggplot2::labs(
      x = var, y = sprintf("Partial effect of %s", var),
      title = sprintf("%s: %s", metric_of_interest, tn),
      subtitle = stats_lab,
      caption = "Centered term; ribbon = 95% CI"
    ) +
    ggplot2::theme_bw()
  print(p)
  invisible(p)
}

# Print partial-effect plots
plot_partial(gam_fit, "gc",   is_smooth_gc)
plot_partial(gam_fit, "cons", is_smooth_cons)

# Basic diagnostics on screen
resid_dev <- residuals(gam_fit, type = "deviance")
fit_vals  <- fitted(gam_fit)
diag_df   <- data.frame(resid = resid_dev, fitted = fit_vals)

print(
  ggplot2::ggplot(diag_df, ggplot2::aes(sample = resid)) +
    ggplot2::stat_qq() + ggplot2::stat_qq_line() +
    ggplot2::theme_bw() + ggplot2::labs(title = "QQ plot (deviance residuals)")
)
print(
  ggplot2::ggplot(diag_df, ggplot2::aes(x = fitted, y = resid)) +
    ggplot2::geom_point(alpha = 0.35, size = 1) +
    ggplot2::geom_smooth(method = "loess", se = FALSE, linewidth = 0.7) +
    ggplot2::geom_hline(yintercept = 0, linetype = 2) +
    ggplot2::theme_bw() + ggplot2::labs(title = "Residuals vs fitted", x = "Fitted", y = "Deviance residuals")
)
print(
  ggplot2::ggplot(diag_df, ggplot2::aes(x = resid)) +
    ggplot2::geom_histogram(bins = 40) +
    ggplot2::theme_bw() + ggplot2::labs(title = "Residuals histogram", x = "Deviance residual", y = "Count")
)

cat("\n--- mgcv::gam.check ---\n"); mgcv::gam.check(gam_fit)
cat("\n--- Concurvity ---\n");    print(mgcv::concurvity(gam_fit, full = TRUE))

## ===================== Single-Index Model (SIM) ===============================
# Fit a SIM: score ~ s(beta1*gc + beta2*cons)
# Choose a modest k for stability; increase if sample size supports it.
k_si <- max(5, min(10, floor(nrow(dat) / 20)))
gam_sim <- mgcv::gam(score ~ s(gc, cons, bs = "si", k = k_si), data = dat, method = "REML")
print(summary(gam_sim))

# Plot the SIM smooth directly (mgcv renders f(Z) vs the estimated index Z with 95% CI)
# This prints to the plots pane; no files written.
plot(gam_sim, pages = 1, residuals = FALSE, shade = TRUE, seWithMean = TRUE)

# Optional (if installed): high-quality ggplot-based rendering
if (requireNamespace("gratia", quietly = TRUE)) {
  gratia::draw(gam_sim, residuals = FALSE)        # f(Z) with CI
  gratia::appraise(gam_sim)                        # diagnostics
}


