library(tidyverse)
library(keras)
library(reticulate)
use_condaenv("tf", required = TRUE)
Sys.setenv(TF_CPP_MIN_LOG_LEVEL = "0")
library(deepG)
library(tensorflow)
library(magrittr)
library(data.table)

# Check GPU visibility
physical_gpus <- tf$config$list_physical_devices("GPU")
print(physical_gpus)

set.seed(42)

filters <- c(32, 64, 64, 128, 256)
kernels <- c(12, 16, 24, 32, 64)
pools <- c(2, 2, 2, 4, 4)

model <- create_model_lstm_cnn(
  maxlen = 1000000,
  layer_lstm = NULL,
  layer_dense = c(128, 64, 2),
  dropout_dense = NULL,
  learning_rate = 0.000001,
  filters = filters,
  kernel_size = kernels,
  pool_size = pools)

target_from_csv <- "synthetic/file_labels.csv"
target_df <- read.csv(target_from_csv)
label_names <- names(target_df)[names(target_df) != "file"]
head(target_df)
print(label_names)
train_path <- "synthetic/data/train"
val_path <- "synthetic/data/validation"
check_path <- "synthetic/checkpoints"

hist <- train_model(
  train_type = "label_csv",
  target_from_csv = target_from_csv,
  model = model,
  path = train_path,
  path_val = val_path,
  path_checkpoint = check_path,
  train_val_ratio = 0.5,
  run_name = "synthetic_old_nodp",
  batch_size = 16,
  epochs = 100,
  steps_per_epoch = 100,
  format = "fasta",
  concat_seq = "",
  vocabulary_label = label_names,
  lr_plateau_factor = 0.8,
  patience = 10
)

plot_path <- file.path(check_path, "plot")
if (!dir.exists(plot_path)) { 
  dir.create(plot_path, recursive = TRUE)
}

png(file.path(plot_path, "training_plot.png"))
plot(hist)
dev.off()
cat("Training plot saved to:", plot_path)
