library(deepG)
model <- keras::load_model_hdf5("synthetic/checkpoints/synthetic_gap_4/Ep.006-val_loss0.31-val_acc0.913.hdf5", compile = TRUE)
keras::k_set_value(model$optimizer$lr, 1e-6)

target_from_csv <- "synthetic/file_labels_updated.csv"
target_df <- read.csv(target_from_csv)
label_names <- names(target_df)[names(target_df) != "file"]
head(target_df)
print(label_names)
train_path <- "synthetic/sep_data/train"
val_path <- "synthetic/sep_data/validation"
check_path <- "synthetic/checkpoints"

# how are samples generated? all random or only start random?
# start at middle, then first half genes lost.
# do fasta files with a 1mbp

gen <- train_model(
  train_type = "label_csv",
  target_from_csv = target_from_csv,
  model = model,
  path = train_path,
  path_val = val_path,
  path_checkpoint = check_path,
  run_name = "synthetic_weiter",
  batch_size = 16,
  epochs = 100,
  steps_per_epoch = 100,
  format = "fasta",
  concat_seq = "",
  vocabulary_label = label_names,
  lr_plateau_factor = 0.8,
  patience = 10,
  initial_epoch = 6,
  train_val_ratio = 0.1,
  return_gen = TRUE
)
