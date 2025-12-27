import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from six.moves import cPickle
import helper
import robust_helper
from tfomics import utils
from sklearn.metrics import roc_auc_score, average_precision_score

#------------------------------------------------------------------------

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--num-trials', type=int, default=10)
parser.add_argument('--output-dir', type=str, default='task3_robust_3models')
args = parser.parse_args()

num_trials = args.num_trials
model_names = ['cnn-dist', 'cnn-local', 'cnn-local-deep']
activations = ['relu', 'exponential']
training_modes = ['robust', 'standard']
flip_fractions = [0.01, 0.05, 0.1, 0.15, 0.2]

results_path = utils.make_directory('koo/results', args.output_dir)
params_path = utils.make_directory(results_path, 'model_params')

#------------------------------------------------------------------------

# setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set deterministic behavior for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# load data
data_path = 'koo/data/synthetic_code_dataset.h5'
data = helper.load_data(data_path)
x_train, y_train, x_valid, y_valid, x_test, y_test = data

# Convert data to torch tensors
x_train = torch.from_numpy(x_train).float()
y_train = torch.from_numpy(y_train).float()
x_valid = torch.from_numpy(x_valid).float()
y_valid = torch.from_numpy(y_valid).float()
x_test = torch.from_numpy(x_test).float()
y_test = torch.from_numpy(y_test).float()

train_dataset = TensorDataset(x_train, y_train)
valid_dataset = TensorDataset(x_valid, y_valid)
test_dataset = TensorDataset(x_test, y_test)

num_workers = 0  # Avoid multiprocessing cleanup errors

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# Set global seed for data loading
g = torch.Generator()
g.manual_seed(42)

train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True, num_workers=num_workers, 
                         pin_memory=True, worker_init_fn=seed_worker, generator=g)
valid_loader = DataLoader(valid_dataset, batch_size=100, shuffle=False, num_workers=num_workers, 
                         pin_memory=True, worker_init_fn=seed_worker)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=num_workers, 
                        pin_memory=True, worker_init_fn=seed_worker)

#------------------------------------------------------------------------

with open(os.path.join(results_path, 'task3_classification_performance.tsv'), 'w') as f:
    f.write('%s\t%s\t%s\n'%('model', 'ave roc', 'ave pr'))

    results = {}
    for model_name in model_names:
        for activation in activations:
            for training_mode in training_modes:
                
                # set flip fraction for robust training
                if training_mode == 'robust':
                    loop_fractions = flip_fractions
                else:
                    loop_fractions = [0.0] # for standard training, loop once without flip

                for flip_fraction in loop_fractions:
                
                    # create base name
                    if training_mode == 'robust':
                        base_name = f"{model_name}_{activation}_robust_{flip_fraction}"
                    else:
                        base_name = f"{model_name}_{activation}_standard"
                    print(base_name)
                    results[base_name] = {}
                
                    trial_roc = []
                    trial_pr = []
                    for trial in range(num_trials):
                        # set seed for reproducibility
                        # Use same seed for ALL models in the same trial for fair comparison
                        seed = trial  # Simple: trial 0 uses seed 0, trial 1 uses seed 1, etc.
                        torch.manual_seed(seed)
                        np.random.seed(seed)
                        random.seed(seed)
                        if torch.cuda.is_available():
                            torch.cuda.manual_seed(seed)
                            torch.cuda.manual_seed_all(seed)
                        
                        # load model
                        model = helper.load_model(model_name, activation=activation, last_layer_mode='flat').to(device)
                        name = base_name+'_'+str(trial)
                        print('model: ' + name)

                        # setup optimizer and criterion
                        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-6)
                        
                        if training_mode == 'robust':
                            lr_patience = 8
                            early_stopping_patience = 30
                        else:
                            lr_patience = 5
                            early_stopping_patience = 20

                        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=lr_patience, min_lr=1e-7)
                        criterion = nn.BCELoss()

                        # early stopping parameters
                        best_val_auroc = 0
                        patience_counter = 0
                        best_epoch = 0

                        # training loop
                        for epoch in range(100):
                            model.train()
                            for inputs, targets in train_loader:
                                inputs, targets = inputs.to(device), targets.to(device)
                                if training_mode == 'robust':
                                    inputs = robust_helper.generate_direct_hotflip_examples_optimized(model, inputs, targets, criterion, flip_fraction)

                                optimizer.zero_grad()
                                outputs = model(inputs)
                                loss = criterion(outputs, targets)
                                loss.backward()
                                optimizer.step()
                            
                            # Validation
                            model.eval()
                            val_preds = []
                            with torch.no_grad():
                                for inputs, _ in valid_loader:
                                    inputs = inputs.to(device)
                                    outputs = model(inputs)
                                    val_preds.append(outputs.detach())
                            
                            val_preds = torch.cat(val_preds).cpu().numpy()
                            val_auroc = roc_auc_score(y_valid.numpy(), val_preds)
                            scheduler.step(val_auroc)

                            if (epoch + 1) % 10 == 0:
                                  print(f'  Epoch [{epoch+1}/100], val_auroc: {val_auroc:.4f}')

                            # Track best model according to paper
                            if val_auroc > best_val_auroc:
                                best_val_auroc = val_auroc
                                best_epoch = epoch + 1
                                patience_counter = 0
                            else:
                                patience_counter += 1
                                if patience_counter >= early_stopping_patience:
                                    print(f'  Early stopping at epoch {epoch+1} (best was epoch {best_epoch})')
                                    break
                        
                        # Save the final model's weights
                        weights_path = os.path.join(params_path, name + '.pth')
                        torch.save(model.state_dict(), weights_path)

                        # predict test sequences and calculate performance metrics
                        model.eval()
                        predictions = []
                        with torch.no_grad():
                            for inputs, _ in test_loader:
                                inputs = inputs.to(device)
                                outputs = model(inputs)
                                predictions.append(outputs.detach())
                        predictions = torch.cat(predictions).cpu().numpy()
                        
                        roc_score = roc_auc_score(y_test.numpy(), predictions)
                        pr_score = average_precision_score(y_test.numpy(), predictions)
                        
                        trial_roc.append(roc_score)
                        trial_pr.append(pr_score)

                    results[base_name] = [np.array(trial_roc), np.array(trial_pr)]
                    f.write("%s\t%.3f+/-%.3f\t%.3f+/-%.3f\n"%(base_name, 
                                                              np.mean(trial_roc),
                                                              np.std(trial_roc), 
                                                              np.mean(trial_pr),
                                                              np.std(trial_pr)))

# save results
file_path = os.path.join(results_path, 'task3_performance_results.pickle')
with open(file_path, 'wb') as f:
    cPickle.dump(results, f, protocol=cPickle.HIGHEST_PROTOCOL)


