import os
import numpy as np
import torch
from six.moves import cPickle
import helper
import robust_helper
from tfomics import utils
from torch.utils.data import TensorDataset, DataLoader

#------------------------------------------------------------------------

num_trials = 10
model_names = ['cnn-dist', 'cnn-local', 'cnn-local-deep']
activations = ['relu', 'exponential']
training_modes = ['robust', 'standard']
flip_fractions = [0.01, 0.05, 0.1, 0.15, 0.2]
PGD_BATCH_SIZE = 100

results_path = os.path.join('koo/results', 'task3_robust_3models')
params_path = os.path.join(results_path, 'model_params')
save_path = utils.make_directory(results_path, 'scores')

#------------------------------------------------------------------------
# setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# load data
data_path = 'koo/data/synthetic_code_dataset.h5'
data = helper.load_data(data_path)
x_train, y_train, x_valid, y_valid, x_test, y_test = data

# load ground truth values
test_model = helper.load_synthetic_models(data_path, dataset='test')
true_index = np.where(y_test[:,0] == 1)[0]

# Convert to torch tensors
X_data = x_test[true_index][:500]
Y_data = y_test[true_index][:500]

X_model = test_model[true_index][:500]
dataset = TensorDataset(torch.from_numpy(X_data).float(), torch.from_numpy(Y_data).float())
data_loader = DataLoader(dataset, batch_size=PGD_BATCH_SIZE)

#------------------------------------------------------------------------

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
                
                # Compute IG with both PGD and shuffle baselines for full ablation
                baseline_methods = ['pgd', 'shuffle']
                
                for baseline_method in baseline_methods:
                    integrated_scores = []
                    
                    for trial in range(num_trials):
                        
                        # load model
                        model = helper.load_model(model_name, activation=activation, last_layer_mode='flat').to(device)
                        name = f"{base_name}_{trial}"
                        print('model: ' + name)

                        # load weights
                        weights_path = os.path.join(params_path, name + '.pth')
                        model.load_state_dict(torch.load(weights_path, map_location=device))
                        model.eval()

                        X_tensor = torch.from_numpy(X_data).float()
                        
                        if baseline_method == 'pgd':
                            # --- Batched PGD Processing ---
                            all_pgd_baselines = []
                            all_pgd_stats = []
                            print(f"  Computing PGD baselines in batches...")
                            for xb_batch, yb_batch in data_loader:
                                xb_batch, yb_batch = xb_batch.to(device), yb_batch.to(device)
                                
                                pgd_baselines_batch, pgd_stats_batch = robust_helper.find_adversarial_baseline_pgd_for_probs_batch_optimized(
                                    model, xb_batch, yb_batch, device
                                )
                                all_pgd_baselines.extend([b.cpu() for b in pgd_baselines_batch])
                                all_pgd_stats.extend(pgd_stats_batch)

                            attackable_samples = [r for r in all_pgd_stats if r['initial_prediction_correct']]
                            pgd_success_count = sum(1 for r in attackable_samples if r['success'])
                            pgd_success_rate = pgd_success_count / len(attackable_samples) if attackable_samples else 0
                            print(f"  PGD baseline success rate: {pgd_success_rate:.3f}")

                            # Use PGD baselines for IG
                            baselines = torch.stack(all_pgd_baselines)
                            successful_pgd = torch.tensor([s['success'] for s in all_pgd_stats], dtype=torch.bool)
                            
                            # Fallback to zero baseline where PGD failed
                            baselines[~successful_pgd] = 0.0
                        else:  # baseline_method == 'shuffle'
                            # --- Shuffle-based baselines ---
                            print(f"  Computing shuffle baselines...")
                            num_background = 10  # number of shuffled baselines to use
                            batch_size = X_tensor.shape[0]
                            
                            # Create shuffle baselines by averaging over multiple shuffled versions
                            shuffle_baselines = []
                            for i in range(batch_size):
                                # Create multiple shuffled versions of this sample
                                sample = X_tensor[i].unsqueeze(0)  # (1, 4, seq_len)
                                shuffled_samples = []
                                
                                for _ in range(num_background):
                                    # Shuffle along the sequence dimension for each channel independently
                                    shuffled = sample.clone()
                                    for ch in range(sample.shape[1]):  # 4 channels
                                        perm = torch.randperm(sample.shape[2])
                                        shuffled[0, ch, :] = sample[0, ch, perm]
                                    shuffled_samples.append(shuffled)
                                
                                # Average the shuffled samples to create baseline
                                baseline = torch.stack(shuffled_samples).mean(dim=0)
                                shuffle_baselines.append(baseline.squeeze(0))
                            
                            baselines = torch.stack(shuffle_baselines)

                        # interpretability performance with integrated gradients
                        print(f'  integrated gradients maps with {baseline_method} baseline')
                        ig_scores = robust_helper.compute_integrated_gradients(model, X_tensor.to(device),
                                                                               baselines=baselines.to(device),
                                                                               target_class_index=0)
                        integrated_scores.append(ig_scores)

                    # save results with baseline method in filename
                    if baseline_method == 'shuffle':
                        file_path = os.path.join(save_path, f"{base_name}_shuffle.pickle")
                    else:
                        file_path = os.path.join(save_path, f"{base_name}.pickle")
                    with open(file_path, 'wb') as f:
                        cPickle.dump(np.array(integrated_scores), f, protocol=cPickle.HIGHEST_PROTOCOL)
