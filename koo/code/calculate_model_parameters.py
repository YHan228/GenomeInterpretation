import torch
from model_zoo.torch_models import CnnDist, CnnLocal

def count_parameters(model):
    """Counts the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# --- CnnDist ---
cnn_dist_flat = CnnDist(last_layer_mode='flat')
num_params_dist_flat = count_parameters(cnn_dist_flat)

cnn_dist_noflat = CnnDist(last_layer_mode='noflat')
num_params_dist_noflat = count_parameters(cnn_dist_noflat)


# --- CnnLocal ---
cnn_local_flat = CnnLocal(last_layer_mode='flat')
num_params_local_flat = count_parameters(cnn_local_flat)

cnn_local_noflat = CnnLocal(last_layer_mode='noflat')
num_params_local_noflat = count_parameters(cnn_local_noflat)


# --- Print Results ---
print("="*40)
print("Model Parameter Comparison")
print("="*40)
print(f"CnnDist (flat):   {num_params_dist_flat:>10,} parameters")
print(f"CnnDist (noflat): {num_params_dist_noflat:>10,} parameters")
print("-"*40)
print(f"CnnLocal (flat):  {num_params_local_flat:>10,} parameters")
print(f"CnnLocal (noflat): {num_params_local_noflat:>10,} parameters")
print("="*40) 