import torch
import numpy as np
from captum.attr import IntegratedGradients

def integrated_gradients(model, X, target_class_index, num_steps=20, reference='shuffle', num_background=10):
    """
    Compute integrated gradients for a batch of sequences.

    Args:
        model (torch.nn.Module): The PyTorch model.
        X (torch.Tensor): Input sequences (batch_size, seq_len, num_features).
        target_class_index (int): The index of the target class for which to
                                  compute attributions.
        num_steps (int): The number of steps for the integration.
        reference (str): The type of reference to use ('shuffle' or 'zeros').
        num_background (int): Number of reference sequences to average over.

    Returns:
        np.ndarray: An array of attribution scores averaged over all references.
    """
    
    # Ensure model is in evaluation mode
    model.eval()

    # Create an IntegratedGradients object
    ig = IntegratedGradients(model)

    # Accumulate attributions over multiple references
    all_attributions = []
    
    for i in range(num_background):
        # Create a baseline (reference)
        if reference == 'shuffle':
            # Create a shuffled baseline for each sequence in the batch
            batch_size, seq_len, _ = X.shape
            # Permute along the sequence length dimension for each item in the batch
            indices = torch.stack([torch.randperm(seq_len) for _ in range(batch_size)])
            # Apply these indices to each item in the batch
            baseline = X[torch.arange(batch_size).unsqueeze(1), indices]
            
        elif reference == 'zeros':
            baseline = torch.zeros_like(X)
        else:
            raise ValueError(f"Unknown reference type: {reference}")

        # Compute attributions for this reference
        attributions, delta = ig.attribute(X, 
                                           baselines=baseline, 
                                           target=target_class_index,
                                           n_steps=num_steps,
                                           return_convergence_delta=True)
        
        all_attributions.append(attributions.cpu().numpy())
    
    # Average attributions across all references
    avg_attributions = np.mean(all_attributions, axis=0)
    
    return avg_attributions 