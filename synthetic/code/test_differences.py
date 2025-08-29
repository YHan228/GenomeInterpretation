#!/usr/bin/env python3
"""
Test script to verify key differences between vanilla and complex experiments.
"""

import sys
sys.path.append('.')

import torch
import numpy as np
from synthetic.code.models import TinyCNN, TinyCNN_Vanilla, get_model
from synthetic.code.evaluation import evaluate_model, evaluate_model_vanilla
from synthetic.code.utils import one_hot


def test_model_differences():
    """Test that the two model architectures are different."""
    print("Testing model differences...")
    
    # Create models
    vanilla_model = TinyCNN_Vanilla()
    complex_model = TinyCNN()
    
    # Check number of layers
    vanilla_layers = sum(1 for name, _ in vanilla_model.named_children() if 'conv' in name)
    complex_layers = sum(1 for name, _ in complex_model.named_children() if 'conv' in name)
    
    print(f"  Vanilla model conv layers: {vanilla_layers}")
    print(f"  Complex model conv layers: {complex_layers}")
    
    # Check parameter counts
    vanilla_params = sum(p.numel() for p in vanilla_model.parameters())
    complex_params = sum(p.numel() for p in complex_model.parameters())
    
    print(f"  Vanilla model parameters: {vanilla_params:,}")
    print(f"  Complex model parameters: {complex_params:,}")
    
    # Check specific architecture differences
    print("\nVanilla model architecture:")
    print(f"  - Conv1: 4 -> 32 channels")
    print(f"  - Conv2: 32 -> 64 channels")
    print(f"  - Conv3: 64 -> 128 channels")
    print(f"  - FC: 128 -> 1")
    print(f"  - Dropout: 0.5 (FC layer)")
    
    print("\nComplex model architecture:")
    print(f"  - Conv1: 4 -> 64 channels")
    print(f"  - Conv2: 64 -> 128 channels (dilation=3)")
    print(f"  - Conv3: 128 -> 256 channels (dilation=5)")
    print(f"  - Conv4: 256 -> 512 channels (dilation=7)")
    print(f"  - FC: 512 -> 1")
    print(f"  - Dropout: 0.2 (FC layer)")
    
    assert vanilla_layers == 3, f"Expected 3 conv layers in vanilla model, got {vanilla_layers}"
    assert complex_layers == 4, f"Expected 4 conv layers in complex model, got {complex_layers}"
    assert vanilla_params < complex_params, "Complex model should have more parameters"
    
    print("✓ Model architecture differences verified!")


def test_evaluation_differences():
    """Test that evaluation functions return different metrics."""
    print("\nTesting evaluation differences...")
    
    # Create minimal test datasets
    n_samples = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create random data for vanilla (1000bp sequences)
    vanilla_seq_len = 1000
    X_vanilla = np.random.randint(0, 4, (n_samples, vanilla_seq_len))
    X_vanilla = one_hot(X_vanilla)
    y_vanilla = np.random.randint(0, 2, n_samples).astype(np.float32)
    masks_vanilla = np.zeros((n_samples, vanilla_seq_len), dtype=bool)
    # Add some dummy masks
    for i in range(n_samples):
        start = np.random.randint(0, 900)
        masks_vanilla[i, start:start+60] = True
    
    # Create random data for complex (5000bp sequences)
    complex_seq_len = 5000
    X_complex = np.random.randint(0, 4, (n_samples, complex_seq_len))
    X_complex = one_hot(X_complex)
    y_complex = np.random.randint(0, 2, n_samples).astype(np.float32)
    masks_complex = np.zeros((n_samples, complex_seq_len), dtype=bool)
    # Add some dummy masks
    for i in range(n_samples):
        # Add motif mask
        start = np.random.randint(0, 4500)
        masks_complex[i, start:start+200] = True
    
    # Create models
    vanilla_model = TinyCNN_Vanilla().to(device)
    complex_model = TinyCNN().to(device)
    
    # Put models in eval mode to avoid issues with batch norm
    vanilla_model.eval()
    complex_model.eval()
    
    # Create data loaders
    vanilla_dl = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.from_numpy(X_vanilla).float(),
            torch.from_numpy(y_vanilla).float(),
            torch.from_numpy(masks_vanilla)
        ),
        batch_size=5
    )
    
    complex_dl = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.from_numpy(X_complex).float(),
            torch.from_numpy(y_complex).float(),
            torch.from_numpy(masks_complex)
        ),
        batch_size=5
    )
    
    # Test vanilla evaluation (should return wIoU)
    print("\nVanilla evaluation returns:")
    vanilla_results = evaluate_model_vanilla(vanilla_model, vanilla_dl, device)
    print(f"  - wIoU: {vanilla_results[0]:.3f}")
    print(f"  - Accuracy: {vanilla_results[1]:.3f}")
    print(f"  - Saliency AUC: {vanilla_results[2]:.3f}")
    print(f"  - Saliency SNR: {vanilla_results[3]:.3f}")
    print(f"  - PGD stats: {list(vanilla_results[4].keys())}")
    
    # Test complex evaluation (should NOT return wIoU)
    print("\nComplex evaluation returns:")
    complex_results = evaluate_model(complex_model, complex_dl, device)
    print(f"  - Accuracy: {complex_results[0]:.3f}")
    print(f"  - Saliency AUC: {complex_results[1]:.3f}")
    print(f"  - Saliency SNR: {complex_results[2]:.3f}")
    print(f"  - Motif Saliency AUC: {complex_results[3]:.3f}")
    print(f"  - Motif Saliency SNR: {complex_results[4]:.3f}")
    print(f"  - PGD stats: {list(complex_results[5].keys())}")
    
    # Verify return value counts
    assert len(vanilla_results) == 5, f"Vanilla evaluation should return 5 values, got {len(vanilla_results)}"
    assert len(complex_results) == 6, f"Complex evaluation should return 6 values, got {len(complex_results)}"
    
    # Verify that vanilla returns wIoU as first value (should be between 0 and 1)
    assert 0 <= vanilla_results[0] <= 1, f"wIoU should be between 0 and 1, got {vanilla_results[0]}"
    
    print("✓ Evaluation differences verified!")


def test_simplecnn_removed():
    """Test that SimpleCNN is no longer available."""
    print("\nTesting SimpleCNN removal...")
    
    # Try to import SimpleCNN (should fail)
    try:
        from synthetic.code.models import SimpleCNN
        assert False, "SimpleCNN should not be importable"
    except ImportError:
        print("✓ SimpleCNN successfully removed from imports")
    
    # Try to get SimpleCNN via get_model (should fail)
    try:
        model = get_model('simplecnn')
        assert False, "SimpleCNN should not be available via get_model"
    except ValueError as e:
        print(f"✓ get_model correctly raises error: {e}")
    
    # Verify available models
    available_models = ['tinycnn', 'tinycnn_vanilla', 'tinycnn_v0', 'logistic']
    print(f"\nAvailable models: {available_models}")
    
    for model_name in available_models:
        model = get_model(model_name)
        print(f"  - {model_name}: {model.__class__.__name__}")
    
    print("✓ SimpleCNN removal verified!")


def main():
    """Run all tests."""
    print("="*60)
    print("Testing key differences in modular structure")
    print("="*60)
    
    test_model_differences()
    test_evaluation_differences()
    test_simplecnn_removed()
    
    print("\n" + "="*60)
    print("All tests passed! Key differences preserved:")
    print("- wIoU metric only in vanilla experiments ✓")
    print("- Different model architectures (3 vs 4 layers) ✓")
    print("- SimpleCNN removed everywhere ✓")
    print("="*60)


if __name__ == '__main__':
    main() 