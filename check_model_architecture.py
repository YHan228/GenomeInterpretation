#!/usr/bin/env python3
import torch
import sys

def check_model_architecture(model_path):
    """Load and examine the architecture of a PyTorch model."""
    try:
        # Load the model
        model = torch.load(model_path, map_location='cpu')
        
        print(f"Model loaded from: {model_path}")
        print("=" * 50)
        
        # Check if it's a state dict or a full model
        if isinstance(model, dict):
            print("Model type: State dictionary")
            print(f"Number of parameters: {len(model)}")
            print("\nParameter names and shapes:")
            print("-" * 30)
            
            for name, param in model.items():
                if isinstance(param, torch.Tensor):
                    print(f"{name}: {param.shape} ({param.dtype})")
                else:
                    print(f"{name}: {type(param)}")
                    
        else:
            print("Model type: Full model object")
            print(f"Model class: {type(model)}")
            
            # Try to print model structure
            try:
                print("\nModel architecture:")
                print("-" * 20)
                print(model)
            except:
                print("Could not print model architecture")
            
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            print(f"\nTotal parameters: {total_params:,}")
            print(f"Trainable parameters: {trainable_params:,}")
            
    except Exception as e:
        print(f"Error loading model: {e}")
        return False
    
    return True

if __name__ == "__main__":
    model_path = "/home/yhan/GenomeInterpretation/phenotype/model/spore_formation/best_model.pth"
    check_model_architecture(model_path)
