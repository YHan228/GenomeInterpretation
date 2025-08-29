import torch
import matplotlib.pyplot as plt
import numpy as np
import os

# --- Configuration ---
CONCENTRATION_VALUES = np.logspace(1, 4, 50)  # Log-spaced values from 10 to 10,000
N_SAMPLES = 10000  # Number of draws for visualization
OUTPUT_DIR = "plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def calculate_epsilon_stats(concentration_major: float, n_samples: int) -> tuple[float, float]:
    """
    Generates samples from a 4-component asymmetric Dirichlet distribution
    and calculates statistics about the deviation from the main component.

    Returns:
        A tuple of (mean_epsilon, std_err_epsilon).
    """
    # Alpha parameters: high for the first component ('A'), low for the others.
    alphas = torch.tensor([concentration_major, 1.0, 1.0, 1.0])
    dist = torch.distributions.Dirichlet(alphas)
    samples = dist.sample((n_samples,))
    
    # Epsilon is the deviation from the probability of the original base.
    # If the original base was p=1, epsilon is 1 - p_after_noise.
    epsilons = 1.0 - samples[:, 0]
    
    mean_epsilon = epsilons.mean().item()
    std_err_epsilon = (epsilons.std() / (n_samples**0.5)).item()
    
    return mean_epsilon, std_err_epsilon

def visualize_epsilon_curve():
    """
    Generates and plots the relationship between Dirichlet concentration and
    the effective epsilon.
    """
    print("Calculating effective epsilon across a range of Dirichlet concentration values...")

    mean_epsilons = []
    std_err_epsilons = []
    
    for conc in CONCENTRATION_VALUES:
        mean, std_err = calculate_epsilon_stats(conc, N_SAMPLES)
        mean_epsilons.append(mean)
        std_err_epsilons.append(std_err)

    mean_epsilons = np.array(mean_epsilons)
    std_err_epsilons = np.array(std_err_epsilons)

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(CONCENTRATION_VALUES, mean_epsilons, marker='o', linestyle='-', markersize=4, label='Mean Epsilon')
    ax.fill_between(
        CONCENTRATION_VALUES,
        mean_epsilons - std_err_epsilons,
        mean_epsilons + std_err_epsilons,
        color='blue',
        alpha=0.2,
        label='Standard Error'
    )
    
    # Formatting
    ax.set_xscale('log')
    ax.set_xlabel("Dirichlet Concentration (log scale)", fontweight='bold')
    ax.set_ylabel("Effective Epsilon (1 - P(original base))", fontweight='bold')
    ax.set_title("Effective Epsilon vs. Dirichlet Concentration for 4-Channel Data", fontsize=16, fontweight='bold')
    ax.grid(True, which="both", ls="--", linewidth=0.5)
    ax.legend()
    sns.despine(ax=ax)
    
    plt.tight_layout()
    
    plot_path = os.path.join(OUTPUT_DIR, "dirichlet_epsilon_curve.png")
    plt.savefig(plot_path, dpi=150)
    
    print(f"\nVisualization saved to: {plot_path}")
    print("The plot shows how the Dirichlet 'concentration' parameter controls the average amount of noise ('epsilon').")
    print("Higher concentration values lead to lower, more stable epsilon values, meaning the smoothed vector is closer to the original one-hot vector.")

if __name__ == "__main__":
    try:
        import seaborn as sns
        sns.set_style("ticks")
    except ImportError:
        pass
    visualize_epsilon_curve() 