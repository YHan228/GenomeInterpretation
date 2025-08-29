import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_skill(auc):
    """Calculates Skill from AUC."""
    return 2 * auc - 1

def calculate_root_skill(skill):
    """Calculates Root-Skill from Skill."""
    return np.sign(skill) * np.sqrt(np.abs(skill))

def calculate_auc_from_root_skill(root_skill):
    """Calculates AUC from Root-Skill."""
    skill = np.sign(root_skill) * (root_skill**2)
    auc = (skill + 1) / 2
    return auc

def main():
    """Main function to run the analysis and visualization."""
    # --- Part 1: Tabulate AUC, Skill, and Root-Skill ---
    print("--- Relationship between AUC, Skill, and Root-Skill ---")
    aucs = np.linspace(0, 1, 21)
    skills = calculate_skill(aucs)
    root_skills = calculate_root_skill(skills)

    df = pd.DataFrame({
        'AUC': aucs,
        'Skill (S = 2p-1)': skills,
        'Root-Skill (Φ = sign(S)·√|S|)': root_skills
    })
    print(df.to_string(index=False))

    # --- Part 2: Visualize the relationship ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot Skill
    color = 'tab:blue'
    ax1.set_xlabel('SaliencyAUC (p)')
    ax1.set_ylabel('Skill (S)', color=color)
    ax1.plot(df['AUC'], df['Skill (S = 2p-1)'], color=color, label='Skill (Linear)')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.axhline(0, color='grey', linestyle='--', linewidth=0.7)
    ax1.axvline(0.5, color='grey', linestyle='--', linewidth=0.7)

    # Instantiate a second y-axis for Root-Skill
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Root-Skill (Φ)', color=color)
    ax2.plot(df['AUC'], df['Root-Skill (Φ = sign(S)·√|S|)'], color=color, linestyle='--', label='Root-Skill (Concave)')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.suptitle('Relationship between SaliencyAUC, Skill, and Root-Skill', fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('synthetic/support_scripts/skill_visualization.png', dpi=300)
    print("\nSaved plot to synthetic/support_scripts/skill_visualization.png")

    # --- Part 3: Tabulate and plot what changes in Effective Root-Skill mean for AUC ---
    print("\n--- Interpreting ΔEffectiveRootSkill ---")
    print("This metric only measures skill in the positive domain (AUC > 0.5).\n")

    # Use a finer range for the plot for smooth curves
    delta_values_plot = np.linspace(-0.75, 0.75, 151)
    
    # Use a coarser range for the table
    delta_values_table = np.array([-0.5, -0.25, -0.1, 0.1, 0.25, 0.5, 0.75])

    initial_aucs = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    plot_results = []
    table_results = []

    # Generate data for both plot and table
    for initial_auc in initial_aucs:
        initial_skill = calculate_skill(initial_auc)
        initial_root_skill = calculate_root_skill(initial_skill)
        initial_effective_root_skill = np.maximum(0, initial_root_skill)
        
        # Plot data (fine-grained)
        for delta in delta_values_plot:
            # We are plotting the change in *Final AUC* as a function of a hypothetical *change in effective skill*
            final_effective_root_skill = initial_effective_root_skill + delta
            # This is then converted back to a regular root_skill for AUC calculation
            final_root_skill = final_effective_root_skill if final_effective_root_skill > 0 else -1 * (final_effective_root_skill**2)

            final_auc = calculate_auc_from_root_skill(final_root_skill)
            plot_results.append({
                'Initial AUC': initial_auc,
                'Final AUC': final_auc,
                'ΔEffectiveRootSkill': delta
            })

        # Table data (coarse-grained)
        for delta in delta_values_table:
            final_effective_root_skill = initial_effective_root_skill + delta
            final_root_skill = final_effective_root_skill if final_effective_root_skill > 0 else -1 * (final_effective_root_skill**2)
            final_auc = calculate_auc_from_root_skill(final_root_skill)
            
            table_results.append({
                'Initial AUC': f"{initial_auc:.2f}",
                'Initial Eff. RootSkill': f"{initial_effective_root_skill:+.2f}",
                'ΔEffectiveRootSkill': f"{delta:+.2f}",
                'Final Eff. RootSkill': f"{final_effective_root_skill:+.2f}",
                'Final AUC': f"{final_auc:.2f}"
            })

    # Print the table
    table_df = pd.DataFrame(table_results)
    print("Table showing what a change in EffectiveRootSkill means for AUC:")
    print(table_df.to_string(index=False))

    # --- Part 4: Visualize the ΔEffectiveRootSkill table ---
    print("\nGenerating plot for interpreting ΔEffectiveRootSkill...")
    plot_df = pd.DataFrame(plot_results)
    
    plt.figure(figsize=(12, 8))
    palette = sns.color_palette("viridis", n_colors=len(initial_aucs))
    
    sns.lineplot(
        data=plot_df,
        x='ΔEffectiveRootSkill',
        y='Final AUC',
        hue='Initial AUC',
        palette=palette,
        linewidth=2.5
    )
    
    plt.title('Impact of ΔEffectiveRootSkill on Final AUC', fontsize=16, weight='bold')
    plt.xlabel('ΔEffectiveRootSkill', fontsize=12)
    plt.ylabel('Resulting Final AUC', fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.axvline(0, color='black', linestyle='--', linewidth=1)
    plt.axhline(0.5, color='grey', linestyle=':', linewidth=1)
    plt.ylim(0, 1.05)
    plt.xlim(min(delta_values_plot), max(delta_values_plot))
    plt.legend(title='Initial AUC', loc='best')
    plt.tight_layout()
    plt.savefig('synthetic/support_scripts/delta_effectiverootskill_visualization.png', dpi=300)
    print("Saved plot to synthetic/support_scripts/delta_effectiverootskill_visualization.png")


if __name__ == '__main__':
    main() 