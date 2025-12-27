#!/usr/bin/env python3
"""
Generate publication-quality workflow diagram for saliency-aware neural network training.
Re-designed for clean logic, orthogonal flow, and professional aesthetics.

Usage:
    python synthetic/docu/generate_workflow_diagram.py
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, BoxStyle
import numpy as np

# =============================================================================
# STYLE CONFIGURATION
# =============================================================================

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica'],
    'font.size': 10,
    'axes.linewidth': 0.0,  # Turn off axis border
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'svg.fonttype': 'none', # Keep text as text in SVG
})

# Professional, muted color palette
COLORS = {
    # Phase 1: Cool Blues
    'p1_bg':      '#F0F4F8',
    'p1_border':  '#8EAEC4',
    'p1_fill':    '#D9E2EC',
    'p1_text':    '#102A43',
    'p1_accent':  '#334E68',

    # Phase 2: Warm Oranges/Terracotta
    'p2_bg':      '#FFF5F2',
    'p2_border':  '#E6B8AF',
    'p2_fill':    '#F7D9C4',
    'p2_text':    '#692518',
    'p2_accent':  '#A64835',

    # Components
    'data':       '#699B6A',  # Muted Sage Green
    'model':      '#D66F62',  # Muted Terra-cotta Red
    'ig':         '#FCC419',  # Yellow/Gold
    'metrics':    '#868E96',  # Grey
    'arrow':      '#495057',
    'white':      '#FFFFFF',
}

# =============================================================================
# DRAWING HELPERS
# =============================================================================

def add_box(ax, x, y, w, h, text, facecolor, edgecolor, 
            fontsize=9, weight='normal', textcolor='black', 
            style='round,pad=0.02,rounding_size=0.03', zorder=10, multiline=False):
    """Draw a styled box with text."""
    # Box
    patch = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle=style,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=1.5,
        zorder=zorder,
        clip_on=False
    )
    ax.add_patch(patch)
    
    # Text
    ax.text(x, y, text, ha='center', va='center', 
            fontsize=fontsize, weight=weight, color=textcolor, zorder=zorder+1,
            linespacing=1.3, clip_on=False)
    return patch

def add_connector(ax, start, end, color=COLORS['arrow'], 
                  style='-|>', conn='arc3,rad=0', lw=1.5, ls='-', zorder=5):
    """Draw a connecting arrow."""
    arrow = FancyArrowPatch(
        start, end,
        arrowstyle=style,
        connectionstyle=conn,
        color=color,
        linewidth=lw,
        linestyle=ls,
        shrinkA=0, shrinkB=0,
        mutation_scale=12,
        zorder=zorder,
        clip_on=False
    )
    ax.add_patch(arrow)
    return arrow

# =============================================================================
# MAIN DIAGRAM
# =============================================================================

def create_workflow_diagram():
    fig, ax = plt.subplots(figsize=(12, 8.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Grid Coordinates (for alignment)
    # Top Row: Phase 1 (Left) and Phase 2 (Right)
    # Bottom Row: Detailed Evaluation
    
    x_p1 = 0.25
    x_p2 = 0.75
    
    # -------------------------------------------------------------------------
    # 1. PHASE CONTAINERS (Backgrounds)
    # -------------------------------------------------------------------------
    
    # Phase 1 Background
    p1_bg = FancyBboxPatch((0.02, 0.45), 0.46, 0.50, boxstyle="round,pad=0.02,rounding_size=0.05",
                           facecolor=COLORS['p1_bg'], edgecolor=COLORS['p1_border'], lw=2, zorder=0, clip_on=False)
    ax.add_patch(p1_bg)
    ax.text(0.05, 0.92, 'PHASE 1: Standard HPO', fontsize=12, weight='bold', color=COLORS['p1_accent'], ha='left', clip_on=False)
    
    # Phase 2 Background
    p2_bg = FancyBboxPatch((0.52, 0.45), 0.46, 0.50, boxstyle="round,pad=0.02,rounding_size=0.05",
                           facecolor=COLORS['p2_bg'], edgecolor=COLORS['p2_border'], lw=2, zorder=0, clip_on=False)
    ax.add_patch(p2_bg)
    ax.text(0.55, 0.92, 'PHASE 2: Robust HPO', fontsize=12, weight='bold', color=COLORS['p2_accent'], ha='left', clip_on=False)

    # -------------------------------------------------------------------------
    # 2. PHASE 1 FLOW (Left)
    # -------------------------------------------------------------------------
    
    # P1 Nodes
    # Search Space
    add_box(ax, x_p1, 0.85, 0.22, 0.08, "Search Space\n(Architecture + Optimizer)", 
            COLORS['white'], COLORS['p1_accent'], fontsize=9, weight='bold')
    
    # Training
    add_box(ax, x_p1, 0.70, 0.22, 0.08, "Standard Training\n(Clean Inputs)", 
            COLORS['white'], COLORS['p1_accent'], fontsize=9)
    
    # Objective
    add_box(ax, x_p1, 0.55, 0.22, 0.06, "Metric: Accuracy", 
            COLORS['p1_fill'], COLORS['p1_accent'], fontsize=9, weight='bold')

    # P1 Connections (The Loop)
    add_connector(ax, (x_p1, 0.81), (x_p1, 0.74)) # Space -> Train
    add_connector(ax, (x_p1, 0.66), (x_p1, 0.58)) # Train -> Obj
    
    # Feedback Loop (Optuna)
    # Draw from bottom of Metric, around left side, back to top of Search Space
    add_connector(ax, (x_p1 - 0.11, 0.55), (x_p1 - 0.11, 0.85), 
                  style='-|>', conn="bar,fraction=-0.2", ls='--', color=COLORS['p1_accent'])
    ax.text(x_p1 - 0.17, 0.70, "Optuna TPE", ha='center', va='center', rotation=90, fontsize=8, color=COLORS['p1_accent'], clip_on=False)

    # Output: Best Arch
    # Exit from the loop to the right
    ax.text(0.49, 0.85, "Best\nArchitecture", ha='center', va='center', fontsize=9, weight='bold', color=COLORS['p1_accent'], clip_on=False)
    add_connector(ax, (x_p1 + 0.11, 0.85), (x_p2 - 0.11, 0.85), lw=2, color=COLORS['p1_accent'])

    # -------------------------------------------------------------------------
    # 3. PHASE 2 FLOW (Right)
    # -------------------------------------------------------------------------
    
    # P2 Nodes
    # Search Space (Modified as requested)
    add_box(ax, x_p2, 0.85, 0.22, 0.08, "Search Space\n(Robust Params, Method)", 
            COLORS['white'], COLORS['p2_accent'], fontsize=9, weight='bold')
    
    # Adversarial Training (Stacked style)
    # Draw lower stack layers first
    add_box(ax, x_p2 + 0.015, 0.70 - 0.015, 0.22, 0.08, "", COLORS['white'], COLORS['p2_accent'], zorder=9)
    add_box(ax, x_p2 + 0.008, 0.70 - 0.008, 0.22, 0.08, "", COLORS['white'], COLORS['p2_accent'], zorder=9.5)
    # Main training box
    add_box(ax, x_p2, 0.70, 0.22, 0.08, "Adversarial Training\n(Various Methods)", 
            COLORS['white'], COLORS['p2_accent'], fontsize=9, zorder=10)
    
    # Evaluation Placeholder (Logical Node)
    add_box(ax, x_p2, 0.55, 0.22, 0.06, "Saliency Evaluation", 
            COLORS['p2_fill'], COLORS['p2_accent'], fontsize=9, weight='bold')

    # P2 Connections
    add_connector(ax, (x_p2, 0.81), (x_p2, 0.74)) # Space -> Train
    add_connector(ax, (x_p2, 0.66), (x_p2, 0.58)) # Train -> Eval
    
    # -------------------------------------------------------------------------
    # 4. DETAILED EVALUATION BLOCK (Bottom)
    # -------------------------------------------------------------------------
    
    # Background for Eval
    eval_bg = FancyBboxPatch((0.02, 0.02), 0.96, 0.38, boxstyle="round,pad=0.02,rounding_size=0.05",
                             facecolor='#FAFAFA', edgecolor='#CCCCCC', lw=1.5, ls=':', zorder=0, clip_on=False)
    ax.add_patch(eval_bg)
    ax.text(0.05, 0.37, 'Saliency-Aware Evaluation Process', fontsize=11, weight='bold', color='#555555', ha='left', clip_on=False)

    # Left Column: Model and Data
    add_box(ax, 0.15, 0.28, 0.14, 0.06, "Trained Model", COLORS['model'], 'none', textcolor='white', weight='bold')
    add_box(ax, 0.15, 0.16, 0.14, 0.06, "Test Data", COLORS['data'], 'none', textcolor='white', weight='bold')

    # Middle-Left: Boundary IG (Restricted to PGD + IG)
    big_x = 0.42
    big_bg = FancyBboxPatch((big_x - 0.11, 0.12), 0.22, 0.20, boxstyle="round,pad=0.02,rounding_size=0.03",
                            facecolor='#FFF9DB', edgecolor='#F08C00', lw=1.5, zorder=1, clip_on=False)
    ax.add_patch(big_bg)
    ax.text(big_x, 0.34, "Boundary Integrated Gradients", ha='center', fontsize=9, weight='bold', color='#F08C00', clip_on=False)
    
    # IG Sub-components
    add_box(ax, big_x, 0.26, 0.12, 0.05, "PGD Baseline", COLORS['white'], '#F08C00', fontsize=8)
    add_box(ax, big_x, 0.17, 0.12, 0.05, "Integrated\nGradients", COLORS['white'], '#F08C00', fontsize=8, multiline=True)
    
    # IG Internal Flow
    add_connector(ax, (big_x, 0.235), (big_x, 0.195)) # PGD -> IG

    # Inputs to IG (Model & Data -> PGD)
    # Model -> PGD
    add_connector(ax, (0.22, 0.28), (big_x - 0.06, 0.28), conn="arc3,rad=0.1", lw=1.2)
    # Data -> PGD
    add_connector(ax, (0.22, 0.16), (big_x - 0.06, 0.26), conn="arc3,rad=-0.1", lw=1.2)

    # Middle-Right: Products (Saliency Maps & Ground Truth)
    prod_x = 0.64
    add_box(ax, prod_x, 0.24, 0.14, 0.05, "Saliency Maps", COLORS['white'], '#F08C00', fontsize=9)
    add_box(ax, prod_x, 0.14, 0.14, 0.05, "Ground Truth", COLORS['white'], COLORS['data'], fontsize=9)
    
    # IG -> Maps
    add_connector(ax, (big_x + 0.06, 0.17), (prod_x - 0.07, 0.24), conn="arc3,rad=-0.1", lw=1.2)
    
    # Test Data -> Ground Truth
    add_connector(ax, (0.22, 0.16), (prod_x - 0.07, 0.14), conn="arc3,rad=0", lw=1.2, ls='--')

    # Right Column: Metrics Stack
    mx = 0.85
    # Define metrics with positions and colors
    metrics_conf = [
        # Label, Y-pos, IsHighlighted, HighlightColor
        ("Accuracy", 0.32, True, COLORS['p1_accent']),
        ("Saliency AUC", 0.26, True, COLORS['p2_accent']),
        ("Overlap (wIoU)", 0.20, False, COLORS['metrics']),
        ("SNR (SaSNR)", 0.14, False, COLORS['metrics']),
    ]
    
    for label, y_pos, highlight, color in metrics_conf:
        border = color if highlight else COLORS['metrics']
        txt_col = color if highlight else 'black'
        weight = 'bold' if highlight else 'normal'
        add_box(ax, mx, y_pos, 0.18, 0.045, label, COLORS['white'], border, 
                fontsize=9, weight=weight, textcolor=txt_col)
        
        # Connections to metrics
        if "Accuracy" in label:
            # Model + Data -> Accuracy
            # Draw from Model/Data area, bypassing IG
            add_connector(ax, (0.22, 0.28), (mx - 0.09, y_pos), conn="arc3,rad=-0.15", lw=1, ls=':', color='#999999')
        elif "AUC" in label or "Overlap" in label:
            # Requires Maps AND GT
            add_connector(ax, (prod_x + 0.07, 0.24), (mx - 0.09, y_pos), conn="arc3,rad=0", lw=0.8)
            add_connector(ax, (prod_x + 0.07, 0.14), (mx - 0.09, y_pos), conn="arc3,rad=0", lw=0.8)
        else: # SNR (Just Maps)
            add_connector(ax, (prod_x + 0.07, 0.24), (mx - 0.09, y_pos), conn="arc3,rad=0", lw=0.8)

    # -------------------------------------------------------------------------
    # 5. FINAL FEEDBACK LOOP
    # -------------------------------------------------------------------------
    
    # Conceptual shape: Bracket style merge
    #
    #    Acc ------|
    #              |-----> Up
    #    AUC ------|
    
    feed_x = 0.96
    
    # Top arm (from Accuracy)
    ax.plot([mx + 0.09, feed_x], [0.32, 0.32], color=COLORS['p2_accent'], lw=1.5, ls='--', clip_on=False)
    
    # Bottom arm (from AUC)
    ax.plot([mx + 0.09, feed_x], [0.26, 0.26], color=COLORS['p2_accent'], lw=1.5, ls='--', clip_on=False)
    
    # Vertical connector closing the bracket
    ax.plot([feed_x, feed_x], [0.26, 0.32], color=COLORS['p2_accent'], lw=1.5, ls='--', clip_on=False)
    
    # Midpoint of bracket
    mid_y = (0.26 + 0.32) / 2
    
    # Line going up from midpoint of the vertical connector
    ax.plot([feed_x, feed_x], [mid_y, 0.85], color=COLORS['p2_accent'], lw=1.5, ls='--', clip_on=False)
    
    # Left arrow back to P2 Search Space
    add_connector(ax, (feed_x, 0.85), (x_p2 + 0.11, 0.85), color=COLORS['p2_accent'], ls='--', style='-|>', lw=1.5)
    
    ax.text(feed_x + 0.01, 0.58, "Optuna TPE", rotation=270, va='center', fontsize=9, color=COLORS['p2_accent'], clip_on=False)

    # Link Top P2 Eval Node to Bottom Eval Block
    add_connector(ax, (x_p2, 0.52), (x_p2, 0.41), style='-|>', lw=2, color=COLORS['arrow'])


    plt.tight_layout()
    return fig, ax

def create_cnn_diagram():
    # Improved detailed architecture diagram
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0, 1.1)
    ax.axis('off')
    
    ax.text(0.5, 1.05, "CNN Architecture (Phase 1 Search Space)", ha='center', fontsize=12, weight='bold', color=COLORS['p1_text'], clip_on=False)

    # Re-use colors
    c_data = COLORS['data']
    c_conv = COLORS['model']
    c_pool = '#868E96'
    
    # Layout configuration
    y_center = 0.55
    
    # Layers: (x, width, height, color, label, sublabel)
    layers = [
        # Input
        (0.06, 0.08, 0.60, c_data, "Input", "4xL"),
        # Conv 1
        (0.20, 0.10, 0.55, c_conv, "Conv1", "k1, c1\nBN, Act, Drop"),
        # Pool 1
        (0.32, 0.06, 0.45, c_pool, "Pool1", "stride1"),
        # Conv 2
        (0.44, 0.08, 0.40, c_conv, "Conv2", "k2, c2\nBN, ReLU, Drop"),
        # Pool 2
        (0.54, 0.05, 0.35, c_pool, "Pool2", ""),
        # Conv 3
        (0.64, 0.08, 0.30, c_conv, "Conv3", "k3, c3\nBN, ReLU, Drop"),
        # Global Pool
        (0.76, 0.06, 0.20, c_pool, "Global\nPool", "Avg/Max"),
        # FC
        (0.88, 0.08, 0.15, COLORS['p1_accent'], "FC", "Logits")
    ]
    
    for i, (x, w, h, col, label, sub) in enumerate(layers):
        # Draw Box
        add_box(ax, x, y_center, w, h, "", col, 'none', zorder=2)
        
        # Label (Top or Center)
        ax.text(x, y_center + h/2 + 0.05, label, ha='center', va='bottom', fontsize=9, weight='bold', clip_on=False)
        
        # Sublabel (Inside or Bottom)
        if sub:
            ax.text(x, y_center, sub, ha='center', va='center', fontsize=7, color='white', linespacing=1.1, clip_on=False)
            
        # Connector
        if i < len(layers)-1:
            nx = layers[i+1][0]
            nw = layers[i+1][1]
            start_x = x + w/2 + 0.01
            end_x = nx - nw/2 - 0.01
            add_connector(ax, (start_x, y_center), (end_x, y_center), lw=1.5)

    # Annotations for dimension changes
    ax.annotate('', xy=(0.06, 0.2), xytext=(0.32, 0.2), 
                arrowprops=dict(arrowstyle='<->', color='#666666', lw=1))
    ax.text(0.19, 0.15, "L -> L/s1", ha='center', fontsize=8, color='#666666', clip_on=False)

    plt.tight_layout()
    return fig, ax

if __name__ == '__main__':
    import os
    out_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("Generating workflow diagram...")
    fig1, _ = create_workflow_diagram()
    fig1.savefig(os.path.join(out_dir, 'workflow_diagram.pdf'), bbox_inches='tight', pad_inches=0.05)
    fig1.savefig(os.path.join(out_dir, 'workflow_diagram.png'), dpi=300, bbox_inches='tight', pad_inches=0.05)
    # Remove bbox_inches='tight' for SVG to avoid clipping groups if possible, but standard practice usually needs it for sizing.
    # However, setting fonttype to none and clip_on=False might help editors enough.
    # User specifically asked for suggestion on clipping. Removing bbox_inches='tight' might leave huge margins.
    # We will try WITH tight but relying on clip_on=False for elements.
    fig1.savefig(os.path.join(out_dir, 'workflow_diagram.svg'), bbox_inches='tight', pad_inches=0.05)
    plt.close(fig1)
    
    print("Generating CNN diagram...")
    fig2, _ = create_cnn_diagram()
    fig2.savefig(os.path.join(out_dir, 'cnn_architecture.pdf'), bbox_inches='tight', pad_inches=0.05)
    fig2.savefig(os.path.join(out_dir, 'cnn_architecture.png'), dpi=300, bbox_inches='tight', pad_inches=0.05)
    fig2.savefig(os.path.join(out_dir, 'cnn_architecture.svg'), bbox_inches='tight', pad_inches=0.05)
    plt.close(fig2)
    
    print("Done.")
