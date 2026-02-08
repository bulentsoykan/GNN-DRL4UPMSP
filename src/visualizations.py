"""
Visualization functions for UPMSP results

Creates figures similar to those in the paper:
- Performance comparison charts
- Pareto front visualization
- Training curves
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List
import os


# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11


def plot_performance_comparison(results: Dict[str, Dict], save_path: str = None):
    """
    Create Figure 2 from the paper: Performance comparison across problem sizes.

    Args:
        results: Dictionary with structure {method_name: {size: metrics}}
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Extract data
    methods = list(results.keys())
    sizes = list(next(iter(results.values())).keys())

    # Prepare data for plotting
    data_twt = {method: [] for method in methods}
    data_tst = {method: [] for method in methods}
    data_time = {method: [] for method in methods}

    for method in methods:
        for size in sizes:
            metrics = results[method][size]
            data_twt[method].append(metrics['avg_twt'])
            data_tst[method].append(metrics['avg_tst'])
            data_time[method].append(metrics['avg_comp_time'])

    x_labels = sizes
    x_pos = np.arange(len(x_labels))
    width = 0.25

    # Plot 1: Average TWT
    ax1 = axes[0, 0]
    for i, method in enumerate(methods):
        ax1.bar(x_pos + i * width, data_twt[method], width, label=method)
    ax1.set_xlabel('Problem Size (n jobs, m machines)')
    ax1.set_ylabel('Average Total Weighted Tardiness (TWT)')
    ax1.set_title('Average Total Weighted Tardiness (TWT)')
    ax1.set_xticks(x_pos + width)
    ax1.set_xticklabels(x_labels)
    ax1.legend()

    # Plot 2: Average TST
    ax2 = axes[0, 1]
    for i, method in enumerate(methods):
        ax2.bar(x_pos + i * width, data_tst[method], width, label=method)
    ax2.set_xlabel('Problem Size (n jobs, m machines)')
    ax2.set_ylabel('Average Total Setup Time (TST)')
    ax2.set_title('Average Total Setup Time (TST)')
    ax2.set_xticks(x_pos + width)
    ax2.set_xticklabels(x_labels)
    ax2.legend()

    # Plot 3: Computation Time (log scale)
    ax3 = axes[1, 0]
    for i, method in enumerate(methods):
        ax3.bar(x_pos + i * width, data_time[method], width, label=method)
    ax3.set_xlabel('Problem Size (n jobs, m machines)')
    ax3.set_ylabel('Average Computation Time (s)')
    ax3.set_title('Average Computation Time (Evaluation)')
    ax3.set_yscale('log')
    ax3.set_xticks(x_pos + width)
    ax3.set_xticklabels(x_labels)
    ax3.legend()

    # Plot 4: Multi-Objective Trade-off (Pareto front)
    ax4 = axes[1, 1]

    # Average across all sizes for each method
    avg_tst = {method: np.mean(data_tst[method]) for method in methods}
    avg_twt = {method: np.mean(data_twt[method]) for method in methods}

    colors = {'ATCSR_Rm': 'blue', 'GA': 'green', 'PPO-GNN': 'red'}
    for method in methods:
        ax4.scatter(
            avg_tst[method], avg_twt[method],
            s=200, label=method,
            color=colors.get(method, 'gray'),
            marker='o', alpha=0.7, edgecolors='black', linewidth=2
        )

    ax4.set_xlabel('Average Total Setup Time (TST)')
    ax4.set_ylabel('Average Total Weighted Tardiness (TWT)')
    ax4.set_title('Multi-Objective Trade-off (Avg TST vs Avg TWT - Averaged Across Sizes)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    return fig


def create_results_table(results: Dict[str, Dict]) -> pd.DataFrame:
    """
    Create Table 1 from the paper: Performance comparison table.

    Args:
        results: Dictionary with structure {method_name: {size: metrics}}

    Returns:
        DataFrame with results
    """
    rows = []

    for method in results:
        for size in results[method]:
            metrics = results[method][size]
            rows.append({
                'Size': size,
                'Method': method,
                'Avg TWT': f"{metrics['avg_twt']:.1f}",
                'Avg TST': f"{metrics['avg_tst']:.1f}",
                'Avg Comp Time (s)': f"{metrics['avg_comp_time']:.2f}",
            })

    df = pd.DataFrame(rows)
    return df


def plot_training_curves(training_history: Dict, save_path: str = None):
    """
    Plot training curves for PPO agent.

    Args:
        training_history: Dictionary with training metrics over episodes
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Episode rewards
    ax1 = axes[0, 0]
    ax1.plot(training_history['episode_rewards'], alpha=0.6)
    ax1.plot(
        pd.Series(training_history['episode_rewards']).rolling(window=50).mean(),
        linewidth=2, label='Moving Average (50 episodes)'
    )
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('Episode Rewards')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # TWT over episodes
    ax2 = axes[0, 1]
    ax2.plot(training_history['episode_twt'], alpha=0.6)
    ax2.plot(
        pd.Series(training_history['episode_twt']).rolling(window=50).mean(),
        linewidth=2, label='Moving Average (50 episodes)'
    )
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Total Weighted Tardiness')
    ax2.set_title('Total Weighted Tardiness per Episode')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # TST over episodes
    ax3 = axes[1, 0]
    ax3.plot(training_history['episode_tst'], alpha=0.6)
    ax3.plot(
        pd.Series(training_history['episode_tst']).rolling(window=50).mean(),
        linewidth=2, label='Moving Average (50 episodes)'
    )
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Total Setup Time')
    ax3.set_title('Total Setup Time per Episode')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Policy loss
    if 'policy_loss' in training_history:
        ax4 = axes[1, 1]
        ax4.plot(training_history['policy_loss'], alpha=0.6)
        ax4.plot(
            pd.Series(training_history['policy_loss']).rolling(window=50).mean(),
            linewidth=2, label='Moving Average (50 updates)'
        )
        ax4.set_xlabel('Update Step')
        ax4.set_ylabel('Policy Loss')
        ax4.set_title('Policy Loss')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")

    return fig


def plot_pareto_front(results: List[Tuple[float, float]], labels: List[str], save_path: str = None):
    """
    Plot Pareto front for multi-objective optimization.

    Args:
        results: List of (TST, TWT) tuples
        labels: List of method labels
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = plt.cm.Set1(np.linspace(0, 1, len(set(labels))))
    color_map = {label: colors[i] for i, label in enumerate(set(labels))}

    for (tst, twt), label in zip(results, labels):
        ax.scatter(tst, twt, s=200, label=label, color=color_map[label],
                  marker='o', alpha=0.7, edgecolors='black', linewidth=2)

    ax.set_xlabel('Total Setup Time (TST)', fontsize=14)
    ax.set_ylabel('Total Weighted Tardiness (TWT)', fontsize=14)
    ax.set_title('Pareto Front: Multi-Objective Trade-off', fontsize=16)
    ax.grid(True, alpha=0.3)

    # Remove duplicate labels
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Pareto front saved to {save_path}")

    return fig


def save_all_visualizations(results: Dict, training_history: Dict = None, output_dir: str = 'results/figures'):
    """
    Generate and save all visualizations.

    Args:
        results: Results dictionary
        training_history: Training history dictionary (optional)
        output_dir: Directory to save figures
    """
    os.makedirs(output_dir, exist_ok=True)

    # Performance comparison
    plot_performance_comparison(results, os.path.join(output_dir, 'performance_comparison.png'))

    # Training curves
    if training_history:
        plot_training_curves(training_history, os.path.join(output_dir, 'training_curves.png'))

    # Results table
    df = create_results_table(results)
    df.to_csv(os.path.join(output_dir, 'results_table.csv'), index=False)
    print(f"Results table saved to {os.path.join(output_dir, 'results_table.csv')}")

    print(f"\nAll visualizations saved to {output_dir}/")
