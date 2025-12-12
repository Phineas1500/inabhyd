"""
Plot zero-shot multiple hypothesis results similar to paper style.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Use the same style as the paper (clean, academic style)
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.titlesize': 12,
    'axes.spines.top': False,
    'axes.spines.right': False,
})


def load_results_from_json(json_file):
    """Load results from JSON file."""
    with open(json_file, 'r') as f:
        all_metrics = json.load(f)

    heights = [1, 2, 3, 4]

    data = {
        'heights': [],
        'weak_accuracy': [],
        'strong_accuracy': [],
        'quality': [],
        'weak_ci_low': [],
        'weak_ci_high': [],
        'strong_ci_low': [],
        'strong_ci_high': [],
    }

    for h in heights:
        key = f'h{h}'
        if key in all_metrics:
            metrics = all_metrics[key]
            data['heights'].append(h)
            data['weak_accuracy'].append(metrics['weak_accuracy'])
            data['strong_accuracy'].append(metrics['strong_accuracy'])
            data['quality'].append(metrics['quality'])
            data['weak_ci_low'].append(metrics['weak_ci'][0])
            data['weak_ci_high'].append(metrics['weak_ci'][1])
            data['strong_ci_low'].append(metrics['strong_ci'][0])
            data['strong_ci_high'].append(metrics['strong_ci'][1])

    return data


def plot_multiple_hypothesis_results(data, output_file='results/figure_multi_gpt4o.pdf'):
    """
    Create a figure for zero-shot multiple hypothesis results.
    1 row × 3 columns (weak accuracy, strong accuracy, quality)
    """

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))

    metrics = ['weak_accuracy', 'strong_accuracy', 'quality']
    metric_labels = ['Weak Accuracy', 'Strong Accuracy', 'Quality']

    # Color for GPT-4o (blue, matching paper style)
    color = '#1f77b4'

    heights = data['heights']

    for col, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[col]

        values = data[metric]

        if len(heights) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            continue

        # Plot line with markers
        ax.plot(heights, values, 'o-', color=color, linewidth=2,
                markersize=8, label='GPT-4o')

        # Add error bars for accuracy metrics
        if metric == 'weak_accuracy' and len(data['weak_ci_low']) > 0:
            ci_low = data['weak_ci_low']
            ci_high = data['weak_ci_high']
            errors = [np.array(values) - np.array(ci_low),
                     np.array(ci_high) - np.array(values)]
            ax.errorbar(heights, values, yerr=errors, fmt='none',
                       color=color, capsize=3, alpha=0.7)
        elif metric == 'strong_accuracy' and len(data['strong_ci_low']) > 0:
            ci_low = data['strong_ci_low']
            ci_high = data['strong_ci_high']
            errors = [np.array(values) - np.array(ci_low),
                     np.array(ci_high) - np.array(values)]
            ax.errorbar(heights, values, yerr=errors, fmt='none',
                       color=color, capsize=3, alpha=0.7)

        # Formatting
        ax.set_xlim(0.5, 4.5)
        ax.set_ylim(0, 1.05)
        ax.set_xticks([1, 2, 3, 4])
        ax.set_yticks([0, 0.25, 0.50, 0.75, 1.00])

        # Labels
        ax.set_xlabel('Height')
        ax.set_title(label)
        if col == 0:
            ax.set_ylabel('Zero-shot Multiple Hypotheses')

        # Grid
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)

    # Add legend at the top
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=1,
               bbox_to_anchor=(0.5, 1.05), frameon=False)

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)

    # Save figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.savefig(output_file.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"Saved figure to {output_file} and {output_file.replace('.pdf', '.png')}")

    plt.close()


def main():
    results_dir = Path('results')
    json_file = results_dir / 'metrics_zeroshot_multi_gpt4o.json'

    # Load data
    print("Loading results from JSON...")
    data = load_results_from_json(json_file)

    # Print table
    print("\n" + "="*60)
    print("ZERO-SHOT MULTIPLE HYPOTHESES RESULTS (GPT-4o)")
    print("="*60)
    print(f"\n{'Height':<10} {'Weak Acc':>12} {'Strong Acc':>12} {'Quality':>12}")
    print("-" * 50)
    for i, h in enumerate(data['heights']):
        print(f"H{h:<9} {data['weak_accuracy'][i]:>12.3f} {data['strong_accuracy'][i]:>12.3f} {data['quality'][i]:>12.3f}")

    # Plot
    print("\nGenerating plot...")
    plot_multiple_hypothesis_results(data, 'results/figure_multi_gpt4o.pdf')


if __name__ == '__main__':
    main()
