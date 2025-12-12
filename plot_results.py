"""
Plot zero-shot single hypothesis results similar to Figure 2 in the paper.
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

    tasks = ['property', 'membership', 'ontology']
    heights = [1, 2, 3, 4]

    data = {}

    for task in tasks:
        data[task] = {
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
            if key in all_metrics.get(task, {}):
                metrics = all_metrics[task][key]
                data[task]['heights'].append(h)
                data[task]['weak_accuracy'].append(metrics['weak_accuracy'])
                data[task]['strong_accuracy'].append(metrics['strong_accuracy'])
                data[task]['quality'].append(metrics['quality'])
                data[task]['weak_ci_low'].append(metrics['weak_ci'][0])
                data[task]['weak_ci_high'].append(metrics['weak_ci'][1])
                data[task]['strong_ci_low'].append(metrics['strong_ci'][0])
                data[task]['strong_ci_high'].append(metrics['strong_ci'][1])

    return data


def plot_single_hypothesis_results(data, output_file='figure2_gpt4o.pdf'):
    """
    Create a figure similar to Figure 2 in the paper.
    3 rows (tasks) × 3 columns (weak accuracy, strong accuracy, quality)
    """

    fig, axes = plt.subplots(3, 3, figsize=(10, 8))

    tasks = ['property', 'membership', 'ontology']
    task_labels = {
        'property': 'Task: infer property',
        'membership': 'Task: infer membership relation',
        'ontology': 'Task: infer subtype relation'
    }

    metrics = ['weak_accuracy', 'strong_accuracy', 'quality']
    metric_labels = ['Weak Accuracy', 'Strong Accuracy', 'Quality']

    # Color for GPT-4o (blue, matching paper style)
    color = '#1f77b4'

    for row, task in enumerate(tasks):
        for col, (metric, label) in enumerate(zip(metrics, metric_labels)):
            ax = axes[row, col]

            heights = data[task]['heights']
            values = data[task][metric]

            if len(heights) == 0:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                continue

            # Plot line with markers
            ax.plot(heights, values, 'o-', color=color, linewidth=2,
                    markersize=8, label='GPT-4o')

            # Add error bars for accuracy metrics
            if metric == 'weak_accuracy' and len(data[task]['weak_ci_low']) > 0:
                ci_low = data[task]['weak_ci_low']
                ci_high = data[task]['weak_ci_high']
                errors = [np.array(values) - np.array(ci_low),
                         np.array(ci_high) - np.array(values)]
                ax.errorbar(heights, values, yerr=errors, fmt='none',
                           color=color, capsize=3, alpha=0.7)
            elif metric == 'strong_accuracy' and len(data[task]['strong_ci_low']) > 0:
                ci_low = data[task]['strong_ci_low']
                ci_high = data[task]['strong_ci_high']
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
            if row == 2:  # Bottom row
                ax.set_xlabel('Height')
            if col == 0:  # Left column
                ax.set_ylabel(task_labels[task], fontsize=10)
            if row == 0:  # Top row
                ax.set_title(label)

            # Grid
            ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            ax.set_axisbelow(True)

    # Add legend at the top
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=1,
               bbox_to_anchor=(0.5, 1.02), frameon=False)

    plt.tight_layout()
    plt.subplots_adjust(top=0.93)

    # Save figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.savefig(output_file.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"Saved figure to {output_file} and {output_file.replace('.pdf', '.png')}")

    plt.close()


def print_results_table(data):
    """Print results in a formatted table."""

    print("\n" + "="*70)
    print("ZERO-SHOT SINGLE HYPOTHESIS RESULTS (GPT-4o)")
    print("="*70)

    tasks = ['property', 'membership', 'ontology']
    task_names = {
        'property': 'Infer Property',
        'membership': 'Infer Membership',
        'ontology': 'Infer Subtype'
    }

    # Weak Accuracy Table
    print("\n--- Weak Accuracy ---")
    print(f"{'Task':<20} {'H1':>10} {'H2':>10} {'H3':>10} {'H4':>10}")
    print("-" * 60)
    for task in tasks:
        row = f"{task_names[task]:<20}"
        for i, h in enumerate(data[task]['heights']):
            val = data[task]['weak_accuracy'][i]
            row += f" {val:>9.3f}"
        print(row)

    # Strong Accuracy Table
    print("\n--- Strong Accuracy ---")
    print(f"{'Task':<20} {'H1':>10} {'H2':>10} {'H3':>10} {'H4':>10}")
    print("-" * 60)
    for task in tasks:
        row = f"{task_names[task]:<20}"
        for i, h in enumerate(data[task]['heights']):
            val = data[task]['strong_accuracy'][i]
            row += f" {val:>9.3f}"
        print(row)

    # Quality Table
    print("\n--- Quality ---")
    print(f"{'Task':<20} {'H1':>10} {'H2':>10} {'H3':>10} {'H4':>10}")
    print("-" * 60)
    for task in tasks:
        row = f"{task_names[task]:<20}"
        for i, h in enumerate(data[task]['heights']):
            val = data[task]['quality'][i]
            row += f" {val:>9.3f}"
        print(row)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Plot INABHYD results')
    parser.add_argument('--results-dir', '-r', type=str, default='results',
                        help='Directory containing result files')
    parser.add_argument('--output', '-o', type=str, default='results/figure2_gpt4o.pdf',
                        help='Output file name')
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    json_file = results_dir / 'metrics_gpt4o.json'

    # Load data
    print("Loading results from JSON...")
    data = load_results_from_json(json_file)

    # Print table
    print_results_table(data)

    # Plot
    print("\nGenerating plot...")
    plot_single_hypothesis_results(data, args.output)


if __name__ == '__main__':
    main()
