"""
Extract metrics from log files and save to JSON for plotting.
"""

import json
import re
from pathlib import Path


def parse_log_file(filepath):
    """Parse a log file and extract metrics for each height."""
    results = {}

    with open(filepath, 'r') as f:
        content = f.read()

    # Pattern to match each experiment block
    block_pattern = r'Zero-shot Single Hypothesis: \w+, Height (\d+).*?Strong Accuracy: ([\d.]+) \[([\d.]+), ([\d.]+)\].*?Weak Accuracy: ([\d.]+) \[([\d.]+), ([\d.]+)\].*?Quality: ([\d.]+)'

    matches = re.findall(block_pattern, content, re.DOTALL)

    for match in matches:
        height = int(match[0])
        results[f'h{height}'] = {
            'strong_accuracy': float(match[1]),
            'strong_ci': [float(match[2]), float(match[3])],
            'weak_accuracy': float(match[4]),
            'weak_ci': [float(match[5]), float(match[6])],
            'quality': float(match[7])
        }

    return results


def main():
    results_dir = Path('results')

    tasks = ['property', 'membership', 'ontology']

    all_metrics = {}

    for task in tasks:
        log_file = results_dir / f'log_zeroshot_single_{task}.txt'

        if log_file.exists():
            print(f"Parsing {log_file}...")
            all_metrics[task] = parse_log_file(log_file)
            print(f"  Found heights: {list(all_metrics[task].keys())}")
        else:
            print(f"Log file not found: {log_file}")
            all_metrics[task] = {}

    # Save to JSON
    output_file = results_dir / 'metrics_gpt4o.json'
    with open(output_file, 'w') as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\nSaved metrics to {output_file}")

    # Print summary table
    print("\n" + "="*70)
    print("ZERO-SHOT SINGLE HYPOTHESIS RESULTS (GPT-4o)")
    print("="*70)

    heights = [1, 2, 3, 4]

    print("\n--- Weak Accuracy ---")
    print(f"{'Task':<20} {'H1':>10} {'H2':>10} {'H3':>10} {'H4':>10}")
    print("-" * 60)
    for task in tasks:
        row = f"{task:<20}"
        for h in heights:
            key = f'h{h}'
            if key in all_metrics[task]:
                val = all_metrics[task][key]['weak_accuracy']
                row += f" {val:>9.3f}"
            else:
                row += f" {'N/A':>9}"
        print(row)

    print("\n--- Strong Accuracy ---")
    print(f"{'Task':<20} {'H1':>10} {'H2':>10} {'H3':>10} {'H4':>10}")
    print("-" * 60)
    for task in tasks:
        row = f"{task:<20}"
        for h in heights:
            key = f'h{h}'
            if key in all_metrics[task]:
                val = all_metrics[task][key]['strong_accuracy']
                row += f" {val:>9.3f}"
            else:
                row += f" {'N/A':>9}"
        print(row)

    print("\n--- Quality ---")
    print(f"{'Task':<20} {'H1':>10} {'H2':>10} {'H3':>10} {'H4':>10}")
    print("-" * 60)
    for task in tasks:
        row = f"{task:<20}"
        for h in heights:
            key = f'h{h}'
            if key in all_metrics[task]:
                val = all_metrics[task][key]['quality']
                row += f" {val:>9.3f}"
            else:
                row += f" {'N/A':>9}"
        print(row)


if __name__ == '__main__':
    main()
