"""
Extract metrics from zero-shot multiple hypothesis log file and save to JSON.
"""

import json
import re
from pathlib import Path


def parse_multi_log_file(filepath):
    """Parse the zero-shot multiple hypothesis log file."""
    results = {}

    with open(filepath, 'r') as f:
        content = f.read()

    # Pattern to match each experiment block
    block_pattern = r'Zero-shot Multiple Hypotheses: Height (\d+).*?Strong Accuracy: ([\d.]+) \[([\d.]+), ([\d.]+)\].*?Weak Accuracy: ([\d.]+) \[([\d.]+), ([\d.]+)\].*?Quality: ([\d.]+)'

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
    log_file = results_dir / 'log_zeroshot_multi.txt'

    if log_file.exists():
        print(f"Parsing {log_file}...")
        metrics = parse_multi_log_file(log_file)
        print(f"  Found heights: {list(metrics.keys())}")
    else:
        print(f"Log file not found: {log_file}")
        return

    # Save to JSON
    output_file = results_dir / 'metrics_zeroshot_multi_gpt4o.json'
    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved metrics to {output_file}")

    # Print summary table
    print("\n" + "="*60)
    print("ZERO-SHOT MULTIPLE HYPOTHESES RESULTS (GPT-4o)")
    print("="*60)

    heights = [1, 2, 3, 4]

    print(f"\n{'Height':<10} {'Weak Acc':>12} {'Strong Acc':>12} {'Quality':>12}")
    print("-" * 50)
    for h in heights:
        key = f'h{h}'
        if key in metrics:
            print(f"H{h:<9} {metrics[key]['weak_accuracy']:>12.3f} {metrics[key]['strong_accuracy']:>12.3f} {metrics[key]['quality']:>12.3f}")


if __name__ == '__main__':
    main()
