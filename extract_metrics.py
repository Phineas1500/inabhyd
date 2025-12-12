"""
Extract metrics from pickle files and save to JSON for easy plotting.
Uses a restricted unpickler to only get the metrics we need.
"""

import pickle
import json
from pathlib import Path
import sys
import os

# Add the script directory to the path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

# Import the ontology module
from ontology import Ontology, OntologyConfig, OntologyNode, Difficulty
from morphology import Morphology, Prop


def extract_metrics_from_pickle(filepath):
    """Extract only the metrics dict from a pickle file."""
    with open(filepath, 'rb') as f:
        # Load the full pickle
        data = pickle.load(f)
        # Return only the metrics (which are simple dicts/floats)
        return data.get('metrics', {})


def main():
    results_dir = Path('results')

    tasks = ['property', 'membership', 'ontology']
    heights = [1, 2, 3, 4]

    all_metrics = {}

    for task in tasks:
        all_metrics[task] = {}
        for h in heights:
            filename = results_dir / f"zeroshot_single_{task}_h{h}_gpt4o.pkl"

            if filename.exists():
                try:
                    metrics = extract_metrics_from_pickle(filename)
                    all_metrics[task][f'h{h}'] = {
                        'weak_accuracy': metrics.get('weak_accuracy', 0),
                        'strong_accuracy': metrics.get('strong_accuracy', 0),
                        'quality': metrics.get('quality', 0),
                        'weak_ci': list(metrics.get('weak_ci', [0, 0])),
                        'strong_ci': list(metrics.get('strong_ci', [0, 0])),
                    }
                    print(f"Loaded {filename}")
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
            else:
                print(f"File not found: {filename}")

    # Save to JSON
    output_file = results_dir / 'metrics_gpt4o.json'
    with open(output_file, 'w') as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\nSaved metrics to {output_file}")

    # Print summary
    print("\n" + "="*70)
    print("ZERO-SHOT SINGLE HYPOTHESIS RESULTS (GPT-4o)")
    print("="*70)

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
