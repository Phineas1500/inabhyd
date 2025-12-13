"""
FOL (First-Order Logic) experiment runner for INABHYD replication.
Generates examples in pure FOL format (skipping NL translation).
Supports OpenAI API and OpenAI-compatible endpoints (e.g., vLLM on Modal).
"""

import os
import sys
import pickle
import argparse
import pathlib
from random import seed, shuffle, randint
from functools import reduce
import numpy as np

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Default endpoint settings
DEFAULT_BASE_URL = os.environ.get("OPENAI_BASE_URL", None)
DEFAULT_API_KEY = os.environ.get("OPENAI_API_KEY", "not-needed")

from ontology import Ontology, OntologyConfig, Difficulty
from evaluate import (
    parse_fol_hypotheses_from_response,
    parse_fol_ground_truth,
    compute_fol_strong_accuracy,
    compute_fol_weak_accuracy,
    wilson_confidence_interval
)

SEED = 62471893
NUM_EXAMPLES = 100

# FOL-specific system prompt
FOL_SYSTEM_PROMPT = """You are a logical reasoning system that performs abduction and induction in first-order logic.
Your job is to produce hypotheses in FOL format that explain observations given theories.
Each hypothesis should take one of these forms:
- predicate(constant) for ground atoms (e.g., dalpist(Amy), rainy(Amy))
- ∀x(P(x) → Q(x)) for universal rules (e.g., ∀x(dalpist(x) → rainy(x)))
- Use ¬ for negation (e.g., ¬slow(Amy), ∀x(dalpist(x) → ¬slow(x)))
Output only FOL hypotheses, one per line."""


def make_fol_user_prompt(ontology):
    """Create FOL format user prompt."""
    return f"Theories: {ontology.fol_theories} Observations: {ontology.fol_observations} Produce hypotheses to explain observations."


def make_fol_completion_prompt(system_prompt, user_prompt):
    """Create a completion-style prompt for base models (no chat template)."""
    return f"""{system_prompt}

{user_prompt}

Hypotheses:"""


# Base models that need completions API instead of chat
BASE_MODELS = ['pythia-160m', 'pythia-410m', 'pythia-1b', 'pythia-1.4b', 'pythia-2.8b', 'pythia-6.9b', 'pythia-12b']


def is_base_model(model_name):
    """Check if model is a base model (needs completions API)."""
    return any(base in model_name.lower() for base in BASE_MODELS)


def sanitize_model_name(model_name):
    """Convert model name to safe filename string."""
    return model_name.replace("-", "").replace("/", "_").replace(":", "_").lower()


def generate_single_example(config):
    """Generate a single reasoning example."""
    while True:
        try:
            ontology = Ontology(config)
            return ontology
        except Exception:
            pass


def get_openai_client(base_url=None):
    """Create OpenAI client with optional custom base URL for vLLM/Modal."""
    from openai import OpenAI

    if base_url or DEFAULT_BASE_URL:
        return OpenAI(
            base_url=base_url or DEFAULT_BASE_URL,
            api_key=DEFAULT_API_KEY,
        )
    return OpenAI()


def run_fol_zero_shot_single_hypothesis(model_name, task_type, height, num_examples=NUM_EXAMPLES, base_url=None):
    """
    Run zero-shot single hypothesis experiment with FOL format.
    task_type: 'property', 'membership', or 'ontology'
    """
    client = get_openai_client(base_url)

    # Seed both Python random and numpy random for reproducibility
    seed(SEED)
    np.random.seed(SEED)

    # Configure based on task type
    config = OntologyConfig(
        hops=height,
        recover_membership=(task_type == 'membership'),
        recover_ontology=(task_type == 'ontology'),
        recover_property=(task_type == 'property'),
        difficulty=Difficulty.SINGLE,
        mix_hops=False
    )

    print(f"\n{'='*60}")
    print(f"FOL Zero-shot Single Hypothesis: {task_type}, Height {height}")
    print(f"{'='*60}")

    examples = []
    replies = []
    results = []

    for i in range(num_examples):
        if i % 10 == 0:
            print(f"  Processing example {i+1}/{num_examples}...")

        ontology = generate_single_example(config)
        examples.append(ontology)

        user_prompt = make_fol_user_prompt(ontology)

        try:
            if is_base_model(model_name):
                # Use completions API for base models
                prompt = make_fol_completion_prompt(FOL_SYSTEM_PROMPT, user_prompt)
                completion = client.completions.create(
                    model=model_name,
                    prompt=prompt,
                    temperature=0,
                    max_tokens=256
                )
                reply = completion.choices[0].text.strip()
            else:
                # Use chat completions API for instruction-tuned models
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": FOL_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0
                )
                reply = completion.choices[0].message.content
        except Exception as e:
            print(f"  Error on example {i+1}: {e}")
            reply = ""

        replies.append(reply)

        # Evaluate using FOL-specific parsing
        pred_hyps = parse_fol_hypotheses_from_response(reply)
        gt_hyps = parse_fol_ground_truth(ontology.fol_hypotheses)

        strong_acc = compute_fol_strong_accuracy(pred_hyps, gt_hyps)
        weak_acc = compute_fol_weak_accuracy(pred_hyps, gt_hyps, ontology.fol_observations, ontology.fol_theories)

        if strong_acc == 1:
            weak_acc = 1

        results.append({
            'strong': strong_acc,
            'weak': weak_acc,
        })

    # Compute aggregate metrics
    strong_mean = np.mean([r['strong'] for r in results])
    weak_mean = np.mean([r['weak'] for r in results])

    strong_ci = wilson_confidence_interval(strong_mean, num_examples)
    weak_ci = wilson_confidence_interval(weak_mean, num_examples)

    print(f"\n  Results:")
    print(f"    Strong Accuracy: {strong_mean:.3f} [{strong_ci[0]:.3f}, {strong_ci[1]:.3f}]")
    print(f"    Weak Accuracy: {weak_mean:.3f} [{weak_ci[0]:.3f}, {weak_ci[1]:.3f}]")

    # Save results
    output_dir = pathlib.Path("results")
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / f"fol_zeroshot_single_{task_type}_h{height}_{sanitize_model_name(model_name)}.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump({
            'examples': examples,
            'replies': replies,
            'results': results,
            'metrics': {
                'strong_accuracy': strong_mean,
                'weak_accuracy': weak_mean,
                'strong_ci': strong_ci,
                'weak_ci': weak_ci
            }
        }, f)

    print(f"  Saved to {output_file}")

    return {
        'strong_accuracy': strong_mean,
        'weak_accuracy': weak_mean,
    }


def run_fol_zero_shot_multiple_hypothesis(model_name, height, num_examples=NUM_EXAMPLES, base_url=None):
    """Run zero-shot multiple hypothesis experiment with FOL format."""
    client = get_openai_client(base_url)

    # Seed both Python random and numpy random for reproducibility
    seed(SEED)
    np.random.seed(SEED)

    config = OntologyConfig(
        hops=height,
        recover_membership=True,
        recover_ontology=True,
        recover_property=True,
        difficulty=Difficulty.EASY,
        mix_hops=True
    )

    print(f"\n{'='*60}")
    print(f"FOL Zero-shot Multiple Hypotheses: Height {height}")
    print(f"{'='*60}")

    examples = []
    replies = []
    results = []

    for i in range(num_examples):
        if i % 10 == 0:
            print(f"  Processing example {i+1}/{num_examples}...")

        ontology = generate_single_example(config)
        examples.append(ontology)

        user_prompt = make_fol_user_prompt(ontology)

        try:
            if is_base_model(model_name):
                # Use completions API for base models
                prompt = make_fol_completion_prompt(FOL_SYSTEM_PROMPT, user_prompt)
                completion = client.completions.create(
                    model=model_name,
                    prompt=prompt,
                    temperature=0,
                    max_tokens=512
                )
                reply = completion.choices[0].text.strip()
            else:
                # Use chat completions API for instruction-tuned models
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": FOL_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0
                )
                reply = completion.choices[0].message.content
        except Exception as e:
            print(f"  Error on example {i+1}: {e}")
            reply = ""

        replies.append(reply)

        # Evaluate using FOL-specific parsing
        pred_hyps = parse_fol_hypotheses_from_response(reply)
        gt_hyps = parse_fol_ground_truth(ontology.fol_hypotheses)

        strong_acc = compute_fol_strong_accuracy(pred_hyps, gt_hyps)
        weak_acc = compute_fol_weak_accuracy(pred_hyps, gt_hyps, ontology.fol_observations, ontology.fol_theories)

        if strong_acc == 1:
            weak_acc = 1

        results.append({
            'strong': strong_acc,
            'weak': weak_acc,
        })

    # Compute aggregate metrics
    strong_mean = np.mean([r['strong'] for r in results])
    weak_mean = np.mean([r['weak'] for r in results])

    strong_ci = wilson_confidence_interval(strong_mean, num_examples)
    weak_ci = wilson_confidence_interval(weak_mean, num_examples)

    print(f"\n  Results:")
    print(f"    Strong Accuracy: {strong_mean:.3f} [{strong_ci[0]:.3f}, {strong_ci[1]:.3f}]")
    print(f"    Weak Accuracy: {weak_mean:.3f} [{weak_ci[0]:.3f}, {weak_ci[1]:.3f}]")

    # Save results
    output_dir = pathlib.Path("results")
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / f"fol_zeroshot_multi_h{height}_{sanitize_model_name(model_name)}.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump({
            'examples': examples,
            'replies': replies,
            'results': results,
            'metrics': {
                'strong_accuracy': strong_mean,
                'weak_accuracy': weak_mean,
                'strong_ci': strong_ci,
                'weak_ci': weak_ci
            }
        }, f)

    print(f"  Saved to {output_file}")

    return {
        'strong_accuracy': strong_mean,
        'weak_accuracy': weak_mean,
    }


def main():
    parser = argparse.ArgumentParser(description='Run INABHYD FOL experiments')
    parser.add_argument('--model', '-m', type=str, default='gpt-4o',
                        help='Model to use (default: gpt-4o, or gemma3-27b for Modal)')
    parser.add_argument('--base-url', '-b', type=str, default=None,
                        help='Base URL for OpenAI-compatible API (e.g., Modal vLLM endpoint)')
    parser.add_argument('--num-examples', '-n', type=int, default=100,
                        help='Number of examples per configuration (default: 100)')
    parser.add_argument('--experiment', '-e', type=str, default='zeroshot_single',
                        choices=['zeroshot_single', 'zeroshot_multi'],
                        help='Which experiment to run')
    parser.add_argument('--task', '-t', type=str, default='all',
                        choices=['all', 'property', 'membership', 'ontology'],
                        help='Task type for single hypothesis experiments')
    parser.add_argument('--height', type=int, default=0,
                        choices=[0, 1, 2, 3, 4],
                        help='Ontology tree height (0 for all heights)')
    args = parser.parse_args()

    if args.experiment == 'zeroshot_single':
        tasks = ['property', 'membership', 'ontology'] if args.task == 'all' else [args.task]
        heights = range(1, 5) if args.height == 0 else [args.height]
        for task in tasks:
            for height in heights:
                run_fol_zero_shot_single_hypothesis(args.model, task, height, args.num_examples, args.base_url)
    elif args.experiment == 'zeroshot_multi':
        heights = range(1, 5) if args.height == 0 else [args.height]
        for height in heights:
            run_fol_zero_shot_multiple_hypothesis(args.model, height, args.num_examples, args.base_url)


if __name__ == '__main__':
    main()
