"""
Fast NL experiment runner with parallel async requests.
Processes multiple examples concurrently for much faster throughput.
"""

import os
import sys
import pickle
import argparse
import pathlib
import asyncio
from random import seed
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DEFAULT_BASE_URL = os.environ.get("OPENAI_BASE_URL", None)
DEFAULT_API_KEY = os.environ.get("OPENAI_API_KEY", "not-needed")

from ontology import Ontology, OntologyConfig, Difficulty
from evaluate import (
    parse_hypotheses_from_response,
    parse_ground_truth,
    compute_strong_accuracy,
    compute_weak_accuracy,
    compute_quality,
    wilson_confidence_interval
)

SEED = 62471893
NUM_EXAMPLES = 100
CONCURRENT_REQUESTS = 20
NO_SYSTEM_PROMPT = False  # For models that don't support system role (e.g., Gemma 2)

SYSTEM_PROMPT = """You are a helpful assitant that performs abduction and induction reasoning.
        Your job is to come up with hypotheses that explain observations with given theories. Each hypothesis should explain as many observations as possible.
        You can come up with multiple hypotheses and each hypothesis should take one line with the format A is B or A is not B.
    .   Only output final hypotheses.
 """


def make_user_prompt(ontology):
    return "Q: " + ontology.theories + " We observe that: " + ontology.observations + " Please come up with hypothesis to explain observations."


def sanitize_model_name(model_name):
    return model_name.replace("-", "").replace("/", "_").replace(":", "_").lower()


def generate_single_example(config):
    while True:
        try:
            ontology = Ontology(config)
            return ontology
        except Exception:
            pass


async def process_single_example(client, model_name, ontology, semaphore):
    """Process a single example with rate limiting via semaphore."""
    async with semaphore:
        user_prompt = make_user_prompt(ontology)

        # Build messages based on whether model supports system role
        if NO_SYSTEM_PROMPT:
            # Merge system prompt into user message for models like Gemma 2
            combined_prompt = f"{SYSTEM_PROMPT}\n\n{user_prompt}"
            messages = [{"role": "user", "content": combined_prompt}]
        else:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ]

        try:
            completion = await client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0,
                max_tokens=3800
            )
            reply = completion.choices[0].message.content
        except Exception as e:
            print(f"  Error: {e}")
            reply = ""

        # Evaluate
        pred_hyps = parse_hypotheses_from_response(reply)
        gt_hyps = parse_ground_truth(ontology.hypotheses)

        strong_acc = compute_strong_accuracy(pred_hyps, gt_hyps)
        weak_acc = compute_weak_accuracy(pred_hyps, gt_hyps, ontology.observations, ontology.theories)
        quality = compute_quality(pred_hyps, gt_hyps, ontology.observations, ontology.theories)

        if strong_acc == 1:
            weak_acc = 1
            quality = 1.0

        return {
            'reply': reply,
            'result': {'strong': strong_acc, 'weak': weak_acc, 'quality': quality}
        }


async def run_zero_shot_single_hypothesis_async(model_name, task_type, height, num_examples=NUM_EXAMPLES, base_url=None):
    """Run zero-shot single hypothesis experiment with parallel async requests."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        base_url=base_url or DEFAULT_BASE_URL,
        api_key=DEFAULT_API_KEY,
    )

    # Seed for reproducibility
    seed(SEED)
    np.random.seed(SEED)

    config = OntologyConfig(
        hops=height,
        recover_membership=(task_type == 'membership'),
        recover_ontology=(task_type == 'ontology'),
        recover_property=(task_type == 'property'),
        difficulty=Difficulty.SINGLE,
        mix_hops=False
    )

    print(f"\n{'='*60}")
    print(f"Zero-shot Single Hypothesis (FAST): {task_type}, Height {height}")
    print(f"{'='*60}")

    # Generate all examples first (must be sequential for reproducibility)
    print(f"  Generating {num_examples} examples...")
    examples = [generate_single_example(config) for _ in range(num_examples)]

    # Process all examples in parallel
    print(f"  Processing {num_examples} examples with {CONCURRENT_REQUESTS} parallel requests...")
    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)

    tasks = [
        process_single_example(client, model_name, ont, semaphore)
        for ont in examples
    ]

    responses = await asyncio.gather(*tasks)

    # Extract results
    replies = [r['reply'] for r in responses]
    results = [r['result'] for r in responses]

    # Compute aggregate metrics
    strong_mean = np.mean([r['strong'] for r in results])
    weak_mean = np.mean([r['weak'] for r in results])
    quality_mean = np.mean([r['quality'] for r in results])

    strong_ci = wilson_confidence_interval(strong_mean, num_examples)
    weak_ci = wilson_confidence_interval(weak_mean, num_examples)

    print(f"\n  Results:")
    print(f"    Strong Accuracy: {strong_mean:.3f} [{strong_ci[0]:.3f}, {strong_ci[1]:.3f}]")
    print(f"    Weak Accuracy: {weak_mean:.3f} [{weak_ci[0]:.3f}, {weak_ci[1]:.3f}]")
    print(f"    Quality: {quality_mean:.3f}")

    # Save results
    output_dir = pathlib.Path("results")
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / f"zeroshot_single_{task_type}_h{height}_{sanitize_model_name(model_name)}.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump({
            'examples': examples,
            'replies': replies,
            'results': results,
            'metrics': {
                'strong_accuracy': strong_mean,
                'weak_accuracy': weak_mean,
                'quality': quality_mean,
                'strong_ci': strong_ci,
                'weak_ci': weak_ci
            }
        }, f)

    print(f"  Saved to {output_file}")

    return {'strong_accuracy': strong_mean, 'weak_accuracy': weak_mean, 'quality': quality_mean}


def main():
    parser = argparse.ArgumentParser(description='Run INABHYD NL experiments (FAST)')
    parser.add_argument('--model', '-m', type=str, required=True)
    parser.add_argument('--base-url', '-b', type=str, required=True)
    parser.add_argument('--num-examples', '-n', type=int, default=100)
    parser.add_argument('--task', '-t', type=str, default='property',
                        choices=['property', 'membership', 'ontology'])
    parser.add_argument('--height', type=int, default=0,
                        choices=[0, 1, 2, 3, 4])
    parser.add_argument('--concurrent', '-c', type=int, default=20,
                        help='Number of concurrent requests (default: 20)')
    parser.add_argument('--no-system-prompt', action='store_true',
                        help='Merge system prompt into user message (for models like Gemma 2 that do not support system role)')
    args = parser.parse_args()

    global CONCURRENT_REQUESTS, NO_SYSTEM_PROMPT
    CONCURRENT_REQUESTS = args.concurrent
    NO_SYSTEM_PROMPT = args.no_system_prompt

    heights = range(1, 5) if args.height == 0 else [args.height]

    for height in heights:
        asyncio.run(run_zero_shot_single_hypothesis_async(
            args.model, args.task, height, args.num_examples, args.base_url
        ))


if __name__ == '__main__':
    main()
