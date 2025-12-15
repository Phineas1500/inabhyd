#!/usr/bin/env python3
"""
Run factorial experiment across all matched pair sets.

Set 1 (Pure): Baseline 2-hop with single child→parent path
Set 2 (Salience): Same structure + direct parent members (tests frequency heuristic)
Set 4 (INABHYD-style): Multiple children → parent + direct member (ceiling)

This tests whether salience/frequency alone explains INABHYD's multi-hop "reasoning".
"""

import os
import sys
import pickle
import argparse
import pathlib
import asyncio
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DEFAULT_BASE_URL = os.environ.get("OPENAI_BASE_URL", None)
DEFAULT_API_KEY = os.environ.get("OPENAI_API_KEY", "not-needed")

from evaluate import parse_hypotheses_from_response, compute_strong_accuracy, wilson_confidence_interval

CONCURRENT_REQUESTS = 20
NO_SYSTEM_PROMPT = False

NL_SYSTEM_PROMPT = """You are a helpful assitant that performs abduction and induction reasoning.
        Your job is to come up with hypotheses that explain observations with given theories. Each hypothesis should explain as many observations as possible.
        You can come up with multiple hypotheses and each hypothesis should take one line with the format A is B or A is not B.
    .   Only output final hypotheses.
"""


def sanitize_model_name(model_name):
    return model_name.replace("-", "").replace("/", "_").replace(":", "_").lower()


async def process_example(client, model_name, example, semaphore):
    """Process a single example."""
    async with semaphore:
        # Build user prompt
        user_prompt = f"Q: {example['theories_nl']} We observe that: {example['observations_nl']} Please come up with hypothesis to explain observations."

        if NO_SYSTEM_PROMPT:
            combined = f"{NL_SYSTEM_PROMPT}\n\n{user_prompt}"
            messages = [{"role": "user", "content": combined}]
        else:
            messages = [
                {"role": "system", "content": NL_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ]

        try:
            completion = await client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0,
                max_tokens=512
            )
            reply = completion.choices[0].message.content
        except Exception as e:
            print(f"  Error: {e}")
            reply = ""

        # Evaluate
        pred_hyps = parse_hypotheses_from_response(reply)
        gt_hyps = [example['gt_hypothesis_nl']]
        strong_acc = compute_strong_accuracy(pred_hyps, gt_hyps)

        return {
            'reply': reply,
            'gt': example['gt_hypothesis_nl'],
            'strong': strong_acc,
            'seed': example['seed'],
            'is_negated': example['is_negated'],
            'parent_mentions': example.get('parent_mentions', 1),
        }


async def process_matched_pair(client, model_name, h1, h2, semaphore):
    """Process a matched pair (Set 1 only)."""
    async with semaphore:
        results = []
        for example in [h1, h2]:
            user_prompt = f"Q: {example['theories_nl']} We observe that: {example['observations_nl']} Please come up with hypothesis to explain observations."

            if NO_SYSTEM_PROMPT:
                combined = f"{NL_SYSTEM_PROMPT}\n\n{user_prompt}"
                messages = [{"role": "user", "content": combined}]
            else:
                messages = [
                    {"role": "system", "content": NL_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ]

            try:
                completion = await client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=0,
                    max_tokens=512
                )
                reply = completion.choices[0].message.content
            except Exception as e:
                print(f"  Error: {e}")
                reply = ""

            pred_hyps = parse_hypotheses_from_response(reply)
            gt_hyps = [example['gt_hypothesis_nl']]
            strong_acc = compute_strong_accuracy(pred_hyps, gt_hyps)

            results.append({
                'reply': reply,
                'gt': example['gt_hypothesis_nl'],
                'strong': strong_acc,
                'depth': example['depth'],
            })

        return results[0], results[1]  # h1_result, h2_result


async def run_set1_experiment(client, model_name, pairs_file):
    """Run Set 1 (Pure) experiment with H1/H2 pairs."""
    with open(pairs_file, 'rb') as f:
        pairs = pickle.load(f)

    print(f"\n--- Set 1 (Pure Multi-hop): {len(pairs)} pairs ---")

    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)
    tasks = [process_matched_pair(client, model_name, h1, h2, semaphore) for h1, h2 in pairs]
    results = await asyncio.gather(*tasks)

    h1_results = [r[0] for r in results]
    h2_results = [r[1] for r in results]

    h1_strong = np.mean([r['strong'] for r in h1_results])
    h2_strong = np.mean([r['strong'] for r in h2_results])

    print(f"  H1 Strong: {h1_strong:.1%}")
    print(f"  H2 Strong: {h2_strong:.1%}")

    return {
        'h1_strong': h1_strong,
        'h2_strong': h2_strong,
        'h1_results': h1_results,
        'h2_results': h2_results,
        'pairs': pairs,
    }


async def run_single_set_experiment(client, model_name, examples_file, set_name):
    """Run experiment on a single set (Set 2 or Set 4)."""
    with open(examples_file, 'rb') as f:
        examples = pickle.load(f)

    print(f"\n--- {set_name}: {len(examples)} examples ---")

    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)
    tasks = [process_example(client, model_name, ex, semaphore) for ex in examples]
    results = await asyncio.gather(*tasks)

    strong = np.mean([r['strong'] for r in results])
    avg_mentions = np.mean([r['parent_mentions'] for r in results])

    print(f"  Strong: {strong:.1%}")
    print(f"  Avg parent mentions: {avg_mentions:.1f}")

    return {
        'strong': strong,
        'results': results,
        'examples': examples,
        'avg_parent_mentions': avg_mentions,
    }


async def run_factorial_experiment(model_name, base_url):
    """Run full factorial experiment across all sets."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(base_url=base_url, api_key=DEFAULT_API_KEY)

    print("=" * 70)
    print(f"FACTORIAL EXPERIMENT: {model_name}")
    print(f"Testing salience hypothesis for multi-hop reasoning")
    print("=" * 70)

    results = {}

    # Set 1: Pure multi-hop
    if os.path.exists('matched_pairs_set1_pure.pkl'):
        results['set1'] = await run_set1_experiment(client, model_name, 'matched_pairs_set1_pure.pkl')
    else:
        print("  Warning: matched_pairs_set1_pure.pkl not found")

    # Set 2: Salience test
    if os.path.exists('matched_pairs_set2_salience.pkl'):
        results['set2'] = await run_single_set_experiment(
            client, model_name, 'matched_pairs_set2_salience.pkl', 'Set 2 (Salience Test)')
    else:
        print("  Warning: matched_pairs_set2_salience.pkl not found")

    # Set 4: INABHYD-style
    if os.path.exists('matched_pairs_set4_inabhyd.pkl'):
        results['set4'] = await run_single_set_experiment(
            client, model_name, 'matched_pairs_set4_inabhyd.pkl', 'Set 4 (INABHYD-style)')
    else:
        print("  Warning: matched_pairs_set4_inabhyd.pkl not found")

    # Set 5: Evidential Path
    if os.path.exists('matched_pairs_set5_evidential.pkl'):
        results['set5'] = await run_single_set_experiment(
            client, model_name, 'matched_pairs_set5_evidential.pkl', 'Set 5 (Evidential Path)')
    else:
        print("  Warning: matched_pairs_set5_evidential.pkl not found")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\n{'Set':<25} {'Parent Mentions':<18} {'Strong Accuracy':<15}")
    print("-" * 60)

    if 'set1' in results:
        print(f"{'Set 1 (Pure) H1':<25} {'1x':<18} {results['set1']['h1_strong']:.1%}")
        print(f"{'Set 1 (Pure) H2':<25} {'1x':<18} {results['set1']['h2_strong']:.1%}")

    if 'set2' in results:
        mentions = results['set2']['avg_parent_mentions']
        print(f"{'Set 2 (Salience)':<25} {f'{mentions:.0f}x':<18} {results['set2']['strong']:.1%}")

    if 'set4' in results:
        mentions = results['set4']['avg_parent_mentions']
        print(f"{'Set 4 (INABHYD-style)':<25} {f'{mentions:.0f}x':<18} {results['set4']['strong']:.1%}")

    if 'set5' in results:
        mentions = results['set5']['avg_parent_mentions']
        print(f"{'Set 5 (Evidential Path)':<25} {f'{mentions:.0f}x':<18} {results['set5']['strong']:.1%}")

    # Interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    if 'set1' in results and 'set2' in results:
        set1_h2 = results['set1']['h2_strong']
        set2 = results['set2']['strong']

        if set2 > set1_h2 + 0.1:
            print(f"\nSet 2 ({set2:.0%}) >> Set 1 H2 ({set1_h2:.0%})")
            print("→ SALIENCE ALONE IMPROVES PERFORMANCE")
            print("→ The model uses frequency/salience heuristics, not compositional reasoning")
        elif abs(set2 - set1_h2) <= 0.1:
            print(f"\nSet 2 ({set2:.0%}) ≈ Set 1 H2 ({set1_h2:.0%})")
            print("→ Salience alone doesn't help")
            print("→ Need to investigate other factors")
        else:
            print(f"\nSet 2 ({set2:.0%}) < Set 1 H2 ({set1_h2:.0%})")
            print("→ Unexpected: distractors hurt performance")

    if 'set2' in results and 'set4' in results:
        set2 = results['set2']['strong']
        set4 = results['set4']['strong']

        if abs(set2 - set4) <= 0.1:
            print(f"\nSet 2 ({set2:.0%}) ≈ Set 4 ({set4:.0%})")
            print("→ Salience alone explains INABHYD performance")
            print("→ Multiple paths don't add anything beyond frequency boost")
        elif set4 > set2 + 0.1:
            print(f"\nSet 4 ({set4:.0%}) > Set 2 ({set2:.0%})")
            print("→ Structure (multiple paths) adds something beyond salience")

    # Set 5 analysis: Test the evidential path hypothesis
    if 'set5' in results:
        set5 = results['set5']['strong']

        if 'set2' in results:
            set2 = results['set2']['strong']
            if set5 > set2 + 0.1:
                print(f"\nSet 5 ({set5:.0%}) >> Set 2 ({set2:.0%})")
                print("→ EVIDENTIAL PATH HYPOTHESIS CONFIRMED")
                print("→ Direct parent members WITH observed properties enable 1-hop shortcut")
                print("→ Salience without evidence is useless")
            elif abs(set5 - set2) <= 0.1:
                print(f"\nSet 5 ({set5:.0%}) ≈ Set 2 ({set2:.0%})")
                print("→ Unexpected: evidential path doesn't help")

        if 'set4' in results:
            set4 = results['set4']['strong']
            if abs(set5 - set4) <= 0.1:
                print(f"\nSet 5 ({set5:.0%}) ≈ Set 4 ({set4:.0%})")
                print("→ STRUCTURE DOESN'T MATTER - only evidential path matters")
                print("→ Model uses: (X is concept) ∧ (X has property) → concept has property")
                print("→ This is CONJUNCTION DETECTION, not compositional reasoning")
            elif set4 > set5 + 0.1:
                print(f"\nSet 4 ({set4:.0%}) > Set 5 ({set5:.0%})")
                print("→ Multiple convergent paths add something beyond single evidential path")

    # Save results
    output_dir = pathlib.Path("factorial_results")
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / f"factorial_{sanitize_model_name(model_name)}.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nSaved to {output_file}")

    return results


def main():
    global NO_SYSTEM_PROMPT, CONCURRENT_REQUESTS

    parser = argparse.ArgumentParser(description='Run factorial experiment on matched pair sets')
    parser.add_argument('--model', '-m', type=str, required=True, help='Model name')
    parser.add_argument('--base-url', '-b', type=str, required=True, help='API base URL')
    parser.add_argument('--no-system-prompt', action='store_true', help='Merge system into user (for Gemma 2)')
    parser.add_argument('--concurrent', '-c', type=int, default=20, help='Concurrent requests')
    args = parser.parse_args()

    NO_SYSTEM_PROMPT = args.no_system_prompt
    CONCURRENT_REQUESTS = args.concurrent

    asyncio.run(run_factorial_experiment(args.model, args.base_url))


if __name__ == "__main__":
    main()
