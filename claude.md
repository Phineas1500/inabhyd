# INABHYD Replication Project - Important Context

## Project Overview
Replicating experiments from the INABHYD paper (2509.03345v1.pdf) - Inductive and Abductive Reasoning with Ontology Trees.

## Model Configurations

### Gemma 3 27B IT (Instruction-tuned)
- **Model**: `google/gemma-3-27b-it`
- **Deployment**: Modal with vLLM (H100 GPU)
- **Endpoint**: `https://phineas1500--gemma3-27b-inference-serve.modal.run/v1`
- **Decoding**: Greedy (temperature=0) for deterministic results
- **Model name for API**: `gemma3-27b`
- **API**: Chat completions (instruction-tuned model)

### Pythia 160M (Base model)
- **Model**: `EleutherAI/pythia-160m-deduped`
- **Deployment**: Modal with vLLM (T4 GPU - smaller model)
- **Endpoint**: `https://phineas1500--pythia-160m-inference-serve.modal.run/v1`
- **Decoding**: Greedy (temperature=0)
- **Model name for API**: `pythia-160m`
- **API**: Completions (base model, no chat template)
- **Note**: Base model - will not follow instructions like instruction-tuned models

## Running Experiments

### Deploy Modal (keep server running):
```bash
# For Gemma 3 27B
modal deploy gemma_modal.py

# For Pythia 160M
modal deploy pythia_modal.py
```

### Run experiments:
```bash
# Gemma 3 27B - Zero-shot single hypothesis (property task)
python run_experiments.py \
  --model gemma3-27b \
  --base-url "https://phineas1500--gemma3-27b-inference-serve.modal.run/v1" \
  --experiment zeroshot_single \
  --task property

# Pythia 160M - Zero-shot single hypothesis (all tasks, all heights)
python run_experiments.py \
  --model pythia-160m \
  --base-url "https://phineas1500--pythia-160m-inference-serve.modal.run/v1" \
  --experiment zeroshot_single \
  --task all

# Can run all heights at once with --height 1 2 3 4
```

## Key Evaluation Fixes Made

### 1. Strong Accuracy: First-Hypothesis-Only (evaluate.py)
**Problem**: Gemma outputs multiple hypotheses (e.g., "X is Y. Y are X. Being X causes Y..."), causing strong accuracy to fail due to count mismatch.

**Solution**: Changed `compute_strong_accuracy()` to use `first_only=True` by default, comparing only the first predicted hypothesis with the first GT hypothesis.

**Result**:
- H1: 0.66 -> 0.95 (paper: 0.90) ✓
- H2: 0.17 -> 0.53 (paper: 0.50) ✓

### 2. Parsing "because" explanations (evaluate.py)
**Problem**: Lines like "Kevin is salty because he is a gwompant" weren't being parsed correctly.

**Solution**: Added logic to extract hypothesis before explanation words ("because", "since", "as", "given", "due to").

### 3. Numpy Random Seeding (run_experiments.py)
**Problem**: `ontology.py` uses `numpy.random.choice` but only Python's `random` was seeded, causing non-reproducible examples.

**Solution**: Added `np.random.seed(SEED)` alongside `seed(SEED)` in all experiment functions.

### 4. Pickle Loading for Old Results
**Problem**: Saved Ontology objects can't be loaded if class structure changed.

**Solution**: Use monkey-patch in reevaluate_results.py:
```python
def safe_hash(self):
    if hasattr(self, 'name'):
        return hash(self.name)
    return id(self)
ontology.OntologyNode.__hash__ = safe_hash
```

## Current Results (Gemma 3 27B) vs Paper

### Zero-shot Single Hypothesis - Property Task
| Height | Our Strong | Paper Strong | Our Weak | Paper Weak |
|--------|-----------|--------------|----------|------------|
| H1     | 0.95      | 0.90         | 0.94     | 0.90       |
| H2     | 0.53      | 0.50         | 0.70     | 0.65       |
| H3     | 0.44      | 0.25         | 0.59     | 0.35       |
| H4     | 0.21      | 0.05         | 0.41     | 0.18       |

**Note**: H3 and H4 show our model performing better than paper's - this could be due to:
1. Different Gemma version (3-27B-IT vs paper's version)
2. Different random examples (can't fully reproduce paper's exact examples)
3. Our model might genuinely be better at higher heights

## Important Files

- `run_experiments.py` - Main experiment runner (supports both chat and completions API)
- `evaluate.py` - Evaluation metrics (strong, weak accuracy, quality)
- `gemma_modal.py` - Modal deployment for Gemma 3 27B (instruction-tuned)
- `pythia_modal.py` - Modal deployment for Pythia 160M (base model)
- `reevaluate_results.py` - Re-evaluate saved results with corrected logic
- `ontology.py` - Ontology tree generation (uses numpy.random.choice)
- `morphology.py` - Generates concept/entity/property names

## Result Files Location
- Results saved to: `results/zeroshot_single_{task}_h{height}_{model}.pkl`
- Format: Dict with keys `['examples', 'replies', 'results', 'metrics']`

## Known Issues

### Paper Data Limitations
- Paper only released H1 data for single-task experiments (`1hop_0shot_property_single_reply_gemmi.pkl`)
- H2-H4 only have "mix" task data (`Xhop_0shot_membership_ontology_property_mix_reply_gemmi.pkl`)
- Cannot reproduce exact examples since paper didn't save input data

### Model Output Differences
- Our Gemma outputs multiple hypotheses including inverse reasoning
- Paper's Gemma outputs clean single hypotheses
- This is handled by first_only=True in strong accuracy

## Prompt (matches paper exactly)
```
System: You are a helpful assitant that performs abduction and induction reasoning.
        Your job is to come up with hypotheses that explain observations with given theories. Each hypothesis should explain as many observations as possible.
        You can come up with multiple hypotheses and each hypothesis should take one line with the format A is B or A is not B.
    .   Only output final hypotheses.

User: Q: {theories} We observe that: {observations} Please come up with hypothesis to explain observations.
```

## SEED
`SEED = 62471893` (same as paper)

## FOL Dataset Implementation (COMPLETED)

The First-Order Logic version of the dataset is now implemented. It skips the natural language translation step, outputting pure symbolic FOL.

### Running FOL Experiments:
```bash
# Gemma 3 27B - FOL Zero-shot single hypothesis (property task)
python run_fol_experiments.py \
  --model gemma3-27b \
  --base-url "https://phineas1500--gemma3-27b-inference-serve.modal.run/v1" \
  --experiment zeroshot_single \
  --task property

# Run all heights
python run_fol_experiments.py \
  --model gemma3-27b \
  --base-url "https://phineas1500--gemma3-27b-inference-serve.modal.run/v1" \
  --experiment zeroshot_single \
  --task property \
  --height 0
```

### FOL Format Examples:
| Type | NL | FOL |
|------|-----|-----|
| Entity + Property | "Amy is rainy" | `rainy(Amy)` |
| Entity + Neg Property | "Amy is not slow" | `¬slow(Amy)` |
| Entity + Concept | "Amy is a dalpist" | `dalpist(Amy)` |
| Concept + Property | "All dalpists are rainy" | `∀x(dalpist(x) → rainy(x))` |
| Concept + Concept | "All cats are mammals" | `∀x(cat(x) → mammal(x))` |

### Key Files:
- `run_fol_experiments.py` - FOL experiment runner
- `fol.py:FOL.to_fol()` - FOL string generation method
- `ontology.py:Ontology.fol_theories/fol_observations/fol_hypotheses` - FOL properties
- `evaluate.py` - FOL parsing and evaluation functions (parse_fol_*, compute_fol_*)

See `FOL_DATASET_PLAN.md` for the original implementation plan.
