# CoT Vectors: Variational Framework

Codebase for Chain-of-Thought Vectors based on the **Variational CoT Vectors** framework.

## Available Methods

- `extracted`: Statistical aggregation method (Eq. 4-5)
- `learnable`: Teacher-student gradient optimization (Eq. 6)
- `ua`: Uncertainty-Aware method with Bayesian shrinkage

## Mathematical Foundation

### Extracted Method
Simple mean aggregation:
```
v_E = (1/N) Σ v_i
```

### Learnable Method
Teacher-student optimization:
```
L = L_align + λ * L_CE
```

### UA Method
Bayesian MAP estimation with structured prior:
```
Prior:      p(z) = N(0, τ²I)
Likelihood: p(μ|z) = N(z, σ²)
Posterior:  z_d = k_d * μ_d
            where k_d = σ²_d / (σ²_d + τ²)
```

The shrinkage coefficient `k_d` acts as adaptive gating:
- High variance (noise) → k_d → 0 (suppress)
- Low variance (signal) → k_d → 1 (preserve)

## Usage

### Single-Layer Extracted
```bash
python main.py \
    --method extracted \
    --layer_idx 10 \
    --dataset gsm8k \
    --model_path /path/to/model
```

### Single-Layer UA
```bash
python main.py \
    --method ua \
    --layer_idx 10 \
    --tau_squared 1.0 \
    --dataset gsm8k
```

### Learnable
```bash
python main.py \
    --method learnable \
    --layer_idx 10 \
    --num_epochs 5 \
    --learning_rate 5e-3 \
    --lambda_val 0.5 \
    --dataset gsm8k
```

### Layer Sweep
```bash
python run_layer_sweep.py \
    --method ua \
    --layers "0,5,10,15,20,25" \
    --tau_squared 1.0 \
    --save_vectors
```

## Project Structure

```
cot_vectors/
├── main.py                 # Main entry point
├── run_layer_sweep.py      # Layer sweep script
├── requirements.txt        # Dependencies
└── src/
    ├── __init__.py
    ├── args.py             # All hyperparameters defined here
    ├── data_utils.py       # Data loading
    ├── eval.py             # Evaluation
    ├── models.py           # Model wrapper with hooks
    ├── utils.py            # Utilities
    └── methods/
        ├── __init__.py     # Method exports
        ├── base.py         # Base class
        ├── extracted.py    # Extracted method
        ├── learnable.py    # Learnable method
        └── ua_vector.py    # UA method
```

## Key Arguments

All hyperparameters are defined in `src/args.py`.

### General Arguments
| Argument | Description | Default |
|----------|-------------|---------|
| `--method` | `extracted`, `learnable`, `ua` | `extracted` |
| `--layer_idx` | Target layer for injection | `0` |
| `--dataset` | Dataset name | `gsm8k` |
| `--mode` | `extract`, `train`, `eval`, `both` | `both` |

### UA Method Arguments
| Argument | Description | Default |
|----------|-------------|---------|
| `--tau_squared` | Prior variance τ² for UA | `1.0` |
| `--min_variance` | Minimum variance threshold | `1e-6` |

### Learnable Method Arguments
| Argument | Description | Default |
|----------|-------------|---------|
| `--lambda_val` | Balance factor λ | `0.5` |
| `--learning_rate` | Learning rate (tiered by model/layer) | `5e-3` |
| `--num_epochs` | Number of training epochs | `5` |
| `--batch_size` | Batch size | `2` |
| `--gradient_accumulation_steps` | Gradient accumulation | `2` |
| `--warmup_ratio` | Warmup ratio for LR scheduler | `0.5` |
| `--weight_decay` | Weight decay | `1e-3` |
| `--max_length` | Max sequence length | `1024` |

### Generation Arguments
| Argument | Description | Default |
|----------|-------------|---------|
| `--max_new_tokens` | Max new tokens to generate | `512` |
| `--num_beams` | Number of beams | `3` |
| `--scaling_factor` | Scaling factor for extracted vectors | `1.0` |

## Output

### Saved Vector Format
```python
{
    "vector": tensor,           # The CoT vector
    "args": {...},              # Arguments used
    "method": "ua",             # Method name
    "statistics": {             # For UA method only
        "mean_vector": tensor,
        "variance_vector": tensor,
        "shrinkage_coefficients": tensor,
        "tau_squared": float,
    },
}
```

## Installation

```bash
pip install -r requirements.txt
```

## Citation

If you use this code, please cite the original CoT Vectors paper.
