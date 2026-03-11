# ABC Vector Posterior Ablation & Diagnostics

## Overview

This enhancement adds posterior information ablation and diagnostic evaluation to the ABC Vector method.

### New Arguments (in `args.py`)

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--posterior_mode` | str | `q_y_qca` | Posterior info ablation: `q_y_qca`, `q_y_qc`, `q_y_q`, `none` |
| `--save_diagnostics` | flag | False | Save per-epoch training diagnostics to JSONL |
| `--diagnostic_split` | str | `both` | Which split(s) to run diagnostic eval: `support`, `test`, `both` |
| `--run_posterior_eval` | flag | False | Run eval with posterior mean (privileged) |
| `--run_prior_eval` | flag | False | Run eval with prior mean (standard) |

### Posterior Modes Explained

```
q_y_qca :  q(z | Q, Y_{Q;C;A})
            Posterior sees full teacher: Question + CoT + Answer.
            Y = mean-pool over answer tokens from teacher forward pass.
            This is the DEFAULT and strongest privileged signal.

q_y_qc  :  q(z | Q, Y_{Q;C})
            Posterior sees Q + CoT only (no answer leak).
            Y = mean-pool over CoT tokens from [Q;CoT] forward pass.
            Tests: Is CoT reasoning sufficient supervision?

q_y_q   :  q(z | Q, Y_Q)
            Posterior sees only question features (Y = r_Q).
            Same input twice to posterior: concat(r_Q, r_Q).
            Tests: Does a second network add capacity?

none    :  No separate posterior. Training uses prior directly.
            z ~ p(z|Q) during training. KL = 0 by construction.
            Tests: Is the prior alone sufficient?
```

---

## Running the Four Key Ablations

### 1) Default: q_y_qca (full privileged info)
```bash
python main.py \
    --method abc \
    --dataset gsm8k \
    --layer_idx 0 \
    --posterior_mode q_y_qca \
    --abc_learning_rate 5e-5 \
    --kl_beta 1.0 \
    --num_epochs 10 \
    --num_support_samples 3000 \
    --num_test_samples 500 \
    --save_diagnostics \
    --run_prior_eval \
    --run_posterior_eval
```

### 2) CoT only: q_y_qc (no answer leak)
```bash
python main.py \
    --method abc \
    --dataset gsm8k \
    --layer_idx 0 \
    --posterior_mode q_y_qc \
    --abc_learning_rate 5e-5 \
    --kl_beta 1.0 \
    --num_epochs 10 \
    --num_support_samples 3000 \
    --num_test_samples 500 \
    --save_diagnostics \
    --run_prior_eval \
    --run_posterior_eval
```

### 3) Question only: q_y_q (posterior capacity test)
```bash
python main.py \
    --method abc \
    --dataset gsm8k \
    --layer_idx 0 \
    --posterior_mode q_y_q \
    --abc_learning_rate 5e-5 \
    --kl_beta 1.0 \
    --num_epochs 10 \
    --num_support_samples 3000 \
    --num_test_samples 500 \
    --save_diagnostics \
    --run_prior_eval \
    --run_posterior_eval
```

### 4) No posterior: none (prior-only training)
```bash
python main.py \
    --method abc \
    --dataset gsm8k \
    --layer_idx 0 \
    --posterior_mode none \
    --abc_learning_rate 5e-5 \
    --kl_beta 1.0 \
    --num_epochs 10 \
    --num_support_samples 3000 \
    --num_test_samples 500 \
    --save_diagnostics \
    --run_prior_eval
```

### Batch Script (run all four)
```bash
#!/bin/bash
COMMON="--method abc --dataset gsm8k --layer_idx 0 \
    --abc_learning_rate 5e-5 --kl_beta 1.0 --num_epochs 10 \
    --num_support_samples 3000 --num_test_samples 500 \
    --save_diagnostics --run_prior_eval --run_posterior_eval"

for MODE in q_y_qca q_y_qc q_y_q none; do
    echo "===== Running posterior_mode=$MODE ====="
    python main.py $COMMON --posterior_mode $MODE
done
```

---

## Output Files

### 1) Training Diagnostics (JSONL)
**Path:** `outputs/{dataset}/diagnostics/train_diagnostics_L{layer}_{mode}.jsonl`

One JSON object per epoch:
```json
{
    "epoch": 1,
    "train_loss": 2.3456,
    "train_nll": 2.1234,
    "train_kl": 0.2222,
    "train_mu_gap_l2": 0.0543,
    "train_mu_gap_cos": 0.9876,
    "train_sigma_phi_mean": 0.6931,
    "train_sigma_psi_mean": 0.6928,
    "gate_value": -0.0760,
    "gate_abs": 0.0760,
    "injected_norm_mean": 1.234,
    "beta_t": 1.0,
    "posterior_mode": "q_y_qca",
    "layer_idx": 0
}
```

| Field | Description |
|-------|-------------|
| `train_loss` | Total ELBO loss = NLL + beta_t * KL |
| `train_nll` | Cross-entropy on answer tokens |
| `train_kl` | KL(q_psi ‖ p_phi), mean over batch |
| `train_mu_gap_l2` | ‖μ_ψ − μ_φ‖₂, batch mean |
| `train_mu_gap_cos` | cos(μ_ψ, μ_φ), batch mean. 1.0 = identical direction |
| `train_sigma_phi_mean` | Prior σ mean (across all dims and batch) |
| `train_sigma_psi_mean` | Posterior σ mean |
| `gate_value` | Raw gate scalar g |
| `gate_abs` | |g| |
| `injected_norm_mean` | ‖g·z‖₂ batch mean (effective injection magnitude) |
| `beta_t` | Current KL weight (after warmup) |

### 2) Diagnostic Eval Comparison (CSV)
**Path:** `outputs/{dataset}/diagnostics/diagnostic_eval_L{layer}_{mode}.csv`

| Field | Description |
|-------|-------------|
| `split` | "test" or "support" |
| `posterior_mode` | Which mode was used for training |
| `layer_idx` | Target injection layer |
| `prior_accuracy` | Accuracy with z* = μ_φ(Q) |
| `posterior_accuracy` | Accuracy with z* = μ_ψ(Q, Y) |
| `accuracy_delta` | posterior_accuracy − prior_accuracy |
| `prior_avg_norm` | Mean ‖g·μ_φ‖ across samples |
| `posterior_avg_norm` | Mean ‖g·μ_ψ‖ across samples |
| `norm_delta` | posterior_norm − prior_norm |
| `gate_value` | Final gate value |
| `num_samples` | Number of eval samples |
| `timestamp` | Evaluation time |

### 3) ABC Checkpoint (PyTorch)
**Path:** `outputs/{dataset}/abc_L{layer}_{mode}_{timestamp}.pt`

Contains: `prior`, `posterior`, `gate`, `layer_idx`, `abc_hidden_dim`, `kl_beta`, `kl_warmup_steps`, `sigma_min`, `posterior_mode`, `args`.

---

## Integration Guide

### Files Modified
1. **`src/args.py`** — Added 5 new arguments (posterior_mode, save_diagnostics, diagnostic_split, run_posterior_eval, run_prior_eval)
2. **`src/methods/abc_vector.py`** — Major enhancements:
   - `ABCDataset` supports `posterior_mode` parameter, builds `teacher_qc` prompt for q_y_qc
   - `abc_collate_fn` handles optional `teacher_qc_*` fields
   - Training loop computes and logs diagnostic metrics per epoch
   - New `_extract_teacher_features_qc()` for CoT-only feature extraction
   - New `extract_teacher_feature()` unified function with `feature_span_mode`
   - New `eval_with_prior_mean()` and `eval_with_posterior_mean()` explicit eval functions
   - New `run_diagnostic_eval()` that runs both and saves comparison CSV
   - `compute_diagnostic_metrics()` utility function
   - `save_diagnostics_jsonl()` and `save_eval_comparison_csv()` I/O utilities
3. **`main.py`** — ABC section updated to pass new args and run diagnostic eval

### Backward Compatibility
- Default `posterior_mode="q_y_qca"` preserves original behavior exactly
- Default `save_diagnostics=False` means no extra I/O overhead
- No existing arguments removed
- `eval()` method unchanged (delegates to `eval_with_prior_mean`)
- Checkpoint format backward-compatible (new `posterior_mode` field added, old checkpoints still load)