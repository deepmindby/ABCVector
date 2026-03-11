"""
Argument parser for CoT Vectors.
All hyperparameters are defined here.

Supports: Extracted, Learnable, Uncertainty-Aware (UA), and ABC methods.

Based on "Variational CoT Vectors" framework:
- Extracted: Statistical aggregation to approximate posterior
- Learnable: Gradient optimization for global reasoning patterns
- UA: Bayesian shrinkage with uncertainty-aware gating
- ABC: Adaptive Bayesian CoT Vector with variational inference
"""

import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="CoT Vectors: Variational and Uncertainty-Aware Methods"
    )
    
    # ==================== General Configuration ====================
    parser.add_argument(
        "--model_path",
        type=str,
        default="/home/haichao/TA/UACoTV/models/Qwen2.5-Math-7B",
        help="Path to the pretrained model"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="qwen",
        choices=["qwen", "llama"],
        help="Model type for architecture-specific handling"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/home/haichao/TA/UACoTV/data",
        help="Path to the data directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Base directory to save outputs (vectors saved to output_dir/{dataset}/)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    # ==================== Method Selection ====================
    parser.add_argument(
        "--method",
        type=str,
        default="extracted",
        choices=["extracted", "learnable", "ua", "abc"],
        help="CoT Vector acquisition method: extracted, learnable, ua, or abc"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=["extract", "train", "eval", "both"],
        help="Operation mode"
    )
    
    # ==================== Dataset Configuration ====================
    parser.add_argument(
        "--dataset",
        type=str,
        default="gsm8k",
        choices=["gsm8k", "math_easy", "math_hard", "mmlu_pro"],
        help="Dataset to use"
    )
    parser.add_argument(
        "--num_support_samples",
        type=int,
        default=3000,
        help="Number of support samples for vector extraction/training"
    )
    parser.add_argument(
        "--num_test_samples",
        type=int,
        default=100,
        help="Number of test samples for evaluation"
    )
    
    # ==================== CoT Vector Configuration ====================
    parser.add_argument(
        "--layer_idx",
        type=int,
        default=0,
        help="Layer index to inject/extract CoT Vector"
    )
    parser.add_argument(
        "--scaling_factor",
        type=float,
        default=1.0,
        help="Scaling factor μ for extracted vectors (Eq. 7 in paper)"
    )
    
    # ==================== UA Vector Configuration ====================
    parser.add_argument(
        "--tau_squared",
        type=float,
        default=1.0,
        help="Prior variance τ² for Bayesian shrinkage. "
             "Smaller values = stronger regularization toward zero"
    )
    parser.add_argument(
        "--min_variance",
        type=float,
        default=1e-6,
        help="Minimum variance threshold for numerical stability"
    )
    
    # ==================== Learnable Vector Configuration ====================
    parser.add_argument(
        "--lambda_val",
        type=float,
        default=0.5,
        help="Balance factor λ between alignment and CE loss (Eq. 6)"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-3,
        help="Learning rate for vector optimization (will be overridden by tiered LR strategy)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size for training"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=2,
        help="Gradient accumulation steps (paper default: 2)"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=5,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.5,
        help="Warmup ratio for LR scheduler (paper default: 0.5 for MATH/LLaMA)"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-3,
        help="Weight decay for AdamW"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
        help="Maximum sequence length for learnable method"
    )
    
    # ==================== ABC Vector Configuration ====================
    parser.add_argument(
        "--abc_hidden_dim",
        type=int,
        default=512,
        help="Hidden dimension for ABC prior/posterior MLP networks"
    )
    parser.add_argument(
        "--kl_beta",
        type=float,
        default=1.0,
        help="KL divergence weight in ELBO objective"
    )
    parser.add_argument(
        "--kl_warmup_steps",
        type=int,
        default=0,
        help="Number of warmup steps for KL weight (0 = no warmup)"
    )
    parser.add_argument(
        "--sigma_min",
        type=float,
        default=1e-4,
        help="Minimum sigma value for numerical stability in ABC"
    )
    parser.add_argument(
        "--abc_learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate for ABC networks (prior, posterior, gate)"
    )
    
    # ==================== ABC Posterior Mode (Ablation) ====================
    # Controls what privileged information the posterior network receives.
    #
    # Theoretical interpretation:
    #   q_y_qca : q(z | Q, Y_{Q;C;A})  — posterior sees full teacher output (Q + CoT + Answer).
    #                                      This is the default and strongest privileged signal.
    #   q_y_qc  : q(z | Q, Y_{Q;C})    — posterior sees Q + CoT only (no answer leak).
    #                                      Tests whether CoT reasoning is sufficient supervision.
    #   q_y_q   : q(z | Q, Y_Q)        — posterior sees only question features (Y = r_Q).
    #                                      Posterior degenerates to f(r_Q, r_Q); tests whether
    #                                      a second network on the same input adds capacity.
    #   none    : q(z | Q) = p(z | Q)  — no separate posterior; training samples z from prior.
    #                                      KL = 0 by construction, pure prior-only training.
    #                                      Equivalent to removing the variational inference.
    parser.add_argument(
        "--posterior_mode",
        type=str,
        default="q_y_qca",
        choices=["q_y_qca", "q_y_qc", "q_y_q", "none"],
        help=(
            "Posterior information ablation mode.\n"
            "  q_y_qca: Y from [Q; CoT; Answer] (default, full privileged info)\n"
            "  q_y_qc:  Y from [Q; CoT] (no answer tokens)\n"
            "  q_y_q:   Y = r_Q (question only, tests posterior capacity)\n"
            "  none:    No posterior, train with prior only (KL=0)"
        )
    )
    
    # ==================== ABC Diagnostics ====================
    parser.add_argument(
        "--save_diagnostics",
        action="store_true",
        default=False,
        help="Save detailed training diagnostics (mu gap, sigma stats, etc.) to jsonl"
    )
    parser.add_argument(
        "--diagnostic_split",
        type=str,
        default="both",
        choices=["support", "test", "both"],
        help="Which split(s) to run diagnostic evaluation on"
    )
    parser.add_argument(
        "--run_posterior_eval",
        action="store_true",
        default=False,
        help="Run evaluation using posterior mean z* = mu_psi(Q, Y) (requires teacher features at eval time)"
    )
    parser.add_argument(
        "--run_prior_eval",
        action="store_true",
        default=False,
        help="Run evaluation using prior mean z* = mu_phi(Q) (this is the standard ABC eval)"
    )
    
    # ==================== Generation Configuration (Evaluation) ====================
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum new tokens to generate"
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=3,
        help="Number of beams (1=greedy, faster)"
    )
    parser.add_argument(
        "--do_sample",
        action="store_true",
        default=False,
        help="Use sampling during generation"
    )
    parser.add_argument(
        "--use_early_stopping",
        action="store_true",
        default=True,
        help="Stop when answer pattern detected"
    )
    
    # ==================== Logging Configuration ====================
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        default=False,
        help="Enable WandB logging"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="cot-vectors-variational",
        help="WandB project name"
    )
    parser.add_argument(
        "--skip_baseline",
        action="store_true",
        default=False,
        help="Skip baseline evaluation"
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=10,
        help="Log every N steps"
    )
    
    # ==================== Vector I/O ====================
    parser.add_argument(
        "--vector_path",
        type=str,
        default=None,
        help="Path to load pre-computed vector"
    )
    parser.add_argument(
        "--save_vector",
        action="store_true",
        default=True,
        help="Save extracted/learned vector"
    )
    parser.add_argument(
        "--abc_checkpoint_path",
        type=str,
        default=None,
        help="Path to load pre-trained ABC checkpoint (prior/posterior/gate)"
    )
    
    # ==================== Layer Sweep Configuration ====================
    parser.add_argument(
        "--layers",
        type=str,
        default=None,
        help="Comma-separated layers to test (e.g., '0,5,10'). "
             "Default: all layers with step"
    )
    parser.add_argument(
        "--layer_step",
        type=int,
        default=2,
        help="Step size when testing all layers (e.g., 2 = test every 2nd layer)"
    )
    parser.add_argument(
        "--baseline_accuracy",
        type=float,
        default=None,
        help="Pre-computed baseline accuracy (use with --skip_baseline)"
    )
    parser.add_argument(
        "--load_vectors_dir",
        type=str,
        default=None,
        help="Load pre-trained vectors from directory (skip extraction/training)"
    )
    
    return parser.parse_args()