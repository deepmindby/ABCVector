"""
Main entry point for CoT Vectors.

Supports four methods based on Variational CoT Vectors framework:
- Extracted: Statistical aggregation of activation differences
- Learnable: Gradient optimization via teacher-student framework
- UA: Uncertainty-Aware with Bayesian shrinkage
- ABC: Adaptive Bayesian CoT Vector with variational inference
"""

import os
import copy
import torch
from datetime import datetime

from src.args import parse_args
from src.models import CoTModelWrapper, load_tokenizer, parse_model_specs
from src.data_utils import load_dataset
from src.methods.extracted import ExtractedCoTVector
from src.methods.learnable import LearnableCoTVector
from src.methods.ua_vector import UACoTVector
from src.methods.abc_vector import ABCCoTVector
from src.eval import run_baseline_evaluation, run_injection_evaluation
from src.utils import set_seed, setup_wandb


def get_output_dir(base_dir: str, dataset: str) -> str:
    """Get dataset-specific output directory: output_dir/{dataset}/"""
    output_dir = os.path.join(base_dir, dataset)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def run_single_model(args, output_dir: str, support_samples, test_samples):
    model_tag = args.model_path.split("/")[-1]

    print("\n" + "=" * 60)
    print("CoT Vectors: Variational Framework")
    print("=" * 60)
    print(f"Model: {model_tag}")
    print(f"Model type: {args.model_name}")
    print(f"Method: {args.method}")
    print(f"Dataset: {args.dataset}")
    print(f"Layer: {args.layer_idx}")
    print(f"Mode: {args.mode}")
    print(f"Output: {output_dir}")
    print(f"Beams: {args.num_beams}, Max tokens: {args.max_new_tokens}")

    if args.method == "learnable":
        print("-" * 60)
        print("Learnable Configuration:")
        print(f"  Epochs: {args.num_epochs}")
        print(f"  Batch size: {args.batch_size}")
        print(f"  Gradient accumulation: {args.gradient_accumulation_steps}")
        print(f"  Learning rate: {args.learning_rate}")
        print(f"  Lambda: {args.lambda_val}")
        print(f"  Max length: {args.max_length}")

    if args.method == "ua":
        print("-" * 60)
        print("Uncertainty-Aware (UA) Configuration:")
        print(f"  Prior variance τ²: {args.tau_squared}")
        print(f"  Min variance: {args.min_variance}")

    if args.method == "abc":
        print("-" * 60)
        print("Adaptive Bayesian CoT (ABC) Configuration:")
        print(f"  Hidden dim: {args.abc_hidden_dim}")
        print(f"  KL beta: {args.kl_beta}")
        print(f"  KL warmup steps: {args.kl_warmup_steps}")
        print(f"  Sigma min: {args.sigma_min}")
        print(f"  Learning rate: {args.abc_learning_rate}")
        print(f"  Epochs: {args.num_epochs}")
        print(f"  Batch size: {args.batch_size}")
        print(f"  Gradient accumulation: {args.gradient_accumulation_steps}")
        print(f"  Max length: {args.max_length}")

    print("=" * 60)

    wandb_run = setup_wandb(args) if args.use_wandb else None

    print("\nLoading model...")
    model_wrapper = CoTModelWrapper(args.model_path, args.model_name)
    tokenizer = load_tokenizer(args.model_path)
    print(f"Model loaded: {model_wrapper.num_layers} layers, hidden_size={model_wrapper.hidden_size}")

    if args.method == "abc":
        abc_method = ABCCoTVector(
            model_wrapper=model_wrapper,
            tokenizer=tokenizer,
            layer_idx=args.layer_idx,
            dataset_type=args.dataset,
            abc_hidden_dim=args.abc_hidden_dim,
            kl_beta=args.kl_beta,
            kl_warmup_steps=args.kl_warmup_steps,
            sigma_min=args.sigma_min,
            learning_rate=args.abc_learning_rate,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_length=args.max_length,
        )

        if args.abc_checkpoint_path:
            print(f"\nLoading ABC checkpoint from {args.abc_checkpoint_path}")
            checkpoint = torch.load(args.abc_checkpoint_path, map_location="cpu")
            abc_method.load_state_dict(checkpoint, device=model_wrapper.device)
            print("ABC checkpoint loaded successfully")

        if args.mode in ["train", "both"] and support_samples:
            print("\nTraining ABC Vector...")
            abc_method.train(support_samples, wandb_run)
            if args.save_vector:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                checkpoint_path = os.path.join(output_dir, f"abc_{model_tag}_L{args.layer_idx}_{timestamp}.pt")
                save_data = {**abc_method.get_state_dict(), "args": vars(args)}
                torch.save(save_data, checkpoint_path)
                print(f"ABC checkpoint saved to {checkpoint_path}")

        if args.mode in ["eval", "both"] and test_samples:
            baseline_results = None
            if not args.skip_baseline:
                baseline_results = run_baseline_evaluation(
                    model_wrapper=model_wrapper,
                    tokenizer=tokenizer,
                    test_samples=test_samples,
                    dataset_type=args.dataset,
                    max_new_tokens=args.max_new_tokens,
                    num_beams=args.num_beams,
                    use_early_stopping=args.use_early_stopping,
                )

            abc_results = abc_method.eval(
                test_samples=test_samples,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
                use_early_stopping=args.use_early_stopping,
            )
            print(f"\n[{model_tag}] ABC accuracy: {abc_results['accuracy']:.2f}%")
            if baseline_results:
                print(f"[{model_tag}] Baseline accuracy: {baseline_results['accuracy']:.2f}%")

        if wandb_run:
            wandb_run.finish()
        return

    vector = None
    method = None

    if args.vector_path:
        print(f"\nLoading vector from {args.vector_path}")
        loaded = torch.load(args.vector_path, map_location="cpu")
        vector = loaded["vector"] if isinstance(loaded, dict) and "vector" in loaded else loaded
        print(f"Loaded vector: shape={vector.shape}, norm={vector.norm().item():.4f}")

    elif args.mode in ["extract", "train", "both"]:
        if args.method == "extracted":
            method = ExtractedCoTVector(model_wrapper, tokenizer, args.layer_idx, args.dataset)
            vector = method.extract(support_samples)
        elif args.method == "learnable":
            method = LearnableCoTVector(
                model_wrapper=model_wrapper,
                tokenizer=tokenizer,
                layer_idx=args.layer_idx,
                dataset_type=args.dataset,
                lambda_val=args.lambda_val,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                warmup_ratio=args.warmup_ratio,
                num_epochs=args.num_epochs,
                batch_size=args.batch_size,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                max_length=args.max_length,
            )
            vector = method.train(support_samples, wandb_run)
        elif args.method == "ua":
            method = UACoTVector(
                model_wrapper=model_wrapper,
                tokenizer=tokenizer,
                layer_idx=args.layer_idx,
                dataset_type=args.dataset,
                tau_squared=args.tau_squared,
                min_variance=args.min_variance,
            )
            vector = method.extract(support_samples)

        if args.save_vector and vector is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            vector_path = os.path.join(output_dir, f"{args.method}_{model_tag}_L{args.layer_idx}_{timestamp}.pt")
            save_data = {"vector": vector.cpu(), "args": vars(args), "method": args.method}
            if args.method == "ua" and hasattr(method, "get_statistics"):
                save_data["statistics"] = method.get_statistics()
            torch.save(save_data, vector_path)
            print(f"Vector saved to {vector_path}")

    if args.mode in ["eval", "both"] and test_samples:
        baseline_results = None
        if not args.skip_baseline:
            baseline_results = run_baseline_evaluation(
                model_wrapper=model_wrapper,
                tokenizer=tokenizer,
                test_samples=test_samples,
                dataset_type=args.dataset,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
                use_early_stopping=args.use_early_stopping,
            )

        injection_results = None
        if vector is not None:
            injection_results = run_injection_evaluation(
                model_wrapper=model_wrapper,
                tokenizer=tokenizer,
                test_samples=test_samples,
                vector=vector,
                layer_idx=args.layer_idx,
                dataset_type=args.dataset,
                scaling_factor=args.scaling_factor,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
                use_early_stopping=args.use_early_stopping,
            )

        print("\n" + "=" * 60)
        print(f"Results Summary [{model_tag}]")
        print("-" * 60)
        if baseline_results:
            print(f"Baseline:   {baseline_results['accuracy']:.2f}% ({baseline_results['correct']}/{baseline_results['total']})")
        if injection_results:
            print(f"Injection:  {injection_results['accuracy']:.2f}% ({injection_results['correct']}/{injection_results['total']})")

    if wandb_run:
        wandb_run.finish()


def main():
    args = parse_args()
    set_seed(args.seed)

    output_dir = get_output_dir(args.output_dir, args.dataset)

    print("\nLoading data...")
    support_samples = None
    test_samples = None
    if args.mode in ["extract", "train", "both"]:
        support_samples = load_dataset(args.data_path, args.dataset, "train", args.num_support_samples)
        print(f"Support set: {len(support_samples)} samples")
    if args.mode in ["eval", "both"]:
        test_samples = load_dataset(args.data_path, args.dataset, "test", args.num_test_samples)
        print(f"Test set: {len(test_samples)} samples")

    model_specs = parse_model_specs(args.model_path, args.model_name)
    print(f"\nResolved {len(model_specs)} model(s): {model_specs}")

    for model_path, model_name in model_specs:
        run_args = copy.deepcopy(args)
        run_args.model_path = model_path
        run_args.model_name = model_name
        run_single_model(run_args, output_dir, support_samples, test_samples)

    print("\nDone!")


if __name__ == "__main__":
    main()
