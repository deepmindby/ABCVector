#!/usr/bin/env python3
"""
Main entry point for CoT Vectors.

Supports methods: extracted, learnable, ua, abc, habc
All hyperparameters are defined in src/args.py

Usage:
    python main.py --method abc --dataset gsm8k --layer_idx 0
    python main.py --method abc --posterior_mode q_y_qc --save_diagnostics
    python main.py --method habc --dataset gsm8k --layer_idx 0 --global_latent_dim 256
"""

import os
import sys
import json
import torch
from datetime import datetime

from src.args import parse_args
from src.models import CoTModelWrapper, load_tokenizer
from src.data_utils import load_dataset
from src.methods.extracted import ExtractedCoTVector
from src.methods.learnable import LearnableCoTVector
from src.methods.ua_vector import UACoTVector
from src.methods.abc_vector import ABCCoTVector
from src.methods.habc_vector import HierarchicalABCCoTVector
from src.eval import run_baseline_evaluation, run_injection_evaluation
from src.utils import set_seed, print_results_summary


def get_output_dir(base_dir: str, dataset: str) -> str:
    """Get dataset-specific output directory: output_dir/{dataset}/"""
    output_dir = os.path.join(base_dir, dataset)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def main():
    args = parse_args()
    
    set_seed(args.seed)
    
    # Create dataset-specific output directory
    output_dir = get_output_dir(args.output_dir, args.dataset)
    
    # Print configuration
    print("=" * 60)
    print("CoT Vectors")
    print("=" * 60)
    print(f"Model:    {args.model_path.split('/')[-1]}")
    print(f"Method:   {args.method}")
    print(f"Dataset:  {args.dataset}")
    print(f"Output:   {output_dir}")
    print(f"Mode:     {args.mode}")
    if args.method == "abc":
        print(f"ABC Config: hidden_dim={args.abc_hidden_dim}, kl_beta={args.kl_beta}, "
              f"kl_warmup={args.kl_warmup_steps}, sigma_min={args.sigma_min}, "
              f"lr={args.abc_learning_rate}, posterior_mode={args.posterior_mode}")
    if args.method == "habc":
        print(f"HABC Config: hidden_dim={args.habc_hidden_dim}, "
              f"global_dim={args.global_latent_dim}, instance_dim={args.instance_latent_dim}, "
              f"beta_g={args.kl_beta_global}, beta_i={args.kl_beta_instance}, "
              f"lr={args.habc_learning_rate}, posterior_mode={args.posterior_mode}")
    print("=" * 60)
    
    # WandB
    wandb_run = None
    if args.use_wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project,
                config=vars(args),
            )
        except ImportError:
            print("Warning: wandb not installed, skipping logging")
    
    # Load model
    print("\nLoading model...")
    model_wrapper = CoTModelWrapper(args.model_path, args.model_name)
    tokenizer = load_tokenizer(args.model_path)
    print(f"Model loaded. Hidden size: {model_wrapper.hidden_size}, "
          f"Layers: {model_wrapper.num_layers}")
    
    # Load data
    support_samples = None
    test_samples = None
    
    if args.mode in ["extract", "train", "both"]:
        support_samples = load_dataset(
            args.data_path, args.dataset, "train", args.num_support_samples
        )
        print(f"Support set: {len(support_samples)} samples")
    
    if args.mode in ["eval", "both"]:
        test_samples = load_dataset(
            args.data_path, args.dataset, "test", args.num_test_samples
        )
        print(f"Test set: {len(test_samples)} samples")
    
    # ==================== Handle ABC method separately ====================
    if args.method == "abc":
        print(f"\n{'='*60}")
        print("ABC Vector Processing")
        print("=" * 60)
        print(f"  Posterior mode: {args.posterior_mode}")
        if args.save_diagnostics:
            print(f"  Diagnostics: ON (split={args.diagnostic_split})")
        if args.run_prior_eval:
            print(f"  Prior eval: ON")
        if args.run_posterior_eval:
            print(f"  Posterior eval: ON")
        
        # Diagnostics output directory
        diagnostics_dir = os.path.join(output_dir, "diagnostics")
        os.makedirs(diagnostics_dir, exist_ok=True)
        
        # Initialize ABC method
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
            # New: posterior mode and diagnostics
            posterior_mode=args.posterior_mode,
            save_diagnostics=args.save_diagnostics,
            diagnostics_dir=diagnostics_dir,
        )
        
        # Load checkpoint if provided
        if args.abc_checkpoint_path:
            print(f"\nLoading ABC checkpoint from {args.abc_checkpoint_path}")
            checkpoint = torch.load(args.abc_checkpoint_path, map_location="cpu")
            target_device = model_wrapper.device
            abc_method.load_state_dict(checkpoint, device=target_device)
            print("ABC checkpoint loaded successfully")
        
        # Training
        if args.mode in ["train", "both"] and support_samples:
            print("\nTraining ABC Vector...")
            abc_method.train(support_samples, wandb_run)
            
            # Save checkpoint
            if args.save_vector:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                checkpoint_filename = f"abc_L{args.layer_idx}_{args.posterior_mode}_{timestamp}.pt"
                checkpoint_path = os.path.join(output_dir, checkpoint_filename)
                
                save_data = {
                    **abc_method.get_state_dict(),
                    "args": vars(args),
                }
                torch.save(save_data, checkpoint_path)
                print(f"ABC checkpoint saved to {checkpoint_path}")
        
        # Evaluation
        if args.mode in ["eval", "both"] and test_samples:
            print(f"\n{'='*60}")
            print("Evaluation")
            print("=" * 60)
            
            # Baseline evaluation
            baseline_results = None
            if not args.skip_baseline:
                print("\n[1/N] Baseline (no injection)...")
                baseline_results = run_baseline_evaluation(
                    model_wrapper=model_wrapper,
                    tokenizer=tokenizer,
                    test_samples=test_samples,
                    dataset_type=args.dataset,
                    max_new_tokens=args.max_new_tokens,
                    num_beams=args.num_beams,
                    use_early_stopping=args.use_early_stopping,
                )
            
            # Standard ABC evaluation (prior mean)
            print(f"\n[2/N] ABC Vector (layer {args.layer_idx}, prior mean z*)...")
            abc_results = abc_method.eval(
                test_samples=test_samples,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
                use_early_stopping=args.use_early_stopping,
            )
            
            # Print results
            print("\n" + "=" * 60)
            print("Results Summary")
            print("-" * 60)
            print(f"Model:      {args.model_path.split('/')[-1]}")
            print(f"Method:     ABC (posterior_mode={args.posterior_mode})")
            print(f"Layer:      {args.layer_idx}")
            print(f"Dataset:    {args.dataset}")
            print(f"Test size:  {len(test_samples)}")
            print("-" * 60)
            
            if baseline_results:
                print(f"Baseline:   {baseline_results['accuracy']:.2f}% "
                      f"({baseline_results['correct']}/{baseline_results['total']})")
            
            if baseline_results:
                diff = abc_results['accuracy'] - baseline_results['accuracy']
                sign = "+" if diff >= 0 else ""
                print(f"ABC(prior): {abc_results['accuracy']:.2f}% "
                      f"({abc_results['correct']}/{abc_results['total']}) [{sign}{diff:.2f}%]")
            else:
                print(f"ABC(prior): {abc_results['accuracy']:.2f}% "
                      f"({abc_results['correct']}/{abc_results['total']})")
            
            print(f"Gate value: {abc_method.gate.item():.4f}")
            print(f"Avg injected norm: {abc_results.get('avg_injected_norm', 0):.4f}")
            print("=" * 60)
            
            # ========== Diagnostic Eval (optional) ==========
            if args.run_prior_eval or args.run_posterior_eval:
                diag_csv_path = os.path.join(
                    diagnostics_dir,
                    f"diagnostic_eval_L{args.layer_idx}_{args.posterior_mode}.csv"
                )
                
                # Determine which splits to evaluate
                eval_splits = {}
                if args.diagnostic_split in ["test", "both"]:
                    eval_splits["test"] = test_samples
                if args.diagnostic_split in ["support", "both"] and support_samples:
                    # Use a subset of support for diagnostic eval
                    diag_support = support_samples[:min(100, len(support_samples))]
                    eval_splits["support"] = diag_support
                
                for split_name, split_samples in eval_splits.items():
                    abc_method.run_diagnostic_eval(
                        test_samples=split_samples,
                        max_new_tokens=args.max_new_tokens,
                        num_beams=args.num_beams,
                        split_name=split_name,
                        save_path=diag_csv_path,
                    )
            
            # Log to WandB
            if wandb_run:
                if baseline_results:
                    wandb_run.log({
                        "eval/baseline_accuracy": baseline_results['accuracy'],
                    })
                wandb_run.log({
                    "eval/abc_accuracy": abc_results['accuracy'],
                    "eval/gate": abc_method.gate.item(),
                    "eval/posterior_mode": args.posterior_mode,
                })
                if baseline_results:
                    wandb_run.log({
                        "eval/improvement": abc_results['accuracy'] - baseline_results['accuracy'],
                    })
                wandb_run.finish()
        
        print("\nDone!")
        return
    
    # ==================== Handle HABC method separately ====================
    if args.method == "habc":
        print(f"\n{'='*60}")
        print("HABC Vector Processing (Hierarchical)")
        print("=" * 60)
        print(f"  Global latent dim: {args.global_latent_dim}")
        print(f"  Instance latent dim: {args.instance_latent_dim}")
        print(f"  Posterior mode: {args.posterior_mode}")
        if args.save_diagnostics:
            print(f"  Diagnostics: ON")
        
        # Diagnostics output directory
        diagnostics_dir = os.path.join(output_dir, "diagnostics")
        os.makedirs(diagnostics_dir, exist_ok=True)
        
        # Initialize HABC method
        habc_method = HierarchicalABCCoTVector(
            model_wrapper=model_wrapper,
            tokenizer=tokenizer,
            layer_idx=args.layer_idx,
            dataset_type=args.dataset,
            habc_hidden_dim=args.habc_hidden_dim,
            global_latent_dim=args.global_latent_dim,
            instance_latent_dim=args.instance_latent_dim,
            kl_beta_global=args.kl_beta_global,
            kl_beta_instance=args.kl_beta_instance,
            kl_warmup_steps_global=args.kl_warmup_steps_global,
            kl_warmup_steps_instance=args.kl_warmup_steps_instance,
            sigma_min_global=args.sigma_min_global,
            sigma_min_instance=args.sigma_min_instance,
            learning_rate=args.habc_learning_rate,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_length=args.max_length,
            posterior_mode=args.posterior_mode,
            save_diagnostics=args.save_diagnostics,
            diagnostics_dir=diagnostics_dir,
        )
        
        # Load checkpoint if provided
        if args.habc_checkpoint_path:
            print(f"\nLoading HABC checkpoint from {args.habc_checkpoint_path}")
            checkpoint = torch.load(args.habc_checkpoint_path, map_location="cpu")
            target_device = model_wrapper.device
            habc_method.load_state_dict(checkpoint, device=target_device)
            print("HABC checkpoint loaded successfully")
        
        # Training
        if args.mode in ["train", "both"] and support_samples:
            print("\nTraining HABC Vector...")
            habc_method.train(support_samples, wandb_run)
            
            # Save checkpoint
            if args.save_vector:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                checkpoint_filename = f"habc_ckpt_layer{args.layer_idx}_{timestamp}.pt"
                checkpoint_path = os.path.join(output_dir, checkpoint_filename)
                
                save_data = {
                    **habc_method.get_state_dict(),
                    "args": vars(args),
                }
                torch.save(save_data, checkpoint_path)
                print(f"HABC checkpoint saved to {checkpoint_path}")
        
        # Evaluation
        if args.mode in ["eval", "both"] and test_samples:
            print(f"\n{'='*60}")
            print("Evaluation")
            print("=" * 60)
            
            # Baseline evaluation
            baseline_results = None
            if not args.skip_baseline:
                print("\n[1/2] Baseline (no injection)...")
                baseline_results = run_baseline_evaluation(
                    model_wrapper=model_wrapper,
                    tokenizer=tokenizer,
                    test_samples=test_samples,
                    dataset_type=args.dataset,
                    max_new_tokens=args.max_new_tokens,
                    num_beams=args.num_beams,
                    use_early_stopping=args.use_early_stopping,
                )
            
            # HABC evaluation
            print(f"\n[2/2] HABC Vector (layer {args.layer_idx})...")
            habc_results = habc_method.eval(
                test_samples=test_samples,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
                use_early_stopping=args.use_early_stopping,
            )
            
            # Print results
            print("\n" + "=" * 60)
            print("Results Summary")
            print("-" * 60)
            print(f"Model:      {args.model_path.split('/')[-1]}")
            print(f"Method:     HABC (posterior_mode={args.posterior_mode})")
            print(f"Layer:      {args.layer_idx}")
            print(f"Dataset:    {args.dataset}")
            print(f"Test size:  {len(test_samples)}")
            print("-" * 60)
            
            if baseline_results:
                print(f"Baseline:   {baseline_results['accuracy']:.2f}% "
                      f"({baseline_results['correct']}/{baseline_results['total']})")
            
            if baseline_results:
                diff = habc_results['accuracy'] - baseline_results['accuracy']
                sign = "+" if diff >= 0 else ""
                print(f"HABC:       {habc_results['accuracy']:.2f}% "
                      f"({habc_results['correct']}/{habc_results['total']}) [{sign}{diff:.2f}%]")
            else:
                print(f"HABC:       {habc_results['accuracy']:.2f}% "
                      f"({habc_results['correct']}/{habc_results['total']})")
            
            print(f"Avg injected norm: {habc_results.get('avg_injected_norm', 0):.4f}")
            print("=" * 60)
            
            # Save results
            if args.save_vector:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                results_filename = f"habc_results_layer{args.layer_idx}_{timestamp}.json"
                results_path = os.path.join(output_dir, results_filename)
                
                results_to_save = {
                    "method": "habc",
                    "layer_idx": args.layer_idx,
                    "dataset": args.dataset,
                    "posterior_mode": args.posterior_mode,
                    "accuracy": habc_results["accuracy"],
                    "correct": habc_results["correct"],
                    "total": habc_results["total"],
                    "avg_injected_norm": habc_results.get("avg_injected_norm", 0),
                    "baseline_accuracy": baseline_results["accuracy"] if baseline_results else None,
                    "config": {
                        "habc_hidden_dim": args.habc_hidden_dim,
                        "global_latent_dim": args.global_latent_dim,
                        "instance_latent_dim": args.instance_latent_dim,
                        "kl_beta_global": args.kl_beta_global,
                        "kl_beta_instance": args.kl_beta_instance,
                        "habc_learning_rate": args.habc_learning_rate,
                        "num_epochs": args.num_epochs,
                    },
                    "timestamp": timestamp,
                }
                with open(results_path, "w") as f:
                    json.dump(results_to_save, f, indent=2)
                print(f"Results saved to {results_path}")
            
            # Log to WandB
            if wandb_run:
                if baseline_results:
                    wandb_run.log({
                        "eval/baseline_accuracy": baseline_results['accuracy'],
                    })
                wandb_run.log({
                    "eval/habc_accuracy": habc_results['accuracy'],
                    "eval/posterior_mode": args.posterior_mode,
                })
                if baseline_results:
                    wandb_run.log({
                        "eval/improvement": habc_results['accuracy'] - baseline_results['accuracy'],
                    })
                wandb_run.finish()
        
        print("\nDone!")
        return
    
    # ==================== Handle non-ABC methods ====================
    # (extracted, learnable, ua — kept unchanged from original main.py)
    
    vector = None
    
    if args.mode in ["extract", "train", "both"]:
        print(f"\n{'='*60}")
        if args.method == "extracted":
            print("Extracting CoT Vector")
        else:
            print("Training CoT Vector")
        print("=" * 60)
        
        if args.method == "extracted":
            method = ExtractedCoTVector(model_wrapper, tokenizer, args.layer_idx, args.dataset)
            vector = method.extract(support_samples, scaling_factor=args.scaling_factor)
        elif args.method == "learnable":
            method = LearnableCoTVector(
                model_wrapper=model_wrapper,
                tokenizer=tokenizer,
                layer_idx=args.layer_idx,
                dataset_type=args.dataset,
                lambda_val=args.lambda_val,
                learning_rate=args.learning_rate,
                num_epochs=args.num_epochs,
                batch_size=args.batch_size,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                warmup_ratio=args.warmup_ratio,
                weight_decay=args.weight_decay,
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
        
        # Save vector
        if vector is not None and args.save_vector:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            vector_filename = f"{args.method}_L{args.layer_idx}_{timestamp}.pt"
            vector_path = os.path.join(output_dir, vector_filename)
            
            from src.utils import save_vector
            save_vector(vector, vector_path, metadata=vars(args))
            print(f"Vector saved to {vector_path}")
    
    # Load vector from file if specified
    if args.vector_path and vector is None:
        from src.utils import load_vector
        vector, _ = load_vector(args.vector_path)
        print(f"Vector loaded from {args.vector_path}")
    
    # Evaluation for non-ABC methods
    if args.mode in ["eval", "both"] and test_samples:
        print(f"\n{'='*60}")
        print("Evaluation")
        print("=" * 60)
        
        # Baseline evaluation
        baseline_results = None
        if not args.skip_baseline:
            print("\n[1/2] Baseline (no injection)...")
            baseline_results = run_baseline_evaluation(
                model_wrapper=model_wrapper,
                tokenizer=tokenizer,
                test_samples=test_samples,
                dataset_type=args.dataset,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
                use_early_stopping=args.use_early_stopping,
            )
        
        # Injection evaluation
        injection_results = None
        if vector is not None:
            print(f"\n[2/2] With CoT Vector (layer {args.layer_idx})...")
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
        
        # Print results
        print("\n" + "=" * 60)
        print("Results Summary")
        print("-" * 60)
        print(f"Model:      {args.model_path.split('/')[-1]}")
        print(f"Method:     {args.method}")
        print(f"Layer:      {args.layer_idx}")
        print(f"Dataset:    {args.dataset}")
        
        if baseline_results:
            print(f"Baseline:   {baseline_results['accuracy']:.2f}%")
        if injection_results:
            print(f"Injection:  {injection_results['accuracy']:.2f}%")
        print("=" * 60)
        
        if wandb_run:
            wandb_run.finish()
    
    print("\nDone!")


if __name__ == "__main__":
    main()