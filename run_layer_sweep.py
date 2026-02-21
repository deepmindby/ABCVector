#!/usr/bin/env python3
"""
Layer sweep script for CoT Vectors.
Evaluates injection at different layers to find optimal performance.

Supports methods: extracted, learnable, ua (uncertainty-aware), abc (adaptive bayesian)

All hyperparameters are defined in src/args.py
"""

import os
import torch
from datetime import datetime
from tqdm import tqdm

from src.args import parse_args
from src.models import CoTModelWrapper, load_tokenizer
from src.data_utils import load_dataset
from src.methods.extracted import ExtractedCoTVector
from src.methods.learnable import LearnableCoTVector
from src.methods.ua_vector import UACoTVector
from src.methods.abc_vector import ABCCoTVector
from src.eval import run_baseline_evaluation, run_injection_evaluation
from src.utils import set_seed


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
    print("=" * 70)
    print("CoT Vector Layer Sweep")
    print("=" * 70)
    print(f"Model:    {args.model_path.split('/')[-1]}")
    print(f"Method:   {args.method}")
    print(f"Dataset:  {args.dataset}")
    print(f"Output:   {output_dir}")
    print(f"Support:  {args.num_support_samples}, Test: {args.num_test_samples}")
    print(f"Skip baseline: {args.skip_baseline}")
    if args.method == "learnable":
        print(f"Learnable Config: epochs={args.num_epochs}, batch={args.batch_size}, "
              f"grad_accum={args.gradient_accumulation_steps}, warmup={args.warmup_ratio}, "
              f"lr={args.learning_rate} (tiered), λ={args.lambda_val}, max_len={args.max_length}")
    if args.method == "ua":
        print(f"UA Config: τ²={args.tau_squared}, min_var={args.min_variance}")
    if args.method == "abc":
        print(f"ABC Config: hidden_dim={args.abc_hidden_dim}, kl_beta={args.kl_beta}, "
              f"kl_warmup={args.kl_warmup_steps}, sigma_min={args.sigma_min}, "
              f"lr={args.abc_learning_rate}, epochs={args.num_epochs}, "
              f"batch={args.batch_size}, grad_accum={args.gradient_accumulation_steps}")
    print("=" * 70)
    
    # Load model
    print("\nLoading model...")
    model_wrapper = CoTModelWrapper(args.model_path, args.model_name)
    tokenizer = load_tokenizer(args.model_path, args.model_name)
    num_layers = model_wrapper.num_layers
    print(f"Model has {num_layers} layers")
    
    # Load data
    print("\nLoading data...")
    support_samples = load_dataset(args.data_path, args.dataset, "train", args.num_support_samples)
    test_samples = load_dataset(args.data_path, args.dataset, "test", args.num_test_samples)
    print(f"Support: {len(support_samples)}, Test: {len(test_samples)}")
    
    # Determine layers to test
    if args.layers:
        layers = [int(l.strip()) for l in args.layers.split(",")]
    else:
        layers = list(range(0, num_layers, args.layer_step))
    
    print(f"\nLayers to test: {layers}")
    print(f"Total: {len(layers)} layers")
    
    # Baseline evaluation (optional)
    baseline_accuracy = args.baseline_accuracy
    
    if not args.skip_baseline:
        print("\n" + "-" * 70)
        print("Running baseline evaluation...")
        baseline = run_baseline_evaluation(
            model_wrapper, tokenizer, test_samples, args.dataset,
            args.max_new_tokens, args.num_beams, args.use_early_stopping
        )
        baseline_accuracy = baseline['accuracy']
        print(f"Baseline accuracy: {baseline_accuracy:.2f}%")
    elif baseline_accuracy is not None:
        print(f"\nUsing provided baseline accuracy: {baseline_accuracy:.2f}%")
    else:
        print("\nSkipping baseline evaluation (no baseline_accuracy provided)")
        baseline_accuracy = 0.0
    
    # Layer sweep
    print("\n" + "-" * 70)
    print(f"Testing {len(layers)} layers with method: {args.method}")
    print("-" * 70)
    
    results = []
    vectors_dict = {}
    
    for layer_idx in tqdm(layers, desc="Layer sweep", ncols=100):
        print(f"\n>>> Layer {layer_idx}")
        
        vector = None
        method = None
        
        # ==================== ABC Method (special handling) ====================
        if args.method == "abc":
            try:
                # Must reinitialize ABC for each layer (fresh prior/posterior/gate)
                abc_method = ABCCoTVector(
                    model_wrapper=model_wrapper,
                    tokenizer=tokenizer,
                    layer_idx=layer_idx,
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
                
                # Check if we should load a pre-trained checkpoint
                if args.load_vectors_dir:
                    checkpoint_path = os.path.join(
                        args.load_vectors_dir,
                        f"abc_L{layer_idx}.pt"
                    )
                    if os.path.exists(checkpoint_path):
                        print(f"  Loading ABC checkpoint from {checkpoint_path}")
                        checkpoint = torch.load(checkpoint_path, map_location="cpu")
                        target_device = model_wrapper.device
                        abc_method.load_state_dict(checkpoint, device=target_device)
                    else:
                        print(f"  Checkpoint not found, training new ABC model...")
                        abc_method.train(support_samples)
                else:
                    # Train ABC
                    abc_method.train(support_samples)
                
                # Save checkpoint if requested
                if args.save_vector:
                    checkpoint_path = os.path.join(
                        output_dir,
                        f"abc_L{layer_idx}.pt"
                    )
                    save_data = {
                        **abc_method.get_state_dict(),
                        "args": vars(args),
                    }
                    torch.save(save_data, checkpoint_path)
                    vectors_dict[layer_idx] = checkpoint_path
                    print(f"  Saved ABC checkpoint to {checkpoint_path}")
                
                # Evaluate ABC
                abc_results = abc_method.eval(
                    test_samples=test_samples,
                    max_new_tokens=args.max_new_tokens,
                    num_beams=args.num_beams,
                    use_early_stopping=args.use_early_stopping,
                )
                
                diff = abc_results['accuracy'] - baseline_accuracy if baseline_accuracy else 0
                gate_val = abc_method.gate.item()
                
                result_entry = {
                    'layer': layer_idx,
                    'accuracy': abc_results['accuracy'],
                    'diff': diff,
                    'correct': abc_results['correct'],
                    'total': abc_results['total'],
                    'gate': gate_val,
                }
                
                results.append(result_entry)
                
                print(f"  Layer {layer_idx:2d}: {abc_results['accuracy']:.2f}% "
                      f"({abc_results['correct']}/{abc_results['total']}) "
                      f"[{diff:+.2f}% vs baseline] gate={gate_val:.3f}")
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                    print(f"  Layer {layer_idx}: CUDA OOM - {e}")
                    torch.cuda.empty_cache()
                    results.append({
                        'layer': layer_idx,
                        'accuracy': 0,
                        'diff': -baseline_accuracy if baseline_accuracy else 0,
                        'error': f"CUDA OOM: {str(e)[:100]}",
                    })
                    continue
                else:
                    print(f"  Layer {layer_idx}: Error - {e}")
                    results.append({
                        'layer': layer_idx,
                        'accuracy': 0,
                        'diff': -baseline_accuracy if baseline_accuracy else 0,
                        'error': str(e),
                    })
                    continue
            except Exception as e:
                print(f"  Layer {layer_idx}: Error - {e}")
                results.append({
                    'layer': layer_idx,
                    'accuracy': 0,
                    'diff': -baseline_accuracy if baseline_accuracy else 0,
                    'error': str(e),
                })
                continue
            
            # Clear CUDA cache after each layer
            torch.cuda.empty_cache()
            continue
        
        # ==================== Other Methods (extracted, learnable, ua) ====================
        # Check if we should load a pre-trained vector
        if args.load_vectors_dir:
            vector_path = os.path.join(
                args.load_vectors_dir,
                f"{args.method}_L{layer_idx}.pt"
            )
            if os.path.exists(vector_path):
                print(f"  Loading vector from {vector_path}")
                loaded = torch.load(vector_path, map_location="cpu")
                if isinstance(loaded, dict) and "vector" in loaded:
                    vector = loaded["vector"]
                else:
                    vector = loaded
            else:
                print(f"  Vector not found at {vector_path}, extracting new one...")
        
        # Extract/Train vector if not loaded
        if vector is None:
            try:
                if args.method == "extracted":
                    method = ExtractedCoTVector(
                        model_wrapper=model_wrapper,
                        tokenizer=tokenizer,
                        layer_idx=layer_idx,
                        dataset_type=args.dataset,
                    )
                    vector = method.extract(support_samples)
                    
                elif args.method == "learnable":
                    method = LearnableCoTVector(
                        model_wrapper=model_wrapper,
                        tokenizer=tokenizer,
                        layer_idx=layer_idx,
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
                    vector = method.train(support_samples)
                    
                elif args.method == "ua":
                    method = UACoTVector(
                        model_wrapper=model_wrapper,
                        tokenizer=tokenizer,
                        layer_idx=layer_idx,
                        dataset_type=args.dataset,
                        tau_squared=args.tau_squared,
                        min_variance=args.min_variance,
                    )
                    vector = method.extract(support_samples)
                    
            except RuntimeError as e:
                if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                    print(f"  Layer {layer_idx}: CUDA OOM - {e}")
                    torch.cuda.empty_cache()
                    results.append({
                        'layer': layer_idx,
                        'accuracy': 0,
                        'diff': -baseline_accuracy if baseline_accuracy else 0,
                        'error': f"CUDA OOM: {str(e)[:100]}",
                    })
                    continue
                else:
                    print(f"  Layer {layer_idx}: Training Error - {e}")
                    results.append({
                        'layer': layer_idx,
                        'accuracy': 0,
                        'diff': -baseline_accuracy if baseline_accuracy else 0,
                        'error': str(e),
                    })
                    continue
            except Exception as e:
                print(f"  Layer {layer_idx}: Error - {e}")
                results.append({
                    'layer': layer_idx,
                    'accuracy': 0,
                    'diff': -baseline_accuracy if baseline_accuracy else 0,
                    'error': str(e),
                })
                continue
        
        # Save vector if requested (to outputs/{dataset}/)
        if args.save_vector and vector is not None:
            vector_path = os.path.join(
                output_dir,
                f"{args.method}_L{layer_idx}.pt"
            )
            save_data = {"vector": vector.cpu(), "layer": layer_idx, "method": args.method}
            
            # Include UA statistics if available
            if args.method == "ua" and method is not None and hasattr(method, 'get_statistics'):
                save_data["statistics"] = method.get_statistics()
            
            torch.save(save_data, vector_path)
            vectors_dict[layer_idx] = vector_path
            print(f"  Saved vector to {vector_path}, norm={vector.norm().item():.4f}")
        
        # Evaluate
        try:
            layer_results = run_injection_evaluation(
                model_wrapper, tokenizer, test_samples, vector,
                layer_idx, args.dataset, 1.0, args.max_new_tokens, 
                args.num_beams, args.use_early_stopping
            )
            
            diff = layer_results['accuracy'] - baseline_accuracy if baseline_accuracy else 0
            vec_norm = vector.norm().item() if vector is not None else 0
            
            result_entry = {
                'layer': layer_idx,
                'accuracy': layer_results['accuracy'],
                'diff': diff,
                'correct': layer_results['correct'],
                'total': layer_results['total'],
                'vector_norm': vec_norm,
            }
            
            results.append(result_entry)
            
            print(f"  Layer {layer_idx:2d}: {layer_results['accuracy']:.2f}% "
                  f"({layer_results['correct']}/{layer_results['total']}) "
                  f"[{diff:+.2f}% vs baseline] norm={vec_norm:.2f}")
            
        except Exception as e:
            print(f"  Layer {layer_idx:2d}: Evaluation Error - {e}")
            results.append({
                'layer': layer_idx,
                'accuracy': 0,
                'diff': -baseline_accuracy if baseline_accuracy else 0,
                'error': str(e),
            })
        
        # Clear CUDA cache after each layer
        torch.cuda.empty_cache()
    
    # Summary
    print("\n" + "=" * 70)
    print("Results Summary")
    print("=" * 70)
    print(f"Method:   {args.method}")
    print(f"Dataset:  {args.dataset}")
    if baseline_accuracy:
        print(f"Baseline: {baseline_accuracy:.2f}%")
    print("-" * 70)
    
    # Filter out errors
    valid_results = [r for r in results if 'error' not in r]
    
    avg_accuracy = 0.0
    avg_metric = 0.0  # norm for others, gate for ABC
    
    if valid_results:
        # Sort by accuracy
        valid_results.sort(key=lambda x: x['accuracy'], reverse=True)
        
        # Layer-wise average
        avg_accuracy = sum(r['accuracy'] for r in valid_results) / len(valid_results)
        
        if args.method == "abc":
            avg_metric = sum(r.get('gate', 0) for r in valid_results) / len(valid_results)
            metric_name = "Average Gate Value"
        else:
            avg_metric = sum(r.get('vector_norm', 0) for r in valid_results) / len(valid_results)
            metric_name = "Average Vector Norm"
        
        print(f"\nLayer-wise Average: {avg_accuracy:.2f}%")
        print(f"{metric_name}: {avg_metric:.2f}")
        
        print("\nTop 5 layers:")
        for r in valid_results[:5]:
            diff_str = f"({r['diff']:+.2f}%)" if baseline_accuracy else ""
            if args.method == "abc":
                metric_str = f"gate={r.get('gate', 0):.3f}"
            else:
                metric_str = f"norm={r.get('vector_norm', 0):.1f}"
            print(f"  Layer {r['layer']:2d}: {r['accuracy']:.2f}% {diff_str} {metric_str}")
        
        if len(valid_results) > 5:
            print("\nBottom 5 layers:")
            for r in valid_results[-5:]:
                diff_str = f"({r['diff']:+.2f}%)" if baseline_accuracy else ""
                if args.method == "abc":
                    metric_str = f"gate={r.get('gate', 0):.3f}"
                else:
                    metric_str = f"norm={r.get('vector_norm', 0):.1f}"
                print(f"  Layer {r['layer']:2d}: {r['accuracy']:.2f}% {diff_str} {metric_str}")
        
        # Best layer
        best = valid_results[0]
        print(f"\n★ Best Layer: {best['layer']} with {best['accuracy']:.2f}%")
        if baseline_accuracy:
            print(f"  Improvement over baseline: {best['diff']:+.2f}%")
        if args.method == "abc":
            print(f"  Gate value: {best.get('gate', 0):.4f}")
        else:
            print(f"  Vector norm: {best.get('vector_norm', 0):.2f}")
    
    # Error summary
    error_results = [r for r in results if 'error' in r]
    if error_results:
        print(f"\nErrors encountered in {len(error_results)} layers:")
        for r in error_results:
            print(f"  Layer {r['layer']}: {r['error'][:50]}...")
    
    # Save results to file (in outputs/{dataset}/)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(
        output_dir, 
        f"layer_sweep_{args.method}_{timestamp}.txt"
    )
    
    with open(result_file, "w") as f:
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Method: {args.method}\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Baseline: {baseline_accuracy:.2f}%\n" if baseline_accuracy else "Baseline: N/A\n")
        f.write(f"Support samples: {args.num_support_samples}\n")
        f.write(f"Test samples: {args.num_test_samples}\n")
        
        if args.method == "learnable":
            f.write(f"\nLearnable Config:\n")
            f.write(f"  Epochs: {args.num_epochs}\n")
            f.write(f"  Batch size: {args.batch_size}\n")
            f.write(f"  Grad accum: {args.gradient_accumulation_steps}\n")
            f.write(f"  Learning rate: {args.learning_rate} (tiered by model/layer)\n")
            f.write(f"  Lambda: {args.lambda_val}\n")
            f.write(f"  Warmup ratio: {args.warmup_ratio}\n")
            f.write(f"  Max length: {args.max_length}\n")
        
        if args.method == "ua":
            f.write(f"\nUA Config:\n")
            f.write(f"  Tau squared: {args.tau_squared}\n")
            f.write(f"  Min variance: {args.min_variance}\n")
        
        if args.method == "abc":
            f.write(f"\nABC Config:\n")
            f.write(f"  Hidden dim: {args.abc_hidden_dim}\n")
            f.write(f"  KL beta: {args.kl_beta}\n")
            f.write(f"  KL warmup steps: {args.kl_warmup_steps}\n")
            f.write(f"  Sigma min: {args.sigma_min}\n")
            f.write(f"  Learning rate: {args.abc_learning_rate}\n")
            f.write(f"  Epochs: {args.num_epochs}\n")
            f.write(f"  Batch size: {args.batch_size}\n")
            f.write(f"  Grad accum: {args.gradient_accumulation_steps}\n")
        
        if args.method == "abc":
            f.write(f"\nLayer\tAccuracy\tDiff\tCorrect\tTotal\tGate\n")
        else:
            f.write(f"\nLayer\tAccuracy\tDiff\tCorrect\tTotal\tNorm\n")
        f.write("-" * 70 + "\n")
        
        for r in sorted(results, key=lambda x: x['layer']):
            if 'error' in r:
                f.write(f"{r['layer']}\tERROR\t-\t-\t-\t-\t{r['error'][:30]}\n")
            else:
                if args.method == "abc":
                    f.write(f"{r['layer']}\t{r['accuracy']:.2f}\t{r['diff']:+.2f}\t"
                            f"{r['correct']}\t{r['total']}\t{r.get('gate', 0):.4f}\n")
                else:
                    f.write(f"{r['layer']}\t{r['accuracy']:.2f}\t{r['diff']:+.2f}\t"
                            f"{r['correct']}\t{r['total']}\t{r.get('vector_norm', 0):.2f}\n")
        
        # Summary at the end
        if valid_results:
            f.write(f"\n" + "=" * 70 + "\n")
            f.write(f"Best Layer: {valid_results[0]['layer']} ({valid_results[0]['accuracy']:.2f}%)\n")
            f.write(f"Layer-wise Average: {avg_accuracy:.2f}%\n")
            if args.method == "abc":
                f.write(f"Average Gate Value: {avg_metric:.4f}\n")
            else:
                f.write(f"Average Vector Norm: {avg_metric:.2f}\n")
    
    print(f"\nResults saved to {result_file}")
    
    # Save vectors/checkpoint paths if saved
    if args.save_vector and vectors_dict:
        if args.method == "abc":
            paths_file = os.path.join(
                output_dir,
                f"checkpoints_paths_{args.method}_{timestamp}.txt"
            )
        else:
            paths_file = os.path.join(
                output_dir,
                f"vectors_paths_{args.method}_{timestamp}.txt"
            )
        with open(paths_file, "w") as f:
            for layer, path in sorted(vectors_dict.items()):
                f.write(f"Layer {layer}: {path}\n")
        print(f"{'Checkpoint' if args.method == 'abc' else 'Vector'} paths saved to {paths_file}")
    
    print("=" * 70)
    print("Done!")


if __name__ == "__main__":
    main()