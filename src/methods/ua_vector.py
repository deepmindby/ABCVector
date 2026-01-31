"""
Uncertainty-Aware (UA) CoT Vector implementation.

Based on "Variational CoT Vectors" framework:
- Uses Bayesian MAP estimation with structured prior
- Implements adaptive gating mechanism based on variance
- Formula: z_d = k_d * μ_d where k_d = σ²_d / (σ²_d + τ²)

Key insight: Dimensions with high variance (noise) are suppressed,
while dimensions with low variance (stable signals) are preserved.
"""

import torch
from typing import List, Optional, Dict, Any
from tqdm import tqdm

from .base import BaseCoTVectorMethod
from ..models import CoTModelWrapper
from ..data_utils import PROMPT_TEMPLATES


class UACoTVector(BaseCoTVectorMethod):
    """
    Uncertainty-Aware CoT Vector with Bayesian shrinkage.
    
    Implements the MAP estimation framework:
    - Prior: p(z) = N(0, τ²I) - sparse prior favoring zero
    - Likelihood: p(μ|z) = N(z, σ²) - teacher distribution from support set
    - Posterior (MAP): z_d = k_d * μ_d where k_d = σ²_d / (σ²_d + τ²)
    
    The shrinkage coefficient k_d acts as an adaptive gate:
    - High variance (noise): k_d → 0, dimension suppressed
    - Low variance (signal): k_d → 1, dimension preserved
    """
    
    def __init__(
        self,
        model_wrapper: CoTModelWrapper,
        tokenizer,
        layer_idx: int,
        dataset_type: str = "gsm8k",
        tau_squared: float = 1.0,
        min_variance: float = 1e-6,
    ):
        """
        Initialize UA CoT Vector extractor.
        
        Args:
            model_wrapper: The CoTModelWrapper instance
            tokenizer: The tokenizer
            layer_idx: Layer index for extraction
            dataset_type: Dataset type (gsm8k, math, mmlu_pro)
            tau_squared: Prior variance τ² for Bayesian shrinkage
                        Smaller = stronger regularization toward zero
            min_variance: Minimum variance for numerical stability
        """
        super().__init__(model_wrapper, tokenizer, layer_idx, dataset_type)
        
        self.tau_squared = tau_squared
        self.min_variance = min_variance
        self.prompt_template = PROMPT_TEMPLATES.get(dataset_type, PROMPT_TEMPLATES["gsm8k"])
        
        # Store statistics for analysis
        self.mean_vector: Optional[torch.Tensor] = None
        self.variance_vector: Optional[torch.Tensor] = None
        self.shrinkage_coefficients: Optional[torch.Tensor] = None
    
    def extract_single(self, sample) -> torch.Tensor:
        """
        Extract activation difference for a single sample.
        
        Returns the difference: α_CoT(answer) - α_NonCoT(answer)
        This represents the "teacher hidden state" for this sample.
        """
        device = self.model_wrapper.device
        
        # Build prompts
        if self.dataset_type == "mmlu_pro":
            cot_prompt = self.prompt_template["cot"].format(
                question=sample.question,
                choices=sample.choices
            ) + sample.cot + f"\nThe answer is {sample.answer}"
            
            non_cot_prompt = self.prompt_template["non_cot"].format(
                question=sample.question,
                choices=sample.choices
            ) + f"The answer is {sample.answer}"
        else:
            cot_prompt = self.prompt_template["cot"].format(
                question=sample.question
            ) + sample.cot + f"\nThe answer is {sample.answer}"
            
            non_cot_prompt = self.prompt_template["non_cot"].format(
                question=sample.question
            ) + f"The answer is {sample.answer}"
        
        # Tokenize
        cot_encoding = self.tokenizer(cot_prompt, return_tensors="pt", truncation=True, max_length=2048)
        non_cot_encoding = self.tokenizer(non_cot_prompt, return_tensors="pt", truncation=True, max_length=2048)
        
        cot_ids = cot_encoding["input_ids"].to(device)
        non_cot_ids = non_cot_encoding["input_ids"].to(device)
        cot_mask = cot_encoding["attention_mask"].to(device)
        non_cot_mask = non_cot_encoding["attention_mask"].to(device)
        
        # Find answer token positions (last N tokens where N = answer length)
        answer_text = f"The answer is {sample.answer}"
        answer_ids = self.tokenizer(answer_text, add_special_tokens=False, return_tensors="pt")["input_ids"]
        answer_len = answer_ids.shape[1]
        
        # Get positions for answer tokens
        cot_answer_pos = list(range(cot_ids.shape[1] - answer_len, cot_ids.shape[1]))
        non_cot_answer_pos = list(range(non_cot_ids.shape[1] - answer_len, non_cot_ids.shape[1]))
        
        # Clear hooks
        self.model_wrapper.clear_hooks()
        
        # Extract CoT activations
        self.model_wrapper.register_extraction_hook(self.layer_idx)
        with torch.no_grad():
            self.model_wrapper(cot_ids, attention_mask=cot_mask)
        cot_activation = self.model_wrapper.get_activations(self.layer_idx)
        cot_answer_activation = cot_activation[:, cot_answer_pos, :].mean(dim=1)  # [1, hidden]
        
        # Clear and extract non-CoT activations
        self.model_wrapper.clear_hooks()
        self.model_wrapper.register_extraction_hook(self.layer_idx)
        with torch.no_grad():
            self.model_wrapper(non_cot_ids, attention_mask=non_cot_mask)
        non_cot_activation = self.model_wrapper.get_activations(self.layer_idx)
        non_cot_answer_activation = non_cot_activation[:, non_cot_answer_pos, :].mean(dim=1)  # [1, hidden]
        
        # Compute difference (teacher hidden state for this sample)
        diff = cot_answer_activation - non_cot_answer_activation  # [1, hidden]
        
        self.model_wrapper.clear_hooks()
        
        return diff.squeeze(0)  # [hidden]
    
    def extract(self, support_samples: List) -> torch.Tensor:
        """
        Extract Uncertainty-Aware CoT Vector from support set.
        
        Implements the Variational CoT Vectors framework:
        1. Compute mean μ = (1/N) Σ v_i (same as Extracted method)
        2. Compute variance σ² = (1/N) Σ (v_i - μ)²
        3. Apply Bayesian shrinkage: z = k ⊙ μ where k_d = τ² / (σ²_d + τ²)
        
        Args:
            support_samples: List of training samples
            
        Returns:
            Uncertainty-aware CoT vector with adaptive gating
        """
        print(f"Extracting UA CoT Vectors from {len(support_samples)} samples at layer {self.layer_idx}...")
        print(f"  Prior variance τ² = {self.tau_squared}")
        
        vectors = []
        for sample in tqdm(support_samples, desc="Extracting", ncols=100):
            try:
                vec = self.extract_single(sample)
                vectors.append(vec)
            except Exception as e:
                # Skip problematic samples
                continue
        
        if not vectors:
            raise ValueError("No vectors extracted!")
        
        # Stack all sample vectors: [N, hidden]
        stacked = torch.stack(vectors, dim=0)
        
        # Step 1: Compute mean vector μ (same as standard Extracted method)
        self.mean_vector = stacked.mean(dim=0)  # [hidden]
        
        # Step 2: Compute variance vector σ²
        # Using unbiased estimator: Var = (1/(N-1)) * Σ(x - μ)²
        if len(vectors) > 1:
            self.variance_vector = stacked.var(dim=0, unbiased=True)  # [hidden]
        else:
            # Single sample: use minimum variance
            self.variance_vector = torch.full_like(self.mean_vector, self.min_variance)
        
        # Ensure minimum variance for numerical stability
        self.variance_vector = torch.clamp(self.variance_vector, min=self.min_variance)
        
        # Step 3: Compute shrinkage coefficients k = τ² / (σ² + τ²)
        self.shrinkage_coefficients = self.tau_squared / (self.variance_vector + self.tau_squared)
        
        # Step 4: Apply Bayesian shrinkage: z = k ⊙ μ
        ua_vector = self.shrinkage_coefficients * self.mean_vector
        
        self.vector = ua_vector
        
        # Print statistics
        self._print_statistics()
        
        return ua_vector
    
    def _print_statistics(self):
        """Print diagnostic statistics about the extracted vector."""
        print(f"\nUA Vector Statistics:")
        print(f"  Mean vector norm: {self.mean_vector.norm().item():.4f}")
        print(f"  UA vector norm: {self.vector.norm().item():.4f}")
        print(f"  Shrinkage ratio: {(self.vector.norm() / self.mean_vector.norm()).item():.4f}")
        
        # Shrinkage coefficient statistics
        k = self.shrinkage_coefficients
        print(f"\nShrinkage Coefficients (k):")
        print(f"  Mean: {k.mean().item():.4f}")
        print(f"  Std: {k.std().item():.4f}")
        print(f"  Min: {k.min().item():.4f}")
        print(f"  Max: {k.max().item():.4f}")
        
        # Count dimensions by shrinkage level
        highly_suppressed = (k < 0.1).sum().item()
        moderately_kept = ((k >= 0.1) & (k < 0.5)).sum().item()
        well_preserved = (k >= 0.5).sum().item()
        
        total_dims = k.numel()
        print(f"\nDimension Classification:")
        print(f"  Highly suppressed (k < 0.1): {highly_suppressed} ({100*highly_suppressed/total_dims:.1f}%)")
        print(f"  Moderately kept (0.1 ≤ k < 0.5): {moderately_kept} ({100*moderately_kept/total_dims:.1f}%)")
        print(f"  Well preserved (k ≥ 0.5): {well_preserved} ({100*well_preserved/total_dims:.1f}%)")
    
    def get_vector(self) -> Optional[torch.Tensor]:
        """Get the extracted UA CoT vector."""
        return self.vector
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get extraction statistics for analysis.
        
        Returns:
            Dictionary containing mean, variance, shrinkage coefficients
        """
        if self.mean_vector is None:
            return {}
        
        return {
            "mean_vector": self.mean_vector.cpu(),
            "variance_vector": self.variance_vector.cpu(),
            "shrinkage_coefficients": self.shrinkage_coefficients.cpu(),
            "tau_squared": self.tau_squared,
            "ua_vector": self.vector.cpu() if self.vector is not None else None,
        }
