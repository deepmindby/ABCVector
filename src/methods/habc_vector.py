"""
Hierarchical Adaptive Bayesian CoT Vector (HABC Vector) implementation.

Two-layer hierarchical variational inference:
- Global latent z_g: task-level, shared across batch
  - Prior: p(z_g) = N(0, I)
  - Posterior: q_psi(z_g | S) where S = batch question aggregate

- Instance latent z_i: sample-level
  - Prior: p_phi(z_i | Q_i, z_g)
  - Posterior: q_psi(z_i | Q_i, Y_i, z_g)

Injection (NO gate):
    v_i = W_g @ z_g + W_i @ z_i
    H^(l) <- H^(l) + v_i

Hierarchical ELBO:
    loss = NLL + beta_g * KL_g + beta_i * KL_i
where:
    KL_g = KL(q(z_g|S) || N(0,I))
    KL_i = (1/B) * sum_i KL(q(z_i|Q_i,Y_i,z_g) || p(z_i|Q_i,z_g))

Test-time:
    z_g* = mu_g  (global posterior mean from test set aggregate)
    z_i* = mu_phi_i  (instance prior mean)
    v_i = W_g(z_g*) + W_i(z_i*)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import List, Optional, Dict, Any
from tqdm import tqdm
import math
import gc
import json
import os
from datetime import datetime

from .base import BaseCoTVectorMethod
from .abc_vector import (
    ABCDataset,
    abc_collate_fn,
    kl_divergence_diag_gaussian,
    save_diagnostics_jsonl,
)
from ..models import CoTModelWrapper
from ..data_utils import PROMPT_TEMPLATES
from ..utils import extract_answer_from_text, compare_answers


# ==================== Hierarchical Networks ====================

class GlobalPosteriorNetwork(nn.Module):
    """
    Global posterior q_psi(z_g | S).
    
    Input: r_S [1, H] or [B_agg, H] — aggregated question representation
    Output: (mu_g, raw_sigma_g) each [1, D_g] or [B_agg, D_g]
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden_dim, output_dim)
        self.sigma_head = nn.Linear(hidden_dim, output_dim)
        
        # Initialize to output near-zero means and small sigma
        nn.init.zeros_(self.mu_head.bias)
        nn.init.normal_(self.mu_head.weight, std=0.01)
        nn.init.zeros_(self.sigma_head.bias)
        nn.init.normal_(self.sigma_head.weight, std=0.01)
    
    def forward(self, r_S: torch.Tensor) -> tuple:
        """
        Args:
            r_S: Aggregated support set representation [*, H]
        Returns:
            mu_g: Global mean [*, D_g]
            raw_sigma_g: Raw sigma before softplus [*, D_g]
        """
        h = self.net(r_S)
        mu = self.mu_head(h)
        raw_sigma = self.sigma_head(h)
        return mu, raw_sigma


class HierPriorNetwork(nn.Module):
    """
    Instance prior p_phi(z_i | Q_i, z_g).
    
    Input: concat([r_Q, z_g]) [B, H + D_g]
    Output: (mu_phi_i, raw_sigma_phi_i) each [B, D_i]
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        # input_dim = H + D_g
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden_dim, output_dim)
        self.sigma_head = nn.Linear(hidden_dim, output_dim)
        
        nn.init.zeros_(self.mu_head.bias)
        nn.init.normal_(self.mu_head.weight, std=0.01)
        nn.init.zeros_(self.sigma_head.bias)
        nn.init.normal_(self.sigma_head.weight, std=0.01)
    
    def forward(self, r_Q: torch.Tensor, z_g: torch.Tensor) -> tuple:
        """
        Args:
            r_Q: Question representation [B, H]
            z_g: Global latent [B, D_g] (broadcast from batch-shared)
        Returns:
            mu_phi_i: Instance prior mean [B, D_i]
            raw_sigma_phi_i: Raw sigma [B, D_i]
        """
        x = torch.cat([r_Q, z_g], dim=-1)  # [B, H + D_g]
        h = self.net(x)
        mu = self.mu_head(h)
        raw_sigma = self.sigma_head(h)
        return mu, raw_sigma


class HierPosteriorNetwork(nn.Module):
    """
    Instance posterior q_psi(z_i | Q_i, Y_i, z_g).
    
    Input: concat([r_Q, Y, z_g]) [B, H + H + D_g] = [B, 2H + D_g]
    Output: (mu_psi_i, raw_sigma_psi_i) each [B, D_i]
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        # input_dim = 2 * H + D_g
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden_dim, output_dim)
        self.sigma_head = nn.Linear(hidden_dim, output_dim)
        
        nn.init.zeros_(self.mu_head.bias)
        nn.init.normal_(self.mu_head.weight, std=0.01)
        nn.init.zeros_(self.sigma_head.bias)
        nn.init.normal_(self.sigma_head.weight, std=0.01)
    
    def forward(self, r_Q: torch.Tensor, Y: torch.Tensor, z_g: torch.Tensor) -> tuple:
        """
        Args:
            r_Q: Question representation [B, H]
            Y: Teacher features [B, H]
            z_g: Global latent [B, D_g] (broadcast from batch-shared)
        Returns:
            mu_psi_i: Instance posterior mean [B, D_i]
            raw_sigma_psi_i: Raw sigma [B, D_i]
        """
        x = torch.cat([r_Q, Y, z_g], dim=-1)  # [B, 2H + D_g]
        h = self.net(x)
        mu = self.mu_head(h)
        raw_sigma = self.sigma_head(h)
        return mu, raw_sigma


# ==================== KL with Standard Normal ====================

def kl_divergence_standard_normal(mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """
    KL(q(z) || N(0, I)) for diagonal Gaussian q = N(mu, diag(sigma^2)).
    
    KL = 0.5 * sum( mu^2 + sigma^2 - log(sigma^2) - 1 )
    
    Args:
        mu: Mean [*, D]
        sigma: Std [*, D]
    Returns:
        KL divergence, summed over D, shape [*]
    """
    var = sigma ** 2
    kl = 0.5 * (mu ** 2 + var - torch.log(var) - 1.0)
    return kl.sum(dim=-1)


# ==================== HABC Vector Method ====================

class HierarchicalABCCoTVector(BaseCoTVectorMethod):
    """
    Hierarchical Adaptive Bayesian CoT Vector with two-layer variational inference.
    
    Key differences from ABCCoTVector:
    - Two-layer latent: global z_g + instance z_i
    - No gate parameter g; amplitude controlled by W_g, W_i projections
    - Injection: v_i = W_g(z_g) + W_i(z_i)
    - Hierarchical ELBO with separate KL terms for global and instance
    
    Training:
        z_g ~ q(z_g | S_batch)  where S_batch = mean(r_Q) over batch
        z_i ~ q(z_i | Q_i, Y_i, z_g)
        loss = NLL + beta_g * KL_g + beta_i * KL_i
    
    Test-time:
        z_g* = mu_g  from GlobalPosterior(mean(r_Q) over test set)
        z_i* = mu_phi_i from HierPrior(Q_i, z_g*)
        v_i = W_g(z_g*) + W_i(z_i*)
    """
    
    def __init__(
        self,
        model_wrapper: CoTModelWrapper,
        tokenizer,
        layer_idx: int,
        dataset_type: str = "gsm8k",
        # HABC-specific hyperparameters
        habc_hidden_dim: int = 512,
        global_latent_dim: int = 256,
        instance_latent_dim: int = 256,
        kl_beta_global: float = 0.1,
        kl_beta_instance: float = 0.1,
        kl_warmup_steps_global: int = 0,
        kl_warmup_steps_instance: int = 0,
        sigma_min_global: float = 1e-4,
        sigma_min_instance: float = 1e-4,
        # Training hyperparameters
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-3,
        warmup_ratio: float = 0.1,
        num_epochs: int = 5,
        batch_size: int = 2,
        gradient_accumulation_steps: int = 2,
        max_length: int = 1024,
        # Posterior mode (reuse ABC's modes for instance posterior)
        posterior_mode: str = "q_y_qca",
        # Diagnostics
        save_diagnostics: bool = False,
        diagnostics_dir: str = "./outputs",
    ):
        super().__init__(model_wrapper, tokenizer, layer_idx, dataset_type)
        
        # HABC hyperparameters
        self.habc_hidden_dim = habc_hidden_dim
        self.global_latent_dim = global_latent_dim
        self.instance_latent_dim = instance_latent_dim
        self.kl_beta_global = kl_beta_global
        self.kl_beta_instance = kl_beta_instance
        self.kl_warmup_steps_global = kl_warmup_steps_global
        self.kl_warmup_steps_instance = kl_warmup_steps_instance
        self.sigma_min_global = sigma_min_global
        self.sigma_min_instance = sigma_min_instance
        
        # Training hyperparameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_length = max_length
        
        # Posterior mode
        self.posterior_mode = posterior_mode
        
        # Diagnostics
        self.save_diagnostics = save_diagnostics
        self.diagnostics_dir = diagnostics_dir
        
        # Model hidden size
        hidden_size = model_wrapper.hidden_size
        self.hidden_size = hidden_size
        
        # ===== Initialize networks =====
        
        # Global posterior: r_S [H] -> (mu_g, sigma_g) [D_g]
        self.global_posterior_net = GlobalPosteriorNetwork(
            input_dim=hidden_size,
            hidden_dim=habc_hidden_dim,
            output_dim=global_latent_dim,
        )
        
        # Instance prior: [r_Q; z_g] -> (mu_phi_i, sigma_phi_i) [D_i]
        self.instance_prior_net = HierPriorNetwork(
            input_dim=hidden_size + global_latent_dim,
            hidden_dim=habc_hidden_dim,
            output_dim=instance_latent_dim,
        )
        
        # Instance posterior: [r_Q; Y; z_g] -> (mu_psi_i, sigma_psi_i) [D_i]
        self.instance_posterior_net = HierPosteriorNetwork(
            input_dim=2 * hidden_size + global_latent_dim,
            hidden_dim=habc_hidden_dim,
            output_dim=instance_latent_dim,
        )
        
        # Projection heads (no bias for clean linear combination)
        self.global_to_hidden = nn.Linear(global_latent_dim, hidden_size, bias=False)
        self.instance_to_hidden = nn.Linear(instance_latent_dim, hidden_size, bias=False)
        
        # Initialize projection heads with small weights for stable start
        nn.init.normal_(self.global_to_hidden.weight, std=0.01)
        nn.init.normal_(self.instance_to_hidden.weight, std=0.01)
        
        # Prompt template
        self.prompt_template = PROMPT_TEMPLATES.get(dataset_type, PROMPT_TEMPLATES["gsm8k"])
        
        # Training state
        self.trained = False
    
    # ==================== Sigma utilities ====================
    
    def _get_sigma_global(self, raw_sigma: torch.Tensor) -> torch.Tensor:
        """Apply softplus + sigma_min for global latent."""
        return F.softplus(raw_sigma) + self.sigma_min_global
    
    def _get_sigma_instance(self, raw_sigma: torch.Tensor) -> torch.Tensor:
        """Apply softplus + sigma_min for instance latent."""
        return F.softplus(raw_sigma) + self.sigma_min_instance
    
    # ==================== Feature extraction (copied from ABC, no ABC modification) ====================
    
    def _extract_question_repr(
        self,
        question_ids: torch.Tensor,
        question_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract question representation r_Q via attention-mask weighted mean pooling.
        Identical to ABCCoTVector._extract_question_repr.
        """
        self.model_wrapper.clear_hooks()
        self.model_wrapper.register_extraction_hook(self.layer_idx, requires_grad=False)
        
        with torch.no_grad():
            self.model_wrapper(question_ids, attention_mask=question_mask)
        
        hidden_states = self.model_wrapper.get_activations(self.layer_idx)
        
        mask_expanded = question_mask.unsqueeze(-1).float()
        r_Q = (hidden_states * mask_expanded).sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-8)
        
        self.model_wrapper.clear_hooks()
        
        return r_Q.detach()
    
    def _extract_teacher_features(
        self,
        teacher_ids: torch.Tensor,
        teacher_mask: torch.Tensor,
        teacher_lens: List[int],
        answer_lens: List[int],
    ) -> torch.Tensor:
        """
        Extract teacher features Y (mean pooling over answer tokens).
        Identical to ABCCoTVector._extract_teacher_features.
        """
        self.model_wrapper.clear_hooks()
        self.model_wrapper.register_extraction_hook(self.layer_idx, requires_grad=False)
        
        with torch.no_grad():
            self.model_wrapper(teacher_ids, attention_mask=teacher_mask)
        
        hidden_states = self.model_wrapper.get_activations(self.layer_idx)
        
        bs = teacher_ids.size(0)
        Y_list = []
        for i in range(bs):
            t_len = teacher_lens[i]
            a_len = answer_lens[i]
            ans_start = max(0, t_len - a_len)
            ans_end = t_len
            if ans_start < ans_end:
                Y_i = hidden_states[i, ans_start:ans_end, :].mean(dim=0)
            else:
                Y_i = hidden_states[i, -1, :]
            Y_list.append(Y_i)
        
        self.model_wrapper.clear_hooks()
        return torch.stack(Y_list).detach()
    
    def _extract_teacher_features_qc(
        self,
        qc_ids: torch.Tensor,
        qc_mask: torch.Tensor,
        qc_lens: List[int],
        cot_lens: List[int],
    ) -> torch.Tensor:
        """
        Extract teacher features from [Q; CoT] (no answer).
        Identical to ABCCoTVector._extract_teacher_features_qc.
        """
        self.model_wrapper.clear_hooks()
        self.model_wrapper.register_extraction_hook(self.layer_idx, requires_grad=False)
        
        with torch.no_grad():
            self.model_wrapper(qc_ids, attention_mask=qc_mask)
        
        hidden_states = self.model_wrapper.get_activations(self.layer_idx)
        
        bs = qc_ids.size(0)
        Y_list = []
        for i in range(bs):
            total_len = qc_lens[i]
            c_len = cot_lens[i]
            cot_start = max(0, total_len - c_len)
            cot_end = total_len
            if cot_start < cot_end:
                Y_i = hidden_states[i, cot_start:cot_end, :].mean(dim=0)
            else:
                Y_i = hidden_states[i, -1, :]
            Y_list.append(Y_i)
        
        self.model_wrapper.clear_hooks()
        return torch.stack(Y_list).detach()
    
    def _compute_ce_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        answer_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss on answer tokens.
        Identical to ABCCoTVector._compute_ce_loss.
        """
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        shift_mask = answer_mask[:, 1:].contiguous()
        
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction='none',
        )
        loss = loss.view(shift_labels.shape)
        
        masked_loss = (loss * shift_mask).sum() / (shift_mask.sum() + 1e-8)
        return masked_loss
    
    # ==================== Device management ====================
    
    def _move_networks_to_device(self, device):
        """Move all HABC networks to target device."""
        self.global_posterior_net = self.global_posterior_net.to(device)
        self.instance_prior_net = self.instance_prior_net.to(device)
        self.instance_posterior_net = self.instance_posterior_net.to(device)
        self.global_to_hidden = self.global_to_hidden.to(device)
        self.instance_to_hidden = self.instance_to_hidden.to(device)
    
    # ==================== Training ====================
    
    def train(
        self,
        support_samples: List,
        wandb_run=None,
    ) -> None:
        """
        Train the HABC Vector networks.
        
        The LLM theta is frozen; only hierarchical networks and projections are optimized.
        No gate parameter.
        
        Args:
            support_samples: List of training samples
            wandb_run: Optional WandB run for logging
        """
        print(f"Training HABC Vector at layer {self.layer_idx}...")
        print(f"  Samples: {len(support_samples)}, Epochs: {self.num_epochs}")
        print(f"  HABC Config: hidden_dim={self.habc_hidden_dim}, "
              f"global_latent_dim={self.global_latent_dim}, "
              f"instance_latent_dim={self.instance_latent_dim}")
        print(f"  KL: beta_g={self.kl_beta_global}, beta_i={self.kl_beta_instance}, "
              f"warmup_g={self.kl_warmup_steps_global}, warmup_i={self.kl_warmup_steps_instance}")
        print(f"  Training: lr={self.learning_rate}, batch={self.batch_size}, "
              f"grad_accum={self.gradient_accumulation_steps}")
        print(f"  Posterior mode: {self.posterior_mode}")
        
        # Create dataset and dataloader (reuse ABC's dataset/collate)
        dataset = ABCDataset(
            support_samples,
            self.tokenizer,
            self.dataset_type,
            max_length=self.max_length,
            posterior_mode=self.posterior_mode,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            collate_fn=abc_collate_fn,
        )
        
        # Get target device
        target_layer = self.model_wrapper._get_layer(self.layer_idx)
        target_device = next(target_layer.parameters()).device
        
        # Move networks to device
        self._move_networks_to_device(target_device)
        
        # Setup optimizer — NO gate, only hierarchical networks + projections
        params = (
            list(self.global_posterior_net.parameters()) +
            list(self.instance_prior_net.parameters()) +
            list(self.instance_posterior_net.parameters()) +
            list(self.global_to_hidden.parameters()) +
            list(self.instance_to_hidden.parameters())
        )
        
        optimizer = torch.optim.AdamW(
            params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        
        # Learning rate scheduler
        total_steps = len(dataloader) * self.num_epochs // self.gradient_accumulation_steps
        warmup_steps = int(total_steps * self.warmup_ratio)
        
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return max(0.1, 0.5 * (1 + math.cos(math.pi * progress)))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        # Diagnostics storage
        epoch_diagnostics = []
        
        # Training loop
        global_step = 0
        best_loss = float('inf')
        
        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            epoch_nll = 0.0
            epoch_kl_g = 0.0
            epoch_kl_i = 0.0
            epoch_injected_norm = 0.0
            num_batches = 0
            
            self.global_posterior_net.train()
            self.instance_prior_net.train()
            self.instance_posterior_net.train()
            self.global_to_hidden.train()
            self.instance_to_hidden.train()
            
            optimizer.zero_grad()
            
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.num_epochs}", ncols=100)
            
            for batch_idx, batch in enumerate(pbar):
                try:
                    # Move to device
                    teacher_ids = batch["teacher_ids"].to(target_device)
                    teacher_mask = batch["teacher_mask"].to(target_device)
                    student_ids = batch["student_ids"].to(target_device)
                    student_mask = batch["student_mask"].to(target_device)
                    question_ids = batch["question_ids"].to(target_device)
                    question_mask = batch["question_mask"].to(target_device)
                    
                    bs = teacher_ids.size(0)
                    
                    # ========== Step 1: Extract r_Q for all samples in batch ==========
                    r_Q = self._extract_question_repr(question_ids, question_mask)  # [B, H]
                    
                    # ========== Step 2: Construct batch-level support aggregate ==========
                    r_S = r_Q.mean(dim=0, keepdim=True)  # [1, H]
                    
                    # ========== Step 3: Global posterior q(z_g | S_batch) ==========
                    mu_g, raw_sigma_g = self.global_posterior_net(r_S)  # [1, D_g]
                    sigma_g = self._get_sigma_global(raw_sigma_g)
                    
                    # Reparameterize global latent
                    eps_g = torch.randn_like(mu_g)
                    z_g = mu_g + eps_g * sigma_g  # [1, D_g]
                    
                    # Global KL: KL(q(z_g|S) || N(0,I))
                    kl_g = kl_divergence_standard_normal(mu_g, sigma_g)  # [1]
                    kl_g = kl_g.squeeze(0)  # scalar
                    
                    # Broadcast z_g to batch size
                    z_g_batch = z_g.expand(bs, -1)  # [B, D_g]
                    
                    # ========== Step 4: Extract Y based on posterior_mode ==========
                    if self.posterior_mode == "q_y_qca":
                        Y = self._extract_teacher_features(
                            teacher_ids, teacher_mask,
                            batch["teacher_len"], batch["answer_len"]
                        )
                    elif self.posterior_mode == "q_y_qc":
                        qc_ids = batch["teacher_qc_ids"].to(target_device)
                        qc_mask = batch["teacher_qc_mask"].to(target_device)
                        Y = self._extract_teacher_features_qc(
                            qc_ids, qc_mask,
                            batch["teacher_qc_len"], batch["cot_len"]
                        )
                    elif self.posterior_mode == "q_y_q":
                        Y = r_Q.clone()
                    else:
                        # posterior_mode == "none"
                        Y = None
                    
                    # ========== Step 5: Instance prior p(z_i | Q_i, z_g) ==========
                    mu_phi_i, raw_sigma_phi_i = self.instance_prior_net(r_Q, z_g_batch)
                    sigma_phi_i = self._get_sigma_instance(raw_sigma_phi_i)
                    
                    # ========== Step 6: Instance posterior or prior-only ==========
                    if self.posterior_mode == "none":
                        mu_psi_i = mu_phi_i.detach()
                        sigma_psi_i = sigma_phi_i.detach()
                        kl_i = torch.tensor(0.0, device=target_device)
                    else:
                        mu_psi_i, raw_sigma_psi_i = self.instance_posterior_net(r_Q, Y, z_g_batch)
                        sigma_psi_i = self._get_sigma_instance(raw_sigma_psi_i)
                        
                        # Instance KL: KL(q(z_i|Q,Y,z_g) || p(z_i|Q,z_g))
                        kl_per_sample = kl_divergence_diag_gaussian(
                            mu_psi_i, sigma_psi_i, mu_phi_i, sigma_phi_i
                        )  # [B]
                        kl_i = kl_per_sample.mean()  # scalar
                    
                    # ========== Step 7: Reparameterize instance latent ==========
                    eps_i = torch.randn_like(mu_psi_i)
                    z_i = mu_psi_i + eps_i * sigma_psi_i  # [B, D_i]
                    
                    # ========== Step 8: Compute injection vectors ==========
                    # v_i = W_g(z_g) + W_i(z_i), NO gate
                    v_global = self.global_to_hidden(z_g_batch)  # [B, H]
                    v_instance = self.instance_to_hidden(z_i)    # [B, H]
                    v = v_global + v_instance  # [B, H]
                    
                    # ========== Step 9: Per-sample injection + NLL ==========
                    nll_losses = []
                    
                    for i in range(bs):
                        self.model_wrapper.clear_hooks()
                        
                        s_ids_i = student_ids[i:i+1]
                        s_mask_i = student_mask[i:i+1]
                        v_i = v[i]  # [H]
                        
                        self.model_wrapper.register_injection_hook(
                            self.layer_idx,
                            vector=v_i,
                            scaling_factor=1.0,
                            requires_grad=True,
                        )
                        
                        outputs = self.model_wrapper(s_ids_i, attention_mask=s_mask_i)
                        logits = outputs.logits
                        
                        s_len = batch["student_len"][i]
                        a_len = batch["answer_len"][i]
                        ans_mask = torch.zeros(s_ids_i.shape[1], device=target_device)
                        ans_start = max(0, s_len - a_len)
                        ans_mask[ans_start:s_len] = 1.0
                        
                        nll_i = self._compute_ce_loss(
                            logits,
                            s_ids_i,
                            ans_mask.unsqueeze(0),
                        )
                        nll_losses.append(nll_i)
                        
                        self.model_wrapper.clear_hooks()
                    
                    nll_loss = torch.stack(nll_losses).mean()
                    
                    # ========== Step 10: Hierarchical ELBO loss ==========
                    beta_g_t = self.kl_beta_global
                    if self.kl_warmup_steps_global > 0:
                        beta_g_t *= min(1.0, global_step / self.kl_warmup_steps_global)
                    
                    beta_i_t = self.kl_beta_instance
                    if self.kl_warmup_steps_instance > 0:
                        beta_i_t *= min(1.0, global_step / self.kl_warmup_steps_instance)
                    
                    loss = nll_loss + beta_g_t * kl_g + beta_i_t * kl_i
                    
                    # Gradient accumulation
                    scaled_loss = loss / self.gradient_accumulation_steps
                    scaled_loss.backward()
                    
                    global_step += 1
                    
                    if global_step % self.gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(params, 1.0)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                    
                    # ========== Accumulate metrics ==========
                    with torch.no_grad():
                        injected_norm = v.norm(dim=-1).mean().item()
                    
                    epoch_loss += loss.item()
                    epoch_nll += nll_loss.item()
                    epoch_kl_g += kl_g.item()
                    epoch_kl_i += kl_i.item() if isinstance(kl_i, torch.Tensor) else kl_i
                    epoch_injected_norm += injected_norm
                    num_batches += 1
                    
                    pbar.set_postfix({
                        "loss": f"{loss.item():.3f}",
                        "nll": f"{nll_loss.item():.3f}",
                        "kl_g": f"{kl_g.item():.3f}",
                        "kl_i": f"{(kl_i.item() if isinstance(kl_i, torch.Tensor) else kl_i):.3f}",
                        "v_norm": f"{injected_norm:.2f}",
                    })
                
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"\n  OOM at batch {batch_idx}, skipping...")
                        self.model_wrapper.clear_hooks()
                        torch.cuda.empty_cache()
                        gc.collect()
                        optimizer.zero_grad()
                        continue
                    else:
                        raise
            
            # End of epoch summary
            n = max(1, num_batches)
            avg_loss = epoch_loss / n
            avg_nll = epoch_nll / n
            avg_kl_g = epoch_kl_g / n
            avg_kl_i = epoch_kl_i / n
            avg_injected_norm = epoch_injected_norm / n
            
            print(f"  Epoch {epoch+1}: loss={avg_loss:.4f}, nll={avg_nll:.4f}, "
                  f"kl_g={avg_kl_g:.4f}, kl_i={avg_kl_i:.4f}, "
                  f"v_norm={avg_injected_norm:.4f}")
            
            # Build epoch diagnostic record
            epoch_record = {
                "epoch": epoch + 1,
                "train_loss": round(avg_loss, 6),
                "train_nll": round(avg_nll, 6),
                "train_kl_global": round(avg_kl_g, 6),
                "train_kl_instance": round(avg_kl_i, 6),
                "injected_norm_mean": round(avg_injected_norm, 6),
                "beta_g_t": round(beta_g_t, 6),
                "beta_i_t": round(beta_i_t, 6),
                "posterior_mode": self.posterior_mode,
                "layer_idx": self.layer_idx,
            }
            epoch_diagnostics.append(epoch_record)
            
            if wandb_run:
                wandb_run.log({
                    "epoch": epoch + 1,
                    "train/loss": avg_loss,
                    "train/nll": avg_nll,
                    "train/kl_global": avg_kl_g,
                    "train/kl_instance": avg_kl_i,
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/beta_g_t": beta_g_t,
                    "train/beta_i_t": beta_i_t,
                    "train/injected_norm_mean": avg_injected_norm,
                })
            
            if avg_loss < best_loss:
                best_loss = avg_loss
            
            torch.cuda.empty_cache()
            gc.collect()
        
        self.trained = True
        print(f"Training complete.")
        
        # Save diagnostics
        if self.save_diagnostics and epoch_diagnostics:
            diag_path = os.path.join(
                self.diagnostics_dir,
                f"habc_train_diagnostics_L{self.layer_idx}_{self.posterior_mode}.jsonl"
            )
            save_diagnostics_jsonl(epoch_diagnostics, diag_path)
            print(f"  Training diagnostics saved to {diag_path}")
        
        return None
    
    # ==================== Evaluation ====================
    
    def eval(
        self,
        test_samples: List,
        max_new_tokens: int = 512,
        num_beams: int = 3,
        use_early_stopping: bool = False,
    ) -> Dict[str, Any]:
        """
        Evaluate HABC Vector on test samples.
        
        Uses deterministic mean inference:
        - z_g* = mu_g from GlobalPosterior(mean(r_Q) over test set)
        - z_i* = mu_phi_i from HierPrior(Q_i, z_g*)
        - v_i = W_g(z_g*) + W_i(z_i*)
        """
        if not self.trained:
            print("Warning: HABC Vector not trained yet!")
        
        self.global_posterior_net.eval()
        self.instance_prior_net.eval()
        self.instance_posterior_net.eval()
        self.global_to_hidden.eval()
        self.instance_to_hidden.eval()
        
        target_layer = self.model_wrapper._get_layer(self.layer_idx)
        target_device = next(target_layer.parameters()).device
        self._move_networks_to_device(target_device)
        
        from transformers import GenerationConfig
        
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "num_beams": num_beams,
            "do_sample": False,
            "temperature": 1.0,
            "top_p": 1.0,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if num_beams > 1:
            gen_kwargs["length_penalty"] = 0.0
        
        generation_config = GenerationConfig(**gen_kwargs)
        
        # ========== Step 1: Compute global z_g* from test set aggregate ==========
        print(f"  Computing global z_g* from {len(test_samples)} test samples...")
        r_Q_all = []
        
        for sample in tqdm(test_samples, desc="Extracting r_Q", ncols=100):
            if self.dataset_type == "mmlu_pro":
                question_prompt = self.prompt_template["non_cot"].format(
                    question=sample.question,
                    choices=sample.choices
                )
            else:
                question_prompt = self.prompt_template["non_cot"].format(
                    question=sample.question
                )
            
            q_enc = self.tokenizer(
                question_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
            ).to(target_device)
            
            r_Q_sample = self._extract_question_repr(
                q_enc["input_ids"],
                q_enc["attention_mask"],
            )  # [1, H]
            r_Q_all.append(r_Q_sample)
        
        r_Q_all_tensor = torch.cat(r_Q_all, dim=0)  # [N, H]
        r_S_eval = r_Q_all_tensor.mean(dim=0, keepdim=True)  # [1, H]
        
        with torch.no_grad():
            mu_g_eval, _ = self.global_posterior_net(r_S_eval)  # [1, D_g]
            z_g_star = mu_g_eval  # deterministic
            v_global_star = self.global_to_hidden(z_g_star)  # [1, H]
        
        print(f"  Global z_g* norm: {z_g_star.norm().item():.4f}, "
              f"W_g(z_g*) norm: {v_global_star.norm().item():.4f}")
        
        # ========== Step 2: Per-sample evaluation ==========
        correct = 0
        total = len(test_samples)
        results = []
        injected_norms = []
        
        desc = f"HABC Eval (L{self.layer_idx})"
        
        for idx, sample in enumerate(tqdm(test_samples, desc=desc, ncols=100)):
            try:
                # Build prompts
                if self.dataset_type == "mmlu_pro":
                    gen_prompt = self.prompt_template["cot"].format(
                        question=sample.question,
                        choices=sample.choices
                    )
                else:
                    gen_prompt = self.prompt_template["cot"].format(
                        question=sample.question
                    )
                
                # Use pre-computed r_Q
                r_Q_i = r_Q_all_tensor[idx:idx+1]  # [1, H]
                
                # Instance prior: z_i* = mu_phi_i
                with torch.no_grad():
                    z_g_for_prior = z_g_star  # [1, D_g]
                    mu_phi_i, _ = self.instance_prior_net(r_Q_i, z_g_for_prior)
                    z_i_star = mu_phi_i  # [1, D_i]
                    
                    v_instance_star = self.instance_to_hidden(z_i_star)  # [1, H]
                    v_i = (v_global_star + v_instance_star).squeeze(0)  # [H]
                    
                    z_norm = v_i.norm().item()
                    injected_norms.append(z_norm)
                
                # Register injection hook
                self.model_wrapper.clear_hooks()
                self.model_wrapper.register_injection_hook(
                    self.layer_idx,
                    vector=v_i,
                    scaling_factor=1.0,
                    requires_grad=False,
                )
                
                # Generate
                gen_enc = self.tokenizer(
                    gen_prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_length,
                ).to(target_device)
                
                input_len = gen_enc["input_ids"].shape[1]
                
                with torch.no_grad():
                    outputs = self.model_wrapper.model.generate(
                        gen_enc["input_ids"],
                        attention_mask=gen_enc["attention_mask"],
                        generation_config=generation_config,
                    )
                
                # Decode
                generated_ids = outputs[0, input_len:]
                generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                
                # Extract and compare answer
                predicted = extract_answer_from_text(generated_text, self.dataset_type)
                is_correct = compare_answers(predicted, sample.answer, self.dataset_type)
                
                self.model_wrapper.clear_hooks()
                
                result = {
                    "predicted": predicted,
                    "ground_truth": sample.answer,
                    "correct": is_correct,
                    "generated_text": generated_text,
                    "injected_norm": z_norm,
                }
                results.append(result)
                
                if is_correct:
                    correct += 1
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"\n  OOM on sample {idx}, skipping...")
                    self.model_wrapper.clear_hooks()
                    torch.cuda.empty_cache()
                    gc.collect()
                    results.append({
                        "predicted": None,
                        "ground_truth": sample.answer,
                        "correct": False,
                        "generated_text": "[OOM]",
                        "injected_norm": 0.0,
                    })
                    continue
                else:
                    raise
            
            torch.cuda.empty_cache()
        
        accuracy = (correct / total * 100) if total > 0 else 0.0
        avg_norm = sum(injected_norms) / len(injected_norms) if injected_norms else 0.0
        
        print(f"\n  HABC Eval: {correct}/{total} = {accuracy:.2f}%")
        print(f"  Average injected norm: {avg_norm:.4f}")
        
        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "results": results,
            "avg_injected_norm": avg_norm,
            "injected_norms": injected_norms,
        }
    
    # ==================== Checkpoint ====================
    
    def get_vector(self) -> Optional[torch.Tensor]:
        """HABC uses dynamic vectors, return None."""
        return None
    
    def get_state_dict(self) -> Dict[str, Any]:
        """Get state dict for saving checkpoint."""
        return {
            "global_posterior": self.global_posterior_net.state_dict(),
            "instance_prior": self.instance_prior_net.state_dict(),
            "instance_posterior": self.instance_posterior_net.state_dict(),
            "global_to_hidden": self.global_to_hidden.state_dict(),
            "instance_to_hidden": self.instance_to_hidden.state_dict(),
            "layer_idx": self.layer_idx,
            "global_latent_dim": self.global_latent_dim,
            "instance_latent_dim": self.instance_latent_dim,
            "habc_hidden_dim": self.habc_hidden_dim,
            "kl_beta_global": self.kl_beta_global,
            "kl_beta_instance": self.kl_beta_instance,
            "kl_warmup_steps_global": self.kl_warmup_steps_global,
            "kl_warmup_steps_instance": self.kl_warmup_steps_instance,
            "sigma_min_global": self.sigma_min_global,
            "sigma_min_instance": self.sigma_min_instance,
            "posterior_mode": self.posterior_mode,
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any], device=None):
        """Load state dict from checkpoint."""
        self.global_posterior_net.load_state_dict(state_dict["global_posterior"])
        self.instance_prior_net.load_state_dict(state_dict["instance_prior"])
        self.instance_posterior_net.load_state_dict(state_dict["instance_posterior"])
        self.global_to_hidden.load_state_dict(state_dict["global_to_hidden"])
        self.instance_to_hidden.load_state_dict(state_dict["instance_to_hidden"])
        
        if device is not None:
            self._move_networks_to_device(device)
        
        self.trained = True