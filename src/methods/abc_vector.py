"""
Adaptive Bayesian CoT Vector (ABC Vector) implementation.

Based on the Variational CoT Vectors framework with:
- Prior Network p_phi(z|Q): predicts z distribution from question only
- Posterior Network q_psi(z|Q,Y): uses privileged teacher features (train-only)
- Gated injection: H_tilde = H + g * z

ELBO objective: J = NLL + beta_t * KL(q_psi || p_phi)
- NLL: Cross-entropy on answer tokens
- KL: Closed-form diagonal Gaussian KL divergence
- KL warmup: beta_t = kl_beta * min(1.0, step / warmup_steps)

Test-time: Use prior mean z* = mu_phi(Q) for injection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from typing import List, Optional, Dict, Any
from tqdm import tqdm
import math
import gc

from .base import BaseCoTVectorMethod
from ..models import CoTModelWrapper
from ..data_utils import PROMPT_TEMPLATES
from ..eval import CoTEvaluator
from ..utils import extract_answer_from_text, compare_answers


# ==================== MLP Networks ====================

class PriorNetwork(nn.Module):
    """
    Prior Network p_phi(z|Q).
    
    Input: r_Q [B, H] - question representation
    Output: (mu_phi, raw_sigma_phi) each [B, H]
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
        
        # Initialize to output near-zero means
        nn.init.zeros_(self.mu_head.bias)
        nn.init.normal_(self.mu_head.weight, std=0.01)
        nn.init.zeros_(self.sigma_head.bias)
        nn.init.normal_(self.sigma_head.weight, std=0.01)
    
    def forward(self, r_Q: torch.Tensor) -> tuple:
        """
        Args:
            r_Q: Question representation [B, H]
        Returns:
            mu_phi: Mean [B, H]
            raw_sigma_phi: Raw sigma (before softplus) [B, H]
        """
        h = self.net(r_Q)
        mu = self.mu_head(h)
        raw_sigma = self.sigma_head(h)
        return mu, raw_sigma


class PosteriorNetwork(nn.Module):
    """
    Posterior Network q_psi(z|Q,Y).
    
    Input: concat([r_Q, Y]) [B, 2H] - question repr + teacher features
    Output: (mu_psi, raw_sigma_psi) each [B, H]
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        # input_dim = 2 * H (concat of r_Q and Y)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden_dim, output_dim)
        self.sigma_head = nn.Linear(hidden_dim, output_dim)
        
        # Initialize to output near-zero means
        nn.init.zeros_(self.mu_head.bias)
        nn.init.normal_(self.mu_head.weight, std=0.01)
        nn.init.zeros_(self.sigma_head.bias)
        nn.init.normal_(self.sigma_head.weight, std=0.01)
    
    def forward(self, r_Q: torch.Tensor, Y: torch.Tensor) -> tuple:
        """
        Args:
            r_Q: Question representation [B, H]
            Y: Teacher features [B, H]
        Returns:
            mu_psi: Mean [B, H]
            raw_sigma_psi: Raw sigma (before softplus) [B, H]
        """
        x = torch.cat([r_Q, Y], dim=-1)  # [B, 2H]
        h = self.net(x)
        mu = self.mu_head(h)
        raw_sigma = self.sigma_head(h)
        return mu, raw_sigma


# ==================== Dataset and Collate ====================

class ABCDataset(Dataset):
    """Dataset for ABC Vector training with three prompt types."""
    
    def __init__(self, samples: List, tokenizer, dataset_type: str, max_length: int = 1024):
        self.samples = samples
        self.tokenizer = tokenizer
        self.dataset_type = dataset_type
        self.max_length = max_length
        self.prompt_template = PROMPT_TEMPLATES.get(dataset_type, PROMPT_TEMPLATES["gsm8k"])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Build three types of prompts
        if self.dataset_type == "mmlu_pro":
            # A) Teacher prompt (with CoT + answer)
            teacher_prompt = self.prompt_template["cot"].format(
                question=sample.question,
                choices=sample.choices
            ) + sample.cot + f"\nThe answer is {sample.answer}"
            
            # B) Student prompt full (non-CoT + answer, for training NLL)
            student_prompt = self.prompt_template["non_cot"].format(
                question=sample.question,
                choices=sample.choices
            ) + f"The answer is {sample.answer}"
            
            # C) Question only prompt (for prior input r_Q)
            question_prompt = self.prompt_template["non_cot"].format(
                question=sample.question,
                choices=sample.choices
            )
        else:
            # A) Teacher prompt
            teacher_prompt = self.prompt_template["cot"].format(
                question=sample.question
            ) + sample.cot + f"\nThe answer is {sample.answer}"
            
            # B) Student prompt full
            student_prompt = self.prompt_template["non_cot"].format(
                question=sample.question
            ) + f"The answer is {sample.answer}"
            
            # C) Question only prompt
            question_prompt = self.prompt_template["non_cot"].format(
                question=sample.question
            )
        
        # Tokenize all three
        teacher_enc = self.tokenizer(
            teacher_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        student_enc = self.tokenizer(
            student_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        question_enc = self.tokenizer(
            question_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        
        # Get answer token positions
        answer_text = f"The answer is {sample.answer}"
        answer_ids = self.tokenizer(answer_text, add_special_tokens=False)["input_ids"]
        answer_len = len(answer_ids)
        
        # Actual sequence lengths
        teacher_len = teacher_enc["input_ids"].shape[1]
        student_len = student_enc["input_ids"].shape[1]
        question_len = question_enc["input_ids"].shape[1]
        
        return {
            "teacher_ids": teacher_enc["input_ids"].squeeze(0),
            "teacher_mask": teacher_enc["attention_mask"].squeeze(0),
            "student_ids": student_enc["input_ids"].squeeze(0),
            "student_mask": student_enc["attention_mask"].squeeze(0),
            "question_ids": question_enc["input_ids"].squeeze(0),
            "question_mask": question_enc["attention_mask"].squeeze(0),
            "teacher_len": teacher_len,
            "student_len": student_len,
            "question_len": question_len,
            "answer_len": answer_len,
        }


def abc_collate_fn(batch):
    """Custom collate function with dynamic padding for ABC dataset."""
    # Find max lengths in this batch
    max_teacher_len = max(item["teacher_len"] for item in batch)
    max_student_len = max(item["student_len"] for item in batch)
    max_question_len = max(item["question_len"] for item in batch)
    
    teacher_ids_list = []
    teacher_mask_list = []
    student_ids_list = []
    student_mask_list = []
    question_ids_list = []
    question_mask_list = []
    teacher_lens = []
    student_lens = []
    question_lens = []
    answer_lens = []
    
    for item in batch:
        # Pad teacher
        t_ids = item["teacher_ids"]
        t_mask = item["teacher_mask"]
        t_pad_len = max_teacher_len - len(t_ids)
        if t_pad_len > 0:
            t_ids = F.pad(t_ids, (0, t_pad_len), value=0)
            t_mask = F.pad(t_mask, (0, t_pad_len), value=0)
        teacher_ids_list.append(t_ids)
        teacher_mask_list.append(t_mask)
        
        # Pad student
        s_ids = item["student_ids"]
        s_mask = item["student_mask"]
        s_pad_len = max_student_len - len(s_ids)
        if s_pad_len > 0:
            s_ids = F.pad(s_ids, (0, s_pad_len), value=0)
            s_mask = F.pad(s_mask, (0, s_pad_len), value=0)
        student_ids_list.append(s_ids)
        student_mask_list.append(s_mask)
        
        # Pad question
        q_ids = item["question_ids"]
        q_mask = item["question_mask"]
        q_pad_len = max_question_len - len(q_ids)
        if q_pad_len > 0:
            q_ids = F.pad(q_ids, (0, q_pad_len), value=0)
            q_mask = F.pad(q_mask, (0, q_pad_len), value=0)
        question_ids_list.append(q_ids)
        question_mask_list.append(q_mask)
        
        teacher_lens.append(item["teacher_len"])
        student_lens.append(item["student_len"])
        question_lens.append(item["question_len"])
        answer_lens.append(item["answer_len"])
    
    return {
        "teacher_ids": torch.stack(teacher_ids_list),
        "teacher_mask": torch.stack(teacher_mask_list),
        "student_ids": torch.stack(student_ids_list),
        "student_mask": torch.stack(student_mask_list),
        "question_ids": torch.stack(question_ids_list),
        "question_mask": torch.stack(question_mask_list),
        "teacher_len": teacher_lens,
        "student_len": student_lens,
        "question_len": question_lens,
        "answer_len": answer_lens,
    }


# ==================== Utility Functions ====================

def compute_kl_divergence(
    mu_q: torch.Tensor,
    sigma_q: torch.Tensor,
    mu_p: torch.Tensor,
    sigma_p: torch.Tensor,
) -> torch.Tensor:
    """
    Compute closed-form KL divergence for diagonal Gaussians.
    KL(q || p) = 0.5 * sum( log(sigma_p^2/sigma_q^2) + (sigma_q^2 + (mu_q-mu_p)^2)/sigma_p^2 - 1 )
    
    Args:
        mu_q: Posterior mean [B, H]
        sigma_q: Posterior std [B, H]
        mu_p: Prior mean [B, H]
        sigma_p: Prior std [B, H]
    
    Returns:
        KL divergence [B]
    """
    var_q = sigma_q ** 2
    var_p = sigma_p ** 2
    
    # KL = 0.5 * sum( log(var_p/var_q) + var_q/var_p + (mu_q-mu_p)^2/var_p - 1 )
    kl = 0.5 * (
        torch.log(var_p / var_q) +
        var_q / var_p +
        ((mu_q - mu_p) ** 2) / var_p -
        1.0
    )
    
    # Sum over hidden dimensions, return [B]
    return kl.sum(dim=-1)


# ==================== ABC Vector Method ====================

class ABCCoTVector(BaseCoTVectorMethod):
    """
    Adaptive Bayesian CoT Vector with variational inference.
    
    Key components:
    - Prior Network p_phi(z|Q): Test-time usable, predicts z from question only
    - Posterior Network q_psi(z|Q,Y): Train-only, uses privileged teacher features
    - Gated Injection: H_tilde = H + g * z
    
    Training objective (ELBO):
        J = NLL + beta_t * KL(q_psi || p_phi)
    
    Test-time inference:
        z* = mu_phi(Q)  (use prior mean, no sampling)
    """
    
    def __init__(
        self,
        model_wrapper: CoTModelWrapper,
        tokenizer,
        layer_idx: int,
        dataset_type: str = "gsm8k",
        # ABC-specific hyperparameters
        abc_hidden_dim: int = 512,
        kl_beta: float = 1.0,
        kl_warmup_steps: int = 0,
        sigma_min: float = 1e-4,
        # Training hyperparameters (reuse from args)
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-3,
        warmup_ratio: float = 0.1,
        num_epochs: int = 5,
        batch_size: int = 2,
        gradient_accumulation_steps: int = 2,
        max_length: int = 1024,
    ):
        super().__init__(model_wrapper, tokenizer, layer_idx, dataset_type)
        
        # ABC hyperparameters
        self.abc_hidden_dim = abc_hidden_dim
        self.kl_beta = kl_beta
        self.kl_warmup_steps = kl_warmup_steps
        self.sigma_min = sigma_min
        
        # Training hyperparameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_length = max_length
        
        # Get hidden size from model
        hidden_size = model_wrapper.hidden_size
        self.hidden_size = hidden_size
        
        # z_dim = hidden_size (no projection, direct injection)
        self.z_dim = hidden_size
        
        # Initialize networks
        # Prior: input H -> output (mu, sigma) each H
        self.prior_net = PriorNetwork(
            input_dim=hidden_size,
            hidden_dim=abc_hidden_dim,
            output_dim=self.z_dim,
        )
        
        # Posterior: input 2H -> output (mu, sigma) each H
        self.posterior_net = PosteriorNetwork(
            input_dim=2 * hidden_size,
            hidden_dim=abc_hidden_dim,
            output_dim=self.z_dim,
        )
        
        # Learnable gate scalar (initialize to 0 for smooth start)
        self.gate = nn.Parameter(torch.tensor(0.0))
        
        # Prompt template
        self.prompt_template = PROMPT_TEMPLATES.get(dataset_type, PROMPT_TEMPLATES["gsm8k"])
        
        # Training state
        self.trained = False
    
    def _get_sigma(self, raw_sigma: torch.Tensor) -> torch.Tensor:
        """Apply softplus + sigma_min for numerical stability."""
        return F.softplus(raw_sigma) + self.sigma_min
    
    def _extract_question_repr(
        self,
        question_ids: torch.Tensor,
        question_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract question representation r_Q.
        
        Uses attention_mask weighted mean pooling over all question tokens
        at the target layer.
        
        Args:
            question_ids: [B, Tq]
            question_mask: [B, Tq]
        
        Returns:
            r_Q: [B, H]
        """
        device = self.model_wrapper.device
        
        # Clear hooks
        self.model_wrapper.clear_hooks()
        
        # Register extraction hook
        self.model_wrapper.register_extraction_hook(self.layer_idx, requires_grad=False)
        
        # Forward pass (frozen)
        with torch.no_grad():
            self.model_wrapper(question_ids, attention_mask=question_mask)
        
        # Get hidden states [B, Tq, H]
        hidden_states = self.model_wrapper.get_activations(self.layer_idx)
        
        # Weighted mean pooling with attention mask
        # mask: [B, Tq] -> [B, Tq, 1]
        mask_expanded = question_mask.unsqueeze(-1).float()
        
        # Weighted sum / sum of weights
        # [B, Tq, H] * [B, Tq, 1] -> sum over Tq -> [B, H]
        r_Q = (hidden_states * mask_expanded).sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-8)
        
        self.model_wrapper.clear_hooks()
        
        return r_Q.detach()  # [B, H]
    
    def _extract_teacher_features(
        self,
        teacher_ids: torch.Tensor,
        teacher_mask: torch.Tensor,
        teacher_lens: List[int],
        answer_lens: List[int],
    ) -> torch.Tensor:
        """
        Extract teacher features Y (privileged, train-only).
        
        Mean pooling over answer token positions at the target layer.
        
        Args:
            teacher_ids: [B, T]
            teacher_mask: [B, T]
            teacher_lens: List of actual sequence lengths
            answer_lens: List of answer lengths
        
        Returns:
            Y: [B, H]
        """
        device = self.model_wrapper.device
        bs = teacher_ids.size(0)
        
        # Clear hooks
        self.model_wrapper.clear_hooks()
        
        # Register extraction hook
        self.model_wrapper.register_extraction_hook(self.layer_idx, requires_grad=False)
        
        # Forward pass (frozen)
        with torch.no_grad():
            self.model_wrapper(teacher_ids, attention_mask=teacher_mask)
        
        # Get hidden states [B, T, H]
        hidden_states = self.model_wrapper.get_activations(self.layer_idx)
        
        # Extract answer positions for each sample and compute mean
        Y_list = []
        for i in range(bs):
            t_len = teacher_lens[i]
            a_len = answer_lens[i]
            # Answer positions: last a_len tokens (before padding)
            ans_start = max(0, t_len - a_len)
            ans_end = t_len
            
            if ans_start < ans_end:
                ans_hidden = hidden_states[i, ans_start:ans_end, :]  # [a_len, H]
                y_i = ans_hidden.mean(dim=0)  # [H]
            else:
                # Fallback: use last token
                y_i = hidden_states[i, t_len - 1, :]  # [H]
            
            Y_list.append(y_i)
        
        Y = torch.stack(Y_list, dim=0)  # [B, H]
        
        self.model_wrapper.clear_hooks()
        
        return Y.detach()  # [B, H]
    
    def _compute_ce_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss on answer tokens.
        Follows the same implementation as learnable.py.
        
        Args:
            logits: [B, T, V]
            labels: [B, T]
            mask: [B, T] - 1 for answer positions, 0 otherwise
        
        Returns:
            CE loss (scalar)
        """
        # Shift for causal LM
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_mask = mask[..., 1:].contiguous()
        
        # Flatten
        vocab_size = shift_logits.size(-1)
        flat_logits = shift_logits.view(-1, vocab_size)
        flat_labels = shift_labels.view(-1)
        flat_mask = shift_mask.view(-1).float()
        
        # Compute loss only on masked positions
        ce_loss = F.cross_entropy(flat_logits, flat_labels, reduction='none')
        masked_loss = (ce_loss * flat_mask).sum() / (flat_mask.sum() + 1e-8)
        
        return masked_loss
    
    def _move_networks_to_device(self, device):
        """Move prior, posterior, and gate to the specified device."""
        self.prior_net = self.prior_net.to(device)
        self.posterior_net = self.posterior_net.to(device)
        self.gate.data = self.gate.data.to(device)
    
    def train(
        self,
        support_samples: List,
        wandb_run=None,
    ) -> None:
        """
        Train the ABC Vector networks (prior, posterior, gate).
        
        The LLM theta is frozen; only phi, psi, g are optimized.
        
        Args:
            support_samples: List of training samples
            wandb_run: Optional WandB run for logging
        
        Returns:
            None (ABC uses dynamic vectors, not a fixed vector)
        """
        print(f"Training ABC Vector at layer {self.layer_idx}...")
        print(f"  Samples: {len(support_samples)}, Epochs: {self.num_epochs}")
        print(f"  ABC Config: hidden_dim={self.abc_hidden_dim}, kl_beta={self.kl_beta}, "
              f"kl_warmup={self.kl_warmup_steps}, sigma_min={self.sigma_min}")
        print(f"  Training: lr={self.learning_rate}, batch={self.batch_size}, "
              f"grad_accum={self.gradient_accumulation_steps}")
        
        # Create dataset and dataloader
        dataset = ABCDataset(
            support_samples,
            self.tokenizer,
            self.dataset_type,
            max_length=self.max_length,
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
        
        # Setup optimizer (only train prior, posterior, gate)
        params = list(self.prior_net.parameters()) + \
                 list(self.posterior_net.parameters()) + \
                 [self.gate]
        
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
        
        # Training loop
        global_step = 0
        best_loss = float('inf')
        
        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            epoch_nll = 0.0
            epoch_kl = 0.0
            num_batches = 0
            
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
                    
                    # ========== Step 1: Extract r_Q (question representation) ==========
                    r_Q = self._extract_question_repr(question_ids, question_mask)  # [B, H]
                    
                    # ========== Step 2: Extract Y (teacher features, train-only) ==========
                    Y = self._extract_teacher_features(
                        teacher_ids, teacher_mask,
                        batch["teacher_len"], batch["answer_len"]
                    )  # [B, H]
                    
                    # ========== Step 3: Prior distribution p_phi(z|Q) ==========
                    mu_phi, raw_sigma_phi = self.prior_net(r_Q)
                    sigma_phi = self._get_sigma(raw_sigma_phi)  # [B, H]
                    
                    # ========== Step 4: Posterior distribution q_psi(z|Q,Y) ==========
                    mu_psi, raw_sigma_psi = self.posterior_net(r_Q, Y)
                    sigma_psi = self._get_sigma(raw_sigma_psi)  # [B, H]
                    
                    # ========== Step 5: Reparameterization trick (MC=1) ==========
                    eps = torch.randn_like(mu_psi)
                    z = mu_psi + eps * sigma_psi  # [B, H]
                    
                    # ========== Step 6: Gated injection and NLL computation ==========
                    # Compute g * z for each sample
                    gated_z = self.gate * z  # [B, H]
                    
                    # Process samples one by one to handle injection properly
                    nll_losses = []
                    
                    for i in range(bs):
                        # Clear hooks
                        self.model_wrapper.clear_hooks()
                        
                        # Get single sample
                        s_ids_i = student_ids[i:i+1]  # [1, T]
                        s_mask_i = student_mask[i:i+1]  # [1, T]
                        gated_z_i = gated_z[i]  # [H]
                        
                        # Register injection hook with gradient
                        self.model_wrapper.register_injection_hook(
                            self.layer_idx,
                            vector=gated_z_i,
                            scaling_factor=1.0,
                            requires_grad=True,
                        )
                        
                        # Forward pass
                        outputs = self.model_wrapper(s_ids_i, attention_mask=s_mask_i)
                        logits = outputs.logits  # [1, T, V]
                        
                        # Create answer mask
                        s_len = batch["student_len"][i]
                        a_len = batch["answer_len"][i]
                        ans_mask = torch.zeros(s_ids_i.shape[1], device=target_device)
                        ans_start = max(0, s_len - a_len)
                        ans_mask[ans_start:s_len] = 1.0
                        
                        # Compute CE loss
                        nll_i = self._compute_ce_loss(
                            logits,
                            s_ids_i,
                            ans_mask.unsqueeze(0),
                        )
                        nll_losses.append(nll_i)
                        
                        # Clear hooks
                        self.model_wrapper.clear_hooks()
                    
                    # Average NLL
                    nll_loss = torch.stack(nll_losses).mean()
                    
                    # ========== Step 7: KL divergence ==========
                    kl_per_sample = compute_kl_divergence(
                        mu_psi, sigma_psi,
                        mu_phi, sigma_phi,
                    )  # [B]
                    kl_loss = kl_per_sample.mean()
                    
                    # ========== Step 8: KL warmup ==========
                    if self.kl_warmup_steps > 0:
                        beta_t = self.kl_beta * min(1.0, global_step / self.kl_warmup_steps)
                    else:
                        beta_t = self.kl_beta
                    
                    # ========== Step 9: Total loss (negative ELBO) ==========
                    loss = nll_loss + beta_t * kl_loss
                    loss = loss / self.gradient_accumulation_steps
                    
                    # Backward
                    loss.backward()
                    
                    # Clean up tensors
                    del r_Q, Y, z, gated_z, nll_losses
                    
                    # Gradient accumulation step
                    if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(params, 1.0)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                        global_step += 1
                        
                        # Clear CUDA cache periodically
                        if global_step % 10 == 0:
                            torch.cuda.empty_cache()
                    
                    # Track losses
                    epoch_loss += loss.item() * self.gradient_accumulation_steps
                    epoch_nll += nll_loss.item()
                    epoch_kl += kl_loss.item()
                    num_batches += 1
                    
                    # Update progress
                    pbar.set_postfix({
                        "loss": f"{epoch_loss/num_batches:.4f}",
                        "nll": f"{epoch_nll/num_batches:.4f}",
                        "kl": f"{epoch_kl/num_batches:.4f}",
                        "g": f"{self.gate.item():.3f}",
                    })
                    
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"\n  Warning: OOM at batch {batch_idx}, clearing cache...")
                        self.model_wrapper.clear_hooks()
                        torch.cuda.empty_cache()
                        gc.collect()
                        optimizer.zero_grad()
                        continue
                    else:
                        raise
            
            # End of epoch
            if (batch_idx + 1) % self.gradient_accumulation_steps != 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            # Epoch summary
            avg_loss = epoch_loss / max(num_batches, 1)
            avg_nll = epoch_nll / max(num_batches, 1)
            avg_kl = epoch_kl / max(num_batches, 1)
            
            print(f"  Epoch {epoch+1}: loss={avg_loss:.4f}, nll={avg_nll:.4f}, "
                  f"kl={avg_kl:.4f}, gate={self.gate.item():.4f}")
            
            if wandb_run:
                wandb_run.log({
                    "epoch": epoch + 1,
                    "train/loss": avg_loss,
                    "train/nll": avg_nll,
                    "train/kl": avg_kl,
                    "train/gate": self.gate.item(),
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/beta_t": beta_t,
                })
            
            # Track best
            if avg_loss < best_loss:
                best_loss = avg_loss
            
            # Clear cache at end of epoch
            torch.cuda.empty_cache()
            gc.collect()
        
        self.trained = True
        print(f"Training complete. Final gate value: {self.gate.item():.4f}")
        
        return None  # ABC returns None (dynamic vectors)
    
    def eval(
        self,
        test_samples: List,
        max_new_tokens: int = 512,
        num_beams: int = 3,
        use_early_stopping: bool = False,
    ) -> Dict[str, Any]:
        """
        Evaluate ABC Vector on test samples.
        
        For each sample:
        1. Compute r_Q from question_only_prompt
        2. Get z* = mu_phi(Q) from prior (no sampling)
        3. Inject g * z* and generate answer
        
        Args:
            test_samples: List of test samples
            max_new_tokens: Max tokens to generate
            num_beams: Number of beams for generation
            use_early_stopping: Use early stopping criteria
        
        Returns:
            Dict with accuracy, correct, total, results
        """
        if not self.trained:
            print("Warning: ABC Vector not trained yet!")
        
        # Set networks to eval mode
        self.prior_net.eval()
        self.posterior_net.eval()
        
        # Get device
        target_layer = self.model_wrapper._get_layer(self.layer_idx)
        target_device = next(target_layer.parameters()).device
        
        # Create evaluator for generation
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
        
        correct = 0
        total = len(test_samples)
        results = []
        
        pbar = tqdm(test_samples, desc=f"ABC Eval (L{self.layer_idx})", ncols=100)
        
        for sample in pbar:
            try:
                # Build question_only_prompt
                if self.dataset_type == "mmlu_pro":
                    question_prompt = self.prompt_template["non_cot"].format(
                        question=sample.question,
                        choices=sample.choices
                    )
                    gen_prompt = self.prompt_template["cot"].format(
                        question=sample.question,
                        choices=sample.choices
                    )
                else:
                    question_prompt = self.prompt_template["non_cot"].format(
                        question=sample.question
                    )
                    gen_prompt = self.prompt_template["cot"].format(
                        question=sample.question
                    )
                
                # Tokenize question for r_Q
                q_enc = self.tokenizer(
                    question_prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_length,
                )
                q_ids = q_enc["input_ids"].to(target_device)
                q_mask = q_enc["attention_mask"].to(target_device)
                
                # Step 1: Extract r_Q
                with torch.no_grad():
                    r_Q = self._extract_question_repr(q_ids, q_mask)  # [1, H]
                
                # Step 2: Get z* = mu_phi(Q) from prior
                with torch.no_grad():
                    mu_phi, _ = self.prior_net(r_Q)
                    z_star = mu_phi  # [1, H]
                
                # Step 3: Gated injection vector
                gated_z_star = self.gate * z_star.squeeze(0)  # [H]
                
                # Tokenize generation prompt
                gen_enc = self.tokenizer(
                    gen_prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_length,
                )
                gen_ids = gen_enc["input_ids"].to(target_device)
                gen_mask = gen_enc["attention_mask"].to(target_device)
                input_len = gen_ids.shape[1]
                
                # Clear hooks and register injection
                self.model_wrapper.clear_hooks()
                self.model_wrapper.register_injection_hook(
                    self.layer_idx,
                    vector=gated_z_star,
                    scaling_factor=1.0,
                    requires_grad=False,
                )
                
                # Generate
                with torch.no_grad():
                    outputs = self.model_wrapper.model.generate(
                        gen_ids,
                        attention_mask=gen_mask,
                        generation_config=generation_config,
                    )
                
                # Decode
                generated_ids = outputs[0, input_len:]
                generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                
                # Extract answer
                predicted = extract_answer_from_text(generated_text, self.dataset_type)
                is_correct = compare_answers(predicted, sample.answer, self.dataset_type)
                
                # Clear hooks
                self.model_wrapper.clear_hooks()
                
                result = {
                    "predicted": predicted,
                    "ground_truth": sample.answer,
                    "correct": is_correct,
                    "generated_text": generated_text,
                    "num_tokens": len(generated_ids),
                }
                results.append(result)
                
                if is_correct:
                    correct += 1
                
                # Update progress
                acc = correct / len(results) * 100
                pbar.set_postfix({"acc": f"{acc:.1f}%"})
                
            except Exception as e:
                print(f"\n  Error evaluating sample: {e}")
                results.append({
                    "predicted": None,
                    "ground_truth": sample.answer,
                    "correct": False,
                    "error": str(e),
                })
                continue
        
        accuracy = correct / total * 100
        
        # Set networks back to train mode
        self.prior_net.train()
        self.posterior_net.train()
        
        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "results": results,
        }
    
    def get_vector(self) -> Optional[torch.Tensor]:
        """
        ABC uses dynamic vectors (z* per sample), so return None.
        """
        return None
    
    def get_state_dict(self) -> Dict[str, Any]:
        """
        Get state dict for saving checkpoint.
        
        Returns:
            Dict with prior, posterior, gate states
        """
        return {
            "prior": self.prior_net.state_dict(),
            "posterior": self.posterior_net.state_dict(),
            "gate": self.gate.detach().cpu(),
            "layer_idx": self.layer_idx,
            "abc_hidden_dim": self.abc_hidden_dim,
            "kl_beta": self.kl_beta,
            "kl_warmup_steps": self.kl_warmup_steps,
            "sigma_min": self.sigma_min,
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any], device=None):
        """
        Load state dict from checkpoint.
        
        Args:
            state_dict: Saved state dict
            device: Target device (optional)
        """
        self.prior_net.load_state_dict(state_dict["prior"])
        self.posterior_net.load_state_dict(state_dict["posterior"])
        self.gate.data = state_dict["gate"]
        
        if device is not None:
            self._move_networks_to_device(device)
        
        self.trained = True