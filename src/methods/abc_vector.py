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

Posterior Mode Ablation (--posterior_mode):
- q_y_qca : q(z | Q, Y_{Q;C;A})  — full privileged info (default)
- q_y_qc  : q(z | Q, Y_{Q;C})    — CoT only, no answer tokens
- q_y_q   : q(z | Q, Y_Q)        — question features only (Y = r_Q)
- none    : no posterior; z sampled from prior during training (KL=0)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from typing import List, Optional, Dict, Any
from tqdm import tqdm
import math
import gc
import json
import os
import csv
from datetime import datetime

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
    """
    Dataset for ABC Vector training with multiple prompt types.
    
    Supports posterior_mode ablation:
      - q_y_qca: teacher = [Q; CoT; Answer], teacher_qc = not needed
      - q_y_qc:  teacher = [Q; CoT; Answer] (kept for NLL), teacher_qc = [Q; CoT]
      - q_y_q:   teacher = [Q; CoT; Answer] (kept for NLL), Y = r_Q at runtime
      - none:    teacher = [Q; CoT; Answer] (kept for NLL), posterior skipped
    """
    
    def __init__(
        self,
        samples: List,
        tokenizer,
        dataset_type: str,
        max_length: int = 1024,
        posterior_mode: str = "q_y_qca",
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.dataset_type = dataset_type
        self.max_length = max_length
        self.posterior_mode = posterior_mode
        self.prompt_template = PROMPT_TEMPLATES.get(dataset_type, PROMPT_TEMPLATES["gsm8k"])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Build prompts depending on dataset type
        if self.dataset_type == "mmlu_pro":
            # A) Teacher prompt (with CoT + answer) — always needed for NLL
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
            
            # D) Teacher QC prompt (Q + CoT, no answer) — for q_y_qc mode
            teacher_qc_prompt = self.prompt_template["cot"].format(
                question=sample.question,
                choices=sample.choices
            ) + sample.cot
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
            
            # D) Teacher QC prompt (Q + CoT, no answer)
            teacher_qc_prompt = self.prompt_template["cot"].format(
                question=sample.question
            ) + sample.cot
        
        # Tokenize core prompts (always needed)
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
        
        result = {
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
        
        # Tokenize teacher_qc only if needed
        if self.posterior_mode == "q_y_qc":
            teacher_qc_enc = self.tokenizer(
                teacher_qc_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
            )
            teacher_qc_len = teacher_qc_enc["input_ids"].shape[1]
            
            # CoT span length = teacher_qc_len - question_len (approximate)
            cot_len = max(1, teacher_qc_len - question_len)
            
            result["teacher_qc_ids"] = teacher_qc_enc["input_ids"].squeeze(0)
            result["teacher_qc_mask"] = teacher_qc_enc["attention_mask"].squeeze(0)
            result["teacher_qc_len"] = teacher_qc_len
            result["cot_len"] = cot_len
        
        return result


def abc_collate_fn(batch):
    """Custom collate function with dynamic padding for ABC dataset."""
    # Determine max lengths
    max_teacher_len = max(item["teacher_ids"].shape[0] for item in batch)
    max_student_len = max(item["student_ids"].shape[0] for item in batch)
    max_question_len = max(item["question_ids"].shape[0] for item in batch)
    
    has_qc = "teacher_qc_ids" in batch[0]
    max_qc_len = 0
    if has_qc:
        max_qc_len = max(item["teacher_qc_ids"].shape[0] for item in batch)
    
    pad_id = 0  # Most models use 0 as pad token
    bs = len(batch)
    
    # Initialize padded tensors
    teacher_ids = torch.full((bs, max_teacher_len), pad_id, dtype=torch.long)
    teacher_mask = torch.zeros((bs, max_teacher_len), dtype=torch.long)
    student_ids = torch.full((bs, max_student_len), pad_id, dtype=torch.long)
    student_mask = torch.zeros((bs, max_student_len), dtype=torch.long)
    question_ids = torch.full((bs, max_question_len), pad_id, dtype=torch.long)
    question_mask = torch.zeros((bs, max_question_len), dtype=torch.long)
    
    teacher_lens = []
    student_lens = []
    question_lens = []
    answer_lens = []
    
    if has_qc:
        qc_ids = torch.full((bs, max_qc_len), pad_id, dtype=torch.long)
        qc_mask = torch.zeros((bs, max_qc_len), dtype=torch.long)
        qc_lens = []
        cot_lens = []
    
    for i, item in enumerate(batch):
        t_len = item["teacher_ids"].shape[0]
        s_len = item["student_ids"].shape[0]
        q_len = item["question_ids"].shape[0]
        
        teacher_ids[i, :t_len] = item["teacher_ids"]
        teacher_mask[i, :t_len] = item["teacher_mask"]
        student_ids[i, :s_len] = item["student_ids"]
        student_mask[i, :s_len] = item["student_mask"]
        question_ids[i, :q_len] = item["question_ids"]
        question_mask[i, :q_len] = item["question_mask"]
        
        teacher_lens.append(item["teacher_len"])
        student_lens.append(item["student_len"])
        question_lens.append(item["question_len"])
        answer_lens.append(item["answer_len"])
        
        if has_qc:
            qc_l = item["teacher_qc_ids"].shape[0]
            qc_ids[i, :qc_l] = item["teacher_qc_ids"]
            qc_mask[i, :qc_l] = item["teacher_qc_mask"]
            qc_lens.append(item["teacher_qc_len"])
            cot_lens.append(item["cot_len"])
    
    result = {
        "teacher_ids": teacher_ids,
        "teacher_mask": teacher_mask,
        "student_ids": student_ids,
        "student_mask": student_mask,
        "question_ids": question_ids,
        "question_mask": question_mask,
        "teacher_len": teacher_lens,
        "student_len": student_lens,
        "question_len": question_lens,
        "answer_len": answer_lens,
    }
    
    if has_qc:
        result["teacher_qc_ids"] = qc_ids
        result["teacher_qc_mask"] = qc_mask
        result["teacher_qc_len"] = qc_lens
        result["cot_len"] = cot_lens
    
    return result


# ==================== KL Divergence ====================

def kl_divergence_diag_gaussian(mu_q, sigma_q, mu_p, sigma_p):
    """
    KL(q || p) for diagonal Gaussian distributions.
    
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
    
    kl = 0.5 * (
        torch.log(var_p / var_q) +
        var_q / var_p +
        ((mu_q - mu_p) ** 2) / var_p -
        1.0
    )
    
    # Sum over hidden dimensions, return [B]
    return kl.sum(dim=-1)


# ==================== Diagnostic Utilities ====================

def compute_diagnostic_metrics(
    mu_phi: torch.Tensor,
    sigma_phi: torch.Tensor,
    mu_psi: torch.Tensor,
    sigma_psi: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute posterior-prior gap diagnostic metrics.
    
    Args:
        mu_phi: Prior mean [B, H]
        sigma_phi: Prior std [B, H]
        mu_psi: Posterior mean [B, H]
        sigma_psi: Posterior std [B, H]
    
    Returns:
        Dict with diagnostic metric values (all scalar floats)
    """
    with torch.no_grad():
        # L2 gap: ||mu_psi - mu_phi||_2, averaged over batch
        mu_diff = mu_psi - mu_phi
        l2_gap = mu_diff.norm(dim=-1).mean().item()  # mean over batch
        
        # Cosine similarity between mu_psi and mu_phi, averaged over batch
        cos_gap = F.cosine_similarity(mu_psi, mu_phi, dim=-1).mean().item()
        
        # Mean sigma values
        sigma_phi_mean = sigma_phi.mean().item()
        sigma_psi_mean = sigma_psi.mean().item()
        
        # KL per sample, averaged
        kl_vals = kl_divergence_diag_gaussian(mu_psi, sigma_psi, mu_phi, sigma_phi)
        kl_mean = kl_vals.mean().item()
    
    return {
        "mu_gap_l2": l2_gap,
        "mu_gap_cos": cos_gap,
        "sigma_phi_mean": sigma_phi_mean,
        "sigma_psi_mean": sigma_psi_mean,
        "kl_per_sample": kl_mean,
    }


def save_diagnostics_jsonl(records: List[Dict], filepath: str):
    """Append diagnostic records to a jsonl file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "a") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def save_eval_comparison_csv(records: List[Dict], filepath: str):
    """Save evaluation comparison results to CSV."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    if not records:
        return
    fieldnames = list(records[0].keys())
    file_exists = os.path.exists(filepath)
    with open(filepath, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(records)


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
    
    Posterior modes for ablation:
        q_y_qca: Full teacher features [Q; CoT; Answer]
        q_y_qc:  Partial teacher features [Q; CoT]
        q_y_q:   Question features only (Y = r_Q)
        none:    No posterior, prior-only training
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
        # Posterior mode ablation
        posterior_mode: str = "q_y_qca",
        # Diagnostics
        save_diagnostics: bool = False,
        diagnostics_dir: str = "./outputs",
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
        
        # Posterior mode
        self.posterior_mode = posterior_mode
        
        # Diagnostics
        self.save_diagnostics = save_diagnostics
        self.diagnostics_dir = diagnostics_dir
        
        # Get hidden size from model
        hidden_size = model_wrapper.hidden_size
        self.hidden_size = hidden_size
        
        # z_dim = hidden_size (no projection, direct injection)
        self.z_dim = hidden_size
        
        # Initialize networks
        self.prior_net = PriorNetwork(
            input_dim=hidden_size,
            hidden_dim=abc_hidden_dim,
            output_dim=self.z_dim,
        )
        
        # Posterior: input 2H -> output (mu, sigma) each H
        # Even for posterior_mode="none", we still create the network
        # (just won't use it) to keep state_dict consistent
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
        Extract teacher features Y (privileged, train-only).
        
        Mean pooling over answer token positions at the target layer.
        Used for posterior_mode = "q_y_qca" (default).
        
        Args:
            teacher_ids: [B, Tt]
            teacher_mask: [B, Tt]
            teacher_lens: List of actual teacher lengths
            answer_lens: List of answer token lengths
        
        Returns:
            Y: [B, H]
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
        Extract teacher features from [Q; CoT] prompt (no answer).
        Mean pooling over CoT token positions at the target layer.
        Used for posterior_mode = "q_y_qc".
        
        Args:
            qc_ids: [B, T_qc]
            qc_mask: [B, T_qc]
            qc_lens: List of actual [Q;CoT] sequence lengths
            cot_lens: List of CoT span lengths
        
        Returns:
            Y: [B, H]
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

    def extract_teacher_feature(
        self,
        input_ids: torch.Tensor,
        attn_mask: torch.Tensor,
        seq_lens: List[int],
        span_lens: List[int],
        feature_span_mode: str = "answer_only",
    ) -> torch.Tensor:
        """
        Unified teacher feature extraction with configurable span mode.
        
        Args:
            input_ids: [B, T]
            attn_mask: [B, T]
            seq_lens: actual sequence lengths per sample
            span_lens: span lengths (answer tokens or cot tokens)
            feature_span_mode: one of:
                - "answer_only": pool over last `span_len` tokens (default)
                - "cot_only": pool over last `span_len` tokens (same logic, different semantics)
                - "cot_answer": pool over last `span_len` tokens (span_len = cot+answer)
                - "full_sequence_mean": weighted mean over all tokens
        
        Returns:
            Y: [B, H]
        """
        self.model_wrapper.clear_hooks()
        self.model_wrapper.register_extraction_hook(self.layer_idx, requires_grad=False)
        
        with torch.no_grad():
            self.model_wrapper(input_ids, attention_mask=attn_mask)
        
        hidden_states = self.model_wrapper.get_activations(self.layer_idx)
        
        bs = input_ids.size(0)
        Y_list = []
        
        if feature_span_mode == "full_sequence_mean":
            mask_expanded = attn_mask.unsqueeze(-1).float()
            Y = (hidden_states * mask_expanded).sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-8)
            self.model_wrapper.clear_hooks()
            return Y.detach()
        
        # For answer_only, cot_only, cot_answer: pool over last span_len tokens
        for i in range(bs):
            s_len = seq_lens[i]
            sp_len = span_lens[i]
            span_start = max(0, s_len - sp_len)
            span_end = s_len
            if span_start < span_end:
                Y_i = hidden_states[i, span_start:span_end, :].mean(dim=0)
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
        
        Args:
            logits: [B, T, V]
            labels: [B, T]
            answer_mask: [B, T] (1 for answer tokens, 0 otherwise)
        
        Returns:
            CE loss scalar
        """
        # Shift for next-token prediction
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
    
    def _move_networks_to_device(self, device):
        """Move prior, posterior, gate to target device."""
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
        Supports posterior_mode ablation and optional diagnostic logging.
        
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
        print(f"  Posterior mode: {self.posterior_mode}")
        
        # Create dataset and dataloader
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
        
        # Setup optimizer (only train prior, posterior, gate)
        if self.posterior_mode == "none":
            # No posterior network training when mode is "none"
            params = list(self.prior_net.parameters()) + [self.gate]
        else:
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
        
        # Diagnostics storage
        epoch_diagnostics = []
        
        # Training loop
        global_step = 0
        best_loss = float('inf')
        
        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            epoch_nll = 0.0
            epoch_kl = 0.0
            epoch_mu_gap_l2 = 0.0
            epoch_mu_gap_cos = 0.0
            epoch_sigma_phi_mean = 0.0
            epoch_sigma_psi_mean = 0.0
            epoch_injected_norm = 0.0
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
                    
                    # ========== Step 2: Extract Y based on posterior_mode ==========
                    if self.posterior_mode == "q_y_qca":
                        # Default: Y from full teacher [Q; CoT; Answer], answer span
                        Y = self._extract_teacher_features(
                            teacher_ids, teacher_mask,
                            batch["teacher_len"], batch["answer_len"]
                        )
                    elif self.posterior_mode == "q_y_qc":
                        # Y from [Q; CoT] only, CoT span
                        qc_ids = batch["teacher_qc_ids"].to(target_device)
                        qc_mask = batch["teacher_qc_mask"].to(target_device)
                        Y = self._extract_teacher_features_qc(
                            qc_ids, qc_mask,
                            batch["teacher_qc_len"], batch["cot_len"]
                        )
                    elif self.posterior_mode == "q_y_q":
                        # Y = r_Q (same question features, tests posterior capacity)
                        Y = r_Q.clone()
                    else:
                        # posterior_mode == "none": will skip posterior below
                        Y = None
                    
                    # ========== Step 3: Prior distribution p_phi(z|Q) ==========
                    mu_phi, raw_sigma_phi = self.prior_net(r_Q)
                    sigma_phi = self._get_sigma(raw_sigma_phi)  # [B, H]
                    
                    # ========== Step 4: Posterior or Prior-only ==========
                    if self.posterior_mode == "none":
                        # No posterior: sample from prior, KL = 0
                        mu_psi = mu_phi.detach()
                        sigma_psi = sigma_phi.detach()
                        kl_loss = torch.tensor(0.0, device=target_device)
                    else:
                        # Standard posterior
                        mu_psi, raw_sigma_psi = self.posterior_net(r_Q, Y)
                        sigma_psi = self._get_sigma(raw_sigma_psi)  # [B, H]
                        
                        # KL divergence
                        kl_per_sample = kl_divergence_diag_gaussian(
                            mu_psi, sigma_psi, mu_phi, sigma_phi
                        )
                        kl_loss = kl_per_sample.mean()
                    
                    # ========== Step 5: Reparameterization trick (MC=1) ==========
                    eps = torch.randn_like(mu_psi)
                    z = mu_psi + eps * sigma_psi  # [B, H]
                    
                    # ========== Step 6: Gated injection and NLL computation ==========
                    gated_z = self.gate * z  # [B, H]
                    
                    nll_losses = []
                    
                    for i in range(bs):
                        self.model_wrapper.clear_hooks()
                        
                        s_ids_i = student_ids[i:i+1]
                        s_mask_i = student_mask[i:i+1]
                        gated_z_i = gated_z[i]
                        
                        self.model_wrapper.register_injection_hook(
                            self.layer_idx,
                            vector=gated_z_i,
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
                    
                    # ========== Step 7: ELBO loss ==========
                    beta_t = self.kl_beta
                    if self.kl_warmup_steps > 0:
                        beta_t *= min(1.0, global_step / self.kl_warmup_steps)
                    
                    loss = nll_loss + beta_t * kl_loss
                    
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
                    epoch_loss += loss.item()
                    epoch_nll += nll_loss.item()
                    epoch_kl += kl_loss.item()
                    num_batches += 1
                    
                    # Diagnostic metrics (computed without grad)
                    with torch.no_grad():
                        injected_norm = gated_z.norm(dim=-1).mean().item()
                        epoch_injected_norm += injected_norm
                        
                        if self.posterior_mode != "none":
                            diag = compute_diagnostic_metrics(
                                mu_phi, sigma_phi, mu_psi, sigma_psi
                            )
                            epoch_mu_gap_l2 += diag["mu_gap_l2"]
                            epoch_mu_gap_cos += diag["mu_gap_cos"]
                            epoch_sigma_phi_mean += diag["sigma_phi_mean"]
                            epoch_sigma_psi_mean += diag["sigma_psi_mean"]
                    
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
            n = max(num_batches, 1)
            avg_loss = epoch_loss / n
            avg_nll = epoch_nll / n
            avg_kl = epoch_kl / n
            avg_mu_gap_l2 = epoch_mu_gap_l2 / n
            avg_mu_gap_cos = epoch_mu_gap_cos / n
            avg_sigma_phi = epoch_sigma_phi_mean / n
            avg_sigma_psi = epoch_sigma_psi_mean / n
            avg_injected_norm = epoch_injected_norm / n
            gate_val = self.gate.item()
            
            print(f"  Epoch {epoch+1}: loss={avg_loss:.4f}, nll={avg_nll:.4f}, "
                  f"kl={avg_kl:.4f}, gate={gate_val:.4f}, "
                  f"mu_gap_l2={avg_mu_gap_l2:.4f}, cos={avg_mu_gap_cos:.4f}")
            
            # Build epoch diagnostic record
            epoch_record = {
                "epoch": epoch + 1,
                "train_loss": round(avg_loss, 6),
                "train_nll": round(avg_nll, 6),
                "train_kl": round(avg_kl, 6),
                "train_mu_gap_l2": round(avg_mu_gap_l2, 6),
                "train_mu_gap_cos": round(avg_mu_gap_cos, 6),
                "train_sigma_phi_mean": round(avg_sigma_phi, 6),
                "train_sigma_psi_mean": round(avg_sigma_psi, 6),
                "gate_value": round(gate_val, 6),
                "gate_abs": round(abs(gate_val), 6),
                "injected_norm_mean": round(avg_injected_norm, 6),
                "beta_t": round(beta_t, 6),
                "posterior_mode": self.posterior_mode,
                "layer_idx": self.layer_idx,
            }
            epoch_diagnostics.append(epoch_record)
            
            if wandb_run:
                wandb_run.log({
                    "epoch": epoch + 1,
                    "train/loss": avg_loss,
                    "train/nll": avg_nll,
                    "train/kl": avg_kl,
                    "train/gate": gate_val,
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/beta_t": beta_t,
                    "train/mu_gap_l2": avg_mu_gap_l2,
                    "train/mu_gap_cos": avg_mu_gap_cos,
                    "train/sigma_phi_mean": avg_sigma_phi,
                    "train/sigma_psi_mean": avg_sigma_psi,
                    "train/injected_norm_mean": avg_injected_norm,
                })
            
            # Track best
            if avg_loss < best_loss:
                best_loss = avg_loss
            
            # Clear cache at end of epoch
            torch.cuda.empty_cache()
            gc.collect()
        
        self.trained = True
        print(f"Training complete. Final gate value: {self.gate.item():.4f}")
        
        # Save diagnostics to jsonl if requested
        if self.save_diagnostics and epoch_diagnostics:
            diag_path = os.path.join(
                self.diagnostics_dir,
                f"train_diagnostics_L{self.layer_idx}_{self.posterior_mode}.jsonl"
            )
            save_diagnostics_jsonl(epoch_diagnostics, diag_path)
            print(f"  Training diagnostics saved to {diag_path}")
        
        return None  # ABC returns None (dynamic vectors)
    
    def eval(
        self,
        test_samples: List,
        max_new_tokens: int = 512,
        num_beams: int = 3,
        use_early_stopping: bool = False,
    ) -> Dict[str, Any]:
        """
        Evaluate ABC Vector on test samples using prior mean (standard).
        
        This is the default evaluation: z* = mu_phi(Q) from prior.
        Equivalent to eval_with_prior_mean().
        
        Args:
            test_samples: List of test samples
            max_new_tokens: Max tokens to generate
            num_beams: Number of beams for generation
            use_early_stopping: Use early stopping criteria
        
        Returns:
            Dict with accuracy, correct, total, results
        """
        return self.eval_with_prior_mean(
            test_samples=test_samples,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            use_early_stopping=use_early_stopping,
        )
    
    def eval_with_prior_mean(
        self,
        test_samples: List,
        max_new_tokens: int = 512,
        num_beams: int = 3,
        use_early_stopping: bool = False,
    ) -> Dict[str, Any]:
        """
        Evaluate ABC Vector using prior mean: z* = mu_phi(Q).
        
        This is the standard test-time inference mode.
        Only requires question input (no teacher/privileged information).
        
        Args:
            test_samples: List of test samples
            max_new_tokens: Max tokens to generate
            num_beams: Number of beams
            use_early_stopping: Early stopping flag
        
        Returns:
            Dict with accuracy, correct, total, results, injected_norms
        """
        return self._eval_with_z_source(
            test_samples=test_samples,
            z_source="prior",
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            use_early_stopping=use_early_stopping,
        )
    
    def eval_with_posterior_mean(
        self,
        test_samples: List,
        max_new_tokens: int = 512,
        num_beams: int = 3,
        use_early_stopping: bool = False,
    ) -> Dict[str, Any]:
        """
        Evaluate ABC Vector using posterior mean: z* = mu_psi(Q, Y).
        
        Requires teacher features at eval time (privileged; not usable in production).
        Useful for diagnosing the posterior-prior gap: if posterior eval >> prior eval,
        the prior has not learned to match the posterior well.
        
        Args:
            test_samples: List of test samples
            max_new_tokens: Max tokens to generate
            num_beams: Number of beams
            use_early_stopping: Early stopping flag
        
        Returns:
            Dict with accuracy, correct, total, results, injected_norms
        """
        return self._eval_with_z_source(
            test_samples=test_samples,
            z_source="posterior",
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            use_early_stopping=use_early_stopping,
        )
    
    def _eval_with_z_source(
        self,
        test_samples: List,
        z_source: str = "prior",
        max_new_tokens: int = 512,
        num_beams: int = 3,
        use_early_stopping: bool = False,
    ) -> Dict[str, Any]:
        """
        Internal evaluation with configurable z source.
        
        Args:
            test_samples: List of test samples
            z_source: "prior" for mu_phi(Q), "posterior" for mu_psi(Q, Y)
            max_new_tokens: Max tokens to generate
            num_beams: Number of beams
            use_early_stopping: Early stopping flag
        
        Returns:
            Dict with accuracy, correct, total, results, injected_norms
        """
        if not self.trained:
            print("Warning: ABC Vector not trained yet!")
        
        self.prior_net.eval()
        self.posterior_net.eval()
        
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
        
        correct = 0
        total = len(test_samples)
        results = []
        injected_norms = []
        
        desc = f"ABC Eval-{z_source} (L{self.layer_idx})"
        pbar = tqdm(test_samples, desc=desc, ncols=100)
        
        for sample in pbar:
            try:
                # Build prompts
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
                ).to(target_device)
                
                # Get r_Q
                r_Q = self._extract_question_repr(
                    q_enc["input_ids"],
                    q_enc["attention_mask"],
                )
                
                # Compute z* depending on source
                with torch.no_grad():
                    if z_source == "prior":
                        mu_phi, raw_sigma_phi = self.prior_net(r_Q)
                        z_star = mu_phi  # [1, H]
                    else:
                        # Posterior: need teacher features Y
                        Y = self._get_eval_teacher_features(sample, target_device)
                        mu_phi, _ = self.prior_net(r_Q)
                        mu_psi, _ = self.posterior_net(r_Q, Y)
                        z_star = mu_psi  # [1, H]
                    
                    gated_z = self.gate * z_star  # [1, H]
                    z_norm = gated_z.norm().item()
                    injected_norms.append(z_norm)
                
                # Register injection hook
                self.model_wrapper.clear_hooks()
                self.model_wrapper.register_injection_hook(
                    self.layer_idx,
                    vector=gated_z.squeeze(0),
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
                
                # Extract answer
                predicted = extract_answer_from_text(generated_text, self.dataset_type)
                is_correct = compare_answers(predicted, sample.answer, self.dataset_type)
                
                self.model_wrapper.clear_hooks()
                
                result = {
                    "predicted": predicted,
                    "ground_truth": sample.answer,
                    "correct": is_correct,
                    "generated_text": generated_text,
                    "num_tokens": len(generated_ids),
                    "injected_norm": z_norm,
                }
                results.append(result)
                
                if is_correct:
                    correct += 1
                
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
        avg_norm = sum(injected_norms) / max(len(injected_norms), 1)
        
        self.prior_net.train()
        self.posterior_net.train()
        
        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "results": results,
            "injected_norms": injected_norms,
            "avg_injected_norm": avg_norm,
            "z_source": z_source,
        }
    
    def _get_eval_teacher_features(
        self,
        sample,
        target_device: torch.device,
    ) -> torch.Tensor:
        """
        Extract teacher features Y for a single sample at eval time.
        Used by eval_with_posterior_mean().
        
        Uses the same posterior_mode as training to determine what Y contains.
        
        Args:
            sample: A CoTSample
            target_device: Target device
        
        Returns:
            Y: [1, H] teacher features
        """
        if self.posterior_mode == "q_y_q" or self.posterior_mode == "none":
            # Y = r_Q for q_y_q mode; for none mode, shouldn't be called
            # but handle gracefully
            if self.dataset_type == "mmlu_pro":
                q_prompt = self.prompt_template["non_cot"].format(
                    question=sample.question,
                    choices=sample.choices
                )
            else:
                q_prompt = self.prompt_template["non_cot"].format(
                    question=sample.question
                )
            q_enc = self.tokenizer(
                q_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
            ).to(target_device)
            Y = self._extract_question_repr(q_enc["input_ids"], q_enc["attention_mask"])
            return Y
        
        if self.posterior_mode == "q_y_qc":
            # Y from [Q; CoT], pool over CoT tokens
            if self.dataset_type == "mmlu_pro":
                qc_prompt = self.prompt_template["cot"].format(
                    question=sample.question,
                    choices=sample.choices
                ) + sample.cot
                q_prompt = self.prompt_template["non_cot"].format(
                    question=sample.question,
                    choices=sample.choices
                )
            else:
                qc_prompt = self.prompt_template["cot"].format(
                    question=sample.question
                ) + sample.cot
                q_prompt = self.prompt_template["non_cot"].format(
                    question=sample.question
                )
            
            qc_enc = self.tokenizer(
                qc_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
            ).to(target_device)
            q_enc = self.tokenizer(
                q_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
            ).to(target_device)
            
            qc_len = qc_enc["input_ids"].shape[1]
            q_len = q_enc["input_ids"].shape[1]
            cot_len = max(1, qc_len - q_len)
            
            Y = self._extract_teacher_features_qc(
                qc_enc["input_ids"],
                qc_enc["attention_mask"],
                [qc_len],
                [cot_len],
            )
            return Y
        
        # Default: q_y_qca — Y from [Q; CoT; Answer], answer span
        if self.dataset_type == "mmlu_pro":
            teacher_prompt = self.prompt_template["cot"].format(
                question=sample.question,
                choices=sample.choices
            ) + sample.cot + f"\nThe answer is {sample.answer}"
        else:
            teacher_prompt = self.prompt_template["cot"].format(
                question=sample.question
            ) + sample.cot + f"\nThe answer is {sample.answer}"
        
        t_enc = self.tokenizer(
            teacher_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        ).to(target_device)
        
        answer_text = f"The answer is {sample.answer}"
        answer_ids = self.tokenizer(answer_text, add_special_tokens=False)["input_ids"]
        a_len = len(answer_ids)
        t_len = t_enc["input_ids"].shape[1]
        
        Y = self._extract_teacher_features(
            t_enc["input_ids"],
            t_enc["attention_mask"],
            [t_len],
            [a_len],
        )
        return Y
    
    def run_diagnostic_eval(
        self,
        test_samples: List,
        max_new_tokens: int = 512,
        num_beams: int = 3,
        split_name: str = "test",
        save_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run diagnostic evaluation comparing prior vs posterior injection.
        
        Outputs:
        - posterior_mean accuracy
        - prior_mean accuracy
        - accuracy delta
        - average injected norm for each
        
        Args:
            test_samples: List of test samples
            max_new_tokens: Max tokens to generate
            num_beams: Number of beams
            split_name: Name of the split (for logging)
            save_path: Optional path to save results CSV
        
        Returns:
            Dict with comparison results
        """
        print(f"\n{'='*60}")
        print(f"Diagnostic Eval on {split_name} ({len(test_samples)} samples)")
        print("=" * 60)
        
        # Prior eval (standard)
        print(f"\n[1/2] Prior mean evaluation (z* = mu_phi(Q))...")
        prior_results = self.eval_with_prior_mean(
            test_samples=test_samples,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
        )
        
        # Posterior eval (privileged)
        if self.posterior_mode != "none":
            print(f"\n[2/2] Posterior mean evaluation (z* = mu_psi(Q, Y))...")
            posterior_results = self.eval_with_posterior_mean(
                test_samples=test_samples,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
            )
        else:
            print(f"\n[2/2] Posterior mode is 'none', skipping posterior eval.")
            posterior_results = prior_results  # same when no posterior
        
        # Comparison
        prior_acc = prior_results["accuracy"]
        posterior_acc = posterior_results["accuracy"]
        acc_delta = posterior_acc - prior_acc
        prior_norm = prior_results["avg_injected_norm"]
        posterior_norm = posterior_results["avg_injected_norm"]
        norm_delta = posterior_norm - prior_norm
        
        comparison = {
            "split": split_name,
            "posterior_mode": self.posterior_mode,
            "layer_idx": self.layer_idx,
            "prior_accuracy": round(prior_acc, 4),
            "posterior_accuracy": round(posterior_acc, 4),
            "accuracy_delta": round(acc_delta, 4),
            "prior_avg_norm": round(prior_norm, 6),
            "posterior_avg_norm": round(posterior_norm, 6),
            "norm_delta": round(norm_delta, 6),
            "gate_value": round(self.gate.item(), 6),
            "num_samples": len(test_samples),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        
        print(f"\n{'='*60}")
        print(f"Diagnostic Results ({split_name})")
        print("-" * 60)
        print(f"  Prior accuracy:     {prior_acc:.2f}%  (norm={prior_norm:.4f})")
        print(f"  Posterior accuracy:  {posterior_acc:.2f}%  (norm={posterior_norm:.4f})")
        print(f"  Accuracy delta:     {acc_delta:+.2f}%")
        print(f"  Norm delta:         {norm_delta:+.4f}")
        print(f"  Gate value:         {self.gate.item():.4f}")
        print("=" * 60)
        
        # Save if path provided
        if save_path:
            save_eval_comparison_csv([comparison], save_path)
            print(f"  Diagnostic results saved to {save_path}")
        
        return {
            "comparison": comparison,
            "prior_results": prior_results,
            "posterior_results": posterior_results,
        }
    
    def get_vector(self) -> Optional[torch.Tensor]:
        """ABC uses dynamic vectors (z* per sample), so return None."""
        return None
    
    def get_state_dict(self) -> Dict[str, Any]:
        """Get state dict for saving checkpoint."""
        return {
            "prior": self.prior_net.state_dict(),
            "posterior": self.posterior_net.state_dict(),
            "gate": self.gate.detach().cpu(),
            "layer_idx": self.layer_idx,
            "abc_hidden_dim": self.abc_hidden_dim,
            "kl_beta": self.kl_beta,
            "kl_warmup_steps": self.kl_warmup_steps,
            "sigma_min": self.sigma_min,
            "posterior_mode": self.posterior_mode,
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any], device=None):
        """Load state dict from checkpoint."""
        self.prior_net.load_state_dict(state_dict["prior"])
        self.posterior_net.load_state_dict(state_dict["posterior"])
        self.gate.data = state_dict["gate"]
        
        if device is not None:
            self._move_networks_to_device(device)
        
        self.trained = True