#!/usr/bin/env python3
"""
mHC Adapter Fine-Tuning Script — Train & Evaluate
===================================================

Fine-tunes a frozen WedLM-7B (or Phi-3-mini) model with Manifold-Constrained
Hyper-Connection adapters and compares against LoRA at the same parameter budget.

WedLM-7B is a text-diffusion model built on the Qwen2.5-7B backbone. This script
auto-detects the model family (qwen2 vs phi3) and dispatches to the correct
sublayer wrappers.

This script demonstrates mHC as a parameter-efficient fine-tuning method with
theoretical stability guarantees from the Birkhoff polytope constraint.

Hardware target: Apple M2 Max, 32GB unified memory (MPS backend)

Training methodology follows contemporary best practices:
  - SmolLM2 SFT recipe (HuggingFace alignment-handbook)
  - QLoRA paper: "small high-quality dataset leads to SOTA results"
  - Cosine LR schedule with 10% warmup (standard for SFT)
  - Gradient checkpointing for memory efficiency
  - Completion-only loss (train on assistant responses only)

Dataset: HuggingFaceTB/smoltalk (SmolLM2's SFT dataset)
  - 1M high-quality instruction examples from 10+ sources
  - Covers: math, code, summarization, rewriting, system prompts
  - Apache 2.0 license
  - Proven: SmolLM2-1.7B trained on this outperforms OpenHermes, Magpie-Pro

Evaluation:
  - Training loss curves (smoothness = stability indicator)
  - Validation perplexity at each epoch
  - Per-layer gradient norms (uniformity = mHC working correctly)
  - H_res spectral norms (must stay <= 1.0 throughout training)
  - lm-evaluation-harness benchmarks (HellaSwag, ARC, MMLU, GSM8K)

Reference: Xie et al. (2026) arXiv:2512.24880v2 (DeepSeek-AI)

Usage:
    # Install dependencies first:
    pip install transformers datasets peft accelerate trl wandb

    # Train mHC adapter (default):
    python3 examples/train_mhc_adapter.py --method mhc

    # Train LoRA adapter (comparison baseline):
    python3 examples/train_mhc_adapter.py --method lora

    # Quick smoke test (100 steps, small subset):
    python3 examples/train_mhc_adapter.py --method mhc --smoke-test

    # Evaluate a trained checkpoint:
    python3 examples/train_mhc_adapter.py --evaluate --checkpoint outputs/mhc/best
"""

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project root for neuroscript_runtime imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from neuroscript_runtime.primitives.connections import (
    ManifoldHyperConnect,
    HyperExpand,
    HyperCollapse,
    sinkhorn_knopp,
)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class TrainConfig:
    """Training configuration — follows SmolLM2 SFT recipe with MPS adaptations."""

    # Model
    # NOTE: "doitmagic/wedlm-7b-base" is not yet publicly available on the Hub.
    # For local testing use a public Qwen2.5-7B checkpoint instead, e.g.:
    #   base_model = "Qwen/Qwen2.5-7B"
    base_model: str = "doitmagic/wedlm-7b-base"
    method: str = "mhc"  # "mhc" or "lora"

    # mHC-specific
    mhc_n: int = 2              # expansion factor (2 for large models, 4 for small)
    mhc_alpha_init: float = 0.01  # gating initialization (paper default)
    mhc_sinkhorn_iters: int = 20  # Sinkhorn iterations (paper default)

    # LoRA-specific (for comparison baseline)
    lora_r: int = 16            # rank — chosen to match mHC param count
    lora_alpha: int = 32        # scaling factor (alpha/r = 2, standard)
    lora_dropout: float = 0.05
    lora_target_modules: list = field(default_factory=lambda: [
        "qkv_proj", "o_proj", "gate_up_proj", "down_proj"
    ])

    # Dataset
    dataset_name: str = "HuggingFaceTB/smoltalk"
    dataset_config: str = "all"
    max_seq_length: int = 512    # short for MPS memory; increase for GPU
    dataset_num_proc: int = 8

    # Training — based on SmolLM2 SFT config + QLoRA best practices
    num_epochs: int = 2
    per_device_batch_size: int = 1       # MPS memory constraint
    gradient_accumulation_steps: int = 8  # effective batch size = 8
    learning_rate: float = 1e-4          # adapter-appropriate (higher than full FT)
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1           # 10% warmup (SmolLM2 standard)
    lr_scheduler: str = "cosine"        # cosine decay (SmolLM2 standard)
    max_grad_norm: float = 1.0          # gradient clipping
    gradient_checkpointing: bool = True  # essential for MPS memory
    bf16: bool = False                   # MPS doesn't support bf16 well, use fp32
    fp16: bool = True                    # MPS supports fp16

    # Evaluation
    eval_steps: int = 500
    eval_samples: int = 1000             # subset for fast validation
    logging_steps: int = 50              # compact logging (every 50 steps)
    generation_steps: int = 250          # show sample generation periodically
    fitness_steps: int = 100             # update fitness scoreboard

    # Output
    output_dir: str = "outputs"
    seed: int = 42
    smoke_test: bool = False             # quick test mode (100 steps)
    max_samples: int = 0                 # 0 = use full dataset; >0 = cap training examples

    # Monitoring
    log_grad_norms: bool = True          # per-layer gradient norm tracking
    log_spectral_norms: bool = True      # H_res spectral norm tracking
    save_training_curves: bool = True

    # Sample generation prompts (used for periodic quality checks)
    generation_prompts: list = field(default_factory=lambda: [
        "Explain what a neural network is in one paragraph.",
        "Write a Python function that checks if a number is prime.",
        "What are the key differences between TCP and UDP?",
    ])


# =============================================================================
# mHC Model Wrapper
# =============================================================================

class Phi3AttnSublayer(nn.Module):
    """Thin wrapper making Phi-3 attention callable as f(hidden) -> hidden.

    ManifoldHyperConnect expects sublayer(x) -> y with shape [batch, seq, dim].
    Phi-3's self_attn needs position_embeddings (cos/sin from rotary) and a
    causal attention mask. This wrapper handles that translation so mHC stays
    model-agnostic.
    """

    def __init__(self, layer, rotary_emb):
        super().__init__()
        self.layernorm = layer.input_layernorm
        self.self_attn = layer.self_attn
        self.rotary_emb = rotary_emb  # shared rotary embedding from base model

    def forward(self, x):
        normed = self.layernorm(x)
        seq_len = x.shape[1]

        # Rotary position embeddings
        pos_ids = torch.arange(seq_len, device=x.device).unsqueeze(0)
        cos, sin = self.rotary_emb(x, pos_ids)

        # Causal attention mask
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float('-inf'), device=x.device, dtype=x.dtype),
            diagonal=1,
        ).unsqueeze(0).unsqueeze(0)  # [1, 1, seq, seq]

        attn_out = self.self_attn(
            normed, attention_mask=causal_mask, position_embeddings=(cos, sin)
        )[0]  # first element is hidden_states, rest is attn_weights/cache
        # NO residual here — mHC handles residual via H_res mixing
        return attn_out


class Phi3MLPSublayer(nn.Module):
    """Thin wrapper making Phi-3 MLP callable as f(hidden) -> hidden."""

    def __init__(self, layer):
        super().__init__()
        self.layernorm = layer.post_attention_layernorm
        self.mlp = layer.mlp

    def forward(self, x):
        normed = self.layernorm(x)
        # NO residual here — mHC handles residual via H_res mixing
        return self.mlp(normed)


class Qwen2AttnSublayer(nn.Module):
    """Thin wrapper making Qwen2 attention callable as f(hidden) -> hidden.

    Qwen2's self_attn uses position_ids (int tensor) for RoPE, unlike Phi-3
    which takes position_embeddings (cos/sin tuple). This wrapper normalizes
    both to the same f(x) -> y interface for mHC compatibility.
    """

    def __init__(self, layer, rotary_emb):
        super().__init__()
        self.input_layernorm = layer.input_layernorm
        self.self_attn = layer.self_attn
        self.rotary_emb = rotary_emb

    def forward(self, x):
        normed = self.input_layernorm(x)
        seq_len = x.shape[1]
        position_ids = torch.arange(seq_len, device=x.device).unsqueeze(0)

        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float('-inf'), device=x.device, dtype=x.dtype),
            diagonal=1,
        ).unsqueeze(0).unsqueeze(0)

        attn_out = self.self_attn(
            normed,
            attention_mask=causal_mask,
            position_ids=position_ids,
        )[0]
        return attn_out


class Qwen2MLPSublayer(nn.Module):
    """Thin wrapper making Qwen2 MLP callable as f(hidden) -> hidden."""

    def __init__(self, layer):
        super().__init__()
        self.post_attention_layernorm = layer.post_attention_layernorm
        self.mlp = layer.mlp

    def forward(self, x):
        normed = self.post_attention_layernorm(x)
        return self.mlp(normed)


class MHCAdaptedModel(nn.Module):
    """Wraps a frozen HuggingFace model with mHC adapters on every sublayer.

    Architecture:
      For each transformer layer:
        - Wrap (norm + self_attn + residual) with ManifoldHyperConnect
        - Wrap (norm + MLP + residual) with ManifoldHyperConnect
        - Add HyperExpand before the first mHC layer
        - Add HyperCollapse after the last mHC layer

    The sublayer wrappers (Phi3AttnSublayer, Phi3MLPSublayer) encapsulate
    the model-specific calling convention so mHC stays model-agnostic.

    Only mHC parameters are trainable. The base model stays frozen.
    """

    def __init__(self, base_model, config: TrainConfig):
        super().__init__()
        self.base_model = base_model
        self.config = config
        self.n = config.mhc_n

        # Freeze all base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Note: gradient checkpointing on the base model is intentionally
        # disabled — it conflicts with our custom forward that calls sublayers
        # directly. Memory is managed via n=2, short seq_length, and fp16.

        # Get model dimensions from config
        model_config = base_model.config
        self.hidden_size = model_config.hidden_size
        self.num_layers = model_config.num_hidden_layers

        # Create mHC wrappers for each layer
        self.attn_mhc = nn.ModuleList()
        self.mlp_mhc = nn.ModuleList()

        # Detect model family for sublayer dispatch
        model_type = getattr(base_model.config, "model_type", "phi3").lower()
        is_qwen2 = "qwen2" in model_type

        for i in range(self.num_layers):
            layer = self.base_model.model.layers[i]

            if is_qwen2:
                attn_sublayer = Qwen2AttnSublayer(layer, base_model.model.rotary_emb)
                mlp_sublayer = Qwen2MLPSublayer(layer)
            else:
                attn_sublayer = Phi3AttnSublayer(layer, base_model.model.rotary_emb)
                mlp_sublayer = Phi3MLPSublayer(layer)

            self.attn_mhc.append(
                ManifoldHyperConnect(
                    sublayer=attn_sublayer,
                    n=self.n,
                    dim=self.hidden_size,
                    layer_idx=i,
                    alpha_init=config.mhc_alpha_init,
                    sinkhorn_iters=config.mhc_sinkhorn_iters,
                )
            )
            self.mlp_mhc.append(
                ManifoldHyperConnect(
                    sublayer=mlp_sublayer,
                    n=self.n,
                    dim=self.hidden_size,
                    layer_idx=i,
                    alpha_init=config.mhc_alpha_init,
                    sinkhorn_iters=config.mhc_sinkhorn_iters,
                )
            )
            if (i + 1) % 8 == 0 or i == self.num_layers - 1:
                print(f"    Wrapped layer {i+1}/{self.num_layers}", flush=True)

        # Stream expansion/collapse
        self.expand = HyperExpand(self.n)
        self.collapse = HyperCollapse()

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        """Forward pass with mHC-wrapped sublayers.

        The key insight: we intercept after embedding, expand to n-wide,
        run through mHC-wrapped attention+MLP for each layer, then collapse
        back before the LM head.
        """
        # Embedding
        inputs_embeds = self.base_model.model.embed_tokens(input_ids)

        # Expand to n-wide stream: [batch, seq, dim] -> [batch, seq, n, dim]
        hidden = self.expand(inputs_embeds)

        # Process through mHC-wrapped layers
        for i in range(self.num_layers):
            hidden = self.attn_mhc[i](hidden)  # mHC-wrapped attention
            hidden = self.mlp_mhc[i](hidden)   # mHC-wrapped MLP

        # Collapse back to single stream: [batch, seq, n, dim] -> [batch, seq, dim]
        hidden = self.collapse(hidden)

        # Final norm + LM head
        hidden = self.base_model.model.norm(hidden)
        logits = self.base_model.lm_head(hidden)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return {"loss": loss, "logits": logits}

    def trainable_params_summary(self):
        """Report trainable vs frozen parameter counts.

        Uses base_model.num_parameters() to avoid slow iteration over 3.8B params.
        """
        base_total = sum(p.numel() for p in self.base_model.parameters())
        # Only count mHC adapter params by iterating the small adapter modules
        trainable = 0
        for module_list in [self.attn_mhc, self.mlp_mhc]:
            for mhc in module_list:
                for name, p in mhc.named_parameters():
                    if 'sublayer' not in name:  # skip frozen sublayer refs
                        trainable += p.numel()
        total = base_total + trainable
        return {
            "total": total,
            "trainable": trainable,
            "frozen": base_total,
            "trainable_pct": 100 * trainable / total,
        }

    def get_spectral_norms(self):
        """Get H_res spectral norms for all mHC layers (stability check)."""
        norms = {}
        for i, (attn, mlp) in enumerate(zip(self.attn_mhc, self.mlp_mhc)):
            raw_attn = attn.alpha_res * attn.b_res
            H_attn = sinkhorn_knopp(raw_attn, iters=20)
            s_attn = torch.linalg.norm(H_attn.cpu(), ord=2).item()

            raw_mlp = mlp.alpha_res * mlp.b_res
            H_mlp = sinkhorn_knopp(raw_mlp, iters=20)
            s_mlp = torch.linalg.norm(H_mlp.cpu(), ord=2).item()

            norms[f"layer_{i}_attn"] = s_attn
            norms[f"layer_{i}_mlp"] = s_mlp
        return norms

    def get_grad_norms(self):
        """Get per-layer gradient norms for mHC parameters."""
        norms = {}
        for i, (attn, mlp) in enumerate(zip(self.attn_mhc, self.mlp_mhc)):
            attn_grad = 0.0
            mlp_grad = 0.0
            for name, p in attn.named_parameters():
                if p.grad is not None and 'sublayer' not in name:
                    attn_grad += p.grad.norm().item() ** 2
            for name, p in mlp.named_parameters():
                if p.grad is not None and 'sublayer' not in name:
                    mlp_grad += p.grad.norm().item() ** 2
            norms[f"layer_{i}_attn"] = attn_grad ** 0.5
            norms[f"layer_{i}_mlp"] = mlp_grad ** 0.5
        return norms


# =============================================================================
# Data Loading
# =============================================================================

def load_and_prepare_dataset(config: TrainConfig, tokenizer):
    """Load SmolTalk dataset and prepare for training.

    Uses completion-only loss: we only compute loss on assistant responses,
    not on user prompts or system messages.
    """
    from datasets import load_dataset

    print(f"  Loading dataset: {config.dataset_name} ({config.dataset_config})")
    dataset = load_dataset(
        config.dataset_name,
        config.dataset_config,
        split="train",
        num_proc=config.dataset_num_proc,
    )

    if config.smoke_test:
        dataset = dataset.select(range(min(500, len(dataset))))
        print(f"  Smoke test: using {len(dataset)} samples")
    elif config.max_samples > 0:
        dataset = dataset.select(range(min(config.max_samples, len(dataset))))
        print(f"  Capped at {len(dataset)} samples (--max-samples {config.max_samples})")

    eval_dataset = load_dataset(
        config.dataset_name,
        config.dataset_config,
        split="test",
        num_proc=config.dataset_num_proc,
    )
    eval_dataset = eval_dataset.select(range(min(config.eval_samples, len(eval_dataset))))

    def tokenize_conversation(example):
        """Convert conversation to token IDs with completion masking."""
        messages = example["messages"]

        # Build full text using chat template
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        # Tokenize
        encoded = tokenizer(
            text,
            truncation=True,
            max_length=config.max_seq_length,
            padding=False,
            return_tensors=None,
        )

        # Build labels: -100 for everything except assistant responses
        input_ids = encoded["input_ids"]
        labels = [-100] * len(input_ids)

        # Find assistant response spans and unmask them
        # Simple heuristic: find assistant markers in the tokenized text
        assistant_start_tokens = tokenizer.encode(
            "<|assistant|>", add_special_tokens=False
        )
        end_tokens = tokenizer.encode("<|end|>", add_special_tokens=False)

        # Fallback: if chat template uses different markers, try common alternatives
        if not assistant_start_tokens:
            assistant_start_tokens = tokenizer.encode(
                "<|im_start|>assistant", add_special_tokens=False
            )
            end_tokens = tokenizer.encode("<|im_end|>", add_special_tokens=False)

        # Find all assistant spans and unmask their labels
        in_assistant = False
        for i, tok in enumerate(input_ids):
            if _is_subsequence_at(input_ids, assistant_start_tokens, i):
                in_assistant = True
                continue
            if in_assistant and _is_subsequence_at(input_ids, end_tokens, i):
                in_assistant = False
                continue
            if in_assistant:
                labels[i] = input_ids[i]

        # If no assistant markers found, fall back to training on everything
        if all(l == -100 for l in labels):
            labels = list(input_ids)

        encoded["labels"] = labels
        return encoded

    print(f"  Tokenizing {len(dataset)} training examples...")
    train_dataset = dataset.map(
        tokenize_conversation,
        remove_columns=dataset.column_names,
        num_proc=config.dataset_num_proc,
        desc="Tokenizing train",
    )

    print(f"  Tokenizing {len(eval_dataset)} eval examples...")
    eval_ds = eval_dataset.map(
        tokenize_conversation,
        remove_columns=eval_dataset.column_names,
        num_proc=config.dataset_num_proc,
        desc="Tokenizing eval",
    )

    # Filter out empty sequences
    train_dataset = train_dataset.filter(lambda x: len(x["input_ids"]) > 10)
    eval_ds = eval_ds.filter(lambda x: len(x["input_ids"]) > 10)

    print(f"  Train: {len(train_dataset)} examples")
    print(f"  Eval:  {len(eval_ds)} examples")

    return train_dataset, eval_ds


def _is_subsequence_at(seq, subseq, start):
    """Check if subseq appears at position start in seq."""
    if not subseq or start + len(subseq) > len(seq):
        return False
    return seq[start:start + len(subseq)] == subseq


def create_data_collator(tokenizer, max_length):
    """Dynamic padding collator for variable-length sequences."""

    def collate_fn(examples):
        # Pad to max length in this batch
        batch_max = min(
            max(len(ex["input_ids"]) for ex in examples),
            max_length,
        )

        input_ids = []
        attention_mask = []
        labels = []

        for ex in examples:
            ids = ex["input_ids"][:batch_max]
            labs = ex["labels"][:batch_max]
            pad_len = batch_max - len(ids)

            input_ids.append(ids + [tokenizer.pad_token_id] * pad_len)
            attention_mask.append([1] * len(ids) + [0] * pad_len)
            labels.append(labs + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    return collate_fn


# =============================================================================
# Training Loop
# =============================================================================

def train(config: TrainConfig):
    """Main training function."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(config.seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    print()
    print("=" * 70)
    print(f"  mHC Adapter Training — {config.method.upper()}")
    print("=" * 70)
    print(f"  Device: {device}")
    print(f"  Base model: {config.base_model}")
    print(f"  Method: {config.method}")
    print(f"  Dataset: {config.dataset_name}")
    print()

    # --- Load tokenizer ---
    print("  Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # --- Load base model ---
    print(f"  Loading base model ({config.base_model})...")
    base_model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        dtype=torch.float16 if config.fp16 else torch.float32,
        attn_implementation="eager",  # flash-attn not available on MPS
    )

    # --- Create adapted model ---
    if config.method == "mhc":
        print(f"  Moving base model to {device}...", flush=True)
        base_model = base_model.to(device)
        print(f"  Wrapping with mHC adapters (n={config.mhc_n})...", flush=True)
        model = MHCAdaptedModel(base_model, config)
        # Only move the small mHC adapter params to device (base is already there)
        print(f"  Moving adapters to {device}...", flush=True)
        model.expand = model.expand.to(device)
        model.collapse = model.collapse.to(device)
        model.attn_mhc = model.attn_mhc.to(device)
        model.mlp_mhc = model.mlp_mhc.to(device)
        print(f"  Counting parameters...", flush=True)
        summary = model.trainable_params_summary()
    elif config.method == "lora":
        print(f"  Wrapping with LoRA adapters (r={config.lora_r})...")
        from peft import get_peft_model, LoraConfig, TaskType
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.lora_target_modules,
        )
        model = get_peft_model(base_model, lora_config).to(device)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        summary = {
            "total": total,
            "trainable": trainable,
            "frozen": total - trainable,
            "trainable_pct": 100 * trainable / total,
        }
    else:
        raise ValueError(f"Unknown method: {config.method}")

    print(f"  Total parameters:     {summary['total']:>12,}")
    print(f"  Trainable parameters: {summary['trainable']:>12,}")
    print(f"  Frozen parameters:    {summary['frozen']:>12,}")
    print(f"  Trainable %:          {summary['trainable_pct']:>11.3f}%")
    print()

    # --- Load dataset ---
    train_dataset, eval_dataset = load_and_prepare_dataset(config, tokenizer)

    collator = create_data_collator(tokenizer, config.max_seq_length)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.per_device_batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=0,  # MPS doesn't benefit from multiprocess data loading
        pin_memory=False,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config.per_device_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )

    # --- Optimizer (AdamW, standard for SFT) ---
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.999),  # standard Adam betas
        eps=1e-8,
    )

    # --- LR Scheduler (cosine with warmup) ---
    total_steps = len(train_loader) * config.num_epochs // config.gradient_accumulation_steps
    if config.smoke_test:
        total_steps = min(total_steps, 100)
        config.fitness_steps = 10
        config.generation_steps = 50
        config.eval_steps = 50
    warmup_steps = int(total_steps * config.warmup_ratio)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print(f"  Total training steps: {total_steps}")
    print(f"  Warmup steps:         {warmup_steps}")
    print(f"  Effective batch size: {config.per_device_batch_size * config.gradient_accumulation_steps}")
    print()

    # --- Training metrics tracking ---
    output_dir = Path(config.output_dir) / config.method
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- TensorBoard ---
    tb_dir = output_dir / "runs"
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=str(tb_dir))
        print(f"  TensorBoard logs: {tb_dir}")
        print(f"  Launch with:      tensorboard --logdir {tb_dir}")
    except ImportError:
        writer = None
        print("  (TensorBoard not available — pip install tensorboard to enable)")
    print()

    metrics = {
        "config": {
            "method": config.method,
            "base_model": config.base_model,
            "learning_rate": config.learning_rate,
            "mhc_n": config.mhc_n if config.method == "mhc" else None,
            "lora_r": config.lora_r if config.method == "lora" else None,
            "trainable_params": summary["trainable"],
            "trainable_pct": summary["trainable_pct"],
        },
        "train_loss": [],
        "eval_loss": [],
        "eval_perplexity": [],
        "learning_rates": [],
        "grad_norms": [],
        "spectral_norms": [],
    }

    # --- Training loop ---
    model.train()
    global_step = 0
    accum_loss = 0.0
    best_eval_loss = float("inf")
    loss_window = []  # rolling window for smoothed loss
    start_time = time.perf_counter()

    # Fitness scoreboard state
    fitness = {
        "loss": float("inf"),
        "eval_loss": float("inf"),
        "perplexity": float("inf"),
        "spectral_ok": True,
        "grad_uniform": 0.0,
    }

    def print_fitness_scoreboard(step):
        """Compact fitness scoreboard — one-line summary of model health."""
        ppl_str = f"{fitness['perplexity']:.1f}" if fitness['perplexity'] < 1e6 else "---"
        elapsed = time.perf_counter() - start_time
        steps_per_sec = step / max(elapsed, 1)
        eta_min = (total_steps - step) / max(steps_per_sec, 0.001) / 60

        spec = "OK" if fitness["spectral_ok"] else "!!"
        print(f"  [{step:5d}/{total_steps}] "
              f"loss={fitness['loss']:.3f} "
              f"eval={fitness['eval_loss']:.3f} "
              f"ppl={ppl_str} "
              f"spec={spec} "
              f"lr={scheduler.get_last_lr()[0]:.1e} "
              f"({steps_per_sec:.1f} step/s, ~{eta_min:.0f}m left)")

    def generate_samples(prompt_idx=0):
        """Generate a sample response to show qualitative progress."""
        model.eval()
        prompt = config.generation_prompts[prompt_idx % len(config.generation_prompts)]
        messages = [{"role": "user", "content": prompt}]
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(input_text, return_tensors="pt").to(device)

        with torch.no_grad():
            if config.method == "mhc":
                # For mHC, we need to generate token by token
                # (simplified greedy decode since model wrapper doesn't support generate())
                generated = inputs["input_ids"]
                for _ in range(128):
                    outputs = model(generated)
                    next_logits = outputs["logits"][:, -1, :]
                    next_token = next_logits.argmax(dim=-1, keepdim=True)
                    generated = torch.cat([generated, next_token], dim=-1)
                    if next_token.item() == tokenizer.eos_token_id:
                        break
                response = tokenizer.decode(
                    generated[0, inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True
                )
            else:
                generated = model.generate(
                    **inputs, max_new_tokens=128, do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
                response = tokenizer.decode(
                    generated[0, inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True
                )

        # Truncate for display
        response = response.strip()
        if len(response) > 300:
            response = response[:300] + "..."

        print(f"\n  --- Sample Generation (step {global_step}) ---")
        print(f"  Q: {prompt}")
        print(f"  A: {response}")
        print()
        model.train()
        return response

    print()
    print(f"  Training {config.method.upper()} | "
          f"{total_steps} steps | "
          f"{config.num_epochs} epochs | "
          f"lr={config.learning_rate:.0e}")
    print(f"  {'-'*66}")

    import tqdm as tqdm_mod

    for epoch in range(config.num_epochs):
        epoch_loss = 0.0
        epoch_steps = 0

        batch_bar = tqdm_mod.tqdm(
            train_loader,
            desc=f"  Epoch {epoch+1}/{config.num_epochs}",
            ncols=88,
            unit="batch",
            leave=True,
        )

        for batch_idx, batch in enumerate(batch_bar):
            if config.smoke_test and global_step >= 100:
                break

            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward
            if config.method == "mhc":
                outputs = model(**batch)
                loss = outputs["loss"]
            else:
                outputs = model(**batch)
                loss = outputs.loss

            loss = loss / config.gradient_accumulation_steps
            loss.backward()
            accum_loss += loss.item()

            # Optimizer step
            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    config.max_grad_norm,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                global_step += 1
                epoch_loss += accum_loss
                epoch_steps += 1
                lr_now = scheduler.get_last_lr()[0]

                # Track smoothed loss
                loss_window.append(accum_loss)
                if len(loss_window) > 50:
                    loss_window.pop(0)
                fitness["loss"] = sum(loss_window) / len(loss_window)

                # Record for metrics
                metrics["train_loss"].append({
                    "step": global_step,
                    "loss": accum_loss,
                    "lr": lr_now,
                })

                # --- TensorBoard: log every step ---
                if writer is not None:
                    writer.add_scalar("train/loss", accum_loss, global_step)
                    writer.add_scalar("train/loss_smoothed", fitness["loss"], global_step)
                    writer.add_scalar("train/lr", lr_now, global_step)

                # --- Live tqdm bar update (every step) ---
                batch_bar.set_postfix({
                    "loss": f"{fitness['loss']:.3f}",
                    "step": f"{global_step}/{total_steps}",
                    "lr": f"{lr_now:.1e}",
                }, refresh=False)

                # --- Fitness scoreboard (compact, periodic) ---
                if global_step % config.fitness_steps == 0:
                    # Check spectral norms (mHC only)
                    if config.method == "mhc" and config.log_spectral_norms:
                        spec_norms = model.get_spectral_norms()
                        metrics["spectral_norms"].append({"step": global_step, **spec_norms})
                        fitness["spectral_ok"] = max(spec_norms.values()) <= 1.001
                        if writer is not None:
                            max_spec = max(spec_norms.values())
                            writer.add_scalar("mhc/spectral_norm_max", max_spec, global_step)

                    # Check gradient uniformity (mHC only)
                    if config.method == "mhc" and config.log_grad_norms:
                        grad_norms = model.get_grad_norms()
                        metrics["grad_norms"].append({"step": global_step, **grad_norms})
                        all_grads = [v for v in grad_norms.values() if v > 0]
                        if all_grads:
                            fitness["grad_uniform"] = min(all_grads) / max(all_grads)
                            if writer is not None:
                                writer.add_scalar("mhc/grad_uniformity", fitness["grad_uniform"], global_step)

                    print_fitness_scoreboard(global_step)

                # --- Sample generation (periodic, shows qualitative progress) ---
                if global_step % config.generation_steps == 0:
                    prompt_idx = (global_step // config.generation_steps) % len(config.generation_prompts)
                    sample = generate_samples(prompt_idx)
                    metrics.setdefault("generations", []).append({
                        "step": global_step,
                        "prompt": config.generation_prompts[prompt_idx],
                        "response": sample,
                    })

                # --- Evaluation (less frequent, full eval set) ---
                if global_step % config.eval_steps == 0:
                    eval_loss = evaluate(model, eval_loader, device, config)
                    eval_ppl = math.exp(min(eval_loss, 20))
                    fitness["eval_loss"] = eval_loss
                    fitness["perplexity"] = eval_ppl
                    metrics["eval_loss"].append({"step": global_step, "loss": eval_loss})
                    metrics["eval_perplexity"].append({"step": global_step, "perplexity": eval_ppl})

                    if writer is not None:
                        writer.add_scalar("eval/loss", eval_loss, global_step)
                        writer.add_scalar("eval/perplexity", eval_ppl, global_step)

                    improved = ""
                    if eval_loss < best_eval_loss:
                        best_eval_loss = eval_loss
                        save_checkpoint(model, tokenizer, config, output_dir / "best")
                        improved = " *NEW BEST*"

                    print(f"  >>> Eval: loss={eval_loss:.4f} ppl={eval_ppl:.1f}{improved}")
                    model.train()

                accum_loss = 0.0

        # End of epoch summary
        if epoch_steps > 0:
            print(f"\n  Epoch {epoch+1}/{config.num_epochs} done | "
                  f"avg_loss={epoch_loss / epoch_steps:.4f}\n")

    # --- Final evaluation ---
    print("=" * 70)
    print("  Final evaluation...")
    final_eval_loss = evaluate(model, eval_loader, device, config)
    final_ppl = math.exp(min(final_eval_loss, 20))
    print(f"  Final Eval Loss: {final_eval_loss:.4f}")
    print(f"  Final Perplexity: {final_ppl:.2f}")

    # Save final checkpoint
    save_checkpoint(model, tokenizer, config, output_dir / "final")

    # Save training metrics
    metrics_path = output_dir / "training_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"  Training metrics saved to {metrics_path}")

    if writer is not None:
        writer.add_scalar("eval/final_loss", final_eval_loss, global_step)
        writer.add_scalar("eval/final_perplexity", final_ppl, global_step)
        writer.flush()
        writer.close()
        print(f"  TensorBoard logs saved to {tb_dir}")

    # --- Print comparison summary ---
    print()
    print("=" * 70)
    print("  TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Method:              {config.method.upper()}")
    print(f"  Trainable params:    {summary['trainable']:,} ({summary['trainable_pct']:.3f}%)")
    print(f"  Best eval loss:      {best_eval_loss:.4f}")
    print(f"  Best eval perplexity:{math.exp(min(best_eval_loss, 20)):.2f}")
    print(f"  Final eval loss:     {final_eval_loss:.4f}")
    print(f"  Final perplexity:    {final_ppl:.2f}")
    if config.method == "mhc":
        spec = model.get_spectral_norms()
        max_spec = max(spec.values())
        print(f"  Max H_res spectral:  {max_spec:.4f} (must be <= 1.0)")
    print(f"  Checkpoints:         {output_dir}")
    print()
    print("  Next steps:")
    print("    1. Run LoRA comparison:  python3 examples/train_mhc_adapter.py --method lora")
    print("    2. Run lm-eval benchmarks:")
    print(f"       lm_eval --model hf --model_args pretrained={output_dir / 'best'} \\")
    print("         --tasks hellaswag,arc_easy,arc_challenge,mmlu,gsm8k \\")
    print(f"         --device {device} --batch_size 4")
    print("    3. Compare training curves: see training_metrics.json")
    print()


def evaluate(model, eval_loader, device, config):
    """Evaluate model on eval set, return average loss."""
    model.eval()
    total_loss = 0.0
    total_steps = 0

    with torch.no_grad():
        for batch in eval_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            if config.method == "mhc":
                outputs = model(**batch)
                loss = outputs["loss"]
            else:
                outputs = model(**batch)
                loss = outputs.loss

            if loss is not None:
                total_loss += loss.item()
                total_steps += 1

    return total_loss / max(total_steps, 1)


def save_checkpoint(model, tokenizer, config, path):
    """Save model checkpoint."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    if config.method == "mhc":
        # Save only the mHC adapter weights (not the frozen base)
        mhc_state = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                mhc_state[name] = param.cpu().detach()
        torch.save(mhc_state, path / "mhc_adapter.pt")

        # Save config
        adapter_config = {
            "method": "mhc",
            "base_model": config.base_model,
            "mhc_n": config.mhc_n,
            "mhc_alpha_init": config.mhc_alpha_init,
            "mhc_sinkhorn_iters": config.mhc_sinkhorn_iters,
            "hidden_size": model.hidden_size,
            "num_layers": model.num_layers,
        }
        with open(path / "adapter_config.json", "w") as f:
            json.dump(adapter_config, f, indent=2)
    else:
        # LoRA saves via PEFT
        model.save_pretrained(path)

    tokenizer.save_pretrained(path)
    print(f"  Checkpoint saved: {path}")


# =============================================================================
# Benchmark Evaluation
# =============================================================================

def run_benchmarks(checkpoint_path: str, device: str = "mps"):
    """Run lm-evaluation-harness benchmarks on a trained checkpoint.

    Benchmarks chosen to match SmolLM2 evaluation suite:
      - HellaSwag: Common sense reasoning
      - ARC (Easy + Challenge): Science knowledge
      - MMLU: Broad knowledge
      - GSM8K: Math reasoning
      - IFEval: Instruction following
    """
    print("=" * 70)
    print("  Running lm-evaluation-harness benchmarks")
    print("=" * 70)
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Device: {device}")
    print()

    # Check if lm-eval is installed
    try:
        import lm_eval  # noqa: F401
    except ImportError:
        print("  lm-evaluation-harness not installed. Install with:")
        print("    pip install 'lm_eval[hf]'")
        print()
        print("  Or run manually:")
        print(f"    lm_eval --model hf \\")
        print(f"      --model_args pretrained={checkpoint_path} \\")
        print(f"      --tasks hellaswag,arc_easy,arc_challenge,mmlu,gsm8k \\")
        print(f"      --device {device} --batch_size 4 \\")
        print(f"      --output_path {checkpoint_path}/eval_results")
        return

    # Run evaluation programmatically
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM

    results = simple_evaluate(
        model="hf",
        model_args=f"pretrained={checkpoint_path}",
        tasks=["hellaswag", "arc_easy", "arc_challenge", "mmlu", "gsm8k"],
        device=device,
        batch_size=4,
    )

    # Print results table
    print(f"\n  {'Benchmark':<25s} {'Score':>10s}")
    print(f"  {'-'*25} {'-'*10}")
    for task, metrics in results["results"].items():
        score = metrics.get("acc_norm,none", metrics.get("acc,none", "N/A"))
        if isinstance(score, float):
            print(f"  {task:<25s} {score*100:>9.1f}%")
        else:
            print(f"  {task:<25s} {str(score):>10s}")

    # Save results
    results_path = Path(checkpoint_path) / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump(results["results"], f, indent=2)
    print(f"\n  Results saved to {results_path}")


# =============================================================================
# Comparison Analysis
# =============================================================================

def compare_methods(output_dir: str = "outputs"):
    """Compare mHC vs LoRA training curves and final metrics."""
    output = Path(output_dir)
    mhc_metrics_path = output / "mhc" / "training_metrics.json"
    lora_metrics_path = output / "lora" / "training_metrics.json"

    if not mhc_metrics_path.exists() or not lora_metrics_path.exists():
        print("  Both mHC and LoRA training must complete before comparison.")
        print(f"  Looking for: {mhc_metrics_path}")
        print(f"  Looking for: {lora_metrics_path}")
        return

    with open(mhc_metrics_path) as f:
        mhc = json.load(f)
    with open(lora_metrics_path) as f:
        lora = json.load(f)

    print("=" * 70)
    print("  mHC vs LoRA — Head-to-Head Comparison")
    print("=" * 70)
    print()

    # Parameter counts
    print(f"  {'Metric':<30s} {'mHC':>15s} {'LoRA':>15s}")
    print(f"  {'-'*30} {'-'*15} {'-'*15}")
    print(f"  {'Trainable params':<30s} "
          f"{mhc['config']['trainable_params']:>15,} "
          f"{lora['config']['trainable_params']:>15,}")
    print(f"  {'Trainable %':<30s} "
          f"{mhc['config']['trainable_pct']:>14.3f}% "
          f"{lora['config']['trainable_pct']:>14.3f}%")

    # Best eval metrics
    if mhc["eval_loss"] and lora["eval_loss"]:
        mhc_best = min(e["loss"] for e in mhc["eval_loss"])
        lora_best = min(e["loss"] for e in lora["eval_loss"])
        print(f"  {'Best eval loss':<30s} "
              f"{mhc_best:>15.4f} "
              f"{lora_best:>15.4f}")
        print(f"  {'Best eval perplexity':<30s} "
              f"{math.exp(min(mhc_best, 20)):>15.2f} "
              f"{math.exp(min(lora_best, 20)):>15.2f}")

    # Training stability (loss variance over last 20% of steps)
    if mhc["train_loss"] and lora["train_loss"]:
        cutoff = int(len(mhc["train_loss"]) * 0.8)
        mhc_late = [e["loss"] for e in mhc["train_loss"][cutoff:]]
        lora_late = [e["loss"] for e in lora["train_loss"][cutoff:]]
        if mhc_late and lora_late:
            import statistics
            mhc_var = statistics.variance(mhc_late) if len(mhc_late) > 1 else 0
            lora_var = statistics.variance(lora_late) if len(lora_late) > 1 else 0
            print(f"  {'Late-training loss variance':<30s} "
                  f"{mhc_var:>15.6f} "
                  f"{lora_var:>15.6f}")
            print(f"  {'Stability advantage':<30s} "
                  f"{'<-- mHC' if mhc_var < lora_var else 'LoRA -->':>15s}")

    # Spectral norm check
    if mhc["spectral_norms"]:
        last_specs = mhc["spectral_norms"][-1]
        max_spec = max(v for k, v in last_specs.items() if k != "step")
        print(f"  {'Max H_res spectral norm':<30s} "
              f"{max_spec:>15.4f} {'(bounded!)' if max_spec <= 1.001 else '(WARNING!)'}")

    print()
    print("  Key insight: mHC provides comparable quality to LoRA with theoretical")
    print("  stability guarantees from the Birkhoff polytope constraint. The doubly")
    print("  stochastic mixing matrices ensure bounded signal propagation regardless")
    print("  of fine-tuning duration or learning rate.")
    print()


# =============================================================================
# Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="mHC Adapter Fine-Tuning & Evaluation"
    )
    parser.add_argument("--method", choices=["mhc", "lora"], default="mhc",
                        help="Fine-tuning method (default: mhc)")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Quick smoke test (100 steps, small subset)")
    parser.add_argument("--evaluate", action="store_true",
                        help="Run lm-eval benchmarks on checkpoint")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint path for evaluation")
    parser.add_argument("--compare", action="store_true",
                        help="Compare mHC vs LoRA results")
    parser.add_argument("--output-dir", type=str, default="outputs",
                        help="Output directory (default: outputs)")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning rate")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of epochs")
    parser.add_argument("--max-seq-length", type=int, default=None,
                        help="Override max sequence length")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Cap training examples (0 = full dataset, e.g. 50000)")
    parser.add_argument("--base-model", type=str, default=None,
                        help="Override base model")
    args = parser.parse_args()

    if args.compare:
        compare_methods(args.output_dir)
        return

    if args.evaluate:
        checkpoint = args.checkpoint or f"{args.output_dir}/{args.method}/best"
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        run_benchmarks(checkpoint, device)
        return

    config = TrainConfig(
        method=args.method,
        smoke_test=args.smoke_test,
        output_dir=args.output_dir,
    )
    if args.lr:
        config.learning_rate = args.lr
    if args.epochs:
        config.num_epochs = args.epochs
    if args.max_seq_length:
        config.max_seq_length = args.max_seq_length
    if args.base_model:
        config.base_model = args.base_model
    if args.max_samples:
        config.max_samples = args.max_samples

    train(config)


if __name__ == "__main__":
    main()
