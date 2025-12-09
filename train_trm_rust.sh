#!/bin/bash
# TRM Training Script for Rust Autocompletion on M2 Max MacBook
# Based on "Less is More: Recursive Reasoning with Tiny Networks"

set -e  # Exit on error

echo "==================================================================="
echo "TRM Rust Autocompletion Training - M2 Max Optimized"
echo "==================================================================="

# Configuration
PROJECT_DIR="/Users/tquick/projects/neuroscript-rs"
DATA_DIR="$PROJECT_DIR/data/rust_code"
CHECKPOINT_DIR="$PROJECT_DIR/checkpoints/trm_rust"
VENV_DIR="$HOME/.venv_trm"

# Model configuration (optimized for M2 Max - 38-core GPU, 96GB RAM)
VOCAB_SIZE=8000          # Start smaller for faster iteration
HIDDEN_SIZE=256          # TRM paper uses 512, but 256 is good for testing
SEQ_LEN=128             # Context window
BATCH_SIZE=128          # Adjust based on M2 Max memory (can go higher)
N_RECURSIONS=6          # Paper's optimal setting
T_CYCLES=3              # Deep recursion cycles (paper's optimal)
MAX_SUPERVISION=16      # Deep supervision steps
LEARNING_RATE=1e-4      # Paper's setting
WEIGHT_DECAY=0.1        # Paper's setting for code (less than 1.0 for puzzles)

# Training configuration
MAX_STEPS=100000        # Total training steps
EVAL_INTERVAL=1000      # Evaluate every N steps
SAVE_INTERVAL=5000      # Save checkpoint every N steps
LOG_INTERVAL=100        # Log every N steps

# M2 Max specific optimizations
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0  # Use all GPU memory
export PYTORCH_ENABLE_MPS_FALLBACK=1         # Fallback to CPU if needed

echo ""
echo "Configuration:"
echo "  Vocab Size: $VOCAB_SIZE"
echo "  Hidden Size: $HIDDEN_SIZE"
echo "  Sequence Length: $SEQ_LEN"
echo "  Batch Size: $BATCH_SIZE"
echo "  Recursions per cycle: $N_RECURSIONS"
echo "  Deep recursion cycles: $T_CYCLES"
echo "  Effective depth: $((T_CYCLES * (N_RECURSIONS + 1) * 2)) layers"
echo ""

# Step 1: Create virtual environment
echo "Step 1: Setting up Python environment..."
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
    echo "  Created virtual environment at $VENV_DIR"
else
    echo "  Using existing virtual environment"
fi

# Activate virtual environment
source "$VENV_DIR/bin/activate"

# Step 2: Install dependencies
echo ""
echo "Step 2: Installing dependencies..."
pip install --upgrade pip setuptools wheel

# Install PyTorch with MPS (Metal Performance Shaders) support for M2 Max
pip install torch torchvision torchaudio

# Install other dependencies
pip install \
    transformers \
    datasets \
    tokenizers \
    accelerate \
    wandb \
    tensorboard \
    numpy \
    tqdm \
    pyyaml \
    sentencepiece

# Install NeuroScript runtime
cd "$PROJECT_DIR"
pip install -e .

echo "  Dependencies installed"

# Step 3: Download and prepare dataset
echo ""
echo "Step 3: Preparing Rust code dataset..."
mkdir -p "$DATA_DIR"

python3 << 'PYTHON_SCRIPT'
import os
from datasets import load_dataset
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
import json

data_dir = os.environ['DATA_DIR']
vocab_size = int(os.environ['VOCAB_SIZE'])
seq_len = int(os.environ['SEQ_LEN'])

print("  Downloading The Stack (Rust subset)...")
# The Stack is gated - user needs to accept terms at:
# https://huggingface.co/datasets/bigcode/the-stack-dedup

# For now, use a smaller public Rust dataset or create synthetic data
print("  Note: The Stack requires authentication. Using alternative...")

# Option 1: Use smaller systems programming dataset
try:
    dataset = load_dataset("dougiefresh/systems_programming_code_conversations", split="train")
    print(f"  Loaded {len(dataset)} examples from systems_programming_code_conversations")
    
    # Extract Rust code
    rust_code = []
    for item in dataset:
        if 'rust' in str(item).lower():
            # Extract code from conversations
            if 'conversation' in item:
                for turn in item['conversation']:
                    if 'content' in turn:
                        rust_code.append(turn['content'])
    
    print(f"  Extracted {len(rust_code)} Rust code snippets")
    
    # Save to file for tokenizer training
    with open(f"{data_dir}/rust_code.txt", "w") as f:
        for code in rust_code[:10000]:  # Limit for faster training
            f.write(code + "\n")
    
except Exception as e:
    print(f"  Could not load dataset: {e}")
    print("  Creating synthetic Rust dataset...")
    
    # Create synthetic Rust code for testing
    synthetic_rust = [
        "fn main() { println!(\"Hello, world!\"); }",
        "struct Point { x: i32, y: i32 }",
        "impl Point { fn new(x: i32, y: i32) -> Self { Point { x, y } } }",
        "use std::collections::HashMap;",
        "let mut map = HashMap::new();",
        "match value { Some(x) => x, None => 0 }",
        "pub fn add(a: i32, b: i32) -> i32 { a + b }",
    ] * 1000  # Repeat for larger dataset
    
    with open(f"{data_dir}/rust_code.txt", "w") as f:
        for code in synthetic_rust:
            f.write(code + "\n")
    
    print(f"  Created {len(synthetic_rust)} synthetic examples")

# Step 4: Train tokenizer
print("  Training BPE tokenizer...")
tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

trainer = trainers.BpeTrainer(
    vocab_size=vocab_size,
    special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
    min_frequency=2
)

# Train on Rust code
tokenizer.train([f"{data_dir}/rust_code.txt"], trainer)
tokenizer.save(f"{data_dir}/rust_tokenizer.json")
print(f"  Tokenizer trained with vocab_size={vocab_size}")

# Step 5: Tokenize and prepare data
print("  Tokenizing dataset...")
with open(f"{data_dir}/rust_code.txt", "r") as f:
    texts = f.readlines()

all_tokens = []
for text in texts:
    encoding = tokenizer.encode(text)
    all_tokens.extend(encoding.ids)

# Create training sequences
sequences = []
for i in range(0, len(all_tokens) - seq_len, seq_len // 2):  # 50% overlap
    seq = all_tokens[i:i + seq_len]
    if len(seq) == seq_len:
        sequences.append(seq)

print(f"  Created {len(sequences)} training sequences of length {seq_len}")

# Split train/val
split_idx = int(len(sequences) * 0.9)
train_seqs = sequences[:split_idx]
val_seqs = sequences[split_idx:]

# Save as JSON
with open(f"{data_dir}/train.json", "w") as f:
    json.dump(train_seqs, f)

with open(f"{data_dir}/val.json", "w") as f:
    json.dump(val_seqs, f)

print(f"  Saved {len(train_seqs)} training, {len(val_seqs)} validation sequences")
print("  Data preparation complete!")

PYTHON_SCRIPT

echo "  Dataset prepared in $DATA_DIR"

# Step 4: Create training script
echo ""
echo "Step 4: Creating TRM training script..."

cat > "$PROJECT_DIR/train_trm.py" << 'TRAIN_SCRIPT'
"""
Tiny Recursive Model (TRM) Training Script
Based on arXiv:2510.04871v1
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from tqdm import tqdm
from datetime import datetime

# M2 Max: Use MPS (Metal Performance Shaders)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Load configuration from environment
VOCAB_SIZE = int(os.environ.get('VOCAB_SIZE', 8000))
HIDDEN_SIZE = int(os.environ.get('HIDDEN_SIZE', 256))
SEQ_LEN = int(os.environ.get('SEQ_LEN', 128))
BATCH_SIZE = int(os.environ.get('BATCH_SIZE', 128))
N_RECURSIONS = int(os.environ.get('N_RECURSIONS', 6))
T_CYCLES = int(os.environ.get('T_CYCLES', 3))
MAX_SUPERVISION = int(os.environ.get('MAX_SUPERVISION', 16))
LR = float(os.environ.get('LEARNING_RATE', 1e-4))
WEIGHT_DECAY = float(os.environ.get('WEIGHT_DECAY', 0.1))
MAX_STEPS = int(os.environ.get('MAX_STEPS', 100000))
EVAL_INTERVAL = int(os.environ.get('EVAL_INTERVAL', 1000))
SAVE_INTERVAL = int(os.environ.get('SAVE_INTERVAL', 5000))
LOG_INTERVAL = int(os.environ.get('LOG_INTERVAL', 100))

DATA_DIR = os.environ['DATA_DIR']
CHECKPOINT_DIR = os.environ['CHECKPOINT_DIR']
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Dataset
class RustCodeDataset(Dataset):
    def __init__(self, filepath):
        with open(filepath, 'r') as f:
            self.sequences = json.load(f)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        # Input: all but last token, Target: all but first token
        x = torch.tensor(seq[:-1], dtype=torch.long)
        y = torch.tensor(seq[1:], dtype=torch.long)
        return x, y

# TRM Model Components
class TinyNet2Layer(nn.Module):
    """2-layer MLP - paper shows this is optimal"""
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * 4)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim * 4, dim)
    
    def forward(self, x):
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class LatentUpdate(nn.Module):
    """Update latent reasoning: z_new = net(x + y + z)"""
    def __init__(self, dim):
        super().__init__()
        self.net = TinyNet2Layer(dim)
    
    def forward(self, x, y, z):
        combined = x + y + z
        return self.net(combined)

class AnswerUpdate(nn.Module):
    """Update answer: y_new = net(y + z)"""
    def __init__(self, dim):
        super().__init__()
        self.net = TinyNet2Layer(dim)
    
    def forward(self, y, z):
        combined = y + z
        return self.net(combined)

class RecursionCycle(nn.Module):
    """One cycle: n latent updates + 1 answer update"""
    def __init__(self, dim, n_recursions=6):
        super().__init__()
        self.n = n_recursions
        self.latent_update = LatentUpdate(dim)
        self.answer_update = AnswerUpdate(dim)
    
    def forward(self, x, y, z):
        # n latent updates
        for _ in range(self.n):
            z = self.latent_update(x, y, z)
        
        # 1 answer update
        y = self.answer_update(y, z)
        
        return y, z

class TRM(nn.Module):
    """Tiny Recursive Model"""
    def __init__(self, vocab_size, dim, seq_len, n_recursions=6, t_cycles=3):
        super().__init__()
        self.dim = dim
        self.t = t_cycles
        
        # Embeddings
        self.embed_x = nn.Embedding(vocab_size, dim)
        self.embed_y = nn.Embedding(vocab_size, dim)
        self.embed_z = nn.Embedding(vocab_size, dim)
        
        # Recursion
        self.cycle = RecursionCycle(dim, n_recursions)
        
        # Output
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, vocab_size)
        
        # Halting head (for ACT)
        self.halt_head = nn.Linear(dim, 1)
    
    def forward(self, input_ids, max_supervision_steps=16):
        """
        Forward with deep supervision
        Returns logits and halting probabilities
        """
        batch_size, seq_len = input_ids.shape
        
        # Initial embeddings
        x = self.embed_x(input_ids)
        y = self.embed_y(input_ids)
        z = self.embed_z(input_ids)
        
        # Deep supervision loop
        all_logits = []
        all_halt_probs = []
        
        for step in range(max_supervision_steps):
            # T-1 cycles without gradients
            with torch.no_grad():
                for _ in range(self.t - 1):
                    y, z = self.cycle(x, y, z)
            
            # Last cycle with gradients
            y, z = self.cycle(x, y, z)
            
            # Project to vocabulary
            logits = self.proj(self.norm(y))
            all_logits.append(logits)
            
            # Halting probability
            halt_logits = self.halt_head(y).squeeze(-1)
            halt_prob = torch.sigmoid(halt_logits).mean()
            all_halt_probs.append(halt_prob)
            
            # Detach for next iteration
            y = y.detach()
            z = z.detach()
            
            # Early stopping (ACT)
            if halt_prob > 0.5:
                break
        
        return all_logits, all_halt_probs

# Training
def train_step(model, batch, optimizer, step):
    """Single training step with deep supervision"""
    x, y = batch
    x, y = x.to(device), y.to(device)
    
    # Forward
    all_logits, all_halt_probs = model(x, MAX_SUPERVISION)
    
    # Loss: sum over all supervision steps
    total_loss = 0
    for logits in all_logits:
        loss = F.cross_entropy(
            logits.reshape(-1, VOCAB_SIZE),
            y.reshape(-1)
        )
        total_loss += loss
    
    # Average over steps
    total_loss = total_loss / len(all_logits)
    
    # Backward
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    # Metrics
    with torch.no_grad():
        # Final step accuracy
        final_logits = all_logits[-1]
        preds = final_logits.argmax(dim=-1)
        acc = (preds == y).float().mean().item()
        
        # Perplexity
        perplexity = torch.exp(total_loss).item()
    
    return {
        'loss': total_loss.item(),
        'accuracy': acc,
        'perplexity': perplexity,
        'supervision_steps': len(all_logits),
        'final_halt_prob': all_halt_probs[-1].item(),
    }

def evaluate(model, dataloader):
    """Evaluation with full supervision"""
    model.eval()
    total_loss = 0
    total_acc = 0
    count = 0
    
    with torch.no_grad():
        for batch in dataloader:
            x, y = batch
            x, y = x.to(device), y.to(device)
            
            # Forward (use max supervision at test time)
            all_logits, _ = model(x, MAX_SUPERVISION)
            
            # Use final prediction
            logits = all_logits[-1]
            loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1))
            
            preds = logits.argmax(dim=-1)
            acc = (preds == y).float().mean().item()
            
            total_loss += loss.item()
            total_acc += acc
            count += 1
    
    model.train()
    return {
        'loss': total_loss / count,
        'accuracy': total_acc / count,
        'perplexity': np.exp(total_loss / count),
    }

def main():
    print(f"\nTRM Training Configuration:")
    print(f"  Device: {device}")
    print(f"  Vocab Size: {VOCAB_SIZE}")
    print(f"  Hidden Size: {HIDDEN_SIZE}")
    print(f"  Sequence Length: {SEQ_LEN}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Recursions per cycle: {N_RECURSIONS}")
    print(f"  Deep recursion cycles: {T_CYCLES}")
    print(f"  Effective depth: {T_CYCLES * (N_RECURSIONS + 1) * 2} layers")
    print(f"  Max supervision steps: {MAX_SUPERVISION}")
    print(f"  Learning Rate: {LR}")
    print(f"  Weight Decay: {WEIGHT_DECAY}")
    print()
    
    # Load data
    print("Loading data...")
    train_dataset = RustCodeDataset(f"{DATA_DIR}/train.json")
    val_dataset = RustCodeDataset(f"{DATA_DIR}/val.json")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # MPS doesn't support multi-process data loading well
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    print(f"  Train: {len(train_dataset)} sequences")
    print(f"  Val: {len(val_dataset)} sequences")
    
    # Model
    print("\nInitializing model...")
    model = TRM(
        vocab_size=VOCAB_SIZE,
        dim=HIDDEN_SIZE,
        seq_len=SEQ_LEN - 1,  # -1 because we shift for next-token prediction
        n_recursions=N_RECURSIONS,
        t_cycles=T_CYCLES
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,} ({n_params/1e6:.2f}M)")
    
    # Optimizer with EMA (paper's key for stability)
    optimizer = AdamW(
        model.parameters(),
        lr=LR,
        betas=(0.9, 0.95),
        weight_decay=WEIGHT_DECAY
    )
    
    # Scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=MAX_STEPS, eta_min=LR * 0.1)
    
    # Training loop
    print("\nStarting training...")
    model.train()
    step = 0
    epoch = 0
    
    log_file = open(f"{CHECKPOINT_DIR}/training_log.txt", "w")
    log_file.write(f"Training started at {datetime.now()}\n")
    log_file.write(f"Parameters: {n_params:,}\n\n")
    log_file.write("Step,Loss,Acc,PPL,SupervisionSteps,HaltProb,ValLoss,ValAcc,ValPPL\n")
    
    while step < MAX_STEPS:
        epoch += 1
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch in pbar:
            metrics = train_step(model, batch, optimizer, step)
            scheduler.step()
            
            # Logging
            if step % LOG_INTERVAL == 0:
                pbar.set_postfix({
                    'loss': f"{metrics['loss']:.3f}",
                    'acc': f"{metrics['accuracy']:.3f}",
                    'ppl': f"{metrics['perplexity']:.2f}",
                    'steps': metrics['supervision_steps'],
                })
            
            # Evaluation
            val_metrics = {}
            if step % EVAL_INTERVAL == 0 and step > 0:
                print(f"\n[Step {step}] Evaluating...")
                val_metrics = evaluate(model, val_loader)
                print(f"  Val Loss: {val_metrics['loss']:.4f}")
                print(f"  Val Acc: {val_metrics['accuracy']:.4f}")
                print(f"  Val PPL: {val_metrics['perplexity']:.2f}")
            
            # Save checkpoint
            if step % SAVE_INTERVAL == 0 and step > 0:
                checkpoint_path = f"{CHECKPOINT_DIR}/checkpoint_step_{step}.pt"
                torch.save({
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_metrics': metrics,
                    'val_metrics': val_metrics,
                }, checkpoint_path)
                print(f"  Saved checkpoint: {checkpoint_path}")
            
            # Log to file
            log_file.write(f"{step},{metrics['loss']:.4f},{metrics['accuracy']:.4f},"
                          f"{metrics['perplexity']:.2f},{metrics['supervision_steps']},"
                          f"{metrics['final_halt_prob']:.3f},"
                          f"{val_metrics.get('loss', '')},"
                          f"{val_metrics.get('accuracy', '')},"
                          f"{val_metrics.get('perplexity', '')}\n")
            log_file.flush()
            
            step += 1
            if step >= MAX_STEPS:
                break
    
    # Final checkpoint
    final_path = f"{CHECKPOINT_DIR}/final_model.pt"
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'config': {
            'vocab_size': VOCAB_SIZE,
            'hidden_size': HIDDEN_SIZE,
            'seq_len': SEQ_LEN,
            'n_recursions': N_RECURSIONS,
            't_cycles': T_CYCLES,
        }
    }, final_path)
    
    log_file.close()
    print(f"\nTraining complete! Final model saved to {final_path}")

if __name__ == "__main__":
    main()

TRAIN_SCRIPT

echo "  Training script created"

# Step 5: Start training
echo ""
echo "Step 5: Starting TRM training..."
echo "  Checkpoints will be saved to: $CHECKPOINT_DIR"
echo "  Logs will be written to: $CHECKPOINT_DIR/training_log.txt"
echo ""

# Export environment variables
export DATA_DIR
export CHECKPOINT_DIR
export VOCAB_SIZE
export HIDDEN_SIZE
export SEQ_LEN
export BATCH_SIZE
export N_RECURSIONS
export T_CYCLES
export MAX_SUPERVISION
export LEARNING_RATE
export WEIGHT_DECAY
export MAX_STEPS
export EVAL_INTERVAL
export SAVE_INTERVAL
export LOG_INTERVAL

# Run training
python3 "$PROJECT_DIR/train_trm.py"

echo ""
echo "==================================================================="
echo "Training complete!"
echo "==================================================================="
echo ""
echo "Checkpoints saved in: $CHECKPOINT_DIR"
echo "Training log: $CHECKPOINT_DIR/training_log.txt"
echo ""
echo "To monitor training progress:"
echo "  tail -f $CHECKPOINT_DIR/training_log.txt"
echo ""
echo "To resume training, load the latest checkpoint in train_trm.py"
echo ""
