#!/usr/bin/env python3
"""Tiny character-level language model using TernaryLinear.

Trains on a small text corpus (TinyShakespeare or inline sample) and compares
ternary vs FP32 perplexity.

Usage:
    uv run python examples/tiny_char_lm_ternary.py [options]

Options:
    --epochs INT       Number of epochs (default: 5)
    --hidden INT       Hidden dimension (default: 256)
    --context INT      Context length (default: 64)
    --batch-size INT   Batch size (default: 32)
    --lr FLOAT         Learning rate (default: 1e-3)
    --compare          Compare ternary vs FP32
    --cuda             Use CUDA
    --download         Download TinyShakespeare (otherwise use inline sample)
"""

import argparse
import math
import time
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Sample text for quick testing (subset of Shakespeare)
SAMPLE_TEXT = """
ROMEO: But, soft! what light through yonder window breaks?
It is the east, and Juliet is the sun.
Arise, fair sun, and kill the envious moon,
Who is already sick and pale with grief,
That thou her maid art far more fair than she:
Be not her maid, since she is envious;
Her vestal livery is but sick and green
And none but fools do wear it; cast it off.
It is my lady, O, it is my love!
O, that she knew she were!
She speaks yet she says nothing: what of that?
Her eye discourses; I will answer it.
I am too bold, 'tis not to me she speaks:
Two of the fairest stars in all the heaven,
Having some business, do entreat her eyes
To twinkle in their spheres till they return.
What if her eyes were there, they in her head?
The brightness of her cheek would shame those stars,
As daylight doth a lamp; her eyes in heaven
Would through the airy region stream so bright
That birds would sing and think it were not night.
See, how she leans her cheek upon her hand!
O, that I were a glove upon that hand,
That I might touch that cheek!

JULIET: Ay me!

ROMEO: She speaks:
O, speak again, bright angel! for thou art
As glorious to this night, being o'er my head
As is a winged messenger of heaven
Unto the white-upturned wondering eyes
Of mortals that fall back to gaze on him
When he bestrides the lazy-pacing clouds
And sails upon the bosom of the air.
"""


class CharDataset(Dataset):
    """Character-level dataset for language modeling."""

    def __init__(self, text: str, context_length: int):
        self.text = text
        self.context_length = context_length

        # Build vocabulary
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}

        # Encode text
        self.data = torch.tensor(
            [self.char_to_idx[ch] for ch in text], dtype=torch.long
        )

    def __len__(self) -> int:
        return len(self.data) - self.context_length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx : idx + self.context_length]
        y = self.data[idx + 1 : idx + self.context_length + 1]
        return x, y


class CharLM(nn.Module):
    """Simple character-level language model (MLP-based)."""

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        context_length: int,
        use_ternary: bool = False,
    ):
        super().__init__()
        self.context_length = context_length
        self.use_ternary = use_ternary

        # Embedding
        self.embedding = nn.Embedding(vocab_size, hidden_dim)

        # Choose linear layer type
        if use_ternary:
            from bittorch.nn import TernaryLinear
            Linear = TernaryLinear
        else:
            Linear = nn.Linear

        # MLP layers
        self.fc1 = Linear(hidden_dim * context_length, hidden_dim * 2)
        self.fc2 = Linear(hidden_dim * 2, hidden_dim)
        self.fc3 = Linear(hidden_dim, vocab_size)

        # Output projection uses standard linear (logits need full precision)
        self.fc3 = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, context_length]
        emb = self.embedding(x)  # [batch, context, hidden]
        emb = emb.view(emb.size(0), -1)  # [batch, context * hidden]

        h = F.relu(self.fc1(emb))
        h = F.relu(self.fc2(h))
        logits = self.fc3(h)  # [batch, vocab_size]

        return logits


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Train for one epoch. Returns (avg_loss, perplexity)."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)

        # Forward
        logits = model(x)  # [batch, vocab_size]

        # Use last position for prediction
        loss = F.cross_entropy(logits, y[:, -1])

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    perplexity = math.exp(avg_loss)
    return avg_loss, perplexity


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate model. Returns (avg_loss, perplexity)."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y[:, -1])
            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    perplexity = math.exp(avg_loss)
    return avg_loss, perplexity


def generate(
    model: nn.Module,
    dataset: CharDataset,
    device: torch.device,
    prompt: str = "ROMEO: ",
    max_new_tokens: int = 100,
) -> str:
    """Generate text from the model."""
    model.eval()

    # Encode prompt
    context = [dataset.char_to_idx.get(ch, 0) for ch in prompt]
    context = context[-dataset.context_length:]  # Truncate if too long

    # Pad if needed
    while len(context) < dataset.context_length:
        context = [0] + context

    context = torch.tensor(context, dtype=torch.long, device=device).unsqueeze(0)

    generated = list(prompt)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(context)
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, 1).item()
            next_char = dataset.idx_to_char[next_idx]
            generated.append(next_char)

            # Update context
            context = torch.cat(
                [context[:, 1:], torch.tensor([[next_idx]], device=device)],
                dim=1,
            )

    return "".join(generated)


def download_tiny_shakespeare() -> str:
    """Download TinyShakespeare dataset."""
    import urllib.request

    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    print("Downloading TinyShakespeare...")
    with urllib.request.urlopen(url) as response:
        text = response.read().decode("utf-8")
    print(f"Downloaded {len(text)} characters")
    return text


def main():
    parser = argparse.ArgumentParser(description="Tiny character-level LM with TernaryLinear")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--context", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--compare", action="store_true", help="Compare ternary vs FP32")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA")
    parser.add_argument("--download", action="store_true", help="Download TinyShakespeare")
    args = parser.parse_args()

    # Device
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Get text
    if args.download:
        text = download_tiny_shakespeare()
    else:
        text = SAMPLE_TEXT * 10  # Repeat sample for more data
        print(f"Using inline sample ({len(text)} chars)")

    # Split train/val
    split_idx = int(len(text) * 0.9)
    train_text = text[:split_idx]
    val_text = text[split_idx:]

    # Create datasets
    train_dataset = CharDataset(train_text, args.context)
    val_dataset = CharDataset(val_text, args.context)

    print(f"Vocab size: {train_dataset.vocab_size}")
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    def train_model(use_ternary: bool) -> Tuple[nn.Module, list]:
        """Train a model and return it with training history."""
        model_name = "TernaryLinear" if use_ternary else "FP32 nn.Linear"
        print(f"\n{'='*60}")
        print(f"Training {model_name}")
        print("=" * 60)

        model = CharLM(
            vocab_size=train_dataset.vocab_size,
            hidden_dim=args.hidden,
            context_length=args.context,
            use_ternary=use_ternary,
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        history = []
        for epoch in range(args.epochs):
            start = time.time()
            train_loss, train_ppl = train_epoch(model, train_loader, optimizer, device)
            val_loss, val_ppl = evaluate(model, val_loader, device)
            elapsed = time.time() - start

            history.append({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_ppl": train_ppl,
                "val_loss": val_loss,
                "val_ppl": val_ppl,
            })

            print(
                f"Epoch {epoch+1}/{args.epochs} | "
                f"Train PPL: {train_ppl:.2f} | Val PPL: {val_ppl:.2f} | "
                f"Time: {elapsed:.2f}s"
            )

        return model, history

    if args.compare:
        # Train both models
        fp32_model, fp32_history = train_model(use_ternary=False)
        ternary_model, ternary_history = train_model(use_ternary=True)

        # Compare final perplexities
        print("\n" + "=" * 60)
        print("COMPARISON RESULTS")
        print("=" * 60)
        print(f"FP32 Final Val PPL:     {fp32_history[-1]['val_ppl']:.2f}")
        print(f"Ternary Final Val PPL:  {ternary_history[-1]['val_ppl']:.2f}")

        ppl_ratio = ternary_history[-1]["val_ppl"] / fp32_history[-1]["val_ppl"]
        print(f"Ratio (Ternary/FP32):   {ppl_ratio:.2f}x")

        # Generate samples
        print("\n" + "-" * 60)
        print("Sample Generation (FP32):")
        print("-" * 60)
        print(generate(fp32_model, train_dataset, device, max_new_tokens=100))

        print("\n" + "-" * 60)
        print("Sample Generation (Ternary):")
        print("-" * 60)
        print(generate(ternary_model, train_dataset, device, max_new_tokens=100))

    else:
        # Train ternary model only
        model, history = train_model(use_ternary=True)

        print("\n" + "-" * 60)
        print("Sample Generation:")
        print("-" * 60)
        print(generate(model, train_dataset, device, max_new_tokens=100))


if __name__ == "__main__":
    main()
