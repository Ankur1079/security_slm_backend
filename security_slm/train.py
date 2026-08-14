"""
Trains the SecuritySLM on the synthetic security-event dataset.

Usage:
    python train.py \
        --train ./artifacts/train.jsonl --val ./artifacts/val.jsonl \
        --tokenizer ./artifacts/tokenizer.json \
        --out ./checkpoints/model.pt --epochs 8 --batch_size 32
"""

import argparse
import math
import os
import torch
from torch.utils.data import DataLoader
from tokenizers import Tokenizer

from model.architecture import SecuritySLM
from dataset import SecurityEventDataset, IGNORE_INDEX
from schema import parse_model_output, LABEL2ID


def get_lr(step, total_steps, warmup_steps, base_lr, min_lr):
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    coeff = 0.5 * (1 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * coeff


@torch.no_grad()
def evaluate_classification_accuracy(model, dataset, tokenizer, device, n_samples=200):
    """Greedily generates on a subset of val examples and checks whether the
    predicted CLASSIFICATION label matches ground truth. This is the metric
    that actually matters for this task, unlike raw token loss."""
    model.eval()
    correct = 0
    total = min(n_samples, len(dataset.examples))
    eos_id = tokenizer.token_to_id("<eos>")
    sep_id = tokenizer.token_to_id("<sep>")

    for i in range(total):
        ex = dataset.examples[i]
        from schema import SecurityEvent, serialize_training_example
        event = SecurityEvent(
            actor=ex["actor"], action=ex["action"], resource=ex["resource"],
            device_trust=ex["device_trust"], location=ex["location"],
            time=ex["time"], prior_events=ex["prior_events"],
        )
        seg = serialize_training_example(event, ex["label"], ex["explanation"])
        input_ids = tokenizer.encode(seg["input"]).ids
        idx = torch.tensor([input_ids], dtype=torch.long, device=device)

        out = model.generate(idx, max_new_tokens=60, temperature=0.3, top_k=10, eos_token_id=eos_id)
        gen_ids = out[0, len(input_ids):].tolist()
        gen_text = tokenizer.decode(gen_ids)
        parsed = parse_model_output(gen_text)
        if parsed["classification"] == ex["label"]:
            correct += 1

    model.train()
    return correct / total if total else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--out", default="./checkpoints/model.pt")
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=384)
    parser.add_argument("--n_layers", type=int, default=8)
    parser.add_argument("--n_heads", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=3e-5)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    tokenizer = Tokenizer.from_file(args.tokenizer)
    vocab_size = tokenizer.get_vocab_size()
    pad_id = tokenizer.token_to_id("<pad>")

    train_ds = SecurityEventDataset(args.train, args.tokenizer, block_size=args.block_size)
    val_ds = SecurityEventDataset(args.val, args.tokenizer, block_size=args.block_size)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    model = SecuritySLM(
        vocab_size=vocab_size, d_model=args.d_model, n_layers=args.n_layers,
        n_heads=args.n_heads, block_size=args.block_size, pad_token_id=pad_id,
    ).to(args.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1, betas=(0.9, 0.95))
    total_steps = len(train_loader) * args.epochs
    use_amp = args.device == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    step = 0
    best_acc = 0.0
    for epoch in range(args.epochs):
        for x, y in train_loader:
            x, y = x.to(args.device), y.to(args.device)
            lr = get_lr(step, total_steps, args.warmup_steps, args.lr, args.min_lr)
            for g in optimizer.param_groups:
                g["lr"] = lr

            with torch.autocast(device_type="cuda" if use_amp else "cpu",
                                 dtype=torch.bfloat16, enabled=use_amp):
                _, loss = model(x, targets=y, ignore_index=IGNORE_INDEX)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            if step % 50 == 0:
                print(f"epoch {epoch} step {step}/{total_steps} loss {loss.item():.4f} lr {lr:.2e}")

            if step > 0 and step % args.eval_every == 0:
                acc = evaluate_classification_accuracy(model, val_ds, tokenizer, args.device)
                print(f"[eval] step {step} classification_accuracy={acc:.3f}")
                if acc > best_acc:
                    best_acc = acc
                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "config": {
                            "vocab_size": vocab_size, "d_model": args.d_model,
                            "n_layers": args.n_layers, "n_heads": args.n_heads,
                            "block_size": args.block_size, "pad_token_id": pad_id,
                        },
                        "step": step, "val_accuracy": acc,
                    }, args.out)
                    print(f"saved new best checkpoint (acc={acc:.3f}) -> {args.out}")

            step += 1

    # final save regardless of whether it beat best_acc, so you always have a usable checkpoint
    final_acc = evaluate_classification_accuracy(model, val_ds, tokenizer, args.device)
    print(f"[final] classification_accuracy={final_acc:.3f}")
    if final_acc >= best_acc:
        torch.save({
            "model_state_dict": model.state_dict(),
            "config": {
                "vocab_size": vocab_size, "d_model": args.d_model,
                "n_layers": args.n_layers, "n_heads": args.n_heads,
                "block_size": args.block_size, "pad_token_id": pad_id,
            },
            "step": step, "val_accuracy": final_acc,
        }, args.out)
        print(f"saved final checkpoint -> {args.out}")


if __name__ == "__main__":
    main()
