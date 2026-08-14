"""
Trains a small byte-level BPE tokenizer on the security-event corpus.
A domain-specific tokenizer keeps sequences short and avoids wasting
capacity on general English vocabulary the model will never need.

Usage:
    python train_tokenizer.py --data ../artifacts/train.jsonl ../artifacts/val.jsonl \
        --vocab_size 4096 --out ../artifacts/tokenizer.json
"""

import argparse
import json
import os
import sys

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from schema import SecurityEvent, SPECIAL_TOKENS, serialize_training_example  # noqa: E402


def jsonl_to_corpus(paths, corpus_path):
    """Flatten JSONL examples into raw text lines the tokenizer trainer can read."""
    with open(corpus_path, "w") as out:
        for path in paths:
            with open(path) as f:
                for line in f:
                    ex = json.loads(line)
                    event = SecurityEvent(
                        actor=ex["actor"], action=ex["action"], resource=ex["resource"],
                        device_trust=ex["device_trust"], location=ex["location"],
                        time=ex["time"], prior_events=ex["prior_events"],
                    )
                    seg = serialize_training_example(event, ex["label"], ex["explanation"])
                    out.write(seg["input"] + seg["target"] + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", nargs="+", required=True, help="JSONL files to train on")
    parser.add_argument("--vocab_size", type=int, default=4096)
    parser.add_argument("--out", type=str, default="../artifacts/tokenizer.json")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    corpus_path = os.path.join(os.path.dirname(args.out), "_corpus.txt")
    jsonl_to_corpus(args.data, corpus_path)

    tokenizer = Tokenizer(models.BPE(unk_token=None))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=2,
    )
    tokenizer.train([corpus_path], trainer)
    tokenizer.save(args.out)
    print(f"Trained tokenizer (vocab_size={tokenizer.get_vocab_size()}) -> {args.out}")

    os.remove(corpus_path)


if __name__ == "__main__":
    main()
