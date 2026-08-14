"""
PyTorch Dataset for the security-event SLM. Tokenizes each (event, label,
explanation) triple into a single sequence and produces a loss mask so the
model is only trained to predict the CLASSIFICATION/EXPLANATION target,
not to reconstruct the input event.
"""

import json
import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer

from schema import SecurityEvent, serialize_training_example

IGNORE_INDEX = -100


class SecurityEventDataset(Dataset):
    def __init__(self, jsonl_path: str, tokenizer_path: str, block_size: int = 512):
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.block_size = block_size
        self.pad_id = self.tokenizer.token_to_id("<pad>")
        self.eos_id = self.tokenizer.token_to_id("<eos>")

        self.examples = []
        with open(jsonl_path) as f:
            for line in f:
                self.examples.append(json.loads(line))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        ex = self.examples[i]
        event = SecurityEvent(
            actor=ex["actor"], action=ex["action"], resource=ex["resource"],
            device_trust=ex["device_trust"], location=ex["location"],
            time=ex["time"], prior_events=ex["prior_events"],
        )
        seg = serialize_training_example(event, ex["label"], ex["explanation"])

        input_ids_in = self.tokenizer.encode(seg["input"]).ids
        input_ids_target = self.tokenizer.encode(seg["target"]).ids

        full = input_ids_in + input_ids_target
        # labels: -100 (ignored) for the input segment, real ids for target segment
        labels = [IGNORE_INDEX] * len(input_ids_in) + input_ids_target[:]

        # truncate/pad to block_size (+1 because we shift by one for next-token prediction)
        full = full[: self.block_size + 1]
        labels = labels[: self.block_size + 1]

        pad_len = (self.block_size + 1) - len(full)
        if pad_len > 0:
            full = full + [self.pad_id] * pad_len
            labels = labels + [IGNORE_INDEX] * pad_len

        x = torch.tensor(full[:-1], dtype=torch.long)
        y_labels = torch.tensor(labels[1:], dtype=torch.long)  # shifted targets
        return x, y_labels
