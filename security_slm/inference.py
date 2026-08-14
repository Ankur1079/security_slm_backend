"""
Inference wrapper around SecuritySLM. Loads a trained checkpoint + tokenizer
and exposes a single classify_event() function used by both a CLI smoke test
and the FastAPI service.
"""

import torch
from tokenizers import Tokenizer

from model.architecture import SecuritySLM
from schema import SecurityEvent, serialize_training_example, parse_model_output


class SecurityEventClassifier:
    def __init__(self, checkpoint_path: str, tokenizer_path: str, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = Tokenizer.from_file(tokenizer_path)

        ckpt = torch.load(checkpoint_path, map_location=self.device)
        cfg = ckpt["config"]
        self.model = SecuritySLM(**cfg).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        self.eos_id = self.tokenizer.token_to_id("<eos>")
        print(f"Loaded checkpoint (step={ckpt.get('step')}, "
              f"val_accuracy={ckpt.get('val_accuracy'):.3f}) on {self.device}")

    @torch.no_grad()
    def classify_event(self, event: SecurityEvent, max_new_tokens=120,
                        temperature=0.3, top_k=10) -> dict:
        seg = serialize_training_example(event, label="", explanation="")
        input_ids = self.tokenizer.encode(seg["input"]).ids
        idx = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        out = self.model.generate(
            idx, max_new_tokens=max_new_tokens, temperature=temperature,
            top_k=top_k, eos_token_id=self.eos_id,
        )
        gen_ids = out[0, len(input_ids):].tolist()
        gen_text = self.tokenizer.decode(gen_ids)
        return parse_model_output(gen_text)


if __name__ == "__main__":
    # Quick smoke test
    clf = SecurityEventClassifier(
        checkpoint_path="./checkpoints/model.pt",
        tokenizer_path="./artifacts/tokenizer.json",
    )
    event = SecurityEvent(
    actor="user:priya", action="view", resource="/engineering/repo_backup.tar",
    device_trust="managed", location="known_ip", time="14:30 UTC",
    prior_events=[],
    )
    result = clf.classify_event(event)
    print(result)
