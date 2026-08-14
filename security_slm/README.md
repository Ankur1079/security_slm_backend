# Security Event SLM — Zero-Trust File Locker

A small decoder-only transformer (GPT-style, PyTorch, built from scratch — no
pretrained weights) that reads a structured security event and outputs both
a **classification** and a natural-language **explanation** in one forward
pass. Designed to slot into a FastAPI backend the same way your other
services (resume pipeline, CI/CD dashboard) are structured.

## How it works

Each event is serialized into text:

```
<bos> EVENT: actor=user:ankur action=bulk_download resource=/finance/q3_report.xlsx
device_trust=unmanaged location=unrecognized_ip time=03:12 UTC
prior_events=failed_mfa x2,new_device_registered <sep>
```

The model autoregressively generates:

```
CLASSIFICATION: exfiltration_risk EXPLANATION: bulk download from an
unmanaged device on an unrecognized IP shortly after failed MFA attempts
matches a credential-compromise exfiltration pattern <eos>
```

Loss is masked on everything up to and including `<sep>`, so the model only
learns to produce the label + explanation, not to reconstruct the input.

Labels (extend in `schema.py`): `benign_unusual`, `exfiltration_risk`,
`privilege_escalation`, `credential_stuffing`, `insider_threat`,
`anomalous_access`, `malware_indicator`, `policy_violation`.

## Project layout

```
security_slm/
├── schema.py                  # Event dataclass, serialization, output parsing
├── dataset.py                 # PyTorch Dataset with loss masking
├── train.py                   # Training loop (AdamW, cosine LR, checkpointing)
├── inference.py                # Load checkpoint -> classify_event()
├── api.py                     # FastAPI service (/classify, /health)
├── requirements.txt
├── data/
│   └── generate_synthetic_data.py   # Template-based synthetic dataset generator
├── tokenizer/
│   └── train_tokenizer.py     # Trains a small domain-specific BPE tokenizer
├── model/
│   └── architecture.py        # SecuritySLM: causal self-attention transformer
├── artifacts/                 # train.jsonl, val.jsonl, tokenizer.json (generated)
└── checkpoints/               # model.pt (generated)
```

## Running the full pipeline

All commands below were run and verified during scaffolding (small-scale
smoke test — 2k examples, 1M-param model, 2 epochs on CPU). The mechanics
(data gen → tokenizer → training loop → loss masking → checkpointing →
inference → parsing) all work correctly end to end. **You'll need a real
training run to get a usable model** — see "Scaling up" below.

### 1. Generate synthetic training data

```bash
cd data
python generate_synthetic_data.py --n 8000 --out ../artifacts/train.jsonl --val_split 0.1
```

This uses field-consistent templates (e.g. `exfiltration_risk` events get
bulk actions + untrusted device/network + suspicious prior events) so the
model learns real signal, not noise. Swap in real logs or LLM-generated
examples later by producing the same JSONL schema:
`{actor, action, resource, device_trust, location, time, prior_events, label, explanation}`.

### 2. Train the tokenizer

```bash
cd ../tokenizer
python train_tokenizer.py --data ../artifacts/train.jsonl ../artifacts/val.jsonl \
    --vocab_size 4096 --out ../artifacts/tokenizer.json
```

### 3. Train the model

```bash
cd ..
python train.py \
    --train ./artifacts/train.jsonl --val ./artifacts/val.jsonl \
    --tokenizer ./artifacts/tokenizer.json --out ./checkpoints/model.pt \
    --block_size 256 --d_model 384 --n_layers 8 --n_heads 6 \
    --batch_size 32 --epochs 15 --eval_every 500
```

Every `eval_every` steps it greedily generates on held-out examples and
computes **classification accuracy** (not just token loss — loss going down
doesn't guarantee the label is right, so this is the metric to actually
watch). Best checkpoint by val accuracy is saved automatically.

On a single consumer GPU (e.g. RTX 3060+), 8000 examples at these settings
should train in well under an hour.

### 4. Run inference

```bash
python inference.py
```

Or import directly:

```python
from inference import SecurityEventClassifier
from schema import SecurityEvent

clf = SecurityEventClassifier("./checkpoints/model.pt", "./artifacts/tokenizer.json")
result = clf.classify_event(SecurityEvent(
    actor="user:ankur", action="bulk_download", resource="/finance/q3_report.xlsx",
    device_trust="unmanaged", location="unrecognized_ip", time="03:12 UTC",
    prior_events=["failed_mfa x2", "new_device_registered"],
))
print(result)  # {"classification": ..., "explanation": ..., "raw": ...}
```

### 5. Serve via FastAPI

```bash
export SLM_CHECKPOINT=./checkpoints/model.pt
export SLM_TOKENIZER=./artifacts/tokenizer.json
uvicorn api:app --host 0.0.0.0 --port 8000
```

```bash
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"actor":"user:ankur","action":"bulk_download","resource":"/finance/q3_report.xlsx",
       "device_trust":"unmanaged","location":"unrecognized_ip","time":"03:12 UTC",
       "prior_events":["failed_mfa x2"]}'
```

## Scaling up for real use

- **Data**: 8000 synthetic examples is a starting point. For production
  reliability, mix in real (anonymized) events from your locker's audit
  log, and/or generate a larger, more varied synthetic set (more resource
  types, more nuanced borderline cases between adjacent labels like
  `anomalous_access` vs `exfiltration_risk`).
- **Model size**: default config (~15-20M params at `d_model=384,
  n_layers=8`) is a good starting point. If accuracy plateaus, try
  `d_model=512, n_layers=10` before adding more data.
- **Explanation faithfulness**: the template explanations are grounded in
  actual event fields by construction. As you add real data, keep enforcing
  that explanations reference concrete fields (device, location, prior
  events) rather than generic language — this is what makes the output
  trustworthy for a security analyst.
- **Deployment**: quantize with `torch.quantization` or export to ONNX for
  lower-latency scoring if this needs to run inline on every file access
  rather than async.
- **Calibration**: for a security tool, add a confidence threshold — if
  top-token probability for the classification is low, route to
  `anomalous_access` (a human-review bucket) rather than guessing.
