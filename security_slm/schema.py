"""
Shared schema + serialization for the Zero-Trust Security Event SLM.

An "event" is a dict describing a file-locker security event. We serialize
it into a flat text format the model reads, and we serialize the
(classification, explanation) pair as the target the model must generate.

Format (single sequence, fed to a causal LM):

<bos> EVENT: actor=... action=... resource=... device_trust=... location=...
time=... prior_events=... <sep> CLASSIFICATION: <label> EXPLANATION: <text> <eos>

At training time we mask the loss on everything up to and including <sep>,
so the model only learns to predict the label + explanation.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<sep>"]

LABELS = [
    "benign_unusual",
    "exfiltration_risk",
    "privilege_escalation",
    "credential_stuffing",
    "insider_threat",
    "anomalous_access",
    "malware_indicator",
    "policy_violation",
]

LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}


@dataclass
class SecurityEvent:
    actor: str
    action: str
    resource: str
    device_trust: str          # managed | unmanaged | unknown
    location: str              # known_ip | unrecognized_ip | tor_exit_node | vpn_corporate
    time: str                  # e.g. "03:12 UTC"
    prior_events: List[str] = field(default_factory=list)

    def to_event_str(self) -> str:
        prior = ",".join(self.prior_events) if self.prior_events else "none"
        return (
            f"EVENT: actor={self.actor} action={self.action} "
            f"resource={self.resource} device_trust={self.device_trust} "
            f"location={self.location} time={self.time} prior_events={prior}"
        )


def serialize_training_example(event: SecurityEvent, label: str, explanation: str) -> Dict[str, str]:
    """Returns the input segment and target segment separately so the
    training script can mask loss correctly."""
    input_segment = f"<bos> {event.to_event_str()} <sep> "
    target_segment = f"CLASSIFICATION: {label} EXPLANATION: {explanation} <eos>"
    return {"input": input_segment, "target": target_segment}


def parse_model_output(text: str) -> Dict[str, Any]:
    """Parse generated text of the form
    'CLASSIFICATION: <label> EXPLANATION: <text> <eos>' back into fields.
    Robust to minor formatting drift from an undertrained model."""
    result = {"classification": None, "explanation": None, "raw": text}
    try:
        text = text.split("<eos>")[0].strip()
        if "EXPLANATION:" in text:
            cls_part, expl_part = text.split("EXPLANATION:", 1)
        else:
            cls_part, expl_part = text, ""
        cls_part = cls_part.replace("CLASSIFICATION:", "").strip()
        # take first whitespace-delimited token as the label
        label_token = cls_part.split()[0] if cls_part.split() else None
        result["classification"] = label_token if label_token in LABEL2ID else cls_part.strip()
        result["explanation"] = expl_part.strip()
    except Exception:
        pass
    return result
