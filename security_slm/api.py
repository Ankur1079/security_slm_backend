"""
FastAPI service for the Security Event SLM. Mirrors the shape of your
existing FastAPI backends (resume pipeline, CI/CD dashboard).

Run:
    uvicorn api:app --host 0.0.0.0 --port 8000

POST /classify
{
  "actor": "user:ankur",
  "action": "bulk_download",
  "resource": "/finance/q3_report.xlsx",
  "device_trust": "unmanaged",
  "location": "unrecognized_ip",
  "time": "03:12 UTC",
  "prior_events": ["failed_mfa x2", "new_device_registered"]
}
"""

import os
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from schema import SecurityEvent, LABELS
from inference import SecurityEventClassifier

CHECKPOINT_PATH = os.environ.get("SLM_CHECKPOINT", "./checkpoints/model.pt")
TOKENIZER_PATH = os.environ.get("SLM_TOKENIZER", "./artifacts/tokenizer.json")

app = FastAPI(title="Zero-Trust Security Event Classifier", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_classifier: Optional[SecurityEventClassifier] = None


class EventRequest(BaseModel):
    actor: str
    action: str
    resource: str
    device_trust: str = Field(pattern="^(managed|unmanaged|unknown)$")
    location: str
    time: str
    prior_events: List[str] = []


class ClassificationResponse(BaseModel):
    classification: Optional[str]
    explanation: Optional[str]
    known_label: bool


@app.on_event("startup")
def load_model():
    global _classifier
    if not os.path.exists(CHECKPOINT_PATH):
        # Don't crash the app if the model hasn't been trained yet — surface
        # a clear error on the endpoint instead.
        print(f"WARNING: checkpoint not found at {CHECKPOINT_PATH}. "
              f"Train the model first (see train.py).")
        return
    _classifier = SecurityEventClassifier(CHECKPOINT_PATH, TOKENIZER_PATH)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _classifier is not None}


@app.post("/classify", response_model=ClassificationResponse)
def classify(req: EventRequest):
    if _classifier is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train a checkpoint and set SLM_CHECKPOINT.",
        )
    event = SecurityEvent(
        actor=req.actor, action=req.action, resource=req.resource,
        device_trust=req.device_trust, location=req.location,
        time=req.time, prior_events=req.prior_events,
    )
    result = _classifier.classify_event(event)
    return ClassificationResponse(
        classification=result["classification"],
        explanation=result["explanation"],
        known_label=result["classification"] in LABELS,
    )
