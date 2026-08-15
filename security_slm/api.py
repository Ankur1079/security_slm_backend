import os
import requests
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from schema import SecurityEvent, LABELS
from inference import SecurityEventClassifier

CHECKPOINT_URL  = os.environ.get("SLM_CHECKPOINT_URL", "")
TOKENIZER_URL   = os.environ.get("SLM_TOKENIZER_URL",  "")
CHECKPOINT_PATH = "/tmp/model.pt"
TOKENIZER_PATH  = "/tmp/tokenizer.json"

def download_file(url, path):
    if os.path.exists(path):
        size = os.path.getsize(path)
        print(f"Already exists: {path} ({size} bytes)")
        if size > 1000:  # valid file
            return
    print(f"Downloading: {url}")
    headers  = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, timeout=300, stream=True)
    response.raise_for_status()
    with open(path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Done: {path} ({os.path.getsize(path)} bytes)")

app = FastAPI(title="Zero-Trust Security Event Classifier", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Vercel deploy ke baad specific URL daalna
    allow_methods=["*"],
    allow_headers=["*"],
)

_classifier: Optional[SecurityEventClassifier] = None

class EventRequest(BaseModel):
    actor:        str
    action:       str
    resource:     str
    device_trust: str = Field(pattern="^(managed|unmanaged|unknown)$")
    location:     str
    time:         str
    prior_events: List[str] = []

class ClassificationResponse(BaseModel):
    classification: Optional[str]
    explanation:    Optional[str]
    known_label:    bool

@app.on_event("startup")
def load_model():
    global _classifier
    try:
        if not CHECKPOINT_URL or not TOKENIZER_URL:
            print("❌ SLM_CHECKPOINT_URL or SLM_TOKENIZER_URL not set")
            return
        download_file(CHECKPOINT_URL, CHECKPOINT_PATH)
        download_file(TOKENIZER_URL,  TOKENIZER_PATH)
        _classifier = SecurityEventClassifier(CHECKPOINT_PATH, TOKENIZER_PATH)
        print("✅ Model loaded successfully")
    except Exception as e:
        import traceback
        print(f"❌ Model load failed: {e}")
        traceback.print_exc()
        _classifier = None

@app.get("/health")
def health():
    return {
        "status":       "ok",
        "model_loaded": _classifier is not None,
        "checkpoint":   os.path.exists(CHECKPOINT_PATH),
        "tokenizer":    os.path.exists(TOKENIZER_PATH),
        "checkpoint_size": os.path.getsize(CHECKPOINT_PATH) if os.path.exists(CHECKPOINT_PATH) else 0,
    }

@app.post("/classify", response_model=ClassificationResponse)
def classify(req: EventRequest):
    if _classifier is None:
        raise HTTPException(status_code=503,
            detail="Model not loaded.")
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
    