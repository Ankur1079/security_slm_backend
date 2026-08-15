import os
import time
import requests
from typing import List, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from schema import SecurityEvent, LABELS
from inference import SecurityEventClassifier

CHECKPOINT_URL    = os.environ.get("SLM_CHECKPOINT_URL", "")
TOKENIZER_URL     = os.environ.get("SLM_TOKENIZER_URL",  "")
CHECKPOINT_PATH   = "/tmp/model.pt"
TOKENIZER_PATH    = "/tmp/tokenizer.json"
VIRUSTOTAL_API_KEY = os.environ.get("VIRUSTOTAL_API_KEY", "")

def download_file(url, path):
    if os.path.exists(path):
        size = os.path.getsize(path)
        print(f"Already exists: {path} ({size} bytes)")
        if size > 1000:
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
    allow_origins=["*"],
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
        "status":          "ok",
        "model_loaded":    _classifier is not None,
        "checkpoint":      os.path.exists(CHECKPOINT_PATH),
        "tokenizer":       os.path.exists(TOKENIZER_PATH),
        "checkpoint_size": os.path.getsize(CHECKPOINT_PATH) if os.path.exists(CHECKPOINT_PATH) else 0,
        "vt_configured":   bool(VIRUSTOTAL_API_KEY),
    }

@app.post("/classify", response_model=ClassificationResponse)
def classify(req: EventRequest):
    if _classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
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

@app.post("/scan-file")
async def scan_file(file: UploadFile = File(...)):
    """VirusTotal scan proxy — browser CORS bypass"""
    if not VIRUSTOTAL_API_KEY:
        return {"status": "skipped", "malicious": 0, "suspicious": 0,
                "message": "VT key not configured"}
    try:
        # Step 1: VT pe upload karo
        content = await file.read()
        upload_res = requests.post(
            "https://www.virustotal.com/api/v3/files",
            headers={"x-apikey": VIRUSTOTAL_API_KEY},
            files={"file": (file.filename, content, file.content_type)},
            timeout=60,
        )
        upload_res.raise_for_status()
        analysis_id = upload_res.json()["data"]["id"]

        # Step 2: Result aane tak wait karo (max 30s)
        for _ in range(10):
            time.sleep(3)
            report_res = requests.get(
                f"https://www.virustotal.com/api/v3/analyses/{analysis_id}",
                headers={"x-apikey": VIRUSTOTAL_API_KEY},
                timeout=30,
            )
            report = report_res.json()
            status = report["data"]["attributes"]["status"]

            if status == "completed":
                stats = report["data"]["attributes"]["stats"]
                return {
                    "status":     "completed",
                    "malicious":  stats.get("malicious",  0),
                    "suspicious": stats.get("suspicious", 0),
                    "harmless":   stats.get("harmless",   0),
                    "undetected": stats.get("undetected", 0),
                    "total":      sum(stats.values()),
                    "permalink":  f"https://www.virustotal.com/gui/file-analysis/{analysis_id}",
                }

        return {"status": "timeout", "malicious": 0, "suspicious": 0}

    except Exception as e:
        print(f"VT scan error: {e}")
        return {"status": "error", "malicious": 0, "suspicious": 0,
                "message": str(e)}