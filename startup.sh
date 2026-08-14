#!/bin/bash
mkdir -p checkpoints artifacts

echo "Downloading model from Hugging Face..."
curl -L "https://huggingface.co/Ankur6394/security-slm/resolve/main/model.pt" \
     -o ./checkpoints/model.pt

echo "Downloading tokenizer..."
curl -L "https://huggingface.co/Ankur6394/security-slm/resolve/main/tokenizer.json" \
     -o ./artifacts/tokenizer.json

echo "Starting server..."
uvicorn api:app --host 0.0.0.0 --port $PORT