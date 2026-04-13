"""
ANLI R2 NLI Classifier — FastAPI Inference Server

Serves predictions for premise-hypothesis pairs using a fine-tuned
DeBERTa-v3-base model trained on ANLI Round 2.

Usage:
    uvicorn src.predict:app --host 0.0.0.0 --port 8000

Example request:
    curl -X POST http://localhost:8000/predict \
      -H "Content-Type: application/json" \
      -d '{"premise": "The cat sat on the mat.", "hypothesis": "An animal was on a surface."}'
"""

import os
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- Configuration ---
MODEL_PATH = os.getenv("MODEL_PATH", "/Users/kj_lomran/Desktop/ANLI-R2-Classification/models/best_checkpoint")
MAX_LENGTH = 256
LABEL_NAMES = ["entailment", "neutral", "contradiction"]

# --- App Setup ---
app = FastAPI(
    title="ANLI R2 NLI Classifier",
    description="3-way Natural Language Inference classification using DeBERTa-v3-base fine-tuned on ANLI Round 2",
    version="1.0.0"
)

# --- Load Model ---
print(f"Loading model from {MODEL_PATH}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model.to(DEVICE)
print(f"Model loaded on {DEVICE}.")


# --- Request / Response Schemas ---
class NLIRequest(BaseModel):
    premise: str = Field(..., min_length=1, description="The premise/context text")
    hypothesis: str = Field(..., min_length=1, description="The hypothesis to evaluate")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "premise": "Idris Sultan (born January 1993) is a Tanzanian Actor and comedian.",
                    "hypothesis": "Idris Sultan was born in the first month of the year preceding 1994."
                }
            ]
        }
    }


class NLIResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: dict


# --- Endpoints ---
@app.get("/")
def root():
    return {"status": "ok", "model": "DeBERTa-v3-base-anli-r2"}


@app.get("/health")
def health():
    return {"status": "healthy", "device": DEVICE}


@app.post("/predict", response_model=NLIResponse)
def predict(request: NLIRequest):
    try:
        inputs = tokenizer(
            request.premise,
            request.hypothesis,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        ).to(DEVICE)

        with torch.no_grad():
            outputs = model(**inputs)

        probs = torch.softmax(outputs.logits, dim=-1).squeeze().cpu().tolist()
        pred_idx = outputs.logits.argmax(dim=-1).item()

        return NLIResponse(
            prediction=LABEL_NAMES[pred_idx],
            confidence=round(probs[pred_idx], 4),
            probabilities={
                name: round(prob, 4) for name, prob in zip(LABEL_NAMES, probs)
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch")
def predict_batch(requests: list[NLIRequest]):
    """Predict on multiple premise-hypothesis pairs at once."""
    results = []
    for req in requests:
        results.append(predict(req))
    return results
