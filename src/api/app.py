from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
import joblib
import datetime

app = FastAPI(title="Invoice Email Classifier API", version="1.0")

project_root = Path(__file__).resolve().parents[2]
model_path = project_root / "models" / "invoice_email_classifier.joblib"
log_path = project_root / "logs" / "api_requests.log"

model = joblib.load(model_path)


class EmailRequest(BaseModel):
    text: str


@app.get("/health")
def health():
    return {"status": "ok", "message": "Invoice Email Classifier API running"}


@app.post("/predict")
def predict(request: EmailRequest):
    pred = model.predict([request.text])[0]

    timestamp = datetime.datetime.now().isoformat()
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"{timestamp} | INPUT={request.text} | PRED={pred}\n")

    return {
        "category": pred,
        "input_text": request.text
    }
