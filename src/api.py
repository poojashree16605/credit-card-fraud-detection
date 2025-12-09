from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()
@app.get("/")
def health():
    return {"status": "ok", "message": "API is running"}
model = joblib.load("src/artifacts/fraud_model.pkl")

class BatchData(BaseModel):
    data: list

@app.post("/predict_batch")
def predict_batch(batch: BatchData):
    df = pd.DataFrame(batch.data)

    # FIX: Remove "Class" column if present
    if "Class" in df.columns:
        df = df.drop(columns=["Class"])

    preds = model.predict(df)
    probas = model.predict_proba(df)[:, 1]

    return {
        "predictions": preds.tolist(),
        "probabilities": probas.tolist()
    }


