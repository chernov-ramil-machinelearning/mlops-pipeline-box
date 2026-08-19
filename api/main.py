"""
FastAPI-слой: /predict + логирование каждого предсказания в БД.

Сервис инференса ML-модели с поддержкой версионирования (MODEL_VERSION)
и асинхронного/синхронного логирования в PostgreSQL.
"""
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import pandas as pd
import psycopg2
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mlops_api")

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://mlops:mlops@db:5432/mlops_box")
MODEL_VERSION = os.environ.get("MODEL_VERSION", "v1")

app = FastAPI(
    title="MLOps Pipeline in a Box API",
    description="Production API for customer churn prediction and experiment inference tracking.",
    version="1.0.0",
)


class PredictRequest(BaseModel):
    CreditScore: int = Field(default=619, description="Credit score of the customer", ge=300, le=850)
    Geography: str = Field(default="France", description="Customer country (e.g. France, Spain, Germany)")
    Gender: str = Field(default="Female", description="Gender (Female / Male)")
    Age: int = Field(default=42, description="Customer age in years", ge=18, le=100)
    Tenure: int = Field(default=2, description="Years customer has been with the bank", ge=0, le=20)
    Balance: float = Field(default=0.0, description="Account balance", ge=0.0)
    NumOfProducts: int = Field(default=1, description="Number of bank products used", ge=1, le=10)
    HasCrCard: int = Field(default=1, description="Credit card flag (1=yes, 0=no)", ge=0, le=1)
    IsActiveMember: int = Field(default=1, description="Active member flag (1=yes, 0=no)", ge=0, le=1)
    EstimatedSalary: float = Field(default=101348.88, description="Estimated annual salary", ge=0.0)
    avg_balance_for_tenure: float = Field(default=76485.88, description="Engineered feature: average balance for tenure")


class PredictResponse(BaseModel):
    churn_prediction: int
    churn_probability: float
    model_version: str


_cached_model = None


def get_model_path() -> Path:
    candidates = [
        Path(f"/app/models/{MODEL_VERSION}.cbm"),
        Path(f"models/{MODEL_VERSION}.cbm"),
        Path(__file__).resolve().parent.parent / "models" / f"{MODEL_VERSION}.cbm",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Model file for version '{MODEL_VERSION}' not found among candidates: {candidates}")


def get_model():
    global _cached_model
    if _cached_model is None:
        model_path = get_model_path()
        logger.info(f"Loading model '{MODEL_VERSION}' from {model_path}...")
        _cached_model = joblib.load(model_path)
    return _cached_model


def log_prediction(payload: Dict[str, Any], prediction: float) -> None:
    try:
        conn = psycopg2.connect(DATABASE_URL)
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO predictions (model_version, input_payload, prediction) VALUES (%s, %s, %s)",
                (MODEL_VERSION, json.dumps(payload), prediction),
            )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.warning(f"Could not log prediction to database: {e}")


@app.get("/health")
def health():
    model_loaded = False
    try:
        _ = get_model()
        model_loaded = True
    except Exception:
        pass
    return {
        "status": "ok",
        "model_version": MODEL_VERSION,
        "model_loaded": model_loaded,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        model = get_model()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")

    input_dict = req.model_dump()
    input_df = pd.DataFrame([input_dict])
    if hasattr(model, "feature_names_") and model.feature_names_:
        # Ensure column order strictly matches model's internal feature order
        input_df = input_df[[col for col in model.feature_names_ if col in input_df.columns]]

    try:
        prediction = int(model.predict(input_df)[0])
        probability = float(model.predict_proba(input_df)[0][1])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference error: {e}")


    log_prediction(input_dict, probability)

    return PredictResponse(
        churn_prediction=prediction,
        churn_probability=round(probability, 4),
        model_version=MODEL_VERSION,
    )

