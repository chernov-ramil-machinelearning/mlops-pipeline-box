"""
Unit-тесты для FastAPI слоя (api/main.py).
Проверяют healthcheck и схему предсказания.
"""
import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from api.main import app

client = TestClient(app)


def test_healthcheck():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model_version" in data


def test_predict_endpoint_success():
    payload = {
        "CreditScore": 650,
        "Geography": "France",
        "Gender": "Female",
        "Age": 38,
        "Tenure": 4,
        "Balance": 50000.0,
        "NumOfProducts": 2,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "EstimatedSalary": 75000.0,
        "avg_balance_for_tenure": 50000.0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "churn_prediction" in data
    assert "churn_probability" in data
    assert data["churn_prediction"] in [0, 1]
    assert 0.0 <= data["churn_probability"] <= 1.0


def test_predict_endpoint_validation_error():
    # Age < 18 or invalid field should trigger 422
    payload = {
        "CreditScore": 650,
        "Geography": "France",
        "Gender": "Female",
        "Age": 10,  # invalid age
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
