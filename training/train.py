"""
Обучение модели на фичах из БД (PostgreSQL) + версионирование и логирование экспериментов.
"""
import argparse
import os
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import pandas as pd
import psycopg as psycopg2
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from experiment_log import log_experiment

ROOT_DIR = Path(__file__).resolve().parent.parent
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://mlops:mlops@localhost:5434/mlops_box")


def load_features(conn) -> pd.DataFrame:
    """Читает features из БД, разворачивает JSONB в структурированные колонки."""
    query = "SELECT customer_id, feature_set FROM features;"
    df = pd.read_sql(query, conn)
    features_flat = pd.json_normalize(df["feature_set"])
    features_flat["customer_id"] = df["customer_id"]
    return features_flat


def train_model(df: pd.DataFrame, hyperparameters: Dict[str, Any]) -> Tuple[CatBoostClassifier, Dict[str, float]]:
    """Обучает CatBoostClassifier и считает метрики ROC-AUC и F1."""
    X = df.drop(columns=["Exited", "customer_id"])
    y = df["Exited"]
    cat_features = ["Geography", "Gender"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = CatBoostClassifier(
        loss_function="Logloss",
        cat_features=cat_features,
        verbose=False,
        **hyperparameters,
    )
    model.fit(X_train, y_train)

    pred_proba = model.predict_proba(X_test)[:, 1]
    pred = model.predict(X_test)

    auc_score = roc_auc_score(y_test, pred_proba)
    f1 = f1_score(y_test, pred)
    metrics_dict = {"roc_auc": float(auc_score), "f1": float(f1)}

    print(f"Trained model with metrics: ROC-AUC={auc_score:.4f}, F1={f1:.4f}")
    return model, metrics_dict


def run_training(model_version: str, hyperparameters: Dict[str, Any]) -> None:
    print(f"Connecting to database: {DATABASE_URL}...")
    conn = psycopg2.connect(DATABASE_URL)
    df = load_features(conn)

    model, metrics = train_model(df, hyperparameters)

    models_dir = ROOT_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"{model_version}.cbm"

    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

    log_experiment(
        conn,
        model_version=model_version,
        hyperparameters=hyperparameters,
        metrics=metrics,
        model_path=str(model_path.relative_to(ROOT_DIR)),
    )
    print(f"Experiment logged to database for version {model_version}")
    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train churn prediction model")
    parser.add_argument("--version", type=str, default="v2", help="Model version tag (e.g. v1, v2)")
    parser.add_argument("--depth", type=int, default=6, help="Tree depth for CatBoost")
    parser.add_argument("--iterations", type=int, default=400, help="Number of iterations")
    parser.add_argument("--learning-rate", type=float, default=0.03, help="Learning rate")
    args = parser.parse_args()

    params = {
        "depth": args.depth,
        "iterations": args.iterations,
        "learning_rate": args.learning_rate,
    }
    run_training(model_version=args.version, hyperparameters=params)

