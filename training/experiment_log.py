"""
Простой трекер экспериментов — не MLflow, но закрывает ту же идею:
после каждого обучения фиксируем что запускали и что получили.
"""
import json


def log_experiment(conn, model_version: str, hyperparameters: dict, metrics: dict, model_path: str) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO experiments (model_version, hyperparameters, metrics, model_path)
            VALUES (%s, %s, %s, %s)
            """,
            (model_version, json.dumps(hyperparameters), json.dumps(metrics), model_path),
        )
    conn.commit()
