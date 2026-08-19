"""
Сравнение model_v1 vs model_v2 на одной отложенной выборке.
Оценка статистической значимости улучшений (Bootstrap CI и критерий Манна-Уитни).
"""
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import psycopg2
from scipy.stats import mannwhitneyu
from sklearn.model_selection import train_test_split

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR / "training"))

from train import load_features

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://mlops:mlops@localhost:5434/mlops_box")


def bootstrap_ci_diff(
    metric_v1: np.ndarray,
    metric_v2: np.ndarray,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    random_state: int = 42,
):
    """
    Bootstrap-доверительный интервал для разницы (metric_v2 - metric_v1)
    по per-sample метрике (например accuracy / per-sample 0-1 loss).

    Возвращает (mean_diff, ci_low, ci_high, significant: bool).
    """
    rng = np.random.default_rng(random_state)
    diffs = []
    n = len(metric_v1)
    for _ in range(n_resamples):
        idx = rng.choice(n, size=n, replace=True)
        diffs.append(metric_v2[idx].mean() - metric_v1[idx].mean())

    diffs = np.array(diffs)
    ci_low, ci_high = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    significant = not (ci_low <= 0 <= ci_high)
    return float(diffs.mean()), float(ci_low), float(ci_high), bool(significant)


def compare_versions(per_sample_metric_v1: np.ndarray, per_sample_metric_v2: np.ndarray):
    """
    Сравнивает две версии моделей статистически.
    """
    mean_diff, ci_low, ci_high, significant = bootstrap_ci_diff(per_sample_metric_v1, per_sample_metric_v2)
    u_stat, p_value = mannwhitneyu(per_sample_metric_v1, per_sample_metric_v2)

    print(f"\n=================== A/B Model Evaluation ===================")
    print(f"Sample size (held-out): {len(per_sample_metric_v1)}")
    print(f"Model v1 accuracy:      {per_sample_metric_v1.mean():.4f}")
    print(f"Model v2 accuracy:      {per_sample_metric_v2.mean():.4f}")
    print(f"Bootstrap mean diff:    {mean_diff:+.4f}")
    print(f"95% Confidence Interval:[{ci_low:+.4f}, {ci_high:+.4f}]")
    print(f"Statistically significant (p < 0.05): {significant}")
    print(f"Mann-Whitney U statistic: {u_stat:.2f}, p-value: {p_value:.4e}")
    print(f"===========================================================\n")
    return {
        "mean_diff": mean_diff,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "significant": significant,
        "p_value": p_value,
    }


if __name__ == "__main__":
    print(f"Connecting to database: {DATABASE_URL}...")
    conn = psycopg2.connect(DATABASE_URL)
    print("Connected. Loading features...")
    df = load_features(conn)
    conn.close()

    print(f"Loaded {len(df)} rows. Splitting hold-out test set...")
    X = df.drop(columns=['Exited', 'customer_id'])
    y = df['Exited']
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("Loading models (v1 and v2)...")
    model_v1 = joblib.load(ROOT_DIR / 'models' / 'v1.cbm')
    model_v2 = joblib.load(ROOT_DIR / 'models' / 'v2.cbm')

    print("Running predictions...")
    pred_v1 = model_v1.predict(X_test)
    pred_v2 = model_v2.predict(X_test)

    per_sample_v1 = (pred_v1 == y_test.values).astype(int)
    per_sample_v2 = (pred_v2 == y_test.values).astype(int)

    compare_versions(per_sample_v1, per_sample_v2)

