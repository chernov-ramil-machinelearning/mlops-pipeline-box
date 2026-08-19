"""
ETL: сырой csv -> очистка -> запись в raw_data / features в PostgreSQL.
"""
import json
import os
from pathlib import Path

import pandas as pd
import psycopg2

ROOT_DIR = Path(__file__).resolve().parent.parent
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://mlops:mlops@localhost:5434/mlops_box")


def load_raw_csv(path: str) -> pd.DataFrame:
    """Загружает сырой csv как есть, без изменений."""
    file_path = Path(path)
    if not file_path.is_absolute():
        file_path = ROOT_DIR / file_path
    return pd.read_csv(file_path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Очистка датасета:
    1. Удаление строк с пропусками (NaN).
    2. Удаление дубликатов по customer_id / CustomerId.
    """
    df_clean = df.dropna().copy()
    id_col = 'CustomerId' if 'CustomerId' in df_clean.columns else 'customer_id'
    if id_col in df_clean.columns:
        df_clean = df_clean.drop_duplicates(subset=[id_col])
    return df_clean


def write_raw_to_db(df: pd.DataFrame, conn) -> None:
    """Пишет каждую строку в raw_data как JSONB."""
    id_col = 'CustomerId' if 'CustomerId' in df.columns else 'customer_id'
    with conn.cursor() as cur:
        for _, row in df.iterrows():
            cust_id = row.get(id_col)
            payload_json = json.dumps(row.to_dict(), default=str)
            cur.execute(
                "INSERT INTO raw_data (customer_id, payload) VALUES (%s, %s)",
                (cust_id, payload_json),
            )
    conn.commit()


def compute_features_sql(conn) -> None:
    """
    Генерирует фичи с использованием SQL оконных функций и сохраняет в features.
    """
    query = """
        INSERT INTO features (customer_id, feature_set)
        SELECT
            customer_id,
            jsonb_build_object(
                'CreditScore', (payload->>'CreditScore')::int,
                'Age', (payload->>'Age')::int,
                'Tenure', (payload->>'Tenure')::int,
                'Balance', (payload->>'Balance')::float,
                'NumOfProducts', (payload->>'NumOfProducts')::int,
                'HasCrCard', (payload->>'HasCrCard')::int,
                'IsActiveMember', (payload->>'IsActiveMember')::int,
                'EstimatedSalary', (payload->>'EstimatedSalary')::float,
                'Exited', (payload->>'Exited')::int,
                'avg_balance_for_tenure',
                    AVG((payload->>'Balance')::float)
                        OVER (PARTITION BY (payload->>'Tenure')::int),
                'Geography', payload->>'Geography',
                'Gender', payload->>'Gender'
            )
        FROM raw_data;
    """
    with conn.cursor() as cur:
        cur.execute(query)
    conn.commit()


def main(csv_path: str = "data/bank_churn.csv") -> None:
    print(f"Connecting to database: {DATABASE_URL}...")
    conn = psycopg2.connect(DATABASE_URL)

    with conn.cursor() as cur:
        cur.execute("TRUNCATE raw_data, features RESTART IDENTITY;")
    conn.commit()
    print("Truncated raw_data and features tables.")

    print(f"Loading raw dataset from {csv_path}...")
    df = load_raw_csv(csv_path)
    df_clean = clean(df)
    print(f"Cleaned data: {len(df_clean)} rows remaining (from original {len(df)} rows).")

    print("Writing raw data to PostgreSQL...")
    write_raw_to_db(df_clean, conn)

    print("Computing features via SQL window functions...")
    compute_features_sql(conn)
    print("ETL complete successfully.")

    conn.close()


if __name__ == "__main__":
    main()

