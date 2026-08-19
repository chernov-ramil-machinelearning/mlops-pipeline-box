"""
Unit-тесты для модуля ETL (etl/load_and_clean.py).
Проверяют корректность базовой логики очистки данных без обращения к БД и ML-модели.
"""
import sys
from pathlib import Path
import pandas as pd
import pytest

# Гарантируем корректный импорт модулей проекта при запуске pytest из любого каталога
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from etl.load_and_clean import clean


def test_clean_drops_rows_with_nan():
    """1. Проверяет, что строки со значениями NaN полностью удаляются."""
    df = pd.DataFrame({
        "customer_id": [1, 2, 3],
        "Balance": [1000.0, None, 3000.0],
        "Age": [25, 30, 45],
    })
    result = clean(df)
    assert result.isna().sum().sum() == 0
    assert len(result) == 2
    assert set(result["customer_id"]) == {1, 3}


def test_clean_drops_duplicate_customer_id():
    """2. Проверяет, что дубликаты по идентификатору клиента удаляются."""
    df = pd.DataFrame({
        "customer_id": [101, 101, 102],
        "Balance": [500.0, 500.0, 1500.0],
        "Age": [40, 40, 35],
    })
    result = clean(df)
    assert len(result) == 2
    assert list(result["customer_id"]) == [101, 102]


def test_clean_returns_dataframe_not_none():
    """3. Проверяет, что функция возвращает валидный DataFrame, а не None."""
    df = pd.DataFrame({
        "customer_id": [1],
        "Balance": [1000.0],
    })
    result = clean(df)
    assert result is not None
    assert isinstance(result, pd.DataFrame)
    assert not result.empty

