-- Схема на старте. Дополняй по ходу разработки — это не финальная версия,
-- а достаточная основа, чтобы начать.

-- Сырые данные как есть, без очистки (загружаются ETL-скриптом)
CREATE TABLE IF NOT EXISTS raw_data (
    id              SERIAL PRIMARY KEY,
    customer_id     INTEGER,
    loaded_at       TIMESTAMP DEFAULT NOW(),
    payload         JSONB NOT NULL  -- сырая строка как JSON, гибко под любой источник
);

-- Очищенные фичи после ETL (результат SQL-запросов с оконными функциями/CTE)
CREATE TABLE IF NOT EXISTS features (
    id              SERIAL PRIMARY KEY,
    customer_id     INTEGER,
    feature_set     JSONB NOT NULL,
    computed_at     TIMESTAMP DEFAULT NOW()
);

-- Лог каждого обучения: что запускали, с какими параметрами, что получили
CREATE TABLE IF NOT EXISTS experiments (
    id              SERIAL PRIMARY KEY,
    model_version   TEXT NOT NULL,          -- 'v1', 'v2', ...
    hyperparameters JSONB,
    metrics         JSONB,                  -- {"roc_auc": 0.87, "f1": 0.81, ...}
    model_path      TEXT,
    trained_at      TIMESTAMP DEFAULT NOW()
);

-- Каждое предсказание, сделанное через API — нужно и для мониторинга, и для A/B
CREATE TABLE IF NOT EXISTS predictions (
    id              SERIAL PRIMARY KEY,
    model_version   TEXT NOT NULL,
    input_payload   JSONB NOT NULL,
    prediction      DOUBLE PRECISION,
    predicted_at    TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_predictions_model_version ON predictions(model_version);
CREATE INDEX IF NOT EXISTS idx_experiments_model_version ON experiments(model_version);
