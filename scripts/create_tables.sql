-- Supabase 建表脚本
-- 在 Supabase Dashboard → SQL Editor 中运行

CREATE TABLE training_sessions (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    stock_code TEXT NOT NULL,
    stock_name TEXT,
    trained_at TIMESTAMPTZ DEFAULT now(),
    forecast_days INT,
    selected_models JSONB,
    ensemble_weights JSONB,
    predictions JSONB,
    config_summary JSONB,
    last_close_price DOUBLE PRECISION,
    stock_data JSONB
);

CREATE TABLE model_results (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    session_id UUID REFERENCES training_sessions(id) ON DELETE CASCADE,
    model_name TEXT NOT NULL,
    cv_metrics JSONB,
    training_time DOUBLE PRECISION,
    future_predictions JSONB,
    future_conf_lower JSONB,
    future_conf_upper JSONB,
    test_predictions JSONB,
    test_actuals JSONB,
    confidence_lower JSONB,
    confidence_upper JSONB
);

CREATE INDEX idx_sessions_stock ON training_sessions(stock_code, trained_at DESC);
CREATE INDEX idx_results_session ON model_results(session_id);

ALTER TABLE training_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE model_results ENABLE ROW LEVEL SECURITY;
CREATE POLICY "public_read_write" ON training_sessions FOR ALL USING (true);
CREATE POLICY "public_read_write" ON model_results FOR ALL USING (true);
