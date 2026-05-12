-- SQLPub / MySQL 建表脚本
-- 在 SQLPub 控制台 → SQL 工具中运行（www.sqlpub.com）

CREATE TABLE training_sessions (
    id VARCHAR(36) PRIMARY KEY,
    stock_code VARCHAR(20) NOT NULL,
    stock_name VARCHAR(100),
    trained_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    forecast_days INT DEFAULT 5,
    selected_models JSON,
    ensemble_weights JSON,
    predictions JSON,
    config_summary JSON,
    last_close_price DOUBLE,
    stock_data JSON
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE model_results (
    id VARCHAR(36) PRIMARY KEY,
    session_id VARCHAR(36) NOT NULL,
    model_name VARCHAR(50) NOT NULL,
    cv_metrics JSON,
    training_time DOUBLE,
    future_predictions JSON,
    future_conf_lower JSON,
    future_conf_upper JSON,
    test_predictions JSON,
    test_actuals JSON,
    test_returns JSON,
    test_returns_actual JSON,
    confidence_lower JSON,
    confidence_upper JSON,
    FOREIGN KEY (session_id) REFERENCES training_sessions(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE INDEX idx_sessions_stock ON training_sessions(stock_code, trained_at);
CREATE INDEX idx_results_session ON model_results(session_id);