-- 涨跌分类预测历史库
-- 用法: mysql -h <host> -u <user> -p <database> < create_clf_tables_mysql.sql

CREATE TABLE IF NOT EXISTS clf_training_sessions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    stock_code VARCHAR(20) NOT NULL,
    stock_name VARCHAR(100),
    trained_at DATETIME NOT NULL,
    forecast_days INT DEFAULT 1,
    threshold FLOAT DEFAULT 0.5,
    look_back INT DEFAULT 20,
    n_splits INT DEFAULT 5,
    selected_models JSON,
    params_json JSON,
    data_start_date DATE,
    data_end_date DATE,
    oos_start_date DATE,
    oos_end_date DATE,
    total_samples INT,
    ensemble_metrics JSON,
    model_metrics JSON,
    latest_date DATE,
    latest_proba FLOAT,
    latest_signal TINYINT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_clf_stock_code (stock_code),
    INDEX idx_clf_trained_at (trained_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS clf_prediction_details (
    id INT AUTO_INCREMENT PRIMARY KEY,
    session_id INT NOT NULL,
    trade_date DATE NOT NULL,
    next_day_proba FLOAT,
    fused_signal TINYINT,
    xgb_proba FLOAT,
    en_proba FLOAT,
    future_ret FLOAT,
    future_ret_valid TINYINT DEFAULT 1,
    next_day_ret FLOAT,
    INDEX idx_clf_session (session_id),
    INDEX idx_clf_trade_date (trade_date),
    FOREIGN KEY (session_id) REFERENCES clf_training_sessions(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS clf_feature_elimination_log (
    id INT AUTO_INCREMENT PRIMARY KEY,
    stock_code VARCHAR(20) NOT NULL,
    feature_name VARCHAR(100) NOT NULL,
    importance DOUBLE,
    rank_in_total INT,
    total_features INT,
    kept_features INT,
    eliminated_at DATETIME NOT NULL,
    INDEX idx_fe_stock (stock_code),
    INDEX idx_fe_feature (feature_name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;