"""
SQLPub / MySQL 云端存储 — 训练结果共享
对 predict_app.py 保持与 supabase_store.py 相同的函数签名
"""

import os
import json
import uuid as _uuid
import numpy as np
from datetime import datetime


def _get_conn():
    try:
        import streamlit as st
        host = st.secrets.get("MYSQL_HOST", os.environ.get("MYSQL_HOST", ""))
        port = int(st.secrets.get("MYSQL_PORT", os.environ.get("MYSQL_PORT", "3306")))
        user = st.secrets.get("MYSQL_USER", os.environ.get("MYSQL_USER", ""))
        password = st.secrets.get("MYSQL_PASSWORD", os.environ.get("MYSQL_PASSWORD", ""))
        database = st.secrets.get("MYSQL_DATABASE", os.environ.get("MYSQL_DATABASE", ""))
    except Exception:
        host = os.environ.get("MYSQL_HOST", "")
        port = int(os.environ.get("MYSQL_PORT", "3306"))
        user = os.environ.get("MYSQL_USER", "")
        password = os.environ.get("MYSQL_PASSWORD", "")
        database = os.environ.get("MYSQL_DATABASE", "")

    if not host or not user or not database:
        return None

    import pymysql
    return pymysql.connect(
        host=host, port=port, user=user, password=password,
        database=database, charset="utf8mb4",
        connect_timeout=10, read_timeout=15, write_timeout=15,
        autocommit=True,
    )


def is_configured() -> bool:
    try:
        import streamlit as st
        host = st.secrets.get("MYSQL_HOST", os.environ.get("MYSQL_HOST", ""))
        user = st.secrets.get("MYSQL_USER", os.environ.get("MYSQL_USER", ""))
        database = st.secrets.get("MYSQL_DATABASE", os.environ.get("MYSQL_DATABASE", ""))
    except Exception:
        host = os.environ.get("MYSQL_HOST", "")
        user = os.environ.get("MYSQL_USER", "")
        database = os.environ.get("MYSQL_DATABASE", "")
    return bool(host and user and database)


def _sanitize(v):
    if isinstance(v, np.ndarray):
        v = v.tolist()
    if isinstance(v, dict):
        return {k: _sanitize(val) for k, val in v.items()}
    if isinstance(v, list):
        return [_sanitize(x) for x in v]
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        f = float(v)
        return None if (np.isnan(f) or np.isinf(f)) else f
    return v


def _json_dumps(obj):
    return json.dumps(obj, ensure_ascii=False, default=str)


def _json_loads(s):
    if s is None:
        return None
    if isinstance(s, (dict, list)):
        return s
    return json.loads(s)


def _serialize_stock_data(df):
    import pandas as pd
    if df is None:
        return None
    trimmed = df.tail(200)
    return {
        "index": [d.strftime("%Y-%m-%d") for d in trimmed.index],
        "columns": list(trimmed.columns),
        "data": _sanitize(trimmed.values.tolist()),
    }


def _deserialize_stock_data(obj):
    import pandas as pd
    if not obj:
        return None
    obj = _json_loads(obj) if isinstance(obj, str) else obj
    idx = pd.to_datetime(obj["index"])
    df = pd.DataFrame(obj["data"], index=idx, columns=obj["columns"])
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ── 对外接口（签名与 supabase_store.py 一致）─────────────────────

def save_training_results(stock_code, stock_name, results, ensemble_weights,
                          predictions, config, forecast_days, selected_models,
                          stock_data=None):
    conn = _get_conn()
    if not conn:
        return None

    preds_json = None
    if predictions:
        preds_json = {}
        for k, v in predictions.items():
            if k == "model_predictions":
                preds_json[k] = {mk: _sanitize(mv) for mk, mv in v.items()}
            elif k == "weights":
                preds_json[k] = _sanitize(v) if isinstance(v, dict) else v
            else:
                preds_json[k] = _sanitize(v)

    config_summary = {
        "look_back": config.look_back,
        "n_features": config.n_features,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "early_stop_patience": config.early_stop_patience,
        "learning_rate": config.learning_rate,
        "dropout": config.dropout,
        "lstm_units": config.lstm_units,
        "gru_units": config.gru_units,
        "cnn_filters": config.cnn_filters,
        "cnn_kernel_size": config.cnn_kernel_size,
        "patchtst_patch_size": config.patchtst_patch_size,
        "patchtst_d_model": config.patchtst_d_model,
        "patchtst_n_heads": config.patchtst_n_heads,
        "patchtst_n_encoder_layers": config.patchtst_n_encoder_layers,
        "patchtst_ff_dim": config.patchtst_ff_dim,
        "patchtst_dropout": config.patchtst_dropout,
        "tft_hidden_size": config.tft_hidden_size,
        "tft_n_heads": config.tft_n_heads,
        "tft_dropout": config.tft_dropout,
        "tft_lstm_layers": config.tft_lstm_layers,
        "cnn_gru_filters": config.cnn_gru_filters,
        "cnn_gru_gru_units": config.cnn_gru_gru_units,
        "cnn_gru_kernel_size": config.cnn_gru_kernel_size,
        "xgboost_n_estimators": config.xgboost_n_estimators,
        "xgboost_max_depth": config.xgboost_max_depth,
        "xgboost_learning_rate": config.xgboost_learning_rate,
        "xgboost_subsample": config.xgboost_subsample,
        "lightgbm_n_estimators": config.lightgbm_n_estimators,
        "lightgbm_max_depth": config.lightgbm_max_depth,
        "lightgbm_learning_rate": config.lightgbm_learning_rate,
        "lightgbm_num_leaves": config.lightgbm_num_leaves,
        "sarima_order": list(config.sarima_order),
        "sarima_seasonal_order": list(config.sarima_seasonal_order),
    }

    session_id = str(_uuid.uuid4())
    session_sql = """
        INSERT INTO training_sessions
        (id, stock_code, stock_name, forecast_days, selected_models,
         ensemble_weights, predictions, config_summary, last_close_price, stock_data)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """
    cur = conn.cursor()
    cur.execute(session_sql, (
        session_id, stock_code, stock_name, forecast_days,
        _json_dumps(selected_models),
        _json_dumps(ensemble_weights),
        _json_dumps(preds_json),
        _json_dumps(config_summary),
        float(predictions["predicted_close"][0]) if predictions else None,
        _json_dumps(_serialize_stock_data(stock_data)),
    ))

    model_sql = """
        INSERT INTO model_results
        (id, session_id, model_name, cv_metrics, training_time,
         future_predictions, future_conf_lower, future_conf_upper,
         test_predictions, test_actuals, test_returns, test_returns_actual,
         confidence_lower, confidence_upper)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """
    for name, r in results.items():
        cur.execute(model_sql, (
            str(_uuid.uuid4()), session_id, name,
            _json_dumps(_sanitize(r.cv_metrics) if r.cv_metrics else {}),
            r.training_time,
            _json_dumps(_sanitize(r.future_predictions)),
            _json_dumps(_sanitize(r.future_conf_lower)),
            _json_dumps(_sanitize(r.future_conf_upper)),
            _json_dumps(_sanitize(r.test_predictions)),
            _json_dumps(_sanitize(r.test_actuals)),
            _json_dumps(_sanitize(r.test_returns)),
            _json_dumps(_sanitize(r.test_returns_actual)),
            _json_dumps(_sanitize(r.confidence_lower)),
            _json_dumps(_sanitize(r.confidence_upper)),
        ))

    cur.close()
    conn.close()
    return session_id


def load_latest_results(stock_code):
    conn = _get_conn()
    if not conn:
        return None

    cur = conn.cursor()
    cur.execute(
        "SELECT * FROM training_sessions WHERE stock_code=%s "
        "ORDER BY trained_at DESC LIMIT 1", (stock_code,))
    cols = [d[0] for d in cur.description]
    row = cur.fetchone()
    if not row:
        cur.close(); conn.close()
        return None
    session = dict(zip(cols, row))

    cur.execute("SELECT * FROM model_results WHERE session_id=%s", (session["id"],))
    mcols = [d[0] for d in cur.description]
    model_rows = [dict(zip(mcols, r)) for r in cur.fetchall()]

    cur.close(); conn.close()
    return session, model_rows


def list_available_stocks():
    conn = _get_conn()
    if not conn:
        return []
    cur = conn.cursor()
    cur.execute(
        "SELECT id, stock_code, stock_name, trained_at, selected_models, forecast_days "
        "FROM training_sessions ORDER BY trained_at DESC LIMIT 50"
    )
    cols = [d[0] for d in cur.description]
    rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    for r in rows:
        if hasattr(r["trained_at"], "isoformat"):
            r["trained_at"] = r["trained_at"].isoformat()
        r["selected_models"] = _json_loads(r["selected_models"])
    cur.close(); conn.close()
    return rows


def load_by_session_id(session_id):
    conn = _get_conn()
    if not conn:
        return None

    cur = conn.cursor()
    cur.execute("SELECT * FROM training_sessions WHERE id=%s", (session_id,))
    cols = [d[0] for d in cur.description]
    row = cur.fetchone()
    if not row:
        cur.close(); conn.close()
        return None
    session = dict(zip(cols, row))

    cur.execute("SELECT * FROM model_results WHERE session_id=%s", (session_id,))
    mcols = [d[0] for d in cur.description]
    model_rows = [dict(zip(mcols, r)) for r in cur.fetchall()]

    cur.close(); conn.close()
    return session, model_rows


def restore_to_session_state(session_row, model_rows):
    import streamlit as st
    from src.predict.training import TrainResult

    st.session_state.stock_code = session_row["stock_code"]
    st.session_state.stock_name = session_row["stock_name"]
    st.session_state.ensemble_weights = _json_loads(session_row.get("ensemble_weights"))

    # 不覆盖已加载的最新 stock_data（可能比训练时保存的更新）
    if st.session_state.get("stock_data") is None:
        st.session_state.stock_data = _deserialize_stock_data(session_row.get("stock_data"))

    predictions_raw = _json_loads(session_row.get("predictions"))
    if predictions_raw:
        restored = {}
        for k, v in predictions_raw.items():
            if k == "model_predictions":
                restored[k] = {mk: np.array(mv) for mk, mv in v.items()}
            elif isinstance(v, list):
                restored[k] = np.array(v)
            else:
                restored[k] = v
        st.session_state.predictions = restored

    results = {}
    for mr in model_rows:
        tr = TrainResult(model_name=mr["model_name"])
        tr.cv_metrics = _json_loads(mr.get("cv_metrics")) or {}
        tr.training_time = mr.get("training_time", 0)
        tr.future_predictions = np.array(_json_loads(mr.get("future_predictions")) or [])
        tr.future_conf_lower = np.array(_json_loads(mr.get("future_conf_lower")) or [])
        tr.future_conf_upper = np.array(_json_loads(mr.get("future_conf_upper")) or [])
        tr.test_predictions = np.array(_json_loads(mr.get("test_predictions")) or [])
        tr.test_actuals = np.array(_json_loads(mr.get("test_actuals")) or [])
        tr.test_returns = np.array(_json_loads(mr.get("test_returns")) or [])
        tr.test_returns_actual = np.array(_json_loads(mr.get("test_returns_actual")) or [])
        tr.confidence_lower = np.array(_json_loads(mr.get("confidence_lower")) or [])
        tr.confidence_upper = np.array(_json_loads(mr.get("confidence_upper")) or [])
        tr.train_history = {}
        tr.feature_cols = []
        tr.n_features = 0
        results[mr["model_name"]] = tr

    st.session_state.train_results = results