"""
Supabase 云端存储 — 训练结果共享
训练完成后自动保存，其他用户可直接加载
"""

import os
import numpy as np
from datetime import datetime


def _get_client():
    try:
        import streamlit as st
        url = st.secrets.get("SUPABASE_URL", os.environ.get("SUPABASE_URL", ""))
        key = st.secrets.get("SUPABASE_KEY", os.environ.get("SUPABASE_KEY", ""))
    except Exception:
        url = os.environ.get("SUPABASE_URL", "")
        key = os.environ.get("SUPABASE_KEY", "")

    if not url or not key:
        return None

    from supabase import create_client
    from supabase.lib.client_options import SyncClientOptions
    return create_client(url, key, options=SyncClientOptions(
        postgrest_client_timeout=15,
    ))


def is_configured() -> bool:
    try:
        import streamlit as st
        url = st.secrets.get("SUPABASE_URL", os.environ.get("SUPABASE_URL", ""))
        key = st.secrets.get("SUPABASE_KEY", os.environ.get("SUPABASE_KEY", ""))
    except Exception:
        url = os.environ.get("SUPABASE_URL", "")
        key = os.environ.get("SUPABASE_KEY", "")
    return bool(url and key)


def _sanitize(v):
    """将 numpy 数组转 list，并把 NaN/Inf 替换为 None（JSON 兼容），递归处理嵌套结构"""
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


def _serialize_stock_data(df):
    """将 DataFrame 序列化为 JSON 兼容的 dict，只保留最近 200 个交易日"""
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
    """从 JSON dict 恢复 DataFrame"""
    import pandas as pd
    if not obj:
        return None
    idx = pd.to_datetime(obj["index"])
    df = pd.DataFrame(obj["data"], index=idx, columns=obj["columns"])
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def save_training_results(stock_code, stock_name, results, ensemble_weights,
                          predictions, config, forecast_days, selected_models,
                          stock_data=None):
    client = _get_client()
    if not client:
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
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "dropout": config.dropout,
    }

    session_data = {
        "stock_code": stock_code,
        "stock_name": stock_name,
        "forecast_days": forecast_days,
        "selected_models": selected_models,
        "ensemble_weights": ensemble_weights,
        "predictions": preds_json,
        "config_summary": config_summary,
        "last_close_price": float(predictions["predicted_close"][0]) if predictions else None,
        "stock_data": _serialize_stock_data(stock_data),
    }

    resp = client.table("training_sessions").insert(session_data).execute()
    session_id = resp.data[0]["id"]

    model_rows = []
    for name, r in results.items():
        model_rows.append({
            "session_id": session_id,
            "model_name": name,
            "cv_metrics": _sanitize(r.cv_metrics) if r.cv_metrics else {},
            "training_time": r.training_time,
            "future_predictions": _sanitize(r.future_predictions),
            "future_conf_lower": _sanitize(r.future_conf_lower),
            "future_conf_upper": _sanitize(r.future_conf_upper),
            "test_predictions": _sanitize(r.test_predictions),
            "test_actuals": _sanitize(r.test_actuals),
            "confidence_lower": _sanitize(r.confidence_lower),
            "confidence_upper": _sanitize(r.confidence_upper),
        })

    if model_rows:
        client.table("model_results").insert(model_rows).execute()

    return session_id


def load_latest_results(stock_code):
    client = _get_client()
    if not client:
        return None

    resp = (client.table("training_sessions")
            .select("*")
            .eq("stock_code", stock_code)
            .order("trained_at", desc=True)
            .limit(1)
            .execute())

    if not resp.data:
        return None

    session = resp.data[0]
    model_resp = (client.table("model_results")
                  .select("*")
                  .eq("session_id", session["id"])
                  .execute())

    return session, model_resp.data


def list_available_stocks():
    client = _get_client()
    if not client:
        return []

    resp = (client.table("training_sessions")
            .select("id, stock_code, stock_name, trained_at, selected_models, forecast_days")
            .order("trained_at", desc=True)
            .limit(50)
            .execute())

    return resp.data or []


def load_by_session_id(session_id):
    client = _get_client()
    if not client:
        return None

    resp = (client.table("training_sessions")
            .select("*")
            .eq("id", session_id)
            .limit(1)
            .execute())

    if not resp.data:
        return None

    session = resp.data[0]
    model_resp = (client.table("model_results")
                  .select("*")
                  .eq("session_id", session["id"])
                  .execute())

    return session, model_resp.data


def restore_to_session_state(session_row, model_rows):
    import streamlit as st
    from src.predict.training import TrainResult

    st.session_state.stock_code = session_row["stock_code"]
    st.session_state.stock_name = session_row["stock_name"]
    st.session_state.ensemble_weights = session_row.get("ensemble_weights")

    # 恢复行情数据
    st.session_state.stock_data = _deserialize_stock_data(session_row.get("stock_data"))

    if session_row.get("predictions"):
        preds = session_row["predictions"]
        restored = {}
        for k, v in preds.items():
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
        tr.cv_metrics = mr.get("cv_metrics", {})
        tr.training_time = mr.get("training_time", 0)
        tr.future_predictions = np.array(mr.get("future_predictions") or [])
        tr.future_conf_lower = np.array(mr.get("future_conf_lower") or [])
        tr.future_conf_upper = np.array(mr.get("future_conf_upper") or [])
        tr.test_predictions = np.array(mr.get("test_predictions") or [])
        tr.test_actuals = np.array(mr.get("test_actuals") or [])
        tr.confidence_lower = np.array(mr.get("confidence_lower") or [])
        tr.confidence_upper = np.array(mr.get("confidence_upper") or [])
        tr.train_history = {}
        tr.feature_cols = []
        tr.n_features = 0
        results[mr["model_name"]] = tr

    st.session_state.train_results = results
