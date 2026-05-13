"""
涨跌分类预测历史 MySQL 存取
save:  训练完成 → clf_training_sessions + clf_prediction_details
list:  某股票所有历史训练记录
load:  单条 session + details
delete: 删除 session（CASCADE 自动清除 details）
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, date

BATCH_SIZE = 500


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
        connect_timeout=10, read_timeout=30, write_timeout=30,
        autocommit=True,
    )


def _safe_float(v):
    if v is None:
        return None
    try:
        f = float(v)
        return None if (np.isnan(f) or np.isinf(f)) else f
    except (TypeError, ValueError):
        return None


def _to_date(v):
    """numpy datetime64 / Timestamp / str → date"""
    if v is None:
        return None
    if isinstance(v, (date, datetime)):
        return v.date() if isinstance(v, datetime) else v
    try:
        return pd.Timestamp(v).date()
    except Exception:
        return None


def save_clf_session(stock_code: str, stock_name: str, session_data: dict) -> int:
    """
    保存一次分类训练 session 到数据库。
    session_data 结构:
      forecast_days, threshold, look_back, n_splits,
      selected_models, params, results (Dict[str, ClfResult]),
      ensemble_result (dict or None), data_dates: tuple(start, end)
    返回 session_id (int)，数据库不可用时返回 -1
    """
    conn = _get_conn()
    if not conn:
        return -1

    cur = conn.cursor()

    trained_at = datetime.now()
    forecast_days = session_data.get("forecast_days", 1)
    threshold = session_data.get("threshold", 0.5)
    look_back = session_data.get("look_back", 20)
    n_splits = session_data.get("n_splits", 5)
    selected_models = session_data.get("selected_models", [])
    params = session_data.get("params", {})
    results = session_data.get("results", {})
    ensemble_result = session_data.get("ensemble_result")
    data_dates = session_data.get("data_dates")

    data_start = _to_date(data_dates[0]) if data_dates else None
    data_end = _to_date(data_dates[1]) if data_dates else None

    # OOS 日期范围
    oos_dates = None
    total_samples = 0
    if ensemble_result and len(ensemble_result.get("oos_dates", [])) > 0:
        oos_dates = ensemble_result["oos_dates"]
        total_samples = len(oos_dates)
    elif results:
        first_model = list(results.keys())[0]
        if hasattr(results[first_model], 'oos_dates') and len(results[first_model].oos_dates) > 0:
            oos_dates = results[first_model].oos_dates
            total_samples = len(oos_dates)

    oos_start = _to_date(oos_dates[0]) if oos_dates is not None and len(oos_dates) > 0 else None
    oos_end = _to_date(oos_dates[-1]) if oos_dates is not None and len(oos_dates) > 0 else None

    # 提取指标
    ensemble_metrics = None
    if ensemble_result:
        ensemble_metrics = _serialize_metrics(ensemble_result.get("metrics", {}))

    model_metrics = {}
    for model_name, r in results.items():
        if hasattr(r, 'overall_metrics'):
            model_metrics[model_name] = _serialize_metrics(r.overall_metrics)
    model_metrics_json = json.dumps(model_metrics, ensure_ascii=False, default=str) if model_metrics else None
    ensemble_metrics_json = json.dumps(ensemble_metrics, ensure_ascii=False, default=str) if ensemble_metrics else None

    # 写入 session
    sql_session = """
        INSERT INTO clf_training_sessions
        (stock_code, stock_name, trained_at, forecast_days, threshold,
         look_back, n_splits, selected_models, params_json,
         data_start_date, data_end_date, oos_start_date, oos_end_date,
         total_samples, ensemble_metrics, model_metrics)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """
    cur.execute(sql_session, (
        stock_code, stock_name, trained_at,
        forecast_days, threshold, look_back, n_splits,
        json.dumps(selected_models, ensure_ascii=False),
        json.dumps(params, ensure_ascii=False, default=str),
        data_start, data_end, oos_start, oos_end,
        total_samples,
        ensemble_metrics_json,
        model_metrics_json,
    ))
    session_id = cur.lastrowid

    # 写入 details
    if oos_dates is not None and total_samples > 0:
        details = _build_details_rows(
            session_id, oos_dates, results, ensemble_result, selected_models
        )
        sql_detail = """
            INSERT INTO clf_prediction_details
            (session_id, trade_date, next_day_proba, fused_signal,
             xgb_proba, en_proba, future_ret, future_ret_valid, next_day_ret)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """
        for i in range(0, len(details), BATCH_SIZE):
            batch = details[i:i + BATCH_SIZE]
            cur.executemany(sql_detail, batch)

    cur.close()
    conn.close()
    return session_id


def _serialize_metrics(m: dict) -> dict:
    """将 metrics dict 中 numpy 值转为 Python 原生类型，confusion 展平"""
    out = {}
    for k, v in m.items():
        if k == "confusion":
            out["confusion_tp"] = int(v.get("tp", 0))
            out["confusion_tn"] = int(v.get("tn", 0))
            out["confusion_fp"] = int(v.get("fp", 0))
            out["confusion_fn"] = int(v.get("fn", 0))
        elif isinstance(v, (np.floating, float)):
            out[k] = float(v) if not np.isnan(v) else None
        elif isinstance(v, (np.integer, int)):
            out[k] = int(v)
        else:
            out[k] = v
    return out


def _build_details_rows(session_id, oos_dates, results, ensemble_result, selected_models):
    """构建 prediction_details 批量写入行"""
    rows = []
    n = len(oos_dates)

    for i in range(n):
        d = _to_date(oos_dates[i])

        # ensemble
        next_day_proba = None
        fused_signal = None
        if ensemble_result and i < len(ensemble_result.get("fused_proba", [])):
            next_day_proba = _safe_float(ensemble_result["fused_proba"][i])
            fused_signal = int(ensemble_result["fused_signal"][i])

        # per-model
        xgb_proba = None
        en_proba = None
        if "XGBoost" in results and i < len(results["XGBoost"].oos_probabilities):
            xgb_proba = _safe_float(results["XGBoost"].oos_probabilities[i])
        if "ElasticNet" in results and i < len(results["ElasticNet"].oos_probabilities):
            en_proba = _safe_float(results["ElasticNet"].oos_probabilities[i])

        # future_ret
        future_ret = None
        future_ret_valid = 1
        next_day_ret = None
        first_model = selected_models[0] if selected_models else list(results.keys())[0]
        if first_model in results and i < len(results[first_model].oos_future_ret):
            fr = results[first_model].oos_future_ret[i]
            if np.isnan(fr):
                future_ret = None
                future_ret_valid = 0
            else:
                future_ret = float(fr)
        if first_model in results and hasattr(results[first_model], 'oos_next_day_ret') and i < len(results[first_model].oos_next_day_ret):
            ndr = results[first_model].oos_next_day_ret[i]
            next_day_ret = None if np.isnan(ndr) else float(ndr)

        rows.append((
            session_id, d,
            next_day_proba, fused_signal,
            xgb_proba, en_proba,
            future_ret, future_ret_valid, next_day_ret,
        ))

    return rows


def list_clf_sessions(stock_code: str) -> list:
    """返回某股票所有历史 session 概览，按时间倒序"""
    conn = _get_conn()
    if not conn:
        return []
    cur = conn.cursor()
    cur.execute(
        "SELECT id, stock_code, stock_name, trained_at, forecast_days, threshold, "
        "look_back, n_splits, selected_models, params_json, "
        "data_start_date, data_end_date, oos_start_date, oos_end_date, "
        "total_samples, ensemble_metrics, model_metrics, created_at "
        "FROM clf_training_sessions WHERE stock_code=%s ORDER BY trained_at DESC",
        (stock_code,))
    cols = [d[0] for d in cur.description]
    rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    cur.close()
    conn.close()

    for r in rows:
        r["session_id"] = r["id"]
        for key in ["selected_models", "params_json", "ensemble_metrics", "model_metrics"]:
            if r.get(key) and isinstance(r[key], str):
                try:
                    r[key.replace("_json", "")] = json.loads(r[key])
                except json.JSONDecodeError:
                    pass
        for date_key in ["trained_at", "data_start_date", "data_end_date",
                         "oos_start_date", "oos_end_date", "created_at"]:
            if r.get(date_key) and hasattr(r[date_key], "strftime"):
                r[date_key] = r[date_key].strftime("%Y-%m-%d %H:%M")
            elif r.get(date_key) and hasattr(r[date_key], "isoformat"):
                r[date_key] = r[date_key].isoformat()

    return rows


def load_clf_session(session_id: int) -> dict:
    """加载完整 session + prediction details"""
    conn = _get_conn()
    if not conn:
        return {}

    cur = conn.cursor()

    # session 主记录
    cur.execute(
        "SELECT * FROM clf_training_sessions WHERE id=%s", (session_id,))
    cols = [d[0] for d in cur.description]
    row = cur.fetchone()
    if not row:
        cur.close()
        conn.close()
        return {}

    session = dict(zip(cols, row))
    session["session_id"] = session["id"]

    # 解析 JSON
    for key in ["selected_models", "params_json", "ensemble_metrics", "model_metrics"]:
        if session.get(key) and isinstance(session[key], str):
            try:
                plain_key = key.replace("_json", "")
                session[plain_key] = json.loads(session[key])
            except json.JSONDecodeError:
                pass

    # 日期格式化
    for dk in ["trained_at", "data_start_date", "data_end_date",
               "oos_start_date", "oos_end_date", "created_at"]:
        if session.get(dk) and hasattr(session[dk], "strftime"):
            session[dk] = session[dk].strftime("%Y-%m-%d %H:%M")

    # details
    cur.execute(
        "SELECT trade_date, next_day_proba, fused_signal, xgb_proba, en_proba, "
        "future_ret, future_ret_valid, next_day_ret "
        "FROM clf_prediction_details WHERE session_id=%s ORDER BY trade_date",
        (session_id,))
    dcols = [d[0] for d in cur.description]
    drows = [dict(zip(dcols, r)) for r in cur.fetchall()]
    for dr in drows:
        if hasattr(dr["trade_date"], "strftime"):
            dr["trade_date"] = dr["trade_date"].strftime("%Y-%m-%d")

    session["details"] = drows

    cur.close()
    conn.close()
    return session


def delete_clf_session(session_id: int):
    """删除 session 及关联 details"""
    conn = _get_conn()
    if not conn:
        return
    cur = conn.cursor()
    cur.execute("DELETE FROM clf_training_sessions WHERE id=%s", (session_id,))
    cur.close()
    conn.close()