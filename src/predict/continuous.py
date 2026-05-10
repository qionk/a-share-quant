"""
价格预测 - 持续学习与性能监控
滚动窗口训练 + 模型性能跟踪
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from .model_store import list_models, save_model, MODEL_DIR
from .models import ModelConfig
from .training import train_all_models, calc_metrics


def rolling_train(stock_code: str, df: pd.DataFrame, selected_models: list,
                  config: ModelConfig, window_size: int = 500,
                  forecast_days: int = 5, progress_callback=None) -> dict:
    """
    滚动窗口训练：取最近 window_size 天数据重新训练所有模型
    自动保存新版本
    返回: {model_name: TrainResult}
    """
    # 截取窗口
    if len(df) > window_size:
        df = df.iloc[-window_size:]

    # 训练
    results = train_all_models(df, selected_models, config,
                               forecast_days=forecast_days,
                               progress_callback=progress_callback)

    # 保存所有模型
    for name, result in results.items():
        if result.model_object is not None:
            try:
                save_model(stock_code, name, result, base_dir=MODEL_DIR)
            except Exception:
                pass

    return results


def track_performance(stock_code: str, base_dir: str = MODEL_DIR) -> pd.DataFrame:
    """
    读取模型元数据，构建性能时序
    返回 DataFrame: [train_date, model_type, mae, rmse, mape, r2]
    """
    models = list_models(stock_code, base_dir)
    records = []
    for m in models:
        metrics = m.metrics or {}
        records.append({
            "train_date": m.train_date,
            "model_type": m.model_type,
            "mae": metrics.get("mae", np.nan),
            "rmse": metrics.get("rmse", np.nan),
            "mape": metrics.get("mape", np.nan),
            "r2": metrics.get("r2", np.nan),
            "file_size_mb": m.file_size / 1024 / 1024,
        })

    df = pd.DataFrame(records)
    if not df.empty:
        df["train_date"] = pd.to_datetime(df["train_date"])
        df = df.sort_values("train_date")
    return df


def should_retrain(stock_code: str, days_threshold: int = 7,
                   base_dir: str = MODEL_DIR) -> dict:
    """
    检查各模型是否需要重新训练
    返回: {model_name: {"needs_update": bool, "reason": str, "last_trained": str}}
    """
    models = list_models(stock_code, base_dir)
    if not models:
        return {"all": {"needs_update": True, "reason": "无已保存模型", "last_trained": "无"}}

    # 按模型类型分组取最新
    latest_by_type = {}
    for m in models:
        if m.model_type not in latest_by_type:
            latest_by_type[m.model_type] = m

    result = {}
    now = datetime.now()
    for model_type, info in latest_by_type.items():
        try:
            train_dt = datetime.fromisoformat(info.train_date)
            days_old = (now - train_dt).days
            needs = days_old >= days_threshold
            reason = f"模型已 {days_old} 天未更新" if needs else "模型状态正常"
        except Exception:
            needs = True
            reason = "无法解析训练日期"
            days_old = -1

        result[model_type] = {
            "needs_update": needs,
            "reason": reason,
            "last_trained": info.train_date,
        }

    return result


def get_model_status(stock_code: str, base_dir: str = MODEL_DIR) -> str:
    """
    获取当前模型状态描述
    返回: "已更新至最新" / "有新数据可用" / "无模型"
    """
    models = list_models(stock_code, base_dir)
    if not models:
        return "无模型"

    latest = models[0]  # 已按时间倒序
    try:
        train_dt = datetime.fromisoformat(latest.train_date)
        days_old = (datetime.now() - train_dt).days
        if days_old <= 1:
            return "已更新至最新"
        elif days_old <= 7:
            return f"最近更新于 {days_old} 天前"
        else:
            return f"有新数据可用（{days_old} 天未更新）"
    except Exception:
        return "状态未知"


def cleanup_old_models(stock_code: str, keep_latest: int = 10,
                       base_dir: str = MODEL_DIR) -> int:
    """
    清理旧模型，保留最近 keep_latest 个版本
    返回删除数量
    """
    from .model_store import delete_model as del_model
    models = list_models(stock_code, base_dir)

    # 按模型类型分组
    by_type = {}
    for m in models:
        by_type.setdefault(m.model_type, []).append(m)

    deleted = 0
    for model_type, model_list in by_type.items():
        if len(model_list) > keep_latest:
            for m in model_list[keep_latest:]:
                if del_model(m.file_path):
                    deleted += 1

    return deleted
