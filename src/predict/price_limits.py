"""
A股涨跌停限制检测与预测修正
"""

import numpy as np


def detect_price_limit_pct(stock_code: str, stock_name: str = "") -> float:
    """
    根据股票代码和名称检测涨跌停比例。
    返回: 涨跌幅限制百分比（如 0.10 = 10%），无法识别返回 None
    """
    code = str(stock_code).strip()

    # ST / *ST 股票 (5%)
    if stock_name and ("ST" in stock_name.upper()):
        return 0.05

    if code.startswith("688"):
        return 0.20  # 科创板 STAR Market
    if code.startswith("300") or code.startswith("301") or code.startswith("302"):
        return 0.20  # 创业板 ChiNext/GEM
    if code.startswith("60") or code.startswith("00"):
        return 0.10  # 沪深主板
    if code.startswith("83") or code.startswith("87") or code.startswith("43"):
        return 0.30  # 北交所/新三板 BSE/NEEQ

    return None


def get_board_name(stock_code: str) -> str:
    """获取板块名称（含涨跌幅信息）"""
    code = str(stock_code).strip()
    if code.startswith("688"):
        return "科创板 (±20%)"
    if code.startswith("300") or code.startswith("301") or code.startswith("302"):
        return "创业板/GEM (±20%)"
    if code.startswith("60"):
        return "沪市主板 (±10%)"
    if code.startswith("00"):
        return "深市主板 (±10%)"
    if code.startswith("83") or code.startswith("87") or code.startswith("43"):
        return "北交所/新三板 (±30%)"
    return "未知板块"


def apply_price_limits(predictions, last_close: float, limit_pct: float):
    """
    将预测价格裁剪到涨跌停范围内。
    predictions: ensemble_predict() 返回的 dict
    返回: 裁剪后的 dict
    """
    if limit_pct is None:
        return predictions

    upper = last_close * (1 + limit_pct)
    lower = last_close * (1 - limit_pct)

    result = predictions.copy()
    for key in ["predicted_close", "confidence_lower", "confidence_upper"]:
        if key in result and result[key] is not None and len(result[key]) > 0:
            arr = result[key] if isinstance(result[key], np.ndarray) else np.array(result[key])
            result[key] = np.clip(arr, lower, upper)

    if "model_predictions" in result and result["model_predictions"]:
        for k, v in result["model_predictions"].items():
            arr = v if isinstance(v, np.ndarray) else np.array(v)
            result["model_predictions"][k] = np.clip(arr, lower, upper)

    return result