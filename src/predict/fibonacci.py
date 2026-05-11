"""
斐波那契回调分析
计算关键水平并生成交易信号
"""

import numpy as np
import pandas as pd

FIB_LEVELS = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.272, 1.618]
FIB_NAMES = {
    0.0: "低点", 0.236: "23.6%", 0.382: "38.2%", 0.5: "50.0%",
    0.618: "61.8%", 0.786: "78.6%", 1.0: "高点",
    1.272: "127.2%", 1.618: "161.8%",
}
FIB_COLORS = {
    0.0: "blue", 0.236: "orange", 0.382: "orange",
    0.5: "gray", 0.618: "green", 0.786: "green",
    1.0: "blue", 1.272: "purple", 1.618: "purple",
}


def calculate_fibonacci_levels(df: pd.DataFrame, lookback_days: int = 90) -> dict:
    """
    从最近 lookback_days 天内计算斐波那契回调水平。

    返回: {
        "levels": {level: price, ...},
        "swing_high": float,
        "swing_low": float,
        "range": float,
        "trend": "up" | "down",
    }
    """
    recent = df.tail(lookback_days)
    swing_high = recent["high"].max()
    swing_low = recent["low"].min()
    swing_range = swing_high - swing_low

    if swing_range < swing_high * 0.01:
        swing_range = swing_high * 0.01

    mid_idx = lookback_days // 2
    trend = "up" if recent["close"].iloc[-1] > recent["close"].iloc[min(mid_idx, len(recent) - 1)] else "down"

    levels = {}
    for level in FIB_LEVELS:
        levels[level] = swing_high - level * swing_range

    return {
        "levels": levels,
        "swing_high": swing_high,
        "swing_low": swing_low,
        "range": swing_range,
        "trend": trend,
    }


def generate_fibonacci_signals(current_price: float, fib_data: dict,
                                 predictions: dict = None) -> list:
    """
    基于斐波那契水平生成买卖信号。

    返回: [{"signal": "buy"/"sell", "level": str, "price": float, "description": str}, ...]
    """
    levels = fib_data["levels"]
    signals = []
    threshold = 0.02

    # 支撑位 -> 买入信号
    for level in [0.618, 0.786]:
        price = levels[level]
        if abs(current_price - price) / current_price < threshold:
            signals.append({
                "signal": "buy",
                "level": FIB_NAMES[level],
                "price": round(price, 2),
                "description": f"价格接近 {FIB_NAMES[level]} 支撑位 (¥{price:.2f})",
            })

    # 阻力位 -> 卖出信号
    for level in [0.236, 0.382]:
        price = levels[level]
        if abs(current_price - price) / current_price < threshold:
            signals.append({
                "signal": "sell",
                "level": FIB_NAMES[level],
                "price": round(price, 2),
                "description": f"价格接近 {FIB_NAMES[level]} 阻力位 (¥{price:.2f})",
            })

    # 与预测交叉检查
    if predictions and predictions.get("predicted_close") is not None \
       and len(predictions["predicted_close"]) > 0:
        pred_prices = predictions["predicted_close"]
        for level in [0.618, 0.786]:
            if any(abs(p - levels[level]) / max(p, 0.01) < 0.03 for p in pred_prices):
                signals.append({
                    "signal": "buy",
                    "level": FIB_NAMES[level],
                    "description": f"预测价格可能触及 {FIB_NAMES[level]} 支撑位",
                })
        for level in [0.236, 0.382]:
            if any(abs(p - levels[level]) / max(p, 0.01) < 0.03 for p in pred_prices):
                signals.append({
                    "signal": "sell",
                    "level": FIB_NAMES[level],
                    "description": f"预测价格可能触及 {FIB_NAMES[level]} 阻力位",
                })

    # 去重
    seen = set()
    unique = []
    for s in signals:
        if s["description"] not in seen:
            seen.add(s["description"])
            unique.append(s)
    return unique