"""
A股量化模型 - 信号层
多因子打分 + 信号平滑 + 市场状态过滤
"""

import pandas as pd
import numpy as np


def normalize_cross_section(factor_df):
    """横截面百分位标准化：每天所有股票互相排名 → 0~1"""
    return factor_df.rank(axis=1, pct=True)


def score_stocks(factors, weights):
    """
    加权打分
    factors: {name: DataFrame(date×code)}
    weights: {name: float}
    """
    total_w = sum(weights.values())
    score = None
    for name, w in weights.items():
        if name not in factors:
            continue
        normed = normalize_cross_section(factors[name])
        weighted = normed * (w / total_w)
        score = weighted if score is None else score.add(weighted, fill_value=0)
    return score


def smooth_signal(score_df, top_n, smooth_days=3):
    """
    信号平滑：连续 smooth_days 天位于 Top N 才确认
    返回 bool DataFrame
    """
    is_top = score_df.rank(axis=1, ascending=False) <= top_n
    rolling_count = is_top.astype(int).rolling(
        window=smooth_days, min_periods=smooth_days
    ).sum()
    return rolling_count >= smooth_days


def generate_signals(factors, breadth, config):
    """
    生成每日交易信号

    返回:
        signals  - bool DataFrame (True = 进入关注池)
        scores   - float DataFrame (综合评分)
        regime   - Series ("normal" / "caution" / "bear")
    """
    sig_cfg = config["signal"]
    mkt_cfg = config["market"]

    # 1. 综合评分
    scores = score_stocks(factors, sig_cfg["weights"])

    # 2. 信号平滑
    signals = smooth_signal(scores, sig_cfg["top_n"], sig_cfg["smooth_days"])

    # 3. 市场状态
    regime = pd.Series("normal", index=breadth.index)
    regime[breadth < mkt_cfg["breadth_threshold_low"]] = "bear"
    regime[
        (breadth >= mkt_cfg["breadth_threshold_low"])
        & (breadth < mkt_cfg["breadth_threshold_high"])
    ] = "caution"

    # 熊市清除买入信号
    bear_dates = regime[regime == "bear"].index
    bear_dates_in_signals = signals.index.intersection(bear_dates)
    if len(bear_dates_in_signals) > 0:
        signals.loc[bear_dates_in_signals] = False

    return signals, scores, regime
