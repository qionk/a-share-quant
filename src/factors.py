"""
A股量化模型 - 因子层
因子计算 + Rank IC 检验
"""

import pandas as pd
import numpy as np
from scipy import stats


# ── 单因子计算（全部向量化，输入输出均为 date×code 的 DataFrame）──


def calc_ma(close_df, window):
    """移动平均"""
    return close_df.rolling(window=window, min_periods=window).mean()


def calc_relative_strength(close_df, window=20):
    """
    相对强度：N 日收益率在全市场的横截面百分位排名
    返回 0~1，越大越强
    """
    ret = close_df.pct_change(window)
    return ret.rank(axis=1, pct=True)


def calc_trend_score(close_df, ma_windows=(5, 20, 60)):
    """
    趋势得分：价格在各均线之上的比例
    close > MA5 > MA20 > MA60 → 得分接近 1
    """
    score = pd.DataFrame(0.0, index=close_df.index, columns=close_df.columns)
    for w in ma_windows:
        ma = calc_ma(close_df, w)
        score += (close_df > ma).astype(float) / len(ma_windows)
    return score


def calc_volatility(close_df, window=20):
    """N 日年化波动率（越低越好，返回负数用于排序）"""
    vol = close_df.pct_change().rolling(window, min_periods=window).std() * np.sqrt(252)
    return -vol  # 取负：低波动 → 高得分


def calc_volume_ratio(volume_df, window=20):
    """量比：当日成交量 / N 日均量"""
    ma = volume_df.rolling(window, min_periods=window).mean()
    return volume_df / ma


def calc_max_drawdown(close_df, window=20):
    """滚动 N 日最大回撤（越小越好，返回正数）"""
    rolling_max = close_df.rolling(window, min_periods=1).max()
    dd = close_df / rolling_max - 1       # 负数
    return -dd                              # 取负：低回撤 → 高得分


def calc_market_breadth(close_df, ma_window=20):
    """市场宽度：站上 N 日均线的股票占比（返回 Series, index=date）"""
    ma = calc_ma(close_df, ma_window)
    above = (close_df > ma).sum(axis=1)
    total = close_df.notna().sum(axis=1)
    return above / total


# ── Rank IC 检验 ─────────────────────────────────────────────


def calc_rank_ic(factor_df, forward_return_df):
    """
    因子 Rank IC: 每天的因子值与未来收益做 Spearman 秩相关
    返回 IC 时间序列
    """
    common_dates = factor_df.index.intersection(forward_return_df.index)
    records = []
    for date in common_dates:
        f = factor_df.loc[date].dropna()
        r = forward_return_df.loc[date].dropna()
        common = f.index.intersection(r.index)
        if len(common) < 30:
            continue
        ic, _ = stats.spearmanr(f[common], r[common])
        records.append({"date": date, "ic": ic})
    if not records:
        return pd.Series(dtype=float)
    return pd.DataFrame(records).set_index("date")["ic"]


# ── 组合调用 ─────────────────────────────────────────────────


def compute_all_factors(panel, config):
    """
    计算全部因子
    返回 (factors_dict, breadth_series)
    """
    close = panel["close"]
    volume = panel["volume"]
    cfg = config["factors"]

    factors = {
        "relative_strength": calc_relative_strength(close, cfg["momentum_window"]),
        "trend":             calc_trend_score(close, cfg["ma_windows"]),
        "volatility":        calc_volatility(close, cfg["volatility_window"]),
        "volume_activity":   calc_volume_ratio(volume, cfg["volume_ma_window"]),
        "max_drawdown":      calc_max_drawdown(close, cfg["momentum_window"]),
    }

    breadth = calc_market_breadth(
        close, config["market"].get("breadth_ma_window", 20)
    )

    return factors, breadth


def evaluate_factors(factors, close_df, forward_days=5):
    """
    对每个因子做 IC 评估
    forward_days: 以未来 N 日收益为目标变量
    """
    fwd_ret = close_df.pct_change(forward_days).shift(-forward_days)
    results = {}
    for name, fdf in factors.items():
        ic = calc_rank_ic(fdf, fwd_ret)
        if len(ic) == 0:
            continue
        results[name] = {
            "ic_mean": ic.mean(),
            "ic_std": ic.std(),
            "icir": ic.mean() / ic.std() if ic.std() > 0 else 0,
            "ic_positive_pct": (ic > 0).mean(),
            "ic_series": ic,
        }
    return results
