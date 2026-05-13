"""
统一数据预处理管线
- 生成日收益率（所有模型的预测目标）
- 生成成交量衍生特征
- 时间序列划分（禁止随机）
"""

import numpy as np
import pandas as pd


def preprocess_data(df: pd.DataFrame, return_clip: float = 0.10,
                     forecast_days: int = 1) -> pd.DataFrame:
    """
    统一数据预处理：计算日收益率目标 + 成交量衍生特征

    返回的 DataFrame 新增列：
      日收益率, future_ret, 目标收益率, 涨跌标签, 目标涨跌,
      成交量变化率, 相对成交量, 量价配合度, 放量上涨, 缩量下跌

    df 必须已有列: close, volume, pct_change（或从close计算）
    forecast_days: 持有天数，N日持有期收益 = close.pct_change(N).shift(-N)
    """
    df = df.copy()

    # ── 1. 日收益率（小数形式，如 0.015 = 1.5%） ──
    if 'pct_change' in df.columns:
        # AKShare/Tencent 返回的 pct_change 是百分比形式（1.5 代表 1.5%），转为小数
        df['日收益率'] = df['pct_change'] / 100.0
    else:
        df['日收益率'] = df['close'].pct_change()  # 小数形式，如 0.015 = 1.5%

    # 截断 ±10%（符合A股涨跌幅限制）
    df['日收益率'] = df['日收益率'].clip(-return_clip, return_clip)

    # ── 2. 涨跌标签 ──────────────────────────────
    df['涨跌标签'] = (df['日收益率'] > 0).astype(int)

    # ── 3. 预测目标：N日持有期方向 ──────────────
    df['future_ret'] = df['close'].pct_change(periods=forecast_days).shift(-forecast_days)
    df['目标涨跌'] = (df['future_ret'] > 0).astype(int)
    df['目标收益率'] = df['future_ret']

    # ── 4. 成交量衍生特征 ─────────────────────────
    # 成交量变化率
    df['成交量变化率'] = df['volume'].pct_change()

    # 相对成交量（与20日均值比）
    vol_ma20 = df['volume'].rolling(window=20).mean().replace(0, np.nan)
    df['相对成交量'] = df['volume'] / vol_ma20

    # 量价配合度（收益率与成交量变化率的滚动相关性）
    df['量价配合度'] = df['日收益率'].rolling(20).corr(df['成交量变化率'])

    # 放量上涨: 涨 + 相对成交量 > 1.5
    df['放量上涨'] = ((df['日收益率'] > 0) & (df['相对成交量'] > 1.5)).astype(int)

    # 缩量下跌: 跌 + 相对成交量 < 0.7
    df['缩量下跌'] = ((df['日收益率'] < 0) & (df['相对成交量'] < 0.7)).astype(int)

    # ── 5. 处理缺失值 ─────────────────────────────
    df = df.dropna(subset=['future_ret'])

    return df


def split_data(df: pd.DataFrame, test_size: float = 0.2) -> dict:
    """
    严格按时间顺序划分训练/测试集（禁止随机）。

    返回 dict:
      - train_df, test_df: 完整 DataFrame
      - split_index: 切分点位置
      - latest_close: 最新收盘价（用于反算预测价格）
    """
    split_idx = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    return {
        'train_df': train_df,
        'test_df': test_df,
        'split_index': split_idx,
        'latest_close': float(df['close'].iloc[-1]),
    }


def returns_to_price_series(last_close: float, predicted_returns: np.ndarray,
                            limit_pct: float = None) -> np.ndarray:
    """
    将预测收益率（%）转为预测收盘价序列。

    predicted_price[t] = prev_price * (1 + return[t]/100)
    可选应用涨跌幅限制。
    """
    if len(predicted_returns) == 0:
        return np.array([])

    prices = np.zeros(len(predicted_returns))
    prev = last_close
    for i, r in enumerate(predicted_returns):
        p = prev * (1 + r / 100)
        if limit_pct is not None:
            upper = last_close * (1 + limit_pct)
            lower = last_close * (1 - limit_pct)
            p = np.clip(p, lower, upper)
        prices[i] = p
        prev = p
    return prices


def calculate_predicted_prices(last_close: float, predicted_returns: np.ndarray,
                               limit_pct: float = None) -> tuple:
    """
    便捷函数：预测收益率 → 预测收盘价 + 日收益率%。

    返回 (predicted_close: np.ndarray, daily_return_pct: np.ndarray)
    """
    prices = returns_to_price_series(last_close, predicted_returns, limit_pct)
    if len(prices) <= 1:
        daily_ret = np.array([])
    else:
        extended = np.concatenate([[last_close], prices])
        daily_ret = (extended[1:] / extended[:-1] - 1) * 100
    return prices, daily_ret