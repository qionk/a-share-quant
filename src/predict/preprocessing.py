"""
统一数据预处理管线
- 生成日收益率（所有模型的预测目标）
- 生成成交量衍生特征
- 时间序列划分（禁止随机）
"""

import numpy as np
import pandas as pd


def preprocess_data(df: pd.DataFrame, return_clip: float = 0.10,
                     forecast_days: int = 1,
                     index_returns: pd.Series = None) -> pd.DataFrame:
    """
    统一数据预处理：计算日收益率目标 + 成交量衍生特征

    返回的 DataFrame 新增列：
      日收益率, future_ret, 目标收益率, 涨跌标签, 目标涨跌,
      成交量变化率, 相对成交量, 量价配合度, 放量上涨, 缩量下跌

    df 必须已有列: close, volume, pct_change（或从close计算）
    forecast_days: 持有天数，N日持有期收益 = close.pct_change(N).shift(-N)
    index_returns: 可选，板块/大盘日收益率Series（index对齐），用于计算相对强弱
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

    # ── 3. 预测目标：下一日涨跌方向（固定1日，与持有期无关） ──
    df['future_ret'] = df['close'].pct_change(periods=forecast_days).shift(-forecast_days)
    df['目标收益率'] = df['future_ret']
    # 训练目标始终是下一日方向
    _next_ret = df['close'].pct_change(periods=1).shift(-1)
    df['目标涨跌'] = _next_ret.gt(0).astype(float)
    df.loc[_next_ret.isna(), '目标涨跌'] = np.nan

    # 策略回测用：下一日收益率（T日收盘买入 → T+1日收盘卖出）
    df['next_day_ret'] = df['日收益率'].shift(-1)

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

    # ── 5. 多周期动量特征 ────────────────────────────
    df['ret_2d'] = df['close'].pct_change(2)
    df['ret_3d'] = df['close'].pct_change(3)
    df['ret_5d'] = df['close'].pct_change(5)
    df['ret_10d'] = df['close'].pct_change(10)

    # ── 6. 均线偏离度 ─────────────────────────────
    ma5 = df['close'].rolling(5).mean()
    ma10 = df['close'].rolling(10).mean()
    ma20 = df['close'].rolling(20).mean()
    df['close_ma5_bias'] = (df['close'] - ma5) / ma5
    df['close_ma10_bias'] = (df['close'] - ma10) / ma10
    df['close_ma20_bias'] = (df['close'] - ma20) / ma20
    df['ma5_ma10_cross'] = (ma5 - ma10) / ma10

    # ── 7. 波动率特征 ─────────────────────────────
    df['volatility_5d'] = df['日收益率'].rolling(5).std()
    df['volatility_10d'] = df['日收益率'].rolling(10).std()
    df['volatility_20d'] = df['日收益率'].rolling(20).std()

    # ── 8. K线形态特征 ─────────────────────────────
    df['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['close']
    df['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['close']
    df['body_ratio'] = (df['close'] - df['open']) / (df['high'] - df['low']).replace(0, np.nan)
    df['high_low_range'] = (df['high'] - df['low']) / df['close']

    # ── 9. 多周期 RSI ─────────────────────────────
    for period in [6, 14]:
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0.0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
        rs = gain / loss.replace(0, np.nan)
        df[f'rsi_{period}'] = 100 - 100 / (1 + rs)

    # ── 10. 成交量多周期特征 ──────────────────────────
    df['vol_chg_3d'] = df['volume'].pct_change(3)
    df['vol_chg_5d'] = df['volume'].pct_change(5)
    vol_ma5 = df['volume'].rolling(5).mean()
    df['vol_ratio_5d'] = df['volume'] / vol_ma5.replace(0, np.nan)

    # ── 11. 科技股专用特征 ─────────────────────────────

    # 换手率系列（科技股情绪核心指标）
    # 仅在 turnover 有效数据超过 50% 时计算，避免 DB 缓存的腾讯源数据全为 NaN
    has_turnover = ('turnover' in df.columns and
                    df['turnover'].notna().mean() > 0.5)
    if has_turnover:
        df['turnover'] = pd.to_numeric(df['turnover'], errors='coerce')
        df['turnover_ma5'] = df['turnover'].rolling(5).mean()
        df['turnover_ma10'] = df['turnover'].rolling(10).mean()
        df['turnover_bias'] = (df['turnover'] - df['turnover_ma5']) / df['turnover_ma5'].replace(0, np.nan)
        df['turnover_accel'] = df['turnover'].diff().rolling(3).mean()
        df['cum_turnover_5d'] = df['turnover'].rolling(5).sum()
        df['cum_turnover_10d'] = df['turnover'].rolling(10).sum()
    elif 'turnover' in df.columns:
        df = df.drop(columns=['turnover'])

    # 跳空缺口
    prev_close = df['close'].shift(1)
    df['gap'] = (df['open'] - prev_close) / prev_close
    df['gap_abs'] = df['gap'].abs()
    df['gap_up'] = (df['gap'] > 0.01).astype(int)
    df['gap_down'] = (df['gap'] < -0.01).astype(int)

    # 价格位置特征
    for n in [5, 10, 20]:
        rolling_high = df['high'].rolling(n).max()
        rolling_low = df['low'].rolling(n).min()
        price_range = (rolling_high - rolling_low).replace(0, np.nan)
        df[f'price_pos_{n}d'] = (df['close'] - rolling_low) / price_range
        df[f'drawdown_{n}d'] = (df['close'] - rolling_high) / rolling_high

    # 连涨连跌天数
    up = (df['日收益率'] > 0).astype(int)
    down = (df['日收益率'] < 0).astype(int)
    streak_up = up.copy()
    streak_down = down.copy()
    for i in range(1, len(df)):
        if up.iloc[i] == 1:
            streak_up.iloc[i] = streak_up.iloc[i - 1] + 1
        if down.iloc[i] == 1:
            streak_down.iloc[i] = streak_down.iloc[i - 1] + 1
    df['streak_up'] = streak_up
    df['streak_down'] = streak_down

    # ATR（平均真实波幅）
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift(1)).abs(),
        (df['low'] - df['close'].shift(1)).abs(),
    ], axis=1).max(axis=1)
    df['atr_5'] = tr.rolling(5).mean()
    df['atr_14'] = tr.rolling(14).mean()
    df['atr_ratio'] = df['atr_5'] / df['atr_14'].replace(0, np.nan)

    # 涨停/跌停统计（科技股涨停板效应）
    df['near_limit_up'] = (df['日收益率'] >= 0.09).astype(int)
    df['near_limit_down'] = (df['日收益率'] <= -0.09).astype(int)
    df['limit_up_count_10d'] = df['near_limit_up'].rolling(10).sum()
    df['limit_down_count_10d'] = df['near_limit_down'].rolling(10).sum()

    # ── 12. 相对强弱（vs 大盘/板块指数） ─────────────────
    if index_returns is not None:
        idx_ret = index_returns.reindex(df.index)
        df['excess_ret'] = df['日收益率'] - idx_ret.fillna(0)
        df['excess_ret_5d'] = df['excess_ret'].rolling(5).sum()
        df['excess_ret_10d'] = df['excess_ret'].rolling(10).sum()
        df['excess_ret_20d'] = df['excess_ret'].rolling(20).sum()

    # ── 13. 处理缺失值 ─────────────────────────────
    # 保留 目标涨跌/future_ret/next_day_ret 为 NaN 的行（最后一天无下日数据），
    # 仅删除特征列（日收益率）为 NaN 的行（首行）
    df = df.dropna(subset=['日收益率'])

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