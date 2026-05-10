"""
A股量化模型 - 回测层
向量化回测，支持 T+1、涨跌停、手续费、止损、市场状态
"""

import pandas as pd
import numpy as np


def backtest(signals, scores, close_df, config):
    """
    策略回测

    逻辑：
    - 收盘后看信号 → 第二天以收盘价近似成交（含滑点）
    - T+1: 买入当天不能卖出
    - 等权分配，最多持有 max_positions 只
    - 单票止损

    参数:
        signals   - bool DataFrame (date×code)
        scores    - float DataFrame (date×code)，用于排优先级
        close_df  - 收盘价 DataFrame (date×code)
        config    - 配置字典

    返回:
        dict: daily_returns, trade_log, final_value
    """
    bt = config["backtest"]
    max_pos = bt["max_positions"]
    commission = bt["commission_rate"]
    stamp_tax = bt["stamp_tax_rate"]
    slippage = bt["slippage"]
    stop_loss = bt["stop_loss"]

    daily_ret = close_df.pct_change()

    # 对齐日期范围
    dates = signals.index.intersection(close_df.index).sort_values()
    if bt.get("start_date"):
        dates = dates[dates >= bt["start_date"]]
    if bt.get("end_date"):
        dates = dates[dates <= bt["end_date"]]

    # 状态
    holdings = {}          # code -> {"entry_price", "entry_date"}
    nav = 1.0
    records = []
    trades = []

    for i in range(1, len(dates)):
        date = dates[i]
        prev = dates[i - 1]

        # ── 当日持仓收益 ─────────────────────────────────
        weight = 1.0 / max_pos if max_pos > 0 else 0
        port_ret = 0.0

        for code in list(holdings.keys()):
            if code in daily_ret.columns and date in daily_ret.index:
                r = daily_ret.loc[date, code]
                if pd.notna(r):
                    port_ret += r * weight

        # ── 止损检查 ─────────────────────────────────────
        to_stop = []
        for code, pos in holdings.items():
            if code in close_df.columns and date in close_df.index:
                cur = close_df.loc[date, code]
                if pd.notna(cur) and pos["entry_price"] > 0:
                    total_ret = (cur - pos["entry_price"]) / pos["entry_price"]
                    if total_ret <= stop_loss:
                        to_stop.append(code)

        for code in to_stop:
            cost = weight * (commission + stamp_tax + slippage)
            port_ret -= cost
            trades.append({"date": date, "code": code, "action": "stop_loss"})
            del holdings[code]

        # ── 卖出：信号消失且满足 T+1 ────────────────────
        if prev in signals.index:
            active = set(signals.loc[prev][signals.loc[prev]].index)
            for code in list(holdings.keys()):
                if code not in active and holdings[code]["entry_date"] < prev:
                    cost = weight * (commission + stamp_tax + slippage)
                    port_ret -= cost
                    trades.append({"date": date, "code": code, "action": "sell"})
                    del holdings[code]

        # ── 买入：新信号，按评分排序 ────────────────────
        if prev in signals.index and prev in scores.index:
            candidates = signals.loc[prev][signals.loc[prev]].index
            # 排除已持有
            candidates = [c for c in candidates if c not in holdings]
            # 按评分降序
            s = scores.loc[prev].reindex(candidates).dropna().sort_values(ascending=False)
            slots = max_pos - len(holdings)
            buy_list = s.index[:slots].tolist()

            for code in buy_list:
                cost = weight * (commission + slippage)
                port_ret -= cost
                price = close_df.loc[date, code] if (
                    code in close_df.columns and date in close_df.index
                ) else np.nan
                holdings[code] = {
                    "entry_price": price if pd.notna(price) else 0,
                    "entry_date": date,
                }
                trades.append({"date": date, "code": code, "action": "buy"})

        # ── 更新净值 ─────────────────────────────────────
        nav *= (1 + port_ret)
        records.append({"date": date, "return": port_ret, "value": nav})

    return {
        "daily_returns": pd.DataFrame(records).set_index("date") if records else pd.DataFrame(),
        "trade_log": pd.DataFrame(trades) if trades else pd.DataFrame(),
        "final_value": nav,
    }


# ── 绩效指标 ────────────────────────────────────────────────


def calc_metrics(daily_df):
    """计算核心回测绩效指标"""
    if daily_df.empty:
        return {}

    ret = daily_df["return"]
    val = daily_df["value"]
    n = len(ret)

    total_return = val.iloc[-1] / val.iloc[0] - 1
    annual_return = (1 + total_return) ** (252 / max(n, 1)) - 1

    peak = val.cummax()
    dd = (val - peak) / peak
    max_dd = dd.min()

    sharpe = ret.mean() / ret.std() * np.sqrt(252) if ret.std() > 0 else 0
    calmar = annual_return / abs(max_dd) if max_dd != 0 else 0
    win_rate = (ret > 0).mean()

    return {
        "总收益":    f"{total_return:.2%}",
        "年化收益":  f"{annual_return:.2%}",
        "最大回撤":  f"{max_dd:.2%}",
        "夏普比率":  f"{sharpe:.2f}",
        "Calmar比率": f"{calmar:.2f}",
        "日胜率":    f"{win_rate:.2%}",
        "交易天数":  n,
    }
