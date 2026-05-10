#!/usr/bin/env python3
"""
A股量化辅助决策模型 - CLI 入口

用法:
    python run.py update        更新行情数据（首次较慢）
    python run.py signal        生成今日关注池
    python run.py backtest      运行策略回测
    python run.py evaluate      因子 IC 评估
    python run.py all           依次执行全部
"""

import os
import sys

# 确保项目根目录在 sys.path 中
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from src.data import load_config, update_all_data, get_stock_pool, load_panel_data
from src.factors import compute_all_factors, evaluate_factors
from src.signal import generate_signals
from src.backtest import backtest, calc_metrics


def _load_pipeline(config):
    """公共流程：加载数据 → 因子 → 信号"""
    pool = get_stock_pool(config)
    print(f"股票池: {len(pool)} 只")
    panel = load_panel_data(config, codes=pool["code"].tolist())
    if not panel:
        print("无数据，请先运行: python run.py update")
        sys.exit(1)
    factors, breadth = compute_all_factors(panel, config)
    signals, scores, regime = generate_signals(factors, breadth, config)
    return pool, panel, factors, breadth, signals, scores, regime


# ── 命令 ─────────────────────────────────────────────────────


def cmd_update(config):
    print("=" * 50)
    print("  数据更新")
    print("=" * 50)
    update_all_data(config)


def cmd_signal(config):
    print("=" * 50)
    print("  生成今日信号")
    print("=" * 50)
    _, panel, factors, breadth, signals, scores, regime = _load_pipeline(config)

    latest = scores.index[-1]
    print(f"\n日期: {latest.strftime('%Y-%m-%d')}")
    print(f"市场宽度: {breadth.iloc[-1]:.1%}")
    print(f"市场状态: {regime.iloc[-1]}")

    today_scores = scores.loc[latest].dropna().sort_values(ascending=False)
    top_n = config["signal"]["top_n"]
    top = today_scores.head(top_n)

    print(f"\nTop {top_n} 关注股票:")
    print("-" * 60)
    print(f"  {'#':>3}  {'代码':<8} {'评分':>8} {'相对强度':>8} {'趋势':>6} {'确认':>4}")
    print("-" * 60)
    for i, (code, score) in enumerate(top.items(), 1):
        rs = factors["relative_strength"].loc[latest].get(code, 0)
        trend = factors["trend"].loc[latest].get(code, 0)
        confirmed = "是" if signals.loc[latest].get(code, False) else "否"
        print(f"  {i:3d}  {code:<8} {score:8.3f} {rs:8.2f} {trend:6.2f} {confirmed:>4}")

    os.makedirs("output", exist_ok=True)
    import pandas as pd
    out = pd.DataFrame({
        "code": top.index,
        "score": top.values,
        "relative_strength": [factors["relative_strength"].loc[latest].get(c, 0) for c in top.index],
        "trend": [factors["trend"].loc[latest].get(c, 0) for c in top.index],
        "confirmed": [signals.loc[latest].get(c, False) for c in top.index],
    })
    out.to_csv("output/signals.csv", index=False)
    print(f"\n信号已保存到 output/signals.csv")


def cmd_backtest(config):
    print("=" * 50)
    print("  策略回测")
    print("=" * 50)
    _, panel, factors, breadth, signals, scores, regime = _load_pipeline(config)

    results = backtest(signals, scores, panel["close"], config)

    if not results["daily_returns"].empty:
        metrics = calc_metrics(results["daily_returns"])
        print("\n回测结果:")
        print("-" * 40)
        for k, v in metrics.items():
            print(f"  {k}: {v}")

        os.makedirs("output", exist_ok=True)
        results["daily_returns"].to_csv("output/backtest_result.csv")
        if not results["trade_log"].empty:
            results["trade_log"].to_csv("output/trade_log.csv", index=False)
        print("\n结果已保存到 output/")
    else:
        print("回测数据不足")


def cmd_evaluate(config):
    print("=" * 50)
    print("  因子 IC 评估")
    print("=" * 50)
    _, panel, factors, breadth, _, _, _ = _load_pipeline(config)

    ic_results = evaluate_factors(factors, panel["close"])
    print(f"\n{'因子':<20} {'IC均值':>10} {'ICIR':>10} {'IC>0':>10}")
    print("-" * 55)
    for name, r in ic_results.items():
        print(f"  {name:<18} {r['ic_mean']:+10.4f} {r['icir']:+10.4f} {r['ic_positive_pct']:10.1%}")


# ── main ─────────────────────────────────────────────────────


def main():
    config = load_config()

    if len(sys.argv) < 2:
        print(__doc__)
        return

    cmd = sys.argv[1]
    dispatch = {
        "update": cmd_update,
        "signal": cmd_signal,
        "backtest": cmd_backtest,
        "evaluate": cmd_evaluate,
    }

    if cmd == "all":
        for fn in [cmd_update, cmd_signal, cmd_evaluate, cmd_backtest]:
            fn(config)
            print()
    elif cmd in dispatch:
        dispatch[cmd](config)
    else:
        print(f"未知命令: {cmd}")
        print(__doc__)


if __name__ == "__main__":
    main()
