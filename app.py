"""
A股量化辅助决策仪表盘
启动: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from src.data import load_config, load_panel_data, load_index_data, get_stock_pool
from src.factors import compute_all_factors, evaluate_factors
from src.signal import generate_signals
from src.backtest import backtest, calc_metrics

st.set_page_config(page_title="A股量化辅助决策", layout="wide")
st.title("A股量化辅助决策仪表盘")


# ── 数据加载（带缓存）──────────────────────────────────────


@st.cache_data(ttl=3600)
def load_all():
    config = load_config()
    pool = get_stock_pool(config)
    panel = load_panel_data(config, codes=pool["code"].tolist())
    index_data = load_index_data(config)
    return config, pool, panel, index_data


try:
    config, pool, panel, index_data = load_all()
except Exception as e:
    st.error(f"数据加载失败: {e}")
    st.info("请先运行 `python run.py update` 获取数据")
    st.stop()

if not panel:
    st.warning("无数据，请先运行 `python run.py update`")
    st.stop()

factors, breadth = compute_all_factors(panel, config)
signals, scores, regime = generate_signals(factors, breadth, config)


# ── 页签 ─────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs(["今日信号", "回测分析", "因子检验", "市场情绪"])


# ═══════ Tab 1: 今日信号 ═══════

with tab1:
    latest = scores.index[-1]
    st.subheader(f"{latest.strftime('%Y-%m-%d')} 关注池")

    # 市场状态
    c1, c2 = st.columns(2)
    regime_map = {"normal": "正常", "caution": "谨慎", "bear": "回避"}
    c1.metric("市场状态", regime_map.get(regime.iloc[-1], regime.iloc[-1]))
    c2.metric("市场宽度", f"{breadth.iloc[-1]:.1%}")

    # Top N
    today_scores = scores.loc[latest].dropna().sort_values(ascending=False)
    top = today_scores.head(config["signal"]["top_n"])

    rows = []
    for code in top.index:
        rows.append({
            "代码": code,
            "综合评分": f"{top[code]:.3f}",
            "相对强度": f'{factors["relative_strength"].loc[latest].get(code, np.nan):.2f}',
            "趋势得分": f'{factors["trend"].loc[latest].get(code, np.nan):.2f}',
            "信号确认": "是" if signals.loc[latest].get(code, False) else "否",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # 单只股票 K 线
    st.subheader("个股走势")
    selected = st.selectbox("选择股票", top.index.tolist())
    if selected and selected in panel["close"].columns:
        price = panel["close"][selected].dropna().tail(120)
        fig = go.Figure(go.Scatter(x=price.index, y=price.values, mode="lines", name=selected))
        for w in config["factors"]["ma_windows"]:
            ma = price.rolling(w).mean()
            fig.add_trace(go.Scatter(x=ma.index, y=ma.values, mode="lines",
                                     name=f"MA{w}", line=dict(dash="dash")))
        fig.update_layout(title=f"{selected} 近 120 日走势", height=400,
                          xaxis_title="日期", yaxis_title="价格（后复权）")
        st.plotly_chart(fig, use_container_width=True)


# ═══════ Tab 2: 回测分析 ═══════

with tab2:
    st.subheader("策略回测")

    results = backtest(signals, scores, panel["close"], config)

    if not results["daily_returns"].empty:
        metrics = calc_metrics(results["daily_returns"])

        # 指标卡片
        cols = st.columns(4)
        items = list(metrics.items())
        for i, (k, v) in enumerate(items[:4]):
            cols[i].metric(k, v)
        if len(items) > 4:
            cols2 = st.columns(4)
            for i, (k, v) in enumerate(items[4:8]):
                cols2[i].metric(k, v)

        # 净值曲线
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=results["daily_returns"].index,
            y=results["daily_returns"]["value"],
            mode="lines", name="策略净值",
        ))
        fig.update_layout(title="策略净值曲线", height=400,
                          xaxis_title="日期", yaxis_title="净值")
        st.plotly_chart(fig, use_container_width=True)

        # 回撤曲线
        val = results["daily_returns"]["value"]
        dd = (val - val.cummax()) / val.cummax()
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=dd.index, y=dd.values,
            fill="tozeroy", fillcolor="rgba(255,0,0,0.1)",
            line=dict(color="red"), name="回撤",
        ))
        fig2.update_layout(title="回撤曲线", height=300,
                           xaxis_title="日期", yaxis_title="回撤")
        st.plotly_chart(fig2, use_container_width=True)

        # 交易日志
        if not results["trade_log"].empty:
            with st.expander("交易日志（最近 50 条）"):
                st.dataframe(results["trade_log"].tail(50), use_container_width=True, hide_index=True)
    else:
        st.warning("回测数据不足")


# ═══════ Tab 3: 因子检验 ═══════

with tab3:
    st.subheader("因子 Rank IC 分析")

    ic_results = evaluate_factors(factors, panel["close"])

    if ic_results:
        summary = pd.DataFrame({
            name: {
                "IC 均值": f'{r["ic_mean"]:.4f}',
                "IC 标准差": f'{r["ic_std"]:.4f}',
                "ICIR": f'{r["icir"]:.4f}',
                "IC>0 占比": f'{r["ic_positive_pct"]:.1%}',
            }
            for name, r in ic_results.items()
        }).T
        st.dataframe(summary, use_container_width=True)

        sel = st.selectbox("选择因子查看 IC 时序", list(ic_results.keys()))
        if sel:
            ic_s = ic_results[sel]["ic_series"]
            fig = go.Figure(go.Bar(x=ic_s.index, y=ic_s.values, name="IC"))
            fig.add_hline(y=ic_s.mean(), line_dash="dash", line_color="red",
                          annotation_text=f"均值: {ic_s.mean():.4f}")
            fig.update_layout(title=f"{sel} - IC 时序", height=400,
                              xaxis_title="日期", yaxis_title="IC")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("因子数据不足，无法评估")


# ═══════ Tab 4: 市场情绪 ═══════

with tab4:
    st.subheader("市场宽度指标")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=breadth.index, y=breadth.values, mode="lines", name="市场宽度"))
    fig.add_hline(y=config["market"]["breadth_threshold_high"],
                  line_dash="dash", line_color="green", annotation_text="多头阈值")
    fig.add_hline(y=config["market"]["breadth_threshold_low"],
                  line_dash="dash", line_color="red", annotation_text="空头阈值")
    fig.update_layout(title="市场宽度（站上 20 日均线股票占比）", height=400,
                      xaxis_title="日期", yaxis_title="占比")
    st.plotly_chart(fig, use_container_width=True)

    # 指数 K 线
    if not index_data.empty:
        fig2 = go.Figure(go.Candlestick(
            x=index_data.index,
            open=index_data["open"], high=index_data["high"],
            low=index_data["low"], close=index_data["close"],
            name=config["market"]["index_code"],
        ))
        fig2.update_layout(
            title=f'指数走势 ({config["market"]["index_code"]})',
            height=450, xaxis_rangeslider_visible=False,
        )
        st.plotly_chart(fig2, use_container_width=True)
