"""
涨跌方向预测 - 二分类模块
========================
独立于回归预测管线，使用 GARCH(1,1) 波动率作为核心特征，
XGBoost + ElasticNet LogisticRegression 二分类器，
严格扩展窗口时间序列验证，智能参数推荐。

防泄漏保证:
1. 仅使用扩展窗口 (expanding window) 划分，禁止随机 shuffle / k-fold
2. GARCH(1,1) 条件波动率仅在各折训练集上拟合，验证集用前向预测
3. 所有量价衍生特征（成交量变化率、相对成交量、量价配合度、放量上涨、缩量下跌）保留
4. create_clf_features 对时刻 i 仅使用 [i-look_back, i-1] 的信息
5. 目标变量 目标涨跌 = 涨跌标签.shift(-1)，预测的是下一日涨跌
"""

import time
import warnings
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Callable

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr

from .features import compute_technical_indicators, time_series_split
from .preprocessing import preprocess_data
from .volatility import fit_garch

warnings.filterwarnings("ignore")

# ── 分类特征列（排除目标涨跌、目标收益率，保留所有量价衍生特征） ──

CLF_FEATURE_COLS = [
    # 价格变化
    "pct_change", "日收益率",
    # 多周期动量
    "ret_2d", "ret_3d", "ret_5d", "ret_10d",
    # 均线偏离度
    "close_ma5_bias", "close_ma10_bias", "close_ma20_bias", "ma5_ma10_cross",
    # 波动率
    "volatility_5d", "volatility_10d", "volatility_20d",
    # 技术指标
    "rsi", "rsi_6", "rsi_14", "dif", "macd",
    # K线形态
    "upper_shadow", "lower_shadow", "body_ratio", "high_low_range",
    # 量价配合
    "成交量变化率", "相对成交量", "量价配合度",
    "放量上涨", "缩量下跌",
    # 成交量多周期
    "vol_chg_3d", "vol_chg_5d", "vol_ratio_5d",
    # ── 科技股专用 ──
    # 换手率系列
    "turnover", "turnover_ma5", "turnover_ma10",
    "turnover_bias", "turnover_accel",
    "cum_turnover_5d", "cum_turnover_10d",
    # 跳空缺口
    "gap", "gap_abs", "gap_up", "gap_down",
    # 价格位置
    "price_pos_5d", "price_pos_10d", "price_pos_20d",
    "drawdown_5d", "drawdown_10d", "drawdown_20d",
    # 连涨连跌
    "streak_up", "streak_down",
    # ATR
    "atr_5", "atr_14", "atr_ratio",
    # 涨停板效应
    "near_limit_up", "near_limit_down",
    "limit_up_count_10d", "limit_down_count_10d",
    # 相对强弱（需指数数据，可选）
    "excess_ret", "excess_ret_5d", "excess_ret_10d", "excess_ret_20d",
]


# ── 数据容器 ──

@dataclass
class ClfResult:
    """单分类器训练结果（所有字段对应 OOS 样本）"""
    model_name: str
    model_object: object = None
    # OOS 预测与标签（按折拼接，时间顺序）
    oos_probabilities: np.ndarray = field(default_factory=lambda: np.array([]))
    oos_predictions: np.ndarray = field(default_factory=lambda: np.array([]))
    oos_actuals: np.ndarray = field(default_factory=lambda: np.array([]))
    oos_returns: np.ndarray = field(default_factory=lambda: np.array([]))
    oos_future_ret: np.ndarray = field(default_factory=lambda: np.array([]))
    oos_next_day_ret: np.ndarray = field(default_factory=lambda: np.array([]))
    oos_dates: np.ndarray = field(default_factory=lambda: np.array([]))
    # CV 指标
    fold_metrics: dict = field(default_factory=dict)
    overall_metrics: dict = field(default_factory=dict)
    # 特征重要性（XGBoost 多折平均）
    feature_importance: dict = field(default_factory=dict)
    training_time: float = 0.0
    feature_names: list = field(default_factory=list)
    # 最新交易日实时预测（目标涨跌为 NaN 的最后一行）
    latest_proba: float = np.nan
    latest_date: object = None


# ── 特征工程 ──

def create_clf_features(
    df: pd.DataFrame,
    feature_cols: List[str],
    look_back: int,
) -> Tuple[np.ndarray, np.ndarray, List[str], pd.DatetimeIndex,
           np.ndarray, pd.DatetimeIndex]:
    """
    构建分类滞后特征（严格防泄漏）。

    对时刻 i，仅使用 [i-look_back, i-1] 的数据构造特征，
    目标 y[i] = df.iloc[i]['目标涨跌'] 预测的是 i→i+1 的涨跌方向。

    最后一天（目标涨跌为 NaN）的特征也会构建，但 y 中对应位置为 NaN，
    单独返回为 X_latest / latest_date，供实时预测使用。

    返回: (X, y, feature_names, dates, X_latest, latest_date)
      X: (n_labeled, n_features * look_back) 有标签样本特征
      y: (n_labeled,) 0/1 标签（仅非 NaN 行）
      feature_names: "col_lagN" 格式的特征名
      dates: 有标签样本的对齐日期
      X_latest: (1, n_features * look_back) 最新日特征（可能为空）
      latest_date: 最新日日期（可能为 None）
    """
    if "目标涨跌" not in df.columns:
        raise ValueError("df 中缺少 '目标涨跌' 列，请先调用 preprocess_data()")

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"df 中缺少以下特征列: {missing}")

    # 只保留需要的列 + 目标
    cols = feature_cols + ["目标涨跌"]
    df_sub = df[cols].copy()
    # 仅删除特征列有 NaN 的行，保留目标涨跌为 NaN 的最后一天
    df_sub = df_sub.dropna(subset=feature_cols)

    if len(df_sub) <= look_back:
        raise ValueError(f"有效数据量 {len(df_sub)} <= look_back {look_back}，无法构造特征")

    n = len(df_sub)
    X_list = []
    for i in range(look_back, n):
        # 仅使用 i-look_back 到 i-1 的数据，不含 i（防泄漏要求 4）
        row = df_sub[feature_cols].iloc[i - look_back:i].values.flatten()
        X_list.append(row)

    X_all = np.array(X_list, dtype=np.float64)
    y_all = df_sub["目标涨跌"].values[look_back:]
    dates_all = df_sub.index[look_back:]

    # 分离有标签和无标签样本
    labeled_mask = ~np.isnan(y_all)
    X = X_all[labeled_mask]
    y = y_all[labeled_mask].astype(int)
    dates = dates_all[labeled_mask]

    # 最新一天（目标涨跌为 NaN）
    X_latest = np.array([], dtype=np.float64)
    latest_date = None
    unlabeled_mask = ~labeled_mask
    if unlabeled_mask.any():
        X_latest = X_all[unlabeled_mask][-1:]
        latest_date = dates_all[unlabeled_mask][-1]

    # 构建特征名
    feature_names = []
    for col in feature_cols:
        for lag in range(look_back, 0, -1):
            feature_names.append(f"{col}_lag{lag}")

    return X, y, feature_names, dates, X_latest, latest_date


# ── 模型构建 ──

def build_xgb_classifier(params: dict):
    """XGBoost 二分类器"""
    import xgboost as xgb
    return xgb.XGBClassifier(
        n_estimators=params.get("n_estimators", 100),
        max_depth=params.get("max_depth", 6),
        learning_rate=params.get("learning_rate", 0.1),
        subsample=params.get("subsample", 0.8),
        colsample_bytree=params.get("colsample_bytree", 0.8),
        min_child_weight=params.get("min_child_weight", 1),
        reg_alpha=params.get("reg_alpha", 0.0),
        reg_lambda=params.get("reg_lambda", 1.0),
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )


def build_elasticnet(params: dict):
    """ElasticNet LogisticRegression"""
    return LogisticRegression(
        C=params.get("C", 1.0),
        l1_ratio=params.get("l1_ratio", 0.15),
        penalty="elasticnet",
        solver="saga",
        max_iter=params.get("max_iter", 5000),
        tol=params.get("tol", 1e-3),
        random_state=42,
        n_jobs=-1,
    )


# ── GARCH 波动率特征（防泄漏） ──

def _expanding_hist_vol(returns: np.ndarray, min_window: int = 20) -> np.ndarray:
    """扩展窗口历史波动率（GARCH 失败时的回退方案）。
    每一点仅使用该点之前的数据。"""
    vol = np.zeros(len(returns))
    for i in range(len(returns)):
        w = max(i + 1, min_window)
        vol[i] = np.std(returns[max(0, i - w + 1):i + 1])
    return vol


def _extract_garch_vol_from_model(garch_result) -> np.ndarray:
    """从已拟合的 GARCH 模型提取条件波动率序列。
    conditional_volatility[t] 是 GARCH 模型对时刻 t 的预测波动率，
    基于 t-1 之前的信息——由 arch 库保证。"""
    cond_vol = np.asarray(garch_result.conditional_volatility, dtype=np.float64)
    # 处理 NaN（GARCH 初始化阶段可能产生）
    mask = np.isnan(cond_vol)
    if mask.any():
        first_valid = cond_vol[~mask][0] if (~mask).any() else 0.01
        cond_vol[mask] = first_valid
    return cond_vol


def _forecast_garch_vol(garch_result, horizon: int) -> np.ndarray:
    """对未来 horizon 期前向预测 GARCH 波动率（仅使用训练集信息）。"""
    try:
        forecast = garch_result.forecast(horizon=horizon)
        var_forecast = forecast.variance.iloc[-1].values
        vol = np.sqrt(np.maximum(var_forecast, 0))
        vol = np.nan_to_num(vol, nan=0.01)
        return vol
    except Exception:
        return np.full(horizon, 0.01)


# ── 评估指标 ──

def calculate_classification_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    returns: np.ndarray,
    predictions: np.ndarray,
    future_ret: np.ndarray = None,
    forecast_days: int = 1,
    next_day_ret: np.ndarray = None,
) -> dict:
    """
    计算分类及策略指标。

    策略逻辑: 预测涨(prob>=threshold)则T日收盘买入、T+1日收盘卖出，
    预测跌则空仓(收益=0)。回测收益 = next_day_ret * predictions（下一日收益）。
    若 next_day_ret 为 None，回退为 returns。

    返回:
      auc, ic, ic_pvalue,
      cum_return, ann_return, ann_volatility, sharpe, max_dd, win_rate,
      profit_loss_ratio, accuracy, precision, recall, confusion
    """
    metrics = {}

    # AUC
    try:
        unique = np.unique(y_true)
        if len(unique) >= 2:
            metrics["auc"] = float(roc_auc_score(y_true, y_pred_proba))
        else:
            metrics["auc"] = np.nan
    except ValueError:
        metrics["auc"] = np.nan

    # IC: 预测概率与下一日实际收益的相关性
    pnl = next_day_ret if next_day_ret is not None else returns
    mask = ~(np.isnan(y_pred_proba) | np.isnan(pnl))
    if mask.sum() >= 10:
        ic, ic_pv = pearsonr(y_pred_proba[mask], pnl[mask])
        metrics["ic"] = float(ic)
        metrics["ic_pvalue"] = float(ic_pv)
    else:
        metrics["ic"] = np.nan
        metrics["ic_pvalue"] = np.nan

    # 策略收益: 预测涨做多(future_ret)，预测跌空仓(收益=0)
    n = len(predictions)
    # 过滤 future_ret 为 NaN 的行（持有期超过数据末尾时无法验证）
    pnl_slice = pnl[:n]
    valid = ~np.isnan(pnl_slice)
    n_valid = valid.sum()
    strategy_returns = pnl_slice[valid] * predictions[valid]
    if n_valid == 0:
        strategy_returns = np.array([0.0])
    cum_series = np.cumprod(1 + strategy_returns)
    cum_return = cum_series[-1] - 1
    metrics["cum_return"] = float(cum_return)

    # 年化收益（252 交易日，按有效样本数）
    if cum_return > -1 and n_valid > 0:
        metrics["ann_return"] = float((1 + cum_return) ** (252 / n_valid) - 1)
    else:
        metrics["ann_return"] = -1.0

    # 年化波动率
    if np.std(strategy_returns) > 0 and n_valid > 0:
        metrics["ann_volatility"] = float(np.std(strategy_returns) * np.sqrt(252))
    else:
        metrics["ann_volatility"] = 0.0

    # Sharpe（无风险利率 = 0）
    if np.std(strategy_returns) > 0 and n_valid > 0:
        metrics["sharpe"] = float(np.sqrt(252) * np.mean(strategy_returns) / np.std(strategy_returns))
    else:
        metrics["sharpe"] = 0.0

    # 最大回撤
    running_max = np.maximum.accumulate(cum_series)
    drawdowns = (cum_series - running_max) / running_max
    metrics["max_dd"] = float(drawdowns.min())

    # 胜率（做多时的正确率，按N日持有期收益方向，仅有效行）
    long_mask = predictions[valid] == 1
    if long_mask.sum() > 0:
        metrics["win_rate"] = float((pnl_slice[valid][long_mask] > 0).mean())
    else:
        metrics["win_rate"] = 0.0

    # 盈亏比: 平均正收益 / 平均负收益的绝对值
    pos_ret = strategy_returns[strategy_returns > 0]
    neg_ret = strategy_returns[strategy_returns < 0]
    if len(neg_ret) > 0 and np.abs(neg_ret.mean()) > 1e-12:
        metrics["profit_loss_ratio"] = float(pos_ret.mean() / np.abs(neg_ret.mean())) if len(pos_ret) > 0 else np.inf
    else:
        metrics["profit_loss_ratio"] = np.inf if len(pos_ret) > 0 else 0.0

    # 混淆矩阵
    tp = int(((predictions == 1) & (y_true[:n] == 1)).sum())
    tn = int(((predictions == 0) & (y_true[:n] == 0)).sum())
    fp = int(((predictions == 1) & (y_true[:n] == 0)).sum())
    fn = int(((predictions == 0) & (y_true[:n] == 1)).sum())
    total = tp + tn + fp + fn
    metrics["confusion"] = {"tp": tp, "tn": tn, "fp": fp, "fn": fn}
    metrics["accuracy"] = (tp + tn) / total if total > 0 else 0.0
    metrics["precision"] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    metrics["recall"] = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return metrics


# ── 扩展窗口训练引擎 ──

def run_expanding_window(
    X: np.ndarray,
    y: np.ndarray,
    returns: np.ndarray,
    dates: np.ndarray,
    feature_names: List[str],
    models: List[str],
    params: Dict[str, dict],
    n_splits: int = 5,
    min_train_size: int = 100,
    progress_cb: Optional[Callable] = None,
    future_ret: np.ndarray = None,
    forecast_days: int = 1,
    next_day_ret: np.ndarray = None,
    X_latest: np.ndarray = None,
    latest_date: object = None,
    all_returns: np.ndarray = None,
) -> Dict[str, ClfResult]:
    """
    严格扩展窗口时间序列训练 + 验证（禁止随机 shuffle / k-fold）。

    每折:
      - 训练集 = [:train_end]（含所有历史数据）
      - 验证集 = [train_end:train_end+fold_size]（紧接训练集之后）
      - GARCH(1,1) 仅在训练集上拟合 → training cond_vol（已由 arch 库保证无前视）
      - 验证集 GARCH vol = 前向预测（仅使用训练集参数）

    X, y, returns, dates: 已对齐的完整数据（由 create_clf_features 产出）
    models: ["XGBoost", "ElasticNet"] 子集
    params: {"XGBoost": {...}, "ElasticNet": {...}}
    future_ret: N日持有期收益（用于回测P&L），None则回退为returns
    forecast_days: 持有天数

    返回: {model_name: ClfResult}
    """
    results = {}

    for model_name in models:
        t0 = time.time()

        result = ClfResult(model_name=model_name)
        result.feature_names = feature_names + ["garch_vol"]

        # 仅使用扩展窗口划分（要求 2+3）
        splits = time_series_split(len(X), n_splits=n_splits, min_train_size=min_train_size)
        if len(splits) == 0:
            split_point = int(len(X) * 0.8)
            splits = [(np.arange(split_point), np.arange(split_point, len(X)))]

        all_oos_probs = []
        all_oos_preds = []
        all_oos_actuals = []
        all_oos_returns = []
        all_oos_dates = []
        all_oos_future_ret = []
        all_oos_next_day_ret = []
        fold_metrics_list = []
        fold_importances = []
        fold_count = len(splits)
        last_fold_scaler = None  # ElasticNet 最后一折的 scaler
        last_fold_garch_model = None  # 最后一折的 GARCH 模型
        last_fold_train_returns = None

        for fold_i, (train_idx, val_idx) in enumerate(splits):
            if progress_cb:
                pct = fold_i / fold_count
                msg = f"[{model_name}] 折 {fold_i + 1}/{fold_count}"
                progress_cb(pct, msg)

            X_train = X[train_idx]
            y_train = y[train_idx]
            X_val = X[val_idx]
            y_val = y[val_idx]
            val_dates = dates[val_idx]
            train_returns = returns[train_idx]
            val_returns = returns[val_idx]
            val_future_ret = future_ret[val_idx] if future_ret is not None else val_returns
            val_next_day_ret = next_day_ret[val_idx] if next_day_ret is not None else val_returns

            # ── A. 仅在训练集上拟合 GARCH（要求 4: 防未来信息泄漏） ──
            garch_model = fit_garch(train_returns, p=1, q=1, dist='t')

            if garch_model is not None:
                garch_vol_train = _extract_garch_vol_from_model(garch_model)
                garch_vol_val = _forecast_garch_vol(garch_model, len(val_idx))
            else:
                # GARCH 不收敛，回退为扩展历史波动率（仍仅用训练集）
                garch_vol_train = _expanding_hist_vol(train_returns, min_window=20)
                garch_vol_val = np.full(len(val_idx), np.std(train_returns[-60:]) if len(train_returns) >= 60 else np.std(train_returns))

            # 长度对齐（GARCH 可能少返回首元素）
            if len(garch_vol_train) < len(X_train):
                pad_len = len(X_train) - len(garch_vol_train)
                garch_vol_train = np.concatenate([np.full(pad_len, garch_vol_train[0]), garch_vol_train])
            elif len(garch_vol_train) > len(X_train):
                garch_vol_train = garch_vol_train[-len(X_train):]

            garch_vol_train = np.nan_to_num(garch_vol_train, nan=0.01)
            garch_vol_val = np.nan_to_num(garch_vol_val, nan=0.01)

            X_train_aug = np.column_stack([X_train, garch_vol_train.reshape(-1, 1)])
            X_val_aug = np.column_stack([X_val, garch_vol_val.reshape(-1, 1)])

            # ── B. 训练 + 预测 ──
            if model_name == "XGBoost":
                clf = build_xgb_classifier(params.get(model_name, {}))
                clf.fit(X_train_aug, y_train)
                proba = clf.predict_proba(X_val_aug)[:, 1]
                # 收集每折特征重要性（后续求平均）
                fold_importances.append(dict(zip(
                    feature_names + ["garch_vol"],
                    clf.feature_importances_,
                )))
            elif model_name == "ElasticNet":
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train_aug)
                X_val_scaled = scaler.transform(X_val_aug)
                clf = build_elasticnet(params.get(model_name, {}))
                clf.fit(X_train_scaled, y_train)
                proba = clf.predict_proba(X_val_scaled)[:, 1]
            else:
                raise ValueError(f"不支持的模型: {model_name}")

            preds = (proba >= 0.5).astype(int)

            all_oos_probs.append(proba)
            all_oos_preds.append(preds)
            all_oos_actuals.append(y_val)
            all_oos_returns.append(val_returns)
            all_oos_future_ret.append(val_future_ret)
            all_oos_next_day_ret.append(val_next_day_ret)
            all_oos_dates.append(val_dates)

            # 折级指标（策略P&L用 next_day_ret）
            fold_metrics = calculate_classification_metrics(
                y_val, proba, val_returns, preds,
                future_ret=val_future_ret, forecast_days=forecast_days,
                next_day_ret=val_next_day_ret,
            )
            fold_metrics_list.append(fold_metrics)

            result.model_object = clf
            if model_name == "ElasticNet":
                last_fold_scaler = scaler
            last_fold_garch_model = garch_model
            last_fold_train_returns = train_returns

        # ── 聚合 OOS（按时序拼接，非末 N 个） ──
        result.oos_probabilities = np.concatenate(all_oos_probs)
        result.oos_predictions = np.concatenate(all_oos_preds).astype(int)
        result.oos_actuals = np.concatenate(all_oos_actuals).astype(int)
        result.oos_returns = np.concatenate(all_oos_returns)
        result.oos_future_ret = np.concatenate(all_oos_future_ret)
        result.oos_next_day_ret = np.concatenate(all_oos_next_day_ret)
        result.oos_dates = np.concatenate(all_oos_dates)

        # ── 最新交易日实时预测 ──
        if X_latest is not None and len(X_latest) > 0 and result.model_object is not None:
            try:
                # 用最后一折的训练集收益率构造 GARCH vol
                _ret_for_garch = all_returns if all_returns is not None else last_fold_train_returns
                if _ret_for_garch is not None:
                    gm = fit_garch(_ret_for_garch, p=1, q=1, dist='t')
                    if gm is not None:
                        garch_vol_latest = _forecast_garch_vol(gm, 1)
                    else:
                        vol_std = np.std(_ret_for_garch[-60:]) if len(_ret_for_garch) >= 60 else np.std(_ret_for_garch)
                        garch_vol_latest = np.array([vol_std])
                else:
                    garch_vol_latest = np.array([0.01])
                garch_vol_latest = np.nan_to_num(garch_vol_latest, nan=0.01)

                X_lat_aug = np.column_stack([X_latest, garch_vol_latest.reshape(-1, 1)])

                if model_name == "ElasticNet" and last_fold_scaler is not None:
                    X_lat_aug = last_fold_scaler.transform(X_lat_aug)

                latest_proba = float(result.model_object.predict_proba(X_lat_aug)[0, 1])
                result.latest_proba = latest_proba
                result.latest_date = latest_date
            except Exception:
                pass

        # 特征重要性：多折平均（XGBoost）
        if fold_importances:
            avg_imp = {}
            for key in fold_importances[0]:
                avg_imp[key] = float(np.mean([fi.get(key, 0) for fi in fold_importances]))
            result.feature_importance = avg_imp

        # 折级指标平均
        if fold_metrics_list:
            avg_metrics = {}
            for key in fold_metrics_list[0]:
                vals = [m[key] for m in fold_metrics_list
                        if not (isinstance(m[key], float) and np.isnan(m[key]))]
                if key == "confusion":
                    avg_metrics[key] = {
                        k: int(np.mean([m["confusion"][k] for m in fold_metrics_list]))
                        for k in fold_metrics_list[0]["confusion"]
                    }
                elif vals:
                    avg_metrics[key] = float(np.mean(vals))
                else:
                    avg_metrics[key] = np.nan
            result.fold_metrics = {
                "fold_scores": fold_metrics_list,
                "cv_avg": avg_metrics,
            }

        # 全量 OOS 指标（策略P&L用 next_day_ret）
        result.overall_metrics = calculate_classification_metrics(
            result.oos_actuals, result.oos_probabilities,
            result.oos_returns, result.oos_predictions,
            future_ret=result.oos_future_ret, forecast_days=forecast_days,
            next_day_ret=result.oos_next_day_ret,
        )

        result.training_time = time.time() - t0
        results[model_name] = result

        if progress_cb:
            progress_cb(1.0, f"[{model_name}] 完成 ({result.training_time:.1f}s)")

    return results


# ── 智能参数推荐 ──

def get_recommended_params(n_samples: int, n_features: int = 30) -> dict:
    """根据有效样本量返回推荐参数（针对科技股优化）。

    科技股特点：波动大、换手高、趋势短 → 短回溯 + 浅树 + 强正则。

    7 档:
      tiny       < 200
      small      200-500
      medium     500-800
      med_large  800-1200
      large      1200-2000
      xlarge     2000-3000
      xxlarge    > 3000
    """
    if n_samples < 200:
        return {
            "mode": "tiny",
            "xgb": {
                "n_estimators": 60, "max_depth": 2, "learning_rate": 0.08,
                "subsample": 0.75, "colsample_bytree": 0.75,
                "min_child_weight": 8, "reg_alpha": 0.5, "reg_lambda": 3.0,
            },
            "elasticnet": {
                "C": 0.5, "l1_ratio": 0.15, "max_iter": 5000, "tol": 1e-3,
            },
            "look_back": 5,
            "n_splits": 3,
        }
    elif n_samples < 500:
        return {
            "mode": "small",
            "xgb": {
                "n_estimators": 80, "max_depth": 2, "learning_rate": 0.06,
                "subsample": 0.75, "colsample_bytree": 0.75,
                "min_child_weight": 8, "reg_alpha": 0.4, "reg_lambda": 3.0,
            },
            "elasticnet": {
                "C": 0.3, "l1_ratio": 0.12, "max_iter": 5000, "tol": 1e-3,
            },
            "look_back": 6,
            "n_splits": 4,
        }
    elif n_samples < 800:
        return {
            "mode": "medium",
            "xgb": {
                "n_estimators": 100, "max_depth": 2, "learning_rate": 0.05,
                "subsample": 0.75, "colsample_bytree": 0.75,
                "min_child_weight": 8, "reg_alpha": 0.4, "reg_lambda": 3.0,
            },
            "elasticnet": {
                "C": 0.2, "l1_ratio": 0.10, "max_iter": 5000, "tol": 1e-4,
            },
            "look_back": 8,
            "n_splits": 5,
        }
    elif n_samples < 1200:
        return {
            "mode": "med_large",
            "xgb": {
                "n_estimators": 120, "max_depth": 2, "learning_rate": 0.05,
                "subsample": 0.75, "colsample_bytree": 0.75,
                "min_child_weight": 8, "reg_alpha": 0.4, "reg_lambda": 3.0,
            },
            "elasticnet": {
                "C": 0.15, "l1_ratio": 0.10, "max_iter": 5000, "tol": 1e-4,
            },
            "look_back": 8,
            "n_splits": 6,
        }
    elif n_samples < 2000:
        return {
            "mode": "large",
            "xgb": {
                "n_estimators": 150, "max_depth": 3, "learning_rate": 0.04,
                "subsample": 0.75, "colsample_bytree": 0.70,
                "min_child_weight": 10, "reg_alpha": 0.4, "reg_lambda": 3.0,
            },
            "elasticnet": {
                "C": 0.10, "l1_ratio": 0.08, "max_iter": 8000, "tol": 1e-4,
            },
            "look_back": 8,
            "n_splits": 7,
        }
    elif n_samples < 3000:
        return {
            "mode": "xlarge",
            "xgb": {
                "n_estimators": 180, "max_depth": 3, "learning_rate": 0.03,
                "subsample": 0.70, "colsample_bytree": 0.65,
                "min_child_weight": 12, "reg_alpha": 0.5, "reg_lambda": 3.5,
            },
            "elasticnet": {
                "C": 0.08, "l1_ratio": 0.06, "max_iter": 8000, "tol": 1e-4,
            },
            "look_back": 10,
            "n_splits": 8,
        }
    else:
        return {
            "mode": "xxlarge",
            "xgb": {
                "n_estimators": 200, "max_depth": 3, "learning_rate": 0.02,
                "subsample": 0.70, "colsample_bytree": 0.60,
                "min_child_weight": 12, "reg_alpha": 0.5, "reg_lambda": 4.0,
            },
            "elasticnet": {
                "C": 0.05, "l1_ratio": 0.05, "max_iter": 10000, "tol": 1e-4,
            },
            "look_back": 10,
            "n_splits": 8,
        }


def check_params_deviation(current: Dict[str, dict], recommended: dict) -> List[str]:
    """对比当前参数与推荐参数，返回警告列表。"""
    warnings_list = []

    xgb_cur = current.get("XGBoost", {})
    xgb_rec = recommended.get("xgb", {})
    for param, rec_val in xgb_rec.items():
        cur_val = xgb_cur.get(param)
        if cur_val is not None and cur_val != rec_val:
            warnings_list.append(
                f"XGBoost.{param}: 当前={cur_val}, 推荐={rec_val}"
            )

    en_cur = current.get("ElasticNet", {})
    en_rec = recommended.get("elasticnet", {})
    for param, rec_val in en_rec.items():
        cur_val = en_cur.get(param)
        if cur_val is not None and cur_val != rec_val:
            warnings_list.append(
                f"ElasticNet.{param}: 当前={cur_val}, 推荐={rec_val}"
            )

    return warnings_list


# ── 动态融合权重 ──

def _compute_ensemble_weights(result_a: 'ClfResult', result_b: 'ClfResult') -> dict:
    """基于 AUC(40%) + Sharpe(30%) + 胜率(30%) 综合评分计算动态权重"""

    def _score(r):
        m = r.overall_metrics
        auc = m.get("auc", 0.5)
        sharpe = m.get("sharpe", 0.0)
        win_rate = m.get("win_rate", 0.5)
        if np.isnan(auc) or auc < 0.5:
            auc = 0.5
        if np.isnan(sharpe) or sharpe < 0:
            sharpe = 0.0
        if np.isnan(win_rate) or win_rate < 0:
            win_rate = 0.0
        return 0.4 * auc + 0.3 * sharpe + 0.3 * win_rate

    score_a = _score(result_a)
    score_b = _score(result_b)
    total = score_a + score_b

    if total <= 0:
        w_a, w_b = 0.5, 0.5
    else:
        w_a = score_a / total
        w_b = score_b / total

    return {result_a.model_name: w_a, result_b.model_name: w_b}


def _load_index_returns(stock_code: str, start_date, end_date) -> Optional[pd.Series]:
    """
    根据股票代码自动选择对应板块指数，返回日收益率 Series。
    科技股(创业板300xxx/科创板688xxx) → 创业板指(399006)
    主板 → 沪深300(000300)
    加载失败返回 None（不影响训练）
    """
    if not stock_code:
        return None
    try:
        import akshare as ak
        if stock_code.startswith("3"):
            index_code = "399006"
        elif stock_code.startswith("688"):
            index_code = "399006"
        else:
            index_code = "000300"

        sd = pd.Timestamp(start_date).strftime("%Y%m%d")
        ed = pd.Timestamp(end_date).strftime("%Y%m%d")
        idx_df = ak.index_zh_a_hist(symbol=index_code, period="daily",
                                     start_date=sd, end_date=ed)
        if idx_df is None or idx_df.empty:
            return None
        idx_df["日期"] = pd.to_datetime(idx_df["日期"])
        idx_df = idx_df.set_index("日期").sort_index()
        idx_ret = idx_df["涨跌幅"] / 100.0
        idx_ret.index.name = None
        return idx_ret
    except Exception:
        return None


# ── 完整管线 ──

def run_classifier_pipeline(
    df: pd.DataFrame,
    selected_models: List[str],
    params: Dict[str, dict],
    look_back: int = 20,
    n_splits: int = 5,
    progress_cb: Optional[Callable] = None,
    forecast_days: int = 1,
    threshold: float = 0.5,
    stock_code: str = None,
):
    """
    分类器完整管线。

    1. preprocess_data → 日收益率 / future_ret / 目标涨跌 / 量价衍生特征
    2. compute_technical_indicators → MA / MACD / RSI / 布林带 / OBV 等
    3. create_clf_features → 滞后特征展平（严格防泄漏）
    4. run_expanding_window → 扩展窗口训练+验证 + GARCH 注入
    5. 概率融合 → final_proba = (proba_a + proba_b) / 2, signal = final_proba >= threshold

    df: 原始 OHLCV DataFrame（index 为日期）
    selected_models: ["XGBoost", "ElasticNet"] 子集
    params: 每个模型的超参数字典
    look_back: 特征回溯天数
    n_splits: 扩展窗口折数
    progress_cb: callable(pct, msg) 进度回调
    forecast_days: 持有天数（N日持有期收益）
    threshold: 融合概率阈值

    返回: (results: Dict[str, ClfResult], ensemble_result: dict | None)
    """
    if progress_cb:
        progress_cb(0.0, "数据预处理中...")

    # 0. 尝试加载板块指数日收益率（用于相对强弱特征）
    index_returns = _load_index_returns(stock_code, df.index.min(), df.index.max())

    # 1. 预处理（添加 日收益率 / future_ret / 目标涨跌 / 成交量变化率 / 相对成交量 /
    #    量价配合度 / 放量上涨 / 缩量下跌）
    df_proc = preprocess_data(df, forecast_days=forecast_days, index_returns=index_returns)

    # 2. 技术指标（添加 MA / MACD / RSI / 布林带 / OBV / vol_ma / vwap 等，
    #    全部仅用历史数据，无前视）
    df_ind = compute_technical_indicators(df_proc)

    # 3. 筛选可用特征（仅保留 df 中实际存在的列）
    available_features = [c for c in CLF_FEATURE_COLS if c in df_ind.columns]
    if len(available_features) < 3:
        raise ValueError(f"可用特征不足 ({len(available_features)})，请检查数据完整性")

    if progress_cb:
        progress_cb(0.1, "构造滞后特征...")

    # 4. 构造特征和目标（X: 纯滞后特征, y: 目标涨跌）
    X, y, feature_names, aligned_dates, X_latest, latest_date = create_clf_features(df_ind, available_features, look_back)

    # 5. 用 create_clf_features 返回的对齐日期做标签索引（防 dropna 偏移）
    returns = df_ind.loc[aligned_dates, "日收益率"].values
    future_ret = df_ind.loc[aligned_dates, "future_ret"].values
    next_day_ret = df_ind.loc[aligned_dates, "next_day_ret"].values
    dates = np.array(aligned_dates)

    # 全量收益率序列（用于 GARCH 拟合最新日预测）
    all_returns = df_ind["日收益率"].dropna().values

    if progress_cb:
        progress_cb(0.15, f"有效样本: {len(X)} 行 × {len(feature_names)} 特征")

    # 6. 扩展窗口训练 + 验证
    results = run_expanding_window(
        X=X, y=y, returns=returns, dates=dates,
        feature_names=feature_names,
        models=selected_models,
        params=params,
        n_splits=n_splits,
        progress_cb=progress_cb,
        future_ret=future_ret,
        forecast_days=forecast_days,
        next_day_ret=next_day_ret,
        X_latest=X_latest,
        latest_date=latest_date,
        all_returns=all_returns,
    )

    # ── 7. 概率融合（如果两个模型都选中） ──
    ensemble_result = None
    if len(selected_models) == 2:
        model_a, model_b = selected_models[0], selected_models[1]
        if model_a in results and model_b in results:
            proba_a = results[model_a].oos_probabilities
            proba_b = results[model_b].oos_probabilities
            n_common = min(len(proba_a), len(proba_b))

            # 动态权重：基于 AUC(40%) + Sharpe(30%) + 胜率(30%) 综合评分
            weights = _compute_ensemble_weights(results[model_a], results[model_b])
            w_a, w_b = weights[model_a], weights[model_b]

            fused_proba = w_a * proba_a[:n_common] + w_b * proba_b[:n_common]
            fused_signal = (fused_proba >= threshold).astype(int)

            y_common = results[model_a].oos_actuals[:n_common]
            ret_common = results[model_a].oos_returns[:n_common]
            fut_common = results[model_a].oos_future_ret[:n_common]
            ndr_common = results[model_a].oos_next_day_ret[:n_common]
            dates_common = results[model_a].oos_dates[:n_common]

            ensemble_metrics = calculate_classification_metrics(
                y_common, fused_proba,
                ret_common, fused_signal,
                future_ret=fut_common,
                forecast_days=forecast_days,
                next_day_ret=ndr_common,
            )

            ensemble_result = {
                "fused_proba": fused_proba,
                "fused_signal": fused_signal,
                "metrics": ensemble_metrics,
                "weights": {model_a: float(w_a), model_b: float(w_b)},
                "oos_dates": dates_common,
                "oos_returns": ret_common,
                "oos_future_ret": fut_common,
                "oos_next_day_ret": ndr_common,
                "oos_actuals": y_common,
                "threshold": threshold,
                "forecast_days": forecast_days,
            }

            # 融合最新日预测
            lat_a = results[model_a].latest_proba
            lat_b = results[model_b].latest_proba
            if not np.isnan(lat_a) and not np.isnan(lat_b):
                ensemble_result["latest_proba"] = float(w_a * lat_a + w_b * lat_b)
                ensemble_result["latest_date"] = results[model_a].latest_date
                ensemble_result["latest_signal"] = int(ensemble_result["latest_proba"] >= threshold)

    if progress_cb:
        progress_cb(1.0, "涨跌预测完成!")

    return results, ensemble_result


# ── 自动调参：随机搜索 + 早停 ──

_TUNE_SEARCH_SPACE = {
    "look_back": [5, 10, 15, 20, 30, 40],
    "forecast_days": [1, 2, 3, 5],
    "n_splits": [4, 5, 6, 7, 8],
    "xgb_learning_rate": [0.01, 0.03, 0.05, 0.08, 0.1],
    "xgb_n_estimators": [80, 100, 150, 200, 300],
    "xgb_max_depth": [3, 4, 5, 6],
    "xgb_subsample": [0.6, 0.7, 0.8, 0.9],
    "xgb_colsample_bytree": [0.5, 0.6, 0.7, 0.8],
    "xgb_min_child_weight": [1, 3, 5, 8, 12],
    "xgb_reg_alpha": [0, 0.1, 0.3, 0.5, 1.0],
    "xgb_reg_lambda": [1.0, 2.0, 3.0, 4.0],
    "en_C": [0.05, 0.1, 0.3, 0.5, 1.0],
    "en_l1_ratio": [0.05, 0.1, 0.15, 0.3, 0.5],
}


def auto_tune_classifier(
    df: pd.DataFrame,
    stock_code: str = None,
    target_auc: float = 0.53,
    max_trials: int = 30,
    selected_models: List[str] = None,
    trial_cb: Optional[Callable] = None,
) -> dict:
    """
    随机搜索参数空间，找到 ensemble AUC >= target_auc 的参数组合。

    trial_cb: callable(trial_idx, max_trials, trial_result_dict) 每轮回调
    返回: {"best_params": {...}, "best_auc": float, "trials": [...], "found": bool}
    """
    import random as _rand

    if selected_models is None:
        selected_models = ["XGBoost", "ElasticNet"]

    trials = []
    best_auc = 0.0
    best_params = None
    tried = set()

    for trial_idx in range(max_trials):
        # 随机采样参数
        sample = {k: _rand.choice(v) for k, v in _TUNE_SEARCH_SPACE.items()}
        sample_key = tuple(sorted(sample.items()))
        if sample_key in tried:
            continue
        tried.add(sample_key)

        params = {
            "XGBoost": {
                "learning_rate": sample["xgb_learning_rate"],
                "n_estimators": sample["xgb_n_estimators"],
                "max_depth": sample["xgb_max_depth"],
                "subsample": sample["xgb_subsample"],
                "colsample_bytree": sample["xgb_colsample_bytree"],
                "min_child_weight": sample["xgb_min_child_weight"],
                "reg_alpha": sample["xgb_reg_alpha"],
                "reg_lambda": sample["xgb_reg_lambda"],
            },
            "ElasticNet": {
                "C": sample["en_C"],
                "l1_ratio": sample["en_l1_ratio"],
                "max_iter": 5000,
                "tol": 1e-3,
            },
        }

        t0 = time.time()
        try:
            results, ensemble_result = run_classifier_pipeline(
                df=df,
                selected_models=selected_models,
                params=params,
                look_back=sample["look_back"],
                n_splits=sample["n_splits"],
                forecast_days=sample["forecast_days"],
                threshold=0.5,
                stock_code=stock_code,
            )
        except Exception:
            continue
        elapsed = round(time.time() - t0, 1)

        # 提取 AUC
        auc = np.nan
        if ensemble_result and ensemble_result.get("metrics"):
            auc = ensemble_result["metrics"].get("auc", np.nan)
        elif results:
            first_model = list(results.keys())[0]
            auc = results[first_model].overall_metrics.get("auc", np.nan)

        if np.isnan(auc):
            continue

        trial_result = {
            "trial": trial_idx + 1,
            "look_back": sample["look_back"],
            "forecast_days": sample["forecast_days"],
            "n_splits": sample["n_splits"],
            "lr": sample["xgb_learning_rate"],
            "depth": sample["xgb_max_depth"],
            "n_est": sample["xgb_n_estimators"],
            "auc": round(auc, 4),
            "elapsed": elapsed,
            "params": params,
            "sample": sample,
        }
        trials.append(trial_result)

        if auc > best_auc:
            best_auc = auc
            best_params = sample

        if trial_cb:
            trial_cb(trial_idx + 1, max_trials, trial_result)

        if auc >= target_auc:
            break

    return {
        "best_params": best_params,
        "best_auc": round(best_auc, 4),
        "trials": trials,
        "found": best_auc >= target_auc,
    }


# ── 贝叶斯优化调参 (Optuna TPE) ──

def auto_tune_optuna(
    df: pd.DataFrame,
    stock_code: str = None,
    target_auc: float = 0.53,
    max_trials: int = 20,
    selected_models: List[str] = None,
    trial_cb: Optional[Callable] = None,
) -> dict:
    """
    使用 Optuna TPE 贝叶斯优化搜索最佳参数，收敛更快。

    trial_cb: callable(trial_idx, max_trials, trial_result_dict) 每轮回调
    返回: {"best_params": {...}, "best_auc": float, "trials": [...], "found": bool}
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    if selected_models is None:
        selected_models = ["XGBoost", "ElasticNet"]

    trials_log = []
    found_early = [False]

    def _objective(trial):
        if found_early[0]:
            raise optuna.TrialPruned()

        sample = {
            "look_back": trial.suggest_categorical("look_back", [5, 10, 15, 20, 30, 40]),
            "forecast_days": trial.suggest_categorical("forecast_days", [1, 2, 3, 5]),
            "n_splits": trial.suggest_categorical("n_splits", [4, 5, 6, 7, 8]),
            "xgb_learning_rate": trial.suggest_categorical("xgb_learning_rate", [0.01, 0.03, 0.05, 0.08, 0.1]),
            "xgb_n_estimators": trial.suggest_categorical("xgb_n_estimators", [80, 100, 150, 200, 300]),
            "xgb_max_depth": trial.suggest_categorical("xgb_max_depth", [3, 4, 5, 6]),
            "xgb_subsample": trial.suggest_categorical("xgb_subsample", [0.6, 0.7, 0.8, 0.9]),
            "xgb_colsample_bytree": trial.suggest_categorical("xgb_colsample_bytree", [0.5, 0.6, 0.7, 0.8]),
            "xgb_min_child_weight": trial.suggest_categorical("xgb_min_child_weight", [1, 3, 5, 8, 12]),
            "xgb_reg_alpha": trial.suggest_categorical("xgb_reg_alpha", [0.0, 0.1, 0.3, 0.5, 1.0]),
            "xgb_reg_lambda": trial.suggest_categorical("xgb_reg_lambda", [1.0, 2.0, 3.0, 4.0]),
            "en_C": trial.suggest_categorical("en_C", [0.05, 0.1, 0.3, 0.5, 1.0]),
            "en_l1_ratio": trial.suggest_categorical("en_l1_ratio", [0.05, 0.1, 0.15, 0.3, 0.5]),
        }

        params = {
            "XGBoost": {
                "learning_rate": sample["xgb_learning_rate"],
                "n_estimators": sample["xgb_n_estimators"],
                "max_depth": sample["xgb_max_depth"],
                "subsample": sample["xgb_subsample"],
                "colsample_bytree": sample["xgb_colsample_bytree"],
                "min_child_weight": sample["xgb_min_child_weight"],
                "reg_alpha": sample["xgb_reg_alpha"],
                "reg_lambda": sample["xgb_reg_lambda"],
            },
            "ElasticNet": {
                "C": sample["en_C"],
                "l1_ratio": sample["en_l1_ratio"],
                "max_iter": 5000,
                "tol": 1e-3,
            },
        }

        t0 = time.time()
        try:
            results, ensemble_result = run_classifier_pipeline(
                df=df,
                selected_models=selected_models,
                params=params,
                look_back=sample["look_back"],
                n_splits=sample["n_splits"],
                forecast_days=sample["forecast_days"],
                threshold=0.5,
                stock_code=stock_code,
            )
        except Exception:
            return 0.0
        elapsed = round(time.time() - t0, 1)

        auc = np.nan
        if ensemble_result and ensemble_result.get("metrics"):
            auc = ensemble_result["metrics"].get("auc", np.nan)
        elif results:
            first_model = list(results.keys())[0]
            auc = results[first_model].overall_metrics.get("auc", np.nan)

        if np.isnan(auc):
            return 0.0

        trial_result = {
            "trial": trial.number + 1,
            "look_back": sample["look_back"],
            "forecast_days": sample["forecast_days"],
            "n_splits": sample["n_splits"],
            "lr": sample["xgb_learning_rate"],
            "depth": sample["xgb_max_depth"],
            "n_est": sample["xgb_n_estimators"],
            "auc": round(auc, 4),
            "elapsed": elapsed,
            "params": params,
            "sample": sample,
        }
        trials_log.append(trial_result)

        if trial_cb:
            trial_cb(trial.number + 1, max_trials, trial_result)

        if auc >= target_auc:
            found_early[0] = True

        return auc

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(_objective, n_trials=max_trials, show_progress_bar=False)

    best_auc = study.best_value if study.best_trial else 0.0
    best_params = None
    if study.best_trial:
        bp = study.best_trial.params
        best_params = {
            "look_back": bp["look_back"],
            "forecast_days": bp["forecast_days"],
            "n_splits": bp["n_splits"],
            "xgb_learning_rate": bp["xgb_learning_rate"],
            "xgb_n_estimators": bp["xgb_n_estimators"],
            "xgb_max_depth": bp["xgb_max_depth"],
            "xgb_subsample": bp["xgb_subsample"],
            "xgb_colsample_bytree": bp["xgb_colsample_bytree"],
            "xgb_min_child_weight": bp["xgb_min_child_weight"],
            "xgb_reg_alpha": bp["xgb_reg_alpha"],
            "xgb_reg_lambda": bp["xgb_reg_lambda"],
            "en_C": bp["en_C"],
            "en_l1_ratio": bp["en_l1_ratio"],
        }

    return {
        "best_params": best_params,
        "best_auc": round(best_auc, 4),
        "trials": trials_log,
        "found": best_auc >= target_auc,
    }