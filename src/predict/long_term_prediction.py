"""
1-3月中长期收益率预测模块
============================
与短期预测(1-5天)完全隔离：
- 数据粒度：日线 → 周线（W-FRI重采样）
- 预测目标：单日收益率 → 未来4/8/12周累计收益率
- 特征体系：高频量价 → 趋势性/周期性特征
- 主力模型：树模型（LightGBM/XGBoost/CatBoost），禁用DL和统计模型
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score

# ─── 周线数据生成 ──────────────────────────────────────────────

def resample_to_weekly(df_daily: pd.DataFrame) -> dict:
    """
    将日线数据重采样为周线。

    参数:
      df_daily: 日线 DataFrame，需包含列 open, high, low, close, volume, amount

    返回:
      dict with:
        - df_weekly: 周线 DataFrame（中文列名）
        - data_warning: 数据不足警告 msg or None
        - total_weeks: 总周数
    """
    rename_map = {
        "open": "开盘", "high": "最高", "low": "最低",
        "close": "收盘", "volume": "成交量", "amount": "成交额",
    }
    cols_needed = list(rename_map.keys())
    available = [c for c in cols_needed if c in df_daily.columns]

    df_weekly = df_daily[available].resample("W-FRI").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum", "amount": "sum",
    })

    # 处理可能缺失的amount列
    df_weekly = df_weekly.rename(columns=rename_map)

    # 周收益率
    df_weekly["周收益率"] = df_weekly["收盘"].pct_change() * 100
    df_weekly = df_weekly.dropna()

    total_weeks = len(df_weekly)
    warning = None
    if total_weeks < 156:
        warning = f"⚠️ 中长期预测建议使用至少3年的历史数据（当前{total_weeks}周），数据量不足可能导致模型效果不佳"

    return {
        "df_weekly": df_weekly,
        "data_warning": warning,
        "total_weeks": total_weeks,
    }


# ─── 预测目标 ──────────────────────────────────────────────────

def create_weekly_targets(df_weekly: pd.DataFrame) -> pd.DataFrame:
    """
    生成未来 4/8/12 周累计收益率目标（用小数表示）。

    返回添加了列的 DataFrame：
      目标_1月, 目标_2月, 目标_3月
    """
    df = df_weekly.copy()
    close = df["收盘"]

    df["目标_1月"] = close.shift(-4) / close - 1
    df["目标_2月"] = close.shift(-8) / close - 1
    df["目标_3月"] = close.shift(-12) / close - 1

    df = df.iloc[:-12]  # 去掉没有未来数据的最后12行
    return df


# ─── 特征工程 ──────────────────────────────────────────────────

def compute_weekly_features(df_weekly: pd.DataFrame) -> pd.DataFrame:
    """
    计算周线级别的中长期特征（~18个）。

    返回: feature DataFrame (index 与 df_weekly 对齐)
    """
    wk = df_weekly.copy()
    close = wk["收盘"]
    volume = wk["成交量"]
    returns = wk["周收益率"]

    features = {}

    # 1. 均线系统（相对价格偏差）
    for w in [5, 10, 20, 60]:
        features[f"MA{w}"] = close.rolling(w).mean() / close - 1

    # 2. 趋势强度
    for w in [10, 20]:
        def _slope(series, window=w):
            out = np.full(len(series), np.nan)
            for i in range(window - 1, len(series)):
                y = series.iloc[i - window + 1:i + 1].values
                x = np.arange(window)
                out[i] = np.polyfit(x, y, 1)[0]
            return out
        features[f"趋势斜率_{w}周"] = _slope(close) / close.values

    # 3. 波动率特征
    features["波动率_4周"] = returns.rolling(4).std()
    features["波动率_12周"] = returns.rolling(12).std()

    # 4. 成交量特征
    features["成交量_4周均值"] = volume.rolling(4).mean() / volume - 1
    features["成交量_12周均值"] = volume.rolling(12).mean() / volume - 1

    # 量价配合度
    ret_ma4 = returns.rolling(4).mean()
    vol_dir = np.sign(features["成交量_4周均值"])
    features["量价配合度_4周"] = ret_ma4 * vol_dir

    # 5. RSI(14)
    delta = returns
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta.where(delta < 0, 0.0))
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    features["RSI_14"] = 100 - (100 / (1 + rs))

    # 6. MACD 周线
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    features["MACD"] = macd_line
    features["MACD信号"] = macd_signal
    features["MACD柱"] = macd_line - macd_signal

    df_feat = pd.DataFrame(features, index=wk.index)
    df_feat = df_feat.fillna(0)
    return df_feat


# ─── 模型构建 ──────────────────────────────────────────────────

def build_longterm_lgb():
    """LightGBM 中长期模型（主力），严格防过拟合参数"""
    import lightgbm as lgb
    return lgb.LGBMRegressor(
        objective="regression",
        metric="rmse",
        num_leaves=15,
        max_depth=4,
        learning_rate=0.01,
        n_estimators=300,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        verbose=-1,
    )


def build_longterm_xgb():
    """XGBoost 中长期模型（辅助）"""
    import xgboost as xgb
    return xgb.XGBRegressor(
        objective="reg:squarederror",
        max_depth=4,
        learning_rate=0.01,
        n_estimators=300,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbosity=0,
    )


def build_longterm_catboost():
    """CatBoost 中长期模型（辅助）"""
    from catboost import CatBoostRegressor
    return CatBoostRegressor(
        depth=4,
        learning_rate=0.01,
        iterations=300,
        subsample=0.7,
        l2_leaf_reg=3,
        random_seed=42,
        verbose=0,
        allow_writing_files=False,
    )


def build_longterm_lr():
    """线性回归（基准模型）"""
    return LinearRegression()


MODEL_BUILDERS = {
    "LightGBM": build_longterm_lgb,
    "XGBoost": build_longterm_xgb,
    "CatBoost": build_longterm_catboost,
    "LinearRegression": build_longterm_lr,
}

# ─── Bootstrap 置信区间（特征扰动法） ─────────────────────────

def _bootstrap_ci(model, X: np.ndarray, n_iter: int = 100) -> tuple:
    """对树模型做特征扰动法 bootstrap，返回 (mean, lower, upper)"""
    rng = np.random.RandomState(42)
    preds = []
    for _ in range(n_iter):
        noise = rng.normal(0, 0.01, X.shape)
        p = model.predict(X + noise)
        preds.append(p)
    preds = np.array(preds)
    mean = preds.mean(axis=0)
    std = preds.std(axis=0)
    return mean, mean - 1.96 * std, mean + 1.96 * std


def _bootstrap_ci_linear(model, X: np.ndarray, n_iter: int = 100) -> tuple:
    """对线性回归做残差 bootstrap"""
    y_pred = model.predict(X)
    residuals = y_pred - y_pred  # 占位，实际用零均值正态
    rng = np.random.RandomState(42)
    preds = []
    for _ in range(n_iter):
        noise = rng.normal(0, np.std(y_pred) * 0.1, len(y_pred))
        preds.append(y_pred + noise)
    preds = np.array(preds)
    std = preds.std(axis=0)
    return y_pred, y_pred - 1.96 * std, y_pred + 1.96 * std


# ─── 训练管线 ──────────────────────────────────────────────────

def train_long_term_models(df_weekly: pd.DataFrame, selected_models: list,
                           horizon_weeks: int = 4,
                           progress_callback=None) -> dict:
    """
    训练所有选中的中长期模型。

    参数:
      df_weekly: 已处理好的周线数据（含目标列）
      selected_models: ["LightGBM", "XGBoost", "CatBoost", "LinearRegression"] 的子集
      horizon_weeks: 4, 8, 或 12
      progress_callback: fn(pct, msg) 进度回调

    返回:
      {model_name: {"model": obj, "cv_rmse": float, "cv_r2": float,
                    "prediction": float (小数), "confidence_interval": (low, high),
                    "direction_accuracy": float, "feature_importance": dict}}
    """
    horizon_map = {4: "目标_1月", 8: "目标_2月", 12: "目标_3月"}
    target_col = horizon_map[horizon_weeks]

    df_with_targets = create_weekly_targets(df_weekly)
    features_df = compute_weekly_features(df_with_targets)

    # 对齐 index
    common_idx = features_df.index.intersection(df_with_targets.index)
    X_all = features_df.loc[common_idx].values
    y_all = df_with_targets.loc[common_idx, target_col].values
    feature_names = list(features_df.columns)
    latest_close = float(df_with_targets.loc[common_idx[-1], "收盘"])

    results = {}
    total = len(selected_models)
    for i, model_name in enumerate(selected_models):
        if progress_callback:
            progress_callback(i / total, f"训练 {model_name}...")

        builder = MODEL_BUILDERS.get(model_name)
        if builder is None:
            continue

        try:
            # TimeSeriesSplit 5折交叉验证
            tscv = TimeSeriesSplit(n_splits=min(5, len(X_all) // 24))
            cv_rmse_scores = []
            cv_r2_scores = []
            direction_accs = []

            for fold_i, (train_idx, val_idx) in enumerate(tscv.split(X_all)):
                X_tr, X_vl = X_all[train_idx], X_all[val_idx]
                y_tr, y_vl = y_all[train_idx], y_all[val_idx]

                model = builder()
                # LightGBM/XGBoost/CatBoost 有 early_stopping_rounds
                if model_name in ("LightGBM", "XGBoost"):
                    eval_set = [(X_vl, y_vl)]
                    model.fit(X_tr, y_tr, eval_set=eval_set, verbose=False)
                elif model_name == "CatBoost":
                    model.fit(X_tr, y_tr, eval_set=(X_vl, y_vl),
                             early_stopping_rounds=20, verbose=False)
                else:  # LinearRegression
                    model.fit(X_tr, y_tr)

                y_pred = model.predict(X_vl)
                cv_rmse_scores.append(np.sqrt(mean_squared_error(y_vl, y_pred)))
                cv_r2_scores.append(r2_score(y_vl, y_pred))

                # 方向准确率（预测方向 vs 实际方向）
                pred_dir = np.sign(y_pred)
                actual_dir = np.sign(y_vl)
                dir_acc = np.mean(pred_dir == actual_dir)
                direction_accs.append(dir_acc)

                if progress_callback and fold_i == 0:
                    progress_callback((i + 0.3) / total, f"{model_name}: CV中...")

            avg_rmse = float(np.mean(cv_rmse_scores))
            avg_r2 = float(np.mean(cv_r2_scores))
            avg_dir_acc = float(np.mean(direction_accs))

            # 全量重训
            final_model = builder()
            if model_name in ("LightGBM", "XGBoost"):
                final_model.fit(X_all, y_all, verbose=False)
            elif model_name == "CatBoost":
                final_model.fit(X_all, y_all, verbose=False)
            else:
                final_model.fit(X_all, y_all)

            # 对未来 horizon_weeks 的预测
            latest_features = X_all[-1:].copy()
            future_pred = float(final_model.predict(latest_features)[0])

            # Bootstrap 置信区间
            if model_name == "LinearRegression":
                _, ci_low, ci_high = _bootstrap_ci_linear(final_model, X_all)
            else:
                _, ci_low, ci_high = _bootstrap_ci(final_model, X_all)
            ci_low = float(ci_low[-1]) if len(ci_low) > 0 else future_pred * 0.8
            ci_high = float(ci_high[-1]) if len(ci_high) > 0 else future_pred * 1.2

            # 特征重要性
            feat_imp = {}
            if model_name == "LightGBM":
                feat_imp = dict(zip(feature_names,
                                   final_model.feature_importances_.tolist()))
            elif model_name == "XGBoost":
                feat_imp = dict(zip(feature_names,
                                   final_model.feature_importances_.tolist()))
            elif model_name == "CatBoost":
                feat_imp = dict(zip(feature_names,
                                   final_model.get_feature_importance().tolist()))

            results[model_name] = {
                "model": final_model,
                "cv_rmse": avg_rmse,
                "cv_r2": avg_r2,
                "prediction": future_pred,  # 小数形式累计收益率
                "confidence_interval": (ci_low, ci_high),
                "direction_accuracy": avg_dir_acc,
                "feature_importance": feat_imp,
            }

        except Exception as e:
            results[model_name] = {
                "model": None,
                "cv_rmse": float("nan"),
                "cv_r2": float("nan"),
                "prediction": 0.0,
                "confidence_interval": (0.0, 0.0),
                "direction_accuracy": 0.0,
                "feature_importance": {},
                "error": str(e),
            }

        if progress_callback:
            progress_callback((i + 1) / total, f"{model_name}: 完成")

    # 集成：等权平均
    valid_preds = [r["prediction"] for r in results.values()
                   if not np.isnan(r.get("cv_rmse", float("nan")))]
    if valid_preds:
        ensemble_pred = np.mean(valid_preds)
    else:
        ensemble_pred = 0.0

    results["_ensemble"] = {
        "prediction": ensemble_pred,
        "latest_close": latest_close,
        "horizon_weeks": horizon_weeks,
    }

    return results


# ─── 风险评估 ──────────────────────────────────────────────────

def assess_risk(df_weekly: pd.DataFrame) -> dict:
    """
    计算中长线风险指标。

    返回:
      annual_vol_pct: 年化波动率(%)
      max_drawdown_pct: 最大回撤(%)
      risk_level: 低/中/高
    """
    returns = df_weekly["周收益率"].dropna()
    if len(returns) < 20:
        return {"annual_vol_pct": 0, "max_drawdown_pct": 0, "risk_level": "数据不足"}

    annual_vol = float(np.std(returns) * np.sqrt(52))

    # 最大回撤
    close = df_weekly["收盘"].values
    peak = np.maximum.accumulate(close)
    drawdown = (close - peak) / peak
    max_dd = float(np.min(drawdown) * 100)

    if annual_vol < 15:
        level = "低"
    elif annual_vol < 25:
        level = "中"
    else:
        level = "高"

    return {
        "annual_vol_pct": round(annual_vol, 1),
        "max_drawdown_pct": round(abs(max_dd), 1),
        "risk_level": level,
    }


def get_rating(pred_return_pct: float, direction_acc: float) -> str:
    """
    根据预测收益率和方向准确率给出综合评级。

    返回: 强烈看涨 / 看涨 / 中性 / 看跌 / 强烈看跌
    """
    if pred_return_pct > 10 and direction_acc > 0.6:
        return "强烈看涨"
    elif pred_return_pct > 3:
        return "看涨"
    elif pred_return_pct > -3:
        return "中性"
    elif pred_return_pct > -10:
        return "看跌"
    else:
        return "强烈看跌"