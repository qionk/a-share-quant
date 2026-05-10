"""
价格预测 - 训练与评估
时序交叉验证 + 模型训练 + 指标计算 + 集成预测
"""

import time
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .features import (
    compute_technical_indicators, prepare_features,
    create_sequences, time_series_split, inverse_transform_predictions,
    DEFAULT_FEATURE_COLS,
)
from .models import (
    ModelConfig, build_lstm, build_gru, build_cnn,
    fit_arima, predict_arima, fit_egarch, predict_egarch, returns_to_prices,
)


@dataclass
class TrainResult:
    """单模型训练结果"""
    model_name: str
    model_object: object = None
    train_history: dict = field(default_factory=dict)
    cv_metrics: dict = field(default_factory=dict)
    test_predictions: np.ndarray = field(default_factory=lambda: np.array([]))
    test_actuals: np.ndarray = field(default_factory=lambda: np.array([]))
    confidence_lower: np.ndarray = field(default_factory=lambda: np.array([]))
    confidence_upper: np.ndarray = field(default_factory=lambda: np.array([]))
    future_predictions: np.ndarray = field(default_factory=lambda: np.array([]))
    future_conf_lower: np.ndarray = field(default_factory=lambda: np.array([]))
    future_conf_upper: np.ndarray = field(default_factory=lambda: np.array([]))
    training_time: float = 0.0
    scaler: object = None
    feature_cols: list = field(default_factory=list)
    n_features: int = 0


def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """计算评估指标: MAE, RMSE, MAPE, R²"""
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[mask], y_pred[mask]

    if len(y_true) == 0:
        return {"mae": np.nan, "rmse": np.nan, "mape": np.nan, "r2": np.nan}

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    nonzero = y_true != 0
    mape = np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100 if nonzero.any() else np.nan
    r2 = r2_score(y_true, y_pred)

    return {"mae": mae, "rmse": rmse, "mape": mape, "r2": r2}


def _train_dl_model(model, X_train, y_train, X_val, y_val, config: ModelConfig,
                    progress_callback=None):
    """训练深度学习模型（LSTM/GRU/CNN 通用）"""
    import tensorflow as tf

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            patience=config.early_stop_patience,
            restore_best_weights=True,
            monitor="val_loss",
        )
    ]

    if progress_callback:
        class ProgressCB(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                progress_callback(epoch + 1, config.epochs, logs or {})
        callbacks.append(ProgressCB())

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config.epochs,
        batch_size=config.batch_size,
        callbacks=callbacks,
        verbose=0,
    )
    return {k: [float(v) for v in vals] for k, vals in history.history.items()}


def _mc_dropout_predict(model, X, n_iter=100):
    """Monte Carlo Dropout 预测（含置信区间）"""
    import tensorflow as tf

    preds = []
    for _ in range(n_iter):
        p = model(X, training=True)
        preds.append(p.numpy().flatten())
    preds = np.array(preds)
    mean = preds.mean(axis=0)
    std = preds.std(axis=0)
    return mean, mean - 1.96 * std, mean + 1.96 * std


def train_dl_model(model_name: str, X, y, config: ModelConfig,
                   n_splits: int = 5, progress_callback=None) -> TrainResult:
    """训练单个深度学习模型（含时序CV）"""
    start = time.time()
    result = TrainResult(model_name=model_name)

    builders = {"LSTM": build_lstm, "GRU": build_gru, "1D-CNN": build_cnn}
    build_fn = builders[model_name]

    # 时序交叉验证
    splits = time_series_split(len(X), n_splits=n_splits)
    cv_scores = []
    all_history = {}

    for fold_i, (train_idx, val_idx) in enumerate(splits):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_vl, y_vl = X[val_idx], y[val_idx]

        model = build_fn(config)

        def fold_progress(epoch, total, logs, _fold=fold_i, _total_folds=len(splits)):
            if progress_callback:
                overall = (_fold * total + epoch) / (_total_folds * total)
                progress_callback(overall, f"Fold {_fold+1}/{_total_folds} - Epoch {epoch}/{total}")

        hist = _train_dl_model(model, X_tr, y_tr, X_vl, y_vl, config, fold_progress)

        y_pred = model.predict(X_vl, verbose=0).flatten()
        fold_metrics = calc_metrics(y_vl, y_pred)
        cv_scores.append(fold_metrics)

        if fold_i == len(splits) - 1:
            all_history = hist

    # 汇总CV指标
    result.cv_metrics = {
        k: np.mean([s[k] for s in cv_scores if not np.isnan(s[k])])
        for k in ["mae", "rmse", "mape", "r2"]
    }

    # 全量训练最终模型
    split_pt = int(len(X) * 0.85)
    X_train, X_test = X[:split_pt], X[split_pt:]
    y_train, y_test = y[:split_pt], y[split_pt:]

    final_model = build_fn(config)
    final_hist = _train_dl_model(final_model, X_train, y_train, X_test, y_test, config)
    result.train_history = final_hist
    result.model_object = final_model

    # 测试集预测 + 置信区间
    mean_pred, lower, upper = _mc_dropout_predict(final_model, X_test)
    result.test_predictions = mean_pred
    result.test_actuals = y_test
    result.confidence_lower = lower
    result.confidence_upper = upper

    result.training_time = time.time() - start
    return result


def train_arima_model(close_prices: np.ndarray, config: ModelConfig,
                      progress_callback=None) -> TrainResult:
    """训练 ARIMA 模型（基于收益率序列，避免随机游走问题）"""
    start = time.time()
    result = TrainResult(model_name="ARIMA")

    if progress_callback:
        progress_callback(0.1, "ARIMA: 拟合中...")

    returns = pd.Series(close_prices).pct_change().dropna().values * 100

    split_pt = int(len(returns) * 0.85)
    train_ret, test_ret = returns[:split_pt], returns[split_pt:]
    test_close = close_prices[split_pt + 1:]

    try:
        model = fit_arima(train_ret)
        result.model_object = model

        pred_rets = []
        for i in range(len(test_ret)):
            fc, _ = predict_arima(model, steps=1)
            pred_rets.append(fc[0])
            model.update(test_ret[i:i+1])

        pred_rets = np.array(pred_rets)
        pred_prices = returns_to_prices(close_prices[split_pt], pred_rets)

        n = min(len(pred_prices), len(test_close))
        result.test_predictions = pred_prices[:n]
        result.test_actuals = test_close[:n]
        result.cv_metrics = calc_metrics(test_close[:n], pred_prices[:n])

        avg_err = np.mean(np.abs(pred_prices[:n] - test_close[:n]))
        result.confidence_lower = pred_prices[:n] - 1.96 * avg_err
        result.confidence_upper = pred_prices[:n] + 1.96 * avg_err

    except Exception as e:
        result.cv_metrics = {"mae": np.nan, "rmse": np.nan, "mape": np.nan, "r2": np.nan}
        result.train_history = {"error": str(e)}

    if progress_callback:
        progress_callback(1.0, "ARIMA: 完成")

    result.training_time = time.time() - start
    return result


def train_egarch_model(returns: np.ndarray, close_prices: np.ndarray,
                       config: ModelConfig, progress_callback=None) -> TrainResult:
    """训练 EGARCH 模型"""
    start = time.time()
    result = TrainResult(model_name="EGARCH")

    if progress_callback:
        progress_callback(0.1, "EGARCH: 拟合中...")

    split_pt = int(len(returns) * 0.85)
    train_ret = returns[:split_pt]
    test_ret = returns[split_pt:]
    test_close = close_prices[split_pt:]

    try:
        model_result = fit_egarch(train_ret)
        result.model_object = model_result

        # 滚动预测
        preds_ret = []
        for i in range(len(test_ret)):
            from arch import arch_model
            am = arch_model(returns[:split_pt + i], vol="EGARCH", p=1, q=1,
                            mean="ARX", dist="normal")
            res = am.fit(disp="off", show_warning=False)
            fc_mean, _ = predict_egarch(res, steps=1)
            preds_ret.append(fc_mean[0])

        preds_ret = np.array(preds_ret)
        pred_prices = returns_to_prices(close_prices[split_pt - 1], preds_ret)

        result.test_predictions = pred_prices[:len(test_close)]
        result.test_actuals = test_close
        result.cv_metrics = calc_metrics(test_close, pred_prices[:len(test_close)])

        avg_err = np.mean(np.abs(pred_prices[:len(test_close)] - test_close))
        result.confidence_lower = pred_prices[:len(test_close)] - 1.96 * avg_err
        result.confidence_upper = pred_prices[:len(test_close)] + 1.96 * avg_err

    except Exception as e:
        result.cv_metrics = {"mae": np.nan, "rmse": np.nan, "mape": np.nan, "r2": np.nan}
        result.train_history = {"error": str(e)}

    if progress_callback:
        progress_callback(1.0, "EGARCH: 完成")

    result.training_time = time.time() - start
    return result


def train_all_models(df: pd.DataFrame, selected_models: list, config: ModelConfig,
                     forecast_days: int = 5, progress_callback=None) -> dict:
    """
    批量训练所有选定模型
    返回: {model_name: TrainResult}
    """
    df_ind = compute_technical_indicators(df)
    scaled, scaler, feature_cols, cleaned_df = prepare_features(df_ind)
    n_features = len(feature_cols)
    config.n_features = n_features

    X, y = create_sequences(scaled, config.look_back, target_col_idx=0)

    results = {}
    dl_models = [m for m in selected_models if m in ("LSTM", "GRU", "1D-CNN")]
    stat_models = [m for m in selected_models if m in ("ARIMA", "EGARCH")]

    total = len(selected_models)
    done = 0

    for name in dl_models:
        if progress_callback:
            progress_callback(done / total, f"训练 {name}...")

        result = train_dl_model(name, X, y, config, n_splits=3)
        result.scaler = scaler
        result.feature_cols = feature_cols
        result.n_features = n_features

        # 反归一化测试集结果
        result.test_predictions = inverse_transform_predictions(
            result.test_predictions, scaler, n_features, 0)
        result.test_actuals = inverse_transform_predictions(
            result.test_actuals, scaler, n_features, 0)
        result.confidence_lower = inverse_transform_predictions(
            result.confidence_lower, scaler, n_features, 0)
        result.confidence_upper = inverse_transform_predictions(
            result.confidence_upper, scaler, n_features, 0)

        # 重新算反归一化后的指标
        result.cv_metrics = calc_metrics(result.test_actuals, result.test_predictions)

        # 未来预测
        last_seq = X[-1:].copy()
        future_preds = _predict_future_dl(result.model_object, last_seq, forecast_days,
                                           scaler, n_features, 0)
        result.future_predictions = future_preds

        results[name] = result
        done += 1

    close_arr = cleaned_df["close"].values if "close" in cleaned_df.columns else df["close"].dropna().values

    if "ARIMA" in stat_models:
        if progress_callback:
            progress_callback(done / total, "训练 ARIMA...")
        result = train_arima_model(close_arr, config, progress_callback=None)
        result.scaler = scaler
        result.feature_cols = feature_cols
        result.n_features = n_features

        # 未来预测（模型已在收益率上训练，预测结果也是收益率）
        try:
            if result.model_object is not None:
                fc_ret, conf_ret = predict_arima(result.model_object, steps=forecast_days)
                last_price = close_arr[-1]
                result.future_predictions = returns_to_prices(last_price, fc_ret)
                result.future_conf_lower = returns_to_prices(last_price, conf_ret[:, 0])
                result.future_conf_upper = returns_to_prices(last_price, conf_ret[:, 1])
        except Exception:
            result.future_predictions = np.array([])

        results["ARIMA"] = result
        done += 1

    if "EGARCH" in stat_models:
        if progress_callback:
            progress_callback(done / total, "训练 EGARCH...")
        returns = pd.Series(close_arr).pct_change().dropna().values * 100
        result = train_egarch_model(returns, close_arr, config, progress_callback=None)
        result.scaler = scaler
        result.feature_cols = feature_cols
        result.n_features = n_features

        # 未来预测
        try:
            if result.model_object is not None:
                mean_fc, vol_fc = predict_egarch(result.model_object, steps=forecast_days)
                result.future_predictions = returns_to_prices(close_arr[-1], mean_fc)
                result.future_conf_lower = returns_to_prices(close_arr[-1], mean_fc - 1.96 * vol_fc)
                result.future_conf_upper = returns_to_prices(close_arr[-1], mean_fc + 1.96 * vol_fc)
        except Exception:
            result.future_predictions = np.array([])

        results["EGARCH"] = result
        done += 1

    if progress_callback:
        progress_callback(1.0, "全部完成")

    return results


def _predict_future_dl(model, last_sequence, steps, scaler, n_features, target_idx):
    """深度学习模型逐步预测未来"""
    predictions = []
    current_seq = last_sequence.copy()

    for _ in range(steps):
        pred = model.predict(current_seq, verbose=0).flatten()[0]
        predictions.append(pred)

        new_row = current_seq[0, -1, :].copy()
        new_row[target_idx] = pred
        current_seq = np.concatenate([current_seq[:, 1:, :],
                                       new_row.reshape(1, 1, -1)], axis=1)

    preds = np.array(predictions)
    return inverse_transform_predictions(preds, scaler, n_features, target_idx)


def compute_ensemble_weights(results: dict) -> dict:
    """基于 CV RMSE 的逆权重"""
    valid = {k: v for k, v in results.items()
             if v.cv_metrics.get("rmse") and not np.isnan(v.cv_metrics["rmse"]) and v.cv_metrics["rmse"] > 0}

    if not valid:
        n = len(results)
        return {k: 1.0 / n for k in results} if n > 0 else {}

    inv_rmse = {k: 1.0 / v.cv_metrics["rmse"] for k, v in valid.items()}
    total = sum(inv_rmse.values())
    return {k: w / total for k, w in inv_rmse.items()}


def ensemble_predict(results: dict, weights: dict, forecast_days: int,
                     last_price: float) -> dict:
    """
    集成预测未来 N 天
    返回包含各项预测结果的字典
    """
    model_preds = {}
    for name, result in results.items():
        if len(result.future_predictions) > 0:
            preds = result.future_predictions[:forecast_days]
            if len(preds) == forecast_days:
                model_preds[name] = preds

    if not model_preds:
        return {"predicted_close": np.array([]),
                "daily_return": np.array([]),
                "cumulative_return": np.array([]),
                "model_predictions": {},
                "weights": weights}

    # 加权平均
    weighted_sum = np.zeros(forecast_days)
    total_weight = 0
    for name, preds in model_preds.items():
        w = weights.get(name, 0)
        if w > 0:
            weighted_sum += preds * w
            total_weight += w

    if total_weight > 0:
        ensemble_prices = weighted_sum / total_weight
    else:
        ensemble_prices = np.mean(list(model_preds.values()), axis=0)

    # 收益率
    price_series = np.concatenate([[last_price], ensemble_prices])
    daily_ret = np.diff(price_series) / price_series[:-1] * 100
    cum_ret = (ensemble_prices / last_price - 1) * 100

    # 置信区间（集成所有模型的区间）
    all_lowers, all_uppers = [], []
    for name, result in results.items():
        if len(result.future_conf_lower) >= forecast_days:
            all_lowers.append(result.future_conf_lower[:forecast_days])
        if len(result.future_conf_upper) >= forecast_days:
            all_uppers.append(result.future_conf_upper[:forecast_days])

    conf_lower = np.min(all_lowers, axis=0) if all_lowers else ensemble_prices * 0.95
    conf_upper = np.max(all_uppers, axis=0) if all_uppers else ensemble_prices * 1.05

    return {
        "predicted_close": ensemble_prices,
        "daily_return": daily_ret,
        "cumulative_return": cum_ret,
        "confidence_lower": conf_lower,
        "confidence_upper": conf_upper,
        "model_predictions": model_preds,
        "weights": weights,
    }


def backtest_predictions(df: pd.DataFrame, results: dict, look_back: int = 30,
                         window_size: int = 200) -> pd.DataFrame:
    """
    历史预测回测：滚动预测下一日，统计方向正确率和误差
    简化版：使用已有测试集预测结果
    """
    records = []
    for name, result in results.items():
        if len(result.test_predictions) == 0 or len(result.test_actuals) == 0:
            continue
        preds = result.test_predictions
        actuals = result.test_actuals
        n = min(len(preds), len(actuals))
        for i in range(1, n):
            actual_dir = 1 if actuals[i] > actuals[i-1] else -1
            pred_dir = 1 if preds[i] > preds[i-1] else -1
            records.append({
                "model": name,
                "step": i,
                "actual": actuals[i],
                "predicted": preds[i],
                "direction_correct": actual_dir == pred_dir,
                "error_pct": abs(preds[i] - actuals[i]) / actuals[i] * 100 if actuals[i] != 0 else 0,
            })
    return pd.DataFrame(records)
