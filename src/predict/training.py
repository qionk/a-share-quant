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
    ModelConfig, build_lstm, build_gru, build_cnn, build_patchtst, build_tft,
    fit_arima, predict_arima, fit_egarch, predict_egarch, returns_to_prices,
)


class TrainingCallbacks:
    """训练回调协议：实时监控接口"""

    def on_training_start(self, model_list: list) -> None:
        pass

    def on_model_start(self, model_name: str, model_index: int, total_models: int) -> None:
        pass

    def on_epoch_end(self, model_name: str, epoch: int, total_epochs: int,
                     train_loss: float, val_loss: float, lr: float,
                     grad_norm: float = None) -> None:
        pass

    def on_model_end(self, model_name: str, result) -> None:
        pass

    def on_training_complete(self, all_results: dict) -> None:
        pass

    def on_overfitting_warning(self, model_name: str, epoch: int,
                                val_loss: float, best_val_loss: float) -> None:
        pass


class LegacyCallbackAdapter(TrainingCallbacks):
    """将旧式 (pct, msg) 回调包装为 TrainingCallbacks"""

    def __init__(self, fn, total_models: int):
        self._fn = fn
        self._total = total_models
        self._idx = 0

    def on_model_start(self, model_name, model_index, total_models):
        self._idx = model_index
        self._fn(model_index / total_models, f"训练 {model_name}...")

    def on_epoch_end(self, model_name, epoch, total_epochs, train_loss, val_loss, lr, grad_norm=None):
        base = self._idx / self._total
        within = (epoch / total_epochs) / self._total
        self._fn(base + within, f"{model_name}: Epoch {epoch}/{total_epochs}")

    def on_training_complete(self, all_results):
        self._fn(1.0, "全部完成")


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
                    model_name: str = "", callbacks: TrainingCallbacks = None):
    """训练深度学习模型（含LR调度、梯度监控、过拟合检测）"""
    import tensorflow as tf

    keras_callbacks = [
        tf.keras.callbacks.EarlyStopping(
            patience=config.early_stop_patience,
            restore_best_weights=True,
            monitor="val_loss",
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=0
        ),
    ]

    if callbacks:
        x_sample = X_train[:min(8, len(X_train))]
        y_sample = y_train[:min(8, len(y_train))]

        class RichProgressCB(tf.keras.callbacks.Callback):
            def __init__(self):
                super().__init__()
                self.best_val_loss = float("inf")
                self.overfit_streak = 0
                self._compute_grads = None

            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                train_loss = logs.get("loss", 0)
                val_loss = logs.get("val_loss", 0)
                lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))

                grad_norm = None
                if self._compute_grads is None:
                    self._compute_grads = self.model.count_params() < 100000
                if self._compute_grads and epoch % 5 == 0:
                    try:
                        with tf.GradientTape() as tape:
                            preds = self.model(x_sample, training=True)
                            loss = tf.reduce_mean(tf.keras.losses.mse(y_sample, tf.squeeze(preds)))
                        grads = tape.gradient(loss, self.model.trainable_variables)
                        total = sum(float(tf.reduce_sum(tf.square(g))) for g in grads if g is not None)
                        grad_norm = float(np.sqrt(total))
                    except Exception:
                        self._compute_grads = False

                callbacks.on_epoch_end(model_name, epoch + 1, config.epochs,
                                       train_loss, val_loss, lr, grad_norm)

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.overfit_streak = 0
                else:
                    self.overfit_streak += 1
                    if self.overfit_streak >= 3 and val_loss > self.best_val_loss * 1.1:
                        callbacks.on_overfitting_warning(
                            model_name, epoch + 1, val_loss, self.best_val_loss)

        keras_callbacks.append(RichProgressCB())

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config.epochs,
        batch_size=config.batch_size,
        callbacks=keras_callbacks,
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
                   n_splits: int = 5, quick: bool = False,
                   callbacks: TrainingCallbacks = None) -> TrainResult:
    """训练单个深度学习模型（含时序CV）"""
    start = time.time()
    result = TrainResult(model_name=model_name)

    builders = {"LSTM": build_lstm, "GRU": build_gru, "1D-CNN": build_cnn,
                "PatchTST": build_patchtst, "TFT": build_tft}
    build_fn = builders[model_name]

    split_pt = int(len(X) * 0.85)
    X_train, X_test = X[:split_pt], X[split_pt:]
    y_train, y_test = y[:split_pt], y[split_pt:]

    if quick:
        # 快速模式: 跳过CV，直接训练一次
        import sys as _sys
        print(f"[DEBUG] {model_name}: build model...", flush=True, file=_sys.stderr)
        final_model = build_fn(config)
        print(f"[DEBUG] {model_name}: params={final_model.count_params()}, start fit...", flush=True, file=_sys.stderr)
        # 快速模式不传 callbacks，避免 Streamlit UI 更新拖慢训练
        final_hist = _train_dl_model(final_model, X_train, y_train, X_test, y_test, config,
                                      model_name=model_name, callbacks=None)
        print(f"[DEBUG] {model_name}: fit done, predict...", flush=True, file=_sys.stderr)
        result.train_history = final_hist
        result.model_object = final_model

        y_pred = final_model.predict(X_test, verbose=0).flatten()
        result.cv_metrics = calc_metrics(y_test, y_pred)
        result.test_predictions = y_pred
        result.test_actuals = y_test

        # 快速模式: MC Dropout 仅 10 次
        print(f"[DEBUG] {model_name}: MC dropout...", flush=True, file=_sys.stderr)
        _, lower, upper = _mc_dropout_predict(final_model, X_test, n_iter=10)
        result.confidence_lower = lower
        result.confidence_upper = upper
    else:
        # 正常模式: 时序交叉验证
        splits = time_series_split(len(X), n_splits=n_splits)
        cv_scores = []

        for fold_i, (train_idx, val_idx) in enumerate(splits):
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_vl, y_vl = X[val_idx], y[val_idx]

            model = build_fn(config)
            hist = _train_dl_model(model, X_tr, y_tr, X_vl, y_vl, config,
                                   model_name=model_name, callbacks=callbacks)

            y_pred = model.predict(X_vl, verbose=0).flatten()
            fold_metrics = calc_metrics(y_vl, y_pred)
            cv_scores.append(fold_metrics)

        result.cv_metrics = {
            k: np.mean([s[k] for s in cv_scores if not np.isnan(s[k])])
            for k in ["mae", "rmse", "mape", "r2"]
        }

        # 全量训练最终模型
        final_model = build_fn(config)
        final_hist = _train_dl_model(final_model, X_train, y_train, X_test, y_test, config,
                                      model_name=model_name, callbacks=callbacks)
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


def _calc_momentum_drift(close_prices: np.ndarray, lookback: int = 20) -> float:
    """计算最近N天的线性趋势斜率，转换为日收益率(%)"""
    recent = close_prices[-lookback:]
    x = np.arange(len(recent))
    coefs = np.polyfit(x, recent, deg=1)
    daily_drift_pct = coefs[0] / close_prices[-1] * 100
    return daily_drift_pct


def train_arima_model(close_prices: np.ndarray, config: ModelConfig,
                      progress_callback=None) -> TrainResult:
    """
    训练 ARIMA(1,1,1) 模型
    直接在价格序列上训练，d=1 自带差分。
    """
    start = time.time()
    result = TrainResult(model_name="ARIMA")

    if progress_callback:
        progress_callback(0.1, "ARIMA: 拟合中...")

    split_pt = int(len(close_prices) * 0.85)
    train_close = close_prices[:split_pt]
    test_close = close_prices[split_pt:]

    try:
        model = fit_arima(train_close)
        result.model_object = model

        # 滚动一步预测
        from statsmodels.tsa.arima.model import ARIMA
        history = list(train_close)
        pred_prices = []
        for i in range(len(test_close)):
            m = ARIMA(history, order=(1, 1, 1), trend='t')
            res = m.fit()
            fc = np.array(res.get_forecast(steps=1).predicted_mean)[0]
            pred_prices.append(fc)
            history.append(test_close[i])

            if progress_callback and i % 20 == 0:
                progress_callback(0.1 + 0.8 * i / len(test_close), f"ARIMA: 滚动预测 {i}/{len(test_close)}")

        pred_prices = np.array(pred_prices)
        result.test_predictions = pred_prices
        result.test_actuals = test_close
        result.cv_metrics = calc_metrics(test_close, pred_prices)
        result.model_object = res

        avg_err = np.mean(np.abs(pred_prices - test_close))
        result.confidence_lower = pred_prices - 1.96 * avg_err
        result.confidence_upper = pred_prices + 1.96 * avg_err

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
                            mean="AR", lags=1, dist="normal")
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
                     forecast_days: int = 5, progress_callback=None,
                     callbacks: TrainingCallbacks = None,
                     quick: bool = False) -> dict:
    """
    批量训练所有选定模型
    返回: {model_name: TrainResult}
    quick: 快速模式，跳过CV和减少MC Dropout
    """
    # 向后兼容：旧式callback自动包装
    if progress_callback and not callbacks:
        callbacks = LegacyCallbackAdapter(progress_callback, len(selected_models))

    df_ind = compute_technical_indicators(df)
    scaled, scaler, feature_cols, cleaned_df = prepare_features(df_ind)

    # 快速模式: 只用 5 个核心特征，大幅减少计算量
    if quick:
        core_names = ["close", "volume", "ma5", "pct_change", "rsi"]
        core_idx = [i for i, c in enumerate(feature_cols) if c in core_names]
        if len(core_idx) < 2:
            core_idx = list(range(min(5, len(feature_cols))))
        if 0 not in core_idx:
            core_idx = [0] + core_idx
        scaled = scaled[:, core_idx]
        feature_cols = [feature_cols[i] for i in core_idx]
        # 重建 scaler 匹配裁剪后的特征维度
        from sklearn.preprocessing import MinMaxScaler
        new_scaler = MinMaxScaler()
        new_scaler.min_ = scaler.min_[core_idx]
        new_scaler.scale_ = scaler.scale_[core_idx]
        new_scaler.data_min_ = scaler.data_min_[core_idx]
        new_scaler.data_max_ = scaler.data_max_[core_idx]
        new_scaler.data_range_ = scaler.data_range_[core_idx]
        new_scaler.n_features_in_ = len(core_idx)
        new_scaler.feature_range = scaler.feature_range
        scaler = new_scaler

    n_features = len(feature_cols)
    config.n_features = n_features

    X, y = create_sequences(scaled, config.look_back, target_col_idx=0)

    import sys as _sys
    print(f"[DEBUG] quick={quick}, features={n_features}, X.shape={X.shape}, "
          f"epochs={config.epochs}, patience={config.early_stop_patience}", flush=True, file=_sys.stderr)

    results = {}
    dl_models = [m for m in selected_models if m in ("LSTM", "GRU", "1D-CNN", "PatchTST", "TFT")]
    stat_models = [m for m in selected_models if m in ("ARIMA", "EGARCH")]

    total = len(selected_models)
    done = 0

    if callbacks:
        callbacks.on_training_start(selected_models)

    for name in dl_models:
        if callbacks:
            callbacks.on_model_start(name, done, total)

        try:
            result = train_dl_model(name, X, y, config, n_splits=3, quick=quick, callbacks=callbacks)
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

        except Exception as e:
            result = TrainResult(model_name=name)
            result.cv_metrics = {"mae": np.nan, "rmse": np.nan, "mape": np.nan, "r2": np.nan}
            result.train_history = {"error": str(e)}
            result.training_time = time.time()

        if callbacks:
            callbacks.on_model_end(name, result)
        results[name] = result
        done += 1

    close_arr = cleaned_df["close"].values if "close" in cleaned_df.columns else df["close"].dropna().values

    if "ARIMA" in stat_models:
        if callbacks:
            callbacks.on_model_start("ARIMA", done, total)
        result = train_arima_model(close_arr, config, progress_callback=None)
        result.scaler = scaler
        result.feature_cols = feature_cols
        result.n_features = n_features

        try:
            if result.model_object is not None:
                fc, conf = predict_arima(result.model_object, steps=forecast_days)
                result.future_predictions = fc
                result.future_conf_lower = conf[:, 0]
                result.future_conf_upper = conf[:, 1]
        except Exception:
            result.future_predictions = np.array([])

        if callbacks:
            callbacks.on_model_end("ARIMA", result)
        results["ARIMA"] = result
        done += 1

    if "EGARCH" in stat_models:
        if callbacks:
            callbacks.on_model_start("EGARCH", done, total)
        returns = pd.Series(close_arr).pct_change().dropna().values * 100
        result = train_egarch_model(returns, close_arr, config, progress_callback=None)
        result.scaler = scaler
        result.feature_cols = feature_cols
        result.n_features = n_features

        try:
            if result.model_object is not None:
                mean_fc, vol_fc = predict_egarch(result.model_object, steps=forecast_days)
                result.future_predictions = returns_to_prices(close_arr[-1], mean_fc)
                result.future_conf_lower = returns_to_prices(close_arr[-1], mean_fc - 1.96 * vol_fc)
                result.future_conf_upper = returns_to_prices(close_arr[-1], mean_fc + 1.96 * vol_fc)
        except Exception:
            result.future_predictions = np.array([])

        if callbacks:
            callbacks.on_model_end("EGARCH", result)
        results["EGARCH"] = result
        done += 1

    if callbacks:
        callbacks.on_training_complete(results)

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
