"""
价格预测 - 训练与评估
时序交叉验证 + 模型训练 + 指标计算 + 集成预测
所有模型统一预测日收益率（目标列 index 0），再反算预测股价
"""

import time
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .features import (
    compute_technical_indicators, prepare_features,
    create_sequences, time_series_split, inverse_transform_predictions,
    create_tabular_features, create_tabular_targets,
    DEFAULT_FEATURE_COLS,
)
from .models import (
    ModelConfig, build_lstm, build_gru, build_cnn, build_cnn_gru,
    build_patchtst, build_tft,
    fit_arima, predict_arima,
    fit_sarima, predict_sarima,
    build_xgboost, build_lightgbm, returns_to_prices,
)
from .preprocessing import preprocess_data, returns_to_price_series


class TrainingCallbacks:
    """训练回调协议：实时监控接口"""

    def on_training_start(self, model_list: list) -> None:
        pass

    def on_model_start(self, model_name: str, model_index: int, total_models: int) -> None:
        pass

    def on_fold_start(self, model_name: str, fold: int, total_folds: int) -> None:
        pass

    def on_fold_end(self, model_name: str, fold: int, fold_metrics: dict) -> None:
        pass

    def on_epoch_end(self, model_name: str, epoch: int, total_epochs: int,
                     train_loss: float, val_loss: float, lr: float,
                     grad_norm: float = None) -> None:
        pass

    def on_early_stop(self, model_name: str, epoch: int, best_epoch: int) -> None:
        pass

    def on_model_end(self, model_name: str, result) -> None:
        pass

    def on_training_complete(self, all_results: dict) -> None:
        pass

    def on_overfitting_warning(self, model_name: str, epoch: int,
                                val_loss: float, best_val_loss: float) -> None:
        pass

    def on_log(self, message: str) -> None:
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
    test_returns: np.ndarray = field(default_factory=lambda: np.array([]))
    test_returns_actual: np.ndarray = field(default_factory=lambda: np.array([]))
    confidence_lower: np.ndarray = field(default_factory=lambda: np.array([]))
    confidence_upper: np.ndarray = field(default_factory=lambda: np.array([]))
    future_predictions: np.ndarray = field(default_factory=lambda: np.array([]))
    future_conf_lower: np.ndarray = field(default_factory=lambda: np.array([]))
    future_conf_upper: np.ndarray = field(default_factory=lambda: np.array([]))
    training_time: float = 0.0
    scaler: object = None
    feature_cols: list = field(default_factory=list)
    n_features: int = 0
    _last_close: float = 0.0  # 用于反算价格的基准价


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

    actual_epochs = len(history.history.get("loss", []))
    if callbacks and actual_epochs < config.epochs:
        best_epoch = actual_epochs - config.early_stop_patience
        callbacks.on_early_stop(model_name, actual_epochs, max(1, best_epoch))
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
            "CNN-GRU": build_cnn_gru, "PatchTST": build_patchtst, "TFT": build_tft}
    build_fn = builders[model_name]

    split_pt = int(len(X) * 0.85)
    X_train, X_test = X[:split_pt], X[split_pt:]
    y_train, y_test = y[:split_pt], y[split_pt:]

    if quick:
        # 快速模式: 跳过CV，直接训练一次
        final_model = build_fn(config)
        if callbacks:
            callbacks.on_log(f"[{model_name}] 快速模式 | 参数量: {final_model.count_params():,} | 训练集: {len(X_train)}, 测试集: {len(X_test)}")
        final_hist = _train_dl_model(final_model, X_train, y_train, X_test, y_test, config,
                                      model_name=model_name, callbacks=callbacks)
        result.train_history = final_hist
        result.model_object = final_model

        y_pred = final_model.predict(X_test, verbose=0).flatten()
        result.cv_metrics = calc_metrics(y_test, y_pred)
        result.test_predictions = y_pred
        result.test_actuals = y_test

        if callbacks:
            callbacks.on_log(f"[{model_name}] MC Dropout 置信区间估算中...")
        _, lower, upper = _mc_dropout_predict(final_model, X_test, n_iter=10)
        result.confidence_lower = lower
        result.confidence_upper = upper
    else:
        # 正常模式: 时序交叉验证
        splits = time_series_split(len(X), n_splits=n_splits)
        cv_scores = []

        if callbacks:
            callbacks.on_log(f"[{model_name}] 正常模式 | {n_splits}折交叉验证 | 训练集: {len(X_train)}, 测试集: {len(X_test)}")

        for fold_i, (train_idx, val_idx) in enumerate(splits):
            if callbacks:
                callbacks.on_fold_start(model_name, fold_i + 1, n_splits)
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_vl, y_vl = X[val_idx], y[val_idx]

            model = build_fn(config)
            hist = _train_dl_model(model, X_tr, y_tr, X_vl, y_vl, config,
                                   model_name=model_name, callbacks=callbacks)

            y_pred = model.predict(X_vl, verbose=0).flatten()
            fold_metrics = calc_metrics(y_vl, y_pred)
            cv_scores.append(fold_metrics)
            if callbacks:
                callbacks.on_fold_end(model_name, fold_i + 1, fold_metrics)

        result.cv_metrics = {
            k: np.mean([s[k] for s in cv_scores if not np.isnan(s[k])])
            for k in ["mae", "rmse", "mape", "r2"]
        }

        if callbacks:
            callbacks.on_log(
                f"[{model_name}] CV完成 | RMSE={result.cv_metrics['rmse']:.4f}, "
                f"R²={result.cv_metrics['r2']:.4f} | 开始全量训练...")

        # 全量训练最终模型
        final_model = build_fn(config)
        if callbacks:
            callbacks.on_log(f"[{model_name}] 最终模型参数量: {final_model.count_params():,}")
        final_hist = _train_dl_model(final_model, X_train, y_train, X_test, y_test, config,
                                      model_name=model_name, callbacks=callbacks)
        result.train_history = final_hist
        result.model_object = final_model

        # 测试集预测 + 置信区间
        if callbacks:
            callbacks.on_log(f"[{model_name}] MC Dropout 置信区间估算中...")
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


def train_garch_model(returns: np.ndarray, close_prices: np.ndarray,
                       config: ModelConfig, progress_callback=None) -> TrainResult:
    """训练 GARCH(1,1) 波动率模型"""
    import warnings
    warnings.filterwarnings("ignore")

    start = time.time()
    result = TrainResult(model_name="GARCH")

    from .volatility import fit_garch, predict_garch_volatility, compute_risk_metrics, historical_volatility_fallback

    if progress_callback:
        progress_callback(0.1, "GARCH: 拟合中...")

    split_pt = int(len(returns) * 0.85)
    train_ret = returns[:split_pt]
    test_ret = returns[split_pt:]
    test_close = close_prices[split_pt:]

    try:
        garch_res = fit_garch(train_ret, config.garch_p, config.garch_q, config.garch_dist)

        if garch_res is None:
            raise RuntimeError("GARCH 未收敛")

        result.model_object = garch_res

        # 预测波动率
        vol_info = predict_garch_volatility(garch_res, horizon=len(test_ret))
        pred_returns = vol_info['mean_forecast'][:len(test_ret)]

        # 存储收益率预测
        result.test_returns = pred_returns
        result.test_returns_actual = test_ret

        # 转为价格用于指标计算
        pred_prices = returns_to_prices(close_prices[split_pt - 1], pred_returns)
        result.test_predictions = pred_prices[:len(test_close)]
        result.test_actuals = test_close
        result.cv_metrics = calc_metrics(test_close, pred_prices[:len(test_close)])

        # 风险指标
        result.train_history['risk_metrics'] = compute_risk_metrics(train_ret, garch_res)

        # 置信区间
        conf_lower_ret = vol_info['confidence_lower'][:len(test_ret)]
        conf_upper_ret = vol_info['confidence_upper'][:len(test_ret)]
        result.confidence_lower = returns_to_prices(close_prices[split_pt - 1], conf_lower_ret)[:len(test_close)]
        result.confidence_upper = returns_to_prices(close_prices[split_pt - 1], conf_upper_ret)[:len(test_close)]

    except Exception as e:
        # GARCH 拟合失败，使用历史波动率替代
        if callable(progress_callback):
            progress_callback(0.5, "GARCH: 拟合失败，使用历史波动率")
        vol_info = historical_volatility_fallback(returns, horizon=len(test_ret))
        pred_returns = vol_info['mean_forecast'][:len(test_ret)]
        result.test_returns = pred_returns
        result.test_returns_actual = test_ret
        result.train_history = {"error": str(e), "fallback": "历史波动率"}
        pred_prices = returns_to_prices(close_prices[split_pt - 1], pred_returns)
        result.test_predictions = pred_prices[:len(test_close)]
        result.test_actuals = test_close
        result.cv_metrics = calc_metrics(test_close, pred_prices[:len(test_close)])

    if progress_callback:
        progress_callback(1.0, "GARCH: 完成")

    result.training_time = time.time() - start
    return result


def validate_training_data(df: pd.DataFrame, config: ModelConfig = None):
    """
    训练前数据质量检查。
    返回: (passed: bool, warnings: list, errors: list)
    """
    warnings = []
    errors = []

    min_rows_warn = 200
    if len(df) < min_rows_warn:
        msg = f"数据量仅 {len(df)} 天 (建议 >= {min_rows_warn})"
        warnings.append(msg)
    if len(df) < 50:
        errors.append(f"数据量仅 {len(df)} 天 (最低要求 50)")

    if "close" in df.columns:
        close = df["close"]
        close_na = close.isna().sum() / len(df)
        if close_na > 0.05:
            warnings.append(f"收盘价缺失率 {close_na:.1%} (偏高)")
        if close_na > 0.3:
            errors.append(f"收盘价缺失率 {close_na:.1%} (过高)")

        if (close < 0).any():
            errors.append(f"存在 {(close < 0).sum()} 个负收盘价")

        # ADF 平稳性检验
        if len(df) > 100:
            try:
                from statsmodels.tsa.stattools import adfuller
                adf_p = adfuller(close.dropna(), maxlag=10)[1]
                if adf_p > 0.05:
                    warnings.append(f"收盘价非平稳 (ADF p={adf_p:.3f})，趋势模型可能更合适")
            except Exception:
                pass

    passed = len(errors) == 0
    return passed, warnings, errors


# ── 树模型训练（XGBoost / LightGBM）─────────────────────────────────────

def _prepare_tabular_data(scaled_data, feature_cols, look_back):
    """准备表格特征，并分割训练/测试集"""
    X, feature_names = create_tabular_features(scaled_data, feature_cols, look_back)
    y = create_tabular_targets(scaled_data, look_back, 0)

    split_pt = int(len(X) * 0.85)
    return (X[:split_pt], X[split_pt:],
            y[:split_pt], y[split_pt:],
            feature_names)


def _bootstrap_ci(model, X, n_iter=100):
    """树模型置信区间：特征扰动法"""
    preds = []
    rng = np.random.RandomState(42)
    for _ in range(n_iter):
        noise = rng.normal(0, 0.01, X.shape)
        p = model.predict(X + noise)
        preds.append(p)
    preds = np.array(preds)
    mean = preds.mean(axis=0)
    std = preds.std(axis=0)
    return mean, mean - 1.96 * std, mean + 1.96 * std


def _predict_future_tree(model, last_features, steps, scaler, n_features,
                          feature_names):
    """
    树模型自回归多步预测。
    last_features: 1D array shape (n_features * look_back,)
    """
    predictions = []
    current = last_features.copy()
    n_lag_features = len(feature_names)

    for _ in range(steps):
        pred_scaled = model.predict(current.reshape(1, -1))[0]
        predictions.append(pred_scaled)

        # 窗口前移：去掉最早的 n_features 个值，把当前预测作为新的 target
        new_lag = current[-n_features:].copy()
        new_lag[0] = pred_scaled  # target_col=0 (close)
        current = np.concatenate([current[n_features:], new_lag])

    preds = np.array(predictions)
    return inverse_transform_predictions(preds, scaler, n_features, 0)


def train_tree_model(model_name, X_seq, y_seq, config, callbacks=None) -> TrainResult:
    """
    训练树模型（XGBoost / LightGBM）。
    X_seq, y_seq: 归一化后的 3D 序列和 1D 目标
    """
    start = time.time()
    result = TrainResult(model_name=model_name)

    builders = {"XGBoost": build_xgboost, "LightGBM": build_lightgbm}
    build_fn = builders.get(model_name)

    if not build_fn:
        result.train_history = {"error": f"未知树模型: {model_name}"}
        return result

    model = build_fn(config)

    if callbacks:
        callbacks.on_log(f"[{model_name}] 树模型训练 | 开始准备表格特征...")

    try:
        # 准备表格数据（用归一化后的特征）
        # 注意：这里需要 scaled_data 和 feature_cols，需要从外层传入
        # 暂时用占位符，实际在 train_all_models 中已准备好 scaled, feature_cols
        result.model_object = model
        result.train_history = {}
        result.cv_metrics = {"mae": np.nan, "rmse": np.nan, "mape": np.nan, "r2": np.nan}
    except Exception as e:
        result.train_history = {"error": str(e)}

    result.training_time = time.time() - start
    return result


def _train_tree_model_full(model, X_train, y_train, X_test, y_test,
                            config, model_name, callbacks) -> TrainResult:
    """树模型完整训练流程（含CV）"""
    start = time.time()
    result = TrainResult(model_name=model_name)

    if callbacks:
        callbacks.on_log(f"[{model_name}] 树模型 | 训练集: {len(X_train)}, 测试集: {len(X_test)}")

    # CV
    folds = time_series_split(len(X_train), n_splits=3)
    fold_metrics = []
    for fi, (tr_idx, vl_idx) in enumerate(folds):
        if callbacks:
            callbacks.on_fold_start(model_name, fi + 1, len(folds))
        X_fold_tr, X_fold_vl = X_train[tr_idx], X_train[vl_idx]
        y_fold_tr, y_fold_vl = y_train[tr_idx], y_train[vl_idx]

        m = build_xgboost(config) if model_name == "XGBoost" else build_lightgbm(config)
        m.fit(X_fold_tr, y_fold_tr)
        y_pred_vl = m.predict(X_fold_vl)
        fm = calc_metrics(y_fold_vl, y_pred_vl)
        fold_metrics.append(fm)
        if callbacks:
            callbacks.on_fold_end(model_name, fi + 1, fm)

    if fold_metrics:
        avg_metrics = {}
        for k in fold_metrics[0]:
            vals = [fm[k] for fm in fold_metrics if not np.isnan(fm[k])]
            avg_metrics[k] = np.mean(vals) if vals else np.nan
        result.cv_metrics = avg_metrics
    else:
        result.cv_metrics = {"mae": np.nan, "rmse": np.nan, "mape": np.nan, "r2": np.nan}

    if callbacks:
        callbacks.on_log(f"[{model_name}] CV完成 | MAE: {result.cv_metrics.get('mae', np.nan):.4f}")

    # 最终训练
    final_model = build_xgboost(config) if model_name == "XGBoost" else build_lightgbm(config)
    final_model.fit(X_train, y_train)
    result.model_object = final_model

    # 测试集评估
    y_pred = final_model.predict(X_test)
    test_metrics = calc_metrics(y_test, y_pred)
    result.test_predictions = y_pred
    result.test_actuals = y_test

    mean_ci, lower_ci, upper_ci = _bootstrap_ci(final_model, X_test)
    result.confidence_lower = lower_ci
    result.confidence_upper = upper_ci

    if callbacks:
        callbacks.on_log(f"[{model_name}] 测试 MAE: {test_metrics.get('mae', np.nan):.4f}")

    result.train_history = {}
    result.training_time = time.time() - start
    return result


# ── SARIMA 训练 ─────────────────────────────────────────────────

def train_sarima_model(close_prices, config, progress_callback=None) -> TrainResult:
    """
    训练 SARIMA 模型（滚动一步预测）
    与 train_arima_model 相同模式但使用 SARIMAX
    """
    import warnings
    warnings.filterwarnings("ignore")

    start = time.time()
    result = TrainResult(model_name="SARIMA")

    if progress_callback:
        progress_callback(0.1, "SARIMA: 拟合中...")

    split_pt = int(len(close_prices) * 0.85)
    train_close = close_prices[:split_pt]
    test_close = close_prices[split_pt:]

    try:
        model = fit_sarima(train_close, config)
        result.model_object = model

        # 滚动一步预测
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        history = list(train_close)
        pred_prices = []
        for i in range(len(test_close)):
            m = SARIMAX(history, order=config.sarima_order,
                        seasonal_order=config.sarima_seasonal_order,
                        trend='t', enforce_stationarity=False,
                        enforce_invertibility=False)
            res = m.fit(disp=False)
            fc = np.array(res.get_forecast(steps=1).predicted_mean)[0]
            pred_prices.append(fc)
            history.append(test_close[i])

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

    # ── 新增：收益率为目标的预处理 ─────────────────────
    df_ret = preprocess_data(df)
    last_close = float(df['close'].iloc[-1])

    df_ind = compute_technical_indicators(df_ret)
    scaled, scaler, feature_cols, cleaned_df = prepare_features(df_ind)

    # 训练前数据质量检查
    passed, val_warnings, val_errors = validate_training_data(cleaned_df, config)
    if callbacks:
        for w in val_warnings:
            callbacks.on_log(f"[数据质量] 警告: {w}")
        for e in val_errors:
            callbacks.on_log(f"[数据质量] 错误: {e}")

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

    results = {}
    dl_models = [m for m in selected_models if m in ("LSTM", "GRU", "1D-CNN", "CNN-GRU", "PatchTST", "TFT")]
    tree_models = [m for m in selected_models if m in ("XGBoost", "LightGBM")]
    stat_models = [m for m in selected_models if m in ("ARIMA", "GARCH", "SARIMA")]
    all_ordered = dl_models + tree_models + stat_models

    total = len(selected_models)
    done = 0

    if callbacks:
        callbacks.on_training_start(selected_models)
        mode_str = "快速模式" if quick else "正常模式"
        callbacks.on_log(
            f"数据准备完成 | {mode_str} | 样本数: {X.shape[0]}, 特征: {n_features}, "
            f"时间步: {config.look_back} | Epochs: {config.epochs}")

    for name in dl_models:
        if callbacks:
            callbacks.on_model_start(name, done, total)

        try:
            result = train_dl_model(name, X, y, config, n_splits=3, quick=quick, callbacks=callbacks)
            result.scaler = scaler
            result.feature_cols = feature_cols
            result.n_features = n_features
            result._last_close = last_close

            # 反归一化 → 收益率%（目标列 index 0 现在是 目标收益率）
            pred_returns = inverse_transform_predictions(
                result.test_predictions, scaler, n_features, 0)
            actual_returns = inverse_transform_predictions(
                result.test_actuals, scaler, n_features, 0)
            result.test_returns = pred_returns
            result.test_returns_actual = actual_returns

            # 收益率 → 价格（用于展示和指标）
            close_arr_full = cleaned_df["close"].values
            split_pt_price = int(len(close_arr_full) * 0.85)
            result.test_predictions = returns_to_price_series(
                close_arr_full[split_pt_price - 1], pred_returns)
            result.test_actuals = close_arr_full[split_pt_price:]
            result.confidence_lower = inverse_transform_predictions(
                result.confidence_lower, scaler, n_features, 0)
            result.confidence_upper = inverse_transform_predictions(
                result.confidence_upper, scaler, n_features, 0)
            # 置信区间也转价格
            conf_lower_p = returns_to_price_series(
                close_arr_full[split_pt_price - 1], result.confidence_lower)
            conf_upper_p = returns_to_price_series(
                close_arr_full[split_pt_price - 1], result.confidence_upper)
            result.confidence_lower = conf_lower_p
            result.confidence_upper = conf_upper_p

            # 重新算反归一化后的指标（基于价格）
            n_min = min(len(result.test_actuals), len(result.test_predictions))
            result.cv_metrics = calc_metrics(
                result.test_actuals[:n_min], result.test_predictions[:n_min])

            # 未来预测 → 收益率%
            last_seq = X[-1:].copy()
            future_preds = _predict_future_dl(result.model_object, last_seq, forecast_days,
                                               scaler, n_features, 0)
            result.future_predictions = future_preds  # 现在是收益率%

        except Exception as e:
            result = TrainResult(model_name=name)
            result.cv_metrics = {"mae": np.nan, "rmse": np.nan, "mape": np.nan, "r2": np.nan}
            result.train_history = {"error": str(e)}
            result.training_time = time.time()

        if callbacks:
            callbacks.on_model_end(name, result)
        results[name] = result
        done += 1

    # ── 树模型训练 ───────────────────────────────────
    for name in tree_models:
        if callbacks:
            callbacks.on_model_start(name, done, total)
            callbacks.on_log(f"[{name}] 表格特征: {n_features * config.look_back} 维")

        try:
            X_tab_train, X_tab_test, y_tab_train, y_tab_test, tab_feature_names = \
                _prepare_tabular_data(scaled, feature_cols, config.look_back)

            result = _train_tree_model_full(
                name, X_tab_train, y_tab_train, X_tab_test, y_tab_test,
                config=config, model_name=name, callbacks=callbacks)
            result.scaler = scaler
            result.feature_cols = tab_feature_names
            result.n_features = n_features
            result._last_close = last_close

            # 反归一化 → 收益率%
            pred_returns = inverse_transform_predictions(
                result.test_predictions, scaler, n_features, 0)
            actual_returns = inverse_transform_predictions(
                result.test_actuals, scaler, n_features, 0)
            result.test_returns = pred_returns
            result.test_returns_actual = actual_returns

            # 收益率 → 价格
            close_arr_full = cleaned_df["close"].values
            split_pt_price = int(len(close_arr_full) * 0.85)
            result.test_predictions = returns_to_price_series(
                close_arr_full[split_pt_price - 1], pred_returns)
            result.test_actuals = close_arr_full[split_pt_price:]
            conf_lower_ret = inverse_transform_predictions(
                result.confidence_lower, scaler, n_features, 0)
            conf_upper_ret = inverse_transform_predictions(
                result.confidence_upper, scaler, n_features, 0)
            result.confidence_lower = returns_to_price_series(
                close_arr_full[split_pt_price - 1], conf_lower_ret)
            result.confidence_upper = returns_to_price_series(
                close_arr_full[split_pt_price - 1], conf_upper_ret)

            n_min = min(len(result.test_actuals), len(result.test_predictions))
            result.cv_metrics = calc_metrics(
                result.test_actuals[:n_min], result.test_predictions[:n_min])

            # 未来预测 → 收益率%
            last_tab_features = X_tab_train[-1].copy()
            future_preds = _predict_future_tree(
                result.model_object, last_tab_features, forecast_days,
                scaler, n_features, tab_feature_names)
            result.future_predictions = future_preds  # 现在是收益率%

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
            callbacks.on_log(f"[ARIMA] 滚动一步预测 | 数据: {len(close_arr)} 天")
        result = train_arima_model(close_arr, config, progress_callback=None)
        result.scaler = scaler
        result.feature_cols = feature_cols
        result.n_features = n_features
        result._last_close = last_close

        try:
            if result.model_object is not None:
                fc, conf = predict_arima(result.model_object, steps=forecast_days)
                # ARIMA 预测价格 → 转为收益率供集成使用
                arima_returns = (fc / close_arr[-1] - 1) * 100
                result.future_predictions = arima_returns
                conf_low_ret = (conf[:, 0] / close_arr[-1] - 1) * 100
                conf_up_ret = (conf[:, 1] / close_arr[-1] - 1) * 100
                result.future_conf_lower = conf_low_ret
                result.future_conf_upper = conf_up_ret
        except Exception:
            result.future_predictions = np.array([])

        if callbacks:
            callbacks.on_model_end("ARIMA", result)
        results["ARIMA"] = result
        done += 1

    if "GARCH" in stat_models:
        if callbacks:
            callbacks.on_model_start("GARCH", done, total)
            callbacks.on_log(f"[GARCH] 波动率建模 | 收益率序列: {len(df_ret['日收益率'].dropna())} 天")
        garch_returns = df_ret['日收益率'].dropna().values
        result = train_garch_model(garch_returns, close_arr, config, progress_callback=None)
        result.scaler = scaler
        result.feature_cols = feature_cols
        result.n_features = n_features
        result._last_close = last_close

        try:
            from .volatility import predict_garch_volatility, historical_volatility_fallback
            if result.model_object is not None:
                vol_info = predict_garch_volatility(result.model_object, steps=forecast_days)
            else:
                vol_info = historical_volatility_fallback(garch_returns, horizon=forecast_days)
            result.future_predictions = vol_info['mean_forecast']
            result.future_conf_lower = vol_info['confidence_lower']
            result.future_conf_upper = vol_info['confidence_upper']
        except Exception:
            result.future_predictions = np.array([])

        if callbacks:
            callbacks.on_model_end("GARCH", result)
        results["GARCH"] = result
        done += 1

    if "SARIMA" in stat_models:
        if callbacks:
            callbacks.on_model_start("SARIMA", done, total)
            callbacks.on_log(f"[SARIMA] 季节性ARIMA | 数据: {len(close_arr)} 天")
        result = train_sarima_model(close_arr, config)
        result.scaler = scaler
        result.feature_cols = feature_cols
        result.n_features = n_features
        result._last_close = last_close

        try:
            if result.model_object is not None:
                fc, conf = predict_sarima(result.model_object, steps=forecast_days)
                # SARIMA 预测价格 → 转为收益率供集成使用
                sarima_returns = (fc / close_arr[-1] - 1) * 100
                result.future_predictions = sarima_returns
                conf_low_ret = (conf[:, 0] / close_arr[-1] - 1) * 100
                conf_up_ret = (conf[:, 1] / close_arr[-1] - 1) * 100
                result.future_conf_lower = conf_low_ret
                result.future_conf_upper = conf_up_ret
        except Exception:
            result.future_predictions = np.array([])

        if callbacks:
            callbacks.on_model_end("SARIMA", result)
        results["SARIMA"] = result
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
    集成预测未来 N 天。
    输入：各模型的 future_predictions 为收益率%数组
    输出：包含价格预测和收益率各项的字典
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

    # 加权平均收益率%
    weighted_sum = np.zeros(forecast_days)
    total_weight = 0
    for name, preds in model_preds.items():
        w = weights.get(name, 0)
        if w > 0:
            weighted_sum += preds * w
            total_weight += w

    if total_weight > 0:
        ensemble_returns = weighted_sum / total_weight
    else:
        ensemble_returns = np.mean(list(model_preds.values()), axis=0)

    # 收益率% → 价格
    ensemble_prices = returns_to_price_series(last_price, ensemble_returns)

    # 日收益率和累计收益（从价格计算）
    price_series = np.concatenate([[last_price], ensemble_prices])
    daily_ret = np.diff(price_series) / price_series[:-1] * 100
    cum_ret = (ensemble_prices / last_price - 1) * 100

    # 置信区间：收集各模型的 future_conf，转价格
    all_lowers, all_uppers = [], []
    for name, result in results.items():
        if len(result.future_conf_lower) >= forecast_days and len(result.future_conf_upper) >= forecast_days:
            # 置信区间也可能是收益率%，需要转为价格
            conf_l = result.future_conf_lower[:forecast_days]
            conf_u = result.future_conf_upper[:forecast_days]
            # 判断是否为收益率（值在 ±20 以内）还是价格
            if np.max(np.abs(conf_l)) < 50 and np.max(np.abs(conf_u)) < 50:
                # 收益率%，转价格
                conf_l = returns_to_price_series(last_price, conf_l)
                conf_u = returns_to_price_series(last_price, conf_u)
            all_lowers.append(conf_l)
            all_uppers.append(conf_u)

    conf_lower = np.min(all_lowers, axis=0) if all_lowers else ensemble_prices * 0.95
    conf_upper = np.max(all_uppers, axis=0) if all_uppers else ensemble_prices * 1.05

    return {
        "predicted_close": ensemble_prices,
        "daily_return": daily_ret,
        "cumulative_return": cum_ret,
        "confidence_lower": conf_lower,
        "confidence_upper": conf_upper,
        "model_predictions": model_preds,  # 各模型的收益率%预测
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
