"""
价格预测 - 模型定义
LSTM / GRU / 1D-CNN / ARIMA / EGARCH
"""

import os
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """模型超参数"""
    look_back: int = 30
    n_features: int = 1
    epochs: int = 100
    batch_size: int = 32
    early_stop_patience: int = 10
    learning_rate: float = 0.001
    dropout: float = 0.2
    lstm_units: list = field(default_factory=lambda: [64, 32])
    gru_units: list = field(default_factory=lambda: [64, 32])
    cnn_filters: list = field(default_factory=lambda: [64, 32])
    cnn_kernel_size: int = 3


def _set_seed(seed=42):
    import tensorflow as tf
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_lstm(config: ModelConfig):
    """
    LSTM 模型:
    LSTM(64, return_sequences) → Dropout → LSTM(32) → Dropout → Dense(1)
    """
    _set_seed()
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input

    model = Sequential([
        Input(shape=(config.look_back, config.n_features)),
        LSTM(config.lstm_units[0], return_sequences=True),
        Dropout(config.dropout),
        LSTM(config.lstm_units[1]),
        Dropout(config.dropout),
        Dense(16, activation="relu"),
        Dense(1),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss="mse",
    )
    return model


def build_gru(config: ModelConfig):
    """
    GRU 模型: 与 LSTM 同构，替换为 GRU 层
    """
    _set_seed()
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import GRU, Dense, Dropout, Input

    model = Sequential([
        Input(shape=(config.look_back, config.n_features)),
        GRU(config.gru_units[0], return_sequences=True),
        Dropout(config.dropout),
        GRU(config.gru_units[1]),
        Dropout(config.dropout),
        Dense(16, activation="relu"),
        Dense(1),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss="mse",
    )
    return model


def build_cnn(config: ModelConfig):
    """
    1D-CNN 模型:
    Conv1D(64,3) → MaxPool → Conv1D(32,3) → MaxPool → Flatten → Dense(32) → Dense(1)
    """
    _set_seed()
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (
        Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Input
    )

    layers = [Input(shape=(config.look_back, config.n_features))]

    seq_len = config.look_back
    layers.append(Conv1D(config.cnn_filters[0], config.cnn_kernel_size,
                         activation="relu", padding="same"))
    if seq_len >= 4:
        layers.append(MaxPooling1D(pool_size=2))
        seq_len = seq_len // 2

    layers.append(Conv1D(config.cnn_filters[1], config.cnn_kernel_size,
                         activation="relu", padding="same"))
    if seq_len >= 4:
        layers.append(MaxPooling1D(pool_size=2))

    layers.extend([
        Flatten(),
        Dense(32, activation="relu"),
        Dropout(config.dropout),
        Dense(1),
    ])

    model = Sequential(layers)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss="mse",
    )
    return model


def fit_arima(train_data: np.ndarray, exog: np.ndarray = None, config: ModelConfig = None):
    """
    拟合 auto_arima
    train_data: 一维收盘价序列
    exog: 外生变量矩阵
    """
    import pmdarima as pm
    import warnings
    warnings.filterwarnings("ignore")

    max_p = 5
    max_q = 5
    max_d = 2
    if config:
        stats = getattr(config, "_stats", None)
        if stats:
            max_p = stats.get("arima_max_p", 5)
            max_q = stats.get("arima_max_q", 5)
            max_d = stats.get("arima_max_d", 2)

    model = pm.auto_arima(
        train_data,
        exogenous=exog,
        start_p=1, start_q=1,
        max_p=max_p, max_q=max_q, max_d=max_d,
        seasonal=False,
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore",
        n_jobs=-1,
    )
    return model


def predict_arima(model, steps: int, exog_future: np.ndarray = None):
    """
    ARIMA 多步预测
    返回: (predictions, confidence_intervals)
    """
    fc, conf = model.predict(n_periods=steps, exogenous=exog_future,
                             return_conf_int=True, alpha=0.05)
    return np.array(fc), np.array(conf)


def fit_egarch(returns: np.ndarray, config: ModelConfig = None):
    """
    拟合 EGARCH(1,1)
    returns: 收益率序列（百分比）
    使用 AR(1) 均值模型捕捉收益率自相关
    """
    from arch import arch_model
    import warnings
    warnings.filterwarnings("ignore")

    p, q = 1, 1
    am = arch_model(returns, vol="EGARCH", p=p, q=q, mean="AR", lags=1, dist="normal")
    result = am.fit(disp="off", show_warning=False)
    return result


def predict_egarch(model_result, steps: int):
    """
    EGARCH 预测
    返回: (mean_forecast, volatility_forecast)
    """
    forecast = model_result.forecast(horizon=steps)
    mean_fc = forecast.mean.iloc[-1].values
    var_fc = forecast.variance.iloc[-1].values
    vol_fc = np.sqrt(var_fc)
    return mean_fc, vol_fc


def returns_to_prices(last_price: float, predicted_returns: np.ndarray) -> np.ndarray:
    """从最后收盘价 + 预测收益率序列 → 价格序列"""
    prices = [last_price]
    for r in predicted_returns:
        prices.append(prices[-1] * (1 + r / 100))
    return np.array(prices[1:])
