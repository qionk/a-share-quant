"""
价格预测 - 模型定义
LSTM / GRU / 1D-CNN / CNN-GRU / PatchTST / TFT /
XGBoost / LightGBM / ARIMA / SARIMA / GARCH
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
    # PatchTST
    patchtst_patch_size: int = 16
    patchtst_d_model: int = 128
    patchtst_n_heads: int = 4
    patchtst_n_encoder_layers: int = 2
    patchtst_ff_dim: int = 256
    patchtst_dropout: float = 0.1
    # TFT
    tft_hidden_size: int = 64
    tft_n_heads: int = 4
    tft_dropout: float = 0.2
    tft_lstm_layers: int = 1
    # CNN-GRU
    cnn_gru_filters: list = field(default_factory=lambda: [64, 32])
    cnn_gru_gru_units: list = field(default_factory=lambda: [64, 32])
    cnn_gru_kernel_size: int = 3
    # XGBoost
    xgboost_n_estimators: int = 100
    xgboost_max_depth: int = 6
    xgboost_learning_rate: float = 0.1
    xgboost_subsample: float = 0.8
    # LightGBM
    lightgbm_n_estimators: int = 100
    lightgbm_max_depth: int = 6
    lightgbm_learning_rate: float = 0.1
    lightgbm_num_leaves: int = 31
    # SARIMA
    sarima_order: tuple = (1, 1, 1)
    sarima_seasonal_order: tuple = (1, 1, 1, 5)
    # GARCH
    garch_p: int = 1
    garch_q: int = 1
    garch_dist: str = 't'


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


def build_cnn_gru(config: ModelConfig):
    """
    CNN-GRU 混合模型:
    Conv1D → MaxPool → Conv1D → MaxPool → GRU → Dropout → GRU → Dropout → Dense(1)
    """
    _set_seed()
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (
        Conv1D, MaxPooling1D, GRU, Dense, Dropout, Input
    )

    seq_len = config.look_back
    layers = [Input(shape=(seq_len, config.n_features))]
    layers.append(Conv1D(config.cnn_gru_filters[0], config.cnn_gru_kernel_size,
                         activation="relu", padding="same"))
    if seq_len >= 4:
        layers.append(MaxPooling1D(pool_size=2))
        seq_len //= 2

    layers.append(Conv1D(config.cnn_gru_filters[1], config.cnn_gru_kernel_size,
                         activation="relu", padding="same"))
    if seq_len >= 4:
        layers.append(MaxPooling1D(pool_size=2))

    layers.append(GRU(config.cnn_gru_gru_units[0], return_sequences=True))
    layers.append(Dropout(config.dropout))
    layers.append(GRU(config.cnn_gru_gru_units[1]))
    layers.append(Dropout(config.dropout))
    layers.append(Dense(16, activation="relu"))
    layers.append(Dense(1))

    model = Sequential(layers)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss="mse",
    )
    return model


def build_patchtst(config: ModelConfig):
    """
    PatchTST: Patch + Transformer Encoder + Linear Head
    """
    _set_seed()
    import tensorflow as tf
    from tensorflow.keras import layers, Model

    look_back = config.look_back
    n_features = config.n_features
    patch_size = config.patchtst_patch_size
    d_model = config.patchtst_d_model
    n_heads = config.patchtst_n_heads
    n_layers = config.patchtst_n_encoder_layers
    ff_dim = config.patchtst_ff_dim
    drop = config.patchtst_dropout

    while look_back // patch_size < 3 and patch_size > 4:
        patch_size = patch_size // 2

    n_patches = (look_back + patch_size - 1) // patch_size
    pad_len = n_patches * patch_size - look_back

    inputs = layers.Input(shape=(look_back, n_features))

    x = inputs
    if pad_len > 0:
        x = layers.ZeroPadding1D(padding=(pad_len, 0))(x)

    x = layers.Reshape((n_patches, patch_size * n_features))(x)
    x = layers.Dense(d_model)(x)

    pos_emb = layers.Embedding(n_patches, d_model)
    positions = tf.range(n_patches)
    x = x + pos_emb(positions)

    for _ in range(n_layers):
        attn_out = layers.MultiHeadAttention(
            num_heads=n_heads, key_dim=d_model // n_heads, dropout=drop
        )(x, x)
        attn_out = layers.Dropout(drop)(attn_out)
        x = layers.LayerNormalization(epsilon=1e-6)(x + attn_out)

        ff_out = layers.Dense(ff_dim, activation="gelu")(x)
        ff_out = layers.Dropout(drop)(ff_out)
        ff_out = layers.Dense(d_model)(ff_out)
        ff_out = layers.Dropout(drop)(ff_out)
        x = layers.LayerNormalization(epsilon=1e-6)(x + ff_out)

    x = layers.Flatten()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(drop)(x)
    outputs = layers.Dense(1)(x)

    model = Model(inputs=inputs, outputs=outputs, name="PatchTST")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss="mse",
    )
    return model


def build_tft(config: ModelConfig):
    """
    Temporal Fusion Transformer (简化版):
    Variable Selection → LSTM Encoder → GRN → Temporal Attention → Output
    """
    _set_seed()
    import tensorflow as tf
    from tensorflow.keras import layers, Model

    look_back = config.look_back
    n_features = config.n_features
    hidden = config.tft_hidden_size
    n_heads = config.tft_n_heads
    drop = config.tft_dropout
    n_lstm = config.tft_lstm_layers

    inputs = layers.Input(shape=(look_back, n_features))

    # Variable Selection Network
    context = layers.Flatten()(inputs)
    context = layers.Dense(hidden, activation="relu")(context)
    var_weights = layers.Dense(n_features, activation="softmax",
                               name="variable_weights")(context)

    # Per-feature projection + weighted sum
    split_proj = []
    for i in range(n_features):
        feat_slice = layers.Lambda(lambda x, idx=i: x[:, :, idx:idx+1])(inputs)
        feat_proj = layers.Dense(hidden)(feat_slice)
        split_proj.append(feat_proj)

    stacked = layers.Lambda(lambda x: tf.stack(x, axis=2))(split_proj)

    var_w_expanded = layers.Lambda(
        lambda x: tf.expand_dims(tf.expand_dims(x, 1), -1)
    )(var_weights)

    weighted = layers.Multiply()([stacked, var_w_expanded])
    selected = layers.Lambda(lambda x: tf.reduce_sum(x, axis=2))(weighted)

    # LSTM Encoder
    lstm_out = selected
    for i in range(n_lstm):
        lstm_out = layers.LSTM(hidden, return_sequences=True,
                               dropout=drop, name=f"tft_lstm_{i}")(lstm_out)

    # Gated Residual Network
    grn_h = layers.Dense(hidden, activation="elu")(lstm_out)
    grn_h = layers.Dense(hidden)(grn_h)
    grn_h = layers.Dropout(drop)(grn_h)
    gate = layers.Dense(hidden, activation="sigmoid")(lstm_out)
    grn_out = layers.Multiply()([gate, grn_h])
    skip = layers.Lambda(lambda x: (1 - x[0]) * x[1])([gate, lstm_out])
    grn_out = layers.Add()([grn_out, skip])
    grn_out = layers.LayerNormalization()(grn_out)

    # Temporal Self-Attention with static enrichment
    static_ctx = layers.GlobalAveragePooling1D()(lstm_out)
    static_ctx = layers.RepeatVector(look_back)(static_ctx)
    enriched = layers.Add()([grn_out, static_ctx])

    attn_out = layers.MultiHeadAttention(
        num_heads=n_heads, key_dim=hidden // n_heads, dropout=drop
    )(enriched, enriched)
    attn_out = layers.Dropout(drop)(attn_out)
    attn_out = layers.Add()([attn_out, enriched])
    attn_out = layers.LayerNormalization()(attn_out)

    # Output
    last_step = layers.Lambda(lambda x: x[:, -1, :])(attn_out)
    out = layers.Dense(hidden // 2, activation="relu")(last_step)
    out = layers.Dropout(drop)(out)
    outputs = layers.Dense(1)(out)

    model = Model(inputs=inputs, outputs=outputs, name="TFT")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss="mse",
    )
    return model


def fit_arima(train_data: np.ndarray, exog: np.ndarray = None, config: ModelConfig = None):
    """
    拟合 ARIMA(1,1,1) + 常数漂移
    train_data: 一维收盘价序列
    """
    from statsmodels.tsa.arima.model import ARIMA
    import warnings
    warnings.filterwarnings("ignore")

    model = ARIMA(train_data, order=(1, 1, 1), exog=exog, trend='t')
    result = model.fit()
    return result


def predict_arima(model, steps: int, exog_future: np.ndarray = None):
    """
    ARIMA 多步预测
    返回: (predictions, confidence_intervals)
    """
    forecast = model.get_forecast(steps=steps, exog=exog_future)
    fc = np.array(forecast.predicted_mean)
    conf = np.array(forecast.conf_int(alpha=0.05))
    return fc, conf


def returns_to_prices(last_price: float, predicted_returns: np.ndarray) -> np.ndarray:
    """从最后收盘价 + 预测收益率序列 → 价格序列"""
    prices = [last_price]
    for r in predicted_returns:
        prices.append(prices[-1] * (1 + r / 100))
    return np.array(prices[1:])


def build_xgboost(config: ModelConfig):
    """XGBoost 回归模型"""
    import xgboost as xgb
    return xgb.XGBRegressor(
        n_estimators=config.xgboost_n_estimators,
        max_depth=config.xgboost_max_depth,
        learning_rate=config.xgboost_learning_rate,
        subsample=config.xgboost_subsample,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )


def build_lightgbm(config: ModelConfig):
    """LightGBM 回归模型"""
    import lightgbm as lgb
    return lgb.LGBMRegressor(
        n_estimators=config.lightgbm_n_estimators,
        max_depth=config.lightgbm_max_depth,
        learning_rate=config.lightgbm_learning_rate,
        num_leaves=config.lightgbm_num_leaves,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )


def fit_sarima(train_data: np.ndarray, config: ModelConfig = None):
    """
    拟合 SARIMA 模型
    train_data: 一维收盘价序列
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    import warnings
    warnings.filterwarnings("ignore")

    order = config.sarima_order if config else (1, 1, 1)
    seasonal_order = config.sarima_seasonal_order if config else (1, 1, 1, 5)

    model = SARIMAX(train_data, order=order, seasonal_order=seasonal_order,
                    trend='t', enforce_stationarity=False, enforce_invertibility=False)
    result = model.fit(disp=False)
    return result


def predict_sarima(model, steps: int):
    """
    SARIMA 多步预测
    返回: (predictions, confidence_intervals)
    """
    forecast = model.get_forecast(steps=steps)
    fc = np.array(forecast.predicted_mean)
    conf = np.array(forecast.conf_int(alpha=0.05))
    return fc, conf
