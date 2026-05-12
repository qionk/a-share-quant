"""
价格预测 - 特征工程
技术指标计算 + 数据归一化 + 滑动窗口
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算技术指标，原地添加列
    输入需包含: open, high, low, close, volume
    """
    df = df.copy()
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # 均线
    for w in [5, 10, 20, 60]:
        df[f"ma{w}"] = close.rolling(w).mean()

    # MACD (12, 26, 9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["dif"] = ema12 - ema26
    df["dea"] = df["dif"].ewm(span=9, adjust=False).mean()
    df["macd"] = (df["dif"] - df["dea"]) * 2

    # RSI(14)
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi"] = 100 - 100 / (1 + rs)

    # 布林带 (20, 2)
    ma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    df["boll_upper"] = ma20 + 2 * std20
    df["boll_mid"] = ma20
    df["boll_lower"] = ma20 - 2 * std20

    # 涨跌幅（若不存在则计算）
    if "pct_change" not in df.columns or df["pct_change"].isna().all():
        df["pct_change"] = close.pct_change()

    # 成交量均线
    df["vol_ma5"] = volume.rolling(5).mean()
    df["vol_ma20"] = volume.rolling(20).mean()

    # OBV (On-Balance Volume)
    df["obv"] = (np.sign(close.diff()) * volume).fillna(0).cumsum()

    # 量比 (volume / vol_ma20)
    df["vol_ratio"] = volume / df["vol_ma20"].replace(0, np.nan)

    # VWAP 近似
    df["vwap"] = (close * volume).cumsum() / volume.cumsum()

    # 量价相关性 (20日滚动)
    df["vol_price_corr"] = volume.rolling(20).corr(close)

    # 成交量动量 (5日变化率)
    df["vol_momentum"] = volume.pct_change(5)

    # 日收益率5日均线（平滑后的收益率趋势）
    if '日收益率' in df.columns:
        df['日收益率_ma5'] = df['日收益率'].rolling(5).mean()

    return df


# 默认特征列（训练用）
DEFAULT_FEATURE_COLS = [
    "目标收益率",  # index 0 = 预测目标
    "close", "open", "high", "low", "volume",
    "ma5", "ma10", "ma20", "ma60",
    "dif", "dea", "macd", "rsi",
    "boll_upper", "boll_mid", "boll_lower",
    "pct_change", "vol_ma5", "vol_ma20",
    "obv", "vol_ratio", "vwap", "vol_price_corr", "vol_momentum",
    "日收益率", "日收益率_ma5",
    "成交量变化率", "相对成交量", "量价配合度",
    "放量上涨", "缩量下跌",
]


def prepare_features(df: pd.DataFrame, feature_cols: list = None):
    """
    准备特征矩阵：选列 → 去NaN → MinMaxScaler 归一化
    返回: (scaled_array, scaler, feature_cols, cleaned_df)
    """
    if feature_cols is None:
        feature_cols = [c for c in DEFAULT_FEATURE_COLS if c in df.columns]

    data = df[feature_cols].copy()
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna()

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data.values)

    return scaled, scaler, feature_cols, data


def create_sequences(data: np.ndarray, look_back: int, target_col_idx: int = 0):
    """
    滑动窗口创建训练序列
    data: shape (n_samples, n_features)
    返回: X (n_seq, look_back, n_features), y (n_seq,)
    """
    X, y = [], []
    for i in range(look_back, len(data)):
        X.append(data[i - look_back:i])
        y.append(data[i, target_col_idx])
    return np.array(X), np.array(y)


def time_series_split(n_samples: int, n_splits: int = 5, min_train_size: int = 100):
    """
    时间序列交叉验证划分（前向扩展）
    返回: [(train_indices, val_indices), ...]
    """
    splits = []
    fold_size = max(1, (n_samples - min_train_size) // (n_splits + 1))

    for i in range(n_splits):
        train_end = min_train_size + fold_size * (i + 1)
        val_end = min(train_end + fold_size, n_samples)
        if train_end >= n_samples or val_end <= train_end:
            break
        train_idx = np.arange(0, train_end)
        val_idx = np.arange(train_end, val_end)
        splits.append((train_idx, val_idx))

    if not splits:
        split_point = int(n_samples * 0.8)
        splits.append((np.arange(0, split_point), np.arange(split_point, n_samples)))

    return splits


def inverse_transform_predictions(predictions: np.ndarray, scaler: MinMaxScaler,
                                   n_features: int, target_col_idx: int = 0) -> np.ndarray:
    """反归一化预测值回原始尺度"""
    dummy = np.zeros((len(predictions), n_features))
    dummy[:, target_col_idx] = predictions.flatten()
    inversed = scaler.inverse_transform(dummy)
    return inversed[:, target_col_idx]


def create_tabular_features(scaled_data: np.ndarray, feature_cols: list,
                             look_back: int) -> tuple:
    """
    将3D序列数据展平为2D表格特征（用于XGBoost/LightGBM等树模型）。
    每个时间步的特征都作为独立的lag列。
    返回: (X_2D, feature_names)
    """
    n_features = len(feature_cols)
    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        row = scaled_data[i - look_back:i].flatten()
        X.append(row)
    X = np.array(X)

    feature_names = []
    for col in feature_cols:
        for lag in range(look_back, 0, -1):
            feature_names.append(f"{col}_lag{lag}")

    return X, feature_names


def create_tabular_targets(scaled_data: np.ndarray, look_back: int,
                            target_col_idx: int = 0) -> np.ndarray:
    """提取表格数据的目标值（归一化后的close价格）"""
    return scaled_data[look_back:, target_col_idx]
