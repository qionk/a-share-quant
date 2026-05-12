"""
A股价格预测工具
==============
功能: 11种模型(LSTM/GRU/1D-CNN/CNN-GRU/PatchTST/TFT/XGBoost/LightGBM/ARIMA/SARIMA/GARCH)预测A股个股未来收盘价
启动: streamlit run predict_app.py
依赖: pip install -r requirements.txt
"""

import os
import sys
import io
import json
import time as _time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.predict.data_input import (
    load_from_akshare, load_from_excel, generate_template, get_stock_name,
)
from src.predict.features import compute_technical_indicators, prepare_features, create_sequences
from src.predict.models import ModelConfig
from src.predict.training import (
    train_all_models, compute_ensemble_weights, ensemble_predict,
    calc_metrics, backtest_predictions, TrainingCallbacks,
    validate_training_data,
)
from src.predict.model_store import save_model, list_models, delete_model, load_model
from src.predict.continuous import (
    rolling_train, track_performance, should_retrain, get_model_status, cleanup_old_models,
)
from src.predict.mysql_store import (
    is_configured as cloud_configured,
    save_training_results as cloud_save,
    load_by_session_id as cloud_load_session,
    list_available_stocks as cloud_list_stocks,
    restore_to_session_state as cloud_restore,
)
from src.predict.stock_data_store import (
    list_db_stocks, load_stock_from_db, fetch_and_store, has_stock_data,
    list_stocks_with_status, list_stock_sessions,
)
from src.predict.price_limits import (
    detect_price_limit_pct, get_board_name, apply_price_limits,
)
from src.predict.fibonacci_wave import (
    detect_wave_levels, calculate_wave_fibonacci, generate_wave_fib_signals,
)
from src.predict.long_term_prediction import (
    resample_to_weekly, train_long_term_models,
    assess_risk, get_rating,
)
from src.data import load_config

# ═══════ 页面配置 ═══════

st.set_page_config(page_title="A股价格预测", page_icon="📈", layout="wide")
st.title("A股价格预测工具")
st.caption("支持 LSTM / GRU / 1D-CNN / CNN-GRU / PatchTST / TFT / XGBoost / LightGBM / ARIMA / SARIMA / GARCH 多模型集成预测")

config = load_config()
predict_cfg = config.get("predict", {})


def _serialize_results() -> bytes:
    """将训练结果序列化为 JSON（不含模型对象，便于下载保存）"""
    data = {
        "stock_code": st.session_state.stock_code,
        "stock_name": st.session_state.stock_name,
        "ensemble_weights": st.session_state.ensemble_weights,
        "save_time": datetime.now().isoformat(),
    }
    if st.session_state.predictions:
        preds = st.session_state.predictions
        data["predictions"] = {
            k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in preds.items()
            if k != "model_predictions"
        }
        if preds.get("model_predictions"):
            data["predictions"]["model_predictions"] = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in preds["model_predictions"].items()
            }

    if st.session_state.train_results:
        data["train_results"] = {}
        for name, r in st.session_state.train_results.items():
            data["train_results"][name] = {
                "model_name": r.model_name,
                "cv_metrics": r.cv_metrics,
                "training_time": r.training_time,
                "test_predictions": r.test_predictions.tolist() if hasattr(r.test_predictions, 'tolist') else [],
                "test_actuals": r.test_actuals.tolist() if hasattr(r.test_actuals, 'tolist') else [],
                "test_returns": r.test_returns.tolist() if hasattr(r.test_returns, 'tolist') else [],
                "test_returns_actual": r.test_returns_actual.tolist() if hasattr(r.test_returns_actual, 'tolist') else [],
                "_last_close": r._last_close,
                "confidence_lower": r.confidence_lower.tolist() if hasattr(r.confidence_lower, 'tolist') else [],
                "confidence_upper": r.confidence_upper.tolist() if hasattr(r.confidence_upper, 'tolist') else [],
                "future_predictions": r.future_predictions.tolist() if hasattr(r.future_predictions, 'tolist') else [],
                "future_conf_lower": r.future_conf_lower.tolist() if hasattr(r.future_conf_lower, 'tolist') else [],
                "future_conf_upper": r.future_conf_upper.tolist() if hasattr(r.future_conf_upper, 'tolist') else [],
                "train_history": r.train_history,
                "feature_cols": r.feature_cols,
                "n_features": r.n_features,
            }

    if st.session_state.stock_data is not None:
        df = st.session_state.stock_data.copy()
        df.index = df.index.strftime("%Y-%m-%d")
        data["stock_data"] = df.to_dict(orient="split")

    return json.dumps(data, ensure_ascii=False, indent=2, default=str).encode("utf-8")


def _deserialize_results(content: bytes):
    """从 JSON 恢复训练结果到 session_state"""
    from src.predict.training import TrainResult
    data = json.loads(content.decode("utf-8"))

    st.session_state.stock_code = data.get("stock_code")
    st.session_state.stock_name = data.get("stock_name")
    st.session_state.ensemble_weights = data.get("ensemble_weights")

    if "stock_data" in data:
        sd = data["stock_data"]
        df = pd.DataFrame(sd["data"], columns=sd["columns"], index=sd["index"])
        df.index = pd.to_datetime(df.index)
        df.index.name = "date"
        st.session_state.stock_data = df

    if "train_results" in data:
        results = {}
        for name, rd in data["train_results"].items():
            tr = TrainResult(model_name=rd["model_name"])
            tr.cv_metrics = rd.get("cv_metrics", {})
            tr.training_time = rd.get("training_time", 0)
            tr.test_predictions = np.array(rd.get("test_predictions", []))
            tr.test_actuals = np.array(rd.get("test_actuals", []))
            tr.test_returns = np.array(rd.get("test_returns", []))
            tr.test_returns_actual = np.array(rd.get("test_returns_actual", []))
            tr._last_close = rd.get("_last_close", 0.0)
            tr.confidence_lower = np.array(rd.get("confidence_lower", []))
            tr.confidence_upper = np.array(rd.get("confidence_upper", []))
            tr.future_predictions = np.array(rd.get("future_predictions", []))
            tr.future_conf_lower = np.array(rd.get("future_conf_lower", []))
            tr.future_conf_upper = np.array(rd.get("future_conf_upper", []))
            tr.train_history = rd.get("train_history", {})
            tr.feature_cols = rd.get("feature_cols", [])
            tr.n_features = rd.get("n_features", 0)
            results[name] = tr
        st.session_state.train_results = results

    if "predictions" in data:
        preds = data["predictions"]
        restored = {}
        for k, v in preds.items():
            if k == "model_predictions":
                restored[k] = {mk: np.array(mv) for mk, mv in v.items()}
            elif isinstance(v, list):
                restored[k] = np.array(v)
            else:
                restored[k] = v
        st.session_state.predictions = restored

# ═══════ Session State 初始化 ═══════

TRAINING_LOCK = os.path.join(ROOT, "models", ".training_lock.json")

for key in ["stock_data", "stock_code", "stock_name", "train_results",
            "ensemble_weights", "predictions", "training_active"]:
    if key not in st.session_state:
        st.session_state[key] = None
if "training_active" not in st.session_state:
    st.session_state.training_active = False
if "cloud_stocks_cache" not in st.session_state:
    st.session_state.cloud_stocks_cache = None
    st.session_state.cloud_stocks_ts = 0
if "db_stocks" not in st.session_state:
    st.session_state.db_stocks = []

# ── 模型参数默认值（供 st.dialog 弹窗使用） ──
DEFAULT_MODEL_PARAMS = {
    "XGBoost": {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1, "subsample": 0.8},
    "LightGBM": {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1, "num_leaves": 31, "subsample": 0.8},
    "1D-CNN": {"look_back": 30, "filters": 32, "kernel_size": 3, "dropout": 0.2, "learning_rate": 0.001},
    "CNN-GRU": {"cnn_filters": 64, "kernel_size": 3, "gru_units": 32, "dropout": 0.2, "learning_rate": 0.001},
    "GRU": {"units": 32, "look_back": 30, "dropout": 0.2, "learning_rate": 0.001},
    "LSTM": {"units": 32, "look_back": 30, "dropout": 0.2, "learning_rate": 0.001},
    "PatchTST": {"d_model": 128, "n_heads": 4, "n_layers": 2, "patch_size": 16, "dropout": 0.1},
    "TFT": {"hidden_size": 64, "n_heads": 4, "dropout": 0.2, "lstm_layers": 1},
    "ARIMA": {"auto": True, "p": 1, "d": 1, "q": 1},
    "SARIMA": {"p": 1, "d": 1, "q": 1, "P": 1, "D": 1, "Q": 1, "s": 5},
    "GARCH": {"p": 1, "q": 1, "dist": "t"},
}
DL_LEARNING_RATE = 0.001
DL_EPOCHS = 100
DL_BATCH_SIZE = 32

if "model_params" not in st.session_state:
    st.session_state.model_params = {k: dict(v) for k, v in DEFAULT_MODEL_PARAMS.items()}
if "modified_models" not in st.session_state:
    st.session_state.modified_models = set()
if "dl_epochs" not in st.session_state:
    st.session_state.dl_epochs = DL_EPOCHS
if "dl_batch_size" not in st.session_state:
    st.session_state.dl_batch_size = DL_BATCH_SIZE
if "dl_learning_rate" not in st.session_state:
    st.session_state.dl_learning_rate = DL_LEARNING_RATE
if "longterm_results" not in st.session_state:
    st.session_state.longterm_results = None


def _param_changed(model_name, key, value, default_val):
    """检查参数是否被修改，更新 modified_models"""
    if value != default_val:
        st.session_state.modified_models.add(model_name)
    else:
        # 检查所有参数是否都恢复默认
        all_default = all(
            st.session_state.model_params[model_name].get(k) == DEFAULT_MODEL_PARAMS[model_name].get(k)
            for k in DEFAULT_MODEL_PARAMS[model_name]
        )
        if all_default:
            st.session_state.modified_models.discard(model_name)


def _training_lock_read():
    if os.path.exists(TRAINING_LOCK):
        import json as _json
        with open(TRAINING_LOCK) as f:
            return _json.load(f)
    return None


def _training_lock_write(data):
    os.makedirs(os.path.dirname(TRAINING_LOCK), exist_ok=True)
    import json as _json
    with open(TRAINING_LOCK, "w") as f:
        _json.dump(data, f)


def _training_lock_clear():
    if os.path.exists(TRAINING_LOCK):
        os.remove(TRAINING_LOCK)


def _check_orphaned_training():
    lock = _training_lock_read()
    if not lock:
        return None
    pid = lock.get("pid")
    if pid:
        try:
            os.kill(pid, 0)  # 不发送信号，只检查存在性
            return "running"
        except OSError:
            pass
    _training_lock_clear()
    return "orphaned"


orphan_status = _check_orphaned_training()
if orphan_status == "running":
    st.session_state.training_active = True
elif orphan_status == "orphaned":
    st.session_state.training_active = False
    st.warning("检测到上次训练中断（可能刷新了页面），已自动重置训练状态")


# ═══════ 模型参数弹窗 (st.dialog) ═══════

@st.dialog("XGBoost 参数设置")
def xgboost_dialog():
    params = st.session_state.model_params["XGBoost"]
    defaults = DEFAULT_MODEL_PARAMS["XGBoost"]
    new_lr = st.slider("学习率", 0.01, 0.30, params["learning_rate"], 0.01, format="%.2f", key="dg_xgb_lr")
    new_n = st.slider("树数量", 50, 300, params["n_estimators"], 10, key="dg_xgb_n")
    new_md = st.slider("最大深度", 3, 10, params["max_depth"], 1, key="dg_xgb_md")
    new_ss = st.slider("子样本", 0.5, 1.0, params["subsample"], 0.05, key="dg_xgb_ss")
    c1, c2 = st.columns(2)
    if c1.button("恢复默认", use_container_width=True):
        st.session_state.model_params["XGBoost"] = dict(defaults)
        st.session_state.modified_models.discard("XGBoost")
        st.rerun()
    if c2.button("确认保存", use_container_width=True, type="primary"):
        st.session_state.model_params["XGBoost"] = {
            "n_estimators": new_n, "max_depth": new_md,
            "learning_rate": new_lr, "subsample": new_ss}
        _param_changed("XGBoost", "n_estimators", new_n, defaults["n_estimators"])
        st.rerun()


@st.dialog("LightGBM 参数设置")
def lightgbm_dialog():
    params = st.session_state.model_params["LightGBM"]
    defaults = DEFAULT_MODEL_PARAMS["LightGBM"]
    new_lr = st.slider("学习率", 0.01, 0.30, params["learning_rate"], 0.01, format="%.2f", key="dg_lgb_lr")
    new_n = st.slider("树数量", 50, 300, params["n_estimators"], 10, key="dg_lgb_n")
    new_md = st.slider("最大深度", 3, 10, params["max_depth"], 1, key="dg_lgb_md")
    new_nl = st.slider("叶子数", 15, 127, params["num_leaves"], 2, key="dg_lgb_nl")
    new_ss = st.slider("子样本", 0.5, 1.0, params["subsample"], 0.05, key="dg_lgb_ss")
    c1, c2 = st.columns(2)
    if c1.button("恢复默认", use_container_width=True):
        st.session_state.model_params["LightGBM"] = dict(defaults)
        st.session_state.modified_models.discard("LightGBM")
        st.rerun()
    if c2.button("确认保存", use_container_width=True, type="primary"):
        st.session_state.model_params["LightGBM"] = {
            "n_estimators": new_n, "max_depth": new_md,
            "learning_rate": new_lr, "num_leaves": new_nl, "subsample": new_ss}
        _param_changed("LightGBM", "n_estimators", new_n, defaults["n_estimators"])
        st.rerun()


@st.dialog("1D-CNN 参数设置")
def cnn_dialog():
    params = st.session_state.model_params["1D-CNN"]
    defaults = DEFAULT_MODEL_PARAMS["1D-CNN"]
    new_lb = st.slider("时间步长", 10, 40, params["look_back"], 5, key="dg_cnn_lb")
    new_fl = st.slider("卷积核", 16, 64, params["filters"], 8, key="dg_cnn_fl")
    new_ks = st.slider("核大小", 2, 5, params["kernel_size"], 1, key="dg_cnn_ks")
    new_do = st.slider("Dropout", 0.1, 0.4, params["dropout"], 0.05, key="dg_cnn_do")
    new_lr = st.slider("学习率", 0.0001, 0.005, params["learning_rate"], 0.0001, format="%.4f", key="dg_cnn_lr")
    c1, c2 = st.columns(2)
    if c1.button("恢复默认", use_container_width=True):
        st.session_state.model_params["1D-CNN"] = dict(defaults)
        st.session_state.modified_models.discard("1D-CNN")
        st.rerun()
    if c2.button("确认保存", use_container_width=True, type="primary"):
        st.session_state.model_params["1D-CNN"] = {
            "look_back": new_lb, "filters": new_fl, "kernel_size": new_ks,
            "dropout": new_do, "learning_rate": new_lr}
        _param_changed("1D-CNN", "filters", new_fl, defaults["filters"])
        st.rerun()


@st.dialog("CNN-GRU 参数设置")
def cnn_gru_dialog():
    params = st.session_state.model_params["CNN-GRU"]
    defaults = DEFAULT_MODEL_PARAMS["CNN-GRU"]
    new_cf = st.slider("卷积核", 16, 64, params["cnn_filters"], 8, key="dg_cg_cf")
    new_ks = st.slider("核大小", 2, 5, params["kernel_size"], 1, key="dg_cg_ks")
    new_gu = st.slider("GRU单元", 16, 64, params["gru_units"], 8, key="dg_cg_gu")
    new_do = st.slider("Dropout", 0.1, 0.4, params["dropout"], 0.05, key="dg_cg_do")
    new_lr = st.slider("学习率", 0.0001, 0.005, params["learning_rate"], 0.0001, format="%.4f", key="dg_cg_lr")
    c1, c2 = st.columns(2)
    if c1.button("恢复默认", use_container_width=True):
        st.session_state.model_params["CNN-GRU"] = dict(defaults)
        st.session_state.modified_models.discard("CNN-GRU")
        st.rerun()
    if c2.button("确认保存", use_container_width=True, type="primary"):
        st.session_state.model_params["CNN-GRU"] = {
            "cnn_filters": new_cf, "kernel_size": new_ks,
            "gru_units": new_gu, "dropout": new_do, "learning_rate": new_lr}
        _param_changed("CNN-GRU", "cnn_filters", new_cf, defaults["cnn_filters"])
        st.rerun()


@st.dialog("GRU 参数设置")
def gru_dialog():
    params = st.session_state.model_params["GRU"]
    defaults = DEFAULT_MODEL_PARAMS["GRU"]
    new_un = st.slider("神经元", 16, 64, params["units"], 8, key="dg_gru_un")
    new_lb = st.slider("时间步长", 10, 40, params["look_back"], 5, key="dg_gru_lb")
    new_do = st.slider("Dropout", 0.1, 0.4, params["dropout"], 0.05, key="dg_gru_do")
    new_lr = st.slider("学习率", 0.0001, 0.005, params["learning_rate"], 0.0001, format="%.4f", key="dg_gru_lr")
    c1, c2 = st.columns(2)
    if c1.button("恢复默认", use_container_width=True):
        st.session_state.model_params["GRU"] = dict(defaults)
        st.session_state.modified_models.discard("GRU")
        st.rerun()
    if c2.button("确认保存", use_container_width=True, type="primary"):
        st.session_state.model_params["GRU"] = {
            "units": new_un, "look_back": new_lb,
            "dropout": new_do, "learning_rate": new_lr}
        _param_changed("GRU", "units", new_un, defaults["units"])
        st.rerun()


@st.dialog("LSTM 参数设置")
def lstm_dialog():
    params = st.session_state.model_params["LSTM"]
    defaults = DEFAULT_MODEL_PARAMS["LSTM"]
    new_un = st.slider("神经元", 16, 64, params["units"], 8, key="dg_lstm_un")
    new_lb = st.slider("时间步长", 10, 40, params["look_back"], 5, key="dg_lstm_lb")
    new_do = st.slider("Dropout", 0.1, 0.4, params["dropout"], 0.05, key="dg_lstm_do")
    new_lr = st.slider("学习率", 0.0001, 0.005, params["learning_rate"], 0.0001, format="%.4f", key="dg_lstm_lr")
    c1, c2 = st.columns(2)
    if c1.button("恢复默认", use_container_width=True):
        st.session_state.model_params["LSTM"] = dict(defaults)
        st.session_state.modified_models.discard("LSTM")
        st.rerun()
    if c2.button("确认保存", use_container_width=True, type="primary"):
        st.session_state.model_params["LSTM"] = {
            "units": new_un, "look_back": new_lb,
            "dropout": new_do, "learning_rate": new_lr}
        _param_changed("LSTM", "units", new_un, defaults["units"])
        st.rerun()


@st.dialog("PatchTST 参数设置")
def patchtst_dialog():
    params = st.session_state.model_params["PatchTST"]
    defaults = DEFAULT_MODEL_PARAMS["PatchTST"]
    new_dm = st.select_slider("d_model", [32, 64, 128, 256],
                              value=params["d_model"], key="dg_pt_dm")
    new_nh = st.select_slider("注意力头数", [2, 4, 8],
                              value=params["n_heads"], key="dg_pt_nh")
    new_nl = st.slider("编码器层数", 1, 4, params["n_layers"], 1, key="dg_pt_nl")
    new_ps = st.select_slider("Patch大小", [8, 16, 32],
                              value=params["patch_size"], key="dg_pt_ps")
    new_do = st.slider("Dropout", 0.05, 0.3, params["dropout"], 0.05, key="dg_pt_do")
    c1, c2 = st.columns(2)
    if c1.button("恢复默认", use_container_width=True):
        st.session_state.model_params["PatchTST"] = dict(defaults)
        st.session_state.modified_models.discard("PatchTST")
        st.rerun()
    if c2.button("确认保存", use_container_width=True, type="primary"):
        st.session_state.model_params["PatchTST"] = {
            "d_model": new_dm, "n_heads": new_nh, "n_layers": new_nl,
            "patch_size": new_ps, "dropout": new_do}
        _param_changed("PatchTST", "d_model", new_dm, defaults["d_model"])
        st.rerun()


@st.dialog("TFT 参数设置")
def tft_dialog():
    params = st.session_state.model_params["TFT"]
    defaults = DEFAULT_MODEL_PARAMS["TFT"]
    new_hs = st.select_slider("隐藏层大小", [32, 64, 128],
                              value=params["hidden_size"], key="dg_tft_hs")
    new_nh = st.select_slider("注意力头数", [2, 4, 8],
                              value=params["n_heads"], key="dg_tft_nh")
    new_do = st.slider("Dropout", 0.1, 0.4, params["dropout"], 0.05, key="dg_tft_do")
    new_nl = st.slider("LSTM层数", 1, 3, params["lstm_layers"], 1, key="dg_tft_nl")
    c1, c2 = st.columns(2)
    if c1.button("恢复默认", use_container_width=True):
        st.session_state.model_params["TFT"] = dict(defaults)
        st.session_state.modified_models.discard("TFT")
        st.rerun()
    if c2.button("确认保存", use_container_width=True, type="primary"):
        st.session_state.model_params["TFT"] = {
            "hidden_size": new_hs, "n_heads": new_nh,
            "dropout": new_do, "lstm_layers": new_nl}
        _param_changed("TFT", "hidden_size", new_hs, defaults["hidden_size"])
        st.rerun()


@st.dialog("ARIMA 参数设置")
def arima_dialog():
    params = st.session_state.model_params["ARIMA"]
    defaults = DEFAULT_MODEL_PARAMS["ARIMA"]
    new_auto = st.toggle("自动选参", value=params["auto"], key="dg_ar_auto")
    if not new_auto:
        c1, c2, c3 = st.columns(3)
        with c1:
            new_p = st.slider("p", 0, 5, params["p"], 1, key="dg_ar_p")
        with c2:
            new_d = st.slider("d", 0, 2, params["d"], 1, key="dg_ar_d")
        with c3:
            new_q = st.slider("q", 0, 5, params["q"], 1, key="dg_ar_q")
    else:
        new_p, new_d, new_q = params["p"], params["d"], params["q"]
    c1, c2 = st.columns(2)
    if c1.button("恢复默认", use_container_width=True):
        st.session_state.model_params["ARIMA"] = dict(defaults)
        st.session_state.modified_models.discard("ARIMA")
        st.rerun()
    if c2.button("确认保存", use_container_width=True, type="primary"):
        st.session_state.model_params["ARIMA"] = {
            "auto": new_auto, "p": new_p, "d": new_d, "q": new_q}
        _param_changed("ARIMA", "auto", new_auto, defaults["auto"])
        st.rerun()


@st.dialog("SARIMA 参数设置")
def sarima_dialog():
    params = st.session_state.model_params["SARIMA"]
    defaults = DEFAULT_MODEL_PARAMS["SARIMA"]
    c1, c2, c3 = st.columns(3)
    with c1:
        new_p = st.slider("p", 0, 3, params["p"], 1, key="dg_sa_p")
        new_d = st.slider("d", 0, 2, params["d"], 1, key="dg_sa_d")
        new_q = st.slider("q", 0, 3, params["q"], 1, key="dg_sa_q")
    with c2:
        new_P = st.slider("季节P", 0, 3, params["P"], 1, key="dg_sa_P")
        new_D = st.slider("季节D", 0, 2, params["D"], 1, key="dg_sa_D")
        new_Q = st.slider("季节Q", 0, 3, params["Q"], 1, key="dg_sa_Q")
    with c3:
        new_s = st.slider("季节周期s", 3, 66, params["s"], 1, key="dg_sa_s")
    c1, c2 = st.columns(2)
    if c1.button("恢复默认", use_container_width=True):
        st.session_state.model_params["SARIMA"] = dict(defaults)
        st.session_state.modified_models.discard("SARIMA")
        st.rerun()
    if c2.button("确认保存", use_container_width=True, type="primary"):
        st.session_state.model_params["SARIMA"] = {
            "p": new_p, "d": new_d, "q": new_q,
            "P": new_P, "D": new_D, "Q": new_Q, "s": new_s}
        _param_changed("SARIMA", "p", new_p, defaults["p"])
        st.rerun()


@st.dialog("GARCH 参数说明")
def garch_dialog():
    st.info("GARCH(1,1) 模型参数固定，不可修改")
    st.markdown("""
    - **p=1**: ARCH阶数
    - **q=1**: GARCH阶数
    - **dist='t'**: 学生t分布（捕获厚尾特性）
    - **mean='constant'**: 允许非零均值收益
    """)
    st.caption("GARCH模型用于波动率预测和风险指标计算")

# ═══════ 侧边栏 ═══════

btn_train = False

with st.sidebar:
    st.header("配置参数")

    # 数据来源
    st.subheader("数据输入")
    data_source = st.radio("数据来源", ["数据库加载", "Excel上传"], horizontal=True)

    if data_source == "数据库加载":
        if st.button("刷新列表", key="refresh_db_stocks"):
            try:
                st.session_state.db_stocks = list_stocks_with_status()
            except Exception as e:
                st.error(f"加载失败: {e}")

        if not st.session_state.db_stocks:
            try:
                st.session_state.db_stocks = list_stocks_with_status()
            except Exception:
                pass

        if st.session_state.db_stocks:
            stock_list = st.session_state.db_stocks
            stock_labels = {}
            for s in stock_list:
                # 数据新鲜度
                from datetime import date
                today_str = date.today().strftime("%Y-%m-%d")
                end_date = s["end_date"]
                if end_date == today_str:
                    freshness = "最新"
                elif end_date >= today_str:
                    freshness = "最新"
                else:
                    days_behind = (date.today() - date.fromisoformat(end_date)).days
                    freshness = f"{days_behind}天前"
                data_part = f"{s['name']} ({s['code']})  {s['rows']}天  最新: {s['end_date']} ({freshness})"
                if s["trained"]:
                    models = ", ".join(s.get("trained_models", []))
                    train_part = f"  |  已训练: {models}"
                else:
                    train_part = "  |  未训练"
                stock_labels[data_part + train_part] = s

            selected_label = st.selectbox(
                "选择股票", options=list(stock_labels.keys()), key="db_stock_select",
                label_visibility="collapsed",
            )
            if selected_label:
                info = stock_labels[selected_label]

                # 训练版本选择
                selected_session_id = None
                if info["trained"]:
                    sessions = list_stock_sessions(info["code"])
                    if sessions:
                        if len(sessions) == 1:
                            selected_session_id = sessions[0]["session_id"]
                            s = sessions[0]
                            models = ", ".join(s.get("trained_models", []))
                            st.caption(f"训练记录: {s['trained_at']}  {models}")
                        else:
                            session_labels = {}
                            for s in sessions:
                                models = ", ".join(s.get("trained_models", []))
                                session_labels[f"{s['trained_at']}  {models}"] = s
                            selected_session = st.selectbox(
                                "训练版本", options=list(session_labels.keys()),
                                index=0, key=f"session_{info['code']}",
                            )
                            selected_session_id = session_labels[selected_session]["session_id"]

                if st.button("加载", type="primary", use_container_width=True, key="load_db_btn"):
                    with st.spinner(f"加载 {info['name']}...并检查数据更新..."):
                        try:
                            from datetime import date as _date
                            _is_fresh = info["end_date"] >= _date.today().strftime("%Y-%m-%d")
                            if not _is_fresh:
                                st.toast(f"正在更新 {info['name']} 的数据...")
                            df, _, _ = fetch_and_store(info["code"])
                            st.session_state.stock_data = df
                            st.session_state.stock_code = info["code"]
                            st.session_state.stock_name = info["name"]

                            if selected_session_id:
                                try:
                                    result = cloud_load_session(selected_session_id)
                                    if result:
                                        cloud_restore(result[0], result[1])
                                    else:
                                        st.session_state.train_results = None
                                        st.session_state.predictions = None
                                except Exception:
                                    st.session_state.train_results = None
                                    st.session_state.predictions = None
                            else:
                                st.session_state.train_results = None
                                st.session_state.predictions = None

                            st.success(f"加载成功: {len(df)} 条")
                            st.rerun()
                        except Exception as e:
                            st.error(f"加载失败: {e}")
        else:
            st.info("暂无数据，请先获取股票")

        st.divider()
        st.caption("添加新股")
        new_code = st.text_input("股票代码", key="new_stock_code", placeholder="6位代码")
        if new_code and len(new_code) == 6:
            exists = False
            try:
                exists = has_stock_data(new_code) or any(s["code"] == new_code for s in st.session_state.db_stocks)
            except Exception:
                pass
            if exists:
                st.info("该股票数据已存在，刷新列表即可看到")
            else:
                if st.button("获取数据", type="primary", use_container_width=True, key="fetch_new_btn"):
                    with st.spinner(f"正在获取 {new_code} 数据..."):
                        try:
                            df, name, _ = fetch_and_store(new_code)
                            st.session_state.stock_data = df
                            st.session_state.stock_code = new_code
                            st.session_state.stock_name = name
                            st.session_state.train_results = None
                            st.session_state.predictions = None
                            st.success(f"获取成功: {name} ({len(df)} 条)")
                            st.session_state.db_stocks = list_stocks_with_status()
                            st.rerun()
                        except Exception as e:
                            st.error(f"获取失败: {e}")

    else:
        st.download_button(
            "下载Excel模板",
            data=generate_template(),
            file_name="stock_data_template.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        uploaded = st.file_uploader("上传Excel文件", type=["xlsx", "xls"])
        if uploaded and st.button("解析数据", type="primary", use_container_width=True):
            with st.spinner("解析中..."):
                try:
                    df = load_from_excel(uploaded)
                    st.session_state.stock_data = df
                    st.session_state.stock_code = "CUSTOM"
                    st.session_state.stock_name = uploaded.name.split(".")[0]
                    st.session_state.train_results = None
                    st.session_state.predictions = None
                    st.success(f"解析成功: {len(df)} 条数据")
                except Exception as e:
                    st.error(str(e))

    st.divider()

    # 模型选择
    st.subheader("模型选择")
    all_models = [
        "LSTM", "GRU", "1D-CNN", "CNN-GRU", "PatchTST", "TFT",
        "XGBoost", "LightGBM",
        "ARIMA", "SARIMA", "GARCH",
    ]
    selected_models = st.multiselect("选择模型", all_models, default=all_models,
                                      help="DL: LSTM/GRU/1D-CNN/CNN-GRU/PatchTST/TFT | 树模型: XGBoost/LightGBM | 统计: ARIMA/SARIMA/GARCH")

    # 模型参数设置
    dialog_map = {
        "XGBoost": xgboost_dialog, "LightGBM": lightgbm_dialog,
        "1D-CNN": cnn_dialog, "CNN-GRU": cnn_gru_dialog,
        "GRU": gru_dialog, "LSTM": lstm_dialog,
        "PatchTST": patchtst_dialog, "TFT": tft_dialog,
        "ARIMA": arima_dialog, "SARIMA": sarima_dialog, "GARCH": garch_dialog,
    }

    st.caption("参数设置（🔵 = 已修改）")
    cols = st.columns(3)
    for i, m in enumerate(all_models):
        with cols[i % 3]:
            mod = " 🔵" if m in st.session_state.modified_models else ""
            if st.button(f"⚙ {m}{mod}", key=f"set_{m}", use_container_width=True):
                dialog_map[m]()

    use_ensemble = st.toggle("集成预测", value=True)

    st.divider()

    # 通用训练参数
    st.subheader("训练参数")
    forecast_days = st.slider("预测天数", 1, 10, 5)
    look_back = st.slider("时间步长(天)", 10, 60, predict_cfg.get("default_look_back", 30))

    quick_mode = st.toggle("快速模式", value=False, help="减少训练轮次和模型参数，适合快速测试")

    # DL通用参数
    col_e, col_b = st.columns(2)
    with col_e:
        st.session_state.dl_epochs = st.slider(
            "训练轮次", 5, 200,
            st.session_state.dl_epochs if not quick_mode else min(st.session_state.dl_epochs, 30),
            5, key="sidebar_epochs",
            help="深度学习模型的训练轮次")
    with col_b:
        st.session_state.dl_batch_size = st.select_slider(
            "批量大小", [8, 16, 32, 64],
            value=st.session_state.dl_batch_size, key="sidebar_batch")

    st.divider()

    # 模型状态
    if st.session_state.stock_code:
        status = get_model_status(st.session_state.stock_code)
        st.info(f"模型状态: {status}")

    # 操作按钮
    st.subheader("操作")
    if st.session_state.training_active:
        st.warning("训练进行中，请勿刷新页面")
        if st.button("强制停止训练", use_container_width=True, type="secondary"):
            lock = _training_lock_read()
            if lock and lock.get("pid"):
                try:
                    import signal
                    os.kill(lock["pid"], signal.SIGKILL)
                except Exception:
                    pass
            _training_lock_clear()
            st.session_state.training_active = False
            st.rerun()
    else:
        btn_train = st.button("训练所有模型", type="primary", use_container_width=True,
                              disabled=st.session_state.stock_data is None)
    btn_export = st.button("导出所有结果", use_container_width=True,
                           disabled=st.session_state.train_results is None)


# ═══════ 构建 ModelConfig ═══════

def _build_config():
    dl_cfg = predict_cfg.get("dl", {})
    qm = predict_cfg.get("quick_mode", {})
    pt_cfg = predict_cfg.get("patchtst", {})
    tf_cfg = predict_cfg.get("tft", {})
    mp = st.session_state.model_params

    epochs = st.session_state.dl_epochs
    batch_size = st.session_state.dl_batch_size

    if quick_mode:
        epochs = min(epochs, qm.get("epochs", 10))
        units_lstm = [qm.get("lstm_units", [32, 16])[0], qm.get("lstm_units", [32, 16])[1]]
        units_gru = [qm.get("gru_units", [32, 16])[0], qm.get("gru_units", [32, 16])[1]]
        cnn_filters = [qm.get("cnn_filters", [32, 16])[0], qm.get("cnn_filters", [32, 16])[1]]
        cnn_gru_cf = [qm.get("cnn_gru_filters", [32, 16])[0], qm.get("cnn_gru_filters", [32, 16])[1]]
        cnn_gru_gu = [qm.get("cnn_gru_gru_units", [32, 16])[0], qm.get("cnn_gru_gru_units", [32, 16])[1]]
        patchtst_d_model = qm.get("patchtst_d_model", 16)
        patchtst_n_layers = qm.get("patchtst_n_encoder_layers", 1)
        tft_hidden = qm.get("tft_hidden_size", 8)
        tft_n_heads_val = qm.get("tft_n_heads", 2)
        xgb_n_estimators = min(mp["XGBoost"]["n_estimators"], 100)
        lgb_n_estimators = min(mp["LightGBM"]["n_estimators"], 100)
        xgb_max_depth = mp["XGBoost"]["max_depth"]
        xgb_lr = mp["XGBoost"]["learning_rate"]
        lgb_max_depth = mp["LightGBM"]["max_depth"]
        lgb_num_leaves = mp["LightGBM"]["num_leaves"]
    else:
        # DL: 使用模型专属参数
        lstm_p = mp["LSTM"]
        gru_p = mp["GRU"]
        cnn_p = mp["1D-CNN"]
        cg_p = mp["CNN-GRU"]
        pt_p = mp["PatchTST"]
        tft_p = mp["TFT"]
        xgb_p = mp["XGBoost"]
        lgb_p = mp["LightGBM"]

        units_lstm = [lstm_p["units"], lstm_p["units"] // 2]
        units_gru = [gru_p["units"], gru_p["units"] // 2]
        cnn_filters = [cnn_p["filters"], cnn_p["filters"] // 2]
        cnn_gru_cf = [cg_p["cnn_filters"], cg_p["cnn_filters"] // 2]
        cnn_gru_gu = [cg_p["gru_units"], cg_p["gru_units"] // 2]
        patchtst_d_model = pt_p["d_model"]
        patchtst_n_layers = pt_p["n_layers"]
        tft_hidden = tft_p["hidden_size"]
        tft_n_heads_val = tft_p["n_heads"]
        xgb_n_estimators = xgb_p["n_estimators"]
        xgb_max_depth = xgb_p["max_depth"]
        xgb_lr = xgb_p["learning_rate"]
        lgb_n_estimators = lgb_p["n_estimators"]
        lgb_max_depth = lgb_p["max_depth"]
        lgb_num_leaves = lgb_p["num_leaves"]

    # 小样本自动检测
    data = st.session_state.stock_data
    sm_cfg = predict_cfg.get("small_sample", {})
    if data is not None and len(data) < sm_cfg.get("threshold", 200):
        units_lstm = sm_cfg.get("lstm_units", [32, 16])
        units_gru = sm_cfg.get("gru_units", [32, 16])

    sarima_p = mp["SARIMA"]
    arima_p = mp["ARIMA"]
    garch_p = mp["GARCH"]

    # dropout 和 learning_rate：从第一个选中的DL模型获取，或使用默认值
    dropout = dl_cfg.get("dropout", 0.2)
    learning_rate = DL_LEARNING_RATE
    dl_selected = [m for m in selected_models if m in ("LSTM", "GRU", "1D-CNN", "CNN-GRU", "PatchTST", "TFT")]
    if dl_selected:
        first_dl = dl_selected[0]
        dl_params = mp.get(first_dl, {})
        dropout = dl_params.get("dropout", dropout)
        learning_rate = dl_params.get("learning_rate", learning_rate)

    return ModelConfig(
        look_back=look_back,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        dropout=dropout,
        lstm_units=units_lstm,
        gru_units=units_gru,
        cnn_filters=cnn_filters,
        cnn_kernel_size=mp.get("1D-CNN", {}).get("kernel_size", dl_cfg.get("cnn_kernel_size", 3)),
        cnn_gru_filters=cnn_gru_cf,
        cnn_gru_gru_units=cnn_gru_gu,
        cnn_gru_kernel_size=mp.get("CNN-GRU", {}).get("kernel_size", dl_cfg.get("cnn_kernel_size", 3)),
        early_stop_patience=3 if quick_mode else dl_cfg.get("early_stop_patience", 10),
        patchtst_patch_size=mp.get("PatchTST", {}).get("patch_size", pt_cfg.get("patch_size", 16)),
        patchtst_d_model=patchtst_d_model,
        patchtst_n_heads=mp.get("PatchTST", {}).get("n_heads", pt_cfg.get("n_heads", 4)),
        patchtst_n_encoder_layers=patchtst_n_layers,
        patchtst_ff_dim=pt_cfg.get("ff_dim", 256),
        patchtst_dropout=mp.get("PatchTST", {}).get("dropout", pt_cfg.get("dropout", 0.1)),
        tft_hidden_size=tft_hidden,
        tft_n_heads=tft_n_heads_val,
        tft_dropout=mp.get("TFT", {}).get("dropout", tf_cfg.get("dropout", 0.2)),
        tft_lstm_layers=mp.get("TFT", {}).get("lstm_layers", tf_cfg.get("lstm_layers", 1)),
        xgboost_n_estimators=xgb_n_estimators,
        xgboost_max_depth=xgb_max_depth,
        xgboost_learning_rate=xgb_lr,
        xgboost_subsample=mp["XGBoost"]["subsample"],
        lightgbm_n_estimators=lgb_n_estimators,
        lightgbm_max_depth=lgb_max_depth,
        lightgbm_learning_rate=mp["LightGBM"]["learning_rate"],
        lightgbm_num_leaves=lgb_num_leaves,
        lightgbm_subsample=mp["LightGBM"]["subsample"],
        sarima_order=(sarima_p["p"], sarima_p["d"], sarima_p["q"]),
        sarima_seasonal_order=(sarima_p["P"], sarima_p["D"], sarima_p["Q"], sarima_p["s"]),
        garch_p=garch_p["p"],
        garch_q=garch_p["q"],
        garch_dist=garch_p["dist"],
    )


# ═══════ 实时训练回调 ═══════

class StreamlitTrainingCallbacks(TrainingCallbacks):
    """将训练回调连接到Streamlit UI容器"""

    def __init__(self, model_containers, overall_progress, overall_status, log_container):
        self.containers = model_containers
        self.progress = overall_progress
        self.status = overall_status
        self.log = log_container
        self.log_lines = []
        self.loss_history = {}
        self.start_time = None
        self.total_models = 0
        self.completed_models = 0
        self.model_times = []

    def on_training_start(self, model_list):
        self.start_time = _time.time()
        self.total_models = len(model_list)
        for name in model_list:
            self.loss_history[name] = {"train": [], "val": [], "lr": [], "grad": []}
        self._add_log(f"训练启动 | 模型: {', '.join(model_list)}")

    def on_model_start(self, model_name, model_index, total_models):
        if model_name in self.containers:
            self.containers[model_name]["status"].info(f"**{model_name}** - 正在训练...")
        pct = model_index / total_models
        self.progress.progress(pct, text=f"训练 {model_name} ({model_index+1}/{total_models})")
        elapsed = _time.time() - self.start_time
        if self.completed_models > 0:
            avg_time = elapsed / self.completed_models
            eta = avg_time * (total_models - model_index)
            self.status.markdown(f"已用: {elapsed:.0f}s | 预计剩余: {eta:.0f}s")
        else:
            self.status.markdown(f"已用: {elapsed:.0f}s")
        self._add_log(f"[{model_name}] 开始训练 ({model_index+1}/{total_models})")

    def on_fold_start(self, model_name, fold, total_folds):
        self._add_log(f"[{model_name}] 交叉验证 Fold {fold}/{total_folds}")

    def on_fold_end(self, model_name, fold, fold_metrics):
        rmse = fold_metrics.get("rmse", 0)
        r2 = fold_metrics.get("r2", 0)
        self._add_log(f"[{model_name}] Fold {fold} 完成 | RMSE={rmse:.4f}, R²={r2:.4f}")

    def on_early_stop(self, model_name, epoch, best_epoch):
        self._add_log(f"[{model_name}] 早停 Epoch {epoch} (最佳 Epoch {best_epoch})")

    def on_log(self, message):
        self._add_log(message)

    def on_epoch_end(self, model_name, epoch, total_epochs,
                     train_loss, val_loss, lr, grad_norm=None):
        if model_name not in self.loss_history:
            self.loss_history[model_name] = {"train": [], "val": [], "lr": [], "grad": []}
        hist = self.loss_history[model_name]
        hist["train"].append(train_loss)
        hist["val"].append(val_loss)
        hist["lr"].append(lr)
        hist["grad"].append(grad_norm)

        if model_name not in self.containers:
            return

        # 快速模式: 只在最后一个 epoch 更新 UI，避免 Plotly 渲染拖慢训练
        if total_epochs <= 15 and epoch < total_epochs:
            # 只更新轻量文本
            self.containers[model_name]["metrics_row"].markdown(
                f"Epoch **{epoch}/{total_epochs}** | "
                f"Train: `{train_loss:.6f}` | Val: `{val_loss:.6f}`")
            base = self.completed_models / self.total_models
            within = (epoch / total_epochs) / self.total_models
            self.progress.progress(min(base + within, 0.99),
                text=f"训练 {model_name} - Epoch {epoch}/{total_epochs}")
            return

        c = self.containers[model_name]

        # 损失曲线
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=hist["train"], name="Train Loss", mode="lines",
                                  line=dict(color="#1f77b4")))
        fig.add_trace(go.Scatter(y=hist["val"], name="Val Loss", mode="lines",
                                  line=dict(color="#ff7f0e", dash="dash")))
        fig.update_layout(height=280, xaxis_title="Epoch", yaxis_title="Loss",
                          margin=dict(t=10, b=30, l=40, r=10))
        c["loss_chart"].plotly_chart(fig, use_container_width=True)

        # 指标行
        c["metrics_row"].markdown(
            f"Epoch **{epoch}/{total_epochs}** | "
            f"Train: `{train_loss:.6f}` | Val: `{val_loss:.6f}`")

        # 学习率 + 梯度范数
        lr_text = f"学习率: `{lr:.2e}`"
        if grad_norm is not None:
            lr_text += f" | 梯度范数: `{grad_norm:.4f}`"
        c["lr_grad"].markdown(lr_text)

        # 进度
        base = self.completed_models / self.total_models
        within = (epoch / total_epochs) / self.total_models
        self.progress.progress(min(base + within, 0.99),
                               text=f"{model_name}: Epoch {epoch}/{total_epochs}")

        if epoch % 5 == 0 or epoch == total_epochs:
            self._add_log(f"[{model_name}] Epoch {epoch}/{total_epochs} - "
                          f"loss: {train_loss:.6f}, val_loss: {val_loss:.6f}, lr: {lr:.2e}")

    def on_overfitting_warning(self, model_name, epoch, val_loss, best_val_loss):
        if model_name in self.containers:
            pct_rise = ((val_loss / best_val_loss) - 1) * 100
            self.containers[model_name]["warning"].warning(
                f"过拟合警告: Epoch {epoch}, val_loss={val_loss:.6f} "
                f"(最佳={best_val_loss:.6f}, 上升 {pct_rise:.1f}%)")
        self._add_log(f"[{model_name}] 过拟合警告 Epoch {epoch}")

    def on_model_end(self, model_name, result):
        self.completed_models += 1
        elapsed = _time.time() - self.start_time
        if model_name in self.containers:
            c = self.containers[model_name]
            rmse = result.cv_metrics.get("rmse", np.nan)
            if not np.isnan(rmse):
                c["status"].success(
                    f"**{model_name}** - 训练完成 | RMSE: {rmse:.4f} | 耗时: {result.training_time:.1f}s")
            else:
                err = result.train_history.get("error", "未知错误")
                c["status"].error(f"**{model_name}** - 训练失败: {err}")
        pct = self.completed_models / self.total_models
        self.progress.progress(pct, text=f"已完成 {self.completed_models}/{self.total_models}")
        rmse = result.cv_metrics.get("rmse", np.nan)
        r2 = result.cv_metrics.get("r2", np.nan)
        metrics_str = ""
        if not np.isnan(rmse):
            metrics_str = f" | RMSE={rmse:.4f}, R²={r2:.4f}"
        self._add_log(f"[{model_name}] ✓ 完成 (耗时 {result.training_time:.1f}s){metrics_str}")

    def on_training_complete(self, all_results):
        total_time = _time.time() - self.start_time
        self.progress.progress(1.0, text="全部训练完成!")
        self.status.markdown(f"总耗时: **{total_time:.1f}s** | 模型数: {len(all_results)}")
        self._add_log(f"全部训练完成, 总耗时 {total_time:.1f}s")

    def _add_log(self, msg):
        self.log_lines.append(msg)
        self.log.code("\n".join(self.log_lines[-30:]), language=None)


# ═══════ 训练触发 ═══════

if btn_train and st.session_state.stock_data is not None:
    _training_lock_write({"pid": os.getpid(),
                          "stock_code": st.session_state.stock_code,
                          "started_at": datetime.now().isoformat()})

    # 杀死可能残留的训练进程，避免抢占 CPU
    import subprocess, signal
    try:
        result = subprocess.run(["pgrep", "-f", "train_all_models"], capture_output=True, text=True)
        for pid in result.stdout.strip().split("\n"):
            if pid:
                os.kill(int(pid), signal.SIGKILL)
    except Exception:
        pass

    st.session_state.training_active = True
    model_config = _build_config()

    # 创建实时监控UI容器
    st.subheader("训练进度")
    overall_progress = st.progress(0, text="准备训练...")
    overall_status = st.empty()

    st.subheader("实时监控")
    dl_selected = [m for m in selected_models if m in ("LSTM", "GRU", "1D-CNN", "CNN-GRU", "PatchTST", "TFT")]
    tree_selected = [m for m in selected_models if m in ("XGBoost", "LightGBM")]
    stat_selected = [m for m in selected_models if m in ("ARIMA", "SARIMA", "GARCH")]
    all_ordered = dl_selected + tree_selected + stat_selected

    model_containers = {}
    if all_ordered:
        model_tabs = st.tabs(all_ordered)
        for i, mname in enumerate(all_ordered):
            with model_tabs[i]:
                model_containers[mname] = {
                    "status": st.empty(),
                    "loss_chart": st.empty(),
                    "metrics_row": st.empty(),
                    "lr_grad": st.empty(),
                    "warning": st.empty(),
                }

    st.subheader("训练日志")
    log_container = st.empty()

    # 构建回调
    sl_callbacks = StreamlitTrainingCallbacks(
        model_containers, overall_progress, overall_status, log_container)

    try:
        if quick_mode:
            # 快速模式: 独立进程训练，避免 Streamlit/TF 线程冲突
            import subprocess, pickle, tempfile
            overall_status.markdown("快速训练中...")

            with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
                input_path = f.name
                pickle.dump({
                    "df": st.session_state.stock_data,
                    "selected_models": selected_models,
                    "config": model_config,
                    "forecast_days": forecast_days,
                }, f)

            output_path = input_path + ".out"
            worker = os.path.join(ROOT, "quick_train_worker.py")

            proc = subprocess.run(
                [sys.executable, worker, input_path, output_path],
                capture_output=True, text=True, timeout=300,
                env={**os.environ, "TF_CPP_MIN_LOG_LEVEL": "3"})

            if proc.returncode != 0:
                raise RuntimeError(f"训练失败:\n{proc.stderr[-1000:]}")

            with open(output_path, "rb") as f:
                results = pickle.load(f)

            os.unlink(input_path)
            os.unlink(output_path)
        else:
            results = train_all_models(
                st.session_state.stock_data,
                selected_models,
                model_config,
                forecast_days=forecast_days,
                callbacks=sl_callbacks,
                quick=False,
            )

        st.session_state.train_results = results
        st.session_state["_last_config"] = model_config

        # 保存loss_history供Tab2展示
        st.session_state["_loss_history"] = sl_callbacks.loss_history

        # 集成预测
        weights = None
        preds = None
        if use_ensemble and len(results) > 1:
            weights = compute_ensemble_weights(results)
            last_price = st.session_state.stock_data["close"].iloc[-1]
            preds = ensemble_predict(results, weights, forecast_days, last_price)
            st.session_state.ensemble_weights = weights
            st.session_state.predictions = preds
        elif results:
            weights = {list(results.keys())[0]: 1.0}
            st.session_state.ensemble_weights = weights
            last_price = st.session_state.stock_data["close"].iloc[-1]
            preds = ensemble_predict(results, weights, forecast_days, last_price)
            st.session_state.predictions = preds

        # 应用A股涨跌停限制
        if preds and last_price:
            stock_code = st.session_state.get("stock_code", "")
            stock_name = st.session_state.get("stock_name", "")
            limit_pct = detect_price_limit_pct(stock_code, stock_name)
            if limit_pct is not None:
                preds = apply_price_limits(preds, last_price, limit_pct)
                st.session_state.predictions = preds
                st.session_state["_price_limit_pct"] = limit_pct
                st.session_state["_board_name"] = get_board_name(stock_code)

        # 保存模型
        for name, result in results.items():
            if result.model_object is not None:
                try:
                    save_model(st.session_state.stock_code, name, result)
                except Exception:
                    pass

        # 保存到云端
        if weights and preds:
            try:
                sid = cloud_save(
                    st.session_state.stock_code, st.session_state.stock_name,
                    results, weights, preds, model_config, forecast_days, selected_models,
                    stock_data=st.session_state.stock_data)
                if sid:
                    st.toast("训练结果已同步到云端")
            except Exception as e:
                st.warning(f"云端保存失败: {e}")

        overall_progress.progress(1.0, text="训练完成!")
    except Exception as e:
        st.error(f"训练失败: {e}")
        import traceback
        st.code(traceback.format_exc())

    st.session_state.training_active = False
    _training_lock_clear()
    st.rerun()


# ═══════ 主界面标签页 ═══════

def _chart_config(fig, height=None, title=None, xaxis_title=None, yaxis_title=None):
    """统一图表交互配置：禁用拖拽、统一悬停、精简工具栏"""
    fig.update_layout(
        dragmode=False,
        hovermode='x unified',
        modebar_add=['zoom2d', 'resetScale2d'],
    )
    if height:
        fig.update_layout(height=height)
    if title:
        fig.update_layout(title=title)
    if xaxis_title:
        fig.update_layout(xaxis_title=xaxis_title)
    if yaxis_title:
        fig.update_layout(yaxis_title=yaxis_title)
    return fig


tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
    ["数据概览", "模型训练", "预测结果", "模型评估", "模型管理", "结果导出", "模型参数详情", "中长期预测"]
)

# ── Tab 1: 数据概览 ──────────────────────────────────────────

with tab1:
    if st.session_state.stock_data is not None:
        df = st.session_state.stock_data
        name = st.session_state.stock_name or ""
        code = st.session_state.stock_code or ""

        # 日期范围选择器
        st.subheader("数据范围")
        df_full = df
        min_date = df_full.index[0].date() if hasattr(df_full.index[0], 'date') else df_full.index[0]
        max_date = df_full.index[-1].date() if hasattr(df_full.index[-1], 'date') else df_full.index[-1]
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            start_date = st.date_input(
                "开始日期",
                value=min_date,
                min_value=min_date,
                max_value=max_date,
                key="tab1_start_date")
        with col_d2:
            end_date = st.date_input(
                "结束日期",
                value=max_date,
                min_value=min_date,
                max_value=max_date,
                key="tab1_end_date")
        mask = (df.index >= pd.Timestamp(start_date)) & (df.index <= pd.Timestamp(end_date))
        df = df[mask]
        if len(df) < 100:
            st.warning(f"选择的时间段仅 {len(df)} 天数据，分析可能不够可靠")

        # 信息卡片
        c1, c2 = st.columns(2)
        c1.metric("股票", f"{name} ({code})")
        c2.metric("数据量", f"{len(df)} 天")
        c3, c4 = st.columns(2)
        c3.metric("最新收盘", f"¥{df['close'].iloc[-1]:.2f}")
        c4.metric("数据区间", f"{df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')}")

        # 小样本警告
        sm_threshold = predict_cfg.get("small_sample", {}).get("threshold", 200)
        if len(df) < sm_threshold:
            st.warning(f"数据量较少（{len(df)} 天），建议开启快速模式或减小时间步长")

        # 收盘价走势
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            row_heights=[0.7, 0.3], vertical_spacing=0.05)
        fig.add_trace(go.Scatter(x=df.index, y=df["close"], name="收盘价",
                                 line=dict(color="#1f77b4")), row=1, col=1)
        if "volume" in df.columns:
            colors = ["red" if df["close"].iloc[i] >= df["close"].iloc[max(0,i-1)]
                      else "green" for i in range(len(df))]
            fig.add_trace(go.Bar(x=df.index, y=df["volume"], name="成交量",
                                 marker_color=colors, opacity=0.5), row=2, col=1)
        fig.update_layout(height=500, title="历史行情", showlegend=True,
                          xaxis2_title="日期", yaxis_title="价格", yaxis2_title="成交量")
        st.plotly_chart(fig, use_container_width=True)

        # 统计信息
        st.subheader("数据统计")
        stats = df[["open", "high", "low", "close", "volume"]].describe().T
        stats.columns = ["计数", "均值", "标准差", "最小", "25%", "50%", "75%", "最大"]
        st.dataframe(stats.style.format("{:.2f}"), use_container_width=True)

        # 数据质量检查
        passed, val_warnings, val_errors = validate_training_data(df)
        if val_warnings:
            for w in val_warnings:
                st.warning(f"数据质量: {w}")
        if val_errors:
            for e in val_errors:
                st.error(f"数据质量: {e}")

        # 成交量分析
        with st.expander("成交量分析", expanded=False):
            if "close" in df.columns and "volume" in df.columns:
                df_ind = compute_technical_indicators(df.head(200) if len(df) > 200 else df)
                recent = df_ind.dropna().tail(100)
                if len(recent) == 0:
                    st.info("数据不足，无法进行成交量分析")
                else:

                    col_v1, col_v2 = st.columns(2)
                    with col_v1:
                        fig_vol = go.Figure()
                        fig_vol.add_trace(go.Scatter(x=recent.index, y=recent["vol_ma5"],
                            name="5日均量", line=dict(color="red")))
                        fig_vol.add_trace(go.Scatter(x=recent.index, y=recent["vol_ma20"],
                            name="20日均量", line=dict(color="blue")))
                        fig_vol.update_layout(height=300, title="成交量趋势",
                            xaxis_title="日期", yaxis_title="成交量")
                        st.plotly_chart(fig_vol, use_container_width=True)

                    with col_v2:
                        fig_ratio = go.Figure()
                        fig_ratio.add_trace(go.Scatter(x=recent.index, y=recent["vol_ratio"],
                            name="量比", line=dict(color="green")))
                        fig_ratio.add_hline(y=1.0, line_dash="dash", line_color="gray",
                            annotation_text="基准线")
                        fig_ratio.add_hline(y=1.5, line_dash="dash", line_color="orange",
                            annotation_text="放量(+50%)")
                        fig_ratio.update_layout(height=300, title="量比 (vol/vol_ma20)",
                            xaxis_title="日期", yaxis_title="量比")
                        st.plotly_chart(fig_ratio, use_container_width=True)

                    # OBV
                    fig_obv = go.Figure()
                    obv_colors = ["red" if recent["close"].iloc[i] >= recent["close"].iloc[max(0, i-1)]
                                  else "green" for i in range(len(recent))]
                    fig_obv.add_trace(go.Scatter(x=recent.index, y=recent["obv"],
                        name="OBV", mode="lines+markers", marker_color=obv_colors))
                    fig_obv.update_layout(height=300, title="能量潮指标 (OBV)",
                        xaxis_title="日期", yaxis_title="OBV")
                    st.plotly_chart(fig_obv, use_container_width=True)

                    # 资金流向信号
                    latest_vol_ratio = recent["vol_ratio"].iloc[-1]
                    latest_corr = recent["vol_price_corr"].iloc[-1]
                    if pd.notna(latest_vol_ratio) and pd.notna(latest_corr):
                        signals = []
                        if latest_vol_ratio > 1.5 and recent["close"].iloc[-1] > recent["close"].iloc[-2]:
                            signals.append("放量上涨，上涨趋势确认")
                        elif latest_vol_ratio > 1.5 and recent["close"].iloc[-1] < recent["close"].iloc[-2]:
                            signals.append("放量下跌，注意风险")
                        elif latest_vol_ratio < 0.5:
                            signals.append("缩量，市场观望情绪浓")
                        if latest_corr > 0.5:
                            signals.append("量价正相关 (量随价涨)")
                        elif latest_corr < -0.5:
                            signals.append("量价负相关 (量价背离)")

                        if signals:
                            st.caption("**资金流向信号:**")
                            for s in signals:
                                if "风险" in s:
                                    st.warning(s)
                                else:
                                    st.info(s)
    else:
        st.info("请先在侧边栏获取或上传数据")


# ── Tab 2: 模型训练 ──────────────────────────────────────────

with tab2:
    if st.session_state.train_results:
        results = st.session_state.train_results

        # 训练总结
        st.subheader("训练总结")
        valid_results = {k: v for k, v in results.items()
                         if v.cv_metrics.get("rmse") and not np.isnan(v.cv_metrics.get("rmse", np.nan))}
        failed_results = {k: v for k, v in results.items() if k not in valid_results}
        total_time = sum(r.training_time for r in results.values())

        col_s1, col_s2, col_s3 = st.columns(3)
        if valid_results:
            best_name = min(valid_results, key=lambda k: valid_results[k].cv_metrics["rmse"])
            col_s1.metric("最佳模型", best_name,
                          delta=f"RMSE: {valid_results[best_name].cv_metrics['rmse']:.4f}")
        col_s2.metric("训练模型数", f"{len(valid_results)}/{len(results)}")
        col_s3.metric("总耗时", f"{total_time:.1f}s")

        if failed_results:
            for name, r in failed_results.items():
                err = r.train_history.get("error", "未知错误")
                st.error(f"{name} 训练失败: {err}")

        # 模型排名表
        st.subheader("模型排名（按 RMSE）")
        def _fmt(v, fmt=".4f"):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return "N/A"
            return f"{v:{fmt}}"

        ranking = []
        for name, r in results.items():
            m = r.cv_metrics
            ranking.append({
                "模型": name,
                "MAE": _fmt(m.get("mae")),
                "RMSE": _fmt(m.get("rmse")),
                "MAPE(%)": _fmt(m.get("mape"), ".2f"),
                "R²": _fmt(m.get("r2")),
                "耗时(秒)": _fmt(r.training_time, ".1f"),
            })
        ranking_df = pd.DataFrame(ranking)
        st.dataframe(ranking_df, use_container_width=True, hide_index=True)

        # Loss 曲线
        st.subheader("训练损失曲线")
        dl_results = {k: v for k, v in results.items()
                      if k in ("LSTM", "GRU", "1D-CNN", "PatchTST", "TFT")}
        if dl_results:
            fig = go.Figure()
            for name, r in dl_results.items():
                if "loss" in r.train_history:
                    fig.add_trace(go.Scatter(
                        y=r.train_history["loss"], name=f"{name}-训练",
                        mode="lines"))
                if "val_loss" in r.train_history:
                    fig.add_trace(go.Scatter(
                        y=r.train_history["val_loss"], name=f"{name}-验证",
                        mode="lines", line=dict(dash="dash")))
            fig.update_layout(height=400, xaxis_title="Epoch", yaxis_title="Loss (MSE)",
                              title="各模型训练/验证损失")
            st.plotly_chart(fig, use_container_width=True)

        # 详细学习率/梯度信息（如果有保存的loss_history）
        loss_hist = st.session_state.get("_loss_history", {})
        if loss_hist:
            with st.expander("学习率 & 梯度范数变化"):
                for name, hist in loss_hist.items():
                    if hist.get("lr") and len(hist["lr"]) > 0:
                        st.markdown(f"**{name}**")
                        col_lr, col_gr = st.columns(2)
                        with col_lr:
                            fig_lr = go.Figure()
                            fig_lr.add_trace(go.Scatter(y=hist["lr"], mode="lines", name="LR"))
                            fig_lr.update_layout(height=200, title="学习率",
                                                  margin=dict(t=30, b=20))
                            st.plotly_chart(fig_lr, use_container_width=True)
                        with col_gr:
                            grads = [g for g in hist.get("grad", []) if g is not None]
                            if grads:
                                fig_gr = go.Figure()
                                fig_gr.add_trace(go.Scatter(y=grads, mode="lines", name="Grad Norm"))
                                fig_gr.update_layout(height=200, title="梯度范数",
                                                      margin=dict(t=30, b=20))
                                st.plotly_chart(fig_gr, use_container_width=True)

        # 集成权重
        if st.session_state.ensemble_weights:
            st.subheader("集成权重")
            weights = st.session_state.ensemble_weights
            fig = go.Figure(go.Pie(
                labels=list(weights.keys()),
                values=list(weights.values()),
                textinfo="label+percent",
            ))
            fig.update_layout(height=300, title="模型权重分配（基于 1/RMSE）")
            st.plotly_chart(fig, use_container_width=True)

        # 重新训练按钮
        if st.button("重新训练所有模型", key="retrain_btn"):
            st.session_state.train_results = None
            st.session_state.predictions = None
            st.session_state.ensemble_weights = None
            st.rerun()
    else:
        st.info("请先在侧边栏点击「训练所有模型」按钮开始训练")
        if st.session_state.stock_data is not None:
            st.markdown(f"**数据量**: {len(st.session_state.stock_data)} 天")
            st.markdown(f"**已选模型**: {', '.join(selected_models)}")


# ── Tab 3: 预测结果 ──────────────────────────────────────────

with tab3:
    if st.session_state.predictions and st.session_state.predictions.get("predicted_close") is not None \
       and len(st.session_state.predictions["predicted_close"]) > 0:
        preds = st.session_state.predictions

        if st.session_state.stock_data is not None:
            last_price = st.session_state.stock_data["close"].iloc[-1]
            last_date = st.session_state.stock_data.index[-1]
        else:
            last_price = preds["predicted_close"][0] / (1 + preds["daily_return"][0] / 100) if preds["daily_return"][0] != 0 else preds["predicted_close"][0]
            last_date = pd.Timestamp.now()

        # 生成未来交易日日期
        future_dates = pd.bdate_range(start=last_date + timedelta(days=1), periods=forecast_days)

        # 预测结果表
        st.subheader("未来预测结果")
        pred_table = pd.DataFrame({
            "日期": future_dates.strftime("%Y-%m-%d"),
            "预测收盘价": [f"¥{p:.2f}" for p in preds["predicted_close"]],
            "日收益率(%)": [f"{r:+.2f}" for r in preds["daily_return"]],
            "累计收益率(%)": [f"{r:+.2f}" for r in preds["cumulative_return"]],
        })
        st.dataframe(pred_table, use_container_width=True, hide_index=True)

        # 关键指标
        c1, c2, c3 = st.columns(3)
        total_ret = preds["cumulative_return"][-1]
        c1.metric("预测总收益", f"{total_ret:+.2f}%",
                  delta="看涨" if total_ret > 0 else "看跌")
        c2.metric("最高预测价", f"¥{max(preds['predicted_close']):.2f}")
        c3.metric("最低预测价", f"¥{min(preds['predicted_close']):.2f}")

        # 收益率柱状图
        fig_ret = go.Figure()
        colors = ["red" if r > 0 else "green" for r in preds["daily_return"]]
        fig_ret.add_trace(go.Bar(
            x=future_dates.strftime("%m-%d"),
            y=preds["daily_return"],
            marker_color=colors,
            name="日收益率",
        ))
        fig_ret.update_layout(height=300, title="未来日收益率预测",
                              xaxis_title="日期", yaxis_title="收益率(%)")
        st.plotly_chart(fig_ret, use_container_width=True)

        # 价格走势图（含置信区间）
        if st.session_state.stock_data is not None:
            hist_close = st.session_state.stock_data["close"].tail(60)
        else:
            hist_close = pd.Series(dtype=float)
        fig_price = go.Figure()

        # 历史
        if not hist_close.empty:
            fig_price.add_trace(go.Scatter(
                x=hist_close.index, y=hist_close.values,
                name="历史收盘价", line=dict(color="#1f77b4")))

        # 预测
        fig_price.add_trace(go.Scatter(
            x=future_dates, y=preds["predicted_close"],
            name="集成预测", line=dict(color="red", width=2, dash="dot"),
            mode="lines+markers"))

        # 置信区间
        if "confidence_lower" in preds and len(preds["confidence_lower"]) > 0:
            fig_price.add_trace(go.Scatter(
                x=list(future_dates) + list(future_dates[::-1]),
                y=list(preds["confidence_upper"]) + list(preds["confidence_lower"][::-1]),
                fill="toself", fillcolor="rgba(255,0,0,0.1)",
                line=dict(color="rgba(255,0,0,0)"), name="95%置信区间"))

        # 涨跌停限制线
        limit_pct_display = st.session_state.get("_price_limit_pct")
        if limit_pct_display is not None:
            board = st.session_state.get("_board_name", "")
            upper_limit = last_price * (1 + limit_pct_display)
            lower_limit = last_price * (1 - limit_pct_display)
            fig_price.add_hline(y=upper_limit, line_dash="dash", line_color="purple",
                                annotation_text=f"涨停(+{limit_pct_display*100:.0f}%)",
                                annotation_position="top right")
            fig_price.add_hline(y=lower_limit, line_dash="dash", line_color="purple",
                                annotation_text=f"跌停(-{limit_pct_display*100:.0f}%)",
                                annotation_position="bottom right")

        # 波段斐波那契水平线
        if st.session_state.stock_data is not None:
            wave_info = detect_wave_levels(st.session_state.stock_data)
            fib_levels = calculate_wave_fibonacci(wave_info)
            for level_info in fib_levels:
                line_width = 2.5 if "黄金" in level_info.get("type", "") else 1.0
                fig_price.add_hline(y=level_info["price"], line_dash="dash",
                    line_color=level_info["color"], opacity=0.5,
                    annotation_text=f"{level_info['name']} ({level_info['price']:.1f})",
                    annotation_position="left",
                    line_width=line_width)

        fig_price.update_layout(height=500, title="收盘价预测走势（含置信区间）",
                                xaxis_title="日期", yaxis_title="价格(¥)")
        st.plotly_chart(fig_price, use_container_width=True)

        # 涨跌停信息
        if limit_pct_display is not None:
            board = st.session_state.get("_board_name", "")
            upper_limit = last_price * (1 + limit_pct_display)
            lower_limit = last_price * (1 - limit_pct_display)
            st.info(f"板块: **{board}** | 涨跌停限制: ±**{limit_pct_display*100:.0f}%** | 涨停价: ¥{upper_limit:.2f} | 跌停价: ¥{lower_limit:.2f}")

        # 波段斐波那契交易信号
        if st.session_state.stock_data is not None:
            wave_info = detect_wave_levels(st.session_state.stock_data)
            fib_levels = calculate_wave_fibonacci(wave_info)
            # 计算相对成交量
            if "相对成交量" in st.session_state.stock_data.columns:
                rel_vol = float(st.session_state.stock_data["相对成交量"].dropna().iloc[-1])
            else:
                vol = st.session_state.stock_data["volume"]
                rel_vol = float(vol.iloc[-1] / vol.tail(20).mean()) if len(vol) >= 20 else 1.0
            model_pred = float(preds["daily_return"][0]) if len(preds["daily_return"]) > 0 else 0
            fib_signals = generate_wave_fib_signals(
                last_price, fib_levels, wave_info, model_pred, rel_vol)
            if fib_signals:
                st.subheader("波段斐波那契交易信号")
                # 波段信息
                st.caption(f"波段趋势: **{wave_info['trend']}** | "
                          f"波段起点: ¥{wave_info['wave_start']:.2f} ({wave_info['start_date'].strftime('%Y-%m-%d') if hasattr(wave_info['start_date'], 'strftime') else wave_info['start_date']}) | "
                          f"波段终点: ¥{wave_info['wave_end']:.2f} ({wave_info['end_date'].strftime('%Y-%m-%d') if hasattr(wave_info['end_date'], 'strftime') else wave_info['end_date']}) | "
                          f"波段涨跌幅: {wave_info['wave_return']*100:.1f}%")

                # 斐波那契价位表格
                st.markdown("**关键黄金分割价位**")
                fib_table_data = []
                for lv in sorted(fib_levels, key=lambda x: x["price"]):
                    bg = None
                    if "黄金" in lv.get("type", ""):
                        bg = "gold"
                    elif abs(lv["price"] - last_price) / last_price < 0.01:
                        bg = "lightblue"
                    fib_table_data.append({
                        "价位名称": lv["name"],
                        "价格": f"¥{lv['price']:.2f}",
                        "类型": lv["type"],
                        "距当前价": f"{(lv['price']/last_price - 1)*100:+.1f}%",
                    })
                fib_df = pd.DataFrame(fib_table_data)
                st.dataframe(fib_df, use_container_width=True, hide_index=True)

                # 信号展示
                for s in fib_signals:
                    signal_type = s["type"]
                    if signal_type in ("强买入", "买入"):
                        st.success(f"**[{signal_type}]** {s['reason']} (置信度: {'★'*s['confidence']})")
                    elif signal_type in ("强卖出", "卖出", "止损/清仓"):
                        st.error(f"**[{signal_type}]** {s['reason']} (置信度: {'★'*s['confidence']})")
                    elif signal_type in ("止盈", "轻仓抄底"):
                        st.warning(f"**[{signal_type}]** {s['reason']} (置信度: {'★'*s['confidence']})")
                    else:
                        st.info(f"**[{signal_type}]** {s['reason']} (置信度: {'★'*s['confidence']})")

        # 各模型对比
        if preds.get("model_predictions"):
            st.subheader("各模型预测对比")
            fig_cmp = go.Figure()
            for name, vals in preds["model_predictions"].items():
                fig_cmp.add_trace(go.Scatter(
                    x=future_dates, y=vals,
                    name=name, mode="lines+markers"))
            if len(preds["predicted_close"]) > 0:
                fig_cmp.add_trace(go.Scatter(
                    x=future_dates, y=preds["predicted_close"],
                    name="集成", line=dict(width=3, dash="dash"), mode="lines"))
            fig_cmp.update_layout(height=400, title="各模型预测价格对比",
                                  xaxis_title="日期", yaxis_title="价格(¥)")
            st.plotly_chart(fig_cmp, use_container_width=True)

        # 中长期预测联动
        if st.session_state.longterm_results is not None:
            lt = st.session_state.longterm_results
            ensemble = lt.get("_ensemble", {})
            lt_pred = ensemble.get("prediction", 0) * 100
            valid_lt_models = [m for m in lt if m != "_ensemble" and not np.isnan(lt[m].get("cv_rmse", float("nan")))]
            lt_dir_acc = np.mean([lt[m]["direction_accuracy"] for m in valid_lt_models]) if valid_lt_models else 0.5
            lt_rating = get_rating(lt_pred, lt_dir_acc)
            lt_period = st.session_state.get("longterm_period", "N/A")
            suggestion = "可继续持有" if lt_pred > 0 else "建议减仓观望"
            if lt_pred > 10:
                suggestion = "强烈建议持有/加仓"
            elif lt_pred < -10:
                suggestion = "强烈建议减仓/止损"
            st.info(f"**中长期预测 ({lt_period})**: 累计收益率预测 **{lt_pred:+.1f}%** | 评级: **{lt_rating}** | {suggestion}")
    else:
        st.info("请先训练模型生成预测")


# ── Tab 4: 模型评估 ──────────────────────────────────────────

with tab4:
    if st.session_state.train_results:
        results = st.session_state.train_results

        # 实际 vs 预测
        st.subheader("测试集: 实际值 vs 预测值")
        model_sel = st.selectbox("选择模型查看", list(results.keys()), key="eval_model")
        r = results[model_sel]

        has_test_data = len(r.test_returns) > 0 and len(r.test_returns_actual) > 0
        if not has_test_data:
            st.info("云端加载的结果不含测试集数据，仅展示模型指标")

        if has_test_data:
            n_test = len(r.test_returns_actual)
            df_full = st.session_state.stock_data
            test_dates = df_full.index[-n_test:] if df_full is not None and len(df_full) >= n_test else list(range(n_test))

            fig_vs = go.Figure()
            fig_vs.add_trace(go.Scatter(x=test_dates, y=r.test_returns_actual,
                                        name="实际值", line=dict(color="blue")))
            fig_vs.add_trace(go.Scatter(x=test_dates, y=r.test_returns,
                                        name="预测值", line=dict(color="red", dash="dash")))
            fig_vs.update_layout(height=400, title=f"{model_sel} - 测试集预测效果",
                                 xaxis_title="日期", yaxis_title="收益率(%)")
            st.plotly_chart(fig_vs, use_container_width=True)

        # 指标对比柱状图
        st.subheader("模型指标对比")
        metric_names = ["MAE", "RMSE", "MAPE(%)", "R²"]
        metric_keys = ["mae", "rmse", "mape", "r2"]
        fig_bar = make_subplots(rows=1, cols=4, subplot_titles=metric_names)
        for i, (mname, mkey) in enumerate(zip(metric_names, metric_keys)):
            vals = [results[m].cv_metrics.get(mkey) or 0 for m in results]
            fig_bar.add_trace(go.Bar(
                x=list(results.keys()), y=vals, name=mname,
                showlegend=False), row=1, col=i+1)
        fig_bar.update_layout(height=350, title="各模型评估指标对比")
        st.plotly_chart(fig_bar, use_container_width=True)

        # 回测
        st.subheader("方向预测准确率")
        if st.session_state.stock_data is None:
            st.info("云端结果无原始行情数据，无法进行方向回测")
        else:
            bt = backtest_predictions(st.session_state.stock_data, results, look_back)
            if not bt.empty:
                acc_by_model = bt.groupby("model")["direction_correct"].mean()
                fig_acc = go.Figure(go.Bar(
                    x=acc_by_model.index.tolist(),
                    y=(acc_by_model.values * 100).tolist(),
                    text=[f"{v:.1f}%" for v in acc_by_model.values * 100],
                    textposition="auto",
                ))
                fig_acc.add_hline(y=50, line_dash="dash", line_color="gray",
                                  annotation_text="随机基准 50%")
                fig_acc.update_layout(height=300, yaxis_title="准确率(%)",
                                      title="各模型方向预测准确率")
                st.plotly_chart(fig_acc, use_container_width=True)
    else:
        st.info("请先训练模型")


# ── Tab 5: 模型管理 ──────────────────────────────────────────

with tab5:
    st.subheader("模型版本管理")

    stock_code_mgr = st.session_state.stock_code or ""
    if stock_code_mgr:
        # 更新检查
        update_info = should_retrain(stock_code_mgr)
        for model_type, info in update_info.items():
            if info["needs_update"]:
                st.warning(f"{model_type}: {info['reason']}")

        # 滚动窗口更新
        col1, col2 = st.columns(2)
        with col1:
            window_size = st.slider("滚动窗口大小", 200, 1000, 500, key="rolling_window")
        with col2:
            if st.button("立即更新模型", type="primary"):
                if st.session_state.stock_data is not None:
                    with st.spinner("滚动窗口训练中..."):
                        model_config = _build_config()
                        new_results = rolling_train(
                            stock_code_mgr,
                            st.session_state.stock_data,
                            selected_models,
                            model_config,
                            window_size=window_size,
                            forecast_days=forecast_days,
                        )
                        st.session_state.train_results = new_results
                        if use_ensemble and len(new_results) > 1:
                            w = compute_ensemble_weights(new_results)
                            last_p = st.session_state.stock_data["close"].iloc[-1]
                            st.session_state.ensemble_weights = w
                            st.session_state.predictions = ensemble_predict(
                                new_results, w, forecast_days, last_p)
                    st.success("模型更新完成!")
                    st.rerun()
                else:
                    st.error("请先加载数据")

        # 已保存模型列表
        models = list_models(stock_code_mgr)
        if models:
            st.subheader(f"已保存模型（{stock_code_mgr}）")
            model_rows = []
            for m in models[:20]:
                model_rows.append({
                    "模型类型": m.model_type,
                    "训练日期": m.train_date[:19],
                    "RMSE": f"{m.metrics.get('rmse', 'N/A')}" if isinstance(m.metrics.get('rmse'), (int, float)) else "N/A",
                    "R²": f"{m.metrics.get('r2', 'N/A')}" if isinstance(m.metrics.get('r2'), (int, float)) else "N/A",
                    "文件大小": f"{m.file_size / 1024:.0f} KB",
                })
            st.dataframe(pd.DataFrame(model_rows), use_container_width=True, hide_index=True)

            # 清理旧模型
            if st.button("清理旧模型（保留最近10个）"):
                n = cleanup_old_models(stock_code_mgr, keep_latest=10)
                st.success(f"已删除 {n} 个旧模型")
                st.rerun()
        else:
            st.info("暂无已保存模型")

        # 性能监控
        perf = track_performance(stock_code_mgr)
        if not perf.empty:
            st.subheader("模型性能趋势")
            fig_perf = go.Figure()
            for mt in perf["model_type"].unique():
                sub = perf[perf["model_type"] == mt]
                fig_perf.add_trace(go.Scatter(
                    x=sub["train_date"], y=sub["rmse"],
                    name=mt, mode="lines+markers"))
            fig_perf.update_layout(height=350, title="RMSE 随时间变化",
                                   xaxis_title="训练日期", yaxis_title="RMSE")
            st.plotly_chart(fig_perf, use_container_width=True)

        # 训练参数记录
        last_config = st.session_state.get("_last_config")
        loaded_config = st.session_state.get("_loaded_config")
        if last_config or loaded_config:
            with st.expander("训练参数记录", expanded=False):
                cfg = loaded_config if loaded_config else last_config
                if isinstance(cfg, dict):
                    st.json(cfg)
                else:
                    st.json({
                        "look_back": cfg.look_back,
                        "n_features": cfg.n_features,
                        "epochs": cfg.epochs,
                        "batch_size": cfg.batch_size,
                        "learning_rate": cfg.learning_rate,
                        "dropout": cfg.dropout,
                        "lstm_units": cfg.lstm_units,
                        "gru_units": cfg.gru_units,
                        "cnn_filters": cfg.cnn_filters,
                        "cnn_gru_filters": cfg.cnn_gru_filters,
                        "cnn_gru_gru_units": cfg.cnn_gru_gru_units,
                        "xgboost_n_estimators": cfg.xgboost_n_estimators,
                        "xgboost_max_depth": cfg.xgboost_max_depth,
                        "lightgbm_n_estimators": cfg.lightgbm_n_estimators,
                        "lightgbm_max_depth": cfg.lightgbm_max_depth,
                        "sarima_order": cfg.sarima_order,
                        "sarima_seasonal_order": cfg.sarima_seasonal_order,
                    })
    else:
        st.info("请先加载数据")


# ── Tab 6: 结果导出 ──────────────────────────────────────────

with tab6:
    st.subheader("导出结果")

    if st.session_state.train_results or st.session_state.predictions:
        # 训练结果存档（JSON，可恢复）
        st.markdown("#### 训练结果存档")
        st.caption('下载后可在侧边栏「恢复历史结果」中上传，无需重新训练')
        result_bytes = _serialize_results()
        st.download_button(
            "下载训练结果存档 (.json)",
            data=result_bytes,
            file_name=f"results_{st.session_state.stock_code}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True,
        )

        st.divider()

        # Excel 导出
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
            # 原始数据
            if st.session_state.stock_data is not None:
                st.session_state.stock_data.to_excel(writer, sheet_name="原始数据")

            # 预测结果
            if st.session_state.predictions and len(st.session_state.predictions.get("predicted_close", [])) > 0:
                preds = st.session_state.predictions
                if st.session_state.stock_data is not None:
                    last_date = st.session_state.stock_data.index[-1]
                else:
                    last_date = pd.Timestamp.now()
                future_dates = pd.bdate_range(start=last_date + timedelta(days=1),
                                              periods=len(preds["predicted_close"]))
                pred_df = pd.DataFrame({
                    "日期": future_dates,
                    "预测收盘价": preds["predicted_close"],
                    "日收益率(%)": preds["daily_return"],
                    "累计收益率(%)": preds["cumulative_return"],
                })
                pred_df.to_excel(writer, sheet_name="预测结果", index=False)

                # 各模型预测
                if preds.get("model_predictions"):
                    mp_df = pd.DataFrame(preds["model_predictions"])
                    mp_df.insert(0, "日期", future_dates)
                    mp_df.to_excel(writer, sheet_name="各模型预测", index=False)

            # 评估指标
            if st.session_state.train_results:
                metrics_rows = []
                for name, r in st.session_state.train_results.items():
                    row = {"模型": name}
                    row.update(r.cv_metrics)
                    row["训练耗时(秒)"] = r.training_time
                    metrics_rows.append(row)
                pd.DataFrame(metrics_rows).to_excel(writer, sheet_name="评估指标", index=False)

            # 中长期预测结果
            if st.session_state.longterm_results is not None:
                lt = st.session_state.longterm_results
                ensemble = lt.get("_ensemble", {})
                lt_pred = ensemble.get("prediction", 0)
                lt_period = st.session_state.get("longterm_period", "N/A")

                # 汇总卡片
                lt_summary = pd.DataFrame({
                    "项目": ["预测周期", "集成预测收益率", "最新周收盘价", "预测目标价"],
                    "值": [lt_period, f"{lt_pred*100:+.2f}%",
                          f"¥{ensemble.get('latest_close', 0):.2f}",
                          f"¥{ensemble.get('latest_close', 0)*(1+lt_pred):.2f}"],
                })
                lt_summary.to_excel(writer, sheet_name="中长期预测", index=False, startrow=0)

                # 模型对比
                lt_model_rows = []
                for name, r in lt.items():
                    if name == "_ensemble":
                        continue
                    lt_model_rows.append({
                        "模型": name,
                        "预测收益率": f"{r['prediction']*100:+.2f}%",
                        "CV_RMSE": r.get("cv_rmse", "N/A"),
                        "CV_R²": r.get("cv_r2", "N/A"),
                        "方向准确率": f"{r.get('direction_accuracy', 0)*100:.1f}%",
                    })
                if lt_model_rows:
                    pd.DataFrame(lt_model_rows).to_excel(
                        writer, sheet_name="中长期预测", index=False,
                        startrow=len(lt_summary) + 3)

        buf.seek(0)
        st.download_button(
            "下载 Excel 报告",
            data=buf.getvalue(),
            file_name=f"prediction_{st.session_state.stock_code}_{datetime.now().strftime('%Y%m%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary",
        )

        # 图表导出提示
        st.info("图表可在各标签页中点击右上角相机图标直接下载 PNG")
    else:
        st.info("请先训练模型生成预测结果")


# ── Tab 7: 模型参数详情 ──────────────────────────────────────────

with tab7:
    st.subheader("模型参数详情")

    last_config = st.session_state.get("_last_config")
    if last_config:
        st.markdown("### 训练配置摘要")
        st.markdown(f"- 时间步长: **{last_config.look_back}** 天")
        st.markdown(f"- 训练轮次: **{last_config.epochs}**")
        st.markdown(f"- 批量大小: **{last_config.batch_size}**")
        st.markdown(f"- 学习率: **{last_config.learning_rate:.6f}**")
        st.markdown(f"- Dropout: **{last_config.dropout}**")

    if st.session_state.train_results:
        st.markdown("---")
        st.markdown("### 各模型参数表")

        all_models_list = [
            "LSTM", "GRU", "1D-CNN", "CNN-GRU", "PatchTST", "TFT",
            "XGBoost", "LightGBM", "ARIMA", "SARIMA", "GARCH",
        ]
        model_labels = {
            "LSTM": "LSTM", "GRU": "GRU", "1D-CNN": "1D-CNN",
            "CNN-GRU": "CNN-GRU", "PatchTST": "PatchTST", "TFT": "TFT",
            "XGBoost": "XGBoost", "LightGBM": "LightGBM",
            "ARIMA": "ARIMA", "SARIMA": "SARIMA", "GARCH": "GARCH",
        }

        mp = st.session_state.model_params
        for model_name in all_models_list:
            params = mp.get(model_name, {})
            defaults = DEFAULT_MODEL_PARAMS.get(model_name, {})
            if not params:
                continue

            is_trained = model_name in st.session_state.train_results
            status_icon = "✅" if is_trained else "⚪"
            is_modified = model_name in st.session_state.modified_models
            mod_icon = " 🔵" if is_modified else ""

            st.markdown(f"**{status_icon} {model_labels.get(model_name, model_name)}{mod_icon}**")

            rows = []
            for key, val in params.items():
                def_val = defaults.get(key, "—")
                if isinstance(val, float):
                    val_str = f"{val:.4f}" if val < 0.01 else f"{val:.2f}"
                else:
                    val_str = str(val)
                if isinstance(def_val, float):
                    def_str = f"{def_val:.4f}" if abs(def_val) < 0.01 else f"{def_val:.2f}"
                else:
                    def_str = str(def_val)
                rows.append({
                    "参数名称": key,
                    "当前值": val_str,
                    "默认值": def_str,
                    "已修改": "是" if val != def_val else "否",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # 训练结果汇总
        st.markdown("---")
        st.markdown("### 训练指标汇总")
        metric_rows = []
        for name, r in st.session_state.train_results.items():
            m = r.cv_metrics
            metric_rows.append({
                "模型": name,
                "MAE": f"{m.get('mae', 'N/A')}",
                "RMSE": f"{m.get('rmse', 'N/A')}",
                "R²": f"{m.get('r2', 'N/A')}",
                "训练耗时(s)": f"{r.training_time:.1f}",
            })
        st.dataframe(pd.DataFrame(metric_rows), use_container_width=True, hide_index=True)
    else:
        st.info("请先训练模型以查看参数详情")


# ── Tab 8: 中长期预测 ──────────────────────────────────────────

with tab8:
    st.subheader("1-3月中长期收益率预测")
    st.caption("基于周线数据，使用树模型（LightGBM/XGBoost/CatBoost）预测未来累计收益率")

    if st.session_state.stock_data is not None:
        df = st.session_state.stock_data

        # ── 参数设置 ──
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            period_label = st.radio(
                "预测周期",
                ["1个月 (4周)", "2个月 (8周)", "3个月 (12周)"],
                key="longterm_period",
                horizontal=True,
            )
        with col_p2:
            lt_models = st.multiselect(
                "选择模型",
                ["LightGBM", "XGBoost", "CatBoost", "LinearRegression"],
                default=["LightGBM", "XGBoost", "CatBoost"],
                key="longterm_models",
            )

        horizon_map = {"1个月 (4周)": 4, "2个月 (8周)": 8, "3个月 (12周)": 12}
        horizon_weeks = horizon_map[period_label]

        # ── 数据准备 ──
        weekly_info = resample_to_weekly(df)
        df_weekly = weekly_info["df_weekly"]
        data_warning = weekly_info["data_warning"]

        if data_warning:
            st.warning(data_warning)

        # ── 训练按钮 ──
        if st.button("开始中长期预测", type="primary", use_container_width=True,
                     disabled=len(lt_models) == 0):
            with st.spinner(f"训练中长期模型（{period_label}）..."):
                try:
                    lt_results = train_long_term_models(
                        df_weekly, lt_models, horizon_weeks,
                    )
                    st.session_state.longterm_results = lt_results
                    st.toast("中长期预测完成!")
                    st.rerun()
                except Exception as e:
                    st.error(f"训练失败: {e}")
                    import traceback
                    st.code(traceback.format_exc())

        # ── 结果展示 ──
        if st.session_state.longterm_results is not None:
            lt = st.session_state.longterm_results
            ensemble = lt.get("_ensemble", {})
            ensemble_pred = ensemble.get("prediction", 0)
            latest_close = ensemble.get("latest_close", df_weekly["收盘"].iloc[-1])

            st.divider()
            st.subheader("预测结果")

            # 1. 综合预测卡片
            pred_return_pct = ensemble_pred * 100  # 转百分比
            valid_models = [m for m in lt if m != "_ensemble" and not np.isnan(lt[m].get("cv_rmse", float("nan")))]
            avg_dir_acc = np.mean([lt[m]["direction_accuracy"] for m in valid_models]) if valid_models else 0.5
            rating = get_rating(pred_return_pct, avg_dir_acc)

            ci_lows = [lt[m]["confidence_interval"][0] * 100 for m in valid_models]
            ci_highs = [lt[m]["confidence_interval"][1] * 100 for m in valid_models]
            ci_low_pct = np.mean(ci_lows) if ci_lows else pred_return_pct * 0.5
            ci_high_pct = np.mean(ci_highs) if ci_highs else pred_return_pct * 1.5
            up_prob = avg_dir_acc

            rating_colors = {
                "强烈看涨": "#006600", "看涨": "#009900", "中性": "#666666",
                "看跌": "#cc0000", "强烈看跌": "#990000",
            }

            card_cols = st.columns(4)
            with card_cols[0]:
                delta_str = f"{pred_return_pct:+.1f}%"
                st.metric("预测累计收益率", delta_str)
            with card_cols[1]:
                st.metric("上涨概率", f"{up_prob*100:.0f}%")
            with card_cols[2]:
                st.metric("95%置信区间", f"{ci_low_pct:+.1f}% ~ {ci_high_pct:+.1f}%")
            with card_cols[3]:
                color = rating_colors.get(rating, "#666666")
                st.markdown(
                    f"<div style='text-align:center;font-size:1.5em;font-weight:bold;color:{color}'>{rating}</div>",
                    unsafe_allow_html=True)
                st.caption("综合评级")

            # 2. 模型对比表
            st.subheader("各模型预测对比")
            model_rows = []
            for name in lt_models:
                if name not in lt or name == "_ensemble":
                    continue
                r = lt[name]
                model_rows.append({
                    "模型": name,
                    "预测收益率": f"{r['prediction']*100:+.2f}%",
                    "RMSE": f"{r.get('cv_rmse', 'N/A'):.4f}" if not np.isnan(r.get('cv_rmse', float('nan'))) else "N/A",
                    "R²": f"{r.get('cv_r2', 'N/A'):.4f}" if not np.isnan(r.get('cv_r2', float('nan'))) else "N/A",
                    "方向准确率": f"{r.get('direction_accuracy', 0)*100:.1f}%",
                })
            model_rows.sort(key=lambda x: float(x["预测收益率"].replace("%", "").replace("+", "")), reverse=True)
            st.dataframe(pd.DataFrame(model_rows), use_container_width=True, hide_index=True)

            # 3. 趋势图（含斐波那契）
            st.subheader("周线趋势预测")
            hist_weeks = min(52, len(df_weekly))
            hist_data = df_weekly.tail(hist_weeks)
            hist_dates = hist_data.index
            hist_close = hist_data["收盘"].values

            future_close = latest_close * (1 + ensemble_pred)
            future_date = hist_dates[-1] + pd.Timedelta(weeks=horizon_weeks)

            fig_weekly = go.Figure()
            fig_weekly.add_trace(go.Scatter(
                x=hist_dates, y=hist_close,
                name="历史周线收盘价", line=dict(color="#1f77b4", width=2),
                mode="lines"))
            fig_weekly.add_trace(go.Scatter(
                x=[hist_dates[-1], future_date],
                y=[latest_close, future_close],
                name=f"预测({horizon_weeks}周)", line=dict(color="red", width=2.5, dash="dot"),
                mode="lines+markers"))

            ci_low_price = latest_close * (1 + ci_low_pct / 100)
            ci_high_price = latest_close * (1 + ci_high_pct / 100)
            fig_weekly.add_trace(go.Scatter(
                x=[hist_dates[-1], future_date, future_date, hist_dates[-1]],
                y=[ci_low_price, ci_low_price, ci_high_price, ci_high_price],
                fill="toself", fillcolor="rgba(255,0,0,0.1)",
                line=dict(color="rgba(255,0,0,0)"), name="95%置信区间"))

            # 斐波那契水平线
            try:
                wave_info = detect_wave_levels(df_weekly.rename(columns={"收盘": "close"}))
                fib_levels = calculate_wave_fibonacci(wave_info)
                for lv in fib_levels:
                    line_w = 2.5 if "黄金" in lv.get("type", "") else 1.0
                    fig_weekly.add_hline(y=lv["price"], line_dash="dash",
                        line_color=lv["color"], opacity=0.4,
                        annotation_text=f"{lv['name']}",
                        annotation_position="left", line_width=line_w)
            except Exception:
                pass

            fig_weekly.update_layout(height=450, title=f"周线收盘价 + {horizon_weeks}周预测",
                                    xaxis_title="日期", yaxis_title="价格(¥)",
                                    hovermode="x unified")
            st.plotly_chart(fig_weekly, use_container_width=True)

            # 4. 风险评估
            risk = assess_risk(df_weekly)
            st.subheader("风险评估")
            risk_cols = st.columns(4)
            risk_cols[0].metric("年化波动率", f"{risk['annual_vol_pct']}%")
            risk_cols[1].metric("最大回撤预期", f"{risk['max_drawdown_pct']}%")
            risk_level_color = {"低": "green", "中": "orange", "高": "red"}
            risk_cols[2].metric("风险等级", risk["risk_level"])

            # 5. 特征重要性
            st.subheader("特征重要性分析 (LightGBM)")
            lgb_result = lt.get("LightGBM", {})
            lgb_feat_imp = lgb_result.get("feature_importance", {})
            if lgb_feat_imp:
                sorted_imp = sorted(lgb_feat_imp.items(), key=lambda x: x[1], reverse=True)[:10]
                fig_imp = go.Figure(go.Bar(
                    x=[v for _, v in sorted_imp],
                    y=[k for k, _ in sorted_imp],
                    orientation="h",
                    marker_color="#1f77b4",
                ))
                fig_imp.update_layout(height=350, title="Top 10 特征重要性",
                                     xaxis_title="重要性", yaxis_title="特征")
                st.plotly_chart(fig_imp, use_container_width=True)
            else:
                st.info("LightGBM 未训练或无法获取特征重要性")
    else:
        st.info("请先在侧边栏加载数据")


# ═══════ 导出按钮处理 ═══════

if btn_export and st.session_state.train_results:
    st.toast("请切换到'结果导出'标签页下载文件")
