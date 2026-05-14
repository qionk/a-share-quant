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
from datetime import datetime, timedelta, date

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.predict.data_input import (
    load_from_akshare, load_from_excel, generate_template, get_stock_name,
)
from src.predict.features import compute_technical_indicators, prepare_features, create_sequences
from src.predict.preprocessing import preprocess_data
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
    list_stocks_with_status, list_stock_sessions, delete_stock_data,
)
from src.predict.fibonacci_wave import (
    detect_wave_levels, calculate_wave_fibonacci, generate_wave_fib_signals,
)
from src.predict.long_term_prediction import (
    resample_to_weekly, train_long_term_models,
    assess_risk, get_rating,
)
from src.predict.ensemble_classifier import (
    run_classifier_pipeline, get_recommended_params,
    check_params_deviation, CLF_FEATURE_COLS, create_clf_features,
    calculate_classification_metrics,
)
from src.data import load_config

# ═══════ 页面配置 ═══════

st.set_page_config(page_title="A股价格预测", page_icon="📈", layout="wide")
st.title("A股价格预测工具")
st.caption("支持 LSTM / GRU / 1D-CNN / CNN-GRU / PatchTST / TFT / XGBoost / LightGBM / ARIMA / SARIMA / GARCH 多模型集成预测")

config = load_config()
predict_cfg = config.get("predict", {})


def validate_target_variable(df: pd.DataFrame = None, silent: bool = False):
    """校验日收益率是否为小数形式（非百分比）。

    在启动时和数据加载后调用，防止回退到价格预测或百分比形式。
    返回 (is_valid: bool, message: str)
    """
    # 检查默认模型参数中是否有价格相关配置残留
    default_price_keys = ["predicted_close", "last_price", "returns_to_price"]
    for key in default_price_keys:
        if key in predict_cfg:
            return False, f"配置中包含已废弃的价格字段 '{key}'，请清理"

    # 检查数据中的日收益率范围
    if df is not None and "日收益率" in df.columns:
        returns = df["日收益率"].dropna()
        if len(returns) > 0:
            abs_mean = abs(returns).mean()
            abs_max = abs(returns).max()

            # 小数形式：均值约0.005-0.03，最大值<0.11（A股涨跌停限制）
            if abs_mean > 1.0:
                return False, (
                    f"日收益率均值为 {abs_mean:.4f}，疑似百分比形式（应为小数）。"
                    f"请检查 preprocessing.py 中是否误乘了 100。"
                )
            if abs_max > 0.15:
                return False, (
                    f"日收益率最大绝对值为 {abs_max:.4f}，超出A股涨跌停范围。"
                    f"请检查数据是否存在异常值。"
                )

    if not silent:
        print("[校验] 日收益率格式检查通过（小数形式）")
    return True, "ok"


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
    "CNN-GRU": {"cnn_filters": 32, "kernel_size": 3, "gru_units": 24, "dropout": 0.2, "look_back": 30, "learning_rate": 0.0006},
    "GRU": {"units": 32, "look_back": 30, "dropout": 0.2, "learning_rate": 0.001},
    "LSTM": {"units": 32, "look_back": 30, "dropout": 0.2, "learning_rate": 0.001},
    "PatchTST": {"d_model": 128, "n_heads": 4, "n_layers": 2, "patch_size": 16, "dropout": 0.1, "look_back": 30, "learning_rate": 0.001},
    "TFT": {"hidden_size": 64, "n_heads": 4, "dropout": 0.2, "lstm_layers": 1, "look_back": 30, "learning_rate": 0.001},
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

# ── 涨跌预测模块状态 ──
if "clf_results" not in st.session_state:
    st.session_state.clf_results = None
if "clf_params" not in st.session_state:
    st.session_state.clf_params = {
        "XGBoost": {
            "n_estimators": 100, "max_depth": 6, "learning_rate": 0.1,
            "subsample": 0.8, "colsample_bytree": 0.8,
            "min_child_weight": 1, "reg_alpha": 0.0, "reg_lambda": 1.0,
        },
        "ElasticNet": {"C": 1.0, "l1_ratio": 0.15, "max_iter": 5000, "tol": 1e-3},
    }
if "clf_recommended_params" not in st.session_state:
    st.session_state.clf_recommended_params = None
if "clf_selected_models" not in st.session_state:
    st.session_state.clf_selected_models = ["XGBoost", "ElasticNet"]
if "clf_look_back" not in st.session_state:
    st.session_state.clf_look_back = 20
if "clf_n_splits" not in st.session_state:
    st.session_state.clf_n_splits = 5
if "clf_modified_models" not in st.session_state:
    st.session_state.clf_modified_models = set()
if "clf_training_active" not in st.session_state:
    st.session_state.clf_training_active = False
if "clf_forecast_days" not in st.session_state:
    st.session_state.clf_forecast_days = 1
if "clf_threshold" not in st.session_state:
    st.session_state.clf_threshold = 0.50
if "clf_ensemble_result" not in st.session_state:
    st.session_state.clf_ensemble_result = None
if "clf_results_stock_code" not in st.session_state:
    st.session_state.clf_results_stock_code = None
if "clf_autotune_active" not in st.session_state:
    st.session_state.clf_autotune_active = False
if "clf_autotune_results" not in st.session_state:
    st.session_state.clf_autotune_results = None


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
    new_lb = st.slider("时间步长", 1, 60, params["look_back"], key="dg_cnn_lb")
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
    new_lb = st.slider("时间步长", 1, 60, params["look_back"], key="dg_cg_lb")
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
            "gru_units": new_gu, "look_back": new_lb,
            "dropout": new_do, "learning_rate": new_lr}
        _param_changed("CNN-GRU", "cnn_filters", new_cf, defaults["cnn_filters"])
        st.rerun()


@st.dialog("GRU 参数设置")
def gru_dialog():
    params = st.session_state.model_params["GRU"]
    defaults = DEFAULT_MODEL_PARAMS["GRU"]
    new_un = st.slider("神经元", 16, 64, params["units"], 8, key="dg_gru_un")
    new_lb = st.slider("时间步长", 1, 60, params["look_back"], key="dg_gru_lb")
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
    new_lb = st.slider("时间步长", 1, 60, params["look_back"], key="dg_lstm_lb")
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
    new_lb = st.slider("时间步长", 1, 60, params["look_back"], key="dg_pt_lb")
    new_do = st.slider("Dropout", 0.05, 0.3, params["dropout"], 0.05, key="dg_pt_do")
    new_lr = st.slider("学习率", 0.0001, 0.005, params["learning_rate"], 0.0001, format="%.4f", key="dg_pt_lr")
    c1, c2 = st.columns(2)
    if c1.button("恢复默认", use_container_width=True):
        st.session_state.model_params["PatchTST"] = dict(defaults)
        st.session_state.modified_models.discard("PatchTST")
        st.rerun()
    if c2.button("确认保存", use_container_width=True, type="primary"):
        st.session_state.model_params["PatchTST"] = {
            "d_model": new_dm, "n_heads": new_nh, "n_layers": new_nl,
            "patch_size": new_ps, "look_back": new_lb,
            "dropout": new_do, "learning_rate": new_lr}
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
    new_lb = st.slider("时间步长", 1, 60, params["look_back"], key="dg_tft_lb")
    new_lr = st.slider("学习率", 0.0001, 0.005, params["learning_rate"], 0.0001, format="%.4f", key="dg_tft_lr")
    c1, c2 = st.columns(2)
    if c1.button("恢复默认", use_container_width=True):
        st.session_state.model_params["TFT"] = dict(defaults)
        st.session_state.modified_models.discard("TFT")
        st.rerun()
    if c2.button("确认保存", use_container_width=True, type="primary"):
        st.session_state.model_params["TFT"] = {
            "hidden_size": new_hs, "n_heads": new_nh,
            "dropout": new_do, "lstm_layers": new_nl,
            "look_back": new_lb, "learning_rate": new_lr}
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


# ── 涨跌预测分类器参数对话框 ──

@st.dialog("XGBoost 分类器参数设置")
def clf_xgboost_dialog():
    params = st.session_state.clf_params["XGBoost"]
    defaults = {
        "n_estimators": 100, "max_depth": 6, "learning_rate": 0.1,
        "subsample": 0.8, "colsample_bytree": 0.8,
        "min_child_weight": 1, "reg_alpha": 0.0, "reg_lambda": 1.0,
    }

    new_lr = st.number_input("学习率 learning_rate", min_value=0.001, max_value=0.50, value=params.get("learning_rate", 0.1), step=0.001, key="dg_clf_xgb_lr", format="%.3f")
    new_n = st.number_input("树数量 n_estimators", min_value=10, max_value=1000, value=params.get("n_estimators", 100), step=10, key="dg_clf_xgb_n")
    new_md = st.number_input("最大深度 max_depth", min_value=2, max_value=15, value=params.get("max_depth", 6), step=1, key="dg_clf_xgb_md")
    new_ss = st.number_input("子样本比例 subsample", min_value=0.3, max_value=1.0, value=params.get("subsample", 0.8), step=0.05, key="dg_clf_xgb_ss", format="%.2f")
    new_cbt = st.number_input("列采样比例 colsample_bytree", min_value=0.3, max_value=1.0, value=params.get("colsample_bytree", 0.8), step=0.05, key="dg_clf_xgb_cbt", format="%.2f")
    new_mcw = st.number_input("最小子节点权重 min_child_weight", min_value=0, max_value=20, value=params.get("min_child_weight", 1), step=1, key="dg_clf_xgb_mcw")
    new_ra = st.number_input("L1正则 reg_alpha", min_value=0.0, max_value=10.0, value=params.get("reg_alpha", 0.0), step=0.01, key="dg_clf_xgb_ra", format="%.2f")
    new_rl = st.number_input("L2正则 reg_lambda", min_value=0.0, max_value=10.0, value=params.get("reg_lambda", 1.0), step=0.01, key="dg_clf_xgb_rl", format="%.2f")

    c1, c2 = st.columns(2)
    if c1.button("恢复默认", use_container_width=True, key="clf_xgb_reset"):
        st.session_state.clf_params["XGBoost"] = dict(defaults)
        st.session_state.clf_modified_models.discard("XGBoost")
        st.rerun()
    if c2.button("确认保存", use_container_width=True, type="primary", key="clf_xgb_save"):
        st.session_state.clf_params["XGBoost"] = {
            "n_estimators": new_n, "max_depth": new_md,
            "learning_rate": new_lr, "subsample": new_ss,
            "colsample_bytree": new_cbt, "min_child_weight": new_mcw,
            "reg_alpha": new_ra, "reg_lambda": new_rl,
        }
        all_default = all(
            st.session_state.clf_params["XGBoost"].get(k) == defaults.get(k)
            for k in defaults
        )
        if all_default:
            st.session_state.clf_modified_models.discard("XGBoost")
        else:
            st.session_state.clf_modified_models.add("XGBoost")
        st.rerun()


@st.dialog("ElasticNet 分类器参数设置")
def clf_elasticnet_dialog():
    params = st.session_state.clf_params["ElasticNet"]
    defaults = {"C": 1.0, "l1_ratio": 0.15, "max_iter": 5000, "tol": 1e-3}

    new_C = st.number_input("正则化强度 C (越小越强)", min_value=0.0, max_value=2.0, value=params.get("C", 1.0), step=0.01,
                      key="dg_clf_en_c", format="%.2f")
    new_l1 = st.number_input("L1比例 l1_ratio", min_value=0.0, max_value=1.0, value=params.get("l1_ratio", 0.15), step=0.01,
                       key="dg_clf_en_l1", format="%.2f")
    new_mi = st.number_input("最大迭代次数 max_iter", min_value=500, max_value=20000, value=params.get("max_iter", 5000), step=500,
                       key="dg_clf_en_mi")
    new_tol = st.number_input("收敛容差 tol", min_value=1e-6, max_value=1e-2, value=params.get("tol", 1e-3), step=1e-4,
                              key="dg_clf_en_tol", format="%.6f")

    c1, c2 = st.columns(2)
    if c1.button("恢复默认", use_container_width=True, key="clf_en_reset"):
        st.session_state.clf_params["ElasticNet"] = dict(defaults)
        st.session_state.clf_modified_models.discard("ElasticNet")
        st.rerun()
    if c2.button("确认保存", use_container_width=True, type="primary", key="clf_en_save"):
        st.session_state.clf_params["ElasticNet"] = {
            "C": new_C, "l1_ratio": new_l1, "max_iter": new_mi, "tol": new_tol}
        all_default = all(
            st.session_state.clf_params["ElasticNet"].get(k) == defaults.get(k)
            for k in defaults
        )
        if all_default:
            st.session_state.clf_modified_models.discard("ElasticNet")
        else:
            st.session_state.clf_modified_models.add("ElasticNet")
        st.rerun()


# ═══════ 侧边栏 ═══════

btn_train = False
btn_clf_train = False
btn_clf_autotune = False
btn_clf_optuna = False

with st.sidebar:
    st.header("配置参数")

    # 数据来源
    st.subheader("数据输入")
    data_source = st.radio("数据来源", ["数据库加载", "Excel上传"], horizontal=True)

    if data_source == "数据库加载":
        # 日期区间选择（max_value 设为明天，确保今天可选）
        _tomorrow = date.today() + __import__('datetime').timedelta(days=1)
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            custom_start = st.date_input("起始日期", value=date(2020, 1, 1),
                                         min_value=date(1990, 1, 1), max_value=_tomorrow,
                                         key="custom_start_date")
        with col_d2:
            custom_end = st.date_input("结束日期", value=date.today(),
                                       min_value=date(1990, 1, 1), max_value=_tomorrow,
                                       key="custom_end_date")
        start_d = custom_start.strftime("%Y%m%d")
        end_d = custom_end.strftime("%Y%m%d")

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
                            _is_fresh = info["end_date"] >= date.today().strftime("%Y-%m-%d")
                            if not _is_fresh:
                                st.toast(f"正在更新 {info['name']} 的数据...")
                            # 若要求的起始日早于DB最早日，则删除重建
                            if start_d < info["start_date"]:
                                delete_stock_data(info["code"])
                            df, _, _ = fetch_and_store(info["code"], start_date=start_d,
                                                        end_date=end_d, max_days=10000)
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
                                        st.session_state.clf_results = None
                                        st.session_state.clf_ensemble_result = None
                                except Exception:
                                    st.session_state.train_results = None
                                    st.session_state.predictions = None
                                    st.session_state.clf_results = None
                                    st.session_state.clf_ensemble_result = None
                            else:
                                st.session_state.train_results = None
                                st.session_state.predictions = None
                                st.session_state.clf_results = None
                                st.session_state.clf_ensemble_result = None

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
                st.info("该股票数据已存在，刷新列表即可看到。如需重新获取请点击下方按钮")
                btn_label = "重新获取数据"
            else:
                btn_label = "获取数据"
            if st.button(btn_label, type="primary", use_container_width=True, key="fetch_new_btn"):
                with st.spinner(f"正在获取 {new_code} ({start_d} ~ {end_d}) 数据..."):
                    try:
                        if exists:
                            delete_stock_data(new_code)
                        df, name, _ = fetch_and_store(new_code, start_date=start_d,
                                                       end_date=end_d, max_days=10000)
                        st.session_state.stock_data = df
                        st.session_state.stock_code = new_code
                        st.session_state.stock_name = name
                        st.session_state.train_results = None
                        st.session_state.predictions = None
                        st.session_state.clf_results = None
                        st.session_state.clf_ensemble_result = None
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
                    st.session_state.clf_results = None
                    st.session_state.clf_ensemble_result = None
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
    look_back = st.slider("时间步长(天)", 1, 60, predict_cfg.get("default_look_back", 30))

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

    # ── 涨跌预测设置 ──
    st.divider()
    st.subheader("涨跌预测设置")

    clf_model_options = ["XGBoost", "ElasticNet"]
    st.session_state.clf_selected_models = st.multiselect(
        "分类模型", clf_model_options,
        default=st.session_state.clf_selected_models,
        help="XGBoost 二分类器 + ElasticNet LogisticRegression")

    st.session_state.clf_look_back = st.number_input(
        "特征回溯天数", min_value=1, max_value=60, value=st.session_state.clf_look_back,
        step=1, key="clf_look_back_slider",
        help="每个样本使用过去 N 天的特征")

    st.session_state.clf_n_splits = st.number_input(
        "扩展窗口折数", min_value=3, max_value=10, value=st.session_state.clf_n_splits,
        step=1, key="clf_n_splits_slider",
        help="时间序列扩展窗口验证的折数")

    st.session_state.clf_forecast_days = st.number_input(
        "预测持有天数", min_value=1, max_value=20, value=st.session_state.clf_forecast_days,
        step=1, key="clf_forecast_days_slider",
        help="T日收盘买入，持有N天后T+N日收盘卖出。1=次日卖出")

    st.session_state.clf_threshold = st.number_input(
        "概率阈值", min_value=0.30, max_value=0.70, value=st.session_state.clf_threshold,
        step=0.01, format="%.2f", key="clf_threshold_slider",
        help="融合概率 >= 阈值时做多，否则空仓（默认0.5）")

    st.caption("分类器参数（🔵 = 已修改）")
    c1, c2 = st.columns(2)
    with c1:
        mod_xgb = " 🔵" if "XGBoost" in st.session_state.clf_modified_models else ""
        if st.button(f"⚙ XGBoost{mod_xgb}", key="set_clf_xgb", use_container_width=True):
            clf_xgboost_dialog()
    with c2:
        mod_en = " 🔵" if "ElasticNet" in st.session_state.clf_modified_models else ""
        if st.button(f"⚙ ElasticNet{mod_en}", key="set_clf_en", use_container_width=True):
            clf_elasticnet_dialog()

    # 智能推荐
    st.divider()
    btn_clf_recommend = st.button("智能推荐", key="clf_smart_recommend", use_container_width=True)
    if btn_clf_recommend:
        if st.session_state.stock_data is not None:
            n_samples = len(st.session_state.stock_data)
            n_features = len([c for c in CLF_FEATURE_COLS if c in st.session_state.stock_data.columns])
            st.session_state.clf_recommended_params = get_recommended_params(n_samples, n_features)
            st.toast(f"推荐模式: {st.session_state.clf_recommended_params['mode']}")

            st.session_state.clf_params["XGBoost"] = dict(
                st.session_state.clf_recommended_params["xgb"])
            st.session_state.clf_params["ElasticNet"] = dict(
                st.session_state.clf_recommended_params["elasticnet"])
            st.session_state.clf_look_back = st.session_state.clf_recommended_params["look_back"]
            st.session_state.clf_n_splits = st.session_state.clf_recommended_params["n_splits"]
            st.rerun()
        else:
            st.warning("请先加载数据")

    if st.session_state.clf_recommended_params is not None:
        rec = st.session_state.clf_recommended_params
        st.info(f"当前推荐: **{rec['mode']}** 模式 (样本数反馈)")
        cur = st.session_state.clf_params
        dev_warnings = check_params_deviation(cur, rec)
        if dev_warnings:
            for w in dev_warnings:
                st.warning(w)

    # 训练触发
    if st.session_state.clf_training_active:
        st.warning("涨跌预测训练中...")
    else:
        btn_clf_train = st.button("开始涨跌训练", type="primary", use_container_width=True,
            disabled=st.session_state.stock_data is None or len(st.session_state.clf_selected_models) == 0)

    # 自动调参
    st.divider()
    st.caption("自动调参")
    clf_target_auc = st.number_input("目标AUC", min_value=0.50, max_value=0.70,
        value=0.53, step=0.01, format="%.2f", key="clf_target_auc")
    _at_c1, _at_c2 = st.columns(2)
    with _at_c1:
        clf_max_trials = st.number_input("随机次数", min_value=5, max_value=100,
            value=30, step=5, key="clf_max_trials")
    with _at_c2:
        clf_max_trials_optuna = st.number_input("贝叶斯次数", min_value=5, max_value=100,
            value=15, step=5, key="clf_max_trials_optuna")
    _at_b1, _at_b2 = st.columns(2)
    with _at_b1:
        btn_clf_autotune = st.button("随机调参", key="btn_clf_autotune", use_container_width=True,
            disabled=st.session_state.stock_data is None or st.session_state.clf_autotune_active)
    with _at_b2:
        btn_clf_optuna = st.button("智能调参(贝叶斯)", key="btn_clf_optuna", use_container_width=True,
        disabled=st.session_state.stock_data is None or st.session_state.clf_autotune_active)


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
        cnn_gru_gu = [qm.get("cnn_gru_gru_units", [24])[0]]
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
        cnn_gru_gu = [cg_p["gru_units"]]
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

    # dropout：从第一个选中的DL模型获取，或使用默认值
    dropout = dl_cfg.get("dropout", 0.2)
    dl_selected = [m for m in selected_models if m in ("LSTM", "GRU", "1D-CNN", "CNN-GRU", "PatchTST", "TFT")]
    if dl_selected:
        first_dl = dl_selected[0]
        dl_params = mp.get(first_dl, {})
        dropout = dl_params.get("dropout", dropout)

    def _get_lr(model_name):
        return mp.get(model_name, {}).get("learning_rate", DL_LEARNING_RATE)

    return ModelConfig(
        look_back=look_back,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=DL_LEARNING_RATE,
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
        # Per-model DL learning rates
        lstm_lr=_get_lr("LSTM"),
        gru_lr=_get_lr("GRU"),
        cnn_lr=_get_lr("1D-CNN"),
        cnn_gru_lr=_get_lr("CNN-GRU"),
        patchtst_lr=_get_lr("PatchTST"),
        tft_lr=_get_lr("TFT"),
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
            preds = ensemble_predict(results, weights, forecast_days)
            st.session_state.ensemble_weights = weights
            st.session_state.predictions = preds
        elif results:
            weights = {list(results.keys())[0]: 1.0}
            st.session_state.ensemble_weights = weights
            preds = ensemble_predict(results, weights, forecast_days)
            st.session_state.predictions = preds

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


# ═══════ 涨跌预测训练触发 ═══════

if btn_clf_train and st.session_state.stock_data is not None:
    st.session_state.clf_training_active = True

    st.subheader("涨跌预测训练进度")
    clf_progress_bar = st.progress(0, text="准备训练...")
    clf_status = st.empty()

    def clf_progress_cb(pct, msg):
        clf_progress_bar.progress(min(pct, 1.0), text=msg)
        clf_status.markdown(msg)

    try:
        clf_progress_cb(0.0, "数据预处理中...")

        results, ensemble_result = run_classifier_pipeline(
            df=st.session_state.stock_data,
            selected_models=st.session_state.clf_selected_models,
            params=st.session_state.clf_params,
            look_back=st.session_state.clf_look_back,
            n_splits=st.session_state.clf_n_splits,
            progress_cb=clf_progress_cb,
            forecast_days=st.session_state.clf_forecast_days,
            threshold=st.session_state.clf_threshold,
            stock_code=st.session_state.stock_code,
        )

        st.session_state.clf_results = results
        st.session_state.clf_ensemble_result = ensemble_result
        st.session_state.clf_results_stock_code = st.session_state.stock_code

        # 保存到历史库
        try:
            from src.predict.clf_history_store import save_clf_session
            _clf_stock_code = st.session_state.stock_code
            _clf_stock_name = st.session_state.get("stock_name", _clf_stock_code)
            session_id = save_clf_session(
                _clf_stock_code, _clf_stock_name,
                {
                    "forecast_days": st.session_state.clf_forecast_days,
                    "threshold": st.session_state.clf_threshold,
                    "look_back": st.session_state.clf_look_back,
                    "n_splits": st.session_state.clf_n_splits,
                    "selected_models": st.session_state.clf_selected_models,
                    "params": st.session_state.clf_params,
                    "results": results,
                    "ensemble_result": ensemble_result,
                    "data_dates": (st.session_state.stock_data.index[0], st.session_state.stock_data.index[-1]),
                },
            )
            if session_id > 0:
                st.toast(f"预测结果已保存 (Session #{session_id})")
        except Exception as save_err:
            st.warning(f"保存历史记录失败: {save_err}")

        clf_progress_bar.progress(1.0, text="涨跌预测训练完成!")
        st.toast("涨跌预测训练完成!")

    except Exception as e:
        st.error(f"涨跌预测训练失败: {e}")
        import traceback
        st.code(traceback.format_exc())

    st.session_state.clf_training_active = False
    st.rerun()


# ═══════ 涨跌预测自动调参执行 ═══════

if btn_clf_autotune and st.session_state.stock_data is not None:
    st.session_state.clf_autotune_active = True
    from src.predict.ensemble_classifier import auto_tune_classifier

    st.subheader("自动调参进行中")
    _tune_progress = st.progress(0, text="准备中...")
    _tune_table_placeholder = st.empty()
    _tune_trials_display = []

    def _tune_trial_cb(trial_idx, max_t, result):
        _tune_progress.progress(trial_idx / max_t, text=f"试验 {trial_idx}/{max_t} | 当前AUC={result['auc']:.4f}")
        _tune_trials_display.append({
            "#": result["trial"],
            "look_back": result["look_back"],
            "forecast": result["forecast_days"],
            "n_splits": result["n_splits"],
            "lr": result["lr"],
            "depth": result["depth"],
            "n_est": result["n_est"],
            "AUC": f"{result['auc']:.4f}",
            "耗时(s)": result["elapsed"],
        })
        _tune_table_placeholder.dataframe(
            pd.DataFrame(_tune_trials_display), use_container_width=True, hide_index=True)

    try:
        tune_result = auto_tune_classifier(
            df=st.session_state.stock_data,
            stock_code=st.session_state.stock_code,
            target_auc=clf_target_auc,
            max_trials=int(clf_max_trials),
            selected_models=st.session_state.clf_selected_models,
            trial_cb=_tune_trial_cb,
        )

        if tune_result["best_params"] is not None:
            bp = tune_result["best_params"]
            st.session_state.clf_look_back = bp["look_back"]
            st.session_state.clf_forecast_days = bp["forecast_days"]
            st.session_state.clf_n_splits = bp["n_splits"]
            st.session_state.clf_params["XGBoost"] = {
                "learning_rate": bp["xgb_learning_rate"],
                "n_estimators": bp["xgb_n_estimators"],
                "max_depth": bp["xgb_max_depth"],
                "subsample": bp["xgb_subsample"],
                "colsample_bytree": bp["xgb_colsample_bytree"],
                "min_child_weight": bp["xgb_min_child_weight"],
                "reg_alpha": bp["xgb_reg_alpha"],
                "reg_lambda": bp["xgb_reg_lambda"],
            }
            st.session_state.clf_params["ElasticNet"] = {
                "C": bp["en_C"],
                "l1_ratio": bp["en_l1_ratio"],
                "max_iter": 5000,
                "tol": 1e-3,
            }

            if tune_result["found"]:
                _tune_progress.progress(1.0, text=f"找到 AUC={tune_result['best_auc']:.4f} 的参数!")
                st.success(f"找到 AUC={tune_result['best_auc']:.2%} 的参数组合！已自动填入侧边栏。")
            else:
                _tune_progress.progress(1.0, text=f"未达标，最佳 AUC={tune_result['best_auc']:.4f}")
                st.warning(f"{int(clf_max_trials)}次试验后最佳 AUC={tune_result['best_auc']:.2%}，未达到目标。已填入最佳参数。")
        else:
            st.error("所有试验均失败，请检查数据。")

        st.session_state.clf_autotune_results = tune_result.get("trials")

    except Exception as e:
        st.error(f"自动调参失败: {e}")
        import traceback
        st.code(traceback.format_exc())

    st.session_state.clf_autotune_active = False


# ═══════ 涨跌预测贝叶斯调参执行 ═══════

if btn_clf_optuna and st.session_state.stock_data is not None:
    st.session_state.clf_autotune_active = True
    from src.predict.ensemble_classifier import auto_tune_optuna

    st.subheader("智能调参 (贝叶斯优化)")
    _opt_progress = st.progress(0, text="准备中...")
    _opt_table_placeholder = st.empty()
    _opt_trials_display = []

    def _optuna_trial_cb(trial_idx, max_t, result):
        _opt_progress.progress(trial_idx / max_t, text=f"试验 {trial_idx}/{max_t} | 当前AUC={result['auc']:.4f}")
        _opt_trials_display.append({
            "#": result["trial"],
            "look_back": result["look_back"],
            "forecast": result["forecast_days"],
            "n_splits": result["n_splits"],
            "lr": result["lr"],
            "depth": result["depth"],
            "n_est": result["n_est"],
            "AUC": f"{result['auc']:.4f}",
            "耗时(s)": result["elapsed"],
        })
        _opt_table_placeholder.dataframe(
            pd.DataFrame(_opt_trials_display), use_container_width=True, hide_index=True)

    try:
        tune_result = auto_tune_optuna(
            df=st.session_state.stock_data,
            stock_code=st.session_state.stock_code,
            target_auc=clf_target_auc,
            max_trials=int(clf_max_trials_optuna),
            selected_models=st.session_state.clf_selected_models,
            trial_cb=_optuna_trial_cb,
        )

        if tune_result["best_params"] is not None:
            bp = tune_result["best_params"]
            st.session_state.clf_look_back = bp["look_back"]
            st.session_state.clf_forecast_days = bp["forecast_days"]
            st.session_state.clf_n_splits = bp["n_splits"]
            st.session_state.clf_params["XGBoost"] = {
                "learning_rate": bp["xgb_learning_rate"],
                "n_estimators": bp["xgb_n_estimators"],
                "max_depth": bp["xgb_max_depth"],
                "subsample": bp["xgb_subsample"],
                "colsample_bytree": bp["xgb_colsample_bytree"],
                "min_child_weight": bp["xgb_min_child_weight"],
                "reg_alpha": bp["xgb_reg_alpha"],
                "reg_lambda": bp["xgb_reg_lambda"],
            }
            st.session_state.clf_params["ElasticNet"] = {
                "C": bp["en_C"],
                "l1_ratio": bp["en_l1_ratio"],
                "max_iter": 5000,
                "tol": 1e-3,
            }

            if tune_result["found"]:
                _opt_progress.progress(1.0, text=f"找到 AUC={tune_result['best_auc']:.4f} 的参数!")
                st.success(f"贝叶斯优化找到 AUC={tune_result['best_auc']:.2%} 的参数组合！已自动填入。")
            else:
                _opt_progress.progress(1.0, text=f"未达标，最佳 AUC={tune_result['best_auc']:.4f}")
                st.warning(f"{int(clf_max_trials_optuna)}次试验后最佳 AUC={tune_result['best_auc']:.2%}，未达到目标。已填入最佳参数。")
        else:
            st.error("所有试验均失败，请检查数据。")

        st.session_state.clf_autotune_results = tune_result.get("trials")

    except ImportError:
        st.error("需要安装 optuna: `pip install optuna`")
    except Exception as e:
        st.error(f"贝叶斯调参失败: {e}")
        import traceback
        st.code(traceback.format_exc())

    st.session_state.clf_autotune_active = False


tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs(
    ["数据概览", "模型训练", "预测结果", "模型评估", "模型管理", "结果导出", "模型参数详情", "中长期预测", "涨跌预测"]
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
            st.session_state.clf_results = None
            st.session_state.clf_ensemble_result = None
            st.rerun()
    else:
        st.info("请先在侧边栏点击「训练所有模型」按钮开始训练")
        if st.session_state.stock_data is not None:
            st.markdown(f"**数据量**: {len(st.session_state.stock_data)} 天")
            st.markdown(f"**已选模型**: {', '.join(selected_models)}")


# ── Tab 3: 预测结果 ──────────────────────────────────────────

with tab3:
    if st.session_state.predictions and st.session_state.predictions.get("predicted_return") is not None \
       and len(st.session_state.predictions["predicted_return"]) > 0:
        preds = st.session_state.predictions

        if st.session_state.stock_data is not None:
            last_price = st.session_state.stock_data["close"].iloc[-1]
            last_date = st.session_state.stock_data.index[-1]
        else:
            last_price = 10.0
            last_date = pd.Timestamp.now()

        # 生成未来交易日日期
        future_dates = pd.bdate_range(start=last_date + timedelta(days=1), periods=forecast_days)

        # 预测结果表
        st.subheader("未来预测结果")
        pred_table = pd.DataFrame({
            "日期": future_dates.strftime("%Y-%m-%d"),
            "预测日收益率": [f"{r*100:+.2f}%" for r in preds["predicted_return"]],
            "日收益率": [f"{r*100:+.2f}%" for r in preds["daily_return"]],
            "累计收益率": [f"{r*100:+.2f}%" for r in preds["cumulative_return"]],
        })
        st.dataframe(pred_table, use_container_width=True, hide_index=True)

        # 关键指标
        c1, c2, c3 = st.columns(3)
        total_ret = preds["cumulative_return"][-1]
        c1.metric("预测累计收益", f"{total_ret*100:+.2f}%",
                  delta="看涨" if total_ret > 0 else "看跌")
        c2.metric("最高日收益率", f"{max(preds['predicted_return'])*100:+.2f}%")
        c3.metric("最低日收益率", f"{min(preds['predicted_return'])*100:+.2f}%")

        # 收益率柱状图
        fig_ret = go.Figure()
        colors = ["red" if r > 0 else "green" for r in preds["daily_return"]]
        fig_ret.add_trace(go.Bar(
            x=future_dates.strftime("%m-%d"),
            y=preds["daily_return"] * 100,
            marker_color=colors,
            name="日收益率",
        ))
        fig_ret.update_layout(height=300, title="未来日收益率预测",
                              xaxis_title="日期", yaxis_title="收益率(%)")
        st.plotly_chart(fig_ret, use_container_width=True)

        # 累计收益率走势图（含置信区间）
        fig_cum = go.Figure()

        # 预测累计收益率
        fig_cum.add_trace(go.Scatter(
            x=future_dates, y=preds["cumulative_return"] * 100,
            name="预测累计收益率", line=dict(color="red", width=2),
            mode="lines+markers"))

        # 置信区间（转为累计收益率，近似：(1+r_lower)累积-1）
        if "confidence_lower" in preds and len(preds["confidence_lower"]) > 0:
            conf_lower_cum = np.cumprod(1 + preds["confidence_lower"]) - 1
            conf_upper_cum = np.cumprod(1 + preds["confidence_upper"]) - 1
            fig_cum.add_trace(go.Scatter(
                x=list(future_dates) + list(future_dates[::-1]),
                y=list(conf_upper_cum * 100) + list(conf_lower_cum * 100)[::-1],
                fill="toself", fillcolor="rgba(255,0,0,0.1)",
                line=dict(color="rgba(255,0,0,0)"), name="置信区间"))

        fig_cum.update_layout(height=500, title="累计收益率预测走势",
                              xaxis_title="日期", yaxis_title="累计收益率(%)")
        st.plotly_chart(fig_cum, use_container_width=True)

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
                    x=future_dates, y=vals * 100,
                    name=name, mode="lines+markers"))
            if len(preds["predicted_return"]) > 0:
                fig_cmp.add_trace(go.Scatter(
                    x=future_dates, y=preds["predicted_return"] * 100,
                    name="集成", line=dict(width=3, dash="dash"), mode="lines"))
            fig_cmp.update_layout(height=400, title="各模型日收益率预测对比",
                                  xaxis_title="日期", yaxis_title="收益率(%)")
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
            fig_vs.add_trace(go.Scatter(x=test_dates, y=r.test_returns_actual * 100,
                                        name="实际值", line=dict(color="blue")))
            fig_vs.add_trace(go.Scatter(x=test_dates, y=r.test_returns * 100,
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
                            st.session_state.ensemble_weights = w
                            st.session_state.predictions = ensemble_predict(
                                new_results, w, forecast_days)
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
                        "dropout": cfg.dropout,
                        "lstm_lr": cfg.lstm_lr,
                        "gru_lr": cfg.gru_lr,
                        "cnn_lr": cfg.cnn_lr,
                        "cnn_gru_lr": cfg.cnn_gru_lr,
                        "patchtst_lr": cfg.patchtst_lr,
                        "tft_lr": cfg.tft_lr,
                        "xgboost_lr": cfg.xgboost_learning_rate,
                        "lightgbm_lr": cfg.lightgbm_learning_rate,
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
            if st.session_state.predictions and len(st.session_state.predictions.get("predicted_return", [])) > 0:
                preds = st.session_state.predictions
                if st.session_state.stock_data is not None:
                    last_date = st.session_state.stock_data.index[-1]
                else:
                    last_date = pd.Timestamp.now()
                future_dates = pd.bdate_range(start=last_date + timedelta(days=1),
                                              periods=len(preds["predicted_return"]))
                pred_df = pd.DataFrame({
                    "日期": future_dates,
                    "预测日收益率": preds["predicted_return"],
                    "日收益率": preds["daily_return"],
                    "累计收益率": preds["cumulative_return"],
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
        st.markdown(f"- Dropout: **{last_config.dropout}**")

        # 显示各DL模型独立学习率
        dl_lr_map = {
            "LSTM": last_config.lstm_lr, "GRU": last_config.gru_lr,
            "1D-CNN": last_config.cnn_lr, "CNN-GRU": last_config.cnn_gru_lr,
            "PatchTST": last_config.patchtst_lr, "TFT": last_config.tft_lr,
        }
        st.markdown("- DL学习率: " + " | ".join(
            f"**{name}**: {lr:.4f}" for name, lr in dl_lr_map.items()
        ))
        tree_lrs = []
        if hasattr(last_config, 'xgboost_learning_rate'):
            tree_lrs.append(f"XGBoost: {last_config.xgboost_learning_rate:.2f}")
        if hasattr(last_config, 'lightgbm_learning_rate'):
            tree_lrs.append(f"LightGBM: {last_config.lightgbm_learning_rate:.2f}")
        if tree_lrs:
            st.markdown("- 树模型学习率: " + " | ".join(tree_lrs))

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


# ── Tab 9: 涨跌预测 ──────────────────────────────────────────

with tab9:
    # 切换股票后自动清除旧结果
    if (st.session_state.clf_results is not None
            and st.session_state.clf_results_stock_code
            and st.session_state.clf_results_stock_code != st.session_state.stock_code):
        st.session_state.clf_results = None
        st.session_state.clf_ensemble_result = None
        st.session_state.clf_results_stock_code = None

    if st.session_state.clf_results is not None:
        results = st.session_state.clf_results
        ens = st.session_state.clf_ensemble_result

        # 阈值变化时实时重算融合信号（无需重新训练）
        if ens is not None and len(st.session_state.clf_selected_models) == 2:
            if (ens.get("threshold") != st.session_state.clf_threshold
                    or ens.get("forecast_days") != st.session_state.clf_forecast_days):
                ens["threshold"] = st.session_state.clf_threshold
                ens["forecast_days"] = st.session_state.clf_forecast_days
                # 用保存的动态权重重算融合概率
                weights = ens.get("weights", {})
                models = st.session_state.clf_selected_models
                if weights and models[0] in weights and models[1] in weights:
                    proba_a = results[models[0]].oos_probabilities
                    proba_b = results[models[1]].oos_probabilities
                    n_common = min(len(proba_a), len(proba_b))
                    ens["fused_proba"] = (weights[models[0]] * proba_a[:n_common]
                                          + weights[models[1]] * proba_b[:n_common])
                ens["fused_signal"] = (ens["fused_proba"] >= st.session_state.clf_threshold).astype(int)
                ens["metrics"] = calculate_classification_metrics(
                    ens["oos_actuals"], ens["fused_proba"],
                    ens["oos_returns"], ens["fused_signal"],
                    future_ret=ens["oos_future_ret"],
                    forecast_days=st.session_state.clf_forecast_days,
                    next_day_ret=ens.get("oos_next_day_ret"),
                )

        st.subheader("涨跌预测结果")

        # 显示本次训练使用的特征数
        _first_model = list(results.keys())[0] if results else None
        if _first_model and results[_first_model].feature_names:
            _n_used = len(results[_first_model].feature_names)
            _has_tr = any("turnover" in f for f in results[_first_model].feature_names)
            _has_excess = any("excess_ret" in f for f in results[_first_model].feature_names)
            _tag_parts = [f"特征维度: {_n_used}"]
            if _has_tr:
                _tag_parts.append("换手率 ✅")
            if _has_excess:
                _tag_parts.append("相对强弱 ✅")
            st.caption(" | ".join(_tag_parts))
        # ── 融合集成指标卡 ──
        show_ensemble = ens is not None and len(st.session_state.clf_selected_models) == 2
        if show_ensemble:
            m = ens["metrics"]
            weights = ens.get("weights", {})
            models = st.session_state.clf_selected_models
            if weights and models[0] in weights and models[1] in weights:
                w_str = f"{models[0]} {weights[models[0]]*100:.0f}% + {models[1]} {weights[models[1]]*100:.0f}%"
            else:
                w_str = "XGBoost + ElasticNet 等权"
            st.markdown(f"### 融合集成 ({w_str})")
            mc1, mc2, mc3, mc4, mc5, mc6, mc7, mc8 = st.columns(8)
            mc1.metric("AUC", f"{m.get('auc', 0):.3f}")
            ic_val = m.get('ic', np.nan)
            mc2.metric("IC", f"{ic_val:.3f}" if not np.isnan(ic_val) else "N/A")
            mc3.metric("年化收益", f"{m.get('ann_return', 0)*100:+.2f}%")
            mc4.metric("年化波动", f"{m.get('ann_volatility', 0)*100:.2f}%")
            mc5.metric("Sharpe", f"{m.get('sharpe', 0):.2f}")
            mc6.metric("最大回撤", f"{m.get('max_dd', 0)*100:.2f}%")
            mc7.metric("胜率", f"{m.get('win_rate', 0)*100:.1f}%")
            mc8.metric("盈亏比", f"{m.get('profit_loss_ratio', 0):.2f}")
            st.caption(
                f"阈值: {ens.get('threshold', 0.5):.2f} | "
                f"持有天数: {ens.get('forecast_days', 1)} | "
                f"准确率: {m.get('accuracy', 0)*100:.1f}% | "
                f"精确率: {m.get('precision', 0)*100:.1f}% | "
                f"召回率: {m.get('recall', 0)*100:.1f}%"
            )

        # ── 每模型指标卡 ──
        for model_name in st.session_state.clf_selected_models:
            if model_name not in results:
                continue
            r = results[model_name]
            m = r.overall_metrics

            st.markdown(f"### {model_name}")
            mc1, mc2, mc3, mc4, mc5, mc6, mc7, mc8 = st.columns(8)
            mc1.metric("AUC", f"{m.get('auc', 0):.3f}")
            ic_val = m.get('ic', np.nan)
            mc2.metric("IC", f"{ic_val:.3f}" if not np.isnan(ic_val) else "N/A")
            mc3.metric("年化收益", f"{m.get('ann_return', 0)*100:+.2f}%")
            mc4.metric("年化波动", f"{m.get('ann_volatility', 0)*100:.2f}%")
            mc5.metric("Sharpe", f"{m.get('sharpe', 0):.2f}")
            mc6.metric("最大回撤", f"{m.get('max_dd', 0)*100:.2f}%")
            mc7.metric("胜率", f"{m.get('win_rate', 0)*100:.1f}%")
            mc8.metric("盈亏比", f"{m.get('profit_loss_ratio', 0):.2f}")

            with st.expander("混淆矩阵 & 分类指标", expanded=False):
                cm = m.get("confusion", {})
                cm_df = pd.DataFrame({
                    "预测涨": [cm.get('tp', 0), cm.get('fp', 0)],
                    "预测跌": [cm.get('fn', 0), cm.get('tn', 0)],
                }, index=["实际涨", "实际跌"])
                st.dataframe(cm_df, use_container_width=True)
                st.caption(
                    f"准确率: {m.get('accuracy', 0)*100:.1f}% | "
                    f"精确率: {m.get('precision', 0)*100:.1f}% | "
                    f"召回率: {m.get('recall', 0)*100:.1f}%")

            if r.fold_metrics.get("cv_avg"):
                cv = r.fold_metrics["cv_avg"]
                with st.expander("交叉验证平均指标", expanded=False):
                    cv_rows = [{
                        "指标": k,
                        "CV均值": f"{v:.4f}" if not isinstance(v, dict) else str(v)
                    } for k, v in cv.items() if k != "confusion"]
                    if cv_rows:
                        st.dataframe(pd.DataFrame(cv_rows), use_container_width=True, hide_index=True)

        # ── 累计收益曲线（使用 future_ret 作回测P&L） ──
        st.subheader("样本外策略日收益率")
        st.caption("策略: T日收盘预测下一日涨则买入，T+1日收盘卖出。信号跌则空仓（未扣除交易费用）")

        fig_cum = go.Figure()
        first_model = st.session_state.clf_selected_models[0]
        if first_model in results:
            base_dates = results[first_model].oos_dates
            base_returns = results[first_model].oos_returns

            # 买入持有基准（每日市场收益）
            fig_cum.add_trace(go.Scatter(
                x=base_dates, y=base_returns * 100,
                name="市场日收益", line=dict(color="gray", dash="dash", width=1.5),
                mode="lines"))

            colors = {"XGBoost": "#1f77b4", "ElasticNet": "#ff7f0e", "Ensemble": "#2ca02c"}
            for model_name in st.session_state.clf_selected_models:
                if model_name not in results:
                    continue
                r = results[model_name]
                # 策略P&L: 下一日收益 × 信号（T日买→T+1日卖）
                pnl = r.oos_next_day_ret if hasattr(r, 'oos_next_day_ret') and r.oos_next_day_ret is not None and len(r.oos_next_day_ret) > 0 else r.oos_returns
                strategy_ret = pnl * r.oos_predictions
                # 过滤 NaN
                valid = ~np.isnan(strategy_ret)
                plot_dates = r.oos_dates[valid] if valid.sum() > 0 else r.oos_dates
                plot_ret = strategy_ret[valid] * 100 if valid.sum() > 0 else strategy_ret * 100

                fig_cum.add_trace(go.Scatter(
                    x=plot_dates, y=plot_ret,
                    name=f"{model_name}", mode="lines",
                    line=dict(color=colors.get(model_name, "#333"), width=2)))

            # 融合集成曲线
            if show_ensemble:
                pnl_ens = ens.get("oos_next_day_ret") if ens.get("oos_next_day_ret") is not None else ens["oos_returns"]
                strategy_ret_ens = pnl_ens * ens["fused_signal"]
                valid_ens = ~np.isnan(strategy_ret_ens)
                plot_dates_ens = ens["oos_dates"][valid_ens] if valid_ens.sum() > 0 else ens["oos_dates"]
                plot_ret_ens = strategy_ret_ens[valid_ens] * 100 if valid_ens.sum() > 0 else strategy_ret_ens * 100
                fig_cum.add_trace(go.Scatter(
                    x=plot_dates_ens, y=plot_ret_ens,
                    name="融合集成", mode="lines",
                    line=dict(color=colors["Ensemble"], width=3)))

        fig_cum.update_layout(
            height=450,
            title="样本外策略日收益率（T日收盘买入，T+1日收盘卖出，未扣费）",
            xaxis_title="日期", yaxis_title="日收益率(%)",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            yaxis=dict(ticksuffix="%"),
        )
        st.plotly_chart(fig_cum, use_container_width=True)

        # ── 特征重要性 ──
        if "XGBoost" in results and results["XGBoost"].feature_importance:
            st.subheader("特征重要性 (XGBoost)")
            imp = results["XGBoost"].feature_importance
            sorted_imp = sorted(imp.items(), key=lambda x: abs(x[1]), reverse=True)[:15]
            show_names = [k[:40] for k, _ in sorted_imp]
            show_vals = [v for _, v in sorted_imp]

            fig_imp = go.Figure(go.Bar(
                x=show_vals, y=show_names, orientation="h",
                marker_color="#1f77b4",
            ))
            fig_imp.update_layout(
                height=400, title="Top 15 特征重要性",
                xaxis_title="重要性", yaxis_title="特征")
            st.plotly_chart(fig_imp, use_container_width=True)

        # ── 近期预测明细表 ──
        st.subheader("近期预测明细")
        first_model = st.session_state.clf_selected_models[0]
        if first_model in results:
            base_dates_f = results[first_model].oos_dates
            base_future_ret_f = results[first_model].oos_future_ret

        # ── 今日实时预测（基于最新交易日特征） ──
        _has_latest = False
        for _mn in st.session_state.clf_selected_models:
            if _mn in results and not np.isnan(getattr(results[_mn], 'latest_proba', np.nan)):
                _has_latest = True
                break
        if _has_latest:
            # 融合概率（使用动态权重）
            _latest_probas = []
            _latest_weights = []
            _ens_weights = ens.get("weights", {}) if ens else {}
            for _mn in st.session_state.clf_selected_models:
                if _mn in results and not np.isnan(getattr(results[_mn], 'latest_proba', np.nan)):
                    _latest_probas.append(results[_mn].latest_proba)
                    _latest_weights.append(_ens_weights.get(_mn, 1.0 / len(st.session_state.clf_selected_models)))
            if len(_latest_probas) > 1:
                _w_sum = sum(_latest_weights)
                _fused_latest = float(sum(p * w for p, w in zip(_latest_probas, _latest_weights)) / _w_sum)
            else:
                _fused_latest = _latest_probas[0]
            _latest_dir = "涨" if _fused_latest >= st.session_state.clf_threshold else "跌"
            _latest_dt = None
            for _mn in st.session_state.clf_selected_models:
                if _mn in results and results[_mn].latest_date is not None:
                    _latest_dt = results[_mn].latest_date
                    break
            _latest_dt_str = "?"
            _predict_dt_str = "?"
            if _latest_dt is not None:
                _ts = pd.Timestamp(_latest_dt)
                _latest_dt_str = _ts.strftime("%Y-%m-%d")
                _next = _ts + pd.Timedelta(days=1)
                while _next.weekday() >= 5:
                    _next += pd.Timedelta(days=1)
                _predict_dt_str = _next.strftime("%Y-%m-%d")
            st.success(f"**基于 {_latest_dt_str} 收盘数据，预测 {_predict_dt_str} 上涨概率: {_fused_latest*100:.1f}%** (→ {_latest_dir})，实际结果待验证")

        # 展示 OOS 最新预测（下一交易日上涨概率）
        if first_model in results and len(base_dates_f) > 0:
            latest_date = pd.Timestamp(base_dates_f[-1]).strftime("%Y-%m-%d") if len(base_dates_f) > 0 else "?"
            if show_ensemble and len(ens["fused_proba"]) > 0:
                latest_proba = ens["fused_proba"][-1]
                latest_dir = "涨" if latest_proba >= st.session_state.clf_threshold else "跌"
                latest_future = base_future_ret_f[-1]
                if bool(np.isnan(latest_future)):
                    st.info(f"**{latest_date} 下一交易日上涨概率: {latest_proba*100:.1f}%** (→ {latest_dir})，实际结果待验证")
                else:
                    actual = "✓ 正确" if (latest_proba >= st.session_state.clf_threshold) == (latest_future > 0) else "✗ 错误"
                    st.info(f"**{latest_date} 下一交易日上涨概率: {latest_proba*100:.1f}%** (→ {latest_dir})，实际: {actual}")
            elif first_model in results and len(results[first_model].oos_probabilities) > 0:
                latest_proba = results[first_model].oos_probabilities[-1]
                latest_dir = "涨" if latest_proba >= st.session_state.clf_threshold else "跌"
                st.info(f"**{latest_date} 下一交易日上涨概率 ({first_model}): {latest_proba*100:.1f}%** (→ {latest_dir})")
        n_recent = st.number_input("显示最近 N 天", min_value=10, max_value=60, value=30, step=5, key="clf_recent_n")

        if first_model in results:
            base_next_day_f = getattr(results[first_model], 'oos_next_day_ret', results[first_model].oos_returns)
            total = len(base_dates_f)
            start_idx = max(0, total - n_recent)

            date_start = pd.Timestamp(base_dates_f[0]).strftime("%Y-%m-%d") if len(base_dates_f) > 0 else "?"
            date_end = pd.Timestamp(base_dates_f[-1]).strftime("%Y-%m-%d") if len(base_dates_f) > 0 else "?"
            st.caption(f"OOS样本区间: {date_start} ~ {date_end}（共 {total} 天）。若日期偏旧，请在侧边栏重新加载股票数据以获取最新行情。")

            table_data = []
            for i in range(start_idx, total):
                d = base_dates_f[i]
                actual_future = base_future_ret_f[i]
                future_is_nan = bool(np.isnan(actual_future))
                actual_dir = "待验证" if future_is_nan else ("涨" if actual_future > 0 else "跌")

                row = {
                    "日期": pd.Timestamp(d).strftime("%Y-%m-%d") if hasattr(pd.Timestamp(d), 'strftime') else str(d)[:10],
                    "实际": actual_dir,
                    f"持有{st.session_state.clf_forecast_days}日收益": "N/A" if future_is_nan else f"{actual_future*100:+.2f}%",
                }

                # 融合信号 + 策略收益（用下一日收益计算）
                if show_ensemble:
                    fused_signal = ens["fused_signal"][i]
                    row["策略信号"] = "多" if fused_signal == 1 else "空"
                    ndr = base_next_day_f[i]
                    ndr_nan = bool(np.isnan(ndr))
                    if ndr_nan:
                        row["策略收益"] = "N/A"
                    elif fused_signal == 1:
                        row["策略收益"] = f"{ndr*100:+.2f}%"
                    else:
                        row["策略收益"] = "0%"
                else:
                    sig = results[first_model].oos_predictions[i]
                    row["策略信号"] = "多" if sig == 1 else "空"
                    ndr = base_next_day_f[i]
                    ndr_nan = bool(np.isnan(ndr))
                    if ndr_nan:
                        row["策略收益"] = "N/A"
                    elif sig == 1:
                        row["策略收益"] = f"{ndr*100:+.2f}%"
                    else:
                        row["策略收益"] = "0%"

                # 融合概率列（模型预测的是下一日上涨概率）
                if show_ensemble:
                    prob_fused = ens["fused_proba"][i]
                    fused_dir = "涨" if prob_fused >= st.session_state.clf_threshold else "跌"
                    if future_is_nan:
                        row["下一日上涨概率"] = f"{prob_fused:.3f} (→ {fused_dir})"
                    else:
                        correct = "✓" if (prob_fused >= st.session_state.clf_threshold) == (actual_future > 0) else "✗"
                        row["下一日上涨概率"] = f"{prob_fused:.3f} (→ {fused_dir} {correct})"

                for model_name in st.session_state.clf_selected_models:
                    if model_name not in results:
                        continue
                    r = results[model_name]
                    prob = r.oos_probabilities[i]
                    pred_dir = "涨" if prob >= st.session_state.clf_threshold else "跌"
                    if future_is_nan:
                        row[f"{model_name}"] = f"{prob:.3f} (→ {pred_dir})"
                    else:
                        correct = "✓" if (prob >= st.session_state.clf_threshold) == (actual_future > 0) else "✗"
                        row[f"{model_name}"] = f"{prob:.3f} (→ {pred_dir} {correct})"

                table_data.append(row)

            st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

        # ── 导出 ──
        st.subheader("导出预测结果")
        if first_model in results:
            base_dates_e = results[first_model].oos_dates
            base_future_ret_e = results[first_model].oos_future_ret
            base_next_day_e = getattr(results[first_model], 'oos_next_day_ret', results[first_model].oos_returns)
            n_total = len(base_dates_e)

            export_rows = []
            for i in range(n_total):
                fut_ret = base_future_ret_e[i]
                fut_is_nan = bool(np.isnan(fut_ret))
                ndr_e = base_next_day_e[i]
                ndr_is_nan = bool(np.isnan(ndr_e))
                row = {
                    "日期": pd.Timestamp(base_dates_e[i]).strftime("%Y-%m-%d") if hasattr(pd.Timestamp(base_dates_e[i]), 'strftime') else str(base_dates_e[i])[:10],
                    f"持有{st.session_state.clf_forecast_days}日收益": None if fut_is_nan else round(float(fut_ret), 6),
                    "实际涨跌": "待验证" if fut_is_nan else ("涨" if fut_ret > 0 else "跌"),
                }
                if show_ensemble and i < len(ens["fused_proba"]):
                    row["下一日上涨概率"] = round(float(ens["fused_proba"][i]), 4)
                    row["融合信号"] = "多" if ens["fused_signal"][i] == 1 else "空"
                    if ndr_is_nan:
                        row["策略收益"] = None
                    elif ens["fused_signal"][i] == 1:
                        row["策略收益"] = round(float(ndr_e), 6)
                    else:
                        row["策略收益"] = 0.0
                elif first_model in results and i < len(results[first_model].oos_predictions):
                    sig = results[first_model].oos_predictions[i]
                    if ndr_is_nan:
                        row["策略收益"] = None
                    elif sig == 1:
                        row["策略收益"] = round(float(ndr_e), 6)
                    else:
                        row["策略收益"] = 0.0
                for model_name in st.session_state.clf_selected_models:
                    if model_name not in results:
                        continue
                    r = results[model_name]
                    if i < len(r.oos_probabilities):
                        row[f"{model_name}_概率"] = round(float(r.oos_probabilities[i]), 4)
                        row[f"{model_name}_信号"] = "涨" if r.oos_predictions[i] == 1 else "跌"
                export_rows.append(row)

            export_df = pd.DataFrame(export_rows)
            csv = export_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "下载预测结果 CSV",
                data=csv,
                file_name=f"clf_predictions_{st.session_state.stock_code}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True,
            )

    else:
        st.info("请在侧边栏「涨跌预测设置」中点击「开始涨跌训练」按钮")
        if st.session_state.stock_data is not None:
            _sd = st.session_state.stock_data
            n = len(_sd)
            n_features = len([c for c in CLF_FEATURE_COLS if c in _sd.columns])
            rec = get_recommended_params(n, n_features)
            st.markdown(f"**当前数据量**: {n} 天, **可用基础特征**: {n_features} 个 (× look_back 展开)")
            st.info(f"推荐训练模式: **{rec['mode']}** (回溯: {rec['look_back']}天, 折数: {rec['n_splits']})")

            # 数据质量提示
            _has_turnover = "turnover" in _sd.columns and _sd["turnover"].notna().mean() > 0.5
            _has_vol = "volume" in _sd.columns and (_sd["volume"] > 0).mean() > 0.5
            _quality_items = []
            _quality_items.append(f"{'✅' if _has_turnover else '❌'} 换手率")
            _quality_items.append(f"{'✅' if _has_vol else '❌'} 成交量")
            _quality_items.append(f"{'✅' if 'pct_change' in _sd.columns else '❌'} 涨跌幅")
            st.caption("数据字段: " + " | ".join(_quality_items)
                       + (" (换手率缺失时自动跳过相关特征)" if not _has_turnover else ""))
            if not _has_turnover and st.session_state.stock_code:
                if st.button("重新获取数据（修复换手率）", key="clf_refetch_turnover"):
                    with st.spinner("重新获取中..."):
                        try:
                            from src.predict.stock_data_store import delete_stock_data, fetch_and_store
                            delete_stock_data(st.session_state.stock_code)
                            df_new, _, _ = fetch_and_store(
                                st.session_state.stock_code,
                                start_date=_sd.index[0].strftime("%Y%m%d"),
                                end_date=_sd.index[-1].strftime("%Y%m%d"),
                                max_days=10000)
                            st.session_state.stock_data = df_new
                            st.success("数据已刷新")
                            st.rerun()
                        except Exception as e:
                            st.error(f"重新获取失败: {e}")

    # ── 预测历史 ──
    st.markdown("---")
    st.subheader("预测历史")

    try:
        from src.predict.clf_history_store import list_clf_sessions, load_clf_session, delete_clf_session
        hist_sessions = list_clf_sessions(st.session_state.stock_code)
    except Exception as _hist_err:
        hist_sessions = []
        st.caption(f"历史记录加载失败: {_hist_err}")

    if not hist_sessions:
        st.caption("暂无历史预测记录，训练涨跌分类器后会自动保存")
    else:
        for s in hist_sessions:
            _sid = s["session_id"]
            _auc_str = ""
            _ens_m = s.get("ensemble_metrics")
            if _ens_m and _ens_m.get("auc") is not None:
                _auc_str = f" | AUC={_ens_m['auc']:.3f}"

            # 下一日预测标签
            _pred_str = ""
            if s.get("latest_proba") is not None:
                _signal_emoji = "🔴" if s.get("latest_signal") == 1 else "🟢"
                _pred_str = f" | {_signal_emoji} 预测{'涨' if s.get('latest_signal') == 1 else '跌'} ({s['latest_proba']:.1%})"

            _title = (
                f"#{_sid} | {s.get('trained_at', '?')} | "
                f"持有{s.get('forecast_days', '?')}天 | 阈值{s.get('threshold', '?')}"
                f"{_auc_str}{_pred_str}"
            )
            with st.expander(_title, expanded=False):
                hist = load_clf_session(_sid)
                if not hist:
                    st.warning("加载失败")
                    continue

                # 下一日预测醒目展示
                if hist.get("latest_proba") is not None:
                    _ldate = hist.get("latest_date", "?")
                    _lproba = hist["latest_proba"]
                    _lsignal = hist.get("latest_signal", 0)
                    if _lsignal == 1:
                        st.success(f"📈 预测下一交易日: **偏涨** | 概率 {_lproba:.1%} | 基于 {_ldate} 数据")
                    else:
                        st.error(f"📉 预测下一交易日: **偏跌** | 概率 {_lproba:.1%} | 基于 {_ldate} 数据")

                st.caption(
                    f"模型: {hist.get('selected_models')} | "
                    f"回溯: {hist.get('look_back')}天 | 折数: {hist.get('n_splits')} | "
                    f"数据: {hist.get('data_start_date')} ~ {hist.get('data_end_date')} | "
                    f"OOS: {hist.get('oos_start_date')} ~ {hist.get('oos_end_date')}"
                )

                params_hist = hist.get("params", {})
                if params_hist:
                    with st.expander("训练参数", expanded=False):
                        st.json(params_hist)

                ens_m = hist.get("ensemble_metrics")
                if ens_m:
                    # 权重信息
                    _w = ens_m.get("weights")
                    if _w:
                        _w_parts = [f"{k} {v*100:.0f}%" for k, v in _w.items()]
                        st.caption(f"融合权重: {' + '.join(_w_parts)}")

                    hc1, hc2, hc3, hc4, hc5, hc6, hc7, hc8 = st.columns(8)
                    hc1.metric("AUC", f"{ens_m.get('auc', 0):.3f}" if ens_m.get('auc') is not None else "N/A")
                    ic_v = ens_m.get('ic')
                    hc2.metric("IC", f"{ic_v:.3f}" if ic_v is not None else "N/A")
                    hc3.metric("年化收益", f"{ens_m.get('ann_return', 0)*100:+.2f}%")
                    hc4.metric("年化波动", f"{ens_m.get('ann_volatility', 0)*100:.2f}%")
                    hc5.metric("Sharpe", f"{ens_m.get('sharpe', 0):.2f}")
                    hc6.metric("最大回撤", f"{ens_m.get('max_dd', 0)*100:.2f}%")
                    hc7.metric("胜率", f"{ens_m.get('win_rate', 0)*100:.1f}%")
                    hc8.metric("盈亏比", f"{ens_m.get('profit_loss_ratio', 0):.2f}")

                details = hist.get("details", [])
                if details:
                    hist_dates = [d["trade_date"] for d in details]
                    hist_signals = [d.get("fused_signal") or 0 for d in details]
                    hist_future = [d.get("future_ret") or 0 for d in details]
                    hist_future_valid = [d.get("future_ret_valid", 1) for d in details]

                    hist_strategy = []
                    for j in range(len(hist_dates)):
                        if hist_future_valid[j] and hist_signals[j] == 1:
                            hist_strategy.append(hist_future[j])
                        else:
                            hist_strategy.append(0.0)
                    hist_cum = np.cumprod(1 + np.array(hist_strategy)) - 1

                    fig_hist = go.Figure()
                    fig_hist.add_trace(go.Scatter(
                        x=hist_dates, y=hist_cum * 100,
                        name="融合集成", mode="lines",
                        line=dict(color="#2ca02c", width=2)))
                    fig_hist.update_layout(
                        height=300,
                        yaxis_title="累计收益率(%)",
                        yaxis=dict(ticksuffix="%"),
                        hovermode="x unified",
                    )
                    st.plotly_chart(fig_hist, use_container_width=True, key=f"clf_hist_chart_{_sid}")

                    st.markdown("**预测明细（最近 30 天）**")
                    hist_table = []
                    for d in details[-30:]:
                        fr = d.get("future_ret")
                        fr_valid = d.get("future_ret_valid", 1)
                        sig = d.get("fused_signal") or 0
                        is_nan = not fr_valid or fr is None
                        strat_ret = 0.0
                        if not is_nan and sig == 1:
                            strat_ret = fr * 100
                        hist_table.append({
                            "日期": d["trade_date"],
                            "下一日上涨概率": f"{d.get('next_day_proba', 0)*100:.1f}%" if d.get('next_day_proba') is not None else "N/A",
                            "信号": "多" if sig == 1 else "空",
                            "持有收益": "N/A" if is_nan else f"{fr*100:+.2f}%",
                            "策略收益": "N/A" if is_nan else f"{strat_ret:+.2f}%",
                        })
                    st.dataframe(pd.DataFrame(hist_table), use_container_width=True, hide_index=True)

                if st.button("删除此记录", key=f"clf_hist_del_{_sid}", type="secondary"):
                    try:
                        delete_clf_session(_sid)
                        st.toast(f"已删除 Session #{_sid}")
                        st.rerun()
                    except Exception as del_err:
                        st.error(f"删除失败: {del_err}")


# ═══════ 导出按钮处理 ═══════

if btn_export and st.session_state.train_results:
    st.toast("请切换到'结果导出'标签页下载文件")
