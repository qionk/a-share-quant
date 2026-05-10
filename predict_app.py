"""
A股价格预测工具
==============
功能: 5种模型(LSTM/GRU/1D-CNN/ARIMA/EGARCH)预测A股个股未来收盘价
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
)
from src.predict.model_store import save_model, list_models, delete_model, load_model
from src.predict.continuous import (
    rolling_train, track_performance, should_retrain, get_model_status, cleanup_old_models,
)
from src.predict.supabase_store import (
    is_configured as supabase_configured,
    save_training_results as supabase_save,
    load_latest_results as supabase_load,
    list_available_stocks as supabase_list_stocks,
    restore_to_session_state as supabase_restore,
)
from src.data import load_config

# ═══════ 页面配置 ═══════

st.set_page_config(page_title="A股价格预测", page_icon="📈", layout="wide")
st.title("A股价格预测工具")
st.caption("支持 LSTM / GRU / 1D-CNN / PatchTST / TFT / ARIMA / EGARCH 多模型集成预测")

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

for key in ["stock_data", "stock_code", "stock_name", "train_results",
            "ensemble_weights", "predictions", "training_active"]:
    if key not in st.session_state:
        st.session_state[key] = None
if "training_active" not in st.session_state:
    st.session_state.training_active = False

# ═══════ 侧边栏 ═══════

with st.sidebar:
    st.header("配置参数")

    # 数据来源
    st.subheader("数据输入")
    data_source = st.radio("数据来源", ["API自动获取", "Excel上传"], horizontal=True)

    if data_source == "API自动获取":
        stock_code = st.text_input("股票代码（6位）", value="603601")
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("开始日期", value=datetime(2020, 1, 1))
        with col2:
            end_date = st.date_input("结束日期", value=datetime.now())

        if st.button("获取数据", type="primary", use_container_width=True):
            with st.spinner("正在获取数据..."):
                try:
                    df = load_from_akshare(
                        stock_code,
                        start_date.strftime("%Y%m%d"),
                        end_date.strftime("%Y%m%d"),
                    )
                    st.session_state.stock_data = df
                    st.session_state.stock_code = stock_code
                    st.session_state.stock_name = get_stock_name(stock_code)
                    st.session_state.train_results = None
                    st.session_state.predictions = None
                    st.success(f"获取成功: {len(df)} 条数据")
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

    # 恢复历史结果
    st.subheader("恢复历史结果")
    uploaded_result = st.file_uploader("上传之前导出的结果文件", type=["json"], key="result_upload")
    if uploaded_result and st.button("恢复结果", use_container_width=True):
        try:
            _deserialize_results(uploaded_result.read())
            st.success("恢复成功！")
            st.rerun()
        except Exception as e:
            st.error(f"恢复失败: {e}")

    st.divider()

    # 云端训练结果
    st.subheader("云端训练结果")
    if supabase_configured():
        try:
            cloud_stocks = supabase_list_stocks()
        except Exception:
            cloud_stocks = []

        if cloud_stocks:
            cloud_options = {
                f"{s['stock_name'] or s['stock_code']} ({s['stock_code']}) - {s['trained_at'][:10]}": s
                for s in cloud_stocks
            }
            selected_cloud = st.selectbox(
                "选择已有结果", options=list(cloud_options.keys()), key="cloud_select"
            )
            if selected_cloud:
                info = cloud_options[selected_cloud]
                models_str = ", ".join(info.get("selected_models", []))
                st.caption(f"模型: {models_str} | 预测{info.get('forecast_days', '?')}天")

            if st.button("加载云端结果", use_container_width=True):
                try:
                    info = cloud_options[selected_cloud]
                    result = supabase_load(info["stock_code"])
                    if result:
                        supabase_restore(result[0], result[1])
                        st.success("云端结果加载成功！")
                        st.rerun()
                    else:
                        st.warning("未找到该股票的云端结果")
                except Exception as e:
                    err_msg = str(e)
                    if "403" in err_msg or "blocked" in err_msg.lower() or "limit" in err_msg.lower():
                        st.error("Supabase 免费版请求超限，请稍后再试（通常几小时后自动恢复）")
                    else:
                        st.error(f"加载失败: {e}")
        else:
            st.caption("暂无云端训练结果")
    else:
        st.caption("未配置云端数据库（设置 SUPABASE_URL/KEY）")

    st.divider()

    # 模型选择
    st.subheader("模型选择")
    all_models = ["LSTM", "GRU", "1D-CNN", "PatchTST", "TFT", "ARIMA", "EGARCH"]
    selected_models = st.multiselect("选择模型", all_models, default=all_models)
    use_ensemble = st.toggle("集成预测", value=True)

    st.divider()

    # 训练参数
    st.subheader("训练参数")
    forecast_days = st.slider("预测天数", 1, 10, 5)
    look_back = st.slider("时间步长(天)", 10, 60, predict_cfg.get("default_look_back", 30))

    quick_mode = st.toggle("快速模式", value=False, help="减少训练轮次和模型参数，适合快速测试")

    with st.expander("高级参数"):
        if quick_mode:
            epochs = st.slider("训练轮次", 5, 30, 10)
        else:
            epochs = st.slider("训练轮次", 20, 200, predict_cfg.get("dl", {}).get("epochs", 100))
        batch_size = st.select_slider("批量大小", [8, 16, 32, 64], value=32)
        learning_rate = st.number_input("学习率", 0.0001, 0.01, 0.001, format="%.4f")
        dropout = st.slider("Dropout", 0.1, 0.5, 0.2)

    with st.expander("Transformer参数"):
        patchtst_cfg = predict_cfg.get("patchtst", {})
        tft_cfg = predict_cfg.get("tft", {})
        if quick_mode:
            patchtst_d_model = st.select_slider("PatchTST d_model", [16, 32, 64], value=16)
            patchtst_n_layers = st.slider("PatchTST编码器层数", 1, 2, 1)
            tft_hidden = st.select_slider("TFT隐藏层", [8, 16, 32], value=8)
            tft_n_heads = st.select_slider("TFT注意力头数", [1, 2, 4], value=2)
        else:
            patchtst_d_model = st.select_slider("PatchTST d_model", [64, 128, 256],
                                                 value=patchtst_cfg.get("d_model", 128))
            patchtst_n_layers = st.slider("PatchTST编码器层数", 1, 4,
                                           patchtst_cfg.get("n_encoder_layers", 2))
            tft_hidden = st.select_slider("TFT隐藏层", [32, 64, 128],
                                           value=tft_cfg.get("hidden_size", 64))
            tft_n_heads = st.select_slider("TFT注意力头数", [2, 4, 8],
                                            value=tft_cfg.get("n_heads", 4))

    st.divider()

    # 模型状态
    if st.session_state.stock_code:
        status = get_model_status(st.session_state.stock_code)
        st.info(f"模型状态: {status}")

    # 操作按钮
    st.subheader("操作")
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

    if quick_mode:
        units_lstm = qm.get("lstm_units", [32, 16])
        units_gru = qm.get("gru_units", [32, 16])
    else:
        units_lstm = dl_cfg.get("lstm_units", [64, 32])
        units_gru = dl_cfg.get("gru_units", [64, 32])

    # 小样本自动检测
    data = st.session_state.stock_data
    sm_cfg = predict_cfg.get("small_sample", {})
    if data is not None and len(data) < sm_cfg.get("threshold", 200):
        units_lstm = sm_cfg.get("lstm_units", [32, 16])
        units_gru = sm_cfg.get("gru_units", [32, 16])

    return ModelConfig(
        look_back=look_back,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        dropout=dropout,
        lstm_units=units_lstm,
        gru_units=units_gru,
        cnn_filters=dl_cfg.get("cnn_filters", [64, 32]),
        cnn_kernel_size=dl_cfg.get("cnn_kernel_size", 3),
        early_stop_patience=3 if quick_mode else dl_cfg.get("early_stop_patience", 10),
        patchtst_patch_size=pt_cfg.get("patch_size", 16),
        patchtst_d_model=qm.get("patchtst_d_model", patchtst_d_model) if quick_mode else patchtst_d_model,
        patchtst_n_heads=pt_cfg.get("n_heads", 4),
        patchtst_n_encoder_layers=qm.get("patchtst_n_encoder_layers", patchtst_n_layers) if quick_mode else patchtst_n_layers,
        patchtst_ff_dim=qm.get("patchtst_ff_dim", 256) if quick_mode else pt_cfg.get("ff_dim", 256),
        patchtst_dropout=pt_cfg.get("dropout", 0.1),
        tft_hidden_size=qm.get("tft_hidden_size", tft_hidden) if quick_mode else tft_hidden,
        tft_n_heads=tft_n_heads,
        tft_dropout=tf_cfg.get("dropout", 0.2),
        tft_lstm_layers=qm.get("tft_lstm_layers", 1) if quick_mode else tf_cfg.get("lstm_layers", 1),
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
        self._add_log(f"[{model_name}] 开始训练")

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
        self._add_log(f"[{model_name}] 完成 (耗时 {result.training_time:.1f}s)")

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
    dl_selected = [m for m in selected_models if m in ("LSTM", "GRU", "1D-CNN", "PatchTST", "TFT")]
    stat_selected = [m for m in selected_models if m in ("ARIMA", "EGARCH")]
    all_ordered = dl_selected + stat_selected

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
                sid = supabase_save(
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
    st.rerun()


# ═══════ 主界面标签页 ═══════

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["数据概览", "模型训练", "预测结果", "模型评估", "模型管理", "结果导出"]
)

# ── Tab 1: 数据概览 ──────────────────────────────────────────

with tab1:
    if st.session_state.stock_data is not None:
        df = st.session_state.stock_data
        name = st.session_state.stock_name or ""
        code = st.session_state.stock_code or ""

        # 信息卡片
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("股票", f"{name} ({code})")
        c2.metric("数据量", f"{len(df)} 天")
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

        fig_price.update_layout(height=450, title="收盘价预测走势（含置信区间）",
                                xaxis_title="日期", yaxis_title="价格(¥)")
        st.plotly_chart(fig_price, use_container_width=True)

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

        has_test_data = len(r.test_predictions) > 0 and len(r.test_actuals) > 0
        if not has_test_data:
            st.info("云端加载的结果不含测试集数据，仅展示模型指标")

        if has_test_data:
            fig_vs = go.Figure()
            x_range = list(range(len(r.test_actuals)))
            fig_vs.add_trace(go.Scatter(x=x_range, y=r.test_actuals,
                                        name="实际值", line=dict(color="blue")))
            fig_vs.add_trace(go.Scatter(x=x_range, y=r.test_predictions,
                                        name="预测值", line=dict(color="red", dash="dash")))
            if len(r.confidence_lower) > 0:
                fig_vs.add_trace(go.Scatter(
                    x=x_range + x_range[::-1],
                    y=list(r.confidence_upper) + list(r.confidence_lower[::-1]),
                    fill="toself", fillcolor="rgba(255,0,0,0.1)",
                    line=dict(color="rgba(0,0,0,0)"), name="95%置信"))
            fig_vs.update_layout(height=400, title=f"{model_sel} - 测试集预测效果",
                                 xaxis_title="样本", yaxis_title="价格")
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


# ═══════ 导出按钮处理 ═══════

if btn_export and st.session_state.train_results:
    st.toast("请切换到'结果导出'标签页下载文件")
