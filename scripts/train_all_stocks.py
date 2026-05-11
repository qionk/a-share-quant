#!/usr/bin/env python3
"""
本地批量训练脚本
训练数据库中所有股票的全部11个模型，结果保存到 MySQL。

用法:
  1. 先设置 MySQL 环境变量:
     set MYSQL_HOST=mysql3.sqlpub.com
     set MYSQL_PORT=3308
     set MYSQL_USER=root_quant
     set MYSQL_PASSWORD=BLnVlQ8qASfhA9xZ
     set MYSQL_DATABASE=a_share_quant

  2. 安装依赖:
     pip install -r requirements.txt

  3. 运行:
     python scripts/train_all_stocks.py
"""
import sys, os, time, logging
from datetime import datetime

# ── 强制 stdout 无缓冲，实时输出 ────────────────────────────
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, "reconfigure") else None

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# ── MySQL 配置（从环境变量读取）────────────────────────────
REQUIRED_ENV = ["MYSQL_HOST", "MYSQL_PORT", "MYSQL_USER", "MYSQL_PASSWORD", "MYSQL_DATABASE"]
# 环境变量默认值由用户在命令行设置，这里只做 fallback
os.environ.setdefault("MYSQL_HOST", os.environ.get("MYSQL_HOST", ""))
os.environ.setdefault("MYSQL_PORT", os.environ.get("MYSQL_PORT", "3306"))
os.environ.setdefault("MYSQL_USER", os.environ.get("MYSQL_USER", ""))
os.environ.setdefault("MYSQL_PASSWORD", os.environ.get("MYSQL_PASSWORD", ""))
os.environ.setdefault("MYSQL_DATABASE", os.environ.get("MYSQL_DATABASE", ""))

# ── 抑制 TensorFlow 日志 ──────────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import pandas as pd
import numpy as np

from src.predict.stock_data_store import fetch_and_store, list_stocks_with_status
from src.predict.training import (
    train_all_models, compute_ensemble_weights, ensemble_predict,
    TrainingCallbacks,
)
from src.predict.models import ModelConfig
from src.predict.mysql_store import save_training_results

# ── 每只股票最多保留多少交易日数据 ────────────────────────
MAX_DATA_DAYS = 500


# ╔══════════════════════════════════════════════════════════╗
# ║              环境检测 & 依赖检查                          ║
# ╚══════════════════════════════════════════════════════════╝

def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def print_section(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def test_environment():
    """启动前全面检测环境，打印诊断信息"""
    print_section("环境诊断")
    ok = True

    # 1. Python 版本
    print(f"\n[1/8] Python 版本")
    py_ver = sys.version_info
    print(f"  Python {py_ver.major}.{py_ver.minor}.{py_ver.micro}  ({sys.executable})")
    if py_ver < (3, 10):
        print(f"  [警告] Python 版本过低，建议 >= 3.10")
        ok = False

    # 2. 基础包检查 (不 import，只查版本)
    print(f"\n[2/8] 核心依赖检查")
    critical_pkgs = {
        "pandas": "pandas", "numpy": "numpy", "tensorflow": "tensorflow",
        "pymysql": "pymysql", "akshare": "akshare", "scikit-learn": "sklearn",
        "xgboost": "xgboost", "lightgbm": "lightgbm", "statsmodels": "statsmodels",
        "arch": "arch", "pmdarima": "pmdarima",
    }
    for display_name, import_name in critical_pkgs.items():
        try:
            m = __import__(import_name)
            ver = getattr(m, "__version__", "?")
            print(f"  [OK] {display_name}=={ver}")
        except ImportError:
            print(f"  [缺失] {display_name} -- 请执行: pip install {display_name}")
            ok = False

    # 3. TensorFlow 详情
    print(f"\n[3/8] TensorFlow 详情")
    try:
        import tensorflow as tf
        print(f"  TensorFlow {tf.__version__}")
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            for gpu in gpus:
                print(f"  GPU: {gpu.name}")
            # 配置 GPU 内存增长
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    print(f"  已启用 GPU 内存增长模式")
                except RuntimeError:
                    pass
            print(f"  [OK] GPU 可用，训练将使用 GPU 加速")
        else:
            print(f"  [INFO] 未检测到 GPU，将使用 CPU 训练 (较慢)")
    except ImportError:
        print(f"  [错误] TensorFlow 未安装")
        ok = False

    # 4. MySQL 连接测试
    print(f"\n[4/8] MySQL 连接测试")
    mysql_host = os.environ.get("MYSQL_HOST", "")
    mysql_port = os.environ.get("MYSQL_PORT", "3306")
    mysql_user = os.environ.get("MYSQL_USER", "")
    mysql_db = os.environ.get("MYSQL_DATABASE", "")
    if not mysql_host or not mysql_user or not mysql_db:
        print(f"  [错误] MySQL 环境变量未完整设置!")
        print(f"    MYSQL_HOST={mysql_host or '(空)'}")
        print(f"    MYSQL_PORT={mysql_port}")
        print(f"    MYSQL_USER={mysql_user or '(空)'}")
        print(f"    MYSQL_DATABASE={mysql_db or '(空)'}")
        print(f"  请在命令行执行:")
        print(f"    set MYSQL_HOST=mysql3.sqlpub.com")
        print(f"    set MYSQL_PORT=3308")
        print(f"    set MYSQL_USER=root_quant")
        print(f"    set MYSQL_PASSWORD=BLnVlQ8qASfhA9xZ")
        print(f"    set MYSQL_DATABASE=a_share_quant")
        ok = False
    else:
        try:
            import pymysql
            conn = pymysql.connect(
                host=mysql_host, port=int(mysql_port), user=mysql_user,
                password=os.environ.get("MYSQL_PASSWORD", ""), database=mysql_db,
                charset="utf8mb4", connect_timeout=10,
            )
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM stock_daily_data")
            stock_count = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM training_sessions")
            session_count = cur.fetchone()[0]
            cur.close(); conn.close()
            print(f"  [OK] MySQL 连接成功")
            print(f"    stock_daily_data 表: {stock_count} 条记录")
            print(f"    training_sessions 表: {session_count} 条记录")
        except Exception as e:
            print(f"  [错误] MySQL 连接失败: {e}")
            ok = False

    # 5. AKShare 连接测试
    print(f"\n[5/8] AKShare 数据源测试")
    try:
        import akshare as ak
        test_df = ak.stock_zh_a_hist(symbol="000001", period="daily", start_date="20260501", end_date="20260510", adjust="hfq")
        if test_df is not None and not test_df.empty:
            print(f"  [OK] AKShare 可用 (测试获取平安银行 000001 成功, {len(test_df)} 条)")
        else:
            print(f"  [WARN] AKShare 返回空数据，训练时可能需要从 MySQL 已有数据加载")
    except Exception as e:
        print(f"  [WARN] AKShare 不可用: {e}")
        print(f"    训练将继续，但新股票数据获取会失败")

    # 6. 工作目录
    print(f"\n[6/8] 工作目录")
    print(f"  项目根目录: {ROOT}")
    data_dir = os.path.join(ROOT, "data")
    os.makedirs(data_dir, exist_ok=True)
    print(f"  数据目录: {data_dir}")
    models_dir = os.path.join(ROOT, "models")
    os.makedirs(models_dir, exist_ok=True)
    print(f"  模型目录: {models_dir}")
    output_dir = os.path.join(ROOT, "output")
    os.makedirs(output_dir, exist_ok=True)
    print(f"  输出目录: {output_dir}")

    # 7. CPU 信息
    print(f"\n[7/8] CPU 信息")
    try:
        cpu_count = os.cpu_count()
        print(f"  CPU 核心数: {cpu_count}")
    except Exception:
        pass

    # 8. 内存信息 (Windows)
    print(f"\n[8/8] 内存信息")
    try:
        import subprocess
        result = subprocess.run(["wmic", "OS", "get", "TotalVisibleMemorySize", "/Value"],
                              capture_output=True, text=True)
        if result.returncode == 0:
            kb = int(result.stdout.strip().split("=")[-1])
            gb = kb / 1024 / 1024
            print(f"  系统内存: {gb:.1f} GB")
    except Exception:
        print(f"  无法获取内存信息")

    print(f"\n{'─'*70}")
    if ok:
        print(f"  环境检测通过! {now()}")
    else:
        print(f"  环境检测发现问题，请修复后再运行")
    print(f"{'─'*70}")
    return ok


# ╔══════════════════════════════════════════════════════════╗
# ║              训练回调 (增强日志)                          ║
# ╚══════════════════════════════════════════════════════════╝

class ConsoleCallbacks(TrainingCallbacks):
    """控制台输出训练进度（增强版）"""

    def __init__(self):
        self.model_start_time = None
        self.fold_start_time = None
        self.epoch_count = 0

    def on_training_start(self, model_list):
        self.epoch_count = 0
        print(f"\n{'─'*60}")
        print(f"[{now()}] 开始训练 {len(model_list)} 个模型")
        print(f"  {' '.join(model_list)}")
        print(f"{'─'*60}")

    def on_model_start(self, model_name, model_index, total_models):
        self.model_start_time = time.time()
        print(f"\n[{now()}] [{model_index+1}/{total_models}] {model_name} 启动训练")

    def on_fold_start(self, model_name, fold, total_folds):
        self.fold_start_time = time.time()

    def on_fold_end(self, model_name, fold, fold_metrics):
        elapsed = time.time() - self.fold_start_time if self.fold_start_time else 0
        mae = fold_metrics.get("mae", 0)
        rmse = fold_metrics.get("rmse", 0)
        print(f"  [{now()}] Fold {fold}/{3} 完成 ({elapsed:.1f}s) | MAE={mae:.4f}  RMSE={rmse:.4f}")

    def on_epoch_end(self, model_name, epoch, total_epochs, train_loss, val_loss, lr, grad_norm):
        self.epoch_count += 1
        if epoch % 5 == 0 or epoch == 1 or epoch == total_epochs:
            gn_str = f"  grad={grad_norm:.4f}" if grad_norm is not None else ""
            print(f"  [{now()}] Epoch {epoch:>3}/{total_epochs}  "
                  f"train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  lr={lr:.8f}{gn_str}")

    def on_early_stop(self, model_name, epoch, best_epoch):
        print(f"  [{now()}] early stop (epoch={epoch}, best={best_epoch})")

    def on_overfitting_warning(self, model_name, epoch, val_loss, best_val_loss):
        print(f"  [{now()}] [WARN] 过拟合迹象 epoch={epoch} (val_loss={val_loss:.6f} > best*1.1={best_val_loss*1.1:.6f})")

    def on_model_end(self, model_name, result):
        elapsed = time.time() - self.model_start_time if self.model_start_time else 0
        m = result.cv_metrics
        print(f"  [{now()}] {model_name} 完成 ({elapsed:.0f}s)", end="")
        if m and not np.isnan(m.get("mae", np.nan)):
            print(f" | MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}  MAPE={m['mape']:.2f}%  R²={m['r2']:.4f}")
        else:
            err = result.train_history.get("error", "未知错误")
            print(f" | 失败: {err}")

    def on_training_complete(self, all_results):
        valid = sum(1 for r in all_results.values()
                    if r.cv_metrics and not np.isnan(r.cv_metrics.get("mae", np.nan)))
        print(f"\n[{now()}] 训练完成: {valid}/{len(all_results)} 个模型成功\n")

    def on_log(self, message):
        print(f"  [{now()}] {message}")


# ╔══════════════════════════════════════════════════════════╗
# ║              单股票训练逻辑                               ║
# ╚══════════════════════════════════════════════════════════╝

def train_one_stock(code, name, config, forecast_days=5):
    """训练单只股票的所有模型"""
    print(f"\n{'#'*70}")
    print(f"# [{now()}] {name} ({code})")
    print(f"{'#'*70}")

    # ── 加载数据 ──────────────────────────────────────
    print(f"[{now()}] 加载数据 (最多保留 {MAX_DATA_DAYS} 天)...")
    t0 = time.time()
    df, stock_name_from_db, _ = fetch_and_store(code, max_days=MAX_DATA_DAYS)
    if df is None or df.empty:
        raise ValueError(f"无法获取 {code} 的数据")

    data_elapsed = time.time() - t0
    data_days = len(df)
    data_start = df.index[0].strftime('%Y-%m-%d')
    data_end = df.index[-1].strftime('%Y-%m-%d')
    last_close = df["close"].iloc[-1]

    print(f"[{now()}] 数据加载完成 ({data_elapsed:.1f}s)")
    print(f"  范围: {data_start} ~ {data_end}  ({data_days} 天)")
    print(f"  最新收盘价: {last_close:.2f}")
    print(f"  涨跌幅范围: {df['close'].pct_change().describe().to_dict() if len(df) > 1 else 'N/A'}")

    # 数据质量概要
    na_pct = df.isna().sum().sum() / (len(df) * len(df.columns)) * 100
    if na_pct > 0:
        print(f"  缺失值比例: {na_pct:.2f}%")
    if data_days < 60:
        print(f"  [WARN] 数据量较少 ({data_days}天)，模型可能欠拟合")
    elif data_days < 200:
        print(f"  [INFO] 数据量一般 ({data_days}天)，建议至少200天")

    # ── 模型列表 ──────────────────────────────────────
    all_models = [
        "LSTM", "GRU", "1D-CNN", "CNN-GRU", "PatchTST", "TFT",
        "XGBoost", "LightGBM",
        "ARIMA", "SARIMA", "EGARCH",
    ]

    callbacks = ConsoleCallbacks()

    # ── 训练 ──────────────────────────────────────────
    print(f"[{now()}] 开始训练...")
    train_t0 = time.time()

    results = train_all_models(
        df, all_models, config,
        forecast_days=forecast_days,
        callbacks=callbacks,
        quick=False,
    )

    train_elapsed = time.time() - train_t0
    print(f"[{now()}] 所有模型训练完成 ({train_elapsed/60:.1f} 分钟)")

    # ── 集成预测 ──────────────────────────────────────
    print(f"\n[{now()}] 计算集成权重和预测...")
    weights = compute_ensemble_weights(results)
    preds = ensemble_predict(results, weights, forecast_days, last_close)

    # 打印各模型权重排名
    print(f"\n  集成权重排名 (逆RMSE加权):")
    for i, (k, w) in enumerate(sorted(weights.items(), key=lambda x: -x[1])):
        bar = "█" * int(w * 50)
        print(f"  {i+1:>2}. {k:<12} {w:>6.1%}  {bar}")

    # 打印预测
    if len(preds["predicted_close"]) > 0:
        print(f"\n  未来 {forecast_days} 天预测:")
        for d, p, r in zip(range(1, forecast_days+1),
                           preds.get("predicted_close", []),
                           preds.get("daily_return", [])):
            conf_low = preds.get("confidence_lower", [])[d-1] if len(preds.get("confidence_lower", [])) >= d else p*0.95
            conf_high = preds.get("confidence_upper", [])[d-1] if len(preds.get("confidence_upper", [])) >= d else p*1.05
            print(f"    D+{d}: ¥{p:.2f}  [{conf_low:.2f} ~ {conf_high:.2f}]  ({r:+.2f}%)")
        cum_ret = preds.get("cumulative_return", np.array([]))
        total_ret = cum_ret[-1] if len(cum_ret) > 0 else 0
        print(f"    累计收益: {total_ret:+.2f}%")
    else:
        print(f"\n  [WARN] 无有效预测结果")

    # ── 保存到 MySQL ──────────────────────────────────
    print(f"\n[{now()}] 保存训练结果到 MySQL...")
    save_t0 = time.time()
    session_id = save_training_results(
        code, name, results, weights, preds,
        config, forecast_days, all_models,
        stock_data=df,
    )
    save_elapsed = time.time() - save_t0

    if session_id:
        print(f"[{now()}] 保存成功! ({save_elapsed:.1f}s) session_id={session_id}")
    else:
        print(f"[{now()}] 保存失败! 请检查 MySQL 配置")

    return results, preds


# ╔══════════════════════════════════════════════════════════╗
# ║              主入口                                       ║
# ╚══════════════════════════════════════════════════════════╝

def main():
    script_start = time.time()
    print(f"\n{'█'*70}")
    print(f"█  A-Share Quant 批量训练系统")
    print(f"█  时间: {now()}")
    print(f"█  项目: {ROOT}")
    print(f"{'█'*70}")

    # ── 步骤 1: 环境检测 ──────────────────────────────
    if not test_environment():
        print(f"\n[{now()}] 环境检测未通过，退出。请修复上述问题后重新运行。")
        sys.exit(1)

    # ── 步骤 2: 检查 MySQL 环境变量 ────────────────────
    missing_env = [k for k in REQUIRED_ENV if not os.environ.get(k)]
    if missing_env:
        print(f"\n[错误] 缺少 MySQL 环境变量: {', '.join(missing_env)}")
        print(f"请在运行前设置:")
        print(f"")
        print(f"  Windows CMD:")
        print(f"    set MYSQL_HOST=mysql3.sqlpub.com")
        print(f"    set MYSQL_PORT=3308")
        print(f"    set MYSQL_USER=root_quant")
        print(f"    set MYSQL_PASSWORD=BLnVlQ8qASfhA9xZ")
        print(f"    set MYSQL_DATABASE=a_share_quant")
        print(f"    python scripts\\train_all_stocks.py")
        print(f"")
        sys.exit(1)

    # ── 步骤 3: 配置 ──────────────────────────────────
    config = ModelConfig(
        look_back=30,
        epochs=100,
        batch_size=32,
        learning_rate=0.001,
        dropout=0.2,
        early_stop_patience=10,
    )
    forecast_days = 5

    print_section("训练配置")
    print(f"  look_back (序列长度): {config.look_back} 天")
    print(f"  epochs: {config.epochs}")
    print(f"  batch_size: {config.batch_size}")
    print(f"  learning_rate: {config.learning_rate}")
    print(f"  dropout: {config.dropout}")
    print(f"  预测天数: {forecast_days} 天")
    print(f"  数据保留: 最近 {MAX_DATA_DAYS} 个交易日")
    print(f"  交叉验证: 3折")
    print(f"  置信区间: MC Dropout 100次 / Bootstrap 100次")

    # ── 步骤 4: 获取股票列表 ──────────────────────────
    print_section("股票列表")
    stocks = list_stocks_with_status()
    if not stocks:
        print(f"\n[错误] 数据库中没有股票数据!")
        print(f"  请先在 Streamlit 网页中添加股票数据:")
        print(f"    streamlit run predict_app.py")
        print(f"  或手动插入股票数据到 stock_daily_data 表")
        sys.exit(1)

    print(f"  共 {len(stocks)} 只股票:")
    for i, s in enumerate(stocks):
        trained_mark = " [已训练]" if s.get("trained") else ""
        print(f"  {i+1:>2}. {s['code']}  {s['name']:<12}  {s['rows']}天  "
              f"到{s['end_date']}{trained_mark}")

    # ── 步骤 5: 批量训练 ──────────────────────────────
    print_section(f"开始批量训练 ({len(stocks)} 只股票)")
    total_start = time.time()
    all_saved = []
    all_failed = []

    for idx, stock in enumerate(stocks):
        code = stock["code"]
        name = stock["name"]
        print(f"\n{'*'*70}")
        print(f"* 股票 [{idx+1}/{len(stocks)}]")
        print(f"{'*'*70}")

        try:
            one_start = time.time()
            results, preds = train_one_stock(code, name, config, forecast_days)
            one_elapsed = time.time() - one_start
            all_saved.append({"code": code, "name": name, "time": one_elapsed})
            print(f"\n  [{now()}] {name} ({code}) 全部完成 ({one_elapsed/60:.1f} 分钟)")
        except KeyboardInterrupt:
            print(f"\n[{now()}] 用户中断训练")
            break
        except Exception as e:
            print(f"\n[{now()}] [失败] {name} ({code}): {e}")
            import traceback
            traceback.print_exc()
            all_failed.append({"code": code, "name": name, "error": str(e)})

    # ── 步骤 6: 汇总报告 ──────────────────────────────
    total_elapsed = time.time() - total_start
    script_elapsed = time.time() - script_start
    print(f"\n{'█'*70}")
    print(f"█  训练完成! {now()}")
    print(f"{'█'*70}")
    print(f"\n  结果汇总:")
    print(f"    总数: {len(stocks)} 只股票")
    print(f"    成功: {len(all_saved)} 只")
    print(f"    失败: {len(all_failed)} 只")
    print(f"    总耗时: {total_elapsed/60:.1f} 分钟 ({script_elapsed/3600:.1f} 小时)")
    print(f"")

    if all_saved:
        print(f"  成功列表:")
        for s in all_saved:
            print(f"    [OK] {s['code']} {s['name']}  ({s['time']/60:.1f} 分钟)")

    if all_failed:
        print(f"  失败列表:")
        for f in all_failed:
            print(f"    [FAIL] {f['code']} {f['name']}  reason: {f['error']}")

    print(f"\n{'█'*70}")
    print(f"█  Done.")
    print(f"{'█'*70}\n")


if __name__ == "__main__":
    main()