#!/usr/bin/env python3
"""
本地批量训练脚本
训练数据库中所有股票的全部11个模型，结果保存到 MySQL。
用法: python scripts/train_all_stocks.py
"""
import sys, os, time

# 强制 stdout 无缓冲，实时输出
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, "reconfigure") else None

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# MySQL 配置 — 从环境变量读取，未设置则用默认值
os.environ.setdefault("MYSQL_HOST", os.environ.get("MYSQL_HOST", ""))
os.environ.setdefault("MYSQL_PORT", os.environ.get("MYSQL_PORT", "3306"))
os.environ.setdefault("MYSQL_USER", os.environ.get("MYSQL_USER", ""))
os.environ.setdefault("MYSQL_PASSWORD", os.environ.get("MYSQL_PASSWORD", ""))
os.environ.setdefault("MYSQL_DATABASE", os.environ.get("MYSQL_DATABASE", ""))

# 抑制 TF 日志
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


class ConsoleCallbacks(TrainingCallbacks):
    """控制台输出训练进度"""

    def __init__(self):
        self.model_start_time = None

    def on_training_start(self, model_list):
        print(f"\n{'='*60}")
        print(f"开始训练 {len(model_list)} 个模型: {', '.join(model_list)}")
        print(f"{'='*60}")

    def on_model_start(self, model_name, model_index, total_models):
        self.model_start_time = time.time()
        print(f"\n[{model_index+1}/{total_models}] {model_name} 开始训练...")

    def on_fold_start(self, model_name, fold, total_folds):
        pass

    def on_fold_end(self, model_name, fold, fold_metrics):
        mae = fold_metrics.get("mae", 0)
        rmse = fold_metrics.get("rmse", 0)
        print(f"  Fold {fold}: MAE={mae:.4f}  RMSE={rmse:.4f}")

    def on_epoch_end(self, model_name, epoch, total_epochs, train_loss, val_loss, lr, grad_norm):
        if epoch % 10 == 0 or epoch == 1 or epoch == total_epochs:
            print(f"  Epoch {epoch}/{total_epochs}  loss={train_loss:.6f}  val_loss={val_loss:.6f}  lr={lr:.6f}")

    def on_early_stop(self, model_name, epoch, best_epoch):
        print(f"  Early stop at epoch {epoch}, best={best_epoch}")

    def on_overfitting_warning(self, model_name, epoch, val_loss, best_val_loss):
        print(f"  [警告] 过拟合迹象 epoch={epoch}")

    def on_model_end(self, model_name, result):
        elapsed = time.time() - self.model_start_time if self.model_start_time else 0
        m = result.cv_metrics
        print(f"  [{model_name}] 完成  耗时: {elapsed:.1f}s")
        if m and not np.isnan(m.get("mae", np.nan)):
            print(f"  MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}  MAPE={m['mape']:.2f}%  R²={m['r2']:.4f}")
        else:
            err = result.train_history.get("error", "未知错误")
            print(f"  [失败] {err}")

    def on_training_complete(self, all_results):
        valid = sum(1 for r in all_results.values()
                    if r.cv_metrics and not np.isnan(r.cv_metrics.get("mae", np.nan)))
        print(f"\n{'='*60}")
        print(f"训练完成: {valid}/{len(all_results)} 个模型成功")
        print(f"{'='*60}")

    def on_log(self, message):
        print(f"  [INFO] {message}")


def train_one_stock(code, name, config, forecast_days=5):
    """训练单只股票的所有模型"""
    print(f"\n{'#'*60}")
    print(f"# {name} ({code})")
    print(f"{'#'*60}")

    # 加载数据（自动增量更新）
    print("加载数据...")
    df, _, _ = fetch_and_store(code)
    print(f"  数据: {len(df)} 天  ({df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')})")

    # 所有11个模型
    all_models = [
        "LSTM", "GRU", "1D-CNN", "CNN-GRU", "PatchTST", "TFT",
        "XGBoost", "LightGBM",
        "ARIMA", "SARIMA", "EGARCH",
    ]

    callbacks = ConsoleCallbacks()

    # 训练
    results = train_all_models(
        df, all_models, config,
        forecast_days=forecast_days,
        callbacks=callbacks,
        quick=False,
    )

    # 集成
    last_price = df["close"].iloc[-1]
    weights = compute_ensemble_weights(results)
    preds = ensemble_predict(results, weights, forecast_days, last_price)

    print(f"\n集成权重:")
    for k, w in sorted(weights.items(), key=lambda x: -x[1]):
        print(f"  {k}: {w:.2%}")

    print(f"\n未来 {forecast_days} 天预测:")
    for d, p, r in zip(range(1, forecast_days+1),
                       preds["predicted_close"],
                       preds["daily_return"]):
        print(f"  D+{d}: ¥{p:.2f}  ({r:+.2f}%)")

    total_return = preds["cumulative_return"][-1] if len(preds["cumulative_return"]) > 0 else 0
    print(f"  累计收益: {total_return:+.2f}%")

    # 保存到 MySQL
    print(f"\n保存到云端...")
    session_id = save_training_results(
        code, name, results, weights, preds,
        config, forecast_days, all_models,
        stock_data=df,
    )

    if session_id:
        print(f"  保存成功! session_id={session_id}")
    else:
        print(f"  保存失败!")

    return results, preds


def main():
    # 配置（正常模式）
    config = ModelConfig(
        look_back=30,
        epochs=100,
        batch_size=32,
        learning_rate=0.001,
        dropout=0.2,
        early_stop_patience=10,
    )

    forecast_days = 5

    # 获取所有股票
    print("获取股票列表...")
    stocks = list_stocks_with_status()
    if not stocks:
        print("数据库中没有股票数据，请先在网页中添加")
        return

    print(f"共 {len(stocks)} 只股票:")
    for s in stocks:
        print(f"  {s['code']} {s['name']}  {s['rows']}天  {s['end_date']}")

    total_start = time.time()
    all_saved = []

    for stock in stocks:
        try:
            results, preds = train_one_stock(
                stock["code"], stock["name"],
                config, forecast_days,
            )
            all_saved.append(stock["code"])
        except Exception as e:
            print(f"\n[错误] {stock['name']} ({stock['code']}) 训练失败: {e}")
            import traceback
            traceback.print_exc()

    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"全部完成! 成功: {len(all_saved)}/{len(stocks)} 只股票")
    print(f"总耗时: {total_elapsed/60:.1f} 分钟")
    print(f"成功: {', '.join(all_saved)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()