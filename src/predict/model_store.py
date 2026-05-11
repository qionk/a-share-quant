"""
价格预测 - 模型存储与版本管理
保存/加载/列表/删除 模型版本
"""

import os
import json
import pickle
from datetime import datetime
from dataclasses import dataclass

MODEL_DIR = "models"


@dataclass
class ModelInfo:
    """模型版本信息"""
    stock_code: str
    model_type: str
    train_date: str
    metrics: dict
    file_path: str
    file_size: int


def _get_model_dir(stock_code: str, base_dir: str = MODEL_DIR) -> str:
    """获取某只股票的模型目录"""
    path = os.path.join(base_dir, stock_code)
    os.makedirs(path, exist_ok=True)
    return path


def save_model(stock_code: str, model_name: str, train_result,
               base_dir: str = MODEL_DIR) -> str:
    """
    保存模型和元数据
    DL模型保存为 .keras, 统计模型保存为 .pkl
    返回保存路径
    """
    model_dir = _get_model_dir(stock_code, base_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"{model_name}_{timestamp}"

    if model_name in ("LSTM", "GRU", "1D-CNN", "CNN-GRU"):
        model_path = os.path.join(model_dir, f"{base_name}.keras")
        train_result.model_object.save(model_path)
    else:
        model_path = os.path.join(model_dir, f"{base_name}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(train_result.model_object, f)

    # 保存 scaler 和元数据
    meta = {
        "stock_code": stock_code,
        "model_type": model_name,
        "train_date": datetime.now().isoformat(),
        "metrics": train_result.cv_metrics,
        "feature_cols": train_result.feature_cols,
        "n_features": train_result.n_features,
        "training_time": train_result.training_time,
    }
    meta_path = os.path.join(model_dir, f"{base_name}_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, default=str)

    # 保存 scaler
    if train_result.scaler is not None:
        scaler_path = os.path.join(model_dir, f"{base_name}_scaler.pkl")
        with open(scaler_path, "wb") as f:
            pickle.dump(train_result.scaler, f)

    return model_path


def load_model(file_path: str) -> dict:
    """
    加载已保存模型
    返回: {"model": ..., "scaler": ..., "meta": ...}
    """
    base = file_path.rsplit(".", 1)[0]
    meta_path = base + "_meta.json"
    scaler_path = base + "_scaler.pkl"

    # 加载模型
    if file_path.endswith(".keras"):
        import tensorflow as tf
        model = tf.keras.models.load_model(file_path)
    else:
        with open(file_path, "rb") as f:
            model = pickle.load(f)

    # 加载元数据
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

    # 加载 scaler
    scaler = None
    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)

    return {"model": model, "scaler": scaler, "meta": meta}


def list_models(stock_code: str = None, base_dir: str = MODEL_DIR) -> list:
    """
    列出已保存模型（按时间倒序）
    stock_code=None 则列出所有股票的模型
    """
    if not os.path.exists(base_dir):
        return []

    results = []
    codes = [stock_code] if stock_code else os.listdir(base_dir)

    for code in codes:
        code_dir = os.path.join(base_dir, code)
        if not os.path.isdir(code_dir):
            continue
        for f in os.listdir(code_dir):
            if f.endswith("_meta.json"):
                meta_path = os.path.join(code_dir, f)
                try:
                    with open(meta_path, "r", encoding="utf-8") as fp:
                        meta = json.load(fp)
                except Exception:
                    continue

                base_name = f.replace("_meta.json", "")
                model_file = None
                for ext in [".keras", ".pkl"]:
                    candidate = os.path.join(code_dir, base_name + ext)
                    if os.path.exists(candidate):
                        model_file = candidate
                        break

                if model_file:
                    results.append(ModelInfo(
                        stock_code=meta.get("stock_code", code),
                        model_type=meta.get("model_type", "unknown"),
                        train_date=meta.get("train_date", ""),
                        metrics=meta.get("metrics", {}),
                        file_path=model_file,
                        file_size=os.path.getsize(model_file),
                    ))

    results.sort(key=lambda x: x.train_date, reverse=True)
    return results


def delete_model(file_path: str) -> bool:
    """删除模型文件及相关文件（meta、scaler）"""
    base = file_path.rsplit(".", 1)[0]
    deleted = False

    for path in [file_path, base + "_meta.json", base + "_scaler.pkl"]:
        if os.path.exists(path):
            os.remove(path)
            deleted = True

    return deleted


def get_latest_model(stock_code: str, model_name: str,
                     base_dir: str = MODEL_DIR) -> str:
    """获取某只股票某类模型的最新版本路径"""
    models = list_models(stock_code, base_dir)
    for m in models:
        if m.model_type == model_name:
            return m.file_path
    return None
