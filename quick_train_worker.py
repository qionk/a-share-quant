"""快速训练 worker — 独立进程，不受 Streamlit 干扰"""
import os, sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# 必须在 import akshare/pandas 之前先 import TF，否则线程池冲突导致死锁
import tensorflow  # noqa: F401

import pickle

if __name__ == "__main__":
    input_path = sys.argv[1]
    output_path = sys.argv[2]

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    with open(input_path, "rb") as f:
        args = pickle.load(f)

    from src.predict.training import train_all_models

    results = train_all_models(
        args["df"], args["selected_models"], args["config"],
        forecast_days=args["forecast_days"], quick=True)

    for r in results.values():
        r.model_object = None
        r.scaler = None

    with open(output_path, "wb") as f:
        pickle.dump(results, f)
