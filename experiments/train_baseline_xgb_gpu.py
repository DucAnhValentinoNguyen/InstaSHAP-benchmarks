import pandas as pd
from src.data import load_adult
from src.models import train_xgboost_gpu
from joblib import dump
import os

def main():
    X_train, X_val, X_test, y_train, y_val, y_test, feats = load_adult()

    model, auc = train_xgboost_gpu(X_train, y_train, X_val, y_val)
    print("XGBoost GPU Validation AUC:", auc)

    os.makedirs("models", exist_ok=True)
    dump(model, "models/xgb_gpu.joblib")

if __name__ == "__main__":
    main()
