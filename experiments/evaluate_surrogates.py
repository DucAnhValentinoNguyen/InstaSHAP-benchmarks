import os
import time
import numpy as np
import pandas as pd
import torch
from joblib import load
from tqdm import tqdm

from src.data import load_adult
from src.surrogate import AdditiveSurrogate


# --------------------------------------------------------------
# Helper: Load a surrogate model
# --------------------------------------------------------------
def load_surrogate(path, feature_names):
    model = AdditiveSurrogate(len(feature_names))
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


# --------------------------------------------------------------
# Compute surrogate predictions
# --------------------------------------------------------------
def surrogate_predict(model, X):
    X_t = torch.tensor(X.values.astype(np.float32))
    with torch.no_grad():
        pred = model(X_t).numpy().squeeze()
    return pred


# --------------------------------------------------------------
# Compute per-instance SHAP-like contributions
# (the surrogate is additive in features)
# --------------------------------------------------------------
def surrogate_shap(model, X):
    X_t = torch.tensor(X.values.astype(np.float32))
    with torch.no_grad():
        contribs = model.feature_contribs(X_t).numpy()
    return contribs


# --------------------------------------------------------------
# Main evaluation script
# --------------------------------------------------------------
def main():

    print("Loading dataset...")
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names = load_adult()

    print("Loading black-box (XGBoost) model...")
    bb = load("models/xgb_gpu.joblib")

    print("Loading full surrogate...")
    full_path = "models/surrogate_gpu.pth"
    full_model = load_surrogate(full_path, feature_names)

    # Precompute benchmark
    print("Computing black-box predictions on test...")
    f_test = bb.predict_proba(X_test)[:, 1]

    print("Computing full surrogate predictions...")
    g_full = surrogate_predict(full_model, X_test)

    print("Full surrogate - MSE:", np.mean((g_full - f_test) ** 2))
    print("Full surrogate - Corr:", np.corrcoef(g_full, f_test)[0, 1])

    # ----------------------------------------------------------
    # SHAP evaluation sample
    # ----------------------------------------------------------
    sample = X_test.sample(300, random_state=0)
    f_test_s = bb.predict_proba(sample)[:, 1]
    shap_full = surrogate_shap(full_model, sample)

    # ----------------------------------------------------------
    # Evaluate compressed surrogates
    # ----------------------------------------------------------
    results = []

    for fname in sorted(os.listdir("models/compressed")):
        if not fname.endswith(".pth"):
            continue

        path = os.path.join("models/compressed", fname)
        print("\nEvaluating:", fname)

        # --- load model ---
        model = load_surrogate(path, feature_names)

        # --- surrogate preds ---
        g_pred = surrogate_predict(model, X_test)

        mse = np.mean((g_pred - f_test) ** 2)
        corr = np.corrcoef(g_pred, f_test)[0, 1]

        # --- runtime ---
        t0 = time.time()
        _ = surrogate_predict(model, sample)
        runtime = (time.time() - t0) * 1000  # ms

        # --- SHAP ---
        shap_c = surrogate_shap(model, sample)

        shap_corr = np.mean([
            np.corrcoef(shap_full[i], shap_c[i])[0, 1]
            for i in range(len(shap_full))
        ])

        # Top-5 feature overlap
        overlap = np.mean([
            len(set(np.argsort(shap_full[i])[-5:]) &
                set(np.argsort(shap_c[i])[-5:])) / 5
            for i in range(len(shap_full))
        ])

        results.append({
            "model": fname,
            "mse": mse,
            "corr": corr,
            "shap_corr": shap_corr,
            "top5_overlap": overlap,
            "runtime_ms": runtime,
        })

    df = pd.DataFrame(results)
    df.to_csv("results/compression_evaluation.csv", index=False)

    print("\nSaved results to results/compression_evaluation.csv")
    print(df)


if __name__ == "__main__":
    main()
