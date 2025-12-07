import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from joblib import load
from sklearn.cluster import KMeans

from src.data import load_adult
from experiments.train_surrogate_gpu import train_surrogate

os.makedirs("models/compressed", exist_ok=True)


# -----------------------------------------------------------
# Helper: Uniform sampling compression
# -----------------------------------------------------------
def compress_uniform(X, k, seed=0):
    return X.sample(n=k, random_state=seed).values.astype(np.float32)


# -----------------------------------------------------------
# Helper: Kmeans compression
# -----------------------------------------------------------
def compress_kmeans(X, k, seed=0):
    km = KMeans(n_clusters=k, random_state=seed)
    km.fit(X.values)
    return km.cluster_centers_.astype(np.float32)


# -----------------------------------------------------------
# Train surrogate with compressed background
# -----------------------------------------------------------
def train_compressed_surrogate(X_train, bb_model, features, background_k, method):

    # 1. Compute compressed background
    if method == "uniform":
        compressed = compress_uniform(X_train, background_k)
    elif method == "kmeans":
        compressed = compress_kmeans(X_train, background_k)
    else:
        raise ValueError("Unknown compression method")

    print(f"Compressed background shape: {compressed.shape}")

    # 2. Compute new baseline = mean of compressed points
    baseline = np.mean(compressed, axis=0).astype(np.float32)

    # 3. Precompute f(x) on original train set
    print("Precomputing f(x) for uncompressed train...")
    fx_all = bb_model.predict_proba(X_train)[:, 1]

    # 4. Train surrogate using the optimized GPU trainer
    model = train_surrogate(
        X_train,
        bb_model,
        fx_all,
        baseline,
        features,
        n_samples=15000,   # smaller because data is compressed
        batch_size=512,
        n_epochs=4,
        lr=1e-3,
    )

    out_path = f"models/compressed/surrogate_k{background_k}_{method}.pth"
    torch.save({"state_dict": model.state_dict(),
                "feature_names": features},
               out_path)

    print(f"Saved compressed surrogate: {out_path}")


# -----------------------------------------------------------
# Main experiment script
# -----------------------------------------------------------
def main():

    X_train, X_val, X_test, y_train, y_val, y_test, features = load_adult()
    print("Loaded dataset")

    bb_model = load("models/xgb_gpu.joblib")
    print("Loaded black-box model")

    # Compression levels
    ks = [5, 10, 20, 50]

    methods = ["uniform", "kmeans"]

    for k in ks:
        for method in methods:
            print("\n========================================")
            print(f"Training compressed surrogate: k={k} method={method}")
            print("========================================")

            train_compressed_surrogate(X_train, bb_model, features, k, method)


if __name__ == "__main__":
    main()
