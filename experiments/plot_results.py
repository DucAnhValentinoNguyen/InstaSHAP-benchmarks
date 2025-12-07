import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs("figures", exist_ok=True)

# -----------------------------------------------------------
# Helper: extract method + k from filename like:
# surrogate_k10_uniform.pth
# -----------------------------------------------------------
def parse_model_name(name):
    match = re.match(r".*k(\d+)_(\w+)\.pth", name)
    if match:
        k = int(match.group(1))
        method = match.group(2)
        return k, method
    return None, None


# -----------------------------------------------------------
# Load results
# -----------------------------------------------------------
df = pd.read_csv("results/compression_evaluation.csv")

# Parse fields
df["k"] = df["model"].apply(lambda x: parse_model_name(x)[0])
df["method"] = df["model"].apply(lambda x: parse_model_name(x)[1])

# Ensure sorted properly
df = df.sort_values(["method", "k"])


# -----------------------------------------------------------
# PLOT 1 — Surrogate MSE vs k
# -----------------------------------------------------------
plt.figure(figsize=(8, 5))
sns.lineplot(data=df, x="k", y="mse", hue="method", marker="o")
plt.title("Surrogate MSE vs Compression Level k")
plt.xlabel("k (Compression Level)")
plt.ylabel("MSE vs Black-Box Predictions")
plt.grid(True)
plt.savefig("figures/mse_vs_k.png", dpi=150)
plt.close()


# -----------------------------------------------------------
# PLOT 2 — Surrogate–BB Correlation vs k
# -----------------------------------------------------------
plt.figure(figsize=(8, 5))
sns.lineplot(data=df, x="k", y="corr", hue="method", marker="o")
plt.title("Correlation with Black-Box Model vs k")
plt.xlabel("k")
plt.ylabel("Correlation")
plt.grid(True)
plt.savefig("figures/corr_vs_k.png", dpi=150)
plt.close()


# -----------------------------------------------------------
# PLOT 3 — SHAP Similarity vs k
# -----------------------------------------------------------
plt.figure(figsize=(8, 5))
sns.lineplot(data=df, x="k", y="shap_corr", hue="method", marker="o")
plt.title("SHAP Vector Similarity vs k")
plt.xlabel("k")
plt.ylabel("Mean SHAP Correlation")
plt.grid(True)
plt.savefig("figures/shap_corr_vs_k.png", dpi=150)
plt.close()


# -----------------------------------------------------------
# PLOT 4 — Top-5 Feature Overlap vs k
# -----------------------------------------------------------
plt.figure(figsize=(8, 5))
sns.lineplot(data=df, x="k", y="top5_overlap", hue="method", marker="o")
plt.title("Top-5 Feature Ranking Overlap vs k")
plt.xlabel("k")
plt.ylabel("Overlap (0–1)")
plt.grid(True)
plt.savefig("figures/top5_overlap_vs_k.png", dpi=150)
plt.close()


# -----------------------------------------------------------
# PLOT 5 — Runtime vs k
# -----------------------------------------------------------
plt.figure(figsize=(8, 5))
sns.lineplot(data=df, x="k", y="runtime_ms", hue="method", marker="o")
plt.title("Runtime to Compute SHAP (300 samples) vs k")
plt.xlabel("k")
plt.ylabel("Runtime (ms)")
plt.grid(True)
plt.savefig("figures/runtime_vs_k.png", dpi=150)
plt.close()


print("Saved plots in directory: figures/")
print(df)
