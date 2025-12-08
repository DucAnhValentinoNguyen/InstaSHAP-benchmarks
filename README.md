![Python](https://img.shields.io/badge/python-3.10-blue)
![Status](https://img.shields.io/badge/status-active-brightgreen)
![License](https://img.shields.io/badge/license-MIT-lightgrey)
# Hybrid InstaSHAP: Combining Clustering and Distribution Compression for Efficient Explainability

This repository contains the implementation, experiments, and analysis for the **Hybrid InstaSHAP** seminar project at **LMU Munich**, supervised by **Dr. Giuseppe Casalicchio**.

The goal of this project is to develop a **hybrid background summarization strategy** for SHAP that combines:

- **Clustering-based prototypes** (semantic representativeness)  
- **Distribution compression via kernel thinning** (statistical representativeness)  
- **InstaSHAP-style additive surrogate models** for instant SHAP approximation  

This hybrid approach improves **runtime efficiency**, **stability**, and **approximation quality** compared to standard background sampling.

---

## 🚀 Motivation

Traditional SHAP struggles with:

- High computational cost  
- Instability due to i.i.d. background sampling  
- Redundant or non-representative background points  

Two existing solutions address parts of this problem:

| Approach | Strength | Weakness |
|----------|----------|----------|
| **Clustering** | Produces semantic prototypes | Poor distributional coverage |
| **Distribution Compression (Kernel Thinning)** | Strong statistical guarantees | No semantic interpretation |

The **Hybrid InstaSHAP** method combines both for a balanced, efficient background set.

---

## 🧠 Method Overview

### 1. **Clustering Step**
Select K-Means (or alternative) cluster centroids as semantic representatives.

### 2. **Distribution Compression Step**
Apply kernel thinning/herding to ensure global distributional representativeness.

### 3. **Surrogate Additive Model (InstaSHAP)**
Train fast additive models to approximate SHAP values directly.

### 4. **Hybrid Background Set**
```
Hybrid = Cluster Representatives ∪ Compressed Samples
```

Benefits:

- ✔ Stable SHAP value estimates  
- ✔ Faster computation  
- ✔ Fewer background samples required  
- ✔ Balances semantic + statistical representativeness  

---

## 📂 Repository Structure

```
Hybrid-InstaSHAP/
│
├── src/               # Implementation of clustering, compression, hybrid SHAP
├── experiments/       # Experiment scripts and configs
├── notebooks/         # Analysis and visualization
├── figures/           # Plots for the seminar report
├── results/           # Metrics and aggregated outputs
├── models/            # Saved clusterers / explainers
├── logs/              # Logging output
├── data/              # Small example datasets
└── requirements.txt   # Dependencies
```

---

## ⚙️ Installation

```bash
git clone https://github.com/DucAnhValentinoNguyen/Hybrid-InstaSHAP
cd Hybrid-InstaSHAP
chmod +x setup.sh
./setup.sh
```

---

## 🧪 Running Experiments

```bash
python experiments/run_experiments.py
```

Config-based run:

```bash
python experiments/run_experiments.py --config experiments/configs/hybrid.yaml
```

Interactive exploration:

```
notebooks/
```

---

## 📊 Results Summary

- Runtime improves compared to raw SHAP and compression baselines  
- Hybrid approach yields **more stable SHAP values**  
- Works well even with **small background sizes**  
- Combines semantic + statistical representativeness  

Figures located in `figures/`.

---

## 📘 Project Information

**Author:** Duc-Anh Nguyen & Shuai Wang

**Supervisor:** Prof. Giuseppe Casalicchio  
**Institution:** LMU Munich, Department of Statistics  
**Course:** XAI Seminar (Winter Semester 2025–26)

---

## 📄 License

MIT License

Copyright (c) 2025 Duc-Anh Nguyen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

## 🔮 Future Work

- Extend to text + tabular multimodal models  
- Evaluate on higher-dimensional datasets  
- Compare against FastSHAP / Amortized SHAP  
- GPU-optimized kernel thinning for large-scale compression  

