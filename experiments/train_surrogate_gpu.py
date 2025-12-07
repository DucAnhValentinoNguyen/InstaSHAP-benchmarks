import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
from datetime import datetime

# ============================================================
# Create logs directory
# ============================================================

os.makedirs("logs", exist_ok=True)

# ============================================================
# Logging Setup
# ============================================================

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
logging.basicConfig(
    filename=f"logs/surrogate_train_{timestamp}.log",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logging.info("Initialized surrogate training script")

# ============================================================
# FAST MaskDataset (no XGBoost inside __getitem__)
# ============================================================

class MaskDataset(Dataset):
    """
    Fast dataset:
    - Returns x, mask, fx
    - No XGBoost calls inside __getitem__
    - XGBoost only runs ONCE per batch inside training loop
    """

    def __init__(self, X, fx_all, baseline, n_samples=20000, random_state=0):
        self.X = X.values.astype(np.float32)
        self.fx_all = fx_all.astype(np.float32).reshape(-1, 1)
        self.baseline = baseline.astype(np.float32)
        self.n_data, self.n_feat = self.X.shape
        self.n_samples = n_samples
        self.rng = np.random.default_rng(random_state)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # random training example
        i = self.rng.integers(0, self.n_data)
        x = self.X[i]     # shape (d,)
        fx = self.fx_all[i]  # shape (1,)

        # random mask 0/1 vector
        mask = (self.rng.random(self.n_feat) < 0.5).astype(np.float32)

        return torch.from_numpy(x), torch.from_numpy(mask), torch.tensor(fx, dtype=torch.float32)


# ============================================================
# Surrogate Model (Additive)
# ============================================================

from src.surrogate import AdditiveSurrogate

# ============================================================
# Training Loop (vectorized masking + batched XGBoost call)
# ============================================================

def train_surrogate(
        X_train, bb_model, fx_all, baseline, feature_names,
        n_samples=20000, batch_size=512, n_epochs=5, lr=1e-3):

    dataset = MaskDataset(X_train, fx_all, baseline, n_samples=n_samples)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,           # IMPORTANT: XGBoost cannot run in worker processes
        pin_memory=True
    )

    model = AdditiveSurrogate(len(feature_names)).cuda()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    print("Starting training...")
    logging.info("Starting surrogate training...")

    for epoch in range(1, n_epochs + 1):
        model.train()
        total = 0.0

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{n_epochs}")

        for x, mask, fx in pbar:
            x = x.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)
            fx = fx.cuda(non_blocking=True).view(-1)   # shape (batch,)

            # -----------------------------------------
            # Construct masked inputs X_masked (GPU)
            # -----------------------------------------
            baseline_gpu = torch.tensor(baseline, device="cuda")
            x_masked = torch.where(mask == 1, x, baseline_gpu)

            # -----------------------------------------
            # XGBoost predict (one batch call)
            # -----------------------------------------
            x_masked_np = x_masked.cpu().numpy()
            f_masked_np = bb_model.predict_proba(
                pd.DataFrame(x_masked_np, columns=feature_names)
            )[:, 1]

            f_masked = torch.tensor(f_masked_np, device="cuda").view(-1)

            # -----------------------------------------
            # Target = f(x) - f(x_masked)
            # -----------------------------------------
            y = fx - f_masked    # shape: (batch,)
            y = y.unsqueeze(1)   # shape: (batch, 1)

            # -----------------------------------------
            # Forward + loss + backward
            # -----------------------------------------
            pred = model(x)  # (batch, 1)
            loss = loss_fn(pred, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss.item()
            pbar.set_postfix({"loss": loss.item()})

        epoch_loss = total / len(loader)
        logging.info(f"Epoch {epoch}/{n_epochs} | Loss {epoch_loss:.6f}")
        print(f"Epoch {epoch}/{n_epochs} - Loss {epoch_loss:.6f}")

    return model


# ============================================================
# Main
# ============================================================

from joblib import load
from src.data import load_adult

def main():
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names = load_adult()
    print("Train shape:", X_train.shape)

    print("Loading black-box model...")
    bb_model = load("models/xgb_gpu.joblib")

    print("Precomputing f(x) for entire dataset...")
    fx_all = bb_model.predict_proba(X_train)[:, 1]
    baseline = X_train.mean(axis=0).values  # baseline feature means

    model = train_surrogate(
        X_train, bb_model, fx_all, baseline, feature_names,
        n_samples=20000, batch_size=512, n_epochs=5
    )

    output_path = "models/surrogate_gpu.pth"
    torch.save({
        "state_dict": model.state_dict(),
        "feature_names": feature_names
    }, output_path)

    print(f"Saved surrogate model to {output_path}")
    logging.info(f"Saved surrogate model to {output_path}")


if __name__ == "__main__":
    main()
