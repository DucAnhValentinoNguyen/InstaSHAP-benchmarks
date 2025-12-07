import os
from src.data import load_adult
from src.models import train_random_forest, save_model

def main():
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names = load_adult()

    model, auc = train_random_forest(X_train, y_train, X_val, y_val)
    print(f"Validation AUC: {auc:.4f}")

    os.makedirs("models", exist_ok=True)
    save_model(model, "models/rf_adult.joblib")
    print("Model saved to models/rf_adult.joblib")

if __name__ == "__main__":
    main()
