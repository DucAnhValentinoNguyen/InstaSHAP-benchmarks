import pandas as pd
from sklearn.model_selection import train_test_split

def load_adult(test_size=0.2, val_size=0.2, random_state=42):
    df = pd.read_csv("data/adult.csv")

    y = df["class"].astype("category").cat.codes
    X = df.drop(columns=["class"])
    X = pd.get_dummies(X, drop_first=True)

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size,
        random_state=random_state, stratify=y_train_val
    )

    return X_train, X_val, X_test, y_train, y_val, y_test, X.columns.tolist()
