import xgboost as xgb
from sklearn.metrics import roc_auc_score

def train_xgboost_gpu(X_train, y_train, X_val, y_val):
    model = xgb.XGBClassifier(
        tree_method="gpu_hist",
        predictor="gpu_predictor",
        gpu_id=0,
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="auc"
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    val_pred = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, val_pred)
    return model, auc
