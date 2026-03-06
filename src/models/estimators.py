import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score


def train_xgboost_model(X_train, y_train, X_test, y_test, use_early_stopping=True):
    """
    Trains an XGBoost classifier with best practices for imbalanced fraud data.
    Uses 'early_stopping_rounds' directly in the constructor.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        X_test (pd.DataFrame): Testing features for early stopping / evaluation.
        y_test (pd.Series): Testing target for early stopping / evaluation.
        use_early_stopping (bool): Whether to use early stopping to prevent overfitting.

    Returns:
        xgb.XGBClassifier: The trained XGBoost model.
    """
    # 1. Calculate scale_pos_weight for handling class imbalance
    scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
    print(f"Scale Pos Weight: {scale_pos_weight:.2f}")

    # 2. Instantiate the XGBoost classifier
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='aucpr',
        scale_pos_weight=scale_pos_weight,
        tree_method='hist',
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=50 if use_early_stopping else None,
    )

    # 3. Train the model
    if use_early_stopping:
        print("Training model with early stopping...")
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )
    else:
        print("Training model without early stopping...")
        model.fit(X_train, y_train)

    # 4. Evaluate on test data
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc_pr  = average_precision_score(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    print(f"Model Performance on Test Set:")
    print(f"  - AUC-PR : {auc_pr:.4f}")
    print(f"  - ROC AUC: {roc_auc:.4f}")

    return model
