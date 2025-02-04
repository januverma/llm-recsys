import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    accuracy_score,
    mean_squared_error,
    mean_absolute_error
)

def binary_classification_metrics(actuals, preds):
    actuals_binary = [1 if x >= 4 else 0 for x in actuals]
    preds_binary = [1 if x == 'Yes' else 0 for x in preds]

    acc = accuracy_score(actuals_binary, preds_binary)
    precision = precision_score(actuals_binary, preds_binary)
    recall = recall_score(actuals_binary, preds_binary)
    f1 = f1_score(actuals_binary, preds_binary)
    auc = roc_auc_score(actuals_binary, preds_binary)

    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"AUC:       {auc:.4f}")
    print(f"Accuracy:  {acc:.4f}")
    return None


def regression_metrics(actuals, preds):
    rmse = np.sqrt(mean_squared_error(actuals, preds))
    mae = mean_absolute_error(actuals, preds)
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    return None