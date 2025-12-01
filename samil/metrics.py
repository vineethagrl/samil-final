from sklearn.metrics import balanced_accuracy_score, roc_auc_score
import numpy as np

def balanced_acc(y_true, y_pred):
    return balanced_accuracy_score(y_true, y_pred)

def auroc_ovr(y_true, probs):
    try:
        return roc_auc_score(y_true, probs, multi_class="ovr")
    except Exception:
        return np.nan
