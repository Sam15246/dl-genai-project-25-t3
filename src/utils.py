import numpy as np
from sklearn.metrics import f1_score, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

def macro_f1(y_true, y_pred):
    return float(np.mean([f1_score(y_true[:, i], y_pred[:, i], zero_division=0)
                          for i in range(y_true.shape[1])]))

def sweep_thresholds(y_true, probs, grid=None):
    if grid is None:
        grid = np.linspace(0.01, 0.99, 99)

    best = []
    for i in range(probs.shape[1]):
        best_thr, best_f1 = 0.5, -1.0
        for thr in grid:
            pred = (probs[:, i] >= thr).astype(int)
            f1 = f1_score(y_true[:, i], pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thr = thr
        best.append(best_thr)
    return np.array(best)
