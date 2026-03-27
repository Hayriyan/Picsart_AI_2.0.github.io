# ML Evaluation Metrics

This file covers the **main machine learning evaluation metrics**, when to use them, and small code examples.

---

## 1) Classification Metrics

Use these when your target is a class label (for example: spam vs not spam).

### Accuracy
- **What it is:** Fraction of correct predictions.
- **Formula:** `(TP + TN) / (TP + TN + FP + FN)`
- **Best for:** Balanced datasets where all errors have similar cost.
- **Watch out:** Can be misleading on imbalanced data.

### Precision
- **What it is:** Of predicted positives, how many are actually positive.
- **Formula:** `TP / (TP + FP)`
- **Best for:** When false positives are costly (for example: fraud alerts).

### Recall (Sensitivity, True Positive Rate)
- **What it is:** Of actual positives, how many are correctly found.
- **Formula:** `TP / (TP + FN)`
- **Best for:** When false negatives are costly (for example: disease detection).

### F1-Score
- **What it is:** Harmonic mean of precision and recall.
- **Formula:** `2 * (Precision * Recall) / (Precision + Recall)`
- **Best for:** Need a balance between precision and recall, especially with class imbalance.

### ROC-AUC
- **What it is:** Area under ROC curve (TPR vs FPR over thresholds).
- **Range:** `0.5` (random) to `1.0` (perfect).
- **Best for:** Overall ranking quality of probabilistic binary classifiers.

### Log Loss (Cross-Entropy)
- **What it is:** Penalizes wrong and overconfident probability predictions.
- **Best for:** Evaluating probability quality, not just hard labels.

### Confusion Matrix
- **What it is:** Table of `TP`, `TN`, `FP`, `FN`.
- **Best for:** Quick error analysis by class.

### Metrics (One-Line Intuition + Math)

1. **Accuracy (Acc)**  
   "How often am I correct overall?"

   `Accuracy = (TP + TN) / (TP + TN + FP + FN)`

2. **Precision (Pr)**  
   "When I predict positive, how often am I right?"

   `Precision = TP / (TP + FP)`

3. **Recall (Sensitivity, TPR)**  
   "Out of all real positives, how many did I catch?"

   `Recall = TP / (TP + FN)`

4. **Specificity (Spec, TNR)**  
   "Out of all real negatives, how many did I correctly reject?"

   `Specificity = TN / (TN + FP)`

5. **F1 Score**  
   "Balanced score between precision and recall."

   `F1 = 2 * (Precision * Recall) / (Precision + Recall)`

### Bonus Intuitions (Very Important)

- `FP` hurts **Precision**.
- `FN` hurts **Recall**.
- Threshold up (`threshold ↑`) -> Precision up, Recall down.
- Threshold down (`threshold ↓`) -> Recall up, Precision down.

#### Classification Code Example (scikit-learn)
```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, confusion_matrix
)

# Example data
y_true = [1, 0, 1, 1, 0, 1, 0, 0]
y_pred = [1, 0, 1, 0, 0, 1, 1, 0]          # predicted labels
y_prob = [0.90, 0.15, 0.80, 0.40, 0.20, 0.95, 0.60, 0.05]  # prob for class 1

print("Accuracy:", accuracy_score(y_true, y_pred))
print("Precision:", precision_score(y_true, y_pred))
print("Recall:", recall_score(y_true, y_pred))
print("F1:", f1_score(y_true, y_pred))
print("ROC-AUC:", roc_auc_score(y_true, y_prob))
print("Log Loss:", log_loss(y_true, y_prob))
print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
```

---

## 2) Regression Metrics

Use these when your target is a continuous number (for example: house price).

### MAE (Mean Absolute Error)
- **What it is:** Average absolute difference between true and predicted values.
- **Formula:** `mean(|y_true - y_pred|)`
- **Best for:** Interpretable average error in target units.

### MSE (Mean Squared Error)
- **What it is:** Average squared difference.
- **Formula:** `mean((y_true - y_pred)^2)`
- **Best for:** Strongly penalizing large errors.

### RMSE (Root Mean Squared Error)
- **What it is:** Square root of MSE.
- **Best for:** Same units as target, still emphasizes large errors.

### R2 Score (Coefficient of Determination)
- **What it is:** Proportion of variance explained by the model.
- **Range:** Usually `<= 1` (can be negative if model is very poor).
- **Best for:** Overall fit comparison between models on same dataset.

### MAPE (Mean Absolute Percentage Error)
- **What it is:** Average absolute percentage error.
- **Formula:** `mean(|(y_true - y_pred) / y_true|) * 100`
- **Best for:** Percentage-based interpretation.
- **Watch out:** Problematic when true values are near zero.

#### Regression Code Example (scikit-learn)
```python
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error
)
import numpy as np

# Example data
y_true = np.array([100, 120, 130, 150, 170])
y_pred = np.array([110, 115, 128, 140, 180])

mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_true, y_pred)
mape = mean_absolute_percentage_error(y_true, y_pred) * 100

print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2:", r2)
print("MAPE (%):", mape)
```

---

## 3) Quick Metric Selection Guide

- **Balanced classification:** Accuracy + F1.
- **Imbalanced classification:** Precision, Recall, F1, ROC-AUC (or PR-AUC).
- **Need calibrated probabilities:** Log Loss + ROC-AUC.
- **General regression:** MAE + RMSE + R2.
- **Business-friendly percentage error:** MAPE (if no near-zero targets).

---

## 4) Best Practices

- Evaluate on a **validation/test set**, not training data.
- Use **cross-validation** for more reliable estimates.
- Track **multiple metrics** (one metric alone can mislead).
- Choose metrics based on **business cost** of error types.

