# Conceptual Perceptron and Support Vector Machines (SVM)

## Perceptron

### Conceptual Overview

The **Perceptron** is one of the simplest linear classifiers, introduced by Frank Rosenblatt in 1958. It's a binary classifier that works by finding a linear decision boundary to separate two classes.

#### Key Concepts:
- **Linear Classifier**: Uses a weighted sum of input features to make predictions
- **Decision Boundary**: A line (2D) or hyperplane (nD) that separates classes
- **Activation Function**: Outputs +1 or -1 based on whether the weighted sum exceeds a threshold
- **Learning Rule**: Updates weights when misclassification occurs
- **Convergence**: Guaranteed only for linearly separable data

#### Mathematical Foundation:
- **Prediction**: $\hat{y} = \text{sign}(w \cdot x + b)$
- **Weight Update**: $w := w + \eta \cdot (y - \hat{y}) \cdot x$ (when error occurs)
- **Bias Update**: $b := b + \eta \cdot (y - \hat{y})$

Where $\eta$ is the learning rate and $y$ is the true label.

### Practical Implementation

```python
from sklearn.linear_model import Perceptron
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Generate sample data
X, y = make_classification(n_samples=100, n_features=2, n_informative=2,
                          n_redundant=0, random_state=42)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                      random_state=42)

# Create and train Perceptron
clf = Perceptron(eta0=0.01, max_iter=100, random_state=42)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))
```

### Key Parameters (Sklearn)
| Parameter | Description |
|-----------|-------------|
| `eta0` | Learning rate |
| `max_iter` | Maximum passes over training set |
| `random_state` | Seed for reproducibility |
| `tol` | Tolerance for stopping criterion |
| `penalty` | Regularization type (None, 'l2', 'l1', 'elasticnet') |

### Practical Advantages & Limitations
| Advantages | Limitations |
|-----------|------------|
| Very fast training | Only for binary classification (basic) |
| Memory efficient | Requires linearly separable data |
| Simple to understand | Sensitive to feature scaling |
| Works online (one sample at a time) | Cannot handle non-linear patterns |

---

## Support Vector Machines (SVM)

### Conceptual Overview

**Support Vector Machines** extend the Perceptron concept by finding the optimal hyperplane that maximizes the margin between classes. They are powerful classifiers that can handle both linear and non-linear problems through kernel tricks.

#### Key Concepts:
- **Maximum Margin Principle**: Finds the decision boundary furthest from both classes
- **Support Vectors**: Data points closest to the decision boundary
- **Kernel Trick**: Maps data to higher dimensions to handle non-linear separation
- **C Parameter**: Trade-off between correct classification and margin maximization
- **Robust**: Works well even when classes aren't perfectly separable

#### Mathematical Foundation:
- **Objective**: Maximize margin while minimizing training errors
- **Margin**: $\frac{2}{||w||}$ (distance between hyperplanes)
- **Decision Function**: $f(x) = \text{sign}(w \cdot \phi(x) + b)$

Where $\phi(x)$ is the kernel transformation of input $x$.

### Practical Implementation

#### Linear SVM
```python
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Generate sample data
X, y = make_classification(n_samples=200, n_features=5, n_informative=3,
                          n_redundant=0, random_state=42)

# Feature scaling (important for SVM!)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2,
                                                      random_state=42)

# Create and train Linear SVM
svm_linear = SVC(kernel='linear', C=1.0, random_state=42)
svm_linear.fit(X_train, y_train)

# Predictions
y_pred = svm_linear.predict(X_test)

# Evaluate
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Number of Support Vectors: {len(svm_linear.support_vectors_)}")
print(classification_report(y_test, y_pred))
```

#### Non-linear SVM with RBF Kernel
```python
from sklearn.svm import SVC
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Generate non-linear data
X, y = make_moons(n_samples=300, noise=0.1, random_state=42)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2,
                                                      random_state=42)

# RBF Kernel SVM (handles non-linear patterns)
svm_rbf = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm_rbf.fit(X_train, y_train)

# Evaluate
y_pred = svm_rbf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
```

### Key Parameters (Sklearn)

| Parameter | Description | Common Values |
|-----------|-------------|---|
| `kernel` | Type of kernel function | 'linear', 'rbf', 'poly', 'sigmoid' |
| `C` | Regularization parameter (inverse) | 0.1, 1, 10, 100 |
| `gamma` | Kernel coefficient for 'rbf', 'poly', 'sigmoid' | 'scale', 'auto', or float |
| `degree` | Degree of polynomial kernel | 2, 3, 4 (default: 3) |
| `random_state` | Seed for reproducibility | Any integer |

### Kernel Functions Explained

| Kernel | Use Case | Formula |
|--------|----------|---------|
| **Linear** | Linearly separable data | $K(x_i, x_j) = x_i \cdot x_j$ |
| **RBF** | Non-linear, general-purpose | $K(x_i, x_j) = e^{-\gamma \|x_i - x_j\|^2}$ |
| **Polynomial** | When non-linearity is polynomial | $K(x_i, x_j) = (x_i \cdot x_j + 1)^d$ |
| **Sigmoid** | Neural network-like behavior | $K(x_i, x_j) = \tanh(\kappa x_i \cdot x_j + \theta)$ |

### Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

# Grid search with cross-validation
grid_search = GridSearchCV(SVC(), param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best CV Score: {grid_search.best_score_:.4f}")

# Use best model
best_svm = grid_search.best_estimator_
y_pred = best_svm.predict(X_test)
```

### Practical Advantages & Limitations
| Advantages | Limitations |
|-----------|------------|
| Works well in high dimensions | Slow for very large datasets |
| Effective with kernel trick | Requires feature scaling |
| Robust to outliers | Hard to interpret results |
| Works for binary & multi-class | Hyperparameter tuning necessary |
| Strong theoretical foundation | Memory intensive (stores support vectors) |

---

## Comparison: Perceptron vs SVM

| Aspect | Perceptron | SVM |
|--------|-----------|-----|
| **Complexity** | Simple | More complex |
| **Performance** | Basic | Superior |
| **Non-linear** | No | Yes (with kernels) |
| **Training Speed** | Very Fast | Slower for large data |
| **Data Requirements** | Linear separation | Flexible |
| **Interpretability** | High | Lower |
| **Use Cases** | Simple classification | Complex patterns |

---

## Practical Workflow

### 1. Data Preparation
```python
from sklearn.preprocessing import StandardScaler

# Always scale features for SVM
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### 2. Model Selection
- Start with **Linear SVM** if data is possibly linearly separable
- Use **RBF SVM** for non-linear patterns
- Try **Perceptron** for simple, fast baselines on large datasets

### 3. Cross-Validation & Tuning
```python
from sklearn.model_selection import cross_validate

# Evaluate with cross-validation
scores = cross_validate(SVC(kernel='rbf'), X_train_scaled, y_train, 
                        cv=5, scoring=['accuracy', 'precision', 'recall'])

print(f"Mean Accuracy: {scores['test_accuracy'].mean():.4f}")
```

### 4. Performance Evaluation
```python
from sklearn.metrics import (accuracy_score, precision_score, 
                             recall_score, f1_score, confusion_matrix)

y_pred = model.predict(X_test_scaled)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
print(confusion_matrix(y_test, y_pred))
```

---

## Key Takeaways for Practice

✓ **Feature Scaling**: Always scale features before using SVM  
✓ **Kernel Selection**: Start with RBF for unknown data patterns  
✓ **Parameter Tuning**: Use GridSearchCV for optimal hyperparameters  
✓ **Cross-Validation**: Prevent overfitting with k-fold cross-validation  
✓ **Perceptron First**: Use as baseline for quick, simple classification tasks  
✓ **SVM for Complex**: Choose SVM when dealing with non-linear relationships
