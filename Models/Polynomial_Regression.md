# Polynomial Regression — Deep Guide

## Overview

Polynomial Regression is a regression technique that extends linear regression by modeling the relationship between input features and a target variable as an $n$-th degree polynomial. Instead of fitting a straight line to data, polynomial regression fits a curve, capturing non-linear relationships while remaining a linear model in its parameters (making it a special case of linear regression with engineered polynomial features).

Polynomial regression is appealing because it maintains interpretability and computational efficiency of linear models while providing additional flexibility to fit curved, non-linear patterns in data. However, high-degree polynomials can lead to severe overfitting, require careful feature scaling, and can exhibit extreme behavior outside the training data range (extrapolation instability). Modern alternatives like splines, kernelized methods (kernel ridge regression), or non-parametric models (random forests, neural networks) often provide better generalization.

## Intuition

The core idea: transform the original feature space by creating polynomial terms (squares, cubes, interactions), then apply linear regression on the augmented feature set.

**Key insight:** Polynomial regression is mathematically linear in its parameters even though it's non-linear in the original features. This allows us to leverage efficient linear algebra solutions while capturing curved relationships.

**Example progression:**

- Linear: $\hat{y} = \beta_0 + \beta_1 x$
- Quadratic (degree 2): $\hat{y} = \beta_0 + \beta_1 x + \beta_2 x^2$
- Cubic (degree 3): $\hat{y} = \beta_0 + \beta_1 x + \beta_2 x^2 + \beta_3 x^3$
- General degree $d$: $\hat{y} = \sum_{k=0}^{d} \beta_k x^k$

## When Polynomial Regression Is Appropriate

**Use polynomial regression when:**

- **Visual exploration suggests curved relationships**: scatter plots show clear non-linear patterns (U-shapes, S-curves, etc.).
- **Interpretability matters**: you need an explicit mathematical formula relating inputs to outputs.
- **Low-dimensional data (1-5 features)**: polynomial terms explode combinatorially with feature count (curse of dimensionality).
- **Extrapolation beyond training range is not required**: polynomial behavior can be extreme outside observed data.
- **Smooth, differentiable relationships**: the underlying process follows a smooth, continuous curve.
- **Physics or domain knowledge suggests polynomial form**: e.g., $v = v_0 + gt + \frac{1}{2}gt^2$ (position under gravity).

**When NOT to use polynomial regression:**

- **High-dimensional data**: number of polynomial features grows exponentially. For $d$ features and degree $p$, the number of polynomial terms is $\mathcal{O}(d^p)$, leading to overfitting and multicollinearity.
- **Prediction requires extrapolation**: polynomial fits can oscillate wildly outside the training domain (Runge's phenomenon).
- **Non-smooth data patterns**: abrupt changes, discontinuities, or locally different behavior; use piecewise methods or tree-based models instead.
- **Extreme overfitting is a concern and you lack sufficient regularization**: higher-degree polynomials fit noise rather than signal.
- **Computational efficiency critical**: consider simpler linear models or tree-based alternatives.

## Mathematical Formulation

### Univariate Polynomial Regression (Single Feature)

Given training data $\{(x_i, y_i)\}_{i=1}^{n}$ with $n$ samples and desired polynomial degree $d$, the first step is to create augmented features by stacking powers of the input:

**Feature Matrix (Augmented with polynomial terms):**

$$
\mathbf{X}_{\text{poly}} = \begin{bmatrix}
1 & x_1 & x_1^2 & \cdots & x_1^d \\
1 & x_2 & x_2^2 & \cdots & x_2^d \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
1 & x_n & x_n^2 & \cdots & x_n^d
\end{bmatrix}
$$

Each row $i$ contains powers of $x_i$ from 0 (constant 1) to $d$.

**Polynomial Model (Prediction Function):**

Once features are constructed, the polynomial regression model predicts as:

$$
\hat{y} = \beta_0 + \beta_1 x + \beta_2 x^2 + \cdots + \beta_d x^d = \sum_{k=0}^{d} \beta_k x^k
$$

where $\beta_k$ are the coefficients (parameters) to be learned from data.

**Parameter Learning via Ordinary Least Squares (OLS):**

Fit the model by minimizing the sum of squared errors (residuals):

$$
\boldsymbol{\beta}^* = \arg\min_{\boldsymbol{\beta}} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 = \arg\min_{\boldsymbol{\beta}} \left\|\mathbf{y} - \mathbf{X}_{\text{poly}} \boldsymbol{\beta}\right\|_2^2
$$

This minimization problem has a closed-form solution (assuming $\mathbf{X}_{\text{poly}}^T \mathbf{X}_{\text{poly}}$ is invertible):

$$
\boldsymbol{\beta}^* = \left(\mathbf{X}_{\text{poly}}^T \mathbf{X}_{\text{poly}}\right)^{-1} \mathbf{X}_{\text{poly}}^T \mathbf{y}
$$

This is the **normal equation** — a single computation gives optimal coefficients without iterative optimization.

### Multivariate Polynomial Regression (Multiple Features)

When you have multiple input features ($m > 1$), polynomial terms include all possible powers and interactions up to degree $d$.

**Example: Two features, degree 2:**

$$
\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \beta_{11} x_1^2 + \beta_{22} x_2^2 + \beta_{12} x_1 x_2
$$

Notice: besides $x_1^2$ and $x_2^2$ (pure quadratic terms), we also include the cross-term $x_1 x_2$.

**Feature Explosion Problem:**

The number of polynomial features grows combinatorially with both number of input features $m$ and degree $d$:

$$
\text{Number of polynomial features} = \binom{m + d}{d} = \frac{(m+d)!}{m! \cdot d!}
$$

**Concrete example:**

- $m=3$ features, $d=3$ degree $\Rightarrow$ $\binom{6}{3} = 20$ polynomial features
- $m=10$ features, $d=3$ degree $\Rightarrow$ $\binom{13}{3} = 286$ polynomial features
- $m=20$ features, $d=3$ degree $\Rightarrow$ $\binom{23}{3} = 1771$ polynomial features

This **curse of dimensionality** is a major limitation of polynomial regression; regularization or feature selection becomes critical.

### Prediction Error and Loss Metrics

After fitting, evaluate model quality using standard regression metrics:

**Mean Squared Error (MSE) — Average squared residual:**

$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} \left(y_i - \hat{y}_i\right)^2
$$

Lower MSE is better; heavily penalizes large errors (due to squaring).

**Root Mean Squared Error (RMSE) — MSE in original units:**

$$
\text{RMSE} = \sqrt{\text{MSE}} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} \left(y_i - \hat{y}_i\right)^2}
$$

More interpretable than MSE because it's in the same units as $y$.

**R-squared ($R^2$) — Proportion of variance explained (0 to 1 scale):**

$$
R^2 = 1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}} = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}
$$

where:

- $\text{SS}_{\text{res}}$ = sum of squared residuals (prediction errors)
- $\text{SS}_{\text{tot}}$ = total sum of squares (baseline variance)
- $\bar{y}$ = mean of all observations

**Interpretation:**

- $R^2 = 1$: Perfect fit (all predictions exact)
- $R^2 = 0.5$: Model explains 50% of variance in $y$
- $R^2 = 0$: Model no better than predicting the constant mean $\bar{y}$
- $R^2 < 0$: Model worse than the mean (rare, indicates severe overfitting)

## Bias-Variance Tradeoff

### Underfitting (High Bias, Low Variance)

- **Degree too low**: the polynomial does not capture the true underlying relationship.
- **Training and test error both high**: model systematically misses patterns.
- **Example:** fitting a linear model to data that is truly quadratic.

### Overfitting (Low Bias, High Variance)

- **Degree too high**: the polynomial fits noise and training peculiarities, not signal.
- **Training error very low, test error high**: model memorizes rather than generalizes.
- **Example:** fitting a degree-15 polynomial to a few dozen points.
- **Runge's phenomenon:** high-degree polynomials oscillate wildly near endpoints, creating unreliable extrapolations.

### Finding the Sweet Spot

Use **cross-validation** (k-fold or train-test split) to select degree $d$:

1. Train models for degrees $d = 1, 2, 3, \ldots, d_{\max}$.
2. Evaluate each on validation data.
3. Choose degree with lowest validation error.
4. Inspect training vs. validation error gap; if large, overfitting is present.

## Regularization Techniques

### Ridge Regression (L2 Regularization)

To prevent overfitting (especially with high-degree polynomials), add a **penalty** on large coefficient magnitudes to the loss function:

$$
\min_{\boldsymbol{\beta}} \left( \underbrace{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}_{\text{Data fit}} + \underbrace{\lambda \sum_{j=1}^{p} \beta_j^2}_{\text{Penalty on coeff. size}} \right)
$$

where:

- $\lambda \geq 0$ is the **regularization strength** (hyperparameter to tune)
- Larger $\lambda$ → stronger penalty → smaller coefficients (reduced variance, higher bias)
- $\lambda = 0$ → standard OLS (no regularization)

**Closed-form Solution (Modified Normal Equation):**

$$
\boldsymbol{\beta}_{\text{ridge}} = \left(\mathbf{X}^T \mathbf{X} + \lambda \mathbf{I}\right)^{-1} \mathbf{X}^T \mathbf{y}
$$

The $+\lambda \mathbf{I}$ term ensures the matrix is invertible (improving numerical stability).

### Lasso Regression (L1 Regularization)

Use L1 penalty (absolute values) instead of L2 (squared):

$$
\min_{\boldsymbol{\beta}} \left( \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^{p} |\beta_j| \right)
$$

**Key difference from Ridge:**

- **L1 encourages sparsity:** Many coefficients become exactly zero (not just small)
- **Automatic feature selection:** Only non-zero coefficients remain; effectively drops less important polynomial terms
- **More interpretable:** Final model uses only a subset of polynomial features
- **No closed-form solution:** Requires iterative optimization algorithms

Lasso is preferred when you want a simpler, sparser model with interpretability.

### Elastic Net (Hybrid: L1 + L2)

Combine both L1 and L2 penalties to get benefits of both Ridge and Lasso:

$$
\min_{\boldsymbol{\beta}} \left( \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 + \lambda_1 \sum_{j=1}^{p} |\beta_j| + \lambda_2 \sum_{j=1}^{p} \beta_j^2 \right)
$$

where:

- $\lambda_1 > 0$ controls L1 sparsity strength
- $\lambda_2 > 0$ controls L2 shrinkage strength

**Advantage:** Balances sparsity (from Lasso) with stability (from Ridge); often outperforms pure L1 or L2 on high-dimensional polynomial features.

## Feature Scaling and Preprocessing

### Standardization (Critical Best Practice)

Polynomial features can have vastly different scales. **Always standardize base features before creating polynomial terms:**

$$
x_{\text{scaled}} = \frac{x - \mu}{\sigma}
$$

where:

- $\mu$ = mean of feature $x$ over training data
- $\sigma$ = standard deviation of feature $x$

**Why standardization matters:**

1. **Numerical stability:** Prevents very large/small numbers from causing computational errors in matrix inversion
2. **Fair regularization:** All regularization penalties apply uniformly; prevents small-scale features from being favored
3. **Faster convergence:** Optimization algorithms converge quicker on standardized data
4. **Better interpretation:** Coefficient magnitudes are comparable across features

**Example:**

- Original: $x \in [1000, 5000]$ (e.g., house square footage). Then $x^2 \in [10^6, 25\times10^6]$ (huge range!).
- Standardized: $x_{\text{scaled}} \in [-2, 2]$ (roughly). Then $x_{\text{scaled}}^2 \in [0, 4]$ (manageable).

### Centering (Optional but Recommended)

Ensure standardization includes centering (setting mean to zero). This reduces **multicollinearity** between raw and polynomial terms:

- Without centering: $x$ and $x^2$ are correlated (both tend to grow together)
- With centering: correlation is reduced, improving coefficient stability

**Mathematical note:** Standardization (as defined above with $(x - \mu)/\sigma$) automatically includes centering.

## Model Selection and Evaluation

### Training vs. Validation Error

Plot both training and validation error curves as functions of polynomial degree to diagnose model quality:

$$
\text{Train Error}(d) \quad \text{vs.} \quad \text{Validation Error}(d), \qquad d = 1,2,\ldots,d_{\max}
$$

**Ideal scenario:**

- Both errors decrease initially (underfitting → good fit)
- Both errors plateau at similar level (sweet spot)
- Validation error stays close to training error (good generalization)

**Warning signs:**

- Training error low, validation error high and widening (overfitting) $\Rightarrow$ Reduce degree or increase regularization
- Both errors high (underfitting) $\Rightarrow$ Increase degree

**Visual pattern:**

- U-shaped validation curve: minimum at optimal degree

### K-Fold Cross-Validation (Detailed)

**Procedure:**

1. **Partition:** Split data randomly into $k$ disjoint folds (e.g., $k=5$)
2. **Iterate:** For each fold $i = 1, \ldots, k$:
   - Train on all folds except $i$
   - Validate on fold $i$
   - Record validation metric (e.g., MSE, $R^2$)
3. **Aggregate:** Compute average validation metric across all $k$ folds
4. **Result:** More robust estimate of generalization performance

**Why k-fold is better than single train-test split:**

- Uses more data for training in each fold
- Reduces variance of error estimates (averaging over multiple splits)
- Especially valuable when data is limited

**Common choice:** $k=5$ or $k=10$ (trade-off between stability and computational cost)

### Residual Analysis

**Definition:** Residuals are the differences between observed and predicted values:

$$
r_i = y_i - \hat{y}_i \quad \text{for each observation } i
$$

Residuals reveal whether the model has captured all signal; ideally they should be random noise.

**1. Residual Plot (Predicted vs. Residuals)**

Scatter plot of $\hat{y}_i$ (x-axis) vs. $r_i$ (y-axis).

**Ideal pattern:** Random scatter symmetrically around the horizontal line $r=0$; no trends.

**Warning signs:**

- **Funnel shape:** Residual variance increases with magnitude of predictions (heteroscedasticity) $\Rightarrow$ Consider weighted least squares or log transformation of $y$
- **Systematic curve:** Residuals are non-random; model misses a trend $\Rightarrow$ Increase polynomial degree or add features
- **Outliers:** A few residuals much larger in magnitude $\Rightarrow$ Investigate; may indicate data errors or need for robust regression

**2. Q-Q Plot (Normality Check)**

Plot quantiles of residuals vs. quantiles of a standard normal distribution.

**Ideal pattern:** Points lie approximately on the diagonal line $y = x$.

**Deviations indicate:**

- **S-shaped Q-Q plot:** Residuals have heavier tails than normal (leptokurtic) $\Rightarrow$ Model may underestimate uncertainty
- **Curved Q-Q plot:** Residuals are non-normal; OLS confidence intervals may be unreliable

**Note:** OLS provides consistent estimators even if residuals are non-normal; however, formal hypothesis tests assume normality.

**3. Heteroscedasticity (Non-Constant Variance)**

If $\text{Var}(r_i | \hat{y}_i)$ varies across predictions (visible as funnel in residual plot), ordinary OLS is inefficient.

**Solutions:**

- **Weighted least squares (WLS):** Assign lower weights to observations with higher residual variance
- **Log or power transformation:** Transform $y$ (e.g., $y' = \log(y)$) to stabilize variance

## Worked Example: Univariate Quadratic Fit

**Scenario:** Real estate dataset with 10 properties. We want to predict house price ($y$) from square footage ($x$). Initial scatter plot suggests a curved relationship (quadratic form).

**Data:**

| $x$ (sqft, 1000s)   | 1.0 | 1.5 | 2.0 | 2.5 | 3.0 | 3.5 | 4.0 | 4.5 | 5.0 | 5.5  |
| ------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---- |
| $y$ (price, $1000s) | 100 | 150 | 230 | 320 | 430 | 550 | 680 | 820 | 980 | 1150 |

**Observation:** Prices increase rapidly (price differences widen as size increases), suggesting quadratic growth $y \propto x^2$.

---

**Step 1: Create Polynomial Feature Matrix**

For degree $d=2$, create three columns: constant, linear, and quadratic terms:

$$
\mathbf{X}_{\text{poly}} = \begin{bmatrix}
1 & 1.0 & 1.0^2 = 1.0 \\
1 & 1.5 & 1.5^2 = 2.25 \\
1 & 2.0 & 2.0^2 = 4.0 \\
1 & 2.5 & 2.5^2 = 6.25 \\
1 & 3.0 & 3.0^2 = 9.0 \\
1 & 3.5 & 3.5^2 = 12.25 \\
1 & 4.0 & 4.0^2 = 16.0 \\
1 & 4.5 & 4.5^2 = 20.25 \\
1 & 5.0 & 5.0^2 = 25.0 \\
1 & 5.5 & 5.5^2 = 30.25
\end{bmatrix}
$$

Result: $10 \times 3$ matrix (10 samples, 3 features including bias).

**Step 2: Solve Normal Equation**

Compute OLS coefficients:

$$
\boldsymbol{\beta} = (\mathbf{X}_{\text{poly}}^T \mathbf{X}_{\text{poly}})^{-1} \mathbf{X}_{\text{poly}}^T \mathbf{y}
$$

_(Computational details omitted, but in practice use numpy/sklearn)_

Suppose the solution is:

$$
\boldsymbol{\beta} = \begin{bmatrix} -50 \\ 30 \\ 90 \end{bmatrix}
$$

Interpretation:

- $\beta_0 = -50$: baseline (intercept, has limited physical meaning here)
- $\beta_1 = 30$: linear effect of $x$ (additional ~$30k per 1000 sqft linearly)
- $\beta_2 = 90$: quadratic effect (nonlinear acceleration; price grows faster at larger sizes)

**Step 3: Write Fitted Model**

$$
\hat{y} = -50 + 30x + 90x^2
$$

**Step 4: Make Predictions**

**On training data** (check fit quality):
For $x = 2.5$:

$$
\hat{y} = -50 + 30(2.5) + 90(2.5)^2 = -50 + 75 + 90(6.25) = -50 + 75 + 562.5 = 587.5
$$

Actual value: $y = 320$. Residual: $r = 320 - 587.5 = -267.5$ (large residual; model underfits this point).

**On new data** (extrapolation example):
Predict price for $x = 3.2$ (outside training range, close to max):

$$
\begin{aligned}
\hat{y} 
&= -50 + 30(3.2) + 90(3.2)^2 \\
&= -50 + 96 + 90(10.24) \\
&= -50 + 96 + 921.6 \\
&= 967.6 \quad \text{(in \$1000s)}
\end{aligned}
$$

---

**Step 5: Goodness-of-Fit Assessment**

Compute $R^2$ (fraction of variance explained):

$$
\text{SS}_{\text{res}} = \sum_i (y_i - \hat{y}_i)^2, \quad \text{SS}_{\text{tot}} = \sum_i (y_i - \bar{y})^2
$$

_(Calculations depend on actual fitted values; let's say we get_ $R^2 = 0.92$_)_

Interpretation: **Model explains 92% of price variance**, which is quite good. The 8% unexplained is likely due to omitted factors (location, condition, etc.).

## Common Pitfalls and How to Avoid Them

### 1. Degree Selection Without Validation

**Problem:** Choosing degree $d$ based solely on training error $\text{MSE}_{\text{train}}$ leads to systematic overfitting. Higher degrees always fit training data better.

**Why it fails:**

$$
\text{MSE}_{\text{train}}(d_1) \geq \text{MSE}_{\text{train}}(d_2) \text{ if } d_1 < d_2 \quad \text{(always true, but misleading!)}
$$

This monotonic decrease doesn't reflect generalization to new data.

**Solution:** Use cross-validation or hold-out test set to select degree fairly. Choose degree based on validation error, not training error:

$$
d^* = \arg\min_d \text{MSE}_{\text{val}}(d)
$$

### 2. Multicollinearity Among Polynomial Terms

**Problem:** Raw polynomial features $x, x^2, x^3, \ldots$ are often highly correlated with each other. For example, if $x \in [1, 5]$, then $\text{Corr}(x, x^2)$ is often $> 0.9$.

**Why it's bad:**

- Inflates coefficient variance: $\text{Var}(\hat{\beta}_j) \propto \frac{1}{1 - R_j^2}$ (where $R_j^2$ is the correlation of feature $j$ with other features)
- Coefficients become unstable and hard to interpret
- Small changes in data lead to large changes in fitted coefficients
- Matrix $\mathbf{X}^T \mathbf{X}$ becomes ill-conditioned (near-singular), causing numerical errors

**Solutions:**

1. **Use orthogonal polynomials:** Chebyshev, Legendre, or Hermite polynomials are designed to be uncorrelated
2. **Apply regularization:** Ridge or Lasso shrinks coefficients, reducing variance inflation
3. **Standardize and center features:** Reduces correlation between $x$ and $x^2$
4. **Inspect correlation matrix:** Before fitting, compute $\text{Corr}(\mathbf{X}_{\text{poly}})$ to diagnose severity

### 3. Extrapolation Instability (Runge's Phenomenon)

**Problem:** High-degree polynomials fit training data well but oscillate **wildly** outside the training range, making predictions unreliable far from observed data.

**Mathematical root cause:** High-degree polynomials have **large derivatives** near the boundaries. Formally, a degree-$d$ polynomial can have up to $d-1$ inflection points, allowing extreme oscillations.

**Classic example (Runge's phenomenon):**

- Function: $f(x) = \frac{1}{1 + 25x^2}$ on $x \in [-1, 1]$
- Fit degree-9 polynomial to 10 equally-spaced samples
- Result: Perfect fit in $[-1, 1]$, but polynomial explodes to $\pm \infty$ near $x = \pm 1.3$

**Solutions:**

1. **Limit polynomial degree:** Use $d \leq 3$ or 4 for extrapolation
2. **Apply regularization:** Ridge/Lasso reduces coefficient magnitudes, dampening oscillations
3. **Use splines:** Piecewise polynomials (e.g., cubic splines) avoid global oscillations
4. **Restrict predictions:** Only predict within $[\min(X_{\text{train}}), \max(X_{\text{train}})]$
5. **Kernel methods:** Kernel ridge regression (e.g., with polynomial kernel) avoids these issues

### 4. Feature Scaling Neglect

**Problem:** Fitting unscaled features leads to two major issues:

1. **Numerical instability:** The matrix $\mathbf{X}^T \mathbf{X}$ has **condition number** $\kappa(\mathbf{X}^T \mathbf{X})$ that may be very large. Computing the inverse becomes numerically unreliable:

   $$
   \boldsymbol{\beta} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}
   $$

   Small errors in data due to floating-point rounding get amplified in the solution.

2. **Unfair regularization:** Regularization penalties don't account for scale. For example, Ridge penalty $\lambda \sum_j \beta_j^2$ will penalize large-scale features more than small-scale ones, leading to biased estimates.

**Solution:** Always standardize features before fitting:

$$
x_{\text{scaled}} = \frac{x - \mu}{\sigma} \quad \text{(apply to all base features before creating polynomial terms)}
$$

This ensures $\mathbf{X}^T \mathbf{X}$ is well-conditioned and regularization is fair.

### 5. Overfitting with Small Datasets

**Problem:** With few training samples $n$ but many polynomial features $p$ (especially for high degree $d$ on multiple inputs), the model has too much flexibility to fit noise rather than signal. Extreme case: $p > n$ (fewer observations than features) leads to underdetermined system with infinite solutions.

**Quantitative warning:** If $p/n > 0.1$ (e.g., 100 features, 1000 samples), watch for overfitting. If $p > n$, overfitting is guaranteed without strong regularization.

**Solutions:**

1. **Reduce polynomial degree:** Lower $d$ directly reduces number of features $p = \binom{m+d}{d}$
2. **Use regularization:** Ridge/Lasso shrinks coefficients, trading bias for lower variance
3. **Collect more data:** Every additional sample helps, especially when $n < p$
4. **Feature selection:** Use Lasso or manual selection to keep only important polynomial terms
5. **Use simpler models:** If linear regression or decision trees fit well, prefer those over high-degree polynomials

## Implementation in Python (scikit-learn)

```python
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import cross_val_score, train_test_split
import matplotlib.pyplot as plt

# Sample data
X = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5]).reshape(-1, 1)
y = np.array([100, 150, 230, 320, 430, 550, 680, 820, 980, 1150])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create polynomial features (degree 2)
poly = PolynomialFeatures(degree=2, include_bias=True)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

# Fit linear regression on polynomial features
model = LinearRegression()
model.fit(X_train_poly, y_train)

# Evaluate
train_score = model.score(X_train_poly, y_train)
test_score = model.score(X_test_poly, y_test)
print(f"Train R²: {train_score:.4f}, Test R²: {test_score:.4f}")

# Cross-validation for degree selection
degrees = range(1, 6)
cv_scores = []
for d in degrees:
    poly_d = PolynomialFeatures(degree=d)
    X_poly = poly_d.fit_transform(X_train_scaled)
    model_d = LinearRegression()
    scores = cross_val_score(model_d, X_poly, y_train, cv=5, scoring='r2')
    cv_scores.append(scores.mean())
    print(f"Degree {d}: CV R² = {scores.mean():.4f} (±{scores.std():.4f})")

# Plot results
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(degrees, cv_scores, 'o-', label='CV Score (R²)')
plt.xlabel('Polynomial Degree')
plt.ylabel('Cross-Validation R²')
plt.legend()
plt.grid()

# Predictions on test set
y_pred = model.predict(X_test_poly)
plt.subplot(1, 2, 2)
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.scatter(X_test, y_pred, color='red', label='Predicted', marker='x')
plt.xlabel('Feature (scaled)')
plt.ylabel('Target')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
```

## Advanced Topics

### Orthogonal Polynomials

Raw polynomial features are highly correlated. Orthogonal polynomials (Chebyshev, Legendre, Hermite) are designed to be uncorrelated and improve numerical stability.

```python
# Using sklearn's built-in orthogonal polynomials via robust preprocessing
from sklearn.preprocessing import PolynomialFeatures

# Orthogonal polynomial regression can be achieved by preprocessing
poly = PolynomialFeatures(degree=3, include_bias=True)
```

### Spline Regression

An alternative to global polynomials is **spline regression**: piecewise polynomials with smooth continuity at knots. Splines avoid Runge's phenomenon and allow localized flexibility.

```python
from scipy.interpolate import UnivariateSpline

# Fit a spline with smoothing parameter s
spline = UnivariateSpline(X.flatten(), y, s=1000)
y_pred = spline(X_test.flatten())
```

### Kernel Ridge Regression

Instead of explicit polynomial features, use kernel methods to implicitly work in high-dimensional polynomial feature spaces while avoiding the curse of dimensionality.

```python
from sklearn.kernel_ridge import KernelRidge

# polynomial kernel ridge regression
kr = KernelRidge(kernel='poly', degree=2, alpha=0.1)
kr.fit(X_train, y_train)
y_pred = kr.predict(X_test)
```

## Comparison with Alternatives

| Method                                  | Interpretability | Computational Cost | Overfitting Risk                  | Extrapolation                    |
| --------------------------------------- | ---------------- | ------------------ | --------------------------------- | -------------------------------- |
| **Linear Regression**                   | Very High        | Very Low           | Low                               | Stable                           |
| **Polynomial Regression (low degree)**  | High             | Low                | Low-Medium                        | Moderate                         |
| **Polynomial Regression (high degree)** | Medium           | Low                | Very High                         | Unstable                         |
| **Spline Regression**                   | Medium           | Low-Medium         | Medium                            | Better than high-degree poly     |
| **Kernel Ridge Regression**             | Low              | Medium             | Medium                            | Data-dependent                   |
| **Neural Networks**                     | Very Low         | High               | Medium-High (with regularization) | Data-dependent                   |
| **Random Forests / Grad. Boosting**     | Low              | Medium-High        | Medium (with tuning)              | Stable but limited extrapolation |

## Summary

**Polynomial regression is ideal for:**

- Curved, non-linear patterns in low-dimensional data.
- Interpretable models requiring an explicit mathematical formula.
- Situations where domain knowledge suggests polynomial relationships.

**To use polynomial regression effectively:**

1. Visualize data; check if polynomial relationship is plausible.
2. Standardize features to avoid scale issues and improve numerical stability.
3. Select polynomial degree via cross-validation; avoid overfitting.
4. Apply regularization (Ridge/Lasso) for high-degree polynomials.
5. Analyze residuals to validate model assumptions.
6. Limit predictions to the training data range to avoid extrapolation instability.
7. For high-dimensional data or severe overfitting, consider alternatives (splines, kernel methods, tree-based models).

Polynomial regression bridges the explainability of linear models with the flexibility to capture non-linear relationships, making it a valuable tool in any machine learning engineer's toolkit—when used judiciously.

---

## Note on GitHub Rendering

This document uses LaTeX math notation (`$...$` for inline, `$$...$$` for display equations).

**GitHub Support:** GitHub natively renders LaTeX in markdown files. All mathematical equations in this guide will display correctly when viewed on GitHub.com.

**Local Viewing:**

- **VS Code:** Install the "Markdown Preview Enhanced" or "Math" extension for preview rendering
- **Raw Text:** If viewing raw markdown, equations will appear as LaTeX source code but are correct

**Browsers:**

- **Chrome/Firefox:** GitHub renders math server-side; no additional plugins needed
- **Safari:** Should work; if not, try refreshing the page
