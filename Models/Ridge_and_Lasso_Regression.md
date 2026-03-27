# Ridge and Lasso Regression — Shrinkage Methods 📉

## Overview

Ridge and Lasso are two of the most popular **shrinkage (regularization) methods** for linear regression. They add a penalty on the size of the coefficients to

- reduce variance when predictors are correlated,
- improve prediction accuracy, and
- (in the case of Lasso) perform variable selection.

These techniques sit between full least-squares and subset selection: instead of keeping or discarding a variable outright, they **continuously shrink** the coefficients
and therefore produce more stable models.

## Why Shrinkage? Motivation

- Ordinary least squares (OLS) may overfit when predictors are numerous or highly correlated.
- Subset selection (e.g. forward/backward selection) is discrete and can have high variance – a small change in the data can flip which variables are chosen.
- Shrinkage methods apply a penalty that encourages smaller coefficients, which stabilizes the estimates and can reduce test error even when all variables are retained.

> **Key idea:** rather than eliminating predictors, pull their coefficients toward zero. The strength of the pull is controlled by a tuning parameter.

## Ridge Regression

### Formulation

Ridge regression solves

$$
\hat{\beta}_{\text{ridge}} = \arg\min_{\beta} \sum_{i=1}^N (y_i - \beta_0 - \sum_{j=1}^p x_{ij}\beta_j)^2 + \lambda \sum_{j=1}^p \beta_j^2,
$$

with penalty parameter $\lambda \ge 0$. Equivalently,

$$
\hat{\beta}_{\text{ridge}} = \arg\min_{\beta} \sum_{i=1}^N (y_i - \beta_0 - x_i^T \beta)^2 \quad\text{subject to } \sum_{j=1}^p \beta_j^2 \le t
$$

where $t$ and $\lambda$ correspond one–to–one. The larger $\lambda$, the stronger the shrinkage; $\lambda=0$ recovers OLS.

> **Important:** the intercept $\beta_0$ is not penalized – the data are usually centered and inputs standardized before fitting.

### Closed‑Form Solution

After centering and standardizing, let $X$ be the $N\times p$ data matrix. Then

$$
\hat{\beta}_{\text{ridge}} = (X^TX + \lambda I)^{-1} X^T y.
$$

Adding $\lambda I$ to $X^TX$ ensures the matrix is invertible and shrinks the eigenvalues. When $X^TX$ is singular (e.g., $p > N$) ridge still gives a unique solution.

### Effect on the Coefficients

- If the predictors are orthonormal, $\hat{\beta}_{\text{ridge}} = \hat{\beta}_{\text{OLS}}/(1+\lambda)$ – all coefficients are scaled down uniformly.
- In general, the shrinkage is larger in directions of smaller variance (see SVD/principal component view below).

### SVD and Degrees of Freedom

Using the SVD $X=UDV^T$ with singular values $d_1\ge \dots\ge d_p$, the fitted values are

$$
X\hat{\beta}_{\text{ridge}} = \sum_{j=1}^p u_j \frac{d_j^2}{d_j^2 + \lambda} u_j^T y.
$$

The **effective degrees of freedom** of the ridge fit is

$$
\mathrm{df}(\lambda) = \operatorname{tr}(X(X^TX+\lambda I)^{-1}X^T) = \sum_{j=1}^p \frac{d_j^2}{d_j^2 + \lambda}.
$$

This quantity decreases from $p$ (when $\lambda=0$) toward 0 as $\lambda\to \infty$.

### Bayesian Interpretation

Ridge regression is equivalent to the posterior mode (and mean) under a Gaussian prior
$\beta_j\sim N(0,\tau^2)$ with $\lambda=\sigma^2/\tau^2$ when the errors are $N(0,\sigma^2)$.

### Practical Notes

- Always standardize predictors before fitting (unit variance).
- Use cross‑validation to choose $\lambda$; ten‑fold CV is common.
- The solution path is smooth; coefficient plots vs. $\log\lambda$ reveal how features shrink.
- Ridge does **not** set coefficients exactly to zero; all predictors remain in the model.

### Simple Example

Suppose two correlated features $x_1, x_2$ and response $y$. OLS may give large but canceling coefficients. With ridge and $\lambda=1$, both coefficients move toward zero, stabilizing predictions.

## Lasso Regression

### Formulation

Lasso solves a similar problem but with an $\ell_1$ penalty:

$$
\hat{\beta}_{\text{lasso}} = \arg\min_{\beta} \sum_{i=1}^N (y_i - \beta_0 - x_i^T\beta)^2 + \lambda \sum_{j=1}^p |\beta_j|.
$$

The $\ell_1$ norm encourages sparsity: as $\lambda$ grows some coefficients become exactly zero, performing variable selection.

### Geometry and Behavior

- The constraint region $\{\beta: \sum |\beta_j| \le t\}$ is a diamond (cross-polytope); the least-squares contours are ellipses. The corners of the diamond align with coordinate axes, so the optimal solution often lands on an axis → zero coefficient.
- The Lasso path is piecewise linear in $\lambda$; efficient algorithms (LARS) exploit this.

### Comparison to Ridge

| Feature            | Ridge                        | Lasso                           |
| ------------------ | ---------------------------- | ------------------------------- |
| Penalty            | $\ell_2$                     | $\ell_1$                        |
| Shrinkage          | Continuous, all coeffs small | Some coeffs zero, others shrunk |
| Variable selection | No                           | Yes (automatic)                 |
| Solution path      | Smooth                       | Piecewise linear                |
| Useful when        | Many correlated predictors   | Expect few relevant predictors  |

### Choosing $\lambda$

Use cross‑validation, information criteria (AIC/BIC) with degrees of freedom approximations, or stability selection. The “one‑standard‑error” rule often picks a sparser model.

### Computational Tools

- `sklearn.linear_model.Ridge`, `RidgeCV` (with built‑in CV).
- `sklearn.linear_model.Lasso`, `LassoCV`, `LassoLarsCV`.
- In R: `glmnet` package handles both simultaneously with elastic net.

## Worked Numeric Example (Prostate Data)

The book excerpt contains coefficients and test errors for different methods (Table 3.3). Ridge with tuned $\lambda$ and Lasso both improved over OLS; Lasso produced a sparse model and the lowest test error.

> **Exercise:** Using any dataset (e.g. Boston housing), fit OLS, ridge, and lasso. Plot coefficient paths vs.
> $\log(\lambda)$, compute CV errors, and compare the number of non-zero coefficients. Discuss what happens when predictors are highly correlated.

## Book Example: Prostate Cancer Data

The excerpt from the book (Section 3.4) provides a concrete demonstration using prostate cancer data:

- **Prediction error curves** (Figure 3.7) compare subset selection, ridge, lasso, principal components regression, and partial least squares as the complexity parameter varies. The horizontal axis is scaled so model complexity increases left‑to‑right; ridge and lasso show smoother curves than subset methods.
- The **chosen model** is the least complex one within one standard error of the best (purple vertical line). For ridge this corresponded to an effective degrees of freedom df(\lambda)=5.0.
- **Table 3.3** reports estimated coefficients and test errors. Ridge and lasso both lowered test error relative to full least squares; lasso produced a sparse model and achieved the smallest test error (0.479 vs 0.492 for ridge).

Key takeaways:

- Shrinkage methods can match or beat the predictive performance of subset selection.
- Ridge tends to keep all variables with smaller coefficients, while lasso discards some (see blank entries in the table).
- The effective degrees of freedom formula (eqn 3.50) provides a interpretable complexity measure that aligns with classical notions of model flexibility.

> **Exercise:** Reproduce the book's analysis by fitting each method on a dataset, plotting CV error vs complexity, and selecting the model using the one‑standard‑error rule. Compare to the values shown in the excerpt.

### When to Use Which

1. **Ridge** when you have many predictors that you believe all have some predictive power and multicollinearity is an issue.
2. **Lasso** when you suspect only a subset of predictors are truly relevant; it also performs feature selection automatically.
3. **Elastic Net** (combination of $\ell_1$ and $\ell_2$ penalties) when predictors are correlated and you want both selection and grouping.

## Practical Usage Cases

- **High‑dimensional data (p ≫ n)**: Ridge is a go‑to when you have more features than samples (e.g., genomics, text embeddings). Lasso can be used when you expect a sparse signal.
- **Multicollinearity**: When predictors are highly correlated (e.g., economic indicators, sensor readings), ridge guards against wildly varying OLS coefficients.
- **Feature selection for interpretability**: Use lasso to automatically identify a small set of predictors (medical risk scores, simplified scoring models).
- **Preprocessing pipelines**: Regularized regression is a stable choice in ML pipelines (e.g., with scaling + CV), and is often used as a baseline model in competitions and prototyping.
- **Model compression / deployment**: Lasso can reduce the number of nonzero weights, making the model cheaper to store and faster to apply.

## Implementation Tips

- Always preprocess: center response and standardize features. Use the same transformation on new data.
- Wrap regularization inside a pipeline so that cross‑validation does not leak information.
- Interpret coefficients cautiously: penalization biases them toward zero; use unpenalized refit on selected features if unbiased estimates are needed.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LassoCV

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("reg", RidgeCV(alphas=[0.1, 1, 10]))
])
pipe.fit(X_train, y_train)

# lasso
lasso = LassoCV(cv=5, alphas=None, max_iter=10000)
lasso_pipe = Pipeline([("scaler", StandardScaler()), ("lasso", lasso)])
lasso_pipe.fit(X_train, y_train)
```

## Visualization and Diagnostics

- Plot coefficient paths versus $\lambda$ or degrees of freedom.
- Trace $\mathrm{df}(\lambda)$ and CV error to pick a model.
- Examine residuals and check for continued over/under‑shrinkage.

## Bias–Variance Tradeoff (Why Shrinkage Helps)

Regularization is a way of managing the bias–variance tradeoff:

- **High variance** models (e.g., OLS with many correlated predictors) can fit noise in the training data; their coefficients can swing widely when the data change.
- **Shrinkage** increases bias (coefficients are pulled toward zero), but reduces variance (the model changes less with new data).

In practice:

- **Ridge** shifts all coefficients toward zero smoothly, decreasing variance with a modest increase in bias.
- **Lasso** does the same but can also set some coefficients exactly to zero, which can substantially reduce variance if the true signal is sparse.

The optimal amount of shrinkage is found by trading off these effects (e.g., via cross‑validation): too little shrinkage leaves high variance; too much shrinkage oversimplifies and increases bias.

## Summary

Shrinkage methods provide a principled way to trade bias for variance.

- **Ridge**: stabilizes coefficients, useful with multicollinearity, but keeps all predictors.
- **Lasso**: adds sparsity, facilitating interpretation and selection.
- Both outperform OLS on data where overfitting is a concern, and are essential tools in the modern statistician's toolbox.

---

_Students should experiment with different data sets and penalty strengths to build intuition about how shrinkage affects prediction and interpretation._
