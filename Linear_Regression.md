# Part I: Linear Regression

This document summarizes key ideas from the linear regression portion of the course. It takes a housing dataset as a running example, then generalizes to the theory and algorithms behind least‑squares regression.

---

## 1. Problem setup and hypothesis representation

- **Training set**: $(x^{(i)}, y^{(i)})$ for $i=1,\ldots,n$. Each input vector $x^{(i)}$ may have multiple components (features). In the housing example we use living area and number of bedrooms.
- **Features**: choosing the components of $x$ is part of the design of a learning problem. Additional features (bathrooms, fireplace, sale date, ...) can be added later. Good feature selection is essential to avoid under/over‑fitting.

### Linear hypothesis

We approximate the target value as a linear function of the features:

\[ h\_{\theta}(x) = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \cdots + \theta_d x_d. \]

To simplify notation introduce the intercept coordinate \(x_0 = 1\) and write

\[ h\_{\theta}(x) = \theta^{\mathrm T} x. \]

The vector \(\theta\in\mathbb R^{d+1}\) collects the parameters (weights).

---

## 2. Cost function and learning

We pick parameters by making predictions close to the training targets. The **least-squares cost** is

\[ J(\theta)=\frac{1}{2n} \sum*{i=1}^n \bigl(h*{\theta}(x^{(i)})-y^{(i)}\bigr)^2. \]

This is a convex quadratic in \(\theta\).

### Gradient descent (LMS/Widrow–Hoff rule)

Initialize \(\theta\) and repeat:

\[ \theta_j := \theta_j - \alpha \frac{\partial}{\partial\theta_j} J(\theta). \]

with learning rate \(\alpha>0\). For a single example the update is

\[ \theta*j := \theta_j + \alpha\,(y^{(i)} - h*{\theta}(x^{(i)}))\, x^{(i)}\_j. \]

and for a batch of size \(n\) the vectorized update is

\[ \theta := \theta + \frac{\alpha}{n} X^{T}(\vec y - X\theta). \]

- **Batch gradient descent**: uses all examples per step. Converges to global minimum since \(J\) is convex.
- **Stochastic gradient descent**: update on each example in sequence. Faster per‑step progress and scalable to large datasets; may oscillate around the optimum unless \(\alpha\) decays.

### Closed‑form solution (normal equations)

Define the design matrix \(X\in\mathbb R^{n\times(d+1)}\) with rows \((x^{(i)})^{\mathrm T}\) and the target vector \(\vec y\). Writing the cost as

\[ J(\theta)=\frac{1}{2n} (X\theta-\vec y)^{\mathrm T}(X\theta-\vec y) \]

setting the gradient to zero yields the **normal equations**

\[ X^{\mathrm T} X \theta = X^{\mathrm T} \vec y \]

and hence

\[ \theta = (X^{\mathrm T} X)^{-1} X^{\mathrm T} \vec y. \]

(Assumes \(X^{\mathrm T}X\) invertible; if not, use regularization or pseudo‑inverse.)

---

## 3. Probabilistic interpretation

Assume

\[ y^{(i)} = \theta^{\mathrm T} x^{(i)} + \epsilon^{(i)}. \]

with noise terms \(\epsilon^{(i)}\) i.i.d. \(\mathcal N(0,\sigma^2)\). The likelihood of the data is

\[ L(\theta) = \prod\_{i=1}^n \frac{1}{\sqrt{2\pi\sigma^2}}\exp\left(-\frac{(y^{(i)}-\theta^{\mathrm T}x^{(i)})^2}{2\sigma^2}\right) \]

Maximizing the (log) likelihood is equivalent to minimizing the least-squares cost; hence ordinary linear regression is the maximum‑likelihood estimator under Gaussian noise.

---

## 4. Model evaluation and goodness of fit

### Coefficient of determination \(R^2\)

Measures fraction of variance explained by the model:

\[ R^2 = 1 - \frac{\sum*{i}(y^{(i)}-\hat y^{(i)})^2}{\sum*{i}(y^{(i)}-\bar y)^2}. \]

where \(\hat y^{(i)}=h\_{\theta}(x^{(i)})\) and \(\bar y\) is the sample mean. \(R^2=1\) for perfect fit, 0 for the constant-mean model; can be negative for bad fits.

### Adjusted \(R^2\)

Penalizes adding irrelevant features:

\[ R^2\_{\text{adj}} = 1 - (1-R^2)\frac{n-1}{n-d-1}. \]

where \(d\) is number of predictors (excluding intercept) and \(n\) number of examples. Used to compare models with different numbers of features.

### Overfitting and underfitting

- **Underfitting**: model too simple (high bias); training error large and algorithm misses structure.
- **Overfitting**: model too complex (high variance); fits noise in training set, poor generalization. e.g., high‑degree polynomial passing through all points.
- Regularization (e.g. ridge, lasso) can mitigate overfitting by penalizing large \(\theta\). Cross‑validation selects hyperparameters.

---

## 5. Locally weighted linear regression (non‑parametric)

Instead of solving a single global fit, LWR computes parameters at each query point by weighting nearby examples:

1. For query \(x\), assign weight
   \[ w^{(i)} = \exp\left(-\frac{\|x^{(i)}-x\|^2}{2\tau^2}\right) \]
2. Fit $\theta$ by minimizing $\sum_{i} w^{(i)} (y^{(i)}-\theta^{\mathrm T}x^{(i)})^2$.

Prediction is \(h(x)=\theta^{\mathrm T}x\). Requires storing full training set; bandwidth \(\tau\) controls locality. As \(\tau\to\infty\) recovers global linear regression.

---

## 6. Key points for students

1. **Understand linear algebra notation**: vectors, matrices, gradients.
2. **Gradient descent vs. normal equations**: iterative versus closed-form.
3. **Feature selection matters**: the choice and scaling of features dictate performance.
4. **Scaling and normalization**: scale features to similar ranges for gradient-based methods.
5. **Evaluation metrics**: use train/test split, cross‑validation, \(R^2\), adjusted \(R^2\), and error measures (MSE, MAE).
6. **Regularization**: introduces bias but reduces variance; essential when \(d\) is large or \(X^{\mathrm T}X\) ill-conditioned.
7. **Assumptions**: normal noise, linear relationship; understand when they fail.
8. **Non-parametric alternatives**: LWR, k‑nearest neighbors, etc., when linear model is inadequate.

---

This summary should give you a compact reference to the main ideas behind linear regression. Experiment with the algorithms on real data and plot results to build intuition.
