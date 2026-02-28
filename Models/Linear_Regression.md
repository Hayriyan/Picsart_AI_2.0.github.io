# CS229 Notes — Part I: Linear Regression

## Linear Regression

To make our housing example more interesting, let’s consider a slightly richer dataset in which we also know the number of bedrooms in each house.

Here, the $x$’s are two-dimensional vectors in $\mathbb{R}^2$.

- $x_1^{(i)}$ — living area of the $i$-th house  
- $x_2^{(i)}$ — number of bedrooms  

When designing a learning problem, you decide which features to use.

---

## Hypothesis Representation

We approximate $y$ as a linear function of $x$:

$$
h_\theta(x) = \theta_0 + \theta_1 x_1 + \theta_2 x_2
$$

The parameters $\theta_i$ are called **weights**.

We introduce:

$$
x_0 = 1
$$

so we can write:

$$
h(x) = \sum_{i=0}^{d} \theta_i x_i = \theta^T x
$$

where:

- $\theta, x$ are vectors
- $d$ is number of features (excluding $x_0$)

---

## Cost Function

We want predictions close to targets.

Define the **least squares cost**:

$$
J(\theta) = \frac{1}{2}\sum_{i=1}^{n} (h_\theta(x^{(i)}) - y^{(i)})^2
$$

This is the **ordinary least squares objective**.

---

# 1. LMS Algorithm (Gradient Descent)

We minimize $J(\theta)$ using gradient descent.

Update rule:

$$
\theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j} J(\theta)
$$

where:

- $\alpha$ = learning rate
- update performed for all $j=0,\dots,d$

---



### Derivative of the Cost Function (Single Training Example)

For one training example $(x,y)$:

$$
J(\theta) = \frac{1}{2}(h_\theta(x)-y)^2
$$

Compute derivative:

$$
\frac{\partial}{\partial \theta_j} J(\theta)
=
\frac{\partial}{\partial \theta_j}
\frac{1}{2}(h_\theta(x)-y)^2
$$

Apply chain rule:

$$
=
2 \cdot \frac{1}{2}(h_\theta(x)-y)
\cdot
\frac{\partial}{\partial \theta_j}(h_\theta(x)-y)
$$

$$
=
(h_\theta(x)-y)
\frac{\partial}{\partial \theta_j}
\left(\sum_{i=0}^{d}\theta_i x_i - y\right)
$$

$$
=
(h_\theta(x)-y)x_j
$$


## LMS Update Rule

$$
\theta_j := \theta_j + \alpha (y^{(i)} - h_\theta(x^{(i)}))x_j^{(i)}
$$

Also called:

- Least Mean Squares (LMS)
- Widrow–Hoff rule

Update size ∝ prediction error.

---

## Batch Gradient Descent

Repeat until convergence:

$$
\theta_j := \theta_j + \alpha \sum_{i=1}^{n}
(y^{(i)} - h_\theta(x^{(i)}))x_j^{(i)}
$$

Vector form:

$$
\theta := \theta + \alpha \sum_{i=1}^{n}
(y^{(i)} - h_\theta(x^{(i)}))x^{(i)}
$$

Properties:

- uses full dataset each step
- always converges for linear regression
- $J(\theta)$ is convex quadratic

---

## Stochastic Gradient Descent (SGD)

Loop through training examples:

$$
\theta_j := \theta_j + \alpha (y^{(i)} - h_\theta(x^{(i)}))x_j^{(i)}
$$

Characteristics:

- faster progress for large datasets
- oscillates near minimum
- often preferred in practice

---

# 2. Normal Equations

Instead of iteration, minimize $J$ analytically.

---

## Matrix Setup

### Design matrix

$$
X =
\begin{bmatrix}
(x^{(1)})^T \\
(x^{(2)})^T \\
\vdots \\
(x^{(n)})^T
\end{bmatrix}
$$

### Target vector

$$
\vec{y} =
\begin{bmatrix}
y^{(1)} \\
y^{(2)} \\
\vdots \\
y^{(n)}
\end{bmatrix}
$$

Prediction error vector:

$$
X\theta - \vec{y}
$$

Cost:

$$
J(\theta)
=
\frac{1}{2}(X\theta-\vec{y})^T(X\theta-\vec{y})
$$

---
## Gradient of Least Squares Cost

We write cost in matrix form:

$$
J(\theta)
=
\frac{1}{2}(X\theta-\vec{y})^T(X\theta-\vec{y})
$$

---

### Step-by-step derivation

$$
\nabla_\theta J(\theta)
=
\nabla_\theta
\frac{1}{2}(X\theta-\vec{y})^T(X\theta-\vec{y})
$$

Expand quadratic:

$$
=
\frac{1}{2}\nabla_\theta
\left(
(X\theta)^TX\theta
-
(X\theta)^T\vec{y}
-
\vec{y}^T(X\theta)
+
\vec{y}^T\vec{y}
\right)
$$

Rewrite using matrix identities:

$$
=
\frac{1}{2}\nabla_\theta
\left(
\theta^T(X^TX)\theta
-
\vec{y}^T(X\theta)
-
\vec{y}^T(X\theta)
\right)
$$

$$
=
\frac{1}{2}\nabla_\theta
\left(
\theta^T(X^TX)\theta
-
2(X^T\vec{y})^T\theta
\right)
$$

Take derivatives:

$$
=
\frac{1}{2}
\left(
2X^TX\theta
-
2X^T\vec{y}
\right)
$$

Final result:

$$
\nabla_\theta J(\theta)
=
X^TX\theta - X^T\vec{y}
$$
---

## Normal Equation Solution

$$
\theta = (X^T X)^{-1} X^T \vec{y}
$$

Requires $X^TX$ to be invertible.

---

# 3. Probabilistic Interpretation

Assume:

$$
y^{(i)} = \theta^T x^{(i)} + \epsilon^{(i)}
$$

with noise:

$$
\epsilon^{(i)} \sim \mathcal{N}(0,\sigma^2)
$$

---
## Log Likelihood Derivation

Likelihood:

$$
L(\theta)
=
\prod_{i=1}^{n}
\frac{1}{\sqrt{2\pi\sigma}}
\exp
\left(
-\frac{(y^{(i)}-\theta^Tx^{(i)})^2}{2\sigma^2}
\right)
$$

---

### Take log

$$
\ell(\theta)=\log L(\theta)
$$

$$
=
\log
\prod_{i=1}^{n}
\frac{1}{\sqrt{2\pi\sigma}}
\exp
\left(
-\frac{(y^{(i)}-\theta^Tx^{(i)})^2}{2\sigma^2}
\right)
$$

Convert product → sum:

$$
=
\sum_{i=1}^{n}
\log
\left(
\frac{1}{\sqrt{2\pi\sigma}}
\exp
\left(
-\frac{(y^{(i)}-\theta^Tx^{(i)})^2}{2\sigma^2}
\right)
\right)
$$

$$
=
n\log\frac{1}{\sqrt{2\pi\sigma}}
-
\frac{1}{2\sigma^2}
\sum_{i=1}^{n}
(y^{(i)}-\theta^Tx^{(i)})^2
$$

---

### Optimization Result

Maximizing $\ell(\theta)$ is equivalent to minimizing:

$$
\frac{1}{2}\sum_{i=1}^{n}(y^{(i)}-\theta^Tx^{(i)})^2
$$

which is the least squares cost.

---

# 4. Locally Weighted Linear Regression

Standard linear regression:

1. Fit $\theta$
2. Output $\theta^T x$

---

## Weighted Version

Minimize:

$$
\sum_i w^{(i)}(y^{(i)}-\theta^Tx^{(i)})^2
$$

Typical weights:

$$
w^{(i)} =
\exp\left(
-\frac{(x^{(i)}-x)^2}{2\tau^2}
\right)
$$

- nearby points → high weight
- distant points → low weight
- $\tau$ = bandwidth

---

## Parametric vs Non-parametric

### Parametric (Linear Regression)
- fixed number of parameters
- no need to store data after training

### Non-parametric (LWR)
- must store training data
- complexity grows with dataset size












