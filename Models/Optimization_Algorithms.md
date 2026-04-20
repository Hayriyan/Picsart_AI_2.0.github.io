# Optimization Algorithms for Deep Learning

## Table of Contents

1. [Convexity](#convexity)
2. [Gradient Descent](#gradient-descent)
3. [Stochastic Gradient Descent (SGD)](#stochastic-gradient-descent)
4. [Minibatch Stochastic Gradient Descent](#minibatch-stochastic-gradient-descent)
5. [Momentum](#momentum)
6. [Nesterov Accelerated Gradient](#nesterov-accelerated-gradient)
7. [Adagrad](#adagrad)
8. [RMSProp](#rmsprop)
9. [Adadelta](#adadelta)
10. [Adam](#adam)
11. [Rprop](#rprop-resilient-backpropagation)

---

## Convexity

### Definition

A set $C$ is **convex** if for any two points $\mathbf{x}, \mathbf{y} \in C$, the line segment between them is also contained in $C$. Mathematically:

$$\lambda \mathbf{x} + (1 - \lambda) \mathbf{y} \in C \quad \text{for all } \lambda \in [0, 1]$$

A function $f$ is **convex** if its domain is a convex set and for all points in its domain:

$$f(\lambda \mathbf{x} + (1 - \lambda) \mathbf{y}) \leq \lambda f(\mathbf{x}) + (1 - \lambda) f(\mathbf{y})$$

### Properties

- **Local minima are global minima**: In convex optimization, any local minimum is also a global minimum
- **Unique global minimum**: Strictly convex functions have at most one global minimum
- **Easier optimization**: Convex problems are generally much easier to solve than non-convex problems
- **Deep neural networks are non-convex**: Training neural networks involves non-convex optimization, which is more challenging but also more flexible

### Jensen's Inequality

An important property of convex functions:

$$f(E[\mathbf{X}]) \leq E[f(\mathbf{X})]$$

This inequality is fundamental in optimization theory and machine learning.

---

## Gradient Descent

### Overview

Gradient descent is the foundational optimization algorithm used to minimize an objective function $J(\theta)$ by updating parameters $\theta$ in the opposite direction of the gradient:

$$\theta = \theta - \eta \cdot \nabla_\theta J(\theta)$$

where:

- $\eta$ is the **learning rate** (step size)
- $\nabla_\theta J(\theta)$ is the gradient of the loss function

### How It Works

1. Start with initial parameters $\theta_0$
2. Compute the gradient $\nabla_\theta J(\theta)$
3. Update parameters: $\theta := \theta - \eta \nabla_\theta J(\theta)$
4. Repeat until convergence

### Key Characteristics

- **Guaranteed convergence to global minimum** for convex error surfaces
- **Convergence to local minimum** for non-convex surfaces (like neural networks)
- Simple and intuitive approach

### Code Example

```python
for i in range(nb_epochs):
    params_grad = evaluate_gradient(loss_function, data, params)
    params = params - learning_rate * params_grad
```

---

## Stochastic Gradient Descent

### Definition

Stochastic Gradient Descent (SGD) performs a parameter update for each training example $(x^{(i)}, y^{(i)})$:

$$\theta = \theta - \eta \cdot \nabla_\theta J(\theta; x^{(i)}; y^{(i)})$$

### Advantages

- **Faster updates**: One update per example instead of all data
- **Can learn online**: Adapt to new examples on-the-fly
- **Lower memory requirements**: Only need one example at a time
- **Better generalization**: Noise in updates can help escape local minima

### Disadvantages

- **High variance**: Frequent updates cause the loss function to fluctuate significantly
- **Slower convergence**: Takes more iterations to converge despite being faster per iteration
- **Difficulty converging**: Overshooting becomes problematic near the minimum

### Code Example

```python
for i in range(nb_epochs):
    np.random.shuffle(data)
    for example in data:
        params_grad = evaluate_gradient(loss_function, example, params)
        params = params - learning_rate * params_grad
```

### Convergence

When learning rate is properly annealed (gradually decreased), SGD converges to:

- Global minimum for convex optimization
- Local minimum for non-convex optimization

---

## Minibatch Stochastic Gradient Descent

### Definition

Minibatch SGD performs an update for every minibatch of $n$ training examples:

$$\theta = \theta - \eta \cdot \nabla_\theta J(\theta; x^{(i:i+n)}; y^{(i:i+n)})$$

### Advantages (Best of Both Worlds)

- **Reduced variance**: More stable parameter updates than pure SGD
- **Computational efficiency**: Leverages optimized matrix operations in modern libraries
- **Practical efficiency**: Common minibatch sizes (50-256) balance computation and stability
- **Industry standard**: Most practical deep learning uses minibatch SGD

### Code Example

```python
for i in range(nb_epochs):
    np.random.shuffle(data)
    for batch in get_batches(data, batch_size=50):
        params_grad = evaluate_gradient(loss_function, batch, params)
        params = params - learning_rate * params_grad
```

### Minibatch Size Selection

- **Small batches** (16-32): Noisier gradients, better generalization, slower per-epoch training
- **Medium batches** (32-256): Balance between noise and stability
- **Large batches** (256+): Cleaner gradients, faster computation, may generalize worse

---

## Key Challenges in Training

Before discussing advanced algorithms, let's identify the main challenges:

### Challenge 1: Learning Rate Selection

- **Too small**: Painfully slow convergence
- **Too large**: Loss oscillates around minimum or diverges
- **Fixed rate**: Cannot adapt to different features or training stages

### Challenge 2: Learning Rate Scheduling

- Schedules must be defined in advance
- Cannot adapt to dataset characteristics automatically
- Different features may need different learning rates

### Challenge 3: Non-uniform Feature Importance

- Sparse features occur infrequently, need larger updates
- Dense features occur frequently, need smaller updates
- Single learning rate treats all parameters equally

### Challenge 4: Non-convex Landscape

- **Local minima**: Suboptimal solutions that trap optimization
- **Saddle points**: More critical than local minima in high dimensions
  - Surrounded by flat plateaus
  - Zero gradients in all directions
  - Hard for SGD to escape

---

## Momentum

### Intuition

Standard gradient descent is like a ball rolling down a hill without inertia—it follows the slope but doesn't build speed. **Momentum** is like adding inertia: the ball accumulates speed as it rolls downhill.

### How It Works

Momentum adds a fraction $\gamma$ of the previous update vector to the current update:

$$v_t = \gamma v_{t-1} + \eta \nabla_\theta J(\theta)$$
$$\theta = \theta - v_t$$

where:

- $v_t$ is the velocity/momentum term
- $\gamma$ is typically 0.9 (momentum coefficient)

### Mathematical Insight

Expansion of the momentum update:
$$\theta = \theta - (\gamma v_{t-1} + \eta \nabla_\theta J(\theta))$$

This shows that:

- **Accelerating dimensions**: Gradients pointing in same direction accumulate → larger updates
- **Oscillating dimensions**: Gradients changing direction → reduced updates

### Benefits

- **Faster convergence**: Accelerates movements in consistent directions
- **Reduced oscillations**: Dampens fluctuations across ravines (steep in one dimension)
- **Better escape from local minima**: Momentum helps escape shallow local minima

### Visualizing Momentum

```
Without momentum: Oscillates across ravine walls, slow progress downhill
With momentum: Builds speed along the hill, smooth path to minimum
```

### Implementation

```python
v = 0
for epoch in range(epochs):
    gradient = compute_gradient(theta)
    v = gamma * v + learning_rate * gradient
    theta = theta - v
```

---

## Nesterov Accelerated Gradient

### Motivation

Standard momentum computes the gradient at the current position and then takes a step based on accumulated momentum. **Nesterov Accelerated Gradient (NAG)** improves upon this by computing the gradient "ahead of time" - looking ahead to where momentum will take us before computing the gradient.

### The Key Insight

We know momentum will take us to an approximate future position: $\theta - \gamma v_{t-1}$. Instead of computing the gradient at the current position, we compute it at this approximate future position:

$$v_t = \gamma v_{t-1} + \eta \nabla_\theta J(\theta - \gamma v_{t-1})$$
$$\theta = \theta - v_t$$

This is called **"Look-ahead" gradient computation**.

### Visual Comparison

**Standard Momentum:**

1. Compute current gradient (blue vector)
2. Take big step in momentum direction (large vector)
3. Result: Often overshoots

**Nesterov Momentum:**

1. Take big step in momentum direction (brown vector)
2. Measure gradient at that position and make correction (red vector)
3. Result: More responsive, prevents overshooting

### The Algorithm

**Step 1: Compute the lookahead position**
$$\tilde{\theta} = \theta - \gamma v_{t-1}$$

**Step 2: Compute gradient at lookahead position**
$$g_t = \nabla_\theta J(\tilde{\theta})$$

**Step 3: Update velocity**
$$v_t = \gamma v_{t-1} + \eta g_t$$

**Step 4: Update parameters**
$$\theta = \theta - v_t$$

### Alternative Formulation (Dozat)

A computationally simpler formulation that's easier to implement:

$$v_t = \gamma v_{t-1} + \eta \nabla_\theta J(\theta)$$
$$\theta = \theta - (\gamma v_t + \eta \nabla_\theta J(\theta))$$

This replaces the momentum from the previous step $v_{t-1}$ with the current momentum $v_t$, achieving similar look-ahead effect.

### Key Benefits

- **Reduced overshooting**: Look-ahead prevents overshooting the minimum
- **Faster convergence**: More responsive than standard momentum
- **Better performance on RNNs**: Significantly improved RNN training
- **Improved responsiveness**: Anticipates the effect of momentum

### Implementation

```python
v = 0
for epoch in range(epochs):
    # Look-ahead position
    theta_lookahead = theta - gamma * v

    # Compute gradient at lookahead
    gradient = compute_gradient(theta_lookahead)

    # Update velocity and parameters
    v = gamma * v + learning_rate * gradient
    theta = theta - v
```

### Hyperparameters

- $\gamma \approx 0.9$ (momentum coefficient)
- $\eta$: learning rate (typically 0.01 or 0.001 depending on context)

### When to Use

- When training **RNNs** and **LSTMs**
- When you have **momentum-based optimization** (SGD + momentum)
- When you want **faster convergence** than standard momentum
- Works well with **learning rate annealing schedules**

---

## Adagrad (Adaptive Gradient)

### Problem It Solves

- Features with different frequencies need different learning rates
- Sparse features should have larger updates (less frequent, more important)
- Dense features should have smaller updates (already well-optimized)

### The Algorithm

Adagrad adapts the learning rate **per-parameter** based on the sum of squared gradients:

$$G_t = \text{diag}(\sum_{s=0}^{t} g_s \odot g_s)$$

$$\theta_{t+1,i} = \theta_{t,i} - \frac{\eta}{\sqrt{G_{t,ii} + \epsilon}} \cdot g_{t,i}$$

Or in vectorized form:
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t + \epsilon}} \odot g_t$$

where:

- $G_t$ is a diagonal matrix of squared gradient accumulation
- $\epsilon$ is a smoothing term (typically $10^{-8}$) for numerical stability
- $\odot$ is element-wise multiplication

### Characteristics

- **Adaptive learning rates**: Different rate for each parameter
- **Large updates for rare features**: Infrequent features have small $G_t$ → large updates
- **Small updates for frequent features**: Frequent features accumulate large $G_t$ → small updates
- **No learning rate tuning**: Default 0.01 works well for most cases

### Critical Weakness

**Monotonic decay of learning rate**: $G_t$ only increases, causing learning rate to shrink over time:

- Accumulated sum grows continuously
- Learning rate becomes infinitesimally small
- Algorithm stops learning in later training stages
- This is the main limitation that led to RMSProp and Adadelta

### Use Cases

- Well-suited for **sparse data** (text, categorical features)
- Good for training large-scale neural networks (used by Google for training massive models)
- Used in GloVe word embeddings training

---

## RMSProp (Root Mean Square Propagation)

### Motivation

RMSProp fixes Adagrad's fatal flaw: the monotonically decreasing learning rate. Instead of accumulating all past squared gradients, it uses an **exponentially decaying average**.

### The Algorithm

RMSProp maintains a decaying average of squared gradients:

$$E[g^2]_t = \gamma E[g^2]_{t-1} + (1 - \gamma) g_t^2$$

$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{E[g^2]_t + \epsilon}} \odot g_t$$

where:

- $\gamma$ is the decay rate (typically 0.9)
- $\eta$ is the learning rate (typically 0.001)
- $E[g^2]_t$ is the exponentially weighted moving average of squared gradients

### Key Points

- **Similarity to Adadelta**: Very similar concept to the first update of Adadelta
- **Short-term memory**: Remembers recent gradients, forgets old ones
- **Prevents learning rate collapse**: Unlike Adagrad, learning rate doesn't shrink to zero
- **Effective for non-convex problems**: Works well in practice for deep neural networks

### Comparison with Adagrad

| Aspect                | Adagrad         | RMSProp               |
| --------------------- | --------------- | --------------------- |
| Gradient accumulation | Sum of all past | Exponential average   |
| Learning rate decay   | Monotonic       | Stable                |
| When to use           | Sparse features | General deep learning |
| Learning rate         | 0.01 (default)  | 0.001 (default)       |

---

## Adadelta (Adaptive Delta)

### Problem It Solves

- Removes dependence on default learning rate
- Fixes Adagrad's aggressive learning rate decay
- Units in gradient descent don't match parameter updates

### The Algorithm

**Step 1: Exponentially decaying average of squared gradients**
$$E[g^2]_t = \gamma E[g^2]_{t-1} + (1 - \gamma) g_t^2$$

**Step 2: Traditional update rule (like Adagrad but with RMS)**
$$\Delta\theta_t = -\frac{\eta}{\text{RMS}[g]_t} g_t$$

where $\text{RMS}[g]_t = \sqrt{E[g^2]_t + \epsilon}$

**Step 3: Address unit mismatch**
Adadelta also tracks exponentially weighted average of squared parameter updates:

$$E[\Delta\theta^2]_t = \gamma E[\Delta\theta^2]_{t-1} + (1 - \gamma) \Delta\theta_t^2$$

$$\text{RMS}[\Delta\theta]_t = \sqrt{E[\Delta\theta^2]_t + \epsilon}$$

**Step 4: Final Adadelta update (no explicit learning rate!)**
$$\Delta\theta_t = -\frac{\text{RMS}[\Delta\theta]_{t-1}}{\text{RMS}[g]_t} g_t$$

$$\theta_{t+1} = \theta_t + \Delta\theta_t$$

### Key Features

- **No learning rate required**: Learning rate is automatically adapted
- **Unit-consistent updates**: Parameter updates have same units as parameters
- **Effective for non-convex optimization**: Works well despite no tuning
- **Resolves Adagrad's issues**: Both learning rate decay and efficiency problems

### Hyperparameters

- $\gamma \approx 0.95$ (smoothing constant for gradient and parameter updates)
- No explicit learning rate $\eta$ needed

---

## Adam (Adaptive Moment Estimation)

### Intuition

Adam can be thought of as a combination of:

- **RMSProp**: Maintains exponentially decaying average of squared gradients
- **Momentum**: Maintains exponentially decaying average of past gradients

The name comes from "**Ad**aptive **M**oment estimation" - it adapts both the first moment (mean) and second moment (variance) of gradients.

### The Algorithm

**Step 1: Compute exponentially weighted moving averages of gradients (momentum)**
$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$

**Step 2: Compute exponentially weighted moving averages of squared gradients**
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$

where:

- $m_t$ is the first moment estimate (mean)
- $v_t$ is the second moment estimate (variance)
- $\beta_1 = 0.9$ (default), $\beta_2 = 0.999$ (default)

**Step 3: Bias correction**
Since $m_t$ and $v_t$ are initialized to 0, they are biased toward zero, especially in early iterations:

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$$

$$\hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

**Step 4: Parameter update**
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

where:

- $\eta$ is the learning rate (typically 0.001)
- $\epsilon = 10^{-8}$ (for numerical stability)

### Why Each Component Matters

- **Momentum term $\hat{m}_t$**: Acts like a heavy ball with friction (prefers flat minima)
- **Adaptive learning rate $\hat{v}_t$**: Larger gradients → smaller updates, sparse features → larger updates
- **Bias correction**: Ensures valid estimates early in training

### Behavior in Different Scenarios

Adam exhibits intelligent behavior in various scenarios:

1. **Consistent gradient direction**: Momentum term accumulates, acceleration occurs
2. **Oscillating dimension**: Gradients cancel out, updates are reduced
3. **Sparse feature**: Small accumulated squared gradient, larger update step
4. **Dense feature**: Large accumulated squared gradient, smaller update step

### Practical Performance

- **Empirically shown to work well** across many architectures and datasets
- **Less sensitive to learning rate** compared to SGD
- **Better than SGD with momentum** on many tasks
- **Converges faster** in practice for non-convex optimization

### Hyperparameters and Defaults

| Parameter  | Default   | Typical Range     | Role                |
| ---------- | --------- | ----------------- | ------------------- |
| $\eta$     | 0.001     | 0.0001-0.01       | Learning rate       |
| $\beta_1$  | 0.9       | 0.85-0.99         | Momentum decay      |
| $\beta_2$  | 0.999     | 0.99-0.999        | Variance decay      |
| $\epsilon$ | $10^{-8}$ | $10^{-8}-10^{-7}$ | Numerical stability |

### Extension: AMSGrad

An important variant of Adam that addresses convergence issues:

- Uses maximum of past second moments instead of exponential average
- Results in non-increasing step sizes
- Solves cases where Adam converges to suboptimal solutions
- Better performance on some datasets, but not universally better

---

## Rprop (Resilient Backpropagation)

### Overview

**Rprop** (Resilient Backpropagation) is an optimization algorithm that focuses on the sign of the gradient rather than its magnitude. It's designed to address the problem of vanishing and exploding gradients by using adaptive learning rates that don't depend on gradient magnitude.

### Motivation

Rprop solves key problems with standard gradient descent:

1. **Gradient magnitude problem**: In steep regions, gradients become very large; in flat regions, they become very small
2. **Step size adaptation**: Different parameters need different step sizes
3. **Batch dependency**: Batch normalization and sign changes in gradients cause learning rate inefficiencies

### The Key Principle

Instead of using gradient magnitude to update parameters, Rprop uses only the **sign of the gradient**:

- If gradient sign is positive: increase the parameter
- If gradient sign is negative: decrease the parameter
- If sign flips between iterations: reduce the step size
- If sign stays the same: increase the step size

### The Algorithm

**Step 1: Compute gradient**
$$g_t^i = \frac{\partial J}{\partial \theta^i}$$

**Step 2: Determine sign change**
$$S_t^i = \text{sign}(g_t^i \cdot g_{t-1}^i)$$

where $S_t^i \in \{-1, 0, +1\}$:

- $+1$: gradient sign is same as previous (accelerate)
- $-1$: gradient sign flipped (decelerate)
- $0$: gradient is zero

**Step 3: Update individual learning rates (step sizes)**

$$
\Delta_t^i = \begin{cases}
    \min(\Delta_{t-1}^i \cdot \eta^+, \Delta_{max}) & \text{if } S_t^i = +1 \\
    \max(\Delta_{t-1}^i \cdot \eta^-, \Delta_{min}) & \text{if } S_t^i = -1 \\
    \Delta_{t-1}^i & \text{if } S_t^i = 0
  \end{cases}
$$

where:

- $\eta^+ \approx 1.2$ (increase factor for consistent gradient direction)
- $\eta^- \approx 0.5$ (decrease factor for sign change)
- $\Delta_{max} = 50$ (maximum step size)
- $\Delta_{min} = 10^{-6}$ (minimum step size)

**Step 4: Update parameters using only gradient sign**
$$\theta_{t+1}^i = \theta_t^i - \Delta_t^i \cdot \text{sign}(g_t^i)$$

### Key Advantages

1. **Robust to gradient magnitude**: Step size independent of gradient scale
2. **Addresses vanishing/exploding gradients**: Doesn't use gradient magnitude directly
3. **Individual learning rates**: Each parameter has its own step size
4. **Fast convergence**: Adaptive step sizes accelerate in good directions
5. **No hyperparameter tuning**: Works well with default values

### Key Disadvantages

1. **Not well-suited for mini-batch**: Designed for batch/full-batch training
2. **Less common in deep learning**: Less flexible than modern methods
3. **High memory**: Requires storing previous gradients and step sizes
4. **Batch effects**: Step sizes don't transfer well across different batch sizes
5. **Poor with batch normalization**: Can interact poorly with gradient normalization

### Implementation

```python
class Rprop:
    def __init__(self, eta_plus=1.2, eta_minus=0.5,
                 delta_max=50, delta_min=1e-6):
        self.eta_plus = eta_plus
        self.eta_minus = eta_minus
        self.delta_max = delta_max
        self.delta_min = delta_min
        self.prev_gradient = None
        self.step_sizes = None

    def update(self, params, gradients):
        if self.prev_gradient is None:
            # Initialize on first call
            self.prev_gradient = [g.copy() for g in gradients]
            self.step_sizes = [0.1 * np.ones_like(g) for g in gradients]
            return

        for i, (param, grad) in enumerate(zip(params, gradients)):
            # Check sign change
            sign_change = np.sign(grad) * np.sign(self.prev_gradient[i])

            # Update step sizes
            self.step_sizes[i] *= np.where(
                sign_change > 0,
                np.minimum(self.eta_plus, self.delta_max / self.step_sizes[i]),
                np.where(
                    sign_change < 0,
                    np.maximum(self.eta_minus, self.delta_min / self.step_sizes[i]),
                    1.0
                )
            )

            # Update parameters using sign
            param -= self.step_sizes[i] * np.sign(grad)

            # Store current gradient for next iteration
            self.prev_gradient[i] = grad.copy()
```

### Hyperparameters

| Parameter        | Default   | Role                         |
| ---------------- | --------- | ---------------------------- |
| $\eta^+$         | 1.2       | Increase factor (accelerate) |
| $\eta^-$         | 0.5       | Decrease factor (decelerate) |
| $\Delta_{max}$   | 50        | Maximum step size            |
| $\Delta_{min}$   | $10^{-6}$ | Minimum step size            |
| Initial $\Delta$ | 0.1       | Initial step sizes           |

### When to Use Rprop

✓ **Good for:**

- Full-batch training (entire dataset at once)
- Classical feedforward neural networks
- Regression tasks
- When you want robust optimization without tuning
- Small to medium-sized networks

✗ **Not suitable for:**

- Mini-batch or stochastic gradient descent
- Large-scale deep learning models
- Convolutional neural networks in modern settings
- Recurrent neural networks
- With batch normalization or layer normalization

### Comparison with Other Algorithms

| Aspect                  | Rprop     | AdaGrad    | Adam        |
| ----------------------- | --------- | ---------- | ----------- |
| Uses gradient magnitude | ✗ No      | ✓ Yes      | ✓ Yes       |
| Mini-batch compatible   | ✗ Poor    | ✓ Yes      | ✓ Yes       |
| Modern usage            | Rare      | Occasional | Very Common |
| For deep networks       | ✗ No      | ✓ Yes      | ✓ Yes       |
| Memory overhead         | High      | Medium     | Medium      |
| Full-batch performance  | Excellent | Good       | Good        |

---

## Comparison of Optimization Algorithms

### Algorithm Comparison Table

| Algorithm        | Learning Rate    | Momentum | Adaptive | Sparse Data | Convergence Speed |
| ---------------- | ---------------- | -------- | -------- | ----------- | ----------------- |
| Gradient Descent | Manual           | ✗        | ✗        | Poor        | Slow              |
| SGD              | Manual           | ✗        | ✗        | OK          | Moderate          |
| SGD + Momentum   | Manual           | ✓        | ✗        | OK          | Fast              |
| Nesterov (NAG)   | Manual           | ✓        | ✗        | OK          | Very Fast         |
| Adagrad          | Auto             | ✗        | ✓        | Excellent   | Fast→Slow         |
| RMSProp          | Auto             | ✗        | ✓        | Good        | Fast              |
| Adadelta         | Auto             | ✗        | ✓        | Good        | Fast              |
| Adam             | Auto             | ✓        | ✓        | Good        | Very Fast         |
| Rprop            | Auto (per-param) | ✗        | ✓        | Good        | Fast (batch mode) |

### Guidance: Which Optimizer to Use

**Choose based on your problem:**

1. **Sparse data** (text, recommendation systems, CTR prediction)
   - Use: **Adagrad** or **Adadelta**
   - Why: Naturally handles varying feature frequencies

2. **General deep learning** (computer vision, most tasks)
   - Use: **Adam**
   - Why: Fast convergence, robust across architectures, minimal tuning

3. **You want maximum generalization** (large-scale models)
   - Use: **SGD with momentum** and learning rate annealing
   - Why: Often finds better minima, may take longer but better final performance

4. **Simple, reliable option** (baseline approach)
   - Use: **Adam with default parameters**
   - Why: Works well out of the box for most problems

5. **Custom optimization** (you have time to tune)
   - Use: **SGD + Momentum** or **RMSProp**
   - Why: Both can outperform Adam with proper tuning

### Performance Characteristics

**Batch gradient descent:**

- Guaranteed convergence for convex problems
- Slow on large datasets
- Used mostly for theoretical analysis

**SGD variants (minibatch SGD):**

- Industry standard for deep learning
- Fast per-iteration and per-epoch
- Stochasticity helps escape bad minima

**Adaptive methods (Adagrad, RMSProp, Adam):**

- Better convergence in practice
- Lower sensitivity to learning rate
- Adam is the most popular choice today

---

## Key Insights and Best Practices

### 1. Learning Rate is Critical

- Most important hyperparameter to tune
- Too high: divergence and loss oscillation
- Too low: extremely slow convergence
- **Use Adam's default (0.001) as starting point**

### 2. Batch Size Matters

- Affects gradient noise and hardware efficiency
- Larger batches: more stable gradients, better GPU utilization
- Smaller batches: noisier gradients, often better generalization
- **Typical range: 32-256**

### 3. Momentum Helps

- Accelerates convergence in consistent directions
- Reduces oscillation (important in non-convex landscapes)
- Works well with SGD (use 0.9 as default)

### 4. Adaptive Methods

- More robust to learning rate choice
- Better for sparse features
- Adam is typically best all-around choice
- Still benefit from learning rate annealing

### 5. Common Pitfalls

- **Forgetting to shuffle data**: Introduces bias into SGD
- **Too large learning rate**: Immediate divergence
- **Too small learning rate**: Training appears stuck
- **Ignoring validation performance**: Overfitting with continued training
- **Using old learning rates from old archives**: Different approaches need different rates

### 6. Modern Practice

- Most practitioners use **Adam** as default
- Use **SGD + momentum** for fine-tuning when maximum performance is needed
- Combine with **learning rate scheduling** for best results
- Consider **batch normalization** alongside optimizer to improve training

---

## Implementation Tips

### Generic Optimizer Template

```python
for epoch in range(num_epochs):
    for batch in get_batches(training_data):
        # Forward pass
        predictions = model(batch)
        loss = loss_function(predictions, batch.labels)

        # Backward pass
        gradients = compute_gradients(loss, model.parameters)

        # Update with chosen optimizer
        optimizer.update(model.parameters, gradients)
```

### Hyperparameter Starting Points

```python
# Adam (recommended for most cases)
optimizer = Adam(learning_rate=0.001, beta1=0.9, beta2=0.999)

# SGD with Momentum (for final fine-tuning)
optimizer = SGD(learning_rate=0.01, momentum=0.9)

# RMSProp (good alternative to Adam)
optimizer = RMSProp(learning_rate=0.001, decay=0.9)
```

---

## (Optional) Recommended Textbooks and Resources 

### "Convex Optimization" by Stephen Boyd and Lieven Vandenberghe

**Stanford University** | [Available Online (Free PDF)](https://web.stanford.edu/~boyd/cvxbook/)

This comprehensive textbook is **highly recommended** for anyone serious about understanding optimization in machine learning and deep learning. While it focuses on convex optimization (not directly applicable to non-convex neural network training), the theoretical foundations and intuitions are invaluable.

#### Key Chapters for ML/DL Practitioners (Focus Areas)

**Chapter 2: Convex Sets**

- Understand the geometry of optimization problems
- Learn about convex sets and their properties
- Foundation for recognizing convex vs. non-convex problems
- Applications to feasible regions in constrained optimization

**Chapter 3: Convex Functions**

- Comprehensive coverage of convex and concave functions
- Essential properties: first-order and second-order characterization
- Understanding function behavior and convergence guarantees
- Key insight: Why convex problems are "easy" to optimize

**Chapter 4: Convex Optimization Problems**

- Standard forms of optimization problems
- Linear and quadratic programming
- Conic and geometric programming
- Understanding problem structure affects solution methods
- Practical problem formulations

**Chapter 5: Duality**

- Lagrange duality and KKT conditions
- **Most important for understanding optimization deeply**
- Foundation for constrained optimization
- Weak and strong duality
- Dual decomposition methods
- Critical for understanding why certain algorithms work

#### Why This Book is Useful for Deep Learning

1. **Theoretical Foundation**: While neural networks are non-convex, understanding convex optimization provides invaluable intuitions about how gradients, step sizes, and convergence work

2. **Convergence Analysis**: Many proofs and convergence guarantees for algorithms like gradient descent, SGD (when applied to convex problems) are directly from these chapters

3. **Problem Formulation**: Learn how to frame optimization problems, which is useful when designing loss functions and constraints

4. **Lagrangian Methods**: Understanding duality is crucial for constrained optimization and regularization techniques

5. **Second-Order Methods**: Background for Newton's method and quasi-Newton methods used in optimization

6. **Practical Algorithms**: Many optimization algorithms used today have roots in convex optimization theory

#### How to Approach

**If you have limited time:**

- Focus on Chapters 2, 3, 4, and 5
- Understand the intuitions rather than all proofs
- Chapter 5 on duality is particularly enlightening

**For deep learning specifically:**

- Chapter 3: Convex functions → understand function behavior
- Chapter 4: Problem formulation → understand loss functions
- Chapter 5: Duality → understand regularization and constraints

**Complementary reading:**

- Section 9.1-9.3: Gradient descent and convergence analysis
- This connects convex theory to the algorithms we use on neural networks

#### Key Takeaways

While you can build neural networks without reading this book, understanding the foundations from Boyd's work will:

- Deepen your intuition about why certain optimizers work
- Help you design better loss functions and constraints
- Enable you to understand convergence behavior
- Provide mathematical rigor to complement practical deep learning
- Aid in debugging optimization issues in your own models

---

## References

1. Ruder, S. (2016). An overview of gradient descent optimization algorithms. arXiv preprint arXiv:1609.04747.
   - Source: https://www.ruder.io/optimizing-gradient-descent/

2. Dive into Deep Learning - Chapter 12: Optimization Algorithms
   - Source: https://d2l.ai/chapter_optimization/index.html

3. **Boyd, S., & Vandenberghe, L. (2004). Convex Optimization.** Cambridge University Press.
   - **Free online version:** https://web.stanford.edu/~boyd/cvxbook/
   - **Recommended chapters for deep learning:** 2 (Convex Sets), 3 (Convex Functions), 4 (Convex Optimization Problems), 5 (Duality)
   - **Why it matters:** Provides theoretical foundation for understanding gradient-based optimization and convergence analysis

4. Key Papers:
   - [Kingma, D. P., & Ba, J. L. (2015). Adam: A Method for Stochastic Optimization. ICLR.](https://arxiv.org/pdf/1412.6980)
   - [Zeiler, M. D. (2012). ADADELTA: An Adaptive Learning Rate Method. arXiv:1212.5701.](https://arxiv.org/pdf/1212.5701)
   - [Duchi, J., Hazan, E., & Singer, Y. (2011). Adaptive Subgradient Methods for Online Learning. JMLR.](https://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)
   - [Tieleman, T., & Hinton, G. (2012). RMSProp: Divide the gradient by a running average. [Coursera Lecture]](https://www.scirp.org/pdf/ojop2024133_12730372.pdf)

---
