# Decision Trees — Deep Guide

## Overview

Decision Trees are a fundamental, interpretable machine learning algorithm used for both classification and regression tasks. A decision tree learns a hierarchical series of if-then-else decision rules by recursively partitioning the feature space. At each internal node, the tree tests a feature and splits data into two (or more) child nodes based on a threshold or categorical value. Leaf nodes represent final predictions (class label for classification, numeric value for regression).

Decision trees are attractive because they are intuitive, require minimal data preprocessing, handle mixed feature types naturally, and provide built-in feature importance estimates. However, trees are prone to overfitting, can be unstable (small data perturbations lead to very different trees), and greedy node splitting may not find globally optimal partitions. Modern ensemble methods (Random Forests, Gradient Boosting) address these limitations.

## Intuition

The core idea: recursively partition data into increasingly homogeneous subsets. Each split is chosen to maximize information gain or reduce impurity. Think of the tree as a series of yes-no questions:

- Is feature X1 > 5?
  - If yes, is feature X2 ∈ {A, B}?
  - If no, is feature X3 < 10?

The tree continues until reaching a stopping criterion (pure node, maximum depth, minimum samples, etc.), at which point we output a prediction.

## When Decision Trees Are Appropriate

- **Interpretability is critical**: business rules, regulatory compliance, explainability required.
- **Mixed feature types**: handles numeric and categorical features natively without extensive preprocessing.
- **Small-to-medium datasets**: trees are fast to train; for large datasets, consider ensembles or subsampling.
- **Non-linear decision boundaries** with clear feature interactions.
- **Feature importance estimation**: easily extract which features matter most.
- **Multi-output regression/classification**: trees naturally handle multiple targets.

When not to use Decision Trees alone:

- **Pure predictive accuracy critical**: ensemble methods (Random Forests, Gradient Boosting) typically outperform single trees.
- **Very large, high-dimensional datasets**: slow training and high memory; use simplified models or approximations.
- **Imbalanced data** without careful tuning (use class weights, SMOTE, or ensemble balancing).
- **Requires generalization on data far from training distribution**: trees can be brittle outside the training region.

## Mathematical Formulation and Splitting Criteria

### Information Gain & Entropy (Classification)

For a node $S$, define entropy as:

$$
H(S) = -\sum_{c=1}^{C} p_c \log_2(p_c)
$$

where $p_c$ is the proportion of samples in $S$ belonging to class $c$, and $C$ is the number of classes.

When considering a split on feature $X$ with threshold $t$:

$$
\text{IG}(S, X, t) = H(S) - \left(\frac{|S_L|}{|S|} H(S_L) + \frac{|S_R|}{|S|} H(S_R)\right)
$$

where $S_L$ and $S_R$ are the left (feature < threshold) and right (feature ≥ threshold) child nodes. The split maximizing information gain is selected.

### Gini Impurity (Classification, often faster)

Gini impurity:

$$
\text{Gini}(S) = 1 - \sum_{c=1}^{C} p_c^2
$$

Gini reduction for a split:

$$
\Delta \text{Gini}(S, X, t) = \text{Gini}(S) - \left(\frac{|S_L|}{|S|} \text{Gini}(S_L) + \frac{|S_R|}{|S|} \text{Gini}(S_R)\right)
$$

Both information gain and Gini reduction favor splits that create more homogeneous child nodes. Gini is computationally slightly faster and is default in scikit-learn.

### Mean Squared Error (Regression)

For regression, minimize MSE over child nodes:

$$
\text{MSE}(S) = \frac{1}{|S|} \sum_{i \in S} (y_i - \bar{y})^2
$$

where $\bar{y}$ is the mean target value in $S$. The criterion for splitting is:

$$
\text{MSE reduction} = \text{MSE}(S) - \left(\frac{|S_L|}{|S|} \text{MSE}(S_L) + \frac{|S_R|}{|S|} \text{MSE}(S_R)\right)
$$

### Mean Absolute Error (Regression, robust alternative)

$$
\text{MAE}(S) = \frac{1}{|S|} \sum_{i \in S} |y_i - \tilde{y}|
$$

where $\tilde{y}$ is the median target value. MAE is more robust to outliers than MSE.

## Tree Growing Process (Recursive Partitioning)

**Algorithm (greedy CART-like approach):**

```
function GROW_TREE(S, depth, max_depth):
    if stopping_criterion(S, depth, max_depth):
        return LEAF(predict(S))

    best_split = None
    best_gain = 0

    for each feature X_j:
        for each possible threshold t:
            S_L = {x in S : X_j(x) < t}
            S_R = {x in S : X_j(x) >= t}
            gain = compute_gain(S, S_L, S_R)

            if gain > best_gain:
                best_gain = gain
                best_split = (X_j, t)

    if best_split is None or best_gain <= 0:
        return LEAF(predict(S))

    X_j, t = best_split
    S_L = {x in S : X_j(x) < t}
    S_R = {x in S : X_j(x) >= t}

    left_child = GROW_TREE(S_L, depth+1, max_depth)
    right_child = GROW_TREE(S_R, depth+1, max_depth)

    return NODE(X_j, t, left_child, right_child)

function stopping_criterion(S, depth, max_depth):
    return (|S| < min_samples_split or
            depth >= max_depth or
            H(S) == 0 or  # pure node
            no valid splits improve gain)
```

## Hyperparameters and Regularization

### Key Hyperparameters

- **max_depth**: maximum tree depth (default None, grows until pure). Smaller depths reduce overfitting but increase bias.
- **min_samples_split**: minimum samples required to split a node (default 2). Larger values reduce overfitting.
- **min_samples_leaf**: minimum samples required at a leaf node (default 1). Prevents overfitting to single outliers.
- **max_features**: number of features to consider at each split (default None = all features). Options:
  - `"sqrt"`: $\sqrt{n_\text{features}}$, reduces variance in ensembles.
  - `"log2"`: $\log_2(n_\text{features})$.
  - `int`: use exact number (e.g., 5).
  - `float`: fraction (e.g., 0.3 = 30% of features).
- **min_impurity_decrease**: minimum gain to split (default 0). Avoids very small improvements.
- **class_weight**: handle imbalanced classification by weighting classes (e.g., `"balanced"` = weight inversely proportional to frequency).

### Regularization Strategy

Overfitting in trees manifests as high depth, complex rules capturing noise. Regularization tactics:

1. **Limit tree depth** (`max_depth`): most effective for reducing overfitting.
2. **Increase min_samples_split / min_samples_leaf**: require more evidence per split.
3. **Decrease max_features**: force consideration of multiple features, reduce variance.
4. **Prune the tree** (cost-complexity pruning): grow a deep tree, then prune branches that don't improve validation performance.
5. **Use ensemble methods** (Random Forests, Gradient Boosting): reduces variance while maintaining low bias.

## Feature Importance

Decision trees provide a built-in measure of feature importance. **Mean Decrease in Impurity (MDI)** for feature $X_j$ is:

$$
\text{Importance}(X_j) = \frac{1}{T} \sum_{t \in T} \left(\text{gain}_t \cdot \frac{|S_t|}{|S_\text{root}|}\right)
$$

where the sum is over all internal nodes $t$ where feature $X_j$ is used for splitting, $\text{gain}_t$ is the impurity reduction at node $t$, and $T$ is the total number of such nodes.

**Interpretation:** features used in early splits affecting many samples are ranked as important. However, MDI has biases:

- Biased toward high-cardinality features.
- Does not account for feature correlations (related features may be interchangeable).

**Alternative:** Permutation Feature Importance — shuffle each feature independently on validation data and measure drop in performance. Less biased and more interpretable.

## Worked Numeric Example (Student-Friendly)

**Binary Classification with 4 Samples:**

| ID  | Age | Income | Approved (y) |
| --- | --- | ------ | ------------ |
| 1   | 25  | Low    | No           |
| 2   | 35  | High   | Yes          |
| 3   | 40  | High   | Yes          |
| 4   | 28  | Low    | No           |

Root node entropy:

$$
H(\text{root}) = -\left(\frac{2}{4} \log_2(0.5) + \frac{2}{4} \log_2(0.5)\right) = 1.0 \text{ bit}
$$

**Option 1: Split on Income = High/Low**

- Left (Low): {1, 4} → both No → $H_L = 0$
- Right (High): {2, 3} → both Yes → $H_R = 0$

Information Gain:

$$
\text{IG} = 1.0 - (0.5 \cdot 0 + 0.5 \cdot 0) = 1.0 \text{ bit}
$$

**Option 2: Split on Age < 30/≥ 30**

- Left (< 30): {1, 4} → No, No → $H_L = 0$
- Right (≥ 30): {2, 3} → Yes, Yes → $H_R = 0$

Information Gain:

$$
\text{IG} = 1.0 - (0.5 \cdot 0 + 0.5 \cdot 0) = 1.0 \text{ bit}
$$

Both splits yield 1.0 bits of information. In practice, tie-breaking depends on implementation or additional criteria. The tree would select one split (say Income first), then the resulting leaves are pure (no further splits needed).

**Final tree:**

```
           root [Approved: 2 No, 2 Yes]
          /                           \
    Income = Low?                 Income = High?
       /                              \
    Leaf: No                        Leaf: Yes
```

## Regression Tree Example

**Simple regression with 4 samples, targets $y = [2, 4, 4, 6]$ and feature $X = [1, 2, 3, 4]$:**

Root MSE:

$$
\text{MSE}_\text{root} = \frac{(2-4)^2 + (4-4)^2 + (4-4)^2 + (6-4)^2}{4} = \frac{8}{4} = 2.0
$$

**Split on X < 2.5:**

- Left: $[2, 4]$ → mean = 3, MSE = $\frac{(2-3)^2 + (4-3)^2}{2} = 1.0$
- Right: $[4, 6]$ → mean = 5, MSE = $\frac{(4-5)^2 + (6-5)^2}{2} = 1.0$

MSE reduction:

$$
\Delta \text{MSE} = 2.0 - (0.5 \cdot 1.0 + 0.5 \cdot 1.0) = 1.0
$$

The tree splits here, resulting in two leaf nodes with predictions 3 and 5.

## Handling Categorical Features

Decision trees natively handle categorical features without requiring one-hot encoding:

- **Binary categorical split**: $X \in \{A, B\}$ vs $X \in \{C, D, E\}$.
- **Ordinal categorical split**: $X \leq \text{"medium"}$ vs $X > \text{"medium"}$ for ordered categories.
- **Multi-way splits** (in some implementations): separate branches for each category value (less common; binary splits more practical).

Scikit-learn's CART algorithm produces binary splits. For categorical variables, consider:

- Ordinal encoding if categories are ordered.
- Grouping rare categories before splitting.

## Pruning: Handling Overfitting

**Cost-Complexity Pruning (Weakest Link Pruning):**

1. Grow a full tree $T_0$ (or constrained tree).
2. For each internal node $t$, compute cost-complexity parameter:

   $$
   \alpha_t = \frac{R(t) - R(T_t)}{|L(T_t)| - 1}
   $$

   where $R(t)$ is error at node $t$ (treating it as a leaf), $R(T_t)$ is error of subtree rooted at $t$, and $|L(T_t)|$ is the number of leaves in subtree.

3. Sequentially prune the node with smallest $\alpha_t$, creating a sequence of nested trees.
4. Use cross-validation to select the best tree from this sequence by validation error.

Scikit-learn provides `cost_complexity_pruning_path()` to automate this.

## Handling Imbalanced Classification

When classes are severely imbalanced:

- **class_weight="balanced"**: weight classes inversely proportional to their frequency. $w_c = \frac{n}{C \cdot n_c}$ where $n$ is total samples, $C$ is number of classes, $n_c$ is samples in class $c$.
- **Threshold moving**: after fitting, adjust the decision threshold for classification.
- **Resampling** before fitting (SMOTE, random undersampling, stratified split).
- **Ensemble methods**: Random Forests or Gradient Boosting often handle imbalance better.

## Complexity and Scalability

**Training Complexity:**

- Time: $O(n_\text{features} \cdot n_\text{samples} \cdot \log(n_\text{samples}) \cdot \text{depth})$ in typical case (sorting features for split search).
- More precisely, $O(n_\text{features} \cdot n_\text{samples} \cdot \text{depth})$ with simplified splitting.
- Memory: $O(n_\text{samples})$ to store the tree structure and indices during construction.

**Prediction Complexity:**

- Time: $O(\text{depth})$ per sample, negligible compared to training.

**Scalability:**

- Reasonably efficient for small-to-medium datasets (millions of samples, hundreds of features).
- For very large datasets (billions of samples) or very wide datasets, consider:
  - Subsampling data during tree construction.
  - Approximate splitting (quantile-based thresholds instead of exact).
  - Distributed tree learning (e.g., LightGBM, XGBoost support this).

## Missing Values

Decision trees can be extended to handle missing values:

- **Surrogate splits**: during training, keep alternative (surrogate) split rules at each node. If a feature is missing at prediction time, use the best surrogate split.
- **Direct missing-value handling** (simpler): pass samples with missing values down both branches (or majority branch), then aggregate predictions.
- **Imputation before training**: fill missing values using mean/mode/KNN/iterative methods before fitting.

Scikit-learn's CART does not natively support missing values; preprocess before fitting. Some other libraries (e.g., catboost, xgboost) have native missing-value strategies.

## Decision Boundaries and Axis-Aligned Splits

Since trees recursively split along feature axes (axis-aligned), they produce **rectilinear (box-like) decision boundaries**. This means:

- Diagonal or complex nonlinear boundaries require many nodes and depth.
- Trees handle axis-aligned patterns efficiently.
- Ensemble methods (Forests, Boosting) mitigate this limitation by combining many trees.

**Illustration:** to approximate a diagonal boundary with tree splits, many small boxes are needed, leading to overfitting. Conversely, simple rectangles are modeled easily.

## Multiple Outputs (Multi-Target Trees)

Trees naturally extend to multiple outputs:

- **Multi-output classification**: predict multiple class labels per sample. Each leaf stores a vector of class predictions.
- **Multi-output regression**: predict multiple numeric targets per sample. Each leaf stores a vector of predicted values.

Splitting criteria are generalized (e.g., weighted average of impurity reductions across targets).

## Practical Tips

- **Always split data into train/validation/test** before tree building (especially for grid search on depth/parameters).
- **Use cross-validation** to select hyperparameters robustly.
- **Start shallow** (`max_depth=3`–`5`) and increase gradually while monitoring validation error.
- **Check feature importance** to understand which features the model relies on; may reveal data quality issues.
- **Visualize the tree** (dot export, plotting) to debug and explain decisions.
- **Consider ensemble methods** (Random Forests, Gradient Boosting) for better predictive accuracy.
- **Be wary of very deep trees** without regularization; likely overfitting.
- **Scale or normalize features** is not critical for trees (unlike distance-based methods), but can help interpretability.

## Pseudocode (Classification)

```text
function PREDICT(tree, x):
    node = tree.root
    while node is not LEAF:
        if x[node.feature] < node.threshold:
            node = node.left_child
        else:
            node = node.right_child
    return node.prediction

function FEATURE_IMPORTANCE(tree, n_features):
    importance = [0] * n_features
    for each node in tree (internal nodes):
        importance[node.feature] += node.gain * (node.n_samples / tree.root.n_samples)
    return importance / sum(importance)
```

## Scikit-learn Examples

**Classification with Hyperparameter Tuning:**

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

dt = DecisionTreeClassifier(random_state=42)
grid = GridSearchCV(dt, param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)

print("Best parameters:", grid.best_params_)
print("Best CV score:", grid.best_score_)
print("Test score:", grid.best_estimator_.score(X_test, y_test))
```

**Regression:**

```python
from sklearn.tree import DecisionTreeRegressor

dt_reg = DecisionTreeRegressor(max_depth=5, min_samples_split=5, random_state=42)
dt_reg.fit(X_train, y_train)
y_pred = dt_reg.predict(X_test)

from sklearn.metrics import mean_squared_error, r2_score
print("MSE:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))
```

**Visualization:**

```python
from sklearn import tree
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(20, 10))
tree.plot_tree(dt, ax=ax, feature_names=feature_names,
               class_names=class_names, filled=True, rounded=True)
plt.show()

# Export to dot format for GraphViz
tree.export_text(dt, feature_names=feature_names)
```

**Feature Importance:**

```python
import pandas as pd

importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': dt.feature_importances_
}).sort_values('importance', ascending=False)

print(importance_df)

import matplotlib.pyplot as plt
plt.barh(importance_df['feature'], importance_df['importance'])
plt.xlabel('Importance')
plt.title('Feature Importance')
plt.show()
```

**Cost-Complexity Pruning:**

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

path = dt.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

trees = []
for ccp_alpha in ccp_alphas:
    tree = DecisionTreeClassifier(random_state=42, ccp_alpha=ccp_alpha)
    tree.fit(X_train, y_train)
    trees.append(tree)

scores = [tree.score(X_test, y_test) for tree in trees]
best_idx = scores.index(max(scores))
best_tree = trees[best_idx]
print(f"Best alpha: {ccp_alphas[best_idx]}, Test score: {scores[best_idx]}")
```

**Handling Class Imbalance:**

```python
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(
    class_weight='balanced',
    max_depth=5,
    min_samples_split=5,
    random_state=42
)
dt.fit(X_train, y_train)

# Evaluate with precision-recall or F1 for imbalanced data
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, dt.predict(X_test)))
```

## Popular Types of Decision Trees

Various decision tree algorithms and variants exist, each with different splitting criteria and design philosophy:

### CART (Classification and Regression Trees)

- Developed by Breiman et al. (1984), the foundation of most modern implementations.
- **Splitting criterion**: Gini impurity for classification, MSE for regression.
- **Binary splits**: always produces binary trees (each node has at most 2 children).
- **Handles mixed data types**: numeric and categorical features naturally.
- **Pruning**: supports cost-complexity pruning.
- **Implementation**: scikit-learn's `DecisionTreeClassifier` and `DecisionTreeRegressor` are based on CART.

### ID3 (Iterative Dichotomiser 3)

- Developed by Quinlan (1986), historical importance but less commonly used now.
- **Splitting criterion**: Information gain (entropy reduction).
- **Limitations**: only handles categorical features (not numeric without discretization), does not support pruning in original form.
- **Multi-way splits**: can create n-ary trees (more than 2 children per node).
- **Use case**: mostly historical/educational; largely superseded by C4.5 and CART.

### C4.5 and C5.0

- Evolution of ID3, developed by Quinlan.
- **Splitting criterion**: Information gain ratio (normalized by split entropy to reduce bias toward high-cardinality features).
- **Improvements**: handles missing values using surrogates, pruning, boosting variant (C5.0), handles numeric features via discretization.
- **Multi-way splits**: creates n-ary trees.
- **Use case**: commercial and research applications; historically popular before CART dominance.

### Conditional Inference Trees (CTree)

- Developed for statistical rigor; uses hypothesis testing for splitting.
- **Splitting criterion**: statistical significance tests (e.g., chi-square for categorical, t-test for numeric).
- **Advantages**: automatic stopping without explicit pruning, unbiased feature selection (no bias toward high-cardinality features).
- **Implementation**: `partykit` R package.
- **Use case**: when statistical inference and hypothesis testing are important; less biased splitting.

### CHAID (Chi-squared Automatic Interaction Detection)

- Uses chi-squared tests for splitting decisions.
- **Splitting criterion**: chi-squared test for categorical variables.
- **Multi-way splits**: can produce n-ary trees.
- **Designed for**: categorical target and features; market research and segmentation.
- **Use case**: exploratory analysis, customer segmentation, when multi-way splits are beneficial.

### Oblique Decision Trees

- Splits are not axis-aligned but along arbitrary hyperplanes (e.g., $a_1 X_1 + a_2 X_2 > t$).
- **Advantage**: more compact trees for problems with diagonal boundaries.
- **Disadvantage**: more complex, harder to interpret, computationally expensive.
- **Example**: Linear Discriminant Analysis (LDA) trees, fuzzy trees.
- **Use case**: problems where axis-aligned splits are inefficient; neural-network-like capability but more interpretable.

### Isolation Forests

- Specialized tree ensemble for anomaly detection.
- **Idea**: anomalies are isolated (require fewer splits than normal points).
- **Splitting criterion**: randomly select features and thresholds (not optimized for purity).
- **Output**: anomaly score based on path length to leaf.
- **Advantage**: efficient, no need for distance or density estimation.
- **Use case**: outlier/anomaly detection in high-dimensional data.

### Extra Trees (Extremely Randomized Trees)

- Randomizes both feature selection and split thresholds.
- **Splitting criterion**: information gain or MSE, but with random threshold selection.
- **Advantage**: faster training, reduced overfitting variance when combined in ensemble.
- **Implementation**: scikit-learn's `ExtraTreeClassifier` and `ExtraTreeRegressor`.
- **Use case**: large datasets, when speed and variance reduction are priorities.

### Survival Trees (Regression Trees for Survival Analysis)

- Adapted for time-to-event data with censoring.
- **Output**: splits to maximize separation of survival curves.
- **Splitting criterion**: log-rank test or similar survival-specific measures.
- **Use case**: medical studies, warranty analysis, customer lifetime value predictions with censoring.

### Gradient Boosting Trees (GBTs)

- Not a single tree type but an ensemble framework where trees correct residuals of previous trees.
- **Popular implementations**: XGBoost, LightGBM, CatBoost, scikit-learn's `GradientBoostingClassifier/Regressor`.
- **Advantages**: strong predictive power, handles complex patterns, supports mixed feature types.
- **Disadvantage**: less interpretable than single trees, more hyperparameters to tune.
- **Use case**: competitions (Kaggle), when accuracy is paramount.

### Random Forest

- Ensemble of independently trained trees on random data/feature subsets.
- **Splitting criterion**: same as individual trees (usually Gini/MSE).
- **Advantage**: reduced variance, robust to overfitting, parallelizable.
- **Implementation**: scikit-learn's `RandomForestClassifier` and `RandomForestRegressor`.
- **Use case**: general-purpose, strong baseline, good balance of accuracy and interpretability.

**Comparison Table (Popular Implementations):**

| Algorithm         | Criterion       | Multi-way? | Pruning               | Missing Vals          | Best For              |
| ----------------- | --------------- | ---------- | --------------------- | --------------------- | --------------------- |
| CART              | Gini/MSE        | Binary     | Yes (cost-complexity) | Imputation/surrogates | Balanced, general use |
| ID3               | Info Gain       | Yes        | No                    | No                    | Categorical data      |
| C4.5              | Info Gain Ratio | Yes        | Yes                   | Surrogate             | Improvement on ID3    |
| CTree             | Stat. Tests     | Binary     | Auto-stop             | Conditional           | Statistical rigor     |
| CHAID             | Chi-squared     | Yes        | Yes                   | Mode                  | Segmentation          |
| Oblique           | Linear combo    | Binary     | Yes                   | Problem-dependent     | Diagonal boundaries   |
| Isolation Forests | Random          | Binary     | No                    | No                    | Anomaly detection     |
| Extra Trees       | Info Gain/MSE   | Binary     | No                    | No                    | Speed & robustness    |
| Gradient Boosting | Varies          | Binary     | Varies                | Depends on impl.      | Accuracy              |
| Random Forest     | Gini/MSE        | Binary     | No                    | No                    | Robust baseline       |

## Extensions and Ensemble Methods

Beyond individual tree types, trees form the basis for powerful ensemble methods:

- **Random Forests**: train multiple trees on random subsets of data and features, average predictions → reduces variance, improves generalization.
- **Gradient Boosting** (XGBoost, LightGBM, CatBoost): sequentially train trees, each correcting errors of previous trees → reduces bias and variance.
- **AdaBoost**: reweight samples, focusing on misclassified examples; combine weak learners into strong ensemble.
- **Extra Trees** (Extremely Randomized Trees): randomize split thresholds instead of searching optimally → faster, less prone to overfitting in high dimensions.

## Advantages and Disadvantages

**Pros:**

- Highly interpretable: decision rules can be extracted and explained to non-technical stakeholders.
- Requires minimal data preprocessing (no scaling, handles mixed feature types).
- Fast prediction (logarithmic in tree size).
- Provides feature importance estimates.
- Non-parametric: makes no distributional assumptions.
- Handles multi-output tasks naturally.

**Cons:**

- Prone to overfitting (high variance), especially on small datasets.
- Greedy splitting does not guarantee globally optimal tree.
- Unstable: small changes in data lead to very different trees.
- Axis-aligned splits lead to inefficient representation of diagonal/smooth boundaries.
- Biased toward high-cardinality features.
- Single trees typically underperform ensemble methods on large datasets.

## Common Pitfalls and Debugging

- **Tree too deep**: symptoms include 100% training accuracy but poor test accuracy. Fix: reduce `max_depth`, increase `min_samples_split/leaf`, use pruning.
- **Features not scaled but feature importance looks skewed**: trees are scale-invariant for splitting, but visualization/interpretation can be confusing. Feature importance reflects splits, not feature magnitude.
- **Imbalanced classification with poor minority accuracy**: use `class_weight='balanced'` or resample before fitting.
- **Missing values during prediction**: preprocess (impute) before fitting or use surrogate splits (manual implementation needed).
- **Model not using expected features**: check feature importance; may indicate data quality issues or feature redundancy.

## Checklist Before Using Decision Trees

- [ ] Understand business/domain requirements for interpretability vs accuracy trade-off.
- [ ] Have sufficient samples per node (rule of thumb: ≥20–30 samples per leaf).
- [ ] Prepare train/validation/test split for hyperparameter tuning.
- [ ] If imbalanced, use `class_weight` or resampling.
- [ ] Tune `max_depth` via cross-validation to balance bias-variance.
- [ ] Visualize final tree to ensure rules are sensible.
- [ ] Check feature importance for unexpected patterns.
- [ ] Consider ensemble methods (Random Forests, Gradient Boosting) for better predictive power.
- [ ] Validate that predictions hold up on new, out-of-sample data.

## Comparative Summary

- **vs. KNN**: Trees require no distance metric, faster prediction, more interpretable; KNN has fewer hyperparameters.
- **vs. Linear Models**: Trees handle nonlinearity and feature interactions without explicit engineering; linear models are faster and more stable on small datasets.
- **vs. Neural Networks**: Trees are interpretable and require less data; NN more powerful for very complex patterns with large data.
- **vs. Ensemble Trees** (Forests, Boosting): single trees prone to overfitting; ensembles are more robust and accurate but less interpretable.

## Short Exercise for Students

Given the following mini dataset with 5 samples, feature X and target y:

| ID  | X   | y   |
| --- | --- | --- |
| 1   | 1.0 | 0   |
| 2   | 2.0 | 1   |
| 3   | 3.0 | 1   |
| 4   | 4.0 | 0   |
| 5   | 2.5 | 1   |

(a) Compute root entropy.
(b) Try splits at X < 1.5, X < 2.5, X < 3.5, and compute information gain for each.
(c) Which split is selected by the greedy algorithm?
(d) Recursively split the resulting left and right child nodes.
(e) Draw the final tree and list the decision rules.

**Bonus:** Verify your tree by predicting the class for a new point X = 3.5.

## References & Further Reading

- Breiman, L., Friedman, J., Stone, C. J., & Olshen, R. A. (1984). Classification and Regression Trees (CART).
- Quinlan, J. R. (1986). Induction of decision trees. Machine Learning.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning.
- Scikit-learn documentation: Decision Trees — https://scikit-learn.org/stable/modules/tree.html
- Molnar, C. (2020). Interpretable Machine Learning — https://christophmolnar.com/books/interpretable-ml/
- Cost-Complexity Pruning: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier.cost_complexity_pruning_path
- MLU-Explain: Decision Trees Interactive Tutorial — https://mlu-explain.github.io/decision-tree/

---
