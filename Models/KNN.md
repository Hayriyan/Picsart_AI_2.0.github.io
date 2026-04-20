# K-Nearest Neighbors (KNN) — Deep Guide

## Overview

K-Nearest Neighbors (KNN) is a simple, non-parametric, instance-based learning algorithm used for classification and regression. It makes predictions by finding the $k$ closest training examples (neighbors) in feature space and aggregating their labels (classification: majority vote; regression: average or weighted average).

KNN is intuitive, requires minimal training (no explicit model fitting), and can capture complex decision boundaries when the data distribution supports it. However, KNN can be computationally expensive at prediction time and sensitive to feature scaling and high-dimensional data.

## Intuition

- Represent each example as a point in a multidimensional feature space.
- For a new example, find the nearest training points according to a distance metric (e.g., Euclidean).
- Use the labels of those nearest points to decide the label/value for the new example.

This leverages the locality assumption: similar inputs have similar outputs.

## When KNN Is Appropriate

- Small to medium-sized datasets where prediction latency is acceptable.
- Low-to-moderate dimensional data (or after dimensionality reduction).
- As a strong, interpretable baseline for classification/regression problems.
- Problems where decision boundaries are irregular and difficult to model parametrically.

When not to use KNN:

- Very large datasets with strict latency requirements (unless using ANN or indexing structures).
- Extremely high-dimensional data without dimensionality reduction (curse of dimensionality).
- Features with mixed or poorly scaled units unless properly preprocessed.

## Mathematical Formulation

Given training data $\{(x_i, y_i)\}_{i=1}^n$ and a query point $x_q$:

1. Compute distances $d(x_q, x_i)$ for all $i$.
2. Select the indices of the $k$ smallest distances: $N_k(x_q)$.
3. Classification prediction (majority vote):

$$
\hat{y} = \arg\max_{c} \sum_{i\in N_k(x_q)} \mathbf{1}(y_i = c)
$$

4. Regression prediction (simple average):

$$
\hat{y} = \frac{1}{k} \sum_{i\in N_k(x_q)} y_i
$$

Weighted regression/classification (distance weighting):

$$
w_i = w\big(d(x_q, x_i)\big) \quad\text{(e.g. } w(d)=\frac{1}{d+\varepsilon}\text{ or } w(d)=e^{-d^2/(2\sigma^2)}\text{)}
$$

$$
\hat{y}_{reg} = \frac{\sum_{i\in N_k(x_q)} w_i y_i}{\sum_{i\in N_k(x_q)} w_i}
$$

For classification, replace sum of values with weighted votes. Concretely, compute the weighted vote for each class $c$:

$$
V_c = \sum_{i\in N_k(x_q)} w_i\,\mathbf{1}(y_i = c)
$$

and predict

$$
\hat{y} = \arg\max_c V_c
$$

Worked numeric example (student-friendly)

Example 1 — Classification (inverse-distance weights):

- Suppose $k=3$ and the 3 nearest neighbors have distances $d = [0.5,\;1.0,\;2.0]$ and labels $y = [A,\;B,\;A]$.
- Use inverse-distance weights $w(d)=1/(d+\varepsilon)$ with $\varepsilon=10^{-6}$.

Compute weights approximately:

$$
w \approx [1/0.5,\;1/1.0,\;1/2.0] = [2.0,\;1.0,\;0.5]
$$

Weighted votes:

$$
V_A = 2.0 + 0.5 = 2.5, \qquad V_B = 1.0
$$

Prediction: $\hat{y}=A$ (since $V_A>V_B$).

Example 2 — Regression (same neighbors but numeric targets):

- Suppose the neighbor target values are $y = [3.0,\;4.0,\;5.0]$ with the same weights $[2.0,1.0,0.5]$.

Compute weighted regression prediction:

$$
\hat{y}_{reg} = \frac{2.0\cdot 3.0 + 1.0\cdot 4.0 + 0.5\cdot 5.0}{2.0 + 1.0 + 0.5} = \frac{12.5}{3.5} \approx 3.571
$$

Short exercise for students

Given $k=4$, neighbor distances $d=[0.2,\;0.8,\;1.5,\;2.5]$ and labels $y=[C,\;A,\;A,\;C]$, use inverse-distance weighting $w(d)=1/(d+10^{-6})$ and answer:

- (a) Compute the weights and the weighted votes $V_A, V_C$. Which class is predicted?
- (b) If the numeric targets were $[2.0,\;3.0,\;4.0,\;5.0]$, compute the weighted regression prediction.

You can compute these by hand or with a short script; this exercise helps see how distance influences the final decision.

## Distance Metrics and Their Use

- Euclidean (L2): $\|x-y\|_2 = \sqrt{\sum_j (x_j - y_j)^2}$. Common for continuous numeric data.
- Manhattan (L1): $\|x-y\|_1 = \sum_j |x_j - y_j|$. More robust to some kinds of noise.
- Minkowski: $\|x-y\|_p = (\sum_j |x_j-y_j|^p)^{1/p}$ generalizes L1/L2.
- Cosine distance: $1 - \frac{x\cdot y}{\|x\| \|y\|}$. Useful for text embeddings where direction matters.
- Mahalanobis: $\sqrt{(x-y)^T S^{-1} (x-y)}$, accounts for covariance between features.
- Hamming: for categorical/binary features (count of mismatches).

Choose Euclidean for typical continuous features after scaling; Cosine for sparse/text vectors; Mahalanobis when covariance structure matters.

## Feature Scaling & Preprocessing

KNN is distance-based, so feature scale matters.

- Standardization (z-score): $x' = (x-\mu)/\sigma$.
- Min-Max scaling: $x' = (x - \min)/(\max-\min)$ rescales to $[0,1]$.
- Robust scaling: subtract median and scale by IQR to mitigate outliers.

Categorical features:

- One-hot encoding: converts categories to orthogonal binary features (increases dimensionality).
- Embeddings / ordinal encoding: use if ordinal relationships exist.
- Use mixed-distance functions or specialized models for mixed data.

Missing values:

- Impute before KNN (mean/mode, iterative imputation). Beware: imputing improperly can distort distances.

Feature weighting:

- Assign weights to features by domain knowledge or via learned metric (metric learning) to emphasize important features.

## KNN Imputer (Missing-value imputation)

KNN can be used to impute missing values by using the non-missing features to find similar rows and then aggregating the neighbors' values for the missing entry.

Algorithm (high level):

- For each sample with a missing entry, compute distances to other samples using only the features that are present in both samples (or use a metric that handles NaNs).
- Select the $k$ nearest neighbors among samples having a non-missing value for the target feature.
- Impute the missing value by aggregating neighbors (numeric: mean or weighted mean; categorical: mode or weighted vote).

Key parameters and implementation notes:

- `n_neighbors` ($k$): number of neighbors used for imputation.
- `weights`: `uniform` or `distance` (distance weighting often preserves local structure better).
- `metric`: distance metric; use scaled features (StandardScaler) before imputation. Scikit-learn's `KNNImputer` uses a `nan_euclidean` metric by default.
- `add_indicator`: whether to add a boolean indicator column for missingness (helps models detect missingness patterns).
- Be careful of data leakage: perform imputation inside cross-validation folds or inside a pipeline so test data statistics are not used to impute training data.

Scikit-learn example:

```python
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Pipeline ensures scaling and imputation occur within CV folds correctly
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("imputer", KNNImputer(n_neighbors=5, weights='distance'))
])

X_imputed = pipe.fit_transform(X_train)
```

Pros:

- Preserves local structure in the data and often produces more realistic imputations than global mean/median.

Cons / caveats:

- Computationally expensive on large datasets.
- Sensitive to scaling and irrelevant features — scale and/or select features first.
- Can unintentionally leak information from the target if target-derived features are present; avoid using target in imputation for training data.

Small numeric example:

Suppose we have three samples and two features, where `NaN` is missing:

| id  | f1  | f2  |
| --- | --- | --- |
| 1   | 1.0 | 2.0 |
| 2   | NaN | 2.5 |
| 3   | 1.2 | 2.1 |

To impute `f1` for sample 2 with $k=2$ and uniform weights, compute distances using `f2` only (since `f1` is missing for sample 2):

- Dist(sample2, sample1) = |2.5 - 2.0| = 0.5
- Dist(sample2, sample3) = |2.5 - 2.1| = 0.4

Neighbors sorted: sample3 (0.4), sample1 (0.5). Impute `f1` as mean of neighbors: $(1.2 + 1.0)/2 = 1.1$.

Best practices:

- Always scale numeric features before KNN imputation.
- Use `add_indicator=True` if missingness itself is informative.
- Impute inside training folds to avoid leakage.
- Compare KNN imputation against simpler strategies (mean/mode) using hold-out validation where you artificially mask values to measure imputation error.

## Choosing $k$ (Number of Neighbors)

- Small $k$ (e.g., $k=1$): low bias, high variance, sensitive to noise/outliers.
- Large $k$: smoother decision boundaries, higher bias, lower variance.

Heuristics:

- $k \approx \sqrt{n}$ is a common starting point.
- Choose odd $k$ for binary classification to avoid ties.
- Use cross-validation (grid search) to tune $k$, often together with metric and weights.

Tie-breaking strategies:

- Use the class of the nearest neighbor among tied groups.
- Use cumulative distances (choose class with smallest summed distance).
- Increase $k$ by 1 to break ties when sensible.

## Weighted KNN

- Distance-weighted voting reduces the influence of far neighbors.
- Common weight functions:
  - Inverse distance: $w(d)=1/(d+\varepsilon)$
  - Gaussian kernel: $w(d)=e^{-d^2/(2\sigma^2)}$
- Weighted KNN often improves robustness for noisy neighbors.

## KNN for Regression

- Predict average (or weighted average) of neighbors' target values.
- Use the same considerations: scaling, $k$ selection, weighting, and distance metric.

Evaluation metrics

- Classification: accuracy.
- Regression: MSE.
- Use stratified CV for imbalanced classes.

## Complexity and Performance

- Naive prediction: $O(n \cdot d)$ per query, where $n$ is training size and $d$ is feature dimension.
- Memory: $O(n \cdot d)$ to store training data.

Indexing structures to accelerate neighbor search:

- [KD-Tree](https://www.youtube.com/watch?v=Glp7THUpGow): efficient for low-dimensional continuous data (typically $d \lesssim 20$); binary space partitioning.
- Ball Tree: good for slightly higher dimensions and different metrics.
- Cover Trees / VP-Trees: for some metric spaces.
- Locality Sensitive Hashing (LSH): approximate for high-dim, large datasets.
- Annoy / Faiss / HNSW: approximate nearest neighbor libraries suitable for very large datasets and embeddings.

Trade-offs:

- Exact nearest neighbor with KD/Ball trees degrades in high dimensions.
- ANN gives large speedups with controlled accuracy loss.

## Curse of Dimensionality

- As $d$ increases, distances become less informative: distances between points concentrate.
- KNN performance often degrades with high $d$.
- Mitigations:
  - Dimensionality reduction: PCA, Truncated SVD (for sparse), UMAP, t-SNE (visualization only).
  - Feature selection: remove irrelevant/noisy features.
  - Metric learning: learn a projection/metric to make neighbors meaningful.

## Practical Tips

- Always scale continuous features.
- Try different distance metrics; use Cosine for sparse/text vectors.
- Use cross-validation to tune `k`, `weights`, and `metric`.
- If dataset large, consider ANN libraries (Faiss, Annoy, NMSLIB, HNSW).
- For mixed numerical and categorical data, consider specialized distances or transform categories carefully.
- If memory critical, use condensed or edited nearest neighbor variants.
- Use neighbor class proportions to produce calibrated probability estimates.

## Dealing with Imbalanced Data

- Use class weighting or distance weighting to emphasize minority class neighbors.
- Oversample minority class (SMOTE) or undersample majority class prior to KNN.
- Use stratified CV and metrics like precision-recall or F1.

## Extensions & Variants

- Condensed Nearest Neighbor: find subset of training data to reduce storage while maintaining performance.
- Edited Nearest Neighbor (ENN): remove noisy instances.
- KD-trees / Ball trees / vantage-point trees: indexing methods for speed.
- Metric learning (e.g., LMNN, triplet loss): learn a distance that makes neighbors more meaningful.
- Probabilistic KNN: model-based smoothing of neighbor votes.

## Implementation Considerations

- Use pipelines to combine scaling + KNN to avoid data leakage.
- Cache neighbor indices for repeated queries when the dataset is static.
- For time-series or structured data, use domain-specific distances (e.g., DTW for time series).

## Pseudocode (Classification)

```text
function KNN_predict(X_train, y_train, x_query, k, metric, weight_fn=None):
    distances = [ metric(x_query, x_i) for x_i in X_train ]
    neighbors_idx = argsort(distances)[:k]
    if weight_fn is None:
        votes = tally(y_train[i] for i in neighbors_idx)
    else:
        votes = defaultdict(0)
        for i in neighbors_idx:
            w = weight_fn(distances[i])
            votes[y_train[i]] += w
    return argmax(votes)
```

## Scikit-learn Examples

Classification (with pipeline and GridSearchCV):

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier())
])

param_grid = {
    'knn__n_neighbors': [3,5,7,9],
    'knn__weights': ['uniform', 'distance'],
    'knn__metric': ['euclidean', 'manhattan', 'minkowski']
}

grid = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)
print(grid.best_params_, grid.best_score_)
```

Regression:

```python
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsRegressor())
])
pipe.fit(X_train, y_train)
```

Using KDTree / BallTree directly for speed:

```python
from sklearn.neighbors import KDTree
import numpy as np

tree = KDTree(X_train, leaf_size=40)
dists, idx = tree.query(X_query, k=5)
# then aggregate y_train[idx] for predictions
```

## Memory and Storage

- Storing the raw training data is required for exact KNN.
- For large persistent datasets, consider disk-backed or memory-mapped representations.
- ANN libraries trade storage and indexing time for query speed.

## Interpretability

- KNN is transparent: predictions can be explained by listing nearby training examples.
- Useful for case-based reasoning or systems where showing nearest examples is valuable.

## Use Cases and Examples

- Recommender systems (simple item-based recommendations using similarity of feature vectors).
- Anomaly detection (distance to k neighbors as an anomaly score).
- Geospatial problems where proximity implies similarity.
- Image or embedding-based retrieval using Cosine or Euclidean distances on learned embeddings.
- Baseline classification/regression where domain knowledge is limited.

## Common Pitfalls

- Forgetting to scale features leading to dominated distances.
- Using KNN in very high dimensions without dimensionality reduction.
- Using Euclidean distance on categorical features encoded with one-hot without considering sparsity and dimensional blow-up.
- Not accounting for class imbalance when interpreting accuracy.

## Checklist Before Using KNN

- [ ] Features scaled appropriately (StandardScaler/MinMax).
- [ ] Choose/experiment with distance metric suitable for data type.
- [ ] Tune k via cross-validation.
- [ ] Consider weighting neighbors by distance.
- [ ] If dataset large, evaluate ANN or indexing methods.
- [ ] If high-dimensional, apply PCA/feature selection or metric learning.

## Short Comparative Summary

- Pros: simple, interpretable, flexible, non-parametric, good baseline.
- Cons: high prediction cost, memory heavy, sensitive to irrelevant features and scaling, struggles in high dimensions.

## References & Further Reading

- Cover, T., & Hart, P. (1967). Nearest neighbor pattern classification.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning.
- Scikit-learn documentation: neighbors — https://scikit-learn.org/stable/modules/neighbors.html
- Indyk, P., & Motwani, R. (1998). Approximate nearest neighbors: toward removing the curse of dimensionality.

---
