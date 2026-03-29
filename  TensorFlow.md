## Introduction to TensorFlow + Keras (for SKLearn/ML students)

Great news: this is a super relevant topic and your baseline (sklearn/ML models) is perfect.  
I’ll give you a clear, structured lecture in English with theory + practical code concepts and GPU/TPU optimization, including logistic regression in TensorFlow for image classification.

---

## 1. Overview

- Goal: move from sklearn mindset to TensorFlow/Keras mindset
  - Add text: the goal is to show how your existing machine learning intuition maps directly to TensorFlow. In sklearn you called `.fit()` on pre-built estimators; in TF you explicitly build tensors and training loops (or use Keras convenience methods) while keeping the same core concepts: data, model, loss, optimizer, metrics.

- Audience: knows sklearn pipelines, linear/logistic models, train/test, metrics
  - Add text: this lesson assumes you already understand `train_test_split`, feature scaling, and evaluation measures like accuracy, precision, recall. We will build on that so you can quickly adapt to deep learning syntax.

- Focus:
  1. TensorFlow concepts (tensors, graphs/eager execution)
     - Add text: explain that a tensor is a multidimensional numeric array, similar to `numpy` but with support for device placement. Graph mode is how TF used to work by compiling a symbolic graph; eager mode is now default so it behaves like regular Python.
  2. Keras high-level API (Sequential, fit, evaluate, predict)
     - Add text: Keras wraps common workflows into a clean sequence. `Sequential` is for simple stack models, and common methods let you train and assess with just a few lines of code.
  3. Logistic regression in TF, applied to images
     - Add text: logistic regression is the simplest neural model, mapping inputs through a weighted sum and activation (sigmoid/softmax). It is a good “first step” before CNNs.
  4. GPU/TPU optimization basics
     - Add text: over time we’ll add one section that shows how easy it is for TF to use higher compute devices and how that changes the pipeline.
  5. Quick demo plan + key tips
     - Add text: at the end we provide a runnable code block students can paste and try.

---

## 2. Why TensorFlow/Keras when you already know sklearn?

- sklearn:
  - CPU-based, great for moderate-data tabular tasks
    - Add text: sklearn models run on a single CPU core unless you explicitly parallelize across data folds; they are great for CSV-like data and well-structured inputs.
  - limited deep learning support, no automatic GPU
    - Add text: sklearn has no built-in support for convolutional or recurrent layers and cannot tap into GPU acceleration to speed up large networks.

- TensorFlow/Keras:
  - automatic differentiation + GPU/TPU support
    - Add text: TF computes gradients for you with `tf.GradientTape` or Keras `fit`, so you don't have to derive backprop by hand. Hardware acceleration is automatic where available.
  - better for images, audio, text, sequences
    - Add text: TF provides dedicated layer types and preprocessing tools for different data modalities.
  - production-ready, model export, distributed training
    - Add text: TF apps can be saved in `SavedModel` format and served in cloud environments, with built-in distribution strategies.

- Analogy:
  - in sklearn, `LogisticRegression` is boiled down to weights + sigmoid
  - in TF, same math, but you can express as tensor operations and easily extend to CNNs/DNNs
    - Add text: this analogy shows that TF is not a completely new algorithm — it is a flexible framework that can represent the same computation plus more.

---

## 3. Core TensorFlow concepts in 5 minutes

- Tensor: N-D array, similar to numpy but can be on CPU/GPU/TPU
  - Add text: tensors carry a shape and dtype, e.g., `tf.Tensor([[1.,2.],[3.,4.]])` is a rank-2 tensor. They are immutable and can be moved between devices.

- `tf.Variable` = trainable weights
  - Add text: variables hold state that persists across training steps and is updated by optimizers.

- `tf.Tensor` = data value
  - Add text: a plain tensor can be an input batch, intermediate activation, or loss; it is not directly mutable.

- `tf.data.Dataset`: pipeline for input, batch, shuffle, performance
  - Add text: `Dataset` is the recommended way to feed data into models; it supports streaming from disk, shuffling, batching, and prefetching.

- Execution modes:
  - Eager: Python-like execution, easy debugging
    - Add text: eager mode runs step-by-step and prints values instantly.
  - Graph (function decorator): compilation + speed
    - Add text: `@tf.function` wraps code into a graph for better performance at scale.

---

## 4. Keras API quick structure

- Models:
  - `tf.keras.Sequential`
    - Add text: easiest API for layer stacks, where each layer has one input and one output.
  - Functional API (for custom graphs)
    - Add text: use the functional API with `Input(...)` for branching models or models with multiple inputs/outputs.

- Core objects:
  - `model.compile(optimizer, loss, metrics)`
    - Add text: this configures training behavior. Optimizer can be `Adam`, `SGD`, etc., and loss can be crossentropy, MSE, etc.
  - `model.fit(train_ds, validation_data=val_ds, epochs=...)`
    - Add text: this runs training for many epochs and reports metrics per step.
  - `model.evaluate(test_ds)`
    - Add text: evaluate measures losses/metrics on holdout data.
  - `model.predict(images)`
    - Add text: returns model outputs for inference.

- Data prep:
  - load dataset, normalize, one-hot labels or sparse
    - Add text: for images, scale values to [0,1]; for categorical targets, use `SparseCategoricalCrossentropy` with integer labels or `CategoricalCrossentropy` with one-hot labels.

---

## 5. Logistic Regression in TensorFlow (math refresher)

- Formula:
  - logits = `W*x + b`
    - Add text: this is the linear part. For images + flatten, `x` is a vector and `W` is a matrix mapping features to classes.
  - probability = `sigmoid(logits)` for binary, `softmax(logits)` for multi-class
    - Add text: sigmoid squeezes outputs to [0,1] for 1 vs all; softmax distributes probability mass across classes.

- Loss:
  - binary: `tf.keras.losses.BinaryCrossentropy(from_logits=True)`
    - Add text: `from_logits=True` means you pass raw scores and TF applies sigmoid inside the loss with numeric stability.
  - multiclass: `tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)`
    - Add text: this is the proper loss for integer labels and raw logits for multiple categories.

---

## 6. Example: MNIST/CIFAR-like image classification with logistic regression

### 6.1 Data prep

- use tf.keras datasets: `mnist` or `cifar10`
  - Add text: these datasets are built-in and good for teaching. `mnist` has 28x28 grayscale digits; `cifar10` has 32x32 RGB objects.
- normalize pixel values `x/255.0`
  - Add text: normalization helps gradient descent converge faster and prevents large learned weights.
- flatten: `tf.keras.layers.Flatten(input_shape=(28,28))`
  - Add text: flatten converts each image to a vector so the dense layer applies a linear transformation across all pixels.

### 6.2 Model

- `tf.keras.Sequential([Flatten, Dense(num_classes, activation='softmax')])`
  - Add text: this is effectively softmax regression (multiclass logistic regression). It has no hidden layers, so it learns only linear decision boundaries.

### 6.3 Compile

- `optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)`
  - Add text: Adam is popular because it adapts the learning rate per parameter, which usually works well out-of-box.
- `loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)` (since softmax output)
  - Add text: with softmax output, we set `from_logits=False`; TF uses the provided probabilities directly.
- `metrics=['accuracy']`
  - Add text: accuracy is easy to interpret for classification.

### 6.4 Fit

- use `model.fit(train_ds, validation_data=val_ds, epochs=5)`
  - Add text: start with 5 epochs and monitor whether loss decreases; more epochs can improve final accuracy.

### 6.5 Evaluate and sample predict

- `model.evaluate(test_ds)`
  - Add text: evaluate returns loss and accuracy; use this as your final model report.
- `model.predict(sample_images)`
  - Add text: predictions are probabilities; convert with `tf.argmax` for class labels.

---

## 7. Code snippet (complete, concise) [Colab Link](https://colab.research.google.com/drive/1iluaZnkzsRO94_5vjsyGvTdpvD-hu9Tz?usp=sharing)
```python
import tensorflow as tf

# 1. load
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# 2. dataset pipeline
batch_size = 128
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)

# 3. logistic regression model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

# 4. train
model.fit(train_ds, validation_data=test_ds, epochs=5)

# 5. evaluate
test_loss, test_acc = model.evaluate(test_ds)
print("test accuracy", test_acc)

# 6. inference
probs = model.predict(x_test[:5])
preds = tf.argmax(probs, axis=1).numpy()
print("preds", preds, "true", y_test[:5])
```

- Add text: this walk-through is intentionally linear and replicates the sklearn pattern: load, preprocess, define model, train, evaluate, predict.

---

## 9. GPU / TPU optimization (key principles)

### 9.1 GPU:

- install CUDA + cuDNN matching TF version
  - Add text: mismatch between TensorFlow and CUDA versions is a common source of failure.
- verify with:
  - `tf.config.list_physical_devices('GPU')`
    - Add text: if this returns an empty list, GPU is not available in your environment.
- set memory growth:

```python
gpus = tf.config.list_physical_devices('GPU')
for g in gpus: tf.config.experimental.set_memory_growth(g, True)
```

- Add text: this avoids TensorFlow pre-allocating all GPU memory and helps run side-by-side processes.
- use accelerated data pipeline:
  - `ds.cache().prefetch(tf.data.AUTOTUNE)`
  - `interleave`, `map(..., num_parallel_calls=tf.data.AUTOTUNE)`
- batch size: bigger yields better throughput (commonly 128, 256)
  - Add text: evaluate GPU memory and adjust batch size until utilization is strong but no OOM.

### 9.2 TPU (Cloud or Colab TPU):

- connect:
  - `resolver = tf.distribute.cluster_resolver.TPUClusterResolver()`
  - `tf.config.experimental_connect_to_cluster(resolver)`
  - `tf.tpu.experimental.initialize_tpu_system(resolver)`
- strategy:
  - `strategy = tf.distribute.TPUStrategy(resolver)`
  - with `strategy.scope():` build + compile model
- use `tf.data` with `ds = ds.batch(global_batch_size, drop_remainder=True)`

- Add text: TPU uses a global batch size across replicas, so adjust accordingly and keep drop_remainder=True for static shape.

### 9.3 Loss scaling + mixed precision

- enable mixed precision:
  - `tf.keras.mixed_precision.set_global_policy('mixed_float16')`
- use compatible optimizers (`Adam`, etc) -> TF handles loss scaling automatically
- good for modern GPU (A100, V100) and TPU.
- Add text: mixed precision can double throughput. Monitor numeric stability.

---

## 10. Practical teaching flow (how to deliver)

1. Start with review: logistic regression in sklearn:
   - `LinearModel` -> probabilities -> `predict_proba`
   - training via gradient descent/closed-form.
   - Add text: remind students how weights are fit and how decision boundaries are formed.
2. Introduce tensor concept, difference graph vs eager.
   - Add text: show direct mapping from numpy arrays to tf tensors
3. Live code: run the MNIST logistic TF snippet.
   - Add text: let students modify epochs and observe training history.
4. Show accuracy (typically ~92% with logistic vs 98% with CNN).
   - Add text: this concrete comparison helps them understand model capacity.
5. Add one conv layer and rerun; show improvement.
   - Add text: highlight architectural changes while keeping training code same.
6. Show `tf.config.list_physical_devices()`, optional GPU demo.
   - Add text: live hardware check reinforces practical setup knowledge.
7. Sum up: TensorFlow is ML from scratch + scale; Keras is easy API.
   - Add text: emphasize confidence raise for working on production workflows.

---

## 11. FAQ / decisions to ask you

- Do you prefer using `MNIST` or `CIFAR-10` for your class (MNIST is simpler; CIFAR more realistic)?
  - Add text: ask students which dataset feels more meaningful — digit classification is easy, object classification is more like real-world tasks.
- Will students run on CPU-only machine, or do you want a live GPU demo for code?
  - Add text: CPU-only is sufficient for learning but GPU/TPU demos speed experiments.
- Should I include a short comparison chart: sklearn vs TF/Keras on logistic regression + images?
  - Add text: a chart helps visual learners quickly spot strengths/limitations.

- Additional resources:
  - URL: https://www.tensorflow.org/tfx/model_analysis/metrics#multi-classmulti-label_classification_metrics
  - Add text: this page explains multi-class/multi-label classification metrics in TensorFlow Model Analysis (TFMA), useful for real-world ML model evaluation.

---
