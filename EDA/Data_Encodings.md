# Data Encodings in Machine Learning

---

## Introduction to Encodings 

- **Definition:** Encoding converts categorical or text data into numerical form so machine learning models can process it.
- **Why needed:** Most ML models (e.g., linear regression, neural networks) require numeric inputs.
- **Common types:** **Label**, **One‑Hot**, **Ordinal**, **Binary**, **Target/Mean**.

---

## Label Encoding 

- **Concept:** Assign a unique integer to each category.

**Example:**

| Color | Encoded |
| ----- | ------- |
| Red   | 0       |
| Blue  | 1       |
| Green | 2       |

- **Advantages:** Simple, fast, memory-efficient.
- **Disadvantages:** Implies ordinal relationships that may not exist.
- **Tips / Use case:** Use for truly ordinal categories (e.g., Low/Medium/High) or with tree-based models that can handle integer categories.

---

## One‑Hot Encoding 

- **Concept:** Convert each category into a separate binary (0/1) column.

**Example:**

| Color | Red | Blue | Green |
| ----- | --- | ---- | ----- |
| Red   | 1   | 0    | 0     |
| Blue  | 0   | 1    | 0     |
| Green | 0   | 0    | 1     |

- **Advantages:** No false ordinal relationships; broadly supported.
- **Disadvantages:** Can cause high dimensionality when categories are many.
- **Tips / Use case:** Use for unordered categorical variables with relatively few unique values (e.g., gender, region).

---

## Ordinal Encoding 

- **Concept:** Assign integers based on an inherent order.

**Example (Size):** Small → 0, Medium → 1, Large → 2

- **Advantages:** Preserves natural order; compact.
- **Disadvantages:** Not suitable for unordered categories; model may treat differences as exact distances.
- **Tips / Use case:** Use when categories have meaningful order (e.g., satisfaction ratings, education levels).

---

## Binary Encoding ⚡

- **Concept:** Convert the label-encoded integer into binary digits, each bit as a separate column.
- **Steps:** Label encode → convert integer to binary → each binary digit → a column.

**Example:**

| Color | Label | Binary1 | Binary2 |
| ----- | ----- | ------- | ------- |
| Red   | 0     | 0       | 0       |
| Blue  | 1     | 0       | 1       |
| Green | 2     | 1       | 0       |

- **Advantages:** Reduces dimensionality versus one‑hot.
- **Disadvantages:** Slightly more complex; may confuse some models if not handled carefully.
- **Tips / Use case:** Use when many categories make one‑hot impractical.

---

## Target / Mean Encoding 

- **Concept:** Replace category with the mean (or statistic) of the target variable for that category.

**Example (house price):**

| Neighborhood | Avg Price |
| ------------ | --------- |
| A            | 300k      |
| B            | 450k      |

- **Advantages:** Captures direct relationship with target; compact.
- **Disadvantages:** High risk of data leakage; requires careful cross-validation and smoothing.
- **Tips / Use case:** Use for categories strongly correlated with the target; apply smoothing and CV; handle unseen categories with global prior.

---

## Choosing the Right Encoding — Quick Reference 

| Encoding Type | Advantages                   | Disadvantages                | Practical Tips                                              |
| ------------- | ---------------------------- | ---------------------------- | ----------------------------------------------------------- |
| Label         | Simple, fast                 | False ordinality             | Use for ordinal categories or tree-based models             |
| One‑Hot       | No false order               | High dimensionality          | Use for unordered categorical variables with few categories |
| Ordinal       | Preserves order              | Not for unordered categories | Use for ordered categories (ratings, size)                  |
| Binary        | Reduces dimensionality       | Slightly complex             | Use for many categories                                     |
| Target/Mean   | Captures target relationship | Risk of data leakage         | Use with strong correlation + cross-validation/smoothing    |

---

## Summary 

- Encoding transforms categorical/text data into numeric form.
- Choose encoding based on: number of categories, whether order matters, and risk of leakage.
- Correct encoding can notably improve model performance — always validate with proper CV and preprocessing.

---

_If you want, I can generate a PDF or a slide deck from these notes._
