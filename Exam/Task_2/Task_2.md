
# Task 2: Customer Orders Data Pipeline & Reporting

**Duration:** 90 Minutes  
**Total Points:** 100  

---

## Overview

Build a **backend data pipeline** for an online store using **Pandas**, **SQLite**, and **SQL**.

You are given two noisy CSV files in `Data/Raw/`:
- `customers_noisy.csv`
- `orders_noisy.csv`

Your tasks include **data cleaning**, **database design**, and **SQL logic implementation**.  
No frontend or API is required.

---

## Part 1: Data Cleaning & Validation (Pandas)

### Input Files
`customers_noisy.csv`, `orders_noisy.csv`

---

### 1. Customers

**Columns:** `CustomerID`, `FirstName`, `LastName`, `Email`, `JoinDate`

**Rules:**
- Drop rows where `CustomerID` is missing or `-1`
- Drop rows where both `FirstName` and `LastName` are missing
- Convert `Email` to lowercase
- Drop rows where `Email` is missing

**Output:**
```

Data/Processed/customers_clean.csv

```

---

### 2. Orders

**Columns:** `OrderID`, `CustomerID`, `OrderDate`, `OrderAmount`, `Status`

**Rules:**
- Drop rows where `OrderID` or `CustomerID` is missing or `-1`
- `OrderAmount`:
  - Convert to numeric
  - Replace negative values with `NaN`
  - Fill `NaN` with median value
- `Status`:
  - Allowed: `Pending`, `Shipped`, `Cancelled`
  - Others → `Pending`

**Output:**
```

Data/Processed/orders_clean.csv

```

---

### 3. Orphan Orders Validation

- Drop orders whose `CustomerID` does not exist in cleaned customers

**Outputs:**
```

Data/Processed/orders_validated.csv
```


---

## Part 2: Database Design (SQLite)

Create database:
```

Data/DB/CustomerOrders.db

```

### Tables

**Customers**
- `CustomerID` (PK)
- `FirstName`
- `LastName`
- `Email` (UNIQUE)
- `JoinDate`

**Orders**
- `OrderID` (PK)
- `CustomerID` (FK → Customers)
- `OrderDate`
- `OrderAmount`
- `Status`

**OrderStatusLog**
- `LogID` (PK, AUTOINCREMENT)
- `OrderID`
- `OldStatus`
- `NewStatus`
- `ChangeDate` (DEFAULT CURRENT_TIMESTAMP)

(Tables may be created via SQL or Python.)

---

## Part 3: Python Integration

Create `Src/task2_pipeline.py` that:
1. Cleans raw CSVs (Part 1)
2. Creates the SQLite database and tables
3. Inserts cleaned customers and validated orders

Script should run end-to-end without CLI arguments.

---

## Part 4: Database Logic (SQL)

### A. Trigger – Status Change Log

Create trigger `LogOrderStatusChange`:
- Fires **AFTER UPDATE OF Status** on `Orders`
- Inserts `OrderID`, `OldStatus`, `NewStatus`, `ChangeDate` into `OrderStatusLog`

---

### B. View – Customer Revenue Summary

Create view `CustomerRevenueSummary`:
- Join `Customers` and `Orders`
- Group by `CustomerID`
- Columns:
  - `CustomerID`
  - `FullName` (`FirstName || ' ' || LastName`)
  - `TotalOrders`
  - `TotalRevenue` (sum of `OrderAmount` where `Status = 'Shipped'`)

---

## Submission Structure

```
├── .venv/ (virtual environment)
├── Src/ (source code)
├── Data/
│   ├── DB/ (database files)
│   ├── Processed/ (cleaned data)
│   └── Row/ (analysis results)
└── notebook/ (Jupyter notebooks for exploration)
```
**Good Luck!**
