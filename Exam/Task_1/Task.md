# Final Exam: Retail Data Pipeline & Database Architecture
**Duration:** 90 Minutes  
**Total Points:** 100  

## Overview
You are tasked with building a backend system for a retail company. You have been given two messy CSV files containing product information and inventory logs. Your goal is to clean this data using **Pandas**, migrate it to a **SQLite** database with a 3-table architecture, and implement automated database logic.

---

## Part 1: Data Cleaning (Pandas) 
**File inputs:** `products_raw.csv` and `inventory_levels.csv`

1.  **Products Data:**
    * Load `products_raw.csv`.
    * **Missing Values:** Fill any missing values with the **median** or **mean**
2.  **Inventory Data:**
    * Load `inventory_levels.csv`.
    * **Missing Values:** Fill any missing values with the median or mean.
3.  **Merge:**
    * Perform a validation check to ensure every `ProductID` in the inventory list exists in the products list.

---

## Part 2: Database Design (SQLite) 
Create a SQLite database named `RetailSystem.db`. Design the following **three tables** with appropriate data types and Primary/Foreign Keys.

1.  **`Products`**
    * `ProductID` (Primary Key, Integer)
    * `ProductName` (Text)
    * `Category` (Text)
    * `Price` (Real)

2.  **`Inventory`**
    * `InventoryID` (Primary Key, Integer)
    * `ProductID` (Foreign Key referencing Products)
    * `WarehouseCode` (Text)
    * `StockLevel` (Integer)

3.  **`Sales`**
    * `SaleID` (Primary Key, Auto-increment)
    * `ProductID` (Foreign Key referencing Products)
    * `QuantitySold` (Integer)
    * `SaleDate` (Text, Defaults to Current Timestamp)

---

## Part 3: Python Integration 
Write a Python script to:
1.  Create to `RetailSystem.db`.
2.  Insert your cleaned **Products** dataframe into the `Products` table.
3.  Insert your cleaned **Inventory** dataframe into the `Inventory` table.

---

## Part 4: Database Logic (SQL) 
You must execute these SQL commands within your Python script or a separate SQL file.

### A. Automation (Trigger)
Create a Trigger named `UpdateStockAfterSale`.
* **Logic:** When a new record is inserted into the **`Sales`** table, the trigger must automatically **subtract** the `QuantitySold` from the `StockLevel` in the **`Inventory`** table for the corresponding `ProductID`.

### B. Reporting (View)
Create a View named `CategoryRevenueSummary`.
* **Logic:** Join the `Products` and `Inventory` tables.
* **Output:** Group by `Category` and show the **Total Potential Revenue** (Sum of `Price` * `StockLevel`) for each category.

### C. Complex Update
* **Task:** Management wants to clear out low-stock items from Warehouse 'WH-A'.
* **Query:** Write a SQL query to decrease the `Price` by **20%** for all products that are located in 'WH-A' AND have a `StockLevel` lower than 40.

---

## Submission Requirements
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