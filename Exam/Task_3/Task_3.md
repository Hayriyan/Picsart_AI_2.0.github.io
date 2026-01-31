# Task 3: Employee Management & Payroll Data Pipeline

**Duration:** 90 Minutes  
**Total Points:** 100

---

## Overview

Build a **backend data pipeline** for an HR management system using **Pandas**, **SQLite**, and **SQL**.

You are given two noisy CSV files in `Data/Raw/`:
- `employees_noisy.csv`
- `payroll_noisy.csv`

Your tasks include **data cleaning**, **database design**, and **SQL logic implementation**.  
No frontend or API is required.

---

## Part 1: Data Cleaning & Validation (Pandas)

### Input Files
`employees_noisy.csv`, `payroll_noisy.csv`

---

### 1. Employees

**Columns:** `EmployeeID`, `FirstName`, `LastName`, `Department`, `HireDate`, `Salary`

**Rules:**
- Drop rows where `EmployeeID` is missing or `-1`
- Drop rows where both `FirstName` and `LastName` are missing
- `Department`:
  - Standardize to title case (e.g., "engineering", "ENGINEERING" → "Engineering")
  - Replace missing values with "Unknown"
- `Salary`:
  - Convert to numeric
  - Replace negative values and zero values with `NaN`
  - Remove outliers: values exceeding 500000 (set to `NaN`)
  - Compute median salary from valid (non-NaN) salaries only
  - Fill `NaN` with computed median salary
- `HireDate`:
  - Convert to datetime format (YYYY-MM-DD)
  - Invalid = cannot be parsed OR HireDate > today's date
  - Drop rows where `HireDate` is missing or invalid

**Output:**
```
Data/Processed/employees_clean.csv
```

---

### 2. Payroll

**Columns:** `PayrollID`, `EmployeeID`, `PayPeriod`, `HoursWorked`, `Bonus`, `Deductions`

**Rules:**
- Drop rows where `PayrollID` or `EmployeeID` is missing or `-1`
- `HoursWorked`:
  - Convert to numeric
  - Replace negative values with `NaN`
  - Fill `NaN` with mean value
  - Cap maximum at 168 hours per pay period (bi-weekly pay cycle)
- `PayPeriod`:
  - Standardize format to `YYYY-MM-DD` (start date of bi-weekly period)
  - If PayPeriod is missing or invalid, drop the row
- `Bonus`:
  - Convert to numeric
  - Replace negative values with 0
  - Fill missing values with 0
- `Deductions`:
  - Convert to numeric
  - Replace negative values with `NaN`
  - Fill `NaN` with median value
  - After merging payroll with cleaned employees (on `EmployeeID`), ensure `Deductions` does not exceed employee `Salary`
  - If `Deductions > Salary`, set `Deductions = Salary` (cap at salary amount)

**Output:**
```

Data/Processed/payroll_clean.csv

```

---

### 3. Referential Integrity Validation & Final Cleanup

- Merge payroll data with cleaned employees on `EmployeeID` (inner join)
- Drop payroll records whose `EmployeeID` does not exist in cleaned employees
- Apply deductions cap check (see rule above)
- SQLite will also enforce foreign key constraints (enabled in Part 3), but Pandas validation ensures clean data insertion

**Outputs:**
```

Data/Processed/payroll_validated.csv
```


---

## Part 2: Database Design (SQLite)

Create database:
```

Data/DB/EmployeeManagement.db

```

### Tables

**Employees**
- `EmployeeID` (PK, INTEGER, NOT NULL)
- `FirstName` (TEXT, NOT NULL)
- `LastName` (TEXT, NOT NULL)
- `Department` (TEXT, NOT NULL)
- `HireDate` (TEXT, NOT NULL, format: YYYY-MM-DD)
- `Salary` (REAL, NOT NULL, CHECK (Salary > 0 AND Salary <= 500000))

**Payroll**
- `PayrollID` (PK, INTEGER, NOT NULL)
- `EmployeeID` (FK → Employees.EmployeeID, INTEGER, NOT NULL)
- `PayPeriod` (TEXT, NOT NULL, format: YYYY-MM-DD)
- `HoursWorked` (REAL, NOT NULL, CHECK (HoursWorked >= 0 AND HoursWorked <= 168))
- `Bonus` (REAL, NOT NULL, CHECK (Bonus >= 0))
- `Deductions` (REAL, NOT NULL, CHECK (Deductions >= 0))

**SalaryHistory**
- `HistoryID` (PK, INTEGER, AUTOINCREMENT)
- `EmployeeID` (FK → Employees.EmployeeID, INTEGER, NOT NULL)
- `OldSalary` (REAL)
- `NewSalary` (REAL, NOT NULL)
- `ChangeDate` (TEXT, DEFAULT CURRENT_TIMESTAMP, NOT NULL)

(Tables may be created via SQL or Python. All constraints must be enforced.)

---

## Part 3: Python Integration

Create `Src/task3_pipeline.py` that:
1. Cleans raw CSVs (Part 1)
2. Creates the SQLite database and tables (Part 2)
3. **Enable foreign key constraints** by executing `PRAGMA foreign_keys = ON;` after connecting to the database
4. Inserts cleaned employees and validated payroll records
5. Verify foreign key constraints are working (attempt to insert invalid `EmployeeID` in Payroll should fail)

**Note:** SQLite foreign keys are **NOT** enabled by default. You must explicitly enable them for referential integrity enforcement.

Script should run end-to-end without CLI arguments.

---

## Part 4: Database Logic (SQL)

### A. Trigger – Salary Change Log

Create trigger `LogSalaryChange`:
- Fires **AFTER UPDATE OF Salary** on `Employees`
- **Only logs when salary actually changes:** `WHEN OLD.Salary IS NOT NEW.Salary AND NEW.Salary IS NOT NULL`
- Inserts `EmployeeID`, `OldSalary`, `NewSalary`, `ChangeDate` into `SalaryHistory`
- Should not log when salary is updated to the same value or to NULL

---

### B. View – Department Payroll Summary

Create view `DepartmentPayrollSummary`:
- Join `Employees` and `Payroll` on `EmployeeID`
- Group by `Department`
- **Important:** Since one employee can have multiple payroll records, use `DISTINCT` or subquery to avoid salary duplication
- Columns:
  - `Department`
  - `TotalEmployees` (count of **distinct** `EmployeeID`)
  - `TotalHoursWorked` (sum of `HoursWorked` from Payroll)
  - `TotalBonus` (sum of `Bonus` from Payroll)
  - `AverageSalary` (average of **distinct** salaries from Employees, using `AVG(DISTINCT Employees.Salary)` or a subquery to calculate from Employees table only)

---

### C. Complex Update Query with Verification

**Task:** Management wants to give a salary increase to long-term employees.

**Requirements:**
1. Write a SQL query (using a transaction) to increase the `Salary` by **10%** for all employees in the "Engineering" department who were hired before 2020-01-01 AND have a current salary less than 100000.
2. **Verification:** After the update, verify that:
   - The trigger `LogSalaryChange` fired correctly
   - Query the `SalaryHistory` table to show the salary changes (old vs new salary)
   - Count how many employees received the raise
3. Use a transaction to ensure atomicity (BEGIN TRANSACTION; UPDATE; SELECT verification; COMMIT;)

**Expected Output:**
- Number of employees who received the raise
- List of affected employees showing old and new salaries from `SalaryHistory`

---

## Submission Structure

```
├── .venv/ (virtual environment)
├── Src/ (source code)
├── Data/
│   ├── DB/ (database files)
│   ├── Processed/ (cleaned data)
│   └── Raw/ (raw data files)
└── notebook/ (Jupyter notebooks for exploration)
```
**Good Luck!**

