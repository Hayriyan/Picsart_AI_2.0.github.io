# SQLite3 Complete Guide

## Table of Contents

1. [Introduction to SQLite3](#introduction-to-sqlite3)
2. [Basic Concepts](#basic-concepts)
3. [Connection Management](#connection-management)
4. [Creating Tables (Schema Design)](#creating-tables-schema-design)
5. [Inserting Data](#inserting-data)
6. [Querying Data](#querying-data)
7. [Updating and Deleting](#updating-and-deleting)
8. [Transactions and Commit](#transactions-and-commit)
9. [Foreign Keys and Relationships](#foreign-keys-and-relationships)
10. [Best Practices](#best-practices)
11. [Real-World Examples from SpaceX Project](#real-world-examples-from-spacex-project)

---

## Introduction to SQLite3

**SQLite3** is a lightweight, serverless, self-contained SQL database engine. Unlike MySQL or PostgreSQL, SQLite doesn't require a separate server process. The entire database is stored in a single file on disk.

### Why SQLite3?

- ✅ **Zero Configuration**: No server setup required
- ✅ **Portable**: Single file database
- ✅ **Fast**: Excellent for small to medium datasets
- ✅ **ACID Compliant**: Ensures data integrity
- ✅ **Built-in Python**: No additional installation needed (comes with Python)

### When to Use SQLite3

- Local data storage
- Development and testing
- Small to medium applications
- Embedded systems
- Data warehousing (like our SpaceX project)

---

## Basic Concepts

### Database File

A SQLite database is a single file (e.g., `spacex.db`). When you connect to it, SQLite creates the file if it doesn't exist.

### Tables

Tables store data in rows and columns, similar to Excel spreadsheets but with strict structure.

### SQL (Structured Query Language)

SQL is the language used to interact with databases. Common operations:

- **CREATE**: Define table structure
- **INSERT**: Add new rows
- **SELECT**: Retrieve data
- **UPDATE**: Modify existing data
- **DELETE**: Remove rows

---

## Connection Management

### Basic Connection

```python
import sqlite3

# Connect to database (creates file if it doesn't exist)
conn = sqlite3.connect("mydatabase.db")

# Always close the connection when done
conn.close()
```

### Connection Context Manager (Recommended)

```python
# Automatically closes connection, even if error occurs
with sqlite3.connect("mydatabase.db") as conn:
    # Do database operations
    pass
# Connection automatically closed here
```

### Our Project's Approach

In `Src/db.py`, we use a helper function:

```python
from pathlib import Path
import sqlite3

DB_PATH = Path("Data/DB/spacex.db")

def get_connection() -> sqlite3.Connection:
    # Create directory if it doesn't exist
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH)
```

**Why this approach?**

- Centralizes database path management
- Ensures directory structure exists
- Easy to change database location in one place

---

## Creating Tables (Schema Design)

### Basic CREATE TABLE Syntax

```python
conn = sqlite3.connect("example.db")
cur = conn.cursor()

cur.execute("""
    CREATE TABLE users (
        id INTEGER PRIMARY KEY,
        name TEXT,
        email TEXT
    )
""")

conn.commit()
conn.close()
```

### Column Types in SQLite

| SQLite Type | Python Type | Description     |
| ----------- | ----------- | --------------- |
| `INTEGER`   | `int`       | Whole numbers   |
| `REAL`      | `float`     | Decimal numbers |
| `TEXT`      | `str`       | Text strings    |
| `BLOB`      | `bytes`     | Binary data     |
| `NULL`      | `None`      | Missing value   |

### Constraints

**PRIMARY KEY**: Uniquely identifies each row

```sql
id INTEGER PRIMARY KEY
```

**AUTOINCREMENT**: Automatically increments the primary key

```sql
id INTEGER PRIMARY KEY AUTOINCREMENT
```

**UNIQUE**: Ensures no duplicate values

```sql
email TEXT UNIQUE
```

**NOT NULL**: Prevents NULL values

```sql
name TEXT NOT NULL
```

**FOREIGN KEY**: References another table

```sql
user_id INTEGER,
FOREIGN KEY (user_id) REFERENCES users(id)
```

### Our Project's Schema

From `Src/db.py`, we create three normalized tables:

```python
def create_schema(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()

    # Table 1: Rockets (Dimension Table)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS rockets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            spacex_id TEXT UNIQUE,
            name TEXT,
            type TEXT,
            active INTEGER
        )
    """)

    # Table 2: Launchpads (Dimension Table)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS launchpads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            spacex_id TEXT UNIQUE,
            name TEXT,
            region TEXT,
            latitude REAL,
            longitude REAL
        )
    """)

    # Table 3: Launches (Fact Table)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS launches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            spacex_id TEXT UNIQUE,
            name TEXT,
            date_utc TEXT,
            success INTEGER,
            rocket_id INTEGER,
            launchpad_id INTEGER,
            details TEXT,
            FOREIGN KEY (rocket_id) REFERENCES rockets(id),
            FOREIGN KEY (launchpad_id) REFERENCES launchpads(id)
        )
    """)

    conn.commit()
```

**Key Design Decisions:**

1. **Normalized Structure**: Separate rockets/launchpads from launches (avoids data duplication)
2. **Internal IDs**: Auto-incrementing integers for efficient joins
3. **SpaceX IDs**: Store original API IDs for reference
4. **Foreign Keys**: Link launches to rockets and launchpads

---

## Inserting Data

### Basic INSERT

```python
cur.execute("""
    INSERT INTO users (name, email)
    VALUES ('John Doe', 'john@example.com')
""")
conn.commit()
```

### Parameterized Queries (CRITICAL for Security)

**❌ NEVER do this (SQL Injection vulnerability):**

```python
name = "John'; DROP TABLE users; --"
cur.execute(f"INSERT INTO users (name) VALUES ('{name}')")  # DANGEROUS!
```

**✅ ALWAYS use parameterized queries:**

```python
name = "John'; DROP TABLE users; --"
cur.execute("INSERT INTO users (name) VALUES (?)", (name,))  # Safe!
```

The `?` is a placeholder. SQLite automatically escapes the value.

### INSERT OR IGNORE

Prevents errors when inserting duplicate unique values:

```python
# If spacex_id already exists, silently skip
cur.execute("""
    INSERT OR IGNORE INTO rockets (spacex_id, name, type, active)
    VALUES (?, ?, ?, ?)
""", (rocket_id, name, rocket_type, 1))
```

### Our Project's Insert Pattern

```python
def load_dimension_tables(conn, rockets_map, launchpads_map):
    cur = conn.cursor()

    # Insert rockets
    for spacex_id, rocket_data in rockets_map.items():
        cur.execute("""
            INSERT OR IGNORE INTO rockets (spacex_id, name, type, active)
            VALUES (?, ?, ?, ?)
        """, (
            spacex_id,
            rocket_data.get("name"),
            rocket_data.get("type"),
            1 if rocket_data.get("active") else 0,
        ))

    conn.commit()  # Save all inserts at once
```

**Why `INSERT OR IGNORE`?**

- Allows re-running ETL without errors
- Idempotent operations (safe to run multiple times)
- Prevents duplicate entries

---

## Querying Data

### Basic SELECT

```python
cur.execute("SELECT * FROM users")
rows = cur.fetchall()  # Returns list of tuples
```

### SELECT with WHERE

```python
cur.execute("SELECT * FROM users WHERE name = ?", ("John",))
row = cur.fetchone()  # Returns single row or None
```

### SELECT Specific Columns

```python
cur.execute("SELECT name, email FROM users")
rows = cur.fetchall()
```

### Building a Dictionary from Results

```python
# Get id and spacex_id for mapping
cur.execute("SELECT id, spacex_id FROM rockets")
rocket_id_map = {row[1]: row[0] for row in cur.fetchall()}
# Result: {"5e9d0d95eda69955f709d1eb": 1, "5e9d0d95eda69973a809d1ec": 2, ...}
```

### Our Project's Query Pattern

```python
# After inserting rockets, build mapping from SpaceX ID to internal ID
cur.execute("SELECT id, spacex_id FROM rockets")
rocket_id_map = {row[1]: row[0] for row in cur.fetchall()}

# Now use this map when inserting launches
for launch in launches:
    rocket_spacex_id = launch.get("rocket")
    rocket_internal_id = rocket_id_map.get(rocket_spacex_id)  # Lookup!

    cur.execute("""
        INSERT INTO launches (rocket_id, ...)
        VALUES (?, ...)
    """, (rocket_internal_id, ...))
```

**Why this mapping?**

- API provides string IDs like `"5e9d0d95eda69955f709d1eb"`
- Database uses integer IDs like `1, 2, 3`
- Need to convert between them when inserting launches

---

## Updating and Deleting

### UPDATE

```python
cur.execute("""
    UPDATE users
    SET email = ?
    WHERE id = ?
""", ("newemail@example.com", 1))
conn.commit()
```

### DELETE

```python
cur.execute("DELETE FROM users WHERE id = ?", (1,))
conn.commit()
```

### DELETE with CASCADE

If foreign keys are set up with `ON DELETE CASCADE`, deleting a parent row automatically deletes child rows.

---

## Transactions and Commit

### What is a Transaction?

A transaction is a sequence of database operations that are treated as a single unit. Either all operations succeed, or all are rolled back.

### Commit

```python
conn = sqlite3.connect("example.db")
cur = conn.cursor()

cur.execute("INSERT INTO users (name) VALUES (?)", ("Alice",))
cur.execute("INSERT INTO users (name) VALUES (?)", ("Bob",))

conn.commit()  # Saves changes to disk
conn.close()
```

**Without `commit()`, changes are NOT saved!**

### Rollback

```python
try:
    cur.execute("INSERT INTO users (name) VALUES (?)", ("Alice",))
    cur.execute("INSERT INTO users (name) VALUES (?)", ("Bob",))
    conn.commit()
except Exception:
    conn.rollback()  # Undo all changes
    raise
```

### Our Project's Transaction Pattern

```python
def load_launches(conn, launches, rocket_id_map, launchpad_id_map):
    cur = conn.cursor()

    for launch in launches:
        # Multiple inserts in a loop
        cur.execute("INSERT INTO launches ...", (...))

    conn.commit()  # Single commit after all inserts (faster!)
```

**Why commit once at the end?**

- Much faster than committing after each insert
- All-or-nothing: if one fails, nothing is saved
- Better performance for bulk operations

---

## Foreign Keys and Relationships

### What are Foreign Keys?

Foreign keys create relationships between tables. They ensure referential integrity (you can't reference a non-existent row).

### Enabling Foreign Keys in SQLite

SQLite requires foreign keys to be explicitly enabled:

```python
conn = sqlite3.connect("example.db")
conn.execute("PRAGMA foreign_keys = ON")
```

### Our Project's Foreign Key Usage

```sql
CREATE TABLE launches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    rocket_id INTEGER,
    launchpad_id INTEGER,
    FOREIGN KEY (rocket_id) REFERENCES rockets(id),
    FOREIGN KEY (launchpad_id) REFERENCES launchpads(id)
)
```

**What this means:**

- `launches.rocket_id` must exist in `rockets.id`
- `launches.launchpad_id` must exist in `launchpads.id`
- Prevents orphaned records

### Joining Tables

```python
# In pandas (from our notebook):
df_launches.merge(
    df_rockets,
    left_on="rocket_id",
    right_on="id",
    how="left"
)
```

Or in pure SQL:

```sql
SELECT launches.*, rockets.name
FROM launches
LEFT JOIN rockets ON launches.rocket_id = rockets.id
```

---

## Best Practices

### 1. Always Use Parameterized Queries

```python
# ✅ Good
cur.execute("SELECT * FROM users WHERE id = ?", (user_id,))

# ❌ Bad
cur.execute(f"SELECT * FROM users WHERE id = {user_id}")
```

### 2. Use Context Managers

```python
# ✅ Good
with sqlite3.connect("db.db") as conn:
    cur = conn.cursor()
    # operations
# Auto-closed

# ❌ Bad
conn = sqlite3.connect("db.db")
# ... operations
conn.close()  # Might be skipped if error occurs
```

### 3. Commit Strategically

- Commit after logical groups of operations
- Don't commit after every single insert (slow)
- Don't forget to commit (data won't be saved)

### 4. Handle Errors

```python
try:
    cur.execute("INSERT INTO ...")
    conn.commit()
except sqlite3.IntegrityError:
    # Handle duplicate key, etc.
    conn.rollback()
except Exception as e:
    conn.rollback()
    raise
```

### 5. Use Transactions for Bulk Operations

```python
# ✅ Good: Single transaction
cur.executemany("INSERT INTO users VALUES (?, ?)", user_list)
conn.commit()

# ❌ Bad: Many transactions
for user in user_list:
    cur.execute("INSERT INTO users VALUES (?, ?)", user)
    conn.commit()  # Slow!
```

### 6. Index Frequently Queried Columns

```python
cur.execute("CREATE INDEX idx_rocket_id ON launches(rocket_id)")
```

---

## Real-World Examples from SpaceX Project

### Example 1: Creating the Schema

**File**: `Src/db.py` → `create_schema()`

```python
def create_schema(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()

    # Create rockets table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS rockets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            spacex_id TEXT UNIQUE,
            name TEXT,
            type TEXT,
            active INTEGER
        )
    """)

    conn.commit()
```

**Key Points:**

- `CREATE TABLE IF NOT EXISTS`: Safe to run multiple times
- `AUTOINCREMENT`: Auto-generates IDs (1, 2, 3, ...)
- `UNIQUE`: Prevents duplicate SpaceX IDs

### Example 2: Loading Dimension Tables

**File**: `Src/db.py` → `load_dimension_tables()`

```python
def load_dimension_tables(conn, rockets_map, launchpads_map):
    cur = conn.cursor()

    # Insert all rockets
    for spacex_id, rocket_data in rockets_map.items():
        cur.execute("""
            INSERT OR IGNORE INTO rockets (spacex_id, name, type, active)
            VALUES (?, ?, ?, ?)
        """, (
            spacex_id,
            rocket_data.get("name"),
            rocket_data.get("type"),
            1 if rocket_data.get("active") else 0,
        ))

    conn.commit()

    # Build mapping: SpaceX ID → Internal ID
    cur.execute("SELECT id, spacex_id FROM rockets")
    rocket_id_map = {row[1]: row[0] for row in cur.fetchall()}

    return rocket_id_map
```

**Why this pattern?**

1. Insert dimension data first (rockets, launchpads)
2. Build ID mapping dictionaries
3. Use mappings when inserting fact table (launches)

### Example 3: Loading Fact Table with Foreign Keys

**File**: `Src/db.py` → `load_launches()`

```python
def load_launches(conn, launches, rocket_id_map, launchpad_id_map):
    cur = conn.cursor()

    for launch in launches:
        # Get SpaceX IDs from API data
        rocket_spacex_id = launch.get("rocket")
        launchpad_spacex_id = launch.get("launchpad")

        # Convert to internal IDs using mapping
        rocket_id = rocket_id_map.get(rocket_spacex_id)
        launchpad_id = launchpad_id_map.get(launchpad_spacex_id)

        # Insert with internal IDs
        cur.execute("""
            INSERT OR IGNORE INTO launches
            (spacex_id, name, date_utc, success, rocket_id, launchpad_id, details)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            launch.get("id"),
            launch.get("name"),
            launch.get("date_utc"),
            1 if launch.get("success") else 0,
            rocket_id,      # Foreign key to rockets.id
            launchpad_id,   # Foreign key to launchpads.id
            launch.get("details"),
        ))

    conn.commit()
```

**Critical Concept:**

- API gives us: `"rocket": "5e9d0d95eda69955f709d1eb"`
- Database needs: `rocket_id: 1` (integer)
- We use the mapping dictionary to convert

### Example 4: Reading Data for Analysis

**File**: `notebook/spacex_analysis.ipynb`

```python
import sqlite3
import pandas as pd

conn = sqlite3.connect("../Data/DB/spacex.db")

# Load entire tables into pandas DataFrames
df_launches = pd.read_sql_query("SELECT * FROM launches", conn)
df_rockets = pd.read_sql_query("SELECT * FROM rockets", conn)
df_launchpads = pd.read_sql_query("SELECT * FROM launchpads", conn)

conn.close()
```

**Why pandas?**

- Easier data manipulation than raw SQL
- Built-in merge/join operations
- Statistical analysis tools

---

## Common SQLite3 Patterns Summary

| Operation        | Pattern                                                      | Example                                   |
| ---------------- | ------------------------------------------------------------ | ----------------------------------------- |
| **Connect**      | `sqlite3.connect(path)`                                      | `conn = sqlite3.connect("db.db")`         |
| **Create Table** | `cur.execute("CREATE TABLE ...")`                            | See schema examples above                 |
| **Insert**       | `cur.execute("INSERT INTO ... VALUES (?, ?)", (val1, val2))` | Parameterized queries                     |
| **Select**       | `cur.execute("SELECT ..."); rows = cur.fetchall()`           | `fetchall()`, `fetchone()`, `fetchmany()` |
| **Update**       | `cur.execute("UPDATE ... SET ... WHERE ...")`                | Always use WHERE clause                   |
| **Delete**       | `cur.execute("DELETE FROM ... WHERE ...")`                   | Always use WHERE clause                   |
| **Commit**       | `conn.commit()`                                              | Required to save changes                  |
| **Close**        | `conn.close()`                                               | Or use context manager                    |

---

## Troubleshooting

### Problem: "database is locked"

**Solution**: Make sure you're closing connections properly. Use context managers.

### Problem: Changes not saved

**Solution**: You forgot to call `conn.commit()`!

### Problem: Foreign key constraint failed

**Solution**: Make sure parent records exist before inserting child records.

### Problem: UNIQUE constraint failed

**Solution**: Use `INSERT OR IGNORE` or check for duplicates first.

---

## Next Steps

1. Practice creating your own tables
2. Experiment with JOINs
3. Learn about indexes for performance
4. Explore SQLite3's advanced features (views, triggers, etc.)

For more information, see the official SQLite documentation: https://www.sqlite.org/docs.html
