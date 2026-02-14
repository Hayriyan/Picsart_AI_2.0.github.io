# Requests and API Guide

## Table of Contents
1. [Introduction to HTTP and APIs](#introduction-to-http-and-apis)
2. [What is the `requests` Library?](#what-is-the-requests-library)
3. [Making Your First API Call](#making-your-first-api-call)
4. [HTTP Methods Explained](#http-methods-explained)
5. [Understanding API Responses](#understanding-api-responses)
6. [Error Handling](#error-handling)
7. [Working with JSON Data](#working-with-json-data)
8. [API Best Practices](#api-best-practices)
9. [Real-World Examples from SpaceX Project](#real-world-examples-from-spacex-project)
10. [Advanced Topics](#advanced-topics)

---

## Introduction to HTTP and APIs

### What is HTTP?

**HTTP (HyperText Transfer Protocol)** is the foundation of data communication on the web. It's how your browser talks to websites, and how applications communicate with APIs.

### HTTP Request Structure

Every HTTP request has:
- **Method**: What action to perform (GET, POST, PUT, DELETE)
- **URL**: Where to send the request
- **Headers**: Metadata about the request
- **Body**: Data to send (optional, for POST/PUT)

### What is an API?

**API (Application Programming Interface)** is a set of rules that allows different software applications to communicate with each other.

**REST API**: A type of API that uses HTTP methods to perform operations on resources.

### Common API Patterns

```
GET    /users          → Get all users
GET    /users/123      → Get user with ID 123
POST   /users          → Create a new user
PUT    /users/123      → Update user 123
DELETE /users/123      → Delete user 123
```

---

## What is the `requests` Library?

The `requests` library is the most popular Python library for making HTTP requests. It's much simpler than Python's built-in `urllib`.

### Installation

```bash
pip install requests
```

### Why Use `requests`?

- ✅ Simple and intuitive API
- ✅ Automatic JSON parsing
- ✅ Built-in error handling
- ✅ Session management
- ✅ Authentication support

---

## Making Your First API Call

### Basic GET Request

```python
import requests

# Simple GET request
response = requests.get("https://api.github.com/users/octocat")
print(response.status_code)  # 200 means success
print(response.json())       # Parse JSON response
```

### Our Project's Base Function

From `Src/api.py`:

```python
import requests

BASE_URL = "https://api.spacexdata.com/v4"

def get_json(endpoint: str) -> Any:
    """GET helper with basic error handling."""
    url = f"{BASE_URL}/{endpoint.lstrip('/')}"
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()  # Raises exception for bad status codes
        return resp.json()
    except requests.RequestException as exc:
        raise SystemExit(f"Error fetching {url}: {exc}") from exc
```

**Key Components:**
- **Base URL**: `https://api.spacexdata.com/v4`
- **Endpoint**: Path like `/launches` or `/rockets/5e9d0d95eda69955f709d1eb`
- **Timeout**: Prevents hanging forever (20 seconds)
- **Error Handling**: Catches network errors and HTTP errors

### Using the Function

```python
# Get all launches
launches = get_json("launches")

# Get specific rocket
rocket = get_json("rockets/5e9d0d95eda69955f709d1eb")
```

---

## HTTP Methods Explained

### GET - Retrieve Data

**Purpose**: Fetch data from the server (read-only)

```python
response = requests.get("https://api.example.com/users")
data = response.json()
```

**Characteristics:**
- No request body
- Idempotent (safe to call multiple times)
- Can be cached

### POST - Create Data

**Purpose**: Send data to create a new resource

```python
new_user = {"name": "John", "email": "john@example.com"}
response = requests.post(
    "https://api.example.com/users",
    json=new_user  # Automatically converts to JSON
)
```

### PUT - Update Data

**Purpose**: Update an existing resource (full replacement)

```python
updated_user = {"name": "John Doe", "email": "john@example.com"}
response = requests.put(
    "https://api.example.com/users/123",
    json=updated_user
)
```

### DELETE - Remove Data

**Purpose**: Delete a resource

```python
response = requests.delete("https://api.example.com/users/123")
```

**Note**: Our SpaceX project only uses GET (read-only API)

---

## Understanding API Responses

### Response Object Properties

```python
response = requests.get("https://api.example.com/data")

# Status Code
print(response.status_code)  # 200, 404, 500, etc.

# Response Headers
print(response.headers)      # Metadata about response

# Response Body (as text)
print(response.text)         # Raw text

# Response Body (as JSON)
print(response.json())       # Parsed JSON (dict/list)

# Response Body (as bytes)
print(response.content)      # Raw bytes
```

### HTTP Status Codes

| Code | Meaning | Description |
|------|---------|-------------|
| **200** | OK | Request successful |
| **201** | Created | Resource created successfully |
| **400** | Bad Request | Invalid request format |
| **401** | Unauthorized | Authentication required |
| **403** | Forbidden | Not allowed to access |
| **404** | Not Found | Resource doesn't exist |
| **500** | Internal Server Error | Server error |
| **503** | Service Unavailable | Server temporarily down |

### Checking Status Codes

```python
response = requests.get("https://api.example.com/data")

# Method 1: Manual check
if response.status_code == 200:
    data = response.json()
else:
    print(f"Error: {response.status_code}")

# Method 2: raise_for_status() (Recommended)
response.raise_for_status()  # Raises exception if status >= 400
data = response.json()
```

---

## Error Handling

### Basic Error Handling

```python
import requests

try:
    response = requests.get("https://api.example.com/data", timeout=10)
    response.raise_for_status()  # Raises HTTPError for bad status codes
    data = response.json()
except requests.exceptions.Timeout:
    print("Request timed out")
except requests.exceptions.ConnectionError:
    print("Could not connect to server")
except requests.exceptions.HTTPError as e:
    print(f"HTTP error: {e}")
except requests.exceptions.RequestException as e:
    print(f"Request failed: {e}")
```

### Our Project's Error Handling

From `Src/api.py`:

```python
def get_json(endpoint: str) -> Any:
    url = f"{BASE_URL}/{endpoint.lstrip('/')}"
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()  # Automatically raises for 4xx/5xx
        return resp.json()
    except requests.RequestException as exc:
        # Catches all request-related errors
        raise SystemExit(f"Error fetching {url}: {exc}") from exc
```

**Why `raise SystemExit`?**
- Stops the entire program if API fails
- Prevents partial data corruption
- Clear error message for debugging

### Common Exception Types

| Exception | When It Occurs |
|-----------|----------------|
| `requests.exceptions.Timeout` | Request took too long |
| `requests.exceptions.ConnectionError` | Network problem (no internet, DNS failure) |
| `requests.exceptions.HTTPError` | HTTP error status (4xx, 5xx) |
| `requests.exceptions.RequestException` | Base class for all request errors |

---

## Working with JSON Data

### What is JSON?

**JSON (JavaScript Object Notation)** is a lightweight data format. It's the most common format for API responses.

### JSON Structure

```json
{
  "name": "Falcon 9",
  "type": "rocket",
  "active": true,
  "stages": 2,
  "boosters": 0
}
```

### Python JSON Handling

```python
import requests
import json

# requests automatically parses JSON
response = requests.get("https://api.example.com/data")
data = response.json()  # Returns Python dict/list

# Access nested data
rocket_name = data["name"]
is_active = data["active"]

# Handle missing keys safely
rocket_name = data.get("name", "Unknown")
```

### Our Project's JSON Usage

```python
# Get launches (returns list of dicts)
launches = get_json("launches")
# Result: [{"id": "...", "name": "...", "rocket": "...", ...}, ...]

# Get specific rocket (returns dict)
rocket = get_json(f"rockets/{rocket_id}")
# Result: {"id": "...", "name": "Falcon 9", "type": "rocket", ...}
```

### Saving JSON to File

```python
import json

data = {"name": "Falcon 9", "active": True}

# Write to file
with open("data.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)

# Read from file
with open("data.json", "r", encoding="utf-8") as f:
    data = json.load(f)
```

**Our Project**: We save raw JSON in `Src/etl.py`:

```python
def save_raw_json(name: str, data: Any) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"{name}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f)
```

---

## API Best Practices

### 1. Always Use Timeouts

```python
# ✅ Good
response = requests.get(url, timeout=20)

# ❌ Bad (can hang forever)
response = requests.get(url)
```

### 2. Handle Errors Gracefully

```python
# ✅ Good
try:
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return response.json()
except requests.RequestException as e:
    logger.error(f"API call failed: {e}")
    return None

# ❌ Bad
response = requests.get(url)
data = response.json()  # Crashes if request fails
```

### 3. Use Session for Multiple Requests

```python
# ✅ Good (reuses connection)
session = requests.Session()
for url in urls:
    response = session.get(url)

# ❌ Bad (creates new connection each time)
for url in urls:
    response = requests.get(url)
```

### 4. Respect Rate Limits

Many APIs limit how many requests you can make per second/minute.

```python
import time

for item in items:
    response = requests.get(f"https://api.example.com/{item}")
    time.sleep(1)  # Wait 1 second between requests
```

### 5. Cache Responses When Possible

```python
# Don't fetch the same data multiple times
if rocket_id not in cache:
    cache[rocket_id] = get_json(f"rockets/{rocket_id}")
return cache[rocket_id]
```

### 6. Use Headers for Authentication

```python
headers = {
    "Authorization": "Bearer YOUR_TOKEN",
    "Content-Type": "application/json"
}
response = requests.get(url, headers=headers)
```

---

## Real-World Examples from SpaceX Project

### Example 1: Fetching All Launches

**File**: `Src/api.py` → `extract_all_launches()`

```python
def extract_all_launches() -> List[Dict[str, Any]]:
    launches = get_json("launches")
    if not isinstance(launches, list):
        raise ValueError("Unexpected launches payload")
    return launches
```

**What happens:**
1. Calls `get_json("launches")` which constructs URL: `https://api.spacexdata.com/v4/launches`
2. Makes GET request
3. Parses JSON response (list of launch dictionaries)
4. Validates it's actually a list
5. Returns the list

**Response Structure:**
```json
[
  {
    "id": "5eb87cd9ffd86e000604b32a",
    "name": "FalconSat",
    "rocket": "5e9d0d95eda69955f709d1eb",
    "launchpad": "5e9e4501f5090910d4566f83",
    "date_utc": "2006-03-24T22:30:00.000Z",
    "success": false,
    "details": "Engine failure..."
  },
  ...
]
```

### Example 2: Extracting Unique IDs

**File**: `Src/api.py` → `extract_unique_ids()`

```python
def extract_unique_ids(launches: List[Dict[str, Any]]) -> tuple[Set[str], Set[str]]:
    rocket_ids: Set[str] = set()
    launchpad_ids: Set[str] = set()
    
    for launch in launches:
        rid = launch.get("rocket")      # Get rocket ID from launch
        pid = launch.get("launchpad")   # Get launchpad ID from launch
        
        if rid:
            rocket_ids.add(rid)          # Add to set (automatically deduplicates)
        if pid:
            launchpad_ids.add(pid)
    
    return rocket_ids, launchpad_ids
```

**Why use a Set?**
- Automatically removes duplicates
- Fast membership testing
- Example: If 200 launches use the same 3 rockets, set will only contain 3 IDs

**Example Flow:**
```python
launches = [
    {"rocket": "rocket_1", "launchpad": "pad_1"},
    {"rocket": "rocket_1", "launchpad": "pad_2"},  # Same rocket, different pad
    {"rocket": "rocket_2", "launchpad": "pad_1"},
]

rocket_ids, launchpad_ids = extract_unique_ids(launches)
# rocket_ids = {"rocket_1", "rocket_2"}  (only 2 unique rockets)
# launchpad_ids = {"pad_1", "pad_2"}    (only 2 unique pads)
```

### Example 3: Fetching Rocket Details

**File**: `Src/api.py` → `fetch_rockets()`

```python
def fetch_rockets(unique_rocket_ids: Set[str]) -> Dict[str, Dict[str, Any]]:
    rockets: Dict[str, Dict[str, Any]] = {}
    
    for rid in unique_rocket_ids:
        rockets[rid] = get_json(f"rockets/{rid}")
    
    return rockets
```

**What happens:**
1. Takes set of unique rocket IDs: `{"5e9d0d95eda69955f709d1eb", "5e9d0d95eda69973a809d1ec"}`
2. For each ID, makes API call: `GET /rockets/5e9d0d95eda69955f709d1eb`
3. Stores result in dictionary: `{rocket_id: rocket_data}`
4. Returns dictionary for easy lookup

**Result Structure:**
```python
{
    "5e9d0d95eda69955f709d1eb": {
        "id": "5e9d0d95eda69955f709d1eb",
        "name": "Falcon 9",
        "type": "rocket",
        "active": True,
        "stages": 2,
        ...
    },
    "5e9d0d95eda69973a809d1ec": {
        "id": "5e9d0d95eda69973a809d1ec",
        "name": "Falcon Heavy",
        ...
    }
}
```

**Why this pattern?**
- Only fetches each rocket once (efficient)
- Dictionary allows fast lookup: `rockets[rocket_id]`
- Avoids redundant API calls

### Example 4: Complete ETL Flow

**File**: `Src/etl.py` → `run_etl()`

```python
def run_etl() -> None:
    # Step 1: Extract all launches
    launches = api.extract_all_launches()
    
    # Step 2: Find unique rocket and launchpad IDs
    rocket_ids, launchpad_ids = api.extract_unique_ids(launches)
    
    # Step 3: Fetch details for each unique rocket (only once per rocket!)
    rockets_map = api.fetch_rockets(rocket_ids)
    
    # Step 4: Fetch details for each unique launchpad (only once per pad!)
    launchpads_map = api.fetch_launchpads(launchpad_ids)
    
    # Step 5: Save raw JSON snapshots
    save_raw_json("launches", launches)
    save_raw_json("rockets", rockets_map)
    save_raw_json("launchpads", launchpads_map)
    
    # Step 6: Load into database (see db.py)
    ...
```

**Efficiency Analysis:**
- If we have 200 launches using 3 rockets and 5 launchpads:
  - **Naive approach**: 200 + 200 + 200 = 600 API calls
  - **Our approach**: 200 + 3 + 5 = 208 API calls
  - **Savings**: 392 fewer API calls!

---

## Advanced Topics

### Query Parameters

Many APIs support filtering via query parameters:

```python
# SpaceX API example (if supported)
response = requests.get(
    "https://api.spacexdata.com/v4/launches",
    params={"limit": 10, "offset": 0}
)
# Results in: https://api.spacexdata.com/v4/launches?limit=10&offset=0
```

### Pagination

Some APIs return data in pages:

```python
page = 1
all_results = []

while True:
    response = requests.get(
        "https://api.example.com/data",
        params={"page": page}
    )
    data = response.json()
    
    if not data:  # No more results
        break
    
    all_results.extend(data)
    page += 1
```

### Authentication

Many APIs require authentication:

```python
# API Key in header
headers = {"X-API-Key": "your-api-key"}
response = requests.get(url, headers=headers)

# Bearer token
headers = {"Authorization": "Bearer your-token"}
response = requests.get(url, headers=headers)

# Basic auth
response = requests.get(url, auth=("username", "password"))
```

### Retry Logic

For unreliable networks:

```python
import time
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

session = requests.Session()
retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504]
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("http://", adapter)
session.mount("https://", adapter)

response = session.get(url)
```

### Streaming Large Responses

For very large files:

```python
response = requests.get(url, stream=True)

with open("large_file.json", "wb") as f:
    for chunk in response.iter_content(chunk_size=8192):
        f.write(chunk)
```

---

## Common Patterns Summary

| Task | Pattern | Example |
|------|---------|---------|
| **Basic GET** | `requests.get(url)` | `requests.get("https://api.example.com/data")` |
| **With Timeout** | `requests.get(url, timeout=10)` | Prevents hanging |
| **Parse JSON** | `response.json()` | Returns Python dict/list |
| **Check Status** | `response.raise_for_status()` | Raises exception if error |
| **Error Handling** | `try/except requests.RequestException` | Catches all request errors |
| **Query Params** | `params={"key": "value"}` | Adds `?key=value` to URL |
| **Headers** | `headers={"Header": "Value"}` | Custom headers |
| **POST with JSON** | `requests.post(url, json=data)` | Auto-converts to JSON |

---

## Troubleshooting

### Problem: "Connection timeout"
**Solution**: Increase timeout or check internet connection
```python
response = requests.get(url, timeout=60)  # Increase timeout
```

### Problem: "404 Not Found"
**Solution**: Check the URL/endpoint is correct
```python
print(response.url)  # See what URL was actually called
```

### Problem: "429 Too Many Requests"
**Solution**: You're hitting rate limits - add delays
```python
import time
time.sleep(1)  # Wait 1 second between requests
```

### Problem: "JSON decode error"
**Solution**: Response might not be JSON - check content type
```python
print(response.headers.get("Content-Type"))
print(response.text[:100])  # See first 100 chars
```

---

## Next Steps

1. Practice with different APIs (GitHub API, OpenWeatherMap, etc.)
2. Learn about API authentication (OAuth, API keys)
3. Explore API documentation tools (Swagger, Postman)
4. Learn about REST API design principles

For more information:
- Requests documentation: https://docs.python-requests.org/
- HTTP status codes: https://httpstatuses.com/
- JSON specification: https://www.json.org/

