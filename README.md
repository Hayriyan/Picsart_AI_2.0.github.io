# Picsart AI 2.0 - Educational Repository

A comprehensive learning resource for students covering **Machine Learning**, **Data Engineering**, **Data Analysis**, and **Python Programming**. This repository contains lecture notes, hands-on exercises, real-world projects, and evaluation tasks.

## 📚 Repository Purpose

This repository serves as an educational platform with:

- **Lecture Notes & Documentation** - Conceptual explanations of ML/DL topics
- **Practical Exercises** - Hands-on NumPy, data manipulation, and analysis
- **Real-World Projects** - SpaceX ETL pipeline demonstrating data engineering
- **Assessment Tasks** - Exam-style problems with datasets
- **Code Examples** - Working implementations of ML algorithms and techniques

## 🗂️ Project Structure

### 📁 [DB/](DB/) - Database & ETL

- **[SpaceX-ETL/](DB/SpaceX-ETL/)** - Complete data pipeline project
  - Extract SpaceX launch data from APIs
  - Transform and clean data using pandas
  - Load into SQLite database
  - Includes notebooks with analysis examples
  - Datasets: launches, launchpads, rockets
  - **Status**: Includes processed data and tutorials

### 📁 [EDA/](EDA/) - Exploratory Data Analysis

- **[Picsart/](EDA/Picsart/)** - ML/DL Program materials
  - 15-Month & 18-Month ML/DL Syllabus
  - Lab planning documents
  - Structured curriculum for deep learning studies

### 📁 [Models/](Models/) - Machine Learning Models

Conceptual documentation on key ML algorithms:

- Decision Trees
- K-Nearest Neighbors (KNN)
- Perceptron & Support Vector Machines (SVM)
- Polynomial & Ridge & Lasso Regression
- Optimization Algorithms
- ML Evaluation Metrics

**📚 Stanford CS229 Lecture Notes:**

- [GLM.pdf](Models/GLM.pdf) - Generalized Linear Models (from Stanford CS229)
- [Linear_Regression.pdf](Models/Linear_Regression.pdf) - Linear Regression (from Stanford CS229)

_These resources are adapted from Andrew Ng's Stanford CS229 course lectures._

### 📁 [NumPy/](NumPy/) - Python Numerical Computing

Interactive Jupyter Notebooks for NumPy mastery:

- NumPy fundamentals & lectures
- Numerical linear algebra
- 2 practice problem sets
- SciPy exercises

### 📁 [Exam/](Exam/) - Assessment Tasks

Real-world evaluation assignments:

- **Task 1** - Inventory & Product data cleaning
- **Task 2** - Additional assessment challenges
- **Task 3** - Employee payroll data processing
- Each task includes raw, noisy datasets for practice

### 📄 Documentation Files

- [TensorFlow.md](TensorFlow.md) - TensorFlow and deep learning notes
- [DB/SQLITE3_GUIDE.md](DB/SQLITE3_GUIDE.md) - SQLite database tutorial
- [DB/REQUESTS_API_GUIDE.md](DB/REQUESTS_API_GUIDE.md) - API interaction guide

## 🚀 Getting Started

### Prerequisites

- **Python 3.14.0** or higher
- Virtual environment (recommended)
- pip or conda package manager

### Installation

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd Picsart_AI_2.0
   ```

2. **Create a virtual environment**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On macOS/Linux
   # or
   .venv\Scripts\activate     # On Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r DB/SpaceX-ETL/requirments.txt
   ```

   Common packages you'll need:

   ```bash
   pip install pandas numpy matplotlib seaborn jupyter scikit-learn sqlite3 requests
   ```

### Quick Start by Topic

#### 🔢 Learning NumPy

1. Start with [NumPy/NumPy_Lecture.ipynb](NumPy/NumPy_Lecture.ipynb)
2. Work through practice sets in [NumPy/](NumPy/) folder
3. Complete SciPy exercises

#### 🔧 Data Engineering (SpaceX ETL)

1. Read [DB/SpaceX-ETL/README.md](DB/SpaceX-ETL/README.md)
2. Review the pipeline: [DB/SpaceX-ETL/etl_pipeline.py](DB/SpaceX-ETL/etl_pipeline.py)
3. Run the step-by-step tutorial: [DB/SpaceX-ETL/notebook/step_by_step_tutorial.ipynb](DB/SpaceX-ETL/notebook/step_by_step_tutorial.ipynb)
4. Explore analysis examples: [DB/SpaceX-ETL/notebook/spacex_analysis.ipynb](DB/SpaceX-ETL/notebook/spacex_analysis.ipynb)

#### 📊 Machine Learning Models

1. Review conceptual docs in [Models/](Models/) folder
2. Study the evaluation metrics guide: [Models/ML_Evaluation_Metrics.md](Models/ML_Evaluation_Metrics.md)
3. Progress from basic (Perceptron) → advanced (SVM, ensemble methods)

#### ✅ Practice Exams

1. Navigate to [Exam/](Exam/) folder
2. Read the task description (e.g., [Exam/Task_1/Task.md](Exam/Task_1/Task.md))
3. Download the dataset from the Data/ subfolder
4. Implement your solution
5. Verify against provided cleaned data

#### 💡 Additional Resources

- Database guide: [DB/SQLITE3_GUIDE.md](DB/SQLITE3_GUIDE.md)
- Working with APIs: [DB/REQUESTS_API_GUIDE.md](DB/REQUESTS_API_GUIDE.md)
- Deep learning: [TensorFlow.md](TensorFlow.md)

## 📖 Learning Path Recommendation

### For Beginners:

1. NumPy fundamentals (NumPy folder)
2. Data cleaning basics (Exam/Task_1)
3. SQL & databases (DB guides)
4. ML Models conceptual understanding (Models folder)

### For Intermediate Learners:

1. Complete SpaceX ETL pipeline
2. Advanced NumPy (linear algebra, scipy)
3. All exam tasks
4. ML model implementations

### For Advanced Learners:

1. Review all ML model documentation
2. Implement custom models from scratch
3. Deep learning with TensorFlow
4. Optimize existing pipelines

## 📝 Usage Examples

### Running a Jupyter Notebook

```bash
jupyter notebook NumPy/NumPy_Lecture.ipynb
```

### Running the SpaceX ETL Pipeline

```bash
cd DB/SpaceX-ETL
python etl_pipeline.py
```

### Loading a SQLite Database

```python
import sqlite3
conn = sqlite3.connect('DB/SpaceX-ETL/spacex.db')
cursor = conn.cursor()
```

## 🤝 Contributing

This is an educational repository. To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m 'Add educational content'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Open a pull request

## 📋 File Organization Tips

- **Practice files**: Save solutions in separate branches or folders
- **Experiment notebooks**: Use numbered versions (e.g., `analysis_v1.ipynb`, `analysis_v2.ipynb`)
- **Data processing**: Keep raw data in `Raw/` and processed in `Procesed/`

## 🆘 Troubleshooting

### Import Errors

```bash
# Verify all packages are installed
pip list
# Reinstall dependencies
pip install -r DB/SpaceX-ETL/requirments.txt --force-reinstall
```

### Jupyter Not Found

```bash
pip install jupyter
```

### Database Issues

Refer to [DB/SQLITE3_GUIDE.md](DB/SQLITE3_GUIDE.md) for detailed SQLite troubleshooting.

## 📚 Key Topics Covered

- Python fundamentals & NumPy
- Data structures & algorithms
- Databases (SQLite, SQL queries)
- ETL/Data pipelines
- Exploratory Data Analysis (EDA)
- Machine Learning algorithms
- ML evaluation & metrics
- Deep Learning with TensorFlow
- Data cleaning & preprocessing
- API integration

## 📄 License

This repository is for educational purposes. Please check individual files for specific license information.

---

## 📚 Acknowledgments & Attribution

This repository incorporates educational materials from various sources:

- **Stanford CS229 - Machine Learning Course** (Andrew Ng)
  - Generalized Linear Models (GLM.pdf) ✓
  - Linear Regression (Linear_Regression.pdf) ✓
  - [Course Website](https://cs229.stanford.edu/)

- **Original Content**: Custom lectures, tutorials, and exercises created for this program
- **Datasets**: SpaceX data sourced from public APIs

> **Attribution Note**: The GLM.pdf and Linear_Regression.pdf documents are educational materials derived from Stanford University's CS229 course. These are used here for learning purposes in accordance with their educational license.

---

**Last Updated**: April 2026  
**Version**: 1.0  
**Status**: Active - Regularly updated with new content
