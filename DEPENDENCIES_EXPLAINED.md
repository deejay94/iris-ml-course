# All Dependencies Explained

## Your Project Dependencies

Based on your `setup.py` and conda environment, here are all the packages you're using:

---

## 📦 Core Dependencies

### 1. **pandas** (`pd`)
```python
import pandas as pd
```

**What it is:** Data manipulation and analysis library

**What it does:**
- Creates DataFrames (like Excel spreadsheets in Python)
- Handles structured data (tables)
- Data cleaning, filtering, grouping
- Reading/writing CSV, Excel files

**Why you need it:**
- `pd.DataFrame()` - Create data tables
- `df.head()`, `df.describe()` - Explore data
- `df.drop()`, `df.copy()` - Manipulate data
- Essential for working with data in ML

**Example in your code:**
```python
df = pd.DataFrame(data['data'], columns=data['feature_names'])
df_train, df_test = train_test_split(df, test_size=0.25)
```

---

### 2. **numpy** (`np`)
```python
import numpy as np
```

**What it is:** Numerical computing library

**What it does:**
- Fast array operations
- Mathematical functions
- Linear algebra
- Foundation for other ML libraries

**Why you need it:**
- `np.array()` - Create arrays
- `np.mean()` - Calculate averages
- `np.array([...])` - Convert lists to arrays
- Fast numerical operations

**Example in your code:**
```python
manual_y_predictions = np.array([...])
np.mean(manual_y_predictions == y_train)
```

---

### 3. **matplotlib** (`plt`)
```python
import matplotlib.pyplot as plt
```

**What it is:** Plotting and visualization library

**What it does:**
- Creates graphs, charts, plots
- Line plots, scatter plots, histograms
- Customizable visualizations
- Save plots as images

**Why you need it:**
- `plt.subplots()` - Create multiple plots
- `plt.show()` - Display plots
- `plt.savefig()` - Save plots
- `plt.tight_layout()` - Adjust spacing

**Example in your code:**
```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
plt.tight_layout()
plt.show()
```

---

### 4. **seaborn** (`sns`)
```python
import seaborn as sns
```

**What it is:** Statistical data visualization library (built on matplotlib)

**What it does:**
- Beautiful, statistical plots
- Easy-to-use plotting functions
- Better default styles
- Statistical visualizations

**Why you need it:**
- `sns.set_theme()` - Set modern style
- `sns.scatterplot()` - Scatter plots
- `sns.relplot()` - Relationship plots
- `sns.pairplot()` - All features vs all features
- Makes matplotlib plots look better

**Example in your code:**
```python
sns.set_theme()
sns.scatterplot(x="petal length (cm)", y="petal width (cm)", hue="pred_label", data=df_predictions, ax=ax1)
sns.pairplot(df, hue="target_name")
```

---

### 5. **scikit-learn** (`sklearn`)
```python
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.linear_model import LogisticRegression
```

**What it is:** Machine learning library

**What it does:**
- ML algorithms (classification, regression)
- Data preprocessing
- Model evaluation
- Built-in datasets

**Why you need it:**
- `datasets.load_iris()` - Load datasets
- `train_test_split()` - Split data
- `LogisticRegression()` - ML model
- `cross_val_score()` - Evaluate model
- `cross_val_predict()` - Get predictions

**Example in your code:**
```python
data = datasets.load_iris()
df_train, df_test = train_test_split(df, test_size=0.25)
model = LogisticRegression(max_iter=200)
accuracies = cross_val_score(model, X_train, y_train, cv=5)
```

---

## 🔗 Dependency Relationships

```
┌─────────────────────────────────────────┐
│         Your Python Code                │
└─────────────────────────────────────────┘
                    ↓
        ┌───────────┴───────────┐
        │                       │
        ↓                       ↓
┌───────────────┐      ┌──────────────────┐
│   pandas      │      │   scikit-learn   │
│   (DataFrames)│      │   (ML Models)    │
└───────────────┘      └──────────────────┘
        │                       │
        └───────────┬───────────┘
                    ↓
            ┌───────────────┐
            │    numpy      │
            │  (Arrays)     │
            └───────────────┘
                    ↓
        ┌───────────┴───────────┐
        │                       │
        ↓                       ↓
┌───────────────┐      ┌───────────────┐
│  matplotlib   │      │   seaborn     │
│  (Plotting)   │      │  (Better plots)│
└───────────────┘      └───────────────┘
```

**Note:** seaborn is built on matplotlib, and both use numpy arrays

---

## 📋 Complete Dependency List

### **Direct Dependencies** (You installed these):
1. **pandas** - Data manipulation
2. **scikit-learn** - Machine learning
3. **matplotlib** - Plotting
4. **seaborn** - Statistical visualization
5. **numpy** - Numerical computing

### **Indirect Dependencies** (Installed automatically):
- **scipy** - Scientific computing (used by scikit-learn)
- **joblib** - Parallel processing (used by scikit-learn)
- **pillow** - Image processing (used by matplotlib)
- **pytz** - Timezone handling (used by pandas)
- **dateutil** - Date parsing (used by pandas)

---

## 🎯 What Each Package Does in Your Project

### **Data Loading & Manipulation:**
- **pandas**: Create DataFrames, manipulate data
- **numpy**: Convert to arrays, numerical operations
- **sklearn.datasets**: Load built-in datasets

### **Machine Learning:**
- **sklearn.linear_model**: Logistic Regression model
- **sklearn.model_selection**: Train/test split, cross-validation

### **Visualization:**
- **matplotlib**: Create plots, subplots, save images
- **seaborn**: Beautiful statistical plots, modern styling

---

## 📦 Installation Commands

### **What you ran:**
```bash
conda create -n iris-course pandas scikit-learn matplotlib seaborn numpy
```

### **What each package provides:**

**pandas:**
- `pd.DataFrame()` - Data tables
- `df.head()`, `df.describe()` - Data exploration
- `df.drop()`, `df.copy()` - Data manipulation

**scikit-learn:**
- `datasets.load_iris()` - Datasets
- `train_test_split()` - Data splitting
- `LogisticRegression()` - ML models
- `cross_val_score()` - Model evaluation

**matplotlib:**
- `plt.subplots()` - Create plots
- `plt.show()` - Display plots
- `plt.savefig()` - Save plots

**seaborn:**
- `sns.set_theme()` - Modern styling
- `sns.scatterplot()` - Scatter plots
- `sns.pairplot()` - Feature relationships

**numpy:**
- `np.array()` - Arrays
- `np.mean()` - Statistics
- Fast numerical operations

---

## 🔍 Package Versions

To check what versions you have:
```bash
conda activate iris-course
conda list
```

Or in Python:
```python
import pandas as pd
import numpy as np
import matplotlib
import seaborn as sns
import sklearn

print(f"pandas: {pd.__version__}")
print(f"numpy: {np.__version__}")
print(f"matplotlib: {matplotlib.__version__}")
print(f"seaborn: {sns.__version__}")
print(f"scikit-learn: {sklearn.__version__}")
```

---

## 💡 Why These Specific Packages?

### **pandas:**
- Industry standard for data manipulation
- Easy to use, powerful
- Works great with ML libraries

### **numpy:**
- Foundation for all numerical Python
- Fast array operations
- Required by pandas, sklearn, matplotlib

### **scikit-learn:**
- Most popular ML library in Python
- Easy to use, well-documented
- Great for beginners

### **matplotlib:**
- Most flexible plotting library
- Can create any type of plot
- Foundation for seaborn

### **seaborn:**
- Makes matplotlib plots look better
- Statistical visualizations
- Easier to use for common plots

---

## 🎓 Learning Order

### **Start with:**
1. **pandas** - Data manipulation (most used)
2. **numpy** - Arrays and math (foundation)
3. **scikit-learn** - ML models (core of ML)

### **Then learn:**
4. **matplotlib** - Basic plotting
5. **seaborn** - Better plots (easier than matplotlib)

---

## 📚 What You Could Add Later

### **For more advanced projects:**
- **jupyter** or **jupyterlab** - Interactive notebooks
- **scipy** - Advanced scientific computing
- **statsmodels** - Statistical modeling
- **plotly** - Interactive visualizations
- **xgboost** - Advanced ML models

### **For specific tasks:**
- **requests** - Download data from web
- **openpyxl** - Read Excel files
- **pillow** - Image processing

---

## 🔧 Troubleshooting

### **If imports fail:**
```bash
# Check if package is installed
conda list | grep pandas

# Reinstall if needed
conda install pandas scikit-learn matplotlib seaborn numpy
```

### **If version conflicts:**
```bash
# Update all packages
conda update --all
```

---

## 📝 Summary

**Your 5 main dependencies:**

1. **pandas** → Data manipulation (DataFrames)
2. **numpy** → Numerical computing (arrays)
3. **scikit-learn** → Machine learning (models, evaluation)
4. **matplotlib** → Plotting (graphs, charts)
5. **seaborn** → Better plots (statistical visualizations)

**All work together:**
- pandas uses numpy arrays
- scikit-learn uses numpy arrays
- matplotlib uses numpy arrays
- seaborn uses matplotlib
- Everything works together seamlessly!

---

## 🎯 Quick Reference

| Package | Purpose | Key Functions |
|---------|---------|---------------|
| **pandas** | Data manipulation | `DataFrame()`, `head()`, `describe()` |
| **numpy** | Numerical computing | `array()`, `mean()`, mathematical ops |
| **scikit-learn** | Machine learning | `LogisticRegression()`, `train_test_split()` |
| **matplotlib** | Plotting | `subplots()`, `show()`, `savefig()` |
| **seaborn** | Better plots | `scatterplot()`, `pairplot()`, `set_theme()` |

**These 5 packages are all you need for your ML project!** 🚀

