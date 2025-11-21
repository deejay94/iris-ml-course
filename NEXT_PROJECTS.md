# Similar Projects to Practice Machine Learning

## 🎯 Projects Similar to Iris Classification

### 1. **Wine Quality Classification** ⭐ (Easiest - Start Here!)

**Dataset:** Wine Quality Dataset (from UCI or sklearn)
- **Similarity:** Multi-class classification (like Iris)
- **Features:** Chemical properties (alcohol, acidity, etc.)
- **Target:** Wine quality (3-9 scale) or wine type (red/white)

**What you'll practice:**
- Same workflow as Iris
- Classification with more features
- Handling different data types

**Code:**
```python
from sklearn.datasets import load_wine
data = load_wine()
# Or download from UCI ML Repository
```

**Skills:** Classification, EDA, model evaluation

---

### 2. **Breast Cancer Classification** ⭐⭐

**Dataset:** Breast Cancer Wisconsin Dataset (built into sklearn)
- **Similarity:** Binary classification (malignant/benign)
- **Features:** Cell measurements (30 features)
- **Target:** Malignant (1) or Benign (0)

**What you'll practice:**
- Binary classification (simpler than multi-class)
- More features (30 vs 4)
- Real-world medical application

**Code:**
```python
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
```

**Skills:** Binary classification, feature importance, medical ML

---

### 3. **Digits Classification** ⭐⭐

**Dataset:** Handwritten Digits (built into sklearn)
- **Similarity:** Multi-class classification (0-9 digits)
- **Features:** 8x8 pixel images (64 features)
- **Target:** Digit (0-9)

**What you'll practice:**
- Image classification basics
- High-dimensional data (64 features)
- Visualization of images

**Code:**
```python
from sklearn.datasets import load_digits
data = load_digits()
```

**Skills:** Image data, high dimensions, visualization

---

### 4. **Titanic Survival Prediction** ⭐⭐⭐ (Most Popular!)

**Dataset:** Titanic Dataset (from Kaggle)
- **Similarity:** Binary classification
- **Features:** Age, gender, class, fare, etc.
- **Target:** Survived (1) or Died (0)

**What you'll practice:**
- Handling missing values
- Categorical features (need encoding)
- Feature engineering
- Real-world messy data

**Get it:**
- Kaggle: https://www.kaggle.com/c/titanic
- Or: `pip install seaborn` then `sns.load_dataset('titanic')`

**Skills:** Data cleaning, feature engineering, categorical data

---

### 5. **House Price Prediction** ⭐⭐⭐ (Regression!)

**Dataset:** Boston Housing or California Housing
- **Similarity:** Same workflow, but REGRESSION (predicting numbers)
- **Features:** House size, location, age, etc.
- **Target:** House price (continuous number)

**What you'll practice:**
- Regression (not classification!)
- Different metrics (MSE, MAE, R²)
- Continuous predictions

**Code:**
```python
from sklearn.datasets import fetch_california_housing
data = fetch_california_housing()
```

**Skills:** Regression, continuous predictions, different metrics

---

### 6. **Customer Churn Prediction** ⭐⭐⭐

**Dataset:** Telco Customer Churn (from Kaggle)
- **Similarity:** Binary classification
- **Features:** Customer demographics, usage, contract info
- **Target:** Churned (1) or Not Churned (0)

**What you'll practice:**
- Business application
- Imbalanced classes (more not-churned than churned)
- Feature importance for business decisions

**Get it:** Kaggle or UCI ML Repository

**Skills:** Business ML, imbalanced data, feature importance

---

## 📊 Project Comparison

| Project | Difficulty | Type | Features | Dataset Source |
|---------|-----------|------|----------|----------------|
| **Wine Quality** | ⭐ | Classification | ~13 | sklearn/UCI |
| **Breast Cancer** | ⭐⭐ | Binary Classification | 30 | sklearn |
| **Digits** | ⭐⭐ | Multi-class | 64 | sklearn |
| **Titanic** | ⭐⭐⭐ | Binary Classification | ~10 | Kaggle |
| **House Prices** | ⭐⭐⭐ | Regression | ~8 | sklearn |
| **Customer Churn** | ⭐⭐⭐ | Binary Classification | ~20 | Kaggle |

---

## 🚀 Recommended Learning Path

### **Step 1: Wine Quality** (After Iris)
- Very similar to Iris
- Slightly more features
- Good confidence builder

### **Step 2: Breast Cancer** (Next)
- Binary classification (simpler)
- More features (30)
- Real-world application

### **Step 3: Digits** (Then)
- Image data
- High dimensions
- Fun visualization

### **Step 4: Titanic** (Challenge)
- Real-world messy data
- Missing values
- Feature engineering

### **Step 5: House Prices** (New Skill)
- Regression (different from classification)
- Continuous predictions
- Different metrics

---

## 💻 Quick Start Templates

### Template 1: Wine Quality
```python
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
data = load_wine()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
df['wine_type'] = df['target'].map({0: 'class_0', 1: 'class_1', 2: 'class_2'})

# Same workflow as Iris!
# ... (follow your Iris code structure)
```

### Template 2: Breast Cancer
```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
import pandas as pd

# Load data
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
df['diagnosis'] = df['target'].map({0: 'benign', 1: 'malignant'})

# Binary classification - simpler!
# ... (same workflow)
```

### Template 3: Digits
```python
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

# Load data
data = load_digits()
X, y = data.data, data.target

# Visualize images
fig, axes = plt.subplots(2, 5, figsize=(10, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(data.images[i], cmap='gray')
    ax.set_title(f'Digit: {data.target[i]}')
plt.show()

# Then same ML workflow!
```

---

## 🎓 What Each Project Teaches

### **Wine Quality:**
- ✅ Multi-class classification
- ✅ More features
- ✅ Feature importance

### **Breast Cancer:**
- ✅ Binary classification
- ✅ Many features (30)
- ✅ Medical application
- ✅ Feature selection

### **Digits:**
- ✅ Image data
- ✅ High dimensions
- ✅ Visualization of images
- ✅ Dimensionality reduction (optional)

### **Titanic:**
- ✅ Data cleaning
- ✅ Missing values
- ✅ Categorical encoding
- ✅ Feature engineering
- ✅ Real-world messy data

### **House Prices:**
- ✅ Regression (new!)
- ✅ Continuous predictions
- ✅ Different metrics (MSE, MAE)
- ✅ Feature scaling importance

### **Customer Churn:**
- ✅ Business application
- ✅ Imbalanced classes
- ✅ Feature importance for decisions
- ✅ Cost-sensitive learning

---

## 📚 Where to Get Datasets

### **Built into sklearn:**
```python
from sklearn.datasets import load_wine, load_breast_cancer, load_digits
from sklearn.datasets import fetch_california_housing
```

### **Kaggle:**
- https://www.kaggle.com/datasets
- Popular: Titanic, Customer Churn, House Prices
- Free account needed

### **UCI ML Repository:**
- https://archive.ics.uci.edu/
- Wine Quality, many others
- Free, no account needed

### **Seaborn built-in:**
```python
import seaborn as sns
df = sns.load_dataset('titanic')  # Some datasets available
```

---

## 🎯 My Recommendation

**Start with Wine Quality!**

**Why:**
1. Very similar to Iris (you'll feel confident)
2. Built into sklearn (easy to load)
3. Slightly more complex (13 features vs 4)
4. Good practice before harder projects

**Then try:**
1. Breast Cancer (binary classification)
2. Digits (image data)
3. Titanic (real-world challenge)

**Finally:**
- House Prices (learn regression)
- Customer Churn (business application)

---

## 💡 Pro Tips

1. **Reuse your Iris code structure** - Same workflow!
2. **Start simple** - Wine Quality is perfect next step
3. **Compare results** - See which algorithms work best
4. **Visualize everything** - Helps understand the data
5. **Document your process** - Like you did with Iris

---

## 🚀 Quick Start: Wine Quality Project

```python
# Complete template - just run it!
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme()

# Load data
data = load_wine()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
df['wine_type'] = df['target'].map({i: f'class_{i}' for i in range(3)})

# Explore
print(df.head())
print(df.describe())

# Visualize
sns.pairplot(df, hue='wine_type', vars=df.columns[:4])
plt.show()

# Split
df_train, df_test = train_test_split(df, test_size=0.25, random_state=42)
X_train = df_train.drop(columns=['target', 'wine_type']).values
y_train = df_train['target'].values
X_test = df_test.drop(columns=['target', 'wine_type']).values
y_test = df_test['target'].values

# Model
model = LogisticRegression(max_iter=200)
scores = cross_val_score(model, X_train, y_train, cv=5)
print(f"CV Accuracy: {np.mean(scores):.3f}")

# Final evaluation
model.fit(X_train, y_train)
test_accuracy = model.score(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.3f}")
```

**You're ready to practice! Start with Wine Quality and work your way up!** 🍷📊

