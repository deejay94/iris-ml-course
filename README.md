# Iris Flower Classification - Machine Learning Project

A comprehensive machine learning project that classifies iris flowers into three species (setosa, versicolor, virginica) using logistic regression. This project demonstrates the complete ML workflow from data exploration to model evaluation.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)
![pandas](https://img.shields.io/badge/pandas-1.3+-green.svg)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technologies](#technologies)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Results](#results)
- [Learning Resources](#learning-resources)
- [Workflow](#workflow)

## 🎯 Overview

This project implements a complete machine learning pipeline for classifying iris flowers based on their sepal and petal measurements. It covers:

- **Data Exploration**: Understanding the dataset through visualizations
- **Data Preparation**: Train/test splitting and feature engineering
- **Model Training**: Logistic regression with cross-validation
- **Model Evaluation**: Accuracy metrics and misclassification analysis
- **Visualization**: Comparing predictions vs actual labels

## ✨ Features

- 📊 **Exploratory Data Analysis (EDA)**
  - Statistical summaries
  - Feature distributions (histograms)
  - Feature relationships (scatter plots, pair plots)
  
- 🤖 **Machine Learning Models**
  - Simple rule-based baseline model
  - Logistic regression classifier
  - Cross-validation for robust evaluation
  
- 📈 **Model Evaluation**
  - 5-fold cross-validation
  - Training and test set accuracy
  - Misclassification analysis
  
- 📉 **Visualizations**
  - Predictions vs actual labels comparison
  - Correct vs incorrect predictions visualization
  - Feature relationship plots

## 🛠 Technologies

- **Python 3.8+**
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **scikit-learn** - Machine learning algorithms
- **matplotlib** - Plotting and visualization
- **seaborn** - Statistical data visualization

## 📦 Dataset

The project uses the famous [Iris dataset](https://archive.ics.uci.edu/ml/datasets/iris) from UCI Machine Learning Repository, which is built into scikit-learn.

**Features:**
- Sepal length (cm)
- Sepal width (cm)
- Petal length (cm)
- Petal width (cm)

**Target:**
- Species: setosa, versicolor, or virginica (3 classes)

**Dataset Size:** 150 samples (50 per species)

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- Conda (recommended) or pip

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/iris-ml-course.git
   cd iris-ml-course
   ```

2. **Create conda environment:**
   ```bash
   conda create -n iris-course pandas scikit-learn matplotlib seaborn numpy
   conda activate iris-course
   ```

   **Or using pip:**
   ```bash
   pip install pandas scikit-learn matplotlib seaborn numpy
   ```

3. **Verify installation:**
   ```bash
   python -c "import pandas, sklearn, matplotlib, seaborn, numpy; print('All packages installed!')"
   ```

## 💻 Usage

### Run the complete workflow:

```bash
# Activate environment
conda activate iris-course

# Run the script
python setup.py
```

### What the script does:

1. Loads the Iris dataset
2. Creates visualizations (EDA)
3. Splits data into train/test sets
4. Trains a logistic regression model
5. Evaluates using cross-validation
6. Makes predictions and visualizes results

### Expected Output:

- Console output with accuracy metrics
- Visualization plots (saved as PNG files)
- Prediction DataFrames (optional CSV export)

## 📁 Project Structure

```
iris-ml-course/
│
├── setup.py                          # Main script with complete ML workflow
├── README.md                         # This file
├── .gitignore                        # Git ignore rules
│
├── Documentation/                    # Learning resources
│   ├── EXPLANATION.md                # Complete workflow explanation
│   ├── STEP_BY_STEP_CHECKLIST.md     # Step-by-step guide
│   ├── DEPENDENCIES_EXPLAINED.md     # Package explanations
│   ├── NEXT_PROJECTS.md              # Similar projects to practice
│   │
│   └── Concept Guides/               # Detailed concept explanations
│       ├── what_is_logistic_regression.md
│       ├── train_test_split_vs_crossval.md
│       ├── cross_validation_explained.md
│       ├── boolean_indexing_explained.md
│       └── ... (more guides)
│
└── Outputs/                          # Generated files (optional)
    ├── scatterplot.png
    └── predictions.csv
```

## 📊 Results

### Model Performance

- **Cross-Validation Accuracy:** ~95-97% (5-fold CV)
- **Test Set Accuracy:** ~95-97%
- **Baseline Model Accuracy:** ~95% (simple rule-based)

### Key Insights

- Petal length and width are the most important features for classification
- Setosa is easily distinguishable from the other two species
- Versicolor and virginica have some overlap, causing occasional misclassifications

## 📚 Learning Resources

This project includes comprehensive documentation for learning:

- **[EXPLANATION.md](EXPLANATION.md)** - Complete breakdown of the ML workflow
- **[STEP_BY_STEP_CHECKLIST.md](STEP_BY_STEP_CHECKLIST.md)** - Detailed step-by-step guide
- **[DEPENDENCIES_EXPLAINED.md](DEPENDENCIES_EXPLAINED.md)** - All packages explained
- **[NEXT_PROJECTS.md](NEXT_PROJECTS.md)** - Similar projects to practice

### Concept Guides

- `what_is_logistic_regression.md` - Understanding logistic regression
- `train_test_split_vs_crossval.md` - Data splitting strategies
- `cross_validation_explained.md` - How cross-validation works
- `choosing_algorithms.md` - When to use which algorithm
- And more...

## 🔄 Workflow

The project follows a standard ML workflow:

```
1. Data Loading
   ↓
2. Exploratory Data Analysis (EDA)
   ↓
3. Data Preparation (Train/Test Split)
   ↓
4. Model Training
   ↓
5. Model Evaluation (Cross-Validation)
   ↓
6. Hyperparameter Tuning
   ↓
7. Final Evaluation (Test Set)
   ↓
8. Visualization & Analysis
```

## 🎓 Learning Objectives

By completing this project, you'll learn:

- ✅ Data exploration and visualization
- ✅ Train/test splitting
- ✅ Cross-validation for model evaluation
- ✅ Logistic regression for classification
- ✅ Hyperparameter tuning
- ✅ Model evaluation metrics
- ✅ Visualization of predictions
- ✅ Identifying misclassifications

## 🔧 Customization

### Try Different Models:

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Replace LogisticRegression with:
model = DecisionTreeClassifier()
# or
model = RandomForestClassifier()
```

### Adjust Hyperparameters:

```python
# In setup.py, modify:
model = LogisticRegression(max_iter=200, C=1.0)  # Try different C values
```

### Change Visualization:

Edit the plotting sections in `setup.py` to customize visualizations.

## 🤝 Contributing

This is a learning project, but suggestions and improvements are welcome!

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📝 License

This project is open source and available for educational purposes.

## 👤 Author

**Your Name**
- GitHub: [@YOUR_USERNAME](https://github.com/YOUR_USERNAME)

## 🙏 Acknowledgments

- UCI Machine Learning Repository for the Iris dataset
- scikit-learn team for excellent ML tools
- The open-source Python community

## 📖 References

- [Iris Dataset](https://archive.ics.uci.edu/ml/datasets/iris)
- [scikit-learn Documentation](https://scikit-learn.org/stable/)
- [pandas Documentation](https://pandas.pydata.org/)
- [seaborn Documentation](https://seaborn.pydata.org/)

---

**Happy Learning! 🚀**

If you find this project helpful, consider giving it a ⭐ on GitHub!

