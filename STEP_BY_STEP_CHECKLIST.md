# Complete Machine Learning Workflow Checklist

## ✅ Phase 1: Environment Setup

- [ ] **Install conda** (if not already installed)
  ```bash
  conda --version  # Check if installed
  ```

- [ ] **Create conda environment**
  ```bash
  conda create -n iris-course pandas scikit-learn matplotlib seaborn numpy
  ```

- [ ] **Activate environment**
  ```bash
  conda activate iris-course
  ```

- [ ] **Verify packages installed**
  ```bash
  conda list
  ```

- [ ] **Configure VS Code/Cursor** (if needed)
  - Python interpreter should point to conda environment
  - Check `.vscode/settings.json` is set correctly

---

## ✅ Phase 2: Data Loading & Exploration

- [ ] **Import libraries**
  ```python
  import pandas as pd
  import numpy as np
  import matplotlib.pyplot as plt
  import seaborn as sns
  from sklearn import datasets
  ```

- [ ] **Set seaborn theme**
  ```python
  sns.set_theme()
  ```

- [ ] **Load Iris dataset**
  ```python
  data = datasets.load_iris()
  ```

- [ ] **Explore data structure**
  ```python
  print(data.keys())
  print(data['feature_names'])
  print(data['target_names'])
  ```

- [ ] **Create DataFrame**
  ```python
  df = pd.DataFrame(data['data'], columns=data['feature_names'])
  df['target'] = data['target']
  ```

- [ ] **Add human-readable labels**
  ```python
  df['target_name'] = df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
  ```

- [ ] **View basic statistics**
  ```python
  print(df.describe())
  print(df.head())
  ```

---

## ✅ Phase 3: Exploratory Data Analysis (EDA)

- [ ] **Create histograms** (optional)
  ```python
  df['sepal length (cm)'].hist()
  plt.show()
  ```

- [ ] **Create relationship plots**
  ```python
  sns.relplot(x='sepal length (cm)', y='target', hue='target_name', data=df)
  sns.relplot(x='sepal width (cm)', y='target', hue='target_name', data=df)
  plt.show()
  ```

- [ ] **Create pair plot** (all features)
  ```python
  sns.pairplot(df, hue="target_name")
  plt.show()
  ```

- [ ] **Analyze patterns**
  - Which features separate species best?
  - Are there any outliers?
  - What relationships do you see?

---

## ✅ Phase 4: Data Preparation

- [ ] **Split into train/test sets**
  ```python
  from sklearn.model_selection import train_test_split
  df_train, df_test = train_test_split(df, test_size=0.25, random_state=42)
  ```

- [ ] **Verify split**
  ```python
  print(f"Training: {df_train.shape}, Test: {df_test.shape}")
  ```

- [ ] **Separate features and labels**
  ```python
  X_train = df_train.drop(columns=["target", "target_name"]).values
  y_train = df_train["target"].values
  X_test = df_test.drop(columns=["target", "target_name"]).values
  y_test = df_test["target"].values
  ```

- [ ] **Verify shapes**
  ```python
  print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
  print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")
  ```

---

## ✅ Phase 5: Build Baseline Model (Optional)

- [ ] **Create simple rule-based model**
  ```python
  def single_feature_prediction(petal_length):
      if petal_length < 2.5:
          return 0
      elif petal_length < 4.8:
          return 1
      else:
          return 2
  ```

- [ ] **Make predictions**
  ```python
  manual_y_predictions = np.array([single_feature_prediction(val) for val in X_train[:, 2]])
  ```

- [ ] **Calculate baseline accuracy**
  ```python
  manual_model_accuracy = np.mean(manual_y_predictions == y_train)
  print(f"Baseline accuracy: {manual_model_accuracy * 100:.2f}%")
  ```

---

## ✅ Phase 6: Build Machine Learning Model

- [ ] **Import model**
  ```python
  from sklearn.linear_model import LogisticRegression
  ```

- [ ] **Create model instance**
  ```python
  model = LogisticRegression(max_iter=200)
  ```

- [ ] **Quick validation split** (optional, for early testing)
  ```python
  Xt, Xv, yt, yv = train_test_split(X_train, y_train, test_size=0.25)
  model.fit(Xt, yt)
  y_pred = model.predict(Xv)
  print(f"Quick validation accuracy: {np.mean(y_pred == yv) * 100:.2f}%")
  ```

---

## ✅ Phase 7: Evaluate Model with Cross-Validation

- [ ] **Import cross-validation functions**
  ```python
  from sklearn.model_selection import cross_val_score, cross_val_predict
  ```

- [ ] **Perform cross-validation**
  ```python
  model = LogisticRegression(max_iter=200)
  accuracies = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
  ```

- [ ] **Calculate and display results**
  ```python
  print(f"CV Mean Accuracy: {np.mean(accuracies) * 100:.2f}%")
  print(f"CV Std Deviation: {np.std(accuracies) * 100:.2f}%")
  print(f"Individual scores: {accuracies}")
  ```

- [ ] **Get predictions for all samples**
  ```python
  y_pred = cross_val_predict(model, X_train, y_train, cv=5)
  ```

- [ ] **Identify misclassifications**
  ```python
  predicted_correctly_mask = y_pred == y_train
  print(f"Correct predictions: {np.sum(predicted_correctly_mask)}")
  print(f"Wrong predictions: {np.sum(~predicted_correctly_mask)}")
  ```

---

## ✅ Phase 8: Analyze Predictions

- [ ] **Add predictions to DataFrame**
  ```python
  df_predictions = df_train.copy()
  df_predictions["correct_preds"] = predicted_correctly_mask
  df_predictions["pred"] = y_pred
  df_predictions["pred_label"] = df_predictions["pred"].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
  ```

- [ ] **View predictions table**
  ```python
  print(df_predictions.head(20))
  # Or save to CSV
  df_predictions.to_csv('predictions.csv', index=False)
  ```

- [ ] **Visualize predictions vs actual**
  ```python
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
  sns.scatterplot(x="petal length (cm)", y="petal width (cm)", hue="pred_label", data=df_predictions, ax=ax1)
  ax1.set_title("Model Predictions")
  sns.scatterplot(x="petal length (cm)", y="petal width (cm)", hue="target_name", data=df_predictions, ax=ax2)
  ax2.set_title("Actual Species")
  plt.tight_layout()
  plt.savefig('predictions_comparison.png')
  plt.show()
  ```

- [ ] **Analyze misclassifications**
  ```python
  wrong_predictions = df_predictions[~predicted_correctly_mask]
  print(f"Misclassified samples:\n{wrong_predictions}")
  ```

---

## ✅ Phase 9: Model Tuning (Hyperparameter Tuning)

- [ ] **Try different hyperparameter values**
  ```python
  best_score = 0
  best_C = None
  
  for C in [0.1, 0.5, 1.0, 1.5, 2.0, 3.0]:
      model = LogisticRegression(max_iter=200, C=C)
      scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
      mean_score = np.mean(scores)
      print(f"C={C}: {mean_score * 100:.2f}%")
      
      if mean_score > best_score:
          best_score = mean_score
          best_C = C
  
  print(f"\nBest C: {best_C} with accuracy: {best_score * 100:.2f}%")
  ```

- [ ] **Select best hyperparameters**
  ```python
  best_model = LogisticRegression(max_iter=200, C=best_C)
  ```

---

## ✅ Phase 10: Final Model Evaluation

- [ ] **Train final model on ALL training data**
  ```python
  model = LogisticRegression(max_iter=200, C=best_C)  # Use best C from tuning
  model.fit(X_train, y_train)
  ```

- [ ] **Make predictions on test set**
  ```python
  y_test_pred = model.predict(X_test)
  ```

- [ ] **Calculate test set accuracy**
  ```python
  test_set_accuracy = np.mean(y_test_pred == y_test)
  print(f"Test Set Accuracy: {test_set_accuracy * 100:.2f}%")
  ```

- [ ] **Compare train vs test accuracy**
  ```python
  train_pred = model.predict(X_train)
  train_accuracy = np.mean(train_pred == y_train)
  print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
  print(f"Test Accuracy: {test_set_accuracy * 100:.2f}%")
  ```

- [ ] **Check for overfitting**
  - If train accuracy >> test accuracy → overfitting
  - If similar → good generalization

---

## ✅ Phase 11: Final Visualization & Reporting

- [ ] **Create test set predictions DataFrame**
  ```python
  df_predictions_test = df_test.copy()
  df_predictions_test["correct_preds"] = y_test_pred == y_test
  df_predictions_test["pred"] = y_test_pred
  df_predictions_test["pred_label"] = df_predictions_test["pred"].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
  ```

- [ ] **Visualize test set results**
  ```python
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
  sns.scatterplot(x="petal length (cm)", y="petal width (cm)", hue="pred_label", data=df_predictions_test, ax=ax1)
  ax1.set_title("Test Set: Predictions")
  sns.scatterplot(x="petal length (cm)", y="petal width (cm)", hue="target_name", data=df_predictions_test, ax=ax2)
  ax2.set_title("Test Set: Actual")
  plt.tight_layout()
  plt.savefig('test_set_results.png')
  plt.show()
  ```

- [ ] **Save final results**
  ```python
  df_predictions_test.to_csv('test_predictions.csv', index=False)
  ```

- [ ] **Create summary report**
  ```python
  print("=" * 50)
  print("FINAL MODEL SUMMARY")
  print("=" * 50)
  print(f"Model: Logistic Regression")
  print(f"Hyperparameter C: {best_C}")
  print(f"Cross-Validation Accuracy: {np.mean(accuracies) * 100:.2f}%")
  print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
  print(f"Test Accuracy: {test_set_accuracy * 100:.2f}%")
  print(f"Misclassified in test set: {np.sum(y_test_pred != y_test)} out of {len(y_test)}")
  print("=" * 50)
  ```

---

## ✅ Phase 12: Optional Enhancements

- [ ] **Try other algorithms**
  ```python
  from sklearn.tree import DecisionTreeClassifier
  from sklearn.ensemble import RandomForestClassifier
  
  # Compare models
  models = {
      'Logistic Regression': LogisticRegression(max_iter=200),
      'Decision Tree': DecisionTreeClassifier(),
      'Random Forest': RandomForestClassifier()
  }
  
  for name, model in models.items():
      scores = cross_val_score(model, X_train, y_train, cv=5)
      print(f"{name}: {np.mean(scores) * 100:.2f}%")
  ```

- [ ] **Feature importance analysis**
  ```python
  model.fit(X_train, y_train)
  feature_importance = model.coef_
  print("Feature importance:", feature_importance)
  ```

- [ ] **Confusion matrix**
  ```python
  from sklearn.metrics import confusion_matrix, classification_report
  cm = confusion_matrix(y_test, y_test_pred)
  print("Confusion Matrix:")
  print(cm)
  print("\nClassification Report:")
  print(classification_report(y_test, y_test_pred, target_names=['setosa', 'versicolor', 'virginica']))
  ```

---

## 🎯 Quick Reference: Running the Code

### To run your script:
```bash
# 1. Activate environment
conda activate iris-course

# 2. Run script
python setup.py
```

### To see plots:
- If `plt.show()` works → plots appear in window
- If not → uncomment `plt.savefig()` lines to save plots as images

---

## 📊 Expected Outputs

After completing all steps, you should have:

1. ✅ Trained model with good accuracy (>90%)
2. ✅ Cross-validation scores
3. ✅ Test set accuracy
4. ✅ Visualization plots (saved as PNG files)
5. ✅ Prediction DataFrames (saved as CSV files)
6. ✅ Understanding of where model makes mistakes

---

## 🎓 Learning Goals Checklist

- [ ] Understand train/test split
- [ ] Understand cross-validation
- [ ] Know how to evaluate model performance
- [ ] Understand hyperparameter tuning
- [ ] Can visualize predictions vs actual
- [ ] Can identify misclassifications
- [ ] Understand overfitting vs generalization

---

**Good luck with your machine learning project!** 🚀

