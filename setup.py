###Notebook setup

# standard imports
import os

# Third-party imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets

sns.set_theme();

### Load Data

data = datasets.load_iris()
data.keys()
# print('data', data['data'][:5])
# print(data['feature_names'])
# print(data['target_names'])


### Create pandas dataframe

df = pd.DataFrame(data['data'], columns=data['feature_names'])
df['target'] = data['target']
# print(pd.DataFrame(data['data'], columns=data['feature_names']))
# print(data['target'])
print('df', df)

### Basic descriptive stats
# print(df.describe())

### Distribution of features and target

# df['sepal length (cm)'].hist()
# plt.show()

### Relationships of data features w/ target

df['target_name'] = df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
# print(df)

col = 'sepal length (cm)'
sns.relplot(x=col, y='target', hue='target_name', data=df)
_ = plt.suptitle(col, y=1.05)
plt.show()

col = 'sepal width (cm)'
sns.relplot(x=col, y='target', hue='target_name', data=df)
_ = plt.suptitle(col, y=1.05)
plt.show()

### Exploratory data analysis (EDA)

sns.pairplot(df, hue="target_name")
# plt.show()

###Train test split

from sklearn.model_selection import train_test_split

df_train, df_test = train_test_split(df, test_size=0.25)
# print(df_train.shape)
# print(df_test.shape)
# print(df_train.head())

### Prepare data for modeling
X_train = df_train.drop(columns=["target", "target_name"]).values
y_train = df_train["target"].values
# print('x train', X_train)
# print(X_train.shape)

### Modeling - simple manual model

def single_feature_prediction(petal_length):
    """Predicts the iris species given the petal length"""
    if petal_length < 2.5:
        return 0
    elif petal_length < 4.8:
        return 1
    else:
        return 2

manual_y_predictions = np.array([single_feature_prediction(val) for val in X_train[:, 2]])
# print(manual_y_predictions)

# print(manual_y_predictions == y_train)

manual_model_accuracy = np.mean(manual_y_predictions == y_train)
# print(f"manual_model_accuracy: {manual_model_accuracy}")

### Modeling logistic regression

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=200)
# print(model.fit(X_train, y_train))
Xt, Xv, yt, yv = train_test_split(X_train, y_train, test_size=0.25)
# print(Xt.shape)

model.fit(Xt, yt)
# print(model.predict(Xv))
# y_pred = model.predict(Xv)
# print('yv', yv)
# print('y pred', y_pred)
# print(y_pred == yv)

# print(np.mean(y_pred == yv))


# model.score()

### Using cross-validation to evaluate our model

from sklearn.model_selection import cross_val_score, cross_val_predict

model = LogisticRegression(max_iter=200)

accuracies = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")

# print(np.mean(accuracies))

### Where are we misclassifying points?

y_pred = cross_val_predict(model, X_train, y_train, cv=5)
# print(y_pred)

predicted_correctly_mask = y_pred == y_train
# print('pred correct', predicted_correctly_mask)

# print('x train pred', X_train[predicted_correctly_mask])

not_predicted_correctly = ~predicted_correctly_mask
# print('correct mask', ~predicted_correctly_mask)

# print(X_train[not_predicted_correctly])

df_predictions = df_train.copy()

df_predictions["correct_preds"] = predicted_correctly_mask

# print(df_predictions.head())

df_predictions["pred"] = y_pred

df_predictions["pred_label"] = df_predictions["pred"].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

# Option 1: Show all rows (set pandas display option)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
# print(df_predictions)

# Option 2: Save to CSV file (uncomment to use)
# df_predictions.to_csv('predictions.csv', index=False)
# print("Saved to predictions.csv")

# Side-by-side plots: predictions vs actual
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Model Predictions
sns.scatterplot(x="petal length (cm)", y="petal width (cm)", hue="pred_label", data=df_predictions, ax=ax1)
ax1.set_title("Model Predictions")
ax1.legend(title="Predicted Species")

# Plot 2: Actual Labels
sns.scatterplot(x="petal length (cm)", y="petal width (cm)", hue="target_name", data=df_predictions, ax=ax2)
ax2.set_title("Actual Species")
ax2.legend(title="Actual Species")

plt.tight_layout()

# Save plot to file (alternative if plt.show() doesn't work)
# plt.savefig('scatterplot.png', dpi=150, bbox_inches='tight')
# print("Plot saved to scatterplot.png")

# Display plot (may not work on some macOS terminals)
# plt.show()  # Required to display the plot

def plot_incorrect_predictions(df_predictions, x_axis_feature, y_axis_feature):
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    axs = axs.flatten()
    sns.scatterplot(x=x_axis_feature, y=y_axis_feature, hue="pred_label", data=df_predictions, ax=axs[0])
    sns.scatterplot(x=x_axis_feature, y=y_axis_feature, hue="target_name", data=df_predictions, ax=axs[1])
    sns.scatterplot(x=x_axis_feature, y=y_axis_feature, hue="correct_preds", data=df_predictions, ax=axs[2])
    axs[3].set_visible(False)
    # plt.show()

plot_incorrect_predictions(df_predictions, "petal length (cm)", "petal width (cm)")


# fig, axs = plt.subplots(2, 2, figsize=(10, 10))
# axs = axs.flatten()
# print(axs.shape)
# plt.show()

### Model tuning

# from sklearn.ensemble import RandomForestClassifier

# model = RandomForestClassifier()

for reg_param in (1, 1.3, 1.8, 2, 2.3, 2.9, 3):
    # print(reg_param)
    model = LogisticRegression(max_iter=200, C=reg_param)
    accuracies = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
    # print(f"Accuracy: {np.mean(accuracies) * 100: .2f}%")


### Final model

model = LogisticRegression(max_iter=200, C=0.9)

X_test = df_test.drop(columns=["target", "target_name"]).values
y_test = df_test["target"].values

# print(X_test.shape)
# print(y_test)

### Train final model using full training data set
model.fit(X_train, y_train)

y_test_pred = model.predict(X_test)
test_set_correctly_classified = y_test_pred == y_test
test_set_accuracy = np.mean(test_set_correctly_classified)

# print(f"Test set accuracy: {test_set_accuracy * 100: .2f}")
# print(test_set_correctly_classified)

df_predictions_test = df_test.copy()
df_predictions_test["correct_preds"] = test_set_correctly_classified
df_predictions_test["pred"] = y_test_pred
df_predictions_test["pred_label"] = df_predictions_test["pred"].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

# print(df_predictions_test.head())

plot_incorrect_predictions(df_predictions_test, x_axis_feature="petal length (cm)", y_axis_feature="petal width (cm)")

 

