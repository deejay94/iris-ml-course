"""
Demo script to show how train_test_split maintains alignment between X and y
Run this to see the correspondence visually!
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

print("=" * 60)
print("DEMO: How train_test_split maintains X-y alignment")
print("=" * 60)

# Create simple example data
# X = features (let's use 2 features for simplicity)
# y = labels (species: 0, 1, 2)
np.random.seed(42)
n_samples = 12

X = np.array([
    [5.1, 3.5],  # Sample 0
    [4.9, 3.0],  # Sample 1
    [4.7, 3.2],  # Sample 2
    [6.4, 3.2],  # Sample 3
    [5.5, 2.3],  # Sample 4
    [6.7, 3.0],  # Sample 5
    [5.0, 3.6],  # Sample 6
    [5.4, 3.9],  # Sample 7
    [6.1, 2.8],  # Sample 8
    [5.8, 2.7],  # Sample 9
    [6.2, 3.4],  # Sample 10
    [5.9, 3.0],  # Sample 11
])

y = np.array([0, 0, 0, 1, 1, 2, 0, 0, 1, 1, 2, 2])  # Labels

print("\n📊 ORIGINAL DATA (before split):")
print("-" * 60)
df_original = pd.DataFrame({
    'Sample': range(len(X)),
    'Feature 1': X[:, 0],
    'Feature 2': X[:, 1],
    'Label': y,
    'Species': ['setosa' if label == 0 else 'versicolor' if label == 1 else 'virginica' for label in y]
})
print(df_original.to_string(index=False))

# Split the data
Xt, Xv, yt, yv = train_test_split(X, y, test_size=0.25, random_state=42)

print("\n\n✅ TRAINING SET (Xt, yt):")
print("-" * 60)
print("Notice: Each row in Xt corresponds to the same row in yt!")
print()
df_train = pd.DataFrame({
    'Index': range(len(Xt)),
    'Feature 1': Xt[:, 0],
    'Feature 2': Xt[:, 1],
    'Label': yt,
    'Species': ['setosa' if label == 0 else 'versicolor' if label == 1 else 'virginica' for label in yt],
    'Original Sample': [i for i in range(len(X)) if i not in [2, 5, 11]]  # These went to validation
})
print(df_train.to_string(index=False))

print("\n\n✅ VALIDATION SET (Xv, yv):")
print("-" * 60)
print("Notice: Each row in Xv corresponds to the same row in yv!")
print()
df_val = pd.DataFrame({
    'Index': range(len(Xv)),
    'Feature 1': Xv[:, 0],
    'Feature 2': Xv[:, 1],
    'Label': yv,
    'Species': ['setosa' if label == 0 else 'versicolor' if label == 1 else 'virginica' for label in yv],
    'Original Sample': [2, 5, 11]  # These went to validation
})
print(df_val.to_string(index=False))

print("\n\n🔍 VERIFICATION:")
print("-" * 60)
print("Let's verify that Xv[0] and yv[0] match the original data:")
print(f"  Xv[0] = {Xv[0]} (features)")
print(f"  yv[0] = {yv[0]} (label)")
print(f"  Original sample {df_val.iloc[0]['Original Sample']}: X = {X[int(df_val.iloc[0]['Original Sample'])]}, y = {y[int(df_val.iloc[0]['Original Sample'])]}")
print(f"  ✓ Match!" if np.array_equal(Xv[0], X[int(df_val.iloc[0]['Original Sample'])]) and yv[0] == y[int(df_val.iloc[0]['Original Sample'])] else "  ✗ No match")

print("\n\n💡 KEY INSIGHT:")
print("-" * 60)
print("When you do: model.predict(Xv)")
print("  - prediction[0] is for Xv[0]")
print("When you compare: prediction[0] == yv[0]")
print("  - You're checking if the prediction for Xv[0] matches the true label yv[0]")
print("  - This works because Xv[0] and yv[0] are the same sample!")
print("\n" + "=" * 60)

