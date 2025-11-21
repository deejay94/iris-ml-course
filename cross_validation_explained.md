# How Cross-Validation Works (`cv=5`)

## Your Understanding (Almost Right!)
You said: "It's splitting it up into 5 parts and using a different part each time to test the data."

**You're close!** But there's a bit more to it...

---

## What Actually Happens

### Step 1: Split into 5 Folds (Parts)
```
Your training data (112 samples)
    ↓
Split into 5 equal parts:
┌─────────┬─────────┬─────────┬─────────┬─────────┐
│ Fold 1  │ Fold 2  │ Fold 3  │ Fold 4  │ Fold 5  │
│ (22-23) │ (22-23) │ (22-23) │ (22-23) │ (22-23) │
└─────────┴─────────┴─────────┴─────────┴─────────┘
```

### Step 2: 5 Iterations (Each Fold Tests Once)

**Iteration 1:**
```
Training: Fold 2 + Fold 3 + Fold 4 + Fold 5  (4 folds = 90 samples)
Testing:  Fold 1                             (1 fold = 22 samples)
↓
Train model on 4 folds, test on Fold 1 → Get score 1
```

**Iteration 2:**
```
Training: Fold 1 + Fold 3 + Fold 4 + Fold 5  (4 folds = 90 samples)
Testing:  Fold 2                             (1 fold = 22 samples)
↓
Train model on 4 folds, test on Fold 2 → Get score 2
```

**Iteration 3:**
```
Training: Fold 1 + Fold 2 + Fold 4 + Fold 5  (4 folds = 90 samples)
Testing:  Fold 3                             (1 fold = 22 samples)
↓
Train model on 4 folds, test on Fold 3 → Get score 3
```

**Iteration 4:**
```
Training: Fold 1 + Fold 2 + Fold 3 + Fold 5  (4 folds = 90 samples)
Testing:  Fold 4                             (1 fold = 22 samples)
↓
Train model on 4 folds, test on Fold 4 → Get score 4
```

**Iteration 5:**
```
Training: Fold 1 + Fold 2 + Fold 3 + Fold 4  (4 folds = 90 samples)
Testing:  Fold 5                             (1 fold = 22 samples)
↓
Train model on 4 folds, test on Fold 5 → Get score 5
```

### Step 3: Return All 5 Scores
```python
accuracies = [score1, score2, score3, score4, score5]
# Example: [0.95, 0.97, 0.96, 0.94, 0.96]
```

---

## Visual Diagram

```
┌─────────────────────────────────────────────────────────┐
│              Your Training Data (112 samples)            │
└─────────────────────────────────────────────────────────┘
                    ↓ Split into 5 folds
┌──────────┬──────────┬──────────┬──────────┬──────────┐
│  Fold 1  │  Fold 2  │  Fold 3  │  Fold 4  │  Fold 5  │
│   (22)   │   (23)   │   (22)   │   (23)   │   (22)   │
└──────────┴──────────┴──────────┴──────────┴──────────┘

Iteration 1:  [Train] [Train] [Train] [Train] [Test]  → Score 1
Iteration 2:  [Train] [Test]  [Train] [Train] [Train]  → Score 2
Iteration 3:  [Train] [Train] [Test]  [Train] [Train]  → Score 3
Iteration 4:  [Train] [Train] [Train] [Test]  [Train]  → Score 4
Iteration 5:  [Train] [Train] [Train] [Train] [Test]   → Score 5
                                                          ↓
                                    accuracies = [score1, score2, score3, score4, score5]
```

---

## Key Points

### ✅ What You Got Right:
- "Splitting into 5 parts" → Correct!
- "Using a different part each time to test" → Correct!

### 📝 What to Add:
- **Each part gets to be the test set ONCE**
- **The other 4 parts are used for training each time**
- **You get 5 scores** (one for each iteration)
- **More reliable** than a single train/test split

---

## Why This is Better Than Single Split

### Single Split (What you did earlier):
```python
Xt, Xv, yt, yv = train_test_split(X_train, y_train, test_size=0.25)
```
- Split once: 75% train, 25% test
- Get 1 score
- **Problem:** Might get lucky/unlucky with that particular split

### Cross-Validation (cv=5):
```python
accuracies = cross_val_score(model, X_train, y_train, cv=5)
```
- Split 5 times (each fold tests once)
- Get 5 scores
- **Benefit:** More reliable, see if model is consistent

---

## Example Output

```python
accuracies = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
print(accuracies)
# Output: [0.95, 0.97, 0.96, 0.94, 0.96]

print(np.mean(accuracies))
# Output: 0.956 (95.6% average accuracy)
```

**What this tells you:**
- Model is consistent (all scores around 0.95-0.97)
- Average accuracy is 95.6%
- More reliable than a single score

---

## The Complete Process

```python
# Step 1: Split data into 5 folds
fold1, fold2, fold3, fold4, fold5 = split_into_5_folds(X_train, y_train)

# Step 2: 5 iterations
for i in range(5):
    # Use fold i as test, others as train
    train_folds = [all folds except fold i]
    test_fold = fold i
    
    # Train model
    model.fit(train_folds)
    
    # Test model
    score = model.score(test_fold)
    scores.append(score)

# Step 3: Return all scores
return scores  # [score1, score2, score3, score4, score5]
```

---

## Comparison

| Method | Train/Test Split | Scores | Reliability |
|--------|-----------------|--------|-------------|
| **Single Split** | 75%/25% once | 1 score | Can be lucky/unlucky |
| **Cross-Validation (cv=5)** | 80%/20% five times | 5 scores | More reliable |

---

## Summary

**Your understanding:**
> "Splitting into 5 parts and using a different part each time to test"

**More precisely:**
> "Splitting into 5 parts, and **each part gets to be the test set once** (while the other 4 parts train), giving you **5 scores** instead of just 1"

**Why it's better:**
- More reliable than single split
- Uses all data for both training and testing (just at different times)
- See consistency across different data splits

---

## Takeaway

✅ You understood the core concept!  
📝 The refinement: Each fold tests **once**, you get **5 scores**, and it's **more reliable** than a single split.

Great question! 🎯


