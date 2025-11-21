# Understanding `plt.subplots(1, 2, figsize=(12, 5))`

## The Line
```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
```

## What It Does
**Creates a figure with 2 side-by-side subplots (plots) that you can draw on.**

---

## Part-by-Part Breakdown

### 1. `plt.subplots()`
- **Function**: Creates a figure and subplots
- **Returns**: Two things (that's why we unpack them)

### 2. `(1, 2)`
- **First number (1)**: Number of **rows** (1 row)
- **Second number (2)**: Number of **columns** (2 columns)
- **Result**: Creates a 1×2 grid (2 plots side by side)

**Visual:**
```
┌─────────────┬─────────────┐
│   Plot 1    │   Plot 2    │
│   (ax1)     │   (ax2)     │
└─────────────┴─────────────┘
```

### 3. `figsize=(12, 5)`
- **First number (12)**: Width in **inches** (12 inches wide)
- **Second number (5)**: Height in **inches** (5 inches tall)
- **Result**: Creates a figure that's 12 inches wide by 5 inches tall

### 4. `fig, (ax1, ax2) =`
- **Unpacking**: The function returns 2 things
- **`fig`**: The figure object (the whole window/canvas)
- **`(ax1, ax2)`**: Two axes objects (individual plots you can draw on)

---

## What Each Variable Does

### `fig` (Figure)
- The **entire window/canvas**
- Contains both subplots
- Use for: `fig.savefig()`, `fig.suptitle()`, overall figure settings

### `ax1` (First Axis)
- The **left plot**
- Use for: `ax1.plot()`, `ax1.scatter()`, `ax1.set_title()`, etc.
- Draw on this for your first visualization

### `ax2` (Second Axis)
- The **right plot**
- Use for: `ax2.plot()`, `ax2.scatter()`, `ax2.set_title()`, etc.
- Draw on this for your second visualization

---

## Visual Representation

```
┌─────────────────────────────────────────────────────────┐
│                         fig                              │
│                    (The whole figure)                     │
│                                                          │
│  ┌──────────────────────┬──────────────────────┐      │
│  │       ax1            │       ax2            │      │
│  │   (Left plot)        │   (Right plot)       │      │
│  │                      │                      │      │
│  │  Your plot here      │  Your plot here      │      │
│  │                      │                      │      │
│  └──────────────────────┴──────────────────────┘      │
│                                                          │
└─────────────────────────────────────────────────────────┘
     12 inches wide × 5 inches tall
```

---

## Example Usage

```python
# Create the figure and axes
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Draw on the first plot (left)
ax1.scatter(x1, y1, hue="species1")
ax1.set_title("Plot 1: Predictions")
ax1.set_xlabel("Petal Length")

# Draw on the second plot (right)
ax2.scatter(x2, y2, hue="species2")
ax2.set_title("Plot 2: Actual Labels")
ax2.set_xlabel("Petal Length")

# Adjust layout so nothing overlaps
plt.tight_layout()

# Show or save
plt.show()
```

---

## Different Configurations

### 1 Row × 2 Columns (Your code):
```python
fig, (ax1, ax2) = plt.subplots(1, 2)
```
**Result:** 2 plots side by side
```
[Plot 1] [Plot 2]
```

### 2 Rows × 1 Column:
```python
fig, (ax1, ax2) = plt.subplots(2, 1)
```
**Result:** 2 plots stacked vertically
```
[Plot 1]
[Plot 2]
```

### 2 Rows × 2 Columns:
```python
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
```
**Result:** 4 plots in a 2×2 grid
```
[Plot 1] [Plot 2]
[Plot 3] [Plot 4]
```

---

## Why Use Subplots?

### Without Subplots (Separate Figures):
```python
plt.figure()
sns.scatterplot(...)  # Plot 1
plt.show()

plt.figure()
sns.scatterplot(...)  # Plot 2
plt.show()
```
- Creates 2 separate windows
- Hard to compare side by side

### With Subplots (Same Figure):
```python
fig, (ax1, ax2) = plt.subplots(1, 2)
sns.scatterplot(..., ax=ax1)  # Plot 1
sns.scatterplot(..., ax=ax2)  # Plot 2
plt.show()
```
- Creates 1 window with 2 plots
- Easy to compare side by side

---

## The `figsize` Parameter

### `figsize=(12, 5)`
- **12 inches wide**: Wide enough for side-by-side plots
- **5 inches tall**: Standard height

### Common Sizes:
```python
figsize=(10, 5)   # Medium, wide
figsize=(12, 6)   # Large, wide
figsize=(8, 6)    # Medium, square-ish
figsize=(6, 4)    # Small, standard
```

**Rule of thumb:** Wider for side-by-side plots, taller for stacked plots

---

## Complete Example from Your Code

```python
# Create figure with 2 side-by-side plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Model Predictions
sns.scatterplot(
    x="petal length (cm)", 
    y="petal width (cm)", 
    hue="pred_label", 
    data=df_predictions, 
    ax=ax1  # ← Draw on first plot (left)
)
ax1.set_title("Model Predictions")

# Plot 2: Actual Labels
sns.scatterplot(
    x="petal length (cm)", 
    y="petal width (cm)", 
    hue="target_name", 
    data=df_predictions, 
    ax=ax2  # ← Draw on second plot (right)
)
ax2.set_title("Actual Species")

plt.tight_layout()  # Adjust spacing
plt.show()
```

---

## Key Points

### 1. **Unpacking**
```python
fig, (ax1, ax2) = plt.subplots(1, 2)
```
- Function returns 2 things
- `fig` = whole figure
- `(ax1, ax2)` = individual plots

### 2. **Grid Layout**
```python
(1, 2)  # 1 row, 2 columns = side by side
(2, 1)  # 2 rows, 1 column = stacked
(2, 2)  # 2 rows, 2 columns = 2×2 grid
```

### 3. **Figure Size**
```python
figsize=(12, 5)  # (width, height) in inches
```

### 4. **Using Axes**
```python
ax1.plot(...)      # Draw on first plot
ax2.scatter(...)   # Draw on second plot
ax1.set_title(...) # Set title on first plot
```

---

## Summary

**`fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))`**

- **`plt.subplots()`**: Creates figure and subplots
- **`(1, 2)`**: 1 row, 2 columns (side by side)
- **`figsize=(12, 5)`**: 12 inches wide, 5 inches tall
- **`fig`**: The whole figure/canvas
- **`ax1, ax2`**: Two individual plots you can draw on

**Result:** One figure with 2 plots side by side, perfect for comparing predictions vs actual labels! 🎯


