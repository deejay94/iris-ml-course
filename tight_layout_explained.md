# What is `plt.tight_layout()`?

## The Line
```python
plt.tight_layout()
```

## What It Does
**Automatically adjusts spacing between subplots and figure edges to prevent overlap.**

---

## The Problem It Solves

### Without `tight_layout()`:
```
┌─────────────────────────────────────┐
│  Title might get cut off            │
│  ┌──────────┬──────────┐           │
│  │ Plot 1   │ Plot 2   │           │
│  │          │          │           │
│  │          │          │           │
│  └──────────┴──────────┘           │
│  Labels might overlap              │
│  X-axis labels cut off             │
└─────────────────────────────────────┘
```

### With `tight_layout()`:
```
┌─────────────────────────────────────┐
│                                     │
│  ┌──────────┬──────────┐          │
│  │ Plot 1   │ Plot 2   │          │
│  │          │          │          │
│  │          │          │          │
│  └──────────┴──────────┘          │
│  X-axis labels visible             │
│  Y-axis labels visible             │
│  Titles properly positioned        │
└─────────────────────────────────────┘
```

---

## What It Adjusts

### 1. **Spacing Between Subplots**
- Prevents plots from being too close together
- Adds padding between multiple plots

### 2. **Figure Margins**
- Ensures labels aren't cut off at edges
- Keeps titles visible
- Makes axes labels readable

### 3. **Legend Positioning**
- Prevents legends from overlapping plots
- Keeps legends within figure bounds

---

## Visual Example

### Before `tight_layout()`:
```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.set_title("Model Predictions")
ax1.set_xlabel("Petal Length (cm)")  # Might get cut off
ax2.set_title("Actual Species")
plt.show()  # Labels might overlap or be cut off
```

**Problems:**
- X-axis labels might be cut off
- Y-axis labels might be cut off
- Titles might overlap
- Legends might be outside the figure

### After `tight_layout()`:
```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.set_title("Model Predictions")
ax1.set_xlabel("Petal Length (cm)")
ax2.set_title("Actual Species")
plt.tight_layout()  # ← Adjusts spacing automatically
plt.show()  # Everything fits nicely!
```

**Fixed:**
- All labels visible
- Titles properly positioned
- Legends within bounds
- No overlapping elements

---

## When to Use It

### ✅ Always use when:
- You have multiple subplots
- You have titles, labels, or legends
- Things might overlap or get cut off

### ✅ Especially important for:
- Multiple subplots (like your 2 side-by-side plots)
- Long axis labels
- Legends
- Titles

### ❌ Not always needed:
- Single simple plot (though it doesn't hurt)
- Very simple plots with short labels

---

## In Your Code Context

```python
# Create 2 side-by-side plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Add plots with titles and legends
sns.scatterplot(..., ax=ax1)
ax1.set_title("Model Predictions")
ax1.legend(title="Predicted Species")

sns.scatterplot(..., ax=ax2)
ax2.set_title("Actual Species")
ax2.legend(title="Actual Species")

# Adjust spacing so nothing overlaps or gets cut off
plt.tight_layout()  # ← Critical for multiple subplots!
```

**Without `tight_layout()`:**
- Legends might overlap plots
- Titles might be too close
- Axis labels might be cut off

**With `tight_layout()`:**
- Everything fits nicely
- All labels visible
- Professional appearance

---

## Alternative: Manual Adjustment

You can also manually adjust spacing:

```python
plt.subplots_adjust(
    left=0.1,      # Left margin
    right=0.9,     # Right margin
    top=0.9,       # Top margin
    bottom=0.1,    # Bottom margin
    wspace=0.3,    # Width spacing between subplots
    hspace=0.3     # Height spacing between subplots
)
```

But `tight_layout()` does this automatically! Much easier.

---

## Parameters (Optional)

You can customize `tight_layout()`:

```python
plt.tight_layout(
    pad=1.0,        # Padding around subplots
    w_pad=0.5,      # Width padding between subplots
    h_pad=0.5       # Height padding between subplots
)
```

**Usually you don't need these** - default works great!

---

## Common Pattern

```python
# Create figure and subplots
fig, axes = plt.subplots(...)

# Add plots, titles, labels
# ...

# Adjust layout (almost always do this!)
plt.tight_layout()

# Show or save
plt.show()
# or
plt.savefig('plot.png')
```

---

## Summary

**`plt.tight_layout()`**

- **What it does:** Automatically adjusts spacing in your figure
- **Why you need it:** Prevents overlap and cut-off labels/titles
- **When to use:** Almost always, especially with multiple subplots
- **Result:** Clean, professional-looking plots where everything is visible

**Think of it as:** "Make everything fit nicely so nothing gets cut off or overlaps!"

**It's a best practice** to include this when you have multiple subplots! 🎯


