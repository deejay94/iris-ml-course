# What is `sns.set_theme()`?

## The Line
```python
sns.set_theme()
```

## What It Does
**Sets seaborn's default visual style for all plots - makes them look better automatically!**

---

## Quick Answer
**It applies seaborn's modern, attractive default theme to all your plots.** Makes them look professional without any extra work.

---

## What It Changes

### Without `sns.set_theme()`:
- Default matplotlib style (more basic)
- Simpler colors
- Basic grid
- Standard fonts

### With `sns.set_theme()`:
- Modern seaborn style
- Better color palette
- Nicer grid (subtle, not distracting)
- Better fonts
- More professional appearance

---

## Visual Comparison

### Before (matplotlib default):
```
Simple plot with basic styling
- Plain white background
- Basic grid lines
- Standard colors
```

### After (seaborn theme):
```
Modern plot with professional styling
- Light gray background
- Subtle grid lines
- Better color palette
- Improved typography
```

---

## What Gets Applied

### 1. **Color Palette**
- More attractive default colors
- Better contrast
- Colorblind-friendly options

### 2. **Grid Style**
- Subtle grid lines (not too bold)
- Helps read values without being distracting
- Professional appearance

### 3. **Background**
- Light gray background (instead of pure white)
- Easier on the eyes
- More modern look

### 4. **Typography**
- Better font choices
- Improved readability

### 5. **Overall Aesthetics**
- More polished appearance
- Publication-ready style

---

## Example

### Without `sns.set_theme()`:
```python
import matplotlib.pyplot as plt
import seaborn as sns

# No theme set
sns.scatterplot(x="petal length", y="petal width", data=df)
plt.show()
# Result: Basic matplotlib style
```

### With `sns.set_theme()`:
```python
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()  # ← Apply theme
sns.scatterplot(x="petal length", y="petal width", data=df)
plt.show()
# Result: Modern seaborn style
```

---

## Customization Options

You can customize the theme:

```python
# Default (what you're using)
sns.set_theme()

# Or customize:
sns.set_theme(
    style="darkgrid",      # Background style
    palette="husl",        # Color palette
    font_scale=1.2,        # Font size
    rc={"figure.figsize": (10, 6)}  # Figure size
)
```

### Style Options:
- `"darkgrid"` - Dark grid on light background (default)
- `"whitegrid"` - White grid on light background
- `"dark"` - Dark background
- `"white"` - White background
- `"ticks"` - Minimal style with ticks

---

## When to Use It

### ✅ Use it when:
- You want better-looking plots automatically
- You're creating multiple plots (applies to all)
- You want a consistent style
- You want publication-ready plots

### ✅ Best practice:
- Call it **once at the beginning** of your script
- Applies to all subsequent plots
- No need to call it for each plot

---

## In Your Code

```python
import seaborn as sns

sns.set_theme()  # ← Set once at the beginning

# All plots after this will use the theme:
sns.relplot(...)      # Uses theme
sns.pairplot(...)    # Uses theme
sns.scatterplot(...)  # Uses theme
```

**All your plots will have the modern seaborn style!**

---

## Common Patterns

### Pattern 1: Default theme (your code)
```python
sns.set_theme()  # Modern, professional style
```

### Pattern 2: Custom style
```python
sns.set_theme(style="whitegrid", palette="muted")
```

### Pattern 3: Reset to default
```python
sns.set_theme()  # Reset to defaults
```

---

## Comparison with matplotlib

### Matplotlib default:
```python
import matplotlib.pyplot as plt
plt.scatter(x, y)  # Basic style
```

### Seaborn with theme:
```python
import seaborn as sns
sns.set_theme()
sns.scatterplot(x=x, y=y)  # Modern style
```

**Seaborn theme makes plots look more professional automatically!**

---

## What Happens Behind the Scenes

When you call `sns.set_theme()`, seaborn:
1. Sets matplotlib's style parameters
2. Applies color palette
3. Configures grid style
4. Sets typography preferences
5. Applies these to all future plots

**It's a one-time setup that affects all plots!**

---

## Summary

**`sns.set_theme()`**

- **What it does:** Applies seaborn's modern, attractive default theme
- **When to use:** Once at the beginning of your script
- **Effect:** All subsequent plots get better styling automatically
- **Result:** More professional, publication-ready plots

**Think of it as:** "Make all my plots look modern and professional automatically!"

**It's a best practice** to include this at the start of your visualization code! 🎨

