# User Guide

This guide covers the main features and usage patterns of RealLabelNormalization.jl.

## Basic Usage

### Single Target Normalization

For regression tasks with a single target variable:

```julia
using RealLabelNormalization

# Your training labels
train_labels = [1.2, 5.8, 3.4, 8.1, 2.3, 100.5]  # Note: 100.5 is an outlier

# Normalize with default settings (min-max to `[-1,1]` with outlier clipping)
normalized = normalize_labels(train_labels)
```

### Multi-Target Normalization

For regression tasks with multiple target variables:

```julia
# Matrix where each column is a different target variable
train_labels = [1.0 10.0 100.0;
                5.0 20.0 200.0;
                3.0 15.0 150.0;
                8.0 25.0 250.0]

# Normalize each target independently
normalized = normalize_labels(train_labels; mode=:columnwise)

# Or normalize globally across all targets
normalized = normalize_labels(train_labels; mode=:global)
```

## Normalization Methods

### Min-Max Normalization

Scales values to a specified range (default: `[-1, 1]`):

```julia
# Default: scale to `[-1, 1]`
normalized = normalize_labels(labels)

# Scale to [0, 1]
normalized = normalize_labels(labels; range=(0, 1))

# Custom range
normalized = normalize_labels(labels; range=(-2, 2))
```

### Z-Score Normalization

Standardizes values to have zero mean and unit variance:

```julia
normalized = normalize_labels(labels; method=:zscore)
# Results in approximately: mean ≈ 0, std ≈ 1
```

## Outlier Handling

### Quantile-Based Clipping

By default, outliers are clipped to the 1st and 99th percentiles before normalization:

```julia
# Default: clip to 1st-99th percentiles
normalized = normalize_labels(labels)

# More aggressive clipping
normalized = normalize_labels(labels; clip_quantiles=(0.05, 0.95))

# No clipping
normalized = normalize_labels(labels; clip_quantiles=nothing)
```

### Why Clip Outliers?

Outliers can severely distort normalization, especially min-max scaling:

```julia
labels = [1, 2, 3, 4, 5, 1000]  # 1000 is an outlier

# Without clipping - outlier dominates the scaling
no_clip = normalize_labels(labels; clip_quantiles=nothing)
# Result: [≈-1, ≈-1, ≈-1, ≈-1, ≈-1, 1] - poor distribution

# With clipping - better distribution
with_clip = normalize_labels(labels)
# Result: more evenly distributed values in `[-1, 1]`
```

## Handling Missing Data (NaN)

The package gracefully handles NaN values:

```julia
labels_with_nan = [1.0, 2.0, NaN, 4.0, 5.0]
normalized = normalize_labels(labels_with_nan)
# NaN values are preserved, statistics computed on valid data only
```

## Train/Test Consistency

For machine learning workflows, it's crucial to use the same normalization parameters across train/validation/test splits:

```julia
# Step 1: Compute normalization statistics on training data only
train_labels = [1.0, 2.0, 3.0, 4.0, 5.0]
stats = compute_normalization_stats(train_labels)

# Step 2: Apply to training data
train_normalized = apply_normalization(train_labels, stats)

# Step 3: Apply same statistics to test data
test_labels = [1.5, 2.5, 3.5, 6.0]  # Different distribution
test_normalized = apply_normalization(test_labels, stats)

# Step 4: Denormalize predictions back to original scale
predictions_normalized = [-0.2, 0.3, 0.8]
predictions_original = denormalize_labels(predictions_normalized, stats)
```

## Best Practices

### When to Use Each Method

- **Min-max normalization**: When you know the expected range of your data or want bounded outputs
- **Z-score normalization**: When your data is approximately normally distributed
- **Global mode**: When all targets should be on the same scale (e.g., related measurements)
- **Column-wise mode**: When targets represent different quantities with different scales

### Recommended Workflow

```julia
# 1. Split your data first
train_indices = 1:800
test_indices = 801:1000

# 2. Compute stats only on training data
stats = compute_normalization_stats(labels[train_indices])

# 3. Apply to all splits
train_normalized = apply_normalization(labels[train_indices], stats)
test_normalized = apply_normalization(labels[test_indices], stats)

# 4. Train your model on normalized data
# ... train model ...

# 5. Denormalize predictions
predictions_original = denormalize_labels(model_predictions, stats)
```

### Handling Extreme Outliers

For datasets with extreme outliers, consider more aggressive clipping:

```julia
# For data with extreme outliers (e.g., financial data)
normalized = normalize_labels(labels; clip_quantiles=(0.1, 0.9))

# For very clean data, you might skip clipping
normalized = normalize_labels(labels; clip_quantiles=nothing)
```
