# User Guide

This guide covers the **stats-based workflow** for leak-free label normalization in machine learning.

## ⚠️ Critical: Always Use the Stats-Based Workflow

**NEVER** use `normalize_labels()` directly on your full dataset. This causes data leakage! Instead, follow this pattern:

1. **Compute stats from training data ONLY**
2. **Apply the same stats to validation/test data**
3. **Denormalize predictions using the same stats**

## The Three-Step Pattern

### Step 1: Compute Normalization Statistics (Training Data Only)

```julia
using RealLabelNormalization

# Your training labels (with outliers)
train_labels = [1.2, 5.8, 3.4, 8.1, 2.3, 100.5]  # 100.5 is an outlier

# Compute stats from training data ONLY
stats = compute_normalization_stats(train_labels; method=:zscore, clip_quantiles=(0.01, 0.99))
```

### Step 2: Apply Stats to All Data Splits

```julia
# Apply SAME stats to training data
train_normalized = apply_normalization(train_labels, stats)

# Apply SAME stats to validation data
val_labels = [2.1, 4.3, 6.7, 9.2]
val_normalized = apply_normalization(val_labels, stats)

# Apply SAME stats to test data  
test_labels = [1.8, 5.1, 7.3]
test_normalized = apply_normalization(test_labels, stats)
```

### Step 3: Denormalize Predictions

```julia
# After training your model on normalized data...
predictions_normalized = model(X_test)  # Model outputs normalized predictions

# Convert back to original scale using SAME stats
predictions_original = denormalize_labels(predictions_normalized, stats)
```

## Multi-Target Regression (Stats-Based)

For multiple target variables, follow the same three-step pattern:

```julia
# Training data: each column is a different target
train_labels = [1.0 10.0 100.0;
                5.0 20.0 200.0;
                3.0 15.0 150.0;
                8.0 25.0 250.0]

# Step 1: Compute stats from training data ONLY
stats = compute_normalization_stats(train_labels; mode=:columnwise, method=:zscore)

# Step 2: Apply SAME stats to all splits
train_normalized = apply_normalization(train_labels, stats)
val_normalized = apply_normalization(val_labels, stats)    # Same stats
test_normalized = apply_normalization(test_labels, stats)  # Same stats

# Step 3: Denormalize predictions using SAME stats
predictions_original = denormalize_labels(predictions_normalized, stats)
```

## Normalization Methods (Stats-Based)

### Min-Max Normalization

Scales values to a specified range (default: `[-1, 1]`):

```julia
# Step 1: Compute stats from training data
stats = compute_normalization_stats(train_labels; method=:minmax, range=(-1, 1))

# Step 2: Apply to all splits
train_norm = apply_normalization(train_labels, stats)
test_norm = apply_normalization(test_labels, stats)

# Step 3: Denormalize predictions
predictions_original = denormalize_labels(predictions_normalized, stats)
```

### Z-Score Normalization

Standardizes values to have zero mean and unit variance:

```julia
# Step 1: Compute stats from training data
stats = compute_normalization_stats(train_labels; method=:zscore)

# Step 2: Apply to all splits  
train_norm = apply_normalization(train_labels, stats)
test_norm = apply_normalization(test_labels, stats)

# Step 3: Denormalize predictions
predictions_original = denormalize_labels(predictions_normalized, stats)
```

## Outlier Handling (Stats-Based)

### Quantile-Based Clipping

Configure clipping when computing stats from training data:

```julia
# Step 1: Compute stats with outlier clipping (training data only)
stats = compute_normalization_stats(train_labels; 
    method=:zscore, 
    clip_quantiles=(0.01, 0.99)  # Default: clip to 1st-99th percentiles
)

# Step 2: Apply to all splits (same clipping applied)
train_norm = apply_normalization(train_labels, stats)
test_norm = apply_normalization(test_labels, stats)

# Step 3: Denormalize predictions
predictions_original = denormalize_labels(predictions_normalized, stats)
```

### Why Clip Outliers?

Outliers can severely distort normalization, especially min-max scaling:

```julia
train_labels = [1, 2, 3, 4, 5, 1000]  # 1000 is an outlier

# Step 1: Compute stats with clipping
stats_with_clip = compute_normalization_stats(train_labels; clip_quantiles=(0.1, 0.9))

# Step 2: Apply to test data
test_labels = [1.5, 2.5, 3.5]
test_norm = apply_normalization(test_labels, stats_with_clip)
# Result: better distribution because outlier was clipped during stats computation
```

## Handling Missing Data (Stats-Based)

NaN values are handled gracefully in the stats-based workflow:

```julia
# Training data with missing values
train_with_nan = [1.0, 2.0, NaN, 4.0, 5.0, 100.0]

# Step 1: Compute stats from valid training data only
stats = compute_normalization_stats(train_with_nan)  # Uses [1.0, 2.0, 4.0, 5.0, 100.0]

# Step 2: Apply to all splits (NaN positions preserved)
train_norm = apply_normalization(train_with_nan, stats)  # NaNs preserved
test_norm = apply_normalization(test_with_nan, stats)    # NaNs preserved, same stats

# Step 3: Denormalize predictions
predictions_original = denormalize_labels(predictions_normalized, stats)
```

## Complete Machine Learning Workflow

Here's the complete pattern for any ML project:

```julia
# Step 1: Compute normalization statistics on training data ONLY
train_labels = [1.0, 2.0, 3.0, 4.0, 5.0, 100.0]  # With outlier
stats = compute_normalization_stats(train_labels; method=:zscore, clip_quantiles=(0.01, 0.99))

# Step 2: Apply SAME stats to all data splits
train_normalized = apply_normalization(train_labels, stats)
val_normalized = apply_normalization(val_labels, stats)    # Same stats
test_normalized = apply_normalization(test_labels, stats)  # Same stats

# Step 3: Train model on normalized data
# model = train_model(X_train, train_normalized)

# Step 4: Make predictions and denormalize using SAME stats
predictions_normalized = model(X_test)
predictions_original = denormalize_labels(predictions_normalized, stats)
```

## Best Practices (Stats-Based Workflow)

### When to Use Each Method

- **Min-max normalization**: When you know the expected range of your data or want bounded outputs
- **Z-score normalization**: When your data is approximately normally distributed
- **Global mode**: When all targets should be on the same scale (e.g., related measurements)
- **Column-wise mode**: When targets represent different quantities with different scales

### The Golden Rule: Always Use Stats-Based Workflow

```julia
# ✅ CORRECT: Stats-based workflow (prevents data leakage)
stats = compute_normalization_stats(train_labels)  # Training data only
train_norm = apply_normalization(train_labels, stats)
test_norm = apply_normalization(test_labels, stats)  # Same stats
predictions_original = denormalize_labels(predictions_normalized, stats)

# ❌ WRONG: Direct normalization (causes data leakage)
# train_norm = normalize_labels(train_labels)
# test_norm = normalize_labels(test_labels)  # Different stats!
```

### Cross-Validation with Consistent Stats

```julia
# For each CV fold, compute stats on training portion only
for fold in 1:5
    train_idx, val_idx = get_cv_indices(fold)
    
    # Step 1: Compute stats on training fold only
    fold_stats = compute_normalization_stats(y_train[train_idx])
    
    # Step 2: Apply to both training and validation portions
    y_train_norm = apply_normalization(y_train[train_idx], fold_stats)
    y_val_norm = apply_normalization(y_train[val_idx], fold_stats)  # Same stats!
    
    # Step 3: Train and validate model
    model = train_model(X_train[train_idx], y_train_norm)
    val_pred_norm = model(X_train[val_idx])
    val_pred_original = denormalize_labels(val_pred_norm, fold_stats)
end
```

### Handling Extreme Outliers

Configure clipping when computing stats from training data:

```julia
# For data with extreme outliers (e.g., financial data)
stats = compute_normalization_stats(train_labels; clip_quantiles=(0.1, 0.9))

# For very clean data, you might skip clipping
stats = compute_normalization_stats(train_labels; clip_quantiles=nothing)

# Apply same clipping to all splits
train_norm = apply_normalization(train_labels, stats)
test_norm = apply_normalization(test_labels, stats)
```
