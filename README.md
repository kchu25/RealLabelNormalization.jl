# RealLabelNormalization.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://kchu25.github.io/RealLabelNormalization.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://kchu25.github.io/RealLabelNormalization.jl/dev/)
[![Build Status](https://github.com/kchu25/RealLabelNormalization.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/kchu25/RealLabelNormalization.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/kchu25/RealLabelNormalization.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/kchu25/RealLabelNormalization.jl)

**Leak-free label normalization for machine learning in Julia.**

Provides robust workflows for normalizing real-valued regression labels while preventing data leakage, handling outliers, and preserving NaNs.

## ⚠️ CRITICAL: Always Use the Stats-Based Workflow

**NEVER** use `normalize_labels()` directly on your full dataset. This causes data leakage! Instead, follow this pattern:

1. **Compute stats from training data ONLY**
2. **Apply the same stats to validation/test data**  
3. **Denormalize predictions using the same stats**

## Why This Package?

```julia
# ❌ Common mistake: data leakage
all_data = [train_data; test_data]
normalized = (all_data .- mean(all_data)) ./ std(all_data)

# ✅ Correct approach: compute stats on training data only
stats = compute_normalization_stats(train_data)
train_norm = apply_normalization(train_data, stats)
test_norm = apply_normalization(test_data, stats)  # Same stats!
```

## Installation

```julia
using Pkg
Pkg.add("RealLabelNormalization")
```

## Quick Start (Stats-Based Workflow)


```julia
using RealLabelNormalization

# Training labels with outlier
train_labels = [1.5, 2.3, 4.1, 3.7, 5.2, 100.0]
test_labels = [2.1, 3.9, 4.5]

# Step 1: Compute stats from TRAINING DATA ONLY
stats = compute_normalization_stats(train_labels; method=:zscore, clip_quantiles=(0.01, 0.99))

# Step 2: Apply SAME stats to training data
train_normalized = apply_normalization(train_labels, stats)

# Step 3: Apply SAME STATS to test data (prevents data leakage!)
test_normalized = apply_normalization(test_labels, stats)

# Step 4: Train model on normalized data
# model = train_your_model(X_train, train_normalized)

# Step 5: Denormalize predictions back to original scale using SAME stats
predictions_normalized = model(X_test)  # Model outputs normalized predictions
predictions_original = denormalize_labels(predictions_normalized, stats)
```

**The Golden Rule:**
1. **Compute stats from training data ONLY**
2. **Apply the same stats to validation/test data**  
3. **Denormalize predictions using the same stats**

## Methods and Modes

| Method | Syntax | Description | Best For |
|--------|--------|-------------|----------|
| **Min-Max** | `method=:minmax, range=(-1,1)` | Scale to specified range | Neural networks, bounded outputs |
| **Z-Score** | `method=:zscore` | Zero mean, unit variance | Linear models, normally distributed data |

| Mode | Syntax | Description | Use Case |
|------|--------|-------------|----------|
| **Global** | `mode=:global` | Single stats across all values | Related measurements, same scale |
| **Column-wise** | `mode=:columnwise` | Per-column normalization | Multi-target with different units |
| **Row-wise** | `mode=:rowwise` | Per-row normalization | Time series, per-sample scaling |

## Usage Examples (Stats-Based Workflow)

### Single-Target Regression
```julia
train_labels = [1.0, 5.0, 3.0, 8.0, 2.0, 100.0]
test_labels = [1.2, 4.8, 6.5]

# Step 1: Compute stats from training data ONLY
stats_minmax = compute_normalization_stats(train_labels; method=:minmax, range=(-1, 1))
stats_zscore = compute_normalization_stats(train_labels; method=:zscore)

# Step 2: Apply SAME stats to both training and test
train_norm = apply_normalization(train_labels, stats_minmax)
test_norm = apply_normalization(test_labels, stats_minmax)  # Same stats!

# Step 3: Denormalize predictions using SAME stats
predictions_norm = model(X_test)
predictions_original = denormalize_labels(predictions_norm, stats_minmax)
```

### Multi-Target Regression
```julia
# Weather data: [temperature, humidity, pressure]
weather_train = [20.5 65.0 1013.2; 22.1 58.3 1015.8; 18.9 72.1 1008.9]
weather_val = [19.8 70.2 1011.5; 23.1 55.0 1018.3]
weather_test = [21.3 62.1 1014.7; 17.2 75.8 1009.2]

# Step 1: Compute stats ONCE from training data
stats = compute_normalization_stats(weather_train; mode=:columnwise, method=:zscore)

# Step 2: Apply SAME stats to all splits
train_norm = apply_normalization(weather_train, stats)
val_norm = apply_normalization(weather_val, stats)      # Same stats
test_norm = apply_normalization(weather_test, stats)    # Same stats

# Step 3: Denormalize predictions using SAME stats
val_pred_original = denormalize_labels(model(X_val), stats)
test_pred_original = denormalize_labels(model(X_test), stats)
```

### Cross-Validation with Consistent Stats
```julia
# For each CV fold, compute stats on training portion only
for fold in 1:5
    train_idx, val_idx = get_cv_indices(fold)
    
    # Step 1: Stats computed on training fold only
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

### Handling Missing Data
```julia
train_with_nan = [1.0, 2.0, NaN, 4.0, 5.0, 100.0]
test_with_nan = [1.5, NaN, 3.2]

# Step 1: Stats computed only on valid (non-NaN) training values
stats = compute_normalization_stats(train_with_nan)  # Uses [1.0, 2.0, 4.0, 5.0, 100.0]

# Step 2: NaN positions preserved in both training and test
train_norm = apply_normalization(train_with_nan, stats)  # NaNs preserved
test_norm = apply_normalization(test_with_nan, stats)    # NaNs preserved, same stats

# Step 3: Denormalize predictions using SAME stats
predictions_original = denormalize_labels(predictions_normalized, stats)
```

## Key Features

- **Prevents data leakage**: Stats computed from training data, applied consistently to val/test
- **Outlier handling**: Configurable quantile-based clipping (default: 1%-99%)
- **NaN preservation**: Statistics skip NaNs, output preserves NaN positions
- **Multi-target support**: Handle matrices with global/column/row-wise modes  
- **Complete workflow**: Compute once, apply everywhere, denormalize predictions
- **Simple API**: Three-step pattern for bulletproof normalization

## Integration with ML Frameworks

### Flux.jl - Complete Workflow
```julia
using Flux

# Step 1: Compute normalization stats from training labels ONLY
train_stats = compute_normalization_stats(Y_train; method=:zscore)

# Step 2: Normalize all data splits using the SAME stats
Y_train_norm = apply_normalization(Y_train, train_stats)
Y_val_norm = apply_normalization(Y_val, train_stats)    # Same stats
Y_test_norm = apply_normalization(Y_test, train_stats)  # Same stats

# Step 3: Create DataLoaders with normalized labels
train_loader = Flux.DataLoader((X_train, Y_train_norm), batchsize=32, shuffle=true)
val_loader = Flux.DataLoader((X_val, Y_val_norm), batchsize=32)

# Step 4: Train model on normalized data
model = Chain(Dense(input_dim => 64, relu), Dense(64 => output_dim))
# ... training loop with normalized labels ...

# Step 5: Make predictions and denormalize using SAME stats
test_pred_norm = model(X_test)
test_pred_original = denormalize_labels(test_pred_norm, train_stats)

# Validation predictions also use the same stats
val_pred_norm = model(X_val)  
val_pred_original = denormalize_labels(val_pred_norm, train_stats)
```

### MLJ.jl Pattern
```julia
# In your MLJ machine/model pipeline
function preprocess_labels(Y_train, Y_val, Y_test)
    # Compute stats only from training data
    stats = compute_normalization_stats(Y_train; method=:zscore, mode=:columnwise)
    
    # Apply consistently across all splits
    return (
        train = apply_normalization(Y_train, stats),
        val = apply_normalization(Y_val, stats),
        test = apply_normalization(Y_test, stats),
        stats = stats  # Store for later denormalization
    )
end

normalized_data = preprocess_labels(Y_train, Y_val, Y_test)
# Train with normalized_data.train, validate with normalized_data.val
# Denormalize predictions: denormalize_labels(predictions, normalized_data.stats)
```

## API Reference

**Core Functions (The Stats-Based Workflow):**
- `compute_normalization_stats(train_data; method, mode, clip_quantiles, range)` - Compute stats from training data ONLY
- `apply_normalization(data, stats)` - Apply precomputed stats to any dataset (train/val/test)
- `denormalize_labels(normalized_predictions, stats)` - Convert predictions back using same stats

**Convenience (Training Data Only):**
- `normalize_labels(train_data; kwargs...)` - One-step normalization for training data (use stats-based workflow for test data)

**Parameters:**
- `method`: `:minmax` (default) or `:zscore`
- `mode`: `:global` (default), `:columnwise`, or `:rowwise`  
- `clip_quantiles`: `(0.01, 0.99)` (default) or `nothing` to disable
- `range`: `(-1, 1)` (default) for min-max scaling

## Documentation

- [**User Guide**](https://kchu25.github.io/RealLabelNormalization.jl/dev/guide/) - Detailed workflows and best practices
- [**API Reference**](https://kchu25.github.io/RealLabelNormalization.jl/dev/api/) - Complete function documentation
- [**Examples**](https://kchu25.github.io/RealLabelNormalization.jl/dev/examples/) - Real-world use cases

## Citation

```bibtex
@software{RealLabelNormalization_jl,
  author = {Shane Kuei-Hsien Chu},
  title = {RealLabelNormalization.jl: Robust normalization for real-valued regression labels},
  url = {https://github.com/kchu25/RealLabelNormalization.jl},
  version = {1.0.0},
  year = {2025}
}
```

## License

MIT License - see [LICENSE](LICENSE) file for details.