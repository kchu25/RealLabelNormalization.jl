```@meta
CurrentModule = RealLabelNormalization
```

# RealLabelNormalization.jl

A Julia package for robust normalization of real-valued labels, commonly used in regression tasks. This package provides various normalization methods with built-in outlier handling and NaN support.

## Features

- **Multiple normalization methods**: Min-max and Z-score normalization
- **Flexible normalization modes**: Global or column-wise normalization
- **Robust outlier handling**: Configurable quantile-based clipping
- **NaN handling**: Preserves NaN values while computing statistics on valid data
- **Consistent train/test normalization**: Save statistics from training data and apply to test data


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

## Installation

```julia
using Pkg
Pkg.add("RealLabelNormalization")
```

## API Reference

```@index
```

```@autodocs
Modules = [RealLabelNormalization]
```
