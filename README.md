# RealLabelNormalization.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://kchu25.github.io/RealLabelNormalization.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://kchu25.github.io/RealLabelNormalization.jl/dev/)
[![Build Status](https://github.com/kchu25/RealLabelNormalization.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/kchu25/RealLabelNormalization.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/kchu25/RealLabelNormalization.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/kchu25/RealLabelNormalization.jl)

Common subroutines for normalizing real-valued labels prior to ML training.

## Why This Package?

Avoiding data leakage, clipping outliers, and handling NaNs isn't hard — but it's tedious, especially when you end up reinventing the same workflow for every dataset (often with tools like [MLUtils.jl](https://github.com/JuliaML/MLUtils.jl)). 

### The Problem
```julia
# Common mistake: computing statistics on entire dataset
all_data = [train_data; test_data]
normalized_all = (all_data .- mean(all_data)) ./ std(all_data)
# This leads to data leakage and poor generalization!
```

### The Solution
```julia
# Correct approach with RealLabelNormalization.jl
stats = compute_normalization_stats(train_data)  # Only training data
train_normalized = apply_normalization(train_data, stats)
test_normalized = apply_normalization(test_data, stats)  # Same stats
```

## Installation

```julia
using Pkg
Pkg.add("RealLabelNormalization")
```

## Quick Start

```julia
using RealLabelNormalization

# Step 1: Compute normalization statistics from training data ONLY
train_labels = [1.5, 2.3, 4.1, 3.7, 5.2, 100.0]  # Contains outlier
stats = compute_normalization_stats(train_labels)

# Step 2: Apply to training data
train_normalized = apply_normalization(train_labels, stats)

# Step 3: Apply same statistics to test data (crucial for consistency!)
test_labels = [2.1, 3.9, 4.5]
test_normalized = apply_normalization(test_labels, stats)

# Step 4: After model prediction, denormalize back to original scale
predictions_normalized = [-0.1, 0.3, 0.7]
predictions_original = denormalize_labels(predictions_normalized, stats)
```

## Key Features

- **Leak-free workflows**: Compute stats on training data, apply to validation/test
- **Automatic outlier handling**: Configurable quantile-based clipping (default: 1st-99th percentiles)
- **NaN preservation**: Statistics computed on valid data only, NaNs preserved in output
- **Multi-target support**: Handle scalar- or vector-valued labels with global/column-wise modes
- **Two normalization methods**: Min-max (configurable range) and Z-score

## Usage Patterns

### Convenience Functions

```julia
using RealLabelNormalization

# Basic min-max normalization to [-1, 1] with outlier clipping
labels = [1.0, 5.0, 3.0, 8.0, 2.0, 100.0]  # 100.0 is an outlier
normalized = normalize_labels(labels)

# Z-score normalization
normalized = normalize_labels(labels; method=:zscore)

# Multi-target normalization (matrix input)
labels_matrix = [1.0 10.0; 5.0 20.0; 3.0 15.0; 8.0 25.0]
normalized = normalize_labels(labels_matrix; mode=:columnwise)
```

### Machine Learning

```julia
# Step 1: Compute normalization statistics from training data ONLY
train_labels = [1.5, 2.3, 4.1, 3.7, 5.2, 100.0]  # Contains outlier
stats = compute_normalization_stats(train_labels)

# Step 2: Apply to training data
train_normalized = apply_normalization(train_labels, stats)

# Step 3: Apply same statistics to test data (crucial for consistency)
test_labels = [2.1, 3.9, 4.5]
test_normalized = apply_normalization(test_labels, stats)

# Step 4: After model prediction, denormalize back to original scale
predictions_normalized = [-0.1, 0.3, 0.7]
predictions_original = denormalize_labels(predictions_normalized, stats)
```

### Vector-valued Regression

```julia
# Matrix labels where each column is a different target
weather_train = [
    20.5  65.0  1013.2;  # [temperature, humidity, pressure]
    22.1  58.3  1015.8;
    18.9  72.1  1008.9;
    25.4  45.2  1020.1
]

weather_test = [
    19.8  70.2  1011.5;
    23.1  55.0  1018.3
]

# Compute stats from training data with custom options
stats = compute_normalization_stats(weather_train; 
    method=:zscore, 
    mode=:columnwise,  # Each target independently (recommended for different units)
    clip_quantiles=(0.05, 0.95)  # More aggressive outlier clipping
)

# Apply to training and test data
train_normalized = apply_normalization(weather_train, stats)
test_normalized = apply_normalization(weather_test, stats)
```

### Row-wise Normalization

```julia
# Each row is normalized independently (useful for time series, per-sample normalization, etc.)
mat = [1.0 2.0 3.0;
       10.0 20.0 30.0;
       -1.0 0.0 1.0]

# Min-max row-wise normalization (each row mapped to [-1, 1])
stats_row = compute_normalization_stats(mat; mode=:rowwise)
normalized_row = apply_normalization(mat, stats_row)

# Z-score row-wise normalization (each row mean≈0, std≈1)
stats_row_z = compute_normalization_stats(mat; mode=:rowwise, method=:zscore)
normalized_row_z = apply_normalization(mat, stats_row_z)

# NaN handling: NaNs are preserved, valid values normalized
mat_nan = [1.0 NaN 3.0; 10.0 20.0 NaN]
stats_nan = compute_normalization_stats(mat_nan; mode=:rowwise)
normalized_nan = apply_normalization(mat_nan, stats_nan)
```

### Other Options

```julia
# Custom clipping percentiles for extreme outliers
normalized = normalize_labels(labels; clip_quantiles=(0.05, 0.95))

# No clipping (not recommended for real data)
normalized = normalize_labels(labels; clip_quantiles=nothing)

# Custom range for min-max normalization
normalized = normalize_labels(labels; range=(0, 1))  # Scale to [0,1]

# Works seamlessly with NaN values
labels_with_nan = [1.0, 2.0, NaN, 4.0, 5.0]
normalized = normalize_labels(labels_with_nan)  # NaN preserved
```

## Methods and Modes

| Method | Description | Use Case | Output Range |
|--------|-------------|----------|--------------|
| **Min-Max** | Scales to specified range | Known data bounds, neural networks | User-defined (default: [-1,1]) |
| **Z-Score** | Zero mean, unit variance | Normally distributed data, linear models | Approximately [-3,3] |

| Mode | Description | Use Case |
|------|-------------|----------|
| **Global** | Single normalization across all features | Related measurements on same scale |
| **Column-wise** | Independent normalization per feature | Different units/scales per target |
| **Row-wise** | Independent normalization per sample/row | Time series, per-sample normalization |

## Integration with ML Workflows

### With Flux.jl

```julia
using Flux

# Normalize labels
stats = compute_normalization_stats(Y_train)
Y_train_norm = apply_normalization(Y_train, stats)

# Create DataLoader
dataloader = Flux.DataLoader((data=X_train, label=Y_train_norm), batchsize=32)

# After training, denormalize predictions
predictions_original = denormalize_labels(model_output, stats)
```

### Label Format Support

This package works with your dataset tuple `(X, Y)` and focuses exclusively on transforming the labels `Y`:

- **Vector labels**: `Y` is a vector for single-target regression
- **Matrix labels**: `Y` is a matrix where `size(Y, 2) = n` (columns = data points) for multi-target regression

## Documentation

- [**User Guide**](https://kchu25.github.io/RealLabelNormalization.jl/dev/guide/): Detailed usage patterns and best practices
- [**Examples**](https://kchu25.github.io/RealLabelNormalization.jl/dev/examples/): Comprehensive examples for various scenarios
- [**API Reference**](https://kchu25.github.io/RealLabelNormalization.jl/dev/api/): Complete function documentation

## Related Packages

- [StatsBase.jl](https://github.com/JuliaStats/StatsBase.jl): General statistical functions
- [MLJ.jl](https://github.com/alan-turing-institute/MLJ.jl): Machine learning framework with preprocessing
- [MLUtils.jl](https://github.com/JuliaML/MLUtils.jl): Machine learning utilities

## Development

### Setup

```julia
# Clone the repository
git clone https://github.com/kchu25/RealLabelNormalization.jl.git
cd RealLabelNormalization.jl

# Activate the environment and install dependencies
julia --project=. -e "using Pkg; Pkg.instantiate()"

# Run tests
julia --project=. -e "using Pkg; Pkg.test()"

# Build documentation
julia --project=docs docs/make.jl
```

## Citation

If you use RealLabelNormalization.jl in your research, please cite:

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

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


