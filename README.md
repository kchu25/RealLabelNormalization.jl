# RealLabelNormalization.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://kchu25.github.io/RealLabelNormalization.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://kchu25.github.io/RealLabelNormalization.jl/dev/)
[![Build Status](https://github.com/kchu25/RealLabelNormalization.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/kchu25/RealLabelNormalization.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/kchu25/RealLabelNormalization.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/kchu25/RealLabelNormalization.jl)


Avoiding data leakage (computing stats on the training set only), clipping outliers, and handling NaNs isn’t hard — but it’s tedious, especially when you end up reinventing the same workflow for every dataset (often with tools like [MLUtils.jl](https://github.com/JuliaML/MLUtils.jl)). This package provides robust normalization of real-valued labels for regression tasks with built-in outlier handling and NaN support, ensuring consistent, leak-free preprocessing across train/validation/test splits.


## A quick way to understand what this package does:

- Your dataset `(X, Y)` consists of `n` (**data point, label**) pairs. This package only operates on the **labels** (`Y`). 
- Two cases for `Y`:
    - **Scalar-valued labels:** `Y` is a vector.
    - **Matrix-valued labels:** `Y` is a matrix, with the second dimension corresponding to data points (`size(Y,2) = n`).

This package normalizes the values in `Y`. Once normalized, you can feed them directly into a Flux DataLoader, e.g.

```julia
using Flux
dataloader = Flux.DataLoader((data=X, label=Y))
```

## Features

- **Multiple normalization methods**: Min-max and Z-score normalization
- **Flexible normalization modes**: Global or column-wise normalization for multi-target regression
- **Robust outlier handling**: Configurable quantile-based clipping to handle extreme values
- **NaN handling**: Preserves NaN values while computing statistics on valid data
- **Consistent train/test normalization**: Save statistics from training data and apply to test data
- **High performance**: Optimized for large datasets with minimal memory overhead
- **Multi-target support**: Handle single or multiple regression targets seamlessly

## Installation

```julia
using Pkg
Pkg.add("RealLabelNormalization")
```

## Quick Start

```julia
using RealLabelNormalization

# Basic min-max normalization to [-1, 1] with outlier clipping
labels = [1.0, 5.0, 3.0, 8.0, 2.0, 100.0]  # 100.0 is an outlier
normalized = normalize_labels(labels)
# Output: [-1.0, -0.2, -0.6, 0.4, -0.8, 1.0]

# Z-score normalization
normalized = normalize_labels(labels; method=:zscore)

# Multi-target normalization (matrix input)
labels_matrix = [1.0 10.0; 5.0 20.0; 3.0 15.0; 8.0 25.0]
normalized = normalize_labels(labels_matrix; mode=:columnwise)
```

## Key Use Cases

### Machine Learning Pipeline

```julia
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

### Handling Different Data Types

```julia
# Works with missing data (NaN)
labels_with_nan = [1.0, 2.0, NaN, 4.0, 5.0]
normalized = normalize_labels(labels_with_nan)  # NaN preserved

# Multi-target regression with different scales
weather_data = [
    20.5  65.0  1013.2;  # [temperature, humidity, pressure]
    22.1  58.3  1015.8;
    18.9  72.1  1008.9;
    25.4  45.2  1020.1
]

# Normalize each target independently
normalized = normalize_labels(weather_data; mode=:columnwise)
```

## Documentation

- [**User Guide**](https://kchu25.github.io/RealLabelNormalization.jl/dev/guide/): Detailed usage patterns and best practices
- [**Examples**](https://kchu25.github.io/RealLabelNormalization.jl/dev/examples/): Comprehensive examples for various scenarios
- [**API Reference**](https://kchu25.github.io/RealLabelNormalization.jl/dev/api/): Complete function documentation

## Why RealLabelNormalization.jl?

### Problem: Inconsistent Normalization in ML Pipelines

```julia
# Common mistake: computing statistics on entire dataset
all_data = [train_data; test_data]
normalized_all = (all_data .- mean(all_data)) ./ std(all_data)
train_norm = normalized_all[1:length(train_data)]
test_norm = normalized_all[length(train_data)+1:end]
# This leads to data leakage and poor generalization!
```

### Solution: Proper Train/Test Separation

```julia
# Correct approach with RealLabelNormalization.jl
stats = compute_normalization_stats(train_data)  # Only training data
train_normalized = apply_normalization(train_data, stats)
test_normalized = apply_normalization(test_data, stats)  # Same stats applied
# Ensures no data leakage and proper generalization
```

## Normalization Methods

| Method | Description | Use Case | Output Range |
|--------|-------------|----------|--------------|
| **Min-Max** | Scales to specified range | Known data bounds, neural networks | User-defined (default: [-1,1]) |
| **Z-Score** | Zero mean, unit variance | Normally distributed data, linear models | Approximately [-3,3] |

## Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| **Global** | Single normalization across all features | Related measurements on same scale |
| **Column-wise** | Independent normalization per feature | Different units/scales per target |

## Advanced Features

### Outlier Clipping

```julia
# Automatic outlier clipping (default: 1st-99th percentiles)
labels = [1, 2, 3, 4, 5, 1000]  # 1000 is extreme outlier
normalized = normalize_labels(labels)  # Outlier automatically clipped

# Custom clipping percentiles
normalized = normalize_labels(labels; clip_quantiles=(0.05, 0.95))

# No clipping
normalized = normalize_labels(labels; clip_quantiles=nothing)
```

### Custom Ranges

```julia
# Scale to [0, 1] instead of [-1, 1]
normalized = normalize_labels(labels; range=(0, 1))

# Scale to custom range
normalized = normalize_labels(labels; range=(-2, 2))
```

## Performance

RealLabelNormalization.jl is optimized for performance:

- **Efficient algorithms**: Vectorized operations with minimal allocations
- **Streaming support**: Process large datasets that don't fit in memory
- **Multi-threading**: Automatic parallelization for large matrices
- **Memory efficient**: In-place operations when possible

## Related Packages

- [StatsBase.jl](https://github.com/JuliaStats/StatsBase.jl): General statistical functions
- [MLJ.jl](https://github.com/alan-turing-institute/MLJ.jl): Machine learning framework with preprocessing
- [MLUtils.jl](https://github.com/JuliaML/MLUtils.jl): Machine learning utilities

### Development Setup

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

## Acknowledgments

- Inspired by scikit-learn's preprocessing module
- Built with the Julia programming language ecosystem
- Thanks to the Julia community for feedback and contributions
