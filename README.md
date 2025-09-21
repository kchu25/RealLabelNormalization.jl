# RealLabelNormalization.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://kchu25.github.io/RealLabelNormalization.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://kchu25.github.io/RealLabelNormalization.jl/dev/)
[![Build Status](https://github.com/kchu25/RealLabelNormalization.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/kchu25/RealLabelNormalization.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/kchu25/RealLabelNormalization.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/kchu25/RealLabelNormalization.jl)

**Leak-free label normalization for machine learning in Julia.**

Provides robust workflows for normalizing real-valued regression labels while preventing data leakage, handling outliers, and preserving NaNs.

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

## Quick Start

```julia
using RealLabelNormalization

# Training labels with outlier
train_labels = [1.5, 2.3, 4.1, 3.7, 5.2, 100.0]

# Step 1: Compute stats from training data only
stats = compute_normalization_stats(train_labels; method=:zscore, clip_quantiles=(0.01, 0.99))

# Step 2: Normalize training data
train_normalized = apply_normalization(train_labels, stats)

# Step 3: Normalize test data with same stats
test_labels = [2.1, 3.9, 4.5]
test_normalized = apply_normalization(test_labels, stats)

# Step 4: Denormalize predictions back to original scale
predictions = denormalize_labels(model_output, stats)
```

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

## Usage Examples

### Single-Target Regression
```julia
labels = [1.0, 5.0, 3.0, 8.0, 2.0, 100.0]

# Min-max to [-1, 1] with outlier clipping (default)
normalize_labels(labels)

# Z-score normalization
normalize_labels(labels; method=:zscore)

# Custom range and clipping
normalize_labels(labels; range=(0, 1), clip_quantiles=(0.05, 0.95))
```

### Multi-Target Regression
```julia
# Weather data: [temperature, humidity, pressure]
weather_train = [20.5 65.0 1013.2; 22.1 58.3 1015.8; 18.9 72.1 1008.9]

# Column-wise: each target normalized independently
stats = compute_normalization_stats(weather_train; mode=:columnwise)
train_norm = apply_normalization(weather_train, stats)

# Apply same stats to test data
weather_test = [19.8 70.2 1011.5; 23.1 55.0 1018.3]
test_norm = apply_normalization(weather_test, stats)
```

### Handling Missing Data
```julia
# NaNs are preserved, stats computed on valid data only
labels_with_nan = [1.0, 2.0, NaN, 4.0, 5.0]
normalized = normalize_labels(labels_with_nan)  # NaN positions preserved
```

## Key Features

- **🛡️ Prevents data leakage**: Stats computed on training data only
- **📊 Outlier handling**: Configurable quantile-based clipping (default: 1%-99%)
- **🔢 NaN preservation**: Statistics skip NaNs, output preserves NaN positions
- **🎯 Multi-target support**: Handle matrices with global/column/row-wise modes
- **⚡ Simple API**: Both step-by-step and convenience functions

## Integration with ML Frameworks

### Flux.jl
```julia
using Flux

# Normalize labels for training
stats = compute_normalization_stats(Y_train)
Y_train_norm = apply_normalization(Y_train, stats)

# Create DataLoader
dataloader = Flux.DataLoader((data=X_train, label=Y_train_norm), batchsize=32)

# Denormalize predictions
predictions_original = denormalize_labels(model(X_test), stats)
```

## API Reference

**Core Functions:**
- `compute_normalization_stats(data; method, mode, clip_quantiles, range)` - Compute normalization parameters
- `apply_normalization(data, stats)` - Apply normalization using stored stats  
- `denormalize_labels(normalized_data, stats)` - Convert back to original scale

**Convenience:**
- `normalize_labels(data; kwargs...)` - One-step normalization (training data only)

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