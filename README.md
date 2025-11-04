# RealLabelNormalization.jl

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://kchu25.github.io/RealLabelNormalization.jl/dev/)
[![Build Status](https://github.com/kchu25/RealLabelNormalization.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/kchu25/RealLabelNormalization.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/kchu25/RealLabelNormalization.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/kchu25/RealLabelNormalization.jl)

Leak-free label normalization for machine learning in Julia.

## Why This Package?

Prevents data leakage by computing statistics on training data only:

```julia
# Wrong: statistics leak from test set
all_data = [train_data; test_data]
normalized = (all_data .- mean(all_data)) ./ std(all_data)

# Correct: stats from training only
stats = compute_normalization_stats(train_data)
train_norm = apply_normalization(train_data, stats)
test_norm = apply_normalization(test_data, stats)  # Same stats
```

## Installation

```julia
using Pkg
Pkg.add("RealLabelNormalization")
```

## Quick Start

```julia
using RealLabelNormalization

# 1. Compute stats from training data ONLY
stats = compute_normalization_stats(train_labels; method=:zscore)

# 2. Apply same stats to all splits
train_norm = apply_normalization(train_labels, stats)
test_norm = apply_normalization(test_labels, stats)

# 3. Train model on normalized data
# model = train(X_train, train_norm)

# 4. Denormalize predictions using same stats
predictions_original = denormalize_labels(model(X_test), stats)
```

## Normalization Methods

| Method | Description | Use Case |
|--------|-------------|----------|
| `:minmax` | Scale to range | Bounded output (e.g., [0,1], [-1,1]) |
| `:zscore` | Zero mean, unit variance | Standard scaling, unbounded |
| `:zscore_minmax` | Z-score then min-max | Outlier handling + bounded range |
| `:log` | Log transform | Skewed distributions |
| `:log_minmax` | Log then min-max | Skewed data + bounded range |

## Modes (Matrices Only)

| Mode | Description |
|------|-------------|
| `:rowwise` (default) | Normalize each row independently |
| `:columnwise` | Normalize each column independently |
| `:global` | Single normalization across all values |

Note: Vectors are always normalized globally.

## Examples

### Single-Target Regression

```julia
train = [1.0, 5.0, 3.0, 8.0, 100.0]  # Outlier present
test = [1.2, 4.8, 6.5]

# Min-max to [0,1]
stats = compute_normalization_stats(train; method=:minmax, range=(0,1))
train_norm = apply_normalization(train, stats)
test_norm = apply_normalization(test, stats)

# Z-score with outlier clipping
stats = compute_normalization_stats(train; method=:zscore, clip_quantiles=(0.01, 0.99))
train_norm = apply_normalization(train, stats)

# Z-score + min-max: best of both worlds
stats = compute_normalization_stats(train; method=:zscore_minmax, range=(0,1))
train_norm = apply_normalization(train, stats)
```

### Multi-Target Regression

```julia
# Weather data: [temperature, humidity, pressure]
weather_train = [20.5 65.0 1013.2; 22.1 58.3 1015.8; 18.9 72.1 1008.9]
weather_test = [21.3 62.1 1014.7; 17.2 75.8 1009.2]

# Normalize each column (feature) independently
stats = compute_normalization_stats(weather_train; mode=:columnwise, method=:zscore)
train_norm = apply_normalization(weather_train, stats)
test_norm = apply_normalization(weather_test, stats)
```

### Handling NaNs

```julia
train = [1.0, 2.0, NaN, 4.0, 5.0]
test = [1.5, NaN, 3.2]

# Stats computed on non-NaN values only
stats = compute_normalization_stats(train)
train_norm = apply_normalization(train, stats)  # NaNs preserved
test_norm = apply_normalization(test, stats)    # NaNs preserved
```

### Log Normalization for Skewed Data

```julia
# Income data (highly skewed)
income_train = [20000.0, 35000.0, 50000.0, 65000.0, 1000000.0]
income_test = [25000.0, 45000.0]

stats = compute_normalization_stats(income_train; method=:log)
train_log = apply_normalization(income_train, stats)
test_log = apply_normalization(income_test, stats)

# Handles negative values automatically
data_with_neg = [-10.0, -5.0, 0.0, 5.0, 100.0]
stats = compute_normalization_stats(data_with_neg; method=:log)
```

### Integration with Flux.jl

```julia
using Flux

# Normalize labels
stats = compute_normalization_stats(Y_train; method=:zscore)
Y_train_norm = apply_normalization(Y_train, stats)
Y_val_norm = apply_normalization(Y_val, stats)

# Create DataLoaders
train_loader = Flux.DataLoader((X_train, Y_train_norm), batchsize=32, shuffle=true)

# Train model
model = Chain(Dense(input_dim => 64, relu), Dense(64 => output_dim))
# ... training loop ...

# Denormalize predictions
predictions_norm = model(X_test)
predictions_original = denormalize_labels(predictions_norm, stats)
```

## API Reference

**Core Functions:**
- `compute_normalization_stats(train_data; kwargs...)` - Compute stats from training data
- `apply_normalization(data, stats)` - Apply precomputed stats
- `denormalize_labels(normalized, stats)` - Convert back to original scale

**Parameters:**
- `method`: `:minmax`, `:zscore`, `:zscore_minmax`, or `:log`
- `mode`: `:rowwise` (default), `:columnwise`, or `:global` (matrices only)
- `range`: `(-1, 1)` (default) for min-max and zscore_minmax
- `clip_quantiles`: `(0.01, 0.99)` (default) or `nothing`

## License

MIT License - see [LICENSE](LICENSE) file for details.
