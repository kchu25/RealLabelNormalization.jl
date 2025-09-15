# Examples

This page provides comprehensive examples of using RealLabelNormalization.jl in various scenarios.

## Basic Examples

### Example 1: Single Target Regression

```julia
using RealLabelNormalization
using Random
Random.seed!(42)

# Simulate house prices with some outliers
house_prices = [200_000, 250_000, 180_000, 320_000, 275_000, 
                190_000, 2_000_000, 210_000, 290_000, 240_000]  # 2M is outlier

println("Original prices: ", house_prices)
println("Min: $(minimum(house_prices)), Max: $(maximum(house_prices))")

# Normalize with outlier clipping
normalized = normalize_labels(house_prices)
println("Normalized: ", normalized)
println("Range: [$(minimum(normalized)), $(maximum(normalized))]")
```

### Example 2: Multi-Target Regression

```julia
# Simulate multi-target regression: [temperature, humidity, pressure]
weather_data = [20.5 65.0 1013.2;
                22.1 58.3 1015.8;
                18.9 72.1 1008.9;
                25.4 45.2 1020.1;
                19.2 68.7 1011.4;
                50.0 30.0 950.0;   # Outlier row
                21.8 61.5 1016.3]

println("Original data shape: ", size(weather_data))

# Column-wise normalization (each variable independently)
normalized_col = normalize_labels(weather_data; mode=:columnwise)
println("Column-wise normalized ranges:")
for i in 1:size(normalized_col, 2)
    col_range = [minimum(normalized_col[:, i]), maximum(normalized_col[:, i])]
    println("  Column $i: $col_range")
end

# Global normalization (all variables on same scale)
normalized_global = normalize_labels(weather_data; mode=:global)
println("Global normalized range: [$(minimum(normalized_global)), $(maximum(normalized_global))]")
```

## Machine Learning Workflow Examples

### Example 3: Complete Train/Validation/Test Pipeline

```julia
using RealLabelNormalization
using Random
Random.seed!(123)

# Simulate a regression dataset
n_samples = 1000
n_features = 5
X = randn(n_samples, n_features)

# Target with some non-linear relationship and outliers
y = 2 * X[:, 1] + 0.5 * X[:, 2].^2 - X[:, 3] + 0.1 * randn(n_samples)
# Add a few outliers
y[1:5] .+= 50 * randn(5)

# Split data
train_idx = 1:600
val_idx = 601:800
test_idx = 801:1000

X_train, y_train = X[train_idx, :], y[train_idx]
X_val, y_val = X[val_idx, :], y[val_idx]
X_test, y_test = X[test_idx, :], y[test_idx]

println("Original target statistics:")
println("  Train: mean=$(mean(y_train)), std=$(std(y_train))")
println("  Val:   mean=$(mean(y_val)), std=$(std(y_val))")
println("  Test:  mean=$(mean(y_test)), std=$(std(y_test))")

# Step 1: Compute normalization statistics from training data ONLY
stats = compute_normalization_stats(y_train; method=:zscore)
println("\\nNormalization statistics computed from training data:")
println(stats)

# Step 2: Apply normalization to all splits
y_train_norm = apply_normalization(y_train, stats)
y_val_norm = apply_normalization(y_val, stats)
y_test_norm = apply_normalization(y_test, stats)

println("\\nNormalized target statistics:")
println("  Train: mean=$(mean(y_train_norm)), std=$(std(y_train_norm))")
println("  Val:   mean=$(mean(y_val_norm)), std=$(std(y_val_norm))")
println("  Test:  mean=$(mean(y_test_norm)), std=$(std(y_test_norm))")

# Step 3: Train model on normalized data (placeholder)
# model = fit_model(X_train, y_train_norm)
# y_pred_norm = predict(model, X_test)

# Simulate some predictions
y_pred_norm = y_test_norm + 0.1 * randn(length(y_test_norm))  # Add some error

# Step 4: Denormalize predictions back to original scale
y_pred_original = denormalize_labels(y_pred_norm, stats)

println("\\nPrediction comparison (first 10 samples):")
println("  True:      ", y_test[1:10])
println("  Predicted: ", y_pred_original[1:10])
println("  Error:     ", abs.(y_test[1:10] - y_pred_original[1:10]))
```

### Example 4: Handling Missing Data

```julia
using RealLabelNormalization

# Data with missing values (NaN)
data_with_missing = [1.0, 2.0, NaN, 4.0, 5.0, 6.0, NaN, 8.0, 100.0, 9.0]

println("Original data: ", data_with_missing)
println("Valid values: ", data_with_missing[.!isnan.(data_with_missing)])

# Normalize - NaN values are preserved, stats computed on valid data
normalized = normalize_labels(data_with_missing)
println("Normalized: ", normalized)

# Check that NaN positions are preserved
println("NaN preserved? ", isnan.(data_with_missing) == isnan.(normalized))

# Compute statistics - works with missing data
stats = compute_normalization_stats(data_with_missing)
println("\\nComputed statistics: ", stats)
```

## Advanced Examples

### Example 5: Comparing Normalization Methods

```julia
using RealLabelNormalization
using Statistics

# Create data with different distributions
uniform_data = rand(100) * 100
normal_data = randn(100) * 20 .+ 50
skewed_data = [rand() < 0.8 ? rand() * 10 : rand() * 100 for _ in 1:100]

datasets = [("Uniform", uniform_data), ("Normal", normal_data), ("Skewed", skewed_data)]

for (name, data) in datasets
    println("\\n=== $name Data ===")
    println("Original: mean=$(mean(data)), std=$(std(data))")
    
    # Min-max normalization
    minmax_norm = normalize_labels(data; method=:minmax)
    println("Min-max: range=[$(minimum(minmax_norm)), $(maximum(minmax_norm))]")
    
    # Z-score normalization
    zscore_norm = normalize_labels(data; method=:zscore)
    println("Z-score: mean=$(mean(zscore_norm)), std=$(std(zscore_norm))")
    
    # Effect of outlier clipping
    no_clip = normalize_labels(data; clip_quantiles=nothing)
    with_clip = normalize_labels(data; clip_quantiles=(0.05, 0.95))
    println("No clip range: [$(minimum(no_clip)), $(maximum(no_clip))]")
    println("With clip range: [$(minimum(with_clip)), $(maximum(with_clip))]")
end
```

### Example 6: Custom Normalization Ranges

```julia
using RealLabelNormalization

# Original data
data = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

println("Original data: ", data)

# Different target ranges
ranges = [(-1, 1), (0, 1), (-2, 2), (-10, 10)]

for range in ranges
    normalized = normalize_labels(data; range=range, clip_quantiles=nothing)
    actual_range = (minimum(normalized), maximum(normalized))
    println("Target $range -> Actual $actual_range")
end
```

### Example 7: Multi-Target with Different Scales

```julia
using RealLabelNormalization

# Multi-target data with very different scales
# Column 1: Small values (0-10)
# Column 2: Medium values (100-1000)  
# Column 3: Large values (10000-100000)
multi_scale_data = [1.0 100.0 10000.0;
                    2.0 200.0 20000.0;
                    3.0 300.0 30000.0;
                    4.0 400.0 40000.0;
                    5.0 500.0 50000.0;
                    100.0 50000.0 5000.0]  # Outlier row

println("Original data ranges per column:")
for i in 1:3
    col_range = [minimum(multi_scale_data[:, i]), maximum(multi_scale_data[:, i])]
    println("  Column $i: $col_range")
end

# Compare global vs column-wise normalization
global_norm = normalize_labels(multi_scale_data; mode=:global)
column_norm = normalize_labels(multi_scale_data; mode=:columnwise)

println("\\nGlobal normalization - range per column:")
for i in 1:3
    col_range = [minimum(global_norm[:, i]), maximum(global_norm[:, i])]
    println("  Column $i: $col_range")
end

println("\\nColumn-wise normalization - range per column:")
for i in 1:3
    col_range = [minimum(column_norm[:, i]), maximum(column_norm[:, i])]
    println("  Column $i: $col_range")
end
```

## Performance Considerations

### Example 8: Large Dataset Handling

```julia
using RealLabelNormalization
using BenchmarkTools

# Simulate large dataset
large_data = randn(100_000, 10)  # 100k samples, 10 targets

println("Dataset size: ", size(large_data))

# Benchmark different operations
println("\\nPerformance benchmarks:")
@btime normalize_labels($large_data; mode=:columnwise)
@btime compute_normalization_stats($large_data; mode=:columnwise)
@btime apply_normalization($large_data, $stats) setup=(stats=compute_normalization_stats($large_data; mode=:columnwise))
```
