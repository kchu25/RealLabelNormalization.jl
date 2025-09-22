# Examples

This page provides comprehensive examples of the **stats-based workflow** for leak-free label normalization.

## ⚠️ Critical: Always Use the Stats-Based Workflow

**NEVER** use `normalize_labels()` directly on your full dataset. This causes data leakage! Always follow the three-step pattern:

1. **Compute stats from training data ONLY**
2. **Apply the same stats to validation/test data**  
3. **Denormalize predictions using the same stats**

## Basic Examples

### Example 1: Single Target Regression (Stats-Based)

```julia
using RealLabelNormalization
using Random
Random.seed!(42)

# Simulate house prices with some outliers
train_prices = [200_000, 250_000, 180_000, 320_000, 275_000, 
                190_000, 2_000_000, 210_000, 290_000, 240_000]  # 2M is outlier
test_prices = [220_000, 280_000, 195_000, 310_000]

println("Training prices: ", train_prices)
println("Test prices: ", test_prices)

# Step 1: Compute stats from training data ONLY
stats = compute_normalization_stats(train_prices; method=:zscore, clip_quantiles=(0.01, 0.99))

# Step 2: Apply SAME stats to both training and test data
train_normalized = apply_normalization(train_prices, stats)
test_normalized = apply_normalization(test_prices, stats)

println("Training normalized: ", train_normalized)
println("Test normalized: ", test_normalized)

# Step 3: Denormalize predictions using SAME stats
predictions_normalized = [0.5, -0.2, 0.8, 0.1]  # Model outputs
predictions_original = denormalize_labels(predictions_normalized, stats)
println("Predictions (original scale): ", predictions_original)
```

### Example 2: Multi-Target Regression (Stats-Based)

```julia
# Simulate multi-target regression: [temperature, humidity, pressure]
weather_train = [20.5 65.0 1013.2;
                22.1 58.3 1015.8;
                18.9 72.1 1008.9;
                25.4 45.2 1020.1;
                19.2 68.7 1011.4;
                50.0 30.0 950.0;   # Outlier row
                21.8 61.5 1016.3]

weather_test = [19.5 70.0 1012.5;
               23.2 55.0 1018.0;
               17.8 75.0 1009.5]

println("Training data shape: ", size(weather_train))
println("Test data shape: ", size(weather_test))

# Step 1: Compute stats from training data ONLY
stats_col = compute_normalization_stats(weather_train; mode=:columnwise, method=:zscore)
stats_global = compute_normalization_stats(weather_train; mode=:global, method=:zscore)

# Step 2: Apply SAME stats to both training and test data
train_norm_col = apply_normalization(weather_train, stats_col)
test_norm_col = apply_normalization(weather_test, stats_col)

train_norm_global = apply_normalization(weather_train, stats_global)
test_norm_global = apply_normalization(weather_test, stats_global)

println("Column-wise normalized ranges (training):")
for i in 1:size(train_norm_col, 2)
    col_range = [minimum(train_norm_col[:, i]), maximum(train_norm_col[:, i])]
    println("  Column $i: $col_range")
end

println("Global normalized range (training): [$(minimum(train_norm_global)), $(maximum(train_norm_global))]")

# Step 3: Denormalize predictions using SAME stats
predictions_norm = [0.5 -0.2 0.8; -0.3 0.7 -0.1]
predictions_original = denormalize_labels(predictions_norm, stats_col)
println("Predictions (original scale): ", predictions_original)
```

## Machine Learning Workflow Examples

### Example 3: Complete Train/Validation/Test Pipeline (Stats-Based)

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
stats = compute_normalization_stats(y_train; method=:zscore, clip_quantiles=(0.01, 0.99))
println("\\nNormalization statistics computed from training data:")
println(stats)

# Step 2: Apply SAME stats to all splits
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

# Step 4: Denormalize predictions back to original scale using SAME stats
y_pred_original = denormalize_labels(y_pred_norm, stats)

println("\\nPrediction comparison (first 10 samples):")
println("  True:      ", y_test[1:10])
println("  Predicted: ", y_pred_original[1:10])
println("  Error:     ", abs.(y_test[1:10] - y_pred_original[1:10]))
```

### Example 4: Handling Missing Data (Stats-Based)

```julia
using RealLabelNormalization

# Training data with missing values (NaN)
train_with_missing = [1.0, 2.0, NaN, 4.0, 5.0, 6.0, NaN, 8.0, 100.0, 9.0]
test_with_missing = [1.5, NaN, 3.2, 4.8, NaN, 7.1]

println("Training data: ", train_with_missing)
println("Test data: ", test_with_missing)
println("Valid training values: ", train_with_missing[.!isnan.(train_with_missing)])

# Step 1: Compute stats from valid training data only
stats = compute_normalization_stats(train_with_missing; method=:zscore, clip_quantiles=(0.01, 0.99))
println("\\nComputed statistics from valid training data: ", stats)

# Step 2: Apply SAME stats to both training and test data
train_norm = apply_normalization(train_with_missing, stats)
test_norm = apply_normalization(test_with_missing, stats)

println("Training normalized: ", train_norm)
println("Test normalized: ", test_norm)

# Check that NaN positions are preserved
println("Training NaN preserved? ", isnan.(train_with_missing) == isnan.(train_norm))
println("Test NaN preserved? ", isnan.(test_with_missing) == isnan.(test_norm))

# Step 3: Denormalize predictions using SAME stats
predictions_norm = [0.5, NaN, -0.2, 0.8, NaN, 0.1]
predictions_original = denormalize_labels(predictions_norm, stats)
println("Predictions (original scale): ", predictions_original)
```

## The Golden Rule: Stats-Based Workflow

### ✅ CORRECT: Always Use This Pattern

```julia
# Step 1: Compute stats from training data ONLY
stats = compute_normalization_stats(train_labels; method=:zscore, clip_quantiles=(0.01, 0.99))

# Step 2: Apply SAME stats to all data splits
train_norm = apply_normalization(train_labels, stats)
val_norm = apply_normalization(val_labels, stats)    # Same stats
test_norm = apply_normalization(test_labels, stats)  # Same stats

# Step 3: Denormalize predictions using SAME stats
predictions_original = denormalize_labels(predictions_normalized, stats)
```

### ❌ WRONG: Direct Normalization (Causes Data Leakage)

```julia
# DON'T DO THIS - causes data leakage!
train_norm = normalize_labels(train_labels)
test_norm = normalize_labels(test_labels)  # Different stats = data leakage!
```

### Why Stats-Based Workflow is Critical

1. **Prevents Data Leakage**: Test data never influences normalization parameters
2. **Consistent Scaling**: All data splits use identical normalization
3. **Proper Validation**: Model performance reflects real-world generalization
4. **Correct Predictions**: Denormalization uses the same parameters as training

## Advanced Examples

### Example 5: Cross-Validation with Consistent Stats

```julia
using RealLabelNormalization
using Statistics

# Simulate a dataset for cross-validation
n_samples = 1000
X = randn(n_samples, 5)
y = 2 * X[:, 1] + 0.5 * X[:, 2].^2 - X[:, 3] + 0.1 * randn(n_samples)

# 5-fold cross-validation
n_folds = 5
fold_size = n_samples ÷ n_folds

for fold in 1:n_folds
    println("\\n=== Fold $fold ===")
    
    # Split data for this fold
    val_start = (fold - 1) * fold_size + 1
    val_end = fold * fold_size
    val_idx = val_start:val_end
    train_idx = setdiff(1:n_samples, val_idx)
    
    X_train, y_train = X[train_idx, :], y[train_idx]
    X_val, y_val = X[val_idx, :], y[val_idx]
    
    # Step 1: Compute stats from training fold ONLY
    stats = compute_normalization_stats(y_train; method=:zscore, clip_quantiles=(0.01, 0.99))
    
    # Step 2: Apply SAME stats to both training and validation
    y_train_norm = apply_normalization(y_train, stats)
    y_val_norm = apply_normalization(y_val, stats)  # Same stats!
    
    println("Training stats: mean=$(mean(y_train)), std=$(std(y_train))")
    println("Validation stats: mean=$(mean(y_val)), std=$(std(y_val))")
    println("Normalized training: mean=$(mean(y_train_norm)), std=$(std(y_train_norm))")
    println("Normalized validation: mean=$(mean(y_val_norm)), std=$(std(y_val_norm))")
    
    # Step 3: Train model and make predictions
    # model = train_model(X_train, y_train_norm)
    # val_pred_norm = model(X_val)
    # val_pred_original = denormalize_labels(val_pred_norm, stats)
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
