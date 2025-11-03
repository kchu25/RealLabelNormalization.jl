# Example: Using scale_back_functor for GPU-compatible denormalization

```julia
using RealLabelNormalization

# Basic example: Vector normalization
train_labels = Float32[1.0, 2.0, 3.0, 4.0, 5.0]
stats = compute_normalization_stats(train_labels; method=:zscore)

# Access the functor
functor = stats.scale_back_functor
println("Functor type: ", typeof(functor))  # ZScoreScaleBack{Float32}

# Use functor directly (element-wise)
normalized_value = 0.0f0  # Some normalized prediction
original_value = functor(normalized_value)
println("Denormalized value: ", original_value)  # Should be close to the mean

# For CPU: existing denormalize_labels still works
predictions_norm = Float32[-1.0, 0.0, 1.0]
predictions_original = denormalize_labels(predictions_norm, stats)

# For GPU (future CUDA.jl integration):
# using CUDA
# 
# function denormalize_kernel!(output, input, functor)
#     i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
#     if i <= length(input)
#         @inbounds output[i] = functor(input[i])
#     end
#     return nothing
# end
#
# predictions_gpu = CuArray(predictions_norm)
# output_gpu = similar(predictions_gpu)
# threads = 256
# blocks = cld(length(predictions_gpu), threads)
# @cuda threads=threads blocks=blocks denormalize_kernel!(
#     output_gpu, predictions_gpu, stats.scale_back_functor
# )
# predictions_original_gpu = Array(output_gpu)
```

## Columnwise Example

```julia
# Matrix with column-wise normalization
weather_train = Float32[20.5 65.0 1013.2; 22.1 58.3 1015.8; 18.9 72.1 1008.9]
stats_col = compute_normalization_stats(weather_train; mode=:columnwise, method=:zscore)

# The functor for columnwise mode
col_functor = stats_col.scale_back_functor
println("Columnwise functor type: ", typeof(col_functor))

# Denormalize specific columns
normalized_temp = 0.0f0  # Normalized temperature (column 1)
original_temp = col_functor(normalized_temp, 1)

normalized_humidity = 1.5f0  # Normalized humidity (column 2)
original_humidity = col_functor(normalized_humidity, 2)

println("Original temperature: ", original_temp)
println("Original humidity: ", original_humidity)
```

## Rowwise Example

```julia
# Matrix with row-wise normalization (default for Flux.jl)
samples = Float32[1.0 2.0 3.0; 10.0 20.0 30.0; -1.0 0.0 1.0]
stats_row = compute_normalization_stats(samples; mode=:rowwise, method=:minmax)

# The functor for rowwise mode
row_functor = stats_row.scale_back_functor
println("Rowwise functor type: ", typeof(row_functor))

# Denormalize specific rows
normalized_sample1 = 0.0f0  # Normalized value from row 1
original_sample1 = row_functor(normalized_sample1, 1)

normalized_sample2 = -0.5f0  # Normalized value from row 2
original_sample2 = row_functor(normalized_sample2, 2)

println("Original sample 1: ", original_sample1)
println("Original sample 2: ", original_sample2)
```

## Benefits

1. **Type stability**: Functors have concrete field types (Float32 by default), enabling compiler optimizations
2. **GPU compatibility**: Can be passed to CUDA kernels without serialization issues
3. **Zero overhead**: Inlines well in both CPU and GPU code
4. **Backward compatible**: Existing `denormalize_labels` function still works as before
5. **Custom kernels**: Advanced users can write custom GPU kernels using the functor

## Future CUDA Integration

When you're ready to add GPU support, the functors are ready to use:

```julia
# Custom CUDA kernel example
using CUDA

function denormalize_matrix_columnwise_kernel!(output, input, functor, n_cols)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    total = length(input)
    
    if idx <= total
        row = ((idx - 1) ÷ n_cols) + 1
        col = ((idx - 1) % n_cols) + 1
        @inbounds output[idx] = functor(input[idx], col)
    end
    return nothing
end

# Usage
predictions_gpu = CuArray(predictions_matrix_norm)
output_gpu = similar(predictions_gpu)
n_cols = size(predictions_matrix_norm, 2)
threads = 256
blocks = cld(length(predictions_gpu), threads)

@cuda threads=threads blocks=blocks denormalize_matrix_columnwise_kernel!(
    output_gpu, predictions_gpu, stats_col.scale_back_functor, n_cols
)
```
