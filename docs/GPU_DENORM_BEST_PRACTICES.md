# GPU Denormalization Best Practices

## The Problem with Large Tuple Functors

While `NTuple`-based functors are technically bitstype and GPU-compatible, there are practical limitations:

1. **Compilation time**: Large tuples (>20 elements) cause slow compilation
2. **Broadcast complexity**: CUDA broadcast with wrapped functors can create complex intermediate types
3. **Memory overhead**: Each tuple element increases kernel parameter size

## Recommended Approaches

### Approach 1: Direct Parameter Arrays (BEST for large datasets)

Instead of using functors in GPU kernels, use the raw normalization parameters directly:

```julia
using CUDA
using RealLabelNormalization

# Compute stats on CPU
labels_cpu = Float32[1.0 2.0 3.0; 10.0 20.0 30.0; 100.0 200.0 300.0]
stats = compute_normalization_stats(labels_cpu; method=:minmax, mode=:rowwise)

# Transfer normalized data and parameters to GPU
normalized_cpu = apply_normalization(labels_cpu, stats)
normalized_gpu = CuArray(normalized_cpu)
min_vals_gpu = CuArray(stats[:min_vals])
max_vals_gpu = CuArray(stats[:max_vals])
range_low = Float32(stats[:range][1])
range_high = Float32(stats[:range][2])

# Custom CUDA kernel - uses raw parameters
function denorm_kernel!(output, input, min_vals, max_vals, range_low, range_high)
    row = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    col = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    
    if row <= size(input, 1) && col <= size(input, 2)
        # Denormalize using formula directly
        normalized_val = input[row, col]
        range_01 = (normalized_val - range_low) / (range_high - range_low)
        @inbounds output[row, col] = min_vals[row] + range_01 * (max_vals[row] - min_vals[row])
    end
    return nothing
end

# Launch kernel
output_gpu = similar(normalized_gpu)
threads = (16, 16)
blocks = cld.((size(normalized_gpu, 1), size(normalized_gpu, 2)), threads)
@cuda threads=threads blocks=blocks denorm_kernel!(
    output_gpu, normalized_gpu, min_vals_gpu, max_vals_gpu, range_low, range_high
)
```

### Approach 2: Simple Functor Broadcasting (GOOD for small datasets)

For datasets with few columns/rows (<10), use the functor directly:

```julia
using CUDA
using RealLabelNormalization

# Vector normalization (no row/column index needed)
labels_cpu = Float32[1.0, 2.0, 3.0, 4.0, 5.0]
stats = compute_normalization_stats(labels_cpu; method=:minmax)
normalized_gpu = CuArray(apply_normalization(labels_cpu, stats))

# Get functor and broadcast directly on GPU
functor = stats[:scale_back_functor]
denormalized_gpu = functor.(normalized_gpu)  # ✅ Works!

# Result back to CPU
result = Array(denormalized_gpu)
```

### Approach 3: Precompute Row/Column Functors (GOOD for medium datasets)

For columnwise/rowwise with moderate size (10-100), precompute per-column/row operations:

```julia
using CUDA
using RealLabelNormalization

labels_cpu = Float32[1.0 10.0; 2.0 20.0; 3.0 30.0]
stats = compute_normalization_stats(labels_cpu; method=:minmax, mode=:columnwise)
normalized_gpu = CuArray(apply_normalization(labels_cpu, stats))

functor = stats[:scale_back_functor]
denormalized_gpu = similar(normalized_gpu)

# Process each column separately (avoids wrapped functor issues)
for col in 1:size(normalized_gpu, 2)
    col_functor = functor.functors[col]  # Extract single-column functor
    denormalized_gpu[:, col] .= col_functor.(normalized_gpu[:, col])  # ✅ Works!
end
```

### Approach 4: Map-based Denormalization (ALTERNATIVE)

Use `map` instead of broadcast to avoid broadcast machinery:

```julia
using CUDA
using RealLabelNormalization

labels_cpu = Float32[1.0 2.0; 10.0 20.0]
stats = compute_normalization_stats(labels_cpu; method=:minmax, mode=:rowwise)
normalized_gpu = CuArray(apply_normalization(labels_cpu, stats))

functor = stats[:scale_back_functor]
denormalized_gpu = similar(normalized_gpu)

# Use map with explicit indices
for row in 1:size(normalized_gpu, 1)
    row_functor = functor.functors[row]
    denormalized_gpu[row, :] .= map(row_functor, normalized_gpu[row, :])
end
```

## When to Use Each Approach

| Dataset Size | Columns/Rows | Recommended Approach | Reason |
|--------------|--------------|----------------------|--------|
| Small | < 10 | Functor broadcasting | Simple, fast compilation |
| Medium | 10-100 | Precompute per-column | Balance of simplicity and performance |
| Large | > 100 | Direct parameters | Avoids large tuples, fast compilation |
| Any | Any | Direct parameters | Most flexible, best performance |

## Example: Complete Rowwise Log Denormalization on GPU

```julia
using CUDA
using RealLabelNormalization

# Setup
labels_cpu = randn(Float32, 1000, 50)  # 1000 rows, 50 columns
labels_cpu .= abs.(labels_cpu)  # Make positive for log
stats = compute_normalization_stats(labels_cpu; method=:log, mode=:rowwise)
normalized_cpu = apply_normalization(labels_cpu, stats)

# Transfer to GPU
normalized_gpu = CuArray(normalized_cpu)
offsets_gpu = CuArray(stats[:offsets])

# Custom kernel for log denormalization
function log_denorm_kernel!(output, input, offsets)
    row = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    col = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    
    if row <= size(input, 1) && col <= size(input, 2)
        @inbounds output[row, col] = exp(input[row, col]) - offsets[row]
    end
    return nothing
end

# Launch
output_gpu = similar(normalized_gpu)
threads = (16, 16)
blocks = cld.((size(normalized_gpu, 1), size(normalized_gpu, 2)), threads)
@cuda threads=threads blocks=blocks log_denorm_kernel!(output_gpu, normalized_gpu, offsets_gpu)

# Verify
CUDA.@allowscalar println("Error: ", maximum(abs.(output_gpu .- CuArray(labels_cpu))))
```

## Summary

**Yes, NTuple makes functors bitstype**, but for practical GPU usage:

1. ✅ **Use functors for simple vector operations** (no row/column indices)
2. ✅ **Use direct parameter arrays for complex operations** (cleaner, faster)
3. ⚠️ **Avoid wrapping functors in broadcasts** (causes type complexity)
4. ✅ **Extract individual tuple elements** when needed (clean and explicit)

The key insight: **being bitstype is necessary but not sufficient** for smooth GPU operation. The simpler the type, the better CUDA's compiler can optimize it.
