# GPU-Compatible Functor Usage

## Overview

RealLabelNormalization provides GPU-compatible functors for efficient denormalization on CUDA devices. All functors are now **bitstype**, meaning they can be passed directly to CUDA kernels without serialization overhead.

## Functor Types

### Simple Functors (Scalar Operations)

These functors work with single values and are fully GPU-compatible:

#### 1. **MinMaxScaleBack{T}**
```julia
functor = MinMaxScaleBack{Float32}(min_val, max_val, range_low, range_high)
original = functor(normalized_value)
```

#### 2. **ZScoreScaleBack{T}**
```julia
functor = ZScoreScaleBack{Float32}(mean, std)
original = functor(normalized_value)
```

#### 3. **LogScaleBack{T}**
```julia
functor = LogScaleBack{Float32}(offset)
original = functor(log_value)
```

### Columnwise/Rowwise Functors (Multi-dimensional Operations)

These functors use **tuples** instead of vectors to maintain bitstype compatibility:

#### **ColumnwiseScaleBack{T, F, N}**
- `N` = number of columns (compile-time constant)
- Uses `NTuple{N,F}` for GPU compatibility

#### **RowwiseScaleBack{T, F, N}**
- `N` = number of rows (compile-time constant)
- Uses `NTuple{N,F}` for GPU compatibility

## Usage Examples

### Basic CPU Usage

```julia
using RealLabelNormalization

# Single vector
labels = Float32[1.0, 2.0, 3.0, 4.0, 5.0]
stats = compute_normalization_stats(labels; method=:minmax)
normalized = apply_normalization(labels, stats)

# Get the functor
functor = stats[:scale_back_functor]

# Denormalize element-wise
original = [functor(x) for x in normalized]
```

### Columnwise Denormalization

```julia
# Multi-column data
labels = Float32[1.0 10.0; 2.0 20.0; 3.0 30.0]
stats = compute_normalization_stats(labels; method=:minmax, mode=:columnwise)
normalized = apply_normalization(labels, stats)

# Get columnwise functor
functor = stats[:scale_back_functor]

# Denormalize with column indices
denormalized = similar(normalized)
for i in 1:size(normalized, 1)
    for j in 1:size(normalized, 2)
        denormalized[i, j] = functor(normalized[i, j], j)
    end
end
```

### GPU/CUDA Usage

```julia
using CUDA
using RealLabelNormalization

# Prepare data on GPU
labels_cpu = Float32[1.0, 2.0, 3.0, 4.0, 5.0]
stats = compute_normalization_stats(labels_cpu; method=:minmax)
normalized_gpu = CuArray(apply_normalization(labels_cpu, stats))

# Get functor (automatically bitstype)
functor = stats[:scale_back_functor]

# Broadcast on GPU - functor can be passed to kernel
denormalized_gpu = functor.(normalized_gpu)

# Or use in custom kernel
function denorm_kernel!(output, input, functor)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= length(input)
        @inbounds output[idx] = functor(input[idx])
    end
    return nothing
end

# Launch kernel
output_gpu = similar(normalized_gpu)
threads = 256
blocks = cld(length(normalized_gpu), threads)
@cuda threads=threads blocks=blocks denorm_kernel!(output_gpu, normalized_gpu, functor)
```

### Rowwise GPU Usage with Custom Wrapper

```julia
# For rowwise operations, you might want to wrap the functor
struct Functor2Arg{F}
    functor::F
    second_arg::Int
end

@inline (f::Functor2Arg)(x) = f.functor(x, f.second_arg)

# Prepare rowwise data
labels = Float32[1.0 2.0 3.0; 10.0 20.0 30.0]
stats = compute_normalization_stats(labels; method=:minmax, mode=:rowwise)
normalized_gpu = CuArray(apply_normalization(labels, stats))

# Get rowwise functor
functor = stats[:scale_back_functor]

# Denormalize each row on GPU
denormalized_gpu = similar(normalized_gpu)
for row in 1:size(normalized_gpu, 1)
    row_functor = Functor2Arg(functor, row)
    denormalized_gpu[row, :] .= row_functor.(normalized_gpu[row, :])
end
```

## Performance Considerations

### For Small Datasets (< 10 columns/rows)
- Use the tuple-based `ColumnwiseScaleBack`/`RowwiseScaleBack` functors
- These are fully bitstype and very efficient on GPU

### For Large Datasets (> 10 columns/rows)
- Consider using the raw parameter arrays from stats directly:
  ```julia
  # Instead of functors, use parameter arrays
  min_vals_gpu = CuArray(stats[:min_vals])
  max_vals_gpu = CuArray(stats[:max_vals])
  range_low = stats[:range][1]
  range_high = stats[:range][2]
  
  # Broadcast denormalization formula
  denormalized = @. min_vals_gpu .+ 
      ((normalized .- range_low) / (range_high - range_low)) .* 
      (max_vals_gpu .- min_vals_gpu)
  ```

### Compilation Time
- Tuple-based functors with many elements (>20) can increase compilation time
- For very large numbers of columns/rows, using parameter arrays directly is more efficient

## Verification

Check if a functor is GPU-compatible:

```julia
using RealLabelNormalization

labels = Float32[1.0, 2.0, 3.0]
stats = compute_normalization_stats(labels; method=:minmax)
functor = stats[:scale_back_functor]

# Should return true
println("Is GPU compatible: ", isbitstype(typeof(functor)))
```

## Summary of Changes

### Before (Non-GPU Compatible)
```julia
struct ColumnwiseScaleBack{T, F}
    functors::Vector{F}  # Not bitstype!
end
```

### After (GPU Compatible)
```julia
struct ColumnwiseScaleBack{T, F, N}
    functors::NTuple{N, F}  # Bitstype!
end
```

All scalar functors (`MinMaxScaleBack`, `ZScoreScaleBack`, `LogScaleBack`) were already bitstype and remain unchanged.
