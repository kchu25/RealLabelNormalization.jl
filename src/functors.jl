# Functors for GPU-compatible denormalization
# These functors can be passed to CUDA kernels for efficient denormalization

"""
    MinMaxScaleBack{T<:AbstractFloat}

Functor for denormalizing min-max normalized values back to original scale.
Compatible with CUDA kernels - all fields are scalars (bitstype).

# Fields
- `min_val::T`: Original minimum value
- `max_val::T`: Original maximum value  
- `range_low::T`: Lower bound of normalized range
- `range_high::T`: Upper bound of normalized range

# Example
```julia
functor = MinMaxScaleBack{Float32}(1.0f0, 5.0f0, -1.0f0, 1.0f0)
original = functor(0.0f0)  # Returns 3.0f0 (midpoint)
```
"""
struct MinMaxScaleBack{T<:AbstractFloat}
    min_val::T
    max_val::T
    range_low::T
    range_high::T
end

function (f::MinMaxScaleBack)(x)
    # Reverse: range -> [0,1] -> original scale
    range_01 = (x - f.range_low) / (f.range_high - f.range_low)
    return f.min_val + range_01 * (f.max_val - f.min_val)
end

"""
    ZScoreScaleBack{T<:AbstractFloat}

Functor for denormalizing z-score normalized values back to original scale.
Compatible with CUDA kernels - all fields are scalars (bitstype).

# Fields
- `mean::T`: Original mean value
- `std::T`: Original standard deviation

# Example
```julia
functor = ZScoreScaleBack{Float32}(3.0f0, 1.5f0)
original = functor(0.0f0)  # Returns 3.0f0 (the mean)
```
"""
struct ZScoreScaleBack{T<:AbstractFloat}
    mean::T
    std::T
end

function (f::ZScoreScaleBack)(x)
    return x * f.std + f.mean
end

"""
    LogScaleBack{T<:AbstractFloat}

Functor for denormalizing log-transformed values back to original scale.
Compatible with CUDA kernels - all fields are scalars (bitstype).

# Fields
- `offset::T`: Offset added before log transformation

# Example
```julia
functor = LogScaleBack{Float32}(1.0f0)
original = functor(0.0f0)  # Returns 0.0f0 (exp(0) - 1)
```
"""
struct LogScaleBack{T<:AbstractFloat}
    offset::T
end

function (f::LogScaleBack)(x)
    return exp(x) - f.offset
end

"""
    ZScoreMinMaxScaleBack{T<:AbstractFloat}

Functor for denormalizing values that were z-score normalized then min-max scaled.
Compatible with CUDA kernels - all fields are scalars (bitstype).

This handles the two-step normalization: z-score → min-max bounds.

# Fields
- `mean::T`: Original mean value (for z-score)
- `std::T`: Original standard deviation (for z-score)
- `z_min::T`: Minimum z-score value from training data
- `z_max::T`: Maximum z-score value from training data
- `range_low::T`: Lower bound applied after z-score
- `range_high::T`: Upper bound applied after z-score

# Example
```julia
functor = ZScoreMinMaxScaleBack{Float32}(3.0f0, 1.5f0, -2.0f0, 2.0f0, 0.0f0, 1.0f0)
# Denormalizes a value that was z-score normalized then scaled to [0,1]
original = functor(0.5f0)  
```
"""
struct ZScoreMinMaxScaleBack{T<:AbstractFloat}
    mean::T
    std::T
    z_min::T
    z_max::T
    range_low::T
    range_high::T
end

function (f::ZScoreMinMaxScaleBack)(x)
    # First reverse the min-max scaling back to z-score range
    range_01 = (x - f.range_low) / (f.range_high - f.range_low)
    z_score = f.z_min + range_01 * (f.z_max - f.z_min)
    # Then reverse z-score to original scale
    return z_score * f.std + f.mean
end

"""
    LogMinMaxScaleBack{T<:AbstractFloat}

Functor for denormalizing values that were log-transformed then min-max scaled.
Compatible with CUDA kernels - all fields are scalars (bitstype).

This handles the two-step normalization: log transformation → min-max bounds.

# Fields
- `offset::T`: Offset added before log transformation
- `log_min::T`: Minimum log value from training data
- `log_max::T`: Maximum log value from training data
- `range_low::T`: Lower bound applied after log transformation
- `range_high::T`: Upper bound applied after log transformation

# Example
```julia
functor = LogMinMaxScaleBack{Float32}(0.0f0, 0.0f0, 4.6f0, 0.0f0, 1.0f0)
# Denormalizes a value that was log-transformed then scaled to [0,1]
original = functor(0.5f0)  
```
"""
struct LogMinMaxScaleBack{T<:AbstractFloat}
    offset::T
    log_min::T
    log_max::T
    range_low::T
    range_high::T
end

function (f::LogMinMaxScaleBack)(x)
    # First reverse the min-max scaling back to log range
    range_01 = (x - f.range_low) / (f.range_high - f.range_low)
    log_val = f.log_min + range_01 * (f.log_max - f.log_min)
    # Then reverse log to original scale
    return exp(log_val) - f.offset
end

"""
    ColumnwiseScaleBack{T<:AbstractFloat, F, N}

Functor for denormalizing columnwise-normalized values.
Uses a tuple of functors for GPU compatibility (bitstype).

# Type Parameters
- `T`: Float type (Float32, Float64)
- `F`: Type of the per-column functor
- `N`: Number of columns (compile-time constant)

# Fields
- `functors::NTuple{N,F}`: Tuple of functors, one per column

# Example
```julia
col_functors = (
    MinMaxScaleBack{Float32}(0.0f0, 10.0f0, -1.0f0, 1.0f0),
    ZScoreScaleBack{Float32}(5.0f0, 2.0f0)
)
functor = ColumnwiseScaleBack(col_functors)
original_col1 = functor(0.0f0, 1)  # Denormalize for column 1
original_col2 = functor(1.5f0, 2)  # Denormalize for column 2
```

# GPU Usage Note
For GPU kernels with large numbers of columns, consider using the raw parameter
arrays (min_vals, max_vals, etc.) directly from the stats object instead of the functor.
"""
struct ColumnwiseScaleBack{T<:AbstractFloat, F, N}
    functors::NTuple{N,F}
end

# Constructor that accepts Vector and converts to tuple
function ColumnwiseScaleBack{T,F}(functors::Vector{F}) where {T<:AbstractFloat, F}
    ColumnwiseScaleBack{T,F,length(functors)}(Tuple(functors))
end

function (f::ColumnwiseScaleBack)(x, col_idx)
    return f.functors[col_idx](x)
end

"""
    RowwiseScaleBack{T<:AbstractFloat, F, N}

Functor for denormalizing rowwise-normalized values.
Uses a tuple of functors for GPU compatibility (bitstype).

# Type Parameters
- `T`: Float type (Float32, Float64)
- `F`: Type of the per-row functor
- `N`: Number of rows (compile-time constant)

# Fields
- `functors::NTuple{N,F}`: Tuple of functors, one per row

# Example
```julia
row_functors = (
    MinMaxScaleBack{Float32}(0.0f0, 10.0f0, -1.0f0, 1.0f0),
    ZScoreScaleBack{Float32}(5.0f0, 2.0f0)
)
functor = RowwiseScaleBack(row_functors)
original_row1 = functor(0.0f0, 1)  # Denormalize for row 1
original_row2 = functor(1.5f0, 2)  # Denormalize for row 2
```

# GPU Usage Note
For GPU kernels with large numbers of rows, consider using the raw parameter
arrays (min_vals, max_vals, etc.) directly from the stats object instead of the functor.
Tuples with many elements can increase compilation time.
"""
struct RowwiseScaleBack{T<:AbstractFloat, F, N}
    functors::NTuple{N,F}
end

# Constructor that accepts Vector and converts to tuple
function RowwiseScaleBack{T,F}(functors::Vector{F}) where {T<:AbstractFloat, F}
    RowwiseScaleBack{T,F,length(functors)}(Tuple(functors))
end

function (f::RowwiseScaleBack)(x, row_idx)
    return f.functors[row_idx](x)
end
