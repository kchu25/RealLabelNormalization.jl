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
