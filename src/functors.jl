# Functors for GPU-compatible denormalization
# These functors can be passed to CUDA kernels for efficient denormalization

"""
    MinMaxScaleBack{T<:AbstractFloat}

Functor for denormalizing min-max normalized values back to original scale.
Compatible with CUDA kernels.

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
Compatible with CUDA kernels.

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
Compatible with CUDA kernels.

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
    ColumnwiseScaleBack{T<:AbstractFloat, F}

Functor for denormalizing columnwise-normalized values.
Contains a vector of per-column functors.

# Fields
- `functors::Vector{F}`: Vector of functors, one per column

# Example
```julia
col_functors = [
    MinMaxScaleBack{Float32}(0.0f0, 10.0f0, -1.0f0, 1.0f0),
    ZScoreScaleBack{Float32}(5.0f0, 2.0f0)
]
functor = ColumnwiseScaleBack(col_functors)
original_col1 = functor(0.0f0, 1)  # Denormalize for column 1
original_col2 = functor(1.5f0, 2)  # Denormalize for column 2
```
"""
struct ColumnwiseScaleBack{T<:AbstractFloat, F}
    functors::Vector{F}
end

function (f::ColumnwiseScaleBack)(x, col_idx)
    return f.functors[col_idx](x)
end

"""
    RowwiseScaleBack{T<:AbstractFloat, F}

Functor for denormalizing rowwise-normalized values.
Contains a vector of per-row functors.

# Fields
- `functors::Vector{F}`: Vector of functors, one per row

# Example
```julia
row_functors = [
    MinMaxScaleBack{Float32}(0.0f0, 10.0f0, -1.0f0, 1.0f0),
    ZScoreScaleBack{Float32}(5.0f0, 2.0f0)
]
functor = RowwiseScaleBack(row_functors)
original_row1 = functor(0.0f0, 1)  # Denormalize for row 1
original_row2 = functor(1.5f0, 2)  # Denormalize for row 2
```
"""
struct RowwiseScaleBack{T<:AbstractFloat, F}
    functors::Vector{F}
end

function (f::RowwiseScaleBack)(x, row_idx)
    return f.functors[row_idx](x)
end
