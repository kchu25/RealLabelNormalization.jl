# Core normalization API

"""
    normalize_labels(labels; method=:minmax, range=(-1, 1), mode=:global, clip_quantiles=(0.01, 0.99))

Normalize labels with various normalization methods and modes. Handles NaN values by ignoring them 
in statistical computations and preserving them in the output.

# Arguments
- `labels`: Vector or matrix where the last dimension is the number of samples
- `method::Symbol`: Normalization method
  - `:minmax`: Min-max normalization (default)
  - `:zscore`: Z-score normalization (mean=0, std=1)
  - `:log`: Log normalization (log-transform with automatic offset for non-positive values)
- `range::Tuple{Real,Real}`: Target range for min-max normalization (default: (-1, 1))
  - `(-1, 1)`: Scaled min-max to `[-1,1]` (default)
  - `(0, 1)`: Standard min-max to [0,1]
  - Custom ranges: e.g., `(-2, 2)`
  - Note: Ignored for :zscore and :log methods
- `mode::Symbol`: Normalization scope
  - `:global`: Normalize across all values (default)
  - `:columnwise`: Normalize each column independently
  - `:rowwise`: Normalize each row independently
- `clip_quantiles::Union{Nothing,Tuple{Real,Real}}`: Percentile values (0-1) for outlier clipping before normalization
  - `(0.01, 0.99)`: Clip to 1st-99th percentiles (default)
  - `(0.05, 0.95)`: Clip to 5th-95th percentiles (more aggressive)
  - `nothing`: No clipping

# NaN Handling
- NaN values are ignored when computing statistics (min, max, mean, std, quantiles)
- NaN values are preserved in the output (remain as NaN)
- If all values in a column are NaN, appropriate warnings are issued and NaN is returned

# Returns
- Normalized labels with same shape as input

# Examples
```julia
# Vector labels (single target)
labels = [1.0, 5.0, 3.0, 8.0, 2.0, 100.0]  # 100.0 is outlier

# Min-max to [-1,1] with outlier clipping (default)
normalized = normalize_labels(labels)

# Min-max to [0,1] 
normalized = normalize_labels(labels; range=(0, 1))

# Z-score normalization with outlier clipping
normalized = normalize_labels(labels; method=:zscore)

# Log normalization (useful for skewed distributions)
normalized = normalize_labels(labels; method=:log)

# Matrix labels (multi-target)
labels_matrix = [1.0 10.0; 5.0 20.0; 3.0 15.0; 8.0 25.0; 1000.0 5.0]  # Outlier in col 1

# Global normalization with clipping
normalized = normalize_labels(labels_matrix; mode=:global)

# Column-wise normalization with clipping 
normalized = normalize_labels(labels_matrix; mode=:columnwise)

# Row-wise normalization with clipping
normalized = normalize_labels(labels_matrix; mode=:rowwise)
```
"""
function normalize_labels(labels::AbstractArray; 
                         method::Symbol=:minmax, 
                         range::Tuple{Real,Real}=(-1, 1),
                         mode::Symbol=:global,
                         clip_quantiles::Union{Nothing,Tuple{Real,Real}}=(0.01, 0.99),
                         warn_on_nan::Bool=true)
    # Input validation
    if method ∉ [:minmax, :zscore, :log]
        throw(ArgumentError("method must be :minmax, :zscore, or :log, got :$method"))
    end
    if mode ∉ [:global, :columnwise, :rowwise]
        throw(ArgumentError("mode must be :global, :columnwise, or :rowwise, got :$mode"))
    end
    if method == :zscore && range != (-1, 1)
        @warn "range parameter input $range is ignored for z-score normalization (produces mean=0, std=1, e.g. ~[-3,3] range)"
    end
    if method == :log && range != (-1, 1)
        @warn "range parameter input $range is ignored for log normalization (produces log-scaled values)"
    end
    if clip_quantiles !== nothing
        if length(clip_quantiles) != 2 || clip_quantiles[1] >= clip_quantiles[2]
            throw(ArgumentError("clip_quantiles must be (lower, upper) with lower < upper"))
        end
        if clip_quantiles[1] < 0 || clip_quantiles[2] > 1
            throw(ArgumentError("clip_quantiles must be between 0 and 1"))
        end
    end
    # Apply clipping if requested
    clipped_labels = clip_quantiles === nothing ? labels : _clip_outliers(labels, clip_quantiles, mode)
    # Handle different input types
    if ndims(clipped_labels) == 1
        return _normalize_vector(clipped_labels, method, range; warn_on_nan=warn_on_nan)
    elseif ndims(clipped_labels) == 2
        if mode == :global
            return _normalize_global(clipped_labels, method, range; warn_on_nan=warn_on_nan)
        elseif mode == :columnwise
            return _normalize_columnwise(clipped_labels, method, range; warn_on_nan=warn_on_nan)
        else # :rowwise
            return _normalize_rowwise(clipped_labels, method, range; warn_on_nan=warn_on_nan)
        end
    else
        throw(ArgumentError("labels must be 1D or 2D array, got $(ndims(clipped_labels))D"))
    end
end

"""
    compute_normalization_stats(labels; method=:minmax, mode=:global, 
    range=(-1, 1), clip_quantiles=(0.01, 0.99))

Compute normalization statistics from training data for later application to validation/test sets.

# Inputs
- `labels`: Vector or matrix where the last dimension is the number of samples
- `method::Symbol`: Normalization method
  - `:minmax`: Min-max normalization (default)
  - `:zscore`: Z-score normalization (mean=0, std=1)
  - `:log`: Log normalization (log-transform with automatic offset for non-positive values)
- `range::Tuple{Real,Real}`: Target range for min-max normalization (default (-1, 1))
    - `(-1, 1)`: Scaled min-max to `[-1,1]` (default)
    - `(0, 1)`: Standard min-max to [0,1]
    - Custom ranges: e.g., `(-2, 2)`
    - Note: Ignored for :zscore and :log methods
- `mode::Symbol`: Normalization scope
  - `:global`: Normalize across all values (default)
  - `:columnwise`: Normalize each column independently
  - `:rowwise`: Normalize each row independently
- `clip_quantiles::Union{Nothing,Tuple{Real,Real}}`: Percentile values (0-1) for outlier clipping before normalization
  - `(0.01, 0.99)`: Clip to 1st-99th percentiles (default)
  - `(0.05, 0.95)`: Clip to 5th-95th percentiles (more aggressive)
  - `nothing`: No clipping

# Returns
- Named tuple with normalization parameters that can be used with `apply_normalization`

# Example
```julia
# Compute stats from training data with outlier clipping
train_stats = compute_normalization_stats(train_labels; method=:zscore, mode=:columnwise, clip_quantiles=(0.05, 0.95))

# Apply to validation/test data (uses same clipping bounds)
val_normalized = apply_normalization(val_labels, train_stats)
test_normalized = apply_normalization(test_labels, train_stats)

# Log normalization for skewed distributions
log_stats = compute_normalization_stats(train_labels; method=:log)
val_log_normalized = apply_normalization(val_labels, log_stats)
```
"""
function compute_normalization_stats(labels::AbstractArray; 
                                   method::Symbol=:minmax,
                                   range::Tuple{Real,Real}=(-1, 1),
                                   mode::Symbol=:global,
                                   clip_quantiles::Union{Nothing,Tuple{Real,Real}}=(0.01, 0.99),
                                   warn_on_nan::Bool=true)
    # Apply clipping if requested
    clipped_labels = clip_quantiles === nothing ? labels : _clip_outliers(labels, clip_quantiles, mode)
    if ndims(clipped_labels) == 1
        return _compute_stats_vector(clipped_labels, method, range, clip_quantiles; warn_on_nan=warn_on_nan)
    elseif ndims(clipped_labels) == 2
        if mode == :global
            return _compute_stats_global(clipped_labels, method, range, clip_quantiles; warn_on_nan=warn_on_nan)
        elseif mode == :columnwise
            return _compute_stats_columnwise(clipped_labels, method, range, clip_quantiles; warn_on_nan=warn_on_nan)
        else # :rowwise
            return _compute_stats_rowwise(clipped_labels, method, range, clip_quantiles; warn_on_nan=warn_on_nan)
        end
    else
        throw(ArgumentError("labels must be 1D or 2D array, got $(ndims(clipped_labels))D"))
    end
end

"""
    apply_normalization(labels, stats)

Apply pre-computed normalization statistics to new data (validation/test sets).

Ensures consistent normalization across train/validation/test splits using only training statistics.
This includes applying the same clipping bounds if they were used during training.
"""
function apply_normalization(labels::AbstractArray, stats::NamedTuple)
    # Apply same clipping as was used during training
    clipped_labels = stats.clip_quantiles === nothing ? labels : _apply_training_clip_bounds(labels, stats)
    
    if stats.method == :minmax
        return _apply_minmax_normalization(clipped_labels, stats)
    elseif stats.method == :zscore
        return _apply_zscore_normalization(clipped_labels, stats)
    elseif stats.method == :log
        return _apply_log_normalization(clipped_labels, stats)
    else
        throw(ArgumentError("Unknown method in stats: $(stats.method)"))
    end
end

"""
    denormalize_labels(normalized_labels, stats)

Convert normalized labels back to original scale using stored statistics.

Useful for interpreting model predictions in original units.
"""
function denormalize_labels(normalized_labels::AbstractArray, stats::NamedTuple)
    if stats.method == :minmax
        return _denormalize_minmax(normalized_labels, stats)
    elseif stats.method == :zscore
        return _denormalize_zscore(normalized_labels, stats)
    elseif stats.method == :log
        return _denormalize_log(normalized_labels, stats)
    else
        throw(ArgumentError("Unknown method in stats: $(stats.method)"))
    end
end
