# Statistics computation functions

"""
Compute normalization statistics for training data.

These functions compute the necessary statistics from training data that will be used 
to normalize training, validation, and test sets consistently.

# Statistics Computed:
- **Min-Max**: `min_val`, `max_val` (or `min_vals`, `max_vals` for columnwise)
- **Z-Score**: `mean`, `std` (or `means`, `stds` for columnwise)
- **All methods**: `method`, `mode`, `range`, `clip_quantiles`, `clip_bounds`

# Clipping Parameters:
- **`clip_quantiles`**: Input percentile values (0-1), e.g., `(0.01, 0.99)` for 1st-99th percentiles
- **`clip_bounds`**: Actual computed quantile values, e.g., `(lower=2.1, upper=98.7)` for clipping

# NaN Handling:
- NaN values are ignored when computing statistics
- If all values are NaN, appropriate NaN statistics are returned with warnings
"""

function _compute_stats_vector(
    labels::AbstractVector, 
    method::Symbol, 
    range::Tuple{Real,Real}, 
    clip_quantiles::Union{Nothing,Tuple{Real,Real}}
)
    T = eltype(labels)
    # Compute clip bounds if clipping is requested
    clip_bounds = nothing
    if clip_quantiles !== nothing
        valid_data = filter(!isnan, labels)
        if !isempty(valid_data)
            lower_bound, upper_bound = quantile(valid_data, [clip_quantiles[1], clip_quantiles[2]])
            lower_bound, upper_bound = convert(T, lower_bound), convert(T, upper_bound)
            clip_bounds = (lower=lower_bound, upper=upper_bound)
        end
    end
    
    if method == :minmax
        min_val, max_val = _safe_extrema(labels)
        min_val, max_val = convert(T, min_val), convert(T, max_val)
        return (
            method=:minmax, 
            min_val=min_val, 
            max_val=max_val, 
            range=range, 
            mode=:vector, 
            clip_quantiles=clip_quantiles,
            clip_bounds=clip_bounds
        )
    else # :zscore
        mu, sigma = _safe_mean_std(labels)
        mu, sigma = convert(T, mu), convert(T, sigma)
        return (
            method=:zscore, 
            mean=mu, 
            std=sigma, 
            mode=:vector, 
            clip_quantiles=clip_quantiles,
            clip_bounds=clip_bounds
        )
    end
end

function _compute_stats_global(
    labels::AbstractMatrix, 
    method::Symbol, 
    range::Tuple{Real,Real}, 
    clip_quantiles::Union{Nothing,Tuple{Real,Real}}
)
    T = eltype(labels)
    # Compute clip bounds if clipping is requested
    clip_bounds = nothing
    if clip_quantiles !== nothing
        valid_data = filter(!isnan, vec(labels))
        if !isempty(valid_data)
            lower_bound, upper_bound = quantile(valid_data, [clip_quantiles[1], clip_quantiles[2]])
            lower_bound, upper_bound = convert(T, lower_bound), convert(T, upper_bound)
            clip_bounds = (lower=lower_bound, upper=upper_bound)
        end
    end
    
    if method == :minmax
        min_val, max_val = _safe_extrema(labels)
        min_val, max_val = convert(T, min_val), convert(T, max_val)
        return (
            method=:minmax, 
            min_val=min_val, 
            max_val=max_val, 
            range=range, 
            mode=:global, 
            clip_quantiles=clip_quantiles,
            clip_bounds=clip_bounds
        )
    else # :zscore
        mu, sigma = _safe_mean_std(labels)
        mu, sigma = convert(T, mu), convert(T, sigma)
        return (
            method=:zscore, 
            mean=mu, 
            std=sigma, 
            mode=:global, 
            clip_quantiles=clip_quantiles,
            clip_bounds=clip_bounds
        )
    end
end

function _compute_stats_columnwise(
    labels::AbstractMatrix, 
    method::Symbol, 
    range::Tuple{Real,Real}, 
    clip_quantiles::Union{Nothing,Tuple{Real,Real}}
)
    n_cols = size(labels, 2)
    T = eltype(labels)
    # Compute clip bounds for each column if clipping is requested
    clip_bounds = nothing
    if clip_quantiles !== nothing
        column_bounds = []
        for col in 1:n_cols
            valid_data = filter(!isnan, labels[:, col])
            if !isempty(valid_data)
                lower_bound, upper_bound = quantile(valid_data, [clip_quantiles[1], clip_quantiles[2]])
                lower_bound, upper_bound = convert(T, lower_bound), convert(T, upper_bound)
                push!(column_bounds, (lower=lower_bound, upper=upper_bound))
            else
                push!(column_bounds, (lower=NaN, upper=NaN))
            end
        end
        clip_bounds = column_bounds
    end
    
    if method == :minmax
        min_vals = T[]
        max_vals = T[]
        for col in 1:n_cols
            min_val, max_val = _safe_extrema(labels[:, col])
            min_val, max_val = convert(T, min_val), convert(T, max_val)
            push!(min_vals, min_val)
            push!(max_vals, max_val)
        end
        return (
            method=:minmax, 
            min_vals=min_vals, 
            max_vals=max_vals, 
            range=range, 
            mode=:columnwise, 
            clip_quantiles=clip_quantiles,
            clip_bounds=clip_bounds
        )
    else # :zscore
        means = T[]
        stds = T[]
        for col in 1:n_cols
            mu, sigma = _safe_mean_std(labels[:, col])
            mu, sigma = convert(T, mu), convert(T, sigma)
            push!(means, mu)
            push!(stds, sigma)
        end
        return (
            method=:zscore, 
            means=means, 
            stds=stds, 
            mode=:columnwise, 
            clip_quantiles=clip_quantiles,
            clip_bounds=clip_bounds
        )
    end
end
