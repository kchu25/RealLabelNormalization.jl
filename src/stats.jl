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
    clip_quantiles::Union{Nothing,Tuple{Real,Real}};
    warn_on_nan::Bool=true
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
        min_val, max_val = _safe_extrema(labels; warn_on_nan=warn_on_nan)
        min_val, max_val = convert(T, min_val), convert(T, max_val)
        functor = MinMaxScaleBack{T}(min_val, max_val, T(range[1]), T(range[2]))
        return (
            method=:minmax, 
            min_val=min_val, 
            max_val=max_val, 
            range=range, 
            mode=:vector, 
            clip_quantiles=clip_quantiles,
            clip_bounds=clip_bounds,
            scale_back_functor=functor
        )
    elseif method == :zscore
        mu, sigma = _safe_mean_std(labels; warn_on_nan=warn_on_nan)
        mu, sigma = convert(T, mu), convert(T, sigma)
        functor = ZScoreScaleBack{T}(mu, sigma)
        return (
            method=:zscore, 
            mean=mu, 
            std=sigma, 
            mode=:vector, 
            clip_quantiles=clip_quantiles,
            clip_bounds=clip_bounds,
            scale_back_functor=functor
        )
    else # :log
        valid_data = filter(!isnan, labels)
        if isempty(valid_data)
            if warn_on_nan
                @warn "All values are NaN, cannot compute log normalization statistics"
            end
            functor = LogScaleBack{T}(convert(T, NaN))
            return (
                method=:log,
                offset=convert(T, NaN),
                mode=:vector,
                clip_quantiles=clip_quantiles,
                clip_bounds=clip_bounds,
                scale_back_functor=functor
            )
        end
        min_val = minimum(valid_data)
        offset = min_val <= 0 ? abs(min_val) + 1.0 : 0.0
        offset = convert(T, offset)
        functor = LogScaleBack{T}(offset)
        return (
            method=:log,
            offset=offset,
            mode=:vector,
            clip_quantiles=clip_quantiles,
            clip_bounds=clip_bounds,
            scale_back_functor=functor
        )
    end
end

function _compute_stats_global(
    labels::AbstractMatrix, 
    method::Symbol, 
    range::Tuple{Real,Real}, 
    clip_quantiles::Union{Nothing,Tuple{Real,Real}};
    warn_on_nan::Bool=true
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
        min_val, max_val = _safe_extrema(labels; warn_on_nan=warn_on_nan)
        min_val, max_val = convert(T, min_val), convert(T, max_val)
        functor = MinMaxScaleBack{T}(min_val, max_val, T(range[1]), T(range[2]))
        return (
            method=:minmax, 
            min_val=min_val, 
            max_val=max_val, 
            range=range, 
            mode=:global, 
            clip_quantiles=clip_quantiles,
            clip_bounds=clip_bounds,
            scale_back_functor=functor
        )
    elseif method == :zscore
        mu, sigma = _safe_mean_std(labels; warn_on_nan=warn_on_nan)
        mu, sigma = convert(T, mu), convert(T, sigma)
        functor = ZScoreScaleBack{T}(mu, sigma)
        return (
            method=:zscore, 
            mean=mu, 
            std=sigma, 
            mode=:global, 
            clip_quantiles=clip_quantiles,
            clip_bounds=clip_bounds,
            scale_back_functor=functor
        )
    else # :log
        valid_data = filter(!isnan, vec(labels))
        if isempty(valid_data)
            if warn_on_nan
                @warn "All values are NaN, cannot compute log normalization statistics"
            end
            functor = LogScaleBack{T}(convert(T, NaN))
            return (
                method=:log,
                offset=convert(T, NaN),
                mode=:global,
                clip_quantiles=clip_quantiles,
                clip_bounds=clip_bounds,
                scale_back_functor=functor
            )
        end
        min_val = minimum(valid_data)
        offset = min_val <= 0 ? abs(min_val) + 1.0 : 0.0
        offset = convert(T, offset)
        functor = LogScaleBack{T}(offset)
        return (
            method=:log,
            offset=offset,
            mode=:global,
            clip_quantiles=clip_quantiles,
            clip_bounds=clip_bounds,
            scale_back_functor=functor
        )
    end
end

function _compute_stats_columnwise(
    labels::AbstractMatrix, 
    method::Symbol, 
    range::Tuple{Real,Real}, 
    clip_quantiles::Union{Nothing,Tuple{Real,Real}};
    warn_on_nan::Bool=true
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
        functors = MinMaxScaleBack{T}[]
        for col in 1:n_cols
            min_val, max_val = _safe_extrema(labels[:, col]; warn_on_nan=warn_on_nan)
            min_val, max_val = convert(T, min_val), convert(T, max_val)
            push!(min_vals, min_val)
            push!(max_vals, max_val)
            push!(functors, MinMaxScaleBack{T}(min_val, max_val, T(range[1]), T(range[2])))
        end
        col_functor = ColumnwiseScaleBack{T, MinMaxScaleBack{T}}(functors)
        return (
            method=:minmax, 
            min_vals=min_vals, 
            max_vals=max_vals, 
            range=range, 
            mode=:columnwise, 
            clip_quantiles=clip_quantiles,
            clip_bounds=clip_bounds,
            scale_back_functor=col_functor
        )
    elseif method == :zscore
        means = T[]
        stds = T[]
        functors = ZScoreScaleBack{T}[]
        for col in 1:n_cols
            mu, sigma = _safe_mean_std(labels[:, col]; warn_on_nan=warn_on_nan)
            mu, sigma = convert(T, mu), convert(T, sigma)
            push!(means, mu)
            push!(stds, sigma)
            push!(functors, ZScoreScaleBack{T}(mu, sigma))
        end
        col_functor = ColumnwiseScaleBack{T, ZScoreScaleBack{T}}(functors)
        return (
            method=:zscore, 
            means=means, 
            stds=stds, 
            mode=:columnwise, 
            clip_quantiles=clip_quantiles,
            clip_bounds=clip_bounds,
            scale_back_functor=col_functor
        )
    else # :log
        offsets = T[]
        functors = LogScaleBack{T}[]
        for col in 1:n_cols
            valid_data = filter(!isnan, labels[:, col])
            if isempty(valid_data)
                if warn_on_nan
                    @warn "Column $col has all NaN values, cannot compute log normalization statistics"
                end
                push!(offsets, convert(T, NaN))
                push!(functors, LogScaleBack{T}(convert(T, NaN)))
            else
                min_val = minimum(valid_data)
                offset = min_val <= 0 ? abs(min_val) + 1.0 : 0.0
                push!(offsets, convert(T, offset))
                push!(functors, LogScaleBack{T}(convert(T, offset)))
            end
        end
        col_functor = ColumnwiseScaleBack{T, LogScaleBack{T}}(functors)
        return (
            method=:log,
            offsets=offsets,
            mode=:columnwise,
            clip_quantiles=clip_quantiles,
            clip_bounds=clip_bounds,
            scale_back_functor=col_functor
        )
    end
end

function _compute_stats_rowwise(
    labels::AbstractMatrix, 
    method::Symbol, 
    range::Tuple{Real,Real}, 
    clip_quantiles::Union{Nothing,Tuple{Real,Real}};
    warn_on_nan::Bool=true
)
    n_rows = size(labels, 1)
    T = eltype(labels)
    # Compute clip bounds for each row if clipping is requested
    clip_bounds = nothing
    if clip_quantiles !== nothing
        row_bounds = []
        for row in 1:n_rows
            valid_data = filter(!isnan, labels[row, :])
            if !isempty(valid_data)
                lower_bound, upper_bound = quantile(valid_data, [clip_quantiles[1], clip_quantiles[2]])
                lower_bound, upper_bound = convert(T, lower_bound), convert(T, upper_bound)
                push!(row_bounds, (lower=lower_bound, upper=upper_bound))
            else
                push!(row_bounds, (lower=NaN, upper=NaN))
            end
        end
        clip_bounds = row_bounds
    end
    if method == :minmax
        min_vals = T[]
        max_vals = T[]
        functors = MinMaxScaleBack{T}[]
        for row in 1:n_rows
            min_val, max_val = _safe_extrema(labels[row, :]; warn_on_nan=warn_on_nan)
            min_val, max_val = convert(T, min_val), convert(T, max_val)
            push!(min_vals, min_val)
            push!(max_vals, max_val)
            push!(functors, MinMaxScaleBack{T}(min_val, max_val, T(range[1]), T(range[2])))
        end
        row_functor = RowwiseScaleBack{T, MinMaxScaleBack{T}}(functors)
        return (
            method=:minmax, 
            min_vals=min_vals, 
            max_vals=max_vals, 
            range=range, 
            mode=:rowwise, 
            clip_quantiles=clip_quantiles,
            clip_bounds=clip_bounds,
            scale_back_functor=row_functor
        )
    elseif method == :zscore
        means = T[]
        stds = T[]
        functors = ZScoreScaleBack{T}[]
        for row in 1:n_rows
            mu, sigma = _safe_mean_std(labels[row, :]; warn_on_nan=warn_on_nan)
            mu, sigma = convert(T, mu), convert(T, sigma)
            push!(means, mu)
            push!(stds, sigma)
            push!(functors, ZScoreScaleBack{T}(mu, sigma))
        end
        row_functor = RowwiseScaleBack{T, ZScoreScaleBack{T}}(functors)
        return (
            method=:zscore, 
            means=means, 
            stds=stds, 
            mode=:rowwise, 
            clip_quantiles=clip_quantiles,
            clip_bounds=clip_bounds,
            scale_back_functor=row_functor
        )
    else # :log
        offsets = T[]
        functors = LogScaleBack{T}[]
        for row in 1:n_rows
            valid_data = filter(!isnan, labels[row, :])
            if isempty(valid_data)
                if warn_on_nan
                    @warn "Row $row has all NaN values, cannot compute log normalization statistics"
                end
                push!(offsets, convert(T, NaN))
                push!(functors, LogScaleBack{T}(convert(T, NaN)))
            else
                min_val = minimum(valid_data)
                offset = min_val <= 0 ? abs(min_val) + 1.0 : 0.0
                push!(offsets, convert(T, offset))
                push!(functors, LogScaleBack{T}(convert(T, offset)))
            end
        end
        row_functor = RowwiseScaleBack{T, LogScaleBack{T}}(functors)
        return (
            method=:log,
            offsets=offsets,
            mode=:rowwise,
            clip_quantiles=clip_quantiles,
            clip_bounds=clip_bounds,
            scale_back_functor=row_functor
        )
    end
end
