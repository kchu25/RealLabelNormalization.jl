# Normalization method implementations

"""
Min-max and z-score normalization functions.
"""

# NaN-safe statistical helper functions
function _safe_extrema(data::AbstractArray; warn_on_nan::Bool=true)
    valid_data = filter(!isnan, data)
    if isempty(valid_data)
        if warn_on_nan
            @warn "All values are NaN, cannot compute extrema"
        end
        return NaN, NaN
    end
    return extrema(valid_data)
end

function _safe_mean_std(data::AbstractArray; warn_on_nan::Bool=true)
    valid_data = filter(!isnan, data)
    if isempty(valid_data)
        if warn_on_nan
            @warn "All values are NaN, cannot compute mean/std"
        end
        return NaN, NaN
    elseif length(valid_data) == 1
        if warn_on_nan
            @warn "Only one valid value, std is zero"
        end
        return mean(valid_data), 0.0
    end
    return mean(valid_data), std(valid_data)
end

# Helper function to avoid code duplication
function _minmax_normalize_to_01(data::AbstractArray; warn_on_nan::Bool=true)
    min_val, max_val = _safe_extrema(data; warn_on_nan=warn_on_nan)
    if isnan(min_val) || isnan(max_val)
        if warn_on_nan
            @warn "Cannot normalize data with all NaN values, returning NaN array"
        end
        return fill(NaN, size(data)), NaN, NaN
    end
    if min_val == max_val
        return zeros(eltype(data), size(data)), min_val, max_val
    end
    normalized_01 = (data .- min_val) ./ (max_val - min_val)
    return normalized_01, min_val, max_val
end

# Helper function to scale [0,1] normalized data to target range
function _scale_to_range(normalized_01::AbstractArray, range::Tuple{Real,Real})
    return range[1] .+ normalized_01 .* (range[2] - range[1])
end

function _normalize_vector(labels::AbstractVector, method::Symbol, range::Tuple{Real,Real}; 
                        warn_on_nan::Bool=true)
    if method == :minmax
        normalized_01, min_val, max_val = _minmax_normalize_to_01(labels; warn_on_nan=warn_on_nan)
        if min_val == max_val
            @warn "All labels have the same value ($min_val), returning zeros"
            return normalized_01  # Already zeros from helper
        end
        # Scale from [0,1] to target range
        return _scale_to_range(normalized_01, range)
    else # :zscore
        mu, sigma = _safe_mean_std(labels; warn_on_nan=warn_on_nan)
        if isnan(mu) || isnan(sigma)
            @warn "Cannot compute z-score with NaN statistics, returning NaN array"
            return fill(NaN, size(labels))
        end
        if sigma == 0
            @warn "Standard deviation is zero, returning zeros"
            return zeros(eltype(labels), size(labels))
        end
        return (labels .- mu) ./ sigma
    end
end

function _normalize_global(labels::AbstractMatrix, method::Symbol, range::Tuple{Real,Real})
    if method == :minmax
        normalized_01, min_val, max_val = _minmax_normalize_to_01(labels)
        if min_val == max_val
            @warn "All labels have the same value ($min_val), returning zeros"
            return normalized_01  # Already zeros from helper
        end
        # Scale from [0,1] to target range
        return _scale_to_range(normalized_01, range)
    else # :zscore
        mu, sigma = _safe_mean_std(labels)
        if isnan(mu) || isnan(sigma)
            @warn "Cannot compute z-score with NaN statistics, returning NaN array"
            return fill(NaN, size(labels))
        end
        if sigma == 0
            @warn "Standard deviation is zero, returning zeros"
            return zeros(eltype(labels), size(labels))
        end
        return (labels .- mu) ./ sigma
    end
end

function _normalize_columnwise(labels::AbstractMatrix, method::Symbol, range::Tuple{Real,Real})
    normalized = similar(labels)
    
    for col in axes(labels, 2) # for col in
        column_data = @view labels[:, col]
        if method == :minmax
            normalized_01, min_val, max_val = _minmax_normalize_to_01(column_data)
            if min_val == max_val
                @warn "Column $col has constant values ($min_val), setting to zeros"
                normalized[:, col] = normalized_01  # Already zeros from helper
            else
                normalized[:, col] = _scale_to_range(normalized_01, range)
            end
        else # :zscore
            mu, sigma = _safe_mean_std(column_data)
            if isnan(mu) || isnan(sigma)
                @warn "Column $col has all NaN values, setting to NaN"
                normalized[:, col] .= NaN
            elseif sigma == 0
                @warn "Column $col has zero standard deviation, setting to zeros"
                normalized[:, col] .= 0
            else
                normalized[:, col] = (column_data .- mu) ./ sigma
            end
        end
    end
    
    return normalized
end

function _normalize_rowwise(labels::AbstractMatrix, method::Symbol, range::Tuple{Real,Real}; warn_on_nan::Bool=true)
    normalized = similar(labels)
    for row in axes(labels, 1)
        row_data = @view labels[row, :]
        if method == :minmax
            normalized_01, min_val, max_val = _minmax_normalize_to_01(row_data; warn_on_nan=warn_on_nan)
            if min_val == max_val
                @warn "Row $row has constant values ($min_val), setting to zeros"
                normalized[row, :] = normalized_01  # Already zeros from helper
            else
                normalized[row, :] = _scale_to_range(normalized_01, range)
            end
        else # :zscore
            mu, sigma = _safe_mean_std(row_data; warn_on_nan=warn_on_nan)
            if isnan(mu) || isnan(sigma)
                @warn "Row $row has all NaN values, setting to NaN"
                normalized[row, :] .= NaN
            elseif sigma == 0
                @warn "Row $row has zero standard deviation, setting to zeros"
                normalized[row, :] .= 0
            else
                normalized[row, :] = (row_data .- mu) ./ sigma
            end
        end
    end
    return normalized
end

# Application helpers

function _apply_minmax_normalization(labels::AbstractArray, stats::NamedTuple)
    if stats.mode == :vector || stats.mode == :global
        min_val, max_val = stats.min_val, stats.max_val
        if min_val == max_val
            return zeros(eltype(labels), size(labels))
        end
        normalized_01 = (labels .- min_val) ./ (max_val - min_val)
        return _scale_to_range(normalized_01, stats.range)
    elseif stats.mode == :columnwise
        normalized = similar(labels)
        for col in axes(labels, 2)
            min_val, max_val = stats.min_vals[col], stats.max_vals[col]
            if min_val == max_val
                normalized[:, col] .= 0
            else
                column_data = @view labels[:, col]
                normalized_01 = (column_data .- min_val) ./ (max_val - min_val)
                normalized[:, col] = _scale_to_range(normalized_01, stats.range)
            end
        end
        return normalized
    else # :rowwise
        normalized = similar(labels)
        for row in axes(labels, 1)
            min_val, max_val = stats.min_vals[row], stats.max_vals[row]
            if min_val == max_val
                normalized[row, :] .= 0
            else
                row_data = @view labels[row, :]
                normalized_01 = (row_data .- min_val) ./ (max_val - min_val)
                normalized[row, :] = _scale_to_range(normalized_01, stats.range)
            end
        end
        return normalized
    end
end

function _apply_zscore_normalization(labels::AbstractArray, stats::NamedTuple)
    if stats.mode == :vector || stats.mode == :global
        mu, sigma = stats.mean, stats.std
        if sigma == 0
            return zeros(eltype(labels), size(labels))
        end
        return (labels .- mu) ./ sigma
    elseif stats.mode == :columnwise
        normalized = similar(labels)
        for col in axes(labels, 2)
            mu, sigma = stats.means[col], stats.stds[col]
            if sigma == 0
                normalized[:, col] .= 0
            else
                normalized[:, col] = @views (labels[:, col] .- mu) ./ sigma
            end
        end
        return normalized
    else # :rowwise
        normalized = similar(labels)
        for row in axes(labels, 1)
            mu, sigma = stats.means[row], stats.stds[row]
            if sigma == 0
                normalized[row, :] .= 0
            else
                normalized[row, :] = @views (labels[row, :] .- mu) ./ sigma
            end
        end
        return normalized
    end
end

# Denormalization helpers

function _denormalize_minmax(normalized_labels::AbstractArray, stats::NamedTuple)
    if stats.mode == :vector || stats.mode == :global
        min_val, max_val = stats.min_val, stats.max_val
        if min_val == max_val
            return fill(min_val, size(normalized_labels))
        end
        # Reverse: range -> [0,1] -> original scale
        range_01 = (normalized_labels .- stats.range[1]) ./ (stats.range[2] - stats.range[1])
        return min_val .+ range_01 .* (max_val - min_val)
    elseif stats.mode == :columnwise
        denormalized = similar(normalized_labels)
        for col in axes(normalized_labels, 2)
            min_val, max_val = stats.min_vals[col], stats.max_vals[col]
            if min_val == max_val
                denormalized[:, col] .= min_val
            else
                column_data = @view normalized_labels[:, col]
                range_01 = (column_data .- stats.range[1]) ./ (stats.range[2] - stats.range[1])
                denormalized[:, col] = min_val .+ range_01 .* (max_val - min_val)
            end
        end
        return denormalized
    else # :rowwise
        denormalized = similar(normalized_labels)
        for row in axes(normalized_labels, 1)
            min_val, max_val = stats.min_vals[row], stats.max_vals[row]
            if min_val == max_val
                denormalized[row, :] .= min_val
            else
                row_data = @view normalized_labels[row, :]
                range_01 = (row_data .- stats.range[1]) ./ (stats.range[2] - stats.range[1])
                denormalized[row, :] = min_val .+ range_01 .* (max_val - min_val)
            end
        end
        return denormalized
    end
end

function _denormalize_zscore(normalized_labels::AbstractArray, stats::NamedTuple)
    if stats.mode == :vector || stats.mode == :global
        mu, sigma = stats.mean, stats.std
        if sigma == 0
            return fill(mu, size(normalized_labels))
        end
        return normalized_labels .* sigma .+ mu
    elseif stats.mode == :columnwise
        denormalized = similar(normalized_labels)
        for col in axes(normalized_labels, 2)
            mu, sigma = stats.means[col], stats.stds[col]
            if sigma == 0
                denormalized[:, col] .= mu
            else
                denormalized[:, col] = @. (@view normalized_labels[:, col]) * sigma + mu
            end
        end
        return denormalized
    else # :rowwise
        denormalized = similar(normalized_labels)
        for row in axes(normalized_labels, 1)
            mu, sigma = stats.means[row], stats.stds[row]
            if sigma == 0
                denormalized[row, :] .= mu
            else
                denormalized[row, :] = @. (@view normalized_labels[row, :]) * sigma + mu
            end
        end
        return denormalized
    end
end
