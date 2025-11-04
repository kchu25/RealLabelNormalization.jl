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

# Helper function for zscore + minmax normalization
function _zscore_minmax_normalize(data::AbstractArray, range::Tuple{Real,Real}; warn_on_nan::Bool=true)
    # Step 1: Z-score normalization
    mu, sigma = _safe_mean_std(data; warn_on_nan=warn_on_nan)
    if isnan(mu) || isnan(sigma)
        if warn_on_nan
            @warn "Cannot compute z-score with NaN statistics, returning NaN array"
        end
        return fill(NaN, size(data)), NaN, NaN, NaN, NaN
    end
    if sigma == 0
        if warn_on_nan
            @warn "Standard deviation is zero, returning zeros"
        end
        return zeros(eltype(data), size(data)), mu, sigma, 0.0, 0.0
    end
    
    z_scores = (data .- mu) ./ sigma
    
    # Step 2: Min-max normalization on the z-scores
    valid_z = filter(!isnan, z_scores)
    if isempty(valid_z)
        return fill(NaN, size(data)), mu, sigma, NaN, NaN
    end
    
    z_min, z_max = extrema(valid_z)
    if z_min == z_max
        # All z-scores are the same (shouldn't happen unless sigma=0, already handled)
        return zeros(eltype(data), size(data)), mu, sigma, z_min, z_max
    end
    
    # Scale z-scores from [z_min, z_max] to target range
    normalized_01 = (z_scores .- z_min) ./ (z_max - z_min)
    result = _scale_to_range(normalized_01, range)
    
    return result, mu, sigma, z_min, z_max
end

function _normalize_vector(labels::AbstractVector, method::Symbol, range::Tuple{Real,Real}, log_shift::Real; 
                        warn_on_nan::Bool=true)
    if method == :minmax
        normalized_01, min_val, max_val = _minmax_normalize_to_01(labels; warn_on_nan=warn_on_nan)
        if min_val == max_val
            @warn "All labels have the same value ($min_val), returning zeros"
            return normalized_01  # Already zeros from helper
        end
        # Scale from [0,1] to target range
        return _scale_to_range(normalized_01, range)
    elseif method == :zscore
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
    elseif method == :zscore_minmax
        result, _, _, _, _ = _zscore_minmax_normalize(labels, range; warn_on_nan=warn_on_nan)
        return result
    else # :log
        # Compute offset to ensure all values are positive
        valid_data = filter(!isnan, labels)
        if isempty(valid_data)
            if warn_on_nan
                @warn "All values are NaN, cannot apply log normalization"
            end
            return fill(NaN, size(labels))
        end
        min_val = minimum(valid_data)
        offset = min_val <= 0 ? abs(min_val) + log_shift : 0.0
        # Apply log transformation
        result = similar(labels)
        for i in eachindex(labels)
            if isnan(labels[i])
                result[i] = NaN
            else
                result[i] = log(labels[i] + offset)
            end
        end
        return result
    end
end

function _normalize_global(labels::AbstractMatrix, method::Symbol, range::Tuple{Real,Real}, log_shift::Real; warn_on_nan::Bool=true)
    if method == :minmax
        normalized_01, min_val, max_val = _minmax_normalize_to_01(labels; warn_on_nan=warn_on_nan)
        if min_val == max_val
            @warn "All labels have the same value ($min_val), returning zeros"
            return normalized_01  # Already zeros from helper
        end
        # Scale from [0,1] to target range
        return _scale_to_range(normalized_01, range)
    elseif method == :zscore
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
    elseif method == :zscore_minmax
        result, _, _, _, _ = _zscore_minmax_normalize(labels, range; warn_on_nan=warn_on_nan)
        return result
    else # :log
        # Compute offset to ensure all values are positive
        valid_data = filter(!isnan, vec(labels))
        if isempty(valid_data)
            @warn "All values are NaN, cannot apply log normalization"
            return fill(NaN, size(labels))
        end
        min_val = minimum(valid_data)
        offset = min_val <= 0 ? abs(min_val) + log_shift : 0.0
        # Apply log transformation
        result = similar(labels)
        for i in eachindex(labels)
            if isnan(labels[i])
                result[i] = NaN
            else
                result[i] = log(labels[i] + offset)
            end
        end
        return result
    end
end

function _normalize_columnwise(labels::AbstractMatrix, method::Symbol, range::Tuple{Real,Real}, log_shift::Real; warn_on_nan::Bool=true)
    normalized = similar(labels)
    
    for col in axes(labels, 2) # for col in
        column_data = @view labels[:, col]
        if method == :minmax
            normalized_01, min_val, max_val = _minmax_normalize_to_01(column_data; warn_on_nan=warn_on_nan)
            if min_val == max_val
                @warn "Column $col has constant values ($min_val), setting to zeros"
                normalized[:, col] = normalized_01  # Already zeros from helper
            else
                normalized[:, col] = _scale_to_range(normalized_01, range)
            end
        elseif method == :zscore
            mu, sigma = _safe_mean_std(column_data; warn_on_nan=warn_on_nan)
            if isnan(mu) || isnan(sigma)
                @warn "Column $col has all NaN values, setting to NaN"
                normalized[:, col] .= NaN
            elseif sigma == 0
                @warn "Column $col has zero standard deviation, setting to zeros"
                normalized[:, col] .= 0
            else
                normalized[:, col] = (column_data .- mu) ./ sigma
            end
        elseif method == :zscore_minmax
            result, _, _, _, _ = _zscore_minmax_normalize(column_data, range; warn_on_nan=warn_on_nan)
            normalized[:, col] = result
        else # :log
            valid_data = filter(!isnan, column_data)
            if isempty(valid_data)
                @warn "Column $col has all NaN values, setting to NaN"
                normalized[:, col] .= NaN
            else
                min_val = minimum(valid_data)
                offset = min_val <= 0 ? abs(min_val) + log_shift : 0.0
                for i in eachindex(column_data)
                    if isnan(column_data[i])
                        normalized[i, col] = NaN
                    else
                        normalized[i, col] = log(column_data[i] + offset)
                    end
                end
            end
        end
    end
    
    return normalized
end

function _normalize_rowwise(labels::AbstractMatrix, method::Symbol, range::Tuple{Real,Real}, log_shift::Real; warn_on_nan::Bool=true)
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
        elseif method == :zscore
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
        elseif method == :zscore_minmax
            result, _, _, _, _ = _zscore_minmax_normalize(row_data, range; warn_on_nan=warn_on_nan)
            normalized[row, :] = result
        else # :log
            valid_data = filter(!isnan, row_data)
            if isempty(valid_data)
                if warn_on_nan
                    @warn "Row $row has all NaN values, setting to NaN"
                end
                normalized[row, :] .= NaN
            else
                min_val = minimum(valid_data)
                offset = min_val <= 0 ? abs(min_val) + log_shift : 0.0
                for i in eachindex(row_data)
                    if isnan(row_data[i])
                        normalized[row, i] = NaN
                    else
                        normalized[row, i] = log(row_data[i] + offset)
                    end
                end
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

function _apply_log_normalization(labels::AbstractArray, stats::NamedTuple)
    if stats.mode == :vector || stats.mode == :global
        offset = stats.offset
        result = similar(labels)
        for i in eachindex(labels)
            if isnan(labels[i])
                result[i] = NaN
            else
                result[i] = log(labels[i] + offset)
            end
        end
        return result
    elseif stats.mode == :columnwise
        normalized = similar(labels)
        for col in axes(labels, 2)
            offset = stats.offsets[col]
            for i in axes(labels, 1)
                if isnan(labels[i, col])
                    normalized[i, col] = NaN
                else
                    normalized[i, col] = log(labels[i, col] + offset)
                end
            end
        end
        return normalized
    else # :rowwise
        normalized = similar(labels)
        for row in axes(labels, 1)
            offset = stats.offsets[row]
            for i in axes(labels, 2)
                if isnan(labels[row, i])
                    normalized[row, i] = NaN
                else
                    normalized[row, i] = log(labels[row, i] + offset)
                end
            end
        end
        return normalized
    end
end

function _apply_zscore_minmax_normalization(labels::AbstractArray, stats::NamedTuple)
    if stats.mode == :vector || stats.mode == :global
        mu = stats.mean
        sigma = stats.std
        z_min = stats.z_min
        z_max = stats.z_max
        range_low, range_high = stats.range
        
        result = similar(labels)
        for i in eachindex(labels)
            if isnan(labels[i])
                result[i] = NaN
            else
                # Step 1: Z-score normalization
                z = sigma == 0.0 ? 0.0 : (labels[i] - mu) / sigma
                # Step 2: Min-max to [0,1]
                norm_01 = (z_max == z_min) ? 0.0 : (z - z_min) / (z_max - z_min)
                # Step 3: Scale to target range
                result[i] = range_low + norm_01 * (range_high - range_low)
            end
        end
        return result
    elseif stats.mode == :columnwise
        normalized = similar(labels)
        for col in axes(labels, 2)
            mu = stats.means[col]
            sigma = stats.stds[col]
            z_min = stats.z_mins[col]
            z_max = stats.z_maxs[col]
            range_low, range_high = stats.range
            
            for i in axes(labels, 1)
                if isnan(labels[i, col])
                    normalized[i, col] = NaN
                else
                    z = sigma == 0.0 ? 0.0 : (labels[i, col] - mu) / sigma
                    norm_01 = (z_max == z_min) ? 0.0 : (z - z_min) / (z_max - z_min)
                    normalized[i, col] = range_low + norm_01 * (range_high - range_low)
                end
            end
        end
        return normalized
    else # :rowwise
        normalized = similar(labels)
        for row in axes(labels, 1)
            mu = stats.means[row]
            sigma = stats.stds[row]
            z_min = stats.z_mins[row]
            z_max = stats.z_maxs[row]
            range_low, range_high = stats.range
            
            for i in axes(labels, 2)
                if isnan(labels[row, i])
                    normalized[row, i] = NaN
                else
                    z = sigma == 0.0 ? 0.0 : (labels[row, i] - mu) / sigma
                    norm_01 = (z_max == z_min) ? 0.0 : (z - z_min) / (z_max - z_min)
                    normalized[row, i] = range_low + norm_01 * (range_high - range_low)
                end
            end
        end
        return normalized
    end
end

