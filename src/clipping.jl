# Outlier clipping functions

"""
Clip outliers using quantiles before normalization.
"""
function _clip_outliers(labels::AbstractVector, clip_quantiles::Tuple{Real,Real}, mode::Symbol)
    valid_data = filter(!isnan, labels)
    if isempty(valid_data)
        @warn "All values are NaN, cannot clip outliers"
        return labels
    end
    T = eltype(labels)
    lower_q, upper_q = quantile(valid_data, [clip_quantiles[1], clip_quantiles[2]])
    lower_q, upper_q = convert(T, lower_q), convert(T, upper_q)
    return clamp.(labels, lower_q, upper_q)
end

function _clip_outliers(labels::AbstractMatrix, clip_quantiles::Tuple{Real,Real}, mode::Symbol)
    T = eltype(labels)
    if mode == :global
        # Clip based on global quantiles
        valid_data = filter(!isnan, vec(labels))
        if isempty(valid_data)
            @warn "All values are NaN, cannot clip outliers"
            return labels
        end
        lower_q, upper_q = quantile(valid_data, [clip_quantiles[1], clip_quantiles[2]])
        lower_q, upper_q = convert(T, lower_q), convert(T, upper_q)
        return clamp.(labels, lower_q, upper_q)
    elseif mode == :columnwise
        # Clip each column independently
        clipped = similar(labels)
        for col in axes(labels, 2)
            column_data = @view labels[:, col]
            valid_data = filter(!isnan, column_data)
            if isempty(valid_data)
                @warn "Column $col has all NaN values, cannot clip outliers"
                clipped[:, col] = column_data
            else
                lower_q, upper_q = quantile(valid_data, [clip_quantiles[1], clip_quantiles[2]])
                lower_q, upper_q = convert(T, lower_q), convert(T, upper_q)
                clipped[:, col] = clamp.(column_data, lower_q, upper_q)
            end
        end
        return clipped
    elseif mode == :rowwise
        # Clip each row independently
        clipped = similar(labels)
        for row in axes(labels, 1)
            row_data = @view labels[row, :]
            valid_data = filter(!isnan, row_data)
            if isempty(valid_data)
                @warn "Row $row has all NaN values, cannot clip outliers"
                clipped[row, :] = row_data
            else
                lower_q, upper_q = quantile(valid_data, [clip_quantiles[1], clip_quantiles[2]])
                lower_q, upper_q = convert(T, lower_q), convert(T, upper_q)
                clipped[row, :] = clamp.(row_data, lower_q, upper_q)
            end
        end
        return clipped
    else
        error("Unsupported mode for _clip_outliers: $mode")
    end
end

"""
Apply training clip bounds to validation/test data.
"""
function _apply_training_clip_bounds(labels::AbstractArray, stats::NamedTuple)
    # Use the stored clip bounds from training data instead of recomputing quantiles
    if stats.clip_bounds === nothing
        return labels  # No clipping was used during training
    end
    
    if stats.mode == :vector || stats.mode == :global
        # Single set of bounds for all data
        lower_bound, upper_bound = stats.clip_bounds.lower, stats.clip_bounds.upper
        if isnan(lower_bound) || isnan(upper_bound)
            @warn "Training clip bounds contain NaN, cannot clip"
            return labels
        end
        return clamp.(labels, lower_bound, upper_bound)
    elseif stats.mode == :columnwise
        # Per-column bounds
        clipped = similar(labels)
        for col in axes(labels, 2)
            if col <= length(stats.clip_bounds)
                lower_bound = stats.clip_bounds[col].lower
                upper_bound = stats.clip_bounds[col].upper
                if isnan(lower_bound) || isnan(upper_bound)
                    @warn "Column $col training clip bounds contain NaN, cannot clip"
                    clipped[:, col] = labels[:, col]
                else
                    clipped[:, col] = clamp.(labels[:, col], lower_bound, upper_bound)
                end
            else
                @warn "Column $col has no training clip bounds, cannot clip"
                clipped[:, col] = labels[:, col]
            end
        end
        return clipped
    elseif stats.mode == :rowwise
        # Per-row bounds
        clipped = similar(labels)
        for row in axes(labels, 1)
            if row <= length(stats.clip_bounds)
                lower_bound = stats.clip_bounds[row].lower
                upper_bound = stats.clip_bounds[row].upper
                if isnan(lower_bound) || isnan(upper_bound)
                    @warn "Row $row training clip bounds contain NaN, cannot clip"
                    clipped[row, :] = labels[row, :]
                else
                    clipped[row, :] = clamp.(labels[row, :], lower_bound, upper_bound)
                end
            else
                @warn "Row $row has no training clip bounds, cannot clip"
                clipped[row, :] = labels[row, :]
            end
        end
        return clipped
    else
        error("Unsupported mode for _apply_training_clip_bounds: $(stats.mode)")
    end
end
