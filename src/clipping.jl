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
    lower_q, upper_q = quantile(valid_data, [clip_quantiles[1], clip_quantiles[2]])
    return clamp.(labels, lower_q, upper_q)
end

function _clip_outliers(labels::AbstractMatrix, clip_quantiles::Tuple{Real,Real}, mode::Symbol)
    if mode == :global
        # Clip based on global quantiles
        valid_data = filter(!isnan, vec(labels))
        if isempty(valid_data)
            @warn "All values are NaN, cannot clip outliers"
            return labels
        end
        lower_q, upper_q = quantile(valid_data, [clip_quantiles[1], clip_quantiles[2]])
        return clamp.(labels, lower_q, upper_q)
    else # :columnwise
        # Clip each column independently
        clipped = similar(labels)
        for col in axes(labels, 2) # for col in
            column_data = @view labels[:, col]
            valid_data = filter(!isnan, column_data)
            if isempty(valid_data)
                @warn "Column $col has all NaN values, cannot clip outliers"
                clipped[:, col] = column_data
            else
                lower_q, upper_q = quantile(valid_data, [clip_quantiles[1], clip_quantiles[2]])
                clipped[:, col] = clamp.(column_data, lower_q, upper_q)
            end
        end
        return clipped
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
    else # :columnwise
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
    end
end
