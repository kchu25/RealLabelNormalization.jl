module RealLabelNormalization

using Statistics
using StatsBase: quantile

include("clipping.jl")    # _clip_outliers functions
include("methods.jl")     # _normalize_*, _apply_*, _denormalize_* functions  
include("stats.jl")       # _compute_stats_* functions

# Include main API last (depends on above)
include("core.jl")        # normalize_labels, compute_normalization_stats, etc.

export normalize_labels, compute_normalization_stats, apply_normalization, denormalize_labels


end
