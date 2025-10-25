using RealLabelNormalization

println("=" ^ 80)
println("Log Normalization Demo")
println("=" ^ 80)

# Example 1: Positive values (no offset needed)
println("\n1. Log normalization with positive values:")
println("-" ^ 60)
positive_data = [1.0, 10.0, 100.0, 1000.0, 10000.0]
println("Original data: ", positive_data)

stats_pos = compute_normalization_stats(positive_data; method=:log, clip_quantiles=nothing)
println("Offset: ", stats_pos.offset, " (no offset needed for positive values)")

normalized_pos = apply_normalization(positive_data, stats_pos)
println("Normalized (log-transformed): ", round.(normalized_pos, digits=4))

denormalized_pos = denormalize_labels(normalized_pos, stats_pos)
println("Denormalized (recovered): ", round.(denormalized_pos, digits=4))
println("Round-trip error: ", maximum(abs.(positive_data .- denormalized_pos)))

# Example 2: Data including negative values (offset required)
println("\n2. Log normalization with negative values:")
println("-" ^ 60)
negative_data = [-10.0, -5.0, 0.0, 5.0, 10.0, 20.0]
println("Original data: ", negative_data)

stats_neg = compute_normalization_stats(negative_data; method=:log, clip_quantiles=nothing)
println("Offset: ", stats_neg.offset, " (offset = |min| + 1 = |-10| + 1 = 11)")

normalized_neg = apply_normalization(negative_data, stats_neg)
println("Normalized (log-transformed): ", round.(normalized_neg, digits=4))

denormalized_neg = denormalize_labels(normalized_neg, stats_neg)
println("Denormalized (recovered): ", round.(denormalized_neg, digits=4))
println("Round-trip error: ", maximum(abs.(negative_data .- denormalized_neg)))

# Example 3: Skewed distribution (where log normalization shines)
println("\n3. Log normalization for highly skewed data:")
println("-" ^ 60)
skewed_data = [1.0, 2.0, 3.0, 5.0, 8.0, 15.0, 50.0, 500.0, 5000.0]
println("Original data (highly skewed): ", skewed_data)
println("Original range: ", maximum(skewed_data) - minimum(skewed_data))

# Compare with min-max normalization
stats_minmax = compute_normalization_stats(skewed_data; method=:minmax, range=(0,1), clip_quantiles=nothing)
normalized_minmax = apply_normalization(skewed_data, stats_minmax)
println("\nMin-max normalized: ", round.(normalized_minmax, digits=4))
println("Notice: Small values (1-15) are all compressed near 0")

# Now with log normalization
stats_log = compute_normalization_stats(skewed_data; method=:log, clip_quantiles=nothing)
normalized_log = apply_normalization(skewed_data, stats_log)
println("\nLog normalized: ", round.(normalized_log, digits=4))
println("Notice: More uniform distribution in log space")

denormalized_log = denormalize_labels(normalized_log, stats_log)
println("\nDenormalized (recovered): ", round.(denormalized_log, digits=4))
println("Round-trip error: ", maximum(abs.(skewed_data .- denormalized_log)))

# Example 4: Multi-column data with different modes
println("\n4. Matrix log normalization (columnwise):")
println("-" ^ 60)
matrix_data = [1.0 10.0; 2.0 100.0; 4.0 1000.0; 8.0 10000.0]
println("Original matrix:")
println(matrix_data)

stats_col = compute_normalization_stats(matrix_data; method=:log, mode=:columnwise, clip_quantiles=nothing)
println("\nOffsets per column: ", stats_col.offsets)

normalized_matrix = apply_normalization(matrix_data, stats_col)
println("\nNormalized matrix:")
println(round.(normalized_matrix, digits=4))

denormalized_matrix = denormalize_labels(normalized_matrix, stats_col)
println("\nDenormalized (recovered) matrix:")
println(round.(denormalized_matrix, digits=4))
println("Round-trip error: ", maximum(abs.(matrix_data .- denormalized_matrix)))

# Example 5: Comparison of all three methods
println("\n5. Comparison: MinMax vs ZScore vs Log:")
println("-" ^ 60)
comparison_data = [1.0, 5.0, 10.0, 50.0, 100.0, 1000.0]
println("Original data: ", comparison_data)

stats_mm = compute_normalization_stats(comparison_data; method=:minmax, range=(-1,1), clip_quantiles=nothing)
stats_zs = compute_normalization_stats(comparison_data; method=:zscore, clip_quantiles=nothing)
stats_lg = compute_normalization_stats(comparison_data; method=:log, clip_quantiles=nothing)

norm_mm = apply_normalization(comparison_data, stats_mm)
norm_zs = apply_normalization(comparison_data, stats_zs)
norm_lg = apply_normalization(comparison_data, stats_lg)

println("\nMinMax [-1,1]: ", round.(norm_mm, digits=4))
println("ZScore:        ", round.(norm_zs, digits=4))
println("Log:           ", round.(norm_lg, digits=4))

println("\n" * "=" ^ 80)
println("Key Takeaways:")
println("=" ^ 80)
println("✓ Log normalization is perfect for skewed/exponential distributions")
println("✓ Automatically handles negative values with offset")
println("✓ Compresses large values while preserving small value differences")
println("✓ Fully reversible (perfect round-trip)")
println("✓ Works with all modes: global, columnwise, rowwise")
println("=" ^ 80)
