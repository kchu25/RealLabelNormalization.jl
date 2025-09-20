using RealLabelNormalization
using Test
using Statistics


@testset "RealLabelNormalization Tests" begin
    
    @testset "Ground Truth Verification - Min-Max Normalization" begin
        # Test with known values and expected results
        labels = [1.0, 2.0, 3.0, 4.0, 5.0]  # min=1, max=5, range=4
        
        # Manual calculation for [-1, 1] range:
        # normalized = 2 * (x - min) / (max - min) - 1
        # For x=1: 2 * (1-1) / 4 - 1 = -1
        # For x=3: 2 * (3-1) / 4 - 1 = 0  
        # For x=5: 2 * (5-1) / 4 - 1 = 1
        expected_minmax = [-1.0, -0.5, 0.0, 0.5, 1.0]
        
        stats = compute_normalization_stats(labels; clip_quantiles=nothing)
        normalized = apply_normalization(labels, stats)
        
        @test isapprox(normalized, expected_minmax, atol=1e-10)
        
        # Test custom range [0, 10]
        stats_custom = compute_normalization_stats(labels; range=(0, 10), clip_quantiles=nothing)
        normalized_custom = apply_normalization(labels, stats_custom)
        # For range [0, 10]: normalized = 10 * (x - min) / (max - min)
        expected_custom = [0.0, 2.5, 5.0, 7.5, 10.0]
        @test isapprox(normalized_custom, expected_custom, atol=1e-10)
    end
    
    @testset "Ground Truth Verification - Z-Score Normalization" begin
        # Test with known statistical properties
        labels = [1.0, 2.0, 3.0, 4.0, 5.0]  # mean=3, std=sqrt(2.5)≈1.58
        
        # Manual calculation: z = (x - mean) / std
        expected_mean = 3.0
        expected_std = sqrt(2.5)  # ≈ 1.5811
        expected_zscore = [(x - expected_mean) / expected_std for x in labels]
        # Should be approximately: [-1.265, -0.632, 0.0, 0.632, 1.265]
        
        stats = compute_normalization_stats(labels; method=:zscore, clip_quantiles=nothing)
        normalized = apply_normalization(labels, stats)
        
        @test isapprox(normalized, expected_zscore, atol=1e-10)
        
        # Verify normalized data has mean≈0, std≈1
        @test abs(Statistics.mean(normalized)) < 1e-10
        @test abs(Statistics.std(normalized) - 1.0) < 1e-10
    end
    
    @testset "Ground Truth Verification - Quantile Clipping" begin
        # Test with known outliers - validate that clipping works, not exact values
        labels = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 100.0]  # 100 is outlier
        
        # Test without clipping
        stats_no_clip = compute_normalization_stats(labels; clip_quantiles=nothing)
        normalized_no_clip = apply_normalization(labels, stats_no_clip)
        denormalized_no_clip = denormalize_labels(normalized_no_clip, stats_no_clip)
        
        # Test with clipping
        stats_with_clip = compute_normalization_stats(labels; clip_quantiles=(0.1, 0.9))
        normalized_with_clip = apply_normalization(labels, stats_with_clip)
        denormalized_with_clip = denormalize_labels(normalized_with_clip, stats_with_clip)
        
        # Key properties to verify:
        # 1. Clipping should reduce the range of the denormalized data
        @test (maximum(denormalized_with_clip) - minimum(denormalized_with_clip)) < 
              (maximum(denormalized_no_clip) - minimum(denormalized_no_clip))
        
        # 2. The outlier should be clipped (last value should not be 100)
        @test denormalized_with_clip[end] < 95.0  # Much less than the original outlier
        
        # 3. Most middle values should remain relatively unchanged  
        @test isapprox(denormalized_with_clip[3:5], labels[3:5], rtol=0.1)
    end
    
    @testset "Ground Truth Verification - Multi-target Column-wise" begin
        # Test matrix with different scales per column
        matrix_labels = [1.0 10.0;   # Column 1: 1-7 (range=6), Column 2: 10-40 (range=30)
                        3.0 20.0; 
                        5.0 30.0;
                        7.0 40.0]
        
        # Expected normalization for column 1 (range 1-7 -> [-1,1]):
        # col1_normalized = 2 * (x - 1) / 6 - 1
        expected_col1 = [-1.0, -1/3, 1/3, 1.0]
        
        # Expected normalization for column 2 (range 10-40 -> [-1,1]):
        # col2_normalized = 2 * (x - 10) / 30 - 1  
        expected_col2 = [-1.0, -1/3, 1/3, 1.0]
        
        stats = compute_normalization_stats(matrix_labels; mode=:columnwise, clip_quantiles=nothing)
        normalized = apply_normalization(matrix_labels, stats)
        
        @test isapprox(normalized[:, 1], expected_col1, atol=1e-10)
        @test isapprox(normalized[:, 2], expected_col2, atol=1e-10)
    end
    
    @testset "Basic Normalization Round-trip" begin
        # Test perfect round-trip without clipping
        labels = [1.0, 2.0, 3.0, 4.0, 5.0]
        stats = compute_normalization_stats(labels; clip_quantiles=(0.0, 1.0))
        normalized = apply_normalization(labels, stats)
        denormalized = denormalize_labels(normalized, stats)
        
        # Test normalization properties
        @test all(-1 ≤ x ≤ 1 for x in normalized)  # Default range [-1, 1]
        @test isapprox(labels, denormalized, atol=1e-10)  # Perfect round-trip
        
        # Test min/max are at boundaries
        @test minimum(normalized) ≈ -1.0
        @test maximum(normalized) ≈ 1.0
    end
    
    @testset "NaN Handling" begin
        # Test with some NaN values (essential edge case)
        labels_with_nan = [1.0, NaN, 3.0, 4.0, NaN, 5.0]
        stats = compute_normalization_stats(labels_with_nan; clip_quantiles=(0.0, 1.0))
        normalized = apply_normalization(labels_with_nan, stats)
        denormalized = denormalize_labels(normalized, stats)
        
        # Essential tests for NaN handling
        @test sum(isnan.(normalized)) == 2  # NaNs preserved in output
        @test sum(isnan.(denormalized)) == 2  # NaNs preserved after round-trip
        
        # Valid values should still normalize correctly  
        valid_mask = .!isnan.(labels_with_nan)
        @test all(-1 ≤ x ≤ 1 for x in normalized[valid_mask])  # Valid values in range
        @test isapprox(labels_with_nan[valid_mask], denormalized[valid_mask], atol=1e-10)  # Round-trip works for valid values
    end
    
    @testset "Multi-output Labels" begin
        # Test matrix labels (multi-target prediction)
        matrix_labels = [1.0 2.0; 3.0 4.0; 5.0 6.0; 7.0 8.0]  # 4 samples, 2 targets each
        
        stats = compute_normalization_stats(matrix_labels; clip_quantiles=(0.0, 1.0))
        normalized = apply_normalization(matrix_labels, stats)
        denormalized = denormalize_labels(normalized, stats)
        
        # Test shapes preserved
        @test size(normalized) == size(matrix_labels)
        @test size(denormalized) == size(matrix_labels)
        
        # Test normalization range
        @test all(-1 ≤ x ≤ 1 for x in normalized)
        
        # Test round-trip accuracy
        @test isapprox(matrix_labels, denormalized, atol=1e-10)
        
        # Test matrix with NaN values
        matrix_with_nan = [1.0 NaN; 3.0 4.0; NaN 6.0]
        matrix_stats = compute_normalization_stats(matrix_with_nan; clip_quantiles=(0.0, 1.0))
        matrix_normalized = apply_normalization(matrix_with_nan, matrix_stats)
        
        @test sum(isnan.(matrix_normalized)) == 2  # NaN count preserved
        @test size(matrix_normalized) == size(matrix_with_nan)  # Shape preserved
    end
    
    @testset "Ground Truth Verification - Edge Cases" begin
        # Test constant values (zero variance)
        constant_labels = [5.0, 5.0, 5.0, 5.0]
        
        # Min-max normalization of constant values should map to middle of range
        # For range [-1, 1], middle is 0
        stats_constant = compute_normalization_stats(constant_labels; clip_quantiles=nothing)
        normalized_constant = apply_normalization(constant_labels, stats_constant)
        @test all(x ≈ 0.0 for x in normalized_constant)  # All values should be 0
        
        # Z-score normalization of constant values should be 0 (since std=0)
        stats_zscore_constant = compute_normalization_stats(constant_labels; method=:zscore, clip_quantiles=nothing)
        normalized_zscore_constant = apply_normalization(constant_labels, stats_zscore_constant)
        @test all(x ≈ 0.0 for x in normalized_zscore_constant)
        
        # Test single value
        single_value = [3.14]
        stats_single = compute_normalization_stats(single_value; clip_quantiles=nothing)
        normalized_single = apply_normalization(single_value, stats_single)
        @test normalized_single[1] ≈ 0.0  # Should map to middle of range
        
        # Test two identical values
        two_identical = [2.0, 2.0]
        stats_two = compute_normalization_stats(two_identical; clip_quantiles=nothing)
        normalized_two = apply_normalization(two_identical, stats_two)
        @test all(x ≈ 0.0 for x in normalized_two)
    end
    
    @testset "Row-wise Normalization" begin
        # Each row is normalized independently
        mat = [1.0 2.0 3.0; 10.0 20.0 30.0; -1.0 0.0 1.0]
        # Min-max: each row should map to [-1, 0, 1]
        expected_minmax = [-1.0 0.0 1.0; -1.0 0.0 1.0; -1.0 0.0 1.0]
        stats_row = compute_normalization_stats(mat; mode=:rowwise, clip_quantiles=nothing)
        normalized_row = apply_normalization(mat, stats_row)
        @test isapprox(normalized_row, expected_minmax, atol=1e-10)
        # Round-trip
        denorm_row = denormalize_labels(normalized_row, stats_row)
        @test isapprox(mat, denorm_row, atol=1e-10)
        # Z-score: each row mean≈0, std≈1
        stats_row_z = compute_normalization_stats(mat; mode=:rowwise, method=:zscore, clip_quantiles=nothing)
        normalized_row_z = apply_normalization(mat, stats_row_z)
        for i in 1:size(mat, 1)
            row = normalized_row_z[i, :]
            @test abs(mean(row)) < 1e-10
            @test abs(std(row) - 1.0) < 1e-10
        end
        # NaN handling: NaNs preserved, valid values normalized
        mat_nan = [1.0 NaN 3.0; 10.0 20.0 NaN]
        stats_nan = compute_normalization_stats(mat_nan; mode=:rowwise, clip_quantiles=nothing)
        normalized_nan = apply_normalization(mat_nan, stats_nan)
        @test sum(isnan.(normalized_nan)) == 2
        # Valid values in each row are in [-1, 1]
        for i in 1:size(mat_nan, 1)
            valid = .!isnan.(mat_nan[i, :])
            @test all(-1 ≤ x ≤ 1 for x in normalized_nan[i, :][valid])
        end
    end

end

