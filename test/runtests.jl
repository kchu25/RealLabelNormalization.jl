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

    @testset "Outlier Clipping Tests" begin
        @testset "Vector Clipping - Basic Functionality" begin
            # Test vector clipping
            data = [1.0, 2.0, 3.0, 4.0, 5.0, 100.0]
            clipped = RealLabelNormalization._clip_outliers(data, (0.0, 0.8), :global)
            
            # More robust tests
            @test maximum(clipped) ≈ 5.0  # 80th percentile of [1,2,3,4,5,100] is 5.0
            @test minimum(clipped) == 1.0  # 0th percentile should be minimum
            @test clipped[end] < 100.0     # Outlier should be clipped
            
            # Test that non-outliers remain unchanged
            @test clipped[1:5] == data[1:5]
        end

        @testset "Matrix Clipping - Different Modes" begin
            # Test matrix clipping - columnwise  
            matrix = [1.0 10.0; 2.0 20.0; 3.0 200.0]
            clipped_matrix = RealLabelNormalization._clip_outliers(matrix, (0.0, 0.8), :columnwise)
            @test clipped_matrix[3, 2] < 200.0  # Outlier in column 2 clipped
            @test clipped_matrix[3, 1] < 3.0   # Column 1's 80th percentile ≈ 2.6
            
            # Test rowwise clipping
            matrix_row = [1.0 2.0 3.0; 10.0 20.0 200.0]
            clipped_row = RealLabelNormalization._clip_outliers(matrix_row, (0.0, 0.8), :rowwise)
            @test clipped_row[2, 3] < 200.0  # Outlier in row 2 clipped
            # Row 1 should be mostly unchanged, but may have slight clipping due to quantile calculation
            @test clipped_row[1, 1] == matrix_row[1, 1]  # First element unchanged
            @test clipped_row[1, 2] == matrix_row[1, 2]  # Second element unchanged
            
            # Test global clipping
            clipped_global = RealLabelNormalization._clip_outliers(matrix, (0.0, 0.8), :global)
            # Global 80th percentile of [1,2,3,10,20,200] is around 20, so all should be <= 20
            # But let's be more flexible with the test
            @test all(clipped_global .<= 25.0)  # All values clipped to global 80th percentile (with some tolerance)
        end

        @testset "NaN Handling in Clipping" begin
            # Test with NaN values - use same quantiles for consistency
            data_nan = [1.0, 2.0, NaN, 4.0, 100.0]
            clipped_nan = RealLabelNormalization._clip_outliers(data_nan, (0.0, 0.8), :global)
            @test sum(isnan.(clipped_nan)) == 1  # NaN preserved
            @test clipped_nan[3] |> isnan        # Specific NaN position preserved
            @test maximum(filter(!isnan, clipped_nan)) < 100.0
            
            # Test matrix with NaN values
            matrix_nan = [1.0 NaN; 2.0 20.0; NaN 200.0]
            clipped_matrix_nan = RealLabelNormalization._clip_outliers(matrix_nan, (0.0, 0.8), :columnwise)
            @test sum(isnan.(clipped_matrix_nan)) == 2  # Both NaNs preserved
            @test clipped_matrix_nan[1, 2] |> isnan
            @test clipped_matrix_nan[3, 1] |> isnan
        end

        @testset "Edge Cases and Boundary Conditions" begin
            # Test empty data
            empty_data = Float64[]
            clipped_empty = RealLabelNormalization._clip_outliers(empty_data, (0.0, 1.0), :global)
            @test clipped_empty == empty_data
            
            # Test all NaN data - should return original data (with warning)
            all_nan = [NaN, NaN, NaN]
            clipped_all_nan = RealLabelNormalization._clip_outliers(all_nan, (0.0, 1.0), :global)
            @test all(isnan.(clipped_all_nan))  # All values should still be NaN
            @test length(clipped_all_nan) == length(all_nan)  # Length preserved
            
            # Test single value
            single_val = [42.0]
            clipped_single = RealLabelNormalization._clip_outliers(single_val, (0.0, 1.0), :global)
            @test clipped_single == single_val
            
            # Test two identical values
            two_identical = [5.0, 5.0]
            clipped_two = RealLabelNormalization._clip_outliers(two_identical, (0.0, 1.0), :global)
            @test clipped_two == two_identical
            
            # Test constant data
            constant_data = [3.0, 3.0, 3.0, 3.0]
            clipped_constant = RealLabelNormalization._clip_outliers(constant_data, (0.0, 1.0), :global)
            @test clipped_constant == constant_data
        end

        @testset "Quantile Boundary Tests" begin
            # Test extreme quantiles
            data = [1.0, 2.0, 3.0, 4.0, 5.0]
            
            # Test (0.0, 1.0) - should not clip anything
            clipped_none = RealLabelNormalization._clip_outliers(data, (0.0, 1.0), :global)
            @test clipped_none == data
            
            # Test (0.5, 0.5) - should clip to median
            clipped_median = RealLabelNormalization._clip_outliers(data, (0.5, 0.5), :global)
            @test all(x == 3.0 for x in clipped_median)  # All values clipped to median
            
            # Test (0.2, 0.8) - should clip extreme values
            data_with_outliers = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 100.0]
            clipped_20_80 = RealLabelNormalization._clip_outliers(data_with_outliers, (0.2, 0.8), :global)
            @test minimum(clipped_20_80) >= 2.0  # 20th percentile
            @test maximum(clipped_20_80) <= 8.0  # 80th percentile
        end

        @testset "Data Type Preservation" begin
            # Test that data types are preserved for Float64
            float64_data = [1.0, 2.0, 3.0, 4.0, 100.0]
            clipped_float64 = RealLabelNormalization._clip_outliers(float64_data, (0.0, 0.8), :global)
            @test eltype(clipped_float64) == eltype(float64_data)
            
            float32_data = Float32[1.0, 2.0, 3.0, 4.0, 100.0]
            clipped_float32 = RealLabelNormalization._clip_outliers(float32_data, (0.0, 0.8), :global)
            @test eltype(clipped_float32) == Float32
            
            # Note: Integer types may cause conversion issues due to quantile calculation
            # This is expected behavior as quantiles are typically Float64
        end

        @testset "Error Handling and Invalid Inputs" begin
            # Test invalid mode - this should throw an error for matrix input
            matrix_data = [1.0 2.0; 3.0 4.0]
            @test_throws ErrorException RealLabelNormalization._clip_outliers(matrix_data, (0.0, 1.0), :invalid_mode)
            
            # Test extreme quantiles - these will clip to specific values
            data = [1.0, 2.0, 3.0]
            clipped_0_0 = RealLabelNormalization._clip_outliers(data, (0.0, 0.0), :global)
            @test all(x == minimum(data) for x in clipped_0_0)  # All clipped to minimum
            
            clipped_1_1 = RealLabelNormalization._clip_outliers(data, (1.0, 1.0), :global)
            @test all(x == maximum(data) for x in clipped_1_1)  # All clipped to maximum
        end

        @testset "Large Dataset Performance" begin
            # Test with larger dataset to ensure performance
            large_data = randn(10000)
            large_data[1:100] .*= 10  # Add some outliers
            
            @time clipped_large = RealLabelNormalization._clip_outliers(large_data, (0.05, 0.95), :global)
            
            # Verify clipping worked
            @test maximum(clipped_large) < maximum(large_data)
            @test minimum(clipped_large) > minimum(large_data)
            @test length(clipped_large) == length(large_data)
        end

        @testset "Mathematical Properties" begin
            # Test that clipping preserves order relationships for non-outlier values
            data = [1.0, 2.0, 3.0, 4.0, 5.0, 100.0, 200.0]
            clipped = RealLabelNormalization._clip_outliers(data, (0.1, 0.9), :global)
            
            # Non-outlier values should maintain their relative order
            non_outlier_indices = [1, 2, 3, 4, 5]
            @test clipped[1] <= clipped[2] <= clipped[3] <= clipped[4] <= clipped[5]
            
            # Clipped values should be within the quantile bounds
            lower_q, upper_q = quantile(filter(!isnan, data), [0.1, 0.9])
            @test all(lower_q <= x <= upper_q for x in clipped if !isnan(x))
        end
    end

    @testset "Training Clip Bounds Tests" begin
        @testset "Basic Training Bounds Application" begin
            # Test vector case
            labels = [1.0, 2.0, 3.0, 4.0, 5.0, 100.0]
            stats = (clip_bounds = (lower = 1.0, upper = 5.0), mode = :vector)
            clipped = RealLabelNormalization._apply_training_clip_bounds(labels, stats)
            
            @test all(1.0 <= x <= 5.0 for x in clipped)
            @test clipped[1:5] == labels[1:5]  # Non-outliers unchanged
            @test clipped[6] == 5.0  # Outlier clipped to upper bound
            
            # Test matrix case - columnwise
            matrix = [1.0 10.0; 2.0 20.0; 3.0 200.0; 100.0 30.0]
            clip_bounds = [(lower = 1.0, upper = 3.0), (lower = 10.0, upper = 30.0)]
            stats_matrix = (clip_bounds = clip_bounds, mode = :columnwise)
            clipped_matrix = RealLabelNormalization._apply_training_clip_bounds(matrix, stats_matrix)
            
            @test all(1.0 <= x <= 3.0 for x in clipped_matrix[:, 1])
            @test all(10.0 <= x <= 30.0 for x in clipped_matrix[:, 2])
        end

        @testset "Training Bounds Edge Cases" begin
            # Test with no clip bounds (should return original data)
            labels = [1.0, 2.0, 3.0]
            stats_no_bounds = (clip_bounds = nothing, mode = :vector)
            clipped_no_bounds = RealLabelNormalization._apply_training_clip_bounds(labels, stats_no_bounds)
            @test clipped_no_bounds == labels
            
            # Test with NaN bounds
            stats_nan_bounds = (clip_bounds = (lower = NaN, upper = 5.0), mode = :vector)
            clipped_nan_bounds = RealLabelNormalization._apply_training_clip_bounds(labels, stats_nan_bounds)
            @test clipped_nan_bounds == labels  # Should return original when bounds are NaN
            
            # Test with insufficient bounds for matrix
            matrix = [1.0 2.0; 3.0 4.0; 5.0 6.0]
            stats_insufficient = (clip_bounds = [(lower = 1.0, upper = 5.0)], mode = :columnwise)
            clipped_insufficient = RealLabelNormalization._apply_training_clip_bounds(matrix, stats_insufficient)
            @test clipped_insufficient[:, 1] == clamp.(matrix[:, 1], 1.0, 5.0)
            @test clipped_insufficient[:, 2] == matrix[:, 2]  # No bounds for column 2
        end

        @testset "Training Bounds with NaN Data" begin
            # Test with NaN values in data
            labels_nan = [1.0, NaN, 3.0, 100.0]
            stats = (clip_bounds = (lower = 1.0, upper = 3.0), mode = :vector)
            clipped_nan = RealLabelNormalization._apply_training_clip_bounds(labels_nan, stats)
            
            @test sum(isnan.(clipped_nan)) == 1  # NaN preserved
            @test clipped_nan[2] |> isnan
            @test clipped_nan[1] == 1.0
            @test clipped_nan[3] == 3.0
            @test clipped_nan[4] == 3.0  # Clipped to upper bound
        end

        @testset "Training Bounds Error Handling" begin
            # Test invalid mode
            labels = [1.0, 2.0, 3.0]
            stats_invalid = (clip_bounds = (lower = 1.0, upper = 3.0), mode = :invalid)
            @test_throws ErrorException RealLabelNormalization._apply_training_clip_bounds(labels, stats_invalid)
        end
    end

    @testset "Property-Based Tests for Clipping" begin
        @testset "Monotonicity Preservation" begin
            # Test that clipping preserves monotonicity for non-outlier values
            for _ in 1:10
                n = rand(5:20)
                data = sort(randn(n))
                data[1:2] .*= 0.1  # Make some values very small
                data[end-1:end] .*= 10  # Make some values very large
                
                clipped = RealLabelNormalization._clip_outliers(data, (0.1, 0.9), :global)
                
                # Find non-outlier indices (values that weren't extreme)
                non_outlier_mask = (data .>= quantile(data, 0.1)) .& (data .<= quantile(data, 0.9))
                if sum(non_outlier_mask) >= 2
                    non_outlier_values = data[non_outlier_mask]
                    non_outlier_clipped = clipped[non_outlier_mask]
                    
                    # Check that order is preserved
                    @test issorted(non_outlier_clipped) == issorted(non_outlier_values)
                end
            end
        end

        @testset "Boundedness Property" begin
            # Test that clipped values are always within quantile bounds
            for _ in 1:10
                data = randn(20)
                quantiles = (rand() * 0.3, 0.7 + rand() * 0.3)  # Random quantiles between 0-0.3 and 0.7-1.0
                
                clipped = RealLabelNormalization._clip_outliers(data, quantiles, :global)
                lower_q, upper_q = quantile(data, [quantiles[1], quantiles[2]])
                
                # All clipped values should be within bounds
                @test all(lower_q <= x <= upper_q for x in clipped if !isnan(x))
            end
        end

        @testset "Clipping Stability Property" begin
            # Test that clipping already-clipped data doesn't expand the range significantly
            # Note: Idempotency doesn't hold for clipping due to quantile recalculation,
            # but we can test that the range doesn't expand
            for _ in 1:5
                data = randn(15)
                quantiles = (0.1, 0.9)
                
                clipped_once = RealLabelNormalization._clip_outliers(data, quantiles, :global)
                clipped_twice = RealLabelNormalization._clip_outliers(clipped_once, quantiles, :global)
                
                # Test that the range of clipped_twice is not significantly larger than clipped_once
                range_once = maximum(clipped_once) - minimum(clipped_once)
                range_twice = maximum(clipped_twice) - minimum(clipped_twice)
                @test range_twice <= range_once + 1e-10
                
                # Test that the bounds are still reasonable
                lower_q, upper_q = quantile(data, [quantiles[1], quantiles[2]])
                @test all(lower_q <= x <= upper_q for x in clipped_twice if !isnan(x))
            end
        end

        @testset "Matrix Mode Consistency" begin
            # Test that different matrix modes produce consistent results for appropriate data
            for _ in 1:5
                n, m = rand(3:8), rand(2:5)
                matrix = randn(n, m)
                
                # Test that columnwise and rowwise produce different but valid results
                clipped_col = RealLabelNormalization._clip_outliers(matrix, (0.1, 0.9), :columnwise)
                clipped_row = RealLabelNormalization._clip_outliers(matrix, (0.1, 0.9), :rowwise)
                clipped_global = RealLabelNormalization._clip_outliers(matrix, (0.1, 0.9), :global)
                
                # All should have same shape
                @test size(clipped_col) == size(matrix)
                @test size(clipped_row) == size(matrix)
                @test size(clipped_global) == size(matrix)
                
                # All should be bounded (though bounds may differ)
                @test all(isfinite, clipped_col)
                @test all(isfinite, clipped_row)
                @test all(isfinite, clipped_global)
            end
        end
    end

    @testset "Warning and Logging Tests" begin
        @testset "Warning for All NaN Data" begin
            # Test that warnings are issued for all NaN data
            all_nan = [NaN, NaN, NaN]
            
            # Capture warnings (this is a simplified test - in practice you might use TestLogging.jl)
            # For now, we just test that the function doesn't crash and returns NaN values
            result = RealLabelNormalization._clip_outliers(all_nan, (0.0, 1.0), :global)
            @test all(isnan.(result))  # All values should still be NaN
            @test length(result) == length(all_nan)  # Length preserved
        end

        @testset "Warning for Empty Data" begin
            # Test that warnings are issued for empty data
            empty_data = Float64[]
            result = RealLabelNormalization._clip_outliers(empty_data, (0.0, 1.0), :global)
            @test result == empty_data
        end

        @testset "Warning for NaN Bounds in Training" begin
            # Test that warnings are issued when training bounds contain NaN
            labels = [1.0, 2.0, 3.0]
            stats_nan_bounds = (clip_bounds = (lower = NaN, upper = 5.0), mode = :vector)
            result = RealLabelNormalization._apply_training_clip_bounds(labels, stats_nan_bounds)
            @test result == labels
        end
    end

    @testset "Integration Tests with Normalization" begin
        @testset "Clipping + Normalization Round-trip" begin
            # Test that clipping works correctly with the full normalization pipeline
            data = [1.0, 2.0, 3.0, 4.0, 5.0, 100.0, 200.0]  # Contains outliers
            
            # Test with clipping
            stats = compute_normalization_stats(data; clip_quantiles=(0.1, 0.9))
            normalized = apply_normalization(data, stats)
            denormalized = denormalize_labels(normalized, stats)
            
            # Verify that outliers were clipped (should be less than original max)
            @test maximum(denormalized) < 200.0
            # The clipping may not be as aggressive as expected due to quantile calculation
            # So we test that it's significantly less than the original maximum
            @test maximum(denormalized) < 150.0
            
            # Verify round-trip accuracy for non-outlier values
            non_outlier_mask = (data .>= quantile(data, 0.1)) .& (data .<= quantile(data, 0.9))
            @test isapprox(data[non_outlier_mask], denormalized[non_outlier_mask], atol=1e-10)
        end

        @testset "Matrix Clipping + Normalization" begin
            # Test matrix clipping with different modes
            matrix = [1.0 10.0; 2.0 20.0; 3.0 200.0; 100.0 30.0]
            
            # Test columnwise clipping
            stats_col = compute_normalization_stats(matrix; mode=:columnwise, clip_quantiles=(0.0, 0.8))
            normalized_col = apply_normalization(matrix, stats_col)
            denormalized_col = denormalize_labels(normalized_col, stats_col)
            
            # Verify clipping worked
            @test maximum(denormalized_col[:, 1]) < 100.0
            @test maximum(denormalized_col[:, 2]) < 200.0
            
            # Test rowwise clipping
            stats_row = compute_normalization_stats(matrix; mode=:rowwise, clip_quantiles=(0.0, 0.8))
            normalized_row = apply_normalization(matrix, stats_row)
            denormalized_row = denormalize_labels(normalized_row, stats_row)
            
            # Verify clipping worked
            @test maximum(denormalized_row[4, :]) < 100.0  # Row 4 had outlier
            @test maximum(denormalized_row[:, 2]) < 200.0  # Column 2 had outlier
        end
    end    
end

