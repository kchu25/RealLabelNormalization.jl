using RealLabelNormalization
using Test


@testset "RealLabelNormalization Tests" begin
    
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

end

