# GPU Functor Tests
# Tests to verify that RealLabelNormalization functors work on CUDA GPUs

using CUDA
using RealLabelNormalization
using Test

@testset "GPU Functor Compatibility" begin
    
    if !CUDA.functional()
        @warn "CUDA is not functional, skipping GPU tests"
        @test_skip false
        return
    end
    
    println("\n✅ Testing on: ", CUDA.name(CUDA.device()), "\n")
    
    @testset "Simple Vector Functors on GPU" begin
        labels = Float32[1.0, 2.0, 3.0, 4.0, 5.0]
        
        # Min-Max
        stats_mm = compute_normalization_stats(labels; method=:minmax, clip_quantiles=nothing)
        normalized_mm = apply_normalization(labels, stats_mm)
        functor_mm = stats_mm[:scale_back_functor]
        
        @test isbitstype(typeof(functor_mm))
        
        normalized_gpu = CuArray(normalized_mm)
        denormalized_gpu = functor_mm.(normalized_gpu)
        denormalized_cpu_expected = denormalize_labels(normalized_mm, stats_mm)
        
        @test Array(denormalized_gpu) ≈ denormalized_cpu_expected atol=1e-5
        
        # Z-Score
        stats_z = compute_normalization_stats(labels; method=:zscore, clip_quantiles=nothing)
        normalized_z = apply_normalization(labels, stats_z)
        functor_z = stats_z[:scale_back_functor]
        
        @test isbitstype(typeof(functor_z))
        
        normalized_z_gpu = CuArray(normalized_z)
        denormalized_z_gpu = functor_z.(normalized_z_gpu)
        denormalized_z_cpu_expected = denormalize_labels(normalized_z, stats_z)
        
        @test Array(denormalized_z_gpu) ≈ denormalized_z_cpu_expected atol=1e-5
        
        # Log
        stats_log = compute_normalization_stats(labels; method=:log, clip_quantiles=nothing)
        normalized_log = apply_normalization(labels, stats_log)
        functor_log = stats_log[:scale_back_functor]
        
        @test isbitstype(typeof(functor_log))
        
        normalized_log_gpu = CuArray(normalized_log)
        denormalized_log_gpu = functor_log.(normalized_log_gpu)
        denormalized_log_cpu_expected = denormalize_labels(normalized_log, stats_log)
        
        @test Array(denormalized_log_gpu) ≈ denormalized_log_cpu_expected atol=1e-5
    end
    
    @testset "Columnwise Functors on GPU" begin
        labels_2d = Float32[1.0 10.0; 2.0 20.0; 3.0 30.0; 4.0 40.0; 5.0 50.0]
        stats = compute_normalization_stats(labels_2d; method=:minmax, mode=:columnwise, clip_quantiles=nothing)
        normalized = apply_normalization(labels_2d, stats)
        functor = stats[:scale_back_functor]
        
        @test isbitstype(typeof(functor))
        
        normalized_gpu = CuArray(normalized)
        denormalized_gpu = similar(normalized_gpu)
        
        # Extract and apply individual column functors
        for col in 1:size(normalized_gpu, 2)
            col_functor = functor.functors[col]
            @test isbitstype(typeof(col_functor))
            denormalized_gpu[:, col] .= col_functor.(normalized_gpu[:, col])
        end
        
        denormalized_cpu_expected = denormalize_labels(normalized, stats)
        @test Array(denormalized_gpu) ≈ denormalized_cpu_expected atol=1e-5
    end
    
    @testset "Rowwise Functors on GPU" begin
        labels_2d = Float32[1.0 2.0 3.0 4.0 5.0; 10.0 20.0 30.0 40.0 50.0; 100.0 200.0 300.0 400.0 500.0]
        stats = compute_normalization_stats(labels_2d; method=:zscore, mode=:rowwise, clip_quantiles=nothing)
        normalized = apply_normalization(labels_2d, stats)
        functor = stats[:scale_back_functor]
        
        @test isbitstype(typeof(functor))
        
        normalized_gpu = CuArray(normalized)
        denormalized_gpu = similar(normalized_gpu)
        
        # Extract and apply individual row functors
        for row in 1:size(normalized_gpu, 1)
            row_functor = functor.functors[row]
            @test isbitstype(typeof(row_functor))
            denormalized_gpu[row, :] .= row_functor.(normalized_gpu[row, :])
        end
        
        denormalized_cpu_expected = denormalize_labels(normalized, stats)
        @test Array(denormalized_gpu) ≈ denormalized_cpu_expected atol=1e-5
    end
    
    @testset "Custom CUDA Kernel with Functor" begin
        # Define custom CUDA kernel
        function denorm_kernel!(output, input, functor)
            idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
            if idx <= length(input)
                @inbounds output[idx] = functor(input[idx])
            end
            return nothing
        end
        
        labels = Float32[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        stats = compute_normalization_stats(labels; method=:minmax, clip_quantiles=nothing)
        normalized = apply_normalization(labels, stats)
        functor = stats[:scale_back_functor]
        
        normalized_gpu = CuArray(normalized)
        output_gpu = similar(normalized_gpu)
        
        threads = 256
        blocks = cld(length(normalized_gpu), threads)
        @cuda threads=threads blocks=blocks denorm_kernel!(output_gpu, normalized_gpu, functor)
        
        denormalized_cpu_expected = denormalize_labels(normalized, stats)
        @test Array(output_gpu) ≈ denormalized_cpu_expected atol=1e-5
    end
    
    @testset "Large Array Performance Test" begin
        # Test with larger array to verify GPU performance benefits
        n = 1_000_000
        labels = Float32.(1:n)
        stats = compute_normalization_stats(labels; method=:zscore, clip_quantiles=nothing)
        normalized = apply_normalization(labels, stats)
        functor = stats[:scale_back_functor]
        
        normalized_gpu = CuArray(normalized)
        
        # Warm-up
        denormalized_gpu = functor.(normalized_gpu)
        CUDA.synchronize()
        
        # Timed GPU operation
        gpu_time = @elapsed begin
            denormalized_gpu = functor.(normalized_gpu)
            CUDA.synchronize()
        end
        
        # Timed CPU operation
        cpu_time = @elapsed begin
            denormalized_cpu = functor.(normalized)
        end
        
        println("  GPU time: ", round(gpu_time * 1000, digits=3), " ms")
        println("  CPU time: ", round(cpu_time * 1000, digits=3), " ms")
        println("  Speedup: ", round(cpu_time / gpu_time, digits=2), "x")
        
        # Verify correctness
        denormalized_cpu_expected = denormalize_labels(normalized, stats)
        @test Array(denormalized_gpu) ≈ denormalized_cpu_expected atol=1e-4
    end
end

println("\n✅ All GPU tests completed successfully!")
