using Test

function compare_optimizer_states(optimizer1::OptimizerCmaEsDeap, optimizer2::OptimizerCmaEs, tolerance)

    @test optimizer1.lambda_ ≈ optimizer2.lambda_ atol = tolerance
    @test optimizer1.dim ≈ optimizer2.dim atol = tolerance
    @test optimizer1.chiN ≈ optimizer2.chiN atol = tolerance
    @test optimizer1.mu ≈ optimizer2.mu atol = tolerance
    @test optimizer1.weights ≈ optimizer2.weights atol = tolerance
    @test optimizer1.mueff ≈ optimizer2.mueff atol = tolerance
    @test optimizer1.cc ≈ optimizer2.cc atol = tolerance
    @test optimizer1.cs ≈ optimizer2.cs atol = tolerance
    @test optimizer1.ps ≈ optimizer2.ps atol = tolerance
    @test optimizer1.pc ≈ optimizer2.pc atol = tolerance
    @test optimizer1.centroid ≈ optimizer2.centroid atol = tolerance
    @test optimizer1.update_count ≈ optimizer2.update_count atol = tolerance
    @test optimizer1.ccov1 ≈ optimizer2.ccov1 atol = tolerance
    @test optimizer1.ccovmu ≈ optimizer2.ccovmu atol = tolerance
    @test optimizer1.C ≈ optimizer2.C atol = tolerance
    @test optimizer1.sigma ≈ optimizer2.sigma atol = tolerance
    @test optimizer1.damps ≈ optimizer2.damps atol = tolerance
    @test optimizer1.diagD ≈ optimizer2.diagD atol = tolerance
    @test optimizer1.B ≈ optimizer2.B atol = tolerance
    @test optimizer1.BD ≈ optimizer2.BD atol = tolerance
    @test optimizer1.genomes ≈ optimizer2.genomes atol = tolerance

end
