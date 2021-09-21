using Test

include("../optimizers/cma_es.jl")
include("optimizers/cma_es_deap.jl")
include("optimizers/compare_optimizer_states.jl")

number_generations = 100
population_size = 200
sigma = 1.5
free_parameters = 1000

tolerance = 0.00001

@testset "Optimizers" begin

    optimizer_configuration = Dict("sigma" => sigma, "population_size" => population_size)

    # Initialize Optimizers
    # OptimizerCmaEsDeap: Original Deap CMA-ES optimizer implemented in Python using PyCall
    # OptimizerCmaEs: CMA-ES optimizer implemented in Julia that we use in our framework
    optimizer1, eigenvectors1, indx1 = OptimizerCmaEsDeap(free_parameters, optimizer_configuration)
    optimizer2, eigenvectors2, indx2, diagD = OptimizerCmaEs(free_parameters, optimizer_configuration, test = true, eigenvectors1 = eigenvectors1, indx1 = indx1)

    # Compare return values
    @test size(indx1) == size(indx2)
    @test diagD[indx1] ≈ diagD[indx2] atol = 0.00001
    @test size(eigenvectors1) == size(eigenvectors2)

    # Compare internal states of both optimizers
    compare_optimizer_states(optimizer1, optimizer2, tolerance)

    for generation = 1:number_generations

        # Ask optimizers for new population
        genomes1, randoms1 = ask(optimizer1)
        genomes2, randoms2 = ask(optimizer2, test = true, randoms1 = randoms1)

        # Compare return values
        @test genomes1 ≈ genomes2 atol = tolerance
        @test size(randoms1) == size(randoms2)
        @test mean(randoms1) ≈ mean(randoms2) atol = 0.1
        @test std(randoms1) ≈ std(randoms2) atol = 0.01

        # Generate random rewards
        rewards_training = rand(population_size)

        # Tell optimizers new rewards
        eigenvectors1, indx1 = tell(optimizer1, rewards_training)
        eigenvectors2, indx2, diagD = tell(optimizer2, rewards_training, test = true, eigenvectors1 = eigenvectors1, indx1 = indx1)

        # Compare return values
        @test size(indx1) == size(indx2)
        @test diagD[indx1] ≈ diagD[indx2] atol = 0.00001
        @test size(eigenvectors1) == size(eigenvectors2)

        # Compare internal states of both optimizers
        compare_optimizer_states(optimizer1, optimizer2, tolerance)

        # Test if C is a Hermitian matrix
        @test optimizer2.C ≈ Hermitian(optimizer2.C) atol = tolerance

    end
end

println("Finished")
