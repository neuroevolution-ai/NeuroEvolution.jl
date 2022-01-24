using Test
using Flux
using CUDA
using DataStructures

include("../brains/feed_forward_nn.jl")


function kernel_test_brain_initialize(individuals, brains)

    initialize(brains, individuals)

    sync_threads()

    return

end

function kernel_test_brain_step(input_all, output_all, brains)

    threadID = threadIdx().x
    blockID = blockIdx().x

    offset_shared_memory = 0

    input = @cuDynamicSharedMem(Float32, brains.input_size, offset_shared_memory)
    offset_shared_memory += sizeof(input)

    output = @cuDynamicSharedMem(Float32, brains.output_size, offset_shared_memory)
    offset_shared_memory += sizeof(output)

    sync_threads()

    # Load inputs to shared memory
    if threadID <= brains.input_size
        input[threadID] = input_all[threadID, blockID]
    end

    sync_threads()

    step(brains, input, output, offset_shared_memory)

    sync_threads()

    # Load outputs from shared memory
    if threadID <= brains.output_size
        output_all[threadID, blockID] = output[threadID]
    end

    return
end

@testset "Feed-Forward Neural Network" begin

    config_brain = OrderedDict()
    config_brain["hidden_layer1_size"] = 32
    config_brain["hidden_layer2_size"] = 16

    number_inputs = 30
    number_outputs = 6
    number_individuals = 100

    number_time_steps = 200

    # Initialize brains
    brains = FeedForwardNN(config_brain, number_inputs, number_outputs, number_individuals)

    individual_size = get_individual_size(brains)

    individuals = randn(number_individuals, individual_size)
    individuals_gpu = CuArray(individuals)

    number_threads = get_required_threads(brains)

    @cuda threads = number_threads blocks = number_individuals kernel_test_brain_initialize(individuals_gpu, brains)

    CUDA.synchronize()

    h1_size = brains.hidden_layer1_size
    h2_size = brains.hidden_layer2_size

    W1 = zeros(h1_size, number_inputs, number_individuals)
    W2 = zeros(h2_size, h1_size, number_individuals)
    W3 = zeros(number_outputs, h2_size, number_individuals)
    b1 = zeros(h1_size, number_individuals)
    b2 = zeros(h2_size, number_individuals)
    b3 = zeros(number_outputs, number_individuals)

    for j = 1:number_individuals

        W1_size = length(W1[:, :, j])
        W2_size = length(W2[:, :, j])
        W3_size = length(W3[:, :, j])
        b1_size = length(b1[:, j])
        b2_size = length(b2[:, j])
        b3_size = length(b3[:, j])

        @test W1_size + W2_size + W3_size + b1_size + b2_size + b3_size == individual_size

        W1[:, :, j] = reshape(view(individuals, j, 1:W1_size), (h1_size, number_inputs))
        offset = W1_size

        b1[:, j] = reshape(view(individuals, j, 1+offset:b1_size+offset), b1_size)
        offset += b1_size

        W2[:, :, j] = reshape(view(individuals, j, 1+offset:W2_size+offset), (h2_size, h1_size))
        offset += W2_size

        b2[:, j] = reshape(view(individuals, j, 1+offset:b2_size+offset), b2_size)
        offset += b2_size

        W3[:, :, j] = reshape(view(individuals, j, 1+offset:W3_size+offset), (number_outputs, h2_size))
        offset += W3_size

        b3[:, j] = reshape(view(individuals, j, 1+offset:b3_size+offset), b3_size)
        offset += b3_size

        @test offset == individual_size

    end

    # Test brain initialization
    @test W1 ≈ Array(brains.W1) rtol = 0.00001
    @test W2 ≈ Array(brains.W2) rtol = 0.00001
    @test W3 ≈ Array(brains.W3) rtol = 0.00001
    @test b1 ≈ Array(brains.b1) rtol = 0.00001
    @test b2 ≈ Array(brains.b2) rtol = 0.00001
    @test b3 ≈ Array(brains.b3) rtol = 0.00001

    for i = 1:number_time_steps

        input = randn(number_inputs, number_individuals)
        input_gpu = CuArray(input)

        output_gpu = CuArray(zeros(number_outputs, number_individuals))

        shared_memory_size = get_memory_requirements(brains) + sizeof(Float32) * (brains.input_size + brains.output_size)

        CUDA.@cuda threads = number_threads blocks = number_individuals shmem = shared_memory_size kernel_test_brain_step(input_gpu, output_gpu, brains)
        CUDA.synchronize()

        for j = 1:number_individuals

            # Calculate output of ffnn
            output = map(tanh, W3[:, :, j] * map(tanh, W2[:, :, j] * map(tanh, W1[:, :, j] * input[:, j] + b1[:, j]) + b2[:, j]) + b3[:, j])

            # Verify output of ffnn using Flux
            hidden_layer1 = Dense(W1[:, :, j], b1[:, j], tanh)
            hidden_layer2 = Dense(W2[:, :, j], b2[:, j], tanh)
            output_layer = Dense(W3[:, :, j], b3[:, j], tanh)
            ffnn_flux = Chain(hidden_layer1, hidden_layer2, output_layer)

            @test output ≈ ffnn_flux(input[:, j]) rtol = 0.00001
            @test output ≈ Array(output_gpu[:, j]) rtol = 0.00001

        end

    end

end

println("Finished")
