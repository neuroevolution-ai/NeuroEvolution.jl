using Test
using Flux
using CUDA
using DataStructures


@testset "Feed-Forward Neural Network" begin

    config_brain = OrderedDict()
    config_brain["hidden_layer1_size"] = 32
    config_brain["hidden_layer2_size"] = 16

    h1_size = config_brain["hidden_layer1_size"]
    h2_size = config_brain["hidden_layer2_size"]

    number_inputs = 30
    number_outputs = 6
    number_individuals = 100

    number_time_steps = 200

    # Initialize brains

    W1_size = h1_size * number_inputs
    W2_size = h2_size * h1_size
    W3_size = number_outputs * h2_size

    individual_size = W1_size + W2_size + W3_size + h1_size + h2_size + number_outputs

    individuals = randn(number_individuals, individual_size)
    individuals_gpu = CuArray(individuals)

    CUDA.synchronize()

    W1 = zeros(h1_size, number_inputs, number_individuals)
    W2 = zeros(h2_size, h1_size, number_individuals)
    W3 = zeros(number_outputs, h2_size, number_individuals)
    b1 = zeros(h1_size, number_individuals)
    b2 = zeros(h2_size, number_individuals)
    b3 = zeros(number_outputs, number_individuals)

    for j = 1:number_individuals

        W1[:, :, j] = reshape(view(individuals, j, 1:W1_size), (h1_size, number_inputs))
        offset = W1_size

        b1[:, j] = reshape(view(individuals, j, 1+offset:h1_size+offset), h1_size)
        offset += h1_size

        W2[:, :, j] = reshape(view(individuals, j, 1+offset:W2_size+offset), (h2_size, h1_size))
        offset += W2_size

        b2[:, j] = reshape(view(individuals, j, 1+offset:h2_size+offset), h2_size)
        offset += h2_size

        W3[:, :, j] = reshape(view(individuals, j, 1+offset:W3_size+offset), (number_outputs, h2_size))
        offset += W3_size

        b3[:, j] = reshape(view(individuals, j, 1+offset:number_outputs+offset), number_outputs)
        offset += number_outputs

        @test offset == individual_size

    end

    # Test brain initialization
    #@test W1 ≈ Array(brains.W1) atol = 0.0001
    #@test W2 ≈ Array(brains.W2) atol = 0.0001
    #@test W3 ≈ Array(brains.W3) atol = 0.0001
    #@test b1 ≈ Array(brains.b1) atol = 0.0001
    #@test b2 ≈ Array(brains.b2) atol = 0.0001
    #@test b3 ≈ Array(brains.b3) atol = 0.0001

    for i = 1:number_time_steps

        input = randn(number_inputs, number_individuals)
        input_gpu = CuArray(input)

        output = zeros(number_outputs, number_individuals)
        output_gpu = CuArray(output)

        #shared_memory_size = get_memory_requirements(brains) + sizeof(Float32) * (brains.input_size + brains.output_size)

        #CUDA.@cuda threads = brains.number_neurons blocks = number_individuals shmem = shared_memory_size kernel_test_brain_step(input_gpu, output_gpu, brains)
        #CUDA.synchronize()

        for j = 1:number_individuals

            # Calculate output of ffnn
            output = map(tanh, W3[:, :, j] * map(tanh, W2[:, :, j] * map(tanh, W1[:, :, j] * input[:, j] + b1[:, j]) + b2[:, j]) + b3[:, j])

            # Verify output of ffnn using Flux
            hidden_layer1 = Dense(W1[:, :, j], b1[:, j], tanh)
            hidden_layer2 = Dense(W2[:, :, j], b2[:, j], tanh)
            output_layer = Dense(W3[:, :, j], b3[:, j], tanh)
            ffnn_flux = Chain(hidden_layer1, hidden_layer2, output_layer)

            @test output ≈ ffnn_flux(input[:, j]) atol = 0.00001

            #@test x[:, j] ≈ Array(brains.x[:, j]) rtol = 0.1
            #@test y[:, j] ≈ Array(output_gpu[:, j]) rtol = 0.1

        end

    end

end

println("Finished")
