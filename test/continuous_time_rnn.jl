using Test
using CUDA
using Random
using JSON

include("../brains/continuous_time_rnn.jl")


function kernel_test_initialize(individuals, brains)

    tx = threadIdx().x
    bx = blockIdx().x

    initialize(tx, bx, individuals, brains)

    sync_threads()

    reset(tx, bx, brains)

    sync_threads()

    return

end

function kernel_test_brain_step(input, brains)

    tx = threadIdx().x
    bx = blockIdx().x

    step(tx, bx, input, brains)

    sync_threads()

    return
end

@testset "Brain Continuous Time RNN" begin

    for differential_eq in ["LiHoChow2005", "NaturalNet"]
        for set_principle_diagonal_elements_of_W_negative in [false, true]

            config_brain = OrderedDict()
            config_brain["delta_t"] = 0.05
            config_brain["number_neurons"] = 50
            config_brain["differential_equation"] = differential_eq
            config_brain["clipping_range_min"] = -1.0
            config_brain["clipping_range_max"] = 1.0
            config_brain["set_principle_diagonal_elements_of_W_negative"] = set_principle_diagonal_elements_of_W_negative
            config_brain["alpha"] = 0.1

            number_inputs = 30
            number_outputs = 15
            number_individuals = 100

            number_time_steps = 200

            # Initialize brains 
            brains = ContinuousTimeRNN(config_brain, number_inputs, number_outputs, number_individuals)

            individual_size = get_individual_size(brains)

            @test individual_size == number_inputs * brains.number_neurons + brains.number_neurons * brains.number_neurons + number_outputs * brains.number_neurons

            v_size = brains.number_neurons * brains.input_size
            w_size = brains.number_neurons * brains.number_neurons
            t_size = brains.output_size * brains.number_neurons

            individuals = rand(number_individuals, individual_size)
            individuals_gpu = CuArray(individuals)

            @cuda threads = brains.number_neurons blocks = number_individuals kernel_test_initialize(individuals_gpu, brains)

            CUDA.synchronize()

            V = zeros(brains.number_neurons, brains.input_size, number_individuals)
            W = zeros(brains.number_neurons, brains.number_neurons, number_individuals)
            T = zeros(brains.output_size, brains.number_neurons, number_individuals)

            for j = 1:number_individuals

                V[:, :, j] = reshape(view(individuals, j, 1:v_size), (brains.number_neurons, brains.input_size))
                W[:, :, j] = reshape(view(individuals, j, v_size+1:v_size+w_size), (brains.number_neurons, brains.number_neurons))
                T[:, :, j] = reshape(view(individuals, j, v_size+w_size+1:v_size+w_size+t_size), (brains.output_size, brains.number_neurons))

                if brains.set_principle_diagonal_elements_of_W_negative == true
                    for i = 1:brains.number_neurons
                        W[i, i, j] = -abs(W[i, i, j])
                    end
                end

            end

            # Test brain initialization
            @test V ≈ Array(brains.V) atol = 0.00001
            @test W ≈ Array(brains.W) atol = 0.00001
            @test T ≈ Array(brains.T) atol = 0.00001

            x = zeros(brains.number_neurons, number_individuals)

            for i = 1:number_time_steps

                input = randn(number_inputs, number_individuals)
                input_gpu = CuArray(input)

                shared_memory = get_memory_requirements(brains)

                CUDA.@cuda threads = brains.number_neurons blocks = number_individuals shmem = shared_memory kernel_test_brain_step(input_gpu, brains)
                CUDA.synchronize()

                for j = 1:number_individuals

                    # Differential Equation
                    if brains.differential_equation == NaturalNet
                        dx_dt = -brains.alpha * x[:, j] + W[:, :, j] * map(tanh, x[:, j]) + V[:, :, j] * input[:, j]
                    elseif brains.differential_equation == LiHoChow2005
                        dx_dt = -brains.alpha * x[:, j] + W[:, :, j] * map(tanh, (x[:, j] + V[:, :, j] * input[:, j]))
                    end

                    # Euler forward discretization
                    x[:, j] += brains.delta_t * dx_dt

                    # Clip x to state boundaries
                    x[:, j] = clamp.(x[:, j], brains.clipping_range_min, brains.clipping_range_max)

                    # Calculate outputs
                    y = map(tanh, T[:, :, j] * x)

                    @test x[:, j] ≈ Array(brains.x[:, j]) rtol = 0.001
                    @test y[:, j] ≈ Array(brains.y[:, j]) rtol = 0.001

                end

            end

        end
    end
end

println("Finished")
