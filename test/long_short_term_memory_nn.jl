using Test
using Flux
using CUDA
using DataStructures

include("../brains/long_short_term_memory_nn.jl")


function kernel_test_brain_initialize(individuals, brains)

    threadID = threadIdx().x
    blockID = blockIdx().x


    initialize(threadID, blockID, brains, individuals)

    sync_threads()

    return

end

function cpu_brain_step()
end    

@testset "Feed-Forward Neural Network" begin

    number_neurons = 10 
    number_inputs = 30
    number_outputs = 6
    number_individuals = 100

    number_time_steps = 200
    
    brains = LongShortTermMemoryNN(number_neurons, number_inputs, number_outputs, number_individuals)

    individual_size = get_individual_size(brains)

    individuals = randn(number_individuals, individual_size)
    individuals_gpu = CuArray(individuals)

    number_threads = get_required_threads(brains)

    #------------------------------------------------------------------------------------------------------------------------
    #Brain initialization tests
    #------------------------------------------------------------------------------------------------------------------------

    @cuda threads = number_threads blocks = number_individuals kernel_test_brain_initialize(individuals_gpu, brains)

    CUDA.synchronize()

    W_i = zeros(number_neurons, number_inputs, number_individuals)
    W_f = zeros(number_neurons, number_inputs, number_individuals)
    W_o = zeros(number_neurons, number_inputs, number_individuals)
    W_c = zeros(number_neurons, number_inputs, number_individuals)
    U_i = zeros(number_neurons, number_neurons, number_individuals)
    U_f = zeros(number_neurons, number_neurons, number_individuals)
    U_o = zeros(number_neurons, number_neurons, number_individuals)
    U_c = zeros(number_neurons, number_neurons, number_individuals)
    b_i = zeros(number_neurons, number_individuals)
    b_f = zeros(number_neurons, number_individuals)
    b_o = zeros(number_neurons, number_individuals)
    b_c = zeros(number_neurons, number_individuals)

    for j = 1:number_individuals

        W_size = length(W_i[:, :, j])
        U_size = length(U_i[:, :, j])
        b_size = length(b_i[:, j])

        @test 4 * W_size + 4* U_size + 4 * b_size == individual_size

        W_i[:, :, j] = reshape(view(individuals, j, 1 : W_size), (number_neurons, number_inputs))
        offset = W_size

        W_f[:, :, j] = reshape(view(individuals, j, offset + 1 : offset + W_size), (number_neurons, number_inputs))
        offset += W_size

        W_o[:, :, j] = reshape(view(individuals, j, offset + 1 : offset + W_size), (number_neurons, number_inputs))
        offset += W_size

        W_c[:, :, j] = reshape(view(individuals, j, offset + 1 : offset + W_size), (number_neurons, number_inputs))
        offset += W_size

        U_i[:, :, j] = reshape(view(individuals, j, offset + 1 : offset + U_size), (number_neurons, number_neurons))
        offset += U_size

        U_f[:, :, j] = reshape(view(individuals, j, offset + 1 : offset + U_size), (number_neurons, number_neurons))
        offset += U_size

        U_o[:, :, j] = reshape(view(individuals, j, offset + 1 : offset + U_size), (number_neurons, number_neurons))
        offset += U_size

        U_c[:, :, j] = reshape(view(individuals, j, offset + 1 : offset + U_size), (number_neurons, number_neurons))
        offset += U_size

        b_i[:, j] = reshape(view(individuals, j, offset + 1 : b_size + offset), b_size)
        offset += b_size

        b_f[:, j] = reshape(view(individuals, j, offset + 1 : b_size + offset), b_size)
        offset += b_size

        b_o[:, j] = reshape(view(individuals, j, offset + 1 : b_size + offset), b_size)
        offset += b_size
        
        b_c[:, j] = reshape(view(individuals, j, offset + 1 : b_size + offset), b_size)
        offset += b_size

        @test offset == individual_size
    end

    @test W_i ≈ Array(brains.W_i) rtol = 0.00001
    @test W_f ≈ Array(brains.W_f) rtol = 0.00001 
    @test W_o ≈ Array(brains.W_o) rtol = 0.00001 
    @test W_c ≈ Array(brains.W_c) rtol = 0.00001

    @test U_i[:,:,1] ≈ Array(brains.U_i[:,:,1]) rtol = 0.00001 
    @test U_f ≈ Array(brains.U_f) rtol = 0.00001 
    @test U_o ≈ Array(brains.U_o) rtol = 0.00001 
    @test U_c ≈ Array(brains.U_c) rtol = 0.00001 

    @test b_i ≈ Array(brains.b_i) rtol = 0.00001 
    @test b_f ≈ Array(brains.b_f) rtol = 0.00001 
    @test b_o ≈ Array(brains.b_o) rtol = 0.00001 
    @test b_c ≈ Array(brains.b_c) rtol = 0.00001 
    
    #------------------------------------------------------------------------------------------------------------------------
    #Brain step tests
    #------------------------------------------------------------------------------------------------------------------------

    hidden_states = zeros(number_neurons, number_individuals)
    
    for i = 1:number_time_steps

        input = randn(Float32, number_inputs, number_individuals)

        for j = 1:number_individuals
            #input Gate
            i = sigmoid.(W_i[:, :, j] * input[:, j] + b_i[:, j] + U_i[:, :, j] * hidden_states[:, j])

            #Forget Gate
            f = sigmoid.(W_f[:, :, j] * input[:, j] + b_f[:, j] + U_f[:, :, j] * hidden_states[:, j])

            #Cell Gate 
            c = tanh.(W_c[:, :, j] * input[:, j] + b_c[:, j] + U_c[:, :, j] * hidden_states[:, j])
            cell_state = f .* hidden_states[:, j] + i .* c

            #Output Gate
            o = sigmoid.(W_o[:, :, j] * input[:, j] + b_o[:, j] + U_o[:, :, j] * hidden_states[:, j])
            hidden_states[:, j] = o .* tanh.(cell_state)

            flux_lstm_layer = LSTM(number_inputs, number_neurons)
            flux_output_layer = Dense(number_neurons, number_outputs)
            flux_lstm = Chain(flux_lstm_layer, flux_output_layer)

            
            output_flux = flux_lstm(input[:, j])

        end

    end

end    