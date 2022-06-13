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

function kernel_test_brain_step(inputs, outputs, brains::LongShortTermMemoryNN, gate_results_all)
    threadID = threadIdx().x
    blockID = blockIdx().x

    offset_memory = 0

    input = @cuDynamicSharedMem(Float32, brains.number_inputs, offset_memory)
    offset_memory += sizeof(input)

    output = @cuDynamicSharedMem(Float32, brains.number_outputs, offset_memory)
    offset_memory += sizeof(output)

    gate_results = @cuDynamicSharedMem(Float32, (4, brains.number_neurons), offset_memory)
    offset_memory += sizeof(gate_results)

    sync_threads()

    if threadID <= brains.number_inputs
        input[threadID] = inputs[threadID, blockID]
    end

    sync_threads()

    step(threadID, blockID, brains, gate_results, input, output)

    sync_threads()

    if threadID <= brains.number_outputs
        outputs[threadID, blockID] = output[threadID]
    end
    sync_threads()

    if threadID <= brains.number_neurons
        for i = 1:4
            gate_results_all[i, threadID, blockID] = gate_results[i, threadID]
        end
    end

    return
end    

@testset "Long Short Term Neural Network" begin

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

    V = zeros(number_outputs, number_neurons, number_individuals)
    b_v = zeros(number_outputs, number_individuals)


    for j = 1:number_individuals

        W_size = length(W_i[:, :, j])
        U_size = length(U_i[:, :, j])
        b_size = length(b_i[:, j])
        V_size = length(V[:, :, j])
        b_v_size = length(b_v[:, j])

        @test 4 * W_size + 4* U_size + 4 * b_size + V_size + b_v_size == individual_size

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

        V[:, :, j] = reshape(view(individuals, j, offset + 1 : V_size + offset), (number_outputs, number_neurons))
        offset += V_size

        b_v[:, j] = reshape(view(individuals, j, offset + 1 : b_v_size + offset), b_v_size)
        offset += b_v_size

        @test offset == individual_size
    end
    
    #Testing weightmatrices of lstm layer
    @test W_i ≈ Array(brains.W_i) rtol = 0.00001
    @test W_f ≈ Array(brains.W_f) rtol = 0.00001 
    @test W_o ≈ Array(brains.W_o) rtol = 0.00001 
    @test W_c ≈ Array(brains.W_c) rtol = 0.00001

    @test U_i ≈ Array(brains.U_i) rtol = 0.00001 
    @test U_f ≈ Array(brains.U_f) rtol = 0.00001 
    @test U_o ≈ Array(brains.U_o) rtol = 0.00001 
    @test U_c ≈ Array(brains.U_c) rtol = 0.00001 

    #Testing biases
    @test b_i ≈ Array(brains.b_i) rtol = 0.00001 
    @test b_f ≈ Array(brains.b_f) rtol = 0.00001 
    @test b_o ≈ Array(brains.b_o) rtol = 0.00001 
    @test b_c ≈ Array(brains.b_c) rtol = 0.00001

    #Testing weightmatrices & bias of output layer
    @test V ≈ Array(brains.V) rtol = 0.00001
    @test b_v ≈ Array(brains.b_v) rtol = 0.00001
    
    
    #------------------------------------------------------------------------------------------------------------------------
    #Brain step tests
    #------------------------------------------------------------------------------------------------------------------------
    #Comparing outputs of the Gpu & Cpu implementations against Flux


    hidden_states = zeros(Float32, (number_neurons, number_individuals))
    cell_states = zeros(Float32, (number_neurons, number_individuals))
    
    output = zeros(number_outputs, number_individuals)
    output_gpu = CuArray(output)
    output_flux = zeros(number_outputs, number_individuals)

    gate_results = CUDA.fill(0.0f0, (4, number_neurons, number_individuals))

    input = randn(Float32, number_inputs, number_individuals)
    input_gpu = CuArray(input)

    shared_memory_size = sizeof(Float32) * brains.number_inputs +
        sizeof(Float32) * brains.number_outputs +
        sizeof(Float32) * brains.number_neurons * 4
    
    #GPU Brain step
    @cuda threads = number_threads blocks = number_individuals shmem = shared_memory_size kernel_test_brain_step(input_gpu, output_gpu, brains, gate_results)

    for j = 1:number_individuals
        #Flux LSTM layer weight initialization
        flux_lstm_cell = Flux.LSTMCell(number_inputs, number_neurons)

        flux_lstm_cell.Wi[1:number_neurons, :] = W_i[:, :, j] 
        flux_lstm_cell.Wi[number_neurons + 1 : 2 * number_neurons, :] = W_f[:, :, j] 
        flux_lstm_cell.Wi[2 * number_neurons + 1 : 3 * number_neurons, :] = W_c[:, :, j]
        flux_lstm_cell.Wi[3 * number_neurons + 1 : 4 * number_neurons, :] = W_o[:, :, j]

        flux_lstm_cell.Wh[1:number_neurons, :] = U_i[:, :, j] 
        flux_lstm_cell.Wh[number_neurons + 1 : 2 * number_neurons, :] = U_f[:, :, j] 
        flux_lstm_cell.Wh[2 * number_neurons + 1 : 3 * number_neurons, :] = U_c[:, :, j]
        flux_lstm_cell.Wh[3 * number_neurons + 1 : 4 * number_neurons, :] = U_o[:, :, j]

        flux_lstm_cell.b[1 : number_neurons] = b_i[:, j]
        flux_lstm_cell.b[number_neurons + 1 : 2 * number_neurons] = b_f[:, j]
        flux_lstm_cell.b[2 * number_neurons + 1 : 3 * number_neurons] = b_c[:, j]
        flux_lstm_cell.b[3 * number_neurons + 1 : 4 * number_neurons] = b_o[:, j]

        flux_lstm_cell.state0[1] .= hidden_states[:, j]

        #Testing initial parameters
        @test flux_lstm_cell.Wi ≈ [W_i[:, :, j]; W_f[:, :, j]; W_c[:, :, j]; W_o[:, :, j]] rtol = 0.00001
        @test flux_lstm_cell.Wh ≈ [U_i[:, :, j]; U_f[:, :, j]; U_c[:, :, j]; U_o[:, :, j]] rtol = 0.00001
        @test flux_lstm_cell.b ≈ [b_i[:, j]; b_f[:, j]; b_c[:, j]; b_o[:, j]] rtol = 0.00001
        @test flux_lstm_cell.state0[1] ≈ hidden_states[:, j] rtol = 0.00001
        @test flux_lstm_cell.state0[2] ≈ cell_states[:, j] rtol = 0.00001    
        
        #Cpu lstm calculation
        #input Gate
        i = sigmoid.(W_i[:, :, j] * input[:, j] + b_i[:, j] + U_i[:, :, j] * hidden_states[:, j])

        #Forget Gate
        f = sigmoid.(W_f[:, :, j] * input[:, j] + b_f[:, j] + U_f[:, :, j] * hidden_states[:, j])

        #Cell Gate 
        c = tanh.(W_c[:, :, j] * input[:, j] + b_c[:, j] + U_c[:, :, j] * hidden_states[:, j])
        cell_states[:, j] = f .* cell_states[:, j] + i .* c

        #Output Gate
        o = sigmoid.(W_o[:, :, j] * input[:, j] + b_o[:, j] + U_o[:, :, j] * hidden_states[:, j])
        hidden_states[:, j] = o .* tanh.(cell_states[:, j])

        #Output Layer 
        output[:, j] = tanh.(V[:, :, j] * hidden_states[:, j] + b_v[:, j])

        #Building Flux LSTM 
        flux_lstm_layer = Flux.Recur(flux_lstm_cell)

        #Testing cpu outputs of lstm layer against flux
        #@test flux_lstm_layer(input[:, j]) ≈ hidden_states[:, j] rtol = 0.00001
        
        flux_output_layer = Dense(V[:, :, j], b_v[:, j], tanh)
        flux_lstm = Chain(flux_lstm_layer, flux_output_layer)

        output_flux[:, j] = flux_lstm(input[:, j])

        #Testing cpu output of output layer against flux
        @test output_flux[:, j] ≈ output[:, j] rtol = 0.00001

        #Testing GPU output against Flux
        @test Array(output_gpu[:, j]) ≈ output_flux[:, j] rtol = 0.00001
    end

end    