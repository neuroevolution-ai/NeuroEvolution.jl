using Test
using Flux
using CUDA
using DataStructures
using Random

include("../brains/long_short_term_memory_nn.jl")

#GGF gegen python testen mit "Pycall"

function kernel_test_brain_initialize(individuals, brains)

    threadID = threadIdx().x
    blockID = blockIdx().x


    initialize(threadID, blockID, brains, individuals)

    sync_threads()

    return

end

function kernel_test_brain_step(
    inputs,
    outputs,
    brains::LongShortTermMemoryNN,
    gate_results_all,
)
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
    sync_threads()

    return
end

function cpu_lstm_step(cell_state, hidden_state, input, W, U, b, individual)
    #Input Gate
    i = sigmoid.(W[1] * input + b[1] + U[1] * hidden_state[:, individual])

    #Forget Gate
    f = sigmoid.(W[2] * input + b[2] + U[2] * hidden_state[:, individual])

    #Cell Gate 
    c = tanh.(W[3] * input + b[3] + U[3] * hidden_state[:, individual])
    cell_state[:, individual] = f .* cell_state[:, individual] + i .* c

    #Output Gate
    o = sigmoid.(W[4] * input + b[4] + U[4] * hidden_state[:, individual])
    hidden_state[:, individual] = o .* tanh.(cell_state[:, individual])
end

@testset "Long Short Term Neural Network" begin

    number_neurons = 10
    number_inputs = 30
    number_outputs = 6
    number_individuals = 100

    number_time_steps = 200

    brains = LongShortTermMemoryNN(
        number_neurons,
        number_inputs,
        number_outputs,
        number_individuals,
    )

    individual_size = get_individual_size(brains)

    #Random.seed!(1)
    individuals = randn(number_individuals, individual_size)
    individuals_gpu = CuArray(individuals)

    number_threads = get_required_threads(brains)

    #------------------------------------------------------------------------------------------------------------------------
    #Brain initialization tests
    #------------------------------------------------------------------------------------------------------------------------

    #------------------
    #GPU initialization
    #------------------
    @cuda threads = number_threads blocks = number_individuals kernel_test_brain_initialize(
        individuals_gpu,
        brains,
    )

    CUDA.synchronize()

    #------------------
    #CPU initialization
    #------------------
    W_i = zeros(Float32, (number_neurons, number_inputs, number_individuals))
    W_i = zeros(Float32, (number_neurons, number_inputs, number_individuals))
    W_f = zeros(Float32, (number_neurons, number_inputs, number_individuals))
    W_o = zeros(Float32, (number_neurons, number_inputs, number_individuals))
    W_c = zeros(Float32, (number_neurons, number_inputs, number_individuals))

    U_i = zeros(Float32, (number_neurons, number_neurons, number_individuals))
    U_f = zeros(Float32, (number_neurons, number_neurons, number_individuals))
    U_o = zeros(Float32, (number_neurons, number_neurons, number_individuals))
    U_c = zeros(Float32, (number_neurons, number_neurons, number_individuals))

    b_i = zeros(Float32, (number_neurons, number_individuals))
    b_f = zeros(Float32, (number_neurons, number_individuals))
    b_o = zeros(Float32, (number_neurons, number_individuals))
    b_c = zeros(Float32, (number_neurons, number_individuals))

    hidden_states = zeros(Float32, (number_neurons, number_individuals))
    cell_states = zeros(Float32, (number_neurons, number_individuals))

    V = zeros(Float32, (number_outputs, number_neurons, number_individuals))
    b_v = zeros(Float32, (number_outputs, number_individuals))


    for j = 1:number_individuals

        W_size = length(W_i[:, :, j])
        U_size = length(U_i[:, :, j])
        b_size = length(b_i[:, j])
        V_size = length(V[:, :, j])
        b_v_size = length(b_v[:, j])

        @test 4 * W_size + 4 * U_size + 4 * b_size + V_size + b_v_size == individual_size

        W_i[:, :, j] =
            reshape(view(individuals, j, 1:W_size), (number_neurons, number_inputs))
        offset = W_size

        W_f[:, :, j] = reshape(
            view(individuals, j, offset+1:offset+W_size),
            (number_neurons, number_inputs),
        )
        offset += W_size

        W_o[:, :, j] = reshape(
            view(individuals, j, offset+1:offset+W_size),
            (number_neurons, number_inputs),
        )
        offset += W_size

        W_c[:, :, j] = reshape(
            view(individuals, j, offset+1:offset+W_size),
            (number_neurons, number_inputs),
        )
        offset += W_size

        U_i[:, :, j] = reshape(
            view(individuals, j, offset+1:offset+U_size),
            (number_neurons, number_neurons),
        )
        offset += U_size

        U_f[:, :, j] = reshape(
            view(individuals, j, offset+1:offset+U_size),
            (number_neurons, number_neurons),
        )
        offset += U_size

        U_o[:, :, j] = reshape(
            view(individuals, j, offset+1:offset+U_size),
            (number_neurons, number_neurons),
        )
        offset += U_size

        U_c[:, :, j] = reshape(
            view(individuals, j, offset+1:offset+U_size),
            (number_neurons, number_neurons),
        )
        offset += U_size

        b_i[:, j] = reshape(view(individuals, j, offset+1:b_size+offset), b_size)
        offset += b_size

        b_f[:, j] = reshape(view(individuals, j, offset+1:b_size+offset), b_size)
        offset += b_size

        b_o[:, j] = reshape(view(individuals, j, offset+1:b_size+offset), b_size)
        offset += b_size

        b_c[:, j] = reshape(view(individuals, j, offset+1:b_size+offset), b_size)
        offset += b_size

        V[:, :, j] = reshape(
            view(individuals, j, offset+1:V_size+offset),
            (number_outputs, number_neurons),
        )
        offset += V_size

        b_v[:, j] = reshape(view(individuals, j, offset+1:b_v_size+offset), b_v_size)
        offset += b_v_size

        @test offset == individual_size
    end

    #Testing weightmatrices of lstm layers between GPU and CPU
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

    #------------------
    #Flux initialization
    #------------------
    flux_lstm = Vector{Chain}(undef, number_individuals)

    for j = 1:number_individuals
        #Initializing Flux LSTM Layer for every individual
        flux_lstm_layer = LSTM(number_inputs, number_neurons)

        #No constructor for weight initialization available for LSTM
        #Weights of LSTM layer are initialized by accessing the LSTM Cell struct
        #https://github.com/FluxML/Flux.jl/blob/master/src/layers/recurrent.jl#L280
        flux_lstm_layer.cell.Wi[1:number_neurons, :] = W_i[:, :, j]
        flux_lstm_layer.cell.Wi[number_neurons+1:2*number_neurons, :] = W_f[:, :, j]
        flux_lstm_layer.cell.Wi[2*number_neurons+1:3*number_neurons, :] = W_c[:, :, j]
        flux_lstm_layer.cell.Wi[3*number_neurons+1:4*number_neurons, :] = W_o[:, :, j]

        flux_lstm_layer.cell.Wh[1:number_neurons, :] = U_i[:, :, j]
        flux_lstm_layer.cell.Wh[number_neurons+1:2*number_neurons, :] = U_f[:, :, j]
        flux_lstm_layer.cell.Wh[2*number_neurons+1:3*number_neurons, :] = U_c[:, :, j]
        flux_lstm_layer.cell.Wh[3*number_neurons+1:4*number_neurons, :] = U_o[:, :, j]

        flux_lstm_layer.cell.b[1:number_neurons] = b_i[:, j]
        flux_lstm_layer.cell.b[number_neurons+1:2*number_neurons] = b_f[:, j]
        flux_lstm_layer.cell.b[2*number_neurons+1:3*number_neurons] = b_c[:, j]
        flux_lstm_layer.cell.b[3*number_neurons+1:4*number_neurons] = b_o[:, j]

        flux_lstm_layer.cell.state0[1] .= hidden_states[:, j]

        #Testing initial parameters of flux LSTM layer
        @test flux_lstm_layer.cell.Wi ≈
              [W_i[:, :, j]; W_f[:, :, j]; W_c[:, :, j]; W_o[:, :, j]] rtol = 0.00001
        @test flux_lstm_layer.cell.Wh ≈
              [U_i[:, :, j]; U_f[:, :, j]; U_c[:, :, j]; U_o[:, :, j]] rtol = 0.00001
        @test flux_lstm_layer.cell.b ≈ [b_i[:, j]; b_f[:, j]; b_c[:, j]; b_o[:, j]] rtol =
            0.00001
        @test flux_lstm_layer.cell.state0[1] ≈ hidden_states[:, j] rtol = 0.00001
        @test flux_lstm_layer.cell.state0[2] ≈ cell_states[:, j] rtol = 0.00001

        #Outputlayer directly initialized
        flux_output_layer = Dense(V[:, :, j], b_v[:, j], tanh)

        #Testing initial parameters of flux output layer
        @test flux_output_layer.weight ≈ V[:, :, j] rtol = 0.00001
        @test flux_output_layer.bias ≈ b_v[:, j] rtol = 0.00001

        #Combining Layers and add to Array
        flux_lstm[j] = Chain(flux_lstm_layer, flux_output_layer)
    end



    #------------------------------------------------------------------------------------------------------------------------
    #Brain step tests
    #Comparing outputs of the Gpu against Cpu implementation and Flux
    #------------------------------------------------------------------------------------------------------------------------

    shared_memory_size =
        sizeof(Float32) * brains.number_inputs +
        sizeof(Float32) * brains.number_outputs +
        sizeof(Float32) * brains.number_neurons * 4

    #Testing for multiple time steps
    for i = 1:number_time_steps
        input = randn(Float32, number_inputs, number_individuals)
        input_gpu = CuArray(input)

        output = zeros(Float32, number_outputs, number_individuals)
        output_flux = zeros(number_outputs, number_individuals)
        output_gpu = CuArray(zeros(Float32, number_outputs, number_individuals))

        gate_results = CUDA.fill(0.0f0, (4, number_neurons, number_individuals))

        #GPU step
        @cuda threads = number_threads blocks = number_individuals shmem = shared_memory_size kernel_test_brain_step(input_gpu, output_gpu, brains, gate_results)
        CUDA.synchronize()

        for j = 1:number_individuals

            #CPU step
            W = (W_i[:, :, j], W_f[:, :, j], W_c[:, :, j], W_o[:, :, j])
            U = (U_i[:, :, j], U_f[:, :, j], U_c[:, :, j], U_o[:, :, j])
            b = (b_i[:, j], b_f[:, j], b_c[:, j], b_o[:, j])

            cpu_lstm_step(cell_states, hidden_states, input[:, j], W, U, b, j)
            output[:, j] = tanh.(V[:, :, j] * hidden_states[:, j] + b_v[:, j])

            #Flux step
            output_flux[:, j] = flux_lstm[j](input[:, j])
            
            #Comparing Outputs
            @test output_flux[:, j] ≈ output[:, j] rtol = 0.00001
            @test output_flux[:, j] ≈ Array(output_gpu[:, j]) rtol = 0.00001

            #Align states since they drift away over time
            flux_lstm[j].layers[1].state[1] .= Array(brains.hidden_state[:, j])
            flux_lstm[j].layers[1].state[2] .= Array(brains.cell_state[:, j])

            cell_states[:, j] .= Array(brains.cell_state[:, j])
            hidden_states[:, j] .= Array(brains.hidden_state[:, j])
        end

    end


end
