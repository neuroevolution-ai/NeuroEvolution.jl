using Test
using Flux
using CUDA
using DataStructures
using Random

include("../brains/elman_network.jl")

function kernel_test_brain_initialize(individuals, brains)

    threadID = threadIdx().x
    blockID = blockIdx().x


    initialize(threadID, blockID, brains, individuals)

    sync_threads()

    return

end

function kernel_test_brain_step(inputs, outputs, brains::ElmanNetwork)
    threadID = threadIdx().x
    blockID = blockIdx().x

    offset_memory = 0

    input = @cuDynamicSharedMem(Float32, brains.number_inputs, offset_memory)
    offset_memory += sizeof(input)

    output = @cuDynamicSharedMem(Float32, brains.number_outputs, offset_memory)
    offset_memory += sizeof(output)

    sync_threads()

    if threadID <= brains.number_inputs
        input[threadID] = inputs[threadID, blockID]
    end

    sync_threads()

    step(threadID, blockID, brains, input, output, offset_memory)

    sync_threads()

    if threadID <= brains.number_outputs
        outputs[threadID, blockID] = output[threadID]
    end
    sync_threads()

    return
end

function cpu_elman_step(hidden_state, input, W, U, b, individual)
    
    #Hidden Layer
    hidden_state[:, individual] = tanh.(W[1] * input + b[1] + U * hidden_state[:, individual])

    #Output Layer
    output = tanh.(W[2] * hidden_state[:, individual] + b[2])

    return output
end

@testset "Elman Network" begin
    config_brain = OrderedDict()
    config_brain["number_neurons"] = 10
    config_brain["number_inputs"] = 30
    config_brain["number_outputs"] = 6

    number_neurons = config_brain["number_neurons"]
    number_inputs = config_brain["number_inputs"]
    number_outputs = config_brain["number_outputs"]

    number_individuals = 100
    number_time_steps = 1000

    brains = ElmanNetwork(
        config_brain,
        number_individuals,
    )

    individual_size = get_individual_size(brains)
    individuals = randn(number_individuals, individual_size)
    individuals_gpu = CuArray(individuals)

    number_threads = get_required_threads(brains)

    #------------------------------------------------------------------------------------------------------------------------
    #Brain initialization tests
    #------------------------------------------------------------------------------------------------------------------------

    #------------------
    #GPU initialization
    #------------------
    @cuda threads = number_threads blocks = number_individuals kernel_test_brain_initialize(individuals_gpu, brains)

    CUDA.synchronize()

    #------------------
    #CPU initialization
    #------------------
    W = zeros(Float32, (number_neurons, number_inputs, number_individuals))

    U = zeros(Float32, (number_neurons, number_neurons, number_individuals))

    b = zeros(Float32, (number_neurons, number_individuals))

    hidden_states = zeros(Float32, (number_neurons, number_individuals))

    V = zeros(Float32, (number_outputs, number_neurons, number_individuals))
    b_v = zeros(Float32, (number_outputs, number_individuals))

    for j = 1:number_individuals

        W_size = length(W[:, :, j])
        U_size = length(U[:, :, j])
        b_size = length(b[:, j])
        V_size = length(V[:, :, j])
        b_v_size = length(b_v[:, j])

        @test W_size + U_size + b_size + V_size + b_v_size == individual_size

        W[:, :, j] =reshape(view(individuals, j, 1:W_size), (number_neurons, number_inputs))
        offset = W_size

        U[:, :, j] = reshape(view(individuals, j, offset+1:offset+U_size), (number_neurons, number_neurons))
        offset += U_size

        b[:, j] = reshape(view(individuals, j, offset+1:b_size+offset), b_size)
        offset += b_size

        V[:, :, j] = reshape(view(individuals, j, offset+1:V_size+offset), (number_outputs, number_neurons))
        offset += V_size

        b_v[:, j] = reshape(view(individuals, j, offset+1:b_v_size+offset), b_v_size)
        offset += b_v_size

        @test offset == individual_size
    end

    #Testing weightmatrices of lstm layers between GPU and CPU
    @test W ≈ Array(brains.W) rtol = 0.00001

    @test U ≈ Array(brains.U) rtol = 0.00001

    #Testing biases
    @test b ≈ Array(brains.b) rtol = 0.00001

    #Testing weightmatrices & bias of output layer
    @test V ≈ Array(brains.V) rtol = 0.00001
    @test b_v ≈ Array(brains.b_v) rtol = 0.00001

    #------------------
    #Flux initialization
    #------------------
    flux_elman = Vector{Chain}(undef, number_individuals)

    for j = 1:number_individuals
        #Initializing Flux RNN Layer for every individual
        flux_rnn_layer = RNN(number_inputs, number_neurons)

        #No constructor for weight initialization available for RNN
        #Weights of RNN layer are initialized by accessing the RNN Cell struct
        #https://github.com/FluxML/Flux.jl/blob/master/src/layers/recurrent.jl#L192
        flux_rnn_layer.cell.Wi[1:number_neurons, :] = W[:, :, j]

        flux_rnn_layer.cell.Wh[1:number_neurons, :] = U[:, :, j]

        flux_rnn_layer.cell.b[1:number_neurons] = b[:, j]

        flux_rnn_layer.cell.state0 .= hidden_states[:, j]

        #Testing initial parameters of flux LSTM layer
        @test flux_rnn_layer.cell.Wi ≈ W[:, :, j] rtol = 0.00001
        @test flux_rnn_layer.cell.Wh ≈ U[:, :, j] rtol = 0.00001
        @test flux_rnn_layer.cell.b ≈ b[:, j] rtol = 0.00001
        @test flux_rnn_layer.cell.state0 ≈ hidden_states[:, j] rtol = 0.00001

        #Outputlayer directly initialized
        flux_output_layer = Dense(V[:, :, j], b_v[:, j], tanh)

        #Testing initial parameters of flux output layer
        @test flux_output_layer.weight ≈ V[:, :, j] rtol = 0.00001
        @test flux_output_layer.bias ≈ b_v[:, j] rtol = 0.00001

        #Combining Layers and add to Array
        flux_elman[j] = Chain(flux_rnn_layer, flux_output_layer)
    end

    #------------------------------------------------------------------------------------------------------------------------
    #Brain step tests
    #------------------------------------------------------------------------------------------------------------------------

    #Comparing outputs of the Gpu & Cpu implementations against Flux

    input = randn(Float32, number_inputs, number_individuals)
    input_gpu = CuArray(input)

    shared_memory_size = get_memory_requirements(brains) + 
        sizeof(Float32) * brains.number_inputs +
        sizeof(Float32) * brains.number_outputs    

    #Testing multiple timesteps
    for i = 1:number_time_steps

        input = randn(Float32, brains.number_inputs, number_individuals)
        input_gpu = CuArray(input)

        output_cpu = zeros(number_outputs, number_individuals)
        output_gpu = CuArray(output_cpu)
        output_flux = zeros(number_outputs, number_individuals)

        #GPU step
        @cuda threads = number_threads blocks = number_individuals shmem = shared_memory_size kernel_test_brain_step(input_gpu, output_gpu, brains)
        CUDA.synchronize()

        for j = 1:number_individuals
            #CPU Brain step
            W_cpu = (W[:, :, j], V[:, :, j])
            U_cpu = U[:, :, j]
            b_cpu = (b[:, j], b_v[:, j])

            output_cpu[:, j] = cpu_elman_step(hidden_states, input[:, j], W_cpu, U_cpu, b_cpu, j)
            
            #Flux Brain step
            output_flux[:, j] = flux_elman[j](input[:, j])

            #Comparing Outputs in every timestep
            @test output_flux[:, j] ≈ output_cpu[:, j] rtol = 0.00001
            @test output_flux[:, j] ≈ Array(output_gpu[:, j]) rtol = 0.00001

            #Align states since they drift away over time
            flux_elman[j].layers[1].state .= Array(brains.hidden_state[:, j])
            hidden_states[:, j] .= Array(brains.hidden_state[:, j])
        end 

    end    

end    