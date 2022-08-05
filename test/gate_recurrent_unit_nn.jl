using Test
using Flux
using CUDA
using DataStructures

include("../brains/gated_recurrent_unit_nn.jl")

function kernel_test_brain_initialize(individuals, brains)

    threadID = threadIdx().x
    blockID = blockIdx().x

    initialize(threadID, blockID, brains, individuals)

    sync_threads()

    return

end

function kernel_test_brain_step(inputs, outputs, brains::GatedRecurrentUnitNN)
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

function cpu_gru_step(hidden_state, input, W, U, b, individual)
    #Reset Gate
    r = sigmoid.(W[1] * input + b[1] + U[1] * hidden_state[:, individual])

    #Update Gate
    u = sigmoid.(W[2] * input + b[2] + U[2] * hidden_state[:, individual])

    #Current Gate 
    c = tanh.(W[3] * input + b[3] + r .* (U[3] * hidden_state[:, individual]))

    hidden_state[:, individual] = u .* hidden_state[:, individual] + (1 .- u) .* c

    #Outputlayer
    output = tanh.(W[4] * hidden_state[:, individual] + b[4])

    return output
end

@testset "Gated Recurrent Unit Neural Network" begin

    config_brain = OrderedDict()
    config_brain["number_neurons"] = 10
    config_brain["number_inputs"] = 30
    config_brain["number_outputs"] = 6

    number_neurons = config_brain["number_neurons"]
    number_inputs = config_brain["number_inputs"]
    number_outputs = config_brain["number_outputs"]

    number_individuals = 100
    number_time_steps = 1000

    brains = GatedRecurrentUnitNN(
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

    W_r = zeros(number_neurons, number_inputs, number_individuals)
    W_u = zeros(number_neurons, number_inputs, number_individuals)
    W_c = zeros(number_neurons, number_inputs, number_individuals)

    U_r = zeros(number_neurons, number_neurons, number_individuals)
    U_u = zeros(number_neurons, number_neurons, number_individuals)
    U_c = zeros(number_neurons, number_neurons, number_individuals)

    b_r = zeros(number_neurons, number_individuals)
    b_u = zeros(number_neurons, number_individuals)
    b_c = zeros(number_neurons, number_individuals)

    hidden_states = zeros(Float32, (number_neurons, number_individuals))

    V = zeros(number_outputs, number_neurons, number_individuals)
    b_v = zeros(number_outputs, number_individuals)


    for j = 1:number_individuals

        W_size = length(W_r[:, :, j])
        U_size = length(U_r[:, :, j])
        b_size = length(b_r[:, j])
        V_size = length(V[:, :, j])
        b_v_size = length(b_v[:, j])

        @test 3 * W_size + 3 * U_size + 3 * b_size + V_size + b_v_size == individual_size

        W_r[:, :, j] = reshape(view(individuals, j, 1:W_size), (number_neurons, number_inputs))
        offset = W_size

        W_u[:, :, j] = reshape(view(individuals, j, offset+1:offset+W_size), (number_neurons, number_inputs))
        offset += W_size

        W_c[:, :, j] = reshape(view(individuals, j, offset+1:offset+W_size), (number_neurons, number_inputs))
        offset += W_size

        U_r[:, :, j] = reshape(view(individuals, j, offset+1:offset+U_size), (number_neurons, number_neurons))
        offset += U_size

        U_u[:, :, j] = reshape(view(individuals, j, offset+1:offset+U_size), (number_neurons, number_neurons))
        offset += U_size

        U_c[:, :, j] = reshape(view(individuals, j, offset+1:offset+U_size), (number_neurons, number_neurons))
        offset += U_size

        b_r[:, j] = reshape(view(individuals, j, offset+1:b_size+offset), b_size)
        offset += b_size

        b_u[:, j] = reshape(view(individuals, j, offset+1:b_size+offset), b_size)
        offset += b_size

        b_c[:, j] = reshape(view(individuals, j, offset+1:b_size+offset), b_size)
        offset += b_size

        V[:, :, j] = reshape(view(individuals, j, offset+1:V_size+offset), (number_outputs, number_neurons))
        offset += V_size

        b_v[:, j] = reshape(view(individuals, j, offset+1:b_v_size+offset), b_v_size)
        offset += b_v_size

        @test offset == individual_size
    end

    #Initialization tests between GPU & CPU
    #Testing weightmatrices
    @test W_r ≈ Array(brains.W_r) rtol = 0.00001
    @test W_u ≈ Array(brains.W_u) rtol = 0.00001
    @test W_c ≈ Array(brains.W_c) rtol = 0.00001

    @test U_r ≈ Array(brains.U_r) rtol = 0.00001
    @test U_u ≈ Array(brains.U_u) rtol = 0.00001
    @test U_c ≈ Array(brains.U_c) rtol = 0.00001

    #Testing biases
    @test b_r ≈ Array(brains.b_r) rtol = 0.00001
    @test b_u ≈ Array(brains.b_u) rtol = 0.00001
    @test b_c ≈ Array(brains.b_c) rtol = 0.00001

    #Testing weightmatrices & bias of output layer
    @test V ≈ Array(brains.V) rtol = 0.00001
    @test b_v ≈ Array(brains.b_v) rtol = 0.00001

    #------------------
    #Flux initialization
    #------------------

    #One flux-chain for each individual
    flux_gru = Vector{Chain}(undef, number_individuals)

    #No constructor for weight initialization available for GRU
    #Weights of GRU layer are initialized by accessing the GRU Cell struct
    #https://github.com/FluxML/Flux.jl/blob/master/src/layers/recurrent.jl#L354

    for j = 1:number_individuals
        #Initializing Flux GRU Layer for every individual
        flux_gru_layer = GRU(number_inputs, number_neurons)

        #In Flux, first "number_neurons"-rows of Wi are for reset gate weight matrix, next update gate and last for current gate
        #Same for Wh and bias
        flux_gru_layer.cell.Wi[1 : number_neurons, :] = W_r[:, :, j]
        flux_gru_layer.cell.Wi[number_neurons + 1 : 2 * number_neurons, :] = W_u[:, :, j]
        flux_gru_layer.cell.Wi[2 * number_neurons + 1 : 3 * number_neurons, :] = W_c[:, :, j]

        flux_gru_layer.cell.Wh[1 : number_neurons, :] = U_r[:, :, j]
        flux_gru_layer.cell.Wh[number_neurons + 1 : 2 *number_neurons, :] = U_u[:, :, j]
        flux_gru_layer.cell.Wh[2 * number_neurons + 1 : 3 * number_neurons, :] = U_c[:, :, j]

        flux_gru_layer.cell.b[1 : number_neurons] = b_r[:, j]
        flux_gru_layer.cell.b[number_neurons + 1 : 2 * number_neurons] = b_u[:, j]
        flux_gru_layer.cell.b[2 * number_neurons + 1 : 3 * number_neurons] = b_c[:, j]

        flux_gru_layer.cell.state0 .= hidden_states[:, j]

        #Testing initial parameters of flux GRU layer
        @test flux_gru_layer.cell.Wi ≈ [W_r[:, :, j]; W_u[:, :, j]; W_c[:, :, j]] rtol = 0.00001
        @test flux_gru_layer.cell.Wh ≈ [U_r[:, :, j]; U_u[:, :, j]; U_c[:, :, j]] rtol = 0.00001
        @test flux_gru_layer.cell.b ≈ [b_r[:, j]; b_u[:, j]; b_c[:, j]] rtol = 0.00001
        @test flux_gru_layer.cell.state0 ≈ hidden_states[:, j] rtol = 0.00001

        #Outputlayer directly initialized
        flux_output_layer = Dense(V[:, :, j], b_v[:, j], tanh)

        #Testing initial parameters of flux output layer
        @test flux_output_layer.weight ≈ V[:, :, j] rtol = 0.00001
        @test flux_output_layer.bias ≈ b_v[:, j] rtol = 0.00001

        #Combining Layers
        flux_gru[j] = Chain(flux_gru_layer, flux_output_layer)
    end


    #------------------------------------------------------------------------------------------------------------------------
    #Brain step tests
    #------------------------------------------------------------------------------------------------------------------------

    #Comparing outputs of the Gpu & Cpu implementations against Flux

    shared_memory_size =
        get_memory_requirements(brains) +
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
            #CPU step
            W = (W_r[:, :, j], W_u[:, :, j], W_c[:, :, j], V[:, :, j])
            U = (U_r[:, :, j], U_u[:, :, j], U_c[:, :, j])
            b = (b_r[:, j], b_u[:, j], b_c[:, j], b_v[:, j])

            output_cpu[:, j] = cpu_gru_step(hidden_states, input[:, j], W, U, b, j)

            #Flux step
            output_flux[:, j] = flux_gru[j](input[:, j])

            #Comparing Outputs in every timestep
            @test output_flux[:, j] ≈ output_cpu[:, j] rtol = 0.00001
            @test output_flux[:, j] ≈ Array(output_gpu[:, j]) rtol = 0.00001

            #Align states since they drift away over time
            flux_gru[j].layers[1].state .= Array(brains.hidden_state[:, j])
            hidden_states[:, j] .= Array(brains.hidden_state[:, j])
        end

    end


end
