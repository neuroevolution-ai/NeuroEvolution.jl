using Adapt
using LinearAlgebra

struct GatedRecurrentUnitNN{A,B,C,D,E}
    W_r::A
    W_u::A
    W_c::A
    U_r::B
    U_u::B
    U_c::B
    b_r::C
    b_u::C
    b_c::C
    hidden_state::C
    V::D
    b_v::E
    number_neurons::Int64
    number_inputs::Int64
    number_outputs::Int64
end

function GatedRecurrentUnitNN(configuration::OrderedDict, number_inputs::Int, number_outputs::Int, number_individuals::Int)

    GatedRecurrentUnitNN(
        CUDA.fill(0.0f0, (configuration["number_neurons"], number_inputs, number_individuals)),
        CUDA.fill(0.0f0, (configuration["number_neurons"], number_inputs, number_individuals)),
        CUDA.fill(0.0f0, (configuration["number_neurons"], number_inputs, number_individuals)),
        CUDA.fill(0.0f0, (configuration["number_neurons"], configuration["number_neurons"], number_individuals)),
        CUDA.fill(0.0f0, (configuration["number_neurons"], configuration["number_neurons"], number_individuals)),
        CUDA.fill(0.0f0, (configuration["number_neurons"], configuration["number_neurons"], number_individuals)),
        CUDA.fill(0.0f0, (configuration["number_neurons"], number_individuals)),
        CUDA.fill(0.0f0, (configuration["number_neurons"], number_individuals)),
        CUDA.fill(0.0f0, (configuration["number_neurons"], number_individuals)),
        CUDA.fill(0.0f0, (configuration["number_neurons"], number_individuals)),
        CUDA.fill(0.0f0, (number_outputs, configuration["number_neurons"], number_individuals)),
        CUDA.fill(0.0f0, (number_outputs, number_individuals)),
        configuration["number_neurons"],
        number_inputs,
        number_outputs,
    )
end

Adapt.@adapt_structure GatedRecurrentUnitNN

function get_required_threads(brains::GatedRecurrentUnitNN)
    return max(brains.number_neurons, brains.number_inputs, brains.number_outputs)
end

function get_memory_requirements(brains::GatedRecurrentUnitNN)
    return sizeof(Float32) * brains.number_neurons * 3
end

function initialize(brains::GatedRecurrentUnitNN, individuals)

    threadID = threadIdx().x
    blockID = blockIdx().x

    w_size = brains.number_inputs * brains.number_neurons
    u_size = brains.number_neurons * brains.number_neurons
    b_size = brains.number_neurons
    v_size = brains.number_neurons * brains.number_outputs

    offset = 0

    if threadID <= brains.number_neurons
        for i = 1:brains.number_inputs
            @inbounds brains.W_r[threadID, i, blockID] = individuals[blockID, threadID+(i-1)*brains.number_neurons+offset]
            @inbounds brains.W_u[threadID, i, blockID] = individuals[blockID, threadID+(i-1)*brains.number_neurons+1*w_size+offset]
            @inbounds brains.W_c[threadID, i, blockID] = individuals[blockID, threadID+(i-1)*brains.number_neurons+2*w_size+offset]
        end

        offset += 3 * w_size

        for i = 1:brains.number_neurons
            @inbounds brains.U_r[threadID, i, blockID] = individuals[blockID, threadID+(i-1)*brains.number_neurons+offset]
            @inbounds brains.U_u[threadID, i, blockID] = individuals[blockID, threadID+(i-1)*brains.number_neurons+1*u_size+offset]
            @inbounds brains.U_c[threadID, i, blockID] = individuals[blockID, threadID+(i-1)*brains.number_neurons+2*u_size+offset]
        end

        offset += 3 * u_size

        @inbounds brains.b_r[threadID, blockID] = individuals[blockID, threadID+offset]
        @inbounds brains.b_u[threadID, blockID] = individuals[blockID, threadID+1*brains.number_neurons+offset]
        @inbounds brains.b_c[threadID, blockID] = individuals[blockID, threadID+2*brains.number_neurons+offset]

        offset += 3 * b_size
    end

    if threadID <= brains.number_outputs
        for i = 1:brains.number_neurons
            @inbounds brains.V[threadID, i, blockID] = individuals[blockID, threadID+(i-1)*brains.number_outputs+offset]
        end

        offset += v_size
        @inbounds brains.b_v[threadID, blockID] = individuals[blockID, threadID+offset]
    end

    sync_threads()

    if threadID <= brains.number_neurons
        reset(brains)
    end

    sync_threads()

end

function reset(brains::GatedRecurrentUnitNN)

    threadID = threadIdx().x
    blockID = blockIdx().x

    if threadID <= brains.number_neurons
        @inbounds brains.hidden_state[threadID, blockID] = 0.0
    end
end

function step(brains::GatedRecurrentUnitNN, input, output, offset_memory)

    threadID = threadIdx().x
    blockID = blockIdx().x

    gate_results = @cuDynamicSharedMem(Float32, (3, brains.number_neurons), offset_memory)
    fill!(gate_results, 0.0f0)
    offset_memory += sizeof(gate_results)

    if threadID <= brains.number_neurons
        #Input calculation for gates
        for i = 1:brains.number_inputs
            #Reset Gate
            @inbounds gate_results[1, threadID] += brains.W_r[threadID, i, blockID] * input[i]
            #Update Gate
            @inbounds gate_results[2, threadID] += brains.W_u[threadID, i, blockID] * input[i]
        end

        #Hidden state calculation for gates 
        for i = 1:brains.number_neurons
            #Reset Gate
            @inbounds gate_results[1, threadID] += brains.U_r[threadID, i, blockID] * brains.hidden_state[i, blockID]
            #Update Gate
            @inbounds gate_results[2, threadID] += brains.U_u[threadID, i, blockID] * brains.hidden_state[i, blockID]
            #Current Gate
            @inbounds gate_results[3, threadID] += brains.U_c[threadID, i, blockID] * brains.hidden_state[i, blockID]
        end


        #Adding Biases
        @inbounds gate_results[1, threadID] += brains.b_r[threadID, blockID]
        @inbounds gate_results[2, threadID] += brains.b_u[threadID, blockID]

        #Applying activation functions
        gate_results[1, threadID] = sigmoid(gate_results[1, threadID])
        gate_results[2, threadID] = sigmoid(gate_results[2, threadID])

        #Current Gate calculations
        gate_results[3, threadID] *= gate_results[1, threadID]

        for i = 1:brains.number_inputs
            @inbounds gate_results[3, threadID] += brains.W_c[threadID, i, blockID] * input[i]
        end
        @inbounds gate_results[3, threadID] += brains.b_c[threadID, blockID]
        gate_results[3, threadID] = tanh(gate_results[3, threadID])

        #Hidden state calculation
        @inbounds brains.hidden_state[threadID, blockID] =
            gate_results[2, threadID] * brains.hidden_state[threadID, blockID] +
            (1 - gate_results[2, threadID]) * gate_results[3, threadID]
    end



    #Output Layer
    if threadID <= brains.number_outputs
        output[threadID] = 0.0
        for i = 1:brains.number_neurons
            @inbounds output[threadID] += brains.V[threadID, i, blockID] * brains.hidden_state[i, blockID]
        end
        @inbounds output[threadID] += brains.b_v[threadID, blockID]
        @inbounds output[threadID] = tanh(output[threadID])
    end

end

function get_individual_size(brains::GatedRecurrentUnitNN)
    return 3 * brains.number_inputs * brains.number_neurons +       #Input weights
           3 * brains.number_neurons * brains.number_neurons +         #hidden_state weights
           3 * brains.number_neurons +                                 #Biases
           brains.number_outputs * brains.number_neurons +             #Output layer weights
           brains.number_outputs                                       #Output bias
end

function sigmoid(x)
    return one(x) / (one(x) + exp(-x))
end
