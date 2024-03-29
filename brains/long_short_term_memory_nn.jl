using Adapt
using LinearAlgebra

struct LongShortTermMemoryNN{A, B, C, D, E}
    W_i::A
    W_f::A
    W_o::A
    W_c::A
    U_i::B
    U_f::B
    U_o::B
    U_c::B
    b_i::C
    b_f::C
    b_o::C
    b_c::C
    hidden_state::C
    cell_state::C
    V::D
    b_v::E
    number_neurons::Int64
    number_inputs::Int64
    number_outputs::Int64
end

function LongShortTermMemoryNN(configuration::OrderedDict, number_inputs::Int, number_outputs::Int, number_individuals::Int)

    LongShortTermMemoryNN(
        CUDA.fill(0.0f0, (configuration["number_neurons"], number_inputs, number_individuals)),
        CUDA.fill(0.0f0, (configuration["number_neurons"], number_inputs, number_individuals)),
        CUDA.fill(0.0f0, (configuration["number_neurons"], number_inputs, number_individuals)),
        CUDA.fill(0.0f0, (configuration["number_neurons"], number_inputs, number_individuals)),
        CUDA.fill(0.0f0, (configuration["number_neurons"], configuration["number_neurons"], number_individuals)),
        CUDA.fill(0.0f0, (configuration["number_neurons"], configuration["number_neurons"], number_individuals)),
        CUDA.fill(0.0f0, (configuration["number_neurons"], configuration["number_neurons"], number_individuals)),
        CUDA.fill(0.0f0, (configuration["number_neurons"], configuration["number_neurons"], number_individuals)),
        CUDA.fill(0.0f0, (configuration["number_neurons"], number_individuals)),
        CUDA.fill(0.0f0, (configuration["number_neurons"], number_individuals)),
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

Adapt.@adapt_structure LongShortTermMemoryNN

function get_required_threads(brains::LongShortTermMemoryNN)
    return max(brains.number_neurons, brains.number_inputs, brains.number_outputs)
end

function get_memory_requirements(brains::LongShortTermMemoryNN)
    return sizeof(Float32) * brains.number_neurons * 4
end

function initialize(brains::LongShortTermMemoryNN, individuals)

    threadID = threadIdx().x
    blockID = blockIdx().x

    w_size = brains.number_inputs * brains.number_neurons
    u_size = brains.number_neurons * brains.number_neurons
    b_size = brains.number_neurons
    v_size = brains.number_neurons * brains.number_outputs

    offset = 0

    if threadID <= brains.number_neurons 
        for i = 1:brains.number_inputs
            @inbounds brains.W_i[threadID, i, blockID] = individuals[blockID, threadID + (i-1) * brains.number_neurons + offset]
            @inbounds brains.W_f[threadID, i, blockID] = individuals[blockID, threadID + (i-1) * brains.number_neurons + 1 * w_size + offset]
            @inbounds brains.W_o[threadID, i, blockID] = individuals[blockID, threadID + (i-1) * brains.number_neurons + 2 * w_size + offset]
            @inbounds brains.W_c[threadID, i, blockID] = individuals[blockID, threadID + (i-1) * brains.number_neurons + 3 * w_size + offset]
        end

        offset += 4* w_size

        for i = 1:brains.number_neurons
            @inbounds brains.U_i[threadID, i, blockID] = individuals[blockID, threadID + (i-1) * brains.number_neurons + offset]
            @inbounds brains.U_f[threadID, i, blockID] = individuals[blockID, threadID + (i-1) * brains.number_neurons + 1 * u_size + offset]
            @inbounds brains.U_o[threadID, i, blockID] = individuals[blockID, threadID + (i-1) * brains.number_neurons + 2 * u_size + offset]
            @inbounds brains.U_c[threadID, i, blockID] = individuals[blockID, threadID + (i-1) * brains.number_neurons + 3 * u_size + offset]
        end

        offset += 4* u_size

        brains.b_i[threadID, blockID] = individuals[blockID, threadID + offset]
        brains.b_f[threadID, blockID] = individuals[blockID, threadID + 1 * brains.number_neurons + offset]
        brains.b_o[threadID, blockID] = individuals[blockID, threadID + 2 * brains.number_neurons + offset]
        brains.b_c[threadID, blockID] = individuals[blockID, threadID + 3 * brains.number_neurons + offset]

        offset += 4* b_size
    end

    if threadID <= brains.number_outputs
        for i = 1:brains.number_neurons
            @inbounds brains.V[threadID, i, blockID] = individuals[blockID, threadID + (i-1) * brains.number_outputs + offset]
        end

        offset += v_size
        brains.b_v[threadID,blockID] = individuals[blockID, threadID + offset]
    end 
    

    sync_threads()

    reset(brains)

    sync_threads()

end

function reset(brains::LongShortTermMemoryNN)

    threadID = threadIdx().x
    blockID = blockIdx().x

    if threadID <= brains.number_neurons
        @inbounds brains.cell_state[threadID, blockID] = 0.0
        @inbounds brains.hidden_state[threadID, blockID] = 0.0
    end 
end

function step(brains::LongShortTermMemoryNN, input, output, offset_memory)

    threadID = threadIdx().x
    blockID = blockIdx().x

    gate_results = @cuDynamicSharedMem(Float32, (4, brains.number_neurons), offset_memory)
    fill!(gate_results, 0.0f0)
    offset_memory += sizeof(gate_results)
    
    if threadID <= brains.number_neurons
        
        #Input calculation for gates
        for i = 1:brains.number_inputs
            #Input Gate
            @inbounds gate_results[1, threadID] += brains.W_i[threadID, i, blockID] * input[i]
            #Forget Gate
            @inbounds gate_results[2, threadID] += brains.W_f[threadID, i, blockID] * input[i]
            #Cell Gate
            @inbounds gate_results[3, threadID] += brains.W_c[threadID, i, blockID] * input[i]
            #Output Gate
            @inbounds gate_results[4, threadID] += brains.W_o[threadID, i, blockID] * input[i]
        end

        #Hidden state calculation for gates 
        for i = 1:brains.number_neurons
            #Input Gate
            @inbounds gate_results[1, threadID] += brains.U_i[threadID, i, blockID] * brains.hidden_state[i, blockID]
            #Forget Gate
            @inbounds gate_results[2, threadID] += brains.U_f[threadID, i, blockID] * brains.hidden_state[i, blockID]
            #Cell Gate
            @inbounds gate_results[3, threadID] += brains.U_c[threadID, i, blockID] * brains.hidden_state[i, blockID]
            #Output Gate
            @inbounds gate_results[4, threadID] += brains.U_o[threadID, i, blockID] * brains.hidden_state[i, blockID]
        end

        #Adding Biases
        gate_results[1, threadID] += brains.b_i[threadID, blockID]
        gate_results[2, threadID] += brains.b_f[threadID, blockID]
        gate_results[3, threadID] += brains.b_c[threadID, blockID]
        gate_results[4, threadID] += brains.b_o[threadID, blockID]

        #Applying activation functions
        @inbounds gate_results[1, threadID] = sigmoid(gate_results[1, threadID])
        @inbounds gate_results[2, threadID] = sigmoid(gate_results[2, threadID])
        @inbounds gate_results[3, threadID] = tanh(gate_results[3, threadID])
        @inbounds gate_results[4, threadID] = sigmoid(gate_results[4, threadID])

        #New Cell state 
        @inbounds brains.cell_state[threadID, blockID] = gate_results[2, threadID] * brains.cell_state[threadID, blockID] + gate_results[1, threadID] * gate_results[3, threadID]

        #Hidden states
        brains.hidden_state[threadID, blockID] = gate_results[4, threadID] * tanh(brains.cell_state[threadID, blockID])

    end

    sync_threads()

    #Output Layer
    if threadID <= brains.number_outputs
        output[threadID] = 0.0
        for i = 1:brains.number_neurons
            @inbounds output[threadID] += brains.V[threadID, i, blockID] * brains.hidden_state[i, blockID]
        end
        output[threadID] += brains.b_v[threadID, blockID]
        output[threadID] = tanh(output[threadID])
    end    

    sync_threads()
end

function get_individual_size(brains::LongShortTermMemoryNN)
    return 4 * brains.number_inputs * brains.number_neurons +       #Input weights
        4 * brains.number_neurons * brains.number_neurons +         #hidden_state weights
        4 * brains.number_neurons +                                 #Biases
        brains.number_outputs * brains.number_neurons +             #Output layer weights
        brains.number_outputs                                       #Output bias
end

function sigmoid(x)
    return one(x) / (one(x) + exp(-x))  
end