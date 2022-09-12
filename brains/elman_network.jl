using Adapt
using LinearAlgebra

struct ElmanNetwork{A, B, C, D, E}
    W::A
    U::B
    b::C
    hidden_state::C
    V::D
    b_v::E
    number_neurons::Int64
    number_inputs::Int64
    number_outputs::Int64
end

function ElmanNetwork(configuration::OrderedDict, number_inputs::Int, number_outputs::Int, number_individuals::Int)

    ElmanNetwork(
        CUDA.fill(0.0f0, (configuration["number_neurons"], number_inputs, number_individuals)),
        CUDA.fill(0.0f0, (configuration["number_neurons"], configuration["number_neurons"], number_individuals)),
        CUDA.fill(0.0f0, (configuration["number_neurons"], number_individuals)),
        CUDA.fill(0.0f0, (configuration["number_neurons"], number_individuals)),
        CUDA.fill(0.0f0, (number_outputs, configuration["number_neurons"], number_individuals)),
        CUDA.fill(0.0f0, (number_outputs, number_individuals)),
        configuration["number_neurons"],
        number_inputs,
        number_outputs
    )

end

Adapt.@adapt_structure ElmanNetwork

function get_required_threads(brains::ElmanNetwork)
    return max(brains.number_neurons, brains.number_inputs, brains.number_outputs)
end

function get_memory_requirements(brains::ElmanNetwork)
    return sizeof(Float32) * brains.number_neurons
end

function initialize(brains::ElmanNetwork, individuals)
    
    threadID = threadIdx().x
    blockID = blockIdx().x

    w_size = brains.number_inputs * brains.number_neurons
    u_size = brains.number_neurons * brains.number_neurons
    b_size = brains.number_neurons
    v_size = brains.number_neurons * brains.number_outputs

    offset = 0

    if threadID <= brains.number_neurons 
        for i = 1:brains.number_inputs
            brains.W[threadID, i, blockID] = individuals[blockID, threadID + (i-1) * brains.number_neurons + offset]
        end

        offset += w_size

        for i = 1:brains.number_neurons
            brains.U[threadID, i, blockID] = individuals[blockID, threadID + (i-1) * brains.number_neurons + offset]
        end

        offset += u_size

        brains.b[threadID, blockID] = individuals[blockID, threadID + offset]

        offset += b_size
    end

    if threadID <= brains.number_outputs
        for i = 1:brains.number_neurons
            brains.V[threadID, i, blockID] = individuals[blockID, threadID + (i-1) * brains.number_outputs + offset]
        end

        offset += v_size
        brains.b_v[threadID,blockID] = individuals[blockID, threadID + offset]
    end

    sync_threads()

    if threadID <= brains.number_neurons
        reset(brains)
    end
    
    sync_threads()
end

function reset(brains::ElmanNetwork)
    
    threadID = threadIdx().x
    blockID = blockIdx().x

    if threadID <= brains.number_neurons
        brains.hidden_state[threadID, blockID] = 0.0
    end 
end

function step(brains::ElmanNetwork, input, output, offset_memory)
    
    threadID = threadIdx().x
    blockID = blockIdx().x

    gate_results = @cuDynamicSharedMem(Float32, brains.number_neurons, offset_memory)
    fill!(gate_results, 0.0f0)
    offset_memory += sizeof(gate_results)
    
    #Rnn Layer
    if threadID <= brains.number_neurons
        
        #Input calculation for gate
        for i = 1:brains.number_inputs
            @inbounds gate_results[threadID] += brains.W[threadID, i, blockID] * input[i]
        end

        #Hidden state calculation for gates 
        for i = 1:brains.number_neurons
            gate_results[threadID] += brains.U[threadID, i, blockID] * brains.hidden_state[i, blockID]
        end

        #Adding Biases
        gate_results[threadID] += brains.b[threadID, blockID]


        #Applying activation function
        gate_results[threadID] = tanh(gate_results[threadID])

        #New Hidden state
        brains.hidden_state[threadID, blockID] = gate_results[threadID]
    end
    
    sync_threads()

    #Output Layer
    if threadID <= brains.number_outputs
        output[threadID] = 0.0
        for i = 1:brains.number_neurons
            output[threadID] += brains.V[threadID, i, blockID] * brains.hidden_state[i, blockID]
        end
        output[threadID] += brains.b_v[threadID, blockID]
        output[threadID] = tanh(output[threadID])
    end    

    sync_threads()
end    

function get_individual_size(brains::ElmanNetwork)
    return brains.number_inputs * brains.number_neurons +       #Input weights
        brains.number_neurons * brains.number_neurons +         #hidden_state weights
        brains.number_neurons +                                 #Biases
        brains.number_outputs * brains.number_neurons +         #Output layer weights
        brains.number_outputs                                   #Output bias
end