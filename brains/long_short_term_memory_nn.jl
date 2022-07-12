using Adapt
using LinearAlgebra

struct LongShortTermMemoryNN{A, B, C, D, E}
    number_neurons::Int64
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
    number_inputs::Int64
    number_outputs::Int64
end

function LongShortTermMemoryNN(number_neurons::Int, number_inputs::Int, number_outputs::Int, number_individuals::Int)

    LongShortTermMemoryNN(
        number_neurons,
        CUDA.fill(0.0f0, (number_neurons, number_inputs, number_individuals)),
        CUDA.fill(0.0f0, (number_neurons, number_inputs, number_individuals)),
        CUDA.fill(0.0f0, (number_neurons, number_inputs, number_individuals)),
        CUDA.fill(0.0f0, (number_neurons, number_inputs, number_individuals)),
        CUDA.fill(0.0f0, (number_neurons, number_neurons, number_individuals)),
        CUDA.fill(0.0f0, (number_neurons, number_neurons, number_individuals)),
        CUDA.fill(0.0f0, (number_neurons, number_neurons, number_individuals)),
        CUDA.fill(0.0f0, (number_neurons, number_neurons, number_individuals)),
        CUDA.fill(0.0f0, (number_neurons, number_individuals)),
        CUDA.fill(0.0f0, (number_neurons, number_individuals)),
        CUDA.fill(0.0f0, (number_neurons, number_individuals)),
        CUDA.fill(0.0f0, (number_neurons, number_individuals)),
        CUDA.fill(0.0f0, (number_neurons, number_individuals)),
        CUDA.fill(0.0f0, (number_neurons, number_individuals)),
        CUDA.fill(0.0f0, (number_outputs, number_neurons, number_individuals)),
        CUDA.fill(0.0f0, (number_outputs, number_individuals)),
        number_inputs,
        number_outputs
    )
end

Adapt.@adapt_structure LongShortTermMemoryNN

function initialize(threadID, blockID, brains::LongShortTermMemoryNN, individuals)

    w_size = brains.number_inputs * brains.number_neurons
    u_size = brains.number_neurons * brains.number_neurons
    b_size = brains.number_neurons
    v_size = brains.number_neurons * brains.number_outputs

    offset = 0

    if threadID <= brains.number_neurons 
        for i = 1:brains.number_inputs
            brains.W_i[threadID, i, blockID] = individuals[blockID, threadID + (i-1) * brains.number_neurons + offset]
            brains.W_f[threadID, i, blockID] = individuals[blockID, threadID + (i-1) * brains.number_neurons + 1 * w_size + offset]
            brains.W_o[threadID, i, blockID] = individuals[blockID, threadID + (i-1) * brains.number_neurons + 2 * w_size + offset]
            brains.W_c[threadID, i, blockID] = individuals[blockID, threadID + (i-1) * brains.number_neurons + 3 * w_size + offset]
        end

        offset += 4* w_size

        for i = 1:brains.number_neurons
            brains.U_i[threadID, i, blockID] = individuals[blockID, threadID + (i-1) * brains.number_neurons + offset]
            brains.U_f[threadID, i, blockID] = individuals[blockID, threadID + (i-1) * brains.number_neurons + 1 * u_size + offset]
            brains.U_o[threadID, i, blockID] = individuals[blockID, threadID + (i-1) * brains.number_neurons + 2 * u_size + offset]
            brains.U_c[threadID, i, blockID] = individuals[blockID, threadID + (i-1) * brains.number_neurons + 3 * u_size + offset]
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
            brains.V[threadID, i, blockID] = individuals[blockID, threadID + (i-1) * brains.number_outputs + offset]
        end

        offset += v_size
        brains.b_v[threadID,blockID] = individuals[blockID, threadID + offset]
    end 
    

    sync_threads()

    reset(threadID, blockID, brains)

    sync_threads()

end

function step(threadID, blockID, brains::LongShortTermMemoryNN, gate_results, input, output)
    
    if threadID <= brains.number_neurons
        gate_results[1, threadID] = 0.0
        gate_results[2, threadID] = 0.0
        gate_results[3, threadID] = 0.0
        gate_results[4, threadID] = 0.0

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
            gate_results[1, threadID] += brains.U_i[threadID, i, blockID] * brains.hidden_state[i, blockID]
            #Forget Gate
            gate_results[2, threadID] += brains.U_f[threadID, i, blockID] * brains.hidden_state[i, blockID]
            #Cell Gate
            gate_results[3, threadID] += brains.U_c[threadID, i, blockID] * brains.hidden_state[i, blockID]
            #Output Gate
            gate_results[4, threadID] += brains.U_o[threadID, i, blockID] * brains.hidden_state[i, blockID]
        end

        #Adding Biases
        gate_results[1, threadID] += brains.b_i[threadID, blockID]
        gate_results[2, threadID] += brains.b_f[threadID, blockID]
        gate_results[3, threadID] += brains.b_c[threadID, blockID]
        gate_results[4, threadID] += brains.b_o[threadID, blockID]

        #Applying activation functions
        gate_results[1, threadID] = sigmoid(gate_results[1, threadID])
        gate_results[2, threadID] = sigmoid(gate_results[2, threadID])
        gate_results[3, threadID] = tanh(gate_results[3, threadID])
        gate_results[4, threadID] = sigmoid(gate_results[4, threadID])

        #New Cell state 
        brains.cell_state[threadID, blockID] = gate_results[2, threadID] * brains.cell_state[threadID, blockID] + gate_results[1, threadID] * gate_results[3, threadID]

        #Hidden states
        brains.hidden_state[threadID, blockID] = gate_results[4, threadID] * tanh(brains.cell_state[threadID, blockID])

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


function reset(threadID, blockID, brains::LongShortTermMemoryNN)
    if threadID <= brains.number_neurons
        brains.cell_state[threadID, blockID] = 0.0
        brains.hidden_state[threadID, blockID] = 0.0
    end 
end

function get_individual_size(brains::LongShortTermMemoryNN)
    return 4 * brains.number_inputs * brains.number_neurons +       #Input weights
        4 * brains.number_neurons * brains.number_neurons +         #hidden_state weights
        4 * brains.number_neurons +                                 #Biases
        brains.number_outputs * brains.number_neurons +             #Output layer weights
        brains.number_outputs                                       #Output bias
end

function get_required_threads(brains::LongShortTermMemoryNN)
    return max(brains.number_neurons, brains.number_inputs, brains.number_outputs)
end

function sigmoid(x)
    return one(x) / (one(x) + exp(-x))  
end