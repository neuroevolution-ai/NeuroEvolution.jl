using Adapt

struct LongShortTermMemoryNN{A, B, C}
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
    hidden::C
    input_size::Int64
    output_size::Int64
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
        number_inputs,
        number_outputs
    )
end

Adapt.@adapt_structure LongShortTermMemoryNN

function initialize(threadID, blockID, brains::LongShortTermMemoryNN, individuals)

    w_size = brains.input_size * brains.number_neurons
    u_size = brains.number_neurons * brains.number_neurons
    offset = 0

    if threadID <= brains.number_neurons
        for i = 1:brains.input_size
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
    end    

    sync_threads()

    reset(threadID, blockID, brains)

    sync_threads()

end

function step(threadID, blockId, brains::LongShortTermMemoryNN)
    #Forget-Gate 

end

function reset(threadID, blockID, brains::LongShortTermMemoryNN)

    if threadID <= brains.number_neurons
        brains.hidden[threadID, blockID] = 0.0
    end 

    sync_threads()
end

function get_individual_size(brains::LongShortTermMemoryNN)
    return 4 * brains.input_size * brains.number_neurons +
        4 * brains.number_neurons * brains.number_neurons + 
        4 * brains.number_neurons
end

function get_required_threads(brains::LongShortTermMemoryNN)
    return brains.number_neurons
end

function sigmoid(x)
    return one(x) / (one(x) + exp(-x))  
end