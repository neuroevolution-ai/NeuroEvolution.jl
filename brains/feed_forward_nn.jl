using CUDA
using Adapt
using DataStructures

struct FeedForwardNN{A,B,C,D,E,F}
    hidden_layer1_size::Int64
    hidden_layer2_size::Int64
    W1::A
    W2::B
    W3::C
    b1::D
    b2::E
    b3::F
    input_size::Int64
    output_size::Int64
end

function FeedForwardNN(configuration::OrderedDict, number_inputs::Int, number_outputs::Int, number_individuals::Int)

    FeedForwardNN(
        configuration["hidden_layer1_size"],
        configuration["hidden_layer2_size"],
        CUDA.fill(0.0f0, (configuration["hidden_layer1_size"], number_inputs, number_individuals)),
        CUDA.fill(0.0f0, (configuration["hidden_layer2_size"], configuration["hidden_layer1_size"], number_individuals)),
        CUDA.fill(0.0f0, (number_outputs, configuration["hidden_layer2_size"], number_individuals)),
        CUDA.fill(0.0f0, (configuration["hidden_layer1_size"], number_individuals)),
        CUDA.fill(0.0f0, (configuration["hidden_layer2_size"], number_individuals)),
        CUDA.fill(0.0f0, (number_outputs, number_individuals)),
        number_inputs,
        number_outputs,
    )
end

Adapt.@adapt_structure FeedForwardNN


function get_required_threads(brains::FeedForwardNN)

    return max(brains.hidden_layer1_size, brains.hidden_layer2_size)
end

function get_memory_requirements(brains::FeedForwardNN)
    
    return sizeof(Float32) * (brains.hidden_layer1_size + brains.hidden_layer2_size)
end

function initialize(brains::FeedForwardNN, individuals)

    threadID = threadIdx().x
    blockID = blockIdx().x

    W1_size = brains.input_size * brains.hidden_layer1_size
    W2_size = brains.hidden_layer1_size * brains.hidden_layer2_size
    W3_size = brains.hidden_layer2_size * brains.output_size
    b1_size = brains.hidden_layer1_size
    b2_size = brains.hidden_layer2_size

    offset = 0

    if threadID <= brains.hidden_layer1_size
        for i = 1:brains.input_size
            @inbounds brains.W1[threadID, i, blockID] = individuals[blockID, threadID+(i-1)*brains.hidden_layer1_size]
        end

        @inbounds brains.b1[threadID, blockID] = individuals[blockID, W1_size+threadID]
    end

    offset += W1_size + b1_size

    if threadID <= brains.hidden_layer2_size
        for i = 1:brains.hidden_layer1_size
            @inbounds brains.W2[threadID, i, blockID] = individuals[blockID, offset+threadID+(i-1)*brains.hidden_layer2_size]
        end

        @inbounds brains.b2[threadID, blockID] = individuals[blockID, offset+W2_size+threadID]
    end

    offset += W2_size + b2_size

    if threadID <= brains.output_size
        for i = 1:brains.hidden_layer2_size
            @inbounds brains.W3[threadID, i, blockID] = individuals[blockID, offset+threadID+(i-1)*brains.output_size]
        end

        @inbounds brains.b3[threadID, blockID] = individuals[blockID, offset+W3_size+threadID]
    end

    sync_threads()

end

function reset(brains::FeedForwardNN)

end

function step(brains::FeedForwardNN, input, output, offset_shared_memory)

    threadID = threadIdx().x
    blockID = blockIdx().x

    hidden_layer1_output = @cuDynamicSharedMem(Float32, brains.hidden_layer1_size, offset_shared_memory)
    offset_shared_memory += sizeof(hidden_layer1_output)

    sync_threads()

    hidden_layer2_output = @cuDynamicSharedMem(Float32, brains.hidden_layer2_size, offset_shared_memory)
    offset_shared_memory += sizeof(hidden_layer2_output)

    sync_threads()

    # Calculate first hidden layer: 
    # hidden_layer1_output = tanh(W1*input + b1)
    if threadID <= brains.hidden_layer1_size
        
        @inbounds  hidden_layer1_output[threadID] = brains.b1[threadID, blockID]
        for i = 1:brains.input_size
            @inbounds  hidden_layer1_output[threadID] += (brains.W1[threadID, i, blockID] * input[i])
        end
        @inbounds  hidden_layer1_output[threadID] = tanh(hidden_layer1_output[threadID])
    end

    sync_threads()

    # Calculate second hidden layer: 
    # hidden_layer2_output = tanh(W2*hidden_layer1_output + b2)
    if threadID <= brains.hidden_layer2_size

        @inbounds  hidden_layer2_output[threadID] = brains.b2[threadID, blockID]
        for i = 1:brains.hidden_layer1_size
            @inbounds  hidden_layer2_output[threadID] += (brains.W2[threadID, i, blockID] * hidden_layer1_output[i])
        end
        @inbounds  hidden_layer2_output[threadID] = tanh(hidden_layer2_output[threadID])
    end

    sync_threads()

    # Calculate outputs: 
    # output = tanh(W3*hidden_layer2_output + b3)
    if threadID <= brains.output_size

        @inbounds  output[threadID] = brains.b3[threadID, blockID]
        for i = 1:brains.hidden_layer2_size
            @inbounds  output[threadID] += (brains.W3[threadID, i, blockID] * hidden_layer2_output[i])
        end
        @inbounds output[threadID] = tanh(output[threadID])
    end

    sync_threads()

end


function get_free_parameter_usage(brains)

    usage_dict = Dict()

    usage_dict["W1"] = brains.input_size * brains.hidden_layer1_size
    usage_dict["W2"] = brains.hidden_layer1_size * brains.hidden_layer2_size
    usage_dict["W3"] = brains.hidden_layer2_size * brains.output_size
    usage_dict["b1"] = brains.hidden_layer1_size
    usage_dict["b2"] = brains.hidden_layer2_size
    usage_dict["b3"] = brains.output_size

    return usage_dict
end

function get_individual_size(brains)

    usage_dict = get_free_parameter_usage(brains)

    return usage_dict["W1"] +
           usage_dict["W2"] +
           usage_dict["W3"] +
           usage_dict["b1"] +
           usage_dict["b2"] +
           usage_dict["b3"]
end
