using CUDA
using Adapt
using DataStructures


struct ChallengerNeuralNetwork{A,B,C,D,E,F}
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

function ChallengerNeuralNetwork(configuration::OrderedDict, number_individuals::Int)

    ChallengerNeuralNetwork(
        configuration["hidden_layer1_size"],
        configuration["hidden_layer2_size"],
        CUDA.fill(0.0f0, (configuration["hidden_layer1_size"], number_inputs, number_individuals)),
        CUDA.fill(0.0f0, (configuration["hidden_layer2_size"], configuration["hidden_layer1_size"], number_individuals)),
        CUDA.fill(0.0f0, (number_outputs, configuration["hidden_layer2_size"], number_individuals)),
        CUDA.fill(0.0f0, (configuration["hidden_layer1_size"], number_individuals)),
        CUDA.fill(0.0f0, (configuration["hidden_layer2_size"], number_individuals)),
        CUDA.fill(0.0f0, (number_outputs, number_individuals)),
        configuration["number_actions"],
        configuration["number_observations"],
    )
end

Adapt.@adapt_structure ChallengerNeuralNetwork


function initialize(environments::ChallengerNeuralNetwork, input, env_seed, offset_shared_memory)

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

    # Initialize inputs
    if threadID <= environments.number_inputs
        input[threadID] = 0.5
    end

    sync_threads()

    return

end

function get_memory_requirements(environments::DummyApp)
    return sizeof(Int32) * environments.number_gui_elements
end

function get_number_inputs(environments::DummyApp)
    return environments.number_inputs
end

function get_number_outputs(environments::DummyApp)
    return environments.number_outputs
end