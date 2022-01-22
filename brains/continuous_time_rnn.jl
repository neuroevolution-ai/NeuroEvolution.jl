using CUDA
using Adapt
using DataStructures

@enum Dif_equation LiHoChow2005 NaturalNet

struct ContinuousTimeRNN{A,B}
    delta_t::Float32
    number_neurons::Int64
    differential_equation::Dif_equation
    clipping_range_min::Float32
    clipping_range_max::Float32
    set_principle_diagonal_elements_of_W_negative::Bool
    alpha::Float32
    V::A
    W::A
    T::A
    x::B
    input_size::Int64
    output_size::Int64
end

function ContinuousTimeRNN(configuration::OrderedDict, number_inputs::Int, number_outputs::Int, number_individuals::Int)

    if configuration["differential_equation"] == "NaturalNet"
        differential_eq = NaturalNet
    elseif configuration["differential_equation"] == "LiHoChow2005"
        differential_eq = LiHoChow2005
    end

    ContinuousTimeRNN(
        convert(Float32, configuration["delta_t"]),
        configuration["number_neurons"],
        differential_eq,
        convert(Float32, configuration["clipping_range_min"]),
        convert(Float32, configuration["clipping_range_max"]),
        convert(Bool, configuration["set_principle_diagonal_elements_of_W_negative"]),
        convert(Float32, configuration["alpha"]),
        CUDA.fill(0.0f0, (configuration["number_neurons"], number_inputs, number_individuals)),
        CUDA.fill(0.0f0, (configuration["number_neurons"], configuration["number_neurons"], number_individuals)),
        CUDA.fill(0.0f0, (number_outputs, configuration["number_neurons"], number_individuals)),
        CUDA.fill(0.0f0, (configuration["number_neurons"], number_individuals)),
        number_inputs,
        number_outputs,
    )
end

Adapt.@adapt_structure ContinuousTimeRNN


function get_memory_required_threads(brains::ContinuousTimeRNN)

    return brains.number_neurons
end

function get_memory_requirements(brains::ContinuousTimeRNN)
    return sizeof(Float32) * brains.number_neurons
end

function initialize(brains::ContinuousTimeRNN, individuals)

    threadID = threadIdx().x
    blockID = blockIdx().x

    v_size = brains.input_size * brains.number_neurons
    w_size = brains.number_neurons * brains.number_neurons

    for i = 1:brains.input_size
        @inbounds brains.V[threadID, i, blockID] = individuals[blockID, threadID+(i-1)*brains.number_neurons]
    end

    for i = 1:brains.number_neurons
        @inbounds brains.W[threadID, i, blockID] = individuals[blockID, v_size+(threadID+(i-1)*brains.number_neurons)]
    end

    sync_threads()

    if brains.set_principle_diagonal_elements_of_W_negative == true
        @inbounds brains.W[threadID, threadID, blockID] = -abs(brains.W[threadID, threadID, blockID])
    end

    for i = 1:brains.output_size
        @inbounds brains.T[i, threadID, blockID] = individuals[blockID, v_size+w_size+(i+(threadID-1)*brains.output_size)]
    end

    sync_threads()

    reset(brains)

    sync_threads()

end

function reset(brains::ContinuousTimeRNN)

    threadID = threadIdx().x
    blockID = blockIdx().x

    if threadID <= brains.number_neurons
        @inbounds brains.x[threadID, blockID] = 0.0
    end

    sync_threads()

end

function step(brains::ContinuousTimeRNN, input, output, offset_shared_memory)

    threadID = threadIdx().x
    blockID = blockIdx().x

    V_value = @cuDynamicSharedMem(Float32, brains.number_neurons, offset_shared_memory)

    if threadID <= brains.number_neurons

        # V_value = V * input (Matrix-Vector-Multiplication)
        V_value[threadID] = 0.0
        for i = 1:brains.input_size
            @inbounds V_value[threadID] += (brains.V[threadID, i, blockID] * input[i])
        end

        sync_threads()

        if brains.differential_equation == NaturalNet

            # W_value = W * tanh(x) (Matrix-Vector-Multiplication)
            W_value = 0.0
            for i = 1:brains.number_neurons
                @inbounds W_value += (brains.W[threadID, i, blockID] * tanh(brains.x[i, blockID]))
            end

            sync_threads()

            # Differential Equation
            dx_dt = -brains.alpha * brains.x[threadID, blockID] + W_value + V_value[threadID]

        elseif brains.differential_equation == LiHoChow2005

            # W_value = W * (tanh(x) + V_value) (Matrix-Vector-Multiplication)
            W_value = 0.0
            for i = 1:brains.number_neurons
                @inbounds W_value += (brains.W[threadID, i, blockID] * (tanh(brains.x[i, blockID] + V_value[i])))
            end

            sync_threads()

            # Differential Equation
            dx_dt = -brains.alpha * brains.x[threadID, blockID] + W_value

        end

        # Euler forward discretization
        @inbounds brains.x[threadID, blockID] += brains.delta_t * dx_dt

        # Clip x to state boundaries
        @inbounds brains.x[threadID, blockID] = clamp(brains.x[threadID, blockID], brains.clipping_range_min, brains.clipping_range_max)

    end

    sync_threads()

    if threadID <= brains.output_size

        # T_value = T * x (Matrix-Vector-Multiplication)
        T_value = 0.0
        for i = 1:brains.number_neurons
            T_value += brains.T[threadID, i, blockID] * brains.x[i, blockID]
        end

        # Calculate outputs
        output[threadID] = tanh(T_value)

    end

    sync_threads()

end

function get_free_parameter_usage(brains)

    usage_dict = Dict()

    usage_dict["V"] = brains.input_size * brains.number_neurons
    usage_dict["W"] = brains.number_neurons * brains.number_neurons
    usage_dict["T"] = brains.number_neurons * brains.output_size

    return usage_dict
end

function get_individual_size(brains)

    usage_dict = get_free_parameter_usage(brains)

    return usage_dict["V"] + usage_dict["W"] + usage_dict["T"]
end