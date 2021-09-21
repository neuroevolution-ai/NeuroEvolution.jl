using CUDA
using Adapt

@enum Dif_equation original separated

struct ContinuousTimeRNN{A,B}
    delta_t::Float32
    number_neurons::Int64
    differential_equation::Dif_equation
    clipping_range_min::Float32
    clipping_range_max::Float32
    alpha::Float32
    V::A
    W::A
    T::A
    x::B
end

function ContinuousTimeRNN(configuration::OrderedDict, number_inputs::Int, number_individuals::Int)

    ContinuousTimeRNN(
        convert(Float32, configuration["delta_t"]),
        configuration["number_neurons"],
        separated,
        convert(Float32, configuration["clipping_range_min"]),
        convert(Float32, configuration["clipping_range_max"]),
        convert(Float32, configuration["alpha"]),
        CUDA.fill(0.0f0, (configuration["number_neurons"], number_inputs, number_individuals)),
        CUDA.fill(0.0f0, (configuration["number_neurons"], configuration["number_neurons"], number_individuals)),
        CUDA.fill(0.0f0, (number_inputs, configuration["number_neurons"], number_individuals)),
        CUDA.fill(0.0f0, (configuration["number_neurons"], number_individuals)))
end

Adapt.@adapt_structure ContinuousTimeRNN


function tensordot(input, x, y, kernel, kernel_size)
    dotproduct = 0.0f0
    for i = 1:kernel_size
        for j = 1:kernel_size
            @inbounds dotproduct += input[y-1+j, x-1+i] * kernel[j, i]
        end
    end
    return dotproduct
end

function custom_conv2(input, result, kernel)
    input_c = size(input, 1)
    input_r = size(input, 2)
    kernel_c = size(kernel, 1)
    kernel_r = size(kernel, 2)
    result_c = size(result, 1)
    result_r = size(result, 2)
    #result hat dimensionen [input_r - kernel_r + 1,input_c - kernel_c + 1]

    for i = 1:result_r
        for j in result_c
            @inbounds result[i, j] = tensordot(input, i, j, kernel, kernel_c)
        end
    end
end


function get_memory_requirements(number_inputs, number_outputs, brain_cfg::ContinuousTimeRNN)
    return sizeof(Float32) * (
        brain_cfg.number_neurons *
        (brain_cfg.number_neurons + number_inputs + number_outputs + 2) +
        number_inputs +
        number_outputs
    )
end

function brain_initialize(threadID, blockID, V, W, T, individuals, brain_cfg::ContinuousTimeRNN)
    number_neurons = size(W, 1)
    input_size = size(V, 2)
    output_size = size(T, 1)
    v_size = input_size * number_neurons
    w_size = number_neurons * number_neurons
    #initialize the brain_masks from the genome and set all Values on the diagonal of W negative
    for i = 1:input_size
        @inbounds V[threadID, i] = individuals[blockID, threadID+((i-1)*number_neurons)]
        @inbounds brain_cfg.V[threadID, i, blockID] =
            individuals[blockID, threadID+((i-1)*number_neurons)]
    end
    sync_threads()
    for i = 1:number_neurons
        @inbounds W[threadID, i] =
            individuals[blockID, v_size+(threadID+((i-1)*number_neurons))]
        @inbounds brain_cfg.W[threadID, i, blockID] =
            individuals[blockID, v_size+(threadID+((i-1)*number_neurons))]
    end
    @inbounds W[threadID, threadID] = -abs(W[threadID, threadID])
    @inbounds brain_cfg.W[threadID, threadID, blockID] = -abs(W[threadID, threadID])
    for i = 1:output_size
        @inbounds T[i, threadID] =
            individuals[blockID, v_size+w_size+(i+((threadID-1)*output_size))]
        @inbounds brain_cfg.T[i, threadID, blockID] =
            individuals[blockID, v_size+w_size+(i+((threadID-1)*output_size))]
    end

    return
end

function brain_step(
    threadID,
    blockID,
    temp_V,
    V,
    W,
    T,
    x,
    input,
    action,
    brain_cfg::ContinuousTimeRNN,
)

    #treat image input as grey_scale

    input_size = size(V, 2)
    output_size = size(T, 1)

    if brain_cfg.differential_equation == original
        #V * input multiplication
        V_value = 0.0f0
        for i = 1:input_size
            @inbounds V_value += V[threadID, i] * input[i]
            #@inbounds V_value += brain_cfg.V[threadID, i,blockID] * input[i]
        end
        #
        @inbounds temp_V[threadID] = tanh(x[threadID] + V_value)


        #W * result of Vmult
        W_value = 0.0f0
        for i = 1:brain_cfg.number_neurons
            @inbounds W_value += W[threadID, i] * temp_V[i]
        end
        #
        dx_dt = (-brain_cfg.alpha * x[threadID]) + W_value
    elseif brain_cfg.differential_equation == separated
        V_value = 0.0f0
        for i = 1:input_size
            @inbounds V_value += V[threadID, i] * input[i]
            #@inbounds V_value += brain_cfg.V[threadID, i,blockID] * input[i]
        end
        @inbounds temp_V[threadID] = V_value
        W_value = 0.0f0
        for i = 1:brain_cfg.number_neurons
            @inbounds W_value += W[threadID, i] * tanh(x[i])
            #@inbounds W_value += brain_cfg.W[threadID, i,blockID] * tanh(brain_cfg.x[i,blockID])
        end
        temp_V[threadID] += W_value
        dx_dt = (-brain_cfg.alpha * x[threadID]) + temp_V[threadID]
        #dx_dt2 = (-brain_cfg.alpha * brain_cfg.x[threadID,blockID]) + temp_V[threadID]
    end
    @inbounds x[threadID] += (brain_cfg.delta_t * dx_dt)
    @inbounds x[threadID] =
        clamp(x[threadID], brain_cfg.clipping_range_min, brain_cfg.clipping_range_max)

    #@inbounds brain_cfg.x[threadID,blockID] += (brain_cfg.delta_t * dx_dt2)
    #@inbounds brain_cfg.x[threadID,blockID] =
    #    clamp(brain_cfg.x[threadID,blockID], brain_cfg.clipping_range_min, brain_cfg.clipping_range_max)
    sync_threads()

    #T*temp_W matmul:
    if threadID <= output_size
        T_value = 0.0f0
        for i = 1:brain_cfg.number_neurons
            @inbounds T_value = T_value + T[threadID, i] * x[i]
            #@inbounds T_value = T_value + brain_cfg.T[threadID, i,blockID] * brain_cfg.x[i,blockID]
        end
        @inbounds action[threadID] = tanh(T_value)
    end
    #
    return
end

function get_masks_from_brain_state(brain_state::Dict) # what format for brain_state
    v_mask = get(brain_state, "v_mask", 1)
    w_mask = get(brain_state, "w_mask", 1)
    t_mask = get(brain_state, "t_mask", 1)

    return v_mask, w_mask, t_mask
end

function _generate_mask(n::Int, m::Int)
    return trues(n, m)
end

function generate_brain_state(input_size, output_size, configuration)
    v_mask = _generate_mask(configuration["number_neurons"], input_size)
    w_mask =
        _generate_mask(configuration["number_neurons"], configuration["number_neurons"])
    t_mask = _generate_mask(output_size, configuration["number_neurons"])

    return get_brain_state_from_masks(v_mask, w_mask, t_mask)
end

function get_brain_state_from_masks(v_mask, w_mask, t_mask)
    return Dict("v_mask" => v_mask, "w_mask" => w_mask, "t_mask" => t_mask)
end

function get_free_parameter_usage(brain_state)
    v_mask, w_mask, t_mask = get_masks_from_brain_state(brain_state)
    #count true values
    free_parameters_v = count(v_mask)
    free_parameters_w = count(w_mask)
    free_parameters_t = count(t_mask)

    free_parameters =
        Dict("V" => free_parameters_v, "W" => free_parameters_w, "T" => free_parameters_t)
    return free_parameters
end

function sum_dict(node)
    sum_ = 0
    for (key, value) in node
        if (value isa Dict)
            sum_ += sum_dict(value)
        else
            sum_ += value
        end

    end
    return sum_
end

function get_individual_size(brain_state)
    usage_dict = get_free_parameter_usage(brain_state)
    return sum_dict(usage_dict)
end