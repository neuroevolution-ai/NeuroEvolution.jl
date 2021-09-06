using CUDA
using Adapt

struct CTRNN_Cfg
    delta_t::Float32
    number_neurons::Int64
    clipping_range_min::Float32
    clipping_range_max::Float32
    alpha::Float32
end
function Adapt.adapt_structure(to,ctrnn::CTRNN_Cfg)
    delta_t = Adapt.adapt_structure(to, ctrnn.delta_t)
    number_neurons = Adapt.adapt_structure(to, ctrnn.number_neurons)
    clipping_range_min = Adapt.adapt_structure(to, ctrnn.clipping_range_min)
    clipping_range_max = Adapt.adapt_structure(to, ctrnn.clipping_range_max)
    alpha = Adapt.adapt_structure(to, ctrnn.alpha)
    CTRNN_Cfg(delta_t,number_neurons,clipping_range_min,clipping_range_max,alpha)
end


function get_memory_requirements(number_inputs,number_outputs, brain_cfg::CTRNN_Cfg)
    return sizeof(Float32) * (brain_cfg.number_neurons*(brain_cfg.number_neurons+number_inputs+number_outputs+2) + number_inputs + number_outputs)
end
function brain_initialize(threadID,blockID, V,W,T, individuals)
    number_neurons = size(W,1)
    input_size = size(V,2)
    output_size = size(T,1)
    v_size = input_size * number_neurons
    w_size = number_neurons * number_neurons
    #initialize the brain_masks from the genome and set all Values on the diagonal of W negative
    for i in 1:input_size
        @inbounds V[threadID,i] = individuals[blockID,threadID+((i-1)*number_neurons)]  
    end
    sync_threads()
    for i in 1:number_neurons
        @inbounds W[threadID,i] = individuals[blockID,v_size+(threadID+((i-1)*number_neurons))]
    end
    @inbounds W[threadID,threadID] = -abs(W[threadID,threadID])
    for i in 1:output_size
        @inbounds T[i,threadID] = individuals[blockID,v_size+w_size+(i+((threadID-1)*output_size))]
    end

    return
end

function brain_step(threadID, temp_V, V, W, T, x, input, action,brain_cfg::CTRNN_Cfg)#alpha, delta_t,clipping_range_min,clipping_range_max)
    input_size = size(V,2)
    output_size = size(T,1)

    #V * input multiplication
    V_value = 0.0f0
    for i = 1:input_size 
        @inbounds V_value += V[threadID, i] * input[i] 
    end
    #
    @inbounds temp_V[threadID] = tanh(x[threadID] + V_value) 

    #W * result of Vmult
    W_value = 0.0f0
    for i = 1:brain_cfg.number_neurons 
        @inbounds W_value = W[threadID, i] * temp_V[i] + W_value
    end
    #
    @inbounds x[threadID] += (brain_cfg.delta_t * ((-brain_cfg.alpha * x[threadID]) + W_value))
    @inbounds x[threadID] = clamp(x[threadID],brain_cfg.clipping_range_min,brain_cfg.clipping_range_max)
    sync_threads()

    #T*temp_W matmul:
    if threadID <= output_size
        T_value = 0.0f0
        for i in 1:brain_cfg.number_neurons
            @inbounds T_value = T_value + T[threadID,i] * x[i]
        end
        @inbounds action[threadID] = tanh(T_value)
    end
    #
    return
end

function get_masks_from_brain_state(brain_state::Dict) # what format for brain_state
    v_mask = get(brain_state,"v_mask",1)
    w_mask = get(brain_state,"w_mask",1)
    t_mask = get(brain_state,"t_mask",1)

    return v_mask, w_mask, t_mask
end

function _generate_mask(n :: Int, m :: Int)
    return trues(n,m)
end

function generate_brain_state(input_size,output_size,configuration::Dict)
    v_mask = _generate_mask(configuration["number_neurons"], input_size)
    w_mask = _generate_mask(configuration["number_neurons"], configuration["number_neurons"])
    t_mask = _generate_mask(output_size, configuration["number_neurons"])

    return get_brain_state_from_masks(v_mask,w_mask,t_mask)
end

function get_brain_state_from_masks(v_mask,w_mask,t_mask)
    return Dict("v_mask"=> v_mask,"w_mask"=>w_mask,"t_mask"=>t_mask)
end

function get_free_parameter_usage(brain_state)
    v_mask, w_mask, t_mask = get_masks_from_brain_state(brain_state)
    #count true values
    free_parameters_v = count(v_mask)
    free_parameters_w = count(w_mask)
    free_parameters_t = count(t_mask)

    free_parameters = Dict("V"=>free_parameters_v,"W"=> free_parameters_w,"T"=>free_parameters_t)
    return free_parameters
end

function sum_dict(node)
    sum_ = 0
    for (key,value) in node
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