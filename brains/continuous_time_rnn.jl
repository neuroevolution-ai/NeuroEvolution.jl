using CUDA

function init_mem(number_neurons,input_size,output_size,offset)
    v_size = input_size * number_neurons
    w_size = number_neurons * number_neurons
    V = @cuDynamicSharedMem(Float32,(number_neurons,input_size),offset)
    W = @cuDynamicSharedMem(Float32,(number_neurons,number_neurons),offset + sizeof(V))
    T = @cuDynamicSharedMem(Float32,(output_size,number_neurons),offset + sizeof(V)+sizeof(W))
    offset += sizeof(V)+sizeof(W)+sizeof(T)
    return V,W,T,offset
end
function brain_initialize(threadID,blockID, V,W,T, individuals)
    number_neurons = size(W,1)
    input_size = size(V,2)
    output_size = size(T,1)
    v_size = input_size * number_neurons
    w_size = number_neurons * number_neurons

    for i in 1:input_size
        @inbounds V[threadID,i] = individuals[blockID,i+((threadID-1)*input_size)]  
    end
    sync_threads()
    for i in 1:number_neurons
        @inbounds W[threadID,i] = individuals[blockID,v_size+(i+((threadID-1)*number_neurons))]
    end
    @inbounds W[threadID,threadID] = -abs(W[threadID,threadID])
    for i in 1:output_size
        @inbounds T[i,threadID] = individuals[blockID,v_size+w_size+(threadID+((i-1)*number_neurons))]
    end

    return
end

function brain_step(threadID, temp_V, V, W, T, x, input, action,alpha, delta_t,clipping_range)
    input_size = size(V,2)
    output_size = size(T,1)
    number_neurons = size(W,1)

            #V*input matmul:
            V_value = 0.0f0
            for i = 1:input_size 
                @inbounds V_value += V[threadID, i] * input[i] 

            end

            @inbounds temp_V[threadID] = tanh(x[threadID] + V_value) 
            #W*temp_V matmul:
            W_value = 0.0f0
            for i = 1:number_neurons 
                @inbounds W_value = W[threadID, i] * temp_V[i] + W_value
            end
            @inbounds x[threadID] += (delta_t * ((-alpha * x[threadID]) + W_value))
            @inbounds x[threadID] = clamp(x[threadID],-clipping_range,clipping_range)
            sync_threads()

            #T*temp_W matmul:
            
            if threadID <= output_size
                T_value = 0.0f0
                for i in 1:number_neurons
                   @inbounds T_value = T_value + T[threadID,i] * x[i]
                end
                @inbounds action[threadID] = tanh(T_value)
            end
            
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
    #config = ContinuousTimeRNNCfg(value for (key,value) in configuration)

    v_mask = _generate_mask(configuration["number_neurons"], input_size)
    w_mask = _generate_mask(configuration["number_neurons"], configuration["number_neurons"])
    t_mask = _generate_mask(output_size, configuration["number_neurons"])

    return get_brain_state_from_masks(v_mask,w_mask,t_mask)
end

function get_brain_state_from_masks(v_mask,w_mask,t_mask)
    return Dict("v_mask"=> v_mask,"w_mask"=>w_mask,"t_mask"=>t_mask)
end

function get_free_parameter_usage(input_size,output_size,configuration,brain_state)
    v_mask, w_mask, t_mask = get_masks_from_brain_state(brain_state)

        #count true values
    free_parameters_v = count(v_mask)
    free_parameters_w = count(w_mask)
    free_parameters_t = count(t_mask)

    free_parameters = Dict("V"=>free_parameters_v,"W"=> free_parameters_w,"T"=>free_parameters_t)
    #=
    if(configuration["optimize_x0"])
        free_parameters["x_0"] = configuration["number_neurons"]
    end
    =#
    return free_parameters
end

function sum_dict(node)
    sum_ = 0
    for (key,value) in node
        if(value isa Dict)
            sum_ += sum_dict(value)
        else
            sum_ += value
        end
        
    end
    return sum_
end

function get_individual_size(input_size,output_size,configuration,brain_state)
    usage_dict = get_free_parameter_usage(input_size,output_size,configuration, brain_state)
    return sum_dict(usage_dict)
end