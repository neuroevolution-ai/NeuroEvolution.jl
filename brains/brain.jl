#module brain
#=
registered_brain_classes = Dict()

include

#continuous_time_rnn.py implementieren
function get_brain_class(brain_class_name:: String)
    if brain_class_name in keys(registered_brain_classes)
        return registered_brain_classes[brain_class_name]
    else
        return 5#RuntimeError("No valid brain")
    end
end


registered_brain_classes["CTRNN"] = "ContinuousTimeRNN"
=#

using CUDA
using Adapt

struct ContinuousTimeRNNCfg
    number_neurons::Int
end

struct ContinuousTimeRNN
    V::CuArray
    W::CuArray
    T::CuArray
    x::CuArray
end
function Adapt.adapt_structure(to, ctrnn::ContinuousTimeRNN)
    V = Adapt.adapt_structure(to, ctrnn.V)
    W = Adapt.adapt_structure(to, ctrnn.W)
    T = Adapt.adapt_structure(to, ctrnn.T)
    x = Adapt.adapt_structure(to, ctrnn.x)
    ContinuousTimeRNN(V, W, T, x)
end

function get_individual_size(brain_type, input_size:: Int, output_size:: Int, configuration:: Dict, brain_state:: Dict)
    #uses context information to calculate the required number of free parameter needed to construct an individual of this class
end

#=
inititalize fucntion fr a specific Type of brain
input_size
output_size
individual
configuration
brain_state
=#
function inititalize(input_size :: Int, output_size :: Int, individual, number_neurons)#, brain_state :: Dict)


number_neurons = 50 #configuration["number_neurons"]

#

v_size = input_size * number_neurons
w_size = number_neurons * number_neurons
t_size = number_neurons * output_size
index = v_size + w_size + t_size 


#inititalize the masks
v_cpu = view(individual,1:v_size)
w_cpu = view(individual,v_size+1:v_size+w_size)
t_cpu = view(individual,v_size+w_size+1:v_size+w_size+t_size)

#reshape the masks to correct dimensions
v_cpu = reshape(v_cpu,number_neurons,input_size)
w_cpu = reshape(w_cpu,number_neurons,number_neurons)
t_cpu = reshape(t_cpu,output_size,number_neurons)

#if(set_principle_diagonal_elements_of_W_negative)
#    w_cpu[diagind(w_cpu)] .= map(-,map(abs,w_cpu[diagind(w_cpu)]))
#end

#load masks onto GPU
V_gpu = CuArray(v_cpu)
W_gpu = CuArray(w_cpu)
T_gpu = CuArray(t_cpu)

x = CUDA.fill(0.0f0, number_neurons)

return ContinuousTimeRNN(V_gpu,W_gpu,T_gpu,x)


end

function reset(x)
    return x = CUDA.fill(0.0f0,length(x))
end

#=step functions
config_array: represents the current state and configuration of the brain. Has format:
input_array: represents current state of environment as given by either the environments step() or reset() function. Has format =
=#
#TODO: Make sure everything is a FLoat32
function step(CTRNN::ContinuousTimeRNN,x, input_array::CuArray)
    #assert input_array is 1-dimensional
    alpha = 0.0
    delta_t = 0.05
    clipping_range = 1.0
    #differential Equations:

    dx_dt = (x .+ -alpha) + CTRNN.W*(map(tanh,CTRNN.V * input_array)) #TODO numpy tanh function find replacement

    # Euler forward discretization
    x = x .+ delta_t * dx_dt
    # Clip y to state boundaries
    x = broadcast(clamp,x, -clipping_range, +clipping_range) #TODO numpy clip() function find replacement  #figure out how to do clamp
    # Calculate outputs
    y = map(tanh,CTRNN.T * x) #TODO tanh() is numpy function find replacement
    #display(y)
    #assert y is 1-dimensional
    #CUDA.unsafe_free!(x)
    return y,x
end

function step(v_mask,w_mask,t_mask,x,input_array)
    #assert input_array is 1-dimensional
    alpha = 0.0
    delta_t = 0.05
    clipping_range = 1.0
    #differential Equations:

    dx_dt = (x .+ -alpha) + w_mask*(map(tanh,v_mask * input_array)) #TODO numpy tanh function find replacement

    # Euler forward discretization
    x = x .+ delta_t * dx_dt
    # Clip y to state boundaries
    x = broadcast(clamp,x, -clipping_range, +clipping_range) #TODO numpy clip() function find replacement  #figure out how to do clamp
    # Calculate outputs
    y = map(tanh,t_mask * x) #TODO tanh() is numpy function find replacement
    #display(y)
    #assert y is 1-dimensional
    #CUDA.unsafe_free!(x)
    return y,x
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