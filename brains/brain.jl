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

#function read_matrix_from_genome(args)
#    body
#end

#function get_activation_function(args)
#    body
#end

function get_individual_size(brain_type, input_size:: Int, output_size:: Int, configuration:: Dict, brain_state:: Dict)
    #uses context information to calculate the required number of free parameter needed to construct an individual of this class
end

#=
function tanh(x::CuArray)
end
=#
function sum_dict(node)
    sum_ = 0
end


function get_free_parameter_usage(brain_type:: CTRNNBRAIN, input_size:: Int, output_size:: Int, configuration:: Dict, brain_state:: Dict) #do Brain typoe be ENUM probably
    body
end

#=
inititalize fucntion fr a specific Type of brain
input_size
output_size
individual
configuration
brain_state
=#
function inititalize(input_size :: Int, output_size :: Int, individual :: Array, configuration :: Dict, brain_state :: Dict)

#get necessary info from congif file
brain_type = configuration['type']
delta_t = configuration['delta_t']
number_neurons = configuration['number_neurons']
differential_equation = configuration['differential_equation']
clipping_range_min = configuration['clipping_range_min']
clipping_range_max = configuration['clipping_range_max']
set_principle_diagonal_elements_of_W_negative = configuration['set_principle_diagonal_elements_of_W_negative']
alpha = configuration['alpha']
#

#set sizes for masks
v_size = input_size * brain_state.number_neurons
w_size = brain_state.number_neurons * brain_state.number_neurons
t_size = brain_state.number_neurons * output_size
index = v_size + w_size + t_size
#


#inititalize the masks
V = CuArray{CuArray}[[element] for element in individual[0:v_size]]
W =
T =
#

#reshape the masks to correct dimensions
V = reshape(V,number_neurons,input_size)
W = reshape(W,number_neurons,number_neurons)
T = reshape(T,output_size,number_neurons)
#

if(set_principle_diagonal_elements_of_W_negative)
    for j in 1:number_neurons
        W[j][j] = -abs(W[j][j])
    end
end
x0 = CUDA.fill(0, number_neurons)

x = x0

brain_config_array = CuArray{Any}[]
end

#=step functions
config_array: represents the current state and configuration of the brain. Has format:
input_array: represents current state of environment as given by either the environments step() or reset() function. Has format =
=#
function step(config_array::Any[], input_array:: Array)
    #assert input_array is 1-dimensional

    #differential Equations:
    if config_array.differential_equation == 'separated'
        dx_dt = -config_array.alpha * config_array.x + config_array.W.dot(tanh(config_array.x)) + config_array.V.dot(input_array) #TODO numpy tanh function find replacement
    elseif config_array.differential_equation == 'original'
        dx_dt = -config_array.alpha * config_array.x + config_array.W.dot(tanh(config_array.x) + config_array.V.dot(input_array)) #TODO numpy tanh function find replacement
    else
        #throw RuntimeError
    end

    # Euler forward discretization
    config_array.x = config_array.x + config_array.delta_t * dx_dt

    # Clip y to state boundaries
    config_array.x = clip(config_array.x, -config_array.clipping_range, +config_array.clipping_range) #TODO numpy clip() function find replacement

    # Calculate outputs
    y = tanh(config_array.T.dot(config_array.x)) #TODO tanh() is numpy function find replacement

    #assert y is 1-dimensional
    return y
end


function get_masks_from_brain_state(brain_state :: Dict) # what format for brain_state
    v_mask = brain_state['v_mask']
    w_mask = brain_state['w_mask']
    t_mask = brain_state['t_mask']

    return v_mask, w_mask, t_mask
end

function _generate_mask(n :: Int, m :: Int, keep_main_diagonal= false :: boolean)
    mask =
    return mask
end
