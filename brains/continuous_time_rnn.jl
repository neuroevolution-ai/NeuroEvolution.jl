import Test
function brain_initialize(threadID,blockID, V, W, T, individuals)

    number_neurons = size(W,1)
    input_size = size(V,2)
    output_size = size(T,1)
    v_size = input_size * number_neurons
    w_size = number_neurons * number_neurons
    #Fill V,W,T with genome data
    #####################################################
    for i in 1:input_size
        @inbounds V[threadID,i] = individuals[blockID,i+((threadID-1)*input_size)]  
    end
    sync_threads()
    for i in 1:number_neurons
        @inbounds W[threadID,i] = individuals[blockID,v_size+(i+((threadID-1)*number_neurons))]
    end
    for i in 1:output_size
        @inbounds T[i,threadID] = individuals[blockID,v_size+w_size+(threadID+((i-1)*number_neurons))]
    end
    #####################################################
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


#Unit tests
##################################################################
#=
using Test
#brain_initialize():
@testset "Initialize Tests" begin
    number_individuals = 112
    input_size = 5
    number_neurons = 25
    output_size = 2
    blockID = 1
    threadID = 1
    individuals = rand(Float32,(number_individuals,number_neurons^2+input_size*number_neurons+output_size*number_neurons))

#Initialize V

#Initialize W

#Initialize T
end
=#

#=
@testset "Step Tests" begin
V =
W = 
T = 
alpha =
x1 =
x2 =
delta_t = 
input =

#Comparison
dx_dt = -alpha * x + W*(tanh(x + (C*input)))
x1 += delta_t * dx_dt
x1 = clamp(x,-clipping_range,clipping_range)
output = tanh(T*x)

#
brain_step
end
=#
