using CUDA
using BenchmarkTools
#using LinearAlgebra
#using Mathematics
#include("gpu_dispatch_example.jl")

#CUDA.versioninfo()
function map_test(a)
    sleep(5)
    print("waited")
    return a+1
end
a = fill(10,5)
display(a)
map(map_test,a)
display(a)

number_neurons = 50
input_size = 100
output_size = 2
alpha = 0.0
#=
x = CUDA.zeros(number_neurons)
output = CUDA.zeros(number_neurons)
W = CUDA.fill(1.0,number_neurons,number_neurons)
V = CUDA.fill(1.0,number_neurons, input_size)
T = CUDA.fill(1.0,output_size,number_neurons)
u = CUDA.fill(1.0, input_size)
=#
#display(x)
function kernel_dif_equation(alpha,x,output,W,V,u)#output ,alpha ,x ,W ,V ,u )
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    #@cuprintln("Index: ", index)
    if index == 1
        output = V*u
    end
    return
end
a5 = CUDA.fill(1.0,2,2)
b5 = CUDA.fill(2.0,2,2)
c5 = CUDA.similar(a5)
env_seed = 5.0
function test_add(a :: CuArray, b :: CuArray, c :: CuArray)
    c = a + b
    return
end

function step(seed::Int)
    return seed+1
end

function kernel_eval_fitness(env_seed,brain_config, action, dx_dt)#genome,number_rounds,env_seed)
    #get Index of current thread
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    #
    if threadIdx().x == 1
        #initialize brain for entire block
        fitness_total = 0
    end

    for i in range 1:number_rounds

        if threadIdx().x == 1
            fitness_current = 0
            done = false
            #initialize environment with environment seed
            #get starting overservation from environment
            #reset brain.... necessary?
        end

        while !done
            #brain step
            dx_dt[1] = diff_eq(brain_config,ob,threadIdx().x)
            if threadIdx().x == 1

            end
            action[threadIdx().x] = brain_step(brain_config,ob,threadIdx().x)
            #brain step
            sync_threads()
            #environment step
            if threadIdx().x == 1
                ob, rew, done, info = env_step(environment_config,action)
                fitness_current += rew
            end
            #environment step
            fitness_current += rew
            sync_threads()
        end
        fitness_total += fitness_current
    end
    return
end
#@cuda threads=10 blocks=2 kernel_eval_fitness(env_seed)

function step(seed::Int)
    return seed+1
end

function test_func()
    return 1,2
end
function kernel_step(array_test)
    @cuprintln()

    @cuprintln()
    return
end



function make_kernel_call(a,b)
    @cuda threads=5 blocks=2 kernel_step(a,b)

end



number_neurons = 100
input_size = 2
output_size = 1000000
alpha = 0.0

x = CUDA.zeros(number_neurons)
output = CUDA.zeros(number_neurons)
#display(x)
#display(output)
W = CUDA.fill(1.0,number_neurons,number_neurons)
V = CUDA.fill(1.0,number_neurons, input_size)
T = CUDA.fill(1.0,output_size,number_neurons)
u = CUDA.fill(1.0, input_size)

#@btime dx_dt = (x.+ -alpha) + W*(map(tanh,V*u))
#result = Array(dx_dt)
#display(result)
x = nothing
output = nothing
#display(x)
#display(output)
W = nothing
V = nothing
T = nothing
u = nothing

x = zeros(number_neurons)
output = zeros(number_neurons)
#display(x)
#display(output)
W = fill(1.0,number_neurons,number_neurons)
V = fill(1.0,number_neurons, input_size)
T = fill(1.0,output_size,number_neurons)
u = fill(1.0, input_size)

function differential_equation(alpha, x, W, V, u)
    #V_dot = map(tanh, (x+ [sumprod = dot(row,u) for row in eachrow(V)]))
    #W_dot = (x.+ -alpha) + [sumprod = dot(row,V_dot) for row in eachrow(W)]
    dx_dt = (x.+ -alpha) + W*(map(tanh,V*u))
    return dx_dt
end

#@btime differential_equation(alpha,x,W,V,u)

#=
function euler_forward_discretization(x,delta_t,dif_equation)
    return x + delta_t * dif_equation
end

function clip(x,clip_range)
    return clamp(x, -clip_range, +clip_range)
end

function calc_outputs(T,x)
    T_dot = [sumprod = dot(row,x) for row in eachrow(T)]
    return map(tanh,T_dot)
end

#TODO write dot function explicit
#display(differential_equation(alpha,x,W,V,u))
function kernel_Euler_forward_discretization(args)

end

function kernel_clip(args)

end

function kernel_calc_outputs(args)

end
=#
