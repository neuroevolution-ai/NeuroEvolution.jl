using CUDA
using LinearAlgebra
#using Mathematics

#CUDA.versioninfo()


number_neurons = 50
input_size = 6
output_size = 2
alpha = 0.0
x = CUDA.zeros(number_neurons)
output = CUDA.zeros(number_neurons)
W = CUDA.fill(1.0,number_neurons,number_neurons)
V = CUDA.fill(1.0,number_neurons, input_size)
u = CUDA.fill(1.0, input_size)

function kernel_dif_equation()#output ,alpha ,x ,W ,V ,u )
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    cuprintln(index)
    return
end
@cuda launch=false threads=50 blocks=1 kernel_dif_equation()


x = zeros(number_neurons)
output = zeros(number_neurons)
#display(x)
#display(output)
W = fill(1.0,number_neurons,number_neurons)
V = fill(1.0,number_neurons, input_size)
T = fill(1.0,output_size,number_neurons)
u = fill(1.0, input_size)

function differential_equation(alpha, x, W, V, u)
    V_dot = map(tanh, (x+ [sumprod = dot(row,u) for row in eachrow(V)]))
    W_dot = (x.+ -alpha) + [sumprod = dot(row,V_dot) for row in eachrow(W)]
    return W_dot
end

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
#=
function kernel_Euler_forward_discretization(args)

end

function kernel_clip(args)

end

function kernel_calc_outputs(args)

end
=#
