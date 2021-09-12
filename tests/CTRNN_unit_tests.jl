using Test
using CUDA
using Random
using JSON

include("D:/NeuroEvolution.jl/brains/continuous_time_rnn.jl")
include("D:/NeuroEvolution.jl/optimizers/optimizer.jl")

function kernel_test_brain_step(V,W,T,x,temp_V,input,action,brain_cfg)
	brain_step(threadIdx().x,temp_V,V,W,T,x,input,action,brain_cfg)
	return
end
function kernel_test_initialize(V,W,T,individuals)
	bx = blockIdx().x


	brain_initialize(threadIdx().x,1,V,W,T,individuals)
	return
end

@testset "Brain step" begin
	input_size = 10
	number_neurons = 50
	output_size = 2
	for i in 1:100
	V = randn((number_neurons,input_size))
	W = randn((number_neurons,number_neurons))
	T = randn((output_size,number_neurons))
	input = randn(input_size)
	x = fill(0.0f0,number_neurons)
	temp_V = CUDA.fill(0.0f0,number_neurons)
	input = randn(input_size)

	delta_t = 0.05
	clipping_range = 1.0
	alpha = 0.0
	action = CUDA.fill(0.0f0,output_size)
	brain_cfg = CTRNN_Cfg(delta_t,number_neurons,original,-clipping_range,clipping_range,alpha)
	@cuda threads=50 kernel_test_brain_step(CuArray(V),CuArray(W),CuArray(T),CuArray(x),temp_V,CuArray(input),action,brain_cfg)
	temp = x + delta_t.*((-alpha.*x)+W*map(tanh,(x+(V*input))))
	temp = clamp.(temp,-clipping_range,clipping_range)
	C_cpu = map(tanh,T*temp)

	CUDA.synchronize()
	@test Array(action) ≈ C_cpu atol=0.00001
	end
end

@testset "Brain initialize" begin
	configuration = JSON.parsefile("D:/NeuroEvolution.jl/configurations/CMA_ES_Deap_CTRNN_Dense.json")
	brain = configuration["brain"]
	optimizer = configuration["optimizer"]
	number_inputs = 10
    number_outputs = 2
	number_neurons = 50
	number_individuals = 112

	brain_state = generate_brain_state(number_inputs,number_outputs,brain)
    free_parameters = get_individual_size(brain_state)

    optimizer = inititalize_optimizer(free_parameters,optimizer)
	genomes = ask(optimizer)


	individuals = fill(0.0f0,number_individuals,free_parameters)
	for i in 1:number_individuals
		for j in 1:free_parameters
			individuals[i,j] = (genomes[i])[j]
		end
	end

	#display(a)
	#display(reshape(a,(number_neurons,number_inputs)))
	v_size = number_neurons * number_inputs
	w_size = number_neurons * number_neurons
	t_size = number_outputs * number_neurons

	V = fill(0.0f0,(number_neurons,number_inputs))
	W = fill(0.0f0,number_neurons,number_neurons)
	T = fill(0.0f0,number_outputs,number_neurons)

	V_gpu=CuArray(V)
	W_gpu=CuArray(W)
	T_gpu =CuArray(T)


	@cuda threads=number_neurons kernel_test_initialize(V_gpu,W_gpu,T_gpu,CuArray(individuals))
	a = reshape(view(individuals,1,1:v_size),(number_neurons,number_inputs))
	b = reshape(view(individuals,1,v_size+1:v_size+w_size),(number_neurons,number_neurons))
	c = reshape(view(individuals,1,v_size+w_size+1:v_size+w_size+t_size),(number_outputs,number_neurons))

	for i in 1:number_neurons
		b[i,i] = -abs(b[i,i])
	end
	CUDA.synchronize()

	@test a == Array(V_gpu)
	@test b == Array(W_gpu)
	@test c == Array(T_gpu)

end