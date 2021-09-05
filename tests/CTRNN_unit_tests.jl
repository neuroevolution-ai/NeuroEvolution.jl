using Test
using CUDA
using Random

include("D:/NeuroEvolution.jl/brains/continuous_time_rnn.jl")


function kernel_test_brain_step(V,W,T,x,temp_V,input,action,brain_cfg)
	brain_step(threadIdx().x,temp_V,V,W,T,x,input,action,brain_cfg)
	return
end
function kernel_test_initialize(V,W,T,individuals)



	brain_initialize(threadIdx().x,blockIdx().x,V,W,T,individuals)
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
	brain_cfg = CTRNN_Cfg(delta_t,number_neurons,-clipping_range,clipping_range,alpha)
	@cuda threads=50 kernel_test_brain_step(CuArray(V),CuArray(W),CuArray(T),CuArray(x),temp_V,CuArray(input),action,brain_cfg)
	temp = x + delta_t.*((-alpha.*x)+W*map(tanh,(x+(V*input))))
	temp = clamp.(temp,-clipping_range,clipping_range)
	C_cpu = map(tanh,T*temp)

	CUDA.synchronize()
	@test Array(action) ≈ C_cpu atol=0.000001
	end
end