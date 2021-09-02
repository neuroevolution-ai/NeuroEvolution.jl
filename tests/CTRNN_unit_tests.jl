using Test
using CUDA
include("D:/NeuroEvolution.jl/NeuroEvolution.jl/brains/continuous_time_rnn.jl")

function test_kernel()
	brain_step(threadIdx().x,)
end

@testset "Initialize Brain" begin
	@test 2+3== 5
end

@testset "Brain step" begin
	input_size = 10
	number_neurons = 50
	output_size = 2
	V = rand(Float32,(number_neurons,input_size))
	W = rand(Float32,(number_neurons,number_neurons))
	T = rand(Float32,(output_size,number_neurons))
	x = zeroes(number_neurons) # or just zeroes
	input = rand(Float32,input_size)

	delta_t = 0.05
	clipping_range = 1.0
	alpha = 0.0
	action = CUDA.fill(0.0f0,output_size)
	@cuda 
	temp = x + delta_t.*((-alpha.*x)+W*map(tanh,(x+(V*u))))
	temp = clamp(temp,-clipping_range,clipping_range)
	C_cpu = map(tanh,T*temp)

	CUDA.synchronize()
	@test action == C_cpu
end