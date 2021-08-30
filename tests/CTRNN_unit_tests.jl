using Test
using CUDA
include("C:/Users/Hendrik/Source/Repos/NeuroEvolution.jl/brains/continuous_time_rnn.jl")


@testset "Initialize Brain" begin
	@test 2+3== 5
end

@testset "Brain step" begin
input_size=
number_neurons =
output_size
V = rand()
W = rand()
T = rand()
x = rand() # or just zeroes
input = rand()

delta_t =
clipping_range =
alpha = 
action = fill(0.0f0,output_size)
C_cpu = 
end