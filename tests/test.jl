using CUDA
using LinearAlgebra

N = 10
x_d = CUDA.fill(1.0f0, N)  # a vector stored on the GPU filled with 1.0 (Float32)
y_d = CUDA.fill(2.0f0, N)  # a vector stored on the GPU filled with 2.0

y_d .+= x_d
@test all(Array(y_d) .== 3.0f0)
