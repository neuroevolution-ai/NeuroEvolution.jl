# Docu for profiling Cuda Code in Julia: https://cuda.juliagpu.org/stable/development/profiling/

#using BenchmarkTools
using CUDA

a = rand(1024,1024,100);

ad = CUDA.rand(1024,1024,100);

@time sin.(a)

sin.(ad);

println("Finished")