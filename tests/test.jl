
using CUDA
using BenchmarkTools


x_d = CUDA.fill(1.0f0,50)  # a vector stored on the GPU filled with 1.0 (Float32)
#display(x_d)
#a = fill(x_d,5)
#display(a)


function kernel_func(x::CuDeviceArray)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    #CUDA.nanosleep(10)
    @cuprintln("Index:",x[index])
    return
end

function kernel_interrupting()
    @cuprintln("Am I interrupting?")
    return
end

function map_call(x::CuArray)
    @cuda kernel_func(x)
end

#@cuda kernel_func(x_d)
a = CUDA.fill(1.0f0,10,10)
b = CUDA.fill(2.0f0,10,10)
c = similar(a)

d = CUDA.fill(3.0f0,10,10)
e = CUDA.fill(4.0f0,10,10)
f = similar(d)

function compute(a,b,c)
    mul!(c,a,b)
    broadcast!(sin,c,c)
    synchronize()
    c
end

MAX_THREADS = CUDA.attribute(device(), CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
println(MAX_THREADS)
#@btime @async compute(d,e,f)
