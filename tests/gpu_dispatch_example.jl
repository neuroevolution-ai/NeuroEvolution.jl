
using CUDA;
#using JSON;

#dicttxt = JSON.parsefile("config_example.json")

#x2 = dicttxt["x"]
#y2 = dicttxt["y"]
N = 256*2
x_d = CUDA.fill(convert(Float64,3), N)  # a vector stored on the GPU filled with 1.0 (Float32)
y_d = CUDA.fill(convert(Float64,2), N)  # a vector stored on the GPU filled with 2.0

function myadd(x::Float32, y::Float32)

    return x+2*y
end

function myadd(x::Float64, y::Float64)
    return x+y
end

using Test
function gpu_add3!(y, x)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    z = myadd(y[index], x[index])

    #@cuprintln(blockIdx().x, "   ", threadIdx().x)

    if threadIdx().x == 1
        @cuprintln("y=", y[index], "  x=", x[index], "  z=", z)
    end

    return
end

numblocks = ceil(Int, N/256)

@cuda threads=6 blocks=numblocks gpu_add3!(y_d, x_d);
