using CUDA
using Adapt
using JSON
using StructArrays


struct Interpolate{CuArray}
    xs::CuArray
    ys::CuArray
end
#=
function (itp::Interpolate)(x)
    i = searchsortedfirst(itp.xs, x)
    i = clamp(i, firstindex(itp.ys), lastindex(itp.ys))
    @inbounds itp.ys[i]
end
=#
function Adapt.adapt_structure(to, itp::Interpolate)
    xs = Adapt.adapt_structure(to, itp.xs)
    ys = Adapt.adapt_structure(to, itp.ys)
    Interpolate(xs, ys)
end


function kernel(itp::Interpolate)
    @cuprintln("1")
    return
end

#@cuda kernel(itp)


struct TestStruct{A}
    z::A
    x::A
    y::A
end
Adapt.Adapt.@adapt_structure TestStruct

struct struct2{A}
    b::Bool
    a::A
end
Adapt.Adapt.@adapt_structure struct2

function kernel(test::struct2)
    @cushow(test.b)
    @cushow(test.a[1].x)
    return
end

struct pt
    x::Float64
    y::Float64
end

function initpt!(a)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    for i in index:stride:length(a)
        a[i] = pt(1.0, 2.0)
    end
    return nothing
end

ndata = 10240
thdx  = 256
numblocks = ceil(Int, ndata/thdx)

println("Starting TEST: Memory allocation and transfer")
#test = StructArray{TestStruct}((1,2))
#display(test)
cua1 = StructArray{TestStruct}((zeros(1),ones(1), ones(1)))
#display(cua1)
#test2 = replace_storage(CuArray,test)
#display(test2)
#@cuda kernel(test2)
s1 = struct2(false,cua1)

cua2 = replace_storage(CuArray, cua1)
s2 = struct2(false,cua2)
display(s2)
@cuda kernel(s2)
