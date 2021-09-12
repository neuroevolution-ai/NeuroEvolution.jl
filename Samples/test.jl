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
    x::Int
    y::A
end
Adapt.Adapt.@adapt_structure TestStruct
function create_struct(a,b)
    test_struct = TestStruct(a,b)
    return test_struct
end

function kernel(test)
    @cushow(test.y[1])
    test.y[1] = 2
    @cushow(test.y[1])
    
    return
end

x = 5
y = CUDA.fill(1,5)
test = TestStruct(x,y)
@cuda kernel(test)
CUDA.synchronize()

