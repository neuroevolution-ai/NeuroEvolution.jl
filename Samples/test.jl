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
    x::A
    y::A
end
Adapt.Adapt.@adapt_structure TestStruct
function create_struct(a,b)
    test_struct = TestStruct(a,b)
    return test_struct
end

function kernel(test)

    return
end

a = TestStruct(3,4)
array1 = CUDA.fill(1,5)
array2 = CUDA.fill(2,5)
b = TestStruct(array1,array2)

s = StructArray{TestStruct}(undef,2)
s[1] = a
s[2] = b
@cuda kernel(b)
CUDA.synchronize()