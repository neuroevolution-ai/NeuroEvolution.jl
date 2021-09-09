using CUDA
using Adapt
using JSON


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


struct TestStruct
    x::Int
    y::Int
end
function Adapt.adapt_structure(to, test::TestStruct)
    x = Adapt.adapt_structure(to, test.x)
    y = Adapt.adapt_structure(to, test.y)
    TestStruct(x, y)
end

function create_struct(a,b)
    test_struct = TestStruct(a,b)
    return test_struct
end

function kernel()
    a = 10
    b = 20
    test = create_struct(a,b)
    @cuprintln(test.a)
    @cuprintln(test.b)
    return
end

@device_code_warntype @cuda kernel()


