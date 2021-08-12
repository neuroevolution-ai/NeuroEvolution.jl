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


struct TestStruct{A}
    x::A
    y::A
    z::Int
end
function Adapt.adapt_structure(to, test::TestStruct)
    x = Adapt.adapt_structure(to, test.x)
    y = Adapt.adapt_structure(to, test.y)
    z = Adapt.adapt_structure(to, test.z)
    TestStruct(x, y, z)
end
function kernel(test::TestStruct)
    
    test2 = TestStruct(test.x,test.y, test.z+1)
    test = test2
    @cuprintln(test.x[1])
    @cuprintln(test.y[1])
    @cuprintln(test.z)

    return
end

x = CUDA.fill(1.0,10)
y = CUDA.fill(2.0,10)
itp = Interpolate(x,y)
#itp = Interpolate(CuArray(xs_cpu), CuArray(ys_cpu))
#display(itp)
#pts = CuArray(pts_cpu);
#display(pts)
#result = itp.(pts)
#display(result)



#=
which structs are necessary:
-Environment
-Brain
-EpisodeRunner



=#
