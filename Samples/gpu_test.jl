using CUDA
using Adapt

mutable struct Test_Cfg
    A::CuArray
    B::CuArray
end
function Adapt.adapt_structure(to, test::Test_Cfg)
    A = Adapt.adapt_structure(to, test.A)
    B = Adapt.adapt_structure(to, test.B)
    Test_Cfg(A, B)
end
struct Kernel_test_Cfg
    A::Float32
    B::Int32
end
function Adapt.adapt_structure(to, test::Kernel_test_Cfg)
    A = Adapt.adapt_structure(to, test.A)
    B = Adapt.adapt_structure(to, test.B)
    Kernel_test_Cfg(A, B)
end

function kernel_test(test)

    @cuprintln(test.A)
    @cuprintln(test.B)
    return
end
test = Kernel_test_Cfg(2.0f0,3)
 @cuda kernel_test(test)