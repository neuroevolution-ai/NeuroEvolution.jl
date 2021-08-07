using CUDA
using Adapt
using JSON


struct TestCfg
    a::CuArray
end

#configuration = JSON.parsefile("configurations/CMA_ES_Deap_CTRNN_Dense.json")
#config = TrainingCfg(number_generations, number_validation_runs, number_rounds,maximum_env_seed,environment,brain,optimizer, -1)


b = fill(1.0,10,10)
#display(b)
#display(test.a)

function Adapt.adapt_structure(to, test::TestCfg)
    a = Adapt.adapt_structure(to, test.a)
    TestCfg(a)
end

test = TestCfg(b)

function kernel_a(test)
    @cuprintln(test.a[1])
    return
end


@cuda kernel_a(test)
