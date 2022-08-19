using Test

@testset "Summary" begin

    # Optimizers
    include("cma_es.jl")

    # Brains
    include("continuous_time_rnn.jl")
    include("feed_forward_nn.jl")
    include("gate_recurrent_unit_nn.jl")
    include("long_short_term_memory_nn.jl")
    include("elman_network.jl")

    # Environments
    include("collect_points.jl")
    include("dummy_app.jl")

end


println("Finished")