using Test

@testset "Summary" begin

include("cma_es.jl")

include("continuous_time_rnn.jl")
include("feed_forward_nn.jl")
include("gated_recurrent_unit_nn.jl")
include("long_short_term_memory_nn.jl")
include("elman_network.jl")

include("collect_points.jl")
include("dummy_app.jl")

end