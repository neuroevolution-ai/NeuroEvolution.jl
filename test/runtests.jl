using Test

@testset "Summary" begin

include("cma_es.jl")
include("collect_points.jl")
include("continuous_time_rnn.jl")
include("dummy_app.jl")

end