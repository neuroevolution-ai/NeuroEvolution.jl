using Test
using Random
include("D:/NeuroEvolution.jl/environments/collect_points_env.jl")


function make_maze_kernel(maze,env_cfg,env_seed)
    if threadIdx().x == 1
        Random.seed!(Random.default_rng(threadIdx().x),env_seed)
    end
    create_maze(maze,env_cfg,0)
    return
end

@testset "Maze" begin
env_seed = 100
maze_columns = 5
maze_rows = 5
maze = CUDA.fill(0,(maze_rows,maze_columns,4))
env_cfg = Collect_Points_Env_Cfg(maze_columns,maze_rows,80,12,8,10.0f0,500.00f0,-700.00f0,1000,10,2)
@cuda shmem=sizeof(Int32) * (maze_columns * maze_rows * 6 + 10) make_maze_kernel(maze,env_cfg,env_seed)
CUDA.synchronize()
maze_cpu = Array(maze)
#test if legitimate maze
display(maze_cpu)

end