using Luxor

include("../tools/play.jl")
include("../test/environments/collect_points_cpu.jl")


config_environment = OrderedDict()
config_environment["maze_columns"] = 7
config_environment["maze_rows"] = 10
config_environment["maze_cell_size"] = 80
config_environment["agent_radius"] = 12
config_environment["point_radius"] = 8
config_environment["agent_movement_range"] = 10.0
config_environment["use_sensors"] = true
config_environment["reward_per_collected_positive_point"] = 500.0
config_environment["reward_per_collected_negative_point"] = -700.0
config_environment["number_time_steps"] = 500

configuration = CollectPointsCpuCfg(config_environment)

width = configuration.maze_cell_size * configuration.maze_columns
height = configuration.maze_cell_size * configuration.maze_rows
number_iterations = configuration.number_time_steps

env = CollectPointsCpu(configuration)

@play width height number_iterations begin

    step(env)

    # Draw white background
    background("white")

    # Draw agent
    sethue("green")
    circle(env.agent_position_x, env.agent_position_y, configuration.agent_radius, :fill)

    # Draw positive point
    sethue("blue")
    circle(env.point_x, env.point_y, configuration.point_radius, :fill)

    # Draw negative point
    sethue("red")
    circle(env.negative_point_x, env.negative_point_y, configuration.point_radius, :fill)

    # Draw maze
    sethue("black")
    setline(2)
    for cell_x = 1:configuration.maze_columns
        for cell_y = 1:configuration.maze_rows

            x_left, x_right, y_top, y_bottom = get_coordinates_maze_cell(cell_x, cell_y, configuration.maze_cell_size)

            cell = env.maze[cell_x, cell_y]

            # Draw walls
            if cell.walls[North] == true
                line(Point(x_left, y_top), Point(x_right, y_top), :stroke)
            end

            if cell.walls[South] == true
                line(Point(x_left, y_bottom), Point(x_right, y_bottom), :stroke)
            end

            if cell.walls[East] == true
                line(Point(x_right, y_top), Point(x_right, y_bottom), :stroke)
            end

            if cell.walls[West] == true
                line(Point(x_left, y_top), Point(x_left, y_bottom), :stroke)
            end

        end
    end

    sleep(0.05)

end

println("Finished")