using Luxor

include("../tools/play.jl")
include("../test/environments/create_maze.jl")

maze_columns = 15
maze_rows = 10
maze_cell_size = 80

width = maze_cell_size * maze_columns
height = maze_cell_size * maze_rows
number_iterations = 100

maze = make_maze(maze_columns, maze_rows)

@play width height number_iterations begin

    # White white background
    background("white")

    # Draw maze
    sethue("black")
    setline(2)

    for cell_x = 1:maze_columns
        for cell_y = 1:maze_rows

            x_left, x_right, y_top, y_bottom = get_coordinates_maze_cell(cell_x, cell_y, maze_cell_size)

            cell = maze[cell_x, cell_y]

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

end

println("Finished")