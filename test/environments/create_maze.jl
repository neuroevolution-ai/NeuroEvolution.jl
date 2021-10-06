# Create a maze using the depth-first algorithm described at
# https://scipython.com/blog/making-a-maze/
# Christian Hill, April 2017.

@enum Direction North South East West

# A cell in the maze.
# A maze "Cell" is a point in the grid which may be surrounded by walls to
# the north, east, south or west.
struct Cell
    x::Int
    y::Int
    walls::Dict

    # Initialize the cell at (x,y). At first it is surrounded by walls.
    function Cell(x, y)
        walls = Dict(North => true, South => true, East => true, West => true)
        new(x, y, walls)
    end
end

# Does this cell still have all its walls?
function has_all_walls(cell::Cell)
    return all(values(cell.walls))
end

# Knock down the wall between cells self and other.
function knock_down_wall(cell::Cell, other::Cell, wall::Direction)

    # A wall separates a pair of cells in the N-S or W-E directions.
    wall_pairs = Dict(North => South, South => North, East => West, West => East)

    cell.walls[wall] = false
    other.walls[wall_pairs[wall]] = false
end

# Return a list of unvisited neighbours to cell
function find_valid_neighbours(maze::Matrix{Cell}, cell::Cell)

    delta = Dict(West => [-1 0], East => [1 0], South => [0 1], North => [0 -1])

    neighbours = []

    nx, ny = size(maze)

    for (direction, (dx, dy)) in delta

        x2, y2 = cell.x + dx, cell.y + dy

        if (1 ≤ x2 ≤ nx) && (1 ≤ y2 ≤ ny)
            neighbour = maze[x2, y2]
            if has_all_walls(neighbour)
                push!(neighbours, (direction, neighbour))
            end
        end
    end

    return neighbours
end

function make_maze(nx::Int, ny::Int)

    # Total number of cells.
    n = nx * ny

    maze = [Cell(x, y) for x = 1:nx, y = 1:ny]
    cell_stack = []

    # Total number of visited cells during maze construction.
    nv = 1

    current_cell = maze[1, 1]

    while nv < n
        neighbours = find_valid_neighbours(maze, current_cell)

        if isempty(neighbours)
            # We've reached a dead end: backtrack.
            current_cell = pop!(cell_stack)
            continue
        end

        # Choose a random neighbouring cell and move to it.
        k = rand(1:length(neighbours))
        direction, next_cell = neighbours[k]
        knock_down_wall(current_cell, next_cell, direction)
        push!(cell_stack, current_cell)
        current_cell = next_cell
        nv += 1
    end

    return maze
end

function get_coordinates_maze_cell(cell_x::Int, cell_y::Int, maze_cell_size::Int)

    x_left = maze_cell_size * (cell_x - 1)
    x_right = maze_cell_size * cell_x
    y_top = maze_cell_size * (cell_y - 1)
    y_bottom = maze_cell_size * cell_y

    return x_left, x_right, y_top, y_bottom
    
end
