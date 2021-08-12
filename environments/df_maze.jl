using CUDA
using Adapt
#=
A cell in the maze.

A maze "Cell" is a point in the grid which may be surrounded by walls to
the north, east, south or west.

=#
#wall_pairs = {'N'=>'S', 'S'=>'N', 'E'=>'W', 'W'=>'E'}
    struct Cell
        coordinates::CuArray
        walls::BitVector
    end
    function Adapt.adapt_structure(to, cell::Cell)
        coordinates = Adapt.adapt_structure(to, cell.coordinates)
        walls = Adapt.adapt_structure(to, cell.walls)
        Cell(coordinates, walls)
    end

    function init_Cell(x,y)
        coordinates = CuArray([x,y])
        walls = CUDA.trues(4)
        cell = Cell(coordinates,walls)
        return cell

    end

    function has_all_walls(walls)
        return all(walls)
    end

    function knock_down_walls(cell,other_cell,wall)
        new_walls = cell.walls
        new_walls[wall] = false
        new_other_walls = other_cell.walls
        new_other_walls[(wall+2) % 4] = false
        new_cell = Cell(cell.coordinates,new_walls)
        new_other_cell = Cell(other_cell.coordinates,new_other_walls)
        return new_cell,new_other_cell
    end
    
    function cell_kernel()
        index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
        x = init_Cell(1,2)
        @cuprintln(x.coordinates[1])
        return
    end
    @cuda threads=5 cell_kernel()
    #dummy = Cell(CUDA.fill(0,2),CUDA.trues(4))
    #maze = CUDA.fill(undef,5,5)
    #dummy = @device_code_warntype @cuda threads = 25 cell_kernel(x)
    #synchronize()
    #display(dummy)
    #maze = CUDA.fill(Cell(CUDA.fill(0,2),CUDA.trues(4)),5,5)
    #maze[1]
    #maze[1] = init_Cell(maze[1],1,1)
    #broadcast(init_Cell,maze,1,1)
    #display(maze)

    #display(maze)
    #for i in eachindex(maze)
    #   maze[i] = Cell(i % size(maze,2), )       
    #end
    #synchronize()
    #display(x)
    #x = CUDA.fill(1.0,2)



#=
module Maze 

#A Maze, represented as a grid of cells.



end
=#