module Cell

"""A cell in the maze.

A maze "Cell" is a point in the grid which may be surrounded by walls to
the north, east, south or west.

"""
#wall_pairs = {'N'=>'S', 'S'=>'N', 'E'=>'W', 'W'=>'E'}

    function init_Cell(x,y)
        coordinates = Array(undef,2)
        walls = trues(4)
        coordinates[1] = x
        coordinates[2] = y

    end

    function has_all_walls(walls)
        return all(walls)
    end

    function knock_down_walls(cell,other_cell,wall)
        body
    end
end
