# Code from here: https://discourse.julialang.org/t/casting-annotations-and-numeric-types-for-cudanative/19853/2

using CUDA

function l_0(x, y, z, w, h)
       return x + y*w + z*w*h
end

function l(x, y, z, w, h)
       _x = x - 1
       _y = y - 1
       _z = z - 1
       return l_0(_x, _y, _z, w, h) + 1
end

function kernel(out)
    x = blockIdx().x
    y = blockIdx().y
    w = gridDim().x
    h = gridDim().y
    z = 1

    arr = @cuDynamicSharedMem(Int32, (w, h, 3))
    arr[x,y,z] = Int32(1)
    linear_index::Int32  = l(x,y,z,w,h)
    arr[linear_index] = Int32(1) # still works

    out[x, y] = linear_index
    return nothing
end

function make_matrix(width :: Int, height :: Int)
    grid = (width, height)
    threads = (1,)

    cu_out = CuArray{Int32, 2}(undef, width, height)

    @cuda blocks=grid threads=threads shmem=sizeof(Int32)*prod(grid) kernel(cu_out)
    out = Array{Int32, 2}(cu_out)
    return out
end

function main()
    width = 10
    height = 10
    matrix = make_matrix(width, height)
    println(matrix)
end

main()
