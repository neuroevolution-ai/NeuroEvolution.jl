using Test
using CUDA
using Random
using Rectangle

include("../environments/dummy_app.jl")


function kernel_test_is_point_in_rect(
    point_x,
    point_y,
    rect_x,
    rect_y,
    width,
    height,
    result,
)

    threadID = threadIdx().x

    result[threadID] = is_point_in_rect(point_x, point_y, rect_x, rect_y, width, height)

    return
end


@testset "Point in Rect" begin

    app_width = 600
    app_height = 400

    result_gpu = CUDA.fill(true, 1)

    for i = 1:10000

        point_x = rand(1:app_width)
        point_y = rand(1:app_width)

        rect_x = rand(1:app_width)
        rect_y = rand(1:app_width)
        rect_width = rand(100:400)
        rect_height = rand(50:200)

        CUDA.@cuda threads = 1 blocks = 1 kernel_test_is_point_in_rect(
            point_x,
            point_y,
            rect_x,
            rect_y,
            rect_width,
            rect_height,
            result_gpu,
        )
        CUDA.synchronize()

        result = Array(result_gpu)

        rect = Rect(rect_x, rect_y, rect_x + rect_width, rect_y + rect_height)
        @test inside((point_x, point_y), rect) == result[1]

    end

end
