using CUDA
using Adapt
using Random
include("../util/Matrix_utils.jl")


struct Button{A,B}
    __relative_coordinate_x::Int32
    __relative_coordinate_y::Int32
    __width::Int32
    __height::Int32
    __matrix_clicked::A
    __matrix_unclicked::A
    __reward::Int32
    __clicked::Bool
    __reward_given::Bool
    on_click_listener::B
    __resettable::Bool
end


struct Dropdown_button_datei
    __button::Button
end

Adapt.@adapt_structure Button


function draw_self_gpu(threadID, button::Button, parent_matrix)
    threadID = threadIdx().x
    if button.__clicked
        kernel_blit_image_inplace(
            threadID,
            parent_matrix,
            button.__matrix_clicked,
            button.__relative_coordinate_x,
            button.__relative_coordinate_y,
        )
    else
        kernel_blit_image_inplace(
            threadID,
            parent_matrix,
            button.__matrix_unclicked,
            button.__relative_coordinate_x,
            button.__relative_coordinate_y,
        )
    end
end

function placeholder_function()
    return
end