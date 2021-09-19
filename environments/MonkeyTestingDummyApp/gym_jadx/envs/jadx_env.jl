using CUDA
using Adapt
using StructArrays
using ImageView, Images
include("../util/Matrix_utils.jl")
include("../util/Image_Container.jl")
include("../ui/Window.jl")
include("../ui/Button.jl")



struct Jadx_Environment{A,B,C,D}
    width::Int
    height::Int
    frame_buffer::A
    windows::B
    all_windows::C
    #buttons
    #all_buttons
    windows_to_be_removed::B
    container::D
end
Adapt.@adapt_structure Jadx_Environment


function init_main_window()
    matrix_self = CuArray(get_array_of_image("main_window.png"))
    current_matrix = similar(matrix_self)
    width = size(matrix_self,3)
    height = size(matrix_self,2)
    return Window(0,0,matrix_self,current_matrix,width,height,true)
end
function initialize()
    total_number_buttons = 28
    total_number_windows = 8
    container = build_image_container()

    return Jadx_Environment()
end

#prepare the Environment inside the Kernel for each round
function env_initialize(env_cfg::Jadx_Environment)

end

function env_step(action,input,env_cfg)
    reward = 0





    return
end

#=
function init_app_close_button(image)
    return Button(convert(Int32,380),convert(Int32,0),convert(Int32,size(image,3)),convert(Int32,size(image,2)),convert(Int32,2),app_close_button,placeholder,false,false,true)
end
function init_dropdown_button_datei(position,image)
    return Button(position,convert(Int32,12),convert(Int32,size(image,3)),convert(Int32,size(image,2)),convert(Int32,2),placeholder,false,false,true)
end
function init_dropdown_button_anzeigen(position,image)
    return Button(position,convert(Int32,12),convert(Int32,size(image,3)),convert(Int32,size(image,2)),convert(Int32,2),placeholder,false,false,true)
end
function init_dropdown_button_navigation(position,image)
    return Button(position,convert(Int32,12),convert(Int32,size(image,3)),convert(Int32,size(image,2)),convert(Int32,2),placeholder,false,false,true)
end
function init_dropdown_button_tools(position,image)
    return Button(position,convert(Int32,12),convert(Int32,size(image,3)),convert(Int32,size(image,2)),convert(Int32,2),placeholder,false,false,true)
end
function init_dropdown_button_hilfe(position,image)
    return Button(position,convert(Int32,12),convert(Int32,size(image,3)),convert(Int32,size(image,2)),convert(Int32,2),placeholder,false,false,true)
end

function init_main_window_small_buttons()
    small_button_array = StructArray{Button}(undef,13)
    i = 1
    j = 1
    while(i<86)
        small_button = Button(convert(Int32,i),convert(Int32,22),convert(Int32,7),convert(Int32,6),convert(Int32,2),placeholder,false,false,true)
        small_button_array[j] = small_button
        j += 1
        i += 7
    end
    return small_button_array
end

=#
function __init_components()

end
function reset()

end


function env_initialize(env::Jadx_Environment)
    tx = threadIdx().x
    #kernel_blit_image_inplace(tx,env.frame_buffer,env.all_windows.main_window.matrix_self,env.all_windows.main_window.relative_x_Coord,env.all_windows.main_window.relative_y_Coord)
    #draw_self_gpu(env.all_windows.main_window)
    #sync_threads()
    #kernel_blit_image_inplace(tx,env.frame_buffer,env.all_windows.main_window.current_matrix,env.all_windows.main_window.relative_x_Coord,env.all_windows.main_window.relative_y_Coord)

end
function eval_fitness(env::Jadx_Environment)
    tx = threadIdx().x

    kernel_blit_image_inplace(tx,env.frame_buffer,get_image(main_window,env.container),0,0)
    #obs = reset()
    return
end

#=
env = initialize()
imshow(colorview(RGB,Array(env.frame_buffer)))
@cuda threads=1024 eval_fitness(env)
CUDA.synchronize()
imshow(colorview(RGB,Array(env.frame_buffer)))
=#