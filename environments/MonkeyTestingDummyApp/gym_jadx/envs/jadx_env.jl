using CUDA
using Adapt
using StructArrays

include("../ui/Button.jl")
include("../ui/Window.jl")

struct Jadx_env{A,B,C}
    __all_buttons::A
    main_window::B
    __windows::C
    __frame_buffer::B
    __width::Int32
    __height::Int32
    __should_restack::Bool
    __last_clicked_index::Int32
    __windows_to_be_removed::C
end

Adapt.@adapt_structure Jadx_env

#
function initialize()#environment configuration
    __all_buttons = StructArray{Button}(undef,100)# length of number of possible buttons
    all_buttons = replace_storage(CuArray,__all_buttons)
    main_window = CUDA.fill(0.0f0,(100,100)) #(x,y) = dimensionen des main_windows
    __windows = StructArray{Window}(undef,100) # length of number of possible windows
    windows = replace_storage(CuArray,__windows)
    __frame_buffer = similar(main_window)
    __width = convert(Int32,100)
    __height = convert(Int32,100)
    __should_restack = false
    __last_clicked_index = convert(Int32,0)
    __windows_to_be_removed = StructArray{Window}(undef,100)# length of number of possible windows
    windows_to_be_removed = replace_storage(CuArray,__windows_to_be_removed)


    Jadx_env(all_buttons,main_window,windows,__frame_buffer,__width,__height,__should_restack,__last_clicked_index,windows_to_be_removed)
end

#prepare the Environment inside the Kernel for each round
function env_initialize(env_cfg::Jadx_env)

end

function env_step(action,input,env_cfg)

end

function __init_components()

end


function init_main_window(test)

    return
end

clicked = CUDA.fill(0.5f0,(10,10,3))
unclicked = CUDA.fill(1.0f0,(10,10,3))
parent = CUDA.fill(0.0f0,(10,10,3))

button = Button(convert(Int32,1),convert(Int32,1),convert(Int32,10),convert(Int32,10),clicked,unclicked,convert(Int32,2),true,false,false)
window = Window(convert(Int32,1),convert(Int32,1),convert(Int32,1),convert(Int32,1),button,button,false)

#@cuda init_main_window(window)

