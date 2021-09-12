using CUDA
using Adapt
using StructArrays

include("../ui/Button.jl")

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
function initialize(environment_configuration)
    __all_buttons

    Jadx_env()
end

#prepare the Environment inside the Kernel for each round
function env_initialize(env_cfg::Jadx_env)

end

function env_step(action,input,env_cfg)

end

function __init_components()

end
