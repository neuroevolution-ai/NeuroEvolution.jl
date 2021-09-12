using CUDA


struct Jadx_env
    __all_buttons
    main_window
    __windows
    __frame_buffer
    __width
    __height
    __should_restack
    __last_clicked_index
    __windows_to_be_removed
end

function Adapt.adapt_structure(to,env_cfg::Jadx_env)
    
    
end

#
function initialize()

end

#prepare the Environment inside the Kernel for each round
function env_initialize()

end

function env_step(action,input,env_cfg)

end