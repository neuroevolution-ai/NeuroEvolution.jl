using CUDA
using Adapt

include("../util/Matrix_utils.jl")
include("Drawables.jl")
include("../util/Enum_Collection.jl")

function get_button(name::Enum,buttons::All_Buttons)
    if name === app_close_button
        return buttons.app_close_button
    elseif name === dropdown_button_datei
        return buttons.dropdown_button_datei
    elseif name === dropdown_button_anzeigen
        return buttons.dropdown_button_anzeigen
    elseif name === dropdown_button_navigation
        return buttons.dropdown_button_navigation
    elseif name === dropdown_button_tools
        return buttons.dropdown_button_tools
    elseif name === dropdown_button_hilfe
        return buttons.dropdown_button_hilfe
    elseif name === small_button_1
        return buttons.small_button_1
    elseif name === small_button_2
        return buttons.small_button_2
    elseif name === small_button_3
        return buttons.small_button_3
    elseif name === small_button_4
        return buttons.small_button_4
    elseif name === small_button_5
        return buttons.small_button_5
    elseif name === small_button_6
        return buttons.small_button_6
    elseif name === small_button_7
        return buttons.small_button_7
    elseif name === small_button_8
        return buttons.small_button_8
    elseif name === small_button_9
        return buttons.small_button_9
    elseif name === small_button_10
        return buttons.small_button_10
    elseif name === small_button_11
        return buttons.small_button_11
    elseif name === small_button_12
        return buttons.small_button_12
    elseif name === small_button_13
        return buttons.small_button_13


    else return buttons.app_close_button
    end
end

function draw_self(button::Button,parent_matrix)
    if !(button.status[1,blockIdx().x])
        kernel_blit_image_inplace(threadIdx().x,blockIdx().x,parent_matrix,button.matrix_unclicked,button.x_Coord,button.y_Coord)
    else 
        kernel_blit_image_inplace(threadIdx().x,blockIdx().x,parent_matrix,button.matrix_clicked,button.x_Coord,button.y_Coord)
    end
    
end

function placeholder_function()
    #@cuprintln("Hello!",threadIdx().x)
end

function listen(func::On_click_listeners)
    if func === placeholder
        placeholder_function()
    else
        return
    end
    return
end

function select_matrix(button::Button)
    if button.status[1,blockIdx().x]
        return button.matrix_clicked
    else
        return button.matrix_unclicked
    end
end

function click(x,y,button,parents_x,parents_y)
    
    if includes_point(x,y,(button.x_Coord+parents_x),(button.y_Coord+parents_y),button.width,button.height)
        if threadIdx().x == 1
            button.status[1,blockIdx().x] = !(button.status[1,blockIdx().x])
        end
        sync_threads()
        if button.status[2,blockIdx().x]
            reward = 0
        else
            button.status[2,blockIdx().x] = true
            reward = button.reward
        end
        listen(button.on_click_listener)
        return reward,true,button.x_Coord,button.y_Coord
        
    end

    return 0,false,0,0
end
