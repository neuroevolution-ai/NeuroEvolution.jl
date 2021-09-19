using CUDA
using Adapt

include("../util/Matrix_utils.jl")
#include("../util/Enum_Collection.jl")


struct Button{A,B,C}
    x_Coord::Int
    y_Coord::Int
    width::Int
    height::Int
    reward::Int
    matrix_clicked::A
    matrix_unclicked::B
    status::C
    resettable::Bool
    on_click_listener::On_click_listeners
end
Adapt.@adapt_structure Button

struct All_Buttons{A}
    app_close_button::A
    dropdown_button_datei::A
    dropdown_button_anzeigen::A
    dropdown_button_navigation::A
    dropdown_button_tools::A
    dropdown_button_hilfe::A
    small_button_1::A
    small_button_2::A
    small_button_3::A
    small_button_4::A
    small_button_5::A
    small_button_6::A
    small_button_7::A
    small_button_8::A
    small_button_9::A
    small_button_10::A
    small_button_11::A
    small_button_12::A
    small_button_13::A
end
Adapt.@adapt_structure All_Buttons

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

function select_matrix(button)
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
#=
struct Button{A}
    relative_x_Coord::Int
    relative_y_Coord::Int
    width::Int
    height::Int
    reward::Int
    button_name::Button_Names
    clicked::Bool
    reward_given::Bool
    resettable::Bool
    #on_click_listener
end
Adapt.@adapt_structure Button

function create_button(x,y,reward,button_name,container)
    image = get_image(button_name,container)
    return Button(x,y,size(image,3),size(image,2),reward,button_name,false,false,true)
end

struct DropDownButton
end


function placeholder_function()
    @cuprintln("Hello!")
end

function get_matrix(env_cfg::Jadx_env,button::Button)
    if button.__button_type == app_close_button
        return env_cfg.image_container.close_window_button_large_unclicked
        
    elseif button.__button_type == dropdown_button_datei
        if button.__clicked
            return env.cfg.image_container.drpdwn_datei_clicked
        else
            return env.cfg.image_container.drpdwn_datei_unclicked
        end
    end
end
function listen(function::On_click_listeners)
end

function click(env_cfg::Jadx_env,x,y,button::Button,parent::Window)
    if includes_point(x, y, button.__relative_x_Coord + parent.__relative_coordinate_x, button.__relative_y_Coord + parent.__relative_coordinate_y, button.__width, button.__height)
        button.__clicked = !button.__clicked
        if button.__reward_given
            reward = 0
        else
            button.__reward_given = true
            reward = button.__reward
        end
        listen(button.__on_click_listener)
        return reward,true
    end
    return 0,false
end
=#
