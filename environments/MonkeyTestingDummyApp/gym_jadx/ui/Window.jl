using CUDA
using Adapt
#using Random
include("../util/Matrix_utils.jl")
include("../util/Enum_Collection.jl")
include("Button.jl")
include("Drawables.jl")


function get_window(name::Enum,windows::All_Windows)#::Window
    if name == main_window
        return windows.main_window
    else 
        return windows.main_window
    end
end



function draw_self(window::Window,windows::All_Windows,buttons::All_Buttons)
    #kernel_blit_image_inplace(threadIdx().x,window.current_matrix,window.matrix_self,window.x_Coord,window.y_Coord)
    kernel_blit_image_inplace(threadIdx().x,blockIdx().x,window.current_matrix,window.matrix_self,window.x_Coord,window.y_Coord)

    for button in window.buttons
        current_button = get_button(button,buttons)
        draw_self(current_button,window.current_matrix)
    end
end

function click(x,y,window::Window,all_buttons::All_Buttons)
    click_on_window = false
    
    if includes_point(x,y,window.x_Coord,window.y_Coord,window.width,window.height) 
        click_on_window = true
        #for button in window.buttons
        for button in window.buttons
            
            current_button = get_button(button,all_buttons)
            reward,click_on_child,coords_x,coords_y = click(x,y,current_button,window.x_Coord,window.y_Coord)
            sync_threads()
            if click_on_child
                if current_button.status[1,blockIdx().x]
                    #kernel_blit_image_inplace(threadIdx().x,blockIdx().x,window.current_matrix,current_button.matrix_clicked,coords_x,coords_y) 
                else
                    #kernel_blit_image_inplace(threadIdx().x,blockIdx().x,window.current_matrix,current_button.matrix_unclicked,coords_x,coords_y) 
                end
                return reward,click_on_window,window.x_Coord,window.y_Coord
            end
        end
    end
    return 0,false,0,0
end













