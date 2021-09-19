using CUDA
using Adapt
#using Random
include("../util/Matrix_utils.jl")
include("../util/Enum_Collection.jl")
include("Button.jl")


struct Window{A,B,C,D}
    matrix_self::A
    current_matrix::D
    x_Coord::Int
    y_Coord::Int
    width::Int
    height::Int
    windows::B
    buttons::C
    modal::Bool
    autoclose::Bool
end
Adapt.@adapt_structure Window

struct All_Windows{A}
    main_window::A
end
Adapt.@adapt_structure All_Windows




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




#=
struct Window{A}
    relative_x_Coord::Int
    relative_y_Coord::Int
    window_name::Window_Names
    width::Int
    height::Int
    #buttons::B#Array containing enums of all buttons in it
    #windows::C #Array containing enums of all windows in it
    
    modal::Bool
end
Adapt.@adapt_structure Window

struct All_Windows{A}
    main_window::A
end
Adapt.@adapt_structure All_Windows

struct Preferences_window
end
Adapt.@adapt_structure Preferences_window
struct Ueber_window
end
Adapt.@adapt_structure Ueber_window


function init_window() 
    draw_self()
end


function draw_self(matrix_self,children,current_matrix)
    temp = copy(matrix_self)
    for child in children
        draw_self(child.__matrix_self,child.__children,current_matrix)
    end
    current_matrix = temp
end


function draw_self_gpu(window::Window)
    
    tx = threadIdx().x

    kernel_blit_image_inplace(
        tx,window.current_matrix,window.matrix_self,                        
        window.relative_x_Coord,
        window.relative_y_Coord,
    )
    sync_threads()
    for child_button in window.buttons
        draw_self_gpu(child_button,window.current_matrix)
    end
    #Iterate over all child Buttons in menu and add them to matrix
    #for child in children
    #    draw_self(child,current_matrix)
    #    sync_threads()
    #end
    return
    
end


function get_matrix(window::Window)

end


function click(window::Window,x,y,parent_x,parent_y)
    
    click_on_window = false
    if includes_point(x,y,window.relative_x_Coord,window.relative_y_Coord,window.width,window.height)
        click_on_window = true
        for child_window in window.windows
            reward,click_on_child,matrix,coords = click(child_window,x,y,window.relative_x_Coord,relative_y_Coord)
            if click_on_child
                kernel_blit_image_inplace(threadIdx().x,window.current_matrix,matrix,coords)
            end 
        end
        for child_button in window.buttons

        end
    end
    
end
=#












