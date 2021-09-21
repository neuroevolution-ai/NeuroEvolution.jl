using CUDA
using Adapt
using StructArrays
using Images
using ImageView
using Random
using Test
include("../util/Matrix_utils.jl")
include("../util/Enum_Collection.jl")
include("../ui/Window.jl")
include("../ui/Button.jl")
include("../ui/Drawables.jl")

struct Jadx_Environment{A,B,C,D}
    all_windows::A
    all_buttons::B
    windows::C
    frame_buffer::D
end
Adapt.@adapt_structure Jadx_Environment

function init_components()
end

function env_initialize(env::Jadx_Environment)

end

function init_app_close_button(number_individuals)
    matrix_unclicked = CuArray(get_array_of_image("close_window_button_large_unclicked.png"))
    return Button(380,1,size(matrix_unclicked,3),size(matrix_unclicked,2),2,matrix_unclicked,matrix_unclicked,CUDA.fill(false,(2,number_individuals)),true,placeholder)
end

function init_dropdown_button_datei(position,number_individuals)
    matrix_unclicked = CuArray(get_array_of_image("drpdwn_datei_unclicked.png"))
    matrix_clicked = CuArray(get_array_of_image("drpdwn_datei_clicked.png"))
    return Button(position,12,size(matrix_unclicked,3),size(matrix_unclicked,2),2,matrix_clicked,matrix_unclicked,CUDA.fill(false,(2,number_individuals)),true,placeholder)
end
function init_dropdown_button_anzeigen(position,number_individuals)
    matrix_unclicked = CuArray(get_array_of_image("drpdwn_anzeigen_unclicked.png"))
    matrix_clicked = CuArray(get_array_of_image("drpdwn_anzeigen.png"))
    return Button(position,12,size(matrix_unclicked,3),size(matrix_unclicked,2),2,matrix_clicked,matrix_unclicked,CUDA.fill(false,(2,number_individuals)),true,placeholder)
end
function init_dropdown_button_navigation(position,number_individuals)
    matrix_unclicked = CuArray(get_array_of_image("drpdwn_navigation_unclicked.png"))
    matrix_clicked = CuArray(get_array_of_image("drpdwn_navigation_clicked.png"))
    return Button(position,12,size(matrix_unclicked,3),size(matrix_unclicked,2),2,matrix_clicked,matrix_unclicked,CUDA.fill(false,(2,number_individuals)),true,placeholder)
end
function init_dropdown_button_tools(position,number_individuals)
    matrix_unclicked = CuArray(get_array_of_image("drpdwn_tools_unclicked.png"))
    matrix_clicked = CuArray(get_array_of_image("drpdwn_tools_clicked.png"))
    return Button(position,12,size(matrix_unclicked,3),size(matrix_unclicked,2),2,matrix_clicked,matrix_unclicked,CUDA.fill(false,(2,number_individuals)),true,placeholder)
end
function init_dropdown_button_hilfe(position,number_individuals)
    matrix_unclicked = CuArray(get_array_of_image("drpdwn_hilfe_unclicked.png"))
    matrix_clicked = CuArray(get_array_of_image("drpdwn_hilfe_clicked.png"))
    return Button(position,12,size(matrix_unclicked,3),size(matrix_unclicked,2),2,matrix_clicked,matrix_unclicked,CUDA.fill(false,(2,number_individuals)),true,placeholder)
end

function init_main_window_small_buttons(number_individuals)
    position=1
    matrix_unclicked = CuArray(get_array_of_image("small_button_1_unclicked.png"))
    matrix_clicked = CuArray(get_array_of_image("small_button_1_clicked.png"))
    small_button_1 = Button(position,22,size(matrix_unclicked,3),size(matrix_unclicked,2),2,matrix_clicked,matrix_unclicked,CUDA.fill(false,(2,number_individuals)),true,placeholder)
    position += 7
    matrix_unclicked = CuArray(get_array_of_image("small_button_2_unclicked.png"))
    matrix_clicked = CuArray(get_array_of_image("small_button_2_clicked.png"))
    small_button_2 = Button(position,22,size(matrix_unclicked,3),size(matrix_unclicked,2),2,matrix_clicked,matrix_unclicked,CUDA.fill(false,(2,number_individuals)),true,placeholder)
    position += 7
    matrix_unclicked = CuArray(get_array_of_image("small_button_3_unclicked.png"))
    matrix_clicked = CuArray(get_array_of_image("small_button_3_clicked.png"))
    small_button_3 = Button(position,22,size(matrix_unclicked,3),size(matrix_unclicked,2),2,matrix_clicked,matrix_unclicked,CUDA.fill(false,(2,number_individuals)),true,placeholder)
    position += 7
    matrix_unclicked = CuArray(get_array_of_image("small_button_4_unclicked.png"))
    matrix_clicked = CuArray(get_array_of_image("small_button_4_clicked.png"))
    small_button_4 = Button(position,22,size(matrix_unclicked,3),size(matrix_unclicked,2),2,matrix_clicked,matrix_unclicked,CUDA.fill(false,(2,number_individuals)),true,placeholder)
    position += 7
    matrix_unclicked = CuArray(get_array_of_image("small_button_5_unclicked.png"))
    matrix_clicked = CuArray(get_array_of_image("small_button_5_clicked.png"))
    small_button_5 = Button(position,22,size(matrix_unclicked,3),size(matrix_unclicked,2),2,matrix_clicked,matrix_unclicked,CUDA.fill(false,(2,number_individuals)),true,placeholder)
    position += 7
    matrix_unclicked = CuArray(get_array_of_image("small_button_6_unclicked.png"))
    matrix_clicked = CuArray(get_array_of_image("small_button_6_clicked.png"))
    small_button_6 = Button(position,22,size(matrix_unclicked,3),size(matrix_unclicked,2),2,matrix_clicked,matrix_unclicked,CUDA.fill(false,(2,number_individuals)),true,placeholder)
    position += 7
    matrix_unclicked = CuArray(get_array_of_image("small_button_7_unclicked.png"))
    matrix_clicked = CuArray(get_array_of_image("small_button_7_clicked.png"))
    small_button_7 = Button(position,22,size(matrix_unclicked,3),size(matrix_unclicked,2),2,matrix_clicked,matrix_unclicked,CUDA.fill(false,(2,number_individuals)),true,placeholder)
    position += 7
    matrix_unclicked = CuArray(get_array_of_image("small_button_8_unclicked.png"))
    matrix_clicked = CuArray(get_array_of_image("small_button_8_clicked.png"))
    small_button_8 = Button(position,22,size(matrix_unclicked,3),size(matrix_unclicked,2),2,matrix_clicked,matrix_unclicked,CUDA.fill(false,(2,number_individuals)),true,placeholder)
    position += 7
    matrix_unclicked = CuArray(get_array_of_image("small_button_9_unclicked.png"))
    matrix_clicked = CuArray(get_array_of_image("small_button_9_clicked.png"))
    small_button_9 = Button(position,22,size(matrix_unclicked,3),size(matrix_unclicked,2),2,matrix_clicked,matrix_unclicked,CUDA.fill(false,(2,number_individuals)),true,placeholder)
    position += 7
    matrix_unclicked = CuArray(get_array_of_image("small_button_10_unclicked.png"))
    matrix_clicked = CuArray(get_array_of_image("small_button_10_clicked.png"))
    small_button_10 = Button(position,22,size(matrix_unclicked,3),size(matrix_unclicked,2),2,matrix_clicked,matrix_unclicked,CUDA.fill(false,(2,number_individuals)),true,placeholder)
    position += 7
    matrix_unclicked = CuArray(get_array_of_image("small_button_11_unclicked.png"))
    matrix_clicked = CuArray(get_array_of_image("small_button_11_clicked.png"))
    small_button_11 = Button(position,22,size(matrix_unclicked,3),size(matrix_unclicked,2),2,matrix_clicked,matrix_unclicked,CUDA.fill(false,(2,number_individuals)),true,placeholder)
    position += 7
    matrix_unclicked = CuArray(get_array_of_image("small_button_12_unclicked.png"))
    matrix_clicked = CuArray(get_array_of_image("small_button_12_clicked.png"))
    small_button_12 = Button(position,22,size(matrix_unclicked,3),size(matrix_unclicked,2),2,matrix_clicked,matrix_unclicked,CUDA.fill(false,(2,number_individuals)),true,placeholder)
    position += 7
    matrix_unclicked = CuArray(get_array_of_image("small_button_13_unclicked.png"))
    matrix_clicked = CuArray(get_array_of_image("small_button_13_clicked.png"))
    small_button_13 = Button(position,22,size(matrix_unclicked,3),size(matrix_unclicked,2),2,matrix_clicked,matrix_unclicked,CUDA.fill(false,(2,number_individuals)),true,placeholder)

    return begin
        small_button_1,
        small_button_2,
        small_button_3,
        small_button_4,
        small_button_5,
        small_button_6,
        small_button_7,
        small_button_8,
        small_button_9,
        small_button_10,
        small_button_11,
        small_button_12,
        small_button_13
    end
end

function init_all_buttons(number_individuals)
    close_button = init_app_close_button(number_individuals)
    position = 1
    button_datei = init_dropdown_button_datei(position,number_individuals)
    position += button_datei.width
    button_anzeigen = init_dropdown_button_anzeigen(position,number_individuals)
    position += button_anzeigen.width
    button_navigation = init_dropdown_button_navigation(position,number_individuals)
    position += button_navigation.width
    button_tools = init_dropdown_button_tools(position,number_individuals)
    position += button_tools.width
    button_hilfe = init_dropdown_button_hilfe(position,number_individuals)
    begin
        small_button_1,
        small_button_2,
        small_button_3,
        small_button_4,
        small_button_5,
        small_button_6,
        small_button_7,
        small_button_8,
        small_button_9,
        small_button_10,
        small_button_11,
        small_button_12,
        small_button_13 = init_main_window_small_buttons(number_individuals)
    end
    return All_Buttons(
        close_button,
        button_datei,
        button_anzeigen,
        button_navigation,
        button_tools,
        button_hilfe,
        small_button_1,
        small_button_2,
        small_button_3,
        small_button_4,
        small_button_5,
        small_button_6,
        small_button_7,
        small_button_8,
        small_button_9,
        small_button_10,
        small_button_11,
        small_button_12,
        small_button_13
        )
end

function init_main_window(number_individuals)
    windows = CuArray{Window_Names}(undef,(5,number_individuals))
    buttons = Array{Button_Names}(undef,19)
    buttons[1] = app_close_button
    buttons[2] = dropdown_button_datei
    buttons[3] = dropdown_button_anzeigen
    buttons[4] = dropdown_button_navigation
    buttons[5] = dropdown_button_tools
    buttons[6] = dropdown_button_hilfe
    buttons[7] = small_button_1
    buttons[8] = small_button_2
    buttons[9] = small_button_3
    buttons[10] = small_button_4
    buttons[11] = small_button_5
    buttons[12] = small_button_6
    buttons[13] = small_button_7
    buttons[14] = small_button_8
    buttons[15] = small_button_9
    buttons[16] = small_button_10
    buttons[17] = small_button_11
    buttons[18] = small_button_12
    buttons[19] = small_button_13
    matrix_self = CuArray(get_array_of_image("main_window.png"))
    current_matrix = CuArray{N0f8}(undef,(size(matrix_self,1),size(matrix_self,2),size(matrix_self,3),number_individuals))

    return Window(matrix_self,current_matrix,1,1,size(matrix_self,3),size(matrix_self,2),windows,CuArray(buttons),true,false)
end

function init_all_windows(number_individuals)
    main = init_main_window(number_individuals)
    return All_Windows(main)
end



function initialize2()
    number_individuals = 100

    all_buttons = init_all_buttons(number_individuals)
    #init All Buttons

    all_windows = init_all_windows(number_individuals)
    #init all Windows
    windows = CuArray{Window_Names}(undef,(5,number_individuals))
    #frame_buffer = similar(all_windows.main_window.matrix_self)
    frame_buffer = CuArray{N0f8}(undef,(size(all_windows.main_window.matrix_self,1),size(all_windows.main_window.matrix_self,2),size(all_windows.main_window.matrix_self,3),number_individuals))
    return Jadx_Environment(all_windows,all_buttons,windows,frame_buffer)
end
#display(env)

function reset2(env::Jadx_Environment)
    for i in 1:size(env.windows,1)
        env.windows[i,blockIdx().x] = no_window
    end
    env.windows[1,blockIdx().x] = main_window
    sync_threads()
    current_window = get_window(env.windows[1,blockIdx().x],env.all_windows)
    
    draw_self(current_window,env.all_windows,env.all_buttons)
    
    sync_threads()
    
    kernel_blit_image_inplace(threadIdx().x,blockIdx().x,env.frame_buffer,current_window.current_matrix,current_window.x_Coord,current_window.y_Coord) 

end


function step(env::Jadx_Environment,action,input)
    reward = 0
    number_of_windows_to_be_removed  = 0
    should_restack = false
    click_x = clamp(ceil(action[1] * 400),0,400)
    click_y = clamp(ceil(action[2] * 268),0,268)
    for i in 1:size(env.windows,1)
        if env.windows[i,blockIdx().x] ≠ no_window
            #index = i - windows_to_be_removed
            #last_clicked_index = size(env.windows,1) - index
            current_window = get_window(env.windows[i,blockIdx().x],env.all_windows)

            reward,window_includes_point,coords_x,coords_y = click(click_x,click_y,current_window,env.all_buttons)
            sync_threads()
            if window_includes_point
                if !should_restack
                    draw_self(current_window,env.all_windows,env.all_buttons)
                    sync_threads()
                    kernel_blit_image_inplace(threadIdx().x,blockIdx().x,env.frame_buffer,current_window.current_matrix,coords_x,coords_y)
                end
                break

            else
                if current_window.modal
                    break
                elseif current_window.autoclose
                    
                end
            end   
        end
    end
    if should_restack
        #stack_windows
    end
    should_restack = false
    sync_threads()
    grayscale(input,env.frame_buffer)
    sync_threads()
    return reward
end

function kernel3(env::Jadx_Environment,input)
    action = @cuDynamicSharedMem(Float32,2)
    fitness_total = 0
    for round in 1:1

        reset2(env)
        for i in 1:1000
            if threadIdx().x == 1
                action[1] = rand(Float32)
                action[2] = rand(Float32)
            end
            sync_threads()
            reward = step(env,action,input)
        end
    end
    return
    
end
env = initialize2()
width = env.all_windows.main_window.width
height = env.all_windows.main_window.height
input = CUDA.fill(0.0f0,(268,400,100))
@cuda threads=400 blocks=100 shmem=sizeof(Float32)*2 kernel3(env,input)
CUDA.synchronize()

display(Array(input))

#display(env.windows)
#imshow(colorview(Gray,Array(input[:,:,1])))



imshow(colorview(RGB,Array(env.frame_buffer[:,:,:,1])))

imshow(colorview(RGB,Array(env.frame_buffer[:,:,:,2])))
imshow(colorview(RGB,Array(env.frame_buffer[:,:,:,3])))
imshow(colorview(RGB,Array(env.frame_buffer[:,:,:,4])))
imshow(colorview(RGB,Array(env.frame_buffer[:,:,:,5])))
imshow(colorview(RGB,Array(env.frame_buffer[:,:,:,6])))
imshow(colorview(RGB,Array(env.frame_buffer[:,:,:,7])))
imshow(colorview(RGB,Array(env.frame_buffer[:,:,:,8])))
imshow(colorview(RGB,Array(env.frame_buffer[:,:,:,9])))
imshow(colorview(RGB,Array(env.frame_buffer[:,:,:,10])))



function episode(number_time_steps,number_rounds,environment_cfg::Jadx_Environment)
    action = @cuDynamicSharedMem(Float32,2)
    fitness_total = 0
    #brain_initialize
    for round in 1:number_rounds
        fitness_current = 0
        #reset brain
        #env_initialize()
        obs = reset2(environment_cfg)
        sync_threads()
        for timestep in 1:number_time_steps
            #action = brain_step(obs)
            if threadIdx().x == 1
                action[1] = rand(Float32)
                action[2] = rand(Float32)
                @cuprintln()
            end
            sync_threads()
            obs,reward = env_step(environment_cfg,action)
        end

    end
end


#@cuda threads=400 blocks=5 shmem=sizeof(Float32)*2 episode(1000,1,env)