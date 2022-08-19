using CUDA
using Adapt
using DataStructures


struct DummyApp{A}
    number_time_steps::Int
    screen_width::Int
    screen_height::Int
    number_buttons_horizontal::Int
    number_buttons_vertical::Int
    buttons_size_horizontal::Int
    buttons_size_vertical::Int
    gui_elements_rectangles::A
end

function DummyApp(configuration::OrderedDict, number_individuals::Int)

    number_buttons = configuration["number_buttons_horizontal"] * configuration["number_buttons_vertical"]

    # Initialize Cuda Array for positions of GUI elements
    gui_elements_rectangles = CUDA.fill(0.0, (number_buttons, 4))

    grid_size_horizontal = configuration["screen_width"] / configuration["number_buttons_horizontal"]
    grid_size_vertical = configuration["screen_height"] / configuration["number_buttons_vertical"]
    border_horizontal = (grid_size_horizontal - configuration["buttons_size_horizontal"]) / 2
    border_vertical = (grid_size_vertical - configuration["buttons_size_vertical"]) / 2

    buttons = [12, 6, 16, 9, 0, 2, 11, 7, 13, 8, 22, 1, 23, 17, 19, 24, 10, 20, 4, 21, 15, 18, 14, 5, 3] .+ 1

    # Place all checkboxes in colums and rows
    n = 1
    for i = 1:configuration["number_buttons_vertical"]
        for j = 1:configuration["number_buttons_horizontal"]

            x = grid_size_horizontal * (i - 1) + border_horizontal
            y = grid_size_vertical * (j - 1) + border_vertical

            button = buttons[n]

            gui_elements_rectangles[button, :] = [x, y, configuration["buttons_size_horizontal"], configuration["buttons_size_vertical"]]
            n += 1
        end
    end

    DummyApp(
        configuration["number_time_steps"],
        configuration["screen_width"],
        configuration["screen_height"],
        configuration["number_buttons_horizontal"],
        configuration["number_buttons_vertical"],
        configuration["buttons_size_horizontal"],
        configuration["buttons_size_vertical"],
        gui_elements_rectangles,
    )
end


Adapt.@adapt_structure DummyApp


function kernel_eval_fitness(individuals, rewards, environment_seeds, number_rounds, brains, environments)

    number_buttons = environments.number_buttons_horizontal * environments.number_buttons_vertical

    threadID = threadIdx().x
    blockID = blockIdx().x

    offset_shared_memory = 0

    reward = @cuDynamicSharedMem(Float32, number_buttons, offset_shared_memory)
    offset_shared_memory += sizeof(reward)

    sync_threads()

    observation = @cuDynamicSharedMem(Float32, get_number_observations(environments) , offset_shared_memory)
    offset_shared_memory += sizeof(observation)

    sync_threads()

    action = @cuDynamicSharedMem(Float32, get_number_actions(environments), offset_shared_memory)
    offset_shared_memory += sizeof(action)

    sync_threads()

    click_position = @cuDynamicSharedMem(Int32, 2, offset_shared_memory)
    offset_shared_memory += sizeof(click_position)

    sync_threads()

    gui_elements_states = @cuDynamicSharedMem(Int32, number_buttons, offset_shared_memory)
    offset_shared_memory += sizeof(gui_elements_states)

    sync_threads()

    # Initialize brains
    initialize(brains, individuals)

    sync_threads()

    # Initialize rewards 
    if threadID <= number_buttons
        reward[threadID] = 0.0
    end

    sync_threads()

    for i = 1:number_rounds

        # Initialize states of gui elements
        if threadID <= number_buttons
            gui_elements_states[threadID] = 0
        end

        sync_threads()

        reset(brains)

        sync_threads()

        # Iterate over given number of time steps
        for time_step = 1:environments.number_time_steps

            # Set observations
            if threadID <= number_buttons
                observation[threadID] = gui_elements_states[threadID]
            end

            sync_threads()

            # Brain step
            step(brains, observation, action, offset_shared_memory)

            #if blockID == 1 && threadID == 1
            #    @cuprintln("tx=", threadID,"   action_x=", action[1], "  action_y=", action[2], "   action_x=", action[3], "  action_y=", action[4])
            #end 

            if threadID <= 2

                # Scale actions to click positions
                random_number = tanh(action[threadID+2]) * random_normal()
                click_position[threadID] = trunc(0.5 * (tanh(action[threadID]) + 1.0 + random_number) * 400.0)

                #click_position[threadID] = rand(1:400)

                #checkbox = rand(1:environments.number_checkboxes)
                #click_position[threadID] = environments.gui_elements_rectangles[checkbox, threadID] + 10 

            end

            sync_threads()

            #if blockID == 1 && threadID == 1
            #    @cuprintln("tx=", threadID,"   action_x=", action[1], "  action_y=", action[2], "  click_position_x=", click_position[1], "  click_position_y=", click_position[2])
            #end   

            # Process mouse click
            process_click(environments, click_position, gui_elements_states, reward, number_buttons)

            sync_threads()

        end
    end

    # Transfer rewards to output vector
    if threadID == 1
        rewards[blockID] = 0.0

        for i in 1:number_buttons
            rewards[blockID] += reward[i]
        end

        rewards[blockID] /= number_rounds
    end

    return
end

function get_required_threads(environments::DummyApp)

    return min(environments.number_buttons_horizontal * environments.number_buttons_vertical, 2)
end

function get_memory_requirements(environments::DummyApp)

    number_buttons = environments.number_buttons_horizontal * environments.number_buttons_vertical

    return sizeof(Float32) * number_buttons +                             # Reward
           sizeof(Float32) * (get_number_observations(environments) + get_number_actions(environments)) +   # Observation + Action
           sizeof(Int32) * (number_buttons + 2)                                # States of gui elements and clicks

end


function get_number_observations(environments::DummyApp)

    return environments.number_buttons_horizontal * environments.number_buttons_vertical
end

function get_number_actions(environments::DummyApp)
    return 4
end

function process_click(environments, point, gui_elements_states, reward, number_buttons)

    button = threadIdx().x

    if button <= number_buttons
        rect_x = environments.gui_elements_rectangles[button, 1]
        rect_y = environments.gui_elements_rectangles[button, 2]
        width = environments.gui_elements_rectangles[button, 3]
        height = environments.gui_elements_rectangles[button, 4]

        if is_point_in_rect(point[1], point[2], rect_x, rect_y, width, height)

            if button == 1
                if gui_elements_states[button] == 0
                    reward[button] = 1.0
                    gui_elements_states[button] = 1
                end
            else
                if gui_elements_states[button] == 0 && gui_elements_states[button-1] == 1
                    reward[button] += 1.0
                    gui_elements_states[button] = 1
                end
            end
        end
    end

    sync_threads()

    #if blockID == 1 && threadID == 1
    #    @cuprintln("tx=", threadID,"   x=", point[1], "  y=", point[2])
    #end

    return
end

function is_point_in_rect(point_x, point_y, rect_x, rect_y, width, height)

    return rect_x <= point_x <= (rect_x + width) && rect_y <= point_y <= (rect_y + height)
end

# https://www.baeldung.com/cs/uniform-to-normal-distribution
# TODO randn() should work now on the gpu in the latest CUDA.jl version
function random_normal()

    u1 = rand()
    u2 = rand()

    return sqrt(-2 * log(u1)) * cos(2 * pi * u2)

end

