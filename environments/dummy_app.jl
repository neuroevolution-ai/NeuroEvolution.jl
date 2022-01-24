using CUDA
using Adapt
using DataStructures


struct DummyApp{A,B}
    number_time_steps::Int32
    number_inputs::Int64
    number_outputs::Int64
    number_checkboxes::Int64
    number_gui_elements::Int64
    gui_elements_rectangles::A
    checkboxes_checked::B
end

function DummyApp(configuration::OrderedDict, number_individuals::Int)

    number_checkboxes = 16
    number_gui_elements = number_checkboxes + 1

    # Initialize Cuda Array for positions of GUI elements
    gui_elements_rectangles = CUDA.fill(0.0, (number_gui_elements, 4))

    checkboxes_size = 100
    checkboxes_grid_size = 100
    checkboxes_border = 0

    # Place all 8 checkboxes in 4 colums and 2 rows
    n = 1
    for i = 1:4
        for j = 1:4

            x = checkboxes_grid_size * (i-1) + checkboxes_border
            y = checkboxes_grid_size * (j-1) + checkboxes_border

            gui_elements_rectangles[n, :] = [x, y, checkboxes_size, checkboxes_size]
            n += 1
        end
    end

    # Place button (below checkboxes)
    gui_elements_rectangles[n, :] = [200, 200, 150, 50]

    DummyApp(
        convert(Int32, configuration["number_time_steps"]),
        number_checkboxes,
        4,
        number_checkboxes,
        number_gui_elements,
        gui_elements_rectangles,
        CUDA.fill(false, (number_checkboxes, number_individuals)),
    )
end


Adapt.@adapt_structure DummyApp


function kernel_eval_fitness(individuals, rewards, environment_seeds, number_rounds, brains, environments)

    threadID = threadIdx().x
    blockID = blockIdx().x

    offset_shared_memory = 0

    reward = @cuDynamicSharedMem(Float32, environments.number_gui_elements, offset_shared_memory)
    offset_shared_memory += sizeof(reward)

    sync_threads()

    observation = @cuDynamicSharedMem(Float32, environments.number_inputs, offset_shared_memory)
    offset_shared_memory += sizeof(observation)

    sync_threads()

    action = @cuDynamicSharedMem(Float32, environments.number_outputs, offset_shared_memory)
    offset_shared_memory += sizeof(action)

    sync_threads()

    click_position = @cuDynamicSharedMem(Int32, environments.number_outputs, offset_shared_memory)
    offset_shared_memory += sizeof(click_position)

    sync_threads()

    gui_elements_states = @cuDynamicSharedMem(Int32, environments.number_gui_elements, offset_shared_memory)
    offset_shared_memory += sizeof(gui_elements_states)

    sync_threads()

    # Initialize brains
    initialize(brains, individuals)

    sync_threads()

    # Initialize states of gui elements
    if threadID <= environments.number_gui_elements
        gui_elements_states[threadID] = 0
    end

    sync_threads()

    # Initialize rewards 
    if threadID <= environments.number_gui_elements
        reward[threadID] = 0.0
    end

    sync_threads()

    # Iterate over given number of time steps
    for time_step = 1:environments.number_time_steps

        # Set observations
        if threadID <= environments.number_checkboxes
            observation[threadID] = gui_elements_states[threadID]
        end

        sync_threads()

        # Brain step
        step(brains, observation, action, offset_shared_memory)

        
        if threadID <= 2
                        
            # Scale actions to click positions
            random_number = action[threadID+2] * random_normal()
            click_position[threadID] = trunc(0.5 * (action[threadID] + 1.0 + random_number) * 400.0)
            
            #click_position[threadID] = rand(1:400)

            #checkbox = rand(1:environments.number_checkboxes)
            #click_position[threadID] = environments.gui_elements_rectangles[checkbox, threadID] + 10 

        end

        sync_threads() 

        #if blockID == 1 && threadID == 1
        #    @cuprintln("tx=", threadID,"   action_x=", action[1], "  action_y=", action[2])
        #end       

        # Process mouse click
        process_click(environments, click_position, gui_elements_states, reward)

        sync_threads()
    
    end

    # Transfer rewards to output vector
    if threadID == 1
        rewards[blockID] = 0.0

        for i in 1:environments.number_gui_elements
            rewards[blockID] += reward[i]
        end
    end

    return
end

function get_required_threads(environments::DummyApp)

    return environments.number_gui_elements
end

function get_memory_requirements(environments::DummyApp)
    return sizeof(Float32) * environments.number_gui_elements +                             # Reward
           sizeof(Float32) * (environments.number_inputs + environments.number_outputs) +   # Observation + Action
           sizeof(Int32) * environments.number_gui_elements                                 # States of gui elements

end

function get_number_inputs(environments::DummyApp)
    return environments.number_inputs
end

function get_number_outputs(environments::DummyApp)
    return environments.number_outputs
end

function process_click(environments, point, gui_elements_states, reward)

    threadID = threadIdx().x

    if threadID <= environments.number_gui_elements
        rect_x = environments.gui_elements_rectangles[threadID, 1]
        rect_y = environments.gui_elements_rectangles[threadID, 2]
        width = environments.gui_elements_rectangles[threadID, 3]
        height = environments.gui_elements_rectangles[threadID, 4]

        if is_point_in_rect(point[1], point[2], rect_x, rect_y, width, height)

            # Toggle checkboxes
            if threadID <= environments.number_checkboxes
                # Bitwise xor with 1 does toggling: https://stackoverflow.com/questions/11604409/how-to-toggle-a-boolean
                #gui_elements_states[threadID] ⊻= 1

                if gui_elements_states[threadID] == 0
                    reward[threadID] += 1.0
                    gui_elements_states[threadID] = 1
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
function random_normal()

    u1 = rand()
    u2 = rand()

    return sqrt(-2*log(u1)) * cos(2*pi*u2)

end

