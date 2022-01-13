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

    number_checkboxes = 8
    number_gui_elements = number_checkboxes + 1

    # Initialize Cuda Array for positions of GUI elements
    gui_elements_rectangles = CUDA.fill(0.0, (number_gui_elements, 4, number_individuals))

    checkboxes_width = 20
    checkboxes_height = 20
    checkboxes_grid_size = 50
    checkboxes_border = 30

    # Place all 8 checkboxes in 4 colums and 2 rows
    n = 1
    for i = 1:4
        for j = 1:2

            x = checkboxes_grid_size * i + checkboxes_border
            y = checkboxes_grid_size * j + checkboxes_border

            gui_elements_rectangles[n, :] = [x, y, checkboxes_width, checkboxes_height]
            n += 1
        end
    end

    # Place button (below checkboxes)
    gui_elements_rectangles[n, :] = [200, 200, 150, 50]

    DummyApp(
        convert(Int32, configuration["number_time_steps"]),
        10,
        2,
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

    reward = @cuDynamicSharedMem(Float32, 1, offset_shared_memory)
    offset_shared_memory += sizeof(reward)

    sync_threads()

    ob = @cuDynamicSharedMem(Float32, environments.number_inputs, offset_shared_memory)
    offset_shared_memory += sizeof(ob)

    sync_threads()

    action = @cuDynamicSharedMem(Float32, environments.number_outputs, offset_shared_memory)
    offset_shared_memory += sizeof(action)

    sync_threads()

    clicked_gui_elements = @cuDynamicSharedMem(Int32, environments.number_gui_elements, offset_shared_memory)
    offset_shared_memory += sizeof(clicked_gui_elements)

    sync_threads()

    # Initialize brains
    initialize(brains, individuals)

    # Uncheck all checkboxes

    # Initialize inputs
    if threadID <= environments.number_inputs
        ob[threadID] = 0.5
    end

    reward = 0.0

    # Iterate over given number of time steps
    for time_step = 1:environments.number_time_steps

        # Brain step
        step(brains, ob, action, offset_shared_memory)

        # Process mouse click
        process_click(environments.number_gui_elements, action[1], action[2], environments.gui_elements_rectangles, clicked_gui_elements)

    end

    if threadID == 1
        rewards[blockID] = reward
    end

    return
end


function get_memory_requirements(environments::DummyApp)
    return sizeof(Float32) * (1 + environments.number_inputs + environments.number_outputs) + sizeof(Int32) * environments.number_gui_elements  # Reward + Observation + Action
            
end

function get_number_inputs(environments::DummyApp)
    return environments.number_inputs
end

function get_number_outputs(environments::DummyApp)
    return environments.number_outputs
end

function process_click(number_gui_elements, point_x, point_y, rectangles, result)

    threadID = threadIdx().x
    blockID = blockIdx().x

    if threadID <= number_gui_elements
        rect_x = rectangles[threadID, 1, blockID]
        rect_y = rectangles[threadID, 2, blockID]
        width = rectangles[threadID, 3, blockID]
        height = rectangles[threadID, 4, blockID]

        result[threadID] = is_point_in_rect(point_x, point_y, rect_x, rect_y, width, height)
    end

    sync_threads()

    return
end

function is_point_in_rect(point_x, point_y, rect_x, rect_y, width, height)

    x1 = rect_x
    y1 = rect_y

    x2 = x1 + width
    y2 = y1 + height

    return x1 <= point_x <= x2 && y1 <= point_y <= y2
end
