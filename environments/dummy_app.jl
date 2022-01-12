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


function initialize(environments::DummyApp, input, env_seed, offset_shared_memory)

    threadID = threadIdx().x
    blockID = blockIdx().x

    # Uncheck all checkboxes
    if threadID <= environments.number_checkboxes
        environments.checkboxes_checked[threadID, blockID] = false
    end

    # Initialize inputs
    if threadID <= environments.number_inputs
        input[threadID] = 0.5
    end

    sync_threads()

    return
end

function step(environments::DummyApp, action, ob, time_step, offset_shared_memory)

    clicked_gui_elements =
        @cuDynamicSharedMem(Bool, environments.number_gui_elements, offset_shared_memory)

    is_point_in_rect(environments, action, environments.gui_elements_rectangles, clicked_gui_elements)

    sync_threads()

    done = false
    rew = 0.0

    return rew, done

end

function get_memory_requirements(environments::DummyApp)
    return sizeof(Int32) * environments.number_gui_elements
end

function get_number_inputs(environments::DummyApp)
    return environments.number_inputs
end

function get_number_outputs(environments::DummyApp)
    return environments.number_outputs
end

function is_point_in_rect(environments::DummyApp, point, rectangles, result)

    threadID = threadIdx().x
    blockID = blockIdx().x

    if threadID <= environments.number_gui_elements
        x1 = rectangles[threadID, 1, blockID]
        y1 = rectangles[threadID, 2, blockID]
        width = rectangles[threadID, 3, blockID]
        height = rectangles[threadID, 4, blockID]

        x2 = x1 + width
        y2 = y1 + height

        x = point[1]
        y = point[2]

        result[threadID] = x1 < x && x < x2 && y1 < y && y < y2
    end

    return
end


