using CUDA
using Adapt
#using Random
include("../util/Matrix_utils.jl")
include("Button.jl")



struct Window{A}
    __relative_coordinate_x::Int32
    __relative_coordinate_y::Int32
    __width::Int32
    __height::Int32
    __matrix_self::A
    __current_matrix::A
    __modal::Bool
end

struct Main_window{A}
    __window::Window
    app_close_button::Button
    dropdown_button_datei::Button
    dropdown_button_anzeigen::Button
    dropdown_button_navigation::Button
    dropdown_button_tools::Button
    dropdown_button_hilfe::Button
    small_button_1::Button
    small_button_2::Button
    small_button_3::Button
    small_button_4::Button
    small_button_5::Button
    small_button_6::Button
    small_button_7::Button
    small_button_8::Button
    small_button_9::Button
    small_button_10::Button
    small_button_11::Button
    small_button_12::Button
    small_button_13::Button
end
struct Ueber_window{}
    __window::Window
end
struct Preferences_window{}
    __window::Window
end
struct Dropdown_menu{}
    __window::Window
end
Adapt.@adapt_structure Window
Adapt.@adapt_structure Main_window


function init_window()
    draw_self()
end

#=
function draw_self(matrix_self,children,current_matrix)
    temp = copy(matrix_self)
    for child in children
        draw_self(child.__matrix_self,child.__children,current_matrix)
    end
    current_matrix = temp
end
=#

function draw_self_gpu(window::Window)
    tx = threadIdx().x

    kernel_blit_image_inplace(
        tx,
        window.__current_matrix,
        window.__matrix_self,
        window.__relative_coordinate_x,
        window.__relative_coordinate_y,
    )
    sync_threads()

    #Iterate over all child Buttons in menu and add them to matrix
    #for child in children
    #    draw_self(child,current_matrix)
    #    sync_threads()
    #end
    return
end

#Init main window
####################################
main_window_array = CuArray(get_array_of_image("main_window.png"))
main_window_current = similar(main_window_array)
main_window = Window(
    convert(Int32, 0),
    convert(Int32, 0),
    convert(Int32, size(main_window_array, 2)),
    convert(Int32, size(main_window_array, 1)),
    main_window_array,
    main_window_current,
    true,
)

#Get app_close_button
close_button_large_array = get_array_of_image("close_window_button_large_unclicked.png")
close_button_large_array_clicked = similar(close_button_large_array)
app_close_button = Button(
    convert(Int32, 0),
    convert(Int32, 380),
    convert(Int32, size(close_button_large_array, 2)),
    convert(Int32, size(close_button_large_array, 1)),
    close_button_large_array_clicked,
    close_button_large_array,
    convert(Int32, 2),
    false,
    false,
    placeholder_function,
    true,
)
#display(app_close_button)
position = 0
#Init Dropdown_button_datei
dropdown_datei_unclicked_array = get_array_of_image("drpdwn_datei_unclicked.png")
dropdown_datei_clicked_array = get_array_of_image("drpdwn_datei_clicked.png")
dropdown_button_datei = Button(
    convert(Int32, 12),
    convert(Int32, position),
    convert(Int32, size(dropdown_datei_clicked_array, 2)),
    convert(Int32, size(dropdown_datei_clicked_array, 1)),
    dropdown_datei_clicked_array,
    dropdown_datei_unclicked_array,
    convert(Int32, 2),
    false,
    false,
    placeholder_function,
    true,
)
position +=convert(Int32, size(dropdown_datei_clicked_array, 2))
#init dropdown_button_anzeigen
dropdown_anzeigen_unclicked_array = get_array_of_image("drpdwn_anzeigen_unclicked.png")
dropdown_anzeigen_clicked_array  = get_array_of_image("drpdwn_anzeigen.png")
dropdown_button_anzeigen = Button(
    convert(Int32, 12),
    convert(Int32, position),
    convert(Int32, size(dropdown_anzeigen_clicked_array, 2)),
    convert(Int32, size(dropdown_anzeigen_clicked_array, 1)),
    dropdown_anzeigen_clicked_array,
    dropdown_anzeigen_unclicked_array,
    convert(Int32, 2),
    false,
    false,
    placeholder_function,
    true,
)
position += convert(Int32, size(dropdown_anzeigen_clicked_array, 2))

#init dropdown_button_navigation
dropdown_navigation_unclicked_array  = get_array_of_image("drpdwn_navigation_unclicked.png")
dropdown_navigation_clicked_array  = get_array_of_image("drpdwn_navigation_clicked.png")
dropdown_button_navigation = Button(
    convert(Int32, 12),
    convert(Int32, position),
    convert(Int32, size(dropdown_navigation_clicked_array, 2)),
    convert(Int32, size(dropdown_navigation_clicked_array, 1)),
    dropdown_navigation_clicked_array,
    dropdown_navigation_unclicked_array,
    convert(Int32, 2),
    false,
    false,
    placeholder_function,
    true,
)
position += convert(Int32, size(dropdown_navigation_clicked_array, 2))

#init dropdown_button_tools
dropdown_tools_unclicked_array  = get_array_of_image("drpdwn_tools_unclicked.png")
dropdown_tools_clicked_array  = get_array_of_image("drpdwn_tools_clicked.png")
dropdown_button_tools = Button(
    convert(Int32, 12),
    convert(Int32, position),
    convert(Int32, size(dropdown_tools_clicked_array, 2)),
    convert(Int32, size(dropdown_tools_clicked_array, 1)),
    dropdown_tools_clicked_array,
    dropdown_tools_unclicked_array,
    convert(Int32, 2),
    false,
    false,
    placeholder_function,
    true,
)
position += convert(Int32, size(dropdown_tools_clicked_array, 2))

#init dropdown_button_hilfe
dropdown_hilfe_unclicked_array = get_array_of_image("drpdwn_hilfe_unclicked.png")
dropdown_hilfe_clicked_array = get_array_of_image("drpdwn_hilfe_clicked.png")
dropdown_button_hilfe  = Button(
    convert(Int32, 12),
    convert(Int32, position),
    convert(Int32, size(dropdown_hilfe_clicked_array, 2)),
    convert(Int32, size(dropdown_hilfe_clicked_array, 1)),
    dropdown_hilfe_clicked_array,
    dropdown_hilfe_unclicked_array,
    convert(Int32, 2),
    false,
    false,
    placeholder_function,
    true,
)
#display(dropdown_button_hilfe.__height)
position += convert(Int32, size(dropdown_hilfe_clicked_array, 2))


#init small buttons












