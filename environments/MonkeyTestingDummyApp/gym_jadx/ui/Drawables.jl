using CUDA
using Adapt

include("../util/Matrix_utils.jl")
include("../util/Enum_Collection.jl")


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