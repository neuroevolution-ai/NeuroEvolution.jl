using CUDA
using Adapt
using ImageView, Images
include("Matrix_utils.jl")
include("Enum_Collection.jl")

struct Image_Container{A}
    button_abbrechen_unclicked::A
    button_aendern_clicked::A
    button_aendern_unclicked::A
    button_bearbeiten_clicked::A
    button_bearbeiten_unclicked::A
    button_clipboard_clicked::A
    button_clipboard_unclicked::A
    button_speichern_unclicked::A
    button_zuruecksetzen_clicked::A
    button_zuruecksetzen_unclicked::A
    chk_clicked::A
    chk_clicked_dis::A
    chk_unclicked::A
    chk_unclicked_dis::A
    close_pref_button::A
    close_ueber_window_button::A
    close_window_button::A
    close_window_button_large_unclicked::A
    close_window_button_unclicked::A
    drpdwn_anzeigen::A
    drpdwn_anzeigen_unclicked::A
    drpdwn_datei_clicked::A
    drpdwn_datei_unclicked::A
    drpdwn_hilfe_clicked::A
    drpdwn_hilfe_unclicked::A
    drpdwn_navigation_clicked::A
    drpdwn_navigation_unclicked::A
    drpdwn_tools_clicked::A
    drpdwn_tools_unclicked::A
    main_window::A   
    menu_button_1::A
    menu_button_1_clicked::A
    menu_button_preferences_clicked::A
    menu_button_preferences_unclicked::A
    menu_ueber_clicked::A
    menu_ueber_unclicked::A
    small_button_1_clicked::A
    small_button_1_unclicked::A
    small_button_2_clicked::A
    small_button_2_unclicked::A
    small_button_3_clicked::A
    small_button_3_unclicked::A
    small_button_4_clicked::A
    small_button_4_unclicked::A
    small_button_5_clicked::A
    small_button_5_unclicked::A
    small_button_6_clicked::A
    small_button_6_unclicked::A
    small_button_7_clicked::A
    small_button_7_unclicked::A
    small_button_8_clicked::A
    small_button_8_unclicked::A
    small_button_9_clicked::A
    small_button_9_unclicked::A
    small_button_10_clicked::A
    small_button_10_unclicked::A
    small_button_11_clicked::A
    small_button_11_unclicked::A
    small_button_12_clicked::A
    small_button_12_unclicked::A
    small_button_13_clicked::A
    small_button_13_unclicked::A
    window_preferences::A
    window_ueber::A

end
Adapt.@adapt_structure Image_Container

function build_image_container()
    #Create Image Container with all image matrices
    button_abbrechen_unclicked = CuArray(get_array_of_image("button_abbrechen_unclicked.png"))
    button_aendern_clicked = CuArray(get_array_of_image("button_aendern_clicked.png"))
    button_aendern_unclicked = CuArray(get_array_of_image("button_aendern_unclicked.png"))
    button_bearbeiten_clicked = CuArray(get_array_of_image("button_bearbeiten_clicked.png"))
    button_bearbeiten_unclicked = CuArray(get_array_of_image("button_bearbeiten_unclicked.png"))
    button_clipboard_clicked = CuArray(get_array_of_image("button_clipboard_clicked.png"))
    button_clipboard_unclicked = CuArray(get_array_of_image("button_clipboard_unclicked.png"))
    button_speichern_unclicked = CuArray(get_array_of_image("button_speichern_unclicked.png"))
    button_zuruecksetzen_clicked = CuArray(get_array_of_image("button_zuruecksetzen_clicked.png"))
    button_zuruecksetzen_unclicked = CuArray(get_array_of_image("button_zuruecksetzen_unclicked.png"))
    chk_clicked = CuArray(get_array_of_image("chk_clicked.png"))
    chk_clicked_dis = CuArray(get_array_of_image("chk_clicked_dis.png"))
    chk_unclicked = CuArray(get_array_of_image("chk_unclicked.png"))
    chk_unclicked_dis = CuArray(get_array_of_image("chk_unclicked_dis.png"))
    close_pref_button = CuArray(get_array_of_image("close_pref_button.png"))
    close_ueber_window_button = CuArray(get_array_of_image("close_ueber_window_button.png"))
    close_window_button = CuArray(get_array_of_image("close_window_button.png"))
    close_window_button_large_unclicked = CuArray(get_array_of_image("close_window_button_large_unclicked.png"))
    close_window_button_unclicked = CuArray(get_array_of_image("close_window_button_unclicked.png"))
    drpdwn_anzeigen = CuArray(get_array_of_image("drpdwn_anzeigen.png"))
    drpdwn_anzeigen_unclicked = CuArray(get_array_of_image("drpdwn_anzeigen_unclicked.png"))
    drpdwn_datei_clicked = CuArray(get_array_of_image("drpdwn_datei_clicked.png"))
    drpdwn_datei_unclicked = CuArray(get_array_of_image("drpdwn_datei_unclicked.png"))
    drpdwn_hilfe_clicked = CuArray(get_array_of_image("drpdwn_hilfe_clicked.png"))
    drpdwn_hilfe_unclicked = CuArray(get_array_of_image("drpdwn_hilfe_unclicked.png"))
    drpdwn_navigation_clicked = CuArray(get_array_of_image("drpdwn_navigation_clicked.png"))
    drpdwn_navigation_unclicked = CuArray(get_array_of_image("drpdwn_navigation_unclicked.png"))
    drpdwn_tools_clicked = CuArray(get_array_of_image("drpdwn_tools_clicked.png"))
    drpdwn_tools_unclicked = CuArray(get_array_of_image("drpdwn_tools_unclicked.png"))
    main_window = CuArray(get_array_of_image("main_window.png"))
    menu_button_1 = CuArray(get_array_of_image("menu_button_1.png"))
    menu_button_1_clicked = CuArray(get_array_of_image("menu_button_1_clicked.png"))
    menu_button_preferences_clicked = CuArray(get_array_of_image("menu_button_preferences_clicked.png"))
    menu_button_preferences_unclicked = CuArray(get_array_of_image("menu_button_preferences_unclicked.png"))
    menu_ueber_clicked = CuArray(get_array_of_image("menu_ueber_clicked.png"))
    menu_ueber_unclicked = CuArray(get_array_of_image("menu_ueber_unclicked.png"))
    small_button_1_clicked = CuArray(get_array_of_image("small_button_1_clicked.png"))
    small_button_1_unclicked = CuArray(get_array_of_image("small_button_1_unclicked.png"))
    small_button_2_clicked = CuArray(get_array_of_image("small_button_2_clicked.png"))
    small_button_2_unclicked = CuArray(get_array_of_image("small_button_2_unclicked.png"))
    small_button_3_clicked = CuArray(get_array_of_image("small_button_3_clicked.png"))
    small_button_3_unclicked = CuArray(get_array_of_image("small_button_3_unclicked.png"))
    small_button_4_clicked = CuArray(get_array_of_image("small_button_4_clicked.png"))
    small_button_4_unclicked = CuArray(get_array_of_image("small_button_4_unclicked.png"))
    small_button_5_clicked = CuArray(get_array_of_image("small_button_5_clicked.png"))
    small_button_5_unclicked = CuArray(get_array_of_image("small_button_5_unclicked.png"))
    small_button_6_clicked = CuArray(get_array_of_image("small_button_6_clicked.png"))
    small_button_6_unclicked = CuArray(get_array_of_image("small_button_6_unclicked.png"))
    small_button_7_clicked = CuArray(get_array_of_image("small_button_7_clicked.png"))
    small_button_7_unclicked = CuArray(get_array_of_image("small_button_7_unclicked.png"))
    small_button_8_clicked = CuArray(get_array_of_image("small_button_8_clicked.png"))
    small_button_8_unclicked = CuArray(get_array_of_image("small_button_8_unclicked.png"))
    small_button_9_clicked = CuArray(get_array_of_image("small_button_9_clicked.png"))
    small_button_9_unclicked = CuArray(get_array_of_image("small_button_9_unclicked.png"))
    small_button_10_clicked = CuArray(get_array_of_image("small_button_10_clicked.png"))
    small_button_10_unclicked = CuArray(get_array_of_image("small_button_10_unclicked.png"))
    small_button_11_clicked = CuArray(get_array_of_image("small_button_11_clicked.png"))
    small_button_11_unclicked = CuArray(get_array_of_image("small_button_11_unclicked.png"))
    small_button_12_clicked = CuArray(get_array_of_image("small_button_12_clicked.png"))
    small_button_12_unclicked = CuArray(get_array_of_image("small_button_12_unclicked.png"))
    small_button_13_clicked = CuArray(get_array_of_image("small_button_13_clicked.png"))
    small_button_13_unclicked = CuArray(get_array_of_image("small_button_13_unclicked.png"))
    window_preferences = CuArray(get_array_of_image("window_preferences.png"))
    window_ueber = CuArray(get_array_of_image("window_ueber.png"))


    
    resized_image_container = Image_Container(
        button_abbrechen_unclicked,
        button_aendern_clicked,
        button_aendern_unclicked,
        button_bearbeiten_clicked,
        button_bearbeiten_unclicked,
        button_clipboard_clicked,
        button_clipboard_unclicked,
        button_speichern_unclicked,
        button_zuruecksetzen_clicked,
        button_zuruecksetzen_unclicked,
        chk_clicked,
        chk_clicked_dis,
        chk_unclicked,
        chk_unclicked_dis,
        close_pref_button,
        close_ueber_window_button,
        close_window_button,
        close_window_button_large_unclicked,
        close_window_button_unclicked,
        drpdwn_anzeigen,
        drpdwn_anzeigen_unclicked,
        drpdwn_datei_clicked,
        drpdwn_datei_unclicked,
        drpdwn_hilfe_clicked,
        drpdwn_hilfe_unclicked,
        drpdwn_navigation_clicked,
        drpdwn_navigation_unclicked,
        drpdwn_tools_clicked,
        drpdwn_tools_unclicked,
        main_window,
        menu_button_1,
        menu_button_1_clicked,
        menu_button_preferences_clicked,
        menu_button_preferences_unclicked,
        menu_ueber_clicked,
        menu_ueber_unclicked,
        small_button_1_clicked,
        small_button_1_unclicked,
        small_button_2_clicked,
        small_button_2_unclicked,
        small_button_3_clicked,
        small_button_3_unclicked,
        small_button_4_clicked,
        small_button_4_unclicked,
        small_button_5_clicked,
        small_button_5_unclicked,
        small_button_6_clicked,
        small_button_6_unclicked,
        small_button_7_clicked,
        small_button_7_unclicked,
        small_button_8_clicked,
        small_button_8_unclicked,
        small_button_9_clicked,
        small_button_9_unclicked,
        small_button_10_clicked,
        small_button_10_unclicked,
        small_button_11_clicked,
        small_button_11_unclicked,
        small_button_12_clicked,
        small_button_12_unclicked,
        small_button_13_clicked,
        small_button_13_unclicked,
        window_preferences,
        window_ueber)

    return resized_image_container    
end

function get_image(name::Enum,container::Image_Container, clicked::Bool=false)
    if name == main_window
        return container.main_window
    elseif name == window_preferences
        return container.window_preferences
    elseif name == window_ueber
        return container.window_ueber
    elseif name == button_abbrechen
        return container.button_abbrechen_unclicked
    elseif name == app_close_button 
        if !clicked
            return container.close_window_button_large_unclicked
        end
        return container.close_window_button
    elseif name == button_aendern 
        if !clicked
            return container.button_aendern_unclicked
        end
        return container.button_aendern_clicked
    elseif name == button_bearbeiten 
        if !clicked
            return container.button_bearbeiten_unclicked
        end
        return container.button_bearbeiten_clicked
    elseif name == button_clipboard 
        if !clicked
            return container.button_clipboard_unclicked
        end
        return container.button_clipboard_clicked
    elseif name == button_speichern 
        return container.button_speichern_unclicked
    elseif name == button_zuruecksetzen 
        if !clicked
            return container.button_zuruecksetzen_unclicked
        end
        return container.button_zuruecksetzen_clicked
    elseif name == dropdown_button_anzeigen 
        if !clicked
            return container.drpdwn_anzeigen_unclicked
        end
        return container.drpdwn_anzeigen
    elseif name == dropdown_button_datei 
        if !clicked
            return container.drpdwn_datei_unclicked
        end
        return container.drpdwn_datei_clicked
    elseif name == dropdown_button_hilfe 
        if !clicked
            return container.drpdwn_hilfe_unclicked
        end
        return container.drpdwn_hilfe_clicked
    elseif name == dropdown_button_navigation 
        if !clicked
            return container.drpdwn_navigation_unclicked
        end
        return container.drpdwn_navigation_clicked
    elseif name == dropdown_button_tools 
        if !clicked
            return container.drpdwn_tools_unclicked
        end
        return container.drpdwn_tools_clicked
    elseif name == menu_button_1 
        if !clicked
            return container.menu_button_1
        end
        return container.menu_button_1_clicked
    elseif name == menu_button_preferences 
        if !clicked
            return container.menu_button_preferences_unclicked
        end
        return container.menu_button_preferences_clicked
    elseif name == menu_ueber 
        if !clicked
            return container.menu_ueber_unclicked
        end
        return container.menu_ueber_clicked
    elseif name == small_button_1 
        if !clicked
            return container.small_button_1_unclicked
        end
        return container.small_button_1_clicked
    elseif name == small_button_2 
        if !clicked
            return container.small_button_2_unclicked
        end
        return container.small_button_2_unclicked
    elseif name == small_button_3 
        if !clicked
            return container.small_button_3_unclicked
        end
        return container.small_button_3_clicked
    elseif name == small_button_4 
        if !clicked
            return container.small_button_4_unclicked
        end
        return container.small_button_4_clicked
    elseif name == small_button_5 
        if !clicked
            return container.small_button_5_unclicked
        end
        return container.small_button_5_clicked
    elseif name == small_button_6 
        if !clicked
            return container.small_button_6_unclicked
        end
        return container.small_button_6_clicked
    elseif name == small_button_7 
        if !clicked
            return container.small_button_7_unclicked
        end
        return container.small_button_7_clicked
    elseif name == small_button_8 
        if !clicked
            return container.small_button_8_unclicked
        end
        return container.small_button_8_clicked
    elseif name == small_button_9 
        if !clicked
            return container.small_button_9_unclicked
        end
        return container.small_button_9_clicked
    elseif name == small_button_10 
        if !clicked
            return container.small_button_10_unclicked
        end
        return container.small_button_10_clicked
    elseif name == small_button_11
        if !clicked
            return container.small_button_11_unclicked
        end
        return container.small_button_11_clicked
    elseif name == small_button_12
        if !clicked
            return container.small_button_12_unclicked
        end
        return container.small_button_12_clicked
    elseif name == small_button_13 
        if !clicked
            return container.small_button_13_unclicked
        end
        return container.small_button_13_clicked
    end
end
