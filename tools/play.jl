# Original file: https://github.com/JuliaGraphics/Luxor.jl/blob/master/src/play.jl
# Replaced whileloop with forloop in play macro to limit number of displayed frames to n 

using MiniFB

function onclick(window, button, mod, isPressed)::Cvoid
    if Bool(isPressed)
        println("mouse clicked")
    end
    mousex = mfb_get_mouse_x(window)
    mousey = mfb_get_mouse_y(window)
    println("x: $mousex y: $mousey")
end

macro play(w, h, n, body)
    quote
        window = mfb_open_ex("Visualization", $(esc(w)), $(esc(h)), MiniFB.WF_RESIZABLE)

        mfb_set_mouse_button_callback(window, onclick)

        buffer = zeros(UInt32, $(esc(w)), $(esc(h)))
        for i = 1:$(esc(n))
            Drawing($(esc(w)), $(esc(h)), :image)
            origin()
            $(esc(body))
            m = permutedims(image_as_matrix!(buffer), (2, 1))
            finish()
            state = mfb_update(window, m)
            if state != MiniFB.STATE_OK
                break
            end
        end
        mfb_close(window)
    end
end
