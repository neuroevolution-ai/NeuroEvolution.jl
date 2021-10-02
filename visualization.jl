using Luxor

include("tools/play.jl")

function main()

    width = 400
    height = 400
    number_iterations = 1000

    θ = 0

    @play width height number_iterations begin

        background("black")
        sethue("white")
        rotate(θ)
        hypotrochoid(200, 110, 37, :stroke)
        θ += π/120
        sleep(0.01)

    end
end


main()