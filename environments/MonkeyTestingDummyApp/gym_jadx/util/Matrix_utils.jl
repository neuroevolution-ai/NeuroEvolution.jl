using Images


include("Path_utils.jl")


function get_array_of_image(filename,resized = true)
    image_path = get_image_path(filename,resized)
    img = load(image_path)
    img_RGB = RGB.(img)
    return permutedims(channelview(img_RGB),(3,2,1))
end



get_array_of_image("menu_über_unclicked.png")