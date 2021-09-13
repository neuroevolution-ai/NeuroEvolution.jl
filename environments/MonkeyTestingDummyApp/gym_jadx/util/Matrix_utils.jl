using Images
using CUDA

function get_array_of_image(filename, resized = true)
    if resized
        image_path = "../../resources/drawables/" * filename
    else
        image_path = "../../resources/drawables/size_original/" * filename
    end
    img = load(image_path)
    return permutedims(channelview(RGB.(img)), (3, 2, 1))
end


function includes_point(click_coordinates, absolute_coordinates, width, height)

    return (
        absolute_coordinates[1] <=
        click_coordinates[1] <=
        (absolute_coordinates[1] + width) &&
        absolute_coordinates[2] <=
        click_coordinates[2] <=
        (absolute_coordinates[2] + height)
    )
end
function includes_point(
    click_coordinate_x,
    click_coordinate_y,
    absolute_coordinate_x,
    absolute_coordinate_y,
    width,
    height,
)

    return (
        absolute_coordinate_x <= click_coordinate_x <= (absolute_coordinate_x + width) &&
        absolute_coordinate_y <= click_coordinate_y <= (absolute_coordinate_y + height)
    )
end

#written for cpu computation, probably needs to be changed for GPU
#function __blit_single_channel_inplace(dest,src,x,y)
#    dest[x:x+size(src,1)-1,y:y+size(src,2)-1] = src
#end

#number threads needs to be bigger than number of rows of src
#TODO make work for src images with more rows than threadnumber
function __blit_single_channel_inplace(threadID, dest, src, channel, x, y)
    temp =
        size(src, 1) % if threadID <= size(src, 1)
            for i = 1:size(src, 2)
                @inbounds dest[x-1+threadID, y-1+i, channel] = src[threadID, i, channel]
            end
        end
    return
end



function kernel_blit_image_inplace(threadID, dest, src, x, y)

    __blit_single_channel_inplace(threadID, dest, src, 1, x, y)
    __blit_single_channel_inplace(threadID, dest, src, 2, x, y)
    __blit_single_channel_inplace(threadID, dest, src, 3, x, y)
    return
end
