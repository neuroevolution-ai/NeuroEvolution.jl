using Images
using CUDA

function get_array_of_image(filename, resized = true)
    if resized
        image_path = "../../resources/drawables/" * filename
    else
        image_path = "../../resources/drawables/size_original/" * filename
    end
    img = load(image_path)
    return channelview(RGB.(img))
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
    if threadID <= size(src, 3)
        #@cushow(x-1+threadID)
        for i = 1:size(src, 2)
            if threadID == 1
                #@cushow(y-1+i)
            end
            dest[channel, y-1+i, x-1+threadID] = src[channel,i, threadID]
        end
    end
    return
end
function __blit_single_channel_inplace(threadID,blockID, dest, src, channel, x, y)
    if ndims(src) == 3
        if threadID <= size(src, 3)
            #@cushow(x-1+threadID)
            for i = 1:size(src, 2)
                if threadID == 1
                    #@cushow(y-1+i)
                end
                dest[channel, y-1+i, x-1+threadID,blockID] = src[channel,i, threadID]
            end
        end
    else 
        if threadID <= size(src, 3)
            #@cushow(x-1+threadID)
            for i = 1:size(src, 2)
                if threadID == 1
                    #@cushow(y-1+i)
                end
                dest[channel, y-1+i, x-1+threadID,blockID] = src[channel,i, threadID,blockID]
            end
        end
    end
    return
end


#function kernel_blit_image_inplace(threadID, dest, src, x, y)

#    __blit_single_channel_inplace(threadID, dest, src, 1, x, y)          
#    __blit_single_channel_inplace(threadID, dest, src, 2, x, y)
#    __blit_single_channel_inplace(threadID, dest, src, 3, x, y)
#    return
#end
function kernel_blit_image_inplace(threadID,blockID, dest, src, x, y)

    __blit_single_channel_inplace(threadID, blockID, dest, src, 1, x, y)          
    __blit_single_channel_inplace(threadID, blockID, dest, src, 2, x, y)
    __blit_single_channel_inplace(threadID, blockID, dest, src, 3, x, y)
    return
end

####
function get_blank_image_as_array(color_red,color_green,_color_blue,width,height)

    result_array = @cuStaticSharedMem(Float32,(height,width,3))
    function fill_array(array,color,channel)
        for i in 1:height
            for j in 1:width
                array[channel,i ,j] = color
            end
        end
    end
    fill_array(result_array,color_red,1)
    fill_array(result_array,color_green,2)
    fill_array(result_array,_color_blue,3)
end
