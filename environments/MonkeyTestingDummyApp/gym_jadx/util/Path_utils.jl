#=

def get_image_path(filename: str, resized=True):
"""
Returns a pathlib.Path object of the specified image file
:param filename: Name of the image file
:param resized: True if the image should be resized, False else
:return: pathlib.Path object of the image
"""
if resized:
    return PathUtils.__get_resized_drawables_path() / filename
else:
    return PathUtils.__get_original_size_drawables_path() / filename
=#

function get_image_path(filename,resized=true)

    if resized
        return "../../resources/drawables/" * filename
    else
        return "../../resources/drawables/size_original/" * filename
    end
end