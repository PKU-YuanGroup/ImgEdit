import os
from openai import OpenAI
from comfy.comfy_types import IO
import time

class ImageResizeToVAESize:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": (IO.IMAGE, ),
                "divide_by": (IO.INT, {"default": 16, "min": 1, "max": 100, "step": 1}),
                "shortest_side": (IO.INT, {"default": 512, "min": 1, "max": 10000, "step": 1}),
            }
        }

    RETURN_TYPES = (IO.IMAGE,)
    CATEGORY = "image_utils"
    FUNCTION = "resize"

    def resize(self, image: Image.Image, divide_by: int = 16, shortest_side: int = 512):
        width, height = image.size
        
        if width < height:
            new_width = shortest_side
            new_height = int(height * (shortest_side / width))
        else:
            new_height = shortest_side 
            new_width = int(width * (shortest_side / height))
        
        image = image.resize((new_width, new_height))
        
        width, height = image.size
        crop_width = (width // divide_by) * divide_by
        crop_height = (height // divide_by) * divide_by
        
        left = (width - crop_width) // 2
        top = (height - crop_height) // 2
        right = left + crop_width
        bottom = top + crop_height
        
        image = image.crop((left, top, right, bottom))
        
        return (image,)
