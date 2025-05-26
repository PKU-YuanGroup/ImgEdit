from .prompt_refiner.nodes import PromptRefiner

prefix = "ImgEdit_"

NODE_CLASS_MAPPINGS = {
    f"{prefix}PromptRefiner": PromptRefiner,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    f"{prefix}PromptRefiner": "Prompt Refiner",
}
