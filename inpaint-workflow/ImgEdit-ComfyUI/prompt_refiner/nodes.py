import os
from openai import OpenAI
from comfy.comfy_types import IO
import time

api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL")
model_name = os.getenv("OPENAI_MODEL_NAME")


class PromptRefiner:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": (IO.STRING, {"multiline": True}),
            }
        }

    RETURN_TYPES = (IO.STRING,)
    CATEGORY = "prompt_refiner"
    FUNCTION = "refine"

    def refine(self, prompt: str):
        client = OpenAI(api_key=api_key, base_url=base_url)
        start_time = time.time()
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": f"{prompt}\n扩写一下上面的细节和描述，用英文。这段文字用于描述一个图片中的主体，因此只需要扩写对于主体的描述，不需要考虑场景。注意只需要返回扩写后的结果，不要输出任何其他内容。",
                }
            ],
        )
        end_time = time.time()
        print(f"Prompt refinement time: {end_time - start_time} seconds")
        return (response.choices[0].message.content,)
