from imgedit.comfyui.base_workflow import ComfyUIWorkflow
from PIL import Image


class ExtractRefWorkflow(ComfyUIWorkflow):
    def __init__(self, endpoint_address: str, workflow_file: str, comfyui_dir: str):
        super().__init__(endpoint_address, workflow_file, comfyui_dir)

    def _parse_result(self, result, result_id):
        return self.get_image(
            result[result_id]["outputs"]["61"]["images"][0]["filename"]
        )

    def _parse_workflow(self, prompt: str, image: Image, mask: Image):
        workflow = self.workflow.copy()

        image = image.convert("RGB")
        rgba_image = Image.new("RGBA", image.size)
        rgba_image.paste(image, (0, 0))
        rgba_image.putalpha(mask)
        image = rgba_image

        workflow["15"]["inputs"]["text"] = prompt
        workflow["61"]["inputs"]["filename_prefix"] = self.client_id
        workflow["59"]["inputs"]["image"] = self.upload_image(image)
        return workflow
