from imgedit.comfyui.base_workflow import ComfyUIWorkflow
from PIL import Image


class StyleTransferWorkflow(ComfyUIWorkflow):
    def __init__(self, endpoint_address: str, workflow_file: str, comfyui_dir: str):
        super().__init__(endpoint_address, workflow_file, comfyui_dir)

    def _parse_result(self, result, result_id):
        return self.get_image(
            result[result_id]["outputs"]["29"]["images"][0]["filename"]
        ), self.get_image(result[result_id]["outputs"]["60"]["images"][0]["filename"])

    def _parse_workflow(self, prompt: str, image: Image):
        workflow = self.workflow.copy()
        image = image.convert("RGB")
        workflow["35"]["inputs"]["text"] = prompt
        workflow["29"]["inputs"]["filename_prefix"] = self.client_id
        workflow["60"]["inputs"]["filename_prefix"] = self.client_id
        workflow["13"]["inputs"]["image"] = self.upload_image(image)
        return workflow
