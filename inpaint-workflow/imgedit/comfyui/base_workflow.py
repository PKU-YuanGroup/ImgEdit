import uuid
import websocket
import json
import urllib.request
import os
from PIL import Image
import shutil
from pathlib import Path
import time

class ComfyUIWorkflow:
    def __init__(self, endpoint_address: str, workflow_file: str, comfyui_dir: str):
        self.endpoint_address = endpoint_address
        self.upload_img_dir = os.path.join(comfyui_dir, "input")
        self.output_img_dir = os.path.join(comfyui_dir, "output")

        self.client_id = None
        with open(workflow_file, "r") as f:
            self.workflow = json.load(f)

    def upload_image(self, image: Image):
        assert self.client_id is not None, "Client ID is not set"
        
        upload_image_path = self.upload_img_dir + f"/{self.client_id}.png"
        image.save(upload_image_path, format="PNG")
        return upload_image_path

    def task_context(self):
        class TaskContext:
            def __init__(self, workflow: ComfyUIWorkflow, client_id: str):
                self.workflow = workflow
                self.client_id = client_id

            def __enter__(self):
                self.workflow.client_id = str(uuid.uuid4())
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                self.clear_cache()

            def clear_cache(self):
                all_input_image = Path(self.workflow.upload_img_dir).glob(f"{self.workflow.client_id}.png")
                all_output_image = Path(self.workflow.output_img_dir).glob(f"{self.workflow.client_id}*.png")
                for image in all_input_image:
                    os.remove(str(image))
                for image in all_output_image:
                    os.remove(str(image))
                self.workflow.client_id = None

        return TaskContext(self, self.client_id)

    def get_image(self, image_path: str):
        image_path = self.output_img_dir + f"/{image_path}"
        return Image.open(image_path)

    def submit_task(self, **kwargs):
        assert self.client_id is not None, "Client ID is not set"
        
        workflow = self._parse_workflow(**kwargs)
        p = {"prompt": workflow, "client_id": self.client_id}
        data = json.dumps(p).encode("utf-8")
        req = urllib.request.Request(
            "http://{}/prompt".format(self.endpoint_address), data=data
        )
        try:
            resp = json.loads(urllib.request.urlopen(req).read())
            prompt_id = resp["prompt_id"]
        except Exception as e:
            print(f"Request http://{self.endpoint_address}/prompt failed: {e}")
            raise e

        ws = websocket.WebSocket()
        ws.connect(
            "ws://{}/ws?clientId={}".format(self.endpoint_address, self.client_id)
        )

        start_time = time.time()
        max_time = 300

        while True:
            if time.time() - start_time > max_time:
                ws.close()
                raise TimeoutError("任务执行超时（超过5分钟）")
            out = ws.recv()
            if isinstance(out, str):
                message = json.loads(out)
                if message["type"] == "executing":
                    data = message["data"]
                    if data["node"] is None and data["prompt_id"] == prompt_id:
                        ws.close()
                        return prompt_id
            time.sleep(0.1)

    def get_result(self, result_id):
        with urllib.request.urlopen(
            "http://{}/history/{}".format(self.endpoint_address, result_id)
        ) as response:
            result = json.loads(response.read())
        return self._parse_result(result, result_id)

    def _parse_workflow(self, **kwargs):
        return self.workflow

    def _parse_result(self, result, result_id):
        pass


if __name__ == "__main__":
    workflow = ComfyUIWorkflow(
        "127.0.0.1:8188",
        "/mnt/data/lzj/codes/imgedit_comfyui/workflows/inpaint.json",
    )
