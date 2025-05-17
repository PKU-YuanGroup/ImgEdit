import os
from PIL import Image
from imgedit.comfyui.style_transfer_sdxl import StyleTransferSDXLWorkflow
from tqdm import tqdm
import argparse
from imgedit.test_dlc.master import dlc_context_runner
from openai import OpenAI
import numpy as np
import base64
import io
import json
args = argparse.ArgumentParser()
args.add_argument("--image_txt", type=str, default="", required=True)
args.add_argument("--style_txt", type=str, default="", required=True)
args.add_argument("--base_img_dir", type=str, default="", required=True)
args.add_argument("--result_dir", type=str, default="", required=True)
args = args.parse_args()


def process_task(workflow_endpoint, image_list, style_list, progress_bar=None):
    workflow = StyleTransferSDXLWorkflow(
        endpoint_address=workflow_endpoint,
        workflow_file="/mnt/data/lzj/codes/imgedit_comfyui/workflows/sdxl_style_transfer_ghibli.json",
        comfyui_dir="/mnt/data/lzj/codes/ComfyUI",
    )

    for image_path in image_list:
        task_id = image_path.replace(base_img_dir + "/", "").split(".")[0].replace("/", "_")
        output_dir = os.path.join(result_dir, task_id)
        
        os.makedirs(output_dir, exist_ok=True)

        if os.path.exists(os.path.join(output_dir, "result.json")):
            progress_bar.update(1)
            continue

        try:
            original_image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Process task {task_id} failed: {str(e)}")
            continue
        style = "Studio Ghibli, hand-drawn animation style, masterpiece, high score."
        prompt = f"{style}"
        
        with workflow.task_context():
            try:
                result_id = workflow.submit_task(
                    prompt=prompt,
                    image=original_image,
                )
                result_image, original_image = workflow.get_result(result_id)
                original_image.save(os.path.join(output_dir, "original.png"))
                result_image.save(os.path.join(output_dir, "result.png"))
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"Process task {task_id} failed: {str(e)}")
                continue
        
        with open(os.path.join(output_dir, "result.json"), "w") as f:
            result = {
                "style": style
            }
            f.write(json.dumps(result))
        
        if progress_bar:
            progress_bar.update(1)


def style_transfer(image_list, style_list, workers):
    from concurrent.futures import ThreadPoolExecutor

    with tqdm(total=len(image_list), desc="Processing tasks") as pbar:
        with ThreadPoolExecutor(max_workers=len(workers)) as executor:
            futures = []
            for i in range(len(workers)):
                future = executor.submit(
                    process_task,
                    workers[i],
                    image_list[i :: len(workers)],
                    style_list,
                    pbar,
                )
                futures.append(future)

            for future in futures:
                future.result()


if __name__ == "__main__":
    image_txt = args.image_txt
    style_txt = args.style_txt
    base_img_dir = args.base_img_dir
    result_dir = args.result_dir

    with open(image_txt, "r") as f:
        json_data = json.load(f)
        image_lines = [json_data[i]["path"] for i in range(len(json_data))]
        # image_lines = f.readlines()

    image_list = []
    for image_line in image_lines:
        image_list.append(os.path.join(base_img_dir, image_line.strip()))
    print("Image list length: ", len(image_list))
    
    
    style_list = []
    with open(style_txt, "r") as f:
        style_lines = f.readlines()
    for style_line in style_lines:
        style_list.append(style_line.strip())
    print("Style list length: ", len(style_list))
    dlc_context_runner(style_transfer, wait_time=120, image_list=image_list, style_list=style_list)
    # style_transfer(image_list, style_list, workers=["10.0.59.17:8188"])
