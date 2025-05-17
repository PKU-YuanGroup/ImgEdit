import json
import os
from PIL import Image
from schemas import ReplaceObjectTask
from imgedit.utils.mask import rle_to_mask
from imgedit.comfyui.flux_inpaint import InpaintWorkflow
import glob
from tqdm import tqdm
import argparse
from imgedit.test_dlc.master import dlc_context_runner

args = argparse.ArgumentParser()
args.add_argument("--json_dir", type=str, default="", required=True)
args.add_argument("--base_img_dir", type=str, default="", required=True)
args.add_argument("--result_dir", type=str, default="", required=True)
args = args.parse_args()

def process_task(workflow_endpoint, data_list, progress_bar=None):
    workflow = InpaintWorkflow(
        endpoint_address=workflow_endpoint,
        workflow_file="/mnt/data/lzj/codes/imgedit_comfyui/workflows/flux_inpaint.json",
        comfyui_dir="/mnt/data/lzj/codes/ComfyUI",
    )

    for data in data_list:
        with open(data, "r") as f:
            data = json.load(f)
        task = ReplaceObjectTask.model_validate(data)
        task_id = task.original_path.split(".")[0].replace("/", "_")

        output_dir = os.path.join(result_dir, task_id)
        os.makedirs(output_dir, exist_ok=True)

        if os.path.exists(os.path.join(output_dir, "result.json")):
            progress_bar.update(1)
            continue

        original_image = Image.open(os.path.join(base_img_dir, task.original_path))
        task.edit_obj.mask = rle_to_mask(
            task.edit_obj.mask, task.resolution.height, task.resolution.width
        )
        task.edit_prompt = task.edit_prompt.replace("'", "")
        task.edit_result = task.edit_result.replace("'", "")
        task.edit_obj.mask.save(os.path.join(output_dir, "mask.png"))

        with workflow.task_context():
            try:
                result_id = workflow.submit_task(
                    prompt=task.edit_result,
                    image=original_image,
                    mask=task.edit_obj.mask,
                )
                result_image, original_image = workflow.get_result(result_id)
                original_image.save(os.path.join(output_dir, "original.png"))
                result_image.save(os.path.join(output_dir, "result.png"))
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"Process task {task_id} failed: {str(e)}")
                continue

        task.edit_obj.mask = ""
        with open(os.path.join(output_dir, "result.json"), "w") as f:
            json.dump(task.model_dump(), f)

        if progress_bar:
            progress_bar.update(1)


def background_change(data_list, workers):
    from concurrent.futures import ThreadPoolExecutor

    with tqdm(total=len(data_list), desc="Processing tasks") as pbar:
        with ThreadPoolExecutor(max_workers=len(workers)) as executor:
            futures = []
            for i in range(len(workers)):
                future = executor.submit(
                    process_task,
                    workers[i],
                    data_list[i :: len(workers)],
                    pbar,
                )
                futures.append(future)

            for future in futures:
                future.result()
                
if __name__ == "__main__":
    json_dir = args.json_dir
    base_img_dir = args.base_img_dir
    result_dir = args.result_dir

    data_list = []
    for i, path in enumerate(
        glob.iglob(
            f"{json_dir}/**/background.json",
            recursive=True,
        )
    ):
        data_list.append(path)
    print(f"Processing {len(data_list)} tasks")
    
    dlc_context_runner(background_change, wait_time=120, data_list=data_list)