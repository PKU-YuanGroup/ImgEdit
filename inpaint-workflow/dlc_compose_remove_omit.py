import json
import os
from PIL import Image
from schemas import OmitTask
from imgedit.utils.mask import rle_to_mask
from imgedit.comfyui.sdxl_inpaint import InpaintWorkflow
import glob
from tqdm import tqdm
import argparse
from imgedit.test_dlc.master import dlc_context_runner

args = argparse.ArgumentParser()
args.add_argument("--json_dir", type=str, default="", required=True)
args.add_argument("--base_img_dir", type=str, default="", required=True)
args.add_argument("--result_dir", type=str, default="", required=True)
args.add_argument("--task_type", type=str, default="omit-refer")

args = args.parse_args()

def process_task(workflow_endpoint, data_list, progress_bar=None):
    workflow = InpaintWorkflow(
        endpoint_address=workflow_endpoint,
        workflow_file="/mnt/data/lzj/codes/imgedit_comfyui/workflows/sdxl_inpaint.json",
        comfyui_dir="/mnt/data/lzj/codes/ComfyUI",
    )

    for data in data_list:
        try:
            with open(data, "r") as f:
                data = json.load(f)
            task = OmitTask.model_validate(data)
        except Exception as e:
            print(f"Error loading {data}: {str(e)}")
            continue
        
        task_id = task.original_path.split(".")[0].replace("/", "_")
        output_dir = os.path.join(result_dir, task_id)
        os.makedirs(output_dir, exist_ok=True)
        
        # If the task has been processed, skip it
        if os.path.exists(os.path.join(output_dir, "result_2.png")):
            progress_bar.update(1)
            continue
        
        if task.edit_type[2] == "remove":
            if not os.path.exists(os.path.join(output_dir, "result_1.png")):
                progress_bar.update(1)
                continue
            original_path = task.original_path
            edit_obj = task.edit_obj
            origin_path = "origin_2.png"
            result_path = "result_2.png"
            mask_path = "mask_2.png"
            final_task = True
            try:
                original_image = Image.open(os.path.join(base_img_dir, original_path))
            except Exception as e:
                print(f"Error loading {original_path}: {str(e)}")
                progress_bar.update(1)
                continue
        else:
            progress_bar.update(1)
            continue

        mask = rle_to_mask(
            edit_obj.mask, task.resolution.height, task.resolution.width
        )
        mask = mask.resize((original_image.width, original_image.height))
        mask.save(os.path.join(output_dir, mask_path))

        with workflow.task_context():
            try:
                result_id = workflow.submit_task(
                    prompt="empty, emptiness, blank",
                    image=original_image,
                    mask=mask,
                )
                result_image, original_image = workflow.get_result(result_id)
                original_image.save(os.path.join(output_dir, origin_path))
                result_image.save(os.path.join(output_dir, result_path))
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"Process task {task_id} failed: {str(e)}")
                continue
        
        if final_task:
            task.edit_obj.mask = ""
            with open(os.path.join(output_dir, "result.json"), "w") as f:
                json.dump(task.model_dump(), f)

        if progress_bar:
            progress_bar.update(1)

def replace(data_list, workers):
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
            f"{json_dir}/**/{args.task_type}.json",
            recursive=True,
        )
    ):
        data_list.append(path)
    print(f"Processing {len(data_list)} tasks")
    
    dlc_context_runner(replace, wait_time=180, data_list=data_list)