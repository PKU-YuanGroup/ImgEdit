import base64
import os
import json
import argparse
from openai import OpenAI
from tqdm import tqdm
from tenacity import retry, wait_exponential, stop_after_attempt
from concurrent.futures import ThreadPoolExecutor, as_completed

# 通用评分模板前缀
prompts_json = "/mnt/workspace/hxy/edit_pipeline/bench/judge_prompt.json"
with open(prompts_json, 'r') as f:
    prompts = json.load(f)

def image_to_base64(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"File {image_path} not found.")
        return None

# @retry(wait=wait_exponential(multiplier=1, min=2, max=2), stop=stop_after_attempt(100))
def call_gpt(original_image_path, result_image_path, edit_prompt, edit_type):
    try:
        original_image_base64 = image_to_base64(original_image_path)
        result_image_base64 = image_to_base64(result_image_path)

        if not original_image_base64 or not result_image_base64:
            return {"error": "Image conversion failed"}

        client = OpenAI(
            api_key="sk-rqUkz5hqK2aIlNTFDe0a039e170f4670816bCc7b8324017c",
            base_url="https://api.bltcy.cn/v1"
        )

        prompt = prompts[edit_type]
        full_prompt = prompt.replace('<edit_prompt>', edit_prompt)

        response = client.chat.completions.create(
            model="gpt-4o",
            stream=False,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": full_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{original_image_base64}"}},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{result_image_base64}"}}
                    ]
                }
            ]
        )

        return response
    except Exception as e:
        print(f"Error in calling GPT API: {e}")
        raise

def process_single_item(key, item, result_img_folder, origin_img_root):
    result_img_name = f"{key}.png"
    result_img_path = os.path.join(result_img_folder, result_img_name)
    origin_img_path = os.path.join(origin_img_root, item['id'])
    edit_prompt = item['prompt']
    edit_type = item['edit_type']

    response = call_gpt(origin_img_path, result_img_path, edit_prompt, edit_type)
    return key, response.choices[0].message.content

def process_json(edit_json, result_img_folder, origin_img_root, num_threads):
    with open(edit_json, 'r') as f:
        edit_infos = json.load(f)

    results = {}
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        future_to_key = {
            executor.submit(process_single_item, key, item, result_img_folder, origin_img_root): key
            for key, item in edit_infos.items()
        }

        for future in tqdm(as_completed(future_to_key), total=len(future_to_key), desc="Processing edits"):
            key = future_to_key[future]
            try:
                k, result = future.result()
                results[k] = result
            except Exception as e:
                print(f"Error processing key {key}: {e}")
                results[key] = {"error": str(e)}

    results_path = os.path.join(result_img_folder, 'result.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)

def main():
    parser = argparse.ArgumentParser(description="Evaluate image edits using GPT")
    parser.add_argument('--result_img_folder', type=str, required=True, help="Folder with subfolders of edited images")
    parser.add_argument('--edit_json', type=str, required=True, help="Path to JSON file mapping keys to metadata")
    parser.add_argument('--origin_img_root', type=str, required=True, help="Root path where original images are stored")
    parser.add_argument('--num_processes', type=int, default=32, help="Number of parallel threads")
    args = parser.parse_args()

    process_json(args.edit_json, args.result_img_folder, args.origin_img_root, args.num_processes)

if __name__ == "__main__":
    main()
