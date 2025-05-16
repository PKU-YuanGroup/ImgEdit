import base64
import os
import json
import argparse
from multiprocessing import Pool
from tenacity import retry, wait_exponential, stop_after_attempt
import re
from tqdm import tqdm
import argparse
from openai import AzureOpenAI 
from azure.identity import AzureCliCredential, ChainedTokenCredential, DefaultAzureCredential, get_bearer_token_provider
from tenacity import retry, stop_after_attempt, wait_exponential
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed


prompt = """
You are a data rater specializing in generate image editing prompt. Given two images of the same person—before and after editing—write a concise prompt that describes only the change in the subject's action, pose, or facial expression. Do not mention any environmental or background details. For example: 'A man swings a golf club.'
Below are the images before and after editing:
"""


# api_version=
model_name='gpt-4o_2024-11-20' #'gpt-4.1_2025-04-14' #'gpt-4o_2024-11-20' #'gpt-4o-mini_2024-07-18'



# Function to convert an image file to Base64
def image_to_base64(image_path):
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return encoded_string
    except FileNotFoundError:
        print(f"File {image_path} not found.")
        return None

# Retry decorator with exponential backoff for call_gpt
@retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(100))
def call_gpt(original_image_path, result_image_path, edit_prompt_json):
    try:
        # Convert images to Base64 encoding
        import pdb; pdb.set_trace()
        original_image_base64 = image_to_base64(original_image_path)
        result_image_base64 = image_to_base64(result_image_path)
        print(original_image_base64)
        if not original_image_base64 or not result_image_base64:
            return {"error": "Image conversion failed"}

        response = aoiclient.chat.completions.create(
            model='gpt-4o_2024-11-20',#'gpt-4.1_2025-04-14',
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    # {
                    #     "type": "image_url",
                    #     "image_url": {"url": f"data:image/jpg;base64,{original_image_base64}"},
                    # },
                    # {
                    #     "type": "image_url",
                    #     "image_url": {"url": f"data:image/jpg;base64,{result_image_base64}"},
                    # }
                ]
            }],
            max_tokens=1024,
        )

        return response.choices[0].message.content

    except Exception as e:
        print(f"Error in calling GPT API: {e}")
        raise  # Reraise the exception to trigger a retry


def process_json(json_path, image_folder):  
    """  
    单个 json 文件的处理函数。  
    注意：必须放在顶层（不能嵌套在函数里），  
    否则 multiprocessing 在 Windows 下会 pickling 失败。  
    """  
    # json_path, image_folder = args          # 拿到参数  
    try:  
        with open(json_path, 'r', encoding='utf-8') as f:  
            data = json.load(f)  
    except Exception as e:  
        print(f"[ERROR] 解析 {json_path} 失败: {e}")  
        return  
  
    # 取出 start_frame / end_frame 相对路径  
    meta = data.get("metadata", {}).get("image_paths", {})  
    start_rel = meta.get("start_frame")  
    end_rel   = meta.get("end_frame")  
  
    if not (start_rel and end_rel):  
        print(f"[WARN] {json_path} 缺少 start_frame / end_frame 字段")  
        return  
  
    start_abs = os.path.join(image_folder, start_rel)  
    end_abs   = os.path.join(image_folder, end_rel)  
  
    if not (os.path.exists(start_abs) and os.path.exists(end_abs)):  
        print(f"[WARN] {json_path} 对应的图片不存在")  
        return  
  
    # 调用 GPT  
    # import pdb; pdb.set_trace()
    response = call_gpt(start_abs, end_abs)  
    print(response)
    data["prompt"] = response  
  
    # 回写 json  
    try:  
        with open(json_path, 'w', encoding='utf-8') as f:  
            json.dump(data, f, ensure_ascii=False, indent=4)  
        return json_path
    except Exception as e:  
        print(f"[ERROR] 写回 {json_path} 失败: {e}")  
        return None
  
  
def process_directory_parallel(json_folder: str,  
                               image_folder: str,  
                               num_processes: int | None = None):  
    """  
    并行遍历并处理 json 文件  
    """  
    json_files = list(Path(json_folder).rglob("*.json"))  
    if not json_files:  
        print("未找到任何 json 文件")  
        return  
  
    num_processes = num_processes or cpu_count()  
    params = [(str(p), image_folder) for p in json_files]  
  
    with Pool(processes=num_processes) as pool:  
        list(tqdm(pool.imap_unordered(process_json, params),  
                  total=len(params),  
                  desc="Processing jsons"))  
  

    
# Main function to handle argument parsing and initiate processing
def main():
    parser = argparse.ArgumentParser(description="Process image editing tasks in a directory")
    parser.add_argument('--json_folder', type=str)
    parser.add_argument('--image_folder', type=str, help="Path to the images containing subfolders with image editing tasks")
    parser.add_argument('--num_processes', type=int, default=1, help="Number of processes to use for parallel processing (default: 4)")

    args = parser.parse_args()

    # Process the directory with the specified number of processes
    # process_directory_parallel(args.json_folder, args.image_folder, args.num_processes)
    json_files = [
        os.path.join(args.json_folder, f)
        for f in os.listdir(args.json_folder)
        if f.endswith(".json")
    ]
    import pdb; pdb.set_trace()
    with ProcessPoolExecutor(max_workers=args.num_processes) as executor:
        future_to_file = {executor.submit(process_json, json_file, args.image_folder): json_file for json_file in json_files}
        for future in tqdm(as_completed(future_to_file), total=len(future_to_file), desc="Processing Files"):
            json_file = future_to_file[future]
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
                else:
                    logging.warning(f"Failed to process file: {json_file}")
            except Exception as exc:
                logging.error(f"File {json_file} generated an exception: {exc}", exc_info=True)
    
    logging.info("Processing complete.")
if __name__ == "__main__":
    main()
