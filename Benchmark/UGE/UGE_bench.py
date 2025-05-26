import base64
import os
import json
import argparse
from openai import OpenAI
from tqdm import tqdm
from tenacity import retry, wait_exponential, stop_after_attempt
from concurrent.futures import ThreadPoolExecutor, as_completed

prompt = """
Task: Evaluation of Image Editing Process

The task is to evaluate the editing process with a focus on precision and naturalness. Pay special attention to the following criteria:
Adherence to Instructions: The edited image should strictly reflect the instructions, with no additional or missing objects. Any deviation from the specified number of elements, colors, or other detailed instructions should be clearly noted. Any unintended changes or misplacement of objects would result in a lower score.
Naturalness: The editing should appear seamless and realistic. Avoid any appearance of artificial manipulation, such as abrupt color transitions, distorted objects, or overly sharp contrasts that would make the image appear unnatural. Editing should not be glaring or distracting to the viewer.
Editing Scope: The editing should be focused on the areas specified in the instructions. If the editing is too extensive or intrudes into other regions not specified, this should be noted. Similarly, editing should not appear excessive in areas where minimal changes were requested.
Non-Edited Areas: No changes should occur in the regions of the image that were not specified for editing. If the non-edited areas are altered in any way, the final score cannot exceed 3, regardless of the quality of the editing itself.
Scoring Criteria:
1 (Poor): Major errors, instructions ignored, objects added/removed incorrectly, unnatural, poorly executed.
2 (Fair): Significant deviations from instructions, unnatural edits, unappealing result.
3 (Acceptable): Minor deviations, some unnatural elements, editing is noticeable but not distracting.
4 (Good): Mostly follows instructions, minor naturalness issues, aesthetically acceptable.
5 (Excellent): Perfect adherence to instructions, no unnatural elements, aesthetically pleasing and precise edits.

Editing Instruction: <edit_prompt>.

Below are the images before and after editing:

Example Response Format:
Brief reasoning: A short explanation of the score based on the criteria above, no more than 20 words.
"""

def image_to_base64(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"File {image_path} not found.")
        return None

@retry(wait=wait_exponential(multiplier=1, min=2, max=5), stop=stop_after_attempt(100))
def call_gpt(original_image_path, result_image_path, edit_prompt):
    try:
        original_image_base64 = image_to_base64(original_image_path)
        result_image_base64 = image_to_base64(result_image_path)

        if not original_image_base64 or not result_image_base64:
            return {"error": "Image conversion failed"}

        client = OpenAI(
            api_key="your api-key",
            base_url="https://api.bltcy.cn/v1"
        )

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

    response = call_gpt(origin_img_path, result_img_path, edit_prompt)
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
    parser.add_argument('--result_img_folder', type=str, required=True, help="Folder of edited images")
    parser.add_argument('--edit_json', type=str, required=True, help="Path to JSON file mapping keys to metadata")
    parser.add_argument('--origin_img_root', type=str, required=True, help="Root path where original images are stored")
    parser.add_argument('--num_processes', type=int, default=32, help="Number of parallel threads")
    args = parser.parse_args()

    process_json(args.edit_json, args.result_img_folder, args.origin_img_root, args.num_processes)

if __name__ == "__main__":
    main()
