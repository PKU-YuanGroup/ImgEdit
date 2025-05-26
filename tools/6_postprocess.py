import base64
import os
import json
import argparse
from multiprocessing import Pool
from tqdm import tqdm
from openai import OpenAI



from openai import AzureOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

client = OpenAI(
            api_key="your api-key",
            base_url="your base-url"
        )

prompt = """
You are a data rater specializing in grading image editing tasks. You will be given two images (before and after editing) and the corresponding editing instructions. Your task is to evaluate the editing effect on a 5-point scale from two perspectives:
Does the editing strictly follow the instructions?
Is the edited image natural, and are no unintended changes made to the areas that were not requested for editing?
Scoring Criteria:
Score 1 (Following Instructions): Evaluate whether the edited image adheres closely to the instructions:
1 (Poor): There are significant editing errors or the instructions are ignored. The object count is incorrect, and the task objectives are completely failed.
2 (Fair): The instructions are not followed well. Some details are completely off-track, deviating significantly from the original instructions, and the object count is incorrect.
3 (Acceptable): The editing contains noticeable deviations. For example, an object might be placed incorrectly, or the number of objects might differ slightly from the instructions.
4 (Good): The editing is mostly as required, with minor deviations. The number of objects is mostly correct, with only slight differences.
5 (Excellent): The editing is done exactly as per the instructions. There are no extra or omitted objects, and the image is modified without any unnaturalness.
Score 2 (Naturalness and No Unintended Changes): Evaluate whether the edited image looks natural and if the areas that were not to be edited have remained unaffected:
1 (Poor): The editing results in an image that is extremely unnatural. Unintended changes affect areas that were not supposed to be edited.
2 (Fair): The image still looks unnatural overall, with noticeable unintended changes to areas that were not meant to be altered.
3 (Acceptable): The image looks mostly natural, but some minor unintended changes have been made to areas not specified for editing.
4 (Good): The image looks natural overall. There may be one or two minor unintended changes, but they do not significantly affect the result.
5 (Excellent): The image looks completely natural with no unintended changes made to areas that were not requested for editing.
Additionally, please assess if the objects in the edited image match what was described in the instructions. Return a boolean value:
True if the objects are correctly matched.
False if there are discrepancies.
Example Response Format:
Brief reasoning: A short explanation of the score based on the criteria above, no more than 20 words.
Score 1: A number from 1 to 5 (Following instructions).
Score 2: A number from 1 to 5 (Naturalness and no unintended changes).
Objects match: Boolean value (True or False).
The editing instruction is: <edit_prompt>.
Below are the images before and after editing:
"""

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
        # Read editing instructions from JSON file
        with open(edit_prompt_json, 'r') as f:
            data = json.load(f)
        edit_prompt = data['edit_prompt']

        # Convert images to Base64 encoding
        original_image_base64 = image_to_base64(original_image_path)
        result_image_base64 = image_to_base64(result_image_path)

        if not original_image_base64 or not result_image_base64:
            return {"error": "Image conversion failed"}

        

        # API request for evaluating image edit
        response = client.chat.completions.create(
            model="gpt-4o",
            stream=False,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt.replace('<edit_prompt>', edit_prompt)
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{original_image_base64}"},
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{result_image_base64}"},
                    }
                ]
            }]
        )

        return response

    except Exception as e:
        print(f"Error in calling GPT API: {e}")
        raise  

def process_folder(folder, api_key):
    original_png = os.path.join(folder, 'original.png')
    result_png = os.path.join(folder, 'result.png')
    result_json = os.path.join(folder, 'result.json')
    judge_json = os.path.join(folder, 'judge_2scores.json')

    if os.path.exists(judge_json):
        print(f"Judge file has been existed")
        return

    # Check if required files exist
    if os.path.exists(original_png) and os.path.exists(result_png) and os.path.exists(result_json):
        response = call_gpt(original_png, result_png, result_json, api_key)

        judge_dict = {}
        judge_dict['score'] = response.choices[0].message.content

        # Save the response to judge.json
        with open(judge_json, 'w') as f:
            json.dump(judge_dict, f, indent=4)
    else:
        print(f"Some files are missing in {folder}")

# Function to process the directory in parallel without lambda
def process_folder_with_api_key(args):
    folder, api_key = args
    process_folder(folder, api_key)

def process_directory_parallel(parent_folder, num_processes, api_key):
    # List all subfolders
    subfolders = [os.path.join(parent_folder, subfolder) for subfolder in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder, subfolder))]
    
    # Use multiprocessing with tqdm to show progress
    with Pool(num_processes) as pool:
        list(tqdm(pool.imap(process_folder_with_api_key, [(folder, api_key) for folder in subfolders]), total=len(subfolders), desc="Processing Folders"))

def process_directory(parent_folder, api_key):
    # Get a list of all subfolders in the parent directory
    subfolders = [os.path.join(parent_folder, subfolder) for subfolder in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder, subfolder))]
    
    # Use tqdm to show a progress bar for processing subfolders
    for folder in tqdm(subfolders, desc="Processing Subfolders"):
        process_folder(folder, api_key)

# Main function to handle argument parsing and initiate processing
def main():
    parser = argparse.ArgumentParser(description="Process image editing tasks in a directory")
    parser.add_argument('--parent_folder', type=str, default="/mnt/workspace/inpaint", help="Path to the parent folder containing subfolders with image editing pairs")
    parser.add_argument('--num_processes', type=int, default=96, help="Number of processes to use for parallel processing (default: 4)")
    parser.add_argument('--api_key', type=str, required=True, help="API key for OpenAI")

    args = parser.parse_args()

    # Process the directory with the specified number of processes
    process_directory_parallel(args.parent_folder, args.num_processes, args.api_key)

if __name__ == "__main__":
    main()
