import base64
from pathlib import Path  

import os
import json
import argparse
from openai import OpenAI
from multiprocessing import Pool
from tqdm import tqdm
# from tenacity import retry, wait_exponential, stop_after_attempt
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from azure.identity import (
    AzureCliCredential,
    ChainedTokenCredential,
    DefaultAzureCredential,
    get_bearer_token_provider,
)
from openai import AzureOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

api_version = "2024-02-15-preview"
api_version="2024-12-01-preview"

# gpt_model_name = "gpt-4o_2024-08-06"
# gpt_model_name = "gpt-4o-mini_2024-07-18"#"gpt-4.1-nano_2025-04-14"

azure_credential = ChainedTokenCredential(
    AzureCliCredential(),
    DefaultAzureCredential(
        exclude_cli_credential=True,
        # Exclude other credentials we are not interested in.
        exclude_environment_credential=True,
        exclude_shared_token_cache_credential=True,
        exclude_developer_cli_credential=True,
        exclude_powershell_credential=True,
        exclude_interactive_browser_credential=True,
        exclude_visual_studio_code_credentials=True,
        # DEFAULT_IDENTITY_CLIENT_ID is a variable exposed in
        # Azure ML Compute jobs that has the client id of the
        # user-assigned managed identity in it.
        # See https://learn.microsoft.com/en-us/azure/machine-learning/how-to-identity-based-service-authentication#compute-cluster
        # In case it is not set the ManagedIdentityCredential will
        # default to using the system-assigned managed identity, if any.
        managed_identity_client_id=os.environ.get("DEFAULT_IDENTITY_CLIENT_ID"),
    ),
)

token_provider = get_bearer_token_provider(azure_credential, "https://cognitiveservices.azure.com/.default")

aoiclient = AzureOpenAI(
    azure_endpoint="https://gcraoai9sw1.openai.azure.com/",
    azure_ad_token_provider=token_provider,
    api_version=api_version,
    max_retries=5,
)

prompt = """
You are a data rater specializing in grading image editing tasks. You will be given one image and a corresponding Cut-out object, which means this object have been cutout from original image. The right side of image is the origin and the left side is the object extract from it. Your task is to evaluate the Cut-out editing effect on a 5-point scale from three perspectives:
    Object Selection & Identity
        1 Wrong object or multiple objects extracted.
        2 Correct class but only part of the object, or obvious intrusions from other items.
        3 Object largely correct yet small pieces missing / extra, identity still recognisable.
        4 Full object with clear identity; only tiny mis-crop (e.g., tip of antenna).
        5 Exact requested object, complete and unmistakably the same instance (ID).
    Mask Precision & Background Purity
        1 Large background remnants, holes in mask, or non-white backdrop dominates.
        2 Noticeable jagged edges, colour fringes, grey/colour patches in white area.
        3 Acceptable mask; minor edge softness or faint halo visible on close look.
        4 Clean, smooth edges; white (#FFFFFF) background uniform, tiny artefacts only when zoomed.
        5 Crisp anti-aliased contour, zero spill or halo; backdrop perfectly pure white throughout.
    Object Integrity & Visual Quality
        1 Severe blur, compression, deformation, or missing parts; unusable.
        2 Moderate noise, colour shift, or slight warping; details clearly degraded.
        3 Overall intact with minor softness or noise; colours mostly preserved.
        4 Sharp detail, accurate colours; negligible artefacts.
        5 Pristine: high-resolution detail, true colours, no artefacts or distortion.
The second and third score should no higher than first score!!!
Example Response Format:
Brief reasoning: A short explanation of the score based on the criteria above, no more than 20 words.
Object Selection & Identity: A number from 1 to 5, no description.
Mask Precision & Background Purity: A number from 1 to 5, no description.
Object Integrity & Visual Quality: A number from 1 to 5, no description.
The Cut-out object is: <edit_prompt>.
Below are the image:
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
def call_gpt(result_image_path, edit_prompt_json, api_key):
    try:
        # Read editing instructions from JSON file
        with open(edit_prompt_json, 'r') as f:
            data = json.load(f)
        edit_prompt = data['edit_result']

        # Convert images to Base64 encoding
        # original_image_base64 = image_to_base64(original_image_path)
        result_image_base64 = image_to_base64(result_image_path)
        # print("dddddddddd")
        if not result_image_base64:
            return {"error": "Image conversion failed"}

        # Initialize OpenAI client
        # base_url = "https://api.bltcy.ai/v1"

        # client = OpenAI(
        #     api_key=api_key,
        #     base_url=base_url
        # )

        # API request for evaluating image edit
        # print(gpt_model_name)
        response = aoiclient.chat.completions.create(
            model = gpt_model_name, #"gpt-4o_2024-08-06", #'gpt-4o-mini_2024-07-18', # 
            # stream=False,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt.replace('<edit_prompt>', edit_prompt)
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
        raise  # Reraise the exception to trigger a retry

# Function to process each folder (image set)
def process_folder(folder, api_key):
    # original_png = os.path.join(folder, 'original.png')
    result_png = os.path.join(folder, 'result.png')
    result_json = os.path.join(folder, 'result.json')
    dirr = os.path.join(api_key, Path(folder).name)
    os.makedirs(dirr, exist_ok=True)
    judge_json = os.path.join(dirr, 'judge_2scores.json')

    if os.path.exists(judge_json):
        print(f"Judge file has been existed")
        return

    # Check if required files exist
    if os.path.exists(result_png) and os.path.exists(result_json):
        response = call_gpt(result_png, result_json, api_key)

        judge_dict = {}
        judge_dict['score'] = response.choices[0].message.content
        judge_dict['folder'] = folder

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
    parser.add_argument('--parent_folder', type=str, default="/mnt/workspace/hxy/edit_pipeline/data/sample/inpaint", help="Path to the parent folder containing subfolders with image editing tasks")
    parser.add_argument('--num_processes', type=int, default=96, help="Number of processes to use for parallel processing (default: 4)")
    parser.add_argument('--api_key', type=str, required=True, help="API key for OpenAI")
    parser.add_argument('--model', type=str, required=True, help="API key for OpenAI")
    args = parser.parse_args()
    global gpt_model_name
    gpt_model_name = args.model
    # Process the directory with the specified number of processes
    process_directory_parallel(args.parent_folder, args.num_processes, args.api_key)

if __name__ == "__main__":
    main()

