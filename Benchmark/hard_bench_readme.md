
# Image Editing Evaluation using GPT

This project evaluates image editing processes using GPT-4o. The system processes a set of original and edited images, comparing them according to a predefined set of criteria, such as instruction adherence, image-editing quality, and detail preservation.

## Overview

The goal of this project is to evaluate the quality of image editing processes using GPT. The evaluation criteria include:
- **Instruction Adherence**: The edit must match the specified editing instructions.
- **Image-editing Quality**: The edit should appear seamless and natural.
- **Detail Preservation**: Regions not specified for editing should remain unchanged.


## Dependencies

The following Python libraries are required for running the script:
- `openai`: For interacting with the OpenAI API.

Install the required dependencies using `pip`:

```bash
pip install base64 tqdm tenacity openai
```

## Setup

1. **OpenAI API Key**: Make sure you have a valid OpenAI API key. Replace `"your api-key"` in the `call_gpt` function with your actual key.

2. **Images and JSON File**: You will need:
   - A folder containing the edited images (`--result_img_folder`).
   - A JSON file mapping keys to metadata and prompts for each image edit (`--edit_json`).
   - A root directory where the original images are stored (`--origin_img_root`).



## Usage

To run the script, use the following command:

```bash
python evaluate_edits.py --result_img_folder <path_to_edited_images> --edit_json <path_to_edit_json> --origin_img_root <path_to_original_images> --num_processes <number_of_threads>
```

### Arguments:
- `--result_img_folder`: The directory containing the edited images.
- `--edit_json`: Path to the JSON file containing metadata and edit instructions.
- `--origin_img_root`: The root directory of the original images.
- `--num_processes`: The number of threads to use for processing. Default is 32.

### Example:

```bash
python evaluate_edits.py --result_img_folder ./edited_images --edit_json ./edits.json --origin_img_root ./original_images --num_processes 4
```
## Example Input/Output

### Input:
A JSON file containing image edit instructions (`edit_json`):

```json
{
    "1": {"id": "000029784.jpg", "prompt": "Remove the lemon slice inside the glass, and turn the slice on the rim into an orange slice."},
    "2": {"id": "000065904.jpg", "prompt": "Remove the person in the front passenger seat."},
    "3": {"id": "000066341.jpg", "prompt": "Remove the small tomato in the center and keep the others."}
}
```

### Output:
A JSON file (`result.json`) with GPT evaluation for each image:

```json
{
    "31": "Brief reasoning: Instructions partially followed. Unnatural edit of people wearing shorts. Heavy alterations led to image distortion. Score: 2.",
    "30": "Brief reasoning: Ferrari logos replaced, but logo near text unchanged; edits slightly unnatural. Score: 3 (Acceptable)",
    "19": "Brief reasoning: The middle car's color changed, but the right car's color remains unchanged, deviating from instructions. Score: 2 (Fair)."
}
```
