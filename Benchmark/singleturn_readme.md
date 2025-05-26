
# Image Editing Evaluation Script

## Requirements

- Python 3.7+
- `openai` library (to interact with GPT-4)
- `tqdm` library (for progress tracking)
- `tenacity` library (for retrying failed requests)
- `concurrent.futures` (for parallel processing)
- `base64` and `os` libraries (for file handling)

To install the required libraries, use:

```bash
pip install openai tqdm tenacity
```

## Setup

1. **OpenAI API Key**: Ensure you have an API key from OpenAI. Replace the placeholder in the code with your actual API key.
2. **Image Files**: You should have a folder containing the original images and another for the edited images. The images should match the format defined in the script.
3. **JSON File**: Prepare a JSON file that maps keys to image metadata. This file should contain information about the images to be evaluated, including their IDs and editing instructions.

### Example of JSON File Format

```json
{
    "1082": {
        "id": "animal/000342021.jpg",
        "prompt": "Change the tortoise's shell texture to a smooth surface.",
        "edit_type": "alter"
    },
    "1068": {
        "id": "animal/000047206.jpg",
        "prompt": "Change the animal's fur color to a darker shade.",
        "edit_type": "alter"
    }
}
```

In the above example, `id` refers to the original image's relative path, and `prompt` contains the editing instructions.

## Usage

The script can be run from the command line. Use the following command format:

```bash
python evaluate_image_edits.py --result_img_folder <path_to_edited_images> --edit_json <path_to_json_file> --origin_img_root <path_to_original_images> --num_processes <number_of_parallel_threads>
```

### Arguments:

- `--result_img_folder`: Folder containing the edited images.
- `--edit_json`: Path to the JSON file mapping keys to metadata for evaluation.
- `--origin_img_root`: Root path where the original images are stored.
- `--num_processes`: Number of parallel threads to process the images (default is 32).


### Example Output:

```json
{
    "449": "Brief reasoning: Entire object extracted flawlessly with precise edges and excellent visual quality.\nObject Identity: 5\nMask Precision: 5\nVisual Quality: 5",
    "48": "Brief reasoning: Person added correctly, slight mismatch in lighting, perspective; minor edge issues visible upon zooming.\nPrompt Compliance: 4  \nVisual Naturalness: 4  \nPhysical & Detail Coherence: 4  ",
    "541": "Brief reasoning: The armchair wasn't removed; only the upholstery pattern was altered.  \nPrompt Compliance: 1  \nVisual Naturalness: 1  \nPhysical & Detail Integrity: 1",
}
```