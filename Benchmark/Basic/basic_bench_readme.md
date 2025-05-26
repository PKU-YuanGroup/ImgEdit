
# Preprocessing
Before evaluating the model, you first need to use the provided JSON file (which contains metadata information) along with the original image files to generate the corresponding edited images by editing model. These edited images should be saved in a folder, with each image's filename prefix corresponding to the key value from the dictionary stored in the JSON file.

## Example Input/Output

### Input
A JSON file containing image edit instructions (`basic_edit`):

```json
{
    "1082": {
        "id": "animal/000342021.jpg",
        "prompt": "Change the tortoise's shell texture to a smooth surface.",
        "edit_type": "adjust"
    },
    "1068": {
        "id": "animal/000047206.jpg",
        "prompt": "Change the animal's fur color to a darker shade.",
        "edit_type": "adjust"
    },
    "673": {
        "id": "style/000278574.jpg",
        "prompt": "Transfer the image into a traditional ukiyo-e woodblock-print style.",
        "edit_type": "style"
    }
}
```


A folder containing original images (`origin_img_root`):

```folder
├── original_images                    
│   ├── animal     
|       |── 000342021.jpg                 
|       |── 000047206.jpg                 
│   ├── style                             
|       |── 000278574.jpg                  
```

### Output:
A folder containing edited images, with filenames prefixed by the key value from the JSON file.

```folder
├── edited_images                    
│   ├── 1082.png                 
│   ├── 1068.png            
│   └── 673.png             
``` 

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
   - A JSON file mapping keys to metadata and prompts for each image edit (`--basic_edit`).
   - A root directory where the original images are stored (`--origin_img_root`).
   - A JSON file containing prompts for each image (`--prompts_json`).



## Usage

To run the script, use the following command:

```bash
python basic_bench.py --result_img_folder <path_to_edited_images> --basic_edit <path_to_basic_edit> --origin_img_root <path_to_original_images> --num_processes <number_of_threads> --prompts_json <path_to_prompts_json>
```

### Arguments:
- `--result_img_folder`: The directory containing the edited images.
- `--prompts_json`: Path to the JSON file containing prompts for each image.
- `--basic_edit`: Path to the JSON file containing metadata and edit instructions.
- `--origin_img_root`: The root directory of the original images.
- `--num_processes`: The number of threads to use for processing. Default is 32.

### Example:

```bash
python basic_bench.py --result_img_folder ./edited_images --basic_edit ./edits.json --origin_img_root ./original_images --num_processes 4 --prompts_json ./prompts.json
```
## Example Input/Output

### Input:
A JSON file containing image edit instructions (`basic_edit`):

```json
{
    "1082": {
        "id": "animal/000342021.jpg",
        "prompt": "Change the tortoise's shell texture to a smooth surface.",
        "edit_type": "adjust"
    },
    "1068": {
        "id": "animal/000047206.jpg",
        "prompt": "Change the animal's fur color to a darker shade.",
        "edit_type": "adjust"
    },
    "673": {
        "id": "style/000278574.jpg",
        "prompt": "Transfer the image into a traditional ukiyo-e woodblock-print style.",
        "edit_type": "style"
    }
}
```

A JSON file containing prompts for each image (`prompts_json`):
```json
{
  "adjust": "\nYou are a data rater specializing in grading attribute alteration edits. You will be given two images ....",
  "style": "\nYou are a data rater specializing in grading style transfer edits. You will be given an input image, a reference style..."
}
```

A folder containing original images (`origin_img_root`):

```folder
├── original_images                    
│   ├── animal     
|       |── 000342021.jpg                 
|       |── 000047206.jpg                 
│   ├── style                             
|       |── 000278574.jpg                  
```


A folder containing edited images, with filenames prefixed by the key value from the JSON file.

```folder
├── edited_images                    
│   ├── 1082.png                 
│   ├── 1068.png            
│   └── 673.png             
``` 

### Output:
A JSON file (`result.json`) with GPT evaluation for each image:

```json
{
    "1082": "Brief reasoning: No texture change applied; shell remains textured; tortoise geometry unchanged.\nPrompt Compliance: 1\nVisual Seamlessness: 1\nPhysical & Detail Fidelity: 1",
    "1068": "Brief reasoning: No fur change; wrong subject; mainly butterfly retained with minor color alteration.\nPrompt Compliance: 1\nVisual Seamlessness: 4\nPhysical & Detail Fidelity: 4",
    "673": "Brief reasoning: Style transfer absent, with excellent content preservation and rendering quality maintaining original image's fidelity.\nStyle Fidelity: 1\nContent Preservation: 5\nRendering Quality: 5"
}
```


# Calculating the score

## Step1  Calculate the average score for all edited images.
The output of gpt is three aspects: Instruction Adherence, Image-editing Quality, and Detail Preservation scores. First, get the average score of these three scores for each editing result, and save the result in the form of a dictionary in a JSON file.

### Example

```bash
python step1_get_avgscore.py --result_json ./result.json --average_score_json ./average_score.json
```

### Input
A JSON file (`result_json`) with GPT evaluation for each image:

```json
{
    "1082": "Brief reasoning: No texture change applied; shell remains textured; tortoise geometry unchanged.\nPrompt Compliance: 1\nVisual Seamlessness: 1\nPhysical & Detail Fidelity: 1",
    "1068": "Brief reasoning: No fur change; wrong subject; mainly butterfly retained with minor color alteration.\nPrompt Compliance: 1\nVisual Seamlessness: 4\nPhysical & Detail Fidelity: 4",
    "673": "Brief reasoning: Style transfer absent, with excellent content preservation and rendering quality maintaining original image's fidelity.\nStyle Fidelity: 1\nContent Preservation: 5\nRendering Quality: 5"
}
```

### Output
A JSON file (`average_score_json`) with average scores for each image:

```json
{
    "1082": 1,
    "1068": 3.3333333333333335,
    "673": 3.6666666666666665
}
```

## Step2  Calculate average score by category
All editing tasks are divided into 9 different categories of editing tasks, such as Addition and Removal. In order to evaluate the model's ability to handle different types of editing tasks, it is necessary to take the average of the results of different categories. Based on the json file that stores the average value obtained in Step 1, the average value is calculated by category.


### Example
```bash
python step2_typescore.py --average_score_json ./average_score.json --typescore_json ./typescore.json --basic_edit ./basic_edit.json
```


### Input
A JSON file (`average_score_json`) with average scores for each image:

```json
{
    "1082": 1,
    "1068": 3.3333333333333335,
    "673": 3.6666666666666665
}
```

A JSON file containing edit type (`basic_edit`):

```json
{
    "1082": {
        "id": "animal/000342021.jpg",
        "prompt": "Change the tortoise's shell texture to a smooth surface.",
        "edit_type": "adjust"
    },
    "1068": {
        "id": "animal/000047206.jpg",
        "prompt": "Change the animal's fur color to a darker shade.",
        "edit_type": "adjust"
    },
    "673": {
        "id": "style/000278574.jpg",
        "prompt": "Transfer the image into a traditional ukiyo-e woodblock-print style.",
        "edit_type": "style"
    }
}
```


### Output
A JSON file (`typescore_json`) with average scores for each category:

```json
{
    "adjust": 2,
    "style": 3.6666666666666665,
}
```
