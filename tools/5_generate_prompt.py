import json
import random
import os
import re
from tqdm import tqdm
import argparse
# from torch.utils.data import Dataset, DataLoader
from openai import AzureOpenAI 
from azure.identity import AzureCliCredential, ChainedTokenCredential, DefaultAzureCredential, get_bearer_token_provider
from tenacity import retry, stop_after_attempt, wait_exponential
import logging
from termcolor import colored
from concurrent.futures import ProcessPoolExecutor, as_completed

import math

edit_type = [
    # "add", 
    # "remove",
    # "replace",
    #  "adjust",
    # "background",
    "version_refer",
    # "extract",
    # "compose",
#    'multi-premise',
    # 'omit-refer',
    # "", prop, add/replace+adjust+adjust/remove,
    # "", prompt ahead , add, ajust, replace,
    # "", version backtrack , 

]

version_refer_prompt = (
"Act as a data annotator who creates multi-turn image-editing prompts from a given scene, three instructions, and one object. Instruction keywords: add - insert the specified object as if it is not yet present, replace - swap the specified object with another one, adjust - modify the object's attributes (color, material, texture, appearance). Prompt should blend naturally into the scene. Workflow: You need to generate three round prompts based on the given instructions and the specific object, The first round is always 'add',  discribe more detail and provide an approximate description of the object's position and size according to its bounding box coordinates and image resolution. Do not use additional sentences to describe the position!!!. The second prompt is 'replace'. In the third round, your prompt should reject the edits from the previous round and, based on the first round, generate a new “adjust” prompt. For example: "
"sence: 'An open laptop on a glass desk beside a notebook and a pen.', resolution: 2560x1600, object: 'laptop', bbox: [500,400,2000,1200], instructions: 'add', 'replace', 'adjust'."
" Output: round1: add a laptop in the central area, round2: replace the laptop with a book, result2: a book, round3: i don't like your edit, adjust the laptop in first round into black color, result3: a black laptop. "
"sence: 'Ceramic mug placed on wooden plate with decorative items against light wall.', resolution: 1440x1920, object: 'ceramic mug', bbox: [931,339,1500,1220], instructions: 'add', 'replace', 'adjust'."
" Output: round1: add a ceramic mug in the central area of the image, round2: replace it into a glass bottle, result2: a glass bottle, round3: withdraw the previous round of modifications, adjust the ceramic mug in round1 into glass material, result3: a glass ceramic mug. "
"{{{}}}"
" Output: "
)

omit_refer_prompt = (
"Act as a data annotator who creates multi-turn image-editing prompts from a given scene, three instructions, and one object. Instruction keywords: remove - delete the object, add - insert the specified object as if it is not yet present, replace - swap the specified object with another one, adjust - modify the object's attributes (color, material, texture, appearance). Prompt should blend naturally into the scene. Workflow: You need to generate three round prompts based on the given instructions and the specific object, The first round is always 'add',  discribe more detail and provide an approximate description of the object's position and size according to its bounding box coordinates and image resolution. Do not use additional sentences to describe the position!!!. In following prompt, you should use pronoun to describe the objects and make prompts concise! "
"sence: 'Ceramic mug placed on wooden plate with decorative items against light wall.', resolution: 1440x1920, object: 'ceramic mug', bbox: [931,339,1500,1220], instructions: 'add', 'ajust', 'replace'."
" Output: round1: add a ceramic mug in the central area of the image, round2: adjust it into glass material, result2: a glass mug, round3: replace it with a glass bottle, result3: a glass bottle. "
"sence: 'An open laptop on a glass desk beside a notebook and a pen.', resolution: 2560x1600, object: 'laptop', bbox: [500,400,2000,1200], instructions: 'add', 'adjust', 'remove'."
" Output: round1: add a laptop in the central area, round2: adjust its color into black, result2: a black laptop, round3: remove it, result3: None "
"{{{}}}"
" Output: "
)

multi_premise_prompt = (
"Act as a data annotator who creates multi-turn image-editing prompts from a given scene, three instructions, and some objects. Instruction keywords: replace - swap the specified object with another one, adjust - modify the object's attributes (color, material, texture, appearance). Prompt should blend naturally into the scene. Workflow: First, output a single overarching premise about object attributes that should follow in all prompts. Next, produce two consecutive instructions that follow the instructions. These instructions must not contain the overarching premise!!! The edit results you provide must contain the premise and only describe the object you edit with concise words. "
"sence: 'Ceramic mug placed on wooden plate with decorative items against light wall.', object1: 'ceramic mug', instruction1: 'replace', object2: 'wooden plate', instruction2: 'replace'"
" Output: premise: All subsequent edits must use glass as the material, round1: replace ceramic mug with a bottle, result1: a glass bottle, round2: replace the wooden plate with a tray, result2: a glass tray. "
"sence: 'An open laptop on a glass desk beside a notebook and a pen.', object1: 'laptop', instruction1: 'adjust', object2: 'notebook', instruction2: 'replace'"
" Output: premise: All subsequent edits must use dark blue as the color, round1: adjust the laptop's color, result1: A dark-blue laptop, round2: adjust the color of notebook, result2: A dark-blue notebook. "
"{{{}}}"
" Output: "
)

add_prompt = (
"You are now acting as a data annotator. Based on the given object, locate its corresponding description in the image caption, including details about location and attributes. Generate an 'add' prompt in no more than 30 words. Additionally, provide an approximate description of the object's position and size according to its bounding box coordinates and image resolution. Do not use additional sentences to describe the position!!!"
" The output format must strictly follow the example, For instance: "
"caption: 'The image depicts a couple dressed in wedding attire standing on a rocky cliff overlooking the ocean. The bride is wearing a long, white, lace gown with an open back, while the groom is dressed in a white suit with a bow tie.' object: 'bride', resolution: 1440x1920, bbox: [931,339,1500,1220] "
"Prompt: 'Add a bride in a long, white lace gown standing on a rocky cliff in the upper-right-middle of the image, occupying about one third of the area.'   "
"{{{}}}"
" Prompt: "
)



replace_prompt = (
"You are now acting as a data annotator and will generate image editing prompts based on a given object and scene. Write a concise instruction to replace the specified object with another that blends naturally into the scene! using at most one meaningful adjective (such as color or material) to describe the replacement. In the prompt, include an approximate description of the object's position and size according to its bounding box and image resolution. Do not use additional sentences to describe the position!!!"
" The output format must strictly follow the example, 'result' should be a noun with or without adjective: "
"summary: 'Ceramic mug placed on surface with decorative items against light wall.' object: 'ceramic mug', resolution: 1672x2140, bbox: [197,208,1707,1417] "
"Output: prompt: replace ceramic mug positioned in the central area of the image with a glass bottle, result: a glass bottle.   "
"{{{}}}"
"Output: "
)

adjust_prompt = (
"You are now acting as a data annotator and need to create image editing prompts based on the given object and scene. Write a concise instruction to adjust the attributes of given object, such as color, material, texture, or appearance. For people, include skin tone, clothing, etc. Do not change the object type. In the prompt, include an approximate description of the object's position and size according to its bounding box and image resolution. Do not use additional sentences to describe the position!!!"
" The output format must strictly follow the example, 'result' should be a noun with or without adjective: "
"Summary: 'Ceramic mug placed on surface with decorative items against light wall.' object: 'ceramic mug', resolution: 1672x2140, bbox: [197,208,1707,1417] "
"Output:  prompt: 'Turn ceramic mug positioned in the central area into glass', result: 'glass mug'"
"Summary: 'An open laptop on a glass desk beside a notebook and a pen.' object: 'laptop', resolution: 2560x1600, bbox: [500,400,2000,1200] "
"Output:  prompt: 'Turn laptop positioned in the central area into black', result: 'black laptop'"
"{{{}}}"
"output:  "
)

background_prompt = (
"You are now acting as a data annotator and need to create image editing prompts based on the given background and scene. Write a concise instruction to change the background. Do not describe objects in generated prompt. The editing prompts must be concise, containing at most one meaningful adjective (such as color or material) to describe the replacement. Avoid using vague or ambiguous terms."
" The output format must strictly follow the example, 'result' should be a noun with or without adjective: "
"summary: A photographer is taking photos of a bride wearing wedding attire by the seaside. background: 'ocean', object in background: 'wedding attire, bride, photographer"
"output:  prompt: 'Change the ocean into earth', result: 'earth'"
"{{{}}}"
"output:  "
)

remove_prompt = (
"You are now acting as a data annotator. Based on the given object and the image caption, generate a prompt to remove that object. You can provide either a descriptive or a concise deletion command. Prompt should be no more than 30 words, strictly using only the information provided. Additionally, provide an approximate description of the object's position and size according to its bounding box coordinates and image resolution. Do not use additional sentences to describe the position!!!"
" The output format must strictly follow the example, For instance: "
"caption: 'The image depicts a couple dressed in wedding attire standing on a rocky cliff overlooking the ocean. The bride is wearing a long, white, lace gown with an open back, while the groom is dressed in a white suit with a bow tie.' object to remove: 'bride', resolution: 1440x1920, bbox: [931,339,1500,1220] "
"Prompt: 'Remove the bride dressed in wedding attire located in the upper right-middle of the image.'   "
"{{{}}}"
" Prompt: "
)

compose_prompt = (
"Act as a data annotator who creates image-editing prompts from a given scene, two instructions, and two objects. Instruction keywords: replace - swap the specified object with another one, remove - delete the object, add - insert the specified object as if it is not yet present, adjust - modify the object's attributes (color, material, texture, appearance). For add or remove, only describe the action; for adjust or replace, write a prompt that blends naturally into the scene. Embed an approximate description of each object's location and size using its bounding box [x_min, y_min, x_max, y_max] and the image resolution, without extra sentences about position. The output must follow the exact example format, and the 'result' field must be either a noun (optionally with adjectives) or “None” for add or remove. "
"sence: 'Ceramic mug placed on wooden plate with decorative items against light wall.', resolution: 1672x2140, object1: 'ceramic mug', bbox1: [197,208,1707,1417], instruction1: 'replace', object2: 'wooden plate', bbox2: [122, 1427, 1783, 1669], instruction2: 'remove'"
" Output: prompt: replace ceramic mug positioned in the central area of the image with a glass bottle and remove the wooden plate underneath, result1: a glass bottle, result2: None. "
"sence: 'An open laptop on a glass desk beside a notebook and a pen.', resolution: 2560x1600, object1: 'laptop', bbox1: [500,400,2000,1200], instruction1: 'add', object2: 'notebook', bbox2: [2010, 480, 2510, 1120], instruction2: 'adjust'"
" Output: prompt: add a laptop in the central area of the image and change the color of notebook on the right into red, result1: None, result2: red notebook. "
"{{{}}}"
" Output: "
)
# summary: 'An open laptop on a glass desk beside a notebook and a pen.' object: 'laptop', resolution: 2560x1600, bbox: [500,400,2000,1200] Output: prompt: replace laptop occupying the central area of the image with a tablet, result: a tablet.
# summary: 'A single white pillow resting on a gray couch against a bright wall.' object: 'white pillow', resolution: 1280x720, bbox: [400,300,900,700] Output: prompt: replace white pillow placed on the center-right of the image with a knitted cushion, result: a knitted cushion.
    # desp = f"caption: {caption} resolution: {resolution_str}, object1: '{object1}',  object1 bbox: {bbox_str1}, instruction1: {instruct[0]}, object2: '{object2}',  object2 bbox: {bbox_str2}, instruction2: {instruct[1]}."  

shit = []

clever_model_name='gpt-4o_2024-11-20'

@retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(100))
def call_gpt(
    prompt, model_name=None, api_key=None, base_url=None
):
    chat_completion = aoiclient.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        max_tokens=1024,
    )
    return chat_completion.choices[0].message.content


# class ImageDataset(Dataset):
#     def __init__(self, json_files):
#         self.json_files = json_files

#     def __len__(self):
#         return len(self.json_files)

#     def __getitem__(self, idx):
#         json_file = self.json_files[idx]
#         with open(json_file, "r") as f:
#             data = json.load(f)
#         data["filename"] = json_file
#         return data

# def custom_collate_fn(batch):
#     return batch

def parse_args():
    parser = argparse.ArgumentParser(description="Instruction Pipeline")
    parser.add_argument(
        "--json_folder",
        required=True,
        type=str,
        help="Path to the folder containing JSON files.",
    )
    parser.add_argument(
        "--model",
        required=True,
        type=str,
        help="gpt model name.",
    )
    parser.add_argument(
        "--output_path", 
        required=True,
        type=str, 
        help="Path to the output JSON files.",
    )
    parser.add_argument("--instruction-type", default='mix',
                        choices=['mix'] + edit_type, help="specify the experiment id.")
    parser.add_argument("--num_workers", type=int, default=1)
    args = parser.parse_args()
    return args

# replace_adjust_pattern = r"^prompt:\s*'(.*?)',?\s*result:\s*'(.*?)'\s*$"
replace_adjust_pattern = r"^prompt:\s*(.*?),?\s*result:\s*(.*?)\s*$"  
compose_pattern = r"^prompt:\s*(.*?),?\s*result1:\s*(.*?),?\s*result2:\s*(.*?)\s*$"  
add_remove_pattern = r"'(.*?)'|(.+)"
multi_premise_pattern = r"^premise:\s*(.*?),?\s*round1:\s*(.*?),?\s*result1:\s*(.*?),?\s*round2:\s*(.*?),?\s*result2:\s*(.*?)\s*$" 
omit_refer_pattern = r"^round1:\s*(.*?),?\s*round2:\s*(.*?),?\s*result2:\s*(.*?),?\s*round3:\s*(.*?),?\s*result3:\s*(.*?)\s*$" 
def valid_obj_candidate(obbox, area, score, clip_score, a_threshold, s_threshold, c_threshold):
    if isinstance(score, list):
        if len(score) > 1 or len(score) == 0:
            print(colored("Error", "yellow"))
        score = score[0]
    obj_area = (obbox[2] - obbox[0]) * (obbox[3] - obbox[1])
    if obj_area <= 0.01 * area:
        return False
    if obj_area > 0.6 * area:
        return False
    if clip_score >= c_threshold:
        return True
    if obj_area >= a_threshold * area and score >= s_threshold:
        return True
    return False

def valid_bg_candidate(obbox, area, score, clip_score, a_threshold, s_threshold, c_threshold):
    if isinstance(score, list):
        if len(score) > 1 or len(score) == 0:
            print(colored("Error", "yellow"))
        score = score[0]
    obj_area = (obbox[2] - obbox[0]) * (obbox[3] - obbox[1])
    if obj_area <= 0.4 * area:
        return False
    if obj_area > 0.7 * area:
        return False
    if clip_score >= c_threshold:
        return True
    if obj_area >= a_threshold * area and score >= s_threshold:
        return True
    return False

def make_replace_adjust_description(summary, object, resolution, bbox):  
    resolution_str = f"{resolution['height']}x{resolution['width']}"  
    # 格式化 bbox 信息，四舍五入为整数  
    bbox_str = f"[{int(round(bbox[0]))},{int(round(bbox[1]))},{int(round(bbox[2]))},{int(round(bbox[3]))}]"  
    # 构建描述字符串  
    description = (  
        f"summary: {summary}, "  
        f"object: '{object}', "  
        f"resolution: {resolution_str}, object bbox: {bbox_str}."  
    )  
    return description 

def make_bg_description(summary, background, object, count=5):  
    selected_object = object if len(object) < count else random.sample(object, count)  
      
    # 构建描述字符串  
    description = (  
        f"summary: {summary}, "  
        f"background: '{background}'"  
        f"object in background: '{', '.join(selected_object)}'."  
    )  
    return description 

def make_omit_refer_description(caption, object, bbox, resolution, instruction):  
    # 格式化 resolution 信息  
    resolution_str = f"{resolution['height']}x{resolution['width']}"  
    # 格式化 bbox 信息，四舍五入为整数  
    bbox_str = f"[{int(round(bbox[0]))},{int(round(bbox[1]))},{int(round(bbox[2]))},{int(round(bbox[3]))}]"  
    # 构建描述字符串  
    desp = f"caption: {caption}, resolution: {resolution_str}, object: '{object}', object bbox: {bbox_str}, instructions: {instruction}"  
    return desp  

def make_version_refer_description(caption, object, bbox, resolution, instruction):  
    # 格式化 resolution 信息  
    resolution_str = f"{resolution['height']}x{resolution['width']}"  
    # 格式化 bbox 信息，四舍五入为整数  
    bbox_str = f"[{int(round(bbox[0]))},{int(round(bbox[1]))},{int(round(bbox[2]))},{int(round(bbox[3]))}]"  
    # 构建描述字符串  
    desp = f"caption: {caption}, resolution: {resolution_str}, object: '{object}', object bbox: {bbox_str}, instructions: {instruction}"  
    return desp 


def make_add_description(caption, object, bbox, resolution):  
    # 格式化 resolution 信息  
    resolution_str = f"{resolution['height']}x{resolution['width']}"  
    # 格式化 bbox 信息，四舍五入为整数  
    bbox_str = f"[{int(round(bbox[0]))},{int(round(bbox[1]))},{int(round(bbox[2]))},{int(round(bbox[3]))}]"  
    # 构建描述字符串  
    desp = f"caption: {caption} object: '{object}', resolution: {resolution_str}, object bbox: {bbox_str}."  
    return desp  

def make_remove_description(caption, object, bbox, resolution):  
    resolution_str = f"{resolution['height']}x{resolution['width']}"  
    if bbox is not None:
        bbox_str = f"[{int(round(bbox[0]))},{int(round(bbox[1]))},{int(round(bbox[2]))},{int(round(bbox[3]))}]"  
        desp = f"caption: {caption} object to remove: '{object}', resolution: {resolution_str}, object bbox: {bbox_str}"  
    else:
        desp = f"caption: {caption} object to remove: '{object}'."  
    return desp


def make_compose_description(caption, object1, object2, bbox1, bbox2, instruct, resolution):  
    # 格式化 resolution 信息  
    resolution_str = f"{resolution['height']}x{resolution['width']}"  
    # 格式化 bbox 信息，四舍五入为整数  
    bbox_str1 = f"[{int(round(bbox1[0]))},{int(round(bbox1[1]))},{int(round(bbox1[2]))},{int(round(bbox1[3]))}]"
    bbox_str2 = f"[{int(round(bbox2[0]))},{int(round(bbox2[1]))},{int(round(bbox2[2]))},{int(round(bbox2[3]))}]"

    # 构建描述字符串  
    desp = f"sence: {caption}, resolution: {resolution_str}, object1: '{object1}', bbox1: {bbox_str1}, instruction1: '{instruct[0]}', object2: '{object2}', bbox2: {bbox_str2}, instruction2: '{instruct[1]}'."  
    return desp 
 
def make_multi_premise_description(caption, object1, object2, instruct):  
    # 构建描述字符串  
    desp = f"sence: {caption}, object1: '{object1}', instruction1: '{instruct[0]}', object2: '{object2}', instruction2: '{instruct[1]}'."  
    return desp 
    
def bbox_far_enough(box1, box2, min_dist=100, xywh=False):  
    # 统一坐标  
    x1_1, y1_1, x2_1, y2_1 =  box1
    x1_2, y1_2, x2_2, y2_2 = box2
  
    # 1) 判是否重叠  
    overlap = not (x2_1 < x1_2 or x2_2 < x1_1 or  
                   y2_1 < y1_2 or y2_2 < y1_1)  
    if overlap:  
        return False          # 有重叠，直接返回不符合要求  
  
    # 2) 计算最近距离  
    dx = 0.0  
    if x2_1 < x1_2:          # box1 在左边  
        dx = x1_2 - x2_1  
    elif x2_2 < x1_1:        # box2 在左边  
        dx = x1_1 - x2_2  
  
    dy = 0.0  
    if y2_1 < y1_2:          # box1 在上面  
        dy = y1_2 - y2_1  
    elif y2_2 < y1_1:        # box2 在上面  
        dy = y1_1 - y2_2  
  
    dist = math.hypot(dx, dy)  # √(dx²+dy²)  
    return dist > min_dist  

def caption(batch):
    filename = batch["filename"]
    resolution = (batch['resolution']['height'], batch['resolution']['width'])
    img_area = resolution[0] * resolution[1]

    output_dir = os.path.join(args.output_path, os.path.basename(filename).replace("_step4_withcount.json", ""))  

    # if os.path.exists(output_json_path):
    #     return
    for instruction_type in edit_type:
        if instruction_type == "background":
            bg_list = batch['segmentation']['background']
            candidate = []
            for i in range(len(bg_list)):
                bbox = bg_list[i]['bbox']
                score = bg_list[i]['score']
                clip_score = bg_list[i]['clip_score']
                if valid_bg_candidate(bbox, img_area, score, clip_score, a_threshold=0.4, s_threshold=0.75, c_threshold=0.7): 
                    candidate.append(i)
            if len(candidate) == 0:
                logging.warning(f"No candidate object for editing")
                continue
            # if object less than xxx, reverse the mask
            edit_obj = batch['segmentation']['background'][random.choice(candidate)]
            desp = make_bg_description(batch['tags']["summary"], edit_obj['class_name'], list(batch["obj_count"].keys()), count=5)
            input_prompt = background_prompt.replace("{{{}}}", desp).replace("\n", "").replace("\t", "").strip()
            model_name = clever_model_name
        else:
            obj_list = batch['segmentation']['object']
            candidate = []
            for i in range(len(obj_list)):
                bbox = obj_list[i]['bbox']
                score = obj_list[i]['score']
                clip_score = obj_list[i]['clip_score']
                if valid_obj_candidate(bbox, img_area, score, clip_score, a_threshold=0.08, s_threshold=0.8, c_threshold=0.8): 
                    candidate.append(i)

            if len(candidate) == 0:
                logging.warning(f"No candidate object for editing")
                break
            if instruction_type == 'compose':
                # import pdb; pdb.set_trace()
                if len(candidate) <= 1:
                    logging.warning(f"No candidate object for editing")
                    break
                edit_obj = [batch['segmentation']['object'][i] for i in random.sample(candidate, 2)]  
                if edit_obj[1]['class_name'] == edit_obj[0]['class_name']:
                    break
                if not bbox_far_enough(edit_obj[0]['bbox'],edit_obj[1]['bbox']):
                    break
                # edit_obj = batch['segmentation']['object'][random.sample(candidate, 2)]
                instructions = random.sample(["add", "remove", "replace", "adjust"], 2)
                desp = make_compose_description(batch['tags']["summary"], edit_obj[0]['class_name'], edit_obj[1]['class_name'], edit_obj[0]['bbox'], edit_obj[1]['bbox'], instructions, batch['resolution'])
                input_prompt = compose_prompt.replace("{{{}}}", desp).replace("\t", "").strip()
                model_name = clever_model_name

            if instruction_type == 'multi-premise':
                if len(candidate) <= 1:
                    logging.warning(f"No candidate object for editing")
                    break
                edit_obj = [batch['segmentation']['object'][i] for i in random.sample(candidate, 2)]  
                # edit_obj = batch['segmentation']['object'][random.sample(candidate, 2)]
                instructions = random.choices(["replace", "adjust"], k=2)
                desp = make_multi_premise_description(batch['tags']["summary"], edit_obj[0]['class_name'], edit_obj[1]['class_name'], instructions)
                input_prompt = multi_premise_prompt.replace("{{{}}}", desp).replace("\t", "").strip()
                model_name = clever_model_name
            
            if instruction_type == 'omit-refer':
                edit_obj = batch['segmentation']['object'][random.choice(candidate)] 
                # edit_obj = batch['segmentation']['object'][random.sample(candidate, 2)]
                instructions = random.choice([  
                    ['add', 'adjust', 'remove'],  
                    ['add', 'adjust', 'replace']  
                ])
                desp = make_omit_refer_description(batch['cap'][0], edit_obj['class_name'], edit_obj['bbox'], batch['resolution'], instructions)
                input_prompt = omit_refer_prompt.replace("{{{}}}", desp).replace("\t", "").strip()
                model_name = clever_model_name
            if instruction_type == 'version_refer':
                edit_obj = batch['segmentation']['object'][random.choice(candidate)] 
                instructions = ['add', 'replace', 'adjust']
                desp = make_version_refer_description(batch['cap'][0], edit_obj['class_name'], edit_obj['bbox'], batch['resolution'], instructions)
                input_prompt = version_refer_prompt.replace("{{{}}}", desp).replace("\t", "").strip()
                model_name = clever_model_name

            if instruction_type == "add":
                edit_obj = batch['segmentation']['object'][random.choice(candidate)]
                desp = make_add_description(batch['cap'][0], edit_obj['class_name'], edit_obj['bbox'], batch['resolution'])
                input_prompt = add_prompt.replace("{{{}}}", desp).replace("\n", "").replace("\t", "").strip()
                model_name = clever_model_name

            elif instruction_type == "remove":
                edit_obj = batch['segmentation']['object'][random.choice(candidate)]
                desp = make_remove_description(batch['cap'][0], edit_obj['class_name'], edit_obj['bbox'], batch['resolution'])
                input_prompt = remove_prompt.replace("{{{}}}", desp).replace("\n", "").replace("\t", "").strip()
                model_name = clever_model_name

            elif instruction_type == "replace":
                edit_obj = batch['segmentation']['object'][random.choice(candidate)]
                desp = make_replace_adjust_description(batch['tags']["summary"], edit_obj['class_name'],batch['resolution'], edit_obj['bbox'])
                input_prompt = replace_prompt.replace("{{{}}}", desp).replace("\n", "").replace("\t", "").strip()
                model_name = clever_model_name

            elif instruction_type == "adjust":
                edit_obj = batch['segmentation']['object'][random.choice(candidate)]
                desp = make_replace_adjust_description(batch['tags']["summary"], edit_obj['class_name'], batch['resolution'], edit_obj['bbox'])
                input_prompt = adjust_prompt.replace("{{{}}}", desp).replace("\n", "").replace("\t", "").strip()
                model_name = clever_model_name
        match = None       
        for attempt in range(3):
            response = call_gpt(input_prompt, model_name=model_name).replace("\n", "").strip("` ").strip()
            if instruction_type in ["adjust", "replace", "background"]:
                pattern = replace_adjust_pattern
                match = re.match(pattern, response)
                try:
                    if match:
                        edit_prompt = match.group(1).strip().rstrip(".")
                        edit_result = match.group(2).strip().rstrip(".")
                        break
                    else:
                        logging.warning(f"Attempt {attempt+1} failed")
                except:
                    print(colored(response+"  match catch error", "red"))
                    break 
            elif instruction_type in ["omit-refer", "version_refer"]:
                # import pdb; pdb.set_trace()
                pattern = omit_refer_pattern
                match = re.match(pattern, response)
                try:
                    if match:
                        round1 = match.group(1).strip().rstrip(".")
                        round2 = match.group(2).strip().rstrip(".")
                        edit_result2 = match.group(3).strip().rstrip(".")
                        round3 = match.group(4).strip().rstrip(".")
                        edit_result3 = match.group(5).strip().rstrip(".")
                        break
                    else:
                        logging.warning(f"Attempt {attempt+1} failed")
                except:
                    print(colored(response+"  match catch error", "red"))
                    break 
            elif instruction_type == "multi-premise":
                pattern = multi_premise_pattern
                match = re.match(pattern, response)
                try:
                    if match:
                        premise = match.group(1).strip().rstrip(".")
                        round1 = match.group(2).strip().rstrip(".")
                        edit_result1 = match.group(3).strip().rstrip(".")
                        round2 = match.group(4).strip().rstrip(".")
                        edit_result2 = match.group(5).strip().rstrip(".")
                        break
                    else:
                        logging.warning(f"Attempt {attempt+1} failed")
                except:
                    print(colored(response+"  match catch error", "red"))
                    break 
            elif instruction_type == "compose":
                pattern = compose_pattern
                match = re.match(pattern, response)
                try:
                    if match:
                        edit_prompt = match.group(1).strip().rstrip(".")
                        edit_result1 = match.group(2).strip().rstrip(".")
                        edit_result2 = match.group(3).strip().rstrip(".")
                        break
                    else:
                        logging.warning(f"Attempt {attempt+1} failed")
                except:
                    print(colored(response+"  match catch error", "red"))
                    break 
            elif instruction_type in ["add", "remove"]:
                pattern = None
                break
        if not match and pattern is not None:
            print(colored(response+"  Prompt generated failed", "blue"))
            break
        # only need path edit_obj edit_type edit_prompt result 
        if instruction_type == "add":
            data_info = {"original_path": batch['path'], "resolution": batch['resolution'], "edit_obj": None, "edit_type": instruction_type, "edit_prompt": response, "edit_result": edit_obj}
        elif instruction_type == "remove":
            data_info = {"original_path": batch['path'], "resolution": batch['resolution'], "edit_obj": edit_obj, "edit_type": instruction_type, "edit_prompt": response, "edit_result": None}
        elif instruction_type in ["adjust", "replace", "background"]:
            data_info = {"original_path": batch['path'], "resolution": batch['resolution'], "edit_obj": edit_obj, "edit_type": instruction_type, "edit_prompt": edit_prompt, "edit_result": edit_result}
        elif instruction_type == "compose":
            data_info = {"original_path": batch['path'], "resolution": batch['resolution'], "edit_obj1": edit_obj[0], "edit_obj2": edit_obj[1], "edit_type": instructions, "edit_prompt": edit_prompt, "edit_result1": edit_result1, "edit_result2": edit_result2}
        elif instruction_type == "multi-premise":
            data_info = {"original_path": batch['path'], "resolution": batch['resolution'], "edit_obj1": edit_obj[0], "edit_obj2": edit_obj[1], "edit_type": instructions, "round1_prompt": round1, "edit_result1": edit_result1, "round2_prompt": round2, "edit_result2": edit_result2, "premise": premise}
        elif instruction_type == "omit-refer":
            data_info = {"original_path": batch['path'], "resolution": batch['resolution'], "edit_obj": edit_obj, "edit_type": instructions, "round1_prompt": round1, "round2_prompt": round2, "edit_result2": edit_result2, "round3_prompt": round3, "edit_result3": edit_result3}
        elif instruction_type == "version_refer":
            data_info = {"original_path": batch['path'], "resolution": batch['resolution'], "edit_obj": edit_obj, "edit_type": instructions, "round1_prompt": round1, "round2_prompt": round2, "edit_result2": edit_result2, "round3_prompt": round3, "edit_result3": edit_result3}
         
        os.makedirs(output_dir, exist_ok=True)  
        with open(os.path.join(output_dir, f"{instruction_type}.json"), "w") as f:
            json.dump(data_info, f, indent=4)

    return  

def process_file(json_file):
    try:
        with open(json_file, "r") as f:
            data = json.load(f)
        data["filename"] = json_file
        caption(data)
        return json_file  # 返回成功处理的文件路径
    except Exception as e:
        logging.error(f"Error processing file {json_file}: {e}", exc_info=True)
        return None
    
if __name__ == '__main__':
    args = parse_args()
    clever_model_name = args.model
    
    os.makedirs(args.output_path, exist_ok=True)

    json_files = [
        os.path.join(args.json_folder, f)
        for f in os.listdir(args.json_folder)
        if f.endswith(".json")
        and not os.path.exists(os.path.join(args.output_path, os.path.basename(f).replace("_step4_withcount.json", "")))
    ]
    results = []
    # for json_file in json_files:
    #     process_file(json_file)

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        future_to_file = {executor.submit(process_file, json_file): json_file for json_file in json_files}
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


                


        # elif instruction_type == "textual":
        #     pass
        # else:
        #     raise NotImplementedError
            

         
           


# summary: 'A sleek smartphone lying on a wooden desk next to a laptop.' object: 'smartphone', resolution: 1920x1080, bbox: [1200,600,1850,1000] Output: prompt: replace smartphone positioned in the bottom-right quadrant of the image with a camera, result: a camera.
# summary: 'A leather wallet resting on a marble countertop with loose change around.' object: 'leather wallet', resolution: 2048x1536, bbox: [500,400,1500,1000] Output: prompt: replace leather wallet positioned centrally at the lower section of the image with a cotton pouch, result: a cotton pouch.
# summary: 'A hardcover book on a side table beside a warm reading lamp in a cozy nook.' object: 'hardcover book', resolution: 1440x2560, bbox: [200,1800,1300,2450] Output: prompt: replace hardcover book positioned in the bottom-left area of the image with a paper notebook, result: a paper notebook.
# summary: 'A green potted plant sits on a windowsill overlooking a lush garden.' object: 'potted plant', resolution: 1080x1920, bbox: [300,200,780,950] Output: prompt: replace potted plant positioned in the upper-left quadrant of the image with a succulent, result: a succulent.
# summary: 'A stainless steel kitchen sink faucet with water droplets on its surface.' object: 'kitchen sink faucet', resolution: 2560x1440, bbox: [100,100,2460,840] Output: prompt: replace kitchen sink faucet spanning the lower half of the image with a black nozzle, result: a black nozzle.
# summary: 'A pair of red high-heeled shoes on a glossy tile floor near a doorway.' object: 'high-heeled shoes', resolution: 1920x1280, bbox: [500,700,1400,1200] Output: prompt: replace high-heeled shoes positioned at the bottom center of the image with a pair of sneakers, result: a pair of sneakers.
# summary: 'A wooden cutting board with sliced vegetables on a rustic kitchen table.' object: 'cutting board', resolution: 1200x1200, bbox: [100,200,900,900] Output: prompt: replace cutting board centered in the image with a metal tray, result: a metal tray.
# summary: 'A colorful children's ball lies in the grass of a sunlit backyard.' object: 'children's ball', resolution: 1600x900, bbox: [200,400,800,1000] Output: prompt: replace children's ball located in the lower-left quadrant of the image with a wooden toy block, result: a wooden toy block.
# summary: 'A single white pillow resting on a gray couch against a bright wall.' object: 'white pillow', resolution: 1280x720, bbox: [400,300,900,700] Output: prompt: replace white pillow placed on the center-right of the image with a knitted cushion, result: a knitted cushion.
# summary: 'An open laptop on a glass desk beside a notebook and a pen.' object: 'laptop', resolution: 2560x1600, bbox: [500,400,2000,1200] Output: prompt: replace laptop occupying the central area of the image with a tablet, result: a tablet.
