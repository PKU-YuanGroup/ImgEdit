import json
import random
import os
import re
from tqdm import tqdm
import argparse
from termcolor import cprint
from torch.utils.data import Dataset, DataLoader
from openai import AzureOpenAI 
from azure.identity import AzureCliCredential, ChainedTokenCredential, DefaultAzureCredential, get_bearer_token_provider
from tenacity import retry, stop_after_attempt, wait_exponential
import pdb
import logging

edit_type = [
    "add", 
    # "add_wo_ambiguity",
    "remove",
    # "remove_wo_ambiguity",
    "replace",
    # "replace_wo_ambiguity",
    "adjust",
    # "adjust":["color_alter", "material_alter", "texture_alter", "appearance_alter", "with reference", "without reference"],
    "background",
    "textual",
]

add_prompt = (
"You are now acting as a data annotator. Based on the given object, find its description in the image caption, including location and attributes, etc. Provide a complete summary in no more than 20 words and do not make anything up."
" For instance: "
"caption: 'The image depicts a couple dressed in wedding attire standing on a rocky cliff overlooking the ocean. The bride is wearing a long, white, lace gown with an open back, while the groom is dressed in a white suit with a bow tie.' object: 'bride' "
"description: 'a bride in a long, white, lace gown standing on a rocky cliff' "
"{{{}}}"
" description: "
)

remove_prompt = (
"You are now acting as a data annotator. Based on the given object and the image caption, generate a prompt to remove that object. You can provide either a descriptive or a concise deletion command. Prompt should be no more than 15 words."
" For instance: "
"caption: 'The image depicts a couple dressed in wedding attire standing on a rocky cliff overlooking the ocean. The bride is wearing a long, white, lace gown with an open back, while the groom is dressed in a white suit with a bow tie.' object to remove: 'bride' "
"prompt: 'remove the bride' or prompt: 'remove the bride dressed in wedding attire' "
"{{{}}}"
" prompt: "
)


replace_prompt = (
"You are now acting as a data annotator and need to create image editing prompts based on the given object and background. Imagine a scene and generate instruction to replace the specified object with another that blends naturally into the background and has a similar size to avoid awkwardness. The editing prompts must be concise, containing at most one meaningful adjective (such as color or material) to describe the replacement. Avoid using vague or ambiguous terms."
" output should adhere to the following format, 'result' should be a noun with or without adjective: "
"object: 'poached egg' , background: 'black plate, wooden base, wooden table' "
"prompt: 'replace poached egg with a fried potato', result: 'fried potato'"
"object: 'old man' , background: 'garden, tree' "
"prompt: 'change the old man to a child', result: 'child' "
"Now is the input: "
"{{{}}}"
)

adjust_prompt = (
"You are now acting as a data annotator and need to create image editing prompts based on the given object and background. Imagine a scene and generate prompts to adjust the attributes of object, such as color, material, texture, or appearance. For people, include skin tone, clothing, etc. Do not change the object type. The editing prompts must be concise, containing at most one meaningful adjective (such as color or material) to describe the replacement and blends naturally into the background. Avoid using vague or ambiguous terms."
" output should adhere to the following format, 'result' should be a noun with or without adjective: "
"object: 'apple' , background: 'black plate, wooden base, wooden table' "
"prompt: 'Turn the apple green', result: 'green apple'"
" object: 'bare trees' , background: 'small village, sky' "
"prompt: 'Make the bare tree leafy', result: 'leafy tree' "
"Now is the input: "
"{{{}}}"
)

background_prompt = (
"You are now acting as a data annotator and need to create image editing prompts based on the given background and object. Imagine a scene and generate prompts to change the background. Do not describe objects in generated prompt. The editing prompts must be concise, containing at most one meaningful adjective (such as color or material) to describe the replacement. Avoid using vague or ambiguous terms."
" output should adhere to the following format, 'result' should be a noun with or without adjective: "
"background: 'ocean' , object: 'wedding attire, bride, couple"
"prompt: 'Change the ocean into earth', result: 'earth'"
" background: 'sky' , object: 'plants, palm trees, sunlight' "
"prompt: 'Darken the sky', result: 'dark sky' "
"Now is the input: "
"{{{}}}"
)


api_version="2024-02-15-preview"
stupid_model_name='gpt-4o-mini_2024-07-18'
clever_model_name='gpt-4o_2024-11-20'


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
    )
)

token_provider = get_bearer_token_provider(azure_credential,
    "https://cognitiveservices.azure.com/.default")

aoiclient = AzureOpenAI( 
    azure_endpoint="https://gcraoai9sw1.openai.azure.com/", 
    azure_ad_token_provider=token_provider, 
    api_version=api_version, 
    max_retries=5, 
)

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


class ImageDataset(Dataset):
    def __init__(self, json_files):
        self.json_files = json_files

    def __len__(self):
        return len(self.json_files)

    def __getitem__(self, idx):
        json_file = self.json_files[idx]
        with open(json_file, "r") as f:
            data = json.load(f)
        data["filename"] = json_file
        return data

def custom_collate_fn(batch):
    return batch

def parse_args():
    parser = argparse.ArgumentParser(description="Instruction Pipeline")
    parser.add_argument(
        "--json_folder",
        required=True,
        type=str,
        help="Path to the folder containing JSON files.",
    )
    parser.add_argument(
        "--output_path", 
        required=True,
        type=str, 
        help="Path to the output JSON files.",
    )
    parser.add_argument("--instruction-type", default='mix',
                        choices=['mix'] + edit_type, help="specify the experiment id.")
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--idx", type=int, default=-101, help="specify the experiment id.")
    args = parser.parse_args()
    return args

replace_adjust_pattern = r"^prompt:\s*'(.*?)',\s*result:\s*'(.*?)'$"
add_remove_pattern = pattern = r"'(.*?)'|(.+)"

def valid_candidate(obbox, area, score, a_threshold, s_threshold):
    obj_area = (obbox[2] - obbox[0]) * (obbox[3] - obbox[1])
    if obj_area >= a_threshold * area and score >= s_threshold:
        return True
    return False

def make_replace_adjust_description(object, background, count=5):  
    if len(background) < count:  
        selected_strings = background  
    else:  
        selected_strings = random.sample(background, count)  
    desp = "object: '" + object + "',  background: '" + ', '.join(selected_strings) +"'"   
    return desp

def make_add_description(caption, object):  
    desp = "caption: " + caption + "  object: '" + object + "'"   
    return desp

def make_remove_description(caption, object):  
    desp = "caption: " + caption + "  object to remove: '" + object + "'"   
    return desp


if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.output_path, exist_ok=True)

    json_files = [
        os.path.join(args.json_folder, f)
        for f in os.listdir(args.json_folder)
        if f.endswith(".json")
        and not os.path.exists(os.path.join(args.output_path, os.path.basename(f).replace("_step4.json", "_step5.json")))
    ]
    dataset = ImageDataset(json_files)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, prefetch_factor=4, pin_memory=True, collate_fn=custom_collate_fn)
    '''
    Special judgment conditions
    edit_type = {
    "add", 
    # "add_wo_ambiguity",
    "remove",
    # "remove_wo_ambiguity",
    "replace",
    # "replace_wo_ambiguity",
    "edit",
    # "edit":["color_alter", "material_alter", "texture_alter", "appearance_alter", "with reference", "without reference"],
    "background",
    "textual",
}

    '''
    instruction_type = args.instruction_type
    for batch in tqdm(dataloader, desc="Processing Files", total=len(dataloader)):
        filename = batch[0]["filename"]
        resolution = (batch[0]['resolution']['height'], batch[0]['resolution']['width'])
        img_area = resolution[0] * resolution[1]
        if instruction_type == "mix":
            instruction_type = random.choice(edit_type)

        if instruction_type == "background":
            bg_list = batch[0]['segmentation']['background']
            candidate = []
            for i in range(len(bg_list)):
                bbox = obj_list[i]['bbox']
                score = obj_list[i]['score']
                if valid_candidate(bbox, img_area, score, a_threshold=0.5, s_threshold=0.7): 
                    candidate.append(i)
            if len(candidate) == 0:
                logging.warning(f"No candidate object for editing")
                continue
            # if object less than xxx, reverse the mask
            edit_bg = batch[0]['segmentation']['background'][random.choice(candidate)]
            desp = make_replace_adjust_description(edit_bg['class_name'], batch[0]['tags']['object'], count=5)
            input_prompt = background_prompt.replace("{{{}}}", desp).replace("\n", "").replace("\t", "").strip()
            model_name = clever_model_name
            
        else:
            obj_list = batch[0]['segmentation']['object']
            candidate = []
            for i in range(len(obj_list)):
                bbox = obj_list[i]['bbox']
                
                score = obj_list[i]['score']
                if valid_candidate(bbox, img_area, score, a_threshold=0.1, s_threshold=0.8): 
                    candidate.append(i)

            if len(candidate) == 0:
                logging.warning(f"No candidate object for editing")
                continue

            edit_obj = batch[0]['segmentation']['object'][random.choice(candidate)]
            
            if instruction_type == "add":
                desp = make_add_description(batch[0]['cap'][0], edit_obj['class_name'])
                input_prompt = add_prompt.replace("{{{}}}", desp).replace("\n", "").replace("\t", "").strip()
                model_name = clever_model_name

            elif instruction_type == "remove":
                desp = make_remove_description(batch[0]['cap'][0], edit_obj['class_name'])
                input_prompt = remove_prompt.replace("{{{}}}", desp).replace("\n", "").replace("\t", "").strip()
                model_name = stupid_model_name

            elif instruction_type == "replace":
                desp = make_replace_adjust_description(edit_obj['class_name'], batch[0]['tags']['background'], count=5)
                input_prompt = replace_prompt.replace("{{{}}}", desp).replace("\n", "").replace("\t", "").strip()
                model_name = clever_model_name

            elif instruction_type == "adjust":
                desp = make_replace_adjust_description(edit_obj['class_name'], batch[0]['tags']['background'], count=5)
                input_prompt = adjust_prompt.replace("{{{}}}", desp).replace("\n", "").replace("\t", "").strip()
                model_name = clever_model_name
                
        for attempt in range(3):
            
            response = call_gpt(input_prompt, model_name=model_name)
            if instruction_type in ["adjust", "replace"]:
                pattern = replace_adjust_pattern
            elif instruction_type in ["add", "remove"]:
                pattern = add_remove_pattern
            
            match = re.match(pattern, response)
            if match:
                pdb.set_trace()
                edit_prompt = match.group(1).strip().rstrip(".")
                edit_result = match.group(2)
                break
            else:
                logging.warning(f"Attempt {attempt+1} failed")
        if not match:
            logging.warning(f"Prompt generated failed")
            continue
        # only need path edit_obj edit_type edit_prompt result 
        if instruction_type == "add":
            data_info = {"edit_obj": None, "edit_type": instruction_type, "edit_prompt": edit_prompt, "edit_result": edit_obj}
        elif instruction_type == "remove":
            data_info = {"edit_obj": edit_obj, "edit_type": instruction_type, "edit_prompt": edit_prompt, "edit_result": None}
        elif instruction_type in ["adjust", "replace", "background"]:
            data_info = {"edit_obj": edit_obj, "edit_type": instruction_type, "edit_prompt": edit_prompt, "edit_result": edit_result.strip().rstrip(".")}
            
        output_json_path = os.path.join(args.output_path, os.path.basename(filename).replace("_step4.json", f"{instruction_type}.json"))
        with open(output_json_path, "w") as f:
            json.dump(data_info, f, indent=4)
                


        # elif instruction_type == "textual":
        #     pass
        # else:
        #     raise NotImplementedError
            

         
           