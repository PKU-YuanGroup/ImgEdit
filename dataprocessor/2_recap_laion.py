import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from azure.identity import AzureCliCredential, ChainedTokenCredential, DefaultAzureCredential, get_bearer_token_provider
from openai import AzureOpenAI 
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm
import spacy 
from nltk.stem import WordNetLemmatizer
from nltk.corpus import words as nltk_words


nlp = spacy.load("en_core_web_sm")  
english_vocab = set(w.lower() for w in nltk_words.words())
lemmatizer = WordNetLemmatizer()
api_version="2024-02-15-preview"
gpt_model_name='gpt-4o-mini_2024-07-18'


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

prompt_format = (
    "Given an image caption, please retrieve the entity words that indicate background and visually separable objects."
    "[Definition of background] The background spaces that appear in most of the image area."
    "[Definition of object] Entities that are visually separable, tangible, and physically present in part of the image, including human and animals."
    "Attention! All entity words need to strictly follow the rules below:"
    "1) The entity word is a singular or plural noun without any quantifier or descriptive phrase."
    "2) The entity word must be an exact subset of the caption, including its characters, words, and symbols. (e.g, 'red top' better than 'top', 'martial arts uniforms' better than 'uniforms')"
    "3) Exclude any part of the body (e.g., 'hands', 'legs', 'feet', 'head')."
    "4) Exclude abstract or non-physical concepts (e.g., 'facial expressions', 'gestures', 'stance', 'details')."
    "5) Exclude actions or descriptions (e.g., 'adjusting', 'imitating')."
    "6) Do not modify or interpret any part of the caption."
    "Here is an example, follow this format to output the results:"
    "Caption: A woman in a mask and coat, with long brown hair, shows a small green-capped bottle to the camera."
    "Output: {'background': [''], 'object': ['woman', 'mask', 'coat', 'long brown hair', 'green-capped bottle']}"
    "Here is the input:"
    "Caption: {{{}}}"
    "Output: "
)

def is_english_word(word):
    return word.lower() in english_vocab

# Global lock for thread-safe file operations
file_lock = Lock()

lemma_word = lemmatizer.lemmatize(i).lower()

def filter_keywords(items, keywords):
    return [item for item in items if item.lower() not in keywords]


def extract_data_from_response(response):
    # Convert all double quotes to single quotes at the beginning
    response = response.replace('"', "'")

    # Regex to match the structure, supporting single quotes for keys and values
    background_pattern = r"'background'\s*:\s*\[(.*?)\]"
    # subject_pattern = r"'subject'\s*:\s*\[(.*?)\]"
    object_pattern = r"'object'\s*:\s*\[(.*?)\]"

    # Extracting background, subject, and object lists using regex
    background_match = re.search(background_pattern, response)
    # subject_match = re.search(subject_pattern, response)
    object_match = re.search(object_pattern, response)

    # Initialize the result dictionary
    result = {"background": [], "object": []}
    if background_match:
        # Split and process background items
        background_items = [item.strip("' ") for item in background_match.group(1).split("', '")]
        filtered_background_items = []
        tmp_set = set()
        for i in background_items:
            doc = nlp(i)
            is_valid_phrase = False
            phrase = []
            for token in doc:
                
                if token.pos_ == 'NOUN' or 'PROPN' and token.ent_type_ != 'PERSON' and is_english_word(token.text):
                    lemma = token.lemma_.lower()
                    phrase.append(lemma)
                    if lemma not in tmp_set and is_valid_phrase is False:
                        is_valid_phrase = True
                        tmp_set.add(lemma)
                else:
                    phrase.append(token)
            if is_valid_phrase:
                filtered_background_items.append(" ".join(phrase).strip())
        result["background"] = filtered_background_items
    
    if object_match:
        # Split and process object items
        object_items = [item.strip("' ") for item in object_match.group(1).split("', '")]
        filtered_object_items = []
        tmp_set = set()
        for i in object_items:
            doc = nlp(i)
            is_valid_phrase = False
            phrase = []
            for token in doc:
                
                if token.pos_ == 'NOUN' or 'PROPN' and token.ent_type_ != 'PERSON' and is_english_word(token.text):
                    lemma = token.lemma_.lower()
                    phrase.append(lemma)
                    if lemma not in tmp_set and is_valid_phrase is False:
                        is_valid_phrase = True
                        tmp_set.add(lemma)
                else:
                    phrase.append(token)
            if is_valid_phrase:
                filtered_object_items.append(" ".join(phrase).strip())
        result["object"] = filtered_object_items # filter_keywords(object_items, keywords_to_remove)

    # Check if all parts were successfully matched
    if background_match and object_match:
        flag = True
    else:
        flag = False

    return result, flag


@retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(100))
def call_gpt(
    prompt, model_name=None, api_key=None, base_url=None
):
    model_name=gpt_model_name
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


def process_file(json_data, model_name, api_key, base_url, output_folder):
    key = json_data[0].replace("/", "").replace(".jpg", "")
    json_path = f"{key}.json"
    dump_data = {}
    image_metadata = json_data[1]
    input_prompt = prompt_format.replace("{{{}}}", image_metadata["cap"][0].strip()).replace("\n", "").replace("\t", "").strip()

    response = call_gpt(input_prompt, model_name=model_name, api_key=api_key, base_url=base_url)

    try:
        response_data = json.loads(response)
    except json.JSONDecodeError:
        response_data, flag = extract_data_from_response(response)
        if not flag:
            print("The response fis not in JSON format, skipping.")
            return

    image_metadata["tags"] = response_data
    dump_data[key] = image_metadata
    with open(os.path.join(output_folder, os.path.basename(json_path)), "w") as f:
        json.dump(dump_data, f, indent=4)


def parse_args():
    parser = argparse.ArgumentParser(description="Process some input and output folders with model settings.")
    parser.add_argument(
        "--input_json", type=str, required=True, help="Path to the caption json."
    )
    parser.add_argument(
        "--output_json_folder", type=str, required=True, help="Path to the output json folder."
    )
    parser.add_argument("--num_worker", type=int, default=32, help="Number of threads for parallel processing")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Config
    num_worker = args.num_worker
    input_json = args.input_json
    output_folder = args.output_json_folder

    # model_name = args.model_name
    # api_key = args.api_key
    # base_url = args.base_url

    with open(input_json, 'r', encoding="utf-8") as f:
        data = json.load(f)

    model_name = None
    api_key = None
    base_url = None

    os.makedirs(output_folder, exist_ok=True)

    fflag = True
    for _ in range(3):
        json_datas = []
        for item in data:
            key = item['path'].replace("/", "").replace(".jpg", "")
            json_path = os.path.join(output_folder, f"{key}.json")
            if not os.path.exists(json_path):
                json_datas.append((key, item))

        if not json_datas:
            fflag = False
            break

        with ThreadPoolExecutor(max_workers=num_worker) as executor:
            futures = []
            with tqdm(total=len(json_datas), desc="Processing JSON files") as pbar:
                for json_data in json_datas:
                    future = executor.submit(process_file, json_data, model_name, api_key, base_url, output_folder)
                    futures.append(future)

                for future in as_completed(futures):
                    future.result()
                    pbar.update(1)

    if fflag:
        print("Processing half completed!")
    else:
        print("Processing full completed!")