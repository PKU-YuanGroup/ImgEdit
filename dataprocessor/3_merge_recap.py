import os
import json
import argparse
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


def process_json_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f) 
    except json.JSONDecodeError:
        print(f"File {os.path.basename(file_path)} is not a json file")
    return data

def extract_metadata_from_json(input_folder, output_file):
    data_list = []  
    
    json_files = [os.path.join(input_folder, filename) for filename in os.listdir(input_folder) if filename.endswith('.json')]

    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_json_file, json_files), total=len(json_files), desc="Processing files"))
        
        for result in results:
            if isinstance(result, list):
                print("shit")
                data_list.extend(result)  
            elif isinstance(result, dict):
                data_list.append(result) 
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data_list, f, indent=4, ensure_ascii=False)

    print(f"data have been saved to {output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract metadata from JSON files using multi-threading.')
    parser.add_argument('--input_folder', type=str, required=True, help='Path to the folder containing JSON files.')
    parser.add_argument('--output_json', type=str, required=True, help='Path to the to the result JSON file.')

    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    extract_metadata_from_json(args.input_folder, args.output_json)