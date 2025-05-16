import json  
import argparse  
import os
from multiprocessing import Pool
from tqdm import tqdm  
from functools import partial  

def parse_args():  
    parser = argparse.ArgumentParser(description="Find item with specific path in JSON.")  
    parser.add_argument("--input_json", type=str, required=True, help="Path to input JSON file.")  
    parser.add_argument("--img_folder", type=str, required=True, help="Path to image folder.") 
    parser.add_argument("--output_json", type=str, required=True, help="Path to output JSON file.") 
    return parser.parse_args()  
  

  
def main():  
    args = parse_args()  
    data_exist = []
    with open(args.input_json, 'r') as file:  
        data = json.load(file)  
    for i in tqdm(data):
        if os.path.isfile(os.path.join(args.img_folder, str(i["path"]))):
            data_exist.append(i)

    with open(args.output_json, 'w', encoding='utf-8') as f:
        json.dump(data_exist, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":  
    main()  