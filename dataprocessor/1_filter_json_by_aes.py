import json  
import argparse  
from multiprocessing import Pool  
from tqdm import tqdm  
from functools import partial  

def parse_args():  
    parser = argparse.ArgumentParser(description="Select data by aes score and resolution")  
    parser.add_argument("--input_json", type=str, required=True, help="Path to input JSON file.")  
    parser.add_argument("--output_json", type=str, required=True, help="Path to output JSON file.")  
    parser.add_argument("--aes", type=float, default=5.0, help="filter aes score.") 
    parser.add_argument("--res", type=int, default=512, help="filter resolution.") 
    parser.add_argument("--num_processes", type=int, default=4, help="Number of processes to use for parallel processing")  
    return parser.parse_args()  
  
def filter_data(item, aes, res):  
    if item['aes'] > aes and item['resolution']['height'] > res and item['resolution']['width'] > res:  
        return item  
    else: 
        return None  
  
def main():  
    args = parse_args()  

    with open(args.input_json, 'r') as file:  
        data = json.load(file)  
    
    filter_data_with_thresholds = partial(filter_data, aes=args.aes, res=args.res)  

    with Pool(args.num_processes) as pool:  
        filtered_data = list(tqdm(pool.imap_unordered(filter_data_with_thresholds, data), total=len(data)))
  

    filtered_data = [item for item in filtered_data if item is not None]  
  

    with open(args.output_json, 'w') as file:  
        json.dump(filtered_data, file, indent=4)  
  
    print(f"Filtered {len(filtered_data)} items with aes > {args.aes} and resolution > {args.res}*{args.res}.")  
  
if __name__ == "__main__":  
    main()  