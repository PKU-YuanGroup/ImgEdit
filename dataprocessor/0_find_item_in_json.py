import json  
import argparse  
from multiprocessing import Pool
from tqdm import tqdm  
from functools import partial  

def parse_args():  
    parser = argparse.ArgumentParser(description="Find item with specific path in JSON.")  
    parser.add_argument("--input_json", type=str, required=True, help="Path to input JSON file.")  
    parser.add_argument("--item", type=str, required=True, help="The item to search for.")  
    parser.add_argument("--value", type=str, required=True, help="The value to search for.")  
    parser.add_argument("--num_processes", type=int, default=4, help="Number of processes to use.")  
    return parser.parse_args()  
  
def search_item(items, value, item):  
    if items.get(item) == value:  
        return items 
    return None  
  
def main():  
    args = parse_args()  
  
    with open(args.input_json, 'r') as file:  
        data = json.load(file)  
    
    search_with_args = partial(search_item, item=args.item, value=args.value)  
    with Pool(args.num_processes) as pool:  
        for result in tqdm(pool.imap_unordered(search_with_args, data), total=len(data)):  
            if result is not None:  
                print(json.dumps(result, indent=4))  
                break 

if __name__ == "__main__":  
    main()  