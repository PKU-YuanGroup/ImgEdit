import json  
import argparse
all_data = {}
def parse_args():
    parser = argparse.ArgumentParser(description="Process some input and output folders with model settings.")
    parser.add_argument(
        "--input_json", type=str, required=True, help="Path to the caption json."
    )
    parser.add_argument(
        "--output_json", type=str, required=True, help="Path to the output json folder."
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    with open(args.input_json, 'r', encoding="utf-8") as f:
        data = json.load(f)

    for item in data:  
        key = item["path"].replace("/", "").replace(".jpg", "")  
        value = item  
        all_data[key] = value  

    with open(args.output_json, "w") as f:
        json.dump(all_data, f, indent=4)