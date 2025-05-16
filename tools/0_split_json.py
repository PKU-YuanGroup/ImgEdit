import json 
import os 
import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Process some input and output folders with model settings.")
    parser.add_argument(
        "--input_json", type=str, required=True, help="Path to the caption json."
    )
    parser.add_argument(
        "--output_folder", type=str, required=True, help="Path to the output json folder."
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_folder, exist_ok=True)
    with open(args.input_json, 'r', encoding="utf-8") as f:
        data = json.load(f)

    for key, value in data.items():  
        with open(os.path.join(args.output_folder, f"{key}.json"), "w") as f:
            json.dump(value, f, indent=4)