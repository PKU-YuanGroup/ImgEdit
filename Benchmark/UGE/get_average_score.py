import json
import argparse

def extract_scores_and_average(entry: str) -> float:
    lines = entry.splitlines()
    scores = []
    for line in lines:
        parts = line.strip().split(': ')
        if len(parts) == 2 and parts[1].isdigit():
            scores.append(int(parts[1]))
    if scores:
        return round(sum(scores) / len(scores), 2)
    return None

def compute_final_average(result_json_dict):
    total_scores = []
    
    for value in result_json_dict.values():
        avg = extract_scores_and_average(value)
        if avg is not None:
            total_scores.append(avg)
    
    if total_scores:
        return round(sum(total_scores) / len(total_scores), 2)
    return None

def main():
    parser = argparse.ArgumentParser(description="Calculate the average score for all keys and print the final average")
    parser.add_argument('--result_json', type=str, required=True, help='Path of result json')

    args = parser.parse_args()

    with open(args.result_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    final_average = compute_final_average(data)

    if final_average is not None:
        print(final_average)
    else:
        print("No valid scores found.")

if __name__ == '__main__':
    main()
