import os
import json
import argparse
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


def process_json_file(file_path):
    """
    处理单个 JSON 文件，提取 metadata 数据
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)  # 读取 JSON 文件内容
    except json.JSONDecodeError:
        print(f"File {os.path.basename(file_path)} is not a json file")
    return data


def extract_metadata_from_json(input_folder, output_file):
    data_list = []  # 用来存储所有的 metadata 元素
    
    # 遍历文件夹中的所有文件
    json_files = [os.path.join(input_folder, filename) for filename in os.listdir(input_folder) if filename.endswith('.json')]
    
    # 使用线程池来并发处理 JSON 文件，并通过 tqdm 显示进度条
    with ThreadPoolExecutor() as executor:
        # tqdm 用于显示进度条
        results = list(tqdm(executor.map(process_json_file, json_files), total=len(json_files), desc="Processing files"))
        
        # 合并所有结果
        for result in results:
            if isinstance(result, list):
                print("shit")
                data_list.extend(result)  # 合并列表
            elif isinstance(result, dict):
                data_list.append(result) 
    
    # 将提取的 metadata 列表保存到一个新的 JSON 文件
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