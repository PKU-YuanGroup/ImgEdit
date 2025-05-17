import os
from imgedit.test_dlc.ip_utils import get_local_ip
import requests
import time
import torch
import subprocess

wait_time = 120
python_path = os.environ.get("PYTHONPATH")
comfyui_path = os.environ.get("COMFYUI_PATH")
master_addr = os.environ.get("MASTER_ADDR")
master_port = os.environ.get("MASTER_PORT")
rank = os.environ.get("RANK")

local_ip = get_local_ip()

if __name__ == "__main__":
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    processes = []
    gpu_count = torch.cuda.device_count()
    for i in range(gpu_count):
        port = 8180 + i
        for _ in range(wait_time):
            try:
                requests.get(
                    f"http://{master_addr}:{master_port}/",
                    params={"worker_addr": f"{local_ip}:{port}"},
                )
                print(f"Worker {local_ip}:{port} registered")
                break
            except Exception as e:
                time.sleep(1)

        cmd = f"{python_path} {os.path.join(comfyui_path, 'main.py')} --port {port} --gpu-only --cuda-device {i} --listen 0.0.0.0 --gpu-only --cache-lru 64"
        log_file = open(os.path.join(log_dir, f"comfyui_{rank}_{i}.log"), "w")
        p = subprocess.Popen(
            cmd.split(), cwd=comfyui_path, shell=False, stdout=log_file, stderr=log_file
        )
        processes.append(p)

    for p in processes:
        p.wait()
