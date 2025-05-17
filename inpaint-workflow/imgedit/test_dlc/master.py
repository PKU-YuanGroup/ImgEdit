import os
import requests
import time
import threading
from flask import Flask, request

master_addr = os.environ.get("MASTER_ADDR")
master_port = os.environ.get("MASTER_PORT")
world_size = int(os.environ.get("WORLD_SIZE"))

workers: list[str] = []
app = Flask(__name__)


@app.route("/")
def ping():
    worker_addr = request.args.get("worker_addr")
    print(f"[Master] Worker {worker_addr} is registering")
    workers.append(worker_addr)
    return "pong"


def run_flask():
    app.run(host=master_addr, port=master_port)


def dlc_context_runner(func, wait_time=120, *args, **kwargs):
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.start()

    while len(workers) < world_size - 1:
        time.sleep(1)
        print(f"[Master] Waiting for {len(workers)}/{world_size} workers to register")

    print(f"[Master] All {world_size} workers registered")

    for _ in range(wait_time):
        try:
            for worker in workers:
                print(f"[Master] Checking worker {worker}")
                resp = requests.post(f"http://{worker}/prompt")
                print(resp.text)
                print(f"[Master] Worker {worker} is ready")
        except Exception as e:
            time.sleep(1)
            continue
        break

    print(f"[Master] All workers are ready")

    func(*args, **kwargs, workers=workers)
