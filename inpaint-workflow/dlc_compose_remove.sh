export RANK=${RANK:-0}
export MASTER_ADDR=${MASTER_ADDR:-localhost}
export MASTER_PORT=${MASTER_PORT:-12345}
export WORLD_SIZE=${WORLD_SIZE:-1}
export WORLD_SIZE=$((WORLD_SIZE * 8))
export PYTHONPATH=/mnt/data/lzj/miniconda3/envs/comfyui/bin/python
export COMFYUI_PATH=/mnt/data/lzj/codes/ComfyUI

TASK_KWARGS="--json_dir /mnt/data/yangye/imgedit/data/step5_final/laion_part2_premise/ \
--base_img_dir /mnt/data/osp/imgdata/laion-aes \
--result_dir results_premise_part2 \
--task_type multi-premise"

mkdir -p logs

echo $TASK_KWARGS

if [ $RANK -eq 0 ]; then
    # if is master
    $PYTHONPATH dlc_compose_remove.py $TASK_KWARGS &
    $PYTHONPATH register_worker.py
else
    # if is worker
    $PYTHONPATH register_worker.py
fi