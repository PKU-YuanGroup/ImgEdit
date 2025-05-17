export RANK=${RANK:-0}
export MASTER_ADDR=${MASTER_ADDR:-localhost}
export MASTER_PORT=${MASTER_PORT:-12345}
export WORLD_SIZE=${WORLD_SIZE:-1}
export WORLD_SIZE=$((WORLD_SIZE * 8))
export PYTHONPATH=/mnt/data/lzj/miniconda3/envs/comfyui/bin/python
export COMFYUI_PATH=/mnt/data/lzj/codes/ComfyUI

TASK_KWARGS="--json_dir /mnt/data/yangye/imgedit/data/step5_final/laion_part0_version_refer/ \
--base_img_dir /mnt/data/osp/imgdata/laion-aes \
--result_dir results_version_refer_part0" 

mkdir -p logs

if [ $RANK -eq 0 ]; then
    # if is master
    $PYTHONPATH dlc_compose_replace_version.py $TASK_KWARGS &
    $PYTHONPATH register_worker.py
else
    # if is worker
    $PYTHONPATH register_worker.py
fi