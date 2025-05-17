export RANK=${RANK:-0}
export MASTER_ADDR=${MASTER_ADDR:-localhost}
export MASTER_PORT=${MASTER_PORT:-12345}
export WORLD_SIZE=${WORLD_SIZE:-1}
export WORLD_SIZE=$((WORLD_SIZE * 8))
export PYTHONPATH=/mnt/data/lzj/miniconda3/envs/comfyui/bin/python
export COMFYUI_PATH=/mnt/data/lzj/codes/ComfyUI

TASK_KWARGS="--image_txt /mnt/data/captions/img_cap/cap_final/cap_merge_final/25_4_6_finetune/1280/laion-nolang/part7_cap36471.json \
--style_txt /mnt/data/lzj/codes/imgedit_comfyui/styles.txt \
--base_img_dir /mnt/data/osp/imgdata/laion-nolang \
--result_dir results_style_transfer_part7_cap36471_ghibli" 

mkdir -p logs

# $PYTHONPATH dlc_style_transfer.py $TASK_KWARGS

if [ $RANK -eq 0 ]; then
    # if is master
    $PYTHONPATH dlc_style_transfer.py $TASK_KWARGS &
    $PYTHONPATH register_worker.py
else
    # if is worker
    $PYTHONPATH register_worker.py
fi