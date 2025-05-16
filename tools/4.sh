python3 tools/4_yolo_seg.py \
    --image_folder "data/img" \
    --json_folder "data/pair_data" \
    --output_path "data/yolo_out" \
    --sam_checkpoint "//yolo/model_zoo/sam2.1/sam2.1_hiera_large.pt" \
    --sam_model_cfg "//yolo/model_zoo/sam2.1/sam2.1_hiera_l.yaml" \
    --yolo_checkpoint "model_zoo/yoloworld/yolo_world_v2_xl_obj365v1_goldg_cc3mlite_pretrain-5daf1395.pth" \
    --yolo_config "model_zoo/yoloworld/configs/yolo_world_v2_xl_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py" \
    --topk 100 \
    --score_thr 0.005 \
    --nms_thr 0.5 \
    --dump_json_results \
    --bbox_annotation \
    --all_annotation

    # --all_annotation \