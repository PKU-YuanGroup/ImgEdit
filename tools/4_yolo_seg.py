import argparse
import json
import os
import cv2
import sys
import numpy as np
import pycocotools.mask as mask_util
import supervision as sv
import torch
from mmdet.apis import init_detector
from mmdet.utils import get_test_pipeline_cfg
from mmengine.config import Config, DictAction
from mmengine.dataset import Compose
from mmengine.runner.amp import autocast
from PIL import Image
from torchvision.ops import nms
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import warnings
warnings.filterwarnings('ignore')

sys.path.append('model_zoo/YOLO-World')  

os.environ["TOKENIZERS_PARALLELISM"] = "false"

BOUNDING_BOX_ANNOTATOR = sv.BoxAnnotator(thickness=1)
MASK_ANNOTATOR = sv.MaskAnnotator()


class LabelAnnotator(sv.LabelAnnotator):
    @staticmethod
    def resolve_text_background_xyxy(
        center_coordinates,
        text_wh,
        position,
    ):
        center_x, center_y = center_coordinates
        text_w, text_h = text_wh
        return center_x, center_y, center_x + text_w, center_y + text_h


LABEL_ANNOTATOR = LabelAnnotator(text_padding=4, text_scale=0.5, text_thickness=1)


class ImageDataset(Dataset):
    def __init__(self, json_files, images_folder, output_path):
        self.json_files = json_files
        self.images_folder = images_folder
        self.output_path = output_path

    def __len__(self):
        return len(self.json_files)

    def __getitem__(self, idx):
        json_file = self.json_files[idx]
        with open(json_file, "r") as f:
            data = json.load(f)

        image_file = os.path.join(self.images_folder, data["path"])
        np_image = np.asarray(Image.open(image_file).convert("RGB"))
        img_name = data["path"].replace("/","")
        local_output_path = os.path.join(self.output_path, img_name.replace(".jpg", ".json"))


        return {
            "img_name": img_name,
            "image": np_image,
            "json_file": json_file,
            "background": [[item] for item in data["tags"]["background"]],
            "object": [[item] for item in data["tags"]["object"]],
            "raw_data": data,
            "local_output_path": local_output_path
        }

def custom_collate_fn(batch):
    return batch

def inference_detector(
    model,
    image_path,
    image_np,
    texts,
    test_pipeline,
    max_dets=100,
    score_thr=0.005,
    nms_thr=0.5,
    output_path="./work_dir",
    use_amp=False,
    annotation=False,
):
    data_info = {"img_id": 0, "img": image_np, "texts": texts}
    data_info = test_pipeline(data_info)
    data_batch = {
        "inputs": data_info["inputs"].unsqueeze(0),
        "data_samples": [data_info["data_samples"]],
    }

    with autocast(enabled=use_amp), torch.no_grad():
        output = model.test_step(data_batch)[0]
        pred_instances = output.pred_instances
        # pred_instances = pred_instances[pred_instances.scores.float() >
        #                                 score_thr]

    keep = nms(pred_instances.bboxes, pred_instances.scores, iou_threshold=nms_thr)
    pred_instances = pred_instances[keep]
    pred_instances = pred_instances[pred_instances.scores.float() > score_thr]

    if len(pred_instances.scores) > max_dets:
        indices = pred_instances.scores.float().topk(max_dets)[1]
        pred_instances = pred_instances[indices]

    pred_instances = pred_instances.cpu().numpy()

    bboxes = pred_instances["bboxes"]
    class_ids = pred_instances["labels"]
    confidence = pred_instances["scores"]

    if annotation:
        if "masks" in pred_instances:
            masks = pred_instances["masks"]
        else:
            masks = None

        detections = sv.Detections(xyxy=bboxes, class_id=class_ids, confidence=confidence, mask=masks)

        labels = [
            f"{texts[class_id][0]} {confidence:0.2f}"
            for class_id, confidence in zip(detections.class_id, detections.confidence)
        ]

        # label images
        image = image_np[..., ::-1].astype(np.uint8)
        image = BOUNDING_BOX_ANNOTATOR.annotate(image, detections)
        image = LABEL_ANNOTATOR.annotate(image, detections, labels=labels)
        if masks is not None:
            image = MASK_ANNOTATOR.annotate(image, detections)

        cv2.imwrite(
            os.path.join(output_path, os.path.basename(image_path).replace(".jpg", "") + "_detection.jpg"), image
        )

    return bboxes, class_ids, confidence


def inference_all(args, yolo_model, test_pipeline, sam_model, texts, image_np, image_path, output_path):
    """
    YoloWorld inference
    """

    # reparameterize texts
    yolo_model.reparameterize(texts)

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float32):
        bboxes, class_ids, confidence = inference_detector(
            model=yolo_model,
            image_path=image_path,
            image_np=image_np,
            texts=texts,
            test_pipeline=test_pipeline,
            max_dets=args.topk,
            score_thr=args.score_thr,
            nms_thr=args.nms_thr,
            output_path=output_path,
            use_amp=args.amp,
            annotation=args.bbox_annotation,
        )

    """
    SAM inference
    """
    input_boxes = bboxes
    class_names = [texts[class_id][0] for class_id in class_ids]
    confidences = torch.from_numpy(confidence)

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        sam_model.set_image(image_np)
        masks, scores, logits = sam_model.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )

    return input_boxes, class_names, confidences, masks, scores


def post_process_all(
    all_annotation,
    image_path,
    image_np,
    input_boxes,
    class_names,
    confidences,
    scores,
    masks,
    output_path,
    dump_single_json_results,
    dump_json_results,
    object,
    background,
):
    """
    Post-process the output of the model to get the masks, scores, and logits for visualization
    """
    # convert the shape to (n, H, W)
    if masks.ndim == 4:
        masks = masks.squeeze(1)

    class_ids = np.array(list(range(len(class_names))))

    labels = [f"{class_name} {confidence:.2f}" for class_name, confidence in zip(class_names, confidences)]

    """
    Visualize image with supervision useful API
    """
    if all_annotation:
        img = image_np[..., ::-1].astype(np.uint8)
        detections = sv.Detections(
            xyxy=input_boxes,  # (n, 4)
            mask=masks.astype(bool),  # (n, h, w)
            class_id=class_ids,
        )

        box_annotator = sv.BoxAnnotator()
        annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)

        label_annotator = sv.LabelAnnotator()
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
        cv2.imwrite(
            os.path.join(output_path, os.path.basename(image_path).replace(".jpg", "") + "_detection.jpg"),
            annotated_frame,
        )

        mask_annotator = sv.MaskAnnotator()
        annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)

        cv2.imwrite(
            os.path.join(output_path, os.path.basename(image_path).replace(".jpg", "") + "_segmentation.jpg"),
            annotated_frame,
        )

    """
    Dump the results in standard format and save as json files
    """

    def single_mask_to_rle(mask):
        rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
        rle["counts"] = rle["counts"].decode("utf-8")
        return rle

    if dump_single_json_results or dump_json_results:
        # convert mask into rle format
        mask_rles = [single_mask_to_rle(mask) for mask in masks]

        input_boxes = input_boxes.tolist()
        scores = scores.tolist()
        # save the results in standard format
        results = {
            "background": [],
            "object": [],
            "box_format": "xyxy",
        }
        for class_name, box, mask_rle, score in zip(class_names, input_boxes, mask_rles, scores):
            annotation = {
                "class_name": class_name,
                "bbox": box,
                "mask": mask_rle["counts"],
                "score": score[0],
            }
            if [class_name] in object:
                results["object"].append(annotation)
            elif [class_name] in background:
                results["background"].append(annotation)
            else:
                raise NotImplementedError


        if dump_json_results:
            return results
        else:
            with open(
                os.path.join(
                    output_path, os.path.basename(image_path).replace(".jpg", "") + ".json"
                ),
                "w",
            ) as f:
                json.dump(results, f, indent=4)
            return None


def parse_args():
    parser = argparse.ArgumentParser(description="YoloWorld-SAM2.1 Model Configuration")

    # all
    parser.add_argument(
        "--json_folder",
        required=True,
        type=str,
        help="Path to the folder containing JSON files.",
    )
    parser.add_argument(
        "--image_folder", 
        required=True,
        type=str, 
        help="Path to the input image",
    )
    parser.add_argument(
        "--output_path",
        required=True,
        type=str,
        help="Path to save the output",
    )

    parser.add_argument(
        "--bbox_annotation", action="store_true", help="save the annotated detection results as YOLO Text format."
    )
    parser.add_argument(
        "--all_annotation",
        action="store_true",
        help="save the annotated detection results as YoloWorld-SAM2.1 format.",
    )
    parser.add_argument("--dump_single_json_results", action="store_true", help="Flag to dump JSON results")
    parser.add_argument("--dump_json_results", action="store_true", help="Flag to dump JSON results")

    # sam2.1
    parser.add_argument(
        "--sam_checkpoint",
        type=str,
        default="./model_zoo/sam2.1/sam2.1_hiera_large.pt",
        help="Path to the model sam_checkpoint",
    )
    parser.add_argument(
        "--sam_model_cfg",
        type=str,
        default="./model_zoo/sam2.1/sam2.1_hiera_l.yaml",
        help="Path to the model configuration file",
    )

    # yolo-world
    parser.add_argument(
        "--yolo_config",
        default="./model_zoo/yoloworld/configs/yolo_world_v2_xl_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py",
        type=str,
        help="test config file path",
    )
    parser.add_argument(
        "--yolo_checkpoint",
        default="./model_zoo/yoloworld/configs/yolo_world_v2_xl_obj365v1_goldg_cc3mlite_pretrain-5daf1395.pth",
        type=str,
        help="checkpoint file",
    )
    parser.add_argument("--topk", default=100, type=int, help="keep topk predictions.")
    parser.add_argument("--score_thr", default=0.005, type=float, help="confidence score threshold for predictions.")
    parser.add_argument("--nms_thr", default=0.5, type=float, help="confidence score threshold for predictions.")
    parser.add_argument("--device", default="cuda", help="device used for inference.")
    parser.add_argument("--amp", action="store_true", help="use mixed precision for inference.")
    parser.add_argument(
        "--cfg_options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file. If the value to "
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        "Note that the quotation marks are necessary and that no white space "
        "is allowed.",
    )

    parser.add_argument("--split_nums", default=8, type=int)
    parser.add_argument("--part", default=0, type=int)

    return parser.parse_args()


def split_list(data, nums, part):
    size = len(data)
    part_size = size // nums
    remainder = size % nums

    start = part * part_size + min(part, remainder)
    end = start + part_size + (1 if part < remainder else 0)

    return data[start:end]


if __name__ == "__main__":
    args = parse_args()

    assert args.dump_single_json_results != args.dump_json_results

    """
    Init input
    """
    sam_checkpoint = args.sam_checkpoint
    sam_model_cfg = args.sam_model_cfg
    image_folder = args.image_folder
    output_path = args.output_path
    dump_single_json_results = args.dump_single_json_results
    dump_json_results = args.dump_json_results
    split_nums = args.split_nums
    part = args.part
    
    os.makedirs(output_path, exist_ok=True)

    json_files = [
        os.path.join(args.json_folder, f)
        for f in os.listdir(args.json_folder)
        if f.endswith(".json")
        and not os.path.exists(os.path.join(output_path, os.path.basename(f).replace(".json", "_step4.json")))
    ]
    # json_files = split_list(json_files, split_nums, part)

    """
    Load YoloWorld
    """

    
    cfg = Config.fromfile(args.yolo_config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    cfg.work_dir = os.path.join("./work_dirs", os.path.splitext(os.path.basename(args.yolo_config))[0])
    # init model
    cfg.load_from = args.yolo_checkpoint
    
    yolo_model = init_detector(cfg, checkpoint=args.yolo_checkpoint, device=args.device)

    # init test pipeline
    test_pipeline_cfg = get_test_pipeline_cfg(cfg=cfg)
    test_pipeline_cfg[0].type = "mmdet.LoadImageFromNDArray"
    test_pipeline = Compose(test_pipeline_cfg)

    """
    Load SAM2.1
    """
    sam_model = SAM2ImagePredictor(build_sam2(sam_model_cfg, sam_checkpoint))
    
    """
    Main Loop DataLoader
        {
            "img_name": img_name,
            "image": np_image,
            "json_file": json_file,
            "background": [[item] for item in data["tags"]["background"]],
            "object": [[item] for item in data["tags"]["object"]],
            "raw_data": data,
            "local_output_path": local_output_path
        }
    """
    # yang modified
    dataset = ImageDataset(json_files, image_folder, output_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, prefetch_factor=4, pin_memory=True, collate_fn=custom_collate_fn)
    
    for batch in tqdm(dataloader, desc="Processing Files", total=len(dataloader)):
        batch = batch[0]
        img_name = batch["img_name"]
        json_file = batch["json_file"]
        image_np = batch["image"]
        background = batch["background"]
        object = batch["object"]
        local_output_path = batch["local_output_path"]
        raw_data = batch["raw_data"]

        """
        Main Inference
        """

        try:
            input_boxes, class_names, confidences, masks, scores = inference_all(
                args=args,
                yolo_model=yolo_model,
                test_pipeline=test_pipeline,
                sam_model=sam_model,
                texts=object+background,
                image_np=image_np,
                image_path=img_name,
                output_path=output_path,
            )
        except Exception as e:
            continue

        """
        Post-process and Save
        """
        result = post_process_all(
            all_annotation=args.all_annotation,
            image_path=img_name,
            image_np=image_np,
            input_boxes=input_boxes,
            class_names=class_names,
            confidences=confidences,
            scores=scores,
            masks=masks,
            output_path=output_path,
            dump_single_json_results=dump_single_json_results,
            dump_json_results=dump_json_results,
            object=object,
            background=background,
        )

        output_json_file = os.path.basename(json_file).replace(".json", "_step4.json")
        raw_data["segmentation"] = result
        output_json_path = os.path.join(output_path, output_json_file)
        with open(output_json_path, "w") as f:
            json.dump(raw_data, f, indent=4)