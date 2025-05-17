import pycocotools.mask as mask_util
from PIL import Image

def rle_to_mask(rle_str, height, width):
    rle = {"counts": rle_str.encode("utf-8"), "size": [height, width]}
    mask = mask_util.decode(rle)
    mask = Image.fromarray((1 - mask) * 255).convert("L")
    return mask
