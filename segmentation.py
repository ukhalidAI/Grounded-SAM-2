from PIL import Image
from lang_sam import LangSAM
import numpy as np
import cv2
def dilate_mask(mask, kernel_size=3, iterations=1):
    if mask.dtype != np.uint8:
        mask = (mask * 255).astype(np.uint8)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=iterations)
    return dilated_mask

def merge_masks(masks):
    merged_mask = np.zeros_like(masks[0])
    for mask in masks:
        merged_mask = np.logical_or(merged_mask, mask)
    dilated_binary_mask = dilate_mask(merged_mask, kernel_size=3, iterations=0)
    return (dilated_binary_mask > 0.5).astype(np.float32)
model = LangSAM()
image_pil = Image.open("./products/lotion.jpeg").convert("RGB")
text_prompt = "lotion"
results = model.predict([image_pil], [text_prompt])
masks = results[0]['masks']
merged_mask = merge_masks(masks)
binary_mask = (merged_mask * 255).astype(np.uint8)
mask_pil = Image.fromarray(binary_mask).convert("L")
mask_pil.save("./masks/lotion.png")