from PIL import Image
from lang_sam import LangSAM
import numpy as np
import cv2
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
# default: Load the model on the available device(s)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-72B-Instruct", torch_dtype="auto", device_map="auto"
)

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-72B-Instruct",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)

# default processer
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-72B-Instruct")
# Messages containing multiple images and a text query
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image",
                "image": "file://./Avatars/beach_girl.jpg",
                "resized_height": 280,
                "resized_width": 420,},
            {"type": "image", "image": "file://./lotion/download.jpeg", "resized_height": 280,
                "resized_width": 420,},
            {"type": "text", "text": "Given we have an image of a person and a product. We want to insert the product in the person's image. Your job is to first identify the suitable size of the product relative to the person's picture and then should give the  coordinates for the center where product should be inserted such that we can show that the person is holding an object. Your coordinates can be in the form of the ratio as compared to the original image. "},
        ],
    }
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

# Inference
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)

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
image_pil = Image.open("./lotion/download.jpeg").convert("RGB")
text_prompt = "bottle"
results = model.predict([image_pil], [text_prompt])
masks = results[0]['masks']
merged_mask = merge_masks(masks)
binary_mask = (merged_mask * 255).astype(np.uint8)
mask_pil = Image.fromarray(binary_mask).convert("L") # Mask
#mask_pil.save("mask_lotion.png")