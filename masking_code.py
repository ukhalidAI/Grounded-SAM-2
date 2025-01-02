import os
import cv2
import torch
import argparse
import numpy as np
from PIL import Image

# Existing imports from your code
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForCausalLM
from utils.supervision_utils import CUSTOM_COLOR_MAP

TASK_PROMPT = {
    "caption": "<CAPTION>",
    "detailed_caption": "<DETAILED_CAPTION>",
    "more_detailed_caption": "<MORE_DETAILED_CAPTION",
    "object_detection": "<OD>",
    "dense_region_caption": "<DENSE_REGION_CAPTION>",
    "region_proposal": "<REGION_PROPOSAL>",
    "phrase_grounding": "<CAPTION_TO_PHRASE_GROUNDING>",
    "referring_expression_segmentation": "<REFERRING_EXPRESSION_SEGMENTATION>",
    "region_to_segmentation": "<REGION_TO_SEGMENTATION>",
    "open_vocabulary_detection": "<OPEN_VOCABULARY_DETECTION>",
    "region_to_category": "<REGION_TO_CATEGORY>",
    "region_to_description": "<REGION_TO_DESCRIPTION>",
    "ocr": "<OCR>",
    "ocr_with_region": "<OCR_WITH_REGION>",
}



# Initialize Models (same as your original initialization)
FLORENCE2_MODEL_ID = "microsoft/Florence-2-large"
SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

florence2_model = AutoModelForCausalLM.from_pretrained(FLORENCE2_MODEL_ID, trust_remote_code=True, torch_dtype='auto').eval().to(device)
florence2_processor = AutoProcessor.from_pretrained(FLORENCE2_MODEL_ID, trust_remote_code=True)

# build sam 2
sam2_model = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device=device)
sam2_predictor = SAM2ImagePredictor(sam2_model)

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
def run_florence2(task_prompt, text_input, model, processor, image):
    assert model is not None, "You should pass the init florence-2 model here"
    assert processor is not None, "You should set florence-2 processor here"

    device = model.device

    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input
    
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch.float16)
    generated_ids = model.generate(
      input_ids=inputs["input_ids"].to(device),
      pixel_values=inputs["pixel_values"].to(device),
      max_new_tokens=1024,
      early_stopping=False,
      do_sample=False,
      num_beams=3,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text, 
        task=task_prompt, 
        image_size=(image.width, image.height)
    )
    return parsed_answer


def extract_frames(video_path, fps=5, skip_start=2, skip_end=2):
    """Extract frames from a video at the specified FPS, skipping first and last frames."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_interval = max(1, video_fps // fps)

    frames = []
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Skip first and last frames
        if frame_idx < skip_start * video_fps or frame_idx > (total_frames - skip_end * video_fps):
            frame_idx += 1
            continue

        # Save frames at the specified interval
        if frame_idx % frame_interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))

        frame_idx += 1

    cap.release()
    return frames


def process_frame_and_save(frame, frame_count, mask_count, text_input, output_dir):
    """Process a single frame, save the frame and its mask."""
    frame_path = os.path.join(output_dir, f"frame_{frame_count}.png")
    mask_path = os.path.join(output_dir, f"mask_{mask_count}.png")

    # Save the frame
    frame.save(frame_path)

    # Generate the mask
    results = run_florence2(
        task_prompt="<OPEN_VOCABULARY_DETECTION>",
        text_input=text_input,
        model=florence2_model,
        processor=florence2_processor,
        image=frame,
    )
    input_boxes = np.array(results["<OPEN_VOCABULARY_DETECTION>"]["bboxes"])
    sam2_predictor.set_image(np.array(frame))
    masks, _, _ = sam2_predictor.predict(
        point_coords=None, point_labels=None, box=input_boxes, multimask_output=False
    )

    if masks.ndim == 4:
        masks = masks.squeeze(1)
    merged_mask = merge_masks(masks)
    binary_mask = (merged_mask * 255).astype(np.uint8)
    mask_pil = Image.fromarray(binary_mask).convert("L")
    mask_pil.save(mask_path)


def process_videos_in_directory(input_dir, output_dir, fps=5, text_input=None):
    """Process all .webm videos in the input directory."""
    video_files = [f for f in os.listdir(input_dir) if f.endswith(".webm")]
    frame_count, mask_count = 0, 0

    for video_idx, video_file in enumerate(video_files):
        video_path = os.path.join(input_dir, video_file)
        print(f"Processing video {video_idx + 1}/{len(video_files)}: {video_file}")

        # Extract frames
        frames = extract_frames(video_path, fps=fps)
        print(f"Extracted {len(frames)} frames from {video_file}")

        # Process each frame
        for frame in frames:
            frame_count += 1
            mask_count += 1
            process_frame_and_save(frame, frame_count, mask_count, text_input, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Video Processing with Mask Generation", add_help=True)
    parser.add_argument("--input_dir", type=str, required=False,default = "./../avatar_data", help="Path to directory with .webm videos")
    parser.add_argument("--output_dir", type=str, default="./../inpainting_data", help="Directory to save frames and masks")
    parser.add_argument("--fps", type=int, default=5, help="Frames per second to process")
    parser.add_argument("--text_input", type=str, required=False,default= "hands" ,help="Text input for mask generation")

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    process_videos_in_directory(args.input_dir, args.output_dir, fps=args.fps, text_input=args.text_input)
