# import torch
# from diffusers import StableDiffusionInpaintPipeline
from PIL import Image

# # Load the pipeline
# pipe = StableDiffusionInpaintPipeline.from_pretrained(
#     "stabilityai/stable-diffusion-2-inpainting",
#     torch_dtype=torch.float16,
# )
# pipe.to("cuda")

# # Define paths to image and mask
# image_path = "./testing/sample_image.jpg"  # Replace with your original image path
# mask_path = "./testing/sample_mask.jpg" 
# # Load the image and mask as PIL images
# image = Image.open(image_path).convert("RGB")
# mask_image = Image.open(mask_path).convert("L")  # Ensure the mask is grayscale

# # Define the inpainting prompt
# prompt = "person holding the product"

# # Run the inpainting pipeline
# result = pipe(prompt=prompt, image=image, mask_image=mask_image).images[0]

# # Save the result
# output_path = "./testing/result.png"
# result.save(output_path)
# print(f"Inpainted image saved at: {output_path}")

import torch
from diffusers.utils import load_image, check_min_version
from diffusers.pipelines import StableDiffusion3ControlNetInpaintingPipeline
from diffusers.models.controlnet_sd3 import SD3ControlNetModel

controlnet = SD3ControlNetModel.from_pretrained(
    "alimama-creative/SD3-Controlnet-Inpainting", use_safetensors=True, extra_conditioning_channels=1
)
pipe = StableDiffusion3ControlNetInpaintingPipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers",
    controlnet=controlnet,
    torch_dtype=torch.float16,
)
pipe.text_encoder.to(torch.float16)
pipe.controlnet.to(torch.float16)
pipe.to("cuda")

image_path = "./testing/sample_image.jpg"  # Replace with your original image path
mask_path = "./testing/sample_mask.jpg" 

image = Image.open(image_path).convert("RGB")
mask = Image.open(mask_path).convert("L")  # Ensure the mask is grayscale

width = 1024
height = 1024
prompt = "A person holding the product"
generator = torch.Generator(device="cuda").manual_seed(24)
res_image = pipe(
    negative_prompt="deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, mutated hands and fingers, disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation, NSFW",
    prompt=prompt,
    height=height,
    width=width,
    control_image=image,
    control_mask=mask,
    num_inference_steps=28,
    generator=generator,
    controlnet_conditioning_scale=0.95,
    guidance_scale=7,
).images[0]
res_image.save(f"./testing/result.png")
