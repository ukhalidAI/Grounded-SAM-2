from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image
import torch
from PIL import Image

pipe = AutoPipelineForInpainting.from_pretrained("diffusers/stable-diffusion-xl-1.0-inpainting-0.1", torch_dtype=torch.float16, variant="fp16").to("cuda")

# img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
# mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"
image_path = "./testing/sample_image.jpg"  # Replace with your original image path
mask_path = "./testing/sample_mask.jpg" 
# Load the image and mask as PIL images
image = Image.open(image_path).convert("RGB").resize((1024, 1024))
mask_image = Image.open(mask_path).convert("L").resize((1024, 1024))  # Ensure the mask is grayscale
# image = load_image(img_url).
# mask_image = load_image(mask_url).resize((1024, 1024))

prompt = "a person holding the product"
generator = torch.Generator(device="cuda").manual_seed(0)

image = pipe(
  prompt=prompt,
  image=image,
  mask_image=mask_image,
  guidance_scale=8.0,
  num_inference_steps=20,  # steps between 15 and 30 work well for us
  strength=0.99,  # make sure to use `strength` below 1.0
  generator=generator,
).images[0]

image.save(f"./testing/result.png")

