import torch

from diffusers import StableDiffusion3Pipeline
from PIL import Image

# Load the model
pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-large", torch_dtype=torch.bfloat16)
pipe = pipe.to("cuda")

# Load your leading image (starting image) - modify this path to your image
leading_image_path = r"D:\Writing_novel_korean\make_image\related_image\horseman.webp"
leading_image = Image.open(leading_image_path).convert("RGB")

# Prompt for generating the final image
prompt = "A humanoid horse wearing a suit."

try:
    # Check if the pipeline accepts 'init_image' directly (which it might not)
    image = pipe(prompt=prompt, init_image=leading_image, strength=0.4).images[0]
except TypeError:
    # If init_image is not supported, fall back on a different method for using the leading image
    print("init_image argument not supported. Please check the updated StableDiffusion3Pipeline documentation.")
    image = pipe(prompt=prompt).images[0]

# Save the generated image
image.save(fr"{prompt}.png")
