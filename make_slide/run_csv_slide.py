import torch
import pandas as pd
from diffusers import StableDiffusion3Pipeline
from PIL import Image

# Load the model
pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3.5-medium", torch_dtype=torch.bfloat16
).to("cuda")

# Read CSV and extract prompts
prompt_csv = pd.read_csv("./counseler_read.csv")

# Assume first column contains prompts; adjust if needed
prompt_list = prompt_csv.iloc[:, 0].dropna().tolist()  
prompt = " ".join(map(str, prompt_list))  # Convert to string

# Additional styling prompts
added_prompts = "Very formal, ppt slide, scatter-plot base"
enhanced_prompt = f"{prompt}, {added_prompts}"

# Generate image
image = pipe(enhanced_prompt).images[0]

# Save the generated image
image.save("check.png")

