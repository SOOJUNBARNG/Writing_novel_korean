import torch

from diffusers import StableDiffusion3Pipeline
from PIL import Image

# Load the model
pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-medium", torch_dtype=torch.bfloat16)
pipe = pipe.to("cuda")

# Load your leading image (starting image) - modify this path to your image
leading_image_path = r"D:\Writing_novel_korean\make_image\related_image\horseman.webp"
leading_image = Image.open(leading_image_path).convert("RGB")

animals_with_clothing = [
    "A male anthropomorphic rat wearing a business suit",
    "A female anthropomorphic rat wearing an elegant dress",
    
    "A male anthropomorphic ox wearing a farmer's outfit",
    "A female anthropomorphic ox wearing a oriental outfit",
    
    "A male anthropomorphic tiger wearing a detective coat",
    "A female anthropomorphic tiger wearing a doctor outfit",
    
    "A male anthropomorphic rabbit dressed in knight general suit",
    "A female anthropomorphic rabbit wearing like a florist",
    
    "A male anthropomorphic dragon in royal Korean robes",
    "A female anthropomorphic dragon with pilates dress",
    
    "A male anthropomorphic snake wearing a bow tie",
    "A female anthropomorphic snake wearing a stylish gown",
    
    "A male horse wearing a royal robe",
    "A female Anthropomorphic horse wearing an elegant ballroom gown",
    
    "A male anthropomorphic goat wearing a scholar’s robe",
    "A female anthropomorphic goat wearing a soft woolen dress",
    
    "A male Anthropomorphic gorila wearing a suit and tie",
    "A female Anthropomorphic gorila wearing a chic blazer and skirt",
    
    "A male Anthropomorphic rooster wearing a pilot uniform",
    "A female Anthropomorphic rooster wearing a stylish flight attendant",
    
    "A male anthropomorphic dog wearing a superhero costume",
    "A female anthropomorphic dog wearing a detective trench coat",
    
    "A male anthropomorphic fit pig dressed as a police officer",
    "A female anthropomorphic fit pig wearing as a lawyer"
]


added_prompts = "Extremely Anthropomorphic,Hominized, Humanized, Nice looking, wearing well-fitted clothing, anime style, white background"

for prompt in animals_with_clothing:
    enhanced_prompt = prompt + ", " + added_prompts
    try:
        image = pipe(prompt=enhanced_prompt, init_image=leading_image, strength=0.8).images[0]
    except TypeError:
        print("init_image argument not supported. Please check the updated StableDiffusion3Pipeline documentation.")
        image = pipe(prompt=enhanced_prompt).images[0]

    # Save the generated image
    image.save(fr"{prompt}.png")
