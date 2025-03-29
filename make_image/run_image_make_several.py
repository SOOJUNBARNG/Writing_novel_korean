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
    "A male but a little like rat wearing a business suit",
    "A female but a little like anthropomorphic rat wearing an elegant dress",
    
    "A male but a little like  anthropomorphic ox wearing a farmer's outfit",
    "A female but a little like  anthropomorphic ox wearing a oriental outfit",
    
    "A male but a little like  anthropomorphic tiger wearing a detective coat",
    "A female but a little like  anthropomorphic tiger wearing a doctor outfit",
    
    "A male but a little like  anthropomorphic rabbit dressed in knight general suit",
    "A female but a little like  anthropomorphic rabbit wearing like a florist",
    
    "A male but a little like  anthropomorphic dragon in royal Korean robes",
    "A female but a little like  anthropomorphic dragon like career woman",
    
    "A male but a little like  anthropomorphic snake wearing a bow tie",
    "A female but a little like  anthropomorphic snake wearing a stylish gown",
    
    "A male but a little like  horse wearing a royal robe",
    "A femalev Anthropomorphic horse wearing an elegant ballroom gown",
    
    "A male but a little like  anthropomorphic goat wearing a scholar’s robe",
    "A female but a little like  anthropomorphic goat wearing a soft woolen dress",
    
    "A male but a little like  Anthropomorphic monkey wearing a suit and tie",
    "A female but a little like  Anthropomorphic monkey wearing a chic blazer and skirt",
    
    "A male but a little like  Anthropomorphic rooster wearing a pilot uniform",
    "A female but a little like  Anthropomorphic hen wearing a stylish flight attendant",
    
    "A male but a little like  anthropomorphic dog wearing a health trainer outfit",
    "A female but a little like  anthropomorphic dog wearing a detective trench coat",
    
    "A male but a little like  anthropomorphic fit pig dressed as a police officer",
    "A female  but a little like  anthropomorphic fit pig wearing as a lawyer"
]


added_prompts = "Extremely Anthropomorphic, Hominized, Humanized, Nice looking, wearing well-fitted clothing, anime style, white background"

for prompt in animals_with_clothing:
    enhanced_prompt = prompt + ", " + added_prompts
    try:
        image = pipe(prompt=enhanced_prompt, init_image=leading_image, strength=0.8).images[0]
    except TypeError:
        print("init_image argument not supported. Please check the updated StableDiffusion3Pipeline documentation.")
        image = pipe(prompt=enhanced_prompt).images[0]

    # Save the generated image
    image.save(fr"{prompt}.png")
