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
    "A male rat wearing a business suit",
    "A female rat wearing an elegant dress",
    
    "A male ox wearing a farmer's outfit",
    "A female ox wearing a traditional Chinese dress",
    
    "A male tiger wearing a detective coat",
    "A female tiger wearing a warrior’s outfit",
    
    "A male rabbit dressed in knight’s armor",
    "A female rabbit wearing a floral dress",
    
    "A male dragon in royal Chinese robes",
    "A female dragon in a flowing ceremonial dress",
    
    "A male snake wearing a bow tie in an upscale restaurant",
    "A female snake wearing a stylish gown",
    
    "A male horse wearing a royal robe",
    "A female horse wearing an elegant ballroom gown",
    
    "A male goat wearing a scholar’s robe",
    "A female goat wearing a soft woolen dress",
    
    "A male monkey wearing a suit and tie in a corporate office",
    "A female monkey wearing a chic blazer and skirt",
    
    "A male rooster wearing a pilot uniform",
    "A female rooster wearing a stylish flight attendant outfit",
    
    "A male dog wearing a superhero costume",
    "A female dog wearing a detective trench coat",
    
    "A male pig dressed as a chef",
    "A female pig wearing a baker’s apron"
]


added_prompts = "Anthropomorphic, Nice looking, delightful face, standing upright, wearing well-fitted clothing, anime style, no background"

for prompt in animals_with_clothing:
    enhanced_prompt = prompt + ", " + added_prompts
    try:
        image = pipe(prompt=enhanced_prompt, init_image=leading_image, strength=0.8).images[0]
    except TypeError:
        print("init_image argument not supported. Please check the updated StableDiffusion3Pipeline documentation.")
        image = pipe(prompt=enhanced_prompt).images[0]

    # Save the generated image
    image.save(fr"{prompt}.png")
