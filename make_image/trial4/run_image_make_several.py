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
    "A female ox wearing a kimono dress",
    
    "A male tiger wearing a detective coat",
    "A female tiger wearing a doctor outfit",
    
    "A male rabbit dressed in knight general suit",
    "A female rabbit wearing like a florist",
    
    "A male dragon in royal Korean robes",
    "A female dragon with pilates dress",
    
    "A male snake wearing a bow tie",
    "A female snake wearing a stylish gown",
    
    "A male horse wearing a royal robe",
    "A female Anthropomorphic horse wearing an elegant ballroom gown",
    
    "A male goat wearing a scholar’s robe",
    "A female goat wearing a soft woolen dress",
    
    "A male Anthropomorphic gorila wearing a suit and tie",
    "A female Anthropomorphic gorila wearing a chic blazer and skirt",
    
    "A male Anthropomorphic rooster wearing a pilot uniform",
    "A female Anthropomorphic rooster wearing a stylish flight attendant",
    
    "A male dog wearing a superhero costume",
    "A female dog wearing a detective trench coat",
    
    "A male fit pig dressed as a police officer",
    "A female fit pig wearing as a lawyer"
]


added_prompts = "Anthropomorphic, Nice looking, slender or muscular, wearing well-fitted clothing, anime style, white background"

for prompt in animals_with_clothing:
    enhanced_prompt = prompt + ", " + added_prompts
    try:
        image = pipe(prompt=enhanced_prompt, init_image=leading_image, strength=0.8).images[0]
    except TypeError:
        print("init_image argument not supported. Please check the updated StableDiffusion3Pipeline documentation.")
        image = pipe(prompt=enhanced_prompt).images[0]

    # Save the generated image
    image.save(fr"{prompt}.png")
