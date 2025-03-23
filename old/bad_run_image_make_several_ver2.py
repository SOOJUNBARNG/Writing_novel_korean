from diffusers import StableDiffusionImg2ImgPipeline
import torch
from PIL import Image

# Load Stable Diffusion Image-to-Image Pipeline
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-2-1")
pipe = pipe.to(device)

# Load leading image
leading_image = Image.open(r"D:\Writing_novel_korean\make_image\related_image\A male tiger wearing a detective coat.png").convert("RGB")  # Ensure it's in RGB mode

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


added_prompts = "Anthropomorphic, good looking, cute, not-realistic, delightful expressive face, standing upright, wearing well-fitted clothing, anime style, no background"


# Iterate through prompts and generate images
for prompt in animals_with_clothing:
    enhanced_prompt = prompt + ", " + added_prompts
    
    try:
        # Image-to-Image generation
        image = pipe(prompt=enhanced_prompt, image=leading_image, strength=0.5).images[0]
    except TypeError:
        print("Error: init_image or image argument not supported. Check pipeline version.")
        image = pipe(prompt=enhanced_prompt).images[0]  # Fallback to text-to-image

    # Save the generated image
    image.save(f"{prompt.replace(' ', '_')}.png")
