import torch

from diffusers import StableDiffusion3Pipeline
from PIL import Image

# Load the model
pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-medium", torch_dtype=torch.bfloat16)
pipe = pipe.to("cuda")

# Load your leading image (starting image) - modify this path to your image
leading_image_path = r"D:\Writing_novel_korean\make_image\related_image\horseman.webp"
leading_image = Image.open(leading_image_path).convert("RGB")

animals_with_clothing = {
    "rat_male": "A handsome man in a sharp business suit with subtly angular features and a clever, rat-like charm",
    "rat_female": "A beautiful woman in an elegant dress with delicate features and a hint of mischievous, rat-like allure",
    
    "ox_male": "A ruggedly handsome man in a farmer's outfit with a strong build and grounded, ox-like presence",
    "ox_female": "A graceful woman in a traditional oriental outfit with calm, dignified features and ox-like strength",
    
    "tiger_male": "A sharp-eyed man in a detective coat with a fierce gaze and a bold, tiger-like charisma",
    "tiger_female": "A confident woman in a white doctor’s coat, with striking eyes and a graceful, tiger-like intensity",
    
    "rabbit_male": "A noble man dressed in a knight general's armor with a gentle but alert expression, evoking rabbit-like agility",
    "rabbit_female": "A gentle woman in a floral dress with soft features and a warm, rabbit-like demeanor",
    
    "dragon_male": "A regal man in traditional Korean royal robes with a powerful presence and subtle dragon-like majesty",
    "dragon_female": "A commanding career woman in a sleek business suit with confident posture and a refined, dragon-like energy",
    
    "snake_male": "A refined man in formalwear with a slim frame and quiet elegance, exuding snake-like charm",
    "snake_female": "A glamorous woman in a flowing gown with sharp eyes and subtle, snake-like mystery",
    
    "horse_male": "A dignified man in a royal robe with a proud stance and a bold, horse-like energy",
    "horse_female": "A radiant woman in a ballroom gown with flowing hair and a graceful, horse-like spirit",
    
    "goat_male": "A scholarly man in traditional robes with wise eyes and a calm, goat-like gentleness",
    "goat_female": "A soft-spoken woman in a cozy woolen dress with serene features and goat-like warmth",
    
    "monkey_male": "A clever man in a modern suit and tie with lively eyes and playful, monkey-like intelligence",
    "monkey_female": "A stylish woman in a chic blazer and skirt with bright features and monkey-like wit",
    
    "rooster_male": "A charismatic man in a pilot uniform with sharp eyes and confident, rooster-like poise",
    "rooster_female": "A fashionable woman in a flight attendant uniform with polished style and rooster-like flair",
    
    "dog_male": "A fit man in a sporty health trainer outfit with loyal eyes and Labrador-like energy",
    "dog_female": "A classy woman in a detective trench coat with an elegant silhouette and clever, poodle-like personality",
    
    "pig_male": "A strong man in a police uniform with bold features and dependable, hog-like steadiness",
    "pig_female": "A confident woman in a lawyer's suit with sharp wit and a composed, pig-like charm"
}

version = "v1"

added_prompts = "Loish and Studio Ghibli, pastel color palette, soft lighting, digital painting, 2D illustration, Very Nice looking, wearing well-fitted clothing, anime style, white background"

# business_topics = [
#     "Business Strategy",
#     "Overview of Corporate Strategy",
#     "Mid-to-Long-Term Vision and Growth Strategy",
#     "Market Analysis and Competitive Environment",
#     "Roadmap for New Business Development",
#     "Growth Strategy for Existing Businesses",
    
#     "Finance & KPI Management",
#     "Revenue and Profit Trends & Analysis",
#     "Business-Specific Revenue Models",
#     "Cost Reduction and Efficiency Measures",
#     "Financial Metrics and KPI Management",
#     "Return on Investment (ROI) Analysis",
    
#     "Organization & Talent Strategy",
#     "Organizational Structure and Governance",
#     "Talent Development and Leadership Training",
#     "Workstyle Reform and Productivity Improvement",
#     "Diversity & Inclusion",
#     "Employee Engagement Enhancement",
    
#     "DX & Technology Utilization",
#     "Digital Transformation (DX) Strategy",
#     "AI & Data Utilization for Operational Efficiency",
#     "System Modernization and IT Infrastructure Optimization",
#     "Customer Data Analysis and Marketing Applications",
#     "Business Improvement through Generative AI",
    
#     "Risk Management & Compliance",
#     "Business Risks and Countermeasures",
#     "Business Continuity Planning (BCP) and Crisis Management",
#     "Strengthening Governance and Internal Controls",
#     "Legal Compliance and Regulatory Adherence",
#     "Cybersecurity Measures",
    
#     "Innovation & Corporate Culture",
#     "Mechanisms for Innovation Creation",
#     "Utilizing Internal Ideas and Creating New Businesses",
#     "Open Innovation Strategy",
#     "Corporate Culture and Vision Penetration",
    
#     "Sustainability & ESG Management"
# ]
# added_prompts = "This is a ppt slide for medical consulting company, Extremely formal, made by consultant, No text,white background"

for i, (key, prompt) in enumerate(animals_with_clothing.items()):
    enhanced_prompt = prompt + ", " + added_prompts
    
    try:
        # If 'leading_image' is not defined, remove init_image from the function call:
        image = pipe(prompt=enhanced_prompt, init_image=None, strength=0.8).images[0]  # Adjust as needed
    except TypeError:
        print("init_image argument not supported. Please check the updated StableDiffusion3Pipeline documentation.")
        image = pipe(prompt=enhanced_prompt).images[0]  # For pure prompt generation

    # Save the generated image
    image.save(fr"{key}_{version}.png")
