import torch

from diffusers import StableDiffusion3Pipeline
from PIL import Image

# Load the model
pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-medium", torch_dtype=torch.bfloat16)
pipe = pipe.to("cuda")

# Load your leading image (starting image) - modify this path to your image
leading_image_path = r"D:\Writing_novel_korean\make_image\related_image\horseman.webp"
leading_image = Image.open(leading_image_path).convert("RGB")

# animals_with_clothing = [
#     "A handsome male but a little like rat wearing a business suit",
#     "A beautiful female but a little like rat wearing an elegant dress",
    
#     "A handsome male  a bit like  ox wearing a farmer's outfit",
#     "A beautiful female  a bit like  ox wearing a oriental outfit",
    
#     "A handsome male  a bit like  tiger wearing a detective coat",
#     "A beautiful female  a bit like  tiger wearing a doctor outfit",
    
#     "A handsome male  a bit like  rabbit dressed in knight general suit",
#     "A beautiful female  a bit like  rabbit wearing like a florist",
    
#     "A handsome male  a bit like  dragon in royal Korean robes",
#     "A beautiful female career woman a bit like dragon in a business suit",
    
#     "A handsome male  a bit like  snake wearing a bow tie",
#     "A beautiful female  a bit like  snake wearing a stylish gown",
    
#     "A handsome male a bit like horse wearing a royal robe",
#     "A beautiful female a bit like horse wearing an elegant ballroom gown",
    
#     "A handsome male a bit like goat wearing a scholar’s robe",
#     "A beautiful female a bit like goat wearing a soft woolen dress",
    
#     "A handsome male like monkey wearing a suit and tie",
#     "A beautiful female like monkey wearing a chic blazer and skirt",
    
#     "A handsome male but a little like rooster wearing a pilot uniform",
#     "A beautiful female but a little like hen wearing a stylish flight attendant",
    
#     "A handsome male a bit like Labrador Retriever wearing a health trainer outfit",
#     "A beautiful female a bit like Poodle wearing a detective trench coat",
    
#     "A handsome male mixed with pig dressed as a police officer",
#     "A beautiful female mixed with pig wearing as a lawyer"
# ]


# added_prompts = "Extremely Anthropomorphic, Humanized, Very Nice looking, wearing well-fitted clothing, anime style, white background"

related_image_basic = [
    "ビジネス戦略関連",
    "経営戦略の全体像",
    "中長期ビジョンと成長戦略",
    "市場分析と競争環境",
    "新規事業開発のロードマップ",
    "既存事業の成長戦略",
    "財務・KPI管理",
    "売上・利益の推移と分析",
    "事業別収益モデル",
    "コスト削減と効率化施策",
    "財務指標とKPI管理",
    "投資対効果（ROI）分析",
    "組織・人材戦略",
    "組織体制とガバナンス",
    "人材育成とリーダーシップ開発",
    "働き方改革と生産性向上",
    "ダイバーシティ＆インクルージョン",
    "従業員エンゲージメント向上策",
    "DX・テクノロジー活用",
    "デジタル変革（DX）戦略",
    "AI・データ活用による業務効率化",
    "システム刷新とITインフラの最適化",
    "顧客データ分析とマーケティング活用",
    "生成AI活用による業務改善",
    "リスク管理・コンプライアンス",
    "事業リスクと対応策",
    "BCP（事業継続計画）と危機管理",
    "ガバナンス強化と内部統制",
    "法規制対応とコンプライアンス",
    "サイバーセキュリティ対策",
    "イノベーション・企業文化",
    "イノベーション創出の仕組み",
    "社内アイデアの活用と新規事業創出",
    "オープンイノベーション戦略",
    "企業文化とビジョンの浸透",
    "サステナビリティとESG経営"
]

added_prompts = "This is a ppt slide for medical frontier company in japan, Extremely formal, made by consultant, No text,white background"

for prompt in related_image_basic:
    enhanced_prompt = prompt + ", " + added_prompts
    try:
        image = pipe(prompt=enhanced_prompt, init_image=leading_image, strength=0.8).images[0]
    except TypeError:
        print("init_image argument not supported. Please check the updated StableDiffusion3Pipeline documentation.")
        image = pipe(prompt=enhanced_prompt).images[0]

    # Save the generated image
    image.save(fr"{prompt}.png")
