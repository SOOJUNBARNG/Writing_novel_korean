# https://soroban.highreso.jp/article/article-060
# https://huggingface.co/elyza/Llama-3-ELYZA-JP-8B-GGUF/tree/main
# https://huggingface.co/learn
# https://zenn.dev/timoneko/books/8a9cab9c5caded/viewer/d01d3e
# https://note.com/elyza/n/n360b6084fdbd

import torch

import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model_name = "microsoft/phi-4"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
)
model.eval()

messages = [
    {
        "role": "system",
        "content": """당신은소설가입니다. 매우 잔혹한 현실속에서 피어나는 사랑과 재치가 가득한 소설을 쓰는것을 좋아하는 소설가입니다. 
                    이 소설가는 또한 매우 빠르게 글을 쓰는 것을 좋아합니다. 이 소설가는 또한 매우 빠르게 글을 쓰는 것을 좋아합니다. """
    },
    {
        "role": "user",
        "content": """
                    목포를 배경으로 하는 소설의 시놉시스를 작성해주세요.
                    배경은 일본의 지배를 여전히 받는 2030년의 대한민국 전라도/경상도이다. 
                    다만, 서울과 강원, 충청도는 일본으로 부터 1990년대에 독립을 하였다.
                    남성의 이름은 장태훈 키 175cm, 몸무게 70kg, 나이 30세
                    남성의 가족은 완도의 김을 일본의 백화점에 납품하는 일을 하고 있다.
                    남성은 동경에서 온 미모의 재일교포가 백화점 남품건으로 이야기를 진행하러 왔을때, 강간을 해버리는 장면으로 시작한다.
                    재일교포(유이)는 겁에 질려서 남성을 칼로 찌르나, 남성은 제대로 된 상처를 입지 않는다.
                    """,
    }
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
token_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")

with torch.no_grad():
    output_ids = model.generate(
        token_ids.to(model.device),
        max_new_tokens=1200,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
output = tokenizer.decode(output_ids.tolist()[0][token_ids.size(1) :], skip_special_tokens=True)
output.to_file("output.txt")