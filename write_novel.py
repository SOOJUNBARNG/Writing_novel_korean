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
    torch_dtype="auto",
    device_map="auto",
)
model.eval()

def generate_long_text(model, tokenizer, total_tokens=50000, chunk_size=15000):
    generated_text = ""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 기존 파일 내용 불러오기
    with open("output.txt", "r", encoding="utf-8") as file:
        text = file.read().strip()

    messages = [
        {
            "role": "system",
            "content": """당신은 소설가입니다. 매우 잔혹한 현실 속에서 피어나는 사랑과 재치가 가득한 소설을 쓰는 것을 좋아하는 소설가입니다. 
                        이 소설가는 또한 매우 빠르게 글을 쓰는 것을 좋아합니다. 100000자 이상의 글을 쓰는 것을 좋아합니다."""
        },
        {
            "role": "user",
            "content": text
        }
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    token_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(device)

    with open("output.txt", "w", encoding="utf-8") as f:
        for _ in range(total_tokens // chunk_size):
            with torch.no_grad():
                output_ids = model.generate(
                    token_ids,
                    max_new_tokens=chunk_size,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                )

            new_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            token_ids = output_ids[:, -chunk_size:].detach()  # 마지막 chunk만 유지

            # 텍스트 저장
            f.write(new_text + "\n")
            f.flush()

            print(new_text[:200])  # 처음 200자 미리보기 출력
            generated_text += new_text

    return generated_text

# 긴 글 생성
long_text = generate_long_text(model, tokenizer)

