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

messages = [
    {
        "role": "system",
        "content": """당신은 소설속 인물입니다. 
                    개인의 경험을 장기간에 걸쳐서 담담히 서술하는 것이 당신의 일입니다 
                    부족한 부분은 자세히 기술을 해주면 좋겠습니다. 2시간 이상의 읽을만한 분량을 쓰는 것을 좋아합니다.
                    100000자 이상의 글을 쓰시오.
                    """
    },
    {
        "role": "user",
        "content": """
                    당신은 총격전의 한 가운데에서 시작합니다.
                    당신은 겁은 많으나 매우 뛰어난 사격 실력을 가지고 있습니다.
                    서울 한 복판에서 러시아 그리고 미국 군인을 죽이면서 당신은 총 앞에서 회상을 시작합니다.

                    시작은 2035년 4월 입니다. 언제나처럼 전복에 미역줄기를 넣으며 전화를 받습니다.
                    전복은 최근에 약재로서 인정을 받아서 매우 고가의 물품입니다.
                    전화를 받으며 당신은 3년생의 전복을 주문하고, 본인이 키우는 4년생의 전복을 납품할 준비를 합니다.

                    다만 운이 않좋게도 강아지가 전원선을 끊어버리고, 전복이 죽어버립니다.
                    급한 당신은 수협에 가서 대출상담을 진행하나, 대출이 불가능하다는 답변을 받습니다.
                    다행히 4년생의 전복의 가격을 인정받아서 보험금은 지급이 되었습니다.

                    다만 3개월후에는 3년생 전복의 가격을 지붏해야 합니다.
                    당신은 고민하다가, 이번에 새로 생긴 전복 선물 거래를 이용하여 금융에 도전합니다.

                    근데 아뿔사 돈을 너무 벌어서, 깡패의 타겟이 되어버립니다.
                    깡패와의 항쟁에서 당신은 깡패를 죽여버리고 이를 감춥니다.

                    허나, 속속들이 좁혀오는 관련자들...
                    당신은 사업을 정리하고 도망가기로 결정합니다.

                    이 이후에 이야기도 흥미진진하게 길게 작성해주십시요.
                    """,
    }
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
token_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")

def generate_long_text(prompt, total_length=100000, chunk_size=1024):
    generated_text = prompt
    for _ in range(total_length // chunk_size):
        inputs = tokenizer(generated_text, return_tensors="pt", truncation=True)
        output = model.generate(**inputs, max_length=len(inputs["input_ids"][0]) + chunk_size, do_sample=True, temperature=0.7)
        new_text = tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Append only new text
        generated_text = new_text[len(generated_text):]

        print(generated_text)  # Print in chunks
    return generated_text

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
token_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(device)

def generate_long_text(model, tokenizer, token_ids, total_tokens=50000, chunk_size=1024):
    generated_text = ""
    
    with open("output_abaloni.txt", "w", encoding="utf-8") as f:
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
            
            print(new_text[:200])  # 처음 200자만 미리보기 출력
            generated_text += new_text
            
    return generated_text

# 긴 글 생성
long_text = generate_long_text(model, tokenizer, token_ids)