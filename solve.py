#!/usr/bin/env python3
import torch
import json
import re
import glob
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel
from generate import generate

def load_model(adapter_path, base_model="GSAI-ML/LLaDA-8B-Instruct", device="cuda"):
    base = AutoModel.from_pretrained(
        base_model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    ).to(device).eval()

    model = PeftModel.from_pretrained(
        base,
        adapter_path,
        torch_dtype=torch.bfloat16
    ).to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    return model, tokenizer

def extract_answer_from_text(text: str) -> str:
    """
    '\\boxed{...}' 혹은 '\\boxed{...}$' 안의 내용을 추출,
    없으면 마지막 숫자를 추출합니다.
    """
    # 1) \boxed{...}$ 우선
    m = re.search(r'\\boxed\{((?:[^{}]|\{[^{}]*\})+)\}\$', text)
    if m:
        return m.group(1).strip()
    # 2) 일반적인 \boxed{...}
    m = re.search(r'\\boxed\{((?:[^{}]|\{[^{}]*\})+)\}', text)
    if m:
        return m.group(1).strip()
    # 3) fallback: 마지막 숫자
    nums = re.findall(r'(-?\d+)', text)
    return nums[-1] if nums else ""

def infer_from_jsonl(jsonl_path="datasets/test.jsonl",
                    output_path="infer_outputsss1.jsonl",
                    adapter_path="modelss/llada_multi_pass_epoch1"):
    device = "cuda"
    model, tokenizer = load_model(adapter_path, device=device)

    block_length = 32
    gen_length = 128
    steps = 128

    # 전체 예제 수 세기
    total = sum(1 for _ in open(jsonl_path, 'r', encoding='utf-8'))

    with open(jsonl_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line in tqdm(fin, total=total, desc="Inference"):
            example = json.loads(line)
            prompt_text = example["prompt"]
            gold_answer = example.get("answer", "").strip()

            # 입력 준비
            messages = [{"role": "user", "content": prompt_text}]
            prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            input_ids = tokenizer(prompt)["input_ids"]
            input_ids = torch.tensor(input_ids).unsqueeze(0).to(device)

            # 생성
            output_ids = generate(
                model,
                input_ids,
                steps=steps,
                gen_length=gen_length,
                block_length=block_length,
                temperature=0.0,
                cfg_scale=0.0,
                remasking="low_confidence"
            )
            output_text = tokenizer.batch_decode(
                output_ids[:, input_ids.shape[1]:],
                skip_special_tokens=True
            )[0].strip()

            # 모델이 낸 정답 추출
            model_answer = extract_answer_from_text(output_text)

            # 채점
            correct = (model_answer == gold_answer)

            # 결과 저장
            result = {
                "prompt": prompt_text,
                "gold_answer": gold_answer,
                "model_answer": model_answer,
                "raw_output": output_text,
                "correct": correct,
                "metadata": example.get("metadata", {})
            }
            fout.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"\n→ Done. Results written to {output_path}")

if __name__ == "__main__":
    infer_from_jsonl()
