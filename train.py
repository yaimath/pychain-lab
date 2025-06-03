import os
import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from peft import LoraConfig, get_peft_model, PeftModel
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

# ───── 설정 ─────
MODEL_NAME = "GSAI-ML/LLaDA-8B-Instruct"
ADAPTER_PATH = "models/llada_multi_pass_epoch0"
MASK_ID = 126336
EOS_ID = 126081
MAX_LEN = 64
BATCH_SIZE = 2
NUM_EPOCHS = 100
LR = 1e-8
WEIGHT_DECAY = 0.1

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ───── 템플릿 적용 ─────
def apply_llada_template(prompt, rationales, answer):
    user_prefix = "<start_id>user<end_id>\n"
    assistant_prefix = "<start_id>assistant<end_id>\n"
    input_sequence = [f"<BOS>{user_prefix}{prompt}<eot_id>"]
    for rationale in rationales:
        input_sequence.append(f"{assistant_prefix}{rationale}<eot_id>")
    if answer:
        input_sequence.append(f"{assistant_prefix}{answer}<EOS>")
    return "".join(input_sequence)

# ───── 데이터셋 ─────
class LLaDADataset(Dataset):
    def __init__(self, path, tokenizer):
        self.samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                prompt = data["prompt"]
                rationales = data["rationales"]
                answer = data["answer"]

                for i in range(len(rationales) + 1):
                    input_r = rationales[:i]
                    target_r = rationales[i] if i < len(rationales) else answer

                    input_str = apply_llada_template(prompt, input_r, "")
                    target_str = f"<start_id>assistant<end_id>\n{target_r}<eot_id>"

                    input_ids = tokenizer(input_str, truncation=True, max_length=MAX_LEN)["input_ids"]
                    target_ids = tokenizer(target_str, truncation=True, max_length=MAX_LEN)["input_ids"]

                    input_ids = torch.tensor(input_ids + [EOS_ID] * (MAX_LEN - len(input_ids)), dtype=torch.long)
                    target_ids = torch.tensor(target_ids + [EOS_ID] * (MAX_LEN - len(target_ids)), dtype=torch.long)

                    self.samples.append((input_ids, target_ids))

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]

def collate_fn(batch):
    input_ids, target_ids = zip(*batch)
    return torch.stack(input_ids), torch.stack(target_ids)

# ───── 마스킹 함수 ─────
def forward_process(batch, mask_id=MASK_ID):
    b, l = batch.shape
    t = torch.randint(1, l + 1, (b,), device=batch.device)
    p_mask = torch.arange(l, device=batch.device).expand(b, l) < t.unsqueeze(1)
    mask = torch.zeros_like(p_mask, dtype=torch.bool)
    for i in range(b):
        idx = torch.randperm(l)[:t[i]]
        mask[i, idx] = True
    noisy = torch.where(mask, mask_id, batch)
    return noisy, mask

# ───── 학습 루프 ─────
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    # base 모델 로딩
    base_model = AutoModel.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    ).to(device)
    
    for param in base_model.parameters():
        param.requires_grad = False

    # ───── LoRA 설정 ─────
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="FEATURE_EXTRACTION"
    )

    # ───── LoRA 적용 ─────
    # ───── LoRA 적용 ─────
    if os.path.exists(ADAPTER_PATH):
        print(f"🔄 Adapter found at '{ADAPTER_PATH}', loading...")
        model = PeftModel.from_pretrained(
            base_model,
            ADAPTER_PATH,
            torch_dtype=torch.bfloat16,
            is_trainable=True  # ✅ 중요! LoRA 파라미터 학습 가능하게 설정
        )
        start_epoch = int(ADAPTER_PATH.split("epoch")[-1]) + 1
    else:
        print(f"🆕 No adapter found at '{ADAPTER_PATH}', initializing from scratch.")
        model = get_peft_model(base_model, lora_config)
        start_epoch = 1

    model = model.to(device)  # ✅ 기존 모델 유지한 채 .to(device)만 수행
    model.train()



    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    dataset = LLaDADataset("datasets/train.jsonl", tokenizer)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    for epoch in range(NUM_EPOCHS):
        total_loss = 0.0
        for step, (input_ids, target_ids) in enumerate(tqdm(dataloader, desc=f"Epoch {start_epoch+epoch}")):
            input_ids, target_ids = input_ids.to(device), target_ids.to(device)

            # 마스킹 적용
            noisy_input, mask = forward_process(input_ids)

            # 모델 예측
            logits = model(input_ids=noisy_input).logits

            # 마스크 위치에 대한 loss만 계산
            loss = F.cross_entropy(
                logits[mask], input_ids[mask], reduction="mean"
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if (step + 1) % 10 == 0:
                print(f"[Epoch {start_epoch+epoch}] Step {step+1} | Loss: {loss.item():.4f}")

        save_path = f"modelss/llada_multi_pass_epoch{start_epoch+epoch}"
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"✅ 모델 저장됨: {save_path} | 평균 Loss: {total_loss / len(dataloader):.4f}")

if __name__ == "__main__":
    main()
