import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType

# === Configuration ===
model_name = "deepseek-ai/deepseek-coder-6.7b-instruct"
dataset_path = "openmathreasoning/train_toolcall_model_instruct.jsonl"
output_dir = "./deepseek-code-toolcall-lora"
max_length = 1024
use_fp16 = torch.cuda.is_available()

# === Load dataset ===
dataset = load_dataset("json", data_files={"train": dataset_path})["train"]

# === Load tokenizer and model ===
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.float16 if use_fp16 else torch.float32)

# === Apply LoRA configuration ===
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # adjust based on model arch
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, peft_config)

# === Tokenization function ===
def tokenize(example):
    prompt = example["input"].strip()
    output = example["output"].strip()
    full_text = prompt + "\n\n" + output
    return tokenizer(full_text, truncation=True, max_length=max_length, padding="max_length")

# === Apply tokenization ===
tokenized_dataset = dataset.map(tokenize, remove_columns=dataset.column_names)

# === Training arguments ===
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=5e-5,
    fp16=use_fp16,
    logging_steps=10,
    save_steps=1000,
    save_total_limit=2,
    warmup_steps=100,
    weight_decay=0.01,
    lr_scheduler_type="linear",
    report_to="none"
)

# === Data collator ===
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# === Trainer ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# === Start training ===
trainer.train()

# === Save only the LoRA adapter ===
model.save_pretrained(output_dir)
