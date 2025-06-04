import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import json
import pandas as pd
from datasets import Dataset
from tqdm import tqdm
import re

# ===== Configuration =====
merged_models_folder = "./" 
subjects_to_evaluate = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]

test_base_folder = "MATH/test" 
results_csv = "evaluation_results.csv"
results_jsonl_folder = "evaluation_outputs"  # Folder to save detailed outputs (.jsonl)

use_bf16 = True
max_length = 512 
max_new_tokens = 384  
sample_ratio = 1.0
device = "cuda" if torch.cuda.is_available() else "cpu"

# ===== Helper: Extract predicted answer =====
def extract_answer(output_text):
    # Try to find boxed answer first
    match = re.search(r"\\boxed{([^}]*)}", output_text)
    if match:
        return match.group(1).strip()

    # Otherwise find the last number
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", output_text)
    if numbers:
        return numbers[-1].strip()

    return None

# ===== Load test datasets =====
def load_test_dataset(folder_path):
    all_data = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_data.append({
                    "problem": data["problem"],
                    "solution": data["solution"],
                })
    return Dataset.from_list(all_data)

# ===== Make output folder if needed =====
os.makedirs(results_jsonl_folder, exist_ok=True)

# ===== Evaluation loop =====
summary_results = []

for subject in subjects_to_evaluate:
    print(f"\nEvaluating subject: {subject}")

    model_path = os.path.join(merged_models_folder, f"merged_{subject}")
    test_folder = os.path.join(test_base_folder, subject)

    if not os.path.exists(model_path):
        print(f"Merged model not found: {model_path}")
        continue

    if not os.path.exists(test_folder):
        print(f"Test data not found: {test_folder}")
        continue

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if use_bf16 else torch.float16,
        trust_remote_code=True,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()

    model.generation_config.temperature = None
    model.generation_config.top_p = None

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Load test data
    test_dataset = load_test_dataset(test_folder)

    # Sample only 10% randomly
    test_dataset = test_dataset.train_test_split(test_size=sample_ratio, seed=42)["test"]

    correct = 0
    total = 0

    output_jsonl_path = os.path.join(results_jsonl_folder, f"{subject}_outputs.jsonl")
    with open(output_jsonl_path, "w", encoding="utf-8") as jsonl_file:

        for example in tqdm(test_dataset, desc=f"Evaluating {subject}"):
            prompt = f"Problem: {example['problem']} Solution:"
            inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=model.config.eos_token_id,
                )

            output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            predicted_answer = extract_answer(output_text)
            correct_answer = str(example["solution"]).strip()

            is_correct = predicted_answer == correct_answer
            if is_correct:
                correct += 1
            total += 1

            # Save each example as one JSON line
            output_entry = {
                "problem": example["problem"],
                "ground_truth": correct_answer,
                "model_output": output_text,
                "predicted_answer": predicted_answer,
                "correct": is_correct,
            }
            jsonl_file.write(json.dumps(output_entry, ensure_ascii=False) + "\n")

    accuracy = correct / total if total > 0 else 0
    print(f"Accuracy for {subject}: {accuracy:.4f}")

    summary_results.append({
        "subject": subject,
        "accuracy": accuracy,
        "total_evaluated_samples": total,
    })

# ===== Save summary results =====
df = pd.DataFrame(summary_results)
df.to_csv(results_csv, index=False)

print(f"\nEvaluation summary saved to {results_csv}")
print(f"Detailed outputs saved to {results_jsonl_folder}/ as .jsonl files")
