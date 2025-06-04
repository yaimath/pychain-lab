import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import json
import pandas as pd
from datasets import Dataset
from tqdm import tqdm
import re

# ===== Configuration =====
merged_models_folder = "./models/"  # Folder containing your merged models
model_paths = [
    #"./models/gen-answer-OpenMathReasoning"
    "./models/gen-answer-OpenMathReasoning"
    #'nvidia/OpenMath-Nemotron-7B'
]
subjects_to_evaluate = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]

test_base_folder = "datasets/MATH/test"  # Path to your MATH test dataset
#results_csv = "evaluation_results-new-contextual.csv"
results_csv = "evaluation_results-new.csv"
results_jsonl_folder = "evaluation_outputs-new"  # Folder to save detailed outputs (.jsonl)

use_bf16 = True
max_length = 1024 + 512 + 256  # Truncate inputs to this length
max_new_tokens = 1024 + 512 + 256  # Allow enough room for full solution generation
sample_ratio = 0.2  # Evaluate only 10% of test set
device = "cuda" if torch.cuda.is_available() else "cpu"

# ===== Helper: Extract predicted answer =====
def extract_answer(output_text):
    # Try to find boxed answer first
    match = re.search(r"\\boxed\{([\s\S]*?)\}\$", output_text)
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

# Load model
model_path = model_paths[0]

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

for subject in subjects_to_evaluate:
    print(f"Evaluating subject: {subject}")

    test_folder = os.path.join(test_base_folder, subject)

    if not os.path.exists(model_path):
        print(f"Merged model not found: {model_path}")
        continue

    if not os.path.exists(test_folder):
        print(f"Test data not found: {test_folder}")
        continue

    # Load test data
    test_dataset = load_test_dataset(test_folder)

    # Sample only 10% randomly
    test_dataset = test_dataset.train_test_split(test_size=sample_ratio, seed=42)["test"]

    correct = 0
    total = 0

    output_jsonl_path = os.path.join(results_jsonl_folder, f"{subject}_outputs.jsonl")
    with open(output_jsonl_path, "w", encoding="utf-8") as jsonl_file:

        for example in tqdm(test_dataset, desc=f"Evaluating {subject}"):
            text_prompt = f"Problem: {example['problem']} Previous Output: "
            inputs = tokenizer(text_prompt, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=model.config.eos_token_id,
                )

            output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            predicted_answer = extract_answer(output_text)
            correct_output = str(example["solution"]).strip()
            correct_answer = extract_answer(correct_output)

            is_correct = predicted_answer == correct_answer
            if is_correct:
                correct += 1
            total += 1

            # Save each example as one JSON line
            output_entry = {
                "problem": example["problem"],
                "correct_output": correct_output,
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

print(f"Evaluation summary saved to {results_csv}")
print(f"Detailed outputs saved to {results_jsonl_folder}/ as .jsonl files")
