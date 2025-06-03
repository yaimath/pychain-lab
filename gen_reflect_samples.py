import argparse
import re
from pathlib import Path
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
from datasets import Dataset, DatasetDict, load_dataset
from transformers import BitsAndBytesConfig

from utils import DatasetFilter

dataset_path = 'nvidia/OpenMathReasoning'
ckpt_path = './models/OpenMathReasoning'
num_samples = 4_000

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

@torch.no_grad
def main():
    train_ds = load_dataset(dataset_path, split='cot', streaming=True).shuffle(seed=43).take(num_samples)
    train_ds = train_ds.with_format('torch')

    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_storage=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        ckpt_path,
        attn_implementation='flash_attention_2',
        quantization_config=bnb_config, 
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
    )

    pipe = pipeline(
        'text-generation', 
        model=model, 
        tokenizer=tokenizer,
        max_length=508,
        truncation=True,
    )

    dataset = []

    for sample in tqdm(train_ds, total=num_samples, mininterval=60, maxinterval=60):
        problem = sample['problem']
        correct_output = sample['generated_solution']
        correct_answer = extract_answer(correct_output)
        output_text = pipe(problem)[0]['generated_text']
        predicted_answer = extract_answer(output_text)
        is_correct = predicted_answer == correct_answer
        dataset.append({
            "problem": problem,
            "correct_output": correct_output,
            "ground_truth": correct_answer,
            "model_output": output_text,
            "predicted_answer": predicted_answer,
            "correct": is_correct,
        })
    dataset = Dataset.from_list(dataset)
    dataset.save_to_disk('./datasets/gen-answer-OpenMathReasoning')

if __name__ == '__main__':
    main()
