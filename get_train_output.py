import argparse
from pathlib import Path
from math import floor

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import LRScheduler, LinearLR, CosineAnnealingLR
from torch.utils.data import DataLoader

import transformers
from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, SchedulerType
from transformers import BitsAndBytesConfig
from peft import LoraConfig, TaskType, get_peft_model
from trl import SFTConfig, SFTTrainer

from train_config import TrainConfig
from utils import PromptFormatter, DataLogCallback

model_paths = [
    "models/OpenMathReasoning"
]

def main():
    config = TrainConfig()

    parser = argparse.ArgumentParser()
    parser.add_argument('--sector', dest='sector', action='store', type=str, default=None)
    args = parser.parse_args()

    if args.sector is not None:
        sector = args.sector
        dataset_path = str(Path(config.dataset_path).parent / sector)
    else:
        sector = Path(config.dataset_path).name
        dataset_path = config.dataset_path

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

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

    train_ds = load_dataset(dataset_path, split='cot', streaming=config.streaming).shuffle(seed=42)
    print(train_ds)
    prompt_formatter = PromptFormatter(
        instruction_col_name='problem',
        response_col_name='generated_solution',
        tokenizer=tokenizer,
        system_instruction=''
    )
    
    if config.streaming:
        num_samples = config.num_samples
    else:
        num_samples = len(train_ds)

    
    
if __name__ == '__main__':
    main()
