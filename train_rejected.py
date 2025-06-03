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

from train_config import ContextualTrainConfig
from utils import ContextualPromptFormatter, DataLogCallback, DatasetFilter

def main():
    config = ContextualTrainConfig()

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

    tokenizer = AutoTokenizer.from_pretrained(config.checkpoint, padding_side='left')
    tokenizer.padding_side = 'left'
    #if tokenizer.pad_token is None:
    #    tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_storage=torch.bfloat16,
    )
    #model = AutoModelForCausalLM.from_pretrained(config.checkpoint, attn_implementation='flash_attention_2')
    model = AutoModelForCausalLM.from_pretrained(
        config.checkpoint, 
        attn_implementation='flash_attention_2',
        quantization_config=bnb_config, 
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
    ).to(device)
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=config.lora_hidden_dim,
        lora_alpha=32,
        lora_dropout=0.1,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    model.enable_input_require_grads()
    model.tokenizer = tokenizer

    train_ds = Dataset.load_from_disk(dataset_path).shuffle(seed=42)
    ds_filter = DatasetFilter(
        correct=lambda sample: sample is False,
    )
    train_ds = train_ds.filter(ds_filter)

    print(train_ds)
    prompt_formatter = ContextualPromptFormatter(
        instruction_col_name='problem',
        wrong_answer_col_name='model_output',
        correct_output_col_name='correct_output',
        tokenizer=tokenizer,
        system_instruction="The problem is given after 'Problem: ' and your previous wrong answer is given after 'Previous Answer'. Reflect the way to improve your skill by critical analysis. Answer with better reasoning.",
    )
    
    if config.streaming:
        num_samples = config.num_samples
    else:
        num_samples = len(train_ds)

    training_args = SFTConfig(
        max_seq_length=config.max_seq_length,
        max_steps=int(floor(num_samples // config.batch_size)),
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        gradient_checkpointing=config.gradient_checkpoint,
        num_train_epochs=config.num_epochs,
        lr_scheduler_type='constant',

        logging_first_step=0,
        logging_steps=1,
        report_to='wandb',

        output_dir='./out',
        save_strategy='best',
        metric_for_best_model='eval_loss',
    )

    trainer = SFTTrainer(
        model,
        training_args,
        train_dataset=train_ds,
        callbacks=[DataLogCallback()],
        formatting_func=prompt_formatter,
    )

    trainer.train()
    trainer.save_model(f'models/{sector}')

if __name__ == '__main__':
    main()
