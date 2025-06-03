import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import LRScheduler, LinearLR, CosineAnnealingLR
from torch.utils.data import DataLoader

import transformers
from datasets import Dataset, DatasetDict, load_dataset, concatenate_datasets
from transformers import AutoModelForSequenceClassification, AutoTokenizer, SchedulerType
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments, Trainer

from train_config import RouterTrainConfig
from utils import TokenizeMapWrapper, compute_accuarcy

def main():
    torch.set_float32_matmul_precision('high')
    config = RouterTrainConfig()

    parser = argparse.ArgumentParser()
    parser.add_argument('--sector', dest='sector', action='store', type=str, default=None)
    args = parser.parse_args()

    if args.sector is not None:
        sector = args.sector
        dataset_root = str(Path(config.dataset_root).parent / sector)
    else:
        sector = Path(config.dataset_root).name
        dataset_root = config.dataset_root
    dataset_root = Path(dataset_root)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    ds_lt = []
    label_lt = []
    for path in dataset_root.glob('*'):
        ds = load_dataset(str(path))['train']
        ds = (
            ds.select_columns(['problem'])
              .rename_column('problem', 'text')
              .add_column('label', [path.name] * len(ds))
        )
        label_lt.append(path.name)
        ds_lt.append(ds)
    ds = concatenate_datasets(ds_lt).shuffle(seed=42)
    
    tokenizer = AutoTokenizer.from_pretrained(config.checkpoint)
    id2label = {i: id for i, id in enumerate(label_lt)}
    label2id = {id: i for i, id in enumerate(label_lt)}
    tokenizer_map = TokenizeMapWrapper(
        tokenizer,
        feature='text',
        option={
            'max_length': config.seq_max_len,
            'truncation': True,
            'padding': 'max_length',
        }
    )
    ds = ds.map(tokenizer_map)
    ds = ds.map(lambda sample: {k: (v if k != 'label' else label2id[v]) for k, v in sample.items()})

    ds_dict = ds.train_test_split(test_size=0.1)
    train_ds = ds_dict['train']
    test_ds = ds_dict['test']

    model = AutoModelForSequenceClassification.from_pretrained(
        config.checkpoint,
        num_labels=len(label_lt),
        id2label=id2label,
        label2id=label2id,
        device_map=device,
    )
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
    )
    training_args = TrainingArguments(
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        gradient_checkpointing=config.gradient_checkpoint,
        num_train_epochs=config.num_epochs,
        weight_decay=config.weight_decay,

        logging_first_step=0,
        logging_steps=1,
        report_to='wandb',

        do_eval=True,
        eval_strategy='epoch',

        output_dir='./out-router',
        save_strategy='best',
        metric_for_best_model='eval_loss',
        
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_accuarcy,
    )

    trainer.train()

if __name__ == '__main__':
    main()
