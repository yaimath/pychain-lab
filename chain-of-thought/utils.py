import inspect
from typing import Optional, Callable

import numpy as np
import pandas as pd

import torch
from transformers import AutoTokenizer, TrainerCallback

class PromptFormatter:
    def __init__(
            self, 
            instruction_col_name, 
            response_col_name, 
            tokenizer: AutoTokenizer,
            system_instruction: Optional[str]=None, 
        ):
        if system_instruction is not None:
            self.system_instruction = system_instruction
        else:
            self.system_instruction = None
        self.instruction_col_name = instruction_col_name
        self.response_col_name = response_col_name
        self.tokenizer = tokenizer

    def __call__(self, example):
        prompt = [
            {'role': 'user', 'content': "Problem: " + example[self.instruction_col_name]},
            {'role': 'assistant', 'content': " Solution: " + example[self.response_col_name]}
        ]
        if self.system_instruction is not None:
            prompt.append({'role': 'system', 'content': self.system_instruction})
        return self.tokenizer.apply_chat_template(prompt, tokenize=False)

class PromptFormatter:
    def __init__(
            self, 
            instruction_col_name, 
            response_col_name, 
            tokenizer: AutoTokenizer,
            system_instruction: Optional[str]=None, 
        ):
        if system_instruction is not None:
            self.system_instruction = system_instruction
        else:
            self.system_instruction = None
        self.instruction_col_name = instruction_col_name
        self.response_col_name = response_col_name
        self.tokenizer = tokenizer

    def __call__(self, example):
        prompt = [
            {'role': 'user', 'content': "Problem: " + example[self.instruction_col_name]},
            {'role': 'assistant', 'content': " Solution: " + example[self.response_col_name]}
        ]
        if self.system_instruction is not None:
            prompt.append({'role': 'system', 'content': self.system_instruction})
        return self.tokenizer.apply_chat_template(prompt, tokenize=False)

class ContextualPromptFormatter:
    def __init__(
            self, 
            instruction_col_name, 
            wrong_answer_col_name,
            correct_output_col_name,
            tokenizer: AutoTokenizer,
            system_instruction: Optional[str]=None, 
        ):
        if system_instruction is not None:
            self.system_instruction = system_instruction
        else:
            self.system_instruction = None
        self.instruction_col_name = instruction_col_name
        self.wrong_answer_col_name = wrong_answer_col_name
        self.correct_output_col_name = correct_output_col_name
        self.tokenizer = tokenizer

    def __call__(self, example):
        prompt = [
            {
                'role': 'user', 
                'content': ("Problem: " 
                            + example[self.wrong_answer_col_name]
                            + " Previous Output: "
                        )
            },
            {
                'role': 'assistant', 
                'content': (example[self.wrong_answer_col_name]
                            + " Better Answer: "
                            + example[self.correct_output_col_name]
                        )
            }
        ]
        if self.system_instruction is not None:
            prompt.append({'role': 'system', 'content': self.system_instruction})
        return self.tokenizer.apply_chat_template(prompt, tokenize=False)

class DataLogCallback(TrainerCallback):
    def on_train_end(self, args, state, control, **kwargs):
        pass

class DatasetFilter:
    def __init__(
            self,
            subject=None,
            #has_answer_extracted=lambda sample: sample is True,
            has_answer_extracted=lambda sample: True,
            pass_rate_72b_tir=lambda sample: True,
            **kwargs,
        ):
        self.filter_conditions = dict()
        if isinstance(subject, Callable):
            self.filter_conditions['subject'] = subject
        if isinstance(has_answer_extracted, Callable):
            self.filter_conditions['has_answer_extracted'] = has_answer_extracted
        if isinstance(pass_rate_72b_tir, Callable):
            self.filter_conditions['pass_rate_72b_tir'] = pass_rate_72b_tir

        for key, value in kwargs.items():
            if isinstance(value, Callable):
                self.filter_conditions[key] = value

    def __call__(self, sample):
        for sample_col, sample_val in sample.items():
            if (sample_col in self.filter_conditions.keys()
                and not self.filter_conditions[sample_col](sample_val)):
                return False
        return True
    
    def __repr__(self):
        return (
            f'{self.__class__.__name__}(' 
            + '\n'.join([f'{key}={inspect.getsourcelines(val)}' for key, val in self.filter_conditions.items()]) 
            + '\n)'
        )
