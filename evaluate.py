import io
import contextlib
import time

import torch
import transformers
from transformers import TextStreamer
import io, contextlib, time
from typing import List, Dict

import torch, transformers
import re



# ------------------------------------------------------------
# 0. Load Model
# ------------------------------------------------------------
MODEL_ID = "nvidia/OpenMath-Nemotron-1.5B"

pipe = transformers.pipeline(
    task="text-generation",
    model=MODEL_ID,
    device_map="auto",
    trust_remote_code=True,      
    model_kwargs={"torch_dtype": torch.bfloat16},
)

# ------------------------------------------------------------
# 1. function to run code between <tool_call> … </tool_call> 
# ------------------------------------------------------------
import ast, subprocess, textwrap, sys

def run_code(code: str) -> str:
    tree = ast.parse(code)             
    last = tree.body[-1]              

    if isinstance(last, ast.Expr):
        expr_src = textwrap.dedent(code.splitlines()[-1])
        code += f"\nprint({expr_src})"

    elif (isinstance(last, ast.Assign) and
          len(last.targets) == 1 and
          isinstance(last.targets[0], ast.Name)):
        var = last.targets[0].id
        code += f"\nprint({var})"

    proc = subprocess.run(
        ["python3", "-c", code],
        capture_output=True,
        text=True
    )

    # if proc.stderr:  # Show stderr
    #     print(proc.stderr, file=sys.stderr)

    return proc.stdout

# ------------------------------------------------------------
# 2. Textstreamer - check "</tool_call>" token
# ------------------------------------------------------------
TARGET_TOKEN = "</tool_call>"         

executed_outputs: List[str] = []       # output of generation 

class TokenDetectorStreamer(TextStreamer):
    def __init__(self, tokenizer):
        super().__init__(tokenizer, skip_prompt=True)
        self.acc_text = ""             # streamed text
        self.detect_cnt = 0
        self.total_code_executions = 5

    def on_finalized_text(self, text: str, stream_end: bool = False):
        global executed_outputs
        self.acc_text += text
        # print(text, end="", flush=True)   

        # check for </tool_call> 
        while TARGET_TOKEN in self.acc_text:
            block, _, rest = self.acc_text.partition(TARGET_TOKEN)
            try:
                code = block.split("<tool_call>")[self.detect_cnt + 1]
                stdout = run_code(code.strip())
                executed_outputs.append(stdout)
                self.total_code_executions -= 1
            except Exception as e:
                executed_outputs.append("")
            self.detect_cnt += 1
            self.acc_text = rest         

# ------------------------------------------------------------
# 3. Instantiate Streamer, Stop criteria 
# ------------------------------------------------------------
streamer = TokenDetectorStreamer(
    pipe.tokenizer                
)

from transformers import StoppingCriteria, StoppingCriteriaList
stop_token_ids = [151658, 151645] # </tool_call>, <|im_end|>
class StopOnTokenSequence(StoppingCriteria):
    def __init__(self, stop_sequence):
        self.stop_sequence = stop_sequence
        self.buffer = []

    def __call__(self, input_ids, scores, **kwargs):
        self.buffer.append(input_ids[0, -1].item())
        if len(self.buffer) > len(self.stop_sequence):
            self.buffer.pop(0)
        return self.buffer == self.stop_sequence

stop_criteria = StoppingCriteriaList([
    StopOnTokenSequence(stop_token_ids)
])


# extract content in \boxed{…} block
def extract_last_boxed_text(s: str) -> str | None:
    key = r'\boxed{'
    start = s.rfind(key)          
    if start == -1:
        return None                

    i = start + len(key)           
    brace_level = 1               

    while i < len(s):
        ch = s[i]
        if ch == '{':
            brace_level += 1
        elif ch == '}':
            brace_level -= 1
            if brace_level == 0:   # reached }
                return s[start + len(key): i]
        i += 1

    # no } 
    return None

import pandas as pd
import json
import os
import time

def evaluate(q, a):
    start = time.time()
    total_code_executions = 3
    problem = q

    answer = extract_last_boxed_text(a)

    input_text = f"""Solve the following math problem, integrating natural language reasoning with Python code executions. 
    You may perform up to {total_code_executions} Python code calls to assist your reasoning.
    Make sure to put the answer (and only answer) inside \boxed{{}}.

    {problem}"""

    messages = [
        {
            "role": "user", 
            "content": input_text},
    ]

    MAX_TURNS = 3
    for turn in range(MAX_TURNS + 1):
        executed_outputs.clear()                     
        streamer = TokenDetectorStreamer(pipe.tokenizer)

        outputs = pipe(messages, max_new_tokens=2048, streamer=streamer, stopping_criteria=stop_criteria)
        assistant_reply = outputs[0]["generated_text"][-1]["content"]
        messages.append({"role": "assistant", "content": assistant_reply})

        if "\\boxed{" in messages[-1]['content'] or turn == MAX_TURNS:
            break
        
        tool_content = "".join(executed_outputs)
        if MAX_TURNS-turn-1 != 0:
            tool_result = ("\n```output\n" 
                            + str(tool_content) 
                            + "```\n```system"
                            + f"\nRemaining code executions: {MAX_TURNS-turn-1}. You will not be able to call code when you run out of executions, so use it wisely. Note that you can still continue solving the problem without code after that."
                            +"\n```"
                            )
        else:
            tool_result = ("\n```output\n" 
                            + str(tool_content) 
                            + "```\n```system"
                            + "Remaining code executions: 0. You have run out of code executions! You can no longer write or execute code. Now you should continue solving the problem by relying on your mathematical reasoning and analytical skills."
                            +"\n```"
                            )
        if executed_outputs:
            messages[-1]['content'] = messages[-1]['content'] + tool_result
        else:
            messages[-1]['content'] = messages[-1]['content'] + tool_result    
    return extract_last_boxed_text(messages[-1]['content']), answer, time.time() - start

results = []

def get_all_file_paths(directory):
    file_paths = []
    for root, dirs, files in os.walk(directory):
        for name in files:
            full_path = os.path.join(root, name)
            file_paths.append(full_path)
    return file_paths

target_dir = "./dataset"
all_files = get_all_file_paths(target_dir)

print("evaluation Start! It may take some time...")
for path in get_all_file_paths(target_dir):
    with open(path, 'r') as f:
        data = json.load(f)

    sub = data['type']
    q = data['problem']
    a = data['solution']

    print(f"evaluating: {q}")

    pred, label, elapsed_time = evaluate(q, a)

    results.append({'subject': sub, 'pred': pred, 'label': label, 'elapsed_time': elapsed_time})
    print(f"answer is ... {pred}\nlabel: {label}")
    torch.cuda.empty_cache()

# list to DataFrame
df = pd.DataFrame(results)
df.to_csv("evaluation_result.csv")
print(f'evaluation complete! \nevaluation_result.csv generated.')