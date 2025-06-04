import json

# 입력 파일
input_file = "datasets/test.jsonl"

# 출력 파일들
prompt_output_file = "prompt.txt"
answer_output_file = "answer.txt"

# 구분자 설정
separator = "<|QUESTION_SEPARATOR|>\n"

with open(input_file, 'r', encoding='utf-8') as infile, \
     open(prompt_output_file, 'w', encoding='utf-8') as prompt_out, \
     open(answer_output_file, 'w', encoding='utf-8') as answer_out:
    
    for idx, line in enumerate(infile):
        data = json.loads(line)
        prompt_out.write(data["prompt"].strip() + "\n")
        answer_out.write(data["answer"].strip() + "\n")
        
        # 마지막 항목이 아니면 구분자 추가
        prompt_out.write(separator)
        answer_out.write(separator)
