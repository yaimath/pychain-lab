
# Verifier for LLaDA-CoT

이 프로젝트는 [`deepseek-ai/deepseek-coder-6.7b-instruct`](https://huggingface.co/deepseek-ai/deepseek-coder-6.7b-instruct) 모델을 기반으로 [`nvidia/OpenMathReasoning`](https://huggingface.co/datasets/nvidia/OpenMathReasoning/viewer/default/tir) 데이터로 훈련시켜 LLaDA-CoT의 output에 대한 verifier를 구현한 것입니다.

---

## 📁 디렉토리 구조

```
.
├── verifier_preprocessing.py             # OpenMathReasoning 데이터셋 preprocessing
├── verifier_train.py                     # LoRA 기반 학습 코드
├── verifier_eval.py                # 정답 추출 및 추론 평가 스크립트
├── requirements.txt        # 실행에 필요한 패키지 목록
└── openmathreasoning/
    ├── test_toolcall_model_instruct.jsonl         # 평가용 jsonl 파일
    └── train_toolcall_model_instruct.jsonl          # 학습용용 jsonl 파일
```

---

## 🚀 설치 및 실행 방법

### 1. 패키지 설치

```bash
pip install -r requirements.txt
```

### 2. 데이터셋 생성

```bash
python verifier_preprocessing.py 
```
- OpenMathReasoning의 output 중 toolcall과 reasoning을 추출합니다.


### 3. 학습 실행

```bash
python verifier_train.py
```


### 4. 추론 실행

```bash
python verifier_eval.py
```
- 생성된 코드의 실행가능 여부와 답의 정확도를 평가합니다.
---

## ⚙️ 주요 구성

### `verifier_preprocessing.py`

- OpenMathReasoning 데이터셋의 generated solution 중 reasoning 부분과 tool call 부분을 추출하여 verifier에 맞는 input 형식으로 변환

### `verifier_train.py`

- LoRA 기반 학습

### `verifier_eval.py`

- 테스트셋에서 `\boxed{...}` 포맷의 답 추출
- 정확도, 실행 가능 여부 평가 및 JSONL 저장

---
