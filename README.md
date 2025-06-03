
# 🧠 LLaDA-CoT: Semi-Autoregressive Reasoning with LLaDA

이 프로젝트는 [`GSAI-ML/LLaDA-8B-Instruct`](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct) 모델을 기반으로, **Step-by-step Chain-of-Thought(CoT) 추론**을 반영한 Semi-Auto Regressive 방식의 문제 해결 방식을 구현한 것입니다.

> Masked Denoising 방식과 LoRA 기반 미세조정을 통해 LLaDA 모델의 수학 문제 해결 능력을 향상시키는 것이 목표입니다.

---

## 📁 디렉토리 구조

```
.
├── generate.py             # 세미 오토리그레시브 방식 생성 함수
├── datamaker.py            # 기존의 MATH 데이터셋을 학습, 평가용 jsonl 파일로 변경경
├── train.py                # LoRA 기반 학습 코드
├── solve.py                # 정답 추출 및 추론 평가 스크립트
├── requirements.txt        # 실행에 필요한 패키지 목록
├── modelss/                # 학습된 adapter(LORA) 저장 경로
└── datasets/
    ├── train.jsonl         # 학습용 jsonl 파일
    └── test.jsonl          # 평가용 jsonl 파일
```

---

## 🚀 설치 및 실행 방법

### 1. 패키지 설치

```bash
pip install -r requirements.txt
```

### 2. 데이터셋 생성

```bash
python datamaker.py 
```
- Option 이지만 MATH dataset을 diffusion model에 CoT하기 적합한 형태로 변형합니다.
- datasets 폴더 하위에 MATH를 두면 됩니다.


### 3. 학습 실행

```bash
python train.py
```

- 학습 데이터: `datasets/train.jsonl`
- 학습 결과: `modelss/llada_multi_pass_epoch*` 에 저장됩니다.

### 4. 추론 실행

```bash
python solve.py
```

- 입력: `datasets/test.jsonl`
- 결과: `infer_outputsss1.jsonl` (정답/예측/출력 포함)

---

## ⚙️ 주요 구성

### `generate.py`

- Semi-autoregressive masked generation 구현
- `generate(...)` 함수는 `block_length` 단위로 마스크된 영역을 점진적으로 채움
- Gumbel noise 및 classifier-free guidance 기능 포함

### `train.py`

- LoRA 기반 Masked Denoising 학습 루프
- Chain-of-Thought 데이터를 토막내어 step-by-step으로 모델에 학습

### `solve.py`

- 테스트셋에서 `\boxed{...}` 포맷의 답 추출
- 정확도 평가 및 JSONL 저장

---

## 📊 예시 결과 (output JSONL)

```json
{
  "prompt": "What is 25 × 4?",
  "gold_answer": "100",
  "model_answer": "100",
  "raw_output": "To calculate 25 × 4, we multiply the numbers: 25 × 4 = \boxed{100}",
  "correct": true
}
```

---

## 📝 데이터 형식

### 학습 데이터 (`train.jsonl`)

```json
{
  "prompt": "Solve 3 + 4",
  "rationales": ["3 + 4 = 7"],
  "answer": "\boxed{7}"
}
```

### 테스트 데이터 (`test.jsonl`)

```json
{
  "prompt": "What is 15 × 2?",
  "answer": "30"
}
```
