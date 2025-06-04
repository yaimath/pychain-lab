# PyChain Lab : Tool-Integrate-Reasoning

데이터셋은 `datasets/` 디렉토리 아래에 넣어주시면 감사하겠습니다. 
이 프로젝트는 `nvidia/OpenMath-Nemotron-1.5B` 모델을 사용해, Tool-Integrate-Reasoning (TIR) 방식의 수학 문제 풀이를 구현한 프로젝트입니다.

- `demo.sh` : `nvidia/OpenMath-Nemotron-1.5B` 모델을 이용한 Tool-Integrate-Reasoning (TIR) 방식의 수학 문제 풀이를 해볼 수 있는 데모입니다.
- `train_config.py` : `dataset/` 디렉토리 안의 문제들에 대한 evaluation을 수행합니다. `nvidia/OpenMath-Nemotron-1.5B` 모델을 이용한 Tool-Integrate-Reasoning (TIR) 방식을 사용합니다.

### 1. 패키지 설치

```bash
pip install -r requirements.txt
```

### 2. 추론 실행

```bash
python demo.py "Find the greatest common factor of $144$ and $405$."

python demo.py "Two chords, $AB$ and $CD,$ meet inside a circle at $P.$  If $AP = CP = 7,$ then what is $\\frac{BP}{DP}$?"

python demo.py "The function $f(x)$ satisfies\n\\[f(f(x)) = 6x - 2005\\]for all real numbers $x.$  There exists an integer $n$ such that $f(n) = 6n - 2005.$  Find $n.$"
```

- Tool-Integrate-Reasoning 방식으로 수학 문제를 풀이합니다.

### 3. 모델 평가 

```bash
python evaluate.py 
```

- `dataset/` 디렉토리 안의 문제들을 풀이하고 결과를 .csv 파일로 저장합니다.
- Tool-Integrate-Reasoning 방식으로 수학 문제를 풀이합니다.
- 요구하는 데이터 형식은 아래와 같습니다.
```json
{
    "problem": "If $a * b = a^b + b^a$, for all positive integer values of $a$ and $b$, then what is the value of $2 * 6$?",
    "type": "Algebra",
    "solution": "\\boxed{100}$."
}
```
