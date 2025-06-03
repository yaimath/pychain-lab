# pychain

데이터셋은 `datasets/` 디렉토리 아래에 넣어주시면 감사하겠습니다. 직접 제작한 `gen-answer-OpenMathReasoning` 데이터셋은 폴더에 포함시켰으나, `MATH`, `OpenMathReasoning`은 저작권 문제로 인해 포함시키지 않았습니다.

- `run-baseline.sh` : `datasets/MATH/train`의 모든 데이터셋을 각각 실행하는 Shell Script입니다.
- `train_config.py` : `TrainConfig` 데이터클래스에서 멤버 변수를 조정하여 학습하실 수 있습니다.
- `train.py` : QLoRA SFT로 학습하는 코드입니다.
- `gen_reflect_samples.py` : 모델을 불러와 text generation을 하는 코드입니다.
- `train_rejected.py` : 틀린 문제들을 바탕으로 새로 학습하는 코드입니다.
- `train_router.py` : 과목을 분류하는 모델을 학습시키는 코드입니다.
- `utils.py` : 기타 유틸리티 코드가 들어있는 파일입니다.

기타 이슈, 요구사항 언제든 말씀해주세요.
