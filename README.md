# pychain

데이터셋은 `datasets/` 디렉토리 아래에 넣어주시면 감사하겠습니다.

- `run-baseline.sh` : `datasets/MATH/train`의 모든 데이터셋을 각각 실행하는 Shell Script입니다.
- `train_config.py` : `TrainConfig` 데이터클래스에서 멤버 변수를 조정하여 학습하실 수 있습니다.
- `train.py` : QLoRA SFT로 학습하는 코드입니다.
- `utils.py` : 유틸리티 코드 넣는 파일입니다. 일단은 Prompt Formatter만 구현되어 있습니다.

기타 이슈, 요구사항 언제든 말씀해주세요.
