# 🧮 LLM Math Reasoning Lab

이 프로젝트는 다양한 방식의 **수학 문제 해결을 위한 LLM 응용 기법**을 연구하고 구현한 프로젝트 모음입니다.  
각 하위 디렉토리는 독립적인 실험 프로젝트로 구성되어 있으며, 다양한 모델 및 추론 방식에 기반하여 LLM을 이용해 수학 문제를 푸는 방법을 탐색합니다.

---

## 📂 서브 디렉토리 소개

| 프로젝트 | 설명 |
|----------|------|
| [`verifier`](verifier/) | LLaDA-CoT의 출력을 검증하는 Verifier. DeepSeek 모델 기반으로 OpenMathReasoning dataset을 학습. |
| [`chain-of-thought`](chain-of-thought/) | QLoRA 기반의 Chain-of-Thought 학습 및 오답 기반 재학습/라우팅 모델 포함. |
| [`discrete-diffusion-llm`](discrete-diffusion-llm/) | Semi-AutoRegressive 방식으로 수학 추론을 수행하는 diffusion-style 모델. |
| [`tool-integrated-reasoning`](tool-integrated-reasoning/) | Tool 호출 기반 수학 추론(TIR)을 Nemotron 모델 기반으로 구현한 실험. |

---

## 📦 패키지 설치

- 각 서브 디렉토리의 `requirements.txt` 파일을 참고해주세요

---

## 🚀 실행 방법

각 프로젝트 디렉토리 안에 있는 `README.md`를 참고하세요.
