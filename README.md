# AESA — 한국어 다축 감정 분석 엔진 프로토타입

> 비정형 한국어 텍스트의 감정·맥락 신호를 구조화하는 AI 분석 프로젝트입니다.

AESA(Advanced Emotion & Sentiment Analyzer)는 한국어 문장에서 단순 긍·부정을 넘어  
복합 감정, 강도, 전이, 상황 맥락, 감정 간 관계를 다축적으로 분석하기 위한 프로토타입입니다.

이 프로젝트는 전직 퍼블리셔의 언어·맥락 감각을 바탕으로,  
AI 모델의 결과를 실제 서비스에서 활용 가능한 구조로 정리하는 것을 목표로 합니다.

> 본 프로젝트는 심리 진단 도구가 아니며, 텍스트 기반 감정·맥락 신호를 분석 후보로 구조화하는 실험적 시스템입니다.

---

## 🎯 프로젝트 개요

| 항목 | 내용 |
|------|------|
| **목표** | 한국어 텍스트의 복합 감정·맥락 신호 분석 |
| **특징** | 11개 분석 관점 기반 모듈형 아키텍처 |
| **기간** | 약 7-8개월 |
| **역할** | 기획, 구조 설계, Python 기반 프로토타입 구현, 데모 제작 |
| **상태** | 포트폴리오용 프로토타입 / 지속 개선 중 |

---

## 🛠 Tech Stack

| Category | Technology |
|----------|------------|
| Language | Python 3.11+ |
| Framework | FastAPI, Uvicorn |
| Deep Learning | PyTorch 2.x |
| NLP | Sentence-Transformers, Transformers (HuggingFace) |
| Korean NLP | KSS, Kiwipiepy |
| Data | Pandas, NumPy, JSON/JSONL |
| Frontend | HTML5, CSS3, Vanilla JS |

---

## 📁 프로젝트 구조

```text
AESA/
├── src/
│   ├── emotion_analysis/      # 11개 감정 분석 모듈
│   │   ├── complex_analyzer.py
│   │   ├── context_extractor.py
│   │   ├── intensity_analyzer.py
│   │   ├── linguistic_matcher.py
│   │   ├── pattern_extractor.py
│   │   ├── psychological_analyzer.py
│   │   ├── situation_analyzer.py
│   │   ├── time_series_analyzer.py
│   │   ├── transition_analyzer.py
│   │   ├── weight_calculator.py
│   │   └── emotion_relationship_analyzer.py
│   │
│   ├── serving/               # FastAPI 웹 서버
│   ├── data_utils.py          # 메인 오케스트레이터
│   ├── config.py              # 설정 관리
│   ├── main.py                # 메타 모델 파이프라인
│   └── sub_classifier.py      # 서브 감정 분류기
│
├── made/                      # 웹 UI / 데모 페이지
└── requirements.txt
```

---

## 🔬 11개 분석 모듈

| # | 모듈 | 역할 |
|---|------|------|
| 1 | **PatternExtractor** | 감정 표현 패턴과 흐름 단서 추출 |
| 2 | **LinguisticMatcher** | 언어적 특성 기반 감정 단서 매칭 |
| 3 | **IntensityAnalyzer** | 감정 강도 측정 및 임베딩 기반 분석 |
| 4 | **ContextExtractor** | 문맥 기반 감정 후보 추론 |
| 5 | **TransitionAnalyzer** | 감정 전이 패턴 분석 |
| 6 | **TimeSeriesAnalyzer** | 시간 흐름에 따른 감정 변화 추적 |
| 7 | **SituationAnalyzer** | 상황 맥락 기반 감정 단서 분석 |
| 8 | **PsychologicalAnalyzer** | 텍스트 내 심리적 경향과 인지 단서 분석 |
| 9 | **WeightCalculator** | 감정 후보별 가중치 계산 |
| 10 | **ComplexAnalyzer** | 복합 감정 통합 분석 |
| 11 | **EmotionRelationshipAnalyzer** | 감정 간 관계성 분석 |

---

## 🏗 아키텍처

```text
[입력 텍스트]
      │
      ▼
[EmotionPipelineOrchestrator]
      │
      ├─→ [PatternExtractor]
      ├─→ [LinguisticMatcher]
      ├─→ [IntensityAnalyzer]
      ├─→ [ContextExtractor]
      ├─→ [TransitionAnalyzer]
      ├─→ [TimeSeriesAnalyzer]
      ├─→ [SituationAnalyzer]
      ├─→ [PsychologicalAnalyzer]
      ├─→ [WeightCalculator]
      ├─→ [ComplexAnalyzer]
      └─→ [EmotionRelationshipAnalyzer]
      │
      ▼
[Payload]
      │
      ▼
[통합 분석 결과]
```

---

## 핵심 설계 포인트

1. **모듈형 분석 구조**  
   감정 분석을 단일 모델 결과에만 의존하지 않고, 패턴·언어·강도·문맥·상황·전이·관계성 등 여러 관점으로 나누어 처리합니다.

2. **표준 Payload 구조**  
   여러 분석 모듈의 입출력을 통일된 컨테이너로 다루어, 결과 통합과 후속 확장을 쉽게 만들고자 했습니다.

3. **동적 호출 구조**  
   각 모듈의 진입점을 관리하는 호출 구조를 두어, 분석 모듈을 추가하거나 교체할 수 있는 방향으로 설계했습니다.

4. **안전한 결과 해석 지향**  
   감정 결과를 단정적 진단으로 제시하기보다, 텍스트에서 관찰되는 감정·맥락 신호의 후보로 다루는 것을 목표로 합니다.

5. **서비스 적용 관점**  
   단순 모델 출력이 아니라, 실제 화면·API·사용자 경험에서 활용 가능한 감정 분석 구조를 실험했습니다.

---

## 🎥 Demo Video

> **Recorded Demo**: https://www.youtube.com/watch?v=R51yFc5YOpA

AESA의 주요 분석 흐름과 결과 화면은 데모 영상과 스크린샷에서 확인할 수 있습니다.

- 주요 감정 분류
- 세부 감정 및 강도 분석
- 감정 전이 및 시간 흐름 추적
- 상황 맥락 기반 감정 해석
- 복합 감정 후보 및 관계성 분석

---

## 🖼 Screenshots

<p align="center">
  <a href="https://github.com/user-attachments/assets/bdcf22d9-e0cd-4530-91f5-3527137f21cd">
    <img src="https://github.com/user-attachments/assets/bdcf22d9-e0cd-4530-91f5-3527137f21cd" alt="AESA analysis result screen" width="100%">
  </a>
</p>

---

## 📌 프로젝트 의의

AESA는 전직 퍼블리셔가 언어와 감정의 결을 AI 서비스 구조로 옮기기 위해 제작한 포트폴리오형 프로토타입입니다.

특히 다음과 같은 문제의식에서 출발했습니다.

- 한국어 감정 표현은 단순 긍정/부정만으로 설명하기 어렵다.
- 하나의 문장 안에서도 여러 감정이 동시에 존재할 수 있다.
- 감정은 단어뿐 아니라 문맥, 상황, 시간 흐름, 관계성 속에서 달라진다.
- AI 서비스에서 감정 분석 결과는 단정이 아니라 해석 가능한 후보로 제시되어야 한다.
- 모델의 출력은 실제 서비스 화면과 사용자 경험에 맞게 구조화되어야 한다.

---

## ⚠️ Limitations

이 프로젝트는 포트폴리오 및 학습 목적의 프로토타입입니다.

현재 버전은 다음과 같은 한계를 가집니다.

- 상용 수준의 대규모 검증 데이터셋은 아직 포함되어 있지 않습니다.
- 감정 분석 결과는 심리 진단이나 의학적 판단으로 사용할 수 없습니다.
- 일부 모듈은 실험적 구조이며, 향후 리팩토링과 평가 기준 보강이 필요합니다.
- 실제 서비스 적용을 위해서는 성능 검증, 예외 처리, 보안, 운영 안정성 개선이 필요합니다.

---

## 📜 License

이 저장소의 코드는 학습 및 포트폴리오 목적으로 공개되었습니다.  
상업적 사용 시 별도 문의 바랍니다.

---

## 👤 Contact

- **Email**: madecompass@outlook.kr
- **Portfolio**: 이력서 참조

---

AESA는 비정형 감정 표현을 분석 가능한 구조로 바꾸고,  
AI 서비스의 응답 품질과 맥락 해석을 개선하기 위한 포트폴리오형 프로토타입입니다.
