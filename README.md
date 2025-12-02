# AESA (Advanced Emotion & Sentiment Analyzer)

> **AI 기술의 라스트마일 실무자** — 모델을 '서비스'로 완성하는 퍼블리셔

한국어 텍스트의 복합 감정을 분석하는 AI 시스템입니다.  
11개의 전문 분석 모듈이 협력하여 단순 긍/부정을 넘어 **심층적인 감정 흐름과 심리 패턴**을 도출합니다.

---

## 🎯 프로젝트 개요

| 항목 | 내용 |
|------|------|
| **목표** | 한국어 텍스트의 복합 감정 분석 |
| **특징** | 11개 전처리 모듈 통합 아키텍처 |
| **기간** | 7-8개월 (1인 개발) |
| **역할** | 기획, 설계, 풀스택 개발 |

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

```
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
├── made/                      # 웹 UI (데모 페이지)
└── requirements.txt
```

---

## 🔬 11개 분석 모듈

| # | 모듈 | 역할 |
|---|------|------|
| 1 | **PatternExtractor** | 감정 패턴 추출 및 흐름 분석 |
| 2 | **LinguisticMatcher** | 언어적 특성 기반 감정 매칭 |
| 3 | **IntensityAnalyzer** | 감정 강도 측정 및 임베딩 생성 |
| 4 | **ContextExtractor** | 문맥 기반 감정 추론 |
| 5 | **TransitionAnalyzer** | 감정 전이 패턴 분석 |
| 6 | **TimeSeriesAnalyzer** | 시계열 감정 변화 추적 |
| 7 | **SituationAnalyzer** | 상황 맥락 기반 감정 추론 |
| 8 | **PsychologicalAnalyzer** | 심리적 패턴 및 인지 편향 분석 |
| 9 | **WeightCalculator** | 감정 가중치 계산 |
| 10 | **ComplexAnalyzer** | 복합 감정 통합 분석 |
| 11 | **EmotionRelationshipAnalyzer** | 감정 간 관계성 분석 |

---

## 🏗 아키텍처

```
[입력 텍스트]
      │
      ▼
[EmotionPipelineOrchestrator]  ← 메인 전처리기
      │
      ├─→ [PatternExtractor]
      ├─→ [LinguisticMatcher]
      ├─→ [IntensityAnalyzer]
      │        ...
      └─→ [11개 모듈 순차/병렬 실행]
      │
      ▼
[Payload]  ← 표준화된 I/O 컨테이너
      │
      ▼
[통합 분석 결과]
```

### 핵심 설계 포인트

1. **Lazy Loading**: 모듈을 실제 호출 시에만 로드하여 부팅 속도 최적화
2. **표준 Payload**: 11개 모듈의 입출력을 단일 형식으로 통일
3. **CALL_TABLE**: 각 모듈의 진입점을 동적으로 관리
4. **안전 호출**: 타임아웃, 폴백, 에러 처리 통합

---

## 🎨 데모

> **Live Demo**: 비공개 (이력서 참조)

데모 페이지에서 한국어 텍스트를 입력하면:
- 주요 감정 (희/노/애/락) 분류
- 세부 감정 도출
- 감정 강도 및 변화 추이
- 심리적 패턴 분석
- 상황 맥락 추론

---

## 📜 License

이 저장소의 코드는 학습 및 포트폴리오 목적으로 공개되었습니다.  
상업적 사용 시 별도 문의 바랍니다.

---

## 👤 Contact

- **Email**: madecompass@outlook.kr
- **Portfolio**: [이력서 참조]

---

*이 프로젝트는 퍼블리셔가 AI 서비스의 "라스트마일"을 어떻게 완성하는지 보여주기 위해 제작되었습니다.*
