# -*- coding: utf-8 -*-
"""
emotion_analysis: 11개 전처리/분석 모듈을 *지연 로딩(Lazy Loading)*으로 안전하게 노출하는 패키지 집약.

[Optimization]
- __all__에 정의된 모듈/함수를 호출할 때만 실제 import가 발생하도록 __getattr__을 구현했습니다.
- 이를 통해 초기 부팅 속도를 극대화하고 순환 참조 문제를 방지합니다.
"""

import sys
import logging
import os
from typing import Any

# ── 로깅 가드 ─────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    logger.addHandler(logging.NullHandler())
logger.propagate = False

# ── Lazy Loading 매핑 테이블 ─────────────────────────────────────────────
# 심볼 이름 -> (모듈 경로, 가져올 객체 이름)
# None인 경우 모듈 경로에서 심볼 이름 그대로 가져옴
_LAZY_MAP = {
    # 1) complex_analyzer
    "ComplexAnalyzerConfig": (".complex_analyzer", "AnalyzerConfig"),
    "PreprocessConfig": (".complex_analyzer", None),
    "SegmenterConfig": (".complex_analyzer", None),
    "FeatureExtractorConfig": (".complex_analyzer", None),
    "ComplexEmotionAnalyzer": (".complex_analyzer", None),
    "analyze_complex_emotions": (".complex_analyzer", None),
    "analyze_emotion_patterns": (".complex_analyzer", None),
    "generate_emotion_embedding": (".complex_analyzer", None),
    "calculate_emotion_complexity": (".complex_analyzer", None),
    "analyze_emotional_conflicts": (".complex_analyzer", None),
    "identify_dominant_emotions": (".complex_analyzer", None),
    "get_emotion_transitions": (".complex_analyzer", None),
    "get_emotional_changes": (".complex_analyzer", None),

    # 2) context_extractor
    "EmotionContext": (".context_extractor", None),
    "EmotionContextManager": (".context_extractor", None),
    "ContextExtractor": (".context_extractor", None),
    "extract_contextual_emotions": (".context_extractor", None),
    "analyze_progressive_context": (".context_extractor", None),
    "analyze_situation_impact": (".context_extractor", None),
    "analyze_context_patterns": (".context_extractor", None),
    "analyze_context_emotion_correlations": (".context_extractor", None),

    # 3) emotion_relationship_analyzer
    "EmotionRelationshipAnalyzer": (".emotion_relationship_analyzer", None),
    "analyze_social_emotion_graph": (".emotion_relationship_analyzer", None),
    "prewarm_pipeline": (".emotion_relationship_analyzer", None),
    "reset_pipeline_cache": (".emotion_relationship_analyzer", None),
    "analyze_emotion_relationships": (".emotion_relationship_analyzer", None),
    "rel_analyze_emotion_progression": (".emotion_relationship_analyzer", "analyze_emotion_progression"),
    "analyze_relationship_expansion": (".emotion_relationship_analyzer", None),
    "analyze_emotion_compatibility": (".emotion_relationship_analyzer", None),
    "identify_emotion_pairs": (".emotion_relationship_analyzer", None),
    "build_social_emotion_graph": (".emotion_relationship_analyzer", None),
    "compute_pairs_compatibility_strength": (".emotion_relationship_analyzer", None),
    "run_main_once": (".emotion_relationship_analyzer", None),
    "EmotionProgressionRelationshipAnalyzer": (".emotion_relationship_analyzer", None),
    "SocialEmotionGraph": (".emotion_relationship_analyzer", None),
    "TextAnalyzer": (".emotion_relationship_analyzer", None),

    # 4) intensity_analyzer
    "EmotionIntensityAnalyzer": (".intensity_analyzer", None),
    "get_intensity_analyzer": (".intensity_analyzer", None),
    "clear_intensity_cache": (".intensity_analyzer", None),
    "analyze_intensity": (".intensity_analyzer", None),
    "analyze_intensity_transitions": (".intensity_analyzer", None),
    "analyze_temporal_patterns": (".intensity_analyzer", None),
    "analyze_situational_intensity": (".intensity_analyzer", None),
    "generate_intensity_embeddings": (".intensity_analyzer", None),
    "embedding_generation": (".intensity_analyzer", "run_embedding_generation"),
    "run_intensity_analysis": (".intensity_analyzer", None),

    # 5) linguistic_matcher
    "EmotionProgressionMatcher": (".linguistic_matcher", None),
    "LinguisticMatcher": (".linguistic_matcher", None),
    "EmotionalAnalyzer": (".linguistic_matcher", None),
    "match_language_patterns": (".linguistic_matcher", None),
    "ling_analyze_emotion_progression": (".linguistic_matcher", "analyze_emotion_progression"),
    "analyze_emotion_progression": (".linguistic_matcher", None),
    "analyze_enhanced_emotions": (".linguistic_matcher", None),
    "analyze_emotions_centrally": (".linguistic_matcher", None),
    "run_linguistic_analysis": (".linguistic_matcher", None),
    "resolve_emotions_json_path": (".linguistic_matcher", None),
    "lm_set_cache_enabled": (".linguistic_matcher", None),
    "lm_clear_cache": (".linguistic_matcher", None),
    "lm_cache_stats": (".linguistic_matcher", None),

    # 6) pattern_extractor
    "PatternContextAnalyzer": (".pattern_extractor", None),
    "EmotionPatternExtractor": (".pattern_extractor", None),
    "PatternExtractor": (".pattern_extractor", None),
    "run_pattern_extraction": (".pattern_extractor", None),
    "analyze_emotion_flow": (".pattern_extractor", None),
    "get_temporal_analysis": (".pattern_extractor", None),
    "extract_emotion_patterns": (".pattern_extractor", None),
    "validate_emotion_patterns": (".pattern_extractor", None),
    "to_data_utils_format": (".pattern_extractor", None), # 기본 포맷터
    "to_pattern_data_utils_format": (".pattern_extractor", "to_data_utils_format"),
    "bench_extract_parallel": (".pattern_extractor", None),

    # 7) psychological_analyzer
    "PsychAnalyzerConfig": (".psychological_analyzer", "AnalyzerConfig"),
    "AnalyzerConfig": (".psychological_analyzer", "AnalyzerConfig"), # 기본 설정
    "SimpleTokenizer": (".psychological_analyzer", None),
    "EvidenceGate": (".psychological_analyzer", None),
    "EmotionsIndex": (".psychological_analyzer", None),
    "load_emotions_json": (".psychological_analyzer", None),
    "build_index": (".psychological_analyzer", None),
    "prepare_emotions_index": (".psychological_analyzer", None),
    "get_default_config": (".psychological_analyzer", None),
    "build_or_load_index": (".psychological_analyzer", None),
    "get_psych_analyzer": (".psychological_analyzer", None),
    "run_psych_analysis": (".psychological_analyzer", None),
    "analyze_sequence_only": (".psychological_analyzer", None),
    "validate_emotions_schema": (".psychological_analyzer", None),
    "to_psych_data_utils_format": (".psychological_analyzer", "to_data_utils_format"),
    "PsychologicalCognitiveAnalyzer": (".psychological_analyzer", None),
    "PsychCogIntegratedAnalyzer": (".psychological_analyzer", None),
    "SignalExtractor": (".psychological_analyzer", None),
    "LabelStats": (".psychological_analyzer", None),
    "WeightCalibrator": (".psychological_analyzer", None),
    "CalibrationSnapshot": (".psychological_analyzer", None),
    "psych_index_stats": (".psychological_analyzer", None),
    "psych_debug_tokens": (".psychological_analyzer", None),

    # 8) situation_analyzer
    "validate_emotion_data": (".situation_analyzer", None),
    "SituationContext": (".situation_analyzer", None),
    "SpatiotemporalContext": (".situation_analyzer", None),
    "EmotionProgressionStage": (".situation_analyzer", None),
    "EmotionFlowPattern": (".situation_analyzer", None),
    "RegexShardMatcher": (".situation_analyzer", None),
    "EmotionProgressionSituationAnalyzer": (".situation_analyzer", None),
    "SituationContextMapper": (".situation_analyzer", None),
    "SituationContextOrchestrator": (".situation_analyzer", None),
    "SituationAnalyzer": (".situation_analyzer", None),
    "run_situation_analysis": (".situation_analyzer", None),
    "run_spatiotemporal_analysis": (".situation_analyzer", None),
    "run_full_situation_analysis": (".situation_analyzer", None),
    "analyze_situations": (".situation_analyzer", None),

    # 9) time_series_analyzer
    "AnalysisResult": (".time_series_analyzer", None),
    "EmotionDataManager": (".time_series_analyzer", None),
    "EmotionSequenceAnalyzer": (".time_series_analyzer", None),
    "CausalityTransitionAnalyzer": (".time_series_analyzer", None),
    "TimeFlowMode": (".time_series_analyzer", None),
    "build_time_series_components": (".time_series_analyzer", None),
    "build_emotion_sequence": (".time_series_analyzer", None),
    "analyze_causality_only": (".time_series_analyzer", None),
    "analyze_time_series_reinforced": (".time_series_analyzer", None),
    "run_emotion_analysis": (".time_series_analyzer", None),
    "run_emotion_analysis_with_components": (".time_series_analyzer", None),
    "run_time_series_analysis": (".time_series_analyzer", None),
    "run_causality_analysis": (".time_series_analyzer", None),
    "run_full_time_series_analysis": (".time_series_analyzer", None),

    # 10) transition_analyzer
    "TransitionMetrics": (".transition_analyzer", None),
    "TransitionAnalyzer": (".transition_analyzer", None),
    "EmotionProgressionAnalyzer": (".transition_analyzer", None),
    "TransitionRelationshipAnalyzer": (".transition_analyzer", "RelationshipAnalyzer"),
    "RelationshipAnalyzer": (".transition_analyzer", "RelationshipAnalyzer"),
    "ComplexTransitionAnalyzer": (".transition_analyzer", None),
    "EmotionNode": (".transition_analyzer", None),
    "TransitionResult": (".transition_analyzer", None),
    "build_transition_components": (".transition_analyzer", None),
    "run_transition_analysis": (".transition_analyzer", None),
    "run_basic_transition_analysis": (".transition_analyzer", None),
    "analyze_emotion_transitions": (".transition_analyzer", None),
    "analyze_corpus_transitions": (".transition_analyzer", None),
    "TransitionComponents": (".transition_analyzer", None),

    # 11) weight_calculator
    "EmotionLogger": (".weight_calculator", None),
    "WeightEmotionDataManager": (".weight_calculator", None),
    "EmotionWeightCalculator": (".weight_calculator", None),
    "run_emotion_weight_calculation": (".weight_calculator", None),
    "run_emotion_weight_quick": (".weight_calculator", None),

    # Aliases
    "AnalyzerConfig_Complex": (".complex_analyzer", "AnalyzerConfig"),
    "AnalyzerConfig_Psych": (".psychological_analyzer", "AnalyzerConfig"),
}

# ── __getattr__을 이용한 지연 로딩 구현 ─────────────────────────────────────
def __getattr__(name: str) -> Any:
    if name in _LAZY_MAP:
        module_path, target_name = _LAZY_MAP[name]
        target_name = target_name or name
        
        # 모듈 import
        try:
            # 상대 경로 import를 위해 package 인자 필요
            from importlib import import_module
            mod = import_module(module_path, package=__name__)
            val = getattr(mod, target_name)
            
            # 캐싱: 다음에 다시 import하지 않도록 현재 모듈의 속성으로 설정
            globals()[name] = val
            return val
        except ImportError as e:
            # 특정 모듈이 없을 때 (예: torch 없음) 유연하게 처리
            logger.warning(f"Lazy loading failed for {name}: {e}")
            raise
        except AttributeError as e:
            logger.warning(f"Attribute {target_name} not found in {module_path}: {e}")
            raise
            
    # 독립 실행 함수들은 __init__.py 내부에 정의되어 있으므로 바로 반환
    # (이 파일의 전역 네임스페이스에 있는 함수들)
    if name in globals():
        return globals()[name]
        
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# ── EMOTIONS.json 경로 해석 통일 (즉시 실행) ────────────────────────────────
def _emotions_path():
    """EMOTIONS.json 경로를 통일된 방식으로 해석"""
    try:
        from src.config import EMOTIONS_JSON_PATH as _EJP
    except ImportError:
        try:
            from config import EMOTIONS_JSON_PATH as _EJP
        except ImportError:
            _EJP = None
    
    if _EJP and os.path.exists(_EJP):
        return str(_EJP)
    else:
        fallback_paths = [
            "EMOTIONS.json", "src/EMOTIONS.json", "src/EMOTIONS.JSON", "EMOTIONS.JSON"
        ]
        for path in fallback_paths:
            if os.path.exists(path):
                return path
        return "EMOTIONS.json"

# ── 독립 실행 함수들 (기존 정의 유지) ────────────────────────────────────────
# 이 함수들은 내부에서 import를 수행하므로 이미 지연 로딩 특성을 가지고 있음
# 그대로 유지해도 무방함

def create_optimized_intensity_analyzer(emotions_data_path: str, **kwargs):
    """성능 최적화된 IntensityAnalyzer 생성 - 모델 캐싱 및 지연 로딩"""
    from .intensity_analyzer import EmotionIntensityAnalyzer
    
    analyzer = EmotionIntensityAnalyzer(emotions_data_path, **kwargs)
    if hasattr(analyzer, '_model_loaded'):
        if analyzer._model_loaded:
            logger.info("[optimized] IntensityAnalyzer - 캐시된 모델 사용")
        else:
            logger.info("[optimized] IntensityAnalyzer - 지연 로딩 활성화")
    return analyzer

def get_model_cache_stats():
    try:
        try:
            from src.config import get_model_cache_stats
        except ImportError:
            from config import get_model_cache_stats
        return get_model_cache_stats()
    except Exception:
        return {}

def clear_model_cache():
    try:
        try:
            from src.config import clear_model_cache
        except ImportError:
            from config import clear_model_cache
        clear_model_cache()
    except Exception:
        pass

def analyze_intensity_independent(text: str) -> dict:
    try:
        from .intensity_analyzer import EmotionIntensityAnalyzer
        emotions_path = _emotions_path()
        analyzer = EmotionIntensityAnalyzer(emotions_path)
        result = analyzer.analyze_emotion_intensity(text)
        if isinstance(result, dict): result["success"] = True
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}

def analyze_situation_independent(text: str) -> dict:
    try:
        # ★★★ 변경: run_situation_analysis 사용 (감정 기반 상황 추론 포함) ★★★
        from .situation_analyzer import run_situation_analysis
        import json
        emotions_path = _emotions_path()
        emotions_data = {}
        if os.path.exists(emotions_path):
            with open(emotions_path, 'r', encoding='utf-8') as f:
                emotions_data = json.load(f)
        # run_situation_analysis는 SituationContextOrchestrator를 사용하여
        # 키워드 매칭 + 감정 기반 상황 추론을 모두 수행함
        result = run_situation_analysis(
            text,
            emotions_data,
            min_conf=0.15,        # 낮은 임계값으로 더 많은 상황 감지
            emit_min_evidence=1,  # 최소 1개 증거로 상황 감지
        )
        if isinstance(result, dict) and "success" not in result: result["success"] = True
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}

def analyze_time_series_independent(text: str, **kwargs) -> dict:
    """
    시계열 분석 독립 함수
    
    Args:
        text: 분석할 텍스트
        **kwargs: data_utils.py에서 전달되는 추가 인자들 (data_manager, sequence 등)
    """
    try:
        from .time_series_analyzer import EmotionSequenceAnalyzer
        
        # data_utils.py에서 전달된 data_manager 사용 (있으면)
        data_manager = kwargs.get("data_manager")
        config = kwargs.get("config", {})
        
        analyzer = EmotionSequenceAnalyzer(data_manager=data_manager, config=config)
        result = analyzer.analyze(text)
        
        if isinstance(result, dict) and "success" not in result: 
            result["success"] = True
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}

def calculate_weights_independent(text: str) -> dict:
    try:
        from .weight_calculator import EmotionWeightCalculator, WeightEmotionDataManager, EmotionLogger
        emotions_path = _emotions_path()
        logger = EmotionLogger('weight_calculator_independent.log')
        dm = WeightEmotionDataManager(logger=logger)
        dm.load_emotions_data(emotions_path)
        use_cuda = (os.getenv("USE_CUDA", "1") == "1")
        analyzer = EmotionWeightCalculator(data_manager=dm, logger=logger, use_cuda=use_cuda)
        result = analyzer.calculate_emotion_weights(text)
        if isinstance(result, dict) and "success" not in result: result["success"] = True
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}

def calculate_weights_fast(text: str) -> dict:
    try:
        import os
        original_simple = os.environ.get("WEIGHT_CALCULATOR_SIMPLE", "0")
        original_fast = os.environ.get("WEIGHT_CALCULATOR_FAST", "0")
        os.environ["WEIGHT_CALCULATOR_SIMPLE"] = "1"
        os.environ["WEIGHT_CALCULATOR_FAST"] = "1"
        try:
            result = calculate_weights_independent(text)
            if isinstance(result, dict):
                result["fast_mode"] = True
                result["optimization"] = "CUDA + Cache + Simple Matching"
            return result
        finally:
            os.environ["WEIGHT_CALCULATOR_SIMPLE"] = original_simple
            os.environ["WEIGHT_CALCULATOR_FAST"] = original_fast
    except Exception as e:
        return {"success": False, "error": str(e), "fast_mode": True}

def analyze_relationships_independent(text: str) -> dict:
    try:
        from .emotion_relationship_analyzer import EmotionRelationshipAnalyzer
        analyzer = EmotionRelationshipAnalyzer()
        result = analyzer.analyze_relationships(text)
        if isinstance(result, dict):
            if "success" not in result:
                if result.get("relationship_analysis") is False: result["success"] = False
                else: result["success"] = True
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}

def analyze_linguistic_patterns_independent(text: str) -> dict:
    try:
        from .linguistic_matcher import LinguisticMatcher
        analyzer = LinguisticMatcher()
        result = analyzer.analyze_linguistic_patterns(text)
        if isinstance(result, dict) and "success" not in result: result["success"] = True
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}

def analyze_transitions_independent(text: str) -> dict:
    try:
        from .transition_analyzer import TransitionAnalyzer
        analyzer = TransitionAnalyzer()
        result = analyzer.analyze_transitions(text)
        if isinstance(result, dict) and "success" not in result: result["success"] = True
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}

def extract_patterns_independent(text: str, *, emotions_data: Any = None, emotions_data_path: Any = None) -> dict:
    try:
        from .pattern_extractor import run_pattern_extraction
        result = run_pattern_extraction(text, emotions_data=emotions_data, emotions_data_path=emotions_data_path)
        if isinstance(result, dict) and "success" not in result: result["success"] = True
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}

def extract_context_independent(text: str) -> dict:
    try:
        from .context_extractor import ContextExtractor
        ce = ContextExtractor()
        fn = getattr(ce, "extract_context_new", None) or getattr(ce, "extract_context", None)
        if fn is None: raise AttributeError("Method not found")
        result = fn(text)
        if isinstance(result, dict) and "success" not in result: result["success"] = True
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}

def analyze_complexity_independent(text: str) -> dict:
    try:
        from .complex_analyzer import ComplexEmotionAnalyzer
        analyzer = ComplexEmotionAnalyzer()
        result = analyzer.analyze_document(text)
        if isinstance(result, dict) and "success" not in result: result["success"] = True
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}

def analyze_psychological_patterns_independent(text: str) -> dict:
    try:
        from .psychological_analyzer import run_psych_analysis
        result = run_psych_analysis(text)
        if isinstance(result, dict) and "success" not in result: result["success"] = True
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}

# --- Full Pipeline Wrappers ---
def run_complex_analysis_independent(text: str) -> dict:
    return analyze_complexity_independent(text)

def run_context_analysis_independent(text: str) -> dict:
    return extract_context_independent(text)

def run_relationship_analysis_independent(text: str) -> dict:
    return analyze_relationships_independent(text)

def run_intensity_analysis_independent(text: str) -> dict:
    try:
        from .intensity_analyzer import run_intensity_analysis
        result = run_intensity_analysis(text)
        if isinstance(result, dict) and "success" not in result: result["success"] = True
        return result
    except Exception as e: return {"success": False, "error": str(e)}

def run_linguistic_analysis_independent(text: str) -> dict:
    try:
        from .linguistic_matcher import run_linguistic_analysis
        result = run_linguistic_analysis(text)
        if isinstance(result, dict) and "success" not in result: result["success"] = True
        return result
    except Exception as e: return {"success": False, "error": str(e)}

def run_pattern_analysis_independent(text: str) -> dict:
    return extract_patterns_independent(text)

def run_psychological_analysis_independent(text: str) -> dict:
    return analyze_psychological_patterns_independent(text)

def run_situation_analysis_independent(text: str) -> dict:
    try:
        from .situation_analyzer import run_situation_analysis
        result = run_situation_analysis(text)
        if isinstance(result, dict) and "success" not in result: result["success"] = True
        return result
    except Exception as e: return {"success": False, "error": str(e)}

def run_time_series_analysis_independent(text: str) -> dict:
    try:
        from .time_series_analyzer import run_time_series_analysis
        result = run_time_series_analysis(text)
        if isinstance(result, dict) and "success" not in result: result["success"] = True
        return result
    except Exception as e: return {"success": False, "error": str(e)}

def run_transition_analysis_independent(text: str) -> dict:
    try:
        from .transition_analyzer import run_transition_analysis
        result = run_transition_analysis(text)
        if isinstance(result, dict) and "success" not in result: result["success"] = True
        return result
    except Exception as e: return {"success": False, "error": str(e)}

def run_weight_calculation_independent(text: str, **kwargs) -> dict:
    try:
        from .weight_calculator import run_emotion_weight_calculation
        result = run_emotion_weight_calculation(text, **kwargs)
        if isinstance(result, dict) and "success" not in result: result["success"] = True
        return result
    except Exception as e: return {"success": False, "error": str(e)}

# ── 공개 심볼 목록 ──────────────────────────────────────────────────────────
__all__ = list(_LAZY_MAP.keys()) + [
    "create_optimized_intensity_analyzer",
    "get_model_cache_stats",
    "clear_model_cache",
    "analyze_intensity_independent",
    "analyze_situation_independent",
    "analyze_time_series_independent",
    "calculate_weights_independent",
    "calculate_weights_fast",
    "analyze_relationships_independent",
    "analyze_linguistic_patterns_independent",
    "analyze_transitions_independent",
    "extract_patterns_independent",
    "extract_context_independent",
    "analyze_complexity_independent",
    "analyze_psychological_patterns_independent",
    "run_complex_analysis_independent",
    "run_context_analysis_independent",
    "run_relationship_analysis_independent",
    "run_intensity_analysis_independent",
    "run_linguistic_analysis_independent",
    "run_pattern_analysis_independent",
    "run_psychological_analysis_independent",
    "run_situation_analysis_independent",
    "run_time_series_analysis_independent",
    "run_transition_analysis_independent",
    "run_weight_calculation_independent",
]
