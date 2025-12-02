# pattern_extractor.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import sys
import json
import time
import torch
import logging
import threading
import re
import math
import unicodedata
import importlib
import numpy as np
from functools import lru_cache
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Dict, Any, Tuple, List, Optional, Set

def _import_config_module():
    """config 모듈을 유연하게 불러오기 (다중 후보 지원)"""
    # 우선순위: 환경변수 → 루트 패키지 → src 패키지 → emotion_analysis 패키지
    candidates = [
        os.getenv("EA_CONFIG_MODULE"),           # e.g. "config" or "src.config"
        "config",
        "src.config",
        "emotion_analysis.config",
    ]
    last_err = None
    for name in candidates:
        if not name:
            continue
        try:
            return importlib.import_module(name)
        except Exception as e:
            last_err = e
    raise ImportError(f"패키지 레이아웃 문제: config 모듈을 찾을 수 없습니다 "
                      f"(tried: {candidates}). last_err={last_err!r}")

def _load_emotions_data(emotions_data=None, emotions_data_path=None):
    """emotions 데이터를 유연하게 로드 (주입값 우선 사용)"""
    # 1) 주입값 우선
    if emotions_data is not None:
        return emotions_data
    if emotions_data_path:
        with open(emotions_data_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    # 2) config 모듈 통해 로드(캐시 함수가 있으면 사용)
    try:
        cfg = _import_config_module()
        path = getattr(cfg, "EMOTIONS_JSON_PATH", os.getenv("EMOTIONS_JSON_PATH", "EMOTIONS.json"))
        load_fn = getattr(cfg, "load_emotions_json", None) or getattr(cfg, "_load_emotions_cached", None)
        if callable(load_fn):
            return load_fn(path)
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logging.warning(f"[pattern_extractor] emotions 데이터 로드 실패: {e}")
        return {}

# =============================================================================
# 성능 프로파일링 (초경량 타이머)
# =============================================================================
from contextlib import contextmanager

PROF_ON = os.environ.get("PE_PROF", "0") in ("1", "true")

@contextmanager
def _tick(tag: str, bucket: Dict[str, float]):
    """초경량 성능 측정 컨텍스트 매니저"""
    if not PROF_ON:
        yield
        return
    t0 = time.perf_counter()
    try:
        yield
    finally:
        ms = (time.perf_counter() - t0) * 1000
        bucket[tag] = bucket.get(tag, 0.0) + ms
        logger.info("[PERF] %s: %.1f ms", tag, ms)

# =============================================================================
# 정규식 컴파일 오버헤드 최적화 - 모듈 전역 컴파일
# =============================================================================
_RE_VARIANTS = re.compile(r"[\uFE0E\uFE0F]")
_RE_TOKENS = re.compile(r"[가-힣A-Za-z0-9]+")

# 문장 분할용 정규식 (연결어 목록을 상수로 처리)
_CONNECTIVES = ["그런데", "그러나", "하지만", "또한", "그런데", "즉", "또", "및", "등", "거의", "약간", "좀",
               "그럼에도", "그렇지만", "그래도", "그나마", "그와는", "또는", "혹은", "그외에", "그러므로", "그러고", "그리고", 
               "그래서", "게다가", "따라서", "결국", "마침내", "점점", "나중에", "이후", "그러면서", "뿐만 아니라"]
_RE_CONNECTIVES = re.compile(rf"\s*(?:{'|'.join(map(re.escape, _CONNECTIVES))})\s*")
_RE_PUNCT_ENDS = re.compile(r"([.?!]+)\s*")

# normalize_keyword용 정규식들
_RE_PUNCT_SUFFIX = re.compile(r"[.,!?…~:;\"'`()\[\]{}·—–/\\|]+$")
_RE_HA_ENDINGS = [
    "하였습니다만", "하였습니다요", "하였습니다.", "하였습니다",
    "합니다만", "합니다", "했습니다만", "했습니다", "했습니",
    "하여서", "하여", "하면서", "하며", "하는", "하기", "하게", "하면", "하니",
    "하였다", "했다", "해요", "하다",
]
_RE_GENERIC_ENDINGS = [
    "였습니다만", "였습니다요", "였습니다.", "였습니다",
    "입니다만", "입니다요", "입니다", "입니다.",
    "습니다만", "습니다", "습니",
    "이었다", "였다", "였네", "였지", "였어", "였니",
    "예요", "아요", "어요", "죠", "네", "다",
]
_RE_PRETERITE = ("였습니", "했습니", "였", "했", "았", "었", "였", "겠")

# =============================================================================
# 토큰화 중복 제거용 LRU 캐시 (정규식 최적화 적용)
# =============================================================================
# 운영 환경에서는 캐시 크기 축소로 메모리 압박 방지
_cache_size = 4096 if (os.getenv("EA_PROFILE") == "prod" or os.getenv("RENDER_DEPLOYMENT") or os.getenv("PRODUCTION_MODE") == "1") else 8192
@lru_cache(maxsize=_cache_size)
def _tok_cache_regex(text: str) -> Tuple[str, ...]:
    """Regex 기반 토큰화 결과 캐싱 (미리 컴파일된 정규식 사용)"""
    text = _RE_VARIANTS.sub("", text or "")
    return tuple(_RE_TOKENS.findall(text))
from logging.handlers import RotatingFileHandler

from concurrent.futures import ThreadPoolExecutor, as_completed

# transformers 의존성 가드 (미설치 환경에서도 모듈 임포트 가능)
try:
    from transformers import AutoModel, AutoTokenizer
except Exception:
    AutoModel = AutoTokenizer = None

# =============================================================================
# 실행기 선택 헬퍼 (PATCH D1)
# =============================================================================
def _choose_executor():
    """워커 수 최적화 (스레드 과다 문제 해결)"""
    cpu_count = os.cpu_count() or 4
    
    # 운영 환경에서는 스레드 수를 제한 (CPU×2 또는 CPU 중 작은 값)
    # Kiwi 미설치 환경에서는 더욱 제한
    try:
        import kiwipiepy
        kiwi_available = True
    except ImportError:
        kiwi_available = False
    
    # Windows/WSL에서 스레드 폭주 억제: CPU×2 정도로 제한
    if kiwi_available:
        # Kiwi 사용 가능: CPU×2 또는 CPU 중 작은 값
        max_workers = min(cpu_count, cpu_count * 2)
    else:
        # Kiwi 미설치: CPU 수만큼만 사용
        max_workers = cpu_count
    
    # Windows 환경에서 추가 제한 (깜빡임/컨텍스트 스위칭 억제)
    if os.name == 'nt':  # Windows
        max_workers = min(max_workers, 8)  # Windows에서는 최대 8개로 제한
    
    # 환경변수로 오버라이드 가능
    max_workers = int(os.environ.get("PE_MAX_WORKERS", str(max_workers)))
    
    return ThreadPoolExecutor(max_workers=max_workers)
try:
    from kiwipiepy import Kiwi
except ImportError:
    Kiwi = None


# =============================================================================
# Logger 설정 (PATCH A: ultra-light logging)
# =============================================================================
def setup_logger() -> logging.Logger:
    logger = logging.getLogger(__name__)
    logger.propagate = False
    
    FAST = os.environ.get("PE_FAST_LOG", "0").lower() in ("1", "true", "yes")
    logger.setLevel(logging.WARNING if FAST else logging.DEBUG)
    
    # 기본: 콘솔 로깅 비활성화 (PE_CONSOLE_LOG=1로 활성화 가능)
    if os.environ.get("PE_CONSOLE_LOG", "0") not in ("0", "false", "no"):
        try:
            ch = logging.StreamHandler()
            ch.setLevel(logging.WARNING if FAST else logging.INFO)
            ch.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s',
                                              '%Y-%m-%d %H:%M:%S'))
            logger.addHandler(ch)
        except Exception:
            pass
    
    # Opt-in: 파일 로깅 (PE_FILE_LOG=1)
    if os.environ.get("PE_FILE_LOG", "0") in ("1", "true", "yes"):
        try:
            from logging.handlers import RotatingFileHandler
            # 통합 로그 관리자 사용 (날짜별 폴더)
            try:
                from log_manager import get_log_manager
                log_manager = get_log_manager()
                log_dir = str(log_manager.get_log_dir("emotion_analysis", use_date_folder=True))
            except ImportError:
                # 폴백: 기존 방식 (날짜별 폴더 추가)
                from datetime import datetime
                current_dir = os.path.dirname(os.path.abspath(__file__))
                base_log_dir = os.path.join(current_dir, 'logs')
                today = datetime.now().strftime("%Y%m%d")
                log_dir = os.path.join(base_log_dir, today)
                os.makedirs(log_dir, exist_ok=True)
            fh = RotatingFileHandler(os.path.join(log_dir, 'pattern_extractor.log'),
                                     maxBytes=10 * 1024 * 1024, backupCount=5, encoding='utf-8')
            fh.setLevel(logging.WARNING if FAST else logging.DEBUG)
            fh.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s [%(filename)s:%(lineno)d] - %(message)s',
                                              datefmt='%Y-%m-%d %H:%M:%S'))
            logger.addHandler(fh)
        except Exception:
            pass
    
    # 핸들러가 없으면 기본 콘솔 핸들러 추가
    if not logger.handlers:
        try:
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            ch.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
            logger.addHandler(ch)
        except Exception:
            logger.addHandler(logging.NullHandler())

    # 시끄러운 서드파티 조용히
    quiet = logging.ERROR if FAST else logging.WARNING
    for name in ("transformers", "urllib3", "kiwipiepy", "torch"):
        logging.getLogger(name).setLevel(quiet)

    return logger
logger = setup_logger()
logger.info("PatternExtractor 로깅 시작")


# =============================================================================
# PatternContextAnalyzer (final, data-driven & hardened)
# =============================================================================
# 전역 PatternContextAnalyzer 캐시
_GLOBAL_PATTERN_ANALYZER_CACHE: Optional['PatternContextAnalyzer'] = None
_GLOBAL_PATTERN_ANALYZER_LOCK = threading.Lock()

# 전역 Kiwi 인스턴스 캐시
_GLOBAL_KIWI_CACHE: Optional[Any] = None
_GLOBAL_KIWI_LOCK = threading.Lock()
_GLOBAL_KIWI_ATTEMPTED = False

class PatternContextAnalyzer:
    _ENDING_FRAG = re.compile(r"(?:[가-힣]*)(?:습니|입니|합니|했습|했었)(?:다)?$")

    @classmethod
    def _get_cached_emotions_data(cls) -> Dict[str, Any]:
        """전역 EMOTIONS 데이터 캐시에서 데이터 반환"""
        # 1차: src.data_utils의 전역 캐시 사용
        try:
            from src.data_utils import get_global_emotions_data
            data = get_global_emotions_data()
            if data:
                return data
        except Exception:
            pass
        
        # 2차: src.config 직접 로드
        try:
            from src.config import _load_emotions_cached
            return _load_emotions_cached()
        except Exception:
            pass
        
        # 3차: 루트 패키지 폴백 (emotion_analysis.data_utils)
        try:
            import emotion_analysis.data_utils as data_utils
            data = getattr(data_utils, 'get_global_emotions_data', lambda: None)()
            if data:
                return data
        except Exception:
            pass
        
        # 4차: 루트 패키지 config 폴백
        try:
            import emotion_analysis.config as config
            return getattr(config, '_load_emotions_cached', lambda: {})()
        except Exception:
            pass
        
        # 최종: 프로덕션 환경에서는 패키지 레이아웃 문제 시 즉시 실패
        try:
            from config import EA_PROFILE, RENDER_DEPLOYMENT
            is_production = EA_PROFILE == "prod" or RENDER_DEPLOYMENT or os.getenv("PRODUCTION_MODE", "0") == "1"
            if is_production:
                raise RuntimeError("패키지 레이아웃 문제: 모든 패키지 경로에서 EMOTIONS 데이터를 찾을 수 없습니다")
        except Exception:
            pass
        
        return {}

    @classmethod
    def get_cached_kiwi(cls) -> Optional[Any]:
        """전역 Kiwi 인스턴스 반환"""
        global _GLOBAL_KIWI_CACHE, _GLOBAL_KIWI_ATTEMPTED
        with _GLOBAL_KIWI_LOCK:
            if _GLOBAL_KIWI_CACHE is not None:
                return _GLOBAL_KIWI_CACHE
            
            if not _GLOBAL_KIWI_ATTEMPTED:
                _GLOBAL_KIWI_ATTEMPTED = True
                try:
                    from kiwipiepy import Kiwi
                    _GLOBAL_KIWI_CACHE = Kiwi()
                    logger.info("전역 Kiwi 인스턴스 초기화 완료")
                except Exception as exc:
                    logger.warning("전역 Kiwi 초기화 실패: %s", exc)
                    _GLOBAL_KIWI_CACHE = None
            
            return _GLOBAL_KIWI_CACHE

    @classmethod
    def get_cached_instance(cls, kiwi_instance: Optional[Any] = None) -> 'PatternContextAnalyzer':
        """캐시된 PatternContextAnalyzer 인스턴스 반환"""
        global _GLOBAL_PATTERN_ANALYZER_CACHE
        with _GLOBAL_PATTERN_ANALYZER_LOCK:
            if _GLOBAL_PATTERN_ANALYZER_CACHE is None:
                emotions_data = cls._get_cached_emotions_data()
                _GLOBAL_PATTERN_ANALYZER_CACHE = cls(emotions_data, kiwi_instance)
            return _GLOBAL_PATTERN_ANALYZER_CACHE

    def __init__(self, emotions_data: Dict[str, Any], kiwi_instance: Optional[Any] = None):
        # 전역 캐시에서 emotions_data 가져오기
        if emotions_data is None or not emotions_data:
            emotions_data = self._get_cached_emotions_data()
        self.emotions_data = emotions_data or {}
        
        # 전역 Kiwi 인스턴스 사용
        if kiwi_instance is None:
            kiwi_instance = self.get_cached_kiwi()
        self.kiwi = kiwi_instance
        self._kiwi_attempted = False  # lazy init 1회 시도 여부
        self._kiwi_cache = None       # per-instance LRU cache 함수
        self.context_window: int = 5

        # 구성값(환경변수로 튜닝)
        self.cfg = {
            "MIN_DELTA": float(os.environ.get("PE_MIN_DELTA", "0.05")),
            "ID_CHANGE_ONLY": os.environ.get("PE_ID_CHANGE_ONLY", "1").lower() in ("1", "true", "yes"),
            "MAX_TRANS_PER_TEXT": int(os.environ.get("PE_MAX_TRANS_PER_TEXT", "6")),
            "TOPK_TEMPORAL": int(os.environ.get("PE_TOPK_TEMPORAL", "20")),
            "TOPK_TRANSITION": int(os.environ.get("PE_TOPK_TRANSITION", "20")),
            "TOPK_INTENSITY": int(os.environ.get("PE_TOPK_INTENSITY", "20")),
            "TOPK_CONTEXT": int(os.environ.get("PE_TOPK_CONTEXT", "25")),
            "TOPK_CONTEXT_PER_SENT": int(os.environ.get("PE_TOPK_CONTEXT_PER_SENT", "5")),
        }
        self.cfg["ALLOW_FALLBACK_LEX"] = os.environ.get("PE_ALLOW_FALLBACK_LEX", "0").strip().lower() in ("1", "true")
        self.calibrator = None

        self.weights = {
            "keyword_base": 2.0,
            "keyword_neutral": 1.0,
            "idf_a": 0.85,
            "idf_b": 0.30,
            "pos_hit": 1.5,
            "neg_hit": -1.0,
            "amp_mul": 1.5,
            "dim_mul": 0.7,
            "guard_pos_scene_neg": 0.45,
            "guard_neg_over_pos": 0.70,
            "relief_gratitude": 1.2,
            "relief_light": 0.6,
            "time_ctx_bonus": 0.03,
            "situation_ctx_bonus": 0.03,
            "overlap_penalty_alpha": 0.35,
        }

        scoring_cfg = (((self.emotions_data.get("analysis_modules") or {}).get("pattern_extractor") or {}).get("scoring") or {})
        if isinstance(scoring_cfg, dict):
            weights_cfg = scoring_cfg.get("weights") if isinstance(scoring_cfg.get("weights"), dict) else scoring_cfg
            for key, value in (weights_cfg or {}).items():
                if key in self.weights:
                    try:
                        self.weights[key] = float(value)
                    except (TypeError, ValueError):
                        pass

        for key in list(self.weights.keys()):
            env_key = f"PE_W_{key.upper()}"
            env_val = os.environ.get(env_key)
            if env_val is not None:
                try:
                    self.weights[key] = float(env_val)
                except ValueError:
                    pass

        # 일반 언어 표지어(라벨 비의존): 전이/시간/강도 관련
        self.emotion_transitions = {
            "emotion_onset": ["처음으로","시작하여","갑자기","드디어","마침내","문득","돌연","불현듯","홀연히","불쑥","느닷없이","예기치 않게","뜻밖에"],
            "emotion_increase": ["점점","더욱","점차","한층","배가 되어","증폭되어","고조되어","강렬해져"],
            "emotion_decrease": ["조금씩","서서히","차츰","완화되며","누그러지며","가라앉으며","쇠퇴하며","약화되며"],
            "emotion_shift":    ["그러다가","하지만","그런데","반면에","오히려","그럼에도","대조적으로","그러나","그에 비해","반대로"],
            "emotion_continuation": ["계속","여전히","변함없이","지속적으로","꾸준히","줄곧","시종일관"],
            "emotion_termination":  ["결국","마침내","끝내","드디어","마지막으로","종국에는","완결되어","마무리되어","종결되어"],
        }
        self.temporal_markers = {
            "순차": ["먼저","그다음","이어서","마지막으로","우선","첫째로","둘째로","셋째로","차례로","순서대로","초기에","중간에","최종적으로"],
            "지속": ["계속","내내","종일","줄곧","항상","영원히","끊임없이","연속적으로","지속적으로","한결같이","시종일관"],
            "빈도": ["자주","가끔","항상","때때로","이따금","간혹","드물게","빈번히","주기적으로","반복적으로","종종","수시로"],
            "시점": ["순간","때","무렵","즈음","시기에","찰나에","시점에","당시에","그때에","순식간","잠시","잠깐","찰나"],
            "변화": ["점점","차츰","서서히","급격히","갑자기","천천히","완만하게","급속도로","격변하여","극적으로"],
        }
        self.extended_temporal_patterns = {
            "pre_dawn": ["새벽","한밤중","동틀녘"],
            "morning": ["아침","오전","이른아침","늦은아침"],
            "noon": ["정오","점심"],
            "afternoon": ["오후","이른오후","늦은오후"],
            "evening": ["저녁","이른저녁","늦은저녁"],
            "night": ["밤","이른밤","늦은밤"],
        }

        # 라벨 → 내부 맵
        self.emotion_categories: Dict[str, Dict[str, Any]] = {}
        self._build_emotion_category_map()
        self._token2eids: Dict[str, Set[str]] = defaultdict(set)
        for emotion_id, info in self.emotion_categories.items():
            node = info.get("data_ref", {}) or {}
            bag: Set[str] = set()
            profile = (node.get("emotion_profile", {}) or {})
            bag.update(profile.get("core_keywords", []) or [])
            bag.update(profile.get("synonyms", []) or [])
            info_meta = info.get("metadata", {}) or {}
            bag.update(info_meta.get("core_keywords", []) or [])
            bag.update(info_meta.get("synonyms", []) or [])
            ctx = (node.get("context_patterns", {}) or {})
            for situation in (ctx.get("situations", {}) or {}).values():
                if not isinstance(situation, Mapping):
                    continue
                bag.update(situation.get("keywords", []) or [])
                bag.update(situation.get("variations", []) or [])
                for example in (situation.get("examples", []) or []):
                    for token in self._tokenize(example):
                        if token:
                            bag.add(token)
            ling = (node.get("linguistic_patterns", {}) or {})
            for phrase in ling.get("key_phrases", []) or []:
                pattern = (phrase.get("pattern") or "")
                if pattern:
                    bag.add(pattern.replace("[XXX]", "").replace("[XX]", ""))
            normalized_terms = {self._normalize_keyword(x) for x in bag if x}
            for normalized in normalized_terms:
                if normalized:
                    self._token2eids[normalized].add(emotion_id)
        self.labeled_transition_rules: List[Dict[str, Any]] = []
        self._build_labeled_transition_rules()

        conn_candidates = set(_CONNECTIVES)
        extra_transitions = self.emotions_data.get("transition_phrases")
        if extra_transitions:
            for phrase in self._flatten_strs(extra_transitions):
                if isinstance(phrase, str) and phrase.strip():
                    conn_candidates.add(phrase.strip())
        for rule in self.labeled_transition_rules or []:
            triggers = rule.get("triggers", []) or []
            for trigger in triggers:
                if isinstance(trigger, str) and trigger.strip():
                    conn_candidates.add(trigger.strip())
        shift_terms = self.emotion_transitions.get("emotion_shift")
        if shift_terms:
            for shift in self._flatten_strs(shift_terms):
                if isinstance(shift, str) and shift.strip():
                    conn_candidates.add(shift.strip())
        pattern_terms = [re.escape(term) for term in sorted({term for term in conn_candidates if term})]
        pattern_body = "|".join(pattern_terms)
        self._re_connectives_dyn = re.compile(rf"\s*(?:{pattern_body})\s*") if pattern_body else _RE_CONNECTIVES

        # ★ 데이터 기반 전역 렉시콘/강도 수정자 구축
        self.scene_tone_lex = self._build_scene_tone_lexicons()
        self.global_intensity_mods = self._build_global_intensity_modifiers()

        # ★ 전역 불용어(라벨 전수 스캔) / 접두 부정 복합어 / 중립 명사
        stopwords = self._build_global_stopwords()
        self.stopwords = stopwords
        self.global_stopwords = stopwords
        self.neg_prefix_terms = self._build_neg_prefix_terms()
        self.neutral_nouns = self._build_neutral_nouns()
        self.label_vocab = self._build_label_vocab()

        self.token_idf = self._build_token_idf()
        self.time_keywords, self.situation_keywords = self._collect_time_situation_from_bone(self.emotions_data)
        self.time_keyword_norms = {self._norm(x) for x in self.time_keywords}
        self.situation_keyword_norms = {self._norm(x) for x in self.situation_keywords}
        self._total_keyword_count = len(self.label_vocab or set())
        self._total_context_count = len(set(self.situation_keywords or []))
        temporal_base = sum(len(v) for v in (self.temporal_markers or {}).values()) if isinstance(self.temporal_markers, Mapping) else 0
        self._total_time_marker_count = temporal_base + len(set(self.time_keywords or []))

        logger.info("PatternContextAnalyzer: 데이터 기반 전역 렉시콘/지표/불용어/접두사/중립명사 준비 완료")

    def inject_calibrator(self, calibrator: Optional[Any]) -> None:
        """Attach optional calibrator used to adjust emotion scores."""
        self.calibrator = calibrator

    # --------------------- utils ---------------------
    @staticmethod
    @lru_cache(maxsize=_cache_size)
    def _norm(s: str) -> str:
        try:
            return unicodedata.normalize("NFC", s or "").strip().lower()
        except Exception:
            return (s or "").strip().lower()


    def _tokenize(self, text: Any) -> List[str]:
        """Kiwi 사용시 lazy init + LRU 캐시, 실패하면 regex로 폴백 (PATCH B2)"""
        if not text:
            return []

        # 입력 normalize 부분은 기존 로직 그대로 유지
        if isinstance(text, (list, tuple, set)):
            parts = [item if isinstance(item, str) else str(item) for item in text]
            text = " ".join(parts)
        elif isinstance(text, Mapping):
            text = " ".join([v if isinstance(v, str) else str(v) for v in text.values()])
        else:
            text = str(text)
        text = text.strip()
        if not text:
            return []

        tokens: List[str] = []

        # Lazy Kiwi init (한 번만 시도)
        if self.kiwi is None and not self._kiwi_attempted and _want_kiwi(default=True) and Kiwi is not None:
            self._kiwi_attempted = True
            try:
                self.kiwi = Kiwi()
                logger.info("Kiwi lazy init 완료")
            except Exception as exc:
                self.kiwi = None
                logger.warning("Kiwi lazy init 실패: %s", exc)

        # Kiwi 경로: per-instance LRU
        if self.kiwi is not None:
            if self._kiwi_cache is None:
                # 운영 환경에서는 Kiwi 캐시 크기 축소
                kiwi_cache_size = 2048 if (os.getenv("EA_PROFILE") == "prod" or os.getenv("RENDER_DEPLOYMENT") or os.getenv("PRODUCTION_MODE") == "1") else int(os.environ.get("PE_KIWI_CACHE", "4096"))
                @lru_cache(maxsize=kiwi_cache_size)
                def _cache_line(s: str) -> Tuple[str, ...]:
                    return tuple(self._norm(getattr(t, 'form', '') or '') for t in self.kiwi.tokenize(s))
                self._kiwi_cache = _cache_line
            try:
                tokens = [t for t in self._kiwi_cache(text) if t]
            except Exception as exc:
                logger.debug("Kiwi tokenization failed for '%s': %s", text[:30], exc)
                tokens = []

        # 폴백: 캐시된 regex
        if not tokens:
            tokens = [self._norm(tok) for tok in _tok_cache_regex(text) if tok]

        tokens = [tok for tok in tokens if tok and not self._is_end_frag(tok)]
        return tokens

    @staticmethod
    @lru_cache(maxsize=_cache_size)
    def _kw_pat(keyword: str) -> Optional[re.Pattern]:
        """
        한국어 조사/어미가 뒤에 붙는 경우(만족+으로/은/는/을/를/다/니다/습니다...)를
        오른쪽 경계로 '허용'하는 안전한 정규식.
        - 왼쪽: 문장 시작 or 공백/문장부호
        - 오른쪽: 문장 끝 or 공백/문장부호 or 대표적인 조사/어미 패턴
        """
        if not keyword:
            return None
        esc = re.escape(PatternContextAnalyzer._norm(keyword))

        # 공백/문장부호(왼쪽 경계용)
        punct = r"\s\.,!?:;\"'`\)\(\[\]\{\}…~\-_/\\|·—–"

        # 대표 조사/어미(오른쪽 경계 허용)
        # 필요시 추가: '처럼','하며','하고','하기','했다' 등
        suf = r"(?:에서|으로|로|에게|보다|까지|부터|만|도|의|와|과|은|는|이|가|을|를|로써|로서|다|니다|습니다)"

        left = rf"(?:(?<=^)|(?<=[{punct}]))"
        right = rf"(?:(?=$)|(?=[{punct}])|(?={suf}))"
        try:
            return re.compile(left + esc + right, flags=re.UNICODE | re.IGNORECASE)
        except re.error:
            return None

    @staticmethod
    @lru_cache(maxsize=_cache_size)
    def _fuzzy_pat(norm_phrase: str) -> Optional[re.Pattern]:
        if not norm_phrase:
            return None
        try:
            fuzzy = re.sub(r"\s+", r".{0,2}", re.escape(norm_phrase))
            return re.compile(fuzzy, flags=re.UNICODE | re.IGNORECASE)
        except re.error:
            return None

    def _get_emotion_node_by_id(self, emotion_id: str) -> Optional[Dict[str, Any]]:
        for top_label, top_data in self.emotions_data.items():
            if (top_data.get("metadata", {}) or {}).get("emotion_id") == emotion_id:
                return top_data
            for _, sub in (top_data.get("sub_emotions", {}) or {}).items():
                if (sub.get("metadata", {}) or {}).get("emotion_id") == emotion_id:
                    return sub
        return None

    def _build_token_idf(self) -> Dict[str, float]:
        from collections import Counter
        df = Counter();
        N = max(1, len(self.emotion_categories))
        # 각 감정 노드별로 한 번만 카운트(문서빈도)
        for _, info in self.emotion_categories.items():
            node = info.get("data_ref", {}) or {}
            bags = []
            ep = (node.get("emotion_profile", {}) or {})
            bags += [ep.get("core_keywords", []) or [], ep.get("synonyms", []) or []]
            sa = (node.get("sentiment_analysis", {}) or {})
            bags += [sa.get("positive_indicators", []) or [], sa.get("negative_indicators", []) or []]
            ling = (node.get("linguistic_patterns", {}) or {})
            bags += [[(kp.get("pattern") or "").replace("[XXX]", "").replace("[XX]", "")] for kp in
                     (ling.get("key_phrases", []) or [])]
            for _, s in ((node.get("context_patterns", {}) or {}).get("situations", {}) or {}).items():
                bags += [s.get("keywords", []) or [], s.get("variations", []) or [], s.get("examples", []) or []]
            seen = set()
            for bag in bags:
                for x in (bag or []):
                    nx = self._normalize_keyword(x)
                    if nx and nx not in seen:
                        seen.add(nx);
                        df[nx] += 1
        # log 스케일(0~1 근처로 쓰기 좋게 노말라이즈)
        import math
        idf = {t: math.log((N + 1) / (c + 1)) / math.log(N + 1) for t, c in df.items()}
        return idf

    def _flatten_strs(self, value: Any) -> List[str]:
        if isinstance(value, str):
            return [value]
        if isinstance(value, (list, tuple, set)):
            out: List[str] = []
            for item in value:
                out.extend(self._flatten_strs(item))
            return out
        if isinstance(value, Mapping):
            out: List[str] = []
            for nested in value.values():
                out.extend(self._flatten_strs(nested))
            return out
        return []

    def _collect_time_situation_from_bone(self, emotions_data: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """EMOTIONS.json 구조에 맞춰 시간/상황 키워드 추출"""
        times: Set[str] = set()
        situs: Set[str] = set()

        if isinstance(emotions_data, Mapping):
            for emotion_key, emotion_data in emotions_data.items():
                if not isinstance(emotion_data, Mapping):
                    continue
                
                # 서브 감정들의 상황 키워드 추출
                sub_emotions = emotion_data.get("sub_emotions", {})
                if isinstance(sub_emotions, dict):
                    for sub_key, sub_data in sub_emotions.items():
                        if not isinstance(sub_data, Mapping):
                            continue
                        
                        # context_patterns.situations 구조에서 키워드 추출
                        context_patterns = sub_data.get("context_patterns", {})
                        if isinstance(context_patterns, dict):
                            situations = context_patterns.get("situations", {})
                            if isinstance(situations, dict):
                                for situation_key, situation_data in situations.items():
                                    if isinstance(situation_data, dict):
                                        # keywords 배열에서 키워드 추출
                                        keywords = situation_data.get("keywords", [])
                                        if isinstance(keywords, list):
                                            for kw in keywords:
                                                if isinstance(kw, str) and kw.strip():
                                                    situs.add(kw.strip())
                                        
                                        # variations 배열에서 키워드 추출
                                        variations = situation_data.get("variations", [])
                                        if isinstance(variations, list):
                                            for var in variations:
                                                if isinstance(var, str) and var.strip():
                                                    # 문장을 단어로 분리하여 키워드 추출
                                                    words = var.split()
                                                    for word in words:
                                                        if len(word) >= 2:  # 최소 2글자 이상
                                                            situs.add(word.strip())
                                        
                                        # description에서도 키워드 추출
                                        description = situation_data.get("description", "")
                                        if isinstance(description, str) and description.strip():
                                            words = description.split()
                                            for word in words:
                                                if len(word) >= 2:
                                                    situs.add(word.strip())

        # 시간 관련 키워드 (기본 패턴들)
        time_patterns = [
            "오늘", "어제", "내일", "지금", "이제", "그때", "언제", "항상", "자주", "가끔",
            "아침", "점심", "저녁", "밤", "새벽", "낮", "밤늦게", "이른", "늦은",
            "월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일",
            "1월", "2월", "3월", "4월", "5월", "6월", "7월", "8월", "9월", "10월", "11월", "12월",
            "봄", "여름", "가을", "겨울", "계절", "연말", "연초", "중순", "말", "초"
        ]
        times.update(time_patterns)

        normalize = lambda xs: sorted({self._norm(x) for x in xs if isinstance(x, str) and x.strip()})
        return normalize(times), normalize(situs)

    def _build_scene_tone_lexicons(self) -> Dict[str, Set[str]]:
        lexicons = {"positive": set(), "negative": set()}
        try:
            sources = []
            modules = (self.emotions_data.get("analysis_modules") or {}).get("pattern_extractor") or {}
            scene_cfg = modules.get("scene_tone_lexicons") if isinstance(modules, Mapping) else None
            if isinstance(scene_cfg, Mapping):
                sources.append(scene_cfg)
            nodes = []
            if isinstance(self.emotions_data, Mapping):
                nodes.extend(self.emotions_data.values())
                for top in self.emotions_data.values():
                    if isinstance(top, Mapping):
                        nodes.extend((top.get("sub_emotions", {}) or {}).values())
            for node in nodes:
                if not isinstance(node, Mapping):
                    continue
                senti = node.get("sentiment_analysis") or {}
                if isinstance(senti, Mapping):
                    sources.append(senti)
            for src in sources:
                pos_items = src.get("positive") or src.get("positive_indicators") or []
                neg_items = src.get("negative") or src.get("negative_indicators") or []
                for item in pos_items or []:
                    norm = self._normalize_keyword(item)
                    if norm:
                        lexicons["positive"].add(norm)
                for item in neg_items or []:
                    norm = self._normalize_keyword(item)
                    if norm:
                        lexicons["negative"].add(norm)
            allow_fallback = self.cfg.get("ALLOW_FALLBACK_LEX", False)
            if not lexicons["positive"] and allow_fallback:
                lexicons["positive"].update({"기쁘다", "행복하다", "즐겁다", "감사하다", "뿌듯하다"})
            if not lexicons["negative"] and allow_fallback:
                lexicons["negative"].update({"슬프다", "화나다", "불안하다", "걱정하다", "짜증나다"})
            return lexicons
        except Exception:
            logger.exception("_build_scene_tone_lexicons 실패")
            return {"positive": set(), "negative": set()}

    def _build_global_intensity_modifiers(self) -> Dict[str, Set[str]]:
        mods = {"amplifiers": set(), "diminishers": set(), "negators": set()}
        try:
            nodes = []
            if isinstance(self.emotions_data, Mapping):
                nodes.extend(self.emotions_data.values())
                for top in self.emotions_data.values():
                    if isinstance(top, Mapping):
                        nodes.extend((top.get("sub_emotions", {}) or {}).values())
            for node in nodes:
                if not isinstance(node, Mapping):
                    continue
                senti = node.get("sentiment_analysis") or {}
                if isinstance(senti, Mapping):
                    imods = senti.get("intensity_modifiers") or {}
                    if isinstance(imods, Mapping):
                        for key in ("amplifiers", "diminishers", "negators"):
                            for word in imods.get(key, []) or []:
                                norm = self._normalize_keyword(word)
                                if norm:
                                    mods[key].add(norm)
            allow_fallback = self.cfg.get("ALLOW_FALLBACK_LEX", False)
            if not mods["amplifiers"] and allow_fallback:
                mods["amplifiers"].update({"매우", "정말", "너무", "굉장히", "엄청"})
            if not mods["diminishers"] and allow_fallback:
                mods["diminishers"].update({"조금", "약간", "다소", "살짝", "그럭저럭"})
            if not mods["negators"] and allow_fallback:
                mods["negators"].update({"아니다", "없다", "못하다", "안", "부정"})
            return mods
        except Exception:
            logger.exception("_build_global_intensity_modifiers 실패")
            return {"amplifiers": set(), "diminishers": set(), "negators": set()}

    def _build_emotion_category_map(self) -> None:
        try:
            categories: Dict[str, Dict[str, Any]] = {}
            if not isinstance(self.emotions_data, Mapping):
                self.emotion_categories = categories
                return

            def register(node: Dict[str, Any], primary_label: str, name: Optional[str], fallback_id: str) -> None:
                meta = node.get("metadata", {}) or {}
                emotion_id = str(meta.get("emotion_id") or fallback_id or name or primary_label)
                categories[emotion_id] = {
                    "primary": primary_label,
                    "sub_emotion_name": name,
                    "metadata": meta,
                    "data_ref": node,
                }

            for primary, top_data in (self.emotions_data or {}).items():
                if not isinstance(top_data, Mapping):
                    continue
                primary_label = str(top_data.get("metadata", {}).get("primary_category") or primary)
                top_meta = top_data.get("metadata", {}) or {}
                top_name = top_meta.get("emotion_name") or top_meta.get("name") or str(primary)
                top_id = str(top_meta.get("emotion_id") or top_name or primary_label)
                register(top_data, primary_label, top_name, top_id)
                for sub_name, sub_data in (top_data.get("sub_emotions", {}) or {}).items():
                    if not isinstance(sub_data, Mapping):
                        continue
                    sub_meta = sub_data.get("metadata", {}) or {}
                    sub_label = sub_meta.get("emotion_name") or sub_name
                    fallback_id = str(sub_meta.get("emotion_id") or f"{top_id}-{sub_label}")
                    register(sub_data, primary_label, sub_label, fallback_id)
            self.emotion_categories = categories
        except Exception:
            logger.exception("_build_emotion_category_map 실패")
            self.emotion_categories = {}

    def _build_labeled_transition_rules(self) -> None:
        rules: List[Dict[str, Any]] = []
        seen: Set[str] = set()
        try:
            if not isinstance(self.emotions_data, Mapping):
                self.labeled_transition_rules = rules
                return

            def collect(node: Dict[str, Any], source_id: str) -> None:
                candidates: List[Any] = []
                for key in ("transition_rules", "transition_patterns"):
                    val = node.get(key)
                    if isinstance(val, list):
                        candidates.append(val)
                profile = node.get("emotion_profile", {}) or {}
                for key in ("transition_rules", "transition_patterns"):
                    val = profile.get(key)
                    if isinstance(val, list):
                        candidates.append(val)
                context = node.get("context_patterns", {}) or {}
                if isinstance(context, Mapping):
                    val = context.get("transition_rules")
                    if isinstance(val, list):
                        candidates.append(val)
                for cand in candidates:
                    for rule in cand or []:
                        if not isinstance(rule, Mapping):
                            continue
                        frm = str(rule.get("from_emotion") or "")
                        to = str(rule.get("to_emotion") or "")
                        triggers = tuple(sorted(rule.get("triggers", []) or []))
                        key = (source_id, frm, to, triggers)
                        if key in seen:
                            continue
                        seen.add(key)
                        r = dict(rule)
                        r.setdefault("source_emotion", source_id)
                        r.setdefault("triggers", list(triggers))
                        rules.append(r)

            for primary, top_data in (self.emotions_data or {}).items():
                if not isinstance(top_data, Mapping):
                    continue
                top_meta = top_data.get("metadata", {}) or {}
                top_id = str(top_meta.get("emotion_id") or primary)
                collect(top_data, top_id)
                for sub_name, sub_data in (top_data.get("sub_emotions", {}) or {}).items():
                    if not isinstance(sub_data, Mapping):
                        continue
                    sub_meta = sub_data.get("metadata", {}) or {}
                    sub_id = str(sub_meta.get("emotion_id") or f"{top_id}-{sub_name}")
                    collect(sub_data, sub_id)
            self.labeled_transition_rules = rules
        except Exception:
            logger.exception("_build_labeled_transition_rules 실패")
            self.labeled_transition_rules = rules

    def _build_label_vocab(self) -> Set[str]:
        vocab: Set[str] = set()

        def add(text: str):
            for t in self._tokenize(text or ""):
                nt = self._normalize_keyword(t)
                if nt and len(nt) >= 2:
                    vocab.add(nt)

        for _, top in self.emotions_data.items():
            ep = top.get("emotion_profile", {}) or {}
            for x in (ep.get("core_keywords", []) or []) + (ep.get("synonyms", []) or []):
                add(x)

            sa = top.get("sentiment_analysis", {}) or {}
            for x in (sa.get("positive_indicators", []) or []) + (sa.get("negative_indicators", []) or []):
                add(x)

            ling = top.get("linguistic_patterns", {}) or {}
            for kp in ling.get("key_phrases", []) or []:
                add((kp.get("pattern") or "").replace("[XXX]", "").replace("[XX]", ""))

            ctx = top.get("context_patterns", {}) or {}
            for s in (ctx.get("situations", {}) or {}).values():
                # ★ examples 는 제외합니다
                for x in (s.get("keywords", []) or []) + (s.get("variations", []) or []):
                    add(x)
                cc = s.get("core_concept")
                if cc:
                    add(cc)

            for _, sub in (top.get("sub_emotions", {}) or {}).items():
                ep = sub.get("emotion_profile", {}) or {}
                for x in (ep.get("core_keywords", []) or []) + (ep.get("synonyms", []) or []):
                    add(x)

                sa = sub.get("sentiment_analysis", {}) or {}
                for x in (sa.get("positive_indicators", []) or []) + (sa.get("negative_indicators", []) or []):
                    add(x)

                ling = sub.get("linguistic_patterns", {}) or {}
                for kp in ling.get("key_phrases", []) or []:
                    add((kp.get("pattern") or "").replace("[XXX]", "").replace("[XX]", ""))

                ctx = sub.get("context_patterns", {}) or {}
                for s in (ctx.get("situations", {}) or {}).values():
                    for x in (s.get("keywords", []) or []) + (s.get("variations", []) or []):
                        add(x)
                    cc = s.get("core_concept")
                    if cc:
                        add(cc)

        return vocab

    # ---------- 불용어/정규화/콘텐츠 토큰 ----------
    def _build_global_stopwords(self) -> Set[str]:
        """전역 감정 라벨 기반 불용어 구성(조사/접사/파편 포함)."""
        from collections import defaultdict

        base = {
            "입니다", "습니", "입니", "합니", "했습", "했었", "하였다", "했다", "였다", "이었다",
            "그리고", "그러나", "하지만", "또한", "그런데", "즉", "또", "및", "등", "거의", "약간", "좀",
        }
        bone_sw = set((self.emotions_data.get("sentiment_analysis") or {}).get("stopwords", []))

        def _tok_stream():
            for _, top in self.emotions_data.items():
                for bag in [
                    (top.get("emotion_profile", {}) or {}).get("core_keywords", []),
                    (top.get("emotion_profile", {}) or {}).get("synonyms", []),
                    (top.get("sentiment_analysis", {}) or {}).get("positive_indicators", []),
                    (top.get("sentiment_analysis", {}) or {}).get("negative_indicators", []),
                ]:
                    for x in (bag or []):
                        for t in self._tokenize(x):
                            yield t
                for _, sub in (top.get("sub_emotions", {}) or {}).items():
                    for bag in [
                        (sub.get("emotion_profile", {}) or {}).get("core_keywords", []),
                        (sub.get("emotion_profile", {}) or {}).get("synonyms", []),
                        (sub.get("sentiment_analysis", {}) or {}).get("positive_indicators", []),
                        (sub.get("sentiment_analysis", {}) or {}).get("negative_indicators", []),
                    ]:
                        for x in (bag or []):
                            for t in self._tokenize(x):
                                yield t
                    for _, sinfo in ((sub.get("context_patterns", {}) or {}).get("situations", {}) or {}).items():
                        for x in (sinfo.get("keywords", []) or []) + (sinfo.get("variations", []) or []) + (sinfo.get("examples", []) or []):
                            for t in self._tokenize(x or ""):
                                yield t

        df = defaultdict(int)
        total_slots = len(self.emotion_categories) or 1
        for t in _tok_stream():
            df[t] += 1

        josa_like = {
            "은", "는", "이", "가", "을", "를", "의", "에", "께", "께서", "에서", "으로", "로", "와", "과", "랑", "이나",
            "하다", "하거나", "하는", "하며", "했던"
        }
        punct = set(list(".,!?~:;\"'`()[]{}<>?/\\|"))

        stop = set()
        for t, c in df.items():
            if (not t) or any(ch in punct for ch in t):
                stop.add(t)
                continue
            if len(t) <= 1:
                stop.add(t)
                continue
            if c >= max(3, int(total_slots * 0.35)):
                stop.add(t)
                continue
            if t in josa_like or t.endswith(("에서", "으로", "로", "와", "과", "랑", "하고", "하며", "하면", "하여", "다면")):
                stop.add(t)
                continue

        normalized_base = {self._norm(w) for w in (base | bone_sw) if w}
        stop.update(normalized_base)
        stop.discard("")

        return stop

    def _build_neg_prefix_terms(self) -> Set[str]:
        """Collect negative-prefix-like tokens used to catch negated expressions."""
        terms: Set[str] = set()
        try:
            modules = (self.emotions_data.get('analysis_modules') or {}).get('pattern_extractor') or {}
            for key in ('neg_prefix_terms', 'negative_prefix_terms'):
                raw = modules.get(key)
                if raw:
                    for t in self._flatten_strs(raw):
                        nt = self._normalize_keyword(t)
                        if nt:
                            terms.add(nt)

            # 폴백 사용 여부는 전역 플래그(ALLOW_FALLBACK_LEX)에 따름 (기본값 True로 변경)
            allow_fallback = bool(self.cfg.get('ALLOW_FALLBACK_LEX', True))

            # 한글 기반의 안전한 기본 접두 부정어(모지바케 제거)
            fallback_defaults = {
                '안', '못', '아니', '불', '비', '무', '탈',
                '불-', '비-', '무-'
            }

            # EMOTIONS.json에 analysis_modules 섹션이 없으면 항상 fallback 사용
            if not terms:
                for default in fallback_defaults:
                    nt = self._normalize_keyword(default)
                    if nt:
                        terms.add(nt)
                if terms:
                    logger.info('PatternContextAnalyzer: using fallback negative prefix terms (EMOTIONS.json analysis_modules not found)')
        except Exception:
            logger.exception('_build_neg_prefix_terms ??')
        return terms

    def _build_neutral_nouns(self) -> Set[str]:
        """Aggregate neutral nouns so we can down-weight them during scoring."""
        neutrals: Set[str] = set()
        try:
            modules = (self.emotions_data.get('analysis_modules') or {}).get('pattern_extractor') or {}
            for key in ('neutral_nouns', 'neutral_keywords', 'neutral_terms'):
                raw = modules.get(key)
                if raw:
                    for t in self._flatten_strs(raw):
                        nt = self._normalize_keyword(t)
                        if nt:
                            neutrals.add(nt)
            nodes: List[Any] = []
            if isinstance(self.emotions_data, Mapping):
                nodes.extend(self.emotions_data.values())
                for top in self.emotions_data.values():
                    if isinstance(top, Mapping):
                        nodes.extend((top.get('sub_emotions', {}) or {}).values())
            for node in nodes:
                if not isinstance(node, Mapping):
                    continue
                ling = node.get('linguistic_patterns') or {}
                if isinstance(ling, Mapping):
                    for key in ('neutral_keywords', 'neutral_nouns'):
                        bag = ling.get(key)
                        if bag:
                            for t in self._flatten_strs(bag):
                                nt = self._normalize_keyword(t)
                                if nt:
                                    neutrals.add(nt)
                context = node.get('context_patterns') or {}
                if isinstance(context, Mapping):
                    situations = context.get('situations') or {}
                    if isinstance(situations, Mapping):
                        for s in situations.values():
                            if isinstance(s, Mapping):
                                bag = s.get('neutral_keywords')
                                if bag:
                                    for t in self._flatten_strs(bag):
                                        nt = self._normalize_keyword(t)
                                        if nt:
                                            neutrals.add(nt)

            # 한글 기반의 안전한 기본 중립 명사(모지바케 제거)
            fallback_defaults = {
                '상황', '상태', '사건', '이야기', '부분', '경험', '시간', '장면', '사실', '문제'
            }
            allow_fallback = bool(self.cfg.get('ALLOW_FALLBACK_LEX', True))
            
            # EMOTIONS.json에 analysis_modules 섹션이 없으면 항상 fallback 사용
            if not neutrals:
                for default in fallback_defaults:
                    nt = self._normalize_keyword(default)
                    if nt:
                        neutrals.add(nt)
                if neutrals:
                    logger.info('PatternContextAnalyzer: using fallback neutral nouns (EMOTIONS.json analysis_modules not found)')
        except Exception:
            logger.exception('_build_neutral_nouns ??')
        return neutrals


    def _normalize_keyword(self, kw: str) -> Optional[str]:
        if not kw:
            return None
        k = self._norm(kw)
        # 끝 문장부호 제거 (미리 컴파일된 정규식 사용)
        k = _RE_PUNCT_SUFFIX.sub("", k)

        # ① '하다' 계열(긴 → 짧은) - 전역 상수 사용
        for suf in sorted(set(_RE_HA_ENDINGS), key=len, reverse=True):
            if k.endswith(suf) and len(k) > len(suf) + 1:
                k = k[:-len(suf)]
                break

        # ② 일반 높임/시제/평서 어미 (긴 → 짧은) - 전역 상수 사용
        for suf in sorted(set(_RE_GENERIC_ENDINGS), key=len, reverse=True):
            if k.endswith(suf) and len(k) > len(suf) + 1:
                k = k[:-len(suf)]
                break

        # ②-b 과거/추정 잔여(엇/았/였/했/겠) 한 번 더 정리 - 전역 상수 사용
        for suf in sorted(set(_RE_PRETERITE), key=len, reverse=True):
            if k.endswith(suf) and len(k) > len(suf) + 1:
                k = k[:-len(suf)]
                break

        k = k.strip()
        # ③ 잔여 '하' 꼬리 정리(산책하 → 산책)
        if k.endswith("하") and len(k) >= 2:
            k = k[:-1]

        # ④ 너무 짧거나 숫자면 폐기
        if not k or len(k) <= 1 or re.fullmatch(r"\d+", k):
            return None
        return k

    def _is_end_frag(self, tok: str) -> bool:
        t = self._norm(tok or "")
        if not t:
            return True
        if len(t) <= 1:
            return True
        if t in getattr(self, "stopwords", set()):
            return True
        return bool(self._ENDING_FRAG.search(t))

    def _is_content_token(self, t: str) -> bool:
        """감성 불용어/조사/1글자 제외."""
        if not t:
            return False
        if self._is_end_frag(t):
            return False
        if re.fullmatch(r"\d+", t):
            return False
        if len(t) == 1:
            return False
        return True

    def _expand_token_forms(self, tokens: List[str]) -> Set[str]:
        """
        Kiwi가 없을 때를 대비한 간단 한국어 조사/어미 스트리핑 확장.
        ★ 변경점:
          - 확장으로 생긴 후보는 label_vocab(또는 전역 톤 렉시콘)에 없으면 버림
          - 조각 토큰(산책하/휩쓸리/가지 등) 제거 효과
        """
        josa_suffixes = [
            "으로써", "으로서", "으로도", "으로만", "으로", "로",
            "에게", "에서", "처럼", "보다", "까지", "부터", "마다", "밖에",
            "이라", "라", "이라서", "라서", "이자", "가요", "네요", "어요", "아요", "예요", "입니다",
            "을", "를", "은", "는", "이", "가", "와", "과", "도", "만", "의", "에",
            "로써", "로서", "로도", "로만",
            "다", "요", "네", "죠", "게", "고", "며", "면", "던", "든", "듯", "함", "함니다"
        ]
        out: Set[str] = set(tokens)  # 원문 토큰은 그대로 보존
        label_vocab = getattr(self, "label_vocab", set())
        pos_lex = self.scene_tone_lex.get("positive", set())
        neg_lex = self.scene_tone_lex.get("negative", set())

        for t in tokens:
            for suf in josa_suffixes:
                if t.endswith(suf) and len(t) > len(suf) + 1:
                    cand = t[: -len(suf)]
                    cn = self._normalize_keyword(cand)
                    # 라벨 어휘/전역 톤 렉시콘 중 하나라도 포함되어야 유지
                    if cn and not self._is_end_frag(cn) and (not label_vocab or (cn in label_vocab or cn in pos_lex or cn in neg_lex)):
                        out.add(cn)
        return out

    def _find_positions(self, text: str, phrase: str) -> List[int]:
        nt, ph = self._norm(text), self._norm(phrase)
        out: List[int] = []
        pat = self._kw_pat(ph)
        if pat:
            for m in pat.finditer(nt):
                out.append(m.start())
                if len(out) >= 5:
                    break
        if not out:
            rgx = self._fuzzy_pat(ph)
            if rgx:
                for m in rgx.finditer(nt):
                    out.append(m.start())
                    if len(out) >= 5:
                        break
        out.sort()
        dedup: List[int] = []
        for p in out:
            if not dedup or p - dedup[-1] > 2:
                dedup.append(p)
                if len(dedup) >= 5:
                    break
        return dedup

    @staticmethod
    @lru_cache(maxsize=1024)
    def _split_sentences_cached(text: str) -> Tuple[str, ...]:
        if not text:
            return tuple()
        # 미리 컴파일된 정규식 사용
        s = _RE_CONNECTIVES.sub(" ? ", (text or "").strip())
        s = _RE_PUNCT_ENDS.sub(r"\1 ? ", s)
        parts = [p.strip(" ,;:") for p in s.split("?") if p and p.strip()]
        return tuple(parts)

    def _split_sentences_ko(self, text: str) -> List[str]:
        if not text:
            return []
        pattern = getattr(self, '_re_connectives_dyn', _RE_CONNECTIVES)
        s = pattern.sub(' ? ', (text or '').strip())
        s = _RE_PUNCT_ENDS.sub(r"\1 ? ", s)
        return [p.strip(' ,;:') for p in s.split('?') if p and p.strip()]

    def analyze_context_window(self, sentences: List[str], position: int, window_size: Optional[int] = None) -> Dict[str, Any]:
        win = window_size or self.context_window
        start = max(0, position - win)
        end = min(len(sentences), position + win + 1)
        context_text = ". ".join(sentences[start:end]).strip()
        focus_sentence = sentences[position] if 0 <= position < len(sentences) else ""
        nt_focus = self._norm(focus_sentence)

        temporal_markers: List[Dict[str, Any]] = []
        transition_indicators: List[Dict[str, Any]] = []
        intensity_modifiers: List[Dict[str, Any]] = []
        intensity_patterns: List[Dict[str, Any]] = []

        ctx = {
            "before": sentences[start:position],
            "after": sentences[position + 1:end],
            "temporal_markers": temporal_markers,
            "transition_indicators": transition_indicators,
            "intensity_modifiers": intensity_modifiers,
            "emotion_intensities": {},
            "intensity_patterns": intensity_patterns,
        }

        for cat, markers in self.temporal_markers.items():
            for m in markers:
                marker_norm = self._norm(m)
                for pos in self._find_positions(nt_focus, marker_norm):
                    bonus = self.weights.get("time_ctx_bonus", 0.03)
                    temporal_markers.append({
                        "category": cat, "marker": m, "position": pos,
                        "text_span": nt_focus[max(0, pos - 10):pos + len(marker_norm) + 10],
                        "bonus": bonus,
                    })

        for t_type, patterns in self.emotion_transitions.items():
            for p in patterns:
                pat_norm = self._norm(p)
                for pos in self._find_positions(nt_focus, pat_norm):
                    bonus = self.weights.get("situation_ctx_bonus", 0.03)
                    transition_indicators.append({
                        "type": t_type, "pattern": p, "position": pos,
                        "text_span": nt_focus[max(0, pos - 10):pos + len(pat_norm) + 10],
                        "bonus": bonus,
                    })

        for w in self.time_keywords or []:
            w_norm = self._norm(w)
            for pos in self._find_positions(nt_focus, w_norm):
                span = nt_focus[max(0, pos - 10):pos + len(w_norm) + 10]
                temporal_markers.append({
                    "category": "label_time",
                    "marker": w,
                    "position": pos,
                    "text_span": span,
                })

        for w in self.situation_keywords or []:
            w_norm = self._norm(w)
            for pos in self._find_positions(nt_focus, w_norm):
                span = nt_focus[max(0, pos - 10):pos + len(w_norm) + 10]
                transition_indicators.append({
                    "type": "label_situation",
                    "pattern": w,
                    "position": pos,
                    "text_span": span,
                })

        for emo_key, emo_data in self.emotions_data.items():
            epi = (emo_data.get("emotion_profile", {}) or {}).get("intensity_levels", {}) or {}
            exs_map = (epi.get("intensity_examples", {}) or {})
            for level, exs in (exs_map.items() if isinstance(exs_map, dict) else []):
                matched = False
                for ex in exs or []:
                    hits = self._find_positions(nt_focus, self._norm(ex))
                    if hits:
                        ctx["emotion_intensities"][emo_key] = {"level": level, "matched_example": ex, "position": hits[0]}
                        matched = True
                        break
                if matched:
                    break

            for sub_name, sub_data in (emo_data.get("sub_emotions", {}) or {}).items():
                spi = ((sub_data.get("emotion_profile", {}) or {}).get("intensity_levels", {}) or {})
                exs_map = (spi.get("intensity_examples", {}) or {})
                for level, exs in (exs_map.items() if isinstance(exs_map, dict) else []):
                    matched = False
                    for ex in exs or []:
                        hits = self._find_positions(nt_focus, self._norm(ex))
                        if hits:
                            key = f"{emo_key}-{sub_name}"
                            ctx["emotion_intensities"][key] = {"level": level, "matched_example": ex, "position": hits[0]}
                            matched = True
                            break
                    if matched:
                        break

        for mod in self.global_intensity_mods.get("amplifiers", set()):
            mod_norm = self._norm(mod)
            for pos in self._find_positions(nt_focus, mod_norm):
                span = nt_focus[max(0, pos - 15):pos + len(mod_norm) + 15]
                intensity_modifiers.append({"type": "amplifier", "modifier": mod, "position": pos, "surrounding_context": span})
                intensity_patterns.append({"modifier": mod, "type": "amplifier", "position": pos, "context": span})
        for mod in self.global_intensity_mods.get("diminishers", set()):
            mod_norm = self._norm(mod)
            for pos in self._find_positions(nt_focus, mod_norm):
                span = nt_focus[max(0, pos - 15):pos + len(mod_norm) + 15]
                intensity_modifiers.append({"type": "diminisher", "modifier": mod, "position": pos, "surrounding_context": span})
                intensity_patterns.append({"modifier": mod, "type": "diminisher", "position": pos, "context": span})

        for k in ("temporal_markers", "transition_indicators", "intensity_modifiers", "intensity_patterns"):
            ctx[k].sort(key=lambda x: x["position"])
        ctx["temporal_markers"]      = ctx["temporal_markers"][: self.cfg["TOPK_TEMPORAL"]]
        ctx["transition_indicators"] = ctx["transition_indicators"][: self.cfg["TOPK_TRANSITION"]]
        ctx["intensity_modifiers"]   = ctx["intensity_modifiers"][: self.cfg["TOPK_INTENSITY"]]
        ctx["intensity_patterns"]    = ctx["intensity_patterns"][: self.cfg["TOPK_INTENSITY"]]
        return ctx

    # --------------------- flow ---------------------
    def _log_summary(self, matched_kw: Set[str], matched_ctx: Set[str], matched_time: Set[str], *, total_kw: int, total_ctx: int, total_time: int) -> None:
        try:
            mk, mc, mt = len(matched_kw), len(matched_ctx), len(matched_time)
            ck = f"{mk}/{total_kw} ({(mk / total_kw * 100.0):.1f}%)" if total_kw else "0/0 (0%)"
            cc = f"{mc}/{total_ctx} ({(mc / total_ctx * 100.0):.1f}%)" if total_ctx else "0/0 (0%)"
            ct = f"{mt}/{total_time} ({(mt / total_time * 100.0):.1f}%)" if total_time else "0/0 (0%)"
            logger.info("[요약] 키워드 매칭: %s | 상황: %s | 시간: %s", ck, cc, ct)
        except Exception as e:
            logger.warning("[요약] 통계 출력 중 오류: %s", e, exc_info=False)

    def analyze_emotion_flow(self, text: str) -> Dict[str, Any]:
        sents = self._split_sentences_ko(text)
        flow = {
            "sequence": [], "transitions": [], "temporal_clusters": [],
            "intensity_changes": [], "emotion_progression": [],
            "transition_patterns": [], "emotional_state_changes": [],
        }
        current_cluster = {"start_idx": 0, "temporal_type": None, "markers": [], "emotions": set(), "transitions": []}
        prev_dom = None; prev_int = {}
        matched_keywords_set: Set[str] = set()
        matched_context_set: Set[str] = set()
        matched_time_set: Set[str] = set()
        
        # 전체 텍스트에서 키워드 매칭 수행
        full_text = " ".join(sents)
        nt_full = self._norm(full_text)
        
        # EMOTIONS.json에서 키워드 추출하여 매칭
        if hasattr(self, 'emotions_data') and isinstance(self.emotions_data, dict):
            for emotion_key, emotion_data in self.emotions_data.items():
                # 코어 키워드 매칭
                core_keywords = emotion_data.get("emotion_profile", {}).get("core_keywords", [])
                for kw in core_keywords:
                    if isinstance(kw, str) and kw.strip():
                        norm_kw = self._norm(kw)
                        if norm_kw in nt_full:
                            matched_keywords_set.add(norm_kw)
                
                # 동의어 매칭
                synonyms = emotion_data.get("emotion_profile", {}).get("synonyms", [])
                for syn in synonyms:
                    if isinstance(syn, str) and syn.strip():
                        norm_syn = self._norm(syn)
                        if norm_syn in nt_full:
                            matched_keywords_set.add(norm_syn)
                
                # 서브 감정 키워드 매칭
                sub_emotions = emotion_data.get("sub_emotions", {})
                if isinstance(sub_emotions, dict):
                    for sub_key, sub_data in sub_emotions.items():
                        sub_keywords = sub_data.get("emotion_profile", {}).get("core_keywords", [])
                        for kw in sub_keywords:
                            if isinstance(kw, str) and kw.strip():
                                norm_kw = self._norm(kw)
                                if norm_kw in nt_full:
                                    matched_keywords_set.add(norm_kw)
        
        # 시간 표현 매칭 개선
        time_patterns = [
            "오늘", "어제", "내일", "지금", "곧", "방금", "오랜만에", "최근에", "요즘",
            "아침", "점심", "저녁", "밤", "새벽", "낮", "밤늦게", "이른", "늦은",
            "월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일",
            "월", "화", "수", "목", "금", "토", "일",
            "1월", "2월", "3월", "4월", "5월", "6월", "7월", "8월", "9월", "10월", "11월", "12월",
            "봄", "여름", "가을", "겨울", "계절", "연말", "연초", "중순", "말", "초"
        ]
        
        for time_pattern in time_patterns:
            norm_time = self._norm(time_pattern)
            if norm_time in nt_full:
                matched_time_set.add(norm_time)
        
        # 총 키워드 수 계산
        total_keywords = 0
        total_time_markers = len(time_patterns)
        total_contexts = 0
        
        if hasattr(self, 'emotions_data') and isinstance(self.emotions_data, dict):
            for emotion_key, emotion_data in self.emotions_data.items():
                # 코어 키워드 수
                core_keywords = emotion_data.get("emotion_profile", {}).get("core_keywords", [])
                total_keywords += len([kw for kw in core_keywords if isinstance(kw, str) and kw.strip()])
                
                # 동의어 수
                synonyms = emotion_data.get("emotion_profile", {}).get("synonyms", [])
                total_keywords += len([syn for syn in synonyms if isinstance(syn, str) and syn.strip()])
                
                # 서브 감정 키워드 수
                sub_emotions = emotion_data.get("sub_emotions", {})
                if isinstance(sub_emotions, dict):
                    for sub_key, sub_data in sub_emotions.items():
                        sub_keywords = sub_data.get("emotion_profile", {}).get("core_keywords", [])
                        total_keywords += len([kw for kw in sub_keywords if isinstance(kw, str) and kw.strip()])
        
        # 속성 설정
        self._total_keyword_count = total_keywords
        self._total_time_marker_count = total_time_markers
        self._total_context_count = total_contexts


        # 문장별 토큰 캐싱
        tokmap = {}
        def _get_tokens(s: str):
            if s not in tokmap:
                tokmap[s] = self._tokenize(s)
            return tokmap[s]

        limit_trans = self.cfg.get("MAX_TRANS_PER_TEXT", 6)
        for idx, sent in enumerate(sents):
            if len(flow["transitions"]) >= limit_trans:
                break
            ctx = self.analyze_context_window(sents, idx, window_size=2)
            tokens = _get_tokens(sent)
            detected = self._identify_emotions_in_sentence(tokens)
            dom = detected[0]["emotion_id"] if detected else None
            for marker in ctx.get("temporal_markers", []):
                marker_text = None
                if isinstance(marker, Mapping):
                    marker_text = marker.get("marker") or marker.get("pattern")
                elif isinstance(marker, str):
                    marker_text = marker
                if isinstance(marker_text, str):
                    norm_marker = self._norm(marker_text)
                    if norm_marker and not self._is_end_frag(norm_marker):
                        matched_time_set.add(norm_marker)

            for indicator in ctx.get("transition_indicators", []):
                pattern_text = None
                if isinstance(indicator, Mapping):
                    pattern_text = indicator.get("pattern") or indicator.get("marker")
                elif isinstance(indicator, str):
                    pattern_text = indicator
                if isinstance(pattern_text, str):
                    norm_pat = self._norm(pattern_text)
                    if norm_pat and not self._is_end_frag(norm_pat):
                        matched_context_set.add(norm_pat)

            for em in detected or []:
                for ev in em.get("evidence", []) or []:
                    value = None
                    if isinstance(ev, Mapping):
                        value = ev.get("value")
                    if isinstance(value, str):
                        norm_val = self._norm(value)
                        if norm_val and not self._is_end_frag(norm_val):
                            matched_keywords_set.add(norm_val)


            curr_int = {}
            for em in detected:
                lvl = em.get("analysis_details", {}).get("dominant_level")
                if not lvl:
                    avg = em.get("analysis_details", {}).get("average_evidence_score", 0)
                    lvl = "high" if avg > 1.5 else ("medium" if avg > 0.8 else "low")
                curr_int[em["emotion_id"]] = {"level": lvl}

            if prev_dom != dom:
                flow["emotional_state_changes"].append({
                    "position": idx, "text": sent,
                    "added_emotions": [dom] if dom else [],
                    "removed_emotions": [prev_dom] if prev_dom else [],
                    "transition_triggers": ctx["transition_indicators"],
                    "temporal_context": ctx["temporal_markers"],
                })

            if (prev_dom or dom) and (prev_dom != dom or not self.cfg["ID_CHANGE_ONLY"]):
                prev_val = self._intensity_value(prev_int.get(prev_dom, {}).get("level")) if prev_dom else 0
                curr_val = self._intensity_value(curr_int.get(dom, {}).get("level")) if dom else 0
                delta = abs(curr_val - prev_val)
                if (prev_dom != dom and delta >= self.cfg["MIN_DELTA"]) or (not self.cfg["ID_CHANGE_ONLY"] and delta >= self.cfg["MIN_DELTA"]):
                    flow["transitions"].append({
                        "from": prev_dom, "to": dom, "delta": delta, "position": idx,
                        "markers": ctx["transition_indicators"], "temporal": ctx["temporal_markers"], "text": sent,
                    })
                    if len(flow["transitions"]) >= limit_trans:
                        break

            flow["sequence"].append({
                "position": idx, "text": sent,
                "temporal_markers": ctx["temporal_markers"],
                "transition_indicators": ctx["transition_indicators"],
                "intensity_modifiers": ctx["intensity_modifiers"],
                "detected_emotions": detected,
            })

            if ctx["temporal_markers"]:
                tm = ctx["temporal_markers"][0]["category"]
                if current_cluster["temporal_type"] != tm and current_cluster["markers"]:
                    current_cluster["transitions"].extend([t for t in flow["transitions"] if current_cluster["start_idx"] <= t["position"] < idx])
                    flow["temporal_clusters"].append(current_cluster)
                    current_cluster = {"start_idx": idx, "temporal_type": tm, "markers": [], "emotions": set(), "transitions": []}
                current_cluster["markers"].extend(ctx["temporal_markers"])
                current_cluster["emotions"].update([em["emotion_id"] for em in detected])

            if detected:
                flow["emotion_progression"].append({
                    "position": idx, "emotions": detected, "text": sent, "temporal_context": ctx["temporal_markers"]
                })

            prev_dom = dom; prev_int = curr_int

        if current_cluster["markers"]:
            current_cluster["transitions"].extend([t for t in flow["transitions"] if t["position"] >= current_cluster["start_idx"]])
            flow["temporal_clusters"].append(current_cluster)

        flow["transition_patterns"] = self.analyze_emotion_transition(text).get("transitions", [])
        flow["transitions"] = flow["transitions"][: self.cfg["MAX_TRANS_PER_TEXT"]]

        self._log_summary(
            matched_kw=matched_keywords_set,
            matched_ctx=matched_context_set,
            matched_time=matched_time_set,
            total_kw=getattr(self, "_total_keyword_count", 0),
            total_ctx=getattr(self, "_total_context_count", 0),
            total_time=getattr(self, "_total_time_marker_count", 0),
        )

        return flow

    def _intensity_value(self, level: Optional[str]) -> int:
        return {"low": 1, "medium": 2, "high": 3}.get((level or "").lower(), 0)

    # --------------------- transitions (rules) ---------------------
    def analyze_emotion_transition(self, text: str) -> Dict[str, Any]:
        sents = self._split_sentences_ko(text)
        transitions = []
        for idx, sent in enumerate(sents):
            nt = self._norm(sent)
            t_info = {"position": idx, "text": sent, "transition_type": None, "markers": [], "context": {}, "matched_transition": []}

            for t_type, pats in self.emotion_transitions.items():
                hits = []
                for p in pats:
                    if self._find_positions(nt, p):
                        hits.append(p)
                if hits and t_info["transition_type"] is None:
                    t_info["transition_type"] = t_type
                t_info["markers"].extend(hits)

            ctx = self.analyze_context_window(sents, idx)
            t_info["context"] = {"before": ctx["before"], "after": ctx["after"], "temporal_markers": ctx["temporal_markers"]}

            for rule in self.labeled_transition_rules:
                trig = rule.get("triggers", []) or []
                trig_hits = []
                for t in trig:
                    if self._find_positions(nt, t):
                        trig_hits.append(t)
                if trig_hits:
                    t_info["matched_transition"].append({
                        "from_emotion": rule.get("from_emotion"),
                        "to_emotion": rule.get("to_emotion"),
                        "trigger_match": trig_hits,
                    })

            if t_info["markers"] or t_info["context"]["temporal_markers"] or t_info["matched_transition"]:
                transitions.append(t_info)

        return {
            "transitions": transitions[: self.cfg["TOPK_TRANSITION"]],
            "total_transitions": len(transitions),
            "transition_types": {t: sum(1 for x in transitions if x.get("transition_type") == t) for t in self.emotion_transitions.keys()},
        }

    # --------------------- temporal ---------------------
    def analyze_temporal_patterns(self, text: str) -> Dict[str, Any]:
        sents = self._split_sentences_ko(text)
        temporal_analysis = {
            "temporal_sequence": [], "emotion_progression": [], "key_temporal_points": [],
            "temporal_clusters": [], "context_patterns": [], "emotion_contexts": {},
            "situation_temporal_mapping": {}, "summary": {},
        }

        TOPK_PER_SENT = self.cfg["TOPK_CONTEXT_PER_SENT"]
        GENERIC = {"자연", "순간", "시간", "때", "무렵", "경치", "풍경"}

        emotion_contexts = {}
        for top, tdata in self.emotions_data.items():
            subs = (tdata.get("sub_emotions", {}) or {})
            for sub_name, sdata in subs.items():
                sits = ((sdata.get("context_patterns", {}) or {}).get("situations", {}) or {})
                for sit_name, sit in sits.items():
                    ek = f"{top}-{sub_name}"
                    emotion_contexts.setdefault(ek, []).append({
                        "name": sit_name, "description": sit.get("description", ""),
                        "core_concept": sit.get("core_concept", ""), "intensity": sit.get("intensity", "medium"),
                        "keywords": sit.get("keywords", []), "variations": sit.get("variations", []),
                        "examples": sit.get("examples", []), "emotion_progression": sit.get("emotion_progression", {}),
                    })

        current_cluster = {"start_idx": 0, "expressions": [], "emotions": set(), "contexts": [], "situations": []}

        # 문장별 토큰 캐싱
        tokmap = {}
        def _get_tokens(s: str):
            if s not in tokmap:
                tokmap[s] = self._tokenize(s)
            return tokmap[s]

        for idx, sent in enumerate(sents):
            tokens = _get_tokens(sent)
            tset = self._expand_token_forms(tokens) | set(tokens)
            norm_tset = {self._norm(tok) for tok in tset}
            temporal_info = {"position": idx, "text": sent, "temporal_markers": [], "emotion_transitions": [],
                             "temporal_type": None, "context_matches": [], "situation_patterns": []}

            for tp, pats in self.extended_temporal_patterns.items():
                for p in pats:
                    if p in sent:
                        temporal_info["temporal_markers"].append(p)
                        temporal_info["temporal_type"] = tp
                        current_cluster["expressions"].append({"type": tp, "pattern": p, "position": idx})

            for w in self.time_keywords:
                hits = self._find_positions(sent, w)
                if hits and w not in temporal_info["temporal_markers"]:
                    temporal_info["temporal_markers"].append(w)
                    if not temporal_info.get("temporal_type"):
                        temporal_info["temporal_type"] = "label_time"
            for w in self.situation_keywords:
                hits = self._find_positions(sent, w)
                if hits and not any(sp.get("keyword") == w for sp in temporal_info["situation_patterns"]):
                    bonus = self.weights.get("situation_ctx_bonus", 0.03)
                    temporal_info["situation_patterns"].append({"keyword": w, "position": hits[0], "bonus": bonus})

            scored = []
            for ek, ctxs in emotion_contexts.items():
                for c in ctxs:
                    kws = set(c["keywords"] or []) | set(c["variations"] or [])
                    kw_hits = [k for k in kws if k in tset]
                    ex_hits = [ex for ex in (c["examples"] or []) if ex and ex in sent]
                    if ex_hits or (len(kw_hits) >= 2) or (len(kw_hits) == 1 and kw_hits[0] not in GENERIC):
                        score = 0.0
                        if ex_hits: score += 2.0
                        score += min(3.0, len(kw_hits)) * 1.0
                        if kw_hits and all(k in GENERIC for k in kw_hits): score *= 0.6
                        if self.time_keyword_norms & norm_tset:
                            score += self.weights.get("time_ctx_bonus", 0.03)
                        if self.situation_keyword_norms & norm_tset:
                            score += self.weights.get("situation_ctx_bonus", 0.03)
                        scored.append({
                            "emotion_key": ek, "situation_name": c["name"],
                            "matched_keywords": kw_hits, "matched_examples": ex_hits,
                            "intensity": c["intensity"], "progression": c["emotion_progression"], "_score": score,
                        })

            scored.sort(key=lambda m: (m["_score"], len(m["matched_keywords"]), bool(m["matched_examples"])), reverse=True)
            pruned, seen = [], set()
            for m in scored:
                key = (m["emotion_key"], m["situation_name"])
                if key in seen:
                    continue
                seen.add(key)
                pruned.append({k: v for k, v in m.items() if not k.startswith("_")})
                if len(pruned) >= TOPK_PER_SENT:
                    break
            temporal_info["context_matches"] = pruned

            identified = self._identify_emotions_in_sentence(_get_tokens(sent))
            if identified:
                current_cluster["emotions"].update([em["emotion_id"] for em in identified])
                temporal_analysis["emotion_progression"].append({
                    "position": idx, "emotions": identified, "text": sent,
                    "temporal_context": temporal_info["temporal_markers"], "context_patterns": temporal_info["context_matches"],
                })

            if temporal_info["temporal_markers"]:
                temporal_analysis["temporal_sequence"].append(temporal_info)
                # 클러스터 경계
                if (idx - current_cluster["start_idx"] > 3) or (current_cluster["expressions"] and temporal_info["temporal_type"] != current_cluster["expressions"][-1]["type"]):
                    if current_cluster["expressions"]:
                        cluster_summary = {
                            "start": current_cluster["start_idx"], "end": idx - 1,
                            "expressions": current_cluster["expressions"],
                            "emotions": list(current_cluster["emotions"]),
                            "contexts": current_cluster["contexts"], "situations": current_cluster["situations"],
                        }
                        prog_sum = {}
                        for ctx in current_cluster["contexts"]:
                            prog = ctx.get("progression", {}) or {}
                            if prog:
                                prog_sum[ctx["emotion_key"]] = {
                                    "trigger": prog.get("trigger"), "development": prog.get("development"),
                                    "peak": prog.get("peak"), "aftermath": prog.get("aftermath"),
                                }
                        if prog_sum:
                            cluster_summary["emotion_progression"] = prog_sum
                        temporal_analysis["temporal_clusters"].append(cluster_summary)
                    current_cluster = {"start_idx": idx, "expressions": [], "emotions": set(), "contexts": [], "situations": []}

        if current_cluster["expressions"]:
            temporal_analysis["temporal_clusters"].append({
                "start": current_cluster["start_idx"], "end": len(sents) - 1,
                "expressions": current_cluster["expressions"], "emotions": list(current_cluster["emotions"]),
                "contexts": current_cluster["contexts"], "situations": current_cluster["situations"],
            })

        temporal_analysis["temporal_sequence"] = temporal_analysis["temporal_sequence"][: self.cfg["TOPK_TEMPORAL"]]
        temporal_analysis["summary"] = {
            "total_sequences": len(temporal_analysis["temporal_sequence"]),
            "total_clusters": len(temporal_analysis["temporal_clusters"]),
            "emotion_context_matches": sum(len(t.get("context_matches", [])) for t in temporal_analysis["temporal_sequence"]),
            "temporal_distribution": {}, "situation_coverage": {},
        }
        for cluster in temporal_analysis["temporal_clusters"]:
            ttype = next((e["type"] for e in cluster.get("expressions", []) if "type" in e), None)
            if ttype:
                temporal_analysis["summary"]["temporal_distribution"][ttype] = temporal_analysis["summary"]["temporal_distribution"].get(ttype, 0) + 1

        for ek, ctxs in emotion_contexts.items():
            matched = 0
            for c in ctxs:
                for seq in temporal_analysis["temporal_sequence"]:
                    if any(m["emotion_key"] == ek and m["situation_name"] == c["name"] for m in seq.get("context_matches", [])):
                        matched += 1; break
            temporal_analysis["summary"]["situation_coverage"][ek] = {"total_situations": len(ctxs), "matched_situations": matched}
        return temporal_analysis

    # --------------------- evidence pruning ---------------------
    def _prune_evidence(self, evidence: List[Dict[str, Any]], k: int = 4) -> List[Dict[str, Any]]:
        """
        - 동일 의미변형(풍요/풍요로움/풍요스러움 등) 루트 기준 중복 제거
        - 일반동사(느끼다/하다/되다 등)는 별도 타입으로 낮은 우선순위 처리
        - 정렬: 점수↓, 타입 우선순위↓, 문자열 길이↓
        """
        import re

        GENERIC_VERBS = {"느끼", "느꼈", "느낀", "느낄", "하다", "했다", "합니다", "되다", "된다", "있다", "없다"}

        def normv(e):
            v = e.get("value")
            return self._normalize_keyword(v) if isinstance(v, str) else v

        def lemma_root(s: str) -> str:
            """한국어 감정/상태명사의 흔한 파생 접미 제거."""
            if not s: return s
            r = self._normalize_keyword(s)
            # 풍요(로움), 감사(함), 자부(심) 등 느슨한 루트화
            r = re.sub(r"(로움|스러움|스러|스럽|스러웠|스러워|함|감|적)$", "", r)
            # 조사/어미 잔여 제거
            r = re.sub(r"(에서|으로|로|에게|보다|까지|부터|만|도|의|와|과|은|는|이|가|을|를|다|니다|습니다)$", "", r)
            return r or s

        # 일반 동사를 evidence에 남기되 우선순위 낮춘 별도 타입으로 변환
        cleaned: List[Dict[str, Any]] = []
        for e in evidence:
            v = normv(e)
            if isinstance(v, str) and v in GENERIC_VERBS and e.get("type") == "keyword":
                e = dict(e)
                e["type"] = "weak_verb"  # 낮은 우선순위
                e["score"] = min(0.6, float(e.get("score", 0.4)))
            cleaned.append(e)

        pri = {
            "keyword": 3,
            "positive_indicator": 2,
            "negative_indicator": 2,
            "amplifier": 1,
            "diminisher": 1,
            "scene_tone": 1,
            "weak_verb": 0,  # 가장 낮게
            "positive_indicator_fallback": 2,
        }

        # 루트 단위 중복 제거
        uniq, seen = [], set()
        sorted_ev = sorted(
            cleaned,
            key=lambda e: (
                abs(float(e.get("score", 0.0))),
                pri.get(e.get("type"), 0),
                len(str(e.get("value", ""))),
            ),
            reverse=True,
        )
        for e in sorted_ev:
            t = e.get("type")
            v = normv(e)
            key = (t, lemma_root(v) if isinstance(v, str) else v)
            if key in seen:
                continue
            seen.add(key)
            uniq.append(e)
            if len(uniq) >= k:
                break
        return uniq

    def _filter_evidence(self, evid: List[Any]) -> List[Any]:
        out: List[Any] = []
        seen: Set[str] = set()
        for e in evid or []:
            value = None
            if isinstance(e, Mapping):
                value = e.get("value") or e.get("marker")
            elif isinstance(e, str):
                value = e
            if isinstance(value, str):
                normed = self._norm(value)
                if self._is_end_frag(normed):
                    continue
                if normed in seen:
                    continue
                seen.add(normed)
            out.append(e)
        return out

    # --------------------- identify emotions (core) ---------------------
    def _apply_evidence_overlap_penalty(self, scores_by_eid: Dict[str, float], evidence_by_eid: Dict[str, List[Any]]) -> Dict[str, float]:
        winner = max(scores_by_eid.items(), key=lambda kv: kv[1])[0] if scores_by_eid else None
        if not winner:
            return scores_by_eid

        def _evset(items: List[Any]) -> Set[str]:
            bag: Set[str] = set()
            for item in items or []:
                raw = None
                if isinstance(item, Mapping):
                    raw = item.get("value") or item.get("marker")
                elif isinstance(item, str):
                    raw = item
                if not isinstance(raw, str):
                    continue
                normed = self._norm(raw)
                if self._is_end_frag(normed):
                    continue
                bag.add(normed)
            return bag

        win_set = _evset(evidence_by_eid.get(winner, []))
        if not win_set:
            return scores_by_eid

        alpha = float(self.weights.get("overlap_penalty_alpha", 0.35))
        updated = dict(scores_by_eid)
        for eid, sc in scores_by_eid.items():
            if eid == winner:
                continue
            overlap = len(win_set & _evset(evidence_by_eid.get(eid, [])))
            if overlap > 0:
                updated[eid] = sc / (1.0 + alpha * overlap)
        return updated

    def _identify_emotions_in_sentence(self, tokens: List[str]) -> List[Dict[str, Any]]:
        """
        문장 내 감정 후보를 보수적으로 식별.
        - 라벨 어휘/톤 렉시콘 기반 필터링
        - 코어/동의어/상황/언어키프레이즈/강도예시 통합
        - 씬톤(scene tone) 보조 가중 (환경변수 PE_SCENETONE_W)
        - 폴라리티 가드 + 락-중립 가드(환경변수 PE_LAK_NEUTRAL_GUARD)
        - '다행/안심' 계열 보정: 희-감사함 +1.2, 락-농담 +0.6
        - 증거 과적재 방지/컨피던스 계산
        - 주감정(primary)별 결과 상한(환경변수 PE_CAP_PER_PRIMARY)
        """
        import os, math
        from collections import Counter

        # ---------- 런타임 다이얼(환경변수) ----------
        scene_w = float(os.environ.get("PE_SCENETONE_W", "0.6"))  # 0.5~0.7 권장
        lak_ng = float(os.environ.get("PE_LAK_NEUTRAL_GUARD", "0.85"))  # 락-중립 가드 강도
        min_conf = float(os.environ.get("PE_MIN_CONF", "0.50"))  # 최소 컨피던스 컷
        cap_per_primary = int(os.environ.get("PE_CAP_PER_PRIMARY", "2"))  # 주감정별 상한

        # ---------- 토큰/렉시콘 준비 ----------
        raw_tokens = (self._expand_token_forms(tokens) or set()) | set(tokens)
        label_vocab = getattr(self, "label_vocab", set())
        pos_lex = self.scene_tone_lex.get("positive", set())
        neg_lex = self.scene_tone_lex.get("negative", set())

        token_set: Set[str] = set()
        for t in raw_tokens:
            nt = self._normalize_keyword(t)
            if not nt or not self._is_content_token(nt):
                continue
            # 라벨 어휘/전역 톤 렉시콘 안에 있을 때만 유지
            if label_vocab and (nt not in label_vocab) and (nt not in pos_lex) and (nt not in neg_lex):
                continue
            token_set.add(nt)

        candidate_tokens = {self._normalize_keyword(t) for t in raw_tokens if t}
        candidate_tokens.update(token_set)
        token_index = getattr(self, '_token2eids', {})
        cand = Counter()
        for norm_token in filter(None, candidate_tokens):
            for eid in token_index.get(norm_token, ()):
                cand[eid] += 1
        try:
            topk = int(os.getenv('PE_CAND_TOPK', '50'))
        except ValueError:
            topk = 50
        topk = max(1, topk)
        candidate_eids = {eid for eid, _ in cand.most_common(topk)} if cand else set(self.emotion_categories.keys())

        # 씬톤 히트(긍/부) 수집(증거는 토큰 단위로 개별 기록)
        scene_pos_hits = [t for t in token_set if t in pos_lex]
        scene_neg_hits = [t for t in token_set if t in neg_lex]

        # 장면 톤을 보조 신호로 base에 가산(너무 과하지 않게 가중)
        # 증거는 문자열 1개씩 기록해 해싱 안전
        base_scene_boost = 0.0
        for st in scene_pos_hits:
            base_scene_boost += scene_w
        for st in scene_neg_hits:
            base_scene_boost -= scene_w * 0.8  # 부정 씬톤은 완만히 감쇠

        # ---------- ML/메타 설정 ----------
        emotion_score_map: Dict[str, Dict[str, Any]] = {}
        negative_prefix_penalty = -1.0
        token_idf: Dict[str, float] = getattr(self, "token_idf", {}) or {}
        weights = getattr(self, "weights", {
            "keyword_base": 2.0,
            "keyword_neutral": 1.0,
            "idf_a": 0.85,
            "idf_b": 0.30,
            "pos_hit": 1.5,
            "neg_hit": -1.0,
            "amp_mul": 1.5,
            "dim_mul": 0.7,
            "guard_pos_scene_neg": 0.45,
            "guard_neg_over_pos": 0.70,
            "relief_gratitude": 1.2,
            "relief_light": 0.6,
            "time_ctx_bonus": 0.03,
            "situation_ctx_bonus": 0.03,
        })

        ml_settings: Dict[str, Any] = {}
        for emotion_id, cat_info in self.emotion_categories.items():
            if emotion_id not in candidate_eids:
                continue
            metadata = cat_info.get("metadata", {})
            ml_metadata = metadata.get("ml_training_metadata", {})
            if ml_metadata:
                ml_settings[emotion_id] = {
                    "confidence_thresholds": ml_metadata.get(
                        "confidence_thresholds",
                        {"basic": 0.7, "complex": 0.8, "subtle": 0.9},
                    ),
                    "pattern_matching": ml_metadata.get(
                        "pattern_matching",
                        {"basic": 0.65, "complex": 0.75, "subtle": 0.85},
                    ),
                    "analysis_modules": ml_metadata.get("analysis_modules", {}),
                }

        # ---------- 감정별 스코어링 ----------
        relief_stems = ("다행", "안도", "안심")
        has_relief = any(any(st in tok for st in relief_stems) for tok in token_set)

        for emotion_id, cat_info in self.emotion_categories.items():
            primary_cat = cat_info.get("primary")
            sub_emo_name = cat_info.get("sub_emotion_name")
            metadata = cat_info.get("metadata", {})
            emo_cplx = metadata.get("emotion_complexity", "basic")

            # 폴라리티 추정
            cat_polarity = metadata.get("sentiment_polarity")
            if not cat_polarity:
                if primary_cat in ("희", "락"):
                    cat_polarity = "pos"
                elif primary_cat in ("노", "애"):
                    cat_polarity = "neg"
                else:
                    cat_polarity = "neutral"

            base = 0.0 + base_scene_boost  # 씬톤은 공통 바닥효과로 반영
            evidence: List[Dict[str, Any]] = []
            node = self._get_emotion_node_by_id(emotion_id) or {}

            # 씬톤 증거(개별 토큰)
            for st in scene_pos_hits:
                evidence.append({"type": "scene_tone", "value": st, "score": scene_w})
            for st in scene_neg_hits:
                evidence.append({"type": "scene_tone", "value": st, "score": -scene_w * 0.8})

            # --- 라벨 병합 키워드 준비(코어/동의어/상황/언어/강도예시) ---
            meta_core = set(metadata.get("core_keywords", []) or [])
            meta_syn = set(metadata.get("synonyms", []) or [])
            prof = node.get("emotion_profile", {}) or {}
            node_core = set(prof.get("core_keywords", []) or [])
            node_syn = set(prof.get("synonyms", []) or [])

            ctx = node.get("context_patterns", {}) or {}
            situ_kws = set()
            for s in (ctx.get("situations", {}) or {}).values():
                situ_kws.update(s.get("keywords", []) or [])
                situ_kws.update(s.get("variations", []) or [])
                core_concept = s.get("core_concept")
                if core_concept:
                    situ_kws.add(core_concept)
                for ex in (s.get("examples", []) or []):
                    for tok in self._tokenize(ex):
                        if len(tok) >= 2:
                            situ_kws.add(tok)

            ling = node.get("linguistic_patterns", {}) or {}
            ling_kws = set()
            for kp in ling.get("key_phrases", []) or []:
                pat = (kp or {}).get("pattern", "")
                if pat:
                    cleaned = pat.replace("[XXX]", "").replace("[XX]", "").strip()
                    if cleaned:
                        ling_kws.add(cleaned)

            int_lv = prof.get("intensity_levels", {}) or {}
            int_ex = set()
            for exs in (int_lv.get("intensity_examples", {}) or {}).values():
                if isinstance(exs, list):
                    int_ex.update(exs or [])

            raw_all = (meta_core | meta_syn | node_core | node_syn | situ_kws | ling_kws | int_ex)

            cleaned_keywords = set()
            for kw in raw_all:
                nk = self._normalize_keyword(kw)
                if not nk or not self._is_content_token(nk):
                    continue
                if label_vocab and (nk not in label_vocab) and (nk not in pos_lex) and (nk not in neg_lex):
                    continue
                cleaned_keywords.add(nk)

            # 코어 키워드 히트(중립명사 감쇠 + IDF 보정)
            core_kw_hits = 0
            for kw in cleaned_keywords:
                if kw in token_set:
                    idf = token_idf.get(kw, 1.0)
                    w_keyword = ((weights.get("keyword_base", 2.0) if kw not in self.neutral_nouns else weights.get("keyword_neutral", 1.0)) * (weights.get("idf_a", 0.85) + weights.get("idf_b", 0.30) * min(1.0, idf)))
                    base += w_keyword
                    core_kw_hits += 1
                    evidence.append({"type": "keyword", "value": kw, "score": w_keyword, "source": "core"})

            # 관련 감정(메타)
            emo_profile_meta = metadata.get("emotion_profile", {}) or {}
            related = emo_profile_meta.get("related_emotions", {}) or {}
            for rel_type, rel_list in related.items():
                w = 1.0 if rel_type == "positive" else (-0.5 if rel_type == "negative" else 0.3)
                for rel in rel_list:
                    nr = self._normalize_keyword(rel)
                    if nr and nr in token_set:
                        base += w
                        evidence.append({"type": "related_emotion", "value": nr, "relationship": rel_type, "score": w})

            # 감성 지표/증폭/감쇠
            def _merge_senti(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
                out: Dict[str, Any] = {}
                out["positive_indicators"] = list(
                    set((a.get("positive_indicators", []) or []) + (b.get("positive_indicators", []) or [])))
                out["negative_indicators"] = list(
                    set((a.get("negative_indicators", []) or []) + (b.get("negative_indicators", []) or [])))
                out["intensity_modifiers"] = {
                    "amplifiers": list(set(((a.get("intensity_modifiers", {}) or {}).get("amplifiers", []) or []) + (
                    ((b.get("intensity_modifiers", {}) or {}).get("amplifiers", []) or [])))),
                    "diminishers": list(set(((a.get("intensity_modifiers", {}) or {}).get("diminishers", []) or []) + (
                    ((b.get("intensity_modifiers", {}) or {}).get("diminishers", []) or [])))),
                }
                return out

            meta_senti = metadata.get("sentiment_analysis", {}) or {}
            node_senti = node.get("sentiment_analysis", {}) or {}
            senti = _merge_senti(meta_senti, node_senti)

            pos_inds = senti.get("positive_indicators", []) or []
            neg_inds = senti.get("negative_indicators", []) or []
            pos_hits = 0
            neg_hits = 0

            for p in pos_inds:
                npk = self._normalize_keyword(p)
                if npk and npk in token_set:
                    delta = weights.get("pos_hit", 1.5)
                    base += delta
                    pos_hits += 1
                    evidence.append({"type": "positive_indicator", "value": npk, "score": delta})
            for n in neg_inds:
                nnk = self._normalize_keyword(n)
                if nnk and nnk in token_set:
                    delta = weights.get("neg_hit", -1.0)
                    base += delta
                    neg_hits += 1
                    evidence.append({"type": "negative_indicator", "value": nnk, "score": delta})

            mods = senti.get("intensity_modifiers", {}) or {}
            amps = set(mods.get("amplifiers", []) or []) | self.global_intensity_mods.get("amplifiers", set())
            dims = set(mods.get("diminishers", []) or []) | self.global_intensity_mods.get("diminishers", set())
            for a in amps:
                na = self._normalize_keyword(a)
                if na and na in token_set:
                    mul = weights.get("amp_mul", 1.5)
                    base *= mul
                    evidence.append({"type": "amplifier", "value": na, "multiplier": mul})
            for d in dims:
                nd = self._normalize_keyword(d)
                if nd and nd in token_set:
                    mul = weights.get("dim_mul", 0.7)
                    base *= mul
                    evidence.append({"type": "diminisher", "value": nd, "multiplier": mul})

            # ML 모듈(옵션)
            ms = ml_settings.get(emotion_id)
            if ms:
                analysis_modules = ms.get("analysis_modules", {}) or {}
                if analysis_modules.get("pattern_extractor", {}).get("enabled"):
                    base += self._apply_pattern_extraction(list(token_set), analysis_modules["pattern_extractor"])
                if analysis_modules.get("intensity_analyzer", {}).get("enabled"):
                    base *= self._apply_intensity_analysis(list(token_set), analysis_modules["intensity_analyzer"],
                                                           evidence)
                if analysis_modules.get("context_extractor", {}).get("enabled"):
                    base += self._apply_context_extraction(list(token_set), analysis_modules["context_extractor"])

            # 부정접두 복합어
            for t in tokens:
                nt = self._normalize_keyword(t)
                if nt and nt in self.neg_prefix_terms:
                    base += negative_prefix_penalty
                    evidence.append({"type": "negative_prefix", "value": nt, "score": negative_prefix_penalty})

            # 폴라리티 가드(긍/부 동작)
            if cat_polarity == "neg":
                if neg_hits == 0 and (pos_hits > 0 or len(scene_pos_hits) > 0):
                    pen = weights.get("guard_pos_scene_neg", 0.45)
                    base *= pen
                    evidence.append({"type": "polarity_guard_v2", "reason": "positive_scene_without_negative_evidence",
                                     "penalty": pen})
            elif cat_polarity == "pos":
                if pos_hits == 0 and neg_hits > 0:
                    pen = weights.get("guard_neg_over_pos", 0.70)
                    base *= pen
                    evidence.append(
                        {"type": "polarity_guard_v2", "reason": "negative_signals_without_positive_evidence",
                         "penalty": pen})

            # 락-중립 가드: 코어 키워드 없이 씬톤/약한 지표에만 의존한 락 계열 완만 감쇠
            if primary_cat == "락" and base > 0:
                only_scene_or_weak = (core_kw_hits == 0) and (pos_hits == 0) and (len(scene_pos_hits) > 0)
                if only_scene_or_weak:
                    base *= lak_ng
                    evidence.append({"type": "guard", "value": "lak_neutral_guard", "multiplier": lak_ng})

            # 다행/안심 계열 보정
            if has_relief:
                name_norm = self._norm(sub_emo_name or "")
                gratitude_terms = ("감사", "고마")
                relief_hit = any(any(term in token for term in gratitude_terms) for token in token_set) or any(term in name_norm for term in gratitude_terms)
                if relief_hit:
                    bonus = weights.get("relief_gratitude", 1.2)
                    base += bonus
                    evidence.append({"type": "relief_bonus", "value": "gratitude", "score": bonus})
                else:
                    bonus = weights.get("relief_light", 0.6)
                    base += bonus
                    evidence.append({"type": "relief_bonus", "value": "relief", "score": bonus})

            # 컨피던스 계산(증거 정리 후)
            if getattr(self, 'calibrator', None) and cleaned_keywords:
                hits = [kw for kw in cleaned_keywords if kw in token_set]
                if hits:
                    prior_fn = getattr(self.calibrator, 'get_prior_adj', None)
                    weight_fn = getattr(self.calibrator, 'get_pattern_weight', None)
                    log_adj = prior_fn(emotion_id) if callable(prior_fn) else 0.0
                    if callable(weight_fn):
                        for kw in hits:
                            log_adj += weight_fn(emotion_id, kw)
                    base *= math.exp(max(-1.5, min(1.5, log_adj)))

            if base > 0 and evidence:
                # 과적재 방지: 서로 다른 타입/가치가 최대치 골고루 남도록 내부 함수가 정리
                evidence = self._prune_evidence(evidence, k=4)
                evidence = self._filter_evidence(evidence)

                # ML 임계값 또는 기본값
                thr = 0.7
                if ms:
                    thr = ms["confidence_thresholds"].get(emo_cplx, 0.7)

                # 로지스틱 스케일 + 분산 보정
                normed = base / (len(evidence) + 1)
                ksig = 5.0
                sig = 1.0 / (1.0 + math.exp(-ksig * (normed - thr)))
                mag = 1.0 - math.exp(-base / 3.5)
                div = len({e["type"] for e in evidence})
                boost = min(1.0, 0.90 + 0.03 * div)
                conf = sig * (0.70 + 0.30 * mag) * boost
                conf = max(0.0, min(conf, 0.97))

                # 스코어 설명 가능성을 위한 breakdown
                score_dbg = {
                    "scene": base_scene_boost,
                    "keywords": core_kw_hits,
                    "pos_hits": pos_hits, 
                    "neg_hits": neg_hits,
                    "mods": {
                        "amp": any(a in token_set for a in amps), 
                        "dim": any(d in token_set for d in dims)
                    },
                    "guards": {
                        "pos_scene_neg": cat_polarity=="neg" and neg_hits==0 and (pos_hits>0 or len(scene_pos_hits)>0),
                        "neg_over_pos": cat_polarity=="pos" and pos_hits==0 and neg_hits>0
                    },
                    "relief": bool(has_relief),
                }

                emotion_score_map[emotion_id] = {
                    "score": base,
                    "confidence": conf,
                    "primary": primary_cat,
                    "sub_emotion": sub_emo_name,
                    "complexity": emo_cplx,
                    "evidence": evidence,
                    "score_breakdown": score_dbg,  # 추가된 설명 가능성 필드
                }

        # ---------- 정렬/컷/주감정 상한 ----------
        if emotion_score_map:
            scores_by_eid = {eid: info["score"] for eid, info in emotion_score_map.items()}
            evidence_by_eid = {eid: info.get("evidence", []) for eid, info in emotion_score_map.items()}
            penalized = self._apply_evidence_overlap_penalty(scores_by_eid, evidence_by_eid)
            for eid, new_score in penalized.items():
                info = emotion_score_map.get(eid)
                if not info:
                    continue
                old_score = info.get("score", 0.0)
                info["score"] = new_score
                if old_score > 0 and new_score < old_score:
                    ratio = max(0.0, min(1.0, new_score / old_score))
                    info["confidence"] = max(0.0, info.get("confidence", 0.0) * ratio)

        sorted_items = sorted(
            emotion_score_map.items(),
            key=lambda x: (x[1]["confidence"], x[1]["score"]),
            reverse=True,
        )

        out: List[Dict[str, Any]] = []
        per_primary_cnt: Dict[str, int] = {}
        for emo_id, info in sorted_items:
            if info["confidence"] < min_conf:
                continue
            p = info["primary"] or ""
            if cap_per_primary > 0:
                if per_primary_cnt.get(p, 0) >= cap_per_primary:
                    continue
                per_primary_cnt[p] = per_primary_cnt.get(p, 0) + 1

            ev = self._filter_evidence(info.get("evidence", []))
            avg_ev = (sum(e.get("score", 0) for e in ev) / len(ev)) if ev else 0.0
            dom = "high" if info["confidence"] >= 0.85 else ("medium" if info["confidence"] >= 0.55 else "low")
            out.append({
                "emotion_id": emo_id,
                "primary": info["primary"],
                "sub_emotion": info["sub_emotion"],
                "score": info["score"],
                "confidence": info["confidence"],
                "complexity": info["complexity"],
                "evidence": ev,
                "analysis_details": {
                    "evidence_count": len(ev),
                    "evidence_types": Counter(e["type"] for e in ev),
                    "average_evidence_score": avg_ev,
                    "dominant_level": dom,
                },
            })
            if len(out) >= 3:  # 전체 상위 3개
                break

        return out

    # --------------------- ML helpers ---------------------
    def _apply_pattern_extraction(self, tokens: List[str], module_settings: Dict[str, Any]) -> float:
        """간단 룰 가중(옵션)."""
        try:
            threshold = float(module_settings.get("threshold", 0.75))
            rules = module_settings.get("additional_rules", []) or []
            score = 0.0
            tset = set(tokens)
            for rule in rules:
                r = self._norm(rule)
                if r in tset:
                    score += min(len(r) * 0.05, 0.6)
            if 0 < score < threshold:
                score *= 0.5
            return score
        except Exception:
            return 0.0

    def _apply_intensity_analysis(self, tokens: List[str], module_settings: Dict[str, Any], existing_evidence: List[Dict[str, Any]]) -> float:
        """데이터 기반 전역 증폭/감쇠 모음으로 강도 스케일 조정(옵션)."""
        try:
            tset = set(tokens)
            amps = [a for a in self.global_intensity_mods.get("amplifiers", set()) if a in tset]
            dims = [d for d in self.global_intensity_mods.get("diminishers", set()) if d in tset]
            score = 1.0
            if amps:
                mul = min(1.5, 1.2 + 0.05 * len(amps))
                score *= mul
                existing_evidence.append({"type": "intensity_amp_global", "hits": amps, "multiplier": mul})
            if dims:
                mul = max(0.7, 0.95 - 0.05 * len(dims))
                score *= mul
                existing_evidence.append({"type": "intensity_dim_global", "hits": dims, "multiplier": mul})
            return max(0.2, min(3.0, score))
        except Exception:
            return 1.0

    def _apply_context_extraction(self, tokens: List[str], module_settings: Dict[str, Any]) -> float:
        """시간/전이 표지어 존재에 따른 소폭 가산(옵션)."""
        try:
            tset = set(tokens)
            temporal = sum(1 for pats in self.temporal_markers.values() for p in pats if self._norm(p) in tset)
            trans = sum(1 for pats in self.emotion_transitions.values() for p in pats if self._norm(p) in tset)
            score = 0.0
            if temporal: score += min(1.0, 0.15 * temporal)
            if trans:    score += min(1.0, 0.20 * trans)
            return max(0.0, min(2.0, score))
        except Exception:
            return 0.0

# =============================================================================
# EmotionPatternExtractor
# =============================================================================
class EmotionPatternExtractor:
    """감정 패턴을 추출하고 임베딩을 생성하는 클래스 (데이터 기반/내결함성 강화본)"""

    # 모듈 전역 캐시: 모델/토크나이저 재사용
    _MODEL_CACHE = {"name": None, "tokenizer": None, "model": None, "device": None}
    _MODEL_LOCK = threading.RLock()

    def __init__(self, emotions_data_path: str = None):
        # EMOTIONS.json 로드 (하드코딩 대신)
        if emotions_data_path is None:
            self.emotions_data = self._load_emotions_data_default()
        else:
            self.emotions_data = self._load_emotions_data_from_path(emotions_data_path)
            
        self.pattern_cache: Dict[str, Any] = {}
        self.cache_lock = threading.Lock()
        self.MAX_PATTERN_CACHE = int(os.environ.get("PE_MAX_PAT_CACHE", "256"))
        
        # EMOTIONS.json 기반 패턴 캐시 초기화
        self._emotion_patterns_cache = {}
        self._extraction_patterns_cache = {}
        self._load_extraction_patterns_recursive()
        self.model = None
        self.tokenizer = None
        self.processed_emotions = set()
        
        # 설정 초기화
        self.cfg = {
            "TOPK_KEYWORDS": int(os.environ.get("PE_TOPK_KEYWORDS", "30")),
            "TOPK_SITUATIONS": int(os.environ.get("PE_TOPK_SITUATIONS", "20")),
            "TOPK_TRIGGERS": int(os.environ.get("PE_TOPK_TRIGGERS", "15")),
            "TOPK_INTENSITY": int(os.environ.get("PE_TOPK_INTENSITY", "20")),
            "TOPK_PHRASES": int(os.environ.get("PE_TOPK_PHRASES", "25")),
            "TOPK_KEYPHRASES": int(os.environ.get("PE_TOPK_KEYPHRASES", "20")),
            "TOPK_INTENSITY_PER_LEVEL": int(os.environ.get("PE_TOPK_INTENSITY_PER_LEVEL", "15")),
            "TOPK_POS_IND": int(os.environ.get("PE_TOPK_POS_IND", "10")),
            "TOPK_NEG_IND": int(os.environ.get("PE_TOPK_NEG_IND", "10")),
            "TOPK_TRANSITION": int(os.environ.get("PE_TOPK_TRANSITION", "20")),
            "MODEL_NAME": os.environ.get("PE_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"),
            "FP16": os.environ.get("PE_FP16", "0").lower() in ("1", "true", "yes"),
            "BATCH_SIZE": int(os.environ.get("PE_BATCH_SIZE", "32")),
            "MAX_LEN": int(os.environ.get("PE_MAX_LEN", "512")),
        }
        
        # PatternContextAnalyzer 초기화 (캐시된 인스턴스 사용)
        self.context_analyzer = PatternContextAnalyzer.get_cached_instance()
        self._use_embedding_cache = os.environ.get("PE_USE_EMBEDDING_CACHE", "1").strip().lower() in ("1", "true", "yes", "on")
        self._embedding_cache_engine = None
        self._embedding_cache_lock = threading.Lock()
        self._embedding_cache_error = False

    def _load_emotions_data_default(self) -> Dict[str, Any]:
        """EMOTIONS.json 파일 로드"""
        try:
            # 전역 캐시에서 먼저 시도
            try:
                from src.data_utils import get_global_emotions_data
                data = get_global_emotions_data()
                if data:
                    return data
            except Exception:
                pass
            
            # config에서 EMOTIONS 데이터 직접 로드 시도
            try:
                from src.config import _load_emotions_cached
                return _load_emotions_cached()
            except Exception:
                pass
            
            # 기본 감정 데이터 반환
            return {
                "희": {"sub_emotions": {}},
                "노": {"sub_emotions": {}},
                "애": {"sub_emotions": {}},
                "락": {"sub_emotions": {}}
            }
        except Exception as e:
            print(f"EMOTIONS.json 로드 실패: {e}")
            return {}

    def _load_emotions_data_from_path(self, emotions_data_path: str) -> Dict[str, Any]:
        """경로에서 EMOTIONS.json 파일 로드"""
        try:
            # emotions_data_path가 dict인 경우 처리
            if isinstance(emotions_data_path, dict):
                return emotions_data_path
            
            # 문자열 경로인 경우에만 파일 로드 시도
            if isinstance(emotions_data_path, str):
                with open(emotions_data_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                print(f"EMOTIONS.json 로드 실패: 잘못된 경로 타입 - {type(emotions_data_path)}")
                return {}
        except Exception as e:
            print(f"EMOTIONS.json 로드 실패: {e}")
            return {}

    def _load_extraction_patterns_recursive(self, emotion_data: Dict[str, Any] = None, path: str = "") -> None:
        """EMOTIONS.json에서 추출 패턴을 재귀적으로 로드"""
        if emotion_data is None:
            emotion_data = self.emotions_data
            
        for emotion_key, emotion_info in emotion_data.items():
            current_path = f"{path}.{emotion_key}" if path else emotion_key
            
            # 하위 감정 재귀 처리
            if isinstance(emotion_info, dict):
                if "sub_emotions" in emotion_info:
                    self._load_extraction_patterns_recursive(emotion_info["sub_emotions"], current_path)
                
                # 추출 패턴 추출
                extraction_patterns = self._extract_extraction_patterns(emotion_info, current_path)
                if extraction_patterns:
                    self._extraction_patterns_cache[current_path] = extraction_patterns
                    
                # 감정 패턴 추출
                emotion_patterns = self._extract_emotion_patterns(emotion_info, current_path)
                if emotion_patterns:
                    self._emotion_patterns_cache[current_path] = emotion_patterns

    def _extract_extraction_patterns(self, emotion_info: Dict[str, Any], emotion_path: str) -> Dict[str, Any]:
        """감정 정보에서 추출 패턴 추출"""
        patterns = {
            "keywords": [],
            "expressions": [],
            "intensity_markers": [],
            "context_clues": [],
            "temporal_markers": []
        }
        
        # intensity_examples에서 패턴 추출
        if "emotion_profile" in emotion_info:
            profile = emotion_info["emotion_profile"]
            if "intensity_levels" in profile:
                intensity_levels = profile["intensity_levels"]
                if "intensity_examples" in intensity_levels:
                    examples = intensity_levels["intensity_examples"]
                    for level, example_list in examples.items():
                        if isinstance(example_list, list):
                            patterns["expressions"].extend(example_list)
                            # 강도 마커 추출
                            if level == "high":
                                patterns["intensity_markers"].extend(["매우", "정말", "너무", "완전히"])
                            elif level == "medium":
                                patterns["intensity_markers"].extend(["꽤", "상당히", "어느 정도"])
        
        # keywords에서 패턴 추출
        if "keywords" in emotion_info:
            patterns["keywords"].extend(emotion_info["keywords"])
            
        # triggers에서 컨텍스트 단서 추출
        if "triggers" in emotion_info:
            patterns["context_clues"].extend(emotion_info["triggers"])
            
        return patterns

    def _extract_emotion_patterns(self, emotion_info: Dict[str, Any], emotion_path: str) -> List[str]:
        """감정 정보에서 기본 패턴 추출"""
        patterns = []
        
        # intensity_examples에서 패턴 추출
        if "emotion_profile" in emotion_info:
            profile = emotion_info["emotion_profile"]
            if "intensity_levels" in profile:
                intensity_levels = profile["intensity_levels"]
                if "intensity_examples" in intensity_levels:
                    examples = intensity_levels["intensity_examples"]
                    for level, example_list in examples.items():
                        if isinstance(example_list, list):
                            patterns.extend(example_list)
        
        # keywords에서 패턴 추출
        if "keywords" in emotion_info:
            patterns.extend(emotion_info["keywords"])
            
        # triggers에서 패턴 추출
        if "triggers" in emotion_info:
            patterns.extend(emotion_info["triggers"])
            
        return patterns

    def extract_patterns(self, text: str) -> Dict[str, Any]:
        """EMOTIONS.json 기반 패턴 추출"""
        try:
            # 문장 분리
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            if not sentences:
                sentences = [text.strip()]
            
            # 각 문장에서 패턴 추출
            sentence_patterns = []
            for i, sentence in enumerate(sentences):
                patterns = self._extract_sentence_patterns(sentence)
                sentence_patterns.append({
                    "sentence_index": i,
                    "sentence": sentence,
                    "patterns": patterns
                })
            
            # 전체 텍스트 패턴 요약
            summary_patterns = self._summarize_extraction_patterns(sentence_patterns)
            
            return {
                "sentence_patterns": sentence_patterns,
                "summary_patterns": summary_patterns,
                "extraction_analysis": {
                    "total_sentences": len(sentences),
                    "pattern_types_found": len(summary_patterns),
                    "extraction_score": self._calculate_extraction_score(sentence_patterns)
                },
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "success": True
            }
            
        except Exception as e:
            return {
                "sentence_patterns": [],
                "summary_patterns": {},
                "extraction_analysis": {
                    "total_sentences": 0,
                    "pattern_types_found": 0,
                    "extraction_score": 0.0
                },
                "error": str(e),
                "success": False
            }

    def _extract_sentence_patterns(self, sentence: str) -> Dict[str, Any]:
        """문장에서 패턴 추출"""
        patterns = {
            "detected_emotions": [],
            "keywords": [],
            "expressions": [],
            "intensity_markers": [],
            "context_clues": []
        }
        
        sentence_lower = sentence.lower()
        
        # 기본 영어 감정 키워드 매핑
        english_emotion_mapping = {
            "happy": "희", "joy": "희", "pleasure": "희", "delight": "희",
            "angry": "노", "mad": "노", "furious": "노", "rage": "노",
            "sad": "애", "depressed": "애", "gloomy": "애", "sorrow": "애",
            "satisfaction": "락", "content": "락", "cheerful": "락"
        }
        
        # 영어 키워드 매칭
        for english_word, emotion in english_emotion_mapping.items():
            if english_word in sentence_lower:
                patterns["detected_emotions"].append({
                    "emotion": emotion,
                    "pattern": english_word,
                    "confidence": 0.8
                })
                patterns["keywords"].append(english_word)
        
        # EMOTIONS.json 패턴 매칭
        for emotion_path, extraction_patterns in self._extraction_patterns_cache.items():
            # 키워드 매칭
            for keyword in extraction_patterns.get("keywords", []):
                if keyword.lower() in sentence_lower:
                    patterns["detected_emotions"].append({
                        "emotion": emotion_path,
                        "pattern": keyword,
                        "confidence": 0.8
                    })
                    patterns["keywords"].append(keyword)
                    break
            
            # 표현 매칭
            for expression in extraction_patterns.get("expressions", []):
                if isinstance(expression, str) and expression.lower() in sentence_lower:
                    patterns["expressions"].append(expression)
            
            # 강도 마커 매칭
            for marker in extraction_patterns.get("intensity_markers", []):
                if marker in sentence:
                    patterns["intensity_markers"].append(marker)
            
            # 컨텍스트 단서 매칭
            for clue in extraction_patterns.get("context_clues", []):
                if clue.lower() in sentence_lower:
                    patterns["context_clues"].append(clue)
        
        return patterns

    def _summarize_extraction_patterns(self, sentence_patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """추출 패턴 요약"""
        summary = {
            "total_emotions": 0,
            "emotion_distribution": {},
            "keywords": [],
            "expressions": [],
            "intensity_markers": [],
            "context_clues": []
        }
        
        for sp in sentence_patterns:
            patterns = sp["patterns"]
            summary["total_emotions"] += len(patterns["detected_emotions"])
            
            # 감정 분포
            for emotion_info in patterns["detected_emotions"]:
                emotion = emotion_info["emotion"]
                summary["emotion_distribution"][emotion] = summary["emotion_distribution"].get(emotion, 0) + 1
            
            # 키워드
            summary["keywords"].extend(patterns["keywords"])
            
            # 표현
            summary["expressions"].extend(patterns["expressions"])
            
            # 강도 마커
            summary["intensity_markers"].extend(patterns["intensity_markers"])
            
            # 컨텍스트 단서
            summary["context_clues"].extend(patterns["context_clues"])
        
        # 중복 제거
        summary["keywords"] = list(set(summary["keywords"]))
        summary["expressions"] = list(set(summary["expressions"]))
        summary["intensity_markers"] = list(set(summary["intensity_markers"]))
        summary["context_clues"] = list(set(summary["context_clues"]))
        
        return summary

    def _calculate_extraction_score(self, sentence_patterns: List[Dict[str, Any]]) -> float:
        """추출 점수 계산"""
        if not sentence_patterns:
            return 0.0
        
        total_patterns = sum(len(sp["patterns"]["detected_emotions"]) for sp in sentence_patterns)
        avg_patterns_per_sentence = total_patterns / len(sentence_patterns)
        
        # 추출 점수 (0.0 ~ 1.0)
        extraction_score = min(1.0, avg_patterns_per_sentence / 2.0)
        return round(extraction_score, 2)

        # 구성 값(환경변수로 튜닝 가능, 상한만 제어)
        self.cfg = {
            # 패턴 상한(출력 경량화)
            "TOPK_KEYWORDS": int(os.environ.get("PE_TOPK_KEYWORDS", "1000")),
            "TOPK_PHRASES": int(os.environ.get("PE_TOPK_PHRASES", "900")),
            "TOPK_SITUATIONS": int(os.environ.get("PE_TOPK_SITUATIONS", "450")),
            "TOPK_INTENSITY_PER_LEVEL": int(os.environ.get("PE_TOPK_INTENSITY_PER_LEVEL", "60")),
            "TOPK_KEYPHRASES": int(os.environ.get("PE_TOPK_KEYPHRASES", "400")),
            "TOPK_POS_IND": int(os.environ.get("PE_TOPK_POS_IND", "200")),
            "TOPK_NEG_IND": int(os.environ.get("PE_TOPK_NEG_IND", "200")),
            # 전이/시간/컨텍스트 요약 상한
            "TOPK_TRANSITION": int(os.environ.get("PE_TOPK_TRANSITION", "20")),
            "TOPK_TEMPORAL": int(os.environ.get("PE_TOPK_TEMPORAL", "20")),
            "TOPK_CONTEXT": int(os.environ.get("PE_TOPK_CONTEXT", "25")),
            # 폴백 렉시콘 설정
            "ALLOW_FALLBACK_LEX": bool(os.environ.get("ALLOW_FALLBACK_LEX", "true").lower() in ("true", "1", "yes")),
            # 임베딩
            "MODEL_NAME": os.environ.get("PE_MODEL_NAME", "klue/bert-base"),
            "BATCH_SIZE": int(os.environ.get("PE_EMB_BATCH", "16")),
            "MAX_LEN": int(os.environ.get("PE_EMB_MAXLEN", "512")),
            "FP16": os.environ.get("PE_FP16", "1").lower() in ("1", "true", "yes"),
        }

        # 전처리 맵 (라벨링 데이터 기반)
        self.synonyms_map = self._build_synonyms_map(self.emotions_data)

        # 형태소기 (PATCH: Kiwi eager/lazy 선택)
        KIWI_EAGER = os.environ.get("PE_KIWI_EAGER", "1").lower() in ("1", "true", "yes")
        if _want_kiwi(default=True) and Kiwi is not None and KIWI_EAGER:
            try:
                self.kiwi = Kiwi()
                logger.info("Kiwi 초기화 완료")
            except Exception as e:
                self.kiwi = None
                logger.warning("Kiwi 초기화 실패, 기능을 비활성화합니다: %s", e)
        else:
            self.kiwi = None
            if Kiwi is None:
                logger.warning("Kiwi 라이브러리가 설치되어 있지 않습니다.")
            elif not KIWI_EAGER:
                logger.info("환경 설정(PE_KIWI_EAGER=0)에 따라 Kiwi는 lazy 모드로 동작합니다.")
            else:
                logger.info("환경 설정(PE_USE_KIWI)에 따라 Kiwi를 사용하지 않습니다.")

        # 컨텍스트/전이/시간 분석기는 데이터 기반 개선본 사용
        self.context_analyzer = PatternContextAnalyzer(self.emotions_data, kiwi_instance=self.kiwi)
        # processed_emotions는 이미 __init__에서 초기화됨
        logger.info("EmotionPatternExtractor 초기화 완료")

# =============================================================================
# Global extractor cache (reuse across runs, no re-embedding required)
# =============================================================================
_EXTRACTOR_CACHE: Dict[str, EmotionPatternExtractor] = {}
_EXTRACTOR_LOCK = threading.Lock()


def _norm_path(path: str) -> str:
    try:
        return os.path.normpath(os.path.abspath(path))
    except Exception:
        return str(path or "")


def get_pattern_extractor(
    emotions_data_path: Optional[str] = None, *, use_cache: bool = True
) -> EmotionPatternExtractor:
    """
    EmotionPatternExtractor 인스턴스를 캐시에서 재사용.
    - emotions_data_path가 None이면 기본 EMOTIONS.json을 사용.
    - use_cache=False로 호출하면 항상 새 인스턴스를 반환.
    """
    target_path = emotions_data_path or "EMOTIONS.json"
    if not use_cache:
        return EmotionPatternExtractor(target_path)

    key = _norm_path(target_path)
    with _EXTRACTOR_LOCK:
        inst = _EXTRACTOR_CACHE.get(key)
        if inst is None:
            inst = EmotionPatternExtractor(target_path)
            _EXTRACTOR_CACHE[key] = inst
        return inst


def clear_pattern_extractor_cache(emotions_data_path: Optional[str] = None) -> None:
    """
    캐시된 추출기 인스턴스를 비운다.
    - emotions_data_path가 지정되지 않으면 전체 캐시를 초기화.
    """
    with _EXTRACTOR_LOCK:
        if emotions_data_path:
            _EXTRACTOR_CACHE.pop(_norm_path(emotions_data_path), None)
        else:
            _EXTRACTOR_CACHE.clear()

    # ---------------------- utils ----------------------
    @staticmethod
    @lru_cache(maxsize=8192)
    def _norm(s: str) -> str:
        try:
            return unicodedata.normalize("NFC", s).strip().lower()
        except Exception:
            return (s or "").strip().lower()

    def _join_limited(self, items: List[Any], topk: int) -> str:
        """리스트를 안전하게 문자열로 결합. dict/숫자 등 아무 타입이 섞여도 동작."""
        if not items:
            return ""
        out: List[str] = []
        limit = max(0, int(topk))
        for it in items[:limit]:
            if it is None:
                continue
            if isinstance(it, str):
                s = it.strip()
                if s:
                    out.append(s)
            elif isinstance(it, (int, float)):
                out.append(str(it))
            elif isinstance(it, dict):
                # 전이 규칙/키워드 dict 등을 사람이 읽을 수 있는 짧은 문자열로 정리
                s = self._stringify_dict(it)
                if s:
                    out.append(s)
            else:
                # 기타 타입은 str()로
                s = str(it).strip()
                if s:
                    out.append(s)
        return " ".join(out)

    @staticmethod
    def _stringify_dict(d: Dict[str, Any]) -> str:
        """
        전이 규칙/패턴 dict를 간결 문자열로 변환:
        - from_emotion/to_emotion, triggers/trigger, pattern, name 등을 합성
        """
        try:
            frm = d.get("from_emotion") or d.get("from") or d.get("src") or ""
            to = d.get("to_emotion") or d.get("to") or d.get("dst") or ""
            pat = d.get("pattern") or ""
            trig = d.get("triggers") or d.get("trigger") or []
            if isinstance(trig, str):
                trig = [trig]
            trig = [t for t in trig if isinstance(t, str)]
            parts = []
            if frm or to:
                parts.append(f"{frm}->{to}".strip("->"))
            if pat:
                parts.append(str(pat))
            if trig:
                parts.append("|".join(trig))
            name = d.get("name") or d.get("id") or ""
            if name:
                parts.append(str(name))
            s = " / ".join(p for p in parts if p)
            return s if s else json.dumps(d, ensure_ascii=False)[:120]
        except Exception:
            return json.dumps(d, ensure_ascii=False)[:120]

    # ---------------------- I/O ------------------------
    def _load_pretrained_model(self, perf_bucket: Optional[Dict[str, float]] = None):
        """사전학습 모델 로드(모듈 캐시 재사용, FP16 선택)"""
        perf = perf_bucket if perf_bucket is not None else {}
        with _tick("load_model", perf):
            with EmotionPatternExtractor._MODEL_LOCK:
                if (
                    EmotionPatternExtractor._MODEL_CACHE["name"] == self.cfg["MODEL_NAME"]
                    and EmotionPatternExtractor._MODEL_CACHE["model"] is not None
                    and EmotionPatternExtractor._MODEL_CACHE["tokenizer"] is not None
                ):
                    self.model = EmotionPatternExtractor._MODEL_CACHE["model"]
                    self.tokenizer = EmotionPatternExtractor._MODEL_CACHE["tokenizer"]
                    return

                try:
                    model_name = self.cfg["MODEL_NAME"]
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    model = AutoModel.from_pretrained(model_name)
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    model.to(device)
                    model.eval()
                    if self.cfg["FP16"] and device.type == "cuda":
                        try:
                            model.half()
                        except Exception:
                            pass

                    EmotionPatternExtractor._MODEL_CACHE.update(
                        {"name": model_name, "tokenizer": tokenizer, "model": model, "device": device}
                    )
                    self.model = model
                    self.tokenizer = tokenizer
                    logger.info("사전학습 모델 '%s' 로드 완료 (device=%s, fp16=%s)", model_name, device, self.cfg['FP16'])
                except Exception as e:
                    logger.error("사전학습 모델 로드 실패: %s", str(e))
                    raise

    def _load_emotions_data(self, emotions_data_path: str) -> Dict[str, Any]:
        try:
            with open(emotions_data_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                logger.info("[라벨링뼈대 로드] 최상위 감정키: %s", list(data.keys()))
                return data
        except Exception as e:
            logger.error("EMOTIONS.json 데이터 로드 실패: %s", str(e))
            raise

    def _fallback_embeddings(self, patterns: Dict[str, Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        HF 모델이 없을 때를 위한 결정론적 임베딩(라벨 텍스트 해시 기반).
        - 입력 텍스트 → SHA256 → seed → N(0,1) 난수 → L2 정규화
        """
        import hashlib
        dim = int(os.environ.get("PE_FALLBACK_DIM", "384"))
        out: Dict[str, torch.Tensor] = {}
        for emo_id, pobj in patterns.items():
            try:
                text = self._gather_embedding_text(emo_id, pobj)
            except Exception as e:
                logger.error("[폴백 임베딩 텍스트 오류] %s: %s", emo_id, e, exc_info=True)
                text = emo_id
            h = hashlib.sha256(text.encode("utf-8")).digest()
            seed = int.from_bytes(h[:8], "little", signed=False)
            g = torch.Generator().manual_seed(seed)
            vec = torch.randn(dim, generator=g)
            vec = vec / (vec.norm() + 1e-9)
            out[emo_id] = vec
        return out

    # ------------------ synonyms map -------------------
    def _build_synonyms_map(self, emotions_data: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        대표/세부 감정의 core_keywords/synonyms/상황 변형 표현/예시/지표를 통합 (데이터 기반)
        - 임베딩 텍스트 구성의 재료로 활용
        """
        synonyms_map: Dict[str, List[str]] = {}
        related_terms_map: Dict[str, Set[str]] = {}
        try:
            for top_label, top_emotion_data in emotions_data.items():
                top_metadata = top_emotion_data.get("metadata", {}) or {}
                top_emotion_profile = top_emotion_data.get("emotion_profile", {}) or {}
                top_emotion_id = top_metadata.get("emotion_id", top_label)
                kset: Set[str] = set()

                # 대표 감정
                kset.update(top_emotion_profile.get("core_keywords", []) or [])
                kset.update(top_emotion_profile.get("synonyms", []) or [])

                rel = top_emotion_profile.get("related_emotions", {}) or {}
                for rtype in ("positive", "neutral", "negative"):
                    terms = rel.get(rtype, []) or []
                    related_terms_map.setdefault(top_emotion_id, set()).update(terms)
                    if rtype == "positive":
                        kset.update(terms)

                ling = top_emotion_data.get("linguistic_patterns", {}) or {}
                for kp in ling.get("key_phrases", []) or []:
                    patt = (kp.get("pattern") or "").replace("[XXX]", "").replace("[XX]", "").strip()
                    if patt:
                        kset.add(patt)

                senti = top_emotion_data.get("sentiment_analysis", {}) or {}
                kset.update(senti.get("positive_indicators", []) or [])
                kset.update(senti.get("negative_indicators", []) or [])

                synonyms_map[top_emotion_id] = list(kset)

                # 세부 감정
                for sub_name, sub_data in (top_emotion_data.get("sub_emotions", {}) or {}).items():
                    sub_meta = sub_data.get("metadata", {}) or {}
                    sub_id = sub_meta.get("emotion_id", sub_name)
                    skset: Set[str] = set()

                    sub_profile = sub_data.get("emotion_profile", {}) or {}
                    skset.update(sub_profile.get("core_keywords", []) or [])
                    skset.update(sub_profile.get("synonyms", []) or [])

                    srel = sub_profile.get("related_emotions", {}) or {}
                    for rtype in ("positive", "neutral", "negative"):
                        terms = srel.get(rtype, []) or []
                        related_terms_map.setdefault(sub_id, set()).update(terms)
                        if rtype == "positive":
                            skset.update(terms)

                    sctx = sub_data.get("context_patterns", {}) or {}
                    for _, sinfo in (sctx.get("situations", {}) or {}).items():
                        skset.update(sinfo.get("keywords", []) or [])
                        skset.update(sinfo.get("variations", []) or [])
                        if sinfo.get("core_concept"):
                            skset.add(sinfo["core_concept"])
                        for ex in sinfo.get("examples", []) or []:
                            ex = (ex or "").strip()
                            if len(ex) >= 2:
                                skset.add(ex)

                    sling = sub_data.get("linguistic_patterns", {}) or {}
                    for kp in sling.get("key_phrases", []) or []:
                        patt = (kp.get("pattern") or "").replace("[XXX]", "").replace("[XX]", "").strip()
                        if patt:
                            skset.add(patt)

                    ssent = sub_data.get("sentiment_analysis", {}) or {}
                    skset.update(ssent.get("positive_indicators", []) or [])
                    skset.update(ssent.get("negative_indicators", []) or [])

                    # 강도 예시
                    ilv = sub_profile.get("intensity_levels", {}) or {}
                    for lv_ex in (ilv.get("intensity_examples", {}) or {}).values():
                        for ex in (lv_ex or []):
                            ex = (ex or "").strip()
                            if len(ex) >= 2:
                                skset.add(ex)

                    synonyms_map[sub_id] = list(skset)

            # 후처리: 정제 + 상한
            for emo_id, kws in synonyms_map.items():
                uniq, seen = [], set()
                for kw in kws:
                    n = self._norm(kw)
                    if not n or all(c in "[]{}()*/\\'\" " for c in n):
                        continue
                    if n not in seen:
                        seen.add(n)
                        uniq.append(kw)
                if emo_id in related_terms_map:
                    uniq.extend(sorted(set(related_terms_map[emo_id]) - set(uniq)))
                synonyms_map[emo_id] = uniq
            logger.info("[_build_synonyms_map] 총 %d개 감정의 키워드 통합 완료", len(synonyms_map))
            logger.debug("감정별 평균 키워드 수: %.1f", sum(len(v) for v in synonyms_map.values()) / max(len(synonyms_map),1))
            return synonyms_map
        except Exception as e:
            logger.error("synonyms_map 구성 중 오류 발생: %s", str(e), exc_info=True)
            return {}

    # ------------------ transitions bundle ------------------
    def analyze_pattern_transitions(self, text: str, perf_bucket: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        텍스트 내 감정 패턴 전이 분석(개선):
        - PatternContextAnalyzer의 문장 분할/문맥·전이/시간 신호를 재사용
        - 전이 포인트 필터(사소 전이 제외), 계층 요약, 분포 요약
        """
        perf = perf_bucket if perf_bucket is not None else {}
        with _tick("analyze_transitions", perf):
            try:
                flow = self.context_analyzer.analyze_emotion_flow(text)
                rules = self.context_analyzer.analyze_emotion_transition(text)

                # 전이 포인트 필터링: 사소 전이 제거
                tpoints = flow.get("emotional_state_changes", []) or []
                as_transition_points = []
                for sc in tpoints:
                    as_transition_points.append({
                        "position": sc.get("position"),
                        "added_emotions": sc.get("added_emotions", []),
                        "removed_emotions": sc.get("removed_emotions", []),
                        "intensity_changes": sc.get("intensity_changes", {})
                    })

                filtered_points = self._filter_transition_points(
                    as_transition_points,
                    min_emotion_diff=int(os.environ.get("PE_MIN_EMO_DIFF", "1")),
                    consider_intensity=True,
                    intensity_threshold=os.environ.get("PE_INTENSITY_JUMP", "medium->high")
                )

                distribution = self._analyze_transition_distribution([
                    {
                        "type": "emotion_shift",
                        "primary_changes": {"added": tp.get("added_emotions", []), "removed": tp.get("removed_emotions", [])},
                        "secondary_changes": {"added": [], "removed": []},
                        "intensity_changes": tp.get("intensity_changes", {}),
                        "position": tp.get("position", 0),
                    }
                    for tp in filtered_points
                ])

                # Top-K 제한
                if hasattr(self.context_analyzer, "cfg"):
                    flow["transitions"] = flow.get("transitions", [])[: self.context_analyzer.cfg["MAX_TRANS_PER_TEXT"]]
                    rules["transitions"] = rules.get("transitions", [])[: self.context_analyzer.cfg["TOPK_TRANSITION"]]

                return {
                    "flow": flow,
                    "rule_based": rules,
                    "filtered_transition_points": filtered_points[: self.cfg["TOPK_TRANSITION"]],
                    "summary": {
                        "flow_transitions": len(flow.get("transitions", [])),
                        "rule_transitions": len(rules.get("transitions", [])),
                        "filtered_transition_points": len(filtered_points),
                        "distribution": distribution.get("transition_type_counts", {}),
                    },
                }
            except Exception as e:
                logger.error("[analyze_pattern_transitions] 오류: %s", e, exc_info=True)
                return {"flow": {}, "rule_based": {}, "filtered_transition_points": [], "summary": {"error": str(e)}}

    def _get_dominant_patterns(self, nested_transitions: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not nested_transitions:
            return {
                "dominant_emotion": ("N/A", 0),
                "dominant_situation": ("N/A", 0),
                "dominant_intensity": ("N/A", 0),
                "emotions_frequency_map": {},
                "situations_frequency_map": {},
                "intensities_frequency_map": {},
            }
        emotions_counter = Counter()
        situations_counter = Counter()
        intensities_counter = Counter()
        for nt in nested_transitions:
            for p in nt.get("patterns", []):
                emotions_counter[p.get("emotion", "unknown_emotion")] += 1
                situations_counter[p.get("situation", "unknown_situation")] += 1
                intensities_counter[p.get("intensity", "medium")] += 1
        return {
            "dominant_emotion": (emotions_counter.most_common(1)[0] if emotions_counter else ("N/A", 0)),
            "dominant_situation": (situations_counter.most_common(1)[0] if situations_counter else ("N/A", 0)),
            "dominant_intensity": (intensities_counter.most_common(1)[0] if intensities_counter else ("N/A", 0)),
            "emotions_frequency_map": dict(emotions_counter),
            "situations_frequency_map": dict(situations_counter),
            "intensities_frequency_map": dict(intensities_counter),
        }

    def _analyze_transition_distribution(self, transition_points: List[Dict[str, Any]]) -> Dict[str, Any]:
        counter = Counter()
        tot_pa = tot_pr = tot_sa = tot_sr = 0
        icc = 0
        pos = []
        for t in transition_points:
            counter[t.get("type", "unknown_transition")] += 1
            pa = len(t.get("primary_changes", {}).get("added", []))
            pr = len(t.get("primary_changes", {}).get("removed", []))
            sa = len(t.get("secondary_changes", {}).get("added", []))
            sr = len(t.get("secondary_changes", {}).get("removed", []))
            tot_pa += pa
            tot_pr += pr
            tot_sa += sa
            tot_sr += sr
            if t.get("intensity_changes"):
                icc += len(t["intensity_changes"])
            pos.append(t.get("position", 0))
        return {
            "transition_type_counts": dict(counter),
            "primary_changes": {"added_total": tot_pa, "removed_total": tot_pr},
            "secondary_changes": {"added_total": tot_sa, "removed_total": tot_sr},
            "intensity_change_count": icc,
            "positions_range": (min(pos) if pos else None, max(pos) if pos else None),
        }

    # ------------------ intensity analysis ------------------
    def _analyze_emotion_intensity(self, text: str, emotion_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        강도 분석(완전 데이터 기반):
        - intensity_examples(라벨) 유사도 + 전역 증폭/감쇠(라벨 합집합) + 상황 키워드 강도(weight)만 사용
        - 하드코딩 모디파이어 제거
        """
        emotion_profile = emotion_data.get("emotion_profile", {}) or {}
        intensity_levels = emotion_profile.get("intensity_levels", {}) or {}
        result = {
            "level": "medium",
            "description": intensity_levels.get("medium", "중간 강도의 감정 상태"),
            "confidence": 0.0,
            "evidence": [],
        }
        try:
            level_desc = {
                "low": intensity_levels.get("low", "낮은 강도의 감정 상태"),
                "medium": intensity_levels.get("medium", "중간 강도의 감정 상태"),
                "high": intensity_levels.get("high", "높은 강도의 감정 상태"),
            }
            il_ex = (intensity_levels.get("intensity_examples", {}) or {})
            scores = {"low": 0.0, "medium": 0.0, "high": 0.0}
            evid = {"low": [], "medium": [], "high": []}

            tokens = set(self.context_analyzer._tokenize(text))

            # ① 예시 기반 점수
            for lv, exs in (il_ex.items() if isinstance(il_ex, dict) else []):
                for ex in exs or []:
                    ex_words = set(self.context_analyzer._tokenize(ex))
                    inter = ex_words & tokens
                    if inter:
                        sim = len(inter) / max(1, len(ex_words))
                        scores[lv] += sim
                        if sim > 0.3:
                            evid[lv].append({
                                "type": "example_match",
                                "text": ex,
                                "similarity": sim,
                                "matched_keywords": list(inter),
                            })

            # ② 전역 증폭/감쇠(라벨 합집합) — PatternContextAnalyzer에서 수집
            gl = getattr(self.context_analyzer, "_build_global_intensity_modifiers", None)
            gl = gl() if callable(gl) else {"amplifiers": set(), "diminishers": set()}
            amp_hits = [a for a in gl.get("amplifiers", set()) if a in tokens]
            dim_hits = [d for d in gl.get("diminishers", set()) if d in tokens]
            if amp_hits:
                scores["high"] += min(2.0, 0.6 + 0.2 * len(amp_hits))
                evid["high"].append({"type": "amplifier_global", "hits": amp_hits})
            if dim_hits:
                scores["low"] += min(2.0, 0.5 + 0.2 * len(dim_hits))
                evid["low"].append({"type": "diminisher_global", "hits": dim_hits})

            # ③ 상황 강도 가중(라벨)
            ctx = emotion_data.get("context_patterns", {}) or {}
            for sname, sinfo in (ctx.get("situations", {}) or {}).items():
                kws = [self._norm(k) for k in (sinfo.get("keywords", []) or [])]
                matched = [k for k in kws if k in tokens]
                if matched:
                    lv = (sinfo.get("intensity") or "medium").lower()
                    add = min(1.5, 0.4 + 0.2 * len(matched))
                    scores[lv] += add
                    evid[lv].append({
                        "type": "situation",
                        "situation": sname,
                        "matched_keywords": matched,
                        "weight": add,
                    })

            total = sum(scores.values())
            if total > 0:
                norm = {k: v / total for k, v in scores.items()}
                mx = max(norm.items(), key=lambda x: x[1])[0]
                conf = norm[mx]
                result = {
                    "level": mx,
                    "description": level_desc.get(mx, ""),
                    "confidence": conf,
                    "evidence": evid[mx],
                    "scores": norm,
                    "analysis_details": {
                        "total_evidence_count": sum(len(v) for v in evid.values()),
                        "level_evidence_counts": {k: len(v) for k, v in evid.items()},
                    },
                }
        except Exception as e:
            logger.error("감정 강도 분석 중 오류 발생: %s", e, exc_info=True)
        return result

    def _intensity_value(self, level: str) -> int:
        return {"low": 1, "medium": 2, "high": 3}.get(level or "", 0)

    # ------------------ filter transitions ------------------
    def _filter_transition_points(
        self,
        transition_points: List[Dict[str, Any]],
        min_emotion_diff: int = 3,
        consider_intensity: bool = True,
        intensity_threshold: str = "medium->high",
    ) -> List[Dict[str, Any]]:
        filtered = []
        parts = intensity_threshold.split("->")
        from_lv, to_lv = (parts[0], parts[1]) if len(parts) == 2 else ("medium", "high")
        for t in transition_points:
            diff_size = len(t.get("added_emotions", [])) + len(t.get("removed_emotions", []))
            keep = diff_size >= min_emotion_diff
            if consider_intensity and not keep:
                for _, ch in (t.get("intensity_changes", {}) or {}).items():
                    if (
                        (ch.get("from", {}).get("level") == from_lv and ch.get("to", {}).get("level") == to_lv)
                        or (ch.get("from", {}).get("level") == "low" and ch.get("to", {}).get("level") == "high")
                    ):
                        keep = True
                        break
            if keep:
                filtered.append(t)
            else:
                logger.debug(
                    f"[filter_transition_points] pos={t.get('position')}, diff_size={diff_size} -> filtered out"
                )
        return filtered

    # ------------------ process patterns ------------------
    def _merge_senti(self, a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
        out = {}
        out["positive_indicators"] = list(set((a.get("positive_indicators", []) or []) + (b.get("positive_indicators", []) or [])))
        out["negative_indicators"] = list(set((a.get("negative_indicators", []) or []) + (b.get("negative_indicators", []) or [])))
        out["intensity_modifiers"] = {
            "amplifiers":  list(set((a.get("intensity_modifiers", {}) or {}).get("amplifiers", [])  + (b.get("intensity_modifiers", {}) or {}).get("amplifiers", []))),
            "diminishers": list(set((a.get("intensity_modifiers", {}) or {}).get("diminishers", []) + (b.get("intensity_modifiers", {}) or {}).get("diminishers", []))),
        }
        return out

    def _stringify_transition_pattern(self, p: Any) -> Optional[str]:
        """전이 패턴 항목을 문자열로 표준화."""
        if p is None:
            return None
        if isinstance(p, str):
            return p.strip() or None
        if isinstance(p, dict):
            return self._stringify_dict(p)
        try:
            return str(p).strip() or None
        except Exception:
            return None

    def _collect_node_patterns(self, node: Dict[str, Any], pat: Dict[str, Any]) -> None:
        """하나의 라벨 노드에서 패턴 수집/병합(대표/세부 공통)"""
        ep = node.get("emotion_profile", {}) or {}
        pat["keywords"].update(ep.get("core_keywords", []) or [])
        pat["keywords"].update(ep.get("synonyms", []) or [])

        # 강도 예시
        il = (ep.get("intensity_levels", {}) or {})
        for lv, lv_ex in ((il.get("intensity_examples", {}) or {}).items()):
            lv_lower = (lv or "").lower()
            if lv_lower in pat["intensity_patterns"]:
                pat["intensity_patterns"][lv_lower].update(lv_ex or [])

        # 상황
        cpx = (node.get("context_patterns", {}) or {})
        for _, sinfo in ((cpx.get("situations", {}) or {}).items()):
            if sinfo.get("description"):
                pat["situations"].add(sinfo["description"])
            pat["keywords"].update(sinfo.get("keywords", []) or [])
            pat["phrases"].update(sinfo.get("examples", []) or [])
            if sinfo.get("variations"):
                pat["keywords"].update(sinfo.get("variations", []) or [])

        # 언어 패턴
        ling = (node.get("linguistic_patterns", {}) or {})
        for kp in ling.get("key_phrases", []) or []:
            patt = (kp.get("pattern") or "").replace("[XXX]", "").replace("[XX]", "").strip()
            if patt:
                pat.setdefault("key_phrases", set()).add(patt)

        # 전이 규칙/감성 지표
        if "emotion_transitions" in node:
            pat.setdefault("emotion_transitions", {}).setdefault("patterns", [])
            raw_pats = (node.get("emotion_transitions", {}) or {}).get("patterns", []) or []
            for rp in raw_pats:
                sp = self._stringify_transition_pattern(rp)
                if sp:
                    pat["emotion_transitions"]["patterns"].append(sp)

        if "sentiment_analysis" in node:
            cur = pat.get("sentiment_analysis", {}) or {}
            pat["sentiment_analysis"] = self._merge_senti(cur, node.get("sentiment_analysis", {}) or {})

    def _process_emotion_pattern(self, emotion_id: str, emotion_data: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """대표 노드 + 모든 세부 노드 라벨을 합쳐 데이터 기반 패턴을 구성"""
        try:
            with self.cache_lock:
                if emotion_id in self.pattern_cache:
                    # 접근시간 갱신
                    self.pattern_cache[emotion_id]["last_access"] = time.time()
                    return emotion_id, self.pattern_cache[emotion_id]

            pat: Dict[str, Any] = {
                "keywords": set(),
                "phrases": set(),
                "situations": set(),
                "intensity_patterns": {"high": set(), "medium": set(), "low": set()},
                "context_patterns": {},
                "linguistic_patterns": {},
                "emotion_transitions": {},
                "sentiment_analysis": {},
                "compiled_patterns": {},
                "key_phrases": set(),
                "weights": {},
                "last_access": time.time(),
            }

            # 대표 노드 수집
            self._collect_node_patterns(emotion_data, pat)

            # 세부 노드 수집
            subs = (emotion_data.get("sub_emotions", {}) or {})
            for _, sdata in subs.items():
                self._collect_node_patterns(sdata, pat)

            # Top-K 상한 적용 + 리스트 변환
            pat["keywords"]    = list(pat["keywords"])[: self.cfg["TOPK_KEYWORDS"]]
            pat["phrases"]     = list(pat["phrases"])[: self.cfg["TOPK_PHRASES"]]
            pat["situations"]  = list(pat["situations"])[: self.cfg["TOPK_SITUATIONS"]]
            pat["key_phrases"] = list(pat.get("key_phrases", set()))[: self.cfg["TOPK_KEYPHRASES"]]

            for lv in ("high", "medium", "low"):
                if lv in pat["intensity_patterns"]:
                    pat["intensity_patterns"][lv] = list(pat["intensity_patterns"][lv])[: self.cfg["TOPK_INTENSITY_PER_LEVEL"]]

            # 감성 지표 상한
            sa = pat.get("sentiment_analysis", {}) or {}
            sa["positive_indicators"] = (sa.get("positive_indicators", []) or [])[: self.cfg["TOPK_POS_IND"]]
            sa["negative_indicators"] = (sa.get("negative_indicators", []) or [])[: self.cfg["TOPK_NEG_IND"]]
            pat["sentiment_analysis"] = sa

            with self.cache_lock:
                self.pattern_cache[emotion_id] = pat
                # 캐시 용량 초과시 LRU 제거
                if len(self.pattern_cache) > self.MAX_PATTERN_CACHE:
                    oldest = min(self.pattern_cache.items(), key=lambda kv: kv[1].get("last_access", 0.0))[0]
                    self.pattern_cache.pop(oldest, None)
            return emotion_id, pat
        except Exception as e:
            logger.error("[오류] _process_emotion_pattern() 실패 (감정 ID=%s): %s", emotion_id, e, exc_info=True)
            return emotion_id, {}

    def extract_emotion_patterns(self, perf_bucket: Optional[Dict[str, float]] = None) -> Dict[str, Dict[str, Any]]:
        """EMOTIONS.json 라벨을 전수 스캔하여 감정별 패턴 뭉치를 구성"""
        perf = perf_bucket if perf_bucket is not None else {}
        with _tick("extract_patterns", perf):
            try:
                self.processed_emotions.clear()
                patterns: Dict[str, Dict[str, Any]] = {}
                if not isinstance(self.emotions_data, dict):
                    logger.error("emotions_data가 dict 형태가 아닙니다.")
                    return {}
                with _choose_executor() as ex:  # PATCH D2: 실행기 선택 헬퍼 사용
                    futures = []
                    for top_label, top_data in self.emotions_data.items():
                        top_meta = top_data.get("metadata", {}) or {}
                        top_id = top_meta.get("emotion_id", top_label)
                        futures.append(ex.submit(self._process_emotion_pattern, top_id, top_data))
                    for fut in as_completed(futures):
                        e_id, p = fut.result()
                        if p:
                            patterns[e_id] = p
                # 오래된 캐시 정리
                with self.cache_lock:
                    now = time.time()
                    keep = {}
                    for cid, cpat in self.pattern_cache.items():
                        if now - cpat.get("last_access", now) <= 3600:
                            keep[cid] = cpat
                    self.pattern_cache = keep
                return patterns
            except Exception as e:
                logger.critical(f"[오류] extract_emotion_patterns() 예외: {e}", exc_info=True)
                return {}

    def validate_emotion_patterns(self, patterns: Dict[str, Dict[str, Any]]) -> bool:
        """구성 필드/레벨 유무만 점검(라벨 기반이므로 내용은 데이터 품질에 의존)"""
        try:
            if not patterns:
                logger.error("패턴이 비어 있습니다.")
                return False
            required = {
                "keywords",
                "phrases",
                "situations",
                "intensity_patterns",
                "context_patterns",
                "linguistic_patterns",
                "emotion_transitions",
                "sentiment_analysis",
                "key_phrases",
            }
            for eid, pobj in patterns.items():
                missing = required - set(pobj.keys())
                if missing:
                    logger.warning(f"감정 {eid} 필드 누락: {missing}")
                if not {"high", "medium", "low"}.issubset(set(pobj.get("intensity_patterns", {}).keys())):
                    logger.error(f"{eid} - 강도 패턴 레벨 누락")
            logger.info("[validate_emotion_patterns] 기본 필드 검증 완료")
            return True
        except Exception as e:
            logger.critical(f"패턴 검증 중 오류: {e}", exc_info=True)
            return False

    # ------------------ embeddings ------------------
    def _gather_embedding_text(self, emo_id: str, pobj: Dict[str, Any]) -> str:
        """
        임베딩 입력 텍스트 구성(데이터 기반):
        - keywords + phrases + situations + key_phrases + sentiment indicators + transition patterns + intensity examples
        - 어떤 리스트에도 dict가 섞여 있어도 안전하게 문자열화하여 결합
        """
        parts: List[str] = []
        parts.append(self._join_limited(pobj.get("keywords", [])    or [], self.cfg["TOPK_KEYWORDS"]))
        parts.append(self._join_limited(pobj.get("phrases", [])     or [], self.cfg["TOPK_PHRASES"]))
        parts.append(self._join_limited(pobj.get("situations", [])  or [], self.cfg["TOPK_SITUATIONS"]))
        parts.append(self._join_limited(pobj.get("key_phrases", []) or [], self.cfg["TOPK_KEYPHRASES"]))

        sa = pobj.get("sentiment_analysis", {}) or {}
        parts.append(self._join_limited(sa.get("positive_indicators", []) or [], self.cfg["TOPK_POS_IND"]))
        parts.append(self._join_limited(sa.get("negative_indicators", []) or [], self.cfg["TOPK_NEG_IND"]))

        tr_pats = (pobj.get("emotion_transitions", {}) or {}).get("patterns", []) or []
        parts.append(self._join_limited(tr_pats, self.cfg["TOPK_TRANSITION"]))

        ints = pobj.get("intensity_patterns", {}) or {}
        for lv in ("high", "medium", "low"):
            parts.append(self._join_limited(ints.get(lv, []) or [], self.cfg["TOPK_INTENSITY_PER_LEVEL"]))

        combined = " ".join([p for p in parts if p]).strip()
        return combined if combined else emo_id

    def _get_embedding_cache_engine(self):
        if not self._use_embedding_cache or self._embedding_cache_error:
            return None
        with self._embedding_cache_lock:
            if self._embedding_cache_engine is not None:
                return self._embedding_cache_engine
            get_global_cache_engine = None
            try:
                from src.embedding_cache import get_global_cache_engine as _engine
                get_global_cache_engine = _engine
            except ImportError:
                try:
                    from embedding_cache import get_global_cache_engine as _engine  # type: ignore
                    get_global_cache_engine = _engine
                except ImportError:
                    get_global_cache_engine = None
            if get_global_cache_engine is None:
                self._embedding_cache_error = True
                return None
            try:
                self._embedding_cache_engine = get_global_cache_engine()
                return self._embedding_cache_engine
            except Exception as e:
                logger.debug("[pattern_extractor] embedding cache init 실패: %s", e, exc_info=True)
                self._embedding_cache_error = True
                return None

    def _compute_embedding_np(self, text: str) -> np.ndarray:
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
        text = text.strip()

        self._load_pretrained_model(None)
        model = EmotionPatternExtractor._MODEL_CACHE["model"]
        tokenizer = EmotionPatternExtractor._MODEL_CACHE["tokenizer"]
        device = EmotionPatternExtractor._MODEL_CACHE["device"]

        if model is None or tokenizer is None:
            return np.zeros((768,), dtype=np.float32)

        dtype = np.float16 if (self.cfg.get("FP16") and device is not None and getattr(device, "type", "") == "cuda") else np.float32
        hidden = getattr(getattr(model, "config", None), "hidden_size", 768)
        if not text:
            return np.zeros((hidden,), dtype=dtype)

        pool = os.environ.get("PE_EMB_POOL", "cls").lower()
        max_len = max(16, int(self.cfg["MAX_LEN"]))
        try:
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
        except Exception as e:
            logger.debug("[pattern_extractor] tokenizer 실패: %s", e, exc_info=True)
            return np.zeros((hidden,), dtype=dtype)

        try:
            inputs = {k: v.to(device) for k, v in inputs.items()}
            ctx = getattr(torch, "inference_mode", torch.no_grad)()
            with ctx:
                outputs = model(**inputs)
                hs = outputs.last_hidden_state  # [B, L, H]
                if pool == "mean" and "attention_mask" in inputs:
                    mask = inputs["attention_mask"].unsqueeze(-1)
                    denom = mask.sum(dim=1).clamp(min=1)
                    emb = (hs * mask).sum(dim=1) / denom
                else:
                    emb = hs[:, 0, :]
                vec = emb.squeeze(0).detach().cpu().numpy()
                if vec.dtype != dtype:
                    vec = vec.astype(dtype, copy=False)
                return vec
        except Exception as e:
            logger.debug("[pattern_extractor] 모델 추론 실패: %s", e, exc_info=True)
            return np.zeros((hidden,), dtype=dtype)
        finally:
            if device is not None and getattr(device, "type", "") == "cuda" and os.environ.get("PE_EMB_OFFLOAD", "1").lower() in ("1", "true", "yes"):
                try:
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                except Exception:
                    pass

    def generate_embeddings(self, patterns: Dict[str, Dict[str, Any]], perf_bucket: Optional[Dict[str, float]] = None) -> Dict[str, torch.Tensor]:
        """사전학습 모델 사용. 실패 시 결정론적 폴백 임베딩으로 대체. (PATCH C: OOM 회복 + inference_mode + 풀링)"""
        perf = perf_bucket if perf_bucket is not None else {}
        with _tick("generate_embeddings", perf):
            # 입력 텍스트 준비 (기존 그대로)
            try:
                all_texts: List[str] = []
                emo_order: List[str] = []
                for emo_id, pobj in patterns.items():
                    try:
                        combined = self._gather_embedding_text(emo_id, pobj)
                    except Exception as e:
                        logger.error("[임베딩 텍스트 구성 오류: %s] %s", emo_id, e, exc_info=True)
                        combined = emo_id
                    all_texts.append(combined)
                    emo_order.append(emo_id)
            except Exception as e:
                logger.error("[임베딩 텍스트 준비 전체 실패] %s", e, exc_info=True)
                try:
                    return self._fallback_embeddings(patterns)
                except Exception as ee:
                    logger.error("[폴백 임베딩 실패] %s", ee, exc_info=True)
                    return {}

            cache_engine = self._get_embedding_cache_engine()
            if cache_engine is not None:
                try:
                    out_map: Dict[str, torch.Tensor] = {}
                    for emo_id, combined in zip(emo_order, all_texts):
                        vec_np = cache_engine.get_embedding(combined, compute_func=self._compute_embedding_np)
                        tensor = torch.from_numpy(vec_np)
                        if tensor.dtype != torch.float32:
                            tensor = tensor.to(torch.float32)
                        out_map[emo_id] = tensor
                    return out_map
                except Exception as cache_exc:
                    logger.debug("[pattern_extractor] embedding cache 경로 실패, 기본 경로 사용: %s", cache_exc, exc_info=True)

            try:
                self._load_pretrained_model(perf)
                model = EmotionPatternExtractor._MODEL_CACHE["model"]
                tokenizer = EmotionPatternExtractor._MODEL_CACHE["tokenizer"]
                device = EmotionPatternExtractor._MODEL_CACHE["device"]

                pool = os.environ.get("PE_EMB_POOL", "cls").lower()  # 'cls' | 'mean'
                bs = max(1, int(self.cfg["BATCH_SIZE"]))
                max_len = max(16, int(self.cfg["MAX_LEN"]))
                out_map: Dict[str, torch.Tensor] = {}

                i = 0
                while i < len(all_texts):
                    try:
                        batch_texts = all_texts[i:i+bs]
                        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True,
                                           truncation=True, max_length=max_len)
                        inputs = {k: v.to(device) for k, v in inputs.items()}
                        # inference_mode가 no_grad보다 더 경량
                        ctx = getattr(torch, "inference_mode", torch.no_grad)()
                        with ctx:
                            outputs = model(**inputs)
                            hs = outputs.last_hidden_state  # [B, L, H]
                            if pool == "mean" and "attention_mask" in inputs:
                                mask = inputs["attention_mask"].unsqueeze(-1)  # [B, L, 1]
                                denom = mask.sum(dim=1).clamp(min=1)
                                emb = (hs * mask).sum(dim=1) / denom
                            else:
                                emb = hs[:, 0, :]  # CLS
                            emb = emb.detach().to("cpu")

                        for j, emo_id in enumerate(emo_order[i:i+bs]):
                            out_map[emo_id] = emb[j]
                        i += bs

                        # GPU 메모리 청소(옵션)
                        if device.type == "cuda" and os.environ.get("PE_EMB_OFFLOAD", "1") in ("1","true","yes"):
                            del inputs, outputs, hs, emb  # type: ignore
                            torch.cuda.synchronize()
                            torch.cuda.empty_cache()

                    except RuntimeError as rexc:
                        # 자동 OOM 회복
                        if "out of memory" in str(rexc).lower() and bs > 1 and device.type == "cuda":
                            bs = max(1, bs // 2)
                            logger.warning("[임베딩] OOM 감지 → 배치 크기 축소: bs=%d", bs)
                            torch.cuda.empty_cache()
                            continue
                        raise

                return out_map

            except Exception as e:
                logger.warning("[임베딩] HF 모델 경로/네트워크 문제로 폴백 사용: %s", e)
                try:
                    return self._fallback_embeddings(patterns)
                except Exception as ee:
                    logger.error("[폴백 임베딩 실패] %s", ee, exc_info=True)
                    return {}


# 별칭 export(상위 설정에서 PatternExtractor로 참조시 호환)
PatternExtractor = EmotionPatternExtractor

# =============================================================================
# Independent Function (FINAL)
# =============================================================================
from typing import Dict, Any, Optional
import json, os, logging


# 환경 다이얼(없으면 안전 기본값으로 동작)
_ENV_SCENETONE_W = float(os.environ.get("PE_SCENETONE_W", "0.6"))       # 0.5~0.7 권장
_ENV_RETURN_EMBED = os.environ.get("PE_RETURN_EMBED", "1") == "1"       # 임베딩 생성 on/off
_ENV_RETURN_TIME  = os.environ.get("PE_RETURN_TEMPORAL", "1") == "1"    # 시간 패턴 on/off
_ENV_USE_KIWI_AUTO = os.environ.get("PE_USE_KIWI", "auto").lower()      # auto/1/0

def _load_emotions(emotions_data_path: str) -> Dict[str, Any]:
    with open(emotions_data_path, "r", encoding="utf-8") as f:
        return json.load(f)

def _want_kiwi(default: Optional[bool] = None) -> bool:
    """
    PE_USE_KIWI=auto|1|0
      - auto: 설치되어 있으면 사용(기본)
      - 1: 강제 사용 시도(미설치면 자동 fallback)
      - 0: 사용 안 함
    """
    try:
        mode = _ENV_USE_KIWI_AUTO
        if mode == "1":
            return True
        if mode == "0":
            return False
        # auto
        return Kiwi is not None if default is None else default and (Kiwi is not None)
    except Exception:
        return False

def resolve_emotions_json_path(preferred_path: Optional[str] = None) -> str:
    """
    EMOTIONS.json 경로를 견고하게 탐색하여 절대경로로 반환.
    우선순위:
      1) 환경변수 EMOTIONS_JSON_PATH
      2) 인자로 받은 경로(preferred_path)
      3) 프로젝트 루트/현재 폴더의 후보명 또는 패턴(glob)
    """
    import glob
    tried: List[str] = []

    def _is_valid_file(p: str) -> bool:
        try:
            return os.path.isfile(p) and os.path.getsize(p) > 0
        except Exception:
            return False

    # 1) ENV
    env_path = os.environ.get("EMOTIONS_JSON_PATH")
    if env_path:
        p = os.path.normpath(os.path.abspath(env_path))
        if _is_valid_file(p):
            return p
        tried.append(p)

    # 2) Preferred path
    if preferred_path:
        p = os.path.normpath(os.path.abspath(preferred_path))
        if _is_valid_file(p):
            return p
        tried.append(p)

    # 3) 후보 경로(프로젝트 루트/현재 폴더)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    candidates = [
        os.path.join(project_root, "EMOTIONS.json"),
        os.path.join(project_root, "EMOTIONS.JSON"),
        os.path.join(current_dir, "EMOTIONS.json"),
        os.path.join(current_dir, "EMOTIONS.JSON"),
    ]
    for base in [project_root, current_dir]:
        candidates.extend(glob.glob(os.path.join(base, "EMOTIONS*.json")))
        candidates.extend(glob.glob(os.path.join(base, "emotions*.json")))

    seen = set()
    for c in candidates:
        c = os.path.normpath(c)
        if c.lower() in seen:
            continue
        seen.add(c.lower())
        if _is_valid_file(c):
            return c
        tried.append(c)

    tried_str = "\n - " + "\n - ".join(tried[:12])
    raise FileNotFoundError(
        "EMOTIONS.json 파일을 찾을 수 없습니다. 아래 경로들을 시도했습니다:" + tried_str
    )


def extract_emotion_patterns(emotions_data_path: str) -> Dict[str, Dict[str, Any]]:
    """전역 함수: 감정 패턴 추출."""
    extractor = EmotionPatternExtractor(emotions_data_path)
    return extractor.extract_emotion_patterns()


def validate_emotion_patterns(patterns: Dict[str, Dict[str, Any]]) -> bool:
    """
    전역 함수(개선본): 파일 로드/클래스 인스턴스화 없이
    extract_emotion_patterns(...) 결과의 '최소 유효성'을 검증.
    """
    try:
        if not isinstance(patterns, dict) or not patterns:
            logger.error("패턴이 비었거나 dict가 아닙니다.")
            return False

        required = {
            "keywords",
            "phrases",
            "situations",
            "intensity_patterns",
            "context_patterns",
            "linguistic_patterns",
            "emotion_transitions",
            "sentiment_analysis",
            "key_phrases",
        }

        ok = True
        for emotion_id, pobj in patterns.items():
            if not isinstance(pobj, dict):
                logger.error(f"{emotion_id} - 패턴 객체가 dict가 아닙니다.")
                ok = False
                continue

            # 필수 키 확인(누락은 경고로만 처리)
            missing = required - set(pobj.keys())
            if missing:
                logger.warning(f"감정 {emotion_id}에 일부 필드가 누락됨: {missing}")

            # 강도 패턴 레벨/형식 확인
            ip = pobj.get("intensity_patterns", {})
            if not isinstance(ip, dict) or not {"high", "medium", "low"}.issubset(set(ip.keys())):
                logger.error(f"{emotion_id} - 강도 패턴에 누락된 레벨이 있거나 형식이 올바르지 않습니다.")
                ok = False

            # 리스트형 필드 형식 경고(치명적 오류 아님)
            for key in ("keywords", "phrases", "situations"):
                if key in pobj and not isinstance(pobj[key], list):
                    logger.warning(
                        f"{emotion_id} - '{key}'는 list여야 합니다. (현재: {type(pobj[key]).__name__})"
                    )

        logger.info("[validate_emotion_patterns] 기본 필드 검증 완료")
        return ok
    except Exception as e:
        logger.critical(f"패턴 검증 중 오류: {str(e)}", exc_info=True)
        return False


def to_data_utils_format(analysis_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    (개선본) analyze_pattern_transitions(...) 최신 출력 구조를
    데이터 유틸 표준 스키마로 안전 변환.
    - emotionFlows: flow.emotion_progression
    - transitionPoints: filtered_transition_points > flow.transitions > rule_based.transitions
    - sequentialPatterns: flow.sequence
    - intensityChanges: flow.intensity_changes
    - timeSeries: flow.sequence, flow.temporal_clusters
    """
    try:
        flows = analysis_result.get("flow") or {}
        rules = analysis_result.get("rule_based") or {}
        filtered = analysis_result.get("filtered_transition_points") or []

        out = {
            "emotionFlows": flows.get("emotion_progression") or [],
            "transitionPoints": (
                filtered
                if filtered
                else (flows.get("transitions") or rules.get("transitions") or [])
            ),
            "sequentialPatterns": flows.get("sequence") or [],
            "intensityChanges": flows.get("intensity_changes") or [],
            "timeSeries": {
                "temporalSequence": flows.get("sequence") or [],
                "temporalClusters": flows.get("temporal_clusters") or [],
            },
            "summary": {
                "flowTransitionCount": len(flows.get("transitions") or []),
                "ruleTransitionCount": len(rules.get("transitions") or []),
                "filteredTransitionCount": len(filtered),
            },
        }
        logger.debug("[to_data_utils_format] 변환 결과 키: %s", list(out.keys()))
        return out
    except Exception as e:
        logger.exception(f"[to_data_utils_format] 변환 중 오류: {str(e)}")
        return {
            "emotionFlows": [],
            "transitionPoints": [],
            "sequentialPatterns": [],
            "intensityChanges": [],
            "timeSeries": {"temporalSequence": [], "temporalClusters": []},
            "summary": {"flowTransitionCount": 0, "ruleTransitionCount": 0, "filteredTransitionCount": 0},
            "error": str(e),
        }


def run_pattern_extraction(
    text: str,
    *,
    emotions_data: Optional[Dict[str, Any]] = None,
    emotions_data_path: Optional[str] = None,
    include_embeddings: Optional[bool] = None,
    include_temporal: Optional[bool] = None,
    return_duf: bool = False,           # True면 to_data_utils_format으로 변환해 반환
) -> Dict[str, Any]:
    """
    독립 함수 #1 (최종):
      - 텍스트에 대해 '감정 패턴 추출 → 검증 → 전이 분석 → (옵션) 시간패턴/임베딩'까지 원샷 수행.
      - EmotionPatternExtractor 클래스를 내부에서 생성.
      - return_duf=True 시, transition 분석 결과를 데이터유틸 표준스키마로 추가 변환하여 포함.
    """
    try:
        # emotions 데이터 로드 (주입값 우선 사용)
        edata = _load_emotions_data(emotions_data, emotions_data_path)
        
        # 빈 입력 즉시 폴백
        if not isinstance(text, str) or not text.strip():
            return {
                "matches": [],
                "count": 0,
                "error": None,
                "fallback": False,
                "success": True,
                "patterns": {},
                "valid": False,
                "transitions": {},
                "temporal_analysis": {},
                "embeddings": {},
                "perf": {}
            }
        
        # 성능 측정용 버킷
        perf = {}
        
        # 0) 옵션 해석(환경 변수 기본값과 병합)
        _inc_embed = _ENV_RETURN_EMBED if include_embeddings is None else bool(include_embeddings)
        _inc_time  = _ENV_RETURN_TIME  if include_temporal  is None else bool(include_temporal)

        # 1) 추출기 인스턴스화
        with _tick("init_extractor", perf):
            if emotions_data is not None:
                extractor = EmotionPatternExtractor(emotions_data_path or "EMOTIONS.json")
                extractor.emotions_data = emotions_data
                extractor.context_analyzer = PatternContextAnalyzer(emotions_data, kiwi_instance=extractor.context_analyzer.kiwi if hasattr(extractor.context_analyzer, "kiwi") else None)
            else:
                extractor = get_pattern_extractor(emotions_data_path or "EMOTIONS.json")

        # 2) 감정 패턴 추출
        extract_fn = getattr(extractor, "extract_emotion_patterns", None)
        if not callable(extract_fn):
            # 캐시에 오래된 인스턴스가 남아 있는 경우 초기화
            logger.warning("[pattern_extractor] extract_emotion_patterns() 미탑재 감지 → 캐시 초기화 후 재시도")
            try:
                clear_pattern_extractor_cache(emotions_data_path or "EMOTIONS.json")
            except Exception:
                logger.debug("[pattern_extractor] 캐시 초기화 실패 (무시 가능)", exc_info=True)
            extractor = EmotionPatternExtractor(emotions_data_path or "EMOTIONS.json")
            extract_fn = getattr(extractor, "extract_emotion_patterns", None)

        if callable(extract_fn):
            raw_patterns = extract_fn(perf)
        else:
            logger.warning("[pattern_extractor] extract_emotion_patterns 재시도 실패 → 사전 로드 캐시 사용")
            raw_patterns = getattr(extractor, "_emotion_patterns_cache", {})

        patterns: Dict[str, Dict[str, Any]] = {}
        wrapped_ids: Set[str] = set()
        if isinstance(raw_patterns, Mapping):
            for emo_id, pobj in raw_patterns.items():
                if isinstance(pobj, Mapping):
                    patterns[str(emo_id)] = dict(pobj)
                    continue
                if isinstance(pobj, Sequence) and not isinstance(pobj, (str, bytes)):
                    if emo_id not in wrapped_ids:
                        logger.warning("%s - 패턴 객체가 list/dict 외 타입(%s) → 간이 래핑", emo_id, type(pobj).__name__)
                        wrapped_ids.add(emo_id)
                    patterns[str(emo_id)] = {
                        "keywords": list(pobj),
                        "phrases": list(pobj),
                        "situations": [],
                        "intensity_patterns": {"high": [], "medium": [], "low": []},
                        "context_patterns": {},
                        "linguistic_patterns": {},
                        "emotion_transitions": {},
                        "sentiment_analysis": {},
                        "key_phrases": [],
                    }
                else:
                    logger.error("%s - 패턴 객체가 dict가 아닙니다.(%s) → 건너뜀", emo_id, type(pobj).__name__)
        else:
            logger.error("[pattern_extractor] 패턴 결과 타입이 dict가 아닙니다: %s", type(raw_patterns).__name__)

        # '실행 성공'을 기준으로 판단: 매치가 0건이어도 정상
        if not patterns:
            patterns = {}

        # 3) 패턴 유효성 검증(보수)
        with _tick("validate", perf):
            is_valid = validate_emotion_patterns(patterns) if patterns else False

        # 4) 감정 전이 분석
        analyze_transitions = getattr(extractor, "analyze_pattern_transitions", None)
        if callable(analyze_transitions):
            transitions = analyze_transitions(text, perf)
        else:
            logger.warning("[pattern_extractor] analyze_pattern_transitions() 미탑재 → 빈 전이 반환")
            transitions = {}

        # 5) (옵션) 시간 패턴
        with _tick("temporal", perf):
            temporal_analysis = extractor.context_analyzer.analyze_temporal_patterns(text) if _inc_time else {}

        # 6) (옵션) 임베딩
        embeddings = {}
        if _inc_embed:
            generate_embeddings_fn = getattr(extractor, "generate_embeddings", None)
            if callable(generate_embeddings_fn):
                embeddings = generate_embeddings_fn(patterns, perf)
            else:
                logger.warning("[pattern_extractor] generate_embeddings() 미탑재 → 임베딩 생략")
        
        # JSON-safe 변환 옵션 (PE_EMBED_JSON=1)
        if embeddings and os.environ.get("PE_EMBED_JSON", "0") in ("1", "true", "yes"):
            try:
                embeddings = {k: (v.detach().cpu().tolist() if hasattr(v, "detach") else v) 
                              for k, v in embeddings.items()}
            except Exception:
                pass

        # 패턴에서 매치 추출 (리스트 형태로 변환)
        matches = []
        if isinstance(patterns, dict):
            for emotion, pattern_data in patterns.items():
                if isinstance(pattern_data, dict) and "matches" in pattern_data:
                    matches.extend(pattern_data["matches"])
                elif isinstance(pattern_data, list):
                    matches.extend(pattern_data)

        out = {
            "matches": matches,
            "count": len(matches),
            "error": None,
            "fallback": False,
            "success": True,  # 실행 성공 기준
            "patterns": patterns,
            "valid": is_valid,
            "transitions": transitions,
            "temporal_analysis": temporal_analysis,
            "embeddings": embeddings,
            "perf": perf,  # 성능 지표 추가
        }

        if return_duf:
            out["transitions_duf"] = to_data_utils_format(transitions)

        return out
    except Exception as e:
        logging.exception(f"[run_pattern_extraction] 오류 발생: {str(e)}")
        return {
            "matches": [],
            "count": 0,
            "error": str(e),
            "fallback": False,
            "success": False,  # 오류 시 실패로 표시
            "patterns": {},
            "valid": False,
            "transitions": {},
            "temporal_analysis": {},
            "embeddings": {},
            "perf": {},
        }


def analyze_emotion_flow(
    text: str,
    emotions_data_path: str,
    *,
    use_kiwi: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    독립 함수 #2 (최종):
      - 문장 단위 감정 흐름만 간단 호출.
      - use_kiwi: True/False/None(auto). auto는 설치 여부에 따라 자동 결정.
    """
    try:
        # 빈 입력 즉시 폴백
        if not isinstance(text, str) or not text.strip():
            return {
                "sequence": [],
                "transitions": [],
                "temporal_clusters": [],
                "intensity_changes": [],
                "emotion_progression": []
            }
        
        emotions_data = _load_emotions(emotions_data_path)
        kiwi = Kiwi() if (_want_kiwi(True if use_kiwi is None else use_kiwi) and Kiwi is not None) else None
        context_analyzer = PatternContextAnalyzer(emotions_data, kiwi_instance=kiwi)
        return context_analyzer.analyze_emotion_flow(text)
    except Exception as e:
        logging.exception(f"[analyze_emotion_flow] 오류 발생: {str(e)}")
        return {"error": str(e)}


def get_temporal_analysis(
    text: str,
    emotions_data_path: str,
    *,
    use_kiwi: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    독립 함수 #3 (최종):
      - 시간 패턴(temporal) 분석만 빠르게 호출.
      - use_kiwi: True/False/None(auto)
    """
    try:
        # 빈 입력 즉시 폴백
        if not isinstance(text, str) or not text.strip():
            return {
                "temporal_sequence": [],
                "emotion_progression": [],
                "key_temporal_points": [],
                "temporal_clusters": [],
                "context_patterns": [],
                "emotion_contexts": {},
                "situation_temporal_mapping": {},
                "summary": {}
            }
        
        emotions_data = _load_emotions(emotions_data_path)
        kiwi = Kiwi() if (_want_kiwi(False if use_kiwi is None else use_kiwi) and Kiwi is not None) else None
        context_analyzer = PatternContextAnalyzer(emotions_data, kiwi_instance=kiwi)
        return context_analyzer.analyze_temporal_patterns(text)
    except Exception as e:
        logging.exception(f"[get_temporal_analysis] 오류: {str(e)}")
        return {"error": str(e)}



# =============================================================================
# A/B 벤치마킹 (PATCH D: 운영에서 바로 측정용)
# =============================================================================
def bench_extract_parallel(emotions_data_path: str, text_sample: str = "샘플 문장입니다.", runs: int = 3) -> dict:
    """실행기 설정별 성능 A/B 벤치마킹"""
    import time
    result = {}
    for name, mw in [("serial", 1), ("thread:auto", os.cpu_count() or 4), ("thread:x8", 8)]:
        os.environ["PE_MAX_WORKERS"] = str(mw)
        t = []
        for _ in range(runs):
            t0 = time.perf_counter()
            run_pattern_extraction(emotions_data_path, text_sample, include_embeddings=False, include_temporal=False)
            t.append((time.perf_counter() - t0) * 1000.0)
        result[name] = {"avg_ms": sum(t)/len(t), "runs_ms": t}
    return result

# =============================================================================
# 미니 검증 훅 (자동 회귀/간이 F1)
# =============================================================================
def _quick_eval(extractor, samples: List[Tuple[str, str]]) -> Dict[str, float]:
    """
    samples: [(text, expected_primary), ...]
    반환: {"acc": .., "coverage": ..}
    """
    ok = hit = 0
    for txt, exp in samples:
        try:
            res = extractor.context_analyzer.analyze_emotion_flow(txt)
            emos = res.get("emotion_progression", [])
            pred = None
            if emos:
                first = emos[0].get("emotions", [])
                if first:
                    pred = first[0].get("primary")
            ok += 1
            hit += 1 if pred == exp else 0
        except Exception as e:
            logger.warning(f"_quick_eval 예외: {e}")
            ok += 1  # 시도한 것으로 카운트
    return {"acc": hit/max(1, ok), "coverage": ok}


# =============================================================================
# Main Function (robust path + single extractor + JSON-safe save)
# =============================================================================
def _json_safe(obj):
    """set/tuple/Counter/torch.Tensor/np 등 JSON 미지원 타입을 안전 변환."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    # torch
    try:
        import torch  # noqa: F401
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
    except Exception:
        pass
    # numpy
    try:
        import numpy as np  # noqa: F401
        if "numpy" in str(type(obj)):
            try:
                return obj.tolist()
            except Exception:
                try:
                    return obj.item()
                except Exception:
                    pass
    except Exception:
        pass
    # set/tuple/list
    if isinstance(obj, (set, tuple, list)):
        return [_json_safe(x) for x in obj]
    # Counter / Mapping
    if isinstance(obj, Counter):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, Mapping):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    # 기타 시퀀스
    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
        return [_json_safe(x) for x in obj]
    # 마지막 안전망
    try:
        return str(obj)
    except Exception:
        return None


def _prune_situation_coverage(summary_dict: Dict[str, Any], keep_top: int = 20) -> None:
    """
    situation_coverage에서 matched_situations==0인 항목을 제거하고,
    매칭된 항목만 최대 keep_top개 유지해 JSON 용량을 줄인다.
    (요약 키는 유지, 값만 정리)
    """
    try:
        cov = summary_dict.get("situation_coverage", {})
        if not isinstance(cov, dict):
            return

        # 매칭 1 이상만 추려내기
        matched = {
            k: v for k, v in cov.items()
            if isinstance(v, dict) and v.get("matched_situations", 0) > 0
        }

        if not matched:
            # 전부 0이면 빈 맵으로 비워서 JSON을 가볍게
            summary_dict["situation_coverage"] = {}
            return

        # 매칭 많은 순으로 상위 keep_top만 유지
        top = sorted(
            matched.items(),
            key=lambda kv: kv[1].get("matched_situations", 0),
            reverse=True
        )[:keep_top]

        summary_dict["situation_coverage"] = dict(top)

    except Exception:
        # 실패해도 저장 흐름 방해하지 않기
        pass



def main():
    """
    감정 패턴 추출기 테스트 메인:
    - 패턴 추출/검증 → 전이 분석 → 시간 패턴 분석 → 임베딩 생성
    - 결과 파일명/경로는 기존 유지: logs/pattern_extractor.json
    """
    try:
        import glob

        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        # 통합 로그 관리자 사용 (날짜별 폴더)
        try:
            from log_manager import get_log_manager
            log_manager = get_log_manager()
            log_dir = str(log_manager.get_log_dir("emotion_analysis", use_date_folder=True))
        except ImportError:
            # 폴백: 기존 방식 (날짜별 폴더 추가)
            from datetime import datetime
            base_log_dir = os.path.join(current_dir, "logs")
            today = datetime.now().strftime("%Y%m%d")
            log_dir = os.path.join(base_log_dir, today)
            os.makedirs(log_dir, exist_ok=True)

        json_filename = "pattern_extractor.json"
        json_path = os.path.join(log_dir, json_filename)

        def _is_valid_file(p: str) -> bool:
            try:
                return os.path.isfile(p) and os.path.getsize(p) > 0
            except Exception:
                return False

        def _resolve_emotions_json_path() -> str:
            tried = []
            # 1) ENV
            env_path = os.environ.get("EMOTIONS_JSON_PATH")
            if env_path:
                p = os.path.normpath(os.path.abspath(env_path))
                if _is_valid_file(p):
                    return p
                tried.append(p)
            # 2) 프로젝트/현재 폴더 후보
            candidates = [
                os.path.join(project_root, "EMOTIONS.json"),
                os.path.join(project_root, "EMOTIONS.JSON"),
                os.path.join(current_dir, "EMOTIONS.json"),
                os.path.join(current_dir, "EMOTIONS.JSON"),
            ]
            # 3) 글롭
            for base in [project_root, current_dir]:
                candidates.extend(glob.glob(os.path.join(base, "EMOTIONS*.json")))
                candidates.extend(glob.glob(os.path.join(base, "emotions*.json")))
            seen = set()
            for c in candidates:
                c = os.path.normpath(c)
                if c.lower() in seen:
                    continue
                seen.add(c.lower())
                if _is_valid_file(c):
                    return c
                tried.append(c)
            tried_str = "\n - " + "\n - ".join(tried[:12])
            raise FileNotFoundError(
                "EMOTIONS.json 파일을 찾을 수 없습니다. 아래 경로들을 시도했습니다:" + tried_str
            )

        emotions_data_path = resolve_emotions_json_path()

        logger.info(f"프로젝트 루트 디렉토리: {project_root}")
        logger.info(f"EMOTIONS.json 경로: {emotions_data_path}")
        logger.info("감정 패턴 추출 테스트 시작")

        # 테스트용 문장들(기존 유지)
        test_texts = [
            """산책하는 순간 자연 속에서 풍요로움을 느꼈습니다. 
            일의 성과에 대한 만족으로 하루를 마감했습니다.""",
            """산책하는 길목에서 바람에 휩쓸리며 마음의 평화를 느꼈습니다.
            가족들과 함께한 저녁 식사는 즐거움으로 가득했습니다.""",
            """산책을 즐기다가 갑작스럽게 비가 내리기 시작했습니다. 하지만 우산을 가지고 있어서 다행이었습니다.""",
            """먼저 산책을 하고, 그다음에 저녁 식사를 했습니다."""
        ]

        # ★ Extractor 1회 초기화(모델/토크나이저 캐시 최대 활용)
        extractor = EmotionPatternExtractor(emotions_data_path)

        all_results = []
        test_success_count = 0

        for idx, text in enumerate(test_texts, 1):
            logger.info(f"\n=== 테스트 케이스 {idx} 분석 시작 ===")

            # 1) 패턴 추출
            patterns = extractor.extract_emotion_patterns()
            if not patterns:
                logger.error(f"테스트 케이스 {idx}: 감정 패턴 추출 실패 (patterns가 비어 있음)")
                continue

            # 2) 패턴 유효성 검증
            if not extractor.validate_emotion_patterns(patterns):
                logger.error(f"테스트 케이스 {idx}: 패턴 유효성 검증 실패")
                continue

            # 3) 전이(transition) 분석 (신/구 구조 모두 호환)
            transitions = extractor.analyze_pattern_transitions(text)

            # 4) 시간(temporal) 패턴 분석
            temporal_patterns = extractor.context_analyzer.analyze_temporal_patterns(text)

            # ✅ 패치 B: 0매칭 커버리지 제거 / 매칭된 항목만 상위 N개 유지
            if isinstance(temporal_patterns, dict) and "summary" in temporal_patterns:
                _prune_situation_coverage(temporal_patterns["summary"], keep_top=20)

            # 5) 임베딩 생성 (배치/캐시)
            embeddings = extractor.generate_embeddings(patterns)

            # --- 성공 여부 판단/로그 (기존 메시지 유지) ---
            # 우선순위: filtered_transition_points → transition_points → flow.transitions
            transition_points_count = 0
            if isinstance(transitions, dict):
                if "filtered_transition_points" in transitions:
                    transition_points_count = len(transitions.get("filtered_transition_points", []))
                elif "transition_points" in transitions:
                    transition_points_count = len(transitions.get("transition_points", []))
                else:
                    transition_points_count = len(transitions.get("flow", {}).get("transitions", []))

            embedding_count = len(embeddings)

            is_passed = True
            if transition_points_count == 0:
                logger.warning(f"테스트 케이스 {idx}: 전이 포인트가 0개입니다.")
            if embedding_count == 0:
                logger.warning(f"테스트 케이스 {idx}: 임베딩 생성이 0개입니다.")

            # 결과 로깅(기존 형식 유지)
            logger.info(f"테스트 케이스 {idx} 분석 결과:")
            logger.info(f"- 추출된 패턴 수: {len(patterns)}")
            logger.info(f"- 감정 전이 포인트 수: {transition_points_count}")
            logger.info(f"- 시간 표현 시퀀스 수: {len(temporal_patterns.get('temporal_sequence', []))}")
            logger.info(f"- 생성된 임베딩 수: {embedding_count}")

            # 대표 감정 패턴 상세(기존 유지)
            if patterns:
                first_emotion_id = list(patterns.keys())[0]
                first_pattern = patterns[first_emotion_id]
                logger.info(f"\n[테스트 케이스 {idx}] 첫 번째 감정 패턴 상세 정보:")
                logger.info(f"- 감정 ID: {first_emotion_id}")
                logger.info(f"- 키워드 수: {len(first_pattern['keywords'])}")
                logger.info(f"- 구문(phrases) 수: {len(first_pattern['phrases'])}")
                logger.info(f"- 상황(situations) 수: {len(first_pattern['situations'])}")

                logger.info(f"\n[테스트 케이스 {idx}] 강도별 패턴 분포:")
                for intensity, patterns_list in first_pattern["intensity_patterns"].items():
                    logger.info(f"- {intensity}: {len(patterns_list)} 패턴")

            if is_passed:
                test_success_count += 1

            # JSON 결과 누적(구조/키 이름 유지)
            case_result = {
                "test_case": idx,
                "input_text": text,
                "patterns_extracted": len(patterns),
                "transitions": transitions,
                "temporal_patterns": temporal_patterns,
                "embedding_count": embedding_count,
                "passed": is_passed,
            }
            all_results.append(case_result)

        # === JSON 저장(원자적 쓰기, 파일명/경로는 기존 유지) ===
        try:
            tmp_path = json_path + ".tmp"
            safe_results = _json_safe(all_results)  # ★ set/tuple/Counter/Tensor 등도 저장 가능
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(safe_results, f, ensure_ascii=False, indent=4)
            os.replace(tmp_path, json_path)
            logger.info(f"테스트 케이스 결과를 JSON으로 저장했습니다: {json_path}")
        except Exception as je:
            logger.exception(f"JSON 저장 중 오류: {je}")

        # 미니 검증 실행 (환경변수)
        if os.environ.get("PE_QUICK_EVAL", "0") == "1":
            logger.info("\n=== 미니 검증 실행 ===")
            samples = [
                ("너무 기쁘고 뿌듯합니다", "희"),
                ("짜증나고 실망스러웠다", "노"),
                ("슬프고 외로웠어요", "애"),
                ("편안하고 자유로운 기분", "락"),
            ]
            try:
                eval_result = _quick_eval(extractor, samples)
                logger.info(f"[quick-eval] {eval_result}")
            except Exception as e:
                logger.error(f"미니 검증 실행 중 오류: {e}")

        # 최종 통계 출력(기존 문구 유지)
        logger.info(f"\n=== 감정 패턴 추출 테스트 완료 ===")
        logger.info(f"총 {len(test_texts)}개 테스트 중 {test_success_count}개 성공.")
        if test_success_count == len(test_texts):
            logger.info("모든 테스트 케이스가 성공적으로 통과했습니다.")
        else:
            logger.warning("일부 테스트 케이스에서 경고 또는 실패가 발생했습니다.")

    except Exception as e:
        logger.error("테스트 실행 중 오류 발생: %s", str(e), exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    try:
        # 빠른 자체 확인 팁
        if os.environ.get("PE_PROF", "0") == "1":
            logger.info("프로파일링 모드 활성화됨 - [PERF] 로그를 확인하세요")
        if os.environ.get("PE_RETURN_EMBED", "1") == "0":
            logger.info("임베딩 생성 비활성화됨 - 더 빠른 테스트")
        
        logger.info("감정 패턴 추출 테스트 시작")
        main()
        logger.info("감정 패턴 추출 테스트 완료")
    except Exception as e:
        logger.error("메인 실행 중 오류 발생: %s", str(e), exc_info=True)
        sys.exit(1)

