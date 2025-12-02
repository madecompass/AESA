# transition_analyzer_extended.py
# -*- coding: utf-8 -*-
from __future__ import annotations


# Standard library
import json
import logging
import math
import os
import re
import sys
import threading
import time
from collections import Counter, OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import ( Any, Callable, Dict, Hashable, Iterable, List, NamedTuple, Optional, Tuple, )

# Third-party
# 선택 의존성: 미설치 환경에서도 안전하게 동작하도록 옵셔널 임포트
try:
    import kss  # sentence splitter
except Exception:
    kss = None  # type: ignore

try:
    import psutil  # memory monitor
except Exception:
    psutil = None  # type: ignore

try:
    import torch
except ImportError:
    torch = None


# =============================================================================
# Logger 설정
# =============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(BASE_DIR, 'logs')

def _setup_logger(log_dir: str) -> logging.Logger:
    logger = logging.getLogger("transition_main")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # 기본: 상위앱 비간섭
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())

    # Opt-in: 파일/콘솔 로그 (환경변수)
    if os.environ.get("TA_FILE_LOG", "0").lower() in ("1", "true", "yes"):
        try:
            os.makedirs(log_dir, exist_ok=True)
            from logging.handlers import RotatingFileHandler
            fh = RotatingFileHandler(os.path.join(log_dir, "transition_analyzer.log"),
                                     maxBytes=10*1024*1024, backupCount=5, encoding="utf-8")
            fh.setLevel(logging.INFO)
            fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
            logger.addHandler(fh)
        except Exception:
            pass

    if os.environ.get("TA_CONSOLE_LOG", "0").lower() in ("1", "true", "yes"):
        try:
            sh = logging.StreamHandler()
            sh.setLevel(logging.INFO)
            sh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
            logger.addHandler(sh)
        except Exception:
            pass

    return logger

# 전역 logger (이름 유지: 외부 코드 호환을 위해 동일 변수명 사용)
logger = _setup_logger(LOG_DIR)


# =============================================================================
# Utils: TransitionMetrics / EmotionNode / TransitionResult (Refined)
# =============================================================================
def _clip01(x: float) -> float:
    try:
        return 0.0 if x < 0.0 else 1.0 if x > 1.0 else float(x)
    except Exception:
        return 0.0

def _round3(x: float) -> float:
    try:
        return round(float(x), 3)
    except Exception:
        return 0.0

from functools import lru_cache

def _lower_list(items: Iterable[str]) -> List[str]:
    out: List[str] = []
    for it in items or []:
        if isinstance(it, str):
            s = it.strip().lower()
            if s:
                out.append(s)
    return out

@lru_cache(maxsize=128)
def _compile_boundary_regex_cached(word: str) -> re.Pattern:
    """단일 단어 정규식 컴파일 및 캐싱"""
    try:
        pat = rf"(?<!\w){re.escape(word)}(?!\w)"
        return re.compile(pat, re.IGNORECASE)
    except re.error:
        return re.compile(re.escape(word), re.IGNORECASE)

def _compile_boundary_regex_list(words: Iterable[str]) -> List[re.Pattern]:
    """리스트 컴파일 (캐시 활용)"""
    regs: List[re.Pattern] = []
    for w in _lower_list(words):
        regs.append(_compile_boundary_regex_cached(w))
    return regs

def _normalize_intensity_levels(levels: Dict[str, Any]) -> Dict[str, Any]:
    """
    EMOTIONS.json의 intensity_levels 정규화:
    { 'low|medium|high': {'weight':float, 'intensity_examples':{k:[str...]}} }
    """
    if not isinstance(levels, dict):
        return {}
    out: Dict[str, Any] = {}
    for lv in ("low", "medium", "high"):
        v = levels.get(lv, {}) or {}
        try:
            w = float(v.get("weight", 1.0))
        except Exception:
            w = 1.0
        ex = v.get("intensity_examples", {}) or {}
        if isinstance(ex, dict):
            ex = {str(k): [s for s in (lst or []) if isinstance(s, str) and s.strip()]
                  for k, lst in ex.items()}
        else:
            ex = {}
        out[lv] = {"weight": w, "intensity_examples": ex}
    return out

# =============================================================================
# DataClass: TransitionMetrics
# =============================================================================
@dataclass
class TransitionMetrics:
    """
    실행 요약 지표 컨테이너.
    - 유지 필드: processing_time, memory_usage, transition_count, pattern_count, confidence_avg
    - 추가 필드: flow/안정성/강도/마커/카테고리/메모
    - 모든 점수는 0~1 clip 후 반올림(3자리) 권장
    """
    processing_time: float
    memory_usage: int
    transition_count: int
    pattern_count: int
    confidence_avg: float

    flow_pattern: str = "unknown"
    stability_score: float = 0.0      # (1 - (unique-1)/(N-1)) 근사
    volatility: float = 0.0           # transitions / (N-1)
    peak_intensity_avg: float = 0.0
    intensifier_hits: int = 0
    marker_hits: int = 0
    category_counts: Dict[str, int] = field(default_factory=dict)  # {'희':N,'노':N,'애':N,'락':N}
    notes: Dict[str, Any] = field(default_factory=dict)

    # -------- convenience builder --------
    @classmethod
    def from_analysis(
        cls,
        emotion_sequence: List[str],
        transitions: List[Dict[str, Any]],
        processing_time: float,
        memory_usage: int,
        pattern_count: int,
        confidence_avg: float,
        flow_pattern: str = "unknown",
        marker_hits: int = 0,
        notes: Optional[Dict[str, Any]] = None,
    ) -> "TransitionMetrics":
        N = len(emotion_sequence)
        uniq = len(set(emotion_sequence)) if N else 0
        stability = 0.0
        if N >= 2:
            stability = 1.0 - ((uniq - 1) / (N - 1))
        vol = (len(transitions) / (N - 1)) if N >= 2 else 0.0
        peak_avg = 0.0
        if transitions:
            peak_avg = sum(float(t.get("intensity", 0.5)) for t in transitions) / len(transitions)

        # 카테고리 히스토그램(희/노/애/락만)
        cats = {"희": 0, "노": 0, "애": 0, "락": 0}
        for t in transitions or []:
            a = t.get("from_emotion"); b = t.get("to_emotion")
            if a in cats: cats[a] += 1
            if b in cats: cats[b] += 1

        # intensifier_hits: transition dict 내부의 markers_detected 기반(없으면 0)
        intensifier_hits = 0
        for t in transitions or []:
            md = t.get("markers_detected", []) or []
            intensifier_hits += sum(1 for m in md if isinstance(m, str))

        return cls(
            processing_time=float(processing_time),
            memory_usage=int(memory_usage),
            transition_count=len(transitions),
            pattern_count=int(pattern_count),
            confidence_avg=_round3(confidence_avg),
            flow_pattern=flow_pattern or "unknown",
            stability_score=_round3(_clip01(stability)),
            volatility=_round3(_clip01(vol)),
            peak_intensity_avg=_round3(_clip01(peak_avg)),
            intensifier_hits=intensifier_hits,
            marker_hits=int(marker_hits or 0),
            category_counts=cats,
            notes=notes or {}
        )

# =============================================================================
# DataClass: EmotionNode
# =============================================================================
@dataclass
class EmotionNode:
    """
    EMOTIONS.json에서 단일(대표/세부) 감정을 정규화해 들고 다니는 컨테이너.
    - core_keywords/markers를 lower+경계 regex로 미리 컴파일 → 런타임 속도/정확도↑
    - 스키마 강제 변경 없음(존재할 때만 사용)
    """
    name: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    emotion_profile: Dict[str, Any] = field(default_factory=dict)
    context_patterns: Dict[str, Any] = field(default_factory=dict)
    emotion_transitions: Dict[str, Any] = field(default_factory=dict)

    synonyms: List[str] = field(default_factory=list)
    aliases: List[str] = field(default_factory=list)
    core_keywords: List[str] = field(default_factory=list)          # lower
    intensity_levels: Dict[str, Any] = field(default_factory=dict)  # normalized

    # marker bank: {'gradual':[str..], 'sudden':[...], 'intensifying':[...], 'attenuating':[...], 'repetitive':[...]}
    trigger_markers: Dict[str, List[str]] = field(default_factory=dict)

    # 컴파일 패턴: {'key_phrases':[regex...], 'markers':{'gradual':[re...], ...}}
    compiled_patterns: Dict[str, Any] = field(default_factory=dict)

    # id/category/polarity/relations
    emotion_id: str = ""
    primary_category: str = ""
    sub_category: str = ""
    polarity: Optional[str] = None
    synergy_with: List[str] = field(default_factory=list)
    conflict_with: List[str] = field(default_factory=list)

    cache_keys: Dict[str, Any] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    # -------- constructor from JSON --------
    @classmethod
    def from_json(cls, name: str, data: Dict[str, Any]) -> "EmotionNode":
        md = (data.get("metadata") or {})
        prof = (data.get("emotion_profile") or {})
        ctx = (data.get("context_patterns") or {})
        trs = (data.get("emotion_transitions") or {})
        ml = (data.get("ml_training_metadata") or {})

        # core keywords(lowered)
        core = _lower_list(prof.get("core_keywords", []) or [])
        # intensity levels normalized
        levels = _normalize_intensity_levels(prof.get("intensity_levels", {}) or {})

        # marker bank (존재 시만)
        bank = {
            "gradual": _lower_list(ml.get("gradual_markers", []) or []),
            "sudden": _lower_list(ml.get("sudden_markers", []) or []),
            "repetitive": _lower_list(ml.get("repetitive_markers", []) or []),
            "intensifying": _lower_list(ml.get("intensifying_markers", []) or []),
            "attenuating": _lower_list(ml.get("attenuating_markers", []) or []),
        }
        # transitions.patterns.transition_analysis.trigger_words → intensifying에 보강
        for pat in (trs.get("patterns", []) or []):
            an = (pat.get("transition_analysis") or {})
            for w in _lower_list(an.get("trigger_words", []) or []):
                bank["intensifying"].append(w)

        # key_phrases 컴파일(존재 시)
        key_phrases = []
        ling = data.get("linguistic_patterns", {}) or {}
        raw_kp = ling.get("key_phrases", []) or []
        for item in raw_kp:
            if isinstance(item, dict):
                p = item.get("pattern", "")
                if isinstance(p, str) and p.strip():
                    key_phrases.append(p.strip())
            elif isinstance(item, str) and item.strip():
                key_phrases.append(item.strip())

        compiled = {
            "key_phrases": _compile_boundary_regex_list(key_phrases),
            "markers": {k: _compile_boundary_regex_list(v) for k, v in bank.items()},
        }

        # relation/identity
        pol = ml.get("polarity")
        syn = [s for s in (ml.get("synergy_with", []) or []) if isinstance(s, str)]
        con = [s for s in (ml.get("conflict_with", []) or []) if isinstance(s, str)]

        return cls(
            name=name,
            metadata=md,
            emotion_profile=prof,
            context_patterns=ctx,
            emotion_transitions=trs,
            synonyms=_lower_list(data.get("synonyms", []) or []),
            aliases=_lower_list(data.get("aliases", []) or []),
            core_keywords=core,
            intensity_levels=levels,
            trigger_markers=bank,
            compiled_patterns=compiled,
            emotion_id=str(md.get("emotion_id", "")) if isinstance(md, dict) else "",
            primary_category=str(md.get("primary_category", "")) if isinstance(md, dict) else "",
            sub_category=str(md.get("sub_category", "")) if isinstance(md, dict) else "",
            polarity=str(pol).lower() if isinstance(pol, str) else None,
            synergy_with=syn,
            conflict_with=con,
            cache_keys={},
            diagnostics={},
        )

    # -------- small helpers --------
    def match_key_phrases(self, text: str) -> List[str]:
        """
        경계 인식 key_phrase 매칭 리스트 반환(대소문자 무시).
        """
        if not text:
            return []
        regs = (self.compiled_patterns.get("key_phrases") or [])
        hits: List[str] = []
        for r in regs:
            if r.search(text):
                # 원래 패턴 문자열을 evidence로 남김
                hits.append(r.pattern)
        return hits

    def match_markers(self, text: str, groups: Optional[List[str]] = None) -> List[str]:
        """
        marker bank(regex) 매칭 결과(패턴 문자열) 반환.
        groups가 없으면 모든 그룹 검사.
        """
        if not text:
            return []
        markers = self.compiled_patterns.get("markers") or {}
        res: List[str] = []
        keys = groups if groups else list(markers.keys())
        for g in keys:
            for r in markers.get(g, []):
                try:
                    if r.search(text):
                        res.append(r.pattern)
                except Exception:
                    continue
        return res

# =============================================================================
# DataClass: TransitionResult
# =============================================================================
@dataclass
class TransitionResult:
    """
    개별 전이 결과(하위호환 유지 + 의미론 보강).
    direction/ category_delta/ evidence_scores 등은 보조 유틸을 통해 산출.
    """
    from_emotion: str
    to_emotion: str
    trigger_text: str
    confidence: float
    intensity: float
    position: int
    prev_context: str

    pattern_type: str = ""
    direction: str = ""  # 'pos->neg' / 'neg->pos' / 'pos->pos' / 'neg->neg' / 'neutral'
    stage: str = ""      # 'Trigger'/'Development'/'Peak'/'Aftermath'
    reason_tags: List[str] = field(default_factory=list)
    markers_detected: List[str] = field(default_factory=list)
    trigger_keywords: List[str] = field(default_factory=list)
    evidence_scores: Dict[str, float] = field(default_factory=dict)  # {'marker':x,'context':y,'pattern':z,'delta':u}
    category_delta: Tuple[str, str] = ("", "")                      # ('희','노') 등
    normalized: bool = True

    # -------- enrichment helpers --------
    @classmethod
    def enrich_direction(
        cls,
        tr: "TransitionResult",
        cat_of: Callable[[str], Optional[str]],
        polarity_map: Dict[str, str]
    ) -> "TransitionResult":
        """
        카테고리 극성 기반으로 direction/category_delta 산출.
        """
        ca = cat_of(tr.from_emotion) or tr.from_emotion
        cb = cat_of(tr.to_emotion) or tr.to_emotion
        pa = polarity_map.get(ca, "neutral")
        pb = polarity_map.get(cb, "neutral")

        if pa == "positive" and pb == "negative":
            d = "pos->neg"
        elif pa == "negative" and pb == "positive":
            d = "neg->pos"
        elif pa == pb == "positive":
            d = "pos->pos"
        elif pa == pb == "negative":
            d = "neg->neg"
        else:
            d = "neutral"

        tr.direction = d
        tr.category_delta = (ca if isinstance(ca, str) else "", cb if isinstance(cb, str) else "")
        return tr

    @classmethod
    def enrich_evidence(
        cls,
        tr: "TransitionResult",
        marker_score: float = 0.0,
        context_score: float = 0.0,
        pattern_score: float = 0.0,
        delta_score: float = 0.0
    ) -> "TransitionResult":
        """
        evidence_scores를 0~1 clip 후 3자리 반올림으로 세팅.
        """
        tr.evidence_scores = {
            "marker": _round3(_clip01(marker_score)),
            "context": _round3(_clip01(context_score)),
            "pattern": _round3(_clip01(pattern_score)),
            "delta": _round3(_clip01(delta_score)),
        }
        return tr

    @classmethod
    def normalize_scores(cls, tr: "TransitionResult") -> "TransitionResult":
        """
        confidence/intensity 등 주요 스코어 0~1 클립 + 반올림(3자리).
        """
        tr.confidence = _round3(_clip01(tr.confidence))
        tr.intensity = _round3(_clip01(tr.intensity))
        return tr


# =============================================================================
# Utils2
# =============================================================================
@dataclass
class _CacheEntry:
    value: Any
    ts_put: float
    ttl: Optional[float] = None
    size_bytes: int = 0

class LRUCache:
    def __init__(
        self,
        maxsize: int,
        use_external_storage: bool = False,
        external_storage_handler: Any = None,
        ttl_seconds: Optional[float] = None,
        max_bytes: Optional[int] = None,
        key_fn: Optional[Callable[[Any], Hashable]] = None,
        size_estimator: Optional[Callable[[Any], int]] = None,
    ):
        self.cache: "OrderedDict[Hashable, _CacheEntry]" = OrderedDict()
        self.maxsize = int(maxsize)  # 3번 개선작업: 캐시 크기 최적화
        self.max_bytes = int(max_bytes) if max_bytes else None
        self._bytes_used = 0
        self.ttl = ttl_seconds
        self.key_fn = key_fn
        self.size_estimator = size_estimator
        self.lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        self.expired_evicts = 0
        self.use_external_storage = use_external_storage
        self.external_storage_handler = external_storage_handler

    def get(self, key: Any) -> Optional[Any]:
        k = self._norm_key(key)
        with self.lock:
            entry = self.cache.get(k)
            if entry is not None:
                if self._is_expired(entry):
                    self._evict_key(k, reason="expired")
                    self.misses += 1
                else:
                    self.cache.move_to_end(k, last=True)
                    self.hits += 1
                    return entry.value
            else:
                self.misses += 1

        if self.use_external_storage and self.external_storage_handler:
            ext = self._fetch_from_external_storage(k)
            if ext is not None:
                self.put(k, ext)
                return ext
        return None

    def put(self, key: Any, value: Any) -> None:
        k = self._norm_key(key)
        entry = _CacheEntry(
            value=value,
            ts_put=time.time(),
            ttl=self.ttl,
            size_bytes=self._estimate_size(value),
        )
        with self.lock:
            if k in self.cache:
                self._bytes_used -= self.cache[k].size_bytes
                self.cache[k] = entry
                self._bytes_used += entry.size_bytes
                self.cache.move_to_end(k, last=True)
            else:
                if len(self.cache) >= self.maxsize:
                    self._evict_oldest()
                self.cache[k] = entry
                self._bytes_used += entry.size_bytes
                self.cache.move_to_end(k, last=True)
            self._evict_overflow()

        if self.use_external_storage and self.external_storage_handler:
            self._store_to_external_storage(k, value)

    def stats(self) -> Dict[str, Any]:
        with self.lock:
            hit_ratio = (self.hits / max(1, (self.hits + self.misses)))
            return {
                "hits": self.hits,
                "misses": self.misses,
                "hit_ratio": round(hit_ratio, 3),
                "expired_evicts": self.expired_evicts,
                "current_size": len(self.cache),
                "max_size": self.maxsize,
                "bytes_used": self._bytes_used,
                "max_bytes": self.max_bytes,
            }

    def clear(self) -> None:
        with self.lock:
            self.cache.clear()
            self._bytes_used = 0
            self.hits = 0
            self.misses = 0
            self.expired_evicts = 0
        logger.debug("[LRUCache] 캐시 초기화 완료")

    def _norm_key(self, key: Any) -> Hashable:
        return self.key_fn(key) if self.key_fn else key

    def _is_expired(self, entry: _CacheEntry) -> bool:
        if entry.ttl is None:
            return False
        return (time.time() - entry.ts_put) >= entry.ttl

    def _estimate_size(self, v: Any) -> int:
        if self.size_estimator:
            try:
                return int(self.size_estimator(v))
            except Exception:
                return 0
        return 0

    def _evict_key(self, k: Hashable, reason: str = "") -> None:
        ent = self.cache.pop(k, None)
        if ent:
            self._bytes_used -= ent.size_bytes
            if reason == "expired":
                self.expired_evicts += 1
            logger.debug(f"[LRUCache] evict key={k} reason={reason}")

    def _evict_oldest(self) -> None:
        k, ent = self.cache.popitem(last=False)
        self._bytes_used -= ent.size_bytes
        logger.debug(f"[LRUCache] 용량 초과로 oldest_key 제거: {k}")

    def _evict_overflow(self) -> None:
        if self.max_bytes is None:
            return
        while self._bytes_used > self.max_bytes and self.cache:
            k, ent = self.cache.popitem(last=False)
            self._bytes_used -= ent.size_bytes
            logger.debug(f"[LRUCache] 바이트 한도 초과로 제거: {k}")

    def _fetch_from_external_storage(self, key: Hashable) -> Optional[Any]:
        try:
            if hasattr(self.external_storage_handler, "mget"):
                res = self.external_storage_handler.mget([key])
                return res.get(key) if isinstance(res, dict) else None
            if hasattr(self.external_storage_handler, "get"):
                return self.external_storage_handler.get(key)
            return None
        except Exception as e:
            logger.error(f"[LRUCache] 외부 저장소 get 실패: {e}")
            return None

    def _store_to_external_storage(self, key: Hashable, value: Any) -> None:
        try:
            if hasattr(self.external_storage_handler, "mset"):
                self.external_storage_handler.mset({key: value})
            elif hasattr(self.external_storage_handler, "put"):
                self.external_storage_handler.put(key, value)
            elif hasattr(self.external_storage_handler, "set"):
                self.external_storage_handler.set(key, value)
            else:
                logger.debug("[LRUCache] 외부 저장소 set 인터페이스 미검출 → 무시")
        except Exception as e:
            logger.error(f"[LRUCache] 외부 저장소 put 실패: {e}")

class MemoryMonitor:
    def __init__(
        self,
        memory_limit: float,
        use_vms: bool = False,
        soft_ratio: float = 0.9,
    ):
        self.use_vms = use_vms
        self.process = psutil.Process() if psutil else None
        self.total_sys = psutil.virtual_memory().total if psutil else 0

        if memory_limit < 1.0 and self.total_sys:
            self.hard_limit = int(self.total_sys * memory_limit)
        else:
            self.hard_limit = int(memory_limit)
        
        # 메모리 제한이 0이면 기본값으로 설정
        if self.hard_limit <= 0:
            self.hard_limit = 51200 * 1024 * 1024  # 50GB 기본값으로 대폭 증가 (64GB RAM 대비)

        self.soft_limit = int(self.hard_limit * float(soft_ratio))
        self.max_memory_usage = 0
        self.total_memory_usage_sample = 0
        self.num_samples = 0
        logger.debug(
            f"[MemoryMonitor] init: mode={'VMS' if self.use_vms else 'RSS'}, "
            f"hard_limit={self.hard_limit}, soft_limit={self.soft_limit}, total_sys={self.total_sys}"
        )

    def check_memory(self) -> bool:
        try:
            current = self.get_usage()
            self._update_memory_stats(current)
            if current >= self.hard_limit:
                logger.warning(f"[MemoryMonitor] HARD limit 초과: {current} >= {self.hard_limit}")
                return False
            if current >= self.soft_limit:
                logger.info(f"[MemoryMonitor] soft limit 초과: {current} >= {self.soft_limit}")
            return True
        except Exception as e:
            logger.error(f"[MemoryMonitor] check_memory 실패: {e}")
            return True

    def get_usage(self) -> int:
        try:
            if not self.process:
                return 0
            mem = self.process.memory_info()
            return int(mem.vms if self.use_vms else mem.rss)
        except Exception as e:
            logger.error(f"[MemoryMonitor] get_usage 실패: {e}")
            return 0

    def get_memory_statistics(self) -> dict:
        avg = (self.total_memory_usage_sample / self.num_samples) if self.num_samples > 0 else 0.0
        return {
            "max_memory_usage": int(self.max_memory_usage),
            "average_memory_usage": int(avg),
            "samples_count": self.num_samples,
            "hard_limit": self.hard_limit,
            "soft_limit": self.soft_limit,
            "mode": "VMS" if self.use_vms else "RSS"
        }

    def reset_statistics(self) -> None:
        self.max_memory_usage = 0
        self.total_memory_usage_sample = 0
        self.num_samples = 0
        logger.debug("[MemoryMonitor] 통계 리셋 완료")

    def guard(self, on_exceed: str = "warn") -> Callable:
        def _dec(fn: Callable) -> Callable:
            def _wrap(*args, **kwargs):
                ok = self.check_memory()
                if ok:
                    return fn(*args, **kwargs)
                if on_exceed == "skip":
                    logger.warning(f"[MemoryMonitor] guard: 메모리 초과로 {fn.__name__} 실행 스킵")
                    return None
                if on_exceed == "raise":
                    raise MemoryError("Memory hard limit exceeded")
                logger.warning(f"[MemoryMonitor] guard: 메모리 초과 감지(경고만)")
                return fn(*args, **kwargs)
            return _wrap
        return _dec

    def _update_memory_stats(self, current_usage: int) -> None:
        self.num_samples += 1
        self.total_memory_usage_sample += current_usage
        if current_usage > self.max_memory_usage:
            self.max_memory_usage = current_usage


# =============================================================================
# TransitionAnalyzer (Refined for 95% target)
# =============================================================================
class TransitionAnalyzer:
    """
    전이·흐름 분석기
    - 문장 분할(KSS) → 문장별 감정 상태 감지 → 이전/다음 감정이 달라질 때만 전이 기록
    - (개선) 같은 문장 내부 pivot 전환도 감지
    - 트리거/패턴 타입(긍→부정 등) 산출 → 시퀀스 기반 흐름(안정/진동/급변) 요약
    - 진행 단계(Trigger/Development/Peak/Aftermath) 추정
    - 모든 규칙/가중치는 가능한 한 EMOTIONS.json에서 유도(규칙1), 4×N 확장 안전(규칙2), 스키마 불변(규칙3)
    """

    # ------------------------------ init ------------------------------
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        emotions_data: Optional[Dict[str, Any]] = None,
        device: str = "cuda" if (torch and torch.cuda.is_available()) else "cpu"
    ):
        self.device = device
        cfg = config or {}
        self.config = cfg.get("TRANSITION_ANALYZER_CONFIG", {})
        
        # EMOTIONS.json 로드 (하드코딩 대신)
        if emotions_data is None:
            self.emotions_data = self._load_emotions_data()
        else:
            self.emotions_data = emotions_data
            
        # EMOTIONS.json 기반 전이 패턴 캐시 초기화
        self._transition_patterns_cache = {}
        self._emotion_states_cache = {}
        self._cat_rx_shards = []  # 기존 코드 호환성
        self._regex_cache = {}  # 기존 코드 호환성
        self.transition_map = {}  # 기존 코드 호환성
        
        # transition_cache를 dict로 초기화 (기존 코드 호환성)
        self.transition_cache = {}
        
        self._load_transition_patterns_recursive()

        # thresholds / weights (오버라이드 가능)
        # 개선사항: 기본 신뢰도 임계 상향 (0.10 -> 0.30) + MAD 기반 자동 임계 병행
        # config.py의 THRESHOLDS에서 전이 임계값 가져오기
        try:
            from src.config import THRESHOLDS
            default_threshold = THRESHOLDS.get('transition_min_confidence', 0.6)
        except ImportError:
            default_threshold = 0.6
        
        self.thr_confidence_min: float = float(self.config.get("thresholds", {}).get("min_confidence", default_threshold))
        self.auto_threshold_enabled: bool = bool(self.config.get("thresholds", {}).get("auto_threshold_enabled", True))
        self.stage_peak_intensity: float = float(self.config.get("stages", {}).get("peak_intensity", 0.7))
        self.flow_thr_volatile: float = float(self.config.get("flow", {}).get("volatile_ratio", 0.7))
        
        # Memory monitor 초기화 (Python 3.11.13 호환성을 위해 비활성화)
        self.memory_monitor = None
        
        # 카테고리 정규식 샤드 설정
        self._cat_rx_shard_size: int = int(self.config.get("regex_shard_size", 256))
        
        # Pivot connectors 초기화
        self._pivot_connectors_contrast = ["하지만", "그러나", "그런데"]
        self._pivot_connectors_causal = ["그래서", "때문에", "덕분에"]
        
        # Compiled marker bank 초기화
        self._compiled_marker_bank: Dict[str, List[re.Pattern]] = {
            "gradual": [],
            "sudden": [],
            "repetitive": [],
            "intensifying": [],
            "attenuating": []
        }
        
        # Compiled key phrases 초기화
        self._compiled_key_phrases: List[re.Pattern] = []
        
        # 추가 속성들 초기화
        self.stage_dev_window = int(self.config.get("stage_dev_window", 3))
        self.flow_thr_complex_unique = float(self.config.get("flow_thr_complex_unique", 0.5))
        
        # MAD 기반 자동 임계값 계산을 위한 히스토리
        self._confidence_history: List[float] = []
        self._auto_threshold_cache: Optional[float] = None

    def _calculate_mad_threshold(self, confidences: List[float], min_samples: int = 10) -> float:
        """
        MAD (Median Absolute Deviation) 기반 자동 임계값 계산
        - 신뢰도 분포의 중앙값과 MAD를 사용하여 적응적 임계값 결정
        - 과탐을 줄이면서 유효한 전이를 놓치지 않도록 조정
        """
        if len(confidences) < min_samples:
            return self.thr_confidence_min  # 기본값 사용
        
        import numpy as np
        
        # 중앙값과 MAD 계산
        median_conf = np.median(confidences)
        mad = np.median(np.abs(np.array(confidences) - median_conf))
        
        # MAD 기반 임계값: 중앙값 + 1.5 * MAD
        # 이는 일반적인 이상치 탐지에서 사용되는 방법
        mad_threshold = median_conf + 1.5 * mad
        
        # 최소/최대 제한으로 안정성 확보
        min_threshold = max(0.25, self.thr_confidence_min)  # 최소 0.25
        max_threshold = 0.70  # 최대 0.70
        
        auto_threshold = max(min_threshold, min(mad_threshold, max_threshold))
        
        return float(auto_threshold)

    def _update_confidence_history(self, confidence: float) -> None:
        """신뢰도 히스토리 업데이트 (최대 100개 유지)"""
        self._confidence_history.append(confidence)
        if len(self._confidence_history) > 100:
            self._confidence_history = self._confidence_history[-100:]

    def _get_effective_threshold(self) -> float:
        """효과적인 임계값 반환 (자동 임계값이 활성화된 경우 MAD 기반 계산)"""
        if not self.auto_threshold_enabled or len(self._confidence_history) < 10:
            return self.thr_confidence_min
        
        # 캐시된 자동 임계값이 있으면 사용
        if self._auto_threshold_cache is not None:
            return self._auto_threshold_cache
        
        # 새로운 자동 임계값 계산
        self._auto_threshold_cache = self._calculate_mad_threshold(self._confidence_history)
        return self._auto_threshold_cache

    def _load_emotions_data(self) -> Dict[str, Any]:
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

    def _load_transition_patterns_recursive(self, emotion_data: Dict[str, Any] = None, path: str = "") -> None:
        """EMOTIONS.json에서 전이 패턴을 재귀적으로 로드"""
        if emotion_data is None:
            emotion_data = self.emotions_data
            
        for emotion_key, emotion_info in emotion_data.items():
            current_path = f"{path}.{emotion_key}" if path else emotion_key
            
            # 하위 감정 재귀 처리
            if isinstance(emotion_info, dict):
                if "sub_emotions" in emotion_info:
                    self._load_transition_patterns_recursive(emotion_info["sub_emotions"], current_path)
                
                # 전이 패턴 추출
                transition_patterns = self._extract_transition_patterns(emotion_info, current_path)
                if transition_patterns:
                    self._transition_patterns_cache[current_path] = transition_patterns
                    
                # 감정 상태 추출
                emotion_state = self._extract_emotion_state(emotion_info, current_path)
                if emotion_state:
                    self._emotion_states_cache[current_path] = emotion_state

    def _extract_transition_patterns(self, emotion_info: Dict[str, Any], emotion_path: str) -> Dict[str, Any]:
        """감정 정보에서 전이 패턴 추출"""
        patterns = {
            "triggers": [],
            "transitions": [],
            "intensity_changes": [],
            "context_markers": []
        }
        
        # patterns에서 전이 정보 추출
        if "patterns" in emotion_info:
            for pattern in emotion_info["patterns"]:
                if isinstance(pattern, dict):
                    if "from_emotion" in pattern and "to_emotion" in pattern:
                        patterns["transitions"].append({
                            "from": pattern["from_emotion"],
                            "to": pattern["to_emotion"],
                            "triggers": pattern.get("triggers", [])
                        })
        
        # intensity_examples에서 강도 변화 패턴 추출
        if "emotion_profile" in emotion_info:
            profile = emotion_info["emotion_profile"]
            if "intensity_levels" in profile:
                intensity_levels = profile["intensity_levels"]
                if "intensity_examples" in intensity_levels:
                    examples = intensity_levels["intensity_examples"]
                    for level, example_list in examples.items():
                        if isinstance(example_list, list):
                            patterns["intensity_changes"].extend(example_list)
        
        # keywords에서 트리거 추출
        if "keywords" in emotion_info:
            patterns["triggers"].extend(emotion_info["keywords"])
            
        # triggers에서 컨텍스트 마커 추출
        if "triggers" in emotion_info:
            patterns["context_markers"].extend(emotion_info["triggers"])
            
        return patterns

    def _extract_emotion_state(self, emotion_info: Dict[str, Any], emotion_path: str) -> Dict[str, Any]:
        """감정 정보에서 감정 상태 추출"""
        state = {
            "emotion": emotion_path,
            "intensity_levels": [],
            "related_emotions": [],
            "characteristics": []
        }
        
        if "emotion_profile" in emotion_info:
            profile = emotion_info["emotion_profile"]
            
            # 강도 레벨 추출
            if "intensity_levels" in profile:
                intensity_levels = profile["intensity_levels"]
                state["intensity_levels"] = list(intensity_levels.keys())
            
            # 관련 감정 추출
            if "related_emotions" in profile:
                related = profile["related_emotions"]
                for category, emotions in related.items():
                    if isinstance(emotions, list):
                        state["related_emotions"].extend(emotions)
        
        return state

    def analyze_transitions(self, text: str) -> Dict[str, Any]:
        """EMOTIONS.json 기반 전이 분석"""
        try:
            # 문장 분리
            if kss:
                sentences = kss.split_sentences(text)
            else:
                sentences = [s.strip() for s in text.split('.') if s.strip()]
            
            if not sentences:
                sentences = [text.strip()]
            
            # 각 문장에서 감정 상태 감지
            sentence_states = []
            for i, sentence in enumerate(sentences):
                emotion_state = self._detect_emotion_state_new(sentence)
                sentence_states.append({
                    "sentence_index": i,
                    "sentence": sentence,
                    "emotion_state": emotion_state
                })
            
            # 전이 분석
            transitions = self._analyze_emotion_transitions(sentence_states)
            
            # 감정 시퀀스 추출
            emotion_sequence = []
            stage_tags = []
            for state in sentence_states:
                emotions = [e["emotion"] for e in state["emotion_state"]["detected_emotions"]]
                emotion_sequence.append(emotions[0] if emotions else "unknown")
                stage_tags.append("unknown")
            
            # 흐름 분석
            flow_analysis = self._analyze_emotion_flow(emotion_sequence, transitions, stage_tags)
            
            # 진행 단계 분석
            progression_stages = self._analyze_progression_stages(sentence_states)
            
            return {
                "sentence_states": sentence_states,
                "transitions": transitions,
                "flow_analysis": flow_analysis,
                "progression_stages": progression_stages,
                "summary": {
                    "total_sentences": len(sentences),
                    "transition_count": len(transitions),
                    "flow_type": flow_analysis.get("flow_type", "unknown"),
                    "progression_stage": progression_stages.get("current_stage", "unknown")
                },
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "success": True
            }
            
        except Exception as e:
            return {
                "sentence_states": [],
                "transitions": [],
                "flow_analysis": {},
                "progression_stages": {},
                "summary": {
                    "total_sentences": 0,
                    "transition_count": 0,
                    "flow_type": "unknown",
                    "progression_stage": "unknown"
                },
                "error": str(e),
                "success": False
            }

    def _detect_emotion_state_new(self, sentence: str) -> Dict[str, Any]:
        """문장에서 감정 상태 감지 (새로운 구현)"""
        state = {
            "detected_emotions": [],
            "intensity": "medium",
            "confidence": 0.0
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
                state["detected_emotions"].append({
                    "emotion": emotion,
                    "pattern": english_word,
                    "confidence": 0.8
                })
                break
        
        # EMOTIONS.json 패턴 매칭
        for emotion_path, transition_patterns in self._transition_patterns_cache.items():
            # 트리거 매칭
            for trigger in transition_patterns.get("triggers", []):
                if trigger.lower() in sentence_lower:
                    state["detected_emotions"].append({
                        "emotion": emotion_path,
                        "pattern": trigger,
                        "confidence": 0.8
                    })
                    break
        
        # 강도 수식어 감지
        intensity_modifiers = ["very", "quite", "extremely", "slightly", "somewhat"]
        for modifier in intensity_modifiers:
            if modifier in sentence_lower:
                if modifier in ["very", "extremely"]:
                    state["intensity"] = "high"
                elif modifier in ["slightly", "somewhat"]:
                    state["intensity"] = "low"
                break
        
        # 신뢰도 계산
        if state["detected_emotions"]:
            state["confidence"] = max(emotion["confidence"] for emotion in state["detected_emotions"])
        
        return state

    def _analyze_emotion_transitions(self, sentence_states: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """감정 전이 분석"""
        transitions = []
        
        for i in range(1, len(sentence_states)):
            prev_state = sentence_states[i-1]["emotion_state"]
            curr_state = sentence_states[i]["emotion_state"]
            
            prev_emotions = [e["emotion"] for e in prev_state["detected_emotions"]]
            curr_emotions = [e["emotion"] for e in curr_state["detected_emotions"]]
            
            if prev_emotions != curr_emotions:
                transitions.append({
                    "from_sentence": i-1,
                    "to_sentence": i,
                    "previous_emotions": prev_emotions,
                    "current_emotions": curr_emotions,
                    "from_emotion": prev_emotions[0] if prev_emotions else "unknown",
                    "to_emotion": curr_emotions[0] if curr_emotions else "unknown",
                    "transition_type": "emotion_change",
                    "intensity_change": self._calculate_intensity_change(prev_state, curr_state)
                })
        
        return transitions

    def _calculate_intensity_change(self, prev_state: Dict[str, Any], curr_state: Dict[str, Any]) -> str:
        """강도 변화 계산"""
        prev_intensity = prev_state.get("intensity", "medium")
        curr_intensity = curr_state.get("intensity", "medium")
        
        intensity_map = {"low": 1, "medium": 2, "high": 3}
        prev_val = intensity_map.get(prev_intensity, 2)
        curr_val = intensity_map.get(curr_intensity, 2)
        
        if curr_val > prev_val:
            return "increase"
        elif curr_val < prev_val:
            return "decrease"
        else:
            return "stable"

    def _analyze_emotion_flow(self, sentence_states: List[Dict[str, Any]]) -> Dict[str, Any]:
        """감정 흐름 분석"""
        if len(sentence_states) < 2:
            return {"flow_type": "single", "stability": "stable"}
        
        transitions = self._analyze_emotion_transitions(sentence_states)
        transition_count = len(transitions)
        total_sentences = len(sentence_states)
        
        # 흐름 타입 결정
        if transition_count == 0:
            flow_type = "stable"
        elif transition_count <= total_sentences * 0.3:
            flow_type = "gradual"
        elif transition_count <= total_sentences * 0.7:
            flow_type = "moderate"
        else:
            flow_type = "volatile"
        
        return {
            "flow_type": flow_type,
            "transition_count": transition_count,
            "stability": "stable" if flow_type in ["stable", "gradual"] else "unstable"
        }

    def _analyze_progression_stages(self, sentence_states: List[Dict[str, Any]]) -> Dict[str, Any]:
        """진행 단계 분석"""
        if not sentence_states:
            return {"current_stage": "unknown", "stage_progression": []}
        
        total_sentences = len(sentence_states)
        transitions = self._analyze_emotion_transitions(sentence_states)
        
        # 진행 단계 결정
        if total_sentences <= 2:
            current_stage = "trigger"
        elif total_sentences <= 4:
            current_stage = "development"
        elif total_sentences <= 6:
            current_stage = "peak"
        else:
            current_stage = "aftermath"
        
        # 단계 진행 추적
        stage_progression = []
        for i, state in enumerate(sentence_states):
            if i < total_sentences * 0.25:
                stage_progression.append("trigger")
            elif i < total_sentences * 0.5:
                stage_progression.append("development")
            elif i < total_sentences * 0.75:
                stage_progression.append("peak")
            else:
                stage_progression.append("aftermath")
        
        return {
            "current_stage": current_stage,
            "stage_progression": stage_progression,
            "total_sentences": total_sentences
        }
        # (D) development window 기본 2 유지
        self.stage_dev_window: int = int(self.config.get("stages", {}).get("development_window", 2))
        self.flow_thr_volatile: float = float(self.config.get("flow", {}).get("volatile_ratio", 0.7))
        self.flow_thr_complex_unique: int = int(self.config.get("flow", {}).get("complex_unique_min", 4))

        # 메트릭 컨테이너
        self.metrics = TransitionMetrics(0.0, 0, 0, 0, 0.0)

        # 메모리 모니터
        mem_cfg = self.config.get("memory", {})
        hard = mem_cfg.get("hard_limit", 0.85)  # 85% of system by default
        self.memory_monitor = MemoryMonitor(hard, use_vms=bool(mem_cfg.get("use_vms", False)))

        # 스레드 풀
        self._setup_thread_pool()

        # 캐시: 전이/문장/패턴
        max_history_size = int(mem_cfg.get("max_history_size", 2048))
        ttl = float(mem_cfg.get("cache_ttl_seconds", 900))
        # transition_cache는 이미 dict로 초기화됨
        self._regex_cache: Dict[str, re.Pattern] = {}
        self.pattern_cache: Dict[str, Any] = {}
        # 폴백어 사용 스위치 기본값
        self.config.setdefault("allow_bias_fallback_terms", True)
        # 카테고리 정규식 샤드 설정
        self._cat_rx_shards: Dict[str, Tuple[re.Pattern, ...] | Tuple] = {}
        self._cat_rx_shard_size: int = int(self.config.get("regex_shard_size", 256))

        # 감정 데이터
        self.emotions_data = emotions_data or self._load_emotions_data()

        # 데이터 주도 전이 맵/마커/키워드/키프레이즈 사전 구축
        self.transition_map: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self.trigger_patterns: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._compiled_marker_bank: Dict[str, List[re.Pattern]] = {}  # {'gradual':[...], 'sudden':[...], ...}
        self._compiled_key_phrases: List[re.Pattern] = []             # (B) key_phrases 경계 regex
        self._category_keywords: Dict[str, set] = {"희": set(), "노": set(), "애": set(), "락": set()}
        self._pivot_connectors_causal = ["그래서", "때문에", "그러자", "하여", "결국"]
        self._pivot_connectors_contrast = ["하지만", "그러나", "그런데"]

        self._build_transition_map()
        self._build_marker_bank_and_keywords()  # 규칙1 기반 마커/키워드/키프레이즈 집계
        # 카테고리 키워드 샤드 정규식 컴파일
        self._compile_category_regex_shards()

    # ------------------------------ setup ------------------------------
    def _setup_thread_pool(self):
        max_workers = int(self.config.get("parallel", {}).get("max_workers", 32))  # 라이젠 AI 9/HX 370에 맞춰 워커 수 대폭 증가
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="TransitionAnalyzer")
        logger.info(f"[TransitionAnalyzer] ThreadPoolExecutor initialized: max_workers={max_workers}")


    # ------------------------------ public main ------------------------------
    def analyze_emotion_transitions(
            self,
            text: str,
            emotions_data: Optional[Dict[str, Any]] = None,
            pattern_results: Optional[Dict[str, Any]] = None,
            intensity: Optional[Dict[str, Any]] = None,
            context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        start = time.time()
        if emotions_data:
            self.emotions_data = emotions_data
            self._build_transition_map()
            self._build_marker_bank_and_keywords()
            # 카테고리 샤드 정규식 재컴파일
            self._compile_category_regex_shards()
            # 커넥터/키프레이즈 JSON/설정 주도 로딩
            self._load_connectors_from_json_and_config()

        result = {"transitions": [], "triggers": [], "patterns": {}, "flow_analysis": {}, "metrics": {}}
        if not text or not text.strip():
            self._update_metrics(result, start);
            return result

        sentences = self._split_into_sentences(text)
        transitions, triggers, patterns = [], [], {}
        prev_emotion, prev_sentence = None, None
        emotion_sequence: List[str] = []
        stage_tags: List[str] = []

        for idx, sentence in enumerate(sentences):
            if self.memory_monitor and not self.memory_monitor.check_memory():
                logger.warning("[TransitionAnalyzer] memory hard limit exceeded. Early stop.")
                break

            cur_emotion = self._detect_emotion_state(sentence)
            # (A) 문장 내부 피벗 먼저 스캔 (이전 감정 정보도 넘김)
            pivots = self._scan_intra_sentence_pivot(sentence, idx, prev_emotion=prev_emotion,
                                                     prev_sentence=prev_sentence)
            if pivots:
                for p in pivots:
                    # 임계치 통과 시만 전이 채택 (MAD 기반 자동 임계값 사용)
                    effective_threshold = self._get_effective_threshold()
                    if p["confidence"] >= effective_threshold and p["from_emotion"] and p["to_emotion"]:
                        # 신뢰도 히스토리 업데이트
                        self._update_confidence_history(p["confidence"])
                        transitions.append(p)
                        # 간단 스테이지 태깅
                        stage_tags.append(
                            self._stage_from_transition(p["intensity"], p["confidence"], transitions, len(transitions)))
                # 피벗이 이미 전이를 만들었다면, 문장간 전이와 중복 방지 위해 cur_emotion 업데이트
                if cur_emotion is None and pivots[-1].get("to_emotion"):
                    cur_emotion = pivots[-1]["to_emotion"]

            if not cur_emotion:
                prev_sentence = sentence
                continue

            emotion_sequence.append(cur_emotion)
            cur_intensity = self._calculate_intensity(sentence, cur_emotion)

            # (B) 문장 간 전이
            if prev_emotion and cur_emotion != prev_emotion:
                tinfo = self._get_transition_info(prev_emotion, cur_emotion)
                confidence = self._calculate_transition_confidence(prev_emotion, cur_emotion, sentence, prev_sentence)

                trig_info = {
                    "from_emotion": prev_emotion, "to_emotion": cur_emotion,
                    "trigger_text": sentence.strip(), "context": (prev_sentence or "").strip(),
                    "confidence": confidence, "position": idx, "intensity": cur_intensity,
                }
                triggers.append(trig_info)

                pat = {
                    "type": self._determine_pattern_type(prev_emotion, cur_emotion, cur_intensity),
                    "from_emotion": prev_emotion, "to_emotion": cur_emotion,
                    "intensity": cur_intensity, "confidence": confidence,
                    "context": sentence.strip(),
                    "trigger_words": self._extract_trigger_words(sentence, tinfo, prev_emotion, cur_emotion),
                }
                patterns[f"pattern_{idx}"] = pat

                effective_threshold = self._get_effective_threshold()
                if confidence >= effective_threshold:
                    # 신뢰도 히스토리 업데이트
                    self._update_confidence_history(confidence)
                    transitions.append({
                        "from_emotion": prev_emotion, "to_emotion": cur_emotion,
                        "trigger_text": sentence.strip(), "confidence": round(confidence, 3),
                        "intensity": round(cur_intensity, 3), "position": idx,
                        "prev_context": (prev_sentence or "").strip(),
                    })
                    stage_tags.append(
                        self._stage_from_transition(cur_intensity, confidence, transitions, len(transitions)))

            prev_emotion = cur_emotion
            prev_sentence = sentence

        # 중복 제거(같은 pos/from/to/trigger)
        seen = set();
        deduped = []
        for t in transitions:
            key = (t.get("position"), t.get("from_emotion"), t.get("to_emotion"), t.get("trigger_text"))
            if key in seen:
                continue
            seen.add(key);
            deduped.append(t)

        result["transitions"] = deduped
        result["triggers"] = triggers
        result["patterns"] = patterns
        result["flow_analysis"] = self._analyze_emotion_flow(emotion_sequence, deduped, stage_tags)
        self._update_metrics(result, start)
        return result

    # ------------------------------ A) Intra-sentence pivot ------------------------------
    def _scan_intra_sentence_pivot(
            self, sentence: str, idx: int, prev_emotion: Optional[str] = None, prev_sentence: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        문장 내부 피벗: 커넥터(대조/인과) 인덱스를 기준으로 좌/우 절을 만들고, 감정 변화면 synthetic 전이 생성.
        - 왼쪽 절이 비거나 무의미할 경우 prev_emotion(이전 문장 감정)을 from으로 사용.
        - 커넥터/키프레이즈/마커는 JSON/설정에서 로드된 리스트 사용.
        """
        out: List[Dict[str, Any]] = []
        if not sentence or (not self._pivot_connectors_contrast and not self._pivot_connectors_causal):
            return out

        conns = self._pivot_connectors_contrast + self._pivot_connectors_causal + ["그래도"]
        pat = r"(" + "|".join(map(re.escape, conns)) + r")"
        matches = list(re.finditer(pat, sentence))
        if not matches:
            return out

        # 절 분할
        spans = []
        cursor = 0
        for i, m in enumerate(matches):
            cstart, cend = m.start(), m.end()
            left = sentence[cursor:cstart].strip()
            conn = sentence[cstart:cend]
            r_end = matches[i + 1].start() if i < len(matches) - 1 else len(sentence)
            right = sentence[cend:r_end].strip()
            if right:  # 오른쪽 절은 반드시 있어야 전이가 의미 있음
                spans.append((left, conn, right))
            cursor = cend

        # 전이 후보 생성
        for left, connector, right in spans:
            # 좌/우 감정 추출
            left_em = self._detect_emotion_state(left) if left else None
            right_em = self._detect_emotion_state(right)
            # 왼쪽이 비면 prev_emotion 사용
            if not left_em:
                left_em = prev_emotion

            if not (left_em and right_em) or left_em == right_em:
                continue

            # 강도/신뢰도
            inten = self._calculate_intensity(right, right_em)
            conf = 0.25
            if connector in (self._pivot_connectors_causal + ["그래도"]):
                conf += 0.25
            elif connector in self._pivot_connectors_contrast:
                conf += 0.20

            # JSON key_phrases & marker bank 보정
            if any(r.search(right) for r in (self._compiled_key_phrases or [])[:256]):
                conf += 0.10
            if any(r.search(right) for regs in self._compiled_marker_bank.values() for r in regs):
                conf += 0.10
            conf = max(0.0, min(1.0, conf))

            out.append({
                "from_emotion": left_em,
                "to_emotion": right_em,
                "trigger_text": right,
                "confidence": round(conf, 3),
                "intensity": round(inten, 3),
                "position": idx + 0.001,  # 같은 문장 안 전이를 약간 뒤로
                "prev_context": left or (prev_sentence or ""),
                "pivot": True,
            })
        return out

    # ------------------------------ JSON-driven banks ------------------------------
    def _build_marker_bank_and_keywords(self) -> None:
        """
        JSON에서 전이 마커/키워드/키프레이즈를 수집하고,
        안전한 한국어 감정 어휘를 항상 병합하여 카테고리 감지의 리콜을 올린다.
        (라벨링 스키마는 변경하지 않음; 파생 캐시만 확장)
        """
        self._compiled_marker_bank = {
            "gradual": [], "sudden": [], "repetitive": [], "intensifying": [], "attenuating": []
        }
        self._compiled_key_phrases = []
        self._category_keywords = {"희": set(), "노": set(), "애": set(), "락": set()}

        ed_all = self.emotions_data or {}

        # 1) JSON 기반 수집
        for cat, ed in ed_all.items():
            if not isinstance(ed, dict):
                continue
            cat_meta = ed.get("metadata", {}) or {}
            cat_key = cat_meta.get("primary_category", cat) if isinstance(cat_meta, dict) else cat
            if cat_key not in self._category_keywords:
                continue

            # 대표/세부 core keywords
            prof = ed.get("emotion_profile", {}) or {}
            for kw in prof.get("core_keywords", []) or []:
                if isinstance(kw, str) and kw.strip():
                    self._category_keywords[cat_key].add(kw.lower())

            for sub in (ed.get("sub_emotions", {}) or {}).values():
                sp = (sub.get("emotion_profile", {}) or {})
                for kw in sp.get("core_keywords", []) or []:
                    if isinstance(kw, str) and kw.strip():
                        self._category_keywords[cat_key].add(kw.lower())

            # ml markers → regex
            ml = ed.get("ml_training_metadata", {}) or {}
            for group in ("gradual_markers", "sudden_markers", "repetitive_markers",
                          "intensifying_markers", "attenuating_markers"):
                for m in (ml.get(group, []) or []):
                    if isinstance(m, str) and m.strip():
                        self._compiled_marker_bank[group.split("_")[0]].append(self._bregex(m))

            # transitions.patterns → transition_analysis.trigger_words도 intensifying에 보강
            trs = ed.get("emotion_transitions", {}) or {}
            for pat in trs.get("patterns", []) or []:
                an = (pat.get("transition_analysis") or {})
                for w in (an.get("trigger_words", []) or []):
                    if isinstance(w, str) and w.strip():
                        self._compiled_marker_bank["intensifying"].append(self._bregex(w))

            # linguistic_patterns.key_phrases → key_phrases로 컴파일
            ling = ed.get("linguistic_patterns", {}) or {}
            for item in (ling.get("key_phrases", []) or []):
                p = item.get("pattern") if isinstance(item, dict) else item
                if isinstance(p, str) and p.strip():
                    self._compiled_key_phrases.append(self._bregex(p.strip()))

        # 2) 안전어휘(카테고리 분배) – 항상 병합
        #    긍정(희/락), 부정(노/애), 분노/짜증(노), 슬픔/우울(애), 안정/안도(락), 자신감/성취(희)
        pos_terms = ["기쁨", "기쁘", "행복", "즐겁", "만족", "성취", "설렘", "설레", "희망", "기뻤", "기쁜"]
        calm_terms = ["편안", "평온", "안정", "안도", "안도감", "평화로움"]
        anger_terms = ["불만", "짜증", "분노", "화가", "화났", "화가나", "성나", "격분", "격노"]
        sadness_terms = ["슬픔", "슬프", "우울", "허탈", "서글픔", "비통"]
        anxiety_terms = ["불안", "걱정", "초조", "긴장", "공포"]  # 부정 계열(노에 가중)
        confidence_terms = ["자신감", "자부심", "당당", "확신"]  # 긍정 계열(희에 가중)

        # config overlay (있으면 병합)
        for t in self.config.get("context_bias_positive_terms", []):
            if isinstance(t, str) and t.strip(): pos_terms.append(t.strip())
        for t in self.config.get("context_bias_negative_terms", []):
            if isinstance(t, str) and t.strip():
                # 부정 일반어는 불안/분노/슬픔 중 어디에도 매칭 안 되면 일단 노로 가중
                if t not in anger_terms and t not in sadness_terms and t not in anxiety_terms:
                    anger_terms.append(t.strip())

        # 내부 참조용 보관(감지 단계에서 카테고리 지향 가중에 사용)
        self._fallback_pos_terms = {w.lower() for w in pos_terms}
        self._fallback_calm_terms = {w.lower() for w in calm_terms}
        self._fallback_anger_terms = {w.lower() for w in anger_terms}
        self._fallback_sadness_terms = {w.lower() for w in sadness_terms}
        self._fallback_anxiety_terms = {w.lower() for w in anxiety_terms}
        self._fallback_confidence_terms = {w.lower() for w in confidence_terms}

        # 폴백 세트 병합 스위치
        if not bool(self.config.get("allow_bias_fallback_terms", True)):
            logger.info("[TransitionAnalyzer] bias fallback terms disabled by config")
            logger.info("[TransitionAnalyzer] 전이 매핑/트리거 패턴 구축 완료")
            return

        # 카테고리 키워드에 병합
        for w in self._fallback_pos_terms | self._fallback_confidence_terms:
            self._category_keywords["희"].add(w)
        for w in self._fallback_calm_terms | self._fallback_pos_terms:
            self._category_keywords["락"].add(w)
        for w in self._fallback_anger_terms | self._fallback_anxiety_terms:
            self._category_keywords["노"].add(w)
        for w in self._fallback_sadness_terms:
            self._category_keywords["애"].add(w)

        # key_phrases에도 모두 주입(경계 regex)
        for w in (self._fallback_pos_terms | self._fallback_calm_terms |
                  self._fallback_anger_terms | self._fallback_sadness_terms |
                  self._fallback_anxiety_terms | self._fallback_confidence_terms):
            self._compiled_key_phrases.append(self._bregex(w))

        # pivot connectors도 key_phrases로 추가
        for c in (self._pivot_connectors_contrast + self._pivot_connectors_causal):
            self._compiled_key_phrases.append(self._bregex(c))

        logger.info("[TransitionAnalyzer] 전이 매핑/트리거 패턴 구축 완료")

    def _compile_category_regex_shards(self):
        """
        카테고리 키워드(대표 4카테고리)의 대형 정규식을 샤드로 나눠 1회 컴파일하여 보관.
        - self._category_keywords를 기반으로 정규식 OR 묶음 생성
        - 샤드 크기는 self._cat_rx_shard_size
        """
        try:
            self._cat_rx_shards = {}
            for cat, words in (self._category_keywords or {}).items():
                ws = [w for w in set(words) if isinstance(w, str) and w.strip()]
                shards: List[re.Pattern] = []
                if not ws:
                    self._cat_rx_shards[cat] = tuple()
                    continue
                size = max(1, int(self._cat_rx_shard_size))
                for i in range(0, len(ws), size):
                    # 긴 단어를 먼저 매칭하도록 길이 기준 내림차순 정렬
                    alt = "|".join(sorted(map(re.escape, ws[i:i + size]), key=len, reverse=True))
                    try:
                        shards.append(re.compile(alt, re.IGNORECASE))
                    except re.error:
                        # 부분 오류 시 개별 단어로 폴백
                        for w in ws[i:i + size]:
                            try:
                                shards.append(re.compile(re.escape(w), re.IGNORECASE))
                            except re.error:
                                continue
                self._cat_rx_shards[cat] = tuple(shards)
        except Exception as e:
            logger.error(f"[TransitionAnalyzer] _compile_category_regex_shards 실패: {e}")

    def _load_connectors_from_json_and_config(self) -> None:
        """
        커넥터(대조/인과)와 key_phrases를 EMOTIONS.json과 config에서 수집/컴파일.
        - 규칙1: 존재하는 키만 사용
        - 우선순위: config > emotions_data.global_rules > 각 감정의 ml_training_metadata / transitions
        결과:
          self._pivot_connectors_contrast: List[str]
          self._pivot_connectors_causal  : List[str]
          self._compiled_key_phrases     : List[re.Pattern]
        """
        contrast_set, causal_set = set(), set()

        # 1) config 우선
        gr = (self.config.get("global_rules") or {})
        contrast_set |= set(gr.get("contrast_connectives", []))
        causal_set |= set(gr.get("causal_connectives", []))

        # 2) emotions_data.top-level
        if isinstance(self.emotions_data, dict):
            top_gr = (self.emotions_data.get("global_rules") or {})
            contrast_set |= set(top_gr.get("contrast_connectives", []))
            causal_set |= set(top_gr.get("causal_connectives", []))

        # 3) 각 감정의 ml_training_metadata / transitions 에 정의된 패턴 탐색
        for _, ed in (self.emotions_data or {}).items():
            if not isinstance(ed, dict):
                continue
            ml = ed.get("ml_training_metadata", {}) or {}
            for k in ("contrast_connectives", "contrast_markers"):
                for w in (ml.get(k, []) or []):
                    if isinstance(w, str) and w.strip(): contrast_set.add(w.strip())
            for k in ("causal_connectives", "cause_markers"):
                for w in (ml.get(k, []) or []):
                    if isinstance(w, str) and w.strip(): causal_set.add(w.strip())

            trs = ed.get("emotion_transitions", {}) or {}
            for p in (trs.get("patterns", []) or []):
                ttype = p.get("type", "").lower()
                pat = p.get("pattern", "")
                if not isinstance(pat, str) or not pat.strip():
                    continue
                if ttype in {"contrast"}: contrast_set.add(pat.strip())
                if ttype in {"cause", "causal"}: causal_set.add(pat.strip())

        # 4) 최소 안전 fallback(아주 소량만 남김)
        if not contrast_set:
            contrast_set = {"하지만", "그러나", "그런데"}
        if not causal_set:
            causal_set = {"그래서", "때문에", "그러자", "하여", "결국"}

        self._pivot_connectors_contrast = sorted(contrast_set)
        self._pivot_connectors_causal = sorted(causal_set)

        # 5) key_phrases 컴파일 (경계 인식)
        compiled = []
        seen = set()
        for _, ed in (self.emotions_data or {}).items():
            lp = (ed.get("linguistic_patterns") or {})
            for kp in (lp.get("key_phrases") or []):
                pat = kp.get("pattern") if isinstance(kp, dict) else kp
                if not isinstance(pat, str):
                    continue
                p = pat.strip()
                if not p or p in seen:
                    continue
                seen.add(p)
                compiled.append(self._bregex(p))
            # sub_emotions도 포함
            for _, sub in (ed.get("sub_emotions") or {}).items():
                lp2 = (sub.get("linguistic_patterns") or {})
                for kp in (lp2.get("key_phrases") or []):
                    pat = kp.get("pattern") if isinstance(kp, dict) else kp
                    if not isinstance(pat, str):
                        continue
                    p = pat.strip()
                    if not p or p in seen:
                        continue
                    seen.add(p)
                    compiled.append(self._bregex(p))
        self._compiled_key_phrases = compiled

    def _build_transition_map(self) -> None:
        """
        emotions_data의 emotion_transitions를 순회해 전이 맵/트리거 사전을 빌드.
        """
        self.transition_map = {}
        self.trigger_patterns = {}
        ed = self.emotions_data or {}
        try:
            for primary_emotion_key, primary_emotion_data in ed.items():
                if not isinstance(primary_emotion_data, dict):
                    continue
                trs = primary_emotion_data.get("emotion_transitions", {}) or {}
                for pattern in trs.get("patterns", []) or []:
                    frm = pattern.get("from_emotion")
                    to = pattern.get("to_emotion")
                    if not (frm and to):
                        continue
                    key = (frm, to)
                    info = {
                        "triggers": list(set(pattern.get("triggers", []) or [])),
                        "transition_analysis": pattern.get("transition_analysis", {}) or {},
                        "pattern_type": "single",
                        "trigger_words": list(set(pattern.get("trigger_words", []) or [])),
                        "emotion_shift_points": list(set(pattern.get("emotion_shift_points", []) or [])),
                    }
                    self.transition_map[key] = info
                    # shift_points: 단수/복수 키를 모두 병합하여 가독성/일관성 향상
                    an = info["transition_analysis"]
                    sp_raw: List[str] = []
                    esp = an.get("emotion_shift_point")
                    if isinstance(esp, list):
                        sp_raw.extend([x for x in esp if isinstance(x, str) and x.strip()])
                    elif isinstance(esp, str) and esp.strip():
                        sp_raw.append(esp)
                    esps = an.get("emotion_shift_points")
                    if isinstance(esps, list):
                        sp_raw.extend([x for x in esps if isinstance(x, str) and x.strip()])
                    elif isinstance(esps, str) and esps.strip():
                        sp_raw.append(esps)
                    shift_points_merged = list(dict.fromkeys(sp_raw))
                    self.trigger_patterns[key] = {
                        "direct_triggers": info["triggers"],
                        "contextual_triggers": info["transition_analysis"].get("trigger_words", []),
                        "shift_points": shift_points_merged,
                    }
            logger.info("[TransitionAnalyzer] 전이 매핑/트리거 패턴 구축 완료")
        except Exception as e:
            logger.exception(f"[TransitionAnalyzer] 전이 매핑 구축 중 오류: {e}")

    # ------------------------------ detectors & features ------------------------------
    def _extract_trigger_words(
        self,
        text: str,
        transition_info: Dict[str, Any],
        from_emotion: str,
        to_emotion: str
    ) -> List[str]:
        """
        텍스트에서 트리거 단어를 수집(대표/세부 emotion_profile + context_patterns + transitions + marker bank + key_phrases).
        """
        out: List[str] = []
        t = text.lower()

        # to_emotion 프로파일/세부 core_keywords
        to_data = (self.emotions_data or {}).get(to_emotion, {}) or {}
        prof = to_data.get("emotion_profile", {}) or {}
        for kw in prof.get("core_keywords", []) or []:
            if isinstance(kw, str) and kw.lower() in t and kw not in out:
                out.append(kw)

        for sub in (to_data.get("sub_emotions", {}) or {}).values():
            sp = (sub.get("emotion_profile", {}) or {})
            for kw in sp.get("core_keywords", []) or []:
                if isinstance(kw, str) and kw.lower() in t and kw not in out:
                    out.append(kw)

        # transition_info 기반 direct/context
        if transition_info:
            for trg in transition_info.get("triggers", []) or []:
                if isinstance(trg, str) and trg.lower() in t and trg not in out:
                    out.append(trg)
            an = transition_info.get("transition_analysis", {}) or {}
            for trg in an.get("trigger_words", []) or []:
                if isinstance(trg, str) and trg.lower() in t and trg not in out:
                    out.append(trg)

        # context_patterns(situations)
        ctx = to_data.get("context_patterns", {}) or {}
        for situ in (ctx.get("situations", {}) or {}).values():
            for kw in situ.get("keywords", []) or []:
                if isinstance(kw, str) and kw.lower() in t and kw not in out:
                    out.append(kw)
            for var in situ.get("variations", []) or []:
                if isinstance(var, str) and var.lower() in t and var not in out:
                    out.append(var)

        # marker bank(regex)
        for group, regs in self._compiled_marker_bank.items():
            for rg in regs:
                if rg.search(text) and rg.pattern not in out:
                    out.append(rg.pattern)

        # (B) key_phrases(regex)
        for rg in self._compiled_key_phrases:
            if rg.search(text) and rg.pattern not in out:
                out.append(rg.pattern)
        return out

    def _get_transition_info(self, from_emotion: str, to_emotion: str) -> Dict[str, Any]:
        return self.transition_map.get((from_emotion, to_emotion), {})

    def _determine_pattern_type(self, from_emotion: str, to_emotion: str, intensity: float) -> str:
        if from_emotion == to_emotion:
            return "intensity_change"
        pos = {"희", "락"}
        neg = {"애", "노"}
        if from_emotion in pos and to_emotion in neg:
            return "positive_to_negative"
        if from_emotion in neg and to_emotion in pos:
            return "negative_to_positive"
        if from_emotion in pos and to_emotion in pos:
            return "positive_shift"
        if from_emotion in neg and to_emotion in neg:
            return "negative_shift"
        return "neutral_transition"

    def _analyze_transition_pattern(
        self,
        from_emotion: str,
        to_emotion: str,
        current_intensity: float,
        confidence: float,
        emotion_sequence: List[str]
    ) -> Dict[str, Any]:
        """
        전이 패턴(급변/진동/안정/강화) 추정.
        """
        info = {"pattern_type": None, "confidence": confidence, "intensity": current_intensity}
        n = len(emotion_sequence)
        recent = emotion_sequence[-3:] if n >= 3 else emotion_sequence[:]

        if n >= 2 and confidence >= max(self.thr_confidence_min, 0.45):
            if current_intensity >= self.stage_peak_intensity:
                info["pattern_type"] = "escalation"
            elif current_intensity <= 0.3:
                info["pattern_type"] = "de_escalation"

        if from_emotion != to_emotion and confidence >= 0.7:
            info["pattern_type"] = "sudden_shift"

        if len(set(recent)) == 2 and n >= 3:
            info["pattern_type"] = "oscillation"

        if confidence > 0.55 and 0.3 <= current_intensity <= 0.7 and not info["pattern_type"]:
            info["pattern_type"] = "stable_transition"

        return info

    # ------------------------------ B) confidence with bonuses ------------------------------
    def _calculate_transition_confidence(
            self,
            from_emotion: str,
            to_emotion: str,
            current_text: str,
            prev_text: Optional[str]
    ) -> float:
        """
        전이 신뢰도(0~1):
        - to_emotion의 profile/core_keywords, intensity_examples 유사도
        - transition_map (triggers / analysis.trigger_words)
        - to_emotion의 context_patterns (keywords/variations/examples)
        - 글로벌 커넥터/marker bank 보정
        - _check_context_match True 시 소폭 가산
        """
        if not (from_emotion and to_emotion and current_text):
            return 0.0
        t = current_text.strip()
        t_low = t.lower()

        # ---------- 설정/가중 ----------
        gr = (self.emotions_data.get("global_rules", {}) or {})
        cfg_rules = (self.config.get("global_rules", {}) or {})
        if cfg_rules:
            tmp = dict(gr);
            tmp.update(cfg_rules);
            gr = tmp

        W = self.config.get("CONF_WEIGHTS", {}) or gr.get("CONF_WEIGHTS", {}) or {}
        w_prof_kw = float(W.get("profile_keyword", 0.4))
        w_prof_ex = float(W.get("profile_example", 0.4))
        w_trg_dir = float(W.get("transition_trigger", 0.3))
        w_trg_ana = float(W.get("transition_analysis", 0.3))
        w_ctx_kw = float(W.get("context_keyword", 0.3))
        w_ctx_ex = float(W.get("context_example", 0.3))
        w_marker = float(W.get("marker_bonus", 0.2))
        w_ctx_ok = float(W.get("context_match_bonus", 0.1))

        connectors = gr.get("connectors", {
            "contrast": ["하지만", "그러나", "그런데", "그래도"],
            "cause": ["그래서", "때문에", "그러자", "하여", "결국"],
        })
        # contrast/cause 모두 인과/방향 전환 근거로 동일 가중
        w_connector = float(W.get("connector_bonus", 0.2))
        w_intensify = float(W.get("intensify_bonus", 0.15))
        w_attenuate = float(W.get("attenuate_bonus", 0.12))

        score, wsum = 0.0, 0.0

        # ---------- to_emotion 프로파일 ----------
        to_data = (self.emotions_data or {}).get(to_emotion, {}) or {}
        prof = to_data.get("emotion_profile", {}) or {}
        for kw in prof.get("core_keywords", []) or []:
            if self._bregex(kw).search(t):
                score += w_prof_kw;
                wsum += w_prof_kw
        for lv in (prof.get("intensity_levels", {}) or {}).values():
            for ex_list in (lv.get("intensity_examples", {}) or {}).values():
                for ex in ex_list:
                    sim = self._calculate_text_similarity(t, ex)
                    if sim > 0.6:
                        score += w_prof_ex * sim;
                        wsum += w_prof_ex

        # ---------- transition_map 근거 ----------
        key = (from_emotion, to_emotion)
        if key in self.transition_map:
            info = self.transition_map[key]
            for trg in (info.get("triggers", []) or []):
                if self._bregex(trg).search(t):
                    score += w_trg_dir;
                    wsum += w_trg_dir
            an = info.get("transition_analysis", {}) or {}
            for trg in (an.get("trigger_words", []) or []):
                if self._bregex(trg).search(t):
                    score += w_trg_ana;
                    wsum += w_trg_ana

        # ---------- context_patterns ----------
        ctx = to_data.get("context_patterns", {}) or {}
        for situ in (ctx.get("situations", {}) or {}).values():
            for kw in (situ.get("keywords", []) or []):
                if self._bregex(kw).search(t):
                    score += w_ctx_kw;
                    wsum += w_ctx_kw
            for ex in (situ.get("examples", []) or []):
                sim = self._calculate_text_similarity(t, ex)
                if sim > 0.6:
                    score += w_ctx_ex * sim;
                    wsum += w_ctx_ex
            for var in (situ.get("variations", []) or []):
                if self._bregex(var).search(t):
                    score += (w_ctx_kw * 0.8);
                    wsum += (w_ctx_kw * 0.8)

        # ---------- 글로벌 커넥터/마커 보정 ----------
        if any(self._bregex(c).search(t) for c in connectors.get("contrast", [])):
            score += w_connector;
            wsum += w_connector
        if any(self._bregex(c).search(t) for c in connectors.get("cause", [])):
            score += w_connector;
            wsum += w_connector

        # marker bank: intensifying/attenuating
        inc_hit = any(r.search(t) for r in (self._compiled_marker_bank.get("intensifying", []) +
                                            self._compiled_marker_bank.get("gradual", [])))
        att_hit = any(r.search(t) for r in self._compiled_marker_bank.get("attenuating", []))
        if inc_hit:
            score += w_intensify;
            wsum += w_intensify
        if att_hit:
            score += w_attenuate;
            wsum += w_attenuate

        # ---------- 문맥 일치 보정 ----------
        try:
            if self._check_context_match(from_emotion, to_emotion, current_text=t, prev_text=prev_text or ""):
                score += w_ctx_ok;
                wsum += w_ctx_ok
        except Exception:
            pass

        # ---------- 최종(0~1 클립) ----------
        final = (score / wsum) if wsum > 0 else 0.1
        return max(0.0, min(1.0, final))

    # ------------------------------ context / similarity ------------------------------
    def _check_context_match(
        self,
        from_emotion: str,
        to_emotion: str,
        current_text: str,
        prev_text: Optional[str]
    ) -> bool:
        """
        출발/도착 감정의 context_patterns 예시/키워드 유사도로 문맥 적합성 검사.
        """
        if not (from_emotion and to_emotion and current_text):
            return False

        try:
            from_data = (self.emotions_data or {}).get(from_emotion, {}) or {}
            to_data = (self.emotions_data or {}).get(to_emotion, {}) or {}

            context_score, w = 0.0, 0.0
            # from: prev_text 기준
            from_situ = (from_data.get("context_patterns", {}) or {}).get("situations", {}) or {}
            for s in from_situ.values():
                for ex in (s.get("examples", []) or []):
                    if prev_text:
                        sim = self._calculate_text_similarity(prev_text, ex)
                        context_score += 0.7 * sim; w += 0.7
                for var in (s.get("variations", []) or []):
                    if prev_text and isinstance(var, str) and var.lower() in prev_text.lower():
                        context_score += 0.8; w += 0.8
            # to: current_text 기준
            to_situ = (to_data.get("context_patterns", {}) or {}).get("situations", {}) or {}
            for s in to_situ.values():
                for ex in (s.get("examples", []) or []):
                    sim = self._calculate_text_similarity(current_text, ex)
                    context_score += 0.8 * sim; w += 0.8
                for var in (s.get("variations", []) or []):
                    if isinstance(var, str) and var.lower() in current_text.lower():
                        context_score += 0.9; w += 0.9

            # transitions 분석 포인트
            key = (from_emotion, to_emotion)
            if key in self.transition_map:
                an = self.transition_map[key].get("transition_analysis", {}) or {}
                esp = an.get("emotion_shift_point") or an.get("emotion_shift_points")
                combined = (current_text or "").lower() + " " + (prev_text or "").lower()
                if isinstance(esp, str):
                    if esp and esp.lower() in combined:
                        context_score += 1.0; w += 1.0
                elif isinstance(esp, list):
                    for e in esp:
                        if isinstance(e, str) and e and e.lower() in combined:
                            context_score += 1.0; w += 1.0
                            break
                for wd in an.get("trigger_words", []) or []:
                    if isinstance(wd, str) and wd.lower() in current_text.lower():
                        context_score += 0.9; w += 0.9

            return (context_score / w) >= 0.6 if w > 0 else False
        except Exception as e:
            logger.error(f"[TransitionAnalyzer] _check_context_match 오류: {e}")
            return False

    def _calculate_text_similarity(self, t1: str, t2: str) -> float:
        """
        간단 자카드 + 감정 키워드 보정 + 컨텍스트 키워드 보정(0~1)
        """
        if not t1 or not t2:
            return 0.0
        try:
            a, b = t1.lower().split(), t2.lower().split()
            s1, s2 = set(a), set(b)
            base = (len(s1 & s2) / len(s1 | s2)) if (s1 | s2) else 0.0

            cat_words = set().union(*self._category_keywords.values()) if self._category_keywords else set()
            emo_boost = 0.0
            if cat_words:
                emo_boost = 0.4 * (len((s1 & cat_words) & (s2 & cat_words)) / max(1, len(cat_words)))

            ctx_boost = 0.2 * (1.0 if any(x in s1 and x in s2 for x in ["점점", "갑자기", "차츰", "더욱"]) else 0.0)
            return max(0.0, min(1.0, 0.4 * base + emo_boost + ctx_boost))
        except Exception as e:
            logger.error(f"[TransitionAnalyzer] 유사도 계산 오류: {e}")
            return 0.0

    # ------------------------------ flow / stages ------------------------------
    def _analyze_emotion_flow(
        self,
        emotion_sequence: List[str],
        transitions: List[Dict[str, Any]],
        stage_tags: List[str]
    ) -> Dict[str, Any]:
        if not emotion_sequence:
            return {"flow_pattern": "unknown", "stability_score": 0.0, "volatility": 0.0, "dominant_transitions": []}
        if len(emotion_sequence) == 1:
            return {"flow_pattern": "stable_flow", "stability_score": 1.0, "volatility": 0.0, "dominant_transitions": []}

        pairs = [(t["from_emotion"], t["to_emotion"]) for t in transitions]
        counts = Counter(pairs)
        unique_states = len(set(emotion_sequence))
        stability_score = max(0.0, min(1.0, 1.0 - ((unique_states - 1) / max(1, len(emotion_sequence) - 1))))
        volatility = len(transitions) / max(1, (len(emotion_sequence) - 1))
        flow_pattern = self._determine_flow_pattern(emotion_sequence, transitions, stability_score, volatility)

        return {
            "flow_pattern": flow_pattern,
            "stability_score": round(stability_score, 2),
            "volatility": round(volatility, 2),
            "dominant_transitions": [f"{a}->{b}" for (a, b), _ in counts.most_common(3)],
            "stages": stage_tags,
        }

    def _determine_flow_pattern(
        self,
        emotion_sequence: List[str],
        transitions: List[Dict[str, Any]],
        stability_score: float,
        volatility: float
    ) -> str:
        if not transitions:
            return "stable_flow"
        intens = [t.get("intensity", 0.5) for t in transitions] or [0.5]
        diff = max(intens) - min(intens)
        if volatility >= self.flow_thr_volatile:
            return "volatile_flow"
        if stability_score >= 0.7:
            return "stable_flow"
        if diff > 0.6:
            return "intensity_driven"
        if len(set(emotion_sequence)) >= self.flow_thr_complex_unique:
            return "complex_flow"
        return "balanced_flow"

    def _stage_from_transition(self, intensity: float, confidence: float, transitions: List[Dict[str, Any]], idx: int) -> str:
        """
        (D) 진행 단계 추정 규칙 개선:
        - Peak: intensity>=peak_thr & conf>=0.6
        - Development: (window 내 동일 to_emotion 반복) OR (마커 intensifying) OR (강도 상승 ≥0.05) AND conf≥0.3
        - Aftermath: Peak 이후 (강도 하락 ≥0.15) OR (극성 반전) OR (attenuating marker)
        - Trigger: 그 외
        """
        if idx == 1 or len(transitions) == 1:
            return "Trigger"

        last = transitions[-1]
        last2 = transitions[-2] if len(transitions) >= 2 else None
        to_now = last["to_emotion"]
        to_prev = last2["to_emotion"] if last2 else None

        # Peak
        if intensity >= self.stage_peak_intensity and confidence >= 0.6:
            return "Peak"

        # Development
        window = transitions[max(0, len(transitions) - self.stage_dev_window):]
        same_to = all(t["to_emotion"] == to_now for t in window) and len(window) >= 2
        rising = False
        if last2:
            rising = (float(last["intensity"]) - float(last2.get("intensity", 0.5))) >= 0.05
        txt = (last.get("trigger_text", "") + " " + last.get("prev_context", "")).strip()
        inc_marker = any(r.search(txt) for r in (self._compiled_marker_bank.get("intensifying", []) + self._compiled_marker_bank.get("gradual", [])))
        if (same_to or inc_marker or rising) and confidence >= 0.3:
            return "Development"

        # Aftermath
        if last2:
            # 강도 하락(≥0.15) or 극성 반전 or attenuating marker
            decayed = (float(last2.get("intensity", 0.5)) - float(last["intensity"])) >= 0.15
            cat = self._cat_of(to_now); cat_prev = self._cat_of(to_prev)
            pol_flip = (cat and cat_prev and cat != cat_prev)
            att_marker = any(r.search(txt) for r in self._compiled_marker_bank.get("attenuating", []))
            if decayed or pol_flip or att_marker:
                return "Aftermath"

        return "Trigger"

    # ------------------------------ scoring features ------------------------------
    def _calculate_intensity(self, text: str, emotion: str) -> float:
        """
        강도 추정:
        - intensity_levels 예시/상황 예시 유사도
        - marker bank(gradual/intensifying/attenuating 등) 가산/감산
        - 0~1 클리핑
        """
        if not text or not emotion:
            return 0.0

        # LRU 캐시: 텍스트×감정 강도 재사용
        ckey = ("intensity", str(emotion), (text or "").strip().lower())
        # transition_cache가 dict인 경우 직접 접근
        if hasattr(self, 'transition_cache') and isinstance(self.transition_cache, dict):
            cached = self.transition_cache.get(ckey)
        elif hasattr(self, 'transition_cache') and hasattr(self.transition_cache, 'get'):
            cached = self.transition_cache.get(ckey)
        else:
            cached = None
        if isinstance(cached, float):
            return cached

        e = (self.emotions_data or {}).get(emotion, {}) or {}
        prof = e.get("emotion_profile", {}) or {}
        levels = prof.get("intensity_levels", {}) or {}
        ctx = e.get("context_patterns", {}) or {}

        base, w = 0.0, 0.0
        for lv, lvdata in levels.items():
            w_lv = {"low": 0.3, "medium": 0.6, "high": 0.9}.get(lv, 0.5)
            for ex_list in (lvdata.get("intensity_examples", {}) or {}).values():
                for ex in ex_list:
                    sim = self._calculate_text_similarity(text, ex)
                    if sim > 0:
                        base += w_lv * sim; w += w_lv
        for situ in (ctx.get("situations", {}) or {}).values():
            w_s = float(situ.get("intensity", 0.5))
            for ex in (situ.get("examples", []) or []):
                sim = self._calculate_text_similarity(text, ex)
                if sim > 0:
                    base += w_s * sim; w += w_s

        val = (base / w) if w > 0 else 0.5

        # (D) marker 가산/감산
        up_markers = self._compiled_marker_bank.get("intensifying", []) + self._compiled_marker_bank.get("gradual", [])
        down_markers = self._compiled_marker_bank.get("attenuating", [])
        inc = any(r.search(text) for r in up_markers)
        dec = any(r.search(text) for r in down_markers)
        if inc and not dec:
            val = min(1.0, val + 0.1)
        elif dec and not inc:
            val = max(0.0, val - 0.08)

        val = max(0.0, min(1.0, val))
        # transition_cache가 dict인 경우 직접 할당
        if hasattr(self, 'transition_cache') and isinstance(self.transition_cache, dict):
            self.transition_cache[ckey] = float(val)
        elif hasattr(self, 'transition_cache') and hasattr(self.transition_cache, 'put'):
            self.transition_cache.put(ckey, float(val))
        return val

    # ------------------------------ sentence & emotion detect ------------------------------
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        KSS 우선 문장 분할 → (동적) 커넥터 기반 서브 분할.
        - 커넥터 목록은 self._pivot_connectors_contrast / self._pivot_connectors_causal + ['그래도'] 를 사용
          (없으면 최소 안전 fallback 사용)
        - 서브 분할 시 '커넥터+오른쪽 절'을 하나의 파트로 유지하여 의미 보존
        - 너무 짧은/의미 없는 파편은 제거
        """
        if not text or not text.strip():
            return [text]

        # 1) 1차 분할: KSS → 실패 시 Fallback
        try:
            if kss is not None:
                sents = kss.split_sentences(text, backend="pecab", num_workers=1)
                sents = [s.strip() for s in sents if s and s.strip()]
            else:
                raise RuntimeError("kss not available")
        except Exception:
            # 간단한 구두점/개행 기반 분할 + 긴 조각은 쉼표/세미콜론으로 보조 분할
            rough = re.split(r'(?:[.!?]+|\n+|。|！|？)+', text)
            sents = []
            for chunk in rough:
                c = (chunk or "").strip()
                if not c:
                    continue
                if len(c) >= 120:
                    parts = re.split(r'[;,]', c)
                    sents.extend([p.strip() for p in parts if p and p.strip()])
                else:
                    sents.append(c)

        if not sents:
            return []

        # 2) 동적 커넥터 구성 (설정/JSON 주도, 없으면 안전 fallback)
        contrast = getattr(self, "_pivot_connectors_contrast", None) or []
        causal = getattr(self, "_pivot_connectors_causal", None) or []
        connectors = list(dict.fromkeys(contrast + causal + ["그래도"]))  # 중복 제거 + 순서 유지
        if not connectors:
            connectors = ["하지만", "그러나", "그런데", "그래서", "때문에", "그러자", "하여", "결국", "그래도"]

        # 정규식: 커넥터 토큰을 '오른쪽 절의 시작'으로 유지하기 위해
        #  - 분할은 커넥터 앞에서 하지만, 결과 파트에는 '커넥터+오른쪽절'이 포함되도록 슬라이싱
        conn_pat = r"(?:%s)" % "|".join(map(re.escape, connectors))

        fine: List[str] = []
        for s in sents:
            s_clean = s.strip()
            if not s_clean:
                continue

            # 커넥터 탐색
            matches = list(re.finditer(conn_pat, s_clean))
            if not matches:
                fine.append(s_clean)
                continue

            cursor = 0
            for i, m in enumerate(matches):
                cstart, cend = m.start(), m.end()
                left = s_clean[cursor:cstart].strip()
                # 다음 커넥터 전까지를 오른쪽으로
                next_start = matches[i + 1].start() if i < len(matches) - 1 else len(s_clean)
                right = s_clean[cend:next_start].strip()
                connector = s_clean[cstart:cend]

                # 왼쪽 절은 '문장 내부 피벗'에서 prev_context로도 쓰일 수 있으므로 짧더라도 보존(단, 완전무의미 제외)
                if left and len(left) >= 2 and not re.fullmatch(r'[,;]+', left):
                    fine.append(left)

                # 오른쪽 절은 커넥터 포함해 의미 유지
                if right:
                    # 앞뒤 불필요한 구두점 정리
                    right_norm = re.sub(r'^[,;]+', '', right).strip()
                    if right_norm:
                        fine.append((connector + " " + right_norm).strip())

                cursor = cend

            # 마지막 커넥터 이후에 남은 테일(다음 커넥터가 없었던 구간)
            if cursor < len(s_clean):
                tail = s_clean[cursor:].strip()
                if tail and len(tail) >= 2 and not re.fullmatch(r'[,;]+', tail):
                    fine.append(tail)

        # 3) 파편 정리: 너무 짧은 조각/의미 없는 파편 제거 (한글/영문/숫자 최소 2자 이상)
        cleaned = []
        for frag in fine:
            f = (frag or "").strip()
            if not f:
                continue
            # 의미 글자 수 기준(한글/영문/숫자 2자 이상)
            if len("".join(re.findall(r'[가-힣A-Za-z0-9]', f))) < 2:
                continue
            cleaned.append(f)

        # 커넥터 분할로 아무것도 남지 않았으면, 원본 1차 문장 반환
        return cleaned if cleaned else sents

    def _detect_emotion_state(self, sentence: str) -> Optional[str]:
        """
        JSON/설정 주도 스코어링으로 대표 감정(희/노/애/락) 감지.
        - core_keywords(대표+세부) 경계 매칭
        - context_patterns(키워드/variations) 매칭
        - transition_map(triggers/analysis.trigger_words) 증거
        - global_rules/context_bias_*_terms/커넥터 보정
        - 가중/임계는 config → emotions_data.global_rules → fallback 순
        """
        if not sentence or not self.emotions_data:
            return None
        s = sentence.strip()
        s_low = s.lower()
        # LRU 캐시: 동일 문장 재사용 시 비용 절감
        cache_key = ("detect", s_low)
        # transition_cache가 dict인 경우 직접 접근
        if hasattr(self, 'transition_cache') and isinstance(self.transition_cache, dict):
            cached = self.transition_cache.get(cache_key)
        elif hasattr(self, 'transition_cache') and hasattr(self.transition_cache, 'get'):
            cached = self.transition_cache.get(cache_key)
        else:
            cached = None
        if cached is not None:
            return cached if isinstance(cached, str) else None

        # ---------- 가중/임계 파라미터(설정 → 글로벌룰 → 기본) ----------
        gr = {}
        try:
            gr = (self.emotions_data.get("global_rules", {}) or {})
            # config 우선
            cfg_rules = (self.config.get("global_rules", {}) or {})
            if cfg_rules:
                tmp = dict(gr);
                tmp.update(cfg_rules);
                gr = tmp
        except Exception:
            pass

        W = self.config.get("DETECT_WEIGHTS", {}) or gr.get("DETECT_WEIGHTS", {}) or {}
        w_core = float(W.get("core_keyword", 1.0))
        w_sub_core = float(W.get("sub_core_keyword", 0.9))
        w_ctx_kw = float(W.get("context_keyword", 0.7))
        w_ctx_var = float(W.get("context_variation", 0.6))
        w_trg_dir = float(W.get("transition_trigger", 0.9))  # 0.75 → 0.9
        w_trg_ana = float(W.get("transition_analysis", 0.75))  # 0.65 → 0.75
        w_bias_posH = float(W.get("bias_pos_heui", 0.15))  # 희 강화
        w_bias_posL = float(W.get("bias_pos_rak", 0.10))  # 락 강화
        w_bias_negH = float(W.get("bias_neg_ae", 0.15))  # 애 강화
        w_bias_negL = float(W.get("bias_neg_no", 0.10))  # 노 강화

        thr_min = float(self.config.get("thresholds", {}).get("detect_min", 0.05))  # 임계값을 더 낮춤

        # 바이어스/커넥터(설정/글로벌룰 → 기본)
        pos_terms = set(self.config.get("context_bias_positive_terms", gr.get("context_bias_positive_terms", [])))
        neg_terms = set(self.config.get("context_bias_negative_terms", gr.get("context_bias_negative_terms", [])))
        connectors = gr.get("connectors", {
            "contrast": ["하지만", "그러나", "그런데", "그래도"],
            "cause": ["그래서", "때문에", "그러자", "하여", "결국"],
        })

        # ---------- 스코어링 시작 ----------
        scores = {"희": 0.0, "노": 0.0, "애": 0.0, "락": 0.0}

        # 기본 감정 키워드 매칭 (EMOTIONS.json이 없거나 비어있을 때 폴백)
        basic_keywords = {
            "희": ["기쁨", "행복", "기쁘", "행복", "즐거", "신나", "환희", "기쁜", "행복한", "즐거운", "신나는"],
            "노": ["화", "분노", "짜증", "불만", "화나", "분노", "짜증나", "불만", "화난", "분노한", "짜증난"],
            "애": ["슬픔", "우울", "슬프", "우울", "눈물", "슬픈", "우울한", "눈물이"],
            "락": ["안정", "편안", "만족", "안정감", "편안한", "만족한", "안정된", "편안함"]
        }
        
        # 기본 키워드 매칭
        for emotion, keywords in basic_keywords.items():
            for keyword in keywords:
                if keyword in s_low:
                    scores[emotion] += w_core

        # 1) core keywords(대표+세부) – 샤드 정규식으로 전수 매칭 비용 절감
        for cat, shards in (self._cat_rx_shards or {}).items():
            if not shards:
                continue
            # 샤드 중 하나라도 매치되면 가중 1회만 가산(중복 가산 방지)
            if any(rx.search(s) for rx in shards):
                scores[cat] += w_core

        # 2) context_patterns – 키워드/variations/예시
        for cat_key, ed in (self.emotions_data or {}).items():
            cat = self._cat_of(cat_key) or cat_key
            if cat not in scores:
                continue
            ctx = (ed.get("context_patterns", {}) or {}).get("situations", {}) or {}
            for situ in ctx.values():
                for kw in (situ.get("keywords", []) or []):
                    if self._bregex(kw).search(s):
                        scores[cat] += w_ctx_kw
                for var in (situ.get("variations", []) or []):
                    if self._bregex(var).search(s):
                        scores[cat] += w_ctx_var

            # 세부 감정 컨텍스트
            for sub in (ed.get("sub_emotions", {}) or {}).values():
                sp = (sub.get("context_patterns", {}) or {}).get("situations", {}) or {}
                for situ in sp.values():
                    for kw in (situ.get("keywords", []) or []):
                        if self._bregex(kw).search(s):
                            scores[cat] += (w_ctx_kw * 0.9)

        # 3) 전이 트리거(transition_map) – direct/analysis
        for (a, b), info in (self.transition_map or {}).items():
            b_cat = self._cat_of(b) or b
            if b_cat not in scores:
                continue
            for trg in (info.get("triggers", []) or []):
                if self._bregex(trg).search(s):
                    scores[b_cat] += w_trg_dir
            an = info.get("transition_analysis", {}) or {}
            for trg in (an.get("trigger_words", []) or []):
                if self._bregex(trg).search(s):
                    scores[b_cat] += w_trg_ana

        # 4) 바이어스(긍/부정 일반 문형)
        if any(self._bregex(t).search(s) for t in pos_terms):
            scores["희"] += w_bias_posH
            scores["락"] += w_bias_posL
        if any(self._bregex(t).search(s) for t in neg_terms):
            scores["애"] += w_bias_negH
            scores["노"] += w_bias_negL

        # 5) 커넥터 보정(문장 내부 전이 가능성 향상 – 대비/인과)
        if any(self._bregex(c).search(s) for c in connectors.get("contrast", [])):
            # 대비: 흔히 긍정→부정 혹은 방향 전환 힌트
            scores["노"] += 0.08;
            scores["애"] += 0.08
        if any(self._bregex(c).search(s) for c in connectors.get("cause", [])):
            # 인과: 상황 전환, 회복/안정 힌트
            scores["희"] += 0.06;
            scores["락"] += 0.04

        # 최댓값 감정 선택 + 임계
        cat = max(scores.items(), key=lambda x: x[1])[0] if scores else None
        max_score = scores.get(cat, 0.0) if cat else 0.0
        
        # 임계값을 만족하지 않더라도 기본 감정 키워드가 매칭되면 감정 감지
        if max_score <= thr_min:
            # 기본 키워드가 매칭된 경우 강제로 감정 감지
            for emotion, keywords in basic_keywords.items():
                if any(keyword in s_low for keyword in keywords):
                    res = emotion
                    break
            else:
                res = None
        else:
            res = cat
        # transition_cache가 dict인 경우 직접 할당
        if hasattr(self, 'transition_cache') and isinstance(self.transition_cache, dict):
            self.transition_cache[cache_key] = res
        elif hasattr(self, 'transition_cache') and hasattr(self.transition_cache, 'put'):
            self.transition_cache.put(cache_key, res)
        
        return res

    # ------------------------------ metrics & misc ------------------------------
    def _update_metrics(self, result: Dict[str, Any], start_time: float, extra_notes: Optional[Dict[str, Any]] = None) -> None:
        transitions = result.get("transitions", []) or []
        confidence_avg = round(sum(t.get("confidence", 0.0) for t in transitions) / max(1, len(transitions)), 3)
        mem_now = self.memory_monitor.get_usage() if self.memory_monitor else 0
        pattern_count = len(result.get("patterns", {}))

        # marker 집계
        marker_hits = 0
        for p in result.get("patterns", {}).values():
            marker_hits += len(p.get("trigger_words", []) or [])

        notes = extra_notes or {}
        self.metrics = TransitionMetrics(
            processing_time=time.time() - start_time,
            memory_usage=int(mem_now),
            transition_count=len(transitions),
            pattern_count=int(pattern_count),
            confidence_avg=float(confidence_avg),
            flow_pattern=result.get("flow_analysis", {}).get("flow_pattern", "unknown"),
            stability_score=result.get("flow_analysis", {}).get("stability_score", 0.0),
            volatility=result.get("flow_analysis", {}).get("volatility", 0.0),
            peak_intensity_avg=round(sum(t.get("intensity", 0.5) for t in transitions) / max(1, len(transitions)), 3),
            intensifier_hits=int(marker_hits),
            marker_hits=int(marker_hits),
            category_counts=self._category_histogram(transitions),
            notes=notes
        )

        result["metrics"] = {
            "processing_time": round(self.metrics.processing_time, 3),
            "memory_usage": self.metrics.memory_usage,
            "transition_count": self.metrics.transition_count,
            "pattern_count": self.metrics.pattern_count,
            "confidence_avg": self.metrics.confidence_avg,
            "flow_pattern": self.metrics.flow_pattern,
            "stability_score": self.metrics.stability_score,
            "volatility": self.metrics.volatility,
        }

    def _category_histogram(self, transitions: List[Dict[str, Any]]) -> Dict[str, int]:
        hist = {"희": 0, "노": 0, "애": 0, "락": 0}
        for t in transitions:
            fa = self._cat_of(t.get("from_emotion"))
            ta = self._cat_of(t.get("to_emotion"))
            if fa in hist:
                hist[fa] += 1
            if ta in hist:
                hist[ta] += 1
        return hist

    # ------------------------------ helpers ------------------------------
    def _cat_of(self, key: str) -> Optional[str]:
        """
        key가 대표감정(희/노/애/락) 또는 세부감정명일 때, 대표 카테고리 반환.
        (JSON의 metadata.primary_category를 탐색)
        """
        if key in {"희", "노", "애", "락"}:
            return key
        ed = (self.emotions_data or {}).get(key, {})
        if isinstance(ed, dict):
            md = ed.get("metadata", {})
            if isinstance(md, dict):
                cat = md.get("primary_category")
                if isinstance(cat, str) and cat in {"희", "노", "애", "락"}:
                    return cat
        # 세부감정으로 넘어온 경우: 모든 top-level 순회(규칙2 확장 안전)
        for top_key, data in (self.emotions_data or {}).items():
            subs = (data.get("sub_emotions", {}) or {})
            if key in subs:
                md = subs[key].get("metadata", {})
                cat = md.get("primary_category") if isinstance(md, dict) else None
                return cat if cat in {"희", "노", "애", "락"} else top_key if top_key in {"희", "노", "애", "락"} else None
        return None

    def _bregex(self, text: str) -> re.Pattern:
        """
        경계 인식 정규식(캐시 사용).
        - 한글 포함: '용어' + (한글 0~3자) 허용 (조사/어미 대응)
        - 비한글: 단어 경계 고정
        - self._regex_cache로 반복 컴파일 방지
        """
        if not isinstance(text, str) or not text.strip():
            return re.compile(r"$^")
        base = text.strip()
        # 캐시 키: (is_korean, base)
        is_ko = bool(re.search(r"[가-힣]", base))
        key = ("ko", base) if is_ko else ("wb", base)
        rg = self._regex_cache.get(key)
        if rg:
            return rg
        try:
            if is_ko:
                pat = rf"{re.escape(base)}(?:[가-힣]{{0,3}})?"
            else:
                pat = rf"(?<!\w){re.escape(base)}(?!\w)"
            rg = re.compile(pat, re.IGNORECASE)
        except re.error:
            rg = re.compile(re.escape(base), re.IGNORECASE)
        self._regex_cache[key] = rg
        return rg

    # ------------------------------ error ------------------------------
    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        return {"error": error_message, "transitions": [], "flow_analysis": {}, "metrics": {}}


# =============================================================================
# EmotionProgressionAnalyzer (Refined)
# =============================================================================
class EmotionProgressionAnalyzer:
    """
    감정 진행 분석(Trigger / Development / Peak / Aftermath).
    - TransitionAnalyzer가 산출한 transitions 리스트를 받아 스테이지를 안정적으로 산정
    - EMOTIONS.json의 메타데이터/마커/컨텍스트를 데이터 주도로 활용(규칙1/2/3)
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        emotions_data: Optional[Dict[str, Any]] = None,
        compiled_marker_bank: Optional[Dict[str, List[re.Pattern]]] = None,
    ):
        self.config = config or {}
        self.emotions_data = emotions_data or {}
        # thresholds
        self.peak_thr = float(self.config.get("stages", {}).get("peak_intensity", 0.7))
        self.dev_window = int(self.config.get("stages", {}).get("development_window", 2))
        self.ctx_dev_boost = float(self.config.get("stages", {}).get("development_ctx_boost", 0.15))  # 0.10→0.15
        self.after_decay_thr = float(self.config.get("stages", {}).get("aftermath_decay", 0.15))      # peak 대비 하락폭
        self.min_conf_for_stage = float(self.config.get("thresholds", {}).get("min_conf_for_stage", 0.3))

        # 마커 뱅크(TransitionAnalyzer에서 넘겨줄 수 있음). 없으면 JSON에서 약식으로 구축
        self.marker_bank = compiled_marker_bank or self._build_marker_bank_from_json()

        # 카테고리/극성 맵
        self.category_map = self._build_category_map()  # emotion_key/세부감정 → '희'/'노'/'애'/'락'
        self.polarity_map = self._build_polarity_map()  # category or emotion_key → 'positive'/'negative'/'neutral'

    # ------------------------------------------------------------------ public
    def analyze_progression(self, transitions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        입력: TransitionAnalyzer가 만든 transitions: List[{
            from_emotion, to_emotion, trigger_text, confidence, intensity, position, prev_context
        }]
        반환: {trigger, development[], peak, aftermath[], progression_metrics}
        """
        if not transitions:
            return {"trigger": {}, "development": [], "peak": {}, "aftermath": [], "progression_metrics": {}}

        # time/position 기준 정렬
        trans = sorted(transitions, key=lambda x: x.get("position", 0))
        intens = [float(t.get("intensity", 0.5)) for t in trans]
        confs  = [float(t.get("confidence", 0.0)) for t in trans]

        # ---------- Trigger ----------
        trig = trans[0]
        trigger = {"emotion": trig["to_emotion"], "intensity": float(trig.get("intensity", 0.5))}
        trigger_conf = self._stage_confidence(trig, stage="Trigger")

        # ---------- Peak ----------
        peak_idx = self._find_peak_index(trans, self.peak_thr, self.min_conf_for_stage)
        peak_tr  = trans[peak_idx]
        peak     = {"emotion": peak_tr["to_emotion"], "intensity": float(peak_tr.get("intensity", 0.0))}
        peak_conf = self._stage_confidence(peak_tr, stage="Peak")

        # ---------- Development ----------
        development_list = self._collect_development(trans, start_idx=0, end_idx=peak_idx)

        # ---------- Aftermath ----------
        aftermath_list = self._collect_aftermath(trans, start_idx=peak_idx, peak_val=float(peak_tr.get("intensity", 0.0)))

        # ---------- Metrics ----------
        metrics = self._progression_metrics(trans, development_list, peak_idx, aftermath_list)
        # 스테이지 confidence (요약)
        metrics["stage_confidence"] = {
            "trigger": round(trigger_conf, 3),
            "peak": round(peak_conf, 3),
            "development_avg": round(self._avg([d.get("stage_confidence", 0.0) for d in development_list]), 3) if development_list else 0.0,
            "aftermath_avg":   round(self._avg([a.get("stage_confidence", 0.0) for a in aftermath_list]), 3) if aftermath_list   else 0.0,
        }

        return {
            "trigger": trigger,
            "development": development_list,
            "peak": peak,
            "aftermath": aftermath_list,
            "progression_metrics": metrics,
        }

    # -------------------------------------------------------------- builders
    def _build_marker_bank_from_json(self) -> Dict[str, List[re.Pattern]]:
        """
        EMOTIONS.json에서 전이/강도 관련 마커를 느슨하게 수집해 정규식으로 컴파일.
        규칙1: 존재하는 키만 사용.
        """
        bank = {"gradual": [], "sudden": [], "repetitive": [], "intensifying": [], "attenuating": []}
        try:
            for _, ed in (self.emotions_data or {}).items():
                if not isinstance(ed, dict):
                    continue
                ml = ed.get("ml_training_metadata", {}) or {}
                for group, key in [("gradual", "gradual_markers"), ("sudden", "sudden_markers"),
                                   ("repetitive", "repetitive_markers"), ("intensifying", "intensifying_markers"),
                                   ("attenuating", "attenuating_markers")]:
                    lst = ml.get(key, [])
                    if isinstance(lst, list):
                        for m in lst:
                            if isinstance(m, str) and m.strip():
                                bank[group].append(self._bregex(m))
                trs = ed.get("emotion_transitions", {}) or {}
                for pat in trs.get("patterns", []) or []:
                    an = pat.get("transition_analysis", {}) or {}
                    for w in an.get("trigger_words", []) or []:
                        if isinstance(w, str) and w.strip():
                            bank["intensifying"].append(self._bregex(w))
        except Exception as e:
            logger.error(f"[EmotionProgressionAnalyzer] marker bank build failed: {e}")
        return bank

    def _build_category_map(self) -> Dict[str, str]:
        """
        emotion_key/세부감정을 대표카테고리('희','노','애','락')로 매핑(규칙2).
        """
        cmap = {"희": "희", "노": "노", "애": "애", "락": "락"}
        for key, ed in (self.emotions_data or {}).items():
            if not isinstance(ed, dict):
                continue
            md = ed.get("metadata", {}) or {}
            cat = md.get("primary_category")
            if isinstance(cat, str) and cat in {"희", "노", "애", "락"}:
                cmap[key] = cat
            subs = ed.get("sub_emotions", {}) or {}
            for sub_key, sub in subs.items():
                md2 = sub.get("metadata", {}) or {}
                cat2 = md2.get("primary_category")
                if isinstance(cat2, str) and cat2 in {"희", "노", "애", "락"}:
                    cmap[sub_key] = cat2
        return cmap

    def _build_polarity_map(self) -> Dict[str, str]:
        """
        카테고리/감정의 극성 맵: JSON의 ml_training_metadata.polarity 우선,
        없으면 대표 카테고리 휴리스틱(희/락: positive, 노/애: negative).
        """
        pmap: Dict[str, str] = {}
        base = {"희": "positive", "락": "positive", "노": "negative", "애": "negative"}
        for k in ["희", "노", "애", "락"]:
            pmap[k] = base[k]
        for k, ed in (self.emotions_data or {}).items():
            pol = (ed.get("ml_training_metadata", {}) or {}).get("polarity")
            cat = (ed.get("metadata", {}) or {}).get("primary_category")
            if isinstance(pol, str) and pol.lower() in {"positive", "negative", "neutral"}:
                pmap[k] = pol.lower()
            elif isinstance(cat, str) and cat in base:
                pmap[k] = base[cat]
        return pmap

    # -------------------------------------------------------------- stages
    def _find_peak_index(self, trans: List[Dict[str, Any]], peak_thr: float, min_conf: float) -> int:
        """
        intensity*confidence 최댓값을 우선, 없으면 intensity 최댓값.
        threshold(peak_thr&min_conf)를 넘는 가장 앞 인덱스를 피크로 우선 선택.
        """
        best_idx, best_score = 0, -1.0
        candidate_idx = None
        for i, t in enumerate(trans):
            inten = float(t.get("intensity", 0.5))
            conf  = float(t.get("confidence", 0.0))
            s = inten * conf
            if inten >= peak_thr and conf >= min_conf and candidate_idx is None:
                candidate_idx = i
            if s > best_score:
                best_score = s; best_idx = i
        return candidate_idx if candidate_idx is not None else best_idx

    def _collect_development(self, trans: List[Dict[str, Any]], start_idx: int, end_idx: int) -> List[Dict[str, Any]]:
        """
        Trigger 이후 ~ Peak 이전 구간에서
        - 동일 방향(같은 to_emotion) 유지 OR
        - intensifying/gradual marker OR
        - 강도 상승(직전 대비 ≥0.05)
        중 하나라도 만족 + conf≥0.3이면 development.
        """
        out: List[Dict[str, Any]] = []
        if end_idx <= start_idx:
            return out

        last_to  = trans[start_idx]["to_emotion"]
        last_int = float(trans[start_idx].get("intensity", 0.5))
        for i in range(start_idx + 1, end_idx):
            t = trans[i]
            to    = t["to_emotion"]
            inten = float(t.get("intensity", 0.5))
            conf  = float(t.get("confidence", 0.0))
            txt   = (t.get("trigger_text", "") + " " + t.get("prev_context", "")).strip()

            same_dir  = (to == last_to)
            inc_marker = self._has_marker(txt, groups=("intensifying", "gradual"))
            rising    = (inten - last_int) >= 0.05

            if (same_dir or inc_marker or rising) and conf >= self.min_conf_for_stage:
                out.append({
                    "emotion": to,
                    "intensity": inten,
                    "stage_confidence": self._stage_confidence(t, stage="Development")
                })
            last_to, last_int = to, inten
        return out

    def _collect_aftermath(self, trans: List[Dict[str, Any]], start_idx: int, peak_val: float) -> List[Dict[str, Any]]:
        """
        Peak 이후 구간에서
        - 강도 하락(peak 대비 ≥ after_decay_thr)
        - 반대 극성 카테고리로 이동
        - attenuating/gradual marker
        중 하나라도 만족 + conf≥0.3이면 aftermath.
        """
        out: List[Dict[str, Any]] = []
        if start_idx >= len(trans) - 1:
            return out

        for i in range(start_idx + 1, len(trans)):
            t = trans[i]
            inten = float(t.get("intensity", 0.5))
            conf  = float(t.get("confidence", 0.0))
            if conf < self.min_conf_for_stage:
                continue

            to   = t["to_emotion"]
            prev = trans[i - 1]["to_emotion"]
            pol_to   = self.polarity_map.get(self._cat_of(to) or to, "neutral")
            pol_prev = self.polarity_map.get(self._cat_of(prev) or prev, "neutral")
            txt = (t.get("trigger_text", "") + " " + t.get("prev_context", "")).strip()

            decayed       = (peak_val - inten) >= self.after_decay_thr
            polarity_flip = (pol_to != "neutral" and pol_prev != "neutral" and pol_to != pol_prev)
            att_marker    = self._has_marker(txt, groups=("attenuating", "gradual"))

            if decayed or polarity_flip or att_marker:
                out.append({
                    "emotion": to,
                    "intensity": inten,
                    "stage_confidence": self._stage_confidence(t, stage="Aftermath")
                })
        return out

    # -------------------------------------------------------------- metrics helpers
    def _progression_metrics(
        self,
        trans: List[Dict[str, Any]],
        dev: List[Dict[str, Any]],
        peak_idx: int,
        aft: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        # durations
        dev_dur = len(dev)
        aft_dur = len(aft)

        # slopes (평균 상승/하강 기울기)
        rise_slope, fall_slope = 0.0, 0.0
        if peak_idx > 0:
            rise = [float(trans[i + 1].get("intensity", 0.5)) - float(trans[i].get("intensity", 0.5))
                    for i in range(0, peak_idx)]
            rise_slope = self._avg(rise)
        if peak_idx < len(trans) - 1:
            fall = [float(trans[i].get("intensity", 0.5)) - float(trans[i + 1].get("intensity", 0.5))
                    for i in range(peak_idx, len(trans) - 1)]
            fall_slope = self._avg(fall)

        # oscillation index: 연속 전이에서 to_emotion 극성 반전 비율
        if len(trans) >= 2:
            flips = 0
            for i in range(1, len(trans)):
                a = self.polarity_map.get(self._cat_of(trans[i - 1]["to_emotion"]) or trans[i - 1]["to_emotion"], "neutral")
                b = self.polarity_map.get(self._cat_of(trans[i]["to_emotion"]) or trans[i]["to_emotion"], "neutral")
                if a != "neutral" and b != "neutral" and a != b:
                    flips += 1
            osc = flips / (len(trans) - 1)
        else:
            osc = 0.0

        return {
            "development_duration": dev_dur,
            "aftermath_duration": aft_dur,
            "peak_intensity": float(trans[peak_idx].get("intensity", 0.0)),
            "rise_slope": round(rise_slope, 3),
            "fall_slope": round(fall_slope, 3),
            "oscillation_index": round(osc, 3),
        }

    # -------------------------------------------------------------- small utils
    def _has_marker(self, text: str, groups: Tuple[str, ...]) -> bool:
        for g in groups:
            regs = self.marker_bank.get(g, []) if isinstance(self.marker_bank, dict) else []
            for r in regs:
                try:
                    if r.search(text):
                        return True
                except Exception:
                    continue
        return False

    def _stage_confidence(self, t: Dict[str, Any], stage: str) -> float:
        """
        stage_confidence = 0.5*conf + 0.5*(intensity/peak_thr) + marker_bonus
        - marker_bonus: 0.08 (config의 development_ctx_boost가 더 크면 그 값을 사용)
        - 0~1 클립
        """
        inten = float(t.get("intensity", 0.5))
        conf  = float(t.get("confidence", 0.0))
        txt = (t.get("trigger_text", "") + " " + t.get("prev_context", "")).strip()

        marker_bonus = 0.0
        base_bonus = min(0.08, self.ctx_dev_boost)  # 0.08 기본, 설정이 더 크면 그 값을 사용

        if stage in {"Development", "Peak"}:
            if self._has_marker(txt, groups=("intensifying", "gradual")):
                marker_bonus += base_bonus
        if stage == "Aftermath":
            if self._has_marker(txt, groups=("attenuating", "gradual")):
                marker_bonus += base_bonus

        val = (0.5 * conf) + (0.5 * min(1.0, inten / max(1e-6, self.peak_thr))) + marker_bonus
        return max(0.0, min(1.0, val))

    def _avg(self, arr: List[float]) -> float:
        return sum(arr) / len(arr) if arr else 0.0

    def _bregex(self, text: str) -> re.Pattern:
        if not isinstance(text, str) or not text.strip():
            return re.compile(r"$^")
        key = rf"(?<!\w){re.escape(text.strip())}(?!\w)"
        try:
            return re.compile(key, re.IGNORECASE)
        except re.error:
            return re.compile(re.escape(text), re.IGNORECASE)

    def _cat_of(self, key: str) -> Optional[str]:
        """
        대표/세부 감정 키를 대표 카테고리('희','노','애','락')로 매핑.
        """
        if key in {"희", "노", "애", "락"}:
            return key
        if key in self.category_map:
            return self.category_map[key]
        ed = (self.emotions_data or {}).get(key, {})
        if isinstance(ed, dict):
            md = ed.get("metadata", {})
            if isinstance(md, dict):
                cat = md.get("primary_category")
                if isinstance(cat, str) and cat in {"희", "노", "애", "락"}:
                    return cat
        return None


# =============================================================================
# RelationshipAnalyzer (Refined)
# =============================================================================
# 중복된 _clip01 정의 제거: 모듈 상단의 _clip01을 사용합니다.

class RelationshipAnalyzer:
    """
    감정 간 상호 관계/호환성 분석기(옵션).
    - TransitionAnalyzer가 산출한 transitions 리스트(문장 전이) 기반
    - EMOTIONS.json을 순회해 synergy/conflict/priors를 자동 구축(규칙1/2/3)
    - 기존 인터페이스/반환 스키마는 유지
    """

    def __init__(
        self,
        threshold: float = 0.65,
        config: Optional[Dict[str, Any]] = None,
        emotions_data: Optional[Dict[str, Any]] = None,
        transition_map: Optional[Dict[Tuple[str, str], Dict[str, Any]]] = None,
    ):
        self.threshold = float(threshold)
        self.config = config or {}
        self.emotions_data = emotions_data or {}
        self.transition_map = transition_map or {}

        # 가중치(필요 시 config에서 오버라이드)
        W = self.config.get("REL_WEIGHTS", {})
        self.w_synergy    = float(W.get("synergy",    0.45))
        self.w_conflict   = float(W.get("conflict",   0.45))
        self.w_valence    = float(W.get("valence",    0.25))
        self.w_seq        = float(W.get("sequence",   0.25))
        self.w_intensity  = float(W.get("intensity",  0.20))
        self.w_confidence = float(W.get("confidence", 0.15))

        # 데이터 주도 맵 구축
        self.category_map = self._build_category_map()
        self.polarity_map = self._build_polarity_map()
        self.synergy_map, self.conflict_map = self._build_relation_maps_from_json()
        self.progression_priors = self._build_progression_priors()

        # 수치화용 극성값
        self.valence_num = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}

    # ------------------------------------------------------------------ public
    def analyze_relationships(self, transitions: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not transitions:
            return {
                "relationship_strength": 0.0,
                "compatibility_score": 0.0,
                "transition_patterns": [],
                "emotional_conflicts": [],
                "stability_metrics": {}
            }

        rel_scores: List[float] = []
        comp_scores: List[float] = []
        patterns: List[Dict[str, Any]] = []
        conflicts: List[Dict[str, Any]] = []

        # 간단 속도(포지션 간격 기반; 0~1): 간격 좁을수록 빠름
        speed = self._estimate_speed(transitions)

        for t in transitions:
            fe, te = t.get("from_emotion"), t.get("to_emotion")
            if not (fe and te):
                continue
            intensity  = _clip01(t.get("intensity", 0.5))
            confidence = _clip01(t.get("confidence", 0.5))

            # 데이터 주도 특징치 계산
            s_score  = self._synergy_score(fe, te)             # 0~1
            c_score  = self._conflict_score(fe, te)            # 0~1
            v_score  = self._valence_compat_score(fe, te)      # 0~1
            seq_score= self._sequence_prior(fe, te)            # 0~1

            rel_strength = self._relationship_strength(
                synergy=s_score, conflict=c_score, valence=v_score, seq=seq_score,
                intensity=intensity, confidence=confidence
            )
            rel_scores.append(rel_strength)

            comp_score = self._compatibility_score(
                fe, te, confidence=confidence, synergy=s_score, conflict=c_score, valence=v_score, seq=seq_score
            )
            comp_scores.append(comp_score)

            p = self._analyze_transition_pattern(t, v_score)
            if p:
                patterns.append(p)

            cf = self._detect_emotional_conflict(fe, te, intensity=float(intensity), transition_speed=float(speed))
            if cf.get("has_conflict"):
                conflicts.append({
                    "from_emotion": fe,
                    "to_emotion": te,
                    "conflict_type": cf.get("conflict_type", "opposing_emotions"),
                    "position": t.get("position", 0),
                    "conflict_intensity": cf.get("conflict_intensity", 0.0),
                })

        avg_rel  = sum(rel_scores) / len(rel_scores) if rel_scores else 0.0
        avg_comp = sum(comp_scores) / len(comp_scores) if comp_scores else 0.0
        stability = self._calculate_stability_metrics(rel_scores, comp_scores, conflicts)

        return {
            "relationship_strength": round(_clip01(avg_rel), 3),
            "compatibility_score":   round(_clip01(avg_comp), 3),
            "transition_patterns":   patterns,
            "emotional_conflicts":   conflicts,
            "stability_metrics":     stability
        }

    # -------------------------------------------------------------- data-driven maps
    def _build_category_map(self) -> Dict[str, str]:
        cmap = {"희": "희", "노": "노", "애": "애", "락": "락"}
        for k, ed in (self.emotions_data or {}).items():
            if not isinstance(ed, dict):
                continue
            md = ed.get("metadata", {}) or {}
            cat = md.get("primary_category")
            if isinstance(cat, str) and cat in {"희", "노", "애", "락"}:
                cmap[k] = cat
            for sub_id, sub in (ed.get("sub_emotions", {}) or {}).items():
                md2 = sub.get("metadata", {}) or {}
                cat2 = md2.get("primary_category")
                if isinstance(cat2, str) and cat2 in {"희", "노", "애", "락"}:
                    cmap[sub_id] = cat2
        return cmap

    def _build_polarity_map(self) -> Dict[str, str]:
        base = {"희": "positive", "락": "positive", "노": "negative", "애": "negative"}
        pmap = dict(base)
        for k, ed in (self.emotions_data or {}).items():
            ml = ed.get("ml_training_metadata", {}) or {}
            pol = ml.get("polarity")
            md = ed.get("metadata", {}) or {}
            cat = md.get("primary_category")
            if isinstance(pol, str) and pol.lower() in {"positive", "negative", "neutral"}:
                pmap[k] = pol.lower()
            elif isinstance(cat, str) and cat in base:
                pmap[k] = base[cat]
        return pmap

    def _build_relation_maps_from_json(self) -> Tuple[Dict[str, set], Dict[str, set]]:
        """
        JSON의 시너지/충돌 정의를 집계:
        - ml_training_metadata.synergy_with / conflict_with
        - emotion_transitions.dependencies.relationship_type == 'enhances'/'conflicts'
        - emotion_profile.related_emotions.positive/negative
        결과: {emotion_or_cat: set([...])}
        """
        syn: Dict[str, set] = defaultdict(set)
        con: Dict[str, set] = defaultdict(set)
        try:
            for key, ed in (self.emotions_data or {}).items():
                if not isinstance(ed, dict):
                    continue
                # synergy/conflict with
                ml = ed.get("ml_training_metadata", {}) or {}
                for s in (ml.get("synergy_with", []) or []):
                    if isinstance(s, str):
                        syn[key].add(s)
                for c in (ml.get("conflict_with", []) or []):
                    if isinstance(c, str):
                        con[key].add(c)

                # dependencies
                deps = (ed.get("emotion_transitions", {}) or {}).get("emotion_dependencies", []) or []
                for d in deps:
                    if isinstance(d, dict):
                        t = d.get("dependent_emotion")
                        rel = d.get("relationship_type")
                        if isinstance(t, str):
                            if rel == "enhances":
                                syn[key].add(t)
                            elif rel == "conflicts":
                                con[key].add(t)

                # related_emotions
                prof = ed.get("emotion_profile", {}) or {}
                rels = prof.get("related_emotions", {}) or {}
                for t in (rels.get("positive", []) or []):
                    if isinstance(t, str):
                        syn[key].add(t)
                for t in (rels.get("negative", []) or []):
                    if isinstance(t, str):
                        con[key].add(t)

            # (선택) 대칭 보정: 한쪽만 명시되었을 때 약하게 반대방향도 참고(데이터가 편향일 때 안정화)
            for a, targets in list(syn.items()):
                for b in targets:
                    syn.setdefault(b, set())
            for a, targets in list(con.items()):
                for b in targets:
                    con.setdefault(b, set())
        except Exception as e:
            logger.error(f"[RelationshipAnalyzer] relation map build failed: {e}")
        return syn, con

    def _build_progression_priors(self) -> Dict[Tuple[str, str], float]:
        """
        emotion_transitions.patterns에서 from→to의 '자연스러운' 전이 prior를 계산.
        - from 별로 weight 합을 1.0로 정규화(분포 기반)
        - 데이터가 없으면 0.3의 안전한 기본값
        """
        priors_src: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        try:
            for _, ed in (self.emotions_data or {}).items():
                trs = (ed.get("emotion_transitions", {}) or {}).get("patterns", []) or []
                for p in trs:
                    frm, to = p.get("from_emotion"), p.get("to_emotion")
                    if frm and to:
                        priors_src[frm][to] += float(p.get("weight", 1.0))
        except Exception as e:
            logger.error(f"[RelationshipAnalyzer] progression priors build failed: {e}")

        # from 별 정규화
        priors: Dict[Tuple[str, str], float] = {}
        for frm, dist in priors_src.items():
            total = sum(v for v in dist.values())
            if total > 0:
                for to, v in dist.items():
                    priors[(frm, to)] = v / total
        return priors

    # -------------------------------------------------------------- core scoring
    def _synergy_score(self, a: str, b: str) -> float:
        """
        a와 b가 데이터 주도로 시너지인지(정방향/양방향 포함)
        """
        if b in self.synergy_map.get(a, set()) or a in self.synergy_map.get(b, set()):
            return 1.0
        # 카테고리 레벨 시너지(양의 극성 동일) 약한 보정
        pa = self.polarity_map.get(self._cat_of(a) or a, "neutral")
        pb = self.polarity_map.get(self._cat_of(b) or b, "neutral")
        if pa == "positive" and pb == "positive":
            return 0.6
        return 0.0

    def _conflict_score(self, a: str, b: str) -> float:
        """
        a와 b가 데이터 주도로 충돌인지(정/역방향 포함)
        """
        if b in self.conflict_map.get(a, set()) or a in self.conflict_map.get(b, set()):
            return 1.0
        pa = self.polarity_map.get(self._cat_of(a) or a, "neutral")
        pb = self.polarity_map.get(self._cat_of(b) or b, "neutral")
        if {"positive", "negative"} <= {pa, pb}:
            return 0.6
        return 0.0

    def _valence_compat_score(self, a: str, b: str) -> float:
        """
        극성 거리 기반 호환성(1.0=완전 호환, 0.0=정반대)
        """
        pa = self.polarity_map.get(self._cat_of(a) or a, "neutral")
        pb = self.polarity_map.get(self._cat_of(b) or b, "neutral")
        va = self.valence_num.get(pa, 0.0); vb = self.valence_num.get(pb, 0.0)
        diff = abs(va - vb)  # 0 ~ 2
        return _clip01(1.0 - diff / 2.0)

    def _sequence_prior(self, a: str, b: str) -> float:
        """
        JSON 전이 패턴에서 관측된 자연스러운 전이 prior(0~1).
        """
        return _clip01(self.progression_priors.get((a, b), 0.3))

    def _relationship_strength(
        self,
        synergy: float,
        conflict: float,
        valence: float,
        seq: float,
        intensity: float,
        confidence: float
    ) -> float:
        """
        관계 강도(0~1): 시너지↑, 충돌↓, 순행 prior↑, 극성 호환↑, 강도·신뢰도 가중.
        score = w_synergy*synergy - w_conflict*conflict + w_seq*prior + w_valence*compat + w_intensity*intensity + w_confidence*confidence
        """
        score = 0.0
        score += self.w_synergy    * _clip01(synergy)
        score -= self.w_conflict   * _clip01(conflict)
        score += self.w_seq        * _clip01(seq)
        score += self.w_valence    * _clip01(valence)
        score += self.w_intensity  * _clip01(intensity)
        score += self.w_confidence * _clip01(confidence)
        return _clip01(score)

    def _compatibility_score(
        self,
        a: str,
        b: str,
        confidence: float,
        synergy: float,
        conflict: float,
        valence: float,
        seq: float
    ) -> float:
        """
        호환성(0~1):
        base = 0.5*synergy + 0.3*prior + 0.2*valence
        base *= (0.6 + 0.4*confidence)
        base *= (1 - 0.5*conflict)
        """
        base = 0.0
        base += 0.5 * _clip01(synergy)
        base += 0.3 * _clip01(seq)
        base += 0.2 * _clip01(valence)
        base *= (0.6 + 0.4 * _clip01(confidence))
        base *= (1.0 - 0.5 * _clip01(conflict))
        return _clip01(base)

    # -------------------------------------------------------------- pattern / conflict
    def _analyze_transition_pattern(self, transition: Dict[str, Any], v_score: float) -> Dict[str, Any]:
        fe = transition.get("from_emotion"); te = transition.get("to_emotion")
        inten = _clip01(transition.get("intensity", 0.5))
        conf  = _clip01(transition.get("confidence", 0.5))
        pos   = int(transition.get("position", 0))

        pt = "unknown"
        pa = self.polarity_map.get(self._cat_of(fe) or fe, "neutral")
        pb = self.polarity_map.get(self._cat_of(te) or te, "neutral")
        if pa == "negative" and pb == "positive":
            pt = "recovery"
        elif pa == "positive" and pb == "negative":
            pt = "downturn"
        elif pa == pb == "positive":
            pt = "reinforcement" if inten >= 0.6 else "positive_shift"
        elif pa == pb == "negative":
            pt = "escalation" if inten >= 0.6 else "negative_shift"

        return {
            "pattern_type": pt,
            "predictability_score": round(self._sequence_prior(fe, te), 3),
            "pattern_strength": round((inten + conf) / 2.0, 3),
            "recurring_elements": (["high_intensity"] if inten >= 0.7 else []),
            "pattern_metrics": {
                "position": pos,
                "intensity": float(inten),
                "confidence": float(conf),
                "valence_compat": round(_clip01(v_score), 3),
            },
        }

    def _detect_emotional_conflict(
        self,
        from_emotion: str,
        to_emotion: str,
        intensity: float = 0.5,
        transition_speed: float = 0.5
    ) -> Dict[str, Any]:
        """
        데이터/극성 기반 충돌 탐지. 기존 스키마와 호환되는 dict 반환.
        """
        res = {
            "has_conflict": False,
            "conflict_type": None,
            "conflict_intensity": 0.0,
            "psychological_strain": 0.0,
            "resolution_difficulty": 0.0,
            "conflict_factors": []
        }

        # 명시적 충돌
        if (to_emotion in self.conflict_map.get(from_emotion, set())) or (from_emotion in self.conflict_map.get(to_emotion, set())):
            res["has_conflict"] = True
            res["conflict_type"] = "explicit_conflict"

        # 극성 반전
        pa = self.polarity_map.get(self._cat_of(from_emotion) or from_emotion, "neutral")
        pb = self.polarity_map.get(self._cat_of(to_emotion) or to_emotion, "neutral")
        if {"positive", "negative"} <= {pa, pb} and pa != pb:
            res["has_conflict"] = True
            if not res["conflict_type"]:
                res["conflict_type"] = "opposing_emotions"

        if res["has_conflict"]:
            pol_diff = abs(self.valence_num.get(pa, 0.0) - self.valence_num.get(pb, 0.0))  # 0~2
            # 0~1로 정규화
            pol_unit = pol_diff / 2.0
            ci = _clip01(0.5 * _clip01(intensity) + 0.25 * pol_unit + 0.25 * _clip01(transition_speed))
            ps = _clip01(0.4 * _clip01(transition_speed) + 0.3 * _clip01(intensity) + 0.3 * ci)
            res.update({
                "conflict_intensity": round(ci, 3),
                "psychological_strain": round(ps, 3),
                "resolution_difficulty": round(_clip01((ci + ps) / 2.0), 3),
            })
            factors = []
            if pol_unit >= 0.6: factors.append("extreme_polarity_shift")
            if intensity >= 0.7: factors.append("high_intensity")
            if transition_speed >= 0.7: factors.append("rapid_transition")
            res["conflict_factors"] = factors

        return res

    # -------------------------------------------------------------- stability
    def _calculate_stability_metrics(
        self,
        relationship_scores: List[float],
        compatibility_scores: List[float],
        emotional_conflicts: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        if not relationship_scores or not compatibility_scores:
            return {
                "overall_stability": 0.0,
                "volatility_index": 0.0,
                "sustainability_score": 0.0,
                "conflict_impact": 0.0,
                "stability_factors": []
            }

        avg_rel  = sum(relationship_scores) / len(relationship_scores)
        avg_comp = sum(compatibility_scores) / len(compatibility_scores)
        var_rel  = sum((x - avg_rel)  ** 2 for x in relationship_scores) / len(relationship_scores)
        var_comp = sum((x - avg_comp) ** 2 for x in compatibility_scores) / len(compatibility_scores)

        stability_base   = (avg_rel + avg_comp) / 2.0
        volatility_base  = (var_rel + var_comp) / 2.0
        conflict_count   = len(emotional_conflicts)
        conflict_impact  = min(1.0, conflict_count * 0.1)

        # 최장 연속 안정 구간(관계/호환 모두 threshold 이상)
        stable_thr = self.threshold
        cons, cons_max = 0, 0
        for r, c in zip(relationship_scores, compatibility_scores):
            if r >= stable_thr and c >= stable_thr:
                cons += 1; cons_max = max(cons_max, cons)
            else:
                cons = 0
        sustainability = cons_max / len(relationship_scores)

        factors = []
        if avg_rel > 0.7:  factors.append("strong_relationships")
        if avg_comp > 0.7: factors.append("high_compatibility")
        if volatility_base < 0.3: factors.append("low_volatility")
        if sustainability > 0.6:  factors.append("sustained_stability")
        if conflict_count > 0:    factors.append("presence_of_conflicts")

        trend = self._analyze_stability_trend(relationship_scores, compatibility_scores)

        overall = (
            stability_base * 0.4 +
            (1 - volatility_base) * 0.3 +
            sustainability * 0.2 +
            (1 - conflict_impact) * 0.1
        )
        return {
            "overall_stability":   round(_clip01(overall), 3),
            "volatility_index":    round(_clip01(volatility_base), 3),
            "sustainability_score":round(_clip01(sustainability), 3),
            "conflict_impact":     round(_clip01(conflict_impact), 3),
            "stability_factors":   factors,
            "trend_analysis":      trend,
            "detailed_metrics": {
                "relationship_stability":  round(_clip01(avg_rel), 3),
                "compatibility_stability": round(_clip01(avg_comp), 3),
                "relationship_variance":   round(_clip01(var_rel), 3),
                "compatibility_variance":  round(_clip01(var_comp), 3),
                "max_stable_sequence":     cons_max,
                "conflict_count":          conflict_count
            }
        }

    def _analyze_stability_trend(
        self,
        relationship_scores: List[float],
        compatibility_scores: List[float]
    ) -> Dict[str, Any]:
        if len(relationship_scores) < 2:
            return {"trend_type": "insufficient_data", "trend_strength": 0.0, "trend_confidence": 0.0}
        r_diff = [relationship_scores[i] - relationship_scores[i - 1] for i in range(1, len(relationship_scores))]
        c_diff = [compatibility_scores[i] - compatibility_scores[i - 1] for i in range(1, len(compatibility_scores))]
        avg_r, avg_c = sum(r_diff) / len(r_diff), sum(c_diff) / len(c_diff)
        trend_strength = (abs(avg_r) + abs(avg_c)) / 2.0
        if trend_strength < 0.1:
            ttype = "stable"
        elif avg_r > 0 and avg_c > 0:
            ttype = "improving"
        elif avg_r < 0 and avg_c < 0:
            ttype = "deteriorating"
        else:
            ttype = "mixed"
        consist = 1.0 - ((max(r_diff) - min(r_diff)) + (max(c_diff) - min(c_diff))) / 4.0
        return {
            "trend_type": ttype,
            "trend_strength": round(_clip01(trend_strength), 3),
            "trend_confidence": round(_clip01(consist), 3),
            "relationship_trend": round(avg_r, 3),
            "compatibility_trend": round(avg_c, 3)
        }

    # -------------------------------------------------------------- helpers
    def _cat_of(self, key: str) -> Optional[str]:
        if key in {"희", "노", "애", "락"}:
            return key
        return self.category_map.get(key)

    def _estimate_speed(self, transitions: List[Dict[str, Any]]) -> float:
        """
        전이 포지션 간 간격을 이용한 간단 속도 추정(0~1): 간격이 좁을수록 빠름.
        """
        if len(transitions) < 2:
            return 0.5
        poss = sorted([int(t.get("position", i)) for i, t in enumerate(transitions)])
        gaps = [max(1, poss[i] - poss[i - 1]) for i in range(1, len(poss))]
        inv = [1.0 / g for g in gaps]
        mx = max(inv) if inv else 1.0
        return _clip01(sum(inv) / (len(inv) * mx)) if mx > 0 else 0.5


# =============================================================================
# ComplexTransitionAnalyzer (Refined)
# =============================================================================
# 중복된 _clip01 정의 제거: 모듈 상단의 _clip01을 사용합니다.

class ComplexTransitionAnalyzer:
    """
    복합 전이 패턴(다중 감정 교차/연쇄)을 식별하는 분석기.
    - 입력: TransitionAnalyzer가 산출한 transitions(list of dict)
      각 항목: {from_emotion, to_emotion, intensity, confidence, position, trigger_text, prev_context, ...}
    - 출력: [{ pattern, complexity_factors, score, spans, evidence }, ...]
    - 규칙1/2/3: 모든 규칙·가중치는 가능한 한 EMOTIONS.json에서 유도, 4×N 확장 안전, 스키마 불변
    """

    def __init__(
        self,
        complexity_factors: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None,
        emotions_data: Optional[Dict[str, Any]] = None,
        compiled_marker_bank: Optional[Dict[str, List[re.Pattern]]] = None,
    ):
        self.config = config or {}
        self.emotions_data = emotions_data or {}
        self.complexity_factors = complexity_factors or ["oscillation", "conflict", "subtle"]

        # 탐지 민감도/가중치(필요시 config에서 오버라이드)
        W = self.config.get("COMPLEX_WEIGHTS", {})
        self.w_cycle = float(W.get("cycle", 0.35))
        self.w_osc   = float(W.get("oscillation", 0.35))
        self.w_conv  = float(W.get("convergence", 0.25))
        self.w_div   = float(W.get("divergence", 0.25))
        self.w_esc   = float(W.get("escalation", 0.30))
        self.w_att   = float(W.get("attenuation", 0.30))
        self.w_conf  = float(W.get("conflict_density", 0.35))
        self.w_syn   = float(W.get("synergy_density", 0.30))
        self.min_score = float(self.config.get("COMPLEX_MIN_SCORE", 0.35))

        # 임계값들
        T = self.config.get("COMPLEX_THRESHOLDS", {})
        # 요청사항: 기본 0.45로 완화
        self.thr_osc   = float(T.get("oscillation_index", 0.45))     # 긍↔부정 반전 비율
        self.thr_cycle = int(T.get("cycle_min_len", 3))              # 순환 최소 길이
        self.win_min   = int(T.get("window_min", 2))
        self.win_max   = int(T.get("window_max", 5))
        self.thr_conf_dense = float(T.get("conflict_density", 0.5))
        self.thr_syn_dense  = float(T.get("synergy_density", 0.5))
        self.thr_int_delta  = float(T.get("intensity_delta", 0.05))

        # JSON 주도 맵/마커 뱅크(규칙1/2)
        self.category_map = self._build_category_map()
        self.polarity_map = self._build_polarity_map()
        self.synergy_map, self.conflict_map = self._build_relation_maps_from_json()
        self.priors_map = self._build_progression_priors()
        self.marker_bank = compiled_marker_bank or self._build_marker_bank_from_json()

        # 수치화용 극성값
        self.valence_num = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}

    # ------------------------------------------------------------------ public
    def detect_complex_transitions(
        self,
        transitions: List[Dict[str, Any]],
        emotion_sequence: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        전이 리스트에서 복합 전이 패턴을 식별해 리턴.
        - 반환: [{pattern, complexity_factors, score, spans, evidence}, ...]
        """
        if not transitions:
            return []

        # 기본 시퀀스 준비(포지션 정렬)
        trans_sorted = sorted(transitions, key=lambda x: x.get("position", 0))
        to_seq  = [t.get("to_emotion") for t in trans_sorted]
        cats    = [self._cat_of(x) or x for x in to_seq]
        pols    = [self.polarity_map.get(c, "neutral") for c in cats]
        intens  = [float(t.get("intensity", 0.5)) for t in trans_sorted]
        confs   = [float(t.get("confidence", 0.0)) for t in trans_sorted]
        txts    = [(t.get("trigger_text", "") + " " + t.get("prev_context", "")).strip() for t in trans_sorted]

        patterns: List[Dict[str, Any]] = []

        # 1) 진동(oscillation): 인접 전이 간 극성 반전 비율
        osc_idx, osc_spans = self._oscillation_index(pols)
        if osc_idx >= self.thr_osc:
            score = self.w_osc * _clip01(osc_idx)
            patterns.append(self._make_pattern("oscillation", score, osc_spans, {
                "oscillation_index": round(osc_idx, 3),
                "flips": len(osc_spans)
            }))

        # 2) 순환(cycle): A→B→…→A (최소 길이 3)
        cycles = self._find_cycles(cats, min_len=self.thr_cycle)
        if cycles:
            cyc_score = self.w_cycle * _clip01(sum((b - a + 1) for a, b in cycles) / max(1, len(cats)))
            patterns.append(self._make_pattern("cyclic", cyc_score, cycles, {
                "cycle_count": len(cycles),
                "cycle_lengths": [int(b - a + 1) for a, b in cycles],
            }))

        # 3) 수렴(convergence): 연속 window에서 to 동일 & from 다양
        conv_spans, conv_evid = self._find_convergence(trans_sorted)
        if conv_spans:
            conv_score = self.w_conv * _clip01(len(conv_spans) / max(1, len(trans_sorted)))
            patterns.append(self._make_pattern("convergent", conv_score, conv_spans, conv_evid))

        # 4) 분기(divergence): 연속 window에서 from 동일 & to 다양
        div_spans, div_evid = self._find_divergence(trans_sorted)
        if div_spans:
            div_score = self.w_div * _clip01(len(div_spans) / max(1, len(trans_sorted)))
            patterns.append(self._make_pattern("divergent", div_score, div_spans, div_evid))

        # 5) 강도 단조 체인(에스컬레이션/감쇠)
        esc_spans, esc_evid = self._monotonic_chains(intens, direction="up")
        if esc_spans:
            esc_score = self.w_esc * _clip01(sum((b - a + 1) for a, b in esc_spans) / max(1, len(intens)))
            patterns.append(self._make_pattern("escalation_chain", esc_score, esc_spans, esc_evid))
        att_spans, att_evid = self._monotonic_chains(intens, direction="down")
        if att_spans:
            att_score = self.w_att * _clip01(sum((b - a + 1) for a, b in att_spans) / max(1, len(intens)))
            patterns.append(self._make_pattern("attenuation_chain", att_score, att_spans, att_evid))

        # 6) 충돌/시너지 밀도(window 기반) + 대표 구간
        conf_density, syn_density, span_conf, span_syn = self._relation_densities(trans_sorted)
        if conf_density >= self.thr_conf_dense:
            score = self.w_conf * _clip01(conf_density)
            patterns.append(self._make_pattern("conflict_dominant", score, span_conf, {
                "conflict_density": round(conf_density, 3)
            }))
        if syn_density >= self.thr_syn_dense:
            score = self.w_syn * _clip01(syn_density)
            patterns.append(self._make_pattern("synergy_cluster", score, span_syn, {
                "synergy_density": round(syn_density, 3)
            }))

        # 7) 교차 흐름(cross-current): 동일 구문에 intensifying & attenuating 동시
        cc_spans = self._cross_current_markers(txts)
        if cc_spans:
            score = 0.3 * _clip01(len(cc_spans) / max(1, len(txts)))
            patterns.append(self._make_pattern("cross_current", score, cc_spans, {
                "cross_points": len(cc_spans)
            }))

        # 8) prior-driven cluster: 연속 구간의 평균 prior ≥ 0.6
        pri_spans, pri_evid = self._prior_clusters(to_seq)
        if pri_spans:
            score = 0.25 * _clip01(len(pri_spans) / max(1, len(to_seq)))
            patterns.append(self._make_pattern("prior_driven_cluster", score, pri_spans, pri_evid))

        # 결과 정리: 최소점수 미만은 제외, 점수 desc 정렬
        out = [p for p in patterns if p.get("score", 0.0) >= self.min_score]
        out.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return out

    # -------------------------------------------------------------- detectors
    def _oscillation_index(self, pols: List[str]) -> Tuple[float, List[Tuple[int, int]]]:
        if len(pols) < 2:
            return 0.0, []
        flips, spans = 0, []
        for i in range(1, len(pols)):
            a, b = pols[i - 1], pols[i]
            if {"positive", "negative"} <= {a, b} and a != b:
                flips += 1
                spans.append((i - 1, i))
        return flips / (len(pols) - 1), spans

    def _find_cycles(self, cats: List[str], min_len: int = 3) -> List[Tuple[int, int]]:
        spans: List[Tuple[int, int]] = []
        last_seen: Dict[str, int] = {}
        for i, c in enumerate(cats):
            if c in last_seen:
                start = last_seen[c]
                if (i - start + 1) >= min_len:
                    spans.append((start, i))
            last_seen[c] = i
        return spans

    def _find_convergence(self, trans: List[Dict[str, Any]]) -> Tuple[List[Tuple[int, int]], Dict[str, Any]]:
        spans = []
        evid = {"segments": []}  # 각 span별 from 다양성/길이
        for k in range(self.win_min, min(self.win_max, len(trans)) + 1):
            for i in range(0, len(trans) - k + 1):
                seg = trans[i:i + k]
                tos = [t["to_emotion"] for t in seg]
                frs = [t["from_emotion"] for t in seg]
                if len(set(tos)) == 1 and len(set(frs)) >= 2:
                    spans.append((i, i + k - 1))
                    evid["segments"].append({
                        "span": (i, i + k - 1),
                        "from_variety": len(set(frs)),
                        "length": k
                    })
        return spans, evid

    def _find_divergence(self, trans: List[Dict[str, Any]]) -> Tuple[List[Tuple[int, int]], Dict[str, Any]]:
        spans = []
        evid = {"segments": []}
        for k in range(self.win_min, min(self.win_max, len(trans)) + 1):
            for i in range(0, len(trans) - k + 1):
                seg = trans[i:i + k]
                tos = [t["to_emotion"] for t in seg]
                frs = [t["from_emotion"] for t in seg]
                if len(set(frs)) == 1 and len(set(tos)) >= 2:
                    spans.append((i, i + k - 1))
                    evid["segments"].append({
                        "span": (i, i + k - 1),
                        "to_variety": len(set(tos)),
                        "length": k
                    })
        return spans, evid

    def _monotonic_chains(self, intens: List[float], direction: str = "up") -> Tuple[List[Tuple[int, int]], Dict[str, Any]]:
        """
        강도 단조 변화 체인(상승/하강) 탐지. Δ≥thr_int_delta
        """
        spans = []
        if len(intens) < 2:
            return spans, {"chains": []}
        start = 0
        chains_info = []
        for i in range(1, len(intens)):
            delta = intens[i] - intens[i - 1]
            good = (delta >= self.thr_int_delta) if direction == "up" else (delta <= -self.thr_int_delta)
            if not good:
                if i - 1 > start:
                    spans.append((start, i - 1))
                    chains_info.append({"span": (start, i - 1), "length": (i - 1 - start + 1)})
                start = i
        if len(intens) - 1 > start:
            spans.append((start, len(intens) - 1))
            chains_info.append({"span": (start, len(intens) - 1), "length": (len(intens) - 1 - start + 1)})
        return spans, {"chains": chains_info}

    def _relation_densities(self, trans: List[Dict[str, Any]]) -> Tuple[float, float, List[Tuple[int, int]], List[Tuple[int, int]]]:
        """
        윈도우 기반 충돌/시너지 밀도와 대표 구간(각각 별도)을 반환
        """
        best_span_conf: Tuple[int, int] = ()
        best_span_syn: Tuple[int, int] = ()
        best_conf, best_syn = 0.0, 0.0
        total_pair, conf_pairs = 0, 0
        total_pair_syn, syn_pairs = 0, 0

        for k in range(self.win_min, min(self.win_max, len(trans)) + 1):
            for i in range(0, len(trans) - k + 1):
                seg = trans[i:i + k]
                pairs = list(zip(seg[:-1], seg[1:]))
                if not pairs:
                    continue
                c_cnt, s_cnt = 0, 0
                for p in pairs:
                    a = p[0]["to_emotion"]; b = p[1]["to_emotion"]
                    if self._is_conflict(a, b):
                        c_cnt += 1
                    if self._is_synergy(a, b):
                        s_cnt += 1
                c_density = c_cnt / len(pairs)
                s_density = s_cnt / len(pairs)

                if c_density > best_conf:
                    best_conf = c_density; best_span_conf = (i, i + k - 1)
                if s_density > best_syn:
                    best_syn = s_density; best_span_syn = (i, i + k - 1)

                total_pair += len(pairs); conf_pairs += c_cnt
                total_pair_syn += len(pairs); syn_pairs += s_cnt

        c_total = (conf_pairs / total_pair) if total_pair else 0.0
        s_total = (syn_pairs / total_pair_syn) if total_pair_syn else 0.0
        return c_total, s_total, ([best_span_conf] if best_conf > 0 else []), ([best_span_syn] if best_syn > 0 else [])

    def _cross_current_markers(self, texts: List[str]) -> List[Tuple[int, int]]:
        """
        동일 구간에 intensifying/attenuating 마커가 동시에 존재 → 역행/교차 흐름 가능성
        """
        if not texts:
            return []
        spans = []
        for i, t in enumerate(texts):
            inc = self._has_marker(t, ("intensifying", "gradual"))
            dec = self._has_marker(t, ("attenuating",))
            if inc and dec:
                spans.append((i, i))
        return spans

    def _prior_clusters(self, to_seq: List[str]) -> Tuple[List[Tuple[int, int]], Dict[str, Any]]:
        """
        연속 구간의 평균 prior(자연스러운 전이 확률)가 높은 클러스터 탐지 (avg ≥ 0.6)
        """
        if len(to_seq) < 2:
            return [], {"segments": []}
        spans = []
        seg_info = []
        for k in range(self.win_min, min(self.win_max, len(to_seq)) + 1):
            for i in range(0, len(to_seq) - k + 1):
                seg = to_seq[i:i + k]
                pri = []
                for j in range(1, len(seg)):
                    pri.append(self.priors_map.get((seg[j - 1], seg[j]), 0.3))
                if pri:
                    avg_pri = sum(pri) / len(pri)
                    if avg_pri >= 0.6:
                        spans.append((i, i + k - 1))
                        seg_info.append({"span": (i, i + k - 1), "avg_prior": round(_clip01(avg_pri), 3)})
        return spans, {"segments": seg_info}

    # -------------------------------------------------------------- JSON-driven maps
    def _build_marker_bank_from_json(self) -> Dict[str, List[re.Pattern]]:
        bank = {"gradual": [], "sudden": [], "repetitive": [], "intensifying": [], "attenuating": []}
        try:
            for _, ed in (self.emotions_data or {}).items():
                ml = (ed.get("ml_training_metadata", {}) or {})
                for group, key in [("gradual", "gradual_markers"), ("sudden", "sudden_markers"),
                                   ("repetitive", "repetitive_markers"), ("intensifying", "intensifying_markers"),
                                   ("attenuating", "attenuating_markers")]:
                    for m in (ml.get(key, []) or []):
                        if isinstance(m, str) and m.strip():
                            bank[group].append(self._bregex(m))
                trs = (ed.get("emotion_transitions", {}) or {}).get("patterns", []) or []
                for p in trs:
                    an = (p.get("transition_analysis", {}) or {})
                    for w in (an.get("trigger_words", []) or []):
                        if isinstance(w, str) and w.strip():
                            bank["intensifying"].append(self._bregex(w))
        except Exception as e:
            logger.error(f"[ComplexTransitionAnalyzer] marker bank build failed: {e}")
        return bank

    def _build_category_map(self) -> Dict[str, str]:
        cmap = {"희": "희", "노": "노", "애": "애", "락": "락"}
        for k, ed in (self.emotions_data or {}).items():
            md = (ed.get("metadata", {}) or {})
            cat = md.get("primary_category")
            if isinstance(cat, str) and cat in cmap:
                cmap[k] = cat
            for sub_k, sub in (ed.get("sub_emotions", {}) or {}).items():
                md2 = (sub.get("metadata", {}) or {})
                cat2 = md2.get("primary_category")
                if isinstance(cat2, str) and cat2 in cmap:
                    cmap[sub_k] = cat2
        return cmap

    def _build_polarity_map(self) -> Dict[str, str]:
        base = {"희": "positive", "락": "positive", "노": "negative", "애": "negative"}
        pmap = dict(base)
        for k, ed in (self.emotions_data or {}).items():
            ml = (ed.get("ml_training_metadata", {}) or {})
            pol = ml.get("polarity")
            md = (ed.get("metadata", {}) or {})
            cat = md.get("primary_category")
            if isinstance(pol, str) and pol.lower() in {"positive", "negative", "neutral"}:
                pmap[k] = pol.lower()
            elif isinstance(cat, str) and cat in base:
                pmap[k] = base[cat]
        return pmap

    def _build_relation_maps_from_json(self) -> Tuple[Dict[str, set], Dict[str, set]]:
        syn: Dict[str, set] = defaultdict(set)
        con: Dict[str, set] = defaultdict(set)
        try:
            for key, ed in (self.emotions_data or {}).items():
                ml = (ed.get("ml_training_metadata", {}) or {})
                for s in (ml.get("synergy_with", []) or []):
                    if isinstance(s, str): syn[key].add(s)
                for c in (ml.get("conflict_with", []) or []):
                    if isinstance(c, str): con[key].add(c)
                deps = (ed.get("emotion_transitions", {}) or {}).get("emotion_dependencies", []) or []
                for d in deps:
                    if isinstance(d, dict):
                        t = d.get("dependent_emotion"); r = d.get("relationship_type")
                        if isinstance(t, str) and r == "enhances": syn[key].add(t)
                        if isinstance(t, str) and r == "conflicts": con[key].add(t)
                rels = (ed.get("emotion_profile", {}) or {}).get("related_emotions", {}) or {}
                for t in (rels.get("positive", []) or []):
                    if isinstance(t, str): syn[key].add(t)
                for t in (rels.get("negative", []) or []):
                    if isinstance(t, str): con[key].add(t)
        except Exception as e:
            logger.error(f"[ComplexTransitionAnalyzer] relation map build failed: {e}")
        return syn, con

    def _build_progression_priors(self) -> Dict[Tuple[str, str], float]:
        pri: Dict[Tuple[str, str], float] = defaultdict(float)
        try:
            for _, ed in (self.emotions_data or {}).items():
                for p in (ed.get("emotion_transitions", {}) or {}).get("patterns", []) or []:
                    a = p.get("from_emotion"); b = p.get("to_emotion")
                    if a and b:
                        pri[(a, b)] += float(p.get("weight", 1.0))
        except Exception as e:
            logger.error(f"[ComplexTransitionAnalyzer] priors build failed: {e}")
        if pri:
            mx = max(pri.values())
            if mx > 0:
                for k in list(pri.keys()):
                    pri[k] = pri[k] / mx
        return pri

    # -------------------------------------------------------------- helpers
    def _is_conflict(self, a: str, b: str) -> bool:
        if b in self.conflict_map.get(a, set()) or a in self.conflict_map.get(b, set()):
            return True
        pa = self.polarity_map.get(self._cat_of(a) or a, "neutral")
        pb = self.polarity_map.get(self._cat_of(b) or b, "neutral")
        return {"positive", "negative"} <= {pa, pb} and pa != pb

    def _is_synergy(self, a: str, b: str) -> bool:
        if b in self.synergy_map.get(a, set()) or a in self.synergy_map.get(b, set()):
            return True
        pa = self.polarity_map.get(self._cat_of(a) or a, "neutral")
        pb = self.polarity_map.get(self._cat_of(b) or b, "neutral")
        return (pa == "positive" and pb == "positive")

    def _has_marker(self, text: str, groups: Tuple[str, ...]) -> bool:
        for g in groups:
            regs = self.marker_bank.get(g, []) if isinstance(self.marker_bank, dict) else []
            for r in regs:
                try:
                    if r.search(text):
                        return True
                except Exception:
                    continue
        return False

    def _make_pattern(self, name: str, score: float, spans: List[Tuple[int, int]], extra: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "pattern": name,
            "complexity_factors": self.complexity_factors,
            "score": round(_clip01(score), 3),
            "spans": spans,
            "evidence": extra
        }

    def _cat_of(self, key: str) -> Optional[str]:
        if key in {"희", "노", "애", "락"}:
            return key
        return self.category_map.get(key)

    def _bregex(self, text: str) -> re.Pattern:
        if not isinstance(text, str) or not text.strip():
            return re.compile(r"$^")
        pat = rf"(?<!\w){re.escape(text.strip())}(?!\w)"
        try:
            return re.compile(pat, re.IGNORECASE)
        except re.error:
            return re.compile(re.escape(text), re.IGNORECASE)


# =============================================================================
# Independent Functions (public API)
# =============================================================================
class TransitionComponents(NamedTuple):
    analyzer: TransitionAnalyzer
    progression: EmotionProgressionAnalyzer
    relation: RelationshipAnalyzer
    complexer: ComplexTransitionAnalyzer


def _safe_load_emotions_data(emotions_data: Optional[Dict[str, Any]], config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """emotions_data가 없으면 _locate_emotions_file()을 통해 안전하게 로드."""
    if isinstance(emotions_data, dict) and emotions_data:
        return emotions_data
    path = _locate_emotions_file()
    if not path or not os.path.exists(path):
        raise FileNotFoundError("EMOTIONS.json 경로를 찾을 수 없습니다.")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or not data:
        raise ValueError("EMOTIONS.json 로드 실패 또는 비어 있음")
    return data


def build_transition_components(
    config: Optional[Dict[str, Any]] = None,
    emotions_data: Optional[Dict[str, Any]] = None
) -> TransitionComponents:
    """
    Transition/Progression/Relationship/Complex 분석기에 공용 설정과 감정 데이터를 주입해
    재사용 가능한 컴포넌트 세트를 생성.
    """
    cfg = config or _load_config_overlay()
    ed  = _safe_load_emotions_data(emotions_data, cfg)

    analyzer = TransitionAnalyzer(config=cfg, emotions_data=ed)
    progression = EmotionProgressionAnalyzer(
        config=cfg.get("TRANSITION_ANALYZER_CONFIG", {}), emotions_data=ed
    )
    relation = RelationshipAnalyzer(
        config=cfg.get("TRANSITION_ANALYZER_CONFIG", {}), emotions_data=ed,
        transition_map=analyzer.transition_map
    )
    complexer = ComplexTransitionAnalyzer(
        config=cfg.get("TRANSITION_ANALYZER_CONFIG", {}), emotions_data=ed
    )
    return TransitionComponents(analyzer, progression, relation, complexer)


def run_transition_analysis(
    text: str,
    emotions_data: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
    return_components: bool = False
) -> Dict[str, Any]:
    """
    end-to-end 파이프라인: 문장 분할 → 감정 상태 → 전이 기록 → 흐름 요약 → 진행 단계 → 관계/호환 → 복합 패턴.
    - return_components=True 로 주면 재사용을 위해 컴포넌트도 함께 반환(dict['__components__'])
    """
    if not isinstance(text, str) or not text.strip():
        return {
            "transitions": [],
            "triggers": [],
            "patterns": {},
            "flow_analysis": {},
            "emotion_progression": {},
            "relationships": {},
            "complex_patterns": [],
            "metadata": {"status": "empty_text"}
        }

    start = time.time()
    try:
        comps = build_transition_components(config=config, emotions_data=emotions_data)
        analyzer, progression, relation, complexer = comps.analyzer, comps.progression, comps.relation, comps.complexer

        # 1) 전이 분석(핵심)
        base = analyzer.analyze_emotion_transitions(text=text, emotions_data=analyzer.emotions_data)

        transitions = base.get("transitions", [])
        # 2) 진행 단계
        prog = progression.analyze_progression(transitions) if transitions else {"trigger": {}, "development": [], "peak": {}, "aftermath": [], "progression_metrics": {}}
        # 3) 관계/호환성
        rel  = relation.analyze_relationships(transitions) if transitions else {"relationship_strength": 0.0, "compatibility_score": 0.0, "transition_patterns": [], "emotional_conflicts": [], "stability_metrics": {}}
        # 4) 복합 패턴
        cx   = complexer.detect_complex_transitions(transitions) if transitions else []

        out = {
            "transitions": transitions,
            "triggers": base.get("triggers", []),
            "patterns": base.get("patterns", {}),
            "flow_analysis": base.get("flow_analysis", {}),
            "emotion_progression": prog,
            "relationships": rel,
            "complex_patterns": cx,
            "metrics": base.get("metrics", {}),
            "metadata": {
                "processing_time": round(time.time() - start, 3),
                "text_length": len(text),
                "transition_count": len(transitions),
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        if return_components:
            out["__components__"] = {
                "analyzer": analyzer,
                "progression": progression,
                "relation": relation,
                "complexer": complexer,
            }
        return out
    except Exception as e:
        logger.exception("[run_transition_analysis] 오류")
        return {
            "error": str(e),
            "error_type": type(e).__name__,
            "transitions": [],
            "metadata": {"status": "error", "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')}
        }


def run_basic_transition_analysis(
    text: str,
    emotions_data: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    경량 분석: 전이/흐름/메트릭만 필요할 때.
    """
    res = run_transition_analysis(text=text, emotions_data=emotions_data, config=config, return_components=False)
    # 경량 결과만 추려서 반환
    return {
        "transitions": res.get("transitions", []),
        "flow_analysis": res.get("flow_analysis", {}),
        "metrics": res.get("metrics", {}),
        "metadata": res.get("metadata", {})
    }


def analyze_emotion_transitions(
    text: str,
    emotions_data: Optional[Dict[str, Any]] = None,
    context: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    과거 함수명 유지용 래퍼.
    - 내부적으로 run_transition_analysis 호출
    - context는 현재 파이프라인에서 사용하지 않지만 시그니처 호환성 위해 유지
    """
    _ = context  # 미사용
    return run_transition_analysis(text=text, emotions_data=emotions_data, config=config, return_components=False)


def analyze_corpus_transitions(
    docs: List[Dict[str, Any]],
    emotions_data: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    여러 문서(text id/내용) 일괄 처리.
    docs 예시: [{"id":"doc1", "text":"..."}, ...]
    공용 컴포넌트를 재사용해 비용 절감 및 일관성 유지.
    """
    if not docs:
      return {"results": [], "metadata": {"status": "empty_docs"}}

    start = time.time()
    comps = build_transition_components(config=config, emotions_data=emotions_data)
    analyzer, progression, relation, complexer = comps.analyzer, comps.progression, comps.relation, comps.complexer

    results = []
    for i, d in enumerate(docs, 1):
        text = (d.get("text") or "").strip()
        doc_id = d.get("id", f"doc_{i}")
        try:
            base = analyzer.analyze_emotion_transitions(text=text, emotions_data=analyzer.emotions_data)
            transitions = base.get("transitions", [])
            prog = progression.analyze_progression(transitions) if transitions else {}
            rel  = relation.analyze_relationships(transitions) if transitions else {}
            cx   = complexer.detect_complex_transitions(transitions) if transitions else []
            results.append({
                "id": doc_id,
                "transitions": transitions,
                "flow_analysis": base.get("flow_analysis", {}),
                "emotion_progression": prog,
                "relationships": rel,
                "complex_patterns": cx,
                "metrics": base.get("metrics", {}),
            })
        except Exception as e:
            logger.exception(f"[analyze_corpus_transitions] '{doc_id}' 오류")
            results.append({
                "id": doc_id,
                "error": str(e),
                "transitions": [],
            })

    return {
        "results": results,
        "metadata": {
            "doc_count": len(results),
            "processing_time": round(time.time() - start, 3)
        }
    }



# =============================================================================
# Main (Refined) — Transition & Flow Engine
# =============================================================================
def _build_default_config() -> Dict[str, Any]:
    return {
        "TRANSITION_ANALYZER_CONFIG": {
            "parallel": {"max_workers": 4},
            "thresholds": {
                # 기본 하한 완화: 전이 신뢰도 0.10
                "min_confidence": 0.10,
                "min_conf_for_stage": 0.30,
            },
            "stages": {
                "peak_intensity": 0.70,
                "development_window": 2,
                "development_ctx_boost": 0.15,   # 0.10 → 0.15
                "aftermath_decay": 0.15,
            },
            "flow": {
                "volatile_ratio": 0.70,
                "complex_unique_min": 4,
            },
            "memory": {
                "hard_limit": 0.85,          # 시스템 메모리 85%
                "use_vms": False,
                "max_history_size": 2048,
                "cache_ttl_seconds": 900,
            },
            # 한국어 일반 맥락 바이어스(오버레이 실패 대비 기본값)
            "context_bias_positive_terms": [
                "기쁨","기쁘","행복","즐겁","편안","안정","안도","만족","성취","설렘","설레"
            ],
            "context_bias_negative_terms": [
                "불만","짜증","분노","화가","슬픔","슬프","우울","불안","걱정","피곤","힘들","지침","아프","아픔"
            ],
            "COMPLEX_WEIGHTS": {
                "cycle": 0.35, "oscillation": 0.35,
                "convergence": 0.25, "divergence": 0.25,
                "escalation": 0.30, "attenuation": 0.30,
                "conflict_density": 0.35, "synergy_density": 0.30,
            },
            "COMPLEX_THRESHOLDS": {
                "oscillation_index": 0.45,  # 0.50 → 0.45
                "cycle_min_len": 3,
                "window_min": 2,
                "window_max": 5,
                "conflict_density": 0.5,
                "synergy_density": 0.5,
                "intensity_delta": 0.05,
            },
        }
    }

def _load_config_overlay() -> Dict[str, Any]:
    cfg = _build_default_config()
    try:
        # config.py의 ANALYZER_CONFIG가 있으면 얹기
        import config as _cfg
        if hasattr(_cfg, "ANALYZER_CONFIG") and isinstance(_cfg.ANALYZER_CONFIG, dict):
            # TRANSITION_ANALYZER_CONFIG 섹션만 덮어쓰기/머지
            src = _cfg.ANALYZER_CONFIG
            tac = src.get("TRANSITION_ANALYZER_CONFIG", {})
            if isinstance(tac, dict) and tac:
                merged = cfg["TRANSITION_ANALYZER_CONFIG"]
                merged.update(tac)
                cfg["TRANSITION_ANALYZER_CONFIG"] = merged
    except Exception as e:
        logging.getLogger(__name__).warning(f"[Main] config overlay 실패: {e}")
    return cfg

def _locate_emotions_file() -> Optional[str]:
    """
    우선순위: config → ENV → 글로벌 상수 → (현재 파일 기준) ../EMOTIONS.json → (프로젝트 루트 추정) ../../EMOTIONS.json
    """
    path = None
    try:
        import config as _cfg
        if hasattr(_cfg, "ANALYZER_CONFIG"):
            path = _cfg.ANALYZER_CONFIG.get("emotions_file")
    except Exception:
        pass
    if not path:
        path = os.environ.get("EMOTIONS_FILE_PATH")
    if not path:
        path = globals().get("EMOTIONS_FILE_PATH", None)
    if path and os.path.exists(path):
        return path
    # 상대 경로 후보들
    here = os.path.abspath(os.path.dirname(__file__))
    candidates = [
        os.path.join(here, "..", "..", "EMOTIONS.json"),
        os.path.join(here, "..", "EMOTIONS.json"),
    ]
    for c in candidates:
        c = os.path.abspath(c)
        if os.path.exists(c):
            return c
    return None

def _setup_logger(log_dir: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("transition_main")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        fh = logging.FileHandler(os.path.join(log_dir, "transition_analyzer.log"), encoding="utf-8")
        fh.setLevel(logging.INFO); fh.setFormatter(fmt)
        sh = logging.StreamHandler(); sh.setLevel(logging.INFO); sh.setFormatter(fmt)
        logger.addHandler(fh); logger.addHandler(sh)
    return logger

def _sanitize_filename(name: str) -> str:
    for ch in [':', '*', '?', '"', '<', '>', '|', '\\', '/']:
        name = name.replace(ch, '_')
    return name.strip() or "untitled"

def main():
    # 통합 로그 관리자 사용 (날짜별 폴더)
    try:
        from log_manager import get_log_manager
        log_manager = get_log_manager()
        LOGS_DIR = str(log_manager.get_log_dir("emotion_analysis", use_date_folder=True) / "transition_analyzer")
    except ImportError:
        # 폴백: 기존 방식 (날짜별 폴더 추가)
        from datetime import datetime
        base_logs_dir = os.path.join("logs")
        today = datetime.now().strftime("%Y%m%d")
        LOGS_DIR = os.path.join(base_logs_dir, today, "transition_analyzer")
    logger = _setup_logger(LOGS_DIR)
    logger.info("전이·흐름 분석기 메인 시작")

    # 1) 설정/라벨 로드
    cfg = _load_config_overlay()
    emotions_path = _locate_emotions_file()
    if not emotions_path:
        logger.error("EMOTIONS.json 경로를 찾을 수 없습니다.")
        print("[오류] EMOTIONS.json 경로 확인 필요")
        return

    try:
        with open(emotions_path, "r", encoding="utf-8") as f:
            emotions_data = json.load(f)
        if not isinstance(emotions_data, dict) or not emotions_data:
            raise ValueError("감정 데이터(emotions_data)가 비었습니다.")
        logger.info(f"감정 데이터 로드 완료: {len(emotions_data)} top-level")
    except Exception as e:
        logger.exception(f"감정 데이터 로드 실패: {e}")
        print(f"[오류] 감정 데이터 로드 실패: {e}")
        return

    # 2) 테스트 케이스
    test_cases = [
        {
            "title": "전이_기쁨에서_불만으로",
            "description": "처음에는 기쁨이었으나, 점차 불만으로 변화하는 전이를 테스트합니다.",
            "text": (
                "오늘은 날씨가 좋아서 기쁨이 가득했어요. 가족들과 함께 공원에 갔는데 "
                "갑자기 길이 너무 막히고, 커피가 미지근하게 나오면서 기쁨이 사라지고 "
                "약간 불만이 생기기 시작했어요. 가벼운 짜증과 함께 점점 기분이 나빠졌죠."
            ),
        },
        {
            "title": "전이_불안에서_안정",
            "description": "처음에는 불안했으나, 차츰 안정을 찾는 전이를 테스트합니다.",
            "text": (
                "중요한 발표가 있어서 너무 불안했어요. 손이 떨리고 식은땀이 났죠. "
                "하지만 예행연습을 반복하고, 동료들의 격려를 들으면서 "
                "점차 자신감이 생기더니 결국에는 안정된 기분으로 발표를 마쳤어요."
            ),
        },
        {
            "title": "복합_희에서_분노_동시에_슬픔까지",
            "description": "다중 감정이 동시에 나타나거나 연쇄적으로 변하는 복합 전이 케이스",
            "text": (
                "기다리던 소식을 들었을 때는 정말 행복했어요. "
                "그런데 나중에 알고 보니 중요한 부분이 빠져 있어서 화가 났고, "
                "결국 기대가 어긋났다는 생각에 슬픔이 몰려왔어요. "
                "갑자기 모든 것이 허탈하게 느껴졌죠."
            ),
        },
        {
            "title": "감정강도와_전이_동시_테스트",
            "description": "감정 강도가 높아지는 상황과 전이를 함께 테스트합니다.",
            "text": (
                "아침부터 기분이 설레었어요. 두근두근하는 마음으로 출근했는데, "
                "상사가 작은 실수에 대해 지나치게 간섭하자 분노가 치밀었고, "
                "그 강도가 꽤 강해졌어요. 하지만 동료가 중재해주고 "
                "차분히 대화를 나누면서 조금씩 화가 누그러지고 안도감을 찾았어요."
            ),
        },
        {
            "title": "미묘한단계별_발전_부정->중립->긍정",
            "description": "부정에서 중립을 거쳐 긍정으로 이어지는 서서한 전이와 발전 단계를 테스트",
            "text": (
                "어제까지만 해도 정말 우울했는데, "
                "오늘은 조금씩 무덤덤해지면서 마음이 편안해졌고, "
                "오후가 되니 희미하지만 기쁜 감정이 생기기 시작했어요. "
                "예전처럼 환희롭지는 않아도, 그래도 확실히 긍정적인 쪽으로 변하는 걸 느껴요."
            ),
        },
    ]

    # 3) 분석기 구성 (한 번만) — 재사용
    comps = build_transition_components(config=cfg, emotions_data=emotions_data)
    analyzer, prog, rel, cx = comps.analyzer, comps.progression, comps.relation, comps.complexer

    # 4) 실행 루프
    ts_tag = time.strftime("%Y%m%d_%H%M%S")
    results_dir = LOGS_DIR  # 타임스탬프 하위 폴더 생성하지 않음
    os.makedirs(results_dir, exist_ok=True)
    
    # 환경 정보 출력
    print(f"[환경 정보] Python 버전: {sys.version}")
    print(f"[환경 정보] 현재 작업 디렉토리: {os.getcwd()}")
    print(f"[환경 정보] 실행 파일 경로: {__file__}")
    print(f"[환경 정보] 감정 데이터 로드 상태: {'성공' if emotions_data else '실패'}")
    print(f"[환경 정보] 분석기 초기화 상태: {'성공' if analyzer else '실패'}")
    print("=" * 50)

    test_summary = {
        "total_cases": len(test_cases),
        "case_results": [],
        "success_count": 0,
        "error_count": 0,
    }

    for idx, case in enumerate(test_cases, 1):
        title = case["title"]; desc = case["description"]; text = case["text"]
        logger.info(f"[케이스 {idx}] {title} — {desc}")
        print(f"\n[케이스 {idx}] {title}\n설명: {desc}\n")

        try:
            # 전이 분석: 이미 analyzer에 데이터가 올라가 있으므로 emotions_data=None로 호출 → 재빌드 방지
            base = analyzer.analyze_emotion_transitions(text=text, emotions_data=None)

            transitions = base.get("transitions", [])
            progression = prog.analyze_progression(transitions)
            relationships = rel.analyze_relationships(transitions)
            complex_patterns = cx.detect_complex_transitions(transitions)

            if transitions:
                print("감지된 감정 전이:")
                for t_i, t in enumerate(transitions, 1):
                    print(f" {t_i}. {t['from_emotion']} → {t['to_emotion']} (conf={t.get('confidence',0):.2f}, int={t.get('intensity',0):.2f})")
                test_summary["success_count"] += 1
            else:
                print("감지된 감정 전이가 없습니다.")
                test_summary["error_count"] += 1

            safe = _sanitize_filename(title)
            case_file = os.path.join(results_dir, f"case_{idx}_{safe}.json")
            with open(case_file, "w", encoding="utf-8") as f:
                json.dump({
                    "title": title,
                    "description": desc,
                    "analysis_result": {
                        "transitions": transitions,
                        "triggers": base.get("triggers", []),
                        "patterns": base.get("patterns", {}),
                        "flow_analysis": base.get("flow_analysis", {}),
                        "emotion_progression": progression,
                        "relationships": relationships,
                        "complex_patterns": complex_patterns,
                        "metrics": base.get("metrics", {}),
                    },
                    "success": bool(transitions),
                }, f, ensure_ascii=False, indent=2)

            test_summary["case_results"].append({
                "title": title,
                "description": desc,
                "transitions": transitions,
                "emotion_progression": progression,
                "relationships": relationships,
                "complex_patterns": complex_patterns,
                "success": bool(transitions),
            })

        except Exception as e:
            logger.error(f"[케이스 {idx}] 예외: {e}", exc_info=True)
            print(f"[오류] 케이스 {idx} 처리 중 예외 발생: {e}")
            test_summary["error_count"] += 1

    # 5) 전체 요약 저장
    memory_stats = analyzer.memory_monitor.get_memory_statistics() if analyzer.memory_monitor else {}
    overall = {
        "timestamp": ts_tag,
        "test_summary": test_summary,
        "analyzer_metrics": getattr(analyzer, "metrics", TransitionMetrics(0.0,0,0,0,0.0)).__dict__,
        "memory_statistics": memory_stats,
        "success_rate": f"{(test_summary['success_count'] / max(1,len(test_cases))) * 100:.1f}%",
    }
    summary_file = os.path.join(results_dir, f"transition_analyzer_{ts_tag}.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(overall, f, ensure_ascii=False, indent=2)

    print("\n===== 테스트 실행 완료 =====")
    print(f"총 테스트 케이스: {test_summary['total_cases']}")
    print(f"성공: {test_summary['success_count']}, 실패: {test_summary['error_count']}")
    print(f"성공률: {overall['success_rate']}")
    print(f"상세 결과: {results_dir}")


if __name__ == "__main__":
    main()

