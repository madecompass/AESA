import re
import heapq
import numpy as np
import unicodedata
import os, sys, json, getpass, logging, glob, math
import hashlib
from threading import RLock
from datetime import datetime
from functools import lru_cache
from typing import Dict, Any, List, Optional, Pattern, Set, Tuple, Iterable
from collections import defaultdict
from logging.handlers import RotatingFileHandler
from pathlib import Path

# config.py와 동일한 환경변수 키 사용 (통일)
EMOTIONS_ENV_KEYS = ("EMOTIONS_JSON", "EMOTIONS_PATH", "EMOTIONS_JSON_PATH")
_EMOTIONS_PATH_CACHE: Optional[Path] = None
_FALLBACK_USED: bool = False
_LAST_ERROR: Optional[str] = None

def _resolve_emotions_json_path() -> Optional[Path]:
    global _EMOTIONS_PATH_CACHE
    if _EMOTIONS_PATH_CACHE and _EMOTIONS_PATH_CACHE.exists():
        return _EMOTIONS_PATH_CACHE
    
    # config.py와 동일한 우선순위로 경로 해결
    for k in EMOTIONS_ENV_KEYS:
        p = os.getenv(k)
        if p:
            cand = Path(p)
            if cand.exists():
                _EMOTIONS_PATH_CACHE = cand
                if k in ("EMOTIONS_PATH", "EMOTIONS_JSON_PATH"):
                    try:
                        logger.warning(f"`{k}` is deprecated; use `EMOTIONS_JSON`.")
                    except Exception:
                        logging.getLogger("linguistic").warning(f"`{k}` is deprecated; use `EMOTIONS_JSON`.")
                return _EMOTIONS_PATH_CACHE
    
    # config.py와 동일한 폴백 경로들
    src = Path(__file__).resolve().parents[2] / "src"
    candidates = [
        src / "EMOTIONS.json",
        src / "EMOTIONS.JSON",
        Path(__file__).resolve().parents[2] / "EMOTIONS.json",
        Path(__file__).resolve().parents[2] / "EMOTIONS.JSON",
    ]
    
    for cand in candidates:
        if cand.exists():
            _EMOTIONS_PATH_CACHE = cand
            return _EMOTIONS_PATH_CACHE
    return None

def _ensure_emotions_data(ed: Optional[Dict[str, Any]]):
    global _FALLBACK_USED, _LAST_ERROR
    _FALLBACK_USED = False
    _LAST_ERROR = None
    
    # 이미 유효한 데이터가 있으면 반환
    if isinstance(ed, dict) and ed:
        return ed
    
    # EMOTIONS.json 파일 로드 시도
    path = _resolve_emotions_json_path()
    if path:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict) and data:
                    return data
        except Exception as e:
            _LAST_ERROR = f"json_load_error: {e}"
            try:
                logger.warning(f"Failed to load EMOTIONS from {path}: {e}")
            except Exception:
                logging.getLogger("linguistic").warning(f"Failed to load EMOTIONS from {path}: {e}")
    
    # 프로덕션 환경에서는 폴백 사용 시 즉시 실패
    from config import EA_PROFILE, RENDER_DEPLOYMENT
    is_production = EA_PROFILE == "prod" or RENDER_DEPLOYMENT or os.getenv("PRODUCTION_MODE", "0") == "1"
    
    if is_production:
        error_msg = f"EMOTIONS.json not found or invalid in production mode. Path: {path}, Error: {_LAST_ERROR}"
        try:
            logger.error(error_msg)
        except Exception:
            logging.getLogger("linguistic").error(error_msg)
        raise RuntimeError(error_msg)
    
    # 개발 환경에서만 폴백 허용
    _FALLBACK_USED = True
    try:
        logger.warning("Using fallback emotions data in development mode")
    except Exception:
        logging.getLogger("linguistic").warning("Using fallback emotions data in development mode")
    
    return {"희": {}, "노": {}, "애": {}, "락": {}}

def _inject_diagnostics(result):
    try:
        if isinstance(result, dict):
            diag = result.setdefault("diagnostics", {})
            diag["emotions_ontology"] = {
                "fallback_used": _FALLBACK_USED,
                "path": str(_EMOTIONS_PATH_CACHE) if _EMOTIONS_PATH_CACHE else None,
                "error": _LAST_ERROR,
                "production_mode": EA_PROFILE == "prod" or RENDER_DEPLOYMENT or os.getenv("PRODUCTION_MODE", "0") == "1",
                "warning": "EMOTIONS.json fallback detected - results may be degraded" if _FALLBACK_USED else None
            }
            
            # 프로덕션에서 폴백 사용 시 결과에 경고 표시
            if _FALLBACK_USED and diag["emotions_ontology"]["production_mode"]:
                result["warning"] = "EMOTIONS.json fallback used in production - results may be unreliable"
                result["success"] = False
    except Exception:
        pass
    return result

def _load_matching_priorities(base: Dict[str, float]) -> Dict[str, float]:
    out = dict(base)
    for k in list(base.keys()):
        env = os.environ.get(f"LM_PRIORITY_{k}")
        if env is None:
            continue
        try:
            out[k] = _clamp01(float(env))
        except Exception:
            pass
    return out

# 기본값(필요 시 환경변수 LM_PRIORITY_* 로 미세조정 가능)
_MATCHING_PRIORITIES_DEFAULT = {
    "EXACT_PHRASE":        0.90,
    "KEYWORD_WITH_CONTEXT":0.70,
    "CORE_EMOTION":        0.60,
    "SITUATION":           0.50,
    "SIMPLE_KEYWORD":      0.30,
}

MATCHING_PRIORITIES: Dict[str, float] = _load_matching_priorities(_MATCHING_PRIORITIES_DEFAULT)


# =============================================================================
# Text Utils (NFC 정규화 & 하위 구간 경계 생성)
# =============================================================================
_PUNCT = r"\s\.,!?:;\"'`\)\(\[\]\{\}…~\-_/\\|·•—–"  # 경계 추정용 심플 클래스

_TOKEN_BOUNDARY = '가-힣A-Za-z0-9'

@lru_cache(maxsize=2048)
def _token_pattern(token: str) -> Pattern:
    if not token:
        return re.compile(r'$^')
    return re.compile(rf'(?<![{_TOKEN_BOUNDARY}]){re.escape(token)}(?![{_TOKEN_BOUNDARY}])')

@lru_cache(maxsize=1024)
def _token_pat(word: str) -> Pattern:
    if not word:
        return re.compile(r"$^")
    return re.compile(rf"(?<![가-힣A-Za-z0-9]){re.escape(word)}(?![가-힣A-Za-z0-9])")

@lru_cache(maxsize=8192)
def normalize(text: str) -> str:
    if text is None:
        return ""
    try:
        return unicodedata.normalize("NFC", text).strip().lower()
    except Exception:
        return (text or "").strip().lower()

def _word_boundary_wrapped(escaped_kw: str) -> str:
    # \b가 한글에서 애매할 때를 고려해, 좌우를 공백/문장부호 경계로도 허용
    left  = rf"(?:(?<=^)|(?<=[{_PUNCT}]))"
    right = rf"(?:(?=$)|(?=[{_PUNCT}]))"
    return rf"{left}(?:{escaped_kw}){right}"

_SENTENCE_SPLIT_TOKENS: Tuple[str, ...] = (
    "\ud558\uc9c0\ub9cc", "\uadf8\ub7ec\ub098", "\ubc18\uba74", "\uadf8\ub7fc\uc5d0\ub3c4", "\uadf8\ub7f0\ub370", "\uadf8\ub798\uc11c", "\uadf8\ub7ec\uc790", "\uadf8\ub7ec\ub2e4\uac00",
    "\ub2e4\ub9cc", "\ud55c\ud3b8", "\uacb0\uad6d", "\ub9c8\uce68\ub0b4", "\uc810\uc810", "\ub354\ub2c8", "\uadf8\ub7ec\ub2c8", "\uadf8\ub7fc",
    "\ud639\uc740", "\ub610\ub294", "\uc9c0\ub9cc", "\ub294\ub370", "\ud588\ub294\ub370", "\uc600\ub294\ub370", "\uc778\ub370", "\uadf8\ub7ac\ub354\ub2c8",
    "\uadf8\ub7ec\ub354\ub2c8"
)
_SENTENCE_TOKEN_PATTERN = "|".join(map(re.escape, _SENTENCE_SPLIT_TOKENS))
_SENTENCE_TOKEN_REGEX = re.compile(rf"\s*(?:{_SENTENCE_TOKEN_PATTERN})\s*")
_DEFAULT_SENTENCE_SPLIT_TOKENS: Tuple[str, ...] = _SENTENCE_SPLIT_TOKENS
_SENTENCE_PUNCT_REGEX = re.compile(r"([.?!]+)\s*")
_SENTENCE_SEPARATOR = "\u27c2"

@lru_cache(maxsize=4096)
def _split_sentences_cached(text: str) -> Tuple[str, ...]:
    if not text:
        return tuple()
    cleaned = text.strip()
    if not cleaned:
        return tuple()
    segmented = _SENTENCE_TOKEN_REGEX.sub(f" {_SENTENCE_SEPARATOR} ", cleaned)
    segmented = _SENTENCE_PUNCT_REGEX.sub("\1 " + _SENTENCE_SEPARATOR + " ", segmented)
    return tuple(
        part.strip(" ,;:")
        for part in segmented.split(_SENTENCE_SEPARATOR)
        if part and part.strip()
    )



def _set_sentence_split_tokens(tokens: Iterable[str]) -> None:
    global _SENTENCE_SPLIT_TOKENS, _SENTENCE_TOKEN_PATTERN, _SENTENCE_TOKEN_REGEX
    tok_tuple = tuple(str(t).strip() for t in (tokens or []) if str(t).strip())
    if not tok_tuple:
        tok_tuple = _DEFAULT_SENTENCE_SPLIT_TOKENS
    if tok_tuple == _SENTENCE_SPLIT_TOKENS:
        return
    _SENTENCE_SPLIT_TOKENS = tok_tuple
    _SENTENCE_TOKEN_PATTERN = "|".join(map(re.escape, _SENTENCE_SPLIT_TOKENS))
    _SENTENCE_TOKEN_REGEX = re.compile(rf"\s*(?:{_SENTENCE_TOKEN_PATTERN})\s*")
    _split_sentences_cached.cache_clear()

# =============================================================================
# CacheManager
# =============================================================================
class CacheManager:
    @staticmethod
    @lru_cache(maxsize=4096)
    def get_compiled_pattern(pattern: str,
                             flags: int = re.UNICODE | re.IGNORECASE,
                             safe: bool = False) -> Optional[Pattern]:
        try:
            p = re.escape(pattern) if safe else pattern
            compiled = re.compile(p, flags=flags)
            return compiled
        except re.error as e:
            logger.debug(f"패턴 컴파일 실패 → safe 재시도: '{pattern}' ({e})")
            try:
                compiled = re.compile(re.escape(pattern), flags=flags)
                return compiled
            except Exception as e2:
                logger.error(f"패턴 컴파일 최종 실패 '{pattern}': {e2}")
                return None

    @staticmethod
    @lru_cache(maxsize=8192)
    def get_keyword_pattern(keyword: str,
                            whole_word: bool = True,
                            case_insensitive: bool = True) -> Optional[Pattern]:
        if not keyword:
            return None
        kw_norm = normalize(keyword)
        flags = re.UNICODE | (re.IGNORECASE if case_insensitive else 0)
        try:
            esc = re.escape(kw_norm)
            patt = _word_boundary_wrapped(esc) if whole_word else esc
            return re.compile(patt, flags=flags)
        except re.error as e:
            logger.error(f"키워드 패턴 컴파일 실패 '{keyword}': {e}")
            return None

    @staticmethod
    def search(pattern: str, text: str, *, safe: bool = False, flags: int = re.UNICODE | re.IGNORECASE) -> Optional[re.Match]:
        if not text:
            return None
        comp = CacheManager.get_compiled_pattern(pattern, flags=flags, safe=safe)
        if comp is None:
            return None
        return comp.search(normalize(text))

    @staticmethod
    def finditer(pattern: str, text: str, *, safe: bool = False,
                 flags: int = re.UNICODE | re.IGNORECASE):
        if not text:
            return iter(())
        comp = CacheManager.get_compiled_pattern(pattern, flags=flags, safe=safe)
        if comp is None:
            return iter(())
        return comp.finditer(normalize(text))

    @staticmethod
    def match_keyword(keyword: str, text: str, *,
                      whole_word: bool = True,
                      case_insensitive: bool = True) -> Optional[re.Match]:
        if not text or not keyword:
            return None
        comp = CacheManager.get_keyword_pattern(keyword, whole_word=whole_word, case_insensitive=case_insensitive)
        if comp is None:
            return None
        return comp.search(normalize(text))


# =============================================================================
# EmotionProgressionMatcher
# =============================================================================
class EmotionProgressionMatcher:
    def __init__(self):
        logger.info("EmotionProgressionMatcher 개선버전 초기화 완료")
        # Shared caches/state used across progression and linguistic matching
        self.emotions_data: Dict[str, Any] = {}
        self._cached_emotions_id: Optional[int] = None
        self._core_keyword_cache: Dict[str, List[str]] = {}
        self._ling_pattern_cache: Dict[str, Dict[str, Any]] = {}
        self._category_stats_cache: Dict[int, Dict[str, Any]] = {}
        self._sub_meta_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._pattern_stats_cache: Dict[int, Dict[str, Dict[str, Any]]] = {}
        self.scfg: Dict[str, Any] = {}
        self.global_positives: Set[str] = set()
        self.global_negatives: Set[str] = set()
        self.global_intensity_mods: Dict[str, Any] = {}
        # progression-specific helper caches
        self._context_keyword_cache: Dict[Tuple[int, str], Tuple[str, ...]] = {}
        self._context_detection_cache: Dict[Tuple[str, Tuple[str, ...]], List[Dict[str, Any]]] = {}
        self._transition_confidence_cache: Dict[Tuple[str, str, str, int], float] = {}
        self._phrase2sub: Dict[str, List[Tuple[str, str]]] = {}
        self._rx_phrase_shards: Tuple[Pattern, ...] = ()
        self._phrase_index_id: Optional[int] = None
        try:
            self._regex_shard_size = int(os.getenv("LM_REGEX_SHARD", "128"))
        except (TypeError, ValueError):
            self._regex_shard_size = 128
        if self._regex_shard_size <= 0:
            self._regex_shard_size = 128
        try:
            self._cand_topk = int(os.getenv("LM_CAND_TOPK", "30"))
        except (TypeError, ValueError):
            self._cand_topk = 30
        if self._cand_topk <= 0:
            self._cand_topk = 30
        # allow helper calls that expect a nested matcher reference
        self.emotion_progression_matcher = self

    def match_emotion_progression(
        self,
        text: str,
        emotions_data: Dict[str, Any],
        sentence_emotion_scores: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        try:
            am = emotions_data.get("analysis_modules", {}) or {}
            pconf = (am.get("progression_analyzer") or {})
            tconf = (am.get("transition_analyzer") or {})
            enabled = bool(pconf.get("enabled", True))
            if not enabled:
                return {"emotion_progression_matches": [], "emotion_transition_matches": []}

            min_transition_score = float(os.environ.get(
                "LM_MIN_TRANSITION_SCORE",
                tconf.get("min_transition_score", 0.6)
            ))
            trigger_intensity = float(os.environ.get(
                "LM_PROGRESSION_TRIGGER_INTENSITY",
                pconf.get("progression_trigger_intensity", 0.7)
            ))
            max_ctx_dist = int(os.environ.get("LM_MAX_CONTEXT_DISTANCE", 20))

            sentiment = emotions_data.get("sentiment_analysis", {}) or {}
            global_pos = sentiment.get("positive_indicators", []) or []
            global_neg = sentiment.get("negative_indicators", []) or []
            # 드문 케이스: dict가 들어오면 키 목록으로 강제
            if isinstance(global_pos, dict):
                global_pos = list(global_pos.keys())
            if isinstance(global_neg, dict):
                global_neg = list(global_neg.keys())
            global_pos = list(global_pos)
            global_neg = list(global_neg)
            global_mod = dict(sentiment.get("intensity_modifiers", {}) or {})

            nt = normalize(text)
            sents = self._split_sentences_ko(nt)
            spans = self._span_sentences(nt, sents)

            time_kw_list = self._collect_context_keywords_from_bone(emotions_data, "time")
            loc_kw_list  = self._collect_context_keywords_from_bone(emotions_data, "location")
            time_ctx     = self._detect_context_items(nt, time_kw_list)
            loc_ctx      = self._detect_context_items(nt, loc_kw_list)

            trans_info  = emotions_data.get("emotion_transitions", {}) or {}
            patterns    = list(trans_info.get("patterns", []) or [])
            transition_matches: List[Dict[str, Any]] = []

            for p in patterns:
                f_e = str(p.get("from_emotion", "") or "")
                t_e = str(p.get("to_emotion", "") or "")
                trig = list(p.get("triggers", []) or [])
                ta   = p.get("transition_analysis", {}) or {}
                shift_str = str(ta.get("emotion_shift_point", "") or "")
                trig_words = list(ta.get("trigger_words", []) or [])
                all_trigs = [t for t in (trig + trig_words) if t]

                seen: Set[int] = set()
                for trig_str in all_trigs:
                    for pos in self._find_positions(nt, trig_str):
                        if pos in seen:
                            continue
                        seen.add(pos)
                        tc = self._find_nearest_context(time_ctx, pos, max_ctx_dist)
                        lc = self._find_nearest_context(loc_ctx, pos, max_ctx_dist)
                        score = self._calculate_transition_confidence(
                            nt, f_e, t_e, trig_str, tc, lc, shift_str,
                            global_pos, global_neg, global_mod,
                            sentence_emotion_scores, pos, spans
                        )
                        transition_matches.append({
                            "from_emotion": f_e,
                            "to_emotion": t_e,
                            "trigger": trig_str,
                            "position": pos,
                            "time_context": tc,
                            "location_context": lc,
                            "transition_analysis": {
                                "shift_point": shift_str,
                                "trigger_words": trig_words,
                                "score": score
                            }
                        })

            transition_matches.sort(key=lambda x: x["position"])
            filtered_trans = [t for t in transition_matches if t["transition_analysis"]["score"] >= min_transition_score]

            prog_results: List[Dict[str, Any]] = []
            for cat_key, cat_data in self._iter_categories(emotions_data):
                sub_emotions = self._get_sub_emotions_dict(cat_data)
                if not sub_emotions:
                    continue

                for sub_name, sub_data in sub_emotions.items():
                    stages_found: List[Dict[str, Any]] = []
                    situations = (sub_data.get("context_patterns", {}) or {}).get("situations", {}) or {}
                    for _, sit in situations.items():
                        eprog = sit.get("emotion_progression", {}) or {}
                        for stage, stage_desc in eprog.items():
                            if not stage_desc:
                                continue
                            phrases = stage_desc if isinstance(stage_desc, list) else [stage_desc]
                            for ph in phrases:
                                for pos in self._find_positions(nt, ph):
                                    score = 1.0
                                    local = self._local_window(nt, pos, 48)
                                    if any(a in local for a in (global_mod.get("amplifiers", []) or [])): score += 0.10
                                    if any(d in local for d in (global_mod.get("diminishers", []) or [])): score -= 0.05
                                    if any(p in local for p in global_pos): score += 0.05
                                    if any(n in local for n in global_neg): score -= 0.05

                                    if sentence_emotion_scores and spans:
                                        idx = self._sent_index_at(pos, spans)
                                        if idx is not None:
                                            for s_info in sentence_emotion_scores:
                                                if int(s_info.get("index", -1)) == idx:
                                                    for e in s_info.get("emotions", []) or []:
                                                        if str(e.get("category", "")) == str(cat_key):
                                                            try:
                                                                score += float(e.get("confidence", 0.0)) * 0.2
                                                            except Exception:
                                                                pass
                                                    break

                                    if str(stage).lower() in ("peak", "climax") and score < trigger_intensity:
                                        continue

                                    tc = self._find_nearest_context(time_ctx, pos, max_ctx_dist)
                                    lc = self._find_nearest_context(loc_ctx, pos, max_ctx_dist)
                                    meta = sub_data.get("metadata", {}) or {}
                                    stages_found.append({
                                        "stage": stage,
                                        "keyword": ph,
                                        "position": pos,
                                        "time_context": tc,
                                        "location_context": lc,
                                        "metadata_used": {
                                            "category_meta": (cat_data.get("metadata", {}) or {}).get("emotion_id", ""),
                                            "sub_meta": meta.get("emotion_id", ""),
                                            "complexity": meta.get("emotion_complexity", "basic"),
                                        },
                                        "score": round(self._clamp01(score), 3)
                                    })

                    if stages_found:
                        stages_found.sort(key=lambda x: x["position"])
                        prog_results.append({
                            "emotion_category": cat_key,
                            "sub_emotion": sub_name,
                            "stages": stages_found
                        })

            return {
                "emotion_progression_matches": prog_results,
                "emotion_transition_matches": filtered_trans
            }

        except Exception:
            logger.exception("감정 발전(Progression) 매칭 중 오류 발생")
            return {"emotion_progression_matches": [], "emotion_transition_matches": []}



    # -------------------------------------------------------------------------
    # Internal
    # -------------------------------------------------------------------------
    def _clamp01(self, x: float) -> float:
        try:
            return max(0.0, min(1.0, float(x)))
        except Exception:
            return 0.0

    def _polarity(self, eid: str) -> int:
        pc = self.scfg.get("polarity_categories", {}) if hasattr(self, "scfg") else {}
        pos_raw = pc.get("pos_main") or ["희", "락"]
        neg_raw = pc.get("neg_main") or ["노", "애"]
        target = str(eid or "")
        target_norm = self._norm(target)
        pos = {self._norm(p) for p in pos_raw}
        if target_norm in pos:
            return 1
        neg = {self._norm(n) for n in neg_raw}
        if target_norm in neg:
            return -1
        return 0

    def _split_sentences_ko(self, text: str) -> List[str]:
        if not text:
            return []
        return list(_split_sentences_cached(text))



    def _span_sentences(self, text: str, sents: List[str]) -> List[Tuple[int, int]]:
        spans: List[Tuple[int, int]] = []
        if not text or not sents:
            return spans
        start = 0
        for s in sents:
            idx = text.find(s, start)
            if idx == -1:
                idx = text.find(s)
            if idx == -1:
                continue
            end = idx + len(s)
            spans.append((idx, end))
            start = end
        return spans

    def _sent_index_at(self, pos: int, spans: List[Tuple[int, int]]) -> Optional[int]:
        for i, (a, b) in enumerate(spans):
            if a <= pos < b:
                return i
        return None

    def _iter_categories(self, emotions_data: Dict[str, Any]) -> Iterable[Tuple[str, Dict[str, Any]]]:
        """Yield (category, data) pairs from emotions_data, ignoring non-dict entries."""
        if isinstance(emotions_data, dict):
            for cat_key, cat_data in emotions_data.items():
                if isinstance(cat_data, dict):
                    yield cat_key, cat_data

    def _collect_context_keywords_from_bone(self, emotions_data: Dict[str, Any], context_type: str) -> Tuple[str, ...]:
        """Collect context keywords (time/location markers) from the emotion skeleton."""
        ctx_key = str(context_type or '').strip().lower()
        cache_key = (id(emotions_data), ctx_key)
        cached = self._context_keyword_cache.get(cache_key)
        if cached is not None:
            return cached

        keywords: Set[str] = set()

        def _collect(value: Any, allow_loose: bool = False) -> None:
            if value is None:
                return
            if isinstance(value, str):
                token = value.strip()
                if token:
                    keywords.add(token)
            elif isinstance(value, (list, tuple, set)):
                for item in value:
                    _collect(item, allow_loose)
            elif isinstance(value, dict):
                for key, nested in value.items():
                    key_str = str(key).lower() if isinstance(key, str) else ''
                    matches_ctx = bool(ctx_key and ctx_key in key_str)
                    _collect(nested, allow_loose or matches_ctx)

        for section_name in ('context_keywords', 'contextual_keywords', 'context_bones', 'context_repository'):
            section = emotions_data.get(section_name) if isinstance(emotions_data, dict) else None
            if isinstance(section, dict):
                _collect(section.get(context_type))
                if ctx_key and ctx_key != context_type:
                    _collect(section.get(ctx_key))

        analysis_modules = emotions_data.get('analysis_modules') if isinstance(emotions_data, dict) else None
        if isinstance(analysis_modules, dict):
            progression_cfg = analysis_modules.get('progression_analyzer') or {}
            if isinstance(progression_cfg, dict):
                repo = progression_cfg.get('context_keywords') or progression_cfg.get('context_repository')
                if isinstance(repo, dict):
                    _collect(repo.get(context_type))
                    if ctx_key and ctx_key != context_type:
                        _collect(repo.get(ctx_key))

        for _, cat_data in self._iter_categories(emotions_data):
            context_patterns = cat_data.get('context_patterns') if isinstance(cat_data, dict) else None
            if isinstance(context_patterns, dict):
                _collect(context_patterns.get(context_type))
                if ctx_key and ctx_key != context_type:
                    _collect(context_patterns.get(ctx_key))
                situations = context_patterns.get('situations')
                if isinstance(situations, dict):
                    for sdata in situations.values():
                        if not isinstance(sdata, dict):
                            continue
                        for key, value in sdata.items():
                            if isinstance(key, str) and ctx_key and ctx_key in key.lower():
                                _collect(value, True)
                        for key in ('keywords', 'variations', 'markers', 'examples'):
                            if isinstance(sdata.get(key), (list, tuple, set)) and ctx_key in key.lower():
                                _collect(sdata.get(key), True)

        deduped = tuple(sorted({token for token in keywords if token}))
        self._context_keyword_cache[cache_key] = deduped
        return deduped

    def _detect_context_items(self, text: str, keywords: Iterable[str]) -> List[Dict[str, Any]]:
        """Locate keyword hits within normalized text and cache the results."""
        if not text or not keywords:
            return []
        norm_text = normalize(text)
        key_tuple = tuple(sorted({kw for kw in keywords if kw}))
        cache_key = (norm_text, key_tuple)
        cached = self._context_detection_cache.get(cache_key)
        if cached is not None:
            return cached

        items: List[Dict[str, Any]] = []
        seen: Set[Tuple[str, int]] = set()
        for keyword in key_tuple:
            patt = CacheManager.get_keyword_pattern(keyword, whole_word=False, case_insensitive=True)
            if patt is None:
                continue
            for match in patt.finditer(norm_text):
                pos = match.start()
                sig = (keyword, pos)
                if sig in seen:
                    continue
                seen.add(sig)
                items.append({'keyword': keyword, 'position': pos})
        items.sort(key=lambda item: item['position'])
        self._context_detection_cache[cache_key] = items
        return items

    def _find_nearest_context(self, context_items: Iterable[Dict[str, Any]], position: int, max_distance: int) -> Optional[Dict[str, Any]]:
        if not context_items:
            return None
        best_item: Optional[Dict[str, Any]] = None
        best_distance = max_distance if max_distance is not None else 0
        for item in context_items:
            ctx_pos = item.get('position') if isinstance(item, dict) else None
            if ctx_pos is None:
                continue
            distance = abs(int(ctx_pos) - int(position))
            if distance > max_distance:
                continue
            if best_item is None or distance < best_distance:
                best_item = item
                best_distance = distance
        return best_item

    def _calculate_transition_confidence(
        self,
        text: str,
        from_emotion: str,
        to_emotion: str,
        trigger: str,
        time_context: Optional[Dict[str, Any]],
        location_context: Optional[Dict[str, Any]],
        shift_phrase: str,
        global_positive: Iterable[str],
        global_negative: Iterable[str],
        global_modifiers: Dict[str, Iterable[str]],
        sentence_emotion_scores: Optional[List[Dict[str, Any]]],
        trigger_position: int,
        sentence_spans: List[Tuple[int, int]],
    ) -> float:
        cache_key = (normalize(from_emotion or ''), normalize(to_emotion or ''), normalize(trigger or ''), int(trigger_position))
        cached = self._transition_confidence_cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            base_score = float(self.scfg.get("transition_base_score", 0.45)) if hasattr(self, "scfg") else 0.45
        except (TypeError, ValueError):
            base_score = 0.45
        score = base_score
        local_window = self._local_window(text, trigger_position, radius=32)
        local_norm = normalize(local_window)
        trigger_norm = normalize(trigger)
        if trigger_norm and trigger_norm in local_norm:
            score += 0.1

        for emo_value, bonus in ((from_emotion, 0.07), (to_emotion, 0.1)):
            emo_norm = normalize(emo_value)
            if emo_norm and emo_norm in local_norm:
                score += bonus

        shift_norm = normalize(shift_phrase)
        if shift_norm and shift_norm in local_norm:
            score += 0.05

        if time_context:
            score += 0.05
        if location_context:
            score += 0.05
        if time_context and location_context:
            score += 0.02

        combined_modifiers: List[str] = []
        if isinstance(global_modifiers, dict):
            for key in ('amplifiers', 'diminishers', 'negators'):
                combined_modifiers.extend(global_modifiers.get(key) or [])
        modifier_hits = sum(1 for token in combined_modifiers if normalize(token) and normalize(token) in local_norm)
        if modifier_hits:
            score += min(modifier_hits * 0.02, 0.06)

        pos_hits = sum(1 for token in (global_positive or []) if normalize(token) and normalize(token) in local_norm)
        neg_hits = sum(1 for token in (global_negative or []) if normalize(token) and normalize(token) in local_norm)
        score += min(pos_hits, 3) * 0.015
        score -= min(neg_hits, 3) * 0.02

        if sentence_emotion_scores and sentence_spans:
            sentence_index = self._sent_index_at(trigger_position, sentence_spans)
            if sentence_index is not None:
                for entry in sentence_emotion_scores:
                    if int(entry.get('index', -1)) != sentence_index:
                        continue
                    for emo_info in entry.get('emotions', []) or []:
                        try:
                            confidence = float(emo_info.get('confidence', 0.0))
                        except (TypeError, ValueError):
                            confidence = 0.0
                        category_norm = normalize(emo_info.get('category', ''))
                        if category_norm and to_emotion and normalize(to_emotion) in category_norm:
                            score += confidence * 0.1
                        if category_norm and from_emotion and normalize(from_emotion) in category_norm:
                            score += confidence * 0.05
                    break

        final_score = self._clamp01(score)
        self._transition_confidence_cache[cache_key] = final_score
        return final_score

    def _local_window(self, text: str, pos: int, radius: int = 40) -> str:
        a = max(0, pos - radius)
        b = min(len(text), pos + radius)
        return text[a:b]


    def _log_exc(self, context: str, exc: Exception, level: str = "error") -> None:
        msg = f"[{context}] 처리 중 오류: {type(exc).__name__}: {exc}"
        if level == "warning":
            logger.warning(msg)
        elif level == "debug":
            logger.debug(msg, exc_info=True)
        else:
            logger.exception(msg)

    def _has_token(self, text: str, tokens: Iterable[str]) -> bool:
        if not text or not tokens:
            return False
        for token in tokens:
            if token and _token_pattern(token).search(text):
                return True
        return False


    def _find_positions(self, text: str, phrase: str) -> List[int]:
        if not phrase:
            return []
        nt = self._norm(text)
        ph = self._norm(phrase)
        out: List[int] = []

        top_k = max(int(self.scfg.get("position_topk", 5)), 1) if hasattr(self, "scfg") else 5
        gap_max = max(int(self.scfg.get("fuzzy_gap_max", 2)), 0) if hasattr(self, "scfg") else 2
        fuzzy_len_max = max(int(self.scfg.get("fuzzy_len_max", 15)), 1) if hasattr(self, "scfg") else 15

        pat = CacheManager.get_keyword_pattern(ph, whole_word=False, case_insensitive=True)
        if pat:
            for m in pat.finditer(nt):
                out.append(m.start())
                if len(out) >= top_k:
                    break

        if not out and 3 <= len(ph) <= fuzzy_len_max:
            fuzzy = re.sub(r"\s+", rf".{{0,{gap_max}}}", re.escape(ph))
            try:
                rgx = re.compile(fuzzy, flags=re.UNICODE | re.IGNORECASE)
                for m in rgx.finditer(nt):
                    out.append(m.start())
                    if len(out) >= top_k:
                        break
            except re.error:
                pass

        out.sort()
        dedup: List[int] = []
        for pos in out:
            if not dedup or abs(pos - dedup[-1]) > 2:
                dedup.append(pos)
                if len(dedup) >= top_k:
                    break
        return dedup

    # ----------------------------
    # Helpers
    # ----------------------------
    def _norm(self, s: str) -> str:
        return normalize(s)

    def _get_sub_emotions_dict(self, cat_data: Dict[str, Any]) -> Dict[str, Any]:
        se = cat_data.get("sub_emotions", {})
        if isinstance(se, dict) and se:
            return se
        se2 = (cat_data.get("emotion_profile", {}) or {}).get("sub_emotions", {})
        if isinstance(se2, dict) and se2:
            return se2
        return {}

    def _get_ling_patterns(self, scope: Dict[str, Any]) -> Dict[str, Any]:
        key = self._sub_cache_key(scope)
        cached = self._ling_pattern_cache.get(key)
        if cached is not None:
            return cached
        lp = scope.get("linguistic_patterns", {})
        if isinstance(lp, dict) and lp:
            result = dict(lp)
        else:
            ep = scope.get("emotion_profile", {}) or {}
            raw_result = ep.get("linguistic_patterns", {}) or {}
            result = dict(raw_result) if isinstance(raw_result, dict) else raw_result
        self._ling_pattern_cache[key] = result
        return result


    def _sub_cache_key(self, scope: Dict[str, Any]) -> str:
        if not isinstance(scope, dict):
            return f"id:{id(scope)}"
        meta = scope.get("metadata") or {}
        if isinstance(meta, dict) and meta.get("emotion_id"):
            return str(meta.get("emotion_id"))
        return f"id:{id(scope)}"


    def _load_scoring_config(self, emotions_data: Dict[str, Any]) -> None:
        am = ((emotions_data.get("analysis_modules") or {}).get("linguistic_matcher") or {}) if isinstance(emotions_data, dict) else {}
        sc = am.get("scoring") or {}
        sa = emotions_data.get("sentiment_analysis") or {} if isinstance(emotions_data, dict) else {}

        def _coerce_float(value: Any, default: float) -> float:
            if value is None:
                return float(default)
            try:
                return float(value)
            except (TypeError, ValueError):
                return float(default)

        def _coerce_int(value: Any, default: int) -> int:
            if value is None:
                return int(default)
            try:
                return int(float(value))
            except (TypeError, ValueError):
                return int(default)

        def _ensure_list(value: Any) -> List[str]:
            if value is None:
                return []
            if isinstance(value, str):
                parts = [p.strip() for p in value.split(',') if p.strip()]
                return parts
            if isinstance(value, (list, tuple, set)):
                return [str(v).strip() for v in value if str(v).strip()]
            return []

        def _env_list(name: str) -> List[str]:
            raw = os.getenv(name)
            return _ensure_list(raw)

        def _norm_list(values: Iterable[str]) -> List[str]:
            return sorted({self._norm(v) for v in (values or []) if v})

        self.scfg = {}

        mp_defaults = {
            "EXACT_PHRASE": 0.90,
            "KEYWORD_WITH_CONTEXT": 0.70,
            "CORE_EMOTION": 0.60,
            "SITUATION": 0.50,
            "SIMPLE_KEYWORD": 0.30,
        }
        mp: Dict[str, float] = {}
        mapping = {
            "EXACT_PHRASE": "LM_PRI_EXACT",
            "KEYWORD_WITH_CONTEXT": "LM_PRI_KW_CTX",
            "CORE_EMOTION": "LM_PRI_CORE",
            "SITUATION": "LM_PRI_SIT",
            "SIMPLE_KEYWORD": "LM_PRI_SIMPLE",
        }
        for key, env_key in mapping.items():
            env_val = os.getenv(env_key)
            mp[key] = _coerce_float(env_val if env_val is not None else sc.get(key.lower()), mp_defaults[key])
        self.scfg["matching_priorities"] = mp

        self.scfg["balance_lambda"] = _coerce_float(os.getenv("LM_BALANCE_LAMBDA", sc.get("balance_lambda")), 0.15)
        len_norm = os.getenv("LM_LEN_NORM", sc.get("len_norm", "log"))
        self.scfg["len_norm"] = str(len_norm).strip().lower() if len_norm else "log"

        self.scfg["position_topk"] = max(_coerce_int(os.getenv("LM_POS_TOPK", sc.get("pos_topk")), 5), 1)
        self.scfg["fuzzy_len_max"] = max(_coerce_int(os.getenv("LM_FUZZY_LEN_MAX", sc.get("fuzzy_len_max")), 15), 1)
        self.scfg["fuzzy_gap_max"] = max(_coerce_int(os.getenv("LM_FUZZY_GAP_MAX", sc.get("fuzzy_gap_max")), 2), 0)
        self.scfg["candidate_topk"] = max(_coerce_int(os.getenv("LM_CAND_TOPK", sc.get("candidate_topk")), 30), 1)

        sim_cfg = sc.get("similarity_thresholds") or {}
        self.scfg["similarity_thresholds"] = {
            "mild": _coerce_float(os.getenv("LM_SIM_G1", sim_cfg.get("mild")), 0.5),
            "strong": _coerce_float(os.getenv("LM_SIM_G2", sim_cfg.get("strong")), 0.3),
            "short_len": _coerce_int(os.getenv("LM_SIM_SHORT_LEN", sim_cfg.get("short_len")), 2),
            "short_penalty": _coerce_float(os.getenv("LM_SIM_SHORT_PENALTY", sim_cfg.get("short_penalty")), 0.85),
        }
        self.scfg["similarity_penalties"] = {
            "mild_penalty": _coerce_float(sim_cfg.get("mild_penalty"), 0.9),
            "strong_penalty": _coerce_float(sim_cfg.get("strong_penalty"), 0.8),
        }

        phrase_tokens = _env_list("LM_PHRASE_BONUS_TOKENS") or _ensure_list(sc.get("phrase_bonus_tokens"))
        emotion_tokens = _env_list("LM_EMOTION_BONUS_TOKENS") or _ensure_list(sc.get("emotion_bonus_tokens"))
        if not phrase_tokens:
            phrase_tokens = _ensure_list(sa.get("positive_indicators"))[:5]
        if not emotion_tokens:
            emotion_tokens = _ensure_list(sa.get("core_emotions"))[:5]
        self.scfg["phrase_bonus_tokens"] = phrase_tokens
        self.scfg["emotion_bonus_tokens"] = emotion_tokens
        self.scfg["phrase_bonus_weight"] = _coerce_float(sc.get("phrase_bonus_weight"), 0.15)
        self.scfg["emotion_bonus_weight"] = _coerce_float(sc.get("emotion_bonus_weight"), 0.10)

        self.scfg["confidence_adjustment"] = {
            "bonus_core": _coerce_float(os.getenv("LM_BONUS_CORE", sc.get("bonus_core")), 0.2),
            "bonus_long_pattern": _coerce_float(os.getenv("LM_BONUS_LONG", sc.get("bonus_long")), 0.1),
            "bonus_intensity": _coerce_float(os.getenv("LM_BONUS_INTENSITY", sc.get("bonus_intensity")), 0.1),
            "bonus_context": _coerce_float(os.getenv("LM_BONUS_CONTEXT", sc.get("bonus_context")), 0.05),
            "penalty_low_sim": _coerce_float(os.getenv("LM_PENALTY_LOW_SIM", sc.get("penalty_low_sim")), 0.15),
            "penalty_mixed_polar": _coerce_float(os.getenv("LM_PENALTY_MIXED", sc.get("penalty_mixed")), 0.2),
            "long_pattern_threshold": _coerce_int(sc.get("long_pattern_threshold"), 10),
            "type_bonus": {
                "intensity_example": _coerce_float(sc.get("type_bonus_intensity_example"), 0.3),
                "keyword": _coerce_float(sc.get("type_bonus_keyword"), 0.2),
                "situation_keyword": _coerce_float(sc.get("type_bonus_situation"), 0.15),
                "default": _coerce_float(sc.get("type_bonus_default"), 0.1),
            },
            "length_bonus_threshold": _coerce_int(sc.get("length_bonus_threshold"), 10),
            "length_bonus": _coerce_float(sc.get("length_bonus"), 0.10),
            "max_confidence": _coerce_float(sc.get("max_confidence"), 0.95),
            "min_confidence": _coerce_float(sc.get("min_confidence"), 0.2),
        }

        polarity_cfg = sc.get("polarity_categories") or {}
        pos_main = polarity_cfg.get("pos_main") or polarity_cfg.get("positive") or ["희", "락"]
        neg_main = polarity_cfg.get("neg_main") or polarity_cfg.get("negative") or ["노", "애"]
        self.scfg["polarity_categories"] = {
            "pos_main": _ensure_list(pos_main) or ["희", "락"],
            "neg_main": _ensure_list(neg_main) or ["노", "애"],
            "positive": _ensure_list(polarity_cfg.get("positive")) or ["희", "락"],
            "negative": _ensure_list(polarity_cfg.get("negative")) or ["노", "애"],
        }

        weights_cfg = sc.get("context_weights") or {}
        self.scfg["context_weights"] = {
            "core_keyword": _coerce_float(weights_cfg.get("core_keyword"), 0.15),
            "situation": _coerce_float(weights_cfg.get("situation"), 0.12),
            "example": _coerce_float(weights_cfg.get("example"), 0.08),
            "intensity": _coerce_float(weights_cfg.get("intensity"), 0.05),
            "interaction": _coerce_float(weights_cfg.get("interaction"), 0.10),
        }

        amplifiers = _env_list("LM_MOD_AMPLIFIERS") or _ensure_list((sa.get("intensity_modifiers") or {}).get("amplifiers"))
        diminishers = _env_list("LM_MOD_DIMINISHERS") or _ensure_list((sa.get("intensity_modifiers") or {}).get("diminishers"))
        negators = _env_list("LM_MOD_NEGATORS") or _ensure_list((sa.get("intensity_modifiers") or {}).get("negators"))
        reversers = _env_list("LM_MOD_REVERSERS") or _ensure_list((sa.get("intensity_modifiers") or {}).get("reversers")) or _ensure_list((sa.get("intensity_modifiers") or {}).get("reverser")) or _ensure_list(sc.get("reversers"))
        amps_norm = _norm_list(amplifiers)
        dims_norm = _norm_list(diminishers)
        neg_norm = _norm_list(negators)
        rev_norm = _norm_list(reversers)
        self.scfg.setdefault("intensity_modifiers", {})
        self.scfg["intensity_modifiers"]["amplifiers"] = amps_norm
        self.scfg["intensity_modifiers"]["diminishers"] = dims_norm
        self.scfg["intensity_modifiers"]["negators"] = neg_norm
        self.scfg["intensity_modifiers"]["reversers"] = rev_norm
        self.scfg["intensity_modifiers"]["reverser"] = rev_norm
        self.scfg["intensity_modifiers"]["all"] = _norm_list(amplifiers + diminishers + negators + reversers)
        self.scfg["reversers"] = [str(r).strip() for r in (reversers or []) if str(r).strip()]

        self.scfg["pos_indicators"] = _norm_list(sa.get("positive_indicators"))
        self.scfg["neg_indicators"] = _norm_list(sa.get("negative_indicators"))

        adj = self.scfg.setdefault("confidence_adjustment", {})
        adj.setdefault("bonus_intensity", 0.05)
        adj.setdefault("penalty_diminisher", 0.05)
        adj.setdefault("bonus_ctx_pos", 0.03)
        adj.setdefault("bonus_time_ctx", 0.03)
        adj.setdefault("penalty_mixed_polar", 0.20)
        adj.setdefault("max_confidence", 0.95)
        adj.setdefault("min_confidence", 0.2)
        adj.setdefault("long_pattern_threshold", 10)

        tk, sk = self._collect_time_situation_from_bone(emotions_data)
        self.scfg["time_keywords"] = tk
        self.scfg["situation_keywords"] = sk

        bridges: List[str] = []
        try:
            am_cfg = ((emotions_data.get("analysis_modules") or {}).get("linguistic_matcher") or {}) if isinstance(emotions_data, dict) else {}
            bridges_cfg = am_cfg.get("bridges")
            if isinstance(bridges_cfg, (list, tuple, set)):
                bridges.extend(str(b) for b in bridges_cfg if b)
            existing = self.scfg.get("bridges") if isinstance(self.scfg, dict) else None
            if isinstance(existing, (list, tuple, set)):
                bridges.extend(str(b) for b in existing if b)
            if not bridges and isinstance(emotions_data, dict):
                root_phrases = emotions_data.get("transition_phrases")
                if isinstance(root_phrases, (list, tuple, set)):
                    bridges.extend(str(b) for b in root_phrases if b)
            if not bridges and isinstance(emotions_data, dict):
                for _, cat_data in (emotions_data.items() if isinstance(emotions_data, dict) else []):
                    subs = self._get_sub_emotions_dict(cat_data)
                    for _, sd in (subs.items() if isinstance(subs, dict) else []):
                        sits = (sd.get("context_patterns") or {}).get("situations", {}) or {}
                        for sit in (sits.values() if isinstance(sits, dict) else []):
                            prog = (sit.get("emotion_progression") or {})
                            for val in (prog.values() if isinstance(prog, dict) else []):
                                if isinstance(val, str) and val.strip():
                                    bridges.append(val)
        except Exception:
            pass

        allow_fb = bool(sc.get("allow_fallback_bridges", False))
        env_allow = os.getenv("LM_ALLOW_BRIDGE_FALLBACK")
        if not allow_fb and isinstance(env_allow, str):
            allow_fb = env_allow.strip().lower() in ("1", "true", "yes")
        bridge_fb = _ensure_list(sc.get("bridge_fallback")) or [
            "오랜 기다림 끝에",
            "며칠 후에",
            "얼마 지나지 않아",
            "시간이 흐른 뒤",
            "한참 후에",
        ]
        if not bridges and allow_fb:
            bridges.extend(bridge_fb)

        bridges_norm = tuple(sorted({self._norm(b) for b in bridges if isinstance(b, str) and b.strip()}))
        self.scfg["allow_fallback_bridges"] = allow_fb
        self.scfg["bridge_fallback"] = tuple(bridge_fb)
        self.scfg["bridges"] = bridges_norm

        sentence_tokens_cfg = sc.get("sentence_split_tokens")
        if not sentence_tokens_cfg and isinstance(emotions_data, dict):
            sentence_tokens_cfg = emotions_data.get("sentence_split_tokens")
        if isinstance(sentence_tokens_cfg, dict):
            sentence_tokens = [str(v).strip() for v in self._flatten_strs(sentence_tokens_cfg) if str(v).strip()]
        else:
            sentence_tokens = [str(v).strip() for v in _ensure_list(sentence_tokens_cfg)]
        if not sentence_tokens:
            sentence_tokens = [str(v).strip() for v in (self.scfg.get("reversers") or []) if str(v).strip()]
        if not sentence_tokens:
            sentence_tokens = [str(v) for v in _DEFAULT_SENTENCE_SPLIT_TOKENS]
        self.scfg["sentence_split_tokens"] = sentence_tokens
        _set_sentence_split_tokens(tuple(sentence_tokens))

        ta_cfg = ((emotions_data.get("analysis_modules") or {}).get("transition_analyzer") or {}) if isinstance(emotions_data, dict) else {}
        try:
            self.scfg["transition_base_score"] = float((ta_cfg or {}).get("base_score", 0.45))
        except (TypeError, ValueError):
            self.scfg["transition_base_score"] = 0.45

    def _flatten_strs(self, value: Any) -> List[str]:
        if isinstance(value, str):
            return [value]
        if isinstance(value, (list, tuple, set)):
            out: List[str] = []
            for item in value:
                out.extend(self._flatten_strs(item))
            return out
        if isinstance(value, dict):
            out: List[str] = []
            for nested in value.values():
                out.extend(self._flatten_strs(nested))
            return out
        return []

    def _collect_time_situation_from_bone(self, emotions_data: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        time_keys_raw = {"time", "temporal", "when", "시점", "시간", "주기", "기간"}
        situ_keys_raw = {"situation", "context", "where", "place", "장소", "상황", "환경"}
        time_keys = {self._norm(k) for k in time_keys_raw}
        situ_keys = {self._norm(k) for k in situ_keys_raw}
        times: Set[str] = set()
        situs: Set[str] = set()

        def scan(obj: Any, key_hint: str = "") -> None:
            if isinstance(obj, dict):
                for key, value in obj.items():
                    key_norm = self._norm(key) if isinstance(key, str) else ""
                    if key_norm in time_keys:
                        times.update(self._flatten_strs(value))
                    if key_norm in situ_keys:
                        situs.update(self._flatten_strs(value))
                    scan(value, key_norm)
            elif isinstance(obj, (list, tuple, set)):
                for item in obj:
                    scan(item, key_hint)

        if isinstance(emotions_data, dict):
            for _, ed in emotions_data.items():
                if not isinstance(ed, dict):
                    continue
                ctx = (ed.get("emotion_profile") or {}).get("context_patterns") or ed.get("context_patterns") or {}
                scan(ctx)

        norm = lambda xs: sorted({self._norm(s) for s in xs if isinstance(s, str) and s.strip()})
        return norm(times), norm(situs)

    def _count_keyword_hits(self, text: str, keywords: Iterable[str]) -> Tuple[int, List[str]]:
        norm_text = self._norm(text or "")
        total = 0
        hits: List[str] = []
        seen: Set[str] = set()
        for kw in keywords or []:
            kw_norm = self._norm(kw)
            if not kw_norm or kw_norm in seen:
                continue
            pattern = _token_pat(kw_norm)
            count = sum(1 for _ in pattern.finditer(norm_text))
            if count:
                total += count
                hits.append(kw_norm)
                seen.add(kw_norm)
        return total, hits

    def _has_time_signal(self, text: str) -> bool:
        scfg = getattr(self, "scfg", {}) or {}
        keywords = scfg.get("time_keywords", [])
        if not keywords:
            return False
        count, _ = self._count_keyword_hits(text, keywords)
        return count > 0

    def _has_situation_signal(self, text: str) -> bool:
        scfg = getattr(self, "scfg", {}) or {}
        keywords = scfg.get("situation_keywords", [])
        if not keywords:
            return False
        count, _ = self._count_keyword_hits(text, keywords)
        return count > 0

    def _get_category_stats(self, emotions_data: Dict[str, Any]) -> Tuple[Dict[str, Dict[str, Any]], float]:
        cache_key = id(emotions_data)
        cached = self._category_stats_cache.get(cache_key)
        if cached:
            return cached['stats'], cached['avg_vocab']

        stats: Dict[str, Dict[str, Any]] = {}
        vocab_sizes: List[int] = []
        for cat, cat_data in emotions_data.items():
            if not isinstance(cat_data, dict):
                continue
            subs = self._get_sub_emotions_dict(cat_data)
            vocab: Set[str] = set()
            for sub_name, sub_data in subs.items():
                vocab.update(self._get_core_keywords(sub_data))
                context = (sub_data.get("context_patterns") or {}).get("situations", {}) or {}
                for sdata in context.values():
                    vocab.update((sdata.get("keywords") or []))
                    vocab.update((sdata.get("variations") or []))
                    vocab.update((sdata.get("examples") or []))
                    progression = sdata.get("emotion_progression") or {}
                    vocab.update(str(v) for v in progression.values() if isinstance(v, str))
                profile = (sub_data.get("emotion_profile") or {})
                levels = (profile.get("intensity_levels") or {}).get("intensity_examples") or {}
                if isinstance(levels, dict):
                    for exs in levels.values():
                        if isinstance(exs, (list, tuple)):
                            vocab.update(map(str, exs))
            vocab_size = max(len(vocab), 1)
            stats[str(cat)] = {
                'vocab': vocab_size,
                'sub_count': max(len(self._get_sub_emotions_dict(cat_data)), 1)
            }
            vocab_sizes.append(vocab_size)
        avg_vocab = sum(vocab_sizes) / len(vocab_sizes) if vocab_sizes else 1.0
        self._category_stats_cache[cache_key] = {'stats': stats, 'avg_vocab': avg_vocab}
        return stats, avg_vocab

    def _get_pattern_statistics(self, emotions_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        cache_key = id(emotions_data)
        cached = self._pattern_stats_cache.get(cache_key)
        if cached is not None:
            return cached
        meta = emotions_data.get("ml_training_metadata") if isinstance(emotions_data, dict) else None
        pattern_stats: Dict[str, Dict[str, Any]] = {}
        if isinstance(meta, dict):
            raw = meta.get("pattern_stats") or meta.get("pattern_statistics") or {}
            if isinstance(raw, dict):
                pattern_stats = raw
            elif isinstance(raw, list):
                for entry in raw:
                    if not isinstance(entry, dict):
                        continue
                    sub_id = str(entry.get("sub_emotion_id") or entry.get("sub_id") or entry.get("emotion_id") or "")
                    pattern = entry.get("pattern")
                    if not sub_id or not pattern:
                        continue
                    bucket = pattern_stats.setdefault(sub_id, {})
                    bucket[pattern] = entry
        self._pattern_stats_cache[cache_key] = pattern_stats
        return pattern_stats

    def _get_sub_meta(self, cat: str, sub_name: str, emotions_data: Dict[str, Any]) -> Dict[str, Any]:
        key = (str(cat), str(sub_name))
        cached = self._sub_meta_cache.get(key)
        if cached is not None:
            return cached
        cat_data = emotions_data.get(cat, {}) if isinstance(emotions_data, dict) else {}
        subs = self._get_sub_emotions_dict(cat_data)
        meta = (subs.get(sub_name, {}) or {}).get("metadata") or {}
        self._sub_meta_cache[key] = meta
        return meta

    def _set_emotions_data_reference(self, emotions_data: Dict[str, Any]) -> None:
        new_id = id(emotions_data)
        if new_id != self._cached_emotions_id:
            self._core_keyword_cache.clear()
            self._ling_pattern_cache.clear()
            self._category_stats_cache.clear()
            self._sub_meta_cache.clear()
            self._pattern_stats_cache.clear()
            self._cached_emotions_id = new_id
        self.emotions_data = emotions_data
        self._build_phrase_index_and_shards(self.emotions_data)
        self._load_scoring_config(self.emotions_data)


    def _build_phrase_index_and_shards(self, emotions_data: Dict[str, Any]) -> None:
        if not isinstance(emotions_data, dict):
            self._phrase2sub = {}
            self._rx_phrase_shards = ()
            self._phrase_index_id = None
            return
        data_id = id(emotions_data)
        if self._phrase_index_id == data_id and self._phrase2sub:
            return

        phrase_map: Dict[str, List[Tuple[str, str]]] = defaultdict(list)

        def push_phrase(phrase: Any, cat: str, sub: str) -> None:
            if not isinstance(phrase, str):
                return
            norm = self._norm(phrase)
            if not norm or len(norm) < 2:
                return
            phrase_map[norm].append((str(cat), str(sub)))

        for cat, cat_data in (emotions_data.items() if isinstance(emotions_data, dict) else []):
            subs = self._get_sub_emotions_dict(cat_data)
            for sub_name, sd in (subs.items() if isinstance(subs, dict) else []):
                ep = (sd.get("emotion_profile") or {})
                for word in (ep.get("core_keywords") or []):
                    push_phrase(word, cat, sub_name)
                sits = (sd.get("context_patterns") or {}).get("situations", {}) or {}
                for situ in (sits.values() if isinstance(sits, dict) else []):
                    for key in ("keywords", "variations", "examples"):
                        for word in (situ.get(key) or []):
                            push_phrase(word, cat, sub_name)
                    prog = (situ.get("emotion_progression") or {})
                    for val in (prog.values() if isinstance(prog, dict) else []):
                        push_phrase(val, cat, sub_name)

        self._phrase2sub = {k: sorted(set(v)) for k, v in phrase_map.items() if v}
        words = list(self._phrase2sub.keys())
        shard_size = max(32, min(2048, int(self._regex_shard_size or 128)))
        shards: List[Pattern] = []
        for i in range(0, len(words), shard_size):
            chunk = words[i:i + shard_size]
            if not chunk:
                continue
            alt = "|".join(sorted((re.escape(w) for w in chunk), key=len, reverse=True))
            try:
                shards.append(re.compile(alt))
            except re.error:
                for word in chunk:
                    try:
                        shards.append(re.compile(re.escape(word)))
                    except re.error:
                        continue
        self._rx_phrase_shards = tuple(shards)
        self._phrase_index_id = data_id

    def _collect_phrase_candidates(self, normalized_text: str) -> Dict[str, Dict[str, int]]:
        if not normalized_text or not self._phrase2sub or not self._rx_phrase_shards:
            return {}
        counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for rx in self._rx_phrase_shards:
            for match in rx.finditer(normalized_text):
                key = self._norm(match.group(0))
                if not key:
                    continue
                for cat, sub in self._phrase2sub.get(key, []):
                    counts[cat][sub] += 1
        return {cat: dict(subs) for cat, subs in counts.items()}

    def _get_core_keywords(self, scope: Dict[str, Any]) -> List[str]:
        key = self._sub_cache_key(scope)
        cached = self._core_keyword_cache.get(key)
        if cached is not None:
            return cached
        ep = scope.get("emotion_profile", {}) or {}
        kws = list(ep.get("core_keywords", []) or scope.get("core_keywords", []) or [])
        self._core_keyword_cache[key] = kws
        return kws


    # ----------------------------
    # Globals
    # ----------------------------

    def _load_global_sentiment_indicators(self, emotions_data: Dict[str, Any]):
        gs = emotions_data.get("sentiment_analysis", {}) or {}
        self.global_positives = set(gs.get("positive_indicators", []) or [])
        self.global_negatives = set(gs.get("negative_indicators", []) or [])
        self.global_intensity_mods = gs.get("intensity_modifiers", {}) or {}

    # ----------------------------
    # Confidence & Scoring
    # ----------------------------
    def _calculate_confidence(self, text: str, pattern: str, emotion_data: Dict[str, Any]) -> float:
        base = MATCHING_PRIORITIES.get("SIMPLE_KEYWORD", 0.3)
        try:
            conf_adj = self.scfg.get("confidence_adjustment", {}) if hasattr(self, "scfg") else {}
            bonus_core = float(conf_adj.get("bonus_core", 0.2))
            bonus_long = float(conf_adj.get("bonus_long_pattern", 0.1))
            bonus_intensity = float(conf_adj.get("bonus_intensity", 0.05))
            bonus_context = float(conf_adj.get("bonus_context", 0.05))
            penalty_mixed = float(conf_adj.get("penalty_mixed_polar", 0.2))
            long_threshold = int(conf_adj.get("long_pattern_threshold", 10))
            max_conf = float(conf_adj.get("max_confidence", 0.95))
            min_conf = float(conf_adj.get("min_confidence", 0.0))

            pos_tokens = [self._norm(tok) for tok in (self.scfg.get("pos_indicators", []) if hasattr(self, "scfg") else [])]
            neg_tokens = [self._norm(tok) for tok in (self.scfg.get("neg_indicators", []) if hasattr(self, "scfg") else [])]
            polarity_cfg = self.scfg.get("polarity_categories", {}) if hasattr(self, "scfg") else {}
            pos_main = set(polarity_cfg.get("pos_main", [])) or {"희", "락"}
            neg_main = set(polarity_cfg.get("neg_main", [])) or {"노", "애"}

            t = self._norm(text)
            p = self._norm(pattern)
            pattern_len = len(p)
            score = base
            if pattern_len >= long_threshold:
                score += bonus_long

            meta = emotion_data.get("metadata", {}) or {}
            eid = meta.get("emotion_id", "")
            complexity = meta.get("emotion_complexity", "basic")
            meta_category = meta.get("emotion_category") or emotion_data.get("emotion_category")
            ver = meta.get("version", "1.0")
            if eid in {"1-1", "1-2"}:
                score += bonus_core / 2.0
            if complexity == "complex":
                score += bonus_context
            elif complexity == "subtle":
                score += bonus_context / 2.0
            try:
                if float(ver) >= 1.2:
                    score += bonus_context / 2.0
            except Exception:
                pass

            ep = emotion_data.get("emotion_profile", {}) or {}
            il = ep.get("intensity_levels", {}) or {}
            iex = il.get("intensity_examples", {}) or {}
            for level, examples in (iex.items() if isinstance(iex, dict) else []):
                for ex in examples or []:
                    if self._norm(ex) in t:
                        if level == "high":
                            score += bonus_intensity * 3.0
                        elif level == "medium":
                            score += bonus_intensity * 2.0
                        else:
                            score += bonus_intensity
                        break

            core_keywords = self._get_core_keywords(emotion_data)
            if core_keywords:
                for kw in core_keywords:
                    pat = CacheManager.get_keyword_pattern(kw, whole_word=True, case_insensitive=True)
                    if pat and pat.search(t):
                        score += bonus_core
                        break

            situations = (emotion_data.get("context_patterns", {}) or {}).get("situations", {}) or {}
            for sdata in situations.values():
                desc = sdata.get("description", "")
                if desc and self._norm(desc) in t:
                    score += bonus_context
                kws = sdata.get("keywords", []) or []
                km = 0
                for kw in kws:
                    pat = CacheManager.get_keyword_pattern(kw, whole_word=True, case_insensitive=True)
                    if pat and pat.search(t):
                        km += 1
                if km:
                    score += min(km * (bonus_context / 2.0), bonus_context)

            ling = self._get_ling_patterns(emotion_data)
            sm = ling.get("sentiment_modifiers", {}) or {}
            amps = sm.get("amplifiers", []) or []
            dims = sm.get("diminishers", []) or []
            if self._has_token(t, [self._norm(a) for a in amps]):
                score += bonus_intensity
            if self._has_token(t, [self._norm(d) for d in dims]):
                score -= bonus_intensity / 2.0

            kps = ling.get("key_phrases", []) or []
            for kp in kps:
                patt = self._norm(kp.get("pattern", ""))
                if patt and patt in t:
                    score += float(kp.get("weight", bonus_context / 2.0))

            combos = ling.get("sentiment_combinations", []) or []
            for combo in combos:
                words = [self._norm(w) for w in (combo.get("words", []) or [])]
                if words and all(w in t for w in words):
                    score += float(combo.get("weight", bonus_context / 2.0))

            pos_hit = self._has_token(t, pos_tokens)
            neg_hit = self._has_token(t, neg_tokens)
            cat_conflict = (meta_category in pos_main and neg_hit) or (meta_category in neg_main and pos_hit)
            if (pos_hit and neg_hit) or cat_conflict:
                score *= max(0.0, 1.0 - penalty_mixed)

            return max(min_conf, min(max_conf, score))
        except Exception as exc:
            self._log_exc("신뢰도 산출", exc)
            return base

    def _calculate_intensity_confidence(self, match: Dict[str, Any], *, text: str) -> float:
        try:
            adj = self.scfg["confidence_adjustment"]
            mods = self.scfg.get("intensity_modifiers", {})
            amps = mods.get("amplifiers", [])
            dims = mods.get("diminishers", [])
            posw = self.scfg.get("pos_indicators", [])

            txt = self._norm(text or "")
            score = float(match.get("intensity_base", 0.0))

            if any(_token_pat(w).search(txt) for w in amps):
                score += adj["bonus_intensity"]
            if any(_token_pat(w).search(txt) for w in dims):
                score *= (1.0 - adj["penalty_diminisher"])
            if any(_token_pat(w).search(txt) for w in posw):
                score += adj["bonus_ctx_pos"]

            if self._has_time_signal(text):
                score += adj.get("bonus_time_ctx", 0.03)
            if self._has_situation_signal(text):
                score += adj.get("bonus_context", 0.05) / 2.0

            return max(0.0, min(1.0, score))
        except Exception as exc:
            self._log_exc("강도 신뢰도 계산", exc)
            base = float(match.get("intensity_base", 0.0))
            return max(0.0, min(1.0, base))

    def _filter_and_rank_matches(self, matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        try:
            thresholds = self.scfg.get("similarity_thresholds", {}) if hasattr(self, "scfg") else {}
            adjust_cfg = self.scfg.get("confidence_adjustment", {}) if hasattr(self, "scfg") else {}
            penalty_low = float(adjust_cfg.get("penalty_low_sim", 0.15))
            penalty_mixed = float(adjust_cfg.get("penalty_mixed_polar", 0.2))
            mild_gate = float(thresholds.get("mild", 0.5))
            strong_gate = float(thresholds.get("strong", 0.3))
            short_len = int(thresholds.get("short_len", 2))
            short_penalty = float(thresholds.get("short_penalty", 0.85))
            phrase_tokens = self.scfg.get("phrase_bonus_tokens", []) if hasattr(self, "scfg") else []
            emotion_tokens = self.scfg.get("emotion_bonus_tokens", []) if hasattr(self, "scfg") else []
            phrase_bonus = float(self.scfg.get("phrase_bonus_weight", 0.15)) if hasattr(self, "scfg") else 0.15
            emotion_bonus = float(self.scfg.get("emotion_bonus_weight", 0.10)) if hasattr(self, "scfg") else 0.10
            length_bonus_threshold = int(adjust_cfg.get("length_bonus_threshold", 10))
            length_bonus_weight = float(adjust_cfg.get("length_bonus", 0.10))
            max_conf = float(adjust_cfg.get("max_confidence", 0.95))
            min_conf = float(adjust_cfg.get("min_confidence", 0.2))

            uniq: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
            for match in matches:
                key = (match["emotion_category"], match["sub_emotion"], match["pattern"])
                base_conf = float(match.get("confidence", 0.3))
                similarity = float(match.get("context_similarity", 1.0))
                if similarity < strong_gate:
                    base_conf *= max(0.0, 1.0 - penalty_low)
                elif similarity < mild_gate:
                    base_conf *= max(0.0, 1.0 - penalty_low / 2.0)

                pattern_text = (match.get("pattern") or "").strip()
                pattern_len = len(pattern_text)
                if 0 < pattern_len <= short_len:
                    base_conf *= short_penalty

                bonus = 0.0
                if phrase_tokens and any(tok in pattern_text for tok in phrase_tokens):
                    bonus += phrase_bonus
                if emotion_tokens and any(tok in pattern_text for tok in emotion_tokens):
                    bonus += emotion_bonus
                if pattern_len > length_bonus_threshold:
                    bonus += length_bonus_weight

                adjusted = min(max(base_conf + bonus, min_conf), max_conf)
                if match.get("mixed_polarity"):
                    adjusted *= max(0.0, 1.0 - penalty_mixed)

                if key not in uniq or adjusted > uniq[key]["confidence"]:
                    uniq[key] = match
                    uniq[key]["confidence"] = adjusted

            filtered = [item for item in uniq.values() if item["confidence"] >= min_conf]
            filtered.sort(key=lambda x: x["confidence"], reverse=True)
            return filtered
        except Exception as exc:
            self._log_exc("매칭 필터/정렬", exc)
            return []
    def _check_emotion_relevance(self, text: str, emotion: str) -> bool:
        try:
            t = self._norm(text)
            e = self._norm(emotion)
            if e in t:
                return True
            for _, cat in self.emotions_data.items():
                subs = self._get_sub_emotions_dict(cat)
                for sd in subs.values():
                    rel = (sd.get("emotion_profile", {}) or {}).get("related_emotions", {}) or {}
                    for lst in rel.values():
                        for item in lst or []:
                            if self._norm(item) in t:
                                return True
            return False
        except Exception:
            logger.exception("감정 관련성 검사 중 오류")
            return False

    def _similarity_check(self, s1: str, s2: str) -> float:
        set1, set2 = set(s1), set(s2)
        inter = set1 & set2
        union = set1 | set2
        if not union:
            return 0.0
        return len(inter) / len(union)

    def _check_context_relevance(self, text: str, pattern: str) -> bool:
        try:
            if len(pattern) < 2:
                return False
            sents = self._split_sentences_ko(text)
            p = self._norm(pattern)
            for s in sents:
                ns = self._norm(s)
                if p in ns:
                    return True
                if self._similarity_check(ns, p) > 0.75:
                    return True
            return False
        except Exception:
            logger.exception("문맥 관련성 검사 중 오류")
            return False

    def _extract_emotion_context(self, text: str, pattern: str, window_size: int = 20) -> Dict[str, str]:
        try:
            t = self._norm(text)
            positions = self._find_positions(t, pattern)
            if not positions:
                return {"before": "", "after": "", "full_context": ""}
            pos = positions[0]
            start = max(0, pos - window_size)
            end = min(len(t), pos + len(self._norm(pattern)) + window_size)
            return {
                "before": t[start:pos].strip(),
                "after": t[pos + len(self._norm(pattern)):end].strip(),
                "full_context": t[start:end].strip(),
            }
        except Exception:
            logger.exception("감정 문맥 추출 중 오류")
            return {"before": "", "after": "", "full_context": ""}

    # ----------------------------
    # Explicit transitions
    # ----------------------------
    def _analyze_explicit_transitions(self, text: str, emotions_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        try:
            t = self._norm(text)
            res = []
            trans = (emotions_data.get("emotion_transitions", {}) or {}).get("patterns", []) or []
            gs = emotions_data.get("sentiment_analysis", {}) or {}
            gpos = gs.get("positive_indicators", []) or []
            gneg = gs.get("negative_indicators", []) or []
            gmods = gs.get("intensity_modifiers", {}) or {}
            sents = self._split_sentences_ko(t)
            spans = self._span_sentences(t, sents)

            tk = self.emotion_progression_matcher._collect_context_keywords_from_bone(emotions_data, "time")
            lk = self.emotion_progression_matcher._collect_context_keywords_from_bone(emotions_data, "location")
            time_ctx = self.emotion_progression_matcher._detect_context_items(t, tk)
            loc_ctx = self.emotion_progression_matcher._detect_context_items(t, lk)

            for info in trans:
                f = str(info.get("from_emotion", "") or "")
                to = str(info.get("to_emotion", "") or "")
                trig = list(info.get("triggers", []) or [])
                ta = info.get("transition_analysis", {}) or {}
                shift = str(ta.get("emotion_shift_point", "") or "")
                trig_words = list(ta.get("trigger_words", []) or [])
                all_trigs = [x for x in (trig + trig_words) if x]
                seen: Set[int] = set()
                for w in all_trigs:
                    for pos in self._find_positions(t, w):
                        if pos in seen:
                            continue
                        seen.add(pos)
                        tc = self.emotion_progression_matcher._find_nearest_context(time_ctx, pos, 20)
                        lc = self.emotion_progression_matcher._find_nearest_context(loc_ctx, pos, 20)
                        score = self.emotion_progression_matcher._calculate_transition_confidence(
                            t, f, to, w, tc, lc, shift, gpos, gneg, gmods,
                            getattr(self, "cached_sentence_emotion_scores", None), pos, spans
                        )
                        res.append({
                            "from": f,
                            "to": to,
                            "trigger": w,
                            "position": pos,
                            "confidence": score,
                            "time_context": tc,
                            "location_context": lc,
                            "from_emotion_in_text": self._norm(f) in t,
                            "to_emotion_in_text": self._norm(to) in t,
                            "shift_point": shift
                        })
            res.sort(key=lambda x: x["position"])
            return res
        except Exception:
            logger.exception("명시적 전이 분석 중 오류")
            return []


    def _levenshtein_distance(self, s1: str, s2: str, early_stop: int = None) -> int:
        if s1 == s2:
            return 0
        l1, l2 = len(s1), len(s2)
        if l1 == 0 or l2 == 0:
            return max(l1, l2)
        d = [[0] * (l2 + 1) for _ in range(l1 + 1)]
        for i in range(l1 + 1):
            d[i][0] = i
        for j in range(l2 + 1):
            d[0][j] = j
        for i in range(1, l1 + 1):
            c1 = s1[i - 1]
            for j in range(1, l2 + 1):
                c2 = s2[j - 1]
                cost = 0 if c1 == c2 else 1
                d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
                if early_stop is not None and d[i][j] > early_stop:
                    return d[i][j]
        return d[l1][l2]

    def _approximate_original_position(self, original_text: str, stripped_text: str, stripped_index: int) -> int:
        import string
        punctuation_and_spaces = string.punctuation + " \t\r\n"
        valid = 0
        for i, ch in enumerate(original_text):
            if ch not in punctuation_and_spaces:
                valid += 1
            if valid == stripped_index:
                return i
        return len(original_text) - 1

    # ----------------------------
    # Time series
    # ----------------------------
    def analyze_time_series(self, text: str, emotions_data: Dict[str, Any]) -> Dict[str, Any]:
        sents = self._split_sentences_ko(text)
        series = []
        prev = None
        for idx, sent in enumerate(sents):
            cur = self._analyze_sentence_emotions(sent, emotions_data)
            ttype = None
            if prev:
                ttype = self._determine_transition_type(prev, cur)
            series.append({"index": idx, "sentence": sent, "emotions": cur, "transition": ttype})
            prev = cur
        return {"time_series": series}

    # ----------------------------
    # Main matching
    # ----------------------------
    def match_linguistic_patterns(
            self,
            text: str,
            emotions_data: Dict[str, Any],
            pattern_results: Optional[Dict[str, Any]] = None,
            context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        self._set_emotions_data_reference(emotions_data)
        self._load_global_sentiment_indicators(self.emotions_data)
        try:
            if not isinstance(text, str) or not isinstance(emotions_data, dict):
                logger.error("잘못된 입력 형식")
                return {}
            if not text.strip():
                logger.warning("빈 텍스트가 입력되었습니다")
                return {}
            nt = self._norm(text)
            cand_counts = self._collect_phrase_candidates(nt)
            topk_limit = max(1, self._cand_topk)
            candidate_map = {
                cat: [sub for sub, _cnt in sorted(subs.items(), key=lambda kv: kv[1], reverse=True)[:topk_limit]]
                for cat, subs in (cand_counts or {}).items()
            }

            result = {
                "matched_phrases": [],
                "sentiment_patterns": [],
                "modifier_effects": {},
                "weighted_scores": {},
                "pattern_analysis": {},
                "dynamic_expressions": {},
                "emotional_expansion": {},
                "contextual_meaning": {},
                "time_series": {},
                "explicit_transitions": []
            }

            matched_phrases = []
            processed = set()

            dyn_kw: Dict[str, str] = {}
            for cat, cat_data in emotions_data.items():
                subs = self._get_sub_emotions_dict(cat_data)
                for _, sd in subs.items():
                    for kw in self._get_core_keywords(sd) or []:
                        if kw not in dyn_kw:
                            dyn_kw[kw] = cat

            pos_tokens = [self._norm(tok) for tok in (self.scfg.get("pos_indicators", []) if hasattr(self, "scfg") else [])]
            neg_tokens = [self._norm(tok) for tok in (self.scfg.get("neg_indicators", []) if hasattr(self, "scfg") else [])]
            polarity_main = self.scfg.get("polarity_categories", {}) if hasattr(self, "scfg") else {}
            pos_main = set(polarity_main.get("pos_main", [])) or {"희", "락"}
            neg_main = set(polarity_main.get("neg_main", [])) or {"노", "애"}
            penalty_mixed = float(self.scfg.get("confidence_adjustment", {}).get("penalty_mixed_polar", 0.2)) if hasattr(self, "scfg") else 0.2
            sentences = list(_split_sentences_cached(text))
            for raw_sent in sentences:
                sent = self._norm(raw_sent)
                if not sent:
                    continue
                if sent in processed:
                    continue
                pos_hit = self._has_token(sent, pos_tokens)
                neg_hit = self._has_token(sent, neg_tokens)
                best = None
                best_conf = 0.0
                for cat, cat_data in emotions_data.items():
                    subs = self._get_sub_emotions_dict(cat_data)
                    if candidate_map:
                        candidate_names = candidate_map.get(cat)
                        if not candidate_names:
                            continue
                        subs = {name: subs[name] for name in candidate_names if name in subs and isinstance(subs[name], dict)}
                        if not subs:
                            continue
                    for sub_name, sd in subs.items():
                        conf = self._calculate_confidence(sent, sent, sd)
                        meta_bonus = 0.0
                        if any(tok in sent for tok in ["처럼", "같이", "듯이", "마냥", "마치"]):
                            meta_bonus += 0.30
                        kw_bonus = 0.0
                        for w in sent.split():
                            if w in dyn_kw and dyn_kw[w] == cat:
                                kw_bonus += 0.20

                        final_conf = min(conf + meta_bonus + kw_bonus, 1.0)
                        mixed = (pos_hit and neg_hit) or (cat in pos_main and neg_hit) or (cat in neg_main and pos_hit)
                        if mixed:
                            final_conf *= max(0.0, 1.0 - penalty_mixed)
                        final_conf = max(0.0, final_conf)

                        if final_conf > best_conf:
                            best_conf = final_conf
                            best = {
                                "emotion_category": cat,
                                "sub_emotion": sub_name,
                                "pattern": sent,
                                "type": "sentence_pattern",
                                "confidence": final_conf,
                                "mixed_polarity": mixed,
                                "main_emotion": cat,
                            }

                        il = (sd.get("emotion_profile", {}) or {}).get("intensity_levels", {}) or {}
                        iex = il.get("intensity_examples", {}) or {}
                        for level, exs in (iex.items() if isinstance(iex, dict) else []):
                            for ex in exs or []:
                                if self._norm(ex) in sent and ex not in processed:
                                    c2 = self._calculate_confidence(sent, ex, sd)
                                    f2 = min(c2 + meta_bonus + kw_bonus, 1.0)
                                    mixed_ex = mixed
                                    if mixed_ex:
                                        f2 *= max(0.0, 1.0 - penalty_mixed)
                                    f2 = max(0.0, f2)
                                    if f2 > best_conf:
                                        best_conf = f2
                                        best = {
                                            "emotion_category": cat,
                                            "sub_emotion": sub_name,
                                            "pattern": ex,
                                            "type": "intensity_example",
                                            "intensity_level": level,
                                            "confidence": f2,
                                            "mixed_polarity": mixed_ex,
                                            "main_emotion": cat,
                                        }

                        sits = (sd.get("context_patterns", {}) or {}).get("situations", {}) or {}
                        for sk, sdata in sits.items():
                            desc = sdata.get("description", "")
                            if desc and self._check_context_relevance(sent, desc):
                                for kw in (sdata.get("keywords", []) or []):
                                    if self._norm(kw) in sent and len(kw) > 2 and kw not in processed:
                                        c3 = self._calculate_confidence(sent, kw, sd)
                                        f3 = min(c3 + meta_bonus + kw_bonus, 1.0)
                                        mixed_kw = mixed
                                        if mixed_kw:
                                            f3 *= max(0.0, 1.0 - penalty_mixed)
                                        f3 = max(0.0, f3)
                                        if f3 > best_conf:
                                            best_conf = f3
                                            best = {
                                                "emotion_category": cat,
                                                "sub_emotion": sub_name,
                                                "pattern": kw,
                                                "type": "situation_keyword",
                                                "situation": sk,
                                                "confidence": f3,
                                                "mixed_polarity": mixed_kw,
                                                "main_emotion": cat,
                                            }

                if best and best["confidence"] >= 0.3:
                    best["context_similarity"] = self._similarity_check(self._norm(sent), self._norm(best.get("pattern", "")))
                    matched_phrases.append(best)
                    processed.add(sent)
                    logger.debug(f"문장 패턴 매칭: {sent}, 신뢰도: {best['confidence']}")


            if matched_phrases:
                matched_phrases = heapq.nlargest(12, matched_phrases, key=lambda x: x.get("confidence", 0.0))
            result["matched_phrases"] = self._filter_and_rank_matches(matched_phrases)
            result["sentiment_patterns"] = self._analyze_sentiment_combinations(nt, emotions_data)
            if result["sentiment_patterns"]:
                result["sentiment_patterns"] = heapq.nlargest(12, result["sentiment_patterns"], key=lambda x: x.get("confidence", 0.0))
            result["modifier_effects"] = self._analyze_sentiment_modifiers(nt, emotions_data)
            result["dynamic_expressions"] = self.analyze_dynamic_expressions(nt, emotions_data)
            result["emotional_expansion"] = self.analyze_emotional_expansion(nt, emotions_data)

            ctx_scores = self._analyze_emotional_context(nt, emotions_data)
            result["contextual_meaning"] = {
                "emotion_scores": ctx_scores,
                "dominant_emotion": max(ctx_scores.items(), key=lambda x: x[1])[0] if ctx_scores else None,
                "analysis_timestamp": datetime.now().isoformat(),
            }

            result["intensity_patterns"] = self._match_intensity_patterns(nt, emotions_data)
            result["explicit_transitions"] = self._analyze_explicit_transitions(nt, emotions_data)
            if result["explicit_transitions"]:
                result["explicit_transitions"] = heapq.nlargest(10, result["explicit_transitions"], key=lambda x: x.get("confidence", 0.0))
            result["time_series"] = self.analyze_time_series(nt, emotions_data)

            weighted_scores = self._calculate_weighted_scores(
                matched_phrases=result["matched_phrases"],
                sentiment_patterns=result["sentiment_patterns"],
                modifier_effects=result["modifier_effects"],
                intensity_patterns=result["intensity_patterns"],
                emotions_data=emotions_data,
            )

            result["weighted_scores"] = weighted_scores
            logger.info(f"언어 패턴 매칭 완료: {len(result['matched_phrases'])}개의 구문 매칭")
            return result
        except Exception:
            logger.exception("언어 패턴 매칭 실패")
            return {}

    # ----------------------------
    # Combinations & Modifiers
    # ----------------------------
    def _analyze_sentiment_combinations(self, text: str, emotions_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        try:
            out = []
            t = self._norm(text)
            for cat, cat_data in emotions_data.items():
                subs = self._get_sub_emotions_dict(cat_data)
                for sub_name, sd in subs.items():
                    rel = (sd.get("emotion_profile", {}) or {}).get("related_emotions", {}) or {}
                    il = (sd.get("emotion_profile", {}) or {}).get("intensity_levels", {}) or {}
                    iex = il.get("intensity_examples", {}) or {}
                    cur_level = "medium"
                    for level, exs in (iex.items() if isinstance(iex, dict) else []):
                        if any(self._norm(ex) in t for ex in exs or []):
                            cur_level = level
                            break
                    for rtype, lst in rel.items():
                        for emo in lst or []:
                            en = self._norm(emo)
                            if en and en in t:
                                pat = CacheManager.get_keyword_pattern(en, whole_word=False, case_insensitive=True)
                                pos = [m.start() for m in pat.finditer(t)] if pat else []
                                ctx = self._extract_context(text, pos, pattern_length=len(emo))
                                conf = self._calculate_confidence(text, emo, cat_data)
                                weight = 1.0
                                ling = self._get_ling_patterns(cat_data)
                                combos = ling.get("sentiment_combinations", []) or []
                                for combo in combos:
                                    if emo in (combo.get("words", []) or []):
                                        weight = combo.get("weight", 1.0)
                                        break
                                out.append({
                                    "emotion_category": cat,
                                    "sub_emotion": sub_name,
                                    "pattern": emo,
                                    "relation_type": rtype,
                                    "intensity_level": cur_level,
                                    "confidence": conf,
                                    "weight": weight,
                                    "positions": pos,
                                    "context": ctx
                                })
            return out
        except Exception:
            logger.exception("감정 표현 조합 분석 중 오류")
            return []

    def _analyze_sentiment_modifiers(self, text: str, emotions_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            t = self._norm(text)
            effects = {"amplifiers": [], "diminishers": [], "negators": [], "total_effect": 1.0}
            amps_glob = (self.global_intensity_mods.get("amplifiers") or [])
            dims_glob = (self.global_intensity_mods.get("diminishers") or [])
            neg_glob  = (self.global_intensity_mods.get("negators") or [])
            def scan_and_push(lst, key):
                for w in lst or []:
                    if self._norm(w) in t:
                        effects[key].append({"modifier": w, "emotion_id": "global"})
            scan_and_push(amps_glob, "amplifiers")
            scan_and_push(dims_glob, "diminishers")
            scan_and_push(neg_glob,  "negators")
            for eid, ed in emotions_data.items():
                sm = self._get_ling_patterns(ed).get("sentiment_modifiers", {}) or {}
                for mtype in ["amplifiers", "diminishers", "negators"]:
                    for w in sm.get(mtype, []) or []:
                        if self._norm(w) in t:
                            effects[mtype].append({"modifier": w, "emotion_id": eid})
            amps = len(effects["amplifiers"])
            dims = len(effects["diminishers"])
            negs = len(effects["negators"])
            total = 1.0 + amps * 0.05 - dims * 0.03 - negs * 0.04
            effects["total_effect"] = max(total, 0.5)
            return effects
        except Exception:
            logger.exception("감정 수정자 분석 중 오류")
            return {"amplifiers": [], "diminishers": [], "negators": [], "total_effect": 1.0}

    # ----------------------------
    # Dynamic expressions
    # ----------------------------
    def analyze_dynamic_expressions(self, text: str, emotions_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            res = {"emotion_transitions": [], "progression_patterns": []}
            sents = self._split_sentences_ko(text)
            for i in range(len(sents) - 1):
                cur = sents[i]
                nxt = sents[i + 1]
                cur_e = self._analyze_sentence_emotions(cur, emotions_data)
                nxt_e = self._analyze_sentence_emotions(nxt, emotions_data)
                bridge = self._extract_bridging_text(text, cur, nxt)
                bridge_list = (self.scfg.get("bridges") if hasattr(self, "scfg") else []) or []
                bridge_norm = self._norm(bridge)
                kw = []
                for phrase in bridge_list:
                    candidate = str(phrase).strip()
                    if not candidate:
                        continue
                    if candidate in bridge or self._norm(candidate) in bridge_norm:
                        kw.append(candidate)
                kw = list(dict.fromkeys(kw))
                if cur_e and nxt_e:
                    res["emotion_transitions"].append({
                        "from_sentence": cur,
                        "to_sentence": nxt,
                        "from_emotions": cur_e,
                        "to_emotions": nxt_e,
                        "transition_type": self._determine_transition_type(cur_e, nxt_e),
                        "confidence": self._calculate_sentence_transition_confidence(cur_e, nxt_e, cur, nxt),
                        "time_context": kw
                    })
            res["progression_patterns"].extend(self._analyze_intensity_patterns(text, emotions_data))
            return res
        except Exception:
            logger.exception("동적 감정 표현 분석 중 오류")
            return {"emotion_transitions": [], "progression_patterns": []}

    def _calculate_sentence_transition_confidence(self, from_emotions, to_emotions, from_sentence, to_sentence) -> float:
        try:
            base_default = 0.5
            try:
                base = float(self.scfg.get("transition_base_score", base_default)) if hasattr(self, "scfg") else base_default
            except (TypeError, ValueError):
                base = base_default
            if not from_emotions or not to_emotions:
                return 0.3
            fmax = max(e.get("intensity", 0.0) for e in from_emotions)
            tmax = max(e.get("intensity", 0.0) for e in to_emotions)
            if abs(tmax - fmax) > 0.3:
                base += 0.1
            polarity_cfg = self.scfg.get("polarity_categories", {}) if hasattr(self, "scfg") else {}
            pos_set = {self._norm(p) for p in (polarity_cfg.get("pos_main") or ["희", "락"])}
            neg_set = {self._norm(n) for n in (polarity_cfg.get("neg_main") or ["노", "애"])}
            fset = {self._norm(e.get("category")) for e in from_emotions}
            tset = {self._norm(e.get("category")) for e in to_emotions}
            if (fset & pos_set and tset & neg_set) or (fset & neg_set and tset & pos_set):
                base += 0.1
            if len(to_sentence) > len(from_sentence) * 1.5:
                base += 0.05
            return min(base, 1.0)
        except Exception:
            logger.exception("문장 전이 신뢰도 계산 중 오류")
            return 0.5

    def _extract_bridging_text(self, full_text: str, current: str, next_sent: str, window: int = 30) -> str:
        try:
            cp = full_text.find(current)
            np = full_text.find(next_sent, cp + len(current))
            if cp == -1 or np == -1:
                return ""
            a = cp + len(current)
            b = min(np, a + window)
            return full_text[a:b].strip()
        except Exception:
            logger.exception("문장 사이 텍스트 추출 중 오류")
            return ""

    # ----------------------------
    # Expansion
    # ----------------------------
    def analyze_emotional_expansion(self, text: str, emotions_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            t = self._norm(text)
            out = {"expanded_emotions": [], "intensity_progression": [], "context_expansion": [], "uncovered_emotions": []}
            mods = self._find_intensity_modifiers(t, emotions_data)
            has_hi = any(m["modifier"] in ["매우", "정말"] for m in mods)
            for cat, cat_data in emotions_data.items():
                subs = self._get_sub_emotions_dict(cat_data)
                for sub_name, sd in subs.items():
                    if self._is_strongly_relevant(t, cat, sub_name, sd):
                        rel = (sd.get("emotion_profile", {}) or {}).get("related_emotions", {}) or {}
                        for rtype, lst in rel.items():
                            for emo in lst or []:
                                if self._check_emotion_relevance(t, emo):
                                    conf = self._calculate_relation_confidence(t, emo)
                                    if conf > 0.7:
                                        out["expanded_emotions"].append({
                                            "base_emotion": {"category": cat, "sub_emotion": sub_name},
                                            "related_emotion": emo,
                                            "relation_type": rtype,
                                            "confidence": conf,
                                            "context": self._extract_emotion_context(t, emo)
                                        })
                        il = (sd.get("emotion_profile", {}) or {}).get("intensity_levels", {}) or {}
                        iex = il.get("intensity_examples", {}) or {}
                        for level, exs in (iex.items() if isinstance(iex, dict) else []):
                            for ex in exs or []:
                                if self._norm(ex) in t:
                                    adj_level = "high" if has_hi else level
                                    base_by_level = {"high": 0.7, "medium": 0.5, "low": 0.3}
                                    match_stub = {
                                        "intensity_base": base_by_level.get(adj_level, 0.5),
                                        "level": adj_level,
                                        "example": ex,
                                        "emotion_category": cat,
                                        "sub_emotion": sub_name,
                                    }
                                    out["intensity_progression"].append({
                                        "emotion_category": cat,
                                        "sub_emotion": sub_name,
                                        "level": adj_level,
                                        "example": ex,
                                        "modifiers": mods,
                                        "confidence": self._calculate_intensity_confidence(match_stub, text=t),
                                        "context": self._extract_emotion_context(t, ex)
                                    })
                        sits = (sd.get("context_patterns", {}) or {}).get("situations", {}) or {}
                        for sk, sdata in sits.items():
                            if self._check_situation_relevance(t, sdata):
                                out["context_expansion"].append({
                                    "emotion_category": cat,
                                    "sub_emotion": sub_name,
                                    "situation": sk,
                                    "description": sdata.get("description", ""),
                                    "keywords": sdata.get("keywords", []),
                                    "examples": sdata.get("examples", []),
                                    "confidence": self._calculate_context_confidence(t, sdata),
                                    "emotion_progression": sdata.get("emotion_progression", {}),
                                    "variations": sdata.get("variations", [])
                                })
            self._verify_all_sub_emotions(t, emotions_data, out)
            out["expanded_emotions"].sort(key=lambda x: x["confidence"], reverse=True)
            out["intensity_progression"].sort(key=lambda x: {"high": 3, "medium": 2, "low": 1}.get(x["level"], 0), reverse=True)
            out["context_expansion"].sort(key=lambda x: x["confidence"], reverse=True)
            return out
        except Exception:
            logger.exception("감정 확장성 분석 중 오류")
            return {"expanded_emotions": [], "intensity_progression": [], "context_expansion": [], "uncovered_emotions": []}

    def _verify_all_sub_emotions(self, text: str, emotions_data: Dict[str, Any], expansion_result: Dict[str, Any]) -> None:
        try:
            covered: Set[Tuple[str, str]] = set()
            for e in expansion_result.get("expanded_emotions", []):
                covered.add((e["base_emotion"]["category"], e["base_emotion"]["sub_emotion"]))
            for p in expansion_result.get("intensity_progression", []):
                covered.add((p["emotion_category"], p["sub_emotion"]))
            for c in expansion_result.get("context_expansion", []):
                covered.add((c["emotion_category"], c["sub_emotion"]))
            for cat, cat_data in emotions_data.items():
                subs = self._get_sub_emotions_dict(cat_data)
                for sub in subs.keys():
                    if (cat, sub) not in covered:
                        expansion_result["uncovered_emotions"].append({
                            "category": cat,
                            "sub_emotion": sub,
                            "reason": "No match found in text"
                        })
        except Exception:
            logger.exception("sub_emotion 누락 확인 중 오류")

    # ----------------------------
    # Weighted scores
    # ----------------------------
    def _calculate_weighted_scores(
        self,
        matched_phrases: List[Dict[str, Any]],
        sentiment_patterns: List[Dict[str, Any]],
        modifier_effects: Dict[str, Any],
        intensity_patterns: List[Dict[str, Any]],
        emotions_data: Dict[str, Any],
    ) -> Dict[str, float]:
        try:
            category_stats, avg_vocab = self._get_category_stats(emotions_data)
            pattern_stats = self._get_pattern_statistics(emotions_data)
            conf_adj = self.scfg.get("confidence_adjustment", {}) if hasattr(self, "scfg") else {}
            pos_tokens_cfg = [self._norm(t) for t in (self.scfg.get("pos_indicators", []) if hasattr(self, "scfg") else [])]
            neg_tokens_cfg = [self._norm(t) for t in (self.scfg.get("neg_indicators", []) if hasattr(self, "scfg") else [])]
            evidence_map: Dict[str, List[float]] = defaultdict(list)

            def _add_evidence(cat: Optional[str], sub_name: Optional[str], pattern: Optional[str], confidence: float, base_weight: float = 1.0) -> None:
                if not cat:
                    return
                cat_key = str(cat)
                stats = category_stats.get(cat_key, {"vocab": 1, "sub_count": 1})
                vocab_size = max(int(stats.get("vocab", 1)), 1)
                hit = max(float(confidence), 0.1)
                tf = hit / math.log(vocab_size + 2.0)
                token_len = max(len((pattern or "").split()), 1)
                length_norm = 1.0 / math.log(token_len + 2.0)
                idf = math.log((avg_vocab + 1.0) / (vocab_size + 1.0)) + 1.0
                log_weight = 1.0
                if pattern:
                    sub_meta = self._get_sub_meta(cat_key, sub_name or "", emotions_data)
                    sub_identifier = str(sub_meta.get("emotion_id") or sub_name or cat_key)
                    stats_for_sub = pattern_stats.get(sub_identifier)
                    if isinstance(stats_for_sub, dict):
                        entry = stats_for_sub.get(pattern)
                        if entry is None:
                            entry = stats_for_sub.get(self._norm(pattern))
                        if isinstance(entry, dict):
                            pos = float(entry.get("df_pos", entry.get("pos", entry.get("positive", 0.0))) or 0.0)
                            neg = float(entry.get("df_neg", entry.get("neg", entry.get("negative", 0.0))) or 0.0)
                            log_odds = math.log((pos + 0.5) / (neg + 0.5))
                            log_weight = max(0.2, min(1.8, 1.0 + (log_odds / 4.0)))
                score = base_weight * tf * idf * length_norm * log_weight
                evidence_map[cat_key].append(score)

            for ph in matched_phrases:
                _add_evidence(
                    ph.get("emotion_category"),
                    ph.get("sub_emotion"),
                    ph.get("pattern"),
                    float(ph.get("confidence", 0.5))
                )

            for ip in intensity_patterns:
                level = str(ip.get("intensity_level", "medium")).lower()
                base_weight = 1.0
                if level == "high":
                    base_weight = 1.2
                elif level == "low":
                    base_weight = 0.9
                _add_evidence(
                    ip.get("emotion_category"),
                    ip.get("sub_emotion"),
                    ip.get("example"),
                    float(ip.get("confidence", 0.6)),
                    base_weight=base_weight
                )

            ws: Dict[str, float] = defaultdict(float)
            for cat, scores in evidence_map.items():
                if scores:
                    ws[cat] = sum(scores)

            for cat_key in category_stats.keys():
                ws.setdefault(cat_key, 0.0)

            if ws:
                mean_score = sum(ws.values()) / max(len(ws), 1)
                balance_lambda = float(self.scfg.get("balance_lambda", 0.15)) if hasattr(self, "scfg") else 0.15
                for cat in list(ws.keys()):
                    ws[cat] = ((1 - balance_lambda) * ws[cat]) + (balance_lambda * mean_score)

            total_effect = float(modifier_effects.get("total_effect", 1.0))
            amp_cnt = len(modifier_effects.get("amplifiers", []) or [])
            dim_cnt = len(modifier_effects.get("diminishers", []) or [])
            total_effect += 0.02 * amp_cnt
            total_effect -= 0.01 * dim_cnt
            total_effect = max(total_effect, 0.5)
            for cat in list(ws.keys()):
                ws[cat] *= total_effect

            for sp in sentiment_patterns:
                cat = str(sp.get("emotion_category"))
                conf = float(sp.get("confidence", 0.4))
                relation = str(sp.get("relation_type", "")).lower()
                delta = 0.0
                if relation == "positive":
                    delta = conf * 0.05
                elif relation == "negative":
                    delta = -conf * 0.05
                ws[cat] += delta

            penalty_mixed = float(conf_adj.get("penalty_mixed_polar", 0.2)) if isinstance(conf_adj, dict) else 0.2
            combined_text = " ".join(str(p.get("pattern", "")) for p in matched_phrases)
            text_conflict = False
            if combined_text.strip():
                combined_norm = self._norm(combined_text)
                text_conflict = self._has_token(combined_norm, pos_tokens_cfg) and self._has_token(combined_norm, neg_tokens_cfg)
            polarity_cfg = self.scfg.get("polarity_categories", {}) if hasattr(self, "scfg") else {}
            pos_main_set = set(polarity_cfg.get("pos_main", [])) or {"희", "락"}
            neg_main_set = set(polarity_cfg.get("neg_main", [])) or {"노", "애"}
            match_cats = {str(p.get("main_emotion") or p.get("emotion_category")) for p in matched_phrases}
            cat_conflict = bool(match_cats & pos_main_set) and bool(match_cats & neg_main_set)
            if text_conflict or cat_conflict:
                for cat in list(ws.keys()):
                    ws[cat] *= max(0.0, 1.0 - penalty_mixed)

            meta = emotions_data.get("ml_training_metadata") if isinstance(emotions_data, dict) else None
            if isinstance(meta, dict):
                class_counts = meta.get("class_counts") or {}
                if isinstance(class_counts, dict) and class_counts:
                    alpha = 1.0
                    total = sum(float(v) for v in class_counts.values())
                    classes = [str(k) for k in class_counts.keys()]
                    uniform = 1.0 / max(len(classes), 1)
                    for cat in list(ws.keys()):
                        count = float(class_counts.get(cat, 0.0))
                        prior = (count + alpha) / (total + alpha * len(classes))
                        log_bias = math.log(max(prior, 1e-6) / max(uniform, 1e-6))
                        ws[cat] += 0.5 * log_bias


            polarity_cfg = self.scfg.get("polarity_categories", {}) if hasattr(self, "scfg") else {}
            pos = set(polarity_cfg.get("positive", []) or ["희", "락"])
            neg = set(polarity_cfg.get("negative", []) or ["노", "애"])
            pos_sum = sum(ws.get(cat, 0.0) for cat in pos)
            neg_sum = sum(ws.get(cat, 0.0) for cat in neg)
            if pos_sum > 0 and neg_sum > 0:
                min_side = min(pos_sum, neg_sum)
                mix_factor = 0.8 if min_side >= 0.1 else 0.9
                for cat in list(ws.keys()):
                    ws[cat] *= mix_factor

            final: Dict[str, float] = {}
            for cat, value in ws.items():
                value = max(0.0, value)
                if value < 0.05:
                    continue
                final[cat] = round(value, 3)
            return final
        except Exception as exc:
            self._log_exc("가중 점수 계산", exc)
            return {}

    def _analyze_intensity_patterns(self, text: str, emotions_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        try:
            out = []
            t = self._norm(text)
            mods = self.scfg.get("intensity_modifiers", {}).get("all", [])
            found = []
            for m in mods:
                nm = self._norm(m)
                if nm in t:
                    pos = t.find(nm)
                    ctx = t[max(0, pos - 10):min(len(t), pos + 10)]
                    found.append({"modifier": m, "position": pos, "context": ctx})
            time_occ, _ = self._count_keyword_hits(text, self.scfg.get("time_keywords", []))
            sents = self._split_sentences_ko(t)
            for i, sent in enumerate(sents):
                local_mod = [m for m in found if self._norm(m["modifier"]) in sent]
                if not local_mod:
                    continue
                for cat, cat_data in emotions_data.items():
                    subs = self._get_sub_emotions_dict(cat_data)
                    for sub_name, sd in subs.items():
                        if self._is_strongly_relevant(sent, cat, sub_name, sd):
                            base_level = "medium"
                            adj = "high"
                            boost = 0.1 if time_occ == 2 else (0.2 if time_occ >= 3 else 0.0)
                            conf = min(0.9 + boost, 1.0)
                            before = sent.split(self._norm(local_mod[0]["modifier"]))[0] if self._norm(local_mod[0]["modifier"]) in sent else ""
                            after  = sent.split(self._norm(local_mod[0]["modifier"]))[1] if self._norm(local_mod[0]["modifier"]) in sent else ""
                            out.append({
                                "type": "intensity_change",
                                "modifiers": [m["modifier"] for m in local_mod],
                                "base_intensity": base_level,
                                "adjusted_intensity": adj,
                                "confidence": conf,
                                "emotion_category": cat,
                                "sub_emotion": sub_name,
                                "context_info": {"before_modifier": before.strip(), "after_modifier": after.strip()},
                                "time_occurrences": time_occ
                            })
            return out
        except Exception:
            logger.exception("강도 변화 패턴 분석 중 오류")
            return []

    def _match_intensity_patterns(self, text: str, emotions_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        try:
            out = []
            t = self._norm(text)
            time_occ, _ = self._count_keyword_hits(text, self.scfg.get("time_keywords", []))
            has_hi = any(self._norm(m) in t for m in ["매우", "정말"])
            for cat, cat_data in emotions_data.items():
                subs = self._get_sub_emotions_dict(cat_data)
                for sub_name, sd in subs.items():
                    il = (sd.get("emotion_profile", {}) or {}).get("intensity_levels", {}) or {}
                    iex = il.get("intensity_examples", {}) or {}
                    poss = []
                    sits = (sd.get("context_patterns", {}) or {}).get("situations", {}) or {}
                    for st in sits.values():
                        poss.append(st.get("emotion_progression", {}) or {})
                    for level, exs in (iex.items() if isinstance(iex, dict) else []):
                        for ex in exs or []:
                            if self._norm(ex) in t:
                                final = "high" if has_hi else level
                                final = self._adjust_level_with_time_occurrences(final, time_occ)
                                out.append({
                                    "emotion_category": cat,
                                    "sub_emotion": sub_name,
                                    "intensity_level": final,
                                    "example": ex,
                                    "confidence": 1.0 if final == "high" else 0.8,
                                    "time_occurrences": time_occ
                                })
                    for ep in poss:
                        peak = ep.get("peak", "")
                        if peak and self._norm(peak) in t:
                            out.append({
                                "emotion_category": cat,
                                "sub_emotion": sub_name,
                                "intensity_level": "high",
                                "example": peak,
                                "confidence": 0.9,
                                "time_occurrences": time_occ
                            })
            return out
        except Exception:
            logger.exception("감정 강도 패턴 매칭 중 오류")
            return []

    def _adjust_level_with_time_occurrences(self, base_level: str, time_occurrences: int) -> str:
        try:
            mapv = {"low": 0, "medium": 1, "high": 2}
            inv = {0: "low", 1: "medium", 2: "high"}
            cur = mapv.get(base_level, 1)
            if time_occurrences >= 1:
                cur = min(cur + 1, 2)
            if time_occurrences >= 2:
                cur = min(cur + 1, 2)
            return inv.get(cur, base_level)
        except Exception:
            logger.exception("시간 키워드 기반 강도 레벨 조정 중 오류")
            return base_level

    def _adjust_level_with_time_keywords(self, current_level: str, time_keywords: List[str]) -> str:
        try:
            mapv = {"low": 0, "medium": 1, "high": 2}
            inv = {0: "low", 1: "medium", 2: "high"}
            cur = mapv.get(current_level, 1)
            if time_keywords and cur < 2:
                cur += 1
            return inv.get(cur, current_level)
        except Exception:
            logger.exception("시간 키워드에 따른 강도 조정 중 오류")
            return current_level

    # ----------------------------
    # Sentence-level emotions
    # ----------------------------
    def _analyze_sentence_emotions(self, sentence: str, emotions_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        try:
            emotions = []
            subsents = list(_split_sentences_cached(sentence))
            neg_conj = set()
            pos_conj = set()
            for _, ed in emotions_data.items():
                kp = self._get_ling_patterns(ed).get("key_phrases", []) or []
                for info in kp:
                    patt = info.get("pattern", "")
                    w = float(info.get("weight", 0.5))
                    ctx = info.get("context_requirement", "")
                    if w >= 0.7 and "긍정" in ctx:
                        pos_conj.add(patt)
                    elif w <= 0.4 and "부정" in ctx:
                        neg_conj.add(patt)

            for ss in subsents:
                for cat, ed in emotions_data.items():
                    subs = self._get_sub_emotions_dict(ed)
                    for sub_name, sd in subs.items():
                        cp = (sd.get("context_patterns", {}) or {})
                        sits = cp.get("situations", {}) or {}

                        situation_match = False
                        matched_sit = None
                        for sk, sd2 in sits.items():
                            desc = sd2.get("description", "")
                            if desc and self._norm(desc) in self._norm(ss):
                                situation_match = True
                                matched_sit = sk
                                break
                            kws = sd2.get("keywords", []) or []
                            if any(self._norm(kw) in self._norm(ss) for kw in kws):
                                situation_match = True
                                matched_sit = sk
                                break
                            exs = sd2.get("examples", []) or []
                            if any(self._norm(ex) in self._norm(ss) for ex in exs):
                                situation_match = True
                                matched_sit = sk
                                break
                        if not situation_match:
                            continue

                        ling = self._get_ling_patterns(sd)
                        key_phrases = ling.get("key_phrases", []) or []
                        phrase_matches = []
                        for info in key_phrases:
                            patt = info.get("pattern", "")
                            if patt and self._norm(patt) in self._norm(ss):
                                phrase_matches.append({"pattern": patt, "weight": info.get("weight", 0.5)})

                        combos = ling.get("sentiment_combinations", []) or []
                        combo_matches = []
                        for combo in combos:
                            words = combo.get("words", []) or []
                            if words and all(self._norm(w) in self._norm(ss) for w in words):
                                combo_matches.append({"words": words, "weight": combo.get("weight", 0.5)})

                        intensity = self._calculate_emotion_intensity(ss, sd)
                        sm = ling.get("sentiment_modifiers", {}) or {}
                        if any(self._norm(a) in self._norm(ss) for a in sm.get("amplifiers", []) or []):
                            intensity *= 1.2
                        if any(self._norm(d) in self._norm(ss) for d in sm.get("diminishers", []) or []):
                            intensity *= 0.8

                        base_conf = self._calculate_confidence(ss, sub_name, ed)
                        if situation_match:
                            base_conf += 0.2
                        if phrase_matches:
                            base_conf += max(p["weight"] for p in phrase_matches) * 0.15
                        if combo_matches:
                            base_conf += max(c["weight"] for c in combo_matches) * 0.1

                        rel = (sd.get("emotion_profile", {}) or {}).get("related_emotions", {}) or {}
                        is_pos = any(self._norm(e) in self._norm(ss) for e in rel.get("positive", []) or [])
                        is_neg = any(self._norm(e) in self._norm(ss) for e in rel.get("negative", []) or [])

                        if cat in ["노", "애"] and is_pos:
                            if any(self._norm(x) in self._norm(ss) for x in neg_conj):
                                base_conf -= 0.1
                            else:
                                base_conf -= 0.25
                        if cat in ["희", "락"] and is_neg:
                            if any(self._norm(x) in self._norm(ss) for x in pos_conj):
                                base_conf -= 0.1
                            else:
                                base_conf -= 0.25

                        final_conf = max(0.0, min(1.0, base_conf))
                        if final_conf >= 0.3:
                            emotions.append({
                                "category": cat,
                                "sub_emotion": sub_name,
                                "intensity": min(max(float(intensity), 0.0), 1.0),
                                "confidence": final_conf,
                                "is_positive": is_pos,
                                "is_negative": is_neg,
                                "matched_situation": matched_sit,
                                "phrase_matches": phrase_matches,
                                "combo_matches": combo_matches,
                                "sub_sentence": ss
                            })

            uniq = []
            seen = set()
            for e in sorted(emotions, key=lambda x: x["confidence"], reverse=True):
                key = (e["category"], e["sub_emotion"])
                if key not in seen:
                    seen.add(key)
                    uniq.append(e)
                    if len(uniq) >= 3:
                        break
            return uniq
        except Exception:
            logger.exception("문장별 감정 분석 중 오류")
            return []

    # ----------------------------
    # Contextual meaning
    # ----------------------------
    def _analyze_emotional_context(self, text: str, emotions_data: Dict[str, Any]) -> Dict[str, float]:
        """
        텍스트의 감정적 문맥을 분석하여 각 감정 카테고리별 점수를 계산.
        - 한글 경계 친화 매칭(self._norm + CacheManager) 사용
        - 긍/부 단어 감지 기반 극성 보정(한쪽만 강하면 해당 극성 강화)
        - 정규화 임계 0.05로 완화(희 일변도 완화)
        - 전부 0이거나 소거되면 간단한 타이브레이크(긍정/부정 단어 카운트) 적용
        """
        try:
            nt = self._norm(text)

            # 1) 초기 스코어 테이블 구성 (대표 4감정만 대상으로 안전화)
            target_cats = [c for c in ["희", "노", "애", "락"] if c in emotions_data]
            if not target_cats:
                target_cats = [k for k, v in emotions_data.items()
                               if isinstance(v, dict) and any(
                        x in v for x in ("emotion_profile", "sub_emotions", "metadata"))]
                target_cats = [c for c in target_cats if c in {"희", "노", "애", "락"}] or ["희", "노", "애", "락"]
            context_scores = {cat: 0.0 for cat in target_cats}
            score_changes = {cat: [] for cat in target_cats}

            # 2) 가중치·표현 사전

            weights_cfg = self.scfg.get("context_weights", {}) if hasattr(self, "scfg") else {}
            WEIGHTS = {
                "CORE_KEYWORD": float(weights_cfg.get("core_keyword", 0.15)),
                "SITUATION": float(weights_cfg.get("situation", 0.12)),
                "EXAMPLE": float(weights_cfg.get("example", 0.08)),
                "INTENSITY": float(weights_cfg.get("intensity", 0.05)),
                "EMOTION_INTERACTION": float(weights_cfg.get("interaction", 0.10)),
            }
            if hasattr(self, "global_intensity_mods"):
                mods = (self.global_intensity_mods.get("amplifiers", []) or []) + (self.global_intensity_mods.get("diminishers", []) or [])
            else:
                mods = self.scfg.get("intensity_modifiers", {}).get("all", [])
            normalized_mods = {self._norm(m) for m in mods if m}
            INTENSITY_MODIFIERS = list(normalized_mods) or [
                "매우", "정말", "너무", "가장", "굉장히", "완전", "진짜", "엄청", "아주", "한없이", "심히", "상당히", "몹시", "되게", "무척",
            ]
            positive_indicators = {self._norm(p) for p in getattr(self, "global_positives", set()) if p} or {
                "즐거운", "행복한", "기쁜", "만족스러운", "흐뭇한", "뿌듯한", "황홀한", "평온한", "신나는", "희망찬", "감사한", "유쾌한", "아늑한", "멋진", "달콤한",
            }
            negative_indicators = {self._norm(n) for n in getattr(self, "global_negatives", set()) if n} or {
                "불안한", "두려운", "슬픈", "불만족스러운", "괴로운", "절망스러운", "불행한", "괴씸한", "무서운", "심란한", "울적한", "씁쓸한", "참담한", "우울한", "비통한",
            }
            # 3) 감정군 전체를 순회하면서 점수 반영
            for category in target_cats:
                cdata = emotions_data.get(category, {}) or {}
                sub_emotions = self._get_sub_emotions_dict(cdata)
                c_score = 0.0

                for _, sd in (sub_emotions.items() if isinstance(sub_emotions, dict) else []):
                    # 3-1) 핵심 키워드
                    core = self._get_core_keywords(sd)
                    kw_matches = 0
                    for kw in core or []:
                        pat = CacheManager.get_keyword_pattern(kw, whole_word=True, case_insensitive=True)
                        if pat and pat.search(nt):
                            kw_matches += 1
                    s_change = kw_matches * WEIGHTS["CORE_KEYWORD"]

                    # 3-2) 상황 패턴(키워드/예시/트리거)
                    sits = (sd.get("context_patterns", {}) or {}).get("situations", {}) or {}
                    s_score = 0.0
                    for st in sits.values():
                        # 키워드
                        for kw in st.get("keywords", []) or []:
                            pat = CacheManager.get_keyword_pattern(kw, whole_word=True, case_insensitive=True)
                            if pat and pat.search(nt):
                                s_score += WEIGHTS["SITUATION"]
                        # 예시
                        for ex in st.get("examples", []) or []:
                            pat = CacheManager.get_keyword_pattern(ex, whole_word=False, case_insensitive=True)
                            if pat and pat.search(nt):
                                s_score += WEIGHTS["EXAMPLE"]
                        # 트리거(진행 서술)
                        prog = st.get("emotion_progression", {}) or {}
                        trig = prog.get("trigger", "")
                        if trig:
                            pat = CacheManager.get_keyword_pattern(trig, whole_word=False, case_insensitive=True)
                            if pat and pat.search(nt):
                                s_score += 0.10

                    total = s_change + s_score
                    if total > 0:
                        score_changes[category].append(total)
                    c_score += total

                # 3-3) 증폭 수식어 개수에 따른 승수 보정
                im_count = 0
                for m in INTENSITY_MODIFIERS:
                    if CacheManager.get_keyword_pattern(m, whole_word=True, case_insensitive=True).search(nt):
                        im_count += 1
                if im_count > 0:
                    c_score *= (1 + WEIGHTS["INTENSITY"] * im_count)

                context_scores[category] = c_score

            # 4) 긍/부 단어 감지 및 극성 보정
            found_positive = any(
                CacheManager.get_keyword_pattern(w, True, True).search(nt) for w in positive_indicators)
            found_negative = any(
                CacheManager.get_keyword_pattern(w, True, True).search(nt) for w in negative_indicators)

            if found_positive and found_negative:
                for k in context_scores:
                    context_scores[k] *= 0.8
            elif found_positive and not found_negative:
                for k in context_scores:
                    if k in {"희", "락"}:
                        context_scores[k] *= 1.10
                    else:
                        context_scores[k] *= 0.95
            elif found_negative and not found_positive:
                for k in context_scores:
                    if k in {"노", "애"}:
                        context_scores[k] *= 1.10
                    else:
                        context_scores[k] *= 0.95

            # 5) 감정 간 상호작용(긍정 우세 시+, 부정 우세 시-) 기존 로직 유지
            pos_cats, neg_cats = ["희", "락"], ["노", "애"]
            pos_sum = sum(context_scores.get(c, 0.0) for c in pos_cats)
            neg_sum = sum(context_scores.get(c, 0.0) for c in neg_cats)
            if pos_sum > neg_sum * 2:
                for pc in pos_cats:
                    context_scores[pc] *= (1 + WEIGHTS["EMOTION_INTERACTION"])
                for nc in neg_cats:
                    context_scores[nc] *= (1 - WEIGHTS["EMOTION_INTERACTION"])

            # 로그(유의 변화만)
            self._log_score_changes(score_changes)

            # 6) 정규화(임계 0.05) - self._normalize_scores를 쓰지 않고, 여기서 직접 처리
            for k in list(context_scores.keys()):
                context_scores[k] = min(max(context_scores[k], 0.0), 1.0)
            total = sum(context_scores.values())
            if total > 0:
                normalized = {k: (v / total) for k, v in context_scores.items()}
                # 임계 0.05 이하 제거
                normalized = {k: v for k, v in normalized.items() if v > 0.05}
                if normalized:
                    top_items = heapq.nlargest(4, normalized.items(), key=lambda x: x[1])
                    t2 = sum(v for _, v in top_items)
                    if t2 > 0:
                        normalized = {k: (v / t2) for k, v in top_items}
                    else:
                        normalized = dict(top_items)
            else:
                normalized = {}

            # 7) 타이브레이크: 전부 0이거나 소거되면 간단한 긍/부 편향으로 추정
            if not normalized or all(v == 0.0 for v in normalized.values()):
                neg_bias = sum(text.count(w) for w in ["실망", "짜증", "분노", "불만", "불안", "우울", "슬픔"])
                pos_bias = sum(text.count(w) for w in ["기쁨", "행복", "환희", "만족", "즐거움"])
                if neg_bias > pos_bias:
                    normalized = {"노": 0.5, "애": 0.5}
                elif pos_bias > neg_bias:
                    normalized = {"희": 0.6, "락": 0.4}
                else:
                    normalized = {"희": 0.34, "락": 0.33, "노": 0.16, "애": 0.17}

            return normalized

        except Exception:
            logger.exception("감정 문맥 분석 중 오류 발생")
            return {'희': 0.0, '노': 0.0, '애': 0.0, '락': 0.0}

    # ----------------------------
    # Misc
    # ----------------------------
    def _analyze_overall_patterns(
        self,
        text: str,
        matched_phrases: List[Dict[str, Any]],
        sentiment_patterns: List[Dict[str, Any]],
        modifier_effects: Dict[str, Any],
        intensity_patterns: List[Dict[str, Any]],
        contextual_meaning: Dict[str, Any],
        emotions_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        try:
            pa = {"dominant_patterns": [], "pattern_distribution": {}, "pattern_coherence": 0.0, "context_relevance": {}, "mixed_emotion": False}
            counts = {}
            for ph in matched_phrases:
                cat = ph["emotion_category"]
                counts[cat] = counts.get(cat, 0) + 1
            dom = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:3]
            pa["dominant_patterns"] = dom
            tot = sum(counts.values())
            if tot > 0:
                pa["pattern_distribution"] = {k: v / tot for k, v in counts.items()}
            pa["pattern_coherence"] = self._calculate_pattern_coherence(matched_phrases, sentiment_patterns)
            pa["context_relevance"] = self._analyze_context_relevance(text, matched_phrases, emotions_data)
            pos = {"희", "락"}
            neg = {"노", "애"}
            if any(k in pos for k in counts.keys()) and any(k in neg for k in counts.keys()):
                pa["mixed_emotion"] = True
            return pa
        except Exception:
            logger.exception("전체 패턴 분석 중 오류")
            return {}

    def _check_strong_emotion_presence(self, text: str, emotion_data: Dict[str, Any]) -> bool:
        try:
            t = self._norm(text)
            score = 0.0
            core = self._get_core_keywords(emotion_data)
            score += sum(1 for kw in core if self._norm(kw) in t) * 0.3
            sits = (emotion_data.get("context_patterns", {}) or {}).get("situations", {}) or {}
            for sd in sits.values():
                if self._norm(sd.get("description", "")) in t:
                    score += 0.4
            exs = []
            for sd in sits.values():
                exs.extend(sd.get("examples", []) or [])
            if any(self._norm(ex) in t for ex in exs):
                score += 0.3
            return score > 0.5
        except Exception:
            logger.exception("감정 존재 여부 확인 중 오류")
            return False

    def _check_emotion_presence(self, text: str, emotion_data: Dict[str, Any]) -> bool:
        try:
            t = self._norm(text)
            core = self._get_core_keywords(emotion_data)
            if any(self._norm(kw) in t for kw in core):
                return True
            sits = (emotion_data.get("context_patterns", {}) or {}).get("situations", {}) or {}
            for sd in sits.values():
                kws = sd.get("keywords", []) or []
                if any(self._norm(kw) in t for kw in kws):
                    return True
            return False
        except Exception:
            logger.exception("감정 존재 여부 확인 중 오류")
            return False

    def _determine_transition_type(self, from_emotions: List[Dict[str, Any]], to_emotions: List[Dict[str, Any]]) -> str:
        if not from_emotions or not to_emotions:
            return "unknown"
        fi = max(e["intensity"] for e in from_emotions) if from_emotions else 0
        ti = max(e["intensity"] for e in to_emotions) if to_emotions else 0
        if ti > fi:
            return "intensifying"
        elif ti < fi:
            return "diminishing"
        else:
            return "maintaining"

    def _analyze_intensity_progression(self, text: str, emotion_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            t = self._norm(text)
            mods = self.scfg.get("intensity_modifiers", {}).get("all", [])
            found = []
            for m in mods:
                nm = self._norm(m)
                if nm in t:
                    p = t.find(nm)
                    found.append({"modifier": m, "position": p, "context": t[max(0, p - 10):min(len(t), p + 10)]})
            time_occ, time_hits = self._count_keyword_hits(text, self.scfg.get("time_keywords", []))
            base = "medium"
            if not found and time_occ == 0:
                return None
            mapv = {"low": 0, "medium": 1, "high": 2}
            cur = mapv.get(base, 1)
            if found:
                cur += 1
            if time_occ >= 2:
                cur += 1
            cur = min(cur, 2)
            inv = {0: "low", 1: "medium", 2: "high"}
            adj = inv[cur]
            conf = min(0.8 + 0.05 * min(time_occ, 3), 1.0)
            return {
                "type": "intensity_change",
                "modifiers": found,
                "time_keywords": time_hits,
                "base_intensity": base,
                "adjusted_intensity": adj,
                "confidence": round(conf, 2),
                "context_info": {
                    "before_modifier": (t.split(self._norm(found[0]["modifier"]))[0].strip() if found else ""),
                    "after_modifier": (t.split(self._norm(found[0]["modifier"]))[1].strip() if found else ""),
                },
            }
        except Exception:
            logger.exception("강도 진행 분석 중 오류")
            return None

    def _is_strongly_relevant(self, text: str, emotion_category: str, sub_emotion: str, emotion_data: Dict[str, Any]) -> bool:
        try:
            t = self._norm(text)
            if "만족" in t and sub_emotion == "만족감":
                return True
            core = self._get_core_keywords(emotion_data)
            if core and sum(1 for kw in core if self._norm(kw) in t) >= 2:
                return True
            sits = (emotion_data.get("context_patterns", {}) or {}).get("situations", {}) or {}
            for sd in sits.values():
                desc = self._norm(sd.get("description", ""))
                kws = [self._norm(k) for k in (sd.get("keywords", []) or [])]
                if desc and desc in t and sum(1 for k in kws if k in t) >= 2:
                    return True
            if emotion_category in ["애", "노"]:
                pos_cfg: List[str] = []
                if hasattr(self, "scfg") and isinstance(self.scfg, dict):
                    pos_cfg = [self._norm(x) for x in (self.scfg.get("pos_indicators") or [])]
                if any(ind and ind in t for ind in pos_cfg):
                    return False
            return False
        except Exception:
            logger.exception("문맥 관련성 검사 중 오류")
            return False

    def _extract_context(self, text: str, positions: List[int], pattern_length: int, window: int = 20) -> List[str]:
        try:
            ctx = []
            for pos in positions:
                a = max(0, pos - window)
                b = min(len(text), pos + pattern_length + window)
                ctx.append(text[a:b])
            return ctx
        except Exception:
            logger.exception("문맥 추출 중 오류")
            return []

    def _find_intensity_modifiers(self, text: str, emotions_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        try:
            t = self._norm(text)
            mods = []
            for ed in emotions_data.values():
                sm = self._get_ling_patterns(ed).get("sentiment_modifiers", {}) or {}
                for mtype in ["amplifiers", "diminishers"]:
                    for w in sm.get(mtype, []) or []:
                        if self._norm(w) in t:
                            mods.append({"modifier": w, "position": t.find(self._norm(w)), "type": mtype, "weight": 1.0 if mtype == "amplifiers" else 0.5})
            return mods
        except Exception:
            logger.exception("강도 수정자 찾기 중 오류")
            return []

    def _find_primary_emotions(self, text: str, emotions_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        primary = []
        t = self._norm(text)
        for cat, ed in emotions_data.items():
            subs = self._get_sub_emotions_dict(ed)
            for sub, sd in subs.items():
                if self._is_strongly_relevant(t, cat, sub, sd):
                    conf = self._calculate_confidence(t, sub, ed)
                    if conf > 0.7:
                        primary.append({"category": cat, "sub_emotion": sub, "confidence": conf})
        return sorted(primary, key=lambda x: x["confidence"], reverse=True)[:2]

    def analyze_emotional_transitions(self, text: str, emotions_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        try:
            out = []
            sents = self._split_sentences_ko(text)
            for i in range(len(sents) - 1):
                cur = sents[i]
                nxt = sents[i + 1]
                ce = self._analyze_sentence_emotions(cur, emotions_data)
                ne = self._analyze_sentence_emotions(nxt, emotions_data)
                if ce and ne:
                    out.append({
                        "from_sentence": cur,
                        "to_sentence": nxt,
                        "from_emotions": ce,
                        "to_emotions": ne,
                        "transition_type": self._determine_transition_type(ce, ne),
                    })
            return out
        except Exception:
            logger.exception("감정 전이 분석 중 오류")
            return []

    def _calculate_emotion_intensity(self, text: str, emotion_data: Dict[str, Any]) -> float:
        try:
            t = self._norm(text)
            base = 0.5
            sm = self._get_ling_patterns(emotion_data).get("sentiment_modifiers", {}) or {}
            for a in sm.get("amplifiers", []) or []:
                if self._norm(a) in t:
                    base *= 1.2
            for d in sm.get("diminishers", []) or []:
                if self._norm(d) in t:
                    base *= 0.8
            sits = (emotion_data.get("context_patterns", {}) or {}).get("situations", {}) or {}
            for st in sits.values():
                if st.get("intensity", "medium") == "high" and any(self._norm(k) in t for k in (st.get("keywords", []) or [])):
                    base *= 1.1
            return min(max(base, 0.0), 1.0)
        except Exception:
            logger.exception("감정 강도 계산 중 오류")
            return 0.5

    def _check_situation_relevance(self, text: str, situation_data: Dict[str, Any]) -> bool:
        try:
            t = self._norm(text)
            desc = self._norm(situation_data.get("description", ""))
            if desc and desc in t:
                return True
            kws = [self._norm(k) for k in (situation_data.get("keywords", []) or [])]
            if sum(1 for k in kws if k in t) >= 2:
                return True
            exs = [self._norm(x) for x in (situation_data.get("examples", []) or [])]
            if any(x in t for x in exs):
                return True
            ep = situation_data.get("emotion_progression", {}) or {}
            if sum(1 for v in ep.values() if self._norm(v) in t) > 0:
                return True
            return False
        except Exception:
            logger.exception("상황 관련성 검사 중 오류")
            return False

    def _calculate_relation_confidence(self, text: str, emotion: str) -> float:
        try:
            t = self._norm(text)
            e = self._norm(emotion)
            c = 0.5
            if e in t:
                c += 0.3
            mods_cfg = {}
            gi_mods = getattr(self, "global_intensity_mods", {})
            if isinstance(gi_mods, dict) and gi_mods:
                mods_cfg = gi_mods
            elif hasattr(self, "scfg") and isinstance(self.scfg, dict):
                mods_cfg = self.scfg.get("intensity_modifiers", {}) or {}
            amp_words = [self._norm(w) for w in (mods_cfg.get("amplifiers") or [])]
            if not amp_words:
                amp_words = [self._norm(w) for w in (mods_cfg.get("all") or [])]
            if any(a and a in t for a in amp_words):
                c += 0.2
            words = t.split()
            pos = -1
            for i, w in enumerate(words):
                if e in w:
                    pos = i
                    break
            if pos >= 0:
                start = max(0, pos - 2)
                end = min(len(words), pos + 3)
                ctx = " ".join(words[start:end])
                p_words = {"좋은", "행복한", "즐거운", "만족스러운"}
                n_words = {"나쁜", "불행한", "슬픈", "불만족스러운"}
                if any(self._norm(w) in ctx for w in p_words):
                    c += 0.1
                elif any(self._norm(w) in ctx for w in n_words):
                    c -= 0.1
            return min(max(c, 0.0), 1.0)
        except Exception:
            logger.exception("관련 감정 신뢰도 계산 중 오류")
            return 0.5

    def _calculate_context_confidence(self, text: str, situation_data: Dict[str, Any]) -> float:
        try:
            t = self._norm(text)
            c = 0.5
            desc = self._norm(situation_data.get("description", ""))
            if desc and desc in t:
                c += 0.3
            kws = [self._norm(k) for k in (situation_data.get("keywords", []) or [])]
            km = sum(1 for k in kws if k in t)
            c += min(km * 0.1, 0.3)
            exs = [self._norm(x) for x in (situation_data.get("examples", []) or [])]
            for ex in exs:
                if ex in t:
                    c += 0.2
                    break
            return min(c, 1.0)
        except Exception:
            logger.exception("문맥 신뢰도 계산 중 오류")
            return 0.5

    def _log_score_changes(self, score_changes: Dict[str, List[float]], threshold: float = 0.1):
        try:
            for cat, ch in score_changes.items():
                total = sum(ch)
                if total > threshold:
                    logger.info(f"카테고리 '{cat}' 총 점수 변화: +{total:.2f}")
                else:
                    logger.debug(f"카테고리 '{cat}' 미미한 점수 변화: +{total:.2f}")
        except Exception:
            logger.exception("점수 변화 로깅 중 오류")

    def _normalize_scores(self, scores: Dict[str, float]) -> Dict[str, float]:
        try:
            MIN_TH = 0.1
            for k in list(scores.keys()):
                scores[k] = min(max(scores[k], 0.0), 1.0)
            tot = sum(scores.values())
            if tot > 0:
                norm = {k: v / tot for k, v in scores.items()}
                norm = {k: v for k, v in norm.items() if v > MIN_TH}
                if norm:
                    tot2 = sum(norm.values())
                    norm = {k: v / tot2 for k, v in norm.items()}
                return norm
            return {k: 0.0 for k in scores}
        except Exception:
            logger.exception("점수 정규화 중 오류")
            return scores

    def _filter_duplicates(self, phrases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        try:
            seen = set()
            out = []
            for ph in phrases:
                key = (ph["emotion_category"], ph["pattern"])
                if key not in seen and ph.get("confidence", 0.0) > 0.6:
                    seen.add(key)
                    out.append(ph)
            return out
        except Exception:
            logger.exception("중복 제거 중 오류")
            return phrases

    def _calculate_average_confidence(self, items: List[Dict[str, Any]]) -> float:
        try:
            if not items:
                return 0.0
            return sum(item.get("confidence", 0.0) for item in items) / len(items)
        except Exception:
            logger.exception("평균 신뢰도 계산 중 오류")
            return 0.0

    def _enhanced_filter_matches(self, matches: List[Dict[str, Any]], text: str) -> List[Dict[str, Any]]:
        uniq = {}
        for m in matches:
            if m["type"] == "keyword":
                if len(m["pattern"]) < 4:
                    continue
                if not self._check_strong_context_relevance(text, m["pattern"]):
                    continue
            key = (m["emotion_category"], m["sub_emotion"], m["pattern"])
            conf = self._calculate_adjusted_confidence(m, text)
            if key not in uniq or conf > uniq[key]["confidence"]:
                uniq[key] = m
                uniq[key]["confidence"] = conf
        out = [v for v in uniq.values() if v["confidence"] > 0.7]
        return sorted(out, key=lambda x: x["confidence"], reverse=True)


    def _calculate_adjusted_confidence(self, match: Dict[str, Any], text: str) -> float:
        try:
            thresholds = self.scfg.get("similarity_thresholds", {}) if hasattr(self, "scfg") else {}
            penalties = self.scfg.get("similarity_penalties", {}) if hasattr(self, "scfg") else {}
            adjust_cfg = self.scfg.get("confidence_adjustment", {}) if hasattr(self, "scfg") else {}
            mild_gate = float(thresholds.get("mild", 0.5))
            strong_gate = float(thresholds.get("strong", 0.3))
            mild_penalty = float(penalties.get("mild_penalty", 0.9))
            strong_penalty = float(penalties.get("strong_penalty", 0.8))
            short_len = int(thresholds.get("short_len", 2))
            short_penalty = float(thresholds.get("short_penalty", 0.85))
            base = float(match.get("confidence", 0.5))
            similarity = float(match.get("context_similarity", 1.0))
            if similarity < strong_gate:
                base *= strong_penalty
            elif similarity < mild_gate:
                base *= mild_penalty

            pattern_text = (match.get("pattern") or "").strip()
            token_count = max(len(pattern_text.split()), 1)
            if 0 < token_count <= short_len:
                base *= short_penalty

            len_mode = (self.scfg.get("len_norm", "log") if hasattr(self, "scfg") else "log").lower()
            if len_mode == "sqrt":
                length_factor = min(token_count / 20.0, 0.2)
            elif len_mode == "none":
                length_factor = 0.0
            else:
                length_factor = min(len(pattern_text) / 50.0, 0.2)

            modifiers: List[str] = []
            mods_cfg = getattr(self, 'global_intensity_mods', {}) or {}
            modifiers.extend(mods_cfg.get("amplifiers", []) or [])
            modifiers.extend(mods_cfg.get("diminishers", []) or [])
            modifiers.extend(mods_cfg.get("negators", []) or [])
            if not modifiers and hasattr(self, "scfg"):
                modifiers = self.scfg.get("intensity_modifiers", {}).get("all", []) or []
            normalized_mods = {self._norm(m) for m in modifiers if m}
            norm_text = self._norm(text)
            intensity_example_bonus = float(adjust_cfg.get("intensity_example_bonus", 0.2))
            intensity_general_bonus = float(adjust_cfg.get("intensity_general_bonus", 0.1))
            intensity_bonus = 0.0
            if normalized_mods and any(mod in norm_text for mod in normalized_mods):
                if match.get("type") == "intensity_example":
                    intensity_bonus = intensity_example_bonus
                else:
                    intensity_bonus = intensity_general_bonus

            type_bonus_cfg = adjust_cfg.get("type_bonus", {})
            type_conf = float(type_bonus_cfg.get(match.get("type"), type_bonus_cfg.get("default", 0.1)))
            ctx_bonus_value = float(adjust_cfg.get("context_bonus", 0.1))
            ctx_bonus = ctx_bonus_value if self._check_strong_context_relevance(text, pattern_text) else 0.0

            adjusted = base + length_factor + intensity_bonus + type_conf + ctx_bonus
            max_conf = float(adjust_cfg.get("max_confidence", 1.0))
            min_conf = float(adjust_cfg.get("min_confidence", 0.0))
            return min(max(adjusted, min_conf), max_conf)
        except Exception:
            logger.exception("신뢰도 보정 계산 중 오류")
            return 0.5


    def _check_strong_context_relevance(self, text: str, pattern: str) -> bool:
        if len(pattern) < 4:
            return False
        words = self._norm(text).split()
        pw = self._norm(pattern).split()
        ctx = []
        for i, w in enumerate(words):
            if self._norm(pattern) in w:
                a = max(0, i - 2)
                b = min(len(words), i + 3)
                ctx.extend(words[a:b])
        return len(ctx) >= 3

    def _calculate_pattern_coherence(self, matched_phrases: List[Dict[str, Any]], sentiment_patterns: List[Dict[str, Any]]) -> float:
        try:
            if not matched_phrases and not sentiment_patterns:
                return 0.0
            em = defaultdict(list)
            for ph in matched_phrases:
                em[ph["emotion_category"]].append(ph)
            cs = []
            for arr in em.values():
                if len(arr) > 1:
                    ws = [p.get("confidence", 0.5) for p in arr]
                    var = np.var(ws) if len(ws) > 1 else 0
                    cs.append(1 / (1 + var))
            return float(np.mean(cs)) if cs else 0.0
        except Exception:
            logger.exception("패턴 일관성 계산 중 오류")
            return 0.0

    def _analyze_context_relevance(self, text: str, matched_phrases: List[Dict[str, Any]], emotions_data: Dict[str, Any]) -> Dict[str, float]:
        try:
            cr = defaultdict(float)
            for ph in matched_phrases:
                cat = ph["emotion_category"]
                ctx = ph.get("context", [])
                if isinstance(ctx, list):
                    for c in ctx:
                        sits = (emotions_data.get(cat, {}) or {}).get("context_patterns", {}) or {}
                        sits = sits.get("situations", {}) or {}
                        for sd in sits.values():
                            if self._norm(sd.get("description", "")) in self._norm(c):
                                cr[cat] += 0.1
            for k in list(cr.keys()):
                cr[k] = min(cr[k], 1.0)
            return dict(cr)
        except Exception:
            logger.exception("문맥 연관성 분석 중 오류")
            return {}



# =============================================================================
# LinguisticMatcher
# =============================================================================
class LinguisticMatcher(EmotionProgressionMatcher):
    def __init__(self, emotions_data: Optional[Dict[str, Any]] = None):
        super().__init__()
        # EMOTIONS.json 로드 (하드코딩 대신)
        if emotions_data is None:
            self.emotions_data = self._load_emotions_data()
        else:
            self.emotions_data = emotions_data
            
        self._cached_emotions_id: Optional[int] = None
        self._core_keyword_cache: Dict[str, List[str]] = {}
        self._ling_pattern_cache: Dict[str, Dict[str, Any]] = {}
        self._category_stats_cache: Dict[int, Dict[str, Any]] = {}
        self._sub_meta_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._pattern_stats_cache: Dict[int, Dict[str, Dict[str, Any]]] = {}
        self.scfg: Dict[str, Any] = {}
        self.global_positives: Set[str] = set()
        self.global_negatives: Set[str] = set()
        self.global_intensity_mods: Dict[str, Any] = {}
        
        # EMOTIONS.json 기반 패턴 캐시 초기화
        self._emotion_patterns_cache = {}
        self._linguistic_patterns_cache = {}
        self._load_linguistic_patterns_recursive()
        
        # Dedicated progression helper (composition) for explicit transition analysis
        self.emotion_progression_matcher: EmotionProgressionMatcher = EmotionProgressionMatcher()
        if self.emotions_data:
            self._set_emotions_data_reference(self.emotions_data)

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
            
            # 파일 경로로 로드 시도
            import os
            emotions_file = os.path.join(os.path.dirname(__file__), '..', 'EMOTIONS.json')
            if os.path.exists(emotions_file):
                with open(emotions_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
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

    def _load_linguistic_patterns_recursive(self, emotion_data: Dict[str, Any] = None, path: str = "") -> None:
        """EMOTIONS.json에서 언어학적 패턴을 재귀적으로 로드"""
        if emotion_data is None:
            emotion_data = self.emotions_data
            
        for emotion_key, emotion_info in emotion_data.items():
            current_path = f"{path}.{emotion_key}" if path else emotion_key
            
            # 하위 감정 재귀 처리
            if isinstance(emotion_info, dict):
                if "sub_emotions" in emotion_info:
                    self._load_linguistic_patterns_recursive(emotion_info["sub_emotions"], current_path)
                
                # 언어학적 패턴 추출
                linguistic_patterns = self._extract_linguistic_patterns(emotion_info, current_path)
                if linguistic_patterns:
                    self._linguistic_patterns_cache[current_path] = linguistic_patterns
                    
                # 감정 패턴 추출
                emotion_patterns = self._extract_emotion_patterns(emotion_info, current_path)
                if emotion_patterns:
                    self._emotion_patterns_cache[current_path] = emotion_patterns

    def _extract_linguistic_patterns(self, emotion_info: Dict[str, Any], emotion_path: str) -> Dict[str, Any]:
        """감정 정보에서 언어학적 패턴 추출"""
        patterns = {
            "keywords": [],
            "intensity_modifiers": [],
            "context_markers": [],
            "metaphors": [],
            "expressions": []
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
                            # 강도 수식어 추출
                            if level == "high":
                                patterns["intensity_modifiers"].extend(["매우", "정말", "너무", "완전히"])
                            elif level == "medium":
                                patterns["intensity_modifiers"].extend(["꽤", "상당히", "어느 정도"])
        
        # keywords에서 패턴 추출
        if "keywords" in emotion_info:
            patterns["keywords"].extend(emotion_info["keywords"])
            
        # triggers에서 패턴 추출
        if "triggers" in emotion_info:
            patterns["context_markers"].extend(emotion_info["triggers"])
            
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

    def analyze_linguistic_patterns(self, text: str) -> Dict[str, Any]:
        """EMOTIONS.json 기반 언어학적 패턴 분석"""
        try:
            # 문장 분리
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            if not sentences:
                sentences = [text.strip()]
            
            # 각 문장에서 언어학적 패턴 분석
            sentence_patterns = []
            for i, sentence in enumerate(sentences):
                patterns = self._analyze_sentence_patterns(sentence)
                sentence_patterns.append({
                    "sentence_index": i,
                    "sentence": sentence,
                    "patterns": patterns
                })
            
            # 전체 텍스트 패턴 요약
            summary_patterns = self._summarize_patterns(sentence_patterns)
            
            return {
                "sentence_patterns": sentence_patterns,
                "summary_patterns": summary_patterns,
                "linguistic_analysis": {
                    "total_sentences": len(sentences),
                    "pattern_types_found": len(summary_patterns),
                    "complexity_score": self._calculate_complexity_score(sentence_patterns)
                },
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "success": True
            }
            
        except Exception as e:
            return {
                "sentence_patterns": [],
                "summary_patterns": {},
                "linguistic_analysis": {
                    "total_sentences": 0,
                    "pattern_types_found": 0,
                    "complexity_score": 0.0
                },
                "error": str(e),
                "success": False
            }

    def _analyze_sentence_patterns(self, sentence: str) -> Dict[str, Any]:
        """문장에서 언어학적 패턴 분석"""
        patterns = {
            "detected_emotions": [],
            "intensity_modifiers": [],
            "context_markers": [],
            "metaphors": [],
            "expressions": []
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
        
        # EMOTIONS.json 패턴 매칭
        for emotion_path, linguistic_patterns in self._linguistic_patterns_cache.items():
            # 키워드 매칭
            for keyword in linguistic_patterns.get("keywords", []):
                if keyword.lower() in sentence_lower:
                    patterns["detected_emotions"].append({
                        "emotion": emotion_path,
                        "pattern": keyword,
                        "confidence": 0.8
                    })
                    break
            
            # 강도 수식어 매칭
            for modifier in linguistic_patterns.get("intensity_modifiers", []):
                if modifier in sentence:
                    patterns["intensity_modifiers"].append(modifier)
            
            # 표현 매칭
            for expression in linguistic_patterns.get("expressions", []):
                if isinstance(expression, str) and expression.lower() in sentence_lower:
                    patterns["expressions"].append(expression)
        
        return patterns

    def _summarize_patterns(self, sentence_patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """패턴 요약"""
        summary = {
            "total_emotions": 0,
            "emotion_distribution": {},
            "intensity_modifiers": [],
            "context_markers": [],
            "expressions": []
        }
        
        for sp in sentence_patterns:
            patterns = sp["patterns"]
            summary["total_emotions"] += len(patterns["detected_emotions"])
            
            # 감정 분포
            for emotion_info in patterns["detected_emotions"]:
                emotion = emotion_info["emotion"]
                summary["emotion_distribution"][emotion] = summary["emotion_distribution"].get(emotion, 0) + 1
            
            # 강도 수식어
            summary["intensity_modifiers"].extend(patterns["intensity_modifiers"])
            
            # 표현
            summary["expressions"].extend(patterns["expressions"])
        
        # 중복 제거
        summary["intensity_modifiers"] = list(set(summary["intensity_modifiers"]))
        summary["expressions"] = list(set(summary["expressions"]))
        
        return summary

    def _calculate_complexity_score(self, sentence_patterns: List[Dict[str, Any]]) -> float:
        """복잡도 점수 계산"""
        if not sentence_patterns:
            return 0.0
        
        total_patterns = sum(len(sp["patterns"]["detected_emotions"]) for sp in sentence_patterns)
        avg_patterns_per_sentence = total_patterns / len(sentence_patterns)
        
        # 복잡도 점수 (0.0 ~ 1.0)
        complexity = min(1.0, avg_patterns_per_sentence / 3.0)
        return round(complexity, 2)

# =============================================================================
# EmotionalAnalyzer
# =============================================================================
class EmotionalAnalyzer:
    def __init__(self, emotions_data: Optional[Dict[str, Any]] = None):
        self.emotions_data = emotions_data or {}
        self._epm = EmotionProgressionMatcher()
        logger.info("개선된 EmotionalAnalyzer 초기화 완료")

    # ----------------------------
    # Helpers
    # ----------------------------
    def set_emotions_data(self, emotions_data: Dict[str, Any]):
        self.emotions_data = emotions_data or {}

    def _norm(self, s: str) -> str:
        return normalize(s)

    def _get_sub_emotions_dict(self, cat_data: Dict[str, Any]) -> Dict[str, Any]:
        se = cat_data.get("sub_emotions", {})
        if isinstance(se, dict) and se:
            return se
        se2 = (cat_data.get("emotion_profile", {}) or {}).get("sub_emotions", {})
        if isinstance(se2, dict) and se2:
            return se2
        return {}

    def _get_ling_patterns(self, scope: Dict[str, Any]) -> Dict[str, Any]:
        lp = scope.get("linguistic_patterns", {})
        if isinstance(lp, dict) and lp:
            return lp
        ep = scope.get("emotion_profile", {}) or {}
        return ep.get("linguistic_patterns", {}) or {}

    def _get_core_keywords(self, scope: Dict[str, Any]) -> List[str]:
        ep = scope.get("emotion_profile", {}) or {}
        kws = ep.get("core_keywords", [])
        if kws:
            return kws
        return scope.get("core_keywords", []) or []

    def _split_sentences_ko(self, text: str) -> List[str]:
        if not text:
            return []
        return list(_split_sentences_cached(text))


    def _span_sentences(self, text: str, sents: List[str]) -> List[Tuple[int, int]]:
        spans: List[Tuple[int, int]] = []
        if not text or not sents:
            return spans
        start = 0
        for s in sents:
            idx = text.find(s, start)
            if idx == -1:
                idx = text.find(s)
            if idx == -1:
                continue
            end = idx + len(s)
            spans.append((idx, end))
            start = end
        return spans

    def _sent_index_at(self, pos: int, spans: List[Tuple[int, int]]) -> Optional[int]:
        for i, (a, b) in enumerate(spans):
            if a <= pos < b:
                return i
        return None

    def _positions_of(self, text: str, phrase: str) -> List[int]:
        nt = self._norm(text)
        ph = self._norm(phrase)
        out: List[int] = []
        pat = CacheManager.get_keyword_pattern(ph, whole_word=False, case_insensitive=True)
        if pat:
            out.extend([m.start() for m in pat.finditer(nt)])
        if not out:
            fuzzy = re.sub(r"\s+", r".{0,2}", re.escape(ph))
            try:
                rgx = re.compile(fuzzy, flags=re.UNICODE | re.IGNORECASE)
                out.extend([m.start() for m in rgx.finditer(nt)])
            except re.error:
                pass
        out.sort()
        dedup: List[int] = []
        for p in out:
            if not dedup or abs(p - dedup[-1]) > 2:
                dedup.append(p)
        return dedup

    def _count_hits(self, text: str, items: Iterable[str]) -> Tuple[int, List[str]]:
        nt = self._norm(text)
        hits: List[str] = []
        for it in items or []:
            if not it:
                continue
            pat = CacheManager.get_keyword_pattern(it, whole_word=False, case_insensitive=True)
            if pat and pat.search(nt):
                hits.append(it)
        return len(hits), hits

    def _local_window(self, text: str, pos: int, radius: int = 40) -> str:
        a = max(0, pos - radius)
        b = min(len(text), pos + radius)
        return text[a:b]

    # ----------------------------
    # Global sentiment
    # ----------------------------
    def _load_globals(self, emotions_data: Dict[str, Any]) -> Tuple[Set[str], Set[str], Dict[str, List[str]]]:
        sa = emotions_data.get("sentiment_analysis", {}) or {}
        pos = set(sa.get("positive_indicators", []) or [])
        neg = set(sa.get("negative_indicators", []) or [])
        mods = sa.get("intensity_modifiers", {}) or {}
        return pos, neg, mods

    # ----------------------------
    # Main: Enhanced Emotions
    # ----------------------------
    def analyze_enhanced_emotions(self, text: str, emotions_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            logger.info("정교한 감정 분석 (intensity + sentiment + context) 시작")
            nt = self._norm(text)
            pos_ind, neg_ind, global_mods = self._load_globals(emotions_data)
            amps_glob = global_mods.get("amplifiers", []) or []
            dims_glob = global_mods.get("diminishers", []) or []
            negators_glob = global_mods.get("negators", []) or []

            sents = self._split_sentences_ko(nt)
            spans = self._span_sentences(nt, sents)

            tk = self._epm._collect_context_keywords_from_bone(emotions_data, "time")
            lk = self._epm._collect_context_keywords_from_bone(emotions_data, "location")
            time_ctx = self._epm._detect_context_items(nt, tk)
            loc_ctx = self._epm._detect_context_items(nt, lk)

            details: List[Dict[str, Any]] = []

            norm_k = lambda xs: [self._norm(x) for x in (xs or [])]
            EVIDENCE_MIN = float(os.environ.get("LM_EVIDENCE_MIN", "0.12"))
            CONF_MIN = float(os.environ.get("LM_CONF_MIN", "0.55"))
            k_norm_default = float(
                (emotions_data.get("ml_training_metadata", {}) or {}).get("intensity_normalizer_k", 1.2))

            for cat, cat_data in emotions_data.items():
                subs = self._get_sub_emotions_dict(cat_data)
                cat_meta = cat_data.get("metadata", {}) or {}
                cat_id = cat_meta.get("emotion_id", "")
                cat_cplx = cat_meta.get("emotion_complexity", "basic")

                for sub_name, sd in subs.items():
                    sub_meta = sd.get("metadata", {}) or {}
                    sub_id = sub_meta.get("emotion_id", "")
                    sub_cplx = sub_meta.get("emotion_complexity", "basic")

                    core_keywords = self._get_core_keywords(sd)
                    core_hits_cnt, core_hits = self._count_hits(nt, core_keywords)

                    ling = self._get_ling_patterns(sd)
                    key_phrases = ling.get("key_phrases", []) or []
                    phrase_hits = []
                    phrase_bonus = 0.0
                    for kp in key_phrases:
                        patt = kp.get("pattern", "")
                        w = float(kp.get("weight", 0.1))
                        if patt and self._norm(patt) in nt:
                            phrase_hits.append({"pattern": patt, "weight": w})
                            phrase_bonus += min(w, 0.2)

                    combos = (ling.get("sentiment_combinations", []) or [])
                    combo_hits = []
                    for combo in combos:
                        words = norm_k(combo.get("words", []) or [])
                        if words and all(w in nt for w in words):
                            combo_hits.append({"words": combo.get("words", []), "weight": combo.get("weight", 0.5)})

                    sm_sub = ling.get("sentiment_modifiers", {}) or {}
                    amps_sub = sm_sub.get("amplifiers", []) or []
                    dims_sub = sm_sub.get("diminishers", []) or []
                    neg_sub = sm_sub.get("negators", []) or []
                    amp_cnt_glob, amp_list_glob = self._count_hits(nt, amps_glob)
                    dim_cnt_glob, dim_list_glob = self._count_hits(nt, dims_glob)
                    amp_cnt_sub, amp_list_sub = self._count_hits(nt, amps_sub)
                    dim_cnt_sub, dim_list_sub = self._count_hits(nt, dims_sub)
                    neg_cnt_sub, neg_list_sub = self._count_hits(nt, neg_sub)
                    amp_cnt = amp_cnt_glob + amp_cnt_sub
                    dim_cnt = dim_cnt_glob + dim_cnt_sub
                    neg_cnt = neg_cnt_sub

                    il = (sd.get("emotion_profile", {}) or {}).get("intensity_levels", {}) or {}
                    iex = il.get("intensity_examples", {}) or {}
                    level_votes = {"low": 0, "medium": 0, "high": 0}
                    level_evidence = []
                    for level, exs in (iex.items() if isinstance(iex, dict) else []):
                        for ex in exs or []:
                            if self._norm(ex) in nt:
                                level_votes[level] += 1
                                level_evidence.append(ex)
                    matched_level = max(level_votes.items(), key=lambda x: x[1])[0] if any(
                        level_votes.values()) else "none"

                    sits = (sd.get("context_patterns", {}) or {}).get("situations", {}) or {}
                    sit_hits = []
                    var_hits = []
                    ex_hits = []
                    trig_hits = []
                    sit_score = 0.0
                    for sk, sdata in sits.items():
                        desc = sdata.get("description", "")
                        if desc and self._norm(desc) in nt:
                            sit_score += 0.12
                            sit_hits.append(desc)
                        kws = sdata.get("keywords", []) or []
                        kw_cnt, kw_list = self._count_hits(nt, kws)
                        if kw_cnt:
                            sit_score += min(kw_cnt * 0.08, 0.24)
                            sit_hits.extend(kw_list)
                        vrs = sdata.get("variations", []) or []
                        v_cnt, v_list = self._count_hits(nt, vrs)
                        if v_cnt:
                            sit_score += min(v_cnt * 0.05, 0.15)
                            var_hits.extend(v_list)
                        exs = sdata.get("examples", []) or []
                        ex_cnt, ex_list = self._count_hits(nt, exs)
                        if ex_cnt:
                            sit_score += min(ex_cnt * 0.08, 0.24)
                            ex_hits.extend(ex_list)
                        eprog = sdata.get("emotion_progression", {}) or {}
                        for stg, stg_desc in eprog.items():
                            if not isinstance(stg_desc, str):
                                continue
                            if self._norm(stg_desc) in nt:
                                trig_hits.append({"stage": stg, "text": stg_desc})
                                if stg.lower() == "peak":
                                    sit_score += 0.15
                                else:
                                    sit_score += 0.08

                    pos_cnt, pos_list = self._count_hits(nt, pos_ind)
                    neg_cnt_i, neg_list_i = self._count_hits(nt, neg_ind)

                    lex_score = min(core_hits_cnt * 0.10 + phrase_bonus + len(combo_hits) * 0.08, 0.50)
                    mod_score = min(amp_cnt * 0.10 - dim_cnt * 0.05 - neg_cnt_i * 0.04, 0.30)
                    sent_score = min(pos_cnt * 0.03 - neg_cnt_i * 0.03, 0.20)
                    inten_score = 0.0
                    if matched_level == "high":
                        inten_score = 0.25
                    elif matched_level == "medium":
                        inten_score = 0.15
                    elif matched_level == "low":
                        inten_score = 0.08

                    time_bonus = 0.0
                    loc_bonus = 0.0
                    for hit in trig_hits:
                        for pos in self._positions_of(nt, hit["text"]):
                            if self._epm._find_nearest_context(time_ctx, pos, 24):
                                time_bonus += 0.03
                            if self._epm._find_nearest_context(loc_ctx, pos, 24):
                                loc_bonus += 0.03
                    prog_score = min(time_bonus + loc_bonus + len(trig_hits) * 0.05, 0.25)

                    raw_intensity = max(0.0, lex_score + sit_score + inten_score + mod_score + sent_score + prog_score)
                    k_norm = float((emotions_data.get("ml_training_metadata", {}) or {}).get("intensity_normalizer_k",
                                                                                             k_norm_default))
                    intensity = raw_intensity / (raw_intensity + k_norm) if (raw_intensity + k_norm) > 0 else 0.0
                    intensity = max(0.0, min(1.0, intensity))

                    base_conf = 0.40
                    facets = 0
                    for v in [lex_score, sit_score, inten_score, mod_score, sent_score, prog_score]:
                        if v > 0:
                            facets += 1
                    conf = base_conf + 0.10 * min(facets, 3) + 0.60 * intensity
                    conf = max(0.0, min(1.0, conf))

                    # 게이트: 증거/신뢰도 부족 항목 제거
                    positive_signal = lex_score + sit_score + inten_score + max(mod_score, 0.0) + max(sent_score,
                                                                                                      0.0) + prog_score
                    evidence_items = (
                            len(core_hits) + len(phrase_hits) + len(combo_hits) +
                            len(sit_hits) + len(var_hits) + len(ex_hits) + len(trig_hits)
                    )
                    if not (
                            (positive_signal >= EVIDENCE_MIN) and
                            (evidence_items >= 1 or matched_level != "none") and
                            (conf >= CONF_MIN)
                    ):
                        continue

                    details.append({
                        "emotion_category": cat,
                        "sub_emotion": sub_name,
                        "sub_emotion_id": sub_id,
                        "complexity": sub_cplx,
                        "matched_intensity_level": matched_level,
                        "calculated_intensity": round(float(intensity), 3),
                        "confidence": round(float(conf), 3),
                        "evidence": {
                            "core_hits": core_hits,
                            "phrase_hits": phrase_hits,
                            "combo_hits": combo_hits,
                            "situation_hits": list(set(sit_hits)),
                            "variation_hits": list(set(var_hits)),
                            "example_hits": list(set(ex_hits)),
                            "progression_triggers": trig_hits,
                            "positive_hits": pos_list,
                            "negative_hits": neg_list_i,
                            "amplifiers": list(set(amp_list_glob + amp_list_sub)),
                            "diminishers": list(set(dim_list_glob + dim_list_sub)),
                            "negators": neg_list_sub,
                            "scores": {
                                "lexical": round(lex_score, 3),
                                "situation": round(sit_score, 3),
                                "intensity_level": round(inten_score, 3),
                                "modifiers": round(mod_score, 3),
                                "sentiment": round(sent_score, 3),
                                "progression": round(prog_score, 3),
                                "raw": round(raw_intensity, 3)
                            }
                        }
                    })

            details.sort(key=lambda x: x["confidence"], reverse=True)
            summary: Dict[str, Any] = {}
            if details:
                top2 = details[:2]
                summary["dominant_emotions"] = [
                    {"category": d["emotion_category"], "sub_emotion": d["sub_emotion"], "confidence": d["confidence"]}
                    for d in top2
                ]
                avg_int = round(sum(d["calculated_intensity"] for d in details) / len(details), 3)
                summary["average_intensity"] = avg_int
                pos_set = {"희", "락"}
                neg_set = {"노", "애"}
                pos_seen = any(d["emotion_category"] in pos_set and d["confidence"] >= 0.5 for d in details[:4])
                neg_seen = any(d["emotion_category"] in neg_set and d["confidence"] >= 0.5 for d in details[:4])
                summary["mixed_polarity"] = bool(pos_seen and neg_seen)
                coverage: Dict[str, int] = {}
                for d in details:
                    coverage[d["emotion_category"]] = coverage.get(d["emotion_category"], 0) + 1
                summary["coverage_counts"] = coverage
            else:
                summary["dominant_emotions"] = []
                summary["average_intensity"] = 0.0
                summary["mixed_polarity"] = False
                summary["coverage_counts"] = {}

            return {"detailed_emotions": details, "overall_summary": summary}
        except Exception:
            logger.exception("정교한 감정 분석 중 오류 발생")
            return {"detailed_emotions": [], "overall_summary": {}}

    # ----------------------------
    # Normalization / Interactions / Utilities
    # ----------------------------
    def normalize_scores(self, scores: Dict[str, float]) -> Dict[str, float]:
        try:
            MINIMUM_THRESHOLD = 0.05
            for k in list(scores.keys()):
                scores[k] = min(max(scores[k], 0.0), 1.0)
            total = sum(scores.values())
            if total > 0:
                norm = {k: v / total for k, v in scores.items()}
                norm = {k: v for k, v in norm.items() if v > MINIMUM_THRESHOLD}
                if norm:
                    t2 = sum(norm.values())
                    norm = {k: v / t2 for k, v in norm.items()}
                return norm
            return {k: 0.0 for k in scores}
        except Exception:
            logger.exception("점수 정규화 중 오류 발생")
            return scores

    def log_score_changes(self, category: str, score_changes: List[float], threshold: float = 0.1):
        try:
            total = sum(score_changes)
            if total > threshold:
                logger.info(f"카테고리 '{category}' 총 점수 변화: +{total:.2f}")
            else:
                logger.debug(f"카테고리 '{category}' 미미한 점수 변화: +{total:.2f}")
        except Exception:
            logger.exception("점수 변화 로깅 중 오류 발생")

    def apply_emotion_interactions(self, context_scores: Dict[str, float]) -> Dict[str, float]:
        try:
            pos = ["희", "락"]
            neg = ["노", "애"]
            pos_score = sum(context_scores.get(c, 0.0) for c in pos) / len(pos)
            neg_score = sum(context_scores.get(c, 0.0) for c in neg) / len(neg)
            RATE = 0.15
            diff = abs(pos_score - neg_score)
            if diff > 0.3:
                if pos_score > neg_score:
                    for c in pos:
                        context_scores[c] *= (1 + RATE)
                    for c in neg:
                        context_scores[c] *= (1 - RATE)
                else:
                    for c in neg:
                        context_scores[c] *= (1 + RATE)
                    for c in pos:
                        context_scores[c] *= (1 - RATE)
            return context_scores
        except Exception:
            logger.exception("감정 상호작용 적용 중 오류 발생")
            return context_scores

    def get_emotion_keywords(self, emotion_data: Dict[str, Any]) -> List[str]:
        try:
            kws = set(self._get_core_keywords(emotion_data))
            sits = (emotion_data.get("context_patterns", {}) or {}).get("situations", {}) or {}
            for st in sits.values():
                kws.update(st.get("keywords", []) or [])
            return list(kws)
        except Exception:
            logger.exception("감정 관련 키워드 추출 중 오류 발생")
            return []

    def calculate_context_confidence(self, text: str, situation_data: Dict[str, Any]) -> float:
        try:
            t = self._norm(text)
            c = 0.5
            desc = self._norm(situation_data.get("description", ""))
            if desc and desc in t:
                c += 0.3
            kws = [self._norm(k) for k in (situation_data.get("keywords", []) or [])]
            c += min(sum(1 for k in kws if k in t) * 0.1, 0.3)
            exs = [self._norm(x) for x in (situation_data.get("examples", []) or [])]
            for e in exs:
                if e in t:
                    c += 0.2
                    break
            return min(c, 1.0)
        except Exception:
            logger.exception("문맥 신뢰도 계산 중 오류 발생")
            return 0.5

    def find_intensity_modifiers(self, text: str, emotions_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        mods = []
        try:
            t = self._norm(text)
            for ed in emotions_data.values():
                sm = self._get_ling_patterns(ed).get("sentiment_modifiers", {}) or {}
                for mtype in ["amplifiers", "diminishers"]:
                    for w in sm.get(mtype, []) or []:
                        if self._norm(w) in t:
                            mods.append({"modifier": w, "position": t.find(self._norm(w)), "type": mtype, "weight": 1.0 if mtype == "amplifiers" else 0.5})
        except Exception:
            logger.exception("강도 수정자 찾기 중 오류 발생")
        return mods

    def calculate_stage_confidence(self, stage: str, text: str, emotions_data: Dict[str, Any]) -> float:
        try:
            STAGE_WEIGHTS_DEFAULT = {"trigger": 0.8, "development": 0.7, "peak": 1.0, "aftermath": 0.6}
            w = self._get_dynamic_stage_weight(stage, emotions_data)
            if w is None:
                w = STAGE_WEIGHTS_DEFAULT.get(stage, 0.7)
            base = 0.5 * float(w)
            norm_text = self._norm(text)
            _, _, mods_cfg = self._load_globals(emotions_data) if isinstance(emotions_data, dict) else (set(), set(), {})
            amp_words = [self._norm(w) for w in (mods_cfg.get("amplifiers") or [])]
            if not amp_words and hasattr(self, "_epm"):
                scfg_obj = getattr(self._epm, "scfg", {}) if hasattr(self._epm, "scfg") else {}
                scfg_mods = scfg_obj.get("intensity_modifiers", {}) if isinstance(scfg_obj, dict) else {}
                fallback = scfg_mods.get("amplifiers") or scfg_mods.get("all") or []
                amp_words = [self._norm(w) for w in fallback]
            if any(a and a in norm_text for a in amp_words):
                base *= 1.2
            return min(base, 1.0)
        except Exception:
            logger.exception("단계 신뢰도 계산 중 오류 발생")
            return 0.5

    def _get_dynamic_stage_weight(self, stage: str, emotions_data: Dict[str, Any]) -> Optional[float]:
        try:
            found = []
            for _, cat_data in emotions_data.items():
                sub = cat_data.get("sub_emotion_profile", {})
                if not sub:
                    sub = (cat_data.get("emotion_profile", {}) or {}).get("sub_emotions", {})
                for _, sd in (sub or {}).items():
                    sits = (sd.get("context_patterns", {}) or {}).get("situations", {}) or {}
                    for _, st in sits.items():
                        ep = st.get("emotion_progression", {}) or {}
                        key = f"{stage}_weight"
                        if key in ep:
                            try:
                                found.append(float(ep[key]))
                            except Exception:
                                pass
            return max(found) if found else None
        except Exception:
            logger.exception("_get_dynamic_stage_weight 오류 발생")
            return None

    def adjust_intensity_level(self, base_level: str, text: str, emotions_data: Dict[str, Any]) -> str:
        try:
            mapv = {"low": 0, "medium": 1, "high": 2}
            val = mapv.get(base_level, 0)
            mods = self.find_intensity_modifiers(text, emotions_data)
            if mods:
                val = min(val + int(max(m["weight"] for m in mods) * 2), 2)
            inv = {0: "low", 1: "medium", 2: "high"}
            return inv[val]
        except Exception:
            logger.exception("강도 레벨 조정 중 오류 발생")
            return base_level

    def calculate_relation_confidence(self, text: str, emotion: str) -> float:
        try:
            t = self._norm(text)
            e = self._norm(emotion)
            c = 0.5
            if e in t:
                c += 0.3
            mods_cfg = {}
            if isinstance(getattr(self, "emotions_data", {}), dict):
                _, _, mods_cfg = self._load_globals(self.emotions_data)
            amp_words = [self._norm(w) for w in (mods_cfg.get("amplifiers") or [])]
            if not amp_words and hasattr(self, "_epm"):
                scfg_obj = getattr(self._epm, "scfg", {}) if hasattr(self._epm, "scfg") else {}
                scfg_mods = scfg_obj.get("intensity_modifiers", {}) if isinstance(scfg_obj, dict) else {}
                fallback = scfg_mods.get("amplifiers") or scfg_mods.get("all") or []
                amp_words = [self._norm(w) for w in fallback]
            if any(a and a in t for a in amp_words):
                c += 0.2
            words = t.split()
            pos = -1
            for i, w in enumerate(words):
                if e in w:
                    pos = i
                    break
            if pos >= 0:
                ctx = " ".join(words[max(0, pos - 2): min(len(words), pos + 3)])
                pset = {"좋은", "행복한", "즐거운", "만족스러운"}
                nset = {"나쁜", "불행한", "슬픈", "불만족스러운"}
                if any(self._norm(w) in ctx for w in pset):
                    c += 0.1
                elif any(self._norm(w) in ctx for w in nset):
                    c -= 0.1
            return min(max(c, 0.0), 1.0)
        except Exception:
            logger.exception("관련 감정 신뢰도 계산 중 오류 발생")
            return 0.5

    # ----------------------------
    # Central Orchestration (optional)
    # ----------------------------
    def analyze_emotions_centrally(self, text: str, emotions_data: Dict[str, Any]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        try:
            logger.info("중앙집중형 감정 분석 시작")
            self.set_emotions_data(emotions_data)
            mltd = emotions_data.get("ml_training_metadata", {}) or {}
            modules = mltd.get("analysis_modules", {}) or {}
            cat_scores = {cat: 0.3 for cat in emotions_data.keys()}

            if modules.get("pattern_extractor", {}).get("enabled", False):
                thr = modules["pattern_extractor"].get("threshold", 0.75)
                rules = modules["pattern_extractor"].get("additional_rules", []) or []
                pats = self._extract_patterns(text, emotions_data, thr, rules)
                result["extracted_patterns"] = pats
                for cat in cat_scores:
                    cat_scores[cat] += sum(1 for p in pats if p["emotion_category"] == cat) * 0.05

            if modules.get("context_extractor", {}).get("enabled", False):
                cache_size = modules["context_extractor"].get("cache_size", 1000)
                mem_lim = modules["context_extractor"].get("memory_limit", 512)
                cinfo = self._extract_context_info(text, emotions_data, cache_size, mem_lim)
                result["context_info"] = cinfo
                for cat in cat_scores:
                    if cat in cinfo.get("dominant_context_emotions", []):
                        cat_scores[cat] += 0.1

            if modules.get("relationship_analyzer", {}).get("enabled", False):
                rs = modules["relationship_analyzer"].get("relationship_strength_threshold", 0.65)
                ct = modules["relationship_analyzer"].get("compatibility_threshold", 0.7)
                rres = self._analyze_relationships(text, emotions_data, rs, ct)
                result["relationship_analysis"] = rres
                for cat, sc in rres.get("relation_scores", {}).items():
                    if len(cat) == 1:
                        cat_scores[cat] += sc * 0.05

            if modules.get("intensity_analyzer", {}).get("enabled", False):
                minc = modules["intensity_analyzer"].get("minimum_confidence", 0.6)
                pm = modules["intensity_analyzer"].get("pattern_match_threshold", 0.7)
                ires = self._analyze_intensity(text, emotions_data, minc, pm)
                result["intensity_analysis"] = ires
                for cat, val in ires.items():
                    cat_scores[cat] += val * 0.1

            cat_scores = self.apply_emotion_interactions(cat_scores)
            cat_scores = self.normalize_scores(cat_scores)
            result["final_scores"] = cat_scores
            return result
        except Exception:
            logger.exception("중앙집중형 감정 분석 중 오류 발생")
            return result

    # ----------------------------
    # Example module bridges
    # ----------------------------
    def _extract_patterns(self, text: str, emotions_data: Dict[str, Any], threshold: float, additional_rules: List[str]) -> List[Dict[str, Any]]:
        out = []
        try:
            sents = self._split_sentences_ko(text)
            for idx, s in enumerate(sents):
                for cat, cdata in emotions_data.items():
                    subs = self._get_sub_emotions_dict(cdata)
                    for sub, sd in subs.items():
                        for kw in self.get_emotion_keywords(sd):
                            if self._norm(kw) in self._norm(s):
                                score = 0.8
                                if score >= threshold:
                                    out.append({"emotion_category": cat, "sub_emotion": sub, "pattern": kw, "sentence_index": idx, "score": score})
        except Exception:
            logger.exception("pattern_extractor 수행 중 오류")
        return out

    def _extract_context_info(self, text: str, emotions_data: Dict[str, Any], cache_size: int, memory_limit: int) -> Dict[str, Any]:
        info = {"dominant_context_emotions": [], "cache_size_used": cache_size, "memory_limit_used": memory_limit}
        try:
            cmap = {}
            for cat, cdata in emotions_data.items():
                subs = self._get_sub_emotions_dict(cdata)
                score = 0.0
                for sd in subs.values():
                    sits = (sd.get("context_patterns", {}) or {}).get("situations", {}) or {}
                    for st in sits.values():
                        desc = st.get("description", "")
                        if desc and self._norm(desc) in self._norm(text):
                            score += 0.2
                cmap[cat] = round(score, 2)
            ranked = sorted(cmap.items(), key=lambda x: x[1], reverse=True)
            if ranked and ranked[0][1] > 0.0:
                info["dominant_context_emotions"].append(ranked[0][0])
            return info
        except Exception:
            logger.exception("context_extractor 수행 중 오류")
            return info

    def _analyze_relationships(self, text: str, emotions_data: Dict[str, Any], rel_strength: float, compat_threshold: float) -> Dict[str, Any]:
        res = {"relation_scores": {}, "relationship_strength_threshold": rel_strength, "compatibility_threshold": compat_threshold}
        try:
            pairs = [("희", "락"), ("노", "애"), ("희", "노"), ("락", "애")]
            for (c1, c2) in pairs:
                n1 = self._count_category_keywords(text, c1)
                n2 = self._count_category_keywords(text, c2)
                res["relation_scores"][f"{c1}-{c2}"] = round(min(n1, n2) * 0.1, 2)
            agg = {}
            for k, v in res["relation_scores"].items():
                for c in k.split("-"):
                    agg[c] = agg.get(c, 0.0) + v
            for c, v in agg.items():
                res["relation_scores"][c] = round(v, 2)
            return res
        except Exception:
            logger.exception("relationship_analyzer 수행 중 오류")
            return res

    def _analyze_intensity(self, text: str, emotions_data: Dict[str, Any], min_conf: float, pattern_thresh: float) -> Dict[str, float]:
        out = {}
        try:
            for cat, cdata in emotions_data.items():
                score = 0.0
                subs = self._get_sub_emotions_dict(cdata)
                for sd in subs.values():
                    if self._check_strong_emotion_presence(text, sd):
                        score += 0.2
                if score >= min_conf:
                    out[cat] = round(score, 2)
            return out
        except Exception:
            logger.exception("intensity_analyzer 수행 중 오류")
            return out

    def _count_category_keywords(self, text: str, category: str) -> int:
        if not self.emotions_data or category not in self.emotions_data:
            return 0
        cdata = self.emotions_data[category]
        subs = self._get_sub_emotions_dict(cdata)
        total = 0
        for sd in subs.values():
            for kw in self.get_emotion_keywords(sd):
                if kw and self._norm(kw) in self._norm(text):
                    total += text.count(kw)
        return total

    def _check_strong_emotion_presence(self, text: str, sub_emotion_data: Dict[str, Any]) -> bool:
        score = 0.0
        try:
            for kw in self._get_core_keywords(sub_emotion_data):
                if self._norm(kw) in self._norm(text):
                    score += 0.1
            return score > 0.3
        except Exception:
            logger.exception("강한 감정 존재 여부 확인 중 오류 발생")
            return False


# =============================================================================
# Independent Functions (with lightweight caching & robust schema)
# =============================================================================
_LM_CACHE_LOCK = RLock()
_LM_CACHE: Dict[str, Dict[str, Any]] = {
    "matcher": {},
    "analyzer": {},
    "progression": {},
}

def _lm_cache_enabled() -> bool:
    return os.environ.get("LM_USE_CACHE", "1") in ("1", "true", "True")

def _make_emodata_key(emotions_data: Dict[str, Any]) -> str:
    try:
        # 가장 빠른 키: 객체 id (프로세스 내 재사용 시 안정적)
        base = f"id:{id(emotions_data)}"
        if os.environ.get("LM_STRICT_CACHE_KEY", "0") in ("1", "true", "True"):
            # 더 엄격한 키(옵션): 키/값 일부 해시(라벨링뼈대가 자주 바뀌면 켜기)
            snap = json.dumps({"k": sorted(list(emotions_data.keys()))[:32]}, ensure_ascii=False)
            base = "sha1:" + hashlib.sha1(snap.encode("utf-8"), usedforsecurity=False).hexdigest()
        return base
    except Exception:
        return f"id:{id(emotions_data)}"

def _get_matcher(emotions_data: Dict[str, Any]) -> LinguisticMatcher:
    if not _lm_cache_enabled():
        return LinguisticMatcher(emotions_data=emotions_data)
    key = _make_emodata_key(emotions_data)
    with _LM_CACHE_LOCK:
        inst = _LM_CACHE["matcher"].get(key)
        if inst is None:
            inst = LinguisticMatcher(emotions_data=emotions_data)
            _LM_CACHE["matcher"][key] = inst
        return inst

def _get_analyzer(emotions_data: Dict[str, Any]) -> EmotionalAnalyzer:
    if not _lm_cache_enabled():
        return EmotionalAnalyzer(emotions_data=emotions_data)
    key = _make_emodata_key(emotions_data)
    with _LM_CACHE_LOCK:
        inst = _LM_CACHE["analyzer"].get(key)
        if inst is None:
            inst = EmotionalAnalyzer(emotions_data=emotions_data)
            _LM_CACHE["analyzer"][key] = inst
        return inst

def _get_progression(emotions_data: Dict[str, Any]) -> EmotionProgressionMatcher:
    if not _lm_cache_enabled():
        return EmotionProgressionMatcher()
    key = _make_emodata_key(emotions_data)
    with _LM_CACHE_LOCK:
        inst = _LM_CACHE["progression"].get(key)
        if inst is None:
            inst = EmotionProgressionMatcher()
            _LM_CACHE["progression"][key] = inst
        return inst

def _ok(payload: Dict[str, Any]) -> Dict[str, Any]:
    payload.setdefault("ok", True)
    return payload

def _err(e: Exception, extra: Dict[str, Any] = None) -> Dict[str, Any]:
    out = {"ok": False, "error": str(e), "error_type": type(e).__name__}
    if extra:
        out.update(extra)
    return out

def match_language_patterns(text: str, emotions_data: Dict[str, Any]) -> Dict[str, Any]:
    """텍스트 내 감정 관련 언어 패턴(키워드, 상황, 강도수식 등) 매칭 결과"""
    emotions_data = _ensure_emotions_data(emotions_data)
    try:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("입력 텍스트가 비어있습니다.")
        if not isinstance(emotions_data, dict) or not emotions_data:
            raise ValueError("emotions_data가 비어있습니다.")
        matcher = _get_matcher(emotions_data)
        res = matcher.match_linguistic_patterns(text, emotions_data)
        return _inject_diagnostics(_ok(res if isinstance(res, dict) else {"result": res}))
    except Exception as e:
        logger.error(f"[match_language_patterns] 오류: {e}", exc_info=True)
        return _inject_diagnostics(_err(e))

def analyze_emotion_progression(
    text: str,
    emotions_data: Dict[str, Any],
    sentence_emotion_scores: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """텍스트 내 감정 전이/발전(Progression) 감지"""
    emotions_data = _ensure_emotions_data(emotions_data)
    try:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("입력 텍스트가 비어있습니다.")
        if not isinstance(emotions_data, dict) or not emotions_data:
            raise ValueError("emotions_data가 비어있습니다.")
        pm = _get_progression(emotions_data)
        res = pm.match_emotion_progression(text, emotions_data, sentence_emotion_scores=sentence_emotion_scores)
        return _inject_diagnostics(_ok(res if isinstance(res, dict) else {"result": res}))
    except Exception as e:
        logger.error(f"[analyze_emotion_progression] 오류: {e}", exc_info=True)
        return _inject_diagnostics(_err(e, {"emotion_progression_matches": [], "emotion_transition_matches": []}))

def analyze_enhanced_emotions(text: str, emotions_data: Dict[str, Any]) -> Dict[str, Any]:
    """정교 분석(강도·문맥·확장성 포함)"""
    emotions_data = _ensure_emotions_data(emotions_data)
    try:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("입력 텍스트가 비어있습니다.")
        if not isinstance(emotions_data, dict) or not emotions_data:
            raise ValueError("emotions_data가 비어있습니다.")
        analyzer = _get_analyzer(emotions_data)
        res = analyzer.analyze_enhanced_emotions(text, emotions_data)
        # 일관 스키마 보장
        if "detailed_emotions" not in res or "overall_summary" not in res:
            res = {"detailed_emotions": res.get("detailed_emotions", []),
                   "overall_summary": res.get("overall_summary", {})}
        return _inject_diagnostics(_ok(res))
    except Exception as e:
        logger.error(f"[analyze_enhanced_emotions] 오류: {e}", exc_info=True)
        return _inject_diagnostics(_err(e, {"detailed_emotions": [], "overall_summary": {}}))

def analyze_emotions_centrally(text: str, emotions_data: Dict[str, Any]) -> Dict[str, Any]:
    """중앙집중형 감정 분석(모듈 통합 실행)"""
    emotions_data = _ensure_emotions_data(emotions_data)
    try:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("입력 텍스트가 비어있습니다.")
        if not isinstance(emotions_data, dict) or not emotions_data:
            raise ValueError("emotions_data가 비어있습니다.")
        analyzer = _get_analyzer(emotions_data)
        res = analyzer.analyze_emotions_centrally(text, emotions_data)
        return _inject_diagnostics(_ok(res if isinstance(res, dict) else {"result": res}))
    except Exception as e:
        logger.error(f"[analyze_emotions_centrally] 오류: {e}", exc_info=True)
        return _inject_diagnostics(_err(e))

def run_linguistic_analysis(
    text: str,
    emotions_data: Dict[str, Any],
    use_progression: bool = False
) -> Dict[str, Any]:
    """통합 언어 분석: 패턴 매칭 + (옵션) 전이/발전"""
    emotions_data = _ensure_emotions_data(emotions_data)
    try:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("입력 텍스트가 비어있습니다.")
        if not isinstance(emotions_data, dict) or not emotions_data:
            raise ValueError("emotions_data가 비어있습니다.")
        matcher = _get_matcher(emotions_data)
        result = matcher.match_linguistic_patterns(text, emotions_data)
        if use_progression:
            pm = _get_progression(emotions_data)
            result["emotion_progression"] = pm.match_emotion_progression(text, emotions_data)
        return _inject_diagnostics(_ok(result if isinstance(result, dict) else {"result": result}))
    except Exception as e:
        logger.error(f"[run_linguistic_analysis] 오류: {e}", exc_info=True)
        out = {"matched_phrases": []}
        if use_progression:
            out["emotion_progression"] = {"emotion_progression_matches": [], "emotion_transition_matches": []}
        return _inject_diagnostics(_err(e, out))

# =============================================================================
# Cache Control Public APIs (optional, for __init__ export)
# =============================================================================
def lm_set_cache_enabled(enabled: bool) -> None:
    """독립함수 캐시 on/off (프로세스 환경변수 기반)."""
    os.environ["LM_USE_CACHE"] = "1" if enabled else "0"

def lm_clear_cache(scope: str = "all") -> int:
    """캐시 비우기: scope in {'matcher','analyzer','progression','all'}.
    반환: 정리된 엔트리 수(int)"""
    cleared = 0
    try:
        with _LM_CACHE_LOCK:  # noqa: F821
            if scope == "all":
                for k in list(_LM_CACHE.keys()):  # noqa: F821
                    cleared += len(_LM_CACHE[k])  # noqa: F821
                    _LM_CACHE[k].clear()
            elif scope in _LM_CACHE:
                cleared = len(_LM_CACHE[scope])
                _LM_CACHE[scope].clear()
            else:
                logger.warning(f"[lm_clear_cache] 알 수 없는 scope='{scope}', 아무 것도 수행하지 않음.")
                cleared = 0
    except Exception as e:
        logger.warning(f"[lm_clear_cache] 실패: {e}")
        cleared = 0
    return int(cleared)

def lm_cache_stats() -> Dict[str, int]:
    """캐시에 잡혀있는 인스턴스 수를 반환."""
    try:
        with _LM_CACHE_LOCK:  # noqa: F821
            return {k: len(v) for k, v in _LM_CACHE.items()}  # noqa: F821
    except Exception:
        return {"matcher": 0, "analyzer": 0, "progression": 0}



# =============================================================================
# Main Function (compact outputs + robust path/logs)
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

# 통합 로그 관리자 사용 (날짜별 폴더)
try:
    from log_manager import get_log_manager
    log_manager = get_log_manager()
    LOGS_DIR = str(log_manager.get_log_dir("emotion_analysis", use_date_folder=True))
except ImportError:
    # 폴백: 기존 방식 (날짜별 폴더 추가)
    base_logs_dir = os.environ.get("LM_LOG_DIR", os.path.join(SCRIPT_DIR, "logs"))
    today = datetime.now().strftime("%Y%m%d")
    LOGS_DIR = os.path.join(base_logs_dir, today)
    os.makedirs(LOGS_DIR, exist_ok=True)

def _setup_logger() -> logging.Logger:
    lg = logging.getLogger("linguistic")
    fast_mode = os.environ.get("LM_FAST_MODE", "").strip().lower() in ("1", "true", "yes")
    target_level = logging.WARNING if fast_mode else logging.INFO
    if not lg.handlers:
        # 기본 비간섭: NullHandler만 부착
        lg.addHandler(logging.NullHandler())

        # Opt-in: 환경변수로 파일/콘솔 핸들러 활성화
        fmt = logging.Formatter("[%(asctime)s|%(levelname)s|%(name)s] %(message)s")
        if os.environ.get("LM_FILE_LOG", "0") == "1":
            try:
                fh = RotatingFileHandler(
                    filename=os.path.join(LOGS_DIR, "linguistic_matcher.log"),
                    maxBytes=5 * 1024 * 1024,
                    backupCount=3,
                    encoding="utf-8",
                )
                fh.setFormatter(fmt)
                lg.addHandler(fh)
            except Exception:
                pass
        if os.environ.get("LM_CONSOLE_LOG", "0") == "1":
            try:
                sh = logging.StreamHandler()
                sh.setFormatter(fmt)
                lg.addHandler(sh)
            except Exception:
                pass
        lg.propagate = False
    lg.setLevel(target_level)
    for handler in lg.handlers:
        handler.setLevel(target_level)
    return lg

logger = _setup_logger()

def _legacy_resolve_emotions_json_path(preferred_path: str = None) -> str:
    tried = []
    def _is_valid(p):
        try:
            return os.path.isfile(p) and os.path.getsize(p) > 0
        except Exception:
            return False

    env_path = os.environ.get("EMOTIONS_JSON_PATH")
    if env_path:
        p = os.path.normpath(os.path.abspath(env_path))
        if _is_valid(p):
            logger.info(f"[config] EMOTIONS.JSON 확정: {p}")
            return p
        tried.append(p)

    if preferred_path:
        p = os.path.normpath(os.path.abspath(preferred_path))
        if _is_valid(p):
            logger.info(f"[config] EMOTIONS.JSON 확정: {p}")
            return p
        tried.append(p)

    candidates = []
    for base in [SCRIPT_DIR, PROJECT_DIR, os.getcwd()]:
        for name in ["EMOTIONS.JSON", "EMOTIONS.json", "Emotions.json"]:
            candidates.append(os.path.join(base, name))
    for base in [SCRIPT_DIR, PROJECT_DIR]:
        for pat in ["EMOTIONS*.JSON", "EMOTIONS*.json", "emotions*.json"]:
            candidates.extend(glob.glob(os.path.join(base, pat)))

    seen = set()
    for c in candidates:
        c = os.path.normpath(c)
        if c.lower() in seen:
            continue
        seen.add(c.lower())
        if _is_valid(c):
            logger.info(f"[config] EMOTIONS.JSON 확정: {c}")
            return c
        tried.append(c)

    tried_list = "\n - ".join(tried[:12])
    raise FileNotFoundError(
        "EMOTIONS.JSON 파일을 찾을 수 없습니다. 시도 경로:\n"
        f" - {tried_list}\n"
        "환경변수 EMOTIONS_JSON_PATH를 파일 경로로 설정하거나 파일을 스크립트/상위 폴더에 배치하세요."
    )

def resolve_emotions_json_path(preferred_path: Optional[str] = None) -> str:
    """공개 유틸: 외부에서 파일 경로 탐색 로직 재사용을 위한 별칭."""
    p = _resolve_emotions_json_path()
    if p:
        return str(p)
    return _legacy_resolve_emotions_json_path(preferred_path)

def load_emotions_data(file_path: str) -> Dict[str, Any]:
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def _round_floats(obj: Any, nd: int = 3) -> Any:
    if isinstance(obj, float):
        return round(obj, nd)
    if isinstance(obj, list):
        return [_round_floats(x, nd) for x in obj]
    if isinstance(obj, dict):
        return {k: _round_floats(v, nd) for k, v in obj.items()}
    return obj

def _topk(lst: List[Dict[str, Any]], k: int, key: str) -> List[Dict[str, Any]]:
    try:
        return sorted(lst, key=lambda x: x.get(key, 0), reverse=True)[:max(0, k)]
    except Exception:
        return lst[:max(0, k)]

def _choose_dominant_context(
    m_weight: Dict[str, float],
    ctx_mean: Optional[str],
    *,
    analyzer_top: Optional[str] = None,
    tie_delta_env: str = "LM_DOMINANT_TIEDELTA",
    analyzer_bias_env: str = "LM_DOMINANT_ANALYZER_DELTA",
) -> Tuple[Optional[str], str]:
    def _fenv(name: str, default: float) -> float:
        try:
            return float(os.environ.get(name, str(default)))
        except Exception:
            return default

    delta = _fenv(tie_delta_env, 0.08)
    a_delta = _fenv(analyzer_bias_env, 0.04)

    reason = "context" if ctx_mean else "weight"
    dom = ctx_mean
    if not isinstance(m_weight, dict) or not m_weight:
        return dom, reason

    w_sorted = sorted(m_weight.items(), key=lambda kv: kv[1], reverse=True)
    w_top, w_top_val = w_sorted[0]
    if not dom:
        dom, reason = w_top, "weight"
    else:
        dom_score = float(m_weight.get(dom, 0.0))
        if w_top != dom and (w_top_val - dom_score) <= delta:
            dom, reason = w_top, "weight_tiebreak"

    if analyzer_top and analyzer_top in m_weight and analyzer_top != dom:
        dom_adv = float(m_weight.get(dom, 0.0)) - float(m_weight.get(analyzer_top, 0.0))
        if dom_adv <= a_delta:
            dom, reason = analyzer_top, "analyzer_tiebreak"

    return dom, reason


def _compact_item(item: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    top_matched = int(cfg.get("top_matched", 5))
    top_detailed = int(cfg.get("top_detailed", 5))
    include_evi = bool(int(cfg.get("include_evidence", 0)))

    text = item.get("text", "")
    mr = item.get("matcher_result", {}) or {}
    ar = item.get("analyzer_result", {}) or {}

    m_matched = mr.get("matched_phrases", []) or []
    m_compact = []
    for mp in _topk(m_matched, top_matched, "confidence"):
        m_compact.append({
            "category": mp.get("emotion_category"),
            "sub": mp.get("sub_emotion"),
            "pattern": mp.get("pattern"),
            "type": mp.get("type"),
            "conf": round(float(mp.get("confidence", 0.0)), 3),
        })

    m_weight = mr.get("weighted_scores", {}) or {}
    ctx_mean = (mr.get("contextual_meaning", {}) or {}).get("dominant_emotion")

    summary = ar.get("overall_summary", {}) or {}
    analyzer_top_cat = None
    try:
        de = summary.get("dominant_emotions", [])
        if isinstance(de, list) and de:
            analyzer_top_cat = de[0].get("category")
    except Exception:
        analyzer_top_cat = None

    dom, dom_reason = _choose_dominant_context(
        m_weight, ctx_mean,
        analyzer_top=analyzer_top_cat,
        tie_delta_env="LM_DOMINANT_TIEDELTA",
        analyzer_bias_env="LM_DOMINANT_ANALYZER_DELTA",
    )

    dyn = mr.get("dynamic_expressions", {}) or {}
    dyn_trans_cnt = len(dyn.get("emotion_transitions", []) or [])
    expl_cnt = len(mr.get("explicit_transitions", []) or [])
    transition_total = max(expl_cnt, dyn_trans_cnt)

    ts = mr.get("time_series", {}) or {}
    ts_len = len(ts.get("time_series", []) or [])

    a_details = ar.get("detailed_emotions", []) or []
    a_top = _topk(a_details, top_detailed, "confidence")
    a_compact = []
    for d in a_top:
        entry = {
            "category": d.get("emotion_category"),
            "sub": d.get("sub_emotion"),
            "intensity": round(float(d.get("calculated_intensity", 0.0)), 3),
            "conf": round(float(d.get("confidence", 0.0)), 3),
            "level": d.get("matched_intensity_level"),
        }
        if include_evi:
            ev = d.get("evidence", {}) or {}
            entry["evidence"] = {
                "core_hits": ev.get("core_hits", [])[:5],
                "phrase_hits": [{"pattern": p.get("pattern"), "w": p.get("weight")} for p in (ev.get("phrase_hits", []) or [])[:5]],
                "situation_hits": (ev.get("situation_hits", []) or [])[:5],
                "progression_triggers": ev.get("progression_triggers", [])[:2],
            }
        a_compact.append(entry)

    compact = {
        "text": text,
        "matcher": {
            "dominant_context": dom,
            "dominant_context_reason": dom_reason,   # ← 추가
            "dominant_context_scores": {             # ← 참고 점수 스냅샷
                "context": ctx_mean,
                "weighted_top": max(m_weight.items(), key=lambda kv: kv[1])[0] if m_weight else None,
                "weighted_scores": _round_floats(m_weight, 3)
            },
            "matched_phrases_top": m_compact,
            "dynamic_counts": {
                "transitions": dyn_trans_cnt,
                "progressions": len(dyn.get("progression_patterns", []) or []),
            },
            "explicit_transition_count": expl_cnt,
            "transition_total": transition_total,
            "time_series_len": ts_len,
        },
        "analyzer": {
            "detailed_top": a_compact,
            "summary": _round_floats({
                "dominant_emotions": summary.get("dominant_emotions", [])[:3],
                "average_intensity": summary.get("average_intensity", 0.0),
                "mixed_polarity": summary.get("mixed_polarity", False),
                "coverage_counts": summary.get("coverage_counts", {}),
            }, 3),
        },
    }
    return compact

def _make_preview(all_results: List[Dict[str, Any]], limit: int = 30) -> Dict[str, Any]:
    preview = []
    for i, item in enumerate(all_results[:max(0, int(limit))]):
        mr = item.get("matcher_result", {}) or {}
        ar = item.get("analyzer_result", {}) or {}

        m_weight = mr.get("weighted_scores", {}) or {}
        ctx_mean = (mr.get("contextual_meaning", {}) or {}).get("dominant_emotion")

        analyzer_top_cat = None
        try:
            de = (ar.get("overall_summary", {}) or {}).get("dominant_emotions", [])
            if isinstance(de, list) and de:
                analyzer_top_cat = de[0].get("category")
        except Exception:
            analyzer_top_cat = None

        dom, dom_reason = _choose_dominant_context(
            m_weight, ctx_mean,
            analyzer_top=analyzer_top_cat,
            tie_delta_env="LM_DOMINANT_TIEDELTA",
            analyzer_bias_env="LM_DOMINANT_ANALYZER_DELTA",
        )

        dyn = mr.get("dynamic_expressions", {}) or {}
        dyn_trans_cnt = len(dyn.get("emotion_transitions", []) or [])
        expl_cnt = len(mr.get("explicit_transitions", []) or [])
        transition_total = max(expl_cnt, dyn_trans_cnt)

        preview.append({
            "text": item.get("text", "")[:120],
            "matched_phrases": len(mr.get("matched_phrases", []) or []),
            "dominant_context": dom,
            "dominant_context_reason": dom_reason,  # ← 추가
            "explicit_transitions": expl_cnt,
            "transition_total": transition_total,
            "detailed_emotions": len(ar.get("detailed_emotions", []) or []),
            "summary": ar.get("overall_summary", {}).get("dominant_emotions", [])[:2],
        })
    return {"preview_count": len(preview), "items": preview}



def _save_json(path: str, obj: Any):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def _save_json_gz(path: str, obj: Any):
    import gzip
    tmp = path + ".tmp"
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

if __name__ == "__main__":
    try:
        emotions_data_path = resolve_emotions_json_path(os.path.join(PROJECT_DIR, "EMOTIONS.JSON"))
        emotions_data = load_emotions_data(emotions_data_path)
        if not emotions_data:
            logger.error("감정 데이터 로드 실패 (emotions_data가 비어있음)")
            sys.exit(1)
        logger.info(f"[라벨링뼈대] 데이터 로드 완료: {len(emotions_data)}개의 대표감정 카테고리")

        logger.info("LinguisticMatcher, EmotionalAnalyzer 초기화...")
        matcher = LinguisticMatcher(emotions_data=emotions_data)
        analyzer = EmotionalAnalyzer(emotions_data=emotions_data)

        test_texts = [
            "바다처럼 끝없이 넘실대는 기쁨, 깊은 숲속 어둠처럼 고요히 내려앉는 슬픔.",
            "이 프로젝트는 오랜 시간 노력을 기울인 끝에 성공하여 환희를 느꼈지만, 동시에 불안감도 조금 있었습니다.",
            "며칠 후가 지나고 나서야 화가 풀렸어요. 결국에는 사랑으로 바뀌었죠.",
            "장례식장에서 만난 그는 분노와 애통함이 교차했지만, 친구들의 위로에 조금씩 마음이 풀렸습니다.",
            "정말 너무 만족스러워요! 가족들과 함께해서 행복하고, 회사에서 인정도 받아서 기분이 좋습니다.",
            "이건 정말 별로예요. 짜증도 나고, 실망도 컸죠. 아무리 생각해도 제 기대에 못 미쳤어요."
        ]
        logger.info(f"총 {len(test_texts)}개의 샘플 텍스트를 이용하여 테스트를 진행합니다.")

        all_results: List[Dict[str, Any]] = []
        for idx, text in enumerate(test_texts, start=1):
            logger.info(f"[테스트#{idx}] 텍스트 분석 시작: {text}")
            lm_result = matcher.match_linguistic_patterns(text, emotions_data)
            logger.info(f"[테스트#{idx}] match_linguistic_patterns: matched_phrases={len(lm_result.get('matched_phrases', []))}, "
                        f"weighted_scores={lm_result.get('weighted_scores', {})}")
            ea_result = analyzer.analyze_enhanced_emotions(text, emotions_data)
            logger.info(f"[테스트#{idx}] analyze_enhanced_emotions: detailed_emotions={len(ea_result.get('detailed_emotions', []))}")
            all_results.append({"text": text, "matcher_result": lm_result, "analyzer_result": ea_result})

        print("\n=== 종합 테스트 결과 ===")
        for i, item in enumerate(all_results, start=1):
            print(f"\n[테스트#{i}] 텍스트: {item['text']}")
            mps = item["matcher_result"].get("matched_phrases", [])
            print(f" - LinguisticMatcher matched_phrases({len(mps)}):")
            for mp in mps[:5]:
                print(f"    * {mp['emotion_category']}/{mp.get('sub_emotion','미정')} - '{mp['pattern']}', conf={mp['confidence']:.2f}")
            eds = item["analyzer_result"].get("detailed_emotions", [])
            print(f" - EmotionalAnalyzer detailed_emotions({len(eds)}):")
            for ed in eds[:3]:
                print(f"    > [{ed['emotion_category']}/{ed['sub_emotion']}] intensity={ed['calculated_intensity']}, conf={ed['confidence']}")

        user = getpass.getuser() or "user"
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        ofmt = os.environ.get("LM_OUTPUT_FORMAT", "both").lower()
        do_compact = os.environ.get("LM_COMPACT", "1") in ("1", "true", "True")
        preview_limit = int(os.environ.get("LM_PREVIEW_LIMIT", "30"))
        top_matched = int(os.environ.get("LM_TOP_MATCHED", "5"))
        top_detailed = int(os.environ.get("LM_TOP_DETAILED", "5"))
        include_evidence = os.environ.get("LM_INCLUDE_EVIDENCE", "0")

        base = os.path.join(LOGS_DIR, f"linguistic_matcher_fulltest_{user}_{ts}")

        if do_compact:
            cfg = {
                "top_matched": top_matched,
                "top_detailed": top_detailed,
                "include_evidence": include_evidence,
            }
            compact_results = [_compact_item(item, cfg) for item in all_results]
            compact_results = _round_floats(compact_results, 3)
            _save_json(base + "_compact.json", compact_results)
            logger.info(f"[저장] 컴팩트 JSON → {base}_compact.json")

            preview = _make_preview(all_results, preview_limit)
            _save_json(base + "_preview.json", preview)
            logger.info(f"[저장] 프리뷰 JSON → {base}_preview.json")

        if ofmt in ("json", "both"):
            _save_json(base + "_full.json", all_results)
            logger.info(f"[저장] 전체 JSON → {base}_full.json")
        if ofmt in ("json.gz", "both"):
            _save_json_gz(base + "_full.json.gz", all_results)
            logger.info(f"[저장] 전체 JSON.GZ → {base}_full.json.gz")

        logger.info("=== 모든 테스트 시나리오 완료 ===")

    except Exception as main_err:
        logger.exception("메인 실행 중 오류")
        print(f"오류 발생: {str(main_err)}")
        sys.exit(1)





