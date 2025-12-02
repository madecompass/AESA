# intensity_analyzer.py
# -*- coding: utf-8 -*-

import sys
import os
import json
import gzip
import shutil
import logging
import tempfile
import argparse
import re
import math
import pickle
import hashlib
from collections import defaultdict
from threading import RLock
from functools import lru_cache
from typing import Any, Dict, Optional, Tuple, List, Set, Iterable
from datetime import datetime

import numpy as np
import torch
import unicodedata as _ud
from contextlib import nullcontext

try:
    from transformers import AutoTokenizer, AutoModel
except Exception:
    AutoTokenizer = None
    AutoModel = None

# Aho-Corasick 자동자 지원 (선택적 의존성)
try:
    import ahocorasick
    _AHO_OK = True
except ImportError:
    _AHO_OK = False

try:
    from filelock import FileLock, Timeout
except Exception:
    FileLock = None
    Timeout = None

# ---------------------------------------------------------------------
# 환경 변수 유틸
# ---------------------------------------------------------------------
def _env_bool(name: str, default: bool = False) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return str(val).strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    val = os.environ.get(name)
    if val is None:
        return default
    try:
        return int(str(val).strip())
    except Exception:
        return default


# ---------------------------------------------------------------------
# 로깅 초기화 (회전 + gzip + 중복핸들러 방지 + 서브모듈 소음 억제)
# ---------------------------------------------------------------------
def setup_logger(
    log_dir: str,
    log_filename: str,
    *,
    level: int = logging.INFO,
    rotate_when: str = "midnight",
    rotate_interval: int = 1,
    backup_count: int = 7,
    max_bytes: int = 0,           # 0이면 시간기반 회전, >0이면 용량기반 회전
    gzip_rotate: bool = True,
    logger_name: str = "emotion"
) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_filename)

    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.propagate = False

    # 이미 구성되어 있으면 재사용
    if getattr(logger, "_configured", False):
        return logger

    # 기본 비간섭: 최소 NullHandler만
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())

    fmt = logging.Formatter("[%(asctime)s|%(levelname)s|%(name)s] %(message)s")

    # Opt-in: 환경변수로 파일/콘솔 로깅 허용
    if os.environ.get("INT_FILE_LOG", "0") == "1":
        try:
            if max_bytes > 0:
                from logging.handlers import RotatingFileHandler
                fh = RotatingFileHandler(
                    log_path, mode="a", maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
                )
            else:
                from logging.handlers import TimedRotatingFileHandler
                fh = TimedRotatingFileHandler(
                    log_path, when=rotate_when, interval=rotate_interval, backupCount=backup_count, encoding="utf-8"
                )
                if gzip_rotate:
                    fh.namer = lambda name: f"{name}.gz"
                    def _rotator(source: str, dest: str) -> None:
                        with open(source, "rb") as sf, gzip.open(dest, "wb") as df:
                            shutil.copyfileobj(sf, df)
                        try:
                            os.remove(source)
                        except OSError:
                            pass
                    fh.rotator = _rotator
            fh.setFormatter(fmt)
            logger.addHandler(fh)
        except Exception:
            pass

    if os.environ.get("INT_CONSOLE_LOG", "0") == "1":
        try:
            ch = logging.StreamHandler()
            ch.setLevel(level)
            ch.setFormatter(fmt)
            logger.addHandler(ch)
        except Exception:
            pass

    # 서드파티 로그 소음 억제는 유지
    for noisy in ("transformers", "huggingface_hub", "urllib3", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logger._configured = True  # type: ignore[attr-defined]
    return logger


# ---------------------------------------------------------------------
# 안전한 JSON 기록 (원자적 쓰기 + 파일락 + 선택적 gzip)
# ---------------------------------------------------------------------
def atomic_write_text(
    text: str,
    path: str,
    *,
    use_lock: bool = True,
    timeout: float = 10.0,
    gzip_by_ext: bool = True
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", dir=os.path.dirname(os.path.abspath(path)))
    os.close(tmp_fd)

    def _write(p: str) -> None:
        if gzip_by_ext and p.endswith(".gz"):
            with gzip.open(p, "wt", encoding="utf-8") as f:
                f.write(text)
        else:
            with open(p, "w", encoding="utf-8") as f:
                f.write(text)

    if use_lock and FileLock is not None:
        lock_path = path + ".lock"
        try:
            # ✅ 올바른 컨텍스트 매니저 사용
            with FileLock(lock_path, timeout=timeout):
                _write(tmp_path)
                os.replace(tmp_path, path)  # 원자적 교체
        finally:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except OSError:
                pass
    else:
        _write(tmp_path)
        os.replace(tmp_path, path)


def dump_json_atomic(
    data: Dict[str, Any],
    path: str,
    *,
    indent: int = 2,
    ensure_ascii: bool = False,
    use_lock: bool = True,
    timeout: float = 10.0
) -> None:
    text = json.dumps(data, ensure_ascii=ensure_ascii, indent=indent)
    atomic_write_text(text, path, use_lock=use_lock, timeout=timeout, gzip_by_ext=True)


# ---------------------------------------------------------------------
# 모듈 전역 로거 (필요 시 외부에서 setup_logger 호출 후 재바인딩 권장)
# ---------------------------------------------------------------------
logger = logging.getLogger("emotion")
logger.addHandler(logging.NullHandler())



# =============================================================================
# TransitionWeightCalculator
# =============================================================================
class TransitionWeightCalculator:

    DEFAULT_CFG: Dict[str, Any] = {
        # 대표 감정(라벨 파일 상의 최상위 키들과 일치해야 합니다)
        "primary_emotions": ["희", "노", "애", "락"],

        # 복잡도 점수(없으면 basic으로 처리)
        "complexity_scores": {"basic": 1.0, "subtle": 1.15, "complex": 1.3},

        # 같은 대표 카테고리/다른 대표 카테고리 보정
        "category_multipliers": {"same": 1.1, "diff": 0.95},

        # 강도 예시 갯수 차이에 대한 패널티 (차이 1당 0.05씩)
        "intensity_diff_factor": 0.05,

        # 키워드 유사도(자카드) 보너스 계수: 1 + jaccard * W
        "keyword_sim_weight": 0.25,

        # progression/prior 보너스: 1 + prior * W (prior는 [0,1] 가정, 없으면 0)
        "prior_weight": 0.3,

        # 명시적 상극(incompatibilities 등) 발견 시 곱하는 패널티 (최대 1 - P)
        "conflict_penalty": 0.25,

        # 발렌스(정/부) 일치/불일치 보정
        "valence_same_multiplier": 1.05,
        "valence_diff_multiplier": 0.95,

        # 최종 클램프
        "min_weight": 0.1,
        "max_weight": 2.0,

        # 토큰화/키워드 추출 관련
        "token_min_len": 2,
    }

    def __init__(self, emotions_data: Dict[str, Any], config: Optional[Dict[str, Any]] = None):
        self.emotions_data = emotions_data or {}
        self.config: Dict[str, Any] = {**TransitionWeightCalculator.DEFAULT_CFG, **(config or {})}
        self.primary_emotions: List[str] = list(self.config["primary_emotions"])
        self._cache: Dict[Tuple[str, str], float] = {}

    # --------------------------------------------------------------------- #
    # public entry-point                                                    #
    # --------------------------------------------------------------------- #
    def generate(self) -> Dict[Tuple[str, str], float]:
        """
        모든 전이 (대표→대표, 세부→세부) 가중치를 계산하여 반환합니다.
        반환: { (from_id, to_id): weight, ... }
        """
        if self._cache:
            return self._cache

        # [GENIUS INSIGHT] File-based Caching Strategy
        # Calculating 80x80 matrix with string processing is CPU intensive (O(N^2)).
        # We use a disk cache to persist this across worker processes.
        cache_key = self._compute_cache_key()
        cache_dir = os.path.join(tempfile.gettempdir(), "made_intensity_cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"weights_{cache_key}.pkl")

        # Try loading from disk
        try:
            if os.path.exists(cache_path):
                with open(cache_path, "rb") as f:
                    self._cache = pickle.load(f)
                    # Simple validation
                    if isinstance(self._cache, dict) and len(self._cache) > 0:
                        return self._cache
        except Exception:
            pass  # Fallback to re-calculation

        # Heavy Calculation
        for key, f_id, t_id, from_d, to_d, f_cat, t_cat in self._iter_pairs():
            self._cache[key] = self._pair_weight(
                from_id=f_id, to_id=t_id,
                from_data=from_d, to_data=to_d,
                parent_from_cat=f_cat, parent_to_cat=t_cat,
            )

        # Save to disk
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(self._cache, f)
        except Exception:
            pass

        return self._cache

    def _compute_cache_key(self) -> str:
        """Generate a unique key based on primary emotions and data keys."""
        try:
            # Use keys of emotions data to detect schema changes
            keys = sorted(self.emotions_data.keys())
            content_str = "|".join(keys) + "|" + str(self.config.get("primary_emotions"))
            return hashlib.md5(content_str.encode("utf-8")).hexdigest()
        except Exception:
            return "default"

    def to_json_ready(self, weights: Optional[Dict[Tuple[str, str], float]] = None, sort_desc: bool = True) -> Dict[str, float]:
        """
        JSON 직렬화를 위해 키를 "from->to" 문자열로 변환합니다.
        """
        w = weights or self._cache or self.generate()
        items = [ (f"{a}->{b}", v) for (a,b), v in w.items() ]
        if sort_desc:
            items.sort(key=lambda kv: kv[1], reverse=True)
        return dict(items)

    def generate_topk(self, k: int = 500) -> Dict[Tuple[str, str], float]:
        """
        상위 K개의 전이만 반환(분석/시각화에 유용). 기존 generate()와 독립적.
        """
        all_w = self.generate()
        return dict(sorted(all_w.items(), key=lambda kv: kv[1], reverse=True)[:max(1, k)])

    # --------------------------------------------------------------------- #
    # helpers                                                               #
    # --------------------------------------------------------------------- #
    @staticmethod
    def _safe_get(obj: Any, path: str, default: Any = None):
        cur = obj
        for part in path.split("."):
            if isinstance(cur, dict):
                cur = cur.get(part)
            else:
                return default
        return default if cur is None else cur

    def _sub_emotions(self, cat_id: str) -> Dict[str, Any]:
        # sub_emotions가 없으면 emotion_profile.sub_emotions → 그 다음 top-level sub_emotions 순으로 시도
        cat = self.emotions_data.get(cat_id, {}) if isinstance(self.emotions_data, dict) else {}
        if not isinstance(cat, dict):
            return {}
        sub = self._safe_get(cat, "emotion_profile.sub_emotions", None)
        if isinstance(sub, dict):
            return sub
        sub2 = cat.get("sub_emotions")
        return sub2 if isinstance(sub2, dict) else {}

    def _iter_pairs(self):
        """
        내부 전이쌍 이터레이터.
        (key, from_id, to_id, from_data, to_data, parent_from_cat, parent_to_cat)
        """
        # 대표 ↔ 대표
        for f in self.primary_emotions:
            for t in self.primary_emotions:
                yield ( (f, t), f, t, self.emotions_data.get(f, {}) or {}, self.emotions_data.get(t, {}) or {}, f, t )

        # 세부 ↔ 세부
        for f_cat in self.primary_emotions:
            fsubs = self._sub_emotions(f_cat)
            if not isinstance(fsubs, dict):
                continue
            for f_name, f_data in fsubs.items():
                for t_cat in self.primary_emotions:
                    tsubs = self._sub_emotions(t_cat)
                    if not isinstance(tsubs, dict):
                        continue
                    for t_name, t_data in tsubs.items():
                        yield ( (f_name, t_name), f_name, t_name, f_data or {}, t_data or {}, f_cat, t_cat )

    # --------------------------------------------------------------------- #
    # core                                                                  #
    # --------------------------------------------------------------------- #
    def _pair_weight(
        self,
        *,
        from_id: str,
        to_id: str,
        from_data: Dict[str, Any],
        to_data: Dict[str, Any],
        parent_from_cat: Optional[str],
        parent_to_cat: Optional[str],
    ) -> float:
        cfg = self.config

        # 1) 복잡도 → base
        fc = self._safe_get(from_data, "metadata.emotion_complexity", "basic")
        tc = self._safe_get(to_data, "metadata.emotion_complexity", "basic")
        cs = cfg["complexity_scores"]
        base = (cs.get(str(fc), 1.0) + cs.get(str(tc), 1.0)) / 2.0

        # 2) 카테고리 동일/상이
        f_cat = self._safe_get(from_data, "metadata.primary_category", parent_from_cat)
        t_cat = self._safe_get(to_data, "metadata.primary_category", parent_to_cat)
        cat_mult = cfg["category_multipliers"]["same"] if f_cat and t_cat and f_cat == t_cat else cfg["category_multipliers"]["diff"]

        # 3) 강도 예시 분포 차이(hi/med/low)
        f_hi, f_md, f_lo = self._intensity_example_counts(from_data)
        t_hi, t_md, t_lo = self._intensity_example_counts(to_data)
        diff = abs(f_hi - t_hi) + 0.5 * abs(f_md - t_md) + 0.25 * abs(f_lo - t_lo)
        inten_mult = max(0.6, 1.0 - diff * float(cfg["intensity_diff_factor"]))

        # 4) 키워드 유사도(자카드)
        f_kw = self._collect_keywords(from_data)
        t_kw = self._collect_keywords(to_data)
        jac = self._jaccard(f_kw, t_kw)
        kw_mult = (1.0 + jac * float(cfg["keyword_sim_weight"]))

        # 5) progression/prior(방향성 사전확률)
        prior = self._lookup_transition_prior(from_id, to_id, from_data, to_data)
        prior_mult = (1.0 + max(0.0, min(1.0, prior)) * float(cfg["prior_weight"]))

        # 6) 명시적 상극/비호환
        conflict = self._is_conflicting_pair(from_id, to_id, from_data, to_data)
        conf_mult = (1.0 - float(cfg["conflict_penalty"])) if conflict else 1.0

        # 7) 발렌스 보정
        fv = self._infer_valence(from_data, fallback_cat=f_cat)
        tv = self._infer_valence(to_data, fallback_cat=t_cat)
        if fv and tv:
            val_mult = cfg["valence_same_multiplier"] if fv == tv else cfg["valence_diff_multiplier"]
        else:
            val_mult = 1.0

        # 가중치 합성
        weight = base * cat_mult * inten_mult * kw_mult * prior_mult * conf_mult * val_mult

        # 클램프 & 반올림
        weight = max(float(cfg["min_weight"]), min(float(cfg["max_weight"]), weight))
        return round(weight, 3)

    # --------------------------------------------------------------------- #
    # feature extractors                                                    #
    # --------------------------------------------------------------------- #
    def _intensity_example_counts(self, emo: Dict[str, Any]) -> Tuple[int, int, int]:
        """
        emotion_profile.intensity_levels.intensity_examples.{high,medium,low}
        의 예시 길이를 집계. 없으면 0.
        """
        ex = self._safe_get(emo, "emotion_profile.intensity_levels.intensity_examples", {}) or {}
        def _len(x): return len(x) if isinstance(x, (list, tuple)) else 0
        return _len(ex.get("high")), _len(ex.get("medium")), _len(ex.get("low"))

    def _collect_keywords(self, emo: Dict[str, Any]) -> Set[str]:
        """
        core_keywords/keywords 와 situations/examples/variations 등에서
        토큰을 넓게 수집(라벨링 뼈대 활용, 하드코딩 최소화).
        """
        toks: Set[str] = set()
        token_re = re.compile(r"[가-힣A-Za-z]{%d,}" % int(self.config["token_min_len"]))

        def add_from_field(x: Any):
            if isinstance(x, str):
                toks.update(token_re.findall(x))
            elif isinstance(x, (list, tuple)):
                for it in x:
                    add_from_field(it)
            elif isinstance(x, dict):
                for v in x.values():
                    add_from_field(v)

        # 핵심 키워드
        for path in (
            "emotion_profile.core_keywords",
            "core_keywords",
            "emotion_profile.keywords",
            "keywords",
        ):
            add_from_field(self._safe_get(emo, path, []))

        # 상황/예시
        cp = self._safe_get(emo, "context_patterns", {}) or {}
        add_from_field(self._safe_get(cp, "situations", {}))
        add_from_field(self._safe_get(cp, "examples", []))
        add_from_field(self._safe_get(cp, "variations", []))

        # 정규화(불필요 조사 제거 / 소문자)
        out: Set[str] = set()
        for t in toks:
            base = self._normalize_token(t)
            if base:
                out.add(base)
        return out

    @staticmethod
    @lru_cache(maxsize=50000)
    def _normalize_token(tok: str) -> Optional[str]:
        if not tok:
            return None
        
        # [Optimization] Use regex directly (re.sub compiles internally, but simple pattern is fast)
        # Or even better, filter without regex if possible, but regex is fine for Python's C implementation.
        # Removing [^가-힣A-Za-z]
        t = re.sub(r"[^가-힣A-Za-z]", "", tok)
        
        if not t:
            return None
        # 매우 짧은 토큰 제외
        if len(t) < 2:
            return None
            
        # 종결 조사 제거(과/와/을/를/은/는/이/가/의 등)
        # Common suffixes optimization
        suffixes = ("들과", "과의", "와의", "으로", "에게", "에서", "라고", "이라", "와", "과", "은", "는", "이", "가", "을", "를", "의")
        
        # Fast check before loop
        for suf in suffixes:
            if t.endswith(suf):
                if len(t) > len(suf) + 1:
                    t = t[: -len(suf)]
                break
                
        return t.lower()

    @staticmethod
    def _jaccard(a: Set[str], b: Set[str]) -> float:
        if not a or not b:
            return 0.0
        inter = len(a & b)
        if inter == 0:
            return 0.0
        union = len(a | b)
        return inter / union if union else 0.0

    def _lookup_transition_prior(self, from_id: str, to_id: str, f: Dict[str, Any], t: Dict[str, Any]) -> float:
        """
        라벨에 사전확률이 명시돼 있으면 반영.
        가능한 위치(유연):
          - f["transition_priors"][to_id] or t["transition_priors_from"][from_id]
          - top-level self.emotions_data["transition_priors"][from_id][to_id]
        없으면 0.0
        """
        # 감정 자신의 prior
        p = self._safe_get(f, f"transition_priors.{to_id}", None)
        if isinstance(p, (int, float)):
            return max(0.0, min(1.0, float(p)))

        p = self._safe_get(t, f"transition_priors_from.{from_id}", None)
        if isinstance(p, (int, float)):
            return max(0.0, min(1.0, float(p)))

        # 전역 prior
        glob = self._safe_get(self.emotions_data, f"transition_priors.{from_id}.{to_id}", None)
        if isinstance(glob, (int, float)):
            return max(0.0, min(1.0, float(glob)))

        return 0.0

    def _is_conflicting_pair(self, from_id: str, to_id: str, f: Dict[str, Any], t: Dict[str, Any]) -> bool:
        """
        명시적 상극/비호환 관계가 JSON에 있으면 True.
        가능한 위치(유연):
          - f["incompatibilities"] / f["conflicts"] 리스트에 to_id 포함
          - t["incompatibilities_from"] 리스트에 from_id 포함
          - 전역 self.emotions_data["incompatibilities"][(from_id, to_id)] 등
        """
        for path in ( "incompatibilities", "conflicts", "antagonisms" ):
            arr = self._safe_get(f, path, []) or []
            if isinstance(arr, list) and to_id in arr:
                return True

        arr = self._safe_get(t, "incompatibilities_from", []) or []
        if isinstance(arr, list) and from_id in arr:
            return True

        # 전역(딕셔너리/리스트 유연 처리)
        glob = self._safe_get(self.emotions_data, "incompatibilities", None)
        if isinstance(glob, dict):
            if glob.get(from_id) == to_id:  # 단일 매핑
                return True
            # dict of lists
            vals = glob.get(from_id, [])
            if isinstance(vals, list) and to_id in vals:
                return True
        elif isinstance(glob, list):
            # [ ["분노","용서"], ... ] 꼴
            for it in glob:
                if isinstance(it, (list, tuple)) and len(it) == 2 and it[0] == from_id and it[1] == to_id:
                    return True
        return False

    def _infer_valence(self, emo: Dict[str, Any], *, fallback_cat: Optional[str]) -> Optional[str]:
        """
        메타에 명시된 valence가 있으면 사용, 없으면 대표 카테고리 기반으로 보수 추정.
        (가능하면 JSON의 metadata.valence를 넣어주세요: "positive"/"negative"/"mixed")
        """
        v = self._safe_get(emo, "metadata.valence", None)
        if isinstance(v, str) and v:
            v = v.strip().lower()
            if v in ("positive", "negative", "mixed"):
                return v

        # 보수적 추정: 대표 카테고리→valence
        # JSON에 일치하는 값이 있으면 그걸 따릅니다.
        cat = self._safe_get(emo, "metadata.primary_category", fallback_cat)
        if not isinstance(cat, str):
            return None

        cat = cat.strip()
        # 가능하면 JSON의 카테고리 메타에서 유추
        meta_val = self._safe_get(self.emotions_data.get(cat, {}), "metadata.valence", None)
        if isinstance(meta_val, str) and meta_val.lower() in ("positive", "negative", "mixed"):
            return meta_val.lower()

        # 최후의 보수적 디폴트(대표4 감정 기준)
        if cat in ("희", "락"):
            return "positive"
        if cat in ("노", "애"):
            return "negative"
        return None


# =============================================================================
# TemporalPatternAnalyzer
# =============================================================================
class TemporalPatternAnalyzer:

    DEFAULT_CONFIG: Dict[str, Any] = {
        # 복잡도 타입별 가중치 (없으면 basic)
        "complexity_weights": {
            "basic":  {"trend": 1.0, "variation": 1.0, "cycle": 0.8},
            "subtle": {"trend": 1.1, "variation": 1.0, "cycle": 1.0},
            "complex":{"trend": 1.1, "variation": 1.2, "cycle": 1.2},
        },

        # 전역 스무딩/샘플링
        "smoothing_window": 3,         # 이동평균 창(>=1)
        "cap_sequence_len": 2048,      # 입력이 더 길면 다운샘플
        "dump_arrays": False,          # True면 일부 배열을 결과에 포함(디버그)
        "max_points_dump": 256,        # dump_arrays가 True일 때 최대 포인트 수

        # 흐름 요약 덤프 상한
        "max_windows_dump": 64,        # intensity_progression 항목 상한
        "max_transitions_dump": 32,    # transition_points 항목 상한

        # FFT/주기 분석
        "min_len_fft": 6,              # FFT 최소 길이
        "cycle_norm_eps": 1e-9,        # 0 나눗셈 방지

        # 전개 단계 추정(전역 타임라인 기반)
        "stage_boundaries": (0.20, 0.65),  # (begin→development, development→climax 시작 후보)
        "stage_slope_sensitivity": 0.75,   # 기울기 영향 비중 (0~1)

        # 발산값 클리핑
        "clip_min": 0.0,
        "clip_max": 1.0,
    }

    def __init__(self, emotions_data: Dict[str, Any], config: Optional[Dict[str, Any]] = None):
        self.emotions_data = emotions_data or {}
        self.config = {**TemporalPatternAnalyzer.DEFAULT_CONFIG, **(config or {})}

    # ──────────────────────────────────────────────────────────────────────
    # public
    # ──────────────────────────────────────────────────────────────────────
    def analyze(self, emotion_sequence: List[Dict[str, float]]) -> Dict[str, Any]:
        """
        emotion_sequence 형식 허용:
          - [{ "희":0.3,"노":0.2,... }, ...]
          - [{ "emotions": {"희":..., "노":...}, ...}, ...]  ← 자동 보정

        반환:
        {
          "complexity_analysis": {emotion_id: {...}},
          "emotion_flow": {emotion_id: {...}},
          "global": {
            "length": int,
            "dominant_emotion_by_step": [str,...],
            "stage_by_step": ["beginning"|...],
            "stage_signals": {"beginning":x,"development":y,"climax":z,"aftermath":w}
          }
          (+ dump_arrays True면 "timeline_debug" 추가)
        }
        """
        # 0) 전처리/보정
        seq = self._coerce_sequence(emotion_sequence)
        if len(seq) < 2:
            return {"complexity_analysis": {}, "emotion_flow": {}, "global": {"length": len(seq)}}

        seq = self._downsample_if_needed(seq, self.config["cap_sequence_len"])
        seq = self._clip_seq(seq, self.config["clip_min"], self.config["clip_max"])
        seq = self._smooth_seq(seq, max(1, int(self.config["smoothing_window"])))

        # 1) 감정 ID 집합
        all_emotion_ids = sorted({k for row in seq for k in row.keys()})

        # 2) 감정별 분석
        complexity: Dict[str, Any] = {}
        flows: Dict[str, Any] = {}
        for eid in all_emotion_ids:
            intens = np.array([float(row.get(eid, 0.0) or 0.0) for row in seq], dtype=np.float32)
            complexity[eid] = self._calculate_complexity_score(eid, intens)
            if len(intens) >= 3:
                flows[eid] = self._analyze_emotion_flow(intens)
            else:
                flows[eid] = {"intensity_progression": [], "transition_points": [], "stability_indices": []}

        # 3) 전역 타임라인/전개 단계 추정
        global_curve, dominant_by_step = self._build_global_curve(seq)
        stage_by_step, stage_signals = self._estimate_stages(global_curve)

        out: Dict[str, Any] = {
            "complexity_analysis": complexity,
            "emotion_flow": self._trim_flow(flows),
            "global": {
                "length": len(seq),
                "dominant_emotion_by_step": dominant_by_step,
                "stage_by_step": stage_by_step,
                "stage_signals": stage_signals,
            },
        }

        if self.config.get("dump_arrays", False):
            # 디버그 배열은 길이를 제한
            max_dump = int(self.config.get("max_points_dump", 256))
            debug_seq = [
                {k: float(v) for k, v in row.items()}
                for row in seq[:max_dump]
            ]
            out["timeline_debug"] = {
                "sequence_head": debug_seq,
                "global_curve_head": [float(x) for x in global_curve[:max_dump]],
            }

        return out

    # ──────────────────────────────────────────────────────────────────────
    # sequence helpers
    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def _coerce_sequence(raw_seq: List[Dict[str, Any]]) -> List[Dict[str, float]]:
        """다양한 입력 포맷을 표준 형태로 보정."""
        seq: List[Dict[str, float]] = []
        for item in (raw_seq or []):
            if isinstance(item, dict) and "emotions" in item and isinstance(item["emotions"], dict):
                seq.append({str(k): float(item["emotions"].get(k, 0.0) or 0.0) for k in item["emotions"].keys()})
            elif isinstance(item, dict):
                # 감정 강도 분석 결과 형태 처리 (Dict[str, Dict[str, Any]] -> Dict[str, float])
                coerced_item = {}
                for k, v in item.items():
                    if isinstance(v, dict):
                        # modified_score 또는 score 필드 추출
                        if 'modified_score' in v:
                            coerced_item[str(k)] = float(v.get('modified_score', 0.0) or 0.0)
                        elif 'score' in v:
                            coerced_item[str(k)] = float(v.get('score', 0.0) or 0.0)
                        elif 'base_score' in v:
                            coerced_item[str(k)] = float(v.get('base_score', 0.0) or 0.0)
                        else:
                            # 딕셔너리 값의 첫 번째 숫자 값 사용
                            numeric_values = [float(val) for val in v.values() if isinstance(val, (int, float))]
                            coerced_item[str(k)] = numeric_values[0] if numeric_values else 0.0
                    else:
                        # 직접 숫자 값인 경우
                        try:
                            coerced_item[str(k)] = float(v or 0.0)
                        except (TypeError, ValueError):
                            coerced_item[str(k)] = 0.0
                seq.append(coerced_item)
        return seq

    @staticmethod
    def _clip_seq(seq: List[Dict[str, float]], vmin: float, vmax: float) -> List[Dict[str, float]]:
        out = []
        for row in seq:
            out.append({k: float(min(max(vmin, v), vmax)) for k, v in row.items()})
        return out

    @staticmethod
    def _downsample_if_needed(seq: List[Dict[str, float]], max_len: int) -> List[Dict[str, float]]:
        n = len(seq)
        if n <= int(max_len):
            return seq
        # 간단한 등간격 다운샘플
        step = math.ceil(n / float(max_len))
        return [seq[i] for i in range(0, n, step)]

    def _smooth_seq(self, seq: List[Dict[str, float]], win: int) -> List[Dict[str, float]]:
        if win <= 1:
            return seq
        keys = sorted({k for row in seq for k in row.keys()})
        mat = np.array([[float(row.get(k, 0.0) or 0.0) for k in keys] for row in seq], dtype=np.float32)
        sm = self._moving_average(mat, win)
        out: List[Dict[str, float]] = []
        for i in range(sm.shape[0]):
            out.append({keys[j]: float(sm[i, j]) for j in range(sm.shape[1])})
        return out

    @staticmethod
    def _moving_average(mat: np.ndarray, win: int) -> np.ndarray:
        """누적합을 활용한 이동 평균을 계산한다. 경계 구간에서는 사용 가능한 길이만큼만 평균을 낸다."""
        if win <= 1:
            return mat
        n, d = mat.shape
        half = win // 2
        indices = np.arange(n)
        starts = np.maximum(0, indices - half)
        ends = np.minimum(n, indices + half + 1)
        cumsum = np.vstack([np.zeros((1, d), dtype=mat.dtype), np.cumsum(mat, axis=0)])
        sums = cumsum[ends] - cumsum[starts]
        lengths = (ends - starts).reshape(-1, 1).astype(mat.dtype, copy=False)
        lengths[lengths == 0] = 1
        out = sums / lengths
        return out

    # ──────────────────────────────────────────────────────────────────────
    # per-emotion metrics
    # ──────────────────────────────────────────────────────────────────────
    def _calculate_complexity_score(self, emotion_id: str, intens: np.ndarray) -> Dict[str, Any]:
        """
        trend/variation/cycle을 라벨 복잡도 가중치에 따라 결합.
        주기 기대치(라벨에 있으면) 가까울수록 보너스.
        """
        if intens.size == 0:
            return {
                "complexity_type": "basic",
                "weighted_trend": 0.0,
                "weighted_variation": 0.0,
                "cycle_strength": 0.0,
                "composite_score": 0.0,
            }

        # 라벨 기반 복잡도 타입 및 가중치
        meta = (self.emotions_data.get(emotion_id, {}) or {}).get("metadata", {}) or {}
        complexity_type = str(meta.get("emotion_complexity", "basic"))
        weights_cfg: Dict[str, Dict[str, float]] = self.config.get("complexity_weights", {})
        weight_map: Dict[str, float] = weights_cfg.get(complexity_type, weights_cfg.get("basic", {}))
        w_trend = float(weight_map.get("trend", 1.0))
        w_var   = float(weight_map.get("variation", 1.0))
        w_cycle = float(weight_map.get("cycle", 1.0))

        # 추세(기울기): polyfit 실패 대비
        try:
            slope = float(np.polyfit(np.arange(intens.size, dtype=np.float32), intens, 1)[0]) if intens.size > 1 else 0.0
        except Exception:
            slope = 0.0
        trend_val = slope * w_trend

        # 변동성: 표준편차
        var_val = float(np.std(intens)) * w_var

        # 주기 강도: FFT 기반 정규화 진폭(DC 제외)
        cycle_val = 0.0
        if intens.size >= int(self.config["min_len_fft"]):
            cycle_val = self._cycle_strength(intens)
            # 라벨 기대 주기와의 적합도(있으면)
            exp_period, tolerance = self._expected_period_meta(emotion_id)
            if exp_period is not None and cycle_val > 0.0:
                closeness = self._period_closeness(intens, exp_period, tolerance)
                # 주기 강도와 적합도를 결합(보너스)
                cycle_val *= (0.9 + 0.2 * closeness)

            cycle_val *= w_cycle

        comp = float(np.mean([trend_val, var_val, cycle_val]))
        return {
            "complexity_type": complexity_type,
            "weighted_trend": float(trend_val),
            "weighted_variation": float(var_val),
            "cycle_strength": float(cycle_val),
            "composite_score": float(comp),
        }

    def _cycle_strength(self, intens: np.ndarray) -> float:
        x = intens.astype(np.float32)
        x = x - x.mean()
        n = x.size
        if n < 2:
            return 0.0
        fft = np.fft.rfft(x)
        amp = np.abs(fft)
        # DC(0) 제외, 정상화
        if amp.size <= 1:
            return 0.0
        nz = amp[1:]
        peak = float(nz.max()) if nz.size else 0.0
        denom = float(nz.mean() + self.config["cycle_norm_eps"])
        return peak / denom if denom > 0 else 0.0

    def _expected_period_meta(self, emotion_id: str) -> Tuple[Optional[int], float]:
        """
        라벨에서 기대 주기 길이(스텝)와 허용 오차를 찾는다.
        가능한 경로를 유연하게 탐색.
        """
        meta = (self.emotions_data.get(emotion_id, {}) or {}).get("metadata", {}) or {}
        # 여러 경로 후보
        paths = [
            ("temporal.expected_period", "temporal.tolerance"),
            ("expected_period", "tolerance"),
            ("temporal_patterns.expected_period", "temporal_patterns.tolerance"),
        ]
        for p_per, p_tol in paths:
            per = self._safe_get(meta, p_per, None)
            tol = self._safe_get(meta, p_tol, 0.15)
            try:
                if per is not None:
                    per_i = int(per)
                    tol_f = float(tol)
                    if per_i > 1:
                        return per_i, max(0.0, min(0.9, tol_f))
            except Exception:
                continue
        return None, 0.15

    @staticmethod
    def _safe_get(obj: Any, path: str, default: Any = None):
        cur = obj
        for part in path.split("."):
            if isinstance(cur, dict):
                cur = cur.get(part)
            else:
                return default
        return default if cur is None else cur

    def _period_closeness(self, intens: np.ndarray, expected_period: int, tol: float) -> float:
        """
        추정된 주요 주기(피크 주파수에서의 주기)와 기대 주기의 근접도(0~1).
        """
        n = intens.size
        if n < 3 or expected_period <= 1:
            return 0.0
        x = intens.astype(np.float32) - float(np.mean(intens))
        fft = np.fft.rfft(x)
        amp = np.abs(fft)
        if amp.size <= 2:
            return 0.0
        # 1..end 중 최대 주파수 인덱스 → 주기 = n / k
        k = int(np.argmax(amp[1:]) + 1)
        if k == 0:
            return 0.0
        est_period = max(2.0, float(n) / float(k))
        # 상대 오차가 tol 이내면 1, 아니면 선형 하락
        rel_err = abs(est_period - expected_period) / max(2.0, float(expected_period))
        if rel_err <= tol:
            return 1.0
        # tol~2*tol 사이에서 1→0 선형 감쇠, 이후 0
        if rel_err >= 2*tol:
            return 0.0
        return float(1.0 - (rel_err - tol) / max(1e-9, tol))

    # ──────────────────────────────────────────────────────────────────────
    # flows & transitions
    # ──────────────────────────────────────────────────────────────────────
    def _analyze_emotion_flow(self, intens: np.ndarray) -> Dict[str, Any]:
        """
        감정 강도 시퀀스 흐름 요약(창 기반 패턴, 전이 포인트, 안정성).
        대용량 방지를 위해 결과 덤프를 제한합니다.
        """
        flow = {"intensity_progression": [], "transition_points": [], "stability_indices": []}
        n = intens.size
        if n == 0:
            return flow

        win = min(3, n)
        max_w = int(self.config["max_windows_dump"])
        max_tp = int(self.config["max_transitions_dump"])

        # 창 기반 패턴
        cnt_w = 0
        for i in range(n - win + 1):
            if cnt_w >= max_w:
                break
            window = intens[i:i + win]
            diffs = np.diff(window)
            if np.all(diffs > 0):
                pattern = "increasing"
            elif np.all(diffs < 0):
                pattern = "decreasing"
            elif np.all(np.abs(diffs) < 1e-6):
                pattern = "stable"
            else:
                pattern = "fluctuating"

            flow["intensity_progression"].append({
                "window_start": int(i),
                "pattern": pattern,
                "intensity_values": [float(x) for x in window]
            })
            cnt_w += 1

        # 전이 포인트(기울기 부호 전환 지점)
        cnt_t = 0
        for i in range(1, n - 1):
            if cnt_t >= max_tp:
                break
            prev_diff = float(intens[i] - intens[i - 1])
            next_diff = float(intens[i + 1] - intens[i])
            if np.sign(prev_diff) != np.sign(next_diff):
                change_mag = abs(prev_diff - next_diff)
                tp_type = "peak" if prev_diff > 0 and next_diff < 0 else ("valley" if prev_diff < 0 and next_diff > 0 else "turn")
                flow["transition_points"].append({
                    "position": int(i),
                    "type": tp_type,
                    "intensity": float(intens[i]),
                    "change_magnitude": float(change_mag)
                })
                cnt_t += 1

        # 안정성 지수(롤링 표준편차 → 1/(1+std))
        stabs: List[float] = []
        for i in range(1, n - win + 1):
            window = intens[i:i + win]
            std_dev = float(np.std(window))
            stabs.append(1.0 / (1.0 + std_dev + 1e-9))
        flow["stability_indices"] = [float(x) for x in stabs[:max_w]]

        return flow

    # ──────────────────────────────────────────────────────────────────────
    # global curve & stages
    # ──────────────────────────────────────────────────────────────────────
    def _build_global_curve(self, seq: List[Dict[str, float]]) -> Tuple[np.ndarray, List[str]]:
        """
        스텝별 전역 강도 곡선과, 스텝별 지배 감정 ID를 반환.
        전역 곡선은 (평균 + 최대)/2 혼합으로 계산(너무 들쭉하지 않게).
        """
        keys = sorted({k for row in seq for k in row.keys()})
        mat = np.array([[float(row.get(k, 0.0)) for k in keys] for row in seq], dtype=np.float32)
        mean_curve = mat.mean(axis=1)
        max_curve = mat.max(axis=1)
        global_curve = 0.5 * (mean_curve + max_curve)

        # 지배 감정
        dom_ids: List[str] = []
        argmax_idx = np.argmax(mat, axis=1)
        for i in range(mat.shape[0]):
            dom_ids.append(str(keys[int(argmax_idx[i])]))
        return global_curve, dom_ids

    def _estimate_stages(self, global_curve: np.ndarray) -> Tuple[List[str], Dict[str, float]]:
        """
        전개 4단계(begin/development/climax/aftermath) 추정.
        - 시간 진행 비율(u)과 기울기(slope)를 혼합해 소프트 신호 계산
        - 각 스텝의 argmax 단계 라벨을 stage_by_step로 제공
        - 전체 신호는 평균으로 요약
        """
        n = global_curve.size
        if n == 0:
            return [], {"beginning": 0.0, "development": 0.0, "climax": 0.0, "aftermath": 0.0}

        # 정규화
        g = global_curve.astype(np.float32)
        gmin, gmax = float(g.min()), float(g.max())
        if gmax - gmin > 1e-9:
            gn = (g - gmin) / (gmax - gmin)
        else:
            gn = g * 0.0

        # 기울기
        slope = np.gradient(gn)
        slope_abs = np.abs(slope)

        # 경계
        b1, b2 = self.config["stage_boundaries"]  # 0~1 기반
        u = np.linspace(0.0, 1.0, n, dtype=np.float32)
        slope_w = float(self.config["stage_slope_sensitivity"])

        # 소프트 신호
        beg_sig = np.clip(1.0 - (u / (b1 + 1e-6)), 0.0, 1.0) * (1.0 - slope_w * slope_abs)
        dev_sig = np.clip((u - b1) / max(1e-6, (b2 - b1)), 0.0, 1.0) * (0.5 + slope_w * slope_abs)
        # 클라이맥스는 최대치와 큰 양의 기울기 직전/직후 강조
        peak_idx = int(np.argmax(gn))
        peak_mask = np.exp(-0.5 * ((np.arange(n) - peak_idx) / max(1.0, 0.15 * n)) ** 2)
        clm_sig = (0.5 * peak_mask + 0.5 * gn) * (0.75 + 0.25 * slope_abs)
        aft_sig = (1.0 - gn) * np.clip((u - b2) / max(1e-6, 1.0 - b2), 0.0, 1.0) * (1.0 - 0.5 * slope_abs)

        # 정규화
        stack = np.vstack([beg_sig, dev_sig, clm_sig, aft_sig]) + 1e-12
        stack = stack / stack.max(axis=0, keepdims=True)

        labels = ["beginning", "development", "climax", "aftermath"]
        stage_idx = stack.argmax(axis=0)
        stage_by_step = [labels[int(i)] for i in stage_idx]
        stage_signals = {labels[i]: float(stack[i].mean()) for i in range(4)}
        return stage_by_step, stage_signals

    # ──────────────────────────────────────────────────────────────────────
    # output trimmer
    # ──────────────────────────────────────────────────────────────────────
    def _trim_flow(self, flows: Dict[str, Any]) -> Dict[str, Any]:
        """
        결과 JSON이 과도하게 커지지 않도록 각 항목 상한을 재확인(방어적).
        """
        mw = int(self.config["max_windows_dump"])
        mt = int(self.config["max_transitions_dump"])
        out: Dict[str, Any] = {}
        for k, v in flows.items():
            if not isinstance(v, dict):
                continue
            prog = v.get("intensity_progression", []) or []
            trans = v.get("transition_points", []) or []
            stab  = v.get("stability_indices", []) or []
            out[k] = {
                "intensity_progression": list(prog[:mw]),
                "transition_points": list(trans[:mt]),
                "stability_indices": [float(x) for x in stab[:mw]],
            }
        return out


# =============================================================================
# TransitionStatisticsCalculator
# =============================================================================
class TransitionStatisticsCalculator:
    DEFAULT_CONFIG: Dict[str, Any] = {
        "pattern_thresholds": {"intensifying": 0.2, "diminishing": -0.2, "fluctuating": 0.3},
        "significance": {
            "base": 1.0,
            "trigger_weight": 1.0,
            "std_weight": 1.0,
            "count_weight": 0.6,
            "category": {"same": 1.1, "diff": 0.9},
        },
        "primary_emotions": ["희", "노", "애", "락"],
        "cap_history": 8,
        "min_count": 1,
        "eps": 1e-9,
        "top_k": 10,
    }

    def __init__(self, config: Dict[str, Any]):
        self.config = {**TransitionStatisticsCalculator.DEFAULT_CONFIG, **(config or {})}
        self.emotions_data: Dict[str, Any] = self.config.get("emotions_data", {}) or {}
        self._cat_index = self._build_category_index()

    def calculate(
        self,
        matrix: Dict[Tuple[str, str], List[float]],
        triggers: Dict[str, int],
    ) -> Dict[str, Any]:
        transition_details = self._analyze_matrix(matrix, triggers)
        transition_details = self._normalize_significance(transition_details)
        significant_changes = self._find_significant_changes(transition_details)
        flow_patterns = self._calculate_flow_patterns(transition_details)
        temporal_analysis = self._perform_temporal_analysis(significant_changes, flow_patterns)

        return {
            "transitions": transition_details,
            "significant_changes": significant_changes[: self.config["top_k"]],
            "emotion_flow_patterns": flow_patterns,
            "temporal_analysis": temporal_analysis,
        }

    def _build_category_index(self) -> Dict[str, str]:
        idx = {}
        prim = set(self.config.get("primary_emotions", []))
        for p in prim:
            idx[p] = p
            sub = ((self.emotions_data.get(p) or {}).get("sub_emotions") or {}).keys()
            for s in sub:
                idx[str(s)] = p
        return idx

    def _emotion_category(self, eid: str) -> Optional[str]:
        return self._cat_index.get(eid)

    def _cat_relation(self, f: str, t: str) -> str:
        cf, ct = self._emotion_category(f), self._emotion_category(t)
        if cf and ct and cf == ct:
            return "same"
        return "diff"

    def _key_str(self, f: str, t: str) -> str:
        return f"{f}->{t}"

    def _analyze_matrix(
        self,
        matrix: Dict[Tuple[str, str], List[float]],
        triggers: Dict[str, int],
    ) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        total_triggers = max(sum(triggers.values()), 1)
        th = self.config["pattern_thresholds"]
        cap = int(self.config["cap_history"])
        min_count = int(self.config["min_count"])
        eps = float(self.config["eps"])
        sig_cfg = self.config["significance"]

        for k, changes in (matrix or {}).items():
            if not isinstance(k, (list, tuple)) or len(k) != 2:
                continue
            f, t = str(k[0]), str(k[1])
            if not changes:
                continue

            arr = np.asarray(changes, dtype=np.float32)
            avg_change = float(np.mean(arr))
            std_dev = float(np.std(arr))
            cnt = int(arr.size)
            if cnt < min_count:
                continue

            if avg_change > float(th.get("intensifying", 0.2)):
                pattern = "intensifying"
            elif avg_change < float(th.get("diminishing", -0.2)):
                pattern = "diminishing"
            elif std_dev > float(th.get("fluctuating", 0.3)):
                pattern = "fluctuating"
            else:
                pattern = "stable"

            trig_key = self._key_str(f, t)
            trig_cnt = int(triggers.get(trig_key, 0))
            cat_rel = self._cat_relation(f, t)

            std_term = 1.0 + float(sig_cfg["std_weight"]) * min(1.0, std_dev)
            trig_term = 1.0 + float(sig_cfg["trigger_weight"]) * (trig_cnt / float(total_triggers))
            cnt_term = 1.0 + float(sig_cfg["count_weight"]) * math.log1p(cnt)
            cat_mul = float(sig_cfg["category"].get(cat_rel, 1.0))

            raw_sig = abs(avg_change) * float(sig_cfg["base"]) * std_term * trig_term * cnt_term * cat_mul
            stability = 1.0 / (1.0 + std_dev + eps)
            direction = "up" if avg_change > 0 else ("down" if avg_change < 0 else "flat")

            out[trig_key] = {
                "from": f,
                "to": t,
                "category_relation": cat_rel,
                "average_change": round(avg_change, 3),
                "standard_deviation": round(std_dev, 3),
                "count": cnt,
                "trigger_frequency": trig_cnt,
                "pattern": pattern,
                "direction": direction,
                "significance": round(float(raw_sig), 3),
                "stability": round(float(stability), 3),
                "change_history": [round(float(x), 3) for x in arr[-cap:].tolist()],
            }

        return out

    def _normalize_significance(self, details: Dict[str, Any]) -> Dict[str, Any]:
        if not details:
            return details
        vals = [float(v.get("significance", 0.0)) for v in details.values()]
        mx = max(vals) if vals else 0.0
        for k, v in details.items():
            s = float(v.get("significance", 0.0))
            v["norm_significance"] = round(float(s / mx), 3) if mx > 0 else 0.0
            cnt = int(v.get("count", 0))
            std = float(v.get("standard_deviation", 0.0))
            conf = (1.0 / (1.0 + std)) * (1.0 / (1.0 + math.exp(-(cnt - 2) / 3.0)))
            v["confidence"] = round(float(conf), 3)
        return details

    def _find_significant_changes(self, transition_details: Dict[str, Any]) -> List[Dict[str, Any]]:
        th = float(self.config.get("significance_threshold", 0.5))
        items = []
        for key, info in transition_details.items():
            sig = float(info.get("norm_significance", info.get("significance", 0.0)))
            if sig > th:
                items.append({
                    "transition": key,
                    "significance": round(sig, 3),
                    "pattern": info.get("pattern", "stable"),
                    "direction": info.get("direction", "flat"),
                })
        items.sort(key=lambda x: x["significance"], reverse=True)
        return items

    def _calculate_flow_patterns(self, transition_details: Dict[str, Any]) -> Dict[str, Any]:
        patterns = defaultdict(int)
        for info in transition_details.values():
            patterns[info.get("pattern", "stable")] += 1
        total = sum(patterns.values()) or 1
        return {p: {"count": c, "percentage": round(100.0 * c / total, 1)} for p, c in patterns.items()}

    def _perform_temporal_analysis(
        self,
        significant_changes: List[Dict[str, Any]],
        flow_patterns: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not flow_patterns:
            return {
                "transition_frequency": 0,
                "average_significance": 0.0,
                "dominant_pattern": "stable",
                "pattern_diversity": 0,
                "balance_index": 0.0,
                "volatility_index": 0.0,
            }

        dom = max(flow_patterns.items(), key=lambda it: it[1]["count"])[0]
        avg_sig = float(np.mean([c["significance"] for c in significant_changes])) if significant_changes else 0.0
        diversity = len([1 for p in flow_patterns.values() if p["count"] > 0])

        inc = flow_patterns.get("intensifying", {}).get("count", 0)
        dim = flow_patterns.get("diminishing", {}).get("count", 0)
        bal = (inc - dim) / float(max(1, inc + dim))

        fluc_pct = flow_patterns.get("fluctuating", {}).get("percentage", 0.0)
        vol = float(fluc_pct) / 100.0

        return {
            "transition_frequency": len(significant_changes),
            "average_significance": round(avg_sig, 3),
            "dominant_pattern": dom,
            "pattern_diversity": diversity,
            "balance_index": round(float(bal), 3),
            "volatility_index": round(float(vol), 3),
        }


# =============================================================================
# EmotionIntensityTransformer
# =============================================================================
class EmotionIntensityTransformer:
    def __init__(self, emotions_data: Dict[str, Any], config: Optional[Dict[str, Any]] = None):
        self.emotions_data = emotions_data or {}

        base = EmotionIntensityTransformer._get_default_config()
        if config:
            for k, v in config.items():
                if isinstance(v, dict) and isinstance(base.get(k), dict):
                    base[k].update(v)
                else:
                    base[k] = v
        self.config = base

        # calculators
        self.weight_calculator = TransitionWeightCalculator(self.emotions_data, self.config["weights"])
        self.temporal_analyzer = TemporalPatternAnalyzer(self.emotions_data, self.config["temporal"])
        self.stats_calculator = TransitionStatisticsCalculator(
            {**self.config["statistics"], "emotions_data": self.emotions_data}
        )

        # caches / matrices
        self.transition_matrix: Dict[Tuple[str, str], List[float]] = defaultdict(list)
        self.sub_emotion_transition_matrix: Dict[Tuple[str, str], List[float]] = defaultdict(list)
        self.transition_triggers: Dict[str, int] = defaultdict(int)
        self.transition_weights_map = self.weight_calculator.generate()
        self._primary_index = self._build_primary_index()

    @staticmethod
    def _get_default_config() -> Dict[str, Any]:
        return {
            "weights": {
                "primary_emotions": ["희", "노", "애", "락"],
                "complexity_scores": {"basic": 1.0, "subtle": 1.2, "complex": 1.4},
                "category_multipliers": {"same": 1.1, "diff": 0.9},
                "intensity_diff_factor": 0.1,
            },
            "temporal": {
                "complexity_weights": {
                    "basic": {"trend": 1.0, "variation": 1.0, "cycle": 1.0},
                    "subtle": {"trend": 1.2, "variation": 1.3, "cycle": 1.2},
                    "complex": {"trend": 1.5, "variation": 1.6, "cycle": 1.4},
                }
            },
            "statistics": {
                "significance_threshold": 0.5,
                "pattern_thresholds": {"intensifying": 0.2, "diminishing": -0.2, "fluctuating": 0.3},
                "top_k": 10,
            },
            "update": {
                "related_emotion_multiplier": 1.2,
                "history_cap": 200,            # per transition pair
                "clip_max_abs": 2.5,           # clamp after weighting/normalization
                "normalize": {"method": "tanh", "gain": 1.0},  # "none" | "tanh"
            },
            "sequence": {"dominant_top_k": 1, "min_delta": 1e-4},
        }

    # ---------- public ----------
    def analyze_temporal_patterns(self, emotion_sequence: List[Dict[str, float]]) -> Dict[str, Any]:
        return self.temporal_analyzer.analyze(emotion_sequence)

    def feed_sequence(self, emotion_sequence: List[Dict[str, float]]) -> None:
        """
        연속된 시점의 감정 분포에서 전이 기록을 추출해 행렬을 갱신합니다.
        규칙: 이전-현재 각 시점의 '지배 감정'(top_k) 쌍으로 전이 기록.
        """
        if not isinstance(emotion_sequence, list) or len(emotion_sequence) < 2:
            return
        top_k = int(self.config["sequence"]["dominant_top_k"])
        min_delta = float(self.config["sequence"]["min_delta"])

        def top_items(d: Dict[str, float], k: int) -> List[Tuple[str, float]]:
            arr = [(str(a), float(b)) for a, b in d.items() if self._is_finite(b)]
            arr.sort(key=lambda x: x[1], reverse=True)
            return arr[: max(1, k)] if arr else []

        for i in range(1, len(emotion_sequence)):
            prev, cur = emotion_sequence[i - 1], emotion_sequence[i]
            prev_tops = top_items(prev, top_k)
            cur_tops = top_items(cur, top_k)
            if not prev_tops or not cur_tops:
                continue

            # 간단: top-1 ↔ top-1 전이
            f_id, f_val = prev_tops[0]
            t_id, t_val = cur_tops[0]
            delta = float(t_val - f_val)
            if abs(delta) >= min_delta:
                self.update_transition_matrix(f_id, t_id, delta, is_sub_emotion=self._is_sub_emotion(f_id) or self._is_sub_emotion(t_id))

    def update_transition_matrix(
        self,
        from_emotion: str,
        to_emotion: str,
        intensity_change: float,
        is_sub_emotion: Optional[bool] = None,
    ) -> float:
        """
        전이 한 건을 행렬에 반영하고, 실제 저장된 변화량을 반환합니다.
        """
        f = str(from_emotion or "").strip()
        t = str(to_emotion or "").strip()
        if not f or not t:
            return 0.0

        val = float(intensity_change or 0.0)
        if not self._is_finite(val):
            return 0.0

        if is_sub_emotion is None:
            is_sub_emotion = self._is_sub_emotion(f) or self._is_sub_emotion(t)

        matrix = self.sub_emotion_transition_matrix if is_sub_emotion else self.transition_matrix
        key = (f, t)

        # weight
        w = float(self.transition_weights_map.get((f, t), self._fallback_pair_weight(f, t)))
        mod = val * w

        # related-emotion boost (primary만 적용)
        if not is_sub_emotion and self._are_related(f, t):
            mod *= float(self.config["update"]["related_emotion_multiplier"])

        # normalize + clip
        mod = self._normalize_change(mod)
        mod = self._clip_change(mod)

        # append with cap
        cap = int(self.config["update"]["history_cap"])
        bucket = matrix[key]
        bucket.append(mod)
        if len(bucket) > cap:
            del bucket[: len(bucket) - cap]

        self.transition_triggers[f"{f}->{t}"] += 1
        return mod

    def get_transition_statistics(self) -> Dict[str, Any]:
        stats_primary = self.stats_calculator.calculate(self.transition_matrix, self.transition_triggers)
        stats_sub = self.stats_calculator.calculate(self.sub_emotion_transition_matrix, self.transition_triggers)

        out = {
            **stats_primary,
            "sub_transitions": stats_sub.get("transitions", {}),
        }

        # merge significant_changes
        merged_sig = (stats_primary.get("significant_changes", []) or []) + (
            stats_sub.get("significant_changes", []) or []
        )
        merged_sig.sort(key=lambda x: x.get("significance", 0.0), reverse=True)
        top_k = int(self.config["statistics"].get("top_k", 10))
        out["significant_changes"] = merged_sig[:top_k]
        return out

    def reset(self) -> None:
        self.transition_matrix.clear()
        self.sub_emotion_transition_matrix.clear()
        self.transition_triggers.clear()

    # ---------- internals ----------
    def _build_primary_index(self) -> Dict[str, str]:
        idx: Dict[str, str] = {}
        for p_id, p_data in (self.emotions_data or {}).items():
            if not isinstance(p_data, dict):
                continue
            p_id = str(p_id)
            idx[p_id] = p_id
            sub = (p_data.get("sub_emotions") or {}) if isinstance(p_data.get("sub_emotions"), dict) else {}
            for s_id in sub.keys():
                idx[str(s_id)] = p_id
        return idx

    def _build_sub_phrase_index(self) -> Dict[str, Dict[str, Set[str]]]:
        index: Dict[str, Dict[str, Set[str]]] = {}
        for pid, pdata in (self.emotions_data or {}).items():
            if not isinstance(pdata, dict):
                continue
            subs = pdata.get("sub_emotions") or {}
            if not isinstance(subs, dict):
                continue
            bucket: Dict[str, Set[str]] = {}
            for sid, sdata in subs.items():
                if not isinstance(sdata, dict):
                    continue
                vocab: Set[str] = set()
                prof = (sdata.get("emotion_profile") or {})
                for key in ("core_keywords", "keywords"):
                    for kw in (prof.get(key) or []):
                        if isinstance(kw, str):
                            kw_norm = kw.strip().lower()
                            if len(kw_norm) >= 2:
                                vocab.add(kw_norm)
                            vocab.update({token for token in self._tokenize_and_normalize(kw) if len(token) >= 2})
                intensity_examples = (prof.get("intensity_levels") or {}).get("intensity_examples", {})
                if isinstance(intensity_examples, dict):
                    for examples in intensity_examples.values():
                        for ex in (examples or []):
                            if isinstance(ex, str):
                                vocab.update({token for token in self._tokenize_and_normalize(ex) if len(token) >= 2})
                cp = (sdata.get("context_patterns") or {})
                situations = cp.get("situations") or {}
                if isinstance(situations, dict):
                    for sit in situations.values():
                        if not isinstance(sit, dict):
                            continue
                        for kw in (sit.get("keywords") or []):
                            if isinstance(kw, str):
                                kw_norm = kw.strip().lower()
                                if len(kw_norm) >= 2:
                                    vocab.add(kw_norm)
                        for ex in (sit.get("examples") or []):
                            if isinstance(ex, str):
                                vocab.update({token for token in self._tokenize_and_normalize(ex) if len(token) >= 2})
                        for var in (sit.get("variations") or []):
                            if isinstance(var, str):
                                vocab.update({token for token in self._tokenize_and_normalize(var) if len(token) >= 2})
                        progression = sit.get("emotion_progression") or {}
                        if isinstance(progression, dict):
                            for desc in progression.values():
                                if isinstance(desc, str):
                                    vocab.update({token for token in self._tokenize_and_normalize(desc) if len(token) >= 2})
                bucket[str(sid)] = {tok for tok in vocab if len(tok) >= 2}
            if bucket:
                index[str(pid)] = bucket
        return index

    def _candidate_subs(self, primary_id: str, token_set: Set[str], topk: int = None) -> List[str]:
        # 환경변수에서 Top-K 설정 가져오기 (기본값 12)
        if topk is None:
            topk = int(os.environ.get("INT_TOPK_SUB", "12"))
        bucket = self._sub_phrase_index.get(str(primary_id), {})
        if not bucket:
            return []
        hits = [(sid, len(vocab & token_set)) for sid, vocab in bucket.items() if vocab & token_set]
        if not hits:
            return list(bucket.keys())[: max(1, topk)]
        hits.sort(key=lambda x: x[1], reverse=True)
        return [sid for sid, _ in hits[: max(1, topk)]]

    @staticmethod
    def _resolve_sub_entry(sub_map: Dict[Any, Any], sub_id: str) -> Tuple[Optional[Any], Optional[Dict[str, Any]]]:
        if sub_id in sub_map:
            return sub_id, sub_map[sub_id]
        for key, value in sub_map.items():
            if str(key) == sub_id:
                return key, value
        return None, None


    def _is_sub_emotion(self, eid: str) -> bool:
        pid = self._primary_index.get(str(eid))
        return bool(pid and pid != str(eid))

    def _resolve_primary(self, eid: str) -> Optional[str]:
        return self._primary_index.get(str(eid))

    def _fallback_pair_weight(self, f: str, t: str) -> float:
        cs = self.config["weights"]["complexity_scores"]
        cm = self.config["weights"]["category_multipliers"]
        fc = self._emotion_complexity(f)
        tc = self._emotion_complexity(t)
        same_cat = (self._resolve_primary(f) == self._resolve_primary(t))
        base = (cs.get(fc, 1.0) + cs.get(tc, 1.0)) / 2.0
        mul = cm["same"] if same_cat else cm["diff"]
        return max(0.1, min(2.0, round(base * mul, 3)))

    def _emotion_complexity(self, eid: str) -> str:
        data = self._lookup_emotion_data(eid)
        meta = data.get("metadata", {}) if isinstance(data, dict) else {}
        return str(meta.get("emotion_complexity", "basic"))

    def _lookup_emotion_data(self, eid: str) -> Dict[str, Any]:
        eid = str(eid)
        if eid in self.emotions_data:
            return self.emotions_data.get(eid, {}) or {}
        pid = self._resolve_primary(eid)
        if not pid:
            return {}
        return ((self.emotions_data.get(pid) or {}).get("sub_emotions") or {}).get(eid, {}) or {}

    def _are_related(self, f: str, t: str) -> bool:
        f_data = self._lookup_emotion_data(f)
        prof = f_data.get("emotion_profile", {}) if isinstance(f_data, dict) else {}
        rel = prof.get("related_emotions", {}) if isinstance(prof, dict) else {}
        for _, arr in (rel.items() if isinstance(rel, dict) else []):
            try:
                if t in (arr or []):
                    return True
            except Exception:
                continue
        return False

    def _normalize_change(self, x: float) -> float:
        norm = self.config["update"]["normalize"] or {}
        method = (norm.get("method") or "none").lower()
        if method == "tanh":
            gain = float(norm.get("gain", 1.0))
            return math.tanh(gain * float(x))
        return float(x)

    def _clip_change(self, x: float) -> float:
        m = float(self.config["update"]["clip_max_abs"])
        if m <= 0:
            return float(x)
        return float(max(-m, min(m, x)))

    @staticmethod
    def _is_finite(x: Any) -> bool:
        try:
            xf = float(x)
            return math.isfinite(xf)
        except Exception:
            return False

# =============================================================================
# EmotionProfileScorer
# =============================================================================
class EmotionProfileScorer:
    """
    EMOTIONS.json 내 세부감정의 emotion_profile을 사용해 텍스트와의 적합도를 점수화.
    - 강도별 예시 매칭 (상위 k개만 사용해 과도한 누적 방지)
    - 핵심/보조 키워드 매칭
    - 관련 감정(극성 일치 시) 매칭
    """

    def __init__(self, tokenize_func, cfg: Optional[Dict[str, Any]] = None):
        self.tokenize = tokenize_func
        cfg = cfg or {}
        self.level_weights = cfg.get("level_weights", {"high": 1.8, "medium": 1.5, "low": 1.2})
        self.keyword_weight: float = float(cfg.get("keyword_weight", 1.5))
        self.related_weight: float = float(cfg.get("related_emotion_weight", 1.3))
        self.example_top_k: int = int(cfg.get("example_top_k", 5))
        self.min_token_len: int = int(cfg.get("min_token_len", 2))
        # 후처리(선택): 점수 정규화 옵션
        norm = cfg.get("normalize", {"method": "none", "gain": 1.0})
        self.norm_method = str(norm.get("method", "none")).lower()
        self.norm_gain = float(norm.get("gain", 1.0))
        self.max_score: Optional[float] = (
            float(cfg["max_score"]) if isinstance(cfg.get("max_score"), (int, float)) else None
        )

    # ---------- public ----------
    def score(self, sub_emotion_data: Dict[str, Any], text_tokens: List[str], is_positive: bool) -> Tuple[float, List[str]]:
        profile = sub_emotion_data.get("emotion_profile", {}) if isinstance(sub_emotion_data, dict) else {}
        if not profile:
            return 0.0, []

        tset = self._norm_tokens(text_tokens)
        if not tset:
            return 0.0, []

        total_score = 0.0
        hits: Set[str] = set()

        levels = (profile.get("intensity_levels") or {})
        examples_by_level = (levels.get("intensity_examples") or {}) if isinstance(levels, dict) else {}
        for level, ex_list in (examples_by_level.items() if isinstance(examples_by_level, dict) else []):
            w = float(self.level_weights.get(level, 1.0))
            sc, ex_hits = self._score_examples(ex_list, tset, base_weight=w)
            total_score += sc
            hits.update(ex_hits)

        ck = set(self._safe_list(profile.get("core_keywords")))
        kw = set(self._safe_list(profile.get("keywords")))
        kset = self._norm_tokens(list(ck | kw))
        if kset:
            matched = kset & tset
            if matched:
                total_score += self.keyword_weight * self._precision_like(tset, kset)
                hits.update(matched)

        rel = profile.get("related_emotions", {}) if isinstance(profile, dict) else {}
        if isinstance(rel, dict):
            for emo_type, phrases in rel.items():
                if (emo_type == "positive" and is_positive) or (emo_type == "negative" and not is_positive) or (emo_type == "neutral"):
                    sc, rel_hits = self._score_examples(phrases, tset, base_weight=self.related_weight)
                    total_score += sc
                    hits.update(rel_hits)

        total_score = self._normalize(total_score)
        if self.max_score is not None:
            total_score = max(0.0, min(self.max_score, total_score))
        return float(total_score), sorted(hits)


    # ---------- internals ----------
    def _score_examples(self, examples: Any, tset: Set[str], base_weight: float) -> Tuple[float, Set[str]]:
        entries: List[Tuple[float, Set[str]]] = []
        for ex in self._safe_list(examples):
            ex_tokens = self._norm_tokens(self.tokenize(ex))
            if not ex_tokens:
                continue
            overlap = tset & ex_tokens
            if not overlap:
                continue
            entries.append((base_weight * self._f1_like(tset, ex_tokens), overlap))

        if not entries:
            return 0.0, set()

        entries.sort(key=lambda x: x[0], reverse=True)
        take = entries[: max(1, self.example_top_k)]
        total = sum(score for score, _ in take)
        hits: Set[str] = set()
        for _, ov in take:
            hits.update(ov)
        return float(total), hits


    @staticmethod
    def _safe_list(x: Any) -> List[str]:
        if isinstance(x, list):
            return [str(t) for t in x if isinstance(t, (str, int, float))]
        if isinstance(x, (str, int, float)):
            return [str(x)]
        return []

    def _norm_tokens(self, tokens: List[str]) -> Set[str]:
        out: Set[str] = set()
        for t in (tokens or []):
            if not isinstance(t, str):
                t = str(t)
            tt = t.strip().lower()
            if len(tt) >= self.min_token_len:
                out.add(tt)
        return out

    @staticmethod
    def _overlap(a: Set[str], b: Set[str]) -> int:
        return len(a & b)

    def _precision_like(self, text_set: Set[str], ref_set: Set[str]) -> float:
        if not ref_set:
            return 0.0
        return self._overlap(text_set, ref_set) / len(ref_set)

    def _f1_like(self, a: Set[str], b: Set[str]) -> float:
        if not a or not b:
            return 0.0
        inter = self._overlap(a, b)
        if inter == 0:
            return 0.0
        p = inter / len(b)
        r = inter / len(a)
        # F1의 근사치로 기하평균 사용(안정적)
        return math.sqrt(max(0.0, p) * max(0.0, r))

    def _normalize(self, x: float) -> float:
        if self.norm_method == "tanh":
            return math.tanh(self.norm_gain * float(x))
        return float(x)


# =============================================================================
# SituationScorer
# =============================================================================
class SituationScorer:
    """
    EMOTIONS.json의 context_patterns.situations를 활용해 텍스트와 상황 적합도를 점수화.
    - description/variations/keywords/progression의 부분일치 기반 가중합
    - required/exclude 키워드 지원(존재 시 엄격 필터)
    - situation.weight / intensity / stage 가중치 반영
    """

    def __init__(self, tokenize_func, cfg: Optional[Dict[str, Any]] = None):
        self.tok = tokenize_func
        self.cfg = cfg or {}
        self.min_token_len: int = int(self.cfg.get("min_token_len", 2))

        # 가중치/임계치
        self.intensity_mul = self.cfg.get("intensity_multiplier", {"high": 1.5, "medium": 1.2, "low": 1.0})
        self.stage_mul = self.cfg.get(
            "stage_weights",
            {"trigger": 1.1, "development": 1.3, "peak": 1.5, "aftermath": 0.9},
        )
        self.weights = self.cfg.get(
            "weights",
            {
                "description": 1.0,
                "variations": 1.0,
                "keywords": 1.0,
                "progression": 1.0,
            },
        )
        self.variation_match_threshold: float = float(self.cfg.get("variation_match_threshold", 0.6))
        self.keyword_weight_per_hit: float = float(self.cfg.get("keyword_weight_per_hit", 0.8))
        self.desc_ratio_gain: float = float(self.cfg.get("description_ratio_gain", 1.0))

    # ---------- public ----------
    def score(self, sub_emotion_data: Dict[str, Any], text_tokens: List[str]) -> float:
        situations = {}
        if isinstance(sub_emotion_data, dict):
            cp = sub_emotion_data.get("context_patterns", {}) or {}
            situations = cp.get("situations", {}) or {}
        if not isinstance(situations, dict) or not situations:
            return 0.0

        tset = self._norm_tokens(text_tokens)
        if not tset:
            return 0.0

        total = 0.0
        for s in situations.values():
            s_score = self._score_one_situation(s, tset)
            total += s_score
        return float(total)

    # ---------- internals ----------
    def _score_one_situation(self, situation: Dict[str, Any], tset: Set[str]) -> float:
        if not isinstance(situation, dict):
            return 0.0

        # 필수/제외 키워드(있으면 우선 필터)
        req = self._norm_tokens(self._safe_list(situation.get("required_keywords")))
        exc = self._norm_tokens(self._safe_list(situation.get("exclude_keywords")))
        if req and not (tset & req):
            return 0.0
        if exc and (tset & exc):
            return 0.0

        # 기본 가중치
        base_w = float(situation.get("weight", 1.0))
        intensity = str(situation.get("intensity", "medium")).lower()
        intensity_w = float(self.intensity_mul.get(intensity, 1.0))

        score = 0.0
        score += self.weights["description"] * self._score_description(situation, tset)
        score += self.weights["variations"] * self._score_variations(situation, tset)
        score += self.weights["keywords"] * self._score_keywords(situation, tset)
        score += self.weights["progression"] * self._score_progression(situation, tset)

        return float(score * base_w * intensity_w)

    def _score_description(self, situation: Dict[str, Any], tset: Set[str]) -> float:
        desc_tokens = self._norm_tokens(self.tok(situation.get("description", "")))
        if not desc_tokens:
            return 0.0
        # 부분일치 비율(설명 길이가 길면 과도 불이익 방지 위해 gain 적용)
        ratio = self._overlap_ratio(tset, desc_tokens) * self.desc_ratio_gain
        return float(ratio)

    def _score_variations(self, situation: Dict[str, Any], tset: Set[str]) -> float:
        total = 0.0
        for v in self._safe_list(situation.get("variations")):
            vt = self._norm_tokens(self.tok(v))
            if not vt:
                continue
            r = self._overlap_ratio(tset, vt)
            if r >= self.variation_match_threshold:
                total += 1.2  # 기본 가중 이득(경험적)
            else:
                # 부분 일치도 일부 반영
                total += 0.6 * r
        return float(total)

    def _score_keywords(self, situation: Dict[str, Any], tset: Set[str]) -> float:
        kws = self._norm_tokens(self._safe_list(situation.get("keywords")))
        if not kws:
            return 0.0
        hit = len(kws & tset)
        if hit == 0:
            return 0.0
        # 키워드는 개수 기반 가산
        return float(self.keyword_weight_per_hit * hit)

    def _score_progression(self, situation: Dict[str, Any], tset: Set[str]) -> float:
        prog = situation.get("emotion_progression", {}) if isinstance(situation, dict) else {}
        if not isinstance(prog, dict) or not prog:
            return 0.0
        sc = 0.0
        for stage, desc in prog.items():
            dt = self._norm_tokens(self.tok(desc))
            if not dt:
                continue
            r = self._overlap_ratio(tset, dt)
            if r <= 0.0:
                continue
            sc += r * float(self.stage_mul.get(stage, 1.0))
        return float(sc)

    # ---------- small utils ----------
    @staticmethod
    def _safe_list(x: Any) -> List[str]:
        if isinstance(x, list):
            return [str(t) for t in x if isinstance(t, (str, int, float))]
        if isinstance(x, (str, int, float)):
            return [str(x)]
        return []

    def _norm_tokens(self, tokens: List[str]) -> Set[str]:
        out: Set[str] = set()
        for t in (tokens or []):
            if not isinstance(t, str):
                t = str(t)
            tt = t.strip().lower()
            if len(tt) >= self.min_token_len:
                out.add(tt)
        return out

    @staticmethod
    def _overlap_ratio(a: Set[str], b: Set[str]) -> float:
        if not a or not b:
            return 0.0
        inter = len(a & b)
        # Jaccard 유사도에 가까운 안정적 비율
        denom = len(a | b)
        return inter / denom if denom else 0.0


# =============================================================================
# TransitionScorer
# =============================================================================
class TransitionScorer:
    _NEG_BOUND = r"(?<![가-힣A-Za-z0-9]){tok}(?![가-힣A-Za-z0-9])"

    @staticmethod
    @lru_cache(maxsize=256)
    def _neg_pat(tok: str) -> re.Pattern:
        return re.compile(TransitionScorer._NEG_BOUND.format(tok=re.escape(tok)))

    def __init__(self, tokenize_func, cfg: Optional[Dict[str, Any]] = None):
        self.tok = tokenize_func
        base_negs = ["않", "못", "아니", "없", "no", "not", "n't"]
        self.cfg = {
            "trigger_base": 1.2,
            "shift_bonus": 0.5,
            "all_triggers_bonus": 0.4,
            "keyword_hit_bonus": 0.2,
            "negation_penalty": 0.6,           # 부정 페널티
            "window_bonus": 0.4,
            "window_size_chars": 12,
            "min_token_len": 2,
            "direction_match_bonus": 0.3,      # (옵션) 방향성과 일치한 경우 추가
            "negations": base_negs.copy(),
        }

        if cfg:
            self.cfg.update(cfg)

    # public -----------------------------------------------------------------
    def score(self, sub_emotion_data: Dict[str, Any],
              text_tokens: List[str], text_lower: str) -> float:
        patterns = {}
        if isinstance(sub_emotion_data, dict):
            et = sub_emotion_data.get("emotion_transitions", {}) or {}
            patterns = et.get("patterns", {}) or et.get("patterns", [])
        if not patterns:
            return 0.0

        tset = self._norm_tokens(text_tokens)
        txt = str(text_lower or "").lower()
        total = 0.0

        # patterns: list[dict] 또는 dict[name]->dict 모두 허용
        iterable = patterns.items() if isinstance(patterns, dict) else enumerate(patterns)
        for _, p in iterable:
            if not isinstance(p, dict):
                continue

            # 트리거 판정(any/all/not 지원)
            trig_ok, hits, all_ok = self._triggers_satisfied(p.get("triggers"), tset)
            if not trig_ok:
                continue

            base = float(self.cfg["trigger_base"]) * float(p.get("weight", 1.0))
            sc = base

            # all 트리거 모두 충족 시 추가 보너스
            if all_ok:
                sc += float(self.cfg["all_triggers_bonus"])

            # 키워드 히트(있다면 소량 가산)
            sc += self._keywords_bonus(p.get("keywords"), tset)

            # 전환 포인트 문구 탐지
            shift_pt = str(
                ((p.get("transition_analysis") or {}).get("emotion_shift_point") or "")
            ).lower()
            if shift_pt and shift_pt in txt:
                sc += float(self.cfg["shift_bonus"])

            # (옵션) from/to 방향정보가 패턴에 존재하면, 텍스트에 두 극성 단서가 함께 있을 때 보너스
            sc += self._direction_hint_bonus(p, tset)

            # 근접(window) 보너스: anchors와 target이 가까운 경우
            sc += self._window_bonus(p.get("window"), txt)

            # 부정어 패널티
            if self._has_negation(txt):
                sc *= float(self.cfg["negation_penalty"])

            total += max(0.0, sc)

        return float(total)

    # internals --------------------------------------------------------------
    def _norm_tokens(self, tokens: List[str]) -> Set[str]:
        out: Set[str] = set()
        m = int(self.cfg["min_token_len"])
        for t in (tokens or []):
            if not isinstance(t, str):
                t = str(t)
            tt = t.strip().lower()
            if len(tt) >= m:
                out.add(tt)
        return out

    def _tokset(self, text: str) -> Set[str]:
        return self._norm_tokens(self.tok(text or ""))

    def _phrase_in_tset(self, phrase: str, tset: Set[str]) -> bool:
        pt = self._tokset(phrase)
        return bool(pt) and pt.issubset(tset)

    def _triggers_satisfied(self, triggers: Any, tset: Set[str]) -> Tuple[bool, int, bool]:
        """
        triggers 형식:
          - ["하지만", "생각해보니"]  → any-매치
          - {"any":[...], "all":[...], "not":[...]}  → 논리 지원
        반환: (충족여부, 매치수, all조건충족여부)
        """
        if not triggers:
            return False, 0, False

        any_list, all_list, not_list = [], [], []
        if isinstance(triggers, dict):
            any_list = triggers.get("any") or []
            all_list = triggers.get("all") or []
            not_list = triggers.get("not") or []
        elif isinstance(triggers, list):
            any_list = triggers
        else:
            any_list = [str(triggers)]

        any_hit = sum(1 for x in any_list if self._phrase_in_tset(str(x), tset))
        all_ok = all(self._phrase_in_tset(str(x), tset) for x in all_list) if all_list else False
        not_ok = any(self._phrase_in_tset(str(x), tset) for x in not_list)

        # not 조건 위배 시 실패
        if not_list and not_ok:
            return False, 0, False

        # all 조건이 존재하면 그것을 우선 충족해야 함
        if all_list:
            return (all_ok, any_hit + (len(all_list) if all_ok else 0), all_ok)

        # 그 외에는 any 매치 1개 이상이면 OK
        return (any_hit > 0, any_hit, False)

    def _keywords_bonus(self, keywords: Any, tset: Set[str]) -> float:
        if not keywords:
            return 0.0
        hits = 0
        if isinstance(keywords, (list, set, tuple)):
            for k in keywords:
                if self._phrase_in_tset(str(k), tset):
                    hits += 1
        elif isinstance(keywords, str):
            if self._phrase_in_tset(keywords, tset):
                hits = 1
        return float(hits * self.cfg["keyword_hit_bonus"])

    def _direction_hint_bonus(self, pattern: Dict[str, Any], tset: Set[str]) -> float:
        ta = pattern.get("transition_analysis") or {}
        frm = str(ta.get("from") or "").strip()
        to_ = str(ta.get("to") or "").strip()
        if not (frm or to_):
            return 0.0
        # from/to에 대략적인 단서(키워드)가 있으면 각각 한 번씩 만족 여부 확인
        bonus = 0.0
        if frm and self._phrase_in_tset(frm, tset):
            bonus += float(self.cfg["direction_match_bonus"]) * 0.5
        if to_ and self._phrase_in_tset(to_, tset):
            bonus += float(self.cfg["direction_match_bonus"]) * 0.5
        return bonus

    def _window_bonus(self, window_cfg: Any, text_lower: str) -> float:
        if not isinstance(window_cfg, dict):
            return 0.0
        anchors = window_cfg.get("anchors") or []
        targets = window_cfg.get("targets") or []
        if not anchors or not targets:
            return 0.0

        span = int(window_cfg.get("span", self.cfg["window_size_chars"]))
        txt = text_lower or ""
        best = 0.0

        # 간단한 문자열 위치 근사(최초 등장 인덱스 기반)
        for a in anchors:
            ai = txt.find(str(a).lower())
            if ai < 0:
                continue
            for b in targets:
                bi = txt.find(str(b).lower())
                if bi < 0:
                    continue
                if abs(ai - bi) <= span:
                    best = max(best, 1.0)

        return float(best * self.cfg["window_bonus"])

    def _has_negation(self, text_lower: str) -> bool:
        if not text_lower:
            return False
        tl = text_lower.lower()
        for n in self.cfg.get("negations", []):
            if TransitionScorer._neg_pat(str(n).lower()).search(tl):
                return True
        return False


# =============================================================================
# EmotionDataLoader
# =============================================================================
class EmotionDataLoader:
    _cache: Dict[str, Tuple[float, int, Dict[str, Any]]] = {}
    _lock = RLock()

    def __init__(self, path: str, autogen: bool = False, *, allow_gzip: bool = True, strict: bool = False):
        self.path = os.path.abspath(path)
        self.autogen = autogen
        self.allow_gzip = allow_gzip
        self.strict = strict

    # ---------- public ----------
    def load(self) -> Dict[str, Any]:
        real_path = self._resolve_path(self.path, self.allow_gzip)

        # autogen when missing
        if real_path is None:
            if not self.autogen:
                raise FileNotFoundError(self.path)
            data = self._skeleton()
            self._write(self.path, data)
            real_path = self._resolve_path(self.path, self.allow_gzip)

        assert real_path is not None, "Internal: resolved path must not be None"

        mtime = os.path.getmtime(real_path)
        fsize = os.path.getsize(real_path)

        with EmotionDataLoader._lock:
            cached = EmotionDataLoader._cache.get(real_path)
            if cached and cached[0] == mtime and cached[1] == fsize:
                return cached[2]

        data = self._read_json(real_path)
        data = self._validate_and_repair(data)  # repair-in-place for older schemas

        with EmotionDataLoader._lock:
            EmotionDataLoader._cache[real_path] = (mtime, fsize, data)
        return data

    # ---------- internals ----------
    @staticmethod
    def _resolve_path(path: str, allow_gzip: bool) -> Optional[str]:
        if os.path.exists(path):
            return path
        if allow_gzip and os.path.exists(path + ".gz"):
            return path + ".gz"
        return None

    @staticmethod
    def _read_json(path: str) -> Dict[str, Any]:
        if path.endswith(".gz"):
            with gzip.open(path, "rt", encoding="utf-8") as f:
                return json.load(f)
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _ensure_dict(x: Any) -> Dict[str, Any]:
        return x if isinstance(x, dict) else {}

    @staticmethod
    def _ensure_list(x: Any) -> List[Any]:
        return x if isinstance(x, list) else []

    def _validate_and_repair(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(data, dict):
            if self.strict:
                raise ValueError("EMOTIONS.json must be an object.")
            data = {}

        required = {"희", "노", "애", "락"}
        missing = [e for e in required if e not in data]
        if missing:
            if self.strict:
                raise ValueError(f"primary emotions missing: {', '.join(missing)}")
            # create missing primaries
            for e in missing:
                data[e] = self._category_default(primary_id=e)

        # normalize each primary & sub
        for prim in ("희", "노", "애", "락"):
            data[prim] = self._normalize_category(prim, self._ensure_dict(data.get(prim)))

        return data

    # ---- category normalization ----
    def _normalize_category(self, primary_id: str, cat: Dict[str, Any]) -> Dict[str, Any]:
        # metadata
        meta = self._ensure_dict(cat.get("metadata"))
        meta.setdefault("emotion_id", primary_id)
        meta.setdefault("primary_category", primary_id)
        meta.setdefault("sub_category", "")
        meta.setdefault("emotion_complexity", "basic")
        meta.setdefault("version", "1.0")
        cat["metadata"] = meta

        # emotion_profile
        prof = self._ensure_dict(cat.get("emotion_profile"))
        prof.setdefault("core_keywords", [])
        relem = self._ensure_dict(prof.get("related_emotions"))
        relem.setdefault("positive", [])
        relem.setdefault("negative", [])
        prof["related_emotions"] = relem

        # intensity_levels → intensity_examples dict(low/medium/high)
        ilevels = self._ensure_dict(prof.get("intensity_levels"))
        iex = ilevels.get("intensity_examples")
        if isinstance(iex, dict):
            pass
        elif isinstance(iex, list):
            # if it was a flat list, put them in "medium"
            iex = {"low": [], "medium": iex, "high": []}
        else:
            iex = {"low": [], "medium": [], "high": []}
        # ensure lists
        for k in ("low", "medium", "high"):
            iex[k] = self._ensure_list(iex.get(k))
        ilevels["intensity_examples"] = iex
        prof["intensity_levels"] = ilevels
        cat["emotion_profile"] = prof

        # context_patterns.situations
        cp = self._ensure_dict(cat.get("context_patterns"))
        cp.setdefault("situations", {})
        cat["context_patterns"] = cp

        # linguistic_patterns
        cat["linguistic_patterns"] = self._ensure_dict(cat.get("linguistic_patterns"))

        # emotion_transitions.patterns (unify to list; keep fallback to dict readable upstream)
        et = self._ensure_dict(cat.get("emotion_transitions"))
        patterns = et.get("patterns")
        if isinstance(patterns, dict):
            # convert dict→list keeping values
            patterns = [self._ensure_dict(v) for v in patterns.values()]
        elif isinstance(patterns, list):
            patterns = [self._ensure_dict(p) for p in patterns]
        else:
            patterns = []
        et["patterns"] = patterns
        cat["emotion_transitions"] = et

        # sentiment/ML metadata
        cat["sentiment_analysis"] = self._ensure_dict(cat.get("sentiment_analysis"))
        cat["ml_training_metadata"] = self._ensure_dict(cat.get("ml_training_metadata"))

        # sub_emotions: enforce dict[name]->subcat
        sub = cat.get("sub_emotions")
        if isinstance(sub, list):
            # list of names → build minimal children
            sub = {str(name): self._category_default(primary_id=primary_id, sub_id=str(name)) for name in sub}
        elif not isinstance(sub, dict):
            sub = {}
        else:
            # normalize each sub
            normalized = {}
            for sub_id, sub_cat in sub.items():
                normalized[str(sub_id)] = self._normalize_sub(primary_id, str(sub_id), self._ensure_dict(sub_cat))
            sub = normalized
        cat["sub_emotions"] = sub

        # dedupe & sanitize keywords
        prof["core_keywords"] = sorted({str(k).strip() for k in self._ensure_list(prof.get("core_keywords")) if k})
        relem["positive"] = sorted({str(k).strip() for k in self._ensure_list(relem.get("positive")) if k})
        relem["negative"] = sorted({str(k).strip() for k in self._ensure_list(relem.get("negative")) if k})

        return cat

    def _normalize_sub(self, primary_id: str, sub_id: str, sub: Dict[str, Any]) -> Dict[str, Any]:
        meta = self._ensure_dict(sub.get("metadata"))
        meta.setdefault("emotion_id", sub_id)
        meta.setdefault("primary_category", primary_id)
        meta.setdefault("sub_category", sub_id)
        meta.setdefault("emotion_complexity", meta.get("emotion_complexity", "basic"))
        meta.setdefault("version", "1.0")
        sub["metadata"] = meta

        prof = self._ensure_dict(sub.get("emotion_profile"))
        prof.setdefault("core_keywords", [])
        ilevels = self._ensure_dict(prof.get("intensity_levels"))
        iex = ilevels.get("intensity_examples")
        if isinstance(iex, dict):
            pass
        elif isinstance(iex, list):
            iex = {"low": [], "medium": iex, "high": []}
        else:
            iex = {"low": [], "medium": [], "high": []}
        for k in ("low", "medium", "high"):
            iex[k] = self._ensure_list(iex.get(k))
        ilevels["intensity_examples"] = iex
        prof["intensity_levels"] = ilevels

        relem = self._ensure_dict(prof.get("related_emotions"))
        relem.setdefault("positive", [])
        relem.setdefault("negative", [])
        prof["related_emotions"] = relem
        sub["emotion_profile"] = prof

        cp = self._ensure_dict(sub.get("context_patterns"))
        cp.setdefault("situations", {})
        sub["context_patterns"] = cp

        et = self._ensure_dict(sub.get("emotion_transitions"))
        pats = et.get("patterns")
        if isinstance(pats, dict):
            pats = [self._ensure_dict(v) for v in pats.values()]
        elif isinstance(pats, list):
            pats = [self._ensure_dict(v) for v in pats]
        else:
            pats = []
        et["patterns"] = pats
        sub["emotion_transitions"] = et

        sub["linguistic_patterns"] = self._ensure_dict(sub.get("linguistic_patterns"))
        sub["sentiment_analysis"] = self._ensure_dict(sub.get("sentiment_analysis"))
        sub["ml_training_metadata"] = self._ensure_dict(sub.get("ml_training_metadata"))

        # keyword dedupe
        prof["core_keywords"] = sorted({str(k).strip() for k in self._ensure_list(prof.get("core_keywords")) if k})
        relem["positive"] = sorted({str(k).strip() for k in self._ensure_list(relem.get("positive")) if k})
        relem["negative"] = sorted({str(k).strip() for k in self._ensure_list(relem.get("negative")) if k})

        return sub

    # ---- defaults / skeleton ----
    def _category_default(self, primary_id: str, sub_id: Optional[str] = None) -> Dict[str, Any]:
        meta = dict(
            emotion_id=sub_id or primary_id,
            primary_category=primary_id,
            sub_category=sub_id or "",
            emotion_complexity="basic",
            version="1.0",
        )
        profile = dict(
            intensity_levels=dict(
                intensity_examples={"low": [], "medium": [], "high": []}
            ),
            related_emotions=dict(positive=[], negative=[]),
            core_keywords=[],
        )
        return dict(
            metadata=meta,
            emotion_profile=profile,
            context_patterns=dict(situations={}),
            linguistic_patterns={},
            emotion_transitions=dict(patterns=[]),
            sentiment_analysis={},
            ml_training_metadata={},
            sub_emotions={} if sub_id is None else None,  # primaries have children; subs don't
        )

    def _skeleton(self) -> Dict[str, Any]:
        skel = {}
        for e in ("희", "노", "애", "락"):
            skel[e] = self._category_default(primary_id=e)
        return skel

    # ---- write ----
    def _write(self, path: str, data: Dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if path.endswith(".gz"):
            with gzip.open(path, "wt", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        else:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

    # ---- cache control (optional) ----
    @classmethod
    def invalidate(cls, path: Optional[str] = None) -> None:
        with cls._lock:
            if path:
                cls._cache.pop(os.path.abspath(path), None)
            else:
                cls._cache.clear()


# =============================================================================
# EmotionIntensityAnalyzer
# =============================================================================
class EmotionIntensityAnalyzer:
    _PRIMARY_REQUIRED = {"희", "노", "애", "락"}
    _CONTEXT_FIELDS = {"context_patterns", "linguistic_patterns", "emotion_transitions", "sentiment_analysis", "ml_training_metadata"}
    _SUB_META_FIELDS = {"emotion_id", "primary_category", "sub_category", "emotion_complexity", "version"}
    _PROFILE_FIELDS = {"intensity_levels", "related_emotions", "core_keywords"}
    _INTENSITY_LEVELS_REQUIRED = {"low", "medium", "high", "intensity_examples"}
    _EMO_PATH_CACHE: Dict[str, str] = {}
    _EMO_PATH_CACHE_LOCK = RLock()

    def _resolve_emotions_json_path(self, path: str) -> str:
        """
        EMOTIONS.JSON 실제 경로를 견고하게 탐색하여 절대경로로 반환.
        우선순위:
          1) 인자로 받은 경로(파일이면 즉시 반환, 디렉터리면 그 안을 탐색)
          2) __file__ 폴더 → 그 부모들(프로젝트 루트까지)
          3) 현재 작업 폴더(CWD) → 그 부모들
        각 디렉터리에서 다음을 검사:
          - 정확한 이름: EMOTIONS.JSON / Emotions.json / emotions.json
          - 패턴: EMOTIONS*.JSON, EMOTIONS*.json, emotions*.json
          - 하위 폴더: config/, configs/ 도 함께 검사
        """
        cache_key = str(path) if path else "__auto__"
        with self._EMO_PATH_CACHE_LOCK:
            cached = self._EMO_PATH_CACHE.get(cache_key)
        if cached:
            return cached

        import os, logging
        from pathlib import Path
        log = logging.getLogger("emotion")
        provided_name: Optional[str] = None
        dirs_to_search: List[Path] = []

        def _is_valid_file(p: Path) -> bool:
            try:
                return p.is_file() and p.stat().st_size > 0
            except Exception:
                return False

        def _cache_and_return(found: Path) -> str:
            resolved_path = str(found.resolve())
            with self._EMO_PATH_CACHE_LOCK:
                self._EMO_PATH_CACHE[cache_key] = resolved_path
            return resolved_path

        # 0) 인자로 온 값 우선 처리
        provided = Path(path) if path else None
        if provided:
            if provided.is_file():
                return _cache_and_return(provided)
            # 절대/상대 디렉터리면 그 안에서 검색
            if provided.is_dir():
                base_dir_hint = provided.resolve()
                dirs_to_search = [base_dir_hint]
            else:
                # 파일명만 주어진 경우를 대비해 이름을 기록
                provided_name = provided.name
                dirs_to_search = []
        else:
            provided_name = None
            dirs_to_search = []

        # 1) __file__ 기준 디렉터리와 부모들
        script_dir = Path(__file__).resolve().parent
        dirs_to_search += [script_dir, *list(script_dir.parents)]

        # 2) CWD와 부모들
        cwd = Path(os.getcwd()).resolve()
        dirs_to_search += [cwd, *list(cwd.parents)]

        # 중복 제거(순서 유지)
        uniq_dirs = []
        seen = set()
        for d in dirs_to_search:
            if d not in seen:
                seen.add(d)
                uniq_dirs.append(d)

        # 검사 대상 이름/패턴/하위폴더
        names = []
        if provided and provided.name and not provided.is_dir():
            names.append(provided.name)
        elif provided_name:
            names.append(provided_name)
        names += ["EMOTIONS.JSON", "Emotions.json", "emotions.json"]

        patterns = ["EMOTIONS*.JSON", "EMOTIONS*.json", "emotions*.json"]
        subfolders = ["", "config", "configs", "Config", "Configs"]

        tried = []
        for d in uniq_dirs:
            for sub in subfolders:
                base = d / sub if sub else d
                # 정확한 이름 우선
                for nm in names:
                    candidate = base / nm
                    tried.append(str(candidate))
                    if _is_valid_file(candidate):
                        log.info(f"[config] EMOTIONS.JSON 경로 확정: {candidate}")
                        return _cache_and_return(candidate)
                # 패턴 검색
                for pat in patterns:
                    for candidate in base.glob(pat):
                        tried.append(str(candidate))
                        if _is_valid_file(candidate):
                            log.info(f"[config] EMOTIONS.JSON 경로 확정: {candidate}")
                            return _cache_and_return(candidate)

        # 실패 시: 시도 경로 일부를 함께 안내
        summary = "\n - ".join(tried[:12])
        raise FileNotFoundError(
            "EMOTIONS.JSON 파일을 찾을 수 없습니다. 아래 경로들을 시도했습니다:\n"
            f" - {summary}\n"
            "해결 방법: (1) 파일을 프로젝트 경로에 배치하거나 (2) 환경변수 EMOTIONS_JSON_PATH를 정확한 파일 경로로 설정하십시오."
        )

    def _load_emotions_data(self, path: str) -> Dict[str, Any]:
        """
        - _resolve_emotions_json_path 로 경로를 먼저 확정
        - EMOTIONS_AUTOGEN=1/true/yes 일 때만 skeleton 생성 허용(기본 OFF)
        """
        import os, logging
        log = logging.getLogger("emotion")

        # 전역 캐시에서 먼저 시도
        try:
            from src.data_utils import get_global_emotions_data
            data = get_global_emotions_data()
            if data:
                log.info("[config] EMOTIONS 데이터 전역 캐시에서 로드 완료")
                return data
        except Exception:
            pass

        autogen = os.environ.get("EMOTIONS_AUTOGEN", "0").strip().lower() in ("1", "true", "yes")
        resolved = self._resolve_emotions_json_path(path)

        try:
            loader = EmotionDataLoader(resolved, autogen=autogen)  # type: ignore
        except TypeError:
            loader = EmotionDataLoader(resolved)

        data = loader.load()
        log.info(f"[config] EMOTIONS 데이터 로드 완료: {resolved}")
        return data

    def __init__(self, emotions_data_path: str, context_weight: float = 0.3):
        self._example_token_pool: Dict[str, Set[str]] = {}
        self.emotions_data = self._load_emotions_data(emotions_data_path)
        self.primary_emotions = set(self._PRIMARY_REQUIRED)
        # STRICT/TOLERANT 모드 노출(운영 안전성)
        self.validation_mode = "strict" if os.environ.get("EAM_VALIDATION_MODE", "warn").lower() == "strict" else "warn"
        self._validate_emotions_data_structure()
        self.modifiers = self._load_modifiers()
        fallback_flag = os.environ.get("ALLOW_FALLBACK_MODIFIERS", "0").strip().lower() in ("1", "true", "yes")
        self._modifier_config = {"allow_fallback_modifiers": fallback_flag}
        self.allow_fallback_min_score = 0.2
        self.auto_scoring = {
            "enabled": True,
            "balance_strength": 0.7,
            "norm_method": "log",
            "topk_detail": 3,
            "clip": 6.0,
        }
        self._V_cache: Tuple[Optional[Dict[str, int]], Optional[float]] = (None, None)

        self._embeddings_enabled = False
        self._device = "cuda" if (hasattr(torch, "cuda") and torch.cuda.is_available()) else "cpu"
        self.tokenizer = None
        self.model = None
        
        # AC 후보 축소 인덱스
        self._phr2ids = None
        self._aho = None
        self._allow_embeddings = os.environ.get("ENABLE_EMBEDDINGS", "0").strip().lower() in ("1", "true", "yes")
        # Embedding cache (전역 캐시 엔진 활용)
        self._use_embedding_cache = os.environ.get("USE_EMBEDDING_CACHE", "1").strip().lower() in ("1", "true", "yes", "on")
        self._context_cache_engine = None  # lazy init via _get_context_cache_engine
        self._context_cache_lock = RLock()
        self._context_cache_error = False
        self._max_context = max(1, _env_int("INTENSITY_MAX_CONTEXT", 8))
        self._max_sub = max(1, _env_int("INTENSITY_MAX_SUB", 5))
        self._tokenizer_max_length = max(64, _env_int("INTENSITY_MAX_LENGTH", 192))
        self._pad_to_multiple = max(0, _env_int("INTENSITY_PAD_MULTIPLE", 8))
        self._batch_size = max(1, _env_int("INTENSITY_BATCH_SIZE", 12))
        self._autocast_enabled = (self._device == "cuda") and _env_bool("INTENSITY_FORCE_FP16", True)
        self._default_tensor_dtype = torch.float16 if self._autocast_enabled else torch.float32
        generate_embed_flag = _env_bool("USE_INTENSITY_EMBED", True)
        self._generate_embeddings_enabled = self._allow_embeddings and generate_embed_flag
        self._prewarm_enabled = _env_bool("INTENSITY_PREWARM", False)
        self._prewarm_samples = max(1, _env_int("INTENSITY_PREWARM_SAMPLES", 6))
        self._raw_context_cache: Dict[str, torch.Tensor] = {}
        try:
            num_threads = _env_int("INTENSITY_TORCH_THREADS", 0)
            if num_threads > 0:
                torch.set_num_threads(num_threads)
        except Exception:
            pass
        
        # 성능 최적화: 지연 로딩 및 캐싱
        self._model_cache_key = None
        self._model_loaded = False
        if self._allow_embeddings:
            self._lazy_load_pretrained_model()

        transformer_config = {
            "weights": {
                "primary_emotions": ["희", "노", "애", "락"],
                "complexity_scores": {"basic": 1.0, "subtle": 1.2, "complex": 1.4},
                "category_multipliers": {"same": 1.1, "diff": 0.9},
                "intensity_diff_factor": 0.1,
            },
            "temporal": {
                "complexity_weights": {
                    "basic": {"trend": 1.0, "variation": 1.0, "cycle": 1.0},
                    "subtle": {"trend": 1.2, "variation": 1.3, "cycle": 1.2},
                    "complex": {"trend": 1.5, "variation": 1.6, "cycle": 1.4},
                }
            },
            "statistics": {
                "significance_threshold": 0.5,
                "pattern_thresholds": {"intensifying": 0.2, "diminishing": -0.2, "fluctuating": 0.3},
            },
            "update": {"related_emotion_multiplier": 1.2},
        }
        self.transformer = EmotionIntensityTransformer(self.emotions_data, config=transformer_config)

        self.context_weight = context_weight
        self.profile_scorer = EmotionProfileScorer(self._tokenize_and_normalize)
        self.situation_scorer = SituationScorer(self._tokenize_and_normalize)
        self.transition_scorer = TransitionScorer(self._tokenize_and_normalize)
        self._ensure_modifier_lexicon()
        try:
            negations = sorted(self._modifier_lexicon.get("negation", set()))
            if negations:
                self.transition_scorer.cfg["negations"] = list(negations)
        except Exception:
            pass

        builder = getattr(self, "_build_sub_phrase_index", None)
        if callable(builder):
            self._sub_phrase_index = builder()
        
        # Limits configuration
        self._limits = {
            "top_sub_per_step": 8,
            "min_sub_score": 0.25,
            "max_sub_pairs_per_step": 200,
        }

        # Robustness/Scaling/Smoothing configuration and caches
        self.robust = {
            "junk_guard": {
                "enabled": True,
                "min_tokens": 2,
                "entropy_th": 4.0,
                "lexicon_cov_th": 0.02,
                "non_alnum_ratio_th": 0.60,
            },
            "scaling": {"method": "logistic", "gain": 1.8, "clip_low": 0.0, "clip_high": 1.0},
            "smoothing": {"method": "ema", "alpha": 0.5, "window": 3},
        }
        
        self._global_lexicon = None  # lazy build via _get_global_lexicon()

        # 라벨 커버리지 기반 Junk-Guard 임계 자동 보정
        try:
            self._calibrate_junk_guard()
        except Exception:
            pass
        
        # [PATCH] QA counters and caches for stability auditing
        if not hasattr(self, "_qa"):
            self._qa = {}
        self._qa.setdefault("borderline_junk", 0)
        self._qa.setdefault("short_text_cap", 0)
        self._qa.setdefault("negation_damp", 0)
        self._qa.setdefault("dyn_gain", 0)
        # negation regex cache / global lexicon cache / QA exposure flag
        self._neg_re_cache = {}
        self._global_lexicon = None
        self._expose_qa = bool(int(os.environ.get("INTENSITY_EXPOSE_QA", "0")))

        if self._prewarm_enabled:
            try:
                self._prewarm_context_embeddings(sample_limit=self._prewarm_samples)
            except Exception as exc:
                logger.debug(f"[intensity_analyzer] prewarm skipped: {exc}", exc_info=True)
    
    def _lazy_load_pretrained_model(self) -> None:
        """지연 로딩으로 모델 로딩 최적화"""
        try:
            # config.py의 모델 캐시 시스템 활용 (src/config 또는 최상위 config 지원)
            get_cached_model = None
            try:
                from src.config import get_cached_model  # type: ignore[attr-defined]
            except ImportError:
                try:
                    import config as _config  # type: ignore
                    get_cached_model = getattr(_config, "get_cached_model", None)
                except ImportError:
                    get_cached_model = None

            model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            self._model_cache_key = f"intensity_analyzer:{model_name}"

            from transformers import AutoTokenizer, AutoModel
            if callable(get_cached_model):
                self.tokenizer = get_cached_model(model_name, AutoTokenizer)
                self.model = get_cached_model(model_name, AutoModel)
            else:
                # 캐시 유틸이 없으면 직접 로드
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModel.from_pretrained(model_name)

            self._embeddings_enabled = True
            self._model_loaded = True

            import logging
            log = logging.getLogger("emotion")
            log.info(f"[intensity_analyzer] 모델 로드 완료 (cache={'on' if callable(get_cached_model) else 'off'}): {model_name}")

        except Exception as e:
            import logging
            log = logging.getLogger("emotion")
            log.warning(f"[intensity_analyzer] 모델 로딩 실패, 임베딩 비활성화: {e}")
            self._embeddings_enabled = False
            self._model_loaded = False
    
    def _ensure_model_loaded(self) -> bool:
        """모델이 로드되었는지 확인하고 필요시 로드"""
        if not self._allow_embeddings:
            return False
        
        if not self._model_loaded:
            self._lazy_load_pretrained_model()
        
        # Limits configuration
        self._limits = {
            "top_sub_per_step": 8,
            "min_sub_score": 0.25,
            "max_sub_pairs_per_step": 200,
        }

        # Robustness/Scaling/Smoothing configuration and caches
        self.robust = {
            "junk_guard": {
                "enabled": True,
                "min_tokens": 2,
                "entropy_th": 4.0,
                "lexicon_cov_th": 0.02,
                "non_alnum_ratio_th": 0.60,
            },
            "scaling": {"method": "logistic", "gain": 1.8, "clip_low": 0.0, "clip_high": 1.0},
            "smoothing": {"method": "ema", "alpha": 0.5, "window": 3},
        }
        
        self._global_lexicon = None  # lazy build via _get_global_lexicon()

        # 라벨 커버리지 기반 Junk-Guard 임계 자동 보정
        try:
            self._calibrate_junk_guard()
        except Exception:
            pass
        
        # [PATCH] QA counters and caches for stability auditing
        if not hasattr(self, "_qa"):
            self._qa = {}
        self._qa.setdefault("borderline_junk", 0)
        self._qa.setdefault("short_text_cap", 0)
        self._qa.setdefault("negation_damp", 0)
        self._qa.setdefault("dyn_gain", 0)
        # negation regex cache / global lexicon cache / QA exposure flag
        self._neg_re_cache = {}
        self._global_lexicon = None
        self._expose_qa = bool(int(os.environ.get("INTENSITY_EXPOSE_QA", "0")))

        logger.info("EmotionIntensityAnalyzer 초기화 완료")
        return self._model_loaded

    def _resolve_limit(self, value: Optional[int], default: int, minimum: int = 1) -> int:
        if value is None:
            return max(minimum, default)
        try:
            return max(minimum, int(value))
        except Exception:
            return max(minimum, default)

    def _autocast_ctx(self):
        if self._autocast_enabled and hasattr(torch, "cuda") and hasattr(torch.cuda, "amp"):
            try:
                return torch.cuda.amp.autocast()
            except Exception:
                return nullcontext()
        return nullcontext()

    def _bulk_compute_embeddings(self, contexts: List[str]) -> None:
        if not self._generate_embeddings_enabled or not contexts:
            return
        contexts = [c.strip() for c in contexts if isinstance(c, str) and c.strip()]
        contexts = [c for c in contexts if c not in self._raw_context_cache]
        if not contexts:
            return
        if not self._ensure_model_loaded():
            return
        if not self._embeddings_enabled or self.tokenizer is None or self.model is None:
            return

        pad_multiple = self._pad_to_multiple if self._pad_to_multiple > 0 else None
        for start in range(0, len(contexts), self._batch_size):
            batch = contexts[start : start + self._batch_size]
            if not batch:
                continue
            try:
                tok_kwargs = {
                    "return_tensors": "pt",
                    "padding": True,
                    "truncation": True,
                    "max_length": self._tokenizer_max_length,
                }
                if pad_multiple:
                    tok_kwargs["pad_to_multiple_of"] = pad_multiple
                toks = self.tokenizer(batch, **tok_kwargs)
                toks = {k: v.to(self._device) for k, v in toks.items()}
                with torch.no_grad():
                    with self._autocast_ctx():
                        outputs = self.model(**toks).last_hidden_state
                pooled = torch.mean(outputs, dim=1)
                for ctx, vec in zip(batch, pooled):
                    self._raw_context_cache[ctx] = vec.detach().to(self._device).to(self._default_tensor_dtype)
            except Exception as exc:
                logger.debug(f"[intensity_analyzer] batch embedding fallback: {exc}", exc_info=True)
                for ctx in batch:
                    try:
                        vec_np = self._compute_context_embedding_np(ctx)
                        tensor = torch.from_numpy(vec_np).to(self._device)
                        if tensor.dtype != self._default_tensor_dtype:
                            tensor = tensor.to(self._default_tensor_dtype)
                        self._raw_context_cache[ctx] = tensor
                    except Exception:
                        self._raw_context_cache.setdefault(
                            ctx,
                            torch.zeros(
                                getattr(getattr(self.model, "config", None), "hidden_size", 768),
                                dtype=self._default_tensor_dtype,
                                device=self._device,
                            ),
                        )

    def _prime_context_embeddings(self, contexts: Iterable[str]) -> None:
        if not self._generate_embeddings_enabled or not contexts:
            return
        normalized: List[str] = []
        seen = set()
        for ctx in contexts:
            if not isinstance(ctx, str):
                continue
            norm = ctx.strip()
            if not norm or norm in seen or norm in self._raw_context_cache:
                continue
            seen.add(norm)
            normalized.append(norm)
        if not normalized:
            return

        cache_engine = self._get_context_cache_engine() if self._use_embedding_cache else None
        pending_local: List[str] = []
        if cache_engine is not None:
            for ctx in normalized:
                try:
                    vec_np = cache_engine.get_embedding(ctx, compute_func=self._compute_context_embedding_np)
                    tensor = torch.from_numpy(vec_np).to(self._device)
                    if tensor.dtype != self._default_tensor_dtype:
                        tensor = tensor.to(self._default_tensor_dtype)
                    self._raw_context_cache[ctx] = tensor
                except Exception:
                    pending_local.append(ctx)
        else:
            pending_local = normalized

        if pending_local:
            self._bulk_compute_embeddings(pending_local)

    def _prewarm_context_embeddings(self, *, sample_limit: int = 6) -> None:
        if not self._generate_embeddings_enabled or not self._allow_embeddings:
            return
        if not self._ensure_model_loaded():
            return
        samples: List[str] = []
        primary_order = ["희", "노", "애", "락"]
        for primary in primary_order:
            data = self.emotions_data.get(primary, {}) if isinstance(self.emotions_data, dict) else {}
            try:
                ctx = self._create_primary_emotion_context(data)
                if ctx:
                    samples.append(ctx)
            except Exception:
                pass
            sub_map = (data.get("sub_emotions") or {}) if isinstance(data, dict) else {}
            if isinstance(sub_map, dict):
                for _, sub_data in list(sub_map.items())[:2]:
                    try:
                        ctx = self._create_sub_emotion_context(sub_data)
                        if ctx:
                            samples.append(ctx)
                    except Exception:
                        continue
            if len(samples) >= sample_limit:
                break
        samples = [s for s in samples if isinstance(s, str) and s.strip()]
        if not samples:
            return
        self._prime_context_embeddings(samples[:sample_limit])

    def warmup(self):
        """
        GPU 워밍업 - 첫 호출 지연을 줄이기 위한 1회 실행
        타임아웃 가능성을 줄이고 첫 호출을 1~2초 빨라지게 함
        """
        try:
            # 아주 짧은 텍스트로 전체 파이프라인을 1회 실행
            _ = self.analyze_emotion_intensity("워밍업 한 줄")
            logger.info("EmotionIntensityAnalyzer warmup 완료")
        except Exception as e:
            # 워밍업 실패는 무시 (실제 실행에 영향 없음)
            logger.debug(f"EmotionIntensityAnalyzer warmup 중 예외 (무시): {e}")
            pass

    @lru_cache(maxsize=4096)
    def _tok_cache(self, text: str) -> Tuple[str, ...]:
        tl = re.sub(r"[\uFE0E\uFE0F]", "", (text or "").lower())
        return tuple(re.findall(r"[가-힣A-Za-z0-9]+", tl))

    def _tokenize_and_normalize(self, text: str) -> List[str]:
        tokens = list(self._tok_cache(text))
        out = []
        for t in tokens:
            if t.endswith("다") and len(t) > 2:
                t = t[:-1]
            elif t.endswith("요") and len(t) > 2:
                t = t[:-1]
            out.append(t)
        return out

    # ──────────────────────────────────────────────────────────────────────
    # Robustness & Junk-Guard utilities
    # ──────────────────────────────────────────────────────────────────────
    def _build_global_lexicon(self) -> Set[str]:
        """EMOTIONS.json에서 긍/부/키워드/예시/패턴 등 전역 토큰 사전을 하나로 모읍니다."""
        if self._global_lexicon:
            return self._global_lexicon
        lx: Set[str] = set()

        def _tok(x: Any) -> Set[str]:
            return set(self._tokenize_and_normalize(str(x))) if isinstance(x, str) else set()

        data = (self.emotions_data or {})
        if isinstance(data, dict):
            for e in data.values():
                if not isinstance(e, dict):
                    continue
                # sentiment analysis indicators
                s = (e.get("sentiment_analysis") or {})
                for bucket in ("positive_indicators", "negative_indicators"):
                    for w in (s.get(bucket) or []):
                        lx |= _tok(w)

                # sub emotions
                subs = (e.get("sub_emotions") or {})
                if isinstance(subs, dict):
                    for sub in subs.values():
                        if not isinstance(sub, dict):
                            continue
                        prof = (sub.get("emotion_profile") or sub.get("profile") or {})
                        for w in (prof.get("core_keywords") or []):
                            lx |= _tok(w)
                        # intensity examples
                        il = (prof.get("intensity_levels") or {})
                        ex_map = (il.get("intensity_examples") or {}) if isinstance(il, dict) else {}
                        if isinstance(ex_map, dict):
                            for exs in ex_map.values():
                                for ex in (exs or []):
                                    lx |= _tok(ex)
                        # key phrases patterns
                        ling = (sub.get("linguistic_patterns") or {})
                        for kp in (ling.get("key_phrases") or []):
                            if isinstance(kp, dict):
                                lx |= _tok(kp.get("pattern", ""))

        self._global_lexicon = lx
        return lx

    # [PATCH] 글로벌 렉시콘 1회 캐시 접근자
    def _get_global_lexicon(self):
        """글로벌 렉시콘을 1회만 빌드해 캐시합니다."""
        gl = getattr(self, "_global_lexicon", None)
        if gl is None:
            gl = self._build_global_lexicon()
            self._global_lexicon = gl
        return gl

    def _shannon_entropy(self, s: str) -> float:
        if not s:
            return 0.0
        from math import log2
        freq: Dict[str, int] = {}
        for ch in s:
            freq[ch] = freq.get(ch, 0) + 1
        n = float(len(s))
        return float(-sum((c / n) * log2(c / n) for c in freq.values()))

    def _calibrate_junk_guard(self) -> None:
        """전역 사전 크기에 따라 lexicon_cov_th를 소폭 조정합니다."""
        try:
            lx_size = len(self._get_global_lexicon())
        except Exception:
            lx_size = 0
        if lx_size < 500:
            self.robust["junk_guard"]["lexicon_cov_th"] = 0.0  # 스파스 환경 완화
        elif lx_size > 5000:
            self.robust["junk_guard"]["lexicon_cov_th"] = 0.03

    def _max_run(self, s: str) -> int:
        if not s:
            return 0
        m, cur = 1, 1
        for i in range(1, len(s)):
            if s[i] == s[i - 1]:
                cur += 1
                m = max(m, cur)
            else:
                cur = 1
        return m

    def _input_stats(self, text: str) -> Dict[str, float]:
        t = (text or "")
        tokens = tuple(self._tok_cache(t))
        n = len(t)
        tn = len(tokens)
        alnum = sum(ch.isalnum() for ch in t)
        non_alnum_ratio = 0.0 if n == 0 else (n - alnum) / n
        han = sum("가" <= ch <= "힣" for ch in t)
        entropy = self._shannon_entropy(t)
        run = self._max_run(t)
        lex = self._get_global_lexicon()
        cov = 0.0
        if tn:
            hits = sum(1 for w in tokens if w in lex)
            cov = hits / tn
        return {
            "len": float(n),
            "tokens": float(tn),
            "non_alnum_ratio": float(non_alnum_ratio),
            "han_ratio": float((han / n) if n else 0.0),
            "entropy": float(entropy),
            "max_run": float(run),
            "lex_cov": float(cov),
        }

    def _junk_guard(self, text: str) -> Tuple[bool, float, Dict[str, float]]:
        """잡음/난수 입력 여부와 점수 반환: (is_junk, score, stats)"""
        try:
            if not self.robust.get("junk_guard", {}).get("enabled", True):
                return (False, 0.0, {})
            st = self._input_stats(text)
            jcfg = self.robust["junk_guard"]
            conds = [
                (st["tokens"] < float(jcfg.get("min_tokens", 2))),
                (st["entropy"] >= float(jcfg.get("entropy_th", 4.0))),
                (st["lex_cov"] < float(jcfg.get("lexicon_cov_th", 0.02))),
                (st["non_alnum_ratio"] >= float(jcfg.get("non_alnum_ratio_th", 0.60))),
                (st["max_run"] >= 8.0),
            ]
            score = sum(1.0 if c else 0.0 for c in conds) / float(len(conds))
            return (score >= 0.6, float(score), st)
        except Exception:
            return (False, 0.0, {})

    def _minimal_output(self, confidence: float, flags: List[str]) -> Dict[str, Dict[str, Any]]:
        """항상 구조를 보장: 4대 정서에 0점, level=low."""
        out: Dict[str, Dict[str, Any]] = {}
        for eid in self.primary_emotions:
            out[eid] = {
                "intensity_score": 0.0,
                "modified_score": 0.0,
                "level": "low",
                "confidence": float(confidence),
                "flags": list(flags),
            }
        return out

    # [PATCH] Borderline Junk mitigation helper
    def _borderline_junk_mitigate(self, jscore: float, base_scores: Dict[str, float]) -> Dict[str, float]:
        """
        Junk-Guard 점수가 0.4~0.6인 회색지대에서 강도를 부드럽게 낮춘다.
        완전 차단 대신 scale∈[0.65..1.0] 적용.
        """
        if 0.4 <= float(jscore) < 0.6 and isinstance(base_scores, dict):
            # 0.4→1.0, 0.6→0.65 로 선형 감쇠
            s = 1.0 - 0.35 * ((float(jscore) - 0.4) / 0.2)
            self._qa["borderline_junk"] += 1
            return {k: float(v) * float(max(0.65, min(1.0, s))) for k, v in base_scores.items()}
        return base_scores

    # [PATCH] 부정표지 경계 정규식 캐시 컴파일러
    def _compile_negation_regex(self, neg_set):
        """부정표지 집합으로 경계 정규식을 컴파일(캐시)."""
        if not neg_set:
            neg_set = {"안", "않", "아니", "못"}
        key = tuple(sorted(neg_set))
        pat = self._neg_re_cache.get(key)
        if pat is None:
            escaped = [re.escape(x) for x in sorted(neg_set, key=len, reverse=True)]
            boundary = r"(?<![가-힣A-Za-z0-9])(?:%s)(?![가-힣A-Za-z0-9])" % "|".join(escaped)
            pat = re.compile(boundary)
            self._neg_re_cache[key] = pat
        return pat

    # ──────────────────────────────────────────────────────────────────────
    # Scaling & surface boost
    # ──────────────────────────────────────────────────────────────────────
    def _compress_intensity(self, x: float) -> float:
        cfg = self.robust.get("scaling", {})
        method = str(cfg.get("method", "logistic")).lower()
        lo = float(cfg.get("clip_low", 0.0)); hi = float(cfg.get("clip_high", 1.0))
        if method == "logistic":
            base_g = float(cfg.get("gain", 1.8))
            # 토큰 길이에 따른 동적 게인(짧으면 더 눌림)
            try:
                tl = max(1, int(getattr(self, "_last_token_len", 8)))
            except Exception:
                tl = 8
            dyn = 1.0 - math.exp(-tl / 12.0)  # 0~1
            # 신뢰도 보정(약신호면 게인 축소)
            base_conf = float(getattr(self, "_last_base_conf", 0.0))
            conf_scale = 0.85 if base_conf < 0.3 else 1.0
            g = base_g * (0.6 + 0.4 * dyn) * conf_scale
            self._qa["dyn_gain"] += 1

            x = max(0.0, min(1.0, float(x)))
            try:
                y = 1.0 / (1.0 + math.exp(-g * (x - 0.5)))
            except OverflowError:
                y = 0.0 if g * (x - 0.5) < 0 else 1.0
            return float(max(lo, min(hi, y)))
        elif method == "tanh":
            g = float(cfg.get("gain", 1.0))
            x = max(0.0, min(1.0, float(x)))
            y = 0.5 * (math.tanh(g * (x - 0.5)) + 1.0)
            return float(max(lo, min(hi, y)))
        return float(max(lo, min(hi, float(x))))

    def _surface_boost(self, text: str) -> float:
        """구두점/반복/이모지 등 표면 신호 가중."""
        t = (text or "")
        boost = 1.0
        excl = t.count("!")
        ques = t.count("?")
        elli = t.count("…") + t.count("...")
        laugh = len(re.findall(r"(ㅋㅋ+|ㅎㅎ+)", t))
        cry = len(re.findall(r"(ㅠㅠ+|ㅜㅜ+)", t))
        boost *= (1.0 + min(0.15, 0.02 * excl))
        boost *= (1.0 + min(0.10, 0.02 * ques))
        boost *= (1.0 + min(0.08, 0.02 * elli))
        boost *= (1.0 + min(0.12, 0.05 * laugh))
        boost *= (1.0 + min(0.12, 0.05 * cry))
        return float(min(1.5, boost))

    # ──────────────────────────────────────────────────────────────────────
    # Sequence smoothing
    # ──────────────────────────────────────────────────────────────────────
    def _apply_sequence_smoothing(self, seq: List[Dict[str, float]]) -> List[Dict[str, float]]:
        cfg = self.robust.get("smoothing", {})
        method = str(cfg.get("method", "ema")).lower()
        if not isinstance(seq, list) or len(seq) < 2:
            return seq
        keys = sorted({k for row in seq for k in row.keys()})
        mat = np.array([[float(row.get(k, 0.0) or 0.0) for k in keys] for row in seq], dtype=np.float32)

        if method == "ema":
            alpha = float(cfg.get("alpha", 0.5))
            out = np.zeros_like(mat)
            out[0] = mat[0]
            for i in range(1, mat.shape[0]):
                out[i] = alpha * mat[i] + (1.0 - alpha) * out[i - 1]
        else:
            win = int(cfg.get("window", 3))
            out = self._moving_average(mat, max(2, win))

        smoothed: List[Dict[str, float]] = []
        for i in range(out.shape[0]):
            smoothed.append({keys[j]: float(out[i, j]) for j in range(out.shape[1])})
        return smoothed

    @staticmethod
    def _moving_average(mat: np.ndarray, win: int) -> np.ndarray:
        if win <= 1:
            return mat
        n, d = mat.shape
        half = win // 2
        indices = np.arange(n)
        starts = np.maximum(0, indices - half)
        ends = np.minimum(n, indices + half + 1)
        cumsum = np.vstack([np.zeros((1, d), dtype=mat.dtype), np.cumsum(mat, axis=0)])
        sums = cumsum[ends] - cumsum[starts]
        lengths = (ends - starts).reshape(-1, 1).astype(mat.dtype, copy=False)
        lengths[lengths == 0] = 1
        out = sums / lengths
        return out

    @staticmethod
    def _data_path_exists(data: Any, path: str) -> bool:
        node = data
        parts = path.split(".")
        for i, part in enumerate(parts):
            if node is None:
                return False
            if part == "*":
                rest = ".".join(parts[i + 1 :])
                if isinstance(node, dict):
                    return any(EmotionIntensityAnalyzer._data_path_exists(v, rest) for v in node.values())
                if isinstance(node, list):
                    return any(EmotionIntensityAnalyzer._data_path_exists(v, rest) for v in node)
                return False
            if isinstance(node, dict) and part in node:
                node = node[part]
            elif isinstance(node, list):
                try:
                    idx = int(part)
                except ValueError:
                    return False
                if 0 <= idx < len(node):
                    node = node[idx]
                else:
                    return False
            else:
                return False
        return True

    def _check_required_fields(self, data: Dict[str, Any], required_fields: Set[str], error_context: str) -> None:
        if not isinstance(data, dict):
            raise TypeError(f"{error_context}: dict 필요")
        missing = [f for f in required_fields if not self._data_path_exists(data, f)]
        if missing:
            raise ValueError(f"{error_context}: 필수 경로 누락 - {', '.join(sorted(missing))}")

    def _validate_primary_emotions_presence(self) -> None:
        if not self._PRIMARY_REQUIRED.issubset(self.emotions_data.keys()):
            missing = self._PRIMARY_REQUIRED - set(self.emotions_data.keys())
            raise ValueError(f"필수 대표감정 누락: {', '.join(sorted(missing))}")

    def _handle_missing(self, is_strict: bool, message: str, data: Dict, key: str, default_value: Any):
        if is_strict:
            raise ValueError(message)
        logger.warning(message + " → 기본값으로 보완")
        data.setdefault(key, default_value)

    def _ensure_sub_emotion_fields(self, sub_name: str, sub_data: Dict, parent_id: str, is_strict: bool):
        context = f"세부감정 '{parent_id}/{sub_name}'"
        all_required_fields = {
            "metadata": self._SUB_META_FIELDS,
            "emotion_profile": self._PROFILE_FIELDS,
            "context_patterns": set(),
            "linguistic_patterns": set(),
            "emotion_transitions": set(),
        }
        for field, required_keys in all_required_fields.items():
            if field not in sub_data:
                self._handle_missing(is_strict, f"{context} 필수 필드 누락: {field}", sub_data, field, {})
            nested = sub_data.get(field, {})
            if required_keys:
                missing = required_keys - nested.keys()
                if missing:
                    for k in missing:
                        default = [] if k == "core_keywords" else {}
                        self._handle_missing(is_strict, f"{context}.{field} 내부 키 누락: {k}", nested, k, default)

    def _validate_emotions_data_structure(self):
        try:
            is_strict = getattr(self, "validation_mode", "strict") == "strict"
            missing_primary = self._PRIMARY_REQUIRED - self.emotions_data.keys()
            for mk in missing_primary:
                self._handle_missing(is_strict, f"필수 대표감정 누락: {mk}", self.emotions_data, mk, {})
            total_sub = 0
            for eid in self._PRIMARY_REQUIRED:
                edata = self.emotions_data.setdefault(eid, {})
                sub = edata.setdefault("sub_emotions", {})
                for sname, sdata in sub.items():
                    self._ensure_sub_emotion_fields(sname, sdata, eid, is_strict)
                total_sub += len(sub)
            if total_sub != 120:
                self._handle_missing(is_strict, f"전체 세부감정 수 120 불일치 (현재 {total_sub})", {}, "dummy", {})
            logger.info(f"라벨링뼈대 검증 완료: 대표={len(self._PRIMARY_REQUIRED)}, 세부={total_sub}, 모드={'strict' if is_strict else 'warn'}")
        except (ValueError, TypeError) as e:
            logger.error(f"라벨링 검증 오류: {e}", exc_info=True)
            raise

    def _load_modifiers(self) -> Dict[str, Dict[str, float]]:
        try:
            default_amplifiers = {"극도로": 1.3, "매우": 1.2, "아주": 1.15, "정말": 1.1}
            default_diminishers = {"약간": 0.9, "조금": 0.8}
            json_amplifiers, json_diminishers = set(), set()
            for edata in self.emotions_data.values():
                for sdata in edata.get("sub_emotions", {}).values():
                    mods = sdata.get("linguistic_patterns", {}).get("sentiment_modifiers", {})
                    amps = mods.get("amplifiers", [])
                    dims = mods.get("diminishers", [])
                    if isinstance(amps, list):
                        json_amplifiers.update(w.lower() for w in amps)
                    if isinstance(dims, list):
                        json_diminishers.update(w.lower() for w in dims)
            out = {"amplifiers": {}, "diminishers": {}}
            for w in json_amplifiers:
                out["amplifiers"][w] = default_amplifiers.get(w, 1.2)
            for w, v in default_amplifiers.items():
                out["amplifiers"].setdefault(w, v)
            for w in json_diminishers:
                out["diminishers"][w] = default_diminishers.get(w, 0.8)
            for w, v in default_diminishers.items():
                out["diminishers"].setdefault(w, v)
            logger.info("수정어 로드 완료")
            return out
        except Exception as e:
            logger.error(f"수정어 로드 실패: {e}", exc_info=True)
            return {"amplifiers": {}, "diminishers": {}}

    def _mod_table(self, mt: Optional[str]) -> Dict[str, float]:
        """
        수정어 타입(mt) → modifiers 테이블을 안전하게 반환.
        JSON 키가 프로젝트마다 조금씩 달라도 별칭을 순회하며 찾아줍니다.
        예: negation → ['negations','negation'], diminisher → ['diminishers','downtoners'] 등
        """
        if not isinstance(getattr(self, "modifiers", None), dict):
            return {}
        alias_map: Dict[str, Tuple[str, ...]] = {
            "amplifier": ("amplifiers", "intensifiers"),
            "diminisher": ("diminishers", "downtoners"),
            "negation": ("negations", "negation"),
            "reverser": ("reversers", "contrastive_markers", "contrast_markers"),
        }
        keys = alias_map.get(mt or "", ())
        for k in keys:
            tbl = self.modifiers.get(k)
            if isinstance(tbl, dict):
                return tbl
        return {}

    def _load_pretrained_model(self):
        try:
            self._device = "cuda" if (hasattr(torch, "cuda") and torch.cuda.is_available()) else "cpu"
            model_name = "klue/bert-base"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.to(self._device)
            self.model.eval()
            self._embeddings_enabled = True
            logger.info(f"모델 로드 완료: {model_name} on {self._device}")
        except Exception as e:
            self._embeddings_enabled = False
            logger.warning(f"임베딩 비활성화(모델 로드 실패): {e}")

    def _is_positive_sentence(self, text: str) -> bool:
        try:
            tl = (text or "").lower()
            pos, neg = 0.0, 0.0

            # EMOTIONS.JSON 상단의 sentiment 사인만 사용
            p_all, n_all, combos = [], [], []
            for e in self.emotions_data.values():
                s = e.get("sentiment_analysis", {}) or {}
                p_all.extend(s.get("positive_indicators", []) or [])
                n_all.extend(s.get("negative_indicators", []) or [])
                combos.extend(s.get("sentiment_combinations", []) or [])

            for kw in p_all:
                if kw and str(kw).lower() in tl:
                    pos += 1.0
            for kw in n_all:
                if kw and str(kw).lower() in tl:
                    neg += 1.0

            for c in combos:
                try:
                    words = [str(w).lower() for w in (c.get("words") or [])]
                    w = float(c.get("weight", 0.0))
                except Exception:
                    continue
                if words and all(wd in tl for wd in words):
                    if w >= 0:
                        pos += abs(w)
                    else:
                        neg += abs(w)

            return pos >= max(neg, 1e-6)
        except Exception:
            # 모호 시 긍/부 중립 판단 → 기본 True 유지 (기존과 호환)
            return True

    def _get_intensity_weight(self, intensity: str, emotion_id: str) -> float:
        try:
            int_levels = self.emotions_data.get(emotion_id, {}).get("emotion_profile", {}).get("intensity_levels", {})
            info = int_levels.get(intensity)
            if isinstance(info, dict):
                return float(info.get("weight", 0.5))
            return {"high": 0.8, "medium": 0.5, "low": 0.3}.get(str(intensity).lower(), 0.5)
        except Exception:
            return 0.5

    def _calibrate_intensity_score(self, emotion_id: str, raw: float) -> float:
        """
        EMOTIONS.JSON의 ml_training_metadata.calibration 설정을 사용해 점수를 보정합니다.
        지원 형태:
          - {"type": "platt", "a": 1.0, "b": 0.0}  # 로지스틱(Platt)
          - {"type": "logistic", "a": 1.0, "b": 0.0}
          - {"type": "isotonic", "table": [[x0,y0],[x1,y1],...]}  # 구간 선형 보간
          - {"희": {...}, "노": {...}}  # 감정별 개별 설정도 허용
        미설정/오류 시 보정 없이 raw 반환.
        """
        try:
            meta_g = (self.emotions_data.get("ml_training_metadata") or {})
            meta_e = ((self.emotions_data.get(emotion_id, {}) or {}).get("ml_training_metadata") or {})
            cal = (meta_e.get("calibration") or {}) or (meta_g.get("calibration") or {})

            # 감정별 개별 설정 우선, 없으면 전체 공통 설정 사용
            cfg = cal.get(emotion_id) if isinstance(cal.get(emotion_id), dict) else cal
            if not isinstance(cfg, dict):
                return float(raw)

            ctype = str(cfg.get("type", "none")).lower()
            if ctype in ("platt", "logistic"):
                a = float(cfg.get("a", 1.0))
                b = float(cfg.get("b", 0.0))
                z = a * float(raw) + b
                # 안전한 시그모이드
                try:
                    val = 1.0 / (1.0 + math.exp(-z))
                except OverflowError:
                    val = 0.0 if z < 0 else 1.0
                return float(val)

            elif ctype in ("isotonic", "piecewise"):
                table = cfg.get("table") or []
                if not isinstance(table, list) or not table:
                    return float(raw)
                pairs = []
                for p in table:
                    if isinstance(p, (list, tuple)) and len(p) >= 2:
                        try:
                            pairs.append((float(p[0]), float(p[1])))
                        except Exception:
                            continue
                if not pairs:
                    return float(raw)
                pairs.sort(key=lambda x: x[0])
                x = max(min(float(raw), pairs[-1][0]), pairs[0][0])
                for i in range(1, len(pairs)):
                    x0, y0 = pairs[i - 1]
                    x1, y1 = pairs[i]
                    if x <= x1:
                        t = (x - x0) / (x1 - x0) if x1 > x0 else 0.0
                        return float(y0 + t * (y1 - y0))
                return float(pairs[-1][1])

            # 유형 미지정 = 패스
            return float(raw)
        except Exception as e:
            logger.warning(f"[calibration] 실패: {e}")
            return max(0.0, min(1.0, float(raw)))

    def _calculate_base_intensity(
        self,
        text: str,
        emotion_data: Dict[str, Any],
        *,
        tokens: Optional[List[str]] = None,
        text_lower: Optional[str] = None,
        is_positive: Optional[bool] = None,
        primary_id: str = "",
    ) -> Tuple[float, Dict[str, List[str]]]:
        try:
            primary = primary_id or str(emotion_data.get("metadata", {}).get("primary_category", "") or "")
            subemotions = emotion_data.get("sub_emotions", {}) or {}
            if not isinstance(subemotions, dict) or not subemotions:
                return 0.0, {}

            toks = tokens if tokens is not None else self._tokenize_and_normalize(text or "")
            tl = text_lower if text_lower is not None else (text or "").lower()
            is_pos = bool(is_positive) if is_positive is not None else self._is_positive_sentence(text)

            matched = 0
            total = 0.0
            detail_hits: Dict[str, List[str]] = {}
            token_set = set(toks)
            primary_key = primary or primary_id or ""
            candidate_topk = int(os.environ.get("INT_TOPK_SUB", "12"))
            cand_ids = self._candidate_subs(primary_key, token_set, topk=candidate_topk) if hasattr(self, "_candidate_subs") else []
            if not cand_ids:
                cand_ids = [str(k) for k in subemotions.keys()]

            for sub_id in cand_ids:
                actual_key, sub_info = self._resolve_sub_entry(subemotions, sub_id)
                if sub_info is None:
                    continue
                ps, hits = self.profile_scorer.score(sub_info, toks, is_pos)
                ss = self.situation_scorer.score(sub_info, toks)
                ts = self.transition_scorer.score(sub_info, toks, tl)
                sc = float(ps) + float(ss) + float(ts)
                if sc > 0:
                    matched += 1
                    total += sc
                    if hits:
                        key = f"{primary}/{actual_key}" if primary else f"{actual_key}"
                        detail_hits[str(key)] = list(dict.fromkeys(hits))

            if matched == 0:
                base = float(self.allow_fallback_min_score)
            else:
                avg = total / matched
                cpx = {"basic": 1.0, "subtle": 1.2, "complex": 1.4}.get(
                    str(emotion_data.get("metadata", {}).get("emotion_complexity", "basic")),
                    1.0,
                )
                ln = len(toks)
                len_mul = 0.8 if ln < 5 else (1.1 if ln > 30 else 1.0)
                sent_mul = 1.2 if is_pos else 0.9
                base = float(avg) * float(cpx) * float(len_mul) * float(sent_mul)

            try:
                pos_ct, neg_ct = self._estimate_polarity_strength(text)
                if pos_ct > 0 and neg_ct > 0 and base > 0.6:
                    base *= 0.85
            except Exception:
                pass

            meta = (self.emotions_data.get("ml_training_metadata") or {})
            k = float(meta.get("intensity_normalizer_k", 1.2))
            base_norm = float(base) / float(base + k) if (base + k) > 0 else 0.0

            calibrated = self._calibrate_intensity_score(
                str((primary_id or emotion_data.get("metadata", {}).get("primary_category", "")) or ""),
                float(base_norm),
            )

            caps_meta = (meta.get("intensity_caps") or {})
            try:
                cap = float(caps_meta.get(str(primary_id), 1.0))
            except Exception:
                cap = 1.0

            final_score = min(max(0.0, float(calibrated)), cap)
            return final_score, detail_hits
        except Exception as e:
            logger.error(f"[기본 강도 계산] 오류: {e}", exc_info=True)
            return 0.0, {}


    def _estimate_polarity_strength(self, text: str) -> Tuple[int, int]:
        """
        EMOTIONS.JSON 상단 sentiment 인디케이터를 사용해 간단히 긍/부 신호 개수를 추정.
        """
        tl = (text or "").lower()
        pos, neg = 0, 0
        for e in self.emotions_data.values():
            s = e.get("sentiment_analysis", {}) or {}
            for kw in (s.get("positive_indicators") or []):
                if kw and str(kw).lower() in tl:
                    pos += 1
            for kw in (s.get("negative_indicators") or []):
                if kw and str(kw).lower() in tl:
                    neg += 1
        return pos, neg

    def _calculate_situational_multiplier(self, token: str, emotion_data: Dict[str, Any], emotion_complexity: str, modifier_type: str) -> float:
        out = 1.0
        for s in emotion_data.get("sub_emotions", {}).values():
            for sit in s.get("context_patterns", {}).get("situations", {}).values():
                if token in (sit.get("keywords", []) or []):
                    lvl = sit.get("intensity", "medium")
                    w = self._calculate_weighted_intensity(emotion_complexity, lvl, modifier_type)
                    out = max(out, w)
        return out

    def _calculate_linguistic_multiplier(self, token: str, text: str, emotion_data: Dict[str, Any]) -> float:
        m = 1.0
        lp = emotion_data.get("linguistic_patterns", {}) or {}
        for kp in lp.get("key_phrases", []) or []:
            pat = (kp.get("pattern", "") or "").lower()
            ctx = (kp.get("context_requirement", "") or "").lower()
            if pat and pat in token and (not ctx or ctx in text):
                m *= (1.0 + float(kp.get("weight", 0.0)))
        for combo in lp.get("sentiment_combinations", []) or []:
            words = [w.lower() for w in combo.get("words", [])]
            if words and all(w in text for w in words):
                m *= (1.0 + float(combo.get("weight", 0.0)))
        return m

    def _calculate_profile_multiplier(self, token: str, emotion_data: Dict[str, Any], emotion_complexity: str, modifier_type: str) -> float:
        m = 1.0
        for s in emotion_data.get("sub_emotions", {}).values():
            prof = s.get("emotion_profile", {}) or {}
            for lvl, exs in (prof.get("intensity_levels", {}).get("intensity_examples", {}) or {}).items():
                for ex in exs or []:
                    if token in (ex or "").lower():
                        m *= self._calculate_weighted_intensity(emotion_complexity, lvl, modifier_type)
            for cat, emos in (prof.get("related_emotions", {}) or {}).items():
                if token in [e.lower() for e in emos or []]:
                    m *= 1.2 if cat == "positive" else 0.8
        return m

    def _calculate_weighted_intensity(self, emotion_complexity: str, intensity_level: str, modifier_type: str) -> float:
        cw = {"basic": {"amplifier": 1.2, "diminisher": 0.8},
              "subtle": {"amplifier": 1.3, "diminisher": 0.7},
              "complex": {"amplifier": 1.4, "diminisher": 0.6}}
        iw = {"high": {"amplifier": 1.3, "diminisher": 0.7},
              "medium": {"amplifier": 1.2, "diminisher": 0.8},
              "low": {"amplifier": 1.1, "diminisher": 0.9}}
        return cw.get(emotion_complexity, {}).get(modifier_type, 1.0) * iw.get(intensity_level, {}).get(modifier_type, 1.0)

    def _analyze_modifiers(self, text: str, emotion_data: Dict[str, Any]) -> float:
        try:
            toks = self._tokenize_and_normalize(text or "")
            ec = emotion_data.get("metadata", {}).get("emotion_complexity", "basic")
            tl = (text or "").lower()
            acc: List[float] = []
            for tok in toks:
                mt = self._get_modifier_type(tok)
                if not mt:
                    continue
                tok_norm = (tok or "").strip().lower()
                base = self._mod_table(mt).get(tok_norm, 1.0)
                sm = self._calculate_situational_multiplier(tok, emotion_data, ec, mt)
                lm = self._calculate_linguistic_multiplier(tok, tl, emotion_data)
                pm = self._calculate_profile_multiplier(tok, emotion_data, ec, mt)
                w = base * sm * lm * pm
                if w != 1.0:
                    acc.append(float(w))
            if acc:
                mean_w = float(np.mean(acc))
                cfx = {"basic": 1.0, "subtle": 1.2, "complex": 1.4}.get(ec, 1.0)
                return float(max(min(mean_w * cfx, 5.0), 0.1))
            return 1.0
        except Exception as e:
            logger.error(f"_analyze_modifiers 오류: {e}", exc_info=True)
            return 1.0

    def _calculate_initial_base_scores(
        self,
        text: str,
        tokens: Optional[List[str]] = None,
        text_lower: Optional[str] = None,
        is_positive: Optional[bool] = None,
    ) -> Tuple[Dict[str, float], Dict[str, List[str]]]:
        tokens = tokens if tokens is not None else self._tokenize_and_normalize(text or "")
        tl = text_lower if text_lower is not None else (text or "").lower()
        is_pos = bool(is_positive) if is_positive is not None else self._is_positive_sentence(text)

        scores: Dict[str, float] = {}
        detail_hits: Dict[str, List[str]] = {}
        for eid in self.primary_emotions:
            base, hits = self._calculate_base_intensity(
                text,
                self.emotions_data.get(eid, {}),
                tokens=tokens,
                text_lower=tl,
                is_positive=is_pos,
                primary_id=eid,
            )
            scores[eid] = base
            detail_hits.update(hits)
        return scores, detail_hits

    def _calculate_transition_compatibility(self, emotion_id: str, emotion_data: Dict[str, Any], current_scores: Dict[str, float], text: str) -> float:
        comp = 1.0
        for p in emotion_data.get("emotion_transitions", {}).get("patterns", []) or []:
            to_e = p.get("to_emotion", "")
            if to_e != emotion_id:
                continue
            if not p.get("from_emotion"):
                continue
            trigs = p.get("transition_analysis", {}).get("trigger_words", []) or []
            if trigs and not any((w.lower() in (text or "").lower()) for w in trigs):
                continue
            ic = str(p.get("transition_analysis", {}).get("intensity_change", "")).lower()
            if "increase" in ic:
                comp *= 1.2
            elif "decrease" in ic:
                comp *= 0.8
        return comp

    def _analyze_multi_emotion_transitions(self, scores: Dict[str, float], text: str) -> Dict[str, float]:
        out = defaultdict(lambda: 1.0)
        tl = (text or "").lower()
        for eid in self.primary_emotions:
            ed = self.emotions_data.get(eid, {})
            for p in ed.get("emotion_transitions", {}).get("multi_emotion_transitions", []) or []:
                frm = p.get("from_emotions", []) or []
                to = p.get("to_emotions", []) or []
                trg = p.get("triggers", []) or []
                if not frm or not to or not any((t.lower() in tl) for t in trg):
                    continue
                if all(scores.get(e, 0.0) > 0.3 for e in frm):
                    ic = str(p.get("transition_analysis", {}).get("intensity_change", "")).lower()
                    for te in to:
                        if "increase" in ic:
                            out[te] *= 1.3
                        elif "decrease" in ic:
                            out[te] *= 0.7
        return out

    def _build_vocab_stats(self) -> Tuple[Dict[str, int], float]:
        V: Dict[str, int] = {}
        sizes: List[int] = []
        for eid in self.primary_emotions:
            data = self.emotions_data.get(eid, {}) or {}
            prof = data.get("emotion_profile", {}) or {}
            kw = 0
            for key in ("core_keywords", "keywords"):
                vals = prof.get(key) or []
                if isinstance(vals, list):
                    kw += len(vals)
            subs = data.get("sub_emotions") or {}
            if isinstance(subs, dict):
                kw += len(subs)
            Ve = max(1, kw)
            V[eid] = Ve
            sizes.append(Ve)
        V_avg = float(sum(sizes) / len(sizes)) if sizes else 1.0
        return V, V_avg

    def _auto_score_from_hits(self, detail_hits: Dict[str, List[str]], token_len: int) -> Dict[str, float]:
        if not detail_hits:
            return {}
        V, V_avg = self._V_cache
        if V is None or V_avg is None:
            V, V_avg = self._build_vocab_stats()
            self._V_cache = (V, V_avg)

        import math

        def _length_norm(L: int, method: str) -> float:
            L = max(1, int(L or 0))
            method = method.lower()
            if method == "none":
                return 1.0
            if method == "sqrt":
                return math.sqrt(float(L))
            return math.log(float(L) + 2.0)

        norm_method = str(self.auto_scoring.get("norm_method", "log"))
        len_norm = _length_norm(token_len, norm_method)

        detail_scores: Dict[str, float] = {}
        for sub_id, hits in detail_hits.items():
            parent = sub_id.split("/", 1)[0] if "/" in sub_id else sub_id
            Ve = float(V.get(parent, 1))
            uniq = list(dict.fromkeys(hits))[: int(self.auto_scoring.get("topk_detail", 3) or 3)]
            if not uniq:
                continue
            count = float(len(uniq))
            tf = count / max(1e-8, math.log(Ve + 2.0))
            idf = math.log((float(V_avg) + 1.0) / (Ve + 1.0)) + 1.0
            score = (tf * idf) / max(1e-8, len_norm)
            detail_scores[sub_id] = score

        per_main: Dict[str, float] = {}
        for sub_id, sc in detail_scores.items():
            parent = sub_id.split("/", 1)[0] if "/" in sub_id else sub_id
            per_main[parent] = per_main.get(parent, 0.0) + float(sc)

        clip_val = float(self.auto_scoring.get("clip", 6.0))
        lam = float(self.auto_scoring.get("balance_strength", 0.0))
        if lam > 0.0 and per_main:
            mean_val = sum(per_main.values()) / len(per_main)
            for k, v in list(per_main.items()):
                per_main[k] = min(clip_val, (1.0 - lam) * float(v) + lam * float(mean_val))
        else:
            for k, v in list(per_main.items()):
                per_main[k] = min(clip_val, float(v))
        return per_main

    def _apply_transition_effects(self, scores: Dict[str, float], text: str) -> Dict[str, float]:
        out = {}
        for eid, base in scores.items():
            ed = self.emotions_data.get(eid, {})
            comp = self._calculate_transition_compatibility(eid, ed, scores, text)
            out[eid] = base * comp
        multi = self._analyze_multi_emotion_transitions(out, text)
        for k, m in multi.items():
            if k in out:
                out[k] *= m
        return out

    def _calculate_emotion_dominance(self, emotion_id: str, score: float, text: str) -> float:
        ed = self.emotions_data.get(emotion_id, {})
        out = score
        tl = (text or "").lower()
        for combo in ed.get("linguistic_patterns", {}).get("sentiment_combinations", []) or []:
            words = [w.lower() for w in combo.get("words", [])]
            if words and all(w in tl for w in words):
                out *= (1.0 + float(combo.get("weight", 0.0)))
        return out

    def _apply_dominance_effects(self, scores: Dict[str, float], text: str) -> Dict[str, float]:
        return {eid: self._calculate_emotion_dominance(eid, sc, text) for eid, sc in scores.items()}

    def _finalize_intensity_results(self, final_scores: Dict[str, float], base_scores: Dict[str, float], text: str) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        sorted_e = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        top3_ids = {e for e, _ in sorted_e[:3]}
        surf = self._surface_boost(text)

        # 간단한 신뢰도 추정: top-gap, 분산, 커버리지
        vals = [float(v) for _, v in sorted_e]
        top = vals[0] if vals else 0.0
        sec = vals[1] if len(vals) > 1 else 0.0
        gap = max(0.0, top - sec)
        cover = sum(1 for v in vals if v >= 0.3) / max(1, len(vals))
        base_conf = min(0.95, 0.40 + 0.40 * top + 0.15 * gap + 0.05 * cover)
        # [PATCH] expose base confidence to scaler
        try:
            self._last_base_conf = float(base_conf)
        except Exception:
            self._last_base_conf = 0.0

        for eid, val in sorted_e:
            if val < 0.3 and eid not in top3_ids:
                continue
            mod_factor = float(self._analyze_modifiers(text, self.emotions_data.get(eid, {})))
            mod_raw = float(val) * mod_factor * float(surf)
            mod = self._compress_intensity(mod_raw)
            # [PATCH] 짧은 텍스트·부정 표지 게이트(경계 매칭) + 최종 안전 클램프
            tl = (text or "").lower()
            tok_len = len(self._tokenize_and_normalize(text))
            base_conf_local = float(getattr(self, "_last_base_conf", 0.0))
            # 부정 표지 경계 매칭
            neg_set = set()
            try:
                self._ensure_modifier_lexicon()
                neg_set = set(self._modifier_lexicon.get("negation", set()))
            except Exception:
                pass
            try:
                pat = self._compile_negation_regex(neg_set)
                neg_present = bool(pat.search(tl))
            except Exception:
                neg_present = any((n in tl) for n in neg_set)

            # 단문 상향 캡(증거 강하면 완화)
            force_medium = False
            if tok_len < 4 and mod > 0.65:
                mod = 0.75 if base_conf_local >= 0.5 else 0.65
                force_medium = True
                self._qa["short_text_cap"] += 1

            # 부정 표지 발견 시 과신 억제
            if neg_present and mod > 0.7:
                mod *= 0.9
                self._qa["negation_damp"] += 1

            # 최종 안전 범위
            if not math.isfinite(mod):
                mod = 0.0
            mod = max(0.0, min(1.0, float(mod)))
            level = "high" if mod >= 0.7 else ("medium" if mod >= 0.3 else "low")
            if force_medium and level == "high":
                level = "medium"

            # flags/관측성 주입
            flags: List[str] = []
            if mod_factor and abs(mod_factor - 1.0) >= 0.2:
                flags.append(f"modifier:{mod_factor:.2f}")
            if float(surf) >= 1.2:
                flags.append(f"surface:{float(surf):.2f}")
            if float(base_scores.get(eid, 0.0)) < 0.15:
                flags.append("weak_evidence")
            if base_conf < 0.30:
                flags.append("low_confidence")

            # 신뢰도 게이팅: 과한 레벨 상향 억제
            lvl_map = {"low": 0, "medium": 1, "high": 2}
            if base_conf < 0.25 and lvl_map.get(level, 0) > 1:
                level = "medium"

            out[eid] = {
                "intensity_score": round(float(base_scores.get(eid, 0.0)), 3),
                "modified_score": round(float(mod), 3),
                "level": level,
                "confidence": round(float(base_conf), 3),
                "flags": flags,
            }

        # 최소 스키마 보장
        if not out:
            out = self._minimal_output(0.20, ["no_evidence"])
        return out

    def ensure_phrase_index(self):
        if getattr(self, "_phr2ids", None) is not None: return
        p2e = {}  # phrase -> set(eid)
        # ... 기존 순회에서 keywords/variations/examples/transition 트리거 수집 ...
        self._phr2ids = p2e
        if _AHO_OK:
            A = ahocorasick.Automaton()
            for ph, ids in p2e.items(): A.add_word(ph, ids)
            A.make_automaton(); self._aho = A
        else:
            self._aho = None

    def get_candidates(self, text: str):
        self.ensure_phrase_index()
        cand = set()
        if self._aho:
            for _, ids in self._aho.iter(text): cand |= set(ids)
        else:
            # 기존 정규식 샤드 경로
            for rx in getattr(self, '_rx_phrase_shards', []):
                for m in rx.finditer(text):
                    cand |= set(self._phr2ids.get((m.group(0) or "").lower(), []))
        return cand

    def analyze_emotion_intensity(self, text: str) -> Dict[str, Dict[str, Any]]:
        try:
            if getattr(self, "_expose_qa", False):
                for k in list(self._qa.keys()):
                    self._qa[k] = 0
            t = _ud.normalize("NFKC", (text or "")).strip()
            if not t:
                return self._minimal_output(0.05, ["empty_input"])
            # Junk-Guard: 난수/잡음 입력이면 안전 수렴
            is_junk, jscore, _ = self._junk_guard(t)
            if is_junk:
                return self._minimal_output(0.10, [f"junk_input:{jscore:.2f}"])
            tokens = self._tokenize_and_normalize(t)
            # [PATCH] 동적 로지스틱 게인을 위한 토큰 길이 힌트 저장
            self._last_token_len = len(tokens)
            tl = (t or "").lower()
            is_pos = self._is_positive_sentence(t)
            
            # 라벨링 순회: EMOTIONS.json의 세부 감정 패턴 매칭
            base, detail_hits = self._calculate_initial_base_scores(t, tokens=tokens, text_lower=tl, is_positive=is_pos)
            
            # 추가 강화: 세부 감정 분석 결과를 기본 점수에 반영
            sub_emotion_scores = self._analyze_sub_emotions(t)
            for emotion_id, sub_score in sub_emotion_scores.items():
                primary_emotion = emotion_id.split('-')[0] if '-' in emotion_id else emotion_id
                if primary_emotion in base:
                    # 세부 감정 점수를 기본 점수에 가중치로 반영
                    base[primary_emotion] = max(base[primary_emotion], sub_score * 0.7)
            
            if self.auto_scoring.get("enabled", False):
                auto_main = self._auto_score_from_hits(detail_hits, token_len=len(tokens))
                # [PATCH] 히트/길이에 따라 동적 가중(0.3→최대 0.5)
                hit_strength = min(1.0, len(detail_hits) / 12.0)
                w = 0.3 + 0.2 * hit_strength
                for eid, score in auto_main.items():
                    base[eid] = (1.0 - w) * float(base.get(eid, 0.0)) + w * float(score)
            # [PATCH] Borderline Junk 완화: 0.4~0.6 회색지대에서 완만 감쇠
            base = self._borderline_junk_mitigate(jscore, base)
            after_tr = self._apply_transition_effects(base, t)
            dom = self._apply_dominance_effects(after_tr, t)
            res = self._finalize_intensity_results(dom, base, t)
            # [PATCH] (선택) 결과에 QA 카운터 노출 (예약 키 사용)
            if getattr(self, "_expose_qa", False):
                res["_meta"] = {"qa": dict(self._qa)}
            return res
        except Exception as e:
            logger.error(f"감정 강도 분석 오류: {e}", exc_info=True)
            return self._minimal_output(0.05, ["error"])

    def _score_sub_emotion_linguistic(self, sub_data: Dict[str, Any], text_lower: str) -> float:
        sc = 0.0
        ling = sub_data.get("linguistic_patterns", {}) or {}
        for kp in ling.get("key_phrases", []) or []:
            pat = (kp.get("pattern", "") or "").lower()
            ctx = (kp.get("context_requirement", "") or "").lower()
            if pat and pat in text_lower and (not ctx or ctx in text_lower):
                sc += float(kp.get("weight", 0.0)) * 1.5
        for combo in ling.get("sentiment_combinations", []) or []:
            words = [w.lower() for w in combo.get("words", [])]
            if words and all(w in text_lower for w in words):
                sc += float(combo.get("weight", 0.0)) * 2.0
        mods = ling.get("sentiment_modifiers", {}) or {}
        for mtype, md in mods.items():
            if isinstance(md, dict):
                for w, v in md.items():
                    if str(w).lower() in text_lower:
                        sc *= float(v)
            elif isinstance(md, list):
                for w in md:
                    if str(w).lower() in text_lower:
                        sc *= 1.2 if mtype == "amplifiers" else 0.8
        return sc

    def _score_sub_emotion_situation(self, sub_data: Dict[str, Any], text_tokens: List[str], text_lower: str) -> float:
        sc = 0.0
        tset = set(text_tokens)
        for sit in sub_data.get("context_patterns", {}).get("situations", {}).values():
            matched = 0.0
            total = 0
            desc = sit.get("description", "")
            if desc and any(tok in (desc or "").lower() for tok in text_tokens):
                matched += 1
            total += 1
            for var in sit.get("variations", []) or []:
                if any(tok in (var or "").lower() for tok in text_tokens):
                    matched += 1
            total += len(sit.get("variations", []) or [])
            kws = [k.lower() for k in (sit.get("keywords", []) or [])]
            matched += sum(1 for kw in kws if kw in text_lower)
            total += len(kws)
            for ex in sit.get("examples", []) or []:
                et = self._tokenize_and_normalize(ex)
                if et:
                    common = tset & set(et)
                    matched += len(common) / len(et)
            total += len(sit.get("examples", []) or [])
            if total > 0:
                s = matched / total
                s *= {"high": 1.5, "medium": 1.2, "low": 1.0}.get(sit.get("intensity", "medium"), 1.0)
                for stg, d in (sit.get("emotion_progression", {}) or {}).items():
                    if d and str(d).lower() in text_lower:
                        s *= {"trigger": 1.1, "development": 1.3, "peak": 1.5, "aftermath": 0.9}.get(stg, 1.0)
                sc += s
        return sc

    def _score_sub_emotion_profile(self, sub_data: Dict[str, Any], text_tokens: List[str], text_lower: str) -> float:
        sc = 0.0
        prof = sub_data.get("emotion_profile", {}) or {}
        tset = set(text_tokens)
        for lvl, exs in (prof.get("intensity_levels", {}).get("intensity_examples", {}) or {}).items():
            w = {"high": 1.5, "medium": 1.2, "low": 1.0}.get(lvl, 1.0)
            for ex in exs or []:
                et = self._tokenize_and_normalize(ex)
                if et:
                    common = tset & set(et)
                    sc += (len(common) / len(et)) * w
        sc += sum(1 for kw in (prof.get("core_keywords", []) or []) if str(kw).lower() in text_lower) * 1.5
        for cat, emos in (prof.get("related_emotions", {}) or {}).items():
            w = 1.2 if cat == "positive" else 1.0
            sc += sum(1 for e in (emos or []) if str(e).lower() in text_lower) * w
        return sc

    def _analyze_sub_emotions(self, text: str) -> Dict[str, float]:
        try:
            tl = (text or "").lower()
            toks = self._tokenize_and_normalize(text or "")
            token_set = set(toks)
            scores: Dict[str, float] = {}
            candidate_limit = min(30, int(self._limits.get("max_sub_pairs_per_step", 200)))
            for eid in self.primary_emotions:
                ed = self.emotions_data.get(eid, {})
                subs = (ed.get("sub_emotions", {}) or {})
                if not isinstance(subs, dict) or not subs:
                    continue
                cand_ids = self._candidate_subs(eid, token_set, topk=candidate_limit) if hasattr(self, "_candidate_subs") else []
                if not cand_ids:
                    cand_ids = [str(k) for k in subs.keys()]
                for sub_id in cand_ids:
                    actual_key, sd = self._resolve_sub_entry(subs, sub_id)
                    if sd is None:
                        continue
                    ls = self._score_sub_emotion_linguistic(sd, tl)
                    ss = self._score_sub_emotion_situation(sd, toks, tl)
                    ps = self._score_sub_emotion_profile(sd, toks, tl)
                    tot = ls + ss + ps
                    if tot > 0:
                        cpx = {"basic": 1.0, "subtle": 1.2, "complex": 1.4}.get(sd.get("metadata", {}).get("emotion_complexity", "basic"), 1.0)
                        final = tot * cpx
                        if final > 0:
                            scores[str(actual_key)] = round(float(final), 3)
            return scores
        except Exception as e:
            logger.error(f"세부감정 분석 오류: {e}", exc_info=True)
            return {}

    def _split_sentences_ko(self, text: str) -> List[str]:
        """
        한국어 문장을 전환 어휘와 문장부호를 기준으로 분할한다.
        전이 토큰(예: '하지만')이 없으면 기본 연결어 목록을 사용한다.
        """
        if not text:
            return []
        import re

        self._ensure_modifier_lexicon()
        reversers = sorted(self._modifier_lexicon.get("reverser", set()))
        fallback = ["하지만", "그러나", "반면", "그래서", "결국", "그러다가", "그런데", "따라서", "이후", "때문에"]
        boundary_tokens = reversers if reversers else fallback
        tok_alt = "|".join(map(re.escape, boundary_tokens)) if boundary_tokens else None
        splitter = " ? "

        s = str(text).strip()
        if not s:
            return []

        if tok_alt:
            s = re.sub(rf"\s*(?:{tok_alt})\s*", splitter, s)
            s = re.sub(rf",\s*(?=(?:{tok_alt})(?:\s|$))", splitter, s)

        s = re.sub(r"([.?!]+)\s*", r"\1" + splitter, s)

        parts = [p.strip(" ,;:") for p in s.split("?") if p and p.strip()]
        try:
            import os
            if os.environ.get("EMOTION_DEBUG_SPLIT") in ("1", "true", "True"):
                logger.info(f"[split] {text} -> {parts}")
        except Exception:
            pass
        return parts

    def analyze_intensity_transitions(self, text: str) -> Dict[str, Any]:
        try:
            # 라벨링 순회: EMOTIONS.json의 전이 패턴 매칭
            sentences = self._split_sentences_ko(text)
            if len(sentences) < 2:
                return {"temporal_patterns": {}, "transitions": [], "sub_transitions": [], "statistics": {}}

            transitions = []
            sub_transitions = []
            temporal_intensities = []
            sub_scores_seq = []
            
            # 각 문장별로 감정 강도 분석
            for i, sentence in enumerate(sentences):
                # 대표 감정 강도 분석
                emotion_intensity = self.analyze_emotion_intensity(sentence)
                temporal_intensities.append(emotion_intensity)
                
                # 세부 감정 분석
                sub_scores = self._analyze_sub_emotions(sentence)
                sub_scores_seq.append(sub_scores)
            
            # 전이 패턴 분석
            for i in range(len(sentences) - 1):
                current_intensities = temporal_intensities[i]
                next_intensities = temporal_intensities[i + 1]
                
                # 주요 감정 전이 감지
                current_dominant = max(current_intensities.keys(), 
                                    key=lambda k: current_intensities[k].get('modified_score', 0.0))
                next_dominant = max(next_intensities.keys(), 
                                 key=lambda k: next_intensities[k].get('modified_score', 0.0))
                
                if current_dominant != next_dominant:
                    current_score = current_intensities[current_dominant].get('modified_score', 0.0)
                    next_score = next_intensities[next_dominant].get('modified_score', 0.0)
                    
                    transitions.append({
                        "from_emotion": current_dominant,
                        "to_emotion": next_dominant,
                        "sentence_index": i + 1,
                        "confidence": min(abs(next_score - current_score) * 2, 1.0),
                        "trigger_text": sentences[i + 1][:50],
                        "intensity_change": next_score - current_score
                    })
                
                # 세부 감정 전이 감지
                current_sub = sub_scores_seq[i]
                next_sub = sub_scores_seq[i + 1]
                
                for sub_emotion in set(current_sub.keys()) | set(next_sub.keys()):
                    current_sub_score = current_sub.get(sub_emotion, 0.0)
                    next_sub_score = next_sub.get(sub_emotion, 0.0)
                    
                    if abs(next_sub_score - current_sub_score) > 0.1:
                        sub_transitions.append({
                            "sub_emotion": sub_emotion,
                            "sentence_index": i + 1,
                            "score_change": next_sub_score - current_sub_score,
                            "confidence": min(abs(next_sub_score - current_sub_score) * 3, 1.0)
                        })
            
            # 시간적 패턴 분석
            temporal_patterns = self.transformer.analyze_temporal_patterns(temporal_intensities)
            
            return {
                "temporal_patterns": temporal_patterns,
                "transitions": transitions,
                "sub_transitions": sub_transitions,
                "statistics": {
                    "total_transitions": len(transitions),
                    "total_sub_transitions": len(sub_transitions),
                    "sentence_count": len(sentences),
                    "avg_intensity_change": sum(t.get('intensity_change', 0.0) for t in transitions) / max(len(transitions), 1)
                }
            }
        except Exception as e:
            logger.error(f"전이 분석 오류: {e}", exc_info=True)
            return {"temporal_patterns": {}, "transitions": [], "sub_transitions": [], "statistics": {}}

    def _map_progression_stage(self, stage: str) -> float:
        m = {"trigger": 1.05, "development": 1.1, "peak": 1.2, "aftermath": 0.9, "end": 0.8, "initial": 1.0, "회복": 1.05, "반전": 1.15}
        key = str(stage or "").lower()
        if key not in m:
            logger.warning(f"알 수 없는 stage '{stage}', 1.0 사용")
        return m.get(key, 1.0)

    def _calculate_example_similarity(self, example: str, text_tokens: List[str]) -> float:
        et = self._tokenize_and_normalize(example or "")
        if not et:
            return 0.0
        tset = set(text_tokens)
        common = tset & set(et)
        if not common:
            return 0.0
        sim = len(common) / len(et)
        mc, run = 0, 0
        for tok in et:
            if tok in tset:
                run += 1
                mc = max(mc, run)
            else:
                run = 0
        return float(sim + (mc / len(et)) * 0.5)

    def _analyze_profile_examples_for_situation(self, emotion_profile: Dict[str, Any], text_tokens: List[str]) -> Dict[str, Any]:
        best = {lvl: {"score": 0.0, "example": ""} for lvl in ("low", "medium", "high")}
        if not text_tokens:
            return best
        tset = set(text_tokens)
        ex_map = (emotion_profile.get("intensity_levels", {}) or {}).get("intensity_examples", {}) or {}
        for lvl, exs in ex_map.items():
            if lvl not in best or not exs:
                continue
            for ex in exs:
                if ex not in self._example_token_pool:
                    self._example_token_pool[ex] = set(self._tokenize_and_normalize(ex))
                eset = self._example_token_pool[ex]
                if not eset:
                    continue
                jac = len(tset & eset) / len(tset | eset)
                if jac > best[lvl]["score"]:
                    best[lvl] = {"score": float(jac), "example": ex}
                    if jac == 1.0:
                        break
        return best

    def _calculate_emotional_relatedness(self, emotion_profile: Dict[str, Any], normalized_text: str) -> float:
        sc = 0.0
        weights = {"positive": 1.2, "negative": 0.8, "neutral": 1.0}
        for cat, emos in (emotion_profile.get("related_emotions", {}) or {}).items():
            w = weights.get(cat, 1.0)
            for e in emos or []:
                if str(e).lower() in normalized_text:
                    sc += w
        return float(sc)

    def _evaluate_context_significance(self, situation_data: Dict[str, Any], profile_matches: Dict[str, Any], normalized_text: str) -> float:
        sc = 0.0
        prog = situation_data.get("emotion_progression", {}) or {}
        wmap = {"trigger": 1.1, "development": 1.2, "peak": 1.5, "aftermath": 0.9}
        for st, desc in prog.items():
            if desc and str(desc).lower() in normalized_text:
                sc += wmap.get(st, 1.0)
                if st == "peak" and profile_matches.get("high", {}).get("score", 0.0) > 0.7:
                    sc *= 1.3
        return float(sc)

    def _calculate_final_situational_weight(self, situation_data: Dict[str, Any], sub_emotion_data: Dict[str, Any], context_scores: Dict[str, Any]) -> float:
        base_lvl = situation_data.get("intensity", "medium")
        eid = sub_emotion_data.get("metadata", {}).get("primary_category")
        w = self._get_intensity_weight(base_lvl, eid)
        w *= (1.0 + float(context_scores.get("context_significance", 0.0)))
        w *= (1.0 + float(context_scores.get("emotional_relatedness", 0.0)) * 0.2)
        cpx = {"basic": 1.0, "subtle": 1.2, "complex": 1.4}.get(sub_emotion_data.get("metadata", {}).get("emotion_complexity", "basic"), 1.0)
        return float(w * cpx)

    def _infer_intensity_label(self, example_similarity: Dict[str, float], ctx_sig: float, related: float) -> str:
        if not isinstance(example_similarity, dict):
            example_similarity = {}
        sim = max(example_similarity.values() or [0.0])
        score = 0.5 * sim + 0.3 * float(ctx_sig) + 0.2 * float(related)
        return "high" if score >= 0.66 else ("medium" if score >= 0.33 else "low")


    def _ensure_modifier_lexicon(self) -> None:
        """
        EMOTIONS.JSON(전역/감정별/세부분류)에서 증폭/완화/부정/전환 표지를 수집해 캐싱.
        사전 구조: self._modifier_lexicon = {
            "amplifier": set(), "diminisher": set(), "negation": set(), "reverser": set()
        }
        """
        if getattr(self, "_modifier_lexicon", None) is not None:
            return

        lex: Dict[str, Set[str]] = {
            "amplifier": set(),
            "diminisher": set(),
            "negation": set(),
            "reverser": set(),  # '하지만/그러나' 등의 담화 전환 표지
        }

        def _add(vals, key: str):
            if not vals:
                return
            if isinstance(vals, dict):
                for v in vals.values():
                    _add(v, key)
                return
            if isinstance(vals, (list, tuple, set)):
                for v in vals:
                    if isinstance(v, str) and v.strip():
                        lex[key].add(v.strip().lower())
                return
            if isinstance(vals, str) and vals.strip():
                lex[key].add(vals.strip().lower())

        # 0) 기존에 self.modifiers 구조가 있다면 우선 사용
        if hasattr(self, "modifiers") and isinstance(self.modifiers, dict):
            _add(self.modifiers.get("amplifiers"), "amplifier")
            _add(self.modifiers.get("intensifiers"), "amplifier")
            _add(self.modifiers.get("diminishers"), "diminisher")
            _add(self.modifiers.get("downtoners"), "diminisher")
            _add(self.modifiers.get("negations"), "negation")
            _add(self.modifiers.get("reversers"), "reverser")

        # 1) 전역 섹션(있다면)
        root_lp = (self.emotions_data.get("linguistic_patterns") or {})
        _add(root_lp.get("amplifiers"), "amplifier")
        _add(root_lp.get("intensifiers"), "amplifier")
        _add(root_lp.get("diminishers"), "diminisher")
        _add(root_lp.get("downtoners"), "diminisher")
        _add(root_lp.get("negations"), "negation")
        _add(root_lp.get("reversers"), "reverser")

        # 2) 1차 감정별
        for ed in (self.emotions_data or {}).values():
            if not isinstance(ed, dict):
                continue
            lp = (ed.get("linguistic_patterns") or {})
            _add(lp.get("amplifiers"), "amplifier")
            _add(lp.get("intensifiers"), "amplifier")
            _add(lp.get("diminishers"), "diminisher")
            _add(lp.get("downtoners"), "diminisher")
            _add(lp.get("negations"), "negation")
            _add(lp.get("reversers"), "reverser")

            subs = ed.get("sub_emotions") or {}
            if isinstance(subs, dict):
                for sd in subs.values():
                    if not isinstance(sd, dict):
                        continue
                    lp2 = (sd.get("linguistic_patterns") or {})
                    _add(lp2.get("amplifiers"), "amplifier")
                    _add(lp2.get("intensifiers"), "amplifier")
                    _add(lp2.get("diminishers"), "diminisher")
                    _add(lp2.get("downtoners"), "diminisher")
                    _add(lp2.get("negations"), "negation")
                    _add(lp2.get("reversers"), "reverser")

        # 3) 기본 폴백(사전이 비어 있을 때만)
        use_fb = bool(getattr(self, "_modifier_config", {}).get("allow_fallback_modifiers", False))
        if use_fb and not lex["amplifier"]:
            lex["amplifier"].update({"매우", "아주", "정말", "너무", "굉장히", "대단히", "무척", "엄청", "완전", "진짜"})
        if use_fb and not lex["diminisher"]:
            lex["diminisher"].update({"조금", "살짝", "다소", "약간", "그다지", "별로"})
        if use_fb and not lex["negation"]:
            lex["negation"].update({"안", "못", "아니", "않", "없", "없다", "아니다", "못하다"})
        if use_fb and not lex["reverser"]:
            lex["reverser"].update({"하지만", "그러나", "반면", "그런데", "다만", "그렇지만"})

        self._modifier_lexicon = lex

    def _get_modifier_type(self, token: str) -> Optional[str]:
        """
        토큰이 증폭/완화/부정/전환 표지인지 판별하여 타입 문자열을 반환.
        반환값: "amplifier" | "diminisher" | "negation" | "reverser" | None
        """
        self._ensure_modifier_lexicon()
        if not token:
            return None

        t = str(token).strip().lower()
        if not t:
            return None

        L = self._modifier_lexicon  # type: ignore[attr-defined]

        if t in L["negation"]:
            return "negation"
        if t in L["amplifier"]:
            return "amplifier"
        if t in L["diminisher"]:
            return "diminisher"
        if t in L["reverser"]:
            return "reverser"

        # 간단한 패턴 폴백 (형태 변형/합성어 대응)
        # 예: "너무나도", "전혀", "별로", "~지 않다"류
        if t.startswith("너무") or t.endswith("나도"):
            return "amplifier"
        if t in {"전혀", "결코"} or ("않" in t) or ("아니" in t) or ("못" in t and len(t) <= 3):
            return "negation"
        if t.startswith("별로") or t.startswith("그다지"):
            return "diminisher"
        if t in {"하지만", "그러나", "반면", "그런데", "다만", "그렇지만"}:
            return "reverser"

        return None

    def _build_sub_phrase_index(self) -> Dict[str, Dict[str, Set[str]]]:
        """
        세부감정 후보 Top-K 선별을 위한 어휘 인덱스를 구축합니다.
        구조: { primary_id: { sub_id: set(tokens >= 2 chars) } }
        """
        index: Dict[str, Dict[str, Set[str]]] = {}
        for pid, pdata in (self.emotions_data or {}).items():
            if not isinstance(pdata, dict):
                continue
            subs = pdata.get("sub_emotions") or {}
            if not isinstance(subs, dict):
                continue
            bucket: Dict[str, Set[str]] = {}
            for sid, sdata in subs.items():
                if not isinstance(sdata, dict):
                    continue
                vocab: Set[str] = set()
                prof = (sdata.get("emotion_profile") or {})
                # 핵심/보조 키워드
                for key in ("core_keywords", "keywords"):
                    for kw in (prof.get(key) or []):
                        if isinstance(kw, str):
                            kw_norm = kw.strip().lower()
                            if len(kw_norm) >= 2:
                                vocab.add(kw_norm)
                            vocab.update({t for t in self._tokenize_and_normalize(kw) if len(t) >= 2})
                # 강도 예시
                intensity_examples = (prof.get("intensity_levels") or {}).get("intensity_examples", {})
                if isinstance(intensity_examples, dict):
                    for examples in intensity_examples.values():
                        for ex in (examples or []):
                            if isinstance(ex, str):
                                vocab.update({t for t in self._tokenize_and_normalize(ex) if len(t) >= 2})
                # 상황 패턴
                cp = (sdata.get("context_patterns") or {})
                situations = cp.get("situations") or {}
                if isinstance(situations, dict):
                    for sit in situations.values():
                        if not isinstance(sit, dict):
                            continue
                        for kw in (sit.get("keywords") or []):
                            if isinstance(kw, str):
                                kw_norm = kw.strip().lower()
                                if len(kw_norm) >= 2:
                                    vocab.add(kw_norm)
                        for ex in (sit.get("examples") or []):
                            if isinstance(ex, str):
                                vocab.update({t for t in self._tokenize_and_normalize(ex) if len(t) >= 2})
                        for var in (sit.get("variations") or []):
                            if isinstance(var, str):
                                vocab.update({t for t in self._tokenize_and_normalize(var) if len(t) >= 2})
                        progression = sit.get("emotion_progression") or {}
                        if isinstance(progression, dict):
                            for desc in progression.values():
                                if isinstance(desc, str):
                                    vocab.update({t for t in self._tokenize_and_normalize(desc) if len(t) >= 2})
                bucket[str(sid)] = {tok for tok in vocab if len(tok) >= 2}
            if bucket:
                index[str(pid)] = bucket
        return index

    def _candidate_subs(self, primary_id: str, token_set: Set[str], topk: int = None) -> List[str]:
        # 환경변수에서 Top-K 설정 가져오기 (기본값 12)
        if topk is None:
            topk = int(os.environ.get("INT_TOPK_SUB", "12"))
        """
        주어진 대표 감정(primary_id) 아래에서 토큰 교집합 수 기준 Top-K 세부감정을 반환.
        교집합이 없으면 임의 Top-K 반환.
        """
        bucket = self._sub_phrase_index.get(str(primary_id), {})
        if not bucket:
            return []
        hits = [(sid, len(vocab & token_set)) for sid, vocab in bucket.items() if vocab & token_set]
        if not hits:
            return list(bucket.keys())[: max(1, topk)]
        hits.sort(key=lambda x: x[1], reverse=True)
        return [sid for sid, _ in hits[: max(1, topk)]]

    @staticmethod
    def _resolve_sub_entry(sub_map: Dict[Any, Any], sub_id: str) -> Tuple[Optional[Any], Optional[Dict[str, Any]]]:
        """
        서브 감정 ID 조회 시 실제 키 타입/문자열 키 모두 유연하게 대응.
        반환: (실제키, 서브데이터) 또는 (None, None)
        """
        if sub_id in sub_map:
            return sub_id, sub_map[sub_id]
        for key, value in sub_map.items():
            if str(key) == sub_id:
                return key, value
        return None, None

    def analyze_situational_intensity(self, text: str) -> Dict[str, Any]:
        try:
            # 라벨링 순회: EMOTIONS.json의 상황 패턴 매칭
            nt = (text or "").strip().lower()
            tokens = self._tokenize_and_normalize(text or "")
            token_set = set(tokens)
            
            results = {}
            
            # EMOTIONS.json의 모든 감정 데이터 순회
            for primary_emotion, emotion_info in self.emotions_data.items():
                sub_emotions = emotion_info.get('sub_emotions', {}) or {}
                
                for sub_emotion, sub_info in sub_emotions.items():
                    context_patterns = sub_info.get('context_patterns', {}) or {}
                    situations = context_patterns.get('situations', {}) or {}
                    
                    for situation_key, situation_data in situations.items():
                        # 상황별 키워드 매칭
                        keywords = situation_data.get('keywords', []) or []
                        examples = situation_data.get('examples', []) or []
                        variations = situation_data.get('variations', []) or []
                        
                        matched_keywords = []
                        matched_examples = []
                        matched_variations = []
                        
                        # 키워드 매칭
                        for kw in keywords:
                            if isinstance(kw, str) and kw.lower() in nt:
                                matched_keywords.append(kw)
                        
                        # 예시 매칭
                        for ex in examples:
                            if isinstance(ex, str) and ex.lower() in nt:
                                matched_examples.append(ex)
                        
                        # 변형 매칭
                        for var in variations:
                            if isinstance(var, str) and var.lower() in nt:
                                matched_variations.append(var)
                        
                        # 매칭이 있으면 결과에 추가
                        if matched_keywords or matched_examples or matched_variations:
                            intensity_level = situation_data.get('intensity', 'medium')
                            intensity_weight = {'low': 0.3, 'medium': 0.6, 'high': 0.9}.get(intensity_level, 0.5)
                            
                            # 매칭 점수 계산
                            score = (len(matched_keywords) * 0.4 + 
                                   len(matched_examples) * 0.3 + 
                                   len(matched_variations) * 0.3) * intensity_weight
                            
                            results[f"{primary_emotion}-{sub_emotion}-{situation_key}"] = {
                                "matched_keywords": matched_keywords,
                                "matched_examples": matched_examples,
                                "matched_variations": matched_variations,
                                "final_intensity_weight": min(score, 1.0),
                                "detected_intensity": intensity_level,
                                "primary_emotion": primary_emotion,
                                "sub_emotion": sub_emotion,
                                "situation_key": situation_key
                            }
            
            return results
            
        except Exception as e:
            logger.error(f"상황 강도 분석 오류: {e}", exc_info=True)
            return {}

    def _safe_stack_mean(self, tensors: List[torch.Tensor]) -> torch.Tensor:
        if not tensors:
            hs = getattr(getattr(getattr(self, "model", None), "config", None), "hidden_size", 768)
            return torch.zeros(hs, dtype=self._default_tensor_dtype, device=self._device)
        stack = [t.to(self._default_tensor_dtype) for t in tensors]
        return torch.mean(torch.stack(stack, dim=0), dim=0)

    def _get_context_cache_engine(self):
        if not self._use_embedding_cache or self._context_cache_error:
            return None
        with self._context_cache_lock:
            if self._context_cache_engine is not None:
                # 캐시 엔진은 1회만 초기화
                return self._context_cache_engine
            try:
                try:
                    from src.embedding_cache import get_global_cache_engine
                except ImportError:
                    from embedding_cache import get_global_cache_engine  # type: ignore
                self._context_cache_engine = get_global_cache_engine()
                return self._context_cache_engine
            except Exception as e:
                logger.debug("[embedding_cache] context cache init failed: %s", e, exc_info=True)
                self._context_cache_error = True
                return None

    def _compute_context_embedding_np(self, context: Any) -> np.ndarray:
        self._ensure_model_loaded()

        if not isinstance(context, str):
            context = str(context) if context is not None else ""
        context = context.strip()

        dtype = np.float16 if self._autocast_enabled else np.float32
        hs = getattr(getattr(getattr(self, "model", None), "config", None), "hidden_size", 768)

        if not context:
            return np.zeros((hs,), dtype=dtype)

        tok_kwargs = {
            "text": context,
            "return_tensors": "pt",
            "padding": True,
            "truncation": True,
            "max_length": self._tokenizer_max_length,
        }
        if self._pad_to_multiple > 0:
            tok_kwargs["pad_to_multiple_of"] = self._pad_to_multiple
        try:
            toks = self.tokenizer(**tok_kwargs)
        except ValueError:
            return np.zeros((hs,), dtype=dtype)
        except Exception as e:
            logger.debug("[embedding_cache] tokenizer failed: %s", e, exc_info=True)
            return np.zeros((hs,), dtype=dtype)

        toks = {k: v.to(self._device) for k, v in toks.items()}
        try:
            with torch.no_grad():
                with self._autocast_ctx():
                    out = self.model(**toks).last_hidden_state
        except Exception as e:
            logger.debug("[embedding_cache] model forward failed: %s", e, exc_info=True)
            return np.zeros((hs,), dtype=dtype)

        vec = torch.mean(out, dim=1).squeeze(0).detach().to(self._default_tensor_dtype)
        base = vec.cpu().numpy()
        if base.dtype != dtype:
            base = base.astype(dtype, copy=False)
        return base

    def _get_context_embedding_tensor(self, context: str) -> torch.Tensor:
        norm = (context or "").strip()
        hs = getattr(getattr(getattr(self, "model", None), "config", None), "hidden_size", 768)
        if not norm or not self._generate_embeddings_enabled:
            return torch.zeros(hs, dtype=self._default_tensor_dtype, device=self._device)

        cached = self._raw_context_cache.get(norm)
        if cached is not None:
            return cached.detach().clone().to(self._default_tensor_dtype)

        cache_engine = self._get_context_cache_engine() if self._use_embedding_cache else None
        tensor: Optional[torch.Tensor] = None
        if cache_engine is not None:
            try:
                vec_np = cache_engine.get_embedding(norm, compute_func=self._compute_context_embedding_np)
                tensor = torch.from_numpy(vec_np).to(self._device)
            except Exception as e:
                logger.debug("[embedding_cache] get_embedding failed: %s", e, exc_info=True)

        if tensor is None:
            self._bulk_compute_embeddings([norm])
            tensor = self._raw_context_cache.get(norm)

        if tensor is None:
            vec_np = self._compute_context_embedding_np(norm)
            tensor = torch.from_numpy(vec_np).to(self._device)

        tensor = tensor.to(self._default_tensor_dtype)
        self._raw_context_cache.setdefault(norm, tensor.detach())
        return tensor.clone()

    @lru_cache(maxsize=4096)
    def _cached_weighted_embedding(self, context: str, intensity: float) -> torch.Tensor:
        if (
            not self._generate_embeddings_enabled
            or not self._embeddings_enabled
            or not isinstance(context, str)
            or not context.strip()
        ):
            hs = getattr(getattr(getattr(self, "model", None), "config", None), "hidden_size", 768)
            return torch.zeros(hs, dtype=self._default_tensor_dtype, device=self._device)

        base = self._get_context_embedding_tensor(context)
        intensity_value = max(0.0, float(intensity))
        if intensity_value == 0.0:
            return torch.zeros_like(base)
        return base * intensity_value

    def generate_embeddings(
        self,
        intensity_results: Dict[str, Dict[str, float]],
        top_k_context: Optional[int] = None,
        top_k_sub: Optional[int] = None,
        fp16: Optional[bool] = None,
    ) -> Dict[str, Any]:
        hs_fallback = getattr(getattr(getattr(self, "model", None), "config", None), "hidden_size", 768)
        if not intensity_results:
            return {"primary": {}, "sub": {}, "contextual": {}, "hierarchical": {}, "combined": {}, "hidden_size": hs_fallback}

        if not self._generate_embeddings_enabled:
            return {"primary": {}, "sub": {}, "contextual": {}, "hierarchical": {}, "combined": {}, "hidden_size": hs_fallback}

        if not self._embeddings_enabled and getattr(self, "_allow_embeddings", False):
            try:
                self._load_pretrained_model()
            except Exception:
                pass

        if (
            not self._ensure_model_loaded()
            or not self._embeddings_enabled
            or self.tokenizer is None
            or self.model is None
        ):
            hs = getattr(getattr(getattr(self, "model", None), "config", None), "hidden_size", hs_fallback)
            return {"primary": {}, "sub": {}, "contextual": {}, "hierarchical": {}, "combined": {}, "hidden_size": hs}

        top_k_context = self._resolve_limit(top_k_context, self._max_context)
        top_k_sub = self._resolve_limit(top_k_sub, self._max_sub)
        use_fp16 = (self._default_tensor_dtype == torch.float16)
        if fp16 is not None:
            use_fp16 = bool(fp16) and (self._device == "cuda")
        target_dtype = torch.float16 if (use_fp16 and self._device == "cuda") else torch.float32

        hs = getattr(self.model.config, "hidden_size", hs_fallback)
        zero_vec = lambda: torch.zeros(hs, dtype=target_dtype, device=self._device)

        primary_plan: Dict[str, Tuple[str, float]] = {}
        sub_plan: Dict[str, Dict[str, Tuple[str, float]]] = {}
        ctx_plan: Dict[str, Dict[str, Tuple[str, float]]] = {}
        hier_plan: Dict[str, Tuple[str, float]] = {}
        prime_candidates: List[str] = []

        for eid, vals in intensity_results.items():
            score = float(vals.get("modified_score", vals.get("score", vals.get("weight", 0.0))))
            edata = self.emotions_data.get(eid, {})

            prim_ctx = self._create_primary_emotion_context(edata)
            if prim_ctx:
                primary_plan[eid] = (prim_ctx, score)
                prime_candidates.append(prim_ctx)

            sub_entries: Dict[str, Tuple[str, float]] = {}
            for sn, sdata in list((edata.get("sub_emotions", {}) or {}).items())[:top_k_sub]:
                sub_ctx = self._create_sub_emotion_context(sdata)
                if sub_ctx:
                    sub_entries[sn] = (sub_ctx, score)
                    prime_candidates.append(sub_ctx)
            if sub_entries:
                sub_plan[eid] = sub_entries

            situations = (edata.get("context_patterns", {}) or {}).get("situations", {}) or {}
            ctx_entries: Dict[str, Tuple[str, float]] = {}
            for name, sdata in list(situations.items())[:top_k_context]:
                ctx = self._create_contextual_situation_context(name, sdata)
                if ctx:
                    ctx_entries[name] = (ctx, score)
                    prime_candidates.append(ctx)
            if ctx_entries:
                ctx_plan[eid] = ctx_entries

            hier_ctx = self._create_hierarchical_context(eid, edata.get("sub_emotions", {}) or {})
            if hier_ctx:
                hier_plan[eid] = (hier_ctx, score)
                prime_candidates.append(hier_ctx)

        self._prime_context_embeddings(prime_candidates)

        primary_tensors: Dict[str, torch.Tensor] = {}
        for eid, (ctx, score) in primary_plan.items():
            primary_tensors[eid] = self._cached_weighted_embedding(ctx, score).to(target_dtype)

        sub_tensors: Dict[str, Dict[str, torch.Tensor]] = {}
        for eid, entries in sub_plan.items():
            tmp: Dict[str, torch.Tensor] = {}
            for sn, (ctx, score) in entries.items():
                tmp[sn] = self._cached_weighted_embedding(ctx, score).to(target_dtype)
            sub_tensors[eid] = tmp

        ctx_tensors: Dict[str, Dict[str, torch.Tensor]] = {}
        for eid, entries in ctx_plan.items():
            tmp: Dict[str, torch.Tensor] = {}
            for name, (ctx, score) in entries.items():
                tmp[name] = self._cached_weighted_embedding(ctx, score).to(target_dtype)
            ctx_tensors[eid] = tmp

        hier_tensors: Dict[str, torch.Tensor] = {}
        for eid, (ctx, score) in hier_plan.items():
            hier_tensors[eid] = self._cached_weighted_embedding(ctx, score).to(target_dtype)

        combined_tensors: Dict[str, torch.Tensor] = {}
        for eid in intensity_results.keys():
            prim_vec = primary_tensors.get(eid, zero_vec())
            sub_mean = self._safe_stack_mean(list(sub_tensors.get(eid, {}).values())).to(target_dtype)
            ctx_mean = self._safe_stack_mean(list(ctx_tensors.get(eid, {}).values())).to(target_dtype)
            hier_vec = hier_tensors.get(eid, zero_vec())
            stack = torch.stack([prim_vec, sub_mean, ctx_mean, hier_vec], dim=0)
            combined_tensors[eid] = torch.mean(stack, dim=0)

        return {
            "primary": primary_tensors,
            "sub": sub_tensors,
            "contextual": ctx_tensors,
            "hierarchical": hier_tensors,
            "combined": combined_tensors,
            "hidden_size": hs,
        }

    def _generate_weighted_embedding(self, context: str, intensity: float) -> torch.Tensor:
        if not self._embeddings_enabled or not context.strip():
            hs = getattr(self.model.config, "hidden_size", 768)
            return torch.zeros(hs, dtype=self._default_tensor_dtype, device=self._device)
        tok_kwargs = {
            "return_tensors": "pt",
            "padding": True,
            "truncation": True,
            "max_length": self._tokenizer_max_length,
        }
        if self._pad_to_multiple > 0:
            tok_kwargs["pad_to_multiple_of"] = self._pad_to_multiple
        inputs = self.tokenizer(context, **tok_kwargs)
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            with self._autocast_ctx():
                outputs = self.model(**inputs).last_hidden_state
        att = torch.tensor(
            [[intensity]],
            dtype=self._default_tensor_dtype,
            device=self._device,
        ).unsqueeze(-1)
        sent = torch.mean(outputs * att, dim=1).squeeze(0).to(self._default_tensor_dtype)
        return sent

    def _create_primary_emotion_context(self, emotion_data: Dict[str, Any]) -> str:
        md = emotion_data.get("metadata", {}) or {}
        prof = emotion_data.get("emotion_profile", {}) or {}
        kws = ", ".join((prof.get("core_keywords", []) or [])[:5])
        return f"Primary Emotion: {md.get('primary_category', '')}, Complexity: {md.get('emotion_complexity', 'basic')}, Keywords: {kws}"

    def _create_sub_emotion_context(self, sub_data: Dict[str, Any]) -> str:
        md = sub_data.get("metadata", {}) or {}
        prof = sub_data.get("emotion_profile", {}) or {}
        rel = prof.get("related_emotions", {}) or {}
        pos = ", ".join((rel.get("positive", []) or [])[:3])
        neg = ", ".join((rel.get("negative", []) or [])[:3])
        return f"Sub-emotion: {md.get('sub_category', '')}, ID: {md.get('emotion_id', '')}, Related Positive: {pos}, Related Negative: {neg}"

    def _create_hierarchical_context(self, emotion_id: str, sub_emotions: Dict[str, Any]) -> str:
        s = [f"Primary Emotion Hierarchy: {emotion_id}"]
        for sn, sd in (sub_emotions or {}).items():
            ex = ((sd.get("emotion_profile", {}) or {}).get("intensity_levels", {}) or {}).get("intensity_examples", {}) or {}
            if ex.get("high"):
                s.append(f"- Sub-emotion {sn} has high intensity examples.")
        return "\n".join(s)

    def _create_contextual_situation_context(self, situation_name: str, situation_data: Dict[str, Any]) -> str:
        return f"Situation: {situation_name}. Description: {situation_data.get('description', '')}. Intensity: {situation_data.get('intensity', 'medium')}"

    def _populate_embeddings(self, embeddings: Dict, intensity_results: Dict) -> None:
        for eid, vals in intensity_results.items():
            iv = float(vals.get("modified_score", 0.0))
            ed = self.emotions_data.get(eid, {})
            subs = ed.get("sub_emotions", {}) or {}
            primary_context = self._create_primary_emotion_context(ed)
            embeddings.setdefault("primary", {})[eid] = self._generate_weighted_embedding(primary_context, iv)
            sub_map = {sn: self._generate_weighted_embedding(self._create_sub_emotion_context(sd), iv) for sn, sd in subs.items()}
            embeddings.setdefault("sub", {})[eid] = sub_map
            ctx_map = {
                nm: self._generate_weighted_embedding(self._create_contextual_situation_context(nm, sd), iv)
                for nm, sd in (ed.get("context_patterns", {}).get("situations", {}) or {}).items()
            }
            embeddings.setdefault("contextual", {})[eid] = ctx_map
            hier_ctx = self._create_hierarchical_context(eid, subs)
            embeddings.setdefault("hierarchical", {})[eid] = self._generate_weighted_embedding(hier_ctx, iv)

    def _combine_embeddings(self, embeddings: Dict) -> None:
        for eid, p in embeddings.get("primary", {}).items():
            sub_t = list(embeddings.get("sub", {}).get(eid, {}).values())
            sub_mean = torch.mean(torch.stack(sub_t), dim=0) if sub_t else torch.zeros_like(p)
            ctx_t = list(embeddings.get("contextual", {}).get(eid, {}).values())
            ctx_mean = torch.mean(torch.stack(ctx_t), dim=0) if ctx_t else torch.zeros_like(p)
            hier = embeddings.get("hierarchical", {}).get(eid, torch.zeros_like(p))
            embeddings.setdefault("combined", {})[eid] = torch.mean(torch.stack([p, sub_mean, ctx_mean, hier], dim=0), dim=0)


# =============================================================================
# Independent Function
# =============================================================================
# 경량 캐시(경로별 Analyzer 1회만 초기화해서 재사용)
_ANALYZER_CACHE: Dict[str, EmotionIntensityAnalyzer] = {}
_ANALYZER_LOCK = RLock()

def _norm_path(p: str) -> str:
    try:
        return os.path.normpath(os.path.abspath(p))
    except Exception:
        return p

def get_intensity_analyzer(emotions_data_path: str, *, use_cache: bool = True,
                           logger: Optional[logging.Logger] = None) -> EmotionIntensityAnalyzer:
    """
    EmotionIntensityAnalyzer 인스턴스를 반환합니다.
    - emotions_data_path: EMOTIONS.JSON 경로(상대/절대 모두 허용)
    - use_cache=True: 동일 경로는 재사용(모델 재로딩 방지, GPU 메모리 절감)
    """
    path_key = _norm_path(emotions_data_path)
    if not use_cache:
        if logger: logger.info("[intensity] caching disabled → new analyzer")
        an = EmotionIntensityAnalyzer(path_key)
        an.warmup()
        return an

    with _ANALYZER_LOCK:
        inst = _ANALYZER_CACHE.get(path_key)
        if inst is not None:
            return inst
        if logger: logger.info(f"[intensity] create analyzer: {path_key}")
        inst = EmotionIntensityAnalyzer(path_key)
        inst.warmup()
        _ANALYZER_CACHE[path_key] = inst
        return inst

def clear_intensity_cache(emotions_data_path: Optional[str] = None) -> None:
    """
    Analyzer 캐시 비우기(메모리 회수/재로딩 강제).
    - emotions_data_path 없으면 전체 캐시를 삭제합니다.
    """
    with _ANALYZER_LOCK:
        if emotions_data_path:
            _ANALYZER_CACHE.pop(_norm_path(emotions_data_path), None)
        else:
            _ANALYZER_CACHE.clear()


def _fallback_minimal_intensity_payload(conf: float = 0.05, reason: str = "error") -> Dict[str, Any]:
    prim = ["희", "노", "애", "락"]
    return {
        e: {
            "intensity_score": 0.0,
            "modified_score": 0.0,
            "level": "low",
            "confidence": float(conf),
            "flags": [reason],
        }
        for e in prim
    }


def analyze_intensity(
    text: str,
    emotions_data_path: str,
    use_cache: bool = True,
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """감정 강도 분석 함수(대표 4감정 스코어 + 보정점수/레벨 포함)."""
    try:
        if not isinstance(text, str) or not text.strip():
            return _fallback_minimal_intensity_payload(0.05, "empty_input")
        analyzer = get_intensity_analyzer(emotions_data_path, use_cache=use_cache, logger=logger)
        results = analyzer.analyze_emotion_intensity(text)
        if logger:
            logger.info(f"[intensity] 완료: {len(results)}개 감정")
        return results
    except Exception as e:
        if logger:
            logger.error(f"[intensity] 오류: {e}", exc_info=True)
        return _fallback_minimal_intensity_payload(0.05, "exception")


def analyze_intensity_transitions(
    text: str,
    emotions_data_path: str,
    min_sentences: int = 2,
    use_cache: bool = True,
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """감정 전이 분석 함수(지배 감정 top-1 전이/통계/전개 패턴 요약)."""
    try:
        if not isinstance(text, str) or not text.strip():
            return {"error": "empty_input", "temporal_patterns": {}, "transitions": [], "sub_transitions": [], "statistics": {}}
        analyzer = get_intensity_analyzer(emotions_data_path, use_cache=use_cache, logger=logger)
        sents = analyzer._split_sentences_ko(text)
        if len(sents) < int(min_sentences):
            return {"error": "need_more_sentences", "temporal_patterns": {}, "transitions": [], "sub_transitions": [], "statistics": {}}
        results = analyzer.analyze_intensity_transitions(text)
        if logger:
            raw_n = len(results.get("transitions", []) or [])
            logger.info(f"[transitions] 완료: {raw_n} 전이")
        return results
    except Exception as e:
        if logger:
            logger.error(f"[transitions] 오류: {e}", exc_info=True)
        return {"error": "exception", "temporal_patterns": {}, "transitions": [], "sub_transitions": [], "statistics": {}}

def analyze_temporal_patterns(
    emotion_sequence: List[Dict[str, float]],
    emotions_data_path: str,
    use_cache: bool = True,
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    이미 산출된 시퀀스에 대해 시계열 패턴만 분석하는 경량 래퍼.
    emotion_sequence 형식: [{ "희":..., "노":...}, ...] 또는 [{ "emotions": {...}}, ...]
    """
    try:
        analyzer = get_intensity_analyzer(emotions_data_path, use_cache=use_cache, logger=logger)
        return analyzer.transformer.analyze_temporal_patterns(emotion_sequence)
    except Exception as e:
        if logger:
            logger.error(f"[temporal] 오류: {e}", exc_info=True)
        return {}



def analyze_situational_intensity(
    text: str,
    emotions_data_path: str,
    threshold: float = 0.3,
    use_cache: bool = True,
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    상황별 감정 강도 분석 함수.
    - threshold: final_intensity_weight 하한(기본 0.3)
    - 내부 Top-K/상한은 환경변수(SITUATION_TOPK_PER_PRIMARY, SITUATION_GLOBAL_CAP, SITUATION_FINAL_WEIGHT_MIN)로 제어
    """
    try:
        if not isinstance(text, str) or not text.strip():
            return {}
        analyzer = get_intensity_analyzer(emotions_data_path, use_cache=use_cache, logger=logger)
        results = analyzer.analyze_situational_intensity(text) or {}
        filtered = {k: v for k, v in results.items() if float(v.get("final_intensity_weight", 0.0)) >= float(threshold)}
        if logger:
            logger.info(f"[situation] 완료: {len(filtered)} 상황 (threshold={threshold})")
        return filtered
    except Exception as e:
        if logger:
            logger.error(f"[situation] 오류: {e}", exc_info=True)
        return {}


def generate_intensity_embeddings(
    text: str,
    emotions_data_path: str,
    embedding_types: Optional[List[str]] = None,
    top_k_context: int = 5,
    top_k_sub: int = 3,
    use_cache: bool = True,
    logger: Optional[logging.Logger] = None,
    enable_embeddings: Optional[bool] = None
) -> Dict[str, Any]:
    """
    감정 강도 기반 임베딩 생성 함수.
    - embedding_types: {'primary','sub','contextual','hierarchical','combined'} 중 선택(None=전부)
    - 반환 시 Tensor는 list로 변환되어 JSON 직렬화 가능
    - enable_embeddings: 환경변수와 무관하게 임베딩 사용 여부를 토글(None=기존 설정 유지)
    """
    try:
        if not isinstance(text, str) or not text.strip():
            return {"intensity_results": _fallback_minimal_intensity_payload(0.05, "empty_input"), "embeddings": {}}
        valid_types = {'primary', 'sub', 'contextual', 'hierarchical', 'combined'}

        analyzer = get_intensity_analyzer(emotions_data_path, use_cache=use_cache, logger=logger)
        # 임베딩 온/오프 런타임 제어(옵션)
        if enable_embeddings is not None:
            if enable_embeddings:
                analyzer._allow_embeddings = True  # type: ignore[attr-defined]
                if not getattr(analyzer, "_embeddings_enabled", False):
                    try:
                        analyzer._load_pretrained_model()  # type: ignore[attr-defined]
                    except Exception:
                        pass
            else:
                analyzer._allow_embeddings = False  # type: ignore[attr-defined]
                setattr(analyzer, "_embeddings_enabled", False)

        intensity_results = analyzer.analyze_emotion_intensity(text)
        embeddings = analyzer.generate_embeddings(
            intensity_results, top_k_context=top_k_context, top_k_sub=top_k_sub
        )
        if embedding_types:
            embeddings = {k: v for k, v in embeddings.items() if k in valid_types and k in set(embedding_types)}

        # Tensor → list 변환
        try:
            embeddings_json = _convert_tensors_to_lists(embeddings)
        except Exception:
            embeddings_json = embeddings  # 혹시 모를 예외 시 원본 유지

        if logger:
            logger.info(f"[embeddings] 완료: keys={list(embeddings_json.keys())}")
        return {"intensity_results": intensity_results, "embeddings": embeddings_json}
    except Exception as e:
        if logger:
            logger.error(f"[embeddings] 오류: {e}", exc_info=True)
        return {"intensity_results": _fallback_minimal_intensity_payload(0.05, "exception"), "embeddings": {}}

# ---------------------------------------------------------------------
# Confidence computation utilities (normalized confidence in [0,1])
# ---------------------------------------------------------------------
def _norm01(x):
    try:
        return max(0.0, min(1.0, float(x)))
    except Exception:
        return 0.0

def _extract_conf_intensity(res: dict) -> float:
    """
    외부 모듈 결과(dict)에서 confidence가 유효하면 그대로 사용하고,
    없거나 비정상일 때 intensity 분포/곡선을 이용해 똑똑한 폴백으로 계산합니다.
    """
    try:
        c = (res.get("confidence") if isinstance(res, dict) else None)
        if isinstance(c, (int, float)) and 0.0 <= float(c) <= 1.0:
            return float(c)

        inten = (res.get("intensity") or {}) if isinstance(res, dict) else {}
        # intensity_distribution 기반 폴백 계산
        import math as _math  # 지역 import는 안전
        dist = inten.get("intensity_distribution") or {}
        vals = [v for v in dist.values() if isinstance(v, (int, float))]
        if vals:
            s = sum(max(0.0, float(v)) for v in vals) or 1.0
            probs = [max(1e-9, float(v) / s) for v in vals]
            p_max = max(probs)
            p_sorted = sorted(probs, reverse=True)
            p_second = p_sorted[1] if len(p_sorted) > 1 else 0.0
            H = -sum(p * _math.log(p) for p in probs)
            Hn = H / _math.log(len(probs))
            contrast = (p_max - p_second) / max(p_max, 1e-9)
            base = 0.6 * p_max + 0.3 * (1.0 - Hn) + 0.1 * contrast
        else:
            base = 0.0

        # intensity_curve(선택) 기반 보정
        vs = []
        for pt in inten.get("intensity_curve") or []:
            v = pt.get("i") if isinstance(pt, dict) else pt
            try:
                vs.append(float(v))
            except Exception:
                pass
        if vs:
            M = max(1.0, max(vs))
            vs = [min(1.0, max(0.0, v / M)) for v in vs]
            mid = sorted(vs)[len(vs) // 2]
            base += 0.1 * (max(vs) - mid)

        return float(max(0.0, min(1.0, base)))
    except Exception:
        return 0.0



def _compute_intensity_confidence(intensity: dict) -> Tuple[float, dict]:
    dist = intensity.get("intensity_distribution") or {}
    vals = [v for v in dist.values() if isinstance(v, (int, float))]
    # 1) 분포 정규화 (합=1, 0~1 클램프)
    if vals:
        s = sum(max(0.0, float(v)) for v in vals) or 1.0
        probs = [max(1e-9, float(v) / s) for v in vals]
    else:
        probs = []

    if probs:
        p_max = max(probs)
        p_sorted = sorted(probs, reverse=True)
        p_second = p_sorted[1] if len(p_sorted) > 1 else 0.0
        H = -sum(p * math.log(p) for p in probs)
        Hn = H / math.log(len(probs))
        contrast = (p_max - p_second) / max(p_max, 1e-9)
        base = 0.6 * p_max + 0.3 * (1.0 - Hn) + 0.1 * contrast
    else:
        p_max = p_second = Hn = contrast = 0.0
        base = 0.0

    # 2) 곡선 샤프니스로 미세 보정
    curve = intensity.get("intensity_curve") or []
    vs = []
    for pt in curve:
        if isinstance(pt, dict):
            v = pt.get("i") or pt.get("value") or pt.get("score")
        else:
            v = pt
        try:
            vs.append(float(v))
        except Exception:
            pass
    if vs:
        M = max(1.0, max(vs))  # 1을 초과하면 스케일에 맞춰 0~1로 재정규화
        vs = [_norm01(v / M) for v in vs]
        vs_sorted = sorted(vs)
        mid = vs_sorted[len(vs) // 2]
        amp = max(vs) - mid
        base += 0.1 * amp  # 미세 가산
    else:
        amp = 0.0

    conf = max(0.0, min(1.0, base))
    return conf, {
        "p_max": round(p_max, 4),
        "entropy_norm": round(Hn, 4),
        "contrast": round(contrast, 4),
        "curve_amp": round(amp if vs else 0.0, 4),
    }



def run_intensity_analysis(
    text: str,
    emotions_data_path: str,
    analysis_config: Optional[Dict[str, Any]] = None,
    use_cache: bool = True,
    logger: Optional[logging.Logger] = None,
    enable_embeddings: Optional[bool] = None
) -> Dict[str, Any]:
    """
    텍스트 전반의 감정 강도를 분석한다(문장·상황·전이·임베딩 포함).
    반환 포맷:
      {
        "timestamp": ...
        "config": {...}
        "emotion_intensity": {...}
        "sub_emotions": {...}
        "situational_intensity": {...}
        "transitions": {...}
        "embeddings": {...}
      }
    """
    try:
        if not isinstance(text, str) or not text.strip():
            return {
                "timestamp": datetime.now().isoformat(),
                "config": (analysis_config or {}),
                "emotion_intensity": _fallback_minimal_intensity_payload(0.05, "empty_input"),
                "sub_emotions": {},
                "situational_intensity": {},
                "transitions": {"temporal_patterns": {}, "transitions": [], "sub_transitions": [], "statistics": {}},
                "embeddings": {},
                "confidence": 0.0,
            }

        default_config = {
            "analyze_situation": True,
            "analyze_transitions": True,
            "generate_embed": False,
            "situation_threshold": 0.3,
            "min_sentences": 2,
            "embedding_types": None,
            "top_k_context": 5,
            "top_k_sub": 3,
        }
        cfg = {**default_config, **(analysis_config or {})}

        analyzer = get_intensity_analyzer(emotions_data_path, use_cache=use_cache, logger=logger)
        # 임베딩 사용 여부 런타임 제어(옵션)
        if enable_embeddings is not None:
            if enable_embeddings:
                analyzer._allow_embeddings = True  # type: ignore[attr-defined]
                if not getattr(analyzer, "_embeddings_enabled", False):
                    try:
                        analyzer._load_pretrained_model()  # type: ignore[attr-defined]
                    except Exception:
                        pass
            else:
                analyzer._allow_embeddings = False  # type: ignore[attr-defined]
                setattr(analyzer, "_embeddings_enabled", False)

        result: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "config": cfg,
        }

        # 성능 모니터링 및 진행 상황 표시
        import time
        start_time = time.time()
        
        print("=== Intensity Analyzer 성능 분석 ===")
        print(f"입력 텍스트: {text[:50]}...")
        print("단계별 분석 진행 중...")
        
        # 1) 대표 4감정 강도
        print("1/5 대표 감정 강도 분석 중...")
        t1 = time.time()
        emo = analyzer.analyze_emotion_intensity(text)
        result["emotion_intensity"] = emo
        step1_time = time.time() - t1
        print(f"   완료 ({step1_time:.2f}초)")

        # 2) 세부감정 점수
        print("2/5 세부 감정 점수 분석 중...")
        t1 = time.time()
        result["sub_emotions"] = analyzer._analyze_sub_emotions(text)
        step2_time = time.time() - t1
        print(f"   완료 ({step2_time:.2f}초)")

        # 3) 상황 강도(필터링 적용)
        if cfg["analyze_situation"]:
            print("3/5 상황 강도 분석 중...")
            t1 = time.time()
            sit = analyzer.analyze_situational_intensity(text) or {}
            result["situational_intensity"] = {
                k: v for k, v in sit.items()
                if float(v.get("final_intensity_weight", 0.0)) >= float(cfg["situation_threshold"])
            }
            step3_time = time.time() - t1
            print(f"   완료 ({step3_time:.2f}초)")
        else:
            step3_time = 0.0

        # 4) 전이
        if cfg["analyze_transitions"]:
            print("4/5 감정 전이 분석 중...")
            t1 = time.time()
            sents = analyzer._split_sentences_ko(text)
            if len(sents) >= int(cfg["min_sentences"]):
                tr_all = analyzer.analyze_intensity_transitions(text) or {}
                # compact 이전 원시 리스트 길이 기반 전이 개수 산정(과대 집계 방지)
                _n_trans = len(tr_all.get("transitions", []) or [])
                _n_sub   = len(tr_all.get("sub_transitions", []) or [])
                tr_all["counts"] = {"transitions": _n_trans, "sub_transitions": _n_sub}
                result["transitions"] = tr_all
            step4_time = time.time() - t1
            print(f"   완료 ({step4_time:.2f}초)")
        else:
            step4_time = 0.0

        # 5) 임베딩
        if cfg["generate_embed"]:
            print("5/5 임베딩 생성 중...")
            t1 = time.time()
            emb = analyzer.generate_embeddings(
                emo,
                top_k_context=int(cfg["top_k_context"]),
                top_k_sub=int(cfg["top_k_sub"])
            )
            if cfg["embedding_types"]:
                emb = {k: v for k, v in emb.items() if k in set(cfg["embedding_types"])}
            try:
                emb = _convert_tensors_to_lists(emb)
            except Exception:
                pass
            result["embeddings"] = emb
            step5_time = time.time() - t1
            print(f"   완료 ({step5_time:.2f}초)")
        else:
            step5_time = 0.0

        # ---- normalized confidence injection (0~1) ----
        try:
            intensity_dist = {
                str(k): float(v.get("modified_score", v.get("intensity_score", 0.0)))
                for k, v in (emo or {}).items()
                if isinstance(v, dict)
            }
        except Exception:
            intensity_dist = {}
        intensity_curve: List[float] = []
        intensity_payload = {
            "intensity": {
                "intensity_distribution": intensity_dist,
                "intensity_curve": intensity_curve,
            }
        }
        conf = _extract_conf_intensity(intensity_payload)
        _, conf_components = _compute_intensity_confidence(
            {
                "intensity_distribution": intensity_dist,
                "intensity_curve": intensity_curve,
            }
        )
        normalized_dist: Dict[str, float] = {}
        if intensity_dist:
            dist_sum = sum(max(0.0, float(x)) for x in intensity_dist.values()) or 1.0
            normalized_dist = {
                key: (max(0.0, float(val)) / dist_sum) for key, val in intensity_dist.items()
            }

        # 성능 분석 결과 출력
        total_time = time.time() - start_time
        print(f"\n=== Intensity Analyzer 성능 분석 결과 ===")
        print(f"총 처리 시간: {total_time:.2f}초")
        print(f"\n단계별 처리 시간:")
        print(f"  1. 대표 감정 강도: {step1_time:.2f}초")
        print(f"  2. 세부 감정 점수: {step2_time:.2f}초")
        print(f"  3. 상황 강도: {step3_time:.2f}초")
        print(f"  4. 감정 전이: {step4_time:.2f}초")
        print(f"  5. 임베딩 생성: {step5_time:.2f}초")
        
        # 결과 요약 출력
        print(f"\n=== 분석 결과 요약 ===")
        print(f"입력 텍스트: {text}")
        print(f"감정 강도:")
        for emotion, data in emo.items():
            if isinstance(data, dict):
                score = data.get("modified_score", 0.0)
                print(f"  {emotion}: {score:.3f}")
        print(f"신뢰도: {conf:.3f}")

        result["confidence"] = conf
        result.setdefault("diagnostics", {})["confidence_components"] = conf_components
        if normalized_dist:
            result["intensity_distribution_normalized"] = normalized_dist

        if logger:
            logger.info("[run] 통합 감정 강도 분석 완료")
        return result

    except Exception as e:
        if logger:
            logger.error(f"[run] 오류: {e}", exc_info=True)
        return {"timestamp": datetime.now().isoformat(), "config": (analysis_config or {}),
                "emotion_intensity": _fallback_minimal_intensity_payload(0.05, "exception"),
                "sub_emotions": {}, "situational_intensity": {},
                "transitions": {"temporal_patterns": {}, "transitions": [], "sub_transitions": [], "statistics": {}},
                "embeddings": {}, "confidence": 0.0}


def run_embedding_generation(payload: Any, config: Dict[str, Any]) -> Dict[str, Any]:
    """DataUtils 호환 래퍼: 임베딩 생성"""
    # payload 객체에서 text 추출
    text = getattr(payload, "text", "") if hasattr(payload, "text") else ""
    if not text and isinstance(payload, dict):
        text = payload.get("text", "")
        
    # EMOTIONS_JSON_PATH는 config나 환경변수에서 가져옴
    import os
    path = os.getenv("EMOTIONS_JSON_PATH", "src/EMOTIONS.json")
    
    return generate_intensity_embeddings(
        text=text,
        emotions_data_path=path,
        enable_embeddings=True # 강제 활성화
    )




# =============================================================================
# Main
# =============================================================================
def _get_test_cases():
    """
    반드시 [{"category": "...", "texts": ["...", ...]}] 형태를 반환.
    - TEST_CASES_PATH(환경변수)가 있으면 그 JSON을 읽어 정규화
    - 그렇지 않으면 내장 샘플 반환
    """
    import os, json

    def _normalize(raw):
        if isinstance(raw, dict):
            return [{"category": str(k), "texts": [str(x) for x in v]} for k, v in raw.items() if isinstance(v, (list, tuple))]
        if isinstance(raw, list) and all(isinstance(x, str) for x in raw):
            return [{"category": "기본", "texts": [str(x) for x in raw]}]
        if isinstance(raw, list):
            out = []
            for it in raw:
                if isinstance(it, dict):
                    cat = str(it.get("category", "무명 카테고리"))
                    tx = it.get("texts", [])
                    if isinstance(tx, (list, tuple)):
                        tx = [str(x) for x in tx]
                    else:
                        tx = [str(tx)]
                    out.append({"category": cat, "texts": tx})
                elif isinstance(it, (list, tuple)) and len(it) == 2:
                    cat, tx = it
                    cat = str(cat)
                    if isinstance(tx, (list, tuple)):
                        tx = [str(x) for x in tx]
                    else:
                        tx = [str(tx)]
                    out.append({"category": cat, "texts": tx})
                else:
                    out.append({"category": "기본", "texts": [str(it)]})
            return out
        return [{"category": "기본", "texts": [str(raw)]}]

    path = os.environ.get("TEST_CASES_PATH")
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        norm = _normalize(data)
        if norm:
            return norm

    return _normalize([
        {"category": "단일 감정", "texts": [
            "오늘 시험에서 만점을 받아서 너무나도 기쁘고 행복합니다.",
            "친구의 배신에 너무나도 화가 나고 분노가 치밉니다.",
            "사랑하는 가족을 잃어서 눈물이 멈추지 않습니다."
        ]},
        {"category": "감정 전이", "texts": [
            "처음에는 화가 났지만, 이야기를 듣고 보니 이해가 되었습니다.",
            "슬픔에 잠겨있다가 친구들이 위로해주어서 조금은 마음이 놓였습니다.",
            "기쁨과 설렘이 가득했는데, 갑자기 불안감이 몰려왔습니다."
        ]},
        {"category": "복합 감정", "texts": [
            "졸업식날 이별의 아쉬움과 새로운 시작에 대한 기대감이 교차합니다.",
            "합격 소식을 들었을 때 기쁨과 동시에 앞으로에 대한 부담감도 느껴졌습니다.",
            "오랜 친구를 만나니 그리움과 반가움, 그리고 미안함이 복잡하게 얽혔습니다."
        ]},
        {"category": "강도 변화", "texts": [
            "조금 짜증났다가, 매우 화가 났다가, 결국에는 폭발할 것 같은 분노가 치밀었습니다.",
            "살짝 기분이 좋아지다가, 점점 더 기뻐지더니, 마침내 환희에 차올랐습니다.",
            "약간의 우울함이 점점 깊어져서 결국 깊은 절망감에 빠져들었습니다."
        ]},
    ])

def _convert_tensors_to_lists(data: Any) -> Any:
    if isinstance(data, dict):
        return {k: _convert_tensors_to_lists(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_convert_tensors_to_lists(v) for v in data]
    if isinstance(data, torch.Tensor):
        return data.tolist()
    return data


def _env_snapshot() -> Dict[str, Any]:
    import sys
    snap = {
        "python": sys.version.split()[0],
        "numpy": getattr(np, "__version__", None),
        "torch": getattr(torch, "__version__", None),
        "transformers": None,
        "cuda_available": bool(torch.cuda.is_available()),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "gpu_name": None,
    }
    try:
        import transformers  # noqa
        snap["transformers"] = getattr(transformers, "__version__", None)
    except Exception:
        pass
    if snap["cuda_available"]:
        try:
            snap["gpu_name"] = torch.cuda.get_device_name(0)
        except Exception:
            pass
    return snap


# =============================================================================
# Main
# =============================================================================
def main():
    """
    실행/저장 옵션이 유연한 러너:
    - 분석기(모델)는 한 번만 초기화해서 전체 케이스에 재사용 → 속도 개선
    - 저장 형식 선택: --output-format {json, json.gz, both}
    - 요약만 저장: --summary-only (summary.json + preview.json)
    - 카테고리별 분할 저장: --split-per-category
    - 결과/로그는 항상 현재 파일 폴더의 logs/ 아래에 저장
    """
    import argparse, logging, os, sys, json, gzip, getpass, re
    from datetime import datetime
    from logging.handlers import RotatingFileHandler
    import numpy as np

    # ---------- CLI ----------
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-embeddings", action="store_true", default=True)
    parser.add_argument("--embed-topk-sub", type=int, default=2)
    parser.add_argument("--embed-topk-ctx", type=int, default=3)
    parser.add_argument("--max-transitions", type=int, default=15)
    parser.add_argument("--texts-per-category", type=int, default=0)
    parser.add_argument("--compact-json", action="store_true", default=True)
    parser.add_argument("--output-format", choices=["json", "json.gz", "both"], default="json.gz",
                        help="출력 형식 선택 (기본: json.gz)")
    parser.add_argument("--summary-only", action="store_true", default=False,
                        help="요약/프리뷰만 저장(전체 test_results는 저장하지 않음)")
    parser.add_argument("--preview-limit", type=int, default=30,
                        help="프리뷰 파일에 포함할 test_results 개수(기본 30)")
    parser.add_argument("--split-per-category", action="store_true", default=False,
                        help="카테고리별로 개별 파일로 분할 저장")
    args = parser.parse_args()

    # ---------- 경로 (통합 로그 관리자 사용 - 날짜별 폴더) ----------
    try:
        from log_manager import get_log_manager
        log_manager = get_log_manager()
        logs_dir = str(log_manager.get_log_dir("emotion_analysis", use_date_folder=True))
    except ImportError:
        # 폴백: 기존 방식 (날짜별 폴더 추가)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        base_logs_dir = os.path.join(base_dir, "logs")
        today = datetime.now().strftime("%Y%m%d")
        logs_dir = os.path.join(base_logs_dir, today)
        os.makedirs(logs_dir, exist_ok=True)

    module_name = os.path.splitext(os.path.basename(__file__))[0]  # intensity_analyzer
    emotions_data_path = os.environ.get("EMOTIONS_JSON_PATH", os.path.join(base_dir, "EMOTIONS.JSON"))

    # ---------- 로거(파일 회전 + 콘솔) ----------
    logger = logging.getLogger("emotion")
    logger.setLevel(logging.INFO)
    logger.handlers = []  # 중복 방지

    file_handler = RotatingFileHandler(
        filename=os.path.join(logs_dir, f"{module_name}.log"),
        maxBytes=5 * 1024 * 1024,  # 5MB
        backupCount=3,
        encoding="utf-8",
    )
    stream_handler = logging.StreamHandler()
    fmt = logging.Formatter("[%(asctime)s|%(levelname)s|%(name)s] %(message)s")
    file_handler.setFormatter(fmt)
    stream_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.propagate = False

    logger.info("=== 감정 강도 분석 테스트 시작 ===")

    # ---------- 헬퍼 ----------
    def _env_snapshot_safe():
        try:
            return _env_snapshot()  # 원본에 있으면 사용
        except Exception:
            import platform
            return {"python": platform.python_version(), "cwd": os.getcwd()}

    def _score_of(v):
        try:
            if isinstance(v, dict):
                return float(v.get("modified_score") or v.get("intensity_score") or v.get("score") or 0.0)
            return float(v)
        except Exception:
            return 0.0

    def _limit_transitions(trn: dict, limit: int) -> dict:
        if not isinstance(trn, dict) or limit is None or limit < 0:
            return trn
        out = dict(trn)
        for k in ("transitions", "sub_transitions"):
            if isinstance(out.get(k), list):
                out[k] = out[k][:limit]
        return out

    def _transition_count_from(trn: dict, compact: bool) -> int:
        if not isinstance(trn, dict):
            return 0
        if not compact and isinstance(trn.get("transitions"), list):
            return int(len(trn["transitions"]))
        st = trn.get("statistics", {})
        if isinstance(st.get("significant_changes"), list):
            return int(len(st["significant_changes"]))
        if isinstance(st.get("transitions"), list):
            return int(len(st["transitions"]))
        if isinstance(st.get("transitions"), dict):
            val = st["transitions"]
            try:
                return int(sum(len(v) if isinstance(v, list) else 1 for v in val.values()))
            except Exception:
                return int(len(val))
        return 0

    def _normalize_test_cases(raw):
        # dict → [{"category":k, "texts":[...]}]
        if isinstance(raw, dict):
            return [{"category": str(k), "texts": [str(x) for x in v]}
                    for k, v in raw.items() if isinstance(v, (list, tuple))]
        # list[str] → 한 카테고리
        if isinstance(raw, list) and all(isinstance(x, str) for x in raw):
            return [{"category": "기본", "texts": [str(x) for x in raw]}]
        # list[...] 혼합
        if isinstance(raw, list):
            out = []
            for it in raw:
                if isinstance(it, dict):
                    cat = str(it.get("category", "무명 카테고리"))
                    tx = it.get("texts", [])
                    tx = [str(x) for x in tx] if isinstance(tx, (list, tuple)) else [str(tx)]
                    out.append({"category": cat, "texts": tx})
                elif isinstance(it, (list, tuple)) and len(it) == 2:
                    cat, tx = it
                    tx = [str(x) for x in tx] if isinstance(tx, (list, tuple)) else [str(tx)]
                    out.append({"category": str(cat), "texts": tx})
                else:
                    out.append({"category": "기본", "texts": [str(it)]})
            return out
        # 기타 → 단일 카테고리
        return [{"category": "기본", "texts": [str(raw)]}]

    def _load_cases():
        path = os.environ.get("TEST_CASES_PATH")
        if path and os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return _normalize_test_cases(json.load(f))
            except Exception as e:
                logger.warning(f"TEST_CASES_PATH 로드 실패: {e} → 기본값 사용")
        # 기본 샘플
        return _normalize_test_cases([
            {"category": "단일 감정", "texts": [
                "오늘 시험에서 만점을 받아서 너무나도 기쁘고 행복합니다.",
                "친구의 배신에 너무나도 화가 나고 분노가 치밉니다.",
                "사랑하는 가족을 잃어서 눈물이 멈추지 않습니다."
            ]},
            {"category": "감정 전이", "texts": [
                "처음에는 화가 났지만, 이야기를 듣고 보니 이해가 되었습니다.",
                "슬픔에 잠겨있다가 친구들이 위로해주어서 조금은 마음이 놓였습니다.",
                "기쁨과 설렘이 가득했는데, 갑자기 불안감이 몰려왔습니다."
            ]},
            {"category": "복합 감정", "texts": [
                "졸업식날 이별의 아쉬움과 새로운 시작에 대한 기대감이 교차합니다.",
                "합격 소식을 들었을 때 기쁨과 동시에 앞으로에 대한 부담감도 느껴졌습니다.",
                "오랜 친구를 만나니 그리움과 반가움, 그리고 미안함이 복잡하게 얽혔습니다."
            ]},
            {"category": "강도 변화", "texts": [
                "조금 짜증났다가, 매우 화가 났다가, 결국에는 폭발할 것 같은 분노가 치밀었습니다.",
                "살짝 기분이 좋아지다가, 점점 더 기뻐지더니, 마침내 환희에 차올랐습니다.",
                "약간의 우울함이 점점 깊어져서 결국 깊은 절망감에 빠져들었습니다."
            ]},
        ])

    def _save_json(path: str, obj: dict):
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)

    def _save_json_gz(path: str, obj: dict):
        tmp = path + ".tmp"
        with gzip.open(tmp, "wt", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)

    def _sanitize(name: str) -> str:
        return re.sub(r"[^0-9A-Za-z가-힣_.-]+", "_", name)[:80]

    def _make_summary(detailed: dict) -> dict:
        return {
            "analysis_timestamp": detailed.get("analysis_timestamp"),
            "module_version": detailed.get("module_version"),
            "environment": detailed.get("environment"),
            "options": detailed.get("options"),
            "test_summary": detailed.get("test_summary"),
        }

    def _make_preview(detailed: dict, limit: int) -> dict:
        trs = detailed.get("test_results", []) or []
        preview = []
        for i, r in enumerate(trs[: max(0, int(limit))]):
            preview.append({
                "category": r.get("category"),
                "input_text": r.get("input_text"),
                "metrics": r.get("metrics"),
            })
        return {"preview_count": len(preview), "test_results_preview": preview}

    # ---------- 테스트 케이스 ----------
    try:
        raw_cases = _get_test_cases()  # 사용자 함수가 있으면 사용
        test_cases = _normalize_test_cases(raw_cases)
    except Exception:
        test_cases = _load_cases()

    # ---------- 분석기 “한 번만” 초기화 ----------
    analyzer = EmotionIntensityAnalyzer(emotions_data_path)

    # ---------- 결과 컨테이너 ----------
    env_snapshot = _env_snapshot_safe()
    detailed_results = {
        "analysis_timestamp": datetime.now().isoformat(),
        "module_version": "1.3",
        "environment": env_snapshot,
        "options": {
            "no_embeddings": bool(args.no_embeddings),
            "embed_topk_sub": int(args.embed_topk_sub),
            "embed_topk_ctx": int(args.embed_topk_ctx),
            "max_transitions": int(args.max_transitions),
            "texts_per_category": int(args.texts_per_category),
            "compact_json": bool(args.compact_json),
        },
        "test_summary": {},
        "test_results": [],
    }

    # ---------- 실행 ----------
    for case in test_cases:
        category_name = str(case.get("category", "무명 카테고리"))
        full_texts = list(case.get("texts", []))
        items = full_texts[: args.texts_per_category] if args.texts_per_category > 0 else full_texts

        logger.info(f"\n=== {category_name} 테스트 시작 ({len(items)}개) ===")
        category_results = []

        for text in items:
            logger.info(f"\n입력 텍스트: {text}")

            intensity = analyzer.analyze_emotion_intensity(text) or {}
            transitions_full = analyzer.analyze_intensity_transitions(text) or {}
            situational = analyzer.analyze_situational_intensity(text) or {}

            transitions_limited = _limit_transitions(transitions_full, args.max_transitions)

            # (A) compact 이전의 '원시 리스트' 길이를 우선 확보
            n_transitions = len(transitions_limited.get("transitions", [])) \
                if isinstance(transitions_limited.get("transitions"), list) else 0
            n_sub_transitions = len(transitions_limited.get("sub_transitions", [])) \
                if isinstance(transitions_limited.get("sub_transitions"), list) else 0

            if args.compact_json:
                embeddings_json = {}  # 용량 절감
                tp = transitions_limited.get("temporal_patterns", {}) or {}
                transitions_json = {
                    "statistics": transitions_limited.get("statistics", {}),
                    "temporal_patterns": tp.get("global", tp),
                }
            else:
                transitions_json = transitions_limited
                if args.no_embeddings:
                    embeddings_json = {}
                else:
                    try:
                        embs = analyzer.generate_embeddings(
                            intensity, top_k_context=args.embed_topk_ctx, top_k_sub=args.embed_topk_sub
                        )
                        try:
                            embeddings_json = _convert_tensors_to_lists(embs)  # 원본 헬퍼가 있으면 사용
                        except Exception:
                            embeddings_json = embs if isinstance(embs, (list, dict)) else {}
                    except Exception as ee:
                        logger.warning(f"[임베딩 스킵] {ee}")
                        embeddings_json = {}

            dominant_emotion = None
            if intensity:
                try:
                    dominant_emotion = max(intensity.items(), key=lambda kv: _score_of(kv[1]))[0]
                except Exception:
                    pass

            max_intensity = max((_score_of(v) for v in intensity.values()), default=0.0)
            emotion_count = len(intensity)

            # (C) 전이 개수 산정: compact 이전 리스트 길이를 우선 사용, 없을 때만 통계로 보완
            if n_transitions > 0:
                transition_count = int(n_transitions)
            else:
                transition_count = int(_transition_count_from(transitions_json, compact=args.compact_json))

            significant_situations = sum(
                1 for s in (situational.values() if isinstance(situational, dict) else [])
                if isinstance(s, dict) and float(s.get("final_intensity_weight", 0.0)) >= 0.5
            )

            entry = {
                "category": category_name,  # ← 분할 저장/프리뷰용으로 카테고리 보존
                "input_text": text,
                "analysis": {
                    "emotion_intensity": intensity,
                    "transitions": transitions_json,
                    "situational": situational if isinstance(situational, dict) else {},
                    "embeddings": embeddings_json,
                },
                "metrics": {
                    "dominant_emotion": dominant_emotion,
                    "max_intensity": round(float(max_intensity), 3),
                    "emotion_count": int(emotion_count),
                    "transition_count": int(transition_count),
                    "significant_situations": int(significant_situations),
                },
            }
            detailed_results["test_results"].append(entry)
            category_results.append(entry)

            if dominant_emotion is not None:
                logger.info(f"주요 감정: {dominant_emotion}")
            logger.info(f"최대 강도: {max_intensity:.3f}")

        if category_results:
            detailed_results["test_summary"][category_name] = {
                "average_intensity": float(np.mean([r["metrics"]["max_intensity"] for r in category_results])),
                "emotion_diversity": float(np.mean([r["metrics"]["emotion_count"] for r in category_results])),
                "transition_frequency": float(np.mean([r["metrics"]["transition_count"] for r in category_results])),
                "situation_richness": float(np.mean([r["metrics"]["significant_situations"] for r in category_results])),
                "samples": int(len(category_results)),
            }

    # ---------- 저장 ----------
    logger.info("\n=== 감정 강도 분석 테스트 완료 ===")
    user = (getpass.getuser() or "user")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.join(logs_dir, f"{module_name}_{user}_{ts}")

    # 1) 요약/프리뷰 저장 (항상 강력 추천: 즉시 열람용 소형 파일)
    summary_obj = _make_summary(detailed_results)
    preview_obj = _make_preview(detailed_results, args.preview_limit)
    _save_json(base + "_summary.json", summary_obj)
    _save_json(base + "_preview.json", preview_obj)
    logger.info(f"[저장 완료] 요약 → {base}_summary.json")
    logger.info(f"[저장 완료] 프리뷰 → {base}_preview.json")

    # 2) 전체 결과 저장
    if not args.summary_only:
        ofmt = getattr(args, "output_format", "json.gz")  # ← 안전하게 한 번만 꺼내서 씁니다.
        if args.split_per_category:
            # 카테고리별로 쪼개 저장
            by_cat = {}
            for r in detailed_results["test_results"]:
                by_cat.setdefault(r.get("category", "기본"), []).append(r)
            for cat, rows in by_cat.items():
                cat_name = _sanitize(cat)
                obj = {
                    "analysis_timestamp": detailed_results["analysis_timestamp"],
                    "module_version": detailed_results["module_version"],
                    "environment": detailed_results["environment"],
                    "options": detailed_results["options"],
                    "test_summary": {cat: detailed_results["test_summary"].get(cat)},
                    "test_results": rows,
                }
                if ofmt in ("json", "both"):
                    _save_json(f"{base}_{cat_name}.json", obj)
                    logger.info(f"[저장 완료] 카테고리 JSON → {base}_{cat_name}.json")
                if ofmt in ("json.gz", "both"):
                    _save_json_gz(f"{base}_{cat_name}.json.gz", obj)
                    logger.info(f"[저장 완료] 카테고리 JSON.GZ → {base}_{cat_name}.json.gz")
        else:
            # 단일 파일로 저장
            if ofmt in ("json", "both"):
                _save_json(base + ".json", detailed_results)
                logger.info(f"[저장 완료] 전체 JSON → {base}.json")
            if ofmt in ("json.gz", "both"):
                _save_json_gz(base + ".json.gz", detailed_results)
                logger.info(f"[저장 완료] 전체 JSON.GZ → {base}.json.gz")

    return detailed_results


if __name__ == "__main__":
    main()

