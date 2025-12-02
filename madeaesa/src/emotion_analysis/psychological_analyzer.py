# psychological_analyzer.py
# -*- coding: utf-8 -*-
"""
Psychological

- 문장·시퀀스 전이/흐름 추적
- 감정 성숙도·심리적 안정성 산출
- 방어기제·인지편향·인지패턴(콤비네이션) 탐지
- 오케스트레이터 친화 리포트 출력
- 모든 키워드/가중/규칙은 EMOTIONS.JSON에서 동적 추출(하드코딩 금지)
"""

from __future__ import annotations

import os
import re
import json
import math
import time
import logging
import io
import cProfile
import pstats
from statistics import mean, median
from logging.handlers import RotatingFileHandler
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Tuple, Optional, Iterable, Set
from copy import deepcopy

# Optional deps
try:
    import ijson  # type: ignore
except Exception:
    ijson = None

try:
    from kiwipiepy import Kiwi  # type: ignore
except Exception:
    Kiwi = None


# -----------------------------------------------------------------------------
# Logger & Run Artifacts
# -----------------------------------------------------------------------------
LOGGER_NAME = "psych_cog_integrated_v3"
def _resolve_script_base() -> str:
    try:
        import sys
        base = os.path.splitext(os.path.basename(sys.argv[0]))[0]
        return base or LOGGER_NAME
    except Exception:
        return LOGGER_NAME

class EnhancedJsonFormatter(logging.Formatter):
    def __init__(self, app_name: str, datefmt: str = "%Y-%m-%d %H:%M:%S"):
        super().__init__(datefmt=datefmt)
        self.app_name = app_name

    def format(self, record: logging.LogRecord) -> str:
        # 지연 임포트로 상단 import 변경 없이 dataclass 지원
        try:
            from dataclasses import is_dataclass, asdict as _asdict
        except Exception:
            is_dataclass = lambda x: False  # noqa
            _asdict = lambda x: str(x)      # noqa

        payload = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "component": record.name,
            "filename": record.filename,
            "function": record.funcName,
            "lineno": record.lineno,
            "message": record.getMessage(),
            "app_name": self.app_name,
            "version": "v3-data-driven",
        }
        if hasattr(record, "data"):
            data = getattr(record, "data")
            try:
                if isinstance(data, dict):
                    payload["data"] = data
                elif is_dataclass(data):
                    payload["data"] = _asdict(data)
                else:
                    payload["data"] = str(data)
            except Exception as e:
                payload["data_error"] = str(e)
        if record.exc_info:
            payload["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]),
                "traceback": self.formatException(record.exc_info),
            }
        return json.dumps(payload, ensure_ascii=False)

def setup_app_logger(logger_name: str = LOGGER_NAME, app_name: str = "psych_cog_integrated") -> logging.Logger:
    logger = logging.getLogger(logger_name)
    if getattr(logger, "_configured", False):
        return logger

    logger.setLevel(logging.INFO)
    logger.propagate = False

    # 기본: 비간섭 (NullHandler만 부착)
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())

    # Opt-in: 파일/JSON/콘솔 로깅 (날짜별 폴더)
    try:
        from log_manager import get_log_manager
        log_manager = get_log_manager()
        log_dir = str(log_manager.get_log_dir("emotion_analysis", use_date_folder=True))
    except ImportError:
        # 폴백: 기존 방식 (날짜별 폴더 추가)
        from datetime import datetime
        base_log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
        today = datetime.now().strftime("%Y%m%d")
        log_dir = os.path.join(base_log_dir, today)
    script_base = _resolve_script_base()
    
    # 파일 로깅 (PSY_FILE_LOG=1)
    if os.environ.get("PSY_FILE_LOG", "0") == "1":
        try:
            os.makedirs(log_dir, exist_ok=True)
            fh = RotatingFileHandler(
                os.path.join(log_dir, f"{script_base}.log"),
                maxBytes=10*1024*1024, backupCount=5, encoding="utf-8"
            )
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(logging.Formatter(
                "[%(asctime)s] %(levelname)s [%(filename)s:%(lineno)d] - %(message)s",
                "%Y-%m-%d %H:%M:%S"
            ))
            logger.addHandler(fh)
        except Exception:
            pass

    # JSON 로깅 (PSY_JSON_LOG=1)
    if os.environ.get("PSY_JSON_LOG", "0") == "1":
        try:
            os.makedirs(log_dir, exist_ok=True)
            jh = RotatingFileHandler(
                os.path.join(log_dir, f"{script_base}.json"),
                maxBytes=10*1024*1024, backupCount=5, encoding="utf-8"
            )
            jh.setLevel(logging.DEBUG)
            jh.setFormatter(EnhancedJsonFormatter(app_name=app_name))
            logger.addHandler(jh)
        except Exception:
            pass

    # 콘솔 로깅 (PSY_CONSOLE_LOG=1)
    if os.environ.get("PSY_CONSOLE_LOG", "0") == "1":
        try:
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            ch.setFormatter(logging.Formatter(
                "[%(asctime)s] %(levelname)s %(message)s",
                "%Y-%m-%d %H:%M:%S"
            ))
            logger.addHandler(ch)
        except Exception:
            pass

    logger._configured = True  # type: ignore
    if os.environ.get("PSY_FILE_LOG", "0") == "1" or os.environ.get("PSY_JSON_LOG", "0") == "1":
        logger.info("로깅 초기화 완료", extra={"data": {"log_dir": log_dir, "script_base": script_base}})
    return logger

logger = setup_app_logger(LOGGER_NAME, app_name="psychological_analyzer")

def save_run_artifacts(report: Dict[str, Any], *, suffix: str = "result", keep_history: bool = True) -> Dict[str, str]:
    script_base = _resolve_script_base()
    # 통합 로그 관리자 사용 (날짜별 폴더)
    try:
        from log_manager import get_log_manager
        log_manager = get_log_manager()
        log_dir = str(log_manager.get_log_dir("emotion_analysis", use_date_folder=True))
    except ImportError:
        # 폴백: 기존 방식 (날짜별 폴더 추가)
        from datetime import datetime
        current_dir = os.path.dirname(os.path.abspath(__file__))
        base_log_dir = os.path.join(current_dir, "logs")
        today = datetime.now().strftime("%Y%m%d")
        log_dir = os.path.join(base_log_dir, today)
        os.makedirs(log_dir, exist_ok=True)

    ts = time.strftime("%Y%m%d_%H%M%S")
    latest_path = os.path.join(log_dir, f"{script_base}.{suffix}.json")
    hist_path   = os.path.join(log_dir, f"{script_base}_{ts}.{suffix}.json")

    try:
        # 최신본
        with open(latest_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        # 히스토리 저장은 옵션
        if keep_history:
            with open(hist_path, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)

        logger.info("분석 결과 저장 완료", extra={"data": {"result_latest": latest_path, **({"result_history": hist_path} if keep_history else {})}})
        return {"latest": latest_path, **({"history": hist_path} if keep_history else {})}
    except Exception:
        logger.exception("분석 결과 저장 실패")
        return {}



# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------
_SENT_SPLIT = re.compile(r'(?<=[.!?…])["”’)\]]*\s+')
_WS = re.compile(r"\s+")
_ZERO_WIDTH = re.compile(r"[\u200B-\u200D\uFEFF]")

def now_ms() -> int:
    return int(time.time() * 1000)

def norm01(x: float) -> float:
    return 0.0 if x <= 0.0 else (1.0 if x >= 1.0 else x)

def softmax(xs: List[float]) -> List[float]:
    if not xs:
        return []
    m = max(xs)
    # 오버플로우 방지
    ex = [math.exp(x - m) for x in xs]
    s = sum(ex) or 1.0
    return [v / s for v in ex]

def _nfkc(s: str) -> str:
    """NFKC 정규화(NFKC가 불가하면 원문 반환)."""
    try:
        import unicodedata as _ud  # 지역 임포트로 상단 임포트 변화 없이 사용
        return _ud.normalize("NFKC", s)
    except Exception:
        return s

def normalize_text(s: str, *, strip_zerowidth: bool = True, collapse_ws: bool = True) -> str:
    """매칭/토크나이징 전 간단 정규화."""
    if not s:
        return ""
    s = _nfkc(s)
    if strip_zerowidth:
        s = _ZERO_WIDTH.sub("", s)
    # 과도 반복 축약: ㅋㅋㅋㅋ → ㅋㅋ, ㅎㅎㅎㅎ → ㅎㅎ, ㅠㅠㅠ → ㅠㅠ, ㅜㅜㅜ → ㅜㅜ
    try:
        s = re.sub(r"(ㅋ)\1{2,}", r"\1\1", s)
        s = re.sub(r"(ㅎ)\1{2,}", r"\1\1", s)
        s = re.sub(r"(ㅠ)\1{2,}", r"\1\1", s)
        s = re.sub(r"(ㅜ)\1{2,}", r"\1\1", s)
    except Exception:
        pass
    if collapse_ws:
        s = _WS.sub(" ", s).strip()
    return s

def split_sentences(text: str) -> List[str]:
    """
    안전한 문장 분리:
    - 기본: 마침표/물음표/느낌표/말줄임표 + 선택적 닫힘기호 뒤 공백
    - 보조: 단일 문장으로 인식되면 개행으로 분리
    - 전처리: 제로폭·중복공백 정리
    """
    text = normalize_text(text)
    if not text:
        return []
    candidates = _SENT_SPLIT.split(text)
    if len(candidates) == 1:
        # 문장부호가 거의 없는 한국어 텍스트 대비: 개행 기준 추가 분리
        candidates = re.split(r"[\n\r]+", text)
    # 불릿/리스트 라인에서 끝부호가 없더라도 문장 취급
    sents = [p.strip(" \"”’)）]") for p in candidates if p and p.strip()]
    return [p for p in sents if p]

def junk_score(s: str) -> float:
    """난수/잡음 점수(0~1). 문자군 비율+단순 문자 엔트로피 기반."""
    s = (s or "").strip()
    if not s:
        return 1.0
    n = len(s)
    try:
        import unicodedata as _ud
        def frac(pred):
            try:
                return sum(1 for ch in s if pred(ch)) / float(max(1, n))
            except Exception:
                return 0.0
        letter = frac(lambda c: _ud.category(c).startswith("L"))
        number = frac(lambda c: _ud.category(c).startswith("N"))
        punct  = frac(lambda c: _ud.category(c).startswith("P"))
    except Exception:
        # 보수적 폴백
        letter = sum(1 for ch in s if ch.isalpha()) / float(n)
        number = sum(1 for ch in s if ch.isdigit()) / float(n)
        punct  = 0.0
    other = max(0.0, 1.0 - (letter + number + punct))
    from collections import Counter
    cnt = Counter(s)
    probs = [c / float(n) for c in cnt.values()]
    ent = -sum(p * math.log(p + 1e-9) for p in probs)
    uniq = len(cnt) / float(n)
    score = 0.45 * (punct + other) + 0.30 * max(0.0, 2.2 - ent) + 0.25 * max(0.0, 0.15 - uniq)
    return max(0.0, min(1.0, float(score)))

def unique_key(obj: Any) -> str:
    """dict/list도 안정 키로 직렬화하여 set/dedup 키로 사용."""
    try:
        if isinstance(obj, (dict, list, tuple, set)):
            # set은 순서 불안정 → 정렬 리스트로 치환
            if isinstance(obj, set):
                obj = sorted(list(obj))
            return json.dumps(obj, ensure_ascii=False, sort_keys=True)
        return str(obj)
    except Exception:
        return str(obj)

def dedup_by(items: List[Dict[str, Any]], key_fields: Tuple[str, ...]) -> List[Dict[str, Any]]:
    """지정 필드 조합으로 안정 중복 제거."""
    seen = set()
    out: List[Dict[str, Any]] = []
    for obj in items:
        key = tuple(unique_key(obj.get(k)) for k in key_fields)
        if key in seen:
            continue
        seen.add(key)
        out.append(obj)
    return out

def safe_get(d: Any, path: List[str], default: Any = None) -> Any:
    """dict 경로 안전 접근."""
    cur = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur

def safe_set(d: Dict[str, Any], path: List[str], value: Any) -> None:
    """dict 경로에 안전 쓰기(중간 노드 자동 생성)."""
    cur = d
    for p in path[:-1]:
        nxt = cur.get(p)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[p] = nxt
        cur = nxt
    cur[path[-1]] = value

def ensure_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, (tuple, set)):
        return list(x)
    return [x]

def iter_phrases(node: Any) -> Iterable[str]:
    """
    JSON의 다양한 형태에서 문자열 구문을 안전 추출:
    - str → 그대로
    - list/tuple/set → 각 요소
    - dict → 흔한 키('text','phrase','value')를 찾아 반환
    중첩 list 한 단계까지 평탄화.
    """
    if not node:
        return
    if isinstance(node, str):
        s = node.strip()
        if s:
            yield s
        return
    if isinstance(node, (list, tuple, set)):
        for v in node:
            if isinstance(v, str):
                s = v.strip()
                if s:
                    yield s
            elif isinstance(v, (list, tuple, set)):
                for w in v:
                    if isinstance(w, str) and w.strip():
                        yield w.strip()
            elif isinstance(v, dict):
                for key in ("text", "phrase", "value"):
                    if isinstance(v.get(key), str) and v[key].strip():
                        yield v[key].strip()
        return
    if isinstance(node, dict):
        for key in ("text", "phrase", "value"):
            val = node.get(key)
            if isinstance(val, str) and val.strip():
                yield val.strip()

def join_tokens(tokens: List[str]) -> str:
    """
    토큰 문자열을 매칭 친화 형태로 결합:
    - 공백 제거 → 한국어/조사 결합 매칭 용이
    - 제로폭 제거
    """
    if not tokens:
        return ""
    s = "".join(tokens)
    s = _ZERO_WIDTH.sub("", s)
    s = _WS.sub("", s)
    return s

def contains_any(hay: str, needles: Iterable[str]) -> bool:
    """부분 문자열 존재 여부(간단·빠른 체크)."""
    if not hay:
        return False
    for n in needles:
        if n and n in hay:
            return True
    return False

def find_all(hay: str, needles: Iterable[str]) -> List[str]:
    """부분 문자열 일치 전체 수집(원소 보존·중복 제거)."""
    found: List[str] = []
    seen = set()
    for n in needles or []:
        if not n:
            continue
        if n in hay and n not in seen:
            seen.add(n)
            found.append(n)
    return found

def batched(iterable: Iterable[Any], n: int) -> Iterable[List[Any]]:
    """이터러블을 n개씩 배치."""
    batch: List[Any] = []
    for x in iterable:
        batch.append(x)
        if len(batch) >= max(1, n):
            yield batch
            batch = []
    if batch:
        yield batch

def median(xs: List[float]) -> float:
    xs2 = sorted(xs)
    if not xs2:
        return 0.0
    m = len(xs2) // 2
    return xs2[m] if len(xs2) % 2 else (xs2[m - 1] + xs2[m]) / 2.0

def mad(values: List[float], *, scale: float = 1.4826) -> Tuple[float, float]:
    """
    중앙값과 MAD(중앙 절대편차) 반환.
    scale=1.4826 → 정규분포에서 표준편차에 근사.
    """
    if not values:
        return 0.0, 0.0
    med = median(values)
    dev = [abs(v - med) for v in values]
    return med, scale * median(dev)

def auto_threshold(values: List[float], *, floor: float = 0.05, k: float = 1.5, default: float = 0.3) -> float:
    """
    전이 임계 자동 산출: thr = max(floor, median + k*MAD)
    값이 부족하면 default.
    """
    if len(values) < 2:
        return default
    med, m = mad(values)
    return max(floor, med + k * m)

# -----------------------------------------------------------------------------
# Utils2
# -----------------------------------------------------------------------------
# [PATCH] --- Validation & Runtime Guards ------------------------------------

@dataclass
class ValidationResult:
    passed: bool
    reason: str = ""
    score: float = 1.0
    flags: list = None

class InputValidator:
    """가벼운 사전 검증: 길이/반복/언어비율/무의미 텍스트 점검."""
    def __init__(self, *, min_len=1, max_len=100000, junk_thr=0.75):
        self.min_len = int(min_len); self.max_len = int(max_len); self.junk_thr = float(junk_thr)

    def _lang_ratio(self, s: str) -> float:
        # 한글/라틴 비율의 합 (문자군 근사)
        if not s: return 0.0
        kor = sum(1 for ch in s if '\uac00' <= ch <= '\ud7a3')
        lat = sum(1 for ch in s if ('a' <= ch.lower() <= 'z'))
        return (kor + lat) / float(len(s))

    def _repetition_score(self, s: str) -> float:
        # 과도 반복: 동일 문자/바이그램 반복 비중 근사
        if not s: return 0.0
        import re
        runs = sum(1 for _ in re.finditer(r'(.)\1{3,}', s))  # 4연속 이상
        bigrams = [s[i:i+2] for i in range(max(0,len(s)-1))]
        uniq = len(set(bigrams)) or 1
        rep = 1.0 - (uniq / float(len(bigrams) or 1))
        return min(1.0, 0.6*rep + 0.4*min(1.0, runs/5.0))

    def validate(self, text: str) -> ValidationResult:
        t = normalize_text(text or "")
        n = len(t)
        if n < self.min_len: return ValidationResult(False, "too_short", 0.0, ["invalid_length"])
        if n > self.max_len: return ValidationResult(False, "too_long", 0.0, ["invalid_length"])
        js = junk_score(t)  # 기존 유틸 재사용
        lang_ok = self._lang_ratio(t) >= 0.35
        rep = self._repetition_score(t)
        if js >= self.junk_thr:
            return ValidationResult(False, "junk_like", 0.0, ["junk_input"])
        if not lang_ok:
            return ValidationResult(False, "lang_mismatch", 0.2, ["low_lang_ratio"])
        if rep >= 0.7:
            return ValidationResult(False, "over_repetition", 0.3, ["pattern_repetition"])
        # 통과 (품질 점수는 참고값)
        score = max(0.1, (1.0 - js) * (0.9 - 0.3*rep))
        return ValidationResult(True, score=score, flags=[])

class AnalysisTransaction:
    """단계별 체크포인트/롤백 유틸(경량)."""
    def __init__(self): 
        self._cp = {}; self._order = []
    def checkpoint(self, stage: str, data: dict):
        self._cp[stage] = deepcopy(data); self._order.append(stage)
    def rollback_to(self, stage: str):
        return deepcopy(self._cp.get(stage, {}))

class MemoryGuard:
    """psutil이 있으면 메모리 한도를 점검, 없으면 무해 폴백."""
    def __init__(self, limit_mb: int = 512):
        self.limit_mb = int(limit_mb)
        try:
            import psutil  # type: ignore
            self._psutil = psutil
        except Exception:
            self._psutil = None

    def check(self) -> dict:
        try:
            if not self._psutil:
                return {"ok": True, "rss_mb": None, "note": "psutil_not_available"}
            proc = self._psutil.Process()
            rss_mb = proc.memory_info().rss / 1024 / 1024
            if rss_mb > self.limit_mb * 1.2:
                return {"ok": False, "rss_mb": rss_mb, "action": "degrade"}
            return {"ok": True, "rss_mb": rss_mb}
        except Exception:
            return {"ok": True, "rss_mb": None, "note": "guard_error"}

class RegexShardMatcher:
    """
    샤딩된 OR-정규식 기반 매칭(기본) + 선택적 Aho–Corasick 전략 토글을 지원하는 매처.

    특징
    - strategy: "regex_shard"(기본) | "aho"
      * "aho"는 'pyahocorasick' 가 설치된 경우에만 사용. 없으면 자동으로 regex_shard로 폴백.
    - ignore_space: True이면 패턴과 텍스트 모두 공백 제거 후 매칭(기존 동작 유지)
    - boundary: True이면 (?<!\\w) ... (?!\\w) 경계를 적용(정규식) / Aho에서는 후처리로 경계검사
    - update(added, removed): 런타임 증분 업데이트 API
      * 추가는 델타 샤드로 누적 컴파일(소량 추가 시 전량 재컴파일 회피)
      * 제거는 허용 집합에서만 제외(기존 샤드에는 남아있더라도 결과 필터링으로 배제)
      * 누적 제거/추가가 많아지면 임계치에서 전체 리빌드 수행
    """

    # ------------------------------ ctor ------------------------------
    def __init__(
        self,
        phrases: List[str],
        shard_size: int = 512,
        *,
        boundary: bool = False,
        ignore_space: bool = True,
        strategy: str = "regex_shard",
        rebuild_threshold: float = 0.35,   # 제거/추가 누적이 전체의 35% 넘으면 풀 리빌드
        min_compile_batch: int = 32,       # 델타 컴파일 최소 배치 크기
    ):
        self.boundary = bool(boundary)
        self.ignore_space = bool(ignore_space)
        self.strategy = strategy if strategy in ("regex_shard", "aho") else "regex_shard"
        self.shard_size = int(max(32, shard_size))
        self.rebuild_threshold = float(max(0.0, min(0.9, rebuild_threshold)))
        self.min_compile_batch = int(max(8, min_compile_batch))

        # 정규화된 패턴 집합(허용되는 키들)
        self._norm_allow: Set[str] = set()
        # 정규식 전략에서 쓸 샤드들
        self._rx_shards: List[re.Pattern] = []
        # 델타 누적(추가분)
        self._delta_patterns: List[str] = []

        # 노멀라이즈 → 정규식 조각 캐시(경계/이스케이프 반영된 문자열)
        self._norm_to_pat: Dict[str, str] = {}

        # Aho–Corasick용 오토마톤
        self._aho = None
        self._aho_available = False
        if self.strategy == "aho":
            try:
                import ahocorasick  # pyahocorasick
                self._ahocorasick = ahocorasick
                self._aho_available = True
            except Exception:
                # 설치 안 되어 있으면 폴백
                self.strategy = "regex_shard"
                self._aho_available = False
                try:
                    logger.warning("[RegexShardMatcher] pyahocorasick not found; fallback to regex_shard")
                except Exception:
                    pass

        # 초기 로드
        self._bulk_load(phrases)

    # --------------------------- public API ---------------------------
    def find_all(self, text: str) -> List[str]:
        """텍스트에서 패턴을 모두 찾아(중복 제거) 노멀라이즈된 매칭 문자열을 반환."""
        if not text:
            return []
        t = re.sub(r"\s+", "", text) if self.ignore_space else text
        if not t:
            return []

        out: List[str] = []
        seen: Set[str] = set()

        if self.strategy == "aho" and self._aho is not None:
            # Aho 결과 후처리(경계/허용집합 필터)
            for end_idx, norm in self._aho.iter(t):
                start_idx = end_idx - len(norm) + 1
                if self.boundary and not self._is_boundary_ok(t, start_idx, end_idx):
                    continue
                if norm not in self._norm_allow:
                    continue
                if norm not in seen:
                    seen.add(norm); out.append(norm)
            return out

        # regex_shard
        for rx in self._rx_shards:
            for m in rx.finditer(t):
                norm = m.group(0)
                if not norm:
                    continue
                if norm not in self._norm_allow:
                    continue
                if norm not in seen:
                    seen.add(norm); out.append(norm)
        return out

    def update(self, added: Optional[Iterable[str]] = None, removed: Optional[Iterable[str]] = None) -> None:
        """
        런타임 증분 업데이트.
        - added: 추가할 패턴 목록
        - removed: 제거할 패턴 목록
        """
        added = list(added or [])
        removed = list(removed or [])
        if not added and not removed:
            return

        # 제거 먼저 반영(허용 집합에서 제외)
        removed_norms: List[str] = []
        for p in removed:
            norm, _ = self._normalize_phrase(p)
            if norm in self._norm_allow:
                self._norm_allow.discard(norm)
                removed_norms.append(norm)
                # 정규식 조각 캐시는 남겨두되, 허용집합에서 빠졌으므로 find_all에서 필터된다.

        # 추가(노멀/패턴 생성 후 허용집합에 등록)
        added_norms: List[str] = []
        for p in added:
            norm, pat = self._normalize_phrase(p)
            if not norm:
                continue
            if norm in self._norm_allow:
                continue
            self._norm_allow.add(norm)
            self._norm_to_pat[norm] = pat
            added_norms.append(norm)

        # 전략별 갱신
        total = max(1, len(self._norm_allow) + len(removed_norms))  # 분모 0 방지
        change_ratio = (len(added_norms) + len(removed_norms)) / float(total)

        if self.strategy == "aho" and self._aho_available:
            # Aho는 재생성이 저렴하므로 일정 임계 넘거나 제거 발생 시 풀 리빌드 권장
            if change_ratio >= self.rebuild_threshold or removed_norms:
                self._build_aho()
            else:
                # 소량 추가만 있을 때 빠른 증분(오토마톤 append는 미지원 → 경량 리빌드)
                self._build_aho()
            return

        # regex_shard 전략
        if change_ratio >= self.rebuild_threshold or removed_norms:
            # 제거가 포함되면 샤드 내부에는 여전히 남기 때문에 깔끔히 풀 리빌드
            self._build_regex_shards()
        else:
            # 추가만 소량 → 델타 샤드로 컴파일 후 append
            # 델타 패턴 누적
            for n in added_norms:
                pat = self._norm_to_pat.get(n)
                if pat:
                    self._delta_patterns.append(pat)
            if len(self._delta_patterns) >= max(self.min_compile_batch, min(self.shard_size, 256)):
                self._rx_shards.append(re.compile("|".join(self._delta_patterns)))
                self._delta_patterns.clear()

    # -------------------------- internal: build -----------------------
    def _bulk_load(self, phrases: List[str]) -> None:
        self._norm_allow.clear()
        self._norm_to_pat.clear()
        for p in (phrases or []):
            norm, pat = self._normalize_phrase(p)
            if not norm:
                continue
            self._norm_allow.add(norm)
            self._norm_to_pat[norm] = pat

        if self.strategy == "aho" and self._aho_available:
            self._build_aho()
        else:
            self.strategy = "regex_shard"  # 안전 폴백
            self._build_regex_shards()

    def _build_regex_shards(self) -> None:
        self._rx_shards = []
        self._delta_patterns = []
        if not self._norm_allow:
            return

        buf: List[str] = []
        limit = max(32, self.shard_size)
        # 허용 집합 기준으로 패턴 재구성(사전 순은 안정성 위해)
        for norm in sorted(self._norm_allow):
            pat = self._norm_to_pat.get(norm)
            if not pat:
                continue
            buf.append(pat)
            if len(buf) >= limit:
                self._rx_shards.append(re.compile("|".join(buf)))
                buf = []
        if buf:
            self._rx_shards.append(re.compile("|".join(buf)))

    def _build_aho(self) -> None:
        if not self._aho_available:
            self._aho = None
            return
        A = self._ahocorasick.Automaton()  # type: ignore[attr-defined]
        for norm in self._norm_allow:
            # Aho는 원 문자열을 키로 저장. 경계 체크는 검색 시 후처리
            A.add_word(norm, norm)
        A.make_automaton()
        self._aho = A

    # ---------------------- internal: normalize -----------------------
    def _normalize_phrase(self, p: str) -> Tuple[str, str]:
        """원문 패턴 → (norm, regex_piece). norm은 공백 제거/정규화된 원형 문자열."""
        if not p:
            return "", ""
        s = normalize_text(p)
        if not s:
            return "", ""
        if self.ignore_space:
            s = re.sub(r"\s+", "", s)
        norm = s  # find_all에서 반환될 문자열(공백 제거 버전)

        # 정규식 조각 생성(이스케이프 후 경계 처리)
        pat = re.escape(s)
        if self.boundary:
            pat = rf"(?<!\w){pat}(?!\w)"
        return norm, pat

    # ---------------------- internal: boundary ------------------------
    @staticmethod
    def _is_word_char(ch: str) -> bool:
        # 간단한 워드 문자 정의: 알파/숫자/밑줄 + 한글/가나/한자
        return bool(re.match(r"[A-Za-z0-9_]", ch)) or bool(re.match(r"[가-힣\u3040-\u30FF\u4E00-\u9FFF]", ch))

    def _is_boundary_ok(self, text: str, start: int, end: int) -> bool:
        """Aho 경로에서 boundary=True일 때 전후 문자가 '워드'가 아닌지 검사."""
        # start/end는 포함 인덱스(ahocorasick에서 end는 포함). 정규식과 일치시키기 위해 다음과 같이 처리:
        # start 앞, end 뒤를 확인
        left_ok = True
        right_ok = True
        if start - 1 >= 0:
            left_ok = not self._is_word_char(text[start - 1])
        if end + 1 < len(text):
            right_ok = not self._is_word_char(text[end + 1])
        return left_ok and right_ok


# -----------------------------------------------------------------------------
# AnalyzerConfig — lean edition (1–8 개선사항을 위한 최소 확장)
# -----------------------------------------------------------------------------
@dataclass(slots=True)
class AnalyzerConfig:
    # --- Evidence Gate ---
    gate_min_types: int = int(os.environ.get("PSY_GATE_MIN_TYPES", "1"))
    gate_min_sum: float = float(os.environ.get("PSY_GATE_MIN_SUM", "1.0"))

    # --- Contribution weights (I, C, M, T) ---
    w_intensity: float = float(os.environ.get("PSY_W_INTENSITY", "0.4"))
    w_context: float = float(os.environ.get("PSY_W_CONTEXT", "0.35"))
    w_modifier: float = float(os.environ.get("PSY_W_MODIFIER", "0.2"))
    w_transition: float = float(os.environ.get("PSY_W_TRANSITION", "0.1"))
    auto_normalize_weights: bool = bool(int(os.environ.get("PSY_W_AUTO_NORM", "1")))

    # --- Per-hit weights (monotonic) ---
    intensity_weights_low: float = float(os.environ.get("PSY_W_INT_LOW", "0.2"))
    intensity_weights_medium: float = float(os.environ.get("PSY_W_INT_MED", "0.5"))
    intensity_weights_high: float = float(os.environ.get("PSY_W_INT_HIGH", "0.8"))
    context_weights_low: float = float(os.environ.get("PSY_W_CTX_LOW", "0.1"))
    context_weights_medium: float = float(os.environ.get("PSY_W_CTX_MED", "0.2"))
    context_weights_high: float = float(os.environ.get("PSY_W_CTX_HIGH", "0.3"))

    # --- Modifiers & Transitions ---
    modifier_per_amp: float = float(os.environ.get("PSY_W_MOD_AMP", "0.05"))
    modifier_per_dim: float = float(os.environ.get("PSY_W_MOD_DIM", "0.05"))
    transition_trigger_bonus: float = float(os.environ.get("PSY_W_T_TRIG", "0.12"))
    transition_shiftpoint_bonus: float = float(os.environ.get("PSY_W_T_SHIFT", "0.05"))

    # Transition sensitivity
    transition_sensitivity: float = float(os.environ.get("PSY_TRANSITION_SENS", "-1.0"))
    trans_auto_floor: float = float(os.environ.get("PSY_TRANS_AUTO_FLOOR", "0.05"))
    trans_auto_k: float = float(os.environ.get("PSY_TRANS_AUTO_K", "1.2"))
    trans_auto_default: float = float(os.environ.get("PSY_TRANS_AUTO_DEFAULT", "0.3"))

    # --- Dependency & Junk-Guard ---
    dep_kiwi_required: bool = bool(int(os.environ.get("PSY_DEP_KIWI_REQUIRED", "0")))
    junk_guard_threshold: float = float(os.environ.get("PSY_JUNK_THR", "0.65"))
    junk_short_len: int = int(os.environ.get("PSY_JUNK_SHORT", "4"))

    # --- Defense gating (반복/윈도/강도) ---
    def_min_repeats: int = int(os.environ.get("PSY_DEF_MIN_REPEATS", "2"))
    def_window: int = int(os.environ.get("PSY_DEF_WINDOW", "2"))
    def_min_intensity: float = float(os.environ.get("PSY_DEF_MIN_INT", "0.20"))
    def_use_adaptive_gate: bool = bool(int(os.environ.get("PSY_DEF_ADAPT", "1")))

    # --- Stability weighting ---
    stab_w_var: float = float(os.environ.get("PSY_STAB_W_VAR", "0.35"))
    stab_w_span: float = float(os.environ.get("PSY_STAB_W_SPAN", "0.25"))
    stab_w_sudden: float = float(os.environ.get("PSY_STAB_W_SUDDEN", "0.25"))
    stab_w_jerk: float = float(os.environ.get("PSY_STAB_W_JERK", "0.15"))
    stab_w_trans_rate: float = float(os.environ.get("PSY_STAB_W_TRATE", "0.20"))
    stab_len_penalty_k: float = float(os.environ.get("PSY_STAB_LENK", "6.0"))

    # --- Context / Caps ---
    context_window: int = int(os.environ.get("PSY_CONTEXT_WIN", "1"))
    max_defense_per_sentence: int = int(os.environ.get("PSY_MAX_DEF_PER_SENT", "2"))
    max_bias_per_sentence: int = int(os.environ.get("PSY_MAX_BIAS_PER_SENT", "2"))
    top_defense: int = int(os.environ.get("PSY_TOP_DEF", "5"))
    top_bias: int = int(os.environ.get("PSY_TOP_BIAS", "5"))

    # --- Locale / Version ---
    language: str = os.environ.get("PSY_LANG", "ko")
    version: str = "v4-hierarchical"
    prefer_kiwi: bool = bool(int(os.environ.get("PSY_PREFER_KIWI", "1")))
    # 스키마 검증 모드: off | warn | raise
    schema_validation_mode: str = "warn"

    # --- Matching/threshold knobs ---
    regex_shard_size: int = int(os.environ.get("PSY_REGEX_SHARD_SIZE", "512"))
    matcher_strategy: str = os.environ.get("PSY_MATCHER_STRATEGY", "regex_shard")  # "regex_shard" | "aho" (미래 대비)
    local_transition_bonus_max: float = float(os.environ.get("PSY_LOCAL_TRANS_BONUS_MAX", "0.25"))
    lex_evidence_cap: float = float(os.environ.get("PSY_LEX_EVIDENCE_CAP", "0.3"))
    adaptive_min_sum_weak: float = float(os.environ.get("PSY_ADAPTIVE_MIN_SUM_WEAK", "0.7"))

    # --- Guards / Parallelism ---
    mem_guard_enabled: bool = bool(int(os.environ.get("PSY_MEM_GUARD", "1")))
    mem_guard_limit_mb: int = int(os.environ.get("PSY_MEM_LIMIT_MB", "51200"))  # 50GB로 대폭 증가 (64GB RAM 대비)
    enable_parallel_sentence_scoring: bool = bool(int(os.environ.get("PSY_PAR_SENT", "1")))  # 병렬 처리 활성화
    parallel_max_workers: int = int(os.environ.get("PSY_PAR_WORKERS", "64"))  # 라이젠 AI 9/HX 370에 맞춰 워커 수 극대화
    parallel_min_sentences: int = int(os.environ.get("PSY_PAR_MIN", "4"))  # 병렬 처리 임계값 낮춤

    # --- Scoring/normalization knobs ---
    unique_match_per_sent: bool = bool(int(os.environ.get("PSY_UNIQUE_MATCH_PER_SENT", "1")))
    use_idf_weighting: bool = bool(int(os.environ.get("PSY_USE_IDF_WEIGHTING", "1")))
    length_norm: str = os.environ.get("PSY_LENGTH_NORM", "log")  # "none"|"sqrt"|"log"
    len_norm_floor: float = float(os.environ.get("PSY_LEN_NORM_FLOOR", "1.2"))  # 초단문 안전판
    clip_score: float = float(os.environ.get("PSY_CLIP_SCORE", "6.0"))
    # [PATCH] 라벨 캘리브레이터 ENV 반영(신규 키 우선, 구 키 폴백)
    calibration_scale: float = float(os.environ.get("PSY_CAL_SCALE", os.environ.get("PSY_CALIBRATION_SCALE", "0.0")))

    # --- Hierarchical Aggregation (NEW) ---
    enable_hierarchy: bool = bool(int(os.environ.get("PSY_ENABLE_HIER", "1")))
    hier_rollup_mode: str = os.environ.get("PSY_HIER_ROLLUP_MODE", "softmax")  # "max"|"mean"|"softmax"
    hier_rollup_tau: float = float(os.environ.get("PSY_HIER_ROLLUP_TAU", "1.0"))
    calibrator_prior_scale: float = float(os.environ.get("PSY_CAL_PRIOR_SCALE", "1.0"))

    # --- Heuristic toggles & thresholds ---
    enable_adversative_heuristic: bool = bool(int(os.environ.get("PSY_ENABLE_ADVERSATIVE_HEUR", "1")))
    adversative_min_score: float = float(os.environ.get("PSY_ADVERSATIVE_MIN_SCORE", "0.28"))
    adversative_evidence_min: float = float(os.environ.get("PSY_ADVERSATIVE_EVIDENCE_MIN", "0.6"))
    flag_emit_heur_hits: bool = bool(int(os.environ.get("PSY_FLAG_EMIT_HEUR_HITS", "1")))
    hide_internal_markers: bool = bool(int(os.environ.get("PSY_HIDE_INTERNAL_MARKERS", "1")))
    mask_evidence_labels: bool = bool(int(os.environ.get("PSY_MASK_EVIDENCE_LABELS", "1")))
    # 휴리스틱 증거 가중 최소치(운영 튜닝용 노출)
    heur_evidence_min: float = float(os.environ.get("PSY_HEUR_EVIDENCE_MIN", "0.6"))

    # --- Heuristic lexicons (fallback; JSON에서 override 가능) ---
    adversative_triggers: List[str] = field(default_factory=lambda: [
        "하지만", "그렇지만", "그러나", "반면", "반대로", "다만", "으나", "지만", "다가도"
    ])
    ko_pos_stems: List[str] = field(default_factory=lambda: [
        "좋", "기쁘", "행복", "편안", "만족", "희망", "나아지", "괜찮"
    ])
    ko_neg_stems: List[str] = field(default_factory=lambda: [
        "슬프", "불안", "우울", "두렵", "걱정", "짜증", "화나", "힘들", "나쁘", "나빠"
    ])

    # --- JSON 경로 (override 우선) ---
    json_path_adversatives: str = os.environ.get("PSY_JSON_PATH_ADVERSATIVES", "global.linguistic_patterns.adversatives")
    json_path_ko_pos_stems: str = os.environ.get("PSY_JSON_PATH_KO_POS", "global.markers.ko_pos_stems")
    json_path_ko_neg_stems: str = os.environ.get("PSY_JSON_PATH_KO_NEG", "global.markers.ko_neg_stems")
    json_path_hierarchy: str = os.environ.get("PSY_JSON_PATH_HIERARCHY", "global.hierarchy")

    # --- 자동 추론 토글 (JSON 비어 있을 때만 시도) ---
    derive_adversatives_from_transitions: bool = bool(int(os.environ.get("PSY_DERIVE_ADV_FROM_TRANS", "1")))
    derive_stems_from_valence_markers: bool = bool(int(os.environ.get("PSY_DERIVE_STEMS_FROM_MARKERS", "1")))

    # transitions에서 adversative 후보를 찾을 때 사용할 태그 키워드(소문자 비교)
    adversative_transition_tags: List[str] = field(default_factory=lambda: ["adversative", "contrast", "역접"])

    # -------------------------------------------------------------------------
    # Validation & normalization
    # -------------------------------------------------------------------------
    def __post_init__(self):
        # ints
        if self.gate_min_types < 1:
            self.gate_min_types = 1
        self.context_window = max(0, int(self.context_window))
        self.max_defense_per_sentence = max(0, int(self.max_defense_per_sentence))
        self.max_bias_per_sentence = max(0, int(self.max_bias_per_sentence))
        self.top_defense = max(0, int(self.top_defense))
        self.top_bias = max(0, int(self.top_bias))
        self.regex_shard_size = max(16, min(8192, int(self.regex_shard_size)))
        # new ints
        self.def_min_repeats = max(0, int(self.def_min_repeats))
        self.def_window = max(0, int(self.def_window))
        self.junk_short_len = max(0, int(self.junk_short_len))

        # floats & NaN guard
        for name in (
            "gate_min_sum",
            "w_intensity", "w_context", "w_modifier", "w_transition",
            "intensity_weights_low", "intensity_weights_medium", "intensity_weights_high",
            "context_weights_low", "context_weights_medium", "context_weights_high",
            "modifier_per_amp", "modifier_per_dim",
            "transition_trigger_bonus", "transition_shiftpoint_bonus",
            "trans_auto_floor", "trans_auto_k", "trans_auto_default",
            "local_transition_bonus_max", "lex_evidence_cap", "adaptive_min_sum_weak",
            "clip_score", "calibration_scale",
            "adversative_min_score", "adversative_evidence_min",
            "len_norm_floor", "hier_rollup_tau", "calibrator_prior_scale",
            "heur_evidence_min",
            # NEW
            "junk_guard_threshold", "def_min_intensity",
            "stab_w_var", "stab_w_span", "stab_w_sudden", "stab_w_jerk", "stab_w_trans_rate",
            "stab_len_penalty_k",
        ):
            v = getattr(self, name)
            if not isinstance(v, (int, float)) or v != v:
                v = 0.0
            setattr(self, name, float(max(0.0, v)))

        # monotonic per-hit weights
        ints = sorted([self.intensity_weights_low, self.intensity_weights_medium, self.intensity_weights_high])
        self.intensity_weights_low, self.intensity_weights_medium, self.intensity_weights_high = ints
        ctxs = sorted([self.context_weights_low, self.context_weights_medium, self.context_weights_high])
        self.context_weights_low, self.context_weights_medium, self.context_weights_high = ctxs

        # I/C/M/T normalize
        if self.auto_normalize_weights:
            total = self.w_intensity + self.w_context + self.w_modifier + self.w_transition
            if total > 0:
                self.w_intensity /= total
                self.w_context /= total
                self.w_modifier /= total
                self.w_transition /= total

        # sensitivity bounds
        if self.transition_sensitivity != -1.0:
            self.transition_sensitivity = max(0.0, float(self.transition_sensitivity))

        # caps
        self.local_transition_bonus_max = float(min(0.5, max(0.0, self.local_transition_bonus_max)))
        self.lex_evidence_cap = float(min(1.0, max(0.0, self.lex_evidence_cap)))
        self.adaptive_min_sum_weak = float(max(0.0, self.adaptive_min_sum_weak))
        self.adversative_min_score = float(min(1.0, max(0.0, self.adversative_min_score)))
        self.adversative_evidence_min = float(min(1.0, max(0.0, self.adversative_evidence_min)))
        self.len_norm_floor = float(max(1.0, self.len_norm_floor))

        # rollup mode
        if self.hier_rollup_mode not in {"max", "mean", "softmax"}:
            self.hier_rollup_mode = "softmax"
        self.hier_rollup_tau = max(0.01, float(self.hier_rollup_tau))

        # length norm mode
        ln = (self.length_norm or "log").lower().strip()
        if ln not in {"none", "sqrt", "log"}:
            ln = "log"
        self.length_norm = ln

        # schema validation mode guard
        mode = str(getattr(self, "schema_validation_mode", "warn")).strip().lower()
        if mode not in {"off", "warn", "raise"}:
            mode = "warn"
        self.schema_validation_mode = mode

        # heuristics evidence minimum clamp (0~1)
        self.heur_evidence_min = float(min(1.0, max(0.0, float(getattr(self, "heur_evidence_min", 0.6)))))

        # optional env overrides for lists
        adv_env = os.environ.get("PSY_ADV_TRIGGERS")
        if adv_env:
            self.adversative_triggers = self._parse_csv_env(adv_env, self.adversative_triggers)
        pos_env = os.environ.get("PSY_KO_POS_STEMS")
        if pos_env:
            self.ko_pos_stems = self._parse_csv_env(pos_env, self.ko_pos_stems)
        neg_env = os.environ.get("PSY_KO_NEG_STEMS")
        if neg_env:
            self.ko_neg_stems = self._parse_csv_env(neg_env, self.ko_neg_stems)

        # matcher strategy guard
        if self.matcher_strategy not in {"regex_shard", "aho"}:
            self.matcher_strategy = "regex_shard"
        # [PATCH] __post_init__ corrections for guard/parallel limits
        self.mem_guard_limit_mb = max(256, int(self.mem_guard_limit_mb))
        self.parallel_max_workers = max(1, int(self.parallel_max_workers))
        self.parallel_min_sentences = max(1, int(self.parallel_min_sentences))
        # [PATCH] calibration_scale 안전 상한
        try:
            self.calibration_scale = float(self.calibration_scale)
        except Exception:
            self.calibration_scale = 0.0
        self.calibration_scale = max(0.0, min(0.2, float(self.calibration_scale)))

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AnalyzerConfig":
        base = cls()
        for f in cls.__dataclass_fields__.keys():  # type: ignore
            if f in d and d[f] is not None:
                setattr(base, f, d[f])
        base.__post_init__()
        return base

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def copy_with(self, **overrides: Any) -> "AnalyzerConfig":
        d = self.to_dict()
        for k, v in overrides.items():
            if k in self.__dataclass_fields__:  # type: ignore
                d[k] = v
        return AnalyzerConfig.from_dict(d)

    def min_sum_for_signal(self, strong_intensity: bool, strong_context: bool) -> float:
        if strong_intensity or strong_context:
            return self.gate_min_sum
        return max(self.adaptive_min_sum_weak, self.trans_auto_default)

    def lex_evidence_weight(self, n_hits: int) -> float:
        return min(self.lex_evidence_cap, 0.10 * float(max(0, n_hits)))

    def local_transition_bonus(self, n_local_hits: int) -> float:
        return min(self.local_transition_bonus_max, 0.05 * float(max(0, min(3, n_local_hits))))

    # -------------------------------------------------------------------------
    # JSON → (명시 경로) → (자동 추론) → (기본값) 우선순위로 휴리스틱 어휘 세팅
    # ※ prepare_emotions_index() 초반에 cfg.apply_linguistic_overrides(emotions_json) 호출 권장
    # -------------------------------------------------------------------------
    def apply_linguistic_overrides(self, emotions_data: Optional[Dict[str, Any]]) -> None:
        if not emotions_data or not isinstance(emotions_data, dict):
            return

        adv = self._get_by_path(emotions_data, self.json_path_adversatives)
        pos = self._get_by_path(emotions_data, self.json_path_ko_pos_stems)
        neg = self._get_by_path(emotions_data, self.json_path_ko_neg_stems)

        # 자동 추론 (JSON 미제공 시)
        if (not self._is_nonempty_str_list(adv)) and self.derive_adversatives_from_transitions:
            adv = self._smart_collect_adversatives_from_transitions(emotions_data)
        if (not self._is_nonempty_str_list(pos)) and self.derive_stems_from_valence_markers:
            pos_markers = self._find_any_list(emotions_data, [
                "global.valence.pos_markers", "valence.pos_markers", "markers.pos_markers", "pos_markers",
            ])
            pos = self._derive_stems_from_markers(pos_markers)
        if (not self._is_nonempty_str_list(neg)) and self.derive_stems_from_valence_markers:
            neg_markers = self._find_any_list(emotions_data, [
                "global.valence.neg_markers", "valence.neg_markers", "markers.neg_markers", "neg_markers",
            ])
            neg = self._derive_stems_from_markers(neg_markers)

        # 반영(없으면 기본값 유지)
        if self._is_nonempty_str_list(adv):
            self.adversative_triggers = self._dedup_strs(adv)
        if self._is_nonempty_str_list(pos):
            self.ko_pos_stems = self._dedup_strs(pos)
        if self._is_nonempty_str_list(neg):
            self.ko_neg_stems = self._dedup_strs(neg)

    # ---------------- internal utils ----------------
    @staticmethod
    def _parse_csv_env(raw: str, fallback: List[str]) -> List[str]:
        try:
            vals = [x.strip() for x in raw.split(",")]
            return [v for v in vals if v]
        except Exception:
            return list(fallback)

    @staticmethod
    def _dedup_strs(items: Iterable[str]) -> List[str]:
        seen, out = set(), []
        for x in items:
            if not isinstance(x, str):
                continue
            s = x.strip()
            if not s or s in seen:
                continue
            seen.add(s)
            out.append(s)
        return out

    @staticmethod
    def _is_nonempty_str_list(x: Any) -> bool:
        return isinstance(x, list) and any(isinstance(y, str) and y.strip() for y in x)

    @staticmethod
    def _get_by_path(d: Dict[str, Any], dotted: Optional[str]) -> Optional[Any]:
        if not dotted:
            return None
        cur: Any = d
        for part in dotted.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return None
            cur = cur[part]
        return cur

    def _find_any_list(self, d: Dict[str, Any], paths: List[str]) -> Optional[List[str]]:
        for p in paths:
            v = self._get_by_path(d, p)
            if self._is_nonempty_str_list(v):
                return v
        return None

    def _smart_collect_adversatives_from_transitions(self, d: Dict[str, Any]) -> Optional[List[str]]:
        trans = d.get("transitions") if isinstance(d.get("transitions"), list) else None
        if not trans:
            trans = self._get_by_path(d, "global.transitions") or self._get_by_path(d, "linguistic.transitions")
        if not trans or not isinstance(trans, list):
            return None
        tags_lower = set(t.lower() for t in self.adversative_transition_tags)
        out: List[str] = []
        for t in trans:
            if not isinstance(t, dict):
                continue
            hint_strs: List[str] = []
            for k in ("tags", "type", "name", "label", "category"):
                v = t.get(k)
                if isinstance(v, str):
                    hint_strs.append(v.lower())
                elif isinstance(v, list):
                    hint_strs.extend([str(x).lower() for x in v if isinstance(x, (str, int, float))])
            if any(any(tag in h for h in hint_strs) for tag in tags_lower):
                trigs = t.get("triggers") or t.get("patterns") or t.get("lexicon")
                if isinstance(trigs, list):
                    out.extend([str(x) for x in trigs if isinstance(x, (str, int, float))])
        out = self._dedup_strs(out)
        return out or None

    def _derive_stems_from_markers(self, markers: Optional[List[str]]) -> Optional[List[str]]:
        if not self._is_nonempty_str_list(markers):
            return None
        stems: List[str] = []
        for m in markers:  # type: ignore
            s = str(m).strip()
            if not s:
                continue
            s = re.sub(r"[^가-힣\s]", "", s).strip()
            if not s:
                continue
            token = s.split()[0]
            stems.append(self._ko_naive_stem(token))
        stems = [t for t in stems if t]
        return self._dedup_strs(stems) or None

    @staticmethod
    def _ko_naive_stem(term: str) -> str:
        # 매우 보수적인 종결/접미 제거
        t = term.strip()
        for suf in ("스럽다", "스럽", "했다", "했어", "하다", "였다", "이다", "습니다", "어요", "아요", "다"):
            if t.endswith(suf) and len(t) > len(suf):
                t = t[: -len(suf)]
                break
        return t.strip()

# -----------------------------------------------------------------------------
# Tokenizer (Kiwi optional) — batch API + unified coarse POS
# -----------------------------------------------------------------------------
class SimpleTokenizer:
    """
    - kiwi 사용 시: 형태소 기반 토큰화(+옵션 표제어/품사 필터)
    - kiwi 미사용 시: URL/이메일/숫자/영문/한글/구두점 포함 폴백 정규식 토큰화
    - normalize_text()로 NFKC/제로폭/다중 공백 정리
    - 캐시: 문장 분할/토큰화 결과 모두 캐싱
    - 목표: Kiwi/폴백 모두 coarse POS를 NOUN/VERB/ADJ/ADV/PART 로 정규화(구두점은 PUNCT)
    """

    # URL/Email 우선 매칭 → 일반 토큰
    _URL_RE   = r"(?:https?://|www\.)\S+"
    _EMAIL_RE = r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
    # 숫자, 영문(축약형 포함), 한글, CJK, 구두점
    _WORD_RE  = r"(?:\d+[\d,.]*|[A-Za-z]+(?:'[A-Za-z]+)?|[가-힣]+|[\u3040-\u30FF\u4E00-\u9FFF]+)"
    _PUNCT_RE = r"[.,!?…:;()\[\]\"'“”‘’—–-]"
    _TOKEN_RE = re.compile(rf"{_URL_RE}|{_EMAIL_RE}|{_WORD_RE}|{_PUNCT_RE}")

    _PUNCT_OR_SYM = re.compile(rf"^(?:{_PUNCT_RE}|[\W_]+)$")

    # 한국어 조사/격/보조 조사, 연결 어미(폴백 PART 판정에 사용)
    _KO_PARTICLES = {
        "은","는","이","가","을","를","에","에서","으로","로","과","와","도",
        "만","처럼","보다","까지","부터","으로서","으로써","밖에","마다","밖에",
        "이며","이고","지만","인데","라서","라면","면서","면서도","거나","거나도",
        "인데도","인데","고","면","으면","지만","는데","는데도"
    }

    def __init__(self, kiwi: Optional[Any] = None, *, prefer_kiwi: Optional[bool] = None):
        self.kiwi = kiwi
        self.prefer_kiwi = True if prefer_kiwi is None else bool(prefer_kiwi)
        # 문장 캐시 / 토큰 캐시
        self._sent_cache: Dict[str, List[str]] = {}
        # 토큰 캐시 키: (text, prefer_kiwi, keep_pos, lemma, drop_punct, pos_filter_tuple)
        self._tok_cache: Dict[Tuple[str, int, int, int, int, Tuple[str, ...]], List[Any]] = {}
        self._max_cache = 8192

    # ---------------- public API ----------------
    def sentences(self, text: str) -> List[str]:
        key = text if isinstance(text, str) else str(text)
        if key in self._sent_cache:
            return self._sent_cache[key]
        sents = split_sentences(text)
        self._cache_put(self._sent_cache, key, sents)
        return sents

    def tokenize(
        self,
        sent: str,
        *,
        keep_pos: bool = False,
        normalize: bool = True,
        lemma: bool = False,
        drop_punct: bool = True,
        pos_filter: Optional[Iterable[str]] = None,
    ) -> List[Any]:
        """
        - keep_pos=True  → (token, coarse_pos) 튜플 반환
        - lemma=True     → (Kiwi) 간단 표제어(동/형용사 종결 '다'만 제거), (폴백) 표면형 유지
        - drop_punct     → 구두점 제거 여부
        - pos_filter     → coarse POS 필터(예: {"ADJ","VERB","NOUN"})
        """
        raw = sent or ""
        text = normalize_text(raw) if normalize else (raw or "")
        pf_tuple: Tuple[str, ...] = tuple(sorted(set(pos_filter))) if pos_filter else tuple()
        ck = (text, int(self.prefer_kiwi), int(keep_pos), int(lemma), int(drop_punct), pf_tuple)
        if ck in self._tok_cache:
            return self._tok_cache[ck][:]

        if self._use_kiwi():
            toks = self._tokenize_with_kiwi(
                text,
                keep_pos=keep_pos,
                lemma=lemma,
                drop_punct=drop_punct,
                pos_filter=set(pos_filter) if pos_filter else None,
            )
        else:
            toks = self._tokenize_fallback(
                text,
                keep_pos=keep_pos,
                drop_punct=drop_punct,
                pos_filter=set(pos_filter) if pos_filter else None,
            )

        self._cache_put(self._tok_cache, ck, toks)
        return toks

    def tokenize_many(
        self,
        sents: Iterable[str],
        *,
        keep_pos: bool = True,
        normalize: bool = True,
        lemma: bool = False,
        drop_punct: bool = True,
        pos_filter: Optional[Iterable[str]] = None,
    ) -> List[List[Any]]:
        """
        문장 리스트를 배치로 토크나이즈.
        - 캐시를 최대한 활용(동일 옵션 키로 문장별 캐시 조회)
        - 반환 형식: List[ List[token] ] 또는 List[ List[(token, pos)] ]
        """
        results: List[List[Any]] = []
        pf_tuple: Tuple[str, ...] = tuple(sorted(set(pos_filter))) if pos_filter else tuple()
        pf_set: Optional[Set[str]] = set(pf_tuple) if pf_tuple else None

        # Kiwi가 배치 API를 제공하지 않아도 캐시/루프 최적화로 오버헤드 최소화
        for raw in sents:
            sent = raw or ""
            text = normalize_text(sent) if normalize else (sent or "")
            ck = (text, int(self.prefer_kiwi), int(keep_pos), int(lemma), int(drop_punct), pf_tuple)
            cached = self._tok_cache.get(ck)
            if cached is not None:
                results.append(cached[:])
                continue

            if self._use_kiwi():
                toks = self._tokenize_with_kiwi(
                    text,
                    keep_pos=keep_pos, lemma=lemma, drop_punct=drop_punct, pos_filter=pf_set
                )
            else:
                toks = self._tokenize_fallback(
                    text,
                    keep_pos=keep_pos, drop_punct=drop_punct, pos_filter=pf_set
                )
            self._cache_put(self._tok_cache, ck, toks)
            results.append(toks)
        return results

    def detokenize(self, tokens: List[str]) -> str:
        if not tokens:
            return ""
        s = " ".join(tokens)
        s = re.sub(r"\s+([.,!?…:;])", r"\1", s)
        return s.strip()

    def set_kiwi(self, kiwi: Any, *, prefer_kiwi: Optional[bool] = None) -> None:
        self.kiwi = kiwi
        if prefer_kiwi is not None:
            self.prefer_kiwi = bool(prefer_kiwi)
        self._tok_cache.clear()

    def is_morph_available(self) -> bool:
        return bool(self.kiwi)

    # ---------------- internal helpers ----------------
    def _use_kiwi(self) -> bool:
        return bool(self.kiwi) and bool(self.prefer_kiwi)

    # Kiwi → coarse POS 매핑(일관화: NOUN/VERB/ADJ/ADV/PART [+ PUNCT for 제거])
    @staticmethod
    def _map_pos(tag: Optional[str]) -> str:
        if not tag:
            return "NOUN"
        t = tag.upper()
        if t.startswith(("NN", "NP", "NR", "SN", "SL", "SH")):
            return "NOUN"
        if t.startswith("VA"):                         # 형용사
            return "ADJ"
        if t.startswith(("VV", "VX", "VCP", "VCN")):  # 동사/보조/계사 → VERB로 평탄화
            return "VERB"
        if t.startswith(("MAG", "MAJ")):
            return "ADV"
        if t.startswith(("J", "E", "IC", "X")):       # 조사/어미/감탄사/접사류 → 기능어로 평탄화
            return "PART"
        if t.startswith(("SF", "SP", "SS", "SE", "SO")):
            return "PUNCT"
        return "NOUN"

    # 매우 보수적인 한국어 표제어 추정
    @staticmethod
    def _ko_lemma_guess(form: str, tag: Optional[str]) -> str:
        if not form or not tag:
            return form
        if tag.upper().startswith(("VV", "VA", "VX", "VCP", "VCN")):
            if form.endswith("다") and len(form) >= 2:
                return form[:-1]
        return form

    @staticmethod
    def _is_hangul(token: str) -> bool:
        return bool(re.search(r"[가-힣]", token))

    @staticmethod
    def _is_cjk(token: str) -> bool:
        return bool(re.search(r"[\u3040-\u30FF\u4E00-\u9FFF]", token))

    @staticmethod
    def _endswith_any(tok: str, suffs: Tuple[str, ...]) -> bool:
        return any(tok.endswith(s) for s in suffs)

    def _fallback_coarse_pos(self, tok: str) -> str:
        # 구두점
        if self._PUNCT_OR_SYM.match(tok):
            return "PUNCT"
        # 숫자/URL/Email → 내용어 취급: NOUN
        if tok.isdigit() or re.match(self._URL_RE, tok) or re.match(self._EMAIL_RE, tok):
            return "NOUN"

        # 영문 휴리스틱: -ly(부사), -ing/-ed(동사), 형용사 접미(-ous,-ful,-able,-ive,-ic,-al,-less,-ish)
        lower = tok.lower()
        if re.fullmatch(r"[a-z]+(?:'[a-z]+)?", lower):
            if lower.endswith("ly"):
                return "ADV"
            if lower.endswith(("ing", "ed")):
                return "VERB"
            if lower.endswith(("ous","ful","able","ive","ic","al","less","ish","y")):
                return "ADJ"
            return "NOUN"

        # 한글/일본어/한자 등 동아시아 스크립트
        if self._is_hangul(tok) or self._is_cjk(tok):
            # 조사/어미(짧은 조사/접속 형태) 판정
            if tok in self._KO_PARTICLES or self._endswith_any(tok, ("은","는","이","가","을","를","에","로","과","와")):
                return "PART"
            # 부사형 흔한 끝: -게/-히
            if tok.endswith(("게", "히")) and len(tok) >= 2:
                return "ADV"
            # 동사/형용사 기본형 종결: -다
            if tok.endswith("다") and len(tok) >= 2:
                return "VERB"
            return "NOUN"

        # 그 외는 명사로 평탄화
        return "NOUN"

    def _tokenize_with_kiwi(
        self,
        text: str,
        *,
        keep_pos: bool,
        lemma: bool,
        drop_punct: bool,
        pos_filter: Optional[Set[str]],
    ) -> List[Any]:
        try:
            seq = self.kiwi.tokenize(text)
        except Exception:
            return self._tokenize_fallback(text, keep_pos=keep_pos, drop_punct=drop_punct, pos_filter=pos_filter)

        want = set(pos_filter) if pos_filter else None
        out: List[Any] = []
        for tk in seq:
            form = tk.form.strip()
            if not form:
                continue
            coarse = self._map_pos(getattr(tk, "tag", None))
            if drop_punct and coarse == "PUNCT":
                continue
            if want and coarse not in want:
                continue
            token = self._ko_lemma_guess(form, getattr(tk, "tag", None)) if lemma else form
            out.append((token, coarse) if keep_pos else token)
        return out

    def _tokenize_fallback(
        self,
        text: str,
        *,
        keep_pos: bool,
        drop_punct: bool,
        pos_filter: Optional[Set[str]],
    ) -> List[Any]:
        want = set(pos_filter) if pos_filter else None
        out: List[Any] = []
        for m in self._TOKEN_RE.finditer(text):
            tok = (m.group(0) or "").strip()
            if not tok:
                continue
            coarse = self._fallback_coarse_pos(tok)
            if drop_punct and coarse == "PUNCT":
                continue
            if want and coarse not in want:
                continue
            out.append((tok, coarse) if keep_pos else tok)
        return out

    def _cache_put(self, cache: Dict[Any, Any], key: Any, value: Any) -> None:
        cache[key] = value
        if len(cache) > self._max_cache:
            # 간단한 reset(필요시 LRU로 교체 가능)
            cache.clear()


# -----------------------------------------------------------------------------
# Evidence Gate
# -----------------------------------------------------------------------------
class EvidenceGate:
    """
    - allow(): 기존과 동일한 고정 임계 통과 판정
    - allow_adaptive(): 신호밀도/텍스트길이/타입지배/caps를 고려한 적응형 판정
    - aggregate(): 타입별 합계/절대합/카운트 집계
    """
    __slots__ = ("min_types", "min_sum", "eps")

    def __init__(self, min_types: int, min_sum: float, eps: float = 1e-8):
        self.min_types = max(1, int(min_types))
        self.min_sum = float(max(0.0, min_sum))
        self.eps = float(max(0.0, eps))

    @staticmethod
    def _sanitize(evidence: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        out: List[Tuple[str, float]] = []
        for item in evidence or []:
            if isinstance(item, dict) and "type" in item:
                t, w = item.get("type"), item.get("weight", item.get("w", 0.0))
            else:
                try:
                    t, w = item  # type: ignore
                except Exception:
                    continue
            if t is None:
                continue
            try:
                w = float(w)
                if w != w:  # NaN guard
                    continue
            except Exception:
                continue
            out.append((str(t), w))
        return out

    def aggregate(
        self, evidence: List[Tuple[str, float]]
    ) -> Tuple[Dict[str, Dict[str, float]], float, float, int]:
        agg: Dict[str, Dict[str, float]] = {}
        total = 0.0
        total_abs = 0.0
        types_count = 0
        for t, w in self._sanitize(evidence):
            d = agg.get(t)
            if d is None:
                d = {"sum": 0.0, "sum_abs": 0.0, "count": 0.0}
                agg[t] = d
                types_count += 1
            d["sum"] += w
            d["sum_abs"] += abs(w)
            d["count"] += 1.0
            total += w
            total_abs += abs(w)
        return agg, total, total_abs, types_count

    def allow(
        self,
        evidence: List[Tuple[str, float]],
        *,
        min_types: Optional[int] = None,
        min_sum: Optional[float] = None,
    ) -> Tuple[bool, Dict[str, Any]]:
        agg, total, total_abs, _ = self.aggregate(evidence)
        m_types = self.min_types if min_types is None else max(1, int(min_types))
        m_sum = self.min_sum if min_sum is None else float(max(0.0, min_sum))
        strong_types = sum(1 for v in agg.values() if v["sum_abs"] > self.eps)
        passed = (strong_types >= m_types) and (total_abs >= max(self.eps, m_sum))
        return passed, {
            "types": {k: v["sum_abs"] for k, v in agg.items()},
            "types_detail": agg,
            "total": total,
            "total_abs": total_abs,
            "strong_types": strong_types,
            "min_types": m_types,
            "min_sum": m_sum,
        }

    # -------------------- NEW: Adaptive Gate --------------------
    def allow_adaptive(
        self,
        evidence: List[Tuple[str, float]],
        *,
        cfg: Any = None,                       # AnalyzerConfig or None
        signal_density: Optional[float] = None, # 0~1; 문장-윈도우 기준 신호 밀도(없으면 추정)
        text_len: Optional[int] = None,         # 현재 문장 길이(문자/토큰 수 등)
        min_types: Optional[int] = None,        # base override
        min_sum: Optional[float] = None,        # base override
        caps: Optional[Dict[str, float]] = None,# 타입별 최대 기여 캡(예: {"lex": cfg.lex_evidence_cap})
        max_type_frac: Optional[float] = None,  # 단일 타입 지배 상한(기본 0.8)
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        적응형 통과 판단:
          - 텍스트가 짧거나(signal_density↓) 약신호면 min_sum 완화
          - 단일 타입 지배 시(total_abs의 max_type_frac 초과) 효율을 낮춰 균형 확보
          - caps로 특정 타입 기여 상한(예: lex)을 안전하게 제한
        """
        agg, total, total_abs, _ = self.aggregate(evidence)
        # 1) 기본 임계
        base_min_types = self.min_types if min_types is None else max(1, int(min_types))
        base_min_sum = self.min_sum if min_sum is None else float(max(0.0, min_sum))

        # 2) 신호 밀도/텍스트 길이 추정
        if signal_density is None:
            # 간단 추정: 타입 수와 개별 evidences 수 기반
            evid_count = sum(int(v["count"]) for v in agg.values())
            distinct_types = len(agg)
            signal_density = max(0.0, min(1.0, 0.25 * distinct_types + 0.1 * evid_count))
        txt_len = max(0, int(text_len or 0))

        # 3) 타입별 caps 적용(예: lex 상한)
        if caps:
            for t, cap in caps.items():
                if t in agg and cap is not None:
                    agg[t]["sum_abs"] = float(min(agg[t]["sum_abs"], float(cap)))
        # 재계산
        total_abs_capped = sum(v["sum_abs"] for v in agg.values())

        # 4) 타입 지배 상한(단일 타입이 전체를 먹는 경우 억제)
        dom_cap = float(max_type_frac if max_type_frac is not None else 0.8)
        dom_cap = 0.5 if dom_cap < 0.5 else (0.95 if dom_cap > 0.95 else dom_cap)
        eff_total_abs = 0.0
        types_eff: Dict[str, float] = {}
        for t, v in agg.items():
            cap_val = dom_cap * total_abs_capped if total_abs_capped > self.eps else v["sum_abs"]
            eff = min(v["sum_abs"], cap_val)
            types_eff[t] = eff
            eff_total_abs += eff

        # 5) 적응형 임계 계산 (개선)
        adapt_min_sum_weak = getattr(cfg, "adaptive_min_sum_weak", 0.7) if cfg is not None else 0.7
        d = float(signal_density)

        # 강신호의 경우 기존처럼 강화(+10%)하지 않고 base 유지
        if d < 0.25 or txt_len < 20:
            m_sum_eff = max(adapt_min_sum_weak, getattr(cfg, "trans_auto_default", 0.3) if cfg is not None else 0.3)
            m_types_eff = max(1, base_min_types)
        elif d < 0.6:
            m_sum_eff = min(base_min_sum * 0.95, base_min_sum)  # 소폭 보정 또는 그대로
            m_types_eff = base_min_types
        else:
            m_sum_eff = base_min_sum  # ★ 강신호는 강화하지 않음
            m_types_eff = base_min_types

        # 보조 규칙: intensity+transition 동시 존재 → min_sum 하한 0.8 보장
        has_intensity = ("intensity" in agg and agg["intensity"]["sum_abs"] > self.eps)
        has_transition = ("transition" in agg and agg["transition"]["sum_abs"] > self.eps)
        if has_intensity and has_transition:
            m_sum_eff = max(m_sum_eff, 0.8)

        # 6) 최종 판정
        strong_types_eff = sum(1 for v in types_eff.values() if v > self.eps)
        passed = (strong_types_eff >= m_types_eff) and (eff_total_abs >= max(self.eps, m_sum_eff))

        # 7) 품질지표(디버깅/튜닝 보조)
        quality = 0.0
        if m_sum_eff > self.eps:
            quality = min(1.0, (eff_total_abs / m_sum_eff)) * min(1.0, strong_types_eff / float(m_types_eff))

        info = {
            "types": types_eff,                  # dom-cap/caps 반영 후 타입별 절대합
            "types_detail": agg,                 # 원본 집계(캡 반영 전 sum_abs는 caps로 덮였을 수 있음)
            "total": total,
            "total_abs_raw": total_abs,
            "total_abs_capped": total_abs_capped,
            "total_abs_eff": eff_total_abs,      # dom-cap 반영 후 합
            "strong_types": strong_types_eff,
            "min_types": m_types_eff,
            "min_sum": m_sum_eff,
            "signal_density": d,
            "text_len": txt_len,
            "dominance_cap": dom_cap,
            "quality": quality,                  # 0~1, 높을수록 근거가 임계 대비 충분
        }
        return passed, info


# -----------------------------------------------------------------------------
# Emotions Index dataclass
# -----------------------------------------------------------------------------
@dataclass(slots=True)
class EmotionsIndex:
    # Core lexicons
    intensity_examples_low:  List[str] = field(default_factory=list)
    intensity_examples_med:  List[str] = field(default_factory=list)
    intensity_examples_high: List[str] = field(default_factory=list)

    context_low:  List[str] = field(default_factory=list)
    context_med:  List[str] = field(default_factory=list)
    context_high: List[str] = field(default_factory=list)

    # stage → phrases (e.g., {"trigger":[...], "development":[...], "peak":[...], "aftermath":[...]})
    context_progressions: Dict[str, List[str]] = field(default_factory=dict)

    key_phrases: List[str] = field(default_factory=list)

    amplifiers:  List[str] = field(default_factory=list)
    diminishers: List[str] = field(default_factory=list)

    sentiment_combinations: List[List[str]] = field(default_factory=list)
    transitions: List[Dict[str, Any]] = field(default_factory=list)

    defenses: Dict[str, List[str]] = field(default_factory=dict)
    biases:   Dict[str, List[str]] = field(default_factory=dict)

    pos_markers: List[str] = field(default_factory=list)
    neg_markers: List[str] = field(default_factory=list)

    # NEW: Heuristic lexicons (prefer JSON → fallback to config → fallback to baked defaults)
    adversatives: List[str] = field(default_factory=list)  # e.g., "하지만", "그러나", ...
    ko_pos_stems: List[str] = field(default_factory=list)  # e.g., "좋", "기쁘", ...
    ko_neg_stems: List[str] = field(default_factory=list)  # e.g., "불안", "슬프", ...

    # NEW: runtime caches (compiled matchers / compiled transition regex)
    matchers: Dict[str, Any] = field(default_factory=dict, repr=False)
    compiled_transitions: List[Dict[str, Any]] = field(default_factory=list, repr=False)

    def any_loaded(self) -> bool:
        return any([
            self.intensity_examples_low, self.intensity_examples_med, self.intensity_examples_high,
            self.context_low, self.context_med, self.context_high,
            self.key_phrases, self.amplifiers, self.diminishers,
            self.transitions, self.defenses, self.biases, self.sentiment_combinations
        ])

    def stats(self) -> Dict[str, int]:
        return {
            "intensity_low":  len(self.intensity_examples_low),
            "intensity_med":  len(self.intensity_examples_med),
            "intensity_high": len(self.intensity_examples_high),
            "context_low":    len(self.context_low),
            "context_med":    len(self.context_med),
            "context_high":   len(self.context_high),
            "progression_stages": sum(len(v) for v in self.context_progressions.values()),
            "key_phrases":    len(self.key_phrases),
            "amplifiers":     len(self.amplifiers),
            "diminishers":    len(self.diminishers),
            "sentiment_combinations": len(self.sentiment_combinations),
            "transitions":    len(self.transitions),
            "defense_terms":  sum(len(v) for v in self.defenses.values()),
            "bias_terms":     sum(len(v) for v in self.biases.values()),
            "pos_markers":    len(self.pos_markers),
            "neg_markers":    len(self.neg_markers),
            "adversatives":   len(self.adversatives),
            "ko_pos_stems":   len(self.ko_pos_stems),
            "ko_neg_stems":   len(self.ko_neg_stems),
        }

    # -------------------------------------------------------------------------
    # Build regex matchers & compiled transitions
    # -------------------------------------------------------------------------
    def build_matchers(self, cfg: "AnalyzerConfig") -> None:
        """
        - RegexShardMatcher / Aho 토글(strategy)은 cfg.matcher_strategy로 제어(없으면 regex_shard).
        - score_sentence()의 FX(cat)와 키가 1:1로 맞도록 **alias 키**까지 모두 생성.
        - progression 단계별 매처도 준비(추후 FX로 바로 조회 가능).
        """
        try:
            _ = RegexShardMatcher  # noqa
        except Exception:
            # 안전 폴백
            self.matchers = {}
            self.compiled_transitions = []
            try:
                logger.warning("[index] RegexShardMatcher unavailable; FX() will fallback to naive find_all.")
            except Exception:
                pass
            return

        # 대형 어휘 자동 전략 전환(샤드↔Aho)
        total_terms = sum(len(getattr(self, k, []) or []) for k in [
            "key_phrases", "context_low", "context_med", "context_high",
            "intensity_examples_low", "intensity_examples_med", "intensity_examples_high"
        ])
        strategy = getattr(cfg, "matcher_strategy", "regex_shard")
        if strategy == "regex_shard" and total_terms >= 50000:
            strategy = "aho"  # pyahocorasick 있으면 사용, 없으면 RegexShardMatcher 내부에서 폴백

        shard = int(getattr(cfg, "regex_shard_size", 512))

        def _m(phrases: List[str], *, boundary=False):
            return RegexShardMatcher(
                phrases or [], shard_size=shard, boundary=boundary, ignore_space=True, strategy=strategy
            )

        m: Dict[str, Any] = {}

        # ---- intensity/context (alias: intensity_examples_*  &  intensity_*) ----
        m["intensity_examples_low"]  = _m(self.intensity_examples_low)
        m["intensity_examples_med"]  = _m(self.intensity_examples_med)
        m["intensity_examples_high"] = _m(self.intensity_examples_high)
        # aliases
        m["intensity_low"]  = m["intensity_examples_low"]
        m["intensity_med"]  = m["intensity_examples_med"]
        m["intensity_high"] = m["intensity_examples_high"]

        m["context_low"]  = _m(self.context_low)
        m["context_med"]  = _m(self.context_med)
        m["context_high"] = _m(self.context_high)

        # ---- misc lexicons ----
        m["key_phrases"] = _m(self.key_phrases)
        m["amplifiers"]  = _m(self.amplifiers)
        m["diminishers"] = _m(self.diminishers)

        m["pos_markers"] = _m(self.pos_markers)
        m["neg_markers"] = _m(self.neg_markers)

        # ---- NEW: heuristics from JSON/config ----
        m["adversatives"] = _m(self.adversatives)
        m["ko_pos_stems"] = _m(self.ko_pos_stems)
        m["ko_neg_stems"] = _m(self.ko_neg_stems)

        # ---- NEW: progression stages matchers (trigger/development/peak/aftermath) ----
        prog = self._normalized_progressions()
        m["progression_trigger"]    = _m(prog.get("trigger", []))
        m["progression_development"] = _m(prog.get("development", []))
        m["progression_peak"]       = _m(prog.get("peak", []))
        m["progression_aftermath"]  = _m(prog.get("aftermath", []))

        # 합본(all)도 편의상 제공
        all_prog = list(dict.fromkeys(
            (prog.get("trigger") or []) +
            (prog.get("development") or []) +
            (prog.get("peak") or []) +
            (prog.get("aftermath") or [])
        ))
        m["progression_all"] = _m(all_prog)

        self.matchers = m

        # ---- transitions: triggers → OR regex (ignore_space) ----
        compiled: List[Dict[str, Any]] = []
        for t in self.transitions or []:
            trigs = t.get("triggers") or []
            if not trigs:
                continue
            parts = []
            norms = []
            for s in trigs:
                s2 = normalize_text(s)
                if not s2:
                    continue
                s2 = re.sub(r"\s+", "", s2)
                parts.append(re.escape(s2))
                norms.append(s2)
            if not parts:
                continue
            rx = re.compile("|".join(parts))
            compiled.append({
                "rx": rx,
                "normalized_triggers": norms,
                "shift_point": t.get("shift_point"),
                "intensity_change": t.get("intensity_change"),
                "triggers": trigs,
            })
        self.compiled_transitions = compiled

        try:
            st = self.stats()
            logger.info("[index] built (intensity=%d, context=%d, key=%d, combos=%d, transitions=%d, defenses=%d, biases=%d)",
                        st["intensity_low"] + st["intensity_med"] + st["intensity_high"],
                        st["context_low"] + st["context_med"] + st["context_high"],
                        st["key_phrases"], st["sentiment_combinations"], st["transitions"],
                        st["defense_terms"], st["bias_terms"])
        except Exception:
            pass

    # NEW: 라벨 매처 자동 빌드(활성화)
    def build_label_matchers(self, cfg) -> None:
        """
        label_lexicons가 존재하면 label_matchers를 자동으로 구성해
        라벨 기반 스코어가 항상 동작하도록 준비합니다.
        """
        if not hasattr(self, "label_lexicons") or not isinstance(getattr(self, "label_lexicons"), dict):
            self.label_matchers = {}
            return
        strategy = getattr(cfg, "matcher_strategy", "regex_shard")
        shard    = int(getattr(cfg, "regex_shard_size", 512))
        m = {}
        for label, terms in getattr(self, "label_lexicons").items():
            m[label] = RegexShardMatcher(terms or [], shard_size=shard, boundary=True, ignore_space=True, strategy=strategy)
        self.label_matchers = m

    # 내부: progression 키 표준화 + 하향식 보완
    def _normalized_progressions(self) -> Dict[str, List[str]]:
        """
        스테이지 키를 표준화하고, 비어있는 단계는 ‘하향식’으로 보완:
        - trigger    ← adversatives + context_low (없으면 [])
        - development← context_med
        - peak       ← context_high
        - aftermath  ← key_phrases (없으면 diminishers)
        """
        raw = self.context_progressions or {}
        out = {
            "trigger":     list(raw.get("trigger")     or []),
            "development": list(raw.get("development") or []),
            "peak":        list(raw.get("peak")        or []),
            "aftermath":   list(raw.get("aftermath")   or []),
        }

        def _dedup(x: List[str]) -> List[str]:
            return list(dict.fromkeys([s for s in x if s]))

        if not out["trigger"]:
            out["trigger"] = _dedup((self.adversatives or []) + (self.context_low or []))
        if not out["development"]:
            out["development"] = _dedup(self.context_med or [])
        if not out["peak"]:
            out["peak"] = _dedup(self.context_high or [])
        if not out["aftermath"]:
            # 여유: 키프레이즈가 과하면 diminishers 우선으로 교체해도 됨
            base = (self.key_phrases or [])
            if not base:
                base = (self.diminishers or [])
            out["aftermath"] = _dedup(base)

        # 최종 중복 제거
        for k in list(out.keys()):
            out[k] = _dedup(out[k])
        return out


# -----------------------------------------------------------------------------
# Helpers to coerce JSON → lists/dicts safely
# -----------------------------------------------------------------------------
def _as_list(x: Any) -> List[str]:
    if not x:
        return []
    if isinstance(x, str):
        return [x]
    if isinstance(x, (list, tuple, set)):
        out: List[str] = []
        for v in x:
            if isinstance(v, str):
                out.append(v)
        return out
    return []

def _as_dict_of_lists(x: Any) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    if not isinstance(x, dict):
        return out
    for k, v in x.items():
        out[str(k)] = _as_list(v)
    return out


# -----------------------------------------------------------------------------
# Build EmotionsIndex from emotions_data + cfg, with schema validation & labels/hierarchy support
# -----------------------------------------------------------------------------
def prepare_emotions_index(emotions_data: Dict[str, Any], cfg: "AnalyzerConfig") -> EmotionsIndex:
    """
    EMOTIONS.JSON을 읽어 EmotionsIndex를 구성 + 간단 스키마 검증.
    - 필수(코어) 슬롯 중 1개 이상 존재해야 함:
        * key_phrases, context_low/med/high, intensity_* , transitions, (또는) label_lexicons
    - 라벨 사전/계층 구조 로드(있으면): label_lexicons, emotion_labels, hierarchy|global.hierarchy
    - JSON > cfg > baked 순으로 역접/극성 어간 폴백
    - context_progressions 키 정규화(trigger/development/peak/aftermath)
    - 마지막에 build_matchers(cfg) + (가능하면) build_label_matchers(cfg) 호출

    검증 동작 모드(cfg.schema_validation_mode):
        "off"  : 검증 끔
        "warn" : 경고 로그만(기본)
        "raise": 오류 시 예외 발생
    """
    # ---------- helpers ----------
    def _dedup_norm_list(xs: Iterable[Any]) -> List[str]:
        out: List[str] = []
        seen: Set[str] = set()
        for x in xs or []:
            s = normalize_text(str(x)).strip()
            if not s:
                continue
            if s not in seen:
                seen.add(s)
                out.append(s)
        return out

    def _as_list_safe(x: Any) -> List[str]:
        return _dedup_norm_list(_as_list(x))

    def _as_dict_of_lists_safe(x: Any) -> Dict[str, List[str]]:
        base = _as_dict_of_lists(x)
        return {str(k): _dedup_norm_list(v) for k, v in (base or {}).items()}

    def _emit_issue(kind: str, msg: str) -> None:
        try:
            if kind == "warn":
                logger.warning(msg)
            else:
                logger.error(msg)
        except Exception:
            pass

    def _handle_issues_or_raise(prefix: str, errors: List[str], warnings: List[str]) -> None:
        mode = str(getattr(cfg, "schema_validation_mode", "warn")).lower()
        if mode == "off":
            return
        if warnings:
            for w in warnings:
                _emit_issue("warn", f"[{prefix}] {w}")
        if errors:
            if mode == "raise":
                raise ValueError(f"{prefix} schema errors: " + "; ".join(errors))
            for e in errors:
                _emit_issue("error", f"[{prefix}] {e}")

    # ---------- start ----------
    data = emotions_data or {}
    errors: List[str] = []
    warns: List[str] = []

    # 최상위 허용 키(스키마 힌트) - 라벨 관련 키 포함
    allowed_top = {
        "intensity_examples_low", "intensity_examples_med", "intensity_examples_high",
        "context_low", "context_med", "context_high",
        "context_progressions",
        "key_phrases", "amplifiers", "diminishers",
        "sentiment_combinations",
        "transitions",
        "defenses", "biases",
        "pos_markers", "neg_markers",
        "global",
        # NEW:
        "label_lexicons", "emotion_labels", "hierarchy",
    }
    unknown_top = [k for k in data.keys() if k not in allowed_top]
    if unknown_top:
        warns.append(f"Unknown top-level keys ignored: {unknown_top}")

    # global 블록 파싱
    g = data.get("global")
    if g is None:
        g = {}
    elif not isinstance(g, dict):
        warns.append("`global` should be an object; got non-dict → ignored")
        g = {}

    patterns = g.get("linguistic_patterns") or {}
    if patterns and not isinstance(patterns, dict):
        warns.append("`global.linguistic_patterns` should be an object; got non-dict → ignored")
        patterns = {}

    markers = g.get("markers") or {}
    if markers and not isinstance(markers, dict):
        warns.append("`global.markers` should be an object; got non-dict → ignored")
        markers = {}

    # 리스트형 슬롯 정규화
    inten_low  = _as_list_safe(data.get("intensity_examples_low"))
    inten_med  = _as_list_safe(data.get("intensity_examples_med"))
    inten_high = _as_list_safe(data.get("intensity_examples_high"))

    ctx_low  = _as_list_safe(data.get("context_low"))
    ctx_med  = _as_list_safe(data.get("context_med"))
    ctx_high = _as_list_safe(data.get("context_high"))

    key_phrases = _as_list_safe(data.get("key_phrases"))
    amplifiers  = _as_list_safe(data.get("amplifiers"))
    diminishers = _as_list_safe(data.get("diminishers"))

    pos_markers = _as_list_safe(data.get("pos_markers"))
    neg_markers = _as_list_safe(data.get("neg_markers"))

    # dict-of-lists 슬롯
    defenses = data.get("defenses") or {}
    if defenses and not isinstance(defenses, dict):
        warns.append("`defenses` should be an object of lists; got non-dict → coerced to empty")
        defenses = {}
    defenses = {str(k): _as_list_safe(v) for k, v in (defenses or {}).items()}

    biases = data.get("biases") or {}
    if biases and not isinstance(biases, dict):
        warns.append("`biases` should be an object of lists; got non-dict → coerced to empty")
        biases = {}
    biases = {str(k): _as_list_safe(v) for k, v in (biases or {}).items()}

    # context_progressions 정규화
    context_progressions = _as_dict_of_lists_safe(data.get("context_progressions"))
    if context_progressions:
        norm_prog: Dict[str, List[str]] = {}
        for k, v in context_progressions.items():
            kk = str(k).strip().lower()
            if kk in ("trigger", "development", "peak", "aftermath"):
                norm_prog[kk] = v
            else:
                warns.append(f"Ignored unknown progression stage: {k}")
        context_progressions = norm_prog

    # transitions
    transitions_raw = data.get("transitions") or []
    transitions: List[Dict[str, Any]] = []
    if transitions_raw and not isinstance(transitions_raw, list):
        warns.append("`transitions` should be a list; got non-list → coerced to empty")
        transitions_raw = []
    if isinstance(transitions_raw, list):
        for i, t in enumerate(transitions_raw):
            if not isinstance(t, dict):
                warns.append(f"`transitions[{i}]` should be an object; got non-dict → skipped")
                continue
            trigs = _as_list_safe(t.get("triggers"))
            if not trigs:
                warns.append(f"`transitions[{i}].triggers` is empty → skipped")
                continue
            t_copy = dict(t)
            t_copy["triggers"] = trigs
            transitions.append(t_copy)

    # sentiment_combinations
    sentiment_combinations = data.get("sentiment_combinations") or []
    if sentiment_combinations and not isinstance(sentiment_combinations, list):
        warns.append("`sentiment_combinations` should be a list; got non-list → coerced to empty")
        sentiment_combinations = []

    # ---------- 라벨 사전 & 계층 구조 로드 (NEW) ----------
    label_lexicons_raw = data.get("label_lexicons") or g.get("label_lexicons") or {}
    if label_lexicons_raw and not isinstance(label_lexicons_raw, dict):
        warns.append("`label_lexicons` should be an object; got non-dict → coerced to empty")
        label_lexicons_raw = {}
    label_lexicons: Dict[str, List[str]] = {}
    for label, terms in label_lexicons_raw.items():
        lbl = str(label).strip()
        if not lbl:
            continue
        label_lexicons[lbl] = _as_list_safe(terms)

    emotion_labels_raw = data.get("emotion_labels") or g.get("emotion_labels") or {}
    emotion_labels: Dict[str, Dict[str, Any]] = {}
    if isinstance(emotion_labels_raw, dict):
        for label, meta in emotion_labels_raw.items():
            lbl = str(label).strip()
            if not lbl:
                continue
            if isinstance(meta, dict):
                emotion_labels[lbl] = meta
            else:
                emotion_labels[lbl] = {"description": str(meta)}

    hierarchy_raw = (data.get("hierarchy") or g.get("hierarchy") or g.get("emotion_hierarchy") or {})
    hierarchy: Dict[str, List[str]] = {}
    if isinstance(hierarchy_raw, dict):
        for parent, children in hierarchy_raw.items():
            p = str(parent).strip()
            if not p:
                continue
            hierarchy[p] = _as_list_safe(children)
    elif isinstance(hierarchy_raw, list):
        warns.append("`hierarchy` is a list; attempting auto-grouping")
        hierarchy = _auto_group_hierarchy(hierarchy_raw)

    # ---------- 최소 코어 슬롯 존재 여부 검증 ----------
    core_present = any([
        bool(key_phrases),
        bool(ctx_low or ctx_med or ctx_high),
        bool(inten_low or inten_med or inten_high),
        bool(transitions),
        bool(label_lexicons),  # 라벨 사전도 코어로 인정
    ])
    if not core_present:
        errors.append(
            "Core slots are all empty (need at least one of: key_phrases, context_*, intensity_*, transitions, label_lexicons)."
        )

    # 1차 검증 결과 출력
    _handle_issues_or_raise("EMOTIONS.JSON", errors, warns)

    # ---------- EmotionsIndex 생성 ----------
    idx = EmotionsIndex(
        intensity_examples_low=inten_low,
        intensity_examples_med=inten_med,
        intensity_examples_high=inten_high,
        context_low=ctx_low,
        context_med=ctx_med,
        context_high=ctx_high,
        context_progressions=context_progressions,
        key_phrases=key_phrases,
        amplifiers=amplifiers,
        diminishers=diminishers,
        sentiment_combinations=sentiment_combinations,
        transitions=transitions,
        defenses=defenses,
        biases=biases,
        pos_markers=pos_markers,
        neg_markers=neg_markers,
    )

    # 라벨/계층 필드는 생성 후 주입(생성자 호환성 보장)
    if label_lexicons:
        setattr(idx, "label_lexicons", label_lexicons)
    if emotion_labels:
        setattr(idx, "emotion_labels", emotion_labels)
    if hierarchy:
        if hasattr(idx, "set_hierarchy") and callable(getattr(idx, "set_hierarchy")):
            idx.set_hierarchy(hierarchy)  # type: ignore[attr-defined]
        else:
            setattr(idx, "hierarchy", hierarchy)
        # Config에도 전달(있을 때만)
        if hasattr(cfg, "emotion_hierarchy"):
            cfg.emotion_hierarchy = hierarchy
    elif getattr(cfg, "derive_hierarchy_from_categories", False) and label_lexicons:
        inferred = _infer_hierarchy_from_labels(label_lexicons, emotion_labels)
        if inferred:
            if hasattr(idx, "set_hierarchy") and callable(getattr(idx, "set_hierarchy")):
                idx.set_hierarchy(inferred)  # type: ignore[attr-defined]
            else:
                setattr(idx, "hierarchy", inferred)
            if hasattr(cfg, "emotion_hierarchy"):
                cfg.emotion_hierarchy = inferred

    # ---------- JSON → cfg → baked 순으로 역접/극성 어간 폴백 ----------
    json_adversatives = _as_list_safe((patterns or {}).get("adversatives"))
    json_ko_pos = _as_list_safe((markers or {}).get("ko_pos_stems"))
    json_ko_neg = _as_list_safe((markers or {}).get("ko_neg_stems"))

    if json_adversatives:
        idx.adversatives = json_adversatives
    else:
        cfg_adv = getattr(cfg, "adversative_triggers", None) or []
        idx.adversatives = _dedup_norm_list(cfg_adv) if cfg_adv else [
            "하지만", "그렇지만", "그러나", "반면", "반대로", "다만", "으나", "지만", "다가도"
        ]

    if json_ko_pos:
        idx.ko_pos_stems = json_ko_pos
    else:
        cfg_pos = getattr(cfg, "ko_pos_stems", None) or []
        idx.ko_pos_stems = _dedup_norm_list(cfg_pos) if cfg_pos else [
            "좋", "기쁘", "행복", "편안", "만족", "희망", "나아지", "괜찮"
        ]

    if json_ko_neg:
        idx.ko_neg_stems = json_ko_neg
    else:
        cfg_neg = getattr(cfg, "ko_neg_stems", None) or []
        idx.ko_neg_stems = _dedup_norm_list(cfg_neg) if cfg_neg else [
            "슬프", "불안", "우울", "두렵", "걱정", "짜증", "화나", "힘들", "나쁘", "나빠"
        ]

    # ---------- 매처/전이 컴파일 ----------
    idx.build_matchers(cfg)

    # ---------- 라벨 매처 빌드 (있으면) ----------
    if hasattr(idx, "build_label_matchers") and callable(getattr(idx, "build_label_matchers")):
        try:
            idx.build_label_matchers(cfg)  # type: ignore[attr-defined]
        except Exception as e:
            _emit_issue("warn", f"[EMOTIONS.JSON] build_label_matchers failed: {e}")

    # ---------- 라벨 커버리지 검증 ----------
    _validate_label_coverage(idx, warns)
    # 2차 경고 출력(라벨 관련)
    _handle_issues_or_raise("LABELS", [], warns)

    # ---------- 로깅 ----------
    try:
        st = idx.stats()
        logger.info(
            "Index stats: intensity=%d ctx=%d key=%d combos=%d trans=%d defenses=%d biases=%d labels=%d",
            st.get("intensity_low", 0) + st.get("intensity_med", 0) + st.get("intensity_high", 0),
            st.get("context_low", 0) + st.get("context_med", 0) + st.get("context_high", 0),
            st.get("key_phrases", 0),
            st.get("sentiment_combinations", 0),
            st.get("transitions", 0),
            st.get("defense_terms", 0),
            st.get("bias_terms", 0),
            st.get("label_terms", 0),
        )
        # 계층 구조 로깅
        if getattr(idx, "hierarchy", None):
            h = getattr(idx, "hierarchy")
            if isinstance(h, dict):
                logger.info("Hierarchy loaded: %d categories / %d emotions",
                            len(h), sum(len(v) for v in h.values()))
    except Exception:
        pass

    return idx


# ---------- 보조 함수들 ----------
def _auto_group_hierarchy(items: List[Any]) -> Dict[str, List[str]]:
    """리스트 형태의 계층을 자동 그룹화(간단 키워드 매칭)."""
    hierarchy: Dict[str, List[str]] = {"joy": [], "anger": [], "sorrow": [], "pleasure": []}
    keyword_map = {
        "joy": ["기쁨", "즐거", "행복", "만족", "희망", "joy", "happy", "glad"],
        "anger": ["분노", "화", "짜증", "anger", "mad", "irritate", "rage"],
        "sorrow": ["슬픔", "우울", "눈물", "sorrow", "sad", "depress", "grief"],
        "pleasure": ["즐거움", "재미", "흥미", "pleasure", "fun", "interest", "enjoy"],
    }
    for item in items:
        if isinstance(item, dict):
            name = str(item.get("name", "")).lower()
            category = str(item.get("category", "")).lower()
        else:
            name = str(item).lower()
            category = ""
        if category in hierarchy:
            hierarchy[category].append(name)
            continue
        for cat, kws in keyword_map.items():
            if any(kw in name for kw in kws):
                hierarchy[cat].append(name)
                break
    return hierarchy


def _infer_hierarchy_from_labels(
    label_lexicons: Dict[str, List[str]], emotion_labels: Dict[str, Dict[str, Any]]
) -> Dict[str, List[str]]:
    """라벨 정보에서 계층 구조 추론(메타 parent/category 우선, 없으면 키워드)."""
    hierarchy: Dict[str, List[str]] = {}
    # 메타에 parent/category가 있으면 우선 사용
    for label, meta in (emotion_labels or {}).items():
        if isinstance(meta, dict):
            parent = meta.get("parent") or meta.get("category")
            if parent:
                hierarchy.setdefault(str(parent), []).append(label)

    # 없으면 긍/부/중립/혼합의 간단 그룹
    if not hierarchy:
        hierarchy = {"positive": [], "negative": [], "neutral": [], "mixed": []}
        for label in (label_lexicons or {}).keys():
            ll = label.lower()
            if any(w in ll for w in ["pos", "긍정", "기쁨", "happy", "joy"]):
                hierarchy["positive"].append(label)
            elif any(w in ll for w in ["neg", "부정", "슬픔", "sad", "anger"]):
                hierarchy["negative"].append(label)
            elif any(w in ll for w in ["neutral", "중립"]):
                hierarchy["neutral"].append(label)
            else:
                hierarchy["mixed"].append(label)
        hierarchy = {k: v for k, v in hierarchy.items() if v}
    return hierarchy


def _validate_label_coverage(idx: EmotionsIndex, warns: List[str]) -> None:
    """라벨 커버리지/정합성 간단 점검(경고만 추가)."""
    label_lex = getattr(idx, "label_lexicons", {})
    if not isinstance(label_lex, dict) or not label_lex:
        return

    # 라벨별 최소 어휘 수 권장
    for label, terms in label_lex.items():
        n = len(terms or [])
        if n < 5:
            warns.append(f"Label '{label}' has only {n} terms (recommend ≥ 5).")

    # 계층(hierarchy)와 라벨 정합성
    h = getattr(idx, "hierarchy", None)
    if isinstance(h, dict) and h:
        all_emotions = set()
        for children in h.values():
            all_emotions.update(children or [])
        missing = all_emotions - set(label_lex.keys())
        if missing:
            warns.append(f"Emotions in hierarchy missing from label_lexicons: {sorted(missing)}")


# ------------------------------ robust loader (already improved) ------------------------------
def load_emotions_json(path: Optional[str]) -> Dict[str, Any]:
    """
    견고한 로더:
    - path가 없거나/잘못되어도 주변 경로 자동 탐색(현재/부모/환경변수/표준 파일명)
    - JSON5스러운 주석(//, /* */) 및 후행 콤마 제거 후 재시도
    - NDJSON(행 단위 JSON)도 허용하여 list로 수집
    - .txt 안에 JSON 블록만 있어도 추출
    - 루트가 list면 dict로 정규화(primary/name/category/id/label/metadata.name 사용)
    """
    def _strip_comments(s: str) -> str:
        s = re.sub(r"/\*.*?\*/", "", s, flags=re.DOTALL)
        s = re.sub(r"(?m)^\s*//.*?$", "", s)
        return s

    def _remove_trailing_commas(s: str) -> str:
        return re.sub(r",\s*([}\]])", r"\1", s)

    def _normalize_root(data: Any) -> Dict[str, Any]:
        if isinstance(data, dict):
            return data
        if isinstance(data, list):
            norm: Dict[str, Any] = {}
            for i, item in enumerate(data):
                if not isinstance(item, dict):
                    continue
                key = (
                    str(item.get("primary"))
                    or str(item.get("name"))
                    or str(item.get("category"))
                    or str(item.get("id"))
                    or str(item.get("label"))
                    or str(item.get("metadata", {}).get("name"))
                    or str(i)
                )
                norm[key] = item
            logger.info("[emotions] root=list → normalized to dict entries=%d", len(norm))
            return norm
        return {}

    def _try_parse_text(text: str) -> Dict[str, Any]:
        try:
            data = json.loads(text)
            return _normalize_root(data)
        except Exception:
            pass
        try:
            cleaned = _remove_trailing_commas(_strip_comments(text))
            data = json.loads(cleaned)
            return _normalize_root(data)
        except Exception:
            pass
        objs: List[Any] = []
        for line in text.splitlines():
            ln = line.strip()
            if not ln or (not ln.startswith("{") and not ln.startswith("[")):
                continue
            try:
                objs.append(json.loads(ln))
            except Exception:
                continue
        if objs:
            return _normalize_root(objs)
        try:
            first = text.find("{")
            last = text.rfind("}")
            if 0 <= first < last:
                frag = text[first:last+1]
                frag = _remove_trailing_commas(_strip_comments(frag))
                data = json.loads(frag)
                return _normalize_root(data)
        except Exception:
            pass
        return {}

    def _iter_default_candidates() -> List[str]:
        cands = []
        envp = os.environ.get("EMOTIONS_JSON")
        if envp:
            cands.append(envp)
        here = os.path.dirname(os.path.abspath(__file__))
        names = [
            "EMOTIONS.JSON", "emotions.json", "EMOTIONS.json", "emotions.JSON",
            "EMOTIONS.ndjson", "emotions.ndjson",
            "EMOTIONS.txt", "EMOTIONS.JSON(일부).txt",
        ]
        for d in (here, os.path.dirname(here), os.path.dirname(os.path.dirname(here))):
            for n in names:
                cands.append(os.path.join(d, n))
            try:
                for fn in os.listdir(d):
                    if fn.lower().endswith((".json", ".ndjson", ".txt")) and "emotion" in fn.lower():
                        cands.append(os.path.join(d, fn))
            except Exception:
                pass
        seen = set(); uniq = []
        for p in cands:
            if not p:
                continue
            ap = os.path.abspath(p)
            if ap in seen:
                continue
            seen.add(ap)
            if os.path.exists(ap):
                uniq.append(ap)
        return uniq

    def _open_and_parse(p: str) -> Dict[str, Any]:
        try:
            with open(p, "r", encoding="utf-8") as f:
                text = f.read()
            data = _try_parse_text(text)
            if data:
                logger.info("[emotions] loaded: %s", p)
            else:
                logger.warning("[emotions] parse failed: %s", p)
            return data
        except Exception as e:
            logger.warning("[emotions] load error for %s: %s", p, e)
            return {}

    candidates: List[str] = []
    if path:
        if os.path.isdir(path):
            candidates.extend(_iter_default_candidates())
        else:
            candidates.append(path)
    if not candidates:
        candidates = _iter_default_candidates()
    for p in candidates:
        data = _open_and_parse(p)
        if data:
            return data
    logger.warning("[emotions] no usable emotions data found (checked %d candidates)", len(candidates))
    return {}


# ------------------------------ helpers ------------------------------
def _dedup_list(xs: List[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for s in xs:
        s2 = normalize_text(s)
        if s2 and s2 not in seen:
            seen.add(s2)
            out.append(s2)
    return out

def _norm_stage(name: str) -> str:
    n = str(name).strip().lower()
    alias = {
        "onset": "trigger", "start": "trigger", "trigger": "trigger",
        "build": "development", "build-up": "development", "development": "development",
        "climax": "peak", "peak": "peak", "apex": "peak",
        "aftermath": "aftermath", "resolution": "aftermath", "cooldown": "aftermath"
    }
    return alias.get(n, n)

def _norm_intensity(name: str) -> str:
    n = str(name).strip().lower()
    if n in {"low", "lo", "약함", "weak"}:
        return "low"
    if n in {"medium", "med", "mid", "중간"}:
        return "medium"
    if n in {"high", "hi", "강함", "strong"}:
        return "high"
    return n

def _extend_norm(dst: List[str], phrases: Iterable[str]) -> None:
    for p in phrases or []:
        p2 = normalize_text(p)
        if p2:
            dst.append(p2)

def _extract_phrases_like(node: Any, keys: List[str]) -> List[str]:
    out: List[str] = []
    if isinstance(node, dict):
        for k in keys:
            if k in node:
                out.extend(iter_phrases(node[k]) or [])
    return out

def _extract_examples_block(block: Any) -> List[str]:
    return _extract_phrases_like(block, ["intensity_examples", "examples", "phrases", "samples", "keywords", "terms"])

def _extract_variants(node: Any) -> List[str]:
    return _extract_phrases_like(node, ["variations", "variants", "aliases", "synonyms"])

def _extract_list(node: Any) -> List[str]:
    return list(iter_phrases(node) or [])


# ------------------------------ index builder ------------------------------
def build_index(data: Any) -> EmotionsIndex:
    idx = EmotionsIndex()

    # 0) normalize root
    if isinstance(data, list):
        norm: Dict[str, Any] = {}
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                continue
            key = (
                str(item.get("primary")) or str(item.get("name")) or str(item.get("category"))
                or str(item.get("id")) or str(item.get("label")) or str(item.get("metadata", {}).get("name"))
                or str(i)
            )
            norm[key] = item
        logger.info("[emotions] root=list → normalized to dict entries=%d", len(norm))
        data = norm
    if not isinstance(data, dict):
        logger.info("[index] empty index (non-dict root)")
        return idx

    # 1) helpers
    def k(node: Dict[str, Any], *alts: str) -> Any:
        if not isinstance(node, dict):
            return None
        lower = {str(kk).lower(): kk for kk in node.keys()}
        for name in alts:
            if name.lower() in lower:
                return node[lower[name.lower()]]
        return None

    def take_phrases(obj: Any, *keys: str) -> List[str]:
        out: List[str] = []
        if isinstance(obj, dict):
            lower = {str(kk).lower(): kk for kk in obj.keys()}
            for name in keys:
                if name.lower() in lower:
                    out.extend(iter_phrases(obj[lower[name.lower()]]) or [])
        return out

    df_counter: Dict[str, int] = {}
    doc_count = 0
    def add_df(terms: Iterable[str]) -> None:
        nonlocal doc_count
        seen: Set[str] = set()
        for t in terms:
            t2 = normalize_text(t)
            if t2: seen.add(t2)
        if seen: doc_count += 1
        for t in seen:
            df_counter[t] = df_counter.get(t, 0) + 1

    # 2) strict parse
    def handle_node(node: Dict[str, Any], *, count_for_df: bool = True) -> None:
        prof = k(node, "emotion_profile", "emotionProfile", "profile")
        levels = k(prof, "intensity_levels", "intensityLevels", "levels") if isinstance(prof, dict) else None
        if isinstance(levels, dict):
            for key, bucket in levels.items():
                keyn = _norm_intensity(key)
                ex = _extract_examples_block(bucket)
                if keyn == "low":    _extend_norm(idx.intensity_examples_low, ex)
                elif keyn == "medium": _extend_norm(idx.intensity_examples_med, ex)
                elif keyn == "high":   _extend_norm(idx.intensity_examples_high, ex)

        ctxp = k(node, "context_patterns", "contextPatterns", "contexts", "situationPatterns")
        sits = k(ctxp, "situations", "scenarios", "cases") if isinstance(ctxp, dict) else ctxp
        iter_sits = list(sits.values()) if isinstance(sits, dict) else (sits if isinstance(sits, list) else [])
        for s in iter_sits:
            if not isinstance(s, dict): continue
            inten = _norm_intensity(k(s, "intensity", "level") or "")
            phrases = take_phrases(s, "keywords", "phrases", "terms", "tokens") + _extract_variants(s)
            if inten == "low":      _extend_norm(idx.context_low, phrases)
            elif inten == "medium": _extend_norm(idx.context_med, phrases)
            elif inten == "high":   _extend_norm(idx.context_high, phrases)
            prog = k(s, "emotion_progression", "progression", "flow")
            if isinstance(prog, dict):
                for stage, examples in prog.items():
                    st = _norm_stage(stage)
                    idx.context_progressions.setdefault(st, [])
                    _extend_norm(idx.context_progressions[st], _extract_list(examples))

        ling = k(node, "linguistic_patterns", "linguisticPatterns", "language_patterns", "languagePatterns")
        kps: List[str] = []
        if isinstance(ling, dict):
            kps = take_phrases(ling, "key_phrases", "keyPhrases", "phrases", "lexicon", "keywords")
            _extend_norm(idx.key_phrases, kps)
            mods = k(ling, "sentiment_modifiers", "sentimentModifiers", "degree_modifiers", "degreeModifiers", "modifiers")
            if isinstance(mods, dict):
                _extend_norm(idx.amplifiers,  take_phrases(mods, "amplifiers", "intensifiers", "boosters", "강화어"))
                _extend_norm(idx.diminishers, take_phrases(mods, "diminishers", "downtoners", "hedges", "완화어"))
            markers = k(ling, "markers", "sentiment_markers")
            pos = k(markers, "positive_markers", "positive", "pos") if isinstance(markers, dict) else None
            neg = k(markers, "negative_markers", "negative", "neg") if isinstance(markers, dict) else None
            _extend_norm(idx.pos_markers, _extract_list(pos or k(ling, "positive_markers", "positive", "pos")))
            _extend_norm(idx.neg_markers, _extract_list(neg or k(ling, "negative_markers", "negative", "neg")))

        trans_root = k(node, "emotion_transitions", "emotionTransitions", "transitions")
        patterns = trans_root if isinstance(trans_root, list) else (k(trans_root, "patterns", "rules", "list") if isinstance(trans_root, dict) else None)
        trig_union: List[str] = []
        if isinstance(patterns, list):
            for t in patterns:
                if not isinstance(t, dict): continue
                triggers = _extract_list(k(t, "triggers", "cues", "signals", "markers"))
                rec = {
                    "from": k(t, "from", "src", "source"),
                    "to": k(t, "to", "dst", "target"),
                    "triggers": [normalize_text(x) for x in triggers if normalize_text(x)],
                    "shift_point": k(t, "shift_point", "shiftPoint", "pivot"),
                    "intensity_change": k(t, "intensity_change", "intensityChange", "change"),
                }
                if rec["triggers"]:
                    idx.transitions.append(rec)
                    trig_union.extend(rec["triggers"])

        psych = k(node, "psychological_patterns", "psychologicalPatterns", "psychPatterns", "cognitive_patterns")
        if isinstance(psych, dict):
            defs = k(psych, "defense_mechanisms", "defenseMechanisms", "defences", "defenses")
            if isinstance(defs, dict):
                for name, obj in defs.items():
                    idx.defenses.setdefault(str(name), [])
                    _extend_norm(idx.defenses[str(name)], take_phrases(obj, "keywords", "phrases", "signals"))
            biases = k(psych, "emotional_biases", "emotionalBiases", "biases", "cognitive_biases", "cognitiveBiases")
            if isinstance(biases, dict):
                for name, obj in biases.items():
                    idx.biases.setdefault(str(name), [])
                    _extend_norm(idx.biases[str(name)], take_phrases(obj, "keywords", "phrases", "signals"))

        rel = k(node, "related_emotions", "relatedEmotions", "affects", "sentiments")
        if isinstance(rel, dict):
            _extend_norm(idx.pos_markers, _extract_list(k(rel, "positive", "pos", "good", "up")))
            _extend_norm(idx.neg_markers, _extract_list(k(rel, "negative", "neg", "bad", "down")))

        if count_for_df:
            add_df(set(kps) | set(trig_union))

    for _, node in data.items():
        if isinstance(node, dict):
            handle_node(node, count_for_df=True)
    for gkey in ("global", "_global", "shared", "common"):
        g = data.get(gkey)
        if isinstance(g, dict):
            handle_node(g, count_for_df=False)

    # 3) loose scan (only if too sparse)
    if not (idx.intensity_examples_low or idx.intensity_examples_med or idx.intensity_examples_high
            or idx.context_low or idx.context_med or idx.context_high or idx.key_phrases):
        def walk_loose(node: Any, path: List[str]):
            if isinstance(node, dict):
                lower = {str(kk).lower(): kk for kk in node.keys()}
                def take(keys: List[str]) -> List[str]:
                    for nm in keys:
                        if nm.lower() in lower:
                            return list(iter_phrases(node[lower[nm.lower()]]) or [])
                    return []
                _extend_norm(idx.key_phrases, take(["key_phrases", "keyPhrases", "phrases", "terms", "lexicon", "keywords"]))
                _extend_norm(idx.pos_markers, take(["positive_markers", "positive", "pos"]))
                _extend_norm(idx.neg_markers, take(["negative_markers", "negative", "neg"]))
                trigs = take(["triggers", "cues", "signals", "markers"])
                if trigs:
                    idx.transitions.append({"from": None, "to": None,
                                            "triggers": [normalize_text(x) for x in trigs if normalize_text(x)],
                                            "shift_point": None, "intensity_change": None})
                for kx, vx in node.items():
                    walk_loose(vx, path + [str(kx)])
            elif isinstance(node, list):
                for i, vx in enumerate(node):
                    walk_loose(vx, path + [str(i)])
        walk_loose(data, [])
        logger.info("[index] loose-mode applied (fallback)")

    # 4) keyphrase noise filtering + context bridging
    def _basic_ok(t: str) -> bool:
        if not t: return False
        t = normalize_text(t)
        if len(t) < 2: return False
        if re.fullmatch(r"[0-9]+", t): return False
        if re.fullmatch(r"[\W_]+", t): return False
        return True

    max_df_ratio = 0.35
    def _df_ok(t: str) -> bool:
        if doc_count <= 0: return True
        df = df_counter.get(t, 0)
        return (df / doc_count) <= max_df_ratio

    kp_clean = [normalize_text(t) for t in idx.key_phrases if _basic_ok(t)]
    kp_clean = [t for t in kp_clean if _df_ok(t)]
    idx.key_phrases = _dedup_list(kp_clean)

    trigger_union: Set[str] = set()
    for tr in idx.transitions:
        for tg in tr.get("triggers") or []:
            tt = normalize_text(tg)
            if tt: trigger_union.add(tt)
    if trigger_union:
        _extend_norm(idx.context_med, sorted(trigger_union))

    context_total = len(idx.context_low) + len(idx.context_med) + len(idx.context_high)
    if context_total < max(50, int(0.05 * len(idx.key_phrases) + 1)):
        def _idf(term: str) -> float:
            if doc_count <= 0: return 1.0
            df = df_counter.get(term, 0) + 1
            return math.log((doc_count + 1) / df) + 1.0
        candidates = list(set(idx.key_phrases).union(trigger_union))
        ranked = sorted(candidates, key=_idf, reverse=True)
        take_n = min(200, len(ranked))
        _extend_norm(idx.context_med, ranked[:take_n])

    # 5) cleanup & stats
    for fld in [
        "intensity_examples_low", "intensity_examples_med", "intensity_examples_high",
        "context_low", "context_med", "context_high", "key_phrases",
        "amplifiers", "diminishers", "pos_markers", "neg_markers"
    ]:
        setattr(idx, fld, _dedup_list(getattr(idx, fld)))
    for stg, arr in list(idx.context_progressions.items()):
        idx.context_progressions[stg] = _dedup_list(arr)
        if not idx.context_progressions[stg]:
            del idx.context_progressions[stg]
    cleaned: List[List[str]] = []
    for combo in idx.sentiment_combinations:
        seen_in: Set[str] = set(); tmp: List[str] = []
        for t in combo:
            t2 = normalize_text(t)
            if t2 and t2 not in seen_in:
                seen_in.add(t2); tmp.append(t2)
        if tmp: cleaned.append(tmp)
    idx.sentiment_combinations = cleaned
    for kx in list(idx.defenses.keys()):
        idx.defenses[kx] = _dedup_list(idx.defenses[kx])
        if not idx.defenses[kx]: del idx.defenses[kx]
    for kx in list(idx.biases.keys()):
        idx.biases[kx] = _dedup_list(idx.biases[kx])
        if not idx.biases[kx]: del idx.biases[kx]

    st = idx.stats()
    if idx.any_loaded():
        logger.info(
            "[index] built (intensity=%d, context=%d, key_phrases=%d, combos=%d, transitions=%d, defenses=%d, biases=%d)",
            st["intensity_low"] + st["intensity_med"] + st["intensity_high"],
            st["context_low"] + st["context_med"] + st["context_high"],
            st["key_phrases"], st["sentiment_combinations"], st["transitions"], st["defense_terms"], st["bias_terms"]
        )
    else:
        logger.info("[index] empty index (check EMOTIONS.JSON schema/path)")
    return idx


# -----------------------------------------------------------------------------
# Calibration
# -----------------------------------------------------------------------------
@dataclass
class LabelStats:
    """
    배치 수집된 통계 스냅샷을 캘리브레이터에 적합한 형태로 전달하기 위한 구조.

    의미:
      - n_docs: 총 문서 수
      - class_counts / subclass_counts: (선택) 라벨 분포(정규 prior 보정 용)
      - df_pos / df_neg: 카테고리별 패턴의 문서 빈도(양성/음성). 예)
          df_pos = {
            "intensity_med": {"기쁘다": 120, "__heur_val_proxy__": 80, ...},
            "context_med":   {"전환": 50,  ...},
            "key_phrases":   {"성취": 200, ...},
          }
          df_neg의 키 구조도 동일.
    """
    n_docs: int = 0
    class_counts: Dict[str, int] = field(default_factory=dict)
    subclass_counts: Dict[str, int] = field(default_factory=dict)
    df_pos: Dict[str, Dict[str, int]] = field(default_factory=dict)
    df_neg: Dict[str, Dict[str, int]] = field(default_factory=dict)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "LabelStats":
        return LabelStats(
            n_docs=int(d.get("n_docs", 0)),
            class_counts=dict(d.get("class_counts", {}) or {}),
            subclass_counts=dict(d.get("subclass_counts", {}) or {}),
            df_pos={k: dict(v or {}) for k, v in (d.get("df_pos", {}) or {}).items()},
            df_neg={k: dict(v or {}) for k, v in (d.get("df_neg", {}) or {}).items()},
        )

@dataclass
class CalibrationSnapshot:
    """
    저장/로드 가능한 경량 스냅샷 포맷.
      - prior_log_adj: 라벨(또는 카테고리)별 prior 보정치(log-ratio)
      - pattern_log_odds: {category: {pattern: weight}}
    """
    version: str = "1"
    created_at: float = field(default_factory=lambda: time.time())
    alpha: float = 1.0
    k:     float = 0.5
    prior_log_adj: Dict[str, float] = field(default_factory=dict)
    pattern_log_odds: Dict[str, Dict[str, float]] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "created_at": self.created_at,
            "alpha": self.alpha,
            "k": self.k,
            "prior_log_adj": dict(self.prior_log_adj),
            "pattern_log_odds": {c: dict(m) for c, m in (self.pattern_log_odds or {}).items()},
            "meta": dict(self.meta or {}),
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "CalibrationSnapshot":
        snap = CalibrationSnapshot()
        snap.version = str(d.get("version", "1"))
        snap.created_at = float(d.get("created_at", time.time()))
        snap.alpha = float(d.get("alpha", 1.0))
        snap.k = float(d.get("k", 0.5))

        # pattern_log_odds 하위호환: list[{"cat":..,"pattern":..,"weight":..}] 형태도 수용
        plo = d.get("pattern_log_odds", {}) or {}
        if isinstance(plo, list):
            tmp: Dict[str, Dict[str, float]] = {}
            for row in plo:
                cat = str(row.get("cat", ""))
                pat = str(row.get("pattern", ""))
                w   = float(row.get("weight", 0.0))
                if not cat or not pat:
                    continue
                tmp.setdefault(cat, {})[pat] = w
            snap.pattern_log_odds = tmp
        else:
            snap.pattern_log_odds = {c: {p: float(w) for p, w in (m or {}).items()}
                                     for c, m in plo.items()}

        snap.prior_log_adj = {k: float(v) for k, v in (d.get("prior_log_adj", {}) or {}).items()}
        snap.meta = dict(d.get("meta", {}) or {})
        return snap

class WeightCalibrator:
    """
    스냅샷 기반 log-odds 캘리브레이터.
    - fit_from_snapshot(LabelStats): 배치 통계로부터 log-odds 산출
    - save_snapshot/load_snapshot: 파일 I/O
    - get_pattern_weight(cat, pat): 런타임 보정치 조회 (score_sentence에서 사용)
    - get_prior_adj(key): (옵션) prior 보정치 조회

    메모:
    - weight_cap: 극단치 클리핑(과신 방지)
    - 하위호환: 과거 (label, pattern) 튜플 키 평면 dict도 읽을 수 있게 대응
    """
    def __init__(self, *, weight_cap: float = 2.5):
        self.weight_cap: float = float(weight_cap)
        self.prior_log_adj: Dict[str, float] = {}
        # 권장 포맷: {category: {pattern: weight}}
        self.pattern_log_odds: Dict[str, Dict[str, float]] = {}
        # 하위호환(있을 수 있음): {(cat_or_label, pattern): weight}
        self._flat_log_odds: Dict[Tuple[str, str], float] = {}
        # 마지막 로드 스냅샷 메타
        self.meta: Dict[str, Any] = {}

    # --- 내부 유틸 ---
    def _clip(self, x: float) -> float:
        cap = max(0.1, float(self.weight_cap))
        return max(-cap, min(cap, float(x)))

    def is_loaded(self) -> bool:
        return bool(self.pattern_log_odds or self._flat_log_odds)

    # --- 핵심: 배치 통계로부터 적합 ---
    def fit_from_snapshot(self, stats: LabelStats, alpha: float = 1.0, k: float = 0.5) -> "WeightCalibrator":
        """
        stats로부터 prior 보정 및 패턴 log-odds를 산출.
        - prior_log_adj: 균일 분포 대비 로그 보정
        - pattern_log_odds: log((df_pos + k) / (df_neg + k))
        """
        # 1) prior 보정(선택: 세부 → 대분류 우선)
        N = max(1, int(stats.n_docs))
        counts = stats.subclass_counts or stats.class_counts or {}
        E = max(1, len(counts))
        self.prior_log_adj.clear()
        if counts:
            for key, c in counts.items():
                prior = (float(c) + float(alpha)) / (float(N) + float(alpha) * float(E))
                # 균일 prior(=1/E) 대비 로그 보정
                adj = math.log(max(1e-9, (1.0 / float(E)) / max(1e-12, prior)))
                self.prior_log_adj[str(key)] = self._clip(adj)

        # 2) 패턴 log-odds
        self.pattern_log_odds.clear()
        self._flat_log_odds.clear()
        df_pos = stats.df_pos or {}
        df_neg = stats.df_neg or {}
        for cat, pos_map in df_pos.items():
            neg_map = df_neg.get(cat, {}) or {}
            for pat, dfp in (pos_map or {}).items():
                dfn = float(neg_map.get(pat, 0))
                dfp = float(dfp)
                w = math.log((dfp + float(k)) / (dfn + float(k)))
                # 카테고리 맵에 저장
                c = str(cat); p = str(pat)
                self.pattern_log_odds.setdefault(c, {})[p] = self._clip(w)

        return self

    # --- 런타임 조회 ---
    def get_prior_adj(self, key: str) -> float:
        """옵션: 라벨/카테고리 수준 prior 보정 조회(현재 파이프라인에선 주로 pattern_weight만 사용)."""
        return float(self.prior_log_adj.get(key, 0.0))

    def get_pattern_weight(self, cat_or_label: str, pattern: str) -> float:
        """
        score_sentence()에서 카테고리 × 패턴으로 호출됨.
        - 새 포맷: pattern_log_odds[cat][pattern]
        - 구 포맷: _flat_log_odds[(cat, pattern)] 또는 (label, pattern)
        """
        c = str(cat_or_label); p = str(pattern)
        # 신 포맷 우선
        m = self.pattern_log_odds.get(c)
        if m:
            w = m.get(p)
            if w is not None:
                return float(w)
        # 하위호환(평면 튜플키)
        if self._flat_log_odds:
            return float(self._flat_log_odds.get((c, p), 0.0))
        return 0.0

    # --- 스냅샷 직렬화 ---
    def to_snapshot(self, *, alpha: float = 1.0, k: float = 0.5, meta: Optional[Dict[str, Any]] = None) -> CalibrationSnapshot:
        snap = CalibrationSnapshot(
            version="1",
            alpha=float(alpha),
            k=float(k),
            prior_log_adj=dict(self.prior_log_adj),
            pattern_log_odds={c: dict(m) for c, m in (self.pattern_log_odds or {}).items()},
            meta=dict(meta or self.meta or {}),
        )
        return snap

    def save_snapshot(self, path: str, *, alpha: float = 1.0, k: float = 0.5, meta: Optional[Dict[str, Any]] = None) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        snap = self.to_snapshot(alpha=alpha, k=k, meta=meta)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(snap.to_dict(), f, ensure_ascii=False, indent=2, sort_keys=True)

    # --- 스냅샷 로드 ---
    def load_snapshot(self, path: str) -> "WeightCalibrator":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # CalibrationSnapshot 스키마 해석
        if "pattern_log_odds" in data or "prior_log_adj" in data:
            snap = CalibrationSnapshot.from_dict(data)
            self.prior_log_adj = {k: self._clip(v) for k, v in (snap.prior_log_adj or {}).items()}
            self.pattern_log_odds = {
                c: {p: self._clip(w) for p, w in (m or {}).items()}
                for c, m in (snap.pattern_log_odds or {}).items()
            }
            self.meta = dict(snap.meta or {})
            self._flat_log_odds.clear()
            return self

        # 하위호환: (label, pattern) 튜플키 기반 평면 dict 덤프였던 경우
        flat = {}
        for k, v in (data or {}).items():
            if isinstance(k, str) and k.startswith("(") and k.endswith(")"):
                # eval을 쓰지 않고 안전 파싱 시도: ('cat','pat') 또는 ("cat","pat")
                try:
                    ks = k.strip("()")
                    lhs, rhs = ks.split(",", 1)
                    c = lhs.strip().strip("'\"")
                    p = rhs.strip().strip("'\"")
                    flat[(c, p)] = self._clip(float(v))
                except Exception:
                    continue
        self._flat_log_odds = flat
        self.prior_log_adj = {}
        self.pattern_log_odds = {}
        self.meta = {}
        return self

    # --- 스냅샷 병합(선택) ---
    def merge_snapshot(self, other: "WeightCalibrator", *, beta: float = 0.5) -> "WeightCalibrator":
        """
        다른 캘리브레이터(또는 방금 로드한 스냅샷)의 가중치를
        가중 평균으로 합성. beta∈[0,1], 1이면 other 100%.
        """
        b = max(0.0, min(1.0, float(beta)))
        # prior
        keys = set(self.prior_log_adj.keys()) | set(other.prior_log_adj.keys())
        for k in keys:
            a = float(self.prior_log_adj.get(k, 0.0))
            o = float(other.prior_log_adj.get(k, 0.0))
            self.prior_log_adj[k] = self._clip((1.0 - b) * a + b * o)
        # pattern
        cats = set(self.pattern_log_odds.keys()) | set(other.pattern_log_odds.keys())
        for c in cats:
            self.pattern_log_odds.setdefault(c, {})
            src_a = self.pattern_log_odds.get(c, {})
            src_b = other.pattern_log_odds.get(c, {})
            pats = set(src_a.keys()) | set(src_b.keys())
            for p in pats:
                a = float(src_a.get(p, 0.0))
                o = float(src_b.get(p, 0.0))
                self.pattern_log_odds[c][p] = self._clip((1.0 - b) * a + b * o)
        return self

# -----------------------------------------------------------------------------
# Core Components
# -----------------------------------------------------------------------------
class SignalExtractor:
    __slots__ = ("idx", "cfg", "gate", "_matchers", "_compiled_transitions", "calibrator")  # NEW: calibrator

    def __init__(self, index: EmotionsIndex, config: AnalyzerConfig,
                 calibrator: Optional["WeightCalibrator"] = None):  # NEW
        self.idx = index
        self.cfg = config
        self.gate = EvidenceGate(config.gate_min_types, config.gate_min_sum)
        # EmotionsIndex.build_matchers() / prepare_emotions_index()가 세팅한 캐시 활용
        self._matchers = getattr(index, "matchers", {}) or {}
        self._compiled_transitions = getattr(index, "compiled_transitions", []) or []
        self.calibrator = calibrator  # NEW

    # -------- internal helpers --------
    def _split_tokens(self, seq: List[Any]) -> Tuple[List[str], List[str]]:
        """
        입력 토큰 시퀀스를 (표면형/표제어 추정 없음) 문자열 리스트와 coarse POS 리스트로 분해.
        - 토큰이 (token, coarse_pos) 형태면 POS를 수집하고 토큰만 결과에 포함
        - 토큰이 str이면 POS='X'
        """
        toks: List[str] = []
        poss: List[str] = []
        for t in seq:
            if t is None:
                continue
            if isinstance(t, (tuple, list)) and len(t) >= 1:
                tok = t[0]
                pos = (t[1] if len(t) > 1 else None)
                if isinstance(tok, str) and tok.strip():
                    toks.append(tok.strip())
                    poss.append(str(pos) if pos else "X")
            elif isinstance(t, str) and t.strip():
                toks.append(t.strip())
                poss.append("X")
        return toks, poss

    def _F_cat(self, cat: str, target: str) -> List[str]:
        """
        카테고리별 매칭:
        - 정규식 샤드 매처 우선
        - 없거나 miss면 부분문자열 find_all
        - (옵션) 한국어 느슨 매칭: 어간 기반으로 0~3자 변형 허용
        """
        m = self._matchers.get(cat)
        if m:
            got = m.find_all(target)
            if got:
                return got

        arr = getattr(self.idx, cat, []) if hasattr(self.idx, cat) else []
        if not arr:
            return []

        # 1) 기본 부분문자열
        hits = find_all(target, arr)
        if hits:
            return hits

        # 2) (NEW) 한국어 느슨 매칭
        if not bool(getattr(self.cfg, "loose_korean_match", True)):
            return []
        try:
            t_ns = re.sub(r"\s+", "", target)
            ko_hits = []
            cap = 2000  # 성능 안전 상한
            for term in arr[:cap]:
                base = term
                for suf in ("하다", "해지다", "해져", "스럽다", "답다", "되다", "다"):
                    if base.endswith(suf) and len(base) > len(suf):
                        base = base[:-len(suf)]
                        break
                if len(base) < 2:
                    continue
                rx = re.compile(re.escape(base) + r"(?:[가-힣]{0,3})")
                if rx.search(t_ns):
                    ko_hits.append(term)
            if ko_hits:
                return list(dict.fromkeys(ko_hits))  # 순서 유지 dedup
        except Exception:
            pass
        return []

    def score_sentence(self, idx: int, toks: List[List[Any]], sents: List[str]) -> Dict[str, Any]:
        """
        Hybrid transition detection with label-based emotion detection and hierarchical rollup.

        NEW:
          - Label-based emotion detection (label_matchers)
          - Hierarchical rollup (fine → parent categories)
          - Prior adjustment & pattern-level calibration
        """
        # -------- config --------
        unique_once = bool(getattr(self.cfg, "unique_match_per_sent", True))
        use_idf = bool(getattr(self.cfg, "use_idf_weighting", True))
        clip_score = float(getattr(self.cfg, "clip_score", 6.0))
        length_norm_mode = str(getattr(self.cfg, "length_norm", "log"))
        len_norm_floor = float(getattr(self.cfg, "len_norm_floor", 1.2))  # make sure AnalyzerConfig has this
        calibration_scale = float(getattr(self.cfg, "calibration_scale", 0.0))
        flag_emit_heur_hits = bool(getattr(self.cfg, "flag_emit_heur_hits", True))
        hide_internal_markers = bool(getattr(self.cfg, "hide_internal_markers", True))
        mask_evidence_labels = bool(getattr(self.cfg, "mask_evidence_labels", True))

        enable_hierarchy = bool(getattr(self.cfg, "enable_hierarchy", True))
        hier_rollup_mode = str(getattr(self.cfg, "hier_rollup_mode", "softmax"))
        hier_rollup_tau = float(getattr(self.cfg, "hier_rollup_tau", 1.0))
        calibrator_prior_scale = float(getattr(self.cfg, "calibrator_prior_scale", 1.0))

        calibrator = getattr(self, "calibrator", None)  # 통일: 이 로컬 변수를 아래서 사용

        w = int(self.cfg.context_window)
        start = max(0, idx - w)
        end = min(len(toks) - 1, idx + w)

        # -------- tokens / POS in window --------
        win_tokens: List[str] = []
        win_pos: List[str] = []
        cur_tokens: List[str] = []
        cur_pos: List[str] = []

        for i in range(start, end + 1):
            ts, ps = self._split_tokens(toks[i])
            win_tokens.extend(ts);
            win_pos.extend(ps)
            if i == idx:
                cur_tokens.extend(ts);
                cur_pos.extend(ps)

        joined = join_tokens(win_tokens)
        joined_cur = join_tokens(cur_tokens) if cur_tokens else joined

        # (1) len_norm 먼저 계산 — 라벨 보정에서 사용하므로
        L_len = max(1, len(cur_tokens))
        if length_norm_mode == "sqrt":
            len_norm = max(1.0, math.sqrt(L_len))
        elif length_norm_mode == "log":
            len_norm = max(1.0, math.log(L_len + 2))
        else:
            len_norm = 1.0
        len_norm = max(float(len_norm_floor), float(len_norm))

        # 매처 헬퍼
        def FX(cat: str, *, use_current: bool = False) -> List[str]:
            target = joined_cur if use_current else joined
            return self._F_cat(cat, target)

        # -------- HEUR namespace --------
        HEUR = {
            "ADV": "__heur_adversative__",
            "MIX": "__heur_mixed__",
            "V_POS": "__heur_ko_pos__",
            "V_NEG": "__heur_ko_neg__",
            "INT_PROXY": "__heur_val_proxy__",
        }

        # -------- hits --------
        hits: Dict[str, Any] = {
            "intensity_low": FX("intensity_examples_low"),
            "intensity_med": FX("intensity_examples_med"),
            "intensity_high": FX("intensity_examples_high"),
            "context_low": FX("context_low"),
            "context_med": FX("context_med"),
            "context_high": FX("context_high"),
            "progression": [],
            "transitions": [],
            "mod_amp": FX("amplifiers"),
            "mod_dim": FX("diminishers"),
            "key_phrases": FX("key_phrases"),
            "cognitive_combos": [],
            "valence_pos": FX("pos_markers"),
            "valence_neg": FX("neg_markers"),
        }

        # progression (stage, phrase)
        if self.idx.context_progressions:
            prog_local: List[Tuple[str, str]] = []
            for stage, phrases in self.idx.context_progressions.items():
                for f in find_all(joined, phrases):
                    prog_local.append((stage, f))
            hits["progression"] = prog_local

        # -------- transitions (compiled first) --------
        trans_trigs_all: List[str] = []
        shift_bonus = 0.0
        local_bonus = 0.0
        j_nospace = re.sub(r"\s+", "", joined)
        jc_nospace = re.sub(r"\s+", "", joined_cur)

        if self._compiled_transitions:
            for comp in self._compiled_transitions:
                rx = comp["rx"]
                found_win = [m.group(0) for m in rx.finditer(j_nospace)]
                if found_win:
                    trans_trigs_all.extend(found_win)
                    if comp.get("shift_point"):
                        shift_bonus += self.cfg.transition_shiftpoint_bonus
                    found_cur = [m.group(0) for m in rx.finditer(jc_nospace)]
                    if found_cur:
                        local_bonus += self.cfg.local_transition_bonus(len(found_cur))
        else:
            for tdef in (self.idx.transitions or []):
                trigs = [re.sub(r"\s+", "", normalize_text(x)) for x in (tdef.get("triggers") or [])]
                found_win = find_all(j_nospace, trigs)
                if found_win:
                    trans_trigs_all.extend(found_win)
                    if tdef.get("shift_point"):
                        shift_bonus += self.cfg.transition_shiftpoint_bonus
                    found_cur = find_all(jc_nospace, trigs)
                    if found_cur:
                        local_bonus += self.cfg.local_transition_bonus(len(found_cur))
        hits["transitions"] = list(dict.fromkeys(trans_trigs_all)) if unique_once else trans_trigs_all

        # -------- label-based emotion detection --------
        label_scores: Dict[str, float] = {}
        if getattr(self.idx, "label_matchers", None):
            for label, matcher in self.idx.label_matchers.items():
                found = matcher.find_all(j_nospace) if matcher else []
                if not found:
                    continue
                # log1p(count)
                score = math.log1p(float(len(found)))
                # prior adj (class imbalance)
                if calibrator and calibrator_prior_scale > 0:
                    score += calibrator_prior_scale * float(calibrator.get_prior_adj(label))
                    # pattern weights (lightly)
                    if calibration_scale > 0:
                        for pattern in found[:10]:
                            score += (calibrator.get_pattern_weight(label,
                                                                    pattern) / len_norm) * calibration_scale * 0.1
                label_scores[label] = score

        # -------- hierarchical rollup (if available) --------
        labels_rollup: Dict[str, float] = {}
        if enable_hierarchy and label_scores:
            if hasattr(self.idx, "rollup_label_scores") and callable(getattr(self.idx, "rollup_label_scores")):
                labels_rollup = self.idx.rollup_label_scores(label_scores, mode=hier_rollup_mode, tau=hier_rollup_tau)
            elif hasattr(self.cfg, "apply_hierarchical_balance"):
                labels_rollup = self.cfg.apply_hierarchical_balance(label_scores)

        # -------- dedup per sentence (option) --------
        if unique_once:
            for k in ("intensity_low", "intensity_med", "intensity_high",
                      "context_low", "context_med", "context_high",
                      "mod_amp", "mod_dim", "key_phrases", "valence_pos", "valence_neg"):
                if isinstance(hits.get(k), list):
                    hits[k] = list(dict.fromkeys(hits[k]))

        # -------- POS boosts --------
        v_adj = sum(1 for p in win_pos if p in ("VERB", "ADJ"))
        adv_n = sum(1 for p in win_pos if p == "ADV")
        pos_boost_I = 0.04 * min(3, v_adj)
        pos_boost_C = 0.03 * min(3, adv_n)

        # -------- vocab-normalized base scores --------
        def _vsize(name: str) -> int:
            arr = getattr(self.idx, name, None)
            return len(arr) if arr else 0

        v_int_low = _vsize("intensity_examples_low")
        v_int_med = _vsize("intensity_examples_med")
        v_int_high = _vsize("intensity_examples_high")
        v_ctx_low = _vsize("context_low")
        v_ctx_med = _vsize("context_med")
        v_ctx_high = _vsize("context_high")

        def _norm_count(n_hits: int, vsize: int, avg_ref: float) -> float:
            base = n_hits / max(1.0, math.log(vsize + 2))
            if use_idf:
                idf = math.log((avg_ref + 1.0) / (vsize + 1.0)) + 1.0
                return base * max(0.0, idf)
            return base

        avg_v_int = max(1.0, (v_int_low + v_int_med + v_int_high) / 3.0)
        avg_v_ctx = max(1.0, (v_ctx_low + v_ctx_med + v_ctx_high) / 3.0)

        I_base = (len(hits["intensity_low"]) * self.cfg.intensity_weights_low +
                  len(hits["intensity_med"]) * self.cfg.intensity_weights_medium +
                  len(hits["intensity_high"]) * self.cfg.intensity_weights_high)
        C_base = (len(hits["context_low"]) * self.cfg.context_weights_low +
                  len(hits["context_med"]) * self.cfg.context_weights_medium +
                  len(hits["context_high"]) * self.cfg.context_weights_high)

        I_adj_core = (
                             _norm_count(len(hits["intensity_low"]), v_int_low,
                                         avg_v_int) * self.cfg.intensity_weights_low +
                             _norm_count(len(hits["intensity_med"]), v_int_med,
                                         avg_v_int) * self.cfg.intensity_weights_medium +
                             _norm_count(len(hits["intensity_high"]), v_int_high,
                                         avg_v_int) * self.cfg.intensity_weights_high
                     ) / len_norm

        C_adj_core = (
                             _norm_count(len(hits["context_low"]), v_ctx_low,
                                         avg_v_ctx) * self.cfg.context_weights_low +
                             _norm_count(len(hits["context_med"]), v_ctx_med,
                                         avg_v_ctx) * self.cfg.context_weights_medium +
                             _norm_count(len(hits["context_high"]), v_ctx_high,
                                         avg_v_ctx) * self.cfg.context_weights_high
                     ) / len_norm

        # valence 보정
        pos_n, neg_n = len(hits["valence_pos"]), len(hits["valence_neg"])
        I_adj = I_adj_core + (pos_n + neg_n) * self.cfg.intensity_weights_medium + pos_boost_I
        C_adj = C_adj_core + (pos_n - neg_n) * self.cfg.context_weights_medium + pos_boost_C

        # -------- heuristics (adversatives/mixed) --------
        enable_adv = bool(getattr(self.cfg, "enable_adversative_heuristic", True))
        heur_fired = False
        if enable_adv:
            adv_win = FX("adversatives")
            adv_cur = FX("adversatives", use_current=True)
            has_adv_win, has_adv_cur = bool(adv_win), bool(adv_cur)
            mixed_val = (pos_n > 0 and neg_n > 0)

            # 중앙부 판정: 등장 위치가 [20%, 80%] 구간
            central_like = False
            if has_adv_cur:
                for a in (getattr(self.idx, "adversatives", []) or []):
                    p = joined_cur.find(a)
                    if p >= 0:
                        frac = p / max(1, len(joined_cur))
                        if 0.2 <= frac <= 0.8:
                            central_like = True
                            break

            if has_adv_cur or (has_adv_win and mixed_val) or mixed_val or central_like:
                if HEUR["ADV"] not in hits["transitions"]:
                    hits["transitions"] = (hits.get("transitions") or []) + [HEUR["ADV"]]
                    heur_fired = True
                if mixed_val and HEUR["MIX"] not in hits["transitions"]:
                    hits["transitions"] = (hits.get("transitions") or []) + [HEUR["MIX"]]
                    heur_fired = True

                max_loc_bonus = float(getattr(self.cfg, "local_transition_bonus_max", 0.2))
                inc = min(max_loc_bonus, 0.06 * (pos_n + neg_n + 1))
                local_bonus = min(max_loc_bonus, local_bonus + inc)

                if not hits.get("valence_pos") and (FX("ko_pos_stems", use_current=True) or FX("ko_pos_stems")):
                    hits["valence_pos"] = (hits.get("valence_pos") or []) + [HEUR["V_POS"]]
                if not hits.get("valence_neg") and (FX("ko_neg_stems", use_current=True) or FX("ko_neg_stems")):
                    hits["valence_neg"] = (hits.get("valence_neg") or []) + [HEUR["V_NEG"]]

                if not (hits["intensity_low"] or hits["intensity_med"] or hits["intensity_high"]):
                    if (hits.get("valence_pos") or hits.get("valence_neg")) and (
                            HEUR["INT_PROXY"] not in (hits.get("intensity_med") or [])):
                        hits["intensity_med"] = (hits.get("intensity_med") or []) + [HEUR["INT_PROXY"]]

        # 휴리스틱 반영 후 재계산
        pos_n, neg_n = len(hits["valence_pos"]), len(hits["valence_neg"])
        I_adj = I_adj_core + (pos_n + neg_n) * self.cfg.intensity_weights_medium + pos_boost_I
        C_adj = C_adj_core + (pos_n - neg_n) * self.cfg.context_weights_medium + pos_boost_C

        # -------- M/T/L/Calib --------
        M = (len(hits["mod_amp"]) * self.cfg.modifier_per_amp -
             len(hits["mod_dim"]) * self.cfg.modifier_per_dim)

        trig_total = len(hits["transitions"]) if hits.get("transitions") else 0
        T = (self.cfg.transition_trigger_bonus * min(3, trig_total) +
             shift_bonus + local_bonus) if trig_total > 0 else 0.0

        # 라벨점수: top-3 합을 약하게 반영 (음수 방지)
        L_score = 0.0
        if label_scores:
            top3 = sorted(label_scores.items(), key=lambda x: x[1], reverse=True)[:3]
            L_score = max(0.0, sum(score for _, score in top3) * 0.1)

        adapt_bonus = 0.0
        if calibrator and calibration_scale > 0.0:
            for cat in ("intensity_low", "intensity_med", "intensity_high",
                        "context_low", "context_med", "context_high", "key_phrases"):
                for pat in hits.get(cat, []) or []:
                    adapt_bonus += calibrator.get_pattern_weight(cat, pat)
            adapt_bonus = (adapt_bonus / len_norm) * calibration_scale

        # -------- final raw score --------
        score_raw = (self.cfg.w_intensity * I_adj +
                     self.cfg.w_context * C_adj +
                     self.cfg.w_modifier * M +
                     self.cfg.w_transition * T +
                     L_score + adapt_bonus)

        if "__heur_adversative__" in (hits.get("transitions") or []):
            adv_min = float(getattr(self.cfg, "adversative_min_score", 0.28))
            score_raw = max(score_raw, adv_min)

        if clip_score is not None:
            score_raw = min(float(score_raw), float(clip_score))

        # -------- evidence --------
        ev: List[Tuple[str, float]] = []
        I_ev = max(0.0, I_base, I_adj)
        C_ev = max(0.0, C_base, C_adj)
        if I_ev > 0: ev.append(("intensity", I_ev))
        if C_ev > 0: ev.append(("context", C_ev))
        if M != 0.0: ev.append(("modifier", abs(M)))
        if T != 0.0: ev.append(("transition", abs(T)))
        if (pos_n + neg_n) > 0: ev.append(("valence", 0.05 * float(pos_n + neg_n)))
        if hits["key_phrases"]: ev.append(("lex", self.cfg.lex_evidence_weight(len(hits["key_phrases"]))))
        if L_score > 0: ev.append(("label", L_score))
        if heur_fired:
            ev.append(("heur", float(getattr(self.cfg, "heur_evidence_min", 0.6))))

        # progression 보간 + 최소 프록시
        if not hits.get("progression") and hits.get("transitions"):
            trig_term = hits["transitions"][0]
            hits["progression"] = [("trigger", trig_term)]
            if hits["context_med"]:
                hits["progression"].append(("development", hits["context_med"][0]))
            else:
                if pos_n > neg_n:
                    hits["progression"].append(("development", "valence_pos"))
                elif neg_n > pos_n:
                    hits["progression"].append(("development", "valence_neg"))

        if not (hits["intensity_low"] or hits["intensity_med"] or hits["intensity_high"]) and (pos_n + neg_n) > 0:
            if HEUR["INT_PROXY"] not in (hits.get("intensity_med") or []):
                hits["intensity_med"] = hits["intensity_med"] + [HEUR["INT_PROXY"]]
        if not (hits["context_low"] or hits["context_med"] or hits["context_high"]) and hits["transitions"]:
            hits["context_med"] = hits["context_med"] + [hits["transitions"][0]]

        # weak-signal gate helper
        ev_sum = sum(w for _, w in ev)
        weak_min = float(getattr(self.cfg, "adaptive_min_sum_weak", 0.7))
        if "__heur_adversative__" in (hits.get("transitions") or []) and ev_sum < weak_min:
            if not any(t == "heur" for t, _ in ev):
                ev.append(("heur", float(getattr(self.cfg, "heur_evidence_min", 0.6))))

        # -------- gate --------
        agg_types = len(set(t for t, _ in ev))
        signal_density = max(0.0, min(1.0, 0.3 * agg_types + 0.1 * len(ev)))
        try:
            passed, info = self.gate.allow_adaptive(
                ev, cfg=self.cfg, signal_density=signal_density,
                text_len=len(sents[idx]),
                caps={"lex": self.cfg.lex_evidence_cap}, max_type_frac=0.8,
            )
        except Exception:
            authI_hits = (len(hits["intensity_low"]) + len(hits["intensity_med"]) + len(hits["intensity_high"])) >= 1
            authC_hits = (len(hits["context_low"]) + len(hits["context_med"]) + len(hits["context_high"])) >= 2
            min_sum_eff = self.cfg.min_sum_for_signal(authI_hits, authC_hits)
            passed, info = self.gate.allow(ev, min_sum=min_sum_eff)

        # ★★★ 개선: gate 미통과 시에도 부분 점수 반영 ★★★
        # 참고: weight_calculator의 스코어링 방식 - quality/evidence 기반 가중치 조정
        if passed:
            score = score_raw
        else:
            # gate 품질(quality) 기반 부분 점수 반영
            # quality: 0~1 범위, evidence 충족도를 나타냄
            gate_quality = float(info.get("quality", 0.0))
            
            # evidence 가중치 합 비율 계산 (min_sum 대비)
            eff_total = float(info.get("eff_total_abs", info.get("total_abs", 0.0)))
            min_sum_used = float(info.get("min_sum_eff", info.get("min_sum", 1.0)))
            evidence_ratio = min(1.0, eff_total / max(0.01, min_sum_used))
            
            # 부분 점수: quality와 evidence_ratio의 평균 * 원래 점수
            # 최소 30%, 최대 70%의 원래 점수 반영
            partial_factor = max(0.3, min(0.7, (gate_quality + evidence_ratio) / 2.0))
            score = score_raw * partial_factor
            
            # 부분 점수 적용 여부 로깅 (디버그용)
            logger.debug(
                "[score_sentence] Gate not passed but partial score applied: "
                "raw=%.3f, quality=%.3f, ev_ratio=%.3f, factor=%.3f, final=%.3f",
                score_raw, gate_quality, evidence_ratio, partial_factor, score
            )

        # -------- output cleanup --------
        def _is_internal_marker(x: Any) -> bool:
            return isinstance(x, str) and x.startswith("__") and x.endswith("__")

        def _is_heur_marker(x: Any) -> bool:
            return isinstance(x, str) and str(x).startswith("__heur_")

        hits_public = hits
        if hide_internal_markers or not flag_emit_heur_hits:
            hits_public = {}
            for k, v in hits.items():
                if isinstance(v, list):
                    if k == "progression" and v and isinstance(v[0], (tuple, list)):
                        keep: List[Tuple[str, str]] = []
                        for (st, term) in v:
                            if hide_internal_markers and _is_internal_marker(term): continue
                            if (not flag_emit_heur_hits) and _is_heur_marker(term): continue
                            keep.append((st, term))
                        hits_public[k] = keep
                    else:
                        keep2 = []
                        for t in v:
                            if hide_internal_markers and _is_internal_marker(t): continue
                            if (not flag_emit_heur_hits) and _is_heur_marker(t): continue
                            keep2.append(t)
                        hits_public[k] = keep2
                else:
                    hits_public[k] = v

        hits_public["labels"] = label_scores or {}
        if labels_rollup:
            hits_public["labels_rollup"] = labels_rollup

        evidence = [{"type": t, "weight": float(w)} for t, w in ev]
        if mask_evidence_labels:
            mapped = []
            for e in evidence:
                et = e.get("type")
                if et == "heur":   et = "transition_hint"
                if et == "label":  et = "emotion_category"
                mapped.append({"type": et, "weight": float(e.get("weight", 0.0))})
            evidence = mapped

        return {
            "hits": hits_public,
            "score_components": {
                "I_base": I_base, "C_base": C_base,
                "I_adj": I_adj, "C_adj": C_adj,
                "M": M, "T": T, "L": L_score,
                "adapt_bonus": adapt_bonus,
                "len_norm": len_norm,
                "score_raw": score_raw, "passed": passed
            },
            "evidence": evidence,
            "gate": info,
            "score": score,
            "label_scores": label_scores,
            "labels_rollup": labels_rollup,
        }


class TimelineAnalyzer:
    __slots__ = ("cfg",)

    def __init__(self, config: AnalyzerConfig):
        self.cfg = config

    def _auto_thr(self, deltas: List[float]) -> float:
        if self.cfg.transition_sensitivity >= 0:
            return float(self.cfg.transition_sensitivity)
        return auto_threshold(
            [abs(d) for d in deltas],
            floor=self.cfg.trans_auto_floor,
            k=self.cfg.trans_auto_k,
            default=self.cfg.trans_auto_default,
        )

    @staticmethod
    def _estimate_quality(feat: Dict[str, Any]) -> float:
        # gate.quality가 있으면 사용, 없으면 evidence 개수/타입 다양성으로 보수적 추정
        q = feat.get("gate", {}).get("quality")
        if isinstance(q, (int, float)):
            return max(0.0, min(1.0, float(q)))
        ev = feat.get("evidence", []) or []
        distinct = len({e.get("type") for e in ev if isinstance(e, dict)})
        return max(0.0, min(1.0, 0.15 * distinct + 0.08 * len(ev)))

    def build(self, toks: List[List[Any]], sents: List[str], extractor: "SignalExtractor") -> List[Dict[str, Any]]:
        n = len(sents)
        if n == 0:
            return []

        # 1) 1차 추출 (옵션: 문장 단위 병렬)
        if getattr(self.cfg, "enable_parallel_sentence_scoring", False) and n >= int(getattr(self.cfg, "parallel_min_sentences", 8)):
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=self.cfg.parallel_max_workers) as ex:
                feats = list(ex.map(lambda i: extractor.score_sentence(i, toks, sents), range(n)))
        else:
            feats = [extractor.score_sentence(i, toks, sents) for i in range(n)]
        raw_scores = [float(f.get("score", 0.0)) for f in feats]
        deltas_for_thr = [raw_scores[i] - raw_scores[i - 1] for i in range(1, n)]
        thr0 = self._auto_thr(deltas_for_thr) if n >= 2 else self.cfg.trans_auto_default
        if thr0 <= 0.0:
            thr0 = self.cfg.trans_auto_default

        # 2) 품질가중 EWMA
        def _quality(f: Dict[str, Any]) -> float:
            q = f.get("gate", {}).get("quality")
            if isinstance(q, (int, float)):
                return max(0.0, min(1.0, float(q)))
            ev = f.get("evidence", []) or []
            distinct = len({e.get("type") for e in ev if isinstance(e, dict)})
            return max(0.0, min(1.0, 0.15 * distinct + 0.08 * len(ev)))

        qualities = [_quality(f) for f in feats]
        smooth: List[float] = []
        prev = 0.0
        for i, s in enumerate(raw_scores):
            alpha = 0.4 + 0.5 * qualities[i]
            val = (1 - alpha) * prev + alpha * s if i > 0 else s
            smooth.append(val)
            prev = val

        # 3) 히스테리시스 완화
        min_hold = 1
        hys_base = 0.12  # 0.15 → 0.12
        flow: List[Dict[str, Any]] = []

        last_state = "continuation"
        last_score = 0.0
        last_change_idx = -999

        for i in range(n):
            s_now = smooth[i]
            delta = s_now - last_score
            ad = abs(delta)

            q = qualities[i]
            hits = feats[i].get("hits", {}) or {}
            has_local_trigger = bool(hits.get("transitions"))

            # 품질/로컬리티 보정 임계
            thr_q = thr0 * (1.0 - 0.25 * q)
            thr_loc = thr_q * (0.90 if has_local_trigger else 1.0)
            hys = hys_base * (1.0 - 0.5 * q)
            thr_up = max(0.05, thr_loc * (1.0 - hys))
            thr_dn = max(0.05, thr_loc * (1.0 + hys))

            can_change = (i - last_change_idx) >= min_hold

            # ---------- 미세 전이(micro shift) 라벨링 ----------
            # full shift 임계(thr_up)에는 못 미치지만 절반 이상이면 micro_shift_up/down
            is_micro = False
            if can_change and ad >= 0.5 * thr_up and ad < thr_up:
                ttype = "micro_shift_up" if delta > 0 else "micro_shift_down"
                is_micro = True
                # micro 전이는 상태를 바꾸지 않음(hold/히스테리시스 유지)
            else:
                # ---------- 기존 전이 판정 ----------
                if can_change and ad >= thr_up:
                    ttype = "shift_up" if delta > 0 else "shift_down"
                    last_state = ttype
                    last_change_idx = i
                else:
                    if i == 0 and s_now != 0.0:
                        ttype = "onset"
                        last_state = "onset"
                        last_change_idx = i
                    elif abs(s_now) < (0.75 * thr_dn) and abs(last_score) >= thr_up and can_change:
                        ttype = "termination"
                        last_state = "termination"
                        last_change_idx = i
                    else:
                        ttype = "continuation"

            flow.append({
                "index": i,
                "text": sents[i],
                "score": round(s_now, 4),
                "delta": round(delta, 4),
                "transition": ttype,
                "is_micro": bool(is_micro),  # 디버깅/가독성용 (옵션)
                "thr_used": round(thr_loc, 4),
                "thr_up_used": round(thr_up, 4),  # 디버깅용 (옵션)
                "thr_dn_used": round(thr_dn, 4),  # 디버깅용 (옵션)
                "score_smooth": round(s_now, 4),
                "hits": feats[i].get("hits", {}),
                "evidence": feats[i].get("evidence", []),
            })
            last_score = s_now

        return flow

class MaturityEstimator:
    __slots__ = ("cfg",)

    def __init__(self, config: AnalyzerConfig):
        self.cfg = config

    def _canon(self) -> List[str]:
        return ["trigger", "development", "peak", "aftermath"]

    def _dedup_consecutive(self, xs: List[str]) -> List[str]:
        out: List[str] = []
        last = None
        for x in xs:
            if x and x != last:
                out.append(x)
                last = x
        return out

    def _lcs_len(self, a: List[str], b: List[str]) -> int:
        na, nb = len(a), len(b)
        dp = [[0]*(nb+1) for _ in range(na+1)]
        for i in range(na):
            ai = a[i]
            for j in range(nb):
                dp[i+1][j+1] = dp[i][j] + 1 if ai == b[j] else max(dp[i+1][j], dp[i][j+1])
        return dp[na][nb]

    def _infer_from_signals(self, flow: List[Dict[str, Any]]) -> Tuple[Set[str], List[str], float]:
        """progression 히트가 없을 때, 전이/발렌스/스코어 곡선으로 단계 추론"""
        n = len(flow)
        if n == 0:
            return set(), [], self.cfg.trans_auto_default

        scores = [float(f.get("score", 0.0)) for f in flow]
        deltas = [scores[i] - scores[i-1] for i in range(1, n)]
        abs_d = [abs(d) for d in deltas] if deltas else []
        thr = (float(self.cfg.transition_sensitivity)
               if self.cfg.transition_sensitivity >= 0
               else auto_threshold(abs_d, floor=self.cfg.trans_auto_floor,
                                   k=self.cfg.trans_auto_k, default=self.cfg.trans_auto_default))

        # 전이/발렌스 신호 집계
        any_trans = any((f.get("hits", {}).get("transitions") for f in flow))
        pos_counts = [len(f.get("hits", {}).get("valence_pos", []) or []) for f in flow]
        neg_counts = [len(f.get("hits", {}).get("valence_neg", []) or []) for f in flow]
        net = [p - n for p, n in zip(pos_counts, neg_counts)]
        sign = [1 if x > 0 else (-1 if x < 0 else 0) for x in net]

        stages_seen: Set[str] = set()
        seq: List[str] = []

        # trigger: 전이 발생 또는 발렌스/스코어 변화가 감지되면
        if any_trans or any(a >= thr for a in abs_d) or any(s != 0 for s in sign):
            stages_seen.add("trigger")
            seq.append("trigger")

        # development: 유의미한 변화(임계 초과) 또는 부호 변화가 있으면
        if (any(a >= thr for a in abs_d) or (len(set(sign) - {0}) > 1)) and n >= 2:
            stages_seen.add("development")
            seq.append("development")

        # peak: |score| 최대가 양 끝이 아니고 충분히 뾰족하면
        if any(scores):
            peak_idx = max(range(n), key=lambda i: abs(scores[i]))
            if 0 < peak_idx < n-1 and abs(scores[peak_idx]) >= max(0.3, 0.5 * (max(abs_d) if abs_d else 0.3)):
                stages_seen.add("peak")
                seq.append("peak")

        # aftermath: 말미에 변화가 작고(임계 미만) 신호가 수렴하면
        if n >= 2 and abs(deltas[-1]) < thr and (sign[-1] != 0 or scores[-1] != 0.0):
            stages_seen.add("aftermath")
            seq.append("aftermath")

        # 최소 보증: 전이만 있고 변화가 작을 때 trigger→development를 기본 제공
        if not stages_seen and any_trans:
            stages_seen.update({"trigger", "development"})
            seq.extend(["trigger", "development"])

        return stages_seen, self._dedup_consecutive(seq), thr

    def score(self, flow: List[Dict[str, Any]]) -> Dict[str, Any]:
        n = len(flow)
        if n == 0:
            return {"value": 0.5, "raw": 0.5, "coverage": 0.0, "gradual_ratio": 0.0,
                    "order_consistency": 0.0, "combo_complexity": 0.0, "confidence": 0.4}

        canon = self._canon()
        stages = set(canon)

        # 1) 원래 progression 히트 기반
        seen_prog: Set[str] = set()
        seq_prog: List[str] = []
        for f in flow:
            prog = f.get("hits", {}).get("progression", []) or []
            if prog:
                per_sent_seen: Set[str] = set()
                for stg, _ in prog:
                    st = _norm_stage(stg)
                    if st in stages and st not in per_sent_seen:
                        per_sent_seen.add(st)
                        seen_prog.add(st)
                        seq_prog.append(st)
        seq_prog = self._dedup_consecutive(seq_prog)

        # 2) progression이 비었으면 신호 기반 추론
        if not seen_prog:
            inferred_seen, inferred_seq, thr = self._infer_from_signals(flow)
            coverage = len(inferred_seen) / len(canon)
            order_consistency = (self._lcs_len(inferred_seq, canon) / len(canon)) if inferred_seq else 0.0
            # 점진성: 임계(thr) 미만 변화 비율
            scores = [float(f.get("score", 0.0)) for f in flow]
            deltas = [scores[i] - scores[i-1] for i in range(1, n)]
            abs_d = [abs(d) for d in deltas] if deltas else []
            gradual_ratio = (sum(1 for a in abs_d if a < thr) / max(1, len(abs_d))) if abs_d else 1.0

            combo_hits = sum(1 for f in flow if f.get("hits", {}).get("cognitive_combos"))
            combo_complexity = combo_hits / n

            raw = (0.45 * coverage +
                   0.25 * order_consistency +
                   0.20 * gradual_ratio +
                   0.10 * combo_complexity)
            value = norm01(raw)

            # 신뢰도: 샘플 수·전이/발렌스 신호 밀도
            signal_sents = sum(1 for f in flow if (f.get("hits", {}).get("transitions")
                                                   or f.get("hits", {}).get("valence_pos")
                                                   or f.get("hits", {}).get("valence_neg")))
            len_factor = min(1.0, n / 6.0)
            signal_factor = min(1.0, signal_sents / max(1.0, n * 0.75))
            confidence = norm01(0.35 + 0.65 * (0.5 * len_factor + 0.5 * signal_factor))

            return {
                "value": value,
                "raw": raw,
                "coverage": coverage,
                "gradual_ratio": gradual_ratio,
                "order_consistency": order_consistency,
                "combo_complexity": combo_complexity,
                "confidence": confidence,
            }

        # 3) progression이 존재하면 기존 방식 + 정렬 일치도 반영
        coverage = len(seen_prog) / len(canon)
        order_consistency = self._lcs_len(seq_prog, canon) / len(canon)
        # 변화 임계 계산
        scores = [float(f.get("score", 0.0)) for f in flow]
        deltas = [scores[i] - scores[i-1] for i in range(1, n)]
        abs_d = [abs(d) for d in deltas] if deltas else []
        thr = (float(self.cfg.transition_sensitivity)
               if self.cfg.transition_sensitivity >= 0
               else auto_threshold(abs_d, floor=self.cfg.trans_auto_floor,
                                   k=self.cfg.trans_auto_k, default=self.cfg.trans_auto_default))
        gradual_ratio = (sum(1 for a in abs_d if a < thr) / max(1, len(abs_d))) if abs_d else 1.0
        combo_hits = sum(1 for f in flow if f.get("hits", {}).get("cognitive_combos"))
        combo_complexity = combo_hits / n

        raw = (0.45 * coverage +
               0.25 * order_consistency +
               0.20 * gradual_ratio +
               0.10 * combo_complexity)
        value = norm01(raw)

        signal_sents = sum(1 for f in flow if (f.get("hits", {}).get("progression")
                                               or f.get("hits", {}).get("cognitive_combos")))
        len_factor = min(1.0, n / 6.0)
        signal_factor = min(1.0, signal_sents / max(1.0, n * 0.75))
        confidence = norm01(0.35 + 0.65 * (0.5 * len_factor + 0.5 * signal_factor))

        return {
            "value": value,
            "raw": raw,
            "coverage": coverage,
            "gradual_ratio": gradual_ratio,
            "order_consistency": order_consistency,
            "combo_complexity": combo_complexity,
            "confidence": confidence,
        }

class StabilityEstimator:
    __slots__ = ("cfg",)

    def __init__(self, config: AnalyzerConfig):
        self.cfg = config

    def _bounded(self, x: float) -> float:
        return x / (abs(x) + 1.0)

    def score(self, flow: List[Dict[str, Any]]) -> Dict[str, Any]:
        n = len(flow)
        scores = [float(f.get("score", 0.0)) for f in flow] or [0.0]

        if n <= 1:
            base = {
                "value": 0.8,
                "confidence": 0.5,
                "variance": 0.0,
                "span": 0.0,
                "sudden_ratio": 0.0,
                "transition_rate": 0.0,
                "delta_median": 0.0,
                "delta_mad": 0.0,
                "jerk_mad": 0.0,
                "threshold": self.cfg.trans_auto_default,
                "note": "single sentence heuristic",
            }
            base["confidence"] = min(0.8, base.get("confidence", 0.5))
            return base

        mean = sum(scores) / n
        var = sum((x - mean) ** 2 for x in scores) / (n - 1)
        span = max(scores) - min(scores)

        deltas = [scores[i] - scores[i - 1] for i in range(1, n)]
        abs_deltas = [abs(d) for d in deltas]
        d_med, d_mad = mad(abs_deltas)  # median of |Δ|, MAD scaled
        thr = (float(self.cfg.transition_sensitivity)
               if self.cfg.transition_sensitivity >= 0
               else auto_threshold(abs_deltas, floor=self.cfg.trans_auto_floor,
                                   k=self.cfg.trans_auto_k, default=self.cfg.trans_auto_default))

        sudden_count = sum(1 for a in abs_deltas if a >= thr)
        sudden_ratio = sudden_count / max(1, len(abs_deltas))

        shifts = [f for f in flow if f.get("transition") in ("shift_up", "shift_down")]
        transition_rate = len(shifts) / max(1, n - 1)

        if len(deltas) >= 2:
            jerks = [deltas[i] - deltas[i - 1] for i in range(1, len(deltas))]
            _, j_mad = mad([abs(j) for j in jerks])
        else:
            j_mad = 0.0

        # NEW: intensity variance and configurable weights
        I = [float(f.get("score_components", {}).get("I_adj", 0.0)) for f in flow] or [0.0]
        meanI = sum(I) / n
        varI = sum((x - meanI) ** 2 for x in I) / (n - 1)

        b = self._bounded
        var_n = b(var)
        span_n = b(span)
        jerk_n = b(j_mad)

        volatility = (
            self.cfg.stab_w_var * var_n
            + self.cfg.stab_w_span * span_n
            + self.cfg.stab_w_sudden * sudden_ratio
            + self.cfg.stab_w_jerk * jerk_n
            + self.cfg.stab_w_trans_rate * transition_rate
            + 0.15 * b(varI)
        )
        value = norm01(1.0 - volatility)

        ev_density = sum(1 for f in flow if f.get("evidence")) / max(1, n)
        len_factor = min(1.0, n / max(1.0, self.cfg.stab_len_penalty_k))
        var_factor = 1.0 - var_n
        confidence = norm01(0.3 + 0.7 * (0.5 * len_factor + 0.3 * (1 - var_n) + 0.2 * (1 - b(varI))))

        return {
            "value": value,
            "confidence": confidence,
            "variance": var,
            "span": span,
            "sudden_ratio": sudden_ratio,
            "transition_rate": transition_rate,
            "delta_median": d_med,
            "delta_mad": d_mad,
            "jerk_mad": j_mad,
            "threshold": thr,
            "intensity_var": varI,
        }

class DefenseBiasDetector:
    __slots__ = ("idx", "cfg", "gate")

    def __init__(self, index: EmotionsIndex, config: AnalyzerConfig):
        self.idx = index
        self.cfg = config
        self.gate = EvidenceGate(config.gate_min_types, config.gate_min_sum)

    def _flatten(self, ts: List[Any]) -> List[str]:
        out: List[str] = []
        for t in ts:
            if t is None:
                continue
            if isinstance(t, (tuple, list)):
                t = t[0]
            if isinstance(t, str) and t.strip():
                out.append(t.strip())
        return out

    def _detect_generic(
        self,
        toks: List[List[Any]],
        sents: List[str],
        table: Dict[str, List[str]],
        key_name: str,
        per_sentence_cap: int,
        flow: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        if not table or not sents:
            return []
        n = len(sents)
        intens = [float(f.get("score_components", {}).get("I_adj", 0.0)) for f in (flow or [{}] * n)]

        # Precompute per-sentence features
        joined_per_sent: List[str] = []
        mods_per_sent: List[Tuple[List[str], List[str], List[str], List[str]]] = []
        for i, ts in enumerate(toks):
            flat = self._flatten(ts)
            joined = join_tokens(flat)
            joined_per_sent.append(joined)
            if not joined:
                mods_per_sent.append(([], [], [], []))
                continue
            amp_hits = find_all(joined, self.idx.amplifiers)
            dim_hits = find_all(joined, self.idx.diminishers)
            pos_hits = find_all(joined, self.idx.pos_markers)
            neg_hits = find_all(joined, self.idx.neg_markers)
            mods_per_sent.append((amp_hits, dim_hits, pos_hits, neg_hits))

        # Build hits map (sentence,name) -> hits
        hits_map: Dict[Tuple[int, str], List[str]] = {}
        names_by_sent: List[Set[str]] = [set() for _ in range(n)]
        for i in range(n):
            joined = joined_per_sent[i]
            if not joined:
                continue
            for name, kws in (table or {}).items():
                if not kws:
                    continue
                hits = find_all(joined, kws)
                if hits:
                    hits_map[(i, name)] = hits
                    names_by_sent[i].add(name)

        if not hits_map:
            return []

        items: List[Dict[str, Any]] = []
        W = int(self.cfg.def_window)
        for (i, name), hits in hits_map.items():
            left, right = max(0, i - W), min(n - 1, i + W)
            repeats = sum(1 for j in range(left, right + 1) if name in names_by_sent[j])
            max_int = max(0.0, max(intens[left:right + 1] or [0.0]))
            if repeats < int(self.cfg.def_min_repeats):
                continue
            if max_int < float(self.cfg.def_min_intensity):
                continue

            ev_gate: List[Tuple[str, float]] = [("rule", 1.0)] * repeats
            if getattr(self.cfg, "def_use_adaptive_gate", True):
                passed, info = self.gate.allow_adaptive(
                    ev_gate,
                    cfg=self.cfg,
                    text_len=len(sents[i]),
                    caps={"lex": self.cfg.lex_evidence_cap},
                    max_type_frac=0.75,
                )
            else:
                passed, info = self.gate.allow(ev_gate)
            if not passed:
                continue

            amp_hits, dim_hits, pos_hits, neg_hits = mods_per_sent[i]
            mod_adj = len(amp_hits) * self.cfg.modifier_per_amp - len(dim_hits) * self.cfg.modifier_per_dim
            conf = 0.55 + 0.05 * min(4, repeats)
            if mod_adj > 0:
                conf += 0.05
            elif mod_adj < 0:
                conf -= 0.04
            if pos_hits and neg_hits:
                conf -= 0.02
            if max_int < 0.35:
                conf *= 0.95
            conf = max(0.5, min(0.9, conf))

            items.append({
                key_name: str(name),
                "sentence_index": i,
                "text": sents[i],
                "evidence": (
                    [{"type": "rule", "term": h, "weight": 1.0} for h in hits]
                    + ([{"type": "modifier", "term": "+", "weight": self.cfg.modifier_per_amp} for _ in amp_hits] if amp_hits else [])
                    + ([{"type": "modifier", "term": "-", "weight": self.cfg.modifier_per_dim} for _ in dim_hits] if dim_hits else [])
                    + ([{"type": "valence", "term": "pos", "weight": 0.05} for _ in pos_hits] if pos_hits else [])
                    + ([{"type": "valence", "term": "neg", "weight": 0.05} for _ in neg_hits] if neg_hits else [])
                ),
                "confidence": conf,
                "gate": info,
            })

        if not items:
            return []

        by_sent: Dict[int, int] = {}
        capped: List[Dict[str, Any]] = []
        for obj in sorted(items, key=lambda x: (x["sentence_index"], -x["confidence"], -len(x.get("evidence", ()))), reverse=False):
            si = obj["sentence_index"]
            if by_sent.get(si, 0) >= per_sentence_cap:
                continue
            capped.append(obj)
            by_sent[si] = by_sent.get(si, 0) + 1

        return dedup_by(capped, (key_name, "sentence_index"))

    def detect_defenses(self, toks: List[List[Any]], sents: List[str], flow: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        # 기본 방어 메커니즘 탐지
        basic_defenses = self._detect_generic(toks, sents, self.idx.defenses, "mechanism", self.cfg.max_defense_per_sentence, flow)
        
        # EMOTIONS.json 기반 방어 메커니즘 탐지 강화
        enhanced_defenses = self._detect_emotions_based_defenses(toks, sents, flow)
        
        # 결과 병합 및 중복 제거
        all_defenses = basic_defenses + enhanced_defenses
        return self._deduplicate_defenses(all_defenses)

    def detect_biases(self, toks: List[List[Any]], sents: List[str], flow: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        # 기본 인지 편향 탐지
        basic_biases = self._detect_generic(toks, sents, self.idx.biases, "bias", self.cfg.max_bias_per_sentence, flow)
        
        # EMOTIONS.json 기반 인지 편향 탐지 강화
        enhanced_biases = self._detect_emotions_based_biases(toks, sents, flow)
        
        # ★★★ NEW: flow 데이터 기반 인지 편향 자동 추론 ★★★
        inferred_biases = self._infer_biases_from_flow(flow, sents)
        
        # 결과 병합 및 중복 제거
        all_biases = basic_biases + enhanced_biases + inferred_biases
        return self._deduplicate_biases(all_biases)
    
    def _infer_biases_from_flow(self, flow: Optional[List[Dict[str, Any]]], sents: List[str]) -> List[Dict[str, Any]]:
        """
        ★★★ 핵심 개선: flow 데이터(감정 분석 결과)에서 인지 편향 자동 추론 ★★★
        
        EMOTIONS.json 임베딩 기반 분석 결과를 활용하여 인지 편향을 추론합니다.
        - 감정 분포 패턴 → 관련 인지 편향
        - 감정 변화 패턴 → 관련 인지 편향
        - 감정 강도 패턴 → 관련 인지 편향
        """
        if not flow or not sents:
            return []
        
        inferred = []
        n = len(flow)
        
        # 1) 전체 감정 분포 집계 (flow의 label_scores 활용)
        emotion_totals = {"희": 0.0, "노": 0.0, "애": 0.0, "락": 0.0}
        emotion_counts = {"희": 0, "노": 0, "애": 0, "락": 0}
        
        for i, f in enumerate(flow):
            # label_scores: 각 문장의 감정 레이블 점수
            label_scores = f.get("label_scores") or f.get("hits", {}).get("labels", {}) or {}
            
            for emo_key in emotion_totals.keys():
                # 감정 키가 포함된 레이블 찾기 (예: "희-기쁨", "노-분노" 등)
                for label, score in label_scores.items():
                    if isinstance(label, str) and label.startswith(emo_key):
                        if isinstance(score, (int, float)) and score > 0:
                            emotion_totals[emo_key] += float(score)
                            emotion_counts[emo_key] += 1
            
            # score 자체도 활용 (양수=긍정, 음수=부정)
            score = float(f.get("score", 0.0))
            if score > 0.1:
                emotion_totals["희"] += score * 0.5
                emotion_counts["희"] += 1
            elif score < -0.1:
                emotion_totals["노"] += abs(score) * 0.3
                emotion_totals["애"] += abs(score) * 0.2
                emotion_counts["노"] += 1
                emotion_counts["애"] += 1
        
        # 감정 분포 정규화
        total_score = sum(emotion_totals.values())
        if total_score > 0:
            emotion_dist = {k: v / total_score for k, v in emotion_totals.items()}
        else:
            emotion_dist = {"희": 0.25, "노": 0.25, "애": 0.25, "락": 0.25}
        
        # 2) 감정 변화 패턴 분석
        deltas = [float(f.get("delta", 0.0)) for f in flow]
        neg_deltas = sum(1 for d in deltas if d < -0.1)
        pos_deltas = sum(1 for d in deltas if d > 0.1)
        volatility = sum(abs(d) for d in deltas) / max(1, len(deltas))
        
        # 3) 감정 분포 기반 인지 편향 추론
        # 노(분노) 지배적 → 귀인 오류, 부정 편향
        if emotion_dist.get("노", 0) > 0.3:
            conf = min(0.85, 0.4 + emotion_dist["노"])
            inferred.append({
                "bias": "귀인 오류",
                "sentence_index": 0,
                "sentence": sents[0] if sents else "",
                "confidence": conf,
                "matched_keywords": [],
                "evidence_count": emotion_counts.get("노", 0),
                "description": "분노 감정이 지배적일 때 원인을 외부에 귀속시키는 경향",
                "type": "flow_inferred",
                "inference_source": f"emotion_dist: 노={emotion_dist.get('노', 0):.2f}"
            })
        
        # 애(슬픔) 지배적 → 피해 의식, 비관 편향
        if emotion_dist.get("애", 0) > 0.3:
            conf = min(0.85, 0.4 + emotion_dist["애"])
            inferred.append({
                "bias": "비관 편향",
                "sentence_index": 0,
                "sentence": sents[0] if sents else "",
                "confidence": conf,
                "matched_keywords": [],
                "evidence_count": emotion_counts.get("애", 0),
                "description": "슬픔 감정이 지배적일 때 상황을 부정적으로 해석하는 경향",
                "type": "flow_inferred",
                "inference_source": f"emotion_dist: 애={emotion_dist.get('애', 0):.2f}"
            })
        
        # 노+애 혼합 → 피해 의식
        if emotion_dist.get("노", 0) > 0.2 and emotion_dist.get("애", 0) > 0.2:
            combined = emotion_dist["노"] + emotion_dist["애"]
            conf = min(0.85, 0.3 + combined * 0.5)
            inferred.append({
                "bias": "피해 의식",
                "sentence_index": 0,
                "sentence": sents[0] if sents else "",
                "confidence": conf,
                "matched_keywords": [],
                "evidence_count": emotion_counts.get("노", 0) + emotion_counts.get("애", 0),
                "description": "분노와 슬픔이 혼합되어 자신이 부당한 대우를 받았다고 인식하는 경향",
                "type": "flow_inferred",
                "inference_source": f"emotion_dist: 노={emotion_dist.get('노', 0):.2f}, 애={emotion_dist.get('애', 0):.2f}"
            })
        
        # 4) 감정 변화 패턴 기반 인지 편향 추론
        # 급격한 부정적 변화 → 감정적 추론 편향
        if neg_deltas >= 2 and volatility > 0.15:
            conf = min(0.75, 0.3 + volatility)
            inferred.append({
                "bias": "감정적 추론",
                "sentence_index": 0,
                "sentence": sents[0] if sents else "",
                "confidence": conf,
                "matched_keywords": [],
                "evidence_count": neg_deltas,
                "description": "감정 변화가 급격할 때 감정에 기반하여 결론을 내리는 경향",
                "type": "flow_inferred",
                "inference_source": f"volatility={volatility:.2f}, neg_deltas={neg_deltas}"
            })
        
        # 지속적 부정 감정 (모든 문장에서 부정) → 부정 편향
        neg_score_ratio = sum(1 for f in flow if float(f.get("score", 0.0)) < 0) / max(1, n)
        if neg_score_ratio > 0.6:
            conf = min(0.8, 0.4 + neg_score_ratio * 0.4)
            inferred.append({
                "bias": "부정 편향",
                "sentence_index": 0,
                "sentence": sents[0] if sents else "",
                "confidence": conf,
                "matched_keywords": [],
                "evidence_count": int(neg_score_ratio * n),
                "description": "지속적으로 부정적 감정이 감지되어 부정적 측면에 집중하는 경향",
                "type": "flow_inferred",
                "inference_source": f"neg_score_ratio={neg_score_ratio:.2f}"
            })
        
        # 5) 희(긍정) 지배적이지만 노 혼합 → 확증 편향 (자기 의견 강화)
        if emotion_dist.get("희", 0) > 0.4 and emotion_dist.get("노", 0) > 0.15:
            conf = min(0.7, 0.3 + emotion_dist["희"] * 0.5)
            inferred.append({
                "bias": "확증 편향",
                "sentence_index": 0,
                "sentence": sents[0] if sents else "",
                "confidence": conf,
                "matched_keywords": [],
                "evidence_count": emotion_counts.get("희", 0),
                "description": "긍정과 분노가 혼합되어 자신의 의견을 강화하려는 경향",
                "type": "flow_inferred",
                "inference_source": f"emotion_dist: 희={emotion_dist.get('희', 0):.2f}, 노={emotion_dist.get('노', 0):.2f}"
            })
        
        return inferred
    
    def _detect_emotions_based_defenses(self, toks: List[List[Any]], sents: List[str], flow: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """EMOTIONS.json 기반 방어 메커니즘 탐지"""
        defenses = []
        
        # EMOTIONS.json에서 방어 메커니즘 관련 패턴 수집
        defense_patterns = self._extract_defense_patterns_from_emotions()
        
        if not defense_patterns:
            return defenses
        
        # 문장별 방어 메커니즘 탐지
        for i, sentence in enumerate(sents):
            sentence_lower = sentence.lower()
            sentence_defenses = []
            
            for mechanism_name, patterns in defense_patterns.items():
                confidence = 0.0
                matched_keywords = []
                
                # 키워드 매칭
                for keyword in patterns.get('keywords', []):
                    if keyword.lower() in sentence_lower:
                        confidence += 0.3
                        matched_keywords.append(keyword)
                
                # 예시 문구 매칭
                for example in patterns.get('examples', []):
                    if example.lower() in sentence_lower:
                        confidence += 0.4
                        matched_keywords.append(example)
                
                # 변형 표현 매칭
                for variation in patterns.get('variations', []):
                    if variation.lower() in sentence_lower:
                        confidence += 0.2
                        matched_keywords.append(variation)
                
                # 설명 매칭
                description = patterns.get('description', '')
                if description and description.lower() in sentence_lower:
                    confidence += 0.1
                    matched_keywords.append(description)
                
                # 신뢰도 임계값 확인
                if confidence >= 0.3 and matched_keywords:
                    # 강도 레벨에 따른 가중치 적용
                    intensity_level = patterns.get('intensity', 'medium')
                    intensity_weight = {'low': 0.3, 'medium': 0.6, 'high': 0.9}.get(intensity_level, 0.5)
                    
                    final_confidence = min(0.9, confidence * intensity_weight)
                    
                    sentence_defenses.append({
                        "mechanism": mechanism_name,
                        "sentence_index": i,
                        "sentence": sentence,
                        "confidence": final_confidence,
                        "matched_keywords": matched_keywords,
                        "evidence_count": len(matched_keywords),
                        "intensity_level": intensity_level,
                        "description": description,
                        "core_concept": patterns.get('core_concept', ''),
                        "type": "emotions_based"
                    })
            
            defenses.extend(sentence_defenses)
        
        return defenses
    
    def _extract_defense_patterns_from_emotions(self) -> Dict[str, Dict[str, Any]]:
        """EMOTIONS.json에서 방어 메커니즘 패턴 추출"""
        defense_patterns = {}
        
        # 기본 방어 메커니즘 패턴 정의
        basic_defense_patterns = {
            "합리화": {
                "keywords": ["합리화", "이유", "변명", "설명", "정당화", "rationalization", "excuse", "justification"],
                "examples": ["그럴 이유가 있어", "다른 사람들도 그렇게 해", "상황이 그랬어"],
                "variations": ["합리적인 이유", "충분한 근거", "타당한 설명"],
                "description": "자신의 행동이나 결정을 합리적으로 설명하려는 방어 메커니즘",
                "core_concept": "합리화",
                "intensity": "medium"
            },
            "투사": {
                "keywords": ["투사", "남탓", "다른 사람", "상대방", "projection", "blame others", "blame someone else"],
                "examples": ["다른 사람이 먼저 그렇게 했어", "상대방이 문제야", "남들이 이해 못해"],
                "variations": ["타인의 책임", "상대방의 잘못", "다른 사람의 문제"],
                "description": "자신의 감정이나 행동을 다른 사람에게 투사하는 방어 메커니즘",
                "core_concept": "투사",
                "intensity": "high"
            },
            "억압": {
                "keywords": ["억압", "숨기기", "감추기", "무시", "repression", "suppression", "hide", "ignore"],
                "examples": ["그런 일은 기억나지 않아", "생각하지 않으려고 해", "무시하고 있어"],
                "variations": ["의식적으로 잊기", "감정 억누르기", "기억 차단"],
                "description": "불쾌한 감정이나 기억을 의식에서 억압하는 방어 메커니즘",
                "core_concept": "억압",
                "intensity": "medium"
            },
            "부인": {
                "keywords": ["부인", "아니다", "그런 게 아니야", "denial", "not true", "that's not right"],
                "examples": ["그런 일 없어", "아니야", "그게 아니야"],
                "variations": ["사실 부정", "현실 거부", "진실 회피"],
                "description": "불쾌한 현실이나 사실을 부인하는 방어 메커니즘",
                "core_concept": "부인",
                "intensity": "high"
            },
            "반동형성": {
                "keywords": ["반동형성", "반대", "정반대", "reaction formation", "opposite", "contrary"],
                "examples": ["정말 좋아하는 척해", "싫어하는데 좋아한다고 해", "반대로 행동해"],
                "variations": ["정반대 행동", "의식적 반대", "감정 반전"],
                "description": "진짜 감정과 반대로 행동하는 방어 메커니즘",
                "core_concept": "반동형성",
                "intensity": "medium"
            },
            "전위": {
                "keywords": ["전위", "다른 곳에", "대상 바꾸기", "displacement", "redirect", "target change"],
                "examples": ["다른 사람에게 화내", "다른 일로 스트레스 풀어", "대상 바꿔서"],
                "variations": ["감정 전환", "대상 변경", "방향 전환"],
                "description": "한 대상에 대한 감정을 다른 대상에게 전위하는 방어 메커니즘",
                "core_concept": "전위",
                "intensity": "medium"
            },
            "승화": {
                "keywords": ["승화", "창작", "예술", "활동", "sublimation", "creation", "art", "activity"],
                "examples": ["그림으로 표현해", "음악으로 풀어", "운동으로 해소해"],
                "variations": ["건설적 표현", "예술적 승화", "활동적 전환"],
                "description": "불쾌한 감정을 건설적인 활동으로 승화시키는 방어 메커니즘",
                "core_concept": "승화",
                "intensity": "low"
            },
            "퇴행": {
                "keywords": ["퇴행", "어린아이", "유치한", "regression", "childish", "immature"],
                "examples": ["어린아이처럼 행동해", "유치하게 굴어", "어려진 것 같아"],
                "variations": ["발달 단계 퇴행", "유아적 행동", "성숙도 저하"],
                "description": "스트레스 상황에서 더 어린 발달 단계로 퇴행하는 방어 메커니즘",
                "core_concept": "퇴행",
                "intensity": "medium"
            }
        }
        
        # EMOTIONS.json에서 방어적 감정 패턴 추출
        emotions_defense_patterns = self._extract_defensive_emotion_patterns()
        
        # 패턴 병합
        defense_patterns.update(basic_defense_patterns)
        defense_patterns.update(emotions_defense_patterns)
        
        return defense_patterns
    
    def _extract_defensive_emotion_patterns(self) -> Dict[str, Dict[str, Any]]:
        """EMOTIONS.json에서 방어적 감정 패턴 추출"""
        defensive_patterns = {}
        
        # EMOTIONS.json 데이터가 있는 경우
        if hasattr(self, 'index') and hasattr(self.index, 'emotions_data'):
            emotions_data = getattr(self.index, 'emotions_data', {})
            
            for emotion_name, emotion_info in emotions_data.items():
                # 방어적 감정 확인
                if self._is_defensive_emotion(emotion_name, emotion_info):
                    sub_emotions = emotion_info.get('sub_emotions', {}) or {}
                    
                    for sub_emotion, sub_info in sub_emotions.items():
                        context_patterns = sub_info.get('context_patterns', {}) or {}
                        situations = context_patterns.get('situations', {}) or {}
                        
                        for situation_key, situation_data in situations.items():
                            # 방어적 상황 패턴 추출
                            if self._is_defensive_situation(situation_key, situation_data):
                                mechanism_name = f"{emotion_name}-{sub_emotion}-{situation_key}"
                                
                                defensive_patterns[mechanism_name] = {
                                    "keywords": situation_data.get('keywords', []) or [],
                                    "examples": situation_data.get('examples', []) or [],
                                    "variations": situation_data.get('variations', []) or [],
                                    "description": situation_data.get('description', '') or '',
                                    "core_concept": situation_data.get('core_concept', '') or '',
                                    "intensity": situation_data.get('intensity', 'medium') or 'medium'
                                }
        
        return defensive_patterns
    
    def _is_defensive_emotion(self, emotion_name: str, emotion_info: Dict[str, Any]) -> bool:
        """감정이 방어적인지 확인"""
        defensive_keywords = ["방어", "보호", "지키", "막", "차단", "거부", "부인", "변명", "합리화"]
        
        # 감정 이름에서 방어적 키워드 확인
        emotion_lower = emotion_name.lower()
        for keyword in defensive_keywords:
            if keyword in emotion_lower:
                return True
        
        # 감정 프로필에서 방어적 키워드 확인
        emotion_profile = emotion_info.get('emotion_profile', {}) or {}
        core_keywords = emotion_profile.get('core_keywords', []) or []
        
        for keyword in core_keywords:
            if isinstance(keyword, str):
                keyword_lower = keyword.lower()
                for defensive_keyword in defensive_keywords:
                    if defensive_keyword in keyword_lower:
                        return True
        
        return False
    
    def _is_defensive_situation(self, situation_key: str, situation_data: Dict[str, Any]) -> bool:
        """상황이 방어적인지 확인"""
        defensive_keywords = ["방어", "보호", "지키", "막", "차단", "거부", "부인", "변명", "합리화", "비판", "대응"]
        
        # 상황 키에서 방어적 키워드 확인
        situation_lower = situation_key.lower()
        for keyword in defensive_keywords:
            if keyword in situation_lower:
                return True
        
        # 상황 설명에서 방어적 키워드 확인
        description = situation_data.get('description', '') or ''
        if description:
            description_lower = description.lower()
            for keyword in defensive_keywords:
                if keyword in description_lower:
                    return True
        
        # 핵심 개념에서 방어적 키워드 확인
        core_concept = situation_data.get('core_concept', '') or ''
        if core_concept:
            core_concept_lower = core_concept.lower()
            for keyword in defensive_keywords:
                if keyword in core_concept_lower:
                    return True
        
        return False
    
    def _deduplicate_defenses(self, defenses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """방어 메커니즘 중복 제거"""
        if not defenses:
            return defenses
        
        # 문장별로 그룹화
        by_sentence = {}
        for defense in defenses:
            sent_idx = defense.get('sentence_index', 0)
            if sent_idx not in by_sentence:
                by_sentence[sent_idx] = []
            by_sentence[sent_idx].append(defense)
        
        # 문장별로 중복 제거
        deduplicated = []
        for sent_idx, sent_defenses in by_sentence.items():
            # 같은 메커니즘 이름으로 그룹화
            by_mechanism = {}
            for defense in sent_defenses:
                mechanism = defense.get('mechanism', '')
                if mechanism not in by_mechanism:
                    by_mechanism[mechanism] = []
                by_mechanism[mechanism].append(defense)
            
            # 각 메커니즘별로 가장 높은 신뢰도 선택
            for mechanism, mechanism_defenses in by_mechanism.items():
                best_defense = max(mechanism_defenses, key=lambda x: x.get('confidence', 0.0))
                deduplicated.append(best_defense)
        
        # 신뢰도 순으로 정렬
        deduplicated.sort(key=lambda x: x.get('confidence', 0.0), reverse=True)
        
        return deduplicated
    
    def _detect_emotions_based_biases(self, toks: List[List[Any]], sents: List[str], flow: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """EMOTIONS.json 기반 인지 편향 탐지"""
        biases = []
        
        # EMOTIONS.json에서 인지 편향 관련 패턴 수집
        bias_patterns = self._extract_bias_patterns_from_emotions()
        
        if not bias_patterns:
            return biases
        
        # 문장별 인지 편향 탐지
        for i, sentence in enumerate(sents):
            sentence_lower = sentence.lower()
            sentence_biases = []
            
            for bias_name, patterns in bias_patterns.items():
                confidence = 0.0
                matched_keywords = []
                
                # 키워드 매칭
                for keyword in patterns.get('keywords', []):
                    if keyword.lower() in sentence_lower:
                        confidence += 0.3
                        matched_keywords.append(keyword)
                
                # 예시 문구 매칭
                for example in patterns.get('examples', []):
                    if example.lower() in sentence_lower:
                        confidence += 0.4
                        matched_keywords.append(example)
                
                # 변형 표현 매칭
                for variation in patterns.get('variations', []):
                    if variation.lower() in sentence_lower:
                        confidence += 0.2
                        matched_keywords.append(variation)
                
                # 설명 매칭
                description = patterns.get('description', '')
                if description and description.lower() in sentence_lower:
                    confidence += 0.1
                    matched_keywords.append(description)
                
                # 신뢰도 임계값 확인
                if confidence >= 0.3 and matched_keywords:
                    # 강도 레벨에 따른 가중치 적용
                    intensity_level = patterns.get('intensity', 'medium')
                    intensity_weight = {'low': 0.3, 'medium': 0.6, 'high': 0.9}.get(intensity_level, 0.5)
                    
                    final_confidence = min(0.9, confidence * intensity_weight)
                    
                    sentence_biases.append({
                        "bias": bias_name,
                        "sentence_index": i,
                        "sentence": sentence,
                        "confidence": final_confidence,
                        "matched_keywords": matched_keywords,
                        "evidence_count": len(matched_keywords),
                        "intensity_level": intensity_level,
                        "description": description,
                        "core_concept": patterns.get('core_concept', ''),
                        "type": "emotions_based"
                    })
            
            biases.extend(sentence_biases)
        
        return biases
    
    def _extract_bias_patterns_from_emotions(self) -> Dict[str, Dict[str, Any]]:
        """EMOTIONS.json에서 인지 편향 패턴 추출"""
        bias_patterns = {}
        
        # 기본 인지 편향 패턴 정의
        basic_bias_patterns = {
            "확증편향": {
                "keywords": ["확증편향", "확인편향", "자신의 생각", "자신의 의견", "confirmation bias", "confirming"],
                "examples": ["내 생각이 맞아", "역시 그럴 줄 알았어", "내가 맞다고 생각해"],
                "variations": ["자신의 의견 확인", "기존 믿음 강화", "상대방 의견 무시"],
                "description": "자신의 기존 믿음이나 가설을 확인하려는 경향",
                "core_concept": "확증편향",
                "intensity": "medium"
            },
            "선택적 지각": {
                "keywords": ["선택적", "원하는 것만", "보고 싶은 것만", "selective perception", "selective"],
                "examples": ["원하는 것만 봐", "보고 싶은 것만 들어", "선택적으로 기억해"],
                "variations": ["편향적 지각", "선택적 기억", "편향적 해석"],
                "description": "자신에게 유리한 정보만 선택적으로 지각하는 편향",
                "core_concept": "선택적 지각",
                "intensity": "medium"
            },
            "후견편향": {
                "keywords": ["후견편향", "당연히", "그럴 줄 알았어", "hindsight bias", "obvious"],
                "examples": ["당연히 그럴 줄 알았어", "그럴 줄 알았는데", "예상했어"],
                "variations": ["사후 지혜", "뒤늦은 깨달음", "예상 가능했다고 생각"],
                "description": "사건이 일어난 후에 그 결과를 예측 가능했다고 생각하는 편향",
                "core_concept": "후견편향",
                "intensity": "low"
            },
            "가용성 휴리스틱": {
                "keywords": ["가용성", "쉽게 떠오르는", "기억나는", "availability heuristic", "easily recalled"],
                "examples": ["쉽게 떠오르는 예시", "기억나는 사건", "최근에 본 것"],
                "variations": ["쉬운 기억", "최근 사건", "인상적인 사례"],
                "description": "쉽게 떠오르는 정보를 과대평가하는 편향",
                "core_concept": "가용성 휴리스틱",
                "intensity": "medium"
            },
            "대표성 휴리스틱": {
                "keywords": ["대표성", "전형적인", "일반적인", "representativeness heuristic", "typical"],
                "examples": ["전형적인 예시", "일반적인 경우", "대표적인 사례"],
                "variations": ["전형성", "일반성", "대표성"],
                "description": "전형적인 예시를 과대평가하는 편향",
                "core_concept": "대표성 휴리스틱",
                "intensity": "medium"
            },
            "기본율 무시": {
                "keywords": ["기본율", "일반적인 확률", "전체적인", "base rate neglect", "base rate"],
                "examples": ["일반적인 경우를 무시해", "전체적인 확률을 고려하지 않아", "기본적인 통계를 무시해"],
                "variations": ["기본 통계 무시", "전체 확률 무시", "일반적 확률 무시"],
                "description": "기본적인 통계 정보를 무시하고 특정 사례에 집중하는 편향",
                "core_concept": "기본율 무시",
                "intensity": "medium"
            },
            "묶음 효과": {
                "keywords": ["묶음", "그룹", "집단", "framing effect", "grouping"],
                "examples": ["같은 그룹이야", "비슷한 사람들", "같은 종류야"],
                "variations": ["그룹화", "집단화", "분류화"],
                "description": "정보를 묶어서 제시할 때 발생하는 인지 편향",
                "core_concept": "묶음 효과",
                "intensity": "low"
            },
            "점화 효과": {
                "keywords": ["점화", "영향받다", "자극받다", "priming effect", "influenced"],
                "examples": ["영향받아서", "자극받아서", "점화되어서"],
                "variations": ["자극 효과", "영향 효과", "점화 현상"],
                "description": "이전에 접한 정보가 이후 판단에 영향을 미치는 현상",
                "core_concept": "점화 효과",
                "intensity": "low"
            }
        }
        
        # EMOTIONS.json에서 편향적 감정 패턴 추출
        emotions_bias_patterns = self._extract_biased_emotion_patterns()
        
        # 패턴 병합
        bias_patterns.update(basic_bias_patterns)
        bias_patterns.update(emotions_bias_patterns)
        
        return bias_patterns
    
    def _extract_biased_emotion_patterns(self) -> Dict[str, Dict[str, Any]]:
        """EMOTIONS.json에서 편향적 감정 패턴 추출"""
        biased_patterns = {}
        
        # EMOTIONS.json 데이터가 있는 경우
        if hasattr(self, 'index') and hasattr(self.index, 'emotions_data'):
            emotions_data = getattr(self.index, 'emotions_data', {})
            
            for emotion_name, emotion_info in emotions_data.items():
                # 편향적 감정 확인
                if self._is_biased_emotion(emotion_name, emotion_info):
                    sub_emotions = emotion_info.get('sub_emotions', {}) or {}
                    
                    for sub_emotion, sub_info in sub_emotions.items():
                        context_patterns = sub_info.get('context_patterns', {}) or {}
                        situations = context_patterns.get('situations', {}) or {}
                        
                        for situation_key, situation_data in situations.items():
                            # 편향적 상황 패턴 추출
                            if self._is_biased_situation(situation_key, situation_data):
                                bias_name = f"{emotion_name}-{sub_emotion}-{situation_key}"
                                
                                biased_patterns[bias_name] = {
                                    "keywords": situation_data.get('keywords', []) or [],
                                    "examples": situation_data.get('examples', []) or [],
                                    "variations": situation_data.get('variations', []) or [],
                                    "description": situation_data.get('description', '') or '',
                                    "core_concept": situation_data.get('core_concept', '') or '',
                                    "intensity": situation_data.get('intensity', 'medium') or 'medium'
                                }
        
        return biased_patterns
    
    def _is_biased_emotion(self, emotion_name: str, emotion_info: Dict[str, Any]) -> bool:
        """감정이 편향적인지 확인"""
        biased_keywords = ["편향", "고정관념", "선입견", "편견", "고정", "고정관념", "선입관"]
        
        # 감정 이름에서 편향적 키워드 확인
        emotion_lower = emotion_name.lower()
        for keyword in biased_keywords:
            if keyword in emotion_lower:
                return True
        
        # 감정 프로필에서 편향적 키워드 확인
        emotion_profile = emotion_info.get('emotion_profile', {}) or {}
        core_keywords = emotion_profile.get('core_keywords', []) or []
        
        for keyword in core_keywords:
            if isinstance(keyword, str):
                keyword_lower = keyword.lower()
                for biased_keyword in biased_keywords:
                    if biased_keyword in keyword_lower:
                        return True
        
        return False
    
    def _is_biased_situation(self, situation_key: str, situation_data: Dict[str, Any]) -> bool:
        """상황이 편향적인지 확인"""
        biased_keywords = ["편향", "고정관념", "선입견", "편견", "고정", "고정관념", "선입관", "편향적"]
        
        # 상황 키에서 편향적 키워드 확인
        situation_lower = situation_key.lower()
        for keyword in biased_keywords:
            if keyword in situation_lower:
                return True
        
        # 상황 설명에서 편향적 키워드 확인
        description = situation_data.get('description', '') or ''
        if description:
            description_lower = description.lower()
            for keyword in biased_keywords:
                if keyword in description_lower:
                    return True
        
        # 핵심 개념에서 편향적 키워드 확인
        core_concept = situation_data.get('core_concept', '') or ''
        if core_concept:
            core_concept_lower = core_concept.lower()
            for keyword in biased_keywords:
                if keyword in core_concept_lower:
                    return True
        
        return False
    
    def _deduplicate_biases(self, biases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """인지 편향 중복 제거"""
        if not biases:
            return biases
        
        # 문장별로 그룹화
        by_sentence = {}
        for bias in biases:
            sent_idx = bias.get('sentence_index', 0)
            if sent_idx not in by_sentence:
                by_sentence[sent_idx] = []
            by_sentence[sent_idx].append(bias)
        
        # 문장별로 중복 제거
        deduplicated = []
        for sent_idx, sent_biases in by_sentence.items():
            # 같은 편향 이름으로 그룹화
            by_bias = {}
            for bias in sent_biases:
                bias_name = bias.get('bias', '')
                if bias_name not in by_bias:
                    by_bias[bias_name] = []
                by_bias[bias_name].append(bias)
            
            # 각 편향별로 가장 높은 신뢰도 선택
            for bias_name, bias_list in by_bias.items():
                best_bias = max(bias_list, key=lambda x: x.get('confidence', 0.0))
                deduplicated.append(best_bias)
        
        # 신뢰도 순으로 정렬
        deduplicated.sort(key=lambda x: x.get('confidence', 0.0), reverse=True)
        
        return deduplicated


# -----------------------------------------------------------------------------
# Orchestrator (prepared index + matcher wiring + POS-aware tokenization + richer diagnostics)
# -----------------------------------------------------------------------------
class PsychCogIntegratedAnalyzer:
    __slots__ = (
        "cfg", "started_at", "kiwi", "tok", "index",
        "extractor", "timeline", "maturity_est", "stability_est", "detector",
        "calibrator",  # ★ NEW
    )

    def __init__(self,
                 config: Optional[AnalyzerConfig] = None,
                 emotions_data_path: Optional[str] = None,
                 emotions_data: Optional[Dict[str, Any]] = None,
                 kiwi_instance: Optional[Any] = None,
                 calibrator: Optional["WeightCalibrator"] = None):  # ★ NEW
        self.cfg = config or AnalyzerConfig()
        self.started_at = now_ms()
        # Kiwi 지연 초기화 실패 경로 개선
        self.kiwi = None
        if kiwi_instance is not None and Kiwi:
            self.kiwi = kiwi_instance
        elif Kiwi and self.cfg.prefer_kiwi:
            try:
                self.kiwi = Kiwi()
            except Exception as e:
                logger.warning(f"Kiwi 초기화 실패: {e} - 폴백 토큰화 사용")
                self.kiwi = None
        self.tok = SimpleTokenizer(self.kiwi, prefer_kiwi=self.cfg.prefer_kiwi)
        self.calibrator = calibrator  # ★ NEW

        # 1) Load & prepare emotions index (includes matchers/compiled transitions)
        if emotions_data is None:
            emotions_data = load_emotions_json(emotions_data_path)
        self.index = prepare_emotions_index(emotions_data or {}, self.cfg)
        if not self.index.any_loaded():
            logger.warning("[index] empty index (check EMOTIONS.JSON schema/path)")

        # Optional schema sanity log (non-fatal)
        try:
            check = validate_emotions_schema(emotions_data or {})
            if not check.get("ok"):
                logger.debug("EMOTIONS schema warnings: %s", check.get("warnings", []))
        except Exception:
            pass

        # 2) Core components (extractor auto-uses index.matchers)
        self.extractor = SignalExtractor(self.index, self.cfg, calibrator)  # ★ NEW: calibrator 주입
        self.timeline = TimelineAnalyzer(self.cfg)
        self.maturity_est = MaturityEstimator(self.cfg)
        self.stability_est = StabilityEstimator(self.cfg)
        self.detector = DefenseBiasDetector(self.index, self.cfg)

        logger.debug("Analyzer initialized with config: %s", self.cfg.to_dict())

    def _dep_health(self) -> Dict[str, Any]:
        """외부 의존성(특히 Kiwi) 헬스 체크."""
        ok = bool(self.kiwi)
        ver = None
        try:
            if self.kiwi:
                ver = getattr(self.kiwi, "__version__", "unknown")
                # 간단 토큰화 프로브(예외 발생 시 폴백 사용)
                _ = list(self.kiwi.tokenize("안녕하세요."))[:1]  # type: ignore[attr-defined]
        except Exception:
            ok = False
        return {"kiwi_ok": bool(ok), "kiwi_ver": ver}

    def analyze(self, text: str) -> Dict[str, Any]:
        t0 = now_ms()
        if not isinstance(text, str):
            text = "" if text is None else str(text)
        text = normalize_text(text)

        # [PATCH] 프로파일링(옵션)
        prof_on = os.environ.get("PSY_PROFILE", "0") == "1"
        if prof_on:
            pr = cProfile.Profile(); pr.enable()

        # --- Dependency health & degrade factor ---
        dep = self._dep_health()
        dep_factor = 1.0 if dep.get("kiwi_ok") else 0.85

        # --- Memory Guard (옵션) ---
        if getattr(self.cfg, "mem_guard_enabled", True):
            mg = MemoryGuard(limit_mb=self.cfg.mem_guard_limit_mb)
            m = mg.check()
            if not m.get("ok", True):
                # 경량 경로 강제: 라벨매칭/캘리브레이터/탐지기 비활성화 힌트로 신호를 약화
                logger.warning("[mem_guard] degrade path: rss=%.1fMB limit=%dMB", m.get("rss_mb", -1.0), self.cfg.mem_guard_limit_mb)
                self.cfg.enable_hierarchy = False
                self.cfg.flag_emit_heur_hits = False

        # --- Input Validation ---
        validator = InputValidator(junk_thr=max(0.70, float(self.cfg.junk_guard_threshold)))
        v = validator.validate(text)
        if not v.passed:
            # 기존 junk 게이트와 동일한 안전 스키마로 조기 종료
            flags = {"flags": list(set((v.flags or []) + ["validator_reject"]))}
            t1 = now_ms()
            meta = {
                "started_at": self.started_at, "finished_at": t1,
                "language": self.cfg.language, "version": self.cfg.version,
                "analysis_time_ms": t1 - t0, "mode": ("strict" if str(getattr(self.cfg, "schema_validation_mode", "warn")).lower() == "raise" else "tolerant"),
                "dep": dep, "validator": {"reason": v.reason, "score": v.score},
            }
            if prof_on:
                try:
                    pr.disable()
                    s = io.StringIO()
                    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
                    ps.print_stats(15)
                    logger.debug("PROFILE:\n%s", s.getvalue())
                except Exception:
                    pass
            return {
                "meta": meta, "config": asdict(self.cfg), "timeline": [],
                "composite_scores": {"maturity": 0.0, "stability": 0.0, "defense_load": 0.0, "bias_load": 0.0},
                "maturity_detail": {}, "stability_detail": {}, "defense_mechanisms": [], "cognitive_biases": [],
                "insights": {"summary": "Undetermined (validation gate)"},
                "evidence_pack": {"flow_evidence": [], "defense_evidence": [], "bias_evidence": []},
                "health_flags": flags,
            }

        # --- Transaction 시작 ---
        txn = AnalysisTransaction()

        # --- Junk-Guard & minimal schema ---
        try:
            js = junk_score(text)
        except Exception:
            js = 0.0
        if len(text) < int(self.cfg.junk_short_len) or js >= float(self.cfg.junk_guard_threshold):
            flags = {"flags": ["junk_input"]}
            t1 = now_ms()
            meta = {
                "started_at": self.started_at,
                "finished_at": t1,
                "language": self.cfg.language,
                "version": self.cfg.version,
                "analysis_time_ms": t1 - t0,
                "mode": ("strict" if str(getattr(self.cfg, "schema_validation_mode", "warn")).lower() == "raise" else "tolerant"),
                "dep": dep,
                "junk_score": js,
            }
            return {
                "meta": meta,
                "config": asdict(self.cfg),
                "timeline": [],
                "composite_scores": {"maturity": 0.0, "stability": 0.0, "defense_load": 0.0, "bias_load": 0.0},
                "maturity_detail": {},
                "stability_detail": {},
                "defense_mechanisms": [],
                "cognitive_biases": [],
                "insights": {"summary": "Undetermined (low evidence)"},
                "evidence_pack": {"flow_evidence": [], "defense_evidence": [], "bias_evidence": []},
                "health_flags": flags,
            }

        # --- Sentence split ---
        sents = self.tok.sentences(text)
        txn.checkpoint("sents", {"sents": sents})
        logger.debug("Input text split into %d sentences.", len(sents))

        # --- Tokenization (POS-aware if Kiwi available) ---
        use_pos = bool(self.kiwi)
        pos_keep = use_pos
        lemma = use_pos
        pos_filter = {"NOUN", "VERB", "ADJ", "ADV"} if use_pos else None
        toks = self.tok.tokenize_many(
            sents,
            keep_pos=use_pos,
            normalize=True,
            lemma=use_pos,
            drop_punct=True,
            pos_filter=pos_filter,
        )
        txn.checkpoint("toks", {"sents": sents, "toks": toks})
        logger.debug("Tokenizing %d sentences... done.", len(sents))

        # --- Timeline / scores ---
        logger.debug("Building timeline...")
        try:
            flow = self.timeline.build(toks, sents, self.extractor)
        except Exception as e:
            logger.exception("timeline build failed; rollback to toks")
            cp = txn.rollback_to("toks")
            sents = cp.get("sents", sents); toks = cp.get("toks", toks)
            flow = []  # 경량 수렴
        txn.checkpoint("flow", {"sents": sents, "toks": toks, "flow": flow})
        logger.debug("Timeline built. Calculating high-level metrics.")

        # [PATCH] 병렬 사용 여부/조건 계산(메타 노출용)
        par_enabled_cfg = bool(getattr(self.cfg, "enable_parallel_sentence_scoring", False))
        par_min = int(getattr(self.cfg, "parallel_min_sentences", 8))
        par_used = bool(par_enabled_cfg and len(sents) >= par_min)
        par_meta = {
            "enabled_cfg": par_enabled_cfg,
            "used": par_used,
            "sentence_count": len(sents),
            "min_sentences": par_min,
            "max_workers": int(getattr(self.cfg, "parallel_max_workers", 4)),
        }

        # [PATCH] Gate quality 요약(평균/중앙/사분위) 계산
        qualities: List[float] = []
        for node in flow or []:
            try:
                q = node.get("gate", {}).get("quality")
            except Exception:
                q = None
            if isinstance(q, (int, float)) and math.isfinite(q):
                qualities.append(float(q))
        if qualities:
            qs = sorted(qualities)
            n_q = len(qs)
            p25 = qs[max(0, (n_q*25)//100 - 1)]
            p75 = qs[min(n_q-1, (n_q*75)//100)]
            gate_quality_summary = {
                "count": n_q,
                "mean": mean(qs),
                "median": median(qs),
                "p25": p25,
                "p75": p75,
            }
        else:
            gate_quality_summary = {"count": 0, "mean": None, "median": None, "p25": None, "p75": None}

        logger.debug("Estimating maturity...")
        maturity = self.maturity_est.score(flow)
        logger.debug("Estimating stability...")
        stability = self.stability_est.score(flow)

        # --- Defense / Bias (data-driven; no result if section absent) ---
        logger.debug("Detecting defense mechanisms and biases...")
        defenses = self.detector.detect_defenses(toks, sents, flow=flow)
        biases = self.detector.detect_biases(toks, sents, flow=flow)

        # --- Apply dependency-based confidence degrade ---
        try:
            for d in defenses:
                d["confidence"] = max(0.0, min(1.0, float(d.get("confidence", 0.6)) * float(dep_factor)))
            for b in biases:
                b["confidence"] = max(0.0, min(1.0, float(b.get("confidence", 0.6)) * float(dep_factor)))
        except Exception:
            pass

        # --- Insights / Diagnostics ---
        insights = self._derive_insights(flow, maturity, stability, defenses, biases)
        flags = self._diagnostic_flags(flow, defenses, biases)

        # --- Summary scores ---
        comp = {
            "maturity": maturity.get("value", 0.0),
            "stability": stability.get("value", 0.0),
            "defense_load": norm01(min(1.0, 0.15 * len(defenses))),
            "bias_load": norm01(min(1.0, 0.15 * len(biases))),
        }

        # [PATCH] 캘리브레이터 scale 적용(아주 약한 보정)
        try:
            scale = float(getattr(self.cfg, "calibration_scale", 0.0))
        except Exception:
            scale = 0.0
        if scale > 0.0:
            for k, v in list(comp.items()):
                if isinstance(v, (int, float)) and math.isfinite(v):
                    comp[k] = float((1.0 - scale) * v + scale * self._calibrator_hint(k, v))

        # --- Runtime meta & optional index stats snapshot ---
        t1 = now_ms()
        logger.info("Analysis complete in %d ms.", t1 - t0)
        # [PATCH] validator 통과 메타 기록
        validator_meta = {"passed": True, "score": float(v.score), "reason": ""}

        meta = {
            "started_at": self.started_at,
            "finished_at": t1,
            "language": self.cfg.language,
            "version": self.cfg.version,
            "analysis_time_ms": t1 - t0,
            "mode": ("strict" if str(getattr(self.cfg, "schema_validation_mode", "warn")).lower() == "raise" else "tolerant"),
            "dep": dep,
            "validator": validator_meta,
            "parallel_info": par_meta,
            "gate_quality_summary": gate_quality_summary,
        }
        try:
            meta["index_stats"] = self.index.stats()
        except Exception:
            pass

        out = {
            "meta": meta,
            "config": asdict(self.cfg),
            "timeline": flow,
            "composite_scores": comp,
            "maturity_detail": maturity,
            "stability_detail": stability,
            "defense_mechanisms": defenses[: self.cfg.top_defense],
            "cognitive_biases": biases[: self.cfg.top_bias],
            "insights": insights,
            "evidence_pack": self._collect_evidence(flow, defenses, biases),
            "health_flags": flags,
        }
        if prof_on:
            try:
                pr.disable()
                s = io.StringIO()
                ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
                ps.print_stats(15)
                logger.debug("PROFILE:\n%s", s.getvalue())
            except Exception:
                pass
        return out

    # [PATCH] 캘리브레이터 힌트(약식)
    def _calibrator_hint(self, key: str, val: float) -> float:
        """
        캘리브레이터 스냅샷이 없는 환경에서도 안전하게 작동하는 약식 힌트.
        실제 스냅샷이 있다면 그 값을 반환하도록 대체하세요.
        """
        try:
            return 0.5 + 0.5 * math.tanh((float(val) - 0.5) * 0.5)
        except Exception:
            return float(val)

    # ---- insights / flags / evidence ----
    def _derive_insights(self,
                         flow: List[Dict[str, Any]],
                         maturity: Dict[str, Any],
                         stability: Dict[str, Any],
                         defenses: List[Dict[str, Any]],
                         biases: List[Dict[str, Any]]) -> Dict[str, Any]:
        pos = sum(1 for f in flow if f.get("score", 0.0) > 0)
        neg = sum(1 for f in flow if f.get("score", 0.0) < 0)
        dom_valence = "positive" if pos > neg else ("negative" if neg > pos else "mixed")
        vol_level = "stable" if stability.get("value", 0.0) >= 0.7 else ("variable" if stability.get("value", 0.0) >= 0.4 else "volatile")
        mat_band = "mature" if maturity.get("value", 0.0) >= 0.7 else ("developing" if maturity.get("value", 0.0) >= 0.4 else "immature")
        return {
            "dominant_valence": dom_valence,
            "volatility_level": vol_level,
            "maturity_band": mat_band,
            "defense_profile": [d.get("mechanism") for d in defenses[:3]],
            "bias_profile": [b.get("bias") for b in biases[:3]],
        }

    def _diagnostic_flags(self,
                          flow: List[Dict[str, Any]],
                          defenses: List[Dict[str, Any]],
                          biases: List[Dict[str, Any]]) -> Dict[str, Any]:
        flags: List[str] = []
        if not flow:
            flags.append("no_timeline")

        sent_count = max(1, len(flow))
        if (len(defenses) + len(biases)) > 4 * sent_count:
            flags.append("over_detection")
        if any(len(f.get("evidence", [])) < self.cfg.gate_min_types for f in flow):
            flags.append("weak_evidence")

        zero_score_ratio = (sum(1 for f in flow if abs(float(f.get("score", 0.0))) < 1e-9) / float(sent_count))
        if zero_score_ratio >= 0.8:
            flags.append("low_signal")

        # 전이 희소성 플래그 (onset/shift_* 거의 없음)
        trans_cnt = sum(1 for f in flow if f.get("transition") in ("onset", "shift_up", "shift_down"))
        if trans_cnt == 0 and sent_count >= 2:
            flags.append("flat_timeline")

        return {"flags": flags, "hint": ("게이트/룰/가중/전이임계 튜닝 권장" if flags else "")}

    def _collect_evidence(self,
                          flow: List[Dict[str, Any]],
                          defenses: List[Dict[str, Any]],
                          biases: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "flow_evidence": [{"index": f.get("index"), "evidence": f.get("evidence", [])} for f in flow],
            "defense_evidence": defenses,
            "bias_evidence": biases,
        }

# =============================================================================
# Public Helpers (drop-in)
# =============================================================================
def get_default_config() -> AnalyzerConfig:
    """기본 설정 반환(ENV 반영)."""
    return AnalyzerConfig()


def build_or_load_index(
    emotions_data_path: Optional[str] = None,
    emotions_data: Optional[Dict[str, Any]] = None,
    config: Optional[AnalyzerConfig] = None,
) -> EmotionsIndex:
    """
    EMOTIONS 데이터 로드/정규화 + 인덱스 구성 + 정규식 샤딩 매처/전이 정규식 준비.
    init 등에서 미리 준비해 재사용할 때 씁니다.
    """
    cfg = config or AnalyzerConfig()
    data = emotions_data if emotions_data is not None else load_emotions_json(emotions_data_path)
    return prepare_emotions_index(data or {}, cfg)


def get_psych_analyzer(
    config: Optional[AnalyzerConfig] = None,
    emotions_data_path: Optional[str] = None,
    emotions_data: Optional[Dict[str, Any]] = None,
    kiwi: Optional[Any] = None,
    calibrator: Optional[WeightCalibrator] = None,  # ★ 추가
) -> "PsychCogIntegratedAnalyzer":
    """
    통합 분석기 팩토리. init에서 인스턴스를 재사용(매처/전이 정규식 캐시)하면 성능↑.
    """
    cfg = config or AnalyzerConfig()
    # Kiwi 지연 초기화 실패 경로 개선
    kiwi_inst = None
    if kiwi is not None and Kiwi:
        kiwi_inst = kiwi
    elif Kiwi and cfg.prefer_kiwi:
        try:
            kiwi_inst = Kiwi()
        except Exception as e:
            logger.warning(f"Kiwi 초기화 실패: {e} - 폴백 토큰화 사용")
            kiwi_inst = None
    return PsychCogIntegratedAnalyzer(
        config=cfg,
        emotions_data_path=emotions_data_path,
        emotions_data=emotions_data,
        kiwi_instance=kiwi_inst,
        calibrator=calibrator,  # ★ 전달
    )


def run_psych_analysis(
    text: str,
    emotions_data_path: Optional[str] = None,
    emotions_data: Optional[Dict[str, Any]] = None,
    config: Optional[AnalyzerConfig] = None,
    *,
    save_artifacts: bool = False,
    artifact_suffix: str = "result",
    keep_history: bool = True,
) -> Dict[str, Any]:
    """
    원샷 통합 분석(전이/성숙도/안정성/방어·편향/인지패턴 → 오케스트레이터 리포트).
    save_artifacts=True면 logs/<파일명>.<suffix>.json 저장.
    """
    analyzer = get_psych_analyzer(config, emotions_data_path, emotions_data)
    try:
        report = analyzer.analyze(text)
    except Exception as e:
        # 최소 스키마 폴백
        logger.exception(f"analyze() failed: {e}")
        report = {
            "meta": {
                "started_at": getattr(analyzer, "started_at", None),
                "finished_at": int(time.time() * 1000)
            },
            "config": analyzer.cfg.to_dict() if hasattr(analyzer.cfg, "to_dict") else {},
            "timeline": [],
            "composite_scores": {
                "maturity": 0.0,
                "stability": 0.0,
                "defense_load": 0.0,
                "bias_load": 0.0
            },
            "maturity_detail": {},
            "stability_detail": {},
            "defense_mechanisms": [],
            "cognitive_biases": [],
            "insights": {"summary": "Undetermined (exception)"},
            "evidence_pack": {
                "flow_evidence": [],
                "defense_evidence": [],
                "bias_evidence": []
            },
            "health_flags": {"flags": ["exception"]},
        }
    
    if save_artifacts:
        try:
            save_run_artifacts(report, suffix=artifact_suffix, keep_history=keep_history)
        except Exception:
            logger.exception("save_run_artifacts failed")
    return report


def analyze_sequence_only(
    text: str,
    emotions_data_path: Optional[str] = None,
    emotions_data: Optional[Dict[str, Any]] = None,
    config: Optional[AnalyzerConfig] = None,
    *,
    pos_filter: Optional[Iterable[str]] = ("NOUN", "VERB", "ADJ", "ADV"),
) -> Dict[str, Any]:
    """
    타임라인만 필요할 때(전이/흐름 가시화). Kiwi가 있으면 품사 기반 토큰화 사용.
    """
    analyzer = get_psych_analyzer(config, emotions_data_path, emotions_data)
    use_pos = bool(analyzer.kiwi)
    pf = set(pos_filter) if (use_pos and pos_filter) else None
    sents = analyzer.tok.sentences(text)
    toks = [
        analyzer.tok.tokenize(s, keep_pos=use_pos, lemma=use_pos, drop_punct=True, pos_filter=pf)
        for s in sents
    ]
    flow = analyzer.timeline.build(toks, sents, analyzer.extractor)
    return {"timeline": flow, "meta": {"sentenceCount": len(sents)}}


def validate_emotions_schema(emotions_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    라이트 스키마 점검(필수 필드 존재 여부). 경고만 리턴, 동작은 느슨 모드로 커버됨.
    """
    req_top = ["emotion_profile", "context_patterns", "linguistic_patterns"]
    warns: List[str] = []
    for cat, node in (emotions_data or {}).items():
        if not isinstance(node, dict):
            warns.append(f"{cat}: node not dict"); continue
        for k in req_top:
            if k not in node:
                warns.append(f"{cat}: missing '{k}'")
    return {"ok": len(warns) == 0, "warnings": warns}


def to_data_utils_format(report: Dict[str, Any]) -> Dict[str, Any]:
    """
    오케스트레이터 표준 포맷:
    - emotionFlows ← timeline
    - transitionPoints ← shift_up/down/onset/termination + micro_shift_* 포함
    - sequentialPatterns ← (index,text)
    """
    tl = report.get("timeline", []) or []
    transitions = [
        t for t in tl
        if t.get("transition") in ("shift_up", "shift_down", "onset", "termination",
                                   "micro_shift_up", "micro_shift_down")
    ]
    return {
        "emotionFlows": tl,
        "transitionPoints": transitions,
        "sequentialPatterns": [{"index": t.get("index"), "text": t.get("text")} for t in tl],
        "summary": {
            "flowTransitionCount": len(transitions),
            "sentenceCount": len(tl),
        },
    }


# ------------------------ Diagnostics (optional, but handy) ------------------------
def psych_index_stats(
    emotions_data_path: Optional[str] = None,
    emotions_data: Optional[Dict[str, Any]] = None,
    config: Optional[AnalyzerConfig] = None,
) -> Dict[str, Any]:
    """인덱스/매처 준비 후 통계 스냅샷."""
    idx = build_or_load_index(emotions_data_path, emotions_data, config)
    try:
        return idx.stats()
    except Exception:
        return {}


def psych_debug_tokens(
    text: str,
    *,
    config: Optional[AnalyzerConfig] = None,
    kiwi: Optional[Any] = None,
    pos_filter: Optional[Iterable[str]] = ("NOUN", "VERB", "ADJ", "ADV"),
) -> Dict[str, Any]:
    """문장/토큰/품사 프린트를 위한 간단 도우미."""
    cfg = config or AnalyzerConfig()
    # Kiwi 지연 초기화 실패 경로 개선
    kiwi_inst = None
    if kiwi and Kiwi:
        kiwi_inst = kiwi
    elif Kiwi and cfg.prefer_kiwi:
        try:
            kiwi_inst = Kiwi()
        except Exception as e:
            logger.warning(f"Kiwi 초기화 실패: {e} - 폴백 토큰화 사용")
            kiwi_inst = None
    
    tokenizer = SimpleTokenizer(kiwi_inst, prefer_kiwi=cfg.prefer_kiwi)
    sents = tokenizer.sentences(text)
    use_pos = bool(tokenizer.is_morph_available())
    pf = set(pos_filter) if (use_pos and pos_filter) else None
    toks = [tokenizer.tokenize(s, keep_pos=use_pos, lemma=use_pos, drop_punct=True, pos_filter=pf) for s in sents]
    return {"sentences": sents, "tokens": toks}


# ------------------------ Backward-compat aliases (optional) ------------------------
# 과거 코드 호환: 외부에서 이 이름을 기대할 수 있으므로 alias 제공
PsychologicalCognitiveAnalyzer = PsychCogIntegratedAnalyzer  # class alias



# =============================================================================
# Main Function
# =============================================================================
def main():
    import time
    start_time = time.time()
    
    logger.setLevel(logging.DEBUG)
    logger.info("심리·인지 통합 분석기 테스트 시작")
    
    print("=" * 80)
    print("심리·인지 통합 분석기 실행 시작")
    print("=" * 80)

    try:
        # 후보 경로에서 EMOTIONS JSON 탐색
        default_path_candidates = [
            os.environ.get("EMOTIONS_JSON"),
            "./EMOTIONS.JSON",
            "./EMOTIONS.json",
            "/mnt/data/EMOTIONS.JSON",
            "/mnt/data/EMOTIONS.JSON(일부).txt",
        ]
        emo_path = next((p for p in default_path_candidates if p and os.path.exists(p)), None)
        print(">> Emotions JSON:", emo_path or "(none)")
        print(f"[1단계] 감정 데이터 로드 완료 - {time.time() - start_time:.2f}초")

        # 분석기 생성
        print("[2단계] 분석기 초기화 중...")
        analyzer_start = time.time()
        kiwi = Kiwi() if 'Kiwi' in globals() and Kiwi else None
        analyzer = PsychCogIntegratedAnalyzer(
            config=AnalyzerConfig(),
            emotions_data_path=emo_path,
            emotions_data=None,
            kiwi_instance=kiwi,
        )
        print(f"[2단계] 분석기 초기화 완료 - {time.time() - analyzer_start:.2f}초")

        # 인덱스 통계 로그
        if hasattr(analyzer, "index") and hasattr(analyzer.index, "stats"):
            st = analyzer.index.stats()
            logger.info(
                "Index stats: intensity=%d ctx=%d key=%d combos=%d trans=%d defenses=%d biases=%d",
                st.get("intensity_low", 0) + st.get("intensity_med", 0) + st.get("intensity_high", 0),
                st.get("context_low", 0) + st.get("context_med", 0) + st.get("context_high", 0),
                st.get("key_phrases", 0),
                st.get("sentiment_combinations", 0),
                st.get("transitions", 0),
                st.get("defense_terms", 0),
                st.get("bias_terms", 0),
            )

        # 테스트 케이스
        test_texts = [
            "오늘은 기분이 좋다가도 슬퍼지는 이상한 날이야. 하지만 내일은 더 좋을 거야.",
            "시험 준비하느라 너무 불안했는데, 막상 결과를 받고나니 마음이 편안해졌어. 정말 기쁘고 만족스러워.",
            "친구와 싸워서 화가 났었는데, 대화를 나누다 보니 서로를 이해하게 되었어. 이제는 오히려 관계가 더 돈독해진 것 같아.",
            "오늘은 약간 불안했지만 다시 생각해보니 괜찮을지도 몰라. 그런데 어떤 말이 계기가 되어 마음이 서서히 편안해졌어."
        ]

        all_ok = True
        total_analysis_time = 0
        
        print(f"[3단계] 테스트 케이스 분석 시작 - 총 {len(test_texts)}개 케이스")
        
        for idx, text in enumerate(test_texts, 1):
            print("\n" + "=" * 80)
            print(f"테스트 케이스 #{idx}")
            print("입력 텍스트:", text)
            print("-" * 80)

            # 분석 실행
            analysis_start = time.time()
            print(f"[{idx}번째 케이스] 분석 시작...")
            report = analyzer.analyze(text)
            analysis_time = time.time() - analysis_start
            total_analysis_time += analysis_time
            print(f"[{idx}번째 케이스] 분석 완료 - {analysis_time:.2f}초")
            
            _ = save_run_artifacts(report, keep_history=False)

            # 요약 프린트
            comp = report.get("composite_scores", {})
            timeline = report.get("timeline", []) or []
            insights = report.get("insights", {}) or {}
            flags = report.get("health_flags", {}) or {}

            print("[요약]")
            print("- 문장 수:", len(timeline))
            print("- 전이 수:", sum(1 for t in timeline if t.get("transition") in ("shift_up", "shift_down", "onset", "termination")))
            print("- 성숙도/안정성:", comp.get("maturity", 0.0), "/", comp.get("stability", 0.0))
            print("- 방어/편향 부하:", comp.get("defense_load", 0.0), "/", comp.get("bias_load", 0.0))
            print("- 인사이트:", insights)
            if flags.get("flags"):
                print("- 헬스 플래그:", flags.get("flags"))

            # 전체 리포트 JSON
            print("\n[전체 리포트(JSON)]")
            print(json.dumps(report, ensure_ascii=False, indent=2))

            # 오케스트레이터 포맷 변환 출력
            print("\n-- data_utils_format --")
            print(json.dumps(to_data_utils_format(report), ensure_ascii=False, indent=2))

            # 간단 구조 점검
            if not isinstance(timeline, list) or not isinstance(comp, dict) or "value" not in report.get("maturity_detail", {}):
                logger.warning("구조 점검 경고: 리포트 스키마에 예상 키가 일부 누락됨")
                all_ok = False
            if any(len(f.get("evidence", [])) == 0 for f in timeline):
                logger.debug("일부 문장에서 evidence가 비어있음")

        # 요약
        total_time = time.time() - start_time
        print("\n" + "=" * 80)
        print("테스트 완료 요약:")
        print("- 처리된 테스트 케이스:", len(test_texts))
        print("- 총 분석 시간:", f"{total_analysis_time:.2f}초")
        print("- 평균 케이스당 분석 시간:", f"{total_analysis_time/len(test_texts):.2f}초")
        print("- 전체 실행 시간:", f"{total_time:.2f}초")
        
        if all_ok:
            print("- 분석기 성능: 정상 작동 (기본 스키마/전이/지표 산출 확인)")
            logger.info("모든 테스트 케이스가 정상적으로 분석되었습니다.")
        else:
            print("- 일부 구조 경고 발생 (로그 확인)")
            logger.warning("일부 케이스에서 경고가 발생했습니다. 로그를 확인하세요.")
        
        print("=" * 80)
        print("심리·인지 통합 분석기 실행 완료")
        print("=" * 80)

    except Exception as e:
        logger.exception("프로그램 실행 중 예외 발생")
        print(f"오류 발생: {e}")
    finally:
        logger.info("프로그램 종료")


# -----------------------------------------------------------------------------
# __main__
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
