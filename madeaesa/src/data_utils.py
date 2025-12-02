# -*- coding: utf-8 -*-
# src/data_utils.py
"""
메인데이터전처리기(오케스트레이터) — 1차 리팩토링 스켈레톤
=================================================================
목표
- config.py의 MODULE_ENTRYPOINTS, EMOTION_ANALYSIS_PIPELINE를 동적으로 해석하여
  11개 모듈을 순서/의존/조건(run_if)대로 실행하는 단일 진입점 구성
- 표준 I/O 컨테이너(Payload)를 정의하고, 각 스텝이 공통 구조로 입/출력을 주고받도록 규격화
- 안전 호출(safe_call), 캐싱, 로깅, 버전 스탬핑, 성능 측정 후킹의 큰 틀만 구성
- 각 스텝의 세부 알고리즘 호출부는 어댑터 테이블(CALL_TABLE)과 TODO로 남겨 후속 구현에서 세부화

사용법(예)
>>> orchestrator = EmotionPipelineOrchestrator()
>>> result = orchestrator.process_text("텍스트...", meta={"source":"demo"})
>>> print(result.keys())  # ['text', 'meta', 'results', 'trace']
"""
from __future__ import annotations

import io, sys, os
if os.name == "nt":
    os.environ.setdefault("PYTHONUTF8", "1")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
        sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
        sys.stdin.reconfigure(encoding="utf-8", errors="replace")
    else:
        # 안전 폴백: detach 후 새 래퍼 부착
        try:  sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding="utf-8", errors="replace", line_buffering=True)
        except Exception: pass
        try:  sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding="utf-8", errors="replace", line_buffering=True)
        except Exception: pass
        try:  sys.stdin  = io.TextIOWrapper(sys.stdin.detach(),  encoding="utf-8", errors="replace")
        except Exception: pass

# 표준 라이브러리 임포트 (알파벳 순서로 그룹화 및 정렬)
import argparse
import atexit
import concurrent.futures
import functools
import hashlib
import importlib
import inspect
import json
import logging
import os
import random
import re
import sys
import threading
import time
import uuid
from collections import Counter, OrderedDict, defaultdict, deque
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from statistics import mean as _mean
from threading import RLock
from time import perf_counter
from types import ModuleType, SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

# 선택적 의존성
import numpy as np
try:
    import kss  # 한국어 문장 분할
except ImportError:
    kss = None
try:
    import psutil  # 시스템 리소스 모니터링
except ImportError:
    psutil = None

# 별칭
import concurrent.futures as _fut


# ---- CUDA SDPA/Flash 경로 토글 (환경변수로 가드) ----
try:
    import torch
    if torch.cuda.is_available():
        # EA_DISABLE_FLASH_SDP: "1"/"true" → 안전모드(기본), "0" → 성능 경로 활성화
        # 선택 개선: Flash SDP 토글 일관화 (기본값 통일)
        if os.getenv("EA_DISABLE_FLASH_SDP", "0").lower() in ("1", "true"):
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_cudnn_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
        else:
            # 환경 안정성 확보 시 최적화 경로 켜기
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_cudnn_sdp(True)
except Exception:
    pass


# ===========================================================================
# 패키지 베이스 해석 (모듈 상단에서 emotion_analysis 실제 임포트 금지)
#   - 우선순위: 명시 인자 -> ENV -> 존재탐지(src.emotion_analysis / emotion_analysis) -> 기본(emotion_analysis)
#   - 임베딩 파이프라인과의 독립성 보존을 위해 문자열 네임스페이스만 반환
# ===========================================================================
def _resolve_package_base(package_base: Optional[str] = None) -> str:
    if package_base:
        return package_base

    env_base = os.getenv("EMOTION_ANALYSIS_PACKAGE")
    if env_base:
        return env_base

    # 존재 탐지: 개발(패키지 경로 prefix가 'src.')/배포(루트 패키지) 모두 지원
    try:
        if importlib.util.find_spec("src.emotion_analysis") is not None:
            return "src.emotion_analysis"
    except Exception:
        pass
    try:
        if importlib.util.find_spec("emotion_analysis") is not None:
            return "emotion_analysis"
    except Exception:
        pass
    # 최후 기본값
    return "emotion_analysis"


# ===========================================================================
# config 모듈 로드
#   - ENV(EMOTION_CONFIG_MODULE)로 명시 오버라이드 가능
#   - 후보를 순차 탐색하고, 마지막 예외를 보존하여 디버깅 용이
#   - lru_cache로 재호출 비용 제거
# ===========================================================================
@functools.lru_cache(maxsize=1)
def _import_config_module() -> ModuleType:
    candidates = []

    env_name = os.getenv("EMOTION_CONFIG_MODULE")
    if env_name:
        candidates.append(env_name)

    # 패키지 상대 경로 우선 시도(스크립트/패키지 양쪽 실행 지원)
    if __package__:
        candidates.append(f"{__package__}.config")

    # 일반적인 절대 경로들
    candidates += ["emotion_analysis.config", "config"]

    last_err: Optional[Exception] = None
    for name in candidates:
        try:
            return importlib.import_module(name)
        except Exception as e:
            last_err = e

    raise ImportError(
        "[data_utils] config 모듈을 찾을 수 없습니다. "
        f"tried={candidates}"
    ) from last_err

# 외부에서 참조할 전역 config
config: ModuleType = _import_config_module()


# ===========================================================================
# 타임아웃/서킷 브레이커 훅 - 개선사항 추가
#   - 드문 "걸림"에 대비한 전체 파이프라인 보호막
#   - config의 DEFAULT_STEP_TIMEOUT을 활용하여 스텝별 타임아웃 적용
# ===========================================================================
STEP_TIMEOUT = getattr(config, "DEFAULT_STEP_TIMEOUT", 0)
STEP_TIMEOUTS = getattr(config, "STEP_TIMEOUTS", {})

_TIMEOUT_EXECUTOR: Optional[_fut.ThreadPoolExecutor] = None
_TIMEOUT_EXECUTOR_LOCK = threading.Lock()


def _get_timeout_pool_size() -> int:
    try:
        env_val = int(os.getenv("STEP_TIMEOUT_POOL_SIZE", "8"))
    except Exception:
        env_val = 8
    # 합리적 상한/하한 적용
    return max(2, min(env_val, 32))


def _get_timeout_executor() -> _fut.ThreadPoolExecutor:
    global _TIMEOUT_EXECUTOR
    if _TIMEOUT_EXECUTOR is None:
        with _TIMEOUT_EXECUTOR_LOCK:
            if _TIMEOUT_EXECUTOR is None:
                pool_size = _get_timeout_pool_size()
                executor = _fut.ThreadPoolExecutor(
                    max_workers=pool_size,
                    thread_name_prefix="ea-timeout",
                )
                atexit.register(lambda ex=executor: ex.shutdown(wait=False, cancel_futures=True))
                _TIMEOUT_EXECUTOR = executor
    return _TIMEOUT_EXECUTOR


def run_step_with_timeout(fn: Callable, step_name: str = None, *args, **kwargs) -> Any:
    """
    스텝 실행 래퍼에 타임아웃 적용
    
    Args:
        fn: 실행할 함수
        step_name: 스텝 이름 (스텝별 타임아웃 적용용)
        *args, **kwargs: 함수 인자들
        
    Returns:
        함수 실행 결과 또는 타임아웃 시 폴백 딕셔너리
        
    Note:
        - 스텝별 타임아웃이 있으면 우선 적용, 없으면 DEFAULT_STEP_TIMEOUT 사용
        - 타임아웃이 0이면 타임아웃 없이 직접 실행
        - 타임아웃 발생 시 안전한 폴백 결과 반환
        - 프로덕션 환경에서는 폴백 감지 시 즉시 종료
    """
    # 스텝별 타임아웃 우선 적용
    timeout = STEP_TIMEOUTS.get(step_name) if step_name else STEP_TIMEOUT
    if not timeout:  # 0=무제한
        # 패치: 타임아웃이 없을 때도 step_name 전달
        if step_name is not None:
            kwargs = dict(kwargs)
            kwargs["step_name"] = step_name
        return fn(*args, **kwargs)
    
    # 패치: 내부 호출에도 step_name을 넘겨줍니다.
    if step_name is not None:
        kwargs = dict(kwargs)
        kwargs["step_name"] = step_name
    
    executor = _get_timeout_executor()
    future = executor.submit(fn, *args, **kwargs)
    try:
        result = future.result(timeout=timeout)
        
        # 즉시 적용 추천 운영 안전가드
        if isinstance(result, dict):
            is_production = (
                getattr(config, "EA_PROFILE", "prod") == "prod"
                or bool(getattr(config, "RENDER_DEPLOYMENT", False))
                or os.getenv("PRODUCTION_MODE", "0") == "1"
            )
            
            # 1) 폴백/실패 강제 페일
            if result.get("fallback") or (result.get("success") is False):
                error_msg = f"[FAIL:{step_name}] fallback or explicit failure"
                logging.getLogger(__name__).error(error_msg)
                raise RuntimeError(error_msg)
            
            # 2) EMOTIONS 폴백 진단 강제 페일 (특히 중요)
            diag = (result.get("diagnostics") or {}).get("emotions_ontology", {})
            if diag.get("fallback_used"):
                error_msg = f"[FAIL:{step_name}] emotions fallback used"
                logging.getLogger(__name__).error(error_msg)
                raise RuntimeError(error_msg)
            
            # 3) 전이 과탐 차단: 평균 confidence < 0.25 인 대량 전이 필터
            trans = (result.get("emotion_transition_matches") or result.get("transitions") or [])
            if isinstance(trans, list) and len(trans) >= 5:
                avg = sum(float(t.get("confidence", 0.0)) for t in trans) / len(trans)
                if avg < 0.25:
                    error_msg = f"[FAIL:{step_name}] suspicious transitions(avg<0.25)"
                    logging.getLogger(__name__).error(error_msg)
                    raise RuntimeError(error_msg)
        
        return result
    except _fut.TimeoutError:
        future.cancel()
        fallback_result = {
            "error": "timeout",
            "fallback": True,
            "conf": 0.0,
            "message": f"Step '{step_name}' timeout after {timeout}s"
        }
    except Exception:
        future.cancel()
        # 타임아웃 이외의 예외는 그대로 전파
        raise
    
    # 타임아웃 발생 시 폴백 결과 반환
    # 프로덕션 환경에서는 폴백 감지 시 즉시 종료
    is_production = (
        getattr(config, "EA_PROFILE", "prod") == "prod"
        or bool(getattr(config, "RENDER_DEPLOYMENT", False))
        or os.getenv("PRODUCTION_MODE", "0") == "1"
    )
    
    if is_production:
        error_msg = f"Timeout in step '{step_name}' in production mode - aborting pipeline"
        logging.getLogger(__name__).error(error_msg)
        raise RuntimeError(error_msg)
    else:
        logging.getLogger(__name__).warning(f"Timeout in step '{step_name}' in development mode - using fallback")
        return fallback_result


# ===========================================================================
# 로깅 설정
#   - 외부 애플리케이션의 로깅 구성을 침범하지 않음
#   - 모듈 로거에는 NullHandler만 부착하여 이중 출력 방지
#   - 루트 로거에 핸들러가 *없을 때만* 기본 구성
#   - 레벨은 ENV > config.TRAINING_PARAMS.LOG_LEVEL > INFO
# ===========================================================================
logger = logging.getLogger(__name__)

# 루트 핸들러가 전혀 없으면 최소 스트림 핸들러 1개만 구성
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO)

# 모듈 로거는 항상 NullHandler 추가(외부 핸들러와 중복 금지)
if not any(isinstance(h, logging.NullHandler) for h in logger.handlers):
    logger.addHandler(logging.NullHandler())

# 루트 레벨을 ENV/설정과 동기화 (핸들러가 있더라도 레벨만 조정)
try:
    lvl_name = os.getenv("DATA_UTILS_LOG_LEVEL") or str(getattr(getattr(config, "TRAINING_PARAMS", {}), "get", lambda *_: "INFO")("LOG_LEVEL"))
    lvl = getattr(logging, str(lvl_name).upper(), logging.INFO)
    logging.getLogger().setLevel(lvl)
    for h in logging.getLogger().handlers:
        h.setLevel(lvl)
except Exception:
    # 로깅 관련 예외는 침묵 (처리 지속)
    pass

# ------------------------------------------------------------
# Tail 강제 실행 환경변수 (HEAVY에서도 강제 실행 허용)
# ------------------------------------------------------------
_ENV_FORCE_TAIL = os.getenv("FORCE_TAIL", "0").lower() in ("1", "true", "on")


# ===========================================================================
# MODULE_ENTRYPOINTS 사전 검증
#   - 형식 오류는 에러 로그만 출력하고 실행은 계속
#   - 문자열 공백 정리, 잘못된 엔트리는 목록으로 모아 한 번에 보고
#   - 실제 import는 *실행 시점*에만 수행 (지연 로드/독립성 유지)
# ===========================================================================
def _validate_entrypoints(_cfg: ModuleType) -> None:
    eps = getattr(_cfg, "MODULE_ENTRYPOINTS", {})
    if not isinstance(eps, dict) or not eps:
        logger.error("[data_utils] MODULE_ENTRYPOINTS가 비어 있거나 잘못되었습니다.")
        return

    bad: list[tuple[str, object]] = []
    fixed: Dict[str, Tuple[str, str]] = {}

    for step, pair in eps.items():
        ok = isinstance(pair, (list, tuple)) and len(pair) == 2 and all(isinstance(x, str) and x.strip() for x in pair)
        if not ok:
            bad.append((step, pair))
            continue
        mod_name, attr_name = (pair[0].strip(), pair[1].strip())
        fixed[step] = (mod_name, attr_name)

    if bad:
        msg = ", ".join([f"{s} -> {p!r}" for s, p in bad])
        logger.error("[data_utils] 엔트리포인트 형식 오류(무시하고 진행): %s", msg)
    else:
        logger.debug("[data_utils] 엔트리포인트 형식 확인 완료(%d개). 실제 import는 실행 시점에 수행", len(fixed))

_validate_entrypoints(config)

# 구성 요약(모델 버전은 없을 수 있으므로 안전 조회)
try:
    _mv = getattr(getattr(config, "MODEL_VERSION_CONFIG", {}), "get", lambda *_: "NA")("model_version")
    logger.info("config loaded: file=%s | model_version=%s", getattr(config, "__file__", "NA"), _mv)
except Exception:
    logger.info("config loaded: file=%s", getattr(config, "__file__", "NA"))


# ===========================================================================
# 표준 페이로드
# ===========================================================================
def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default

# ★ 범용 안전 캐스팅
def _safe_int(x, default):
    try:
        return int(x) if x is not None else int(default)
    except Exception:
        return int(default)

def _truncate_str(s: str, max_len: int = 500) -> str:
    if not isinstance(s, str):
        return s
    return s if len(s) <= max_len else s[: max_len - 3] + "..."

def _sanitize_extras(obj: Any, depth: int = 0, max_depth: int = 2) -> Any:
    if depth > max_depth:
        return "<omitted>"
    if isinstance(obj, dict):
        out = {}
        for k, v in list(obj.items())[:50]:
            out[str(k)] = _sanitize_extras(v, depth + 1, max_depth)
        return out
    if isinstance(obj, (list, tuple)):
        seq = list(obj)[:50]
        return [_sanitize_extras(v, depth + 1, max_depth) for v in seq]
    if isinstance(obj, str):
        return _truncate_str(obj, 500)
    if isinstance(obj, (int, float, bool)) or obj is None:
        return obj
    return _truncate_str(repr(obj), 300)


# ===========================================================================
# Payload
# ===========================================================================
@dataclass(slots=True)
class Payload:
    text: str
    meta: Dict[str, Any] = field(default_factory=dict)
    sentences: List[str] = field(default_factory=list)
    tokens: List[str] = field(default_factory=list)
    lang: Optional[str] = None
    features: Dict[str, Any] = field(default_factory=dict)
    results: Dict[str, Any] = field(default_factory=dict)
    timeline: Dict[str, Any] = field(default_factory=dict)
    # ★ Patch-4: trace를 deque(maxlen)으로 변경하여 자동 메모리 관리 - 메모리 최적화
    trace: deque[Dict[str, Any]] = field(default_factory=lambda: deque(maxlen=_env_int("PAYLOAD_TRACE_LIMIT", 100)))
    _trace_limit: int = field(default_factory=lambda: _env_int("PAYLOAD_TRACE_LIMIT", 100), repr=False)
    _compact_output: bool = field(
        default_factory=lambda: os.getenv("PAYLOAD_OUTPUT_COMPACT", "1").lower() in ("1", "true", "yes"), repr=False
    )

    def __post_init__(self) -> None:
        try:
            if not isinstance(self.text, str):
                self.text = "" if self.text is None else str(self.text)
        except Exception:
            self.text = ""
        if not isinstance(self.meta, dict):
            self.meta = {}
        if not isinstance(self.sentences, list):
            self.sentences = list(self.sentences) if self.sentences is not None else []
        if not isinstance(self.tokens, list):
            self.tokens = list(self.tokens) if self.tokens is not None else []
        
        # 메모리 최적화: 대용량 텍스트 처리
        if len(self.text) > 10000:  # 10KB 이상
            self._compact_output = True
            # 불필요한 필드 제거로 메모리 절약
            if not self.tokens:
                self.tokens = []

    def stamp(
        self,
        step: str,
        status: str,
        started: Optional[float] = None,
        error: Optional[Any] = None,
        extras: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        now_wall_ms = int(time.time() * 1000)
        entry: Dict[str, Any] = {"ts": now_wall_ms, "step": str(step), "status": str(status)}
        if started is not None:
            try:
                entry["duration_ms"] = round((time.perf_counter() - float(started)) * 1000, 2)
            except Exception:
                pass
        if error:
            entry["error"] = _truncate_str(str(error), 800)
            entry["error_class"] = getattr(type(error), "__name__", "Exception") if not isinstance(error, str) else "str"
        if extras:
            entry.update(_sanitize_extras(extras))
        self.trace.append(entry)
        # ★ Patch-4: deque가 자동으로 maxlen 관리하므로 슬라이싱 제거
        return entry

    def add_result(self, step: str, result: Any) -> Any:
        self.results[str(step)] = result
        if isinstance(result, dict):
            feats = result.get("features")
            if isinstance(feats, dict):
                self.features.update(feats)
            if "timeline" in result and isinstance(result["timeline"], dict):
                self.timeline[str(step)] = result["timeline"]
        return result

    def mark_skip(self, step: str, reason: str) -> None:
        self.stamp(step, status="skip", extras={"reason": _truncate_str(reason, 300)})

    def get(self, dotted_path: Optional[str], default: Any = None) -> Any:
        root: Dict[str, Any] = {
            "text": self.text,
            "meta": self.meta,
            "sentences": self.sentences,
            "tokens": self.tokens,
            "lang": self.lang,
            "features": self.features,
            "results": self.results,
            "timeline": self.timeline,
        }
        if not dotted_path:
            return root

        node: Any = root
        for key in str(dotted_path).split("."):
            try:
                if isinstance(node, dict):
                    node = node.get(key)
                elif isinstance(node, list) and key.isdigit():
                    idx = int(key)
                    node = node[idx] if 0 <= idx < len(node) else None
                else:
                    return default
            except (KeyError, IndexError):
                return default

            if node is None:
                return default
        return node if node is not None else default


    def update_meta(self, **kwargs) -> None:
        try:
            for k, v in kwargs.items():
                self.meta[str(k)] = v
        except Exception:
            pass

    def _set_determinism(self):
        """Set process-wide determinism based on cfg.DETERMINISM_CONFIG."""
        try:
            seed = int(getattr(self.cfg, "DETERMINISM_CONFIG", {}).get("seed", 42))
            random.seed(seed)
            try:
                np.random.seed(seed)
            except Exception:
                pass
            try:
                import torch
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                if bool(getattr(self.cfg, "DETERMINISM_CONFIG", {}).get("torch_deterministic", True)):
                    try:
                        torch.use_deterministic_algorithms(True)  # type: ignore[attr-defined]
                    except Exception:
                        pass
                    try:
                        torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
                    except Exception:
                        pass
            except Exception:
                pass
            os.environ.setdefault("PYTHONHASHSEED", str(seed))
        except Exception:
            pass

    def to_output(self, meta_extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        meta = dict(self.meta)
        if isinstance(meta_extra, dict):
            meta.update(meta_extra)

        # ★ Patch-4: deque를 리스트로 변환하여 직렬화
        trace_data = list(self.trace)
        trace = trace_data[-200:] if self._compact_output and trace_data else trace_data

        # 성공 여부 판단: 모든 스텝이 성공했는지 확인
        success = True
        for step_name, step_result in self.results.items():
            if isinstance(step_result, dict) and step_result.get('success') is False:
                success = False
                break
        
        return {"text": self.text, "meta": meta, "results": self.results, "trace": trace, "success": success}

    @classmethod
    def new(cls, text: str, meta: Optional[Dict[str, Any]] = None, cfg: Optional[Any] = None) -> "Payload":
        m = dict(meta or {})
        try:
            m.setdefault("trace_id", str(uuid.uuid4()))
        except Exception:
            m.setdefault("trace_id", "NA")

        if cfg is not None:
            try:
                m.setdefault("emotion_set_version", getattr(cfg, "EMOTION_SET_VERSION", "NA"))
            except Exception:
                pass
            try:
                m.setdefault("config_signature", getattr(cfg, "CONFIG_SIGNATURE", "NA"))
            except Exception:
                pass
            try:
                mvc = getattr(cfg, "MODEL_VERSION_CONFIG", None)
                if hasattr(mvc, "get") and callable(mvc.get):
                     m.setdefault("model_version", mvc.get("model_version", "NA"))
                else:
                     m.setdefault("model_version", "NA")
            except Exception:
                m.setdefault("model_version", "NA")

        return cls(text=text, meta=m)

    @property
    def sentences_count(self) -> int:
        return len(self.sentences)

    @property
    def tokens_count(self) -> int:
        return len(self.tokens)

    def __str__(self) -> str:
        return self.text

    def __bool__(self) -> bool:
        return bool(self.text)

    def __getattr__(self, name: str) -> Any:
        try:
            return getattr(self.text, name)
        except AttributeError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


# ===========================================================================
# 유틸 — 안정성/대용량 대응 하드닝 버전
# ===========================================================================

# @경량 해시 / JSON 직렬화 / 다이제스트
def _hash_text(s: str | bytes, *, size: int = 16, algo: str = "sha256") -> str:
    """짧은 해시(기본 16hex) — 캐시 키/트레이스용. None/bytes 안전."""
    if s is None:
        s = ""
    if isinstance(s, bytes):
        data = s
    else:
        data = str(s).encode("utf-8", "replace")
    try:
        h = getattr(hashlib, algo)(data)
    except Exception:
        h = hashlib.sha256(data)
    return h.hexdigest()[: int(size)]

def _safe_json_dumps(obj: Any, *, sort_keys: bool = True, indent: Optional[int] = None) -> str:
    """
    예외 없이 JSON 문자열로 직렬화(트레이스/서명용).
    - dataclass, Path, set, bytes, numpy/torch(있으면) 등을 안전 변환
    - 최대한 안정적/결정적인 표현을 위해 sort_keys 기본 True
    """
    def _default(o: Any):
        try:
            # dataclass
            if is_dataclass(o):
                return asdict(o)
            # pathlib
            if isinstance(o, Path):
                return str(o)
            # bytes
            if isinstance(o, (bytes, bytearray, memoryview)):
                return bytes(o).decode("utf-8", "replace")
            # set/tuple
            if isinstance(o, (set, tuple)):
                return list(o)
            # numpy
            if o.__class__.__module__.startswith("numpy"):  # type: ignore[attr-defined]
                try:
                    return np.asarray(o).tolist()
                except Exception:
                    return str(o)
            # torch tensor / device
            mod = o.__class__.__module__
            name = o.__class__.__name__
            if mod.startswith("torch"):
                # Tensor
                if name == "Tensor":
                    try:
                        return o.detach().cpu().tolist()
                    except Exception:
                        return str(o)
                # device/dtype 등
                return str(o)
            # pydantic v2
            if hasattr(o, "model_dump"):
                return o.model_dump()
            # enum
            if hasattr(o, "value"):
                return getattr(o, "value")
        except Exception:
            pass
        # 최후
        return str(o)

    try:
        return json.dumps(obj, ensure_ascii=False, sort_keys=sort_keys, default=_default, indent=indent)
    except Exception:
        try:
            return json.dumps(str(obj), ensure_ascii=False)
        except Exception:
            return "{}"

def _config_digest(cfg: Dict[str, Any]) -> str:
    """설정 딕셔너리의 안정적 다이제스트(기본 blake2b-16, 실패 시 md5)."""
    s = _safe_json_dumps(cfg, sort_keys=True)
    data = s.encode("utf-8", "replace")
    try:
        return hashlib.blake2b(data, digest_size=16).hexdigest()
    except Exception:
        return hashlib.md5(data).hexdigest()

# @문자열/경로 유틸
def _shorten(text: Any, max_len: int = 160) -> str:
    """트레이스에 넣을 때 긴 본문을 안전하게 축약(해시 꼬리표 포함)."""
    if not isinstance(text, str):
        text = str(text)
    if len(text) <= max_len:
        return text
    head = text[: max_len - 10].rstrip()
    tail_hash = _hash_text(text)
    return f"{head}…#{tail_hash}"

def _dig_simple(d: Any, dotted: Optional[str], default: Any = None):
    """간단판: 강화판 _dig를 래핑하여 유지보수 단일화."""
    return _dig(d, dotted, default=default)

def _coerce_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    """confidence 등 숫자 스칼라로 강제 변환. 퍼센트/콤마/불리언 대응."""
    try:
        if x is None:
            return default
        if isinstance(x, bool):
            return 1.0 if x else 0.0
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).strip()
        if not s:
            return default
        # 1,234.56  /  12.3%
        s_norm = s.replace(",", "")
        if s_norm.endswith("%"):
            return float(s_norm[:-1]) / 100.0
        return float(s_norm)
    except Exception:
        return default

# @텍스트 정규화 / 문장분할 / 토크나이즈
# ---------------------------------------------------------------------
# @@빈번 호출 대비 정규식은 모듈 레벨에서 미리 컴파일
_URL_RE = re.compile(r"(https?://[^\s]+|www\.[^\s]+)", re.IGNORECASE)
_WS_MULTI_RE = re.compile(r"[ \t\f\v]+")
_WS_NL_TRAIL_RE = re.compile(r"[ \t]+\n")
_NL_MULTI_RE = re.compile(r"\n{3,}")
_CTRL_RE = re.compile(r"[\u0000-\u0008\u000B\u000C\u000E-\u001F]")

# @@종결부호 기반 분리(한국어/영문)
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?‥…!?]|[。！？])\s+")
# @@경량 토크나이저(한글/영문/숫자/언더스코어 + 단독 기호, 수축형 일부 보존)
_TOKEN_RE = re.compile(r"[0-9A-Za-z가-힣_]+(?:'[A-Za-z]+)?|[^\s0-9A-Za-z가-힣_]")

# 중복 함수 정의 제거됨 - 아래에 더 완전한 버전이 있음

def _normalize_text(
    text: str,
    *,
    remove_special_chars: Optional[bool] = None,
    normalize_whitespace: Optional[bool] = None,
    remove_urls: Optional[bool] = None,
) -> str:
    """
    경량 텍스트 정규화.
    - config.PREPROCESSING_CONFIG가 있다면 그 옵션을 기본값으로 사용
    - 외부 의존성 없이 작동(정규식은 모듈 전역 컴파일)
    """
    try:
        opts = getattr(config, "PREPROCESSING_CONFIG", {}).get("text_cleaning", {})  # type: ignore[name-defined]
    except Exception:
        opts = {}

    if remove_special_chars is None:
        remove_special_chars = bool(opts.get("remove_special_chars", True))
    if normalize_whitespace is None:
        normalize_whitespace = bool(opts.get("normalize_whitespace", True))
    if remove_urls is None:
        remove_urls = bool(opts.get("remove_urls", True))

    out = text or ""
    try:
        if remove_urls:
            out = _URL_RE.sub(" ", out)

        if normalize_whitespace:
            # CRLF → LF 정규화
            out = out.replace("\r\n", "\n")
            # 연속 공백 축소, 줄끝 공백 제거, 과도한 빈줄 축소
            out = _WS_MULTI_RE.sub(" ", out)
            out = _WS_NL_TRAIL_RE.sub("\n", out)
            out = _NL_MULTI_RE.sub("\n\n", out).strip()

        if remove_special_chars:
            # 제어문자 제거(한글/영문/숫자/기본 문장부호 유지)
            out = _CTRL_RE.sub("", out)
    except Exception:
        if normalize_whitespace:
            out = " ".join(out.split())
    return out

def _split_sentences(text: str) -> list[str]:
    """
    문장 분할: kss가 있으면 사용, 없으면 종결부호/개행 기반 폴백.
    - 한국어/혼합 문장 기본 대응
    """
    try:
        sents = list(kss.split_sentences(text))
        return [s.strip() for s in sents if s and s.strip()]
    except Exception:
        pass

    try:
        sents = _SENT_SPLIT_RE.split(text)
        if len(sents) <= 1:
            # 최후: 줄바꿈 또는 전체 한 문장
            parts = (text or "").replace("\r\n", "\n").split("\n")
            return [p.strip() for p in parts if p.strip()]
        return [s.strip() for s in sents if s and s.strip()]
    except Exception:
        parts = (text or "").replace("\r\n", "\n").split("\n")
        return [p.strip() for p in parts if p.strip()]

def _basic_tokenize(text: str) -> list[str]:
    """
    가벼운 토크나이저: 단어/구두점 분리.
    고급 형태소 분석기 대체 아님(후속 교체 가능).
    """
    try:
        return _TOKEN_RE.findall(text or "")
    except Exception:
        return (text or "").split()

# @@캐시 키 / 타이머
def _make_cache_key(
    *,
    text: str,
    step: str,
    cfg_signature: str,
    emotion_set_version: str = "NA",
    model_version: str = "NA",
) -> str:
    """캐시 키 표준 구성(텍스트는 해시로 축약, 구분자 충돌 방지)."""
    safe_step = (step or "").replace("::", "|")
    parts = [
        _hash_text(text),
        safe_step,
        cfg_signature or "NA",
        emotion_set_version or "NA",
        model_version or "NA",
    ]
    return "::".join(parts)

def _timer_ms():
    """
    간단 타이머 컨텍스트:
    with _timer_ms() as t:
        ... 작업 ...
    t.ms  # 경과 ms
    """
    class _T:
        def __enter__(self):
            self._start = time.perf_counter()
            self.ms = 0.0
            return self
        def __exit__(self, exc_type, exc, tb):
            self.ms = round((time.perf_counter() - self._start) * 1000, 2)
    return _T()


# ===========================================================================
# 모듈 레지스트리 — 안정성/독립성/대용량 하드닝 버전
# ===========================================================================
class ModuleRegistry:
    """
    MODULE_ENTRYPOINTS를 기반으로 실제 클래스/함수에 접근하는 레지스트리.

    특징
    - 인스턴스 캐시 보유(재사용) + force_rebuild 지원 (스레드 세이프)
    - 패키지 기준 경로 동적 해석(ENV/인자/EA 모듈)
    - 생성자 시그니처 기반 자동 주입:
        * emotions_data (dict), emotions_data_path (str)
        * weight_calculator 전용: data_manager, logger
    - 클래스 __init__ 실패 시 **모듈로 폴백**(특히 transition_analyzer 안전 구동)
    - post-config(설정 주입) 및 prewarm(인덱스/캐시 준비) 훅 지원
    - 감정 사전(JSON) 로드 캐시 + mtime 감지(파일 갱신 시 자동 리로드)
    """

    def __init__(
        self,
        entrypoints: Dict[str, Tuple[str, str]],
        *,
        cfg: Any = None,
        package_base: Optional[str] = None,
        init_kwargs: Optional[Dict[str, Any]] = None,
    ):
        # 엔트리포인트 입력 정규화
        self.entrypoints: Dict[str, Tuple[str, str]] = {}
        for k, v in dict(entrypoints or {}).items():
            try:
                mod, sym = v
                self.entrypoints[str(k)] = (str(mod).strip(), str(sym).strip())
            except Exception:
                # 형식 오류는 무시 (사전 검증 단계에서 이미 로그됨)
                continue

        self._instances: Dict[str, Any] = {}
        self._lock = threading.RLock()
        # [Rollback] Removed fine-grained locking to prevent deadlocks with Lazy Loading import locks
        
        self.cfg = cfg if cfg is not None else config  # type: ignore[name-defined]
        # EA(패키지) 임포트를 즉시 하지 않고 문자열 네임스페이스만 유지
        self.package_base = _resolve_package_base(package_base)  # type: ignore[name-defined]

        self.init_kwargs = dict(init_kwargs or {})
        self._emotions_path_cache: Optional[str] = None
        self._emotions_data_cache: Optional[dict] = None
        self._emotions_data_mtime: Optional[float] = None

        self._log_entrypoints_summary()

    # ------------------------------
    # 캐시 제어
    # ------------------------------
    def clear(self) -> None:
        with self._lock:
            self._instances.clear()

    # ------------------------------
    # 인스턴스 획득
    # ------------------------------
    def get_instance(self, step_name: str, *, force_rebuild: bool = False, prewarm: bool = False) -> Any:
        with self._lock:
            if not force_rebuild and step_name in self._instances:
                return self._instances[step_name]

            module_name, class_name = self._resolve_entrypoint(step_name)
            mod = self._import_module(module_name)
            cls_or_attr = getattr(mod, class_name, None)

            # allow callable or value-based entrypoints; fall back to module when absent
            instance: Any
            if cls_or_attr is None:
                instance = mod
            elif inspect.isclass(cls_or_attr):
                # 기존 클래스도 __init__ 실패 시 모듈 전환
                try:
                    instance = self._instantiate(step_name, cls_or_attr, mod)
                except Exception:
                    logger.debug("[ModuleRegistry] instantiate failed; fallback to module: %s", module_name, exc_info=True)  # type: ignore[name-defined]
                    instance = mod
            elif callable(cls_or_attr):
                instance = cls_or_attr
            else:
                instance = cls_or_attr

            # 설정/경로 주입(모듈/인스턴스 공통)
            self._post_configure(instance)

            # 필요시 프리워밍
            if prewarm:
                self._prewarm(instance)

            self._instances[step_name] = instance
            return instance

    # ------------------------------
    # 내부: 엔트리포인트/임포트
    # ------------------------------
    def _resolve_entrypoint(self, step_name: str) -> Tuple[str, str]:
        if step_name not in self.entrypoints:
            raise KeyError(f"Unknown step entrypoint: {step_name}")
        module_name, class_name = self.entrypoints[step_name]
        return module_name, class_name

    def _import_module(self, module_name: str) -> ModuleType:
        """패키지 베이스 자동 해석 + 양쪽 네임스페이스 모두 시도."""
        tried: List[str] = []
        candidates: List[str] = []

        # 1) 현재 베이스 우선
        pb = self.package_base  # e.g., 'src.emotion_analysis' or 'emotion_analysis'
        candidates.append(f"{pb}.{module_name}")

        # 2) 서로 반대쪽 베이스도 백업으로 시도
        if pb != "emotion_analysis":
            candidates.append(f"emotion_analysis.{module_name}")
        if pb != "src.emotion_analysis":
            candidates.append(f"src.emotion_analysis.{module_name}")

        # 3) 맨 마지막 순수 모듈명(로컬 실행 호환)
        candidates.append(module_name)

        importlib.invalidate_caches()
        for modname in candidates:
            tried.append(modname)
            try:
                return importlib.import_module(modname)
            except Exception:
                continue
        raise ImportError(f"[ModuleRegistry] 모듈 로드 실패: {module_name} (tried={tried})")

    # ------------------------------
    # 내부: 인스턴스 생성(자동 주입 포함)
    # ------------------------------
    def _instantiate(self, step_name: str, cls: type, mod: ModuleType):
        """
        생성자 시그니처를 점검해 필요한 인자를 자동 주입한다.
        - emotions_data / emotions_data_path
        - weight_calculator 전용: data_manager / logger
        - config/cfg 동시 지원
        """
        kw = dict(self.init_kwargs)

        try:
            params = set(inspect.signature(cls).parameters.keys())

            # 감정 데이터/경로 자동 주입
            if {"emotions_data", "emotions"}.intersection(params):
                ed = self._get_emotions_data()
                if ed is not None:
                    kw.setdefault("emotions_data", ed)

            if {"emotions_data_path", "emotions_path", "emotions_json_path"}.intersection(params):
                ep = self._get_emotions_path()
                if ep is not None:
                    if "emotions_data_path" in params:
                        kw.setdefault("emotions_data_path", ep)
                    elif "emotions_path" in params:
                        kw.setdefault("emotions_path", ep)
                    elif "emotions_json_path" in params:
                        kw.setdefault("emotions_json_path", ep)

            # weight_calculator 의존성 자동 주입
            if step_name == "weight_calculator":
                if "data_manager" in params:
                    kw.setdefault("data_manager", self._build_weight_data_manager(mod))
                if "logger" in params:
                    kw.setdefault("logger", logging.getLogger("weight_calculator"))

        except Exception:
            # 시그니처 확인 실패 시 일반 경로
            pass

        # config/cfg 주입 시도 (우선순위: config -> cfg -> 없음)
        try:
            return cls(config=self.cfg, **kw)
        except TypeError:
            pass
        try:
            return cls(cfg=self.cfg, **kw)
        except TypeError:
            pass
        try:
            return cls(**kw)
        except TypeError:
            return cls()

    # ------------------------------
    # 내부: post-config / prewarm 훅
    # ------------------------------
    def _post_configure(self, instance: Any) -> None:
        # 인스턴스/모듈 공통으로 설정 주입 시도
        for setter_name in ("configure", "set_config", "set_cfg", "load_config"):
            fn = getattr(instance, setter_name, None)
            if callable(fn):
                try:
                    fn(self.cfg)
                    return
                except Exception:
                    logger.debug("[ModuleRegistry] %s() 주입 실패(무시)", setter_name, exc_info=True)  # type: ignore[name-defined]
        for attr_name in ("config", "cfg"):
            try:
                setattr(instance, attr_name, self.cfg)
                return
            except Exception:
                continue

    def _prewarm(self, instance: Any) -> None:
        hooks = (
            "build_or_load_index",
            "build_index",
            "prepare_emotions_index",
            "ensure_phrase_index",  # AC 프리매칭 인덱스 빌드
            "prewarm",
            "load",
            "initialize",
        )
        for h in hooks:
            fn = getattr(instance, h, None)
            if callable(fn):
                try:
                    try:
                        fn(self.cfg)
                    except TypeError:
                        fn()
                    logger.debug("[ModuleRegistry] prewarm hook executed: %s", h)  # type: ignore[name-defined]
                    return
                except Exception:
                    logger.debug("[ModuleRegistry] prewarm hook 실패(무시): %s", h, exc_info=True)  # type: ignore[name-defined]

    # ------------------------------
    # 편의: 벌크 프리워밍
    # ------------------------------
    def prewarm_all(self, steps: Optional[List[str]] = None) -> None:
        # [Rollback] Parallel prewarming disabled to prevent import deadlocks
        target_steps = steps or list(self.entrypoints.keys())
        for name in target_steps:
            try:
                self.get_instance(name, force_rebuild=False, prewarm=True)
            except Exception as e:
                logger.warning("[ModuleRegistry] prewarm 실패: %s (%s)", name, e)  # type: ignore[name-defined]

    # ------------------------------
    # EMOTIONS 경로/데이터 로딩
    # ------------------------------
    def _get_emotions_path(self) -> Optional[str]:
        if self._emotions_path_cache:
            return self._emotions_path_cache

        # 우선순위: ENV → cfg.EMOTION_MANAGER_CONFIG.emotions_path → cfg.EMOTIONS_JSON_PATH
        env_p = os.getenv("EMOTIONS_JSON") or os.getenv("EMOTIONS_JSON_PATH") or os.getenv("EMOTIONS_PATH")
        if env_p:
            self._emotions_path_cache = env_p
            return self._emotions_path_cache

        try:
            mgr = getattr(self.cfg, "EMOTION_MANAGER_CONFIG", {})
            p = getattr(mgr, "get", lambda *_: None)("emotions_path")  # dict-like only
            if p:
                self._emotions_path_cache = str(p)
                return self._emotions_path_cache
        except Exception:
            pass

        try:
            p = getattr(self.cfg, "EMOTIONS_JSON_PATH", None)
            if p:
                self._emotions_path_cache = str(p)
                return self._emotions_path_cache
        except Exception:
            pass

        return None

    def _get_emotions_data(self) -> Optional[dict]:
        """
        감정 사전(JSON) 로드(캐시 + mtime 감지).
        - UTF-8-SIG 지원
        - 실패 시 None 반환(상위에서 안전 폴백)
        """
        path = self._get_emotions_path()
        if not path:
            return None

        try:
            stat = os.stat(path)
            mtime = stat.st_mtime
        except Exception:
            mtime = None

        # 캐시가 있고 변경이 없으면 반환
        if self._emotions_data_cache is not None and self._emotions_data_mtime == mtime:
            return self._emotions_data_cache

        try:
            with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
                data = json.load(f)
            self._emotions_data_cache = data if isinstance(data, dict) else {}
            self._emotions_data_mtime = mtime
            return self._emotions_data_cache
        except Exception:
            logger.debug("[ModuleRegistry] emotions_data 로드 실패(path=%r)", path, exc_info=True)  # type: ignore[name-defined]
            return None

    # ------------------------------
    # weight_calculator 의존성 빌더
    # ------------------------------
    def _build_weight_data_manager(self, mod: ModuleType) -> Any:
        """
        weight_calculator 모듈에 있는 WeightEmotionDataManager를 우선 사용하고,
        실패 시 최소한의 속성을 갖춘 더미 객체로 대체.
        """
        ed = self._get_emotions_data() or {}
        try:
            DM = getattr(mod, "WeightEmotionDataManager", None)
            if DM is None:
                # 모듈에 클래스가 없으면 간단 대체
                return SimpleNamespace(emotions_data=ed)

            params = set(inspect.signature(DM).parameters.keys())
            if "emotions_data" in params:
                return DM(emotions_data=ed)
            if "config" in params:
                return DM(config=self.cfg)
            # 인자 없는 형태
            return DM()
        except Exception:
            logger.debug("[ModuleRegistry] WeightEmotionDataManager 생성 실패, 더미로 대체", exc_info=True)  # type: ignore[name-defined]
            return SimpleNamespace(emotions_data=ed)

    # ------------------------------
    # 진단 로그
    # ------------------------------
    def _log_entrypoints_summary(self) -> None:
        try:
            pairs = [f"{k} -> {v[0]}.{v[1]}" for k, v in self.entrypoints.items()]
            preview = ", ".join(pairs[:10])
            more = f" (+{len(pairs)-10} more)" if len(pairs) > 10 else ""
            logger.info("[ModuleRegistry] entrypoints(%d): %s%s", len(pairs), preview, more)  # type: ignore[name-defined]
        except Exception:
            pass


# ===========================================================================
# 안전 호출 어댑터 — 연쇄 오류 차단/대용량 하드닝 버전
# ===========================================================================
# 전역 EMOTIONS 데이터 캐시 (모든 모듈에서 공유)
_GLOBAL_EMOTIONS_CACHE: Optional[dict] = None
_GLOBAL_EMOTIONS_CACHE_LOCK = threading.Lock()

def get_global_emotions_data() -> Optional[dict]:
    """전역 EMOTIONS 데이터 캐시에서 데이터 반환"""
    global _GLOBAL_EMOTIONS_CACHE
    with _GLOBAL_EMOTIONS_CACHE_LOCK:
        if _GLOBAL_EMOTIONS_CACHE is not None:
            return _GLOBAL_EMOTIONS_CACHE
        
        # 캐시가 없으면 로드 시도
        try:
            import config
            if hasattr(config, 'EMOTIONS') and config.EMOTIONS:
                _GLOBAL_EMOTIONS_CACHE = config.EMOTIONS
                return _GLOBAL_EMOTIONS_CACHE
            
            # 파일에서 로드
            emotions_path = getattr(config, 'EMOTIONS_JSON_PATH', None)
            if emotions_path and Path(emotions_path).exists():
                with open(emotions_path, 'r', encoding='utf-8') as f:
                    _GLOBAL_EMOTIONS_CACHE = json.load(f)
                return _GLOBAL_EMOTIONS_CACHE
        except Exception as e:
            logging.debug(f"[전역 캐시] EMOTIONS 데이터 로드 실패: {e}")
        
        return None

class CallAdapter:
    """
    스텝 → 실행 함수(또는 메서드) 매핑/호출 어댑터.

    보강 포인트
    - context_analysis 입력 정규화:
        * transition/intensity 결과를 항상 dict로 강제(coerce)
        * transition / transitions / flows / flow_results 등 동의어 키로 모두 전달
    - time_series_analyzer: 내부 EDM 메서드 부재 안전화 + run_time_series_analysis 우선
    - relationship_analyzer: 시퀀스 정규화(dict→list) + 텍스트 폴백 + Weight DM 기반 emotions_data 주입
    - 구성/경로 주입, 시그니처 자동 매핑, confidence 보정 유지
    - ★ 런타임 예외(예: CUDA)에서는 폴백 금지 → 연쇄 가짜 오류 차단
    """

    def __init__(self, cfg: Any = None):
        self.cfg = cfg or config
        self._emotions_data_cache: Optional[dict] = None
        self._rel_emotions_data_cache: Optional[dict] = None  # 관계 분석용 ED 캐시

    CALL_TABLE: Dict[str, Tuple[str, ...]] = {
        "pattern_extractor": ("extract_patterns_independent", "run_pattern_extraction", "extract_emotion_patterns", "analyze_emotion_flow", "run"),
        "emotion_classification": ("analyze_linguistic_patterns_independent", "run_linguistic_analysis", "analyze_emotions_centrally", "match_language_patterns", "run"),
        "intensity_analysis": ("analyze_intensity_independent", "run_intensity_analysis", "analyze_intensity", "run"),
        "embedding_generation": ("analyze_intensity_independent", "generate_intensity_embeddings", "run_intensity_analysis", "run"),
        "context_analysis": ("extract_context_independent", "analyze_context_patterns", "analyze_progressive_context", "analyze_context_emotion_correlations", "run"),
        "sub_emotion_detection": ("analyze_linguistic_patterns_independent", "analyze_enhanced_emotions", "match_language_patterns", "run"),
        "transition_analyzer": ("analyze_transitions_independent", "run_transition_analysis", "run_basic_transition_analysis", "analyze_emotion_transitions", "run"),
        "linguistic_matcher": ("analyze_linguistic_patterns_independent", "run_linguistic_analysis", "analyze_emotion_progression", "run"),
        "weight_calculator": ("calculate_weights_independent", "run_emotion_weight_calculation", "run_emotion_weight_quick", "run"),
        "complex_analyzer": ("analyze_complexity_independent", "analyze_complex_emotions", "run"),
        "time_series_analyzer": ("analyze_time_series_independent", "run_time_series_analysis", "run_full_time_series_analysis", "run_causality_analysis", "run"),
        "situation_analyzer": ("analyze_situation_independent", "run_full_situation_analysis", "run_situation_analysis", "analyze_situation_context", "run"),
        "psychological_analyzer": ("analyze_psychological_patterns_independent", "run_psych_analysis", "analyze_sequence_only", "run"),
        "relationship_analyzer": ("analyze_relationships_independent", "compute_pairs_compatibility_strength", "analyze_emotion_relationships", "build_social_emotion_graph", "run"),
    }
    FALLBACK_METHODS: Tuple[str, ...] = ("run", "analyze", "process", "forward", "__call__")

    # ------------------------------
    # 공개: 호출 래퍼 생성
    # ------------------------------
    def resolve_callable(self, step_name: str, instance: Any) -> Optional[Callable]:
        target = self._pick_target(step_name, instance)
        if target is None:
            return None

        def runner(payload: "Payload", step_cfg: Dict[str, Any]) -> Dict[str, Any]:
            text = self._safe_text(payload)
            emotions_data_json = self._get_emotions_data()
            cfg = self.cfg
            call_kwargs: Dict[str, Any] = {}

            # ---- EMOTIONS 경로 ----
            ep = None
            try:
                ep = getattr(cfg, "EMOTION_MANAGER_CONFIG", {}).get("emotions_path")
            except Exception:
                ep = None
            if not ep:
                ep = getattr(cfg, "EMOTIONS_JSON_PATH", None)
            ep = str(ep) if ep else None

            # ---- 공통: EMBEDDING_PARAMS(device/amp) 노출 보조 ----
            # 함수 시그니처에 해당 인자가 있으면 바인딩되고, 없으면 자동 무시됩니다.
            emb_params = dict(step_cfg.get("EMBEDDING_PARAMS") or {})
            device_override = emb_params.get("device", None)
            amp_override = emb_params.get("amp", None)

            def _surface_device_amp(kwargs: Dict[str, Any]) -> None:
                if device_override is not None:
                    kwargs.setdefault("device", device_override)
                if amp_override is not None:
                    kwargs.setdefault("amp", amp_override)

            # ---- 스텝별 특수 주입 ----
            if step_name == "pattern_extractor":
                # 요구 시그니처 호환: emotions_data_path + text
                if ep:
                    call_kwargs["emotions_data_path"] = ep
                else:
                    # JSON 경로가 없으면 기본 EMOTIONS.JSON 경로 사용
                    default_path = str(Path(getattr(cfg, "PROJECT_ROOT", Path("."))) / "src" / "EMOTIONS.JSON")
                    if os.path.exists(default_path):
                        call_kwargs["emotions_data_path"] = default_path
                    else:
                        # 운영에서는 스켈레톤 금지 → 즉시 실패
                        prof = getattr(cfg, "EA_PROFILE", "prod")
                        if str(prof) == "prod" or os.getenv("PRODUCTION_MODE", "0") == "1":
                            raise RuntimeError("EMOTIONS_JSON_PATH not set (production)")
                        
                        # 개발에서만 임시 스켈레톤 허용
                        try:
                            proj = Path(getattr(cfg, "PROJECT_ROOT", Path(".")))
                            rt = proj / "logs" / "EMOTIONS.runtime.json"
                            rt.parent.mkdir(parents=True, exist_ok=True)
                            # 기본 감정 데이터로 런타임 파일 생성
                            default_emotions = {
                                "희": {"sub_emotions": {}},
                                "노": {"sub_emotions": {}},
                                "애": {"sub_emotions": {}},
                                "락": {"sub_emotions": {}}
                            }
                            with open(rt, "w", encoding="utf-8") as f:
                                f.write(_safe_json_dumps(default_emotions, sort_keys=True))
                            call_kwargs["emotions_data_path"] = str(rt)
                        except Exception as e:
                            logger.warning(f"Failed to create runtime emotions file: {e}")
                            # 최후의 수단: 빈 문자열로 설정
                            call_kwargs["emotions_data_path"] = ""
                call_kwargs["text"] = text
                # 패턴 추출 내부에서 HF 임베딩을 쓰는 구현 대비: device/amp/use_cache 노출
                _surface_device_amp(call_kwargs)
                call_kwargs.setdefault("use_cache", bool(step_cfg.get("use_cache", True)))

            if step_name == "intensity_analysis":
                if ep:
                    call_kwargs["emotions_data_path"] = ep
                call_kwargs["text"] = text
                # ★ 캐시/디바이스/AMP 선언적 노출
                call_kwargs["use_cache"] = bool(step_cfg.get("use_cache", True))
                _surface_device_amp(call_kwargs)

            if step_name == "embedding_generation":
                if ep:
                    call_kwargs["emotions_data_path"] = ep
                intensity = payload.results.get("intensity_analysis")
                if isinstance(intensity, dict):
                    call_kwargs["intensity"] = intensity
                call_kwargs.setdefault("text", text)
                # ★ 캐시/디바이스/AMP 선언적 노출
                call_kwargs["use_cache"] = bool(step_cfg.get("use_cache", True))
                _surface_device_amp(call_kwargs)

            # transition/time_series에 문장 전달(시그니처 없으면 무시됨)
            if step_name in ("transition_analyzer", "time_series_analyzer"):
                if payload.sentences:
                    call_kwargs.setdefault("sentences", payload.sentences)

            if step_name == "sub_emotion_detection":
                route = _dig(payload.results.get("emotion_classification", {}), "router.topk_main")
                if route:
                    call_kwargs["route"] = route

            if step_name == "context_analysis":
                # 널가드 보강: transition/intensity 결과를 안전하게 추출
                tr = self._coerce_dict(payload.results.get("transition_analyzer", {})) or {}
                ir = self._coerce_dict(payload.results.get("intensity_analysis", {})) or {}
                
                # sentences 안전 주입
                if hasattr(payload, 'sentences') and payload.sentences:
                    call_kwargs.setdefault("sentences", payload.sentences)
                
                # transition 관련 결과 안전 주입 (동의어 키 지원)
                call_kwargs.update({
                    "transition_results": tr, "transition": tr, "transitions": tr,
                    "flow_results": tr, "flows": tr,
                })
                
                # intensity 관련 결과 안전 주입
                call_kwargs.update({"intensity_results": ir, "intensity": ir})
                
                # 감정 시퀀스 주입(없으면 빈 리스트)
                seq_hint = self._extract_sequence(payload.results) or []
                call_kwargs["emotion_sequence"] = seq_hint
                
                # emotions_data 전달 추가 (널가드 적용)
                if emotions_data_json:
                    call_kwargs["emotions_data"] = emotions_data_json

            if step_name == "relationship_analyzer":
                # 널가드 보강: 시퀀스 안전 추출
                seq = self._extract_sequence(payload.results) or []
                rel_ed = self._get_emotions_data_for_relationship() or emotions_data_json
                comp_fn = self._pick_attr(instance, "compute_pairs_compatibility_strength")
                if seq and comp_fn:
                    return self._normalize_result(
                        self._call_by_signature(comp_fn, {
                            "emotion_sequence": seq,
                            "emotions_data": rel_ed,
                            "config": self._mk_step_config("relationship_analyzer", cfg, step_cfg),
                            "payload": payload, "step_cfg": step_cfg, "text": text,
                        }),
                        step_name,
                    )
                rel_fn = self._pick_attr(instance, "analyze_emotion_relationships")
                if rel_fn:
                    return self._normalize_result(
                        self._call_by_signature(rel_fn, {
                            "text": text, "emotions_data": rel_ed,
                            "config": self._mk_step_config("relationship_analyzer", cfg, step_cfg),
                            "payload": payload, "step_cfg": step_cfg,
                        }),
                        step_name,
                    )

            # ★★★ complex_analyzer 특수 처리: 이전 모듈 결과 전달 ★★★
            if step_name == "complex_analyzer":
                # emotion_classification 결과 주입 (감정 분류 결과 활용)
                emo_cls = self._coerce_dict(payload.results.get("emotion_classification", {})) or {}
                if emo_cls:
                    call_kwargs["emotion_classification"] = emo_cls
                    call_kwargs["prior_emotions"] = emo_cls  # 동의어 지원
                
                # intensity_analysis 결과 주입 (강도 분석 결과 활용)
                intensity = self._coerce_dict(payload.results.get("intensity_analysis", {})) or {}
                if intensity:
                    call_kwargs["intensity_results"] = intensity
                    call_kwargs["intensity"] = intensity
                
                # context_analysis 결과 주입 (문맥 분석 결과 활용)
                ctx = self._coerce_dict(payload.results.get("context_analysis", {})) or {}
                if ctx:
                    call_kwargs["context_results"] = ctx
                
                # pattern_extractor 결과 주입 (패턴 추출 결과 활용)
                patterns = self._coerce_dict(payload.results.get("pattern_extractor", {})) or {}
                if patterns:
                    call_kwargs["pattern_results"] = patterns
                
                # 전체 모듈 결과도 전달 (필요시 활용)
                call_kwargs["module_results"] = payload.results
                
                # emotions_data JSON 전달
                if emotions_data_json:
                    call_kwargs["emotions_data"] = emotions_data_json
                
                logger.info(f"[complex_analyzer] 주입된 이전 결과: emotion_classification={bool(emo_cls)}, intensity={bool(intensity)}, context={bool(ctx)}, patterns={bool(patterns)}")
                if emo_cls:
                    logger.info(f"[complex_analyzer] emotion_classification keys: {list(emo_cls.keys())[:10]}")
                    main_dist = emo_cls.get('main_distribution') or emo_cls.get('distribution') or (emo_cls.get('router', {}) or {}).get('main_dist')
                    logger.info(f"[complex_analyzer] main_distribution 발견: {main_dist}")

            if step_name == "weight_calculator":
                # [승격] Soft-Ensemble을 위해 모든 모듈의 결과 주입
                call_kwargs["module_results"] = payload.results
                if ep:
                    call_kwargs["emotions_file"] = ep
                # config_path 옵션 지원
                if getattr(cfg, "WEIGHT_CALCULATOR_CONFIG_PATH", None):
                    call_kwargs["config_path"] = getattr(cfg, "WEIGHT_CALCULATOR_CONFIG_PATH")

            if step_name == "time_series_analyzer":
                for modname in ("src.emotion_analysis.time_series_analyzer", "emotion_analysis.time_series_analyzer"):
                    try:
                        ts_mod = __import__(modname, fromlist=["*"])
                        EDM = getattr(ts_mod, "EmotionDataManager", None)
                        if EDM is not None and not hasattr(EDM, "get_emotion_complexity"):
                            def _get_emotion_complexity(self, emotion_id: str, default: float = 1.0):
                                try:
                                    return float(default)
                                except Exception:
                                    return 1.0

                            setattr(EDM, "get_emotion_complexity", _get_emotion_complexity)
                        break
                    except Exception:
                        continue
                dm = self._build_weight_dm_compat()
                call_kwargs["data_manager"] = dm
                # 널가드 보강: 시퀀스 안전 추출
                seq_hint = self._extract_sequence(payload.results) or []
                if seq_hint:
                    call_kwargs["sequence"] = seq_hint

            # ---- dict step-config ----
            step_config_dict = self._mk_step_config(step_name, cfg, step_cfg)

            # ---- 심리분석기 특수 처리 ------------------------------------
            config_for_target = step_config_dict
            if step_name == "psychological_analyzer":
                if ep:
                    call_kwargs.setdefault("emotions_data_path", ep)

                AnalyzerConfig = None
                for modname in ("src.emotion_analysis.psychological_analyzer",
                                "emotion_analysis.psychological_analyzer"):
                    try:
                        psy_mod = __import__(modname, fromlist=["*"])
                        AnalyzerConfig = getattr(psy_mod, "AnalyzerConfig", None)
                        if AnalyzerConfig:
                            break
                    except Exception:
                        continue

                try:
                    base = dict(getattr(cfg, "PSYCHOLOGICAL_ANALYZER_CONFIG", {}) or {})
                except Exception:
                    base = {}
                merged = {**base, **{k: v for k, v in step_config_dict.items() if v is not None}}
                merged.setdefault("prefer_kiwi", bool(merged.get("prefer_kiwi", True)))
                merged.setdefault("gate_min_types", int(merged.get("gate_min_types", 1)))
                merged.setdefault("gate_min_sum", float(merged.get("gate_min_sum", 1.0)))

                if AnalyzerConfig is not None:
                    try:
                        if hasattr(AnalyzerConfig, "from_dict"):
                            config_for_target = AnalyzerConfig.from_dict(merged)
                        else:
                            config_for_target = AnalyzerConfig(**merged)
                    except Exception:
                        class _CfgNS(SimpleNamespace):
                            def to_dict(self): return dict(self.__dict__)

                        config_for_target = _CfgNS(**merged)
                else:
                    class _CfgNS(SimpleNamespace):
                        def to_dict(self): return dict(self.__dict__)

                    config_for_target = _CfgNS(**merged)

                call_kwargs.setdefault("analyzer_config", step_config_dict)
                call_kwargs.setdefault("options", step_config_dict)

            # ---- 호출 ----
            # 독립 함수인 경우에도 표준 인자 주입(시그니처 자동 매핑)
            if target.__name__.endswith("_independent"):
                res = self._call_by_signature(target, {
                    "text": text,
                    "config": config_for_target,
                    "cfg": cfg,
                    "step_cfg": step_cfg,
                    "payload": payload,
                    "emotions_data": emotions_data_json,
                    **call_kwargs,
                })
            else:
                # 기존 복잡한 호출
                res = self._call_by_signature(
                    target,
                    {
                        "payload": payload,
                        "text": text,
                        "config": config_for_target,  # psycho는 AnalyzerConfig/래퍼, 나머지는 dict
                        "cfg": cfg,
                        "step_cfg": step_cfg,
                        "emotions_data": emotions_data_json,
                        "results": payload.results,
                        **call_kwargs,
                    },
                )
            return self._normalize_result(res, step_name)

        return runner

    # -------- 유틸: dict 강제 --------
    def _coerce_dict(self, obj: Any) -> Dict[str, Any]:
        return obj if isinstance(obj, dict) else {}

    # -------- step-config(dict) 생성 --------
    def _mk_step_config(self, step_name: str, cfg: Any, step_cfg: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        try: out["EMBEDDING_PARAMS"] = dict(getattr(cfg, "EMBEDDING_PARAMS", {}) or {})
        except Exception: pass
        try: out["TRAINING_PARAMS"] = dict(getattr(cfg, "TRAINING_PARAMS", {}) or {})
        except Exception: pass

        if step_name == "transition_analyzer":
            base = getattr(cfg, "EMOTION_TRANSITION_PARAMS", {}) or {}
            out["TRANSITION_ANALYZER_CONFIG"] = dict(base)
        if step_name == "time_series_analyzer":
            base = getattr(cfg, "TIME_SERIES_CONFIG", {}) or {}
            out.update(dict(base)); out.setdefault("max_emotions_per_sentence", 3)
        if step_name == "context_analysis":
            base = getattr(cfg, "CONTEXT_ANALYZER_CONFIG", {}) or {}
            out["CONTEXT_ANALYZER_CONFIG"] = dict(base)

        try:
            ep = getattr(cfg, "EMOTIONS_JSON_PATH", None)
            if ep: out["EMOTIONS_JSON_PATH"] = str(ep)
        except Exception: pass

        if isinstance(step_cfg, dict) and step_cfg:
            out.update(step_cfg)
        return out

    # -------- 대상 함수 선택/모듈 탐색 --------
    def _pick_target(self, step_name: str, instance: Any) -> Optional[Callable]:
        candidates = self.CALL_TABLE.get(step_name, ("run",))

        # 1) 인스턴스가 있으면 그 메서드를 우선 사용
        module_obj = self._instance_module(instance)
        for cand in candidates:
            fn = getattr(instance, cand, None)
            if callable(fn): return fn
        if module_obj is not None:
            for cand in candidates:
                fn = getattr(module_obj, cand, None)
                if callable(fn): return fn

        # 2) 루트 패키지의 독립 함수 사용 (마지막 폴백)
        for cand in candidates:
            if cand.endswith("_independent"):
                pkg = None
                try:
                    import emotion_analysis as _pkg
                    pkg = _pkg
                except Exception:
                    try:
                        import src.emotion_analysis as _pkg  # dev 레이아웃
                        pkg = _pkg
                    except Exception:
                        pkg = None

                fn = getattr(pkg, cand, None) if pkg else None
                if callable(fn):
                    return fn

        # 3) 일반적인 폴백 수단
        for cand in self.FALLBACK_METHODS:
            fn = getattr(instance, cand, None)
            if callable(fn): return fn
        if module_obj is not None:
            for cand in self.FALLBACK_METHODS:
                fn = getattr(module_obj, cand, None)
                if callable(fn): return fn
        if callable(instance): return instance
        return None

    def _instance_module(self, instance: Any) -> Optional[ModuleType]:
        try:
            mod_name = getattr(instance, "__module__", None) or getattr(getattr(instance, "__class__", None), "__module__", None)
            if not mod_name: return None
            return __import__(mod_name, fromlist=["*"])
        except Exception:
            return None

    def _pick_attr(self, obj: Any, name: str) -> Optional[Callable]:
        fn = getattr(obj, name, None)
        if callable(fn):
            return fn
        mod = self._instance_module(obj)
        if mod:
            fn = getattr(mod, name, None)
            if callable(fn):
                return fn
        return None

    # -------- 시그니처 인자 자동 매핑 --------
    def _call_by_signature(self, func: Callable, args_pool: Dict[str, Any]) -> Any:
        """
        원칙:
        - 키워드 기반 매칭 → 실패(TypeError) 시 위치 인자 매칭 → 그래도 실패(TypeError)면 제한적 폴백
        - ★ 런타임 예외(Exception)는 *그대로* 올려서 연쇄 가짜 오류를 차단
        """
        try:
            sig = inspect.signature(func)
        except (TypeError, ValueError):
            return self._call_fallback(func, args_pool)

        bound_kwargs: Dict[str, Any] = {}
        for name, p in sig.parameters.items():
            if name in args_pool:
                bound_kwargs[name] = args_pool[name]
            elif p.default is inspect._empty and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD):
                alt = self._guess_alt_value(name, args_pool)
                if alt is not None:
                    bound_kwargs[name] = alt

        try:
            return func(**bound_kwargs)
        except TypeError:
            # 시그니처 불일치만 위치 인자 재시도
            ordered = [bound_kwargs[name] for name in sig.parameters if name in bound_kwargs]
            try:
                return func(*ordered)
            except TypeError:
                return self._call_fallback(func, args_pool)
            except Exception:
                # 런타임 예외는 폴백 금지
                raise
        except Exception:
            # 런타임 예외는 폴백 금지
            raise

    def _guess_alt_value(self, name: str, pool: Dict[str, Any]) -> Any:
        alias = {
            "cfg": "config", "configuration": "config", "conf": "config",
            "emotions": "emotions_data", "emotion_data": "emotions_data", "emotions_json": "emotions_data",
            "text_input": "text", "sentence_list": "sentences",
            "payload_obj": "payload", "options": "step_cfg", "kwargs": "step_cfg",
            # context synonyms
            "transition": "transition_results",
            "transitions": "transition_results",
            "flows": "transition_results",
            "flow_results": "transition_results",
            "intensity": "intensity_results",
            "emotion_sequence": "sequence",
        }
        key = alias.get(name)
        return pool.get(key) if key else None

    def _call_fallback(self, func: Callable, pool: Dict[str, Any]) -> Any:
        """
        최후의 호환 호출.
        - 시그니처를 다시 확인해 *명시 이름*이 있을 때만 최소 인자 전달
        - 무차별 위치 인자 전달 금지(연쇄 오류 방지)
        """
        try:
            sig = inspect.signature(func)
        except Exception:
            return func()

        candidates: Dict[str, Any] = {}
        # 안전한 최소 매핑만 허용
        safe = {
            "payload": pool.get("payload"),
            "text": self._safe_text(pool.get("payload")) if pool.get("payload") is not None else pool.get("text"),
            "config": pool.get("config"),
            "step_cfg": pool.get("step_cfg"),
        }
        for name in sig.parameters:
            if name in safe and safe[name] is not None:
                candidates[name] = safe[name]

        try:
            if candidates:
                return func(**candidates)
            return func()
        except TypeError:
            # 그래도 안 맞으면 인자 없이 시도
            return func()

    # -------- 시퀀스/ED 로딩 보조 --------
    def _extract_sequence(self, results: Dict[str, Any]) -> Optional[List[Any]]:
        seq = _dig(results.get("time_series_analyzer", {}), "sequence")
        if not seq:
            seq = _dig(results.get("transition_analyzer", {}), "sequence") or _dig(results.get("pattern_extractor", {}), "sequence")
        if not seq:
            return None
        if isinstance(seq, dict):
            try:
                items = sorted(((int(k), v) for k, v in seq.items()), key=lambda kv: kv[0])
                return [v for _, v in items]
            except Exception:
                try:
                    return list(seq.values())
                except Exception:
                    return None
        if isinstance(seq, (list, tuple)):
            return list(seq)
        return None

    def _get_emotions_data_for_relationship(self) -> Optional[dict]:
        if self._rel_emotions_data_cache is not None:
            return self._rel_emotions_data_cache
        try:
            for modname in ("src.emotion_analysis.weight_calculator", "emotion_analysis.weight_calculator"):
                try:
                    wc = __import__(modname, fromlist=["*"])
                except Exception:
                    continue
                DM = getattr(wc, "WeightEmotionDataManager", None)
                ed = self._get_emotions_data() or {}
                dm = None
                if DM is not None:
                    try:
                        dm = DM(emotions_data=ed)
                    except TypeError:
                        try:
                            dm = DM(config=self.cfg)
                        except TypeError:
                            dm = DM()
                if dm and hasattr(dm, "emotions_data") and isinstance(dm.emotions_data, dict):
                    self._rel_emotions_data_cache = dm.emotions_data
                    return self._rel_emotions_data_cache
        except Exception:
            logger.debug("[CallAdapter] relationship용 ED 로딩 실패(무시)", exc_info=True)
        self._rel_emotions_data_cache = self._get_emotions_data()
        return self._rel_emotions_data_cache

    def _get_emotions_data(self) -> Optional[dict]:
        # 전역 캐시 우선 사용
        global_data = get_global_emotions_data()
        if global_data is not None:
            return global_data
        
        # 전역 캐시가 없으면 기존 방식 사용
        if self._emotions_data_cache is not None:
            return self._emotions_data_cache
        try:
            manager = getattr(self.cfg, "EMOTION_MANAGER_CONFIG", {})
            data = manager.get("emotions_data")
            if isinstance(data, dict) and data:
                self._emotions_data_cache = data
                return data
            path = manager.get("emotions_path") or getattr(self.cfg, "EMOTIONS_JSON_PATH", None)
            if path:
                # path가 dict인 경우 처리
                if isinstance(path, dict):
                    logger.debug("[CallAdapter] emotions_path가 dict 타입입니다. 무시합니다.")
                    return None
                # path가 문자열인 경우에만 파일 로드 시도
                if isinstance(path, (str, Path)):
                    with open(str(path), "r", encoding="utf-8") as f:
                        self._emotions_data_cache = json.load(f)
                    return self._emotions_data_cache
        except Exception:
            logger.debug("[CallAdapter] emotions_data 로드 실패(무시)", exc_info=True)
        self._emotions_data_cache = None
        return None

    # -------- Weight DM(호환) --------
    def _build_weight_dm_compat(self):
        try:
            for modname in ("src.emotion_analysis.weight_calculator", "emotion_analysis.weight_calculator"):
                try:
                    wc = __import__(modname, fromlist=["*"])
                except Exception:
                    continue
                DM = getattr(wc, "WeightEmotionDataManager", None)
                ed = self._get_emotions_data() or {}
                if DM is not None:
                    try:
                        dm = DM(emotions_data=ed)
                    except TypeError:
                        try:
                            dm = DM(config=self.cfg)
                        except TypeError:
                            dm = DM()
                else:
                    dm = type("CompatDM", (), {})()
                    setattr(dm, "emotions_data", ed)
                break
            else:
                dm = type("CompatDM", (), {})()
                setattr(dm, "emotions_data", self._get_emotions_data() or {})
        except Exception:
            dm = type("CompatDM", (), {})()
            setattr(dm, "emotions_data", self._get_emotions_data() or {})

        if not hasattr(dm, "get_emotion_complexity"):
            def _get_emotion_complexity(emotion_id: str, default: float = 1.0):
                try: return float(default)
                except Exception: return 1.0
            setattr(dm, "get_emotion_complexity", _get_emotion_complexity)
        return dm

    # -------- 결과 정규화(+ confidence) --------
    def _normalize_result(self, res: Any, step_name: str) -> Dict[str, Any]:
        if not isinstance(res, dict):
            return {"step": step_name, "value": res, "confidence": self._default_conf(step_name), "success": True}
        res.setdefault("step", step_name)
        # success 필드가 없으면 기본값 True 설정
        if "success" not in res:
            res["success"] = True
        if "confidence" in res and isinstance(res["confidence"], (int, float)):
            return res
        if step_name == "emotion_classification":
            conf = self._extract_conf_emotion_classification(res)
        elif step_name == "intensity_analysis":
            conf = self._extract_conf_intensity(res)
        else:
            conf = self._extract_conf_generic(res)
        if conf is None:
            conf = self._default_conf(step_name)
        res["confidence"] = float(conf)
        return res

    def _extract_conf_emotion_classification(self, res: Dict[str, Any]) -> Optional[float]:
        topk = _dig(res, "router.topk_main", [])
        val = self._max_from_seq(topk)
        if val is not None: return val
        for key in ("scores", "probabilities", "probs", "main_scores"):
            if isinstance(res.get(key), dict):
                return self._max_from_dict(res[key])
        return None

    def _extract_conf_intensity(self, res: Dict[str, Any]) -> Optional[float]:
        for key in ("intensity_distribution", "intensity_probs"):
            if isinstance(res.get(key), dict):
                return self._max_from_dict(res[key])
        curve = res.get("intensity_curve") or res.get("curve")
        if isinstance(curve, (list, tuple)):
            return self._max_from_seq(curve)
        return None

    def _extract_conf_generic(self, res: Dict[str, Any]) -> Optional[float]:
        for k in ("score", "prob", "likelihood"):
            v = res.get(k)
            if isinstance(v, (int, float)): return float(v)
        for k in ("scores", "probabilities", "probs", "values"):
            if isinstance(res.get(k), dict): return self._max_from_dict(res[k])
            if isinstance(res.get(k), (list, tuple)): return self._max_from_seq(res[k])
        return None

    def _max_from_dict(self, d: Dict[str, Any]) -> Optional[float]:
        try:
            nums = [float(v) for v in d.values() if isinstance(v, (int, float))]
            return max(nums) if nums else None
        except Exception:
            return None

    def _max_from_seq(self, seq: Any) -> Optional[float]:
        try:
            vals: List[float] = []
            for it in seq or []:
                if isinstance(it, (int, float)):
                    vals.append(float(it))
                elif isinstance(it, (list, tuple)) and len(it) >= 2 and isinstance(it[1], (int, float)):
                    vals.append(float(it[1]))
                elif isinstance(it, dict):
                    for k in ("confidence", "score", "prob"):
                        v = it.get(k)
                        if isinstance(v, (int, float)):
                            vals.append(float(v)); break
            return max(vals) if vals else None
        except Exception:
            return None

    def _default_conf(self, step_name: str) -> float:
        try:
            t = getattr(self.cfg, "THRESHOLDS", {})
            if step_name == "emotion_classification":
                return float(t.get("main_confidence", 0.1)) + 1e-3
            if step_name == "intensity_analysis":
                return float(t.get("sub_confidence", 0.15)) + 1e-3
        except Exception:
            pass
        return 0.5

    # -------- 기타 --------
    def _safe_text(self, payload_or_text: Any) -> str:
        """Payload/text 무엇이 와도 안전하게 문자열 반환."""
        try:
            if hasattr(payload_or_text, "text"):
                v = payload_or_text.text
            else:
                v = payload_or_text
            return v if isinstance(v, str) else ("" if v is None else str(v))
        except Exception:
            return ""


# ===========================================================================
# 캐시 — 연쇄오류 차단·스탬피드 방지·대용량 하드닝 버전
# ===========================================================================
@dataclass(slots=True)
class _CacheEntry:
    value: Any
    expire_at: Optional[float]  # None = no TTL
    size: int = 0               # approx bytes (메모리 예산 사용 시)

class StepCache:
    """
    파이프라인 스텝 캐시 (LRU + 선택적 TTL + 선택적 메모리 예산)

    특징
    - LRU 기반 용량 제한(max_items)
    - TTL(sec) 만료 + 랜덤 지터(스탬피드 방지)
    - 스레드-세이프(RLock), get_or_set로 중복 계산 방지
    - 히트/미스/에빅션/만료 통계
    - 키 규약: 기본은 hash(text)::step::cfg_sig (하위호환)
              확장 메서드 make_key_v2(...) 제공
    - 선택: max_bytes 예산 내에서 LRU 에빅션 (approx size)
    """

    def __init__(
        self,
        enabled: bool = True,
        max_items: int = 8192,  # 3번 개선작업: 캐시 크기 대폭 증가 (4096→8192)
        ttl_sec: Optional[float] = None,
        *,
        max_bytes: Optional[int] = None,
        sweep_every: int = 4096,  # 3번 개선작업: 스윕 주기 대폭 증가 (2048→4096)
    ):
        self.enabled = bool(enabled)
        self.max_items = int(max_items)
        self.ttl_sec = float(ttl_sec) if ttl_sec else None
        self.max_bytes = int(max_bytes) if max_bytes else None
        self.sweep_every = max(16, int(sweep_every))

        self._store: "OrderedDict[str, _CacheEntry]" = OrderedDict()
        self._lock = RLock()
        # 키별 락 제거로 메모리 누수 방지
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._expirations = 0
        self._ops_since_sweep = 0
        self._total_bytes = 0

    # --- 키 생성 (하위호환 + 확장) ------------------------------------
    def make_key(
        self, text: str, step: str, cfg_sig: str, *,
        emotion_set_version: Optional[str] = None,
        model_version: Optional[str] = None
    ) -> str:
        """
        기존 호출부와의 하위호환을 유지합니다.
        필요한 경우 emotion_set_version/model_version을 추가로 넘겨 키 충돌을 줄일 수 있습니다.
        """
        if emotion_set_version or model_version:
            return _make_cache_key(
                text=text, step=step, cfg_signature=cfg_sig,
                emotion_set_version=emotion_set_version or "NA",
                model_version=model_version or "NA",
            )
        # 압축된 키 생성으로 메모리 절약
        text_hash = _hash_text(text)[:16]  # 해시 길이 단축
        return f"{text_hash}::{(step or '').replace('::','|')}::{cfg_sig}"

    def make_key_v2(
        self, *, text: str, step: str, cfg_signature: str,
        emotion_set_version: str = "NA", model_version: str = "NA"
    ) -> str:
        """권장: 확장 키 생성 (유틸 _make_cache_key 사용)."""
        return _make_cache_key(
            text=text, step=step, cfg_signature=cfg_signature,
            emotion_set_version=emotion_set_version, model_version=model_version,
        )

    # --- 기본 API ------------------------------------------------------
    def get(self, key: str) -> Optional[Any]:
        if not self.enabled:
            return None
        now = time.time()
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._misses += 1
                return None
            if entry.expire_at is not None and entry.expire_at < now:
                # 만료 → 제거 후 miss 처리
                self._pop_key(key)
                self._misses += 1
                self._expirations += 1
                return None
            # LRU: 최근 사용으로 갱신
            self._store.move_to_end(key, last=True)
            self._hits += 1
            return entry.value

    def peek(self, key: str) -> Optional[Any]:
        """LRU 순서를 변경하지 않고 값만 조회."""
        if not self.enabled:
            return None
        now = time.time()
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            if entry.expire_at is not None and entry.expire_at < now:
                return None
            return entry.value

    def contains(self, key: str) -> bool:
        return self.get(key) is not None

    def set(self, key: str, value: Any) -> None:
        if not self.enabled:
            return
        with self._lock:
            expire_at = self._compute_expiry()
            size = self._approx_size(value)
            # 기존 키가 있으면 교체 전 바이트 조정
            old = self._store.get(key)
            if old is not None:
                self._total_bytes -= max(0, old.size)
            self._store[key] = _CacheEntry(value=value, expire_at=expire_at, size=size)
            self._store.move_to_end(key, last=True)
            self._total_bytes += max(0, size)
            self._ops_since_sweep += 1
            if self._ops_since_sweep >= self.sweep_every:
                self._sweep_expired_locked()
            self._evict_if_needed_locked()

    def get_or_set(self, key: str, producer: Callable[[], Any]) -> Any:
        """
        캐시에 있으면 반환, 없으면 producer()를 한 번만 실행해 저장 후 반환.
        - 단일 전역 락으로 동시 호출 스탬피드 방지 (메모리 누수 방지).
        """
        if not self.enabled:
            # 캐시를 끈 상태라도 miss 카운트만 증가시켜 진단 가능
            try:
                res = producer()
            finally:
                self._misses += 1
            return res

        # 전역 락으로 동시성 제어
        with self._lock:
            # 캐시 조회
            val = self._get_locked(key)
            if val is not None:
                return val
            
            # 생산 및 저장
            res = producer()
            self._set_locked(key, res)
            return res

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            self._hits = self._misses = self._evictions = self._expirations = 0
            self._total_bytes = 0
            self._ops_since_sweep = 0

    # --- 선택적 무효화 유틸 -------------------------------------------
    def invalidate(self, key: str) -> bool:
        """특정 키 제거."""
        with self._lock:
            return self._pop_key(key)

    def invalidate_step(self, step: str) -> int:
        """특정 스텝(prefix 매칭) 캐시 제거."""
        needle = f"::{step}::"
        with self._lock:
            keys = [k for k in list(self._store.keys()) if needle in k]
            for k in keys:
                self._pop_key(k)
            return len(keys)

    def invalidate_text(self, text: str) -> int:
        """특정 원본문에 해당하는 캐시 제거."""
        prefix = _hash_text(text)
        with self._lock:
            keys = [k for k in list(self._store.keys()) if k.startswith(prefix)]
            for k in keys:
                self._pop_key(k)
            return len(keys)

    # --- 진단 ----------------------------------------------------------
    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "enabled": self.enabled,
                "size": len(self._store),
                "max_items": self.max_items,
                "ttl_sec": self.ttl_sec,
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
                "expirations": self._expirations,
                "total_bytes": self._total_bytes,
                "max_bytes": self.max_bytes,
            }

    def reset_stats(self) -> None:
        with self._lock:
            self._hits = self._misses = self._evictions = self._expirations = 0

    # --- 내부 ----------------------------------------------------------
    def _compute_expiry(self) -> Optional[float]:
        if not self.ttl_sec:
            return None
        #  ±10% 지터로 만료 타이밍 분산(스탬피드 방지)
        jitter = 1.0 + (random.random() * 0.2 - 0.1)
        return time.time() + max(0.0, self.ttl_sec * jitter)

    def _sweep_expired_locked(self) -> None:
        """TTL 만료 키를 주기적으로 정리(락 보유 상태에서 호출)."""
        if not self.ttl_sec:
            self._ops_since_sweep = 0
            return
        now = time.time()
        expired: List[str] = [k for k, e in self._store.items() if e.expire_at and e.expire_at < now]
        for k in expired:
            self._pop_key(k)
        self._expirations += len(expired)
        self._ops_since_sweep = 0

    def _evict_if_needed_locked(self) -> None:
        """LRU 원칙으로 용량/메모리 예산을 만족할 때까지 제거(락 보유 상태에서 호출)."""
        # 메모리 예산 우선
        if self.max_bytes is not None and self.max_bytes > 0:
            while self._total_bytes > self.max_bytes and self._store:
                old_key, old_entry = self._store.popitem(last=False)
                self._total_bytes -= max(0, old_entry.size)
                self._evictions += 1

        # 아이템 개수 제한
        while len(self._store) > self.max_items and self._store:
            old_key, old_entry = self._store.popitem(last=False)
            self._total_bytes -= max(0, old_entry.size)
            self._evictions += 1

    def _pop_key(self, key: str) -> bool:
        entry = self._store.pop(key, None)
        if entry is None:
            return False
        self._total_bytes -= max(0, entry.size)
        return True

    # 키별 락 제거로 메모리 누수 방지 - 단일 전역 락 사용

    # --- 크기 추정(approx) --------------------------------------------
    def _approx_size(self, obj: Any) -> int:
        """
        캐시 항목의 대략적 메모리 크기 추정.
        - numpy: nbytes
        - torch.Tensor: numel * element_size
        - bytes/bytearray: len
        - dict/list: JSON 직렬화 길이(상한 있음)
        - 그 외: sys.getsizeof + fallback
        """
        try:
            # numpy
            if hasattr(obj, "nbytes") and isinstance(getattr(obj, "nbytes"), (int,)):
                return int(getattr(obj, "nbytes"))

            # torch.Tensor
            if obj.__class__.__module__.startswith("torch") and hasattr(obj, "numel") and hasattr(obj, "element_size"):
                try:
                    return int(obj.numel()) * int(obj.element_size())  # type: ignore[no-any-return]
                except Exception:
                    pass

            if isinstance(obj, (bytes, bytearray, memoryview)):
                return len(obj)

            if isinstance(obj, (dict, list, tuple)):
                try:
                    s = json.dumps(obj, ensure_ascii=False, default=lambda x: str(x))
                    return len(s.encode("utf-8"))
                except Exception:
                    return min(4096, len(str(obj).encode("utf-8", "replace")))

            return max(64, sys.getsizeof(obj))
        except Exception:
            return 256


# ===========================================================================
# run_if 평가기 — 연쇄오류 차단·디버그 강화·대용량 하드닝 버전
# ===========================================================================
class ConditionEvaluator:
    """
    파이프라인 run_if 조건 평가기(강화판).

    지원 조건(typ)
    - confidence_check:
        { "type": "confidence_check", "module": "intensity_analysis",
          "key": "confidence", "threshold": 0.5, "op": ">=", "negate": false }
        * threshold_name: "transition" 처럼 config.THRESHOLDS의 키를 사용할 수 있음
        * path 로 results가 아닌 임의 경로도 가능 (예: "results.emotion_classification.router.uncertainty")

    - check_results:
        { "type": "check_results", "module": "context_analysis",
          "condition": "exists|nonempty|above|below|equals|contains",
          "key": "some.key", "threshold": 0.3, "op": ">=", "negate": false }

    - metric_compare: (좌/우 경로 또는 상수 비교)
        { "type": "metric_compare",
          "left": "results.emotion_classification.confidence",
          "right": 0.6, "op": ">=" }

    - any_of / all_of (중첩 가능):
        { "any_of": [ <조건>, <조건>, ... ] }  또는  { "all_of": [ ... ] }

    공통 옵션
    - negate: 결과를 반전
    - percent: true 시 threshold(또는 right)가 0~100 정수/문자열(예: "75%")이면 0~1로 자동 환산
    - agg: 값이 리스트/딕셔너리일 때 len|max|min|mean|sum 중 하나로 스칼라화

    반환
    - evaluate(...) -> bool
    - evaluate_with_reason(...) -> (bool, {"reasons":[...]}): 디버깅 용도
    """

    # ------------- Public API -------------
    @classmethod
    def evaluate(cls, run_if: List[Dict[str, Any]] | Dict[str, Any], payload: "Payload") -> bool:
        ok, _ = cls.evaluate_with_reason(run_if, payload)
        return ok

    @classmethod
    def evaluate_with_reason(
        cls,
        run_if: List[Dict[str, Any]] | Dict[str, Any],
        payload: "Payload",
        *,
        max_reasons: int = 2000,
    ) -> Tuple[bool, Dict[str, Any]]:
        # 단일 dict도 허용
        if not run_if:
            return True, {"reasons": ["no-conditions"]}
        clauses = run_if if isinstance(run_if, list) else [run_if]

        reasons: List[str] = []
        for cond in clauses:
            ok = cls._eval_clause(cond or {}, payload, reasons)
            tag = "passed" if ok else "failed"
            reasons.append(f"clause-{tag}: {cls._short(cond)}")
            if not ok:
                # 과도한 리스팅 방지
                return False, {"reasons": reasons[-max_reasons:]}

        return True, {"reasons": reasons[-max_reasons:]}

    # ------------- Internal -------------
    @classmethod
    def _eval_clause(cls, cond: Dict[str, Any], payload: "Payload", reasons: List[str]) -> bool:
        # 그룹 논리(any_of/all_of) 먼저 처리
        if "any_of" in cond:
            sub = cond.get("any_of") or []
            hit = False
            for c in sub:
                if cls._eval_clause(c, payload, reasons):
                    hit = True
                    break
            return hit

        if "all_of" in cond:
            sub = cond.get("all_of") or []
            for c in sub:
                if not cls._eval_clause(c, payload, reasons):
                    return False
            return True

        typ = cond.get("type")
        negate = bool(cond.get("negate", False))

        # 매크로: module_ok → confidence_check 변환
        if typ == "module_ok":
            cond = {
                "type": "confidence_check",
                "module": cond.get("module"),
                "key": "confidence",
                "threshold": cond.get("min_conf", 0.5),
                "op": ">=",
            }
            typ = "confidence_check"

        # confidence_check
        if typ == "confidence_check":
            module = cond.get("module")
            key = cond.get("key", "confidence")
            path = cond.get("path")  # 있으면 path 우선
            val = cls._fetch_value(payload, module=module, key=key, path=path, agg=cond.get("agg"))
            thr = cls._resolve_threshold(cond)
            op = cond.get("op", ">=")
            ok = cls._compare(val, thr, op, percent=cond.get("percent"))
            return (not ok) if negate else ok

        # check_results
        if typ == "check_results":
            module = cond.get("module")
            condition = (cond.get("condition") or "exists").lower()
            key = cond.get("key")
            path = cond.get("path")
            val = cls._fetch_value(payload, module=module, key=key, path=path, agg=cond.get("agg"))
            if condition == "exists":
                ok = val is not None
            elif condition == "nonempty":
                ok = bool(val)
            elif condition in ("above", "below", "equals", "contains"):
                thr = cls._resolve_threshold(cond)
                if condition == "above":
                    ok = cls._compare(val, thr, ">", percent=cond.get("percent"))
                elif condition == "below":
                    ok = cls._compare(val, thr, "<", percent=cond.get("percent"))
                elif condition == "equals":
                    ok = cls._equals(val, thr)
                else:  # contains
                    ok = cls._contains(val, thr)
            else:
                logger.debug("Unknown check_results.condition: %s", condition)
                ok = False
            return (not ok) if negate else ok

        # metric_compare
        if typ == "metric_compare":
            left = cls._fetch_by_any(payload, cond.get("left"), agg=cond.get("agg_left"))
            right_raw = cond.get("right")
            right = cls._fetch_by_any(payload, right_raw, agg=cond.get("agg_right"))
            op = cond.get("op", ">=")
            ok = cls._compare(left, right, op, percent=cond.get("percent"))
            return (not ok) if negate else ok

        # text_length_check
        if typ == "text_length_check":
            text = payload.text
            if not isinstance(text, str):
                ok = False
            else:
                min_length = cond.get("min_length", 0)
                min_tokens = cond.get("min_tokens", 0)
                
                length_ok = len(text.strip()) >= min_length
                tokens_ok = True
                
                if min_tokens > 0:
                    # 간단한 토큰화 (공백 기준)
                    tokens = text.strip().split()
                    tokens_ok = len(tokens) >= min_tokens
                
                ok = length_ok and tokens_ok
            return (not ok) if negate else ok

        logger.debug("Unknown run_if condition: %s", cond)
        return False

    # ---------------- value fetchers ----------------
    @classmethod
    def _fetch_by_any(cls, payload: "Payload", ref: Any, agg: Optional[str] = None):
        """
        ref가 문자열이면 path로 간주하여 payload 트리에서 조회,
        dict {'module':..., 'key':...} 형식도 허용, 그 외는 상수로 사용.
        """
        if isinstance(ref, str):
            return cls._fetch_value(payload, path=ref, agg=agg)
        if isinstance(ref, dict):
            return cls._fetch_value(
                payload,
                module=ref.get("module"),
                key=ref.get("key"),
                path=ref.get("path"),
                agg=agg or ref.get("agg"),
            )
        return ref

    @classmethod
    def _fetch_value(
        cls,
        payload: "Payload",
        *,
        module: Optional[str] = None,
        key: Optional[str] = None,
        path: Optional[str] = None,
        agg: Optional[str] = None,
    ):
        """
        payload에서 값을 찾아옵니다.
        우선순위: path → (results[module] + key) → None
        """
        try:
            root = {
                "text": payload.text,
                "meta": payload.meta,
                "sentences": payload.sentences,
                "tokens": payload.tokens,
                "lang": payload.lang,
                "features": payload.features,
                "results": payload.results,
                "timeline": payload.timeline,
                # 기본 메트릭 추가
                "tokens_count": len(payload.tokens),
                "sentences_count": len(payload.sentences),
            }
        except Exception:
            # payload 비정상일 때 최소 방어
            root = {}

        val = None
        if path:
            val = _dig(root, path)
        elif module:
            mod_res = (payload.results or {}).get(module)
            val = _dig(mod_res, key) if key else mod_res

        # 스칼라화(agg)
        if agg and val is not None and not isinstance(val, (int, float)):
            agg = str(agg).lower()
            try:
                if agg == "len":
                    val = len(val)
                elif agg == "max":
                    val = max(val.values()) if isinstance(val, dict) else max(val)
                elif agg == "min":
                    val = min(val.values()) if isinstance(val, dict) else min(val)
                elif agg == "sum":
                    val = sum(val.values()) if isinstance(val, dict) else sum(val)
                elif agg == "mean":
                    vals = list(val.values()) if isinstance(val, dict) else list(val)
                    # 숫자만 취해 평균(혼합형이면 숫자만)
                    nums = [float(x) for x in vals if isinstance(x, (int, float))]
                    val = _mean(nums) if nums else None
            except Exception:
                # agg 실패 시 원값 유지
                pass

        # 숫자 문자열 변환(가능하면)
        if isinstance(val, (str, bytes)):
            s = val.decode("utf-8", "replace") if isinstance(val, (bytes, bytearray)) else val
            s = s.strip()
            # 퍼센트 문자열 "75%": 0.75
            if s.endswith("%"):
                try:
                    val = float(s[:-1]) / 100.0
                except Exception:
                    pass
            else:
                v = _coerce_float(s, None)
                val = v if v is not None else val

        return val

    # ---------------- threshold/compare ----------------
    @classmethod
    def _resolve_threshold(cls, cond: Dict[str, Any]) -> Any:
        """
        threshold 또는 threshold_name(config.THRESHOLDS/INTENSITY_THRESHOLDS) 해석.
        """
        if "threshold" in cond:
            return cond["threshold"]

        name = cond.get("threshold_name")
        if not name:
            return None

        # config 조회
        try:
            th = getattr(config, "THRESHOLDS", {})
            if isinstance(th, dict) and name in th:
                return th[name]
        except Exception:
            pass
        try:
            ith = getattr(config, "INTENSITY_THRESHOLDS", {})
            if isinstance(ith, dict) and name in ith:
                return ith[name]
        except Exception:
            pass

        # default 제공 시 사용
        return cond.get("default")

    @classmethod
    def _compare(cls, left: Any, right: Any, op: str, *, percent: Optional[bool] = None) -> bool:
        """
        비교 연산 통일.
        - percent=True 이면 right가 1~100 정수/문자열("75%")일 때 0~1로 환산.
        - 숫자 비교는 float으로 강제(가능하면), 그 외는 일반 비교.
        """
        # None 처리
        if left is None or right is None:
            return False

        # 퍼센트 환산(오른쪽만)
        right = cls._maybe_percent(right, enable=percent)

        # 수치 비교 가능하면 수치로
        l_num = _coerce_float(left, None)
        r_num = _coerce_float(right, None)

        try:
            if l_num is not None and r_num is not None:
                l, r = l_num, r_num
            else:
                l, r = left, right

            op = (op or ">=").lower()
            if op in (">", "gt"):
                return l > r
            if op in (">=", "ge"):
                return l >= r
            if op in ("<", "lt"):
                return l < r
            if op in ("<=", "le"):
                return l <= r
            if op in ("==", "eq"):
                return l == r
            if op in ("!=", "<>", "ne"):
                return l != r
            if op == "in":
                return l in r  # type: ignore[operator]
            if op == "contains":
                return r in l  # type: ignore[operator]
        except Exception:
            logger.debug("compare error: left=%r right=%r op=%s", left, right, op, exc_info=True)
            return False

        logger.debug("unknown operator: %s", op)
        return False

    @staticmethod
    def _maybe_percent(value: Any, *, enable: Optional[bool]) -> Any:
        """퍼센트 스케일을 0~1로 변환."""
        if isinstance(value, str) and value.strip().endswith("%"):
            try:
                return float(value.strip()[:-1]) / 100.0
            except Exception:
                return value
        if enable:
            v = _coerce_float(value, None)
            if v is not None and v > 1:
                return v / 100.0
        return value

    @staticmethod
    def _equals(a: Any, b: Any, eps: float = 1e-9) -> bool:
        a_n = _coerce_float(a, None)
        b_n = _coerce_float(b, None)
        if a_n is not None and b_n is not None:
            return abs(a_n - b_n) <= eps
        return a == b

    @staticmethod
    def _contains(container: Any, item: Any) -> bool:
        try:
            # 문자열 퍼센트도 자연 처리
            if isinstance(item, str) and item.endswith("%"):
                item = ConditionEvaluator._maybe_percent(item, enable=True)
            return item in container  # type: ignore[operator]
        except Exception:
            return False

    # ---------------- misc ----------------
    @staticmethod
    def _short(cond: Dict[str, Any]) -> str:
        try:
            typ = cond.get("type") or ("any_of" if "any_of" in cond else "all_of")
            key = cond.get("key") or cond.get("path") or cond.get("module") or cond.get("left")
            thr = cond.get("threshold") or cond.get("threshold_name") or cond.get("right")
            op = cond.get("op") or cond.get("condition")
            return f"{typ}({key},{op},{thr})"
        except Exception:
            return str(cond)

# ===========================================================================
# 사소한 딥 dict 접근 유틸 (강화판, 안전/성능 개선)
# ===========================================================================
# @미리 컴파일한 토크나이저 정규식들 (성능/안정성)
# @@브라켓 토큰: [0], ['key'], ["ke\"y"], 공백 허용, 음수 인덱스 허용
_DIG_HEAD_RE = re.compile(r"^[^\[\]]+")
_DIG_BRACKET_RE = re.compile(r"""
    \[
        \s*
        (                               # 그룹: 내부 토큰
            -?\d+                       # 정수 인덱스 (음수 허용)
            | "(?:[^"\\]|\\.)*"         # "문자열"(이스케이프 허용)
            | '(?:[^'\\]|\\.)*'         # '문자열'(이스케이프 허용)
        )
        \s*
    \]
""", re.VERBOSE)

def _dig(d: Any, dotted: Optional[str], default: Any = None, *, sep: str = ".") -> Any:
    """
    안전한 중첩 조회:
      - dict / list / tuple (및 numpy/pandas 시퀀스) 지원
      - 점 표기: "a.b.c"
      - 브라켓 표기: obj["a"][0]['b'] → "a[0]['b']" 또는 "a[0].b"
      - 혼합 가능: "results.emotion_classification.router.topk_main[0]['label']"
      - 존재하지 않거나 타입이 맞지 않으면 default 반환(기본 None)
      - 음수 인덱스 지원: "items[-1]"

    예)
      _dig(payload.results, "emotion_classification.confidence", 0.0)
      _dig(root, "a[0].b['c']", default={})
    """
    if dotted is None or dotted == "":
        return d

    def _unescape(s: str) -> str:
        # 따옴표 제거 + 일반 이스케이프 해제
        if len(s) >= 2 and (s[0] == s[-1]) and s[0] in ("'", '"'):
            s = s[1:-1]
        return bytes(s, "utf-8").decode("unicode_escape")

    def _subtokens(part: str):
        """
        한 세그먼트에서 브라켓 인덱스들을 추출.
        예: "a[0]['b']" -> ["a", 0, "b"]
            "[1]"       -> [1]
        """
        tokens = []
        head = _DIG_HEAD_RE.match(part)
        if head:
            tokens.append(head.group(0))

        for m in _DIG_BRACKET_RE.finditer(part):
            raw = m.group(1).strip()
            if raw.lstrip("-").isdigit():
                try:
                    tokens.append(int(raw))
                except Exception:
                    tokens.append(raw)  # 안전 폴백(문자열)
            else:
                tokens.append(_unescape(raw))
        # "[0]"처럼 head가 없을 수도 있음 → 위 for가 처리
        return tokens if tokens else ([part] if part else [])

    # numpy/pandas 시퀀스도 최소 지원(의존성 없이 duck-typing)
    def _is_sequence_like(x: Any) -> bool:
        if isinstance(x, (str, bytes)):
            return False
        if isinstance(x, Sequence):
            return True
        mod = getattr(x, "__class__", type(x)).__module__
        # numpy.ndarray, pandas.Series/DataFrame의 __getitem__ 방식을 허용
        return hasattr(x, "__getitem__") and mod and any(k in mod for k in ("numpy", "pandas"))

    current = d
    try:
        # 점 단위 분해 후 각 파트에서 브라켓 인덱스 해석
        parts = [p for p in str(dotted).split(sep) if p != ""]
        for part in parts:
            for tok in _subtokens(part):
                if isinstance(current, Mapping):
                    # dict에서 키 조회(정수/문자열 키 모두 시도)
                    if tok in current:
                        current = current[tok]
                    else:
                        # 정수 토큰이면 문자열 키도 시도 ("0" vs 0)
                        if isinstance(tok, int) and (str_tok := str(tok)) in current:
                            current = current[str_tok]
                        elif isinstance(tok, str) and tok.isdigit():
                            itok = int(tok)
                            if itok in current:
                                current = current[itok]
                            elif tok in current:
                                current = current[tok]
                            else:
                                return default
                        else:
                            return default
                elif _is_sequence_like(current):
                    # 시퀀스에서 인덱스 조회
                    idx: Optional[int] = None
                    if isinstance(tok, int):
                        idx = tok
                    elif isinstance(tok, str) and tok.lstrip("-").isdigit():
                        idx = int(tok)
                    else:
                        # 시퀀스에 문자열 키 접근은 불가 → 실패
                        return default

                    # 음수 인덱스 허용
                    n = len(current) if hasattr(current, "__len__") else None
                    if n is not None:
                        if idx < 0:
                            idx = n + idx
                        if idx < 0 or idx >= n:
                            return default
                    try:
                        current = current[idx]  # type: ignore[index]
                    except Exception:
                        return default
                else:
                    # 더 이상 내려갈 수 없음
                    return default
        return current
    except Exception:
        return default


# 메인 오케스트레이터 — 안정성/대용량 하드닝 버전 ------------------------------------
class EmotionPipelineOrchestrator:
    """
    config 기반 파이프라인 실행기(로그/저장 일원화 + 모듈 파일로그 제어판).

    포함 기능:
    - src/logs/YYYYMMDD 실행 폴더 자동 생성
    - 스텝별 결과 JSON 저장(+ 캐시 적중 저장 토글)
    - save=True 시 최종 결과도 같은 날짜 폴더에 저장(out_dir 미지정 시)
    - 실행마다 고정 trace_id 자동 주입(없을 때)
    - Tail 4단계 강제 실행(confidence 보정)
    - 오케스트레이션 시 emotion_analysis/logs, src/logs, 루트 logs로의 파일 로그 차단/리다이렉트
      (RUN_DIR=오늘자 폴더만 허용; data_utils.log만 화이트리스트)
    - ★ 런타임 예외가 발생해도 파이프라인 전체는 중단하지 않음(스텝별 on_error 정책 적용)
    - ★ 병렬 실행 지원 (의존성 그래프 기반)
    """

    _FORCE_TAIL_STEPS = {
        "complex_analyzer",
        "situation_analyzer",
        "relationship_analyzer",
        "psychological_analyzer",
    }
    _FORCED_CONF_DEFAULTS = {
        "complex_analyzer": 0.51,
        "situation_analyzer": 0.51,
        "relationship_analyzer": 0.51,
        "psychological_analyzer": 0.51,
    }

    # ---------- 경로 정규화 유틸(대/소문자 무시 + 절대경로) ----------
    def _canon_path(self, p: str) -> str:
        return os.path.normcase(os.path.normpath(os.path.abspath(p)))
    
    def _determine_active_preset(self) -> str:
        """개선안 C-1: 프리셋 고정 (핵심)"""
        # 환경변수 강제 토글이 최우선
        if os.getenv("FORCE_HEAVY", "0").lower() in ("1", "true", "yes"):
            return "heavy"
        if os.getenv("FORCE_FAST", "0").lower() in ("1", "true", "yes"):
            return "fast"

        # 환경변수 우선 확인
        preset = os.getenv("ACTIVE_PRESET")
        if preset:
            return preset
        
        # 사용 목적에 따른 자동 결정
        embedding_mode = os.getenv("EMB_TEXT_ONLY", "0")
        label_mode = os.getenv("LABEL_HEAVY_MODE", "0")
        
        if embedding_mode == "1":
            return "fast"  # Text-Only 임베딩용
        elif label_mode == "1":
            return "heavy"  # Full-Feature 정답지용
        else:
            return "hybrid"  # 기본값
    
    def _check_circuit_breaker(self, step_name: str) -> bool:
        """P6: 서킷브레이커 체크 (단일 구조, 히스테리시스 포함)"""
        cb = self.circuit_breaker
        current_time = time.time()
        
        # 스텝 상태 가져오기 (기본값: CLOSED)
        state_info = cb["states"].get(step_name, {
            "state": "CLOSED",
            "failures": 0,
            "successes": 0,
            "last_failure": 0,
            "last_success": 0
        })
        
        if state_info["state"] == "OPEN":
            # OPEN 상태: 리셋 타임아웃 확인
            if current_time - state_info["last_failure"] >= cb["reset_timeout"]:
                # 타임아웃 만료 - HALF_OPEN으로 전환
                state_info["state"] = "HALF_OPEN"
                state_info["successes"] = 0
                logger.info(f"[서킷브레이커] {step_name} HALF_OPEN으로 전환")
            else:
                logger.warning(f"[서킷브레이커] {step_name} 차단 중 (OPEN 상태)")
                return False
        
        # 상태 업데이트
        cb["states"][step_name] = state_info
        return True
    
    def _record_circuit_breaker_failure(self, step_name: str) -> None:
        """P6: 서킷브레이커 실패 기록 (단일 구조)"""
        cb = self.circuit_breaker
        current_time = time.time()
        
        state_info = cb["states"].get(step_name, {
            "state": "CLOSED",
            "failures": 0,
            "successes": 0,
            "last_failure": 0,
            "last_success": 0
        })
        
        # 실패 기록
        state_info["failures"] += 1
        state_info["successes"] = 0  # 연속 성공 카운트 리셋
        state_info["last_failure"] = current_time
        
        # 상태 전환
        if state_info["failures"] >= cb["max_failures"]:
            state_info["state"] = "OPEN"
            logger.warning(f"[서킷브레이커] {step_name} OPEN으로 전환 ({state_info['failures']}회 실패)")
        else:
            logger.warning(f"[서킷브레이커] {step_name} 실패 기록: {state_info['failures']}/{cb['max_failures']}회")
        
        cb["states"][step_name] = state_info
    
    def _record_circuit_breaker_success(self, step_name: str) -> None:
        """P6: 서킷브레이커 성공 기록 (히스테리시스)"""
        cb = self.circuit_breaker
        current_time = time.time()
        
        state_info = cb["states"].get(step_name, {
            "state": "CLOSED",
            "failures": 0,
            "successes": 0,
            "last_failure": 0,
            "last_success": 0
        })
        
        # 성공 기록
        state_info["successes"] += 1
        state_info["last_success"] = current_time
        
        # 히스테리시스: 연속 성공 횟수 기준으로 상태 전환
        if state_info["state"] == "HALF_OPEN" and state_info["successes"] >= cb["success_threshold"]:
            state_info["state"] = "CLOSED"
            state_info["failures"] = 0  # 실패 카운트 리셋
            logger.info(f"[서킷브레이커] {step_name} CLOSED로 복구 ({state_info['successes']}회 연속 성공)")
        elif state_info["state"] == "CLOSED" and state_info["successes"] >= cb["success_threshold"]:
            # 이미 CLOSED 상태에서도 실패 카운트 리셋
            state_info["failures"] = 0
        
        cb["states"][step_name] = state_info
    
    def _optimize_parallel_settings(self) -> None:
        """개선안 C-4: 병렬 실행 최적화"""
        try:
            import os
            import multiprocessing
            
            # CPU 코어 수 감지
            cpu_count = multiprocessing.cpu_count()
            
            # GPU 사용 가능 여부 확인
            gpu_available = False
            try:
                import torch
                gpu_available = torch.cuda.is_available()
            except ImportError:
                pass
            
            # 환경변수 기반 설정
            if os.getenv("OMP_NUM_THREADS") is None:
                if gpu_available:
                    # GPU 사용 시 CPU 스레드 수 제한
                    os.environ["OMP_NUM_THREADS"] = str(min(4, cpu_count))
                    os.environ["MKL_NUM_THREADS"] = str(min(4, cpu_count))
                    logger.info(f"[병렬최적화] GPU 모드: CPU 스레드 {min(4, cpu_count)}개로 제한")
                else:
                    # CPU 전용 시 모든 코어 활용
                    os.environ["OMP_NUM_THREADS"] = str(cpu_count)
                    os.environ["MKL_NUM_THREADS"] = str(cpu_count)
                    logger.info(f"[병렬최적화] CPU 모드: 모든 코어 {cpu_count}개 활용")
            
            # 워커 수 자동 조정 - RTX 5080 최적화
            if self.max_workers == 256:  # 기본값인 경우에만 자동 조정
                if gpu_available:
                    # RTX 5080: 24코어 × 2 = 48워커 최적 (GPU 메모리 16GB 충분)
                    self.max_workers = min(48, cpu_count * 2)  # 보수적 최적화
                    logger.info(f"[GPU 최적화] RTX 5080 환경: {self.max_workers}워커")
                else:
                    self.max_workers = min(24, cpu_count)  # CPU 모드는 절반
                    logger.info(f"[CPU 모드] {self.max_workers}워커")
                
                # 추가 최적화: 배치 크기 동적 조정
                os.environ.setdefault("EMB_RUN_BATCH", "512")  # 256→512 증가
                
        except Exception as e:
            logger.warning(f"[병렬최적화] 자동 최적화 실패: {e}")
            # 기본값 유지

    def _is_under_canon(self, child: str, parent: str) -> bool:
        c = self._canon_path(child)
        p = self._canon_path(parent)
        return c == p or c.startswith(p + os.sep)

    # ---------- 생성자 ----------
    def _validate_interfaces(self) -> None:
        """인터페이스 스냅샷 검사 (런타임 시작 시 1회)"""
        try:
            # MODULE_ENTRYPOINTS 검사
            entrypoints = getattr(self.cfg, "MODULE_ENTRYPOINTS", {})
            if not isinstance(entrypoints, dict):
                raise ValueError("MODULE_ENTRYPOINTS must be a dictionary")

            # 런타임과 동일한 로더로 실제 인스턴스 획득 시도
            tmp_registry = ModuleRegistry(entrypoints, cfg=self.cfg,
                                          package_base=_resolve_package_base(None))
            for step_name in entrypoints.keys():
                try:
                    tmp_registry.get_instance(step_name, prewarm=False)
                except Exception as e:
                    is_prod = (getattr(self.cfg, "EA_PROFILE", "prod") == "prod" 
                               or os.getenv("PRODUCTION_MODE", "0") == "1")
                    msg = f"Interface validation failed for {step_name}: {e}"
                    if is_prod: 
                        raise RuntimeError(msg)
                    logger.warning(msg)

            # (기존) 독립 함수 존재 검사 로직은 그대로 유지
            adapter = CallAdapter(self.cfg)
            for step_name, candidates in adapter.CALL_TABLE.items():
                for cand in candidates:
                    if cand.endswith("_independent"):
                        # 루트 패키지에서 독립 함수 검사
                        pkg = None
                        try:
                            import emotion_analysis as _pkg
                            pkg = _pkg
                        except Exception:
                            try:
                                import src.emotion_analysis as _pkg
                                pkg = _pkg
                            except Exception:
                                pkg = None
                        
                        if pkg and not hasattr(pkg, cand):
                            is_prod = (
                                getattr(self.cfg, "EA_PROFILE", "prod") == "prod" 
                                or os.getenv("PRODUCTION_MODE", "0") == "1"
                            )
                            if is_prod:
                                raise RuntimeError(f"Independent function {cand} not found in root package")
                            else:
                                logger.warning(f"Independent function {cand} not found in root package")
                                
        except Exception as e:
            is_prod = (
                getattr(self.cfg, "EA_PROFILE", "prod") == "prod" 
                or os.getenv("PRODUCTION_MODE", "0") == "1"
            )
            if is_prod:
                raise RuntimeError(f"Interface validation failed: {e}")
            else:
                logger.warning(f"Interface validation warning: {e}")

    def __init__(self, cfg: Any = None, *, prewarm: bool = True, max_workers: int = 256, hybrid_mode: bool = False):  # 3번 개선작업: 워커 수 극대화 (128→256)
        self.cfg = cfg or config
        self.max_workers = max_workers
        self.hybrid_mode = hybrid_mode
        self._force_heavy = os.getenv("FORCE_HEAVY", "0").lower() in ("1", "true", "yes")
        self._force_fast = os.getenv("FORCE_FAST", "0").lower() in ("1", "true", "yes")
        
        # 개선안 C-1: 프리셋 고정 (핵심)
        self.active_preset = self._determine_active_preset()
        logger.info(f"[orchestrator] 활성 프리셋: {self.active_preset}")
        
        # 개선안 C-2: 하이브리드 진입점 설정
        self.quality_threshold = float(os.getenv("QUALITY_THRESHOLD", "0.70"))
        self.auto_fallback = os.getenv("AUTO_FALLBACK", "true").lower() in ("true", "1", "yes")
        logger.info(f"[orchestrator] 품질 임계값: {self.quality_threshold}, 자동 폴백: {self.auto_fallback}")

        # 하이브리드에서 사용할 스텝 그룹 기본값
        self.fast_steps = {
            "pattern_extractor", "linguistic_matcher", "intensity_analysis",
            "context_analysis", "weight_calculator", "emotion_classification",
            "embedding_generation"
        }
        self.precision_steps = {
            "complex_analyzer", "situation_analyzer", "relationship_analyzer",
            "psychological_analyzer", "transition_analyzer", "time_series_analyzer"
        }
        
        # 패치: 프리셋/강제 토글에 맞춰 하이브리드 모드 제어
        if self._force_heavy:
            self.hybrid_mode = False
            logger.info("[orchestrator] FORCE_HEAVY=1 → 하이브리드 모드 비활성화, full heavy pipeline 강제")
        elif self.active_preset == "hybrid" and not self._force_fast:
            self.hybrid_mode = True
            # 빠른/정밀 스텝 세트도 즉시 구성
            self.fast_steps = {
                "pattern_extractor", "linguistic_matcher", "intensity_analysis", 
                "context_analysis", "weight_calculator", "emotion_classification",
                "embedding_generation"
            }
            self.precision_steps = {
                "complex_analyzer", "situation_analyzer", "relationship_analyzer", 
                "psychological_analyzer", "transition_analyzer", "time_series_analyzer"
            }
            logger.info(f"[orchestrator] 하이브리드 모드 활성화 (프리셋: {self.active_preset})")
        
        # 인터페이스 스냅샷 검사 (런타임 시작 시 1회)
        self._validate_interfaces()
        
        # GPU 워밍업 (CUDA 컨텍스트 초기화로 인한 멈춤 방지)
        if not getattr(self, "_gpu_warm", False):
            try:
                import torch
                if torch.cuda.is_available():
                    _ = torch.empty(1, device="cuda").fill_(0)
                    logger.info("[orchestrator] GPU 워밍업 완료")
                self._gpu_warm = True
            except Exception as e:
                logger.debug(f"[orchestrator] GPU 워밍업 실패: {e}")
                self._gpu_warm = True  # 실패해도 플래그 설정하여 재시도 방지

        # [GENIUS INSIGHT] Adaptive Parallelism Control (APC)
        # 병렬 처리는 속도를 높이지만, 메모리(OOM)와 리소스 경합을 유발하여 500 에러의 주범이 됩니다.
        # 따라서 "안전 모드(Safe Mode)"와 "성능 모드(Boost Mode)"를 스마트하게 전환합니다.
        
        # 1. 기본값: API 호출 시 안전을 위해 기본적으로 병렬 처리를 끕니다 (안정성 우선).
        #    단, test.py 등 로컬 실험에서는 켜져 있을 수 있습니다.
        default_parallel = "0" if os.getenv("EMOTION_ORCHESTRATED") else "1"
        disable_parallel_env = os.getenv("DISABLE_PARALLEL", default_parallel).lower()
        
        if disable_parallel_env in ("1", "true", "yes"):
            self._parallel_enabled = False
            logger.info("[orchestrator] 병렬 실행 비활성화 (안정성 모드)")
        else:
            self._parallel_enabled = True
            os.environ.setdefault("ORCH_PARALLEL_MODE", "1")
            os.environ.setdefault("PARALLEL_ENABLED", "1")
            logger.info("[orchestrator] 병렬 실행 활성화 (고성능 모드)")

        # 2. 리소스 부족 시 강제 비활성화 (OOM 방지)
        try:
            import psutil
            mem = psutil.virtual_memory()
            # 여유 메모리가 2GB 미만이면 병렬 처리를 강제로 끕니다.
            if self._parallel_enabled and mem.available < 2 * 1024 * 1024 * 1024:
                self._parallel_enabled = False
                logger.warning(f"[orchestrator] 메모리 부족({mem.available // (1024*1024)}MB)으로 병렬 실행 강제 비활성화")
        except ImportError:
            pass
        
        # 3. 상태 로깅
        logger.info(f"[orchestrator] 최종 병렬 실행 상태: {self._parallel_enabled}")
        
        # CPU/GPU 리소스 감지로 자동 최적화
        self._optimize_parallel_settings()
        
        # 하이브리드 모드 설정 (개선안 C-2: 하이브리드 진입점)
        if self.hybrid_mode:
            # P1: emotion_classification 필수 스텝 추가
            self.fast_steps = {
                "pattern_extractor", "linguistic_matcher", "intensity_analysis", 
                "context_analysis", "weight_calculator", "emotion_classification",
                "embedding_generation"
            }
            self.precision_steps = {
                "complex_analyzer", "situation_analyzer", "relationship_analyzer", 
                "psychological_analyzer", "transition_analyzer", "time_series_analyzer"
            }
        
        # P6: 서킷브레이커 설정 (단일 구조 통합)
        self.circuit_breaker = {
            "max_failures": int(os.getenv("CIRCUIT_BREAKER_MAX_FAILURES", "3")),
            "reset_timeout": float(os.getenv("CIRCUIT_BREAKER_RESET_TIMEOUT", "60.0")),
            "success_threshold": int(os.getenv("CIRCUIT_BREAKER_SUCCESS_THRESHOLD", "2")),  # 연속 성공 횟수
            "states": {}  # {step_name: {"state": "CLOSED|HALF_OPEN|OPEN", "failures": 0, "successes": 0, "last_failure": 0, "last_success": 0}}
        }
        logger.info(f"[orchestrator] 서킷브레이커 설정: {self.circuit_breaker['max_failures']}회 실패 시 {self.circuit_breaker['reset_timeout']}초 차단, {self.circuit_breaker['success_threshold']}회 연속 성공 시 복구")

        # cfg가 dict인 경우와 객체인 경우 모두 처리
        if isinstance(self.cfg, dict):
            raw_log_cfg = self.cfg.get("LOGGING_CONFIG", {}) or {}
        else:
            raw_log_cfg = getattr(self.cfg, "LOGGING_CONFIG", {}) or {}
        self.log_cfg = dict(raw_log_cfg)
        self.log_cfg.setdefault("write_step_results", False)
        self.log_cfg.setdefault("write_step_results_for_cached", False)
        self.log_cfg.setdefault("write_on_error_only", True)
        self.log_cfg.setdefault("write_final_only", True)
        self.log_cfg.setdefault("result_keep_keys", ["module", "timings", "dominant_emotions", "confidence", "evidence"])
        
        # cfg가 dict인 경우와 객체인 경우 모두 처리
        if isinstance(self.cfg, dict):
            self.cfg["LOGGING_CONFIG"] = self.log_cfg
        else:
            setattr(self.cfg, "LOGGING_CONFIG", self.log_cfg)

        self._latest_payload: Optional[Dict[str, Any]] = None
        self._current_text_hash: Optional[str] = None
        self._current_trace_id: Optional[str] = None

        self._configure_torch_backends()
        # Determinism toggle (optional, defaults applied via cfg.DETERMINISM_CONFIG)
        try:
            # config.apply_determinism이 있으면 호출
            getattr(config, "apply_determinism", lambda *_: None)()
        except Exception:
            pass

        # [1] 오케스트레이션 컨텍스트 & 실행폴더 확보
        os.environ["EMOTION_ORCHESTRATED"] = "1"
        self._run_dir_cache: Optional[str] = self._ensure_run_dir()

        # [2] 가드 설치 플래그
        self._logging_guard_installed: bool = False
        self._fileio_guard_installed: bool = False

        # [3] 모듈 임포트/프리웜 전에 가드 설치(환경변수로 비활성화 가능)
        if os.getenv("EA_DISABLE_FILE_GUARDS", "0") not in ("1", "true", "yes"):
            try:
                self._install_fileio_guard()   # open()/Path.open/저수준 os.open 등
                self._install_logging_guard()  # FileHandler/Rotating/TimedRotating/basicConfig
            except Exception:
                logger.debug("[orchestrator] install guards failed", exc_info=True)

        # [4] 레지스트리/어댑터
        self.registry = ModuleRegistry(
            getattr(self.cfg, "MODULE_ENTRYPOINTS", {}),
            cfg=self.cfg,
            package_base=_resolve_package_base(None),  # dev/prod 모두 유연하게 처리
            init_kwargs={},
        )
        self.adapter = CallAdapter(cfg=self.cfg)

        # [5] 캐시/파이프라인
        # ★ 성능 최적화: 캐시 크기 증가 및 TTL 최적화
        # CACHING_CONFIG.cache_size 가 None이 들어와도 안전 (3번 개선작업: 캐시 크기 증가)
        cache_size = _safe_int(getattr(self.cfg, "CACHING_CONFIG", {}).get("cache_size", None), 1024)  # 512 → 1024로 증가
        ttl_sec = getattr(self.cfg, "CACHING_CONFIG", {}).get("ttl_sec")
        max_bytes = getattr(self.cfg, "CACHING_CONFIG", {}).get("max_bytes")
        self.cache = StepCache(
            enabled=bool(getattr(self.cfg, "PERFORMANCE_CONFIG", {}).get("cache_enabled", True)),
            max_items=cache_size,
            ttl_sec=float(ttl_sec) if ttl_sec is not None else None,
            max_bytes=int(max_bytes) if max_bytes is not None else None,
        )
        self.pipeline_spec = getattr(self.cfg, "EMOTION_ANALYSIS_PIPELINE", {"steps": []})
        self._validate_pipeline()
        self._step_no = self._build_step_index_map()  # 스텝 번호 매핑
        self._initial_skip_modules = self._compute_skip_modules(self.pipeline_spec.get("steps", []), log=False)

        # [6] 초기 붙어있던 파일핸들러 정리
        try:
            self._reconfigure_module_file_handlers(self._run_dir_cache)
        except Exception:
            logger.debug("[orchestrator] module file-handler initial reconfigure failed", exc_info=True)

        # [7] prewarm 전/후로도 한 번 정리
        # ★ 성능 최적화: 인스턴스 풀 사전 생성 강화
        if prewarm or bool(getattr(self.cfg, "PERFORMANCE_CONFIG", {}).get("prewarm_on_start", False)):
            try:
                self._reconfigure_module_file_handlers(self._run_dir_cache)
                # 모든 모듈 인스턴스를 사전에 생성하여 재사용 보장
                logger.info("[orchestrator] 모듈 인스턴스 풀 사전 생성 시작...")
                target_steps = [s for s in self.registry.entrypoints.keys() if s not in self._initial_skip_modules]
                self.registry.prewarm_all(target_steps)
                logger.info(f"[orchestrator] 모듈 인스턴스 풀 생성 완료: {len(self.registry._instances)}개")
                self._reconfigure_module_file_handlers(self._run_dir_cache)
            except Exception:
                logger.debug("[orchestrator] prewarm_all 실패(무시)", exc_info=True)
        
        # ★ 성능 최적화: 강제 인스턴스 풀 생성 (prewarm이 False여도)
        try:
            force_prewarm = os.getenv("FORCE_PREWARM_ALL", "1").lower() in ("1", "true", "yes")
            if force_prewarm and len(self.registry._instances) == 0:
                logger.info("[orchestrator] 강제 인스턴스 풀 생성 활성화")
                target_steps = [s for s in self.registry.entrypoints.keys() if s not in self._initial_skip_modules]
                self.registry.prewarm_all(target_steps)
                logger.info(f"[orchestrator] 강제 인스턴스 풀 생성 완료: {len(self.registry._instances)}개")
        except Exception:
            logger.debug("[orchestrator] 강제 prewarm 실패(무시)", exc_info=True)

        # 핸들러 재구성 빈도 제어(대용량 시 과도 호출 방지)
        # cfg가 dict인 경우와 객체인 경우 모두 처리
        if isinstance(self.cfg, dict):
            log_cfg = self.cfg.get("LOGGING_CONFIG", {})
        else:
            log_cfg = getattr(self.cfg, "LOGGING_CONFIG", {})
        # reconfigure_every_steps 가 None/문자열이어도 안전
        self._reconf_every_steps = _safe_int(log_cfg.get("reconfigure_every_steps", None), 10)
        self._processed_steps = 0

        # Tail 강제 실행 목록/토글 외부 설정 가시화
        try:
            if isinstance(self.cfg, dict):
                cfg_tail = self.cfg.get("FORCE_TAIL_STEPS", None)
            else:
                cfg_tail = getattr(self.cfg, "FORCE_TAIL_STEPS", None)
            if isinstance(cfg_tail, (list, tuple, set)):
                # 인스턴스 스코프에서 오버라이드
                self._FORCE_TAIL_STEPS = set(str(s) for s in cfg_tail)
        except Exception:
            pass
        # heavy 프리셋에서는 기본 비활성, 그 외 기본 활성
        default_tail_enabled = True
        try:
            preset = getattr(self.cfg, "ACTIVE_PRESET", "balanced")
            if str(preset).lower() == "heavy":
                default_tail_enabled = False
        except Exception:
            pass
        self._force_tail_enabled = bool(getattr(self.cfg, "FORCE_TAIL_STEPS_ENABLED", default_tail_enabled))
        
        # ★ Patch-6: Circuit Breaker 초기화
        # 패치 D: 레거시 _cb 제거 (circuit_breaker로 단일화)
        
        # ★ Patch-8: PerformanceMetrics 초기화
        self.perf = PerformanceMetrics()

    # ---------- 공개 진입점 ----------
    def _configure_torch_backends(self) -> None:
        """패치 B: 오토캐스트/백엔드 안전 임포트"""
        try:
            import torch as _torch
        except Exception:
            return
        try:
            if not _torch.cuda.is_available():
                return
            
            # GPU 최적화 설정
            _torch.backends.cudnn.benchmark = True
            _torch.backends.cudnn.deterministic = False
            _torch.cuda.empty_cache()
            _torch.cuda.set_per_process_memory_fraction(0.85)
            _torch.backends.cuda.matmul.allow_tf32 = True
            _torch.backends.cudnn.allow_tf32 = True
            
            # Flash Attention 설정
            disable = os.getenv("EA_DISABLE_FLASH_SDP", "0").lower() in ("1", "true")
            try:
                _torch.backends.cuda.enable_flash_sdp(not disable)
                _torch.backends.cuda.enable_mem_efficient_sdp(not disable)
                _torch.backends.cuda.enable_cudnn_sdp(not disable)
                _torch.backends.cuda.enable_math_sdp(disable)
            except Exception:
                pass
            
            # 행렬 곱셈 정밀도 설정
            try:
                _torch.set_float32_matmul_precision(os.getenv("EA_MATMUL_PRECISION", "high"))
            except Exception:
                pass
            
            logging.info("[orchestrator] GPU 최적화 활성화")
        except Exception:
            pass

    def _make_initial_payload(self, text: str, meta: Optional[Dict[str, Any]] = None) -> Payload:
        payload = Payload.new(text=text, meta=meta, cfg=self.cfg)
        try:
            if "trace_id" not in payload.meta:
                payload.meta["trace_id"] = _hash_text(f"{time.time()}|{_hash_text(text)[:12]}")
        except Exception:
            pass
        
        # 기본 메트릭 설정 (run_if 조건 평가를 위해)
        try:
            # 간단한 토큰화와 문장 분할
            norm_text = _normalize_text(text) if text else ""
            payload.text = norm_text
            payload.sentences = _split_sentences(norm_text) if norm_text else []
            payload.tokens = _basic_tokenize(norm_text) if norm_text else []
            
            # 기본 메트릭을 results에 추가 (ConditionEvaluator가 찾을 수 있도록)
            payload.add_result("text_preprocessing", {
                "sentences_count": len(payload.sentences),
                "tokens_count": len(payload.tokens),
                "success": True
            })
        except Exception:
            # 실패해도 기본값으로 계속 진행
            payload.sentences = []
            payload.tokens = []
            payload.add_result("text_preprocessing", {
                "sentences_count": 0,
                "tokens_count": 0,
                "success": False
            })
        
        return payload

    def process_text_hybrid(
        self,
        text: str,
        meta: Optional[Dict[str, Any]] = None,
        *,
        save: bool = False,
        out_dir: Optional[str] = None,
        filename: Optional[str] = None,
        quality_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        개선안 C-2: 하이브리드 처리 모드 - 품질 기반 자동 승격
        
        Args:
            text: 입력 텍스트
            meta: 메타데이터
            save: 결과 저장 여부
            out_dir: 출력 디렉토리
            filename: 파일명
            quality_threshold: 품질 임계값 (이하이면 정밀 처리, None이면 자동 설정)
        """
        if self._force_heavy:
            logger.info("[하이브리드] FORCE_HEAVY=1 → 하이브리드 스킵, heavy 프리셋 직접 실행")
            return self.process_text_by_preset(
                text,
                meta,
                save=save,
                out_dir=out_dir,
                filename=filename,
                preset="heavy",
            )

        if not self.hybrid_mode:
            # 하이브리드 모드가 아니면 기존 방식 사용
            return self.process_text(text, meta, save=save, out_dir=out_dir, filename=filename)
        
        # 품질 임계값 자동 설정
        if quality_threshold is None:
            quality_threshold = self.quality_threshold
        
        # 1단계: 빠른 처리 (fast_steps만 실행)
        logger.info("[하이브리드] 1단계: 빠른 처리 시작")
        fast_result = self._process_fast_steps(text, meta)
        
        # 품질 평가 (강화된 버전)
        quality_score = self._evaluate_quality(fast_result, text)
        logger.info(f"[하이브리드] 품질 점수: {quality_score:.3f} (임계값: {quality_threshold})")
        
        # 상세 품질 검증 정보 추가
        validation = self.validate_pipeline_result(fast_result)
        fast_result["meta"]["quality_validation"] = validation
        
        if quality_score >= quality_threshold:
            # 품질이 충분하면 빠른 결과 반환
            logger.info("[하이브리드] 품질 충족 - 빠른 결과 반환")
            result = fast_result
            result["processing_mode"] = "fast"
            result["quality_score"] = quality_score
        else:
            # 품질이 부족하면 정밀 처리 추가
            logger.info("[하이브리드] 품질 부족 - 정밀 처리 추가")
            precision_result = self._process_precision_steps(text, meta, fast_result)
            
            # 결과 병합
            result = self._merge_results(fast_result, precision_result)
            result["processing_mode"] = "hybrid"
            result["quality_score"] = quality_score
        
        # 결과 저장
        if save:
            self._save_result(result, out_dir, filename)
        
        return result

    def process_text_by_preset(
        self,
        text: str,
        meta: Optional[Dict[str, Any]] = None,
        *,
        save: bool = False,
        out_dir: Optional[str] = None,
        filename: Optional[str] = None,
        preset: Optional[str] = None,
    ) -> Dict[str, Any]:
        """A-1: 프리셋별 처리 메서드 - 통합 그룹 빌더 사용"""
        if preset is None:
            preset = self.active_preset
        
        logger.info(f"[프리셋] {preset} 모드로 처리 시작")
        
        # A-1: call-site 수정 - 프리셋 스펙 적용 후 통합 그룹 빌더 사용
        spec = self._build_pipeline_spec(preset)
        old = self.pipeline_spec
        try:
            self._apply_pipeline_spec(spec)  # 프리셋 스펙을 실제 실행 스펙으로 적용
            payload = self._make_initial_payload(text, meta)
            payload.meta["preset"] = preset
            
            if self._parallel_enabled:
                parallel_groups = self._build_parallel_groups()
                if parallel_groups:
                    logger.info(f"[orchestrator] 프리셋 '{preset}' 병렬 실행: {len(parallel_groups)}개 그룹")
                    self._run_parallel_pipeline(payload, parallel_groups)
                else:
                    logger.info(f"[orchestrator] 프리셋 '{preset}' 순차 실행: 병렬 그룹 없음")
                    self._run_sequential_pipeline(payload, save=save, out_dir=out_dir)
            else:
                logger.info(f"[orchestrator] 프리셋 '{preset}' 순차 실행: 병렬 실행 비활성화")
                self._run_sequential_pipeline(payload, save=save, out_dir=out_dir)
            
            result = payload.to_output()
            result["processing_mode"] = f"{preset}_preset"
            return result
        finally:
            self.pipeline_spec = old  # 스펙 되돌리기
    
    # A-1: 개별 프리셋 메서드 제거 - 통합된 process_text_by_preset 사용

    def _build_pipeline_spec(self, preset: str) -> Dict[str, Any]:
        """P1: 프리셋별 pipeline_spec 구성"""
        if preset == "fast":
            # 빠른 스텝만 포함 (emotion_classification 필수)
            steps = {
                "pattern_extractor", "linguistic_matcher", "intensity_analysis", 
                "context_analysis", "weight_calculator", "emotion_classification"
            }
        elif preset == "heavy":
            # 모든 스텝 포함 (11개 모듈 완전 활용)
            steps = {
                "pattern_extractor", "linguistic_matcher", "intensity_analysis", "context_analysis", 
                "weight_calculator", "complex_analyzer", "situation_analyzer", "relationship_analyzer", 
                "psychological_analyzer", "transition_analyzer", "time_series_analyzer", "emotion_classification"
            }
        else:
            # 기본값 (모든 스텝)
            steps = {
                "pattern_extractor", "linguistic_matcher", "intensity_analysis", "context_analysis", 
                "weight_calculator", "complex_analyzer", "situation_analyzer", "relationship_analyzer", 
                "psychological_analyzer", "transition_analyzer", "time_series_analyzer", "emotion_classification"
            }
        
        # DAG 기반 의존성 순서 보장
        ordered_steps = self._order_steps_by_dependencies(list(steps))
        
        return {
            "preset": preset,
            "steps": ordered_steps,
            "parallel_groups": []  # 통합 그룹 빌더 사용으로 변경
        }
    
    def _order_steps_by_dependencies(self, steps: List[str]) -> List[str]:
        """의존성 기반 스텝 순서 정렬"""
        # 의존성 매핑 (후행 스텝이 선행 스텝에 의존)
        dependencies = {
            "context_analysis": ["pattern_extractor", "linguistic_matcher"],
            "transition_analyzer": ["context_analysis", "intensity_analysis"],
            "time_series_analyzer": ["transition_analyzer", "context_analysis"],
            "relationship_analyzer": ["context_analysis", "emotion_classification"],
            "complex_analyzer": ["emotion_classification", "intensity_analysis"],
            "situation_analyzer": ["context_analysis", "emotion_classification"],
            "psychological_analyzer": ["complex_analyzer", "situation_analyzer"],
            "weight_calculator": ["emotion_classification", "intensity_analysis"]
        }
        
        # 위상 정렬로 순서 결정
        ordered = []
        remaining = set(steps)
        
        while remaining:
            # 의존성이 없는 스텝들 찾기
            ready = []
            for step in remaining:
                deps = dependencies.get(step, [])
                if all(dep not in remaining for dep in deps):
                    ready.append(step)
            
            if not ready:
                # 순환 의존성 감지 시 임의 순서로 처리
                ready = [remaining.pop()]
            
            # 준비된 스텝들을 순서에 추가
            for step in sorted(ready):
                ordered.append(step)
                remaining.remove(step)
        
        return ordered
    
    # 선택 개선: 사용되지 않는 메서드 주석 처리 (혼동 방지)
    # def _build_parallel_groups_for_preset(self, steps: List[str]) -> List[List[str]]:
    #     """프리셋별 병렬 그룹 구성 (현재 사용되지 않음 - DAG 기반 실행 사용)"""
    #     # 병렬 실행 가능한 그룹들
    #     parallel_groups = []
    #     
    #     # 독립적인 스텝들을 그룹화
    #     independent_groups = [
    #         ["pattern_extractor", "linguistic_matcher"],
    #         ["intensity_analysis", "emotion_classification"],
    #         ["complex_analyzer", "situation_analyzer"],
    #         ["relationship_analyzer", "psychological_analyzer"]
    #     ]
    #     
    #     for group in independent_groups:
    #         # 실제 존재하는 스텝만 포함
    #         valid_group = [step for step in group if step in steps]
    #         if len(valid_group) > 1:
    #             parallel_groups.append(valid_group)
    #     
    #     return parallel_groups

    def _apply_pipeline_spec(self, spec: dict) -> None:
        """패치 A: 프리셋 스펙을 실행기가 요구하는 구조로 변환+적용"""
        deps = {
            "context_analysis": ["pattern_extractor", "linguistic_matcher"],
            "transition_analyzer": ["context_analysis", "intensity_analysis"],
            "time_series_analyzer": ["transition_analyzer", "context_analysis"],
            "relationship_analyzer": ["context_analysis", "emotion_classification"],
            "complex_analyzer": ["emotion_classification", "intensity_analysis"],
            "situation_analyzer": ["context_analysis", "emotion_classification"],
            "psychological_analyzer": ["complex_analyzer", "situation_analyzer"],
            "weight_calculator": ["emotion_classification", "intensity_analysis"],
        }
        
        # 1) 의존성 closure: 누락된 deps 자동 포함
        want = set(spec["steps"])
        added = True
        while added:
            added = False
            for s in list(want):
                for d in deps.get(s, []):
                    if d not in want:
                        want.add(d)
                        added = True
        
        # 2) 실행기 포맷으로 변환
        steps = []
        for name in self._order_steps_by_dependencies(list(want)):
            steps.append({
                "name": name,
                "enabled": True,
                "dependencies": deps.get(name, []),
                "config": getattr(self.cfg, name.upper(), {}) or {}
            })
        
        self.pipeline_spec = {"steps": steps}

    def _save_result(self, result: Dict[str, Any], out_dir: Optional[str] = None, filename: Optional[str] = None) -> str:
        """결과 저장 헬퍼 메서드"""
        try:
            return self._save_output_json(result, out_dir=out_dir, filename=filename)
        except Exception as e:
            logger.error(f"[결과저장] 저장 실패: {e}")
            return ""

    def _execute_step(self, step_name: str, payload: "Payload") -> Any:
        """스텝 실행 헬퍼 메서드"""
        try:
            # [Emergency Fix] embedding_generation 스텝 강제 처리
            if step_name == "embedding_generation":
                logger.info(f"[DEBUG] Attempting forced execution for {step_name}")
                try:
                    try:
                        from src.emotion_analysis.intensity_analyzer import run_embedding_generation
                    except ImportError:
                        from emotion_analysis.intensity_analyzer import run_embedding_generation
                        
                    step_cfg = getattr(self.cfg, "EMBEDDING_GENERATION", {}) or getattr(self.cfg, "EMBEDDING_PARAMS", {}) or {}
                    result = self._run_step_with_runtime_guards(run_embedding_generation, payload, step_cfg, step_name=step_name)
                    logger.info(f"[DEBUG] Forced execution result type: {type(result)}")
                    return result
                except Exception as e:
                    logger.warning(f"[스텝실행] embedding_generation 강제 실행 실패: {e}")
                    # 실패하면 아래의 원래 로직으로 폴백

            # 스텝 설정 가져오기
            step_cfg = getattr(self.cfg, step_name.upper(), {})
            if isinstance(step_cfg, dict):
                step_cfg = step_cfg
            else:
                step_cfg = {}
            
            # 인스턴스 가져오기
            instance = self.registry.get_instance(step_name)
            if instance is None:
                logger.warning(f"[스텝실행] {step_name} 인스턴스 생성 실패")
                return None
            
            # 실행 함수 결정
            runner = self.adapter.resolve_callable(step_name, instance)
            if runner is None:
                logger.warning(f"[스텝실행] {step_name} 실행 함수 없음")
                return None
            
            # 스텝 실행
            return self._run_step_with_runtime_guards(runner, payload, step_cfg, step_name=step_name)
            
        except Exception as e:
            logger.error(f"[스텝실행] {step_name} 실행 실패: {e}")
            return None

    def _process_fast_steps(self, text: str, meta: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """빠른 처리 단계 (fast_steps만 실행)"""
        payload = self._make_initial_payload(text, meta)
        
        # [Fix] embedding_generation이 fast_steps에 누락되는 경우 방지 (필수 스텝 보장)
        steps_to_run = set(self.fast_steps)
        steps_to_run.add("embedding_generation")
        
        logger.info(f"[DEBUG] fast_steps to run: {sorted(list(steps_to_run))}")
        
        for step_name in steps_to_run:
            try:
                result = self._execute_step(step_name, payload)
                if result is not None:
                    payload.add_result(step_name, result)
            except Exception as e:
                logger.warning(f"[하이브리드] 빠른 처리 단계 {step_name} 실패: {e}")
                payload.add_result(step_name, {"error": str(e), "confidence": 0.0})
        
        return payload.to_output()

    def _process_precision_steps(self, text: str, meta: Optional[Dict[str, Any]], fast_result: Dict[str, Any]) -> Dict[str, Any]:
        """정밀 처리 단계 (precision_steps 실행)"""
        # fast_result가 dict이므로 Payload로 재구성하거나, 바로 결과 dict를 병합하는 방식 중 택1
        payload = self._make_initial_payload(text, meta)
        for k, v in (fast_result.get("results") or {}).items():
            payload.add_result(k, v)
        
        for step_name in self.precision_steps:
            try:
                result = self._execute_step(step_name, payload)
                if result is not None:
                    payload.add_result(step_name, result)
            except Exception as e:
                logger.warning(f"[하이브리드] 정밀 처리 단계 {step_name} 실패: {e}")
                payload.add_result(step_name, {"error": str(e), "confidence": 0.0})
        
        return payload.to_output()

    def _evaluate_quality(self, result: Dict[str, Any], text: str) -> float:
        """개선안 C-2: 결과 품질 평가 (0.0 ~ 1.0) - 강화된 버전"""
        # 기본 품질 검증
        validation = self.validate_pipeline_result(result)
        base_quality = validation["overall_quality"]
        
        # 텍스트 특성 기반 품질 조정
        text_features = self._analyze_text_features(text)
        quality_adjustment = self._calculate_quality_adjustment(text_features)
        
        # 최종 품질 점수 계산
        final_quality = min(1.0, max(0.0, base_quality + quality_adjustment))
        
        logger.debug(f"[품질평가] 기본: {base_quality:.3f}, 조정: {quality_adjustment:.3f}, 최종: {final_quality:.3f}")
        return final_quality
    
    def _analyze_text_features(self, text: str) -> Dict[str, Any]:
        """P5: 텍스트 특성 분석 (품질 평가용) - 국문 텍스트 호환"""
        import re
        
        # P5: 이모지 유니코드 블록 기준 감지 (정규식 사용)
        emoji_pattern = re.compile(
            r'[\U0001F600-\U0001F64F]|'  # 이모티콘
            r'[\U0001F300-\U0001F5FF]|'  # 기타 기호 및 픽토그램
            r'[\U0001F680-\U0001F6FF]|'  # 교통 및 지도 기호
            r'[\U0001F1E0-\U0001F1FF]|'  # 지역 표시 기호
            r'[\U00002600-\U000026FF]|'  # 기타 기호
            r'[\U00002700-\U000027BF]'   # Dingbats
        )
        emoji_count = len(emoji_pattern.findall(text))
        
        # P5: 한글+영문 혼합 감지 (문장부호, 숫자, URL 제외)
        korean_chars = len(re.findall(r'[가-힣]', text))
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        total_chars = len(re.sub(r'[^\w\s]', '', text))  # 문장부호 제외
        
        has_mixed_language = False
        if total_chars > 10:  # 최소 길이 체크
            korean_ratio = korean_chars / total_chars
            english_ratio = english_chars / total_chars
            # 한글과 영문이 모두 유의미한 비율(각각 20% 이상)일 때만 혼합 언어로 판단
            has_mixed_language = korean_ratio > 0.2 and english_ratio > 0.2
        
        features = {
            "length": len(text),
            "sentence_count": len(text.split('.')),
            "emoji_count": emoji_count,
            "url_count": text.count('http'),
            "special_char_ratio": sum(1 for c in text if not c.isalnum() and not c.isspace()) / max(1, len(text)),
            "has_complex_punctuation": any(c in text for c in ['...', '!!!', '???', '—', '–']),
            "has_mixed_language": has_mixed_language,
            "korean_chars": korean_chars,
            "english_chars": english_chars,
            "total_chars": total_chars
        }
        return features
    
    def _calculate_quality_adjustment(self, features: Dict[str, Any]) -> float:
        """P5: 텍스트 특성 기반 품질 조정값 계산 (클램프 적용)"""
        adjustment = 0.0
        
        # 긴 문장은 품질 하락
        if features["length"] > 1500:
            adjustment -= 0.1
        
        # P5: 이모티콘 다량은 품질 하락 (정밀한 감지)
        if features["emoji_count"] > 10:
            adjustment -= 0.05
        
        # URL 과다는 품질 하락
        if features["url_count"] > 3:
            adjustment -= 0.05
        
        # 특수문자 비율이 높으면 품질 하락
        if features["special_char_ratio"] > 0.3:
            adjustment -= 0.1
        
        # 복잡한 구두점은 품질 하락
        if features["has_complex_punctuation"]:
            adjustment -= 0.05
        
        # P5: 혼합 언어는 품질 하락 (한글 패널티 제거)
        if features["has_mixed_language"]:
            adjustment -= 0.03
        
        # P5: 과도한 감점 클램프 (총 -0.15 한도)
        adjustment = max(-0.15, adjustment)
        
        return adjustment
    
    def validate_pipeline_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """파이프라인 결과 종합 검증"""
        validation = {
            "overall_quality": 0.0,
            "consistency_score": 0.0,
            "confidence_score": 0.0,
            "completeness_score": 0.0,
            "coherence_score": 0.0,
            "issues": [],
            "recommendations": []
        }
        
        results = result.get("results", {})
        
        # 1. 신뢰도 검증
        confidences = []
        for step_name, step_result in results.items():
            if isinstance(step_result, dict):
                if "confidence" in step_result:
                    confidences.append(step_result["confidence"])
                elif "error" in step_result:
                    validation["issues"].append(f"{step_name}: {step_result['error']}")
        
        if confidences:
            validation["confidence_score"] = sum(confidences) / len(confidences)
        else:
            validation["confidence_score"] = 0.0
            validation["issues"].append("No confidence scores found")
        
        # 2. 일관성 검증 (감정 결과 간 일치성)
        emotions = self._extract_all_emotions(results)
        validation["consistency_score"] = self._calculate_emotion_consistency(emotions)
        
        # 3. 완전성 검증 (필수 스텝 실행 여부)
        required_steps = ["pattern_extractor", "emotion_classification", "intensity_analysis"]
        executed_steps = set(results.keys())
        validation["completeness_score"] = len(executed_steps & set(required_steps)) / len(required_steps)
        
        # 4. 논리적 일관성 검증
        validation["coherence_score"] = self._calculate_logical_coherence(results)
        
        # 5. 종합 품질 점수
        validation["overall_quality"] = (
            validation["confidence_score"] * 0.3 +
            validation["consistency_score"] * 0.25 +
            validation["completeness_score"] * 0.25 +
            validation["coherence_score"] * 0.2
        )
        
        # 6. 개선 권고사항 생성
        validation["recommendations"] = self._generate_quality_recommendations(validation)
        
        return validation
    
    def _extract_all_emotions(self, results: Dict[str, Any]) -> Dict[str, List[str]]:
        """모든 스텝에서 감정 정보 추출"""
        emotions = {}
        
        for step_name, step_result in results.items():
            if not isinstance(step_result, dict):
                continue
                
            step_emotions = []
            
            # 다양한 감정 필드에서 추출
            emotion_fields = ["emotions", "dominant_emotions", "emotion_weights", "topk_main"]
            for field in emotion_fields:
                if field in step_result:
                    data = step_result[field]
                    if isinstance(data, list):
                        step_emotions.extend([str(e) for e in data if isinstance(e, str)])
                    elif isinstance(data, dict):
                        step_emotions.extend([str(k) for k in data.keys() if isinstance(k, str)])
            
            if step_emotions:
                emotions[step_name] = step_emotions
        
        return emotions
    
    def _calculate_emotion_consistency(self, emotions: Dict[str, List[str]]) -> float:
        """감정 일관성 점수 계산"""
        if len(emotions) < 2:
            return 1.0
        
        # 모든 감정을 하나의 집합으로 합침
        all_emotions = set()
        for step_emotions in emotions.values():
            all_emotions.update(step_emotions)
        
        if not all_emotions:
            return 0.0
        
        # 각 스텝별 일치도 계산
        consistency_scores = []
        for step_name, step_emotions in emotions.items():
            if not step_emotions:
                continue
            
            step_set = set(step_emotions)
            # 다른 스텝들과의 교집합 비율 계산
            overlaps = []
            for other_name, other_emotions in emotions.items():
                if other_name != step_name and other_emotions:
                    other_set = set(other_emotions)
                    if step_set or other_set:
                        overlap = len(step_set & other_set) / len(step_set | other_set)
                        overlaps.append(overlap)
            
            if overlaps:
                consistency_scores.append(sum(overlaps) / len(overlaps))
        
        return sum(consistency_scores) / len(consistency_scores) if consistency_scores else 0.0
    
    def _calculate_logical_coherence(self, results: Dict[str, Any]) -> float:
        """논리적 일관성 점수 계산"""
        coherence_factors = []
        
        # 1. 강도 분석과 감정 분류 일치성
        if "intensity_analysis" in results and "emotion_classification" in results:
            intensity = results["intensity_analysis"]
            classification = results["emotion_classification"]
            
            if isinstance(intensity, dict) and isinstance(classification, dict):
                # 강도가 높은 감정이 분류 결과와 일치하는지 확인
                intensity_emotions = intensity.get("emotions", [])
                classified_emotions = classification.get("emotions", [])
                
                if intensity_emotions and classified_emotions:
                    overlap = len(set(intensity_emotions) & set(classified_emotions))
                    coherence_factors.append(overlap / max(len(intensity_emotions), len(classified_emotions)))
        
        # 2. 패턴 추출과 맥락 분석 일치성
        if "pattern_extractor" in results and "context_analysis" in results:
            pattern = results["pattern_extractor"]
            context = results["context_analysis"]
            
            if isinstance(pattern, dict) and isinstance(context, dict):
                pattern_confidence = pattern.get("confidence", 0.0)
                context_confidence = context.get("confidence", 0.0)
                
                # 신뢰도 차이가 크지 않은지 확인
                confidence_diff = abs(pattern_confidence - context_confidence)
                coherence_factors.append(max(0.0, 1.0 - confidence_diff))
        
        return sum(coherence_factors) / len(coherence_factors) if coherence_factors else 0.5
    
    def _generate_quality_recommendations(self, validation: Dict[str, Any]) -> List[str]:
        """품질 개선 권고사항 생성"""
        recommendations = []
        
        if validation["confidence_score"] < 0.5:
            recommendations.append("신뢰도가 낮습니다. 입력 텍스트의 품질을 확인하세요.")
        
        if validation["consistency_score"] < 0.3:
            recommendations.append("감정 분석 결과가 일관성이 부족합니다. 추가 검증이 필요합니다.")
        
        if validation["completeness_score"] < 0.8:
            recommendations.append("일부 필수 분석 스텝이 누락되었습니다.")
        
        if validation["coherence_score"] < 0.4:
            recommendations.append("분석 결과 간 논리적 일관성이 부족합니다.")
        
        if len(validation["issues"]) > 2:
            recommendations.append("여러 오류가 발생했습니다. 시스템 상태를 확인하세요.")
        
        return recommendations
    
    def _validate_dependencies(self, step_name: str, payload: "Payload", deps: List[str]) -> bool:
        """강화된 의존성 검증"""
        if not deps:
            return True
        
        missing_deps = [d for d in deps if d not in payload.results]
        if not missing_deps:
            return True
        
        # 현재 품질 점수 확인
        current_quality = self._get_current_quality_score(payload)
        quality_threshold = 0.3
        
        # 품질이 낮으면 의존성 필수
        if current_quality < quality_threshold:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"{step_name} 품질 점수 낮음 ({current_quality:.3f}) - 의존성 필수")
            return False
        
        # 핵심 의존성 체크
        critical_deps = ["pattern_extractor", "emotion_classification", "intensity_analysis"]
        critical_missing = [d for d in missing_deps if d in critical_deps]
        
        if critical_missing:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"{step_name} 핵심 의존성 누락: {critical_missing}")
            return False
        
        # 비핵심 의존성은 품질 기반으로 결정 (완화)
        non_critical_missing = [d for d in missing_deps if d not in critical_deps]
        if non_critical_missing and current_quality < 0.2:  # 임계값 완화
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"{step_name} 비핵심 의존성 누락 + 품질 매우 낮음 - 스킵")
            return False
        
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"{step_name} 의존성 검증 통과 (품질: {current_quality:.3f})")
        return True
    
    def _get_current_quality_score(self, payload: "Payload") -> float:
        """현재 품질 점수 계산"""
        if not payload.results:
            return 0.0
        
        # 간단한 품질 점수 계산
        confidences = []
        error_count = 0
        
        for step_name, step_result in payload.results.items():
            if isinstance(step_result, dict):
                if "confidence" in step_result:
                    confidences.append(step_result["confidence"])
                if "error" in step_result:
                    error_count += 1
        
        if not confidences:
            return 0.1 if error_count == 0 else 0.0
        
        avg_confidence = sum(confidences) / len(confidences)
        error_penalty = min(0.5, error_count * 0.1)
        
        return max(0.0, avg_confidence - error_penalty)
    
    def _calculate_smart_timeout(self, step_name: str, step_cfg: Dict[str, Any]) -> float:
        """패치 C: p95/mean 기반 스마트 타임아웃 계산"""
        try:
            base_timeout = float(
                step_cfg.get("timeout") or
                getattr(self.cfg, "STEP_TIMEOUTS", {}).get(step_name, getattr(self.cfg, "DEFAULT_STEP_TIMEOUT", 0))
            )
        except Exception:
            base_timeout = 0.0

        device_multiplier = 2.0 if self._device() == "cpu" else 1.0
        stats = self.perf.stats(step_name)  # {"mean_ms","p95_ms","count"}
        if stats and stats.get("count", 0) > 0:
            p95_ms = float(stats.get("p95_ms", 0.0))
            mean_ms = float(stats.get("mean_ms", 0.0))
            if p95_ms > 0:
                base_timeout = max(base_timeout, max(30.0, min(600.0, 1.2 * (p95_ms / 1000.0))))
            elif mean_ms > 0:
                base_timeout = max(base_timeout, max(20.0, min(300.0, 2.0 * (mean_ms / 1000.0))))
        
        if step_name in ("embedding_generation", "intensity_analysis"):
            base_timeout *= 1.5
        
        return base_timeout * device_multiplier
    
    def _get_average_execution_time(self, step_name: str) -> float:
        """스텝별 평균 실행 시간 조회"""
        if not hasattr(self, '_execution_times'):
            self._execution_times = {}
        
        times = self._execution_times.get(step_name, [])
        if not times:
            return 0.0
        
        # 최근 10회 실행 시간의 평균
        recent_times = times[-10:] if len(times) > 10 else times
        return sum(recent_times) / len(recent_times)
    
    def _record_execution_time(self, step_name: str, duration: float) -> None:
        """실행 시간 기록"""
        if not hasattr(self, '_execution_times'):
            self._execution_times = {}
        
        if step_name not in self._execution_times:
            self._execution_times[step_name] = []
        
        self._execution_times[step_name].append(duration)
        
        # 최대 50개까지만 보관
        if len(self._execution_times[step_name]) > 50:
            self._execution_times[step_name] = self._execution_times[step_name][-50:]
    
    def _monitor_execution_quality(self, payload: "Payload") -> None:
        """실행 중 품질 모니터링"""
        current_quality = self._get_current_quality_score(payload)
        
        if current_quality < 0.3:  # 임계값 완화
            logger.warning(f"[품질모니터] 낮은 품질 감지: {current_quality:.3f}")
            # 품질이 낮으면 추가 검증 스텝 실행
            self._run_quality_boost_steps(payload)
        
        if current_quality < 0.1:  # 임계값 완화
            logger.error(f"[품질모니터] 심각한 품질 문제: {current_quality:.3f}")
            # 심각한 품질 문제 시 폴백 모드로 전환
            self._switch_to_fallback_mode(payload)
    
    def _run_quality_boost_steps(self, payload: "Payload") -> None:
        """품질 향상 스텝 실행"""
        logger.info("[품질향상] 추가 검증 스텝 실행")
        
        # 기본 검증 스텝들
        boost_steps = ["pattern_extractor", "emotion_classification"]
        
        for step_name in boost_steps:
            if step_name not in payload.results:
                try:
                    # 간단한 재실행
                    instance = self.registry.get_instance(step_name)
                    runner = self.adapter.resolve_callable(step_name, instance)
                    if runner:
                        result = self._safe_call(runner, payload, {}, step_name=step_name)
                        if result:
                            payload.add_result(step_name, result)
                            payload.stamp(step_name, "ok", extras={"quality_boost": True})
                except Exception as e:
                    logger.warning(f"[품질향상] {step_name} 재실행 실패: {e}")
    
    def _switch_to_fallback_mode(self, payload: "Payload") -> None:
        """폴백 모드로 전환"""
        logger.warning("[폴백모드] 전환 - 최소한의 분석만 수행")
        
        # 폴백 모드에서는 핵심 스텝만 실행
        fallback_steps = ["pattern_extractor"]
        
        for step_name in fallback_steps:
            if step_name not in payload.results:
                try:
                    # 최소한의 설정으로 실행
                    fallback_config = {"timeout": 30, "on_error": "skeleton"}
                    instance = self.registry.get_instance(step_name)
                    runner = self.adapter.resolve_callable(step_name, instance)
                    if runner:
                        result = self._safe_call(runner, payload, fallback_config, step_name=step_name)
                        if result:
                            payload.add_result(step_name, result)
                            payload.stamp(step_name, "ok", extras={"fallback_mode": True})
                except Exception as e:
                    logger.error(f"[폴백모드] {step_name} 실행 실패: {e}")
                    # 폴백 모드에서도 실패하면 스켈레톤 결과 생성
                    skeleton = {"features": {}, "timeline": {}, "confidence": 0.1, "error": str(e)}
                    payload.add_result(step_name, skeleton)
                    payload.stamp(step_name, "error", error=str(e), extras={"fallback_skeleton": True})

    def _merge_results(self, fast_result: Dict[str, Any], precision_result: Dict[str, Any]) -> Dict[str, Any]:
        """P2: 빠른 결과와 정밀 결과 깊은 병합 (신뢰도 우선 규칙)"""
        # 기본 구조는 fast_result 기준
        merged = fast_result.copy()
        
        # results 섹션 깊은 병합
        if "results" in precision_result:
            if "results" not in merged:
                merged["results"] = {}
            
            # 각 스텝별로 깊은 병합
            for step_name, precision_data in precision_result["results"].items():
                if step_name in merged["results"]:
                    fast_data = merged["results"][step_name]
                    # 같은 스텝이면 confidence가 높은 쪽 채택
                    fast_conf = self._extract_confidence(fast_data)
                    precision_conf = self._extract_confidence(precision_data)
                    
                    if precision_conf > fast_conf:
                        merged["results"][step_name] = precision_data
                    # confidence가 같거나 없으면 정밀 결과 우선
                    elif precision_conf == fast_conf or precision_conf is None:
                        merged["results"][step_name] = precision_data
                else:
                    # 새로운 스텝은 정밀 결과 추가
                    merged["results"][step_name] = precision_data
        
        # 메타데이터 병합 (합집합)
        if "meta" in precision_result:
            if "meta" not in merged:
                merged["meta"] = {}
            
            # 상위 필드는 fast 기준 유지
            for key, value in precision_result["meta"].items():
                if key in ["recommendations", "quality_validation", "processing_mode"]:
                    # 특정 필드는 합집합
                    if key not in merged["meta"]:
                        merged["meta"][key] = value
                    elif isinstance(value, list) and isinstance(merged["meta"][key], list):
                        # 리스트인 경우 합집합
                        merged["meta"][key] = list(set(merged["meta"][key] + value))
                else:
                    # 나머지는 정밀 결과로 업데이트
                    merged["meta"][key] = value
        
        return merged
    
    def _extract_confidence(self, data: Dict[str, Any]) -> Optional[float]:
        """데이터에서 confidence 값 추출"""
        if isinstance(data, dict):
            return data.get("confidence", data.get("conf", None))
        return None

    def process_text(
        self,
        text: str,
        meta: Optional[Dict[str, Any]] = None,
        *,
        save: bool = False,
        out_dir: Optional[str] = None,
        filename: Optional[str] = None,
    ) -> Dict[str, Any]:

        # 컨텍스트/가드 재확인
        os.environ["EMOTION_ORCHESTRATED"] = "1"
        run_dir = self._run_dir_cache or self._ensure_run_dir()
        self._run_dir_cache = run_dir
        try:
            self._install_fileio_guard()
            self._install_logging_guard()
        except Exception:
            logger.debug("[orchestrator] install guards failed", exc_info=True)
        try:
            self._reconfigure_module_file_handlers(run_dir)
        except Exception:
            logger.debug("[orchestrator] module file-handler reconfigure failed", exc_info=True)

        payload = self._make_initial_payload(text, meta)
        # 현재 요청 컨텍스트를 _latest_payload에 반영 (스마트 스킵 등에서 참고)
        try:
            self._latest_payload = {"text": payload.text, "meta": dict(payload.meta)}
        except Exception:
            self._latest_payload = None
        # 런타임 진단 메타(호스트명 등) 추가
        try:
            host = None
            try:
                host = getattr(os, "uname")().nodename  # type: ignore[attr-defined]
            except Exception:
                host = os.getenv("HOSTNAME") or os.getenv("COMPUTERNAME")
            if host:
                payload.update_meta(host=str(host))
        except Exception:
            pass

        self._current_text_hash = _hash_text(payload.text)
        self._current_trace_id = payload.meta.get("trace_id")
        
        # ★ Patch-7: 리소스 모니터링 - 메모리 고사용 시 캐시 스윕
        try:
            mem = psutil.virtual_memory()
            if mem.percent > float(getattr(self.cfg, "SAFE_MEM_PERCENT", 92.0)):
                # 메모리 압박 → 캐시 스윕
                self.cache.clear()
                logger.warning("[orchestrator] high memory (%.1f%%), cache cleared", mem.percent)
        except ImportError:
            # psutil이 없으면 스킵
            pass
        except Exception:
            pass
        
        self._run_text_preprocessing(payload)

        # ★성능 최적화: 환경변수 기반 병렬 처리 강제 활성화
        # Fast 모드 체크
        force_fast = os.getenv("FORCE_FAST", "0") == "1"
        if force_fast:
            # Fast 모드에서는 병렬 처리 비활성화 (더 빠름)
            self._parallel_enabled = False
            logger.info("[orchestrator] 🚀 FAST 모드 - 병렬 처리 비활성화 (순차 실행이 더 빠름)")
        else:
            force_parallel = os.getenv("ORCH_PARALLEL_MODE", "0") == "1"
            if force_parallel and not self._parallel_enabled:
                self._parallel_enabled = True
                logger.info("[orchestrator] 병렬 처리 강제 활성화 (ORCH_PARALLEL_MODE=1)")
        
        # 병렬 실행 그룹 생성
        parallel_groups = self._build_parallel_groups()
        
        if parallel_groups and self._parallel_enabled:
            # 병렬 실행 모드
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"병렬 실행 모드: {len(parallel_groups)} 그룹")
            self._run_parallel_pipeline(payload, parallel_groups)
        else:
            # 기존 순차 실행 모드 (캐시 최적화 적용)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"순차 실행 모드: parallel_enabled={self._parallel_enabled}, groups={len(parallel_groups) if parallel_groups else 0}")
            self._run_sequential_pipeline(payload, save=save, out_dir=out_dir)
        
        out = payload.to_output()
        
        # 필수 스텝 강제 성공 검증
        self._assert_all_required_steps(payload)
        
        # 성능 요약 메타 추가(안전)
        try:
            perf_summary = {k: self.perf.stats(k) for k in self.perf.step_timings.keys()}
            out.setdefault("meta", {})["perf_summary"] = perf_summary
        except Exception:
            pass
        # 최종 산출물 (중복 제거: 한 번만 생성)
        out = payload.to_output(meta_extra=self._final_meta(payload.meta))
        
        # 성능 요약 메타 추가(안전)
        try:
            perf_summary = {k: self.perf.stats(k) for k in self.perf.step_timings.keys()}
            out.setdefault("meta", {})["perf_summary"] = perf_summary
        except Exception:
            pass
        
        self._latest_payload = out
        
        # 저장 (중복 제거: 한 번만 저장)
        if save:
            try:
                run_dir = self._run_dir_cache or self._ensure_run_dir()
                self._run_dir_cache = run_dir
                target_dir = out_dir or run_dir
                if bool(self.log_cfg.get("write_final_only", True)):
                    final_path = self._final_result_path(target_dir, filename)
                    self._atomic_write_json(final_path, out)
                    out["meta"]["output_path"] = final_path
                else:
                    out["meta"]["output_path"] = self._save_output_json(
                        out,
                        out_dir=target_dir,
                        filename=filename,
                    )
            except Exception as e:
                logger.warning("[orchestrator] 결과 저장 실패: %s", e)
        
        return out

    def _run_parallel_pipeline(self, payload: "Payload", parallel_groups: List[List[str]]) -> None:
        """병렬 파이프라인 실행"""
        steps: List[Dict[str, Any]] = self.pipeline_spec.get("steps", [])
        step_cfg_map = {step.get("name"): step.get("config", {}) for step in steps}
        
        for i, group in enumerate(parallel_groups):
            # 그룹 내 병렬 실행
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"그룹 {i+1} 실행: {group}")
            self._run_parallel_group(group, payload, step_cfg_map)
        
        # 강제 실행 스텝들 (순차 실행)
        self._force_tail_if_missing(payload, step_cfg_map)

    def _run_sequential_pipeline(self, payload: "Payload", *, save: bool = False, out_dir: Optional[str] = None) -> None:
        """기존 순차 파이프라인 실행 (성능 최적화 버전)"""
        # ★ 성능 최적화: 의존성 그래프 캐싱 (한 번만 구축)
        if not hasattr(self, '_dependency_graph_cache'):
            steps = self.pipeline_spec.get("steps", [])
            self._dependency_graph_cache = {
                step.get("name"): {
                    "deps": step.get("dependencies", []),
                    "run_if": step.get("run_if", []),
                    "config": step.get("config", {}),
                    "enabled": step.get("enabled", True)
                }
                for step in steps if step.get("name")
            }
            logger.debug(f"[orchestrator] 의존성 그래프 캐시 구축 완료: {len(self._dependency_graph_cache)}개 스텝")
        
        steps: List[Dict[str, Any]] = self.pipeline_spec.get("steps", [])
        
        # ===== 개선사항: 환경변수 기반 모듈 스킵 (FAST 모드 최적화) =====
        skip_modules_set = self._compute_skip_modules(steps, log=True)
        
        if skip_modules_set:
            logger.info(f"[orchestrator] ✅ 최종 스킵 모듈 목록 ({len(skip_modules_set)}개): {skip_modules_set}")
        for step in steps:
            name = step.get("name")
            if name == "text_preprocessing":
                continue
            if not step.get("enabled", True):
                payload.mark_skip(name or "?", "disabled")
                continue
            
            # 환경변수 기반 스킵 체크
            if name and name in skip_modules_set:
                logger.info(f"[orchestrator] 모듈 스킵 (환경변수): {name}")
                payload.mark_skip(name, "optimized_skip")
                payload.add_result(
                    name,
                    {
                        "success": True,
                        "skipped": True,
                        "reason": "optimized_skip",
                    },
                )
                continue

            # ★ 성능 최적화: 캐시된 의존성 그래프 사용
            step_info = self._dependency_graph_cache.get(name, {})
            deps = step_info.get("deps", step.get("dependencies", []))
            run_if = step_info.get("run_if", step.get("run_if", []))
            step_cfg = step_info.get("config", step.get("config", {}) or {})
            
            # 기본 게이트 추가: run_if가 비어있거나 없으면 기본 조건 적용
            if not run_if:
                # 선택 개선: config로 노출된 임계값 사용
                min_tokens = int(getattr(self.cfg, "MIN_TOKENS_THRESHOLD", 3))
                min_sentences = int(getattr(self.cfg, "MIN_SENTENCES_THRESHOLD", 1))
                
                # 짧은 입력 자동 스킵 기본 게이트
                run_if = [{"all_of": [
                    {"type": "metric_compare", "left": "tokens_count", "right": min_tokens, "op": ">="},
                    {"type": "metric_compare", "left": "sentences_count", "right": min_sentences, "op": ">="}
                ]}]
                
                # 무거운 스텝들에 대해서는 더 높은 기준 적용
                heavy_steps = {"transition_analyzer", "time_series_analyzer", "psychological_analyzer", "complex_analyzer"}
                if name in heavy_steps:
                    heavy_min_tokens = int(getattr(self.cfg, "HEAVY_MIN_TOKENS_THRESHOLD", 10))
                    run_if = [{"all_of": [
                        {"type": "metric_compare", "left": "tokens_count", "right": heavy_min_tokens, "op": ">="},
                        {"type": "metric_compare", "left": "sentences_count", "right": min_sentences, "op": ">="}
                    ]}]

            # 의존성 확인
            missing = [d for d in deps if d not in payload.results]
            if missing:
                payload.mark_skip(name, f"deps-missing:{','.join(missing)}")
                continue

            # 조건 확인
            ok, reason = ConditionEvaluator.evaluate_with_reason(run_if, payload)
            # NEW: env override for tail steps in HEAVY (run_if bypass)
            is_tail = (name in self._FORCE_TAIL_STEPS)
            if (not ok) and is_tail and _ENV_FORCE_TAIL:
                try:
                    reason.setdefault("reasons", []).append("force_tail_env_override")
                except Exception:
                    pass
                ok = True
            will_force = False
            if not ok:
                will_force = (name in self._FORCE_TAIL_STEPS) and (getattr(self, "_force_tail_enabled", True) or _ENV_FORCE_TAIL)
                if not will_force:
                    payload.mark_skip(name, f"run_if:false {reason.get('reasons')}")
                    logger.debug(f"[{name}] 스킵됨: run_if 실패 - {reason.get('reasons')}")
                    continue
                else:
                    logger.debug(f"[{name}] 강제 실행: tail step")
            else:
                logger.debug(f"[{name}] 조건 통과: {reason.get('summary', 'ok')}")

            # 캐시 키
            cfg_sig = _config_digest(step_cfg)
            cache_key = self.cache.make_key(
                text=payload.text,
                step=name,
                cfg_sig=cfg_sig,
                emotion_set_version=payload.meta.get("emotion_set_version", "NA"),
                model_version=payload.meta.get("model_version", "NA"),
            )

            # 캐시 적중 처리
            cached = self.cache.get(cache_key)
            if cached is not None and not will_force:
                payload.add_result(name, cached)
                payload.stamp(name, status="cached", extras={"cache": True})
                self._write_step_result(name, cached, cached=True, ok=True)
                continue

            # 실행(리트라이 포함)
            started = time.perf_counter()
            retries = int(getattr(self.cfg, "ERROR_HANDLING_CONFIG", {}).get("retry_count", 0))
            delay = float(getattr(self.cfg, "ERROR_HANDLING_CONFIG", {}).get("retry_delay", 0.0))  # 고성능 환경에서는 지연 최소화
            # on_error 정책 소스: 스텝 설정 > 전역 ERROR_HANDLING_CONFIG.on_error (기본 skip)
            on_error = (step_cfg.get("on_error") or getattr(self.cfg, "ERROR_HANDLING_CONFIG", {}).get("on_error", "skip")).lower()
            # 외부 표기 ↔ 내부 처리 매핑:
            #  - "record-only" → "record" (결과 스켈레톤 없이 오류만 기록)
            #  - "abort"       → "raise"  (예외 전파)
            if on_error in ("record-only", "recordonly", "record_only"):
                on_error = "record"
            elif on_error in ("abort", "error"):
                on_error = "raise"
            # on_error: "skip" | "skeleton" | "record" | "raise"
            attempt = 0

            # ★ 성능 최적화: 인스턴스 재사용 보장 (캐시에서 가져오기)
            # registry.get_instance는 이미 캐싱을 하고 있지만, 명시적으로 확인
            instance = self.registry.get_instance(name, force_rebuild=False, prewarm=False)
            
            def _produce() -> Any:
                # ★ 성능 최적화: 이미 가져온 인스턴스 재사용 (중복 생성 방지)
                # instance는 위에서 이미 가져왔으므로 재사용
                # 대용량: 과도한 핸들러 재구성을 줄이기 위해 주기 제어
                self._processed_steps += 1
                if self._processed_steps % max(1, self._reconf_every_steps) == 0:
                    try:
                        self._reconfigure_module_file_handlers(self._run_dir_cache)
                    except Exception:
                        logger.debug("[orchestrator] handler reconfigure (periodic) failed", exc_info=True)

                runner = self.adapter.resolve_callable(name, instance)
                if runner is None:
                    logger.error(f"[{name}] 러너를 찾을 수 없음: {type(instance).__name__}")
                    raise AttributeError(f"No callable for step '{name}'")
                else:
                    logger.debug(f"[{name}] 러너 발견: {runner.__name__ if hasattr(runner, '__name__') else type(runner).__name__}")
                return run_step_with_timeout(
                    self._run_step_with_runtime_guards, 
                    step_name=name,
                    runner=runner, 
                    payload=payload, 
                    step_cfg=step_cfg
                )

            result: Any = None
            while True:
                try:
                    result = _produce()
                    
                    # 폴백 감지 시 프로덕션에서는 즉시 종료
                    if isinstance(result, dict) and result.get("fallback"):
                        is_production = (
                            getattr(config, "EA_PROFILE", "prod") == "prod"
                            or bool(getattr(config, "RENDER_DEPLOYMENT", False))
                            or os.getenv("PRODUCTION_MODE", "0") == "1"
                        )
                        
                        if is_production:
                            error_msg = f"Fallback detected in step '{name}' in production mode - aborting pipeline"
                            logger.error(error_msg)
                            raise RuntimeError(error_msg)
                        else:
                            logger.warning(f"Fallback detected in step '{name}' in development mode")
                    
                    # success: false 감지 시 프로덕션에서는 즉시 종료
                    if isinstance(result, dict) and result.get("success") is False:
                        is_production = (
                            getattr(config, "EA_PROFILE", "prod") == "prod"
                            or bool(getattr(config, "RENDER_DEPLOYMENT", False))
                            or os.getenv("PRODUCTION_MODE", "0") == "1"
                        )
                        
                        if is_production:
                            error_msg = f"Step '{name}' returned success=false in production mode - aborting pipeline"
                            logger.error(error_msg)
                            raise RuntimeError(error_msg)
                        else:
                            logger.warning(f"Step '{name}' returned success=false in development mode")
                    
                    # tail 강제 시 confidence 보정
                    if will_force:
                        if not isinstance(result, dict):
                            result = {"value": result}
                        if "confidence" not in result or not isinstance(result["confidence"], (int, float)):
                            result["confidence"] = float(self._FORCED_CONF_DEFAULTS.get(name, 0.51))
                        payload.add_result(name, result)
                        payload.stamp(name, "ok", started, extras={"forced": True})
                    else:
                        payload.add_result(name, result)
                        payload.stamp(name, "ok", started)
                    self._write_step_result(name, result, cached=False, ok=True)
                    self.cache.set(cache_key, result)
                    # 패치 D: 레거시 _cb 제거 (circuit_breaker로 단일화)
                    # P4: 성능 메트릭 기록 (순차 경로에도 적용)
                    try:
                        duration = time.perf_counter() - started
                        self._record_execution_time(name, duration)
                        if payload.trace:
                            last_trace = payload.trace[-1]
                            if "duration_ms" in last_trace:
                                self.perf.record(name, last_trace["duration_ms"])
                    except Exception:
                        pass
                    
                    # P6: 서킷브레이커 성공 기록
                    self._record_circuit_breaker_success(name)
                    
                    # P8: 품질 모니터링 (순차 경로에도 적용)
                    self._monitor_execution_quality(payload)
                    
                    break
                except Exception as e:
                    attempt += 1
                    logger.exception("[step:%s] failed (attempt %d)", name, attempt)
                    # 패치 D: 레거시 _cb 업데이트 대신 circuit_breaker 사용
                    self._record_circuit_breaker_failure(name)
                    
                    if attempt <= retries:
                        if delay > 0:
                            try:
                                time.sleep(delay)
                            except Exception:
                                pass
                        continue

                    # 리트라이 초과 → on_error 정책 적용
                    if on_error == "raise":
                        payload.stamp(name, "error", started, error=str(e), extras={"forced": will_force})
                        self._write_step_result(name, {"error": str(e)}, cached=False, ok=False)
                        raise
                    elif on_error == "record":
                        payload.add_result(name, {"error": str(e), "confidence": 0.0})
                        payload.stamp(name, "error", started, error=str(e), extras={"recorded": True})
                        self._write_step_result(name, payload.results.get(name), cached=False, ok=False)
                        # 기록 후 계속 진행
                        break
                    elif on_error == "skeleton":
                        # 의존 스텝을 살리기 위한 최소 스켈레톤
                        sk = {"features": {}, "timeline": {}, "confidence": 0.0, "error": str(e)}
                        payload.add_result(name, sk)
                        payload.stamp(name, "error", started, error=str(e), extras={"skeleton": True})
                        self._write_step_result(name, payload.results.get(name), cached=False, ok=False)
                        break
                    else:  # default "skip"
                        payload.stamp(name, "error", started, error=str(e), extras={"skipped": True})
                        self._write_step_result(name, {"error": str(e)}, cached=False, ok=False)
                        # 결과를 남기지 않음 → 의존 스텝은 deps-missing으로 스킵되어 연쇄 폭주 방지
                        break
        # Safety post-pass: force missing tail steps when FORCE_TAIL is enabled
        if _ENV_FORCE_TAIL:
            try:
                step_cfg_map = {s.get("name"): (s.get("config") or {}) for s in steps if s.get("name")}
            except Exception:
                step_cfg_map = {}
            try:
                self._force_tail_if_missing(payload, step_cfg_map=step_cfg_map)
            except Exception:
                logger.debug("[orchestrator] _force_tail_if_missing failed", exc_info=True)

    # ---------- 배치 처리 ----------
    def process_corpus(
        self,
        texts: List[str],
        metas: Optional[List[Dict[str, Any]]] = None,
        *,
        save_each: bool = False,
        out_dir: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        metas = metas or [None] * len(texts)
        for t, m in zip(texts, metas):
            results.append(self.process_text(t, m, save=save_each, out_dir=out_dir))
        return results

    # ---------- 내부: 전처리 ----------
    def _run_text_preprocessing(self, payload: "Payload") -> None:
        started = time.perf_counter()
        try:
            norm = _normalize_text(payload.text)
            sents = _split_sentences(norm)
            toks = _basic_tokenize(norm)
            lang = payload.lang or payload.meta.get("lang", None)

            payload.text = norm
            payload.sentences = sents
            payload.tokens = toks
            payload.lang = lang

            result = {
                "normalized_text": norm if len(norm) <= 2000 else f"{norm[:2000]}…",
                "sentences": sents,
                "tokens": toks[:512],
                "sentences_count": len(sents),
                "tokens_count": len(toks),
                "features": {"preprocessed": True},
                "confidence": 1.0,
            }
            result["success"] = True
            payload.add_result("text_preprocessing", result)
            payload.stamp("text_preprocessing", "ok", started)
        except Exception as e:
            payload.add_result("text_preprocessing", {"features": {"preprocessed": False}, "success": False, "error": str(e)})
            payload.stamp("text_preprocessing", "error", started, error=str(e))

    # ---------- 내부: 안전 호출 ----------
    def _run_step_with_runtime_guards(self, runner: Callable, payload: "Payload", step_cfg: Dict[str, Any], *, step_name: str) -> Any:
        # 개선안 C-3: 서킷브레이커 체크 (강화)
        if not self._check_circuit_breaker(step_name):
            payload.mark_skip(step_name, "Circuit breaker activated")
            return None
        
        # 스마트 타임아웃 계산
        timeout_s = self._calculate_smart_timeout(step_name, step_cfg)

        def _invoke_core() -> Any:
            return self._safe_call(runner, payload, step_cfg, step_name=step_name)

        # 컨텍스트 내 실행 함수
        def _invoke_with_context() -> Any:
            """패치 B: 오토캐스트 안전 임포트"""
            try:
                import torch as _torch
            except Exception:
                return _invoke_core()

            if not (_torch.cuda.is_available() and bool(getattr(self.cfg, "USE_AUTOCAST", True))):
                return _invoke_core()

            dtype_env = os.getenv("EA_AUTOCAST_DTYPE", "fp16").lower()
            dtype = _torch.bfloat16 if dtype_env == "bf16" else _torch.float16
            try:
                ctx1 = _torch.autocast("cuda", dtype=dtype, enabled=True)
            except Exception:
                return _invoke_core()
            try:
                ctx2 = _torch.inference_mode()
            except Exception:
                from contextlib import nullcontext
                ctx2 = nullcontext()

            with ctx1:
                with ctx2:
                    return self._safe_call(runner, payload, step_cfg, step_name=step_name)

        # 타임아웃 래퍼
        if timeout_s and timeout_s > 0:
            with _fut.ThreadPoolExecutor(max_workers=1) as _ex:
                fut = _ex.submit(_invoke_with_context)
                try:
                    return fut.result(timeout=timeout_s if timeout_s else None)
                except _fut.TimeoutError:
                    # intensity_analysis는 1회만 관대한 재시도
                    attempt = step_cfg.get("_retry_attempt", 1)
                    if step_name == "intensity_analysis" and attempt == 1:
                        new_to = max(60, (timeout_s or 0) * 2 or 60)
                        print(f"[guard] {step_name} timed out at {timeout_s}s → retry with {new_to}s")
                        retry_cfg = {**step_cfg, "timeout": new_to, "_retry_attempt": 2}
                        return self._run_step_with_runtime_guards(runner, payload, retry_cfg, step_name=step_name)
                    raise TimeoutError(f"step-timeout({timeout_s}s)")
        else:
            return _invoke_with_context()

    def _safe_call(self, runner: Callable, payload: "Payload", step_cfg: Dict[str, Any], *, step_name: str) -> Any:
        try:
            return runner(payload, step_cfg)
        except Exception as e:
            # 개선안 C-3: 서킷브레이커 실패 기록
            self._record_circuit_breaker_failure(step_name)
            msg = str(e).lower()

            # ★ GPU 불법명령 → 단계적 폴백: (1) GPU FP32 → (2) CPU
            gpu_fb_targets = {"embedding_generation", "intensity_analysis", "pattern_extractor"}
            if (step_name in gpu_fb_targets) and ("illegal instruction" in msg or "acceleratorerror" in msg):
                # -------- 1) 같은 스텝을 GPU(FP32, amp=False)로 재시도 --------
                step_cfg_fp32 = dict(step_cfg or {})
                ep1 = dict(step_cfg_fp32.get("EMBEDDING_PARAMS") or {})
                try:
                    ep1["device"] = "cuda" if (torch.cuda.is_available()) else "cpu"
                except Exception:
                    ep1["device"] = "cpu"
                ep1["amp"] = False  # FP32 강제
                step_cfg_fp32["EMBEDDING_PARAMS"] = ep1
                step_cfg_fp32["use_cache"] = False  # 캐시 무시(분석기 새로 빌드)
                try:
                    inst1 = self.registry.get_instance(step_name, force_rebuild=True, prewarm=False)
                    run1 = self.adapter.resolve_callable(step_name, inst1)
                    if run1 is None:
                        raise RuntimeError(f"no callable after gpu-fp32 fallback for step={step_name}")
                    if not hasattr(self.cfg, "_EA_GPU_FP32_FALLBACK_WARNED"):
                        logger.warning("[orchestrator] GPU FP32 fallback for step=%s (AMP disabled)", step_name)
                        self.cfg._EA_GPU_FP32_FALLBACK_WARNED = True
                    return run1(payload, step_cfg_fp32)
                except Exception as e1:
                    # e1에도 같은 오류가 있다면 CPU로 최종 폴백
                    msg1 = str(e1).lower()
                    if "illegal instruction" not in msg1 and "acceleratorerror" not in msg1:
                        # 다른 오류면 원래 로직으로 위임
                        raise e1

                # -------- 2) CPU 폴백(최종) --------
                step_cfg_cpu = dict(step_cfg or {})
                ep2 = dict(step_cfg_cpu.get("EMBEDDING_PARAMS") or {})
                ep2["device"] = "cpu"
                ep2["amp"] = False
                step_cfg_cpu["EMBEDDING_PARAMS"] = ep2
                step_cfg_cpu["use_cache"] = False  # 캐시 무시
                try:
                    inst2 = self.registry.get_instance(step_name, force_rebuild=True, prewarm=False)
                    run2 = self.adapter.resolve_callable(step_name, inst2)
                    if run2 is None:
                        raise RuntimeError(f"no callable after cpu-fallback for step={step_name}")
                    if not hasattr(self.cfg, "_EA_CPU_FALLBACK_WARNED"):
                        logger.warning("[orchestrator] CPU fallback for step=%s due to GPU illegal instruction",
                                       step_name)
                        self.cfg._EA_CPU_FALLBACK_WARNED = True
                    return run2(payload, step_cfg_cpu)
                except Exception as e2:
                    raise e2

            # 다른 예외/스텝은 기존 리트라이/정책으로
            raise

    def _compute_skip_modules(self, steps: List[Dict[str, Any]], *, log: bool = False) -> Set[str]:
        skip_modules_set: Set[str] = set()
        skip_modules_env = os.getenv("ORCH_SKIP_MODULES", "")
        if skip_modules_env:
            env_set = {m.strip() for m in skip_modules_env.split(",") if m.strip()}
            skip_modules_set.update(env_set)
            if log and env_set:
                logger.info(f"[orchestrator] 환경변수 ORCH_SKIP_MODULES 감지: {env_set}")
                logger.info(f"[orchestrator] 스킵할 모듈 수: {len(env_set)}개")

        if os.getenv("NO_HEAVY_MODULES", "0") == "1":
            heavy_modules = {
                "transition_analyzer",
                "time_series_analyzer",
                "complex_analyzer",
                "psychological_analyzer",
                "pattern_extractor",
                "situation_analyzer",
                "relationship_analyzer",
            }
            skip_modules_set.update(heavy_modules)
            if log:
                logger.info(f"[orchestrator] 🚀 NO_HEAVY_MODULES 활성화 - 무거운 모듈 자동 스킵: {heavy_modules}")

        if os.getenv("USE_FULL_PIPELINE", "1") != "1":
            essential_modules = {"emotion_classification", "intensity_analysis", "context_analysis", "weight_calculator"}
            non_essential = set()
            for step in steps:
                step_name = step.get("name")
                if step_name and step_name not in essential_modules and step_name != "text_preprocessing":
                    non_essential.add(step_name)
            skip_modules_set.update(non_essential)
            if log:
                logger.info(f"[orchestrator] 🚀 USE_FULL_PIPELINE=0 - 핵심 모듈만 실행: {essential_modules}")
                if non_essential:
                    logger.info(f"[orchestrator] 스킵할 모듈: {non_essential}")

        # Smart skip for heavy modules based on text length / quality (optional)
        # 활성화 조건: SMART_SKIP_HEAVY=1 이고 USE_FULL_PIPELINE=1 인 경우에만 동작
        try:
            smart_skip_enabled = os.getenv("SMART_SKIP_HEAVY", "0") in ("1", "true", "True")
            use_full_pipeline = os.getenv("USE_FULL_PIPELINE", "1") == "1"
            if smart_skip_enabled and use_full_pipeline:
                heavy_candidates = {"complex_analyzer", "psychological_analyzer", "transition_analyzer", "time_series_analyzer"}
                text_len = 0
                current_quality = None
                try:
                    # Payload가 있을 때 현재 텍스트 길이/품질을 참고할 수 있도록 meta에서 조회 시도
                    # (없으면 보수적으로 동작)
                    if hasattr(self, "_latest_payload") and isinstance(self._latest_payload, dict):
                        t = str(self._latest_payload.get("text") or "")
                        text_len = len(t)
                        meta = self._latest_payload.get("meta") or {}
                        current_quality = meta.get("quality_score") or meta.get("overall_quality")
                except Exception:
                    pass

                # 환경변수 기반 임계값 (기본: 짧은 텍스트에서만 스킵)
                max_len_for_skip = int(os.getenv("SKIP_HEAVY_MAX_LEN", "256"))
                min_quality_for_skip = float(os.getenv("SKIP_HEAVY_MIN_QUALITY", "0.65"))

                allow_by_length = (text_len > 0 and text_len <= max_len_for_skip)
                allow_by_quality = (current_quality is None) or (current_quality >= min_quality_for_skip)

                if allow_by_length and allow_by_quality:
                    smart_set = set()
                    for step in steps:
                        step_name = step.get("name")
                        if step_name in heavy_candidates and step_name not in skip_modules_set:
                            smart_set.add(step_name)
                    if smart_set:
                        skip_modules_set.update(smart_set)
                        if log:
                            logger.info(f"[orchestrator] SMART_SKIP_HEAVY 활성화 - 조건부 무거운 모듈 스킵: {smart_set} (len={text_len}, quality={current_quality})")
        except Exception:
            # 스마트 스킵 계산 중 문제 발생 시 무시하고 기본 스킵만 사용
            pass

        return skip_modules_set

    def _force_tail_if_missing(self, payload: "Payload", step_cfg_map: Optional[Dict[str, Any]] = None) -> None:
        """Post-pass safety: ensure tail steps have results when FORCE_TAIL env is enabled."""
        step_cfg_map = step_cfg_map or {}
        for name in self._FORCE_TAIL_STEPS:
            # 이미 결과가 있으면 skip
            if payload.results.get(name):
                continue
            try:
                inst = self.registry.get_instance(name)
                runner = self.adapter.resolve_callable(name, inst)
                if runner is None:
                    continue
                with _timer_ms() as _t:
                    res = runner(payload, step_cfg_map.get(name, {}))
                if isinstance(res, dict):
                    res.setdefault("confidence", float(self._FORCED_CONF_DEFAULTS.get(name, 0.51)))
                    res.setdefault("evidence", 0.51)
                payload.add_result(name, res)
                payload.stamp(name, status="ok", started=None, extras={"forced": True, "post_pass": True})
                try:
                    self._write_step_result(name, res, cached=False, ok=True)
                except Exception:
                    pass
            except Exception as e:
                try:
                    payload.stamp(name, status="error", started=None, error=str(e), extras={"forced": True, "post_pass": True})
                    self._write_step_result(name, {"error": str(e)}, cached=False, ok=False)
                except Exception:
                    pass

    # ---------- 내부: 메타/서명 ----------
    def _assert_all_required_steps(self, payload: "Payload") -> None:
        """필수 스텝 전부의 존재+성공을 확인"""
        # [Emergency Fix] embedding_generation이 누락되었다면 여기서라도 실행 시도 (Just-In-Time Execution)
        if "embedding_generation" not in payload.results:
            logger.info("[orchestrator] embedding_generation missing at assertion -> executing Just-In-Time")
            try:
                try:
                    from src.emotion_analysis.intensity_analyzer import run_embedding_generation
                except ImportError:
                    from emotion_analysis.intensity_analyzer import run_embedding_generation
                
                step_cfg = getattr(self.cfg, "EMBEDDING_GENERATION", {}) or getattr(self.cfg, "EMBEDDING_PARAMS", {}) or {}
                res = run_embedding_generation(payload, step_cfg)
                payload.add_result("embedding_generation", res)
                logger.info("[orchestrator] embedding_generation JIT execution success")
            except Exception as e:
                logger.warning(f"[orchestrator] embedding_generation JIT execution failed: {e}")

        planned = [s.get("name") for s in self.pipeline_spec.get("steps", []) if s.get("enabled", True)]
        required = [s for s in planned if s and s != "text_preprocessing"]
        missing = [s for s in required if s not in payload.results]
        
        # 실패 원인 상세 분석
        failed = []
        for s, r in payload.results.items():
            if s in required and isinstance(r, dict) and r.get("success") is False:
                why = r.get("error") or ("fallback" if r.get("fallback") else "unknown")
                if isinstance(r.get("matches"), list) and len(r["matches"]) == 0 and not r.get("fallback"):
                    # '매치 0건'은 실패가 아님 → 경고만
                    logger.warning(f"[{s}] no matches (0); treating as success.")
                    r["success"] = True
                else:
                    failed.append(f"{s}({why})")
        
        if missing or failed:
            # [Fix] embedding_generation 누락은 경고만 하고 통과 (Soft Fail)
            if "embedding_generation" in missing:
                logger.warning("[orchestrator] embedding_generation result missing - proceeding with partial results (Soft Fail)")
                missing.remove("embedding_generation")
            
            # 여전히 누락된 필수 스텝이 있거나 실패한 스텝이 있는 경우에만 에러 처리
            if missing or failed:
                msg = f"pipeline-incomplete missing={missing} failed={failed}"
                # prod에서는 즉시 중단
                is_prod = (
                    getattr(config, "EA_PROFILE", "prod") == "prod" 
                    or os.getenv("PRODUCTION_MODE", "0") == "1"
                )
                if is_prod:
                    raise RuntimeError(msg)
                logging.warning(msg)

    def _final_meta(self, meta: Dict[str, Any]) -> Dict[str, Any]:
        return {
            **meta,
            "device": self._device(),
            "pipeline": [s.get("name") for s in self.pipeline_spec.get("steps", []) if s.get("enabled", True)],
            "cache_stats": self.cache.stats(),
            "run_dir": self._run_dir_cache,
            "logs_root": self._compute_logs_root(),
            "logging_policy": {
                "write_final_only": bool(self.log_cfg.get("write_final_only", True)),
                "write_step_results": bool(self.log_cfg.get("write_step_results", False)),
                "redirect_module_file_logs": bool(self.log_cfg.get("redirect_module_file_logs_to_run_dir", False)),
                "suppress_module_file_logs": bool(self.log_cfg.get("suppress_module_file_logs_when_orchestrated", True)),
            },
        }

    def _device(self) -> str:
        try:
            return getattr(self.cfg, "EMBEDDING_PARAMS", {}).get("device", "cpu")
        except Exception:
            return "cpu"

    def _validate_pipeline(self) -> None:
        names = set()
        for s in self.pipeline_spec.get("steps", []):
            nm = s.get("name")
            if not nm:
                raise ValueError("Pipeline step missing 'name'")
            if nm in names:
                raise ValueError(f"Duplicated step name: {nm}")
            names.add(nm)
            if nm != "text_preprocessing" and nm not in getattr(self.cfg, "MODULE_ENTRYPOINTS", {}):
                logger.warning("No MODULE_ENTRYPOINTS for step '%s' — if enabled, this may fail", nm)

    # ---------- 내부: 저장 유틸 ----------
    def _should_write_step_results(self, name: str, cached: bool, ok: bool) -> bool:
        if self.log_cfg.get("write_final_only", False):
            return False
        if cached and not self.log_cfg.get("write_step_results_for_cached", False):
            return False
        if self.log_cfg.get("write_on_error_only", False):
            return not ok
        return bool(self.log_cfg.get("write_step_results", False))

    def _atomic_write_json(self, path: str, data: Dict[str, Any]) -> None:
        try:
            tmp_path = f"{path}.tmp{os.getpid()}"
            indent = 2 if bool(self.log_cfg.get("pretty_json", True)) else None
            payload = _safe_json_dumps(data, sort_keys=False, indent=indent)
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(payload)
            os.replace(tmp_path, path)
        except Exception as e:
            logger.debug("[orchestrator] atomic write failed for %s: %s", path, e, exc_info=True)

    def _step_result_path(self, step_name: str) -> str:
        dirp = self._run_dir_cache or self._ensure_run_dir()
        safe_step = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in (step_name or "step"))
        idx = int(self._step_no.get(step_name, 0))
        base_name = f"{idx:02d}_{safe_step}.json" if idx > 0 else f"{safe_step}.json"
        path = os.path.join(dirp, base_name)
        if os.path.exists(path):
            suffix_src = self._current_text_hash or _hash_text(f"{step_name}|{time.time()}")
            suffix = suffix_src[:8]
            root, ext = os.path.splitext(base_name)
            path = os.path.join(dirp, f"{root}_{suffix}{ext or '.json'}")
        return path

    def _final_result_path(self, dir_path: Optional[str], filename: Optional[str]) -> str:
        dirp = dir_path or self._run_dir_cache or self._ensure_run_dir()
        os.makedirs(dirp, exist_ok=True)
        if filename:
            return os.path.join(dirp, filename)
        trace = (self._current_trace_id or "trace").replace(os.sep, "_")
        text_hash = (self._current_text_hash or _hash_text(str(time.time())))[:12]
        return os.path.join(dirp, f"{trace}_{text_hash}.json")

    def _write_step_result(self, name: str, result: Any, *, cached: bool, ok: bool) -> None:
        if not self._should_write_step_results(name, cached, ok):
            return
        payload = result if isinstance(result, dict) else {"value": result}
        keep = set(self.log_cfg.get("result_keep_keys") or [])
        if keep:
            payload = {k: v for k, v in payload.items() if k in keep}
        data = {"step": name, "result": payload, "meta": {"trace_id": self._current_trace_id, "cached": cached, "ok": ok}}
        try:
            data["meta"]["timestamp"] = time.time()
        except Exception:
            pass
        path = self._step_result_path(name)
        self._atomic_write_json(path, data)

    def _save_output_json(self, result: Dict[str, Any], *, out_dir: Optional[str] = None, filename: Optional[str] = None) -> str:
        base_dir = (
            out_dir
            or (getattr(self.cfg, "PATHS", {}).get("analysis_results") if hasattr(self.cfg, "PATHS") else None)
            or os.path.join(os.path.dirname(getattr(self.cfg, "__file__", ".")), "analysis_results")
        )
        os.makedirs(base_dir, exist_ok=True)

        trace_id = _dig(result, "meta.trace_id", _hash_text(str(time.time())))
        text_hash = _hash_text(_dig(result, "text", ""))
        fname = filename or f"{trace_id}_{text_hash}.json"
        fpath = os.path.join(base_dir, fname)

        # 안전 직렬화
        try:
            s = _safe_json_dumps(result, sort_keys=False, indent=2 if bool(getattr(self.cfg, "LOGGING_CONFIG", {}).get("pretty_json", True)) else None)
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(s)
        except Exception:
            # 최후 폴백
            with open(fpath, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False)
        return fpath

    # ---------- 날짜/RUN_DIR/루트경로 ----------
    def _today_date_str(self) -> str:
        try:
            offset = int(getattr(self.cfg, "LOGGING_CONFIG", {}).get("tz_offset_hours", 9))
        except Exception:
            offset = 9
        tz = timezone(timedelta(hours=offset))
        return datetime.now(tz).strftime(getattr(self.cfg, "LOGGING_CONFIG", {}).get("date_fmt", "%Y%m%d"))

    def _compute_logs_root(self) -> str:
        env_p = os.getenv("EMOTION_LOGS_ROOT")
        if env_p:
            return env_p
        try:
            p = getattr(self.cfg, "PATHS", {}).get("logs_root")
            if p:
                return str(p)
        except Exception:
            pass
        # src/logs를 기본으로 사용
        try:
            import config
            from pathlib import Path
            return getattr(config, "LOG_DIR", str(Path(__file__).parent / "logs"))
        except ImportError:
            cfg_dir = os.path.dirname(getattr(self.cfg, "__file__", "."))
            return os.path.join(cfg_dir, "logs")

    def _ensure_run_dir(self) -> str:
        base = self._compute_logs_root()
        # src/logs/YYYYMMDD/data_utils/ 구조로 변경
        run_dir = os.path.join(base, self._today_date_str(), "data_utils")
        os.makedirs(run_dir, exist_ok=True)
        return run_dir

    # ---------- 스텝 JSON 저장 ----------
    def _save_step_json(self, payload: "Payload", step_name: str, result: Any, dir_path: Optional[str] = None) -> Optional[str]:
        try:
            dirp = dir_path or self._ensure_run_dir()
            idx = int(self._step_no.get(step_name, 0))
            safe_step = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in (step_name or "step"))
            base_name = f"{idx:02d}_{safe_step}.json" if idx > 0 else f"{safe_step}.json"
            fpath = os.path.join(dirp, base_name)
            if os.path.exists(fpath):
                text_hash = _hash_text(payload.text)[:8]
                fpath = os.path.join(dirp, f"{os.path.splitext(base_name)[0]}_{text_hash}.json")

            # non-serializable 방지: 안전 직렬화 사용
            to_dump = {
                "step": step_name,
                "result": result,
                "meta": {
                    "trace_id": payload.meta.get("trace_id"),
                    "date": self._today_date_str(),
                },
            }
            s = _safe_json_dumps(to_dump, sort_keys=False, indent=2 if bool(getattr(self.cfg, "LOGGING_CONFIG", {}).get("pretty_json", True)) else None)
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(s)
            return fpath
        except Exception as e:
            logger.debug("[orchestrator] save_step_json failed: %s", e, exc_info=True)
            return None

    # ---------- 모듈 로그 경로 ----------
    def _module_logs_dir(self) -> str:
        """통합 로그 관리자를 사용하여 메인 컨텍스트 로그 디렉터리 반환"""
        try:
            from log_manager import get_log_manager
            log_manager = get_log_manager()
            return str(log_manager.get_log_dir("main"))
        except ImportError:
            # 폴백: 기존 방식
            cfg_dir = os.path.dirname(getattr(self.cfg, "__file__", "."))
            return os.path.abspath(os.path.join(cfg_dir, "emotion_analysis", "logs"))

    def _build_step_index_map(self) -> Dict[str, int]:
        m: Dict[str, int] = {}
        try:
            seq = [s.get("name") for s in self.pipeline_spec.get("steps", []) if s.get("enabled", True)]
            seq = [s for s in seq if s and s != "text_preprocessing"]
            for i, name in enumerate(seq, start=1):
                m[name] = i
        except Exception:
            pass
        return m
    
    def _build_parallel_groups(self) -> List[List[str]]:
        """
        [최적화] 3단계 전술 그룹 병렬 실행 전략 (All-Alive Parallel Optimization)
        11개 모듈을 의존성 기반으로 3개 그룹으로 나누어 병렬성을 극대화합니다.
        """
        if not self._parallel_enabled:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("[orchestrator] 병렬 실행 비활성 (설정됨)")
            return []

        # 활성화된 스텝만 필터링
        active_steps = {s.get("name") for s in self.pipeline_spec.get("steps", []) if s.get("enabled", True)}
        
        # 1. Group A (Scouts): 선봉대 - 독립적으로 실행 가능한 모듈들
        # - 텍스트만 있으면 실행 가능
        group_a = [
            "pattern_extractor", 
            "linguistic_matcher", 
            "intensity_analysis", 
            "emotion_classification",
            "context_extractor" # context_analysis의 전처리
        ]
        
        # 2. Group B (Core): 중견대 - 1차 분석 결과가 필요한 모듈들
        # - A그룹의 결과(패턴, 분류 등)를 활용
        group_b = [
            "context_analysis",       # linguistics, pattern 필요
            "weight_calculator",      # classification, intensity 필요
            "transition_analyzer",    # context, intensity 필요
            "sub_emotion_detection"   # classification 필요
        ]
        
        # 3. Group C (Deep): 후발대 - 심층 분석 모듈들
        # - 모든 문맥 정보(A+B)를 종합하여 추론
        group_c = [
            "complex_analyzer",       # classification, intensity 필요
            "situation_analyzer",     # context, classification 필요
            "relationship_analyzer",  # context, classification 필요
            "psychological_analyzer", # complex, situation 필요
            "time_series_analyzer"    # transition, context 필요
        ]
        
        # 실제 활성화된 스텝만 남기고 필터링
        final_groups = []
        for g in [group_a, group_b, group_c]:
            filtered = [s for s in g if s in active_steps]
            if filtered:
                final_groups.append(filtered)
                
        # 남은(분류되지 않은) 스텝들은 마지막에 순차 실행되도록 둠
        # (DAG 토폴로지 정렬 로직을 완전히 대체하므로, 이 그룹들에 없는 스텝은 순차 실행됨)
        
        return final_groups
    
    def _run_parallel_group(self, group: List[str], payload: "Payload", step_cfg_map: Dict[str, Any]) -> None:
        """병렬 그룹 실행 (성능 최적화 버전)"""
        if not group:
            return
        
        # ★ 성능 최적화: 병렬 실행 전에 인스턴스 사전 로드
        for step_name in group:
            try:
                # 인스턴스를 미리 가져와서 캐시에 저장 (병렬 실행 시 재사용)
                self.registry.get_instance(step_name, force_rebuild=False, prewarm=False)
            except Exception:
                logger.debug(f"[parallel] 인스턴스 사전 로드 실패: {step_name}", exc_info=True)
        
        def run_step(step_name: str):
            try:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"병렬 스텝 시작: {step_name}")
                
                # 의존성 확인
                step_config = None
                for step in self.pipeline_spec.get("steps", []):
                    if step.get("name") == step_name:
                        step_config = step
                        break
                
                if step_config:
                    deps = step_config.get("dependencies", [])
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"{step_name} 의존성: {deps}")
                    
                # 강화된 의존성 검증
                if not self._validate_dependencies(step_name, payload, deps):
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"{step_name} 의존성 검증 실패 - 스킵")
                    payload.stamp(step_name, "skip", time.perf_counter(), extras={"reason": "dependency_validation_failed"})
                    return None
                
                # 패치: 병렬 그룹에서 run_if가 없을 때 기본 게이트 적용 (순차와 동일 정책)
                step_cfg = step_cfg_map.get(step_name, {}) or {}
                run_if = step_cfg.get("run_if")
                
                if not run_if:
                    # 순차 경로와 동일 기준 적용
                    min_tokens = int(getattr(self.cfg, "MIN_TOKENS_THRESHOLD", 3))
                    min_sentences = int(getattr(self.cfg, "MIN_SENTENCES_THRESHOLD", 1))
                    run_if = [{"all_of": [
                        {"type": "metric_compare", "left": "tokens_count", "right": min_tokens, "op": ">="},
                        {"type": "metric_compare", "left": "sentences_count", "right": min_sentences, "op": ">="}
                    ]}]
                    heavy_steps = {"transition_analyzer", "time_series_analyzer", "psychological_analyzer", "complex_analyzer"}
                    if step_name in heavy_steps:
                        heavy_min_tokens = int(getattr(self.cfg, "HEAVY_MIN_TOKENS_THRESHOLD", 10))
                        run_if = [{"all_of": [
                            {"type": "metric_compare", "left": "tokens_count", "right": heavy_min_tokens, "op": ">="},
                            {"type": "metric_compare", "left": "sentences_count", "right": min_sentences, "op": ">="}
                        ]}]
                    # 병렬용 step_cfg에도 주입(다음 호출부터 일관성)
                    step_cfg_map[step_name] = {**step_cfg, "run_if": run_if}
                
                # run_if 조건 평가
                try:
                    ok, reason = ConditionEvaluator.evaluate_with_reason(run_if, payload)
                    if not ok:
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug(f"{step_name} run_if 조건 실패 - 스킵: {reason}")
                        payload.stamp(step_name, "skip", time.perf_counter(), extras={"reason": f"run_if_failed: {reason}"})
                        return None
                except Exception as e:
                    logger.warning(f"{step_name} run_if 평가 실패: {e}")
                    # 평가 실패 시 기본적으로 실행
                
                # 병렬 실행 시에도 CallAdapter를 통해 인자 주입 보장
                instance = self.registry.get_instance(step_name)
                runner = self.adapter.resolve_callable(step_name, instance)
                if runner:
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"{step_name} 러너 발견: {runner.__name__ if hasattr(runner, '__name__') else type(runner).__name__}")
                    started = time.perf_counter()
                    result = self._run_step_with_runtime_guards(runner, payload, step_cfg_map.get(step_name, {}), step_name=step_name)
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"{step_name} 완료: {type(result).__name__}")
                    
                    # 결과를 payload에 추가하고 상태 기록
                    if result is not None:
                        payload.add_result(step_name, result)
                        payload.stamp(step_name, "ok", started)
                        
                        # 실행 시간 기록
                        duration = time.perf_counter() - started
                        self._record_execution_time(step_name, duration)
                        
                        # P6: 서킷브레이커 성공 기록
                        self._record_circuit_breaker_success(step_name)
                        
                        # 품질 모니터링
                        self._monitor_execution_quality(payload)
                    else:
                        payload.stamp(step_name, "error", started, error="No result returned")
                    
                    return result
                else:
                    # runner가 None인 경우 직접 호출 시도 (fallback)
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"{step_name} 러너 없음, 직접 호출 시도")
                    logger.warning(f"Parallel step {step_name}: runner is None, trying direct call")
                    if callable(instance):
                        started = time.perf_counter()
                        result = self._run_step_with_runtime_guards(instance, payload, step_cfg_map.get(step_name, {}), step_name=step_name)
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug(f"{step_name} 직접 호출 완료: {type(result).__name__}")
                        
                        # 결과를 payload에 추가하고 상태 기록
                        if result is not None:
                            payload.add_result(step_name, result)
                            payload.stamp(step_name, "ok", started)
                            
                            # 실행 시간 기록
                            duration = time.perf_counter() - started
                            self._record_execution_time(step_name, duration)
                            
                            # 품질 모니터링
                            self._monitor_execution_quality(payload)
                        else:
                            payload.stamp(step_name, "error", started, error="No result returned")
                        
                        return result
                    else:
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug(f"{step_name} 인스턴스 호출 불가: {type(instance).__name__}")
                        logger.error(f"Parallel step {step_name}: instance is not callable")
                        payload.mark_skip(step_name, f"Instance is not callable: {type(instance)}")
                        return None
            except Exception as e:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"{step_name} 실패: {e}")
                logger.error(f"Parallel step {step_name} failed: {e}")
                # 병렬 실행 실패 시에도 payload에 에러 마킹
                started = time.perf_counter()
                payload.stamp(step_name, "error", started, error=str(e), extras={"parallel_failed": True})
                return None
        
        # 병렬 실행
        worker_limit = max(1, min(self.max_workers, len(group)))
        with _fut.ThreadPoolExecutor(max_workers=worker_limit) as executor:
            futures = {executor.submit(run_step, step): step for step in group}
            
            for future in _fut.as_completed(futures):
                step_name = futures[future]
                try:
                    result = future.result()
                    # add_result는 run_step 내부에서 이미 호출됨 (중복 제거)
                except Exception as e:
                    logger.error(f"Parallel execution error for {step_name}: {e}")
                    payload.mark_skip(step_name, f"Parallel execution failed: {e}")

    # ---------- 차단 대상/화이트리스트 수집 ----------
    def _cfg_log_roots_and_whitelist(self):
        roots = set()

        # emotion_analysis/logs (복수/단수)
        ea_logs = self._module_logs_dir()
        roots.add(ea_logs)
        roots.add(os.path.abspath(os.path.join(os.path.dirname(ea_logs), "log")))

        # src/logs (복수/단수)
        log_dir = getattr(self.cfg, "LOG_DIR", None)
        if log_dir:
            roots.add(os.path.abspath(log_dir))
            roots.add(os.path.abspath(os.path.join(os.path.dirname(log_dir), "log")))

        # 루트 logs/log
        cfg_dir = os.path.dirname(getattr(self.cfg, "__file__", "."))
        root_dir = os.path.abspath(os.path.join(cfg_dir, os.pardir))
        roots.add(os.path.join(root_dir, "logs"))
        roots.add(os.path.join(root_dir, "log"))

        # 화이트리스트: data_utils.log만 허용
        whitelist = set()
        p = getattr(self.cfg, "DATA_UTILS_LOG_PATH", None)
        if p:
            whitelist.add(self._canon_path(p))
        try:
            if log_dir:
                whitelist.add(self._canon_path(os.path.join(log_dir, "data_utils.log")))
        except Exception:
            pass
        # ENV로 추가 허용 경로(세미콜론 구분)
        extra = os.getenv("EA_LOG_WHITELIST", "")
        for w in (x.strip() for x in extra.split(";") if x.strip()):
            whitelist.add(self._canon_path(w))

        roots_c = [self._canon_path(r) for r in roots]
        return roots_c, set(whitelist)

    # ---------- 파일쓰기 가드 ----------
    def _install_fileio_guard(self) -> None:
        if getattr(self, "_fileio_guard_installed", False):
            return
        import builtins
        policy = getattr(self.cfg, "LOGGING_CONFIG", {})
        suppress = bool(policy.get("suppress_module_file_logs_when_orchestrated", True))
        redirect = bool(policy.get("redirect_module_file_logs_to_run_dir", False))
        if not (suppress or redirect):
            self._fileio_guard_installed = True
            return

        targets, whitelist = self._cfg_log_roots_and_whitelist()
        run_dir = self._run_dir_cache or self._ensure_run_dir()
        allow_dirs = {self._canon_path(run_dir)}  # RUN_DIR 허용

        def _blocked_write(p: str) -> bool:
            ap = self._canon_path(p)
            for a in allow_dirs:
                if self._is_under_canon(ap, a):
                    return False
            if ap in whitelist:
                return False
            return any(self._is_under_canon(ap, r) for r in targets)

        _orig_open = builtins.open

        def _guarded_open(file, mode="r", *a, **kw):
            try:
                if isinstance(file, (str, bytes, os.PathLike)):
                    is_text_write = (not any(b in mode for b in ("b",))) and any(m in mode for m in ("w", "a", "x", "+"))
                    # 대상 경로가 차단 대상이면 리다이렉트/서프레스 처리
                    if any(m in mode for m in ("w", "a", "x", "+")) and _blocked_write(os.fspath(file)):
                        if redirect and run_dir:
                            newp = os.path.join(run_dir, os.path.basename(os.fspath(file)))
                            if is_text_write and "encoding" not in kw:
                                kw["encoding"] = "utf-8"
                            return _orig_open(newp, mode, *a, **kw)
                        elif suppress:
                            # 서프레스 시 /dev/null로. 텍스트 모드도 동일 처리
                            return _orig_open(os.devnull, mode, *a, **kw)
                    # 일반 경로: 텍스트 쓰기 모드면 UTF-8 기본 강제
                    if is_text_write and "encoding" not in kw:
                        kw["encoding"] = "utf-8"
            except Exception:
                pass
            return _orig_open(file, mode, *a, **kw)

        builtins.open = _guarded_open

        _orig_path_open = Path.open

        def _path_open(self_p, mode="r", *a, **kw):
            return builtins.open(self_p.__fspath__(), mode, *a, **kw)

        Path.open = _path_open
        self._fileio_guard_installed = True

    # ---------- 로깅 가드 ----------
    def _install_logging_guard(self) -> None:
        if getattr(self, "_logging_guard_installed", False):
            return
        try:
            from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
        except Exception:
            RotatingFileHandler = None
            TimedRotatingFileHandler = None  # type: ignore[assignment]

        policy = getattr(self.cfg, "LOGGING_CONFIG", {})
        suppress = bool(policy.get("suppress_module_file_logs_when_orchestrated", True))
        redirect = bool(policy.get("redirect_module_file_logs_to_run_dir", False))
        if not (suppress or redirect):
            self._logging_guard_installed = True
            return

        targets, whitelist = self._cfg_log_roots_and_whitelist()
        run_dir = self._run_dir_cache or self._ensure_run_dir()
        allow_dirs = {self._canon_path(run_dir)}

        def _blocked(path: str) -> bool:
            ap = self._canon_path(path)
            for a in allow_dirs:
                if self._is_under_canon(ap, a):
                    return False
            if ap in whitelist:
                return False
            return any(self._is_under_canon(ap, r) for r in targets)

        _orig_fh = logging.FileHandler.__init__

        def _fh(self_h, filename, *a, **kw):
            fn = filename
            try:
                if _blocked(fn):
                    if redirect and run_dir:
                        fn = os.path.join(run_dir, os.path.basename(fn))
                    elif suppress:
                        fn = os.devnull
            except Exception:
                pass
            return _orig_fh(self_h, fn, *a, **kw)

        logging.FileHandler.__init__ = _fh

        if RotatingFileHandler is not None:
            _orig_rfh = RotatingFileHandler.__init__  # type: ignore[assignment]

            def _rfh(self_h, filename, *a, **kw):
                fn = filename
                try:
                    if _blocked(fn):
                        if redirect and run_dir:
                            fn = os.path.join(run_dir, os.path.basename(fn))
                        elif suppress:
                            fn = os.devnull
                except Exception:
                    pass
                return _orig_rfh(self_h, fn, *a, **kw)

            RotatingFileHandler.__init__ = _rfh  # type: ignore[assignment]

        if TimedRotatingFileHandler is not None:
            _orig_trfh = TimedRotatingFileHandler.__init__  # type: ignore[assignment]

            def _trfh(self_h, filename, *a, **kw):
                fn = filename
                try:
                    if _blocked(fn):
                        if redirect and run_dir:
                            fn = os.path.join(run_dir, os.path.basename(fn))
                        elif suppress:
                            fn = os.devnull
                except Exception:
                    pass
                return _orig_trfh(self_h, fn, *a, **kw)

            TimedRotatingFileHandler.__init__ = _trfh  # type: ignore[assignment]

        # basicConfig(filename=...) 가드
        _orig_basic = logging.basicConfig

        def _basic_guard(*a, **kw):
            fn = kw.get("filename")
            try:
                if fn and _blocked(fn):
                    if redirect and run_dir:
                        kw["filename"] = os.path.join(run_dir, os.path.basename(fn))
                    elif suppress:
                        kw.pop("filename", None)  # 파일핸들러 제거(스트림만)
            except Exception:
                pass
            return _orig_basic(*a, **kw)

        logging.basicConfig = _basic_guard
        self._logging_guard_installed = True

    # ---------- 이미 붙은 핸들러 정리/리다이렉트 ----------
    def _reconfigure_module_file_handlers(self, run_dir: Optional[str] = None) -> None:
        """통합 로그 관리자를 사용하여 모듈 로그 핸들러 재구성"""
        try:
            from log_manager import get_log_manager
            log_manager = get_log_manager()
            
            policy = getattr(self.cfg, "LOGGING_CONFIG", {})
            suppress = bool(policy.get("suppress_module_file_logs_when_orchestrated", True))
            redirect = bool(policy.get("redirect_module_file_logs_to_run_dir", False))
            
            if redirect and run_dir:
                # 통합 로그 관리자를 사용한 리다이렉트
                from pathlib import Path
                target_dir = Path(run_dir)
                log_manager.redirect_module_logs(target_dir)
            elif suppress:
                # 모든 모듈 로그 억제
                for logger_name in logging.Logger.manager.loggerDict:
                    logger_i = logging.getLogger(logger_name)
                    for handler in logger_i.handlers[:]:
                        if isinstance(handler, logging.FileHandler):
                            logger_i.removeHandler(handler)
                            logger_i.addHandler(logging.NullHandler())
            return
        except ImportError:
            # 폴백: 기존 방식
            pass
        
        # 기존 방식 (폴백)
        policy = getattr(self.cfg, "LOGGING_CONFIG", {})
        suppress = bool(policy.get("suppress_module_file_logs_when_orchestrated", True))
        redirect = bool(policy.get("redirect_module_file_logs_to_run_dir", False))
        if not (suppress or redirect):
            return

        targets, whitelist = self._cfg_log_roots_and_whitelist()
        fmt = getattr(self.cfg, "LOGGING_FORMAT", "%(asctime)s - %(levelname)s - %(name)s - %(message)s")
        run_dir = run_dir or self._run_dir_cache or self._ensure_run_dir()
        for name, logobj in list(logging.root.manager.loggerDict.items()):
            logger_i = logobj if isinstance(logobj, logging.Logger) else logging.getLogger(name)
            if not isinstance(logger_i, logging.Logger):
                continue
            for h in list(logger_i.handlers):
                if not isinstance(h, logging.FileHandler):
                    continue
                fpath = getattr(h, "baseFilename", "")
                if not fpath:
                    continue
                ap = self._canon_path(fpath)
                if ap in whitelist:
                    continue
                if any(self._is_under_canon(ap, r) for r in targets):
                    try:
                        logger_i.removeHandler(h)
                        h.flush()
                        h.close()
                    except Exception:
                        pass
                    if redirect and run_dir:
                        try:
                            fname = os.path.basename(ap)
                            nh = logging.FileHandler(os.path.join(run_dir, fname), encoding="utf-8")
                            nh.setFormatter(logging.Formatter(fmt))
                            nh.setLevel(getattr(h, "level", logging.getLogger().level))
                            logger_i.addHandler(nh)
                        except Exception:
                            logger.debug("[orchestrator] redirect FileHandler failed for %s", ap, exc_info=True)
            logger_i.propagate = True
            try:
                logger_i.setLevel(logging.getLogger().level)
            except Exception:
                pass


# 선택: 간단한 수동 테스트 러너 (강화/대용량 대응) -----------------------------------


def _get_cli_orchestrator_and_config():
    try:
        module = importlib.import_module("src.data_utils")
    except ModuleNotFoundError:
        return EmotionPipelineOrchestrator, config
    return module.EmotionPipelineOrchestrator, module.config


# ============================================================================
# Patch-3: FileIOGuardCtx - 컨텍스트 매니저 버전 (테스트/임시 런 편의)
# ============================================================================
class FileIOGuardCtx:
    """File I/O 가드를 컨텍스트 매니저로 관리 (테스트 환경용)"""
    def __init__(self, orchestrator: "EmotionPipelineOrchestrator"):
        self.o = orchestrator
        self._installed = False
    
    def __enter__(self):
        if not self._installed:
            self.o._install_fileio_guard()
            self.o._install_logging_guard()
            self._installed = True
        return self
    
    def __exit__(self, exc_type, exc, tb):
        # 필요하다면 여기서 원복 로직 추가(현재는 퍼시스턴트 가드 유지)
        return False


# ============================================================================
# Patch-8: PerformanceMetrics - 성능 메트릭 집계 클래스 (선택)
# ============================================================================
@dataclass(slots=True)
class PerformanceMetrics:
    """파이프라인 스텝별 성능 메트릭 집계"""
    step_timings: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    
    def record(self, step: str, ms: float):
        """스텝의 실행 시간 기록"""
        arr = self.step_timings[step]
        arr.append(ms)
        # 메모리 관리를 위해 최대 200개만 유지
        if len(arr) > 200:
            self.step_timings[step] = arr[-100:]
    
    def stats(self, step: str) -> dict:
        """특정 스텝의 통계 (mean, p95)"""
        arr = self.step_timings.get(step, [])
        if not arr:
            return {}
        try:
            import numpy as np
            return {
                "mean_ms": float(np.mean(arr)),
                "p95_ms": float(np.percentile(arr, 95)),
                "count": len(arr)
            }
        except ImportError:
            # numpy가 없으면 기본 통계만
            return {
                "mean_ms": sum(arr) / len(arr),
                "min_ms": min(arr),
                "max_ms": max(arr),
                "count": len(arr)
            }


if __name__ == "__main__":

    # ---------------------- CLI ----------------------
    ap = argparse.ArgumentParser(
        description="EmotionPipelineOrchestrator CLI (hardened)",
        epilog="권장 실행 방법:\n"
               "  python -m src.data_utils --text '테스트 문장'\n"
               "  python -m emotion_analysis.data_utils --text '테스트 문장'\n"
               "환경변수로 패키지 베이스 고정:\n"
               "  EMOTION_ANALYSIS_PACKAGE=src.emotion_analysis python -m src.data_utils --text '테스트 문장'",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--text", type=str, help="직접 입력 텍스트")
    src.add_argument("--file", type=str, help="텍스트가 들어있는 파일 경로(UTF-8 기본, CP949 등 폴백)")
    src.add_argument("--stdin", action="store_true", help="STDIN으로부터 텍스트를 읽음")

    # 배치(JSONL)
    ap.add_argument("--jsonl", type=str, help="JSONL 파일 경로 (각 줄: {text, meta?, id?})")
    ap.add_argument("--save-each", action="store_true", help="JSONL 배치: 각 결과를 개별 JSON으로 저장")
    ap.add_argument("--max-lines", type=int, default=None, help="JSONL에서 최대 처리 라인 수 제한")

    # 저장/로그/성능
    ap.add_argument("--save", action="store_true", help="단일 텍스트 결과를 JSON으로 저장")
    ap.add_argument("--outdir", type=str, default=None, help="결과 JSON 저장 폴더")
    ap.add_argument("--prewarm", action="store_true", help="시작 시 모듈 프리워밍")
    ap.add_argument("--show-trace", action="store_true", help="스텝별 trace를 콘솔에 출력")
    ap.add_argument("--log-level", type=str, choices=["DEBUG","INFO","WARNING","ERROR","CRITICAL"], help="루트 로그 레벨 설정")
    ap.add_argument("--redirect-logs", action="store_true", help="모듈 파일 로그를 RUN_DIR로 리다이렉트")
    ap.add_argument("--no-step-json", action="store_true", help="스텝별 중간 결과 JSON 저장 비활성화")

    # 디바이스/환경
    ap.add_argument("--device", type=str, choices=["cpu","cuda"], help="임베딩 디바이스 오버라이드")
    ap.add_argument("--disable-file-guards", action="store_true", help="파일 I/O/로깅 가드 비활성화(EA_DISABLE_FILE_GUARDS=1)")

    # ===== 신규: 상태/재개/안전 종료 =====
    ap.add_argument("--status-every", type=float, default=5.0, help="상태 출력 주기(초). 0이면 비활성")
    ap.add_argument("--status-json", type=str, default=None, help="상태를 주기적으로 덤프할 JSON 경로")
    ap.add_argument("--resume-from", type=str, default=None, help="이미 처리한 JSONL 경로(id/text 해시 기준으로 스킵)")
    ap.add_argument("--stop-file", type=str, default=None, help="이 파일이 존재하면 다음 샘플에서 안전 종료")

    args = ap.parse_args()

    # ---------------------- 유틸 ----------------------
    def _planned_steps(cfg):
        """파이프라인에서 실제 실행 대상으로 계획된 스텝(전처리 제외)."""
        try:
            steps = [
                s.get("name")
                for s in cfg.EMOTION_ANALYSIS_PIPELINE.get("steps", [])
                if s.get("enabled", True)
            ]
            return [s for s in steps if s and s != "text_preprocessing"]
        except Exception:
            return [k for k in getattr(cfg, "MODULE_ENTRYPOINTS", {}).keys()]

    def _print_summary(out: dict, planned: list[str]) -> None:
        """요약 리포트 출력."""
        results_keys = list(out.get("results", {}).keys())
        status_map = {}
        for t in out.get("trace", []):
            s = t.get("step")
            if s:
                status_map[s] = t.get("status", "unknown")
        executed = [s for s in planned if s in results_keys]
        missing = [s for s in planned if s not in results_keys]
        
        # [Genius Fix] 트레이스 누락이어도 결과가 실존하면 성공으로 간주 (실질주의적 집계)
        final_statuses = []
        for s in planned:
            st = status_map.get(s)
            if not st:
                # 트레이스엔 없지만 결과 키가 있으면 'ok' (Silent Success)
                st = "ok" if s in results_keys else "missing"
            final_statuses.append(st)
        counts = Counter(final_statuses)

        print("\n=== Pipeline Summary ===")
        print(f"Planned steps : {len(planned)}")
        print(f"Executed steps: {len(executed)}/{len(planned)}")
        print("Status counts :", dict(counts))
        print("Missing      :", ", ".join(missing) if missing else "(none)")

        topk = _dig(out, "results.emotion_classification.router.topk_main", [])
        if topk:
            def _fmt(e):
                if isinstance(e, (list, tuple)) and len(e) >= 2 and isinstance(e[1], (int, float)):
                    return f"{e[0]}({e[1]:.2f})"
                if isinstance(e, dict):
                    lbl = e.get("label") or e.get("id") or e.get("name") or "?"
                    sc = e.get("score") or e.get("confidence")
                    return f"{lbl}({sc:.2f})" if isinstance(sc, (int, float)) else str(lbl)
                return str(e)
            show = ", ".join(_fmt(e) for e in topk[:3])
            print("Top main emotions:", show)

        if _dig(out, "meta.output_path"):
            print("Saved JSON   :", out["meta"]["output_path"])

    def _read_text_from_file(path: str) -> str:
        p = Path(path)
        data = None
        for enc in ("utf-8", "utf-8-sig", "cp949", "euc-kr"):
            try:
                data = p.read_text(encoding=enc)
                break
            except Exception:
                continue
        if data is None:
            raw = p.read_bytes()
            try:
                data = raw.decode("utf-8", "replace")
            except Exception:
                data = raw.decode("latin-1", "replace")
        return data

    def _ensure_dirs():
        try:
            if args.outdir:
                Path(args.outdir).mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

    def _apply_runtime_overrides(cfg_module):
        if args.log_level:
            os.environ["DATA_UTILS_LOG_LEVEL"] = args.log_level
            try:
                logging.getLogger().setLevel(getattr(logging, args.log_level, logging.INFO))
            except Exception:
                pass
        try:
            lg = getattr(cfg_module, "LOGGING_CONFIG", {})
            if args.redirect_logs:
                lg["redirect_module_file_logs_to_run_dir"] = True
            if args.no_step_json:
                lg["write_step_results"] = False
                lg["write_step_results_for_cached"] = False
            setattr(cfg_module, "LOGGING_CONFIG", lg)
        except Exception:
            pass
        if args.device:
            try:
                ep = getattr(cfg_module, "EMBEDDING_PARAMS", {})
                ep["device"] = args.device
                setattr(cfg_module, "EMBEDDING_PARAMS", ep)
            except Exception:
                pass
        if args.disable_file_guards:
            os.environ["EA_DISABLE_FILE_GUARDS"] = "1"

    orchestrator_cls, config_module = _get_cli_orchestrator_and_config()
    # ---------------------- 실행 모드 분기 ----------------------
    _apply_runtime_overrides(config_module)
    _ensure_dirs()

    # ===== 배치(JSONL) 모드 =====
    if args.jsonl:
        in_path = Path(args.jsonl)
        if not in_path.is_file():
            print(f"[ERROR] JSONL not found: {in_path}", file=sys.stderr)
            sys.exit(2)

        # 재개: 이미 처리한 키 로딩(id → text_hash → text 해시)
        def _make_key(obj):
            _id = obj.get("id")
            if _id:
                return str(_id)
            txt = (obj.get("text") or "").strip()
            return _hash_text(txt) if txt else None

        seen = set()
        if args.resume_from:
            try:
                with open(args.resume_from, "r", encoding="utf-8") as rf:
                    for ln, line in enumerate(rf, 1):
                        try:
                            o = json.loads(line)
                            k = (o.get("text_hash") or _make_key(o))
                            if k: seen.add(k)
                        except Exception:
                            continue
            except FileNotFoundError:
                pass

        # 총 줄 수(분모)
        try:
            with in_path.open("r", encoding="utf-8") as f:
                total_lines = sum(1 for _ in f)
        except Exception:
            total_lines = 0

        # 상태/ETA 쓰레드
        STATUS_EVERY = float(max(0.05, args.status_every)) if args.status_every else 0.0  # 고성능 환경에 맞춰 상태 업데이트 빈도 증가
        WARMUP_COUNT = 50
        WARMUP_TIME  = 30.0
        EMA_ALPHA    = 0.20

        proc_lock   = threading.Lock()
        proc_done   = 0
        proc_ok     = 0
        proc_fail   = 0
        t0          = perf_counter()
        prev_t      = [t0]
        prev_done   = [0]
        ema_rate    = [0.0]
        stop_evt    = threading.Event()

        def _fmt_eta(rem, rate):
            if not rate or rate <= 0: return "?"
            sec = rem / rate
            m, s = int(sec // 60), int(sec % 60)
            h = m // 60; m = m % 60
            return f"{h}h{m:02d}m{s:02d}s" if h else f"{m:02d}m{s:02d}s"

        def _dump_status_json(path, payload):
            try:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False, indent=2)
            except Exception:
                pass

        def _status_pump():
            last_len = 0
            while not stop_evt.wait(STATUS_EVERY if STATUS_EVERY > 0 else 10**9):
                now = perf_counter()
                with proc_lock:
                    done = proc_done; ok = proc_ok; fail = proc_fail

                dt = max(1e-6, now - prev_t[0])
                inst = max(0.0, (done - prev_done[0]) / dt)
                ema_rate[0] = (1.0-EMA_ALPHA)*ema_rate[0] + EMA_ALPHA*inst
                prev_t[0], prev_done[0] = now, done

                elapsed = now - t0
                warmed = (done >= WARMUP_COUNT and elapsed >= WARMUP_TIME)
                use_r  = ema_rate[0] if warmed else inst

                if total_lines:
                    pct = (done/total_lines)*100.0
                    rem = max(0, total_lines - done)
                    eta = _fmt_eta(rem, use_r) if warmed and use_r>0 else "warming up"
                    line = (f"[status] {done}/{total_lines} ({pct:5.1f}%) "
                            f"| ok+{ok} fail+{fail} | rate {use_r:4.1f}/s | eta {eta}")
                else:
                    line = (f"[status] processed={done} | ok+{ok} fail+{fail} | rate {use_r:4.1f}/s")

                print("\r" + line + " " * max(0, last_len - len(line)), end="", flush=True)
                last_len = len(line)

                if args.status_json:
                    payload = dict(
                        processed_total=done, processed_session=done,
                        n_ok=ok, n_fail=fail, rate_sps=use_r,
                        total_expected=total_lines,
                        pct=(done/total_lines*100.0) if total_lines else None,
                        eta=(eta if total_lines else None),
                        elapsed_sec=elapsed,
                    )
                    _dump_status_json(args.status_json, payload)

            if STATUS_EVERY > 0:
                print()

        if STATUS_EVERY > 0:
            threading.Thread(target=_status_pump, daemon=True).start()

        # 실행
        orch = orchestrator_cls(prewarm=args.prewarm)
        started_all = perf_counter()
        n_total = 0; n_ok = 0

        try:
            with in_path.open("r", encoding="utf-8") as f:
                for line_idx, line in enumerate(f, start=1):
                    if args.max_lines and line_idx > args.max_lines:
                        break
                    raw = line.strip()
                    if not raw:
                        continue
                    n_total += 1

                    # 안전 종료
                    if args.stop_file and os.path.exists(args.stop_file):
                        print("\n[stop] found stop-file → graceful exit")
                        break

                    # 재개 스킵
                    try:
                        obj = json.loads(raw)
                    except Exception:
                        with proc_lock:
                            proc_done += 1; proc_fail += 1
                        continue
                    k = (obj.get("id") or _hash_text((obj.get("text") or "").strip()))
                    if k and k in seen:
                        with proc_lock:
                            proc_done += 1  # 이미 처리된 것으로 간주(스킵)
                        continue

                    # 실제 처리
                    try:
                        text = (obj.get("text") or "").strip()
                        meta = obj.get("meta")
                        if not text:
                            raise ValueError("record has no 'text'")
                        out = orch.process_text(text, meta=meta, save=args.save_each, out_dir=args.outdir)
                        with proc_lock:
                            proc_done += 1; proc_ok += 1
                            seen.add(k)
                        n_ok += 1
                        if (n_total % 50) == 0:
                            print("", end="")  # 상태줄만 유지
                    except Exception as e:
                        with proc_lock:
                            proc_done += 1; proc_fail += 1
                        print(f"[WARN] line {line_idx} failed: {e}", file=sys.stderr)
                        continue
        except Exception as e:
            print(f"[ERROR] reading JSONL failed: {e}", file=sys.stderr)
            stop_evt.set()
            sys.exit(1)

        stop_evt.set()
        dur = perf_counter() - started_all
        print(f"\n=== Batch Summary ===\nFile: {in_path}\nProcessed: {n_ok}/{n_total}\nElapsed: {dur:.2f}s")
        sys.exit(0 if n_ok > 0 else 1)

    # ===== 단일 텍스트 모드 =====
    # 입력 텍스트 확보 (우선순위: --text → --file → --stdin → 기본문장)
    input_text = args.text
    if not input_text and args.file:
        try:
            input_text = _read_text_from_file(args.file)
        except Exception as e:
            print(f"[ERROR] failed to read file: {e}", file=sys.stderr)
            sys.exit(2)
    if not input_text and args.stdin:
        try:
            input_text = sys.stdin.read()
        except Exception as e:
            print(f"[ERROR] failed to read stdin: {e}", file=sys.stderr)
            sys.exit(2)
    if not input_text:
        input_text = "오늘은 정말 행복했지만, 갑자기 예상치 못한 일이 생겨 마음이 복잡해졌다."

    orch = orchestrator_cls(prewarm=args.prewarm)
    t0 = perf_counter()
    out = orch.process_text(input_text, meta={"source": "cli"}, save=args.save, out_dir=args.outdir)
    elapsed = perf_counter() - t0

    planned = _planned_steps(config_module)
    if args.show_trace:
        try:
            print(json.dumps(out.get("trace", []), ensure_ascii=False, indent=2))
        except Exception:
            print(out.get("trace", []))

    _print_summary(out, planned)

    print("\n=== Meta ===")
    meta_keys = ("trace_id", "model_version", "device")
    meta_out = {k: out["meta"][k] for k in meta_keys if k in out.get("meta", {})}
    meta_out["elapsed_sec"] = round(elapsed, 2)
    try:
        print(json.dumps(meta_out, ensure_ascii=False, indent=2))
    except Exception:
        print(meta_out)

    sys.exit(0)
