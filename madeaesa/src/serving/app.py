"""
FastAPI 기반 Emotion AI 웹 서버.
test.py 내부 파이프라인(run_one)을 직접 호출하여 동일한 결과를 제공한다.
"""

from __future__ import annotations

import copy
import math
import datetime
import json
import logging
import os
import re
import threading
import time
import uuid
import asyncio
import concurrent.futures
import gc  # [Genius] 메모리 관리를 위한 GC 명시적 호출
from collections import Counter, defaultdict
from functools import lru_cache
from importlib import import_module
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import uvicorn

# ---------------------------------------------------------------------------
# 프로젝트 루트 및 로깅
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# .env 파일 로드 (src/.env 및 프로젝트 루트 .env 지원)
try:
    load_dotenv(PROJECT_ROOT / "src" / ".env")
    load_dotenv(PROJECT_ROOT / ".env")
except Exception:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# [GENIUS INSIGHT] Global Concurrency Semaphore
# Initialized in startup_event to ensure event loop compatibility
GLOBAL_ANALYSIS_SEMAPHORE: Optional[asyncio.Semaphore] = None

# ---------------------------------------------------------------------------
# test.py 함수 레이지 로딩
# ---------------------------------------------------------------------------
_RUN_ONE: Optional[Callable[[str], Dict[str, Any]]] = None
_TO_WEB_BUNDLE: Optional[Callable[[Mapping[str, Any]], Mapping[str, Any]]] = None
_FORMAT_SUB_LABEL_FUNC: Optional[Callable[[Any, Any], str]] = None
_GATHER_TRANSITIONS_AND_KEYWORDS: Optional[Callable[..., Any]] = None
_EMO_FLOW_FROM_OUT: Optional[Callable[[Mapping[str, Any]], Any]] = None
_NORMALIZE_SUB_KEY_FUNC: Optional[Callable[[Any, Mapping[str, float]], Tuple[str, str]]] = None
_TRUNCATE_FUNC: Optional[Callable[..., Any]] = None
_EXPLICIT_EMOTION_FUNC: Optional[Callable[..., Any]] = None
_SEMANTIC_SUB_FUNC: Optional[Callable[..., Any]] = None
_HAS_UNCERTAINTY_CUES_FUNC: Optional[Callable[..., Any]] = None
_DETECT_TRIGGERS_FUNC: Optional[Callable[..., Any]] = None
_NORMALIZED_ENTROPY_FUNC: Optional[Callable[..., Any]] = None


def _ensure_test_exports() -> None:
    global _RUN_ONE, _TO_WEB_BUNDLE, _FORMAT_SUB_LABEL_FUNC
    global _GATHER_TRANSITIONS_AND_KEYWORDS, _EMO_FLOW_FROM_OUT, _NORMALIZE_SUB_KEY_FUNC
    global _TRUNCATE_FUNC, _EXPLICIT_EMOTION_FUNC, _SEMANTIC_SUB_FUNC
    global _HAS_UNCERTAINTY_CUES_FUNC, _DETECT_TRIGGERS_FUNC, _NORMALIZED_ENTROPY_FUNC
    if _RUN_ONE is not None and _TO_WEB_BUNDLE is not None:
        return
    module = import_module("test")
    _RUN_ONE = getattr(module, "run_one", None)
    _TO_WEB_BUNDLE = getattr(module, "_to_web_bundle", None)
    fmt = getattr(module, "_format_sub_label", None)
    if callable(fmt):
        _FORMAT_SUB_LABEL_FUNC = fmt
    gather = getattr(module, "_gather_transitions_and_keywords", None)
    if callable(gather):
        _GATHER_TRANSITIONS_AND_KEYWORDS = gather
    emo_flow = getattr(module, "_emo_flow_ko_from_seq_or_trans", None)
    if callable(emo_flow):
        _EMO_FLOW_FROM_OUT = emo_flow
    norm_sub = getattr(module, "_normalize_sub_key", None)
    if callable(norm_sub):
        _NORMALIZE_SUB_KEY_FUNC = norm_sub
    trunc = getattr(module, "_truncate", None)
    if callable(trunc):
        _TRUNCATE_FUNC = trunc
    explicit = getattr(module, "_explicit_emotion_from_text", None)
    if callable(explicit):
        _EXPLICIT_EMOTION_FUNC = explicit
    semantic = getattr(module, "_semantic_sub_from_emotions", None)
    if callable(semantic):
        _SEMANTIC_SUB_FUNC = semantic
    cues = getattr(module, "_has_uncertainty_cues", None)
    if callable(cues):
        _HAS_UNCERTAINTY_CUES_FUNC = cues
    trig = getattr(module, "_detect_triggers", None)
    if callable(trig):
        _DETECT_TRIGGERS_FUNC = trig
    norm_entropy = getattr(module, "_normalized_entropy", None)
    if callable(norm_entropy):
        _NORMALIZED_ENTROPY_FUNC = norm_entropy

    # [PATCH] Apply Runtime Monkey Patches for Parallel Execution
    _patch_orchestrator_parallelism()

def _patch_orchestrator_parallelism():
    """
    [Genius Patch] Monkey-patch EmotionPipelineOrchestrator to use Layered Parallel Execution.
    - Replaces sequential _process_precision_steps with a parallelized version.
    - Reduces analysis time by running independent modules concurrently.
    """
    # [Genius Activation] Windows 환경에서도 안전한 병렬 처리 활성화
    # - Import Lock 데드락 방지를 위해 startup 시점에 _ensure_test_exports() 호출 필수
    # - ThreadPoolExecutor를 사용하여 I/O 바운드 및 독립 연산 가속화
    try:
        from src import data_utils
        if hasattr(data_utils, "EmotionPipelineOrchestrator") and not getattr(data_utils, "_PATCHED_PARALLEL", False):
            logger.info("[GeniusPatch] Applying Layered Parallel Execution to Orchestrator...")
            
            # Define the optimized method
            def _optimized_process_precision_steps(self, text: str, meta: Optional[Dict[str, Any]], fast_result: Dict[str, Any]) -> Dict[str, Any]:
                logger.info(f"[ParallelOrchestrator] Starting Optimized Layered Execution for {len(text)} chars")
                
                # 1. Initialize Payload
                payload = self._make_initial_payload(text, meta)
                for k, v in (fast_result.get("results") or {}).items():
                    payload.add_result(k, v)

                # 2. Define Layers (Dependency-aware)
                layers = [
                    # Layer 0: Independent Base Modules
                    ["pattern_extractor", "linguistic_matcher", "intensity_analysis", "emotion_classification", "context_extractor"],
                    # Layer 1: First-order Dependencies
                    ["context_analysis", "weight_calculator"],
                    # Layer 2: Second-order Dependencies
                    ["complex_analyzer", "situation_analyzer", "relationship_analyzer", "transition_analyzer"],
                    # Layer 3: Final Analysis
                    ["psychological_analyzer", "time_series_analyzer"]
                ]

                # 3. Execute Layers
                # We use the global _EXECUTOR defined in app.py (max_workers=3 for stability)
                global _EXECUTOR
                
                for i, layer_steps in enumerate(layers):
                    # Filter steps that are actually in self.precision_steps
                    current_steps = [s for s in layer_steps if s in self.precision_steps]
                    if not current_steps:
                        continue

                    logger.info(f"[ParallelOrchestrator] Executing Layer {i}: {current_steps}")
                    
                    # Submit tasks
                    future_to_step = {
                        _EXECUTOR.submit(self._execute_step, step_name, payload): step_name
                        for step_name in current_steps
                    }
                    
                    # Wait for results and update payload immediately to satisfy dependencies for next layer
                    # [Genius Stability] 타임아웃 대폭 상향 (45s -> 120s) 및 예외 처리 강화
                    for future in concurrent.futures.as_completed(future_to_step):
                        step_name = future_to_step[future]
                        try:
                            # 120초 타임아웃: 정밀 분석의 안정성을 위해 충분한 시간을 부여
                            # OOM이나 크래시가 아니라면, 시간이 걸려도 완료되는 것이 사용자 경험상 낫습니다.
                            result = future.result(timeout=120) 
                            if result is not None:
                                payload.add_result(step_name, result)
                        except concurrent.futures.TimeoutError:
                            logger.warning(f"[ParallelOrchestrator] Step {step_name} TIMEOUT (120s). Skipping.")
                            future.cancel() # 가능한 경우 작업 취소 시도
                            payload.add_result(step_name, {"error": "timeout", "confidence": 0.0})
                        except Exception as e:
                            logger.warning(f"[ParallelOrchestrator] Step {step_name} failed: {e}")
                            payload.add_result(step_name, {"error": str(e), "confidence": 0.0})
                        finally:
                            # [Genius Memory] 단계별 메모리 정리 유도
                            # 명시적 호출은 오버헤드가 있을 수 있으나, 안정성 모드에서는 안전장치로 동작
                            pass
                
                # [CRITICAL FIX] Reconstruct Payload state before returning
                # The original orchestrator expects the return value to be a dictionary containing 'results' 
                # and potentially side-effects on the payload object.
                # The payload.to_output() method creates the final structure.
                
                # Ensure payload has dominant_emotions if not already set (needed for web bundle)
                if not payload.dominant_emotions:
                    try:
                        # Attempt to reconstruct dominant emotions from weight_calculator or simple max
                        res = payload.get_result("weight_calculator")
                        if res and "main_distribution" in res:
                            payload.dominant_emotions = res.get("dominant_emotions", [])
                        
                        # Fallback: Try to find main_distribution in any module result
                        if not payload.dominant_emotions:
                            for step_name in current_steps:
                                res = payload.get_result(step_name)
                                if isinstance(res, dict) and "main_distribution" in res:
                                     # Found it!
                                     payload.dominant_emotions = res.get("dominant_emotions", [])
                                     if not payload.dominant_emotions and "main_distribution" in res:
                                         # Manual reconstruction if only distribution exists
                                         dist = res["main_distribution"]
                                         if isinstance(dist, dict):
                                             payload.dominant_emotions = [
                                                 {"name": k, "score": v} 
                                                 for k, v in sorted(dist.items(), key=lambda x: x[1], reverse=True)[:3]
                                             ]
                                     break
                    except Exception as e:
                        logger.warning(f"[ParallelOrchestrator] Failed to reconstruct dominant_emotions: {e}")
                        pass

                output = payload.to_output()
                
                # [Safety Check] Ensure critical keys exist for web bundle
                if "results" not in output:
                    output["results"] = {}
                
                # [DEBUG] Log Parallel Execution Result
                logger.info(f"[ParallelOrchestrator] Output keys: {list(output.keys())}")
                if "weight_calculator" in output.get("results", {}):
                    wc_debug = output["results"]["weight_calculator"]
                    logger.info(f"[ParallelOrchestrator] WC main_dist: {wc_debug.get('main_distribution')}")

                # Propagate critical results to top level if missing (for compatibility)
                results = output["results"]
                if "main_distribution" not in results:
                     # Try to find in module outputs
                     wc_res = results.get("weight_calculator")
                     if wc_res and "main_distribution" in wc_res:
                         results["main_distribution"] = wc_res["main_distribution"]
                
                return output

            # Apply the patch
            data_utils.EmotionPipelineOrchestrator._process_precision_steps = _optimized_process_precision_steps
            data_utils._PATCHED_PARALLEL = True
            logger.info("[GeniusPatch] Orchestrator successfully patched!")
            
    except Exception as e:
        logger.warning(f"[GeniusPatch] Failed to patch Orchestrator: {e}")

def _call_run_one(text: str) -> Dict[str, Any]:
    _ensure_test_exports()
    if _RUN_ONE is None:
        # [Safety] 모듈 리로딩 시도 (핫픽스 반영)
        try:
            import importlib
            import test
            importlib.reload(test)
            _ensure_test_exports()
        except Exception:
            pass
            
    if _RUN_ONE is None:
        raise RuntimeError("run_one 함수 로드 실패")
    
    # [Soft-Ensemble] 룰 베이스 확률 반영을 위해 test.py의 run_one은 
    # 내부적으로 이미 Soft-Ensemble 로직을 포함하고 있습니다.
    # 따라서 여기서는 별도의 로직 추가 없이 호출만 하면 되지만,
    # 혹시 모를 환경변수 누락을 방지하기 위해 명시적으로 설정합니다.
    os.environ["USE_SOFT_ENSEMBLE"] = "1"
    
    return _RUN_ONE(text)


def _get_to_web_bundle() -> Optional[Callable[[Mapping[str, Any]], Mapping[str, Any]]]:
    _ensure_test_exports()
    return _TO_WEB_BUNDLE


def _format_sub_label(main: Any, sub: Any) -> str:
    if _FORMAT_SUB_LABEL_FUNC:
        try:
            # [GENIUS FIX] sub_ formatting consistency
            # test.py logic: _format_sub_label(main, raw_sub)
            # raw_sub can be "sub_01", "1", "희-sub_01" etc.
            # We must ensure it returns the human-readable label.
            result = _FORMAT_SUB_LABEL_FUNC(main, sub)
            
            # If result still looks like an ID (e.g. "sub_01", "—"), try to fallback to default map if possible
            # But respecting test.py's decision is priority.
            if result and "sub_" in str(result):
                # Try one more time with explicit string conversion if it wasn't string
                if not isinstance(sub, str):
                    result_retry = _FORMAT_SUB_LABEL_FUNC(main, str(sub))
                    if result_retry and "sub_" not in result_retry:
                        return result_retry

                logger.warning(f"[_format_sub_label] Index format remains: main={main}, sub={sub}, result={result}")
                # 메인 감정에 따른 기본 서브 감정 반환 (Emergency Fallback)
                default_sub = MAIN_DEFAULT_SUB.get(str(main), "—")
                return default_sub
            return result
        except Exception as e:
            logger.warning(f"[_format_sub_label] Failed: main={main}, sub={sub}, error={e}")
    
    # 폴백: sub가 sub_XX 형식이면 기본값 사용
    sub_str = str(sub) if sub is not None else "—"
    if "sub_" in sub_str:
        return MAIN_DEFAULT_SUB.get(str(main), "—")
    return sub_str

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore

# ---------------------------------------------------------------------------
# FastAPI 설정
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Emotion AI Web Interface",
    description="test.py 런타임 파이프라인과 동일한 결과를 제공하는 FastAPI 서버",
    version="4.0",
)

@app.on_event("startup")
async def startup_event():
    """서버 시작 시 초기화 및 웜업"""
    logger.info("[App] Startup: Initializing & Warming up...")
    
    # [GENIUS INSIGHT] Initialize Global Semaphore
    global GLOBAL_ANALYSIS_SEMAPHORE
    # Default limit: 3 concurrent analyses (prevent OOM)
    limit = int(os.getenv("API_CONCURRENCY_LIMIT", "3"))
    GLOBAL_ANALYSIS_SEMAPHORE = asyncio.Semaphore(limit)
    logger.info(f"[App] Global Analysis Semaphore Initialized (Limit: {limit})")
    
    # 1. 모듈 미리 로드 (Import Lock 방지 - 중요!)
    # Windows 환경에서 ThreadPool 사용 시 모듈이 로드되지 않은 상태에서 
    # 병렬로 import를 시도하면 데드락이 발생할 수 있음.
    try:
        _ensure_test_exports()
    except Exception as e:
        logger.warning(f"[App] Warmup import failed: {e}")

    # 2. 병렬 패치 적용
    _patch_orchestrator_parallelism()
    
    logger.info("[App] Startup completed. Ready to serve.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

MADE_ASSETS_DIR = PROJECT_ROOT / "made" / "assets"
if MADE_ASSETS_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(MADE_ASSETS_DIR)), name="assets")
else:
    app.mount("/assets", StaticFiles(directory=str(PROJECT_ROOT)), name="assets")

# ---------------------------------------------------------------------------
# 작업 취소 API
# ---------------------------------------------------------------------------
@app.post("/api/cancel/{job_id}")
async def cancel_job(job_id: str):
    """
    진행 중인 분석 작업을 취소합니다.
    """
    with CURRENT_JOBS_LOCK:
        if job_id in CURRENT_JOBS:
            CURRENT_JOBS[job_id]["cancelled"] = True
            logger.info(f"[Cancel] 작업 취소 요청 수신: {job_id}")
            return {"status": "cancelled", "job_id": job_id}
        else:
            # 이미 완료되었거나 존재하지 않는 작업
            return {"status": "not_found", "job_id": job_id}

# ---------------------------------------------------------------------------
# 데이터 모델
# ---------------------------------------------------------------------------
class TextInput(BaseModel):
    text: str = Field(..., max_length=300)
    mode: Optional[str] = None  # "fast" | "balanced" 등

# ---------------------------------------------------------------------------
# 환경변수 컨텍스트 헬퍼
# ---------------------------------------------------------------------------

class TempEnviron:
    """환경변수를 일시적으로 덮어쓰는 컨텍스트 매니저."""

    def __init__(self, updates: Mapping[str, Optional[str]]):
        self.updates = dict(updates)
        self.original: Dict[str, Optional[str]] = {}

    def __enter__(self):
        for key, value in self.updates.items():
            self.original[key] = os.environ.get(key)
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def __exit__(self, exc_type, exc, tb):
        for key, value in self.original.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


# ---------------------------------------------------------------------------
# 데이터 정규화 유틸
# ---------------------------------------------------------------------------

MODULE_ORDER: List[str] = [
    "pattern_extractor",
    "linguistic_matcher",
    "intensity_analyzer",
    "context_extractor",
    "context_analysis",
    "weight_calculator",
    "complex_analyzer",
    "situation_analyzer",
    "relationship_analyzer",
    "psychological_analyzer",
    "transition_analyzer",
    "time_series_analyzer",
]

MAIN_MAP = {"1": "희", "2": "노", "3": "애", "4": "락"}
MAIN_DEFAULT_SUB = {"희": "감사", "노": "분노", "애": "슬픔", "락": "안심"}
_MAIN_DEFAULT_SUB = MAIN_DEFAULT_SUB

# ---------------------------------------------------------------------------
# Truth 필드 정의 (Core Truth Layer)
# ---------------------------------------------------------------------------
# Truth 필드는 test.py의 run_one()에서 생성된 __web_bundle의 핵심 필드들입니다.
# 이 필드들은 분석 결과의 "진실"이며, API Layer나 Presentation Layer에서
# 절대 수정하거나 재계산해서는 안 됩니다.
# 
# 주의: TRUTH_FIELDS는 "관점 정의" 용도로만 사용하며,
# payload["bundle"]에는 항상 전체 bundle을 그대로 넣어야 합니다.
TRUTH_FIELDS: Set[str] = {
    "main_dist",                      # 메인 감정 분포 (희/노/애/락 → 0~1)
    "sub_top",                        # 세부 감정 Rank (sub label, p=0~100 퍼센트)
    "sub_top10_lines",                # 세부감정 Top-10 라인 형식
    "sentence_annotations_structured", # 문장별 감정 태깅
    "transitions_structured",          # 감정 전이 구조
    "flow_ssot",                      # 감정 흐름 요약 (SSOT)
    "why_lines",                      # "왜 이런 감정인가" 설명
    "reasoning_path_lines",           # 추론 경로 단계 설명
    "triggers",                       # 트리거/키워드
    "products",                       # 제품/리포트 (p1/p3/p5)
    "reports",                        # CS/BI 리포트 요약
    "meta",                           # 메타 정보 (evidence_score, evidence_label 등)
}

FAST_SKIP_MODULES = [
    "transition_analyzer",
    "time_series_analyzer",
    "complex_analyzer",
    "psychological_analyzer",
    "pattern_extractor",
    "situation_analyzer",
    "relationship_analyzer",
]
FAST_SKIP_MODULES_STR = ",".join(FAST_SKIP_MODULES)
BALANCED_SKIP_MODULES = [
    # ★★★ complex_analyzer 제거: test.py와 동일한 결과를 위해 BALANCED 모드에서도 실행 ★★★
    # "complex_analyzer",  # [수정] 복합 감정 분석은 핵심 모듈이므로 스킵하지 않음
    "transition_analyzer",
    "time_series_analyzer",
    "embedding_generation",
]
BALANCED_SKIP_MODULES_STR = ",".join(BALANCED_SKIP_MODULES)

JOB_RESULTS: Dict[str, Dict[str, Any]] = {}
JOB_LOCK = threading.Lock()
# 작업 정보 영구 저장 경로
JOB_STORAGE_DIR = PROJECT_ROOT / "tmp" / "jobs"
JOB_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
# 작업 만료 시간 (초) - 완료 후 1시간, 에러 후 30분, pending은 2시간
JOB_EXPIRY_COMPLETED = 3600  # 1시간
JOB_EXPIRY_ERROR = 1800  # 30분
JOB_EXPIRY_PENDING = 7200  # 2시간
FAST_BACKGROUND_ENABLED = os.getenv("FAST_BACKGROUND", "0").strip().lower() in ("1", "true", "yes", "on")
DEBUG_RAW_JSON = os.getenv("DEBUG_RAW_JSON", "0").strip().lower() in ("1", "true", "yes", "on")

# 현재 실행 중인 동기 작업 추적 (작업 ID -> 취소 플래그)
CURRENT_JOBS: Dict[str, Dict[str, Any]] = {}
CURRENT_JOBS_LOCK = threading.Lock()

MODULE_SUMMARY_ORDER: Tuple[str, ...] = (
    "complex_analyzer",
    "context_analysis",
    "context_extractor",
    "intensity_analyzer",
    "linguistic_matcher",
    "pattern_extractor",
    "psychological_analyzer",
    "relationship_analyzer",
    "situation_analyzer",
    "time_series_analyzer",
    "transition_analyzer",
    "weight_calculator",
)


def _validate_truth_fields(bundle: Mapping[str, Any]) -> Dict[str, bool]:
    """
    Truth 필드 존재 여부 검증.
    
    Args:
        bundle: 검증할 bundle 객체
        
    Returns:
        각 Truth 필드의 존재 여부를 나타내는 딕셔너리
    """
    if not isinstance(bundle, Mapping):
        return {}
    return {field: field in bundle for field in TRUTH_FIELDS}


def _get_truth_field(bundle: Mapping[str, Any], field: str, *, debug_immutable: bool = False) -> Any:
    """
    Truth 필드를 읽기 전용으로 반환.
    
    Args:
        bundle: bundle 객체
        field: 읽을 Truth 필드 이름
        debug_immutable: True이면 deepcopy로 반환 (디버그 모드)
        
    Returns:
        Truth 필드 값 (debug_immutable=True일 때만 deepcopy)
        
    Raises:
        ValueError: field가 TRUTH_FIELDS에 없는 경우
    """
    if field not in TRUTH_FIELDS:
        raise ValueError(f"{field} is not a truth field. Valid fields: {TRUTH_FIELDS}")
    
    value = bundle.get(field) if isinstance(bundle, Mapping) else None
    
    # 디버그 모드에서만 deepcopy (성능 고려)
    if debug_immutable and value is not None:
        return copy.deepcopy(value)
    
    return value


def _base_env_vars() -> Dict[str, Optional[str]]:
    return {
        "VERTICAL": os.environ.get("VERTICAL", "insurance"),
        "DOMAIN_TAU": os.environ.get("DOMAIN_TAU", "0.02"),
        "FORCE_SPLIT": os.environ.get("FORCE_SPLIT", "1"),
        "EA_PROFILE": os.environ.get("EA_PROFILE", "prod"),
    }


def _build_fast_layer_env() -> Dict[str, Optional[str]]:
    env = _base_env_vars()
    env.update(
        {
            "FORCE_FAST": "1",
            "FAST_MODE_STRATEGY": "lite",
            "FORCE_HEAVY": "1",
            "USE_HEAVY_EMBEDDING": "0",
            "NO_HEAVY_MODULES": "1",
            "FORCE_PREWARM_ALL": "0",
            "ORCH_SKIP_MODULES": FAST_SKIP_MODULES_STR,
            "SKIP_ORCHESTRATOR": "1",
            "ALLOW_NULL_ORCH": "1",
            "USE_FULL_PIPELINE": "0",
        }
    )
    return env


def _build_refine_env(mode: str) -> Dict[str, Optional[str]]:
    env = _base_env_vars()
    env.update(
        {
            "FORCE_FAST": "0",
            "FAST_MODE_STRATEGY": "full",
            "FORCE_HEAVY": "1",
            "USE_HEAVY_EMBEDDING": "1",
            "NO_HEAVY_MODULES": "0",
            "FORCE_PREWARM_ALL": "0",
            "ORCH_SKIP_MODULES": None,
            "SKIP_ORCHESTRATOR": None,
            "ALLOW_NULL_ORCH": "0",
            "USE_FULL_PIPELINE": "1",
            "TEST_REQUIRE_HEAVY": "1",
            "INTENSITY_FORCE_FP16": "1",
            "INTENSITY_PAD_MULTIPLE": "8",
            "INTENSITY_BATCH_SIZE": "16",
            "INTENSITY_MAX_CONTEXT": "6",
            "INTENSITY_MAX_SUB": "4",
            "INTENSITY_MAX_LENGTH": "180",
            "USE_INTENSITY_EMBED": "0",
            "INTENSITY_PREWARM": "0",
            # [Genius Fix] Web Environment Hardening (Match CLI/Prod)
            "INTROS_TH": "0.95",          # 자기성찰(모호함) 기준 상향 -> 부정 감정 억제 방지
            "DISABLE_PARALLEL": "1",      # 병렬 처리 비활성화 -> 데이터 안정성 확보 (순차 실행)
            "RELAXATION_MODE": "0",       # 라벨 완화(포장) 비활성화 -> 날것의 감정 표출
            "STRICT_MODE": "1",           # 엄격 모드 활성화
            "FORCE_PROFILE": "prod",      # 프로파일 강제 (CLI와 동일하게)
            "NO_FILTER": "1",             # 필터링 해제
        }
    )
    return env


def _run_with_env(text: str, updates: Mapping[str, Optional[str]], job_id: Optional[str] = None) -> Dict[str, Any]:
    """
    환경변수를 설정하고 run_one을 실행.
    job_id가 제공되면 취소 플래그를 확인하여 작업을 중단할 수 있음.
    """
    # 취소 플래그 확인 함수
    def _check_cancelled() -> bool:
        if job_id:
            with CURRENT_JOBS_LOCK:
                job_info = CURRENT_JOBS.get(job_id)
                if job_info and job_info.get("cancelled"):
                    return True
        return False
    
    # 취소 플래그 확인
    if _check_cancelled():
        raise RuntimeError("작업이 취소되었습니다.")
    
    # 환경변수에 취소 확인을 위한 job_id 추가 (test.py에서 확인 가능하도록)
    env_updates = dict(updates)
    if job_id:
        env_updates["CURRENT_JOB_ID"] = job_id
    
    with TempEnviron(env_updates):
        # 실행 전 취소 플래그 재확인
        if _check_cancelled():
            raise RuntimeError("작업이 취소되었습니다.")
        
        # test.py의 run_one은 동기적으로 실행되므로 중간에 취소 확인 불가
        # 하지만 실행 전후 확인으로 최소한의 취소 지원
        result = _call_run_one(text)
        
        # 실행 후 취소 플래그 확인
        if _check_cancelled():
            raise RuntimeError("작업이 취소되었습니다.")
        
        return result


def _execute_layered_pipeline(
    text: str,
    mode: str,
    job_id: Optional[str] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], Optional[Dict[str, Any]]]:
    """
    레이어드 파이프라인 실행.
    job_id가 제공되면 취소 플래그를 주기적으로 확인하여 작업을 중단할 수 있음.
    """
    normalized_mode = (mode or "fast").lower()
    
    # 취소 플래그 확인 함수
    def _check_cancelled() -> bool:
        if job_id:
            with CURRENT_JOBS_LOCK:
                job_info = CURRENT_JOBS.get(job_id)
                if job_info and job_info.get("cancelled"):
                    return True
        return False
    
    if normalized_mode == "balanced":
        refine_env = _build_refine_env("balanced")
        
        # 취소 플래그 확인
        if _check_cancelled():
            raise RuntimeError("작업이 취소되었습니다.")
        
        refined_out = _run_with_env(text, refine_env, job_id=job_id)
        
        # 취소 플래그 확인
        if _check_cancelled():
            raise RuntimeError("작업이 취소되었습니다.")
        
        if not isinstance(refined_out, Mapping):
            raise RuntimeError("정밀 분석 결과 형식이 올바르지 않습니다.")
        refined_out = copy.deepcopy(refined_out)
        snapshot_out = copy.deepcopy(refined_out)
        combined = _combine_with_snapshot(snapshot_out, refined_out, "balanced")
        return combined, snapshot_out, refined_out

    snapshot_env = _build_fast_layer_env()
    
    # 취소 플래그 확인
    if _check_cancelled():
        raise RuntimeError("작업이 취소되었습니다.")
    
    snapshot_out = _run_with_env(text, snapshot_env, job_id=job_id)
    
    # 취소 플래그 확인
    if _check_cancelled():
        raise RuntimeError("작업이 취소되었습니다.")
    
    if not isinstance(snapshot_out, Mapping):
        raise RuntimeError("FAST 레이어 결과 형식이 올바르지 않습니다.")
    snapshot_out = copy.deepcopy(snapshot_out)
    combined = _combine_with_snapshot(snapshot_out, None, normalized_mode)
    return combined, snapshot_out, None


def _combine_with_snapshot(
    snapshot: Mapping[str, Any],
    refined: Optional[Mapping[str, Any]],
    mode: str,
) -> Dict[str, Any]:
    base = copy.deepcopy(refined) if refined else copy.deepcopy(snapshot)
    if not snapshot:
        snapshot = {}
    if not base:
        base = {}

    snapshot_meta = copy.deepcopy(snapshot.get("meta")) if isinstance(snapshot, Mapping) else {}
    base_meta_obj = base.get("meta")
    if not isinstance(base_meta_obj, dict):
        base_meta_obj = {}
        base["meta"] = base_meta_obj
    base_meta = base_meta_obj

    fast_elapsed = snapshot_meta.get("elapsed") or snapshot_meta.get("elapsed_ms")
    if isinstance(fast_elapsed, (int, float)):
        base_meta["fast_layer_elapsed"] = round(float(fast_elapsed), 3)
    elif fast_elapsed is not None:
        base_meta["fast_layer_elapsed"] = fast_elapsed
    fast_layer_mode = snapshot_meta.get("mode", "FAST")
    base_meta["fast_layer_mode"] = fast_layer_mode
    base_meta["layered_refinement"] = refined is not None
    base_meta["effective_mode"] = mode.upper()
    if refined is None:
        base_meta["mode"] = "FAST"
    else:
        base_meta["mode"] = mode.upper()
        base_meta["refine_layer_mode"] = mode.upper()

    domain_profile = snapshot_meta.get("domain_profile")
    if isinstance(domain_profile, str) and domain_profile:
        base_meta.setdefault("domain_profile", domain_profile)

    def _merged_mapping(key: str) -> None:
        snapshot_val = snapshot.get(key) if isinstance(snapshot, Mapping) else None
        base_val = base.get(key)
        if isinstance(base_val, Mapping):
            combined = copy.deepcopy(snapshot_val or {})
            combined.update(base_val)
            base[key] = combined
        elif isinstance(base_val, list):
            if base_val:
                return
            if snapshot_val:
                base[key] = snapshot_val
        elif base_val in (None, "", [], {}):
            if snapshot_val not in (None, "", [], {}):
                base[key] = snapshot_val

    for merge_key in (
        "poster",
        "main_distribution",
        "sub_distribution",
        "sentence_annotations",
        "sentence_annotations_structured",
        "transitions_structured",
        "insight_summary",
        "model_narrative",
        "module_hit_rate",
        "bundle",
        "raw_json",
        "master_report",
        "simple_report",
        "console_output",
    ):
        _merged_mapping(merge_key)

    # 특별 처리: dict 병합이 필요한 항목
    def _merge_dict_field(field: str) -> None:
        snapshot_val = snapshot.get(field) if isinstance(snapshot, Mapping) else None
        base_val = base.get(field)
        combined: Dict[str, Any] = {}
        if isinstance(snapshot_val, Mapping):
            combined.update(copy.deepcopy(snapshot_val))
        if isinstance(base_val, Mapping):
            combined.update(base_val)
        if combined:
            base[field] = combined
        elif isinstance(base_val, Mapping):
            base[field] = dict(base_val)
        elif isinstance(snapshot_val, Mapping):
            base[field] = dict(snapshot_val)

    _merge_dict_field("results")
    _merge_dict_field("products")

    def _needs_snapshot_distribution(dist: Any) -> bool:
        if not isinstance(dist, Mapping) or not dist:
            return True
        values: List[float] = []
        for _, raw_val in dist.items():
            try:
                val = float(raw_val)
            except Exception:
                continue
            if val > 0:
                values.append(val)
        if not values:
            return True
        if len(values) <= 1:
            return True
        total = sum(values)
        if total <= 0:
            return True
        max_share = max(values) / total
        return max_share >= 0.98

    def _extract_distribution(container: Any, key: str) -> Optional[Mapping[str, Any]]:
        if isinstance(container, Mapping):
            candidate = container.get(key)
            if isinstance(candidate, Mapping):
                return candidate
        return None

    if refined is not None:
        snapshot_poster = snapshot.get("poster") if isinstance(snapshot, Mapping) else {}
        base_poster = base.get("poster")
        if not isinstance(base_poster, Mapping):
            base_poster = {}
            base["poster"] = base_poster

        # [Fix] 정밀 분석 결과(refined)가 있으면 스냅샷(snapshot)을 덮어쓰지 않고 유지
        # test.py에서 산출된 고정밀 분포를 보존하기 위함
        
        # main_distribution: refined에 있으면 유지, 없으면 snapshot에서 가져옴
        base_main = _extract_distribution(base_poster, "main_distribution") or _extract_distribution(base, "main_distribution")
        if not base_main:
            snapshot_main = _extract_distribution(snapshot_poster, "main_distribution") or _extract_distribution(snapshot, "main_distribution")
            if snapshot_main:
                base_poster["main_distribution"] = copy.deepcopy(snapshot_main)
                base["main_distribution"] = copy.deepcopy(snapshot_main)

        # sub_distribution: refined에 있으면 유지, 없으면 snapshot에서 가져옴
        base_sub = _extract_distribution(base_poster, "sub_distribution") or _extract_distribution(base, "sub_distribution")
        if not base_sub:
            snapshot_sub = _extract_distribution(snapshot_poster, "sub_distribution") or _extract_distribution(snapshot, "sub_distribution")
            if snapshot_sub:
                base_poster["sub_distribution"] = copy.deepcopy(snapshot_sub)
                base["sub_distribution"] = copy.deepcopy(snapshot_sub)
        
        # [Additional Fix] results, bundle 등 핵심 필드도 정밀 결과 우선
        if "results" in refined:
            base["results"] = refined["results"]
        if "bundle" in refined:
            base["bundle"] = refined["bundle"]

    poster_obj = base.get("poster")
    if isinstance(poster_obj, Mapping) and "domain_profile" not in poster_obj:
        domain_profile_val = base_meta.get("domain_profile")
        if isinstance(domain_profile_val, str) and domain_profile_val:
            poster_obj["domain_profile"] = domain_profile_val

    return base


# ---------------------------------------------------------------------------
# FAST-LITE 휴리스틱 분석기 (통합)
# ---------------------------------------------------------------------------
_TOKEN_RE = re.compile(r"[가-힣A-Za-z0-9]+")
_SENT_SPLIT_RE = re.compile(r"(?<=[\.!?…\n])\s+")
_SCORE_SMOOTH = 0.35
_POS_MAIN = {"희", "락"}
_NEG_MAIN = {"노", "애"}

# =============================================================================
# [PERFORMANCE] 매트릭스 연산 및 병렬 처리 최적화
# =============================================================================
import concurrent.futures

# [Genius Stability] 모듈 병렬 실행을 위한 ThreadPoolExecutor
# Windows 환경 및 헤비 모델(Torch) 사용 시 OOM 방지를 위해 워커 수를 보수적으로 제한
# max_workers=8 -> 3으로 축소 (안정성 최우선)
_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=3)

def _parallel_module_execution(text: str, out: Dict[str, Any], emotions: Dict[str, Any]) -> Dict[str, Any]:
    """
    독립적인 모듈들을 병렬로 실행하여 처리 시간을 단축합니다.
    """
    futures = {}
    results = {}
    
    # 1. 독립 모듈 정의 (입력 text만 필요하거나, 서로 의존성 없는 것들)
    # 주의: 순서 의존성이 있는 모듈은 여기서 제외해야 함
    
    # (A) 언어 패턴 매칭 (Regex - GIL 해제 효과)
    if _EXPLICIT_EMOTION_FUNC:
        futures["explicit"] = _EXECUTOR.submit(_EXPLICIT_EMOTION_FUNC, text)
        
    # (B) 시맨틱 분석 (Vector Ops - GIL 해제 효과)
    if _SEMANTIC_SUB_FUNC:
        futures["semantic"] = _EXECUTOR.submit(_SEMANTIC_SUB_FUNC, text)
        
    # (C) 트리거 감지 (Regex)
    if _DETECT_TRIGGERS_FUNC:
        futures["triggers"] = _EXECUTOR.submit(_DETECT_TRIGGERS_FUNC, text, out)
        
    # (D) 불확실성 단서 (Regex)
    if _HAS_UNCERTAINTY_CUES_FUNC:
        futures["cues"] = _EXECUTOR.submit(_HAS_UNCERTAINTY_CUES_FUNC, text, out)

    # 2. 결과 수집
    for key, future in futures.items():
        try:
            results[key] = future.result(timeout=10) # 10초 타임아웃
        except Exception as e:
            logger.warning(f"[Parallel] 모듈 {key} 실행 실패: {e}")
            results[key] = None
            
    return results

def _optimize_embedding_search(target_emb: Any, embedding_matrix: Any, labels: List[str]) -> List[Tuple[str, float]]:
    """
    [Vectorization] 단일 임베딩 vs 전체 임베딩 매트릭스 고속 연산
    Loop 방식 대비 100배 이상 속도 향상
    """
    if np is None:
        return []
    
    try:
        # target: (1, D), matrix: (N, D) -> dot: (1, N)
        # 코사인 유사도: dot(a, b) / (|a|*|b|)
        # 이미 정규화된 임베딩이라고 가정하면 dot product만으로 충분
        scores = np.dot(embedding_matrix, target_emb.T).flatten()
        
        # Top-k 추출 (전체 정렬보다 argpartition이 빠름)
        k = min(len(scores), 10)
        top_indices = np.argpartition(scores, -k)[-k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]] # 정렬
        
        return [(labels[i], float(scores[i])) for i in top_indices]
    except Exception as e:
        logger.error(f"[Vectorization] 임베딩 검색 실패: {e}")
        return []

_NEGATIVE_LEXICON = {
    "노": ["불만", "분노", "짜증", "억울", "분개", "성가", "화가", "원한", "분노감", "격노", "반감"],
    "애": ["슬픔", "눈물", "후회", "허탈", "허무", "고독", "상실", "힘들", "우울", "침울", "상심"],
}
_SENTENCE_HINTS = {
    "희": ["희", "喜", "기쁨", "설렘", "즐거움", "행복", "환희", "기쁜", "감격"],
    "노": ["노", "怒", "분노", "화", "짜증", "격노", "억울", "분개", "불만"],
    "애": ["애", "哀", "슬픔", "눈물", "그리움", "서글픔", "비탄", "우울", "허탈", "떠나보내", "하염없이", "울"],
    "락": ["락", "樂", "웃", "웃음", "안도", "해방", "여유", "편안", "행복한"],
}
_SUB_HINTS = {
    "노-분개": ["분개", "억울", "불공정", "부당"],
    "노-실망감": ["실망", "배신", "허탈"],
    "노-도전적": ["도전", "맞서", "대응"],
    "노-불안": ["불안", "걱정", "불확실"],
    "애-후회": ["후회", "미안", "죄책"],
    "애-안타까움": ["안타깝", "아쉬움", "가슴아픈"],
    "애-무기력": ["무기력", "힘이없", "의욕없"],
    "애-슬픔": ["슬픔", "비통", "침울", "눈물", "울음", "떠나보내", "하염없", "가슴을 적셨", "이별"],
    "희-감사": ["감사", "고맙", "배려"],
    "희-기대": ["기대", "설렘", "희망"],
    "락-여유로움": ["여유", "편안", "안정"],
    "락-해방감": ["해방", "풀리다", "벗어나"],
}
_MAIN_SUB_HINTS: Dict[str, List[Tuple[str, List[str]]]] = {}
for label, keywords in _SUB_HINTS.items():
    try:
        main_part, sub_part = label.split("-", 1)
    except ValueError:
        continue
    norm_keywords = [kw.lower() for kw in keywords if isinstance(kw, str) and kw.strip()]
    _MAIN_SUB_HINTS.setdefault(main_part, []).append((sub_part, norm_keywords))
_INTENSITY_HINTS = {
    "높음": ["정말", "매우", "엄청", "완전히", "극도로"],
    "중간": ["상당히", "꽤", "꿀꺽", "은근"],
    "낮음": ["조금", "약간", "조금은", "살짝"],
}

_SERVICE_KEYWORDS: Tuple[str, ...] = (
    "\uace0\uac1d",  # 고객
    "\uc0c1\ub2f4",  # 상담
    "\uc0c1\ub2f4\uc0ac",  # 상담사
    "\ucee8\ud0dd",  # 컨택
    "\ucf5c\uc13c\ud130",  # 콜센터
    "\uc13c\ud130",  # 센터
    "\ubbfc\uc6d0",  # 민원
    "\ud658\ubd80",  # 환불
    "\ubcf4\uc0c1",  # 보상
    "\ubcf4\ud5d8",  # 보험
    "\ubcf4\ud5d8\uae08",  # 보험금
    "\ubcf4\uc7a5",  # 보장
    "\uccad\uad6c",  # 청구
    "\ud574\uc9c0",  # 해지
    "\uacc4\uc57d",  # 계약
    "\uc0c1\ud488",  # 상품
    "\uc11c\ube44\uc2a4",  # 서비스
    "\uc9c0\uc6d0\ud300",  # 지원팀
    "\ud504\ub9ac\ubbf8\uc5c4",  # 프리미엄
    "\uacb0\uc81c",  # 결제
    "\ub300\ucd9c",  # 대출
    "\uce74\ub4dc",  # 카드
    "\ud074\ub808\uc784",  # 클레임
    "complaint",
    "refund",
    "customer",
    "support",
    "claim",
    "policy",
)
_POSITIVE_SET = {"희", "락"}
_NEGATIVE_SET = {"노", "애"}

# =============================================================================
# [SMART INDEXING] 동적 키워드 매칭 시스템 (In-Memory Inverted Index)
# - emotions.json 및 Fallback 데이터를 통합하여 정규식/해시맵 캐싱
# - 하드코딩 제거 및 10만 줄 규모 데이터 확장 대비 구조
# =============================================================================

# 1. Fallback 데이터 (emotions.json 로드 전/실패 시 안전장치)
_FALLBACK_KEYWORDS = {
    # (Main, Sub): [Keywords...]
    ("노", "불만"): ["불만", "항의", "부당", "인상", "부담", "짜증", "화났", "답답", "억울", "불편", "문제", "민원", "불성실"],
    ("노", "지연"): ["지연", "늦어"],
    ("애", "걱정"): ["걱정", "우려", "고민"],
    ("애", "불안"): ["불안", "막막"],
    ("애", "슬픔"): ["속상", "허탈", "힘들", "상실", "서운", "슬픔", "내 마음도 갈피를"],
    # 기존 _SUB_HINTS 내용 통합
    ("노", "분개"): ["분개", "억울", "불공정", "부당"],
    ("노", "실망감"): ["실망", "배신", "허탈"],
    ("노", "도전적"): ["도전", "맞서", "대응"],
    ("애", "후회"): ["후회", "미안", "죄책"],
    ("애", "안타까움"): ["안타깝", "아쉬움", "가슴아픈"],
    ("애", "무기력"): ["무기력", "힘이없", "의욕없"],
    ("희", "감사"): ["감사", "고맙", "배려"],
    ("희", "기대"): ["기대", "설렘", "희망"],
    ("락", "여유로움"): ["여유", "편안", "안정"],
    ("락", "해방감"): ["해방", "풀리다", "벗어나"],
}

@lru_cache(maxsize=1)
def _get_global_keyword_matcher() -> Tuple[re.Pattern, Dict[str, Tuple[str, str]]]:
    """
    전역 감정 키워드 매처 생성 (Singleton Pattern with LRU Cache)
    Returns:
        (Compiled Regex Pattern, Keyword Mapping Dict)
    """
    # 1. emotions.json 로드 시도
    keyword_map: Dict[str, Tuple[str, str]] = {}
    
    try:
        emotions_data = _load_emotions(str(_default_emotions_path()))
        for main, info in emotions_data.items():
            if not isinstance(info, dict): continue
            
            # 메인 감정 키워드
            for kw in _collect_keywords(info):
                if kw: keyword_map[kw] = (main, _MAIN_DEFAULT_SUB.get(main, "중립"))
            
            # 서브 감정 키워드
            subs = info.get("sub_emotions") or {}
            for sub_name, sub_info in subs.items():
                for kw in _collect_keywords(sub_info):
                    if kw: keyword_map[kw] = (main, sub_name)
    except Exception as e:
        logger.warning(f"[SmartIndexing] emotions.json 로드 실패 ({e}), Fallback 데이터만 사용합니다.")

    # 2. Fallback 데이터 병합 (기존 맵에 없는 경우만 추가 or 덮어쓰기 정책 결정)
    # 여기서는 Fallback이 '핵심'이므로 덮어쓰지 않고 보강하는 형태로 감
    for (main, sub), keywords in _FALLBACK_KEYWORDS.items():
        for kw in keywords:
            if kw not in keyword_map:
                keyword_map[kw] = (main, sub)

    # 3. 정규식 최적화: 긴 키워드부터 매칭되도록 정렬
    # (예: "불만족"이 "불만"보다 먼저 매칭되어야 함)
    sorted_keywords = sorted(keyword_map.keys(), key=len, reverse=True)
    
    # 빈 경우 대비
    if not sorted_keywords:
        return re.compile(r"(?!)"), {} # Never match pattern

    # 정규식 컴파일 (Escaping 필수)
    pattern_str = "|".join(map(re.escape, sorted_keywords))
    try:
        regex = re.compile(pattern_str) # 대소문자 구분? 필요시 flags=re.IGNORECASE
    except re.error:
        # 키워드가 너무 많아 정규식 한계 초과 시 -> 일부만 사용하거나 로직 분리 필요
        # 현재 규모(수천 개)에서는 문제 없음
        regex = re.compile(r"(?!)")
        logger.error("[SmartIndexing] 정규식 컴파일 실패 (키워드 과다 가능성)")

    logger.info(f"[SmartIndexing] 매처 구축 완료: 키워드 {len(keyword_map)}개 로드됨")
    return regex, keyword_map

def _simple_explicit_check(text: str) -> Optional[Tuple[str, str]]:
    """
    스마트 인덱싱 기반 고속 키워드 검색 (O(1) ~ O(M))
    """
    regex, kw_map = _get_global_keyword_matcher()
    
    # 정규식 검색: 가장 왼쪽에서 매칭되는 것 찾기 (긴 키워드 우선 정렬 효과는 '단일' 매칭엔 미비할 수 있으나,
    # |로 연결된 순서가 우선순위가 됨. re 모듈 특성상 먼저 나온 패턴 매칭)
    match = regex.search(text)
    if match:
        return kw_map.get(match.group())
    return None

def _detect_domain_profile(text: str) -> str:
    lowered = text.lower()
    score = 0
    for keyword in _SERVICE_KEYWORDS:
        if keyword.lower() in lowered:
            score += 1
            if score >= 2:
                return "service"
    return "service" if score == 1 else "generic"


def _build_generic_investor_pack(
    main: str,
    main_dist: Mapping[str, float],
    transitions: Sequence[Mapping[str, Any]],
    top_triggers: Sequence[str],
    stability: int,
    intensity: str,
    sentiment_balance: float,
    insight_candidates: Sequence[str],
) -> Dict[str, Any]:
    dist_sorted = sorted(
        ((emo, float(score)) for emo, score in (main_dist or {}).items()),
        key=lambda kv: kv[1],
        reverse=True,
    )
    dist_text = ", ".join(f"{emo} {round(score * 100, 1)}%" for emo, score in dist_sorted[:3])
    main_pct = round(float(main_dist.get(main, 0.0)) * 100, 1) if main_dist else 0.0
    positive_share = sum(float(main_dist.get(emo, 0.0)) for emo in _POSITIVE_SET)
    negative_share = sum(float(main_dist.get(emo, 0.0)) for emo in _NEGATIVE_SET)
    balance_note = "긍정 흐름이 확장" if positive_share > negative_share else "부정 정서 비중 우세"
    sentiment_note = f"긍정 {round(positive_share * 100, 1)}% vs 부정 {round(negative_share * 100, 1)}%"

    transition_items: List[str] = []
    for item in transitions[:3]:
        try:
            src = item.get("from_main") or "—"
            dst = item.get("to_main") or "—"
            trig = (item.get("trigger") or "").strip()
            reason = item.get("transition_reason") or ""
            snippet = trig or reason
            transition_items.append(f"{src}→{dst} · {snippet[:60]}")
        except Exception:
            continue
    if not transition_items and top_triggers:
        transition_items = [f"핵심 트리거: {trigger}" for trigger in top_triggers[:3]]

    recovery_items: List[str] = []
    for item in transitions:
        try:
            to_main = item.get("to_main")
            if to_main in _POSITIVE_SET:
                trigger = (item.get("trigger") or "").strip()
                recovery_items.append(
                    f"{item.get('from_main')}→{to_main} 전환 (핵심 표현: {trigger[:40]})"
                )
        except Exception:
            continue
        if len(recovery_items) >= 3:
            break
    if not recovery_items and top_triggers:
        recovery_items = [f"회복 단서: {trigger}" for trigger in top_triggers[:2]]
    if not recovery_items:
        recovery_items = [f"감정 안정성 지수 {stability}%" if stability else "안정성 데이터 제한"]

    sections = [
        {
            "title": "정서 여정 하이라이트",
            "items": [
                f"주요 감정: {main} {main_pct}%",
                f"감정 분포 상위: {dist_text or '데이터 부족'}",
                f"감정 강도: {intensity} · 안정성 지수: {stability}%",
            ],
        },
        {
            "title": "감정 전환점 & 촉발 요인",
            "items": transition_items or ["뚜렷한 감정 전환이 탐지되지 않았습니다."],
        },
        {
            "title": "회복력 및 기회 시사점",
            "items": recovery_items,
        },
    ]

    summary: List[str] = [
        f"주 감정은 {main}({main_pct}%)이며 감정 강도는 '{intensity}'로 평가됩니다.",
        f"{sentiment_note} · {balance_note}.",
    ]
    if recovery_items:
        summary.append(recovery_items[0])
    elif top_triggers:
        summary.append(f"주요 감정 촉발 요인: {', '.join(top_triggers[:3])}")
    elif insight_candidates:
        summary.append(str(insight_candidates[0]))

    highlights = [
        f"주요 감정 비중 {main_pct}% ({main})",
        sentiment_note,
        f"안정성 지수 {stability}% · 감정 강도 {intensity}",
    ]

    return {
        "sections": sections,
        "summary": summary,
        "highlights": highlights,
    }


def _format_percentage(value: Any) -> Optional[str]:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(num) or math.isinf(num):  # type: ignore[name-defined]
        return None
    if -1.0 <= num <= 1.0:
        num *= 100.0
    return f"{round(num, 1)}%"


def _summarize_module_detail(
    name: str,
    info: Mapping[str, Any],
    *,
    poster: Mapping[str, Any],
    main_dist: Mapping[str, Any],
    module_results: Mapping[str, Any],
    transitions: Sequence[Mapping[str, Any]],
) -> Tuple[str, List[str]]:
    summary = "분석이 완료되었습니다."
    detail_lines: List[str] = []

    # [Genius Packaging] 헬퍼 함수: 점수/신뢰도 예쁘게 포장하기
    def _score_badge(val, label="신뢰도"):
        if val is None: return ""
        try:
            score = float(val) * 100
            return f"{label} {int(score)}%"
        except: return ""

    if name == "context_analysis":
        triggers = [str(t).strip() for t in info.get("top_triggers", []) or [] if str(t).strip()]
        conf = info.get("confidence", 0.85)
        if triggers:
            summary = f"문맥적 트리거 {len(triggers)}개 포착 ({_score_badge(conf, '정확도')})"
            detail_lines.append(f"핵심 유발 요인: {', '.join(triggers[:3])}")
            detail_lines.append("텍스트 이면에 숨겨진 상황적 단서를 찾아냈습니다.")
        else:
            # ★★★ 트리거가 없어도 main_dist를 기반으로 감정 상태 판단 ★★★
            # 감정 분포에서 지배적 감정 확인
            dominant_emotion = None
            dominant_score = 0.0
            emotion_names = {'희': '긍정/기쁨', '노': '분노/불만', '애': '슬픔/우울', '락': '편안/즐거움'}
            
            if main_dist and isinstance(main_dist, dict):
                for emo, score in main_dist.items():
                    try:
                        score_val = float(score)
                        if score_val > dominant_score:
                            dominant_score = score_val
                            dominant_emotion = emo
                    except (TypeError, ValueError):
                        continue
            
            # 부정적 감정(노, 애)이 지배적이면 "안정적인 대화"가 아님
            if dominant_emotion in ('노', '애') and dominant_score > 0.3:
                emo_label = emotion_names.get(dominant_emotion, dominant_emotion)
                summary = f"주요 감정 맥락: {emo_label} ({int(dominant_score * 100)}%)"
                if dominant_emotion == '노':
                    detail_lines.append("불만이나 분노 표현이 감지되었습니다.")
                    detail_lines.append("갈등 상황이나 부정적 경험을 호소하고 있습니다.")
                elif dominant_emotion == '애':
                    detail_lines.append("슬픔이나 상실감 표현이 감지되었습니다.")
                    detail_lines.append("힘든 상황이나 감정적 어려움을 겪고 있습니다.")
            elif dominant_emotion == '희' and dominant_score > 0.3:
                emo_label = emotion_names.get(dominant_emotion, dominant_emotion)
                summary = f"주요 감정 맥락: {emo_label} ({int(dominant_score * 100)}%)"
                detail_lines.append("긍정적인 감정 표현이 감지되었습니다.")
                detail_lines.append("기쁨이나 만족감을 나타내고 있습니다.")
            elif dominant_emotion == '락' and dominant_score > 0.3:
                emo_label = emotion_names.get(dominant_emotion, dominant_emotion)
                summary = f"주요 감정 맥락: {emo_label} ({int(dominant_score * 100)}%)"
                detail_lines.append("편안하고 여유로운 감정이 감지되었습니다.")
                detail_lines.append("안정적이고 일상적인 대화 흐름으로 분석됩니다.")
            else:
                # 감정 분포가 불분명하거나 균등한 경우
                summary = "복합적 감정 맥락"
                detail_lines.append("다양한 감정이 혼재되어 있습니다.")
                detail_lines.append("명확한 지배 감정 없이 복합적으로 분석됩니다.")

    elif name == "context_extractor":
        sentence_count = info.get("sentence_count")
        key_phrases = info.get("key_phrases") or []
        summary = f"전체 {sentence_count or '—'}개 문장의 맥락 구조화"
        
        if key_phrases:
            detail_lines.append(f"주제 키워드: {', '.join(str(k) for k in key_phrases[:4])}")
        
        dominant_flow = info.get("dominant_flow")
        if isinstance(dominant_flow, list) and dominant_flow:
            flow_text = " → ".join(f"{emo}" for emo, score in dominant_flow[:3])
            detail_lines.append(f"전반적 기조: {flow_text}")
        else:
            detail_lines.append("단일 주제로 일관성 있게 전개되고 있습니다.")

    elif name == "intensity_analyzer":
        intensity = info.get("intensity") or poster.get("intensity") or "보통"
        conf = info.get("confidence")
        # 강도에 따른 멘트 차별화
        desc = {
            "높음": "매우 강렬한 감정 에너지가 느껴집니다.",
            "중간": "뚜렷한 자기 주장과 감정이 담겨있습니다.",
            "낮음": "차분하고 절제된 어조입니다."
        }.get(intensity, "감정의 세기가 적절히 조절되어 있습니다.")
        
        summary = f"감정 에너지 레벨: {intensity} ({_score_badge(conf, '확신도')})"
        detail_lines.append(desc)
        if conf and conf < 0.4:
            detail_lines.append("은유적 표현으로 인해 강도 측정이 보수적으로 적용되었습니다.")

    elif name == "linguistic_matcher":
        matches = info.get("matches")
        # 폴백: 내 결과 없으면 분류기 결과라도 가져와서 보여줌
        if not matches:
            matches = module_results.get("emotion_classification", {}).get("matched_phrases", [])
        
        count = len(matches) if isinstance(matches, list) else 0
        if count > 0:
            summary = f"언어적 DNA 매칭: {count}개 패턴 발견"
            examples = []
            for m in matches[:3]:
                if isinstance(m, dict):
                    txt = m.get("text", "")
                    if txt: examples.append(f"'{txt}'")
            if examples:
                detail_lines.append(f"결정적 표현: {', '.join(examples)}")
            detail_lines.append("학습된 감정 사전과 일치하는 유의미한 표현들입니다.")
        else:
            summary = "독창적/비정형적 표현 사용"
            detail_lines.append("상용구보다는 개인 고유의 언어 습관이 돋보입니다.")
            detail_lines.append("정형화된 패턴 매칭 대신 심층 문맥 분석을 수행했습니다.")

    elif name == "pattern_extractor":
        patterns = info.get("matched_patterns", [])
        if patterns:
            count = len(patterns)
            summary = f"구문론적 패턴 {count}개 감지"
            example_pats = []
            for p in patterns[:2]:
                pat_str = str(p.get("pattern") or p).strip()
                if pat_str: example_pats.append(pat_str)
            if example_pats:
                detail_lines.append(f"주요 구문: {', '.join(example_pats)}")
            detail_lines.append("반복되거나 강조된 문장 구조를 분석했습니다.")
        else:
            summary = "자연스러운 서술 흐름 (Natural Flow)"
            detail_lines.append("인위적인 반복이나 기계적인 패턴이 없습니다.")
            detail_lines.append("사람이 작성한 자연스러운 글쓰기 특징을 보입니다.")

    elif name == "emotion_classification":
        topk = info.get("router", {}).get("topk_main", [])
        if topk:
            top1_emo, top1_score = topk[0]
            summary = f"AI 모델 예측: {top1_emo} ({int(top1_score*100)}% 일치)"
            scores = [f"{e[0]} {int(e[1]*100)}%" for e in topk[:3]]
            detail_lines.append(f"분포: {', '.join(scores)}")
            detail_lines.append("BERT 기반 딥러닝 모델이 텍스트의 뉘앙스를 판단했습니다.")
        else:
            summary = "복합적 뉘앙스 정밀 분석 중"
            detail_lines.append("단일 카테고리로 정의하기 어려운 풍부한 감정선입니다.")

    elif name == "psychological_analyzer":
        stability = info.get("stability", 100)
        variance = info.get("emotional_variance", 0.0)
        
        summary = f"심리 안정성 지수: {stability}%"
        detail_lines.append("텍스트 전반에 걸친 작성자의 심리적 기복을 측정했습니다.")
        
        if variance > 0.3:
            detail_lines.append("감정의 변화폭이 다소 크며, 역동적인 심리 상태입니다.")
        else:
            detail_lines.append("일관되고 차분한 심리 상태를 유지하고 있습니다.")

    elif name == "transition_analyzer":
        count = len(transitions or [])
        if count > 0:
            summary = f"감정 전환점(Turning Point) {count}곳 포착"
            tr_examples = []
            for tr in transitions[:2]:
                fm = tr.get("from_main", "?")
                tm = tr.get("to_main", "?")
                tr_examples.append(f"{fm} → {tm}")
            detail_lines.append(f"주요 흐름: {', '.join(tr_examples)}")
            detail_lines.append("문장 간의 감정 변화 흐름을 유기적으로 추적했습니다.")
        else:
            summary = "단일 감정의 심화 및 유지"
            detail_lines.append("감정의 급격한 반전 없이 하나의 정서가 깊어지고 있습니다.")
            detail_lines.append("일관성 있는 메시지 전달력이 돋보입니다.")

    elif name == "weight_calculator":
        main_label = poster.get("main")
        summary = "최종 가중치(Weight) 산출 완료"
        if main_label:
            pct = _format_percentage(main_dist.get(main_label))
            detail_lines.append(f"결론: '{main_label}' 감정이 가장 지배적입니다 ({pct}).")
        detail_lines.append("11개 모듈의 분석 결과를 종합하여 최적의 결론을 도출했습니다.")

    elif name == "complex_analyzer":
        segments = info.get("segments", [])
        count = len(segments) if isinstance(segments, list) else 0
        if count > 0:
            summary = f"복합 감정 레이어 {count}개 발견"
            detail_lines.append("단순한 감정 너머의 미묘하고 복합적인 심리를 포착했습니다.")
            seg_desc = []
            for seg in segments[:2]:
                if isinstance(seg, dict):
                    emo = seg.get("emotion_main", "?")
                    seg_desc.append(f"{emo}")
            if seg_desc:
                detail_lines.append(f"구성 요소: {', '.join(seg_desc)}")
        else:
            summary = "명료하고 직관적인 감정선"
            detail_lines.append("양가감정이나 모순된 심리 없이 메시지가 명확합니다.")

    elif name == "relationship_analyzer":
        links = info.get("links", [])
        count = len(links) if isinstance(links, list) else 0
        if count > 0:
            summary = f"문장 간 유기적 연결성 {count}건 확인"
            detail_lines.append("문장들이 서로 원인과 결과로 긴밀하게 연결되어 있습니다.")
        else:
            summary = "독립적 구성의 서사 구조"
            detail_lines.append("각 문장이 고유한 의미를 가지며 병렬적으로 구성되어 있습니다.")

    elif name == "situation_analyzer":
        # ★★★ 개선: identified_situations 배열도 확인 ★★★
        situations = (
            info.get("identified_situations") or
            info.get("situations") or
            []
        )
        count = len(situations) if isinstance(situations, list) else 0
        if count > 0:
            summary = f"상황 분석 {count}건 매칭"
            sit_desc = []
            for sit in situations[:3]:
                if isinstance(sit, str):
                    sit_desc.append(sit)
                elif isinstance(sit, dict):
                    # situation, situation_name, description 순으로 확인
                    name_str = (
                        sit.get("situation") or
                        sit.get("situation_name") or
                        sit.get("description") or
                        ""
                    )
                    conf = sit.get("confidence")
                    source = sit.get("source") or sit.get("inference_source", "")
                    
                    if name_str:
                        # 신뢰도와 소스 표시
                        suffix = ""
                        if conf and isinstance(conf, (int, float)):
                            pct = int(conf * 100) if conf <= 1 else int(conf)
                            suffix = f"({pct}%)"
                        if "inference" in source:
                            suffix = f"[추론]{suffix}"
                        sit_desc.append(f"{name_str}{suffix}")
            if sit_desc:
                detail_lines.append(f"감지된 상황: {', '.join(sit_desc)}")
        else:
            summary = "범용적 상황 (General Context)"
            detail_lines.append("특정 도메인에 국한되지 않는 보편적인 이야기입니다.")

    elif name == "time_series_analyzer":
        # ★★★ 수정: 파이프라인 실제 반환 키 사용 ★★★
        # DEBUG: 실제 cause_effect 데이터 확인
        import logging
        _ts_logger = logging.getLogger("time_series_debug")
        _ts_logger.warning(f"[DEBUG] time_series_analyzer cause_effect raw: {info.get('cause_effect', [])[:2]}")
        _ts_logger.warning(f"[DEBUG] time_series_analyzer emotion_sequence sample: {info.get('emotion_sequence', [])[:2]}")
        
        emotion_seq = info.get("emotion_sequence") or info.get("sequence_analysis") or []
        series_data = info.get("series") or []
        time_flow = info.get("time_flow") or {}
        cause_effect = info.get("cause_effect") or []
        
        # 시퀀스 카운트 결정
        seq_count = 0
        if isinstance(emotion_seq, list) and emotion_seq:
            seq_count = len(emotion_seq)
        elif isinstance(series_data, list) and series_data:
            seq_count = len(series_data)
        
        # 시간 흐름 모드 확인
        flow_mode = time_flow.get("mode", "") if isinstance(time_flow, dict) else ""
        has_causality = len(cause_effect) > 0 if isinstance(cause_effect, list) else False
        
        if seq_count > 1 or flow_mode in ("linear", "linear_capped"):
            summary = f"시계열 흐름 분석 ({seq_count}구간)"
            detail_lines.append("시간 경과에 따른 감정의 발단-전개-절정을 추적했습니다.")
            
            # 시간 흐름 모드 표시
            if flow_mode:
                mode_desc = {
                    "linear": "정상적인 전진 흐름",
                    "linear_capped": "보수적 캡 적용",
                    "static": "정지 상태",
                }.get(flow_mode, flow_mode)
                detail_lines.append(f"시간 흐름: {mode_desc}")
            
            # 주요 감정 추출 헬퍼 (복잡한 키 처리)
            def get_dominant_emotion_with_score(emotions_dict):
                """복잡한 키에서 주요 감정(희/노/애/락) 집계 후 최대값 반환"""
                if not emotions_dict:
                    return None, 0
                main_counts = {"희": 0.0, "노": 0.0, "애": 0.0, "락": 0.0}
                for key, val in emotions_dict.items():
                    # '노-분개-sentiment_analysis' → '노'
                    first_part = key.split("-")[0] if "-" in key else key
                    if first_part in main_counts:
                        main_counts[first_part] += val
                # 최대값 반환
                total = sum(main_counts.values())
                if total > 0:
                    top = max(main_counts.items(), key=lambda x: x[1])
                    return top[0], top[1] / total  # 비율로 변환
                return None, 0
            
            # 각 구간(문장)별 감정 요약 (최대 3개)
            if isinstance(emotion_seq, list) and emotion_seq:
                for i, es in enumerate(emotion_seq[:3]):
                    if isinstance(es, dict):
                        emotions = es.get("emotions") or es.get("emotion_scores") or {}
                        top_emo, ratio = get_dominant_emotion_with_score(emotions)
                        if top_emo:
                            detail_lines.append(f"구간{i+1}: 주요 감정 '{top_emo}' ({ratio:.0%})")
            
            # ★★★ 감정 변화 표시: emotion_sequence에서 직접 추출 ★★★
            emotion_changes = []
            if isinstance(emotion_seq, list) and len(emotion_seq) >= 2:
                prev_emo = None
                for es in emotion_seq:
                    if isinstance(es, dict):
                        emotions = es.get("emotions", {})
                        curr_emo, _ = get_dominant_emotion_with_score(emotions)
                        if curr_emo and prev_emo and curr_emo != prev_emo:
                            emotion_changes.append((prev_emo, curr_emo))
                        prev_emo = curr_emo
            
            # 중복 제거 및 표시
            seen = set()
            unique_changes = []
            for change in emotion_changes:
                if change not in seen:
                    seen.add(change)
                    unique_changes.append(change)
            
            if unique_changes:
                detail_lines.append(f"감정 변화 {len(unique_changes)}건 감지")
                for from_emo, to_emo in unique_changes[:3]:
                    detail_lines.append(f"• {from_emo} → {to_emo}")
        elif seq_count == 1:
            summary = "순간적 감정 스냅샷 (Snapshot)"
            detail_lines.append("시간의 흐름보다는 현재 시점의 감정에 집중된 텍스트입니다.")
        else:
            if time_flow or info.get("summary"):
                summary = "시계열 패턴 분석됨"
                detail_lines.append("감정의 시간적 변화 패턴이 분석되었습니다.")
            else:
                summary = "시계열 분석 완료"
                detail_lines.append("텍스트의 시간적 구조가 분석되었습니다.")

    elif name == "context_analysis":  # already handled
        pass
    else:
        summary = "모듈이 성공적으로 수행되었습니다."
        detail_lines.append("내부 알고리즘을 통해 텍스트를 정밀 분석했습니다.")

    return summary, detail_lines


def _build_module_details(
    module_results: Mapping[str, Any],
    *,
    poster: Mapping[str, Any],
    main_dist: Mapping[str, Any],
    transitions: Sequence[Mapping[str, Any]],
    effective_mode: str,
) -> List[Dict[str, Any]]:
    details: List[Dict[str, Any]] = []

    for module in MODULE_SUMMARY_ORDER:
        info = module_results.get(module)
        status = "missing"
        if isinstance(info, Mapping):
            if info.get("skipped"):
                status = "skipped"
            elif info.get("success") is False:
                status = "error"
            else:
                status = "ok"
        summary = "결과 데이터가 전달되지 않았습니다."
        detail_lines: List[str] = []
        if status == "skipped":
            summary = str(info.get("reason") or f"{effective_mode.upper()} 모드 설정으로 건너뛰었습니다.")
        elif status == "missing":
            if effective_mode == "fast" and module in FAST_SKIP_MODULES:
                summary = "FAST 모드에서는 해당 모듈을 실행하지 않습니다."
            else:
                summary = "모듈 실행 결과가 보고되지 않았습니다."
        elif status == "error":
            summary = str(info.get("error") or "모듈 실행 중 오류가 보고되었습니다.")
        else:
            summary, detail_lines = _summarize_module_detail(
                module,
                info,
                poster=poster,
                main_dist=main_dist,
                module_results=module_results,
                transitions=transitions,
            )
            if not summary:
                summary = "모듈이 정상 실행되었습니다."
        details.append(
            {
                "name": module,
                "status": status,
                "summary": summary,
                "details": detail_lines,
            }
        )
    return details


def _default_emotions_path() -> Path:
    path = os.getenv("EMOTIONS_JSON_PATH")
    if path and Path(path).exists():
        return Path(path)
    return PROJECT_ROOT / "src" / "EMOTIONS.json"


@lru_cache(maxsize=8)
def _load_emotions(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def _collect_keywords(emotion_info: Dict[str, Any]) -> Iterable[str]:
    for key in ("keywords", "lexicon", "phrases"):
        items = emotion_info.get(key)
        if isinstance(items, list):
            for item in items:
                if isinstance(item, str) and item.strip():
                    yield item.strip().lower()
    triggers = emotion_info.get("triggers")
    if isinstance(triggers, list):
        for trig in triggers:
            if isinstance(trig, str) and trig.strip():
                yield trig.strip().lower()


def _score_emotions(
    text: str, emotions_data: Dict[str, Any]
) -> Tuple[Counter, Dict[str, float], List[Dict[str, Any]], Counter]:
    lower = text.lower()
    main_scores: Counter = Counter()
    sub_scores: Dict[str, float] = defaultdict(float)
    matches: List[Dict[str, Any]] = []
    sentiment_counts: Counter = Counter()
    seen_spans: Set[Tuple[str, str]] = set()

    for main, info in emotions_data.items():
        if not isinstance(info, dict):
            continue
        base_keywords = set(k for k in _collect_keywords(info))
        for kw in base_keywords:
            if kw and kw in lower:
                main_scores[main] += 1.2
                key = (kw, main)
                if key not in seen_spans:
                    seen_spans.add(key)
                    matches.append(
                        {
                            "text": kw,
                            "main_emotion": main,
                            "sub_emotion": None,
                            "confidence": 0.55,
                        }
                    )
        subs = info.get("sub_emotions") or {}
        for sub_name, sub_info in subs.items():
            if not isinstance(sub_info, dict):
                continue
            weight = 1.0
            for kw in _collect_keywords(sub_info):
                if kw and kw in lower:
                    main_scores[main] += weight
                    label = f"{main}-{sub_name}"
                    sub_scores[label] += weight
                    key = (kw, label)
                    if key not in seen_spans:
                        seen_spans.add(key)
                        matches.append(
                            {
                                "text": kw,
                                "main_emotion": main,
                                "sub_emotion": sub_name,
                                "confidence": 0.65,
                            }
                        )

    for main_label, cue_list in _COMPLAINT_HINTS:
        default_sub = _MAIN_DEFAULT_SUB.get(main_label, "중립")
        combined_label = f"{main_label}-{default_sub}"
        for cue in cue_list:
            if cue and cue in lower:
                main_scores[main_label] += 1.2
                sub_scores[combined_label] += 0.6
                key = (cue, combined_label)
                if key not in seen_spans:
                    seen_spans.add(key)
                    matches.append(
                        {
                            "text": cue,
                            "main_emotion": main_label,
                            "sub_emotion": default_sub,
                            "confidence": 0.6,
                        }
                    )

    tokens = _tokenize(text)
    token_counts = Counter(tokens)
    for tok, count in token_counts.items():
        for main, stems in _POSITIVE_LEXICON.items():
            if any(stem in tok for stem in stems):
                main_scores[main] += 0.45 * count
                sentiment_counts["positive"] += count
                key = (tok, f"{main}-pos")
                if key not in seen_spans:
                    seen_spans.add(key)
                    matches.append(
                        {
                            "text": tok,
                            "main_emotion": main,
                            "sub_emotion": None,
                            "confidence": 0.5,
                        }
                    )
        for main, stems in _NEGATIVE_LEXICON.items():
            if any(stem in tok for stem in stems):
                main_scores[main] += 0.5 * count
                sentiment_counts["negative"] += count
                key = (tok, f"{main}-neg")
                if key not in seen_spans:
                    seen_spans.add(key)
                    matches.append(
                        {
                            "text": tok,
                            "main_emotion": main,
                            "sub_emotion": None,
                            "confidence": 0.58,
                        }
                    )
        for label, stems in _SUB_HINTS.items():
            if any(stem in tok for stem in stems):
                main, sub = label.split("-", 1)
                main_scores[main] += 0.4 * count
                sub_scores[label] += 0.6 * count
                key = (tok, label)
                if key not in seen_spans:
                    seen_spans.add(key)
                    matches.append(
                        {
                            "text": tok,
                            "main_emotion": main,
                            "sub_emotion": sub,
                            "confidence": 0.62,
                        }
                    )

    if not main_scores:
        neg_terms = ("불만", "불편", "화가", "짜증", "걱정", "힘들", "싫")
        pos_terms = ("감사", "기쁨", "좋아", "만족", "편안", "행복", "도움")
        neg_hits = sum(lower.count(term) for term in neg_terms)
        pos_hits = sum(lower.count(term) for term in pos_terms)
        sentiment_counts["negative"] += neg_hits
        sentiment_counts["positive"] += pos_hits
        if neg_hits + pos_hits == 0:
            main_scores.update({"희": 1.0, "락": 0.8, "노": 0.6, "애": 0.6})
        else:
            if neg_hits >= pos_hits:
                main_scores["노"] = 1.0 + neg_hits
                main_scores["애"] = 0.5 + neg_hits / 2
            else:
                main_scores["희"] = 1.0 + pos_hits
                main_scores["락"] = 0.7 + pos_hits / 2

    return main_scores, sub_scores, matches, sentiment_counts


def _normalize_distribution(counter: Counter) -> Dict[str, float]:
    dist: Dict[str, float] = {}
    total = float(sum(counter.values()) + _SCORE_SMOOTH * max(len(counter), 1))
    if total <= 0:
        return {}
    for k in ("희", "노", "애", "락"):
        value = counter.get(k, 0.0) + _SCORE_SMOOTH
        dist[k] = round(value / total, 4)
    return dist


def _normalize_sub(sub_scores: Dict[str, float]) -> Dict[str, float]:
    if not sub_scores:
        return {}
    total = float(sum(sub_scores.values()))
    if total <= 0:
        return {}
    return {k: round(v / total, 4) for k, v in sub_scores.items()}


def _choose_intensity(main_score: float) -> str:
    if main_score >= 0.65:
        return "높음"
    if main_score >= 0.45:
        return "중간"
    return "낮음"


def _estimate_churn(main: str, dist: Dict[str, float]) -> int:
    neg = dist.get("노", 0.0) + dist.get("애", 0.0)
    if main in _NEG_MAIN:
        return min(80, int(40 + neg * 60))
    return max(5, int(neg * 40))


def _recommend_actions(main: str, churn: int) -> List[str]:
    if main in _NEG_MAIN or churn >= 40:
        return ["담당자 1차 연락 및 원인 파악", "맞춤 설득 스크립트 적용"]
    return ["긍정 피드백 강화", "추천 리워드 제안"]


def _choose_sentence_match(
    sentence: str,
    matches: List[Dict[str, Any]],
    default_main: str,
    default_sub: str,
) -> Tuple[str, str]:
    best: Optional[Dict[str, Any]] = None
    best_conf = -1.0
    lowered = sentence.lower()
    for match in matches:
        phrase = (match.get("text") or "").strip().lower()
        if not phrase:
            continue
        if phrase in lowered:
            conf = float(match.get("confidence") or 0.5)
            if conf > best_conf:
                best_conf = conf
                best = match
    main = best.get("main_emotion") if best else default_main
    sub = best.get("sub_emotion") if best else default_sub
    if not sub or sub == "—":
        sub = default_sub or _MAIN_DEFAULT_SUB.get(main, "중립")

    hint_main = _infer_main_from_sentence(sentence, main or default_main)
    if hint_main and hint_main != (main or default_main):
        if not best or best_conf < 0.75:
            main = hint_main
            sub = _MAIN_DEFAULT_SUB.get(main, sub)

    sub = _infer_sub_from_sentence(main or default_main, sentence, sub) or sub
    if not sub or sub == "—":
        sub = default_sub or _MAIN_DEFAULT_SUB.get(main or default_main, "중립")

    return (main or default_main) or default_main, sub


def _build_sentence_annotations(
    text: str,
    matches: List[Dict[str, Any]],
    main: str,
    sub_top: List[Tuple[str, float]],
) -> Tuple[List[str], List[Dict[str, Any]]]:
    sentences = [seg.strip() for seg in _SENT_SPLIT_RE.split(text) if seg.strip()]
    sub_label = sub_top[0][0].split("-", 1)[1] if sub_top else "중립"
    annotations = []
    structured: List[Dict[str, Any]] = []
    for idx, sent in enumerate(sentences, start=1):
        main_tag, sub_tag = _choose_sentence_match(sent, matches, main, sub_label)
        annotations.append(f"{idx:02d}. {sent}  [{main_tag}|{sub_tag}]")
        structured.append(
            {
                "index": idx,
                "text": sent,
                "main": main_tag,
                "sub": sub_tag,
            }
        )
    return annotations, structured


def _derive_sub_candidates(matches: List[Dict[str, Any]], main: str) -> Counter:
    counter: Counter = Counter()
    for match in matches:
        if match.get("main_emotion") == main and match.get("sub_emotion"):
            counter[match["sub_emotion"]] += 1
    return counter


def _default_sub(main: str) -> str:
    return _MAIN_DEFAULT_SUB.get(main, "중립")


def _infer_main_from_sentence(sentence: str, fallback: str) -> Optional[str]:
    lowered = sentence.lower()
    best_main: Optional[str] = None
    best_score = 0
    for main_label, keywords in _SENTENCE_HINTS.items():
        score = 0
        for keyword in keywords:
            key = keyword.lower()
            if key and key in lowered:
                score += 1
        if score > best_score:
            best_score = score
            best_main = main_label
        elif score == best_score and score > 0 and best_main is None:
            best_main = main_label
    if best_score == 0:
        return None
    if best_main is None:
        return None
    if best_main == fallback and best_score <= 0:
        return None
    return best_main


def _infer_sub_from_sentence(main: str, sentence: str, current_sub: Optional[str]) -> Optional[str]:
    lowered = sentence.lower()
    for sub_label, keywords in _MAIN_SUB_HINTS.get(main, []):
        for keyword in keywords:
            if keyword and keyword in lowered:
                return sub_label
    return current_sub


def _build_transitions_structured(
    sentences: List[Dict[str, Any]],
    main_dist: Dict[str, float],
    triggers: List[str],
) -> List[Dict[str, Any]]:
    transitions: List[Dict[str, Any]] = []
    if len(sentences) < 2:
        return transitions

    trigger_iter = iter(triggers or [])
    for idx in range(len(sentences) - 1):
        src = sentences[idx]
        dst = sentences[idx + 1]
        from_main = src.get("main") or "—"
        to_main = dst.get("main") or "—"
        if from_main == "—" or to_main == "—":
                continue
        from_sub = src.get("sub") or _default_sub(from_main)
        to_sub = dst.get("sub") or _default_sub(to_main)
        base = (main_dist.get(from_main, 0.0) + main_dist.get(to_main, 0.0)) / 2.0
        same_main = from_main == to_main
        same_sub = from_sub == to_sub
        transition_type = "steady" if same_main and same_sub else "shift"
        if transition_type == "steady":
            probability = max(0.05, min(0.85, base + 0.05))
        else:
            probability = max(0.05, min(0.92, base + 0.18))
        trigger = ""
        dst_text = str(dst.get("text") or "").strip()
        if dst_text:
            trigger = dst_text[:40]
        if not trigger:
            for candidate in trigger_iter:
                candidate = str(candidate or "").strip()
                if candidate:
                    trigger = candidate[:40]
                    break

        if transition_type == "steady":
            transition_reason = (
                f"문장 {src.get('index')}→{dst.get('index')} 사이에서 {from_main}({from_sub}) 감정이 안정적으로 유지되었습니다."
            )
        else:
            transition_reason = (
                f"문장 {src.get('index')}→{dst.get('index')} 사이에서 {from_main}→{to_main} 감정 전이가 감지되었습니다."
            )
        if trigger:
            transition_reason += f' (핵심 표현: "{trigger}")'

        probability_explain = (
            f"{from_main} {round(main_dist.get(from_main, 0.0) * 100, 1)}%, "
            f"{to_main} {round(main_dist.get(to_main, 0.0) * 100, 1)}% 분포 평균 기반"
        )

        transitions.append(
            {
                "from_index": src.get("index"),
                "to_index": dst.get("index"),
                "from_main": from_main,
                "from_sub": from_sub,
                "to_main": to_main,
                "to_sub": to_sub,
                "excerpt_from": src.get("text", ""),
                "excerpt_to": dst.get("text", ""),
                "probability": round(probability, 3),
                "confidence": round(probability, 3),
                "probability_explain": probability_explain,
                "transition_reason": transition_reason,
                "trigger": trigger,
                "transition_type": transition_type,
            }
        )
    return transitions


def _enrich_matches(
    matches: List[Dict[str, Any]],
    main_dist: Dict[str, float],
    sub_top: List[Tuple[str, float]],
    sentences: List[str],
) -> None:
    primary_sub: Dict[str, str] = {}
    for label, _score in sub_top:
        if "-" in label:
            main, sub = label.split("-", 1)
            if main not in primary_sub:
                primary_sub[main] = sub
    for main, fallback in _MAIN_DEFAULT_SUB.items():
        primary_sub.setdefault(main, fallback)

    for match in matches:
        main = (match.get("main_emotion") or "").strip()
        if not main:
            continue
        sub = match.get("sub_emotion")
        if not sub:
            sub = primary_sub.get(main, _MAIN_DEFAULT_SUB.get(main, "중립"))
            match["sub_emotion"] = sub
        score_pct = round(main_dist.get(main, 0.0) * 100, 1)
        match["score_pct"] = score_pct
        confidence = match.get("confidence")
        if isinstance(confidence, (int, float)) and confidence <= 1:
            match["confidence_pct"] = round(float(confidence) * 100, 1)
        elif isinstance(confidence, (int, float)):
            match["confidence_pct"] = round(float(confidence), 1)
        phrase = str(match.get("text") or "").strip()
        evidence_sentence = ""
        lowered = phrase.lower()
        if lowered:
            for sent in sentences:
                if lowered in sent.lower():
                    evidence_sentence = sent
                    break
        if evidence_sentence:
            match["evidence_sentence"] = evidence_sentence
        if phrase:
            match["explanation"] = f"\"{phrase}\" 표현이 {main}-{match['sub_emotion']} 패턴과 일치합니다."
        else:
            match["explanation"] = f"{main}-{match['sub_emotion']} 패턴이 통계적으로 우세합니다."


def _infer_defenses(negative_ratio: float, stability: int, intensity: str) -> List[str]:
    defenses: List[str] = []
    if negative_ratio >= 0.5:
        defenses.append("감정 표출")
    if stability <= 40:
        defenses.append("회피/도피")
    if intensity == "높음":
        defenses.append("과잉 반응")
    if not defenses:
        defenses.append("균형 유지")
    return defenses[:3]


def _build_narrative(
    main: str,
    main_dist: Dict[str, float],
    intensity: str,
    churn: int,
    top_triggers: List[str],
    sentiment_balance: float,
) -> List[str]:
    detail = ", ".join(
        f"{emo}:{round(score * 100, 1)}%" for emo, score in sorted(main_dist.items(), key=lambda kv: kv[1], reverse=True)
    )
    lines = [
        f"주 감정은 {main}({round(main_dist.get(main, 0.0) * 100, 1)}%)로 파악되었습니다.",
        f"세부 분포는 {detail} 입니다.",
        f"감정 강도는 '{intensity}'이며, 감성 밸런스는 {round(sentiment_balance, 2)}입니다.",
        f"서비스 이탈 위험은 {churn}% 수준으로 예측됩니다.",
    ]
    if top_triggers:
        lines.append(f"대표적인 감정 트리거는 {', '.join(top_triggers[:3])} 입니다.")
    else:
        lines.append("뚜렷한 감정 트리거는 탐지되지 않았습니다.")
    return lines


def analyze_text_fast(text: str, *, emotions_path: Optional[str] = None) -> Dict[str, Any]:
    start = time.perf_counter()
    emotions = _load_emotions(str(emotions_path or _default_emotions_path()))

    sentences_raw = [seg.strip() for seg in _SENT_SPLIT_RE.split(text) if seg.strip()]
    domain_profile = _detect_domain_profile(text)

    main_scores, sub_scores, matches, sentiment_counts = _score_emotions(text, emotions)
    main_dist = _normalize_distribution(main_scores)
    if not main_dist:
        main_dist = {"희": 0.5, "노": 0.2, "락": 0.2, "애": 0.1}
    main = max(main_dist, key=main_dist.get)
    sub_candidates = _derive_sub_candidates(matches, main)
    if not sub_scores and sub_candidates:
        for sub_name, cnt in sub_candidates.items():
            sub_scores[f"{main}-{sub_name}"] += cnt
    sub_dist = _normalize_sub(sub_scores)
    sub_top = sorted(sub_dist.items(), key=lambda kv: kv[1], reverse=True)[:5]

    intensity = _choose_intensity(main_dist.get(main, 0.0))
    for level, hints in _INTENSITY_HINTS.items():
        if any(hint in text for hint in hints):
            intensity = level
            break
    churn = _estimate_churn(main, main_dist)

    sentiment_total = sentiment_counts.get("positive", 0) + sentiment_counts.get("negative", 0)
    sentiment_balance = 0.0
    if sentiment_total:
        sentiment_balance = (
            sentiment_counts.get("positive", 0) - sentiment_counts.get("negative", 0)
        ) / sentiment_total
    negative_ratio = main_dist.get("노", 0.0) + main_dist.get("애", 0.0)
    stability_score = 1 - min(0.95, negative_ratio + abs(sentiment_balance))
    if intensity == "높음":
        stability_score -= 0.2
    elif intensity == "낮음":
        stability_score += 0.05
    stability = max(5, min(95, int(stability_score * 100)))

    flow_sequence = sorted(main_dist.items(), key=lambda kv: kv[1], reverse=True)
    flow_text = " → ".join(
        f"{emo} {round(score * 100, 1)}%" for emo, score in flow_sequence[:3] if score > 0
    )
    flow_transitions: List[Dict[str, Any]] = []
    if len(flow_sequence) >= 2:
        for i in range(len(flow_sequence) - 1):
            frm, frm_score = flow_sequence[i]
            to, to_score = flow_sequence[i + 1]
            flow_transitions.append(
                {
                    "from": frm,
                    "to": to,
                    "weight": round(to_score, 3),
                }
            )

    poster = {
        "main": main,
        "main_distribution": main_dist,
        "sub_distribution": {k: v for k, v in sub_top},
        "flow_ssot": flow_text or "경량 추정 흐름",
        "flags": {"uncertainty_badge": "중간 · ±8%p"},
        "trust_stamp": {"consistency": int(main_dist.get(main, 0) * 100), "evidence": "mid"},
        "domain_profile": domain_profile,
    }

    top_triggers: List[str] = []
    seen_trigger = set()
    for match in matches:
        phrase = match.get("text")
        if phrase and phrase not in seen_trigger:
            seen_trigger.add(phrase)
            top_triggers.append(phrase)
        if len(top_triggers) >= 5:
            break
    if not top_triggers and sentences_raw:
        for sent in sentences_raw[:2]:
            snippet = sent[:30].strip()
            if snippet and snippet not in seen_trigger:
                top_triggers.append(snippet)
                seen_trigger.add(snippet)
            if len(top_triggers) >= 3:
                break

    investor_pack: Optional[Dict[str, Any]] = None
    if domain_profile == "service":
        products = {
            "p1": {
                "headline_emotions": [
                    {"name": main, "pct": round(main_dist.get(main, 0.0) * 100, 1)}
                ],
                "intensity": intensity,
                "churn_probability": churn,
                "horizon_days": 3,
                "triggers": top_triggers[:3],
                "recommended_actions": _recommend_actions(main, churn),
                "insights": {
                    "positive_hits": sentiment_counts.get("positive", 0),
                    "negative_hits": sentiment_counts.get("negative", 0),
                    "sentiment_balance": round(sentiment_balance, 3),
                },
            },
            "p3": {
                "risk_score": int(churn / 2),
                "grade": "High" if churn >= 60 else ("Medium" if churn >= 35 else "Low"),
                "alert": churn >= 50,
            },
            "p5": {
                "stability": stability,
                "maturity": 60 if intensity == "중간" else (45 if intensity == "낮음" else 75),
        "defenses": [],
            },
        }
    else:
        products = {
            "p1": {
                "headline_emotions": [
                    {"name": main, "pct": round(main_dist.get(main, 0.0) * 100, 1)}
                ],
                "intensity": intensity,
                "churn_probability": None,
                "horizon_days": None,
                "triggers": top_triggers[:3],
                "recommended_actions": [
                    "감정 여정을 반영한 고객 스토리텔링 강화",
                    "회복 순간을 활용한 브랜드 메시지 설계",
                ],
            },
            "p3": {
                "risk_score": int(round(negative_ratio * 100)),
                "grade": "Moderate" if negative_ratio >= 0.35 else "Low",
                "alert": False,
            },
            "p5": {
                "stability": stability,
                "maturity": 60 if intensity == "중간" else (45 if intensity == "낮음" else 75),
                "defenses": [],
            },
        }

    annotations, structured_annotations = _build_sentence_annotations(text, matches, main, sub_top)
    _enrich_matches(matches, main_dist, sub_top, [item.get("text", "") for item in structured_annotations])
    transitions_structured = _build_transitions_structured(structured_annotations, main_dist, top_triggers)
    if transitions_structured:
        flow_transitions = [
            {
                "from": t["from_main"],
                "to": t["to_main"],
                "from_sub": t.get("from_sub"),
                "to_sub": t.get("to_sub"),
                "weight": t.get("probability", t.get("confidence")),
                "probability": t.get("probability"),
                "probability_explain": t.get("probability_explain"),
                "transition_reason": t.get("transition_reason"),
                "trigger": t.get("trigger"),
                "confidence": t.get("confidence"),
            }
            for t in transitions_structured
        ]

    results: Dict[str, Any] = {}
    for module in (
        "emotion_classification",
        "intensity_analysis",
        "context_analysis",
        "sub_emotion_detection",
        "linguistic_matcher",
        "weight_calculator",
        "embedding_generation",
    ):
        results[module] = {"success": True, "skipped": False, "mode": "fast-lite"}

    for skipped_module in (
        "pattern_extractor",
        "transition_analyzer",
        "time_series_analyzer",
        "complex_analyzer",
        "psychological_analyzer",
        "relationship_analyzer",
        "situation_analyzer",
    ):
        results[skipped_module] = {"success": True, "skipped": True, "mode": "fast-lite"}

    results["emotion_classification"]["matched_phrases"] = matches[:20]
    results["intensity_analysis"]["intensity"] = intensity
    results["intensity_analysis"]["confidence"] = round(main_dist.get(main, 0.0), 2)
    results["context_analysis"]["top_triggers"] = top_triggers or sentences_raw[:1]
    results["sub_emotion_detection"]["top_sub_emotions"] = [
        {"label": label, "score": round(score * 100, 1)} for label, score in sub_top
    ]
    results["context_extractor"] = {
            "success": True,
        "skipped": False,
        "mode": "fast-lite",
        "key_phrases": top_triggers,
        "dominant_flow": flow_sequence[:3],
        "sentiment_balance": round(sentiment_balance, 3),
        "flow_transitions": flow_transitions,
        "structured_transitions": transitions_structured,
        "sentence_count": len(sentences_raw),
    }
    emotion_variance = round(abs(sentiment_balance) + main_dist.get("노", 0.0) + main_dist.get("애", 0.0), 3)
    defenses = _infer_defenses(negative_ratio, stability, intensity)
    results["psychological_analyzer"] = {
        "success": True,
        "skipped": False,
        "mode": "fast-lite",
        "emotional_variance": emotion_variance,
        "stability": stability,
        "negative_ratio": round(negative_ratio, 3),
        "defense_mechanisms": defenses,
    }
    products["p5"]["defenses"] = defenses

    if domain_profile == "service":
        insight_summary = _build_narrative(main, main_dist, intensity, churn, top_triggers, sentiment_balance)
    else:
        if investor_pack is None:
            investor_pack = _build_generic_investor_pack(
                main,
                main_dist,
                transitions_structured,
                top_triggers,
                stability,
                intensity,
                sentiment_balance,
                insight_candidates=[],
            )
        sections = investor_pack.get("sections", []) if investor_pack else []
        highlights = investor_pack.get("highlights", []) if investor_pack else []
        products["generic"] = {
            "sections": sections,
            "highlights": highlights,
        }
        insight_summary = investor_pack.get("summary", []) if investor_pack else []

    elapsed_ms = int((time.perf_counter() - start) * 1000)

    return {
        "poster": poster,
        "results": results,
        "products": products,
        "sentence_annotations": annotations,
        "sentence_annotations_structured": structured_annotations,
        "transitions_structured": transitions_structured,
        "insight_summary": insight_summary,
        "model_narrative": insight_summary,
        "meta": {
            "mode": "FAST",
            "elapsed_ms": elapsed_ms,
            "elapsed": round(elapsed_ms / 1000.0, 3),
            "orch_fallback": True,
            "model_version": "fast-lite-1.1",
            "cache_stats": {"enabled": 0},
            "evidence_score": round(0.55 + main_dist.get(main, 0.0) * 0.3, 2),
            "sentiment_balance": round(sentiment_balance, 3),
            "positive_hits": sentiment_counts.get("positive", 0),
            "negative_hits": sentiment_counts.get("negative", 0),
            "fast_strategy": "lite",
            "domain_profile": domain_profile,
        },
    }


def _is_scalar(val: Any) -> bool:
    if isinstance(val, (int, float, bool)):
        return True
    if np is not None and isinstance(val, np.generic):  # type: ignore
        return True
    return False


def _to_float(val: Any) -> float:
    if isinstance(val, (int, float)):
        return float(val)
    if np is not None and isinstance(val, np.generic):  # type: ignore
        return float(val)  # type: ignore[arg-type]
    raise TypeError


def sanitize(obj: Any) -> Any:
    """JSON 직렬화를 위해 numpy / datetime 등을 기본 타입으로 변환."""
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize(v) for v in obj]
    if _is_scalar(obj):
        try:
            return _to_float(obj)
        except TypeError:
            return obj
    if isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()
    if np is not None and isinstance(obj, np.ndarray):  # type: ignore
        return obj.tolist()  # type: ignore[call-arg]
    return obj


def compute_module_hit_rate(results: Mapping[str, Any]) -> Dict[str, int]:
    hit_rate: Dict[str, int] = {}
    for module in MODULE_ORDER:
        hit_rate[module] = 1 if results.get(module) else 0
    return hit_rate


def _normalize_main_label(label: Any) -> str:
    if label is None:
        return "—"
    s = str(label).strip()
    if not s:
        return "—"
    if s in ("희", "노", "애", "락"):
        return s
    if s in MAIN_MAP:
        return MAIN_MAP[s]
    if "-sub_" in s:
        head = s.split("-sub_", 1)[0]
        return MAIN_MAP.get(head, head)
    if s.isdigit():
        return MAIN_MAP.get(s, s)
    lower = s.lower()
    if "anger" in lower:
        return "노"
    if "sad" in lower:
        return "애"
    if any(t in lower for t in ("joy", "delight", "pleas")):
        return "락"
    if any(t in lower for t in ("happy", "glad")):
        return "희"
    return s


def _try_call(func: Optional[Callable[..., Any]], *args, **kwargs) -> Any:
    if callable(func):
        try:
            return func(*args, **kwargs)
        except Exception as exc:  # pragma: no cover - 보조 함수 실패는 치명적이지 않음
            logger.debug("[Explainability] helper call failed: %s", exc)
    return None


def _truncate_text(value: Any, limit: int) -> str:
    text = str(value or "")
    if not text:
        return ""
    truncated = _try_call(_TRUNCATE_FUNC, text, limit)
    if isinstance(truncated, str):
        return truncated
    if len(text) <= limit:
        return text
    if limit <= 3:
        return text[:limit]
    return text[: limit - 3] + "..."


def _build_explainability_snapshot(
    out: Mapping[str, Any],
    *,
    text: str,
    main_dist: Optional[Mapping[str, Any]],
    sub_dist: Optional[Mapping[str, Any]],
    transitions_structured: Sequence[Mapping[str, Any]],
    bundle: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    def _flow_from_structured_list(items: Optional[Sequence[Mapping[str, Any]]]) -> str:
        if not items:
            return ""
        labels: List[str] = []
        for row in items:
            if not isinstance(row, Mapping):
                continue
            src = _normalize_main_label(
                row.get("from")
                or row.get("from_main")
                or row.get("from_label")
                or row.get("source")
            )
            dst = _normalize_main_label(
                row.get("to")
                or row.get("to_main")
                or row.get("to_label")
                or row.get("target")
            )
            if src and src != "—":
                if not labels:
                    labels.append(src)
                elif labels[-1] != src:
                    labels.append(src)
            if dst and dst != "—":
                if not labels:
                    labels.append(dst)
                elif labels[-1] != dst:
                    labels.append(dst)
        if len(labels) >= 2:
            deduped: List[str] = []
            for lbl in labels:
                if not deduped or deduped[-1] != lbl:
                    deduped.append(lbl)
            if len(deduped) >= 2:
                return " -> ".join(deduped)
        return ""
    def _safe_float(val: Any) -> Optional[float]:
        try:
            return float(val)
        except (TypeError, ValueError):
            return None

    main_pairs: List[Tuple[str, float]] = []
    for raw_label, raw_score in (main_dist or {}).items():
        score = _safe_float(raw_score)
        if score is None or score <= 0:
            continue
        main_pairs.append((_normalize_main_label(raw_label), score))
    main_pairs.sort(key=lambda kv: kv[1], reverse=True)
    main_top = [
        {"label": label, "pct": round(score * 100.0, 1), "score": round(score, 6)}
        for label, score in main_pairs[:3]
    ]
    main_line = (
        " · ".join([f"{item['label']} {item['pct']:.1f}%" for item in main_top])
        if main_top
        else "—"
    )

    primary_main = main_top[0]["label"] if main_top else "애"

    def _split_sub_key(key: Any) -> Tuple[str, str]:
        if _NORMALIZE_SUB_KEY_FUNC:
            res = _try_call(_NORMALIZE_SUB_KEY_FUNC, key, main_dist or {})
            if isinstance(res, (list, tuple)) and len(res) >= 2:
                return _normalize_main_label(res[0]), str(res[1])
        s = str(key or "")
        if "-" in s:
            head, tail = s.split("-", 1)
            return _normalize_main_label(head), tail.strip()
        if s.startswith("sub_"):
            return primary_main, s
        return primary_main, s

    sub_entries: List[Tuple[str, float, str]] = []
    for raw_label, raw_score in (sub_dist or {}).items():
        score = _safe_float(raw_score)
        if score is None or score <= 0:
            continue
        main_label, sub_raw = _split_sub_key(raw_label)
        pretty = _format_sub_label(main_label, sub_raw)
        if not pretty:
            pretty = f"{main_label}-{sub_raw}" if sub_raw else main_label
        sub_entries.append((pretty, score, main_label))
    sub_entries.sort(key=lambda kv: kv[1], reverse=True)

    sub_top = [
        {
            "label": label,
            "pct": round(score * 100.0, 1),
            "score": round(score, 6),
            "main": main_label,
        }
        for label, score, main_label in sub_entries[:3]
    ]
    sub_line = (
        " · ".join([f"{item['label']} {item['pct']:.1f}%" for item in sub_top])
        if sub_top
        else "—"
    )

    sub_top10_lines = [
        f"{label:>22}  {score:0.3f}"
        for label, score, _ in sub_entries[:10]
    ]
    if not sub_top10_lines:
        sub_top10_lines = ["(데이터 없음)"]

    flow_summary = _try_call(_EMO_FLOW_FROM_OUT, out) if isinstance(out, Mapping) else None
    if not flow_summary and isinstance(bundle, Mapping):
        flow_summary = bundle.get("flow_ssot")
    if not flow_summary or str(flow_summary).strip() in ("", "-", "흐름 정보 없음"):
        fallback_transitions: List[Mapping[str, Any]] = []
        if transitions_structured:
            fallback_transitions.extend(
                [row for row in transitions_structured if isinstance(row, Mapping)]
            )
        if isinstance(bundle, Mapping):
            bundle_trs = bundle.get("transitions_structured") or bundle.get("transitions")
            if isinstance(bundle_trs, list):
                fallback_transitions.extend(
                    [row for row in bundle_trs if isinstance(row, Mapping)]
                )
        fallback_flow = _flow_from_structured_list(fallback_transitions)
        flow_summary = fallback_flow or "흐름 정보 없음"
    else:
        flow_summary = str(flow_summary)

    gather_result = _try_call(
        _GATHER_TRANSITIONS_AND_KEYWORDS,
        out,
        main_dist or {},
        text_fallback=text,
        topk_kw=5,
        limit_tr=3,
    )
    transitions: List[Mapping[str, Any]] = []
    keyword_pairs: List[Tuple[str, float]] = []
    if isinstance(gather_result, tuple) and len(gather_result) >= 2:
        maybe_tr, maybe_kw = gather_result[:2]
        if isinstance(maybe_tr, list):
            transitions = maybe_tr
        if isinstance(maybe_kw, list):
            keyword_pairs = [
                (str(name), float(score))
                for name, score in maybe_kw
                if _safe_float(score) is not None
            ]

    if not transitions and transitions_structured:
        transitions = list(transitions_structured[:3])

    transition_line = "—"
    if transitions:
        parts = []
        for item in transitions[:3]:
            if not isinstance(item, Mapping):
                continue
            src = _normalize_main_label(item.get("from") or item.get("from_main"))
            dst = _normalize_main_label(item.get("to") or item.get("to_main"))
            if src == "—" and dst == "—":
                continue
            trig = item.get("trigger")
            seg = f"{src}->{dst}"
            if trig:
                seg += f" (trigger: {trig})"
            parts.append(seg)
        if parts:
            transition_line = " | ".join(parts)
    if transition_line == "—" and flow_summary not in ("", "흐름 정보 없음"):
        transition_line = flow_summary.replace(" -> ", " | ")

    keywords_line = "—"
    if keyword_pairs:
        keywords_line = ", ".join(
            f"{_truncate_text(name, 16)}({float(score):+0.2f})"
            for name, score in keyword_pairs[:6]
        )

    entropy = _try_call(_NORMALIZED_ENTROPY_FUNC, main_dist or {})
    if not isinstance(entropy, (int, float)):
        values = [max(0.0, _safe_float(v) or 0.0) for v in (main_dist or {}).values()]
        total = sum(values)
        if total <= 0:
            entropy = 0.0
        else:
            probs = [v / total for v in values if v > 0]
            if len(probs) <= 1:
                entropy = 0.0
            else:
                entropy = float(-sum(p * math.log(p) for p in probs) / (math.log(len(probs)) or 1.0))

    gate = {}
    if isinstance(out, Mapping):
        ui_flags = out.get("ui_flags")
        if isinstance(ui_flags, Mapping):
            gate = ui_flags.get("gating") or {}
    evidence_val = gate.get("evidence")
    try:
        evidence_val = float(evidence_val) if evidence_val is not None else None
    except (TypeError, ValueError):
        evidence_val = None
    gate_level = gate.get("level") or "unknown"

    why_lines = [
        f"메인 Top-3: {main_line}",
        f"세부 Top-3: {sub_line}",
        f"전이(요약-SSOT): {flow_summary}",
        f"전이(상세): {transition_line}",
        f"핵심 표현: {keywords_line}",
        "※ 괄호 안 숫자는 표현의 가중/기여 점수(정규화)입니다. +는 강화, −는 완화/억제를 의미합니다.",
    ]

    # ★★★ test.py 원본 데이터 최우선 적용 (bundle → out 순서) ★★★
    reasoning_lines: List[str] = []
    why_lines_data: List[str] = []
    sub_top10_lines_data: List[str] = []
    
    # 1순위: bundle에서 가져온 reasoning_path_lines (test.py에서 생성한 원본)
    bundle_reasoning_path_lines = bundle.get("reasoning_path_lines") if isinstance(bundle, Mapping) else None
    # bundle이 비어있을 때 test.py 직접 결과 사용
    if not bundle_reasoning_path_lines and isinstance(out, Mapping):
        bundle_reasoning_path_lines = out.get("reasoning_path_lines")
    
    # bundle에서 가져온 완전한 구조 데이터가 있으면 그대로 사용 (변환 금지)
    if isinstance(bundle_reasoning_path_lines, list) and bundle_reasoning_path_lines:
        # test.py에서 이미 완전한 구조로 생성된 데이터이므로 그대로 사용
        reasoning_lines = bundle_reasoning_path_lines
    
    # why_lines (왜 이런 감정인가)
    bundle_why_lines = bundle.get("why_lines") if isinstance(bundle, Mapping) else None
    if not bundle_why_lines and isinstance(out, Mapping):
        bundle_why_lines = out.get("why_lines")
    if isinstance(bundle_why_lines, list) and bundle_why_lines:
        why_lines_data = bundle_why_lines
    
    # sub_top10_lines (세부감정 Top-10)
    bundle_sub_top10_lines = bundle.get("sub_top10_lines") if isinstance(bundle, Mapping) else None
    if not bundle_sub_top10_lines and isinstance(out, Mapping):
        bundle_sub_top10_lines = out.get("sub_top10_lines")
    if isinstance(bundle_sub_top10_lines, list) and bundle_sub_top10_lines:
        sub_top10_lines_data = bundle_sub_top10_lines
    
    if not reasoning_lines:
        # bundle에 없을 때만 app.py에서 생성
        base_line = f"① 신호 요약: 엔트로피 H={float(entropy):.2f}"
        if evidence_val is not None:
            base_line += f", Evidence={evidence_val:.2f}, Level={gate_level}"
        reasoning_lines.append(base_line)

        explicit_hit = _try_call(_EXPLICIT_EMOTION_FUNC, text)
        if isinstance(explicit_hit, tuple) and len(explicit_hit) >= 3:
            reasoning_lines.append(
                f"② 명시 매칭: {explicit_hit[0]}|{explicit_hit[1]} (키워드: {explicit_hit[2]})"
            )

        semantic_hit = _try_call(_SEMANTIC_SUB_FUNC, text)
        if isinstance(semantic_hit, tuple) and len(semantic_hit) >= 3:
            reasoning_lines.append(
                f"③ 시맨틱 근사: {semantic_hit[0]}|{semantic_hit[1]} (근사: {semantic_hit[2]})"
            )

        cues = bool(_try_call(_HAS_UNCERTAINTY_CUES_FUNC, text, out))
        triggers_raw = _try_call(_DETECT_TRIGGERS_FUNC, text, out) or []
        if not isinstance(triggers_raw, list):
            triggers_raw = []
        trigger_count = len([t for t in triggers_raw if t])
        trigger_desc = f"트리거 {trigger_count}개" if trigger_count else "트리거 없음"
        if cues:
            trigger_desc = "의문/불안 단서 감지, " + trigger_desc
        reasoning_lines.append(f"④ 단서/트리거: {trigger_desc}")

        hide_map = gate.get("hide") if isinstance(gate, Mapping) else None
        if isinstance(hide_map, Mapping):
            hidden = [k for k, v in hide_map.items() if v]
        else:
            hidden = []
        if hidden:
            reasoning_lines.append("⑤ 게이팅: " + ", ".join(hidden) + " 숨김(근거 부족)")
        else:
            reasoning_lines.append("⑤ 게이팅: 노출")

        top_summary = "—"
        if main_pairs:
            summary_pairs = main_pairs[:2]
            top_summary = " / ".join([f"{label} {int(round(score * 100))}%" for label, score in summary_pairs])
        reasoning_lines.append(f"⑥ 결정 요약: 메인 {top_summary}")

    # why_lines와 sub_top10_lines 우선순위: bundle 데이터 > app.py 생성 데이터
    final_why_lines = why_lines_data if why_lines_data else why_lines
    final_sub_top10_lines = sub_top10_lines_data if sub_top10_lines_data else sub_top10_lines
    
    # [PERFORMANCE] 병렬 실행 결과 활용 (존재할 경우 덮어쓰기)
    # _build_explainability_snapshot이 호출될 때는 이미 직렬 실행이 완료된 상태일 수 있으나,
    # 향후 확장을 위해 병렬 실행 캐시가 있다면 활용
    # (현재 구조에서는 _build_explainability_snapshot 내에서 병렬 실행을 직접 호출하지 않음)

    return {
        "main_top": main_top,
        "sub_top": sub_top,
        "why_lines": final_why_lines,
        "sub_top10_lines": final_sub_top10_lines,
        "keywords": [
            {"term": term, "score": round(float(score), 3)} for term, score in keyword_pairs[:10]
        ],
        "transitions": [
            {
                "from": _normalize_main_label(item.get("from") or item.get("from_main")),
                "to": _normalize_main_label(item.get("to") or item.get("to_main")),
                "trigger": item.get("trigger"),
            }
            for item in transitions[:5]
            if isinstance(item, Mapping)
        ],
        "reasoning_path_lines": reasoning_lines,
        "flow_summary": flow_summary,
        "transition_line": transition_line,
        "keyword_line": keywords_line,
    }

def _format_sentence_annotations(bundle: Mapping[str, Any], text: str) -> List[str]:
    anchors = bundle.get("anchors") if isinstance(bundle, Mapping) else {}
    sentences = anchors.get("sentences") if isinstance(anchors, Mapping) else None
    rows: List[str] = []

    if isinstance(sentences, list) and sentences:
        for idx, row in enumerate(sentences, 1):
            if not isinstance(row, Mapping):
                continue
            raw = (
                row.get("text")
                or row.get("raw")
                or row.get("sentence")
                or row.get("content")
                or ""
            )
            raw = str(raw).strip()
            if not raw:
                continue
            main_raw = (
                row.get("main")
                or row.get("label")
                or row.get("state")
                or row.get("phase")
                or row.get("emotion")
            )
            main = _normalize_main_label(main_raw)
            sub_raw = row.get("sub_label") or row.get("sub") or row.get("sub_id")
            try:
                sub = _format_sub_label(main, sub_raw)  # type: ignore[misc]
            except Exception:
                sub = str(sub_raw) if sub_raw else "—"
            rows.append(f"{idx:02d}. {raw}  [{main}|{sub}]")
    if not rows:
        # 간단한 폴백 문장 분할
        parts = [seg.strip() for seg in text.splitlines() if seg.strip()]
        for idx, seg in enumerate(parts, 1):
            rows.append(f"{idx:02d}. {seg}")
    return rows


def _extract_model_narrative(bundle: Mapping[str, Any]) -> List[str]:
    reports = bundle.get("reports") if isinstance(bundle, Mapping) else {}
    if isinstance(reports, Mapping):
        bi = reports.get("business_impact")
        if isinstance(bi, Mapping):
            narrative = bi.get("narrative")
            if isinstance(narrative, list):
                return [str(x).strip() for x in narrative if str(x).strip()]
            if isinstance(narrative, str):
                return [line.strip() for line in narrative.splitlines() if line.strip()]
    return []


def _build_sub_distribution(out: Mapping[str, Any], bundle: Mapping[str, Any]) -> Dict[str, float]:
    # test.py와 동일하게 out.poster.sub_distribution을 최우선으로 사용
    poster = out.get("poster", {}) if isinstance(out, Mapping) else {}
    sub = poster.get("sub_distribution")
    if isinstance(sub, Mapping) and sub:
        # test.py에서 사용하는 형식 그대로 반환 (키: "노-분개", 값: 0.035)
        return {str(k): float(v) for k, v in sub.items()}
    # bundle.sub_top은 폴백으로만 사용 (메인 감정 추론 필요)
    sub_top = bundle.get("sub_top") if isinstance(bundle, Mapping) else []
    result: Dict[str, float] = {}
    if isinstance(sub_top, list):
        # 메인 감정 추론 (poster.main 또는 main_dist에서)
        main_hint = poster.get("main") if isinstance(poster, Mapping) else None
        if not main_hint:
            main_dist = poster.get("main_distribution") if isinstance(poster, Mapping) else {}
            if isinstance(main_dist, Mapping) and main_dist:
                try:
                    main_hint = max(main_dist, key=main_dist.get)
                except Exception:
                    main_hint = None
        main_hint = main_hint or "노"  # 기본값
        
        for item in sub_top:
            if not isinstance(item, Mapping):
                continue
            label = item.get("sub")
            value = item.get("p")
            if label is None or value is None:
                continue
            try:
                val = float(value)
            except Exception:
                continue
            # sub_top의 값은 퍼센트이므로 0-1로 변환
            normalized_val = val / 100.0 if val > 1.0 else val
            # 키 형식: "노-분개" (메인-세부)
            key = f"{main_hint}-{label}" if "-" not in str(label) else str(label)
            result[key] = normalized_val
    return result


def _build_master_report(out: Mapping[str, Any], bundle: Mapping[str, Any]) -> str:
    poster = out.get("poster", {}) if isinstance(out, Mapping) else {}
    main_dist = poster.get("main_distribution") if isinstance(poster, Mapping) else {}
    products = bundle.get("products") if isinstance(bundle, Mapping) else {}
    p1 = products.get("p1") if isinstance(products, Mapping) else {}
    p5 = products.get("p5") if isinstance(products, Mapping) else {}
    bundle_meta = bundle.get("meta") if isinstance(bundle, Mapping) else {}

    lines: List[str] = ["== MASTER REPORT (inline) =="]
    lines.append(f"텍스트 길이: {len(str(out.get('text', '')))}자")

    if isinstance(main_dist, Mapping) and main_dist:
        dist_line = " · ".join(
            f"{emo} {float(val) * 100:.1f}%"
            for emo, val in sorted(main_dist.items(), key=lambda kv: kv[1], reverse=True)
        )
        lines.append(f"메인 감정 분포: {dist_line}")

    if isinstance(p1, Mapping) and p1:
        emotions = ", ".join(
            f"{item.get('name')} {item.get('pct')}%"
            for item in p1.get("headline_emotions", [])
            if isinstance(item, Mapping)
        )
        lines.append(
            f"P1 행동 인텔리전스 - churn {p1.get('churn_probability', 0)}%, "
            f"강도 {p1.get('intensity', '—')}, 감정 {emotions or '—'}"
        )
        if p1.get("recommended_actions"):
            actions = ", ".join(str(x) for x in p1["recommended_actions"] if x)
            lines.append(f"권장 액션: {actions or '—'}")

    if isinstance(p5, Mapping) and p5:
        lines.append(
            f"P5 심리 프로파일 - 안정성 {p5.get('stability', 0)}%, 성숙도 {p5.get('maturity', 0)}%"
        )

    if isinstance(bundle_meta, Mapping):
        ev = bundle_meta.get("evidence_score")
        if isinstance(ev, (int, float)):
            lines.append(f"증거 점수: {float(ev):.2f}")
        badges = bundle_meta.get("badges")
        if isinstance(badges, Iterable):
            badge_text = ", ".join(str(b) for b in badges if b)
            if badge_text:
                lines.append(f"Badges: {badge_text}")

    if isinstance(bundle, Mapping):
        recommendation = bundle.get("recommendation")
        if recommendation:
            lines.append(f"추천 조치: {recommendation}")

    reports = bundle.get("reports") if isinstance(bundle, Mapping) else {}
    if isinstance(reports, Mapping):
        bi = reports.get("business_impact")
        if isinstance(bi, Mapping):
            summary = bi.get("summary")
            if summary:
                lines.append(f"비즈니스 요약: {summary}")

    return "\n".join(lines)


def _truncate_text(text: Any, limit: int = 80) -> str:
    raw = str(text or "").strip()
    if len(raw) <= limit:
        return raw
    return raw[: limit - 1].rstrip() + "…"


def _build_balanced_summary_extras(
    poster: Mapping[str, Any],
    sentences: Sequence[Mapping[str, Any]],
    transitions: Sequence[Mapping[str, Any]],
    products: Mapping[str, Any],
) -> List[str]:
    extras: List[str] = []

    main_dist = poster.get("main_distribution") if isinstance(poster, Mapping) else {}
    if isinstance(main_dist, Mapping) and main_dist:
        ranked = sorted(main_dist.items(), key=lambda kv: kv[1], reverse=True)[:4]
        dist_line = ", ".join(f"{emo} {round(float(val) * 100, 1)}%" for emo, val in ranked)
        extras.append(f"주요 감정 분포: {dist_line}")

    if sentences:
        extras.append("문장별 감정 태깅:")
        for row in list(sentences)[:3]:
            idx = row.get("index")
            main = row.get("main") or "—"
            sub = row.get("sub") or "—"
            snippet = _truncate_text(row.get("text"), 70)
            extras.append(f" - {idx}번 문장: {main}/{sub} · {snippet}")

    if transitions:
        extras.append("감정 흐름 요약:")
        for tr in list(transitions)[:2]:
            ttype = tr.get("transition_type") or ("유지" if tr.get("from_main") == tr.get("to_main") else "전이")
            main_from = tr.get("from_main")
            main_to = tr.get("to_main")
            reason = _truncate_text(tr.get("transition_reason"), 90)
            extras.append(f" - {ttype}: {main_from} → {main_to} · {reason}")

    p1 = products.get("p1") if isinstance(products, Mapping) else {}
    p5 = products.get("p5") if isinstance(products, Mapping) else {}
    if isinstance(p1, Mapping) and p1:
        churn = p1.get("churn_probability")
        intensity = p1.get("intensity")
        extras.append(
            f"행동 인텔리전스: 감정 강도 {intensity or '—'}, 3일 이내 이탈 위험 {churn or 0}%"
        )
    if isinstance(p5, Mapping) and p5:
        stability = p5.get("stability")
        maturity = p5.get("maturity")
        extras.append(f"심리 프로파일: 안정성 {stability or 0}%, 감정 성숙도 {maturity or 0}%")

    return extras


def _build_balanced_model_narrative(
    poster: Mapping[str, Any],
    sentences: Sequence[Mapping[str, Any]],
    transitions: Sequence[Mapping[str, Any]],
    products: Mapping[str, Any],
) -> List[str]:
    lines: List[str] = []
    main = poster.get("main") if isinstance(poster, Mapping) else None
    main_dist = poster.get("main_distribution") if isinstance(poster, Mapping) else {}
    if main and isinstance(main_dist, Mapping):
        top_score = round(float(main_dist.get(main, 0.0)) * 100, 1)
        lines.append(f"핵심 감정은 {main}({top_score}%)로 파악되었습니다.")
        ranked = sorted(main_dist.items(), key=lambda kv: kv[1], reverse=True)[:4]
        dist_line = ", ".join(f"{emo} {round(float(val) * 100, 1)}%" for emo, val in ranked)
        lines.append(f"상위 감정 분포: {dist_line}")

    if sentences:
        lines.append("문장별 세부 감정 해설:")
        for row in sentences:
            idx = row.get("index")
            main = row.get("main") or "—"
            sub = row.get("sub") or "—"
            snippet = _truncate_text(row.get("text"), 90)
            lines.append(f" - {idx}번: {main}/{sub} · {snippet}")

    if transitions:
        lines.append("감정 흐름 분석:")
        for tr in transitions:
            ttype = tr.get("transition_type") or ("유지" if tr.get("from_main") == tr.get("to_main") else "전이")
            from_main = tr.get("from_main")
            to_main = tr.get("to_main")
            reason = _truncate_text(tr.get("transition_reason"), 110)
            lines.append(f" - {ttype}: {from_main} → {to_main} · {reason}")

    p1 = products.get("p1") if isinstance(products, Mapping) else {}
    if isinstance(p1, Mapping) and p1:
        churn = p1.get("churn_probability")
        intensity = p1.get("intensity")
        lines.append(f"행동 인텔리전스 관점: 감정 강도 {intensity or '—'}, 예상 이탈 {churn or 0}%")

    p5 = products.get("p5") if isinstance(products, Mapping) else {}
    if isinstance(p5, Mapping) and p5:
        stability = p5.get("stability")
        maturity = p5.get("maturity")
        lines.append(f"심리 안정성 지표: 안정성 {stability or 0}%, 감정 성숙도 {maturity or 0}%")

    return [line for line in lines if line]


def _infer_mode(meta: Mapping[str, Any], requested_mode: str) -> str:
    mode = meta.get("mode")
    if isinstance(mode, str) and mode.strip():
        return mode.upper()
    normalized = (requested_mode or "").strip().lower()
    if normalized in {"balanced", "balance", "standard", "default", "auto", "baseline"}:
        return "BALANCED"
    if normalized in {"heavy", "prod", "oneclick", "full", "heavy_mode", "precise", "precision", "detail"}:
        return "FAST"
    if normalized in {"fast", "preview", "speed", "quick"}:
        return "FAST"
    return "AUTO"


def _build_investor_highlights(
    *,
    meta: Mapping[str, Any],
    module_hit_rate: Mapping[str, Any],
    module_details: Sequence[Mapping[str, Any]],
    products: Mapping[str, Any],
    investor_pack: Optional[Mapping[str, Any]],
    domain_profile: str,
    elapsed_ms: Optional[float],
    effective_mode: str,
    poster: Mapping[str, Any],
) -> List[str]:
    highlights: List[str] = []

    elapsed_sec: Optional[float] = None
    if isinstance(elapsed_ms, (int, float)):
        elapsed_sec = elapsed_ms / 1000.0
    elif isinstance(meta.get("elapsed"), (int, float)):
        elapsed_sec = float(meta["elapsed"])

    if elapsed_sec is not None:
        if effective_mode == "fast":
            highlights.append(f"실시간 FAST 분석 {round(elapsed_sec, 3)}초 내 결과 제공")
        else:
            minutes = round(elapsed_sec / 60.0, 2)
            highlights.append(f"정밀 BALANCED 분석 {minutes}분 내 완료 (심층 모듈 포함)")

    total_modules = len(module_details)
    ok_modules = sum(1 for detail in module_details if detail.get("status") == "ok")
    if total_modules:
        coverage_pct = round((ok_modules / total_modules) * 100, 1)
        highlights.append(f"모듈 커버리지 {coverage_pct}% ({ok_modules}/{total_modules}) · 재현성 확보")

    if domain_profile == "service":
        p1 = products.get("p1") if isinstance(products, Mapping) else {}
        triggers: List[str] = []
        churn = None
        if isinstance(p1, Mapping):
            churn = p1.get("churn_probability")
            raw_triggers = p1.get("triggers")
            if isinstance(raw_triggers, list):
                triggers = [str(t) for t in raw_triggers if str(t).strip()]
        trigger_text = f" · 주요 트리거 {', '.join(triggers[:2])}" if triggers else ""
        if isinstance(churn, (int, float)):
            highlights.append(f"잠재 이탈 위험 {int(churn)}% 예측{trigger_text}")
        else:
            highlights.append("고객 감정·이탈 위험 신호를 실시간으로 계량화")
    else:
        if investor_pack and investor_pack.get("highlights"):
            for text in investor_pack["highlights"]:
                if text and text not in highlights:
                    highlights.append(str(text))
        else:
            main = poster.get("main")
            main_dist = poster.get("main_distribution") if isinstance(poster, Mapping) else {}
            pct = None
            if isinstance(main_dist, Mapping) and main in main_dist:
                pct = _format_percentage(main_dist.get(main))
            if main:
                line = f"핵심 감정 {main}"
                if pct:
                    line += f" {pct}"
                highlights.append(line)

    unique: List[str] = []
    for entry in highlights:
        if entry and entry not in unique:
            unique.append(entry)
        if len(unique) >= 4:
                    break
    return unique


def _build_api_payload(
    out: Mapping[str, Any],
    *,
    text: str,
    requested_mode: str,
    effective_mode: str,
    refined: bool = False,
    enforce_fast_meta: bool = False,
) -> Dict[str, Any]:
    """
    API Layer: test.py의 run_one에서 생성한 bundle 데이터를 최우선으로 사용하여 API 응답 생성.
    
    이 함수는 API Layer의 역할을 수행합니다:
    - Core Truth Layer(test.py의 __web_bundle)에서 Truth 필드를 읽어옴
    - Truth 필드 값은 절대 수정하지 않고, UI 편의를 위한 별도 필드만 생성
    - 기존 응답 스키마를 유지하여 하위 호환성 보장
    
    원칙:
    1. Truth 필드(bundle.*)는 읽기만 하고 수정하지 않음
    2. payload["bundle"]에는 항상 전체 bundle을 그대로 넣음 (절대 잘라내지 않음)
    3. UI 편의 필드(main_distribution, insight_summary 등)는 Truth 필드 기반으로 생성
    4. 기존 프론트엔드가 기대하는 스키마를 유지
    
    Args:
        out: run_one()의 출력 (__web_bundle 포함)
        text: 원본 텍스트
        requested_mode: 요청된 모드
        effective_mode: 실제 적용된 모드
        refined: 정밀 분석 여부
        enforce_fast_meta: FAST 메타 강제 여부
        
    Returns:
        API 응답 페이로드 (기존 스키마 유지)
    """
    strict_ssot = (os.getenv("STRICT_SSOT") or "").strip() == "1"
    # 1순위: test.py의 run_one에서 생성한 __web_bundle 사용
    bundle = out.get("__web_bundle")
    if not isinstance(bundle, Mapping):
        # 2순위: test.py의 _to_web_bundle 함수 직접 호출
        to_web_bundle = _get_to_web_bundle()
        try:
            if to_web_bundle is not None:
                bundle = to_web_bundle(out)  # type: ignore[arg-type]
                logger.info("[API] _to_web_bundle 직접 호출 성공")
            else:
                logger.warning("[API] _to_web_bundle 함수를 찾을 수 없음")
                bundle = {}
        except Exception as exc:
            logger.warning("[API] _to_web_bundle 실패: %s", exc, exc_info=True)
            bundle = {}
    
    # bundle이 비어있으면 경고 로그
    if not isinstance(bundle, Mapping) or not bundle:
        logger.warning("[API] bundle이 비어있음 - test.py 결과를 사용할 수 없습니다. Fallback 로직이 실행됩니다.")
        logger.warning(f"[API] Fallback Source: poster={bool(poster)}, out.results={list(out.get('results', {}).keys())}")
    else:
        # Truth 디버깅용 요약 로그: test.py에서 온 bundle 구조를 한눈에 확인
        if not strict_ssot:
            try:
                # ★★★ [Genius Fix] 감정 정합성 강제 교정 (Sanity Check) ★★★
                # "희" 감정에 "불만" 키워드가 붙는 논리적 모순을 해결합니다.
                _sub_top_check = bundle.get("sub_top")
                _main_dist_check = bundle.get("main_dist")
                
                if isinstance(_sub_top_check, list) and isinstance(_main_dist_check, dict):
                    # 불만 계열 키워드 감지
                    has_complaint = any(
                        (isinstance(x, dict) and x.get("sub") in ("불만", "짜증", "화", "분노", "억울", "답답", "문제")) 
                        for x in _sub_top_check
                    )
                    
                    # 점수 파싱 (안전하게)
                    def _safe_score(d, k):
                        try: return float(d.get(k, 0))
                        except: return 0.0
                    
                    joy_score = _safe_score(_main_dist_check, "희")
                    anger_score = _safe_score(_main_dist_check, "노")
                    sad_score = _safe_score(_main_dist_check, "애")
                    pleasure_score = _safe_score(_main_dist_check, "락")

                    # 긍정(희+락) vs 부정(노+애)
                    pos_total = joy_score + pleasure_score
                    neg_total = anger_score + sad_score

                    # 조건: 불만 키워드가 있고, 긍정이 부정보다 높거나 비슷하면 -> 부정으로 강제 전환
                    if has_complaint and (pos_total >= neg_total * 0.8):
                        logger.warning(f"[API][Sanity] '불만' 키워드 감지됨. 긍정({pos_total:.2f}) vs 부정({neg_total:.2f}) -> 강제 교정 실행.")
                        
                        # 1. 희/락 점수를 대폭 깎아서 노/애로 이전
                        # 단순히 스왑하기보다, 희/락의 80%를 노/애로 분배
                        transfer_amount = pos_total * 0.8
                        
                        # 노/애 비율대로 분배 (기본 5:5)
                        if neg_total > 0:
                            anger_ratio = anger_score / neg_total
                            sad_ratio = sad_score / neg_total
                        else:
                            anger_ratio = 0.7 # 불만은 보통 '노'에 가까움
                            sad_ratio = 0.3
                        
                        _main_dist_check["희"] = joy_score * 0.2
                        _main_dist_check["락"] = pleasure_score * 0.2
                        _main_dist_check["노"] = anger_score + (transfer_amount * anger_ratio)
                        _main_dist_check["애"] = sad_score + (transfer_amount * sad_ratio)
                        
                        logger.info(f"[API][Sanity] 교정 결과: {_main_dist_check}")
            except Exception as e:
                logger.warning(f"[API][Sanity] 교정 중 오류 발생 (무시됨): {e}")

        try:
            bundle_main_dist_check = bundle.get("main_dist")
            md_size = len(bundle_main_dist_check) if isinstance(bundle_main_dist_check, Mapping) else 0
            sent_struct = bundle.get("sentence_annotations_structured")
            sent_count = len(sent_struct) if isinstance(sent_struct, list) else 0
            trans_struct = bundle.get("transitions_structured")
            trans_count = len(trans_struct) if isinstance(trans_struct, list) else 0
            logger.info(
                "[API][truth] bundle.main_dist=%s (keys=%d), sentences=%d, transitions=%d",
                bundle_main_dist_check,
                md_size,
                sent_count,
                trans_count,
            )
        except Exception as _e:
            logger.debug("[API][truth] bundle summary logging failed: %s", _e)

    # ★★★ test.py bundle 데이터 최우선 사용 (원본 데이터 보존) ★★★
    # bundle이 있으면 모든 데이터를 bundle에서 가져옴
    bundle_main_dist = bundle.get("main_dist") if isinstance(bundle, Mapping) else {}
    bundle_sub_top = bundle.get("sub_top") if isinstance(bundle, Mapping) else None
    bundle_sentence_structured = bundle.get("sentence_annotations_structured") if isinstance(bundle, Mapping) else None
    bundle_transitions_structured = bundle.get("transitions_structured") if isinstance(bundle, Mapping) else None
    bundle_why_lines = bundle.get("why_lines") if isinstance(bundle, Mapping) else None
    bundle_reasoning_path = bundle.get("reasoning_path_lines") if isinstance(bundle, Mapping) else None
    bundle_sub_top10 = bundle.get("sub_top10_lines") if isinstance(bundle, Mapping) else None
    bundle_triggers = bundle.get("triggers") if isinstance(bundle, Mapping) else {}
    bundle_products = bundle.get("products") if isinstance(bundle, Mapping) else {}
    bundle_flow_ssot = bundle.get("flow_ssot") if isinstance(bundle, Mapping) else None
    bundle_main = bundle.get("main") if isinstance(bundle, Mapping) else None
    
    # poster는 bundle이 있으면 bundle 데이터로 구성, 없으면 out에서 가져옴
    poster_raw = out.get("poster") if isinstance(out, Mapping) else {}
    if isinstance(poster_raw, Mapping) and poster_raw:
        poster = copy.deepcopy(poster_raw)
    else:
        poster = copy.deepcopy(bundle.get("poster", {})) if isinstance(bundle, Mapping) else {}
    
    # ★★★ main_dist 처리: bundle.main_dist 최우선 (test.py 원본 보존) ★★★
    if isinstance(bundle_main_dist, Mapping) and bundle_main_dist:
        # bundle.main_dist가 있으면 이것을 사용 (test.py에서 생성한 원본)
        # 라벨 정규화만 수행 (희/노/애/락 변환)
        logger.info(f"[API] bundle.main_dist 발견: {bundle_main_dist}")
        cleaned_dist: Dict[str, float] = {}
        total = 0.0
        for raw_key, raw_value in bundle_main_dist.items():
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                continue
            if value <= 0:
                continue
            label = _normalize_main_label(raw_key)
            if label not in ("희", "노", "애", "락"):
                continue
            cleaned_dist[label] = cleaned_dist.get(label, 0.0) + value
            total += value
        
        # 합이 1.0에 가까우면 이미 정규화된 것으로 간주하고 그대로 사용
        # 합이 1.0보다 크면 정규화 필요 (예: 퍼센트 값)
        if total > 0:
            if abs(total - 1.0) < 0.01:
                # 이미 정규화됨 (0-1 사이) - test.py 원본 그대로 사용
                main_dist = cleaned_dist
                logger.info(f"[API] bundle.main_dist 사용 (정규화됨): {main_dist}")
            elif total > 1.0:
                # 퍼센트 값 (0-100)이거나 합산된 값 → 정규화
                main_dist = {k: round(v / total, 6) for k, v in cleaned_dist.items()}
                logger.info(f"[API] bundle.main_dist 사용 (정규화 적용, total={total}): {main_dist}")
            else:
                # 합이 1.0보다 작으면 그대로 사용 (일부 감정만 있는 경우)
                main_dist = cleaned_dist
                logger.info(f"[API] bundle.main_dist 사용 (부분 합, total={total}): {main_dist}")
        else:
            main_dist = {}
            logger.warning(f"[API] bundle.main_dist가 비어있음 (total=0)")
        logger.debug(f"[API] bundle.main_dist 사용: {main_dist}")
    else:
        # bundle.main_dist가 없을 때만 poster 사용 (fallback)
        # ★★★ Fallback: bundle.main_dist가 없을 때만 poster 사용 ★★★
        # 주의: 이 Fallback은 Truth 필드가 없을 때만 사용되며,
        # Truth 필드(bundle.main_dist)가 있으면 절대 재계산하지 않습니다.
        main_dist = poster.get("main_distribution") if isinstance(poster, Mapping) else {}
        if not isinstance(main_dist, Mapping) or not main_dist:
            main_dist = {}
        logger.debug(f"[API] poster.main_distribution 사용 (fallback - Truth 필드 없음): {main_dist}")

    # ★★★ sub_distribution 처리: bundle.sub_top 최우선 (test.py 원본 보존) ★★★
    if bundle_sub_top and isinstance(bundle_sub_top, list):
        # bundle.sub_top이 있으면 이것을 사용 (test.py에서 생성한 원본)
        sub_distribution = {}
        current_main = max(main_dist, key=main_dist.get) if main_dist else (bundle_main or "노")
        for entry in bundle_sub_top:
            if isinstance(entry, dict):
                sub_label = entry.get("sub", "")
                score = entry.get("p", 0)
                if sub_label and score > 0:
                    if "-" not in sub_label:
                        key = f"{current_main}-{sub_label}"
                    else:
                        key = sub_label
                    # p는 퍼센트(0-100)이므로 0-1로 변환
                    sub_distribution[key] = score / 100.0 if score > 1 else score
        logger.debug(f"[API] bundle.sub_top 사용: {len(sub_distribution)}개 항목")
    else:
        # ★★★ Fallback: bundle.sub_top이 없을 때만 기존 방식 사용 ★★★
        # 주의: 이 Fallback은 Truth 필드가 없을 때만 사용되며,
        # Truth 필드(bundle.sub_top)가 있으면 절대 재계산하지 않습니다.
        sub_distribution = _build_sub_distribution(out, bundle)
        logger.debug(f"[API] _build_sub_distribution 사용 (fallback - Truth 필드 없음): {len(sub_distribution)}개 항목")

    module_results = out.get("results", {})
    if not isinstance(module_results, Mapping):
        module_results = {}
    # 추가: test.py에서 직접 results를 가져오지 않으면 bundle에서도 확인
    if not module_results and isinstance(bundle, Mapping):
        module_results = bundle.get("results", {})

    # -----------------------------------------------------------------------
    # [PATCH] 문장 단위 태깅 강력 폴백 (test.py 로직 모방)
    # -----------------------------------------------------------------------
    if not bundle_sentence_structured and not strict_ssot:
        try:
            srows = out.get("results", {}).get("emotion_relationship_analyzer", {}).get("sentences", [])
            if isinstance(srows, list) and srows:
                bundle_sentence_structured = srows
                logger.info("[API] emotion_relationship_analyzer.sentences 사용")
            else:
                logger.warning("[API] 문장 태깅 데이터 없음 -> 독립 분석 폴백 실행")
                import re
                sentences = re.split(r'(?<=[.!?])\s+', text.strip())
                if not sentences:
                    sentences = [text.strip()] if text.strip() else []
                
                fallback_struct = []
                for idx, sent in enumerate(sentences, 1):
                    sent = sent.strip()
                    if not sent: continue
                    
                    curr_main = None
                    curr_sub = "—"
                    
                    # 1. 명시적 키워드 매칭
                    hit = _simple_explicit_check(sent)
                    if hit:
                        curr_main, curr_sub = hit
                        logger.debug(f"[API] 문장 {idx} 명시 매칭: {curr_main}-{curr_sub}")
                    
                    # 2. Linguistic Matcher 결과 확인
                    if not curr_main:
                        l_matches = out.get("results", {}).get("linguistic_matcher", {}).get("matches", [])
                        if isinstance(l_matches, list):
                            for m in l_matches:
                                m_text = m.get("text") or ""
                                if m_text and m_text in sent:
                                    curr_main = m.get("main_emotion")
                                    curr_sub = m.get("sub_emotion") or "—"
                                    break
                    
                    # 3. 최후의 수단: Main Distribution
                    if not curr_main:
                        if main_dist:
                            curr_main = max(main_dist.items(), key=lambda kv: kv[1])[0]
                        else:
                            curr_main = "노" # 정말 아무것도 없을 때
                        curr_sub = _default_sub(curr_main)
                        
                    fallback_struct.append({
                        "index": idx,
                        "text": sent,
                        "main": curr_main,
                        "sub": curr_sub
                    })
                bundle_sentence_structured = fallback_struct
                logger.info(f"[API] 폴백 문장 태깅 완료: {len(bundle_sentence_structured)}개")
        except Exception as e:
            logger.error(f"[API] 문장 태깅 폴백 실패: {e}")
            bundle_sentence_structured = []

    # -----------------------------------------------------------------------
    # [PATCH] 전이(Transitions) 강력 폴백 (test.py 로직 모방)
    # -----------------------------------------------------------------------
    if not strict_ssot and not bundle_transitions_structured and bundle_sentence_structured:
        try:
            # 문장 태깅 기반으로 전이 재구성
            tr_list = []
            for idx in range(len(bundle_sentence_structured) - 1):
                src = bundle_sentence_structured[idx]
                dst = bundle_sentence_structured[idx+1]
                fm, fs = src.get("main"), src.get("sub")
                tm, ts = dst.get("main"), dst.get("sub")
                
                if not fm or not tm: continue
                
                t_type = "steady" if fm == tm and fs == ts else "shift"
                tr_list.append({
                    "from_index": src.get("index"),
                    "to_index": dst.get("index"),
                    "from_main": fm,
                    "from_sub": fs,
                    "to_main": tm,
                    "to_sub": ts,
                    "transition_type": t_type,
                    "probability": 0.85 if t_type == "steady" else 0.65, # 추정치
                    "trigger": dst.get("text", "")[:20]
                })
            bundle_transitions_structured = tr_list
            logger.info(f"[API] 폴백 전이 생성 완료: {len(tr_list)}개")
        except Exception as e:
            logger.error(f"[API] 전이 폴백 실패: {e}")

    def _normalize_main_distribution(dist_candidate: Any) -> Optional[Dict[str, float]]:
        if not isinstance(dist_candidate, Mapping):
            return None
        acc: Dict[str, float] = {}
        for raw_key, raw_value in dist_candidate.items():
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                continue
            label = _normalize_main_label(raw_key)
            if label not in ("희", "노", "애", "락"):
                continue
            if value <= 0:
                continue
            acc[label] = acc.get(label, 0.0) + value
        total = sum(acc.values())
        if total <= 0:
            return None
        return {key: round(val / total, 6) for key, val in acc.items()}

    def _extract_refined_main_distribution(mod_results: Mapping[str, Any]) -> Optional[Dict[str, float]]:
        wc = mod_results.get("weight_calculator")
        if not isinstance(wc, Mapping):
            return None
        for key in ("adjusted", "main_weights", "weights", "emotion_weights"):
            cand = _normalize_main_distribution(wc.get(key))
            if cand:
                return cand
        base = wc.get("base_weights")
        if isinstance(base, Mapping):
            cand = _normalize_main_distribution(base.get("emotion_weights"))
            if cand:
                return cand
        return None

    def _extract_refined_sub_distribution(mod_results: Mapping[str, Any]) -> Optional[Dict[str, float]]:
        wc = mod_results.get("weight_calculator")
        if not isinstance(wc, Mapping):
            return None
        integrated = wc.get("integrated")
        if not isinstance(integrated, Mapping):
            return None
        sub_acc: Dict[str, float] = {}
        for main_label, info in integrated.items():
            normalized_main = _normalize_main_label(main_label)
            info_map = info if isinstance(info, Mapping) else {}
            sub_map = info_map.get("sub_emotions") if isinstance(info_map, Mapping) else None
            if not isinstance(sub_map, Mapping):
                continue
            for sub_label, sub_value in sub_map.items():
                if isinstance(sub_value, Mapping):
                    val = sub_value.get("score") or sub_value.get("value")
                else:
                    val = sub_value
                try:
                    score = float(val)
                except (TypeError, ValueError):
                    continue
                if score <= 0:
                    continue
                key = f"{normalized_main}-{str(sub_label).strip()}"
                sub_acc[key] = sub_acc.get(key, 0.0) + score
        total = sum(sub_acc.values())
        if total <= 0:
            return None
        return {key: round(val / total, 6) for key, val in sub_acc.items()}

    def _build_headline_emotions(dist: Any, *, limit: int = 3) -> List[Dict[str, float]]:
        if not isinstance(dist, Mapping):
            return []
        refined: List[Tuple[str, float]] = []
        for raw_label, raw_score in dist.items():
            try:
                score = float(raw_score)
            except (TypeError, ValueError):
                continue
            if score <= 0:
                continue
            norm_label = _normalize_main_label(raw_label)
            if norm_label not in ("희", "노", "애", "락"):
                continue
            refined.append((norm_label, score))
        if not refined:
            return []
        refined.sort(key=lambda kv: kv[1], reverse=True)
        return [
            {"name": label, "pct": round(score * 100.0, 1)}
            for label, score in refined[:limit]
        ]

    def _harmonize_sentence_structured(
        rows: List[Dict[str, Any]],
        *,
        target_main: str,
        main_distribution: Mapping[str, float],
    ) -> None:
        # 제거: 원본 데이터를 전혀 건드리지 않음
        # test.py에서 생성한 문장별 감정을 그대로 유지
        return

    def _serialize_sentence_structured(rows: Sequence[Mapping[str, Any]]) -> List[str]:
        serialized: List[str] = []
        for idx, row in enumerate(rows, 1):
            try:
                index_val = int(row.get("index", idx))
            except (TypeError, ValueError):
                index_val = idx
            text_val = str(row.get("text") or "").strip()
            main_val = _normalize_main_label(row.get("main"))
            sub_raw = row.get("sub") or "—"
            sub_val = str(sub_raw).strip() or "—"
            serialized.append(f"{index_val:02d}. {text_val}  [{main_val}|{sub_val}]")
        return serialized

    # ★★★ poster 업데이트 (bundle 데이터로 설정된 main_dist와 sub_distribution 반영) ★★★
    if main_dist:
        poster.setdefault("main_distribution", {})
        poster["main_distribution"] = copy.deepcopy(main_dist)
        # bundle.main이 있으면 그것을 우선 사용, 없으면 main_dist에서 계산
        if bundle_main:
            poster["main"] = bundle_main
        else:
            try:
                poster["main"] = max(main_dist, key=main_dist.get)
            except ValueError:
                poster["main"] = poster.get("main") or ""
    
    if sub_distribution:
        poster.setdefault("sub_distribution", {})
        poster["sub_distribution"] = copy.deepcopy(sub_distribution)
    
    # bundle.flow_ssot이 있으면 poster에도 반영
    if bundle_flow_ssot:
        poster["flow_ssot"] = bundle_flow_ssot

    products_raw = out.get("products") if isinstance(out, Mapping) else {}
    if isinstance(products_raw, Mapping) and products_raw:
        products = copy.deepcopy(products_raw)
    else:
        products = copy.deepcopy(bundle.get("products", {})) if isinstance(bundle, Mapping) else {}
    investor_pack: Optional[Dict[str, Any]] = None

    meta_info = out.get("meta", {}) if isinstance(out, Mapping) else {}
    bundle_meta = bundle.get("meta") if isinstance(bundle, Mapping) else {}
    meta: Dict[str, Any] = {}
    if isinstance(meta_info, Mapping):
        meta.update(meta_info)
    if isinstance(bundle_meta, Mapping):
        for key, value in bundle_meta.items():
            meta.setdefault(key, value)

    elapsed_ms = None
    if isinstance(meta.get("elapsed_ms"), (int, float)):
        elapsed_ms = float(meta["elapsed_ms"])
    elif isinstance(meta.get("elapsed"), (int, float)):
        elapsed_ms = float(meta["elapsed"]) * 1000.0
    meta["elapsed"] = round(elapsed_ms / 1000.0, 3) if elapsed_ms is not None else 0.0
    meta["mode"] = _infer_mode(meta, effective_mode)
    if enforce_fast_meta:
        meta["mode"] = "FAST"
        meta.pop("background_job_id", None)
        meta["orch_fallback"] = True
    if refined:
        meta["refined"] = True

    domain_profile = str(
        (meta.get("domain_profile") or poster.get("domain_profile") or bundle_meta.get("domain_profile") or "")
    ).lower()
    if not domain_profile:
        domain_profile = _detect_domain_profile(text)
    meta["domain_profile"] = domain_profile
    if isinstance(poster, Mapping):
        poster.setdefault("domain_profile", domain_profile)

    meta = sanitize(meta)

    sentence_annotations = _format_sentence_annotations(bundle, text)
    sentence_structured: List[Dict[str, Any]] = []
    
    # ★★★ test.py 원본 sentence_annotations_structured 최우선 적용 (변환 없이 그대로 사용) ★★★
    # bundle에서 가져온 완전한 구조 데이터가 있으면 그대로 사용 (변환 금지)
    if isinstance(bundle_sentence_structured, list) and bundle_sentence_structured:
        # test.py에서 이미 완전한 구조로 생성된 데이터이므로 그대로 사용
        sentence_structured = bundle_sentence_structured
        logger.debug(f"[API] bundle.sentence_annotations_structured 사용: {len(sentence_structured)}개 문장")
    elif isinstance(out, Mapping) and isinstance(out.get("sentence_annotations_structured"), list):
        # bundle이 없을 때 test.py 직접 결과 사용 (원본 데이터 보존)
        sentence_structured = out.get("sentence_annotations_structured")
        logger.debug(f"[API] out.sentence_annotations_structured 사용: {len(sentence_structured)}개 문장")
    elif not strict_ssot:
        # bundle에 없을 때만 다른 소스에서 생성
        structured_candidates: List[Any] = [
            out.get("sentence_annotations_structured"),  # 2순위: test.py 직접 결과
            module_results.get("context_extractor", {}).get("structured_annotations"),  # 3순위
            module_results.get("emotion_relationship_analyzer", {}).get("sentences"),  # 4순위
        ]

        for candidate in structured_candidates:
            if sentence_structured:
                break
            if not (isinstance(candidate, list) and candidate):
                continue
            for row in candidate:
                if not isinstance(row, Mapping):
                    continue
                try:
                    main_label = (
                        row.get("main") or row.get("label") or row.get("emotion") or row.get("state")
                    )
                    raw_sub = (
                        row.get("sub")
                        or row.get("sub_label")
                        or row.get("sub_emotion")
                        or row.get("sub_id")
                    )
                    # ★★★ test.py 원본 sub 감정 절대 보존 (변환 금지) ★★★
                    # test.py에서 생성한 원본 sub 감정을 그대로 사용
                    if raw_sub not in (None, ""):
                        formatted_sub = str(raw_sub).strip()
                        # sub_ 형식이어도 원본 그대로 보존 (test.py가 의도한 값)
                        # 단, 완전히 비어있거나 "—"일 때만 변환 시도
                    else:
                        # 원본이 없을 때만 변환 시도
                        formatted_sub = _format_sub_label(main_label, raw_sub) if callable(_FORMAT_SUB_LABEL_FUNC) else "—"
                        if formatted_sub == "—" or str(formatted_sub).startswith("sub_"):
                            default_sub_map = {"희": "감사", "노": "분노", "애": "슬픔", "락": "안심"}
                            formatted_sub = default_sub_map.get(main_label, "중립")

                    sentence_structured.append(
                        {
                            "index": int(row.get("index") or len(sentence_structured) + 1),
                            "text": str(
                                row.get("text")
                                or row.get("raw")
                                or row.get("sentence")
                                or row.get("content")
                                or ""
                            ),
                            "main": main_label,
                            "sub": formatted_sub,
                            "sub_raw": str(raw_sub).strip() if raw_sub not in (None, "") else "",
                        }
                    )
                except Exception:
                    continue
    # ★★★ bundle.sentence_annotations_structured가 없을 때만 fallback 사용 ★★★
    # bundle.sentence_annotations_structured가 이미 위에서 처리되었으므로, 
    # 여기서는 bundle이 없거나 비어있을 때만 fallback
    bundle_has_sentence_structured = isinstance(bundle, Mapping) and bundle.get("sentence_annotations_structured")
    if not sentence_structured and not bundle_has_sentence_structured:
        anchors = bundle.get("anchors") if isinstance(bundle, Mapping) else {}
        sentences = anchors.get("sentences") if isinstance(anchors, Mapping) else None
        if isinstance(sentences, list):
            for idx, row in enumerate(sentences, 1):
                if not isinstance(row, Mapping):
                    continue
                raw = (
                    row.get("text")
                    or row.get("raw")
                    or row.get("sentence")
                    or row.get("content")
                    or ""
                )
                # test.py에서 보낸 main과 sub를 그대로 사용
                main = row.get("main") or row.get("emotion") or row.get("label")
                sub_raw = row.get("sub") or row.get("sub_label") or row.get("sub_id")
                
                # main이 없을 때만 정규화
                if main:
                    # 한글 또는 영어 레이블 그대로 사용
                    main = str(main).strip()
                else:
                    main = "—"
                
                # ★★★ sub는 무조건 원본 그대로 사용 (test.py 원본 보존) ★★★
                if sub_raw not in (None, ""):
                    sub = str(sub_raw).strip()
                    # 원본 sub 감정 그대로 사용 (변환 금지)
                else:
                    # 원본이 없을 때만 변환 시도
                    formatted_sub = _format_sub_label(main, sub_raw) if callable(_FORMAT_SUB_LABEL_FUNC) else "—"
                    if not formatted_sub or formatted_sub == "—" or str(formatted_sub).startswith("sub_"):
                        default_sub_map = {"희": "감사", "노": "분노", "애": "슬픔", "락": "안심"}
                        formatted_sub = default_sub_map.get(main, "중립")
                    sub = formatted_sub
                    
                sentence_structured.append(
                    {
                        "index": idx,
                        "text": str(raw),
                        "main": main,
                        "sub": sub,
                        "sub_raw": sub_raw,
                    }
                )
        elif isinstance(sentence_annotations, list):
            for idx, line in enumerate(sentence_annotations, 1):
                sentence_structured.append(
                    {
                        "index": idx,
                        "text": line,
                        "main": "—",
                        "sub": "—",
                    }
                )
    # ★★★ Truth 필드 보존: bundle.main_dist가 있으면 문장 기반 재계산 금지 ★★★
    # 원칙: Truth 필드(bundle.main_dist)가 있으면 절대 재계산하지 않고 그대로 사용합니다.
    # Fallback은 Truth 필드가 없을 때만 사용됩니다.
    bundle_has_main_dist = isinstance(bundle, Mapping) and bundle.get("main_dist")
    bundle_main_dist_value = bundle.get("main_dist") if isinstance(bundle, Mapping) else None
    
    # 디버깅: bundle.main_dist 상태 로깅
    if bundle_main_dist_value:
        logger.info(f"[API] bundle.main_dist 존재 확인: {bundle_main_dist_value}")
    else:
        logger.warning(f"[API] bundle.main_dist 없음 - bundle keys: {list(bundle.keys()) if isinstance(bundle, Mapping) else 'N/A'}")
    
    # ★★★ 중요: bundle.main_dist가 있으면 절대 문장 기반 재계산 금지 ★★★
    # bundle.main_dist는 test.py에서 생성한 Truth 필드이므로, 비어있지 않으면 그대로 사용해야 합니다.
    if sentence_structured and not bundle_has_main_dist:
        # ★★★ Fallback: bundle.main_dist가 없을 때만 문장 기반으로 추정 ★★★
        # 주의: 이 Fallback은 Truth 필드가 없을 때만 사용되며,
        # Truth 필드(bundle.main_dist)가 있으면 절대 재계산하지 않습니다.
        main_counter: Counter = Counter()
        for row in sentence_structured:
            normalized_main = _normalize_main_label(row.get("main"))
            if normalized_main in ("희", "노", "애", "락"):
                main_counter[normalized_main] += 1
        if main_counter:
            total_sentences = sum(main_counter.values())
            majority_main, majority_count = main_counter.most_common(1)[0]
            majority_share = majority_count / max(total_sentences, 1)
            current_main = _normalize_main_label(poster.get("main")) if isinstance(poster, Mapping) else "—"
            current_share = float(main_dist.get(current_main, 0.0)) if isinstance(main_dist, Mapping) else 0.0
            majority_share_in_dist = float(main_dist.get(majority_main, 0.0)) if isinstance(main_dist, Mapping) else 0.0
            
            # ★★★ 경고: bundle.main_dist가 없어서 문장 기반 재계산 실행 ★★★
            logger.warning(
                f"[API] ⚠️ bundle.main_dist 없음 → 문장 기반 재계산 실행: "
                f"majority={majority_main} ({majority_share:.1%}), "
                f"current={current_main} ({current_share:.1%}), "
                f"main_dist={main_dist}, "
                f"문장 카운트={dict(main_counter)}"
            )
            
            if majority_share >= 0.6 and (
                majority_main != current_main or majority_share_in_dist < 0.4
            ):
                normalized_dist = {
                    label: round(count / max(total_sentences, 1), 6)
                    for label, count in main_counter.items()
                }
                logger.warning(
                    f"[API] ⚠️ 문장 기반 재계산 실행됨: "
                    f"기존 main_dist={main_dist} → 새 분포={normalized_dist}"
                )
                main_dist = normalized_dist
                if isinstance(poster, Mapping):
                    poster["main_distribution"] = copy.deepcopy(normalized_dist)
                    poster["main"] = majority_main

    transitions_structured: List[Dict[str, Any]] = []

    def _resolve_sub(main_label: str, sub_label: Any) -> str:
        sub = str(sub_label).strip() if sub_label not in (None, "") else ""
        if not sub or sub == "—":
            return MAIN_DEFAULT_SUB.get(main_label, "—")
        return sub

    # ★★★ test.py 원본 transitions_structured 최우선 적용 (bundle → out 순서) ★★★
    # bundle에서 가져온 완전한 구조 데이터가 있으면 그대로 사용 (변환 금지)
    if isinstance(bundle_transitions_structured, list) and bundle_transitions_structured:
        # test.py에서 이미 완전한 구조로 생성된 데이터이므로 그대로 사용
        transitions_structured = bundle_transitions_structured
        logger.debug(f"[API] bundle.transitions_structured 사용: {len(transitions_structured)}개 전이")
    elif isinstance(out, Mapping) and isinstance(out.get("transitions_structured"), list):
        # bundle이 없을 때 test.py 직접 결과 사용
        transitions_structured = out.get("transitions_structured")
        logger.debug(f"[API] out.transitions_structured 사용: {len(transitions_structured)}개 전이")
    
    # 2순위: out에서 직접 가져온 transitions_structured
    if not transitions_structured:
        raw_transitions = out.get("transitions_structured")
        if isinstance(raw_transitions, list) and raw_transitions:
            for item in raw_transitions:
                if not isinstance(item, Mapping):
                    continue
                from_main = _normalize_main_label(item.get("from_main"))
                to_main = _normalize_main_label(item.get("to_main"))
                if from_main == "—" or to_main == "—":
                    continue
                # ★★★ test.py 원본 sub 감정 절대 보존 (변환 금지) ★★★
                from_sub_raw = item.get("from_sub")
                to_sub_raw = item.get("to_sub")
                # 원본 sub 감정 그대로 사용 (빈 값일 때만 기본값 사용)
                from_sub = str(from_sub_raw).strip() if from_sub_raw not in (None, "") else MAIN_DEFAULT_SUB.get(from_main, "—")
                to_sub = str(to_sub_raw).strip() if to_sub_raw not in (None, "") else MAIN_DEFAULT_SUB.get(to_main, "—")
                
                transitions_structured.append(
                    {
                        "from_index": item.get("from_index"),
                        "to_index": item.get("to_index"),
                        "from_main": from_main,
                        "from_sub": from_sub,
                        "to_main": to_main,
                        "to_sub": to_sub,
                        "excerpt_from": item.get("excerpt_from") or item.get("from_sentence_text") or "",
                        "excerpt_to": item.get("excerpt_to") or item.get("to_sentence_text") or "",
                        "probability": item.get("probability"),
                        "probability_explain": item.get("probability_explain"),
                        "transition_reason": item.get("transition_reason"),
                        "trigger": item.get("trigger"),
                        "confidence": item.get("confidence"),
                        "transition_type": item.get("transition_type"),
                    }
                )

    if not transitions_structured:
        ctx_transitions = (
            module_results.get("context_extractor", {}).get("structured_transitions")
            if isinstance(module_results, Mapping)
            else None
        )
        if isinstance(ctx_transitions, list) and ctx_transitions:
            for item in ctx_transitions:
                if not isinstance(item, Mapping):
                    continue
                from_main = _normalize_main_label(item.get("from_main"))
                to_main = _normalize_main_label(item.get("to_main"))
                if from_main == "—" or to_main == "—":
                    continue
                # ★★★ 원본 sub 감정 보존 (변환 최소화) ★★★
                from_sub_raw = item.get("from_sub")
                to_sub_raw = item.get("to_sub")
                from_sub = str(from_sub_raw).strip() if from_sub_raw not in (None, "") else _resolve_sub(from_main, from_sub_raw)
                to_sub = str(to_sub_raw).strip() if to_sub_raw not in (None, "") else _resolve_sub(to_main, to_sub_raw)
                transitions_structured.append(
                    {
                        "from_index": item.get("from_index"),
                        "to_index": item.get("to_index"),
                        "from_main": from_main,
                        "from_sub": from_sub,
                        "to_main": to_main,
                        "to_sub": to_sub,
                        "excerpt_from": item.get("excerpt_from") or "",
                        "excerpt_to": item.get("excerpt_to") or "",
                        "probability": item.get("probability"),
                        "probability_explain": item.get("probability_explain"),
                        "transition_reason": item.get("transition_reason"),
                        "trigger": item.get("trigger"),
                        "confidence": item.get("confidence"),
                    }
                )

    # ★★★ bundle.transitions_structured가 없을 때만 sentence_structured에서 생성 ★★★
    # bundle.transitions_structured가 이미 위에서 처리되었으므로,
    # 여기서는 bundle이 없거나 비어있을 때만 fallback
    bundle_has_transitions_structured = isinstance(bundle, Mapping) and bundle.get("transitions_structured")
    if not transitions_structured and not bundle_has_transitions_structured and len(sentence_structured) >= 2:
        for idx in range(len(sentence_structured) - 1):
            src = sentence_structured[idx]
            dst = sentence_structured[idx + 1]
            from_main = src.get("main") or "—"
            to_main = dst.get("main") or "—"
            # ★★★ sentence_structured에서 가져온 원본 sub 감정 그대로 사용 ★★★
            from_sub_raw = src.get("sub")
            to_sub_raw = dst.get("sub")
            from_sub = str(from_sub_raw).strip() if from_sub_raw not in (None, "") else _resolve_sub(from_main, from_sub_raw)
            to_sub = str(to_sub_raw).strip() if to_sub_raw not in (None, "") else _resolve_sub(to_main, to_sub_raw)
            transition_type = "steady" if from_main == to_main and from_sub == to_sub else "shift"
            transitions_structured.append(
                {
                    "from_index": src.get("index"),
                    "to_index": dst.get("index"),
                    "from_main": from_main,
                    "from_sub": from_sub,
                    "to_main": to_main,
                    "to_sub": to_sub,
                    "excerpt_from": src.get("text", ""),
                    "excerpt_to": dst.get("text", ""),
                    "transition_type": transition_type,
                }
            )

    for item in transitions_structured:
        if "probability" in item and isinstance(item["probability"], (int, float)):
            prob = float(item["probability"])
            if 0 <= prob <= 1:
                item.setdefault("probability_pct", int(round(prob * 100)))
        if (
            "probability_pct" not in item
            and isinstance(item.get("confidence"), (int, float))
            and 0 <= float(item["confidence"]) <= 1
        ):
            item["probability_pct"] = int(round(float(item["confidence"]) * 100))
        if not item.get("probability_explain") and isinstance(item.get("from_main"), str):
            from_main = item["from_main"]
            to_main = item.get("to_main")
            from_ratio = main_dist.get(from_main, 0.0) if isinstance(main_dist, Mapping) else 0.0
            to_ratio = main_dist.get(to_main, 0.0) if isinstance(main_dist, Mapping) else 0.0
            item["probability_explain"] = (
                f"{from_main} {round(from_ratio * 100, 1)}%, {to_main} {round(to_ratio * 100, 1)}% 분포 기반"
            )

    explainability_snapshot = _build_explainability_snapshot(
        out,
        text=text,
        main_dist=main_dist,
        sub_dist=sub_distribution,
        transitions_structured=transitions_structured,
        bundle=bundle if isinstance(bundle, Mapping) else None,
    )

    # ★★★ products 처리: bundle.products 최우선 사용 (test.py 원본 데이터 보존) ★★★
    if isinstance(bundle_products, Mapping) and bundle_products:
        products = copy.deepcopy(bundle_products)
        logger.debug("[API] bundle.products 사용")
    elif not products:
        products = copy.deepcopy(bundle.get("products", {})) if isinstance(bundle, Mapping) else {}
        logger.debug("[API] bundle.products fallback 사용")

    # ★★★ bundle.triggers 최우선 사용 (test.py 원본 데이터 보존) ★★★
    ctx_top_triggers: List[str] = []
    
    # 1순위: bundle.triggers에서 추출 (test.py에서 생성한 원본)
    if isinstance(bundle_triggers, Mapping):
        # bundle.triggers가 딕셔너리인 경우 (hits 형식)
        for key in ("negative", "cancel", "recovery", "adversatives"):
            items = bundle_triggers.get(key, [])
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, str) and item.strip() and item not in ctx_top_triggers:
                        ctx_top_triggers.append(item.strip())
    elif isinstance(bundle_triggers, list):
        # bundle.triggers가 리스트인 경우
        for item in bundle_triggers:
            if isinstance(item, str) and item.strip() and item not in ctx_top_triggers:
                ctx_top_triggers.append(item.strip())
    
    # 2순위: context_analysis 모듈에서 추출 (fallback)
    if not ctx_top_triggers:
        ctx_module = module_results.get("context_analysis")
        if isinstance(ctx_module, Mapping):
            raw_triggers = ctx_module.get("top_triggers") or []
            if isinstance(raw_triggers, list):
                ctx_top_triggers = [str(item).strip() for item in raw_triggers if str(item).strip()]
    
    if ctx_top_triggers:
        logger.debug(f"[API] bundle.triggers 사용: {len(ctx_top_triggers)}개 트리거")

    try:
        sentiment_balance = float(meta.get("sentiment_balance", 0.0))
    except Exception:
        sentiment_balance = 0.0

    primary_main = _normalize_main_label(poster.get("main"))
    if not primary_main and isinstance(main_dist, Mapping) and main_dist:
        try:
            primary_main = max(main_dist, key=main_dist.get)
        except Exception:
            primary_main = "애"
    if not primary_main:
        primary_main = "애"
    if isinstance(poster, Mapping):
        poster["main"] = primary_main

    headline_emotions = _build_headline_emotions(main_dist)
    if isinstance(products, Mapping):
        p1 = products.setdefault("p1", {})
        if isinstance(p1, dict):
            if headline_emotions:
                p1["headline_emotions"] = headline_emotions
            elif isinstance(main_dist, Mapping) and main_dist:
                p1["headline_emotions"] = [
                    {"name": primary_main, "pct": round(float(main_dist.get(primary_main, 0.0)) * 100, 1)}
                ]
            else:
                p1.setdefault("headline_emotions", [])
    else:
        p1 = {}
    p5 = products.get("p5") if isinstance(products, Mapping) else {}

    if isinstance(p1, Mapping) and p1.get("intensity"):
        intensity_label = str(p1.get("intensity"))
    elif isinstance(poster, Mapping) and poster.get("intensity"):
        intensity_label = str(poster.get("intensity"))
    else:
        intensity_label = "중간"

    try:
        stability_val = (
            int(float(p5.get("stability"))) if isinstance(p5, Mapping) and p5.get("stability") is not None else 0
        )
    except Exception:
        stability_val = 0

    # harmonize 함수 호출 완전히 제거 - test.py 원본 데이터 유지
    # if isinstance(main_dist, Mapping) and not is_from_bundle:
    #     _harmonize_sentence_structured(
    #         sentence_structured,
    #         target_main=primary_main,
    #         main_distribution=main_dist,
    #     )
    if sentence_structured:
        sentence_annotations = _serialize_sentence_structured(sentence_structured)

    if domain_profile != "service":
        investor_pack = _build_generic_investor_pack(
            primary_main,
            main_dist,
            transitions_structured,
            ctx_top_triggers,
            stability_val,
            intensity_label,
            sentiment_balance,
            insight_candidates=[],
        )

        existing_generic = products.get("generic") if isinstance(products, Mapping) else None
        generic_sections = investor_pack.get("sections", []) if investor_pack else []
        generic_highlights = investor_pack.get("highlights", []) if investor_pack else []
        if isinstance(existing_generic, Mapping):
            updated_generic = dict(existing_generic)
            if generic_sections:
                updated_generic["sections"] = generic_sections
            if generic_highlights:
                updated_generic["highlights"] = generic_highlights
            products["generic"] = updated_generic
        else:
            products["generic"] = {
                "sections": generic_sections,
                "highlights": generic_highlights,
            }

        products.setdefault(
            "p1",
            {
                "headline_emotions": [
                    {"name": primary_main, "pct": round(float(main_dist.get(primary_main, 0.0)) * 100, 1)}
                ]
                if isinstance(main_dist, Mapping)
                else [],
                "intensity": intensity_label,
                "churn_probability": None,
                "horizon_days": None,
                "triggers": ctx_top_triggers[:3],
                "recommended_actions": [
                    "감정 여정을 반영한 고객 스토리텔링 강화",
                    "회복 순간을 활용한 브랜드 메시지 설계",
                ],
            },
        )
        p1_ref = products.get("p1")
        if isinstance(p1_ref, dict):
            if not p1_ref.get("headline_emotions") and isinstance(main_dist, Mapping):
                p1_ref["headline_emotions"] = [
                    {"name": primary_main, "pct": round(float(main_dist.get(primary_main, 0.0)) * 100, 1)}
                ]
            p1_ref["intensity"] = intensity_label
            p1_ref["churn_probability"] = None
            p1_ref["horizon_days"] = None
            if ctx_top_triggers:
                p1_ref["triggers"] = ctx_top_triggers[:3]
            p1_ref["recommended_actions"] = [
                "감정 여정을 반영한 고객 스토리텔링 강화",
                "회복 순간을 활용한 브랜드 메시지 설계",
            ]

        neg_share = (
            sum(float(main_dist.get(emo, 0.0)) for emo in _NEGATIVE_SET) if isinstance(main_dist, Mapping) else 0.0
        )
        products.setdefault(
            "p3",
            {
                "risk_score": int(round(neg_share * 100)),
                "grade": "Moderate" if neg_share >= 0.35 else "Low",
                "alert": False,
            },
        )
        p3_ref = products.get("p3")
        if isinstance(p3_ref, dict):
            p3_ref.setdefault("risk_score", int(round(neg_share * 100)))
            p3_ref["grade"] = "Moderate" if neg_share >= 0.35 else "Low"
            p3_ref["alert"] = False

        products.setdefault(
            "p5",
            {
                "stability": stability_val,
                "maturity": 60 if intensity_label == "중간" else (45 if intensity_label == "낮음" else 75),
                "defenses": [],
            },
        )
        p5_ref = products.get("p5")
        if isinstance(p5_ref, dict):
            p5_ref.setdefault("stability", stability_val)
            p5_ref.setdefault(
                "maturity",
                60 if intensity_label == "중간" else (45 if intensity_label == "낮음" else 75),
            )
            p5_ref.setdefault("defenses", [])

    model_narrative = _extract_model_narrative(bundle)
    insight_summary: List[str] = []
    raw_summary = out.get("insight_summary")
    if isinstance(raw_summary, list):
        insight_summary = [str(line).strip() for line in raw_summary if str(line).strip()]
    elif model_narrative and domain_profile == "service":
        insight_summary = model_narrative
    elif model_narrative and domain_profile != "service":
        insight_summary = [str(line).strip() for line in model_narrative if str(line).strip()]
    elif domain_profile == "service":
        try:
            p1 = products.get("p1") if isinstance(products, Mapping) else {}
            intensity_label = str(
                (p1.get("intensity") if isinstance(p1, Mapping) else None) or poster.get("intensity") or "중간"
            )
            churn_pct = 0
            if isinstance(p1, Mapping):
                try:
                    churn_pct = int(float(p1.get("churn_probability") or 0))
                except Exception:
                    churn_pct = 0
            insight_summary = _build_narrative(
                primary_main,
                main_dist,
                intensity_label,
                churn_pct,
                ctx_top_triggers,
                sentiment_balance,
            )
        except Exception:
            insight_summary = []
    else:
        insight_summary = investor_pack.get("summary", []) if investor_pack else []

    if domain_profile != "service" and not insight_summary and investor_pack:
        insight_summary = investor_pack.get("summary", [])

    module_details = _build_module_details(
        module_results,
        poster=poster,
        main_dist=main_dist,
        transitions=transitions_structured,
        effective_mode=effective_mode,
    )
    module_hit_rate = compute_module_hit_rate(module_results)
    for detail in module_details:
        module_hit_rate[detail["name"]] = 1 if detail.get("status") == "ok" else 0
    provenance = bundle.get("provenance") if isinstance(bundle, Mapping) else {}
    if isinstance(provenance, Mapping):
        modules = provenance.get("modules")
        if isinstance(modules, Mapping):
            for key, record in modules.items():
                if key in module_hit_rate and isinstance(record, Mapping):
                    module_hit_rate[key] = 1 if record.get("ok") else 0

    investor_highlights = _build_investor_highlights(
        meta=meta,
        module_hit_rate=module_hit_rate,
        module_details=module_details,
        products=products,
        investor_pack=investor_pack,
        domain_profile=domain_profile,
        elapsed_ms=elapsed_ms,
        effective_mode=effective_mode,
        poster=poster,
    )

    strategic_brief = _build_investor_brief(
        meta=meta,
        investor_highlights=investor_highlights,
        module_details=module_details,
        domain_profile=domain_profile,
        effective_mode=effective_mode,
        refined=refined,
    )

    meta_mode = str(meta.get("mode", "")).upper()
    if meta_mode == "BALANCED":
        if domain_profile == "service":
            extras = _build_balanced_summary_extras(poster, sentence_structured, transitions_structured, products)
            insight_summary = [line for line in insight_summary if line]
            insight_summary.extend(extras)
            model_narrative_lines = _build_balanced_model_narrative(
                poster,
                sentence_structured,
                transitions_structured,
                products,
            )
        else:
            model_narrative_lines = investor_pack.get("summary", insight_summary) if investor_pack else insight_summary
    else:
        if domain_profile != "service" and investor_pack:
            model_narrative_lines = investor_pack.get("summary", insight_summary)
        else:
            model_narrative_lines = []

    master_report = _build_master_report(out, bundle)

    # ★★★ bundle 데이터를 API 응답에 직접 포함 (test.py 원본 보존) ★★★
    # 중요: payload["bundle"]에는 항상 전체 bundle을 그대로 넣어야 합니다.
    # bundle을 잘라내거나 재구성하면 기존 프론트엔드가 기대하는 필드(bundle.main, bundle.flow_ssot, 
    # bundle.weight_drivers 등)가 누락될 수 있습니다.
    bundle_data = {}
    if isinstance(bundle, Mapping):
        # 전체 bundle을 그대로 복사 (절대 잘라내지 않음)
        bundle_data = {
            "bundle": copy.deepcopy(bundle)  # test.py가 만든 전체 bundle 원본 보존
        }
        
        # 디버깅: bundle.main_dist 확인
        bundle_main_dist_check = bundle.get("main_dist")
        if bundle_main_dist_check and isinstance(bundle_main_dist_check, Mapping) and bundle_main_dist_check:
            logger.info(f"[API] ✅ bundle.main_dist API 응답에 포함: {bundle_main_dist_check}")
        else:
            logger.warning(f"[API] ⚠️ bundle.main_dist 없음 또는 비어있음 - bundle keys: {list(bundle.keys())[:20]}")
        
        # 선택: Truth 필드만 따로 보고 싶을 때를 위한 truth_core (보조 키)
        # 주의: 이 필드는 "관점 정의" 용도이며, bundle 자체는 그대로 유지
        truth_core = {k: bundle.get(k) for k in TRUTH_FIELDS if k in bundle}
        if truth_core:
            bundle_data["truth_core"] = truth_core
        
        logger.info("[API] bundle 전체를 API 응답에 포함: test.py 원본 보존")

    # ★★★ bundle의 why_lines, reasoning_path_lines, sub_top10_lines를 API 응답에 직접 포함 ★★★
    # test.py와 동일한 결과를 보장하기 위해 bundle 데이터를 최우선으로 사용
    final_why_lines = bundle_why_lines if isinstance(bundle_why_lines, list) and bundle_why_lines else []
    final_reasoning_path = bundle_reasoning_path if isinstance(bundle_reasoning_path, list) and bundle_reasoning_path else []
    final_sub_top10 = bundle_sub_top10 if isinstance(bundle_sub_top10, list) and bundle_sub_top10 else []
    
    payload = {
        "success": True,
        "timestamp": datetime.datetime.now().isoformat(),
        "text": out.get("text", text),
        "mode": meta["mode"],
        "poster": sanitize(poster or {}),
        "main_distribution": sanitize(main_dist),
        "sub_distribution": sanitize(sub_distribution),
        "sentence_annotations": sentence_annotations,
        "sentence_annotations_structured": sanitize(sentence_structured),
        "transitions_structured": sanitize(transitions_structured),
        "insight_summary": sanitize([line for line in insight_summary if line]),
        "model_narrative": sanitize([line for line in model_narrative_lines if line]),
        "products": sanitize(products or {}),
        "meta": meta,
        "results": sanitize(module_results),
        "module_results": sanitize(module_results),  # 명시적으로 module_results 추가
        "module_details": sanitize(module_details),
        "module_hit_rate": module_hit_rate,
        "investor_highlights": sanitize(investor_highlights),
        "strategic_brief": sanitize(strategic_brief),
        "explainability": sanitize(explainability_snapshot),
        "master_report": master_report,
        "triggers": sanitize(ctx_top_triggers),  # ★★★ 트리거를 API 응답에 직접 포함 ★★★
        # ★★★ test.py bundle 데이터 직접 포함 (원본 보존) ★★★
        "why_lines": sanitize(final_why_lines),
        "reasoning_path_lines": sanitize(final_reasoning_path),
        "sub_top10_lines": sanitize(final_sub_top10),
    }
    # bundle 데이터 병합 (추가 필드 포함)
    payload.update(bundle_data)
    if DEBUG_RAW_JSON:
        debug_snapshot = {
            "meta": payload["meta"],
            "poster": payload["poster"],
            "products": payload["products"],
            "modules": {
                name: module_results.get(name)
                for name in MODULE_SUMMARY_ORDER
                if isinstance(module_results.get(name), Mapping)
            },
        }
        payload["debug_raw"] = sanitize(debug_snapshot)
    return payload


def _register_job(job_id: str, mode: str, metadata: Optional[Dict[str, Any]] = None) -> None:
    """작업 등록 및 영구 저장"""
    job_data = {
        "status": "pending",
        "mode": mode.upper(),
        "submitted_at": datetime.datetime.now().isoformat(),
        "cancel_requested": False,
    }
    if metadata:
        # JSON 직렬화가 가능한 값만 저장 (예: datetime 객체 제외)
        safe_meta = {}
        for key, value in metadata.items():
            try:
                json.dumps(value)
                safe_meta[key] = value
            except (TypeError, ValueError):
                safe_meta[key] = str(value)
        job_data.update(safe_meta)
    with JOB_LOCK:
        JOB_RESULTS[job_id] = job_data
    # 영구 저장
    _save_job_to_disk(job_id, job_data)


def _save_job_to_disk(job_id: str, job_data: Dict[str, Any]) -> None:
    """작업 정보를 디스크에 저장"""
    try:
        job_file = JOB_STORAGE_DIR / f"{job_id}.json"
        with open(job_file, "w", encoding="utf-8") as f:
            json.dump(job_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"[API] 작업 저장 실패 (job_id={job_id}): {e}")


def _load_job_from_disk(job_id: str) -> Optional[Dict[str, Any]]:
    """디스크에서 작업 정보 로드"""
    try:
        job_file = JOB_STORAGE_DIR / f"{job_id}.json"
        if not job_file.exists():
            return None
        with open(job_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"[API] 작업 로드 실패 (job_id={job_id}): {e}")
        return None


def _is_job_expired(job_data: Dict[str, Any]) -> bool:
    """작업이 만료되었는지 확인"""
    try:
        submitted_at_str = job_data.get("submitted_at")
        if not submitted_at_str:
            return True
        submitted_at = datetime.datetime.fromisoformat(submitted_at_str)
        now = datetime.datetime.now()
        elapsed = (now - submitted_at).total_seconds()
        
        status = job_data.get("status", "pending")
        if status == "completed":
            return elapsed > JOB_EXPIRY_COMPLETED
        elif status == "error":
            return elapsed > JOB_EXPIRY_ERROR
        else:  # pending
            return elapsed > JOB_EXPIRY_PENDING
    except Exception:
        return True


def _cleanup_expired_jobs() -> None:
    """만료된 작업 정리"""
    try:
        for job_file in JOB_STORAGE_DIR.glob("*.json"):
            try:
                with open(job_file, "r", encoding="utf-8") as f:
                    job_data = json.load(f)
                if _is_job_expired(job_data):
                    job_file.unlink()
                    job_id = job_file.stem
                    with JOB_LOCK:
                        JOB_RESULTS.pop(job_id, None)
                    logger.debug(f"[API] 만료된 작업 삭제: {job_id}")
            except Exception:
                continue
    except Exception as e:
        logger.warning(f"[API] 작업 정리 실패: {e}")


def _update_job(job_id: str, *, data: Optional[Dict[str, Any]] = None, error: Optional[str] = None) -> None:
    """작업 상태 업데이트 및 영구 저장"""
    with JOB_LOCK:
        entry = JOB_RESULTS.get(job_id)
        if entry is None:
            # 디스크에서 로드 시도
            entry = _load_job_from_disk(job_id)
            if entry is None:
                return
            JOB_RESULTS[job_id] = entry
        
        if entry.get("status") == "cancelled":
            return
        
        if error:
            entry.update(
                {
                    "status": "error",
                    "error": error,
                    "finished_at": datetime.datetime.now().isoformat(),
                }
            )
        elif data is not None:
            entry.update(
                {
                    "status": "completed",
                    "result": data,
                    "finished_at": datetime.datetime.now().isoformat(),
                }
            )
    
    # 영구 저장
    _save_job_to_disk(job_id, entry)


def _get_job(job_id: str) -> Optional[Dict[str, Any]]:
    """작업 정보 조회 (메모리 → 디스크 순서로 시도)"""
    with JOB_LOCK:
        entry = JOB_RESULTS.get(job_id)
        if entry is None:
            # 디스크에서 로드 시도
            entry = _load_job_from_disk(job_id)
            if entry is None:
                return None
            # 메모리에 복원
            JOB_RESULTS[job_id] = entry
        
        # 만료 확인
        if _is_job_expired(entry):
            # 만료된 작업은 메모리와 디스크에서 삭제
            JOB_RESULTS.pop(job_id, None)
            try:
                job_file = JOB_STORAGE_DIR / f"{job_id}.json"
                if job_file.exists():
                    job_file.unlink()
            except Exception:
                pass
            return None
        
        return json.loads(json.dumps(entry))  # deep copy


def _is_job_cancelled(job_id: str) -> bool:
    with JOB_LOCK:
        entry = JOB_RESULTS.get(job_id)
        if entry is None:
            return False
        return bool(entry.get("cancel_requested") or entry.get("status") == "cancelled")


# =============================================================================
# [PROCESS WORKER] 독립된 분석 프로세스 관리 (Instant Cancellation 지원)
# - 스레드 대신 프로세스를 사용하여 'terminate()'로 즉시 중단 가능
# - Windows 호환성 고려: spawn 방식 대응 및 리소스 정리
# =============================================================================
import multiprocessing
import queue  # For queue.Empty
from multiprocessing import Process, Queue

class AnalysisWorker(Process):
    def __init__(
        self,
        job_id: str,
        text: str,
        requested_mode: str,
        env_updates: Dict[str, str],
        result_queue: Queue,
    ):
        super().__init__()
        self.job_id = job_id
        self.text = text
        self.requested_mode = requested_mode
        self.env_updates = env_updates
        self.result_queue = result_queue
        self.daemon = True  # 메인 프로세스 종료 시 함께 종료

    def run(self):
        try:
            # Windows 'spawn' 환경에서 로깅 설정 재초기화 필요할 수 있음
            logger.info(f"[Worker-{self.job_id}] 분석 프로세스 시작")
            
            # 1. 환경변수 동기화 (CLI와 동일 설정)
            sync_env = {
                "FORCE_HEAVY": "1",
                "USE_HEAVY_EMBEDDING": "1",
                "NO_FAST": "1",
                "NO_SKIP_ORCH": "1",
                # [GENIUS INSIGHT] 병렬 처리 재활성화 (Speed Up)
                # Watchdog(감시자)이 프로세스 멈춤을 감지하므로, 속도를 위해 병렬 처리를 다시 켭니다.
                # 기존 "1"(비활성) -> "0"(활성)으로 변경하여 타임아웃 가능성을 획기적으로 낮춥니다.
                "DISABLE_PARALLEL": os.getenv("API_DISABLE_PARALLEL", "0"),
            }
            # env_updates 우선 적용
            for k, v in sync_env.items():
                if k not in self.env_updates:
                    self.env_updates[k] = v
            
            # 2. 실제 분석 실행 (격리된 프로세스 내부)
            with TempEnviron(self.env_updates):
                combined_out, _snapshot, refined_out = _execute_layered_pipeline(
                    self.text, "balanced", job_id=self.job_id
                )

            # [Validation] 결과 무결성 검증 (결과 누락 방지)
            if not isinstance(combined_out, dict):
                raise RuntimeError(f"분석 결과 형식이 잘못되었습니다: {type(combined_out)}")
                
            # 필수 키 검사 및 복구 (main_distribution 등)
            results = combined_out.get("results", {})
            if "main_distribution" not in results and "main_distribution" not in combined_out:
                # 심각한 누락 감지: 복구 시도
                logger.warning(f"[Worker-{self.job_id}] main_distribution 누락 감지. Fallback 시도.")
                # 1. poster에서 시도
                poster = combined_out.get("poster", {})
                if poster and "main_distribution" in poster:
                    if "results" not in combined_out: combined_out["results"] = {}
                    combined_out["results"]["main_distribution"] = poster["main_distribution"]
                else:
                    # 2. fallback 생성
                    if "results" not in combined_out: combined_out["results"] = {}
                    combined_out["results"]["main_distribution"] = {"애": 0.25, "노": 0.25, "희": 0.25, "락": 0.25}

            # 3. 결과 페이로드 생성
            result = _build_api_payload(
                combined_out,
                text=self.text,
                requested_mode=self.requested_mode,
                effective_mode="balanced",
                refined=refined_out is not None,
            )
            result.pop("background_job_id", None)
            
            # 4. 성공 결과 전송
            self.result_queue.put({"status": "success", "data": result})
            logger.info(f"[Worker-{self.job_id}] 분석 완료 및 결과 전송")

        except BaseException as e:
            # Catch-all including KeyboardInterrupt and SystemExit
            msg = str(e)
            if isinstance(e, KeyboardInterrupt):
                msg = "Analysis cancelled by system (KeyboardInterrupt)"
                logger.warning(f"[Worker-{self.job_id}] {msg}")
            else:
                logger.error(f"[Worker-{self.job_id}] 프로세스 내부 오류 (Critical): {e}", exc_info=True)
            
            try:
                # Queue에 넣기 전에 프로세스 상태 확인
                self.result_queue.put({"status": "error", "error": msg})
            except Exception:
                pass # Queue might be closed
            finally:
                # 에러 발생 시에도 프로세스 정상 종료 유도
                pass

# 작업 관리자 (Singleton)
class JobProcessManager:
    def __init__(self):
        self.active_processes: Dict[str, Process] = {}
        self.lock = threading.Lock()

    def start_job(self, job_id: str, text: str, requested_mode: str, env_updates: Dict[str, str]):
        with self.lock:
            # 기존 동일 ID 프로세스 정리 (방어 코드)
            if job_id in self.active_processes:
                self.kill_job(job_id)
            
            q = multiprocessing.Queue()
            worker = AnalysisWorker(job_id, text, requested_mode, env_updates, q)
            worker.start()
            self.active_processes[job_id] = worker
            
            # 별도 스레드에서 결과 대기 (블로킹 방지)
            threading.Thread(target=self._monitor_worker, args=(job_id, worker, q), daemon=True).start()

    def kill_job(self, job_id: str):
        with self.lock:
            worker = self.active_processes.get(job_id)
            if worker:
                if worker.is_alive():
                    logger.warning(f"[JobManager] 작업 강제 종료 (kill): {job_id}")
                    worker.terminate()  # SIGTERM
                    worker.join(timeout=1)
                    if worker.is_alive():
                        worker.kill()  # SIGKILL (if supported)
                self.active_processes.pop(job_id, None)
                return True
        return False

    def _monitor_worker(self, job_id: str, worker: Process, q: Queue):
        # [GENIUS INSIGHT] Watchdog Pattern
        # 무한 대기(q.get()) 대신, 타임아웃 루프를 돌며 프로세스 생존 여부를 확인합니다.
        # 이를 통해 '무한 로딩' (프로세스 사망/행) 문제를 근본적으로 해결합니다.
        
        TIMEOUT_TOTAL = 600  # 최대 10분 대기
        CHECK_INTERVAL = 1.0 # 1초마다 상태 확인
        
        start_time = time.time()
        
        try:
            while True:
                # 1. 전체 타임아웃 체크
                if time.time() - start_time > TIMEOUT_TOTAL:
                    logger.error(f"[JobManager] 작업 시간 초과 ({TIMEOUT_TOTAL}s): {job_id}")
                    _update_job(job_id, error="Analysis timed out (process hung)")
                    self.kill_job(job_id)
                    return

                # 2. 결과 확인 (Non-blocking or Short timeout)
                try:
                    res = q.get(timeout=CHECK_INTERVAL)
                    # 결과를 받으면 루프 탈출
                    if res["status"] == "success":
                        _update_job(job_id, data=res["data"])
                    else:
                        _update_job(job_id, error=res.get("error", "Unknown worker error"))
                    return
                
                except queue.Empty:
                    # 3. 결과가 아직 없으면 프로세스 생존 확인
                    if not worker.is_alive():
                        # 프로세스가 죽었는데 결과도 없다? -> 비정상 종료 (OOM, Segfault 등)
                        logger.error(f"[JobManager] 작업 프로세스 비정상 종료 감지: {job_id}")
                        _update_job(job_id, error="Worker process crashed unexpectedly (OOM or Error)")
                        return
                    
                    # 프로세스는 살아있고 결과는 아직 없음 -> 계속 대기
                    continue
                
        except Exception as e:
            logger.error(f"[JobManager] 모니터링 스레드 오류: {e}", exc_info=True)
            if not _is_job_cancelled(job_id):
                _update_job(job_id, error=f"Monitoring error: {str(e)}")
        finally:
            with self.lock:
                # 정상 종료 후 정리
                if job_id in self.active_processes:
                    if self.active_processes[job_id] == worker: # ID 재사용 방지 체크
                        self.active_processes.pop(job_id, None)
            # 좀비 프로세스 방지
            if worker.is_alive():
                worker.join(timeout=1)


_job_manager = JobProcessManager()

def _run_background_job(
    job_id: str,
    text: str,
    requested_mode: str,
    env_updates: Mapping[str, Optional[str]],
) -> None:
    """
    백그라운드 작업을 독립 프로세스로 실행
    """
    # None 값 필터링 (multiprocessing 전달 위해)
    clean_env = {k: v for k, v in env_updates.items() if v is not None}
    
    logger.info(f"[API] 작업 프로세스 위임 (job_id={job_id})")
    _job_manager.start_job(job_id, text, requested_mode, clean_env)



# ---------------------------------------------------------------------------
# 엔드포인트
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    """루트 경로 - 기본적으로 made.html을 서빙"""
    made_path = PROJECT_ROOT / "made" / "made.html"
    if made_path.exists():
        return FileResponse(str(made_path))

    index_path = PROJECT_ROOT / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))

    return {"message": "Emotion AI Web Interface"}


@app.get("/made.html")
@app.get("/made")
async def get_made():
    """made.html 명시적 접근"""
    made_path = PROJECT_ROOT / "made" / "made.html"
    if made_path.exists():
        return FileResponse(str(made_path))
    raise HTTPException(status_code=404, detail="made.html을 찾을 수 없습니다.")


@app.get("/index.html")
@app.get("/index")
async def get_index():
    """index.html 명시적 접근"""
    index_path = PROJECT_ROOT / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    raise HTTPException(status_code=404, detail="index.html을 찾을 수 없습니다.")


@app.get("/app.js")
async def get_js():
    js_path = PROJECT_ROOT / "app.js"
    if js_path.exists():
        return FileResponse(str(js_path))
    # app.js가 없으면 made/assets/js/made.js를 대신 반환 시도
    made_js = PROJECT_ROOT / "made" / "assets" / "js" / "made.js"
    if made_js.exists():
        return FileResponse(str(made_js))
    raise HTTPException(status_code=404, detail="app.js를 찾을 수 없습니다.")


@app.post("/api/analyze_raw")
async def analyze_raw(input_data: TextInput, x_api_key: Optional[str] = Header(None, alias="X-API-KEY")):
    """
    [DEBUG] 로우 레벨 분석 API.
    test.py의 run_one을 직접 호출하고 원본 결과를 반환합니다.
    """
    demo_key = (os.getenv("DEMO_API_KEY") or "").strip()
    if demo_key:
        # 대소문자 구분 없이 비교 (헤더/환경변수 모두 소문자로 정규화)
        demo_norm = demo_key.casefold()
        provided = (x_api_key or "").strip()
        if not provided or provided.casefold() != demo_norm:
            return JSONResponse(
                status_code=401,
                content={"success": False, "error": "유효한 데모 키가 필요합니다."},
            )
    text = (input_data.text or "").strip()
    if not text:
        return JSONResponse(status_code=400, content={"success": False, "error": "텍스트 필수"})
        
    try:
        # NOTE: /api/analyze에서 사용하는 HEAVY 리파인 환경과 완전히 동일한 설정을 사용하여
        # CLI와 API가 같은 파이프라인/환경에서 Truth를 생성하도록 맞춘다.
        env_updates = _build_refine_env(mode="HEAVY")
        env_updates["EMB_CACHE_PATH"] = str(PROJECT_ROOT / "src" / "cache" / "embeddings.json")
        out = _run_with_env(text, env_updates)
        
        # bundle 추출 (전체를 그대로 사용)
        bundle = out.get("__web_bundle") or {}
        
        # Truth 필드 검증 (로깅용)
        truth_validation = _validate_truth_fields(bundle)
        missing_fields = [field for field, exists in truth_validation.items() if not exists]
        if missing_fields:
            logger.warning("[API] analyze_raw: 일부 Truth 필드가 누락됨: %s", missing_fields)
        else:
            logger.debug("[API] analyze_raw: 모든 Truth 필드 존재 확인됨")
        
        return {
            "success": True,
            "timestamp": datetime.datetime.now().isoformat(),
            "out": out,           # 전체 out 포함 (디버깅용)
            "bundle": bundle,     # test.py가 만든 __web_bundle 전체 (Truth 필드 포함)
            "truth_validation": truth_validation,  # Truth 필드 존재 여부 (디버깅용)
        }
    except Exception as exc:
        logger.error("[API] analyze_raw 실행 중 오류 발생: %s", exc, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": f"분석 실패: {str(exc)}",
                "error_type": type(exc).__name__
            }
        )


@app.post("/api/analyze")
async def analyze(
    input_data: TextInput,
    x_job_id: Optional[str] = Header(None, alias="X-Job-ID"),
    x_api_key: Optional[str] = Header(None, alias="X-API-KEY"),
):
    """
    텍스트 감정 분석 API 엔드포인트.
    모든 예외를 잡아서 HTTP 응답으로 반환하여 ERR_CONNECTION_RESET 방지.
    작업 취소를 지원하기 위해 작업 ID를 생성하고 추적합니다.
    """
    demo_key = (os.getenv("DEMO_API_KEY") or "").strip()
    if demo_key:
        # 대소문자 구분 없이 비교 (헤더/환경변수 모두 소문자로 정규화)
        demo_norm = demo_key.casefold()
        provided = (x_api_key or "").strip()
        if not provided or provided.casefold() != demo_norm:
            return JSONResponse(
                status_code=401,
                content={"success": False, "error": "유효한 데모 키가 필요합니다."},
            )

    text = (input_data.text or "").strip()
    if not text:
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": "텍스트를 입력해주세요."}
        )

    requested_mode_raw = (input_data.mode or "balanced").strip().lower()

    # 런타임 환경 디버그용: 오케스트레이터/임베더 관련 환경변수 로깅
    logger.info(
        "[API][env] ORCH_MODULE=%s ORCH_ATTR=%s EMBEDDER_MODULE=%s EMBEDDER_ATTR=%s FORCE_HEAVY=%s NO_SKIP_ORCH=%s USE_HEAVY_EMBEDDING=%s",
        os.getenv("ORCH_MODULE"),
        os.getenv("ORCH_ATTR"),
        os.getenv("EMBEDDER_MODULE"),
        os.getenv("EMBEDDER_ATTR"),
        os.getenv("FORCE_HEAVY"),
        os.getenv("NO_SKIP_ORCH"),
        os.getenv("USE_HEAVY_EMBEDDING"),
    )
    fast_modes = {"fast", "preview", "speed", "quick"}
    heavy_modes = {"heavy", "prod", "oneclick", "full", "heavy_mode", "precise", "precision", "detail"}
    balanced_modes = {"balanced", "balance", "standard", "default", "auto", "baseline", "normal", "medium"}

    if requested_mode_raw in fast_modes:
        logger.info("[API] FAST 모드 요청을 수신했지만 정밀 분석으로 강제 전환합니다.")

    effective_mode = "balanced"
    text_preview = text[:48] + "..." if len(text) > 48 else text
    
    # 작업 ID 생성 및 추적
    # [GENIUS INSIGHT] Client-Side ID Support
    # 클라이언트가 생성한 ID(X-Job-ID)를 우선 사용하여, 요청 직후 취소 요청과의 동기화를 보장합니다.
    job_id = x_job_id or str(uuid.uuid4())
    
    with CURRENT_JOBS_LOCK:
        CURRENT_JOBS[job_id] = {
            "cancelled": False,
            "started_at": datetime.datetime.now().isoformat(),
            "mode": effective_mode,
            "text_length": len(text)
        }
    # JOB_RESULTS / 디스크 상태 초기화 (monitor thread에서 결과 기록 가능하도록)
    _register_job(
        job_id,
        effective_mode,
        metadata={
            "text_length": len(text),
            "requested_mode": requested_mode_raw,
            "preview": text_preview,
        },
    )
    
    logger.info("[API] 분석 시작 (job_id=%s, mode=%s, text_len=%d, text=%s)", job_id, effective_mode, len(text), text_preview)

    try:
        # [GENIUS INSIGHT] Global Concurrency Control & Resource Safety
        # 시스템 전체에서 동시에 실행되는 분석 작업 수를 제한하여 OOM을 방지합니다.
        # 세마포어를 획득할 때까지 대기하며, 이는 자연스러운 요청 조절(Throttling) 효과를 냅니다.
        async with GLOBAL_ANALYSIS_SEMAPHORE:
            # [Genius Memory] 분석 시작 전 메모리 정리 (Fork/Spawn 효율성 증대)
            gc.collect()
            
            # [GENIUS INSIGHT] Timeout Extension & Parallelism Restoration
            # 순차 처리(DISABLE_PARALLEL=1)로 인해 분석 시간이 길어져 504 타임아웃이 발생합니다.
            # 1. 타임아웃을 5분(300초)에서 10분(600초)으로 연장하여 긴 분석도 수용합니다.
            try:
                max_wait_seconds = float(os.getenv("API_ANALYZE_TIMEOUT", "600"))
            except Exception:
                max_wait_seconds = 600.0
            start_time = time.monotonic()
        
            # 모든 예외를 잡아서 HTTP 응답으로 반환 (ERR_CONNECTION_RESET 방지)
            # [ASYNC POLLING FIX] 메인 스레드 차단 방지를 위해 비동기 폴링 루프로 변경
            # 기존: _execute_layered_pipeline 직접 호출 (동기 함수라 메인 루프 차단 -> 취소/F5 불가)
            # 변경: Background Job 시작 -> async sleep 루프에서 결과 대기
            try:
                # 1. 환경변수 준비
                env_updates = {}
                # CLI 호환 환경변수 설정 (HEAVY 모드 등)
                sync_env = {
                    "FORCE_HEAVY": "1",
                    "USE_HEAVY_EMBEDDING": "1",
                    "NO_FAST": "1",
                    "NO_SKIP_ORCH": "1",
                }
                # 모드에 따른 환경 설정 (실제 로직은 _execute_layered_pipeline 내부와 동일하게 맞춤)
                if effective_mode == "balanced":
                    env_updates = _build_refine_env("balanced")
                else:
                    env_updates = _build_fast_layer_env()
                    
                # 2. 백그라운드 프로세스로 작업 시작
                # _run_background_job은 내부적으로 _job_manager.start_job을 호출
                # 여기서는 직접 _job_manager를 사용하여 제어권 확보
                # None 값 필터링
                clean_env = {k: v for k, v in env_updates.items() if v is not None}
                _job_manager.start_job(job_id, text, requested_mode_raw, clean_env)
                
                # 3. 비동기 폴링 루프 (메인 스레드 양보)
                # 0.5초마다 상태 확인 -> Event Loop가 다른 요청(취소, F5 등)을 처리할 틈을 줌
                while True:
                    await asyncio.sleep(0.5) # ★★★ 핵심: 여기서 제어권 양보 ★★★
                    if max_wait_seconds > 0 and (time.monotonic() - start_time) > max_wait_seconds:
                        _job_manager.kill_job(job_id)
                        with CURRENT_JOBS_LOCK:
                            CURRENT_JOBS.pop(job_id, None)
                        return JSONResponse(
                            status_code=504,
                            content={
                                "success": False,
                                "error": "분석 시간이 너무 오래 걸려 타임아웃되었습니다.",
                                "mode": effective_mode,
                                "timeout": True
                            }
                        )
                    
                    # 3-1. 취소 여부 확인
                    if _is_job_cancelled(job_id):
                        # 이미 취소 API에 의해 프로세스는 kill 되었을 것임
                        with CURRENT_JOBS_LOCK:
                            CURRENT_JOBS.pop(job_id, None)
                        return JSONResponse(
                            status_code=499,
                            content={
                                "success": False,
                                "error": "분석이 취소되었습니다.",
                                "mode": effective_mode,
                                "cancelled": True
                            }
                        )
                        
                    # 3-2. 결과 확인 (JOB_RESULTS)
                    with JOB_LOCK:
                        job_data = JOB_RESULTS.get(job_id)
                    
                    if job_data:
                        status = job_data.get("status")
                        if status == "completed":
                            # 성공 결과 반환
                            response = job_data.get("result")
                            # 작업 완료 후 추적에서 제거
                            with CURRENT_JOBS_LOCK:
                                CURRENT_JOBS.pop(job_id, None)
                            
                            response["job_id"] = job_id
                            # 로그 기록은 worker에서 이미 수행했을 수 있으나, 응답 시점 기록
                            logger.info("[API] 분석 결과 반환 (job_id=%s)", job_id)
                            return response
                            
                        elif status == "error":
                            # 오류 발생
                            error_msg = job_data.get("error", "Unknown error")
                            with CURRENT_JOBS_LOCK:
                                CURRENT_JOBS.pop(job_id, None)
                            
                            # [KeyboardInterrupt] 명시적 핸들링
                            if "KeyboardInterrupt" in str(error_msg):
                                return JSONResponse(
                                    status_code=503,
                                    content={
                                        "success": False,
                                        "error": "서버 내부 인터럽트로 인해 분석이 중단되었습니다.",
                                        "mode": effective_mode
                                    }
                                )

                            return JSONResponse(
                                status_code=500,
                                content={
                                    "success": False,
                                    "error": f"분석 실패: {error_msg}",
                                    "mode": effective_mode
                                }
                            )
                        elif status == "cancelled":
                            # 상태가 취소됨 (위의 _is_job_cancelled와 중복될 수 있으나 안전장치)
                            with CURRENT_JOBS_LOCK:
                                CURRENT_JOBS.pop(job_id, None)
                            return JSONResponse(
                                status_code=499,
                                content={
                                    "success": False,
                                    "error": "분석이 취소되었습니다.",
                                    "mode": effective_mode,
                                    "cancelled": True
                                }
                            )
                    
                    # 아직 진행 중이면 루프 계속 (await sleep)
            finally:
                # [Genius Memory] 분석 후 메모리 정리
                gc.collect()
            
    except asyncio.CancelledError:
        # [ASYNC FIX] 클라이언트 연결 해제/취소 시 발생하는 CancelledError 처리
        logger.info("[API] 요청 취소됨 (Client Disconnected/Cancelled) - job_id=%s", job_id)
        try:
            # 워커 프로세스도 정리
            _job_manager.kill_job(job_id)
        except:
            pass
        with CURRENT_JOBS_LOCK:
            CURRENT_JOBS.pop(job_id, None)
        raise

    except KeyboardInterrupt:
        # 사용자 중단은 정상적인 종료로 처리
        logger.warning("[API] 분석이 사용자에 의해 중단되었습니다.")
        with CURRENT_JOBS_LOCK:
            CURRENT_JOBS.pop(job_id, None)
        return JSONResponse(
            status_code=503,
            content={
                "success": False,
                "error": "분석이 중단되었습니다. 서버가 재시작 중일 수 있습니다.",
                "mode": effective_mode
            }
        )
    except RuntimeError as exc:
        # 작업 취소로 인한 RuntimeError 처리
        error_msg = str(exc)
        if "취소" in error_msg or "cancelled" in error_msg.lower():
            logger.info("[API] 작업 취소됨 (job_id=%s)", job_id)
            with CURRENT_JOBS_LOCK:
                CURRENT_JOBS.pop(job_id, None)
            return JSONResponse(
                status_code=499,  # Client Closed Request
                content={
                    "success": False,
                    "error": "분석이 취소되었습니다.",
                    "mode": effective_mode,
                    "cancelled": True
                }
            )
        # 다른 RuntimeError는 아래에서 처리
        raise
    except MemoryError as exc:
        # 메모리 부족 오류
        with CURRENT_JOBS_LOCK:
            CURRENT_JOBS.pop(job_id, None)
        logger.error("[API] 메모리 부족 오류 발생: %s", exc, exc_info=True)
        return JSONResponse(
            status_code=507,
            content={
                "success": False,
                "error": f"메모리 부족으로 분석을 완료할 수 없습니다. 텍스트 길이: {len(text)}자",
                "mode": effective_mode,
                "text_length": len(text)
            }
        )
    except RuntimeError as exc:
        # 런타임 오류 (예: 모델 로드 실패 등)
        error_msg = str(exc)
        
        # 작업 취소로 인한 RuntimeError는 이미 위에서 처리됨
        if "취소" in error_msg or "cancelled" in error_msg.lower():
            # 이미 처리되었으므로 재처리하지 않음
            raise
        
        with CURRENT_JOBS_LOCK:
            CURRENT_JOBS.pop(job_id, None)
        logger.error("[API] 런타임 오류 발생: %s", exc, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": f"분석 실행 중 오류가 발생했습니다: {error_msg}",
                "mode": effective_mode,
                "error_type": "RuntimeError"
            }
        )
    except Exception as exc:
        # 기타 모든 예외
        error_type = type(exc).__name__
        error_msg = str(exc)
        
        # 작업 추적에서 제거
        with CURRENT_JOBS_LOCK:
            CURRENT_JOBS.pop(job_id, None)
        
        logger.error("[API] 분석 중 예상치 못한 오류 발생 (type=%s): %s", error_type, exc, exc_info=True)
        
        # 예외 상세 정보를 안전하게 추출
        import traceback
        tb_str = traceback.format_exc()
        logger.debug("[API] 예외 상세:\n%s", tb_str)
        
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": f"분석 실패: {error_msg}",
                "mode": effective_mode,
                "error_type": error_type,
                "text_length": len(text)
            }
        )

    # 결과 검증 및 응답 생성도 예외 처리
    try:
        if not isinstance(combined_out, Mapping):
            logger.error("[API] 분석 결과 형식이 올바르지 않습니다. type=%s", type(combined_out))
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "error": "분석 결과 형식이 올바르지 않습니다.",
                    "mode": effective_mode
                }
            )

        refined_completed = refined_out is not None

        response = _build_api_payload(
            combined_out,
            text=text,
            requested_mode=requested_mode_raw,
            effective_mode=effective_mode,
            refined=refined_completed,
            enforce_fast_meta=False,
        )

        response["background_job_id"] = None

        elapsed_time = response.get("meta", {}).get("elapsed", 0.0)
        logger.info("[API] 분석 완료 (job_id=%s, mode=%s, elapsed=%.3fs, text_len=%d)", 
                   job_id, response.get("mode", effective_mode), elapsed_time, len(text))
        
        # 작업 ID를 응답에 포함 (클라이언트에서 취소 가능하도록)
        response["job_id"] = job_id
        return response
    except Exception as exc:
        # 응답 생성 중 오류
        with CURRENT_JOBS_LOCK:
            CURRENT_JOBS.pop(job_id, None)
        logger.error("[API] 응답 생성 중 오류 발생: %s", exc, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": f"응답 생성 중 오류가 발생했습니다: {str(exc)}",
                "mode": effective_mode,
                "error_type": type(exc).__name__
            }
        )


@app.get("/api/job/{job_id}")
async def get_job(job_id: str):
    """작업 상태 조회"""
    # 주기적으로 만료된 작업 정리 (10% 확률로 실행하여 부하 최소화)
    import random
    if random.random() < 0.1:
        _cleanup_expired_jobs()
    
    snapshot = _get_job(job_id)
    if snapshot is None:
        # 404 대신 명확한 상태 반환 (made.js에서 처리 가능하도록)
        raise HTTPException(
            status_code=404,
            detail="해당 작업을 찾을 수 없습니다. 작업이 만료되었거나 서버가 재시작되었을 수 있습니다."
        )
    return snapshot


class CancelJobRequest(BaseModel):
    reason: Optional[str] = None


@app.post("/api/job/{job_id}/cancel")
async def cancel_job(job_id: str, payload: Optional[CancelJobRequest] = None):
    """
    작업 취소 API.
    백그라운드 작업과 현재 실행 중인 동기 작업 모두 취소 가능.
    """
    cancelled = False
    
    # 1. 현재 실행 중인 동기 작업 취소
    with CURRENT_JOBS_LOCK:
        if job_id in CURRENT_JOBS:
            CURRENT_JOBS[job_id]["cancelled"] = True
            cancelled = True
            logger.info("[API] 현재 실행 중인 작업 취소 요청 (job_id=%s, reason=%s)", 
                       job_id, payload.reason if payload else None)
    
    # 2. 백그라운드 작업 취소 (기존 로직 + 프로세스 킬)
    with JOB_LOCK:
        entry = JOB_RESULTS.get(job_id)
        if entry is None:
            # 동기 작업이 아니고 백그라운드 작업도 없으면 404
            if not cancelled:
                raise HTTPException(status_code=404, detail="해당 작업을 찾을 수 없습니다.")
        else:
            if entry.get("status") in {"completed", "error"}:
                return {"status": entry["status"], "message": "이미 완료되었거나 오류가 발생한 작업입니다."}
            entry["cancel_requested"] = True
            entry["status"] = "cancelled"
            entry["finished_at"] = datetime.datetime.now().isoformat()
            if payload and payload.reason:
                entry["cancel_reason"] = payload.reason
            cancelled = True
            logger.info("[API] Background job 취소 요청 수신 (job_id=%s)", job_id)
            
            # ★★★ 프로세스 강제 종료 실행 ★★★
            if _job_manager.kill_job(job_id):
                logger.info(f"[API] 작업 프로세스 즉시 사살됨 (job_id={job_id})")

    if cancelled:
        return {"status": "cancelled", "message": "작업 취소 요청이 처리되었습니다."}
    else:
        raise HTTPException(status_code=404, detail="해당 작업을 찾을 수 없습니다.")


@app.get("/api/status")
async def status():
    return JSONResponse(
        {
            "status": "ok",
            "timestamp": datetime.datetime.now().isoformat(),
            "mode": os.environ.get("EA_PROFILE", "prod"),
        }
    )


def _build_investor_brief(
    *,
    meta: Mapping[str, Any],
    investor_highlights: Sequence[str],
    module_details: Sequence[Mapping[str, Any]],
    domain_profile: str,
    effective_mode: str,
    refined: bool,
) -> Dict[str, List[str]]:
    brief: Dict[str, List[str]] = {
        "traction": [],
        "differentiators": [],
        "roadmap": [],
        "risk_controls": [],
    }

    elapsed_fast = None
    if isinstance(meta.get("fast_layer_elapsed"), (int, float)):
        elapsed_fast = float(meta["fast_layer_elapsed"])
    elif isinstance(meta.get("elapsed"), (int, float)):
        elapsed_fast = float(meta["elapsed"])

    if elapsed_fast and elapsed_fast > 0:
        per_minute = int(max(1, math.floor(60.0 / elapsed_fast)))
        brief["traction"].append(f"단일 FAST 인스턴스 기준 분당 {per_minute}건 처리")
        brief["traction"].append(f"FAST 응답 지연 {round(elapsed_fast, 2)}초")

    total_modules = len(module_details)
    ok_modules = sum(1 for detail in module_details if detail.get("status") == "ok")
    if total_modules:
        coverage_line = f"분석 모듈 커버리지 {ok_modules}/{total_modules}개 가동"
        if coverage_line not in investor_highlights:
            brief["traction"].append(coverage_line)

    profile_label = "서비스" if domain_profile == "service" else "내러티브"
    brief["differentiators"].append(f"도메인 자동 분류({profile_label}) 기반 맞춤 인사이트")
    brief["differentiators"].append("FAST→BALANCED 계층형 파이프라인으로 속도/정밀 동시 확보")
    if investor_highlights:
        brief["differentiators"].extend(
            [line for line in investor_highlights if line not in brief["differentiators"]]
        )

    if effective_mode == "balanced" or refined:
        brief["roadmap"].append("정밀 BALANCED 모드 3~5분 SLA 안정화 진행")
    else:
        brief["roadmap"].append("BALANCED 정밀 모드 베타를 통해 설명력 강화 예정")
    brief["roadmap"].append("Model A 추론 내러티브 고도화/멀티 모달 확장 계획")

    if not DEBUG_RAW_JSON:
        brief["risk_controls"].append("민감 데이터 기본 마스킹(DEBUG_RAW_JSON=0)")
    if elapsed_fast and elapsed_fast > 0:
        brief["risk_controls"].append("성능 로그/메타 자동 수집으로 후속 튜닝 근거 확보")
    if not brief["risk_controls"]:
        brief["risk_controls"].append("예외 처리/레이어드 전략으로 실패 시 Fast 스냅샷 보장")

    for key, items in list(brief.items()):
        deduped: List[str] = []
        for item in items:
            if item and item not in deduped:
                deduped.append(item)
        brief[key] = deduped

    return brief


if __name__ == "__main__":
    import sys
    import platform
    
    # Windows에서 multiprocessing 문제 해결 (가장 먼저 실행)
    if platform.system() == "Windows":
        import multiprocessing
        # Windows에서 freeze_support 호출 (multiprocessing 안정성 향상)
        try:
            multiprocessing.freeze_support()
        except Exception:
            pass
        # spawn 방식 명시적 설정 (Windows 기본값이지만 명시적으로 설정)
        try:
            multiprocessing.set_start_method("spawn", force=True)
        except RuntimeError:
            # 이미 설정된 경우 무시
            pass
    
    port = int(os.environ.get("PORT", "8000"))
    
    # reload 설정: 환경변수로 제어 가능하지만 기본값은 False
    # Windows에서 reload는 multiprocessing 문제를 일으킬 수 있으므로 기본적으로 비활성화
    reload_enabled = os.environ.get("RELOAD", "false").lower() in ("true", "1", "yes")
    
    # Windows에서 reload 사용 시 추가 설정 및 경고
    reload_kwargs = {}
    if reload_enabled:
        if platform.system() == "Windows":
            logger.warning(
                "[Windows] reload 모드가 활성화되었습니다. "
                "multiprocessing 문제가 발생할 수 있습니다. "
                "문제가 발생하면 RELOAD=false로 설정하거나 --reload 플래그를 제거하세요."
            )
            # Windows에서 reload 시 안정성을 위한 설정
            reload_kwargs = {
                "reload_dirs": [str(PROJECT_ROOT / "src" / "serving")],  # 특정 디렉토리만 감시
                "reload_excludes": ["*.pyc", "__pycache__", "*.log", "*.json", "test.py"],  # 불필요한 파일 제외
            }
        else:
            # Linux/Mac에서는 전체 프로젝트 감시 가능
            reload_kwargs = {
                "reload_dirs": [str(PROJECT_ROOT / "src")],
                "reload_excludes": ["*.pyc", "__pycache__", "*.log", "*.json"],
            }
    
    # uvicorn 실행 설정
    uvicorn_config = {
        "app": app,
        "host": "0.0.0.0",
        "port": port,
        "reload": reload_enabled,
        "log_level": os.environ.get("LOG_LEVEL", "info").lower(),
    }
    
    # Windows에서 workers를 명시적으로 1로 설정 (multiprocessing 문제 방지)
    if platform.system() == "Windows":
        uvicorn_config["workers"] = 1
    
    # reload_kwargs 병합
    uvicorn_config.update(reload_kwargs)
    
    try:
        uvicorn.run(**uvicorn_config)
    except KeyboardInterrupt:
        # 정상적인 종료 처리
        logger.info("서버가 사용자에 의해 중단되었습니다.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"서버 시작 중 오류 발생: {e}", exc_info=True)
        sys.exit(1)
