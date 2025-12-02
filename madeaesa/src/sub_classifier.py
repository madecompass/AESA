# src/sub_classifier.py
# -*- coding: utf-8 -*-

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

"""
Sub-emotion trainer (v2, minimal & extensible)
- 입력: 임베딩 JSONL(run.jsonl) + 정답지(labels_programmatic.jsonl)
- 대상: 메인 감정(희/노/애/락)별 서브 감정(단일 라벨) 분류기
- 특징: EMOTIONS.JSON에서 서브 목록 동적 로드, confidence 가중, fallback 제외 옵션, 불균형 가중
- 확장 포인트: 11개 분석 모듈 특성/컨텍스트, 증강, k-fold, MLP/Transformer Head 교체
"""

# 표준 라이브러리
import os
import json
import re
import time
import random
import math
import glob
import shutil
import unicodedata
import traceback
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any
from collections import Counter
from copy import deepcopy

# 써드파티 라이브러리
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Sampler
from sklearn.model_selection import StratifiedKFold

def _now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")

# ============================
# 개선된 로깅 시스템
# ============================
def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """구조화된 로깅 시스템 설정"""
    logger = logging.getLogger("sub_classifier")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    # 기존 핸들러 제거
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 포맷터 설정
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 파일 핸들러 (선택적)
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def get_system_info() -> Dict[str, Any]:
    """시스템 정보 수집"""
    info = {
        "cpu_count": os.cpu_count(),
        "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    
    if torch.cuda.is_available():
        info.update({
            "cuda_version": torch.version.cuda,
            "gpu_count": torch.cuda.device_count(),
            "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "gpu_memory_gb": [round(torch.cuda.get_device_properties(i).total_memory / (1024**3), 2) 
                             for i in range(torch.cuda.device_count())]
        })
    
    return info

def log_performance_metrics(logger: logging.Logger, metrics: Dict[str, Any]) -> None:
    """성능 메트릭 로깅"""
    logger.info("=== Performance Metrics ===")
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            logger.info(f"{key}: {value}")
        else:
            logger.info(f"{key}: {value}")
    logger.info("=" * 30)

# ============================
# 개선된 에러 처리 시스템
# ============================
class SubClassifierError(Exception):
    """서브학습기 기본 예외 클래스"""
    pass

class DataLoadError(SubClassifierError):
    """데이터 로딩 관련 예외"""
    pass

class ModelTrainingError(SubClassifierError):
    """모델 학습 관련 예외"""
    pass

class ValidationError(SubClassifierError):
    """검증 관련 예외"""
    pass

def handle_error(logger: logging.Logger, error: Exception, context: str = "", 
                 recoverable: bool = True) -> bool:
    """통합 에러 처리 함수"""
    error_type = type(error).__name__
    error_msg = str(error)
    
    logger.error(f"[{context}] {error_type}: {error_msg}")
    
    if recoverable:
        logger.info(f"[{context}] Attempting recovery...")
        return True
    else:
        logger.critical(f"[{context}] Non-recoverable error. Stopping execution.")
        return False

def validate_data_quality(X: np.ndarray, y: np.ndarray, w: np.ndarray, 
                         logger: logging.Logger) -> Dict[str, Any]:
    """데이터 품질 검증"""
    quality_report = {
        "total_samples": len(X),
        "feature_dim": X.shape[1] if X.ndim > 1 else 1,
        "num_classes": len(np.unique(y)),
        "class_distribution": dict(Counter(y)),
        "issues": [],
        "warnings": []
    }
    
    # NaN/Inf 검사
    if np.isnan(X).any():
        quality_report["issues"].append("NaN values found in features")
    if np.isinf(X).any():
        quality_report["issues"].append("Inf values found in features")
    
    # 클래스 불균형 검사
    class_counts = Counter(y)
    min_count = min(class_counts.values())
    max_count = max(class_counts.values())
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
    
    if imbalance_ratio > 10:
        quality_report["warnings"].append(f"High class imbalance ratio: {imbalance_ratio:.2f}")
    
    # 가중치 검사
    if w is not None:
        if np.any(w <= 0):
            quality_report["issues"].append("Non-positive weights found")
        if np.any(np.isnan(w)):
            quality_report["issues"].append("NaN weights found")
    
    # 품질 점수 계산
    quality_score = 1.0
    quality_score -= len(quality_report["issues"]) * 0.3
    quality_score -= len(quality_report["warnings"]) * 0.1
    quality_report["quality_score"] = max(0.0, quality_score)
    
    logger.info(f"Data quality score: {quality_score:.3f}")
    if quality_report["issues"]:
        logger.warning(f"Data quality issues: {quality_report['issues']}")
    if quality_report["warnings"]:
        logger.warning(f"Data quality warnings: {quality_report['warnings']}")
    
    return quality_report

# ============================
# 성능 최적화 시스템
# ============================
def optimize_batch_size(model: nn.Module, device: torch.device, 
                       input_shape: Tuple[int, ...], 
                       max_batch_size: int = 1024) -> int:
    """자동 배치 크기 최적화"""
    model.eval()
    optimal_batch_size = 1
    
    for batch_size in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
        if batch_size > max_batch_size:
            break
            
        try:
            # 테스트 입력 생성
            test_input = torch.randn(batch_size, *input_shape).to(device)
            
            # 메모리 사용량 측정
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            
            with torch.no_grad():
                _ = model(test_input)
            
            if device.type == 'cuda':
                memory_used = torch.cuda.max_memory_allocated() / (1024**2)  # MB
                if memory_used > 1000:  # 1GB 제한
                    break
            
            optimal_batch_size = batch_size
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                break
            else:
                raise e
    
    return optimal_batch_size

def monitor_memory_usage(logger: logging.Logger) -> Dict[str, float]:
    """메모리 사용량 모니터링"""
    memory_info = {}
    
    if torch.cuda.is_available():
        memory_info.update({
            "gpu_memory_allocated_mb": torch.cuda.memory_allocated() / (1024**2),
            "gpu_memory_reserved_mb": torch.cuda.memory_reserved() / (1024**2),
            "gpu_memory_cached_mb": torch.cuda.memory_cached() / (1024**2),
        })
    
    return memory_info

def log_memory_usage(logger: logging.Logger, context: str = "") -> None:
    """메모리 사용량 로깅"""
    memory_info = monitor_memory_usage(logger)
    logger.info(f"[{context}] Memory usage: {memory_info}")

def auto_optimize_device_settings(device: torch.device, logger: logging.Logger) -> Dict[str, Any]:
    """디바이스 설정 자동 최적화"""
    settings = {}
    
    if device.type == 'cuda':
        # CUDA 최적화 설정
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        settings["cudnn_benchmark"] = True
        settings["cudnn_deterministic"] = False
        
        # 메모리 최적화
        torch.cuda.empty_cache()
        settings["memory_cleared"] = True
        
        # GPU 정보 로깅
        gpu_name = torch.cuda.get_device_name()
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"Using GPU: {gpu_name} ({gpu_memory:.1f}GB)")
        
    else:
        # CPU 최적화 설정
        torch.set_num_threads(min(32, os.cpu_count() or 1))  # 라이젠 AI 9/HX 370에 맞춰 스레드 수 대폭 증가
        settings["num_threads"] = torch.get_num_threads()
        logger.info(f"Using CPU with {settings['num_threads']} threads")
    
    return settings

# robust JSON 문자열 로더(주석/트레일링 콤마/잘린 JSON 복구)
_RE_TCOMMA1 = re.compile(r",\s*([}\]])")          # trailing comma
_RE_LINECOM = re.compile(r"//.*?$", re.MULTILINE) # // comment
_RE_BLKCOM  = re.compile(r"/\*.*?\*/", re.DOTALL) # /* */ comment

def _read_json_robust(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        txt = p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        try:
            txt = p.read_text(encoding="latin-1", errors="ignore")
        except Exception:
            return {}
    txt = txt.lstrip("\ufeff").strip()
    if not txt:
        return {}
    # 1차: 그대로 시도
    try:
        obj = json.loads(txt);
        return obj if isinstance(obj, dict) else {}
    except Exception:
        pass
    # 전처리: 주석/트레일링 콤마 제거
    t = _RE_LINECOM.sub("", txt)
    t = _RE_BLKCOM.sub("", t)
    t = _RE_TCOMMA1.sub(r"\1", t)
    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        pass
    # 잘린 JSON 복구: 가장 바깥 중괄호 범위
    s, e = t.find("{"), t.rfind("}")
    if s != -1 and e != -1 and e > s:
        chunk = t[s:e+1]
        chunk = _RE_TCOMMA1.sub(r"\1", chunk)
        try:
            obj = json.loads(chunk)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    # 라인 단위 복구
    for line in t.splitlines():
        line = line.strip()
        if not line or "{" not in line:
            continue
        try:
            line = _RE_TCOMMA1.sub(r"\1", line)
            obj = json.loads(line)
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    return {}

def resolve_device(device: str = "auto") -> str:
    d = (device or "auto").lower()
    if d != "auto":
        return d
    if torch.cuda.is_available():
        # 안정적 fp32 매트멀 정밀도(2.x)
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def ensure_outdir(base: str, safe_overwrite: bool = True) -> Path:
    """
    디렉터리 생성 유틸.
    - safe_overwrite=True이고 동일 폴더가 있으면 1회만 타임스탬프 붙여 새 폴더 생성
    """
    p = Path(base)
    if p.exists() and safe_overwrite:
        # 이미 타임스탬프가 붙어있더라도 무조건 1회만 새로 만듦
        p = p.with_name(p.stem + f".{_now_tag()}" + p.suffix)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.mkdir(parents=True, exist_ok=True)
    return p

def _read_pointer_file(p: str) -> str:
    """포인터 파일(.path)을 읽어 실제 라벨 파일 경로로 해석.
    유효한 파일을 가리키면 해당 경로를 반환, 아니면 빈 문자열.
    """
    try:
        q = Path(p)
        if q.is_file():
            target = q.read_text(encoding="utf-8", errors="ignore").strip()
            if target and Path(target).is_file():
                print(f"[labels] pointer: {p} -> {target}")
                return str(Path(target))
    except Exception:
        pass
    return ""

def resolve_labels_path(path: str) -> str:
    """
    --labels-jsonl 경로 해석(포인터/퀵/밸런스 자동 탐색 포함):
      - LABELS_JSONL 환경변수 보조 사용
      - 포인터(.path) 파일 우선 해석
      - 명시 파일이면 그대로 사용
      - 기본 포인터들(data/labels.*.latest.path) 자동 해석
      - 디렉터리/패턴/기본 검색 경로에서 최신 파일 선택
    """
    s = (path or os.getenv("LABELS_JSONL", "")).strip()

    # (A) 명시 경로가 포인터(.path)이면 즉시 해석
    if s and s.lower().endswith(".path"):
        r = _read_pointer_file(s)
        if r:
            return r

    # (B) 명시 경로가 파일이면 통과
    p = Path(s) if s else None
    if p and p.is_file():
        return str(p)
    # (B2) 명시 경로가 글롭 패턴이면 우선 검색
    if s and any(ch in s for ch in "*?[]"):
        hits = sorted(glob.glob(s), key=os.path.getmtime, reverse=True)
        if hits:
            pick = hits[0]
            print(f"[labels] resolved: '{s}' -> '{pick}'")
            return pick

    # (C) 비었거나 디렉터리/글롭 패턴이면 자동탐색
    # 1) 기본 포인터들 우선
    for ptr in (
        "data/labels.programmatic.latest.path",
        "data/labels.quick.latest.path",
        "data/labels.latest.path",
    ):
        r = _read_pointer_file(ptr)
        if r:
            return r

    # 2) 디렉터리/패턴 탐색 (밸런스 우선 → 퀵 → 일반)
    search_dirs = [s] if (s and os.path.isdir(s)) else ["data", "src/data", "."]
    patterns = [
        "labels*.bal.jsonl",
        "labels.programmatic*.bal.jsonl",
        "labels.quick*.jsonl",
        "labels*.jsonl",
    ]
    for d in search_dirs:
        for pat in patterns:
            hits = sorted(glob.glob(os.path.join(d, pat)), key=os.path.getmtime, reverse=True)
            if hits:
                pick = hits[0]
                print(f"[labels] resolved: '{d}/{pat}' -> '{pick}'")
                return pick

    # 마지막으로 원본 문자열 반환(상대/패턴일 수 있으므로)
    return s

def resolve_emb_path(path: str) -> str:
    """
    --emb-jsonl 편의 해석:
      - 존재하면 그대로 사용
      - 글롭 패턴(*,?,[])이면 최신 파일 선택
      - 디렉터리면 *.jsonl 중 최신 선택
      - 기본 폴백: src/embeddings/run*.jsonl
    """
    s = str(path or "").strip()
    if not s:
        # 기본 탐색 패턴만 사용
        candidates = sorted(glob.glob("src/embeddings/run*.jsonl"),
                           key=os.path.getmtime, reverse=True)
        return candidates[0] if candidates else ""
    
    p = Path(s)
    if p.exists() and p.is_file():  # exists → is_file로 변경
        return str(p)
    
    candidates = []
    if any(ch in s for ch in "*?[]"):
        candidates = sorted(glob.glob(s), key=os.path.getmtime, reverse=True)
    elif os.path.isdir(s):
        candidates = sorted(glob.glob(os.path.join(s, "*.jsonl")),
                           key=os.path.getmtime, reverse=True)
    
    if candidates:
        pick = candidates[0]
        print(f"[emb] resolved: '{s}' -> '{pick}'")
        return pick
    return s

def set_seed(seed: int, deterministic: bool = True):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # CUDA 결정론 모드
    if deterministic:
        if torch.cuda.is_available() and "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        try:
            torch.use_deterministic_algorithms(True)
        except Exception as e:
            print(f"[warn] deterministic algorithms not fully enabled: {e} -> fallback safe mode")
            try:
                torch.use_deterministic_algorithms(False)
            except Exception:
                pass
    # CuDNN 권장 플래그
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not bool(deterministic)

# 텍스트 정규화(한/영/숫자/하이픈 유지)
def _norm_ko(s: str) -> str:
    s = unicodedata.normalize("NFKC", str(s)).replace("\u200b","").strip().lower()
    s = re.sub(r"[^\w\s\u3131-\u318e\uac00-\ud7a3\-]", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def _sub_variants(name: str) -> set:
    """'만족감'→{'만족감','만족'}, '희열-감'→{'희열-감','희열'}. '스러움/감정' 포함."""
    n = _norm_ko(name)
    v = {n}
    if "-" in n:
        v.add(n.split("-", 1)[-1])
    for suf in ("스러움","감정","감","함"):
        if n.endswith(suf) and len(n) > len(suf):
            v.add(n[:-len(suf)])
    if " " in n:
        v.add(n.replace(" ", ""))
    return v

def build_sub_alias_map(sub_list: list) -> dict:
    """EMOTIONS의 sub_list에서 alias → canonical 매핑 생성"""
    alias = {}
    for s in sub_list:
        for v in _sub_variants(s):
            alias[v] = s
    return alias

def map_sub_alias(label: str, alias_map: dict) -> Optional[str]:
    """라벨 문자열을 canonical sub로 매핑(정확/경계 기반 부분일치). 짧은 키워드는 exact만."""
    if not label:
        return None
    t = _norm_ko(label)
    # 1) exact
    if t in alias_map:
        return alias_map[t]
    # 2) 경계 기반 부분일치(3자 이상)
    for k, v in alias_map.items():
        if len(k) >= 3:
            pat = rf"(?<![0-9A-Za-z\u3131-\u318e\uac00-\ud7a3]){re.escape(k)}(?![0-9A-Za-z\u3131-\u318e\uac00-\ud7a3])"
            if re.search(pat, t):
                return v
    return None

def macro_f1_masked(y_true, y_pred, n_cls, present_only=False, min_support=0):
    """등장 클래스만 혹은 지원수 하한을 두고 macro F1 계산."""
    import numpy as np
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    supports = np.bincount(y_true, minlength=n_cls)
    mask = np.ones(n_cls, dtype=bool)
    if present_only: mask &= (supports > 0)
    if min_support > 0: mask &= (supports >= min_support)
    f1s = []
    for c in range(n_cls):
        if not mask[c]:
            continue
        tp = np.sum((y_true==c)&(y_pred==c))
        fp = np.sum((y_true!=c)&(y_pred==c))
        fn = np.sum((y_true==c)&(y_pred!=c))
        p = tp/max(1,tp+fp); r = tp/max(1,tp+fn)
        f = 2*p*r/max(1e-9,p+r)
        f1s.append(f)
    return float(np.mean(f1s)) if f1s else 0.0

def _safe_float(x, default=0.0):
    try:
        v = float(x)
        if v != v or v in (float("inf"), float("-inf")):  # NaN/Inf
            return default
        return v
    except Exception:
        return default

def _label_confidence(o: dict) -> float:
    """라벨 신뢰도 추출: reasons.confidence > debug_meta.conf > score_main > score"""
    r  = (o.get("reasons")    or {})
    dm = (o.get("debug_meta") or {})
    return _safe_float(
        r.get("confidence")
        or dm.get("conf")
        or o.get("score_main")
        or o.get("score")
        or 0.0
    )

def _label_fallback(o: dict) -> bool:
    """폴백 사용 여부: reasons.fallback_used > debug_meta.fallback"""
    r  = (o.get("reasons")    or {})
    dm = (o.get("debug_meta") or {})
    return bool(r.get("fallback_used") or dm.get("fallback"))

def _resolve_record_id(o: dict, preferred: str | None = "id") -> Optional[str]:
    """
    1) preferred(id 등) → 2) meta.trace_id → 3) trace_id → 4) sample_id
    → 5) text_hash → 6) id(재확인) → 7) hash(text)
    """
    if not isinstance(o, dict):
        return None
    rid = None
    if preferred and o.get(preferred) is not None:
        rid = o.get(preferred)
    if rid is None:
        rid = ((o.get("meta") or {}).get("trace_id")
               or o.get("trace_id")
               or o.get("sample_id")
               or o.get("text_hash"))
    if rid is None and o.get("id") is not None:  # ★ preferred=None일 때도 id를 마지막에 한 번 더 본다
        rid = o.get("id")
    if rid is None:
        t = o.get("text")
        if isinstance(t, str) and t:
            import hashlib
            rid = hashlib.md5(t.encode("utf-8","ignore")).hexdigest()
    return str(rid) if rid is not None else None

# ---- 구성 (FINAL+) ----
@dataclass
class TrainCfg:
    # 필수
    emb_jsonl: str
    labels_jsonl: str

    # 참조 스키마
    emotions_json: str = "src/EMOTIONS.JSON"          # 메인→서브 정의
    session_dir: Optional[str] = None                 # (선택) 세션 피처 결합 시 사용

    # 학습 타깃
    main: Optional[str] = None                        # '희'|'노'|'애'|'락' (None이면 외부 로직에서 all-mains)
    outdir: str = "src/models/sub_v2"                 # 항상 src/models 하위로 정착

    # 조인/필드
    id_field: str = "id"
    emb_field: str = "embedding"

    # 라벨 필터링/가중
    min_conf: float = 0.30                            # reasons.confidence 하한
    exclude_fallback: bool = True                     # fallback_used 제외
    conf_power: float = 2.0                           # 샘플 가중: w = clip(conf**conf_power, 0.2, 1.0)

    # 데이터 규모/다양성 하한
    min_join: Optional[int] = 20                      # 최소 조인 샘플 수(None이면 자동 하한 사용)
    min_classes: int = 1                              # 최소 서로 다른 서브 라벨 수

    # 학습 하이퍼파라미터
    batch: int = 128
    epochs: int = 15
    lr: float = 2e-4
    weight_decay: float = 1e-4
    val_ratio: float = 0.2
    seed: int = 42

    # 조기 종료(옵션)
    early_stop_patience: Optional[int] = 5            # 0/None이면 조기 종료 끔
    min_epochs: int = 5                               # 조기 종료를 보더라도 최소 보장 에포크
    early_stop_metric: str = "macro_f1"               # 'macro_f1'|'present_f1'|'loss'

    # 클래스 불균형/샘플러(옵션)
    class_weight_mode: str = "effective"              # 'none'|'freq_inv'|'effective'|'cb-focal'
    beta: float = 0.999                               # effective-number beta
    use_weighted_sampler: bool = False
    label_smoothing: float = 0.0
    focal_gamma: float = 2.0

    # 검증 지표 해석 보조(옵션)
    eval_present_only: bool = False                   # present-only F1을 보조 지표로 활용
    eval_min_support: int = 0                         # sup>=K 클래스만 평균

    # 안정성/AMP/누적(옵션)
    grad_clip: float = 0.0                            # >0이면 해당 값으로 clip
    grad_accum: int = 1
    amp: bool = False
    amp_dtype: str = "fp16"                           # 'fp16'|'bf16'
    amp_fallback: bool = False

     # --- 새로운 개선사항 ---
    mixup_alpha: float = 0.0 
    kfold: int = 0                 # k>1이면 K-Fold 학습
    export_oof: Optional[str] = None  # "data/oof.sub.{main}.npz" 등 경로
    normalize: str = "l2"   # 'l2'|'zscore'|'none'

    # 스케줄러(옵션)
    sched: str = "onecycle"                           # 'none'|'onecycle'|'cosine'
    min_lr: float = 1e-5

    # 실행/저장
    device: str = "auto"                              # 'auto' | 'cuda' | 'cpu' | 'mps'
    safe_overwrite: bool = True                       # 같은 outdir가 있으면 타임스탬프로 새 폴더 생성(단, 1회만)
    save_history: bool = True
    log_interval: int = 50
    debug: int = 0
    resume_from: Optional[str] = None                 # warm-start 체크포인트 경로(선택)

    # 내부 계산용(런타임에서 채움)
    _device_resolved: str = "cpu"
    _prepared: bool = False                           # 한 번만 준비(idempotent)
    _run_dir: Optional[str] = None                    # 이번 실행이 쓰는 최종 저장 루트
    
    # 개선된 로깅 및 모니터링 옵션
    log_level: str = "INFO"                           # 로깅 레벨
    log_file: Optional[str] = None                    # 로그 파일 경로
    enable_performance_monitoring: bool = True        # 성능 모니터링 활성화
    enable_memory_monitoring: bool = True             # 메모리 모니터링 활성화
    auto_optimize_batch_size: bool = True             # 자동 배치 크기 최적화
    quality_threshold: float = 0.7                    # 데이터 품질 임계값

    # ---- public helpers ----
    def as_dict(self) -> Dict:
        d = asdict(self)
        d["_device_resolved"] = self._device_resolved
        return d

    def effective_min_join(self, sub_count: int) -> int:
        """
        자동 하한 계산(작은 셋 친화). min_join이 명시되면 그 값을 그대로 사용.
        """
        if self.min_join is not None:
            return int(self.min_join)
        # 경험칙: 최소 12, 최대 60, 서브*2
        return max(12, min(60, int(sub_count) * 2))

    def resolve_run_dir(self) -> Path:
        """
        준비 후 사용: 이번 실행이 저장할 최종 루트 폴더(Path).
        """
        return Path(self._run_dir or self.outdir)

    def validate_and_prepare(self) -> None:
        """
        실행 직전 1회만 호출되는 준비 단계.
        - 경로 검사
        - outdir을 항상 src/models 하위에 1회만 생성(타임스탬프 중복 방지)
        - 디바이스/시드/필드/범위 보정
        - 조기 종료/로그 옵션 보정
        """
        if self._prepared:
            return

        # 1) 경로 검사(+라벨/임베딩 자동 탐색)
        self.emb_jsonl = resolve_emb_path(self.emb_jsonl)
        self.labels_jsonl = resolve_labels_path(self.labels_jsonl)
        
        if not Path(self.emb_jsonl).is_file():  # exists → is_file로 변경
            raise FileNotFoundError(f"emb_jsonl not found: {self.emb_jsonl}")
        if not Path(self.labels_jsonl).is_file():  # exists → is_file로 변경
            raise FileNotFoundError(f"labels_jsonl not found: {self.labels_jsonl}")
        if self.emotions_json and not Path(self.emotions_json).exists():
            print(f"[warn] emotions_json not found: {self.emotions_json} (fallback to empty schema)")

        # 2) outdir 정규화: 항상 src/models 하위로 고정 + 타임스탬프 1회만 부여
        out = Path(self.outdir)
        if not out.is_absolute():
            s = str(out).replace("\\", "/")
            if not s.startswith("src/"):
                out = Path("src") / out
        out = ensure_outdir(str(out), safe_overwrite=self.safe_overwrite)   # 최초 1회만 TS 부여
        self.outdir = str(out)
        self._run_dir = self.outdir  # 런 디렉터리 보존

        # 3) 디바이스/시드
        self._device_resolved = resolve_device(self.device)
        set_seed(self.seed, deterministic=True)

        # 4) 필드/범위 보정
        self.id_field = self.id_field or "id"
        self.emb_field = self.emb_field or "embedding"

        # 값 범위
        try:
            self.min_conf  = float(max(0.0, min(1.0, self.min_conf)))
        except Exception:
            self.min_conf = 0.30
        try:
            self.val_ratio = float(max(0.05, min(0.5, self.val_ratio)))
        except Exception:
            self.val_ratio = 0.2
        try:
            self.batch     = int(max(8, self.batch))
        except Exception:
            self.batch = 128
        try:
            self.epochs    = int(max(1, self.epochs))
        except Exception:
            self.epochs = 15
        
        # 배치 크기 경고
        if self.debug and self.batch > 8192:
            print(f"[warn] batch={self.batch} (unusually large) → GPU OOM 위험")

        # 하한 보정
        try:
            self.min_classes = max(1, int(self.min_classes))
        except Exception:
            self.min_classes = 1

        if self.min_join is not None:
            try:
                mj = int(self.min_join)
                self.min_join = mj if mj >= 1 else None
            except Exception:
                self.min_join = None

        # 조기 종료/지표 보정
        try:
            if self.early_stop_patience is not None:
                self.early_stop_patience = max(0, int(self.early_stop_patience))
        except Exception:
            self.early_stop_patience = 5

        try:
            self.min_epochs = max(1, int(self.min_epochs))
            if self.min_epochs > self.epochs:
                self.min_epochs = self.epochs
        except Exception:
            self.min_epochs = max(1, min(10, self.epochs // 4 or 1))

        self.early_stop_metric = (self.early_stop_metric or "macro_f1").lower()
        if self.early_stop_metric not in ("macro_f1","present_f1","loss"):
            self.early_stop_metric = "macro_f1"

        # 클래스 가중/샘플러 모드 보정
        self.class_weight_mode = (self.class_weight_mode or "effective").lower()
        if self.class_weight_mode not in ("none","freq_inv","effective","cb-focal","balanced"):
            self.class_weight_mode = "effective"
        # 'balanced'는 과거 호환 -> 'freq_inv'로 처리
        if self.class_weight_mode == "balanced":
            self.class_weight_mode = "freq_inv"
        self.use_weighted_sampler = bool(self.use_weighted_sampler)

        # 검증 보조 파라미터 보정
        self.eval_present_only = bool(self.eval_present_only)
        try:
            self.eval_min_support = max(0, int(self.eval_min_support))
        except Exception:
            self.eval_min_support = 0

        # 안정성 옵션
        try:
            self.grad_clip = float(self.grad_clip)
            if self.grad_clip < 0:
                self.grad_clip = 0.0
        except Exception:
            self.grad_clip = 0.0

        # 로그/히스토리
        self.save_history = bool(self.save_history)
        try:
            self.log_interval = max(1, int(self.log_interval))
        except Exception:
            self.log_interval = 50

        if self.debug:
            print("[cfg]", json.dumps(self.as_dict(), ensure_ascii=False, indent=2))

        self._prepared = True

# ---- EMOTIONS.JSON 로더 ----
def load_emotions_map(path: str) -> Dict[str, List[str]]:
    """
    EMOTIONS.JSON에서 메인→서브 목록 로드.
    지원 포맷:
      { "희": { "sub_emotions": { "기쁨": {...}, ... } }, ... }
      { "희": ["기쁨","행복",...], ... }
    누락/형식불일치시 안전한 기본값 반환.
    """
    obj = _read_json_robust(path) if path and Path(path).exists() else {}
    out: Dict[str, List[str]] = {}

    def _norm_list(x) -> List[str]:
        if isinstance(x, list):
            return [str(t) for t in x if str(t).strip()]
        if isinstance(x, dict):
            return [str(k) for k in x.keys()]
        return []

    if isinstance(obj, dict):
        for main, spec in obj.items():
            subs: List[str] = []
            if isinstance(spec, dict):
                if "sub_emotions" in spec:
                    subs = _norm_list(spec.get("sub_emotions"))
                elif "subs" in spec:
                    subs = _norm_list(spec.get("subs"))
                elif isinstance(spec.get("sub_emotions"), list):
                    subs = _norm_list(spec.get("sub_emotions"))
            elif isinstance(spec, list):
                subs = _norm_list(spec)
            out[str(main)] = subs

    # 필수 메인 보정(없으면 빈 리스트라도 생성)
    for m in ("희","노","애","락"):
        out.setdefault(m, [])

    # 중복 제거 + 안정 정렬
    for m in list(out.keys()):
        seen = set(); uniq = []
        for t in out[m]:
            if t not in seen and str(t).strip():
                seen.add(t); uniq.append(t)
        out[m] = uniq

    return out


# --------------------
# 1) 데이터 로딩 (robust)
# --------------------
def load_embeddings_jsonl(path: str, id_field: str, emb_field: str) -> Tuple[np.ndarray, List[str]]:
    """
    run.jsonl에서 임베딩을 로드.
    - id 추출: 우선 id_field, 없으면 meta.trace_id -> trace_id -> sample_id
    - 방어 로직:
      * NaN/Inf 제거
      * 영벡터(norm==0) 제거
      * 차원 불일치 시 '최빈 차원'만 채택(나머지는 버림)
      * 중복 id는 '뒤에서 읽은 레코드'로 덮어씀(최신 우선)
    """
    ids: List[str] = []
    vecs: List[np.ndarray] = []
    dims: Counter = Counter()
    raw_records: List[Tuple[str, np.ndarray]] = []

    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                o = json.loads(ln)
            except Exception:
                continue

            _id = _resolve_record_id(o, preferred=id_field)   # ← 핵심 한 줄: text_hash/trace_id 우선, 그 다음 id
            emb = o.get(emb_field)

            if not _id:
                continue
            if isinstance(emb, list):
                v = np.asarray(emb, dtype=np.float32)
            elif isinstance(emb, np.ndarray):
                v = emb.astype(np.float32, copy=False)
            else:
                continue

            if v.ndim != 1 or v.size == 0:
                continue
            if not np.isfinite(v).all():
                continue
            if float(np.linalg.norm(v)) == 0.0:
                continue

            raw_records.append((_id, v))
            dims.update([v.size])

    if not raw_records:
        raise ValueError(f"No valid embeddings in {path}")

    # 최빈 차원(다수결)만 수집
    dim_mode, _ = dims.most_common(1)[0]
    uniq: Dict[str, np.ndarray] = {}
    for _id, v in raw_records:
        if v.size != dim_mode:
            continue
        # 동일 id가 여러 번 나오면 '마지막 것'을 채택(보통 최신)
        uniq[_id] = v

    if not uniq:
        raise ValueError(f"All embeddings filtered by dimension; expected dim={dim_mode}")

    # 리스트로 정렬(안정적 재현성 위해 id 정렬)
    ids = sorted(uniq.keys())
    X = np.vstack([uniq[i] for i in ids]).astype(np.float32, copy=False)

    print(f"[emb-load] Loaded {len(ids)} embeddings from {path} (dim={X.shape[1]}, filtered={len(raw_records)-len(ids)} invalid/duplicate)")

    return X, ids


def load_labels_jsonl(
    path: str,
    *,
    min_conf: float,
    exclude_fallback: bool,
    target_main: str,
    conf_power: float = 2.0,
    sub_alias_map: dict | None = None,   # ★ 추가
) -> Tuple[Dict[str, str], Dict[str, float]]:
    y_map: Dict[str, str] = {}
    w_map: Dict[str, float] = {}
    conf_map: Dict[str, float] = {}

    # 진단용 카운트
    cand_total = 0
    kept_conf = 0
    kept_fb = 0

    # BOM 섞일 수 있어 utf-8-sig 로딩
    with open(path, "r", encoding="utf-8-sig") as f:
        for ln in f:
            ln = ln.strip()
            if not ln: 
                continue
            try:
                o = json.loads(ln)
            except Exception:
                continue
            if o.get("main") != target_main:
                continue

            # 후보 카운트(필터 전)
            cand_total += 1
            conf = _label_confidence(o)
            if conf >= float(min_conf):
                kept_conf += 1
            if not _label_fallback(o):
                kept_fb += 1

            _id = _resolve_record_id(o, preferred="id")  # meta.trace_id → trace_id → sample_id → text_hash → (id) 순으로 해석
            if not _id:
                continue
            sub = (o.get("sub")
                   or o.get("label_sub")
                   or (o.get("reasons") or {}).get("label_sub")
                   or o.get("label")
                   or o.get("pred_sub"))
            if not isinstance(sub, str) or not sub.strip():
                continue
            sub = sub.strip()
            if sub_alias_map:
                mapped = map_sub_alias(sub, sub_alias_map)
                if mapped: 
                    sub = mapped
            if conf < float(min_conf):
                continue
            if bool(exclude_fallback) and _label_fallback(o):
                continue

            if (_id not in conf_map) or (conf > conf_map[_id]):
                y_map[_id] = sub
                conf_map[_id] = conf
                w_map[_id] = float(max(0.2, min(1.0, (conf ** float(conf_power)))))

    # 라벨 조인 직전 진단 로그(요청된 2줄)
    print(f"[lab-load] main={target_main} | total_candidates={cand_total} | kept_by_conf={kept_conf} | kept_by_fallback={kept_fb} | final_labels={len(y_map)}")
    print(f"[diag] main={target_main} | cand={cand_total} | kept_conf={kept_conf}")
    print(f"[diag] main={target_main} | kept_fb={kept_fb} (exclude_fallback={exclude_fallback})")

    return y_map, w_map


def join_xy(
    X: np.ndarray,
    ids: List[str],
    y_map: Dict[str, str],
    w_map: Dict[str, float],
    sub_list: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    임베딩 X/ids와 라벨 y_map을 조인하여 학습용 (feats, labels, weights) 반환.
    - sub_list에 없는 라벨은 버림
    - 반환: Xj[N, D], yj[N], wj[N], hit(조인 성공 개수)
    """
    if len(ids) != X.shape[0]:
        raise ValueError(f"ids length {len(ids)} != X rows {X.shape[0]}")

    sub_to_idx = {s: i for i, s in enumerate(sub_list)}
    feats, labels, weights = [], [], []
    hit = 0

    for i, _id in enumerate(ids):
        sub = y_map.get(_id)
        if sub is None:
            continue
        idx = sub_to_idx.get(sub)
        if idx is None:
            # 알 수 없는 서브라벨이면 스킵
            continue
        feats.append(X[i])
        labels.append(idx)
        weights.append(w_map.get(_id, 1.0))
        hit += 1

    if hit == 0:
        raise ValueError("No joined samples. Check id/label mapping and sub_list coverage.")

    Xj = np.asarray(feats, dtype=np.float32)
    yj = np.asarray(labels, dtype=np.int64)
    wj = np.asarray(weights, dtype=np.float32)
    return Xj, yj, wj, hit


# --- NEW: 11모듈 표준 피처 로더 (NxF float32) ---
def load_module_features(session_dir: Optional[str], ids: List[str]) -> Optional[np.ndarray]:
    """
    표준: session_dir 하위 또는 data/features.latest.(npz|npy)
    - npz: {"ids": [...], "X": (N,F)} 또는 {"<any>":(N,F), "ids":[...]}
    - 누락/미스매치 시 None
    """
    try:
        paths = []
        if session_dir:
            for p in ("feature_bank.npz", "features.latest.npz", "features.latest.npy"):
                paths.append(Path(session_dir) / p)
        paths += [Path("data/features.latest.npz"), Path("data/features.latest.npy")]
        
        # 디버깅: 경로 확인
        checked_paths = [str(p) for p in paths]
        hit = next((p for p in paths if p.exists()), None)
        if not hit:
            # 경로 확인 로그 (디버그 모드에서만)
            import os
            if os.getenv("DEBUG_MODULE_FEATURES", "0") == "1":
                print(f"[module-features] Checked paths: {checked_paths}")
                print(f"[module-features] None of the paths exist")
            return None
        if hit.suffix == ".npz":
            z = np.load(str(hit))
            if "X" in z and "ids" in z:
                id2row = {i: k for k, i in enumerate(list(z["ids"]))}
                rows = [id2row.get(i, -1) for i in ids]
                if any(r < 0 for r in rows):  # 일부라도 없으면 무시
                    return None
                return z["X"][rows].astype(np.float32, copy=False)
            # 단일 배열만 있을 때
            arr = next((z[k] for k in z.files if k != "ids"), None)
            if arr is not None and arr.shape[0] == len(ids):
                return arr.astype(np.float32, copy=False)
            return None
        else:
            X = np.load(str(hit))
            return X.astype(np.float32, copy=False) if X.shape[0] == len(ids) else None
    except Exception:
        return None


# --------------------
# 2) Dataset / Model
# --------------------
class ClassBalancedBatchSampler(Sampler):
    """각 클래스에서 균등 샘플링하여 배치를 구성 (잔여는 순환)."""
    def __init__(self, labels: np.ndarray, batch_size: int):
        self.labels = labels
        self.bs = int(batch_size)
        self.by_cls = {c: np.where(labels == c)[0].tolist() for c in np.unique(labels)}
        for c in self.by_cls:
            random.shuffle(self.by_cls[c])
        self._p = {c: 0 for c in self.by_cls}
        self._order = list(self.by_cls.keys())
    
    def __iter__(self):
        n = len(self.labels)
        i = 0
        while i < n:
            take = []
            # 각 클래스에서 1개씩 뽑기 → 남은 칸은 라운드로빈
            for c in self._order:
                if len(take) >= self.bs: 
                    break
                pool = self.by_cls[c]
                p = self._p[c]
                if not pool: 
                    continue
                take.append(pool[p % len(pool)])
                self._p[c] = p + 1
            # 남은 칸 채우기
            j = 0
            while len(take) < self.bs:
                c = self._order[j % len(self._order)]
                pool = self.by_cls[c]
                p = self._p[c]
                if pool:
                    take.append(pool[p % len(pool)])
                    self._p[c] = p + 1
                j += 1
            i += len(take)
            yield from take
    
    def __len__(self):
        return int(math.ceil(len(self.labels) / max(1, self.bs)))


class SubDataset(Dataset):
    """
    임베딩 기반 서브감정 학습용 데이터셋.
    - X: [N, D] float32
    - y: [N]     int64 (클래스 인덱스)
    - w: [N]     float32 (샘플 가중치)
    - normalize: None | 'l2' | 'zscore'
        * 'l2'     : 각 벡터를 벡터 노름으로 정규화
        * 'zscore' : 전체 특성에 대해 (x - mean) / std (std<eps이면 1로 처리)
      -> 정규화는 __init__ 시 **사전 적용**되어 런타임 오버헤드가 없음.
    """
    def __init__(self,
                 X: np.ndarray,
                 y: np.ndarray,
                 w: np.ndarray,
                 normalize: Optional[str] = None,
                 eps: float = 1e-8):
        assert isinstance(X, np.ndarray) and X.ndim == 2, f"X must be [N,D] ndarray, got {type(X)} {getattr(X, 'shape', None)}"
        assert isinstance(y, np.ndarray) and y.ndim == 1, f"y must be [N] ndarray"
        assert isinstance(w, np.ndarray) and w.ndim == 1, f"w must be [N] ndarray"
        assert X.shape[0] == y.shape[0] == w.shape[0], f"length mismatch: X={X.shape[0]} y={y.shape[0]} w={w.shape[0]}"

        X = X.astype(np.float32, copy=False)
        y = y.astype(np.int64, copy=False)
        w = w.astype(np.float32, copy=False)

        self.stats_: Dict[str, Any] = {"normalize": normalize or "none"}

        if normalize is not None:
            norm = normalize.lower()
            if norm == "l2":
                norms = np.linalg.norm(X, axis=1, keepdims=True).astype(np.float32)
                np.maximum(norms, eps, out=norms)
                X = X / norms
                self.stats_["avg_l2norm_after"] = float(np.linalg.norm(X, axis=1).mean())
            elif norm == "zscore":
                mean = X.mean(axis=0, dtype=np.float64)
                std  = X.std(axis=0, dtype=np.float64)
                std[std < eps] = 1.0
                X = ((X - mean) / std).astype(np.float32, copy=False)
                self.stats_["z_mean"] = float(np.abs(X.mean()).item())
                self.stats_["z_std"]  = float(X.std().item())
            else:
                raise ValueError(f"normalize must be None|'l2'|'zscore', got {normalize}")

        self.X = torch.from_numpy(X).contiguous()
        self.y = torch.from_numpy(y).contiguous()
        self.w = torch.from_numpy(w).contiguous()

        self.stats_["N"] = int(self.X.shape[0])
        self.stats_["D"] = int(self.X.shape[1])
        self.stats_["avg_l2norm_before"] = None

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, i: int):
        return self.X[i], self.y[i], self.w[i]


class MLPHead(nn.Module):
    """
    심플하지만 안정적인 MLP 분류 헤드.
    구성: Linear -> LayerNorm -> GELU -> Dropout -> Linear -> GELU -> Dropout -> Linear
    - in_dim : 입력 차원(임베딩 차원)
    - n_cls  : 클래스 수
    - hidden : 첫 히든 폭(기본 512)
    - drop   : 드롭아웃 비율(기본 0.2)
    - act    : 활성함수 ('gelu'|'relu'|'silu')
    - logit_scale: 로짓 스케일(캘리브레이션용), 기본 1.0(비활성)
    """
    def __init__(self,
                 in_dim: int,
                 n_cls: int,
                 hidden: int = 512,
                 drop: float = 0.2,
                 act: str = "gelu",
                 logit_scale: float = 1.0):
        super().__init__()
        assert in_dim > 0 and n_cls > 1

        if act.lower() == "relu":
            Act = nn.ReLU
        elif act.lower() == "silu":
            Act = nn.SiLU
        else:
            Act = nn.GELU

        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            Act(),
            nn.Dropout(drop),
            nn.Linear(hidden, hidden // 2),
            Act(),
            nn.Dropout(drop),
            nn.Linear(hidden // 2, n_cls),
        )
        self.register_buffer("_logit_scale", torch.tensor(float(logit_scale), dtype=torch.float32))
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            if m.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0.0
                nn.init.uniform_(m.bias, -bound, bound)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            if hasattr(m, "weight") and m.weight is not None:
                nn.init.ones_(m.weight)
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.float32:
            x = x.float()
        logits = self.backbone(x)
        if self._logit_scale is not None:
            logits = logits * self._logit_scale
        return logits


class TemperatureCalibrator:
    """클래스별 온도 스케일링으로 희귀 서브 과신 완화"""
    def __init__(self, n_cls: int):
        self.T = np.ones((n_cls,), dtype=np.float32)
        self.fitted = False
    
    def fit(self, logits: np.ndarray, y: np.ndarray, max_iter: int = 200):
        """간단한 좌표강하(클래스별 NLL 최소화)"""
        n, c = logits.shape
        T = np.ones((c,), dtype=np.float32)
        for k in range(c):
            idx = (y == k)
            if not idx.any(): 
                continue
            z = logits[idx]  # [Nk,C]
            # 1D 탐색
            t = 1.0
            for _ in range(max_iter):
                p = np.exp(z[:, k] / t) / np.exp(z / t).sum(1, keepdims=True)
                grad = -np.mean((1 - p) * z[:, k] / (t * t))
                t = max(0.05, min(5.0, t - 0.1 * grad))
            T[k] = t
        self.T = T
        self.fitted = True
    
    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        """온도 스케일링 적용"""
        if not self.fitted: 
            return logits
        T = torch.as_tensor(self.T, device=logits.device, dtype=logits.dtype)
        return logits / T


class CBFocalLoss(torch.nn.Module):
    """Class-Balanced Focal Loss with optional label smoothing."""
    def __init__(self, class_weights: torch.Tensor, gamma: float = 2.0,
                 label_smoothing: float = 0.0, reduction: str = "none"):
        super().__init__()
        self.register_buffer("alpha", class_weights / (class_weights.mean() + 1e-12))
        self.gamma = float(gamma)
        self.smooth = float(label_smoothing)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        num_classes = int(logits.size(1))
        with torch.no_grad():
            y = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1.0)
            if self.smooth > 0.0:
                y = y * (1.0 - self.smooth) + (self.smooth / num_classes)
        logp = torch.log_softmax(logits, dim=1)
        p = logp.exp()
        pt = (y * p).sum(dim=1)
        focal = (1.0 - pt).clamp(min=1e-6).pow(self.gamma)
        alpha_t = self.alpha[target]
        per_sample = -(alpha_t * focal * (y * logp).sum(dim=1))  # [B]

        if self.reduction == "none":
            return per_sample
        if self.reduction == "sum":
            return per_sample.sum()
        return per_sample.mean()


from torch.utils.data import WeightedRandomSampler

def make_weighted_sampler(train_labels: list[int] | np.ndarray, class_weights: torch.Tensor) -> WeightedRandomSampler:
    """
    클래스 가중치로부터 per-sample 가중치를 만들어 샘플러 구성.
    """
    if isinstance(train_labels, np.ndarray):
        tl = torch.from_numpy(train_labels.astype(np.int64))
    else:
        tl = torch.tensor(train_labels, dtype=torch.long)
    sample_w = class_weights[tl].to(dtype=torch.double)
    # 합 1.0 근처로 정규화(선택)
    sample_w = sample_w * (sample_w.numel() / (sample_w.sum().clamp_min(1e-9)))
    return WeightedRandomSampler(weights=sample_w, num_samples=int(sample_w.numel()), replacement=True)


# --------------------
# 3) 학습/평가 루프 (robust & metrics & early-stopping)
# --------------------
def compute_class_weights(labels: list[int] | np.ndarray, num_classes: int, mode: str="effective", beta: float=0.999) -> torch.Tensor:
    """
    클래스 가중치 계산
    - none       : 모두 1
    - freq_inv   : 1 / n_c  (평균 1로 정규화)
    - effective  : (1-beta) / (1 - beta^{n_c})  (CVPR'19 Class-Balanced Loss)
    - cb-focal   : effective과 동일한 w를 alpha로 사용
    """
    import numpy as _np
    from collections import Counter as _Counter
    y = _np.asarray(labels, dtype=_np.int64)
    cnt = _np.zeros(num_classes, dtype=_np.float64)
    c = _Counter(y.tolist())
    for k, v in c.items():
        if 0 <= int(k) < num_classes:
            cnt[int(k)] = float(v)

    if mode == "none":
        w = _np.ones(num_classes, dtype=_np.float64)
    elif mode == "freq_inv" or mode == "balanced":
        w = 1.0 / _np.clip(cnt, 1.0, None)
    elif mode in ("effective", "cb-focal"):
        w = (1.0 - float(beta)) / (1.0 - _np.power(float(beta), _np.clip(cnt, 1.0, None)))
    else:
        w = _np.ones(num_classes, dtype=_np.float64)

    # (기존 w 계산 후) 희귀 서브 보너스: cnt < rare_th 이면 × rare_boost
    cfg_th = int(os.getenv("RARE_TH", "0"))
    cfg_boost = float(os.getenv("RARE_BOOST", "0"))
    rare_th = cfg_th if cfg_th > 0 else max(3, int(0.005 * max(1, len(labels))))
    rare_boost = cfg_boost if cfg_boost > 0 else 1.5
    for c in range(num_classes):
        if cnt[c] > 0 and cnt[c] < rare_th:
            w[c] *= rare_boost
    
    # 평균 1.0으로 정규화
    w = w * (num_classes / (w.sum() + 1e-12))
    return torch.tensor(w, dtype=torch.float32)


def split_train_val(n_or_y, ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    학습/검증 분할.
    - y 배열이 들어오면 '계층적 분할(각 클래스 비율 유지)' 시도.
    - 정수 N이 들어오면 기존 방식대로 단순 셔플 분할.
    """
    rng = np.random.RandomState(seed)
    if isinstance(n_or_y, np.ndarray):
        y = n_or_y
        idx = np.arange(len(y))
        val_idx = []
        for c in np.unique(y):
            cls_idx = idx[y == c]
            rng.shuffle(cls_idx)
            k = max(1, int(len(cls_idx) * ratio))
            val_idx.append(cls_idx[:k])
        val_idx = np.concatenate(val_idx, axis=0) if val_idx else np.array([], dtype=int)
        val_idx = np.unique(val_idx)
        train_mask = np.ones(len(y), dtype=bool)
        train_mask[val_idx] = False
        tr_idx = np.where(train_mask)[0]
        rng.shuffle(tr_idx); rng.shuffle(val_idx)
        if val_idx.size == 0:
            val_idx = np.array([tr_idx[0]]); tr_idx = tr_idx[1:]
        return tr_idx, val_idx
    else:
        n = int(n_or_y)
        idx = np.arange(n); rng.shuffle(idx)
        val = max(1, int(n * ratio))
        return idx[val:], idx[:val]


@torch.no_grad()
def _compute_macro_f1(targets: np.ndarray, preds: np.ndarray, n_cls: int) -> float:
    f1s = []
    for c in range(n_cls):
        tp = np.sum((targets == c) & (preds == c))
        fp = np.sum((targets != c) & (preds == c))
        fn = np.sum((targets == c) & (preds != c))
        precision = tp / max(1, tp + fp)
        recall    = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-9, precision + recall)
        f1s.append(f1)
    return float(np.mean(f1s)) if f1s else 0.0


@torch.no_grad()
def mine_hard_examples(y_true: np.ndarray, y_pred: np.ndarray, topk: int = 3) -> List[Tuple[int,int]]:
    """검증 혼동행렬 기반으로 상위 오탑 서브 페어를 추출"""
    from collections import Counter
    bad = [(int(t), int(p)) for t, p in zip(y_true, y_pred) if t != p]
    cnt = Counter(bad)
    return [p for p,_ in cnt.most_common(topk)]


def evaluate(model, dl, device, n_cls: int, *, present_only=False, min_support: int = 0,
             present_mask: Optional[np.ndarray] = None) -> Tuple[float, float, float, float, float]:
    """
    검증 루프.
    반환: (loss_mean, acc, macro_f1, f1_present, f1_supK)
    """
    model.eval()
    ce = nn.CrossEntropyLoss(reduction="mean")
    tot_loss = 0.0
    tot = 0
    correct = 0

    all_t = []; all_p = []

    for xb, yb, wb in dl:
        xb = xb.to(device); yb = yb.to(device)
        logits = model(xb)
        
        # ===== 개선사항 2: 검증 시 present_mask 안정화 =====
        if present_mask is not None:
            mask = torch.as_tensor(present_mask, device=logits.device, dtype=torch.bool)  # [C]
            keep = mask.nonzero(as_tuple=False).squeeze(1)  # [C_present]
            logits = logits.index_select(1, keep)  # [B, C_present]
            
            # 라벨을 C_present 로컬 인덱스로 매핑
            remap = torch.full((mask.numel(),), -1, device=logits.device, dtype=torch.long)
            remap[keep] = torch.arange(keep.numel(), device=logits.device)
            yb_local = remap[yb]
            
            # (안전) 혹시 -1이 있으면 해당 배치는 스킵
            valid = (yb_local >= 0)
            if not valid.all():
                logits = logits[valid]
                yb_local = yb_local[valid]
                xb = xb[valid]  # 나중에 size 계산을 위해
                yb = yb[valid]  # 원본 라벨도 필터링
        else:
            yb_local = yb
        
        if logits.size(0) == 0:  # 모든 샘플이 필터링된 경우
            continue
            
        loss = ce(logits, yb_local)
        # ===== 개선사항 2 끝 =====

        tot_loss += float(loss) * xb.size(0)
        pred = logits.argmax(dim=1)
        
        # present_mask가 있을 때는 원래 인덱스로 역매핑
        if present_mask is not None:
            keep = mask.nonzero(as_tuple=False).squeeze(1)
            pred_orig = keep[pred]
            correct += int((pred_orig == yb).sum().item())
            all_p.append(pred_orig.cpu().numpy())
        else:
            correct += int((pred == yb).sum().item())
            all_p.append(pred.cpu().numpy())
        
        tot += int(yb.size(0))
        all_t.append(yb.cpu().numpy())

    if tot == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    targets = np.concatenate(all_t, axis=0) if all_t else np.array([], dtype=int)
    preds   = np.concatenate(all_p, axis=0) if all_p else np.array([], dtype=int)
    macro_f1 = _compute_macro_f1(targets, preds, n_cls=n_cls)
    f1_present = macro_f1_masked(targets, preds, n_cls, present_only=True)
    f1_supK    = macro_f1_masked(targets, preds, n_cls, present_only=False, min_support=max(0,int(min_support)))
    return tot_loss / tot, correct / tot, macro_f1, f1_present, f1_supK


def eval_metrics(model: nn.Module, dl: DataLoader, device: torch.device, num_classes: int, 
                 present_mask: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    sklearn 기반 보조 지표 산출
    """
    try:
        from sklearn.metrics import f1_score, precision_recall_fscore_support, balanced_accuracy_score, confusion_matrix
    except Exception:
        return {}
    model.eval()
    ys = []; preds = []
    with torch.no_grad():
        for xb, yb, _ in dl:
            xb = xb.to(device); yb = yb.to(device)
            logits = model(xb)
            # ===== 개선사항 2와 일관된 처리 =====
            if present_mask is not None:
                mask = torch.as_tensor(present_mask, device=logits.device, dtype=torch.bool)
                logits[:, ~mask] = logits[:, ~mask] - 1e4  # -1e9 대신 -1e4로 안정화
            pr = logits.argmax(1)
            ys.append(yb.cpu().numpy()); preds.append(pr.cpu().numpy())
    if not ys:
        return {}
    y = np.concatenate(ys); p = np.concatenate(preds)
    macro_f1 = f1_score(y, p, average="macro")
    micro_f1 = f1_score(y, p, average="micro")
    bal_acc = balanced_accuracy_score(y, p)
    prc, rc, f1c, _ = precision_recall_fscore_support(y, p, labels=list(range(num_classes)), zero_division=0)
    cm = confusion_matrix(y, p, labels=list(range(num_classes)))
    return dict(macro_f1=float(macro_f1), micro_f1=float(micro_f1), bal_acc=float(bal_acc),
                precision=prc.tolist(), recall=rc.tolist(), f1_per_cls=f1c.tolist(), cm=cm.tolist())


def save_ckpt(path: Path | str, model: nn.Module, opt: torch.optim.Optimizer,
              epoch: int, metrics: Dict[str, Any], cfg: TrainCfg,
              class_weights: torch.Tensor | None = None, extra_meta: Dict[str, Any] | None = None) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "epoch": int(epoch),
        "metrics": metrics,
        "cfg": cfg.as_dict() if hasattr(cfg, "as_dict") else dict(cfg),
        "class_weights": (class_weights.detach().cpu() if isinstance(class_weights, torch.Tensor) else None),
        "meta": (extra_meta or {}),
        "rng": {
            "py": random.getstate(),
            "np": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "torch_cuda": (torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
        }
    }, path)
    
    # NEW: metrics snapshot 별도 저장
    try:
        snap = dict(metrics=metrics, cfg=cfg.as_dict() if hasattr(cfg, "as_dict") else dict(cfg), meta=extra_meta or {})
        Path(path).with_suffix(".metrics.json").write_text(
            json.dumps(snap, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception:
        pass


# --- Torch 버전 호환 ReduceLROnPlateau 생성기 ---
def _make_plateau_scheduler(opt, mode: str = "max", debug: int = 0):
    import inspect
    kwargs = dict(mode=mode, factor=0.5, patience=2)
    sig = inspect.signature(torch.optim.lr_scheduler.ReduceLROnPlateau.__init__)
    if "verbose" in sig.parameters:
        kwargs["verbose"] = bool(debug > 0)
    return torch.optim.lr_scheduler.ReduceLROnPlateau(opt, **kwargs)


def _make_step_scheduler(opt, sched: str, max_lr: float, epochs: int, steps_per_epoch: int, min_lr: float):
    """
    onecycle / cosine 스케줄러 생성(미지정 시 None).
    """
    if sched == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=max_lr, epochs=int(epochs), steps_per_epoch=max(1, int(steps_per_epoch)))
    if sched == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(epochs * max(1, steps_per_epoch)), eta_min=float(min_lr))
    return None


def train_one(cfg: TrainCfg, sub_list: List[str],
              X: np.ndarray, y: np.ndarray, w: np.ndarray,
              save_path: Path,
              pre_split: tuple | None = None):
    """
    단일 메인 서브 분류기 학습.
    pre_split: (X_tr, y_tr, w_tr, X_va, y_va, w_va) 를 넘기면 내부 split을 생략하고 그대로 사용.
    - 샘플 가중치(w) + 클래스 가중치 동시 반영
    - 조기 종료(옵션화), Plateau 스케줄러(선택 지표 기반)
    - CUDA일 때 AMP 자동(on): torch.amp 우선, 구버전은 torch.cuda.amp 폴백
    - 학습 히스토리를 JSON으로 함께 저장
    """
    # ---------------- 개선된 초기화 ----------------
    # 로깅 시스템 설정
    logger = setup_logging(cfg.log_level, cfg.log_file)
    logger.info("=== Sub-Classifier Training Started ===")
    
    # 시스템 정보 로깅
    system_info = get_system_info()
    logger.info(f"System info: {system_info}")
    
    # 데이터 품질 검증
    quality_report = validate_data_quality(X, y, w, logger)
    if quality_report["quality_score"] < cfg.quality_threshold:
        logger.warning(f"Data quality score ({quality_report['quality_score']:.3f}) below threshold ({cfg.quality_threshold})")
    
    set_seed(cfg.seed)
    device = torch.device(getattr(cfg, "_device_resolved",
                                  "cuda" if torch.cuda.is_available() else "cpu"))
    
    # 디바이스 최적화
    device_settings = auto_optimize_device_settings(device, logger)
    
    in_dim = int(X.shape[1])
    n_cls  = len(sub_list)

    # ── (A) 사전 분할이 있으면 그대로 사용
    if pre_split is not None:
        (X_tr, y_tr, w_tr, X_va, y_va, w_va) = pre_split
    else:
        # ── (B) 아니면 기존처럼 내부 split
        tr_idx, va_idx = split_train_val(y, cfg.val_ratio, cfg.seed)
        X_tr, y_tr, w_tr = X[tr_idx], y[tr_idx], w[tr_idx]
        X_va, y_va, w_va = X[va_idx], y[va_idx], w[va_idx]

    # ===== 개선사항 2: 훈련 시 미등장 서브 로짓 마스킹 준비 =====
    present_mask_np = np.zeros(n_cls, dtype=bool)
    present_mask_np[sorted(set(y_tr.tolist()))] = True
    present_mask_tr = torch.as_tensor(present_mask_np, device=device, dtype=torch.bool)
    # ===== 개선사항 2 끝 =====

    # 11모듈 피처 수평결합 (있으면)
    # 주의: train_one 함수는 이미 조인된 X를 받으므로, 여기서는 추가 피처만 결합
    X_extra = None
    if hasattr(cfg, "session_dir") and cfg.session_dir:
        # 원본 ids가 필요하지만 train_one에서는 받지 않으므로, 임시로 인덱스 기반 처리
        try:
            # 전체 데이터에서 피처 로드 시도
            X_extra = load_module_features(cfg.session_dir, [f"idx_{i}" for i in range(len(X))])
            if X_extra is not None and X_extra.shape[0] == len(X):
                # 조인된 행만 남기도록 재결합
                X_tr_extra = X_extra[tr_idx] if len(tr_idx) < len(X) else X_extra
                X_va_extra = X_extra[va_idx] if len(va_idx) < len(X) else X_extra
                # NaN 방지 + float32
                X_tr_extra = np.nan_to_num(X_tr_extra, copy=False).astype(np.float32, copy=False)
                X_va_extra = np.nan_to_num(X_va_extra, copy=False).astype(np.float32, copy=False)
                # 수평결합
                X_tr = np.hstack([X_tr, X_tr_extra]).astype(np.float32, copy=False)
                X_va = np.hstack([X_va, X_va_extra]).astype(np.float32, copy=False)
                in_dim = int(X_tr.shape[1])  # 차원 업데이트
                print(f"[module-features] Loaded {X_extra.shape[1]} additional features, new dim: {in_dim}")
        except Exception as e:
            print(f"[module-features] Failed to load: {e}")

    # Dataset (정규화 방식 선택)
    normalize_mode = cfg.normalize if cfg.normalize != "none" else None
    ds_tr = SubDataset(X_tr, y_tr, w_tr, normalize=normalize_mode)
    ds_va = SubDataset(X_va, y_va, w_va, normalize=normalize_mode)
    
    # NEW: 표준화 파라미터 저장(있을 때)
    extra_meta = {"feature_norm": ds_tr.stats_}

    # 평가 설정 요약
    print(f"[eval] present_only={bool(getattr(cfg,'eval_present_only', False))} | min_support={int(getattr(cfg,'eval_min_support', 0) or 0)}")

    # Sampler (옵션) - 개선된 우선순위 적용
    sampler = None
    mode_cw = getattr(cfg, "class_weight_mode", "effective")
    # 클래스 가중치 먼저 산출
    class_w = compute_class_weights(y_tr, n_cls, mode=mode_cw, beta=float(getattr(cfg, "beta", 0.999))).to(device)
    
    # sampler 우선순위: ClassBalanced > WeightedRandom > shuffle
    if getattr(cfg, "use_weighted_sampler", False):
        sampler = make_weighted_sampler(y_tr, class_w.detach().cpu())
    elif (np.bincount(y_tr, minlength=n_cls) > 0).sum() > 1 and np.bincount(y_tr, minlength=n_cls).min() < 3:
        print(f"[auto-sampler] Rare class detected (min_count={np.bincount(y_tr, minlength=n_cls).min()}) → enabling class-balanced sampler")
        sampler = ClassBalancedBatchSampler(y_tr, batch_size=min(cfg.batch, max(1, len(ds_tr))))
    # ===== 개선사항 4 끝 =====

    # 손실 쪽 클래스 가중 사용 여부(샘플러 사용 시 과보정 방지)
    use_loss_class_weight = True
    if (sampler is not None) and (mode_cw in ("freq_inv", "effective", "cb-focal")):
        use_loss_class_weight = False  # ★ sampler on이면 loss class weight off

    # Dataloader
    # Windows에서는 multiprocessing 문제로 인해 num_workers=0으로 강제 설정
    is_windows = (os.name == "nt")
    if is_windows:
        # Windows 환경에서는 기본적으로 num_workers=0 (환경 변수로 오버라이드 가능)
        num_workers = os.getenv("SUB_DATALOADER_WORKERS")
        try:
            num_workers = int(num_workers) if num_workers is not None else 0
        except ValueError:
            num_workers = 0
        if cfg.debug and num_workers == 0:
            print("[dataloader] Windows detected → num_workers forced to 0 (set SUB_DATALOADER_WORKERS to override)")
    else:
        # Linux/Mac에서는 기존 로직 유지
        num_workers = max(0, min(32, (os.cpu_count() or 1) - 1))
        if device.type != "cuda":
            cpu_worker_cap = os.getenv("SUB_DATALOADER_CPU_WORKERS")
            try:
                cpu_worker_cap = int(cpu_worker_cap) if cpu_worker_cap is not None else 0
            except ValueError:
                cpu_worker_cap = 0
            num_workers = max(0, min(num_workers, cpu_worker_cap))
            if cpu_worker_cap == 0 and cfg.debug:
                print("[dataloader] CPU mode detected → num_workers forced to 0 (set SUB_DATALOADER_CPU_WORKERS to override)")
    pin = (device.type == "cuda")
    if cfg.debug and cfg.batch > len(ds_tr):
        print(f"[warn] batch({cfg.batch}) > train_size({len(ds_tr)}); single-batch per epoch.")

    dl_tr = DataLoader(ds_tr, batch_size=min(cfg.batch, max(1, len(ds_tr))),
                       shuffle=(sampler is None), sampler=sampler,
                       num_workers=num_workers, pin_memory=pin, persistent_workers=(num_workers > 0 and not is_windows))
    dl_va = DataLoader(ds_va, batch_size=min(cfg.batch, max(1, len(ds_va))),
                       shuffle=False, num_workers=num_workers,
                       pin_memory=pin, persistent_workers=False)

    # Model / loss / opt / scheduler
    model = MLPHead(in_dim, n_cls, hidden=512, drop=0.2, act="gelu", logit_scale=1.0).to(device)
    calibrator = TemperatureCalibrator(n_cls)
    
    # 자동 배치 크기 최적화
    if cfg.auto_optimize_batch_size:
        optimal_batch_size = optimize_batch_size(model, device, (in_dim,))
        if optimal_batch_size != cfg.batch:
            logger.info(f"Auto-optimized batch size: {cfg.batch} -> {optimal_batch_size}")
            print(f"[batch-optimize] Original batch={cfg.batch}, Optimized batch={optimal_batch_size}, Train size={len(ds_tr)}, Batches per epoch={len(dl_tr)}")
            cfg.batch = optimal_batch_size
            # DataLoader 재생성
            dl_tr = DataLoader(ds_tr, batch_size=min(cfg.batch, max(1, len(ds_tr))),
                             shuffle=(sampler is None), sampler=sampler,
                             num_workers=num_workers, pin_memory=pin, persistent_workers=(num_workers > 0 and not is_windows))
            dl_va = DataLoader(ds_va, batch_size=min(cfg.batch, max(1, len(ds_va))),
                             shuffle=False, num_workers=num_workers,
                             pin_memory=pin, persistent_workers=False)
        else:
            print(f"[batch-optimize] Using original batch size={cfg.batch}, Batches per epoch={len(dl_tr)}")
    
    # ---------------- 학습 루프 준비 ----------------
    # patience와 min_epochs를 함수 시작 부분에서 정의하여 예외 발생 시에도 안전하게 참조 가능하도록 함
    patience   = int(getattr(cfg, "early_stop_patience", 5) or 0)
    min_epochs = int(getattr(cfg, "min_epochs", max(3, min(10, cfg.epochs // 4))))
    
    # 학습 설정 요약 출력
    print(f"\n{'='*60}")
    print(f"[TRAINING CONFIG] main={cfg.main}")
    print(f"{'='*60}")
    print(f"  Model: MLPHead(input_dim={in_dim}, num_classes={n_cls})")
    print(f"  Training samples: {len(ds_tr)}")
    print(f"  Validation samples: {len(ds_va)}")
    print(f"  Batch size: {cfg.batch}")
    print(f"  Batches per epoch: train={len(dl_tr)}, val={len(ds_va)}")
    print(f"  Max epochs: {cfg.epochs}")
    print(f"  Learning rate: {cfg.lr}")
    print(f"  Early stop patience: {patience}")
    print(f"  Min epochs: {min_epochs}")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")
    
    # ★ warm-start: 가중치만 불러와 이어가기
    try:
        if getattr(cfg, "resume_from", None):
            sd = torch.load(str(Path(cfg.resume_from)), map_location="cpu")
            state = sd.get("state_dict") or sd.get("model") or sd
            if isinstance(state, dict):
                model.load_state_dict(state, strict=False)
                print(f"[resume] loaded weights from {cfg.resume_from}")
    except Exception as e:
        print(f"[resume][warn] failed to load: {e}")

    # Criterion 선택(항상 reduction='none' → per-sample 벡터)
    if mode_cw == "cb-focal":
        cw_for_focal = class_w if use_loss_class_weight else torch.ones_like(class_w)
        criterion = CBFocalLoss(cw_for_focal.to(device),
                                gamma=float(getattr(cfg, "focal_gamma", 2.0)),
                                label_smoothing=float(getattr(cfg, "label_smoothing", 0.0)),
                                reduction="none")
    else:
        use_w = class_w.to(device) if (use_loss_class_weight and mode_cw != "none") else None
        criterion = nn.CrossEntropyLoss(reduction="none", weight=use_w,
                                        label_smoothing=float(getattr(cfg, "label_smoothing", 0.0)))

    # 불균형/손실/샘플러 설정 요약
    print(f"[imbalance] mode={mode_cw} | sampler={'on' if sampler is not None else 'off'} | use_loss_cw={use_loss_class_weight} | focal_gamma={float(getattr(cfg, 'focal_gamma', 2.0)):.3g} | label_smoothing={float(getattr(cfg, 'label_smoothing', 0.0)):.3g}")
    if (sampler is not None) and (not use_loss_class_weight):
        print("[imbalance] weighted-sampler enabled -> disabled class weights in loss to avoid over-correction")

    opt = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # 스텝형 스케줄러(onecycle/cosine) 준비
    steps_per_epoch = max(1, len(dl_tr))
    sched = _make_step_scheduler(opt, getattr(cfg, "sched", "onecycle"), max_lr=float(cfg.lr), epochs=int(cfg.epochs), steps_per_epoch=steps_per_epoch, min_lr=float(getattr(cfg, "min_lr", 1e-5)))

    # AMP 설정
    device_cuda = (device.type == "cuda")
    amp_enabled = bool(getattr(cfg, "amp", False)) and device_cuda
    amp_dtype = (torch.bfloat16 if str(getattr(cfg, "amp_dtype", "fp16")).lower() == "bf16" else torch.float16)
    scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled)

    # ---------------- 학습 루프 ----------------
    history: List[Dict[str, float]] = []
    best_metric = -1e9  # maximize 기본
    best_acc    = 0.0
    best_epoch  = 0
    best_state  = None
    bad = 0

    metric_name = getattr(cfg, "early_stop_metric", "macro_f1").lower()
    grad_accum = max(1, int(getattr(cfg, "grad_accum", 1)))
    gc_thresh = float(getattr(cfg, "grad_clip", 0.0) or 0.0)

    hard_pairs: List[Tuple[int,int]] = []
    hard_boost = np.ones((n_cls,), dtype=np.float32)  # 클래스별 부스트(기본 1.0)
    
    # 학습 시간 측정 시작
    training_start_time = time.time()
    print(f"[training] Starting training at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    for ep in range(1, cfg.epochs + 1):   # ★ 들여쓰기 정상
        epoch_start_time = time.time()
        # 에포크 시작 시 메모리 모니터링
        if cfg.enable_memory_monitoring and ep % 5 == 1:
            log_memory_usage(logger, f"Epoch {ep} start")
            
        model.train()
        ep_loss = 0.0; seen = 0
        opt.zero_grad(set_to_none=True)

        # (옵션) hard_pairs 반영: 다음 epoch에 해당 클래스 표본이 조금 더 자주 보이도록
        if hard_pairs:
            for t, p in hard_pairs:
                hard_boost[t] = min(1.5, hard_boost[t] * 1.10)  # 타겟 클래스에 10%씩 상한 1.5
        class_w_eff = compute_class_weights(y_tr, n_cls, mode=mode_cw,
                                            beta=float(getattr(cfg, "beta", 0.999))).to(device)
        class_w_eff = class_w_eff * torch.from_numpy(hard_boost).to(device)
        if getattr(cfg, "use_weighted_sampler", False):
            sampler = make_weighted_sampler(y_tr, class_w_eff.detach().cpu())
            dl_tr = DataLoader(ds_tr, batch_size=min(cfg.batch, max(1, len(ds_tr))),
                               shuffle=False, sampler=sampler, num_workers=num_workers,
                               pin_memory=pin, persistent_workers=(num_workers > 0))

        for it, (xb, yb, wb) in enumerate(dl_tr, start=1):
            xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True); wb = wb.to(device, non_blocking=True)

            # === PATCH B: 임베딩 MixUp (로짓 계산 직전) ===
            mixup_alpha = float(getattr(cfg, "mixup_alpha", 0.0) or 0.0)
            use_mixup = (mixup_alpha > 0.0) and (xb.size(0) >= 2)
            if use_mixup:
                lam = np.random.beta(mixup_alpha, mixup_alpha)
                perm = torch.randperm(xb.size(0), device=xb.device)
                xb = lam * xb + (1 - lam) * xb[perm]
                yb2 = yb[perm]
            # === END PATCH B ===

            try:
                with torch.amp.autocast(device_type='cuda', enabled=amp_enabled, dtype=amp_dtype):
                    logits = model(xb)
                    
                    # ===== 개선사항 2: 훈련 시 미등장 서브 로짓 마스킹 (더 안정적인 값) =====
                    if present_mask_tr is not None:
                        logits[:, ~present_mask_tr] = logits[:, ~present_mask_tr] - 1e4  # -1e9 대신 -1e4로 안정화
                    # ===== 개선사항 2 끝 =====
                    
                    if use_mixup:
                        # CE/CB-Focal 공통 mixup 손실
                        loss_vec = lam * criterion(logits, yb) + (1 - lam) * criterion(logits, yb2)
                    else:
                        loss_vec = criterion(logits, yb)   # 항상 [B] 벡터
                    # 샘플 가중치 정규화(평균 1.0 근처)
                    wb_norm = wb * (wb.numel() / (wb.sum() + 1e-9))
                    loss = (loss_vec * wb_norm).mean() / grad_accum
                scaler.scale(loss).backward()
                if it % grad_accum == 0:
                    if gc_thresh > 0:
                        scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), gc_thresh)
                    scaler.step(opt); scaler.update()
                    opt.zero_grad(set_to_none=True)
                    if sched is not None:
                        try:
                            sched.step()
                        except ValueError as e:
                            # OneCycleLR가 총 스텝 수를 초과했을 때 (조기 종료로 인해)
                            if "total_steps" in str(e).lower() or "step" in str(e).lower():
                                if cfg.debug:
                                    print(f"[scheduler] Early stop detected, scheduler step skipped: {e}")
                                pass  # 조기 종료 시 스케줄러 에러 무시
                            else:
                                raise
            except RuntimeError as e:
                msg = str(e).lower()
                oom = ("out of memory" in msg) or ("cuda error" in msg)
                if amp_enabled and bool(getattr(cfg, "amp_fallback", False)) and oom:
                    print("[AMP] OOM → fallback to FP32 for the rest of epoch")
                    amp_enabled = False
                    scaler = torch.amp.GradScaler('cuda', enabled=False)
                    torch.cuda.empty_cache()
                    # FP32 재시도(같은 배치)
                    logits = model(xb)
                    loss_vec = criterion(logits, yb)   # [B]
                    wb_norm = wb * (wb.numel() / (wb.sum() + 1e-9))
                    loss = (loss_vec * wb_norm).mean() / grad_accum
                    loss.backward()
                    if it % grad_accum == 0:
                        if gc_thresh > 0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), gc_thresh)
                        opt.step(); opt.zero_grad(set_to_none=True)
                        if sched is not None:
                            try:
                                sched.step()
                            except ValueError as e:
                                # OneCycleLR가 총 스텝 수를 초과했을 때 (조기 종료로 인해)
                                if "total_steps" in str(e).lower() or "step" in str(e).lower():
                                    if cfg.debug:
                                        print(f"[scheduler] Early stop detected, scheduler step skipped: {e}")
                                    pass  # 조기 종료 시 스케줄러 에러 무시
                                else:
                                    raise
                else:
                    raise

            if not torch.isfinite(loss):
                if cfg.debug: print("[warn] non-finite loss; skip")
                continue

            ep_loss += float(loss.detach().item()) * xb.size(0)
            seen += xb.size(0)

            if cfg.debug and (it % max(1, int(cfg.log_interval)) == 0):
                cur_lr = opt.param_groups[0]["lr"]
                print(f"[iter {it:05d}] loss={float(loss):.4f} | lr={cur_lr:.6g}")

        # [PATCH 1] grad_accum 잔여 스텝 flush
        if (it % grad_accum) != 0:
            if gc_thresh > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gc_thresh)
            scaler.step(opt); scaler.update()
            opt.zero_grad(set_to_none=True)
            if sched is not None:
                sched.step()

        # Validate
        present_mask = np.zeros(n_cls, dtype=bool)
        present_mask[sorted(set(y_tr.tolist()))] = True
        va_loss, va_acc, va_f1, va_f1_present, va_f1_supK = evaluate(
            model, dl_va, device, n_cls=n_cls,
            present_only=bool(getattr(cfg, "eval_present_only", False)),
            min_support=int(getattr(cfg, "eval_min_support", 0) or 0),
            present_mask=present_mask                                  # ★ 추가
        )

        tr_loss_mean = ep_loss / max(1, seen)
        
        # 에포크 소요 시간 계산
        epoch_elapsed = time.time() - epoch_start_time
        
        # ===== 개선사항 5: 가시성 개선을 위한 포맷 변경 =====
        epoch_metrics = {
            "epoch": ep,
            "train_loss": tr_loss_mean,
            "val_loss": va_loss,
            "val_acc": va_acc,
            "val_macro_f1": va_f1,
            "val_present_f1": va_f1_present,
            "val_supK_f1": va_f1_supK
        }
        
        logger.info(f"[{save_path.stem}] ep {ep:02d}/{cfg.epochs} ({epoch_elapsed:.2f}s) "
                   f"| train_loss={tr_loss_mean:.6f} "
                   f"| val_loss={va_loss:.6f} | val_acc={va_acc:.4f} "
                   f"| val_macro_f1={va_f1:.4f} | val_present_f1={va_f1_present:.4f} | val_supK_f1={va_f1_supK:.4f}")
        
        # 성능 메트릭 로깅
        if cfg.enable_performance_monitoring and ep % 5 == 0:
            log_performance_metrics(logger, epoch_metrics)

        # 보조 지표 수집
        aux_metrics = eval_metrics(model, dl_va, device, n_cls, 
                                 present_mask=np.isin(range(n_cls), sorted(set(y.tolist()))))
        
        # hard-example mining: 다음 epoch에서 가중치 미세 조정
        try:
            y_all = np.concatenate([b[1].cpu().numpy() for b in dl_va], axis=0)
            with torch.no_grad():
                preds = []
                for xb, yb, _ in dl_va:
                    pr = model(xb.to(device)).argmax(1).cpu().numpy()
                    preds.append(pr)
            p_all = np.concatenate(preds, axis=0)
            hard_pairs = mine_hard_examples(y_all, p_all, topk=3)
        except Exception:
            hard_pairs = []
        cur_lr = opt.param_groups[0]["lr"]
        history.append({
            "epoch": ep, "train_loss": tr_loss_mean,
            "val_loss": float(va_loss), "val_acc": float(va_acc),
            "val_macro_f1": float(va_f1),
            "val_present_f1": float(va_f1_present),
            "val_supK_f1": float(va_f1_supK),
            "lr": float(cur_lr),
            **({} if not aux_metrics else {"aux": aux_metrics}),
            "metric_used": metric_name,
        })

        # 스케줄러(Plateau 모드 사용시) — 선택적: 유지하고 싶으면 아래 주석 해제
        # sched_metric = va_loss if metric_name == "loss" else (va_f1_present if metric_name == "present_f1" else va_f1)
        # if isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau):
        #     sched.step(sched_metric)

        # Save best/last
        sched_metric = va_loss if metric_name == "loss" else (va_f1_present if metric_name == "present_f1" else va_f1)
        current = sched_metric if metric_name != "loss" else -sched_metric  # loss는 작을수록 좋음
        improved = (current > best_metric + 1e-6) or (abs(current - best_metric) <= 1e-6 and va_acc > best_acc)

        last_metrics = {
            "train_loss": tr_loss_mean, "val_loss": float(va_loss), "val_acc": float(va_acc),
            "val_macro_f1": float(va_f1), "val_present_f1": float(va_f1_present), "val_supK_f1": float(va_f1_supK),
            **({} if not aux_metrics else aux_metrics)
        }
        save_ckpt(Path(cfg.outdir) / f"sub_{cfg.main}_last.pt", model, opt, ep, last_metrics, cfg, class_weights=class_w)

        if improved:
            best_metric = current
            best_acc    = float(va_acc)
            best_epoch  = ep
            best_state  = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            # per-class T 적합(검증 로짓 수집)
            try:
                logits_val = []
                ys_val = []
                with torch.no_grad():
                    for xb, yb, _ in dl_va:
                        lg = model(xb.to(device)).cpu().numpy()
                        logits_val.append(lg)
                        ys_val.append(yb.numpy())
                L = np.vstack(logits_val)
                Y = np.concatenate(ys_val)
                calibrator.fit(L, Y, max_iter=100)
            except Exception:
                pass
            
            extra_meta.update({"present_mask": present_mask_np.tolist(), "per_class_T": calibrator.T.tolist()})
            save_ckpt(Path(cfg.outdir) / f"sub_{cfg.main}_best.pt", model, opt, ep, last_metrics, cfg, 
                     class_weights=class_w, extra_meta=extra_meta)

        # 조기 종료(최소 에포크 보장)
        if patience > 0 and ep >= min_epochs and not improved:
            bad += 1
            if bad >= patience:
                if cfg.debug:
                    print(f"[early-stop] no improvement {patience} epochs (min_epochs={min_epochs}, best_epoch={best_epoch}).")
                break
        else:
            bad = 0

    # 학습 완료 시간 측정
    training_elapsed = time.time() - training_start_time
    training_minutes = int(training_elapsed // 60)
    training_seconds = int(training_elapsed % 60)
    print(f"\n{'='*60}")
    print(f"[TRAINING COMPLETE] main={cfg.main}")
    print(f"{'='*60}")
    print(f"  Total training time: {training_minutes}m {training_seconds}s ({training_elapsed:.2f}s)")
    print(f"  Epochs completed: {ep}/{cfg.epochs}")
    print(f"  Best epoch: {best_epoch}")
    print(f"  Best metric ({metric_name}): {best_metric:.6f}")
    print(f"  Best accuracy: {best_acc:.4f}")
    if ep < cfg.epochs:
        print(f"  Early stopped: Yes (patience={patience})")
    else:
        print(f"  Early stopped: No (completed all epochs)")
    print(f"{'='*60}\n")

    # === PATCH C: 검증셋으로 로짓 스케일 캘리브레이션 ===
    def _fit_logit_scale(model, dl, device):
        """간단 LBFGS로 scalar s를 학습(softmax(logits*s))"""
        # ===== 개선사항 3: 스칼라 텐서로 수정 =====
        s = torch.tensor(1.0, device=device, requires_grad=True)  # 스칼라
        opt = torch.optim.LBFGS([s], lr=0.1, max_iter=50, line_search_fn="strong_wolfe")
        ce = nn.CrossEntropyLoss(reduction="mean")

        def _closure():
            opt.zero_grad(set_to_none=True)
            tot = 0.0; n = 0
            for xb, yb, _ in dl:
                xb = xb.to(device); yb = yb.to(device)
                logits = model(xb) * s.clamp(0.25, 4.0)
                loss = ce(logits, yb)
                tot = tot + loss; n += 1
                if n >= 32: break  # 너무 오래 안 걸리게 일부만 사용
            (tot / max(1,n)).backward()
            return tot / max(1,n)

        try:
            opt.step(_closure)
            with torch.no_grad():
                model._logit_scale.copy_(s.clamp(0.25, 4.0).detach())
        except Exception as e:
            print(f"[calib][warn] logit-scale fit failed: {e}")

    # … 에폭 루프 종료 후, best_state 로드해서 검증로더(dl_va)로 캘리브레이션
    if best_state is None:
        best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    else:
        model.load_state_dict(best_state, strict=False)
    
    # 캘리브레이션
    _fit_logit_scale(model, dl_va, device)
    best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    # === END PATCH C ===

    # ---------------- 저장 ----------------
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if best_state is None:
        best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    best_f1_hist = max((h.get("val_macro_f1", 0.0) for h in history), default=0.0)
    # === 존재하지 않는 서브 마스크 계산 ===
    present_idx = sorted(set(y.tolist()))  # 이번 학습 셋에서 실제 등장한 서브 index
    
    meta = {
        "sub_list": sub_list,
        "in_dim": in_dim,
        "best_epoch": int(best_epoch),
         "best_macro_f1": float(best_f1_hist),  # ← history의 최고값으로 교체
        "best_val_acc": float(best_acc),
        "present_idx": present_idx,              # ★ 추가: 존재 서브 인덱스
        "cfg": {
            "batch": cfg.batch, "epochs": cfg.epochs, "lr": cfg.lr,
            "val_ratio": cfg.val_ratio, "min_conf": cfg.min_conf,
            "exclude_fallback": cfg.exclude_fallback,
            "early_stop_patience": int(patience),
            "min_epochs": int(min_epochs),
            "early_stop_metric": metric_name,
            "class_weight_mode": getattr(cfg, "class_weight_mode","effective"),
            "beta": float(getattr(cfg, "beta", 0.999)),
            "use_weighted_sampler": bool(getattr(cfg,"use_weighted_sampler",False)),
            "label_smoothing": float(getattr(cfg, "label_smoothing", 0.0)),
            "focal_gamma": float(getattr(cfg, "focal_gamma", 2.0)),
            "sched": str(getattr(cfg, "sched", "onecycle")),
            "min_lr": float(getattr(cfg, "min_lr", 1e-5)),
            "grad_accum": int(getattr(cfg, "grad_accum", 1)),
            "amp": bool(getattr(cfg, "amp", False)),
            "amp_dtype": str(getattr(cfg, "amp_dtype", "fp16")),
            "amp_fallback": bool(getattr(cfg, "amp_fallback", False)),
            "eval_present_only": bool(getattr(cfg, "eval_present_only", False)),
            "eval_min_support": int(getattr(cfg, "eval_min_support", 0) or 0),
        }
    }
    # 최종 모델 저장 (원자적 저장)
    try:
        tmp_path = save_path.with_suffix(".tmp")
        torch.save({"state_dict": best_state, "meta": meta}, tmp_path)
        # 원자적 교체
        tmp_path.replace(save_path)
        file_size_mb = save_path.stat().st_size / (1024 * 1024)
        print(f"\n{'='*60}")
        print(f"[MODEL SAVED] main={cfg.main}")
        print(f"{'='*60}")
        print(f"  File: {save_path.name}")
        print(f"  Size: {file_size_mb:.2f} MB")
        print(f"  Training samples used: {len(ds_tr)}")
        print(f"  Validation samples used: {len(ds_va)}")
        print(f"  Classes present: {len(present_idx)}/{len(sub_list)}")
        print(f"  Best epoch: {best_epoch}/{ep}")
        print(f"  Best F1: {best_f1_hist:.4f}")
        print(f"  Best accuracy: {best_acc:.4f}")
        print(f"  Total training time: {training_minutes}m {training_seconds}s")
        print(f"{'='*60}\n")
        print(f"[✓ 저장 완료] {save_path.name} ({file_size_mb:.2f} MB) | present_idx={len(present_idx)}/{len(sub_list)} | best_epoch={best_epoch} | best_f1={best_f1_hist:.4f}")
    except Exception as e:
        print(f"[✗ 저장 실패] {save_path}: {e}")
        raise

    # 히스토리도 옆에 저장
    if getattr(cfg, "save_history", True):
        hist_path = save_path.with_suffix(".history.json")
        hist_path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")



# --------------------
# 4) 엔드포인트 (robust & diagnostics)
# --------------------
def _attach_context_if_available(cfg: TrainCfg, ids: List[str], X: np.ndarray) -> np.ndarray:
    """
    선택적 컨텍스트 피처 결합 훅.
    - load_module_features(session_dir:str, ids:List[str]) -> np.ndarray[N, F]
      가 전역에 정의되어 있으면 자동 호출하여 X에 np.hstack.
    - 정의/호출 실패 시 경고만 출력하고 계속 진행.
    """
    if not getattr(cfg, "session_dir", None):
        return X
    fn = globals().get("load_module_features", None)
    if not callable(fn):
        if cfg.debug:
            print("[ctx] load_module_features not found; skipping context attach.")
        return X
    try:
        C = fn(cfg.session_dir, ids)  # 기대형태: np.ndarray [N, F]
        if isinstance(C, np.ndarray) and C.ndim == 2 and C.shape[0] == len(ids) and C.size > 0:
            C = C.astype(np.float32, copy=False)
            if not isinstance(X, np.ndarray) or X.ndim != 2 or X.shape[0] != len(ids):
                print(f"[ctx][warn] base X shape invalid: {None if not isinstance(X,np.ndarray) else X.shape}")
                return X
            X2 = np.hstack([X, C])
            print(f"[ctx] attached context features: {C.shape[1]} dims -> X {X.shape} -> {X2.shape}")
            return X2
        else:
            if cfg.debug:
                print(f"[ctx] unexpected context shape: {None if not isinstance(C,np.ndarray) else C.shape}; skip.")
            return X
    except Exception as e:
        print(f"[ctx][warn] context attach failed: {e}")
        return X


def _save_join_summary(out_dir: Path,
                       main: str,
                       ids_total: int,
                       hit: int,
                       sub_list: List[str],
                       yj: np.ndarray) -> None:
    """조인/클래스 분포를 요약 파일로 저장."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cnt = Counter(yj.tolist()) if yj is not None and yj.size else {}
    by_class = {sub_list[i]: int(cnt.get(i, 0)) for i in range(len(sub_list))}
    # [PATCH 3] 희귀 서브 목록 계산
    rare_th = max(3, int(0.005 * max(1, ids_total)))
    rare_subs = [sub_list[i] for i, c in cnt.items() if c > 0 and c < rare_th]
    
    info = {
        "main": main,
        "ids_total": ids_total,
        "hit": hit,
        "hit_ratio": float(hit / max(1, ids_total)),
        "class_counts": by_class,
        "n_classes_present": int(sum(v > 0 for v in cnt.values())),
        "rare_threshold": int(rare_th),
        "rare_subs": rare_subs,
    }
    (out_dir / f"join_summary_{main}.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def train_for_main(cfg: TrainCfg):
    """
    단일 메인(예: '노')의 서브 분류기 학습 엔드포인트.
    - EMOTIONS에서 서브 라벨 로딩
    - 임베딩/라벨 로딩 & 조인
    - (있다면) 컨텍스트 피처 결합
    - (부족 시) 자동 완화 시도
    - 학습 및 저장, 요약 JSON 기록
    """
    from copy import deepcopy

    cfg = deepcopy(cfg)                 # 호출 간 설정 오염 방지
    cfg.validate_and_prepare()          # outdir/device/seed 등 정리

    if not cfg.main:
        raise ValueError("cfg.main must be set ('희'|'노'|'애'|'락').")

    emo_map = load_emotions_map(cfg.emotions_json)
    if cfg.main not in emo_map:
        print(f"[skip] main={cfg.main} not in EMOTIONS.JSON")
        return

    sub_list = emo_map.get(cfg.main, [])
    if not sub_list:
        print(f"[skip] main={cfg.main} has no sub emotions in EMOTIONS.JSON")
        return

    # 서브 동의어/변형 매핑(조인율 개선)
    sub_alias = build_sub_alias_map(sub_list)

    print(f"[info] main={cfg.main} | sub_count={len(sub_list)}")

    # 1) 임베딩 로드
    X, ids = load_embeddings_jsonl(cfg.emb_jsonl, cfg.id_field, cfg.emb_field)
    # 컨텍스트 피처는 아래 _try_join 단계에서 조인된 행 기준으로 1회만 결합

    # 하한 계산
    need_join = cfg.effective_min_join(len(sub_list))
    need_classes = max(1, int(getattr(cfg, "min_classes", 2)))

    # --- 조인 함수 ---
    # ===== 개선사항 4: 데이터 안정화 권장사항 =====
    # 조인 수가 적을 경우 다음 설정을 권장:
    # 1) --min-conf 0.40 --include-fallback (폴백 포함하여 데이터 늘리기)
    # 2) --eval-min-support 5 (극소 표본 클래스 필터링)
    # 3) --normalize l2 (정규화 유지)
    # 4) 데이터가 매우 적은 경우 --kfold 5 (K-Fold 교차검증)
    # ===== 개선사항 4 끝 =====
    def _try_join(min_conf: float, exclude_fallback: bool, tag: str):
        y_map, w_map = load_labels_jsonl(
            cfg.labels_jsonl,
            min_conf=min_conf,
            exclude_fallback=exclude_fallback,
            target_main=cfg.main,
            conf_power=getattr(cfg, "conf_power", 2.0),
            sub_alias_map=sub_alias,
        )
        # 디버그: 조인 전 겹침 수 확인
        if cfg.debug:
            emb_set = set(ids)
            lab_set = set(y_map.keys())
            print(f"[join/diag] emb_ids={len(emb_set)} lab_ids={len(lab_set)} overlap={len(emb_set & lab_set)}")
        # 11모듈 피처 수평결합 (있으면)
        X_extra = None
        if hasattr(cfg, "session_dir") and cfg.session_dir:
            X_extra = load_module_features(cfg.session_dir, ids)
            if X_extra is None:
                # 피처 파일을 찾지 못한 경우 경로 확인
                print(f"[module-features] Not found in session_dir={cfg.session_dir}, checking default paths...")
                # 기본 경로도 확인
                X_extra = load_module_features(None, ids)
        
        if X_extra is not None and X_extra.shape[0] == len(ids):
            # 조인된 행만 남기도록 join 후 재결합
            Xj, yj, wj, hit = join_xy(X, ids, y_map, w_map, sub_list)
            Xej = X_extra[[ids.index(i) for i in ids if i in y_map]]
            # NaN 방지 + float32
            Xej = np.nan_to_num(Xej, copy=False).astype(np.float32, copy=False)
            Xj = np.hstack([Xj, Xej]).astype(np.float32, copy=False)
            print(f"[module-features] ✓ Loaded {X_extra.shape[1]} additional features, new dim: {Xj.shape[1]} (was {X.shape[1]})")
        else:
            Xj, yj, wj, hit = join_xy(X, ids, y_map, w_map, sub_list)
            if X_extra is None:
                print(f"[module-features] ✗ Module features not loaded (file not found or shape mismatch)")
            elif X_extra.shape[0] != len(ids):
                print(f"[module-features] ✗ Shape mismatch: features={X_extra.shape[0]}, ids={len(ids)}")
        classes_present = len(set(yj.tolist()))
        print(f"[join/{tag}] main={cfg.main} | hit={hit}/{len(ids)} ({hit/len(ids)*100:.1f}%) | X={Xj.shape} | "
              f"classes_present={classes_present}/{len(sub_list)} | "
              f"min_conf={min_conf} | exclude_fallback={exclude_fallback}")
        print(f"[join/{tag}] Training samples: train={int(hit*(1-cfg.val_ratio))}, val={int(hit*cfg.val_ratio)}")
        return Xj, yj, wj, hit, classes_present, dict(y_map)

    # 2) 1차 시도(기본 필터)
    best = None
    try:
        base = _try_join(cfg.min_conf, cfg.exclude_fallback, tag="base")
        best = ("base",) + base  # (tag, Xj, yj, wj, hit, classes_present, y_map_used)
    except Exception as e:
        if cfg.debug:
            print(f"[warn] base join failed: {e}")

    # 3) (부족 시) 자동 완화 시도(2단계)
    need_retry = (best is None)
    if not need_retry:
        _, _, _, _, hit_b, cls_b, _ = best
        if (cls_b < need_classes) or (hit_b < need_join):
            need_retry = True

    if need_retry:
        # relax-1: min_conf ↓, fallback 포함
        try_min_conf = max(0.0, cfg.min_conf - 0.2)
        try:
            relax = _try_join(try_min_conf, False, tag="relax")
            if best is None:
                best = ("relax",) + relax
            else:
                _, _, _, _, hit_b, cls_b, _ = best
                _, Xr, yr, wr, hit_r, cls_r, _ = ("relax",) + relax
                if (cls_r > cls_b) or (cls_r == cls_b and hit_r > hit_b):
                    best = ("relax",) + relax
        except Exception as e:
            if cfg.debug:
                print(f"[warn] relax join failed: {e}")

    # relax-2(여전히 모자라면) : min_conf 더 내림
    if best is not None:
        _, _, _, _, hit_b, cls_b, _ = best
        if (cls_b < need_classes) or (hit_b < need_join):
            try:
                relax2 = _try_join(max(0.0, (cfg.min_conf - 0.3)), False, tag="relax2")
                _, _, _, _, hit_r, cls_r, _ = ("relax2",) + relax2
                if (cls_r > cls_b) or (cls_r == cls_b and hit_r > hit_b):
                    best = ("relax2",) + relax2
            except Exception as e:
                if cfg.debug:
                    print(f"[warn] relax2 join failed: {e}")

    if best is None:
        print(f"[error] main={cfg.main}: No joined samples at all.")
        return

    # ===== 개선사항 1: 조인 선택 로직 - 작은 셋에서 relax 우선 채택 =====
    PREFER_RELAX_HIT = int(os.getenv("PREFER_RELAX_HIT", "30"))  # 임계 기본 30
    if best is not None:
        tag_b, Xb, yb, wb, hit_b, cls_b, _ = best
        # relax 재시도 결과가 있으면 비교
        for cand in ("relax2", "relax"):
            try:
                # cand 변수가 채워져 있을 때만 비교
                if 'relax' in locals() and cand == "relax":
                    _, Xr, yr, wr, hit_r, cls_r, _ = ("relax",) + relax
                    if hit_b < PREFER_RELAX_HIT and ((cls_r > cls_b) or (cls_r == cls_b and hit_r > hit_b)):
                        best = (cand, Xr, yr, wr, hit_r, cls_r, _)  # relax 쪽으로 승격
                        tag_b, Xb, yb, wb, hit_b, cls_b, _ = best
                if 'relax2' in locals() and cand == "relax2":
                    _, Xr, yr, wr, hit_r, cls_r, _ = ("relax2",) + relax2
                    if hit_b < PREFER_RELAX_HIT and ((cls_r > cls_b) or (cls_r == cls_b and hit_r > hit_b)):
                        best = (cand, Xr, yr, wr, hit_r, cls_r, _)  # relax 쪽으로 승격
                        tag_b, Xb, yb, wb, hit_b, cls_b, _ = best
            except Exception:
                pass
    # ===== 개선사항 1 끝 =====

    tag_used, Xj, yj, wj, hit, classes_present, ymap_used = best
    
    # 학습 시작 전 데이터 사용량 요약
    train_size = int(hit * (1 - cfg.val_ratio))
    val_size = int(hit * cfg.val_ratio)
    print(f"\n{'='*60}")
    print(f"[DATA SUMMARY] main={cfg.main}")
    print(f"{'='*60}")
    print(f"  Total embeddings loaded: {len(ids)}")
    print(f"  Labels matched: {hit} ({hit/len(ids)*100:.1f}%)")
    print(f"  Training samples: {train_size}")
    print(f"  Validation samples: {val_size}")
    print(f"  Feature dimension: {Xj.shape[1]}")
    print(f"  Classes present: {classes_present}/{len(sub_list)}")
    print(f"  Filter used: {tag_used}")
    print(f"{'='*60}\n")
    
    print(f"[use] main={cfg.main} | filter='{tag_used}' | hit={hit} | classes_present={classes_present}")

    # 최소 조건 확인
    warn_only = False
    if classes_present < need_classes:
        if classes_present == 1 and hit >= need_join:
            print(f"[warn] main={cfg.main}: classes_present=1, but hit={hit}>=min_join({need_join}). Proceed anyway.")
            warn_only = True
        else:
            print(f"[skip] main={cfg.main}: classes_present={classes_present} < min_classes={need_classes}.")
            _save_join_summary(Path(cfg.outdir), cfg.main, len(ids), hit, sub_list, yj)
            (Path(cfg.outdir) / f"join_filter_{cfg.main}.json").write_text(
                json.dumps({"main": cfg.main, "filter_used": tag_used,
                            "min_classes": int(need_classes), "classes_present": int(classes_present),
                            "min_join": int(need_join), "hit": int(hit)}, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
            return

    # ===== 개선사항 3: 극소표본 자동 진입 가드 (이미 구현되어 있음, 확인) =====
    if hit < need_join and not warn_only:
        small = int(os.getenv("MINI_JOIN", "10"))  # 기본 10
        if hit >= small:
            print(f"[warn] main={cfg.main}: hit={hit} < need_join({need_join}) → small-sample mode(proceed, thr={small})")
        else:
            print(f"[skip] main={cfg.main}: joined samples too small (hit={hit} < {need_join}, thr={small}).")
            _save_join_summary(Path(cfg.outdir), cfg.main, len(ids), hit, sub_list, yj)
            (Path(cfg.outdir) / f"join_filter_{cfg.main}.json").write_text(
                json.dumps({"main": cfg.main, "filter_used": tag_used,
                            "min_classes": int(need_classes), "classes_present": int(classes_present),
                            "min_join": int(need_join), "hit": int(hit)}, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
            return

    # 4) 학습
    out_model = Path(cfg.outdir) / f"sub_{cfg.main}.pt"
    train_one(cfg, sub_list, Xj, yj, wj, out_model)

    # 5) 요약 저장(추가 정보 포함)
    _save_join_summary(Path(cfg.outdir), cfg.main, len(ids), hit, sub_list, yj)
    extra = {
        "main": cfg.main,
        "filter_used": tag_used,
        "min_conf_used": (cfg.min_conf if tag_used == "base" else max(0.0, cfg.min_conf - 0.2)),
        "exclude_fallback_used": (cfg.exclude_fallback if tag_used == "base" else False),
        "classes_present": int(classes_present),
        "hit": int(hit),
        "need_join": int(need_join),
        "need_classes": int(need_classes),
    }
    (Path(cfg.outdir) / f"join_filter_{cfg.main}.json").write_text(
        json.dumps(extra, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def train_kfold(cfg: TrainCfg, sub_list: List[str],
                X: np.ndarray, y: np.ndarray, w: np.ndarray, ids: List[str]):
    """K-Fold OOF 확률 내보내기"""
    if cfg.kfold <= 1 or not cfg.export_oof:
        return
    kf = StratifiedKFold(n_splits=int(cfg.kfold), shuffle=True, random_state=int(cfg.seed))
    oof = np.zeros((len(y), len(sub_list)), dtype=np.float32)
    for fi, (tr, va) in enumerate(kf.split(X, y), start=1):
        pre = (X[tr], y[tr], w[tr], X[va], y[va], w[va])
        out_dir = Path(cfg.outdir) / f"fold{fi:02d}"
        train_one(cfg, sub_list, X, y, w, save_path=out_dir/"best.pt", pre_split=pre)
        # infer on val
        dev = torch.device(cfg._device_resolved)
        mdl = MLPHead(X.shape[1], len(sub_list))
        sd = torch.load(str(out_dir/"best.pt"), map_location="cpu")
        state = sd.get("state_dict") if isinstance(sd, dict) else None
        if state is None and isinstance(sd, dict):
            state = sd.get("model")
        if isinstance(state, dict):
            mdl.load_state_dict(state, strict=False)
        else:
            # 호환성: 파일 자체가 state_dict인 경우
            mdl.load_state_dict(sd if isinstance(sd, dict) else sd, strict=False)
        mdl.to(dev).eval()
        with torch.no_grad():
            for st in range(0, len(va), 8192):
                chunk = torch.from_numpy(X[va][st:st+8192]).to(dev)
                oof[va][st:st+8192] = torch.softmax(mdl(chunk), dim=1).cpu().numpy()
    np.savez(cfg.export_oof, probs=oof, y=y, ids=np.array(ids))


def train_kfold_for_main(cfg: TrainCfg, k: int = 5):
    cfg = deepcopy(cfg); cfg.validate_and_prepare()
    assert cfg.main, "cfg.main required"

    emo_map  = load_emotions_map(cfg.emotions_json)
    sub_list = emo_map.get(cfg.main, [])
    if not sub_list:
        print(f"[skip] main={cfg.main} has no sub emotions in EMOTIONS.JSON"); return

    X, ids = load_embeddings_jsonl(cfg.emb_jsonl, cfg.id_field, cfg.emb_field)
    sub_alias = build_sub_alias_map(sub_list)
    y_map, w_map = load_labels_jsonl(cfg.labels_jsonl,
                                     min_conf=cfg.min_conf,
                                     exclude_fallback=cfg.exclude_fallback,
                                     target_main=cfg.main,
                                     conf_power=cfg.conf_power,
                                     sub_alias_map=sub_alias)
    Xj, yj, wj, hit = join_xy(X, ids, y_map, w_map, sub_list)
    print(f"[kfold/join] main={cfg.main} | hit={hit}/{len(ids)} | X={Xj.shape}")

    skf = StratifiedKFold(n_splits=int(k), shuffle=True, random_state=cfg.seed)
    fold_paths = []
    for f, (tr_idx, va_idx) in enumerate(skf.split(Xj, yj), start=1):
        sub_cfg = deepcopy(cfg); sub_cfg._prepared = True
        sub_cfg.outdir = str(Path(cfg.outdir) / f"fold{f}")
        Path(sub_cfg.outdir).mkdir(parents=True, exist_ok=True)

        X_tr, y_tr, w_tr = Xj[tr_idx], yj[tr_idx], wj[tr_idx]
        X_va, y_va, w_va = Xj[va_idx], yj[va_idx], wj[va_idx]

        save_path = Path(sub_cfg.outdir) / f"sub_{cfg.main}.pt"
        # ★ 핵심: pre_split로 fold 분할을 그대로 전달
        train_one(sub_cfg, sub_list, Xj, yj, wj, save_path,
                  pre_split=(X_tr, y_tr, w_tr, X_va, y_va, w_va))
        fold_paths.append(str(save_path))

    meta = {"main": cfg.main, "k": int(k), "folds": fold_paths}
    (Path(cfg.outdir) / f"sub_{cfg.main}_ensemble.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[kfold] saved ensemble meta -> {Path(cfg.outdir) / f'sub_{cfg.main}_ensemble.json'}")

    # 선택: 폴드 중 최고의 모델을 루트에 복사하여 호환성 유지 (sub_{main}.pt)
    try:
        best_score = -1e9
        best_dir: Path | None = None
        for fi in range(1, int(k) + 1):
            out_dir = Path(cfg.outdir) / f"fold{fi:02d}"
            cand_score = None
            # 1) metrics 스냅샷 우선 사용
            metrics_path = out_dir / f"sub_{cfg.main}_best.pt.metrics.json"
            if metrics_path.exists():
                try:
                    snap = json.loads(metrics_path.read_text(encoding="utf-8", errors="ignore"))
                    cand_score = float(((snap.get("metrics") or {}).get("val_macro_f1")) or 0.0)
                except Exception:
                    cand_score = None
            # 2) 없으면 history에서 최고 F1 사용
            if cand_score is None:
                hist_path = out_dir / "best.pt.history.json"
                if hist_path.exists():
                    try:
                        hist = json.loads(hist_path.read_text(encoding="utf-8", errors="ignore"))
                        if isinstance(hist, list) and hist:
                            cand_score = max(float(h.get("val_macro_f1", 0.0)) for h in hist if isinstance(h, dict))
                    except Exception:
                        cand_score = None
            if cand_score is None:
                cand_score = 0.0
            if float(cand_score) > best_score:
                best_score = float(cand_score)
                best_dir = out_dir
        if best_dir is not None:
            src = best_dir / "best.pt"
            dst = Path(cfg.outdir) / f"sub_{cfg.main}.pt"
            if src.exists():
                shutil.copyfile(src, dst)
                (Path(cfg.outdir) / f"sub_{cfg.main}_kfold_best.json").write_text(
                    json.dumps({
                        "main": cfg.main,
                        "k": int(k),
                        "selected_fold": best_dir.name,
                        "val_macro_f1": float(best_score)
                    }, ensure_ascii=False, indent=2),
                    encoding="utf-8"
                )
                print(f"[kfold] best fold selected -> {best_dir.name}; copied to {dst.name}")
            else:
                print(f"[kfold][warn] best.pt not found under {best_dir}")
    except Exception as e:
        print(f"[kfold][warn] failed to select/copy best fold: {e}")



def train_all_mains(cfg: TrainCfg):
    from copy import deepcopy
    cfg = deepcopy(cfg)
    cfg.validate_and_prepare()
    run_root = Path(cfg.outdir)
    run_root.mkdir(parents=True, exist_ok=True)

    emo_map = load_emotions_map(cfg.emotions_json)
    mains_order = [m for m in ("희","노","애","락") if m in emo_map]

    summary = []
    failed_mains = []
    for main in mains_order:
        sub_cfg = deepcopy(cfg)
        sub_cfg.main = main
        sub_cfg.outdir = str(run_root)
        sub_cfg._prepared = True  # 이미 validate됨

        print(f"\n{'='*60}")
        print(f"=== [main={main}] 학습 시작 ===")
        print(f"{'='*60}")
        try:
            before = set(os.listdir(run_root))
            k_val = int(getattr(sub_cfg, "kfold", 0) or 0)
            if k_val > 1:
                train_kfold_for_main(sub_cfg, k=k_val)
            else:
                train_for_main(sub_cfg)
            after = set(os.listdir(run_root))
            
            # 모델 파일 생성 검증 강화
            model_file = f"sub_{main}.pt"
            model_path = run_root / model_file
            ensemble_meta = run_root / f"sub_{main}_ensemble.json"
            made = (model_path.exists() and model_path.stat().st_size > 0) or ensemble_meta.exists()
            
            if made:
                file_size_mb = (model_path.stat().st_size / (1024 * 1024)) if model_path.exists() else 0.0
                print(f"[✓ 완료] {main} 모델 저장됨: {model_file if model_path.exists() else ensemble_meta.name} ({file_size_mb:.2f} MB)")
                summary.append({"main": main, "status": "ok", "model_file": (model_file if model_path.exists() else ensemble_meta.name), "size_mb": round(file_size_mb, 2)})
            else:
                print(f"[✗ 실패] {main} 모델 파일이 생성되지 않았습니다: {model_file}")
                summary.append({"main": main, "status": "failed", "reason": "model_file_not_created"})
                failed_mains.append(main)
        except KeyboardInterrupt:
            print(f"\n[중단] 사용자에 의해 {main} 학습이 중단되었습니다.")
            summary.append({"main": main, "status": "interrupted"})
            failed_mains.append(main)
            raise
        except Exception as e:
            import traceback
            error_msg = str(e)
            error_trace = traceback.format_exc()
            print(f"[✗ 오류] {main} 학습 중 오류 발생:")
            print(f"  오류 메시지: {error_msg}")
            if cfg.debug:
                print(f"  상세 추적:\n{error_trace}")
            summary.append({"main": main, "status": "error", "msg": error_msg})
            failed_mains.append(main)
    
    # 최종 요약 출력
    print(f"\n{'='*60}")
    print(f"=== 전체 학습 요약 ===")
    print(f"{'='*60}")
    success_count = sum(1 for s in summary if s.get("status") == "ok")
    print(f"성공: {success_count}/{len(mains_order)}")
    if failed_mains:
        print(f"실패: {', '.join(failed_mains)}")
    print(f"{'='*60}\n")

    (run_root / "summary_all_mains.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] all-mains summary written to {run_root / 'summary_all_mains.json'}")


# --------------------
# 5) CLI (robust & tunable, aligned with TrainCfg)
# --------------------
def parse_args() -> TrainCfg:
    import argparse
    ap = argparse.ArgumentParser(
        "Sub-emotion trainer (v2)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # --- 데이터 하한(TrainCfg와 동일 기본값) ---
    ap.add_argument("--min-join", type=int, default=20,
                    help="최소 조인 샘플 수(없으면 자동 하한 대신, 이 값을 최소로 사용)")
    ap.add_argument("--min-classes", type=int, default=1,
                    help="최소 서로 다른 서브 라벨 수")

    # --- 필수 입력 ---
    ap.add_argument("--emb-jsonl", default="", help="임베딩 JSONL(run.jsonl). 비우면 자동 탐색")
    ap.add_argument("--labels-jsonl", default="", help="정답지 JSONL(.jsonl 또는 포인터 .path). 비우면 자동 탐색하며 LABELS_JSONL 환경변수도 참조")
    ap.add_argument("--oneclick", choices=["quick","prod","imbalanced"], default=None,
                    help="추천 프리셋 자동 적용(입력 경로/재현성/주요 하이퍼파라미터 일괄 설정)")

    # --- 스키마/세션 ---
    ap.add_argument("--emotions-json", default="src/EMOTIONS.JSON", help="메인→서브 정의 파일")
    ap.add_argument("--session-dir", default=None, help="(선택) 세션 루트: 컨텍스트 피처 결합 시 사용")

    # --- 타깃 (상호배타) ---
    tgt = ap.add_mutually_exclusive_group()
    tgt.add_argument("--main", default=None, help="특정 메인만 학습: 희|노|애|락")
    tgt.add_argument("--all-mains", action="store_true", help="4개 메인 모두 학습")

    # --- 출력/실행 ---
    ap.add_argument("--outdir", default="src/models/sub_v2", help="출력 폴더(모델/요약 저장 루트)")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu", "mps"], help="학습 디바이스")
    ap.add_argument("--overwrite", action="store_true", help="같은 outdir가 존재하면 덮어쓰기(기본은 타임스탬프 새 폴더 생성)")
    ap.add_argument("--debug", type=int, default=0, help="디버그 레벨(>0이면 상세 로그)")

    # --- 조인 필드 ---
    ap.add_argument("--id-field", default="id", help="임베딩 JSONL의 id 필드명")
    ap.add_argument("--emb-field", default="embedding", help="임베딩 벡터 필드명")
    
    # --- 정규화 방식 ---
    ap.add_argument("--normalize", choices=["l2","zscore","none"], default="l2",
                    help="입력 임베딩 정규화 방식")

    # --- 라벨 필터링/가중 (TrainCfg 기본값과 일치) ---
    ap.add_argument("--min-conf", type=float, default=0.3, help="라벨 confidence 하한")
    fb = ap.add_mutually_exclusive_group()
    fb.add_argument("--exclude-fallback", dest="exclude_fallback", action="store_true",
                    help="fallback_used가 있는 샘플 제외")
    fb.add_argument("--include-fallback", dest="exclude_fallback", action="store_false",
                    help="fallback 샘플도 포함")
    ap.set_defaults(exclude_fallback=True)
    ap.add_argument("--conf-power", type=float, default=2.0, help="샘플 가중 지수 conf**p (p>0)")

    # --- 학습 하이퍼파라미터 ---
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)

    # --- 조기 종료/지표/불균형(옵션화) ---
    ap.add_argument("--early-stop-patience", type=int, default=5,
                    help="검증 성능이 개선되지 않을 때 기다릴 에포크 수(0이면 조기 종료 끔)")
    ap.add_argument("--min-epochs", type=int, default=5,
                    help="조기 종료를 보더라도 최소 보장 학습 에포크 수")
    ap.add_argument("--early-stop-metric", choices=["macro_f1","present_f1","loss"], default="macro_f1",
                    help="Plateau 스케줄러/조기 종료 기준 지표")

    # 클래스 불균형/샘플러/로스
    ap.add_argument("--class-weight-mode", choices=["none","freq_inv","effective","cb-focal"], default="effective",
                    help="클래스 가중치 계산 모드")
    ap.add_argument("--beta", type=float, default=0.999, help="effective-number beta (0.9~0.9999)")
    ap.add_argument("--use-weighted-sampler", action="store_true",
                    help="불균형일 때 가중 샘플러 사용")
    ap.add_argument("--label-smoothing", type=float, default=0.0, help="Label smoothing for CE/CB-Focal")
    ap.add_argument("--focal-gamma", type=float, default=2.0, help="Focal loss gamma")
    
    # 희귀 클래스 보너스 파라미터
    ap.add_argument("--rare-th", type=int, default=0,
        help="희귀 클래스로 간주할 지원수 하한(0이면 자동 0.5% 규칙)")
    ap.add_argument("--rare-boost", type=float, default=1.5,
        help="희귀 클래스 가중 보너스 배율(1.0=미사용)")
    
    # MixUp 및 K-Fold 옵션 추가
    ap.add_argument("--mixup-alpha", type=float, default=0.0,
                    help="임베딩 MixUp alpha(0이면 비활성)")
    ap.add_argument("--kfold", type=int, default=0,
                    help="k>1이면 Stratified K-Fold 학습(개별 폴드 모델 + 앙상블 메타 저장)")

    # 검증 보조
    ap.add_argument("--eval-present-only", action="store_true",
                    help="검증 시 present-only F1을 보조로 사용")
    ap.add_argument("--eval-min-support", type=int, default=0,
                    help="검증 sup>=K macro-F1 계산에 사용할 K(0이면 미사용)")

    # AMP/누적/클립
    ap.add_argument("--amp", action="store_true", help="Enable AMP (CUDA only)")
    ap.add_argument("--amp-dtype", choices=["fp16","bf16"], default="fp16", help="AMP dtype")
    ap.add_argument("--amp-fallback", action="store_true", help="AMP OOM 시 FP32 폴백")
    ap.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation steps")
    ap.add_argument("--grad-clip", type=float, default=0.0,
                    help=">0이면 그라디언트 클립 적용")

    # 스케줄러
    ap.add_argument("--sched", choices=["none","onecycle","cosine"], default="onecycle", help="LR scheduler")
    ap.add_argument("--min-lr", type=float, default=1e-5, help="Cosine 등에서 사용할 최소 LR")

    ap.add_argument("--save-history", action="store_true", help="학습 히스토리 JSON 저장")
    ap.add_argument("--log-interval", type=int, default=50, help="디버그 로그 출력 간격(iter)")
    ap.add_argument("--resume-from", default=None, help="warm-start 체크포인트 경로(선택)")
    
    # 개선된 로깅 및 모니터링 옵션
    ap.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO",
                    help="로깅 레벨")
    ap.add_argument("--log-file", default=None, help="로그 파일 경로")
    ap.add_argument("--enable-performance-monitoring", action="store_true", default=True,
                    help="성능 모니터링 활성화")
    ap.add_argument("--disable-performance-monitoring", dest="enable_performance_monitoring", action="store_false",
                    help="성능 모니터링 비활성화")
    ap.add_argument("--enable-memory-monitoring", action="store_true", default=True,
                    help="메모리 모니터링 활성화")
    ap.add_argument("--disable-memory-monitoring", dest="enable_memory_monitoring", action="store_false",
                    help="메모리 모니터링 비활성화")
    ap.add_argument("--auto-optimize-batch-size", action="store_true", default=True,
                    help="자동 배치 크기 최적화")
    ap.add_argument("--disable-auto-optimize-batch-size", dest="auto_optimize_batch_size", action="store_false",
                    help="자동 배치 크기 최적화 비활성화")
    ap.add_argument("--quality-threshold", type=float, default=0.7,
                    help="데이터 품질 임계값 (0.0-1.0)")

    args = ap.parse_args()

    # TrainCfg에 존재하는 필드만 골라 전달
    cfg_fields = set(TrainCfg.__dataclass_fields__.keys())
    cfg_kwargs = {k: v for k, v in vars(args).items() if k in cfg_fields}
    cfg = TrainCfg(**cfg_kwargs)
    
    # --- 원클릭 프리셋 ---
    if args.oneclick:
        # 공통
        cfg.all_mains = True if getattr(args,"all_mains",False) or (getattr(args,"main",None) is None) else False  # type: ignore
        cfg.normalize = "l2"; cfg.save_history = True; cfg.seed = 2025
        if args.oneclick == "quick":
            cfg.outdir = "src/models/sub_v2_smoke"
            cfg.min_conf = 0.40; cfg.exclude_fallback = True; cfg.conf_power = 2.0
            cfg.min_classes = 2; cfg.val_ratio = 0.15
            cfg.epochs = 8; cfg.batch = 256
            cfg.class_weight_mode = "effective"
            cfg.mixup_alpha = 0.15
            cfg.amp = True; cfg.amp_dtype = "fp16"; cfg.amp_fallback = True
            cfg.eval_present_only = True; cfg.eval_min_support = 3
            # 개선된 모니터링 설정
            cfg.log_level = "INFO"; cfg.enable_performance_monitoring = True
            cfg.enable_memory_monitoring = True; cfg.auto_optimize_batch_size = True
            cfg.quality_threshold = 0.6  # 빠른 테스트를 위해 낮춤
        elif args.oneclick == "prod":
            cfg.outdir = "src/models/sub_v2"
            cfg.min_conf = 0.55; cfg.exclude_fallback = True; cfg.conf_power = 2.5
            cfg.min_classes = 2; cfg.val_ratio = 0.20
            cfg.epochs = 20; cfg.batch = 128; cfg.lr = 2e-4; cfg.weight_decay = 1e-4
            cfg.sched = "onecycle"; cfg.min_lr = 1e-5
            cfg.mixup_alpha = 0.20; cfg.label_smoothing = 0.05
            cfg.class_weight_mode = "effective"; cfg.beta = 0.9995
            cfg.amp = True; cfg.amp_dtype = "fp16"; cfg.amp_fallback = True
            cfg.early_stop_patience = 5; cfg.min_epochs = 5; cfg.early_stop_metric = "macro_f1"
            cfg.eval_present_only = True; cfg.eval_min_support = 5
            # 개선된 모니터링 설정
            cfg.log_level = "INFO"; cfg.enable_performance_monitoring = True
            cfg.enable_memory_monitoring = True; cfg.auto_optimize_batch_size = True
            cfg.quality_threshold = 0.7; cfg.log_file = f"training_{_now_tag()}.log"
        elif args.oneclick == "imbalanced":
            cfg.outdir = "src/models/sub_v2_imbalanced"
            cfg.min_conf = 0.55; cfg.exclude_fallback = True; cfg.conf_power = 3.0
            cfg.min_classes = 2
            cfg.class_weight_mode = "freq_inv"; cfg.use_weighted_sampler = True
            cfg.epochs = 20; cfg.batch = 128
            cfg.mixup_alpha = 0.20; cfg.label_smoothing = 0.05
            cfg.sched = "onecycle"; cfg.eval_present_only = True; cfg.eval_min_support = 5
            # 개선된 모니터링 설정
            cfg.log_level = "INFO"; cfg.enable_performance_monitoring = True
            cfg.enable_memory_monitoring = True; cfg.auto_optimize_batch_size = True
            cfg.quality_threshold = 0.5  # 불균형 데이터를 위해 낮춤

    # 'overwrite'는 TrainCfg에 없으므로 safe_overwrite만 반영
    cfg.safe_overwrite = (not args.overwrite)
    return cfg


if __name__ == "__main__":
    cfg = parse_args()
    try:
        k = int(getattr(cfg, "kfold", 0) or 0)
        if getattr(cfg, "main", None):
            if k and k > 1:
                train_kfold_for_main(cfg, k=k)
            else:
                train_for_main(cfg)
        else:
            # all-mains + kfold는 보통 시간이 길어지므로 main별 루프 안에서 개별로 kfold 호출 권장
            train_all_mains(cfg)
    except Exception as e:
        print(f"[fatal] {e}")
        if getattr(cfg, "debug", 0):
            traceback.print_exc()
