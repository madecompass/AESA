# config.py
# -*- coding: utf-8 -*-

import io, sys, os

# --- GPU 최적화 환경변수 자동 설정 ---
os.environ.setdefault("ENABLE_EMBEDDINGS", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:512")
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "0")
os.environ.setdefault("TORCH_CUDNN_V8_API_ENABLED", "1")

if os.name == "nt":
    os.environ.setdefault("PYTHONUTF8", "1")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
        sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
        sys.stdin.reconfigure(encoding="utf-8", errors="replace")
    else:
        # 안전 폴백: detach 후 새 래퍼 부착
        try:
            sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding="utf-8", errors="replace", line_buffering=True)
        except Exception:
            pass
        try:
            sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding="utf-8", errors="replace", line_buffering=True)
        except Exception:
            pass
        try:
            sys.stdin = io.TextIOWrapper(sys.stdin.detach(), encoding="utf-8", errors="replace")
        except Exception:
            pass

"""
설계 목표 (요약)
- EMOTIONS.JSON을 ‘단일 진실원(SoT)’으로 두고, config.py는 토글/스칼라 하이퍼파라미터와 엔트리 포인트만 관리
- 기존 키/섹션을 최대한 유지하면서, 식별자·경로·디바이스·임계값의 불일치/중복을 해소
- 임포트 부작용 최소화(읽기만 수행, 쓰기는 별도 로직에서)
- 계층 분류(희/노/애/락 → 세부 → 단일기) 라우팅/불확실성 임계값을 단일 THRESHOLDS 네임스페이스로 통일
"""

import os
import json
from importlib.util import find_spec
import logging
import hashlib
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Iterable, Union, List


# --------------------------------
# 환경변수 헬퍼 함수 (시연 모드 지원)
# --------------------------------
def _env_bool(k, d):
    """환경변수를 bool로 변환 (1/true/yes/y → True)"""
    return str(os.getenv(k, str(int(d)))).lower() in ("1", "true", "yes", "y")


def _env_float(k, d):
    """환경변수를 float로 변환"""
    return float(os.getenv(k, str(d)))


def _env_str(k, d):
    """환경변수를 문자열로 변환"""
    return os.getenv(k, d)


def _env_int(k, d):
    """환경변수를 int로 변환"""
    try:
        return int(os.getenv(k, str(d)))
    except (ValueError, TypeError):
        return d

# ★ 안전 캐스팅(환경변수/설정/JSON 값이 None·문자열이어도 안전)
def _safe_int(x, default):
    try:
        return int(x) if x is not None else int(default)
    except Exception:
        return int(default)


# --------------------------------
# 시연 모드 기본 설정
# --------------------------------
DEMO_MODE = _env_bool("DEMO_MODE", False)  # 고성능 데스크탑용 프로덕션 모드
EMOTIONS_JSON_ENV = _env_str("EMOTIONS_JSON", "EMOTIONS.JSON")

# 정확성 우선 임계값 (시연 프리셋에서 상향)
EMOTION_ALIAS_MODE = _env_str("EMOTION_ALIAS_MODE", "strict_neutral_lak" if DEMO_MODE else "baseline")
ENABLE_TEMP_SCALING = _env_bool("ENABLE_TEMP_SCALING", True if DEMO_MODE else False)
REJECT_FALLBACK_UNDER = _env_float("REJECT_FALLBACK_UNDER", 0.50 if DEMO_MODE else 0.35)
REQUIRE_MIN_CONF = _env_float("REQUIRE_MIN_CONF", 0.50 if DEMO_MODE else 0.40)
SUB_MIN_POST = _env_float("SUB_MIN_POST", 0.40 if DEMO_MODE else 0.35)

# 긴 스텝 방지용 타임아웃
# 웹 UI에서는 충분한 시간을 주어야 함 (intensity_analysis가 오래 걸림)
DEFAULT_STEP_TIMEOUT = _env_int("DEFAULT_STEP_TIMEOUT", 0)  # 고성능 환경에서는 타임아웃 비활성화

# 스텝별 개별 타임아웃 설정 (JSON 형식 환경변수)
STEP_TIMEOUTS = {}
_raw = os.getenv("STEP_TIMEOUTS", "").strip()
if _raw:
    try:
        STEP_TIMEOUTS = {str(k): int(v) for k, v in json.loads(_raw).items()}
    except Exception as e:
        print(f"[config] WARN invalid STEP_TIMEOUTS: {e}")

# 스텝별 타임아웃 기본값 설정 (중량 스텝에 실시간 타임아웃 부여)
DEFAULT_STEP_TIMEOUTS = {
    # GPU 로딩/임베딩 등 장시간 작업 스텝만 타임아웃 유지
    "intensity_analysis": 360,
    "embedding_generation": 300,
    "pattern_extractor": 360,
    "transition_analyzer": 60,
    "complex_analyzer": 90,
    "time_series_analyzer": 75,
    "context_analysis": 120,  # context_extractor 타임아웃 추가
    "context_extractor": 120,  # context_extractor 타임아웃 추가 (별칭)
}

# 경량 스텝은 기본 0으로 두어 불필요한 타임아웃 실행기를 생성하지 않음
# 주의: context_analysis는 DEFAULT_STEP_TIMEOUTS에 명시적으로 설정되어 있으므로 여기서 제외
_ZERO_TIMEOUT_STEPS = [
    "psychological_analyzer",
    "relationship_analyzer",
    # "context_analysis",  # 제거: DEFAULT_STEP_TIMEOUTS에서 120초로 설정됨
    "linguistic_matcher",
    "situation_analyzer",
    "sub_emotion_detection",
    "weight_calculator",
]
for _step in _ZERO_TIMEOUT_STEPS:
    DEFAULT_STEP_TIMEOUTS[_step] = 0

# 환경변수에서 설정된 값이 없으면 기본값 사용 (0도 포함)
for step, default_timeout in DEFAULT_STEP_TIMEOUTS.items():
    if step not in STEP_TIMEOUTS:
        STEP_TIMEOUTS[step] = default_timeout

# --------------------------------
# Torch 디바이스 (있으면 cuda, 없으면 cpu)
# --------------------------------
try:
    import torch  # noqa

    _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:
    _DEVICE = "cpu"

# ============================
# 기본 디렉토리 및 경로 설정 (Render-friendly)
#  - 임포트 시 쓰기 부작용 제거
#  - 경로는 레포 루트 기준 + ENV 오버라이드
# ============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 레포 루트(render/ 등) 기준
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

DATA_FOLDER = os.path.join(BASE_DIR, "talk")
RESULTS_LOG_PATH = os.path.join(BASE_DIR, "emotion_analysis_results.txt")
SUB_RESULTS_LOG_PATH = os.path.join(BASE_DIR, "sub_emotion_analysis_results.txt")

# [임포트 시 디렉터리 생성 금지]
STRICT_LABEL_DIR = os.getenv("STRICT_LABEL_DIR", "1") == "1"

# [개선사항 1] 엄격 모드 토글 - 운영환경에서 필수 리소스 없으면 즉시 실패
CFG_STRICT_ON_START = os.getenv("CFG_STRICT_ON_START", "0") == "1"
STRICT_SUBS = os.getenv("STRICT_SUBS", "0") == "1"  # 서브4 모두 필수
MODEL_DIR = os.getenv("MODEL_DIR") or str(SRC_DIR / "models")
LOG_DIR = os.getenv("LOG_DIR") or str(SRC_DIR / "logs")
EMBEDDINGS_DIR = os.getenv("EMBEDDINGS_DIR") or str(Path(BASE_DIR) / "embeddings")
# 라벨 원천 디렉터리(없어도 기동 가능하도록 None 허용)
# Render 배포 환경에서는 환경변수로 경로 설정
LABEL_DIR = os.getenv("LABEL_DIR") or (str(PROJECT_ROOT / "label") if STRICT_LABEL_DIR else None)

# 프로파일 및 배포 환경 설정 (최상단에 배치)
EA_PROFILE = os.getenv("EA_PROFILE", "prod")  # 'dev'|'pilot'|'prod'
RENDER_DEPLOYMENT = os.getenv("RENDER_DEPLOYMENT", "0") == "1"
USE_CUDA = os.getenv("USE_CUDA", "1") == "1"  # CUDA 사용 기본값 완화

# Render 배포 시 자동 폴더 생성 비활성화
if RENDER_DEPLOYMENT:
    STRICT_LABEL_DIR = False  # Render에서는 라벨 폴더 없이도 동작


# 필요 시(서버 기동 시점) 한 번만 호출하여 생성
def ensure_runtime_dirs():
    for p in [LOG_DIR, MODEL_DIR]:
        if p and not os.path.isdir(p):
            os.makedirs(p, exist_ok=True)


# === 모델 파일 중앙 매핑 (ENV로 덮어쓰기 가능) ===
MODEL_FILENAMES = {
    "unified": os.getenv("UNIFIED_MODEL", "unified_model.pt"),
    "sub": {
        "희": os.getenv("SUB_HI", "sub_희.pt"),
        "노": os.getenv("SUB_NO", "sub_노.pt"),
        "애": os.getenv("SUB_AE", "sub_애.pt"),
        "락": os.getenv("SUB_RAK", "sub_락.pt"),
    }
}


def get_model_path(kind: str, key: Optional[str] = None) -> str:
    """kind in {'unified','sub'}; key in {'희','노','애','락'} when kind='sub'"""
    base = Path(MODEL_DIR)
    if kind == "unified":
        return str(base / MODEL_FILENAMES["unified"])
    if kind == "sub" and key in MODEL_FILENAMES["sub"]:
        return str(base / MODEL_FILENAMES["sub"][key])
    raise KeyError(f"Unknown model kind/key: {kind}/{key}")


MAPPING_SAVE_PATH = {
    'sub_emotion_to_index': os.path.join(MODEL_DIR, 'sub_emotion_to_index.json'),
    'index_to_sub_emotion': os.path.join(MODEL_DIR, 'index_to_sub_emotion.json'),
    'intensity_to_index': os.path.join(MODEL_DIR, 'intensity_to_index.json'),
    'index_to_intensity': os.path.join(MODEL_DIR, 'index_to_intensity.json'),
}

EMBEDDINGS_PATH = os.path.join(EMBEDDINGS_DIR, "emotion_embeddings.pt")

# ============================
# 메인데이터전처리 로깅
# ============================
LOGGING_CONFIG = {
    # 스텝별 결과 JSON: 세션 빌더의 필수 재료
    "write_step_results": True,
    "write_step_results_for_cached": False,  # 기본은 억제(대량 배치 I/O 절감)

    # 날짜/타임존
    "date_fmt": "%Y%m%d",
    "tz_offset_hours": 9,

    # 파일 로테이션
    "rotate_main": {"max_bytes": 50 * 1024 * 1024, "backup_count": 10},  # 고성능 환경에 맞춰 로그 크기 증가
    "rotate_sub_classifier": {"max_bytes": 20 * 1024 * 1024, "backup_count": 5},
    "rotate_data_utils": {"max_bytes": 20 * 1024 * 1024, "backup_count": 5},

    # 모듈 파일로그 억제/리다이렉트 (원하시면 둘 중 하나만 True)
    "suppress_module_file_logs_when_orchestrated": True,
    "redirect_module_file_logs_to_run_dir": False,

    # 구조화(JSON) 로그 옵션(머신 파싱용; 기본 False)
    "json_logs": False,
}


# --------------------------------
# EMOTIONS.JSON 경로 해석(대소문자·ENV 대응)
# --------------------------------
def _resolve_emotions_json_path(project_root: Path) -> Optional[Path]:
    """
    우선순위:
      1) ENV: EMOTIONS_JSON
      2) src/EMOTIONS.json (프로젝트 구조에 맞춘 경로)
      3) project_root 하위: EMOTIONS.json / EMOTIONS.JSON / emotions.json
      4) /mnt/data 하위(데모·개발 환경용): 동일한 파일명 후보
      5) Render 배포 환경: 환경변수 경로 우선
    """
    # EMOTIONS_JSON_ENV는 이미 상단에서 _env_str로 파싱됨
    env_path = EMOTIONS_JSON_ENV
    if env_path and Path(env_path).is_file():
        return Path(env_path)

    # 프로젝트 구조에 맞춘 경로 우선 확인
    src_candidates = [
        project_root / "src" / "EMOTIONS.json",
        project_root / "src" / "EMOTIONS.JSON",
        project_root / "src" / "emotions.json",
    ]
    for p in src_candidates:
        if p.is_file():
            return p

    # Render 배포 환경에서는 추가 경로 체크
    if RENDER_DEPLOYMENT:
        render_candidates = [
            Path("/opt/render/project/src/EMOTIONS.json"),
            Path("/opt/render/project/src/EMOTIONS.JSON"),
            Path("/opt/render/project/EMOTIONS.json"),
            Path("/opt/render/project/EMOTIONS.JSON"),
        ]
        for p in render_candidates:
            if p.is_file():
                return p

    candidates = [
        project_root / "EMOTIONS.json",
        project_root / "EMOTIONS.JSON",
        project_root / "emotions.json",
        Path("/mnt/data/EMOTIONS.json"),
        Path("/mnt/data/EMOTIONS.JSON"),
        Path("/mnt/data/emotions.json"),
    ]
    for p in candidates:
        if p.is_file():
            return p
    return None


EMOTIONS_JSON_PATH = _resolve_emotions_json_path(PROJECT_ROOT)

# EMOTIONS_JSON_PATH가 None인 경우 src/EMOTIONS.json으로 기본 설정
if EMOTIONS_JSON_PATH is None:
    EMOTIONS_JSON_PATH = PROJECT_ROOT / "src" / "EMOTIONS.json"

# ============================
# 로깅 설정 (환경변수/기본값)
# ============================

JSON_LOGS = (os.getenv("JSON_LOGS", "0") == "1")
LOGGING_CONFIG["json_logs"] = bool(JSON_LOGS)
if JSON_LOGS:
    LOGGING_FORMAT = '{"time":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","message":"%(message)s"}'
else:
    LOGGING_FORMAT = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
SUB_CLASSIFIER_LOG_PATH = os.path.join(LOG_DIR, 'sub_classifier.log')
MAIN_LOG_PATH = os.path.join(LOG_DIR, 'main.log')
DATA_UTILS_LOG_PATH = os.path.join(LOG_DIR, 'data_utils.log')

# ============================
# 통합 로그 관리 설정
# ============================

# 로그 컨텍스트별 디렉터리 구조
LOG_CONTEXT_CONFIG = {
    # 독립 모듈 테스트: emotion_analysis/logs/
    "emotion_analysis": {
        "base_dir": SRC_DIR / "emotion_analysis" / "logs",
        "description": "독립 모듈 테스트용 로그"
    },
    # 메인 데이터 전처리: logs/emotion_analysis/
    "main": {
        "base_dir": Path(LOG_DIR) / "emotion_analysis", 
        "description": "메인 데이터 전처리용 로그"
    },
    # 학습기: logs/{학습기이름}/
    "trainer": {
        "base_dir": Path(LOG_DIR) / "trainers",
        "description": "학습기용 로그"
    },
    # 웹 서비스: logs/web/
    "web": {
        "base_dir": Path(LOG_DIR) / "web",
        "description": "웹 서비스용 로그"
    },
    # 기본: logs/default/
    "default": {
        "base_dir": Path(LOG_DIR) / "default",
        "description": "기본 로그"
    }
}

# 로그 리다이렉트 설정
LOGGING_CONFIG.update({
    # 모듈 로그 억제/리다이렉트 설정
    "suppress_module_file_logs_when_orchestrated": True,
    "redirect_module_file_logs_to_run_dir": False,
    
    # 통합 로그 관리자 사용
    "use_unified_log_manager": True,
    "auto_detect_context": True,
    
    # 컨텍스트별 로그 설정
    "context_logging": {
        "emotion_analysis": {
            "console": True,
            "file": True,
            "max_bytes": 100 * 1024 * 1024,  # 고성능 환경에 맞춰 로그 크기 증가
            "backup_count": 10
        },
        "main": {
            "console": True,
            "file": True,
            "max_bytes": 200 * 1024 * 1024,  # 고성능 환경에 맞춰 로그 크기 증가
            "backup_count": 20
        },
        "trainer": {
            "console": False,
            "file": True,
            "max_bytes": 500 * 1024 * 1024,  # 고성능 환경에 맞춰 로그 크기 증가
            "backup_count": 50
        },
        "web": {
            "console": False,
            "file": True,
            "max_bytes": 200 * 1024 * 1024,  # 고성능 환경에 맞춰 로그 크기 증가
            "backup_count": 20
        }
    }
})

_DEFAULT_LOG_LEVEL = os.getenv("CONFIG_LOG_LEVEL", "INFO").upper()


def _make_handlers():
    hs = [logging.StreamHandler()]

    # main.log
    rot = LOGGING_CONFIG.get("rotate_main")
    if rot:
        hs.append(RotatingFileHandler(
            MAIN_LOG_PATH,
            maxBytes=_safe_int(rot.get("max_bytes"), 5 * 1024 * 1024),
            backupCount=_safe_int(rot.get("backup_count"), 5),
            encoding="utf-8"
        ))
    else:
        hs.append(logging.FileHandler(MAIN_LOG_PATH, encoding="utf-8"))

    # sub_classifier.log
    rot_sc = LOGGING_CONFIG.get("rotate_sub_classifier")
    if rot_sc:
        hs.append(RotatingFileHandler(
            SUB_CLASSIFIER_LOG_PATH,
            maxBytes=_safe_int(rot_sc.get("max_bytes"), 2 * 1024 * 1024),
            backupCount=_safe_int(rot_sc.get("backup_count"), 3),
            encoding="utf-8"
        ))
    else:
        hs.append(logging.FileHandler(SUB_CLASSIFIER_LOG_PATH, encoding="utf-8"))

    # data_utils.log
    rot_du = LOGGING_CONFIG.get("rotate_data_utils")
    if rot_du:
        hs.append(RotatingFileHandler(
            DATA_UTILS_LOG_PATH,
            maxBytes=_safe_int(rot_du.get("max_bytes"), 2 * 1024 * 1024),
            backupCount=_safe_int(rot_du.get("backup_count"), 3),
            encoding="utf-8"
        ))
    else:
        hs.append(logging.FileHandler(DATA_UTILS_LOG_PATH, encoding="utf-8"))

    return hs


# [A-1] 루트 로깅 초기화 가드(비간섭 옵션화)
if os.getenv("CONFIG_INIT_LOGGING", "0") == "1" and not logging.getLogger().handlers:
    ensure_runtime_dirs()
    logging.basicConfig(
        level=getattr(logging, _DEFAULT_LOG_LEVEL, logging.INFO),
        format=LOGGING_FORMAT,
        handlers=_make_handlers()
    )
# else: 상위 앱/CLI가 로깅을 구성하도록 비간섭 유지

logger = logging.getLogger(__name__)


# ============================
# 유틸: 딥 머지/해시
# ============================

def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _sha256_of_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _sha256_of_obj(obj: Any) -> str:
    try:
        data = json.dumps(obj, ensure_ascii=False, sort_keys=True).encode("utf-8")
        return _sha256_of_bytes(data)
    except Exception:
        return "NA"


# ============================
# 임베딩 관련 설정
# ============================
EMBEDDING_PARAMS = {
    'model_name': 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
    'pooling_strategy': 'mean',
    'similarity_metric': 'cosine',
    'embedding_dim': 384,  # ★ 모델 hidden_size와 일치
    # 추천 설정(속도/정확 타협): 긴 문장 윈도우링에 유리
    'max_sequence_length': 512,  # TRAINING_PARAMS.MAX_LEN과 일치
    'device': _DEVICE,  # [A-4] 자동 감지 장치로 통일
    'doc_stride': 128,  # 윈도우 중첩
    # 윈도우 미니배치(텍스트 인코더 내부 윈도우 묶음 처리) - GPU 활용 극대화
    'text_batch': 512,  # CPU 최적화: 256→512 증가 (2배)
    # 성능 최적화 설정
    'adaptive_batch_size': True,  # 동적 배치 크기 조정
    'min_batch_size': 256,  # CPU 최적화: 128→256 증가 (2배)
    'max_batch_size': 4096,  # CPU 최적화: 2048→4096 증가 (2배)
    'memory_threshold_mb': 81920,  # CPU 최적화: 61440→81920 증가 (80GB)
    'performance_monitoring': True,  # 성능 모니터링 활성화
    'cache_embeddings': True,
    'embedding_cache_path': os.path.join(EMBEDDINGS_DIR, "cached_embeddings.pt"),
    'cache_size': 8192,  # CPU 최적화: 4096→8192 증가 (2배)
    'text_cleaning': True,
    'normalize_whitespace': True,
    'remove_special_chars': True,
    # (zero shot 등 기타 키 유지)
    'zero_shot_model_name': 'joeddav/xlm-roberta-large-xnli',

    'batch_size': 2048  # CPU 최적화: 1024→2048 증가 (2배)
}

# ============================
# 모델 캐싱 시스템 (성능 최적화)
# ============================
import threading
from typing import Dict, Any, Optional

# 전역 모델 캐시
_model_cache: Dict[str, Any] = {}
_model_cache_lock = threading.RLock()
_model_cache_stats = {
    'hits': 0,
    'misses': 0,
    'loads': 0,
    'unloads': 0
}

# 운영 환경 판별 변수 (MODEL_CACHE_CONFIG에서 사용)
is_prod = (EA_PROFILE == "prod") or RENDER_DEPLOYMENT

MODEL_CACHE_CONFIG = {
    'enabled': _env_bool("MODEL_CACHE_ENABLED", True),
    # 운영 환경에서는 캐시 상한 축소로 메모리 압박 방지
    'max_models': _env_int("MODEL_CACHE_MAX_MODELS", 16 if is_prod else 32),
    'unload_delay': _env_int("MODEL_CACHE_UNLOAD_DELAY", 1800 if is_prod else 3600),  # 운영: 30분, 개발: 60분
    'memory_threshold': _env_float("MODEL_CACHE_MEMORY_THRESHOLD", 0.85 if is_prod else 0.95),  # 운영: 85%, 개발: 95%
    'preload_models': ['unified', 'sub_v2'] if is_prod else ['unified', 'sub_v2', 'intensity', 'pattern', 'context'],  # 운영: 필수만, 개발: 전체
    'auto_cleanup': _env_bool("MODEL_CACHE_AUTO_CLEANUP", True),  # 3번 개선작업: 자동 정리 활성화
}

def get_cached_model(model_name: str, model_class: Any, **kwargs) -> Any:
    """캐시된 모델 반환 - 성능 최적화"""
    if not MODEL_CACHE_CONFIG['enabled']:
        return model_class.from_pretrained(model_name, **kwargs)
    
    with _model_cache_lock:
        cache_key = f"{model_name}:{hash(str(kwargs))}"
        
        if cache_key in _model_cache:
            _model_cache_stats['hits'] += 1
            return _model_cache[cache_key]
        
        # 캐시 미스 - 새 모델 로드
        _model_cache_stats['misses'] += 1
        _model_cache_stats['loads'] += 1
        
        # 메모리 사용량 체크
        if len(_model_cache) >= MODEL_CACHE_CONFIG['max_models']:
            _cleanup_model_cache()
        
        model = model_class.from_pretrained(model_name, **kwargs)
        _model_cache[cache_key] = model
        
        return model

def _cleanup_model_cache() -> None:
    """모델 캐시 정리 - 메모리 관리"""
    if not MODEL_CACHE_CONFIG['auto_cleanup']:
        return
    
    with _model_cache_lock:
        if len(_model_cache) >= MODEL_CACHE_CONFIG['max_models']:
            # 가장 오래된 모델 제거 (FIFO)
            oldest_key = next(iter(_model_cache))
            del _model_cache[oldest_key]
            _model_cache_stats['unloads'] += 1

def get_model_cache_stats() -> Dict[str, Any]:
    """모델 캐시 통계 반환"""
    with _model_cache_lock:
        total_requests = _model_cache_stats['hits'] + _model_cache_stats['misses']
        hit_rate = _model_cache_stats['hits'] / total_requests if total_requests > 0 else 0.0
        
        return {
            'cache_size': len(_model_cache),
            'hit_rate': hit_rate,
            'total_requests': total_requests,
            'hits': _model_cache_stats['hits'],
            'misses': _model_cache_stats['misses'],
            'loads': _model_cache_stats['loads'],
            'unloads': _model_cache_stats['unloads']
        }

def clear_model_cache() -> None:
    """모델 캐시 완전 정리"""
    with _model_cache_lock:
        _model_cache.clear()
        _model_cache_stats.update({'hits': 0, 'misses': 0, 'loads': 0, 'unloads': 0})

# ============================
# 성능 최적화 설정
# ============================
PERFORMANCE_OPTIMIZATIONS = {
    'model_caching': {
        'enabled': _env_bool("MODEL_CACHE_ENABLED", True),
        'max_models': _env_int("MODEL_CACHE_MAX_MODELS", 5),
        'unload_delay': _env_int("MODEL_CACHE_UNLOAD_DELAY", 300),
        'memory_threshold': _env_float("MODEL_CACHE_MEMORY_THRESHOLD", 0.8),
        'preload_models': ['unified'],
        'auto_cleanup': _env_bool("MODEL_CACHE_AUTO_CLEANUP", True),
    },
    'pipeline_parallelization': {
        'enabled': _env_bool("PIPELINE_PARALLELIZATION", True),  # ★ 성능 최적화: 기본값 True로 강제 활성화
        'max_workers': _env_int("PIPELINE_MAX_WORKERS", 4),
        'parallel_groups': [
            ['pattern_extractor', 'linguistic_matcher'],
            ['intensity_analysis', 'context_analysis'],
            ['transition_analyzer', 'sub_emotion_detection'],
            ['weight_calculator', 'complex_analyzer', 'situation_analyzer', 'relationship_analyzer', 'psychological_analyzer']
        ],
        # ★ 성능 최적화: 인스턴스 풀 사전 생성 활성화
        'prewarm_on_start': _env_bool("PREWARM_ON_START", True),  # 오케스트레이터 시작 시 모든 모듈 인스턴스 사전 생성
    },
    'memory_management': {
        'payload_trace_limit': _env_int("PAYLOAD_TRACE_LIMIT", 100),
        'cache_max_items': _env_int("CACHE_MAX_ITEMS", 4096),  # 4배 증가
        'gc_threshold': _env_float("GC_MEMORY_THRESHOLD", 0.8),
        'compact_output': _env_bool("PAYLOAD_OUTPUT_COMPACT", True),
        'large_text_threshold': _env_int("LARGE_TEXT_THRESHOLD", 50000),  # 고성능 환경에 맞춰 대용량 텍스트 임계값 증가
    },
    'caching_strategy': {
        'enabled': _env_bool("CACHING_ENABLED", True),
        'ttl_seconds': _env_int("CACHE_TTL_SECONDS", 3600),
        'max_size_mb': _env_int("CACHE_MAX_SIZE_MB", 8192),  # 8GB로 대폭 증가
        'compression': _env_bool("CACHE_COMPRESSION", True),
        'sweep_every': _env_int("CACHE_SWEEP_EVERY", 2048),  # 고성능 환경에 맞춰 스윕 주기 증가
    }
}

# 성능 모드 설정
PERFORMANCE_MODE = os.getenv("PERFORMANCE_MODE", "balanced")  # fast, balanced, quality

if PERFORMANCE_MODE == "fast":
    PERFORMANCE_OPTIMIZATIONS['model_caching']['max_models'] = 3
    PERFORMANCE_OPTIMIZATIONS['pipeline_parallelization']['max_workers'] = 6
    PERFORMANCE_OPTIMIZATIONS['memory_management']['payload_trace_limit'] = 50
    PERFORMANCE_OPTIMIZATIONS['memory_management']['cache_max_items'] = 256
elif PERFORMANCE_MODE == "quality":
    PERFORMANCE_OPTIMIZATIONS['model_caching']['max_models'] = 8
    PERFORMANCE_OPTIMIZATIONS['pipeline_parallelization']['max_workers'] = 2
    PERFORMANCE_OPTIMIZATIONS['memory_management']['payload_trace_limit'] = 200
    PERFORMANCE_OPTIMIZATIONS['memory_management']['cache_max_items'] = 1024
elif PERFORMANCE_MODE == "balanced":
    # 기본값 유지
    pass

# 성능 모니터링 설정
PERFORMANCE_MONITORING = {
    'enabled': _env_bool("PERFORMANCE_MONITORING", True),
    'log_timings': _env_bool("LOG_PERFORMANCE_TIMINGS", True),
    'memory_monitoring': _env_bool("MEMORY_MONITORING", True),
    'cache_stats': _env_bool("CACHE_STATS_LOGGING", True),
    'parallel_stats': _env_bool("PARALLEL_STATS_LOGGING", True),
}

def get_performance_config() -> Dict[str, Any]:
    """성능 설정 반환"""
    return PERFORMANCE_OPTIMIZATIONS

def get_performance_mode() -> str:
    """현재 성능 모드 반환"""
    return PERFORMANCE_MODE

def is_performance_mode(mode: str) -> bool:
    """특정 성능 모드인지 확인"""
    return PERFORMANCE_MODE == mode
ZERO_SHOT_MODEL_NAME = EMBEDDING_PARAMS['zero_shot_model_name']

# 서빙 모드에서 임베딩 스텝 사용 여부(기본 비활성)
USE_EMBEDDINGS_AT_SERVE = (os.getenv("USE_EMBEDDINGS_AT_SERVE", "1") == "1")

# RTX 5080 CUDA 최적화:
if str(_DEVICE).startswith('cuda'):
    EMBEDDING_PARAMS['batch_size'] = 1536  # RTX 5080 16GB 메모리 최적화 배치 크기 (1024→1536)
    EMBEDDING_PARAMS['max_batch_size'] = 3072  # RTX 5080 16GB 메모리 최대 배치 크기 (2048→3072)

# 임베딩 캐시 크기 설정
EMBEDDING_CACHE_SIZE = _env_int("EMBEDDING_CACHE_SIZE", 1000)

# 패턴 추출기 폴백 렉시콘 설정
ALLOW_FALLBACK_LEX = _env_bool("ALLOW_FALLBACK_LEX", True)

EMBEDDING_SCHEMA_VERSION = "v1"
EMBEDDING_EXPORT = {"schema_version": EMBEDDING_SCHEMA_VERSION}

# === C. LABEL_BUILDER 설정(11모듈 스코어러) ===
LABEL_BUILDER_CONFIG = {
    # 가중치: 프로젝트 정책값 (필요 시 조정)
    'weights': {
        'ec': 0.45,  # emotion_classification prior (topk 분포)
        'ctx': 0.25,  # 문맥/키워드(hit ratio)
        'sit': 0.10,  # 상황 분석
        'trn': 0.08,  # 전이 신호
        'int': 0.06,  # 강도 확신도
        'rel': 0.03,  # 관계
        'psy': 0.02,  # 심리
        'cpx': 0.01,  # 복합
    },
    # 판정 임계: 1위-2위 마진/서브 임계
    'margin': 0.15,
    'sub_threshold': 0.20,
    # 컨텍스트/키워드 최소 매칭수(weak label 생성시)
    'min_keyword_hit': 1,
    # 근거 로그(flag)
    'explain': True,
}

# === [신규] Soft-Ensemble 모듈 가중치 설정 ===
# 각 분석 모듈이 최종 감정 판단에 미치는 영향력을 제어합니다.
# weight_calculator.calculate_integrated_weights()에서 참조합니다.
ENSEMBLE_WEIGHTS = {
    # 1. 핵심 신호 (가장 신뢰도 높음)
    "intensity_analysis": 1.5,      # 강도 분석 (매우 중요)
    "pattern_extractor": 1.3,       # 패턴 매칭 (명확한 신호)
    "linguistic_matcher": 1.2,      # 언어적 표현 (명시적)

    # 2. 문맥 및 상황 (중요)
    "context_analysis": 1.0,        # 문맥 흐름
    "situation_analyzer": 1.0,      # 상황적 요인
    "emotion_classification": 1.0,  # 1차 모델 분류 결과

    # 3. 관계 및 구조 (보조)
    "relationship_analyzer": 0.9,   # 감정 관계
    "transition_analyzer": 0.8,     # 전이 패턴
    "time_series_analyzer": 0.8,    # 시계열 변동

    # 4. 심층 해석 (참고용)
    "complex_analyzer": 0.7,        # 복합 감정
    "psychological_analyzer": 0.6,  # 심리적 기제
    
    # 기본값 (설정에 없는 모듈)
    "default": 0.5
}

# 설정 버전 (변경 이력 추적용)
CONFIG_VERSION = "1.5.0-soft-ensemble"

# === D. VALIDATION / SMOKE 게이트 ===
VALIDATION_GATES = {
    'min_probe_acc': 0.90,  # 선형 프로브 최소 정확도
    'min_knn': 0.60,  # kNN@1 최소 일치율(옵션)
    'max_nan_inf': 0.0,  # NaN/Inf 허용 퍼센트
    'max_zero_pct': 1.0,  # 영벡터 허용 퍼센트
}

SMOKE_DEFAULTS = {
    'epochs': 3,
    'batch': 64,
    'lr': 1e-3
}

# === E. DATASET/라벨 조인 정책 ===
# [A-6] 조인/ID 계약(항상 내보내기)
DATASET_EXPORT = {
    'id_source': 'trace_id',  # 'trace_id' | 'text_hash' | 'id'
    'include_label_main': True,  # 임베딩 JSONL에 label_main 필수 기록 권장
    'emit_trace_id': True,
    'emit_text_hash': True,
}

LABEL_JOIN_POLICY = {
    'prefer_keys': ['id', 'trace_id', 'text_hash'],  # 조인 시도 순서
    'allow_text_hash': True
}

# ============================
# 임베딩 레코드 표준화
# ============================
EMBEDDING_RECORD_STANDARD = {
    'required_fields': ['id', 'text_hash', 'emotions.topk'],  # 필수 필드
    'preferred_fields': ['trace_id', 'text', 'meta.trace_id'],  # 선호 필드
    'fallback_fields': ['sample_id', 'record.id', 'record.text'],  # 폴백 필드
    'validation': {
        'id_format': 'string',  # ID 형식 검증
        'text_hash_length': 64,  # SHA256 해시 길이
        'emotions_topk_min': 4,  # 최소 감정 개수
    }
}

# ============================
# CONTEXT_ANALYZER_CONFIG
# ============================

CONTEXT_ANALYZER_CONFIG = {
    'cache_size': 128,
    'memory_limit': 500 * 1024 * 1024,
    'thresholds': {
        'minimum_context_length': 10,
        'confidence': 0.5
    },
    'weights': {
        'keyword_match': 1.0,
        'pattern_match': 1.0,
        'context_relevance': 1.0,
        'temporal_coherence': 1.0
    }
}

# ============================
# 대표 감정 및 서브 감정 구조(리스트)
#  - 실제 서브 사전/패턴은 EMOTIONS.JSON에서 읽고,
#    아래 목록은 라벨 디렉토리 탐색/검증 및 매핑 생성에 사용
# ============================

MAIN_EMOTIONS = {
    '희': [
        '만족감', '환희', '자긍심', '행복', '긍정적', '희열', '열정', '감격', '감사함', '유쾌함',
        '활력', '자신감', '충만함', '평온함', '낙관적', '기대감', '용기', '성취감', '따뜻함', '우정',
        '희망적', '기쁨', '자유로움', '충족감', '영감', '다정함', '성장감', '자부심', '호기심', '응원'
    ],
    '노': [
        '분노', '불쾌감', '적개심', '불만', '원한', '분개', '질투', '공격성', '조바심', '냉담함',
        '반감', '실망감', '방어적', '적대감', '짜증', '성가심', '경멸', '배신감', '불안정', '억울함',
        '증오', '분함', '도전적', '복수심', '경계심', '긴장감', '모욕감', '불평', '거부감', '좌절감'
    ],
    '애': [
        '슬픔', '비통함', '상실감', '고독', '외로움', '애도', '허무함', '후회', '낙담', '우울함',
        '동정심', '자책감', '연민', '혼란스러움', '절망감', '멍함', '비탄', '상심', '애처로움', '힘겨움',
        '아쉬움', '그리움', '고립감', '무기력', '소외감', '눈물', '허탈감', '패배감', '실의', '안타까움'
    ],
    '락': [
        '즐거움', '해방감', '놀이', '신남', '자유로움', '희열감', '동료애', '모험심', '편안함', '흥미로움',
        '자발성', '쾌활함', '감동', '기분좋음', '농담', '여유로움', '낙천적', '안락함', '행복감', '친밀감',
        '흥분', '열린 마음', '재미', '평화로움', '동지애', '기대감', '감탄스러움', '활기', '안심', '정적'
    ]
}

# ============================
# 라벨 매핑 생성
# ============================

ID_SEP = "-"  # 식별자 구분자(하이픈으로 통일)

# EMOTIONS 데이터 로드 (EMOTIONS_JSON_PATH 우선, 없으면 폴백) - 전역 캐시 적용
from functools import lru_cache

# 첫 로드 여부를 추적하는 전역 변수
_EMOTIONS_FIRST_LOAD = True

@lru_cache(maxsize=1)
def _load_emotions_cached(path: str) -> dict:
    """EMOTIONS.json 파일을 LRU 캐시로 로드 (프로세스 내 1회만 읽기)"""
    global _EMOTIONS_FIRST_LOAD
    EMOTIONS = {}
    if path and os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                EMOTIONS = json.load(f)
            # 첫 로드 시에만 로그 출력
            if _EMOTIONS_FIRST_LOAD:
                logger.info(f"[config] EMOTIONS 데이터 로드 완료: {path}")
                _EMOTIONS_FIRST_LOAD = False
        except Exception as e:
            logger.warning(f"[config] EMOTIONS 데이터 로드 실패: {e}")
            EMOTIONS = {}
    return EMOTIONS

def _load_emotions_cached_wrapper():
    """EMOTIONS_JSON_PATH를 사용한 래퍼 함수"""
    return _load_emotions_cached(str(EMOTIONS_JSON_PATH) if EMOTIONS_JSON_PATH else "")

# 전역 모델/토크나이저 캐시 (중복 로드 방지)
_HF_MODEL_CACHE = {}
_HF_TOKENIZER_CACHE = {}

@lru_cache(maxsize=8)
def get_hf_model(model_name: str):
    """HuggingFace 모델을 전역 캐시로 로드 (중복 로드 방지)"""
    if model_name in _HF_MODEL_CACHE:
        return _HF_MODEL_CACHE[model_name]
    
    try:
        from transformers import AutoModel
        model = AutoModel.from_pretrained(model_name)
        _HF_MODEL_CACHE[model_name] = model
        logger.info(f"[config] 모델 캐시에 로드 완료: {model_name}")
        return model
    except Exception as e:
        logger.warning(f"[config] 모델 로드 실패: {model_name} - {e}")
        return None

@lru_cache(maxsize=8)
def get_hf_tokenizer(model_name: str):
    """HuggingFace 토크나이저를 전역 캐시로 로드 (중복 로드 방지)"""
    if model_name in _HF_TOKENIZER_CACHE:
        return _HF_TOKENIZER_CACHE[model_name]
    
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        _HF_TOKENIZER_CACHE[model_name] = tokenizer
        logger.info(f"[config] 토크나이저 캐시에 로드 완료: {model_name}")
        return tokenizer
    except Exception as e:
        logger.warning(f"[config] 토크나이저 로드 실패: {model_name} - {e}")
        return None

EMOTIONS = _load_emotions_cached_wrapper()

# EMOTIONS가 비어있으면 추가 경로 시도
if not EMOTIONS:
    # 추가 경로 시도
    additional_paths = [
        PROJECT_ROOT / "src" / "EMOTIONS.json",
        PROJECT_ROOT / "EMOTIONS.json",
        Path("src/EMOTIONS.json"),
        Path("EMOTIONS.json"),
    ]
    
    for path in additional_paths:
        if path.is_file():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    EMOTIONS = json.load(f)
                logger.info(f"[config] EMOTIONS 데이터 로드 완료 (추가 경로): {path}")
                EMOTIONS_JSON_PATH = path  # 경로 업데이트
                # 캐시 업데이트
                _EMOTIONS_CACHE = EMOTIONS
                _EMOTIONS_CACHE_PATH = EMOTIONS_JSON_PATH
                break
            except Exception as e:
                logger.debug(f"[config] 추가 경로 로드 실패 {path}: {e}")
                continue
    
    if not EMOTIONS:
        logger.warning(f"[config] EMOTIONS.json 파일을 찾을 수 없음. 시도한 경로: {[str(p) for p in additional_paths]}")
        # 최소한의 폴백 구조 생성
        EMOTIONS = {
            "희": {
                "metadata": {"primary_category": "희", "emotion_id": "희", "version": "1.0"},
                "sub_emotions": {}
            },
            "노": {
                "metadata": {"primary_category": "노", "emotion_id": "노", "version": "1.0"},
                "sub_emotions": {}
            },
            "애": {
                "metadata": {"primary_category": "애", "emotion_id": "애", "version": "1.0"},
                "sub_emotions": {}
            },
            "락": {
                "metadata": {"primary_category": "락", "emotion_id": "락", "version": "1.0"},
                "sub_emotions": {}
            }
        }
        # 캐시 업데이트
        _EMOTIONS_CACHE = EMOTIONS
        _EMOTIONS_CACHE_PATH = EMOTIONS_JSON_PATH

SUB_EMOTION_TO_LABEL: Dict[str, str] = {}
LABEL_TO_SUB_EMOTION: Dict[str, str] = {}
# 교차 메인 동명이인 충돌 방지용 맵
LABEL_TO_SUB_EMOTION_BY_MAIN: Dict[str, Dict[str, str]] = {}


def register_label_mapping(main_emotion: str, sub_label: str, emotion_id: str):
    global LABEL_TO_SUB_EMOTION, LABEL_TO_SUB_EMOTION_BY_MAIN
    # 하위호환(기존 딕셔너리) 유지하되 충돌 시 경고
    if sub_label in LABEL_TO_SUB_EMOTION and LABEL_TO_SUB_EMOTION[sub_label] != emotion_id:
        # 로컬 개발 환경에서는 경고 억제 (불필요한 경고)
        if RENDER_DEPLOYMENT:
            logging.warning(f"Duplicate sub label '{sub_label}' across mains; prefer BY_MAIN resolver.")
    LABEL_TO_SUB_EMOTION[sub_label] = emotion_id
    LABEL_TO_SUB_EMOTION_BY_MAIN.setdefault(main_emotion, {})[sub_label] = emotion_id


def resolve_sub_id(main_emotion: str, sub_label: str) -> Optional[str]:
    return LABEL_TO_SUB_EMOTION_BY_MAIN.get(main_emotion, {}).get(sub_label)


def _init_minimal_emotions(main_emotions: dict,
                           emotions: dict,
                           sub_to_label: dict,
                           label_to_sub: dict) -> None:
    """라벨/EMOTIONS.JSON 부재 시에도 축소 모드로 기동(Unknown 서브 자동 구성)."""
    for main in main_emotions.keys():
        emotions.setdefault(main, {'label': main, 'sub_emotions': {}})
        unknown_id = f"{main}{ID_SEP}unknown"
        emotions[main]['sub_emotions'][unknown_id] = {
            'sub_category': 'Unknown', 'intensity_levels': {}, 'keywords': [],
            'examples': [], 'contexts': {},
            'related_emotions': {'positive': [], 'negative': [], 'neutral': []},
            'emotion_transitions': {'patterns': []},
            'sentiment_analysis': {'positive_indicators': [], 'negative_indicators': [],
                                   'intensity_modifiers': {'amplifiers': [], 'diminishers': []}}
        }
        sub_to_label[unknown_id] = 'Unknown'
        register_label_mapping(main, 'Unknown', unknown_id)
        # 비충돌 키 별칭 유지
        label_to_sub[f"{main}::Unknown"] = unknown_id


def create_emotion_label_mappings() -> Dict[str, Any]:
    """
    LABEL_DIR/{main}/{sub}.json 을 읽어 서브 감정 매핑을 구성.
    - emotion_id 생성 규칙을 'main-sub' 로 **단일화**(이중 접두어 제거)
    - Unknown 서브 감정을 main마다 준비
    """
    mappings = {
        'main_to_sub': {},
        'sub_to_main': {},
        'id_to_emotion': {},
        'emotion_to_id': {}
    }

    # LABEL_DIR이 없을 때도 최소 맵 구성 보장
    if not LABEL_DIR:
        _init_minimal_emotions(MAIN_EMOTIONS, EMOTIONS, SUB_EMOTION_TO_LABEL, LABEL_TO_SUB_EMOTION)
        for main_emotion in MAIN_EMOTIONS.keys():
            unknown_id = f"{main_emotion}{ID_SEP}unknown"
            mappings['main_to_sub'].setdefault(main_emotion, []).append(unknown_id)
            mappings['sub_to_main'][unknown_id] = main_emotion
            mappings['id_to_emotion'][unknown_id] = {'main': main_emotion, 'sub': 'Unknown'}
            mappings['emotion_to_id'][f"{main_emotion}_{unknown_id}"] = unknown_id
        return mappings

    for main_emotion, sub_emotions_list in MAIN_EMOTIONS.items():
        main_emotion_path = os.path.join(LABEL_DIR, main_emotion)
        if not os.path.isdir(main_emotion_path):
            # 로컬 개발 환경에서는 경고 억제 (불필요한 경고)
            if RENDER_DEPLOYMENT:
                logging.warning(f"Main emotion folder '{main_emotion_path}'가 존재하지 않습니다.")
            continue

        EMOTIONS[main_emotion] = {'label': main_emotion, 'sub_emotions': {}}

        # Unknown 서브
        unknown_id = f"{main_emotion}{ID_SEP}unknown"
        EMOTIONS[main_emotion]['sub_emotions'][unknown_id] = {
            'sub_category': 'Unknown',
            'intensity_levels': {},
            'keywords': [],
            'examples': [],
            'contexts': {},
            'related_emotions': {'positive': [], 'negative': [], 'neutral': []},
            'emotion_transitions': {'patterns': []},
            'sentiment_analysis': {
                'positive_indicators': [],
                'negative_indicators': [],
                'intensity_modifiers': {'amplifiers': [], 'diminishers': []}
            }
        }
        SUB_EMOTION_TO_LABEL[unknown_id] = 'Unknown'
        register_label_mapping(main_emotion, 'Unknown', unknown_id)
        mappings['main_to_sub'].setdefault(main_emotion, []).append(unknown_id)
        mappings['sub_to_main'][unknown_id] = main_emotion
        mappings['id_to_emotion'][unknown_id] = {'main': main_emotion, 'sub': 'Unknown'}
        mappings['emotion_to_id'][f"{main_emotion}_{unknown_id}"] = unknown_id

        # 정의된 서브 항목들
        for sub_emotion in sub_emotions_list:
            sub_emotion_file = f"{sub_emotion}.json"
            sub_emotion_path = os.path.join(main_emotion_path, sub_emotion_file)
            if not os.path.isfile(sub_emotion_path):
                logging.error(f"서브 감정 파일 '{sub_emotion_file}'을 찾을 수 없습니다.")
                continue

            try:
                with open(sub_emotion_path, 'r', encoding='utf-8') as f:
                    sub_emotion_data = json.load(f)

                sub_emotion_label = sub_emotion_data.get('metadata', {}).get('sub_category')
                if not sub_emotion_label:
                    logging.error(f"'{sub_emotion_file}' 서브 감정에 'sub_category' 키가 없습니다.")
                    continue

                # **중복 없는** 표준 emotion_id 생성: "main-sub"
                emotion_id = f"{main_emotion}{ID_SEP}{sub_emotion_label}"
                if emotion_id in LABEL_TO_SUB_EMOTION:
                    logging.error(f"서브 감정 레이블 '{emotion_id}'이 중복되었습니다.")
                    continue

                EMOTIONS[main_emotion]['sub_emotions'][emotion_id] = {
                    'sub_category': sub_emotion_label,
                    'intensity_levels': sub_emotion_data.get('emotion_profile', {}).get('intensity_levels', {}),
                    'keywords': sub_emotion_data.get('emotion_profile', {}).get('core_keywords', []),
                    'examples': sub_emotion_data.get('context_patterns', {}).get('examples', []),
                    'contexts': sub_emotion_data.get('context_patterns', {}),
                    'related_emotions': sub_emotion_data.get('emotion_profile', {}).get('related_emotions', {
                        'positive': [], 'negative': [], 'neutral': []
                    }),
                    'emotion_transitions': sub_emotion_data.get('emotion_transitions', {'patterns': []}),
                    'sentiment_analysis': sub_emotion_data.get('sentiment_analysis', {
                        'positive_indicators': [],
                        'negative_indicators': [],
                        'intensity_modifiers': {'amplifiers': [], 'diminishers': []}
                    })
                }

                SUB_EMOTION_TO_LABEL[emotion_id] = sub_emotion_label
                register_label_mapping(main_emotion, sub_emotion_label, emotion_id)

                mappings['main_to_sub'].setdefault(main_emotion, []).append(emotion_id)
                mappings['sub_to_main'][emotion_id] = main_emotion
                mappings['id_to_emotion'][emotion_id] = {'main': main_emotion, 'sub': sub_emotion_label}
                mappings['emotion_to_id'][f"{main_emotion}_{emotion_id}"] = emotion_id

            except json.JSONDecodeError as e:
                logging.error(f"JSON decode error in file {sub_emotion_path}: {e}")
            except Exception as e:
                logging.error(f"Error reading file {sub_emotion_path}: {e}")

    return mappings


EMOTION_MAPPINGS = create_emotion_label_mappings()

# 라벨/JSON 부재 시 축소 모드로 기동
if not EMOTIONS:
    # 프로덕션 환경에서는 폴백 사용 시 즉시 실패
    is_production = EA_PROFILE == "prod" or RENDER_DEPLOYMENT or os.getenv("PRODUCTION_MODE", "0") == "1"
    
    if is_production:
        error_msg = f"EMOTIONS.json not found or invalid in production mode. Path: {EMOTIONS_JSON_PATH}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    
    # 개발 환경에서만 폴백 허용
    logger.warning("EMOTIONS 딕셔너리가 비어 있어 minimal fallback으로 전환합니다.")
    _init_minimal_emotions(MAIN_EMOTIONS, EMOTIONS, SUB_EMOTION_TO_LABEL, LABEL_TO_SUB_EMOTION)

# 표준 라벨·인덱스
EMOTION_TO_LABEL = {k: k for k in EMOTIONS.keys()}
LABEL_TO_EMOTION = {v: k for k, v in EMOTION_TO_LABEL.items()}
EMOTION_LABELS = list(EMOTION_TO_LABEL.keys())

# 재현성을 위해 정렬(기존: dict 순서 → 환경에 따라 달라질 수 있음)
SUB_EMOTION_LABELS = sorted(SUB_EMOTION_TO_LABEL.keys())

sub_emotion_to_index = {label: idx for idx, label in enumerate(SUB_EMOTION_LABELS)}
index_to_sub_emotion = {idx: label for label, idx in sub_emotion_to_index.items()}
num_sub_classes = len(SUB_EMOTION_LABELS)

SCORE_TO_INTENSITY = {1: 'low', 2: 'medium', 3: 'high'}
intensity_to_index = {v: (k - 1) for k, v in SCORE_TO_INTENSITY.items()}
index_to_intensity = {(k - 1): v for k, v in SCORE_TO_INTENSITY.items()}
num_intensity_classes = len(SCORE_TO_INTENSITY)

# ============================
# 모듈 엔트리포인트(누락 보완)
# ============================
MODULE_ENTRYPOINTS = {
    "pattern_extractor": ("pattern_extractor", "run_pattern_extraction"),  # 함수형 엔트리포인트
    "emotion_classification": ("linguistic_matcher", "LinguisticMatcher"),  # 메인 감정 분류
    "intensity_analysis": ("intensity_analyzer", "analyze_intensity"),  # 함수형 엔트리포인트로 변경
    "embedding_generation": ("intensity_analyzer", "analyze_intensity"),  # 함수형 엔트리포인트로 변경
    "context_analysis": ("context_extractor", "ContextExtractor"),
    "sub_emotion_detection": ("linguistic_matcher", "LinguisticMatcher"),  # 서브 감정 탐지
    "transition_analyzer": ("transition_analyzer", "TransitionAnalyzer"),
    "linguistic_matcher": ("linguistic_matcher", "LinguisticMatcher"),  # 언어 패턴 매칭
    "weight_calculator": ("weight_calculator", "EmotionWeightCalculator"),
    "complex_analyzer": ("complex_analyzer", "ComplexEmotionAnalyzer"),
    "time_series_analyzer": ("time_series_analyzer", "run_time_series_analysis"),  # 함수형 엔트리포인트
    "situation_analyzer": ("situation_analyzer", "SituationAnalyzer"),
    "psychological_analyzer": ("psychological_analyzer", "PsychCogIntegratedAnalyzer"),
    "relationship_analyzer": ("emotion_relationship_analyzer", "EmotionRelationshipAnalyzer"),
}

# ============================
# ANALYZER_CONFIG
# ============================
ANALYZER_CONFIG = {
    "contrast_turn_cfg": {
        "both_base": 0.22, "both_scale": 0.12, "both_max": 0.40,
        "single_base": 0.14, "single_scale": 0.10, "single_max": 0.30
    },
    "context_bias_positive_terms": ["칭찬", "축하", "기뻤", "기쁜", "행복", "만족", "성취"],
    "context_bias_negative_terms": ["피곤", "피곤함", "지침", "아프", "아픈", "힘들", "불안", "걱정", "우울"],
    "paragraph_top_k": 3,
    "paragraph_boost_alpha": 0.10,
    "conflict_min_intensity": 0.35,
    "conflict_weaken_factor": 0.85,
    # "feature_weights": {...}  # 필요 시 오버라이드

    # ▼▼▼▼▼ [여기에 추가] ▼▼▼▼▼
    "connectives": {
        "cause": ["~로 인하여", "덕분에"],
        "contrast": ["반대로", "대신에"],
        "sequence": ["그런 다음", "이윽고"],
    },
    "connective_scores": {"cause": 0.48, "contrast": 0.42, "sequence": 0.36},
}

# ============================
# 임계값 단일 네임스페이스
# ============================
THRESHOLDS = {
    'transition': 0.4,
    'main_confidence': 0.10,
    'sub_confidence': 0.15,
    'similarity': 0.30,
    'sub_min_similarity': 0.30,
    'unknown': 0.20,
    # 개선사항: 전이 검출 임계 상향 (0.4 -> 0.6)
    'transition_confidence': 0.6,  # 전이 신뢰도 최소값
    'transition_min_confidence': 0.6,  # 전이 분석기용
    # 시연 모드 전용 임계값 추가
    'reject_fallback_under': REJECT_FALLBACK_UNDER,
    'require_min_conf': REQUIRE_MIN_CONF,
    'sub_min_post': SUB_MIN_POST,
}

# [개선사항 2] 라벨 최종 가드 중앙집중화
LABELING_GUARDS = {
    "min_conf": float(os.getenv("LBL_MIN_CONF", str(REQUIRE_MIN_CONF))),
    "reject_fallback_under": float(os.getenv("LBL_REJECT_FB_UNDER", str(REJECT_FALLBACK_UNDER))),
    "enable_temp_scaling": ENABLE_TEMP_SCALING,
    "emotion_alias_mode": EMOTION_ALIAS_MODE,
}

# [개선사항 4] 프로파일(DEV/PILOT/PROD)로 임계값/정책 스위칭
if EA_PROFILE == "dev":
    THRESHOLDS.update(main_confidence=0.05, sub_confidence=0.10, unknown=0.30)
    LABELING_GUARDS.update(min_conf=0.45, reject_fallback_under=0.40)
elif EA_PROFILE == "pilot":
    THRESHOLDS.update(main_confidence=0.08, sub_confidence=0.12)
    LABELING_GUARDS.update(min_conf=0.50, reject_fallback_under=0.45)
# prod는 기본값 유지

# ============================
# TRAINING_PARAMS (기존 키 유지)
# - 일부 값은 THRESHOLDS 참조로 일원화
# ============================
TRAINING_PARAMS = {
    'BERT_MODEL_NAME': 'klue/bert-base',
    'MAX_LEN': 512,
    'BATCH_SIZE': 4,
    'LEARNING_RATE': 1e-5,
    'MIN_LR': 1e-7,
    'WEIGHT_DECAY': 5e-3,
    'WARMUP_RATIO': 0.1,
    'WARMUP_STEPS': 2000,
    'NUM_EPOCHS': 15,
    'DROPOUT_RATE': 0.7,
    'EARLY_STOPPING': True,
    'PATIENCE': 2,
    'MIN_DELTA': 0.01,
    'CONTEXT_WINDOW_SIZE': 3,
    'EMOTION_TRANSITION_THRESHOLD': THRESHOLDS['transition'],
    'MAIN_EMOTION_CONFIDENCE_THRESHOLD': THRESHOLDS['main_confidence'],
    'SUB_EMOTION_CONFIDENCE_THRESHOLD': THRESHOLDS['sub_confidence'],
    'SIMILARITY_THRESHOLD': THRESHOLDS['similarity'],
    'SUB_EMOTION_MIN_SIMILARITY': THRESHOLDS['sub_min_similarity'],
    'UNKNOWN_THRESHOLD': THRESHOLDS['unknown'],
    'MIN_CONFIDENCE_FOR_SUB': THRESHOLDS['sub_confidence'],
    'MIN_SEGMENT_LENGTH': 0,
    'EMOTION_FLOW_WINDOW': 5,
    'POOLING_METHOD': 'attention',
    'ATTENTION_HEADS': 8,
    'CONTEXT_EMBEDDING_DIM': 256,
    'EMOTION_EMBEDDING_DIM': 768,
    'NUM_AUGMENTED': 4,
    'MIN_DATA_SAMPLES': 100,
    'K_FOLDS': 5,
    'SEED': 42,
    'SUB_EMOTION_BATCH_SIZE': 16,
    'SUB_EMOTION_LEARNING_RATE': 1e-5,
    'SUB_EMOTION_NUM_EPOCHS': 15,
    'HIDDEN_SIZE': 768,
    'INTERMEDIATE_SIZE': 1024,
    'NUM_ATTENTION_HEADS': 12,
    'NUM_HIDDEN_LAYERS': 4,
    'LOG_LEVEL': os.getenv("TRAINING_LOG_LEVEL", "INFO"),
    'CLASS_WEIGHTS': {'main': True, 'sub': True, 'intensity': True},
    'LOSS_WEIGHTS': {'main': 1.0, 'sub': 2.0, 'intensity': 0.5}
}

# 기존 변수명(하위호환)
MAIN_EMOTION_CONFIDENCE_THRESHOLD = TRAINING_PARAMS['MAIN_EMOTION_CONFIDENCE_THRESHOLD']
SUB_EMOTION_CONFIDENCE_THRESHOLD = TRAINING_PARAMS['SUB_EMOTION_CONFIDENCE_THRESHOLD']
SIMILARITY_THRESHOLD = TRAINING_PARAMS['SIMILARITY_THRESHOLD']

N_CLUSTERS = 2
INTENSITY_THRESHOLDS = {'low': 0.2, 'medium': 0.5, 'high': 0.7}
CONTEXT_WEIGHTS = {'emotional_keywords': 0.4, 'temporal_context': 0.3, 'situational_context': 0.3}
EMOTION_TRANSITION_PARAMS = {
    'min_transition_score': THRESHOLDS['transition'],  # 단일화
    'context_window': 3,
    'smoothing_factor': 0.2
}

SUB_EMOTION_MATCHING = {
    'min_keyword_match': 1,
    'context_weight': 0.6,
    'keyword_weight': 0.4,
    'fallback_to_unknown_threshold': THRESHOLDS['unknown']  # 단일화
}

# ============================
# EMOTION GUIDELINES
# - SoT: EMOTIONS.JSON
# - 아래 DEFAULTS는 백업/오버라이드 용도
# ============================

EMOTION_RULES_POLICY = {
    "source": os.getenv("EMOTION_RULES_SOURCE", "json"),  # "json" | "config"
    "allow_overrides": True,
    "override_scopes": ["thresholds", "weights", "modifiers"]  # 확장 가능
}

# ---- Defaults (기존 정의 최대한 유지) ----
EMOTION_GUIDELINES_DEFAULTS = {
    'sentiment_analysis': {
        'positive_indicators': [
            "좋다", "행복", "기쁨", "기뻐", "즐거움", "감사", "환희", "만족감", "유쾌함", "긍정적",
            "희열", "열정", "감격", "자긍심", "감사함", "활력", "자신감", "충만함", "평온함", "낙관적",
            "기대감", "용기", "성취감", "따뜻함", "우정", "희망적", "기쁨", "자유로움", "충족감", "영감",
            "다정함", "성장감", "자부심", "호기심", "응원", "활기", "행운", "의욕", "기대", "편안함",
            "희망적", "기쁨의 환희", "충만함", "감격", "사랑스러움", "미소", "소망", "평화", "기운", "성취감",
            "감동", "포근함", "설레임", "긍정적 기대", "기쁨의 파동", "희망의 빛", "즐거운 기대", "환희의 순간",
            "행복한 순간", "즐거움의 확산", "활기찬 하루"
        ],
        'negative_indicators': [
            "나쁘다", "슬픔", "분노", "짜증", "불만", "좌절", "분개", "질투", "공격성", "조바심",
            "냉담함", "반감", "실망감", "방어적", "적대감", "성가심", "경멸", "배신감", "불안정",
            "억울함", "증오", "분함", "도전적", "복수심", "경계심", "긴장감", "모욕감", "불평", "거부감",
            "좌절감", "불쾌감", "적개심", "원한"
        ],
        'intensity_modifiers': {
            'amplifiers': ["매우", "정말", "너무", "굉장히", "진짜", "엄청", "극도로", "대단히", "무척"],
            'diminishers': ["약간", "조금", "다소", "좀", "조금씩", "어느 정도"]
        }
    },
    'linguistic_patterns': {
        'key_phrases': [
            {"pattern": "정말 행복해", "weight": 1.0, "context_requirement": "축하"},
            {"pattern": "너무 슬퍼", "weight": 0.8, "context_requirement": "비극"},
            {"pattern": "매우 분노", "weight": 1.0, "context_requirement": "대립"},
            {"pattern": "조금 우울해", "weight": 0.5, "context_requirement": "일상"},
            {"pattern": "깊은 슬픔", "weight": 0.9, "context_requirement": "비극"},
            {"pattern": "극도의 불만", "weight": 1.0, "context_requirement": "불만족"},
            {"pattern": "상심하다", "weight": 0.7, "context_requirement": "애도"},
            {"pattern": "극심한 분노", "weight": 1.0, "context_requirement": "대립"},
            {"pattern": "조금 실망했어", "weight": 0.4, "context_requirement": "실망"},
            {"pattern": "엄청 불만족스러워", "weight": 1.0, "context_requirement": "불만족"},
            {"pattern": "완전 행복해", "weight": 1.0, "context_requirement": "축하"},
            {"pattern": "소소하게 행복해", "weight": 0.6, "context_requirement": "일상"},
            {"pattern": "매우 불쾌하다", "weight": 0.9, "context_requirement": "대립"},
            {"pattern": "깊은 외로움", "weight": 0.8, "context_requirement": "비극"},
            {"pattern": "좀 괜찮아졌어", "weight": 0.5, "context_requirement": "회복"},
            {"pattern": "완전히 미쳐가", "weight": 1.0, "context_requirement": "대립"},
            {"pattern": "조금 짜증나", "weight": 0.3, "context_requirement": "일상"},
            {"pattern": "마음이 무겁다", "weight": 0.7, "context_requirement": "비극"},
            {"pattern": "너무나도 기쁘다", "weight": 1.0, "context_requirement": "축하"},
            {"pattern": "약간 실망스럽다", "weight": 0.4, "context_requirement": "실망"},
            {"pattern": "마음이 답답하다", "weight": 0.8, "context_requirement": "불만족"},
            {"pattern": "매우 설레", "weight": 0.9, "context_requirement": "기대"},
            {"pattern": "좀 우울해져", "weight": 0.6, "context_requirement": "일상"},
            {"pattern": "심각하게 화났어", "weight": 1.0, "context_requirement": "대립"},
            {"pattern": "기분이 처져", "weight": 0.7, "context_requirement": "일상"},
            {"pattern": "완전히 지쳤어", "weight": 0.8, "context_requirement": "피로"},
            {"pattern": "너무 실망스러워", "weight": 0.9, "context_requirement": "실망"},
            {"pattern": "감동했어", "weight": 0.8, "context_requirement": "감동"},
            {"pattern": "완전히 무너졌어", "weight": 1.0, "context_requirement": "비극"},
            {"pattern": "매우 고마워", "weight": 1.0, "context_requirement": "감사"},
            {"pattern": "깊이 감사해", "weight": 1.0, "context_requirement": "감사"},
            {"pattern": "한없이 기쁘다", "weight": 0.9, "context_requirement": "축하"},
            {"pattern": "충격적이다", "weight": 0.8, "context_requirement": "놀람"},
            {"pattern": "깊은 충격을 받다", "weight": 0.9, "context_requirement": "비극"},
            {"pattern": "긴장돼", "weight": 0.7, "context_requirement": "불안"},
            {"pattern": "어이없다", "weight": 0.6, "context_requirement": "불만족"},
            {"pattern": "편안해졌어", "weight": 0.8, "context_requirement": "안도"},
            {"pattern": "너무 지쳤다", "weight": 0.9, "context_requirement": "피로"},
            {"pattern": "마음이 복잡하다", "weight": 0.7, "context_requirement": "혼란"},
            {"pattern": "어느 정도 만족해", "weight": 0.5, "context_requirement": "만족"},
            {"pattern": "진정한 행복", "weight": 1.0, "context_requirement": "축하"},
            {"pattern": "너무 억울해", "weight": 0.9, "context_requirement": "대립"},
            {"pattern": "기대돼", "weight": 0.7, "context_requirement": "기대"},
            {"pattern": "걱정돼", "weight": 0.6, "context_requirement": "불안"},
            {"pattern": "완전 행복해", "weight": 1.0, "context_requirement": "축하"},
            {"pattern": "매우 서운해", "weight": 0.8, "context_requirement": "실망"},
            {"pattern": "너무 충격적이야", "weight": 0.9, "context_requirement": "비극"},
            {"pattern": "기대보다 나아", "weight": 0.7, "context_requirement": "긍정적"},
            {"pattern": "완전 짜증나", "weight": 1.0, "context_requirement": "대립"},
            {"pattern": "어느 정도 괜찮아", "weight": 0.5, "context_requirement": "회복"},
            {"pattern": "조금 무서워", "weight": 0.4, "context_requirement": "불안"},
            {"pattern": "너무 만족스러워", "weight": 0.9, "context_requirement": "만족"},
            {"pattern": "깊은 감동", "weight": 1.0, "context_requirement": "감동"},
            {"pattern": "마음이 따뜻해", "weight": 0.8, "context_requirement": "긍정적"},
        ],
        'sentiment_combinations': [
            {"words": ["기쁨", "행복"], "weight": 0.8},
            {"words": ["슬픔", "비탄"], "weight": 0.7},
            {"words": ["분노", "짜증"], "weight": 0.75},
            {"words": ["즐거움", "유쾌함"], "weight": 0.85},
            {"words": ["외로움", "고독"], "weight": 0.6},
            {"words": ["허무함", "절망감"], "weight": 0.7},
            {"words": ["적개심", "혐오"], "weight": 0.65},
            {"words": ["유희감", "자유로움"], "weight": 0.75},
            {"words": ["감사함", "기쁨"], "weight": 0.8},
            {"words": ["불만", "좌절"], "weight": 0.7},
            {"words": ["동정심", "연민"], "weight": 0.7},
            {"words": ["감사함", "희열"], "weight": 0.85},
            {"words": ["자부심", "자긍심"], "weight": 0.8},
            {"words": ["연민", "애정"], "weight": 0.7},
            {"words": ["기대감", "용기"], "weight": 0.75},
            {"words": ["희망적", "영감"], "weight": 0.8},
            {"words": ["행복감", "자유로움"], "weight": 0.85},
            {"words": ["성취감", "충만함"], "weight": 0.8},
            {"words": ["기대감", "여유"], "weight": 0.75},
            {"words": ["낙관적", "평화로움"], "weight": 0.8},
            {"words": ["사랑", "감사함"], "weight": 0.85},
            {"words": ["긴장", "불안"], "weight": 0.7},
            {"words": ["질투", "불쾌함"], "weight": 0.75},
            {"words": ["경멸", "혐오"], "weight": 0.8},
            {"words": ["안도", "평온함"], "weight": 0.9},
            {"words": ["흥미", "기대"], "weight": 0.8},
            {"words": ["미소", "따뜻함"], "weight": 0.85},
            {"words": ["좌절", "상실감"], "weight": 0.7},
            {"words": ["슬픔", "후회"], "weight": 0.75},
            {"words": ["무기력", "허탈감"], "weight": 0.8},
            {"words": ["짜증", "분개"], "weight": 0.7},
            {"words": ["반감", "냉담함"], "weight": 0.65},
            {"words": ["열정", "희망"], "weight": 0.85},
            {"words": ["동료애", "협력"], "weight": 0.9},
            {"words": ["감동", "따뜻함"], "weight": 0.85},
            {"words": ["행복", "환희"], "weight": 0.9},
            {"words": ["고립감", "소외감"], "weight": 0.65},
            {"words": ["증오", "복수심"], "weight": 0.75},
            {"words": ["기쁨", "사랑"], "weight": 0.9},
            {"words": ["감격", "자부심"], "weight": 0.85},
            {"words": ["공포", "불안"], "weight": 0.7},
            {"words": ["쾌활함", "자유로움"], "weight": 0.8},
            {"words": ["희망", "용기"], "weight": 0.85},
            {"words": ["우울함", "상심"], "weight": 0.75},
            {"words": ["성가심", "실망"], "weight": 0.7},
            {"words": ["흥분", "기대감"], "weight": 0.85},
            {"words": ["격려", "희망"], "weight": 0.8},
            {"words": ["사랑", "연민"], "weight": 0.75},
            {"words": ["감사", "평화"], "weight": 0.9},
            {"words": ["우정", "따뜻함"], "weight": 0.85},
            {"words": ["비난", "반감"], "weight": 0.7},
            {"words": ["억울함", "불만"], "weight": 0.75},
            {"words": ["비애", "고독"], "weight": 0.8},
            {"words": ["불안", "긴장"], "weight": 0.7},
            {"words": ["친밀감", "사랑"], "weight": 0.9},
            {"words": ["평화", "안락함"], "weight": 0.85},
            {"words": ["성취감", "자긍심"], "weight": 0.85},
            {"words": ["기쁨", "활기"], "weight": 0.8},
            {"words": ["의욕", "기대"], "weight": 0.75},
            {"words": ["절망", "후회"], "weight": 0.7},
            {"words": ["혐오", "적대감"], "weight": 0.75},
            {"words": ["낙천적", "희망"], "weight": 0.8},
            {"words": ["동정", "애정"], "weight": 0.75},
            {"words": ["활기", "열정"], "weight": 0.85},
            {"words": ["조바심", "걱정"], "weight": 0.7},
            {"words": ["고요", "평온"], "weight": 0.85},
            {"words": ["열린 마음", "유쾌함"], "weight": 0.8},
        ],
        'sentiment_modifiers': {
            'amplifiers': ["매우", "정말", "너무", "굉장히", "진짜", "엄청", "극도로", "대단히", "무척"],
            'diminishers': ["약간", "조금", "다소", "좀", "조금씩", "어느 정도"]
        }
    },
    'emotion_transitions': {
        'patterns': [
            {'from_emotion': '슬픔', 'to_emotion': '기쁨',
             'triggers': ['다시 웃다', '기분 전환', '새로운 시작'],
             'transition_analysis': {'trigger_words': ['다시 웃다', '기분 전환'],
                                     'emotion_shift_point': '문장 중간',
                                     'intensity_change': '슬픔에서 기쁨으로 강도 증가'}},
            {'from_emotion': '기쁨', 'to_emotion': '슬픔',
             'triggers': ['실망하다', '좌절하다', '비극적 사건'],
             'transition_analysis': {'trigger_words': ['실망하다', '좌절하다'],
                                     'emotion_shift_point': '문장 끝',
                                     'intensity_change': '기쁨에서 슬픔으로 강도 감소'}},
            {'from_emotion': '분노', 'to_emotion': '슬픔',
             'triggers': ['후회하다', '상실감'],
             'transition_analysis': {'trigger_words': ['후회하다', '상실감'],
                                     'emotion_shift_point': '문장 끝',
                                     'intensity_change': '분노에서 슬픔으로 강도 변화'}},
            {'from_emotion': '슬픔', 'to_emotion': '분노',
             'triggers': ['억울함을 느끼다', '불만을 가지다'],
             'transition_analysis': {'trigger_words': ['억울함을 느끼다', '불만을 가지다'],
                                     'emotion_shift_point': '문장 중간',
                                     'intensity_change': '슬픔에서 분노로 강도 증가'}},
            {'from_emotion': '분노', 'to_emotion': '기쁨',
             'triggers': ['갈등 해결', '이해를 하다'],
             'transition_analysis': {'trigger_words': ['갈등 해결', '이해를 하다'],
                                     'emotion_shift_point': '문장 끝',
                                     'intensity_change': '분노에서 기쁨으로 강도 감소'}},
            {'from_emotion': '기쁨', 'to_emotion': '분노',
             'triggers': ['이해받지 못하다', '기대 실패'],
             'transition_analysis': {'trigger_words': ['이해받지 못하다', '기대 실패'],
                                     'emotion_shift_point': '문장 중간',
                                     'intensity_change': '기쁨에서 분노로 강도 증가'}},
            {'from_emotion': '불안', 'to_emotion': '평온',
             'triggers': ['안심하다', '긴장을 풀다'],
             'transition_analysis': {'trigger_words': ['안심하다', '긴장을 풀다'],
                                     'emotion_shift_point': '문장 끝',
                                     'intensity_change': '불안에서 평온으로 강도 감소'}},
            {'from_emotion': '평온', 'to_emotion': '불안',
             'triggers': ['걱정되다', '불안해지다'],
             'transition_analysis': {'trigger_words': ['걱정되다', '불안해지다'],
                                     'emotion_shift_point': '문장 중간',
                                     'intensity_change': '평온에서 불안으로 강도 증가'}},
            {'from_emotion': '불안', 'to_emotion': '분노',
             'triggers': ['더는 못 참다', '위협에 대응', '분노가 치밀어 오르다'],
             'transition_analysis': {'trigger_words': ['더는 못 참다', '위협에 대응', '분노가 치밀어 오르다'],
                                     'emotion_shift_point': '문장 중간',
                                     'intensity_change': '불안에서 분노로 강도 증가'}},
            {'from_emotion': '분노', 'to_emotion': '불안',
             'triggers': ['이후가 걱정되다', '복수 후에 불안감', '판단이 흔들리다'],
             'transition_analysis': {'trigger_words': ['이후가 걱정되다', '불안감', '흔들리다'],
                                     'emotion_shift_point': '문장 끝',
                                     'intensity_change': '분노에서 불안으로 강도 변화'}},
            {'from_emotion': '슬픔', 'to_emotion': '불안',
             'triggers': ['앞날이 막막하다', '점점 초조해지다'],
             'transition_analysis': {'trigger_words': ['막막하다', '초조해지다'],
                                     'emotion_shift_point': '문장 중간',
                                     'intensity_change': '슬픔에서 불안으로 강도 상승'}},
            {'from_emotion': '불안', 'to_emotion': '슬픔',
             'triggers': ['모든 것이 실패하다', '절망감이 커지다'],
             'transition_analysis': {'trigger_words': ['실패하다', '절망감'],
                                     'emotion_shift_point': '문장 끝',
                                     'intensity_change': '불안에서 슬픔으로 강도 증가'}},
            {'from_emotion': '기쁨', 'to_emotion': '평온',
             'triggers': ['안정된 상태로 가라앉다', '행복의 여운이 잔잔해지다'],
             'transition_analysis': {'trigger_words': ['안정된 상태', '여운이 잔잔해지다'],
                                     'emotion_shift_point': '문장 중간',
                                     'intensity_change': '기쁨에서 평온으로 감정 완화'}},
            {'from_emotion': '평온', 'to_emotion': '기쁨',
             'triggers': ['들뜬 기분', '잔잔한 행복이 커지다'],
             'transition_analysis': {'trigger_words': ['들뜬 기분', '행복이 커지다'],
                                     'emotion_shift_point': '문장 끝',
                                     'intensity_change': '평온에서 기쁨으로 강도 상승'}},
            {'from_emotion': '분노', 'to_emotion': '평온',
             'triggers': ['마음을 가라앉히다', '상황이 해결되다', '타협점 찾다'],
             'transition_analysis': {'trigger_words': ['가라앉히다', '해결되다', '타협점'],
                                     'emotion_shift_point': '문장 끝',
                                     'intensity_change': '분노에서 평온으로 강도 감소'}},
            {'from_emotion': '평온', 'to_emotion': '분노',
             'triggers': ['갑작스런 도발', '억울한 상황 발견', '원인 모를 짜증'],
             'transition_analysis': {'trigger_words': ['도발', '억울한', '짜증'],
                                     'emotion_shift_point': '문장 중간',
                                     'intensity_change': '평온에서 분노로 강도 급상승'}},
        ]
    }
}


def _load_guidelines_from_json(path: Optional[Path]) -> Optional[dict]:
    """
    EMOTIONS.JSON에서 전역 가이드라인 섹션을 탐색해 로드.
    - 실제 스키마에 따라 최상위 키가 다를 수 있으므로, 존재 시 그대로 사용.
    - 불명확하면 None 반환(기본값 사용).
    """
    if not path or not path.is_file():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # 후보 키들 중 하나라도 있으면 해당 블록 사용
        candidates = ["EMOTION_GUIDELINES", "guidelines", "emotion_guidelines"]
        for k in candidates:
            if isinstance(data.get(k), dict):
                return data[k]
        # 직접 필수 섹션이 최상위에 존재하는 경우
        if all(isinstance(data.get(k), dict) for k in
               ["sentiment_analysis", "linguistic_patterns", "emotion_transitions"]):
            return {
                "sentiment_analysis": data["sentiment_analysis"],
                "linguistic_patterns": data["linguistic_patterns"],
                "emotion_transitions": data["emotion_transitions"],
            }
    except Exception as e:
        logging.warning(f"EMOTIONS.JSON 가이드라인 로드 실패: {e}")
    return None


_JSON_GUIDELINES = _load_guidelines_from_json(EMOTIONS_JSON_PATH)
if EMOTION_RULES_POLICY["source"] == "json" and _JSON_GUIDELINES:
    EMOTION_GUIDELINES = _JSON_GUIDELINES if not EMOTION_RULES_POLICY.get("allow_overrides", True) \
        else _deep_merge(_JSON_GUIDELINES, EMOTION_GUIDELINES_DEFAULTS)
    GUIDELINES_SOURCE_USED = "json"
else:
    EMOTION_GUIDELINES = EMOTION_GUIDELINES_DEFAULTS
    GUIDELINES_SOURCE_USED = "config-defaults"


# ============================
# 버전/서명
# ============================

def _compute_emotion_set_version(path: Optional[Path]) -> str:
    if path and path.is_file():
        try:
            return _sha256_of_bytes(path.read_bytes())
        except Exception:
            return "NA"
    # 라벨 파일들의 서브셋으로 해시 대체
    seed = _sha256_of_obj(EMOTION_MAPPINGS)
    return seed


EMOTION_SET_VERSION = _compute_emotion_set_version(EMOTIONS_JSON_PATH)

# 설정 스키마 버전(스키마 진화 추적)
CONFIG_SCHEMA_VERSION = "v2.2"

CONFIG_SIGNATURE = _sha256_of_obj({
    "schema": CONFIG_SCHEMA_VERSION,
    "MODULE_ENTRYPOINTS": MODULE_ENTRYPOINTS,
    "TRAINING_PARAMS": TRAINING_PARAMS,
    "EMBEDDING_PARAMS": {k: EMBEDDING_PARAMS[k] for k in (
        "model_name", "pooling_strategy", "similarity_metric", "embedding_dim", "device"
    )},
    "ANALYZER_CONFIG": ANALYZER_CONFIG,
    "THRESHOLDS": THRESHOLDS,
    "GUIDELINES_SOURCE": GUIDELINES_SOURCE_USED
})

# ============================
# 최종 로깅 레벨 정합(TRAINING_PARAMS 반영)
# ============================
if os.getenv("CONFIG_INIT_LOGGING", "0") == "1":
    try:
        _lvl = getattr(logging, str(TRAINING_PARAMS.get("LOG_LEVEL", "INFO")).upper(), logging.INFO)
        logging.getLogger().setLevel(_lvl)
        for h in logging.getLogger().handlers:
            h.setLevel(_lvl)
    except Exception:
        pass

# ===== [2] =====
# 아래 섹션은 [1]에서 정의한 공통 상수/유틸(THRESHOLDS, INTENSITY_THRESHOLDS, CONTEXT_WEIGHTS,
# EMOTIONS_JSON_PATH, EMOTION_MAPPINGS, EMOTION_GUIDELINES, TRAINING_PARAMS, EMBEDDING_PARAMS,
# ZERO_SHOT_MODEL_NAME, _deep_merge 등)을 그대로 활용하여
# 중복/불일치를 줄이고, SoT(EMOTIONS.JSON) 우선 로딩 정책에 맞게 개선했습니다.


# -----------------------------
# JSON 로더(키 후보를 순회 탐색)
# -----------------------------
def _load_json_key(path, keys) -> Optional[dict]:
    if not path or not getattr(path, "is_file", lambda: False)():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for k in keys:
            if isinstance(data.get(k), dict):
                return data[k]
        return None
    except Exception as e:
        logging.warning(f"[config] JSON key 로드 실패(keys={keys}): {e}")
        return None


# ============================
# EMOTION_METRICS (일원화)
#  - intensity_scale: INTENSITY_THRESHOLDS로부터 파생
#  - confidence_thresholds: THRESHOLDS 참조
#  - context_weights: CONTEXT_WEIGHTS 참조
# ============================

# INTENSITY_THRESHOLDS = {'low': 0.2, 'medium': 0.5, 'high': 0.7}
_low = float(INTENSITY_THRESHOLDS['low'])
_med = float(INTENSITY_THRESHOLDS['medium'])
_high = float(INTENSITY_THRESHOLDS['high'])

EMOTION_METRICS = {
    'intensity_scale': {
        'low': (0.0, _med),
        'medium': (_med, _high),
        'high': (_high, 1.0)
    },
    'confidence_thresholds': {
        'main_emotion': float(THRESHOLDS['main_confidence']),
        'sub_emotion': float(THRESHOLDS['sub_confidence']),
        'intensity': float(INTENSITY_THRESHOLDS['medium'])
    },
    'context_weights': dict(CONTEXT_WEIGHTS)
}

# ============================
# RELATED_EMOTIONS (SoT: JSON → 기본값 병합)
# ============================

RELATED_EMOTIONS_DEFAULTS = {
    "축하": [
        "만족감", "환희", "자긍심", "행복", "긍정적", "희열", "열정", "감격", "감사함", "유쾌함",
        "활력", "자신감", "충만함", "평온함", "낙관적", "기대감", "용기", "성취감", "따뜻함", "우정",
        "희망적", "기쁨", "자유로움", "충족감", "영감", "다정함", "성장감", "자부심", "호기심", "응원",
        "기쁨의 눈물", "긍정의 기운", "환호", "찬사", "뿌듯함", "긍지", "의기양양함", "사기 충전", "축복받음", "기념"
    ],
    "비극": [
        "슬픔", "비통함", "상실감", "고독", "외로움", "애도", "허무함", "후회", "낙담", "우울함",
        "동정심", "자책감", "연민", "혼란스러움", "절망감", "멍함", "비탄", "상심", "애처로움", "힘겨움",
        "아쉬움", "그리움", "고립감", "무기력", "소외감", "눈물", "허탈감", "패배감", "실의", "안타까움",
        "허탈함", "쓸쓸함", "상처", "좌절", "애잔함", "쓰라림", "고통", "실망", "애통", "상실의 아픔"
    ],
    "사랑": [
        "애정", "그리움", "연민", "자유로움", "친밀감", "우정", "감사함", "동정심", "동료애", "다정함",
        "사모함", "헌신", "애착", "애정 어린", "열애", "애틋함", "배려심", "포용", "동경", "공감",
        "친밀한 유대", "서로를 위함", "유대감", "로맨스", "믿음", "귀여움", "설레임", "친애", "돌봄", "연대"
    ],
    "평화": [
        "평온함", "여유로움", "감동", "평화로움", "기분좋음", "안심", "편안함", "안락함", "감탄스러움", "안정감",
        "휴식", "조화", "느긋함", "자연스러움", "잔잔함", "차분함", "화합", "평정", "온화함", "무사",
        "안온함", "조용함", "균형", "정적", "부드러움", "평화로운 분위기", "태평", "명상", "치유", "평온한 기운"
    ],
    "도전": [
        "열망", "목표의식", "자신감", "위험 감수", "열정", "용기", "결단력", "극복 의지", "모험심", "적극적",
        "노력", "경쟁심", "도전 의식", "실험적", "포부", "성취 목표", "확신", "탐험", "극복", "의욕",
        "승리의 열망", "긴장과 설렘", "새로운 시도", "목표 달성", "끈기", "불굴의 의지", "결심", "성장 욕구", "혁신", "실천"
    ],
    "분노": [
        "불쾌감", "분개", "격분", "적대감", "짜증", "공격성", "반발심", "폭발적 감정", "격노", "억울함",
        "불만", "분노", "불평", "좌절감", "성난", "원망", "복수심", "긴장감", "위협", "적의",
        "증오", "분함", "비난", "비판", "항의", "속상함", "불안정한 분노", "감정 폭발", "적개심", "대항심"
    ]
}

_RELATED_EMOTIONS_JSON = _load_json_key(EMOTIONS_JSON_PATH, ["related_emotions_global", "related_emotions"])
RELATED_EMOTIONS = _deep_merge(RELATED_EMOTIONS_DEFAULTS, _RELATED_EMOTIONS_JSON or {})


# ============================
# 매핑 저장(임포트 시 비작동; 필요 시 호출)
# ============================

def save_mappings(save_on_import: bool = False):
    try:
        with open(MAPPING_SAVE_PATH['sub_emotion_to_index'], 'w', encoding='utf-8') as f:
            json.dump(sub_emotion_to_index, f, ensure_ascii=False, indent=4)
        with open(MAPPING_SAVE_PATH['index_to_sub_emotion'], 'w', encoding='utf-8') as f:
            json.dump(index_to_sub_emotion, f, ensure_ascii=False, indent=4)
        with open(MAPPING_SAVE_PATH['intensity_to_index'], 'w', encoding='utf-8') as f:
            json.dump(intensity_to_index, f, ensure_ascii=False, indent=4)
        with open(MAPPING_SAVE_PATH['index_to_intensity'], 'w', encoding='utf-8') as f:
            json.dump(index_to_intensity, f, ensure_ascii=False, indent=4)
        logging.info("[config] 매핑 딕셔너리 저장 완료.")
    except Exception as e:
        logging.error(f"[config] 매핑 딕셔너리 저장 중 오류 발생: {e}")


if os.getenv("SAVE_ON_IMPORT", "0") == "1":
    save_mappings(save_on_import=True)

# ============================
# 경로 모음
# ============================

PATHS = {
    'emotion_structure': os.path.join(BASE_DIR, 'emotion_structure.json'),
    'emotion_mappings': os.path.join(BASE_DIR, 'emotion_mappings.json'),
    'analysis_results': os.path.join(BASE_DIR, 'analysis_results'),
    # [A-2] 로그 루트 ENV 오버라이드
    'logs_root': os.getenv('EMOTION_LOGS_ROOT') or LOG_DIR,
    # 레거시 경로 추가
    'label_dir': LABEL_DIR,
    'log_dir': LOG_DIR,
    'model_dir': MODEL_DIR,
    'embeddings_dir': EMBEDDINGS_DIR,
}

# ============================
# EMOTION_BASE_SCORES (SoT: JSON → 기본값 병합)
#  - 과거 하드코딩을 DEFAULTS로 유지하되 JSON 우선
# ============================

EMOTION_BASE_SCORES_DEFAULTS = {
    # [희]
    "만족감": 1.00, "환희": 1.10, "자긍심": 1.05, "행복": 1.05, "긍정적": 1.00, "희열": 1.10, "열정": 1.05,
    "감격": 1.05, "감사함": 1.05, "유쾌함": 1.00, "활력": 1.00, "자신감": 1.05, "충만함": 1.05, "평온함": 1.00,
    "낙관적": 1.00, "기대감": 1.00, "용기": 1.00, "성취감": 1.05, "따뜻함": 1.00, "우정": 1.00, "희망적": 1.00,
    "기쁨": 1.00, "자유로움": 1.00, "충족감": 1.00, "영감": 1.05, "다정함": 1.00, "성장감": 1.00, "자부심": 1.05,
    "호기심": 1.00, "응원": 1.00,
    # [노]
    "분노": 1.10, "불쾌감": 0.95, "적개심": 1.05, "불만": 0.95, "원한": 1.10, "분개": 1.05, "질투": 0.90,
    "공격성": 1.10, "조바심": 0.90, "냉담함": 0.95, "반감": 0.95, "실망감": 0.90, "방어적": 0.90, "적대감": 1.05,
    "짜증": 0.90, "성가심": 0.90, "경멸": 1.05, "배신감": 1.05, "불안정": 0.95, "억울함": 1.05, "증오": 1.10,
    "분함": 1.00, "도전적": 1.00, "복수심": 1.10, "경계심": 1.00, "긴장감": 0.95, "모욕감": 1.05, "불평": 0.90,
    "거부감": 0.90, "좌절감": 0.95,
    # [애]
    "슬픔": 1.05, "비통함": 1.10, "상실감": 1.10, "고독": 1.00, "외로움": 1.00, "애도": 1.05, "허무함": 1.00,
    "후회": 1.00, "낙담": 1.05, "우울함": 1.00, "동정심": 0.90, "자책감": 1.00, "연민": 1.00, "혼란스러움": 0.90,
    "절망감": 1.10, "멍함": 0.95, "비탄": 1.10, "상심": 1.00, "애처로움": 1.00, "힘겨움": 1.00, "아쉬움": 1.00,
    "그리움": 1.00, "고립감": 1.00, "무기력": 1.00, "소외감": 1.00, "눈물": 1.00, "허탈감": 1.00, "패배감": 1.05,
    "실의": 1.05, "안타까움": 1.00,
    # [락]
    "즐거움": 1.00, "해방감": 1.05, "놀이": 1.00, "신남": 1.00, "자유로움": 1.05, "희열감": 1.10, "동료애": 1.00,
    "모험심": 1.05, "편안함": 1.00, "흥미로움": 1.00, "자발성": 1.00, "쾌활함": 1.00, "감동": 1.05, "기분좋음": 1.00,
    "농담": 1.00, "여유로움": 1.00, "낙천적": 1.00, "안락함": 1.00, "행복감": 1.05, "친밀감": 1.00, "흥분": 1.00,
    "열린 마음": 1.00, "재미": 1.00, "평화로움": 1.00, "동지애": 1.00, "감탄스러움": 1.05, "활기": 1.00, "안심": 1.00,
    "정적": 0.90,
}

_EMOTION_BASE_SCORES_JSON = _load_json_key(EMOTIONS_JSON_PATH, ["emotion_base_scores", "base_scores"])
EMOTION_BASE_SCORES = _deep_merge(EMOTION_BASE_SCORES_DEFAULTS, _EMOTION_BASE_SCORES_JSON or {})

# ============================
# EMOTION_ANALYSIS_PATTERNS (분류/룰 기반 피쳐용 기본 패턴)
#  - JSON 제공 시 병합하여 사용
# ============================

EMOTION_ANALYSIS_PATTERNS_DEFAULTS = {
    '희': {
        'keywords': [
            '기쁘다', '좋다', '행복하다', '즐겁다', '신나다', '날아갈 듯', '뿌듯하다', '만족스럽다', '설레다', '기분 짱',
            '감동', '감격스럽다', '든든하다', '의욕이 생기다', '희망차다', '유쾌하다', '정말 좋다', '웃음이 난다',
            '환희롭다', '흥분된다(긍정)', '싱글벙글', '가슴 벅차다', '살맛 난다', '감사하다', '희열을 느끼다',
            '기분 최고', '미소가 지어진다', '가슴이 두근거리다(행복)', '마음이 따뜻해진다', '희망으로 가득하다',
            '감탄스럽다', '설렘이 가득하다'
        ],
        'negative_patterns': [
            '않다', '못하다', '싫다', '아니다', '실망스럽다', '마음이 어둡다', '감흥이 없다', '기쁨이 사라지다',
            '즐겁지 않다', '행복하지 않다', '웃음이 안 나온다', '흥이 깨지다', '의욕이 사라지다',
            '희망이 꺾이다', '벅참이 식었다', '기쁨이 무색해졌다'
        ],
        'intensity_words': [
            '매우', '너무', '굉장히', '정말', '엄청', '한없이', '상당히', '꽤', '어느 정도', '슬쩍',
            '무척', '비교적', '제법', '약간', '조금', '충분히', '몹시', '가득(히)', '완전히'
        ],
        'threshold': max(THRESHOLDS['main_confidence'], 0.6)  # 기존 의미 보존
    },
    '노': {
        'keywords': [
            '화나다', '짜증', '분노', '열받다', '성질나다', '격분하다', '울화가 치밀다', '버럭', '욱하다', '불쾌하다',
            '분개하다', '원망스럽다', '미치고 환장하겠다', '밉다', '격앙되다', '근질근질하다(화)', '폭발할 것 같다',
            '분통이 터지다', '화가 머리끝까지 치밀다', '욱신욱신거리다(분노)', '속이 뒤집히다', '열이 오른다',
            '짜증 폭발', '신경이 곤두선다', '속이 끓는다', '정말 화가 난다', '분이 안 풀린다'
        ],
        'negative_patterns': [
            '괜찮다', '문제없다', '화가 풀렸다', '진정됐다', '아니다', '분노할 이유가 없다', '짜증 없어', '좀 참을 만하다',
            '거슬리지 않는다', '전혀 화날 것 없다', '참을 수 있다', '마음이 가라앉았다', '분노감이 사라졌다',
            '화가 잦아들었다', '불쾌하지 않다', '짜증스러운 기분이 아니다'
        ],
        'intensity_words': [
            '완전', '극도로', '몹시', '엄청', '심각하게', '매우', '아주', '너무', '진짜', '대단히',
            '지독하게', '격렬하게', '심하게', '불같이', '강렬하게', '조금', '살짝', '약간', '그나마', '보통'
        ],
        'threshold': 0.5
    },
    '애': {
        'keywords': [
            '슬프다', '우울하다', '서글프다', '외롭다', '울적하다', '눈물이 난다', '허무하다', '비통하다', '가슴 아프다',
            '안타깝다', '눈물이 고이다', '침울하다', '낙담하다', '괴롭다', '절망적이다', '상심하다', '그립다',
            '먹먹하다', '스산하다(우울)', '시린 마음', '가슴이 무너지다', '눈시울이 붉어지다', '울컥하다',
            '머리가 멍하다(슬픔)', '속이 허하다', '마음이 쪼그라든다', '공허하다'
        ],
        'negative_patterns': [
            '슬프지 않다', '아니다', '눈물 안 난다', '버틸만하다', '괜찮아졌다', '우울감 사라지다', '의연하다',
            '크게 슬프지 않다', '충분히 견딜 만하다', '눈물이 마르다', '속상하지 않다', '마음이 다잡혔다', '더는 울고 싶지 않다'
        ],
        'intensity_words': [
            '너무', '정말', '대단히', '지독하게', '완전', '매우', '심하게', '잔뜩', '가슴 저릿하게', '살짝', '약간',
            '한없이', '깊이', '몹시', '하염없이', '억누를 수 없을 정도로', '심각하게', '은근히', '서서히', '조금', '어느 정도'
        ],
        'threshold': max(THRESHOLDS['main_confidence'], 0.6)
    },
    '락': {
        'keywords': [
            '즐거움', '해방감', '신난다', '유쾌함', '흐뭇하다', '웃기다', '웃음 터지다', '여유롭다', '반가움', '홀가분',
            '기분 최고', '재미있다', '흥미롭다', '마음이 가볍다', '따뜻한 기분', '마음이 편안하다',
            '신바람 난다', '들뜬다', '들썩들썩거리다(흥)', '깨가 쏟아진다', '유쾌한 농담', '웃음꽃 피다',
            '가슴이 두근(신남)', '활기차다', '맘껏 즐기다', '기쁨이 차오르다', '춤추고 싶다(비유)'
        ],
        'negative_patterns': [
            '따분하다', '지겹다', '지루하다', '즐겁지 않다', '아니다', '우울해진다', '마음이 무겁다', '즐길 기분이 아님',
            '흥이 깨지다', '재미없다', '신나지 않다', '웃음이 나오지 않는다', '심드렁하다', '흥겨움이 사라졌다', '흥미를 못 느낀다'
        ],
        'intensity_words': [
            '매우', '정말', '엄청', '어마어마하게', '진짜', '너무', '조금', '약간', '살짝', '상당히', '제법',
            '한껏', '몹시', '적당히', '꽤나', '아주', '한참', '높게', '강렬하게(신남)', '서서히', '살포시'
        ],
        'threshold': 0.55
    }
}

_EMOTION_ANALYSIS_PATTERNS_JSON = _load_json_key(EMOTIONS_JSON_PATH, ["emotion_analysis_patterns", "patterns"])
EMOTION_ANALYSIS_PATTERNS = _deep_merge(EMOTION_ANALYSIS_PATTERNS_DEFAULTS, _EMOTION_ANALYSIS_PATTERNS_JSON or {})

# ============================
# 가중·임계 (일관화 유지)
# ============================

EMOTION_ANALYSIS_WEIGHTS = {
    'keyword_weight': 0.4,
    'context_weight': 0.3,
    'intensity_weight': 0.3
}

# 하위 호환을 위한 미러(실제 값은 THRESHOLDS 참조)
EMOTION_THRESHOLDS = {
    'main_emotion': THRESHOLDS['main_confidence'],
    'sub_emotion': THRESHOLDS['sub_confidence'],
    'intensity': INTENSITY_THRESHOLDS['medium'],
    'context': 0.15
}

# ============================
# 파이프라인 (라우팅 정책·임계값·의존성 정리)
#  - 새로 추가: emotion_classification 단계에 라우팅 설정(topk/route_policy)
#  - run_if 임계값을 THRESHOLDS로 일원화
# ============================

EMOTION_ANALYSIS_PIPELINE = {
    'steps': [
        {
            'name': 'text_preprocessing',
            'enabled': True,
            'dependencies': [],
            'run_if': [],
            'config': {
                'preprocessing': {
                    'text_cleaning': {
                        'remove_special_chars': True,
                        'normalize_whitespace': True,
                        'remove_urls': True
                    },
                    'tokenization': {
                        'max_length': TRAINING_PARAMS.get('MAX_LEN', 512),
                        'truncation': True,
                        'padding': True
                    }
                }
            }
        },
        {
            'name': 'pattern_extractor',
            'enabled': True,
            'dependencies': ['text_preprocessing'],
            'run_if': [],
            'config': {}
        },
        {
            'name': 'emotion_classification',  # Stage A: 메인(희/노/애/락) 라우팅
            'enabled': True,
            'dependencies': [],  # 독립 실행으로 변경
            'run_if': [],
            'config': {
                'task': 'main_routing',
                'multi_label': True,
                'confidence_threshold': THRESHOLDS['main_confidence'],
                'topk': 2,
                'route_policy': 'topk_if_uncertain',  # top1 확실→단일, 아니면 top2 병렬
                'uncertainty_key': 'entropy',  # 엔진이 지원 시
                'outputs': {
                    'topk_main_key': 'router.topk_main',
                    'uncertainty_key': 'router.uncertainty'
                }
            }
        },
        {
            'name': 'intensity_analysis',
            'enabled': True,
            'dependencies': [],  # 독립 실행으로 변경
            'run_if': [],
            'config': {
                'thresholds': {
                    'low': INTENSITY_THRESHOLDS['low'],
                    'medium': INTENSITY_THRESHOLDS['medium'],
                    'high': INTENSITY_THRESHOLDS['high']
                }
            }
        },
        {
            'name': 'embedding_generation',
            'enabled': True,
            'dependencies': [],  # 독립 실행으로 변경
            'run_if': [],
            'config': EMBEDDING_PARAMS
        },
        {
            'name': 'context_analysis',
            'enabled': True,
            'dependencies': [],  # 독립 실행으로 변경
            'run_if': [],
            'config': {
                'weights': {
                    'situation': CONTEXT_WEIGHTS['situational_context'],
                    'progression': 0.3,
                    'linguistic': 0.3
                }
            }
        },
        {
            'name': 'sub_emotion_detection',  # Stage B: 서브 분기
            'enabled': True,
            'dependencies': ['context_analysis'],
            'run_if': [
                {'type': 'confidence_check', 'module': 'context_analysis',
                 'threshold': THRESHOLDS['sub_confidence']}
            ],
            'config': {
                'use_embeddings': True,
                'use_keywords': True,
                'use_context': True,
                'fallback_strategy': 'hybrid',
                'thresholds': {
                    **SUB_EMOTION_MATCHING,
                    'route_with': 'router.topk_main',  # 상위 라우팅 결과 사용
                    'unknown_threshold': THRESHOLDS['unknown']
                }
            }
        },
        {
            'name': 'transition_analyzer',
            'enabled': True,
            'dependencies': [],  # 독립 실행으로 변경
            'run_if': [],
            'config': dict(EMOTION_TRANSITION_PARAMS)
        },
        {
            'name': 'linguistic_matcher',
            'enabled': True,
            'dependencies': [],  # 독립 실행으로 변경
            'run_if': [],
            'config': {'patterns': EMOTION_ANALYSIS_PATTERNS}
        },
        {
            'name': 'weight_calculator',
            'enabled': True,
            'dependencies': [],  # 독립 실행으로 변경
            'run_if': [],
            'config': {}  # WEIGHT_CALCULATOR_CONFIG 아래 별도 섹션 참조
        },
        {
            'name': 'complex_analyzer',  # Stage C: 최종 집계(단일기 대행)
            'enabled': True,
            'dependencies': [],  # 독립 실행으로 변경
            'run_if': [],
            'config': {'fusion': {'use_time_series': True, 'use_relationships': True}}
        },
        {
            'name': 'situation_analyzer',
            'enabled': True,
            'dependencies': [],  # 독립 실행으로 변경
            'run_if': [],
            'config': {}
        },
        {
            'name': 'time_series_analyzer',
            'enabled': True,
            'dependencies': [],  # 독립 실행으로 변경
            'run_if': [],
            'config': {}
        },
        {
            'name': 'relationship_analyzer',
            'enabled': True,
            'dependencies': [],  # 독립 실행으로 변경
            'run_if': [],
            'config': {}
        },
        {
            'name': 'psychological_analyzer',
            'enabled': True,
            'dependencies': [],  # 독립 실행으로 변경
            'run_if': [],
            'config': {}
        }
    ],
    'output_format': {
        'include_metadata': True,
        'include_confidence_scores': True,
        'include_context_analysis': True,
        # 표준화 키(권장): 엔진에서 지원 시 활용
        'include_transition_points': True,
        'include_intensity_curve': True
    }
}

# -------- 파이프라인 프리셋 & 활성 프리셋 적용 --------
PIPELINE_PRESETS = {
    "fast": ["text_preprocessing", "pattern_extractor", "emotion_classification", "intensity_analysis",
             "embedding_generation"],
    "balanced": ["text_preprocessing", "pattern_extractor", "emotion_classification", "intensity_analysis",
                 "embedding_generation", "context_analysis", "sub_emotion_detection"],
    "heavy": [s["name"] for s in EMOTION_ANALYSIS_PIPELINE.get("steps", [])],
}
ACTIVE_PRESET = os.getenv("EA_PIPELINE_PRESET", "heavy")

try:
    _enabled_names = set(PIPELINE_PRESETS.get(ACTIVE_PRESET, PIPELINE_PRESETS["balanced"]))
    if ACTIVE_PRESET == "heavy":
        for _st in EMOTION_ANALYSIS_PIPELINE.get("steps", []):
            _st["enabled"] = True
    else:
        for _st in EMOTION_ANALYSIS_PIPELINE.get("steps", []):
            _st["enabled"] = (_st.get("name") in _enabled_names)
except Exception as _e:
    logging.getLogger(__name__).warning(f"[config] PIPELINE_PRESETS 적용 실패({ACTIVE_PRESET}): {_e}")

# 특정 스텝 라우팅 정책 ENV 덮어쓰기
try:
    for _st in EMOTION_ANALYSIS_PIPELINE.get("steps", []):
        if _st.get("name") == "emotion_classification":
            _cfg = _st.setdefault("config", {})
            _cfg["route_policy"] = os.getenv("EA_ROUTE_POLICY", _cfg.get("route_policy", "topk_if_uncertain"))
            break
except Exception as _e:
    logging.getLogger(__name__).warning(f"[config] EA_ROUTE_POLICY 적용 실패: {_e}")

# USE_EMBEDDINGS_AT_SERVE === "0" 이면 임베딩 스텝 비활성화 + 의존성 완화
try:
    if not USE_EMBEDDINGS_AT_SERVE:
        for _st in EMOTION_ANALYSIS_PIPELINE.get("steps", []):
            if _st.get("name") == "embedding_generation":
                _st["enabled"] = False
        for _st in EMOTION_ANALYSIS_PIPELINE.get("steps", []):
            if _st.get("name") == "context_analysis":
                # 임베딩 없이도 돌 수 있도록 의존성 완화
                _st["dependencies"] = ['pattern_extractor']
                _st["run_if"] = []
except Exception as _e:
    logging.getLogger(__name__).warning(f"[config] USE_EMBEDDINGS_AT_SERVE 적용 실패: {_e}")

# 임베딩 생성 타임아웃 방지를 위한 환경변수 옵션
EMBEDDING_TIMEOUT_DISABLE = _env_bool("EMBEDDING_TIMEOUT_DISABLE", False)
if EMBEDDING_TIMEOUT_DISABLE:
    # 임베딩 생성 스텝을 비활성화하여 타임아웃 방지
    for _st in EMOTION_ANALYSIS_PIPELINE.get("steps", []):
        if _st.get("name") == "embedding_generation":
            _st["enabled"] = False
            logging.getLogger(__name__).info("[config] 임베딩 생성 스텝 비활성화 (타임아웃 방지)")

# [개선사항 5] 임베딩 ON/OFF 블록 위치 고정
# NOTE: 아래 USE_EMBEDDINGS_AT_SERVE 처리 블록은 반드시
# WEIGHT_CALCULATOR_CONFIG 정의 이후에 위치해야 합니다.
# USE_EMBEDDINGS_AT_SERVE 적용 블록 바로 아래: 서브/웨이트 설정도 함께 비활성화


# ============================
# 성능/캐시/시간·흐름 설정(일관화)
# ============================

PERFORMANCE_CONFIG = {
    'cache_enabled': True,
    'parallel_processing': True,
    'max_workers': 32,  # CPU 최적화: 4→32 증가 (8배)
    'embedding_batch_size': 32,  # CPU 최적화: 16→32 증가 (2배)
    # ★ 성능 최적화: 인스턴스 풀 사전 생성 활성화
    'prewarm_on_start': True,  # 오케스트레이터 시작 시 모든 모듈 인스턴스 사전 생성
}

CACHING_CONFIG = {
    # ★ 성능 최적화: 캐시 크기 증가 (data_utils.py 기본값 1024보다 크게 설정)
    'cache_size': 8192,  # CPU 최적화: 2048→8192 증가 (4배) / data_utils.py 기본값 1024보다 우선
    'weights': {
        'keyword_match': 0.4,
        'pattern_match': 0.3,
        'context_relevance': 0.2,
        'temporal_coherence': 0.1
    },
    'thresholds': {
        'confidence': 0.6,
        'relevance': 0.4,
        'minimum_context_length': 10
    },
    'memory_limit': 2048 * 1024 * 1024,  # CPU 최적화: 512MB→2GB 증가 (4배)
    # TTL 설정 (선택적)
    'ttl_sec': None,  # None이면 TTL 없음 (영구 캐시)
    'max_bytes': None,  # None이면 메모리 제한 없음
}

ANALYZER_THRESHOLDS = {
    'intensity_analyzer': {
        'minimum_confidence': THRESHOLDS['sub_confidence'],
        'pattern_match_threshold': 0.6,
        'context_match_threshold': 0.4
    }
}

WEIGHT_CALCULATOR_CONFIG = {
    'dynamic_weights': {
        'emotion_flow': 0.3,
        'situation_context': 0.3,
        'progression_stage': 0.4
    },
    'use_embeddings_for_weights': True,
    'embedding_weight_factor': 0.5
}
if os.getenv("__DEFER_EMBED_OFF_WEIGHTS") == "1":
    WEIGHT_CALCULATOR_CONFIG["use_embeddings_for_weights"] = False
    WEIGHT_CALCULATOR_CONFIG["embedding_weight_factor"] = 0.0
    os.environ.pop("__DEFER_EMBED_OFF_WEIGHTS", None)

# [개선사항 5] 임베딩 ON/OFF — 정의 이후에 적용 (NameError 방지)
try:
    if not USE_EMBEDDINGS_AT_SERVE:
        for _st in EMOTION_ANALYSIS_PIPELINE.get("steps", []):
            if _st.get("name") == "sub_emotion_detection":
                _cfg = _st.setdefault("config", {})
                _cfg["use_embeddings"] = False
        WEIGHT_CALCULATOR_CONFIG["use_embeddings_for_weights"] = False
        WEIGHT_CALCULATOR_CONFIG["embedding_weight_factor"] = 0.0
except Exception as _e:
    logging.getLogger(__name__).warning(f"[config] sub/weights embed-off 적용 실패: {_e}")

TIME_SERIES_CONFIG = {
    'window_size': 5,
    'min_sequence_length': 3,
    'smoothing_factor': 0.2
}

EMOTIONAL_FLOW_CONFIG = {
    'transition_window': EMOTION_TRANSITION_PARAMS.get('context_window', 3),
    'min_transition_confidence': THRESHOLDS['transition']
}

# ★ 성능 최적화: 병렬 실행 설정 (data_utils.py에서 직접 참조)
# 최상위 레벨에 추가하여 getattr(self.cfg, "PIPELINE_PARALLELIZATION", True)에서 접근 가능
PIPELINE_PARALLELIZATION = _env_bool("PIPELINE_PARALLELIZATION", True)  # 기본값 True로 강제 활성화

MODEL_VERSION_CONFIG = {
    'checkpoints_dir': os.path.join(BASE_DIR, 'checkpoints'),
    'model_version': '1.0.0',
    'emotion_set_version': EMOTION_SET_VERSION,
    'config_signature': CONFIG_SIGNATURE
}

# [개선사항 8] 메트릭/버전 스탬프 표준화(운영 가시성)
RUN_META = {
    "config_signature": CONFIG_SIGNATURE,
    "config_schema": CONFIG_SCHEMA_VERSION,
    "emotion_set_version": EMOTION_SET_VERSION,
    "embedding_dim": EMBEDDING_PARAMS.get('embedding_dim', 384),
    "ea_profile": EA_PROFILE,
    "device": _DEVICE,
    # 시연 모드 메타데이터 추가
    "demo_mode": DEMO_MODE,
    "emotion_alias_mode": EMOTION_ALIAS_MODE,
    "enable_temp_scaling": ENABLE_TEMP_SCALING,
    "default_step_timeout": DEFAULT_STEP_TIMEOUT,
}

PREPROCESSING_CONFIG = {
    'text_cleaning': {
        'remove_special_chars': True,
        'normalize_whitespace': True,
        'remove_urls': True
    }
}

# [개선사항 7] 오류 정책 일원화(기록·중단·스킵)
ERROR_HANDLING_CONFIG = {
    'retry_count': 3,
    'retry_delay': 1.0,
    # 기본 정책: "skip" | "abort" | "record-only"
    'on_error': os.getenv("EA_ON_ERROR", "record-only"),
    'max_retries': int(os.getenv("EA_MAX_RETRIES", "2")),
    'retry_backoff': float(os.getenv("EA_RETRY_BACKOFF", "1.5")),
}

# ============================
# EMOTION_STRUCTURE_MAPPING (스키마 템플릿)
# ============================

EMOTION_STRUCTURE_MAPPING = {
    'metadata': {
        'emotion_id': '',
        'primary_category': '',
        'sub_category': '',
        'emotion_complexity': '',
        'version': ''
    },
    'emotion_profile': {
        'intensity_levels': {
            'low': {'description': '낮은 강도의 감정 상태', 'examples': []},
            'medium': {'description': '중간 강도의 감정 상태', 'examples': []},
            'high': {'description': '높은 강도의 감정 상태', 'examples': []}
        },
        'related_emotions': {
            'positive': [],
            'negative': [],
            'neutral': []
        },
        'core_keywords': []
    },
    'context_patterns': {
        'description': '',
        'intensity': '',
        'core_concept': '',
        'variations': [],
        'keywords': [],
        'examples': [],
        'emotion_progression': {
            'trigger': '',
            'development': '',
            'peak': '',
            'aftermath': ''
        }
    },
    'linguistic_patterns': {
        'key_phrases': [],
        'sentiment_combinations': [],
        'sentiment_modifiers': {
            'amplifiers': [],
            'diminishers': []
        }
    },
    'emotion_transitions': {
        'patterns': []
    }
}

# 심리분석기 기본 옵션(내부에서 config.gate_min_* / prefer_kiwi 속성 접근)
PSYCHOLOGICAL_ANALYZER_CONFIG = {
    "prefer_kiwi": True,
    "gate_min_types": 1,
    "gate_min_sum": 1.0,
}

# ============================
# 통합 매니저 구성(SoT 우선, 기본값 병합)
# ============================
EMOTION_MANAGER_CONFIG = {
    'emotions_data': {},  # 런타임 로더가 채움
    'emotions_path': EMOTIONS_JSON_PATH,
    'emotion_mappings': EMOTION_MAPPINGS,
    'emotion_patterns': EMOTION_ANALYSIS_PATTERNS,
    'emotion_structure_mapping': EMOTION_STRUCTURE_MAPPING,
    'emotion_metrics': EMOTION_METRICS,
    'emotion_guidelines': EMOTION_GUIDELINES,
    'training_params': TRAINING_PARAMS,
    'embedding_params': EMBEDDING_PARAMS,
    'psychological_analyzer_config': PSYCHOLOGICAL_ANALYZER_CONFIG,
    'emotion_base_scores': EMOTION_BASE_SCORES,
    'related_emotions': RELATED_EMOTIONS,
    'zero_shot_model_name': ZERO_SHOT_MODEL_NAME,
    'emotion_analysis_pipeline': EMOTION_ANALYSIS_PIPELINE,
    'paths': PATHS,
    'intensity_thresholds': INTENSITY_THRESHOLDS,
    'context_weights': CONTEXT_WEIGHTS,
    'emotion_transition_params': EMOTION_TRANSITION_PARAMS,
    'analyzer_thresholds': ANALYZER_THRESHOLDS,
    'weight_calculator_config': WEIGHT_CALCULATOR_CONFIG,
    'time_series_config': TIME_SERIES_CONFIG,
    'performance_config': PERFORMANCE_CONFIG,
    'emotional_flow_config': EMOTIONAL_FLOW_CONFIG,
    'model_version_config': MODEL_VERSION_CONFIG,
    'preprocessing_config': PREPROCESSING_CONFIG,
    'error_handling_config': ERROR_HANDLING_CONFIG,
    # ★ 성능 최적화: 병렬 실행 설정 추가
    'PIPELINE_PARALLELIZATION': PIPELINE_PARALLELIZATION,
}


# -------- 엔트리포인트 정적 검증(가벼운 스펙 체크) --------
def validate_entrypoints(eps: dict) -> None:
    # 동적/정적 접두사 후보
    pkg = (__package__ or "").strip(".")
    bases = [b for b in {pkg, "src", "emotion_analysis", "src.emotion_analysis", ""} if b]
    bad = []
    for step, (mod, attr) in eps.items():
        ok = False
        for base in [""] + [b for b in bases if b != ""]:  # "" 먼저 시도
            name = f"{base + '.' if base else ''}{mod}"
            try:
                if find_spec(name) is not None:
                    ok = True
                    break
            except Exception:
                # 일부 환경에서 find_spec가 예외를 던지기도 함 → 무시하고 다음 후보
                continue
        if not ok:
            bad.append((step, mod, attr, "module-not-found"))
    if bad:
        logging.getLogger(__name__).warning("MODULE_ENTRYPOINTS unresolved: %s", bad)


# 임시 가드: 환경에 따라 검증 실패 시 전체 초기화를 막지 않음
try:
    validate_entrypoints(MODULE_ENTRYPOINTS)
except Exception as e:
    logging.getLogger(__name__).warning("validate_entrypoints skipped: %s", e)

logging.info("최종 통합 config([2] 영역) 로딩 및 개선 완료.")

# ============================
# [A-3] 스피드/결정론 토글 + 간단 헬퍼
# ============================

# 스피드/결정론 프로필 설정 (한 번만 넣어두면 끝)
EMB_PROFILE = os.getenv("EMB_PROFILE", "speed")  # "speed" | "deterministic"

if EMB_PROFILE == "speed":
    # 빠름(정확도 동일) — 재현성은 엄밀히 보장 X
    os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)    # 결정론 강제 해제
    try:
        import torch
        torch.use_deterministic_algorithms(False)
        torch.backends.cuda.matmul.allow_tf32 = True       # 속도 ↑
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass
else:
    # 느림(재현성 보장)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        import torch
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    except Exception:
        pass

DETERMINISM_CONFIG = {
    "seed": 42,
    "torch_deterministic": EMB_PROFILE == "deterministic",
    "cudnn_benchmark": EMB_PROFILE == "speed"
}

# ENV 토글 노출(속도 vs 재현성 즉시 전환)
DETERMINISM_CONFIG.update({
    "seed": int(os.getenv("SEED", DETERMINISM_CONFIG.get("seed", 42))),
    "torch_deterministic": os.getenv("TORCH_DETERMINISTIC", str(int(EMB_PROFILE == "deterministic"))),
    "cudnn_benchmark": os.getenv("CUDNN_BENCHMARK", str(int(EMB_PROFILE == "speed"))),
})


def apply_determinism(cfg: dict = None):
    """전역 결정론 설정 적용 (재현성 보장)"""
    if cfg is None:
        cfg = DETERMINISM_CONFIG
    try:
        import random
        random.seed(int(cfg.get("seed", 42)))
        os.environ.setdefault("PYTHONHASHSEED", str(cfg.get("seed", 42)))
        try:
            import numpy as np
            np.random.seed(int(cfg.get("seed", 42)))
        except Exception:
            pass
        try:
            import torch
            torch.manual_seed(int(cfg.get("seed", 42)))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(cfg.get("seed", 42)))
            if bool(cfg.get("torch_deterministic", True)):
                torch.use_deterministic_algorithms(True)
                import torch.backends.cudnn as cudnn
                cudnn.benchmark = bool(cfg.get("cudnn_benchmark", False))
        except Exception:
            pass
    except Exception:
        pass


# ============================
# [A-7] 스키마/버전 합치 체크
# ============================
def assert_runtime_contracts(embedding_dim_expected: int = None) -> None:
    """런타임 스키마/버전 계약 검증"""
    if embedding_dim_expected is None:
        embedding_dim_expected = EMBEDDING_PARAMS.get('embedding_dim', 384)

    try:
        assert isinstance(EMOTION_SET_VERSION, str) and len(EMOTION_SET_VERSION) > 0
        assert int(embedding_dim_expected) == int(EMBEDDING_PARAMS['embedding_dim'])
        # 버전 시그니처 확인
        assert isinstance(CONFIG_SIGNATURE, str) and len(CONFIG_SIGNATURE) > 0
        assert isinstance(EMBEDDING_SCHEMA_VERSION, str) and len(EMBEDDING_SCHEMA_VERSION) > 0
        logging.getLogger(__name__).debug(
            f"[contract] validated: emotion_v={EMOTION_SET_VERSION}, "
            f"emb_dim={embedding_dim_expected}, schema_v={EMBEDDING_SCHEMA_VERSION}"
        )
    except AssertionError as e:
        logging.getLogger(__name__).warning(f"[contract] schema or dim mismatch: {e}")


# ============================
# 프로젝트 상태 체크 헬퍼
# ============================
def check_project_health() -> Dict[str, Any]:
    """프로젝트 필수 구성요소 상태 점검"""
    health = {
        "status": "healthy",
        "issues": [],
        "warnings": [],
        "components": {}
    }

    # 1. 필수 디렉토리 체크
    required_dirs = {
        "label": LABEL_DIR,
        "logs": LOG_DIR,
        "models": MODEL_DIR,
        "embeddings": EMBEDDINGS_DIR
    }
    # 임베딩 비사용 서빙 모드일 때는 정상 비활성로 간주
    if not USE_EMBEDDINGS_AT_SERVE:
        required_dirs["embeddings"] = None

    for name, path in required_dirs.items():
        if name == "embeddings" and not USE_EMBEDDINGS_AT_SERVE:
            health["warnings"].append("embeddings disabled at serve")
            health["components"][name] = "DISABLED"
            continue
        if path and os.path.isdir(path):
            health["components"][name] = "OK"
        elif path is None:
            health["warnings"].append(f"{name} directory deferred (STRICT_LABEL_DIR=0)")
            health["components"][name] = "DEFERRED"
        else:
            health["issues"].append(f"{name} directory missing: {path}")
            health["components"][name] = "MISSING"

    # 2. EMOTIONS.JSON 체크
    if EMOTIONS_JSON_PATH and os.path.isfile(EMOTIONS_JSON_PATH):
        health["components"]["emotions_json"] = "OK"
    else:
        health["issues"].append("EMOTIONS.JSON not found")
        health["components"]["emotions_json"] = "MISSING"

    # 3. 디바이스 체크
    health["components"]["device"] = _DEVICE

    # 4. 임베딩 모델 체크
    health["components"]["embedding_model"] = EMBEDDING_PARAMS.get('model_name', 'UNKNOWN')
    health["components"]["embedding_dim"] = EMBEDDING_PARAMS.get('embedding_dim', 0)

    # 전체 상태 결정
    if health["issues"]:
        health["status"] = "critical" if len(health["issues"]) > 2 else "degraded"
    elif health["warnings"]:
        health["status"] = "warning"

    return health


# [개선사항 3] 파이프라인/엔트리포인트 사전 검증 강화
def validate_pipeline_spec() -> List[str]:
    """파이프라인 스텝과 엔트리포인트 일치 검증"""
    issues = []
    steps = (EMOTION_ANALYSIS_PIPELINE or {}).get("steps", [])
    eps = MODULE_ENTRYPOINTS
    names = set()
    for s in steps:
        nm = s.get("name")
        if not nm:
            issues.append("pipeline: step without name")
            continue
        if nm in names:
            issues.append(f"pipeline: duplicated step '{nm}'")
        names.add(nm)
        if nm != "text_preprocessing" and nm not in eps:
            issues.append(f"pipeline: no entrypoint for '{nm}'")
    return issues


# ============================
# 구성 검증(서빙 전 점검)
# ============================
def validate_config(strict: bool = False) -> List[str]:
    issues: List[str] = []
    # unified
    try:
        mp = get_model_path("unified")
    except Exception:
        mp = None
    if mp and not os.path.isfile(mp):
        issues.append(f"Missing model: {mp}")

    # sub 4종
    for key in ("희", "노", "애", "락"):
        try:
            sp = get_model_path("sub", key)
            if not os.path.isfile(sp):
                issues.append(f"Missing sub model ({key}): {sp}")
        except Exception:
            issues.append(f"Sub model path resolve failed: {key}")

    # 라벨 맵/EMOTIONS.JSON 경로 확인(없으면 축소 모드 경고)
    if EMOTIONS_JSON_PATH and not os.path.isfile(EMOTIONS_JSON_PATH):
        issues.append("EMOTIONS.JSON not found (fallback to minimal mapping)")

    # 파이프라인 스펙 검증 추가
    issues.extend(validate_pipeline_spec())

    if strict and issues:
        raise RuntimeError("Invalid configuration:\n - " + "\n - ".join(issues))
    return issues


# [개선사항 1] enforce_runtime_contracts - 서빙 전 최종 점검
def enforce_runtime_contracts(strict: Optional[bool] = None) -> List[str]:
    """서빙 전 최종 점검: 스키마/버전/모델/경로"""
    issues = []
    if strict is None:
        strict = CFG_STRICT_ON_START

    # 런타임 계약 검증
    try:
        assert_runtime_contracts()  # emb_dim/서명/셋버전 확인
    except Exception as e:
        issues.append(f"contract: {e}")

    # 기본 검증 + 필요 시 서브 4종 강제
    v = validate_config(strict=bool(strict))
    issues.extend(v)

    if STRICT_SUBS:
        # validate_config가 이미 sub 4종 체크함 → strict=True면 raise
        if not strict and any("Missing sub model" in m for m in v):
            raise RuntimeError("STRICT_SUBS=1: all sub models required")

    return issues


# [개선사항 6] 운영 가드 + 재현성 원클릭
def harden_for_prod() -> dict:
    """운영 직전 한 번 호출하면 좋은 종합 점검/세팅"""
    apply_determinism()  # seed/cudnn/torch 결정론
    ensure_runtime_dirs()
    issues = enforce_runtime_contracts(strict=CFG_STRICT_ON_START)
    health = check_project_health()
    return {"issues": issues, "health": health, "run_meta": RUN_META}

# ============================
# 프리셋 설정 (prod 기본값 활성화)
# ============================

# 중복 제거: 위의 DEFAULT_STEP_TIMEOUTS와 병합됨

PRESET_CONFIGS = {
    'prod': {
        'PIPELINE_PARALLELIZATION': True,
        'PIPELINE_MAX_WORKERS': 32,  # CPU 최적화: 8→32 증가 (4배)
        'MODEL_CACHE_ENABLED': True,
        'MODEL_CACHE_MAX_MODELS': 16,  # CPU 최적화: 8→16 증가 (2배)
        'CACHE_SIZE': 8192,  # CPU 최적화: 2048→8192 증가 (4배)
        'DEFAULT_STEP_TIMEOUT': 45,
        'STEP_TIMEOUTS': DEFAULT_STEP_TIMEOUTS,
        'USE_EMBEDDINGS_AT_SERVE': True,
        'EMBEDDING_BATCH_SIZE': 64,  # CPU 최적화: 32→64 증가 (2배)
        'ENABLE_SHORT_TEXT_GATING': True,
        'ENABLE_AC_OPTIMIZATION': True,
        # ★ 성능 최적화: 인스턴스 풀 사전 생성 활성화
        'PREWARM_ON_START': True,  # 오케스트레이터 시작 시 모든 모듈 인스턴스 사전 생성
    },
    'dev': {
        'PIPELINE_PARALLELIZATION': False,
        'PIPELINE_MAX_WORKERS': 2,
        'MODEL_CACHE_ENABLED': False,
        'CACHE_SIZE': 256,
        'DEFAULT_STEP_TIMEOUT': 30,
        'STEP_TIMEOUTS': {k: v // 2 for k, v in DEFAULT_STEP_TIMEOUTS.items()},  # 개발 환경은 절반으로 단축
        'USE_EMBEDDINGS_AT_SERVE': False,
        'EMBEDDING_BATCH_SIZE': 8,
        'ENABLE_SHORT_TEXT_GATING': False,
        'ENABLE_AC_OPTIMIZATION': False,
    }
}

def apply_preset(preset_name: str = 'prod') -> None:
    """프리셋 설정 적용"""
    if preset_name not in PRESET_CONFIGS:
        logging.warning(f"[config] Unknown preset: {preset_name}, using 'prod'")
        preset_name = 'prod'
    
    preset = PRESET_CONFIGS[preset_name]
    for key, value in preset.items():
        os.environ.setdefault(key, str(value))
        logging.info(f"[config] Applied preset {preset_name}: {key}={value}")

# 기본적으로 prod 프리셋 적용 (환경변수로 제어 가능)
if os.getenv("CONFIG_APPLY_PRESET", "1") == "1":
    apply_preset('prod')

# EMOTIONS 경로 환경변수 확정 설정 (모듈 경고 최소화)
_ejp = str(EMOTIONS_JSON_PATH) if EMOTIONS_JSON_PATH else ""
if _ejp:
    os.environ.setdefault("EMOTIONS_JSON", _ejp)
    os.environ.setdefault("EMOTIONS_PATH", _ejp)

