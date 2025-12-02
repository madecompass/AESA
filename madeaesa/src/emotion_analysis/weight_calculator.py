# -*- coding: utf-8 -*-
from __future__ import annotations

# 표준 라이브러리
import os
import re
import json
import time
import math
import logging
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple, Set
from collections import defaultdict, deque
from functools import lru_cache
from logging.handlers import RotatingFileHandler
from concurrent.futures import ThreadPoolExecutor

# 선택(옵션) 의존성: 없으면 graceful fallback
try:
    import numpy as np
except Exception:  # numpy 미설치 시 안전 처리
    np = None  # type: ignore

try:
    import torch
    from torch import Tensor
    from torch.nn import Embedding
    _TORCH_OK = True
    # CUDA 가용성 체크
    _CUDA_AVAILABLE = torch.cuda.is_available() if _TORCH_OK else False
    _CUDA_DEVICE_COUNT = torch.cuda.device_count() if _CUDA_AVAILABLE else 0
except Exception:  # torch 미설치 시 안전 처리
    _TORCH_OK = False
    _CUDA_AVAILABLE = False
    _CUDA_DEVICE_COUNT = 0

# Aho-Corasick 자동자 지원 (선택적 의존성)
try:
    import ahocorasick
    _AHO_OK = True
except ImportError:
    _AHO_OK = False
    Tensor = Any  # type: ignore
    Embedding = object  # type: ignore

# =============================================================================
# EmotionLogger
# =============================================================================
class EmotionLogger:
    """
    - 텍스트 로그: 회전(rotating) 파일 + 콘솔
    - JSON 로그: 원자적 저장(임시 파일 -> replace), 손상 시 안전 복구
    - 로그 경로 우선순위: 인자(log_dir) > ENV(EMOTION_LOG_DIR/LOGS_DIR) > 모듈 기준 ./logs
    - 중복 핸들러 자동 방지, propagate=False
    """

    def __init__(
        self,
        log_file_name: str = "weight_calculator.log",
        log_dir: Optional[str | Path] = None,
        max_bytes: int = 10 * 1024 * 1024,
        backup_count: int = 5,
        console_level: int = logging.INFO,
        file_level: int = logging.DEBUG,
        logger_name: str = "weight_calculator"
    ):
        self._session_id = time.strftime("%Y%m%d_%H%M%S")

        # 1) 로그 디렉터리 결정 (통합 로그 관리자 사용 - 날짜별 폴더)
        try:
            from log_manager import get_log_manager
            log_manager = get_log_manager()
            # 독립 모듈 테스트 컨텍스트 사용 (날짜별 폴더 적용)
            self.base_dir = log_manager.get_log_dir("emotion_analysis", use_date_folder=True)
        except ImportError:
            # 폴백: 기존 방식 (날짜별 폴더 추가)
            base_log_dir = Path(
                log_dir
                or os.environ.get("EMOTION_LOG_DIR")
                or os.environ.get("LOGS_DIR")
                or Path(__file__).parent / "logs"
            ).resolve()
            # 날짜별 폴더 추가
            today = time.strftime("%Y%m%d")
            self.base_dir = base_log_dir / today

        # 2) 로거 준비(중복 핸들러 방지)
        self.logger = logging.getLogger(logger_name)
        self.logger.propagate = False
        self.logger.setLevel(logging.DEBUG)

        if not getattr(self.logger, "_initialized", False):
            # 포맷터
            fmt = logging.Formatter(
                fmt="[%(asctime)s] %(levelname)s %(name)s [%(filename)s:%(lineno)d] "
                    "(session=%(session)s) - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            # 기본: 비간섭(핸들러 없음이면 NullHandler 1개만)
            if not self.logger.handlers:
                self.logger.addHandler(logging.NullHandler())

            # Opt-in: 환경변수로 파일/콘솔 로그 활성화
            if os.environ.get("EWC_FILE_LOG", "0").lower() in ("1", "true", "yes"):
                try:
                    self.base_dir.mkdir(parents=True, exist_ok=True)
                    fh = RotatingFileHandler(
                        filename=str(self.base_dir / log_file_name),
                        maxBytes=max_bytes,
                        backupCount=backup_count,
                        encoding="utf-8"
                    )
                    fh.setLevel(file_level)
                    fh.setFormatter(fmt)
                    self.logger.addHandler(fh)
                except Exception:
                    pass

            if os.environ.get("EWC_CONSOLE_LOG", "0").lower() in ("1", "true", "yes"):
                try:
                    ch = logging.StreamHandler()
                    ch.setLevel(console_level)
                    ch.setFormatter(fmt)
                    self.logger.addHandler(ch)
                except Exception:
                    pass

            # 세션 컨텍스트 주입용 필터
            class _SessionFilter(logging.Filter):
                def __init__(self, session: str):
                    super().__init__()
                    self._session = session
                def filter(self, record: logging.LogRecord) -> bool:
                    # extra 필드 보강: %(session)s
                    if not hasattr(record, "session"):
                        record.session = self._session
                    return True

            self.logger.addFilter(_SessionFilter(self._session_id))
            self.logger._initialized = True

        self.logger.debug(f"EmotionLogger initialized. log_dir={self.base_dir}")

    # ----------------------------- public API -----------------------------

    def get_logger(self) -> logging.Logger:
        return self.logger

    def set_log_level(self, level: str) -> None:
        """
        파일/콘솔 모두 동일 레벨로 재설정.
        """
        level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }
        chosen = level_map.get(level.upper(), logging.INFO)
        self.logger.setLevel(chosen)
        for h in self.logger.handlers:
            # 파일/콘솔 각각의 성격을 유지하고 싶다면 조건 분기 가능
            h.setLevel(chosen)
        self.logger.info(f"로그 레벨 변경 → {level.upper()}")

    def save_json_log(
        self,
        data: Dict[str, Any],
        filename: str = "weight_calculator.json",
        keep_last: int = 1000
    ) -> None:
        """
        JSON 로그에 항목을 append(원자적 저장).
        - 파일 손상(fail/partial write)에도 안전하도록 temp 파일로 저장 후 replace
        - keep_last: 과도한 성장 방지(최근 N개만 유지)
        """
        try:
            log_path = self.base_dir / filename
            # 디렉토리 보장(NullHandler만 사용 중이어도 저장 시에는 필요)
            try:
                log_path.parent.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
            payload = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "data": data
            }

            # 기존 데이터 로드(손상 시 복구)
            existing: List[Dict[str, Any]] = []
            if log_path.exists():
                try:
                    with open(log_path, "r", encoding="utf-8") as rf:
                        existing = json.load(rf)
                        if not isinstance(existing, list):
                            existing = []
                except Exception as e:
                    self.logger.warning(f"기존 JSON 로그 파싱 실패(복구): {e}")
                    # 손상된 파일 백업
                    backup = log_path.with_suffix(".corrupted.json")
                    try:
                        log_path.replace(backup)
                        self.logger.warning(f"손상 로그 백업: {backup}")
                    except Exception:
                        pass
                    existing = []

            existing.append(payload)
            # 최근 N개만 유지
            if keep_last and len(existing) > keep_last:
                existing = existing[-keep_last:]

            # 원자적 저장: temp -> replace
            tmp_path = log_path.with_suffix(".tmp")
            with open(tmp_path, "w", encoding="utf-8") as wf:
                json.dump(existing, wf, ensure_ascii=False, indent=2)
                wf.flush()
                os.fsync(wf.fileno())
            tmp_path.replace(log_path)

            self.logger.info(f"JSON 로그 저장 완료: {log_path.name} (entries={len(existing)})")

        except Exception as e:
            self.logger.exception(f"JSON 로그 저장 실패: {e}")

    def info(self, msg: str, **extra):
        self._log(logging.INFO, msg, **extra)

    def debug(self, msg: str, **extra):
        self._log(logging.DEBUG, msg, **extra)

    def warning(self, msg: str, **extra):
        self._log(logging.WARNING, msg, **extra)

    def error(self, msg: str, **extra):
        self._log(logging.ERROR, msg, **extra)

    def critical(self, msg: str, **extra):
        self._log(logging.CRITICAL, msg, **extra)

    # ---------------------------- internal utils ----------------------------

    def _log(self, level: int, msg: str, **extra):
        """
        extra로 세션 외 임의 키를 안전 주입.
        """
        if extra:
            # logging.makeLogRecord를 쓰지 않고 LoggerAdapter 쓰는 방법도 가능.
            self.logger.log(level, msg, extra=extra)
        else:
            self.logger.log(level, msg)


# =============================================================================
# WeightEmotionDataManager
# =============================================================================
class WeightEmotionDataManager:
    """
    - EMOTIONS.json 로딩/검증/인덱스 구축/캐싱을 담당
    - 규칙1: 가능한 한 JSON 데이터에서 유도(하드코딩 최소화)
    - 규칙2: 4×N(희/노/애/락 × 가변 세부감정) 확장 안전
    - 규칙3: 라벨링 스키마 변경 금지(필요시 보정은 in-memory에서만)
    """

    # --------------------------------------------------------------------- init
    def __init__(
        self,
        logger: "EmotionLogger",
        config_path: Optional[str] = None,
        emotions_file: Optional[str | Path] = None,
        cache_timeout: int = 300
    ):
        self.logger = logger.get_logger()
        self.cache_timeout = int(cache_timeout)
        self.cache: Dict[str, Any] = {}
        self.cache_timestamps: Dict[str, float] = {}

        # 메모리 캐시/인덱스
        self.emotions_data: Dict[str, Any] = {}
        self._flat_index: Dict[str, Dict[str, Any]] = {}       # emotion_id → node(dict)
        self._category_map: Dict[str, str] = {}                # emotion_id → 대표카테고리('희','노','애','락')
        self._top_ids: List[str] = []                          # 상위(대표) emotion ids (희/노/애/락)
        self.id_mapping_cache: Dict[str, str] = {}             # emotion_id → 대표카테고리
        self.emotion_data_cache: Dict[str, Any] = {}
        self.sub_emotion_cache: Dict[str, Any] = {}

        # 표준 검증 룰
        self._initialize_validation_rules_v2()

        # 선택: config 오버레이
        if config_path:
            self._load_config_v1(config_path)
            self._load_config_v2(config_path)

        # 파일 로드(있으면) → 없으면 유저가 나중에 load_emotions_data 호출
        try:
            if emotions_file:
                self._load_emotions_json(Path(emotions_file))
            else:
                auto = self._locate_emotions_file()
                if auto:
                    self._load_emotions_json(auto)
            if self.emotions_data:
                self._rebuild_indexes()
                self._build_id_mapping_cache_v2()
            self.logger.info(
                f"WeightEmotionDataManager 초기화 완료 - 대표:{len(self._top_ids)}개, "
                f"총 노드:{len(self._flat_index)}개"
            )
        except Exception as e:
            self.logger.error(f"초기 로드 실패: {e}", exc_info=True)

    # ----------------------------------------------------------- config loaders
    def _initialize_validation_rules_v2(self):
        self.validation_rules = {
            "required_keys": [
                "metadata", "emotion_profile", "context_patterns",
                "linguistic_patterns", "emotion_transitions",
                "sentiment_analysis", "ml_training_metadata"
            ],
            "metadata_keys": [
                "emotion_id", "primary_category", "sub_category",
                "emotion_complexity", "version"
            ],
            "intensity_range": [0.0, 1.0],
            "complexity_levels": ["basic", "complex", "subtle"],
            "version_pattern": r"^\d+\.\d+$"
        }
        # 정보성 매핑(로직 의존 금지)
        self.emotion_mapping = {
            "1": {"category": "희", "description": "긍정적이고 기쁜 감정", "intensity_range": [0.0, 1.0], "sub_count": 30},
            "2": {"category": "노", "description": "화나고 부정적인 감정", "intensity_range": [0.0, 1.0], "sub_count": 30},
            "3": {"category": "애", "description": "슬프고 우울한 감정", "intensity_range": [0.0, 1.0], "sub_count": 30},
            "4": {"category": "락", "description": "즐겁고 행복한 감정", "intensity_range": [0.0, 1.0], "sub_count": 30},
        }
        self.emotion_profiles: Dict[str, Any] = {}

    def _load_config_v1(self, config_path: str) -> None:
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            self.validation_rules.update(config.get("validation_rules", {}))
            self.emotion_profiles.update(config.get("emotion_profiles", {}))
            self.logger.info(f"[config_v1] 설정 로드: {config_path}")
        except Exception as e:
            self.logger.error(f"[config_v1] 설정 로드 실패: {e}")

    def _load_config_v2(self, config_path: str) -> None:
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            if "validation_rules" in config:
                self.validation_rules.update(config["validation_rules"])
            if "emotion_profiles" in config:
                self.emotion_profiles.update(config["emotion_profiles"])
            self.logger.info(f"[config_v2] 설정 로드: {config_path}")
        except Exception as e:
            self.logger.error(f"[config_v2] 설정 로드 실패: {e}")

    # --------------------------------------------------------------- file I/O
    def _locate_emotions_file(self) -> Optional[Path]:
        """
        우선순위: ENV → 글로벌 상수 → 프로젝트 상대 경로 후보
        """
        candidates: List[Path] = []
        env = os.environ.get("EMOTIONS_FILE_PATH")
        if env:
            candidates.append(Path(env))
        g = globals().get("EMOTIONS_FILE_PATH")
        if g:
            candidates.append(Path(g))
        here = Path(__file__).resolve().parent
        candidates += [
            here.parent / "EMOTIONS.json",
            here.parent / "src" / "EMOTIONS.json",
            here.parent.parent / "EMOTIONS.json",
        ]
        for c in candidates:
            if c and c.exists():
                return c
        return None

    def _load_emotions_json(self, path: Path) -> bool:
        """
        EMOTIONS.json 로드 → 최상위 대표 감정 노드에 누락 섹션(backfill) 보정 → 인덱스 재구성.
        - 최상위(희/노/애/락)는 실제 현업 JSON에서 보통 요약 메타만 두고, 상세는 하위에 두는 경우가 많음.
          이 패턴을 존중하면서도 파이프라인이 요구하는 키는 빈 dict로 보정해 경고를 제거한다.
        """
        try:
            # 전역 캐시에서 먼저 시도
            try:
                from src.data_utils import get_global_emotions_data
                data = get_global_emotions_data()
                if data:
                    path = data  # 캐시된 데이터 사용
            except Exception:
                pass
            
            # path가 dict인 경우 처리
            if isinstance(path, dict):
                data = path
            else:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            if not isinstance(data, dict) or not data:
                raise ValueError("EMOTIONS.json 형식 오류 또는 비어있음.")

            # 1) 최상위 대표 감정 노드 보정(backfill)
            def _backfill_top(node: Dict[str, Any]) -> None:
                # 스키마 키들 중 최상위에서 없어도 되는 키는 빈 dict로 보정
                must_have_empty_ok = [
                    "emotion_profile", "context_patterns", "linguistic_patterns",
                    "emotion_transitions", "sentiment_analysis", "ml_training_metadata"
                ]
                for k in must_have_empty_ok:
                    if k not in node:
                        node[k] = {}

                # 필수 구조 안전 가드
                node.setdefault("metadata", {})
                node.setdefault("sub_emotions", {})

            for top_id, top_node in list(data.items()):
                if isinstance(top_node, dict):
                    _backfill_top(top_node)

            self.emotions_data = data

            # 2) 인덱스 재구성 및 id 맵 캐시
            self._rebuild_indexes()
            self._build_id_mapping_cache_v2()

            # 3) 스키마 검증(운영 모드에서는 엄격 검증 실패 시 hard-fail)
            from config import EA_PROFILE, RENDER_DEPLOYMENT
            is_production = EA_PROFILE == "prod" or RENDER_DEPLOYMENT or os.getenv("PRODUCTION_MODE", "0") == "1"
            
            if not self.validate_emotions_schema_v1():
                if is_production:
                    error_msg = "EMOTIONS.json 스키마 검증 실패 - 운영 모드에서는 허용되지 않음"
                    self.logger.error(error_msg)
                    raise RuntimeError(error_msg)
                else:
                    self.logger.warning("스키마 경고: 일부 항목이 누락일 수 있음(보정 완료, 진행 계속).")
            self.logger.info(f"EMOTIONS.json 로드 완료: {path}")
            return True
        except Exception as e:
            self.logger.error(f"EMOTIONS.json 로드 실패: {e}", exc_info=True)
            self.emotions_data = {}
            return False

    # ----------------------------------------------------------------- builder
    def load_emotions_data(self, file_path: Optional[str] = None) -> bool:
        """
        라벨 디렉토리(./label/희|노|애|락/*.json)를 스캔해 통합 EMOTIONS.json 생성 + 로드.
        - file_path가 주어지면 해당 경로를 우선 로드
        - 없고, 기존 EMOTIONS.json가 있으면 그것을 우선 로드
        - 둘 다 없으면 label 디렉토리에서 빌드
        """
        try:
            if file_path:
                ok = self._load_emotions_json(Path(file_path))
                if ok:
                    self._rebuild_indexes()
                    self._build_id_mapping_cache_v2()
                return ok

            # 기존 EMOTIONS.json 먼저 시도
            default_json = self._locate_emotions_file()
            if default_json and default_json.exists():
                ok = self._load_emotions_json(default_json)
                if ok:
                    self._rebuild_indexes()
                    self._build_id_mapping_cache_v2()
                    return True

            # 라벨에서 생성
            current_dir = Path(__file__).parent
            src_dir = current_dir.parent
            labels_dir = src_dir / "label"
            emotions_json_path = src_dir / "EMOTIONS.json"

            self.logger.info(f"Labels 디렉토리 경로: {labels_dir}")
            self.logger.info(f"EMOTIONS.json 경로: {emotions_json_path}")

            emotions_data: Dict[str, Any] = {}
            primary_categories = ["희", "노", "애", "락"]
            for category in primary_categories:
                category_path = labels_dir / category
                if not category_path.exists():
                    self.logger.warning(f"카테고리 디렉토리를 찾을 수 없음: {category_path}")
                    continue

                category_data = self._initialize_category_data(category)
                for json_file in category_path.glob("*.json"):
                    try:
                        sub_emotion = self._process_sub_emotion_file(json_file, category)
                        if sub_emotion:
                            sub_name = json_file.stem
                            category_data["sub_emotions"][sub_name] = sub_emotion
                    except Exception as e:
                        self.logger.error(f"파일 처리 오류({json_file.name}): {e}")

                emotions_data[category] = category_data

            if not emotions_data:
                self.logger.error("라벨로부터 감정 데이터를 구성할 수 없습니다.")
                self.emotions_data = {}
                return False

            self._save_emotions_json(emotions_json_path, emotions_data)
            self.emotions_data = emotions_data
            self._log_load_completion(emotions_data)
            self._rebuild_indexes()
            self._build_id_mapping_cache_v2()
            return True

        except Exception as e:
            self.logger.error(f"감정 데이터 로딩 중 오류: {e}", exc_info=True)
            self.emotions_data = {}
            return False

    def ensure_top_level_profiles(self) -> None:
        """
        최상위 대표감정(희/노/애/락)에 필수 키가 없을 때 안전한 빈 구조로 백필하고,
        기존 WARNING을 INFO로 내려 소음 감소. (동작 무관)
        """
        if not isinstance(self.emotions_data, dict):
            return
        required = [
            "emotion_profile",
            "context_patterns",
            "linguistic_patterns",
            "emotion_transitions",
            "sentiment_analysis",
            "ml_training_metadata",
            "sub_emotions",
            "metadata"
        ]
        changed = False
        for top_id, top_data in self.emotions_data.items():
            if not isinstance(top_data, dict):
                continue
            for k in required:
                if k not in top_data or top_data[k] is None:
                    # 백필
                    if k == "metadata":
                        meta = top_data.get("metadata", {}) or {}
                        meta.setdefault("primary_category", top_id)
                        meta.setdefault("emotion_id", top_id)
                        meta.setdefault("version", "1.0")
                        top_data["metadata"] = meta
                    elif k == "sub_emotions":
                        top_data["sub_emotions"] = {}
                    else:
                        top_data[k] = {}
                    self.logger.info(f"[ensure_top_level_profiles] '{top_id}'에 '{k}'가 없어 기본 구조를 백필했습니다.")
                    changed = True

            # 예외적으로 가장 많이 등장하는 누락 로그(예: emotion_profile)만 INFO로 집계
            if "emotion_profile" not in top_data or not isinstance(top_data.get("emotion_profile"), dict):
                self.logger.info(f"[ensure_top_level_profiles] 대표 '{top_id}'의 'emotion_profile'이 비어있어 빈 프로필을 사용합니다.")

        if changed:
            self.logger.info("[ensure_top_level_profiles] 최상위 대표감정 스키마 보강/백필 완료.")

    def _initialize_category_data(self, category: str) -> Dict[str, Any]:
        return {
            "metadata": {"primary_category": category, "emotion_id": category, "version": "1.0"},
            "emotion_profile": {},
            "context_patterns": {},
            "linguistic_patterns": {},
            "emotion_transitions": {},
            "sentiment_analysis": {},
            "ml_training_metadata": {},
            "sub_emotions": {},
        }

    def _process_sub_emotion_file(self, json_file: Path, category: str) -> Optional[Dict[str, Any]]:
        try:
            with json_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                self.logger.warning(f"잘못된 JSON 형식: {json_file.name}")
                return None

            md = data.get("metadata", {}) or {}
            md.update({
                "emotion_id": f"{category}-{json_file.stem}",
                "primary_category": category,
                "sub_category": json_file.stem
            })
            data["metadata"] = md

            # 필수 섹션 보정(스키마 유지, 없는 키만 빈 dict로)
            for sec in self.validation_rules["required_keys"]:
                if sec not in data:
                    data[sec] = {}

            return data
        except Exception as e:
            self.logger.error(f"하위 감정 파일 처리 오류({json_file.name}): {e}")
            return None

    def _save_emotions_json(self, file_path: Path, emotions_data: Dict[str, Any]) -> None:
        with file_path.open("w", encoding="utf-8") as f:
            json.dump(emotions_data, f, ensure_ascii=False, indent=2)
        self.logger.info(f"통합 EMOTIONS.json 저장: {file_path}")

    def _log_load_completion(self, emotions_data: Dict[str, Any]) -> None:
        total_categories = len(emotions_data)
        total_sub = sum(len(cat.get("sub_emotions", {})) for cat in emotions_data.values())
        self.logger.info(f"감정 데이터 로드 완료: 대표 {total_categories}개, 하위 {total_sub}개")

    # --------------------------------------------------------------- validators
    # WeightEmotionDataManager 내 교체
    def validate_emotions_schema_v1(self) -> bool:
        """
        최상위 노드(희/노/애/락)는 일부 섹션 없음 허용(백필),
        하위 감정은 기존 규칙대로 엄격 검증.
        """
        if not isinstance(self.emotions_data, dict):
            self.logger.warning("emotions_data가 dict 형식이 아닙니다.")
            return False

        required_top_keys = [
            "metadata",
            "sub_emotions",
            "emotion_profile",
            "context_patterns",
            "linguistic_patterns",
            "emotion_transitions",
            "sentiment_analysis",
            "ml_training_metadata"
        ]
        required_sub_keys = [
            "metadata",
            "emotion_profile",
            "context_patterns",
            "linguistic_patterns",
            "emotion_transitions",
            "sentiment_analysis",
            "ml_training_metadata"
        ]
        meta_keys = ["emotion_id", "primary_category", "sub_category", "emotion_complexity", "version"]

        ok = True
        for top_id, top_data in self.emotions_data.items():
            if not isinstance(top_data, dict):
                self.logger.warning(f"{top_id} 감정 데이터가 dict가 아닙니다. 구조 이상.")
                ok = False
                continue

            # 최상위에 대해 누락 항목은 백필
            for k in required_top_keys:
                if k not in top_data:
                    top_data[k] = {} if k != "sub_emotions" else {}
                    self.logger.debug(f"[schema] top-level {top_id}에 '{k}' 백필")

            # 하위 감정은 엄격히
            sub_emotions = top_data.get("sub_emotions", {})
            if not isinstance(sub_emotions, dict):
                self.logger.info(f"{top_id}의 sub_emotions가 dict가 아님 → 빈 dict로 보정")
                top_data["sub_emotions"] = {}
                sub_emotions = {}

            for sub_name, sub_data in sub_emotions.items():
                if not isinstance(sub_data, dict):
                    self.logger.warning(f"하위 감정 {sub_name} 데이터가 dict가 아닙니다. 구조 이상.")
                    ok = False
                    continue
                for srk in required_sub_keys:
                    if srk not in sub_data:
                        self.logger.warning(f"{sub_name} 하위 감정 데이터에 '{srk}' 누락")
                        ok = False
                meta = sub_data.get("metadata", {})
                for mk in meta_keys:
                    if mk not in meta:
                        self.logger.warning(f"{sub_name} metadata에 '{mk}' 누락")
                        ok = False

        if ok:
            self.logger.info("emotions_data 스키마 검증 완료(최상위 백필 적용).")
        else:
            self.logger.warning("스키마 검증 중 일부 항목 부족이 발견되었습니다(최상위는 백필 처리).")
        return ok

    def _validate_emotions_data_v2(self, data: Dict[str, Any]) -> bool:
        if not isinstance(data, dict):
            return False
        for _, node in data.items():
            if not self._validate_emotion_structure_v2(node):
                return False
            if not self._validate_metadata_v2(node.get("metadata", {})):
                return False
            if not self._validate_sub_emotions_v2(node.get("sub_emotions", {})):
                return False
        return True

    def _validate_emotion_structure_v2(self, node: Dict[str, Any]) -> bool:
        for rk in self.validation_rules["required_keys"]:
            if rk not in node:
                return False
        return True

    def _validate_metadata_v2(self, md: Dict[str, Any]) -> bool:
        for k in self.validation_rules["metadata_keys"]:
            if k not in md:
                return False
        return True

    def _validate_sub_emotions_v2(self, sub: Dict[str, Any]) -> bool:
        if not isinstance(sub, dict):
            return False
        for _, info in sub.items():
            if not isinstance(info, dict):
                return False
        return True

    # ------------------------------------------------------------------- index
    def _rebuild_indexes(self) -> None:
        """emotions_data를 평탄화하여 빠른 조회 인덱스 구축."""
        self._flat_index.clear()
        self._category_map.clear()
        self._top_ids = []

        def walk(node: Dict[str, Any]):
            md = node.get("metadata", {}) or {}
            eid = md.get("emotion_id")
            cat = md.get("primary_category")
            if isinstance(eid, str) and eid:
                self._flat_index[eid] = node
                if isinstance(cat, str) and cat:
                    self._category_map[eid] = cat

            for _, child in (node.get("sub_emotions", {}) or {}).items():
                if isinstance(child, dict):
                    walk(child)

        for top_id, top_node in (self.emotions_data or {}).items():
            if isinstance(top_node, dict):
                self._top_ids.append(top_id)
                walk(top_node)

        # 대표 카테고리 키 보강(희/노/애/락가 top_id가 아닐 수 있는 JSON도 안전)
        for k in ["희", "노", "애", "락"]:
            if k in self.emotions_data:
                self._category_map.setdefault(k, k)
                self._flat_index.setdefault(k, self.emotions_data[k])

        self.logger.debug(
            f"인덱스 구축 완료: flat={len(self._flat_index)}, cat_map={len(self._category_map)}, tops={len(self._top_ids)}"
        )

    # ------------------------------------------------------------------ query
    def _normalize_emotion_id_v1(self, eid: str) -> str:
        if not eid:
            return ""
        return eid.replace("_", "-")

    def _create_default_emotion_result_v2(self) -> Dict[str, Any]:
        return {
            "primary_id": None,
            "primary_category": "기타",
            "emotion_mapping": {},
            "intensity": None,
            "related_info": None,
        }

    def _extract_related_emotions_v2(self, emotion_data: Optional[Dict[str, Any]]) -> Dict[str, List[str]]:
        if not emotion_data:
            return {"positive": [], "negative": [], "neutral": []}
        prof = emotion_data.get("emotion_profile", {}) or {}
        rel = prof.get("related_emotions", {}) or {}
        return {
            "positive": rel.get("positive", []) or [],
            "negative": rel.get("negative", []) or [],
            "neutral": rel.get("neutral", []) or [],
        }

    def _build_id_mapping_cache_v2(self):
        """emotion_id -> primary_category 매핑 캐시 구축"""
        self.id_mapping_cache = dict(self._category_map)


# =============================================================================
# EmotionWeightCalculator (Soft-Ensemble Core)
# =============================================================================
class EmotionWeightCalculator:
    """
    11개 전처리 모듈의 결과를 종합하여 'Soft-Ensemble' 가중치를 계산하는 핵심 클래스.
    config.ENSEMBLE_WEIGHTS 설정을 기반으로 각 모듈의 신뢰도를 조정하고,
    최종적으로 통합된 감정 분포와 근거(contribution)를 산출한다.
    """

    def __init__(
        self, 
        data_manager: "WeightEmotionDataManager", 
        logger: "EmotionLogger",
        use_cuda: bool = False
    ):
        self.dm = data_manager
        self.logger = logger.get_logger()
        self.use_cuda = use_cuda
        
        # 설정 로드 (가중치)
        try:
            from config import ENSEMBLE_WEIGHTS
            self.ensemble_weights = ENSEMBLE_WEIGHTS
        except ImportError:
            self.ensemble_weights = {"default": 0.5}
            self.logger.warning("config.ENSEMBLE_WEIGHTS 로드 실패, 기본값 사용")

    def calculate_emotion_weights(self, text: str) -> Dict[str, Any]:
        """
        단독 실행용 메서드 (하위 호환성).
        실제로는 calculate_integrated_weights를 호출해야 함.
        """
        return self.calculate_integrated_weights(text, module_results={})

    def calculate_integrated_weights(
        self, 
        text: str, 
        module_results: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        [Soft-Ensemble] 11개 모듈의 결과를 통합하여 최종 감정 분포 산출.
        
        Args:
            text: 입력 텍스트
            module_results: 오케스트레이터가 수집한 각 모듈의 실행 결과 dict
            
        Returns:
            dict: {
                "main_distribution": {"희": 0.3, "노": 0.7, ...},
                "contribution": {"intensity_analysis": 0.4, ...},
                "success": True
            }
        """
        if not module_results:
            module_results = {}
            
        scores: Dict[str, float] = defaultdict(float)
        contributions: Dict[str, float] = defaultdict(float)
        total_weight_sum = 0.0
        
        # 1. 모듈별 점수 집계
        for module_name, result in module_results.items():
            if not isinstance(result, dict):
                continue
                
            weight = self.ensemble_weights.get(module_name, self.ensemble_weights.get("default", 0.5))
            if weight <= 0:
                continue

            # 모듈별 감정 추출 로직 (어댑터)
            extracted_probs = self._extract_probs_from_module(module_name, result)
            
            if extracted_probs:
                # 정규화된 확률 * 모듈 가중치
                for emo, prob in extracted_probs.items():
                    score = prob * weight
                    scores[emo] += score
                    
                # 기여도 기록 (단순 합산)
                contributions[module_name] += weight
                total_weight_sum += weight

        # 2. 최종 분포 정규화 (Softmax or Ratio)
        final_dist = self._normalize_distribution(scores)
        
        # 3. 대표 감정 도출
        if final_dist:
            dominant = max(final_dist.items(), key=lambda x: x[1])
            main_emotion = dominant[0]
            confidence = dominant[1]
        else:
            main_emotion = None
            confidence = 0.0

        return {
            "success": True,
            "main_distribution": final_dist,
            "dominant_emotion": main_emotion,
            "confidence": confidence,
            "contribution": dict(contributions),
            "meta": {
                "total_modules": len(module_results),
                "active_weight_sum": total_weight_sum
            }
        }

    def _extract_probs_from_module(self, module_name: str, result: Dict[str, Any]) -> Dict[str, float]:
        """각 모듈의 상이한 결과 포맷에서 표준 감정 확률(희/노/애/락) 추출"""
        probs = {}
        
        # A. emotion_classification (이미 확률 분포임)
        if module_name == "emotion_classification":
            # router.probs > topk_main > main_emotion
            if "router" in result and "probs" in result["router"]:
                return result["router"]["probs"]
            
        # B. intensity_analysis (감정별 강도)
        elif module_name == "intensity_analysis":
            # {"metrics": {"노": {"intensity": 0.8, ...}}}
            metrics = result.get("metrics", {})
            for k, v in metrics.items():
                if k in ["희", "노", "애", "락"] and isinstance(v, dict):
                    probs[k] = float(v.get("intensity", 0.0))
                    
        # C. pattern_extractor (매칭된 패턴의 감정)
        elif module_name == "pattern_extractor":
            # matches: [{"emotion": "노", "score": 1.0}, ...]
            matches = result.get("matches", [])
            for m in matches:
                e = m.get("emotion")
                s = float(m.get("score", 1.0))
                if e in ["희", "노", "애", "락"]:
                    probs[e] = probs.get(e, 0.0) + s
                    
        # D. linguistic_matcher (언어적 매칭)
        elif module_name == "linguistic_matcher":
            # matches list or direct keys
            matches = result.get("matches", [])
            for m in matches:
                # {"label": "노", ...}
                e = m.get("label")
                if e in ["희", "노", "애", "락"]:
                    probs[e] = probs.get(e, 0.0) + 1.0
                    
        # E. context_analysis (부정/긍정 점수)
        elif module_name == "context_analysis":
            # context_score: -1.0 ~ 1.0 (음수: 노/애, 양수: 희/락)
            score = float(result.get("context_score", 0.0))
            if score < 0:
                probs["노"] = abs(score) * 0.6
                probs["애"] = abs(score) * 0.4
            elif score > 0:
                probs["희"] = score * 0.6
                probs["락"] = score * 0.4
                
        # F. situation_analyzer / complex_analyzer / etc.
        # (필요 시 추가 구현, 기본적으로 main_emotion 키 확인)
        else:
            main = result.get("main_emotion") or result.get("primary_emotion")
            if main in ["희", "노", "애", "락"]:
                probs[main] = 1.0
                
        # 모듈 내부 정규화 (합이 1이 되도록)
        total = sum(probs.values())
        if total > 0:
            return {k: v / total for k, v in probs.items()}
        return probs

    def _normalize_distribution(self, scores: Dict[str, float]) -> Dict[str, float]:
        """점수 -> 확률 정규화 (Softmax 유사 효과)"""
        # 0점 방지 및 스무딩
        target_keys = ["희", "노", "애", "락"]
        safe_scores = {k: scores.get(k, 0.0) for k in target_keys}
        
        total = sum(safe_scores.values())
        if total <= 0:
            # 정보 없음 -> 균등 분포 (또는 중립)
            return {k: 0.25 for k in target_keys}
            
        return {k: v / total for k, v in safe_scores.items()}



    def _find_top_id_by_emotion_id_v1(self, mapping: Dict[str, Dict[str, Any]], emotion_id: str) -> Optional[str]:
        if emotion_id in mapping:
            return emotion_id
        for tid, tdata in mapping.items():
            if emotion_id in tdata.get("sub_emotions", {}):
                return tid
        return None

    def _build_recursive_mapping_v1(self) -> Dict[str, Dict[str, Any]]:
        base: Dict[str, Dict[str, Any]] = {}
        for top_id, top_data in (self.emotions_data or {}).items():
            meta = top_data.get("metadata", {}) or {}
            base[top_id] = {
                "category": meta.get("primary_category", ""),
                "sub_emotions": {},
                "intensity_range": [0.0, 1.0],
            }
            self._populate_recursive_sub_emotions_v1(base, top_id, top_data)
        return base

    def _populate_recursive_sub_emotions_v1(self, base: Dict[str, Dict[str, Any]], top_id: str, cur: Dict[str, Any]):
        for sub_name, sub in (cur.get("sub_emotions", {}) or {}).items():
            md = sub.get("metadata", {}) or {}
            sid = md.get("emotion_id", "")
            cplx = md.get("emotion_complexity", "")
            prof = sub.get("emotion_profile", {}) or {}
            intensities = prof.get("intensity_levels", {}) or {}
            keywords = prof.get("core_keywords", []) or []
            if sid:
                base[top_id]["sub_emotions"][sid] = {
                    "name": sub_name,
                    "complexity": cplx,
                    "intensity_levels": intensities,
                    "core_keywords": keywords,
                }
            if isinstance(sub.get("sub_emotions", {}), dict):
                self._populate_recursive_sub_emotions_v1(base, top_id, sub)

    def find_emotion_data_v1(self, emotion_id: str) -> Optional[Dict[str, Any]]:
        try:
            norm = self._normalize_emotion_id_v1(emotion_id)
            # 인덱스 우선
            node = self._flat_index.get(norm)
            if node:
                return self._extract_node_payload(node)

            # fallback: 탐색
            for _, top_node in (self.emotions_data or {}).items():
                res = self._search_sub_emotion_v1(top_node, norm)
                if res:
                    return res

            self.logger.warning(f"emotion_id {emotion_id} ({norm}) 데이터 없음")
            return None
        except Exception as e:
            self.logger.error(f"find_emotion_data_v1 오류: {e}", exc_info=True)
            return None

    def _search_sub_emotion_v1(self, node: Dict[str, Any], target_id: str) -> Optional[Dict[str, Any]]:
        try:
            for _, sub in (node.get("sub_emotions", {}) or {}).items():
                md = sub.get("metadata", {}) or {}
                sid = md.get("emotion_id", "")
                if sid == target_id:
                    return self._extract_node_payload(sub)
                if isinstance(sub.get("sub_emotions", {}), dict):
                    r = self._search_sub_emotion_v1(sub, target_id)
                    if r:
                        return r
            return None
        except Exception as e:
            self.logger.error(f"_search_sub_emotion_v1 오류: {e}", exc_info=True)
            return None

    def _extract_node_payload(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "metadata": data.get("metadata", {}),
            "emotion_profile": {
                "intensity_levels": (data.get("emotion_profile", {}) or {}).get("intensity_levels", {}),
                "related_emotions": (data.get("emotion_profile", {}) or {}).get("related_emotions", {}),
                "core_keywords": (data.get("emotion_profile", {}) or {}).get("core_keywords", []),
            },
            "context_patterns": {"situations": (data.get("context_patterns", {}) or {}).get("situations", {})},
            "linguistic_patterns": data.get("linguistic_patterns", {}),
            "emotion_transitions": data.get("emotion_transitions", {}),
            "sentiment_analysis": data.get("sentiment_analysis", {}),
            "ml_training_metadata": data.get("ml_training_metadata", {}),
        }

    # ----------------------------------------------------------------- primary
    def get_primary_emotion_v1(self, emotion_id: str) -> Dict[str, Any]:
        try:
            mapping = self._build_recursive_mapping_v1()
            top_id = self._find_top_id_by_emotion_id_v1(mapping, emotion_id)
            if not top_id or top_id not in mapping:
                self.logger.warning(f"알 수 없는 대표 감정 ID: {emotion_id}")
                return self._create_default_emotion_result_v2()

            primary_info = mapping[top_id]
            sub_info = primary_info["sub_emotions"].get(emotion_id)
            emotion_data = self.find_emotion_data_v1(emotion_id)
            related_emotions = self._extract_related_emotions_v2(emotion_data)
            return {
                "primary_id": top_id,
                "primary_category": primary_info.get("category", "기타"),
                "emotion_mapping": {
                    "sub_emotion": sub_info,
                    "intensity_range": primary_info.get("intensity_range", [0.0, 1.0]),
                },
                "intensity": {"levels": sub_info["intensity_levels"] if sub_info else None, "current": None},
                "related_info": {
                    "keywords": sub_info["core_keywords"] if sub_info else [],
                    "related_emotions": related_emotions,
                },
            }
        except Exception as e:
            self.logger.error(f"get_primary_emotion_v1 오류: {e}", exc_info=True)
            return self._create_default_emotion_result_v2()

    @lru_cache(maxsize=1024)
    def get_primary_emotion(self, emotion_id: str) -> Dict[str, Any]:
        cache_key = f"primary_{emotion_id}"
        c = self._get_from_cache_v2(cache_key)
        if c:
            return c
        try:
            mapping = self._build_recursive_mapping_v1()
            top_id = self._find_top_id_by_emotion_id_v1(mapping, emotion_id)
            res = self._create_default_emotion_result_v2()
            if top_id and top_id in mapping:
                res = self._build_primary_emotion_result_v2(top_id, emotion_id, mapping)
            self._add_to_cache_v2(cache_key, res)
            return res
        except Exception as e:
            self.logger.error(f"get_primary_emotion 오류: {e}", exc_info=True)
            return self._create_default_emotion_result_v2()

    def _build_primary_emotion_result_v2(self, top_id: str, emotion_id: str, mapping: Dict[str, Any]) -> Dict[str, Any]:
        primary_info = mapping[top_id]
        sub_info = primary_info["sub_emotions"].get(emotion_id)
        emotion_data = self.find_emotion_data(emotion_id)
        related_emotions = self._extract_related_emotions_v2(emotion_data)
        return {
            "primary_id": top_id,
            "primary_category": primary_info.get("category", "기타"),
            "emotion_mapping": {
                "sub_emotion": sub_info,
                "intensity_range": primary_info.get("intensity_range", [0.0, 1.0]),
            },
            "intensity": {"levels": sub_info["intensity_levels"] if sub_info else None, "current": None},
            "related_info": {
                "keywords": sub_info["core_keywords"] if sub_info else [],
                "related_emotions": related_emotions,
            },
        }

    # ------------------------------------------------------------- public find
    @lru_cache(maxsize=1024)
    def find_emotion_data(self, emotion_id: str) -> Optional[Dict[str, Any]]:
        cache_key = f"emotion_data_{emotion_id}"
        c = self._get_from_cache_v2(cache_key)
        if c:
            return c
        try:
            res = self._find_emotion_data_internal_v2(emotion_id)
            if res:
                self._add_to_cache_v2(cache_key, res)
            return res
        except Exception as e:
            self.logger.error(f"find_emotion_data 오류: {e}", exc_info=True)
            return None

    def _find_emotion_data_internal_v2(self, emotion_id: str) -> Optional[Dict[str, Any]]:
        return self.find_emotion_data_v1(emotion_id)

    # --------------------------------------------------------------- id-mapping
    def _build_id_mapping_cache_v2(self):
        self.id_mapping_cache.clear()
        for eid, node in self._flat_index.items():
            cat = (node.get("metadata", {}) or {}).get("primary_category")
            if isinstance(cat, str) and cat in {"희", "노", "애", "락"}:
                self.id_mapping_cache[eid] = cat

    # -------------------------------------------------------------- cache utils
    def _clear_all_caches_v2(self):
        self.cache.clear()
        self.cache_timestamps.clear()
        self.get_primary_emotion.cache_clear()
        self.find_emotion_data.cache_clear()

    def _get_from_cache_v2(self, key: str) -> Optional[Any]:
        if key in self.cache:
            if time.time() - self.cache_timestamps.get(key, 0) < self.cache_timeout:
                return self.cache[key]
            # TTL 만료
            self.cache.pop(key, None)
            self.cache_timestamps.pop(key, None)
        return None

    def _add_to_cache_v2(self, key: str, value: Any):
        self.cache[key] = value
        self.cache_timestamps[key] = time.time()
        if len(self.cache) > 10000:
            # 가장 오래된 항목 제거
            oldest = min(self.cache_timestamps.items(), key=lambda x: x[1])[0]
            self.cache.pop(oldest, None)
            self.cache_timestamps.pop(oldest, None)



# =============================================================================
# EmotionWeightCalculator
# =============================================================================
class EmotionWeightCalculator:
    """ 가중치·통합 오케스트레이터 (CUDA 가속 지원) """

    # --------------------------------------------------------------------- init
    def __init__(
            self,
            data_manager: WeightEmotionDataManager = None,
            logger: EmotionLogger = None,
            use_parallel: bool = False,
            max_workers: int = 256,  # 3번 개선작업: 워커 수 극대화 (128→256)
            thresholds: Optional[Dict[str, float]] = None,
            seed: Optional[int] = None,
            use_cuda: bool = True,  # CUDA 사용 여부 (기본값: True)
    ):
        # 독립 모듈을 위한 기본값 설정
        if logger is None:
            import logging
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger.get_logger()
            
        if data_manager is None:
            self.data_manager = self._create_default_data_manager()
        else:
            self.data_manager = data_manager

        # CUDA 설정 (조건부 활성화)
        self.use_cuda = use_cuda and _CUDA_AVAILABLE
        self.device = None
        if self.use_cuda:
            try:
                self.device = torch.device("cuda:0")
                # 간단한 CUDA 테스트
                test_tensor = torch.randn(10, 10, device=self.device)
                _ = torch.matmul(test_tensor, test_tensor.t())
                torch.cuda.empty_cache()
                self.logger.info("[EmotionWeightCalculator] CUDA 가속 활성화 완료")
            except Exception as e:
                self.use_cuda = False
                self.device = None
                self.logger.warning(f"[EmotionWeightCalculator] CUDA 활성화 실패, CPU로 폴백: {e}")
        else:
            self.logger.info("[EmotionWeightCalculator] CPU 모드로 실행")
            
        # 성능 최적화 플래그 (환경변수 기반)
        self.fast_mode = os.environ.get("EWC_FAST", "0") == "1"
        self.topk_sub_limit = int(os.environ.get("INT_TOPK_SUB", "12"))
        self.pattern_similarity_threshold = float(os.environ.get("EWC_PATTERN_THRESHOLD", "0.0"))
        self.stage_similarity_threshold = float(os.environ.get("EWC_STAGE_THRESHOLD", "0.0"))
        
        # Aho-Corasick 자동자 초기화
        self._aho_automaton = None
        self._phrase2eids = None
        self._phr2ids = None
        
        # 토큰셋 캐시 (유사도 O(1)화)
        self._token_cache = {}
        self._pattern_token_cache = {}
            
        self.emotions_data = self.data_manager.emotions_data
        self.use_parallel = use_parallel or os.environ.get("EWC_PARALLEL", "0") == "1"
        self.max_workers = max_workers or int(os.environ.get("EWC_MAX_WORKERS", "128"))  # CPU 최적화: 32→128 증가 (4배)
        
        # 인덱스 초기화
        self._phrase2eids = None
        self._rx_phrase_shards = None
        
        # 경계 인식 정규식 캐시
        self._regex_cache: Dict[str, re.Pattern] = {}
        # 후보 프리필터용 샤드 크기 및 인덱스 캐시
        self._cand_shard = int(os.getenv("EWC_REGEX_SHARD", "256"))
        
        # QA 카운터 초기화
        self._qa = defaultdict(int)
        
        # 임계값 초기화
        self.thresholds = thresholds or {
            "stage_text_similarity": 0.3,
            "emotion_weight_min": 0.1,
            "situation_weight_min": 0.05
        }
        
        # 수치 계산 관련 속성 초기화
        self._diag_eps_base = 1e-8
        self._eig_max_retries = 3
        
        # 스레드 풀 실행기 초기화
        self.executor = None
        
        # 성능 최적화: 인덱싱된 패턴 매칭 (지연 로딩)
        self._pattern_index = {}
        self._emotion_patterns = {}
        self._pattern_index_built = False

        # CUDA 가속을 위한 추가 메서드 정의
        self._compute_interactions_cuda = self._create_cuda_interaction_computer()
        
        # 성능 최적화: 캐시 초기화
        self._calculation_cache = {}
        self._cache_max_size = 1000
        
        # 성능 최적화: 빠른 모드 설정
        self._fast_mode = os.environ.get("WEIGHT_CALCULATOR_FAST", "1") == "1"
        self._skip_complex_analysis = os.environ.get("WEIGHT_CALCULATOR_SIMPLE", "0") == "1"

    def _create_cuda_interaction_computer(self):
        """CUDA 기반 상호작용 계산기 생성"""
        if not self.use_cuda or not _TORCH_OK:
            return None
        
        def compute_interactions_cuda(pairs: List[Tuple[str, str]], weights: Dict[str, float], text: str) -> List[float]:
            """CUDA에서 배치 상호작용 계산"""
            try:
                # 텍스트를 토큰화하여 텐서로 변환
                text_tokens = (text or "").lower().split()
                text_tensor = torch.tensor([hash(token) % 1000 for token in text_tokens[:50]], 
                                         device=self.device, dtype=torch.float32)
                
                # 감정별 가중치 텐서 생성
                emotion_names = ["희", "노", "애", "락"]
                weight_values = [weights.get(emotion, 0.0) for emotion in emotion_names]
                weight_tensor = torch.tensor(weight_values, device=self.device, dtype=torch.float32)
                
                # 상호작용 계산 (간단한 벡터 연산)
                interactions = []
                for e1, e2 in pairs:
                    try:
                        idx1 = emotion_names.index(e1) if e1 in emotion_names else 0
                        idx2 = emotion_names.index(e2) if e2 in emotion_names else 0
                        
                        # 간단한 상호작용 모델 (CUDA에서 계산)
                        w1 = weight_tensor[idx1]
                        w2 = weight_tensor[idx2]
                        
                        # 텍스트 기반 보정
                        text_factor = torch.mean(text_tensor) if len(text_tensor) > 0 else torch.tensor(0.0, device=self.device)
                        interaction = w1 * w2 * 0.5 + text_factor * 0.1
                        
                        interactions.append(max(0.05, float(interaction.cpu().item())))
                    except Exception:
                        interactions.append(0.05)
                
                # 메모리 정리
                torch.cuda.empty_cache()
                return interactions
                
            except Exception as e:
                self.logger.warning(f"[_compute_interactions_cuda] CUDA 계산 실패: {e}")
                # CPU로 폴백
                return [0.05] * len(pairs)
        
        return compute_interactions_cuda

    def _calculate_emotion_weights_fast(self, text: str, text_hash: int) -> Dict[str, Any]:
        """빠른 모드: 단순 키워드 매칭 기반 감정 가중치 계산"""
        t0 = time.time()
        
        # 간단한 키워드 매칭
        text_lower = text.lower()
        
        # 기본 감정 키워드 (간단한 매핑)
        emotion_keywords = {
            "희": ["happy", "joy", "excited", "pleased", "delighted", "cheerful", "glad"],
            "노": ["angry", "mad", "furious", "irritated", "annoyed", "upset", "frustrated"],
            "애": ["sad", "depressed", "melancholy", "gloomy", "sorrowful", "unhappy", "blue"],
            "락": ["fun", "enjoy", "amused", "entertained", "pleasure", "recreation", "playful"]
        }
        
        # 키워드 매칭 카운트
        emotion_counts = {}
        for emotion, keywords in emotion_keywords.items():
            count = sum(1 for keyword in keywords if keyword in text_lower)
            emotion_counts[emotion] = count
        
        # 정규화
        total_count = sum(emotion_counts.values())
        if total_count > 0:
            for emotion in emotion_counts:
                emotion_counts[emotion] = emotion_counts[emotion] / total_count
        else:
            # 기본 균등 분배
            for emotion in emotion_keywords:
                emotion_counts[emotion] = 0.25
        
        # 결과 구성
        result = {
            "text": text,
            "matches": {"matched_phrases": [], "contextual": {}},
            "emotion_weights": emotion_counts,
            "weights": dict(emotion_counts),
            "metrics": {
                "processing_time": round(time.time() - t0, 6),
                "emotion_frequencies": emotion_counts,
                "context_scores": {},
                "fast_mode": True
            },
            "qa": {"fast_mode_used": 1}
        }
        
        # 캐시에 저장
        if len(self._calculation_cache) < self._cache_max_size:
            self._calculation_cache[text_hash] = result
        
        return result

    def _ensure_pattern_index(self):
        """패턴 인덱스가 구축되었는지 확인하고 필요시 구축"""
        if not self._pattern_index_built:
            self._build_pattern_index()
            self._pattern_index_built = True

    def _build_pattern_index(self):
        """패턴 인덱스 구축 - 성능 최적화 (지연 로딩)"""
        try:
            for primary_emotion, emotion_info in self.emotions_data.items():
                sub_emotions = emotion_info.get('sub_emotions', {}) or {}
                patterns = []
                
                for sub_emotion, sub_info in sub_emotions.items():
                    e_id = sub_info.get('metadata', {}).get('emotion_id', '')
                    emotion_profile = sub_info.get('emotion_profile', {}) or {}
                    intensity_levels = emotion_profile.get('intensity_levels', {}) or {}
                    
                    # intensity_examples 패턴 수집
                    for level in ('low', 'medium', 'high'):
                        exs = (intensity_levels.get('intensity_examples', {}) or {}).get(level, []) or []
                        for pattern in exs:
                            patterns.append({
                                'pattern': str(pattern).lower(),
                                'emotion_id': e_id,
                                'primary_emotion': primary_emotion,
                                'intensity': level,
                                'weight': {'low': 0.3, 'medium': 0.6, 'high': 1.0}.get(level, 0.5)
                            })
                    
                    # linguistic_patterns 패턴 수집
                    ling = sub_info.get('linguistic_patterns', {}) or {}
                    for kp in ling.get('key_phrases', []) or []:
                        phrase = str(kp.get('pattern', '')).lower()
                        if phrase:
                            patterns.append({
                                'pattern': phrase,
                                'emotion_id': e_id,
                                'primary_emotion': primary_emotion,
                                'intensity': 'medium',
                                'weight': float(kp.get('weight', 0.5))
                            })
                
                self._emotion_patterns[primary_emotion] = patterns
                
        except Exception as e:
            self.logger.warning(f"[_build_pattern_index] 패턴 인덱스 구축 실패: {e}")
            # 기본 패턴으로 폴백
            self._emotion_patterns = {
                '희': [{'pattern': '기쁨', 'emotion_id': 'joy', 'primary_emotion': '희', 'intensity': 'medium', 'weight': 0.8}],
                '노': [{'pattern': '분노', 'emotion_id': 'anger', 'primary_emotion': '노', 'intensity': 'medium', 'weight': 0.8}],
                '애': [{'pattern': '슬픔', 'emotion_id': 'sadness', 'primary_emotion': '애', 'intensity': 'medium', 'weight': 0.8}],
                '락': [{'pattern': '안정', 'emotion_id': 'calm', 'primary_emotion': '락', 'intensity': 'medium', 'weight': 0.8}]
            }

    def _create_default_data_manager(self):
        """독립 모듈을 위한 기본 데이터 매니저 생성"""
        class DefaultWeightDataManager:
            def __init__(self):
                self.emotions_data = {
                    "희": {"patterns": ["happy", "joy", "행복", "기쁨"], "weight": 1.0},
                    "노": {"patterns": ["angry", "mad", "분노", "화남"], "weight": 1.0},
                    "애": {"patterns": ["sad", "depressed", "슬픔", "우울"], "weight": 1.0},
                    "락": {"patterns": ["pleasure", "satisfaction", "만족", "환희"], "weight": 1.0},
                }
        
        return DefaultWeightDataManager()

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
            for rx in self._rx_phrase_shards:
                for m in rx.finditer(text):
                    cand |= set(self._phr2ids.get((m.group(0) or "").lower(), []))
        return cand

    def calculate_weights(self, text: str) -> Dict[str, Any]:
        """독립 모듈을 위한 간단한 가중치 계산"""
        try:
            # 기본 감정 패턴 매칭
            emotion_patterns = {
                "희": ["happy", "joy", "행복", "기쁨", "pleasure", "delight"],
                "노": ["angry", "mad", "분노", "화남", "furious", "rage"],
                "애": ["sad", "depressed", "슬픔", "우울", "gloomy", "sorrow"],
                "락": ["pleasure", "satisfaction", "만족", "환희", "content", "cheerful"],
            }
            
            text_lower = text.lower()
            emotion_weights = {}
            total_matches = 0
            
            for emotion, patterns in emotion_patterns.items():
                matches = sum(1 for pattern in patterns if pattern in text_lower)
                emotion_weights[emotion] = matches
                total_matches += matches
            
            # 정규화
            if total_matches > 0:
                for emotion in emotion_weights:
                    emotion_weights[emotion] = emotion_weights[emotion] / total_matches
            
            # 지배 감정 찾기
            dominant_emotion = max(emotion_weights.items(), key=lambda x: x[1])[0] if emotion_weights else None
            
            return {
                "emotion_weights": emotion_weights,
                "dominant_emotion": dominant_emotion,
                "total_matches": total_matches,
                "confidence": max(emotion_weights.values()) if emotion_weights else 0.0,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "success": True
            }
            
        except Exception as e:
            return {
                "emotion_weights": {},
                "dominant_emotion": None,
                "total_matches": 0,
                "confidence": 0.0,
                "error": str(e),
                "success": False
            }

    def _normalize_distribution(self, weights: Dict[str, float]) -> Dict[str, float]:
        """가중치 분포 정규화 (합이 1이 되도록)"""
        if not weights:
            return {"희": 0.25, "노": 0.25, "애": 0.25, "락": 0.25}
        
        # 음수 제거
        safe_weights = {k: max(0.0, v) for k, v in weights.items()}
        total = sum(safe_weights.values())
        
        if total <= 1e-9:
            return {"희": 0.25, "노": 0.25, "애": 0.25, "락": 0.25}
            
        return {k: v / total for k, v in safe_weights.items()}

    def calculate_integrated_weights(
        self,
        text: str,
        module_results: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        [전략적 승격] Soft-Ensemble 통합 가중치 계산
        11개 모듈의 분석 결과를 종합하여 보정된 확률 분포를 산출합니다.
        """
        # 1. 기본 텍스트 기반 가중치 (Base Layer)
        # calculate_weights 대신 더 정교한 calculate_emotion_weights 사용 권장 (존재한다면)
        # 여기서는 안전하게 현재 클래스의 메서드 호출
        if hasattr(self, 'calculate_emotion_weights'):
            base_res = self.calculate_emotion_weights(text)
        else:
            base_res = self.calculate_weights(text)
            
        # 베이스 가중치 추출
        base_weights = base_res.get("emotion_weights", {}) or base_res.get("weights", {})
        if not base_weights:
            base_weights = {"희": 0.25, "노": 0.25, "애": 0.25, "락": 0.25}
            
        # 모듈 결과가 없으면 베이스 반환
        if not module_results:
            norm_weights = self._normalize_distribution(base_weights)
            return {
                "weights": norm_weights,
                "main_distribution": norm_weights,
                "details": base_res,
                "contribution": {"text_base": 1.0}
            }

        # 2. 모듈별 가중치 통합 (Soft Ensemble Layer)
        integrated = dict(base_weights)
        contributions = {"text_base": 1.0}
        
        # Helper: 가중치 가산 함수
        def _add_weight(emotion, score, source, weight_factor=1.0):
            if emotion in integrated:
                val = score * weight_factor
                integrated[emotion] += val
                contributions[source] = contributions.get(source, 0.0) + val

        # A. Emotion Classification (모델 1차 확률) - 신뢰도 높음
        if "emotion_classification" in module_results:
            ec = module_results["emotion_classification"]
            # topk_main 처리
            topk = ec.get("topk_main", []) or ec.get("topk", [])
            if isinstance(topk, list):
                for item in topk:
                    if isinstance(item, dict):
                        emo = item.get("label") or item.get("emotion")
                        prob = float(item.get("score") or item.get("probability") or 0.0)
                        if emo in integrated:
                            _add_weight(emo, prob, "classification", 1.5) # 1.5배 가중

        # B. Intensity Analyzer (강도 부스터)
        if "intensity_analyzer" in module_results:
            ia = module_results["intensity_analyzer"]
            # intensity_levels 활용
            levels = ia.get("intensity_levels", {})
            if isinstance(levels, dict):
                for emo, level in levels.items():
                    if emo in integrated and isinstance(level, str):
                        # High=0.3, Medium=0.15, Low=0.05 가산
                        boost = {"high": 0.3, "medium": 0.15, "low": 0.05}.get(level.lower(), 0.0)
                        _add_weight(emo, boost, "intensity")

        # C. Linguistic Matcher (언어적 확신)
        if "linguistic_matcher" in module_results:
            lm = module_results["linguistic_matcher"]
            # pattern_matches 활용
            matches = lm.get("pattern_matches", [])
            if matches:
                # 매칭된 패턴 수만큼 소폭 가산
                for m in matches:
                    emo = m.get("primary_emotion")
                    if emo in integrated:
                        _add_weight(emo, 0.1, "linguistic")

        # D. Context Analysis (맥락적 단서)
        if "context_analysis" in module_results:
            ca = module_results["context_analysis"]
            # detected_situations 활용
            sits = ca.get("detected_situations", [])
            if sits:
                for s in sits:
                    emo = s.get("related_emotion")
                    if emo in integrated:
                        _add_weight(emo, 0.2, "context")

        # E. Complex Analyzer (복합 감정)
        if "complex_analyzer" in module_results:
            cpx = module_results["complex_analyzer"]
            # dominant_emotions 활용
            doms = cpx.get("dominant_emotions", [])
            if doms:
                for d in doms:
                    emo = d.get("emotion")
                    score = float(d.get("score", 0.0))
                    if emo in integrated:
                        _add_weight(emo, score * 0.5, "complex")

        # 3. 최종 정규화
        final_weights = self._normalize_distribution(integrated)
        
        # 4. 엔트로피 기반 불확실성 계산 (Optional)
        import math
        probs = list(final_weights.values())
        entropy = -sum(p * math.log(p + 1e-9) for p in probs)
        
        return {
            "weights": final_weights,           # 호환성용
            "main_distribution": final_weights, # Soft Ensemble용 핵심 결과
            "raw_scores": integrated,           # 정규화 전 원본 점수
            "contribution": contributions,      # XAI (설명 가능성) 데이터
            "entropy": entropy,
            "details": base_res
        }

        # numpy 존재 여부 플래그(디버그 가독성)
        try:
            self._qa["np_missing"] = 1 if np is None else 0
        except Exception:
            self._qa["np_missing"] = 0

    # ----------------------------------------------------------------- lifecycle
    def shutdown_executor(self) -> None:
        if self.executor:
            try:
                self.executor.shutdown(wait=True)
                self.logger.info("[EWC] ThreadPoolExecutor 정상 종료")
            except Exception as e:
                self.logger.error(f"[EWC] ThreadPoolExecutor 종료 오류: {e}")

    # ---------------------------------------------------------------- text utils
    @lru_cache(maxsize=5000)
    def _tokenize_text(self, text: str) -> List[str]:
        return re.findall(r"[가-힣]+|[a-zA-Z]+|\d+", text)

    def _bregex(self, text: str) -> re.Pattern:
        if not isinstance(text, str) or not text.strip():
            return re.compile(r"$^")
        base = text.strip()
        is_ko = bool(re.search(r"[가-힣]", base))
        key = f"b::{'ko' if is_ko else 'wb'}::{base}"
        rg = self._regex_cache.get(key)
        if rg:
            return rg
        try:
            if is_ko:
                # 한국어: 후행 조사/어미 0~2자 허용 + 보수적 경계 대안
                pat = rf"(?<![가-힣A-Za-z0-9]){re.escape(base)}(?![가-힣A-Za-z0-9])|{re.escape(base)}[가-힣]{{0,2}}"
            else:
                pat = rf"(?<!\w){re.escape(base)}(?!\w)"
            rg = re.compile(pat, re.IGNORECASE)
        except re.error:
            rg = re.compile(re.escape(base), re.IGNORECASE)
        self._regex_cache[key] = rg
        return rg

    def _get_cached_tokens(self, text: str) -> Set[str]:
        """토큰화 결과를 캐시하여 반복 계산 방지"""
        if text not in self._token_cache:
            self._token_cache[text] = set(self._tokenize_text(text))
        return self._token_cache[text]
    
    def _get_cached_pattern_tokens(self, pattern: str) -> Set[str]:
        """패턴 토큰화 결과를 캐시하여 반복 계산 방지"""
        if pattern not in self._pattern_token_cache:
            self._pattern_token_cache[pattern] = set(self._tokenize_text(pattern))
        return self._pattern_token_cache[pattern]

    def _calculate_text_similarity_fast(self, text_a: str, text_b: str) -> float:
        """캐시된 토큰셋을 사용한 빠른 유사도 계산"""
        if not text_a or not text_b:
            return 0.0
        a = self._get_cached_tokens(text_a.lower())
        b = self._get_cached_tokens(text_b.lower())
        if not a or not b:
            return 0.0
        # Jaccard
        j = len(a & b) / len(a | b)
        # Boundary regex hit ratio (in b tokens against a text)
        hits = 0
        for tok in b:
            if self._bregex(tok).search(text_a):
                hits += 1
        dice = (2 * hits) / (len(a) + len(b))
        return min(1.0, 0.6 * j + 0.4 * dice)

    def compute_text_similarity(self, text_a: str, text_b: str) -> float:
        return self._calculate_text_similarity_fast(text_a, text_b)

    def _ensure_phrase_index(self):
        if self._phrase2eids is not None and self._rx_phrase_shards is not None:
            return
        from collections import defaultdict
        bank: List[str] = []
        p2e: Dict[str, Set[str]] = defaultdict(set)
        try:
            for _, em in (self.emotions_data or {}).items():
                for _, sd in (em.get('sub_emotions', {}) or {}).items():
                    eid = (sd.get('metadata', {}) or {}).get('emotion_id', '')
                    ep = (sd.get('emotion_profile', {}) or {})
                    core = ep.get('core_keywords', []) or []
                    ctx = (sd.get('context_patterns', {}) or {}).get('situations', {}) or {}
                    for w in core:
                        w = str(w).lower().strip()
                        if w:
                            bank.append(w); p2e[w].add(eid)
                    for s in ctx.values():
                        for k in ('keywords', 'variations', 'examples'):
                            for w in (s.get(k) or []):
                                w = str(w).lower().strip()
                                if w:
                                    bank.append(w); p2e[w].add(eid)
            
            # Aho-Corasick 자동자 구축 (가능한 경우)
            if _AHO_OK and bank:
                try:
                    automaton = ahocorasick.Automaton()
                    unique_words = set(bank)
                    for word in unique_words:
                        eids = list(p2e[word])
                        automaton.add_word(word, eids)
                    automaton.make_automaton()
                    self._aho_automaton = automaton
                    self.logger.debug(f"Aho-Corasick automaton built with {len(unique_words)} phrases")
                except Exception as e:
                    self.logger.warning(f"Failed to build Aho-Corasick automaton: {e}")
                    self._aho_automaton = None
            
            import re as _re
            words = sorted(set(bank))
            shards: List[re.Pattern] = []
            step = max(1, int(getattr(self, "_cand_shard", 256)))
            for i in range(0, len(words), step):
                alt = "|".join(sorted(map(_re.escape, words[i:i + step]), key=len, reverse=True))
                shards.append(_re.compile(alt))
            self._phrase2eids, self._rx_phrase_shards = p2e, tuple(shards)
        except Exception:
            # 실패 시 비활성화
            self._phrase2eids, self._rx_phrase_shards = {}, tuple()

    # ---------------------------------------------------------- matching (JSON)
    def _compute_compound_factor(self, emotion_a: str, emotion_b: str, weight_a: float, weight_b: float) -> float:
        """
        JSON 주도 복합 보정치(시너지/충돌/전이 단서) 계산.
        - 입력은 대표 감정명(희/노/애/락) 또는 세부 ID일 수 있음. 여기서는 대표 카테고리명을 받도록 calculate_compound_emotions에서 호출함.
        - 내부에서 해당 대표 카테고리의 전체 하위 노드를 순회하여 evidence를 집계.
        - 결과는 [-0.10, +0.15] 범위 내에서 반환되며, min(weight_a, weight_b)에 따라 약하게 스케일 조정.
        규칙:
          1) 하드코딩 최소화: synergy/conflict/dependencies/related_emotions/transitions 전부 EMOTIONS.json에서 유도
          2) 4×N 확장 안전: 모든 하위 노드 순회
          3) 스키마 변경 없음: 누락키는 안전 보정
        """
        # ---------------------- 기본 방어 ----------------------
        try:
            if not self.emotions_data:
                return 0.0

            # 대표 카테고리 top id 확인
            top_a = self._find_top_id_by_primary_category(emotion_a) or emotion_a
            top_b = self._find_top_id_by_primary_category(emotion_b) or emotion_b
            node_a = self.emotions_data.get(top_a)
            node_b = self.emotions_data.get(top_b)
            if not isinstance(node_a, dict) or not isinstance(node_b, dict):
                return 0.0

            # ---------------------- 유틸 ----------------------
            def _iter_nodes(root: Dict[str, Any]):
                """루트(대표) 포함, 모든 하위 노드를 깊이우선 순회."""
                stack = [root]
                while stack:
                    cur = stack.pop()
                    yield cur
                    for _, child in (cur.get("sub_emotions", {}) or {}).items():
                        if isinstance(child, dict):
                            stack.append(child)

            def _tokset(x: Any) -> set:
                """문자/리스트 혼용을 대비한 소문자 토큰 집합화."""
                out = set()
                if isinstance(x, str):
                    if x.strip():
                        out.add(x.lower().strip())
                elif isinstance(x, list):
                    for it in x:
                        if isinstance(it, str) and it.strip():
                            out.add(it.lower().strip())
                return out

            def _collect_alias_tokens(n: Dict[str, Any]) -> set:
                """해당 노드를 대표하는 문자열 후보(이름/ID/서브카테고리 등) 토큰."""
                md = (n.get("metadata", {}) or {})
                prof = (n.get("emotion_profile", {}) or {})
                toks = set()
                for k in ["emotion_id", "primary_category", "sub_category"]:
                    v = md.get(k)
                    if isinstance(v, str) and v.strip():
                        toks.add(v.lower().strip())
                # core_keywords도 약한 별칭으로 사용
                for kw in (prof.get("core_keywords", []) or []):
                    if isinstance(kw, str) and kw.strip():
                        toks.add(kw.lower().strip())
                return toks

            # polarity 보조(둘 다 양성/음성 여부)
            def _polarity(cat: str) -> str:
                if cat in {"희", "락"}:
                    return "positive"
                if cat in {"노", "애"}:
                    return "negative"
                return "neutral"

            pol_a = _polarity(emotion_a)
            pol_b = _polarity(emotion_b)

            # ---------------------- 증거 집계 ----------------------
            synergy_hits = 0.0
            conflict_hits = 0.0
            trans_link_hits = 0.0

            # B집합 별칭 토큰(텍스트 문자열 기반 참조가 많기 때문)
            alias_b_all = set()
            for nb in _iter_nodes(node_b):
                alias_b_all |= _collect_alias_tokens(nb)

            # A의 전체 노드 순회하며 evidence 수집
            for na in _iter_nodes(node_a):
                md_a = (na.get("metadata", {}) or {})
                id_a = str(md_a.get("emotion_id", "")).lower().strip()

                # --- 1) synergy/conflict (ml_training_metadata, dependencies, related_emotions) ---
                ml_a = (na.get("ml_training_metadata", {}) or {})
                syn_a = _tokset(ml_a.get("synergy_with", []))
                con_a = _tokset(ml_a.get("conflict_with", []))

                # dependencies
                for dep in (na.get("emotion_transitions", {}) or {}).get("emotion_dependencies", []) or []:
                    if not isinstance(dep, dict):
                        continue
                    tgt = str(dep.get("dependent_emotion", "")).lower().strip()
                    rel = str(dep.get("relationship_type", "")).lower().strip()
                    if tgt:
                        if rel == "enhances":
                            syn_a.add(tgt)
                        elif rel == "conflicts":
                            con_a.add(tgt)

                # related_emotions
                rel_a = (na.get("emotion_profile", {}) or {}).get("related_emotions", {}) or {}
                for t in _tokset(rel_a.get("positive", [])):
                    syn_a.add(t)
                for t in _tokset(rel_a.get("negative", [])):
                    con_a.add(t)
                # neutral은 약한 연결로 보되 점수에는 반영하지 않음(노이즈 방지)

                # B alias에 포함되면 히트
                # - 직접 문자열 비교 외에도 대표 카테고리명(희/노/애/락) 매칭도 허용
                if syn_a & alias_b_all or emotion_b in syn_a:
                    synergy_hits += 1.0
                if con_a & alias_b_all or emotion_b in con_a:
                    conflict_hits += 1.0

                # --- 2) transition pattern (from/to 문자열 연결) ---
                trs = (na.get("emotion_transitions", {}) or {}).get("patterns", []) or []
                for p in trs:
                    if not isinstance(p, dict):
                        continue
                    fa = str(p.get("from_emotion", "")).lower().strip()
                    ta = str(p.get("to_emotion", "")).lower().strip()
                    # B 별칭/카테고리가 from/to에 걸리면 한쪽 방향 연결로 가산
                    if fa and (fa in alias_b_all or fa == emotion_b.lower()):
                        trans_link_hits += 0.5
                    if ta and (ta in alias_b_all or ta == emotion_b.lower()):
                        trans_link_hits += 0.5

            # ---------------------- 스코어 산출 ----------------------
            # 기본 가중(히트수 × 단위기여)
            base = (0.02 * synergy_hits) - (0.02 * conflict_hits) + (0.01 * trans_link_hits)

            # polarity 보조: 양의-양의 약 +, 양의-음성 약 -
            if pol_a == "positive" and pol_b == "positive":
                base += 0.01
            elif {"positive", "negative"} <= {pol_a, pol_b}:
                base -= 0.01

            # 상호 가중치 반영(두 감정이 모두 의미있을 때만 조금 더 반영)
            scale = 0.5 + min(max(weight_a, 0.0), max(weight_b, 0.0))
            score = base * scale

            # 클리핑
            return max(-0.10, min(0.15, score))

        except Exception as e:
            # 안전 실패 시 보정 0
            self.logger.error(f"_compute_compound_factor 오류: {e}", exc_info=True)
            return 0.0

    # EmotionWeightCalculator 내부에 추가 (기존 메서드 유지)
    def _calculate_intensity_match(
            self,
            intensity1: Dict[str, Any],
            intensity2: Dict[str, Any],
            text: str
    ) -> float:
        """
        호환 래퍼: 예전 이름(_calculate_intensity_match)으로 호출되는 곳을
        새로운 점수 함수(_intensity_match_score)에 연결.
        """
        try:
            return float(self._intensity_match_score(intensity1, intensity2, text))
        except Exception:
            # 안전 실패 시 0.0
            return 0.0

    def _calculate_context_match(
            self,
            context1: Dict[str, Any],
            context2: Dict[str, Any],
            text: str
    ) -> float:
        """
        호환 래퍼: 예전 이름(_calculate_context_match)으로 호출되는 곳을
        새로운 점수 함수(_context_match_score)에 연결.
        """
        try:
            return float(self._context_match_score(context1, context2, text))
        except Exception:
            # 안전 실패 시 0.0
            return 0.0

    def blend_weights_dynamic(
            self,
            compound: Dict[str, float],
            progression: Dict[str, float],
            situation: Dict[str, float],
            text: str,
            min_floor: float = 0.05
    ) -> Dict[str, float]:
        """
        텍스트 길이/정보량에 따라 (compound/progression/situation) 가중을 동적으로 조정.
        - 짧은 텍스트(토큰 < 18): 0.45 / 0.25 / 0.30
        - 중간(18~50):           0.40 / 0.30 / 0.30
        - 긴(>50):               0.35 / 0.35 / 0.30
        이후 floor 적용 및 정규화.
        """
        tokens = re.findall(r"[가-힣]+|[a-zA-Z]+|\d+", (text or "").strip().lower())
        L = len(tokens)
        if L < 18:
            wc, wp, ws = 0.45, 0.25, 0.30
        elif L > 50:
            wc, wp, ws = 0.35, 0.35, 0.30
        else:
            wc, wp, ws = 0.40, 0.30, 0.30

        # 모든 키 통합 셋
        keys = set(compound) | set(progression) | set(situation)
        out = {}
        for k in keys:
            c = float(compound.get(k, 0.0))
            p = float(progression.get(k, 0.0))
            s = float(situation.get(k, 0.0))
            out[k] = wc * c + wp * p + ws * s

        # floor 적용
        for k in out:
            if out[k] < min_floor:
                out[k] = min_floor

        # 정규화
        tot = sum(out.values())
        if tot > 0:
            for k in out:
                out[k] /= tot
        return out

        # [PATCH 2/5] 라벨 백필 + ReLU→L1 정규화 유틸
    def _ensure_top4_prior(self, d: Dict[str, float], min_prior: float = 0.01) -> Dict[str, float]:
        """상위 4대정서(희/노/애/락) prior 백필. 음수/비유한 값은 0으로."""
        changed = False
        out: Dict[str, float] = {}
        for k in d.keys() | {"희", "노", "애", "락"}:
            v = float(d.get(k, 0.0))
            if not math.isfinite(v) or v < 0:
                v = 0.0
            out[k] = v
        for core in ("희", "노", "애", "락"):
            if out.get(core, 0.0) <= 0.0:
                # STRICT 모드에선 백필하지 않고 경고만 남김
                if str(getattr(self, "mode", "TOLERANT")).upper() == "STRICT":
                    try:
                        self.logger.warning(f"[EWC][STRICT] prior backfill skipped for core emotion '{core}'.")
                    except Exception:
                        pass
                else:
                    out[core] = max(min_prior, 0.0)  # 균등 prior
                    changed = True
        if changed:
            self._qa["prior_backfilled"] += 1
        return out

    def _relu_l1_normalize(self, d: Dict[str, float]) -> Dict[str, float]:
        """ReLU로 음수 제거 → L1 정규화 + NaN/Inf 방지. 합계=1.0(±1e-6) 보장."""
        out = {k: (float(v) if math.isfinite(v) else 0.0) for k, v in (d or {}).items()}
        out = {k: (v if v > 0.0 else 0.0) for k, v in out.items()}  # ReLU
        s = sum(out.values())
        if s <= 0.0:
            # 전부 0이면 균등 분배
            n = max(len(out), 1)
            out = {k: 1.0 / n for k in out} if out else {"희": 0.25, "노": 0.25, "애": 0.25, "락": 0.25}
        else:
            inv = 1.0 / s
            out = {k: v * inv for k, v in out.items()}
        self._qa["relu_l1_applied"] += 1
        return out

    def create_matches_data(self, text: str) -> Dict[str, Any]:
        """
        패턴/키워드/상황 매칭 → evidence 수집 - 캐시 최적화.
        (임계치: _dynamic_thresholds로 길이 기반 가변)
        """
        # 패턴 인덱스 지연 로딩
        self._ensure_pattern_index()
        
        matches = {
            "matched_phrases": [],
            "text": text,
            "contextual": {},
            "emotion_intensities": {}
        }

        processed_text = text.lower()
        tokens = self._tokenize_text(processed_text)
        token_set = set(tokens)

        dyn = self._dynamic_thresholds(len(processed_text))
        thr = dyn["pattern_similarity"]

        # 후보 Top-K 프리필터: 문장에서 등장 가능한 하위감정 후보만 선별
        self._ensure_phrase_index()
        cand_eids: Set[str] = set()
        
        # Aho-Corasick 자동자 사용 (우선)
        if getattr(self, "_aho_automaton", None):
            try:
                for _, eids in self._aho_automaton.iter(processed_text):
                    cand_eids |= set(eids)
            except Exception:
                pass
        
        # 정규식 샤드 폴백
        if not cand_eids and getattr(self, "_rx_phrase_shards", None):
            for rx in self._rx_phrase_shards:
                try:
                    for m in rx.finditer(processed_text):
                        key = (m.group(0) or "").lower()
                        if key and key in self._phrase2eids:
                            cand_eids |= set(self._phrase2eids.get(key, set()))
                except Exception:
                    continue

        # 라벨링 데이터 순회: EMOTIONS.json의 세부 감정 패턴 매칭
        for primary_emotion, emotion_info in self.emotions_data.items():
            sub_emotions = emotion_info.get('sub_emotions', {}) or {}
            matches["emotion_intensities"][primary_emotion] = 0.0

            iter_items = list(sub_emotions.items())
            if cand_eids:
                try:
                    iter_items = [
                        (sn, sd) for sn, sd in sub_emotions.items()
                        if (sd.get('metadata', {}) or {}).get('emotion_id', '') in cand_eids
                    ]
                except Exception:
                    iter_items = list(sub_emotions.items())
            
            for sub_emotion, sub_info in iter_items:
                e_id = sub_info.get('metadata', {}).get('emotion_id', '')
                emotion_profile = sub_info.get('emotion_profile', {}) or {}
                intensity_levels = emotion_profile.get('intensity_levels', {}) or {}

                # intensity_examples 패턴 매칭
                example_patterns = []
                for level in ('low', 'medium', 'high'):
                    exs = (intensity_levels.get('intensity_examples', {}) or {}).get(level, []) or []
                    example_patterns.extend([(ex, level) for ex in exs])

                for pat, intensity_level in example_patterns:
                    pat_lower = str(pat).lower()
                    if pat_lower in processed_text:
                        sim = 0.9
                    else:
                        # FAST 모드: 정확 매칭이 없으면 유사도 계산 스킵
                        if self.fast_mode:
                            continue
                        sim = self._calculate_text_similarity_fast(processed_text, pat_lower)
                    
                    if sim >= thr:
                        intensity_factor = {'low': 0.3, 'medium': 0.6, 'high': 1.0}.get(intensity_level, 0.5)
                        w = float(intensity_factor * sim)
                        matches["matched_phrases"].append({
                            "emotion_id": e_id,
                            "pattern": pat,
                            "weight": w,
                            "intensity_level": intensity_level
                        })
                        matches["emotion_intensities"][primary_emotion] = max(
                            matches["emotion_intensities"][primary_emotion],
                            w
                        )

                # linguistic_patterns.key_phrases 매칭
                ling = sub_info.get('linguistic_patterns', {}) or {}
                for kp in ling.get('key_phrases', []) or []:
                    phrase = str(kp.get('pattern', '')).lower()
                    if not phrase:
                        continue
                    
                    if phrase in processed_text:
                        sim = 0.9
                    else:
                        # FAST 모드: 정확 매칭이 없으면 유사도 계산 스킵
                        if self.fast_mode:
                            continue
                        sim = self._calculate_text_similarity_fast(processed_text, phrase)
                    
                    if sim >= thr:
                        context_req = str(kp.get('context_requirement', '')).lower()
                        if context_req and not any(cr in processed_text for cr in context_req.split()):
                            continue
                        matches["matched_phrases"].append({
                            "emotion_id": e_id,
                            "pattern": phrase,
                            "weight": float(kp.get('weight', 0.5)) * sim,
                            "context": context_req
                        })

                # core_keywords 매칭
                core_keywords = emotion_profile.get('core_keywords', []) or []
                for kw in core_keywords:
                    kw_l = str(kw).lower()
                    if kw_l in token_set:
                        matches["matched_phrases"].append({
                            "emotion_id": e_id,
                            "pattern": kw,
                            "weight": 0.6,
                            "type": "core_keyword"
                        })
                    else:
                        # FAST 모드: 정확 매칭이 없으면 유사도 계산 스킵
                        if self.fast_mode:
                            continue
                        sim = self._calculate_text_similarity_fast(processed_text, kw_l)
                        if sim >= thr:
                            matches["matched_phrases"].append({
                                "emotion_id": e_id,
                                "pattern": kw,
                                "weight": 0.4 * sim,
                                "type": "partial_keyword"
                            })

                # context_patterns.situations 매칭
                ctx = sub_info.get('context_patterns', {}) or {}
                for sit_key, sit in (ctx.get('situations', {}) or {}).items():
                    int_level = str(sit.get('intensity', 'medium'))
                    inten_w = {'low': 0.3, 'medium': 0.6, 'high': 0.9}.get(int_level, 0.5)
                    variations = sit.get('variations', []) or []
                    examples = sit.get('examples', []) or []
                    keywords = sit.get('keywords', []) or []
                    for pat in (variations + examples):
                        pat_lower = str(pat).lower()
                        if pat_lower in processed_text:
                            sim = 0.9
                        else:
                            # FAST 모드: 정확 매칭이 없으면 유사도 계산 스킵
                            if self.fast_mode:
                                continue
                            sim = self._calculate_text_similarity_fast(processed_text, pat_lower)
                        
                        if sim >= thr:
                            matches["contextual"].setdefault(e_id, []).append({
                                "situation_key": sit_key,
                                "variation": pat,
                                "intensity": int_level,
                                "weight": inten_w * sim,
                                "keywords": keywords
                            })
        self.logger.debug(f"[create_matches_data] dyn_thr={thr} matched={len(matches['matched_phrases'])}")
        return matches

    # --------------------------------------------------------------- weight core
    def calculate_emotion_weights(self, text: str) -> Dict[str, Any]:
        """
        대표감정 가중치 계산 (희/노/애/락) - 캐시 최적화
        - create_matches_data → 요인 번들 계산 → 감정별 평균강도/빈도/문맥 보정 → 정규화
        """
        # 캐시 확인
        text_hash = hash(text.strip())
        if text_hash in self._calculation_cache:
            return self._calculation_cache[text_hash]
        
        # 빠른 모드: 단순 키워드 매칭
        if self._skip_complex_analysis:
            return self._calculate_emotion_weights_fast(text, text_hash)
        
        # 패턴 인덱스 지연 로딩
        self._ensure_pattern_index()
        
        t0 = time.time()
        base = {"희": 0.05, "노": 0.05, "애": 0.05, "락": 0.05}
        matches = self.create_matches_data(text)
        intens_sum: Dict[str, float] = defaultdict(float)
        freq: Dict[str, int] = defaultdict(int)
        ctx_scores: Dict[str, float] = defaultdict(float)

        for m in matches.get("matched_phrases", []):
            e_id = m.get("emotion_id", "")
            if not e_id:
                continue
            prim = self.data_manager.get_primary_emotion(e_id)
            if not isinstance(prim, dict):
                continue
            cat = prim.get("primary_category")
            if cat not in base:
                continue

            pattern_w = float(m.get("weight", 0.0))
            lvl = str(m.get("intensity_level", "medium")).lower()
            ptype = m.get("type", "default")

            lvl_mult = {"low": 0.7, "medium": 1.0, "high": 1.3}.get(lvl, 1.0)
            type_mult = {"core_keyword": 1.2, "partial_keyword": 0.8, "sentiment_combination": 1.1}.get(ptype, 1.0)

            emo_data = self.data_manager.find_emotion_data(e_id)
            bundle = self._calculate_factor_bundle(emo_data, text, pattern_w)
            adjusted = bundle * lvl_mult * type_mult

            intens_sum[cat] += adjusted
            freq[cat] += 1

            # contextual extras
            ctx = emo_data.get("context_patterns", {}) if emo_data else {}
            if ctx:
                ctx_scores[cat] += self._calculate_context_factor(ctx, text)

        final = dict(base)
        for cat in base:
            if freq[cat] > 0:
                avg = intens_sum[cat] / freq[cat]
                cboost = min(ctx_scores[cat], 0.3)
                fboost = min(freq[cat] / 10.0, 1.0)
                
                # 추가 강화: EMOTIONS.json의 감정 복잡도 및 메타데이터 반영
                emotion_data = self.emotions_data.get(cat, {})
                complexity = emotion_data.get('metadata', {}).get('emotion_complexity', 'basic')
                complexity_mult = {'basic': 1.0, 'subtle': 1.1, 'complex': 1.2}.get(complexity, 1.0)
                
                # 감정 프로필의 추가 가중치 반영
                emotion_profile = emotion_data.get('emotion_profile', {})
                profile_boost = 0.0
                if emotion_profile:
                    # 관련 감정 수에 따른 보너스
                    related_emotions = emotion_profile.get('related_emotions', {})
                    if related_emotions:
                        profile_boost += min(len(related_emotions) * 0.05, 0.2)
                    
                    # 강도 레벨 다양성에 따른 보너스
                    intensity_levels = emotion_profile.get('intensity_levels', {})
                    if intensity_levels:
                        level_count = len(intensity_levels)
                        profile_boost += min(level_count * 0.03, 0.15)
                
                final[cat] = max(avg * (1 + cboost) * (1 + fboost) * complexity_mult * (1 + profile_boost), 0.05)

        # normalize, min-floor 0.05, renormalize
        def _normalize_floor(d: Dict[str, float]) -> Dict[str, float]:
            s = sum(d.values())
            if s > 0:
                for k in d:
                    d[k] /= s
            for k in d:
                d[k] = max(d[k], 0.05)
            s2 = sum(d.values())
            if s2 > 0:
                for k in d:
                    d[k] /= s2
            return d

        final = _normalize_floor(final)
        # [PATCH 4/5] prior 백필 + ReLU→L1 정규화 적용
        final = self._ensure_top4_prior(final, min_prior=0.01)
        final = self._relu_l1_normalize(final)

        self._last_matches = matches
        
        result = {
            "text": text,
            "matches": matches,
            "emotion_weights": final,
            "weights": dict(final),
            "metrics": {
                "processing_time": round(time.time() - t0, 6),
                "emotion_frequencies": dict(freq),
                "context_scores": {k: round(v, 4) for k, v in ctx_scores.items()},
            },
            "qa": dict(self._qa),
        }
        
        # 캐시에 저장 (크기 제한)
        if len(self._calculation_cache) < self._cache_max_size:
            self._calculation_cache[text_hash] = result
        
        return result

    # -------------------------------------- parallel(옵션) helper & chunk compute
    def _calculate_emotion_weights_serial(self, matches: Dict[str, Any], text: str) -> Dict[str, float]:
        base = defaultdict(lambda: 0.05, {"희": 0.05, "노": 0.05, "애": 0.05, "락": 0.05})
        cnt = defaultdict(int)
        intens = defaultdict(list)

        for item in matches.get("matched_phrases", []):
            e_id = item.get("emotion_id")
            if not e_id:
                continue
            prim = self.data_manager.get_primary_emotion(e_id)
            if not isinstance(prim, dict):
                continue
            cat = prim.get("primary_category")
            if cat not in {"희", "노", "애", "락"}:
                continue

            lvl = str(item.get("intensity_level", "medium")).lower()
            lvl_mult = {"low": 0.7, "medium": 1.0, "high": 1.3}.get(lvl, 1.0)
            sub = self.data_manager.find_emotion_data(e_id)
            w = self._calculate_factor_bundle(sub, text, float(item.get("weight", 0.0))) * lvl_mult

            base[cat] += w
            cnt[cat] += 1
            intens[cat].append(w)

        # avg intensity boost
        for cat in base:
            if cnt[cat] > 0:
                avg = sum(intens[cat]) / cnt[cat]
                base[cat] = max(base[cat] * (1 + avg), 0.05)

        # contextual global boost
        ctx_boost = self._calculate_contextual_boost(matches.get("contextual", {}))
        for cat, boost in ctx_boost.items():
            base[cat] *= (1 + boost)

        # normalize + floor
        s = sum(base.values())
        if s > 0:
            for k in base:
                base[k] /= s
        # floor
        for k in base:
            if base[k] < 0.05:
                base[k] = 0.05
        # renormalize
        s2 = sum(base.values())
        if s2 > 0:
            for k in base:
                base[k] /= s2
        # [PATCH 4/5] prior 백필 + ReLU→L1 정규화 적용(직렬 경로)
        final_base = self._ensure_top4_prior(dict(base), min_prior=0.01)
        final_base = self._relu_l1_normalize(final_base)
        return dict(final_base)

    def _calculate_contextual_boost(self, contextual: Dict[str, List[Dict[str, Any]]]) -> Dict[str, float]:
        INTEN = {"low": 0.3, "medium": 0.5, "high": 0.7}
        boosts = defaultdict(float)
        for e_id, ctxs in (contextual or {}).items():
            prim = self.data_manager.get_primary_emotion(e_id)
            if not isinstance(prim, dict):
                continue
            cat = prim.get("primary_category")
            if not cat:
                continue
            score = 0.0
            for c in ctxs:
                w = float(c.get("weight", 0.0))
                inten = str(c.get("intensity", "medium")).lower()
                score += w * INTEN.get(inten, 0.5)
            boosts[cat] += min(score, 0.5)
        return dict(boosts)

    # -------------------------------------------------------------- factor bundle
    def _calculate_factor_bundle(
        self, sub_emotion_data: Optional[Dict[str, Any]], text: str, base_weight: float
    ) -> float:
        """
        JSON 주도 요인 결합:
        intensity + related + sentiment + transition + context + ml + (keywords, complexity, progression, compound, temporal)
        """
        if not sub_emotion_data:
            return max(base_weight, 0.05)

        prof = sub_emotion_data.get("emotion_profile", {}) or {}
        levels = prof.get("intensity_levels", {}) or {}
        rel = prof.get("related_emotions", {}) or {}
        core = prof.get("core_keywords", []) or []

        senti = sub_emotion_data.get("sentiment_analysis", {}) or {}
        ml = sub_emotion_data.get("ml_training_metadata", {}) or {}
        trans = sub_emotion_data.get("emotion_transitions", {}) or {}
        ctx = sub_emotion_data.get("context_patterns", {}) or {}

        w = {
            "intensity": 0.30, "related": 0.15, "sentiment": 0.20,
            "transition": 0.15, "context": 0.15, "ml": 0.05
        }

        f_int = self._calculate_intensity_factor(levels, text)
        f_rel = self._calculate_related_emotion_factor(rel, text)
        f_sent = self._calculate_sentiment_factor(senti, text)
        f_tran = self._calculate_transition_factor(trans, text)
        f_ctx = self._calculate_context_factor(ctx, text)
        f_ml = self._calculate_ml_factor(ml, text)

        keyword_boost = 1.0
        if core:
            matched = sum(1 for kw in core if self._bregex(str(kw)).search(text.lower()))
            if matched > 0:
                keyword_boost = 1.0 + matched * 0.15

        # weighted sum
        mixed = (
            f_int * w["intensity"] + f_rel * w["related"] + f_sent * w["sentiment"] +
            f_tran * w["transition"] + f_ctx * w["context"] + f_ml * w["ml"]
        )

        # complexity multiplier
        cpx = (sub_emotion_data.get("metadata", {}) or {}).get("emotion_complexity", "basic")
        cpx_mult = {"basic": 1.0, "complex": 1.3, "subtle": 1.5}.get(str(cpx), 1.0)

        # progression marker
        prog_boost = 1.0
        sits = (ctx.get("situations", {}) or {})
        hit = 0
        for s in sits.values():
            prog = s.get("emotion_progression", {}) or {}
            for st in prog.values():
                if st and self._bregex(str(st)).search(text.lower()):
                    hit += 1
        if hit > 0:
            prog_boost = 1.0 + hit * 0.2

        # compound pairs (from/to words appear together)
        comp_mult = 1.0
        for p in (trans.get("patterns", []) or []):
            fr = str(p.get("from_emotion", "")).lower()
            to = str(p.get("to_emotion", "")).lower()
            if fr and to and self._bregex(fr).search(text.lower()) and self._bregex(to).search(text.lower()):
                comp_mult += 0.25

        # temporal pattern
        tmp_mult = 1.0
        for tp in (ctx.get("temporal_patterns", []) or []):
            seq = tp.get("sequence", []) or []
            if seq and any(self._bregex(str(s)).search(text.lower()) for s in seq):
                tmp_mult += 0.15

        final_factor = mixed * keyword_boost * cpx_mult * prog_boost * comp_mult * tmp_mult
        if abs(final_factor - 1.0) < 0.1:
            final_factor += 0.15
        return min(max(base_weight * final_factor, 0.05), 1.0)

    # -------------- individual factor contributors (JSON-driven, light heuristics)
    def _calculate_intensity_factor(self, intensity_levels: Dict[str, Any], text: str) -> float:
        f = 1.0
        exs = (intensity_levels or {}).get("intensity_examples", {}) or {}
        for lvl, arr in exs.items():
            for ex in arr or []:
                sim = self._calculate_text_similarity_fast(text.lower(), str(ex).lower())
                if sim >= 0.2:
                    f += {"low": 0.1, "medium": 0.2, "high": 0.3}.get(lvl, 0.15) * sim
        return max(f, 0.1)

    def _calculate_related_emotion_factor(self, related: Dict[str, List[str]], text: str) -> float:
        f = 1.0
        lowered = text.lower()
        for p in (related.get("positive", []) or []):
            if self._bregex(str(p)).search(lowered):
                f += 0.05
        for n in (related.get("negative", []) or []):
            if self._bregex(str(n)).search(lowered):
                f -= 0.05
        for u in (related.get("neutral", []) or []):
            if self._bregex(str(u)).search(lowered):
                f += 0.01
        return max(f, 0.1)

    def _calculate_sentiment_factor(self, senti: Dict[str, Any], text: str) -> float:
        f = 1.0
        lowered = text.lower()
        for w in (senti.get("positive_indicators", []) or []):
            if self._bregex(str(w)).search(lowered):
                f += 0.05
        for w in (senti.get("negative_indicators", []) or []):
            if self._bregex(str(w)).search(lowered):
                f -= 0.05
        mods = senti.get("intensity_modifiers", {}) or {}
        for w in (mods.get("amplifiers", []) or []):
            if self._bregex(str(w)).search(lowered):
                f += 0.02
        for w in (mods.get("diminishers", []) or []):
            if self._bregex(str(w)).search(lowered):
                f -= 0.02
        return max(f, 0.1)

    def _calculate_transition_factor(self, trans: Dict[str, Any], text: str) -> float:
        f = 1.0
        lowered = text.lower()
        for p in (trans.get("patterns", []) or []):
            for trg in (p.get("triggers", []) or []):
                if self._bregex(str(trg)).search(lowered):
                    f += 0.03
        for m in (trans.get("multi_emotion_transitions", []) or []):
            for trg in (m.get("triggers", []) or []):
                if self._bregex(str(trg)).search(lowered):
                    f += 0.03
        return max(f, 0.1)

    def _calculate_context_factor(self, ctx: Dict[str, Any], text: str) -> float:
        f = 1.0
        lowered = text.lower()
        for sit in (ctx.get("situations", {}) or {}).values():
            desc = str(sit.get("description", "")).lower()
            if desc and self._bregex(desc).search(lowered):
                f += 0.03
            for v in (sit.get("variations", []) or []):
                if self._bregex(str(v)).search(lowered):
                    f += 0.03
            for kw in (sit.get("keywords", []) or []):
                if self._bregex(str(kw)).search(lowered):
                    f += 0.02
        return max(f, 0.1)

    def _calculate_ml_factor(self, ml: Dict[str, Any], text: str) -> float:
        f = 1.0
        if not ml:
            return f
        lowered = text.lower()
        req = ml.get("context_requirements", {}) or {}
        min_len = int(req.get("minimum_length", 0))
        max_len = int(req.get("maximum_length", 10**9))
        if len(lowered) < min_len:
            f -= 0.1
        if len(lowered) > max_len:
            f -= 0.1
        rk = req.get("required_keywords", {}) or {}
        if isinstance(rk, dict) and rk:
            hits = 0
            required = 0
            for k, need in rk.items():
                if not isinstance(need, int) or need <= 0:
                    continue
                required += 1
                if self._bregex(str(k).lower()).search(lowered):
                    hits += 1
            if hits < required:
                f -= 0.02 * (required - hits)
        if (ml.get("analysis_modules", {}) or {}).get("context_extractor", {}).get("enabled", False):
            if "상황" in lowered:
                f += 0.05
        return max(f, 0.1)

    # ---------------------------------------------------------- progressions
    def calculate_progression(self, text: str) -> Dict[str, float]:
        """
        EMOTIONS.json의 situations.emotion_progression를 이용해 스테이지 매칭 기반 점수 산출.
        - 텍스트 길이 기반 동적 임계치(_dynamic_thresholds) 적용
        - 스테이지별 최고 점수(Trigger/Development/Peak/Aftermath) + 시퀀스 보너스
        - 시간적 일관성/모멘텀 보정
        """
        from collections import defaultdict

        # 라벨링 순회: EMOTIONS.json의 감정 진행 패턴 매칭
        processed = (text or "").lower()
        tokens = self._tokenize_text(processed)
        token_set = set(tokens)
        
        scores = {"희": 0.0, "노": 0.0, "애": 0.0, "락": 0.0}
        
        if self.emotions_data:
            # EMOTIONS.json의 모든 감정 데이터 순회
            for primary_emotion, emotion_info in self.emotions_data.items():
                if primary_emotion not in scores:
                    continue
                    
                sub_emotions = emotion_info.get('sub_emotions', {}) or {}
                
                for sub_emotion, sub_info in sub_emotions.items():
                    context_patterns = sub_info.get('context_patterns', {}) or {}
                    situations = context_patterns.get('situations', {}) or {}
                    
                    for situation_key, situation_data in situations.items():
                        emotion_progression = situation_data.get('emotion_progression', {}) or {}
                        
                        # 진행 단계별 매칭
                        stage_scores = {}
                        for stage_name, stage_data in emotion_progression.items():
                            if isinstance(stage_data, dict):
                                # 단계별 키워드 매칭
                                keywords = stage_data.get('keywords', []) or []
                                examples = stage_data.get('examples', []) or []
                                description = stage_data.get('description', '') or ''
                                
                                stage_score = 0.0
                                
                                # 키워드 매칭
                                for keyword in keywords:
                                    if isinstance(keyword, str) and keyword.lower() in processed:
                                        stage_score += 0.4
                                
                                # 예시 매칭
                                for example in examples:
                                    if isinstance(example, str) and example.lower() in processed:
                                        stage_score += 0.3
                                
                                # 설명 매칭
                                if description and description.lower() in processed:
                                    stage_score += 0.2
                                
                                stage_scores[stage_name] = min(stage_score, 1.0)
                            elif isinstance(stage_data, str):
                                # 문자열 형태의 단계 데이터
                                if stage_data.lower() in processed:
                                    stage_scores[stage_name] = 0.5
                        
                        # 최고 점수 단계를 감정 점수에 반영
                        if stage_scores:
                            max_stage_score = max(stage_scores.values())
                            intensity_level = situation_data.get('intensity', 'medium')
                            intensity_weight = {'low': 0.3, 'medium': 0.6, 'high': 0.9}.get(intensity_level, 0.5)
                            
                            scores[primary_emotion] += max_stage_score * intensity_weight
        else:
            # 폴백: 기본 감정별 간단한 매칭
            emotion_keywords = {
                '희': ['기쁨', '행복', '즐거', '신나', '환희', '축하', '기쁘'],
                '노': ['화', '분노', '짜증', '불만', '화나', '격분'],
                '애': ['슬픔', '우울', '슬프', '눈물', '우울'],
                '락': ['안정', '편안', '만족', '안정감', '평온']
            }
            
            for emotion, keywords in emotion_keywords.items():
                score = 0.0
                for keyword in keywords:
                    if keyword in processed:
                        score += 0.3
                scores[emotion] = min(score, 1.0)
        
        # normalize + floor(0.05)
        s = sum(scores.values())
        if s > 0:
            for k in scores:
                scores[k] /= s
        for k in list(scores.keys()):
            scores[k] = max(scores[k], 0.05)
        
        return dict(scores)

    def _stage_match_score(
            self,
            stage_text: str,
            tokens: List[str],
            token_set: Set[str],
            dyn_stage: Optional[float] = None
    ) -> float:
        """
        스테이지 문구(stage_text)와 입력 토큰열(tokens) 간 유사도 점수.
        - exact(토큰 부분집합) 일치: 1.0
        - 부분 일치: 슬라이딩 윈도우 최대 유사도(자카드+경계매치 혼합)
        - 임계치: dyn_stage가 우선, 없으면 self.thresholds['stage_text_similarity'] 사용
          * 임계 미달이면 0.8 감쇠를 적용해 약한 신호도 반영
        """
        if not stage_text:
            return 0.0

        # 임계치 결정
        thr = float((getattr(self, "thresholds", {}) or {}).get("stage_text_similarity", 0.20))
        if isinstance(dyn_stage, (int, float)):
            thr = float(dyn_stage)

        stoks = self._tokenize_text(stage_text.lower())
        if not stoks:
            return 0.0

        # 정확 포함(스테이지 토큰이 전부 포함되면 1.0)
        if set(stoks).issubset(token_set):
            return 1.0

        # 부분 일치: 슬라이딩 윈도우 최대 유사도
        best = 0.0
        L = len(stoks)
        if len(tokens) >= L:
            for i in range(len(tokens) - L + 1):
                win = tokens[i:i + L]
                sim = self._calculate_text_similarity_fast(" ".join(win), " ".join(stoks))
                if sim > best:
                    best = sim
        else:
            # 텍스트가 더 짧으면 전체 토큰으로만 비교
            best = self._calculate_text_similarity_fast(" ".join(tokens), " ".join(stoks))

        # 임계 이상이면 그대로, 미만이면 완만 감쇠
        score = best if best >= thr else best * 0.8
        return max(0.0, min(1.0, score))

    def _sequence_score(self, stage_scores: Dict[str, float], seq: Dict[str, Dict[str, float]], inten: float) -> float:
        acc = 0.0
        for cur, nxts in seq.items():
            if cur not in stage_scores:
                continue
            for nxt, w in nxts.items():
                if nxt in stage_scores:
                    acc += min(stage_scores[cur], stage_scores[nxt]) * w * inten
        return min(acc, 1.0)

    def _temporal_coherence(self, stage_best: Dict[str, float]) -> float:
        order = ["trigger", "development", "peak", "aftermath"]
        score = 1.0
        for i in range(len(order) - 1):
            s1, s2 = stage_best.get(order[i], 0.0), stage_best.get(order[i + 1], 0.0)
            if s1 and s2:
                if s1 > s2 * 1.5 or s2 > s1 * 1.5:
                    score *= 0.8
        return max(score, 0.6)

    def _emotional_momentum(self, stage_best: Dict[str, float]) -> float:
        order = ["trigger", "development", "peak", "aftermath"]
        arr = [stage_best.get(s, 0.0) for s in order]
        windows = []
        for i in range(len(arr) - 2):
            a, b, c = arr[i], arr[i + 1], arr[i + 2]
            if a and b and c:
                prog = ((b - a) + (c - b)) / 2.0
                windows.append(1.0 + prog)
        if not windows:
            return 1.0
        mom = sum(windows) / len(windows)
        return max(0.7, min(1.5, mom))

    def _embedding_index_for(self, key: str) -> int:
        """
        감정 키(대표/세부/텍스트)를 임베딩 인덱스로 안정 매핑.
        - 가능하면 hashlib.md5로 안정 해시, 불가 시 내장 hash()로 fallback
        - bucket size는 self._embedding_bucket_size
        """
        k = (key or "").strip()
        if not k:
            return 0
        # hashlib 사용 시 안정적(세션 간 동일), 실패 시 내장 hash()로 대체
        try:
            import hashlib  # 파일 상단에 있어도 되고, 여기서 동작해도 됨
            h = int(hashlib.md5(k.encode("utf-8")).hexdigest(), 16)
        except Exception:
            h = abs(hash(k))
        return h % max(1, getattr(self, "_embedding_bucket_size", 8192))

    # -------------------------------------------------------------- situations
    def calculate_situation(self, text: str) -> Dict[str, float]:
        """
        상황 패턴을 대표감정별로 점수화(직접/컨텍스트/키워드/예시/전개/시간·문맥 요건).
        """
        # 라벨링 순회: EMOTIONS.json의 상황 패턴 매칭
        processed = text.lower()
        tokens = self._tokenize_text(processed)
        token_set = set(tokens)
        
        scores = {"희": 0.0, "노": 0.0, "애": 0.0, "락": 0.0}
        
        if self.emotions_data:
            # EMOTIONS.json의 모든 감정 데이터 순회
            for primary_emotion, emotion_info in self.emotions_data.items():
                if primary_emotion not in scores:
                    continue
                    
                sub_emotions = emotion_info.get('sub_emotions', {}) or {}
                
                for sub_emotion, sub_info in sub_emotions.items():
                    context_patterns = sub_info.get('context_patterns', {}) or {}
                    situations = context_patterns.get('situations', {}) or {}
                    
                    for situation_key, situation_data in situations.items():
                        # 상황별 키워드 매칭
                        keywords = situation_data.get('keywords', []) or []
                        examples = situation_data.get('examples', []) or []
                        variations = situation_data.get('variations', []) or []
                        description = situation_data.get('description', '') or ''
                        
                        # 키워드 매칭
                        keyword_matches = 0
                        for keyword in keywords:
                            if isinstance(keyword, str) and keyword.lower() in processed:
                                keyword_matches += 1
                        
                        # 예시 매칭
                        example_matches = 0
                        for example in examples:
                            if isinstance(example, str) and example.lower() in processed:
                                example_matches += 1
                        
                        # 변형 매칭
                        variation_matches = 0
                        for variation in variations:
                            if isinstance(variation, str) and variation.lower() in processed:
                                variation_matches += 1
                        
                        # 설명 매칭
                        description_match = 0
                        if description and description.lower() in processed:
                            description_match = 1
                        
                        # 매칭 점수 계산
                        if keyword_matches > 0 or example_matches > 0 or variation_matches > 0 or description_match > 0:
                            intensity_level = situation_data.get('intensity', 'medium')
                            intensity_weight = {'low': 0.3, 'medium': 0.6, 'high': 0.9}.get(intensity_level, 0.5)
                            
                            # 가중치 계산
                            total_matches = keyword_matches + example_matches + variation_matches + description_match
                            match_score = (keyword_matches * 0.4 + example_matches * 0.3 + 
                                         variation_matches * 0.2 + description_match * 0.1) * intensity_weight
                            
                            scores[primary_emotion] += min(match_score, 1.0)
        else:
            # 폴백: 기본 감정별 간단한 상황 매칭
            situation_keywords = {
                '희': ['결혼', '축하', '기쁨', '행복', '즐거', '신나', '환희', '축하'],
                '노': ['화', '분노', '짜증', '불만', '화나', '격분', '싸움', '갈등'],
                '애': ['슬픔', '우울', '슬프', '눈물', '우울', '이별', '상실'],
                '락': ['안정', '편안', '만족', '안정감', '평온', '휴식', '평화']
            }
            
            for emotion, keywords in situation_keywords.items():
                score = 0.0
                for keyword in keywords:
                    if keyword in processed:
                        score += 0.25
                scores[emotion] = min(score, 1.0)
        
        # normalize + floor(0.05)
        s = sum(scores.values())
        if s > 0:
            for k in scores:
                scores[k] /= s
        for k in list(scores.keys()):
            scores[k] = max(scores[k], 0.05)
        return dict(scores)

    def _progression_match_score(self, prog: Dict[str, str], text: str) -> float:
        stage_w = {"initial": 0.2, "development": 0.3, "peak": 0.3, "resolution": 0.2}
        # stage 동의어 매핑(일관성 향상): trigger→initial, resolution→aftermath(가중 매핑 시 키 보정)
        alias = {"trigger": "initial", "resolution": "aftermath"}
        tot, cnt = 0.0, 0
        for stage, st in (prog or {}).items():
            if not st:
                continue
            sim = self._calculate_text_similarity_fast(text, str(st).lower())
            if sim > self.thresholds["stage_text_similarity"]:
                w_key = alias.get(stage, stage)
                tot += sim * stage_w.get(w_key, 0.25)
                cnt += 1
        if cnt > 1:
            tot *= (1 + 0.1 * cnt)
        return min(tot, 1.0)

    def _context_requirements_score(self, req: Dict[str, Any], text: str) -> float:
        score = 0.0
        lowered = (text or "").lower()
        token_set = set(self._tokenize_text(lowered))
        reqw = [str(w).lower() for w in (req.get("required_words", []) or [])]
        if reqw:
            m = 0
            for w in reqw:
                if w in token_set or self._bregex(w).search(lowered):
                    m += 1
            score += 0.1 * (m / len(reqw))
        exw = [str(w).lower() for w in (req.get("excluded_words", []) or [])]
        if exw:
            m = 0
            for w in exw:
                if w in token_set or self._bregex(w).search(lowered):
                    m += 1
            score -= 0.1 * (m / len(exw))
        return max(score, -0.5)

    def _temporal_patterns_score(self, patterns: List[Dict[str, Any]], text: str) -> float:
        score = 0.0
        for p in (patterns or []):
            seq = [str(s).lower() for s in (p.get("sequence", []) or [])]
            if not seq:
                continue
            hits = sum(1 for s in seq if self._bregex(s).search(text))
            if hits:
                score += 0.1 * (hits / len(seq))
        return min(score, 0.5)

    def _calculate_keyword_boost(self, situation_data: Dict[str, Any], token_set: Set[str]) -> float:
        kws = [str(k).lower() for k in (situation_data.get("keywords", []) or [])]
        if not kws:
            return 0.0
        hits = sum(1 for k in kws if k in token_set)
        return min(hits * 0.15, 0.45)

    def _calculate_context_boost(self, situation_data: Dict[str, Any], tokens: List[str]) -> float:
        base = 0.0
        desc = str(situation_data.get("description", "")).lower()
        if desc:
            base += self._calculate_text_similarity_fast(" ".join(tokens), desc) * 0.3
        var = [str(v).lower() for v in (situation_data.get("variations", []) or [])]
        if var:
            base += (max([self._calculate_text_similarity_fast(" ".join(tokens), v) for v in var] + [0.0])) * 0.2
        return min(base, 0.5)

    # -------------------------------------------------------- compound & matrix
    def calculate_compound_emotions(self, base_weights: Dict[str, float]) -> Dict[str, float]:
        """
        하위감정 relation/transition에 기반하여 감정쌍 상호 보정(경량).
        """
        adjusted = dict(base_weights)
        ems = list(adjusted.keys())

        def _resolve_primary(name: str) -> str:
            if name in {"희", "노", "애", "락"}:
                return name
            data = self.data_manager.get_primary_emotion(name)
            return data.get("primary_category", name) if isinstance(data, dict) else name

        for i in range(len(ems)):
            for j in range(i + 1, len(ems)):
                a, b = ems[i], ems[j]
                w1, w2 = adjusted[a], adjusted[b]
                if w1 <= 0 or w2 <= 0:
                    continue
                ra, rb = _resolve_primary(a), _resolve_primary(b)
                cf = self._compute_compound_factor(ra, rb, w1, w2)
                adjusted[a] += cf * w2
                adjusted[b] += cf * w1

        s = sum(adjusted.values())
        if s > 0:
            for k in adjusted:
                adjusted[k] /= s
        return adjusted

    def calculate_emotion_matrix(self, text: str, base_weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        감정 상호작용 행렬 + 고유값/벡터 + 상관 + 지표
        - base_weights 제공 시 재계산 없이 이를 사용(안전 정규화 적용)
        """
        if isinstance(base_weights, dict) and base_weights:
            weights = self._ensure_top4_prior(dict(base_weights), min_prior=0.01)
            weights = self._relu_l1_normalize(dict(weights))
        else:
            weights = self.calculate_emotion_weights(text)["emotion_weights"]
        prog = self.calculate_progression(text)
        sit = self.calculate_situation(text)

        # 대표 카테고리 동적 추출(4×N 확장 안전)
        cats: List[str] = []
        try:
            # 데이터 매니저에서 추출(권장)
            tops = list(getattr(self.data_manager, "_top_ids", []) or [])
            cats = sorted(set(tops)) if tops else []
        except Exception:
            cats = []
        if not cats:
            # fallback: 현재 분포 키들의 합집합
            cats = sorted(set(list(weights.keys()) + list(prog.keys()) + list(sit.keys())))
        if not cats:
            cats = ["희", "노", "애", "락"]

        inter = defaultdict(lambda: defaultdict(float))
        # 대각 원소(자기 상호작용)는 평균값으로 채움
        for e1 in cats:
            d = (weights.get(e1, 0.0) + prog.get(e1, 0.0) + sit.get(e1, 0.0)) / 3.0
            inter[e1][e1] = max(d, 0.05)

        # 오프대각 쌍 병렬 계산 (CUDA 가속 지원)
        pairs = [(e1, e2) for e1 in cats for e2 in cats if e1 != e2]

        def _compute_pair(args: Tuple[str, str]) -> Tuple[str, str, float]:
            e1, e2 = args
            d1 = self.data_manager.find_emotion_data(e1)
            d2 = self.data_manager.find_emotion_data(e2)
            if not d1 or not d2:
                return e1, e2, 0.05
            val = self._calculate_emotion_interaction(d1, d2, text, weights.get(e1, 0.0), weights.get(e2, 0.0))
            return e1, e2, max(val, 0.05)

        # CUDA 가속 병렬 계산 시도
        if self.use_cuda and _TORCH_OK and len(pairs) > 4 and self._compute_interactions_cuda:
            try:
                # CUDA에서 배치 처리
                interaction_values = self._compute_interactions_cuda(pairs, weights, text)
                for i, (e1, e2) in enumerate(pairs):
                    inter[e1][e2] = interaction_values[i]
            except Exception as e:
                self.logger.warning(f"[calculate_emotion_matrix] CUDA 병렬 계산 실패, CPU로 폴백: {e}")
                # CPU 병렬 처리로 폴백
                if self.use_parallel and getattr(self, "executor", None) is not None:
                    try:
                        for e1, e2, v in self.executor.map(_compute_pair, pairs):
                            inter[e1][e2] = v
                    except Exception:
                        # 실패 시 직렬로 폴백
                        for e1, e2 in pairs:
                            _, _, v = _compute_pair((e1, e2))
                            inter[e1][e2] = v
                else:
                    for e1, e2 in pairs:
                        _, _, v = _compute_pair((e1, e2))
                        inter[e1][e2] = v
        else:
            # 기존 CPU 병렬 처리
            if self.use_parallel and getattr(self, "executor", None) is not None:
                try:
                    for e1, e2, v in self.executor.map(_compute_pair, pairs):
                        inter[e1][e2] = v
                except Exception:
                    # 실패 시 직렬로 폴백
                    for e1, e2 in pairs:
                        _, _, v = _compute_pair((e1, e2))
                        inter[e1][e2] = v
            else:
                for e1, e2 in pairs:
                    _, _, v = _compute_pair((e1, e2))
                    inter[e1][e2] = v

        norm = self._normalize_interaction_matrix(inter)
        eigvals, eigvecs = self._calculate_matrix_properties(norm)
        corrs = self._calculate_emotion_correlations(norm)
        patterns = self._identify_dominant_patterns(norm, eigvals, eigvecs)

        return {
            "weights": dict(weights),
            "interaction_matrix": {k: dict(v) for k, v in norm.items()},
            "matrix": {k: dict(v) for k, v in norm.items()},  # 출력 일관성 보조(호환 위해 중복 제공)
            "eigenvalues": eigvals,
            "eigenvectors": eigvecs,
            "correlations": corrs,
            "dominant_patterns": patterns,
            "metrics": {
                "matrix_stability": self._calculate_matrix_stability(eigvals),
                "emotion_balance": self._calculate_emotion_balance(norm),
                "interaction_strength": self._calculate_interaction_strength(norm),
            },
            "qa": dict(self._qa),
        }

    def _calculate_emotion_interaction(self,
                                       emotion1_data: Dict[str, Any],
                                       emotion2_data: Dict[str, Any],
                                       text: str,
                                       weight1: float,
                                       weight2: float) -> float:
        """
        시너지(+)/충돌(−)/전이(+)/강도·문맥(+)/가중치 모멘텀(+)
        → 0~1 클립
        """
        t_low = (text or "").lower()

        # ---- 시너지/충돌(related_emotions + ml.conflict_with) ----
        def _rel_sets(d):
            rp = (d.get("emotion_profile", {}) or {}).get("related_emotions", {}) or {}
            pos = set(x.lower() for x in (rp.get("positive", []) or []))
            neg = set(x.lower() for x in (rp.get("negative", []) or []))
            return pos, neg

        p1, n1 = _rel_sets(emotion1_data)
        p2, n2 = _rel_sets(emotion2_data)

        # 겹침(같은 긍정/같은 부정) + 상반 교차(positive vs negative)
        pos_ol = len(p1 & p2)
        neg_ol = len(n1 & n2)
        cross_conf = len(p1 & n2) + len(p2 & n1)

        # ml.conflict_with
        def _conf_set(d):
            return set(x.lower() for x in ((d.get("ml_training_metadata", {}) or {}).get("conflict_with", []) or []))

        c1 = _conf_set(emotion1_data);
        c2 = _conf_set(emotion2_data)
        explicit_conf = int(bool(c1 & {x.lower() for x in p2 | n2} or c2 & {x.lower() for x in p1 | n1}))

        # ---- 전이 prior(간단 추정: triggers가 텍스트에 등장하면 +) ----
        def _trig_bonus(d_from, d_to):
            pats = (d_from.get("emotion_transitions", {}) or {}).get("patterns", []) or []
            bonus = 0.0
            for p in pats:
                if (p.get("to_emotion") or "").strip() == (d_to.get("metadata", {}) or {}).get("primary_category", ""):
                    for trg in (p.get("triggers", []) or []):
                        if isinstance(trg, str) and trg.lower() in t_low:
                            bonus += 0.06
                    for trg in ((p.get("transition_analysis", {}) or {}).get("trigger_words", []) or []):
                        if isinstance(trg, str) and trg.lower() in t_low:
                            bonus += 0.04
            return min(bonus, 0.18)

        prior_bonus = _trig_bonus(emotion1_data, emotion2_data) + _trig_bonus(emotion2_data, emotion1_data)

        # ---- 강도/문맥 일치(기존 로직 재사용) ----
        intensity_match = self._intensity_match_score(
            (emotion1_data.get("emotion_profile", {}) or {}).get("intensity_levels", {}) or {},
            (emotion2_data.get("emotion_profile", {}) or {}).get("intensity_levels", {}) or {},
            text,
        )
        context_match   = self._context_match_score(
            emotion1_data.get("context_patterns", {}) or {},
            emotion2_data.get("context_patterns", {}) or {},
            text,
        )

        # ---- 가중치 모멘텀(두 감정이 모두 큰 비중이면 더 상호작용↑) ----
        momentum = min(weight1, weight2) * 0.6

        # ---- 종합 스코어 ----
        synergy = 0.0
        synergy += 0.12 * pos_ol
        synergy += 0.06 * neg_ol  # 같은 부정끼리의 공진도 약간 반영
        synergy += prior_bonus
        synergy += 0.30 * intensity_match
        synergy += 0.34 * context_match
        synergy += momentum

        conflict = 0.0
        conflict += 0.10 * cross_conf
        conflict += 0.18 * explicit_conf

        score = synergy - conflict
        return max(0.0, min(1.0, score))

    def _calculate_intensity_distribution(self, intensity_examples: Dict[str, Any]) -> float:
        try:
            # _intensity_dist_score는 emotion_profile 형태를 기대하므로 최소 구조로 감싼다
            prof = {"intensity_levels": {"intensity_examples": intensity_examples or {}}}
            return float(self._intensity_dist_score(prof))
        except Exception:
            return 0.0

    def _calculate_context_relevance(self, context_patterns: Dict[str, Any]) -> float:
        try:
            return float(self._context_rel_score(context_patterns or {}))
        except Exception:
            return 0.0


    def _check_related_emotions(self, rel1: Dict[str, List[str]], rel2: Dict[str, List[str]]) -> bool:
        if not rel1 or not rel2:
            return False
        def _norm_set(lst): return set(str(x).lower() for x in (lst or []))
        p1, n1, u1 = _norm_set(rel1.get("positive")), _norm_set(rel1.get("negative")), _norm_set(rel1.get("neutral"))
        p2, n2, u2 = _norm_set(rel2.get("positive")), _norm_set(rel2.get("negative")), _norm_set(rel2.get("neutral"))
        pos = len(p1 & p2); neg = len(n1 & n2); neu = len(u1 & u2)
        conflict = len(p1 & n2) + len(p2 & n1)
        total = len(p1) + len(n1) + len(u1)
        if total == 0:
            return False
        sim = (pos + neg + 0.5 * neu - 0.7 * conflict) / total
        return sim >= 0.3

    def _check_emotion_transitions(self, T1: List[Dict[str, Any]], T2: List[Dict[str, Any]], text: str) -> bool:
        if not T1 or not T2:
            return False
        lowered = text.lower()
        def _hit(pt: Dict[str, Any]) -> bool:
            for tr in (pt.get("triggers", []) or []):
                if self._bregex(str(tr)).search(lowered):
                    return True
            return False
        match, total = 0.0, 0.0
        for a in T1:
            for b in T2:
                total += 1.0
                if _hit(a) and _hit(b):
                    match += 1.0
                if (a.get("from_emotion") == b.get("to_emotion")) or (a.get("to_emotion") == b.get("from_emotion")):
                    match += 0.5
        if total == 0:
            return False
        return (match / total) >= 0.3

    def _intensity_match_score(self, I1: Dict[str, Any], I2: Dict[str, Any], text: str) -> float:
        if not I1 or not I2:
            return 0.0
        def _best(examples: Dict[str, List[str]]) -> Tuple[float, str]:
            best, lvl_best = 0.0, "medium"
            lw = {"low": 0.7, "medium": 1.0, "high": 1.3}
            exs = (examples or {}).get("intensity_examples", {}) or {}
            for lvl, arr in exs.items():
                for ex in (arr or []):
                    sim = self._calculate_text_similarity_fast(text.lower(), str(ex).lower())
                    val = sim * lw.get(lvl, 1.0)
                    if val > best:
                        best, lvl_best = val, lvl
            return best, lvl_best
        s1, l1 = _best(I1); s2, l2 = _best(I2)
        lvl_match = 1.0 if l1 == l2 else 0.5
        return min(((s1 + s2) / 2.0) * lvl_match, 1.0)

    def _context_match_score(self, C1: Dict[str, Any], C2: Dict[str, Any], text: str) -> float:
        def _score_sit(sit: Dict[str, Any]) -> float:
            sc = 0.0; lowered = text.lower()
            desc = str(sit.get("description", "")).lower()
            if desc:
                sc += self._calculate_text_similarity_fast(lowered, desc) * 0.3
            var = [str(v).lower() for v in (sit.get("variations", []) or [])]
            if var:
                sc += (max([self._calculate_text_similarity_fast(lowered, v) for v in var] + [0.0])) * 0.2
            kws = [str(k).lower() for k in (sit.get("keywords", []) or [])]
            if kws:
                hit = sum(1 for k in kws if self._bregex(k).search(lowered))
                sc += (hit / len(kws)) * 0.3
            exs = [str(e).lower() for e in (sit.get("examples", []) or [])]
            if exs:
                sc += (max([self._calculate_text_similarity_fast(lowered, e) for e in exs] + [0.0])) * 0.2
            return min(sc, 1.0)
        s1 = max([_score_sit(s) for s in (C1.get("situations", {}) or {}).values()] + [0.0])
        s2 = max([_score_sit(s) for s in (C2.get("situations", {}) or {}).values()] + [0.0])
        return min((s1 + s2) / 2.0, 1.0)

    def _normalize_interaction_matrix(self, matrix: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """
        행 단위 정규화(각 감정 기준으로 확률 분포처럼 합=1)
        - 작은 epsilon 스무딩으로 0행/동일행 문제 완화
        """
        normalized = defaultdict(lambda: defaultdict(float))
        eps = 1e-8
        # [보강] 누락 열 키 집합화(희/노/애/락 ∪ 관측된 행/열 키 전체)
        all_cols = sorted({c for row in (matrix or {}).values() for c in (row or {}).keys()} | set((matrix or {}).keys()) | {"희", "노", "애", "락"})
        for row_em, row in (matrix or {}).items():
            row_sum = sum(max(v, 0.0) for v in row.values())
            if row_sum <= 0:
                # 전부 동일 최소값으로 채움(모든 열 키 기준 균등)
                n = len(all_cols) or 1
                for col_em in all_cols:
                    normalized[row_em][col_em] = 1.0 / n
            else:
                for col_em, val in row.items():
                    normalized[row_em][col_em] = max(val, 0.0) / (row_sum + eps)
        return normalized

        # [PATCH 3/5] 고유분해 안전화: 대칭화 + Diagonal Loading + SVD 폴백
    def _safe_eig_or_svd(self, A: "np.ndarray") -> Tuple[List[float], List[List[float]]]:
        """
        1) 유한성 보정(NaN/Inf→0)
        2) 대칭화 S = (A + A.T)/2
        3) Diagonal Loading(εI) 점증하며 eigh 시도
        4) 실패 시 SVD(PCA) 폴백: U, s, Vt -> evals = s**2, evecs = U
        """
        if np is None:
            return [], []

        S = np.asarray(A, dtype=np.float64)
        # 유한성 보정
        bad_mask = ~np.isfinite(S)
        if bad_mask.any():
            S[bad_mask] = 0.0
            self._qa["finite_repair"] += 1

        # 대칭화
        S = 0.5 * (S + S.T)

        # eigh + diagonal loading 점증
        I = np.eye(S.shape[0], dtype=np.float64)
        eps = float(self._diag_eps_base)
        for retry in range(int(self._eig_max_retries)):
            try:
                evals, evecs = np.linalg.eigh(S + eps * I)
                # 유효성 체크
                if not (np.isfinite(evals).all() and np.isfinite(evecs).all()):
                    raise FloatingPointError("non-finite eigen output")
                # 내림차순 정렬(절댓값 기준)
                order = np.argsort(-np.abs(evals))
                evals = evals[order]
                evecs = evecs[:, order]
                if retry > 0:
                    self._qa["eig_retries"] += retry
                    self._qa["diag_loading"] += 1
                return [float(x) for x in evals], [list(map(float, evecs[:, k])) for k in range(evecs.shape[1])]
            except Exception:
                # 다음 재시도: eps 10배
                eps = min(eps * 10.0, float(self._diag_eps_max))

        # SVD(PCA) 폴백
        try:
            U, s, _ = np.linalg.svd(S, full_matrices=False)
            evals = (s ** 2)  # 분산 기여도 유사
            order = np.argsort(-evals)
            evals = evals[order]
            U = U[:, order]
            self._qa["fallback_svd"] += 1
            return [float(x) for x in evals], [list(map(float, U[:, k])) for k in range(U.shape[1])]
        except Exception:
            # 최후의 안전: 빈 결과
            self._qa["fallback_svd"] += 1
            return [], []

    # EmotionWeightCalculator 내 교체
    def _calculate_matrix_properties(self, matrix: Dict[str, Dict[str, float]]) -> Tuple[
        List[float], List[List[float]]]:
        """
        고유값/고유벡터 계산(정렬/안정화):
          - numpy 행렬로 변환
          - 안전 eig(eigh) 시도 후 실패 시 SVD 폴백 사용
          - float 변환/리스트화는 _safe_eig_or_svd 내부에서 처리
        """
        if np is None:
            return [], []
        emotions = sorted(matrix.keys())
        size = len(emotions)
        if size == 0:
            return [], []

        A = np.zeros((size, size), dtype=np.float64)
        for i, em1 in enumerate(emotions):
            for j, em2 in enumerate(emotions):
                A[i, j] = float(matrix[em1].get(em2, 0.0))
        evals, evecs = self._safe_eig_or_svd(A)
        return evals, evecs

    # EmotionWeightCalculator 내 교체
    def _calculate_emotion_correlations(
            self,
            matrix: Dict[str, Dict[str, float]],
            eps: float = 1e-4
    ) -> Dict[str, Dict[str, float]]:
        """
        행(감정) 벡터 간 상관(피어슨). 행이 지나치게 유사한 경우를 완화하기 위해
        ε-smoothing을 적용한 뒤 재정규화한다. (eps 기본 1e-4)
        반환: {emotion1: {emotion2: corr, ...}, ...}
        """
        emotions = sorted(matrix.keys())
        if not emotions:
            return {}

        # 1) ε-smoothing + 재정규화
        smoothed: Dict[str, List[float]] = {}
        for em in emotions:
            row = [float(matrix[em].get(e2, 0.0)) for e2 in emotions]
            row = [max(x, 0.0) + eps for x in row]
            s = sum(row)
            row = [x / s if s > 0 else 0.0 for x in row]
            smoothed[em] = row

        # 2) 평균 제거(센터링)
        centered: Dict[str, List[float]] = {}
        for em in emotions:
            r = smoothed[em]
            mu = sum(r) / len(r) if r else 0.0
            centered[em] = [x - mu for x in r]

        # 3) 분산/상관 계산
        corrs: Dict[str, Dict[str, float]] = {em: {} for em in emotions}
        for i, a in enumerate(emotions):
            va = centered[a]
            var_a = sum(x * x for x in va)
            for j, b in enumerate(emotions):
                if a == b:
                    corrs[a][b] = 1.0
                    continue
                vb = centered[b]
                var_b = sum(x * x for x in vb)
                if var_a <= 1e-12 or var_b <= 1e-12:  # 상수 벡터 등
                    corrs[a][b] = 0.0
                else:
                    num = sum(x * y for x, y in zip(va, vb))
                    corrs[a][b] = max(-1.0, min(1.0, num / (var_a ** 0.5 * var_b ** 0.5)))
        return corrs

    def _dynamic_thresholds(self, text_len: int) -> Dict[str, float]:
        """
        텍스트 길이에 따라 매칭 임계값을 가변 설정.
        - 짧음(<=60): recall↑ → 임계값 낮춤
        - 보통(61~240): 기본
        - 김(>240): precision↑ → 임계값 높임
        """
        base_pat = float(getattr(self, "thresholds", {}).get("pattern_similarity", 0.15))
        base_stage = float(getattr(self, "thresholds", {}).get("stage_text_similarity", 0.22))
        
        # 환경변수로 임계치 상향 조정 (FAST 모드)
        if self.fast_mode:
            base_pat += self.pattern_similarity_threshold
            base_stage += self.stage_similarity_threshold

        if text_len <= 60:  # 짧은 입력
            scale_pat, scale_stage = 0.8, 0.82
        elif text_len <= 240:  # 중간 길이
            scale_pat, scale_stage = 1.0, 1.0
        else:  # 긴 입력
            scale_pat, scale_stage = 1.2, 1.18

        return {
            "pattern_similarity": max(0.05, min(0.9, base_pat * scale_pat)),
            "stage_text_similarity": max(0.05, min(0.95, base_stage * scale_stage)),
        }

    def _calculate_correlation(self, v1: List[float], v2: List[float]) -> float:
        if len(v1) != len(v2) or len(v1) == 0:
            return 0.0
        m1 = sum(v1) / len(v1)
        m2 = sum(v2) / len(v2)
        num = sum((x - m1) * (y - m2) for x, y in zip(v1, v2))
        den = (sum((x - m1) ** 2 for x in v1) * sum((y - m2) ** 2 for y in v2)) ** 0.5
        return (num / den) if den > 0 else 0.0

    # EmotionWeightCalculator 내 교체
    def _identify_dominant_patterns(
            self,
            matrix: Dict[str, Dict[str, float]],
            eigenvalues: List[float],
            eigenvectors: List[List[float]]
    ) -> List[Dict[str, Any]]:
        """
        고유벡터 해석 보강:
          - (eigenvalue, eigenvector) 쌍을 eigenvalue 내림차순 정렬
          - 각 고유벡터를 절댓값 기준으로 정규화하여 weight 합=1
          - 소량값은 컷오프(예: 0.15)로 노이즈 제거
        """
        if np is None:
            return []
        emotions = sorted(matrix.keys())
        if not emotions or not eigenvalues or not eigenvectors:
            return []

        # numpy array 변환
        vals = np.asarray(eigenvalues, dtype=np.float64)
        vecs = np.asarray(eigenvectors, dtype=np.float64)

        # 길이 점검
        if vecs.shape[0] != vals.shape[0]:
            # 불일치 시 안전처리: 최소 길이 기준 슬라이스
            m = min(vecs.shape[0], vals.shape[0])
            vals = vals[:m]
            vecs = vecs[:m, :]

        # 정렬(내림차순)
        order = np.argsort(-np.abs(vals))
        vals = vals[order]
        vecs = vecs[order]

        out_patterns: List[Dict[str, Any]] = []
        cutoff = 0.15  # 소량 컷오프

        for i, (ev, vec) in enumerate(zip(vals, vecs), start=1):
            # 절댓값 정규화 (합=1)
            v = np.abs(np.asarray(vec, dtype=np.float64))
            v_sum = float(v.sum())
            if v_sum <= 1e-12:
                continue
            w = (v / v_sum)

            # 컷오프 적용 후 상위만 수집
            dom = []
            for j, emo in enumerate(emotions):
                if j < w.shape[0] and float(w[j]) >= cutoff:
                    dom.append({"emotion": emo, "weight": float(w[j])})

            if not dom:
                # 컷오프가 너무 높으면 가장 큰 한 개는 강제로 포함
                jmax = int(np.argmax(w))
                dom = [{"emotion": emotions[jmax], "weight": float(w[jmax])}]

            dom_sorted = sorted(dom, key=lambda x: x["weight"], reverse=True)
            out_patterns.append({
                "pattern_id": i,
                "eigenvalue": float(np.abs(ev)),
                "dominant_emotions": dom_sorted
            })

        # eigenvalue desc로 재정렬(보장)
        out_patterns.sort(key=lambda x: x["eigenvalue"], reverse=True)
        return out_patterns

    def _calculate_matrix_stability(self, eigvals: List[float]) -> float:
        if not eigvals:
            return 1.0
        me = max(abs(x) for x in eigvals)
        return 1.0 / (1.0 + me)

    def _calculate_emotion_balance(self, M: Dict[str, Dict[str, float]]) -> float:
        ems = sorted(M.keys())
        rs = [sum(M[e].values()) for e in ems]
        if not rs:
            return 1.0
        mean = sum(rs) / len(rs)
        var = sum((x - mean) ** 2 for x in rs) / len(rs)
        return 1.0 / (1.0 + var)

    def _calculate_interaction_strength(self, M: Dict[str, Dict[str, float]]) -> float:
        ems = sorted(M.keys())
        off = sum(M[a][b] for a in ems for b in ems if a != b)
        denom = len(ems) * (len(ems) - 1)
        return (off / denom) if denom > 0 else 0.0

    # -------------------------------------------------------- sub emotion merge
    def integrate_sub_emotion_scores(
            self,
            emotion_weights: Dict[str, float],
            text: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        대표감정별 하위감정 Top-K 분배 (K=3~7 가변).
        - text가 주어지면 길이 기준, 없으면 엔트로피로 K 결정
        - 기존 호출과 호환(옵션 파라미터)
        """
        result: Dict[str, Any] = {}
        min_weight_threshold = 0.05
        baseline_threshold = 0.03

        # 감정별 K 산정(텍스트 단위 동일 K 사용)
        top_k_global = self._choose_topk_for_text(text, emotion_weights)

        for main_emotion, total_score in (emotion_weights or {}).items():
            if total_score <= 0:
                continue

            top_id = self._find_top_id_by_primary_category(main_emotion)
            if not top_id or top_id not in self.emotions_data:
                result[main_emotion] = {"score": max(total_score, min_weight_threshold), "sub_emotions": {}}
                continue

            sub_emotions = (self.emotions_data[top_id].get('sub_emotions', {}) or {})
            if not sub_emotions:
                result[main_emotion] = {"score": max(total_score, min_weight_threshold), "sub_emotions": {}}
                continue

            # 하위감정 베이스 스코어 산출(강도/문맥 요인)
            density_map: Dict[str, Dict[str, float]] = {}
            total_base = 0.0
            for sub_name, sub_data in sub_emotions.items():
                meta = sub_data.get('metadata', {}) or {}
                prof = sub_data.get('emotion_profile', {}) or {}
                ctx = sub_data.get('context_patterns', {}) or {}

                # 강도 분포/문맥 적합
                inten = (prof.get('intensity_levels', {}) or {}).get('intensity_examples', {}) or {}
                inten_score = self._calculate_intensity_distribution(inten)  # [-?,+?] 평균화 지표
                ctx_score = self._calculate_context_relevance(ctx)  # [0,1] 정도

                # 복합/키워드 근거(있으면 가점) — 너무 과도하지 않게 0.0~0.2 정도
                kw = len(prof.get('core_keywords', []) or [])
                extra_bonus = min(0.2, 0.03 * kw)

                base_w = max(1e-6, (1.0 + inten_score) * (1.0 + ctx_score) + extra_bonus)
                total_base += base_w
                density_map[sub_name] = {
                    "base": base_w,
                    "inten": inten_score,
                    "ctx": ctx_score,
                    "bonus": extra_bonus
                }

            if total_base <= 0:
                result[main_emotion] = {"score": max(total_score, min_weight_threshold), "sub_emotions": {}}
                continue

            # Top-K 선택
            k = min(top_k_global, len(density_map))
            sorted_subs = sorted(density_map.items(), key=lambda x: x[1]["base"], reverse=True)[:k]

            # 배분
            remain = total_score
            alloc: Dict[str, float] = {}
            for sub_name, comp in sorted_subs:
                # 비율 배분 + 강도/문맥/보너스 결합
                ratio = comp["base"] / total_base
                raw = total_score * ratio
                # 완만 가중
                adj = raw * (1.0 + 0.5 * comp["ctx"]) * (1.0 + 0.3 * comp["inten"]) + comp["bonus"] * 0.1
                # 바닥선 유지
                adj = max(adj, baseline_threshold)
                if adj > remain:
                    adj = remain
                alloc[sub_name] = adj
                remain -= adj
                if remain <= 0:
                    break

            # 남은 잔여 배분(있으면)
            if alloc and remain > 0:
                split = remain / len(alloc)
                for s in alloc:
                    alloc[s] += split
                remain = 0.0

            # 비율 스케일링(총합 = total_score)
            ssum = sum(alloc.values())
            if ssum > 0:
                scale = total_score / ssum
                for s in alloc:
                    alloc[s] *= scale

            # 임계 미만 컷 + 재스케일
            alloc = {n: v for n, v in alloc.items() if v >= min_weight_threshold}
            ssum = sum(alloc.values())
            if ssum > 0:
                scale = total_score / ssum
                for s in alloc:
                    alloc[s] *= scale

            result[main_emotion] = {"score": max(total_score, min_weight_threshold), "sub_emotions": alloc}
        return result

    def _choose_topk_for_text(self, text: Optional[str], weights: Dict[str, float]) -> int:
        """
        Top-K(3~7) 동적 결정:
        - 텍스트가 있으면 토큰 수를 기준으로,
        - 없으면 가중치 엔트로피(정규화)로 추정.
        """
        try:
            if text:
                n_tok = len(self._tokenize_text(text.lower()))
                if n_tok < 15:   return 3
                if n_tok < 35:   return 4
                if n_tok < 80:   return 5
                if n_tok < 150:  return 6
                return 7
            # entropy 기반
            ps = [max(float(p), 1e-12) for p in weights.values()]
            s = sum(ps)
            ps = [p / s for p in ps] if s > 0 else [1.0 / max(1, len(ps))] * len(ps)
            import math
            H = -sum(p * math.log(p + 1e-12) for p in ps) / (math.log(len(ps) + 1e-12) if len(ps) > 1 else 1.0)
            # H∈[0,1] → 3..7 매핑
            k = 3 + int(round(4 * max(0.0, min(1.0, H))))
            return max(3, min(7, k))
        except Exception:
            return 5

    def _intensity_dist_score(self, prof: Dict[str, Any]) -> float:
        ex = (prof.get("intensity_levels", {}) or {}).get("intensity_examples", {}) or {}
        if not ex:
            return 0.0
        weights = {"high": 1.3, "medium": 1.0, "low": 0.7}
        num, den = 0.0, 0
        for lvl, arr in ex.items():
            if arr:
                num += weights.get(lvl, 1.0) * len(arr)
                den += len(arr)
        return ((num / max(1, den)) - 1.0) if den else 0.0

    def _context_rel_score(self, ctx: Dict[str, Any]) -> float:
        sits = (ctx.get("situations", {}) or {})
        if not sits:
            return 0.0
        tot, n = 0.0, 0
        for s in sits.values():
            sc = 0.0
            if s.get("description"): sc += 0.3
            if s.get("variations"): sc += min(0.2 * len(s["variations"]) / 5, 0.2)
            if s.get("keywords"): sc += min(0.2 * len(s["keywords"]) / 10, 0.2)
            if s.get("examples"): sc += min(0.3 * len(s["examples"]) / 5, 0.3)
            tot += min(sc, 1.0); n += 1
        return (tot / n) if n else 0.0

    # -------------------------------------------------------------- top id utils
    def _find_top_id_by_primary_category(self, category_name: str) -> str:
        if not self.emotions_data:
            return ""
        tid = self._search_top_id_direct(category_name)
        if tid:
            return tid
        return self._search_top_id_by_variants(category_name)

    def _search_top_id_direct(self, category_name: str) -> str:
        for tid, tdata in (self.emotions_data or {}).items():
            if (tdata.get("metadata", {}) or {}).get("primary_category") == category_name:
                return tid
        return ""

    def _search_top_id_by_variants(self, category_name: str) -> str:
        target = category_name.strip()
        for tid, tdata in (self.emotions_data or {}).items():
            if target in self._collect_category_variants(tdata):
                return tid
        return ""

    def _collect_category_variants(self, data: Dict[str, Any]) -> set:
        variants = set()
        md = (data.get("metadata", {}) or {})
        if md.get("primary_category"): variants.add(str(md["primary_category"]).strip())
        if md.get("sub_category"): variants.add(str(md["sub_category"]).strip())

        prof = (data.get("emotion_profile", {}) or {})
        rel = (prof.get("related_emotions", {}) or {})
        for key in ["positive", "negative", "neutral"]:
            for it in (rel.get(key, []) or []):
                if isinstance(it, str) and it.strip():
                    variants.add(it.strip())

        ling = (data.get("linguistic_patterns", {}) or {})
        for combo in (ling.get("sentiment_combinations", []) or []):
            for w in (combo.get("words", []) or []):
                if isinstance(w, str) and w.strip():
                    variants.add(w.strip())

        trs = (data.get("emotion_transitions", {}) or {})
        for p in (trs.get("patterns", []) or []):
            for k in ["from_emotion", "to_emotion"]:
                v = p.get(k)
                if isinstance(v, str) and v.strip():
                    variants.add(v.strip())
        for m in (trs.get("multi_emotion_transitions", []) or []):
            for v in (m.get("from_emotions", []) or []) + (m.get("to_emotions", []) or []):
                if isinstance(v, str) and v.strip():
                    variants.add(v.strip())
        return variants

    # ----------------------------------------------------------------- embedding
    def _ensure_embedding_capacity(self, need: int) -> None:
        if not _TORCH_OK or self.embedding_layer is None:
            return
        cur = self.embedding_layer.num_embeddings
        if need <= cur:
            return
        # grow
        with torch.no_grad():
            new_layer = Embedding(num_embeddings=need, embedding_dim=self.embedding_dim).to(self.device)
            new_layer.weight[:cur].copy_(self.embedding_layer.weight.data)
            self.embedding_layer = new_layer
            self.logger.info(f"[EWC] Embedding 크기 확장: {cur} → {need}")

    def generate_embedding(self, emotion_weights: Dict[str, float]) -> "Tensor | np.ndarray":
        """
        안정 해시 기반 고정-버킷 임베딩 가중합.
        - torch 사용 시: self.embedding_layer (고정 bucket)에서 가중합 후 L2 정규화
        - torch 미사용 시: numpy 난수 벡터를 md5 시드로 생성해 가중합 후 L2 정규화
        - UserWarning(grad tensor → scalar) 방지: no_grad + .item() 사용
        """
        dim = getattr(self, "embedding_dim", 64)

        # 빈 입력 처리
        if not emotion_weights:
            if _TORCH_OK and getattr(self, "embedding_layer", None) is not None:
                # grad 불필요 → no_grad
                with torch.no_grad():
                    return torch.zeros(dim, device=self.device)
            return np.zeros((dim,), dtype=float) if np is not None else [0.0] * dim

        # 안정 인덱스 매핑(고정 버킷)
        def _stable_idx(key: str, bucket: int) -> int:
            k = (key or "").strip()
            if not k:
                return 0
            try:
                import hashlib
                h = int(hashlib.md5(k.encode("utf-8")).hexdigest(), 16)
            except Exception:
                h = abs(hash(k))
            return h % max(1, bucket)

        keys = list(emotion_weights.keys())

        # ---- torch 경로 (CUDA 가속) ----
        if _TORCH_OK and getattr(self, "embedding_layer", None) is not None:
            try:
                # CUDA 디바이스 설정
                target_device = self.device if self.use_cuda else torch.device("cpu")
                
                bucket = int(getattr(self, "_embedding_bucket_size",
                                     getattr(self.embedding_layer, "num_embeddings", 8192)))
                idxs = [_stable_idx(k, bucket) for k in keys]

                # 완전 추론 모드
                self.embedding_layer.eval()
                with torch.no_grad():
                    idx_tensor = torch.tensor(idxs, dtype=torch.long, device=target_device)  # [K]
                    em: "Tensor" = self.embedding_layer(idx_tensor)  # [K, D]
                    w: "Tensor" = torch.tensor(
                        [float(emotion_weights[k]) for k in keys],
                        dtype=torch.float32, device=target_device
                    ).unsqueeze(1)  # [K, 1]
                    out = (em * w).sum(dim=0)  # [D]

                    # L2 정규화 (item()으로 스칼라 추출 → 경고 방지)
                    norm = out.norm()
                    nval = float(norm.item()) if hasattr(norm, "item") else float(norm)
                    if nval > 0.0:
                        out = out / nval
                    
                    # CUDA 사용 시 메모리 정리
                    if self.use_cuda:
                        torch.cuda.empty_cache()
                    
                    return out  # grad 붙지 않은 텐서
            except Exception as e:
                self.logger.warning(f"[generate_embedding] CUDA 실패, CPU로 폴백: {e}")
                # CPU로 폴백
                with torch.no_grad():
                    idx_tensor = torch.tensor(idxs, dtype=torch.long, device=torch.device("cpu"))
                    em: "Tensor" = self.embedding_layer(idx_tensor)
                    w: "Tensor" = torch.tensor(
                        [float(emotion_weights[k]) for k in keys],
                        dtype=torch.float32, device=torch.device("cpu")
                    ).unsqueeze(1)
                    out = (em * w).sum(dim=0)
                    
                    norm = out.norm()
                    nval = float(norm.item()) if hasattr(norm, "item") else float(norm)
                    if nval > 0.0:
                        out = out / nval
                    return out

        # ---- numpy 경로 ----
        if np is not None:
            vec = np.zeros((dim,), dtype=float)
            for k in keys:
                # md5 → 32bit 시드 → 재현 가능한 가짜 임베딩
                try:
                    import hashlib
                    seed = int(hashlib.md5(k.encode("utf-8")).hexdigest(), 16) % (2 ** 32)
                except Exception:
                    seed = abs(hash(k)) % (2 ** 32)
                rng = np.random.default_rng(seed)
                v = rng.standard_normal(dim)  # [D]
                vec += v * float(emotion_weights[k])

            nrm = float(np.linalg.norm(vec))
            if nrm > 0.0:
                vec = vec / nrm
            return vec

        # 최후 fallback: 파이썬 리스트
        dense = [float(emotion_weights[k]) for k in keys][:dim]
        if len(dense) < dim:
            dense += [0.0] * (dim - len(dense))
        return dense


# =============================================================================
# Independent Functions (drop-in)
# =============================================================================
def run_emotion_weight_calculation(
    text: str,
    *,
    emotions_file: Optional[str] = None,
    config_path: Optional[str] = None,
    use_parallel: bool = False,
    max_workers: int = 256,  # 3번 개선작업: 워커 수 극대화 (128→256)
    thresholds: Optional[Dict[str, float]] = None,
    log_file_name: str = "weight_calculator_independent.log",
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    통합 오케스트레이터를 단일 호출로 실행해 결과를 dict로 반환합니다.
    - EMOTIONS.json: 명시(emotions_file) > ENV(EMOTIONS_FILE_PATH) > 프로젝트 상대 경로 후보
    - thresholds: {"pattern_similarity": float, "stage_text_similarity": float} 오버라이드 가능
    - 반환: 가중치/전개/상황/복합/행렬/임베딩 등 전체 결과
    """
    # 빈 입력 즉시 폴백(스키마 보존)
    if not isinstance(text, str) or not text.strip():
        uni = {"희": 0.25, "노": 0.25, "애": 0.25, "락": 0.25}
        return {
            "text": text,
            "base_weights": {"emotion_weights": dict(uni), "metrics": {}},
            "progression": {},
            "situation": {},
            "compound": {},
            "adjusted": dict(uni),
            "integrated": {k: {"score": v, "sub_emotions": {}} for k, v in uni.items()},
            "matrix": {
                "weights": dict(uni),
                "interaction_matrix": {},
                "eigenvalues": [],
                "eigenvectors": [],
                "correlations": {},
                "dominant_patterns": [],
                "metrics": {},
            },
            "embedding": [0.0] * 64,
        }
    # 1) 로거
    emotion_logger = EmotionLogger(log_file_name=log_file_name)
    logger = emotion_logger.get_logger()
    logger.info("[run_emotion_weight_calculation] 독립함수 시작")

    try:
        # 2) 데이터 매니저
        dm = WeightEmotionDataManager(
            logger=emotion_logger,
            config_path=config_path,
            emotions_file=emotions_file,
        )
        # 존재하지 않으면 라벨에서 구성 후 로드
        if not dm.emotions_data:
            if not dm.load_emotions_data(emotions_file):
                logger.warning("감정 데이터 로드 실패")
                return {}

        # 안전 보강(최상위 대표 감정에 필수 키 빈 구조 백필)
        dm.ensure_top_level_profiles()
        dm.validate_emotions_schema_v1()  # 경고만 남기고 진행

        # 3) 계산기
        calc: Optional[EmotionWeightCalculator] = None
        try:
            calc = EmotionWeightCalculator(
                data_manager=dm,
                logger=emotion_logger,
                use_parallel=use_parallel,
                max_workers=max_workers,
                thresholds=thresholds,
                seed=seed,
            )

            # 4) 단계별 계산
            base_result       = calc.calculate_emotion_weights(text)          # 대표 가중치
            progression       = calc.calculate_progression(text)              # 전개(Trigger/Dev/Peak/After)
            situation         = calc.calculate_situation(text)                # 상황 매칭
            compound          = calc.calculate_compound_emotions(base_result["emotion_weights"])
            matrix            = calc.calculate_emotion_matrix(text, base_weights=base_result["emotion_weights"])  # 상호작용 행렬/고유값/상관/패턴
            adjusted          = calc.blend_weights_dynamic(compound, progression, situation, text=text)
            # Final safety normalization (uniform invariant): ensure 4-core presence + L1
            adjusted = calc._ensure_top4_prior(dict(adjusted), min_prior=0.01)
            adjusted = calc._relu_l1_normalize(dict(adjusted))
            integrated        = calc.integrate_sub_emotion_scores(adjusted, text=text)
            embedding_vec     = calc.generate_embedding(adjusted)

            # 5) 결과 조립
            out: Dict[str, Any] = {
                "text": text,
                "base_weights": base_result,          # {"emotion_weights": {...}, "metrics": ...}
                "progression": progression,           # {"희": ..., "노": ...}
                "situation": situation,               # {"희": ..., "노": ...}
                "compound": compound,                 # 복합 보정 후 분포
                "adjusted": adjusted,                 # 길이 가중까지 반영된 최종 대표 가중치
                "integrated": integrated,             # 하위 감정 top-K 통합 점수
                "matrix": matrix,                     # 상호작용 행렬/고유값/벡터/상관/패턴/지표
                "embedding": (
                    embedding_vec.tolist()
                    if hasattr(embedding_vec, "tolist") else embedding_vec
                ),
            }
            logger.info("[run_emotion_weight_calculation] 독립함수 종료")
            return out
        finally:
            try:
                if calc is not None:
                    calc.shutdown_executor()
            except Exception:
                pass

    except Exception as e:
        logger.exception(f"[run_emotion_weight_calculation] 오류: {e}")
        return {}


def run_emotion_weight_quick(
    text: str,
    *,
    emotions_file: Optional[str] = None,
    log_file_name: str = "weight_calculator_quick.log",
) -> Dict[str, Any]:
    """
    경량 버전: 대표 가중치 + 하위감정 통합까지만 빠르게 필요할 때.
    """
    # 빈 입력 즉시 폴백(스키마 보존)
    if not isinstance(text, str) or not text.strip():
        uni = {"희": 0.25, "노": 0.25, "애": 0.25, "락": 0.25}
        return {
            "text": text,
            "weights": dict(uni),
            "integrated": {k: {"score": v, "sub_emotions": {}} for k, v in uni.items()},
        }
    emotion_logger = EmotionLogger(log_file_name=log_file_name)
    dm = WeightEmotionDataManager(logger=emotion_logger, emotions_file=emotions_file)
    if not dm.emotions_data:
        if not dm.load_emotions_data(emotions_file):
            return {}
    dm.ensure_top_level_profiles()
    # 경량 경로에서도 스키마 검증 로그 노출(경고 기반)
    try:
        dm.validate_emotions_schema_v1()
    except Exception:
        pass

    calc = EmotionWeightCalculator(dm, emotion_logger, use_parallel=False)
    base  = calc.calculate_emotion_weights(text)
    comp  = calc.calculate_compound_emotions(base["emotion_weights"])
    adj   = calc.blend_weights_dynamic(comp, {}, {}, text=text)
    # Final safety normalization (uniform invariant): ensure 4-core presence + L1
    adj = calc._ensure_top4_prior(dict(adj), min_prior=0.01)
    adj = calc._relu_l1_normalize(dict(adj))
    integ = calc.integrate_sub_emotion_scores(adj, text=text)
    calc.shutdown_executor()
    return {
        "text": text,
        "weights": adj,
        "integrated": integ
    }



# =============================================================================
# Main (Weight Orchestrator) — run end-to-end and persist results
# =============================================================================
def _safe_list_shape(x) -> List[int]:
    """torch.Tensor / np.ndarray / list 모두 안전하게 shape를 반환."""
    try:
        import torch  # noqa
        if hasattr(x, "size") and callable(x.size):
            # torch.Tensor
            return list(x.size())
    except Exception:
        pass
    try:
        import numpy as np  # noqa
        if hasattr(x, "shape"):
            return list(x.shape)
    except Exception:
        pass
    if isinstance(x, list):
        return [len(x)]
    return []

def _safe_param_count(calculator: "EmotionWeightCalculator") -> int:
    """임베딩 계층 파라미터 수(없으면 0)."""
    try:
        import torch  # noqa
        if getattr(calculator, "embedding_layer", None) is not None:
            return sum(p.numel() for p in calculator.embedding_layer.parameters())
    except Exception:
        pass
    return 0

def _locate_emotions_json() -> str | None:
    """
    EMOTIONS.json 탐색(존재 시 경로 반환).
    우선순위: ENV → 모듈 기준 상대 경로 후보
    """
    cand = os.environ.get("EMOTIONS_FILE_PATH")
    if cand and os.path.exists(cand):
        return cand
    here = os.path.abspath(os.path.dirname(__file__))
    candidates = [
        os.path.join(here, "EMOTIONS.json"),
        os.path.join(here, "..", "EMOTIONS.json"),
    ]
    for c in candidates:
        c = os.path.abspath(c)
        if os.path.exists(c):
            return c
    return None

def _ensure_logs_dir() -> str:
    """emotion_analysis/logs 디렉토리 보장 (날짜별 폴더 사용)."""
    try:
        from log_manager import get_log_manager
        log_manager = get_log_manager()
        logs_dir = str(log_manager.get_log_dir("emotion_analysis", use_date_folder=True))
    except ImportError:
        # 폴백: 기존 방식 (날짜별 폴더 추가)
        base_logs_dir = os.path.join(os.path.dirname(__file__), "logs")
        today = time.strftime("%Y%m%d")
        logs_dir = os.path.join(base_logs_dir, today)
        os.makedirs(logs_dir, exist_ok=True)
    return logs_dir

def main():
    t0 = time.time()

    # 1) 로거 준비 (파일/콘솔 동시)
    logs_dir = _ensure_logs_dir()
    # 파일명만 바꿔 전달하면 EmotionLogger가 emotion_analysis/logs/ 아래 생성
    emotion_logger = EmotionLogger(log_file_name="weight_orchestrator.log")
    logger = emotion_logger.get_logger()
    logger.info("[Main] 가중치·통합 오케스트레이터 시작")

    # 2) 데이터 매니저 준비 (EMOTIONS.json 있으면 로드, 없으면 생성)
    dm = WeightEmotionDataManager(logger=emotion_logger)

    emotions_path = _locate_emotions_json()
    if emotions_path and os.path.exists(emotions_path):
        try:
            with open(emotions_path, "r", encoding="utf-8") as f:
                dm.emotions_data = json.load(f)
            logger.info(f"[Main] EMOTIONS.json 로드: {emotions_path}")
        except Exception as e:
            logger.warning(f"[Main] 기존 EMOTIONS.json 로드 실패 → 생성 시도: {e}")
            dm.load_emotions_data()  # labels/* 기반 생성
    else:
        logger.info("[Main] EMOTIONS.json 미발견 → 생성 시도")
        dm.load_emotions_data()

    # ✅ 최상위 대표감정(희/노/애/락) 필수키 빈 구조 백필 → 경고 소음 제거
    dm.ensure_top_level_profiles()

    # 스키마 검증(운영 모드에서는 엄격 검증 실패 시 hard-fail)
    from config import EA_PROFILE, RENDER_DEPLOYMENT
    is_production = EA_PROFILE == "prod" or RENDER_DEPLOYMENT or os.getenv("PRODUCTION_MODE", "0") == "1"
    
    if not dm.validate_emotions_schema_v1():
        if is_production:
            error_msg = "[Main] EMOTIONS 데이터 스키마 검증 실패 - 운영 모드에서는 허용되지 않음"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        else:
            logger.warning("[Main] EMOTIONS 데이터 스키마 경고: 일부 키가 없을 수 있습니다(진행은 계속).")

    # 3) 계산기 준비 (병렬/임베딩 옵션은 내부에서 자동 감지)
    calc = EmotionWeightCalculator(
        dm,
        logger=emotion_logger,
        use_parallel=True,
        max_workers=32,  # 라이젠 AI 9/HX 370에 맞춰 워커 수 대폭 증가
        thresholds={  # ✅ 임계치 외부 오버라이드(선택)
            "pattern_similarity": 0.15,
            "stage_text_similarity": 0.20,
        }
    )

    # 4) 샘플 텍스트 (원하면 이 부분을 외부 입력/배치로 대체 가능)
    sample_text = "친구의 결혼식 초대장을 받았을 때, 기쁨 속에서 축하 메시지를 작성했다."

    # 5) 단계별 실행 및 타이밍
    times: Dict[str, float] = {}

    print("단계별 분석 진행 중...")
    
    print("1/7 감정 가중치 계산 중...")
    t1 = time.time()
    initial = calc.calculate_emotion_weights(sample_text)
    initial_weights = initial["emotion_weights"]
    times["initial_calculation"] = time.time() - t1
    print(f"   완료 ({times['initial_calculation']:.2f}초)")

    print("2/7 감정 전개 분석 중...")
    t1 = time.time()
    prog = calc.calculate_progression(sample_text)
    times["progression_analysis"] = time.time() - t1
    print(f"   완료 ({times['progression_analysis']:.2f}초)")

    print("3/7 상황 분석 중...")
    t1 = time.time()
    situ = calc.calculate_situation(sample_text)
    times["situation_analysis"] = time.time() - t1
    print(f"   완료 ({times['situation_analysis']:.2f}초)")

    print("4/7 복합 감정 계산 중...")
    t1 = time.time()
    comp = calc.calculate_compound_emotions(initial_weights)
    times["compound_calculation"] = time.time() - t1
    print(f"   완료 ({times['compound_calculation']:.2f}초)")

    print("5/7 감정 행렬 분석 중...")
    t1 = time.time()
    matrix = calc.calculate_emotion_matrix(sample_text, base_weights=initial_weights)
    times["matrix_analysis"] = time.time() - t1
    print(f"   완료 ({times['matrix_analysis']:.2f}초)")

    # 6) 통합/보정 (가중 합성)
    adjusted = calc.blend_weights_dynamic(
        compound=comp,
        progression=prog,
        situation=situ,
        text=sample_text,  # 길이/정보량에 따라 가중 자동 조정
        min_floor=0.05
    )

    print("6/7 감정 통합 중...")
    t1 = time.time()
    integrated = calc.integrate_sub_emotion_scores(dict(adjusted))
    times["integration"] = time.time() - t1
    print(f"   완료 ({times['integration']:.2f}초)")

    # 7) 임베딩
    print("7/7 임베딩 생성 중...")
    t1 = time.time()
    embedding_vec = calc.generate_embedding(dict(adjusted))
    times["embedding_generation"] = time.time() - t1
    print(f"   완료 ({times['embedding_generation']:.2f}초)")

    # 8) 메트릭/요약
    active_cnt = sum(1 for v in adjusted.values() if v > 0.05)
    ecount = len(initial_weights)
    total_proc = time.time() - t0

    emb_shape = _safe_list_shape(embedding_vec)
    param_count = _safe_param_count(calc)

    perf = {
        "total_processing_time": round(total_proc, 3),
        "component_times": {k: round(v, 3) for k, v in times.items()},
        "emotion_metrics": {
            "total_emotions": ecount,
            "active_emotions": active_cnt,
            "average_weight": round(sum(adjusted.values()) / max(1, ecount), 3),
            "max_weight": round(max(adjusted.values() or [0.0]), 3),
            "min_weight": round(min(adjusted.values() or [0.0]), 3),
        },
        "matrix_metrics": matrix.get("metrics", {}),
        "memory_usage": {
            "embedding_size": emb_shape,
            "total_parameters": param_count,
        },
    }

    final = {
        "original_text": sample_text,
        "initial_weights": initial_weights,
        "progression_scores": prog,
        "situation_scores": situ,
        "compound_emotions": comp,
        "adjusted_weights": dict(adjusted),
        "integrated_sub_emotions": integrated,
        "emotion_matrix_analysis": matrix,
        "embedding_shape": emb_shape,
        "metrics": perf,
    }

    # 9) 저장 (logs/weight_orchestrator_YYYYMMDD_HHMMSS.json)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_name = f"weight_orchestrator_{ts}.json"
    out_path = os.path.join(logs_dir, out_name)

    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump([{"timestamp": ts, "data": final}], f, ensure_ascii=False, indent=2)
        logger.info(f"[Main] 결과 JSON 저장: {out_path}")
    except Exception as e:
        logger.exception(f"[Main] 결과 저장 실패: {e}")

    # 추가: EmotionLogger의 append-style JSON에도 남기고 싶으면(옵션)
    try:
        emotion_logger.save_json_log(final, filename="weight_orchestrator.json")
    except Exception:
        pass

    calc.shutdown_executor()
    logger.info(f"[Main] 완료 — 총 처리 시간 {perf['total_processing_time']}s, "
                f"성공 저장: logs/{out_name}")
    
    # 성능 분석 출력
    print(f"\n=== Weight Calculator 성능 분석 ===")
    print(f"총 처리 시간: {perf['total_processing_time']}초")
    print(f"\n단계별 처리 시간:")
    for step, duration in perf['component_times'].items():
        print(f"  {step}: {duration}초")
    print(f"\n감정 메트릭:")
    for metric, value in perf['emotion_metrics'].items():
        print(f"  {metric}: {value}")
    
    # 결과 요약 출력
    print(f"\n=== 분석 결과 요약 ===")
    print(f"원본 텍스트: {final['original_text']}")
    print(f"감정 가중치:")
    for emotion, weight in final['adjusted_weights'].items():
        print(f"  {emotion}: {weight:.3f}")
    print(f"임베딩 차원: {final['embedding_shape']}")
    print(f"결과 파일: {out_path}")

    return final


if __name__ == "__main__":
    main()
