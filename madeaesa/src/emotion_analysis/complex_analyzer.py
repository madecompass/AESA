# -*- coding: utf-8 -*-
from __future__ import annotations

# =========================
# Standard library imports
# =========================
import functools
import argparse
import datetime
import datetime as _dt
import gc
import html
import json
import logging
import math
import os
import random
import re
import sys
import unicodedata
import difflib
from types import SimpleNamespace
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Pattern, Literal, Set

# =========================
# Third-party imports
# =========================
import numpy as np

try:
    import torch  # type: ignore

    _HAS_TORCH = True
except Exception:
    torch = None  # type: ignore
    _HAS_TORCH = False

# =========================
# Logging setup
# =========================
_SCRIPT_DIR = Path(__file__).resolve().parent
# 통합 로그 관리자 사용 (날짜별 폴더)
try:
    from log_manager import get_log_manager
    log_manager = get_log_manager()
    _LOG_DIR = log_manager.get_log_dir("emotion_analysis", use_date_folder=True)
except ImportError:
    # 폴백: 기존 방식 (날짜별 폴더 추가)
    from datetime import datetime
    base_log_dir = Path(os.environ.get("CA_LOG_DIR") or (_SCRIPT_DIR / "logs"))
    today = datetime.now().strftime("%Y%m%d")
    _LOG_DIR = base_log_dir / today
 # 기본: 상위앱 로깅 비간섭 (디렉토리 생성도 지연)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = False

if not logger.handlers:
    # 기본: 비간섭 (핸들러 미부착 시 NullHandler로 흡수)
    logger.addHandler(logging.NullHandler())

    # 선택: 파일 로그(회전) - Opt-in via CA_FILE_LOG=1
    if os.environ.get("CA_FILE_LOG", "0") == "1":
        try:
            os.makedirs(str(_LOG_DIR), exist_ok=True)
        except Exception:
            pass
        fh = RotatingFileHandler(
            str(_LOG_DIR / "complex_analyzer.log"),
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(
            "[%(asctime)s|%(levelname)s|%(name)s:%(lineno)d] %(message)s",
            "%Y-%m-%d %H:%M:%S"))
        logger.addHandler(fh)

    # 선택: 콘솔 로그 - Opt-in via CA_CONSOLE_LOG=1
    if os.environ.get("CA_CONSOLE_LOG", "0") == "1":
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(ch)

logger.info("ComplexAnalyzer 로깅 시작")

# (기존 상수/코드 계속)
EMOTION_AXES: List[str] = ["희", "노", "애", "락"]


# ------------------------------- 인코딩 패치 (UTF-8) -------------------------------
def save_json_utf8(path: str, data: dict, cls=None) -> None:
    dir_ = os.path.dirname(path)
    if dir_:
        os.makedirs(dir_, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, cls=cls)


def load_text_utf8(path: str) -> str:
    """
    UTF-8로 텍스트 파일을 읽습니다. BOM(utf-8-sig) 우선 시도 후,
    일반 utf-8로 폴백, 그래도 실패하면 replacement로 깨짐 없이 로드합니다.
    """
    try:
        with open(path, "r", encoding="utf-8-sig") as f:  # BOM 안전
            return f.read()
    except UnicodeDecodeError:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        # 마지막 안전망: 바이트로 읽어 utf-8 replacement
        with open(path, "rb") as f:
            data = f.read()
        try:
            return data.decode("utf-8")
        except Exception:
            return data.decode("utf-8", errors="replace")


# 콘솔 UTF-8(Windows 안전)
try:
    # stdout/stderr/stdin 모두 가능할 때 reconfigure
    for _stream in (getattr(sys, "stdout", None), getattr(sys, "stderr", None), getattr(sys, "stdin", None)):
        if _stream is not None and hasattr(_stream, "reconfigure"):
            _stream.reconfigure(encoding="utf-8")
except Exception:
    pass


# ------------------------------- utils -------------------------------

def _log(msg: str) -> None:
    sys.stderr.write(f"[complex_analyzer] {msg}\n")
    sys.stderr.flush()


def softmax_temperature(xs: Sequence[float], temperature: float = 1.0) -> List[float]:
    if not xs:
        return []
    t = max(1e-6, float(temperature))
    arr = np.array(xs, dtype=np.float64) / t
    arr -= arr.max()
    exps = np.exp(arr)
    s = float(exps.sum()) or 1.0
    return [float(v) for v in (exps / s)]


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(vec))
    return vec if n <= 0 else vec / n


def deterministic_rng(seed_src: str) -> random.Random:
    import hashlib
    seed = int(hashlib.sha256(seed_src.encode("utf-8")).hexdigest()[:8], 16)
    return random.Random(seed)


def _pair(a: str, b: str) -> Tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def _now_iso() -> str:
    try:
        # Python 3.11+
        from datetime import UTC
        return datetime.datetime.now(UTC).replace(microsecond=0).isoformat().replace('+00:00', 'Z')
    except ImportError:
        # Python 3.8+
        from datetime import timezone
        return datetime.datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')


# [OPTIMIZATION] 로그 연산 캐싱 (자주 쓰이는 값)
_LOG_V_E_CACHE = {}
_LOG_DOCS_CACHE = {}

def score_subemotion_base(hits_e: int, V_e: int, df_e: int | None, N_docs: int | None) -> float:
    """
    세부 감정의 기본 점수(base score)를 계산합니다.
    (정규화된 TF) * (데이터 기반 IDF) 형태이며, log1p로 안정화됩니다.
    """
    if hits_e <= 0: return 0.0
    
    # TF: 키워드 등장 횟수를 로그 정규화된 어휘 크기로 나눔
    # [OPTIMIZATION] 자주 쓰이는 V_e 로그값 캐싱
    if V_e not in _LOG_V_E_CACHE:
        _LOG_V_E_CACHE[V_e] = max(1.0, math.log(V_e + 2))
    tf = hits_e / _LOG_V_E_CACHE[V_e]

    # IDF: DF 정보가 있으면 표준 IDF를, 없으면 1.0(영향 없음)을 사용
    if df_e is not None and N_docs is not None:
        # [OPTIMIZATION] IDF 계산 단순화 (빈번한 log 호출 억제)
        # N_docs가 고정값이라면 더 최적화 가능하나, 여기선 수식만 유지
        idf = math.log((N_docs + 1) / (df_e + 1))
    else:
        idf = 1.0

    # 최종 점수: TF-IDF 결과에 log1p를 적용하여 점수 폭주를 막고 안정성 확보
    # [OPTIMIZATION] math.log1p(x)는 x가 작을 때 log(1+x)와 같음. 근사식 사용 안 함 (정확도 유지)
    return math.log1p(max(0.0, tf * idf))


def apply_calibration(
        emo_key: str,  # "희/설렘" 같은 세부감정 키 (없으면 "희")
        base_score: float,  # score_subemotion_base 결과 (이미 log1p 상태)
        matched_patterns: dict[str, int],  # {패턴키: 매칭횟수} - 세부감정 기준
        calibrator: "WeightCalibrator | None",
        cfg: "AnalyzerConfig"
) -> float:
    """
    base_score에 데이터 기반 Calibrator(사전 확률, 패턴 가중치)를 적용합니다.
    안정적인 로그 도메인에서 연산이 수행됩니다.
    """
    # 타입 힌트를 위해 문자열로 참조 ("WeightCalibrator | None", "AnalyzerConfig")
    # 실제 실행 시에는 이 클래스들이 정의된 후에 호출되므로 문제가 없습니다.

    if not (cfg.use_calibrator and calibrator):
        return base_score  # Calibrator를 사용하지 않으면 base_score 그대로 반환

    # 1) prior 조정 (세부감정 → 없으면 대분류로 백오프)
    def _get_prior(emo: str) -> float:
        if emo in calibrator.prior_log_adj:
            return calibrator.prior_log_adj[emo]
        if "/" in emo:
            parent = emo.split("/", 1)[0]
            return calibrator.prior_log_adj.get(parent, 0.0)
        return 0.0

    # base_score는 log1p(x) = log(x+1) 상태이므로, 다시 logit으로 변환
    # log1p(x) 상태인 base_score를 안정적인 역변환(math.expm1)을 사용해 다시 log(x) 도메인으로 변환
    logit = math.log(max(cfg.epsilon, math.expm1(base_score)) + cfg.epsilon)
    logit += cfg.prior_strength * _get_prior(emo_key)

    # 2) 패턴 log-odds 합산 (binary/count 모드 + 캡)
    patt_sum = 0.0
    for pat, cnt in (matched_patterns or {}).items():
        w = calibrator.get_pattern_weight(emo_key, pat)
        if cfg.pattern_mode == "binary":
            patt_sum += w * (1 if cnt > 0 else 0)
        else:
            patt_sum += w * cnt  # count 모드

    # 점수 폭주 방지를 위해 패턴 기여도에 상한/하한(캡) 적용
    patt_sum = max(-cfg.max_pattern_contrib, min(cfg.max_pattern_contrib, patt_sum))
    logit += cfg.pattern_strength * patt_sum

    # 3) 다시 양수 스코어로 변환
    calibrated = math.exp(logit) - cfg.epsilon
    return max(0.0, calibrated)


def aggregate_category(
        cat_to_subs: dict[str, list[str]],
        sub_scores: dict[str, float]
) -> dict[str, float]:
    """
    세부 감정 점수들을 대분류(희/노/애/락)별로 균등 평균하여 집계합니다.
    """
    S = {}
    for cat, subs in cat_to_subs.items():
        if not subs:
            S[cat] = 0.0
        else:
            # 백오프 규칙으로 세부 감정 점수를 조회합니다.
            # 우선순위: 직접 키("설렘") → 접두 키("희/설렘") → 역프리픽스 보호("희/설렘" 형태로 들어온 경우 분리)
            def _fetch(sub: str, cat_name: str) -> float:
                v = sub_scores.get(sub)
                if v is not None:
                    return float(v)
                v = sub_scores.get(f"{cat_name}/{sub}")
                if v is not None:
                    return float(v)
                if "/" in sub:
                    tail = sub.split("/", 1)[-1]
                    v = sub_scores.get(tail)
                    if v is not None:
                        return float(v)
                return 0.0

            denom = max(1, len(subs))
            S[cat] = sum(_fetch(s, cat) for s in subs) / denom
    return S


def z_standardize(score: float, stats: Dict[str, float]) -> float:
    """사전 계산된 평균, 표준편차로 점수를 z-score 변환합니다."""
    mean = stats.get('mean', 0.0)
    std = stats.get('std', 1.0)
    return (score - mean) / max(1e-8, std)


def sigmoid(x: float) -> float:
    """Z-score를 [0, 1] 범위의 안정적인 점수로 변환합니다."""
    if x < -20.0: return 0.0
    if x > 20.0: return 1.0
    return 1.0 / (1.0 + math.exp(-x))


# ------------------------------- coverage-normalization utils -------------------------------
_RE_NON_ALNUM_KO = re.compile(r"[^0-9A-Za-z\uAC00-\uD7A3]+")

def _normalize_token_simple(s: str) -> str:
    """NFKC → lower → keep only Hangul/Latin/digits; collapse spaces.
    커버리지용 기본 정규화(보수적)."""
    x = unicodedata.normalize("NFKC", (s or "")).strip().lower()
    x = _RE_NON_ALNUM_KO.sub(" ", x)
    x = " ".join(x.split())
    return x


_KO_JOSA_1 = (
    "이", "가", "은", "는", "을", "를", "도", "와", "과", "의",
    "에", "로", "처럼",  # 타이포 방지용 소수 포함
)
_KO_JOSA_2 = (
    "에서", "에게", "한테", "처럼", "까지", "부터", "만", "뿐", "조차", "마저", "으로",
)
_KO_NOMINAL = ("함", "감", "함이", "감이")
_KO_ENDINGS = (
    # 매우 흔한 서술/연결 어미(보수적, 말단 1~2음절)
    "였다", "였다가", "였다면", "였다면야", "였다가도",
    "했다", "했고", "해서", "하며", "하다", "하다가", "한다", "한데",
    "였다가", "였다네", "였지", "였고",
    "다", "죠", "요", "네", "까", "구나",
)

def _normalize_token_core_for_coverage(s: str) -> str:
    """커버리지 전용 코어 정규화.
    - NFKC → lower → 한/영/숫자만 유지(간단 정규화)
    - 한국어 조사/명사화 접미/흔한 어미를 말단에서 보수적으로 제거
    - 최종 길이 < 2면 원형 유지
    """
    base = _normalize_token_simple(s)
    if not base:
        return base

    t = base
    changed = True
    while changed and len(t) >= 2:
        changed = False
        # 2글자 조사 우선 제거
        for suf in _KO_JOSA_2:
            if t.endswith(suf) and len(t) - len(suf) >= 2:
                t = t[: -len(suf)]
                changed = True
                break
        if changed:
            continue
        # 1글자 조사
        for suf in _KO_JOSA_1:
            if t.endswith(suf) and len(t) - len(suf) >= 2:
                t = t[: -len(suf)]
                changed = True
                break
        if changed:
            continue
        # 명사화 접미(함/감 계열)
        for suf in _KO_NOMINAL:
            if t.endswith(suf) and len(t) - len(suf) >= 2:
                t = t[: -len(suf)]
                changed = True
                break
        if changed:
            continue
        # 흔한 서술/연결 어미(보수적으로 말단만 제거)
        for suf in _KO_ENDINGS:
            if t.endswith(suf) and len(t) - len(suf) >= 2:
                t = t[: -len(suf)]
                changed = True
                break

    # 과다 제거 방지
    if len(t) < 2:
        return base
    return t


def length_norm(L: int, method: str = "auto") -> float:
    """문서/세그먼트 길이에 따른 정규화 스케일러."""
    method = (method or "auto").lower()
    if method == "none": return 1.0
    if method == "sqrt": return max(1.0, L) ** 0.5
    if method == "log":  return math.log(max(2, L) + 0.0)
    # auto (기본 정책)
    if L < 10: return 1.0
    if L < 50: return max(1.0, L) ** 0.5
    return math.log(max(2, L) + 0.0)


def tfidf_signal(count: float, V_e: int, V_avg: float, L: int, norm_method: str) -> float:
    """TF-IDF와 길이 정규화를 결합한 표준 신호 강도 계산."""
    # tf = count / log(V_e+2), idf ≈ log((V̄+1)/(V_e+1)) + 1
    tf = float(count) / max(1e-8, math.log(max(2, V_e) + 0.0))
    idf = math.log((float(V_avg) + 1.0) / (float(V_e) + 1.0)) + 1.0
    len_norm_val = length_norm(L, norm_method)
    return (tf * idf) / max(1e-8, len_norm_val)


# @class Preprocessor
def _pp_log(msg: str) -> None:
    try:
        _log(f"[Preprocessor] {msg}")  # use global logger if present
    except Exception:
        sys.stderr.write(f"[Preprocessor] {msg}\n")
        sys.stderr.flush()


# ------------------------------- emotions/rules -------------------------------

@dataclass
class EmotionRule:
    emotion_id: str
    keywords: List[str] = field(default_factory=list)
    sub_emotions: List[str] = field(default_factory=list)
    incompatible_with: List[str] = field(default_factory=list)
    compatible_with: List[str] = field(default_factory=list)


# ------------------------------- config (updated) -------------------------------
@dataclass
class AnalyzerConfig:
    """분석기 전체 파이프라인의 동작을 제어하는 설정 클래스."""

    # [기본 필드 - 하위 호환 유지]
    emotions_path: Optional[str] = None
    score_temperature: float = 0.75  # auto_temperature=True면 런타임에서 자동 스케일로 대체 가능
    score_min_floor: float = 0.01
    score_max_cap: float = 0.99
    min_signal_threshold: float = 0.05  # auto_min_signal=True면 런타임에서 동적으로 재설정 가능
    device: Optional[str] = None
    seed: int = 1337
    embed_dim: int = 768
    include_segments: bool = False

    # [적응형 스코어링 설정 (사용자 노출)]
    adaptive_mode: bool = True  # 모든 동적 조정 on/off
    balance_strength: float = 0.25  # 0=없음, 1=완전균형 (실사용 권장 0.2~0.3)
    norm_method: Literal["auto", "none", "sqrt", "log"] = "auto"  # 길이 정규화 방식

    # [신규: 데이터 기반 보정(Calibrator) 설정]
    use_calibrator: bool = True
    prior_strength: float = 1.0  # 사전(클래스 분포) 보정 강도
    pattern_strength: float = 1.0  # 패턴 log-odds 반영 강도
    pattern_mode: Literal["binary", "count"] = "binary"  # 패턴 가중치 적용 방식 ["binary", "count"]
    max_pattern_contrib: float = 3.0  # 패턴 가중치 합산의 상한/하한 (점수 폭주 방지)
    epsilon: float = 1e-6  # 수치 안정성을 위한 작은 값 (로그/지수 변환 시)

    use_zscore_fusion: bool = True  # True일 경우 키워드 외 다른 신호(문맥 등)를 Z-Score로 결합
    z_channel_weights: Dict[str, float] = field(default_factory=lambda: {"keyword": 0.6, "context": 0.4})

    # [내부 기본값 - 파이프라인 내부에서만 사용 (repr=False)]
    auto_temperature: bool = field(default=True, repr=False)  # 감정/클러스터 수에 따른 온도 자동화
    auto_min_signal: bool = field(default=True, repr=False)  # 세부감정 규모에 따른 임계 자동화
    clip_score: float = field(default=6.0, repr=False)

    # [운영 파라미터: 커버리지/게이트 튜닝(옵션, 기본 BALANCED)]
    CA_JUNK_MIN_COVERAGE: float = 0.03
    CA_JUNK_J_THRESHOLD: float = 0.85
    CA_JUNK_MIN_VALID_SENTENCES: int = 1

    # [그래프 컷/피드백 파라미터]
    graph_edge_quantile: float = 0.75  # 권장 기본: 0.75 (중간값보다 보수적으로 컷)
    graph_q_minmax: Tuple[float, float] = (0.30, 0.90)
    graph_q_step: float = 0.05
    graph_target_keep_ratio: Tuple[float, float] = (0.40, 0.60)


# ------------------------------- preprocessing -------------------------------

@dataclass
class PreprocessConfig:
    strip_html: bool = True
    strip_scripts_styles: bool = True
    html_unescape: bool = True
    normalize_whitespace: bool = True
    collapse_newlines: bool = True
    normalize_quotes_dashes: bool = True
    collapse_punct_runs: bool = True
    collapse_laughter: bool = True
    remove_urls: bool = True
    remove_emails: bool = True
    remove_handles: bool = True  # @username
    remove_phones: bool = False
    remove_emojis: bool = False  # 보통은 근거로 남겨둠
    lowercase_latin: bool = True
    preserve_case_korean: bool = True
    trim_edges: bool = True
    max_len: Optional[int] = None  # 너무 긴 텍스트 커팅 방지용 (None이면 무제한)
    build_alignment: bool = False  # True면 cleaned→original char index 맵 작성
    laughter_keep: int = 2  # 반복 문자(ㅋㅋㅋ/ㅎㅎㅎ 등)에서 남길 최대 글자 수
    punct_run_keep: int = 2  # 연속 구두점(!!!, ???, ……)을 몇 개로 줄일지
    newline_collapse_to: int = 2  # 연속 개행을 몇 줄로 축약할지
    ellipsis_keep: int = 3  # "…" 또는 "...." 등 엘리시스를 몇 개의 '.'으로 통일할지
    url_replacement: str = " "  # URL 제거 시 대체 문자열
    emoji_replacement: str = " "  # 이모지 제거 시 대체 문자열


@dataclass
class PreprocessStats:
    orig_len: int = 0
    clean_len: int = 0
    removed_urls: int = 0
    removed_emails: int = 0
    removed_handles: int = 0
    removed_phones: int = 0
    removed_emojis: int = 0
    collapsed_spaces: int = 0
    collapsed_punct: int = 0
    token_count: int = 0  # cleaned 전체 토큰 수(어절/숫자 포함, 구두점 제외)
    sent_lens: List[int] = field(default_factory=list)  # 문장별 토큰 수


@dataclass
class PreprocessResult:
    original: str
    cleaned: str
    segments: List[str] = field(default_factory=list)
    segment_spans: List[Tuple[int, int]] = field(default_factory=list)  # in cleaned text
    alignment_map: Optional[List[int]] = None  # cleaned_char_idx -> original_char_idx
    stats: PreprocessStats = field(default_factory=PreprocessStats)
    lang_profile: Dict[str, float] = field(default_factory=dict)  # {"ko":0.xx,"en":0.xx,"num":0.xx}


# ------------------------------- segmentation -------------------------------

@dataclass
class Segment:
    index: int
    start: int
    end: int
    text: str
    para_index: int
    sent_index: int
    kind: str = "sentence"
    delimiter: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SegmenterConfig:
    unit: str = "hybrid"  # sentence|paragraph|hybrid
    keep_delimiters: bool = True  # keep final punct in segment text
    break_on_nl: int = 2  # paragraphs split on >= this many newlines
    min_chars: int = 6  # merge very short segments
    max_chars: int = 600  # split overly long segments
    merge_short: bool = True  # enable short merge
    split_long: bool = True  # enable long split
    normalize_ws: bool = True  # normalize inner whitespace
    attach_right_quotes: bool = True  # attach closing quotes/brackets to segment
    dedent_bullets: bool = True  # strip bullet markers
    language: str = "auto"  # auto|ko|en
    window_size: int = 0  # optional rolling window size
    window_stride: int = 0  # optional rolling window stride (0->=size)
    protect_abbrev_en: bool = True  # prevent splits after common abbreviations
    protect_decimals: bool = True  # 3.14 not a boundary
    protect_initials: bool = True  # J.R.R. not a boundary
    ko_clause_split: bool = True  # allow '다/요/까' clause heuristics for ko


# ------------------------------- keyword matching -------------------------------

@dataclass
class KeywordMatcherConfig:
    """키워드 매칭 동작 옵션."""
    case_sensitive: bool = False  # 대소문자 구분 여부(영문)
    include_sub_emotions_as_keywords: bool = False  # 서브감정을 키워드로도 간주
    dedupe_hits: bool = True  # 동일 토큰 중복 제거
    ignore_overlap: bool = True  # 동일 감정 내 중복 겹침 무시
    allow_ko_suffix: bool = True  # 한글 키워드 뒤 소수의 활용/조사 허용
    ko_suffix_max: int = 2  # 허용 음절 수
    phrase_weight: float = 1.5  # 공백 포함(구/구절) 가중치
    negation_window_chars: int = 8  # 부정어 주변 N자 내 매치 무시
    negation_tokens: List[str] = field(default_factory=lambda: [
        "아니", "아닌", "아니다", "않", "않다", "않는", "못", "못하다", "못한",
        "없", "없다",
        "불가", "불가능", "불만", "불편", "불행",
        "비효율", "비정상", "비관", "비극",
        "무가치", "무의미", "무능", "무책임",
        "no", "not", "never"
    ])
    min_token_len: int = 1  # 너무 짧은 토큰 필터
    use_word_boundary_for_latin: bool = True  # 라틴 토큰은 경계(\b)로 감싸기
    max_regex_len: int = 200000  # 안전장치: 과도한 정규식 길이 방지
    escape_regex: bool = True  # 키워드 이스케이프
    compile_flags: int = re.IGNORECASE  # 정규식 플래그(기본: 대소문자 무시)


@dataclass
class KeywordHit:
    """개별 히트 정보."""
    emotion_id: str
    token: str
    start: int
    end: int
    weight: float = 1.0
    negated: bool = False


HANGUL_RANGE = r"\uAC00-\uD7A3"


def _is_latin_like(token: str) -> bool:
    return re.fullmatch(r"[A-Za-z0-9\s]+", token or "") is not None


@functools.lru_cache(maxsize=100_000)
def _cached_token_pattern(
        token: str,
        allow_ko_suffix: bool,
        ko_suffix_max: int,
        escape_regex: bool,
        use_word_boundary_for_latin: bool,
) -> str:
    """토큰 하나를 느슨 매칭용 정규식 조각으로 변환(옵션 조합까지 캐시 키에 포함)."""
    t = (token or "").strip()
    if not t:
        return ""

    if _is_latin_like(t):
        base = re.escape(t) if escape_regex else t
        base = base.replace(r"\ ", r"\s+")  # 공백 유연화
        return rf"\b{base}\b" if use_word_boundary_for_latin else base

    # 한글: 글자 사이 \s* 허용 + (옵션) 접미 허용
    chars = [re.escape(ch) for ch in t]
    body = r"\s*".join(chars)
    suffix = ""
    if allow_ko_suffix and ko_suffix_max > 0:
        suffix = rf"[{HANGUL_RANGE}]{{0,{int(ko_suffix_max)}}}"
    return rf"(?:{body}){suffix}"


# ------------------------------- feature extraction -------------------------------

@dataclass
class FeatureExtractorConfig:
    """세그먼트 단위 통계/휴리스틱 피처 추출 설정."""
    max_len_clip: int = 800
    max_tokens_clip: int = 120
    normalize_base_chars: int = 200
    normalize_base_tokens: int = 40
    include_emoticons: bool = True
    include_emoji: bool = True
    include_temporal: bool = True
    include_stopword_ratio: bool = True
    include_josa_count: bool = True
    include_booster_hedge: bool = True
    include_repetition: bool = True
    include_case_ratio: bool = True
    include_script_ratio: bool = True
    include_basic_punct: bool = True
    default_norm_method: str = "auto"
    # B-1) 길이 정규화 임계값 설정 추가
    ln_auto_small: int = 10      # < small → none
    ln_auto_medium: int = 50     # small ≤ L < medium → sqrt, 그 이상 log


# =============================================================================
# RuleLoader
# =============================================================================
class RuleLoader:
    # 영어/한자/국문 동의어 매핑(느슨한 정규화)
    _EID_SYNONYMS = {
        "희": {"희", "기쁨", "joy", "joyful", "happiness", "happy", "喜", "pleasure", "delight", "희열"},
        "노": {"노", "분노", "anger", "angry", "rage", "怒", "irritation", "resentment"},
        "애": {"애", "슬픔", "sadness", "sad", "哀", "sorrow", "grief", "depress"},
        "락": {"락", "즐거움", "fun", "pleasure", "樂", "amusement", "relief", "안도", "흥분"},
    }

    # 확장자 후보
    _EXTS = (".json", ".jsonl", ".txt")

    def __init__(self, path: Optional[str]):
        self.path = path

    # ------------------------------ public ------------------------------
    def load(self) -> Dict[str, EmotionRule]:
        default = self._default_rules()
        if not self.path:
            return default

        sources = self._resolve_sources(self.path)
        if not sources:
            _log(f"emotions source not found: {self.path} (using defaults)")
            return default

        rules: Dict[str, EmotionRule] = dict(default)
        loaded_any = False
        for fp in sources:
            try:
                raw = self._read_text(fp)
                if not raw.strip():
                    _log(f"empty file skipped: {fp}")
                    continue
                raw = self._strip_json_comments(raw)
                obj = self._parse_json_or_jsonl(raw)
                if obj is None:
                    _log(f"parse failed (skip): {fp}")
                    continue
                rules = self._merge_payload(rules, obj)
                loaded_any = True
            except Exception as e:
                _log(f"failed to load {fp}: {e} (skipped)")

        # 최종 정리(정규화/검증/누락보완)
        rules = self._finalize(rules, ensure_all_axes=True)
        self._log_summary(rules, sources, used_default=not loaded_any)
        return rules

    # ------------------------------ default seed ------------------------------
    def _default_rules(self) -> Dict[str, EmotionRule]:
        return {
            "희": EmotionRule("희",
                             keywords=["기쁨", "행복", "만족", "긍정", "환희", "설렘", "희망", "안도"],
                             sub_emotions=["기쁨", "만족", "희열", "충만함", "자신감", "평온함", "안도감"],
                             incompatible_with=["노", "애"],
                             compatible_with=["락"]),
            "노": EmotionRule("노",
                             keywords=["분노", "짜증", "화", "격분", "반발", "적대", "불만", "불안"],
                             sub_emotions=["분노", "좌절", "적개심", "질투", "거부감", "불안"],
                             incompatible_with=["희", "락"],
                             compatible_with=["애"]),
            "애": EmotionRule("애",
                             keywords=["슬픔", "우울", "상실", "눈물", "허무", "절망", "걱정"],
                             sub_emotions=["슬픔", "상실감", "우울함", "허탈감", "힘겨움", "걱정"],
                             incompatible_with=["락", "희"],
                             compatible_with=["노"]),
            "락": EmotionRule("락",
                             keywords=["즐거움", "재미", "행복감", "여유", "안심", "편안", "쾌활", "흥분"],
                             sub_emotions=["즐거움", "재미", "흥분", "해방감", "활기", "여유로움"],
                             incompatible_with=["노", "애"],
                             compatible_with=["희"]),
        }

    # ------------------------------ source resolving ------------------------------
    def _resolve_sources(self, spec: str) -> List[str]:
        """
        spec 예시:
          - "/path/EMOTIONS.json"
          - "/path/dir"  -> dir 내 *.json|*.jsonl|*.txt
          - "a.json,b.json"  -> 콤마/세미콜론 구분
          - "rules/*.json"  -> glob
        """
        import glob
        parts: List[str] = []
        for token in re.split(r"[;,]", spec):
            token = token.strip()
            if not token:
                continue
            parts.append(token)

        out: List[str] = []
        for p in parts:
            if os.path.isdir(p):
                for ext in self._EXTS:
                    out.extend(sorted(glob.glob(os.path.join(p, f"*{ext}"))))
                continue
            # glob 패턴
            g = glob.glob(p)
            if g:
                out.extend(sorted(g))
                continue
            # 단일 파일
            if os.path.isfile(p):
                out.append(p)
            else:
                # 확장자 생략 시 후보 시도
                for ext in self._EXTS:
                    cand = f"{p}{ext}"
                    if os.path.isfile(cand):
                        out.append(cand)
                        break
        # 중복 제거(순서 유지)
        seen = set()
        uniq = []
        for x in out:
            if x not in seen:
                seen.add(x)
                uniq.append(x)
        return uniq

    # ------------------------------ io/parse helpers ------------------------------
    def _read_text(self, path: str) -> str:
        # UTF-8(BOM 포함) 안전 읽기 — 유틸에 위임
        return load_text_utf8(path)

    def _strip_json_comments(self, s: str) -> str:
        # // 한줄, # 한줄, /* 블록 */ 제거 (문자열 내부까지 완벽히 구분하진 않지만 일반 규칙 파일에는 충분)
        s = re.sub(r"/\*.*?\*/", "", s, flags=re.DOTALL)
        s = re.sub(r"^\s*//.*?$", "", s, flags=re.MULTILINE)
        s = re.sub(r"^\s*#.*?$", "", s, flags=re.MULTILINE)
        return s

    def _parse_json_or_jsonl(self, raw: str) -> Optional[Any]:
        raw_strip = raw.strip()
        if not raw_strip:
            return None
        # JSON 객체/배열 우선
        if raw_strip[0] in "{[":
            try:
                return json.loads(raw_strip)
            except json.JSONDecodeError:
                pass
        # JSONL 시도(유연 모드: 빈줄/주석 제거됨)
        items = []
        ok = False
        for line in raw_strip.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
                ok = True
            except Exception:
                # jsonl 라인 중 일부만 파싱 가능한 경우: 가능한 라인만 사용
                continue
        if ok:
            return items
        return None

    # ------------------------------ merging logic ------------------------------
    def _merge_payload(self, base: Dict[str, EmotionRule], payload: Any) -> Dict[str, EmotionRule]:
        """
        payload 형태:
          1) dict: { "희": {...}, "노": {...} }
          2) list(jsonl): [ {"emotion_id":"희", "keywords":[...]}, {...}, {"희": {...}} ...]
        기본 정책: 병합(유니온). 노드에 "replace": true 가 있으면 해당 필드는 교체.
        """
        rules = dict(base)

        if isinstance(payload, dict):
            rules = self._merge_dict(rules, payload)
        elif isinstance(payload, list):
            for item in payload:
                if isinstance(item, dict):
                    if "emotion_id" in item or "emotion" in item:
                        rules = self._merge_line_object(rules, item)
                    else:
                        # {"희": {...}} 같은 형태의 단일 dict 라인
                        rules = self._merge_dict(rules, item)
                else:
                    continue
        else:
            # 알 수 없는 구조는 무시
            pass

        return rules

    def _merge_dict(self, rules: Dict[str, EmotionRule], obj: Dict[str, Any]) -> Dict[str, EmotionRule]:
        out = dict(rules)
        for k, v in obj.items():
            if not isinstance(v, dict):
                continue
            eid = self._canon_eid(k)
            if not eid:
                # 감정키가 아닌 경우 무시
                continue
            out = self._upsert_rule(out, eid, v)
        return out

    def _merge_line_object(self, rules: Dict[str, EmotionRule], item: Dict[str, Any]) -> Dict[str, EmotionRule]:
        out = dict(rules)
        raw_eid = item.get("emotion_id") or item.get("emotion")
        eid = self._canon_eid(str(raw_eid)) if raw_eid is not None else None
        if not eid:
            return out
        node = {k: v for k, v in item.items() if k not in ("emotion_id", "emotion")}
        out = self._upsert_rule(out, eid, node)
        return out

    def _upsert_rule(self, rules: Dict[str, EmotionRule], eid: str, node: Dict[str, Any]) -> Dict[str, EmotionRule]:
        """
        node 예시:
          {
            "keywords": [...], "sub_emotions":[...],
            "incompatible_with":["노"], "compatible_with":["락"],
            "replace": false | true  # true면 각 필드 교체, 아니면 유니온
          }
        """
        out = dict(rules)
        cur = out.get(eid, EmotionRule(eid))
        replace = bool(node.get("replace", False))

        # keywords
        if "keywords" in node and isinstance(node["keywords"], (list, tuple)):
            vals = self._sanitize_tokens(node["keywords"])
            cur.keywords = vals if replace else self._union_preserve(cur.keywords, vals)

        # sub_emotions
        if "sub_emotions" in node and isinstance(node["sub_emotions"], (list, tuple)):
            vals = self._sanitize_tokens(node["sub_emotions"])
            cur.sub_emotions = vals if replace else self._union_preserve(cur.sub_emotions, vals)

        # incompatible_with / compatible_with (정규화 필요)
        if "incompatible_with" in node and isinstance(node["incompatible_with"], (list, tuple)):
            vals = self._normalize_relation_list(node["incompatible_with"])
            cur.incompatible_with = vals if replace else self._union_preserve(cur.incompatible_with, vals)

        if "compatible_with" in node and isinstance(node["compatible_with"], (list, tuple)):
            vals = self._normalize_relation_list(node["compatible_with"])
            cur.compatible_with = vals if replace else self._union_preserve(cur.compatible_with, vals)

        out[eid] = cur
        return out

    # ------------------------------ normalization ------------------------------
    def _canon_eid(self, raw: str) -> Optional[str]:
        if raw is None:
            return None
        s = str(raw).strip()
        if not s:
            return None
        low = s.lower()
        # 정확 일치 우선
        if s in EMOTION_AXES:
            return s
        # 동의어 탐색
        for canon, syns in self._EID_SYNONYMS.items():
            if s in syns or low in syns:
                return canon
        # 한 글자 한자(喜怒哀樂) 처리
        mapping = {"喜": "희", "怒": "노", "哀": "애", "樂": "락"}
        if s in mapping:
            return mapping[s]
        return None

    def _sanitize_tokens(self, seq: Sequence[Any], max_items: int = 2048, max_len: int = 128) -> List[str]:
        out: List[str] = []
        seen = set()
        for x in seq:
            if not isinstance(x, str):
                try:
                    x = str(x)
                except Exception:
                    continue
            t = x.strip()
            if not t:
                continue
            if len(t) > max_len:
                t = t[:max_len]
            if t not in seen:
                seen.add(t)
                out.append(t)
            if len(out) >= max_items:
                break
        return out

    def _union_preserve(self, a: Sequence[str], b: Sequence[str], max_items: int = 4096) -> List[str]:
        out: List[str] = []
        seen = set()
        for x in list(a) + list(b):
            if x in seen:
                continue
            seen.add(x)
            out.append(x)
            if len(out) >= max_items:
                break
        return out

    def _normalize_relation_list(self, seq: Sequence[Any], max_items: int = 16) -> List[str]:
        out: List[str] = []
        seen = set()
        for x in seq:
            eid = self._canon_eid(str(x))
            if not eid:
                continue
            if eid not in seen:
                seen.add(eid)
                out.append(eid)
            if len(out) >= max_items:
                break
        return out

    # ------------------------------ finalize/validate ------------------------------
    def _finalize(self, rules: Dict[str, EmotionRule], ensure_all_axes: bool = True) -> Dict[str, EmotionRule]:
        out: Dict[str, EmotionRule] = {}
        for eid, r in rules.items():
            if eid not in EMOTION_AXES:
                # 비정규 키는 무시
                continue
            kw = self._sanitize_tokens(r.keywords)
            subs = self._sanitize_tokens(r.sub_emotions)
            inc = self._normalize_relation_list([x for x in (r.incompatible_with or []) if x != eid])
            com = self._normalize_relation_list([x for x in (r.compatible_with or []) if x != eid])

            # 상호 모순 정리: 같은 축이 inc/com 모두에 있으면 inc 우선
            com = [x for x in com if x not in set(inc)]

            out[eid] = EmotionRule(
                emotion_id=eid,
                keywords=kw,
                sub_emotions=subs,
                incompatible_with=inc,
                compatible_with=com,
            )

        if ensure_all_axes:
            # 누락 축 보완
            for eid in EMOTION_AXES:
                if eid not in out:
                    out[eid] = EmotionRule(eid)

        return out

    # ------------------------------ logging ------------------------------
    def _log_summary(self, rules: Dict[str, EmotionRule], sources: List[str], used_default: bool) -> None:
        try:
            src_info = f"{len(sources)} file(s)" if sources else "0 file"
            parts = []
            for eid in EMOTION_AXES:
                r = rules.get(eid) or EmotionRule(eid)
                parts.append(
                    f"{eid}: kw={len(r.keywords)}, sub={len(r.sub_emotions)}, "
                    f"inc={len(r.incompatible_with)}, com={len(r.compatible_with)}"
                )
            default_tag = "defaults only" if used_default else "merged"
            _log(f"rules loaded [{default_tag}] from {src_info} | " + " | ".join(parts))
        except Exception:
            pass


# =============================================================================
# Preprocessor (개선 적용)
# =============================================================================
class Preprocessor:
    _RE_URL = re.compile(r"(?i)\b((?:https?://|ftp://|www\.)[^\s<>\"']+)", re.UNICODE)
    _RE_EMAIL = re.compile(r"(?i)\b[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}\b")
    _RE_HANDLE = re.compile(r"(^|(?<=\s))@[A-Za-z0-9_]{2,}\b")
    _RE_PHONE = re.compile(r"\b(?:\+?\d[\d\-\.\s]{7,}\d)\b")
    _RE_EMOJI = re.compile(
        "["  # 기본 유니코드 이모지/기호 범위
        "\U0001F300-\U0001FAFF"
        "\U00002600-\U000026FF"
        "\U00002700-\U000027BF"
        "\U0001F900-\U0001F9FF"
        "\U0001F1E6-\U0001F1FF"
        "\uFE0E-\uFE0F"
        "]+", flags=re.UNICODE
    )
    _RE_VS = re.compile(r"[\uFE0E\uFE0F]")

    _RE_ZW = re.compile(r"[\u200B\u200C\u200D\u2060\uFEFF]")
    _RE_CTRL = re.compile(r"[\u0000-\u0008\u000B\u000C\u000E-\u001F]")
    _RE_SPACE = re.compile(r"[ \t\f\v]+")
    _RE_NL = re.compile(r"\s*\n\s*")
    _RE_MULTI_NL = re.compile(r"\n{2,}")
    _RE_MULTI_PUNCT = re.compile(r"([!?.,…])\1{1,}")
    _RE_ELLIPSIS = re.compile(r"…|\.{3,}")
    _RE_LAUGH = re.compile(r"([ㅋㅎ]+)\1{1,}", flags=re.IGNORECASE)
    _RE_SENT_SPLIT = re.compile(r"([.!?…。！？])\s*|(\n+)|((?:다|요)\s+)")
    _RE_TOKEN = re.compile(r"[가-힣]+|[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:[.,]\d+)*|[^\s\w]", re.UNICODE)

    def __init__(self, cfg: Optional[PreprocessConfig] = None):
        self.cfg = cfg or PreprocessConfig()
        self._last: Optional[PreprocessResult] = None

    # === 외부 API ===
    def preprocess(self, text: str) -> str:
        res = self._run_pipeline(text)
        self._last = res
        return res.cleaned

    def get_last_result(self) -> Optional[PreprocessResult]:
        return self._last

    def segment(self, text: Optional[str] = None) -> List[str]:
        if text is not None:
            self.preprocess(text)
        return list(self._last.segments) if self._last else []

    def segment_with_spans(self, text: Optional[str] = None) -> List[Tuple[int, int, str]]:
        if text is not None:
            self.preprocess(text)
        if not self._last:
            return []
        return [(s, e, self._last.cleaned[s:e]) for (s, e) in self._last.segment_spans]

    def tokens_for_rules(self, text: Optional[str] = None, drop_stopwords: bool = False) -> List[Dict[str, Any]]:
        if text is not None:
            self.preprocess(text)
        if not self._last:
            return []
        toks: List[Dict[str, Any]] = []
        for m in self._RE_TOKEN.finditer(self._last.cleaned):
            tok = m.group(0)
            if drop_stopwords and self._is_stopword(tok):
                continue
            toks.append({"text": tok, "start": m.start(), "end": m.end()})
        return toks

    # === 키워드 유틸 ===
    def build_keyword_regex(self, keywords: List[str]) -> re.Pattern:
        safe: List[str] = []
        for k in keywords:
            k_str = str(k or "").strip()
            if k_str:
                safe.append(re.escape(k_str))
        if not safe:
            return re.compile(r"$^")
        pattern = r"\b(" + "|".join(safe) + r")\b"
        return re.compile(pattern, flags=re.IGNORECASE)

    def iter_keyword_hits(self, text: Optional[str], pat: re.Pattern):
        if text is not None:
            self.preprocess(text)
        if not self._last:
            return
        for m in pat.finditer(self._last.cleaned):
            yield {"match": m.group(1), "start": m.start(1), "end": m.end(1)}

    # === 파이프라인 ===
    def _run_pipeline(self, text: str) -> PreprocessResult:
        stats = PreprocessStats(orig_len=len(text))
        t = text

        # 1) 길이 제한
        if self.cfg.max_len is not None and self.cfg.max_len > 0:
            t = t[:self.cfg.max_len]

        # 2) HTML 제거/정리
        if self.cfg.strip_html:
            t = self._strip_html_keep_text(t, scripts=self.cfg.strip_scripts_styles)
        if self.cfg.html_unescape:
            try:
                t = html.unescape(t)
            except Exception:
                pass

        # 3) 기본 정규화
        t = t.replace("\r\n", "\n").replace("\r", "\n")
        t = self._normalize_chars(t)

        # 4) 제거/치환(강도/대체어 반영)
        if self.cfg.remove_urls:
            t, c = self._subn(self._RE_URL, t, self.cfg.url_replacement)
            stats.removed_urls += c
        if self.cfg.remove_emails:
            t, c = self._subn(self._RE_EMAIL, t, " ")
            stats.removed_emails += c
        if self.cfg.remove_handles:
            t, c = self._subn(self._RE_HANDLE, t, " ")
            stats.removed_handles += c
        if self.cfg.remove_phones:
            t, c = self._subn(self._RE_PHONE, t, " ")
            stats.removed_phones += c
        if self.cfg.remove_emojis:
            t, c = self._subn(self._RE_EMOJI, t, self.cfg.emoji_replacement)
            stats.removed_emojis += c
            t = self._RE_VS.sub("", t)

        # 5) 반복 문자/웃음, 공백/개행/구두점 런(강도 반영)
        if self.cfg.collapse_laughter:
            keep = max(1, int(self.cfg.laughter_keep))
            # "ㅋㅋㅋㅋ" -> "ㅋㅋ" (keep 개수로 절단)
            t, _ = self._subn(self._RE_LAUGH, t, lambda m: m.group(1)[:keep])

        if self.cfg.normalize_whitespace:
            t, c = self._collapse_spaces(t)
            stats.collapsed_spaces += c
            if self.cfg.collapse_newlines:
                prev = t
                # 주변 공백 제거 + 다중 개행 축약
                t = self._RE_NL.sub("\n", t)
                target_nl = "\n" * max(1, int(self.cfg.newline_collapse_to))
                t = self._RE_MULTI_NL.sub(target_nl, t)
                stats.collapsed_spaces += max(0, len(prev) - len(t))

        if self.cfg.collapse_punct_runs:
            before = len(t)
            keep = max(1, int(self.cfg.punct_run_keep))
            # "!!!" -> "!!", "???" -> "??", "……" -> "..."
            t = self._RE_MULTI_PUNCT.sub(lambda m: m.group(1) * keep, t)
            dots = "." * max(1, int(self.cfg.ellipsis_keep))
            t = self._RE_ELLIPSIS.sub(dots, t)
            stats.collapsed_punct += max(0, before - len(t))

        if self.cfg.trim_edges:
            t = t.strip()
        if self.cfg.lowercase_latin:
            t = self._lowercase_latin_only(t)
        t = self._strip_zero_width_and_ctrl(t)

        # 6) 문장 분할
        seg_spans, segs = self._split_sentences(t)

        # 7) 언어 비율/정렬 맵(옵션)
        profile = self._language_profile(t)
        align_map = self._build_alignment_map(
            text[:(self.cfg.max_len or len(text))], t
        ) if self.cfg.build_alignment else None

        # 8) 길이/토큰 통계(길이 정규화에 활용)
        stats.clean_len = len(t)
        stats.token_count = self._count_tokens_text(t)
        stats.sent_lens = [self._count_tokens_text(s) for s in segs]

        return PreprocessResult(
            original=text, cleaned=t, segments=segs, segment_spans=seg_spans,
            alignment_map=align_map, stats=stats, lang_profile=profile
        )

    # === HTML/문자 정규화 ===
    def _strip_html_keep_text(self, s: str, scripts: bool = True) -> str:
        if scripts:
            s = re.sub(r"(?is)<(script|style)\b.*?>.*?</\1>", " ", s)
        s = re.sub(r"(?i)</?(p|br|div|h[1-6]|li|ul|ol|blockquote|section|article)\b[^>]*>", "\n", s)
        s = re.sub(r"<[^>]+>", " ", s)
        return s

    def _normalize_chars(self, s: str) -> str:
        s = s.replace("\u3000", " ")
        s = unicodedata.normalize("NFKC", s)
        if self.cfg.normalize_quotes_dashes:
            s = s.translate(str.maketrans({
                "“": '"', "”": '"', "‟": '"', "„": '"', "＂": '"',
                "‘": "'", "’": "'", "‚": "'", "‛": "'", "＇": "'",
                "–": "-", "—": "-", "―": "-", "−": "-",
            }))
            s = self._RE_ELLIPSIS.sub("." * max(1, int(self.cfg.ellipsis_keep)), s)
        return s

    def _collapse_spaces(self, s: str) -> Tuple[str, int]:
        before = len(s)
        s = self._RE_SPACE.sub(" ", s)
        return s, max(0, before - len(s))

    def _lowercase_latin_only(self, s: str) -> str:
        return "".join(ch.lower() if 'A' <= ch <= 'Z' else ch for ch in s)

    def _strip_zero_width_and_ctrl(self, s: str) -> str:
        s = self._RE_ZW.sub("", s)
        s = self._RE_CTRL.sub("", s)
        return s

    # === 분할/통계 ===
    def _split_sentences(self, s: str) -> Tuple[List[Tuple[int, int]], List[str]]:
        if not s.strip():
            return [], []
        parts = self._RE_SENT_SPLIT.split(s)
        sentences, current = [], ""
        for part in filter(None, parts):
            current += part
            sp = part.strip()
            if sp and sp[-1] in ".!?…。！？요다\n":
                sentences.append(current.strip())
                current = ""
        if current.strip():
            sentences.append(current.strip())
        if not sentences and s.strip():
            sentences = [s.strip()]

        # spans 복원
        spans, last_pos = [], 0
        for sent in sentences:
            try:
                start = s.index(sent, last_pos)
                end = start + len(sent)
                spans.append((start, end))
                last_pos = end
            except ValueError:
                spans.append((last_pos, last_pos + len(sent)))
                last_pos += len(sent) + 1
        return spans, sentences

    def _language_profile(self, s: str) -> Dict[str, float]:
        if not s:
            return {"ko": 0.0, "en": 0.0, "num": 0.0, "other": 0.0}
        total = max(1, len(s))
        ko = sum(1 for ch in s if "가" <= ch <= "힣")
        en = sum(1 for ch in s if "a" <= ch.lower() <= "z")
        num = sum(1 for ch in s if ch.isdigit())
        return {"ko": ko / total, "en": en / total, "num": num / total, "other": (total - ko - en - num) / total}

    def _build_alignment_map(self, original: str, cleaned: str) -> List[int]:
        sm = difflib.SequenceMatcher(a=original, b=cleaned, autojunk=False)
        mapping: List[int] = [-1] * len(cleaned)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag in ("equal", "replace"):
                for dj, di in zip(range(j1, j2), range(i1, i2)):
                    if dj < len(mapping):
                        mapping[dj] = di
        last = 0
        for i in range(len(mapping)):
            if mapping[i] == -1:
                mapping[i] = last
            else:
                last = mapping[i]
        return mapping

    # === 토큰/정규 치환 헬퍼 ===
    def _is_stopword(self, tok: str) -> bool:
        stop_ko = {"그리고", "그러나", "하지만", "또는", "또", "및", "에서", "으로", "에게", "하다", "있는", "없는", "했다"}
        stop_en = {"the", "a", "an", "and", "or", "but", "to", "in", "of", "on", "for", "as", "is", "are", "was",
                   "were"}
        if not tok or tok.isspace():
            return True
        # 구두점 단독 토큰/순수 기호는 stop 취급
        if not any(c.isalnum() for c in tok):
            return True
        lower_tok = tok.lower()
        if 'a' <= lower_tok[0] <= 'z' and lower_tok in stop_en:
            return True
        if '가' <= tok[0] <= '힣' and tok in stop_ko:
            return True
        return False

    def _count_tokens_text(self, s: str) -> int:
        """구두점 단독 토큰은 제외하고 어절/숫자 중심으로 카운트."""
        n = 0
        for m in self._RE_TOKEN.finditer(s):
            tok = m.group(0)
            if any(c.isalnum() for c in tok):  # 문자/숫자 포함 토큰만
                n += 1
        return n

    def _subn(self, pat: re.Pattern, s: str, repl: Any) -> Tuple[str, int]:
        if callable(repl):
            count = 0

            def _wrap(m):
                nonlocal count
                count += 1
                return repl(m)

            return pat.sub(_wrap, s), count
        else:
            return pat.subn(repl, s)


# =============================================================================
# Segmenter (improved)
# =============================================================================
class Segmenter:
    _END_PUNCT = ".?!…。！？"
    _QUOTES_RIGHT = "”’'\"》〉』】)]}」"
    _BULLETS = re.compile(r"^\s*([•◦▪︎▫︎●■□\-–—*+]|\(\d+\)|\d+\.)\s+")
    _WS = re.compile(r"[ \t\f\v]+")
    _NL_MULTI = re.compile(r"\n{2,}")
    _ELLIPSIS = re.compile(r"\.\.+|…")

    _EN_ABBREV = {
        "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "vs.", "etc.", "e.g.", "i.e.", "fig.", "al.", "no.",
        "jan.", "feb.", "mar.", "apr.", "jun.", "jul.", "aug.", "sep.", "oct.", "nov.", "dec."
    }
    _KO_JOINERS = {"그리고", "그러나", "하지만", "그래서", "그러면", "또한", "때문에"}

    # 한국어 절 경계(후방탐색 미사용: 모바일/임베디드 호환성↑)
    _KO_CLAUSE_SPLIT = re.compile(r"([가-힣](?:다|요|죠|임|까|네|래|군|습니다|합니다|합다))\s+")

    _SPLIT_HINTS = re.compile(r"[;:，、／/|，、]|(?<!\d),(?!\d)")  # 숫자 쉼표 제외
    _DIGIT_PERIOD = re.compile(r"\d\.\d")
    _INITIALS = re.compile(r"(?:[A-Z]\.){2,}")
    _ENDS_WITH_QUOTE = re.compile(r"[\"'”’)\]}]+$")
    _TOKEN = re.compile(r"[가-힣]+|[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:[.,]\d+)*|[^\s\w]")

    def __init__(self, cfg: Optional[SegmenterConfig] = None):
        self.cfg = cfg or SegmenterConfig()
        # 선택: 토큰 상한(문자 상한과 병행). 기존 Config에 없던 값이므로 내부 기본값으로만 사용.
        self._max_tokens_internal = 0  # 0이면 비활성

    # --- 외부 API ---
    def segment(self, text: str) -> List[str]:
        segs = self.segment_with_spans(text)
        return [s.text for s in segs] if segs else ([text.strip()] if text.strip() else [])

    def segment_with_spans(self, text: str) -> List[Segment]:
        clean = self._normalize_text(text)
        para_spans = self._split_paragraphs(clean)

        segments: List[Segment] = []
        seg_idx = 0
        for p_idx, (p_start, p_end) in enumerate(para_spans):
            p_txt = clean[p_start:p_end]

            if self.cfg.unit == "paragraph":
                units = [(p_start, p_end, p_txt, "", {"para_only": True})]
            else:
                units = self._split_sentences_in_para(p_txt, p_start)

            if self.cfg.merge_short:
                units = self._merge_short(units)
            if self.cfg.split_long:
                units = self._split_long(units)

            for s_idx, (s, e, t, delim, meta) in enumerate(units):
                clean_text = t.strip()
                if not clean_text:
                    continue

                # 공백 제거 전후 인덱스 보정
                start_offset = t.find(clean_text)
                end_offset = start_offset + len(clean_text)

                # 절(clauses) 경계 절대 좌표로 기록 (변화량/지속성 계산 재사용)
                clause_spans_rel = self._scan_clauses_in_sentence(clean_text)
                clause_spans_abs = [(s + start_offset + cs, s + start_offset + ce) for (cs, ce) in clause_spans_rel]

                # 토큰 길이 기록(성능/정규화에 유용)
                tok_len = self._token_count(clean_text)

                meta_out = {
                    **meta,
                    "para_span": (p_start, p_end),
                    "len_chars": len(clean_text),
                    "len_tokens": tok_len,
                    "clause_spans": clause_spans_abs,  # 절 경계 절대 인덱스
                    "sent_delim": delim,
                }

                segments.append(Segment(
                    index=seg_idx, start=s + start_offset, end=s + end_offset, text=clean_text,
                    para_index=p_idx, sent_index=s_idx,
                    kind="sentence" if self.cfg.unit != "paragraph" else "paragraph",
                    delimiter=delim, meta=meta_out
                ))
                seg_idx += 1

        return segments

    def windows(self, segments: Sequence[str], size: Optional[int] = None, stride: Optional[int] = None) -> List[
        List[str]]:
        ws = size if size is not None else (self.cfg.window_size or 0)
        st = stride if stride is not None else (self.cfg.window_stride or ws)
        if ws <= 0 or not segments:
            return []
        st = max(1, st)
        return [list(segments[i: i + ws]) for i in range(0, len(segments), st)]

    # --- 정규화 ---
    def _normalize_text(self, t: str) -> str:
        x = t.replace("\r\n", "\n").replace("\r", "\n")
        x = self._ELLIPSIS.sub("...", x)
        if self.cfg.normalize_ws:
            x = self._WS.sub(" ", x)
            x = self._NL_MULTI.sub("\n\n", x)
        return x.strip()

    # --- 단락 분리 ---
    def _split_paragraphs(self, s: str) -> List[Tuple[int, int]]:
        if not s:
            return []
        break_re = re.compile(r"\n{" + str(max(1, self.cfg.break_on_nl)) + ",}")
        spans: List[Tuple[int, int]] = []
        last = 0
        for m in break_re.finditer(s):
            start, end = last, m.start()
            if s[start:end].strip():
                spans.append((start, end))
            last = m.end()
        if last < len(s) and s[last:].strip():
            spans.append((last, len(s)))
        return spans if spans else ([(0, len(s))] if s.strip() else [])

    # --- 문장 분리 ---
    def _split_sentences_in_para(self, p_txt: str, offset: int) -> List[Tuple[int, int, str, str, Dict[str, Any]]]:
        units = self._scan_sentence_boundaries(p_txt, offset)
        out: List[Tuple[int, int, str, str, Dict[str, Any]]] = []
        for s, e, delim in units:
            raw = p_txt[s - offset: e - offset]
            if self.cfg.dedent_bullets:
                raw = self._BULLETS.sub("", raw)
            if raw.strip():
                out.append((s, e, raw, delim, {}))
        return out

    def _scan_sentence_boundaries(self, txt: str, base: int) -> List[Tuple[int, int, str]]:
        boundaries = [0]

        # 1) 종결부호/개행
        for m in re.finditer(r"([.!?…。！？])\s*|(\n+)", txt):
            boundaries.append(m.end())

        # 2) 한국어 절 분리(문장 내부 서브 경계 힌트)
        if self.cfg.ko_clause_split and self._needs_clause_split(txt):
            for m in self._KO_CLAUSE_SPLIT.finditer(txt):
                boundaries.append(m.end())

        boundaries = sorted(set(boundaries + [len(txt)]))

        units: List[Tuple[int, int, str]] = []
        last_b = 0
        for b in boundaries:
            if b <= last_b:
                continue

            segment_text = txt[last_b:b]
            end_char_pos = b - 1

            # 보호 로직(소수점, 이니셜, 영문 약어)
            if segment_text.endswith("."):
                if self.cfg.protect_decimals and self._is_decimal_point(txt, end_char_pos):
                    last_b = b
                    continue
                if self.cfg.protect_initials and self._is_initials_near(txt, end_char_pos):
                    last_b = b
                    continue
                if self.cfg.protect_abbrev_en and self._looks_like_abbrev(txt, end_char_pos):
                    last_b = b
                    continue

            if segment_text.strip():
                delimiter = txt[b - 1:b] if txt[b - 1:b].strip() in self._END_PUNCT else ""
                units.append((base + last_b, base + b, delimiter))
            last_b = b

        return units

    # --- 절(clauses) 스캔: 문장 내부 서브 스팬(상대 좌표) ---
    def _scan_clauses_in_sentence(self, sent: str) -> List[Tuple[int, int]]:
        if not sent.strip():
            return []
        # 힌트: 한국어 종결(다/요 계열), 세미콜론/쉼표류, 슬래시 등
        hints = [0]
        for m in self._SPLIT_HINTS.finditer(sent):
            hints.append(m.end())
        if self.cfg.ko_clause_split:
            for m in self._KO_CLAUSE_SPLIT.finditer(sent):
                hints.append(m.end())
        hints = sorted(set(hints + [len(sent)]))

        clauses: List[Tuple[int, int]] = []
        last = 0
        for h in hints:
            if h <= last:
                continue
            frag = sent[last:h].strip()
            if frag:
                # 절단 시 공백 제외 인덱스 복원
                start_rel = last + (sent[last:h].find(frag))
                end_rel = start_rel + len(frag)
                clauses.append((start_rel, end_rel))
            last = h
        # 보정: 절이 하나도 없으면 전체를 1절로
        return clauses or [(0, len(sent.strip()))]

    # --- 보호/휴리스틱 ---
    def _is_decimal_point(self, s: str, i: int) -> bool:
        return i > 0 and i < len(s) - 1 and s[i - 1].isdigit() and s[i + 1].isdigit()

    def _is_initials_near(self, s: str, i: int) -> bool:
        return bool(self._INITIALS.search(s, max(0, i - 10), i + 2))

    def _looks_like_abbrev(self, s: str, i: int) -> bool:
        match = re.search(r"\b([A-Za-z]{1,5})\.$", s[:i + 1])
        return match is not None and match.group(0).lower() in self._EN_ABBREV

    def _needs_clause_split(self, seg_text: str) -> bool:
        if self.cfg.language not in ("auto", "ko"):
            return False
        if not any("가" <= c <= "힣" for c in seg_text):
            return False
        return len(seg_text) > 80 and any(k in seg_text for k in self._KO_JOINERS)

    # --- 단축/분할 ---
    def _merge_short(self, units: List[Tuple[int, int, str, str, Dict]]) -> List[Tuple[int, int, str, str, Dict]]:
        if not units or self.cfg.min_chars <= 0:
            return units

        merged_units: List[Tuple[int, int, str, str, Dict]] = []
        buffer = list(units[0])

        for i in range(1, len(units)):
            if len(buffer[2]) < self.cfg.min_chars:
                next_unit = units[i]
                buffer[1] = next_unit[1]  # end
                buffer[2] += " " + next_unit[2]  # text
                buffer[3] = next_unit[3]  # delimiter
                buffer[4]["merged"] = True
            else:
                merged_units.append(tuple(buffer))
                buffer = list(units[i])

        if buffer:
            merged_units.append(tuple(buffer))

        return merged_units

    def _split_long(self, units: List[Tuple[int, int, str, str, Dict]]) -> List[Tuple[int, int, str, str, Dict]]:
        """문자 상한(max_chars) + (내부) 토큰 상한을 고려해 긴 문장을 안정적으로 분할."""
        max_chars = max(0, int(self.cfg.max_chars))
        max_tokens = max(0, int(self._max_tokens_internal))

        if max_chars <= 0 and max_tokens <= 0:
            return units

        final_units: List[Tuple[int, int, str, str, Dict]] = []
        for s, e, t, d, m in units:
            if (max_chars <= 0 or len(t) <= max_chars) and (max_tokens <= 0 or self._token_count(t) <= max_tokens):
                final_units.append((s, e, t, d, m))
                continue

            # 반복적으로 자르기
            current_offset = 0
            while True:
                remain = t[current_offset:]
                if not remain:
                    break

                need_cut = False
                if max_chars > 0 and len(remain) > max_chars:
                    need_cut = True
                    slice_text = remain[:max_chars]
                else:
                    slice_text = remain

                if max_tokens > 0:
                    if self._token_count(slice_text) > max_tokens:
                        need_cut = True
                        # 토큰 기반 범위 재조정
                        slice_text = self._slice_by_tokens(remain, max_tokens)

                if not need_cut:
                    # 남은 부분을 tail로 추가
                    tail = remain.strip()
                    if tail:
                        ps = s + current_offset + remain.find(tail)
                        pe = ps + len(tail)
                        final_units.append((ps, pe, tail, d, {**m, "split_long_tail": True}))
                    break

                # cut 위치 계산(힌트/공백/문장 경계 우선)
                cut_pos = self._find_long_cut(slice_text)
                part = slice_text[:cut_pos].strip()
                if not part:
                    # 안전장치: 빈 조각 방지
                    part = slice_text.strip()
                    cut_pos = len(slice_text)

                part_start = s + current_offset + (slice_text.find(part))
                part_end = part_start + len(part)
                final_units.append((part_start, part_end, part, "", {**m, "split_long": True}))

                current_offset += cut_pos

        return final_units

    def _find_long_cut(self, chunk: str) -> int:
        # 1) 힌트(;, :, 쉼표류, 슬래시 등)
        hints = list(self._SPLIT_HINTS.finditer(chunk))
        if hints:
            return hints[-1].end()
        # 2) 공백
        last_space = chunk.rfind(" ")
        if last_space > len(chunk) * 0.4:
            return last_space + 1
        # 3) 토큰 경계 추정(알파벳/한글/숫자 변경 시점)
        for i in range(len(chunk) - 1, max(5, int(len(chunk) * 0.6)), -1):
            if self._is_boundary_like(chunk, i):
                return i + 1
        # 4) 최후: 끝까지
        return len(chunk)

    # --- 유틸 ---
    def _is_boundary_like(self, s: str, i: int) -> bool:
        a, b = s[i - 1], s[i]

        # 문자 클래스가 바뀌는 지점 / 공백 / 구두점
        def _class(ch: str) -> int:
            if "가" <= ch <= "힣":
                return 1
            if ch.isalpha():
                return 2
            if ch.isdigit():
                return 3
            if ch.isspace():
                return 4
            return 5

        return _class(a) != _class(b) or a.isspace() or b.isspace() or (a in ",;:/" or b in ",;:/")

    def _token_count(self, s: str) -> int:
        return sum(1 for m in self._TOKEN.finditer(s) if any(c.isalnum() for c in m.group(0)))

    def _slice_by_tokens(self, s: str, max_tokens: int) -> str:
        """앞에서부터 max_tokens 토큰을 포함하는 부분 문자열 반환."""
        cnt = 0
        last_end = 0
        for m in self._TOKEN.finditer(s):
            tok = m.group(0)
            if any(c.isalnum() for c in tok):
                cnt += 1
            last_end = m.end()
            if cnt >= max_tokens:
                break
        return s[:last_end] if last_end else s


# =============================================================================
# KeywordMatcher (improved)
# =============================================================================
class KeywordMatcher:
    """룰 키워드를 공백/조사 변형에 견고하도록 '느슨한 한글 매칭' + 가벼운 한국어 접미 추림을 적용.
       adaptive_mode=True일 때 문장 단위 유니크 매치 기반 TF 가중 누적을 수행한다."""
    HANGUL_RANGE = r"\uAC00-\uD7A3"

    _KO_SUFFIXES: Tuple[str, ...] = (
        "함이", "함", "감이", "감", "스러웠", "스러워서", "스러워", "스러움",
        "돼서", "해서", "해서요", "라서", "이라서", "에서", "으로", "으로는",
        "까지", "조차", "라도", "처럼", "에게", "에게서", "으로서", "으로써",
        "도", "만", "은", "는", "이", "가", "을", "를", "과", "와", "마다",
        "밖에", "부터", "보다", "뿐", "께", "께서", "야", "나마", "이나마"
    )

    _RE_SENT_END = re.compile(r"([.!?…。！？])\s*|(\n+)")
    _RE_TOKEN = re.compile(r"[가-힣]+|[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:[.,]\d+)*|[^\s\w]", re.UNICODE)

    def __init__(self, rules: Dict[str, "EmotionRule"], cfg: Optional[Any] = None):
        self.rules = rules
        self.cfg = cfg or SimpleNamespace(
            include_sub_emotions_as_keywords=True,
            min_token_len=1,
            max_regex_len=200000,
            compile_flags=re.IGNORECASE,
            escape_regex=True,
            use_word_boundary_for_latin=True,
            allow_ko_suffix=True,
            ko_suffix_max=2,
            ignore_overlap=True,
            dedupe_hits=True,
            negation_window_chars=8,
            negation_tokens=["아니", "아닌", "아니다", "않", "않다", "않는", "못", "못하다", "못한", "없", "없다", "불가", "불가능", "불만", "불편",
                             "불행", "비효율", "비정상", "비관", "비극", "무가치", "무의미", "무능", "무책임", "no", "not", "never"],
            phrase_weight=1.25,
            adaptive_mode=True,
            norm_method="log",
        )
        
        # 미니패치 2: 길이 정규화 임계값 설정 (FeatureExtractor와 일관성 유지)
        self.cfg.ln_auto_small = getattr(self.cfg, "ln_auto_small", 10)
        self.cfg.ln_auto_medium = getattr(self.cfg, "ln_auto_medium", 50)
        
        self.regex: Dict[str, re.Pattern] = {}
        self._kw_index: Dict[str, List[str]] = {}
        self._vocab_total: Dict[str, int] = {}  # V_e
        self._vocab_avg_total: float = 1.0  # V̄

        self._last_token_count: int = 0
        self._build()

    # ------------------------------ Public API ------------------------------

    def match(self, text: str) -> Tuple[Dict[str, float], Dict[str, List[str]]]:
        """adaptive_mode=True: 문장 단위 유니크 매치 + TF 가중 누적(부동소수 반환)."""
        if getattr(self.cfg, "adaptive_mode", False):
            return self._adaptive_match(text)
        return self._legacy_match(text)

    def match_detailed(self, text: str) -> Dict[str, Dict[str, Any]]:
        """감정별 상세 매치 정보(위치/가중치/부정 여부)와 커버리지 문장 길이 계산(기존 동작 유지)."""
        results: Dict[str, Dict[str, Any]] = {
            eid: {"hits": [], "coverage_chars": 0, "occupied": []} for eid in EMOTION_AXES
        }

        # 1) 원문 기준 탐색
        all_hits: List[Any] = []
        for eid, pattern in self.regex.items():
            if not pattern:
                continue
            for match in pattern.finditer(text):
                matched_text = match.group(1)
                start, end = match.span(1)
                if self._is_negated(text, start, end):
                    continue
                token = self._recover_token(eid, matched_text) or matched_text
                weight = self._weight_for_token(token)
                all_hits.append(self._make_hit(eid, token, start, end, weight))

        # 2) stemmed 텍스트 보조 탐색
        text_proc = self._preprocess_text(text)
        for eid, pattern in self.regex.items():
            if any((h.get("emotion_id") if isinstance(h, dict) else getattr(h, "emotion_id", None)) == eid for h in
                   all_hits):
                continue
            for match in pattern.finditer(text_proc):
                stemmed = match.group(1)
                rough = re.escape(stemmed).replace(r"\ ", r"\s+")
                m2 = re.search(rough, text, flags=re.IGNORECASE)
                if not m2:
                    continue
                s2, e2 = m2.span()
                if self._is_negated(text, s2, e2):
                    continue
                token = self._recover_token(eid, stemmed) or stemmed
                weight = self._weight_for_token(token)
                all_hits.append(self._make_hit(eid, token, s2, e2, weight))

        # 3) 겹침/중복 정리
        sorted_hits = sorted(
            all_hits,
            key=lambda h: ((h["start"] if isinstance(h, dict) else getattr(h, "start")),
                           -((h["end"] if isinstance(h, dict) else getattr(h, "end")) - (
                               h["start"] if isinstance(h, dict) else getattr(h, "start"))))
        )
        occupied: List[Tuple[int, int]] = []
        for h in sorted_hits:
            s = h["start"] if isinstance(h, dict) else getattr(h, "start")
            e = h["end"] if isinstance(h, dict) else getattr(h, "end")
            if self.cfg.ignore_overlap and self._is_overlapping(occupied, s, e):
                continue
            eid = h["emotion_id"] if isinstance(h, dict) else getattr(h, "emotion_id")
            results[eid]["hits"].append(h)
            occupied.append((s, e))

        # 4) 커버리지(문자 길이)
        for eid in EMOTION_AXES:
            spans = [
                (h["start"] if isinstance(h, dict) else getattr(h, "start"),
                 h["end"] if isinstance(h, dict) else getattr(h, "end"))
                for h in results[eid]["hits"]
            ]
            results[eid]["coverage_chars"] = self._coverage_len(spans)
            results[eid]["occupied"] = spans

        return results

    # ------------------------------ Adaptive/Legacy ------------------------------

    def _adaptive_match(self, text: str) -> Tuple[Dict[str, float], Dict[str, List[str]]]:
        """문장 단위 유니크 매치 기반 TF 가중 누적. 부동소수 스코어를 반환."""
        self._last_token_count = self._token_count(text)

        sentences = self._split_sentences_simple(text)
        counts: Dict[str, float] = {e: 0.0 for e in EMOTION_AXES}
        hits_set: Dict[str, Set[str]] = {e: set() for e in EMOTION_AXES}
        
        # 미니패치 1: 문장별 유니크 히트 기록
        per_sent_hits: List[Dict[str, Set[str]]] = []

        for sent in sentences:
            sent_hits: Dict[str, Set[str]] = {}
            for eid, pat in self.regex.items():
                matches = pat.findall(sent)
                if not matches:
                    continue
                uniq = set((x if isinstance(x, str) else x[0]).strip() for x in matches if x)
                if not uniq:
                    continue
                sent_hits[eid] = uniq
                hits_set[eid].update(uniq)
                
                # TF 가중 누적
                V_e = max(1, self._vocab_total.get(eid, 1))
                tf_weight = len(uniq) / max(1.0, math.log(V_e + 2))
                counts[eid] += tf_weight
                
            per_sent_hits.append(sent_hits)

        # 미니패치 1: 문장 단위 보조 가중(대비/강조) - 정확도 향상
        contrast_tokens = ("하지만", "그러나", "그런데")
        intensity_tokens = ("정말", "진짜", "너무", "엄청", "완전")

        for sent, sh in zip(sentences, per_sent_hits):
            # 대비 접속사가 있는 문장은 상충 가능성을 고려해 모든 감정 신호를 소폭 증폭
            if any(tok in sent for tok in contrast_tokens):
                for eid in EMOTION_AXES:
                    if counts[eid] > 0:
                        counts[eid] *= 1.05  # +5%

            # 강조 부사가 있는 문장은 해당 문장에만 감정 키워드가 있을 때 가중
            if any(tok in sent for tok in intensity_tokens):
                for eid in EMOTION_AXES:
                    # 이 문장의 히트에만 기반하여 정확한 가중 적용
                    if eid in sh:  # 이 문장에 해당 감정 히트가 있을 때만
                        counts[eid] *= 1.08  # +8%

        hits: Dict[str, List[str]] = {k: sorted(list(v))[:64] for k, v in hits_set.items()}
        return counts, hits

    def _legacy_match(self, text: str) -> Tuple[Dict[str, float], Dict[str, List[str]]]:
        """기존 findall 기반 단순 카운팅(정수→부동소수로 캐스팅만)."""
        text_proc = self._preprocess_text(text)
        self._last_token_count = self._token_count(text)

        counts: Dict[str, float] = {e: 0.0 for e in EMOTION_AXES}
        hits: Dict[str, List[str]] = {e: [] for e in EMOTION_AXES}

        for eid, pat in self.regex.items():
            m = pat.findall(text_proc)
            if not m:
                continue
            counts[eid] = float(len(m))
            uniq = list({(x if isinstance(x, str) else x[0]).strip() for x in m})
            hits[eid] = uniq[:64]
        return counts, hits

    # ------------------------------ Coverage (TF–IDF형) ------------------------------

    def _kw_coverage(self, eid: str, kw_hits: List[str]) -> float:
        """TF–IDF형 커버리지: tf=uniq/log(V_e+2), idf≈log((V̄+1)/(V_e+1))+1, len_norm∈{1,√L,log(L+2)}."""
        uniq = len({(k or "").strip().lower() for k in (kw_hits or []) if k})
        V_e = max(1, self._vocab_total.get(eid, 1))
        V_avg = max(1.0, self._vocab_avg_total)

        tf = uniq / max(1.0, math.log(V_e + 2))
        idf = math.log((V_avg + 1.0) / (V_e + 1.0)) + 1.0

        L = max(1, int(self._last_token_count or 1))
        len_norm = self._len_norm(L, getattr(self.cfg, "norm_method", "log"))
        return float((tf * idf) / max(1e-8, len_norm))

    # ------------------------------ Rule update ------------------------------

    def _clear_caches(self) -> None:
        """룰/옵션 변경 시 캐시된 토큰→패턴 변환기를 초기화."""
        try:
            _cached_token_pattern.cache_clear()
        except Exception:
            pass

    def update_rules(self, rules: Dict[str, "EmotionRule"]) -> None:
        self.rules = rules or {}
        self._clear_caches()
        self._build()

    def add_keywords(self, eid: str, tokens: List[str]) -> None:
        if eid not in self.rules:
            self.rules[eid] = EmotionRule(emotion_id=eid)
        cur = set(self.rules[eid].keywords or [])
        for t in tokens:
            if t:
                cur.add(t)
        self.rules[eid].keywords = sorted(cur)
        self._clear_caches()
        self._build()

    # ------------------------------ Internals ------------------------------

    def _build(self) -> None:
        """규칙으로부터 느슨 정규식을 생성/컴파일 + 어휘 통계(V_e, V̄) 계산."""
        self.regex.clear()
        self._kw_index.clear()
        self._vocab_total.clear()

        for eid, rule in (self.rules or {}).items():
            kw = list(rule.keywords or [])
            if getattr(self.cfg, "include_sub_emotions_as_keywords", False):
                kw.extend([x for x in (rule.sub_emotions or []) if isinstance(x, str)])

            tokens = self._sanitize_tokens(kw)
            if not tokens:
                continue

            tokens.sort(key=len, reverse=True)
            parts: List[str] = []
            total_len = 0
            max_len = int(self.cfg.max_regex_len)
            compiled_tokens: List[str] = []
            for t in tokens:
                p = self._token_to_pattern(t)
                if not p:
                    continue
                if total_len + len(p) + 1 > max_len:
                    continue  # skip this token but still consider remaining shorter tokens
                parts.append(p)
                compiled_tokens.append(t)
                total_len += len(p) + 1

            if not parts:
                continue

            self._kw_index[eid] = compiled_tokens
            self._vocab_total[eid] = len(compiled_tokens)

            final = "(" + "|".join(parts) + ")"
            try:
                self.regex[eid] = re.compile(final, int(self.cfg.compile_flags))
            except re.error:
                esc = "(" + "|".join(re.escape(t) for t in compiled_tokens) + ")"
                self.regex[eid] = re.compile(esc, re.IGNORECASE)

        vals = list(self._vocab_total.values())
        self._vocab_avg_total = float(sum(vals) / len(vals)) if vals else 1.0

    def _token_to_pattern(self, token: str) -> str:
        """캐시된 토큰→패턴 변환기로 교체."""
        t = (token or "").strip()
        if not t or len(t) < int(self.cfg.min_token_len):
            return ""
        # utils에 올려둔 _cached_token_pattern 호출
        return _cached_token_pattern(
            t,
            bool(getattr(self.cfg, "allow_ko_suffix", True)),
            int(getattr(self.cfg, "ko_suffix_max", 2)),
            bool(getattr(self.cfg, "escape_regex", True)),
            bool(getattr(self.cfg, "use_word_boundary_for_latin", True)),
        )

    # ----- text preprocessing (rough stemming) -----
    def _preprocess_text(self, text: str) -> str:
        toks = re.findall(r"[가-힣A-Za-z0-9]+", text)
        stemmed = [self._ko_stem_rough(t) for t in toks]
        return " ".join(stemmed)

    def _ko_stem_rough(self, tok: str) -> str:
        for suf in self._KO_SUFFIXES:
            if tok.endswith(suf) and len(tok) > len(suf) + 1:
                return tok[: -len(suf)]
        return tok

    # ----- sentence / token utils -----
    def _split_sentences_simple(self, s: str) -> List[str]:
        if not s.strip():
            return []
        parts = self._RE_SENT_END.split(s)
        sentences, cur = [], ""
        for part in filter(None, parts):
            cur += part
            if part.strip() and (part.strip()[-1:] in ".!?…。！？" or part.count("\n") > 0):
                if cur.strip():
                    sentences.append(cur.strip())
                cur = ""
        if cur.strip():
            sentences.append(cur.strip())
        return sentences

    def _token_count(self, s: str) -> int:
        return sum(1 for m in self._RE_TOKEN.finditer(s) if any(c.isalnum() for c in m.group(0)))

    # ----- generic utils -----
    def _sanitize_tokens(self, tokens: List[str]) -> List[str]:
        out, seen = [], set()
        for t in tokens:
            s = (t or "").strip()
            if not s:
                continue
            k = s.lower()
            if k in seen:
                continue
            seen.add(k)
            out.append(s)
        return out

    def _recover_token(self, eid: str, matched: str) -> Optional[str]:
        cands = self._kw_index.get(eid, [])
        nm = self._normalize_token(matched)
        best, best_score = None, 0.0
        for c in cands:
            nc = self._normalize_token(c)
            if nm.startswith(nc):
                sc = len(nc) / max(1, len(nm))
                if sc > best_score:
                    best, best_score = c, sc
        return best

    def _normalize_token(self, s: str) -> str:
        x = (s or "").strip().lower()
        return re.sub(r"[^\w\s" + self.HANGUL_RANGE + "]", "", x)

    def _is_negated(self, text: str, start: int, end: int) -> bool:
        if int(self.cfg.negation_window_chars) <= 0:
            return False
        w0 = max(0, start - int(self.cfg.negation_window_chars))
        w1 = min(len(text), end + int(self.cfg.negation_window_chars))
        window = text[w0:w1]
        toks = [re.escape(n) for n in (self.cfg.negation_tokens or []) if n]
        if not toks:
            return False
        pat = r"(" + "|".join(toks) + r")"
        return bool(re.search(pat, window, re.IGNORECASE))

    def _weight_for_token(self, token: str) -> float:
        return float(self.cfg.phrase_weight) if (" " in (token or "")) else 1.0

    def _is_overlapping(self, occupied: List[Tuple[int, int]], s: int, e: int) -> bool:
        for os, oe in occupied:
            if s < oe and e > os:
                return True
        return False

    def _coverage_len(self, spans: List[Tuple[int, int]]) -> int:
        if not spans:
            return 0
        spans = sorted(spans)
        merged = []
        cs, ce = spans[0]
        for ns, ne in spans[1:]:
            if ns < ce:
                ce = max(ce, ne)
            else:
                merged.append((cs, ce))
                cs, ce = ns, ne
        merged.append((cs, ce))
        return sum(e - s for s, e in merged)

    def _len_norm(self, L: int, method: str) -> float:
        """길이 정규화 스케일: none|sqrt|log|auto(default=log if unknown)."""
        m = (method or "").lower()
        if m == "none":
            return 1.0
        if m == "sqrt":
            return math.sqrt(max(1, L))
        if m == "log":
            return math.log(max(1, L) + 2.0)
        # 미니패치 2: auto 모드에서 설정값 사용 (FeatureExtractor와 일관성)
        if m == "auto":
            small = int(getattr(self.cfg, "ln_auto_small", 10))
            medium = int(getattr(self.cfg, "ln_auto_medium", 50))
            if L < small:
                return 1.0
            if L < medium:
                return math.sqrt(L)
            return math.log(L + 2.0)
        # fallback for unknown method
        return math.log(max(1, L) + 2.0)

    # KeywordHit 호환(없으면 dict 반환)
    def _make_hit(self, eid: str, token: str, start: int, end: int, weight: float) -> Any:
        KH = globals().get("KeywordHit")
        if KH is not None:
            try:
                return KH(eid, token, start, end, weight)
            except Exception:
                pass
        return {"emotion_id": eid, "token": token, "start": start, "end": end, "weight": weight}


# =============================================================================
# FeatureExtractor
# =============================================================================
class FeatureExtractor:
    """세그먼트 통계·휴리스틱 피처 추출기.
       - 길이 정규화 스칼라(len_norm)를 계산해 스코어러로 전달
       - (옵션) kw_hits, pattern_weights를 받아 유니크 키워드/가중합을 계산해 전달
    """

    # 최소 한국어/영어 불용어·부정어·강조어·완화어
    _KO_STOP = {
        "은", "는", "이", "가", "의", "을", "를", "에", "에서", "으로", "로", "와", "과", "도", "만", "뿐", "마다", "까지", "부터", "보다",
        "그리고", "그러나", "하지만", "또한", "그래서", "때문에", "정도", "좀", "조금", "매우", "아주", "그", "이", "저", "그것", "이것", "저것",
        "했다", "하다", "하게", "되다", "된다", "했다가", "하며", "하며도", "라고", "이며", "거나", "거나도", "보다도"
    }
    _EN_STOP = {
        "a", "an", "the", "and", "or", "but", "so", "because", "to", "of", "for", "in", "on", "at", "with", "by",
        "from", "that", "this", "it", "is", "are", "was", "were", "be", "been", "being", "as", "if", "then", "than",
        "very", "really", "just", "only", "also", "too"
    }
    _KO_NEG = {"아니", "아닌", "아니다", "않", "못", "무", "없", "미", "비", "불", "비-", "부정", "노"}
    _EN_NEG = {"no", "not", "never", "none", "nothing", "nobody", "nowhere", "hardly", "scarcely", "barely", "n't"}
    _KO_BOOST = {"매우", "아주", "정말", "진짜", "완전", "너무", "엄청", "격하게", "극도로", "대단히", "몹시", "굉장히", "특별히"}
    _EN_BOOST = {"very", "really", "so", "extremely", "utterly", "totally", "completely", "highly", "greatly",
                 "strongly", "super"}
    _KO_HEDGE = {"아마", "아마도", "대체로", "대략", "혹시", "어쩌면", "대충", "약간", "다소", "왠지", "느낌", "같다", "듯하다"}
    _EN_HEDGE = {"maybe", "perhaps", "likely", "roughly", "somewhat", "slightly", "kinda", "sorta", "seems", "seem",
                 "appears"}

    _KO_JOSA = {"은", "는", "이", "가", "을", "를", "에", "에서", "에게", "께", "으로", "로", "와", "과", "도", "만", "뿐", "마다", "까지",
                "부터", "보다", "처럼", "같이"}

    # 중복된 "ㅋㅋㅋ", "ㅎㅎㅎ" 등
    _KO_LAUGHTER = {"ㅋㅋ", "ㅎㅎ", "하하", "헤헤", "호호"}
    _EN_LAUGHTER = {"lol", "lmao", "haha", "hehe", "rofl"}

    _TEMPORAL = {"어제", "오늘", "내일", "방금", "지금", "현재", "곧", "이후", "이전", "곧바로", "잠시후", "방금전", "최근", "요즘", "과거", "미래",
                 "곧장", "yesterday", "today", "tomorrow", "now", "soon", "later", "earlier", "recently", "currently",
                 "future", "past", "immediately"}

    _EMOJI_MAP: Dict[str, set] = {
        "희": set(list("😀😃😄😁😆😊🙂☺️🤗✨🎉🥳🤩👍")),
        "노": set(list("😠😡🤬👿💢😤👊🖐️😾")),
        "애": set(list("😢😭😞😔😟☹️🙁🫤")),
        "락": set(list("😎🤙🎈🎊🎮🎵😜😝😛🤪")),
    }

    _EMOTICON_POS = re.compile(r"(?:\^_\^|\^\^|:-\)|:\)|;\)|\(:|\(\^|\^o\^|ㅎㅎ+|ㅋㅋ+)", re.I)
    _EMOTICON_NEG = re.compile(r"(?:ㅠ+|ㅜ+|T_T|TT|;_;|:-\(|:\()", re.I)
    _EMOTICON_NEU = re.compile(r"(?:-_-|=_=|\.\._|\.__\.)")

    _RE_TOK = re.compile(r"[A-Za-z0-9]+|[\uac00-\ud7a3]+|[^\s]", re.UNICODE)
    _RE_EXCL = re.compile(r"!+")
    _RE_QUES = re.compile(r"\?+")
    _RE_ELL = re.compile(r"(?:\.{3,}|…+)")
    _RE_QUOTES = re.compile(r"[\"“”‘’']")
    _RE_PARENS = re.compile(r"[\(\)\[\]\{\}]")
    _RE_COMMA = re.compile(r",")
    _RE_PERIOD = re.compile(r"\.")
    _RE_DASH = re.compile(r"[–—-]")
    _RE_SEMI = re.compile(r";")
    _RE_COLON = re.compile(r":")
    _RE_REPEAT = re.compile(r"(.)\1{2,}", re.UNICODE)
    _RE_DIGIT = re.compile(r"\d")
    _RE_LINE = re.compile(r"\n")

    def __init__(self, cfg: Optional[FeatureExtractorConfig] = None):
        self.cfg = cfg or FeatureExtractorConfig()
        
        # B-2) 부정/강조/완화 토큰을 환경변수로 확장 가능
        import os
        extra_boost = os.getenv("EXTRA_BOOST_TOKENS", "")
        if extra_boost.strip():
            self._KO_BOOST = self._KO_BOOST | set(x.strip() for x in extra_boost.split(",") if x.strip())
        
        extra_hedge = os.getenv("EXTRA_HEDGE_TOKENS", "")
        if extra_hedge.strip():
            self._KO_HEDGE = self._KO_HEDGE | set(x.strip() for x in extra_hedge.split(",") if x.strip())
        
        extra_neg = os.getenv("EXTRA_NEG_TOKENS", "")
        if extra_neg.strip():
            self._KO_NEG = self._KO_NEG | set(x.strip() for x in extra_neg.split(",") if x.strip())
            
        # 미니패치 3: 영문 토큰 확장도 환경변수로 허용
        extra_boost_en = os.getenv("EXTRA_BOOST_TOKENS_EN", "")
        if extra_boost_en.strip():
            self._EN_BOOST = self._EN_BOOST | set(x.strip().lower() for x in extra_boost_en.split(",") if x.strip())
        
        extra_hedge_en = os.getenv("EXTRA_HEDGE_TOKENS_EN", "")
        if extra_hedge_en.strip():
            self._EN_HEDGE = self._EN_HEDGE | set(x.strip().lower() for x in extra_hedge_en.split(",") if x.strip())
        
        extra_neg_en = os.getenv("EXTRA_NEG_TOKENS_EN", "")
        if extra_neg_en.strip():
            self._EN_NEG = self._EN_NEG | set(x.strip().lower() for x in extra_neg_en.split(",") if x.strip())

    # --------------------------------------------------------------------- #
    # 핵심 API
    # --------------------------------------------------------------------- #
    def extract(
            self,
            segment: str,
            *,
            norm_method: Optional[str] = None,
            kw_hits: Optional[Dict[str, List[str]]] = None,
            pattern_weights: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> Dict[str, Any]:
        """
        세그먼트 피처 추출 + (옵션) 정규화/중복 제거/패턴 가중치 정보 계산.

        Args:
            segment: 분석 대상 텍스트(단일 세그먼트).
            norm_method: 길이 정규화 방식("auto"|"none"|"sqrt"|"log"). None이면 cfg.default_norm_method.
            kw_hits: (옵션) 감정ID -> [키워드 텍스트] 목록. 유니크 키워드/가중합 계산에 사용.
            pattern_weights: (옵션, Phase 2) 감정ID -> {키워드: weight}. 제공 시 유니크 키워드의 가중 합 계산.
        """
        text = segment or ""
        L = len(text)
        toks = self._tokenize(text)
        T = len(toks)

        # ---- 문자/스크립트/구두점/규칙성 통계 ----
        hangul_chars = sum(1 for ch in text if '\uac00' <= ch <= '\ud7a3')
        latin_chars = sum(1 for ch in text if ('A' <= ch <= 'Z') or ('a' <= ch <= 'z'))
        digit_chars = len(self._RE_DIGIT.findall(text))
        other_chars = max(0, L - hangul_chars - latin_chars - digit_chars)

        exclam = len(self._RE_EXCL.findall(text)) if self.cfg.include_basic_punct else 0
        ques = len(self._RE_QUES.findall(text)) if self.cfg.include_basic_punct else 0
        ell = len(self._RE_ELL.findall(text)) if self.cfg.include_basic_punct else 0
        quotes = len(self._RE_QUOTES.findall(text)) if self.cfg.include_basic_punct else 0
        parens = len(self._RE_PARENS.findall(text)) if self.cfg.include_basic_punct else 0
        comma = len(self._RE_COMMA.findall(text)) if self.cfg.include_basic_punct else 0
        period = len(self._RE_PERIOD.findall(text)) if self.cfg.include_basic_punct else 0
        dash = len(self._RE_DASH.findall(text)) if self.cfg.include_basic_punct else 0
        semi = len(self._RE_SEMI.findall(text)) if self.cfg.include_basic_punct else 0
        colon = len(self._RE_COLON.findall(text)) if self.cfg.include_basic_punct else 0

        lines = len(self._RE_LINE.findall(text)) + (1 if L > 0 else 0)

        # 문장 수 추정(간단): !, ?, . 의 등장횟수
        sent_enders = exclam + ques + period
        sentences_est = max(1, sent_enders) if text.strip() else 0

        repeats = self._RE_REPEAT.findall(text) if self.cfg.include_repetition else []
        repeat_runs = len(repeats)
        repeat_longest = max((len(m) for m in repeats), default=0)

        upper_tokens = sum(1 for w in toks if w.isalpha() and w.upper() == w and any('A' <= c <= 'Z' for c in w))
        upper_chars = sum(1 for ch in text if 'A' <= ch <= 'Z')
        lower_chars = sum(1 for ch in text if 'a' <= ch <= 'z')

        josa_cnt = self._count_josa(toks) if self.cfg.include_josa_count else 0
        stop_cnt = self._count_stopwords(toks) if self.cfg.include_stopword_ratio else 0

        boosters = hedges = negs = 0
        if self.cfg.include_booster_hedge:
            boosters = self._count_in_set(toks, self._KO_BOOST | self._EN_BOOST)
            hedges = self._count_in_set(toks, self._KO_HEDGE | self._EN_HEDGE)
            negs = self._count_in_set(toks, self._KO_NEG | self._EN_NEG)

        temporal = self._count_in_set(toks, self._TEMPORAL) if self.cfg.include_temporal else 0

        laugh_cnt = 0
        if self.cfg.include_booster_hedge:
            laugh_cnt += self._count_laughter(text)

        emo_emoji = {"희": 0, "노": 0, "애": 0, "락": 0}
        emoji_total = 0
        if self.cfg.include_emoji:
            emoji_total, emo_emoji = self._count_emoji(text)

        emo_pos = emo_neg = emo_neu = 0
        if self.cfg.include_emoticons:
            emo_pos = len(self._EMOTICON_POS.findall(text))
            emo_neg = len(self._EMOTICON_NEG.findall(text))
            emo_neu = len(self._EMOTICON_NEU.findall(text))

        uniq = len(set(toks)) if T else 0
        avg_tok_len = (sum(len(w) for w in toks) / T) if T else 0.0

        # ---- 길이 정규화 스칼라 (스코어러 직접 전달용) ----
        nm = (norm_method or self.cfg.default_norm_method or "auto").lower()
        len_norm_tokens = self._len_norm(T, nm)  # 권장: 토큰 수 기반
        len_norm_chars = self._len_norm(L, nm)  # 참고: 문자 수 기반(디버그용)

        # ---- (옵션) 키워드 중복 제거 및 패턴 가중 합 ----
        kw_summary = {"uniq_total": 0, "by_emotion": {}, "weighted_by_emotion": {}, "weighted_total": 0.0}
        if kw_hits:
            uniq_total = 0
            weighted_total = 0.0
            by_emotion: Dict[str, int] = {}
            weighted_by_emotion: Dict[str, float] = {}

            for eid, lst in kw_hits.items():
                uniq_set = {str(x).strip().lower() for x in (lst or []) if x}
                by_emotion[eid] = len(uniq_set)
                uniq_total += by_emotion[eid]

                # 패턴 가중치 합(제공된 경우)
                if pattern_weights and eid in pattern_weights:
                    weights = pattern_weights[eid] or {}
                    wsum = 0.0
                    for tok in uniq_set:
                        # 키워드 표준화 방식은 KeywordMatcher._normalize_token과 유사하게 소문자 기준
                        wsum += float(weights.get(tok, 0.0))
                    weighted_by_emotion[eid] = wsum
                    weighted_total += wsum

            kw_summary["uniq_total"] = uniq_total
            kw_summary["by_emotion"] = by_emotion
            if weighted_by_emotion:
                kw_summary["weighted_by_emotion"] = {k: round(v, 6) for k, v in weighted_by_emotion.items()}
                kw_summary["weighted_total"] = round(weighted_total, 6)

        # ---- 파생 지표 ----
        exclam_norm = min(1.0, exclam / 3.0)
        ques_norm = min(1.0, ques / 3.0)
        ell_norm = min(1.0, ell / 2.0)
        repeat_norm = min(1.0, repeat_runs / 2.0)
        caps_ratio = (upper_chars / max(1, upper_chars + lower_chars)) if self.cfg.include_case_ratio else 0.0
        hangul_ratio = (hangul_chars / max(1, L)) if self.cfg.include_script_ratio else 0.0
        latin_ratio = (latin_chars / max(1, L)) if self.cfg.include_script_ratio else 0.0

        booster_norm = min(1.0, boosters / 3.0)
        neg_norm = min(1.0, negs / 3.0)
        hedge_norm = min(1.0, hedges / 3.0)
        laugh_norm = min(1.0, laugh_cnt / 3.0)
        emoji_h_norm = min(1.0, emo_emoji["희"] / 4.0)
        emoji_a_norm = min(1.0, emo_emoji["노"] / 4.0)
        emoji_s_norm = min(1.0, emo_emoji["애"] / 4.0)
        emoji_f_norm = min(1.0, emo_emoji["락"] / 4.0)

        intensity = self._clip01(
            0.30 * exclam_norm +
            0.20 * booster_norm +
            0.15 * repeat_norm +
            0.15 * caps_ratio +
            0.10 * emoji_h_norm +
            0.10 * emoji_f_norm
        )
        uncertainty = self._clip01(
            0.35 * ques_norm +
            0.25 * ell_norm +
            0.25 * hedge_norm +
            0.15 * (1.0 if "?" in text and "!" in text else 0.0)
        )
        negativity_hint = self._clip01(
            0.45 * neg_norm +
            0.25 * emoji_a_norm +
            0.20 * (emo_neg / max(1.0, emo_pos + emo_neg + emo_neu)) +
            0.10 * (1.0 if ("않" in text or "없" in text) else 0.0)
        )
        positivity_hint = self._clip01(
            0.35 * laugh_norm +
            0.30 * emoji_h_norm +
            0.20 * emoji_f_norm +
            0.15 * max(0.0, exclam_norm - neg_norm * 0.5)
        )

        feature_vector: List[float] = [
            min(1.0, max(1, L) / self.cfg.max_len_clip),
            min(1.0, max(1, T) / self.cfg.max_tokens_clip),
            exclam_norm, ques_norm, ell_norm,
            repeat_norm, caps_ratio, hangul_ratio, latin_ratio,
            booster_norm, neg_norm, hedge_norm, laugh_norm,
            emoji_h_norm, emoji_a_norm, emoji_s_norm, emoji_f_norm,
            intensity, uncertainty, negativity_hint, positivity_hint,
        ]

        return {
            "basic": {
                "len_chars": L,
                "len_tokens": T,
                "uniq_tokens": uniq,
                "avg_token_len": round(avg_tok_len, 4),
                "lines": lines,
                "sentences_est": sentences_est,
            },
            "punct": {
                "exclam": exclam,
                "question": ques,
                "ellipsis": ell,
                "quotes": quotes,
                "parens": parens,
                "comma": comma,
                "period": period,
                "dash": dash,
                "semicolon": semi,
                "colon": colon,
            },
            "lexical": {
                "stopword_count": stop_cnt,
                "stopword_ratio": round(stop_cnt / max(1, T), 4) if self.cfg.include_stopword_ratio else 0.0,
                "uppercase_tokens": upper_tokens,
                "uppercase_chars": upper_chars,
                "lowercase_chars": lower_chars,
                "digit_chars": digit_chars,
                "type_token_ratio": round(uniq / max(1, T), 4) if T else 0.0,
            },
            "script": {
                "hangul_chars": hangul_chars,
                "latin_chars": latin_chars,
                "other_chars": other_chars,
                "hangul_ratio": round(hangul_ratio, 4),
                "latin_ratio": round(latin_ratio, 4),
            },
            "cues": {
                "boosters": boosters,
                "hedges": hedges,
                "negations": negs,
                "temporal": temporal,
                "josa": josa_cnt,
                "laughter": laugh_cnt,
                "repetition_runs": repeat_runs,
                "repetition_longest": repeat_longest,
                "emoticon_pos": emo_pos,
                "emoticon_neg": emo_neg,
                "emoticon_neu": emo_neu,
            },
            "emoji": {
                "total": emoji_total,
                "by_axis": dict(emo_emoji),
            },
            "derived": {
                "intensity": round(intensity, 6),
                "uncertainty": round(uncertainty, 6),
                "negativity_hint": round(negativity_hint, 6),
                "positivity_hint": round(positivity_hint, 6),
            },
            "feature_vector": [round(x, 6) for x in feature_vector],

            # === 추가 전달 항목: 길이 정규화 & 키워드 요약 ===
            "norm": {
                "method": nm,
                "len_norm_tokens": float(round(len_norm_tokens, 6)),
                "len_norm_chars": float(round(len_norm_chars, 6)),
            },
            # kw_hits가 주어지면, 세그먼트 차원에서 중복 제거된 유니크 키워드 개수/가중합 제공
            "kw_summary": kw_summary,
        }

    def extract_batch(
            self,
            segments: List[str],
            *,
            norm_method: Optional[str] = None,
            kw_hits_batch: Optional[List[Optional[Dict[str, List[str]]]]] = None,
            pattern_weights: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> List[Dict[str, Any]]:
        """배치 추출. kw_hits_batch가 주어지면 각 세그먼트에 매칭되는 kw_hits를 함께 전달."""
        out = []
        kw_hits_batch = kw_hits_batch or [None] * len(segments)
        for seg, kwh in zip(segments, kw_hits_batch):
            out.append(self.extract(seg, norm_method=norm_method, kw_hits=kwh, pattern_weights=pattern_weights))
        return out

    # --------------------------------------------------------------------- #
    # Helpers
    # --------------------------------------------------------------------- #
    def _tokenize(self, text: str) -> List[str]:
        return [m.group(0) for m in self._RE_TOK.finditer(text)]

    def _count_josa(self, toks: List[str]) -> int:
        return sum(1 for t in toks if t in self._KO_JOSA)

    def _count_stopwords(self, toks: List[str]) -> int:
        c = 0
        en_low = self._EN_STOP
        for t in toks:
            tl = t.lower()
            if t in self._KO_STOP or tl in en_low:
                c += 1
        return c

    def _count_in_set(self, toks: List[str], vocab: set) -> int:
        v_low = {x.lower() for x in vocab}
        return sum(1 for t in toks if (t in vocab) or (t.lower() in v_low))

    def _count_laughter(self, text: str) -> int:
        cnt = 0
        for token in self._KO_LAUGHTER:
            cnt += text.count(token)
        for token in self._EN_LAUGHTER:
            cnt += len(re.findall(rf"\b{re.escape(token)}\b", text, flags=re.I))
        return cnt

    def _count_emoji(self, text: str) -> Tuple[int, Dict[str, int]]:
        total = 0
        by_axis = {"희": 0, "노": 0, "애": 0, "락": 0}
        for ch in text:
            matched = False
            for axis, s in self._EMOJI_MAP.items():
                if ch in s:
                    by_axis[axis] += 1
                    matched = True
                    break
            if matched:
                total += 1
        return total, by_axis

    def _len_norm(self, length: int, method: str) -> float:
        """길이 정규화 스칼라."""
        m = (method or "auto").lower()
        L = max(1, int(length))
        if m == "none":
            return 1.0
        if m == "sqrt":
            return math.sqrt(L)
        if m == "log":
            return math.log(L + 2.0)
        # auto → cfg의 임계로 결정 (B-1 개선사항 적용)
        small = int(getattr(self.cfg, "ln_auto_small", 10))
        medium = int(getattr(self.cfg, "ln_auto_medium", 50))
        if L < small:
            return 1.0
        if L < medium:
            return math.sqrt(L)
        return math.log(L + 2.0)

    def _clip01(self, x: float) -> float:
        return 0.0 if x <= 0.0 else (1.0 if x >= 1.0 else x)


# =============================================================================
# EmotionScorer
# =============================================================================
class EmotionScorer:
    """
    키워드 기반 점수 + (키워드 무적중 시) 한국어 휴리스틱 폴백 신호.
    adaptive_mode=True일 때:
      - TF–IDF 스코어링: tf = count/log(V_e+2), idf ≈ log((V̄+1)/(V_e+1))+1
      - 길이 정규화(len_norm): {1, sqrt(L), log(L+2)} or FeatureExtractor의 len_norm_tokens
      - 대분류 균형 보정: S_m ← min(S_m * [(1-λ) + λ * (S̄/S_m)], clip)
      - 동적 임계치: min_signal ≈ c / √(N/30), N=세부감정 총량
      - 서브감정 선택: 누적 확률 0.8 기준(결정론적), 정보 보존 중심
    """

    def __init__(
            self,
            cfg: "AnalyzerConfig",
            rules: Dict[str, "EmotionRule"],
            feature_extractor: Optional["FeatureExtractor"] = None,
    ):
        self.cfg = cfg
        self.rules = rules or {}
        self.fe = feature_extractor

        # ---- 간소화 Config(3종) + 내부 기본값(하위호환) ----
        self.adaptive = getattr(cfg, "adaptive_mode", True)
        self.balance_strength = float(getattr(cfg, "balance_strength", 0.7))
        self.norm_method = getattr(cfg, "norm_method", "auto")

        # 내부 기본값(코드 내부에서만 사용; 0) 공통 설계와 일치)
        self._clip_score = float(getattr(cfg, "clip_score", 6.0))
        self._score_min_floor = float(getattr(cfg, "score_min_floor", 0.01))
        self._score_max_cap = float(getattr(cfg, "score_max_cap", 0.99))
        self._auto_temperature = bool(getattr(cfg, "auto_temperature", True))
        self._auto_min_signal = bool(getattr(cfg, "auto_min_signal", True))

        # 구버전 호환 필드
        self._legacy_temperature = float(getattr(cfg, "score_temperature", 0.75))
        self._legacy_min_signal = float(getattr(cfg, "min_signal_threshold", 0.05))
        self._seed = int(getattr(cfg, "seed", 1337))

        # 어휘 통계: V_e, V̄, 세부감정 총량 N
        self._vocab_sizes: Dict[str, int] = {}
        self._avg_vocab_size: float = 1.0
        self._total_sub_count: int = 0
        self._init_vocab_stats()

        # 폴백용 간단 어휘
        self.fallback = {
            "희": ["기쁘", "좋아", "다행", "안도", "설레", "들뜨", "행복", "반가", "뿌듯"],
            "노": ["짜증", "화가", "분노", "열받", "억울", "분개", "거슬리", "빡치"],
            "애": ["슬픔", "우울", "서글프", "상실", "허무", "서운", "눈물", "쓸쓸"],
            "락": ["즐거", "재밌", "여유", "편안", "해방감", "흥분", "신나", "홀가분"],
            "_worry": ["불안", "초조", "걱정", "근심", "긴장"],
        }

        # '락' 보조 단서(문장 일부 매칭)
        self._rak_cues = [
            "안도", "안도감", "편안", "여유", "평온",
            "후련", "해방감", "휴식", "긴장이 풀", "한숨 돌리",
        ]

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def score_segment(
            self,
            segment: str,
            kw_counts: Dict[str, float],
            kw_hits: Dict[str, List[str]],
    ) -> List[Dict[str, Any]]:
        """
        Args:
          segment: 세그먼트 원문
          kw_counts: 감정ID -> (가중 TF 포함 가능) 카운트
          kw_hits:   감정ID -> 유니크 키워드 텍스트 목록
        Returns:
          [{"emotion_id","score","confidence","keywords","sub_emotions","context"}, ...]
        """
        if not self.adaptive:
            return self._legacy_score(segment, kw_counts, kw_hits)

        # 1) 길이 정규화 스칼라(len_norm) 계산
        len_norm = self._len_norm_from_fe(segment)

        # 2) TF–IDF 기반 원시 점수 S_e 계산
        scores_raw: Dict[str, float] = {}
        for eid in EMOTION_AXES:
            c = float(kw_counts.get(eid, 0.0))
            if c <= 0.0:
                scores_raw[eid] = 0.0
                continue
            V_e = max(1, self._vocab_sizes.get(eid, 1))
            V_avg = max(1.0, self._avg_vocab_size)
            tf = c / max(1.0, math.log(V_e + 2.0))
            idf = math.log((V_avg + 1.0) / (V_e + 1.0)) + 1.0
            scores_raw[eid] = (tf * idf) / max(1e-8, len_norm)

        # 3) 모든 값이 0이면 폴백 신호
        if sum(scores_raw.values()) <= 0.0:
            fb = self._fallback_signal(segment)
            scores_raw = {e: fb.get(e, 0.0) for e in EMOTION_AXES}

        # '락' 보조 부스팅(안도/편안/여유/후련/해방감 등 맥락에서 약한 락을 소폭 가산)
        try:
            scores_raw = self._soft_relaxation_boost(scores_raw, segment, kw_hits)
        except Exception:
            pass

        # 4) 대분류 균형 보정 + 클리핑
        scores_bal = self._apply_balance(scores_raw, lam=self.balance_strength, clip=self._clip_score)

        # 5) Softmax 확률화(온도: 감정 수 기반 자동/수동)
        temperature = self._get_temperature()
        probs = self._softmax_temperature([scores_bal[e] for e in EMOTION_AXES], temperature)
        prob_map = {eid: p for eid, p in zip(EMOTION_AXES, probs)}

        # 6) 동적 임계치 결정
        min_signal = self._get_min_signal()

        # 7) 결과 구성 + 서브감정 선택(누적확률 0.8, 결정론적)
        out: List[Dict[str, Any]] = []
        for eid in EMOTION_AXES:
            p = float(prob_map.get(eid, 0.0))
            if p < min_signal:
                continue
            subs = self._select_sub_emotions_cumprob(eid, p, kw_hits.get(eid, []))
            out.append({
                "emotion_id": eid,
                "score": float(max(self._score_min_floor, min(self._score_max_cap, p))),
                "confidence": 0.0,  # Calibrator 단계에서 채움
                "keywords": kw_hits.get(eid, [])[:64],
                "sub_emotions": subs,  # [{"name":..., "score":...}, ...]
                "context": {"situations": [], "relationships": [], "environment": [], "triggers": [], "modifiers": []},
            })
        return out

    def get_min_signal(self) -> float:
        return self._get_min_signal()

    # ------------------------------------------------------------------ #
    # Legacy path (하위호환)
    # ------------------------------------------------------------------ #
    def _legacy_score(
            self,
            segment: str,
            kw_counts: Dict[str, float],
            kw_hits: Dict[str, List[str]],
    ) -> List[Dict[str, Any]]:
        base = [float(kw_counts.get(e, 0.0)) for e in EMOTION_AXES]
        if sum(base) <= 0.0:
            fb = self._fallback_signal(segment)
            base = [fb.get(e, 0.0) for e in EMOTION_AXES]
        if sum(base) <= 0.0:
            # 균등 탈출(결정론적 상수)
            base = [0.001 + (i * 0.0003) for i in range(len(EMOTION_AXES))]

        probs = self._softmax_temperature(base, self._legacy_temperature)
        min_signal = getattr(self.cfg, "min_signal_threshold", self._legacy_min_signal)

        out: List[Dict[str, Any]] = []
        for eid, p in zip(EMOTION_AXES, probs):
            if p < min_signal:
                continue
            subs = self._select_sub_emotions_cumprob(eid, p, kw_hits.get(eid, []))
            out.append({
                "emotion_id": eid,
                "score": float(max(self._score_min_floor, min(self._score_max_cap, p))),
                "confidence": 0.0,
                "keywords": kw_hits.get(eid, [])[:64],
                "sub_emotions": subs,
                "context": {"situations": [], "relationships": [], "environment": [], "triggers": [], "modifiers": []},
            })
        return out

    # ------------------------------------------------------------------ #
    # Helpers: stats / normalization / balance
    # ------------------------------------------------------------------ #
    def _init_vocab_stats(self) -> None:
        sizes = []
        sub_total = 0
        for eid, rule in (self.rules or {}).items():
            kw = len(getattr(rule, "keywords", []) or [])
            subs = len(getattr(rule, "sub_emotions", []) or [])
            size = kw + subs  # 키워드+서브감정 크기를 어휘량 근사치로 사용
            self._vocab_sizes[eid] = max(1, size)
            sizes.append(max(1, size))
            sub_total += subs
        self._avg_vocab_size = (sum(sizes) / len(sizes)) if sizes else 1.0
        self._total_sub_count = int(sub_total)

    def _len_norm_from_fe(self, segment: str) -> float:
        # FeatureExtractor가 있으면 그 값을 우선 사용
        try:
            if self.fe is not None:
                fe_res = self.fe.extract(segment, norm_method=self.norm_method)
                ln = float(fe_res.get("norm", {}).get("len_norm_tokens", 0.0) or 0.0)
                if ln > 0.0:
                    return ln
        except Exception:
            pass
        # 폴백: 간단 토큰 카운트 기반
        T = self._simple_token_count(segment)
        return self._len_norm(T, self.norm_method)

    def _apply_balance(self, scores: Dict[str, float], lam: float = 0.7, clip: float = 6.0) -> Dict[str, float]:
        vals = [v for v in scores.values() if v > 0]
        mean_v = (sum(vals) / len(vals)) if vals else 0.0
        out: Dict[str, float] = {}
        for eid, s in scores.items():
            if s <= 0.0:
                out[eid] = 0.0
                continue
            # bal(m) = (1-λ) + λ * (S̄ / S_m)
            bal = (1.0 - lam) + lam * (mean_v / max(1e-8, s)) if mean_v > 0 else 1.0
            out[eid] = min(s * bal, clip)
        return out

    def _get_temperature(self) -> float:
        if not self._auto_temperature:
            return self._legacy_temperature
        # 감정 수 기반 동적 온도
        n = max(1, len([e for e in EMOTION_AXES]))
        if n <= 4:
            return 0.75
        if n <= 10:
            return 0.90
        # sqrt 스케일, 상한 1.2
        return min(1.2, 0.75 * math.sqrt(n / 4.0))

    def _get_min_signal(self) -> float:
        if not self._auto_min_signal:
            return self._legacy_min_signal
        N = max(1, self._total_sub_count)  # 세부감정 총량
        c = 0.10  # 기본 상수
        val = c / math.sqrt(max(1.0, N / 30.0))
        # 안정적 범위로 클램프
        return float(min(0.15, max(0.02, val)))

    # ------------------------------------------------------------------ #
    # Helpers: sub-emotion selection (cumulative prob 0.8)
    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    # Helpers: sub-emotion selection (cumulative prob 0.8)
    # ------------------------------------------------------------------ #
    def _select_sub_emotions_cumprob(self, eid: str, parent_prob: float, kw_e_hits: List[str]) -> List[Dict[str, Any]]:
        """
        [개선됨] 누적 확률 0.8에 도달할 때까지 서브감정을 선택합니다.
        - 모든 서브감정에 초기에는 동일한 가중치를 부여합니다.
        - 랜덤 샘플링이나 복잡한 해시 없이 결정론적으로 동작하여 재현성을 보장합니다.
        """
        rule = self.rules.get(eid)
        subs = list(getattr(rule, "sub_emotions", []) or [])
        if not subs:
            return []

        # 1. 모든 서브감정 후보에 기본 가중치 1.0을 부여합니다.
        base_scores = []
        for name in subs:
            name = str(name).strip()
            if not name:
                continue
            # 향후 패턴 가중치 등을 적용할 수 있는 확장 지점입니다. (현재는 균등)
            base_scores.append({"name": name, "p": 1.0})

        # 2. 가중치를 정규화하여 전체 합이 1이 되도록 합니다.
        total_w = sum(x["p"] for x in base_scores) or 1.0
        for x in base_scores:
            x["p"] = x["p"] / total_w

        # 3. 정규화된 가중치(확률)를 기준으로 내림차순 정렬합니다.
        base_scores.sort(key=lambda x: x["p"], reverse=True)

        # 4. 누적 확률이 0.8이 될 때까지 순서대로 서브감정을 선택합니다.
        selected_subs, cumulative_prob = [], 0.0
        threshold = 0.80
        for x in base_scores:
            # 최종 점수는 부모 감정의 확률에 서브감정의 정규화된 가중치를 곱하여 계산합니다.
            score = float(round(parent_prob * x["p"], 6))
            selected_subs.append({"name": x["name"], "score": score})

            cumulative_prob += x["p"]
            if cumulative_prob >= threshold:
                break

        return selected_subs

    # ------------------------------------------------------------------ #
    # Helpers: token/len norm/softmax/fallback
    # ------------------------------------------------------------------ #
    def _simple_token_count(self, s: str) -> int:
        # 간단 토큰 카운트(한글/영문/숫자/기타 단독문자)
        return sum(1 for _ in re.finditer(r"[가-힣]+|[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:[.,]\d+)*|[^\s\w]", s or ""))

    def _len_norm(self, L: int, method: str) -> float:
        m = (method or "auto").lower()
        L = max(1, int(L))
        if m == "none":
            return 1.0
        if m == "sqrt":
            return math.sqrt(L)
        if m == "log":
            return math.log(L + 2.0)
        # auto
        if L < 10:
            return 1.0
        if L < 50:
            return math.sqrt(L)
        return math.log(L + 2.0)

    def _softmax_temperature(self, vec: List[float], temperature: float) -> List[float]:
        # 외부 util이 있으면 그걸, 없으면 내부 구현 사용
        try:
            fn = globals().get("softmax_temperature", None)
            if callable(fn):
                return list(fn(vec, temperature=temperature))
        except Exception:
            pass
        t = max(1e-6, float(temperature))
        m = max(vec) if vec else 0.0
        exps = [math.exp((v - m) / t) for v in (vec or [])]
        Z = sum(exps) or 1.0
        return [x / Z for x in exps]

    def _fallback_signal(self, text: str) -> Dict[str, float]:
        t = unicodedata.normalize("NFKC", text or "")
        sig = {e: 0.0 for e in EMOTION_AXES}

        def bump(keys: List[str], w: float, targets: List[str]):
            c = sum(t.count(k) for k in keys)
            if c > 0:
                for e in targets:
                    sig[e] += w * c

        bump(self.fallback["희"], 1.0, ["희", "락"])
        bump(self.fallback["락"], 0.8, ["락", "희"])
        bump(self.fallback["노"], 1.0, ["노"])
        bump(self.fallback["애"], 1.0, ["애"])

        wc = sum(t.count(k) for k in self.fallback["_worry"])
        if wc > 0:
            sig["노"] += 0.6 * wc
            sig["애"] += 0.4 * wc

        ex = t.count("!") + t.count("!!")
        qn = t.count("?")
        el = t.count("…") + t.count("...")

        if ex > 0:
            sig["희"] += 0.3 * ex
            sig["락"] += 0.2 * ex
        if qn > 0:
            sig["노"] += 0.15 * qn
            sig["애"] += 0.15 * qn
        if el > 0:
            sig["애"] += 0.2 * el

        return sig

    # --- 보조: '락' 소프트 부스팅 ---
    def _soft_relaxation_boost(self,
                               scores_raw: Dict[str, float],
                               segment: str,
                               kw_hits: Dict[str, List[str]]) -> Dict[str, float]:
        """
        '락'이 임계 미만이고 '희'가 상대적으로 강하며, 문장에 '안도/편안/여유...' 단서가 있으면
        '희' 점수의 일부를 '락' 쪽으로 이동(소량 가산)합니다.
        - 라벨 파일 수정 없음, 키워드 카운트/임베딩과 충돌 없음.
        """
        try:
            t = unicodedata.normalize("NFKC", segment or "")
        except Exception:
            t = segment or ""

        s_hee = float(scores_raw.get("희", 0.0))
        s_rak = float(scores_raw.get("락", 0.0))

        # 이미 락이 충분하거나 희가 약하면 종료
        if s_rak >= s_hee * 0.8 or s_hee <= 1e-8:
            return scores_raw

        cue_hits = sum(1 for c in self._rak_cues if c in t)
        if cue_hits <= 0:
            return scores_raw

        # 희가 충분히 있는데 락이 약한 경우만 보정
        boost_ratio = min(0.12, 0.05 * cue_hits)   # 1~2개=5~10%, 3개+=최대 12%
        boost = s_hee * boost_ratio

        if boost <= 0.0:
            return scores_raw

        # 보정 적용(총합 보존까지 원한다면 희에서 일부 감산 옵션을 고려)
        scores_raw["락"] = s_rak + boost
        # scores_raw["희"] = max(0.0, s_hee - 0.5 * boost)  # 선택적 역감산

        return scores_raw

    # ------------------------------------------------------------------ #
    # Helpers: deterministic weight for sub-emotions (no randomness)
    # ------------------------------------------------------------------ #
    def _normalize_str(self, s: str) -> str:
        x = unicodedata.normalize("NFKC", s or "").strip().lower()
        return re.sub(r"[^\w\s\uAC00-\uD7A3]", "", x)

    def _stable_weight(self, key: str) -> float:
        """
        문자열 기반 결정론 가중치(0.5~1.5). 랜덤/샘플링 없이 재현성 보장.
        """
        import hashlib
        h = hashlib.md5((key + f"|{self._seed}").encode("utf-8")).hexdigest()
        # 0..1 → 0.5..1.5
        v = int(h[:8], 16) / 0xFFFFFFFF
        return 0.5 + v  # [0.5, 1.5]


# =============================================================================
# EvidenceAggregator (finalized)
# =============================================================================
class EvidenceAggregator:
    """세그먼트 증거를 문서 레벨로 집계·균형 보정·정규화(softmax)까지 수행.
    - 세그먼트별 기여도(w*score)를 누적 비중 0.8까지 합산해 문서 원시 점수를 산출
    - 대분류 균형 보정 bal(m) = (1-λ) + λ*(S̄/S_m) 후 softmax 정규화
    - 자동 temperature 산정(클래스 수 기반) + '실사용값/출처' 기록
    """

    def __init__(self, cfg: "AnalyzerConfig"):
        self.cfg = cfg

        # ── Config (간단 3종 + 하위호환) ─────────────────────────────────
        self.balance_strength = float(getattr(cfg, "balance_strength", 0.7))  # λ
        self._clip_score = float(getattr(cfg, "clip_score", 6.0))
        self._auto_temperature_flag = bool(getattr(cfg, "auto_temperature", True))
        self._legacy_temperature = float(getattr(cfg, "score_temperature", 0.75))
        self._score_min_floor = float(getattr(cfg, "score_min_floor", 0.01))
        self._score_max_cap = float(getattr(cfg, "score_max_cap", 0.99))

        # ── 세그 가중/휴리스틱 파라미터 (현행 유지) ───────────────────────
        self._FREQ_GATE_MIN = 0.65
        self._FREQ_GATE_MAX = 1.00
        self._RECENCY_BOOST = 0.15
        self._PUNCT_MAX = 1.50

        # ── 출력 상한 ────────────────────────────────────────────────────
        self._KW_TOPK = 64

        # ── softmax에 '실제 사용된' temperature 및 출처 기록 ─────────────
        self._last_used_temperature: Optional[float] = None
        self._last_temperature_source: Optional[str] = None  # "config" | "auto" | "fallback"

    # ------------------------------------------------------------------ #
    # Public
    # ------------------------------------------------------------------ #
    def aggregate_document(self, seg_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        세그먼트별 감정 기여도(w*score)를 누적 비중 0.8까지만 합산 → 균형 보정(상황 적응형 λ) → softmax.
        softmax 호출 '직전'에 실제 temperature를 self._last_used_temperature에 기록합니다.
        """
        n_seg = max(1, len(seg_results))

        # --- 누적 버킷(키워드/서브감정/신뢰/기여도) ----------------------
        acc: Dict[str, Dict[str, Any]] = {
            e: {
                "wconf": 0.0,
                "seg_hits": 0,
                "kw_set": set(),
                "sub_counter": Counter(),
                "first_idx": None,
                "last_idx": None,
                "score_contributions": [],  # 세그먼트별 기여도(w * score)
            }
            for e in EMOTION_AXES
        }

        # --- 1) 세그먼트별 기여도 및 부가정보 수집 -----------------------
        for i, seg in enumerate(seg_results):
            w = self._segment_weight(seg, i, n_seg)
            emo_list = seg.get("detected_emotions") or []
            present_eids = set()

            for emo in emo_list:
                eid = emo.get("emotion_id")
                if not eid or eid not in acc:
                    continue

                score = float(emo.get("score", 0.0))
                if score > 0.0:
                    acc[eid]["score_contributions"].append(w * score)

                conf = float(emo.get("confidence", 0.0))
                kws = emo.get("keywords") or []
                subs = emo.get("sub_emotions") or []

                acc[eid]["wconf"] += w * conf
                acc[eid]["kw_set"].update(kws)
                for se in subs:
                    n = se.get("name")
                    s = float(se.get("score", 0.0))
                    if n:
                        acc[eid]["sub_counter"][n] += w * s

                if acc[eid]["first_idx"] is None:
                    acc[eid]["first_idx"] = i
                acc[eid]["last_idx"] = i
                present_eids.add(eid)

            for eid in present_eids:
                acc[eid]["seg_hits"] += 1

        # --- 2) 누적 비중 0.8까지 합산해 원시 점수 계산 -------------------
        sum_map: Dict[str, float] = {}
        for eid in EMOTION_AXES:
            contributions = sorted(acc[eid]["score_contributions"], reverse=True)
            total_sum = sum(contributions)
            if total_sum <= 0:
                sum_map[eid] = 0.0
                continue

            cum = 0.0
            cur = 0.0
            for x in contributions:
                if (cum / total_sum) >= 0.80:
                    break
                cur += x
                cum += x
            sum_map[eid] = cur

        raw_scores = [sum_map.get(e, 0.0) for e in EMOTION_AXES]

        # --- 3) 결과 객체(문서 레벨) 뼈대 구성 ---------------------------
        detected: List[Dict[str, Any]] = []
        for eid in EMOTION_AXES:
            raw = float(sum_map.get(eid, 0.0))

            # 신뢰도 평균 계산 가중치(세그 재스캔)
            total_w_for_conf = 0.0
            for i, seg in enumerate(seg_results):
                if any(emo.get("emotion_id") == eid for emo in seg.get("detected_emotions", [])):
                    total_w_for_conf += self._segment_weight(seg, i, n_seg)

            spread = acc[eid]["seg_hits"] / float(n_seg)
            sub_shares = self._select_subs_cumprob(acc[eid]["sub_counter"])
            kws_sorted = sorted(list(acc[eid]["kw_set"]))[: self._KW_TOPK]
            conf = self._document_level_confidence(
                eid=eid,
                avg_conf=(acc[eid]["wconf"] / total_w_for_conf) if total_w_for_conf > 0 else 0.0,
                kw_count=len(kws_sorted),
                sub_count=len(acc[eid]["sub_counter"]),
                spread=spread,
            )
            context = {"triggers": kws_sorted[: int(getattr(self.cfg, "context_top_k", 5) or 5)]}

            detected.append({
                "emotion_id": eid,
                "score": raw,  # 후단 softmax로 정규화됨
                "confidence": conf,
                "keywords": kws_sorted,
                "sub_emotions": [{"name": n, "share": round(s, 6)} for n, s in sub_shares],
                "context": context,
                "span": {
                    "first_index": acc[eid]["first_idx"],
                    "last_index": acc[eid]["last_idx"],
                    "coverage_ratio": round(spread, 6),
                },
            })

        # --- 4) 균형 보정(상황 적응형 λ) → softmax(자동 온도) -------------
        if any(s > 0 for s in raw_scores):
            # (추가) 원시 점수 산포 기반 λ 적응
            vals = [v for v in raw_scores if v > 0]
            std = float(np.std(vals)) if vals else 0.0
            base_lam = float(getattr(self.cfg, "balance_strength", 0.7))
            # std가 낮을수록(=너무 평평) λ 줄여 샤프닝, 범위 [0.30, base_lam]
            lam = max(0.30, min(base_lam, base_lam - 0.5 * max(0.0, 0.20 - std)))
            # 기록(선택): 실제 사용 λ
            try:
                self._last_balance_lambda = float(lam)
            except Exception:
                pass

            balanced = self._apply_balance(raw_scores, lam=lam, clip=self._clip_score)

            # 감지된 양수 스코어 개수 K
            K = len([s for s in balanced if s > 0])

            # 설정값 우선 → 자동 온도 → 폴백 순으로 temperature 결정
            cfg_tau = getattr(self.cfg, "score_temperature", None)
            if isinstance(cfg_tau, (int, float)):
                tau = float(cfg_tau)
                src = "config"
            elif self._auto_temperature_flag:
                tau = self._auto_temperature(K)
                src = "auto"
            else:
                tau = 0.75
                src = "fallback"

            # [중요] 실제 사용 온도/출처 기록 (오케스트레이터에서 읽어감)
            self._last_used_temperature = float(tau)
            self._last_temperature_source = src
            self._last_temperature = self._last_used_temperature  # backward compat

            # softmax 적용(외부 util 우선)
            try:
                fn = globals().get("softmax_temperature", None)
                if callable(fn):
                    final_probs = list(fn(balanced, temperature=tau))
                else:
                    raise RuntimeError
            except Exception:
                final_probs = self._softmax_temperature(balanced, temperature=tau)
        else:
            final_probs = [0.0] * len(raw_scores)

        # --- 5) 최종 점수/서브감정 점수 반영(캡/플로어) -------------------
        for d, s in zip(detected, final_probs):
            s_cap = float(max(self._score_min_floor, min(self._score_max_cap, s))) if any(final_probs) else 0.0

            # 서브감정 share → 최종 score로 분배
            subs = d.get("sub_emotions") or []
            if subs:
                tot_share = sum(x.get("share", 0.0) for x in subs) or 1.0
                d["sub_emotions"] = [
                    {"name": x["name"], "score": round(s_cap * (x["share"] / tot_share), 6)}
                    for x in subs
                ]
            d["score"] = s_cap

        return detected

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _segment_weight(self, seg: Dict[str, Any], idx: int, n_seg: int) -> float:
        """길이/문장부호/최근성 기반 세그 가중치."""
        t = seg.get("text") or ""
        L = max(1, len(t))
        len_norm = max(0.25, min(2.0, L / 120.0)) ** 0.5
        ex = t.count("!")
        qn = t.count("?")
        punct = min(self._PUNCT_MAX, 1.0 + 0.10 * ex + 0.05 * qn)
        recency = 1.0 + (self._RECENCY_BOOST * (idx / max(1, n_seg - 1))) if n_seg > 1 else 1.0
        return len_norm * punct * recency

    def _document_level_confidence(
            self,
            eid: str,
            avg_conf: float,
            kw_count: int,
            sub_count: int,
            spread: float,
    ) -> float:
        """라벨 신뢰도 + 키워드/서브감정 다양성 + 분포 신호를 혼합."""
        kw_strength = min(1.0, kw_count / 10.0)
        sub_div = min(1.0, sub_count / 10.0)
        spread_sig = min(1.0, math.sqrt(max(0.0, spread)))
        conf = 0.50 * avg_conf + 0.25 * kw_strength + 0.15 * sub_div + 0.10 * spread_sig
        return round(max(0.0, min(1.0, conf)), 6)

    # EvidenceAggregator._select_subs_cumprob(...) 교체

    def _select_subs_cumprob(self, sub_counter: Counter, threshold: float = 0.80) -> List[Tuple[str, float]]:
        if not sub_counter: return []
        # 기본값
        items = [(n, float(v)) for n, v in sub_counter.items() if v > 0]
        if not items: return []
        # (추가) 미세 가중: 강도/길이/빈도 기반
        # 길이가 긴 서브명이나 반복 등장(값 큰 것)에 소폭 보정
        adj = []
        for n, v in items:
            boost = 1.0 + 0.05 * min(10, len(n))  # 이름 길이 보너스(최대 +50%)
            adj.append((n, v * boost))
        items = adj
        items.sort(key=lambda x: x[1], reverse=True)
        total = sum(v for _, v in items) or 1.0

        out, cum = [], 0.0
        for name, val in items:
            share = val / total
            out.append((name, share))
            cum += share
            if cum >= threshold:
                break
        return out

    def _apply_balance(self, scores: List[float], lam: float = 0.7, clip: float = 6.0) -> List[float]:
        vals = [v for v in scores if v > 0.0]
        mean_v = (sum(vals) / len(vals)) if vals else 0.0
        out: List[float] = []
        for s in scores:
            if s <= 0.0:
                out.append(0.0)
                continue
            bal = (1.0 - lam) + lam * (mean_v / max(1e-8, s)) if mean_v > 0 else 1.0
            out.append(min(s * bal, clip))
        return out

    def _auto_temperature(self, k: int) -> float:
        k = max(1, int(k))
        # 이전: if k <= 4: return 0.90
        if k <= 4:
            return 0.70  # 4클래스는 조금 더 날카롭게
        if k <= 10:
            return 0.95 + 0.03 * (k - 4)  # 5:0.98 ~ 10:1.13
        return 1.20

    def _softmax_temperature(self, vec: List[float], temperature: float) -> List[float]:
        t = max(1e-6, float(temperature))
        if not vec:
            return []
        m = max(vec)
        exps = [math.exp((v - m) / t) for v in vec]
        Z = sum(exps) or 1.0
        return [x / Z for x in exps]


# =============================================================================
# RelationGraphBuilder (final: neutralized compat mod + synergy guarantee + quantile logs)
# =============================================================================
class RelationGraphBuilder:
    """감정 간 호환성/강도/신뢰도를 반영해 관계 그래프(간선 리스트)를 생성."""

    def __init__(
            self,
            rules: Dict[str, "EmotionRule"],
            *,
            min_edge_strength: float = 0.05,  # 절대 하한(이 미만은 후보에도 포함 X)
            confidence_gate: float = 0.50,  # 강도에 반영될 신뢰도 게이트(0~1)
            compat_mod_amp: float = 0.30,  # 호환성 편차 증폭 폭
            edge_quantile: float = 0.50,  # 기본 q (중앙값 컷)
            # --- 선택: 목표 간선 비율 피드백(문서별 밀도 적응) ---
            target_keep_ratio: Tuple[float, float] = (0.40, 0.60),  # kept/total 목표 범위
            q_step: float = 0.05,  # q 조정 보폭
            q_minmax: Tuple[float, float] = (0.30, 0.80),  # q 하한/상한
    ):
        self.rules = rules or {}
        self.min_edge_strength = float(max(0.0, min(1.0, min_edge_strength)))
        self.confidence_gate = float(max(0.0, min(1.0, confidence_gate)))
        self.compat_mod_amp = float(max(0.0, min(1.0, compat_mod_amp)))

        # 기록용 상태
        self._last_edge_quantiles: Optional[Dict[str, Any]] = None
        self._last_q_used: float = float(max(0.0, min(1.0, edge_quantile)))

        # 피드백 파라미터
        self._target_keep_lo = float(max(0.0, min(1.0, target_keep_ratio[0])))
        self._target_keep_hi = float(max(0.0, min(1.0, target_keep_ratio[1])))
        if self._target_keep_lo > self._target_keep_hi:
            self._target_keep_lo, self._target_keep_hi = self._target_keep_hi, self._target_keep_lo
        self._q_step = float(max(0.0, min(0.25, q_step)))
        self._q_min = float(max(0.0, min(1.0, q_minmax[0])))
        self._q_max = float(max(0.0, min(1.0, q_minmax[1])))

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def build(self, detected: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        det = {d.get("emotion_id"): d for d in (detected or []) if d and d.get("emotion_id")}
        ids = [e for e in EMOTION_AXES if e in det]
        n = len(ids)
        if n < 2:
            self._last_edge_quantiles = None
            return []

        # 1) 모든 간선 후보 계산(필터링은 나중에)
        candidates: List[Dict[str, Any]] = []
        strengths: List[float] = []

        for i in range(n - 1):
            a = ids[i]
            for j in range(i + 1, n):
                b = ids[j]
                da, db = det[a], det[b]
                sa, sb = float(da.get("score", 0.0)), float(db.get("score", 0.0))
                if sa <= 0.0 or sb <= 0.0:
                    continue

                # 기본 강도: 기하평균
                base = math.sqrt(sa * sb)

                # 호환성 및 타입
                c = self._compat(a, b)  # 0~1
                rtype = self._rtype_by_compat(c, sa, sb)

                # --- [개선] 호환성 편차 보정 '중립화' ---
                # 양 극단(시너지/갈등) 모두 가중하되 과도치 완화(감마<1)
                compat_deviation = abs(c - 0.5) * 2.0  # [0..1]
                gamma = 0.8
                compat_mod = 1.0 + self.compat_mod_amp * ((compat_deviation ** gamma) - (0.5 ** gamma))
                # 시너지 약한 보정(친화 0.65 이상이면 5% 가산)
                if c >= 0.65:
                    compat_mod *= 1.05

                # 신뢰도 게이트
                ca, cb = float(da.get("confidence", 0.0)), float(db.get("confidence", 0.0))
                conf_avg = max(0.0, min(1.0, 0.5 * (ca + cb)))
                conf_mod = 1.0 - self.confidence_gate + self.confidence_gate * (0.5 + 0.5 * conf_avg)

                strength = max(0.0, min(1.0, base * compat_mod * conf_mod))
                # 절대 하한 컷
                if strength < self.min_edge_strength:
                    continue

                # 키워드 겹침(설명용)
                kw_a = set(da.get("keywords") or [])
                kw_b = set(db.get("keywords") or [])
                denom = len(kw_a | kw_b)
                overlap = (len(kw_a & kw_b) / float(denom)) if denom > 0 else 0.0
                inter = list(sorted(kw_a & kw_b))[:8]

                candidates.append({
                    "source": a,
                    "target": b,
                    "relationship_type": rtype,
                    "strength": round(float(strength), 6),
                    "compatibility": round(float(c), 6),
                    "edge_confidence": round(float(conf_avg), 6),
                    "keyword_overlap": round(float(overlap), 6),
                    "shared_keywords": inter,
                })
                strengths.append(strength)

        if not candidates:
            self._last_edge_quantiles = None
            return []

        # 2) 상대 임계(분위수) 계산
        strengths_np = np.array(strengths, dtype=np.float64)
        q_used = float(max(0.0, min(1.0, self._last_q_used))) or 0.50
        thr = float(np.quantile(strengths_np, q_used))

        # 3) 1차 필터링
        kept = [it for it in candidates if it["strength"] >= thr]

        # 4) (선택) 목표 간선 비율 피드백: kept/total이 목표 범위를 벗어나면 q 미세 조정(1회)
        total_cand = len(candidates)
        kept_cnt = len(kept)
        kept_ratio = (kept_cnt / float(total_cand)) if total_cand > 0 else 0.0

        tuned = False
        if total_cand >= 3:
            if kept_ratio < self._target_keep_lo and q_used > self._q_min:
                q_used = max(self._q_min, round(q_used - self._q_step, 3))
                thr = float(np.quantile(strengths_np, q_used))
                kept = [it for it in candidates if it["strength"] >= thr]
                kept_cnt = len(kept)
                kept_ratio = kept_cnt / float(total_cand)
                tuned = True
            elif kept_ratio > self._target_keep_hi and q_used < self._q_max:
                q_used = min(self._q_max, round(q_used + self._q_step, 3))
                thr = float(np.quantile(strengths_np, q_used))
                kept = [it for it in candidates if it["strength"] >= thr]
                kept_cnt = len(kept)
                kept_ratio = kept_cnt / float(total_cand)
                tuned = True

        # --- [개선] 시너지 최소 1개 보장 ---
        synergy_cands = [it for it in candidates if it["relationship_type"] == "synergy"]
        if synergy_cands and not any(e["relationship_type"] == "synergy" for e in kept):
            best_syn = max(synergy_cands, key=lambda x: x["strength"])
            pair_set = {(e["source"], e["target"]) for e in kept}
            if (best_syn["source"], best_syn["target"]) not in pair_set:
                kept.append(best_syn)
                kept_cnt += 1
                kept_ratio = kept_cnt / float(total_cand) if total_cand > 0 else 0.0

        # 5) 로그(분위수 경계 + 밀도) 기록
        self._last_edge_quantiles = {
            "q_used": round(q_used, 3),
            "thr": round(thr, 6),
            "min": round(float(strengths_np.min()), 6),
            "median": round(float(np.quantile(strengths_np, 0.5)), 6),
            "max": round(float(strengths_np.max()), 6),
            "total_candidates": int(total_cand),
            "kept_edges": int(kept_cnt),
            "kept_ratio": round(float(kept_ratio), 6),
            "feedback_applied": bool(tuned),
        }
        # (오케스트레이터 과거 키와도 합치기)
        self._last_edge_threshold = self._last_edge_quantiles["thr"]
        self._last_quantile = self._last_edge_quantiles["q_used"]

        kept.sort(key=lambda e: (e["strength"], e["compatibility"]), reverse=True)

        return kept

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _compat(self, a: str, b: str) -> float:
        ra, rb = self.rules.get(a), self.rules.get(b)
        if rb and a in (rb.incompatible_with or []): return 0.15
        if ra and b in (ra.incompatible_with or []): return 0.15
        if rb and a in (rb.compatible_with or []):   return 0.70
        if ra and b in (ra.compatible_with or []):   return 0.70
        return 0.50

    def _rtype_by_compat(self, c: float, sa: float, sb: float) -> str:
        if c <= 0.30:
            return "conflict"
        if c < 0.55:
            return "tension"
        if c >= 0.62 or (c >= 0.58 and min(sa, sb) >= 0.25):
            return "synergy"
        return "neutral"


# =============================================================================
# ConflictResolver
# =============================================================================
class ConflictResolver:
    """
    관계 그래프와 문서 레벨 증거를 기반으로 충돌/긴장을 산출.
    개선점:
      - severity: (증거강도, 관계강도, 분산, 불균형)을 z-score 표준화 후 가중합 → sigmoid로 [0,1]
      - 선택: 상위 K 고정 대신 누적 비중 0.8까지 선택(밀도 적응)
      - 서브감정 페어도 누적 0.8 기준으로 선택
    """

    def __init__(
            self,
            severity_threshold: float = 0.60,  # 이 값 이상이면 resolution_needed=True
            max_conflicts: int = 10,  # 안전 상한(선택 로직은 누적 0.8 우선)
            sub_pairs: int = 4,  # (하위호환) 서브감정 페어 기본 상한
            top_kw: int = 8,  # 설명용 공유 키워드 상한
            compat_amp: float = 0.35,  # 설명 지표용: 호환성 편차 영향
            overlap_amp: float = 0.25,  # 설명 지표용: 키워드 중첩 영향
            balance_amp: float = 0.20,  # 설명 지표용: 균형(유사 강도) 영향

            # --- 새 옵션 ---
            cumulative_ratio: float = 0.80,  # 누적 비중 선택 기준
            w_edge: float = 0.35,  # severity 가중치: 관계강도
            w_evidence: float = 0.30,  # severity 가중치: 증거강도
            w_dispersion: float = 0.20,  # severity 가중치: 분산(세그 커버리지)
            w_imbalance: float = 0.15,  # severity 가중치: 불균형(|sa - sb|)
    ):
        self.severity_threshold = float(max(0.0, min(1.0, severity_threshold)))
        self.max_conflicts = int(max(1, max_conflicts))
        self.sub_pairs_legacy = int(max(0, sub_pairs))  # 누적 0.8 사용하되 안전상한으로 활용
        self.top_kw = int(max(0, top_kw))
        self.compat_amp = float(max(0.0, min(1.0, compat_amp)))
        self.overlap_amp = float(max(0.0, min(1.0, overlap_amp)))
        self.balance_amp = float(max(0.0, min(1.0, balance_amp)))

        self.cumulative_ratio = float(max(0.5, min(0.95, cumulative_ratio)))
        # severity 가중치 정규화
        total_w = max(1e-8, (w_edge + w_evidence + w_dispersion + w_imbalance))
        self.w_edge = float(w_edge) / total_w
        self.w_evidence = float(w_evidence) / total_w
        self.w_dispersion = float(w_dispersion) / total_w
        self.w_imbalance = float(w_imbalance) / total_w

    # ------------------------------------------------------------------ #
    # Public
    # ------------------------------------------------------------------ #
    def derive(self, detected: List[Dict[str, Any]], relations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # 맵 준비
        score_map = {d["emotion_id"]: float(d.get("score", 0.0)) for d in (detected or []) if d.get("emotion_id")}
        conf_map = {d["emotion_id"]: float(d.get("confidence", 0.0)) for d in (detected or []) if d.get("emotion_id")}
        kw_map = {d["emotion_id"]: set(d.get("keywords") or []) for d in (detected or []) if d.get("emotion_id")}
        span_cov = {d["emotion_id"]: float(((d.get("span") or {}).get("coverage_ratio", 0.0))) for d in (detected or [])
                    if d.get("emotion_id")}
        subs_map = {
            d["emotion_id"]: [(se.get("name"), float(se.get("score", 0.0))) for se in (d.get("sub_emotions") or [])]
            for d in (detected or []) if d.get("emotion_id")
        }

        # 1) 후보 수집(충돌/긴장만)
        candidates: List[Dict[str, Any]] = []
        for r in relations or []:
            rtype = r.get("relationship_type")
            if rtype not in ("conflict", "tension"):
                continue

            a, b = r.get("source"), r.get("target")
            if not a or not b:
                continue

            base = float(r.get("strength", 0.0))  # 관계강도(0~1)
            if base <= 0.0:
                continue

            # 증거강도: 간선 평균 신뢰도(없으면 감정 신뢰도 평균)
            edge_conf = r.get("edge_confidence", None)
            conf_avg = float(edge_conf) if edge_conf is not None else 0.5 * (
                        conf_map.get(a, 0.0) + conf_map.get(b, 0.0))
            conf_avg = max(0.0, min(1.0, conf_avg))

            # 분산(coverage): 두 감정의 세그 커버리지 평균
            cov_a = span_cov.get(a, 0.0)
            cov_b = span_cov.get(b, 0.0)
            dispersion = max(0.0, min(1.0, 0.5 * (cov_a + cov_b)))

            # 불균형: |sa - sb|
            sa = float(score_map.get(a, 0.0))
            sb = float(score_map.get(b, 0.0))
            imbalance = abs(sa - sb)

            # 키워드 중첩(설명용)
            if "keyword_overlap" in r:
                overlap = float(r.get("keyword_overlap", 0.0))
                shared = list(r.get("shared_keywords") or [])[: self.top_kw]
            else:
                a_k, b_k = kw_map.get(a, set()), kw_map.get(b, set())
                shared = sorted(a_k & b_k)[: self.top_kw]
                overlap = self._jaccard(a_k, b_k)

            # 호환성(설명용)
            compat = float(r.get("compatibility", 0.5))
            compat = max(0.0, min(1.0, compat))
            compat_dev = abs(compat - 0.5) * 2.0  # 0~1

            candidates.append({
                "a": a, "b": b, "rtype": rtype,
                "edge_strength": base,  # 관계강도
                "evidence_strength": conf_avg,  # 증거강도(간선/감정 신뢰도)
                "dispersion": dispersion,  # 세그 커버리지
                "imbalance": imbalance,  # |sa-sb|
                "sa": sa, "sb": sb,
                "compat": compat, "compat_dev": compat_dev,
                "overlap": max(0.0, min(1.0, overlap)),
                "shared": shared,
            })

        if not candidates:
            return []

        # 2) z-score 표준화
        z_edge = self._zscore([c["edge_strength"] for c in candidates])
        z_evidence = self._zscore([c["evidence_strength"] for c in candidates])
        z_dispersion = self._zscore([c["dispersion"] for c in candidates])
        z_imbalance = self._zscore([c["imbalance"] for c in candidates])

        # 3) 가중합 → sigmoid로 [0,1] 매핑
        out_rows: List[Dict[str, Any]] = []
        for idx, c in enumerate(candidates):
            lin = (
                    self.w_edge * z_edge[idx] +
                    self.w_evidence * z_evidence[idx] +
                    self.w_dispersion * z_dispersion[idx] +
                    self.w_imbalance * z_imbalance[idx]
            )
            severity = self._sigmoid(lin)  # 0~1

            # 설명용 지표(가시성): 기존 로직 유지
            explain_strength = c["edge_strength"] * (
                    1.0
                    + self.compat_amp * (c["compat_dev"] - 0.5)
                    + self.overlap_amp * (c["overlap"] - 0.5)
                    + self.balance_amp * ((1.0 - c["imbalance"]) - 0.5)
            )
            explain_strength = max(0.0, min(1.0, explain_strength))

            # 서브감정 충돌(누적 0.8)
            sub_conflicts = self._derive_sub_conflicts_cum(
                c["a"], c["b"], subs_map, conf_gate=c["evidence_strength"],
                max_fallback=self.sub_pairs_legacy
            )

            out_rows.append({
                "emotions": [c["a"], c["b"]],
                "type": "대립" if c["rtype"] == "conflict" else "긴장",
                "severity": round(float(severity), 6),
                "intensity": round(float(c["edge_strength"]), 6),  # 관계강도(원시)
                "resolution_needed": bool(severity >= self.severity_threshold),

                # 참고/설명 필드
                "compatibility": round(float(c["compat"]), 6),
                "edge_confidence": round(float(c["evidence_strength"]), 6),
                "keyword_overlap": round(float(c["overlap"]), 6),
                "shared_keywords": list(c["shared"]),
                "balance": round(float(1.0 - c["imbalance"]), 6),
                "stability": round(
                    float(0.4 * c["evidence_strength"] + 0.3 * (1.0 - c["imbalance"]) + 0.3 * min(c["sa"], c["sb"])),
                    6),
                "explain_strength": round(float(explain_strength), 6),
                "components": {  # 디버그/점검용
                    "edge_z": round(z_edge[idx], 4),
                    "evidence_z": round(z_evidence[idx], 4),
                    "dispersion_z": round(z_dispersion[idx], 4),
                    "imbalance_z": round(z_imbalance[idx], 4),
                    "linear": round(lin, 4),
                },
                "sub_conflicts": sub_conflicts,
            })

        # 4) 누적 0.8 선택(문서별 적응)
        out_rows.sort(key=lambda x: x["severity"], reverse=True)
        total = sum(r["severity"] for r in out_rows) or 1.0
        kept, cum = [], 0.0
        for r in out_rows:
            kept.append(r)
            cum += r["severity"]
            if cum / total >= self.cumulative_ratio:
                break

        # 안전 상한 적용(너무 많을 때만 컷)
        if len(kept) > self.max_conflicts:
            kept = kept[: self.max_conflicts]

        return kept

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _derive_sub_conflicts_cum(
            self,
            a: str,
            b: str,
            subs_map: Dict[str, List[Tuple[str, float]]],
            conf_gate: float,
            max_fallback: int,
            threshold_ratio: float = 0.80
    ) -> List[Dict[str, Any]]:
        """서브감정 페어를 점수 내림차순으로 누적 0.8까지 선택(없으면 상위 max_fallback 사용)."""
        subs_a = [(n, s) for (n, s) in (subs_map.get(a) or []) if n]
        subs_b = [(n, s) for (n, s) in (subs_map.get(b) or []) if n]
        if not subs_a or not subs_b:
            return []

        # 상위 일부만 전처리로 컷(안정성), 이후 누적 0.8
        subs_a = sorted(subs_a, key=lambda x: x[1], reverse=True)[: min(12, len(subs_a))]
        subs_b = sorted(subs_b, key=lambda x: x[1], reverse=True)[: min(12, len(subs_b))]

        pairs: List[Tuple[str, str, float]] = []
        gate = (0.5 + 0.5 * max(0.0, min(1.0, conf_gate)))  # 신뢰도 게이트
        for na, sa in subs_a:
            for nb, sb in subs_b:
                sc = math.sqrt(max(0.0, sa) * max(0.0, sb)) * gate
                pairs.append((na, nb, sc))

        pairs.sort(key=lambda x: x[2], reverse=True)
        total = sum(sc for _, _, sc in pairs) or 1.0

        out: List[Dict[str, Any]] = []
        cum = 0.0
        for na, nb, sc in pairs:
            out.append({"a_sub": na, "b_sub": nb, "score": round(float(max(0.0, min(1.0, sc))), 6)})
            cum += sc
            if cum / total >= threshold_ratio:
                break

        # 예외적으로 너무 적으면 레거시 상한으로 보완
        if not out and max_fallback > 0:
            for na, nb, sc in pairs[:max_fallback]:
                out.append({"a_sub": na, "b_sub": nb, "score": round(float(max(0.0, min(1.0, sc))), 6)})

        return out

    def _zscore(self, xs: List[float]) -> List[float]:
        n = len(xs)
        if n == 0:
            return []
        mean = sum(xs) / n
        var = sum((x - mean) ** 2 for x in xs) / n
        std = math.sqrt(var)
        if std < 1e-8:
            return [0.0] * n
        return [(x - mean) / std for x in xs]

    def _sigmoid(self, x: float) -> float:
        # 수치 안정화
        if x < -20:
            return 0.0
        if x > 20:
            return 1.0
        return 1.0 / (1.0 + math.exp(-x))

    def _jaccard(self, a: set, b: set) -> float:
        if not a and not b:
            return 0.0
        inter = len(a & b)
        union = len(a | b)
        if union <= 0:
            return 0.0
        return inter / float(union)


# =============================================================================
# DominantEmotionRanker
# =============================================================================
class DominantEmotionRanker:
    """ 라벨 점수 + (지지-충돌) 연결 + 파워 중심성으로 지배 감정을 산출하고,
    상황에 맞게 역할(role)을 부여합니다. """

    # 폴백 EMOTION_AXES (외부 상수가 없을 때)
    _DEFAULT_AXES = ("희", "노", "애", "락")

    def __init__(
            self,
            # 합성 스코어 가중치
            score_weight: float = 0.50,  # 원시 감정 점수 가중
            degree_weight: float = 0.25,  # (지지-충돌) 연결 가중
            centrality_weight: float = 0.25,  # 파워 중심성 가중

            # 간선 타입 가중/패널티
            conflict_penalty: float = 0.35,  # 충돌 간선 감점
            synergy_boost: float = 0.10,  # 시너지 간선 가점

            # 중심성 파라미터
            damping: float = 0.85,  # 파워 반복 감쇠
            iterations: int = 25,  # 파워 반복 횟수

            # 최종 스코어 소프트맥스
            temperature: float = 0.75,

            # 리포트 설정
            top_related: int = 3,  # 각 감정별 노출할 상위 관련 간선
            degree_strategy: str = "count",  # 'count' | 'weighted' (지지/대립 비율 집계 전략)

            # 역할 다양화를 위한 임계
            driver_inf_min: float = 0.25,  # 드라이버로 볼 최소 영향도(정규화 후)
            co_driver_gap: float = 0.05,  # 1위 대비 co-driver 허용 갭(5%)
            supporter_min_ratio: float = 0.30,  # supporter로 볼 최소 지지 비율
            pressured_ratio: float = 0.60  # pressured로 볼 최소 대립 비율
    ):
        # --- 가중치 정규화 ---
        self.w_score = float(max(0.0, min(1.0, score_weight)))
        self.w_degree = float(max(0.0, min(1.0, degree_weight)))
        self.w_central = float(max(0.0, min(1.0, centrality_weight)))
        s = self.w_score + self.w_degree + self.w_central
        if s <= 0.0:
            self.w_score, self.w_degree, self.w_central = 1.0, 0.0, 0.0
        else:
            self.w_score /= s
            self.w_degree /= s
            self.w_central /= s

        # --- 하이퍼/옵션 ---
        self.conflict_penalty = float(max(0.0, min(1.0, conflict_penalty)))
        self.synergy_boost = float(max(0.0, min(1.0, synergy_boost)))
        self.damping = float(max(0.0, min(1.0, damping)))
        self.iterations = int(max(1, iterations))
        self.temperature = max(1e-6, float(temperature))
        self.top_related = int(max(0, top_related))
        self.degree_strategy = degree_strategy if degree_strategy in ("count", "weighted") else "count"

        # --- 간선 타입 가중/판정 ---
        self._rtype_weight = {
            "synergy": 1.0 + self.synergy_boost,
            "neutral": 0.60,
            "tension": 0.50,
            "conflict": max(0.10, 0.50 - self.conflict_penalty),
        }
        self._rtype_support = {"synergy": 1.0, "neutral": 0.20, "tension": 0.0, "conflict": 0.0}
        self._rtype_oppose = {"synergy": 0.0, "neutral": 0.0, "tension": 0.60, "conflict": 1.0}

        # --- 역할 임계 ---
        self.driver_inf_min = float(driver_inf_min)
        self.co_driver_gap = float(co_driver_gap)
        self.supporter_min_ratio = float(supporter_min_ratio)
        self.pressured_ratio = float(pressured_ratio)

        # EMOTION_AXES 폴백 준비
        try:
            # 외부에서 정의되어 있으면 그대로 사용
            self.EMOTION_AXES = tuple(EMOTION_AXES)  # type: ignore[name-defined]
        except Exception:
            self.EMOTION_AXES = self._DEFAULT_AXES

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def rank(
            self,
            detected: List[Dict[str, Any]],
            relations: List[Dict[str, Any]],
            *,
            nodes: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        detected: [{"emotion_id","score","confidence", ...}, ...]
        relations: [{"source","target","relationship_type","strength","edge_confidence", ...}, ...]
        nodes: 사용할 축 명시(없으면 EMOTION_AXES→detected→관계에서 추론)
        """
        # --- 노드 셋 결정 ---
        axes = self._resolve_nodes(nodes, detected, relations)

        # --- 점수/신뢰도 맵 ---
        score_map = {d["emotion_id"]: float(d.get("score", 0.0)) for d in (detected or []) if
                     d.get("emotion_id") in axes}
        conf_map = {d["emotion_id"]: float(d.get("confidence", 0.0)) for d in (detected or []) if
                    d.get("emotion_id") in axes}

        # --- 1) 지지/충돌 누적 + 인접행렬 ---
        support_deg = {e: 0.0 for e in axes}
        oppose_deg = {e: 0.0 for e in axes}
        adj, edge_book = self._build_adjacency(axes, relations, conf_map, accumulate=(support_deg, oppose_deg))

        # --- 2) (지지 - 패널티*충돌) 정규화 ---
        deg_pos = {e: max(0.0, support_deg[e] - self.conflict_penalty * oppose_deg[e]) for e in axes}
        deg_norm = self._normalize(deg_pos)
        base_norm = self._normalize(score_map)

        # --- 3) 파워 중심성 ---
        central = self._power_centrality(axes, adj, base_norm, self.damping, self.iterations)
        central_norm = self._normalize(central)

        # --- 4) 합성 스코어 (문서 레벨 confidence 게이팅) ---
        raw = {}
        for e in axes:
            conf_gate = 0.5 + 0.5 * max(0.0, min(1.0, conf_map.get(e, 0.0)))
            v = (
                    self.w_score * base_norm.get(e, 0.0) +
                    self.w_degree * deg_norm.get(e, 0.0) +
                    self.w_central * central_norm.get(e, 0.0)
            )
            raw[e] = max(0.0, min(1.0, v * conf_gate))

        # --- 5) 소프트맥스 정규화 ---
        final_scores = self._softmax_map(raw, temperature=self.temperature)

        # --- 6) 노드별 간선 수집/정렬 (관련 간선 상위 노출) ---
        rels_by_node = {e: [] for e in axes}
        for ed in edge_book:
            # edge_book: dict items already augmented with 'edge_weight'
            a, b = ed["source"], ed["target"]
            if a in rels_by_node:
                rels_by_node[a].append(ed)
            if b in rels_by_node:
                rels_by_node[b].append(ed)
        for e in axes:
            rels_by_node[e].sort(key=lambda x: (x.get("edge_weight", 0.0), x.get("strength", 0.0)), reverse=True)

        # --- 7) 비율 계산(집계 전략: count | weighted) ---
        ratios = self._calc_ratios(axes, relations, edge_book)

        # --- 8) 역할(role) 부여 ---
        top_id = max(final_scores, key=final_scores.get) if final_scores else None
        top_val = final_scores.get(top_id, 0.0) if top_id else 0.0
        max_cent = max(central_norm.values() or [0.0])

        rows = []
        # 안정적 정렬(동률 시 중심성/베이스/이름으로 결정)
        order = sorted(
            final_scores.items(),
            key=lambda x: (x[1], central_norm.get(x[0], 0.0), base_norm.get(x[0], 0.0), -ord(x[0][0])),
            reverse=True
        )
        for eid, infl in order:
            sup_ratio = ratios[eid]["sup_ratio"]
            opp_ratio = ratios[eid]["opp_ratio"]
            role = self._role_of(
                e=eid,
                influence=infl,
                centrality=central_norm.get(eid, 0.0),
                sup_ratio=sup_ratio,
                opp_ratio=opp_ratio,
                is_top=(eid == top_id),
                top_val=top_val,
                max_cent=max_cent,
            )
            rows.append({
                "emotion_id": eid,
                "influence_score": round(float(infl), 6),
                "related_emotions": rels_by_node.get(eid, [])[: self.top_related],
                "support_degree": round(sup_ratio, 6),
                "conflict_degree": round(opp_ratio, 6),
                "centrality": round(central_norm.get(eid, 0.0), 6),
                "confidence": round(conf_map.get(eid, 0.0), 6),
                "role": role,
            })
        return rows

    # --------------------------------------------------------------------- #
    # Internals
    # --------------------------------------------------------------------- #
    def _resolve_nodes(
            self,
            nodes: Optional[List[str]],
            detected: List[Dict[str, Any]],
            relations: List[Dict[str, Any]],
    ) -> Tuple[str, ...]:
        if nodes:
            return tuple(nodes)
        # 1) 외부/폴백 상수
        axes = list(self.EMOTION_AXES)
        # 2) detected 보강
        for d in detected or []:
            eid = d.get("emotion_id")
            if eid and eid not in axes:
                axes.append(eid)
        # 3) relations 보강
        for r in relations or []:
            a, b = r.get("source"), r.get("target")
            if a and a not in axes:
                axes.append(a)
            if b and b not in axes:
                axes.append(b)
        return tuple(axes)

    def _build_adjacency(
            self,
            nodes: Tuple[str, ...],
            relations: List[Dict[str, Any]],
            conf_map: Dict[str, float],
            accumulate: Optional[Tuple[Dict[str, float], Dict[str, float]]] = None,
    ) -> Tuple[Dict[Tuple[str, str], float], List[Dict[str, Any]]]:
        """
        반환:
          adj: 가중 인접행렬 (양방향 합산)
          edge_book: 각 간선에 edge_weight(정렬용)를 부여한 리스트
        accumulate=(support_deg, oppose_deg) 가 주어지면 가중 누적도 병행
        """
        adj = {(a, b): 0.0 for a in nodes for b in nodes if a != b}
        edge_book: List[Dict[str, Any]] = []

        for r in relations or []:
            a, b = r.get("source"), r.get("target")
            if not a or not b or a == b or a not in nodes or b not in nodes:
                continue
            s = float(r.get("strength", 0.0))
            if s <= 0.0:
                continue
            typ = r.get("relationship_type", "neutral")
            base_w = self._rtype_weight.get(typ, 0.60)

            edge_conf = r.get("edge_confidence")
            if edge_conf is None:
                edge_conf = 0.5 * (conf_map.get(a, 0.0) + conf_map.get(b, 0.0))
            gate = 0.5 + 0.5 * max(0.0, min(1.0, float(edge_conf)))

            # 최종 간선 가중(인접행렬용)
            w = s * base_w * gate
            adj[(a, b)] += w
            adj[(b, a)] += w

            # 지지/대립 가중(비율용)
            sup_w = s * self._rtype_support.get(typ, 0.2) * gate
            opp_w = s * self._rtype_oppose.get(typ, 0.0) * gate
            if accumulate is not None:
                support_deg, oppose_deg = accumulate
                support_deg[a] += sup_w
                support_deg[b] += sup_w
                oppose_deg[a] += opp_w
                oppose_deg[b] += opp_w

            # 리포트용으로 간선 weight 부착(정렬/노출)
            rb = dict(r)
            rb["edge_weight"] = w
            edge_book.append(rb)

        return adj, edge_book

    def _power_centrality(
            self,
            nodes: Tuple[str, ...],
            adj: Dict[Tuple[str, str], float],
            base: Dict[str, float],
            damping: float,
            iters: int,
    ) -> Dict[str, float]:
        if not nodes:
            return {}
        # 초기 벡터(균등)
        v = {n: 1.0 / len(nodes) for n in nodes}
        for _ in range(iters):
            nv = {n: 0.0 for n in nodes}
            for i in nodes:
                s = 0.0
                for j in nodes:
                    if i == j:
                        continue
                    s += adj.get((j, i), 0.0) * v[j]
                nv[i] = damping * s + (1.0 - damping) * base.get(i, 0.0)
            z = sum(nv.values()) or 1.0
            v = {k: (nv[k] / z) for k in nodes}
        return v

    def _calc_ratios(
            self,
            nodes: Tuple[str, ...],
            relations: List[Dict[str, Any]],
            edge_book: List[Dict[str, Any]],
    ) -> Dict[str, Dict[str, float]]:
        out = {e: {"sup_ratio": 0.0, "opp_ratio": 0.0} for e in nodes}

        if self.degree_strategy == "weighted":
            sup_sum = {e: 0.0 for e in nodes}
            opp_sum = {e: 0.0 for e in nodes}
            for r in edge_book:
                a, b = r["source"], r["target"]
                typ = r.get("relationship_type", "neutral")
                s = float(r.get("strength", 0.0))
                gate = (r.get("edge_weight", 0.0) / max(1e-9,
                                                        float(self._rtype_weight.get(typ, 0.60)))) if s > 0 else 0.0
                sup = s * self._rtype_support.get(typ, 0.2) * gate
                opp = s * self._rtype_oppose.get(typ, 0.0) * gate
                sup_sum[a] += sup
                sup_sum[b] += sup
                opp_sum[a] += opp
                opp_sum[b] += opp
            for e in nodes:
                tot = sup_sum[e] + opp_sum[e]
                if tot <= 0:
                    out[e]["sup_ratio"] = 0.0
                    out[e]["opp_ratio"] = 0.0
                else:
                    out[e]["sup_ratio"] = sup_sum[e] / tot
                    out[e]["opp_ratio"] = opp_sum[e] / tot
            return out

        # count 전략(기존 리포트와 동일한 느낌 유지)
        counts = {e: {"sup": 0, "opp": 0, "tot": 0} for e in nodes}
        for r in relations or []:
            a, b = r.get("source"), r.get("target")
            if not a or not b or a not in nodes or b not in nodes or a == b:
                continue
            t = r.get("relationship_type", "neutral")
            if t == "synergy":
                counts[a]["sup"] += 1
                counts[b]["sup"] += 1
            elif t in ("conflict", "tension"):
                counts[a]["opp"] += 1
                counts[b]["opp"] += 1
            counts[a]["tot"] += 1
            counts[b]["tot"] += 1
        for e in nodes:
            tot = max(1, counts[e]["tot"])
            out[e]["sup_ratio"] = counts[e]["sup"] / tot
            out[e]["opp_ratio"] = counts[e]["opp"] / tot
        return out

    def _role_of(
            self,
            e: str,
            influence: float,
            centrality: float,
            sup_ratio: float,
            opp_ratio: float,
            is_top: bool,
            top_val: float,
            max_cent: float,
    ) -> str:
        # 1) 최상위 축 co-driver 승격(영향도 상위 & 중심성 상위일 때)
        if is_top and (influence >= self.driver_inf_min or influence >= top_val * (1 - self.co_driver_gap)) \
                and (centrality >= max(0.6, 0.9 * max_cent)):
            return "co-driver"
        # 2) 지지 충분 → supporter
        if sup_ratio >= self.supporter_min_ratio and influence >= 0.22:
            return "supporter"
        # 3) 대립 강함 → pressured
        if opp_ratio >= self.pressured_ratio:
            return "pressured"
        # 4) 상위권 보조 드라이버 (2차 타이브레이커)
        if influence >= max(0.35, top_val * (1 - 2 * self.co_driver_gap)):
            return "co-driver"
        return "balanced"

    # -------------------- helpers -------------------- #
    def _normalize(self, m: Dict[str, float]) -> Dict[str, float]:
        if not m:
            return {}
        mx = max(0.0, max(m.values()))
        if mx <= 0.0:
            return {k: 0.0 for k in m}
        return {k: max(0.0, min(1.0, v / mx)) for k, v in m.items()}

    def _softmax_map(self, m: Dict[str, float], temperature: float) -> Dict[str, float]:
        keys = list(m.keys())
        vals = [float(m[k]) for k in keys]
        probs = self._softmax_temperature(vals, temperature=temperature)
        return {k: p for k, p in zip(keys, probs)}

    @staticmethod
    def _softmax_temperature(x: List[float], temperature: float = 1.0) -> List[float]:
        """수치안정 softmax (max-shift)."""
        if temperature <= 0:
            temperature = 1e-6
        x_scaled = [v / temperature for v in x]
        m = max(x_scaled) if x_scaled else 0.0
        exps = [math.exp(v - m) for v in x_scaled]
        s = sum(exps) or 1.0
        return [v / s for v in exps]


# =============================================================================
# Clusterer
# =============================================================================
class Clusterer:
    """감정 축 중심의 책임도 기반 소군집 생성(결정론적·라벨링 우선)."""

    def __init__(
            self,
            w_score: float = 0.55,  # 중심-구성원 강도(점수기반) 가중
            w_kw: float = 0.25,  # 키워드 유사도 가중
            w_sub: float = 0.20,  # 서브감정 유사도 가중
            temperature_centerwise: float = 0.85,  # 클러스터 내부 softmax 온도
            temperature_crosscenter: float = 0.85,  # 교차 중심 softmax 온도
            center_floor: float = 0.90,  # 중심 감정의 최소 membership 하한
            min_keep: int = 1,  # 각 클러스터 최소 구성원 수
            max_members: int = 4  # 각 클러스터 최대 구성원 수
    ):
        s = max(1e-9, w_score + w_kw + w_sub)
        self.w_score = float(w_score / s)
        self.w_kw = float(w_kw / s)
        self.w_sub = float(w_sub / s)
        self.t_center = max(1e-6, float(temperature_centerwise))
        self.t_cross = max(1e-6, float(temperature_crosscenter))
        self.center_floor = max(0.0, min(1.0, float(center_floor)))
        self.min_keep = int(max(0, min_keep))
        self.max_members = int(max(1, max_members))

    def cluster(self, detected: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        det = {d["emotion_id"]: d for d in (detected or []) if d.get("emotion_id")}
        if not det:
            return []

        score_map = {e: float(det[e].get("score", 0.0)) for e in EMOTION_AXES if e in det}
        conf_map = {e: float(det[e].get("confidence", 0.0)) for e in EMOTION_AXES if e in det}
        kw_map = {e: self._kw_set(det[e].get("keywords")) for e in EMOTION_AXES if e in det}
        sub_map = {e: self._sub_map(det[e].get("sub_emotions")) for e in EMOTION_AXES if e in det}

        # 중심-구성원 유사도 행렬 S[c][e] 계산
        S: Dict[str, Dict[str, float]] = {c: {e: 0.0 for e in EMOTION_AXES if e in det} for c in EMOTION_AXES if
                                          c in det}
        for c in S.keys():
            for e in S[c].keys():
                S[c][e] = self._pair_similarity(
                    c, e,
                    score_map, conf_map,
                    kw_map, sub_map
                )

        # 클러스터 내부 정규화(중심 기준 softmax)로 membership_row[c][e] 산출
        membership_row: Dict[str, Dict[str, float]] = {}
        for c, row in S.items():
            keys = list(row.keys())
            vals = [row[k] for k in keys]
            probs = softmax_temperature(vals, temperature=self.t_center)
            m = {k: float(p) for k, p in zip(keys, probs)}
            # 중심 감정 하한 적용
            if c in m:
                m[c] = max(self.center_floor, m[c])
                # 재정규화
                z = sum(m.values()) or 1.0
                m = {k: v / z for k, v in m.items()}
            membership_row[c] = m

        # 교차 중심 정규화로 각 감정의 책임도(어느 중심에 더 속하는가)
        membership_col: Dict[str, Dict[str, float]] = {e: {} for e in det.keys()}
        for e in det.keys():
            keys = list(membership_row.keys())  # centers
            vals = [membership_row[c].get(e, 0.0) for c in keys]
            probs = softmax_temperature(vals, temperature=self.t_cross)
            membership_col[e] = {c: float(p) for c, p in zip(keys, probs)}

        # 분리도 계산을 위해 각 감정의 1/2순위 중심 점수 확보
        first_second_gap: Dict[str, float] = {}
        for e, col in membership_col.items():
            if not col:
                first_second_gap[e] = 0.0
            else:
                sorted_cs = sorted(col.items(), key=lambda x: x[1], reverse=True)
                top = sorted_cs[0][1]
                sec = sorted_cs[1][1] if len(sorted_cs) > 1 else 0.0
                first_second_gap[e] = max(0.0, top - sec)

        # 클러스터별 멤버 정리 및 지표 산출
        clusters: List[Dict[str, Any]] = []
        for c in membership_row.keys():
            row = membership_row[c]
            # 멤버 후보 정렬
            members_sorted = sorted(row.items(), key=lambda x: x[1], reverse=True)
            # 최소/최대 멤버 수 적용
            chosen = members_sorted[: max(self.min_keep, self.max_members)]
            # 멤버 상세 구성
            members = []
            for eid, w in chosen:
                members.append(
                    {
                        "emotion_id": eid,
                        "score": float(score_map.get(eid, 0.0)),
                        "sub_emotions": det[eid].get("sub_emotions", []),
                        "keywords": det[eid].get("keywords", []),
                        "weight": float(w),
                        "responsibility": float(membership_col[eid].get(c, 0.0)),
                    }
                )

            # 지표: 평균 점수(가중), 응집도(중심과의 유사도 가중 평균), 분리도(책임도 1-2위 차 평균)
            avg_score = self._weighted_mean([score_map.get(m["emotion_id"], 0.0) for m in members],
                                            [m["weight"] for m in members])
            cohesion = self._weighted_mean(
                [S[c][m["emotion_id"]] for m in members],
                [m["weight"] for m in members]
            )
            separation = float(
                np.mean([first_second_gap.get(m["emotion_id"], 0.0) for m in members])) if members else 0.0
            cluster_strength = max(0.0, min(1.0, 0.65 * cohesion + 0.35 * separation))

            # 대표 키워드/서브감정 요약
            top_keywords = self._top_keywords(members, topk=6)
            rep_subs = self._top_sub_emotions(members, topk=6)

            clusters.append(
                {
                    "center_emotion": c,
                    "members": members,
                    "average_score": round(float(avg_score), 6),
                    "cluster_strength": round(float(cluster_strength), 6),
                    "cohesion": round(float(cohesion), 6),
                    "separation": round(float(separation), 6),
                    "top_keywords": top_keywords,
                    "representative_sub_emotions": rep_subs,
                }
            )

        # Clusterer.cluster(...) 결과 만들기 직전
        clusters = [c for c in clusters if len(c.get("members", [])) >= 2 or c.get("cohesion", 0) >= 0.25]

        return clusters

    # 유사도 = 점수강도(기하평균) + 키워드 자카드 + 서브감정 가중 자카드, 신뢰도 게이팅
    def _pair_similarity(
            self,
            c: str,
            e: str,
            score_map: Dict[str, float],
            conf_map: Dict[str, float],
            kw_map: Dict[str, set],
            sub_map: Dict[str, Dict[str, float]],
    ) -> float:
        if c not in score_map or e not in score_map:
            return 0.0
        s_center = float(score_map.get(c, 0.0))
        s_member = float(score_map.get(e, 0.0))
        base = math.sqrt(max(0.0, s_center) * max(0.0, s_member))
        kw_sim = self._jaccard(kw_map.get(c, set()), kw_map.get(e, set()))
        sub_sim = self._weighted_jaccard(sub_map.get(c, {}), sub_map.get(e, {}))
        conf_gate = 0.5 * (max(0.0, min(1.0, conf_map.get(c, 0.0))) + max(0.0, min(1.0, conf_map.get(e, 0.0))))
        sim = self.w_score * base + self.w_kw * kw_sim + self.w_sub * sub_sim
        sim *= (0.5 + 0.5 * conf_gate)
        return max(0.0, min(1.0, float(sim)))

    def _kw_set(self, kws: Any) -> set:
        if not kws:
            return set()
        if isinstance(kws, (list, tuple, set)):
            return {str(x).strip().lower() for x in kws if x}
        return set()

    def _sub_map(self, subs: Any) -> Dict[str, float]:
        out: Dict[str, float] = {}
        if not subs:
            return out
        if isinstance(subs, (list, tuple)):
            for it in subs:
                if not isinstance(it, dict):
                    continue
                name = str(it.get("name", "")).strip().lower()
                if not name:
                    continue
                sc = float(it.get("score", 0.0))
                if sc > 0:
                    out[name] = max(out.get(name, 0.0), sc)
        return out

    def _jaccard(self, a: set, b: set) -> float:
        if not a and not b:
            return 0.0
        inter = len(a & b)
        union = len(a | b) or 1
        return float(inter / union)

    def _weighted_jaccard(self, a: Dict[str, float], b: Dict[str, float]) -> float:
        if not a and not b:
            return 0.0
        keys = set(a.keys()) | set(b.keys())
        num = 0.0
        den = 0.0
        for k in keys:
            va = float(a.get(k, 0.0))
            vb = float(b.get(k, 0.0))
            num += min(va, vb)
            den += max(va, vb)
        if den <= 1e-12:
            return 0.0
        return float(num / den)

    def _weighted_mean(self, xs: List[float], ws: List[float]) -> float:
        if not xs or not ws or len(xs) != len(ws):
            return 0.0
        wsum = float(sum(ws)) or 1.0
        return float(sum(x * w for x, w in zip(xs, ws)) / wsum)

    def _top_keywords(self, members: List[Dict[str, Any]], topk: int = 6) -> List[str]:
        freq: Dict[str, float] = {}
        for m in members:
            w = float(m.get("weight", 0.0))
            for kw in m.get("keywords") or []:
                key = str(kw).strip().lower()
                if not key:
                    continue
                freq[key] = freq.get(key, 0.0) + w
        items = sorted(freq.items(), key=lambda x: x[1], reverse=True)[: max(0, topk)]
        return [k for k, _ in items]

    def _top_sub_emotions(self, members: List[Dict[str, Any]], topk: int = 6) -> List[Dict[str, Any]]:
        accum: Dict[str, float] = {}
        for m in members:
            w = float(m.get("weight", 0.0))
            for se in m.get("sub_emotions") or []:
                name = str(se.get("name", "")).strip().lower()
                sc = float(se.get("score", 0.0))
                if not name or sc <= 0:
                    continue
                accum[name] = max(accum.get(name, 0.0), sc * w)
        items = sorted(accum.items(), key=lambda x: x[1], reverse=True)[: max(0, topk)]
        return [{"name": n, "score": round(float(s), 6)} for n, s in items]


# =============================================================================
# EmbeddingGenerator
# =============================================================================
class EmbeddingGenerator:
    """라벨링 근거를 우선으로 전역 D차원 감정 임베딩을 결정론적으로 생성 + 앵커 정렬 지원."""

    def __init__(
            self,
            cfg,
            embed_dim: int | None = None,
            w_score: float = 0.60,
            w_kw: float = 0.25,
            w_sub: float = 0.15,
    ):
        self.cfg = cfg
        self.dim = int(embed_dim if embed_dim is not None else getattr(cfg, "embed_dim", 768))
        self.dim = max(64, self.dim)
        s = max(1e-9, w_score + w_kw + w_sub)
        self.w_score = float(w_score / s)
        self.w_kw = float(w_kw / s)
        self.w_sub = float(w_sub / s)

        # 장치 표기(수치계산은 numpy로 수행하지만 메타에는 장치 표기)
        try:
            use_cuda = bool(getattr(cfg, "device", None) == "cuda")
            has_cuda = bool("_HAS_TORCH" in globals() and globals()["_HAS_TORCH"] and "torch" in globals() and getattr(
                globals()["torch"], "cuda", None) and globals()["torch"].cuda.is_available())
            self.device = "cuda" if (use_cuda and has_cuda) else "cpu"
        except Exception:
            self.device = "cpu"

        # 섹션 분할(희/노/애/락)
        self.sections = self._split_sections(self.dim)

        # ── 앵커 정렬 상태 ────────────────────────────────────────────────
        self._A: Optional[np.ndarray] = None  # (d_out x d_in) 선형변환
        self._b: Optional[np.ndarray] = None  # (d_out,) 바이어스
        self._out_dim: int = self.dim  # 정렬 후 출력 차원(기본=내부 차원)
        self._align_method: Optional[str] = None
        self._fit_meta: Dict[str, Any] = {}

    # ------------------------------------------------------------------ #
    # Public: 기존 API (합성 임베딩 생성)
    # ------------------------------------------------------------------ #
    def create(self, detected: List[Dict[str, Any]]) -> Tuple[List[float], Dict[str, Any]]:
        """문서 레벨 감정 리스트(detected)를 받아 내부 임베딩(정렬 전)을 생성하고 메타를 반환."""
        dim = self.dim
        if not detected:
            zero = np.zeros(dim, dtype=np.float64)
            meta = {
                "confidence": 0.0,
                "metadata": {
                    "timestamp": self._now_iso(),
                    "detected_count": 0.0,
                    "device": self.device,
                    "dim": float(dim),
                    "sections": {k: [float(s), float(e)] for k, (s, e) in self.sections.items()},
                    "weights": {"score": self.w_score, "keyword": self.w_kw, "sub": self.w_sub},
                },
            }
            return zero.astype(float).tolist(), meta

        emb = np.zeros(dim, dtype=np.float64)

        # 맵 준비
        score_map = {d.get("emotion_id"): float(d.get("score", 0.0)) for d in detected if d.get("emotion_id")}
        conf_map = {d.get("emotion_id"): float(d.get("confidence", 0.0)) for d in detected if d.get("emotion_id")}
        kw_map = {d.get("emotion_id"): (d.get("keywords") or []) for d in detected if d.get("emotion_id")}
        subs_map = {d.get("emotion_id"): (d.get("sub_emotions") or []) for d in detected if d.get("emotion_id")}

        # 메인: 감정별 섹션 가중 합성
        for emo in detected:
            eid = emo.get("emotion_id")
            if not eid:
                continue
            score = float(emo.get("score", 0.0))
            if score <= 0:
                continue
            conf = float(emo.get("confidence", 0.0))
            kws = list((emo.get("keywords") or [])[:8])
            subs = list((emo.get("sub_emotions") or [])[:8])

            # 결정론적 난수 벡터(전역 seed 원천: eid + topK근거 + score)
            seed_src = self._seed_for(eid, score, kws, subs)
            base_vec = self._randn(seed_src, dim)
            base_vec = l2_normalize(base_vec)

            # 가중치: 근거 신뢰도 게이팅
            gate = 0.5 + 0.5 * max(0.0, min(1.0, conf))
            weight_score = self.w_score * score * gate

            # 키워드 보강(동일 섹션 내 미세 변조)
            kw_boost = np.zeros(dim, dtype=np.float64)
            if kws:
                m = min(8, len(kws))
                acc = np.zeros(dim, dtype=np.float64)
                for k in sorted(kws[:m]):
                    kv = self._randn(f"kw|{eid}|{k}", dim)
                    kv = l2_normalize(kv)
                    acc += kv
                kw_boost = l2_normalize(acc) if np.linalg.norm(acc) > 0 else acc
            weight_kw = self.w_kw * gate * min(1.0, len(kws) / 8.0) * (0.65 + 0.35 * score)

            # 서브감정 보강(점수 가중 평균)
            sub_boost = np.zeros(dim, dtype=np.float64)
            if subs:
                acc = np.zeros(dim, dtype=np.float64)
                wsum = 0.0
                for se in subs:
                    n = str(se.get("name", "")).strip()
                    sc = float(se.get("score", 0.0))
                    if not n or sc <= 0:
                        continue
                    sv = self._randn(f"sub|{eid}|{n}|{sc:.4f}", dim)
                    sv = l2_normalize(sv)
                    acc += sv * sc
                    wsum += sc
                if wsum > 0:
                    sub_boost = l2_normalize(acc)
            sub_cov = min(1.0, len([s for s in subs if float(s.get("score", 0.0)) > 0]) / 8.0) if subs else 0.0
            weight_sub = self.w_sub * gate * (0.5 + 0.5 * sub_cov) * (0.5 + 0.5 * score)

            # 섹션 합성(희/노/애/락 슬라이스만 갱신)
            s, e = self.sections.get(eid, (0, dim))
            emb[s:e] += base_vec[s:e] * weight_score
            if kw_boost is not None:
                emb[s:e] += kw_boost[s:e] * weight_kw
            if sub_boost is not None:
                emb[s:e] += sub_boost[s:e] * weight_sub

        # 정규화
        if np.linalg.norm(emb) > 0:
            emb = l2_normalize(emb)

        # 신뢰도 산정(근거량·분산·내부 confidence 통합)
        kw_cov_mean = float(np.mean([min(1.0, len(kw_map[eid]) / 8.0) for eid in kw_map])) if kw_map else 0.0
        sub_cov_mean = float(np.mean([min(1.0, len(subs_map[eid]) / 8.0) for eid in subs_map])) if subs_map else 0.0
        score_std = float(np.std([max(0.0, min(1.0, score_map[eid])) for eid in score_map])) if score_map else 0.0
        conf_mean_in = float(np.mean([max(0.0, min(1.0, conf_map[eid])) for eid in conf_map])) if conf_map else 0.0
        coverage = float(len([eid for eid, sc in score_map.items() if sc > 0.0])) / float(len(EMOTION_AXES))
        confidence = (
                0.22 + 0.30 * kw_cov_mean + 0.18 * sub_cov_mean + 0.20 * conf_mean_in + 0.10 * min(1.0, score_std * 3.0)
        )
        confidence *= (0.7 + 0.3 * coverage)
        confidence = float(max(0.0, min(1.0, round(confidence, 6))))

        meta = {
            "confidence": confidence,
            "metadata": {
                "timestamp": self._now_iso(),
                "detected_count": float(len([eid for eid, sc in score_map.items() if sc > 0.0])),
                "device": self.device,
                "dim": float(dim),
                "sections": {k: [float(s), float(e)] for k, (s, e) in self.sections.items()},
                "weights": {"score": self.w_score, "keyword": self.w_kw, "sub": self.w_sub},
                "coverage": round(coverage, 6),
                "kw_coverage_mean": round(kw_cov_mean, 6),
                "sub_coverage_mean": round(sub_cov_mean, 6),
                "score_std": round(score_std, 6),
                "input_confidence_mean": round(conf_mean_in, 6),
            },
        }
        return emb.astype(float).tolist(), meta

    # ------------------------------------------------------------------ #
    # NEW: encode(document_vector)
    # ------------------------------------------------------------------ #
    def encode(
            self,
            document_vector: Any,
            *,
            apply_alignment: bool = True,
            normalize: bool = True,
    ) -> List[float]:
        """
        문서 입력을 임베딩으로 인코딩한다.
        - document_vector가 list[dict] (detected emotions) → create() 경유
        - document_vector가 dict이며 'detected'/'detected_emotions' 키 보유 → 해당 값 사용
        - document_vector가 list[float] → 내부 벡터로 간주
        - alignment가 학습되어 있으면 A@x + b 적용
        """
        # 1) 내부(base) 임베딩 확보
        x = None

        if isinstance(document_vector, list) and document_vector and isinstance(document_vector[0], dict):
            x, _ = self.create(document_vector)
            x = np.asarray(x, dtype=np.float64)
        elif isinstance(document_vector, dict) and (
                "detected" in document_vector or "detected_emotions" in document_vector
        ):
            det = document_vector.get("detected") or document_vector.get("detected_emotions") or []
            x, _ = self.create(det)
            x = np.asarray(x, dtype=np.float64)
        elif isinstance(document_vector, (list, tuple, np.ndarray)) and (
                len(document_vector) > 0 and isinstance(document_vector[0], (int, float, np.floating))
        ):
            x = np.asarray(document_vector, dtype=np.float64)
        else:
            # 입력 형태를 알 수 없으면 빈 벡터 반환(호환성)
            x = np.zeros(self.dim, dtype=np.float64)

        # 차원 보정(내부 차원으로 패딩/컷)
        if x.ndim != 1:
            x = x.reshape(-1)
        if x.shape[0] < self.dim:
            x = np.pad(x, (0, self.dim - x.shape[0]))
        elif x.shape[0] > self.dim:
            x = x[: self.dim]

        if normalize and np.linalg.norm(x) > 0:
            x = l2_normalize(x)

        # 2) 앵커 정렬이 있으면 적용
        if apply_alignment and (self._A is not None):
            y = (self._A @ x) + (self._b if self._b is not None else 0.0)
            if normalize and np.linalg.norm(y) > 0:
                y = l2_normalize(y)
            return y.astype(float).tolist()

        # 정렬 없으면 내부(base) 벡터 반환
        return x.astype(float).tolist()

    # ------------------------------------------------------------------ #
    # NEW: fit(anchors) — 앵커 정렬 학습
    # ------------------------------------------------------------------ #
    def fit(
            self,
            anchors: Sequence[Any],
            *,
            method: str = "auto",  # "procrustes" | "ridge" | "auto"
            ridge: float = 1e-3,  # ridge 계수(Linear 모드)
            allow_scale: bool = True,  # Procrustes에서 scale 허용
    ) -> Dict[str, Any]:
        """
        anchors: 다음 중 하나의 시퀀스
          - [(input, target), ...]
          - [{"input": <detected/list/vec>, "target": <vec>}, ...]
        target: 외부 임베딩 벡터(list/np.ndarray)
        정렬 결과는 encode(..., apply_alignment=True)에서 자동 적용됨.
        """
        # 1) 데이터 적재
        X_list: List[np.ndarray] = []
        Y_list: List[np.ndarray] = []

        for item in anchors or []:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                inp, tgt = item[0], item[1]
            elif isinstance(item, dict):
                inp = item.get("input") or item.get("detected") or item.get("embedding")
                tgt = item.get("target") or item.get("external") or item.get("y")
            else:
                continue

            xi = np.asarray(self.encode(inp, apply_alignment=False, normalize=True), dtype=np.float64).reshape(1, -1)
            yi = np.asarray(tgt, dtype=np.float64).reshape(1, -1)
            if yi.size == 0 or xi.size == 0:
                continue
            X_list.append(xi)
            Y_list.append(yi)

        if not X_list:
            # 학습할 것이 없으면 정렬 해제
            self._A, self._b, self._out_dim, self._align_method, self._fit_meta = None, None, self.dim, None, {}
            return {"ok": False, "reason": "no_anchors"}

        X = np.vstack(X_list)  # (n x d_in)
        Y = np.vstack(Y_list)  # (n x d_out)
        n, d_in = X.shape
        d_out = Y.shape[1]

        # 2) 방법 선택
        chosen = method
        if method == "auto":
            # 같은 차원 & 샘플 수 >= 2 → Procrustes, 그 외 Ridge
            chosen = "procrustes" if (d_in == d_out and n >= 2) else "ridge"

        # 3) 변환 학습
        if chosen == "procrustes":
            # 중심화
            Xc = X - X.mean(axis=0, keepdims=True)
            Yc = Y - Y.mean(axis=0, keepdims=True)
            # Xc^T Yc = U Σ V^T
            M = Xc.T @ Yc
            U, S, Vt = np.linalg.svd(M, full_matrices=False)
            R = U @ Vt  # 회전
            if allow_scale:
                scale = (S.sum() / (np.square(np.linalg.norm(Xc, axis=1))).sum()) if np.linalg.norm(Xc) > 0 else 1.0
            else:
                scale = 1.0
            A = (scale * R)  # (d x d)
            b = (Y.mean(axis=0) - X.mean(axis=0) @ A).reshape(-1)  # (d,)
            # 훈련 오차
            pred = (X @ A) + b
            mse = float(np.mean((pred - Y) ** 2))
        else:
            # Ridge Linear: Y ≈ A X + b
            # X_aug = [X | 1], A_ext = argmin ||X_aug A_ext - Y||^2 + λ||A_ext||^2
            X_aug = np.hstack([X, np.ones((n, 1), dtype=np.float64)])
            I = np.eye(d_in + 1, dtype=np.float64)
            A_ext = np.linalg.pinv(X_aug.T @ X_aug + ridge * I) @ X_aug.T @ Y  # (d_in+1 x d_out)
            A = A_ext[:-1, :].T  # (d_out x d_in)
            b = A_ext[-1, :].reshape(-1)  # (d_out,)
            pred = (X @ A.T) + b
            mse = float(np.mean((pred - Y) ** 2))

        # 4) 상태 저장
        self._A = A if chosen == "procrustes" else A  # (d_out x d_in)
        self._b = b
        self._out_dim = d_out if chosen == "ridge" or (chosen == "procrustes" and d_out != d_in) else d_in
        self._align_method = chosen
        self._fit_meta = {
            "method": chosen,
            "n_anchors": int(n),
            "in_dim": int(d_in),
            "out_dim": int(d_out),
            "mse": round(mse, 8),
            "allow_scale": bool(allow_scale if chosen == "procrustes" else True),
        }
        return {"ok": True, **self._fit_meta}

    def reset_alignment(self) -> None:
        """학습된 앵커 정렬을 해제합니다."""
        self._A, self._b, self._out_dim, self._align_method, self._fit_meta = None, None, self.dim, None, {}

    # ------------------------------------------------------------------ #
    # Utils
    # ------------------------------------------------------------------ #
    def _split_sections(self, dim: int) -> Dict[str, Tuple[int, int]]:
        q = dim // 4
        r = dim % 4
        sizes = [q, q, q, q]
        for i in range(r):
            sizes[-(i + 1)] += 1
        bounds = []
        cur = 0
        for s in sizes:
            bounds.append((cur, cur + s))
            cur += s
        return {"희": bounds[0], "노": bounds[1], "애": bounds[2], "락": bounds[3]}

    def _seed_for(self, eid: str, score: float, kws: List[str], subs: List[Dict[str, Any]]) -> str:
        topk = ",".join(sorted([str(k) for k in (kws or [])][:6]))
        subs_names = ",".join(sorted([str(se.get("name", "")) for se in (subs or [])][:6]))
        return f"{eid}|{topk}|{subs_names}|{score:.4f}"

    def _randn(self, seed_src: str, n: int) -> np.ndarray:
        # 외부 제공 deterministic_rng가 있으면 사용
        try:
            r = deterministic_rng(seed_src)
            xs = np.array([r.random() for _ in range(max(2, n) * 2)], dtype=np.float64)
            xs = (xs - 0.5) * 2.0
            xs = (xs - xs.mean()) / (xs.std() + 1e-8)
            return xs[:n]
        except Exception:
            # 폴백(결정론 보장): config.seed + seed_src로 해시
            import hashlib, random as _rnd
            base_seed = int(getattr(self.cfg, "seed", 1337))
            h = int(hashlib.sha256((seed_src + f"|{base_seed}").encode("utf-8")).hexdigest()[:16], 16)
            rnd = _rnd.Random(h)
            xs = np.array([rnd.random() for _ in range(max(2, n) * 2)], dtype=np.float64)
            xs = (xs - 0.5) * 2.0
            xs = (xs - xs.mean()) / (xs.std() + 1e-8)
            return xs[:n]

    def _now_iso(self) -> str:
        import datetime as _dt
        return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


# =============================================================================
# QualityAssessor
# =============================================================================
class QualityAssessor:
    """복잡도·품질·변화량을 계산해 전역 신뢰도와 완성도를 평가."""

    def __init__(self, cfg, rel_strength_threshold: float = 0.08):
        self.cfg = cfg
        self.rel_strength_threshold = float(max(0.0, min(1.0, rel_strength_threshold)))

    # ---- helpers --------------------------------------------------------------
    def _clamp01(self, x: float) -> float:
        return 0.0 if x < 0.0 else 1.0 if x > 1.0 else float(x)

    def _safe_mean(self, xs: List[float]) -> float:
        return float(np.mean(xs)) if xs else 0.0

    def _norm_entropy01(self, scores: List[float]) -> float:
        p = np.array([max(0.0, float(v)) for v in scores], dtype=np.float64)
        s = float(np.sum(p))
        if s <= 0.0:
            return 0.0
        p = p / s
        ent = float(-np.sum(p * (np.log(p + 1e-12))))
        max_ent = math.log(max(1, len(p)))
        return 0.0 if max_ent <= 0 else float(ent / max_ent)

    # ---- complexity -----------------------------------------------------------
    def complexity(
            self,
            detected: List[Dict[str, Any]],
            relations: List[Dict[str, Any]],
            conflicts: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        # 커버리지
        min_sig = float(getattr(self.cfg, "min_signal_threshold", 0.05))
        coverage = len([d for d in detected if float(d.get("score", 0.0)) >= min_sig]) / float(len(EMOTION_AXES) or 1)

        # 관계 밀도(원시/유효)
        n = len([d for d in detected if float(d.get("score", 0.0)) > 0.0])
        max_edges = (n * (n - 1) / 2.0) if n > 1 else 1.0
        rel_count = len(relations)
        rel_density = float(min(rel_count / max_edges, 1.0)) if max_edges > 0 else 0.0
        strong_edges = [r for r in relations if float(r.get("strength", 0.0)) >= self.rel_strength_threshold]
        rel_density_eff = float(min(len(strong_edges) / max_edges, 1.0)) if max_edges > 0 else 0.0

        # 충돌 강도/커버리지
        conflict_intensity = self._safe_mean([float(c.get("severity", 0.0)) for c in conflicts]) if conflicts else 0.0
        conflict_edges = [r for r in relations if r.get("relationship_type") in ("conflict", "tension")]
        conflict_coverage = float(len(conflict_edges) / max(1, rel_count)) if rel_count > 0 else 0.0
        synergy_edges = [r for r in relations if r.get("relationship_type") == "synergy"]
        synergy_ratio = float(len(synergy_edges) / max(1, rel_count)) if rel_count > 0 else 0.0

        # 밸런스(엔트로피)와 지배 격차
        scores = [float(d.get("score", 0.0)) for d in detected] or [0.0] * len(EMOTION_AXES)
        balance_entropy = self._norm_entropy01(scores)
        top2 = sorted(scores, reverse=True)[:2] + [0.0, 0.0]
        dominance_gap = float(max(0.0, top2[0] - top2[1]))

        # 종합 복잡도
        w_cov, w_rel, w_conf, w_bal, w_dom = 0.27, 0.23, 0.27, 0.13, 0.10
        overall = (
                w_cov * coverage
                + w_rel * rel_density_eff
                + w_conf * conflict_intensity
                + w_bal * balance_entropy
                + w_dom * (1.0 - dominance_gap)
        )
        overall = self._clamp01(overall)

        return {
            "emotional_diversity": round(coverage, 6),
            "relation_density": round(rel_density, 6),
            "effective_relation_density": round(rel_density_eff, 6),
            "conflict_intensity": round(conflict_intensity, 6),
            "conflict_coverage": round(conflict_coverage, 6),
            "synergy_ratio": round(synergy_ratio, 6),
            "balance_entropy": round(balance_entropy, 6),
            "dominance_gap": round(dominance_gap, 6),
            "overall_complexity": round(overall, 6),
        }

    # ---- quality --------------------------------------------------------------
    def quality(
            self,
            detected: List[Dict[str, Any]],
            cm: Dict[str, Any],
            dominance: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        emo_conf = self._safe_mean([float(d.get("confidence", 0.0)) for d in detected]) if detected else 0.0
        relation_density = float(cm.get("effective_relation_density", cm.get("relation_density", 0.0)))
        coverage = float(cm.get("emotional_diversity", 0.0))

        # 분석 심도: 우세 감정의 영향력 분포(상위 k 평균)
        top_infl = sorted([float(x.get("influence_score", 0.0)) for x in dominance], reverse=True)[
            :max(1, len(EMOTION_AXES))]
        analysis_depth = min(1.0, self._safe_mean(top_infl) * 1.0)

        # 맥락 신뢰도: 관계망 밀도 기반 게이팅
        context_rel = min(1.0, 0.5 + 0.5 * relation_density)

        # 안정도: 감정 점수 표준편차의 역수
        scores = [float(d.get("score", 0.0)) for d in detected]
        std = float(np.std(scores)) if scores else 0.0
        stability = self._clamp01(1.0 - min(1.0, std * 2.0))

        # 종합 품질
        overall = emo_conf * 0.45 + context_rel * 0.30 + coverage * 0.25
        overall = self._clamp01(float(round(overall, 6)))

        # 경고 플래그
        flags: List[str] = []
        if coverage < 0.25:
            flags.append("low_coverage")
        if emo_conf < 0.30:
            flags.append("low_confidence")
        if float(cm.get("conflict_intensity", 0.0)) > 0.70:
            flags.append("high_conflict")
        if float(cm.get("dominance_gap", 0.0)) > 0.70:
            flags.append("single_emotion_dominates")
        if relation_density < 0.10:
            flags.append("sparse_relation_graph")

        return {
            "coverage": {"unique_emotions": len(detected), "emotion_coverage": round(coverage, 6)},
            "consistency": {
                "relation_density": round(relation_density, 6),
                "analysis_depth": round(analysis_depth, 6),
                "stability": round(stability, 6),
            },
            "reliability": {
                "emotion_confidence": round(emo_conf, 6),
                "context_reliability": round(context_rel, 6),
                "overall_quality": overall,
                "flags": flags,
            },
        }

    # ---- emotional change -----------------------------------------------------
    # QualityAssessor.emotional_changes 교체(핵심만)
    def emotional_changes(self, seg_results):
        if not seg_results or len(seg_results) < 2:
            return []
        out = []
        prev = {e: 0.0 for e in EMOTION_AXES}
        # 간단 지수평활
        alpha = 0.45  # 0.5~0.7 추천
        sm_prev = prev.copy()
        for i, seg in enumerate(seg_results):
            cur = prev.copy()
            for emo in seg.get("detected_emotions", []):
                cur[emo["emotion_id"]] = float(emo.get("score", 0.0))
            sm_cur = {}
            for e in EMOTION_AXES:
                sm_cur[e] = (1 - alpha) * sm_prev[e] + alpha * cur[e]
            if i > 0:
                delta = {}
                mag = 0.0
                for e in EMOTION_AXES:
                    d = sm_cur[e] - sm_prev[e]
                    # 과도 변화 클립(데모 안정화)
                    d = max(-0.25, min(0.25, d))
                    delta[e] = round(d, 6)
                    mag += abs(d)
                # 상위 변화 두 개
                leaders = sorted(
                    [{"emotion_id": e, "abs_change": abs(delta[e]),
                      "direction": ("up" if delta[e] > 0 else ("down" if delta[e] < 0 else "flat"))}
                     for e in EMOTION_AXES],
                    key=lambda x: x["abs_change"], reverse=True
                )[:2]
                out.append({"from": i - 1, "to": i, "delta": delta, "magnitude": round(mag, 6), "leaders": leaders})
            sm_prev = sm_cur
            prev = cur
        return out


# =============================================================================
# Calibration (Label-driven weighting)
# =============================================================================
@dataclass  # slots=True 는 환경에 따라 충돌 가능 → 우선 미사용 권장
class LabelStats:
    n_docs: int
    class_counts: Dict[str, int]  # 대분류(희/노/애/락)
    subclass_counts: Dict[str, int]  # "희/설렘" 형태의 세부감정 키
    df_pos: Dict[str, Dict[str, int]]  # 감정별 패턴의 양성 DF
    df_neg: Dict[str, Dict[str, int]]  # 감정별 패턴의 음성 DF


class WeightCalibrator:
    def __init__(self):
        self.prior_log_adj: Dict[str, float] = {}
        self.pattern_log_odds: Dict[Tuple[str, str], float] = {}

    def fit_from_snapshot(self, stats: LabelStats, alpha: float = 1.0, k: float = 0.5) -> "WeightCalibrator":
        # 1) prior 보정 Δ_e
        N = max(1, stats.n_docs)
        # 대/세부 모두 지원: 세부 우선, 없으면 대분류
        E = max(1, len(stats.subclass_counts) or len(stats.class_counts))
        counts = stats.subclass_counts or stats.class_counts
        for emo, c in counts.items():
            prior = (c + alpha) / (N + alpha * E)
            self.prior_log_adj[emo] = math.log(max(1e-9, (1.0 / E) / prior))
        # 2) 패턴 log-odds
        for emo, pos_map in (stats.df_pos or {}).items():
            neg_map = stats.df_neg.get(emo, {})
            for pat, dfp in pos_map.items():
                dfn = neg_map.get(pat, 0)
                w = math.log((dfp + k) / (dfn + k))
                self.pattern_log_odds[(emo, pat)] = w
        return self

    def get_prior_adj(self, emo: str) -> float:
        return self.prior_log_adj.get(emo, 0.0)

    def get_pattern_weight(self, emo: str, pat: str) -> float:
        return self.pattern_log_odds.get((emo, pat), 0.0)


# =============================================================================
# ConfidenceCalibrator
# =============================================================================
class ConfidenceCalibrator:
    """라벨링 근거·하위정서 분포·점수 강도·지속성 기반으로 감정별 confidence를 산출."""

    def __init__(self, cfg, rules: Optional[Dict[str, "EmotionRule"]] = None):
        self.cfg = cfg
        self.rules: Dict[str, "EmotionRule"] = rules or {}

        # 특성 가중치(합≈1)
        self.w_kw = 0.34  # 키워드 커버리지
        self.w_sub = 0.20  # 하위정서 다양성/균일도
        self.w_score = 0.30  # 점수 강도(자체 신호)
        self.w_persist = 0.10  # 세그먼트 지속성
        self.w_entropy = 0.06  # 하위정서 분포 엔트로피

        # 페널티 (B-3: 환경변수로 조정 가능)
        import os
        self.pen_no_evidence = float(os.getenv("CAL_PEN_NO_EVIDENCE", "0.18"))  # 증거 전무 시
        self.pen_low_kw = float(os.getenv("CAL_PEN_LOW_KW", "0.10"))  # 키워드 빈약
        self.pen_low_sub = 0.08  # 하위정서 빈약

        # 세그먼트 기반 보조통계(선택)
        self._persist_ratio: Dict[str, float] = {}  # eid -> (#present seg)/(#segs)
        self._persist_score: Dict[str, float] = {}  # eid -> 평균 score
        self._seg_count: int = 0

        # --- coverage lexicon caches (정규화/코어 사전) ---
        self._lexicon_full_norm_by_eid: Dict[str, set] = {}
        self._lexicon_core_2plus_by_eid: Dict[str, set] = {}
        try:
            self._on_rules_updated()
        except Exception:
            # 초기 규칙 부재 시 조용히 진행
            self._lexicon_full_norm_by_eid = {}
            self._lexicon_core_2plus_by_eid = {}

    # ------------------------------------------------------------
    # 세그먼트 통계 적재(선택 사용)
    # ------------------------------------------------------------
    def reset_segments(self) -> None:
        self._persist_ratio.clear()
        self._persist_score.clear()
        self._seg_count = 0

    def fit_segments(self, seg_results: List[Dict[str, Any]]) -> None:
        """세그먼트 레벨 지속성 통계를 준비."""
        self.reset_segments()
        if not seg_results:
            return
        min_sig = float(getattr(self.cfg, "min_signal_threshold", 0.05))
        present: Dict[str, int] = {e: 0 for e in EMOTION_AXES}
        score_sum: Dict[str, float] = {e: 0.0 for e in EMOTION_AXES}
        seg_n = len(seg_results)
        for seg in seg_results:
            seen_this_seg = set()
            for emo in seg.get("detected_emotions", []) or []:
                eid = emo.get("emotion_id")
                if not eid:
                    continue
                sc = float(emo.get("score", 0.0))
                score_sum[eid] += sc
                if sc >= min_sig and eid not in seen_this_seg:
                    present[eid] += 1
                    seen_this_seg.add(eid)
        for eid in EMOTION_AXES:
            self._persist_ratio[eid] = float(present[eid]) / float(seg_n or 1)
            cnt_for_avg = max(1, present[eid])  # 존재 세그먼트 기준 평균
            self._persist_score[eid] = float(score_sum[eid]) / float(cnt_for_avg)
        self._seg_count = seg_n

    # ------------------------------------------------------------
    # 내부 특성 계산
    # ------------------------------------------------------------
    def _clip01(self, x: float) -> float:
        return 0.0 if x < 0.0 else 1.0 if x > 1.0 else float(x)

    def _safe_len(self, x) -> int:
        try:
            return len(x)
        except Exception:
            return 0

    def _kw_coverage(self, eid: str, kw_hits: List[str]) -> float:
        """룰 키워드 대비 커버리지(한국어 친화 느슨 매칭).
        계층: (a) 정확일치 → (b) 코어일치 → (c) 부분일치(len≥2, 축키 제외)."""
        try:
            self._ensure_lexicon_caches()
            lex_full = self._lexicon_full_norm_by_eid.get(eid, set())
            lex_core = self._lexicon_core_2plus_by_eid.get(eid, set())

            seen_hits: set = set()
            for tok in (kw_hits or []):
                if not tok:
                    continue
                t_norm = _normalize_token_simple(tok)
                if not t_norm:
                    continue
                hit = False
                key = None
                # (a) 정확일치
                if t_norm in lex_full:
                    hit = True
                    key = t_norm
                else:
                    # (b) 코어일치
                    core = _normalize_token_core_for_coverage(t_norm)
                    if core in lex_full:
                        hit = True
                        key = core
                    # (c) 부분일치(서브스트링) – 길이≥2 & 축키 제외
                    if not hit and len(core) >= 2 and core not in EMOTION_AXES:
                        for lx in lex_core:
                            if len(lx) >= 2 and (lx in core or core in lx):
                                hit = True
                                key = core
                                break
                if hit and key:
                    seen_hits.add(key)

            uniq = len(seen_hits)
            rule_total = len(lex_full)
            # 룰이 있으면 룰 전체 대비, 없으면 경험적 상한(=10) 대비
            denom = rule_total if rule_total > 0 else 10
            cov = uniq / float(max(1, min(denom, 16)))
            # 완만한 포화: 1 - exp(-a*x)
            a = 2.2
            return self._clip01(1.0 - math.exp(-a * cov))
        except Exception:
            # 안전 폴백: 기존 보수적 방식
            uniq = len(set([str(k).lower() for k in (kw_hits or []) if k]))
            rule_total = self._safe_len(getattr(self.rules.get(eid, None), "keywords", []))
            denom = rule_total if rule_total > 0 else 10
            cov = uniq / float(max(1, min(denom, 16)))
            a = 2.2
            return self._clip01(1.0 - math.exp(-a * cov))

    # --- internal: lexicon caches ---
    def _ensure_lexicon_caches(self) -> None:
        if not self._lexicon_full_norm_by_eid or not self._lexicon_core_2plus_by_eid:
            self._on_rules_updated()

    def _on_rules_updated(self) -> None:
        full_map: Dict[str, set] = {}
        core_map: Dict[str, set] = {}
        for eid, rule in (self.rules or {}).items():
            toks = [t for t in (getattr(rule, "keywords", []) or []) if isinstance(t, str)]
            full = { _normalize_token_simple(t) for t in toks if _normalize_token_simple(t) }
            core = { _normalize_token_core_for_coverage(t) for t in full if len(_normalize_token_core_for_coverage(t)) >= 2 }
            # 길이 2 이상만 부분일치 후보로 사용
            core = {c for c in core if len(c) >= 2}
            full_map[eid] = full
            core_map[eid] = core
        self._lexicon_full_norm_by_eid = full_map
        self._lexicon_core_2plus_by_eid = core_map

    def _sub_diversity(self, sub_emotions: List[Dict[str, Any]]) -> Tuple[float, float]:
        """하위정서의 (고유수/균일도) -> (diversity, entropy_norm)"""
        if not sub_emotions:
            return 0.0, 0.0
        names = [str(se.get("name", "")).strip().lower() for se in sub_emotions if se.get("name")]
        uniq = len(set(names))
        diversity = self._clip01(uniq / 10.0)

        weights = [float(se.get("score", 0.0)) for se in sub_emotions]
        s = float(sum(max(0.0, w) for w in weights))
        if s <= 0.0:
            return diversity, 0.0
        p = np.array([max(0.0, w) / s for w in weights], dtype=np.float64)
        ent = float(-np.sum(p * (np.log(p + 1e-12))))
        max_ent = math.log(len(p)) if len(p) > 0 else 1.0
        ent_norm = self._clip01(ent / max(1e-12, max_ent))
        return diversity, ent_norm

    def _score_strength(self, score: float) -> float:
        """점수 강도의 소프트 스케일."""
        # sqrt로 완만히 포화(약한 점수도 조금 반영, 높은 점수는 과포화 방지)
        return self._clip01(math.sqrt(max(0.0, float(score))))

    def _persistence(self, eid: str) -> float:
        """세그먼트 지속성."""
        return self._clip01(float(self._persist_ratio.get(eid, 0.0)))

    # ------------------------------------------------------------
    # 메인 API
    # ------------------------------------------------------------
    def apply(self, detected: List[Dict[str, Any]], use_persistence: bool = True) -> None:
        """감정별 confidence 값을 산출해 제자리 갱신."""
        if not detected:
            return

        min_sig = float(getattr(self.cfg, "min_signal_threshold", 0.05))
        
        # B-3) 단문 게이팅 완화 토글
        import os
        if os.getenv("LENIENT_SHORT", "0") == "1":
            # 세그 수가 1~2개면 게이트 완화
            if self._seg_count <= 2:
                self.pen_no_evidence = min(self.pen_no_evidence, 0.10)
                self.pen_low_kw = min(self.pen_low_kw, 0.06)

        for d in detected:
            eid = d.get("emotion_id")
            if not eid:
                d["confidence"] = 0.0
                continue

            score = float(d.get("score", 0.0))
            kw_hits = d.get("keywords") or []
            subs = d.get("sub_emotions") or []

            kw_cov = self._kw_coverage(eid, kw_hits)
            sub_div, ent = self._sub_diversity(subs)
            strength = self._score_strength(score)
            persist = self._persistence(eid) if use_persistence else 0.0

            # 기본 스코어(가중 합)
            base = (
                    self.w_kw * kw_cov +
                    self.w_sub * sub_div +
                    self.w_score * strength +
                    self.w_persist * persist +
                    self.w_entropy * ent
            )

            # 페널티
            penalty = 0.0
            if self._safe_len(kw_hits) == 0 and self._safe_len(subs) == 0:
                penalty += self.pen_no_evidence
            if kw_cov < 0.20 and score < (min_sig * 2.0):
                penalty += self.pen_low_kw
            if sub_div < 0.20 and ent < 0.20:
                penalty += self.pen_low_sub

            # 게이팅(0~1)
            conf = base * (1.0 - self._clip01(penalty))

            # 경계 보정: 강한 점수/지속성인데 증거 부족인 경우 약간 보정
            if conf < 0.35 and (score >= 0.45 or persist >= 0.5):
                conf = min(0.55, conf + 0.08 + 0.12 * max(score, persist))

            d["confidence"] = round(self._clip01(conf), 6)

    # ------------------------------------------------------------
    # 배치 설명용(선택) - 디버그 시 도움
    # ------------------------------------------------------------
    def explain(self, detected: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """각 감정에 대한 confidence 기여요인(요약)을 반환."""
        out: List[Dict[str, Any]] = []
        for d in detected or []:
            eid = d.get("emotion_id", "")
            kw_hits = d.get("keywords") or []
            subs = d.get("sub_emotions") or []
            score = float(d.get("score", 0.0))
            kw_cov = self._kw_coverage(eid, kw_hits)
            sub_div, ent = self._sub_diversity(subs)
            strength = self._score_strength(score)
            persist = self._persistence(eid)
            out.append({
                "emotion_id": eid,
                "kw_coverage": round(kw_cov, 6),
                "sub_diversity": round(sub_div, 6),
                "sub_entropy": round(ent, 6),
                "score_strength": round(strength, 6),
                "persistence": round(persist, 6),
                "segment_count": self._seg_count,
            })
        return out


# =============================================================================
# ComplexEmotionAnalyzer (improved orchestrator)
# =============================================================================
class ComplexEmotionAnalyzer:
    """
    전처리 → 세분화 → 키워드 → 피처 → 세그먼트 스코어 → 문서 집계
    → 관계/충돌 → 지배/클러스터 → 임베딩 → 품질/변화 오케스트라.
    """

    def __init__(self, cfg: Optional[AnalyzerConfig] = None, rules: Optional[Dict[str, EmotionRule]] = None):
        # --- Config 기본값 보정 ---
        self.cfg = cfg or AnalyzerConfig()
        _defaults = dict(
            adaptive_mode=True,  # 동적 조정 on/off
            balance_strength=0.25,  # 0=없음, 1=완전균형 (실사용 권장 0.2~0.3)
            norm_method="auto",  # "auto"|"none"|"sqrt"|"log"
            # 내부 기본
            clip_score=6.0,
            score_min_floor=0.01,
            score_max_cap=0.99,
            auto_temperature=True,
            auto_min_signal=True,
            # --- Calibrator 관련 디폴트 ---
            use_calibrator=True,
            prior_strength=1.0,
            pattern_strength=1.0,
            pattern_mode="binary",  # "binary"|"count"
            max_pattern_contrib=3.0,
            epsilon=1e-6,
        )
        for k, v in _defaults.items():
            if not hasattr(self.cfg, k):
                setattr(self.cfg, k, v)

        # --- Rules ---
        self.rules = rules or RuleLoader(self.cfg.emotions_path).load()

        # --- Stages ---
        self.pre = Preprocessor()  # PreprocessConfig는 외부에서 override 가능
        self.seg = Segmenter()  # SegmenterConfig override 가능

        # KeywordMatcher
        self.kw = KeywordMatcher(
            self.rules,
            cfg=SimpleNamespace(
                include_sub_emotions_as_keywords=True,
                min_token_len=1,
                max_regex_len=200000,
                compile_flags=re.IGNORECASE,
                escape_regex=True,
                use_word_boundary_for_latin=True,
                allow_ko_suffix=True,
                ko_suffix_max=2,
                ignore_overlap=True,
                dedupe_hits=True,
                negation_window_chars=8,
                negation_tokens=["아니", "아닌", "아니다", "않", "않다", "않는", "못", "못하다", "못한", "없", "없다", "불가", "불가능", "불만",
                                 "불편", "불행", "비효율", "비정상", "비관", "비극", "무가치", "무의미", "무능", "무책임", "no", "not", "never"],
                phrase_weight=1.25,
                adaptive_mode=bool(getattr(self.cfg, "adaptive_mode", True)),
            ),
        )

        self.fe = FeatureExtractor()
        self.scorer = EmotionScorer(self.cfg, self.rules, feature_extractor=self.fe)
        self.agg = EvidenceAggregator(self.cfg)
        self.rels = RelationGraphBuilder(
            self.rules,
            edge_quantile=float(getattr(self.cfg, "graph_edge_quantile", 0.75)),
            target_keep_ratio=tuple(getattr(self.cfg, "graph_target_keep_ratio", (0.40, 0.60))),
            q_step=float(getattr(self.cfg, "graph_q_step", 0.05)),
            q_minmax=tuple(getattr(self.cfg, "graph_q_minmax", (0.30, 0.90))),
        )
        self.conf = ConflictResolver()
        self.rank = DominantEmotionRanker()
        self.clus = Clusterer()
        self.emb = EmbeddingGenerator(self.cfg)
        self.qa = QualityAssessor(self.cfg)
        self.cal = ConfidenceCalibrator(self.cfg, rules=self.rules)

        # --- Label-driven Weight Calibrator ---
        self.weight_calibrator = WeightCalibrator()

        # === NEW: 미리 캘리브레이터 적합(룰 스냅샷 기반) ===
        if getattr(self.cfg, "use_calibrator", True):
            try:
                # 1) 고급: 별도 헬퍼가 있다면 우선 사용
                if hasattr(self, "_build_label_stats_from_rules"):
                    stats = self._build_label_stats_from_rules(self.rules)  # LabelStats
                else:
                    # 2) 내장: 룰로부터 '중립' 스냅샷 생성
                    top_cats = list(self.rules.keys()) if self.rules else ["희", "노", "애", "락"]
                    class_counts: Dict[str, int] = {c: 1 for c in top_cats}
                    subclass_counts: Dict[str, int] = {}
                    for cat, rule in (self.rules or {}).items():
                        subs = getattr(rule, "sub_emotions", None) or []
                        for s in subs:
                            subclass_counts[f"{cat}/{s}"] = 1
                    n_docs = max(1, len(subclass_counts) or len(class_counts))
                    stats = LabelStats(
                        n_docs=n_docs,
                        class_counts=class_counts,
                        subclass_counts=subclass_counts,
                        df_pos={},  # k-스무딩으로 0도 안전
                        df_neg={},
                    )

                # 사전 확률/패턴 로그오즈 학습 (alpha,k는 보수적 기본)
                self.weight_calibrator.fit_from_snapshot(stats, alpha=1.0, k=0.5)
                # P1) IDF 활성화 - 서브감정 DF 주입: 스냅샷 보관
                self._label_stats = stats  # LabelStats 그대로 저장
                logger.debug("[Calibrator] Prefit complete with rule snapshot.")
            except Exception as e:
                logger.warning(f"[Calibrator] Prefit skipped: {e}")

        # (선택) Calibrator 중심 운용 시 과도한 이중 균형화 방지
        # if getattr(self.cfg, "use_calibrator", True) and self.cfg.balance_strength > 0.3:
        #     self.cfg.balance_strength = 0.3
        #     logger.debug("[Calibrator] balance_strength clamped to 0.3 for double-balance prevention.")

        # --- 채널 통계(초기값; 실제 서비스에서는 러닝 통계로 교체 권장) ---
        self.signal_stats = {
            "keyword_channel": {"mean": 0.15, "std": 0.08},
            "context_channel": {"mean": 0.25, "std": 0.12},
        }

    # === NEW: Calibrator 사전학습용 스냅샷 생성 ===
    def _build_label_stats_from_rules(self, rules: Dict[str, "EmotionRule"]) -> "LabelStats":
        """
        규칙만으로 만들 수 있는 '중립' 스냅샷.
        - subclass_counts: 각 대분류/서브감정을 모두 1로 둔 균등 분포
        - class_counts: 대분류도 1씩
        - df_pos/df_neg: 비워두거나(0) 필요시 키워드를 1로 채움
        """
        # 대분류 목록
        top_cats = list(rules.keys()) if rules else list(getattr(self, "EMOTION_AXES", [])) or ["희", "노", "애", "락"]
        class_counts = {c: 1 for c in top_cats}

        # 서브감정 키("희/설렘" 형태)
        subclass_counts: Dict[str, int] = {}
        for c in top_cats:
            subs = (rules.get(c).sub_emotions if (rules and rules.get(c)) else []) or []
            if subs:
                for s in subs:
                    subclass_counts[f"{c}/{s}"] = 1

        # 문서 수는 보수적으로 '클래스(또는 서브클래스) 개수'와 동일하게
        n_docs = max(1, len(subclass_counts) or len(class_counts))

        # 패턴 DF는 기본 0. 필요 시 규칙 키워드를 1로 두어 log-odds 0에 가깝게 유지 가능
        # (k-스무딩 덕분에 0으로 두어도 수치적으로 안전)
        df_pos: Dict[str, Dict[str, int]] = {}
        df_neg: Dict[str, Dict[str, int]] = {}

        # 선택) 서브감정별 키워드가 없다면 대분류 기준으로 DF를 채울 수도 있음
        # 여기서는 보수적으로 비워둡니다.

        return LabelStats(
            n_docs=n_docs,
            class_counts=class_counts,
            subclass_counts=subclass_counts,
            df_pos=df_pos,
            df_neg=df_neg,
        )

    # ------------------------------------------------------------------ #
    # 문서 단위 전체 분석(옵션: config 오버라이드, 디버그 모드)
    # ------------------------------------------------------------------ #
    def analyze_document(self, text: str, config: Optional[Dict[str, Any]] = None, debug: bool = False) -> Dict[
        str, Any]:
        if config:
            self._apply_overrides(config)

        timings: Dict[str, float] = {}
        _t0 = self._now()

        # 1) & 2) 전처리 및 세그먼트화 (기존과 동일)
        clean = self.pre.preprocess(text)
        timings["preprocess_ms"] = self._elapsed_ms(_t0)
        _t0 = self._now()
        segments = self.seg.segment(clean)
        timings["segment_ms"] = self._elapsed_ms(_t0)
        _t0 = self._now()

        # 3) 세그먼트별 스코어링
        seg_results: List[Dict[str, Any]] = []
        cat_to_subs = {eid: rule.sub_emotions for eid, rule in self.rules.items()}

        for i, seg in enumerate(segments):
            kw_counts, kw_hits = self.kw.match(seg)
            features = self.fe.extract(seg)  # Z-Score 결합을 위해 피처 추출을 앞으로 이동

            if debug:
                logger.debug(f"[KW_DEBUG] seg#{i} | counts={kw_counts} | hits={kw_hits}")

            detected_emotions: List[Dict[str, Any]]

            if self.cfg.use_calibrator:
                # --- 신규 데이터 기반 스코어링 파이프라인 ---
                # 1~3단계: 키워드 채널 점수 계산 (Calibrator 파이프라인)
                # P1) IDF 활성화 - 서브감정 DF 주입: df_map 및 N_docs 사용
                df_map = (getattr(self, "_label_stats", None).subclass_counts) if getattr(self, "_label_stats",
                                                                                          None) else {}
                N_docs = (getattr(self, "_label_stats", None).n_docs) if getattr(self, "_label_stats", None) else None

                sub_scores = {}
                for emo_key, hits in kw_hits.items():
                    if not hits:
                        continue
                    df_e = df_map.get(emo_key, None)  # "희/설렘" 형태 키
                    base = score_subemotion_base(len(hits), self.scorer._vocab_sizes.get(emo_key, 1), df_e, N_docs)
                    s_cal = apply_calibration(emo_key, base, Counter(hits), self.weight_calibrator, self.cfg)
                    sub_scores[emo_key] = s_cal

                cat_scores = aggregate_category(cat_to_subs, sub_scores)
                context_signals = {
                    "희": features["derived"]["positivity_hint"], "락": features["derived"]["positivity_hint"],
                    "노": features["derived"]["negativity_hint"], "애": features["derived"]["negativity_hint"]
                }

                # P3 즉시 반영: cat_scores와 context_signals 저장, detected_emotions는 나중에 재작성
                detected_emotions = []  # 임시 빈 배열

            else:  # Calibrator 미사용 시
                detected_emotions = self.scorer.score_segment(seg, kw_counts, kw_hits)

            # calibrator 사용 시 추가 데이터 저장
            seg_data = {
                "index": i, "text": seg, "features": features,
                "keyword_counts": kw_counts, "keyword_hits": kw_hits,
                "detected_emotions": detected_emotions
            }
            if self.cfg.use_calibrator:
                seg_data["cat_scores"] = cat_scores if 'cat_scores' in locals() else {}
                seg_data["context_signals"] = context_signals if 'context_signals' in locals() else {}

            seg_results.append(seg_data)
        timings["segment_scoring_ms"] = self._elapsed_ms(_t0)
        _t0 = self._now()

        # P3) Z-Score 채널 통계 동적화 - 문서 단위 러닝 통계 산출
        if self.cfg.use_zscore_fusion:
            kw_vals = []
            ctx_vals = []
            for seg in seg_results:
                # 키워드 채널 대용: 감정별 cat_scores 평균 또는 kw_summary.uniq_total 등
                kw_vals.append(sum(seg.get("cat_scores", {}).values()) / max(1, len(seg.get("cat_scores", {}))))
                # 맥락 채널: features["derived"]["positivity_hint"/"negativity_hint"] 평균
                d = seg["features"]["derived"]
                ctx_vals.append(0.5 * (d["positivity_hint"] + d["negativity_hint"]))

            def _mean_std(xs):
                import statistics as st
                m = st.fmean(xs) if xs else 0.0
                s = (st.pstdev(xs) if xs else 1.0) or 1.0
                return {"mean": float(m), "std": float(s)}

            self.signal_stats = {
                "keyword_channel": _mean_std(kw_vals),
                "context_channel": _mean_std(ctx_vals),
            }

            # P3 즉시 반영: 갱신된 signal_stats로 재융합하여 detected_emotions 재작성
            if self.cfg.use_calibrator:
                w_kw = self.cfg.z_channel_weights.get("keyword", 0.5)
                w_ctx = self.cfg.z_channel_weights.get("context", 0.5)

                for seg in seg_results:
                    if "cat_scores" in seg and "context_signals" in seg:
                        cat_scores = seg["cat_scores"]
                        ctx = seg["context_signals"]
                        fused = {}
                        for emo in EMOTION_AXES:
                            z_kw = z_standardize(cat_scores.get(emo, 0.0), self.signal_stats["keyword_channel"])
                            z_ctx = z_standardize(ctx.get(emo, 0.0), self.signal_stats["context_channel"])
                            z = (w_kw * z_kw + w_ctx * z_ctx) / max(1e-8, w_kw + w_ctx)
                            fused[emo] = sigmoid(z)

                        # 밸런스 → softmax → 임계 → 최종 세그 감정 재작성
                        scores_bal = self.scorer._apply_balance(fused, lam=0.0)
                        tau = self.scorer._get_temperature()
                        probs = self.scorer._softmax_temperature([scores_bal.get(e, 0.0) for e in EMOTION_AXES], tau)
                        prob_map = {eid: p for eid, p in zip(EMOTION_AXES, probs)}
                        min_sig = self.scorer.get_min_signal()

                        detected_emotions = []
                        for eid, p in prob_map.items():
                            if p >= min_sig:
                                subs = self.scorer._select_sub_emotions_cumprob(eid, p,
                                                                                seg["keyword_hits"].get(eid, []))
                                detected_emotions.append({
                                    "emotion_id": eid,
                                    "score": float(max(self.cfg.score_min_floor, min(self.cfg.score_max_cap, p))),
                                    "confidence": 0.0,
                                    "keywords": seg["keyword_hits"].get(eid, [])[:64],
                                    "sub_emotions": subs,
                                    "context": {}
                                })
                        seg["detected_emotions"] = detected_emotions

        # 4) 문서 레벨 집계 ~ 9) 최종 결과 조립 (기존과 동일)
        self.cal.fit_segments(seg_results)
        detected_doc = self.agg.aggregate_document(seg_results)
        self.cal.apply(detected_doc, use_persistence=True)
        timings["aggregate_calibrate_ms"] = self._elapsed_ms(_t0)
        _t0 = self._now()

        relations = self.rels.build(detected_doc)
        conflicts = self.conf.derive(detected_doc, relations)
        timings["relations_conflicts_ms"] = self._elapsed_ms(_t0)
        _t0 = self._now()

        dominance = self.rank.rank(detected_doc, relations)
        clusters = self.clus.cluster(detected_doc)
        timings["ranking_clustering_ms"] = self._elapsed_ms(_t0)
        _t0 = self._now()

        emb_vec, emb_meta = self.emb.create(detected_doc)
        timings["embedding_ms"] = self._elapsed_ms(_t0)
        _t0 = self._now()

        cm = self.qa.complexity(detected_doc, relations, conflicts)
        changes = self.qa.emotional_changes(seg_results)
        quality = self.qa.quality(detected_doc, cm, dominance)
        timings["quality_ms"] = self._elapsed_ms(_t0)

        calib_meta = self._build_calibration_meta()
        calib_meta["summary"] = self._calibration_summary_text(calib_meta)

        result: Dict[str, Any] = {
            "detected_emotions": detected_doc, "emotion_relations": relations,
            "emotional_conflicts": conflicts, "dominant_emotions": dominance,
            "emotion_clusters": clusters, "complexity_metrics": cm,
            "emotional_changes": changes,
            "embedding_info": {"embedding_vector": emb_vec, **emb_meta},
            "quality": quality, "meta": {"calibration": calib_meta}
        }
        if self.cfg.include_segments or debug:
            result["segments"] = seg_results
        if debug:
            result.setdefault("debug", {})
            result["debug"].update({
                "timings_ms": timings, "confidence_explain": self.cal.explain(detected_doc),
                "calibration_log": calib_meta, "config_snapshot": self._snapshot_config(),
            })

        return result

    # P2) 패턴 log-odds 실사용 - df_pos/df_neg 적재 훅
    def load_label_stats(self, path: str):
        """
        라벨 코퍼스 집계 결과(JSON)를 읽어 LabelStats에 넣어주는 유틸 함수.
        패턴 가중치가 실제 통계 기반으로 활성화됩니다.
        """
        import json
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        stats = LabelStats(
            n_docs=int(obj["n_docs"]),
            class_counts=obj["class_counts"],
            subclass_counts=obj["subclass_counts"],
            df_pos=obj.get("df_pos", {}),
            df_neg=obj.get("df_neg", {}),
        )
        self.weight_calibrator.fit_from_snapshot(stats, alpha=1.0, k=0.5)
        self._label_stats = stats

    # ---- helper: 한 줄 요약 문자열 ---------------------------------------
    def _calibration_summary_text(self, calib_meta: Dict[str, Any]) -> str:
        tau = calib_meta.get("used_temperature", None)
        src = calib_meta.get("temperature_source", None)
        q = None
        kept = None
        total = None
        eq = calib_meta.get("edge_quantiles") or {}
        if isinstance(eq, dict):
            q = eq.get("q_used", None)
            kept = eq.get("kept_edges", None)
            total = eq.get("total_candidates", None)
        parts = []
        parts.append(f"autoT:{'on' if calib_meta.get('auto_temperature') else 'off'}")
        if tau is not None:
            parts.append(f"tau:{tau:.2f}")
        if src:
            parts.append(f"src:{src}")
        if q is not None:
            parts.append(f"q:{q:.2f}")
        if kept is not None and total is not None:
            parts.append(f"edges:{kept}/{total}")
        return ", ".join(parts)

    # ------------------------------------------------------------------ #
    # 내부 유틸
    # ------------------------------------------------------------------ #
    def _build_calibration_meta(self) -> Dict[str, Any]:
        """
        동일 포맷으로 캘리브레이션 메타를 구성하는 헬퍼.
        - EvidenceAggregator가 softmax 직전에 기록한 실제 온도(_last_used_temperature)를 사용
        - RelationGraphBuilder가 저장한 분위수 로그(kept_edges/total_candidates 포함)를 그대로 노출
        - Scorer의 동적 임계값(있으면) 우선 사용
        """
        # --- Scorer: 사용된 최소 신호 임계 ---
        try:
            used_min_signal = float(self.scorer.get_min_signal())
            min_signal_strategy = "auto"
        except Exception:
            used_min_signal = float(getattr(self.cfg, "min_signal_threshold", 0.05))
            min_signal_strategy = "fixed"

        # --- Aggregator: softmax 실제 사용 온도(일원화) ---
        used_tau = getattr(self.agg, "_last_used_temperature", None)
        # (선택적) 소스 기록: 'config' | 'auto' | 'fallback' 등을 어그리게이터에서 남긴 경우
        tau_source = getattr(self.agg, "_last_temperature_source", None)

        # --- Relation graph: 마지막 분위수 로그(보강된 kept_edges/total_candidates 포함 가능) ---
        edge_q = getattr(self.rels, "_last_edge_quantiles", None)
        # 안전 캐스팅/정렬(있을 때만)
        if isinstance(edge_q, dict):
            edge_q = {
                k: (round(float(v), 6) if isinstance(v, (int, float)) else v)
                for k, v in edge_q.items()
            }

        # --- 스코어 캡/클립(참고용) ---
        score_floor = float(getattr(self.cfg, "score_min_floor", 0.01))
        score_cap = float(getattr(self.cfg, "score_max_cap", 0.99))
        clip_score = float(getattr(self.cfg, "clip_score", 6.0))

        return {
            # 동작 모드/정규화/균형
            "adaptive_mode": bool(getattr(self.cfg, "adaptive_mode", True)),
            "balance_strength": float(getattr(self.cfg, "balance_strength", 0.7)),
            "norm_method": str(getattr(self.cfg, "norm_method", "auto")),

            # 온도: 자동 여부/실제 사용값/소스
            "auto_temperature": bool(getattr(self.cfg, "auto_temperature", True)),
            "used_temperature": used_tau,
            "temperature_source": tau_source,  # 어그리게이터가 기록했을 때만 값이 존재

            # 스코어 임계(Scorer)
            "min_signal_threshold_used": used_min_signal,
            "min_signal_strategy": min_signal_strategy,

            # 관계 그래프 분위수 로그(분위수 컷 + 분포 + 유지 간선 수)
            # 예: {"q_used":0.5,"thr":0.123,"min":0.01,"median":0.08,"max":0.42,"kept_edges":4,"total_candidates":7,"kept_ratio":0.571}
            "edge_quantiles": edge_q,

            # 충돌 누적 컷(문서마다 유연 선택)
            "conflict_cum_ratio": 0.80,

            # 참고: 점수 캡/클립
            "score_caps": {"floor": score_floor, "cap": score_cap, "clip_score": clip_score},
        }

    def _calibration_log_snapshot(self) -> Dict[str, Any]:
        """
        디버그용 보조 로그. 현재는 calibration 메타를 그대로 반환하되
        확장 여지를 위해 함수로 분리.
        """
        return dict(self._build_calibration_meta())

    # 규칙/설정 동적 오버라이드(파일 경로·룰·서브컴포넌트 설정 포함)
    def _apply_overrides(self, overrides: Dict[str, Any]) -> None:
        if not overrides:
            return
        # AnalyzerConfig 필드 반영
        for k, v in list(overrides.items()):
            if hasattr(self.cfg, k):
                setattr(self.cfg, k, v)

        # include_segments 즉시 반영
        if "include_segments" in overrides:
            self.cfg.include_segments = bool(overrides["include_segments"])

        # emotions_path 지정 시 재로딩
        if "emotions_path" in overrides and overrides["emotions_path"] != getattr(self.cfg, "emotions_path", None):
            self.cfg.emotions_path = overrides["emotions_path"]
            self._reload_rules(RuleLoader(self.cfg.emotions_path).load())

        # rules 직접 주입
        if "rules" in overrides and overrides["rules"]:
            self._reload_rules(overrides["rules"])

        # 서브컴포넌트 설정(있으면 필드 단위로 갱신)
        sub_cfg_map = {
            "preprocess_config": getattr(self.pre, "cfg", None),
            "segmenter_config": getattr(self.seg, "cfg", None),
            "keyword_matcher_config": getattr(self.kw, "cfg", None),
            "feature_extractor_config": getattr(self.fe, "cfg", None),
        }
        for key, obj in sub_cfg_map.items():
            if key in overrides and isinstance(overrides[key], dict) and obj is not None:
                for fk, fv in overrides[key].items():
                    if hasattr(obj, fk):
                        setattr(obj, fk, fv)
                # 키워드매처는 규칙/옵션 변경 시 재컴파일
                if key == "keyword_matcher_config":
                    self.kw.update_rules(self.rules)
                    # 미니패치 2: 길이 정규화 임계값도 전파
                    kw_cfg = overrides["keyword_matcher_config"]
                    if "ln_auto_small" in kw_cfg:
                        self.kw.cfg.ln_auto_small = int(kw_cfg["ln_auto_small"])
                    if "ln_auto_medium" in kw_cfg:
                        self.kw.cfg.ln_auto_medium = int(kw_cfg["ln_auto_medium"])
                    
                # B-2) FeatureExtractor에 추가 토큰 주입
                if key == "feature_extractor_config":
                    fe_cfg = overrides["feature_extractor_config"]
                    if isinstance(fe_cfg.get("extra_boost", None), (list, tuple)):
                        self.fe._KO_BOOST |= set(map(str, fe_cfg["extra_boost"]))
                    if isinstance(fe_cfg.get("extra_hedge", None), (list, tuple)):
                        self.fe._KO_HEDGE |= set(map(str, fe_cfg["extra_hedge"]))
                    if isinstance(fe_cfg.get("extra_neg", None), (list, tuple)):
                        self.fe._KO_NEG |= set(map(str, fe_cfg["extra_neg"]))
                    
                    # 미니패치 3: 영문 토큰도 config로 확장 가능
                    if isinstance(fe_cfg.get("extra_boost_en", None), (list, tuple)):
                        self.fe._EN_BOOST |= set(map(lambda x: str(x).lower(), fe_cfg["extra_boost_en"]))
                    if isinstance(fe_cfg.get("extra_hedge_en", None), (list, tuple)):
                        self.fe._EN_HEDGE |= set(map(lambda x: str(x).lower(), fe_cfg["extra_hedge_en"]))
                    if isinstance(fe_cfg.get("extra_neg_en", None), (list, tuple)):
                        self.fe._EN_NEG |= set(map(lambda x: str(x).lower(), fe_cfg["extra_neg_en"]))
                    
                    # 미니패치 2: FeatureExtractor의 임계값도 KeywordMatcher에 전파
                    if "ln_auto_small" in fe_cfg and hasattr(self.kw, "cfg"):
                        self.kw.cfg.ln_auto_small = int(fe_cfg["ln_auto_small"])
                    if "ln_auto_medium" in fe_cfg and hasattr(self.kw, "cfg"):
                        self.kw.cfg.ln_auto_medium = int(fe_cfg["ln_auto_medium"])

        # adaptive_mode/balance_strength/norm_method 전파 보장
        if hasattr(self.kw, "cfg"):
            setattr(self.kw.cfg, "adaptive_mode", bool(getattr(self.cfg, "adaptive_mode", True)))
        if hasattr(self.scorer, "cfg"):
            self.scorer.cfg = self.cfg
        if hasattr(self.agg, "cfg"):
            self.agg.cfg = self.cfg

    # 규칙 재주입 및 종속 컴포넌트 갱신
    def _reload_rules(self, rules: Dict[str, EmotionRule]) -> None:
        self.rules = rules or {}
        self.kw.update_rules(self.rules)
        self.scorer.rules = self.rules
        self.rels.rules = self.rules
        self.cal.rules = self.rules
        # coverage lexicon caches 갱신(ConfidenceCalibrator)
        try:
            if hasattr(self.cal, "_on_rules_updated"):
                self.cal._on_rules_updated()
        except Exception:
            pass

        # === NEW: 룰이 바뀌면 캘리브레이터도 다시 맞춤 ===
        if getattr(self.cfg, "use_calibrator", True) and hasattr(self, "weight_calibrator"):
            try:
                stats = self._build_label_stats_from_rules(self.rules)
                self.weight_calibrator.fit_from_snapshot(stats, alpha=1.0, k=0.5)
                logger.debug("[Calibrator] Refit complete after rules reload.")
            except Exception as e:
                logger.warning(f"[Calibrator] Refit after reload skipped: {e}")

    # 구성 스냅샷
    def _snapshot_config(self) -> Dict[str, Any]:
        return {
            "analyzer": {
                "adaptive_mode": bool(getattr(self.cfg, "adaptive_mode", True)),
                "balance_strength": float(getattr(self.cfg, "balance_strength", 0.7)),
                "norm_method": str(getattr(self.cfg, "norm_method", "auto")),
                "score_temperature": float(getattr(self.cfg, "score_temperature", 0.75)),
                "score_min_floor": float(getattr(self.cfg, "score_min_floor", 0.01)),
                "score_max_cap": float(getattr(self.cfg, "score_max_cap", 0.99)),
                "min_signal_threshold": float(getattr(self.cfg, "min_signal_threshold", 0.05)),
                "auto_temperature": bool(getattr(self.cfg, "auto_temperature", True)),
                "auto_min_signal": bool(getattr(self.cfg, "auto_min_signal", True)),
                "device": getattr(self.cfg, "device", None),
                "seed": int(getattr(self.cfg, "seed", 1337)),
                "embed_dim": int(getattr(self.cfg, "embed_dim", 768)),
                "include_segments": bool(getattr(self.cfg, "include_segments", False)),
                "emotions_path": getattr(self.cfg, "emotions_path", None),
            },
            "preprocess": getattr(self.pre, "cfg", None).__dict__ if getattr(self.pre, "cfg", None) else {},
            "segmenter": getattr(self.seg, "cfg", None).__dict__ if getattr(self.seg, "cfg", None) else {},
            "keyword_matcher": getattr(self.kw, "cfg", None).__dict__ if getattr(self.kw, "cfg", None) else {},
            "feature_extractor": getattr(self.fe, "cfg", None).__dict__ if getattr(self.fe, "cfg", None) else {},
        }

    # 타이밍 유틸
    def _now(self) -> float:
        import time
        return time.perf_counter()

    def _elapsed_ms(self, t0: float) -> float:
        import time
        return round((time.perf_counter() - t0) * 1000.0, 3)

    def _sharpen_if_flat(self, detected_doc, tau: float = 0.55, rng_thr: float = 0.02) -> None:
        scores = [float(d.get("score", 0.0)) for d in (detected_doc or [])]
        if not scores:
            return
        rng = max(scores) - min(scores)
        if rng >= rng_thr:  # 충분히 차이나면 패스
            return
        # power transform: s' ∝ s^(1/tau)
        base = [max(1e-9, s) for s in scores]
        sharp = [s ** (1.0 / max(1e-6, tau)) for s in base]
        Z = sum(sharp) or 1.0
        new_scores = [x / Z for x in sharp]
        for d, s in zip(detected_doc, new_scores):
            d["score"] = float(s)


# =============================================================================
# Independent Functions (public, stable API)
# =============================================================================

def _empty_payload(reason: str = "error", meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    실패/조기종료 시에도 하위 파이프라인이 기대하는 최소 스키마를 보장하는 안전 페이로드.
    """
    payload = {
        "detected_emotions": [],
        "emotion_relations": [],
        "emotional_conflicts": [],
        "dominant_emotions": [],
        "emotion_clusters": [],
        "complexity_metrics": {
            "emotional_diversity": 0.0,
            "relation_density": 0.0,
            "effective_relation_density": 0.0,
            "conflict_intensity": 0.0,
            "conflict_coverage": 0.0,
            "synergy_ratio": 0.0,
            "balance_entropy": 0.0,
            "dominance_gap": 0.0,
            "overall_complexity": 0.0,
        },
        "emotional_changes": [],
        "embedding_info": {"embedding_vector": [], "confidence": 0.0, "metadata": {}},
        "quality": {
            "coverage": {"unique_emotions": 0, "emotion_coverage": 0.0},
            "consistency": {"relation_density": 0.0, "analysis_depth": 0.0, "stability": 0.0},
            "reliability": {"emotion_confidence": 0.0, "context_reliability": 0.0, "overall_quality": 0.0, "flags": ["empty_result", reason]},
        },
        "meta": {"error": reason},
    }
    if isinstance(meta, dict):
        payload["meta"].update(meta)
    return payload


def analyze_complex_emotions(
        text: str,
        emotions_data: Dict[str, Any] = None,  # ← 호환용
        config: Optional[Dict[str, Any]] = None,
        *,
        # ★★★ 이전 모듈 결과 수신 파라미터 추가 ★★★
        emotion_classification: Dict[str, Any] = None,
        prior_emotions: Dict[str, Any] = None,
        intensity_results: Dict[str, Any] = None,
        intensity: Dict[str, Any] = None,
        context_results: Dict[str, Any] = None,
        pattern_results: Dict[str, Any] = None,
        module_results: Dict[str, Any] = None,
        **kwargs,  # 추가 인자 무시
) -> Dict[str, Any]:
    """
    전체 분석 원샷 실행 (세그/그래프/충돌/랭킹/클러스터/임베딩/품질/변화).
    실패 시에도 최소 스키마를 보장합니다.
    
    ★ 이전 모듈의 emotion_classification 결과가 있으면 이를 우선 활용합니다.
    """
    # 빈/공백 입력 시 즉시 안전 폴백
    if not isinstance(text, str) or not text.strip():
        return _empty_payload("empty_input")
    
    # ★★★ 이전 모듈 결과에서 감정 분류 추출 ★★★
    prior_emo = emotion_classification or prior_emotions or {}
    intensity_data = intensity_results or intensity or {}
    
    try:
        analyzer = ComplexEmotionAnalyzer()
        res = analyzer.analyze_document(text, config or {})
        
        # 빈 dict가 들어오거나 핵심키가 없을 때도 최소 스키마 보정
        if not isinstance(res, dict) or not res.get("detected_emotions"):
            return _empty_payload("empty_result")
        
        # ★★★ 균등 분포(분석 실패) 감지 및 이전 결과 활용 ★★★
        detected = res.get("detected_emotions", [])
        cm = res.get("complexity_metrics", {})
        entropy = cm.get("balance_entropy", 0)
        
        # 균등 분포 판정: entropy > 0.95 또는 dominance_gap < 0.05
        is_uniform = entropy > 0.95 or cm.get("dominance_gap", 1) < 0.05
        
        # 키워드 매칭 여부 확인
        has_keywords = any(
            bool(emo.get("keywords")) 
            for emo in detected 
            if isinstance(emo, dict)
        )
        
        # ★ 균등 분포 + 키워드 없음 = 분석 실패 → 이전 결과 활용 시도
        if is_uniform and not has_keywords:
            logger.info("[analyze_complex_emotions] 균등 분포 감지 - 이전 모듈 결과 활용 시도")
            
            # ★★★ 여러 소스에서 main_distribution 탐색 ★★★
            main_dist = None
            all_results = module_results or {}
            
            # 1순위: weight_calculator 결과
            wc = all_results.get("weight_calculator", {}) or {}
            if isinstance(wc, dict):
                main_dist = wc.get("main_distribution") or wc.get("distribution") or wc.get("main_dist")
                if main_dist:
                    logger.info(f"[analyze_complex_emotions] weight_calculator에서 main_dist 발견: {main_dist}")
            
            # 2순위: intensity_analysis 결과
            if not main_dist:
                ia = all_results.get("intensity_analysis", {}) or {}
                if isinstance(ia, dict):
                    main_dist = ia.get("main_distribution") or ia.get("distribution") or ia.get("main_dist")
                    if main_dist:
                        logger.info(f"[analyze_complex_emotions] intensity_analysis에서 main_dist 발견: {main_dist}")
            
            # 3순위: emotion_classification 결과
            if not main_dist and prior_emo:
                main_dist = prior_emo.get("main_distribution") or prior_emo.get("distribution") or {}
                if not main_dist:
                    router = prior_emo.get("router", {})
                    main_dist = router.get("main_dist") or router.get("distribution") or {}
                if main_dist:
                    logger.info(f"[analyze_complex_emotions] emotion_classification에서 main_dist 발견: {main_dist}")
            
            # 4순위: context_analysis 결과
            if not main_dist:
                ctx = all_results.get("context_analysis", {}) or {}
                if isinstance(ctx, dict):
                    main_dist = ctx.get("main_distribution") or ctx.get("distribution") or ctx.get("emotion_scores")
                    if main_dist:
                        logger.info(f"[analyze_complex_emotions] context_analysis에서 main_dist 발견: {main_dist}")
            
            if not main_dist:
                logger.warning("[analyze_complex_emotions] 어떤 모듈에서도 main_distribution을 찾지 못함")
            
            if main_dist and isinstance(main_dist, dict):
                # 이전 결과로 detected_emotions 재구성
                new_detected = []
                for emo_id in EMOTION_AXES:
                    score = float(main_dist.get(emo_id, 0))
                    if score > 0.01:  # 1% 이상만 포함
                        new_detected.append({
                            "emotion_id": emo_id,
                            "score": score,
                            "confidence": 0.6,  # 이전 결과 기반이므로 중간 신뢰도
                            "keywords": [],
                            "sub_emotions": [],
                            "context": {"source": "prior_emotion_classification"},
                            "span": {"first_index": 0, "last_index": 0, "coverage_ratio": 1}
                        })
                
                if new_detected:
                    # 점수 순 정렬
                    new_detected.sort(key=lambda x: x["score"], reverse=True)
                    res["detected_emotions"] = new_detected
                    res["_analysis_source"] = "prior_emotion_classification"
                    
                    # complexity_metrics 재계산
                    scores = [e["score"] for e in new_detected]
                    if len(scores) >= 2:
                        import math
                        total = sum(scores)
                        probs = [s/total for s in scores] if total > 0 else scores
                        new_entropy = -sum(p * math.log(p + 1e-10) for p in probs if p > 0) / math.log(len(probs))
                        res["complexity_metrics"]["balance_entropy"] = new_entropy
                        res["complexity_metrics"]["dominance_gap"] = max(scores) - min(scores) if scores else 0
                    
                    logger.info(f"[analyze_complex_emotions] 이전 결과로 대체됨: {len(new_detected)}개 감정")
        
        return res
    except Exception as e:
        logger.error(f"[analyze_complex_emotions] 오류: {e}", exc_info=True)
        return _empty_payload("exception", {"exception": str(e)})


def generate_emotion_embedding(
        text: str,
        emotions_data: Dict[str, Any] = None,  # ← 호환용, 미사용(유지)
        config: Optional[Dict[str, Any]] = None,
):
    """
    문서 임베딩을 생성합니다.
    - torch 가 있으면 torch.Tensor, 없으면 list[float] 반환
    - config에 {"return_meta": True}를 주면 (vec, meta) 튜플을 반환합니다. ← 비호환 방지용 선택적 확장
    """
    # 빈/공백 입력 시 즉시 안전 폴백(임베딩 0 벡터)
    try:
        emb_size = 768
        if isinstance(config, dict):
            emb_size = int(config.get("embed_dim") or config.get("embedding_size") or emb_size)
    except Exception:
        emb_size = 768
    want_meta = bool(isinstance(config, dict) and config.get("return_meta"))
    if not isinstance(text, str) or not text.strip():
        if '_HAS_TORCH' in globals() and globals()['_HAS_TORCH'] and torch is not None:
            z = torch.zeros(emb_size, dtype=torch.float32)
            return (z, {"error": "empty_input"}) if want_meta else z
        z = [0.0] * emb_size
        return (z, {"error": "empty_input"}) if want_meta else z
    try:
        analyzer = ComplexEmotionAnalyzer()
        cfg = dict(config or {})
        want_meta = bool(cfg.pop("return_meta", False))
        res = analyzer.analyze_document(text, cfg)
        emb = (res.get("embedding_info") or {}).get("embedding_vector")
        if not emb:
            detected = res.get("detected_emotions", []) or []
            emb, meta = analyzer.emb.create(detected)
        else:
            meta = (res.get("embedding_info") or {})
        if '_HAS_TORCH' in globals() and globals()['_HAS_TORCH'] and torch is not None:
            tensor = torch.tensor(emb, dtype=torch.float32)
            return (tensor, meta) if want_meta else tensor
        return (emb, meta) if want_meta else emb
    except Exception as e:
        logger.error(f"[generate_emotion_embedding] 오류: {e}", exc_info=True)
        # 안전 폴백
        emb_size = 768
        try:
            if isinstance(config, dict):
                emb_size = int(config.get("embed_dim") or config.get("embedding_size") or emb_size)
        except Exception:
            pass
        if '_HAS_TORCH' in globals() and globals()['_HAS_TORCH'] and torch is not None:
            z = torch.zeros(emb_size, dtype=torch.float32)
            return (z, {"error": str(e)}) if isinstance(config, dict) and config.get("return_meta") else z
        z = [0.0] * emb_size
        return (z, {"error": str(e)}) if isinstance(config, dict) and config.get("return_meta") else z


def analyze_emotion_patterns(
        text: str,
        emotions_data: Dict[str, Any] = None,  # ← 호환용, 미사용(유지)
        config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    세그먼트/키워드/서브감정 분포 등 패턴 중심 상세분석.
    """
    if not isinstance(text, str) or not text.strip():
        return _empty_payload("empty_input")
    cfg = dict(config or {})
    cfg["include_segments"] = True
    return analyze_complex_emotions(text, emotions_data, cfg)


def get_emotion_transitions(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """결과에서 감정 전이만 추출해 반환."""
    try:
        out: List[Dict[str, Any]] = []
        changes = (result or {}).get("emotional_changes") or []
        if changes:
            for ch in changes:
                out.append({
                    "from": ch.get("from"),
                    "to": ch.get("to"),
                    "magnitude": float(ch.get("magnitude", 0.0)),
                    "leaders": ch.get("leaders", []),
                    "delta": ch.get("delta", {}),
                })
            return out

        # ▼ 보수적 백오프 1: time_flow.detected_events 기반 전이 재구성
        tf = ((result or {}).get("time_flow") or {}).get("detected_events") or []
        for ev in tf:
            frm = ev.get("from") or ev.get("prev") or ev.get("from_emotion")
            to  = ev.get("to")   or ev.get("next") or ev.get("to_emotion")
            if frm and to and frm != to:
                out.append({
                    "from": frm,
                    "to": to,
                    "magnitude": float(ev.get("confidence", ev.get("change", 0.0))),
                    "leaders": ev.get("leaders", []),
                    "delta": ev.get("delta", {}),
                })

        # ▼ 보수적 백오프 2: cause_effect에서 최상위 쌍 근사 전이 생성
        if not out:
            for link in ((result or {}).get("cause_effect") or []):
                f = link.get("from_emotions") or {}
                t = link.get("to_emotions") or {}
                if isinstance(f, dict) and isinstance(t, dict) and f and t:
                    try:
                        frm = max(f, key=f.get)
                        to = max(t, key=t.get)
                    except Exception:
                        frm = to = None
                    if frm and to and frm != to:
                        out.append({
                            "from": frm,
                            "to": to,
                            "magnitude": float(link.get("combined_score", 0.0)),
                            "leaders": [],
                            "delta": {},
                        })
        return out
    except Exception as e:
        logger.error(f"[get_emotion_transitions] 오류: {e}", exc_info=True)
        return []


def analyze_emotional_conflicts(
        analysis_result: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """감정 충돌(대립/긴장)을 재계산하거나, 결과에서 추출해 반환."""
    try:
        if analysis_result.get("emotional_conflicts"):
            return analysis_result["emotional_conflicts"]
        analyzer = ComplexEmotionAnalyzer()
        detected = analysis_result.get("detected_emotions", []) or []
        relations = analysis_result.get("emotion_relations", []) or analyzer.rels.build(detected)
        return analyzer.conf.derive(detected, relations)
    except Exception as e:
        logger.error(f"[analyze_emotional_conflicts] 오류: {e}", exc_info=True)
        return []


def calculate_emotion_complexity(
        analysis_result: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """복잡도 메트릭만 따로 계산/반환."""
    try:
        if analysis_result.get("complexity_metrics"):
            return analysis_result["complexity_metrics"]
        analyzer = ComplexEmotionAnalyzer()
        detected = analysis_result.get("detected_emotions", []) or []
        relations = analysis_result.get("emotion_relations", []) or analyzer.rels.build(detected)
        conflicts = analysis_result.get("emotional_conflicts", []) or analyzer.conf.derive(detected, relations)
        return analyzer.qa.complexity(detected, relations, conflicts)
    except Exception as e:
        logger.error(f"[calculate_emotion_complexity] 오류: {e}", exc_info=True)
        # 최소 스키마 유지
        return {
            "emotional_diversity": 0.0,
            "relation_density": 0.0,
            "effective_relation_density": 0.0,
            "conflict_intensity": 0.0,
            "conflict_coverage": 0.0,
            "synergy_ratio": 0.0,
            "balance_entropy": 0.0,
            "dominance_gap": 0.0,
            "overall_complexity": 0.0,
        }


def identify_dominant_emotions(
        analysis_result: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """지배 감정(랭킹 + role)을 재계산하거나 결과에서 추출."""
    try:
        if analysis_result.get("dominant_emotions"):
            return analysis_result["dominant_emotions"]
        analyzer = ComplexEmotionAnalyzer()
        detected = analysis_result.get("detected_emotions", []) or []
        relations = analysis_result.get("emotion_relations", []) or analyzer.rels.build(detected)
        return analyzer.rank.rank(detected, relations)
    except Exception as e:
        logger.error(f"[identify_dominant_emotions] 오류: {e}", exc_info=True)
        return []


def get_emotional_changes(
        text: str,
        analysis_result: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """텍스트의 시계열 전이(세그 기반). 결과에 없으면 필요 최소한으로 재계산."""
    if not isinstance(text, str) or not text.strip():
        return []
    try:
        if analysis_result.get("emotional_changes"):
            return analysis_result["emotional_changes"]
        analyzer = ComplexEmotionAnalyzer()
        if analysis_result.get("segments"):
            return analyzer.qa.emotional_changes(analysis_result["segments"])
        cfg = dict(config or {}); cfg["include_segments"] = True
        res = analyzer.analyze_document(text, cfg)
        return res.get("emotional_changes", [])
    except Exception as e:
        logger.error(f"[get_emotional_changes] 오류: {e}", exc_info=True)
        return []



# =============================================================================
# CLI 유틸
# =============================================================================
def _timestamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")

def _default_emotions_path() -> str:
    """
    EMOTIONS.json/JOSN 후보 경로 중 존재하는 첫 경로 리턴.
    존재하지 않아도 분석은 진행 가능(룰 로더가 디폴트 사용).
    """
    cands = [
        os.environ.get("EMOTIONS_JSON_PATH"),
        str(_SCRIPT_DIR / "EMOTIONS.json"),
        str(_SCRIPT_DIR / "EMOTIONS.JSON"),
        str((_SCRIPT_DIR / "data" / "EMOTIONS.json")),
        str((_SCRIPT_DIR.parent / "EMOTIONS.json")),
    ]
    for p in cands:
        if p and os.path.exists(p):
            return p
    # 마지막 후보(존재할 수도, 없을 수도) 반환: 경고만 찍고 진행
    return cands[1]  # _SCRIPT_DIR / "EMOTIONS.json"

def _read_text_from_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def _read_text_from_stdin() -> str:
    return sys.stdin.read()

def _demo_text() -> str:
    return (
        "오늘은 기쁘면서도 왠지 모르게 불안함이 스며드는 날이네요. "
        "기분이 들뜨다가도 한편으론 걱정돼서 마음이 복잡해요. "
        "끝내고 나니 안도감도 느껴집니다."
    )


# =============================================================================
# 기존 객체/타입 호환을 위한 안전 인코더
# =============================================================================
class _CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        try:
            from typing import get_type_hints  # noqa
        except Exception:
            pass
        # EmotionRelation 같은 커스텀 객체 -> dict 변환 시도
        for attr_set in (("to_dict",), ("__dict__",)):
            for attr in attr_set:
                if hasattr(obj, attr):
                    try:
                        if callable(getattr(obj, "to_dict", None)):
                            return obj.to_dict()
                        return dict(obj.__dict__)
                    except Exception:
                        break
        # torch 텐서
        if _HAS_TORCH and "torch" in str(type(obj)):
            try:
                return obj.tolist()
            except Exception:
                return str(obj)
        # datetime
        if isinstance(obj, _dt.datetime):
            return obj.isoformat()
        return super().default(obj)


# =============================================================================
# 결과 표시/저장
# =============================================================================
def _print_analysis_results(results: dict) -> None:
    print("=" * 56)
    print("[복합 감정 분석 결과]\n")

    detected = results.get("detected_emotions") or []
    print("1) 감지된 감정(detected_emotions):")
    if detected:
        for idx, d in enumerate(detected, 1):
            if isinstance(d, dict):
                eid = d.get("emotion_id", "")
                score = float(d.get("score", 0.0) or 0.0)
                conf = float(d.get("confidence", score) or score)
            else:
                eid = getattr(d, "emotion_id", "")
                score = float(getattr(d, "score", 0.0) or 0.0)
                conf = float(getattr(d, "confidence", score) or score)
            print(f"   {idx}. Emotion ID: {eid}, Score: {score:.3f}, Confidence: {conf:.3f}")
    else:
        print("   감지된 감정이 없습니다.")
    print()

    print("2) 감정 간 관계(emotion_relations):")
    rels = results.get("emotion_relations") or []

    def _to_rel_dict(rel):
        if isinstance(rel, dict):
            return rel
        return {
            "source": getattr(rel, "source", None),
            "target": getattr(rel, "target", None),
            "relationship_type": getattr(rel, "relationship_type", getattr(rel, "type", "unknown")),
            "strength": float(getattr(rel, "strength", 0.0) or 0.0),
            "compatibility": float(getattr(rel, "compatibility", 0.0) or 0.0),
        }

    if rels:
        for i, rel in enumerate([_to_rel_dict(r) for r in rels], 1):
            print(
                f"   {i}. {rel.get('source')} → {rel.get('target')}, "
                f"type: {rel.get('relationship_type')}, "
                f"strength: {float(rel.get('strength', 0.0)):.3f}, "
                f"compat: {float(rel.get('compatibility', 0.0)):.3f}"
            )
    else:
        print("   감정 관계 데이터가 없습니다.")
    print()

    print("3) 감정 충돌(emotional_conflicts):")
    conflicts = results.get("emotional_conflicts") or []
    if conflicts:
        for i, c in enumerate(conflicts, 1):
            ems = c.get("emotions", [])
            ctype = c.get("type", "unknown")
            sev = float(c.get("severity", 0.0) or 0.0)
            inten = float(c.get("intensity", 0.0) or 0.0)
            print(f"   {i}. Emotions: {ems}, Type: {ctype}, Severity: {sev:.3f}, Intensity: {inten:.3f}")
    else:
        print("   감정 충돌이 감지되지 않았습니다.")
    print()

    print("4) 복잡도 메트릭(complexity_metrics):")
    cm = results.get("complexity_metrics") or {}
    if cm:
        for k, v in cm.items():
            try:
                print(f"   - {k}: {float(v):.3f}")
            except Exception:
                print(f"   - {k}: {v}")
    else:
        print("   복잡도 메트릭이 없습니다.")
    print("=" * 56)


def _save_analysis_results(results: dict, out_path: str = None) -> str:
    """
    결과 JSON 저장. out_path 없으면 logs/complex_analyzer_타임스탬프.json 로 저장.
    """
    if out_path:
        out_file = Path(out_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_file = _LOG_DIR / f"complex_analyzer_{_timestamp()}.json"
    try:
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2, cls=_CustomJSONEncoder)
        logger.info(f"분석 결과 저장: {out_file}")
    except Exception as e:
        logger.error(f"분석 결과 저장 실패: {e}")
    return str(out_file)


# =============================================================================
# argparse
# =============================================================================
def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Complex Emotion Analyzer — 문서 전반 감정/관계/동역학 종합 분석"
    )
    src = p.add_argument_group("입력 소스")
    src.add_argument("--input", type=str, help="분석할 텍스트 파일 경로")
    src.add_argument("--stdin", action="store_true", help="표준입력으로부터 읽기 (파이프/리다이렉션)")
    src.add_argument("--text", type=str, help="직접 전달한 문자열을 분석")
    src.add_argument("--demo", action="store_true", help="내장 데모 텍스트 사용(옵션 생략 시에도 기본으로 사용)")

    conf = p.add_argument_group("모델 설정")
    conf.add_argument("--emotions", type=str, default=None, help="EMOTIONS.json 경로 (없으면 자동 탐색)")
    conf.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"], help="연산 디바이스")
    conf.add_argument("--seed", type=int, default=1337, help="결정론 시드")
    conf.add_argument("--temp", type=float, default=0.75, help="temperature")
    conf.add_argument("--min-signal", type=float, default=0.05, help="감정 최소 신호 임계치")
    conf.add_argument("--embed-dim", type=int, default=768, help="임베딩 차원")
    conf.add_argument("--include-segments", action="store_true", help="세그먼트별 결과 포함")

    cal = p.add_argument_group("캘리브레이션/분화")
    cal.add_argument("--balance-strength", type=float, default=0.25,
                     help="대분류 균형 보정 강도(0=무효, 1=완전균등). 실사용 권장: 0.2~0.3")

    graph = p.add_argument_group("그래프 컷/피드백")
    graph.add_argument("--edge-q", type=float, default=0.75,
                       help="관계 간선 컷 분위수(0~1). 높을수록 약한 간선 제거")
    graph.add_argument("--edge-q-min", type=float, default=0.30, help="q 하한")
    graph.add_argument("--edge-q-max", type=float, default=0.90, help="q 상한")
    graph.add_argument("--edge-q-step", type=float, default=0.05, help="q 피드백 보폭")
    graph.add_argument("--edge-keep-lo", type=float, default=0.40, help="목표 간선 유지비 하한")
    graph.add_argument("--edge-keep-hi", type=float, default=0.60, help="목표 간선 유지비 상한")

    out = p.add_argument_group("출력/로깅")
    out.add_argument("--out", type=str, default=None, help="결과 JSON 저장 경로(기본: logs/complex_analyzer_타임스탬프.json)")
    out.add_argument("--log-dir", type=str, default=None, help="로그/기본 결과 저장 폴더(기본: 스크립트 하위 logs)")
    out.add_argument("--debug", action="store_true", help="콘솔 로깅을 DEBUG로")

    return p.parse_args(argv)


def _apply_log_dir_override(args) -> None:
    global _LOG_DIR
    if args.log_dir:
        new_dir = Path(args.log_dir)
        new_dir.mkdir(parents=True, exist_ok=True)
        _LOG_DIR = new_dir
        # 파일 핸들러 경로만 교체
        for h in list(logger.handlers):
            if isinstance(h, RotatingFileHandler):
                logger.removeHandler(h)
        fh = RotatingFileHandler(
            _LOG_DIR / "complex_analyzer.log",
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        fh.setLevel(logging.DEBUG)
        fmt = logging.Formatter(
            "[%(asctime)s] %(levelname)s [%(filename)s:%(lineno)d] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.info(f"로그 디렉토리 변경: {_LOG_DIR}")


def _resolve_text(args) -> tuple[str, str]:
    """
    (source, text) 반환. source는 'text'/'file'/'stdin'/'demo' 중 하나.
    아무 입력도 없으면 demo로 자동 전환.
    """
    # 우선순위: --text > --input > --stdin/파이프 > --demo > 자동 demo
    # 파이프 여부
    piped = not sys.stdin.isatty()
    if args.text:
        return "text", args.text
    if args.input:
        if not os.path.exists(args.input):
            logger.error(f"입력 파일을 찾을 수 없습니다: {args.input}")
            raise FileNotFoundError(args.input)
        return "file", _read_text_from_file(args.input)
    if args.stdin or piped:
        return "stdin", _read_text_from_stdin()
    # 명시적 데모 혹은 자동 데모
    return "demo", _demo_text()


def debug_rules_compat_matrix(rules):
    ids = EMOTION_AXES
    rows = []
    print("\n[DEBUG] Emotion Compatibility Matrix:")
    header = "       " + "   ".join(ids)
    print(header)
    print("    " + "-" * len(header))
    for a in ids:
        ra = rules.get(a)
        row_str = f" {a} |"
        for b in ids:
            if a == b:
                row_str += "  --  "
                continue
            c = 0.5
            rb = rules.get(b)
            if ra and b in (ra.incompatible_with or []):
                c = 0.15
            elif rb and a in (rb.incompatible_with or []):
                c = 0.15  # 양방향 확인
            elif ra and b in (ra.compatible_with or []):
                c = 0.70
            elif rb and a in (rb.compatible_with or []):
                c = 0.70  # 양방향 확인
            row_str += f" {c:.2f} "
        print(row_str)
    print("-" * len(header))
    print()


# =============================================================================
# Main
# =============================================================================
def main(argv=None) -> int:
    args = _parse_args(argv)

    # 로그 디렉토리 오버라이드
    _apply_log_dir_override(args)

    # 콘솔 로그 레벨 조정
    if args.debug:
        for h in logger.handlers:
            if isinstance(h, logging.StreamHandler):
                h.setLevel(logging.DEBUG)

    logger.info("Complex Emotion Analyzer 시작")

    # 입력 텍스트 결정(없으면 자동으로 demo 사용)
    try:
        source, text = _resolve_text(args)
    except Exception as e:
        logger.error(f"입력 소스 준비 실패: {e}")
        print("사용 예) --demo | --text \"문장\" | --input 파일 | 파이프/리다이렉션 또는 --stdin", file=sys.stderr)
        return 2

    # EMOTIONS 경로 결정(없어도 동작: RuleLoader가 디폴트 룰 사용)
    emo_path = args.emotions or _default_emotions_path()
    if not emo_path or not os.path.exists(emo_path):
        logger.warning(f"EMOTIONS 데이터셋을 찾지 못했습니다(진행은 가능): {emo_path}")

    # AnalyzerConfig 구성
    try:
        cfg = AnalyzerConfig(
            emotions_path=emo_path,
            score_temperature=float(args.temp),
            min_signal_threshold=float(args.min_signal),
            device=args.device or ("cuda" if (_HAS_TORCH and torch and torch.cuda.is_available()) else "cpu"),
            seed=int(args.seed),
        )
        cfg.use_calibrator = True
        cfg.prior_strength = 1.0
        cfg.pattern_strength = 1.0
        # CLI에서 넘어온 분화 강도를 채택
        try:
            cfg.balance_strength = float(args.balance_strength)
        except Exception:
            pass

        # 그래프 컷/피드백 파라미터 바인딩
        try:
            cfg.graph_edge_quantile = float(args.edge_q)
            cfg.graph_q_minmax = (float(args.edge_q_min), float(args.edge_q_max))
            cfg.graph_q_step = float(args.edge_q_step)
            cfg.graph_target_keep_ratio = (float(args.edge_keep_lo), float(args.edge_keep_hi))
        except Exception:
            pass

        try:
            setattr(cfg, "embed_dim", int(args.embed_dim))
        except Exception:
            pass
        try:
            setattr(cfg, "include_segments", bool(args.include_segments))
        except Exception:
            pass
    except Exception as e:
        logger.error(f"AnalyzerConfig 생성 실패: {e}")
        return 2

    # 분석 실행
    try:
        analyzer = ComplexEmotionAnalyzer(cfg)
        debug_rules_compat_matrix(analyzer.rules)
        results = analyzer.analyze_document(
            text,
            config={
                "auto_temperature": False,
                "score_temperature": 0.52,  # 0.50~0.55 권장
                "adaptive_mode": True,
                "norm_method": "log",
                # min_signal_threshold 및 balance_strength는 CLI/Config 값을 존중
            },
            debug=args.debug
        )
    except Exception as e:
        logger.error(f"분석 중 오류: {e}", exc_info=True)
        return 1

    # 콘솔 출력 + 파일 저장
    try:
        _print_analysis_results(results)
        out_path = args.out  # None이면 자동 경로 사용
        saved = _save_analysis_results(results, out_path)
        logger.info(f"[완료] 입력소스={source}, 저장={saved}")
    except Exception as e:
        logger.error(f"결과 처리 중 오류: {e}")

    # 자원 정리
    try:
        if _HAS_TORCH and torch and torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()
    return 0


# =============================================================================
# Entrypoint
# =============================================================================
if __name__ == "__main__":
    sys.exit(main())




