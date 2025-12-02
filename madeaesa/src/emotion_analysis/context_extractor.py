# context_extractor.py
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import json
import gc
import re
try:
    import psutil
except Exception:
    psutil = None
import copy
import math
import time
import logging
from pathlib import Path
from collections import Counter, defaultdict
from logging.handlers import RotatingFileHandler
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Dict, Any, List, Optional, Tuple, TypedDict, Set, Sequence, cast, Mapping, Union, Iterable, Pattern, DefaultDict
from dataclasses import dataclass, field


# =============================================================================
# Mini-Helper
# =============================================================================
SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+|\r?\n+')

try: import kss
except Exception: kss = None
try: from sklearn.feature_extraction.text import TfidfVectorizer
except Exception: TfidfVectorizer = None
def _split_korean_sentences_fallback(text: str) -> List[str]:
    if not text:
        return []
    parts = SENT_SPLIT_RE.split(text)
    return [s.strip() for s in parts if s and s.strip()]

def _split_into_sentences(text: str) -> List[str]:
    """한국어 특화 문장 분리 함수"""
    try:
        # UTF-8 인코딩 강제 설정
        import sys
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8')
        
        # 텍스트 정리
        if isinstance(text, bytes):
            text = text.decode('utf-8', errors='ignore')
        
        # 간단한 문장 분리 (KSS 의존성 제거)
        sentences = []
        # 한국어 문장부호 기준 분리
        parts = text.replace('!', '.').replace('?', '.').replace('ㅋㅋ', '.').replace('ㅎㅎ', '.').split('.')
        for part in parts:
            part = part.strip()
            if part and len(part) > 1:  # 너무 짧은 부분 제외
                sentences.append(part)
        
        return sentences if sentences else [text.strip()]
    except Exception as e:
        print(f"문장 분리 오류: {e}")
        return [text.strip()]


# =============================================================================
# Logger 설정
# =============================================================================
# 통합 로그 관리자 사용 (날짜별 폴더)
try:
    from log_manager import get_log_manager
    log_manager = get_log_manager()
    log_dir = str(log_manager.get_log_dir("emotion_analysis", use_date_folder=True))
except ImportError:
    # 폴백: 기존 방식 (날짜별 폴더 추가)
    base_log_dir = 'logs'
    today = datetime.now().strftime("%Y%m%d")
    log_dir = os.path.join(base_log_dir, today)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# 비간섭: 기본은 NullHandler만 부착
if not any(isinstance(h, logging.NullHandler) for h in logger.handlers):
    logger.addHandler(logging.NullHandler())

# 선택적 로깅(환경변수로만 활성화)
_fmt = logging.Formatter(
    '[%(asctime)s] %(levelname)s [%(filename)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
if os.environ.get("CTX_FILE_LOG", "0") == "1":
    try:
        os.makedirs(log_dir, exist_ok=True)
        fh = RotatingFileHandler(
            os.path.join(log_dir, 'context_extractor.log'),
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding='utf-8'
        )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(_fmt)
        logger.addHandler(fh)
    except Exception:
        pass
if os.environ.get("CTX_CONSOLE_LOG", "0") == "1":
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(_fmt)
    logger.addHandler(ch)

# 표준 스트림 UTF-8 재설정(콘솔/파이프 한글 깨짐 방지)
try:
    for _stream in (getattr(sys, "stdout", None), getattr(sys, "stderr", None), getattr(sys, "stdin", None)):
        if _stream is not None and hasattr(_stream, "reconfigure"):
            _stream.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class _WeightObj(TypedDict, total=False):
    weight: float
    score: float
    w: float
    label: str
    count: int
_WeightVal = Union[float, int, str, _WeightObj]

def _limit_situations_global(result: dict, k: int = 5) -> None:
    try:
        sa = (result.get("situation_impact_analysis") or {}).get("situation_analysis") or []
        sa = sorted(sa, key=lambda x: float(x.get("impact_score", 0.0)), reverse=True)
        (result.setdefault("situation_impact_analysis", {}))["situation_analysis"] = sa[:k]
    except Exception:
        pass

def _as_float_weight(v: _WeightVal) -> Optional[float]:
    if isinstance(v, (int, float)): return float(v)
    if isinstance(v, str):
        try: return float(v)
        except ValueError: return None
    if isinstance(v, dict):
        for k in ("weight", "score", "w"):
            raw = v.get(k)
            if isinstance(raw, (int, float)):
                return float(raw)
            if isinstance(raw, str):
                try:
                    return float(raw)
                except ValueError:
                    pass
    return None

def _normalize_transition_weights( all_transitions: Optional[Mapping[str, Mapping[str, _WeightVal]]] ) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    if not all_transitions:
        return out
    for f, to_map in all_transitions.items():
        try:
            items = to_map.items()
        except AttributeError:
            try:
                items = dict(to_map).items()
            except Exception:
                continue
        inner: Dict[str, float] = {}
        for t, raw in items:
            w = _as_float_weight(raw)
            if w is None:
                continue
            if not math.isfinite(w):
                continue
            inner[t] = inner.get(t, 0.0) + float(w)
        if inner:
            out[f] = inner
    return out

# =============================================================================
# Static / Defaults
# =============================================================================
_POS_ROOTS = ("희", "락")
_NEG_ROOTS = ("노", "애")

DEFAULT_CONFIG = {
    "sequence_top_k": 5,
    "sequence_min_score": 0.21,
    "polarity_gate_enabled": True,
    "polarity_dominance_threshold": 0.65,
    "polarity_opposite_keep_max": 1,
    "polarity_opposite_downscale": 0.6,
    "analysis_top_k": 5,
    "emit_all_taxonomy": False,
    "sub_emotions_aggregate": True,
    "round_digits": 3,
    "strip_empty": True,
    "cap_lengths": {
        "situation_analysis_per_emotion": 5,
        "emotion_sequence_per_sentence": 5
    },
    "sequence_keep_candidates_full": False,
    "sequence_candidates_full_cap": 100,
}
def _safe_float(v, default=0.0) -> float:
    try: return float(v)
    except Exception: return float(default)

def _safe_int(v, default=0) -> int:
    try: return int(v)
    except Exception: return int(default)


_PP_LOG = logging.getLogger("emotion_post")

def _tokenize_ko(text: str) -> List[str]:
    if not isinstance(text, str) or not text: return []
    toks = re.split(r"[^\w가-힣]+", text)
    return [t for t in toks if t]

def _round_floats_inplace(obj: Any, ndigits: int = 3) -> Any:
    if isinstance(obj, dict):
        for k, v in obj.items():
            obj[k] = _round_floats_inplace(v, ndigits)
        return obj
    if isinstance(obj, list):
        for i in range(len(obj)):
            obj[i] = _round_floats_inplace(obj[i], ndigits)
        return obj
    if isinstance(obj, float):
        try:
            return round(obj, ndigits)
        except Exception:
            return obj
    return obj


def _pick_situation_name(result: Dict[str, Any]) -> Optional[str]:
    try:
        sia = (result.get("situation_impact_analysis") or {}).get("situation_analysis") or []
        if isinstance(sia, list) and sia:
            top = max(sia, key=lambda x: float(x.get("impact_score", 0.0) or 0.0))
            if isinstance(top, dict):
                name = top.get("situation")
                if isinstance(name, str) and name.strip():
                    return name.strip()
    except Exception:
        pass
    try:
        rec = (result.get("context_summary") or {}).get("recent_situations") or []
        if isinstance(rec, list) and rec:
            name = rec[0].get("situation")
            if isinstance(name, str) and name.strip():
                return name.strip()
    except Exception:
        pass
    return None


def _rename_context_fields(context_info: Dict[str, Any], *, backcompat: bool = True) -> None:
    if not isinstance(context_info, dict):
        return
    stype = context_info.get("situation_type")
    if stype and "context_anchor" not in context_info:
        context_info["context_anchor"] = stype
    # situation_name은 외부에서 채워주므로 여기서는 빈 자리만 보장
    context_info.setdefault("situation_name", None)
    if not backcompat:
        # 완전 이전키 제거를 원할 때만
        context_info.pop("situation_type", None)


def _recompute_primary_from_sequence(seq: List[Dict[str, Any]]) -> Optional[Tuple[str, str, str]]:
    """
    ② 표출 일관화: 최종 emotion_sequence를 기준으로 primary_emotion 재산출
    반환: (primary_emotion, primary_category, sub_category)
    """
    best_id = None
    best_score = -1.0
    for item in seq or []:
        if not isinstance(item, dict):
            continue
        for em in item.get("emotions", []):
            if not isinstance(em, dict):
                continue
            emid = em.get("emotion_id")
            conf = float(em.get("confidence", 0.0) or 0.0)
            if emid and conf > best_score:
                best_score = conf
                best_id = emid

    if not best_id:
        return None

    # "희-환희" → primary_category="희", sub_category="환희"
    if "-" in best_id:
        cat, sub = best_id.split("-", 1)
    else:
        cat, sub = best_id, ""
    return best_id, cat, sub


def _unify_emotion_sequences(result: Dict[str, Any]) -> None:
    """ ② 값 일관화 """
    seq = result.get("emotion_sequence")
    if not isinstance(seq, list): return
    ea = result.setdefault("emotion_analysis", {})
    ea["emotion_sequence"] = copy.deepcopy(seq)
    prim = _recompute_primary_from_sequence(seq)
    if prim: result["primary_emotion"], result["primary_category"], result["sub_category"] = prim

def _attach_evidence_to_situations( result: Dict[str, Any], text: str ) -> None:
    """ ③ 상황 추천에 텍스트 근거 부착: """
    try:
        sia = (result.get("situation_impact_analysis") or {}).get("situation_analysis") or []
        if not isinstance(sia, list) or not sia:
            return

        # 입력 문장 토큰화
        sentences = re.split(r'(?<=[.!?])\s+|[\n\r]+', text.strip()) if text else []
        sent_tokens = [set(_tokenize_ko(s)) for s in sentences]

        for item in sia:
            if not isinstance(item, dict):
                continue
            sname = item.get("situation")
            if not isinstance(sname, str) or not sname.strip():
                continue
            skeys = set(_tokenize_ko(sname))
            if not skeys:
                continue

            matched: List[str] = []
            hit_idx: List[int] = []
            for i, toks in enumerate(sent_tokens):
                inter = skeys & toks
                if inter:
                    hit_idx.append(i)
                    matched.extend(sorted(inter))

            matched = sorted(list(set(matched)))
            coverage = (len(matched) / len(skeys)) if skeys else 0.0
            item["evidence"] = {
                "matched_keywords": matched,
                "sentence_indexes": hit_idx,
                "coverage": round(coverage, 3),
            }
    except Exception as e:
        _PP_LOG.warning(f"[evidence] attach failed: {e}")


def _hide_low_confidence_sections(
    result: Dict[str, Any],
    *,
    threshold: float = 0.20
) -> None:
    """ ④ 저신뢰 섹션 축소 """
    ep = result.get("emotion_patterns")
    if isinstance(ep, dict):
        meta = ep.get("metadata") or {}
        cs = float(meta.get("confidence_score", 1.0) or 1.0)
        if cs < threshold:
            ep["omitted_due_to_low_confidence"] = True
            # 본문 축소
            if "emotional_transitions" in ep:
                ep["emotional_transitions"]["transitions"] = {}
                # statistics는 개수 정도만 유지
                stats = ep["emotional_transitions"].get("statistics") or {}
                for k in list(stats.keys()):
                    if k not in ("total_transitions", "avg_intensity", "labeling_match_ratio"):
                        stats.pop(k, None)
            if "emotion_sequences" in ep:
                es = ep["emotion_sequences"]
                es["sequences"] = es.get("sequences", [])[:2]  # 2개만
            if "transition_dynamics" in ep:
                # 카운트만 유지
                td = ep["transition_dynamics"]
                for k in list(td.keys()):
                    if k not in ("total_transitions",):
                        td.pop(k, None)
    cs = (result.get("context_summary") or {}).get("context_patterns")
    if isinstance(cs, dict):
        meta = cs.get("metadata") or {}
        conf = float(meta.get("confidence_score", 1.0) or 1.0)
        if conf < threshold:
            cs["omitted_due_to_low_confidence"] = True
            if "emotional_transitions" in cs:
                cs["emotional_transitions"]["transitions"] = {}
            if "emotion_sequences" in cs:
                cs["emotion_sequences"]["sequences"] = cs["emotion_sequences"].get("sequences", [])[:2]
            if "transition_dynamics" in cs:
                td = cs["transition_dynamics"]
                for k in list(td.keys()):
                    if k not in ("total_transitions",):
                        td.pop(k, None)


def _prune_and_round_payload(
    result: Dict[str, Any],
    *,
    max_situations: int = 10,
    ndigits: int = 3
) -> None:
    """
    ⑤ 페이로드 경량화: """
    try:
        sia = (result.get("situation_impact_analysis") or {}).get("situation_analysis")
        if isinstance(sia, list) and len(sia) > max_situations:
            # impact_score 기준 상위 max_situations
            top = sorted(sia, key=lambda x: float(x.get("impact_score", 0.0) or 0.0), reverse=True)[:max_situations]
            result["situation_impact_analysis"]["situation_analysis"] = top
    except Exception:
        pass
    # 눈에 띄는 중복/임시키 제거 예시
    for k in ("_cache", "_raw_candidates", "_debug"):
        result.pop(k, None)
    # 반올림
    _round_floats_inplace(result, ndigits=ndigits)

def _derive_progression_signals_from_seq(seq):
    """
    emotion_sequence의 intensity를 이용해 단계(beginning/development/climax/aftermath) 신호를
    간단히 추정합니다. progressive_analysis에 직접 점수가 있으면 그걸 우선 사용합니다.
    """
    try:
        # seq 기반 추정
        if not isinstance(seq, list) or not seq:
            return {}

        intensities = [
            float((s.get("intensity") or {}).get("score", 0.0) or 0.0)
            for s in seq if isinstance(s, dict)
        ]
        if not intensities or not any(intensities):
            return {}

        total = sum(intensities) or 1.0
        beg = intensities[0] / total
        aft = (intensities[-1] / total) if len(intensities) >= 2 else 0.0
        cli = max(intensities) / total
        dev = max(0.0, 1.0 - (beg + cli + aft))
        return {
            "beginning": beg,
            "development": dev,
            "climax": cli,
            "aftermath": aft,
        }
    except Exception:
        return {}

def _surface_progression_signals(result: dict) -> None:
    """
    progression_signals를 생성/부착.
    1) progressive_analysis에 숫자가 있으면 정규화해서 사용
    2) 없으면 emotion_sequence의 intensity를 완만 스무딩 후
       시작/전개/클라이맥스/여파로 분할해 평균값을 정규화
    """
    try:
        stage_keys = ("beginning", "development", "climax", "aftermath")
        pa = result.get("progressive_analysis") or {}
        stage_scores = {k: float(pa.get(k, 0.0) or 0.0) for k in stage_keys}
        s = sum(v for v in stage_scores.values() if v > 0)
        if s > 0:
            result["progression_signals"] = {k: (v / s) if s > 0 else 0.0 for k, v in stage_scores.items()}
            return

        seq = result.get("emotion_sequence") or []
        intens = [float((it.get("intensity") or {}).get("score", 0.0) or 0.0) for it in seq if isinstance(it, dict)]
        n = len(intens)
        if n == 0:
            return
        if n == 1:
            # 단문 케이스는 시작/클라이맥스를 동등 가중
            result["progression_signals"] = {"beginning": 0.5, "development": 0.0, "climax": 0.5, "aftermath": 0.0}
            return

        # 간단 이동평균 스무딩(window=3)
        smoothed: List[float] = []
        for i in range(n):
            L = max(0, i - 1)
            R = min(n, i + 2)
            smoothed.append(sum(intens[L:R]) / max(1, (R - L)))

        peak_idx = max(range(n), key=lambda i: smoothed[i])
        head_end = max(1, int(round(n * 0.25)))

        beg = sum(smoothed[:head_end]) / max(1, head_end)
        dev = sum(smoothed[head_end:peak_idx]) / max(1, (peak_idx - head_end)) if peak_idx - head_end > 0 else 0.0
        cli = smoothed[peak_idx]
        aft = sum(smoothed[peak_idx + 1:]) / max(1, (n - peak_idx - 1)) if (n - peak_idx - 1) > 0 else 0.0

        total = beg + dev + cli + aft
        if total > 0:
            result["progression_signals"] = {
                "beginning": beg / total,
                "development": dev / total,
                "climax": cli / total,
                "aftermath": aft / total
            }
    except Exception:
        pass

def _as_list_str(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, (list, tuple, set)):
        out: List[str] = []
        for v in x:
            s = str(v).strip()
            if s:
                out.append(s)
        return out
    s = str(x).strip()
    return [s] if s else []

def _dedup_preserve(seq: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for s in seq:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out

def _extract_korean_tokens(s: str) -> List[str]:
    # 한글 2자 이상 토큰, + 영문/숫자 단어(3자 이상)
    toks_ko = re.findall(r"[가-힣]{2,}", s or "")
    toks_en = re.findall(r"[A-Za-z0-9_]{3,}", s or "")
    return _dedup_preserve(toks_ko + toks_en)

def _ensure_min_emotions_per_sentence(
    result: Dict[str, Any],
    *,
    min_keep: int = 1,
    conf_floor: float = 0.12,
    dominant_ratio_thr: float = 0.65,  # 문서 지배 극성 임계
    ndigits: int = 3,
    coverage_threshold: float = 0.3  # 커버리지 임계값 (개선사항)
) -> None:
    """
    각 문장(segment)에 최소 1개 감정 후보가 있도록 보장.
    개선사항: 커버리지가 낮은 문장에만 제한적으로 백필 적용
    보강점:
      - 문서 지배 극성(pos/neg)과 정합되게 백필(필요 시 반대 극성 1개까지 허용)
      - 중복 emotion_id 병합 및 prev_emotion 보강
      - intensity(score/label) 재산출(최대 신뢰도 기반)
    우선순위: seg.candidates → emotion_analysis.emotion_sequence → debug.candidates_full → primary_emotion
    """

    def _polarity_of(eid: Optional[str]) -> str:
        if not isinstance(eid, str) or not eid:
            return "neu"
        head = eid.split("-", 1)[0]
        if head in ("희", "락"): return "pos"
        if head in ("노", "애"): return "neg"
        return "neu"
    
    def _calculate_sentence_coverage(sentence_text: str, emotion_candidates: List[Dict[str, Any]]) -> float:
        """
        문장의 감정 커버리지 계산
        - 감정 키워드/패턴이 문장에서 얼마나 많이 매칭되는지 측정
        - 커버리지가 낮을수록 백필이 필요
        """
        if not sentence_text or not emotion_candidates:
            return 0.0
        
        # 감정 키워드 추출
        emotion_keywords = set()
        for candidate in emotion_candidates:
            emotion_id = candidate.get("emotion_id", "")
            if emotion_id:
                # 감정 ID에서 키워드 추출 (예: "희-행복" -> "행복")
                parts = emotion_id.split("-")
                if len(parts) > 1:
                    emotion_keywords.add(parts[1])
                emotion_keywords.add(emotion_id)
        
        if not emotion_keywords:
            return 0.0
        
        # 문장에서 키워드 매칭 확인
        sentence_lower = sentence_text.lower()
        matched_keywords = sum(1 for keyword in emotion_keywords if keyword.lower() in sentence_lower)
        
        # 커버리지 = 매칭된 키워드 수 / 전체 키워드 수
        coverage = matched_keywords / len(emotion_keywords)
        return min(coverage, 1.0)

    def _doc_dominant_polarity(seq: List[Dict[str, Any]]) -> Tuple[Optional[str], float]:
        pos = neg = 0.0
        for it in seq or []:
            for e in it.get("emotions", []) or []:
                pol = _polarity_of(e.get("emotion_id"))
                w = float(e.get("confidence", 0.0) or 0.0)
                if pol == "pos": pos += w
                elif pol == "neg": neg += w
        tot = pos + neg
        if tot <= 0: return None, 0.0
        if pos >= neg: return "pos", pos / tot
        return "neg", neg / tot

    def _norm_item(item: Any, *, fallback_conf: float, intensity_score: Optional[float]) -> Optional[Dict[str, Any]]:
        if not isinstance(item, dict):
            return None
        eid = item.get("emotion_id") or item.get("id")
        if not isinstance(eid, str) or not eid:
            return None
        try:
            conf = float(item.get("confidence", item.get("score", fallback_conf)))
        except Exception:
            conf = float(fallback_conf)
        conf = max(conf, conf_floor)
        # label 통일: 'mid' → 'medium'
        ilabel = item.get("intensity_label") or (
            "high" if (isinstance(intensity_score, (int, float)) and intensity_score >= 0.66)
            else "medium" if (isinstance(intensity_score, (int, float)) and intensity_score >= 0.33)
            else "low"
        )
        if ilabel == "mid":
            ilabel = "medium"
        return {
            "emotion_id": eid,
            "confidence": round(conf, ndigits),
            "emotion_complexity": item.get("emotion_complexity") or "subtle",
            "intensity_label": ilabel
        }

    seq = result.get("emotion_sequence") or []
    if not isinstance(seq, list):
        return

    # 보조 소스
    ea_seq = ((result.get("emotion_analysis") or {}).get("emotion_sequence") or [])
    dbg_full = (result.get("debug") or {}).get("candidates_full")
    primary_stub: Optional[Dict[str, Any]] = None
    pe = result.get("primary_emotion")
    if isinstance(pe, str) and pe:
        primary_stub = {"emotion_id": pe, "confidence": conf_floor}

    # 문서 지배 극성 추정
    doc_pol, doc_ratio = _doc_dominant_polarity(seq)

    prev_top: Optional[str] = None
    for i, seg in enumerate(seq):
        if not isinstance(seg, dict):
            continue
        if seg.get("emotions"):  # 이미 후보 있으면 prev_emotion만 보강하고 다음
            if "prev_emotion" not in seg:
                seg["prev_emotion"] = prev_top
            if seg.get("emotions"):
                top = seg["emotions"][0].get("emotion_id")
                if isinstance(top, str) and top:
                    prev_top = top
            # intensity label 'mid' 정규화
            inten = seg.get("intensity") or {}
            lbl = inten.get("label")
            if lbl == "mid":
                inten["label"] = "medium"
                seg["intensity"] = inten
            continue

        # 문장 강도(있으면) 로직에 활용
        try:
            intensity_score = float((seg.get("intensity") or {}).get("score"))
        except Exception:
            intensity_score = None

        # 후보 수집
        picked: List[Dict[str, Any]] = []

        def _collect_from(arr: Any) -> List[Dict[str, Any]]:
            out: List[Dict[str, Any]] = []
            if not isinstance(arr, list):
                return out
            for it in arr:
                norm = _norm_item(it, fallback_conf=0.15, intensity_score=intensity_score)
                if norm:
                    out.append(norm)
            return out

        # 1) seg.candidates
        seg_cands = _collect_from(seg.get("candidates") or [])
        # 2) emotion_analysis.emotion_sequence[i]
        ea_cands = []
        if isinstance(ea_seq, list) and i < len(ea_seq):
            ea_item = ea_seq[i] if isinstance(ea_seq[i], dict) else {}
            ea_cands = _collect_from(ea_item.get("emotions") or [])
        # 3) debug.candidates_full[i]
        df_cands = []
        if isinstance(dbg_full, list) and i < len(dbg_full):
            df = dbg_full[i]
            if isinstance(df, dict):
                if isinstance(df.get("emotions"), list):
                    df_cands = _collect_from(df["emotions"])
                elif isinstance(df.get("candidates"), list):
                    df_cands = _collect_from(df["candidates"])

        pool = seg_cands or ea_cands or df_cands
        
        # 개선사항: 커버리지가 낮은 문장에만 백필 적용
        sentence_text = seg.get("sentence", "")
        sentence_coverage = _calculate_sentence_coverage(sentence_text, pool)
        
        # 커버리지가 임계값 이하인 경우에만 백필 적용
        should_backfill = sentence_coverage < coverage_threshold
        
        if not pool and primary_stub and should_backfill:
            stub = _norm_item(primary_stub, fallback_conf=conf_floor, intensity_score=intensity_score)
            pool = [stub] if stub else []
        elif not pool and not should_backfill:
            # 커버리지가 충분한 경우 백필하지 않음
            continue

        # 극성 게이트(문서 지배 극성이 충분히 강할 때만)
        if pool and (doc_pol in ("pos", "neg")) and (doc_ratio >= dominant_ratio_thr):
            same_pol = [e for e in pool if _polarity_of(e.get("emotion_id")) == doc_pol]
            if same_pol:
                pool = same_pol
            else:
                # 반대 극성 1개까지만 허용(점수 하향)
                opp = sorted(pool, key=lambda x: float(x.get("confidence", 0.0)), reverse=True)[:1]
                for e in opp:
                    e["confidence"] = round(max(conf_floor, float(e.get("confidence", 0.0)) * 0.6), ndigits)
                pool = opp

        # 중복 병합(emotion_id 최대 신뢰도만 유지)
        uniq: Dict[str, Dict[str, Any]] = {}
        for e in pool or []:
            eid = e.get("emotion_id")
            if not isinstance(eid, str) or not eid:
                continue
            if (eid not in uniq) or (float(e.get("confidence", 0.0)) > float(uniq[eid].get("confidence", 0.0))):
                uniq[eid] = e
        picked = sorted(uniq.values(), key=lambda x: float(x.get("confidence", 0.0)), reverse=True)[:max(1, int(min_keep))]

        if picked:
            seg["emotions"] = picked
            # prev_emotion 연결
            seg["prev_emotion"] = prev_top
            prev_top = picked[0]["emotion_id"]

            # intensity 재산출(최대 신뢰도 기반)
            top_conf = max(float(e.get("confidence", 0.0)) for e in picked)
            inten = seg.get("intensity") or {}
            inten["score"] = float(top_conf)
            inten["label"] = "high" if top_conf >= 0.66 else ("medium" if top_conf >= 0.33 else "low")
            seg["intensity"] = inten



def _normalize_core_schema(result: Dict[str, Any], *, backcompat: bool = True) -> None:
    """
    결과 스키마를 안전하게 정규화(키 보장/타입 정리/필드명 통일).
    - context_info 필수 키 보장 + 리스트화
    - emotion_sequence: text -> sentence 필드 통일, emotions 형식 표준화, intensity 보정
    - situation_impact_analysis: 컨테이너 보장
    """
    if not isinstance(result, dict):
        return

    # 1) context_info 정규화
    ci = result.get("context_info")
    if not isinstance(ci, dict):
        # 과거 호환: 'context'를 context_info로 승격
        if isinstance(result.get("context"), dict):
            ci = dict(result["context"])
            result["context_info"] = ci
        else:
            ci = {}
            result["context_info"] = ci

    for k in ("time_indicators", "location_indicators", "social_context", "emotional_triggers"):
        ci[k] = _dedup_preserve(_as_list_str(ci.get(k)))

    # 과거 호환 키 유지
    if backcompat:
        # situation_type은 이미 있으면 그대로, 없으면 context_anchor나 situation_name으로 보완 X(보수적)
        pass

    # 2) emotion_sequence 정규화
    seq = result.get("emotion_sequence")
    if not isinstance(seq, list):
        seq = []
        result["emotion_sequence"] = seq

    norm_seq: List[Dict[str, Any]] = []
    for it in seq:
        if not isinstance(it, dict):
            continue
        sent = it.get("sentence") or it.get("text")
        if not isinstance(sent, str) or not sent.strip():
            continue
        it["sentence"] = sent.strip()
        it.pop("text", None)

        emos = it.get("emotions", [])
        if isinstance(emos, dict):
            # {"희-행복": 0.21, ...} → [{"emotion_id":"희-행복", "confidence":0.21}, ...]
            lst: List[Dict[str, Any]] = []
            for k, v in emos.items():
                try:
                    conf = float(v)
                except Exception:
                    conf = 0.0
                lst.append({"emotion_id": str(k), "confidence": conf})
            emos = lst
        elif isinstance(emos, list):
            lst = []
            for e in emos:
                if not isinstance(e, dict):
                    continue
                eid = e.get("emotion_id") or e.get("id")
                if not isinstance(eid, str) or not eid.strip():
                    continue
                try:
                    conf = float(e.get("confidence", e.get("score", e.get("value", 0.0))))
                except Exception:
                    conf = 0.0
                ne = dict(e)
                ne["emotion_id"] = eid
                ne["confidence"] = conf
                lst.append(ne)
            emos = lst
        else:
            emos = []
        it["emotions"] = emos

        # intensity 보정
        inten = it.get("intensity")
        if not isinstance(inten, dict):
            mx = max((float(e.get("confidence", 0.0)) for e in emos), default=0.0)
            it["intensity"] = {
                "score": mx,
                "label": "low" if mx < 0.33 else ("medium" if mx < 0.66 else "high")
            }
        else:
            try:
                sc = float(inten.get("score", 0.0))
            except Exception:
                sc = 0.0
            inten["score"] = sc
            if "label" not in inten:
                inten["label"] = "low" if sc < 0.33 else ("medium" if sc < 0.66 else "high")
            it["intensity"] = inten

        norm_seq.append(it)
    result["emotion_sequence"] = norm_seq

    # 3) situation_impact_analysis 컨테이너 보장
    sia = result.get("situation_impact_analysis")
    if not isinstance(sia, dict):
        # 루트에 situation_analysis가 있으면 흡수
        sa = result.pop("situation_analysis", None)
        result["situation_impact_analysis"] = {
            "situation_analysis": sa if isinstance(sa, list) else [],
            "impact_scores": {}
        }
    else:
        if not isinstance(sia.get("situation_analysis"), list):
            sia["situation_analysis"] = []
        if not isinstance(sia.get("impact_scores"), dict):
            sia["impact_scores"] = {}

def _attach_evidence(result: Dict[str, Any], text: str) -> None:
    """
    상황 영향 항목에 근거(evidence) 자동 부착(고도화):
      - 상황명 토큰 + context_info.emotional_triggers를 키워드 풀로 결합
      - 각 문장과 키워드 교집합 기반 sentence_scores 산출
      - matched_sentence_examples(최대 2개) 제공
      - 기존 evidence가 있으면 보존·결측만 채움
    """
    if not isinstance(result, dict):
        return

    # 문장 수집
    seq: List[Dict[str, Any]] = result.get("emotion_sequence") or []
    sentences: List[str] = [s.get("sentence", "") for s in seq if isinstance(s, dict) and isinstance(s.get("sentence"), str)]
    if not sentences:
        sentences = _split_into_sentences(text or "") or []

    # 문장 토큰 캐시
    sent_tokens: List[Set[str]] = [set(_extract_korean_tokens(s)) for s in sentences]

    sia = result.get("situation_impact_analysis")
    if not isinstance(sia, dict):
        return
    sa_list = sia.get("situation_analysis")
    if not isinstance(sa_list, list):
        return

    ci = result.get("context_info") or {}
    trig_pool = []
    if isinstance(ci, dict):
        trig_pool = _as_list_str(ci.get("emotional_triggers"))

    for item in sa_list:
        if not isinstance(item, dict):
            continue

        sit = item.get("situation", "")
        kw_pool = _dedup_preserve(_extract_korean_tokens(sit) + trig_pool)
        if not kw_pool:
            # 숫자 보정만
            try: item["impact_score"] = float(item.get("impact_score", 0.0))
            except Exception: item["impact_score"] = 0.0
            continue

        matched_keywords: Set[str] = set()
        hit_idx: List[int] = []
        sentence_scores: List[float] = []

        kw_set = set(kw_pool)
        for i, toks in enumerate(sent_tokens):
            inter = kw_set & toks
            if inter:
                hit_idx.append(i)
                matched_keywords.update(inter)
                sentence_scores.append(round(len(inter) / max(1, len(kw_set)), 3))
            else:
                sentence_scores.append(0.0)

        coverage = (len(matched_keywords) / max(1, len(kw_set)))
        examples = [sentences[j] for j in hit_idx[:2]] if sentences else []

        ev = item.get("evidence") if isinstance(item.get("evidence"), dict) else {}
        ev = dict(ev or {})
        ev.setdefault("matched_keywords", sorted(matched_keywords))
        ev.setdefault("sentence_indexes", hit_idx)
        ev.setdefault("coverage", round(float(coverage), 3))
        if sentence_scores:
            ev.setdefault("sentence_scores", sentence_scores)
        if examples:
            ev.setdefault("matched_sentence_examples", examples)
        item["evidence"] = ev

        # impact_score 숫자 보정만 수행(리웨이트는 _filter_low_evidence_situations에서)
        try:
            item["impact_score"] = float(item.get("impact_score", 0.0))
        except Exception:
            item["impact_score"] = 0.0


def postprocess_context_flow_output(
    result: Dict[str, Any],
    text: str,
    *,
    low_conf_hide_thresh: float = 0.15,
    ndigits: int = 3,
    backcompat: bool = True
) -> Dict[str, Any]:
    try:
        # ① 스키마 정규화
        _normalize_core_schema(result, backcompat=backcompat)

        # ② 증거 부착
        _attach_evidence(result, text)

        # ③ 상황영향 필터/Top-K (문장 수 기반 동적 topk)
        seq_len = len(result.get("emotion_sequence") or [])
        dyn_topk = 5 if seq_len >= 2 else 3
        result = _filter_low_evidence_situations(
            result, min_cov=0.08, reweight=True, topk=dyn_topk
        )

        # ③-보강) location_indicators 정규화
        ci = result.setdefault("context_info", {})
        locs = ci.get("location_indicators")
        if isinstance(locs, set):
            ci["location_indicators"] = sorted(list(locs))
        elif isinstance(locs, tuple):
            ci["location_indicators"] = list(locs)
        elif not isinstance(locs, list):
            ci["location_indicators"] = ([] if not locs else [locs])

        # ④ 단계 신호 표면화
        _surface_progression_signals(result)

        # ⑤ 저신뢰 섹션 축소
        _hide_low_confidence_sections(result, threshold=low_conf_hide_thresh)

        # ✅ ⑤-보강) 문장별 최소 1개 감정 보장 (운영 스위치로 제어)
        # 개선사항: 운영에서는 백필 비활성화, 리포트 모드에서만 활성화
        enable_backfill = os.getenv("EMOTION_BACKFILL_ENABLED", "0") == "1"
        is_report_mode = os.getenv("REPORT_MODE", "0") == "1"
        
        if enable_backfill or is_report_mode:
            _ensure_min_emotions_per_sentence(
                result,
                min_keep=1,
                conf_floor=low_conf_hide_thresh,
                ndigits=ndigits
            )

        # ⑥ 반올림/경량화
        _prune_and_round_payload(result, max_situations=max(dyn_topk, 5), ndigits=ndigits)
        return result

    except Exception:
        try:
            _round_floats_inplace(result, ndigits=ndigits)
        except Exception:
            pass
        return result


def _filter_low_evidence_situations(payload: Dict[str, Any], *, min_cov: float = 0.08, reweight: bool = True, topk: int = 8) -> Dict[str, Any]:
    """
    evidence.coverage가 낮은 상황 후보는 제거/하향. 선택적으로 가중 재산정 후 상위 topk만 유지.
    impact_scores도 kept 항목 기준 재계산하여 동기화.
    """
    try:
        sia = payload.get("situation_impact_analysis")
        if not isinstance(sia, dict):
            return payload
        rows = sia.get("situation_analysis")
        if not isinstance(rows, list):
            return payload

        kept: List[Dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            ev = row.get("evidence", {}) if isinstance(row.get("evidence"), dict) else {}
            cov = float(ev.get("coverage", 0.0) or 0.0)
            if cov < min_cov:
                continue
            if reweight:
                base = float(row.get("impact_score", 0.0) or 0.0)
                row["impact_score"] = round(base * (0.2 + 0.8 * cov), 3)  # 증거 강할수록↑
            kept.append(row)

        if not kept:
            # 아무것도 안 남으면 원본 유지
            return payload

        kept.sort(key=lambda r: r.get("impact_score", 0.0), reverse=True)
        kept = kept[:topk]
        sia["situation_analysis"] = kept

        # impact_scores 동기화
        impact_scores = {r.get("situation"): r.get("impact_score", 0.0) for r in kept if isinstance(r.get("situation"), str)}
        sia["impact_scores"] = impact_scores
        return payload
    except Exception:
        return payload


def _drop_redundant_sequence(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    emotion_sequence가 emotion_analysis.emotion_sequence와 완전 동일하면
    후자(중복)를 삭제해 결과를 간결하게.
    """
    try:
        seq = payload.get("emotion_sequence")
        ea = payload.get("emotion_analysis")
        if isinstance(ea, dict) and ea.get("emotion_sequence") == seq:
            ea.pop("emotion_sequence", None)
        return payload
    except Exception:
        return payload


# =============================================================================
# TypedDict / Dataclass 모델
# =============================================================================
class _ContextInfo(TypedDict):
    situation_type: Optional[str]
    time_indicators: List[str]
    location_indicators: List[str]
    social_context: List[str]
    emotional_triggers: List[str]
    intensity_markers: List[str]

class EmotionEntry(TypedDict, total=False):
    emotion: str
    intensity_label: str
    intensity_score: float
    timestamp: datetime
    weight: float
    context: Optional[dict]
    situation: Optional[dict]

@dataclass
class EmotionProgressionData:
    """감정 진행 데이터를 저장하는 클래스"""
    initial_emotion: str
    current_emotion: str
    start_time: datetime = field(default_factory=datetime.now)
    progression_path: List[str] = field(default_factory=list)
    duration: timedelta = field(default_factory=timedelta)
    intensity_changes: List[float] = field(default_factory=list)
    context_factors: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SituationData:
    """상황 데이터를 저장하는 클래스"""
    situation_type: str
    impact_score: float
    related_emotions: List[str]
    temporal_factors: Dict[str, float]
    context_weight: float


# =============================================================================
# EmotionContext
# =============================================================================
class EmotionContext:

    def __init__(self, emotions_data: Dict[str, Any]):
        self.emotions_data = emotions_data
        self.intensity_priors = {"low": 0.95, "medium": 1.0, "high": 1.05}
        self.history: List[EmotionEntry] = []
        self.transitions: Dict[Tuple[str, str], float] = defaultdict(float)
        self.emotion_counts: Dict[str, float] = defaultdict(float)
        self.emotion_flow: List[str] = []
        self.emotion_progressions: Dict[str, EmotionProgressionData] = {}
        self.significant_transitions: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
        self.situation_impacts: Dict[str, SituationData] = {}
        # 메모리 관리용 캐시(방어 로직과 정합)
        self._cache: Dict[str, Any] = {}

    def inject_priors(self, priors: Dict[str, Any]) -> None:
        try:
            updates = {k: float(v) for k, v in (priors or {}).items() if k in ("low", "medium", "high")}
            if updates:
                self.intensity_priors.update(updates)
        except Exception:
            pass

    def update(self, current_emotions: List[Dict[str, Any]]) -> None:  # 감정 문맥 업데이트
        try:
            if not current_emotions:
                logger.warning("감정 데이터가 비어있어 업데이트를 수행할 수 없습니다.")
                return
            current_timestamp = datetime.now()
            total_emotions = len(current_emotions)
            base_weights = self._calculate_base_weights(current_emotions)

            for idx, emotion_data in enumerate(current_emotions):
                current_emotion = emotion_data.get("emotion")
                if not current_emotion:
                    logger.warning(f"감정 정보가 누락됨: {emotion_data}")
                    continue

                intensity_info = self._process_emotion_intensity(emotion_data, idx, total_emotions)
                context_weight = self._calculate_context_weight(emotion_data, current_timestamp)
                temporal_weight = self._calculate_temporal_weight(current_timestamp)
                interaction_weight = self._calculate_interaction_weight(
                    current_emotion,
                    current_emotions,
                    idx
                )

                final_weight = self._calculate_final_weight(
                    base_weight=base_weights[idx],
                    intensity_weight=intensity_info["weight"],
                    context_weight=context_weight,
                    temporal_weight=temporal_weight,
                    interaction_weight=interaction_weight
                )

                self._update_emotion_history(
                    emotion=current_emotion,
                    intensity_info=intensity_info,
                    timestamp=current_timestamp,
                    weight=final_weight,
                    emotion_data=emotion_data
                )

                self._update_transition_patterns(
                    current_emotion=current_emotion,
                    intensity_score=intensity_info["score"],
                    timestamp=current_timestamp
                )

            self._cleanup_old_data(current_timestamp)
            self._manage_memory()
            logger.debug(f"감정 문맥 업데이트 완료: {len(current_emotions)}개 감정 처리됨")
        except Exception as e:
            logger.error(f"감정 문맥 업데이트 중 오류 발생: {str(e)}")
            raise

    def _calculate_base_weights(self, emotions: List[Dict[str, Any]]) -> List[float]:  # 기본 가중치 계산
        try:
            total_emotions = len(emotions)
            if total_emotions == 1:
                return [1.0]
            weights = []
            for idx, emotion_data in enumerate(emotions):
                position_weight = 1.0 - (idx * 0.1)
                intensity_score = emotion_data.get("intensity_score", 0.5)
                intensity_weight = intensity_score / total_emotions
                confidence = emotion_data.get("confidence", 1.0)
                confidence_weight = confidence * 0.5
                base_weight = (position_weight + intensity_weight + confidence_weight) / 3
                weights.append(max(0.1, min(base_weight, 1.0)))
            return weights
        except Exception as e:
            logger.error(f"기본 가중치 계산 중 오류: {str(e)}")
            # 안전한 균등 분배
            n = max(1, len(emotions))
            return [1.0 / n] * n

    def _process_emotion_intensity(
        self,
        emotion_data: Dict[str, Any],
        index: int,
        total_emotions: int
    ) -> Dict[str, Any]:  # 감정 강도 처리
        try:
            intensity_label = emotion_data.get("intensity_label", "medium")
            intensity_score = emotion_data.get("intensity_score", 0.5)
            base_intensity_weight = self.intensity_priors.get(intensity_label, 1.0)
            position_factor = 1.0 - (index * 0.1 / max(1, total_emotions))
            normalized_score = max(0.0, min(1.0, float(intensity_score)))
            final_intensity_weight = max(0.0, base_intensity_weight * position_factor * normalized_score)
            return {
                "label": intensity_label,
                "score": normalized_score,
                "weight": final_intensity_weight
            }
        except Exception as e:
            logger.error(f"감정 강도 정보를 계산하는 중 오류가 발생했습니다: {str(e)}")
            return {
                "label": "medium",
                "score": 0.5,
                "weight": 1.0
            }

    def _calculate_context_weight(self, emotion_data: Dict[str, Any], timestamp: datetime) -> float:  # 문맥 기반 가중치 계산
        try:
            context_weight = 1.0
            situation = emotion_data.get("situation", "")
            if situation:
                situation_impact = self.situation_impacts.get(situation)
                if situation_impact:
                    context_weight *= (1.0 + situation_impact.impact_score * 0.2)

            emotion = emotion_data.get("emotion")
            if emotion in self.emotion_counts:
                frequency = self.emotion_counts[emotion]
                frequency_weight = 1.0 + (math.log1p(frequency) * 0.1)
                context_weight *= frequency_weight

            hour = timestamp.hour
            if 9 <= hour <= 18:
                context_weight *= 1.2
            elif 0 <= hour <= 5:
                context_weight *= 0.8

            if self.history:
                prev_emotion = self.history[-1]["emotion"]
                if prev_emotion == emotion:
                    context_weight *= 1.1

            return max(0.5, min(context_weight, 2.0))
        except Exception as e:
            logger.error(f"문맥 가중치 계산 중 오류: {str(e)}")
            return 1.0

    def _calculate_temporal_weight(self, current_timestamp: datetime) -> float:  # 시간 기반 가중치 계산
        try:
            temporal_weight = 1.0
            if self.history:
                last_timestamp = self.history[-1]["timestamp"]
                time_diff = (current_timestamp - last_timestamp).total_seconds()
                if time_diff < 60:
                    temporal_weight = 1.2
                elif time_diff < 300:
                    temporal_weight = 1.1
                elif time_diff > 86400:
                    temporal_weight = 0.8
                elif time_diff > 3600:
                    temporal_weight = 0.9
            return temporal_weight
        except Exception as e:
            logger.error(f"시간 가중치 계산 중 오류: {str(e)}")
            return 1.0

    def _calculate_interaction_weight(
        self,
        current_emotion: str,
        all_emotions: List[Dict[str, Any]],
        current_index: int
    ) -> float:
        # 복합 감정 상호작용 가중치 계산
        try:
            interaction_weight = 1.0
            for idx, emotion_data in enumerate(all_emotions):
                if idx == current_index:
                    continue
                other_emotion = emotion_data.get("emotion")
                if not other_emotion:
                    continue
                compatibility_score = self._calculate_emotion_compatibility(
                    current_emotion,
                    other_emotion
                )
                distance = abs(current_index - idx)
                distance_factor = 1.0 / (1.0 + distance)
                interaction_weight *= (1.0 + (compatibility_score * distance_factor * 0.1))
            return max(0.5, min(interaction_weight, 1.5))
        except Exception as e:
            logger.error(f"상호작용 가중치 계산 중 오류: {str(e)}")
            return 1.0

    def _calculate_emotion_compatibility(self, emotion1: str, emotion2: str) -> float:  # 두 감정 간의 호환성 계산
        try:
            compatibility = 0.5
            primary1 = emotion1.split('-')[0] if '-' in emotion1 else emotion1
            primary2 = emotion2.split('-')[0] if '-' in emotion2 else emotion2
            if primary1 == primary2:
                compatibility += 0.3
            if (emotion1, emotion2) in self.transitions:
                compatibility += 0.2

            emotion1_data = self._get_emotion_data(emotion1)
            emotion2_data = self._get_emotion_data(emotion2)
            if emotion1_data and emotion2_data:
                related_emotions1 = set(emotion1_data.get("related_emotions", {}).get("positive", []))
                related_emotions2 = set(emotion2_data.get("related_emotions", {}).get("positive", []))
                if related_emotions1.intersection(related_emotions2):
                    compatibility += 0.1
            return max(0.0, min(compatibility, 1.0))
        except Exception as e:
            logger.error(f"감정 호환성 계산 중 오류: {str(e)}")
            return 0.5

    def _get_emotion_data(self, emotion_id: str) -> Dict[str, Any]:
        data: Dict[str, Any] = {}
        try:
            if not emotion_id:
                return data
            if '-' in emotion_id:
                primary, sub = emotion_id.split('-', 1)
            else:
                primary, sub = emotion_id, None

            cat_data = self.emotions_data.get(primary, {})
            if not cat_data:
                return data

            if sub:
                sub_data = cat_data.get('sub_emotions', {}).get(sub, {})
                if not sub_data:
                    sub_data = cat_data.get('emotion_profile', {}).get('sub_emotions', {}).get(sub, {})
                data["metadata"] = sub_data.get("metadata", {})
                data["emotion_profile"] = sub_data.get("emotion_profile", {})
                related_emotions = sub_data.get("emotion_profile", {}).get("related_emotions", {})
            else:
                data["metadata"] = cat_data.get("metadata", {})
                data["emotion_profile"] = cat_data.get("emotion_profile", {})
                related_emotions = cat_data.get("emotion_profile", {}).get("related_emotions", {})

            data["related_emotions"] = related_emotions
            return data
        except Exception as e:
            logger.error(f"_get_emotion_data 중 오류 발생: {str(e)}", exc_info=True)
            return data

    def _calculate_final_weight(
        self,
        base_weight: float,
        intensity_weight: float,
        context_weight: float,
        temporal_weight: float,
        interaction_weight: float
    ) -> float:  # 최종 가중치 계산
        try:
            weights = {
                "base": 0.3,
                "intensity": 0.2,
                "context": 0.2,
                "temporal": 0.15,
                "interaction": 0.15
            }
            final_weight = (
                base_weight * weights["base"] +
                intensity_weight * weights["intensity"] +
                context_weight * weights["context"] +
                temporal_weight * weights["temporal"] +
                interaction_weight * weights["interaction"]
            )
            normalized_weight = max(0.1, min(final_weight, 2.0))
            return normalized_weight
        except Exception as e:
            logger.error(f"최종 가중치 계산 중 오류: {str(e)}")
            return 1.0

    def _update_emotion_history(
        self,
        emotion: str,
        intensity_info: Dict[str, Any],
        timestamp: datetime,
        weight: float,
        emotion_data: Dict[str, Any]
    ) -> None:
        try:
            entry: EmotionEntry = {
                "emotion": emotion,
                "intensity_label": intensity_info["label"],
                "intensity_score": float(intensity_info["score"]),
                "timestamp": timestamp,
                "weight": float(weight),
            }
            if "context" in emotion_data:
                entry["context"] = emotion_data["context"]
            if "situation" in emotion_data:
                entry["situation"] = emotion_data["situation"]

            self.history.append(entry)
            self.emotion_flow.append(emotion)
            self.emotion_counts[emotion] += weight

            # 최근 5개만 유지
            self.history = self.history[-5:]
            self.emotion_flow = self.emotion_flow[-5:]

            logger.debug("EmotionContext - history updated: %s", emotion)
        except Exception as exc:
            logger.error("EmotionContext - history update failed: %s", exc, exc_info=True)

    def _update_transition_patterns(self, current_emotion: str, intensity_score: float, timestamp: datetime) -> None:
        try:
            if len(self.history) >= 2:
                prev_emotion = self.history[-2]["emotion"]
                prev_intensity = self.history[-2]["intensity_score"]

                transition = (prev_emotion, current_emotion)
                transition_intensity = (prev_intensity + intensity_score) / 2

                time_diff = (timestamp - self.history[-2]["timestamp"]).total_seconds()
                time_weight = 1.0
                if time_diff < 60:
                    time_weight = 1.2
                elif time_diff > 3600:
                    time_weight = 0.8

                transition_weight = transition_intensity * time_weight
                self.transitions[transition] += transition_weight

                intensity_change = abs(prev_intensity - intensity_score)
                if intensity_change > 0.5:
                    self._handle_significant_transition(
                        prev_emotion,
                        current_emotion,
                        intensity_change,
                        timestamp
                    )
            logger.debug(f"전이 패턴 업데이트 완료: {current_emotion}")
        except Exception as e:
            logger.error(f"전이 패턴 업데이트 중 오류: {str(e)}")

    def _handle_significant_transition(
        self,
        from_emotion: str,
        to_emotion: str,
        intensity_change: float,
        timestamp: datetime
    ) -> None:
        try:
            transition_key = (from_emotion, to_emotion)
            if not hasattr(self, 'significant_transitions'):
                self.significant_transitions = defaultdict(list)

            transition_info = {
                'timestamp': timestamp,
                'intensity_change': intensity_change,
                'from_emotion': from_emotion,
                'to_emotion': to_emotion
            }
            if hasattr(self, 'context_info'):
                transition_info['context'] = self.context_info.get(from_emotion, {})

            self.significant_transitions[transition_key].append(transition_info)

            if len(self.significant_transitions[transition_key]) >= 2:
                self._analyze_transition_pattern(transition_key)
            logger.debug(f"중요 전이 처리: {from_emotion} -> {to_emotion}")
        except Exception as e:
            logger.error(f"중요 전이 처리 중 오류: {str(e)}")

    def _analyze_transition_pattern(self, transition_key: Tuple[str, str]) -> None:  # 전이 패턴 분석
        try:
            transitions = self.significant_transitions[transition_key]
            intervals = []
            for i in range(1, len(transitions)):
                interval = transitions[i]['timestamp'] - transitions[i - 1]['timestamp']
                intervals.append(interval.total_seconds())

            intensity_changes = [t['intensity_change'] for t in transitions]
            pattern_stats = {
                'avg_interval': sum(intervals) / len(intervals) if intervals else 0,
                'avg_intensity_change': sum(intensity_changes) / len(intensity_changes),
                'occurrence_count': len(transitions),
                'latest_timestamp': transitions[-1]['timestamp']
            }
            if not hasattr(self, 'transition_patterns'):
                self.transition_patterns = {}
            self.transition_patterns[transition_key] = pattern_stats
            logger.debug(f"전이 패턴 분석 완료: {transition_key}")
        except Exception as e:
            logger.error(f"전이 패턴 분석 중 오류: {str(e)}")

    def _cleanup_old_data(self, current_timestamp: datetime) -> None:  # 오래된 데이터 정리
        try:
            retention_period = timedelta(hours=24)
            cutoff_time = current_timestamp - retention_period

            self.history = [entry for entry in self.history if entry["timestamp"] > cutoff_time]

            if hasattr(self, 'significant_transitions'):
                for transition_key in list(self.significant_transitions.keys()):
                    self.significant_transitions[transition_key] = [
                        t for t in self.significant_transitions[transition_key]
                        if t['timestamp'] > cutoff_time
                    ]
                    if not self.significant_transitions[transition_key]:
                        del self.significant_transitions[transition_key]

            if hasattr(self, 'transition_patterns'):
                for key in list(self.transition_patterns.keys()):
                    if self.transition_patterns[key]['latest_timestamp'] < cutoff_time:
                        del self.transition_patterns[key]

            logger.debug("오래된 데이터 정리 완료")
        except Exception as e:
            logger.error(f"데이터 정리 중 오류: {str(e)}")

    def _manage_memory(self) -> None:  # 메모리 관리
        try:
            if psutil is None:
                return
            process = psutil.Process()
            memory_info = process.memory_info()
            current_memory = memory_info.rss / 1024 / 1024
            memory_threshold = 1024  # 3번 개선작업: 메모리 임계값 증가 (500MB→1GB)
            if current_memory > memory_threshold:
                for attr in ['context_info', 'transition_patterns']:
                    if hasattr(self, attr):
                        delattr(self, attr)
                if hasattr(self, '_cache'):
                    self._cache.clear()
                gc.collect()
            logger.debug(f"메모리 관리 완료: {current_memory:.2f}MB 사용 중")
        except Exception as e:
            logger.error(f"메모리 관리 중 오류: {str(e)}")

    def _cleanup_old_transitions(self) -> None:
        """오래된 전이 데이터 정리"""
        if hasattr(self, 'significant_transitions'):
            cutoff_time = datetime.now() - timedelta(hours=24)
            for transition_key in list(self.significant_transitions.keys()):
                self.significant_transitions[transition_key] = [
                    t for t in self.significant_transitions[transition_key]
                    if t['timestamp'] > cutoff_time
                ]
                if not self.significant_transitions[transition_key]:
                    del self.significant_transitions[transition_key]

    def get_dominant_emotion(self) -> Dict[str, Any]:
        """ 복합 감정을 고려하여 지배적인 감정을 판단하는 개선된 함수 """
        try:
            if not self.emotion_counts:
                logger.warning("감정 기록이 없습니다.")
                return {
                    "primary": None,
                    "secondary": None,
                    "intensity": 0.0,
                    "confidence": 0.0
                }

            emotion_scores = self._calculate_comprehensive_scores()
            sorted_emotions = sorted(
                emotion_scores.items(),
                key=lambda x: x[1]["final_score"],
                reverse=True
            )

            primary_emotion = sorted_emotions[0][0]
            primary_score = sorted_emotions[0][1]["final_score"]

            secondary_emotion = None
            if len(sorted_emotions) > 1:
                score_diff = primary_score - sorted_emotions[1][1]["final_score"]
                if score_diff < 0.2:  # 점수 차이가 작으면 복합 감정으로 판단
                    secondary_emotion = sorted_emotions[1][0]

            confidence = self._calculate_emotion_confidence(
                primary_emotion,
                secondary_emotion,
                sorted_emotions
            )

            intensity = self._calculate_emotion_intensity(
                primary_emotion,
                secondary_emotion
            )

            result = {
                "primary": primary_emotion,
                "secondary": secondary_emotion,
                "intensity": intensity,
                "confidence": confidence,
                "details": {
                    "primary_score": primary_score,
                    "interaction_factors": self._get_emotion_interactions(primary_emotion, secondary_emotion),
                    "temporal_pattern": self._get_temporal_pattern(primary_emotion)
                }
            }
            logger.debug(f"지배적 감정 분석 결과: {result}")
            return result
        except Exception as e:
            logger.error(f"지배적 감정 분석 중 오류 발생: {str(e)}")
            return {
                "primary": None,
                "secondary": None,
                "intensity": 0.0,
                "confidence": 0.0
            }

    def _get_emotion_interactions(self, emotion1: Optional[str], emotion2: Optional[str]) -> Dict[str, Any]:
        interactions: Dict[str, Any] = {}
        if not emotion1 and not emotion2:
            return interactions

        if emotion1 and not emotion2:
            data1 = self._get_emotion_data(emotion1)
            interactions["single_emotion_data"] = {
                "emotion": emotion1,
                "metadata": data1.get("metadata", {}),
                "emotion_profile": data1.get("emotion_profile", {})
            }
            return interactions

        data1 = self._get_emotion_data(emotion1) if emotion1 else {}
        data2 = self._get_emotion_data(emotion2) if emotion2 else {}

        from_cat, _ = self._split_emotion_id(emotion1) if emotion1 else (None, None)
        to_cat, _ = self._split_emotion_id(emotion2) if emotion2 else (None, None)
        interactions["transition_info"] = {}
        if from_cat and to_cat:
            labeling_transition = self._find_transition_in_labeling_data(emotion1, emotion2)
            if labeling_transition:
                interactions["transition_info"]["labeling_data"] = labeling_transition

        interactions["combined_metadata"] = {
            "emotion1_metadata": data1.get("metadata", {}),
            "emotion2_metadata": data2.get("metadata", {})
        }
        interactions["combined_profiles"] = {
            "emotion1_profile": data1.get("emotion_profile", {}),
            "emotion2_profile": data2.get("emotion_profile", {})
        }

        if emotion1 and emotion2:
            compatibility = self._calculate_emotion_compatibility(emotion1, emotion2)
            interactions["compatibility"] = compatibility
        return interactions

    def _get_temporal_pattern(self, emotion: Optional[str]) -> Dict[str, Any]:
        pattern: Dict[str, Any] = {}
        if not emotion or not self.history:
            return pattern

        emotion_history = [h for h in self.history if h["emotion"] == emotion]
        if not emotion_history:
            return pattern

        sorted_hist = sorted(emotion_history, key=lambda x: x["timestamp"])
        pattern["occurrences"] = len(sorted_hist)
        pattern["first_timestamp"] = sorted_hist[0]["timestamp"].isoformat()
        pattern["last_timestamp"] = sorted_hist[-1]["timestamp"].isoformat()

        intensities = [h["intensity_score"] for h in sorted_hist]
        pattern["min_intensity"] = round(min(intensities), 3)
        pattern["max_intensity"] = round(max(intensities), 3)
        pattern["avg_intensity"] = round(sum(intensities) / len(intensities), 3)

        labeling_info = self._find_labeling_emotion_info(emotion)
        if labeling_info:
            pattern["labeling_profile"] = labeling_info.get("emotion_profile", {})
            pattern["labeling_metadata"] = labeling_info.get("metadata", {})
        return pattern

    def _calculate_comprehensive_scores(self) -> Dict[str, Dict[str, float]]:
        """감정별 종합 점수 계산"""
        try:
            scores: Dict[str, Dict[str, float]] = {}
            total = sum(self.emotion_counts.values()) or 1.0
            for emotion, count in self.emotion_counts.items():
                base_score = count / total
                recency_weight = self._calculate_recency_weight(emotion)
                persistence_weight = self._calculate_persistence_weight(emotion)
                context_weight = self._calculate_context_relevance(emotion)
                final_score = (base_score * 0.4 +
                               recency_weight * 0.3 +
                               persistence_weight * 0.2 +
                               context_weight * 0.1)
                scores[emotion] = {
                    "base_score": base_score,
                    "recency_weight": recency_weight,
                    "persistence_weight": persistence_weight,
                    "context_weight": context_weight,
                    "final_score": final_score
                }
            return scores
        except Exception as e:
            logger.error(f"종합 점수 계산 중 오류: {str(e)}")
            return {}

    def _calculate_context_relevance(self, emotion: str) -> float:
        relevance = 0.0
        total_transitions = sum(self.transitions.values())
        if total_transitions > 0:
            for (from_e, to_e), weight in self.transitions.items():
                if from_e == emotion or to_e == emotion:
                    relevance += (weight / total_transitions) * 0.3

        recent_records = [h for h in self.history[-5:] if h["emotion"] == emotion]
        for record in recent_records:
            intensity = record["intensity_score"]
            if intensity >= 0.7:
                relevance += 0.2
            elif intensity >= 0.4:
                relevance += 0.1
            else:
                relevance += 0.05

        if hasattr(self, "significant_transitions"):
            for (from_e, to_e), changes in self.significant_transitions.items():
                if from_e == emotion or to_e == emotion:
                    relevance += 0.05 * len(changes)

        if relevance > 1.0:
            relevance = 1.0
        return round(relevance, 3)

    def _calculate_emotion_confidence(
        self,
        primary_emotion: Optional[str],
        secondary_emotion: Optional[str],
        sorted_emotions: List[Tuple[str, Dict[str, float]]]
    ) -> float:
        """감정 신뢰도 계산"""
        try:
            if not primary_emotion:
                return 0.0

            score_distribution = [e[1]["final_score"] for e in sorted_emotions]
            if not score_distribution:
                return 0.0

            max_score = score_distribution[0]
            if max_score <= 0:
                return 0.0

            mean_score = sum(score_distribution) / len(score_distribution)
            score_variance = sum((s - mean_score) ** 2 for s in score_distribution) / len(score_distribution)

            temporal_consistency = self._check_temporal_consistency(primary_emotion)
            relationship_factor = 1.0
            if secondary_emotion:
                relationship_factor = self._check_emotion_relationship(primary_emotion, secondary_emotion)

            confidence = (0.4 * (max_score - mean_score) / max_score +
                          0.3 * (1 - math.sqrt(score_variance)) +
                          0.2 * temporal_consistency +
                          0.1 * relationship_factor)
            return max(0.0, min(1.0, confidence))
        except Exception as e:
            logger.error(f"신뢰도 계산 중 오류: {str(e)}")
            return 0.0

    def _calculate_emotion_intensity(
        self,
        primary_emotion: Optional[str],
        secondary_emotion: Optional[str]
    ) -> float:
        """감정 강도 계산"""
        try:
            if not primary_emotion:
                return 0.0

            # 1. 기본 강도 계산
            base_intensity = 0.0
            if self.history:
                recent_intensities = [
                    entry["intensity_score"]
                    for entry in self.history[-3:]  # 최근 3개 기록 참조
                    if entry["emotion"] == primary_emotion
                ]
                if recent_intensities:
                    base_intensity = sum(recent_intensities) / len(recent_intensities)

            intensity_modifier = 1.0
            if secondary_emotion:
                intensity_modifier = self._calculate_intensity_modification(primary_emotion, secondary_emotion)

            final_intensity = base_intensity * intensity_modifier
            return max(0.0, min(1.0, final_intensity))
        except Exception as e:
            logger.error(f"강도 계산 중 오류: {str(e)}")
            return 0.0

    def _split_emotion_id(self, emotion_id: str) -> Tuple[str, Optional[str]]:
        if '-' in emotion_id:
            category, sub_emotion = emotion_id.split('-', 1)
            return category, sub_emotion
        return emotion_id, None

    def _calculate_intensity_modification(self, primary_emotion: str, secondary_emotion: str) -> float:
        modification = 1.0
        from_cat, from_sub = self._split_emotion_id(primary_emotion)
        to_cat, to_sub = self._split_emotion_id(secondary_emotion)

        if from_cat in self.emotions_data:
            cat_data = self.emotions_data[from_cat]
            transitions = cat_data.get("emotion_transitions", {})
            if not transitions:
                transitions = cat_data.get("emotion_profile", {}).get("emotion_transitions", {})

            for pattern in transitions.get("patterns", []):
                fe = pattern.get("from_emotion")
                te = pattern.get("to_emotion")
                if (fe == from_cat or fe == primary_emotion) and (te == to_cat or te == secondary_emotion):
                    triggers = pattern.get("triggers", [])
                    modification += 0.05 * len(triggers)

            if from_sub:
                sub_data = cat_data.get("sub_emotions", {}).get(from_sub, {})
                if not sub_data:
                    sub_data = cat_data.get("emotion_profile", {}).get("sub_emotions", {}).get(from_sub, {})
                if sub_data:
                    sentiment_modifiers = sub_data.get("sentiment_analysis", {}).get("intensity_modifiers", {})
                    amps = sentiment_modifiers.get("amplifiers", [])
                    dims = sentiment_modifiers.get("diminishers", [])
                    if secondary_emotion in amps:
                        modification *= 1.2
                    if secondary_emotion in dims:
                        modification *= 0.8
        return round(modification, 3)

    def _calculate_recency_weight(self, emotion: str) -> float:
        """최근성 가중치 계산"""
        try:
            if not self.history:
                return 0.0
            current_time = datetime.now()
            weights = []
            for entry in self.history:
                if entry["emotion"] == emotion:
                    time_diff = (current_time - entry["timestamp"]).total_seconds()
                    # 시간 차이가 작을수록 높은 가중치
                    weight = math.exp(-time_diff / 3600)  # 1시간 기준
                    weights.append(weight)
            return max(weights) if weights else 0.0
        except Exception as e:
            logger.error(f"최근성 가중치 계산 중 오류: {str(e)}")
            return 0.0

    def _calculate_persistence_weight(self, emotion: str) -> float:
        """지속성 가중치 계산"""
        try:
            if not self.history:
                return 0.0
            consecutive_count = 0
            max_consecutive = 0
            for entry in self.history:
                if entry["emotion"] == emotion:
                    consecutive_count += 1
                    max_consecutive = max(max_consecutive, consecutive_count)
                else:
                    consecutive_count = 0
            return min(max_consecutive / 5, 1.0)  # 최대 5회 연속을 기준으로 정규화
        except Exception as e:
            logger.error(f"지속성 가중치 계산 중 오류: {str(e)}")
            return 0.0

    def _check_temporal_consistency(self, emotion: str) -> float:
        """시간적 일관성 검사"""
        try:
            if len(self.history) < 2:
                return 1.0
            consistent_transitions = 0
            total_transitions = 0
            for i in range(1, len(self.history)):
                if self.history[i]["emotion"] == emotion:
                    if self.history[i - 1]["emotion"] == emotion:
                        consistent_transitions += 1
                    total_transitions += 1
            return consistent_transitions / total_transitions if total_transitions > 0 else 0.0
        except Exception as e:
            logger.error(f"시간적 일관성 검사 중 오류: {str(e)}")
            return 0.0

    def _check_emotion_relationship(self, emotion1: str, emotion2: str) -> float:
        """감정 관계 검사"""
        try:
            # 1. 전이 빈도 확인
            transition_freq = self.transitions.get((emotion1, emotion2), 0) + \
                              self.transitions.get((emotion2, emotion1), 0)

            # 2. 윈도우 동시 출현율
            cooccurrence = 0
            total_windows = 0
            window_size = 3  # 3개 단위로 윈도우 설정
            for i in range(len(self.history) - window_size + 1):
                window = self.history[i:i + window_size]
                emotions_in_window = {entry["emotion"] for entry in window}
                if emotion1 in emotions_in_window and emotion2 in emotions_in_window:
                    cooccurrence += 1
                total_windows += 1
            cooccurrence_rate = cooccurrence / total_windows if total_windows > 0 else 0

            relationship_score = 0.7 * transition_freq + 0.3 * cooccurrence_rate
            return min(1.0, relationship_score)

        except Exception as e:
            logger.error(f"감정 관계 검사 중 오류: {str(e)}")
            return 0.0

    def get_transition_probability(self, from_emotion: str, to_emotion: str) -> float:
        """특정 전이의 확률 계산"""
        total_transitions = sum(self.transitions.values())
        if total_transitions == 0:
            return 0.0
        return self.transitions.get((from_emotion, to_emotion), 0) / total_transitions

    def track_emotion_flow(self) -> Dict[str, Any]:
        """감정 흐름 추적 및 라벨링 뼈대와의 매핑을 수행하는 개선된 함수"""
        try:
            # 1. 기본 흐름 분석
            flow_analysis: Dict[str, Any] = {
                "emotion_sequence": self._analyze_emotion_sequence(),
                "transition_patterns": self._analyze_detailed_transitions(),
                "dominant_emotions": self._analyze_dominant_emotions(),
                "flow_characteristics": self._analyze_flow_characteristics(),
                "temporal_patterns": self._analyze_temporal_patterns()
            }

            # 2. 라벨링 뼈대와의 매핑 분석
            labeling_mapping = self._analyze_labeling_structure_mapping()
            if labeling_mapping:
                flow_analysis["labeling_mapping"] = labeling_mapping

            # 3. 감정 강도 패턴 분석
            intensity_patterns = self._analyze_intensity_patterns()
            if intensity_patterns:
                flow_analysis["intensity_patterns"] = intensity_patterns

            # 4. 복합 감정 패턴 분석
            complex_patterns = self._analyze_complex_emotion_patterns()
            if complex_patterns:
                flow_analysis["complex_patterns"] = complex_patterns

            # 5. 감정 전이 연관성 분석
            transition_correlations = self._analyze_transition_correlations()
            if transition_correlations:
                flow_analysis["transition_correlations"] = transition_correlations

            # 6. 시계열 기반 패턴 분석
            temporal_emotion_patterns = self._analyze_temporal_emotion_patterns()
            if temporal_emotion_patterns:
                flow_analysis["temporal_emotion_patterns"] = temporal_emotion_patterns

            # 7. 최종 검증 및 정규화(직렬화-safe 변환 포함)
            flow_analysis = self._validate_and_normalize_flow_analysis(flow_analysis)
            return flow_analysis
        except Exception as e:
            logger.error(f"감정 흐름 추적 중 오류 발생: {str(e)}")
            return {}

    def _analyze_complex_emotion_patterns(self) -> Dict[str, Any]:
        """ 복합 감정 패턴 분석 """
        if not self.history:
            return {}
        try:
            mixed = self._analyze_mixed_emotions()
            cycles = self._analyze_emotion_cycles()
            intensity_corr = self._analyze_intensity_correlations()

            labeling_mapped = []
            for mix_info in mixed:
                primary_e = mix_info.get("primary_emotion")
                concurrent_list = mix_info.get("concurrent_emotions", [])
                labeling_mapped.append(
                    self._map_emotions_to_labeling(primary_e, [ce["emotion"] for ce in concurrent_list])
                )

            result = {
                "mixed_emotions": mixed,
                "emotion_cycles": cycles,
                "intensity_correlations": intensity_corr,
                "labeling_mapped_info": labeling_mapped  # [라벨링뼈대] 매핑 결과 (예시)
            }
            return result
        except Exception as e:
            logger.error(f"복합 감정 패턴 분석 중 오류 발생: {str(e)}")
            return {}

    def _map_emotions_to_labeling(
        self,
        primary_emotion: str,
        concurrent_emotions: Sequence[str],
    ) -> Dict[str, Any]:
        details: List[Dict[str, Any]] = []
        mapping_result: Dict[str, Any] = {
            "primary_emotion": primary_emotion,
            "concurrent_emotions": list(concurrent_emotions),
            "details": details,
        }
        try:
            def _push(emotion_id: str) -> None:
                if not emotion_id:
                    return
                info = self._find_labeling_emotion_info(emotion_id)
                if info:
                    details.append({"emotion": emotion_id, "labeling_info": info})

            _push(primary_emotion)
            for emo in sorted(set(concurrent_emotions) - {primary_emotion}):
                _push(emo)
        except Exception as exc:
            logger.error("라벨링 매핑 실패: %s", exc, exc_info=True)

        return mapping_result

    def _find_labeling_emotion_info(self, emotion_id: str) -> Dict[str, Any]:
        info: Dict[str, Any] = {}
        try:
            if not hasattr(self, 'emotions_data'):
                logger.warning("self.emotions_data가 정의되지 않았습니다. 라벨링 데이터를 로드했는지 확인하세요.")
                return info
            if not isinstance(self.emotions_data, dict):
                logger.warning("self.emotions_data의 자료형이 Dict가 아닙니다. 올바른 라벨링 데이터를 로드했는지 확인하세요.")
                return info
            if not emotion_id:
                logger.warning("감정 ID(emotion_id)가 비어있습니다.")
                return info

            if '-' in emotion_id:
                primary, sub = emotion_id.split('-', 1)
            else:
                primary, sub = emotion_id, None

            cat_data = self.emotions_data.get(primary)
            if not cat_data:
                logger.warning(f"'{primary}'에 해당하는 감정 데이터가 [라벨링뼈대]에 존재하지 않습니다.")
                return info

            if sub:
                sub_emotion_data = cat_data.get('sub_emotions', {}).get(sub, {})
                if not sub_emotion_data:
                    sub_emotion_data = cat_data.get('emotion_profile', {}).get('sub_emotions', {}).get(sub, {})
                info["metadata"] = sub_emotion_data.get("metadata", {})
                info["emotion_profile"] = sub_emotion_data.get("emotion_profile", {})
            else:
                info["metadata"] = cat_data.get("metadata", {})
                info["emotion_profile"] = cat_data.get("emotion_profile", {})
            return info
        except Exception as e:
            logger.error(f"라벨링 감정 정보 검색 중 오류 발생: {str(e)}", exc_info=True)
            return info

    def _analyze_transition_correlations(self) -> Dict[str, Any]:
        """ 감정 전이 연관성 분석 """
        if not self.transitions:
            return {}
        try:
            correlation_result: Dict[str, Any] = {
                "transition_details": [],
                "labeling_insights": []
            }
            sorted_transitions = sorted(self.transitions.items(), key=lambda x: x[1], reverse=True)
            for (from_e, to_e), weight in sorted_transitions:
                detail = {
                    "from": from_e,
                    "to": to_e,
                    "accumulated_weight": round(weight, 3)
                }
                correlation_result["transition_details"].append(detail)

            for (from_e, to_e), weight in self.transitions.items():
                labeling_transition = self._find_transition_in_labeling_data(from_e, to_e)
                if labeling_transition:
                    correlation_result["labeling_insights"].append({
                        "transition": f"{from_e}->{to_e}",
                        "weight": round(weight, 3),
                        "labeling_info": labeling_transition
                    })
            return correlation_result
        except Exception as e:
            logger.error(f"감정 전이 연관성 분석 중 오류 발생: {str(e)}")
            return {}

    def _analyze_temporal_emotion_patterns(self) -> Dict[str, Any]:
        if not self.history:
            return {}
        try:
            result: Dict[str, Any] = {
                "time_based_sequence": [],
                "peak_emotions_by_time": [],
                "progression_stages": []
            }

            # 1) 시계열 정렬
            sorted_history = sorted(self.history, key=lambda x: x["timestamp"])
            for entry in sorted_history:
                result["time_based_sequence"].append({
                    "emotion": entry["emotion"],
                    "intensity": entry["intensity_score"],
                    "time": entry["timestamp"].isoformat()
                })

            # 2) 특정 시간대(peak) 감정 찾기
            peak_threshold = 0.7
            peak_data = [
                {
                    "emotion": h["emotion"],
                    "time": h["timestamp"].isoformat(),
                    "intensity_score": h["intensity_score"]
                }
                for h in sorted_history
                if h["intensity_score"] >= peak_threshold
            ]
            result["peak_emotions_by_time"] = peak_data

            # 3) 감정 진행 단계 분석(라벨링 기반)
            progression_stages: List[Dict[str, Any]] = []
            for entry in sorted_history:
                e_id = entry["emotion"]
                labeling_data = self._find_labeling_emotion_info(e_id)
                intensity_value = entry["intensity_score"]
                progression_map: Dict[str, Any] = {}

                if "emotion_profile" in labeling_data:
                    profile = labeling_data["emotion_profile"]
                    emotion_progression = {}
                    if "emotion_progression" in profile:
                        emotion_progression = profile["emotion_progression"]

                    if intensity_value < 0.3:
                        if "trigger" in emotion_progression:
                            progression_map["stage"] = "trigger"
                            progression_map["description"] = emotion_progression["trigger"]
                        else:
                            progression_map["stage"] = "trigger"
                            progression_map["description"] = "Low intensity stage"
                    elif 0.3 <= intensity_value < 0.6:
                        if "development" in emotion_progression:
                            progression_map["stage"] = "development"
                            progression_map["description"] = emotion_progression["development"]
                        else:
                            progression_map["stage"] = "development"
                            progression_map["description"] = "Growing intensity stage"
                    elif 0.6 <= intensity_value < 0.8:
                        if "peak" in emotion_progression:
                            progression_map["stage"] = "peak"
                            progression_map["description"] = emotion_progression["peak"]
                        else:
                            progression_map["stage"] = "peak"
                            progression_map["description"] = "High intensity stage"
                    else:
                        if "aftermath" in emotion_progression:
                            progression_map["stage"] = "aftermath"
                            progression_map["description"] = emotion_progression["aftermath"]
                        else:
                            progression_map["stage"] = "aftermath"
                            progression_map["description"] = "Post-peak stage"

                progression_stages.append({
                    "emotion": e_id,
                    "time": entry["timestamp"].isoformat(),
                    "intensity": intensity_value,
                    "possible_progression_stages": progression_map
                })

            result["progression_stages"] = progression_stages
            return result

        except Exception as e:
            logger.error(f"시계열 기반 패턴 분석 중 오류 발생: {str(e)}")
            return {}

    def _analyze_labeling_structure_mapping(self) -> Dict[str, Any]:
        """라벨링 뼈대 구조와의 매핑 분석을 수행하는 함수"""
        try:
            mapping_result: Dict[str, Any] = {
                "emotion_mappings": {},
                "transition_mappings": {},
                "intensity_mappings": {},
                "context_mappings": {}
            }

            for emotion in self.emotion_flow:
                if '-' in emotion:
                    primary, sub = emotion.split('-')
                    if primary in self.emotions_data:
                        sub_emotions = self.emotions_data[primary].get('sub_emotions', {})
                        if sub in sub_emotions:
                            mapping_result["emotion_mappings"][emotion] = {
                                "primary": primary,
                                "sub": sub,
                                "metadata": sub_emotions[sub].get('metadata', {}),
                                "profile": sub_emotions[sub].get('emotion_profile', {})
                            }

            for (from_emotion, to_emotion), weight in self.transitions.items():
                transition_key = f"{from_emotion}->{to_emotion}"
                transition_info = self._find_transition_in_labeling_data(from_emotion, to_emotion)
                if transition_info:
                    mapping_result["transition_mappings"][transition_key] = {
                        "weight": weight,
                        "labeling_info": transition_info
                    }

            for entry in self.history:
                emotion = entry.get("emotion")
                intensity = entry.get("intensity_score", 0)
                if emotion in mapping_result["emotion_mappings"]:
                    if "intensity_levels" not in mapping_result["intensity_mappings"]:
                        mapping_result["intensity_mappings"]["intensity_levels"] = {}
                    mapping_result["intensity_mappings"]["intensity_levels"][emotion] = {
                        "score": intensity,
                        "label": self._get_intensity_label_from_labeling(emotion, intensity)
                    }

            context_patterns = self._extract_context_patterns_from_labeling()
            if context_patterns:
                mapping_result["context_mappings"] = context_patterns
            return mapping_result
        except Exception as e:
            logger.error(f"라벨링 구조 매핑 분석 중 오류 발생: {str(e)}")
            return {}

    def _find_transition_in_labeling_data(self, from_emotion: str, to_emotion: str) -> Dict[str, Any]:
        """라벨링 데이터에서 감정 전이 정보를 찾는 함수"""
        try:
            transition_info: Dict[str, Any] = {}
            if '-' in from_emotion:
                primary_from, sub_from = from_emotion.split('-')
            else:
                primary_from, sub_from = from_emotion, None

            if '-' in to_emotion:
                primary_to, sub_to = to_emotion.split('-')
            else:
                primary_to, sub_to = to_emotion, None

            if primary_from in self.emotions_data and primary_to in self.emotions_data:
                from_data = self.emotions_data[primary_from]
                transitions = from_data.get('emotion_transitions', {}).get('patterns', [])
                for pattern in transitions:
                    if pattern.get('from_emotion') == primary_from and pattern.get('to_emotion') == primary_to:
                        transition_info['primary_level'] = pattern

            if sub_from and sub_to:
                from_sub_data = self.emotions_data[primary_from].get('sub_emotions', {}).get(sub_from, {})
                if not from_sub_data:
                    from_sub_data = self.emotions_data[primary_from].get('emotion_profile', {}).get('sub_emotions', {}).get(sub_from, {})
                sub_transitions = from_sub_data.get('emotion_transitions', {}).get('patterns', [])
                for pattern in sub_transitions:
                    if pattern.get('from_emotion') == from_emotion and pattern.get('to_emotion') == to_emotion:
                        transition_info['sub_level'] = pattern
            return transition_info
        except Exception as e:
            logger.error(f"전이 정보 검색 중 오류 발생: {str(e)}")
            return {}

    def _get_intensity_label_from_labeling(self, emotion: str, intensity_score: float) -> str:
        """라벨링 데이터 기반으로 강도 레이블을 결정하는 함수"""
        try:
            if '-' in emotion:
                primary, sub = emotion.split('-')
            else:
                return 'medium'  # 기본값

            emotion_data = self.emotions_data.get(primary, {})
            sub_data = emotion_data.get('sub_emotions', {}).get(sub, {})
            if not sub_data:
                sub_data = emotion_data.get('emotion_profile', {}).get('sub_emotions', {}).get(sub, {})

            intensity_levels = sub_data.get('emotion_profile', {}).get('intensity_levels', {})
            thresholds = {
                'low': 0.3,   # 기본값
                'medium': 0.7 # 기본값
            }
            if 'thresholds' in intensity_levels:
                thresholds = intensity_levels['thresholds']

            if intensity_score >= thresholds.get('medium', 0.7):
                return 'high'
            elif intensity_score >= thresholds.get('low', 0.3):
                return 'medium'
            else:
                return 'low'
        except Exception as e:
            logger.error(f"강도 레이블 결정 중 오류 발생: {str(e)}")
            return 'medium'

    def _extract_context_patterns_from_labeling(self) -> Dict[str, Any]:
        """라벨링 데이터에서 문맥 패턴을 추출하는 함수"""
        try:
            context_patterns: Dict[str, Any] = {
                "situation_patterns": {},
                "linguistic_patterns": {},
                "emotional_triggers": set(),
                "intensity_modifiers": {}
            }

            for primary, data in self.emotions_data.items():
                self._collect_situation_patterns(primary, data, context_patterns)
                self._collect_linguistic_patterns(primary, data, context_patterns)
                self._collect_emotional_triggers(primary, data, context_patterns)
                self._collect_intensity_modifiers(primary, data, context_patterns)

            # set → list 변환 등 직렬화 안전화
            self._sets_to_lists_inplace(context_patterns)
            return context_patterns
        except Exception as e:
            logger.error(f"문맥 패턴 추출 중 오류 발생: {str(e)}")
            return {}

    def _collect_situation_patterns(
        self,
        primary: str,
        data: Dict[str, Any],
        context_patterns: Dict[str, Any]
    ) -> None:
        """ 1) 상황(situations) 정보를 수집 """
        try:
            primary_context = data.get("context_patterns", {})
            primary_situations = primary_context.get("situations", {})
            for sit_name, sit_data in primary_situations.items():
                if sit_name not in context_patterns["situation_patterns"]:
                    context_patterns["situation_patterns"][sit_name] = {
                        "emotion_category": primary,
                        "description": sit_data.get("description", ""),
                        "keywords": set(sit_data.get("keywords", [])),
                        "examples": set(sit_data.get("examples", [])),
                        "variations": set(sit_data.get("variations", [])),
                        "intensity": sit_data.get("intensity", "medium")
                    }
                else:
                    context_patterns["situation_patterns"][sit_name]["keywords"].update(sit_data.get("keywords", []))
                    context_patterns["situation_patterns"][sit_name]["examples"].update(sit_data.get("examples", []))
                    context_patterns["situation_patterns"][sit_name]["variations"].update(sit_data.get("variations", []))

            sub_emotions = data.get("sub_emotions", {})
            if not sub_emotions:
                sub_emotions = data.get("emotion_profile", {}).get("sub_emotions", {})

            for sub_name, sub_data in sub_emotions.items():
                sub_ctx = sub_data.get("context_patterns", {})
                sub_situations = sub_ctx.get("situations", {})
                for sit_name, sit_data in sub_situations.items():
                    if sit_name not in context_patterns["situation_patterns"]:
                        context_patterns["situation_patterns"][sit_name] = {
                            "emotion_category": f"{primary}-{sub_name}",
                            "description": sit_data.get("description", ""),
                            "keywords": set(sit_data.get("keywords", [])),
                            "examples": set(sit_data.get("examples", [])),
                            "variations": set(sit_data.get("variations", [])),
                            "intensity": sit_data.get("intensity", "medium")
                        }
                    else:
                        context_patterns["situation_patterns"][sit_name]["keywords"].update(sit_data.get("keywords", []))
                        context_patterns["situation_patterns"][sit_name]["examples"].update(sit_data.get("examples", []))
                        context_patterns["situation_patterns"][sit_name]["variations"].update(sit_data.get("variations", []))

        except Exception as e:
            logger.error(f"_collect_situation_patterns 오류: {str(e)}")

    def _collect_linguistic_patterns(
        self,
        primary: str,
        data: Dict[str, Any],
        context_patterns: Dict[str, Any]
    ) -> None:
        """ 2) 언어 패턴(linguistic_patterns) 정보를 수집 """
        try:
            primary_linguistic = data.get("linguistic_patterns", {})
            key_phrases = primary_linguistic.get("key_phrases", [])
            if key_phrases:
                if "key_phrases" not in context_patterns["linguistic_patterns"]:
                    context_patterns["linguistic_patterns"]["key_phrases"] = []
                for phrase_info in key_phrases:
                    context_patterns["linguistic_patterns"]["key_phrases"].append({
                        "pattern": phrase_info.get("pattern", ""),
                        "weight": phrase_info.get("weight", 1.0),
                        "context_requirement": phrase_info.get("context_requirement", "")
                    })

            sentiment_combos = primary_linguistic.get("sentiment_combinations", [])
            if sentiment_combos:
                if "sentiment_combinations" not in context_patterns["linguistic_patterns"]:
                    context_patterns["linguistic_patterns"]["sentiment_combinations"] = []
                for combo in sentiment_combos:
                    context_patterns["linguistic_patterns"]["sentiment_combinations"].append({
                        "words": combo.get("words", []),
                        "weight": combo.get("weight", 1.0)
                    })

            sentiment_modifiers = primary_linguistic.get("sentiment_modifiers", {})
            if sentiment_modifiers:
                if "sentiment_modifiers" not in context_patterns["linguistic_patterns"]:
                    context_patterns["linguistic_patterns"]["sentiment_modifiers"] = {}
                # amplifiers
                amps = sentiment_modifiers.get("amplifiers", [])
                if amps:
                    context_patterns["linguistic_patterns"]["sentiment_modifiers"].setdefault("amplifiers", set())
                    for amp in amps:
                        context_patterns["linguistic_patterns"]["sentiment_modifiers"]["amplifiers"].add(amp)
                # diminishers
                dims = sentiment_modifiers.get("diminishers", [])
                if dims:
                    context_patterns["linguistic_patterns"]["sentiment_modifiers"].setdefault("diminishers", set())
                    for d in dims:
                        context_patterns["linguistic_patterns"]["sentiment_modifiers"]["diminishers"].add(d)

            # sub_emotions에도 linguistic_patterns가 있을 수 있으므로 처리
            sub_emotions = data.get("sub_emotions", {})
            if not sub_emotions:
                sub_emotions = data.get("emotion_profile", {}).get("sub_emotions", {})

            for sub_name, sub_data in sub_emotions.items():
                sub_ling = sub_data.get("linguistic_patterns", {})

                key_phrases = sub_ling.get("key_phrases", [])
                if key_phrases:
                    context_patterns["linguistic_patterns"].setdefault("key_phrases", [])
                    for phrase_info in key_phrases:
                        context_patterns["linguistic_patterns"]["key_phrases"].append({
                            "pattern": phrase_info.get("pattern", ""),
                            "weight": phrase_info.get("weight", 1.0),
                            "context_requirement": phrase_info.get("context_requirement", "")
                        })

                sentiment_combos = sub_ling.get("sentiment_combinations", [])
                if sentiment_combos:
                    context_patterns["linguistic_patterns"].setdefault("sentiment_combinations", [])
                    for combo in sentiment_combos:
                        context_patterns["linguistic_patterns"]["sentiment_combinations"].append({
                            "words": combo.get("words", []),
                            "weight": combo.get("weight", 1.0)
                        })

                sentiment_modifiers = sub_ling.get("sentiment_modifiers", {})
                if sentiment_modifiers:
                    context_patterns["linguistic_patterns"].setdefault("sentiment_modifiers", {})
                    amps = sentiment_modifiers.get("amplifiers", [])
                    dims = sentiment_modifiers.get("diminishers", [])
                    if amps:
                        context_patterns["linguistic_patterns"]["sentiment_modifiers"].setdefault("amplifiers", set())
                        for amp in amps:
                            context_patterns["linguistic_patterns"]["sentiment_modifiers"]["amplifiers"].add(amp)
                    if dims:
                        context_patterns["linguistic_patterns"]["sentiment_modifiers"].setdefault("diminishers", set())
                        for d in dims:
                            context_patterns["linguistic_patterns"]["sentiment_modifiers"]["diminishers"].add(d)

        except Exception as e:
            logger.error(f"_collect_linguistic_patterns 오류: {str(e)}")

    def _collect_emotional_triggers(
        self,
        primary: str,
        data: Dict[str, Any],
        context_patterns: Dict[str, Any]
    ) -> None:
        """ 3) 감정 트리거(emotional_triggers) 정보를 수집 """
        try:
            # 1. 대표 감정 레벨 전이(transition)에서 triggers 확인
            emotion_transitions = data.get("emotion_transitions", {})
            if not emotion_transitions:
                emotion_transitions = data.get("emotion_profile", {}).get("emotion_transitions", {})
            patterns = emotion_transitions.get("patterns", [])
            for patt in patterns:
                for trig in patt.get("triggers", []):
                    context_patterns["emotional_triggers"].add(trig)

            # 2. sub_emotions에도 전이 정보가 있을 수 있음
            sub_emotions = data.get("sub_emotions", {})
            if not sub_emotions:
                sub_emotions = data.get("emotion_profile", {}).get("sub_emotions", {})

            for sub_name, sub_data in sub_emotions.items():
                sub_trans = sub_data.get("emotion_transitions", {})
                if not sub_trans:
                    sub_trans = sub_data.get("emotion_profile", {}).get("emotion_transitions", {})
                sub_patterns = sub_trans.get("patterns", [])
                for patt in sub_patterns:
                    for trig in patt.get("triggers", []):
                        context_patterns["emotional_triggers"].add(trig)

        except Exception as e:
            logger.error(f"_collect_emotional_triggers 오류: {str(e)}")

    def _collect_intensity_modifiers(
        self,
        primary: str,
        data: Dict[str, Any],
        context_patterns: Dict[str, Any]
    ) -> None:
        """ 4) 강도 수정자(intensity_modifiers) 정보 수집 """
        try:
            # sentiment_analysis에 intensity_modifiers 있을 수 있음
            sentiment_analysis = data.get("sentiment_analysis", {})
            if sentiment_analysis:
                self._extract_modifiers_from_sentiment(
                    sentiment_analysis,
                    context_patterns["intensity_modifiers"]
                )

            # 대표 감정 레벨에서 linguistic_patterns도 확인
            ling_patterns = data.get("linguistic_patterns", {})
            if ling_patterns:
                mods = ling_patterns.get("sentiment_modifiers", {})
                self._extract_modifiers_dict(
                    mods,
                    context_patterns["intensity_modifiers"]
                )

            # sub_emotions도 확인
            sub_emotions = data.get("sub_emotions", {})
            if not sub_emotions:
                sub_emotions = data.get("emotion_profile", {}).get("sub_emotions", {})

            for sub_name, sub_data in sub_emotions.items():
                # sub-level sentiment_analysis
                sub_sentiment = sub_data.get("sentiment_analysis", {})
                if sub_sentiment:
                    self._extract_modifiers_from_sentiment(
                        sub_sentiment,
                        context_patterns["intensity_modifiers"]
                    )
                sub_ling = sub_data.get("linguistic_patterns", {})
                if sub_ling:
                    sub_mods = sub_ling.get("sentiment_modifiers", {})
                    self._extract_modifiers_dict(
                        sub_mods,
                        context_patterns["intensity_modifiers"]
                    )
        except Exception as e:
            logger.error(f"_collect_intensity_modifiers 오류: {str(e)}")

    def _extract_modifiers_from_sentiment(
        self,
        sentiment_analysis: Dict[str, Any],
        global_modifiers: Dict[str, Set[str]]
    ) -> None:
        intens_mods = sentiment_analysis.get("intensity_modifiers", {})
        if not intens_mods:
            return
        amps = intens_mods.get("amplifiers", [])
        dims = intens_mods.get("diminishers", [])
        if "amplifiers" not in global_modifiers:
            global_modifiers["amplifiers"] = set()
        if "diminishers" not in global_modifiers:
            global_modifiers["diminishers"] = set()
        for amp in amps:
            global_modifiers["amplifiers"].add(amp)
        for dim in dims:
            global_modifiers["diminishers"].add(dim)

    def _extract_modifiers_dict(
        self,
        modifiers_dict: Dict[str, List[str]],
        global_modifiers: Dict[str, Set[str]]
    ) -> None:
        amps = modifiers_dict.get("amplifiers", [])
        dims = modifiers_dict.get("diminishers", [])

        if "amplifiers" not in global_modifiers:
            global_modifiers["amplifiers"] = set()
        if "diminishers" not in global_modifiers:
            global_modifiers["diminishers"] = set()

        for amp in amps:
            global_modifiers["amplifiers"].add(amp)
        for dim in dims:
            global_modifiers["diminishers"].add(dim)

    def _validate_and_normalize_flow_analysis(self, flow_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """흐름 분석 결과를 검증하고 정규화하는 함수"""
        try:
            validated_analysis: Dict[str, Any] = {}
            required_keys = {
                "emotion_sequence",
                "transition_patterns",
                "dominant_emotions"
            }
            for key in required_keys:
                if key in flow_analysis:
                    validated_analysis[key] = flow_analysis[key]
                else:
                    logger.warning(f"필수 키 누락: {key}")
                    validated_analysis[key] = {}

            for key, value in flow_analysis.items():
                if key not in validated_analysis:
                    if isinstance(value, dict):
                        validated_analysis[key] = self._normalize_dict_values(value)
                    elif isinstance(value, list):
                        validated_analysis[key] = self._normalize_list_values(value)
                    else:
                        # datetime 등 단일 값 방어
                        if isinstance(value, datetime):
                            validated_analysis[key] = value.isoformat()
                        elif isinstance(value, set):
                            validated_analysis[key] = sorted(list(value))
                        else:
                            validated_analysis[key] = value

            validated_analysis = self._ensure_analysis_consistency(validated_analysis)
            return validated_analysis
        except Exception as e:
            logger.error(f"흐름 분석 검증 중 오류 발생: {str(e)}")
            return flow_analysis

    def _normalize_dict_values(self, data: Dict[str, Any]) -> Dict[str, Any]:
        normalized: Dict[str, Any] = {}
        try:
            for key, value in data.items():
                if isinstance(value, dict):
                    normalized[key] = self._normalize_dict_values(value)
                elif isinstance(value, list):
                    normalized[key] = self._normalize_list_values(value)
                else:
                    if isinstance(value, datetime):
                        normalized[key] = value.isoformat()
                    elif isinstance(value, set):
                        normalized[key] = sorted(list(value))
                    else:
                        normalized[key] = value
            return normalized
        except Exception as e:
            logger.error(f"_normalize_dict_values 오류: {str(e)}")
            return data

    def _normalize_list_values(self, data: List[Any]) -> List[Any]:
        normalized: List[Any] = []
        try:
            for item in data:
                if isinstance(item, dict):
                    normalized.append(self._normalize_dict_values(item))
                elif isinstance(item, list):
                    normalized.append(self._normalize_list_values(item))
                else:
                    if isinstance(item, datetime):
                        normalized.append(item.isoformat())
                    elif isinstance(item, set):
                        normalized.append(sorted(list(item)))
                    else:
                        normalized.append(item)
            return normalized
        except Exception as e:
            logger.error(f"_normalize_list_values 오류: {str(e)}")
            return data

    def _ensure_analysis_consistency(self, validated_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """ flow_analysis의 주요 구조들이 일관성(consistency)을 갖추도록 확인/보정하는 함수 """
        try:
            # dominant_emotions → dict 기대
            if "dominant_emotions" in validated_analysis and not isinstance(validated_analysis["dominant_emotions"], dict):
                logger.warning("'dominant_emotions'가 dict가 아님, 빈 dict로 교체합니다.")
                validated_analysis["dominant_emotions"] = {}

            # emotion_sequence → list 기대
            if "emotion_sequence" in validated_analysis and not isinstance(validated_analysis["emotion_sequence"], list):
                logger.warning("'emotion_sequence'가 list가 아님, 빈 list로 교체합니다.")
                validated_analysis["emotion_sequence"] = []

            # transition_patterns → dict 기대
            if "transition_patterns" in validated_analysis and not isinstance(validated_analysis["transition_patterns"], dict):
                logger.warning("'transition_patterns'가 dict가 아님, 빈 dict로 교체합니다.")
                validated_analysis["transition_patterns"] = {}

            return validated_analysis
        except Exception as e:
            logger.error(f"_ensure_analysis_consistency 오류: {str(e)}")
            return validated_analysis

    def _map_flow_to_labeling_data(self, flow_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """ 추출된 complex_patterns, intensity_patterns 등을 [라벨링뼈대]와 매핑하거나 """
        try:
            mapping_result: Dict[str, Any] = {}
            if "complex_patterns" in flow_analysis:
                complex_map = []
                for pattern_key, pattern_data in flow_analysis["complex_patterns"].items():
                    complex_map.append({
                        "pattern_type": pattern_key,
                        "details": pattern_data
                    })
                mapping_result["complex_mapped"] = complex_map

            # 강도 패턴 매핑
            if "intensity_patterns" in flow_analysis:
                intensity_map = []
                for pattern_key, pattern_val in flow_analysis["intensity_patterns"].items():
                    intensity_map.append({
                        "pattern_key": pattern_key,
                        "pattern_value": pattern_val
                    })
                mapping_result["intensity_mapped"] = intensity_map
            return mapping_result
        except Exception as e:
            logger.error(f"라벨링뼈대 매핑 중 오류 발생: {str(e)}")
            return {}

    def _analyze_complex_patterns(self) -> Dict[str, Any]:
        """복합 감정 패턴 분석 (기존 유지: 내부적으로 상위 구현 재사용)"""
        if not self.history:
            return {}
        # 기존 이름 유지 + 로직은 중앙화
        return self._analyze_complex_emotion_patterns()

    def _analyze_mixed_emotions(self) -> List[Dict[str, Any]]:
        """동시 발생 감정 분석"""
        mixed_emotions: List[Dict[str, Any]] = []
        time_window = timedelta(minutes=5)
        for i, entry in enumerate(self.history):
            current_time = entry["timestamp"]
            concurrent_emotions = []
            for j, other_entry in enumerate(self.history):
                if i != j and abs(current_time - other_entry["timestamp"]) <= time_window:
                    concurrent_emotions.append({
                        "emotion": other_entry["emotion"],
                        "intensity": other_entry["intensity_score"]
                    })
            if concurrent_emotions:
                mixed_emotions.append({
                    "primary_emotion": entry["emotion"],
                    "concurrent_emotions": concurrent_emotions,
                    "timestamp": current_time.isoformat()
                })
        return mixed_emotions

    def _analyze_emotion_cycles(self) -> List[Dict[str, Any]]:
        """감정 순환 패턴 분석"""
        emotion_sequence = [entry["emotion"] for entry in self.history]
        timestamps = [entry["timestamp"] for entry in self.history]
        cycles: List[Dict[str, Any]] = []
        for cycle_length in range(2, min(len(emotion_sequence), 5)):
            for i in range(len(emotion_sequence) - cycle_length):
                if emotion_sequence[i:i + cycle_length] == emotion_sequence[i + cycle_length:i + 2 * cycle_length]:
                    cycles.append({
                        "pattern": emotion_sequence[i:i + cycle_length],
                        "start_time": timestamps[i].isoformat(),
                        "duration": (timestamps[i + cycle_length] - timestamps[i]).total_seconds(),
                        "repetitions": 2
                    })
        return cycles

    def _analyze_intensity_correlations(self) -> Dict[str, float]:
        """
        연속 감정(pair) 간 강도 상관관계 분석.

        Returns
        -------
        Dict[str, float]
            key : "감정A->감정B"
            val : 피어슨 상관계수 (-1.0 ~ 1.0)
        """
        # ── 0) 샘플 부족 시 빠른 반환 ────────────────────────────
        if len(self.history) < 2:
            return {}

        # ── 1) 감정쌍별 (x, y) 수집 ───────────────────────────
        pair_xy: Dict[str, List[Tuple[float, float]]] = defaultdict(list)

        for curr, nxt in zip(self.history, self.history[1:]):
            key = f"{curr['emotion']}->{nxt['emotion']}"
            pair_xy[key].append((curr['intensity_score'], nxt['intensity_score']))

        # ── 2) 피어슨 r 계산 ─────────────────────────────────
        correlations: Dict[str, float] = {}
        for pair, xy in pair_xy.items():
            if len(xy) < 2:  # 표본 2-개 미만 → 신뢰도 낮아 skip
                continue

            x_vals, y_vals = zip(*xy)

            # 분산이 0이면 상관계수 정의 불가 → 0 처리
            if len(set(x_vals)) == 1 or len(set(y_vals)) == 1:
                correlations[pair] = 0.0
                continue

            correlations[pair] = self._calculate_correlation(x_vals, y_vals)

        return correlations

    @staticmethod
    def _calculate_correlation(
        x: Sequence[float],
        y: Sequence[float],
        method: str = "pearson"
    ) -> float:
        if len(x) < 2 or len(y) < 2:
            return 0.0

        if method == "pearson":
            mx, my = sum(x) / len(x), sum(y) / len(y)
            num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
            den = (sum((xi - mx) ** 2 for xi in x) * sum((yi - my) ** 2 for yi in y)) ** 0.5
            return 0.0 if den == 0 else num / den

        raise ValueError(f"Unknown method: {method}")

    def _analyze_emotion_sequence(self) -> List[Dict[str, Any]]:
        """
        self.history 에 기록된 감정 시퀀스를 구조화된 리스트로 반환
        """
        result: List[Dict[str, Any]] = []
        ctx = getattr(self, "context_info", {}) or {}

        for record in self.history:  # record 는 EmotionEntry
            result.append({
                "emotion": record["emotion"],
                "timestamp": record["timestamp"].isoformat(),  # 직렬화 안전
                "intensity": {
                    "label": record["intensity_label"],
                    "score": record["intensity_score"],
                },
                "context": ctx.get(record["emotion"], {}),
            })
        return result

    def _analyze_detailed_transitions(self) -> Dict[str, Any]:
        """상세 전이 패턴 분석"""
        transition_analysis: Dict[str, Any] = {
            "patterns": dict(self.transitions),
            "significant_changes": [],
            "frequent_sequences": [],
            "circular_patterns": []
        }

        if hasattr(self, "significant_transitions"):
            for (from_emotion, to_emotion), changes in self.significant_transitions.items():
                if changes:
                    avg_intensity_change = sum(c["intensity_change"] for c in changes) / len(changes)
                    transition_analysis["significant_changes"].append({
                        "from": from_emotion,
                        "to": to_emotion,
                        "avg_intensity_change": avg_intensity_change,
                        "occurrence_count": len(changes)
                    })

        # 빈발 시퀀스 패턴 분석
        emotion_sequence = [entry["emotion"] for entry in self.history]
        transition_analysis["frequent_sequences"] = self._find_frequent_sequences(emotion_sequence)
        transition_analysis["circular_patterns"] = self._detect_circular_patterns(emotion_sequence)
        return transition_analysis

    def _find_frequent_sequences(
        self,
        sequence: List[str],
        min_length: int = 2,
        min_support: int = 2
    ) -> List[Dict[str, Any]]:
        """빈발 감정 시퀀스 탐지"""
        frequent_sequences: List[Dict[str, Any]] = []
        sequence_len = len(sequence)
        for length in range(min_length, min(sequence_len + 1, 5)):  # 최대 길이 4까지 제한
            for i in range(sequence_len - length + 1):
                current_seq = tuple(sequence[i:i + length])
                count = 0
                for j in range(sequence_len - length + 1):
                    if tuple(sequence[j:j + length]) == current_seq:
                        count += 1
                if count >= min_support:
                    frequent_sequences.append({
                        "sequence": list(current_seq),
                        "count": count,
                        "support": count / (sequence_len - length + 1)
                    })
        return frequent_sequences

    def _detect_circular_patterns(self, sequence: List[str], max_length: int = 4) -> List[Dict[str, Any]]:
        """순환 감정 패턴 탐지"""
        circular_patterns: List[Dict[str, Any]] = []
        sequence_len = len(sequence)
        for length in range(2, min(sequence_len + 1, max_length + 1)):
            for i in range(sequence_len - length):
                current_pattern = sequence[i:i + length]
                if current_pattern[0] == current_pattern[-1]:
                    occurrences = self._count_pattern_occurrences(sequence, current_pattern)
                    if occurrences > 1:  # 최소 2번 이상 발생한 패턴만 포함
                        circular_patterns.append({
                            "pattern": current_pattern,
                            "length": length,
                            "occurrences": occurrences
                        })
        return circular_patterns

    def _count_pattern_occurrences(self, sequence: List[str], pattern: List[str]) -> int:
        """특정 패턴의 출현 빈도 계산"""
        count = 0
        pattern_len = len(pattern)
        for i in range(len(sequence) - pattern_len + 1):
            if sequence[i:i + pattern_len] == pattern:
                count += 1
        return count

    def _analyze_intensity_patterns(self) -> Dict[str, Any]:
        """감정 강도 패턴 분석"""
        if not self.history:
            return {}
        intensity_patterns: Dict[str, Any] = {
            "overall_trend": self._calculate_intensity_trend(),
            "peak_points": self._identify_peak_points(),
            "intensity_distribution": self._analyze_intensity_distribution()
        }
        return intensity_patterns

    def _calculate_intensity_trend(self) -> Dict[str, float]:
        """감정 강도 추세 분석"""
        intensities: List[float] = [entry["intensity_score"] for entry in self.history]
        if not intensities:
            return {"slope": 0.0, "variance": 0.0}

        slope: float = 0.0
        variance: float = 0.0
        x = list(range(len(intensities)))
        mean_y = sum(intensities) / len(intensities)

        if len(x) > 1:
            mean_x = sum(x) / len(x)
            numerator = sum((x[i] - mean_x) * (intensities[i] - mean_y) for i in range(len(x)))
            denominator = sum((x[i] - mean_x) ** 2 for i in range(len(x)))
            slope = numerator / denominator if denominator != 0 else 0.0

        if len(intensities) > 1:
            variance = sum((y - mean_y) ** 2 for y in intensities) / len(intensities)

        result: Dict[str, float] = {"slope": slope, "variance": variance}
        return result

    def _identify_peak_points(self) -> List[Dict[str, Any]]:
        """감정 강도의 피크 포인트 식별"""
        peaks: List[Dict[str, Any]] = []
        if len(self.history) < 3:
            return peaks

        for i in range(1, len(self.history) - 1):
            current_intensity = self.history[i]["intensity_score"]
            prev_intensity = self.history[i - 1]["intensity_score"]
            next_intensity = self.history[i + 1]["intensity_score"]

            if prev_intensity < current_intensity and current_intensity > next_intensity:
                peaks.append({
                    "emotion": self.history[i]["emotion"],
                    "intensity": current_intensity,
                    "timestamp": self.history[i]["timestamp"].isoformat()
                })
        return peaks

    def _analyze_intensity_distribution(self) -> Dict[str, Any]:
        """감정 강도 분포 분석"""
        if not self.history:
            return {}
        intensities = [entry["intensity_score"] for entry in self.history]
        return {
            "min": min(intensities),
            "max": max(intensities),
            "mean": sum(intensities) / len(intensities),
            "distribution": {
                "low": len([i for i in intensities if i < 0.4]),
                "medium": len([i for i in intensities if 0.4 <= i < 0.7]),
                "high": len([i for i in intensities if i >= 0.7])
            }
        }

    def _analyze_dominant_emotions(self) -> Dict[str, float]:
        """ 지배적 감정 분석 """
        total_occurrences = sum(self.emotion_counts.values())
        if total_occurrences <= 0:
            return {}
        return {
            emotion: count / total_occurrences
            for emotion, count in self.emotion_counts.items()
        }

    def _analyze_flow_characteristics(self) -> Dict[str, Any]:
        """감정 흐름 특성 분석"""
        if not self.history:
            return {}
        return {
            "volatility": self._calculate_emotion_volatility(),
            "persistence": self._calculate_emotion_persistence(),
            "transition_speed": self._calculate_transition_speed()
        }

    def _calculate_emotion_volatility(self) -> float:
        """감정 변동성 계산"""
        if len(self.history) < 2:
            return 0.0
        changes = 0
        for i in range(len(self.history) - 1):
            if self.history[i]["emotion"] != self.history[i + 1]["emotion"]:
                changes += 1
        return changes / (len(self.history) - 1)

    def _calculate_emotion_persistence(self) -> Dict[str, float]:
        persistence: Dict[str, float] = {}
        current_emotion, current_count = None, 0
        for entry in self.history:
            emo = entry["emotion"]
            if emo == current_emotion:
                current_count += 1
            else:
                if current_emotion is not None:
                    persistence[current_emotion] = max(persistence.get(current_emotion, 0), current_count)
                current_emotion, current_count = emo, 1
        if current_emotion is not None:
            persistence[current_emotion] = max(persistence.get(current_emotion, 0), current_count)
        return persistence

    def _calculate_transition_speed(self) -> float:
        """감정 전이 속도 계산"""
        if not self.transitions:
            return 0.0
        total_transitions = sum(self.transitions.values())
        unique_transitions = len(self.transitions)
        return total_transitions / unique_transitions if unique_transitions > 0 else 0.0

    def _analyze_temporal_patterns(self) -> Dict[str, Any]:
        """시간적 패턴 분석"""
        if not self.history:
            return {}
        sequence_length = len(self.history)
        unique_emotions = len({h["emotion"] for h in self.history})
        emotion_distribution = self._analyze_dominant_emotions()

        return {
            "sequence_length": sequence_length,
            "unique_emotions": unique_emotions,
            "emotion_distribution": emotion_distribution
        }

    # ─────────────────────────────────────────────────────────────────────
    # 내부 헬퍼: set → list 변환(직렬화 안전화)
    # ─────────────────────────────────────────────────────────────────────
    def _sets_to_lists_inplace(self, obj: Any) -> Any:
        if isinstance(obj, dict):
            for k, v in list(obj.items()):
                obj[k] = self._sets_to_lists_inplace(v)
            return obj
        elif isinstance(obj, list):
            return [self._sets_to_lists_inplace(x) for x in obj]
        elif isinstance(obj, set):
            return sorted(list(obj))
        else:
            return obj


# =============================================================================
# EmotionContextManager
# =============================================================================
class EmotionContextManager:
    """ 감정 문맥 관리 클래스 (안정성/직렬화/가중치 결합 보강) """

    def __init__(self, emotions_data: Dict[str, Any], retention_period: int = 24):
        # EMOTIONS.json 데이터
        self.emotions_data = emotions_data
        # 데이터 보관 주기(시간)
        self.retention_period = int(retention_period)
        # 감정 히스토리: (timestamp, situation, intensity) 리스트
        self.emotion_history: Dict[str, List[Tuple[datetime, str, float]]] = defaultdict(list)
        # 상황별 영향도 저장 (SituationData 또는 dict 폴백)
        self.situation_impacts: Dict[str, Any] = {}
        # 감정 진행 경로 (EmotionProgressionData 또는 dict 폴백)
        self.emotion_progressions: Dict[str, Any] = {}
        # 기타 컨텍스트 저장소
        self.context_store: Dict[str, Dict[str, Any]] = defaultdict(dict)
        # 상황 기본 가중의 완전 자동화 - intensity priors 추가
        self._intensity_priors = {'low': 0.95, 'medium': 1.0, 'high': 1.05}
        # 간단 메트릭
        self.metrics: Dict[str, Any] = {
            "updates": 0,
            "last_update": None,
            "cleanup_runs": 0,
            "dropped_history_items": 0
        }
        logger.info("EmotionContextManager 초기화 완료")

    # 상황 기본 가중의 완전 자동화 - priors 주입 메서드
    def inject_intensity_priors(self, priors: Dict[str, Any]) -> None:
        """외부에서 intensity priors를 주입합니다."""
        try:
            self._intensity_priors.update({k: float(v) for k, v in (priors or {}).items() if k in ('low', 'medium', 'high')})
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 내부 공통 유틸
    # ------------------------------------------------------------------
    @staticmethod
    def _safe_float(x: Any, default: float = 0.0) -> float:
        try:
            if x is None:
                return default
            return float(x)
        except Exception:
            return default

    @staticmethod
    def _now() -> datetime:
        return datetime.now()

    @staticmethod
    def _get(obj: Any, key: str, default: Any = None) -> Any:
        """dataclass/obj/dict 어디든 안전 접근"""
        try:
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)
        except Exception:
            return default

    @staticmethod
    def _set(obj: Any, key: str, value: Any) -> None:
        try:
            if isinstance(obj, dict):
                obj[key] = value
            else:
                setattr(obj, key, value)
        except Exception:
            pass

    def _mk_situation_data(self, situation: str, impact_score: float, related_emotions: Optional[List[str]] = None,
                           temporal_factors: Optional[Dict[str, int]] = None, context_weight: float = 1.0) -> Any:
        """SituationData 생성(없으면 dict 폴백)"""
        related_emotions = list(dict.fromkeys(related_emotions or []))  # unique 순서 유지
        temporal_factors = dict(temporal_factors or {})
        try:
            # dataclass 존재 시 사용
            return SituationData(
                situation_type=situation,
                impact_score=float(impact_score),
                related_emotions=related_emotions,
                temporal_factors=temporal_factors,
                context_weight=float(context_weight)
            )
        except Exception:
            return {
                "situation_type": situation,
                "impact_score": float(impact_score),
                "related_emotions": related_emotions,
                "temporal_factors": temporal_factors,
                "context_weight": float(context_weight)
            }

    def _mk_progression_data(self, emotion: str, intensity: float, context_factors: Dict[str, Any]) -> Any:
        """EmotionProgressionData 생성(없으면 dict 폴백)"""
        try:
            return EmotionProgressionData(
                initial_emotion=emotion,
                current_emotion=emotion,
                progression_path=[emotion],
                intensity_changes=[float(intensity)],
                context_factors=dict(context_factors),
                start_time=self._now()
            )
        except Exception:
            return {
                "initial_emotion": emotion,
                "current_emotion": emotion,
                "progression_path": [emotion],
                "intensity_changes": [float(intensity)],
                "context_factors": dict(context_factors),
                "start_time": self._now()
            }

    # ------------------------------------------------------------------
    # 메인 업데이트
    # ------------------------------------------------------------------
    def update_emotion_context(
        self,
        emotion: str,
        intensity: float,
        situation: str,
        weight: float = 1.0,
        context_factors: Optional[Dict[str, Any]] = None,
        current_time: Optional[datetime] = None
    ) -> None:
        """
        감정 문맥 정보 업데이트 - 상황별/진행/문맥 가중치 결합 + 히스토리/전이/정리까지 일괄 처리
        """
        try:
            context_factors = dict(context_factors or {})
            current_time = current_time or self._now()
            intensity = max(0.0, min(1.0, self._safe_float(intensity, 0.0)))
            weight = max(0.0, self._safe_float(weight, 1.0))

            # 상황 정보 매칭
            situation_info = self._find_matching_situation(emotion, situation)

            # 가중치 계산
            base_weight = self._calculate_base_situation_weight(emotion, intensity, situation_info)
            progression_weight = self._calculate_progression_weight(emotion, situation_info)
            context_weight = self._calculate_context_weight(context_factors, situation_info)
            final_weight = self._combine_weights(base_weight, progression_weight, context_weight, weight)

            # 히스토리 적재
            self.emotion_history[emotion].append((current_time, situation, intensity))

            # 영향/전이/경로 갱신 (final_weight 반영)
            eff_intensity = max(0.0, min(2.0, intensity * final_weight))
            self._update_situation_impact(situation, emotion, eff_intensity)
            self._update_detailed_situation_impact(
                situation, emotion, eff_intensity, final_weight, situation_info, context_factors
            )
            self._update_progression_path(emotion, intensity, context_factors)
            self._cleanup_old_data(current_time)

            # 메트릭
            self.metrics["updates"] += 1
            self.metrics["last_update"] = current_time.isoformat()

            # 디버그 로그
            logger.debug(
                "감정 문맥 업데이트 | emotion=%s situation=%s intensity=%.3f "
                "base=%.3f prog=%.3f ctx=%.3f ext=%.3f -> final=%.3f eff=%.3f",
                emotion, situation, intensity, base_weight, progression_weight, context_weight, weight, final_weight, eff_intensity
            )

        except Exception as e:
            logger.error(f"감정 문맥 업데이트 중 오류: {str(e)}")
            raise

    # ------------------------------------------------------------------
    # 상황 매칭/가중치 로직
    # ------------------------------------------------------------------
    def _find_matching_situation(self, emotion: str, situation: str) -> Dict[str, Any]:
        """라벨링 데이터에서 상황 정보 매칭"""
        try:
            if '-' in emotion:
                category, sub_emotion = emotion.split('-', 1)
            else:
                category, sub_emotion = emotion, None
            emotion_data = self.emotions_data.get(category, {})
            if sub_emotion:
                sub_emotions = emotion_data.get('sub_emotions', {}) or emotion_data.get('emotion_profile', {}).get('sub_emotions', {})
                sub_data = sub_emotions.get(sub_emotion, {})
                situations = (sub_data.get('context_patterns', {}) or {}).get('situations', {}) or {}
                for sit_name, sit_data in situations.items():
                    name_hit = situation.lower() in str(sit_name).lower()
                    kw_hit = any(str(kw).lower() in situation.lower() for kw in (sit_data.get('keywords', []) or []))
                    if name_hit or kw_hit:
                        return sit_data
            return {}
        except Exception as e:
            logger.error(f"상황 매칭 중 오류: {str(e)}")
            return {}

    def _calculate_base_situation_weight(self, emotion: str, intensity: float, situation_info: Dict[str, Any]) -> float:
        """기본 상황 가중치 계산"""
        try:
            base_weight = 1.0
            situation_intensity = (situation_info or {}).get('intensity', 'medium')
            # 상황 기본 가중의 완전 자동화 - priors 재사용
            base_weight *= float(self._intensity_priors.get(situation_intensity, 1.0))
            base_weight *= (0.5 + float(intensity) * 0.5)  # 0.5 ~ 1.0
            return max(0.25, min(base_weight, 2.0))
        except Exception as e:
            logger.error(f"기본 가중치 계산 중 오류: {str(e)}")
            return 1.0

    def _calculate_progression_weight(self, emotion: str, situation_info: Dict[str, Any]) -> float:
        """감정 진행 단계별 가중치 계산"""
        try:
            progression = (situation_info or {}).get('emotion_progression', {}) or {}
            if not progression:
                return 1.0
            
            # 진행 단계 가중의 라벨 유도화 - 단계별 키워드/예시 수 기반 동적 계산
            stage_weights = {}
            for stage in ['trigger', 'development', 'peak', 'aftermath']:
                stage_data = progression.get(stage, {})
                if isinstance(stage_data, dict):
                    # 키워드 수와 예시 수를 합산해서 보정값 계산
                    keywords_count = len(stage_data.get('keywords', []) or [])
                    examples_count = len(stage_data.get('examples', []) or [])
                    description_len = len(str(stage_data.get('description', '')))
                    
                    # 정규화된 가중치 계산 (0.05~0.3 범위 보정)
                    total_features = keywords_count + examples_count + (description_len // 50)
                    weight_adj = min(0.3, max(0.05, total_features * 0.02))
                    
                    # 기본 가중치 + 동적 보정
                    base_weights = {'trigger': 1.1, 'development': 1.2, 'peak': 1.3, 'aftermath': 1.0}
                    stage_weights[stage] = base_weights.get(stage, 1.0) + weight_adj
                else:
                    # 폴백: 기본 가중치
                    base_weights = {'trigger': 1.1, 'development': 1.2, 'peak': 1.3, 'aftermath': 1.0}
                    stage_weights[stage] = base_weights.get(stage, 1.0)
            
            current_stage = self._determine_current_stage(emotion)
            return float(stage_weights.get(current_stage, 1.0))
        except Exception as e:
            logger.error(f"진행 가중치 계산 중 오류: {str(e)}")
            return 1.0

    def _calculate_context_weight(self, context_factors: Dict[str, Any], situation_info: Dict[str, Any]) -> float:
        """문맥 요소 가중치 계산 (불리언/카운트 모두 지원)"""
        try:
            w = 1.0
            # 불리언 신호
            if bool(context_factors.get('time_relevant')) and (situation_info or {}).get('time_indicators'):
                w *= 1.1
            if bool(context_factors.get('location_relevant')) and (situation_info or {}).get('location_indicators'):
                w *= 1.1
            if bool(context_factors.get('social_relevant')) and (situation_info or {}).get('social_context'):
                w *= 1.1

            # 카운트형 신호(예: ContextExtractor에서 전달하는 길이 값들)
            for key, coef, cap in [
                ("time_indicators", 0.02, 1.25),
                ("location_indicators", 0.02, 1.25),
                ("social_context", 0.02, 1.25),
                ("emotional_triggers", 0.03, 1.35),
            ]:
                cnt = self._safe_float(context_factors.get(key), 0.0)
                if cnt > 0:
                    w *= min(1.0 + cnt * coef, cap)
            return max(0.5, min(w, 2.0))
        except Exception as e:
            logger.error(f"문맥 가중치 계산 중 오류: {str(e)}")
            return 1.0

    def _combine_weights(self, base_weight: float, progression_weight: float, context_weight: float, external_weight: float) -> float:
        """가중치 결합 (+ 외부 가중치 반영)"""
        try:
            weights = {'base': 0.5, 'progression': 0.3, 'context': 0.2}
            combined = (base_weight * weights['base'] +
                        progression_weight * weights['progression'] +
                        context_weight * weights['context'])
            combined *= max(0.0, float(external_weight))
            return max(0.5, min(combined, 2.0))
        except Exception as e:
            logger.error(f"가중치 결합 중 오류: {str(e)}")
            return 1.0

    def _determine_current_stage(self, emotion: str) -> Optional[str]:
        """현재 감정 단계 판단"""
        try:
            prog = self.emotion_progressions.get(emotion)
            if not prog:
                return None
            start_time = self._get(prog, "start_time")
            if not isinstance(start_time, datetime):
                return None
            duration = self._now() - start_time
            if duration < timedelta(minutes=5):
                return 'trigger'
            elif duration < timedelta(minutes=15):
                return 'development'
            elif duration < timedelta(minutes=30):
                return 'peak'
            else:
                return 'aftermath'
        except Exception as e:
            logger.error(f"감정 단계 판단 중 오류: {str(e)}")
            return None

    def _refine_situation_emotion_mapping(self, emotion: str, current_situation: str) -> Dict[str, Any]:
        """ 세부 감정-상황 매핑 보조 """
        try:
            if '-' in emotion:
                category, _ = emotion.split('-', 1)
            else:
                category = emotion
            emotion_data = self.emotions_data.get(category, {}) or {}
            sub_emotion_data = emotion_data.get('sub_emotions', {}) or emotion_data.get('emotion_profile', {}).get('sub_emotions', {}) or {}
            best_match, best_score = {}, 0.0
            for _, sub_data in sub_emotion_data.items():
                situations_dict = (sub_data.get('context_patterns', {}) or {}).get('situations', {}) or {}
                for situation_name, situation_info in situations_dict.items():
                    score = 0.0
                    if str(situation_name).lower() == current_situation.lower():
                        score += 1.0
                    for k in (situation_info.get('keywords', []) or []):
                        if str(k).lower() in current_situation.lower():
                            score += 0.5
                    for v in (situation_info.get('variations', []) or []):
                        if str(v).lower() in current_situation.lower():
                            score += 0.3
                    if score > best_score:
                        best_score, best_match = score, situation_info
            return best_match
        except Exception as e:
            logger.error(f"_refine_situation_emotion_mapping 중 오류: {str(e)}")
            return {}

    def _find_situation_in_labeling_data(self, emotion: str, situation: str) -> Dict[str, Any]:
        """라벨링 데이터에서 상황 정보 검색"""
        try:
            if '-' in emotion:
                category, sub_emotion = emotion.split('-', 1)
            else:
                category, sub_emotion = emotion, None
            emotion_data = self.emotions_data.get(category, {}) or {}
            if sub_emotion:
                sub_emotion_data = emotion_data.get('sub_emotions', {}) or emotion_data.get('emotion_profile', {}).get('sub_emotions', {}) or {}
                situations = (sub_emotion_data.get('context_patterns', {}) or {}).get('situations', {}) or {}
                return self._search_situation_recursively(situations, situation)
            return {}
        except Exception as e:
            logger.error(f"상황 정보 검색 중 오류: {str(e)}")
            return {}

    def _search_situation_recursively(self, situations: Dict[str, Any], target_situation: str) -> Dict[str, Any]:
        """재귀적으로 상황 정보 검색"""
        for situation_name, situation_data in (situations or {}).items():
            if str(situation_name).lower() == target_situation.lower():
                return situation_data
            if any(str(keyword).lower() in target_situation.lower() for keyword in (situation_data.get('keywords', []) or [])):
                return situation_data
            if any(str(variation).lower() in target_situation.lower() for variation in (situation_data.get('variations', []) or [])):
                return situation_data
        return {}

    # ------------------------------------------------------------------
    # 상황 영향/경로 업데이트
    # ------------------------------------------------------------------
    def _update_detailed_situation_impact(
        self,
        situation: str,
        emotion: str,
        intensity: float,
        situation_weight: float,
        situation_info: Dict[str, Any],
        context_factors: Dict[str, Any]
    ) -> None:
        """상세 상황 영향도 업데이트 (라벨링·문맥 반영)"""
        impact = self.situation_impacts.get(situation)
        if not impact:
            impact = self._mk_situation_data(
                situation=situation,
                impact_score=float(intensity) * float(situation_weight),
                related_emotions=[emotion],
                temporal_factors={},
                context_weight=1.0
            )
            self.situation_impacts[situation] = impact
        else:
            # 지수 평활
            current = self._get(impact, "impact_score", 0.0)
            updated = (current * 0.3) + (float(intensity) * float(situation_weight) * 0.7)
            self._set(impact, "impact_score", updated)

            # 관련 감정 집합 업데이트(중복 제거)
            rel = list(dict.fromkeys(list(self._get(impact, "related_emotions", [])) + [emotion]))
            self._set(impact, "related_emotions", rel)

            # 시간적 요소 스냅샷
            cur_key = self._now().strftime("%H:%M")
            tf = dict(self._get(impact, "temporal_factors", {}))
            tf[cur_key] = int(tf.get(cur_key, 0)) + 1
            self._set(impact, "temporal_factors", tf)

        # 라벨링 기반 컨텍스트 가중
        if situation_info:
            intensity_level = (situation_info or {}).get('intensity', 'medium')
            cw = float(self._get(impact, "context_weight", 1.0)) * float(self._intensity_priors.get(intensity_level, 1.0))
            progression = (situation_info or {}).get('emotion_progression', {}) or {}
            # 진행 단계 가중의 라벨 유도화 - 단계별 키워드/예시 수 기반 동적 계산
            for stage in ['trigger', 'development', 'peak', 'aftermath']:
                if stage in progression:
                    stage_data = progression.get(stage, {})
                    if isinstance(stage_data, dict):
                        # 키워드 수와 예시 수를 합산해서 보정값 계산
                        keywords_count = len(stage_data.get('keywords', []) or [])
                        examples_count = len(stage_data.get('examples', []) or [])
                        description_len = len(str(stage_data.get('description', '')))
                        
                        # 정규화된 가중치 계산 (0.05~0.3 범위 보정)
                        total_features = keywords_count + examples_count + (description_len // 50)
                        weight_adj = min(0.3, max(0.05, total_features * 0.02))
                        
                        # 기본 가중치 + 동적 보정
                        base_weights = {'trigger': 1.1, 'development': 1.2, 'peak': 1.3, 'aftermath': 1.0}
                        w = base_weights.get(stage, 1.0) + weight_adj
                    else:
                        # 폴백: 기본 가중치
                        base_weights = {'trigger': 1.1, 'development': 1.2, 'peak': 1.3, 'aftermath': 1.0}
                        w = base_weights.get(stage, 1.0)
                    cw *= float(w)
            self._set(impact, "context_weight", cw)

        # 문맥 요소(숫자형) 반영
        cw = float(self._get(impact, "context_weight", 1.0))
        for k in ("time_indicators", "location_indicators", "social_context", "emotional_triggers"):
            v = self._safe_float(context_factors.get(k), 0.0)
            if v > 0:
                cw *= (1.0 + min(v * 0.02, 0.3))
        self._set(impact, "context_weight", cw)

    def _cleanup_old_data(self, current_time: datetime) -> None:
        """오래된 데이터 정리(보관주기 초과 항목 제거)"""
        cutoff_time = current_time - timedelta(hours=self.retention_period)
        dropped = 0
        for emotion in list(self.emotion_history.keys()):
            new_list = [(t, s, i) for (t, s, i) in self.emotion_history[emotion] if isinstance(t, datetime) and t > cutoff_time]
            dropped += max(0, len(self.emotion_history[emotion]) - len(new_list))
            if new_list:
                self.emotion_history[emotion] = new_list
            else:
                del self.emotion_history[emotion]
        self.metrics["cleanup_runs"] += 1
        self.metrics["dropped_history_items"] += dropped

    def _update_progression_path(self, emotion: str, intensity: float, context_factors: Dict[str, Any]) -> None:
        """감정 진행 경로 업데이트 (폴백 지원)"""
        if emotion not in self.emotion_progressions:
            self.emotion_progressions[emotion] = self._mk_progression_data(emotion, intensity, context_factors)
            return
        prog = self.emotion_progressions[emotion]
        # path
        path = list(self._get(prog, "progression_path", [])) + [emotion]
        self._set(prog, "progression_path", path)
        # intensity
        ints = list(self._get(prog, "intensity_changes", [])) + [float(intensity)]
        self._set(prog, "intensity_changes", ints)
        # current
        self._set(prog, "current_emotion", emotion)
        # context decay update
        ctx = dict(self._get(prog, "context_factors", {}))
        for k, v in (context_factors or {}).items():
            prev = self._safe_float(ctx.get(k), 0.0)
            ctx[k] = prev * 0.7 + self._safe_float(v, 0.0) * 0.3
        self._set(prog, "context_factors", ctx)
        # start_time 보정
        if not isinstance(self._get(prog, "start_time"), datetime):
            self._set(prog, "start_time", self._now())

    def _update_situation_impact(self, situation: str, emotion: str, intensity: float) -> None:
        """상황 영향도(간단형) 업데이트"""
        impact = self.situation_impacts.get(situation)
        if not impact:
            impact = self._mk_situation_data(situation, intensity, [emotion], {}, 1.0)
            self.situation_impacts[situation] = impact
        else:
            current = self._get(impact, "impact_score", 0.0)
            self._set(impact, "impact_score", current * 0.7 + float(intensity) * 0.3)
            rel = list(dict.fromkeys(list(self._get(impact, "related_emotions", [])) + [emotion]))
            self._set(impact, "related_emotions", rel)

    # ------------------------------------------------------------------
    # 패턴/전이 분석
    # ------------------------------------------------------------------
    def _analyze_situation_correlations(self) -> Dict[str, Any]:
        """상황별 영향과 관련 감정을 요약합니다."""
        try:
            result = {"by_situation": {}, "top_situations": []}
            for situation, impact in (self.situation_impacts or {}).items():
                score = float(self._get(impact, "impact_score", 0.0))
                related = list(self._get(impact, "related_emotions", []))
                result["by_situation"][situation] = {
                    "impact_score": round(score, 3),
                    "related_emotions": related,
                }
            tops = sorted(
                result["by_situation"].items(),
                key=lambda kv: kv[1]["impact_score"],
                reverse=True,
            )[:10]
            result["top_situations"] = [
                {"situation": name, **info} for name, info in tops
            ]
            return result
        except Exception as e:
            logger.error(f"상황 상관 분석 중 오류 발생: {str(e)}")
            return {"by_situation": {}, "top_situations": []}

    def analyze_emotion_patterns(self) -> Dict[str, Any]:
        """ 감정 패턴 분석 (라벨링 전이 활용 강화 + 메타데이터) """
        try:
            patterns = {
                "emotional_transitions": self._analyze_emotional_transitions(),
                "emotion_sequences": self._analyze_emotion_sequences(),
                "transition_dynamics": self._analyze_transition_dynamics()
            }
            patterns["metadata"] = {
                "analysis_timestamp": self._now().isoformat(),
                "data_points": sum(len(v) for v in self.emotion_history.values()),
                "confidence_score": self._calculate_analysis_confidence()
            }
            logger.debug("감정 패턴 분석 완료")
            return patterns
        except Exception as e:
            logger.error(f"감정 패턴 분석 중 오류: {str(e)}")
            return {}

    def _analyze_emotion_sequences(self) -> Dict[str, Any]:
        """ 연속 동일 감정 구간 탐지 """
        results = {"sequences": [], "sequence_count": 0}
        all_entries: List[Tuple[datetime, str, str, float]] = []
        for emotion, records in self.emotion_history.items():
            for (ts, sit, intensity) in records:
                all_entries.append((ts, emotion, sit, intensity))
        all_entries.sort(key=lambda x: x[0])
        if not all_entries:
            return results

        current_emotion = all_entries[0][1]
        current_start_time = all_entries[0][0]
        run_length = 1
        for i in range(1, len(all_entries)):
            _, emo, _, _ = all_entries[i]
            if emo == current_emotion:
                run_length += 1
            else:
                results["sequences"].append({
                    "emotion": current_emotion,
                    "start_time": current_start_time.isoformat(),
                    "run_length": run_length
                })
                current_emotion = emo
                current_start_time = all_entries[i][0]
                run_length = 1
        results["sequences"].append({
            "emotion": current_emotion,
            "start_time": current_start_time.isoformat(),
            "run_length": run_length
        })
        results["sequence_count"] = len(results["sequences"])
        return results

    def _analyze_transition_dynamics(self) -> Dict[str, Any]:
        """ 전이(transition)의 속도/간격/빈도 추적 """
        result = {
            "total_transitions": 0,
            "avg_time_between_transitions": 0.0,
            "longest_gap_seconds": 0.0,
            "shortest_gap_seconds": 0.0
        }
        all_entries: List[Tuple[datetime, str, str, float]] = []
        for emotion, records in self.emotion_history.items():
            for (ts, sit, intensity) in records:
                all_entries.append((ts, emotion, sit, intensity))
        all_entries.sort(key=lambda x: x[0])
        if len(all_entries) < 2:
            return result

        transition_count = 0
        time_gaps: List[float] = []
        for i in range(1, len(all_entries)):
            prev_ts, prev_emo, _, _ = all_entries[i - 1]
            curr_ts, curr_emo, _, _ = all_entries[i]
            if curr_emo != prev_emo:
                transition_count += 1
                gap_seconds = max(0.0, (curr_ts - prev_ts).total_seconds())
                time_gaps.append(gap_seconds)

        if time_gaps:
            result["total_transitions"] = transition_count
            result["avg_time_between_transitions"] = round(sum(time_gaps) / len(time_gaps), 3)
            result["longest_gap_seconds"] = round(max(time_gaps), 3)
            result["shortest_gap_seconds"] = round(min(time_gaps), 3)
        return result

    def _analyze_emotional_transitions(self) -> Dict[str, Any]:
        """ 감정 전이 패턴 분석 """
        try:
            from collections import defaultdict
            all_entries = []
            for emotion, record_list in self.emotion_history.items():
                for (ts, situation, intensity) in record_list:
                    all_entries.append({
                        "emotion": emotion,
                        "timestamp": ts,
                        "intensity_score": float(intensity),
                        "situation": situation
                    })
            all_entries.sort(key=lambda x: x["timestamp"])

            transitions = defaultdict(lambda: {
                "count": 0,
                "avg_intensity": 0.0,
                "labeling_match": False,
                "supporting_patterns": []
            })

            for i in range(len(all_entries) - 1):
                from_emo = all_entries[i]["emotion"]
                from_int = all_entries[i]["intensity_score"]
                to_emo = all_entries[i + 1]["emotion"]
                to_int = all_entries[i + 1]["intensity_score"]

                key = (from_emo, to_emo)
                tr = transitions[key]
                tr["count"] += 1
                pair_avg = (from_int + to_int) / 2.0
                tr["avg_intensity"] = ((tr["avg_intensity"] * (tr["count"] - 1)) + pair_avg) / tr["count"]

            valid_transitions = self._match_with_labeling_transitions(transitions)
            stats = self._calculate_transition_statistics(valid_transitions)
            return {"transitions": dict(valid_transitions), "statistics": stats}
        except Exception as e:
            logger.error(f"감정 전이 분석 중 오류: {str(e)}")
            return {}

    def _match_with_labeling_transitions(self, transitions: Dict[Tuple[str, str], Dict[str, Any]]) -> Dict[Tuple[str, str], Dict[str, Any]]:
        """실제 전이를 라벨링 데이터와 매칭"""
        try:
            valid = {}
            for (from_emotion, to_emotion), tdata in transitions.items():
                labeling_match = False
                supporting = []

                from_cat = from_emotion.split('-')[0]
                to_cat = to_emotion.split('-')[0]
                from_data = self.emotions_data.get(from_cat, {}) or {}

                if self._check_category_transition(from_data, from_cat, to_cat):
                    labeling_match = True
                    supporting.append("category_level_match")

                if '-' in from_emotion and '-' in to_emotion:
                    if self._check_detailed_transition(from_data, from_emotion, to_emotion):
                        labeling_match = True
                        supporting.append("detailed_level_match")

                if labeling_match:
                    valid[(from_emotion, to_emotion)] = {
                        **tdata,
                        "labeling_match": True,
                        "supporting_patterns": supporting
                    }
            return valid
        except Exception as e:
            logger.error(f"전이 매칭 중 오류: {str(e)}")
            return {}

    def _check_category_transition(self, from_data: Dict[str, Any], from_category: str, to_category: str) -> bool:
        """대표감정 레벨 전이 확인"""
        try:
            transitions = from_data.get('emotion_transitions', {}) or from_data.get('emotion_profile', {}).get('emotion_transitions', {})
            patterns = (transitions or {}).get('patterns', []) or []
            for pattern in patterns:
                if (pattern.get('from_emotion') == from_category and pattern.get('to_emotion') == to_category):
                    return True
            return False
        except Exception as e:
            logger.error(f"대표감정 전이 확인 중 오류: {str(e)}")
            return False

    def _check_detailed_transition(self, from_data: Dict[str, Any], from_emotion: str, to_emotion: str) -> bool:
        """세부감정 레벨 전이 확인"""
        try:
            subs = from_data.get('sub_emotions', {}) or from_data.get('emotion_profile', {}).get('sub_emotions', {}) or {}
            if not subs:
                return False
            from_sub = from_emotion.split('-', 1)[1]
            sub_data = subs.get(from_sub, {}) or {}
            patterns = (sub_data.get('emotion_transitions', {}) or {}).get('patterns', []) or []
            for pattern in patterns:
                if (pattern.get('from_emotion') == from_emotion and pattern.get('to_emotion') == to_emotion):
                    return True
            return False
        except Exception as e:
            logger.error(f"세부감정 전이 확인 중 오류: {str(e)}")
            return False

    def _calculate_transition_statistics(self, transitions: Dict[Tuple[str, str], Dict[str, Any]]) -> Dict[str, Any]:
        """전이 패턴 통계 계산"""
        try:
            stats = {"total_transitions": len(transitions), "labeling_match_ratio": 0.0, "avg_intensity": 0.0, "common_patterns": []}
            if not transitions:
                return stats
            matched = sum(1 for t in transitions.values() if t.get("labeling_match"))
            stats["labeling_match_ratio"] = matched / len(transitions)
            stats["avg_intensity"] = sum(float(t.get("avg_intensity", 0.0)) for t in transitions.values()) / len(transitions)
            pattern_counts = Counter()
            for t in transitions.values():
                for p in t.get("supporting_patterns", []):
                    pattern_counts[p] += 1
            stats["common_patterns"] = [{"pattern": p, "count": c} for p, c in pattern_counts.most_common(3)]
            return stats
        except Exception as e:
            logger.error(f"전이 통계 계산 중 오류: {str(e)}")
            return {}

    def _calculate_analysis_confidence(self) -> float:
        """분석 신뢰도 계산"""
        try:
            if not self.emotion_history:
                return 0.0
            data_conf = min(sum(len(v) for v in self.emotion_history.values()) / 20, 1.0)
            analysis = self._analyze_emotional_transitions() or {}
            stats = analysis.get("statistics", {}) or {}
            match_conf = float(stats.get("labeling_match_ratio", 0.0))
            return round(data_conf * 0.4 + match_conf * 0.6, 2)
        except Exception as e:
            logger.error(f"신뢰도 계산 중 오류: {str(e)}")
            return 0.0

    def _load_transition_scores_from_labeling_data(self, transitions: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """라벨링 전이 패턴으로 점수 보정"""
        try:
            for category_name, category_data in self.emotions_data.items():
                emotion_transitions = category_data.get("emotion_transitions", {}) or category_data.get("emotion_profile", {}).get("emotion_transitions", {}) or {}
                patterns_list = (emotion_transitions.get("patterns", []) or [])
                for pattern in patterns_list:
                    from_e = pattern.get("from_emotion")
                    to_e = pattern.get("to_emotion")
                    if not from_e or not to_e:
                        continue
                    triggers = pattern.get("triggers", []) or []
                    factor = 1.0 + (0.05 * len(triggers))
                    if from_e in transitions and to_e in transitions[from_e]:
                        transitions[from_e][to_e] *= factor
            return transitions
        except Exception as e:
            logger.error(f"전이 점수 보정 중 오류 발생: {str(e)}")
            return transitions

    def _get_dominant_emotions(self) -> Dict[str, float]:
        """ 지배적 감정(평균 강도) """
        strengths = {}
        for emotion, history in self.emotion_history.items():
            if history:
                avg = sum(self._safe_float(i, 0.0) for _, _, i in history) / max(1, len(history))
                strengths[emotion] = round(avg, 3)
            else:
                strengths[emotion] = 0.0
        return strengths

    def _analyze_transitions(self) -> Dict[str, Dict[str, Any]]:
        """ 심층 전이 패턴(라벨링+실측+체인/문맥/시간/통계) """
        transitions = defaultdict(lambda: defaultdict(float))
        try:
            self._extract_transitions_from_labeling_data(transitions)
            self._analyze_actual_transitions(transitions)
            chain_patterns = self._analyze_chain_patterns()
            contextual_transitions = self._analyze_contextual_transitions()
            temporal_transitions = self._analyze_temporal_transitions()
            statistical_metrics = self._calculate_overall_transition_statistics()
            return {
                "transitions": {k: dict(v) for k, v in transitions.items()},
                "chain_patterns": chain_patterns,
                "contextual_transitions": contextual_transitions,
                "temporal_transitions": temporal_transitions,
                "statistical_metrics": statistical_metrics
            }
        except Exception as e:
            logger.error(f"전이 패턴 분석 중 오류 발생: {str(e)}")
            return {}

    def _extract_transitions_from_labeling_data(self, transitions: Dict) -> None:
        """라벨링 전이 정보를 transitions에 누적"""
        try:
            for _, emotion_data in self.emotions_data.items():
                emotion_transitions = emotion_data.get('emotion_transitions', {}) or emotion_data.get('emotion_profile', {}).get('emotion_transitions', {}) or {}
                for pattern in (emotion_transitions.get('patterns', []) or []):
                    from_e = pattern.get('from_emotion')
                    to_e = pattern.get('to_emotion')
                    if not from_e or not to_e:
                        continue
                    triggers = pattern.get('triggers', []) or []
                    weight = 1.0 + (len(triggers) * 0.1)
                    transitions[from_e][to_e] += weight
        except Exception as e:
            logger.error(f"라벨링 데이터 전이 정보 추출 중 오류: {str(e)}")

    def _analyze_actual_transitions(self, transitions: Dict) -> None:
        """실제 히스토리 기반 전이 가중 누적"""
        try:
            entries = []
            for emotion, history in self.emotion_history.items():
                for ts, situation, intensity in history:
                    entries.append((ts, emotion, self._safe_float(intensity)))
            entries.sort(key=lambda x: x[0])
            for i in range(len(entries) - 1):
                cur_ts, from_emotion, cur_int = entries[i]
                next_ts, to_emotion, next_int = entries[i + 1]
                time_diff = max(0.0, (next_ts - cur_ts).total_seconds())
                time_weight = 1.0 if time_diff <= 3600 else 0.5
                intensity_weight = (cur_int + next_int) / 2.0
                transitions[from_emotion][to_emotion] += time_weight * intensity_weight
        except Exception as e:
            logger.error(f"실제 전이 패턴 분석 중 오류: {str(e)}")


    def _analyze_chain_patterns(self) -> Dict[str, Any]:
        """연쇄 전이 패턴 분석"""
        chain_patterns = {"frequent_chains": [], "cyclic_patterns": [], "transition_sequences": []}
        try:
            seq: List[Tuple[str, datetime]] = []
            for _, history in self.emotion_history.items():
                for entry in sorted(history, key=lambda x: x[0]):
                    seq.append((entry[1], entry[0]))
            seq.sort(key=lambda x: x[1])
            chain_patterns["frequent_chains"] = self._find_frequent_chains(seq)
            chain_patterns["cyclic_patterns"] = self._find_cyclic_patterns(seq)
            chain_patterns["transition_sequences"] = self._analyze_transition_sequences(seq)
        except Exception as e:
            logger.error(f"연쇄 패턴 분석 중 오류: {str(e)}")
        return chain_patterns

    def _find_frequent_chains(self, emotion_sequence: List[Tuple[str, datetime]]) -> List[Dict[str, Any]]:
        """ 빈발 연쇄 패턴 탐지 """
        results: List[Dict[str, Any]] = []
        if not emotion_sequence:
            return results
        min_length, max_length, min_support = 2, 4, 2
        seq_len = len(emotion_sequence)
        emo_only = [e for e, _ in emotion_sequence]
        chain_counts: Dict[Tuple[str, ...], int] = {}
        for length in range(min_length, min(seq_len, max_length) + 1):
            for start in range(seq_len - length + 1):
                chain = tuple(emo_only[start:start + length])
                chain_counts[chain] = chain_counts.get(chain, 0) + 1
        for chain, count in chain_counts.items():
            if count >= min_support:
                support = count / max(1, (seq_len - len(chain) + 1))
                results.append({"sequence": list(chain), "count": count, "support": round(support, 3)})
        return sorted(results, key=lambda x: x["count"], reverse=True)

    def _find_cyclic_patterns(self, emotion_sequence: List[Tuple[str, datetime]]) -> List[Dict[str, Any]]:
        """ 순환 패턴 탐지 """
        results: List[Dict[str, Any]] = []
        if not emotion_sequence:
            return results
        min_cycle_length, max_cycle_length = 2, 5
        emo_only = [e for e, _ in emotion_sequence]
        n = len(emo_only)
        for L in range(min_cycle_length, min(n, max_cycle_length) + 1):
            for i in range(n - L + 1):
                slice_ = emo_only[i:i + L]
                if slice_[0] != slice_[-1]:
                    continue
                occ = 1
                for j in range(i + 1, n - L + 1):
                    if emo_only[j:j + L] == slice_:
                        occ += 1
                if occ >= 2:
                    results.append({"pattern": slice_, "length": L, "occurrences": occ})
        return sorted(results, key=lambda x: x["occurrences"], reverse=True)

    def _analyze_transition_sequences(self, emotion_sequence: List[Tuple[str, datetime]]) -> List[Dict[str, Any]]:
        """ 2-step / 3-step 전이 시퀀스 분석 """
        results: List[Dict[str, Any]] = []
        if not emotion_sequence or len(emotion_sequence) < 2:
            return results
        n = len(emotion_sequence)
        two, three = defaultdict(int), defaultdict(int)
        for i in range(n - 1):
            e1 = emotion_sequence[i][0]
            e2 = emotion_sequence[i + 1][0]
            two[(e1, e2)] += 1
            if i + 2 < n:
                e3 = emotion_sequence[i + 2][0]
                three[(e1, e2, e3)] += 1
        results.append({"two_step_transitions": [{"transition": list(k), "count": v} for k, v in two.items()]})
        results.append({"three_step_transitions": [{"transition": list(k), "count": v} for k, v in three.items()]})
        return results

    def _analyze_contextual_transitions(self) -> Dict[str, Any]:
        """문맥 기반 전이 분석"""
        patterns: Dict[str, Any] = {}
        try:
            for situation, impact in self.situation_impacts.items():
                rel = list(self._get(impact, "related_emotions", []))
                ctx_w = float(self._get(impact, "context_weight", 1.0))
                trans = defaultdict(lambda: defaultdict(float))
                for i in range(len(rel) - 1):
                    trans[rel[i]][rel[i + 1]] += ctx_w
                patterns[situation] = {k: dict(v) for k, v in trans.items()}
        except Exception as e:
            logger.error(f"문맥 기반 전이 분석 중 오류: {str(e)}")
        return patterns

    def _analyze_temporal_transitions(self) -> Dict[str, Any]:
        """시간 기반 전이 분석"""
        hourly = defaultdict(lambda: defaultdict(float))
        daily = defaultdict(lambda: defaultdict(float))
        try:
            entries = []
            for emotion, history in self.emotion_history.items():
                for ts, situation, intensity in history:
                    entries.append((ts, emotion))
            entries.sort(key=lambda x: x[0])
            for i in range(len(entries) - 1):
                cur_ts, from_emotion = entries[i]
                next_ts, to_emotion = entries[i + 1]
                hour = cur_ts.hour
                day = cur_ts.strftime('%A')
                hourly[hour][(from_emotion, to_emotion)] += 1
                daily[day][(from_emotion, to_emotion)] += 1
            return {
                "hourly_transitions": {h: dict(v) for h, v in hourly.items()},
                "daily_transitions": {d: dict(v) for d, v in daily.items()},
                "trend_analysis": self._analyze_transition_trends()
            }
        except Exception as e:
            logger.error(f"시간 기반 전이 분석 중 오류: {str(e)}")
            return {"hourly_transitions": {}, "daily_transitions": {}, "trend_analysis": {}}


    def _analyze_transition_trends(self) -> Dict[str, Any]:
        """ 전이 트렌드 분석 """
        results = {"hourly_extremes": {}, "daily_extremes": {}, "hour_trend": 0, "day_trend": 0}
        hourly_counts: Dict[int, int] = defaultdict(int)
        day_counts: Dict[str, int] = defaultdict(int)
        for _, history in self.emotion_history.items():
            for ts, _, _ in sorted(history, key=lambda x: x[0]):
                hourly_counts[ts.hour] += 1
                day_counts[ts.strftime('%A')] += 1

        def extremes(d: Dict[Any, int]) -> Dict[str, Any]:
            if not d:
                return {"max": None, "min": None}
            items = sorted(d.items(), key=lambda x: x[1], reverse=True)
            return {"max": {"key": items[0][0], "value": items[0][1]},
                    "min": {"key": items[-1][0], "value": items[-1][1]}}

        results["hourly_extremes"] = extremes(hourly_counts)
        results["daily_extremes"] = extremes(day_counts)
        first_half = sum(hourly_counts[h] for h in range(0, 12))
        second_half = sum(hourly_counts[h] for h in range(12, 24))
        results["hour_trend"] = int(second_half - first_half)
        weekdays = {"Monday", "Tuesday", "Wednesday", "Thursday", "Friday"}
        weekend = {"Saturday", "Sunday"}
        weekday_sum = sum(v for k, v in day_counts.items() if k in weekdays)
        weekend_sum = sum(v for k, v in day_counts.items() if k in weekend)
        results["day_trend"] = int(weekend_sum - weekday_sum)
        return results

    def _calculate_overall_transition_statistics(self) -> Dict[str, Any]:
        """전이 패턴의 통계 지표"""
        stats = {"transition_probabilities": {}, "stability_metrics": {}, "diversity_metrics": {}}
        try:
            all_transitions = self._collect_all_transitions()
            stats["transition_probabilities"] = self._calculate_transition_probabilities(all_transitions)
            stats["stability_metrics"] = self._calculate_stability_metrics(all_transitions)
            stats["diversity_metrics"] = self._calculate_diversity_metrics(all_transitions)
        except Exception as e:
            logger.error(f"전이 통계 계산 중 오류: {str(e)}")
        return stats

    def _collect_all_transitions(self) -> Dict[str, Dict[str, float]]:
        """ 모든 전이를 모아 하나의 구조로 """
        all_transitions: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))

        # 라벨링 기반
        for _, emotion_data in self.emotions_data.items():
            transitions_data = emotion_data.get('emotion_transitions', {}) or emotion_data.get('emotion_profile', {}).get('emotion_transitions', {}) or {}
            for pattern in (transitions_data.get('patterns', []) or []):
                from_e = pattern.get('from_emotion')
                to_e = pattern.get('to_emotion')
                if not from_e or not to_e:
                    continue
                triggers = pattern.get('triggers', []) or []
                all_transitions[from_e][to_e] += 1.0 + (len(triggers) * 0.1)

        # 실측 기반
        entries = []
        for emotion, history in self.emotion_history.items():
            for ts, situation, intensity in history:
                entries.append((ts, emotion, self._safe_float(intensity)))
        entries.sort(key=lambda x: x[0])
        for i in range(len(entries) - 1):
            cur_ts, from_emotion, cur_int = entries[i]
            next_ts, to_emotion, next_int = entries[i + 1]
            time_diff = max(0.0, (next_ts - cur_ts).total_seconds())
            time_weight = 1.0 if time_diff <= 3600 else 0.5
            intensity_weight = (cur_int + next_int) / 2.0
            all_transitions[from_emotion][to_emotion] += time_weight * intensity_weight

        return {k: dict(v) for k, v in all_transitions.items()}


    def _calculate_transition_probabilities(self, all_transitions: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        probs: Dict[str, Dict[str, float]] = {}
        for from_e, to_dict in (all_transitions or {}).items():
            total = sum(to_dict.values())
            if total <= 0:
                continue
            probs[from_e] = {to_e: round(w / total, 3) for to_e, w in to_dict.items()}
        return probs

    def _calculate_stability_metrics(
            self,
            all_transitions: Optional[Mapping[str, Mapping[str, _WeightVal]]]
    ) -> Dict[str, Any]:
        at = _normalize_transition_weights(all_transitions)

        result: Dict[str, Any] = {"self_transition_ratio": 0.0, "top_transition": None}
        total = 0.0
        self_sum = 0.0
        top_key: Optional[Tuple[str, str]] = None
        top_w = 0.0

        for f, to_dict in at.items():
            for t, w in to_dict.items():  # 여긴 항상 float
                total += w
                if f == t:
                    self_sum += w
                if w > top_w:
                    top_w, top_key = w, (f, t)

        result["self_transition_ratio"] = round(self_sum / total, 3) if total > 0 else 0.0
        if top_key is not None:
            result["top_transition"] = {
                "from": top_key[0],
                "to": top_key[1],
                "weight": round(top_w, 3),
            }
        return result

    def _calculate_diversity_metrics(self, all_transitions: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        result = {"distinct_from_count": 0, "distinct_to_count": 0, "transition_entropy": 0.0}
        from_set = set(all_transitions.keys())
        to_set = set()
        wsum = 0.0
        dist: List[float] = []
        for _, to_dict in (all_transitions or {}).items():
            for to_e, w in to_dict.items():
                to_set.add(to_e)
                wsum += w
                dist.append(w)
        result["distinct_from_count"] = len(from_set)
        result["distinct_to_count"] = len(to_set)
        if wsum > 0 and dist:
            import math
            ent = -sum((w / wsum) * math.log2(max(1e-12, w / wsum)) for w in dist)
            result["transition_entropy"] = round(ent, 3)
        return result

    # ------------------------------------------------------------------
    # 시간/지속/요약
    # ------------------------------------------------------------------
    def _analyze_temporal_patterns(self) -> Dict[str, Any]:
        """시간적 패턴 분석"""
        return {
            "daily_patterns": self._analyze_daily_patterns(),
            "emotion_duration": self._analyze_emotion_duration(),  # seconds
            "intensity_changes": self._analyze_intensity_changes()
        }

    def _analyze_daily_patterns(self) -> Dict[str, Dict[int, float]]:
        """일중 감정 패턴(시간대 평균 강도)"""
        hourly: Dict[str, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
        for emotion, history in self.emotion_history.items():
            for ts, _, intensity in history:
                hourly[emotion][ts.hour].append(self._safe_float(intensity, 0.0))
        return {
            emo: {h: (sum(vals) / max(1, len(vals))) for h, vals in hours.items()}
            for emo, hours in hourly.items()
        }

    def _analyze_emotion_duration(self) -> Dict[str, float]:
        """감정 지속 시간(초)"""
        durations: Dict[str, float] = {}
        for emotion, history in self.emotion_history.items():
            if not history:
                continue
            first = min(t for t, _, _ in history if isinstance(t, datetime))
            last = max(t for t, _, _ in history if isinstance(t, datetime))
            durations[emotion] = max(0.0, (last - first).total_seconds())
        return durations

    def _analyze_intensity_changes(self) -> Dict[str, List[float]]:
        """감정 강도 변화(진행 데이터 기준)"""
        out: Dict[str, List[float]] = {}
        for emotion, prog in self.emotion_progressions.items():
            out[emotion] = list(self._get(prog, "intensity_changes", []))
        return out

    def get_context_summary(self) -> Dict[str, Any]:
        """문맥 정보 요약 (JSON 안전)"""
        return {
            "active_emotions": self._get_dominant_emotions(),
            "recent_situations": self._get_recent_situations(),
            "context_patterns": self.analyze_emotion_patterns(),
            "metrics": dict(self.metrics)
        }

    def _get_recent_situations(self) -> List[Dict[str, Any]]:
        """최근 상황 정보(impact 내림차순)"""
        items = []
        for situation, impact in self.situation_impacts.items():
            items.append({
                "situation": situation,
                "impact_score": float(self._get(impact, "impact_score", 0.0)),
                "related_emotions": list(self._get(impact, "related_emotions", [])),
                "context_weight": float(self._get(impact, "context_weight", 1.0))
            })
        return sorted(items, key=lambda x: x["impact_score"], reverse=True)

    def track_all_emotion_flows(self) -> Dict[str, Any]:
        """모든 감정 흐름 추적 (datetime→ISO 직렬화 안전)"""
        hist: Dict[str, List[Dict[str, Any]]] = {}
        for emo, records in self.emotion_history.items():
            hist[emo] = [{"timestamp": t.isoformat() if isinstance(t, datetime) else str(t),
                          "situation": s, "intensity": float(i)} for (t, s, i) in records]
        prog = {}
        for emo, data in self.emotion_progressions.items():
            prog[emo] = {
                "path": list(self._get(data, "progression_path", [])),
                "intensities": list(self._get(data, "intensity_changes", []))
            }
        return {"emotion_histories": hist, "progression_data": prog}


# =============================================================================
# ContextExtractor
# =============================================================================
class ContextExtractor:
    def __init__(self, emotions_data: Dict[str, Any] = None):
        # EMOTIONS.json 로드 (하드코딩 대신)
        if emotions_data is None:
            self.emotions_data = self._load_emotions_data()
        else:
            self.emotions_data = emotions_data
            
        self.emotion_cache: Dict[str, Any] = {}
        self._intensity_prior_weights = {'low': 0.95, 'medium': 1.0, 'high': 1.05}
        self.calibrator = None
        self._rx_time: Optional[Pattern[str]] = None
        self._rx_location: Optional[Pattern[str]] = None
        self._rx_social: Optional[Pattern[str]] = None
        self._rx_triggers: Optional[Pattern[str]] = None
        self._rx_phrase: Optional[Pattern[str]] = None
        self._phrase2emo = defaultdict(list)
        
        # EMOTIONS.json 기반 컨텍스트 패턴 캐시 초기화
        self._context_patterns_cache = {}
        self._temporal_patterns_cache = {}
        self._spatial_patterns_cache = {}
        self._social_patterns_cache = {}
        
        # 메트릭 초기화
        self.metrics = {
            'context_scores': defaultdict(float),
            'confidence_scores': defaultdict(float),
            'relevance_scores': defaultdict(float),
            'processing_time': 0.0,
            'memory_usage': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # 의존성 모듈 초기화
        self.pattern_extractor = None
        self.intensity_analyzer = None
        self.transition_analyzer = None
        self.is_ready = False
        
        # 메모리 모니터 초기화
        try:
            self.memory_monitor = MemoryMonitor(512 * 1024 * 1024)  # 512MB 기본값
        except Exception:
            self.memory_monitor = None
        
        # 안전한 패턴 로딩
        try:
            self._load_context_patterns_recursive()
        except Exception as e:
            print(f"패턴 로딩 실패: {e}")
            # 기본 패턴으로 폴백
            self._load_default_patterns()

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

    def _load_default_patterns(self) -> None:
        """기본 패턴 로딩 (폴백)"""
        # 기본 시간적 패턴
        self._temporal_patterns_cache["default"] = [
            "오전", "오후", "아침", "저녁", "밤", "주말", "평일", 
            "morning", "afternoon", "evening", "night", "weekend", "weekday"
        ]
        
        # 기본 공간적 패턴
        self._spatial_patterns_cache["default"] = [
            "서울", "부산", "카페", "집", "회사", "학교", "병원", "공원",
            "Seoul", "Busan", "cafe", "home", "office", "school", "hospital", "park"
        ]
        
        # 기본 사회적 패턴
        self._social_patterns_cache["default"] = [
            "친구", "가족", "동료", "선생님", "부모", "아이",
            "friend", "family", "colleague", "teacher", "parent", "child"
        ]
        
        # 기본 컨텍스트 패턴
        self._context_patterns_cache["default"] = {
            "triggers": ["happy", "sad", "angry", "joy", "pleasure"],
            "situations": ["at home", "at work", "with family", "with friends"],
            "context_clues": ["today", "now", "always", "never"],
            "emotional_markers": ["feeling", "emotion", "mood"]
        }

    def _load_context_patterns_recursive(self, emotion_data: Dict[str, Any] = None, path: str = "") -> None:
        """EMOTIONS.json에서 컨텍스트 패턴을 재귀적으로 로드"""
        if emotion_data is None:
            emotion_data = self.emotions_data
            
        for emotion_key, emotion_info in emotion_data.items():
            current_path = f"{path}.{emotion_key}" if path else emotion_key
            
            # 하위 감정 재귀 처리
            if isinstance(emotion_info, dict):
                if "sub_emotions" in emotion_info:
                    self._load_context_patterns_recursive(emotion_info["sub_emotions"], current_path)
                
                # 컨텍스트 패턴 추출
                context_patterns = self._extract_context_patterns(emotion_info)
                if context_patterns:
                    self._context_patterns_cache[current_path] = context_patterns
                    
                # 시간적 패턴 추출
                temporal_patterns = self._extract_temporal_patterns(emotion_info)
                if temporal_patterns:
                    self._temporal_patterns_cache[current_path] = temporal_patterns
                    
                # 공간적 패턴 추출
                spatial_patterns = self._extract_spatial_patterns(emotion_info)
                if spatial_patterns:
                    self._spatial_patterns_cache[current_path] = spatial_patterns
                    
                # 사회적 패턴 추출
                social_patterns = self._extract_social_patterns(emotion_info)
                if social_patterns:
                    self._social_patterns_cache[current_path] = social_patterns

    def _extract_context_patterns(self, emotion_info: Dict[str, Any], emotion_path: str = "") -> Dict[str, Any]:
        """감정 정보에서 컨텍스트 패턴 추출"""
        patterns = {
            "triggers": [],
            "situations": [],
            "context_clues": [],
            "emotional_markers": []
        }
        
        # intensity_examples에서 컨텍스트 추출
        if "emotion_profile" in emotion_info:
            profile = emotion_info["emotion_profile"]
            if "intensity_levels" in profile:
                intensity_levels = profile["intensity_levels"]
                if "intensity_examples" in intensity_levels:
                    examples = intensity_levels["intensity_examples"]
                    for level, example_list in examples.items():
                        if isinstance(example_list, list):
                            patterns["situations"].extend(example_list)
        
        # keywords에서 트리거 추출
        if "keywords" in emotion_info:
            patterns["triggers"].extend(emotion_info["keywords"])
            
        # triggers에서 컨텍스트 단서 추출
        if "triggers" in emotion_info:
            patterns["context_clues"].extend(emotion_info["triggers"])
            
        return patterns

    def _extract_temporal_patterns(self, emotion_info: Dict[str, Any], emotion_path: str = "") -> List[str]:
        """감정 정보에서 시간적 패턴 추출"""
        patterns = []
        
        # 시간 관련 키워드 (하드코딩된 트리거)
        temporal_keywords = ["오전", "오후", "아침", "저녁", "밤", "주말", "평일", "morning", "afternoon", "evening", "night", "weekend", "weekday"]
        patterns.extend(temporal_keywords)
        
        # intensity_examples에서 시간적 컨텍스트 추출
        if "emotion_profile" in emotion_info:
            profile = emotion_info["emotion_profile"]
            if "intensity_levels" in profile:
                intensity_levels = profile["intensity_levels"]
                if "intensity_examples" in intensity_levels:
                    examples = intensity_levels["intensity_examples"]
                    for level, example_list in examples.items():
                        if isinstance(example_list, list):
                            for example in example_list:
                                if isinstance(example, str):
                                    # 시간 관련 표현 추출
                                    for keyword in temporal_keywords:
                                        if keyword in example:
                                            patterns.append(example)
            
        return patterns

    def _extract_spatial_patterns(self, emotion_info: Dict[str, Any], emotion_path: str = "") -> List[str]:
        """감정 정보에서 공간적 패턴 추출"""
        patterns = []
        
        # 공간 관련 키워드 (하드코딩된 트리거)
        spatial_keywords = ["서울", "부산", "카페", "집", "회사", "학교", "병원", "공원", "Seoul", "Busan", "cafe", "home", "office", "school", "hospital", "park"]
        patterns.extend(spatial_keywords)
        
        # intensity_examples에서 공간적 컨텍스트 추출
        if "emotion_profile" in emotion_info:
            profile = emotion_info["emotion_profile"]
            if "intensity_levels" in profile:
                intensity_levels = profile["intensity_levels"]
                if "intensity_examples" in intensity_levels:
                    examples = intensity_levels["intensity_examples"]
                    for level, example_list in examples.items():
                        if isinstance(example_list, list):
                            for example in example_list:
                                if isinstance(example, str):
                                    # 공간 관련 표현 추출
                                    for keyword in spatial_keywords:
                                        if keyword in example:
                                            patterns.append(example)
            
        return patterns

    def _extract_social_patterns(self, emotion_info: Dict[str, Any], emotion_path: str = "") -> List[str]:
        """감정 정보에서 사회적 패턴 추출"""
        patterns = []
        
        # 사회적 키워드 (하드코딩된 트리거)
        social_keywords = ["친구", "가족", "동료", "선생님", "부모", "아이", "friend", "family", "colleague", "teacher", "parent", "child"]
        patterns.extend(social_keywords)
        
        # intensity_examples에서 사회적 컨텍스트 추출
        if "emotion_profile" in emotion_info:
            profile = emotion_info["emotion_profile"]
            if "intensity_levels" in profile:
                intensity_levels = profile["intensity_levels"]
                if "intensity_examples" in intensity_levels:
                    examples = intensity_levels["intensity_examples"]
                    for level, example_list in examples.items():
                        if isinstance(example_list, list):
                            for example in example_list:
                                if isinstance(example, str):
                                    # 사회적 표현 추출
                                    for keyword in social_keywords:
                                        if keyword in example:
                                            patterns.append(example)
            
        return patterns

    def extract_context_new(self, text: str) -> Dict[str, Any]:
        """EMOTIONS.json 기반 컨텍스트 추출 (새로운 구현)"""
        try:
            # 문장 분리
            sentences = _split_into_sentences(text)
            if not sentences:
                sentences = [text.strip()]
            
            # 각 문장에서 컨텍스트 추출
            sentence_contexts = []
            for i, sentence in enumerate(sentences):
                context = self._extract_sentence_context(sentence)
                sentence_contexts.append({
                    "sentence_index": i,
                    "sentence": sentence,
                    "context": context
                })
            
            # 전체 텍스트 컨텍스트 요약
            summary_context = self._summarize_context(sentence_contexts)
            
            return {
                "sentence_contexts": sentence_contexts,
                "summary_context": summary_context,
                "context_analysis": {
                    "total_sentences": len(sentences),
                    "context_types_found": len(summary_context),
                    "context_score": self._calculate_context_score(sentence_contexts)
                },
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "success": True
            }
            
        except Exception as e:
            return {
                "sentence_contexts": [],
                "summary_context": {},
                "context_analysis": {
                    "total_sentences": 0,
                    "context_types_found": 0,
                    "context_score": 0.0
                },
                "error": str(e),
                "success": False
            }

    def _extract_sentence_context(self, sentence: str) -> Dict[str, Any]:
        """문장에서 컨텍스트 추출"""
        context = {
            "temporal_markers": [],
            "spatial_markers": [],
            "social_markers": [],
            "emotional_context": [],
            "situational_context": []
        }
        
        sentence_lower = sentence.lower()
        
        # 시간적 마커 감지 (한국어 특화)
        korean_temporal_keywords = [
            "오전", "오후", "아침", "저녁", "밤", "새벽", "점심", "식사시간",
            "주말", "평일", "휴일", "공휴일", "방학", "학기",
            "오늘", "어제", "내일", "모레", "글피", "이번주", "다음주", "지난주",
            "월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일",
            "1월", "2월", "3월", "4월", "5월", "6월", "7월", "8월", "9월", "10월", "11월", "12월",
            "봄", "여름", "가을", "겨울", "계절"
        ]
        english_temporal_keywords = ["morning", "afternoon", "evening", "night", "weekend", "weekday", "today", "yesterday", "tomorrow"]
        
        # 한국어 키워드 매칭 (부분 매칭 허용)
        for keyword in korean_temporal_keywords:
            if keyword in sentence:
                context["temporal_markers"].append({
                    "pattern": keyword,
                    "emotion": "시간적",
                    "confidence": 0.9,
                    "language": "korean"
                })
        
        # 영어 키워드 매칭
        for keyword in english_temporal_keywords:
            if keyword in sentence_lower:
                context["temporal_markers"].append({
                    "pattern": keyword,
                    "emotion": "시간적",
                    "confidence": 0.8,
                    "language": "english"
                })
        
        # EMOTIONS.json 시간적 패턴 매칭
        for emotion_path, temporal_patterns in self._temporal_patterns_cache.items():
            for pattern in temporal_patterns:
                if pattern.lower() in sentence_lower:
                    context["temporal_markers"].append({
                        "pattern": pattern,
                        "emotion": emotion_path,
                        "confidence": 0.8
                    })
        
        # 공간적 마커 감지 (한국어 특화)
        korean_spatial_keywords = [
            # 장소
            "집", "회사", "학교", "병원", "공원", "카페", "도서관", "상점", "마트", "백화점",
            "식당", "극장", "영화관", "놀이공원", "해변", "산", "강", "호수",
            # 한국 도시
            "서울", "부산", "대구", "인천", "광주", "대전", "울산", "세종", "수원", "성남",
            "고양", "용인", "부천", "화성", "안산", "안양", "평택", "시흥", "김포", "의정부",
            # 방향/위치
            "집에서", "회사에서", "학교에서", "병원에서", "공원에서", "카페에서",
            "위에", "아래에", "앞에", "뒤에", "옆에", "가운데", "중앙에",
            "안에", "밖에", "근처에", "멀리", "가까이"
        ]
        english_spatial_keywords = ["home", "office", "school", "hospital", "park", "cafe", "Seoul", "Busan", "at home", "at work"]
        
        # 한국어 키워드 매칭
        for keyword in korean_spatial_keywords:
            if keyword in sentence:
                context["spatial_markers"].append({
                    "pattern": keyword,
                    "emotion": "공간적",
                    "confidence": 0.9,
                    "language": "korean"
                })
        
        # 영어 키워드 매칭
        for keyword in english_spatial_keywords:
            if keyword in sentence_lower:
                context["spatial_markers"].append({
                    "pattern": keyword,
                    "emotion": "공간적",
                    "confidence": 0.8,
                    "language": "english"
                })
        
        # EMOTIONS.json 공간적 패턴 매칭
        for emotion_path, spatial_patterns in self._spatial_patterns_cache.items():
            for pattern in spatial_patterns:
                if pattern.lower() in sentence_lower:
                    context["spatial_markers"].append({
                        "pattern": pattern,
                        "emotion": emotion_path,
                        "confidence": 0.8
                    })
        
        # 사회적 마커 감지 (한국어 특화)
        korean_social_keywords = [
            # 가족 관계
            "가족", "부모", "아버지", "어머니", "아빠", "엄마", "형", "누나", "동생", "언니", "오빠",
            "할아버지", "할머니", "조부모", "손자", "손녀", "조카", "삼촌", "이모", "고모",
            # 사회 관계
            "친구", "동료", "선생님", "교수", "상사", "부하", "후배", "선배", "동급생", "급우",
            "이웃", "사장", "직원", "고객", "손님", "회원", "멤버",
            # 관계 표현
            "가족과", "친구와", "동료와", "선생님과", "부모와", "아이와",
            "함께", "같이", "단체로", "모두", "우리", "저희"
        ]
        english_social_keywords = ["friend", "family", "colleague", "teacher", "parent", "child", "with family", "with friends", "with colleagues"]
        
        # 한국어 키워드 매칭
        for keyword in korean_social_keywords:
            if keyword in sentence:
                context["social_markers"].append({
                    "pattern": keyword,
                    "emotion": "사회적",
                    "confidence": 0.9,
                    "language": "korean"
                })
        
        # 영어 키워드 매칭
        for keyword in english_social_keywords:
            if keyword in sentence_lower:
                context["social_markers"].append({
                    "pattern": keyword,
                    "emotion": "사회적",
                    "confidence": 0.8,
                    "language": "english"
                })
        
        # EMOTIONS.json 사회적 패턴 매칭
        for emotion_path, social_patterns in self._social_patterns_cache.items():
            for pattern in social_patterns:
                if pattern.lower() in sentence_lower:
                    context["social_markers"].append({
                        "pattern": pattern,
                        "emotion": emotion_path,
                        "confidence": 0.8
                    })
        
        # 감정적 컨텍스트 감지
        english_emotion_mapping = {
            "happy": "희", "joy": "희", "pleasure": "희", "delight": "희",
            "angry": "노", "mad": "노", "furious": "노", "rage": "노",
            "sad": "애", "depressed": "애", "gloomy": "애", "sorrow": "애",
            "satisfaction": "락", "content": "락", "cheerful": "락"
        }
        
        # 한국어 감정 키워드 매핑 (확장)
        korean_emotion_mapping = {
            # 희 (기쁨)
            "행복": "희", "기쁨": "희", "즐거움": "희", "만족": "희", "환희": "희", "신남": "희",
            "웃음": "희", "미소": "희", "웃겨": "희", "재미": "희", "유쾌": "희", "상쾌": "희",
            "감동": "희", "감탄": "희", "놀라움": "희", "신기": "희", "멋져": "희", "좋아": "희",
            # 노 (분노)
            "분노": "노", "화남": "노", "짜증": "노", "불만": "노", "격분": "노", "성남": "노",
            "화가": "노", "열받": "노", "빡쳐": "노", "싫어": "노", "증오": "노", "혐오": "노",
            "실망": "노", "좌절": "노", "억울": "노", "억울함": "노", "불공평": "노",
            # 애 (슬픔)
            "슬픔": "애", "우울": "애", "절망": "애", "실망": "애", "우울함": "애", "침울": "애",
            "눈물": "애", "울음": "애", "울고": "애", "눈물나": "애", "서글픔": "애", "서운": "애",
            "외로움": "애", "쓸쓸": "애", "허탈": "애", "무력": "애", "힘들어": "애", "지쳐": "애",
            # 락 (안정)
            "안정": "락", "평온": "락", "편안": "락", "만족": "락", "평화": "락", "고요": "락",
            "차분": "락", "여유": "락", "안심": "락", "안락": "락", "쾌적": "락", "시원": "락",
            "깔끔": "락", "정돈": "락", "정리": "락", "완성": "락", "성취": "락", "달성": "락"
        }
        
        # 영어 키워드 매칭
        for english_word, emotion in english_emotion_mapping.items():
            if english_word in sentence_lower:
                context["emotional_context"].append({
                    "emotion": emotion,
                    "pattern": english_word,
                    "confidence": 0.8,
                    "language": "english"
                })
        
        # 한국어 키워드 매칭
        for korean_word, emotion in korean_emotion_mapping.items():
            if korean_word in sentence:
                context["emotional_context"].append({
                    "emotion": emotion,
                    "pattern": korean_word,
                    "confidence": 0.9,
                    "language": "korean"
                })
        
        # EMOTIONS.json 패턴 매칭
        for emotion_path, context_patterns in self._context_patterns_cache.items():
            # 트리거 매칭
            for trigger in context_patterns.get("triggers", []):
                if trigger.lower() in sentence_lower:
                    context["emotional_context"].append({
                        "emotion": emotion_path,
                        "pattern": trigger,
                        "confidence": 0.8
                    })
                    break
            
            # 상황 매칭
            for situation in context_patterns.get("situations", []):
                if isinstance(situation, str) and situation.lower() in sentence_lower:
                    context["situational_context"].append(situation)
        
        return context

    def _summarize_context(self, sentence_contexts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """컨텍스트 요약"""
        summary = {
            "temporal_markers": [],
            "spatial_markers": [],
            "social_markers": [],
            "emotional_context": [],
            "situational_context": []
        }
        
        for sc in sentence_contexts:
            context = sc["context"]
            
            # 시간적 마커
            summary["temporal_markers"].extend(context["temporal_markers"])
            
            # 공간적 마커
            summary["spatial_markers"].extend(context["spatial_markers"])
            
            # 사회적 마커
            summary["social_markers"].extend(context["social_markers"])
            
            # 감정적 컨텍스트
            summary["emotional_context"].extend(context["emotional_context"])
            
            # 상황적 컨텍스트
            summary["situational_context"].extend(context["situational_context"])
        
        # 중복 제거
        summary["situational_context"] = list(set(summary["situational_context"]))
        
        return summary

    def _calculate_context_score(self, sentence_contexts: List[Dict[str, Any]]) -> float:
        """컨텍스트 점수 계산"""
        if not sentence_contexts:
            return 0.0
        
        total_contexts = sum(
            len(sc["context"]["temporal_markers"]) + 
            len(sc["context"]["spatial_markers"]) + 
            len(sc["context"]["social_markers"]) + 
            len(sc["context"]["emotional_context"])
            for sc in sentence_contexts
        )
        avg_contexts_per_sentence = total_contexts / len(sentence_contexts)
        
        # 컨텍스트 점수 (0.0 ~ 1.0)
        context_score = min(1.0, avg_contexts_per_sentence / 3.0)
        return round(context_score, 2)

        # config 대신 직접 설정
        self.config = {
            'cache_size': 4096,
            'memory_limit': 500 * 1024 * 1024,  # 500MB
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

        # 메트릭은 이미 __init__에서 초기화됨

        self._setup_cache()  # ← 실제로 쓰는 캐시로 정비
        # memory_monitor는 이미 __init__에서 초기화됨

        # B4) 불필요한 Heavy 의존 제거/지연 - TfidfVectorizer 지연 로딩
        self.vectorizer = None  # 실제 사용 시에만 지역 임포트/초기화

        # EmotionContextManager 초기화(외부 의존 가정 그대로 유지)
        self.context_manager = EmotionContextManager(self.emotions_data)
        # 상황 기본 가중의 완전 자동화 - priors 주입
        self.context_manager.inject_intensity_priors(self._intensity_prior_weights)

        # 의존성 모듈은 이미 __init__에서 초기화됨

        self._ensure_post_cfg_defaults()
        logger.info("ContextExtractor 초기화 완료")

    def inject_calibrator(self, calibrator) -> None:
        """Complex 모듈의 WeightCalibrator/ConfidenceCalibrator를 연결(옵션)."""
        self.calibrator = calibrator

    # ---------------------------------------------------------------------
    # 내부 공통 유틸
    # ---------------------------------------------------------------------
    def _polarity_roots(self) -> Tuple[Set[str], Set[str]]:
        """_POS_ROOTS/_NEG_ROOTS 값을 바탕으로 기본 긍정/부정 감정 루트를 구축한다."""
        if hasattr(self, "_pos_roots") and hasattr(self, "_neg_roots"):
            return self._pos_roots, self._neg_roots

        pos: Set[str] = set()
        neg: Set[str] = set()
        try:
            for cat, cdata in (self.emotions_data or {}).items():
                rel = (cdata.get("context_patterns", {}) or {}).get("related_emotions") \
                      or (cdata.get("emotion_profile", {}) or {}).get("related_emotions") \
                      or {}
                if rel.get("positive"):
                    pos.add(cat)
                if rel.get("negative"):
                    neg.add(cat)
        except Exception:
            pass
        # H2) 극성 루트 폴백 수정 - 안전 폴백으로 교체
        if not pos:
            pos = {"희", "락"}
        if not neg:
            neg = {"노", "애"}

        self._pos_roots, self._neg_roots = pos, neg
        return pos, neg

    @staticmethod
    def _safe_float(x: Any, default: float = 0.0) -> float:
        try:
            if x is None:
                return default
            return float(x)
        except Exception:
            return default

    @staticmethod
    def _safe_len(x: Any) -> int:
        try:
            if x is None:
                return 0
            if isinstance(x, (list, tuple, set, dict)):
                return len(x)
            if isinstance(x, str):
                return 1 if x.strip() else 0
            return len(x)  # type: ignore[arg-type]
        except Exception:
            return 0

    # ---------------------------------------------------------------------
    # 극성/폴라리티 관련
    # ---------------------------------------------------------------------
    def _emotion_polarity(self, emotion_id: str) -> str:
        """루트 카테고리 기준 polarity 판단: pos/neg/neu"""
        if not emotion_id:
            return "neu"
        root = emotion_id.split("-", 1)[0]
        _POS_ROOTS, _NEG_ROOTS = self._polarity_roots()
        if root in _POS_ROOTS:
            return "pos"
        if root in _NEG_ROOTS:
            return "neg"
        return "neu"

    def _compute_global_polarity(self, items: List[Dict[str, Any]]) -> Dict[str, float]:
        """시퀀스 전체 positive/negative 합산 스코어"""
        pos = neg = 0.0
        for it in items:
            for emo in it.get("emotions", []):
                pol = self._emotion_polarity(emo.get("emotion_id", ""))
                if pol == "pos":
                    pos += self._safe_float(emo.get("confidence"), 0.0)
                elif pol == "neg":
                    neg += self._safe_float(emo.get("confidence"), 0.0)
        total = pos + neg
        return {
            "pos": pos,
            "neg": neg,
            "total": total,
            "dominant": "pos" if pos >= neg else "neg",
            "ratio": (max(pos, neg) / total) if total > 0 else 0.0
        }

    # ---------------------------------------------------------------------
    # 시퀀스 후처리(정규화·게이팅·집계) + prev_emotion 연결
    # ---------------------------------------------------------------------
    def _postprocess_emotion_sequence(
        self,
        raw_sequence: List[Dict[str, Any]],
        context_info: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        if not raw_sequence:
            return []

        cfg = getattr(self, "config", {}) or {}
        top_k = int(cfg.get("sequence_top_k", 5))
        min_score = float(cfg.get("sequence_min_score", 0.21))
        polarity_gate_enabled = bool(cfg.get("polarity_gate_enabled", True))
        polarity_scope = cfg.get("polarity_scope", "both")  # "off"|"sentence"|"document"|"both"
        dominance_thr = float(cfg.get("polarity_dominance_threshold", 0.65))
        opposite_keep_max = int(cfg.get("polarity_opposite_keep_max", 1))
        opposite_downscale = float(cfg.get("polarity_opposite_downscale", 0.6))
        keep_candidates_full = bool(cfg.get("sequence_keep_candidates_full", False))
        full_cap = int(cfg.get("sequence_candidates_full_cap", 100))
        round_digits = int(cfg.get("round_digits", 3))
        intensity_agg = cfg.get("intensity_agg", "avg")  # "avg"|"max"
        per_sentence_cap = int(cfg.get("cap_lengths", {}).get("emotion_sequence_per_sentence", top_k))

        # 동일 문장 병합 → coalesce → 폴라리티 게이트 → 캡핑
        seq = self._merge_duplicate_sentences(raw_sequence)
        processed: List[Dict[str, Any]] = []

        for item in seq:
            sent = item.get("sentence", "")
            cands = list(item.get("candidates_full") or item.get("emotions") or [])
            cands = self._coalesce_emotions(cands)

            if polarity_gate_enabled and polarity_scope in ("sentence", "both") and cands:
                dom_pol, ratio = self._infer_sentence_polarity(cands)
                if dom_pol in ("pos", "neg") and ratio >= dominance_thr:
                    cands = self._apply_polarity_gate_emotions(
                        cands,
                        dominant_polarity=dom_pol,
                        keep_max_opposite=opposite_keep_max,
                        downscale=opposite_downscale
                    )

            # 점수 필터/정렬/캡
            cands = [e for e in cands if self._safe_float(e.get("confidence"), 0.0) >= min_score]
            cands.sort(key=lambda x: self._safe_float(x.get("confidence"), 0.0), reverse=True)
            if per_sentence_cap > 0:
                cands = cands[:min(top_k, per_sentence_cap)]

            # intensity 산출
            if cands:
                if intensity_agg == "max":
                    score = max(self._safe_float(e.get("confidence"), 0.0) for e in cands)
                else:
                    score = sum(self._safe_float(e.get("confidence"), 0.0) for e in cands) / max(1, len(cands))
                score = round(min(max(score, 0.0), 1.0), round_digits)
                label = self._label_intensity(score)
            else:
                score = round(self._safe_float(item.get("intensity", {}).get("score"), 0.0), round_digits)
                score = min(max(score, 0.0), 1.0)
                label = self._label_intensity(score)

            out_item = {
                "sentence": sent,
                "emotions": cands,
                "intensity": {"score": score, "label": label}
            }
            if keep_candidates_full:
                full = list(item.get("candidates_full") or item.get("emotions") or [])
                if full_cap > 0 and len(full) > full_cap:
                    full = full[:full_cap]
                out_item["candidates_full"] = full
            processed.append(out_item)

        # 문서 단위 지배 극성 기반 추가 게이트
        if polarity_gate_enabled and polarity_scope in ("document", "both"):
            dom = self._compute_document_dominant_polarity(processed)
            if dom and dom.get("ratio", 0.0) >= dominance_thr and dom.get("polarity") in ("pos", "neg"):
                new_processed = []
                for item in processed:
                    gated = self._apply_polarity_gate_emotions(
                        list(item.get("emotions") or []),
                        dominant_polarity=dom["polarity"],
                        keep_max_opposite=opposite_keep_max,
                        downscale=opposite_downscale
                    )
                    if gated:
                        if intensity_agg == "max":
                            score = max(self._safe_float(e.get("confidence"), 0.0) for e in gated)
                        else:
                            score = sum(self._safe_float(e.get("confidence"), 0.0) for e in gated) / max(1, len(gated))
                        score = round(min(max(score, 0.0), 1.0), round_digits)
                        label = self._label_intensity(score)
                    else:
                        score = round(self._safe_float(item.get("intensity", {}).get("score"), 0.0), round_digits)
                        score = min(max(score, 0.0), 1.0)
                        label = self._label_intensity(score)
                    new_processed.append({
                        **item,
                        "emotions": gated,
                        "intensity": {"score": score, "label": label}
                    })
                processed = new_processed

        # prev_emotion 연결(상황 영향 가중치용)
        prev_top = None
        for it in processed:
            it["prev_emotion"] = prev_top
            top = (it.get("emotions") or [{}])[0].get("emotion_id")
            if isinstance(top, str) and top:
                prev_top = top

        return processed

    # ② 동일 문장 병합
    def _merge_duplicate_sentences(self, seq: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        bucket: Dict[str, Dict[str, Any]] = {}
        for it in seq or []:
            s = it.get("sentence", "")
            if s not in bucket:
                bucket[s] = {
                    "sentence": s,
                    "candidates_full": list(it.get("candidates_full") or it.get("emotions") or []),
                    "intensity": {"score": self._safe_float(it.get("intensity", {}).get("score"), 0.0)}
                }
            else:
                bucket[s]["candidates_full"].extend(list(it.get("candidates_full") or it.get("emotions") or []))
                old = self._safe_float(bucket[s]["intensity"]["score"], 0.0)
                cur = self._safe_float(it.get("intensity", {}).get("score"), 0.0)
                if cur > old:
                    bucket[s]["intensity"]["score"] = cur
        return list(bucket.values())

    # ③ 감정 후보 coalesce (emotion_id 기준 중복 병합)
    def _coalesce_emotions(self, cands: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        by_id: Dict[str, Dict[str, Any]] = {}
        for c in cands or []:
            eid = c.get("emotion_id")
            if not isinstance(eid, str) or not eid:
                continue
            conf = self._safe_float(c.get("confidence"), 0.0)
            if eid not in by_id or conf > self._safe_float(by_id[eid].get("confidence"), 0.0):
                by_id[eid] = {
                    "emotion_id": eid,
                    "confidence": conf,
                    "emotion_complexity": c.get("emotion_complexity"),
                    "intensity_label": c.get("intensity_label")
                }
        out = list(by_id.values())
        out.sort(key=lambda x: self._safe_float(x.get("confidence"), 0.0), reverse=True)
        return out

    # ④ 문장 단위 지배 극성 추정(pos/neg/neutral, ratio)
    def _infer_sentence_polarity(self, cands: List[Dict[str, Any]]) -> Tuple[str, float]:
        pos = neg = 0.0
        for c in cands or []:
            eid = c.get("emotion_id", "")
            pol = self._get_emotion_polarity(eid)
            if pol == "pos":
                pos += self._safe_float(c.get("confidence"), 0.0)
            elif pol == "neg":
                neg += self._safe_float(c.get("confidence"), 0.0)
        total = pos + neg
        if total <= 0:
            return "neutral", 0.0
        return ("pos", pos / total) if pos >= neg else ("neg", neg / total)

    # ⑤ 감정ID → 극성 변환(간단 규칙: "희","락"=pos / "노","애"=neg / 그 외 neutral)
    def _get_emotion_polarity(self, emotion_id: str) -> str:
        if not emotion_id:
            return "neutral"
        head = emotion_id.split("-", 1)[0]
        if head in ("희", "락"):
            return "pos"
        if head in ("노", "애"):
            return "neg"
        return "neutral"

    # ⑥ (단일 문장용) 극성 게이트
    def _apply_polarity_gate_emotions(
        self,
        cands: List[Dict[str, Any]],
        dominant_polarity: str,
        keep_max_opposite: Optional[int] = None,
        downscale: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        cfg = getattr(self, "config", {}) or {}
        if keep_max_opposite is None:
            keep_max_opposite = int(cfg.get("polarity_opposite_keep_max", 1))
        if downscale is None:
            downscale = float(cfg.get("polarity_opposite_downscale", 0.6))

        cands = list(cands or [])
        if not cands or dominant_polarity not in ("pos", "neg"):
            cands.sort(key=lambda x: self._safe_float(x.get("confidence"), 0.0), reverse=True)
            return cands[: int(cfg.get("sequence_top_k", 5))]

        pos, neg, neu = [], [], []
        for e in cands:
            pol = self._get_emotion_polarity(e.get("emotion_id"))
            (pos if pol == "pos" else neg if pol == "neg" else neu).append(e)

        if dominant_polarity == "pos":
            neg.sort(key=lambda x: self._safe_float(x.get("confidence"), 0.0), reverse=True)
            keep = neg[: max(0, keep_max_opposite)]
            gated = pos + neu + [{**x, "confidence": self._safe_float(x.get("confidence"), 0.0) * downscale} for x in keep]
        else:
            pos.sort(key=lambda x: self._safe_float(x.get("confidence"), 0.0), reverse=True)
            keep = pos[: max(0, keep_max_opposite)]
            gated = neg + neu + [{**x, "confidence": self._safe_float(x.get("confidence"), 0.0) * downscale} for x in keep]

        gated.sort(key=lambda x: self._safe_float(x.get("confidence"), 0.0), reverse=True)
        return gated[: int(cfg.get("sequence_top_k", 5))]

    def _apply_polarity_gate_sequence(
        self,
        emotion_sequence: List[Dict[str, Any]],
        dominant_polarity: str,
        keep_max_opposite: Optional[int] = None,
        downscale: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        cfg = getattr(self, "config", {}) or {}
        rd = int(cfg.get("round_digits", 3))
        out = []
        for item in emotion_sequence or []:
            cands = list(item.get("emotions") or [])
            gated = self._apply_polarity_gate_emotions(
                cands,
                dominant_polarity=dominant_polarity,
                keep_max_opposite=keep_max_opposite,
                downscale=downscale,
            )
            sent_intensity = max((self._safe_float(e.get("confidence"), 0.0) for e in gated), default=0.0)
            out.append({
                "sentence": item.get("sentence", ""),
                "emotions": gated,
                "intensity": {
                    "score": round(min(max(sent_intensity, 0.0), 1.0), rd),
                    "label": self._label_intensity(sent_intensity)
                }
            })
        return out

    def _apply_polarity_gate(self, target: List[Dict[str, Any]], dominant_polarity: str,
                             keep_max_opposite: Optional[int] = None, downscale: Optional[float] = None):
        if not isinstance(target, list) or not target or not isinstance(target[0], dict):
            return target
        first = target[0]
        if "sentence" in first and "emotions" in first:
            return self._apply_polarity_gate_sequence(
                emotion_sequence=target,
                dominant_polarity=dominant_polarity,
                keep_max_opposite=keep_max_opposite,
                downscale=downscale,
            )
        if "emotion_id" in first:
            return self._apply_polarity_gate_emotions(
                cands=target,
                dominant_polarity=dominant_polarity,
                keep_max_opposite=keep_max_opposite,
                downscale=downscale,
            )
        return target

    # ⑦ 문서단위 지배 극성 계산
    def _compute_document_dominant_polarity(self, processed: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        pos = neg = 0.0
        for it in processed or []:
            for e in it.get("emotions", []):
                pol = self._get_emotion_polarity(e.get("emotion_id"))
                if pol == "pos":
                    pos += self._safe_float(e.get("confidence"), 0.0)
                elif pol == "neg":
                    neg += self._safe_float(e.get("confidence"), 0.0)
        total = pos + neg
        if total <= 0:
            return None
        return {"polarity": "pos", "ratio": pos / total} if pos >= neg else {"polarity": "neg", "ratio": neg / total}

    # ---------------------------------------------------------------------
    # 선택·라벨·정규화 유틸
    # ---------------------------------------------------------------------
    def _label_intensity(self, score: float) -> str:
        return self._intensity_label(score)

    def _determine_polarity(self, emotion_id: Optional[str]) -> str:
        if not emotion_id or not isinstance(emotion_id, str):
            return "neu"
        head = emotion_id.split("-")[0].strip()
        if head in ("희", "락"):
            return "pos"
        if head in ("노", "애"):
            return "neg"
        return "neu"

    def _select_topk_candidates(
        self,
        candidates: List[Dict[str, Any]],
        top_k: int,
        min_score: float,
        sort_key: str = "confidence"
    ) -> List[Dict[str, Any]]:
        filtered = []
        for c in candidates or []:
            conf = self._safe_float(c.get(sort_key, c.get("score", 0.0)), 0.0)
            if conf >= float(min_score):
                c = dict(c)
                c["confidence"] = self._safe_float(c.get("confidence", conf), conf)
                filtered.append(c)
        if not filtered and candidates:
            best = max(candidates, key=lambda x: self._safe_float(x.get(sort_key, x.get("score", 0.0)), 0.0))
            best = dict(best)
            best["confidence"] = self._safe_float(best.get("confidence", best.get("score", 0.0)), 0.0)
            filtered = [best]
        filtered.sort(key=lambda x: self._safe_float(x.get("confidence", x.get("score", 0.0)), 0.0), reverse=True)
        if top_k > 0:
            filtered = filtered[:int(top_k)]
        return filtered

    def _apply_polarity_gate_candidates(
        self,
        candidates: List[Dict[str, Any]],
        dominant_polarity: str,
        keep_max_opposite: int,
        downscale: float
    ) -> List[Dict[str, Any]]:
        if dominant_polarity not in ("pos", "neg"):
            return candidates
        pos_list, neg_list, neu_list = [], [], []
        for c in candidates or []:
            pol = self._determine_polarity(c.get("emotion_id"))
            if pol == "pos":
                pos_list.append(c)
            elif pol == "neg":
                neg_list.append(c)
            else:
                neu_list.append(c)

        if dominant_polarity == "pos":
            neg_list.sort(key=lambda x: self._safe_float(x.get("confidence", x.get("score", 0.0))), reverse=True)
            keep = neg_list[:max(0, int(keep_max_opposite))]
            for c in keep:
                c["confidence"] = max(0.0, self._safe_float(c.get("confidence", c.get("score", 0.0))) * float(downscale))
            gated = pos_list + keep + neu_list
        else:
            pos_list.sort(key=lambda x: self._safe_float(x.get("confidence", x.get("score", 0.0))), reverse=True)
            keep = pos_list[:max(0, int(keep_max_opposite))]
            for c in keep:
                c["confidence"] = max(0.0, self._safe_float(c.get("confidence", c.get("score", 0.0))) * float(downscale))
            gated = neg_list + keep + neu_list

        gated.sort(key=lambda x: self._safe_float(x.get("confidence", x.get("score", 0.0))), reverse=True)
        return gated

    # ---------------------------------------------------------------------
    # 폴백 전이 패턴
    # ---------------------------------------------------------------------
    def _fallback_progressive_patterns(self, emotion_sequence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        patterns = []
        if not emotion_sequence or len(emotion_sequence) < 2:
            return patterns

        def _get_id(seq_item: Dict[str, Any]) -> Optional[str]:
            ems = seq_item.get("emotions")
            if isinstance(ems, list) and len(ems) > 0:
                first = ems[0]
                if isinstance(first, dict):
                    for k in ("emotion_id", "id", "emotion", "label"):
                        v = first.get(k)
                        if isinstance(v, str) and v:
                            return v
                elif isinstance(first, str) and first:
                    return first
            for k in ("emotion_id", "id", "emotion", "label"):
                v = seq_item.get(k)
                if isinstance(v, str) and v:
                    return v
            return None

        prev_id = _get_id(emotion_sequence[0])
        prev_int = self._safe_float(emotion_sequence[0].get("intensity", {}).get("score"), 0.0)
        for item in emotion_sequence[1:]:
            cur_id = _get_id(item)
            cur_int = self._safe_float(item.get("intensity", {}).get("score"), 0.0)
            if prev_id and cur_id:
                patterns.append({
                    "from": prev_id,
                    "to": cur_id,
                    "avg_intensity": round((prev_int + cur_int) / 2.0, 3),
                    "count": 1
                })
            prev_id, prev_int = cur_id, cur_int
        return patterns

    # ---------------------------------------------------------------------
    # 외부 의존 주입
    # ---------------------------------------------------------------------
    def set_dependencies(self, pattern_extractor=None, intensity_analyzer=None, transition_analyzer=None):
        self.pattern_extractor = pattern_extractor
        self.intensity_analyzer = intensity_analyzer
        self.transition_analyzer = transition_analyzer
        self.is_ready = bool(self.transition_analyzer)
        logger.info("ContextExtractor 의존성 주입 완료 (transition_analyzer=%s)",
                    type(self.transition_analyzer).__name__ if self.transition_analyzer else None)

    # ---------------------------------------------------------------------
    # 메인 엔트리
    # ---------------------------------------------------------------------
    def extract_contextual_emotions(
            self,
            text: str,
            emotions_data: Optional[Dict[str, Any]] = None,
            max_depth: int = 3
    ) -> Dict[str, Any]:
        """재귀적 감정 분석을 통해 문맥 기반 감정을 추출 (후처리 + 압축 + 페이로드 정리 포함)"""
        try:
            start_time = time.time()
            if emotions_data is None:
                emotions_data = self.emotions_data

            if hasattr(self, "memory_monitor") and not self.memory_monitor.check_memory():
                self._cleanup_resources()

            if not self._validate_input(text):
                return self._create_error_response("Invalid input")

            sentences: List[str] = self._preprocess_text(text)

            # 캐시 활용
            context_info: Dict[str, Any] = self._analyze_context(tuple(sentences))

            # 재귀 분석
            emotion_analysis_raw: Dict[str, Any] = self._recursive_emotion_analysis(
                sentences=sentences,
                emotions_data=emotions_data,
                context_info=context_info,
                current_depth=0,
                max_depth=max_depth
            )

            raw_sequence: List[Dict[str, Any]] = list(emotion_analysis_raw.get("emotion_sequence", []))

            keep_full = bool(self.config.get("sequence_keep_candidates_full", False))
            candidates_full = self._cap_candidates(
                raw_sequence, cap=int(self.config.get("sequence_candidates_full_cap", 100))
            ) if keep_full else None

            emotion_sequence: List[Dict[str, Any]] = self._postprocess_emotion_sequence(
                raw_sequence=raw_sequence,
                context_info=context_info
            )

            confidence_scores: Dict[str, float] = self._calculate_weighted_confidence(
                emotion_sequence=emotion_sequence,
                context_info=context_info
            )

            primary_emotion: Optional[str] = self._determine_primary_emotion(
                emotion_sequence=emotion_sequence,
                context_info=context_info
            )

            primary_category: Optional[str] = None
            sub_category: Optional[str] = None
            if primary_emotion:
                if "-" in primary_emotion:
                    primary_category, sub_category = primary_emotion.split("-", 1)
                else:
                    primary_category = primary_emotion

            # Context Manager에 학습 피드
            if hasattr(self, "context_manager"):
                for seq in emotion_sequence:
                    if seq.get("emotions"):
                        emotion = seq["emotions"][0].get("emotion_id")
                        intensity_value = self._safe_float(seq.get("intensity", {}).get("score"), 0.0)
                        situation = context_info.get("situation_type", "default")
                        context_factors = {
                            "time_indicators": self._safe_len(context_info.get("time_indicators")),
                            "location_indicators": self._safe_len(context_info.get("location_indicators")),
                            "social_context": self._safe_len(context_info.get("social_context")),
                            "emotional_triggers": self._safe_len(context_info.get("emotional_triggers")),
                        }
                        if emotion:
                            self.context_manager.update_emotion_context(
                                emotion=emotion,
                                intensity=float(intensity_value),
                                situation=situation,
                                weight=1.0,
                                context_factors=context_factors,
                            )

            progressive_analysis = self.analyze_progressive_context(
                text=text,
                emotion_sequence=emotion_sequence,
                context_info=context_info
            )

            situation_impact = self.analyze_situation_impact(
                text=text,
                emotion_sequence=emotion_sequence,
                context_info=context_info
            )

            # compact
            emotion_analysis_compact = self._compact_emotion_analysis(
                analysis=emotion_analysis_raw,
                top_k=int(self.config.get("analysis_top_k", 5)),
                per_sentence_cap=int(self.config.get("cap_lengths", {}).get("emotion_sequence_per_sentence", 5)),
                emit_all_taxonomy=bool(self.config.get("emit_all_taxonomy", False)),
                aggregate_subs=bool(self.config.get("sub_emotions_aggregate", True))
            )

            # 메트릭 업데이트
            self._update_metrics(time.time() - start_time)

            payload: Dict[str, Any] = {
                "primary_emotion": primary_emotion,
                "primary_category": primary_category,
                "sub_category": sub_category,
                "context_info": context_info,
                "emotion_sequence": emotion_sequence,
                "confidence_scores": confidence_scores,
                "emotion_analysis": emotion_analysis_compact,
                "progressive_analysis": progressive_analysis,
                "situation_impact_analysis": situation_impact,
                "metrics": self._get_metrics(),
            }
            if keep_full and candidates_full is not None:
                payload.setdefault("debug", {})["candidates_full"] = candidates_full

            payload = self._finalize_payload_cleanup(payload)

            # B3) Fast/Heavy 분기 스위치 - context_manager 부가 결과는 옵션에 따라 붙임
            if hasattr(self, "context_manager"):
                try:
                    if bool(self.config.get("attach_flows", True)):
                        payload["emotion_flows"] = self.context_manager.track_all_emotion_flows()
                    if bool(self.config.get("attach_patterns", True)):
                        payload["emotion_patterns"] = self.context_manager.analyze_emotion_patterns()
                    payload["context_summary"] = self.context_manager.get_context_summary()
                except Exception:
                    payload.setdefault("debug", {})[
                        "context_manager_warning"] = "context_manager extensions unavailable"

            # ✅ 1) 스키마/값 일관화(라운딩, 백워드 호환 키 유지)
            payload = postprocess_context_flow_output(
                payload, text, low_conf_hide_thresh=0.15, ndigits=int(self.config.get("round_digits", 3)),
                backcompat=True
            )

            # ✅ 2) 저신뢰 상황 후보 필터/리웨이트 + topk 정리
            payload = _filter_low_evidence_situations(
                payload,
                min_cov=float(self.config.get("min_evidence_coverage", 0.08)),
                reweight=bool(self.config.get("reweight_by_evidence", True)),
                topk=int(self.config.get("impact_topk", 8))
            )

            # ✅ 3) 중복 섹션 정리(선택)
            payload = _drop_redundant_sequence(payload)

            return payload

        except Exception as e:
            logger.critical(f"Context extraction failed: {str(e)}", exc_info=True)
            return self._create_error_response("Internal error")

    def _cap_candidates(self, arr: List[Any], cap: int = 100) -> List[Any]:
        if not isinstance(arr, list) or cap <= 0:
            return arr
        return arr[:cap]

    def _ensure_post_cfg_defaults(self) -> None:
        cfg = self.config = dict(getattr(self, "config", {}) or {})
        cfg.setdefault("sequence_top_k", 5)
        cfg.setdefault("sequence_min_score", 0.21)
        cfg.setdefault("intensity_bins", [0.33, 0.66])
        cfg.setdefault("merge_strategy", "max")
        cfg.setdefault("normalize_enabled", True)
        cfg.setdefault("polarity_gate_enabled", True)
        cfg.setdefault("polarity_dominance_threshold", 0.65)
        cfg.setdefault("polarity_opposite_keep_max", 1)
        cfg.setdefault("polarity_opposite_downscale", 0.6)
        cfg.setdefault("analysis_top_k", 5)
        cfg.setdefault("emit_all_taxonomy", False)
        cfg.setdefault("sub_emotions_aggregate", True)
        cfg.setdefault("round_digits", 3)
        cfg.setdefault("strip_empty", True)
        cfg.setdefault("cap_lengths", {
            "situation_analysis_per_emotion": 5,
            "emotion_sequence_per_sentence": 5
        })
        cfg.setdefault("schema_version", "1.1-post")
        cfg.setdefault("compact_emotion_analysis", True)
        cfg.setdefault("allow_fallback_indicators", False)
        cfg.setdefault("candidate_topk", 50)
        cfg.setdefault("attach_patterns", True)
        cfg.setdefault("attach_flows", True)
        cfg.setdefault("normalize_hardcode_aliases", False)

    def _normalize_emotion_id(self, emotion_id: str) -> str:
        if not emotion_id or "-" not in emotion_id:
            return emotion_id
        cat, label = emotion_id.split("-", 1)
        label = label.strip()
        
        # H5) 정규화/동의어 하드코딩 축소 - phrase 인덱스 우선 사용
        if hasattr(self, '_phrase2emo') and self._phrase2emo:
            # phrase 인덱스에서 검색
            for phrase, candidates in self._phrase2emo.items():
                if phrase == label:
                    for emo_id, _ in candidates:
                        if emo_id.startswith(f"{cat}-"):
                            return emo_id
        
        # 하드코딩 표는 옵션으로만 사용
        if self.config.get("normalize_hardcode_aliases", False):
            repl = {
                "행복감": "행복",
                "희열감": "희열",
                "자긍심": "자긍심",
                "감탄스러움": "감탄",
                "쾌활함": "쾌활함",
                "기분좋음": "기분좋음",
            }
            if label in repl:
                norm = repl[label]
            elif label.endswith("감") and len(label) >= 3:
                norm = label[:-1]
            elif label.endswith("스러움") and len(label) >= 5:
                norm = label[:-3]
            else:
                norm = label
            return f"{cat}-{norm}"
        
        # 기본: 원본 반환
        return emotion_id

    def _intensity_label(self, score: float) -> str:
        b = self.config.get("intensity_bins", [0.33, 0.66])
        score = max(0.0, min(1.0, float(score)))
        if score < b[0]:
            return "low"
        if score < b[1]:
            return "medium"
        return "high"

    def _compute_polarity_of_id(self, emotion_id: str) -> str:
        if not emotion_id or "-" not in emotion_id:
            return "unk"
        root = emotion_id.split("-", 1)[0]
        if root in ("희", "락"):
            return "pos"
        if root in ("노", "애"):
            return "neg"
        return "unk"

    def _compute_dominant_polarity(self, emotion_sequence):
        return self._compute_document_dominant_polarity(emotion_sequence)

    # ---------------------------------------------------------------------
    # compact 변환
    # ---------------------------------------------------------------------
    def _compact_emotion_analysis(
        self,
        analysis: Dict[str, Any],
        top_k: int = 5,
        per_sentence_cap: int = 5,
        emit_all_taxonomy: bool = False,
        aggregate_subs: bool = True
    ) -> Dict[str, Any]:
        if not analysis:
            return {}
        seq = list(analysis.get("emotion_sequence", []))
        if not seq:
            return analysis

        by_sent: Dict[str, List[Dict[str, Any]]] = {}
        for row in seq:
            sent = row.get("sentence", "")
            for emo in row.get("emotions", []):
                by_sent.setdefault(sent, []).append(emo)

        compact_seq: List[Dict[str, Any]] = []
        appeared_emotions = set()
        for sent, emos in by_sent.items():
            emos = sorted(emos, key=lambda x: self._safe_float(x.get("confidence"), 0.0), reverse=True)
            emos = emos[:max(1, top_k)]
            intensity_score = max([self._safe_float(e.get("confidence"), 0.0) for e in emos], default=0.0)
            label = "high" if intensity_score >= 0.66 else ("medium" if intensity_score >= 0.33 else "low")
            for e in emos:
                eid = e.get("emotion_id")
                if eid:
                    appeared_emotions.add(eid)
            compact_seq.append({
                "sentence": sent,
                "emotions": emos[:per_sentence_cap],
                "intensity": {"score": intensity_score, "label": label}
            })

        if aggregate_subs:
            subs = {eid: [] for eid in sorted(appeared_emotions)}
        else:
            subs = dict(analysis.get("sub_emotions", {})) if emit_all_taxonomy else {eid: [] for eid in sorted(appeared_emotions)}

        if emit_all_taxonomy:
            hierarchy = dict(analysis.get("emotion_hierarchy", {}))
        else:
            roots = sorted({eid.split("-", 1)[0] for eid in appeared_emotions if "-" in eid} |
                           {eid for eid in appeared_emotions if "-" not in eid})
            hierarchy = {r: {} for r in roots}

        out = {
            "emotion_sequence": self._round_and_strip(compact_seq),
            "sub_emotions": subs,
            "emotion_hierarchy": hierarchy
        }
        return out

    def _round_and_strip(self, obj: Any) -> Any:
        digits = int((self.config or {}).get("round_digits", 3))
        strip = bool((self.config or {}).get("strip_empty", True))

        def round_num(x):
            try:
                return round(float(x), digits)
            except Exception:
                return x

        if isinstance(obj, dict):
            out = {k: self._round_and_strip(v) for k, v in obj.items()}
            if strip:
                out = {k: v for k, v in out.items() if v not in (None, [], {})}
            return out
        elif isinstance(obj, list):
            out = [self._round_and_strip(v) for v in obj]
            if strip:
                out = [v for v in out if v not in (None, [], {})]
            return out
        elif isinstance(obj, (int, float)):
            return round_num(obj)
        else:
            return obj

    # ---------------------------------------------------------------------
    # 페이로드 마무리(반올림/캡/압축/정리 + 직렬화 안전화)
    # ---------------------------------------------------------------------
    def _finalize_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            return payload

        cfg = getattr(self, "config", {}) or {}
        rd = int(cfg.get("round_digits", 3))
        caps = dict(cfg.get("cap_lengths", {}) or {})
        cap_sa_per_emotion = int(caps.get("situation_analysis_per_emotion", 5))
        cap_seq_per_sentence = int(caps.get("emotion_sequence_per_sentence", 5))

        # 1) 반올림
        self._round_floats_inplace(payload, rd)

        # 2) 길이 상한
        self._cap_situation_analysis_inplace(payload, cap_sa_per_emotion)
        self._cap_emotions_per_sentence_inplace(payload, cap_seq_per_sentence)

        # 3) (옵션) emotion_analysis 압축
        if bool(cfg.get("compact_emotion_analysis", True)):
            analysis_top_k = int(cfg.get("analysis_top_k", 5))
            self._compact_emotion_analysis_inplace(payload, analysis_top_k, rd)

        # 4) (옵션) 빈 값 제거
        if bool(cfg.get("strip_empty", True)):
            payload = self._strip_empty(payload)

        # 5) 직렬화 안전화(set/datetime/defaultdict 등)
        self._json_safe_inplace(payload)

        # 6) 스키마 버전
        payload["schema_version"] = cfg.get("schema_version", "1.2")

        return payload

    def _finalize_payload_cleanup(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._finalize_payload(payload)

    # ---- helpers ---------------------------------------------------------
    def _round_floats_inplace(self, obj: Any, ndigits: int = 3) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, float):
                    obj[k] = round(v, ndigits)
                else:
                    self._round_floats_inplace(v, ndigits)
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                if isinstance(v, float):
                    obj[i] = round(v, ndigits)
                else:
                    self._round_floats_inplace(v, ndigits)

    def _cap_situation_analysis_inplace(self, payload: Dict[str, Any], per_emotion_cap: int) -> None:
        sia = payload.get("situation_impact_analysis")
        if not isinstance(sia, dict):
            return
        sa = sia.get("situation_analysis")
        if not isinstance(sa, list) or per_emotion_cap <= 0:
            return

        from collections import defaultdict
        buckets = defaultdict(list)
        for row in sa:
            if not isinstance(row, dict):
                continue
            eid = row.get("emotion_id", "__NO_EMOTION_ID__")
            buckets[eid].append(row)

        trimmed = []
        for _, rows in buckets.items():
            rows.sort(key=lambda x: self._safe_float(x.get("impact_score"), 0.0), reverse=True)
            trimmed.extend(rows[:per_emotion_cap])

        trimmed.sort(key=lambda x: self._safe_float(x.get("impact_score"), 0.0), reverse=True)
        sia["situation_analysis"] = trimmed

    def _cap_emotions_per_sentence_inplace(self, payload: Dict[str, Any], per_sentence_cap: int) -> None:
        seq = payload.get("emotion_sequence")
        if not isinstance(seq, list) or per_sentence_cap <= 0:
            return
        for row in seq:
            if not isinstance(row, dict):
                continue
            emos = row.get("emotions")
            if isinstance(emos, list):
                emos.sort(key=lambda x: self._safe_float(x.get("confidence"), 0.0), reverse=True)
                row["emotions"] = emos[:per_sentence_cap]

    def _compact_emotion_analysis_inplace(self, payload: Dict[str, Any], top_k: int, rd: int) -> None:
        ea = payload.get("emotion_analysis")
        if not isinstance(ea, dict):
            return
        seq = ea.get("emotion_sequence")
        if not isinstance(seq, list) or top_k <= 0:
            return

        from collections import defaultdict
        sent_bucket = defaultdict(list)
        for row in seq:
            if not isinstance(row, dict):
                continue
            sent = row.get("sentence", "")
            emos = row.get("emotions") or []
            for e in emos:
                if not isinstance(e, dict):
                    continue
                eid = e.get("emotion_id")
                if not eid:
                    continue
                conf = self._safe_float(e.get("confidence"), 0.0)
                sent_bucket[sent].append({
                    "emotion_id": eid,
                    "confidence": conf,
                    "emotion_complexity": e.get("emotion_complexity"),
                    "intensity_label": e.get("intensity_label")
                })

        compact_seq = []
        for sent, emos in sent_bucket.items():
            if not emos:
                continue
            best = {}
            for e in emos:
                eid = e["emotion_id"]
                if eid not in best or e["confidence"] > best[eid]["confidence"]:
                    best[eid] = e
            merged = list(best.values())
            merged.sort(key=lambda x: self._safe_float(x.get("confidence"), 0.0), reverse=True)
            merged = merged[:top_k]
            score = sum(self._safe_float(x.get("confidence"), 0.0) for x in merged) / max(1, len(merged))
            score = round(min(max(score, 0.0), 1.0), rd)
            compact_seq.append({
                "sentence": sent,
                "emotions": merged,
                "intensity": {"score": score, "label": self._label_intensity(score)}
            })

        if compact_seq:
            order = {s: i for i, s in enumerate(sent_bucket.keys())}
            compact_seq.sort(key=lambda r: order.get(r.get("sentence", ""), 10 ** 9))
            ea["emotion_sequence"] = compact_seq

    def _strip_empty(self, x):
        if isinstance(x, dict):
            out = {}
            for k, v in x.items():
                vv = self._strip_empty(v)
                if vv in (None, "", [], {}):
                    continue
                out[k] = vv
            return out
        if isinstance(x, list):
            out = [self._strip_empty(v) for v in x]
            out = [v for v in out if v not in (None, "", [], {})]
            return out
        return x

    def _strip_empty_inplace(self, obj):
        PRESERVE_KEYS = {"primary_emotion", "primary_category", "sub_category", "context_info"}
        if isinstance(obj, dict):
            for k in list(obj.keys()):
                if k in PRESERVE_KEYS:
                    self._strip_empty_inplace(obj[k])
                    continue
                v = obj[k]
                if isinstance(v, (dict, list)):
                    self._strip_empty_inplace(v)
                    if v == {} or v == []:
                        obj.pop(k, None)
                elif v is None:
                    obj.pop(k, None)
        elif isinstance(obj, list):
            to_del = []
            for i, v in enumerate(obj):
                if isinstance(v, (dict, list)):
                    self._strip_empty_inplace(v)
                    if v == {} or v == []:
                        to_del.append(i)
                elif v is None:
                    to_del.append(i)
            for i in reversed(to_del):
                del obj[i]

    # ---------------------------------------------------------------------
    # 진행형 분석 & 상황 영향
    # ---------------------------------------------------------------------
    def analyze_progressive_context(
        self,
        text: str,
        emotion_sequence: List[Dict[str, Any]],
        context_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        진행형 감정 분석 + 상황 영향도를 종합.
        - transition_analyzer 없을 때도 폴백 분석을 반드시 수행.
        """
        result: Dict[str, Any] = {
            "progressive_patterns": [],
            "transition_details": {},
            "integrated_situation_impact": {}
        }
        try:
            if context_info is None:
                sentences = tuple(self._preprocess_text(text))
                context_info = self._analyze_context(sentences)

            self._get_emotion_cache()

            try:
                if self.transition_analyzer is not None:
                    transitions = self.transition_analyzer.get_progressive_patterns(emotion_sequence)
                else:
                    transitions = self._fallback_progressive_patterns(emotion_sequence)
            except Exception as e:
                logger.warning(f"전이 분석 실패, 폴백 사용: {e}")
                transitions = self._fallback_progressive_patterns(emotion_sequence)

            transitions_filtered = [
                t for t in transitions
                if self._validate_transition_in_labeling_data(t.get("from", ""), t.get("to", ""))
            ]
            result["progressive_patterns"] = transitions_filtered
            result["transition_details"] = self._analyze_transition_details(transitions_filtered)

            situation_impact_result = self.analyze_situation_impact(text, emotion_sequence, context_info)
            integrated_map: Dict[str, Any] = {}
            for tr in transitions_filtered:
                from_e, to_e = tr["from"], tr["to"]
                max_impact, top_situation = 0.0, None
                for item in situation_impact_result.get("situation_analysis", []):
                    impact = self._safe_float(item.get("impact_score"), 0.0)
                    if impact > max_impact:
                        max_impact = impact
                        top_situation = item.get("situation")
                integrated_map[f"{from_e}->{to_e}"] = {
                    "impact_score": round(max_impact, 3),
                    "top_situation": top_situation
                }
            result["integrated_situation_impact"] = integrated_map
            return result

        except Exception as e:
            logger.error(f"진행형 감정 분석(통합) 중 오류 발생: {str(e)}")
            return result

    def _validate_transition_in_labeling_data(self, from_emotion: str, to_emotion: str) -> bool:
        try:
            from_cat, from_sub = self._split_emotion_id(from_emotion)
            to_cat, to_sub = self._split_emotion_id(to_emotion)
            from_data = self.emotions_data.get(from_cat, {})
            if not from_data:
                return False
            transitions = {}
            if from_sub:
                sub_data = from_data.get('sub_emotions', {}).get(from_sub, {})
                if not sub_data:
                    sub_data = from_data.get('emotion_profile', {}).get('sub_emotions', {}).get(from_sub, {})
                transitions = sub_data.get('emotion_transitions', {}) or {}
            else:
                transitions = from_data.get('emotion_transitions', {}) or {}
            patterns = transitions.get('patterns', [])
            for p in patterns:
                if p.get('from_emotion') in (from_cat, from_emotion):
                    if p.get('to_emotion') in (to_cat, to_emotion):
                        return True
            return False
        except Exception:
            return False

    def _split_emotion_id(self, emotion_id: str) -> Tuple[str, Optional[str]]:
        if '-' in emotion_id:
            cat, sub = emotion_id.split('-', 1)
            return cat, sub
        else:
            return emotion_id, None

    def _analyze_transition_details(self, transitions: List[Dict[str, Any]]) -> Dict[str, Any]:
        details = defaultdict(int)
        for t in transitions:
            trans_key = f"{t['from']}->{t['to']}"
            details[trans_key] += 1
        sorted_details = sorted(details.items(), key=lambda x: x[1], reverse=True)
        return {"transition_counts": sorted_details}

    def analyze_situation_impact(self, text: str, emotion_sequence: List[Dict[str, Any]], context_info: Dict[str, Any]) -> Dict[str, Any]:
        result = {
            "situation_analysis": [],
            "impact_scores": {},
            "transition_based_adjustment": {}
        }
        try:
            emotion_cache = self._get_emotion_cache()
            transitions_map = emotion_cache.get("transitions_map", {})
            for seq in emotion_sequence:
                e_id = (seq.get('emotions') or [{}])[0].get('emotion_id')
                if not e_id:
                    continue
                matched_situations = self._match_situations_in_labeling_data(e_id)
                for st in matched_situations:
                    base_intensity = self._safe_float(seq.get('intensity', {}).get('score'), 0.0)
                    impact_score = self._calculate_situation_score(e_id, st, base_intensity)
                    from_emotion = seq.get('prev_emotion')
                    if from_emotion and from_emotion in transitions_map:
                        for t_info in transitions_map[from_emotion]:
                            if t_info["to"] == e_id:
                                impact_score += 0.1 * len(t_info["triggers"])
                                key_name = f"{from_emotion}->{e_id}"
                                result["transition_based_adjustment"].setdefault(key_name, 0.0)
                                result["transition_based_adjustment"][key_name] += 0.1 * len(t_info["triggers"])
                    result["situation_analysis"].append({
                        "emotion_id": e_id,
                        "situation": st.get("name"),
                        "impact_score": round(impact_score, 3)
                    })
            situation_group = defaultdict(list)
            for item in result["situation_analysis"]:
                situation_group[item["situation"]].append(self._safe_float(item["impact_score"], 0.0))
            for st_name, scores in situation_group.items():
                if scores:
                    result["impact_scores"][st_name] = sum(scores) / len(scores)
            return result
        except Exception as e:
            logger.error(f"상황 영향도 분석(전이 반영) 중 오류 발생: {str(e)}")
            return result

    def _match_situations_in_labeling_data(self, emotion_id: str) -> List[Dict[str, Any]]:
        matched_situations = []
        cat, sub = self._split_emotion_id(emotion_id)
        if cat not in self.emotions_data:
            return matched_situations
        cat_data = self.emotions_data[cat]
        if sub:
            sub_data = cat_data.get("sub_emotions", {}).get(sub, {})
            if not sub_data:
                sub_data = cat_data.get("emotion_profile", {}).get("sub_emotions", {}).get(sub, {})
            situations = sub_data.get('context_patterns', {}).get('situations', {})
        else:
            situations = cat_data.get('context_patterns', {}).get('situations', {})
        for name, sdata in (situations or {}).items():
            matched_situations.append({
                "name": name,
                "keywords": sdata.get("keywords", []),
                "intensity": sdata.get("intensity", "medium")
            })
        return matched_situations

    def _calculate_situation_score(self, emotion_id: str, situation_info: Dict[str, Any], intensity_score: float) -> float:
        base = self._intensity_prior_weights.get(situation_info.get("intensity", "medium"), 1.0)

        calibrator = getattr(self, "calibrator", None)
        if calibrator:
            cat, sub = self._split_emotion_id(emotion_id)
            emo_key = f"{cat}-{sub}" if sub else cat
            log_adj = 0.0
            if hasattr(calibrator, "get_pattern_weight"):
                for kw in situation_info.get("keywords", []):
                    if isinstance(kw, str) and kw:
                        log_adj += calibrator.get_pattern_weight(emo_key, kw)
            if hasattr(calibrator, "get_prior_adj"):
                log_adj += calibrator.get_prior_adj(emo_key)
            if log_adj:
                base *= math.exp(log_adj)

        s = max(0.0, min(1.0, float(intensity_score)))
        return round(base * s, 3)

    def _traverse_labeling_data(self, data: Dict[str, Any], path: Optional[str] = None) -> None:
        if not path:
            path = ""
        if not isinstance(data, dict):
            return
        for key, value in data.items():
            current_path = f"{path}.{key}" if path else key
            if key == "sub_emotions" and isinstance(value, dict):
                for sub_key, sub_val in value.items():
                    sub_path = f"{current_path}.{sub_key}"
                    self._traverse_labeling_data(sub_val, sub_path)
            elif key == "context_patterns" and isinstance(value, dict):
                if "situations" in value:
                    pass
            else:
                if isinstance(value, dict):
                    self._traverse_labeling_data(value, current_path)
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            self._traverse_labeling_data(item, current_path)

    def calculate_context_metrics(self, context_data: Dict[str, Any]) -> Dict[str, float]:
        metrics = {}
        if context_data:
            try:
                metrics['context_score'] = sum(
                    self.config['weights'][key] * float(context_data.get(key, 0.0))
                    for key in self.config['weights']
                ) / max(1, len(self.config['weights']))
                metrics['confidence_score'] = min(
                    metrics['context_score'],
                    float(self.config['thresholds']['confidence'])
                )
                metrics['relevance_score'] = metrics['context_score'] * metrics['confidence_score']
            except Exception:
                metrics = {'context_score': 0.0, 'confidence_score': 0.0, 'relevance_score': 0.0}
        else:
            metrics = {'context_score': 0.0, 'confidence_score': 0.0, 'relevance_score': 0.0}
        return metrics

    # ---------------------------------------------------------------------
    # 문맥 캐시 세팅/사용/클린업(정비)
    # ---------------------------------------------------------------------
    def _setup_cache(self):
        """캐시 설정 초기화 (실제로 사용하는 래퍼로 세팅)"""
        size = int(self.config.get('cache_size', 4096))

        # 내부 분석 함수를 lru_cache로 감쌈(바운드 메서드 안전)
        self._context_cache = lru_cache(maxsize=size)(self._analyze_context_internal)
        logger.info(f"Context cache initialized with size: {size}")

    def _estimate_intensity_priors(self, emotion_cache: Dict[str, Any]) -> Dict[str, float]:
        cnt = Counter()
        for situations in (emotion_cache.get("situations") or {}).values():
            for situ in situations:
                cnt[situ.get("intensity", "medium")] += 1
        if not cnt:
            return self._intensity_prior_weights
        total = sum(cnt.values())
        raw = {k: (total / max(1, v)) for k, v in cnt.items()}
        mean = sum(raw.values()) / max(1, len(raw))
        return {k: raw.get(k, 1.0) / max(1e-9, mean) for k in ("low", "medium", "high")}

    def _compile_phrase_indexes(self, ec: Dict[str, Any]) -> None:
        """문장 파트 시각 스케어를 위한 정규식 커플리어"""
        def _compile_alt(phrases: Optional[Iterable[str]]) -> Optional[Pattern[str]]:
            items = [p for p in (phrases or []) if isinstance(p, str) and p]
            if not items:
                return None
            ordered = sorted(items, key=len, reverse=True)
            pattern = "|".join(re.escape(p) for p in ordered)
            return re.compile(pattern) if pattern else None

        # 정규식 인덱스 폭주 대비 가드 - 분할 컴파일 지원
        def _compile_alt_large(phrases: Optional[Iterable[str]], max_patterns: int = 5000) -> Optional[Pattern[str]]:
            items = [p for p in (phrases or []) if isinstance(p, str) and p]
            if not items:
                return None
            
            # 대규모 대응: Top-N 우선순위화 및 길이별 분할
            if len(items) > max_patterns:
                # 1) 빈도 기반 상위 N개만 선택 (phrase2emo에서 가중치 합계 기준)
                weighted_items = []
                for item in items:
                    weight_sum = sum(w for _, w in self._phrase2emo.get(item, []))
                    weighted_items.append((item, weight_sum))
                
                # 가중치 순으로 정렬하여 상위 max_patterns개만 선택
                weighted_items.sort(key=lambda x: x[1], reverse=True)
                items = [item for item, _ in weighted_items[:max_patterns]]
            
            # 2) 길이별 분할 (짧은 것 우선 - 더 정확한 매칭)
            short_items = [p for p in items if len(p) <= 10]
            long_items = [p for p in items if len(p) > 10]
            
            patterns = []
            if short_items:
                ordered_short = sorted(short_items, key=len, reverse=True)
                pattern_short = "|".join(re.escape(p) for p in ordered_short)
                if pattern_short:
                    patterns.append(pattern_short)
            
            if long_items:
                ordered_long = sorted(long_items, key=len, reverse=True)
                pattern_long = "|".join(re.escape(p) for p in ordered_long)
                if pattern_long:
                    patterns.append(pattern_long)
            
            if patterns:
                combined_pattern = "|".join(f"({p})" for p in patterns)
                return re.compile(combined_pattern)
            return None

        inds = ec.get("indicators", {}) or {}
        self._rx_time = _compile_alt(inds.get("time"))
        self._rx_location = _compile_alt(inds.get("location"))
        self._rx_social = _compile_alt(inds.get("social"))
        self._rx_triggers = _compile_alt(inds.get("triggers"))

        phrase_map: DefaultDict[str, List[Tuple[str, float]]] = defaultdict(list)
        for emo_id, situations in (ec.get("situations") or {}).items():
            for situ in situations:
                for kw in (situ.get("keywords") or []):
                    if isinstance(kw, str) and kw:
                        phrase_map[kw].append((emo_id, 1.0))
                for ex in (situ.get("examples") or []):
                    if isinstance(ex, str) and ex:
                        phrase_map[ex].append((emo_id, 0.5))
                for vr in (situ.get("variations") or []):
                    if isinstance(vr, str) and vr:
                        phrase_map[vr].append((emo_id, 0.5))
        self._phrase2emo = phrase_map
        
        # 정규식 인덱스 폭주 대비 가드 - 대규모 라벨 대응
        phrases = list(phrase_map.keys())
        if len(phrases) > 1000:  # 1000개 이상이면 최적화 적용
            self._rx_phrase = _compile_alt_large(phrases)
        else:
            self._rx_phrase = _compile_alt(phrases)

    def _get_emotion_cache(self) -> Dict[str, Any]:
        if self.emotion_cache:
            if self._rx_phrase is None and self.emotion_cache:
                self._compile_phrase_indexes(self.emotion_cache)
            return self.emotion_cache
        cache = self._build_emotion_cache() or {}
        self._compile_phrase_indexes(cache)
        weights = self._estimate_intensity_priors(cache)
        if weights:
            self._intensity_prior_weights = weights
            try:
                self.context_manager.inject_intensity_priors(weights)
            except Exception:
                pass
        return cache

    def get_shared_index(self) -> Dict[str, Any]:
        return {
            "rx_phrase": self._rx_phrase,
            "phrase2emo": dict(self._phrase2emo),
            "rx_time": self._rx_time,
            "rx_location": self._rx_location,
            "rx_social": self._rx_social,
            "rx_triggers": self._rx_triggers,
        }

    def _analyze_context_internal(self, sentences: tuple) -> Dict[str, Any]:
        """내부 문맥 분석 함수(캐시 대상)."""
        return self._analyze_context_compute(sentences)

    def _analyze_context(self, sentences: Tuple[str, ...]) -> Dict[str, Any]:
        """ 캐시를 경유하여 문맥 분석을 수행 """
        if not isinstance(sentences, tuple):
            sentences = tuple(sentences or ())

        # cache hit/miss 메트릭 갱신
        try:
            before = self._context_cache.cache_info()
            out = self._context_cache(sentences)
            after = self._context_cache.cache_info()
            self.metrics['cache_hits'] += max(0, after.hits - before.hits)
            self.metrics['cache_misses'] += max(0, after.misses - before.misses)
            return out
        except Exception:
            # 캐시 미세그먼트 오류 시 직접 호출
            return self._analyze_context_compute(sentences)

    def _analyze_context_compute(self, sentences: Tuple[str, ...]) -> Dict[str, Any]:
        """ 라벨링 데이터 기반의 최적화된 문맥 분석(실제 계산) """
        try:
            context_info: Dict[str, Any] = {
                "situation_type": None,
                "time_indicators": [],
                "location_indicators": [],
                "social_context": [],
                "emotional_triggers": [],
                "intensity_markers": []
            }
            emotion_cache = self._get_emotion_cache()

            norm_sentences: Tuple[str, ...] = tuple(
                s.strip() for s in sentences if isinstance(s, str) and s.strip()
            )
            for sentence in norm_sentences:
                situation_type = self._identify_situation_recursively(sentence, emotion_cache)
                if isinstance(situation_type, str) and situation_type:
                    context_info["situation_type"] = situation_type
                self._extract_context_elements(sentence, emotion_cache, context_info)

            cleaned = self._clean_and_sort_context_info(context_info)
            if logger.isEnabledFor(logging.DEBUG):
                try:
                    logger.debug("정리된 문맥 정보: %s", json.dumps(cleaned, ensure_ascii=False))
                except Exception:
                    pass
            return cleaned
        except Exception as e:
            logger.error(f"문맥 분석 중 오류 발생: {str(e)}")
            return {}

    # ---------------------------------------------------------------------
    # 감정 캐시 구축 (내부용, 반환은 내부에서만 사용)
    # ---------------------------------------------------------------------
    def _build_emotion_cache(self) -> Dict[str, Any]:
        cfg = getattr(self, "config", {}) or {}
        if not isinstance(cfg, dict):
            import types as _types
            if isinstance(cfg, _types.ModuleType):
                cfg = {k: getattr(cfg, k) for k in dir(cfg) if k.isupper()}
            else:
                try:
                    cfg = dict(cfg)
                except Exception:
                    cfg = {}
        self.config = cfg
        try:
            from collections import defaultdict
            from typing import TypedDict, Dict as _Dict, Any as _Any, List as _List, Set as _Set, DefaultDict as _DefaultDict, Iterable, Tuple as _Tuple

            class SituationEntry(TypedDict):
                name: str
                keywords: _Set[str]
                examples: _Set[str]
                variations: _Set[str]

            Indicators = _Dict[str, _Set[str]]
            TransItem = _Dict[str, Any]
            situations_dd: _DefaultDict[str, _List[SituationEntry]] = defaultdict(list)
            indicators: Indicators = {"time": set(), "location": set(), "social": set(), "triggers": set()}
            transitions_map_dd: _DefaultDict[str, _List[TransItem]] = defaultdict(list)
            sub_emotions_cache: Dict[str, Dict[str, Any]] = {}

            def _extend_set(target: Set[str], values: Iterable[Any]) -> None:
                for v in values or []:
                    if isinstance(v, str) and v.strip():
                        target.add(v.strip())

            def _declared_from(obj: Dict[str, Any]) -> Indicators:
                out: Indicators = {"time": set(), "location": set(), "social": set(), "triggers": set()}
                candidates = []
                if isinstance(obj, dict):
                    candidates.append(obj.get("indicators", {}))
                    candidates.append(obj.get("context_indicators", {}))
                    for k in ("time_indicators", "location_indicators", "social_indicators", "emotional_triggers"):
                        if k in obj:
                            candidates.append({k: obj.get(k)})
                for c in candidates:
                    if not isinstance(c, dict):
                        continue
                    _extend_set(out["time"], c.get("time") or c.get("time_indicators") or [])
                    _extend_set(out["location"], c.get("location") or c.get("location_indicators") or [])
                    _extend_set(out["social"], c.get("social") or c.get("social_indicators") or c.get("social_context") or [])
                    _extend_set(out["triggers"], c.get("triggers") or c.get("emotional_triggers") or [])
                return out

            global_declared: Indicators = {"time": set(), "location": set(), "social": set(), "triggers": set()}
            fallback_seed: Indicators = {
                "time": {"오늘", "어제", "지금", "곧", "방금", "오랜만에"},
                "location": {"집", "학교", "회사", "밖", "안", "여기"},
                "social": {"가족", "친구", "동료", "함께", "우리"},
                "triggers": set(),
            }

            if not isinstance(self.emotions_data, dict):
                logger.error("emotions_data가 dict 형식이 아닙니다.")
                return {}
                
            for category_name, category_data in self.emotions_data.items():
                for source_key in ("context_patterns", "metadata", "emotion_profile"):
                    src = category_data.get(source_key, {}) or {}
                    dec = _declared_from(src)
                    for k in indicators.keys():
                        indicators[k] |= dec[k]
                        global_declared[k] |= dec[k]

                cat_transitions = category_data.get('emotion_transitions', {}) or {}
                patterns = cat_transitions.get('patterns', []) or []
                for p in patterns:
                    from_e = p.get('from_emotion')
                    to_e = p.get('to_emotion')
                    if from_e and to_e:
                        trig_val = p.get('triggers') or []
                        trig_list = [trig_val] if isinstance(trig_val, str) else [str(t) for t in (trig_val or [])]
                        ta = p.get('transition_analysis', {}) or {}
                        _extend_set(indicators["triggers"], trig_list)
                        _extend_set(indicators["triggers"], ta.get("trigger_words", []))
                        transitions_map_dd[from_e].append({"to": to_e, "triggers": trig_list})

                raw_subs = category_data.get('sub_emotions') or category_data.get('emotion_profile', {}).get('sub_emotions', {})
                if isinstance(raw_subs, dict):
                    subs_iter: Iterable[_Tuple[str, Dict[str, Any]]] = raw_subs.items()
                elif isinstance(raw_subs, list):
                    subs_iter = ((str(i), x) for i, x in enumerate(raw_subs) if isinstance(x, dict))
                else:
                    subs_iter = []

                for sub_emotion_name, sub_data in subs_iter:
                    emotion_id = f"{category_name}-{sub_emotion_name}"
                    sub_emotions_cache[emotion_id] = {
                        "metadata": sub_data.get("metadata", {}),
                        "core_keywords": sub_data.get("core_keywords", []),
                        "context_patterns": sub_data.get("context_patterns", {}),
                        "sentiment_analysis": sub_data.get("sentiment_analysis", {})
                    }
                    for source_key in ("context_patterns", "metadata", "emotion_profile"):
                        src = sub_data.get(source_key, {}) or {}
                        dec = _declared_from(src)
                        for k in indicators.keys():
                            indicators[k] |= dec[k]
                            global_declared[k] |= dec[k]

                    context_patterns = sub_data.get('context_patterns', {}) or {}
                    raw_situations = context_patterns.get('situations', {}) or {}
                    if isinstance(raw_situations, dict):
                        sit_iter: Iterable[_Tuple[str, Any]] = raw_situations.items()
                    elif isinstance(raw_situations, list):
                        sit_iter = ((f"situation_{i}", s) for i, s in enumerate(raw_situations))
                    else:
                        sit_iter = []
                    for situation_name, situation_data in sit_iter:
                        if not isinstance(situation_data, dict):
                            if isinstance(situation_data, list):
                                situation_data = {"keywords": situation_data, "examples": [], "variations": []}
                            else:
                                continue
                        keywords_set: Set[str] = set(situation_data.get('keywords', []) or [])
                        examples_set: Set[str] = set(situation_data.get('examples', []) or [])
                        variations_set: Set[str] = set(situation_data.get('variations', []) or [])
                        situations_dd[emotion_id].append({
                            "name": situation_name,
                            "keywords": keywords_set,
                            "examples": examples_set,
                            "variations": variations_set
                        })
                        dec = _declared_from(situation_data)
                        for k in indicators.keys():
                            indicators[k] |= dec[k]
                        for text in list(examples_set) + list(variations_set) + list(keywords_set):
                            for tw in global_declared["time"]:
                                if tw in text:
                                    indicators["time"].add(tw)
                            for lw in global_declared["location"]:
                                if lw in text:
                                    indicators["location"].add(lw)
                            for sw in global_declared["social"]:
                                if sw in text:
                                    indicators["social"].add(sw)
                        trig = (situation_data.get('emotion_progression', {}) or {}).get('trigger')
                        if isinstance(trig, str) and trig:
                            indicators["triggers"].add(trig)

                    sub_transitions = sub_data.get('emotion_transitions', {}) or {}
                    sub_patterns = sub_transitions.get('patterns', []) or []
                    for sp in sub_patterns:
                        f_e = sp.get('from_emotion')
                        t_e = sp.get('to_emotion')
                        if f_e and t_e:
                            trig_val = sp.get('triggers') or []
                            trig_list = [trig_val] if isinstance(trig_val, str) else [str(t) for t in (trig_val or [])]
                            ta = sp.get('transition_analysis', {}) or {}
                            _extend_set(indicators["triggers"], trig_list)
                            _extend_set(indicators["triggers"], ta.get("trigger_words", []))
                            transitions_map_dd[f_e].append({"to": t_e, "triggers": trig_list})

            for k in ("time", "location", "social"):
                if not indicators[k] and bool(self.config.get("allow_fallback_indicators", False)):
                    indicators[k] |= fallback_seed[k]

            self.emotion_cache = {
                "situations": situations_dd,
                "indicators": indicators,
                "sub_emotions": sub_emotions_cache,
                "transitions_map": transitions_map_dd
            }
            return self.emotion_cache
        except Exception as e:
            logger.error(f"감정 캐시 구축 중 오류 발생: {str(e)}")
            return {}

    def _identify_situation_recursively(self, sentence: str, emotion_cache: Dict[str, Any]) -> Optional[str]:
        try:
            rx = getattr(self, "_rx_phrase", None)
            if not rx:
                return None
            scores: DefaultDict[str, float] = defaultdict(float)
            for match in rx.finditer(sentence):
                phrase = match.group(0)
                for emo_id, weight in self._phrase2emo.get(phrase, ()):
                    scores[emo_id] += weight
            if not scores:
                return None
            return max(scores.items(), key=lambda x: x[1])[0]
        except Exception as e:
            logger.error(f"상황 유형을 판별하는 중 오류가 발생했습니다: {str(e)}")
            return None

    def _extract_context_elements(self, sentence: str, emotion_cache: Dict[str, Any], context_info: Dict[str, Any]) -> None:
        try:
            rx_time, rx_loc = self._rx_time, self._rx_location
            rx_soc, rx_tri = self._rx_social, self._rx_triggers

            if rx_time:
                context_info["time_indicators"] += [m.group(0) for m in rx_time.finditer(sentence)]
            if rx_loc:
                context_info["location_indicators"] += [m.group(0) for m in rx_loc.finditer(sentence)]
            if rx_soc:
                context_info["social_context"] += [m.group(0) for m in rx_soc.finditer(sentence)]
            if rx_tri:
                context_info["emotional_triggers"] += [m.group(0) for m in rx_tri.finditer(sentence)]
        except Exception as e:
            logger.error(f"문맥 요소를 추출하는 중 오류가 발생했습니다: {str(e)}")

    def _clean_and_sort_context_info(self, context_info: Dict[str, Any]) -> Dict[str, Any]:
        try:
            cleaned_info = {}
            for key, value in context_info.items():
                if isinstance(value, list):
                    cleaned_info[key] = sorted(list(set(value)))
                else:
                    cleaned_info[key] = value
            return cleaned_info
        except Exception as e:
            logger.error(f"문맥 정보 정리 중 오류 발생: {str(e)}")
            return context_info

    def _preprocess_text(self, text: str) -> List[str]:
        try:
            sentences = self._split_into_sentences(text)
            return sentences
        except Exception as e:
            logger.exception(f"텍스트 전처리 중 오류 발생: {str(e)}")
            return []

    def _split_into_sentences(self, text: str) -> List[str]:
        """kss 폴백 처리"""
        try:
            import kss  # 지역 임포트
            sentences = kss.split_sentences(text)
            split_sentences = [sent.strip() for sent in sentences if isinstance(sent, str) and sent.strip()]
            logger.debug(f"문장 분리 결과: {split_sentences}")
            return split_sentences
        except Exception:
            # 간단한 폴백: 문장부호 기준 분할
            rough = re.split(r'(?<=[.!?])\s+|\n+', str(text))
            split_sentences = [s.strip() for s in rough if s and s.strip()]
            logger.debug(f"(fallback) 문장 분리 결과: {split_sentences}")
            return split_sentences or [text]

    def _validate_input(self, text: str) -> bool:
        if not isinstance(text, str):
            logger.error("입력된 텍스트가 문자열이 아닙니다.")
            return False
        if len(text) < self.config.get('thresholds', {}).get('minimum_context_length', 10):
            logger.warning("입력된 텍스트가 너무 짧습니다.")
            return False
        return True

    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        return {
            "error": error_message,
            "context_info": {},
            "emotion_sequence": [],
            "confidence_scores": {},
            "metrics": self._get_metrics(),
            "emotion_flows": {},
            "emotion_patterns": {},
            "context_summary": {}
        }

    def _get_metrics(self) -> Dict[str, Any]:
        return {
            "memory_usage_mb": self.metrics['memory_usage'] / (1024 * 1024),
            "cache_hits": self.metrics['cache_hits'],
            "cache_misses": self.metrics['cache_misses'],
            "processing_time_ms": self.metrics['processing_time'] * 1000
        }

    def _cleanup_resources(self):
        try:
            if hasattr(self, "_context_cache") and hasattr(self._context_cache, "cache_clear"):
                self._context_cache.cache_clear()
        except Exception:
            pass
        gc.collect()
        logger.info("Resources cleaned up")

    # ---------------------------------------------------------------------
    # 재귀 분석
    # ---------------------------------------------------------------------
    def _recursive_emotion_analysis(
        self,
        sentences: List[str],
        emotions_data: Dict[str, Any],
        context_info: Dict[str, Any],
        current_depth: int = 0,
        max_depth: int = 3,
        parent_emotion: Optional[str] = None
    ) -> Dict[str, Any]:
        if current_depth >= max_depth:
            return {}
        try:
            analysis_results = {
                "emotion_sequence": [],
                "sub_emotions": defaultdict(list),
                "emotion_hierarchy": defaultdict(dict)
            }
            for sentence in sentences:
                sentence_emotions = self._analyze_sentence_emotions(
                    sentence,
                    emotions_data,
                    context_info,
                    parent_emotion
                )
                if sentence_emotions:
                    analysis_results["emotion_sequence"].extend(sentence_emotions)
                    for emotion_entry in sentence_emotions:
                        if "emotions" in emotion_entry and emotion_entry["emotions"]:
                            current_emotion = emotion_entry["emotions"][0].get("emotion_id")
                            if current_emotion:
                                sub_emotion_data = self._get_sub_emotion_data(emotions_data, current_emotion)
                                if sub_emotion_data:
                                    sub_analysis = self._recursive_emotion_analysis(
                                        sentences=[sentence],
                                        emotions_data=sub_emotion_data,
                                        context_info=context_info,
                                        current_depth=current_depth + 1,
                                        max_depth=max_depth,
                                        parent_emotion=current_emotion
                                    )
                                    if sub_analysis:
                                        analysis_results["sub_emotions"][current_emotion].extend(
                                            sub_analysis.get("emotion_sequence", [])
                                        )
                                        analysis_results["emotion_hierarchy"][current_emotion].update(
                                            sub_analysis.get("emotion_hierarchy", {})
                                        )
            return analysis_results
        except Exception as e:
            logger.error(f"재귀적 감정 분석 중 오류 발생: {str(e)}")
            return {}

    def _get_sub_emotion_data(self, emotions_data: Dict[str, Any], emotion_id: str) -> Dict[str, Any]:
        try:
            if '-' in emotion_id:
                category, sub_emotion = emotion_id.split('-', 1)
            else:
                category, sub_emotion = emotion_id, None

            emotion_data = emotions_data.get(category, {})
            if sub_emotion:
                sub_emotions = emotion_data.get('sub_emotions', {})
                if not sub_emotions:
                    sub_emotions = emotion_data.get('emotion_profile', {}).get('sub_emotions', {})
                return sub_emotions.get(sub_emotion, {})
            return emotion_data
        except Exception as e:
            logger.error(f"세부 감정 데이터 추출 중 오류 발생: {str(e)}")
            return {}

    def _analyze_sentence_emotions(
        self,
        sentence: str,
        emotions_data: Dict[str, Any],
        context_info: Dict[str, Any],
        parent_emotion: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        1) 패턴 추출 모듈(self.pattern_extractor) 활용 가능
        2) 강도 분석(self.intensity_analyzer) 활용 가능
        3) [라벨링뼈대] 기반 감정 분석(폴백 포함)
        """
        emotion_sequence = []
        
        # B1) 문장→후보 감정 사전 필터링(전수 스캔 제거)
        # 1) 인덱스 기반 후보 추출
        rx = getattr(self, "_rx_phrase", None)
        cand_ids = []
        if rx:
            from collections import defaultdict
            hits = defaultdict(float)
            for m in rx.finditer(sentence):
                ph = m.group(0)
                for emo_id, w in self._phrase2emo.get(ph, []):
                    hits[emo_id] += w
            cand_ids = [eid for eid, _ in sorted(hits.items(), key=lambda x: x[1], reverse=True)
                        [:int(self.config.get("candidate_topk", 50))]]

        def iter_subs():
            if cand_ids:
                for emo_id in cand_ids:
                    cat, sub = self._split_emotion_id(emo_id)
                    subs = emotions_data.get(cat, {}).get('sub_emotions', {}) \
                           or emotions_data.get(cat, {}).get('emotion_profile', {}).get('sub_emotions', {})
                    sd = subs.get(sub, {})
                    if sd: 
                        yield cat, sub, sd
            else:
                # 폴백: 전수 스캔
                for cat, item in emotions_data.items():
                    subs = item.get('sub_emotions') or item.get('emotion_profile', {}).get('sub_emotions', {})
                    for sub, sd in (subs or {}).items():
                        yield cat, sub, sd

        for cat, sub, sub_data in iter_subs():
            emo_id = f"{cat}-{sub}"
            intensity = self._calculate_emotion_intensity(sentence, sub_data, emo_id)
            if intensity > 0:
                emotion_complexity = sub_data.get('metadata', {}).get('emotion_complexity', 'basic')
                intensity_label = self._get_intensity_label(intensity, sub_data)
                emotion_sequence.append({
                    'sentence': sentence,
                    'emotions': [{
                        'emotion_id': emo_id,
                        'confidence': float(min(max(intensity, 0.0), 1.0)),
                        'emotion_complexity': emotion_complexity,
                        'intensity_label': intensity_label
                    }],
                    'intensity': {'score': float(min(max(intensity, 0.0), 1.0))}
                })
        return emotion_sequence

    def _calculate_emotion_intensity(self, sentence: str, emotion_data: Dict[str, Any], emo_id: Optional[str] = None) -> float:
        """
        라벨링/문맥/패턴/수정자/관련감정을 종합한 강도 산출 (0~1)
        - 한글 경계/파생(어미/접미) 허용
        - context_patterns(list)과 raw(dict)를 분리해 안전 사용
        - (4) 상황 패턴 매칭 파트, (9) 패턴 종합 점수 반영 보강
        """
        try:
            if not sentence or not isinstance(emotion_data, dict):
                return 0.1

            intensity_components = {
                "keyword_match": 0.0,
                "pattern_match": 0.0,
                "situation_match": 0.0,
                "modifier_impact": 1.0,  # 곱셈
                "intensity_level": 0.0,
                "linguistic_match": 0.0
            }

            # --- 패턴 준비: list(정규화된 패턴) vs dict(raw 컨테이너) ---
            cp_list = self._extract_context_patterns(emotion_data) or []  # List[dict]
            cp_raw = (emotion_data.get('context_patterns') or {}) if isinstance(emotion_data.get('context_patterns'),
                                                                                dict) else {}

            # --- (1) 핵심 키워드 매칭 (정확/부분 + 한글 경계/파생 허용) ---
            if emo_id:
                keyword_source = self._cached_core_keywords(emo_id)
            else:
                keyword_source = self._extract_core_keywords(emotion_data) or []
            keywords = [kw for kw in keyword_source if isinstance(kw, str) and kw]
            matched_keywords: Set[str] = set()
            if keywords:
                import re
                def has_hangul_boundary(kw: str, sent: str) -> bool:
                    return bool(re.search(rf'(?<![\uac00-\ud7a3A-Za-z0-9]){re.escape(kw)}(?![\uac00-\ud7a3A-Za-z0-9])', sent))

                def has_suffix_variant(kw: str, sent: str) -> bool:
                    return bool(re.search(rf'{re.escape(kw)}[가-힣A-Za-z0-9]{{0,2}}', sent))

                exact_cnt, partial_cnt = 0, 0
                for kw in keywords:
                    if kw in sentence:
                        matched_keywords.add(kw)
                        if has_hangul_boundary(kw, sentence) or has_suffix_variant(kw, sentence):
                            exact_cnt += 1
                        else:
                            partial_cnt += 1

                toks = re.findall(r'[가-힣A-Za-z0-9]+', sentence)
                length_norm = max(1.0, len(set(toks)) ** 0.5)

                exact_score = (exact_cnt * 0.35) / length_norm
                partial_score = (partial_cnt * 0.12) / length_norm
                intensity_components["keyword_match"] = min(1.0, exact_score + partial_score)
            else:
                matched_keywords = set()
            # --- (2) 라벨 intensity_levels 예시 유사도 ---
            levels = (emotion_data.get('emotion_profile') or {}).get('intensity_levels') or {}
            if levels:
                best = 0.0
                for level in ('low', 'medium', 'high'):
                    examples = levels.get(f'{level}_examples') \
                               or (levels.get('intensity_examples') or {}).get(level, [])
                    if not examples:
                        continue
                    for ex in examples:
                        if not isinstance(ex, str):
                            continue
                        sim = self._calculate_sentence_similarity(sentence, ex)  # 0~1
                        lvl_w = {'low': 0.3, 'medium': 0.6, 'high': 1.0}.get(level, 0.5)
                        best = max(best, sim * lvl_w)
                intensity_components["intensity_level"] = best

            # === (4) 상황 패턴 매칭 파트 (list 패턴 + 진행 보너스) ===
            # 4-1) list 패턴 점수
            if cp_list:
                pattern_scores: List[float] = []
                for p in cp_list:
                    try:
                        inten = (p.get('intensity') or 'medium').lower()
                        kws = [k for k in (p.get('keywords') or []) if isinstance(k, str)]
                        exs = [e for e in (p.get('examples') or []) if isinstance(e, str)]
                        vars_ = [v for v in (p.get('variations') or []) if isinstance(v, str)]

                        score = 0.0
                        if kws:
                            match_cnt = sum(1 for k in kws if k and (k in sentence))
                            score += (match_cnt / len(kws)) * 0.4
                        if exs and any((ex and (ex in sentence)) for ex in exs):
                            score += 0.5
                        if vars_:
                            var_cnt = sum(1 for v in vars_ if v and (v in sentence))
                            if var_cnt:
                                score += 0.3 * min(1.0, var_cnt / len(vars_))

                        mult = {'low': 0.5, 'medium': 0.8, 'high': 1.2}.get(inten, 0.8)
                        pattern_scores.append(score * mult)
                    except Exception:
                        continue

                if pattern_scores:
                    intensity_components["situation_match"] = min(1.0, max(pattern_scores))

            # 4-2) 진행 보너스 (dict 기반: situations[].emotion_progression)
            try:
                situations = (cp_raw.get('situations') or {}) if isinstance(cp_raw, dict) else {}
                prog_bonus = 0.0
                stage_w = {'trigger': 0.2, 'development': 0.4, 'peak': 0.8, 'aftermath': 0.3}
                for s in situations.values():
                    prog = (s or {}).get('emotion_progression') or {}
                    for stage, w in stage_w.items():
                        st_text = prog.get(stage)
                        if isinstance(st_text, str) and st_text:
                            kws = self._extract_keywords_from_text(st_text)  # 한글 2글자+
                            if not kws:
                                continue
                            matches = sum(1 for k in kws if k in sentence)
                            if matches:
                                prog_bonus = max(prog_bonus, w * (matches / len(kws)))
                if prog_bonus > 0:
                    intensity_components["situation_match"] = min(
                        1.0, intensity_components["situation_match"] + prog_bonus * 0.3
                    )
            except Exception:
                pass

            # --- (5) 언어적 패턴(linguistic_patterns) ---
            lp = emotion_data.get('linguistic_patterns') or {}
            if lp:
                phrase_best = 0.0
                for ph in (lp.get('key_phrases') or []):
                    try:
                        pat = (ph or {}).get('pattern', '')
                        if pat and isinstance(pat, str) and pat in sentence:
                            phrase_best = max(phrase_best, float(ph.get('weight', 0.5)))
                    except Exception:
                        continue
                intensity_components["linguistic_match"] = min(1.0, phrase_best)

            # === (9) 패턴 매칭 종합 점수 (기존 함수 안전 호출) ===
            # 우리 버전은 list/dict 모두 수용하므로 일관되게 list 전달
            try:
                pm = float(self._calculate_pattern_matches(sentence, cp_list))
                intensity_components["pattern_match"] = max(0.0, min(1.0, pm))
            except Exception:
                # 혹시 커스텀 버전이 dict를 더 잘 처리한다면 예비 시도
                try:
                    pm = float(self._calculate_pattern_matches(sentence, cp_raw))
                    intensity_components["pattern_match"] = max(0.0, min(1.0, pm))
                except Exception:
                    pass

            # --- (6) 증폭/감쇠 수정자 ---
            sa = (emotion_data.get('sentiment_analysis')
                  or (emotion_data.get('emotion_profile') or {}).get('sentiment_analysis') or {})
            modifiers = (sa.get('intensity_modifiers') or {}) if isinstance(sa, dict) else {}
            for amp in (modifiers.get('amplifiers') or []):
                if isinstance(amp, str) and amp and amp in sentence:
                    intensity_components["modifier_impact"] *= 1.5
            for dim in (modifiers.get('diminishers') or []):
                if isinstance(dim, str) and dim and dim in sentence:
                    intensity_components["modifier_impact"] *= 0.6
            if any(neg in sentence for neg in ("않", "못", "아니")) and intensity_components["keyword_match"] > 0:
                intensity_components["modifier_impact"] *= 0.8

            # --- (7) 관련 감정 보너스 ---
            rel = (emotion_data.get('emotion_profile') or {}).get('related_emotions') or {}
            related_bonus = 0.0
            for pe in (rel.get('positive') or []):
                if isinstance(pe, str) and pe in sentence:
                    related_bonus += 0.1
            for ne in (rel.get('negative') or []):
                if isinstance(ne, str) and ne in sentence:
                    related_bonus -= 0.05
            related_bonus = max(-0.1, min(0.1, related_bonus))

            # --- (8) 가중 합산 → 곱셈 → 바닥값 → [0,1] ---
            weights = {
                "keyword_match": 0.25,
                "intensity_level": 0.20,
                "situation_match": 0.20,
                "linguistic_match": 0.15,
                "pattern_match": 0.15,
                "related_bonus": 0.05
            }
            ws = (
                    intensity_components["keyword_match"] * weights["keyword_match"] +
                    intensity_components["intensity_level"] * weights["intensity_level"] +
                    intensity_components["situation_match"] * weights["situation_match"] +
                    intensity_components["linguistic_match"] * weights["linguistic_match"] +
                    intensity_components["pattern_match"] * weights["pattern_match"] +
                    related_bonus * weights["related_bonus"]
            )
            ws *= float(intensity_components["modifier_impact"])

            if 0 < ws < 0.12:  # 너무 낮게 깎이는 현상 방지
                ws = 0.12

            final_intensity = max(0.0, min(1.0, ws))

            # H4) 캘리브레이터 가중 곱(라벨 기반 자동 보정)
            if emo_id and getattr(self, "calibrator", None) and matched_keywords:
                try:
                    import math
                    cat, sub = self._split_emotion_id(emo_id)
                    emo_key = f"{cat}-{sub}" if sub else cat
                    log_adj = 0.0
                    for kw in matched_keywords:
                        log_adj += self.calibrator.get_pattern_weight(emo_key, kw)
                    log_adj += self.calibrator.get_prior_adj(emo_key)
                    final_intensity *= math.exp(max(-1.5, min(1.5, log_adj)))  # 캡
                    final_intensity = max(0.0, min(1.0, final_intensity))
                except Exception:
                    pass

            logger.debug(
                "강도계산 | sent='%s' | kw=%.3f lvl=%.3f sit=%.3f ling=%.3f pat=%.3f rel=%.3f x mod=%.3f -> %.3f",
                (sentence[:60] + "…") if len(sentence) > 60 else sentence,
                intensity_components["keyword_match"],
                intensity_components["intensity_level"],
                intensity_components["situation_match"],
                intensity_components["linguistic_match"],
                intensity_components["pattern_match"],
                related_bonus,
                intensity_components["modifier_impact"],
                final_intensity
            )
            return final_intensity

        except Exception as e:
            logger.error(f"감정 강도 계산 중 오류: {e}", exc_info=True)
            return 0.1

    def _calculate_sentence_similarity(self, sentence1: str, sentence2: str) -> float:
        """두 문장의 Jaccard 유사도(한글 토큰). 간단/안전."""
        try:
            t1 = set(re.findall(r'[가-힣]+', sentence1))
            t2 = set(re.findall(r'[가-힣]+', sentence2))
            if not t1 or not t2:
                return 0.0
            inter = len(t1 & t2)
            union = len(t1 | t2)
            return (inter / union) if union else 0.0
        except Exception:
            return 0.0

    def _extract_keywords_from_text(self, text: str) -> List[str]:
        """간단 키워드 추출(한글 2+글자, 불용어 제거)"""
        try:
            kws = re.findall(r'[가-힣]{2,}', text or "")
            stop = {'하다', '있다', '되다', '이다', '그', '저', '것'}
            return [k for k in kws if k not in stop]
        except Exception:
            return []

    def _cached_core_keywords(self, emo_id: str) -> Set[str]:
        # B2) 패턴/키워드 추출 LRU 캐시화
        if not hasattr(self, '_keyword_cache'):
            from functools import lru_cache
            
            @lru_cache(maxsize=10000)
            def _get_cached_keywords(emo_id: str) -> tuple:
                keywords: Set[str] = set()
                try:
                    sub_map = (self.emotion_cache.get("sub_emotions") or {})
                    sub_data = sub_map.get(emo_id, {})
                    if isinstance(sub_data, dict):
                        keywords.update(sub_data.get("core_keywords", []) or [])
                        cp = (sub_data.get("context_patterns") or {}).get("situations", {}) or {}
                        for item in cp.values():
                            keywords.update(item.get("keywords", []) or [])
                except Exception:
                    pass
                return tuple(keywords)  # Set은 해시 불가능하므로 tuple로 변환
            
            self._keyword_cache = _get_cached_keywords
        
        return set(self._keyword_cache(emo_id))

    def _extract_core_keywords(self, emotion_data: Dict[str, Any]) -> Set[str]:
        keywords = set()
        try:
            keywords.update(emotion_data.get('core_keywords', []))
            profile_keywords = emotion_data.get('emotion_profile', {}).get('core_keywords', [])
            keywords.update(profile_keywords)
            situations = emotion_data.get('context_patterns', {}).get('situations', {})
            for situation in (situations or {}).values():
                keywords.update(situation.get('keywords', []))
            return keywords
        except Exception as e:
            logger.error(f"핵심 키워드 추출 중 오류 발생: {str(e)}")
            return set()

    def _extract_context_patterns_legacy(self, emotion_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        다양한 스키마(raw)를 안전하게 정규화해서
        [{'name': str, 'keywords': List[str], 'examples': List[str], 'variations': List[str], 'intensity': str}]로 반환
        """

        def _coerce(item: Any, name: str = "") -> Dict[str, Any]:
            # 기본 골격
            pat = {"name": name or "", "keywords": [], "examples": [], "variations": [], "intensity": "medium"}
            if isinstance(item, str):
                pat["keywords"] = [item]
            elif isinstance(item, (list, tuple, set)):
                # 전부 문자열이면 키워드로, dict 섞여 있으면 풀어서 흡수
                if all(isinstance(x, str) for x in item):
                    pat["keywords"] = list(item)
                else:
                    for x in item:
                        if isinstance(x, str):
                            pat["keywords"].append(x)
                        elif isinstance(x, dict):
                            pat["keywords"] += list(map(str, x.get("keywords") or []))
                            pat["examples"] += list(map(str, x.get("examples") or []))
                            pat["variations"] += list(map(str, x.get("variations") or []))
                            if isinstance(x.get("intensity"), str):
                                pat["intensity"] = x["intensity"]
            elif isinstance(item, dict):
                # 일반적인 dict 스키마 흡수
                kws = item.get("keywords") or []
                if isinstance(kws, str): kws = [kws]
                if isinstance(kws, (list, tuple, set)): pat["keywords"] = list(map(str, kws))

                exs = item.get("examples") or []
                if isinstance(exs, str): exs = [exs]
                if isinstance(exs, (list, tuple, set)): pat["examples"] = list(map(str, exs))

                vars_ = item.get("variations") or []
                if isinstance(vars_, str): vars_ = [vars_]
                if isinstance(vars_, (list, tuple, set)): pat["variations"] = list(map(str, vars_))

                if isinstance(item.get("intensity"), str):
                    pat["intensity"] = item["intensity"]
                if not pat["name"] and isinstance(item.get("name"), str):
                    pat["name"] = item["name"]
            return pat

        patterns: List[Dict[str, Any]] = []
        raw = (
                (emotion_data.get("context_patterns") or {})
                or (emotion_data.get("emotion_profile", {}).get("context_patterns") or {})
        )

        # case 1: dict with 'situations'
        if isinstance(raw, dict) and isinstance(raw.get("situations"), dict):
            for name, item in raw["situations"].items():
                patterns.append(_coerce(item, name=str(name)))
            return patterns

        # case 2: plain dict (name -> spec)
        if isinstance(raw, dict):
            for name, item in raw.items():
                patterns.append(_coerce(item, name=str(name)))
            return patterns

        # case 3: list 형태
        if isinstance(raw, list):
            for i, item in enumerate(raw):
                patterns.append(_coerce(item, name=f"pattern_{i}"))
            return patterns

        # 아무것도 못읽으면 빈 리스트
        return []

    def _calculate_pattern_matches(self, sentence: str, patterns: Any) -> float:
        """
        문자열/리스트/딕셔너리 어느 형태의 패턴이 와도 안전하게 스코어 산출.
        반환값은 [0.0, 1.0] 범위.
        """
        try:
            # 패턴이 비정규화로 들어오면 정규화 시도
            if not isinstance(patterns, list):
                # emotion_data 전체를 실수로 넘긴 경우도 방어
                if isinstance(patterns, dict) and ("situations" in patterns or "keywords" in patterns):
                    patterns = self._extract_context_patterns({"context_patterns": patterns})
                else:
                    # 문자열/기타 → 리스트로
                    patterns = [patterns]

            max_score = 0.0
            for pat in patterns:
                # pat 정규화
                if isinstance(pat, str):
                    pat_dict = {"keywords": [pat], "examples": [], "variations": [], "intensity": "medium"}
                elif isinstance(pat, (list, tuple, set)):
                    if all(isinstance(x, str) for x in pat):
                        pat_dict = {"keywords": list(pat), "examples": [], "variations": [], "intensity": "medium"}
                    else:
                        # 혼합이면 최소 스키마만
                        flat_kws = [str(x) for x in pat if isinstance(x, str)]
                        pat_dict = {"keywords": flat_kws, "examples": [], "variations": [], "intensity": "medium"}
                elif isinstance(pat, dict):
                    pat_dict = {
                        "keywords": list(map(str, (pat.get("keywords") or []))),
                        "examples": list(map(str, (pat.get("examples") or []))),
                        "variations": list(map(str, (pat.get("variations") or []))),
                        "intensity": str(pat.get("intensity") or "medium"),
                    }
                else:
                    continue

                # 스코어링
                score = 0.0
                kws = pat_dict["keywords"]
                exs = pat_dict["examples"]
                vars_ = pat_dict["variations"]
                inten = pat_dict["intensity"]

                # 키워드 커버리지(부분문자열 매칭; 한국어에서 \b 경계가 부정확하므로 단순 포함으로)
                if kws:
                    match_cnt = sum(1 for kw in kws if isinstance(kw, str) and kw and (kw in sentence))
                    score += (match_cnt / max(1, len(kws))) * 0.4

                # 예시 하나라도 포함되면 강한 신호
                if exs and any((isinstance(ex, str) and ex and ex in sentence) for ex in exs):
                    score += 0.5

                # 변형(variations) 일부라도 포함되면 보너스
                if vars_:
                    var_cnt = sum(1 for v in vars_ if isinstance(v, str) and v and (v in sentence))
                    if var_cnt:
                        score += 0.3 * min(1.0, var_cnt / max(1, len(vars_)))

                # intensity에 따른 배수
                mul = {"low": 0.6, "medium": 0.9, "high": 1.2}.get(inten, 0.9)
                score *= mul

                max_score = max(max_score, score)

            return float(max(0.0, min(1.0, max_score)))
        except Exception as e:
            logger.error(f"패턴 매칭 점수 계산 중 오류 발생: {e}")
            return 0.0

    def _calculate_situation_intensity(self, sentence: str, emotion_data: Dict[str, Any]) -> float:
        try:
            situations = emotion_data.get('context_patterns', {}).get('situations', {}) or {}
            max_intensity = 0.0
            for situation in situations.values():
                current_intensity = 0.0
                progression = situation.get('emotion_progression', {}) or {}
                if progression:
                    # 진행 단계 가중의 라벨 유도화 - 단계별 키워드/예시 수 기반 동적 계산
                    for stage in ['trigger', 'development', 'peak', 'aftermath']:
                        v = progression.get(stage)
                        base_weights = {'trigger': 0.3, 'development': 0.5, 'peak': 1.0, 'aftermath': 0.7}
                        weight = 0.0
                        # 1) 문자열 설명 매칭
                        if isinstance(v, str) and v and v in sentence:
                            stage_data = progression.get(stage, {})
                            if isinstance(stage_data, dict):
                                # 키워드/예시/설명 길이를 보정 값으로 사용
                                keywords_count = len(stage_data.get('keywords', []) or [])
                                examples_count = len(stage_data.get('examples', []) or [])
                                description_len = len(str(stage_data.get('description', '')))
                                total_features = keywords_count + examples_count + (description_len // 50)
                                weight_adj = min(0.15, max(0.02, total_features * 0.01))
                                weight = base_weights.get(stage, 0.5) + weight_adj
                            else:
                                weight = base_weights.get(stage, 0.5)
                        # 2) dict 스키마 매칭(키워드/예시/변형)
                        elif isinstance(v, dict):
                            kws = [k for k in (v.get('keywords') or []) if isinstance(k, str)]
                            exs = [e for e in (v.get('examples') or []) if isinstance(e, str)]
                            vars_ = [w for w in (v.get('variations') or []) if isinstance(w, str)]
                            score = 0.0
                            if kws:
                                score += (sum(1 for k in kws if k and k in sentence) / max(1, len(kws))) * 0.4
                            if exs and any((e and e in sentence) for e in exs):
                                score += 0.5
                            if vars_:
                                hit = sum(1 for w in vars_ if w and w in sentence)
                                if hit:
                                    score += 0.3 * min(1.0, hit / max(1, len(vars_)))
                            weight = score * base_weights.get(stage, 0.5)
                        # 현 단계에서 계산된 최대치 반영
                        current_intensity = max(current_intensity, float(weight))
                max_intensity = max(max_intensity, current_intensity)
            return max_intensity
        except Exception as e:
            logger.error(f"상황 강도 계산 중 오류 발생: {str(e)}")
            return 0.0

    def _calculate_modifier_impact(self, sentence: str, emotion_data: Dict[str, Any]) -> float:
        try:
            modifier_impact = 1.0
            sentiment_modifiers = emotion_data.get('sentiment_analysis', {}).get('intensity_modifiers', {})
            for amplifier in sentiment_modifiers.get('amplifiers', []) or []:
                if isinstance(amplifier, str) and amplifier in sentence:
                    modifier_impact *= 1.2
            for diminisher in sentiment_modifiers.get('diminishers', []) or []:
                if isinstance(diminisher, str) and diminisher in sentence:
                    modifier_impact *= 0.8
            return modifier_impact
        except Exception as e:
            logger.error(f"수정자 영향 계산 중 오류 발생: {str(e)}")
            return 1.0

    def _apply_intensity_weights(self, components: Dict[str, float]) -> float:
        try:
            weights = {
                "keyword_match": 0.3,
                "pattern_match": 0.3,
                "situation_match": 0.2,
                "modifier_impact": 0.2
            }
            weighted_sum = 0.0
            for k, v in components.items():
                weighted_sum += float(v) * float(weights.get(k, 0.0))
            return weighted_sum
        except Exception as e:
            logger.error(f"강도 가중치 적용 중 오류 발생: {str(e)}")
            return 0.0

    def _normalize_intensity(self, intensity: float) -> float:
        try:
            normalized = max(0.0, min(1.0, float(intensity)))
            return round(normalized, 3)
        except Exception as e:
            logger.error(f"강도 정규화 중 오류 발생: {str(e)}")
            return 0.0

    def _get_intensity_label(self, intensity_value: float, sub_emotion_data: Dict[str, Any]) -> str:
        intensity_config = sub_emotion_data.get('emotion_profile', {}).get('intensity_levels', {})
        thresholds = intensity_config.get('thresholds', {})
        low_thr = self._safe_float(thresholds.get('low'), 0.3)
        med_thr = self._safe_float(thresholds.get('medium'), 0.7)
        if intensity_value >= med_thr:
            return "high"
        elif intensity_value >= low_thr:
            return "medium"
        else:
            return "low"

    # ---------------------------------------------------------------------
    # 신뢰도/주요 감정
    # ---------------------------------------------------------------------
    def _compute_confidence(self, seq: Dict[str, Any], context_info: Dict[str, Any]) -> float:
        try:
            intensity = seq.get("intensity") or {}
            base_confidence: float = self._safe_float(intensity.get("score"), 0.0)
            base_confidence = max(0.0, min(base_confidence, 1.0))

            emotion_id = ""
            emotions = seq.get("emotions") or []
            if isinstance(emotions, list) and emotions:
                first = emotions[0]
                if isinstance(first, dict):
                    maybe_id = first.get("emotion_id")
                    if isinstance(maybe_id, str):
                        emotion_id = maybe_id

            context_weights = {
                "time_indicators": min(self._safe_len(context_info.get("time_indicators")) * 0.05, 0.25),
                "location_indicators": min(self._safe_len(context_info.get("location_indicators")) * 0.05, 0.25),
                "social_context": min(self._safe_len(context_info.get("social_context")) * 0.05, 0.25),
                "emotional_triggers": min(self._safe_len(context_info.get("emotional_triggers")) * 0.05, 0.25),
            }
            context_weight: float = float(sum(context_weights.values()))

            final_confidence: float = (base_confidence * 0.6) + (context_weight * 0.4)
            final_confidence = max(0.0, min(final_confidence, 1.0))

            logger.info(
                "=== 감정 신뢰도 분석 ===\n감정: %s | 기본: %.2f | 문맥가중: %.2f | 최종: %.2f",
                emotion_id or 'N/A', base_confidence, context_weight, final_confidence
            )
            return final_confidence

        except Exception as e:
            logger.error(f"신뢰도 계산 중 오류 발생: {str(e)}")
            return 0.0

    def _determine_primary_emotion(self, emotion_sequence: List[Dict[str, Any]], context_info: Dict[str, Any]) -> Optional[str]:
        try:
            if not emotion_sequence:
                logger.warning("감정 시퀀스가 비어있습니다.")
                return None
            emotion_scores = defaultdict(float)
            for seq in emotion_sequence:
                if not seq.get("emotions"):
                    continue
                emotion_id = seq['emotions'][0].get('emotion_id')
                if not emotion_id:
                    continue
                base_score = self._safe_float(seq['intensity'].get('score'), 0.0)
                context_score = 0.0
                if context_info.get('situation_type') == emotion_id:
                    context_score += 0.2
                emotion_scores[emotion_id] += (base_score + context_score)
            if not emotion_scores:
                logger.warning("감정 점수 계산 결과가 없습니다.")
                return None
            primary_emotion = max(emotion_scores.items(), key=lambda x: x[1])[0]
            logger.info("주요 감정 결정: %s / 점수=%s", primary_emotion, dict(emotion_scores))
            return primary_emotion
        except Exception as e:
            logger.error(f"주요 감정 결정 중 오류 발생: {str(e)}")
            return None

    def _calculate_weighted_confidence(self, emotion_sequence: List[Dict[str, Any]], context_info: Dict[str, Any]) -> Dict[str, float]:
        confidence_scores: Dict[str, float] = {}
        for seq in emotion_sequence:
            if not seq.get("emotions"):
                continue
            emotion_id = seq['emotions'][0].get('emotion_id')

            # 임계값 조회: 카테고리/서브 모두 탐색
            cat, sub = self._split_emotion_id(emotion_id)
            meta = (self.emotions_data.get(cat, {}) or {}).get('ml_training_metadata', {})
            if sub:
                sub_meta = (self.emotions_data.get(cat, {}) or {}).get('sub_emotions', {})
                if not sub_meta:
                    sub_meta = (self.emotions_data.get(cat, {}) or {}).get('emotion_profile', {}).get('sub_emotions', {})
                meta = (sub_meta.get(sub, {}) or {}).get('ml_training_metadata', {}) or meta

            confidence_threshold = self._safe_float((meta.get('confidence_thresholds', {}) or {}).get('basic'), 0.7)

            confidence_score = self._compute_confidence(seq, context_info)
            if confidence_score >= confidence_threshold:
                confidence_scores[emotion_id] = confidence_score
        return confidence_scores

    def _update_metrics(self, processing_time: float):
        self.metrics['processing_time'] = processing_time
        try:
            if self.memory_monitor is not None:
                self.metrics['memory_usage'] = self.memory_monitor.get_usage()
            else:
                self.metrics['memory_usage'] = 0
        except Exception:
            self.metrics['memory_usage'] = 0

    # ---------------------------------------------------------------------
    # 직렬화 안전화 유틸
    # ---------------------------------------------------------------------
    def _json_safe_inplace(self, obj: Any) -> Any:
        """dict/list 내부의 datetime, set, defaultdict 등을 직렬화 안전 타입으로 변환(제자리)"""
        try:
            from datetime import datetime, date
        except Exception:
            datetime = type("dt", (), {})  # type: ignore
            date = datetime  # type: ignore

        if isinstance(obj, dict):
            # defaultdict → dict
            if isinstance(obj, defaultdict):
                obj = dict(obj)
            for k in list(obj.keys()):
                v = obj[k]
                obj[k] = self._json_safe_inplace(v)
            return obj

        if isinstance(obj, list):
            for i, v in enumerate(obj):
                obj[i] = self._json_safe_inplace(v)
            return obj

        if isinstance(obj, set):
            return sorted(list(obj))

        # datetime/date → isoformat
        try:
            if isinstance(obj, (datetime, date)):
                return obj.isoformat()
        except Exception:
            pass

        return obj


# =============================================================================
# MemoryMonitor
# =============================================================================
class MemoryMonitor:
    def __init__(self, memory_limit: int):
        self.memory_limit = memory_limit
        self._use_psutil = psutil is not None
        self.process = psutil.Process() if self._use_psutil else None

    def check_memory(self) -> bool:
        """메모리 사용량 체크"""
        if self._use_psutil and self.process is not None:
            try:
                current_memory = self.process.memory_info().rss
                return current_memory < self.memory_limit
            except Exception:
                return True
        # psutil이 없으면 보수적으로 통과
        return True

    def get_usage(self) -> int:
        """현재 메모리 사용량 반환"""
        if self._use_psutil and self.process is not None:
            try:
                return self.process.memory_info().rss
            except Exception:
                return 0
        return 0

# =============================================================================
# Independent Function
# =============================================================================
def _clamp_depth(depth: int, lo: int = 1, hi: int = 5) -> int:
    try:
        d = int(depth)
    except Exception:
        d = 3
    return max(lo, min(d, hi))

def _join_pretokenized(pretokenized: Optional[List[str]], fallback_text: str) -> str:
    if isinstance(pretokenized, list) and pretokenized:
        return " ".join(s for s in pretokenized if isinstance(s, str))
    return fallback_text

def _get_or_create_extractor(
    emotions_data: Dict[str, Any],
    extractor: Optional["ContextExtractor"] = None,
    config: Optional[Dict[str, Any]] = None,
    dependencies: Optional[Dict[str, Any]] = None,
) -> "ContextExtractor":
    """
    - 외부에서 주입된 extractor가 있으면 그대로 사용
    - 없으면 새로 생성
    - config/dpendencies 가 있으면 적용
    """
    ext = extractor or ContextExtractor(emotions_data)
    if isinstance(config, dict):
        # 사용자가 넘긴 설정을 가볍게 merge(덮어쓰기)
        try:
            ext.config.update(config)
        except Exception:
            pass
        # 후처리용 기본값 보강
        try:
            ext._ensure_post_cfg_defaults()
        except Exception:
            logger.debug("failed to ensure post cfg defaults", exc_info=True)

    if isinstance(dependencies, dict):
        try:
            ext.set_dependencies(
                pattern_extractor=dependencies.get("pattern_extractor"),
                intensity_analyzer=dependencies.get("intensity_analyzer"),
                transition_analyzer=dependencies.get("transition_analyzer"),
            )
        except Exception:
            logger.debug("failed to set dependencies", exc_info=True)
    return ext

def _error_payload(msg: str, extractor: Optional["ContextExtractor"] = None) -> Dict[str, Any]:
    """클래스의 에러 스키마와 최대한 일관되게 반환"""
    try:
        if extractor is not None:
            return extractor._create_error_response(msg)
    except Exception:
        pass
    # extractor 생성조차 실패했을 때의 폴백 스키마
    return {
        "error": msg,
        "context_info": {},
        "emotion_sequence": [],
        "confidence_scores": {},
        "metrics": {"memory_usage_mb": 0.0, "cache_hits": 0, "cache_misses": 0, "processing_time_ms": 0.0},
        "emotion_flows": {},
        "emotion_patterns": {},
        "context_summary": {},
        "schema_version": "indep-1.0"
    }


# -----------------------------------------------------------------------------
# 1) extract_contextual_emotions
# -----------------------------------------------------------------------------
def extract_contextual_emotions(
    text: str,
    emotions_data: Dict[str, Any],
    max_depth: int = 3,
    *,
    config: Optional[Dict[str, Any]] = None,
    dependencies: Optional[Dict[str, Any]] = None,
    extractor: Optional["ContextExtractor"] = None,
    pretokenized: Optional[List[str]] = None
) -> Dict[str, Any]:
    """text + 라벨링 감정 데이터로 문맥 기반 감정을 추출 (안전성/옵션 강화)
    사용 팁: 최소 길이 검증(기본 10자)을 오버라이드하려면
      config={'thresholds': {'minimum_context_length': 5}} 와 같이 전달하세요.
    """
    ext = None
    try:
        ext = _get_or_create_extractor(emotions_data, extractor, config, dependencies)
        # pretokenized가 있으면 결합해서 사용(문장분리 비용 회피 가능)
        normalized_text = _join_pretokenized(pretokenized, text)

        # 공백/비문자 입력 즉시 폴백
        if not isinstance(normalized_text, str) or not normalized_text.strip():
            return ext._create_error_response("Invalid input (empty)")

        # 클래스의 입력 검증 기준을 그대로 따름
        if not ext._validate_input(normalized_text):
            return ext._create_error_response("Invalid input (too short or not a string)")

        depth = _clamp_depth(max_depth)
        return ext.extract_contextual_emotions(normalized_text, emotions_data, max_depth=depth)
    except Exception as e:
        logger.error(f"[extract_contextual_emotions] 오류: {str(e)}", exc_info=True)
        return _error_payload(str(e), ext)


# -----------------------------------------------------------------------------
# 2) analyze_progressive_context
# -----------------------------------------------------------------------------
def analyze_progressive_context(
    text: str,
    emotions_data: Dict[str, Any],
    *,
    config: Optional[Dict[str, Any]] = None,
    dependencies: Optional[Dict[str, Any]] = None,
    extractor: Optional["ContextExtractor"] = None,
    pretokenized: Optional[List[str]] = None,
    include_base: bool = False
) -> Dict[str, Any]:
    """
    진행형 감정 분석(전이·상황 통합).
    - base 추출 1회만 수행, 그 결과의 시퀀스/컨텍스트로 분석
    - include_base=True 시 base 결과도 함께 반환
    - 최소 길이 검증 오버라이드 예: config={'thresholds': {'minimum_context_length': 5}}
    """
    ext = None
    try:
        ext = _get_or_create_extractor(emotions_data, extractor, config, dependencies)
        normalized_text = _join_pretokenized(pretokenized, text)

        if not isinstance(normalized_text, str) or not normalized_text.strip():
            return ext._create_error_response("Invalid input (empty)")

        if not ext._validate_input(normalized_text):
            return ext._create_error_response("Invalid input (too short or not a string)")

        base = ext.extract_contextual_emotions(normalized_text, emotions_data, max_depth=2)
        emotion_sequence: List[Dict[str, Any]] = cast(List[Dict[str, Any]], base.get("emotion_sequence", []))
        context_info: Dict[str, Any] = cast(Dict[str, Any], base.get("context_info", {}))

        result = ext.analyze_progressive_context(
            text=normalized_text,
            emotion_sequence=emotion_sequence,
            context_info=context_info
        )
        if include_base:
            result = {"base": base, "progressive_analysis": result}
        # 메트릭 동봉(가벼운 가시성)
        result.setdefault("metrics", ext._get_metrics())
        result.setdefault("schema_version", ext.config.get("schema_version", "1.2"))
        return result
    except Exception as e:
        logger.error(f"[analyze_progressive_context] 오류: {str(e)}", exc_info=True)
        return _error_payload(str(e), ext)


# -----------------------------------------------------------------------------
# 3) analyze_situation_impact
# -----------------------------------------------------------------------------
def analyze_situation_impact(
    text: str,
    emotions_data: Dict[str, Any],
    *,
    config: Optional[Dict[str, Any]] = None,
    dependencies: Optional[Dict[str, Any]] = None,
    extractor: Optional["ContextExtractor"] = None,
    pretokenized: Optional[List[str]] = None,
    include_base: bool = False
) -> Dict[str, Any]:
    """상황(문맥)과 감정 전이 정보를 가중치로 계산
    사용 팁: 최소 길이 검증(기본 10자) 오버라이드 예:
      config={'thresholds': {'minimum_context_length': 5}}
    """
    ext = None
    try:
        ext = _get_or_create_extractor(emotions_data, extractor, config, dependencies)
        normalized_text = _join_pretokenized(pretokenized, text)

        if not isinstance(normalized_text, str) or not normalized_text.strip():
            return ext._create_error_response("Invalid input (empty)")

        if not ext._validate_input(normalized_text):
            return ext._create_error_response("Invalid input (too short or not a string)")

        base = ext.extract_contextual_emotions(normalized_text, emotions_data, max_depth=2)
        emotion_sequence = list(base.get("emotion_sequence", []))
        context_info = dict(base.get("context_info", {}))

        result = ext.analyze_situation_impact(normalized_text, emotion_sequence, context_info)
        if include_base:
            result = {"base": base, "situation_impact": result}
        result.setdefault("metrics", ext._get_metrics())
        result.setdefault("schema_version", ext.config.get("schema_version", "1.2"))
        return result
    except Exception as e:
        logger.error(f"[analyze_situation_impact] 오류: {str(e)}", exc_info=True)
        return _error_payload(str(e), ext)


# -----------------------------------------------------------------------------
# 4) analyze_context_patterns
# -----------------------------------------------------------------------------
def analyze_context_patterns(
    text: str,
    emotions_data: Dict[str, Any],
    *,
    config: Optional[Dict[str, Any]] = None,
    dependencies: Optional[Dict[str, Any]] = None,
    extractor: Optional["ContextExtractor"] = None,
    pretokenized: Optional[List[str]] = None
) -> Dict[str, Any]:
    """문맥 패턴을 분석하고 관련 메트릭 반환(클래스 내부 컨텍스트 분석을 그대로 활용)
    사용 팁: 최소 길이 검증 오버라이드 예: config={'thresholds': {'minimum_context_length': 5}}
    """
    ext = None
    try:
        ext = _get_or_create_extractor(emotions_data, extractor, config, dependencies)
        normalized_text = _join_pretokenized(pretokenized, text)

        if not isinstance(normalized_text, str) or not normalized_text.strip():
            return ext._create_error_response("Invalid input (empty)")

        if not ext._validate_input(normalized_text):
            return ext._create_error_response("Invalid input (too short or not a string)")

        sentences = tuple(ext._preprocess_text(normalized_text))
        context_info = ext._analyze_context(sentences)
        return {
            "context_patterns": context_info,
            "metrics": ext._get_metrics(),
            "schema_version": ext.config.get("schema_version", "1.2")
        }
    except Exception as e:
        logger.error(f"[analyze_context_patterns] 오류: {str(e)}", exc_info=True)
        return _error_payload(str(e), ext)


# -----------------------------------------------------------------------------
# 5) analyze_context_emotion_correlations
# -----------------------------------------------------------------------------
def analyze_context_emotion_correlations(
    text: str,
    emotions_data: Dict[str, Any],
    *,
    config: Optional[Dict[str, Any]] = None,
    dependencies: Optional[Dict[str, Any]] = None,
    extractor: Optional["ContextExtractor"] = None,
    pretokenized: Optional[List[str]] = None,
    include_flows: bool = False
) -> Dict[str, Any]:
    """상황과 감정 간의 연관성 분석(상황 상관/요약/선택적 흐름 포함)
    사용 팁: 최소 길이 검증 오버라이드 예: config={'thresholds': {'minimum_context_length': 5}}
    """
    ext = None
    try:
        ext = _get_or_create_extractor(emotions_data, extractor, config, dependencies)
        normalized_text = _join_pretokenized(pretokenized, text)

        if not isinstance(normalized_text, str) or not normalized_text.strip():
            return ext._create_error_response("Invalid input (empty)")

        if not ext._validate_input(normalized_text):
            return ext._create_error_response("Invalid input (too short or not a string)")

        # 내부 상태(컨텍스트 매니저)를 채우기 위해 한 번 돌림
        _ = ext.extract_contextual_emotions(normalized_text, emotions_data, max_depth=2)

        out = {
            "situation_correlations": ext.context_manager._analyze_situation_correlations(),
            "context_summary": ext.context_manager.get_context_summary(),
            "metrics": ext._get_metrics(),
            "schema_version": ext.config.get("schema_version", "1.2")
        }
        if include_flows:
            out["emotion_flows"] = ext.context_manager.track_all_emotion_flows()
        return out
    except Exception as e:
        logger.error(f"[analyze_context_emotion_correlations] 오류: {str(e)}", exc_info=True)
        return _error_payload(str(e), ext)

# =============================================================================
# Main Function
# =============================================================================
logger = logging.getLogger("context_extractor")
if not any(isinstance(h, logging.NullHandler) for h in logger.handlers):
    logger.addHandler(logging.NullHandler())


# ---- Helper: dict 키를 문자열화 ---------------------------------------------------------
def _convert_dict_keys_to_str(data):
    """재귀적으로 딕셔너리의 비문자 키(예: tuple)를 문자열 키로 변환."""
    if isinstance(data, dict):
        new_dict = {}
        for k, v in data.items():
            if isinstance(k, tuple):
                key_str = "->".join(str(x) for x in k)
            else:
                key_str = str(k)
            new_dict[key_str] = _convert_dict_keys_to_str(v)
        return new_dict
    elif isinstance(data, list):
        return [_convert_dict_keys_to_str(item) for item in data]
    return data


# ---- Helper: JSON 인코더 ---------------------------------------------------------------
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, timedelta):
            return obj.total_seconds()
        if isinstance(obj, set):
            return list(obj)
        return super().default(obj)


# ---- Helper: 로거 설정 ------------------------------------------------------------------
def _setup_logger(log_dir: Path, level: str = "INFO", to_file: bool = True) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    lvl = getattr(logging, level.upper(), logging.INFO)

    # 중복 핸들러 방지
    if logger.handlers:
        for h in list(logger.handlers):
            logger.removeHandler(h)

    logger.setLevel(lvl)

    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(lvl)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if to_file:
        file_handler = RotatingFileHandler(
            str(log_dir / 'context_extractor.log'),
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)  # 파일은 상세
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# ---- Helper: 텍스트 로딩 ---------------------------------------------------------------
def _resolve_base_dir() -> Path:
    try:
        return Path(__file__).resolve().parents[2]
    except Exception:
        return Path.cwd()

def _read_text(args: argparse.Namespace, fallback: str) -> str:
    if args.text is not None:
        return args.text
    if args.text_file:
        p = Path(args.text_file)
        try:
            return p.read_text(encoding=args.encoding or "utf-8")
        except Exception as e:
            logger.error(f"텍스트 파일 읽기 실패: {p} ({e})")
            sys.exit(2)
    if args.stdin:
        try:
            data = sys.stdin.read()
            return data
        except Exception as e:
            logger.error(f"STDIN 읽기 실패: {e}")
            sys.exit(2)
    return fallback


# ---- Helper: emotions_data 로딩 --------------------------------------------------------
def _load_emotions_data(path: Path, encoding: str = "utf-8") -> Dict[str, Any]:
    try:
        with path.open('r', encoding=encoding) as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("EMOTIONS.json 최상위 구조는 dict 여야 합니다.")
        return data
    except FileNotFoundError:
        logger.error(f"EMOTIONS.json을 찾을 수 없습니다: {path}")
        sys.exit(1)
    except json.JSONDecodeError as je:
        logger.error(f"EMOTIONS.json 파싱 오류: {str(je)}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"라벨링 감정 데이터 로딩 중 오류: {str(e)}", exc_info=True)
        sys.exit(1)


# ---- Helper: config 오버라이드 ---------------------------------------------------------
def _load_config_override(config_json_path: Optional[str]) -> Dict[str, Any]:
    if not config_json_path:
        return {}
    try:
        p = Path(config_json_path)
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("config JSON은 최상위 dict 여야 합니다.")
        return data
    except Exception as e:
        logger.error(f"config JSON 로딩 실패: {e}")
        return {}


# ---- 콘솔 요약 출력 --------------------------------------------------------------------
def _print_summary(result: Dict[str, Any]) -> None:
    print("\n=== 감정 분석 결과 (콘솔 요약) ===")
    print(f"주요 감정(대표-세부): {result.get('primary_emotion')}")
    print(f"대표 감정: {result.get('primary_category')}")
    print(f"세부 감정: {result.get('sub_category')}")
    print("문맥 정보:")
    print(json.dumps(result.get('context_info'), ensure_ascii=False, indent=2, cls=CustomJSONEncoder))
    print("감정 시퀀스:")
    print(json.dumps(result.get('emotion_sequence'), ensure_ascii=False, indent=2, cls=CustomJSONEncoder))

    if "progressive_analysis" in result:
        print("진행형 감정 분석:")
        print(json.dumps(result.get('progressive_analysis'), ensure_ascii=False, indent=2, cls=CustomJSONEncoder))
    if "situation_impact_analysis" in result:
        print("상황 영향도 분석:")
        print(json.dumps(result.get('situation_impact_analysis'), ensure_ascii=False, indent=2, cls=CustomJSONEncoder))


# ---- 파일 저장 ------------------------------------------------------------------------
def _save_json(output_path: Optional[str], log_dir: Path, data: Dict[str, Any]) -> Optional[Path]:
    if not output_path:
        return None
    # '-' 이면 파일 저장 대신 STDOUT에만 출력
    if output_path.strip() == "-":
        print(json.dumps(_convert_dict_keys_to_str(data), ensure_ascii=False, indent=2, cls=CustomJSONEncoder))
        return None

    out_path = Path(output_path)
    # 상대 경로로 온 경우 log_dir 밑에 저장(기존 동작 호환)
    if not out_path.is_absolute():
        out_path = log_dir / out_path.name

    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open('w', encoding='utf-8') as json_file:
            json.dump(_convert_dict_keys_to_str(data), json_file, ensure_ascii=False, indent=2, cls=CustomJSONEncoder)
        logger.info(f"감정 분석 결과를 JSON 파일로 저장했습니다: {out_path}")
        return out_path
    except Exception as e:
        logger.error(f"JSON 파일 작성 중 오류 발생: {str(e)}", exc_info=True)
        return None


# ---- Main -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Emotion Context Extraction Tester")
    parser.add_argument("--text", type=str, default=None, help="분석 대상 텍스트(직접 입력)")
    parser.add_argument("--text_file", type=str, default=None, help="분석 텍스트 파일 경로")
    parser.add_argument("--stdin", action="store_true", help="STDIN으로 텍스트를 읽음")
    parser.add_argument("--encoding", type=str, default="utf-8", help="텍스트/JSON 파일 인코딩")

    parser.add_argument("--emotions_data_path", type=str, default=None, help="라벨링 감정 데이터(EMOTIONS.json) 경로")
    parser.add_argument("--config_json", type=str, default=None, help="ContextExtractor 설정 오버라이드 JSON 경로")
    parser.add_argument("--max_depth", type=int, default=3, help="재귀 분석 최대 깊이")

    parser.add_argument("--mode", type=str, choices=["extract", "progressive", "situation", "correlations", "patterns"],
                        default="extract", help="실행 모드")
    parser.add_argument("--include_base", action="store_true", help="progressive/situation 결과에 base 포함")

    parser.add_argument("--output_json", type=str, default="context_extractor.json",
                        help="결과 저장 경로. '-'면 STDOUT에만 출력")
    # 통합 로그 관리자 사용 (날짜별 폴더)
    try:
        from log_manager import get_log_manager
        log_manager = get_log_manager()
        default_log_dir = str(log_manager.get_log_dir("emotion_analysis", use_date_folder=True))
    except ImportError:
        # 폴백: 기존 방식 (날짜별 폴더 추가)
        from datetime import datetime
        base_log_dir = 'logs'
        today = datetime.now().strftime("%Y%m%d")
        default_log_dir = os.path.join(base_log_dir, today)
    
    parser.add_argument("--log_dir", type=str, default=default_log_dir, help="로그/기본 출력 디렉터리")
    parser.add_argument("--log_level", type=str, default="INFO", help="로그 레벨(INFO/DEBUG/WARN 등)")
    parser.add_argument("--no_file_log", action="store_true", help="로그 파일 기록 비활성화")
    args = parser.parse_args()

    # 0) 로거
    log_dir = Path(args.log_dir)
    _setup_logger(log_dir, level=args.log_level, to_file=not args.no_file_log)

    logger.info("=== 감정 분석 시작 ===")
    base_dir = _resolve_base_dir()

    # 1) 입력/데이터 로딩
    default_emotions_data_path = base_dir / 'src' / 'EMOTIONS.json'
    emotions_data_path = Path(args.emotions_data_path) if args.emotions_data_path else default_emotions_data_path
    test_text_fallback = "오늘 가족들과 함께 맛있는 음식을 먹으며 즐거운 시간을 보냈습니다. 특히 오랜만에 만난 가족들과 이야기를 나누며 만족감을 느꼈습니다."
    test_text = _read_text(args, test_text_fallback)
    logger.info(f"분석할 텍스트: {test_text[:200]}{'...' if len(test_text) > 200 else ''}")
    logger.info(f"라벨링 감정 데이터 경로: {emotions_data_path}")

    emotions_data = _load_emotions_data(emotions_data_path, encoding=args.encoding)
    config_override = _load_config_override(args.config_json)

    # 2) 실행
    from typing import cast, List
    try:
        # 옵션: 독립 함수들을 사용(이전 단계에서 개선)
        # (모드에 따라 호출 함수 변경)
        if args.mode == "extract":
            result = extract_contextual_emotions(
                text=test_text,
                emotions_data=emotions_data,
                max_depth=args.max_depth,
                config=config_override
            )
        elif args.mode == "progressive":
            base = extract_contextual_emotions(
                text=test_text,
                emotions_data=emotions_data,
                max_depth=2,
                config=config_override
            )
            emotion_sequence = cast(List[Dict[str, Any]], base.get("emotion_sequence", []))
            context_info = cast(Dict[str, Any], base.get("context_info", {}))
            result = analyze_progressive_context(
                text=test_text,
                emotions_data=emotions_data,
                config=config_override,
                include_base=args.include_base
            )
            # 호환: top-level에서도 확인 가능하도록 병합
            result.setdefault("emotion_sequence", emotion_sequence)
            result.setdefault("context_info", context_info)
        elif args.mode == "situation":
            result = analyze_situation_impact(
                text=test_text,
                emotions_data=emotions_data,
                config=config_override,
                include_base=args.include_base
            )
        elif args.mode == "correlations":
            result = analyze_context_emotion_correlations(
                text=test_text,
                emotions_data=emotions_data,
                config=config_override,
                include_flows=True
            )
        else:  # patterns
            result = analyze_context_patterns(
                text=test_text,
                emotions_data=emotions_data,
                config=config_override
            )

        # 3) 결과 검증/로그
        if isinstance(result, dict) and result.get("error"):
            logger.error(f"실행 오류: {result.get('error')}")
            sys.exit(1)

        logger.info("=== 감정 분석 결과 ===")
        logger.info(f"주요 감정(대표-세부): {result.get('primary_emotion')}")
        logger.info(f"대표 감정: {result.get('primary_category')}")
        logger.info(f"세부 감정: {result.get('sub_category')}")
        logger.info(f"문맥 정보: {json.dumps(result.get('context_info'), ensure_ascii=False, cls=CustomJSONEncoder)}")

        if "emotion_sequence" in result:
            logger.info(f"감정 시퀀스 길이: {len(result.get('emotion_sequence', []))}")
        if "progressive_analysis" in result:
            logger.info(f"진행형 감정 분석 결과: {json.dumps(result.get('progressive_analysis'), ensure_ascii=False, cls=CustomJSONEncoder)}")
        if "situation_impact_analysis" in result:
            logger.info(f"상황 영향도 분석 결과: {json.dumps(result.get('situation_impact_analysis'), ensure_ascii=False, cls=CustomJSONEncoder)}")

        # 필수 키 점검(가벼운 헤드업)
        for rk in ["primary_emotion", "context_info", "emotion_sequence"]:
            if rk not in result:
                logger.warning(f"결과에서 '{rk}' 키가 누락되었습니다. 구현을 확인하세요.")

        # 4) 저장/콘솔 요약
        _save_json(args.output_json, log_dir, result)
        _print_summary(result)

        # 5) 정리
        gc.collect()
        logger.info("=== 감정 분석 종료 ===")
        sys.exit(0)

    except KeyboardInterrupt:
        logger.warning("사용자에 의해 중단되었습니다 (KeyboardInterrupt).")
        sys.exit(130)
    except Exception as e:
        logger.error(f"예상치 못한 오류: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
