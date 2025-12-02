# situation_analyzer.py
# -*- coding: utf-8 -*-

# ── Standard library
import os
import sys
import re
import json
import time
import math
import logging
import warnings
import copy
import unicodedata
from pathlib import Path
from collections import defaultdict, Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Set, Union, Iterable
from functools import lru_cache

# ── Third-party
import numpy as np
try:
    import kss  # OK면 그대로 사용
except Exception:
    kss = None  # 미설치면 None으로 폴백(클래스 내부 로직에서 이미 안전 처리)

try:
    import torch
except ImportError:
    torch = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None  # _lazy_load_model()에서 try/except로 안전 처리됨

# ── Module logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.addHandler(logging.NullHandler())
logger.propagate = False

# --- [Situation Analyzer Robustness Add-ons] ---
# Utilities, constants, and soft-fail friendly helpers for situation matching
# without external dependencies. These operate independently and never hard-exit.
# Observability/metrics skeleton always present in outputs
DEFAULT_METRICS = {
    "used_levels": [],
    "fallbacks": {"lexicon": False, "alias": False},
    "coverage": 0.0,
    "jaccard_mean": 0.0,
    "warnings": []
}

# (옵션) Fallback 별칭 맵: 외부 설정(emotions_config)에 있으면 우선 사용, 없으면 이 상수로 보완
FALLBACK_SITUATION_ALIASES = {
    "갈등": {"충돌", "분쟁", "언쟁", "대립"},
    "축하": {"경축", "파티", "치하"},
    "상실": {"이별", "상심", "슬픔"},
    "협상": {"타협", "조율", "교섭"},
}

LEVEL_WEIGHTS = {"L0": 1.0, "L1": 0.9, "L2": 0.8, "L3": 0.6, "L4": 0.5}

def _empty_situation_report(reason: str) -> Dict[str, Any]:
    """빈/오류 입력 시에도 스키마를 보존하는 경량 리포트"""
    return {
        "identified_situations": [],
        "context_mapping": {},
        "situation_metrics": {},
        "emotional_context": {},
        "spatiotemporal_context": {},
        "situational_triggers": [],
        "metrics": {
            "processing_time": 0.0,
            "memory_usage_kb": 0,
            "situations_identified": 0,
            "success_rate": False,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "warnings": [reason],
        },
        "error": reason,
    }

def nfkc_norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    return " ".join(s.split()).lower()

def is_junk_text(s: str):
    """ 간단 Junk 감지: 문자군 비율, 유니크/길이, 반복 런, 간이 엔트로피 """
    if not s or s.strip() == "":
        return True, "EMPTY"
    s_norm = nfkc_norm(s)
    n = len(s_norm)
    if n < 3:
        return True, "TOO_SHORT"
    letters = sum(ch.isalpha() for ch in s_norm)
    digits = sum(ch.isdigit() for ch in s_norm)
    puncts = sum(not (ch.isalnum() or ch.isspace()) for ch in s_norm)
    alpha_ratio = letters / max(n, 1)
    punct_ratio = puncts / max(n, 1)
    uniq_ratio = len(set(s_norm)) / max(n, 1)
    # 간이 엔트로피
    from collections import Counter as _Ctr
    c = _Ctr(s_norm)
    ent = -sum((cnt / n) * math.log((cnt / n) + 1e-12) for cnt in c.values())
    # 반복 런(같은 문자 4회 이상)
    max_run, run, prev = 1, 1, None
    for ch in s_norm:
        if ch == prev:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 1
            prev = ch
    # 휴리스틱 임계
    if punct_ratio > 0.55 or uniq_ratio < 0.15 or max_run >= 6 or ent < 1.2:
        return True, f"NOISY(punct={punct_ratio:.2f},uniq={uniq_ratio:.2f},run={max_run},H={ent:.2f})"
    return False, ""

def bigrams(s: str):
    s = nfkc_norm(s).replace(" ", "")
    return {s[i:i+2] for i in range(len(s)-1)} if len(s) > 1 else set()

def jaccard(a: set, b: set):
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / (union or 1)

def edit_distance_le2(a: str, b: str, max_d=2):
    """ 제한적 Levenshtein: 거리 2 초과 시 즉시 중단 (길이 24자 가드 권장) """
    a, b = nfkc_norm(a), nfkc_norm(b)
    if abs(len(a) - len(b)) > max_d:
        return max_d + 1
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        prev, dp[0] = dp[0], i
        for j, cb in enumerate(b, start=1):
            cur = dp[j]
            cost = 0 if ca == cb else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = cur
        if min(dp) > max_d:
            return max_d + 1
    return dp[-1]

def load_aliases_from_emotions(emotions: dict):
    # emotions["situations"]["aliases"] 구조가 있으면 사용
    try:
        aliases = emotions.get("situations", {}).get("aliases", {})
        # 표준화: set으로 캐스팅
        return {k: set(v) for k, v in aliases.items() if isinstance(v, (list, set))}
    except Exception:
        return {}

def level_match(text: str, key: str, level: str):
    """ 레벨별 매칭 여부/점수 """
    if level == "L0":
        hit = key in text
        return hit, 1.0 if hit else 0.0
    if level == "L1":
        hit = nfkc_norm(key) in nfkc_norm(text)
        return hit, 0.9 if hit else 0.0
    if level == "L3":
        j = jaccard(bigrams(text), bigrams(key))
        return j >= 0.45, min(0.6 + (j - 0.45) * 1.0, 0.8)  # 0.6~0.8
    if level == "L4":
        if len(key) > 24:
            return False, 0.0
        d = edit_distance_le2(text, key, 2)
        return d <= 2, 0.5 if d <= 2 else 0.0
    return False, 0.0

def infer_simple_signals(text: str):
    t = nfkc_norm(text)
    # 간이 강도/정서 힌트
    pos_hint = any(k in t for k in ["축하", "경축", "행복", "기쁨", "고마"])
    neg_hint = any(k in t for k in ["분노", "짜증", "화가", "불안", "슬픔", "상실"])
    amp_hint = any(k in t for k in ["!!!", "??", "ㅠㅠ", "ㅋㅋ", "ㅎㅎ"]) or t.count("!") >= 2
    return {
        "valence": (1 if pos_hint and not neg_hint else (-1 if neg_hint and not pos_hint else 0)),
        "amplified": bool(amp_hint)
    }

# 상황 카테고리의 기대 신호(예시, 실제 프로젝트 용어에 맞게 확장)
EXPECTED = {
    "갈등": {"valence": -1, "amplified": True},
    "축하": {"valence": +1, "amplified": True},
    "상실": {"valence": -1, "amplified": False},
    "협상": {"valence": 0,  "amplified": False},
}

def reconcile_with_emotions(situation: str, snippet: str, intensity_profile=None, transitions=None):
    """ 전이/강도 신호와의 충돌 시 가중 하향 """
    exp = EXPECTED.get(situation, None)
    if not exp:
        return 1.0, None  # 기대치 없음 → 변경 없음

    sig = infer_simple_signals(snippet)
    # 외부 신호가 있으면 우선
    if intensity_profile and "mean" in (intensity_profile or {}):
        try:
            sig["amplified"] = bool(float(intensity_profile.get("mean", 0.0)) >= 0.6)
        except Exception:
            pass
    if transitions and "count" in (transitions or {}):
        try:
            # 전이 많으면 amplified 힌트로 본다
            sig["amplified"] = sig["amplified"] or int(transitions.get("count", 0)) >= 2
        except Exception:
            pass

    penalty = 1.0
    notes = []
    if exp["valence"] != 0 and sig["valence"] != 0 and (exp["valence"] != sig["valence"]):
        penalty *= 0.6; notes.append("VALENCE_MISMATCH")
    if exp["amplified"] and not sig["amplified"]:
        penalty *= 0.7; notes.append("AMPLITUDE_MISMATCH")
    if not exp["amplified"] and sig["amplified"]:
        penalty *= 0.8; notes.append("OVER_EXCITED")

    return max(penalty, 0.4), (";".join(notes) if notes else None)


# =============================================================================
# ★★★ 범용 유틸리티: 감정 분석 결과에서 상황 역추론 ★★★
# 다른 모듈(psychological_analyzer, complex_analyzer 등)에서도 재사용 가능
# =============================================================================
def infer_situations_from_emotion_data(
    emotion_results: Dict[str, Any],
    emotions_data: Dict[str, Any],
    *,
    min_confidence: float = 0.3,
    max_situations: int = 5,
    boost_by_intensity: bool = True,
) -> List[Dict[str, Any]]:
    """
    ★★★ 핵심 알고리즘: 감정 분석 결과에서 연관 상황을 역추론 ★★★
    
    원리:
    1. emotion_results에서 감지된 감정(main_distribution, detected_emotions 등) 추출
    2. EMOTIONS.json에서 해당 감정의 context_patterns.situations 정보 조회
    3. 감정 신뢰도에 기반하여 상황 신뢰도 계산
    4. 상황 목록 반환 (중복 제거, 신뢰도 정렬)
    
    Args:
        emotion_results: 감정 분석 결과 (main_distribution, emotion_intensity 등)
        emotions_data: EMOTIONS.json 데이터
        min_confidence: 최소 신뢰도 임계값
        max_situations: 최대 반환 상황 수
        boost_by_intensity: 감정 강도로 신뢰도 부스트 여부
    
    Returns:
        List[Dict]: 추론된 상황 목록
        [
            {
                "situation": "서비스 불만",
                "primary_emotion": "노",
                "sub_emotion": "분노-억울함",
                "confidence": 0.72,
                "inference_source": "emotion_distribution",
                "description": "...",
            },
            ...
        ]
    """
    if not emotion_results or not emotions_data:
        return []
    
    inferred_situations: List[Dict[str, Any]] = []
    situation_scores: Dict[str, Dict[str, Any]] = {}  # situation_name -> {score, sources}
    
    # 1) 감정 분포 추출 (다양한 키 지원)
    main_dist = (
        emotion_results.get("main_distribution") or
        emotion_results.get("emotion_distribution") or
        emotion_results.get("distribution") or
        {}
    )
    
    # 감정 강도 정보 (있으면 부스트에 활용)
    emotion_intensity = emotion_results.get("emotion_intensity") or {}
    
    # 감지된 세부 감정 목록
    detected_emotions = (
        emotion_results.get("detected_emotions") or
        emotion_results.get("emotions") or
        []
    )
    
    # 2) EMOTIONS.json 순회하며 감정-상황 매핑
    for primary_emo, emo_info in emotions_data.items():
        if not isinstance(emo_info, dict):
            continue
        
        # 주 감정 점수 확인
        primary_score = float(main_dist.get(primary_emo, 0.0))
        if primary_score < 0.1:
            continue  # 낮은 점수는 스킵
        
        # 세부 감정 순회
        sub_emotions = emo_info.get("sub_emotions") or {}
        for sub_emo_name, sub_info in sub_emotions.items():
            if not isinstance(sub_info, dict):
                continue
            
            # context_patterns에서 situations 추출
            context_patterns = sub_info.get("context_patterns") or {}
            situations = context_patterns.get("situations") or {}
            
            for sit_name, sit_data in situations.items():
                if not isinstance(sit_data, dict):
                    continue
                
                # 상황 신뢰도 계산
                # 기본: 주 감정 점수 * 0.7 + 고정 0.2
                base_conf = primary_score * 0.7 + 0.2
                
                # 강도 부스트 (해당 감정의 강도가 high면 부스트)
                if boost_by_intensity:
                    intensity_info = emotion_intensity.get(primary_emo) or {}
                    if isinstance(intensity_info, dict):
                        level = intensity_info.get("level", "medium")
                        if level == "high":
                            base_conf *= 1.15
                        elif level == "low":
                            base_conf *= 0.85
                
                # 상황 데이터의 intensity 가중치
                sit_intensity = sit_data.get("intensity", "medium")
                intensity_weight = {"low": 0.8, "medium": 1.0, "high": 1.1}.get(sit_intensity, 1.0)
                
                final_conf = min(0.95, base_conf * intensity_weight)
                
                if final_conf < min_confidence:
                    continue
                
                # 중복 처리: 동일 상황이면 더 높은 신뢰도 유지
                sit_key = sit_name
                # ★★★ 중요: situation_id 추가 (다른 함수와의 호환성) ★★★
                situation_id = f"{primary_emo}-{sub_emo_name}:{sit_name}"
                
                if sit_key in situation_scores:
                    if final_conf > situation_scores[sit_key]["confidence"]:
                        situation_scores[sit_key] = {
                            "situation_id": situation_id,
                            "situation": sit_name,
                            "situation_name": sit_name,
                            "primary_emotion": primary_emo,
                            "sub_emotion_name": sub_emo_name,
                            "sub_emotion": f"{primary_emo}-{sub_emo_name}",
                            "confidence": round(final_conf, 3),
                            "inference_source": "emotion_distribution",
                            "description": sit_data.get("description", ""),
                            "keywords": sit_data.get("keywords", [])[:3],
                        }
                else:
                    situation_scores[sit_key] = {
                        "situation_id": situation_id,
                        "situation": sit_name,
                        "situation_name": sit_name,
                        "primary_emotion": primary_emo,
                        "sub_emotion_name": sub_emo_name,
                        "sub_emotion": f"{primary_emo}-{sub_emo_name}",
                        "confidence": round(final_conf, 3),
                        "inference_source": "emotion_distribution",
                        "description": sit_data.get("description", ""),
                        "keywords": sit_data.get("keywords", [])[:3],
                    }
    
    # 3) 신뢰도 순 정렬 및 상위 N개 반환
    sorted_situations = sorted(
        situation_scores.values(),
        key=lambda x: x["confidence"],
        reverse=True
    )[:max_situations]
    
    return sorted_situations


def merge_situation_results(
    keyword_matches: List[Dict[str, Any]],
    inferred_situations: List[Dict[str, Any]],
    *,
    keyword_weight: float = 0.6,
    inference_weight: float = 0.4,
    max_results: int = 5,
) -> List[Dict[str, Any]]:
    """
    키워드 매칭 결과와 감정 기반 추론 결과를 병합
    
    Args:
        keyword_matches: 키워드 매칭 결과
        inferred_situations: 감정 기반 추론 결과
        keyword_weight: 키워드 매칭 가중치
        inference_weight: 감정 추론 가중치
        max_results: 최대 결과 수
    
    Returns:
        병합된 상황 목록
    """
    merged: Dict[str, Dict[str, Any]] = {}
    
    # 키워드 매칭 결과 추가
    for item in (keyword_matches or []):
        sit_name = item.get("situation") or item.get("situation_name", "")
        if not sit_name:
            continue
        conf = float(item.get("confidence", 0.5)) * keyword_weight
        merged[sit_name] = {
            **item,
            "situation": sit_name,
            "confidence": conf,
            "source": "keyword_match",
        }
    
    # 감정 추론 결과 병합
    for item in (inferred_situations or []):
        sit_name = item.get("situation", "")
        if not sit_name:
            continue
        conf = float(item.get("confidence", 0.5)) * inference_weight
        
        if sit_name in merged:
            # 이미 있으면 신뢰도 합산 (시너지 효과)
            merged[sit_name]["confidence"] += conf
            merged[sit_name]["source"] = "keyword+inference"
        else:
            merged[sit_name] = {
                **item,
                "situation": sit_name,
                "confidence": conf,
                "source": "emotion_inference",
            }
    
    # 정렬 및 반환
    sorted_merged = sorted(
        merged.values(),
        key=lambda x: x["confidence"],
        reverse=True
    )[:max_results]
    
    # 신뢰도 정규화 (0~1 범위)
    for item in sorted_merged:
        item["confidence"] = min(0.95, round(item["confidence"], 3))
    
    return sorted_merged


def analyze_situations(text: str, emotions_config: dict = None,
                       intensity_profile: dict = None, transitions: dict = None,
                       mode: str = None):
    """
    Multi-level tolerant situation matcher with Junk-Guard and conflict down-weighting.
    Returns a schema-stable dict: {status, matches[], confidence, metrics{...}}.
    mode: "STRICT" | "TOLERANT" (기본 TOLERANT)
    """
    mode = (mode or os.environ.get("SITUATION_MODE") or "TOLERANT").upper()
    metrics = copy.deepcopy(DEFAULT_METRICS)

    # ENV 튜닝 가능한 Junk-Guard로 일원화
    is_junk, why = _sit_is_junk_text(text)
    if is_junk:
        out = {
            "status": "junk_input",
            "confidence": 0.2,
            "matches": [],
            "metrics": {**metrics, "warnings": metrics["warnings"] + [f"JUNK_INPUT:{why}"]}
        }
        return out

    # 1) 사전/별칭 준비 (없으면 Soft-fail)
    try:
        situation_dict = (emotions_config or {}).get("situations", {}).get("lexicon", {})
        alias_from_cfg = load_aliases_from_emotions(emotions_config or {}) if emotions_config else {}
    except Exception as e:
        out = {
            "status": "missing_lexicon",
            "confidence": 0.0,
            "matches": [],
            "metrics": {**metrics, "warnings": metrics["warnings"] + [f"MISSING_LEXICON:{e}"]}
        }
        return out

    aliases = FALLBACK_SITUATION_ALIASES.copy()
    if alias_from_cfg:
        for k, v in alias_from_cfg.items():
            aliases.setdefault(k, set()).update(v)
        metrics["fallbacks"]["alias"] = True

    t_norm = nfkc_norm(text)
    matches = []

    # 2) 멀티레벨 매칭
    # [OPTIMIZATION] 텍스트 전처리 1회 수행으로 반복 연산 제거
    # 텍스트 길이/유니크 비율 등은 이미 위에서 계산됨.
    # nfkc_norm은 t_norm에 이미 있음.
    
    # Bigram 집합 미리 생성 (L3용)
    text_bigrams = bigrams(t_norm) if len(t_norm) > 1 else set()
    
    def add_match(name, span, level, score, ev):
        matches.append({"situation": name, "span": span, "level": level, "score": float(score), "evidence": ev})
        metrics["used_levels"].append(level)

    for name, patterns in (situation_dict or {}).items():
        # patterns: ["정확키", ...] 가정. 없으면 name 자체로 시도.
        keys = patterns if isinstance(patterns, (list, tuple)) else [name]
        found = False

        # L0/L1: 정밀/정규화 포함 스캔
        # [OPTIMIZATION] L0(raw text)와 L1(normalized text) 분리 최적화
        
        # L0: Raw text search
        for key in keys:
            if key in text:
                add_match(name, None, "L0", 1.0, [key]); found = True; break
        if found: continue

        # L1: Normalized text search (t_norm 재사용)
        for key in keys:
            # key 정규화는 캐싱되거나 미리 되어있어야 함 (여기서는 런타임 비용 감수하되 text 정규화 비용은 제거)
            k_norm = nfkc_norm(key) 
            if k_norm in t_norm:
                add_match(name, None, "L1", 0.9, [key]); found = True; break
        if found: continue

        # L2: 별칭/동의어 (L1과 동일 로직)
        if name in aliases:
            for alias in aliases[name]:
                a_norm = nfkc_norm(alias)
                if a_norm in t_norm:
                    add_match(name, None, "L2", 0.8, [alias]); found = True; break
        if found: continue

        # L3: bi-gram Jaccard (미리 계산된 text_bigrams 사용)
        # [OPTIMIZATION] 키워드 정규화 및 Bigram 계산 최소화
        for key in keys[:3]:  # 시간 가드
            k_norm = nfkc_norm(key)
            if len(k_norm) < 2: continue
            
            # 키워드 bigram은 캐싱되면 좋겠지만, 여기서는 즉시 계산
            k_bigrams = {k_norm[i:i+2] for i in range(len(k_norm)-1)}
            
            # Jaccard 수동 계산 (함수 호출 오버헤드 제거)
            if not k_bigrams: continue
            inter = len(text_bigrams & k_bigrams)
            union = len(text_bigrams | k_bigrams)
            j = inter / (union or 1)
            
            if j >= 0.45:
                sc = min(0.6 + (j - 0.45) * 1.0, 0.8)
                add_match(name, None, "L3", sc, [key]); found = True; break
        if found: continue

        # L4: 편집거리 (문자열 길이 제한)
        # [OPTIMIZATION] 길이 차이로 빠른 기각 (Early Exit)
        t_len = len(t_norm)
        for key in keys[:2]:  # 시간 가드
            if len(key) > 24: continue
            
            k_norm = nfkc_norm(key)
            k_len = len(k_norm)
            
            # 길이 차이가 2 초과면 편집거리 2 이하일 수 없음
            if abs(t_len - k_len) > 2: continue
            
            d = edit_distance_le2(t_norm, k_norm, 2)
            if d <= 2:
                add_match(name, None, "L4", 0.5, [key]); found = True; break

    # 3) NMS 유사 중복 억제(상황 중복 시 높은 score 우선)
    matches.sort(key=lambda m: m["score"], reverse=True)
    uniq = {}
    dedup = []
    for m in matches:
        n = m["situation"]
        if n not in uniq:
            uniq[n] = True
            dedup.append(m)
    matches = dedup

    # 4) 상황-감정 연동 검증 및 가중 하향 (기본 OFF; 오케스트레이터가 책임)
    penalties = []
    for m in matches:
        if int(os.environ.get("SIT_USE_SIMPLE_RECONCILE", "0")):
            p, note = reconcile_with_emotions(m["situation"], text, intensity_profile, transitions)
            if note:
                metrics["warnings"].append(f"REWEIGHT:{m['situation']}:{note}")
            m["score"] *= float(p)
            penalties.append(p)

    # 5) confidence/coverage 산출
    if matches:
        coverage = min(1.0, len(matches) / max(1, len(situation_dict) if isinstance(situation_dict, dict) else 1))
        metrics["coverage"] = float(coverage)
        try:
            metrics["jaccard_mean"] = round(sum(max(0.0, (m["score"] - 0.5) / 0.5) for m in matches) / len(matches), 3)
        except Exception:
            metrics["jaccard_mean"] = 0.0
        confidence = min(1.0, 0.5 + 0.5 * float(metrics["jaccard_mean"]))
        status = "ok"
        if any(float(p) < 0.8 for p in penalties):
            status = "mismatch_adjusted"
        out = {"status": status, "confidence": float(confidence), "matches": matches, "metrics": metrics}
    else:
        out = {"status": "no_match", "confidence": 0.3, "matches": [], "metrics": metrics}

    return out

# -----------------------------------------------------------------------------
# 1) Validation — 스키마 불변 전제의 관대한 검사기 (role readiness 포함, drop-in)
#    * 루트 + sub_emotions 모두 순회하여 coverage/ready%를 현실적으로 계산
# -----------------------------------------------------------------------------
def validate_emotion_data(data: Any) -> Dict[str, Any]:
    """
    상황 컨텍스트 오케스트레이터용 라벨링 데이터 검증기.
    - 루트가 list/dict 모두 허용(자동 정규화)
    - 스키마 동의어(snake/camel/국문) 관대 대응
    - (중요) 루트 + sub_emotions 모두 순회하여 상황/전이/강도/단계/시공간 커버리지 집계
    - 역할 적합도(readiness) % 산출
    반환:
      {
        "ok": bool,
        "errors": [str,...],
        "warnings": [str,...],
        "summary": {...},
        "samples": {...},
        "coverage": {...},
        "readiness": {"score": float, "percent": int, "weights": {...}},
        "hints": [str,...]
      }
    """
    # ---------- 내장 헬퍼 ----------
    def _normalize_root(d: Any) -> Dict[str, Any]:
        if isinstance(d, dict):
            return d
        if isinstance(d, list):
            out = {}
            for i, item in enumerate(d):
                if isinstance(item, dict):
                    key = (
                        str(item.get("id"))
                        or str(item.get("name"))
                        or str(item.get("primary"))
                        or str(item.get("label"))
                        or str(item.get("metadata", {}).get("name"))
                        or f"node_{i}"
                    )
                    out[key] = item
            return out
        return {}

    def _iter_phrases(node: Any) -> Iterable[str]:
        if not node:
            return
        if isinstance(node, str):
            s = node.strip()
            if s:
                yield s
            return
        if isinstance(node, (list, tuple, set)):
            for v in node:
                if isinstance(v, str) and v.strip():
                    yield v.strip()
        elif isinstance(node, dict):
            for k in ("text", "phrase", "value", "token"):
                v = node.get(k)
                if isinstance(v, str) and v.strip():
                    yield v.strip()

    def _k(node: Dict[str, Any], *alts: str) -> Any:
        if not isinstance(node, dict):
            return None
        lower = {str(kk).lower(): kk for kk in node.keys()}
        for nm in alts:
            if nm.lower() in lower:
                return node[lower[nm.lower()]]
        return None

    def _dedup_keep_order(xs: Iterable[str]) -> List[str]:
        seen, out = set(), []
        for s in xs:
            s2 = str(s).strip()
            if s2 and s2 not in seen:
                seen.add(s2)
                out.append(s2)
        return out

    def _basic_ok_token(t: str) -> bool:
        if not t: return False
        t = str(t).strip()
        if len(t) < 2: return False
        import re as _re
        if _re.fullmatch(r"[0-9]+", t): return False
        if _re.fullmatch(r"[\W_]+", t): return False
        return True

    def _norm_stage(name: Any) -> str:
        n = str(name).strip().lower() if name else ""
        alias = {
            "onset": "trigger", "start": "trigger", "trigger": "trigger",
            "build": "development", "build-up": "development", "development": "development",
            "climax": "peak", "apex": "peak", "peak": "peak",
            "aftermath": "aftermath", "resolution": "aftermath", "cooldown": "aftermath"
        }
        return alias.get(n, n or "trigger")

    # ---------- 정규화 ----------
    root = _normalize_root(data)
    errors: List[str] = []
    warnings: List[str] = []
    hints: List[str] = []

    if not root:
        return {
            "ok": False,
            "errors": ["root is empty or unsupported type; expected dict/list of nodes"],
            "warnings": [],
            "summary": {},
            "samples": {},
            "coverage": {},
            "readiness": {"score": 0.0, "percent": 0, "weights": {}},
            "hints": ["데이터 루트에 dict(list-of-dict 허용)가 필요합니다."]
        }

    # ---------- 통계/커버리지 집계자 ----------
    n_nodes = 0
    n_situations = 0
    n_triggers = 0
    n_ctx = 0
    n_progress = 0
    n_transitions = 0
    n_keywords = 0
    n_synergy = 0
    n_conflicts = 0

    nodes_with_situations = 0
    nodes_with_transitions = 0
    nodes_with_intensity_examples = 0
    nodes_with_progression = 0
    nodes_with_spatiotemporal = 0

    sample_first = {"situation": None, "triggers": None, "context": None}
    known_ids: Set[str] = set()
    known_names: Set[str] = set()
    dup_ids: Set[str] = set()
    dup_names: Set[str] = set()

    import re as _re

    # ---- 루트 + sub_emotions 재귀 집계 함수 ----
    def _accumulate_from_node(key: str, node: Dict[str, Any], *, count_df: bool = True) -> None:
        nonlocal n_nodes, n_situations, n_triggers, n_ctx, n_progress, n_transitions
        nonlocal n_keywords, n_synergy, n_conflicts
        nonlocal nodes_with_situations, nodes_with_transitions, nodes_with_intensity_examples
        nonlocal nodes_with_progression, nodes_with_spatiotemporal
        nonlocal sample_first, known_ids, known_names

        # 대표 이름/ID
        nid = str(node.get("id") or key).strip()
        nname = str(node.get("name") or node.get("label") or node.get("primary") or key).strip()
        if nid in known_ids:  dup_ids.add(nid)
        if nname in known_names: dup_names.add(nname)
        known_ids.add(nid)
        known_names.add(nname)

        # intensity_levels
        prof = _k(node, "emotion_profile", "emotionProfile", "profile")
        levels = _k(prof, "intensity_levels", "intensityLevels", "levels") if isinstance(prof, dict) else None
        node_has_intensity = False
        if isinstance(levels, dict):
            if "intensity_examples" in levels and isinstance(levels["intensity_examples"], dict):
                node_has_intensity = any(bool(list(_iter_phrases(levels["intensity_examples"].get(lv)))) for lv in ("low","medium","high"))
            else:
                for lv in ("low","medium","high"):
                    block = levels.get(lv)
                    if isinstance(block, dict) and list(_iter_phrases(block.get("intensity_examples"))):
                        node_has_intensity = True; break
        if node_has_intensity:
            nodes_with_intensity_examples += 1

        # context_patterns.situations
        ctxp = _k(node, "context_patterns", "contextPatterns", "contexts", "situationPatterns")
        sits = _k(ctxp, "situations", "scenarios", "cases") if isinstance(ctxp, dict) else None
        it = list(sits.values()) if isinstance(sits, dict) else (sits if isinstance(sits, list) else [])
        if it:
            nodes_with_situations += 1
            n_situations += len(it)
            if sample_first["situation"] is None:
                sample_first["situation"] = str(nname)
        node_has_prog = False
        for s in it or []:
            if not isinstance(s, dict): continue
            ks = list(_iter_phrases(_k(s, "keywords", "phrases", "terms", "tokens")))
            vs = list(_iter_phrases(_k(s, "variations", "variants", "aliases", "synonyms")))
            n_keywords += len(ks) + len(vs)
            exs = list(_iter_phrases(_k(s, "examples", "samples")))
            n_ctx += len(exs)
            prog = _k(s, "emotion_progression", "progression", "flow")
            if isinstance(prog, dict):
                for stg, ex in prog.items():
                    n_progress += len(list(_iter_phrases(ex)))
                    node_has_prog = True
        if node_has_prog:
            nodes_with_progression += 1

        # transitions
        trans_root = _k(node, "emotion_transitions", "emotionTransitions", "transitions")
        patterns = trans_root if isinstance(trans_root, list) else (_k(trans_root, "patterns", "rules", "list") if isinstance(trans_root, dict) else None)
        if isinstance(patterns, list) and patterns:
            nodes_with_transitions += 1
            for t in patterns:
                if not isinstance(t, dict): continue
                t_trigs = list(_iter_phrases(_k(t, "triggers", "cues", "signals", "markers")))
                n_triggers += len(t_trigs)
                n_transitions += 1
                if sample_first["triggers"] is None and t_trigs:
                    sample_first["triggers"] = t_trigs[:5]

        # 스파시오템포럴 힌트
        spatiotemporal_hit = False
        for cand in [node, prof or {}, ctxp or {}, trans_root or {}]:
            if not isinstance(cand, dict): continue
            st_keys = ("spatiotemporal", "temporal_markers", "temporal_aspects", "location_markers",
                       "time_markers", "location", "time", "date", "duration", "sequence")
            for kx in cand.keys():
                if any(s in str(kx).lower() for s in st_keys):
                    spatiotemporal_hit = True; break
            if spatiotemporal_hit: break
        if spatiotemporal_hit:
            nodes_with_spatiotemporal += 1

        # synergy/conflicts
        sync = _k(node, "synergy", "synergy_with", "synergyWith", "cooperate_with")
        conf = _k(node, "conflicts", "conflicts_with", "conflictsWith", "interfere_with")
        def _as_names_list(obj: Any) -> List[str]:
            if isinstance(obj, dict):
                return _dedup_keep_order(obj.keys())
            return _dedup_keep_order(list(_iter_phrases(obj)))
        if isinstance(sync, (list, dict)): n_synergy += len(_as_names_list(sync))
        if isinstance(conf, (list, dict)): n_conflicts += len(_as_names_list(conf))

        # unknown stage 경고
        if it:
            for s in it:
                prog = _k(s, "emotion_progression", "progression", "flow")
                if isinstance(prog, dict):
                    unknown = [st for st in prog.keys() if _norm_stage(st) not in ("trigger","development","peak","aftermath")]
                    if unknown:
                        warnings.append(f"{key}: unknown progression stage(s) {unknown}")

    # ---- 루트/서브 감정 순회 ----
    for key, node in root.items():
        if not isinstance(node, dict):
            warnings.append(f"{key}: node is not dict; skipped")
            continue
        n_nodes += 1
        _accumulate_from_node(key, node, count_df=True)

        # sub_emotions도 커버리지 포함
        subs = node.get("sub_emotions", {}) or (node.get("emotion_profile", {}) or {}).get("sub_emotions", {})
        if isinstance(subs, dict):
            for sub_name, sub_node in subs.items():
                if isinstance(sub_node, dict):
                    _accumulate_from_node(f"{key}.{sub_name}", sub_node, count_df=False)

    # 품질 경고
    if n_situations == 0 and n_keywords == 0 and n_ctx == 0:
        errors.append("no situations/context keywords found (context_patterns.situations.* missing or empty)")
    if n_triggers == 0 and n_transitions > 0:
        warnings.append("transition patterns exist but have no triggers")

    # 나쁜 토큰 샘플
    sample_kws = []
    for key, node in root.items():
        ctxp = _k(node, "context_patterns", "contextPatterns", "contexts", "situationPatterns")
        sits = _k(ctxp, "situations", "scenarios", "cases") if isinstance(ctxp, dict) else None
        it = list(sits.values()) if isinstance(sits, dict) else (sits if isinstance(sits, list) else [])
        for s in it or []:
            if isinstance(s, dict):
                sample_kws.extend(list(_iter_phrases(_k(s, "keywords", "phrases", "terms", "tokens"))))
        # sub_emotions의 situations도 샘플 포함
        subs = node.get("sub_emotions", {}) or (node.get("emotion_profile", {}) or {}).get("sub_emotions", {})
        if isinstance(subs, dict):
            for _sn, _sub in subs.items():
                ctxp2 = _k(_sub, "context_patterns", "contextPatterns", "contexts", "situationPatterns")
                sits2 = _k(ctxp2, "situations", "scenarios", "cases") if isinstance(ctxp2, dict) else None
                it2 = list(sits2.values()) if isinstance(sits2, dict) else (sits2 if isinstance(sits2, list) else [])
                for s2 in it2 or []:
                    if isinstance(s2, dict):
                        sample_kws.extend(list(_iter_phrases(_k(s2, "keywords", "phrases", "terms", "tokens"))))
    bad = [x for x in _dedup_keep_order(sample_kws) if not _basic_ok_token(x)]
    if bad:
        warnings.append(f"too generic/invalid tokens detected (sample): {bad[:10]}")

    # 중복 이름/ID
    if dup_ids:
        warnings.append(f"duplicated ids: {sorted(list(dup_ids))[:10]}")
    if dup_names:
        warnings.append(f"duplicated names: {sorted(list(dup_names))[:10]}")

    # coverage/summary
    coverage = {
        "nodes": n_nodes,
        "nodes_with_situations": nodes_with_situations,
        "nodes_with_transitions": nodes_with_transitions,
        "nodes_with_intensity_examples": nodes_with_intensity_examples,
        "nodes_with_progression": nodes_with_progression,
        "nodes_with_spatiotemporal": nodes_with_spatiotemporal,
        "ratios": {
            "situations": (nodes_with_situations / n_nodes) if n_nodes else 0.0,
            "transitions": (nodes_with_transitions / n_nodes) if n_nodes else 0.0,
            "intensity": (nodes_with_intensity_examples / n_nodes) if n_nodes else 0.0,
            "progression": (nodes_with_progression / n_nodes) if n_nodes else 0.0,
            "spatiotemporal": (nodes_with_spatiotemporal / n_nodes) if n_nodes else 0.0,
        }
    }
    summary = {
        "nodes": n_nodes,
        "situations": n_situations,
        "context_examples": n_ctx,
        "progression_examples": n_progress,
        "transitions": n_transitions,
        "triggers": n_triggers,
        "keywords": n_keywords,
        "synergy_refs": n_synergy,
        "conflict_refs": n_conflicts,
    }

    # readiness (역할 적합도)
    W = {"situations": 0.25, "transitions": 0.25, "intensity": 0.20, "progression": 0.15, "spatiotemporal": 0.15}
    r = coverage["ratios"]
    def _score(x: float) -> float:
        return min(1.0, 0.3 + 0.7 * x) if x > 0.0 else 0.0
    score = (W["situations"]     * _score(r["situations"]) +
             W["transitions"]    * _score(r["transitions"]) +
             W["intensity"]      * _score(r["intensity"]) +
             W["progression"]    * _score(r["progression"]) +
             W["spatiotemporal"] * _score(r["spatiotemporal"]))
    readiness = {"score": round(score, 3), "percent": int(round(score * 100)), "weights": W}

    # 힌트
    if readiness["percent"] < 90:
        if r["situations"] == 0.0:   hints.append("context_patterns.situations.* 를 최소 1개 이상 채워 상황 패턴을 제공하세요.")
        if r["transitions"] == 0.0:  hints.append("emotion_transitions.patterns[*].triggers 를 채워 전이 트리거를 제공하세요.")
        if r["intensity"] == 0.0:    hints.append("emotion_profile.intensity_levels 에 intensity_examples 를 채우세요 (low/medium/high).")
        if r["progression"] == 0.0:  hints.append("situations[*].emotion_progression 에 trigger/development/peak/aftermath 예시를 추가하세요.")
        if r["spatiotemporal"] == 0.0: hints.append("temporal_aspects/location/time/duration/sequence 등 시공간 표지를 추가하세요.")
        if dup_ids or dup_names:     hints.append("중복 id/name 을 해소해 참조 일관성을 높이세요.")
        if n_triggers == 0 and n_transitions > 0: hints.append("전이 패턴에 triggers 가 없습니다. 최소 1개 이상 등록하세요.")
        if bad:                      hints.append("너무 짧거나 숫자/기호 위주의 키워드를 정제하세요.")

    ok = (len(errors) == 0)
    return {
        "ok": ok,
        "errors": errors,
        "warnings": warnings,
        "summary": summary,
        "samples": {k: v for k, v in sample_first.items() if v},
        "coverage": coverage,
        "readiness": readiness,
        "hints": hints[:5],
    }



# ---------------------------------------------------------------------
# RegexShardMatcher (portable, dependency-free)
#   - ignore_space=True  : 공백 무시 매칭 (한글 합성/띄어쓰기 변동에 강건)
#   - boundary=False     : 경계 미사용(부분일치). 필요시 단어경계로 확장 가능
#   - shard_size=512     : 라벨이 많을 때 OR-정규식 샤딩
# find_all(text) -> List[str]  : 원문(비정규화) 구절 리스트 반환(중복 제거)
# ---------------------------------------------------------------------
import unicodedata as _ud
import re as _re

def _nfkc_no_space(s: str) -> str:
    try:
        s = _ud.normalize("NFKC", s or "")
    except Exception:
        s = s or ""
    return _re.sub(r"\s+", "", s)

class RegexShardMatcher:
    def __init__(
        self,
        phrases: List[str],
        shard_size: int = 512,
        boundary: bool = False,
        ignore_space: bool = True,
    ):
        self._orig: List[str] = [p for p in (phrases or []) if isinstance(p, str) and p.strip()]
        self._ignore_space = bool(ignore_space)
        self._norm_map: Dict[str, str] = {}   # norm_phrase -> original phrase
        self._rx_list: List[Any] = []

        if not self._orig:
            return

        # 정규화(공백 제거+NFKC)
        tokens = []
        for p in self._orig:
            key = _nfkc_no_space(p) if self._ignore_space else p
            if key:
                # 같은 정규화 키가 여러 원문을 가리킬 수 있으나, 우선 1:1 맵만 유지(중복 제거 목적)
                self._norm_map.setdefault(key, p)
                tokens.append(key)

        # 샤딩하여 OR-정규식 컴파일
        # boundary를 쓰려면 (\b…) 같은 경계로 감싼 패턴을 생성하는 로직을 여기에 추가
        for i in range(0, len(tokens), max(1, int(shard_size))):
            chunk = tokens[i : i + shard_size]
            # 공백무시 비교를 위해 텍스트도 공백 제거 후 검색하므로, 여기선 chunk를 그대로 escape
            pat = "|".join(_re.escape(t) for t in chunk)
            self._rx_list.append(_re.compile(pat))

    def find_all(self, text: str) -> List[str]:
        if not self._rx_list:
            return []
        t = _nfkc_no_space(text) if self._ignore_space else (text or "")
        hits: List[str] = []
        seen = set()
        for rx in self._rx_list:
            for m in rx.finditer(t):
                norm_hit = m.group(0)
                orig = self._norm_map.get(norm_hit)
                if not orig:
                    # 혹시 정규화 키가 바로 매핑 안되면, 동등성 검사(희귀 케이스)
                    orig = next((v for k, v in self._norm_map.items() if k == norm_hit), None)
                if orig and orig not in seen:
                    seen.add(orig)
                    hits.append(orig)
        return hits

# LRU 캐시: RegexShardMatcher 재사용으로 생성 비용 최소화
@lru_cache(maxsize=100_000)
def _rx_cache(key: Tuple[Any, ...], phrases: Tuple[str, ...]) -> RegexShardMatcher:
    return RegexShardMatcher(list(phrases or []), shard_size=256, boundary=False, ignore_space=True)


# === [PATCH A] Situation Analyzer: Junk-Guard + thresholds (non-breaking) ===
def _sit_junk_cfg():
    def _getf(name, default):
        try:
            return float(os.environ.get(name, str(default)))
        except Exception:
            return float(default)
    def _geti(name, default):
        try:
            return int(os.environ.get(name, str(default)))
        except Exception:
            return int(default)
    return {
        "PUNCT": _getf("SIT_JUNK_PUNCT", 0.55),   # 구두점 비율 상한
        "UNIQ":  _getf("SIT_JUNK_UNIQ", 0.15),    # 유니크 문자 비율 하한
        "ENT":   _getf("SIT_JUNK_ENT", 1.20),     # 간이 엔트로피 하한
        "RUN":   _geti("SIT_JUNK_RUN", 6),        # 동일문자 연속 길이 상한
    }

def _sit_is_junk_text(s: str):
    """NFKC·공백제거 후 문자군/엔트로피/반복런으로 난수·무의미 텍스트 감지."""
    s = s or ""
    s_norm = (_nfkc_no_space(s) or "").lower()
    n = len(s_norm)
    if n < 3:
        return True, "TOO_SHORT"
    letters = sum(ch.isalpha() for ch in s_norm)
    puncts  = sum(not ch.isalnum() for ch in s_norm)
    uniq    = len(set(s_norm))
    from collections import Counter as _C
    c = _C(s_norm)
    ent = -sum((cnt/n)*math.log((cnt/n)+1e-12) for cnt in c.values())
    # max-run
    run = max_run = 1; prev = None
    for ch in s_norm:
        if ch == prev:
            run += 1; max_run = max(max_run, run)
        else:
            run = 1; prev = ch
    cfg = _sit_junk_cfg()
    if (puncts/max(n,1)) > cfg["PUNCT"] or (uniq/max(n,1)) < cfg["UNIQ"] or ent < cfg["ENT"] or max_run >= cfg["RUN"]:
        return True, f"NOISY(punct={puncts/n:.2f},uniq={uniq/n:.2f},H={ent:.2f},run={max_run})"
    return False, ""


# =============================================================================
# Data Classes  —  evidence-aware containers for situation orchestration
# =============================================================================

# --- local helpers (이 블록 내부에서만 사용; 외부 유틸 없을 때도 안전 동작) ---
def _sc_clip01(x: float) -> float:
    try:
        return 0.0 if x is None else (1.0 if x > 1.0 else (0.0 if x < 0.0 else float(x)))
    except Exception:
        return 0.0

def _sc_norm_intensity(v: Any, default: str = "medium") -> str:
    s = str(v).strip().lower() if v is not None else ""
    if s in {"low", "lo", "약함", "weak"}: return "low"
    if s in {"medium", "med", "mid", "중간"}: return "medium"
    if s in {"high", "hi", "강함", "strong"}: return "high"
    return default

def _sc_dedup_str_list(xs: Any) -> List[str]:
    out, seen = [], set()
    if not xs: return out
    for t in xs:
        try:
            v = str(t).strip()
        except Exception:
            continue
        if v and v not in seen:
            seen.add(v); out.append(v)
    return out

def _sc_stage_norm(name: Any) -> str:
    n = str(name).strip().lower() if name else ""
    alias = {
        "onset": "trigger", "start": "trigger", "trigger": "trigger",
        "build": "development", "build-up": "development", "development": "development",
        "climax": "peak", "apex": "peak", "peak": "peak",
        "aftermath": "aftermath", "resolution": "aftermath", "cooldown": "aftermath"
    }
    return alias.get(n, n or "trigger")

# -------------------------------- Base mixin --------------------------------
@dataclass(slots=True)
class _SerializableMixin:
    def to_dict(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__.keys()}  # type: ignore


# ---------------------------- SituationContext -------------------------------
@dataclass(slots=True)
class SituationContext(_SerializableMixin):
    """
    상황 컨텍스트(라벨 노드 매핑 + 런타임 매칭 결과)를 담는 핵심 컨테이너.
    - 라벨 스키마 동의어를 관대하게 흡수하고, 누락/중복을 보정.
    - 증거(evidence)와 트리거/매치 스팬을 함께 집계해 상위 오케스트레이터가 바로 사용 가능.
    """
    situation_id: str
    core_concept: str
    keywords: List[str]
    variations: List[str]
    emotion_links: Dict[str, float]
    intensity: str = "medium"
    context_patterns: Dict[str, Any] = field(default_factory=dict)
    temporal_aspects: Optional[Dict[str, Any]] = None

    # runtime fields
    triggers: List[str] = field(default_factory=list)                 # 매칭된 트리거(정규화)
    matches: List[Dict[str, Any]] = field(default_factory=list)       # {"type","text","start","end","weight"}
    evidence: List[Dict[str, Any]] = field(default_factory=list)      # {"type","weight","term"?}
    synergy_with: List[str] = field(default_factory=list)             # 시너지 참조(이름/ID)
    conflicts_with: List[str] = field(default_factory=list)           # 충돌 참조(이름/ID)
    confidence: float = 0.0                                           # 0~1
    source: Optional[Dict[str, Any]] = None                           # 원본 라벨 노드(옵션)

    # derived aggregates (finalize 후 채워짐)
    evidence_totals: Dict[str, float] = field(default_factory=dict)   # 타입별 증거합
    match_score: float = 0.0                                          # 키워드/변형/예시 기반 매치 스코어(0~1)

    def __post_init__(self):
        self.situation_id   = str(self.situation_id).strip()
        self.core_concept   = str(self.core_concept).strip() if self.core_concept else ""
        self.intensity      = _sc_norm_intensity(self.intensity, "medium")
        self.keywords       = _sc_dedup_str_list(self.keywords)
        self.variations     = _sc_dedup_str_list(self.variations)
        self.triggers       = _sc_dedup_str_list(self.triggers)
        self.synergy_with   = _sc_dedup_str_list(self.synergy_with)
        self.conflicts_with = _sc_dedup_str_list(self.conflicts_with)
        # emotion_links 값 클립
        self.emotion_links  = {str(k): float(v) for (k, v) in (self.emotion_links or {}).items()}
        self.emotion_links  = {k: _sc_clip01(v) for k, v in self.emotion_links.items()}
        self.confidence     = _sc_clip01(self.confidence)
        # context_patterns가 None/비 dict인 경우 빈 dict로
        if not isinstance(self.context_patterns, dict):
            self.context_patterns = {}

    # ---- runtime ops ----
    def add_trigger(self, trig: str, weight: float = 0.12) -> None:
        t = str(trig).strip()
        if t and t not in self.triggers:
            self.triggers.append(t)
        self.add_evidence("trigger", weight, term=t)

    def add_keyword_hit(self, term: str, weight: float = 0.08) -> None:
        t = str(term).strip()
        if t:
            self.add_match("keyword", t, None, None, weight)
            self.add_evidence("keyword", weight, term=t)

    def add_variation_hit(self, term: str, weight: float = 0.05) -> None:
        t = str(term).strip()
        if t:
            self.add_match("variation", t, None, None, weight)
            self.add_evidence("variation", weight, term=t)

    def add_example_hit(self, term: str, weight: float = 0.10) -> None:
        t = str(term).strip()
        if t:
            self.add_match("example", t, None, None, weight)
            self.add_evidence("example", weight, term=t)

    def add_match(self, mtype: str, text: str, start: Optional[int] = None, end: Optional[int] = None, weight: float = 0.0) -> None:
        self.matches.append({
            "type": str(mtype),
            "text": str(text),
            "start": None if start is None else int(start),
            "end": None if end is None else int(end),
            "weight": float(weight),
        })

    def add_evidence(self, etype: str, weight: float, term: Optional[str] = None) -> None:
        self.evidence.append({"type": str(etype), "weight": float(weight), **({"term": str(term)} if term else {})})

    def merge_from(self, other: "SituationContext") -> None:
        """같은 상황에 대한 부분 결과 병합(증거/트리거/매치/링크/확신도)."""
        if not isinstance(other, SituationContext):
            return
        self.triggers       = _sc_dedup_str_list(self.triggers + other.triggers)
        self.synergy_with   = _sc_dedup_str_list(self.synergy_with + other.synergy_with)
        self.conflicts_with = _sc_dedup_str_list(self.conflicts_with + other.conflicts_with)
        self.keywords       = _sc_dedup_str_list(self.keywords + other.keywords)
        self.variations     = _sc_dedup_str_list(self.variations + other.variations)
        self.matches.extend(other.matches or [])
        self.evidence.extend(other.evidence or [])
        # emotion_links: 최대값 우선(데이터 라벨 보수적 결합)
        for k, v in (other.emotion_links or {}).items():
            self.emotion_links[k] = max(self.emotion_links.get(k, 0.0), _sc_clip01(v))
        # confidence: 이후 finalize_confidence()에서 재산출

    def finalize_confidence(self,
                            weight_map: Optional[Dict[str, float]] = None,
                            *,
                            cap_per_type: Optional[Dict[str, float]] = None,
                            min_sum: float = 0.0) -> None:
        """
        수집된 evidence/trigger/매치로부터 match_score & confidence 산출.
        - weight_map: 타입별 가중치(기본값은 보수적)
        - cap_per_type: 타입별 최대 기여(lex 과다 억제 등)
        """
        wm = weight_map or {
            "trigger":   0.12,   # add_trigger 기본값과 일치
            "keyword":   0.08,
            "variation": 0.05,
            "example":   0.10,
        }
        caps = cap_per_type or {
            "keyword": 0.40,
            "variation": 0.25,
            "example": 0.40,
            "trigger": 0.36,     # 최대 3회(0.12×3) 정도
        }

        totals: Dict[str, float] = {}
        for ev in (self.evidence or []):
            et = str(ev.get("type"))
            w  = float(ev.get("weight", 0.0))
            if et in wm:
                totals[et] = totals.get(et, 0.0) + w

        # 타입별 cap 적용
        for et, cap in caps.items():
            if et in totals:
                totals[et] = min(totals[et], float(cap))

        # match_score: 키워드/변형/예시 중심 (0~1 스케일)
        lex_sum = totals.get("keyword", 0.0) + totals.get("variation", 0.0) + totals.get("example", 0.0)
        trig_sum = totals.get("trigger", 0.0)
        self.match_score = _sc_clip01(lex_sum + 0.5 * trig_sum)

        # confidence: evidence 합 (최소치 이상이면 sigmoid 없이 선형 클립)
        total = sum(totals.values())
        if total < min_sum:
            self.confidence = _sc_clip01(total)
        else:
            self.confidence = _sc_clip01(total)

        self.evidence_totals = totals

    # ---- factory: 라벨 노드 → SituationContext ----
    @classmethod
    def from_label_node(cls, node_id: str, node: Dict[str, Any]) -> "SituationContext":
        # 동의어 기반 안전 접근
        def k(obj: Dict[str, Any], *alts: str) -> Any:
            lower = {str(kk).lower(): kk for kk in obj.keys()}
            for nm in alts:
                if nm.lower() in lower:
                    return obj[lower[nm.lower()]]
            return None

        core = k(node, "name", "label", "primary") or node_id
        ctxp = k(node, "context_patterns", "contextPatterns", "contexts", "situationPatterns") or {}

        # situations.*.keywords/variations/… 를 평탄화(키워드/변형 풀)
        kws, vars_ = [], []
        if isinstance(ctxp, dict):
            sits = k(ctxp, "situations", "scenarios", "cases")
            it = list(sits.values()) if isinstance(sits, dict) else (sits if isinstance(sits, list) else [])
            for s in it or []:
                if not isinstance(s, dict): continue
                for vv in ("keywords", "phrases", "terms", "tokens"):
                    v = s.get(vv)
                    kws.extend([str(x).strip() for x in v or [] if str(x).strip()])
                for vv in ("variations", "variants", "aliases", "synonyms"):
                    v = s.get(vv)
                    vars_.extend([str(x).strip() for x in v or [] if str(x).strip()])

        # emotion_links
        e_links = k(node, "emotion_links", "emotionLinks", "related_emotions", "relatedEmotions") or {}
        if not isinstance(e_links, dict): e_links = {}

        # intensity (노드/메타데이터에서 탐색)
        meta = node.get("metadata") if isinstance(node.get("metadata"), dict) else {}
        intensity = k(node, "intensity") or k(meta, "intensity") or "medium"

        # synergy/conflicts
        synergy = k(node, "synergy", "synergy_with", "synergyWith", "cooperate_with")
        conflict = k(node, "conflicts", "conflicts_with", "conflictsWith", "interfere_with")
        synergy_list = list(synergy.keys()) if isinstance(synergy, dict) else list(synergy or [])
        conflict_list = list(conflict.keys()) if isinstance(conflict, dict) else list(conflict or [])

        return cls(
            situation_id=str(node_id),
            core_concept=str(core),
            keywords=_sc_dedup_str_list(kws),
            variations=_sc_dedup_str_list(vars_),
            emotion_links={str(k): _sc_clip01(v) for k, v in e_links.items()},
            intensity=_sc_norm_intensity(intensity, "medium"),
            context_patterns=(ctxp if isinstance(ctxp, dict) else {}),
            temporal_aspects=(meta.get("temporal_aspects") if isinstance(meta, dict) else None),
            triggers=[],
            matches=[],
            evidence=[],
            synergy_with=_sc_dedup_str_list(synergy_list),
            conflicts_with=_sc_dedup_str_list(conflict_list),
            confidence=0.0,
            source=node if isinstance(node, dict) else None,
        )


# -------------------------- SpatiotemporalContext ---------------------------
@dataclass(slots=True)
class SpatiotemporalContext(_SerializableMixin):
    """
    시공간 컨텍스트. 입력은 자유로운 문자열/사전이어도 되고, 파이프라인에서 정규화된 값을 채워넣습니다.
    """
    location: Optional[str]
    time: Optional[str]
    date: Optional[str]
    duration: Optional[str] = None
    temporal_sequence: Optional[List[str]] = None
    location_details: Optional[Dict[str, str]] = None
    timezone: Optional[str] = None
    start_index: Optional[int] = None
    end_index: Optional[int] = None
    confidence: float = 0.0

    def __post_init__(self):
        if isinstance(self.temporal_sequence, list):
            self.temporal_sequence = _sc_dedup_str_list(self.temporal_sequence)
        if isinstance(self.location_details, dict):
            self.location_details = {str(k): str(v) for k, v in self.location_details.items()}
        self.confidence = _sc_clip01(self.confidence)

    # 편의: 표지 누적 후 자동 요약
    def add_factor(self, *, type: str, subtype: str, text: str, position: Optional[int] = None) -> None:
        if not isinstance(self.temporal_sequence, list):
            self.temporal_sequence = []
        # 단순 시퀀스 표식만 누적(상세 구조는 상위에서 관리)
        if type in ("temporal", "change"):
            if text not in self.temporal_sequence:
                self.temporal_sequence.append(text)
        # 간단 장소 유형 추정(실내/실외)
        if type == "spatial" and isinstance(self.location_details, (dict, type(None))):
            ld = self.location_details or {}
            if any(w in (self.location or "") for w in ("카페", "식당", "도서관", "회사", "학교")):
                ld["type"] = ld.get("type") or "indoor"
            self.location_details = ld

    def finalize(self) -> None:
        # 시/분/장소가 채워져 있으면 신뢰도 보수 산정
        score = 0.0
        score += 0.4 if self.time else 0.0
        score += 0.3 if self.location else 0.0
        score += 0.2 if (self.temporal_sequence and len(self.temporal_sequence) > 0) else 0.0
        self.confidence = _sc_clip01(score)


# ------------------------ EmotionProgressionStage ---------------------------
@dataclass(slots=True)
class EmotionProgressionStage(_SerializableMixin):
    """
    감정 진행 단계. 외부 스키마의 stage 이름을 표준화(trigger/development/peak/aftermath)합니다.
    """
    stage: str
    description: str
    matched_text: Optional[str]
    position: Optional[int]
    intensity: str = "medium"
    confidence: float = 0.0
    transitions: List[Dict[str, Any]] = field(default_factory=list)
    context_factors: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.stage = _sc_stage_norm(self.stage)
        self.intensity = _sc_norm_intensity(self.intensity, "medium")
        self.confidence = _sc_clip01(self.confidence)
        if not isinstance(self.transitions, list):
            self.transitions = []
        if not isinstance(self.context_factors, dict):
            self.context_factors = {}

    @classmethod
    def from_label(cls, stage: str, examples: Any, *, default_desc: str = "") -> List["EmotionProgressionStage"]:
        out: List[EmotionProgressionStage] = []
        stg = _sc_stage_norm(stage)
        if isinstance(examples, (list, tuple)):
            for ex in examples:
                desc = default_desc or f"{stg} example"
                out.append(cls(stage=stg, description=desc, matched_text=str(ex), position=None))
        elif isinstance(examples, str):
            out.append(cls(stage=stg, description=default_desc or f"{stg} example", matched_text=examples, position=None))
        return out

    # 편의: 전이/컨텍스트 요인 추가
    def add_transition(self, t: Dict[str, Any]) -> None:
        self.transitions.append(t)

    def add_context_factor(self, k: str, v: Any) -> None:
        self.context_factors[k] = v


# --------------------------- EmotionFlowPattern -----------------------------
@dataclass(slots=True)
class EmotionFlowPattern(_SerializableMixin):
    """
    감정 흐름 패턴(상승/하강/변동/안정 + micro/shift 카운트 포함).
    오케스트레이터가 타임라인에서 바로 요약해 담을 수 있도록 설계.
    """
    pattern_type: str                     # e.g., "increasing", "decreasing", "volatile", "stable", "mixed"
    start_position: int
    end_position: int
    intensities: List[str]                # ["low","medium","high"] 시퀀스
    transitions: List[Dict[str, Any]]     # {"type":"shift_up","index":..., "delta":..., "is_micro": bool}
    duration: int
    confidence: float
    micro_shifts: int = 0
    major_shifts: int = 0
    avg_intensity: Optional[str] = None
    stages: Optional[List[str]] = None    # ["trigger","development",...]

    def __post_init__(self):
        self.start_position = int(self.start_position)
        self.end_position   = int(self.end_position)
        self.duration       = max(0, int(self.duration))
        self.confidence     = _sc_clip01(self.confidence)
        self.intensities    = [_sc_norm_intensity(x, "medium") for x in (self.intensities or [])]
        if not isinstance(self.transitions, list):
            self.transitions = []
        # 평균 강도 대략치
        if self.intensities:
            score = sum({"low": 0.0, "medium": 0.5, "high": 1.0}.get(x, 0.5) for x in self.intensities) / len(self.intensities)
            self.avg_intensity = "low" if score < 0.25 else ("high" if score > 0.75 else "medium")
        # 시퀀스 단계 정규화
        if self.stages:
            self.stages = [_sc_stage_norm(s) for s in self.stages]

    def finalize_from_transitions(self) -> None:
        """transitions 목록으로 micro/major 카운트 자동 집계."""
        ms, maj = 0, 0
        for t in (self.transitions or []):
            if str(t.get("type")) in ("micro_shift_up", "micro_shift_down"):
                ms += 1
            elif str(t.get("type")) in ("shift_up", "shift_down"):
                maj += 1
        self.micro_shifts = ms
        self.major_shifts = maj



# =============================================================================
# EmotionProgressionSituationAnalyzer — independent, label-driven
# =============================================================================
class EmotionProgressionSituationAnalyzer:
    """
    역할: 시공간·상황 패턴 매핑 → 감정 강도 흐름/전이 추적 → 지표 산출(중간 집계/조율기)
    - 라벨링 뼈대(스키마)는 변경하지 않고, 내부 로직을 라벨 데이터에 맞춰 동작
    - 정규식 샤딩 매처(가능 시)로 키워드/트리거/예시 매칭 정확도·속도 개선(없으면 안전 폴백)
    - 전이/강도/단계 추정은 데이터 기반 + 보수적 규칙으로 결선
    """

    # ------------------------------ ctor / params ------------------------------
    def __init__(self, embedding_model=None, emotions_data: Dict[str, Any] = None):
        try:
            os.environ["KSS_SUPPRESS_WARNINGS"] = "1"
            os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"
            logging.getLogger("kss").setLevel(logging.ERROR)
        except Exception:
            pass

        self.kss = kss if "kss" in globals() else None
        self.embedding_model = embedding_model
        self.emotions_data = emotions_data or {}
        self.warned_ids: Set[str] = set()

        # 튜닝 파라미터(독립, ENV 반영)
        self.params = {
            "regex_shard_size": int(os.environ.get("SIT_REGEX_SHARD_SIZE", "512")),
            "transition_conf_thr": float(os.environ.get("SIT_TRANS_CONF_THR", "0.6")),
            "micro_shift_half": 0.5,  # 미세 전이: full 임계의 절반 이상일 때
        }

        # 캐시
        self.emotion_intensity_cache: Dict[str, Tuple[str, float]] = {}
        self.transition_pattern_cache: Dict[str, Any] = {}
        self.pattern_match_cache: Dict[str, Any] = {}
        self.intensity_analysis_cache: Dict[str, Any] = {}

        # 라벨 기반 인덱스/매처 준비
        self._build_indices(self.emotions_data)

        # (데이터에 전이 힌트가 없을 때만) 보조 트리거
        self._fallback_transitions = {
            "contrast": ["하지만", "그러나", "반면"],
            "cause": ["때문에", "그래서", "따라서"],
            "addition": ["또한", "게다가", "더불어"],
            "temporal": ["먼저", "이후", "마지막으로"],
        }

    # ------------------------------ Index/Matchers ------------------------------
    def _build_indices(self, data: Dict[str, Any]) -> None:
        # RegexShardMatcher 사용 가능여부
        try:
            _ = RegexShardMatcher  # type: ignore
            self._rx_available = True
        except Exception:
            self._rx_available = False

        low, med, high = [], [], []
        ctx_low, ctx_med, ctx_high = [], [], []
        stage_map: Dict[str, List[str]] = {"trigger": [], "development": [], "peak": [], "aftermath": []}
        triggers_union: List[str] = []

        root = data if isinstance(data, dict) else {}
        for _, node in root.items():
            if not isinstance(node, dict):
                continue

            # intensity_levels
            prof = self._k(node, "emotion_profile", "emotionProfile", "profile")
            levels = self._k(prof, "intensity_levels", "intensityLevels", "levels") if isinstance(prof, dict) else None
            if isinstance(levels, dict):
                ex = self._extract_intensity_examples(levels)
                low.extend(ex.get("low", []))
                med.extend(ex.get("medium", []))
                high.extend(ex.get("high", []))

            # context_patterns.situations
            ctxp = self._k(node, "context_patterns", "contextPatterns", "contexts", "situationPatterns")
            if isinstance(ctxp, dict):
                sits = self._k(ctxp, "situations", "scenarios", "cases")
                it = list(sits.values()) if isinstance(sits, dict) else (sits if isinstance(sits, list) else [])
                for s in it or []:
                    if not isinstance(s, dict):
                        continue
                    inten = self._norm_intensity(self._k(s, "intensity", "level") or "medium")
                    phrases = list(self._iter_phrases(self._k(s, "keywords", "phrases", "terms", "tokens"))) + \
                              list(self._iter_phrases(self._k(s, "variations", "variants", "aliases", "synonyms")))
                    if inten == "low":
                        ctx_low.extend(phrases)
                    elif inten == "high":
                        ctx_high.extend(phrases)
                    else:
                        ctx_med.extend(phrases)
                    prog = self._k(s, "emotion_progression", "progression", "flow")
                    if isinstance(prog, dict):
                        for stg, examples in prog.items():
                            st = self._norm_stage(stg)
                            stage_map.setdefault(st, []).extend(list(self._iter_phrases(examples)))

            # transitions.patterns
            trans_root = self._k(node, "emotion_transitions", "emotionTransitions", "transitions")
            patterns = trans_root if isinstance(trans_root, list) else (self._k(trans_root, "patterns", "rules", "list") if isinstance(trans_root, dict) else None)
            if isinstance(patterns, list):
                for p in patterns:
                    ts = list(self._iter_phrases(self._k(p, "triggers", "cues", "signals", "markers")))
                    if ts:
                        triggers_union.extend(ts)

        # 중복 제거
        self._int_low, self._int_med, self._int_high = map(self._dedup, (low, med, high))
        self._ctx_low, self._ctx_med, self._ctx_high = map(self._dedup, (ctx_low, ctx_med, ctx_high))
        for k in list(stage_map.keys()):
            stage_map[k] = self._dedup(stage_map.get(k, []))
        self._stage_map = stage_map
        self._triggers = self._dedup(triggers_union)

        # 정규식 샤드 매처 준비
        self._matchers: Dict[str, Any] = {}
        shard = int(self.params["regex_shard_size"])
        if self._rx_available:
            self._matchers["int_low"] = RegexShardMatcher(self._int_low, shard_size=shard, boundary=False, ignore_space=True)
            self._matchers["int_med"] = RegexShardMatcher(self._int_med, shard_size=shard, boundary=False, ignore_space=True)
            self._matchers["int_high"] = RegexShardMatcher(self._int_high, shard_size=shard, boundary=False, ignore_space=True)

            self._matchers["ctx_low"] = RegexShardMatcher(self._ctx_low, shard_size=shard, boundary=False, ignore_space=True)
            self._matchers["ctx_med"] = RegexShardMatcher(self._ctx_med, shard_size=shard, boundary=False, ignore_space=True)
            self._matchers["ctx_high"] = RegexShardMatcher(self._ctx_high, shard_size=shard, boundary=False, ignore_space=True)

            for stg, phrases in self._stage_map.items():
                self._matchers[f"stage_{stg}"] = RegexShardMatcher(phrases, shard_size=shard, boundary=False, ignore_space=True)

            if self._triggers:
                parts = [re.escape(re.sub(r"\s+", "", self._norm_txt(x))) for x in self._triggers]
                self._rx_triggers = re.compile("|".join(parts)) if parts else None
            else:
                self._rx_triggers = None
        else:
            self._rx_triggers = None

    # ------------------------------- Public API -------------------------------
    def analyze(self, text: str, emotion_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        일회성 통합 분석: 진행/전이/강도/시공간·문맥 표지까지 묶어서 반환.
        오케스트레이터에서 바로 합성할 수 있는 형태의 중간 산출물을 만든다.
        """
        sents = self._split_text_into_sentences(text)
        confidences = [0.5] * len(sents)  # 필요 시 호출부에서 넣을 수 있음

        # ✅ 함수명 오타 수정: _analyze_emotion_progression → _analyze_emotional_progression
        prog = self._analyze_emotional_progression(text, emotion_data, confidences)
        ctx = self._analyze_emotional_context_internal(text, emotion_data)
        flow = self._analyze_emotion_flows_enhanced(text, emotion_data)

        # 시공간 표지 집계
        st_ctx = {
            "location": None, "time": None, "date": None,
            "temporal_sequence": [], "location_details": {}
        }
        for i, s in enumerate(sents):
            for f in self._extract_contextual_factors(s):
                if f["type"] == "spatial" and not st_ctx["location"]:
                    st_ctx["location"] = f["text"]
                if f["type"] == "temporal" and f["subtype"] in ("absolute", "relative") and not st_ctx["time"]:
                    st_ctx["time"] = f["text"]
                if f["type"] in ("temporal", "change"):
                    st_ctx["temporal_sequence"].append({"marker": f["text"], "position": i, "text": s})

        return {
            "sentences": sents,
            "progression": prog,
            "emotional_context": ctx,
            "flows": flow,
            "spatiotemporal": st_ctx,
        }

    # EmotionProgressionSituationAnalyzer 클래스 안쪽(다른 메서드와 같은 들여쓰기 레벨)에 추가
    def _analyze_emotional_progression(self, text: str, emotion_data: Dict[str, Any],
                                       sentence_confidences: List[float]) -> Dict[str, Any]:
        """
        감정 진행 분석(데이터 중심):
          - 라벨의 progression/transition/intensity 예시를 활용
          - 문장별 강도/단계/전이를 추정하고, 변화 지점을 축적
        ENV:
          SIT_TRANS_CONF_THR (float, default=0.5) : 전이 확정 임계
          SIT_MICRO_SHIFT_HALF (float, default=0.5) : micro 전이 임계 = full × 이 값
          SIT_TRANS_HOLD_MIN (int, default=1) : 전이 최소 유지 구간
        """
        results = {
            "stages": [],
            "transitions": [],
            "intensity_changes": [],
            "contextual_factors": [],
            "stage_confidences": {}
        }
        try:
            sents = self._split_text_into_sentences(text)
            if not sents:
                return results

            # 라벨 기반 패턴
            patterns = []
            if isinstance(emotion_data, dict):
                trans_root = emotion_data.get("emotion_transitions") or {}
                patterns = trans_root.get("patterns", [])

            # 임계/홀드 (ENV 우선)
            full_thr = float(os.environ.get("SIT_TRANS_CONF_THR", "0.5"))
            micro_h = float(os.environ.get("SIT_MICRO_SHIFT_HALF", "0.5"))
            hold_min = int(os.environ.get("SIT_TRANS_HOLD_MIN", "1"))
            half_thr = max(0.0, full_thr * micro_h)

            current_stage = "trigger"
            stage_bucket: List[Dict[str, Any]] = []
            last_change_idx = -999

            for idx, sent in enumerate(sents):
                base_conf = sentence_confidences[idx] if idx < len(sentence_confidences) else 0.5

                # (1) 강도/단계 추정
                intensity, conf = self._get_emotion_intensity_from_structure(emotion_data, base_conf, sent)
                stage_guess = self._guess_stage_from_examples(sent) or self._map_intensity_to_stage(intensity)

                # (2) 전이 탐지
                trans = self._detect_transition_patterns(sent, patterns, current_stage, intensity)
                strongest = max(trans, key=lambda x: x["confidence"]) if trans else None

                # (3) 버킷 적재
                stage_bucket.append({
                    "text": sent,
                    "intensity": intensity,
                    "confidence": conf,
                    "stage_guess": stage_guess,
                    "transitions": trans,
                    "position": idx
                })

                # (4) 전이 판단: hold 이후에만 반영
                if strongest and (idx - last_change_idx) >= hold_min:
                    if strongest["confidence"] >= full_thr:
                        results["transitions"].append({
                            "from_stage": current_stage,
                            "to_stage": self._norm_stage(strongest.get("to_emotion", current_stage)),
                            "trigger": strongest.get("trigger"),
                            "confidence": strongest["confidence"],
                            "position": idx,
                            "type": "shift",
                            "analysis": strongest.get("transition_analysis", {})
                        })
                        # 현재 단계 마감
                        if stage_bucket:
                            avg_conf = round(sum(s["confidence"] for s in stage_bucket) / len(stage_bucket), 3)
                            results["stages"].append({
                                "stage_name": current_stage,
                                "sentences": stage_bucket.copy(),
                                "confidence": avg_conf,
                                "duration": len(stage_bucket)
                            })
                            stage_bucket.clear()
                        current_stage = self._norm_stage(strongest.get("to_emotion", current_stage))
                        last_change_idx = idx

                    elif strongest["confidence"] >= half_thr:
                        results["transitions"].append({
                            "from_stage": current_stage,
                            "to_stage": current_stage,  # micro 전이는 단계 고정
                            "trigger": strongest.get("trigger"),
                            "confidence": strongest["confidence"],
                            "position": idx,
                            "type": "micro_shift",
                            "analysis": strongest.get("transition_analysis", {})
                        })
                        last_change_idx = idx

                # (5) 강도 변화 기록
                if len(stage_bucket) >= 2:
                    prev_int = stage_bucket[-2]["intensity"]
                    if prev_int != intensity:
                        results["intensity_changes"].append({
                            "from_intensity": prev_int,
                            "to_intensity": intensity,
                            "position": idx,
                            "confidence": conf
                        })

            # 마지막 단계 마감
            if stage_bucket:
                avg_conf = round(sum(s["confidence"] for s in stage_bucket) / len(stage_bucket), 3)
                results["stages"].append({
                    "stage_name": current_stage,
                    "sentences": stage_bucket.copy(),
                    "confidence": avg_conf,
                    "duration": len(stage_bucket)
                })

            # 단계별 평균 신뢰도
            for st in results["stages"]:
                results["stage_confidences"][st["stage_name"]] = st["confidence"]

            return results

        except Exception as e:
            logger.error(f"[EPSA:_analyze_emotional_progression] 오류: {e}")
            return results

    def _calculate_sub_emotion_score(self, sentence: str, sub_emotion_data: Dict[str, Any]) -> float:
        """
        세부 감정(sub_emotion)에 대한 보수적 점수(0~1)를 계산.
        - core_keywords, intensity_examples, situations.keywords/triggers, progression 예시,
          emotion_transitions.patterns.triggers 를 순회하여 가산.
        - RegexShardMatcher가 있으면 공백무시 부분일치로 강건 매칭.
        """
        score = 0.0
        try:
            j = re.sub(r"\s+", "", self._norm_txt(sentence))

            # --- core_keywords ---
            core_keywords = sub_emotion_data.get("core_keywords", []) or \
                            (sub_emotion_data.get("emotion_profile", {}) or {}).get("core_keywords", [])
            if core_keywords:
                if getattr(self, "_rx_available", False):
                    rx = _rx_cache(("core", tuple(core_keywords)), tuple(core_keywords))
                    hits = rx.find_all(j)
                    if hits:
                        ratio = min(1.0, len(hits) / max(1, len(core_keywords)))
                        score += ratio * 0.4
                else:
                    matched = [kw for kw in core_keywords if re.sub(r"\s+", "", self._norm_txt(kw)) in j]
                    if matched:
                        ratio = len(matched) / max(1, len(core_keywords))
                        score += ratio * 0.4

            # --- intensity examples ---
            levels = sub_emotion_data.get("intensity_levels", {}) or \
                     (sub_emotion_data.get("emotion_profile", {}) or {}).get("intensity_levels", {})
            ex = self._extract_intensity_examples(levels)
            for lv, bonus in (("high", 0.30), ("medium", 0.15), ("low", 0.08)):
                for w in ex.get(lv, []):
                    if re.sub(r"\s+", "", self._norm_txt(w)) in j:
                        score += bonus
                        break  # 한 레벨당 첫 매치만

            # --- context_patterns.situations ---
            ctxp = sub_emotion_data.get("context_patterns", {}) or \
                   (sub_emotion_data.get("emotion_profile", {}) or {}).get("context_patterns", {})
            sits = ctxp.get("situations", {})
            if isinstance(sits, dict):
                for _, s in sits.items():
                    if not isinstance(s, dict):
                        continue

                    # keywords
                    kws = s.get("keywords", []) or []
                    if kws:
                        matched = 0
                        if getattr(self, "_rx_available", False):
                            rx = _rx_cache(("kws", tuple(kws)), tuple(kws))
                            matched = len(rx.find_all(j))
                        else:
                            matched = sum(1 for kw in kws if re.sub(r"\s+", "", self._norm_txt(kw)) in j)
                        if matched:
                            score += min(0.3, (matched / max(1, len(kws))) * 0.3)

                    # triggers
                    trigs = s.get("triggers", []) or []
                    if trigs:
                        mt = 0
                        if getattr(self, "_rx_available", False):
                            rx = _rx_cache(("trigs", tuple(trigs)), tuple(trigs))
                            mt = len(rx.find_all(j))
                        else:
                            mt = sum(1 for t in trigs if re.sub(r"\s+", "", self._norm_txt(t)) in j)
                        if mt:
                            score += min(0.2, 0.1 * mt)

                    # progression 예시
                    prog = s.get("emotion_progression", {}) or {}
                    if isinstance(prog, dict):
                        for _, exs in prog.items():
                            for ex_s in self._iter_phrases(exs):
                                if re.sub(r"\s+", "", self._norm_txt(ex_s)) in j:
                                    score += 0.05
                                    break

            # --- emotion_transitions.patterns ---
            et = sub_emotion_data.get("emotion_transitions", {}) or {}
            patterns = et.get("patterns", []) or []
            for p in patterns:
                trigs = p.get("triggers", []) or []
                if not trigs:
                    continue
                mt = 0
                if getattr(self, "_rx_available", False):
                    rx = _rx_cache(("ptrigs", tuple(trigs)), tuple(trigs))
                    mt = len(rx.find_all(j))
                else:
                    mt = sum(1 for t in trigs if re.sub(r"\s+", "", self._norm_txt(t)) in j)
                if mt:
                    base_conf = float((p.get("transition_analysis") or {}).get("pattern_confidence", 0.5))
                    ic = str((p.get("transition_analysis") or {}).get("intensity_change", "")).lower()
                    if any(k in ic for k in ("급감", "하락", "decrease")):
                        base_conf += 0.05
                    elif any(k in ic for k in ("증가", "상승", "increase")):
                        base_conf += 0.03
                    score += min(0.22, base_conf * 0.1 + 0.02 * mt)

            return round(min(1.0, max(0.0, score)), 3)

        except Exception as e:
            logger.error(f"[EPSA:_calculate_sub_emotion_score] 오류: {e}")
            return 0.0

    def _check_multi_emotion_transition(
            self,
            matched_emotions: List[str],
            sentence: str,
            multi_transitions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        한 문장에 from_emotions 전부가 감지되고, triggers 중 1개 이상이 나타나면
        to_emotions로의 전이가 있었다고 판단해 간단 카드로 반환.
        """
        out: List[Dict[str, Any]] = []
        try:
            j = re.sub(r"\s+", "", self._norm_txt(sentence))
            mset = set(matched_emotions or [])
            for mt in (multi_transitions or []):
                from_emos = mt.get("from_emotions", []) or []
                if not set(from_emos).issubset(mset):
                    continue
                trigs = mt.get("triggers", []) or []
                found = [t for t in trigs if re.sub(r"\s+", "", self._norm_txt(t)) in j]
                if found:
                    out.append({
                        "from_emotions": from_emos,
                        "to_emotions": mt.get("to_emotions", []) or [],
                        "trigger": found[0],
                        "analysis": mt.get("transition_analysis", {}) or {},
                    })
            return out
        except Exception as e:
            logger.error(f"[EPSA:_check_multi_emotion_transition] 오류: {e}")
            return out

    # --------------------------- Internal Detectors ---------------------------
    def _detect_multiple_emotions(self, sentence: str, sub_emotion_data: Dict[str, Any]) -> List[str]:
        """
        문장에서 sub_emotion의 core_keywords(또는 emotion_profile.core_keywords)를 찾아 리스트로 반환.
        """
        detected_emotions: List[str] = []
        try:
            core_kw = sub_emotion_data.get("core_keywords", []) or \
                      (sub_emotion_data.get("emotion_profile", {}) or {}).get("core_keywords", [])
            if not core_kw:
                return detected_emotions
            j = re.sub(r"\s+", "", self._norm_txt(sentence))
            for kw in core_kw:
                k = re.sub(r"\s+", "", self._norm_txt(kw))
                if k and k in j and kw not in detected_emotions:
                    detected_emotions.append(kw)
            return detected_emotions
        except Exception:
            return detected_emotions

    def _detect_transition_patterns(self, sentence: str, transition_patterns: List[Dict[str, Any]],
                                    current_stage: str, current_intensity: str) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        try:
            raw = self._norm_txt(sentence)
            j = re.sub(r"\s+", "", raw)

            sudden = bool(re.search(r"(갑자기|돌연|순간)", raw))
            gradual = bool(re.search(r"(점점|차츰|서서히)", raw))

            # 정규식 트리거 감지(전역)
            trig_hits = []
            if self._rx_available and getattr(self, "_triggers", None):
                rx = getattr(self, "_rx_triggers", None)
                if rx:
                    trig_hits = [m.group(0) for m in rx.finditer(j)]

            # 1) 라벨 패턴 우선
            for pattern in transition_patterns or []:
                trigs = list(self._iter_phrases(self._k(pattern, "triggers", "cues", "signals", "markers")))
                matched = 0
                found = None
                if trigs:
                    norm_trigs = [re.sub(r"\s+", "", self._norm_txt(t)) for t in trigs]
                    if trig_hits:
                        inter = [t for t in trig_hits if t in norm_trigs]
                        matched = len(inter)
                        found = inter[0] if inter else None
                    else:
                        found = next((t for t in norm_trigs if t and t in j), None)
                        matched = 1 if found else 0

                if matched > 0:
                    analysis = pattern.get("transition_analysis", {}) or {}
                    base_conf = float(analysis.get("pattern_confidence", 0.5))
                    ic = str(analysis.get("intensity_change", "")).lower()
                    if any(k in ic for k in ("급감", "하락", "decrease", "sudden")):
                        base_conf += 0.08
                    elif any(k in ic for k in ("증가", "상승", "increase", "gradual")):
                        base_conf += 0.05
                    sp = str(analysis.get("emotion_shift_point", "")).lower()
                    if "peak" in sp and current_stage in ("development", "trigger"): base_conf += 0.07
                    if "aftermath" in sp and current_stage == "peak": base_conf += 0.10
                    if current_intensity == "high" and "increase" in ic: base_conf += 0.02
                    base_conf += min(0.05, 0.02 * (matched - 1))
                    if sudden: base_conf += 0.04
                    if gradual: base_conf += 0.02

                    out.append({
                        "from_emotion": pattern.get("from_emotion") or pattern.get("from"),
                        "to_emotion": pattern.get("to_emotion") or pattern.get("to"),
                        "trigger": found or trigs[0],
                        "confidence": round(max(0.0, min(base_conf, 1.0)), 3),
                        "transition_analysis": analysis
                    })

            # 2) 라벨 전이가 없거나 트리거가 빈약할 때: 보조 트리거
            if not out:
                allow_fb = os.environ.get("SIT_ALLOW_FALLBACK_TRIGGERS", "1") != "0"
                if allow_fb:
                    fb = ["하지만", "그러나", "반면", "갑자기", "돌연", "순간", "서서히", "차츰", "점점", "그래도"]
                    fb_norm = [re.sub(r"\s+", "", self._norm_txt(t)) for t in fb]
                    hit = next((t for t in fb_norm if t in j), None)
                    if hit:
                        base = 0.55
                        if sudden: base += 0.05
                        if gradual: base += 0.03
                        out.append({
                            "from_emotion": current_stage,
                            "to_emotion": current_stage,  # 단계 고정: 안전
                            "trigger": hit,
                            "confidence": round(min(0.95, base), 3),
                            "transition_analysis": {"origin": "fallback"}
                        })

            return out
        except Exception as e:
            logger.error(f"[EPSA:_detect_transition_patterns] 오류: {e}")
            return out

    # ----------------------------- Context/Intensity ---------------------------
    def _get_emotion_intensity_from_structure(self, emotion_data: Dict[str, Any], confidence: float, sentence: str) -> Tuple[str, float]:
        """라벨 intensity 예시 + 보수적 확률 보정으로 강도 추정"""
        try:
            levels = emotion_data.get("intensity_levels") or (emotion_data.get("emotion_profile", {}) or {}).get("intensity_levels", {})
            ex = self._extract_intensity_examples(levels)

            j = re.sub(r"\s+", "", self._norm_txt(sentence))
            found = {"low": False, "medium": False, "high": False}
            if self._rx_available:
                if self._matchers.get("int_high") and self._matchers["int_high"].find_all(j): found["high"] = True
                if self._matchers.get("int_med") and self._matchers["int_med"].find_all(j): found["medium"] = True
                if self._matchers.get("int_low") and self._matchers["int_low"].find_all(j): found["low"] = True
            else:
                for lv in ("high", "medium", "low"):
                    for w in ex.get(lv, []):
                        if re.sub(r"\s+", "", w) in j:
                            found[lv] = True; break

            if confidence >= 0.7:
                intensity = "high"; adj = min(confidence * 1.05, 1.0)
            elif confidence >= 0.4:
                intensity = "medium"; adj = confidence
            else:
                intensity = "low"; adj = max(confidence * 0.95, 0.0)

            if found["high"]:
                intensity, adj = "high", min(adj + 0.08, 1.0)
            elif found["medium"] and intensity != "high":
                intensity, adj = "medium", min(adj + 0.05, 1.0)
            elif found["low"] and intensity not in ("medium", "high"):
                intensity, adj = "low", min(adj + 0.02, 1.0)

            return intensity, round(float(adj), 3)
        except Exception as e:
            logger.error(f"[EPSA:_get_emotion_intensity_from_structure] 오류: {e}")
            return "medium", confidence

    # ------------------------------- Context utils ------------------------------
    def _guess_stage_from_examples(self, sentence: str) -> Optional[str]:
        j = re.sub(r"\s+", "", self._norm_txt(sentence))
        for stg in ("trigger", "development", "peak", "aftermath"):
            phrases = self._stage_map.get(stg, [])
            if not phrases:
                continue
            if self._rx_available and self._matchers.get(f"stage_{stg}"):
                if self._matchers[f"stage_{stg}"].find_all(j):
                    return stg
            else:
                for p in phrases:
                    if re.sub(r"\s+", "", p) in j:
                        return stg
        return None

    def _map_intensity_to_stage(self, intensity: str) -> str:
        return {"low": "trigger", "medium": "development", "high": "peak"}.get(intensity, "development")

    # ------------------------------- Misc helpers -------------------------------
    @staticmethod
    def _norm_txt(s: str) -> str:
        try:
            import unicodedata as _ud
            s = _ud.normalize("NFKC", s or "")
        except Exception:
            pass
        return s

    @staticmethod
    def _iter_phrases(node: Any) -> Iterable[str]:
        if not node:
            return
        if isinstance(node, str):
            s = node.strip()
            if s:
                yield s; return
        if isinstance(node, (list, tuple, set)):
            for v in node:
                if isinstance(v, str) and v.strip():
                    yield v.strip()
        elif isinstance(node, dict):
            for k in ("text", "phrase", "value", "token"):
                v = node.get(k)
                if isinstance(v, str) and v.strip():
                    yield v.strip()

    @staticmethod
    def _dedup(xs: Iterable[str]) -> List[str]:
        out, seen = [], set()
        for s in xs or []:
            s2 = str(s).strip()
            if s2 and s2 not in seen:
                seen.add(s2); out.append(s2)
        return out

    @staticmethod
    def _k(node: Dict[str, Any], *alts: str) -> Any:
        if not isinstance(node, dict):
            return None
        lower = {str(kk).lower(): kk for kk in node.keys()}
        for nm in alts:
            if nm.lower() in lower:
                return node[lower[nm.lower()]]
        return None

    @staticmethod
    def _norm_stage(name: Any) -> str:
        n = str(name).strip().lower() if name else ""
        alias = {
            "onset": "trigger", "start": "trigger", "trigger": "trigger",
            "build": "development", "build-up": "development", "development": "development",
            "climax": "peak", "apex": "peak", "peak": "peak",
            "aftermath": "aftermath", "resolution": "aftermath", "cooldown": "aftermath"
        }
        return alias.get(n, n or "trigger")

    @staticmethod
    def _norm_intensity(v: Any) -> str:
        s = str(v).strip().lower() if v is not None else ""
        if s in {"low", "lo", "약함", "weak"}: return "low"
        if s in {"medium", "med", "mid", "중간"}: return "medium"
        if s in {"high", "hi", "강함", "strong"}: return "high"
        return "medium"

    @staticmethod
    def _extract_intensity_examples(levels: Dict[str, Any]) -> Dict[str, List[str]]:
        out = {"low": [], "medium": [], "high": []}
        if not isinstance(levels, dict):
            return out
        if "intensity_examples" in levels and isinstance(levels["intensity_examples"], dict):
            for lv in ("low", "medium", "high"):
                out[lv].extend(list(EmotionProgressionSituationAnalyzer._iter_phrases(levels["intensity_examples"].get(lv))))
            return out
        for lv in ("low", "medium", "high"):
            block = levels.get(lv) or {}
            if isinstance(block, dict):
                out[lv].extend(list(EmotionProgressionSituationAnalyzer._iter_phrases(block.get("intensity_examples"))))
        return out

    # ------------------------- Context & Flow Analyzers -------------------------
    def _analyze_emotional_context_internal(self, text: str, sub_emotion_data: Dict[str, Any]) -> Dict[str, Any]:
        results = {
            "emotional_contexts": [],
            "intensity_progression": [],
            "context_transitions": [],
            "multi_transitions_detected": [],
            "emotional_transitions": [],
            "dominant_emotions": [],
        }
        try:
            sents = self._split_text_into_sentences(text)
            if not sents:
                return results

            subs = sub_emotion_data.get("sub_emotions", {}) or \
                   (sub_emotion_data.get("emotion_profile", {}) or {}).get("sub_emotions", {})
            if not isinstance(subs, dict) or not subs:
                merged = {}
                for _pri, _emo in (self.emotions_data or {}).items():
                    if not isinstance(_emo, dict): continue
                    _subs = _emo.get("sub_emotions", {}) or (_emo.get("emotion_profile", {}) or {}).get("sub_emotions",
                                                                                                        {})
                    if isinstance(_subs, dict): merged.update(_subs)
                subs = merged

            if not hasattr(self, "_val_pos"):
                pos, neg = set(), set()
                for _pri, _emo in (self.emotions_data or {}).items():
                    if not isinstance(_emo, dict): continue
                    ling = (_emo.get("linguistic_patterns") or {})
                    pos.update([w for w in (ling.get("positive_markers") or []) if isinstance(w, str)])
                    neg.update([w for w in (ling.get("negative_markers") or []) if isinstance(w, str)])
                self._val_pos, self._val_neg = list(pos), list(neg)

            def _valence_score(sentence: str) -> float:
                j = re.sub(r"\s+", "", self._norm_txt(sentence))
                pos_hits = sum(1 for w in self._val_pos if re.sub(r"\s+", "", self._norm_txt(w)) in j)
                neg_hits = sum(1 for w in self._val_neg if re.sub(r"\s+", "", self._norm_txt(w)) in j)
                if pos_hits == 0 and neg_hits == 0:
                    return 0.0
                return (pos_hits - neg_hits) / float(max(1, pos_hits + neg_hits))

            flows = []
            trans_records = []
            dom_counter = defaultdict(int)

            for idx, sentence in enumerate(sents):
                if not sentence.strip():
                    continue

                raw_scores: Dict[str, float] = {}
                cat_map: Dict[str, str] = {}
                for sub_name, sub_data in (subs or {}).items():
                    if not isinstance(sub_data, dict): continue
                    sub_id = str((sub_data.get("metadata") or {}).get("emotion_id") or sub_name)
                    raw_scores[sub_id] = round(self._calculate_sub_emotion_score(sentence, sub_data), 3)
                    cat_map[sub_id] = str((sub_data.get("metadata") or {}).get("primary_category") or "")

                v = _valence_score(sentence)
                adj_scores = dict(raw_scores)
                if v < -0.15:
                    for sid, sc in raw_scores.items():
                        cat = cat_map.get(sid, "")
                        if cat in ("희", "락"):
                            adj_scores[sid] = round(sc * 0.6, 3)
                        elif cat in ("애", "노", "우"):
                            adj_scores[sid] = round(min(1.0, sc * 1.25), 3)
                elif v > 0.15:
                    for sid, sc in raw_scores.items():
                        cat = cat_map.get(sid, "")
                        if cat in ("애", "노", "우"):
                            adj_scores[sid] = round(sc * 0.7, 3)
                        elif cat in ("희", "락"):
                            adj_scores[sid] = round(min(1.0, sc * 1.15), 3)

                top_id, top_score = ("none", 0.0)
                if adj_scores:
                    top_id, top_score = max(adj_scores.items(), key=lambda x: x[1])
                    if top_score < 0.18:
                        top_id, top_score = ("none", 0.0)

                if top_id != "none":
                    intensity = self._get_emotion_intensity_from_structure(
                        subs.get(top_id, {}) if top_id in subs else {}, top_score, sentence
                    )[0]
                else:
                    intensity = "low"

                results["emotional_contexts"].append({
                    "index": idx,
                    "text": sentence,
                    "dominant_sub_emotion": top_id,
                    "dominant_score": top_score,
                    "sub_emotion_scores": adj_scores,
                    "intensity": intensity,
                    "valence": round(v, 3),
                })

                flows.append({
                    "sentence": sentence,
                    "dominant_emotion": {"emotion": top_id, "score": top_score},
                    "position": idx
                })
                if top_id != "none":
                    dom_counter[top_id] += 1

            for i in range(1, len(results["emotional_contexts"])):
                prev = results["emotional_contexts"][i - 1]
                cur = results["emotional_contexts"][i]
                if prev["dominant_sub_emotion"] != "none" and cur["dominant_sub_emotion"] != "none":
                    if prev["dominant_sub_emotion"] != cur["dominant_sub_emotion"]:
                        conf = min(prev["dominant_score"], cur["dominant_score"])
                        trans_records.append({
                            "from_emotion": prev["dominant_sub_emotion"],
                            "to_emotion": cur["dominant_sub_emotion"],
                            "position": i,
                            "confidence": round(conf, 3)
                        })

            multi_trans_rules = []
            for _sub in subs.values():
                et = _sub.get("emotion_transitions", {})
                rules = et.get("multi_emotion_transitions", [])
                if isinstance(rules, list):
                    multi_trans_rules.extend(rules)
            if multi_trans_rules:
                for i, row in enumerate(results["emotional_contexts"]):
                    matched = [eid for eid, sc in (row.get("sub_emotion_scores") or {}).items() if sc >= 0.3]
                    det = self._check_multi_emotion_transition(matched, row["text"], multi_trans_rules)
                    if det:
                        for d in det:
                            trans_records.append({
                                "from_emotion": "/".join(d.get("from_emotions") or []),
                                "to_emotion": "/".join(d.get("to_emotions") or []),
                                "position": i,
                                "confidence": 0.6,
                                "trigger": d.get("trigger")
                            })
                        results["multi_transitions_detected"].extend(det)

            results["emotional_transitions"] = trans_records

            for row in results["emotional_contexts"]:
                results["intensity_progression"].append({
                    "position": row["index"],
                    "intensity": row["intensity"],
                    "score": row["dominant_score"]
                })

            total = max(1, len(flows))
            dom_sorted = sorted(dom_counter.items(), key=lambda x: x[1], reverse=True)
            results["dominant_emotions"] = [
                {"emotion": k, "frequency": v, "ratio": round(v / total, 3)}
                for k, v in dom_sorted[:3]
            ]

            results.setdefault("emotion_flows", flows)

            return results

        except Exception as e:
            logger.error(f"[EPSA:_analyze_emotional_context_internal] 오류: {e}")
            return results

    # ------------------------------ Flow (enhanced) ------------------------------
    def _analyze_emotion_flows_enhanced(self, text: str, sub_emotion_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            flow_results = {"flow_patterns": [], "emotion_sequences": [], "intensity_changes": [], "temporal_aspects": []}
            sents = self._split_text_into_sentences(text)
            if not sents:
                return flow_results

            emotion_transitions = (sub_emotion_data.get("emotion_transitions", {}) or {}).get("patterns", [])
            levels = (sub_emotion_data.get("emotion_profile", {}) or {}).get("intensity_levels", {})
            ex = self._extract_intensity_examples(levels)

            current_flow = {"pattern_type": None, "start_position": 0, "intensities": [], "transitions": []}

            for idx, sentence in enumerate(sents):
                j = re.sub(r"\s+", "", self._norm_txt(sentence))
                cur = "medium"
                if self._rx_available:
                    if RegexShardMatcher(ex.get("high", []), 256, False, True).find_all(j):   cur = "high"
                    elif RegexShardMatcher(ex.get("medium", []), 256, False, True).find_all(j): cur = "medium"
                    elif RegexShardMatcher(ex.get("low", []), 256, False, True).find_all(j):    cur = "low"
                else:
                    for lv in ("high", "medium", "low"):
                        if any(re.sub(r"\s+", "", self._norm_txt(w)) in j for w in ex.get(lv, [])):
                            cur = lv; break

                matched_transitions = []
                for p in emotion_transitions or []:
                    trigs = p.get("triggers", [])
                    trig = next((t for t in trigs if re.sub(r"\s+", "", self._norm_txt(t)) in j), None)
                    if trig:
                        matched_transitions.append({
                            "from_emotion": p.get("from_emotion") or p.get("from"),
                            "to_emotion": p.get("to_emotion") or p.get("to"),
                            "trigger": trig, "position": idx
                        })

                # 시간표지
                temporal_markers = {"gradual": r"(점점|차츰|서서히)", "sudden": r"(갑자기|돌연|순간)", "repetitive": r"(자주|계속|반복)"}
                for aspect, rx in temporal_markers.items():
                    if re.search(rx, sentence):
                        flow_results["temporal_aspects"].append({"type": aspect, "position": idx, "text": sentence})

                current_flow["intensities"].append(cur)
                if matched_transitions:
                    current_flow["transitions"].extend(matched_transitions)

                if idx > 0:
                    prev = current_flow["intensities"][-2]
                    if prev != cur:
                        pattern_type = self._determine_flow_pattern_type(current_flow["intensities"])
                        flow_results["flow_patterns"].append({
                            "pattern_type": pattern_type,
                            "start_position": current_flow["start_position"],
                            "end_position": idx - 1,
                            "intensities": current_flow["intensities"][:-1],
                            "transitions": current_flow["transitions"]
                        })
                        current_flow = {"pattern_type": None, "start_position": idx, "intensities": [cur], "transitions": []}

            if current_flow["intensities"]:
                pattern_type = self._determine_flow_pattern_type(current_flow["intensities"])
                flow_results["flow_patterns"].append({
                    "pattern_type": pattern_type,
                    "start_position": current_flow["start_position"],
                    "end_position": len(sents) - 1,
                    "intensities": current_flow["intensities"],
                    "transitions": current_flow["transitions"]
                })

            flow_results["emotion_sequences"] = self._analyze_emotion_sequences(flow_results["flow_patterns"])
            return flow_results

        except Exception as e:
            logger.error(f"[EPSA:_analyze_emotion_flows_enhanced] 오류: {e}")
            return {"flow_patterns": [], "emotion_sequences": [], "intensity_changes": [], "temporal_aspects": []}

    # ------------------------------ Flow helpers ------------------------------
    def _determine_flow_pattern_type(self, intensities: List[str]) -> str:
        if len(intensities) < 2: return "stable"
        values = [{"low": 1, "medium": 2, "high": 3}.get(i, 2) for i in intensities]
        diffs = []
        for i in range(1, len(values)):
            diffs.append(values[i] - values[i-1])
        if all(d > 0 for d in diffs): return "increasing"
        if all(d < 0 for d in diffs): return "decreasing"
        if all(d == 0 for d in diffs): return "stable"
        if len(set(diffs)) > 2: return "fluctuating"
        return "mixed"

    def _analyze_emotion_sequences(self, flow_patterns: List[Dict]) -> List[Dict]:
        seqs: List[Dict[str, Any]] = []
        if not flow_patterns: return seqs
        cur = {"pattern_types": [], "duration": 0, "transitions_count": 0}
        for p in flow_patterns:
            cur["pattern_types"].append(p["pattern_type"])
            cur["duration"] += len(p.get("intensities", []))
            cur["transitions_count"] += len(p.get("transitions", []))
            if len(cur["pattern_types"]) >= 3:
                seqs.append({
                    "sequence_type": self._classify_sequence_type(cur["pattern_types"][-3:]),
                    "patterns": cur["pattern_types"][-3:],
                    "duration": cur["duration"],
                    "transitions_count": cur["transitions_count"],
                })
        return seqs

    def _classify_sequence_type(self, patterns: List[str]) -> str:
        ps = "_".join(patterns)
        if "increasing" in patterns and "decreasing" in patterns: return "volatile"
        if patterns.count("stable") >= 2: return "consistent"
        if ps.startswith("increasing"): return "progressive"
        if ps.startswith("decreasing"): return "regressive"
        return "mixed"

    def _determine_next_stage_from_transition(self, strongest_transition: Dict[str, Any], current_stage: str) -> str:
        try:
            nxt = strongest_transition.get("to_emotion") or strongest_transition.get("to")
            return self._norm_stage(nxt) if nxt else current_stage
        except Exception:
            return current_stage

    def _determine_next_stage_from_intensity(self, intensity: str) -> str:
        return self._map_intensity_to_stage(intensity)

    # ------------------------------ Sentence split ------------------------------
    def _split_text_into_sentences(self, text: str) -> List[str]:
        try:
            txt = self._norm_txt(text or "")
            if self.kss:
                return [s.strip() for s in self.kss.split_sentences(txt) if s and s.strip()]
            out = re.split(r'(?<=[.!?…])["”’)\]]*\s+|[\r\n]+', txt)
            return [s.strip() for s in out if s and s.strip()]
        except Exception as e:
            logger.error(f"[EPSA:_split_text_into_sentences] 오류: {e}")
            return [t.strip() for t in (text or "").split(".") if t.strip()]

    # ---------------------------- Contextual factors ----------------------------
    def _extract_contextual_factors(self, sentence: str) -> list:
        factors = []
        try:
            patterns = {
                "temporal": {
                    "absolute": r"\d{1,2}시\s*\d{0,2}분?",
                    "relative": r"(아침|점심|저녁|밤|새벽|정오|자정)",
                    "duration": r"(\d+\s*(시간|분|초))",
                    "sequence": r"(먼저|이후|마지막|이전|다음)",
                },
                "spatial": {
                    "specific": r"(서울|부산|대구|인천|광주|대전|울산|세종)",
                    "general": r"(시|군|구|동|읍|면)",
                    "place": r"(학교|회사|공원|카페|식당|도서관)",
                },
                "causal": {
                    "cause": r"(때문에|로 인해|덕분에)",
                    "effect": r"(그래서|따라서|결과적으로)",
                    "condition": r"(만약|[가-힣]+면|[가-힣]+다면)",
                },
                "change": {
                    "gradual": r"(점점|차츰|서서히)",
                    "sudden": r"(갑자기|돌연|순간)",
                    "repetitive": r"(자주|계속|반복)",
                },
            }
            # 라벨 메타의 global_markers 우선 병합(도메인 적응)
            try:
                g = (self.emotions_data.get("analysis_modules", {})  # type: ignore
                     .get("situation_analyzer", {}) or {}).get("global_markers", {}) or {}
                for cat, sub in (g.items() if isinstance(g, dict) else []):
                    for subcat, terms in (sub or {}).items():
                        if isinstance(terms, list) and terms:
                            pat = "|".join(map(re.escape, terms))
                            existing = patterns.get(cat, {}).get(subcat, "")
                            combined = f"({pat})" if not existing else f"({pat})|({existing})"
                            patterns.setdefault(cat, {})[subcat] = combined
            except Exception:
                pass

            for cat, sub in patterns.items():
                for subcat, rx in sub.items():
                    for m in re.finditer(rx, sentence):
                        factors.append({
                            "type": cat, "subtype": subcat,
                            "text": m.group(), "position": m.start(),
                            "context": sentence[max(0, m.start() - 10) : m.end() + 10],
                        })
            if len(factors) > 1:
                self._analyze_factor_relationships(factors)
            return factors
        except Exception as e:
            logger.error(f"[EPSA:_extract_contextual_factors] 오류: {e}")
            return []

    def _analyze_factor_relationships(self, factors: list) -> None:
        try:
            for i, f1 in enumerate(factors):
                for f2 in factors[i + 1 :]:
                    if abs(f1["position"] - f2["position"]) < 20:
                        if f1["type"] == "temporal" and f2["type"] == "spatial":
                            f1["related_to"] = f2
                            f2["related_to"] = f1
                        elif f1["type"] == "causal" and f2["type"] == "change":
                            f1["causes"] = f2
                            f2["caused_by"] = f1
        except Exception as e:
            logger.error(f"[EPSA:_analyze_factor_relationships] 오류: {e}")

    # ------------------------------- Semantic match ------------------------------
    def _find_stage_sentence(self, sentences: list, sentence_embeddings: list, description: str, threshold: float = 0.7):
        try:
            if sentence_embeddings and self.embedding_model:
                desc_embedding = self.embedding_model.encode([description], show_progress_bar=False)[0]
                best_score, best_idx = 0.0, None
                semantic_matches = []
                for idx, emb in enumerate(sentence_embeddings):
                    denom = (np.linalg.norm(desc_embedding) * np.linalg.norm(emb) + 1e-12)
                    cs = float(np.dot(desc_embedding, emb) / denom)
                    if cs > threshold * 0.8:
                        semantic_matches.append((idx, cs))
                    if cs > best_score:
                        best_score, best_idx = cs, idx
                if semantic_matches:
                    pick = self._select_best_contextual_match(sentences, semantic_matches, description) or 0
                    return sentences[pick], pick
                if best_score >= threshold and best_idx is not None:
                    return sentences[best_idx], best_idx
            keys = [k for k in re.split(r"\s+", description.strip()) if k]
            for idx, s in enumerate(sentences):
                m = sum(1 for k in keys if k in s)
                if keys and (m / len(keys)) >= 0.5:
                    return s, idx
            return None, None
        except Exception as e:
            logger.error(f"[EPSA:_find_stage_sentence] 오류: {e}")
            return None, None

    def _select_best_contextual_match(self, sentences: list, semantic_matches: list, description: str):
        try:
            best_idx, best_score = None, 0.0
            for idx, sim in semantic_matches:
                ctx = 0.0
                if idx > 0 and any(w in sentences[idx - 1] for w in self._fallback_transitions.get("temporal", [])):
                    ctx += 0.2
                if idx < len(sentences) - 1 and any(w in sentences[idx + 1] for w in self._fallback_transitions.get("cause", [])):
                    ctx += 0.2
                final = sim * 0.7 + ctx * 0.3
                if final > best_score:
                    best_idx, best_score = idx, final
            return best_idx
        except Exception as e:
            logger.error(f"[EPSA:_select_best_contextual_match] 오류: {e}")
            return None


# =============================================================================
# SituationContextMapper — FIX: add helpers + stable index keys + matcher wiring
# =============================================================================
class SituationContextMapper:
    def __init__(self, emotions_data: Dict[str, Any], *, config: Optional[Dict[str, Any]] = None):
        self.kss = kss if "kss" in globals() else None
        self.emotions_data = emotions_data or {}
        self.config = config or {}
        self.params = {
            "regex_shard_size": int(os.environ.get("SIT_REGEX_SHARD_SIZE", "512")),
            "min_conf_to_emit": float(os.environ.get("SIT_MIN_CONF", "0.10")),
            "top_k": int(os.environ.get("SIT_TOPK", "50")),
        }
        # --- Top-K harmonization: Mapper와 Orchestrator 간 ENV 정합 ---
        _env_topk = os.environ.get("SIT_TOPK") or os.environ.get("SIT_ORCH_TOPK")
        try:
            _env_topk = int(_env_topk) if _env_topk is not None else None
        except Exception:
            _env_topk = None
        if _env_topk is not None:
            self.params["top_k"] = _env_topk
            self.params["top_k_cards"] = _env_topk
            self.params["topk_phrases"] = _env_topk
        else:
            # 기본값 보강(없으면 50으로)
            self.params["top_k"] = int(self.params.get("top_k", self.params.get("top_k_cards", 50)))
        # Regex Shard Size도 폴백 허용
        _env_shard = os.environ.get("SIT_REGEX_SHARD_SIZE") or os.environ.get("SIT_REGEX_SHARDSIZE")
        if _env_shard:
            try:
                self.params["regex_shard_size"] = int(_env_shard)
            except Exception:
                pass
        # --- [ADD] Matching tune knobs & aliases (schema-free) ---
        self.tune = {
            "contain_thr_short": float(os.environ.get("SIT_CONTAIN_THR_SHORT", "0.50")),
            "contain_thr_long":  float(os.environ.get("SIT_CONTAIN_THR_LONG",  "0.40")),
            "fuzzy_use":         bool(int(os.environ.get("SIT_FUZZY_USE", "1"))),
            "fuzzy_max_gap":     int(os.environ.get("SIT_FUZZY_MAX_GAP", "2")),
            "fuzzy_max_len":     int(os.environ.get("SIT_FUZZY_MAX_LEN", "24")),
        }
        self.aliases = self._build_alias_map(self.emotions_data)
        # RegexShardMatcher 사용여부
        try:
            _ = RegexShardMatcher  # type: ignore
            self._rx_available = True
        except Exception:
            self._rx_available = False

        self._index = self._build_situation_index(self.emotions_data)
        self._matchers = self._build_situation_matchers(self._index)

        # Calibrator hook (optional)
        self.calibrator = None

        # 전역 프레이즈 → sid 인덱스 및 샤드 매처(Top-K 프리필터용)
        self._phrase2sid = defaultdict(set)
        bank_kw, bank_var, bank_ex, bank_trg, bank_stage = [], [], [], [], []
        try:
            for sid, si in self._index.items():
                for w in si.get("keywords", []) or []:
                    self._phrase2sid[self._norm_txt(w)].add(sid); bank_kw.append(w)
                for w in si.get("variations", []) or []:
                    self._phrase2sid[self._norm_txt(w)].add(sid); bank_var.append(w)
                for w in si.get("examples", []) or []:
                    self._phrase2sid[self._norm_txt(w)].add(sid); bank_ex.append(w)
                for w in si.get("triggers", []) or []:
                    self._phrase2sid[self._norm_txt(w)].add(sid); bank_trg.append(w)
                for st in ("stage_trigger", "stage_development", "stage_peak", "stage_aftermath"):
                    for w in si.get(st, []) or []:
                        self._phrase2sid[self._norm_txt(w)].add(sid); bank_stage.append(w)
            shard = int(self.params["regex_shard_size"])
            if self._rx_available:
                self._rx_all = {
                    "kw":    RegexShardMatcher(bank_kw,    shard_size=shard, ignore_space=True),
                    "var":   RegexShardMatcher(bank_var,   shard_size=shard, ignore_space=True),
                    "ex":    RegexShardMatcher(bank_ex,    shard_size=shard, ignore_space=True),
                    "trg":   RegexShardMatcher(bank_trg,   shard_size=shard, ignore_space=True),
                    "stage": RegexShardMatcher(bank_stage, shard_size=shard, ignore_space=True),
                }
            else:
                self._rx_all = {}
        except Exception:
            # 안전 폴백
            self._rx_all = {}

    # Calibrator 주입 API
    def inject_calibrator(self, calibrator) -> None:
        self.calibrator = calibrator

    # ----------------------------- Public API -----------------------------
    def map(self, text: str) -> Dict[str, Any]:
        txt = self._norm_txt(text or "")
        if not txt:
            return {"identified_situations": [], "context_mapping": {}}
        sents = self._split_sentences(txt)

        out_cards, ctx_map = [], {}

        # 후보 sid 상위 K 선별(전역 1회 스캔)
        joined = "".join(sents)
        cand = Counter()
        if getattr(self, "_rx_available", False) and getattr(self, "_rx_all", None):
            norm_joined = self._norm_txt(joined)
            for tag in ("kw", "var", "ex", "trg", "stage"):
                matcher = self._rx_all.get(tag)
                if matcher:
                    for hit in matcher.find_all(norm_joined):
                        for sid_c in self._phrase2sid.get(self._norm_txt(hit), ()):
                            cand[sid_c] += 1
        TOPK = max(1, int(self.params.get("top_k", 50)))
        sid_iter = [sid for sid, _ in cand.most_common(TOPK)] if cand else list(self._index.keys())

        for sid in sid_iter:
            si = self._index[sid]
            # 증거 수집
            matched_kw, matched_var, matched_ex, matched_trigs, matched_stages = [], [], [], [], []
            # 간단 증거 가중치
            w_kw, w_var, w_ex, w_trg, w_stg = 0.08, 0.05, 0.10, 0.12, 0.10
            type_cap = {"keyword": 0.40, "variation": 0.25, "example": 0.40, "trigger": 0.36, "progression": 0.30}

            e_sum = {"keyword": 0.0, "variation": 0.0, "example": 0.0, "trigger": 0.0, "progression": 0.0}
            for sent in sents:
                j = re.sub(r"\s+", "", self._norm_txt(sent))
                for w in self._find_all(si, "kw", j, sid):   # keywords
                    if w not in matched_kw:
                        matched_kw.append(w); e_sum["keyword"] = min(type_cap["keyword"], e_sum["keyword"] + w_kw)
                for w in self._find_all(si, "var", j, sid):  # variations
                    if w not in matched_var:
                        matched_var.append(w); e_sum["variation"] = min(type_cap["variation"], e_sum["variation"] + w_var)
                for w in self._find_all(si, "ex", j, sid):   # examples
                    if w not in matched_ex:
                        matched_ex.append(w); e_sum["example"] = min(type_cap["example"], e_sum["example"] + w_ex)
                for w in self._find_all(si, "trg", j, sid):  # triggers
                    if w not in matched_trigs:
                        matched_trigs.append(w); e_sum["trigger"] = min(type_cap["trigger"], e_sum["trigger"] + w_trg)
                for stg in ("trigger", "development", "peak", "aftermath"):
                    phs = self._find_all(si, f"stage_{stg}", j, sid)
                    if phs:
                        matched_stages.append(stg)
                        e_sum["progression"] = min(type_cap["progression"], e_sum["progression"] + w_stg)

            # 최종 신뢰도(간단 합)
            conf = float(sum(e_sum.values()))

            # (옵션) 캘리브레이터 기반 자동 보정
            if getattr(self, "calibrator", None):
                try:
                    log_adj = 0.0
                    emo_key = si.get("sub_emotion_id") or si.get("sub_emotion_name")
                    for term in (matched_kw + matched_var + matched_ex + matched_trigs):
                        log_adj += float(self.calibrator.get_pattern_weight(str(emo_key), term))
                    log_adj += float(self.calibrator.get_prior_adj(str(emo_key)))
                    conf = float(min(1.0, max(0.0, conf * math.exp(max(-1.5, min(1.5, log_adj))))))  # clip & exp scaling
                except Exception:
                    pass

            if conf >= self.params["min_conf_to_emit"]:
                out_cards.append({
                    "situation_id": sid,
                    "primary_emotion": si["primary_emotion"],
                    "sub_emotion_name": si["sub_emotion_name"],
                    "situation_name": si["situation_name"],
                    "confidence": round(conf, 3),
                })
                ctx_map[sid] = {
                    "primary_emotion": si["primary_emotion"],
                    "sub_emotion_name": si["sub_emotion_name"],
                    "situation_name": si["situation_name"],
                    "matched_keywords": matched_kw,
                    "matched_variations": matched_var,
                    "matched_examples": matched_ex,
                    "matched_triggers": matched_trigs,
                    "matched_progression_stages": sorted(list(set(matched_stages))),
                    "evidence_counts": {
                        "keywords": len(matched_kw),
                        "variations": len(matched_var),
                        "examples": len(matched_ex),
                        "triggers": len(matched_trigs),
                        "stages": len(set(matched_stages)),
                    },
                    "match_score": round(conf, 3),
                }

        out_cards.sort(key=lambda x: x["confidence"], reverse=True)
        K = int(self.params.get("top_k", 50))
        if K > 0:
            out_cards = out_cards[: K]
        return {"identified_situations": out_cards, "context_mapping": ctx_map}

    # ------------------------ Index & Matchers ------------------------
    def _build_situation_index(self, data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        idx: Dict[str, Dict[str, Any]] = {}

        def k(obj: Dict[str, Any], *alts: str) -> Any:
            lower = {str(kk).lower(): kk for kk in obj.keys()}
            for nm in alts:
                if nm.lower() in lower:
                    return obj[lower[nm.lower()]]
            return None

        for primary, emo in (data or {}).items():
            if not isinstance(emo, dict):
                continue
            subs = emo.get("sub_emotions", {}) or (emo.get("emotion_profile", {}) or {}).get("sub_emotions", {})
            if not isinstance(subs, dict):
                continue
            for sub_name, sub in subs.items():
                if not isinstance(sub, dict):
                    continue
                sub_id = str((sub.get("metadata") or {}).get("emotion_id") or "").strip()
                if not sub_id:
                    continue
                ctxp = sub.get("context_patterns", {}) or (sub.get("emotion_profile", {}) or {}).get("context_patterns", {})
                sits = ctxp.get("situations", {})
                if not isinstance(sits, dict):
                    continue

                # intensity 힌트(옵션)
                intensity = self._norm_intensity(sub.get("intensity") or (sub.get("metadata") or {}).get("intensity") or "medium")

                for sit_key, sit in sits.items():
                    if not isinstance(sit, dict):
                        continue
                    sid = f"{sub_id}:{sit_key}"
                    keywords   = self._dedup_list(sit.get("keywords", []))
                    variations = self._dedup_list(sit.get("variations", []))
                    examples   = self._dedup_list(sit.get("examples", []))
                    triggers   = self._dedup_list(sit.get("triggers", []))
                    # --- [ADD] 라벨 불변: 런타임 파생 변형/별칭 주입 ---
                    if bool(int(os.environ.get("SIT_GEN_MORPH", "1"))):
                        gen = set()
                        for base in (keywords + variations):
                            gen.update(self._ko_variants(base))
                        variations = self._dedup_list(list(set(variations) | gen))
                    # 별칭 합치기(라벨에 없으면 fallback만)
                    if sit_key in self.aliases:
                        variations = self._dedup_list(list(set(variations) | set(self.aliases[sit_key])))

                    stages_map = {"trigger": [], "development": [], "peak": [], "aftermath": []}
                    prog = sit.get("emotion_progression", {})
                    if isinstance(prog, dict):
                        for stg, exs in prog.items():
                            ns = self._norm_stage(stg)
                            stages_map.setdefault(ns, []).extend(list(self._iter_phrases(exs)))

                    idx[sid] = {
                        "situation_id": sid,  # <-- 매처 참조를 위해 명시 저장
                        "primary_emotion": primary,
                        "sub_emotion_name": sub_name,
                        "sub_emotion_id": sub_id,
                        "situation_name": sit_key,
                        "core_concept": str(sit.get("core_concept", sit_key)).strip(),
                        "intensity": intensity,
                        "keywords": keywords,
                        "variations": variations,
                        "examples": examples,
                        "triggers": triggers,
                        "stage_trigger": self._dedup_list(stages_map["trigger"]),
                        "stage_development": self._dedup_list(stages_map["development"]),
                        "stage_peak": self._dedup_list(stages_map["peak"]),
                        "stage_aftermath": self._dedup_list(stages_map["aftermath"]),
                        "raw_situation": sit,
                    }
        return idx

    def _build_situation_matchers(self, idx: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        matchers: Dict[str, Dict[str, Any]] = {}
        if not self._rx_available:
            return matchers
        shard = int(self.params["regex_shard_size"])

        def rx(phrases: List[str]):
            return RegexShardMatcher(phrases, shard_size=shard, boundary=False, ignore_space=True) if phrases else None

        for sid, si in idx.items():
            matchers[sid] = {
                "kw": rx(si.get("keywords", [])),
                "var": rx(si.get("variations", [])),
                "ex":  rx(si.get("examples", [])),
                "trg": rx(si.get("triggers", [])),
                "stage_trigger":     rx(si.get("stage_trigger", [])),
                "stage_development": rx(si.get("stage_development", [])),
                "stage_peak":        rx(si.get("stage_peak", [])),
                "stage_aftermath":   rx(si.get("stage_aftermath", [])),
            }
        return matchers

    # -------------------------- Matching helper --------------------------
    def _find_all(self, si: Dict[str, Any], field: str, joined_text: str, sid: Optional[str]) -> List[str]:
        """
        상황별 매처/폴백으로 일치 항목 추출.
        joined_text: 공백 제거된 문장
        sid: situation_id (인덱스 키)
        """
        out: List[str] = []
        if self._rx_available and sid and sid in self._matchers:
            rx = self._matchers[sid].get(field)
            if rx:
                return rx.find_all(joined_text)

        # 폴백: 목록 직접 부분매칭
        phrases = []
        if field == "kw":   phrases = si.get("keywords", [])
        elif field == "var": phrases = si.get("variations", [])
        elif field == "ex":  phrases = si.get("examples", [])
        elif field == "trg": phrases = si.get("triggers", [])
        elif field.startswith("stage_"): phrases = si.get(field, [])

        for p in phrases:
            p2 = re.sub(r"\s+", "", self._norm_txt(p))
            if not p2:
                continue
            # L0: substring (공백 제거, NFKC)
            if p2 in joined_text:
                out.append(p); continue
            # L3: containment (char bi-gram)
            if self._containment_hit(joined_text, p2,
                                     self.tune["contain_thr_short"], self.tune["contain_thr_long"]):
                out.append(p); continue
            # L4: gap-fuzzy (경량 편집 보정)
            if self.tune["fuzzy_use"] and self._fuzzy_hit(joined_text, p2,
                                                          self.tune["fuzzy_max_gap"], self.tune["fuzzy_max_len"]):
                out.append(p)
        return out

    # --- [ADD] aliases loader (라벨 스키마 불변; 있으면 흡수, 없으면 안전 기본셋) ---
    def _build_alias_map(self, emotions_data: Dict[str, Any]) -> Dict[str, Set[str]]:
        out: Dict[str, Set[str]] = {}
        try:
            # 있으면 사용 (스키마에 없더라도 안전)
            ali = (emotions_data.get("situations") or {}).get("aliases", {})
            if isinstance(ali, dict):
                for k, vs in ali.items():
                    if isinstance(vs, (list, set, tuple)):
                        out[str(k)] = {str(x).strip() for x in vs if isinstance(x, str) and x.strip()}
        except Exception:
            pass
        # 안전 기본 별칭(필요 시 확장 가능; 라벨 파일은 변경하지 않음)
        fallback = {
            "갈등": {"충돌", "분쟁", "언쟁", "대립"},
            "축하": {"경축", "치하", "파티"},
            "상실": {"이별", "슬픔", "상심"},
            "협상": {"교섭", "조율", "타협"},
        }
        for k, vs in fallback.items():
            out.setdefault(k, set()).update(vs)
        return out

    # --- [ADD] 한국어 조사/어미 변형 파생 (라벨 변경 없이 매칭만 강화) ---
    def _ko_variants(self, s: str) -> List[str]:
        s = str(s or "").strip()
        if not s:
            return []
        bases = {s, re.sub(r"\s+", "", s)}
        tails = ("을","를","은","는","이","가","과","와","도","만","의","에","에서","으로","로",
                 "에게","께서","께","한테","밖에","조차","마저","까지")
        for t in tails:
            if s.endswith(t) and len(s) > len(t):
                stem = s[: -len(t)]
                bases.update({stem, re.sub(r"\s+", "", stem)})
        return [x for x in bases if x and x != s]

    # --- [ADD] 경량 유사도/퍼지 매칭 (Jaccard containment + gap fuzzy) ---
    def _containment_hit(self, text_norm: str, token_norm: str, thr_short: float, thr_long: float) -> bool:
        if not token_norm or not text_norm:
            return False
        # char bi-grams containment: |P ∩ T| / |P|
        def _bg(z): return {z[i:i+2] for i in range(len(z)-1)} if len(z) > 1 else set()
        pb = _bg(token_norm)
        if not pb:
            return False
        tb = _bg(text_norm)
        contain = len(pb & tb) / max(1, len(pb))
        thr = thr_short if len(token_norm) <= 5 else thr_long
        return contain >= thr

    def _fuzzy_hit(self, text_norm: str, token_norm: str, max_gap: int, max_len: int) -> bool:
        if not token_norm or not text_norm:
            return False
        if len(token_norm) > max_len:
            return False
        step = max(2, len(token_norm)//3)
        parts = [re.escape(token_norm[i:i+step]) for i in range(0, len(token_norm), step)]
        pat = ".*?".join(parts) if parts else ""
        return bool(pat and re.search(pat, text_norm))

    # ------------------------------ Utils ------------------------------
    @staticmethod
    def _norm_txt(s: str) -> str:
        try:
            import unicodedata as _ud
            s = _ud.normalize("NFKC", s or "")
        except Exception:
            s = s or ""
        return s

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        try:
            return [t.strip() for t in kss.split_sentences(text) if t and t.strip()]
        except Exception:
            return [t.strip() for t in re.split(r'(?<=[.!?…])["”’)\]]*\s+|[\r\n]+', text) if t and t.strip()]

    @staticmethod
    def _norm_intensity(v: Any, default: str = "medium") -> str:
        s = str(v).strip().lower() if v is not None else ""
        if s in {"low", "lo", "약함", "weak"}: return "low"
        if s in {"medium", "med", "mid", "중간"}: return "medium"
        if s in {"high", "hi", "강함", "strong"}: return "high"
        return default

    @staticmethod
    def _norm_stage(name: Any) -> str:
        n = str(name).strip().lower() if name else ""
        alias = {
            "onset": "trigger", "start": "trigger", "trigger": "trigger",
            "build": "development", "build-up": "development", "development": "development",
            "climax": "peak", "apex": "peak", "peak": "peak",
            "aftermath": "aftermath", "resolution": "aftermath", "cooldown": "aftermath",
        }
        return alias.get(n, n or "trigger")

    @staticmethod
    def _iter_phrases(node: Any):
        if not node:
            return
        if isinstance(node, str):
            s = node.strip()
            if s:
                yield s
            return
        if isinstance(node, (list, tuple, set)):
            for v in node:
                if isinstance(v, str) and v.strip():
                    yield v.strip()
        elif isinstance(node, dict):
            for k in ("text", "phrase", "value", "token"):
                v = node.get(k)
                if isinstance(v, str) and v.strip():
                    yield v.strip()

    @staticmethod
    def _dedup_list(xs: List[str]) -> List[str]:
        out, seen = [], set()
        for s in xs or []:
            s2 = str(s).strip()
            if s2 and s2 not in seen:
                seen.add(s2); out.append(s2)
        return out


# =============================================================================
# SituationContextOrchestrator — mapper+EPSA 합성, 최종 리포트 생성 (독립형)
# =============================================================================
class SituationContextOrchestrator:
    """
    이 클래스는 두 축을 합쳐 최종 리포트를 만듭니다.
      1) SituationContextMapper  → 상황 식별 + 컨텍스트 매핑(증거 기반)
      2) EmotionProgressionSituationAnalyzer(EPSA) → 감정 흐름/전이/강도/시공간 표지

    출력은 샘플 JSON 스키마와 호환:
      {
        identified_situations: [...],
        context_mapping: {...},
        situation_metrics: {...},
        emotional_context: {...},
        spatiotemporal_context: {...},
        situational_triggers: [...],
        metrics: {...}
      }
    """

    def __init__(
        self,
        emotions_data: Dict[str, Any],
        *,
        config: Optional[Dict[str, Any]] = None,
        embedding_model: Any = None,
    ):
        self.emotions_data = emotions_data or {}
        self.config = config or {}
        self.mapper = SituationContextMapper(self.emotions_data, config=self.config)
        self.epsa = EmotionProgressionSituationAnalyzer(
            embedding_model=embedding_model, emotions_data=self.emotions_data
        )

        # 독립 파라미터(ENV/설정)
        self.params = {
            "top_k_cards": int(os.environ.get("SIT_ORCH_TOPK", "50")),
            "card_alpha": float(os.environ.get("SIT_CARD_ALPHA", "0.6")),  # mapper.conf vs match_score 조합
            "min_emit_conf": float(os.environ.get("SIT_ORCH_MIN_CONF", "0.10")),
        }

    # --- [ADD] 간이 정서 신호 추정(텍스트 기반) ---
    def _infer_simple_signals(self, text: str) -> Dict[str, Any]:
        t = (_nfkc_no_space(text or "") or "").lower()
        pos_hint = any(k in t for k in ("축하","경축","고맙","감사","행복","기쁨"))
        neg_hint = any(k in t for k in ("화가","분노","짜증","불안","슬픔","상실","후회"))
        amp_hint = ("!!" in t) or ("??" in t) or ("ㅠㅠ" in t) or ("ㅋㅋ" in t) or ("ㅎㅎ" in t)
        return {"valence": (1 if pos_hint and not neg_hint else (-1 if neg_hint and not pos_hint else 0)),
                "amplified": bool(amp_hint)}

    # --- [ADD] 정서 신호/카테고리 충돌 시 가중 하향 ---
    def _penalty_from_signals(self, primary_emotion: str, text: str) -> Tuple[float, Optional[str]]:
        # 희/락=+1, 애/노/우=-1, 그 외=0
        pe = str(primary_emotion or "")
        expected = 0
        if pe in ("희","락"): expected = +1
        elif pe in ("애","노","우"): expected = -1
        sig = self._infer_simple_signals(text)
        penalty = 1.0; notes: List[str] = []
        if expected != 0 and sig["valence"] != 0 and (expected != sig["valence"]):
            penalty *= float(os.environ.get("SIT_VALENCE_PENALTY", "0.60")); notes.append("VALENCE_MISMATCH")
        if expected == 0 and sig["amplified"]:
            penalty *= float(os.environ.get("SIT_OVEREXCITED_PENALTY", "0.85")); notes.append("OVER_EXCITED")
        return max(0.4, min(1.0, penalty)), (";".join(notes) if notes else None)

    # --- [ADD] 텍스트에서 직접 감정 추론 (폴백용) ---
    def _infer_emotion_from_text(self, text: str) -> Dict[str, float]:
        """
        텍스트에서 키워드 기반으로 감정 분포를 추론합니다.
        emotion_flows가 비어있을 때 폴백으로 사용됩니다.
        """
        t = (_nfkc_no_space(text or "") or "").lower()
        scores: Dict[str, float] = {}
        
        # 감정별 키워드 패턴 (확장 가능)
        emotion_keywords = {
            "노": [
                "불만", "화", "짜증", "분노", "화나", "열받", "억울", "답답",
                "인상", "갱신", "비싸", "부당", "불공정", "항의", "불쾌",
                "싫", "미움", "혐오", "거부", "안되", "못하", "왜", "어이없",
            ],
            "애": [
                "슬프", "슬픔", "우울", "눈물", "아쉬", "그리", "외로",
                "상실", "이별", "후회", "미안", "안타깝", "걱정", "불안",
            ],
            "희": [
                "기쁘", "행복", "좋아", "사랑", "감사", "축하", "즐거",
                "희망", "설레", "감동", "뿌듯", "만족", "다행",
            ],
            "락": [
                "재미", "신나", "웃", "즐거", "흥미", "활기", "유쾌",
            ],
            "욕": [
                "원해", "원하", "바라", "갖고싶", "되고싶", "하고싶",
            ],
            "오": [
                "오만", "자만", "자부", "뻐기", "자랑",
            ],
            "애증": [
                "복잡", "애증", "사랑미움",
            ],
        }
        
        for emo, keywords in emotion_keywords.items():
            count = sum(1 for kw in keywords if kw in t)
            if count > 0:
                scores[emo] = min(0.9, 0.3 + count * 0.15)
        
        # 정규화
        total = sum(scores.values()) or 1.0
        return {k: round(v / total, 3) for k, v in scores.items()} if scores else {}

    # ------------------------------- Public API -------------------------------
    def analyze(self, text: str) -> Dict[str, Any]:
        """
        텍스트 1건에 대해 최종 리포트를 생성.
        """
        t0 = time.time()
        try:
            # [NEW] Junk-Guard (라벨/의존성 무관, 구조 출력 보장)
            is_junk, why = _sit_is_junk_text(text)
            if is_junk:
                m = {"processing_time": round(time.time() - t0, 3),
                     "memory_usage_kb": self._get_memory_kb(),
                     "situations_identified": 0,
                     "success_rate": False,
                     "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                     "warnings": [f"JUNK_INPUT:{why}"]}
                return {
                    "identified_situations": [],
                    "context_mapping": {},
                    "situation_metrics": {},
                    "emotional_context": {},
                    "spatiotemporal_context": {},
                    "situational_triggers": [],
                    "metrics": m,
                }

            # 1) 상황 식별/컨텍스트 매핑
            m = self.mapper.map(text)
            context_mapping: Dict[str, Dict[str, Any]] = m.get("context_mapping", {}) or {}
            # 카드 조정에서 원문 텍스트 활용
            for _sid in list(context_mapping.keys()):
                context_mapping[_sid]["full_text"] = text

            # 2) EPSA: 진행/전이/강도/시공간
            e = self.epsa.analyze(text, self.emotions_data)

            # 3) 감정 컨텍스트 정리
            emotional_context = self._compose_emotional_context(e)

            # 4) 상황 카드 합성(신뢰도 정규화 + 신호 패널티 적용)
            identified_situations = self._synthesize_cards(m.get("identified_situations", []), context_mapping, emotional_context)

            # ★★★ NEW: 키워드 매칭 결과가 부족하면 감정 기반 상황 추론 적용 ★★★
            if len(identified_situations) < 2:
                # 감정 분포 계산 (emotion_flows에서 추출)
                flows = emotional_context.get("emotion_flows") or []
                dom_counts: Dict[str, float] = {}
                for flow in flows:
                    dom = flow.get("dominant_emotion", {})
                    emo = dom.get("emotion") if isinstance(dom, dict) else None
                    score = dom.get("score", 0.5) if isinstance(dom, dict) else 0.5
                    if emo and emo != "none":
                        # 주 감정 추출 (예: "노-분노-억울함" → "노")
                        primary = emo.split("-")[0] if "-" in emo else emo
                        dom_counts[primary] = dom_counts.get(primary, 0) + score
                
                # 정규화하여 분포 생성
                total = sum(dom_counts.values()) or 1.0
                main_dist = {k: round(v / total, 3) for k, v in dom_counts.items()}
                
                # EPSA 결과에서 추가 정보 추출
                epsa_dist = (e.get("emotional_context") or {}).get("emotion_distribution") or {}
                if epsa_dist and not main_dist:
                    main_dist = epsa_dist
                
                # ★★★ 폴백: 감정 분포가 비어있으면 텍스트에서 직접 감정 추론 ★★★
                if not main_dist:
                    main_dist = self._infer_emotion_from_text(text)
                    logger.debug("[Orchestrator] 텍스트 기반 감정 추론: %s", main_dist)
                
                # 감정 분석 결과 구성
                emotion_results = {
                    "main_distribution": main_dist,
                    "emotion_intensity": emotional_context.get("emotion_intensities") or {},
                    "detected_emotions": emotional_context.get("dominant_emotions") or [],
                }
                
                logger.debug(
                    "[Orchestrator] 감정 분포 추출: %s", main_dist
                )
                
                # 감정 기반 상황 추론
                inferred = infer_situations_from_emotion_data(
                    emotion_results,
                    self.emotions_data,
                    min_confidence=0.30,  # 임계값 낮춤
                    max_situations=5,
                )
                
                # 기존 결과와 병합
                if inferred:
                    identified_situations = merge_situation_results(
                        identified_situations,
                        inferred,
                        keyword_weight=0.7,
                        inference_weight=0.5,
                        max_results=5,
                    )
                    logger.debug(
                        "[Orchestrator] 감정 기반 상황 추론 적용: %d개 추론됨",
                        len(inferred)
                    )

            # 5) 시공간 컨텍스트
            spatiotemporal_context = self._compose_spatiotemporal(e, text)

            # 6) 상황 트리거 (컨텍스트 매핑 기반)
            situational_triggers = self._compose_situational_triggers(context_mapping, identified_situations)

            # 7) 상황 메트릭
            situation_metrics = self._compute_situation_metrics(
                identified_situations, emotional_context, spatiotemporal_context
            )

            # 8) 시스템 메트릭
            metrics = self._compute_metrics(t0, len(identified_situations))

            result = {
                "identified_situations": identified_situations,
                "context_mapping": context_mapping,
                "situation_metrics": situation_metrics,
                "emotional_context": emotional_context,
                "spatiotemporal_context": spatiotemporal_context,
                "situational_triggers": situational_triggers,
                "metrics": metrics,
            }
            # Micro-Patch 2-4: STRICT 모드에서 카드 없음 경고 표면화
            mode = (os.environ.get("SIT_MODE") or os.environ.get("SITUATION_MODE") or "TOLERANT").upper()
            try:
                identified = result.get("identified_situations", [])
                if mode == "STRICT" and not identified:
                    result.setdefault("metrics", {}).setdefault("warnings", []).append("STRICT_NO_CARD")
            except Exception:
                pass
            return result

        except Exception as ex:
            logger.exception(f"[Orchestrator:analyze] 오류: {ex}")
            return {
                "identified_situations": [],
                "context_mapping": {},
                "situation_metrics": {},
                "emotional_context": {},
                "spatiotemporal_context": {},
                "situational_triggers": [],
                "metrics": self._compute_metrics(t0, 0),
                "error": str(ex),
            }

    # ------------------------------ Synthesizers ------------------------------
    def _synthesize_cards(self, mapper_cards: List[Dict[str, Any]], ctx_map: Dict[str, Dict[str, Any]],
                          emotional_context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        alpha = float(self.params.get("card_alpha", 0.6))
        min_conf = float(os.environ.get("SIT_MIN_CONF", self.params.get("min_emit_conf", 0.20)))
        top_k = int(os.environ.get("SIT_ORCH_TOPK", self.params.get("top_k_cards", 8)))
        min_evid = int(os.environ.get("SIT_EMIT_MIN_EVID", self.params.get("min_emit_evidence", 2)))  # ← NEW
        # STRICT 모드면 약간 보수적
        mode = (os.environ.get("SIT_MODE") or os.environ.get("SITUATION_MODE") or "TOLERANT").upper()
        if mode == "STRICT":
            min_conf = min(1.0, min_conf + 0.05)
            min_evid = max(min_evid, 3)

        cards_source = mapper_cards[:] if mapper_cards else []
        if not cards_source and ctx_map:
            for sid, mp in ctx_map.items():
                cards_source.append({
                    "situation_id": sid,
                    "primary_emotion": mp.get("primary_emotion"),
                    "sub_emotion_name": mp.get("sub_emotion_name"),
                    "situation_name": mp.get("situation_name"),
                    "confidence": 0.0,
                })

        cards: List[Dict[str, Any]] = []
        for card in cards_source or []:
            sid = card.get("situation_id")
            mp = ctx_map.get(sid, {})

            # --- NEW: 증거 최소 요건(키워드/변형/예시/트리거/단계 중 2가지 이상)
            evid_count = 0
            evid_count += 1 if (mp.get("matched_keywords")) else 0
            evid_count += 1 if (mp.get("matched_variations")) else 0
            evid_count += 1 if (mp.get("matched_examples")) else 0
            evid_count += 1 if (mp.get("matched_triggers")) else 0
            evid_count += 1 if (mp.get("matched_progression_stages")) else 0
            if evid_count < min_evid:
                continue

            match_score = float(mp.get("match_score", 0.0))
            mapper_conf = float(card.get("confidence", 0.0))
            trig_bonus = 0.02 * len(mp.get("matched_triggers", []) or [])
            prog_bonus = 0.03 if (mp.get("matched_progression_stages")) else 0.0

            conf_final = alpha * mapper_conf + (1.0 - alpha) * match_score + trig_bonus + prog_bonus
            # 신호 기반 하향 조정(텍스트 전역 특징 활용)
            penalty, note = self._penalty_from_signals(
                card.get("primary_emotion") or mp.get("primary_emotion"),
                mp.get("full_text") or ""
            )
            conf_final *= penalty
            if note:
                metrics = mp.setdefault("internal_metrics", {})
                metrics.setdefault("warnings", []).append(f"REWEIGHT:{sid}:{note}")
            conf_final = max(0.0, min(1.0, conf_final))

            # --- NEW: 캘리브레이션(상위 카드가 0.6 이상으로 보이도록 재스케일)
            # min_conf 이상인 값들을 0.6~1.0 구간으로 선형 매핑(읽기 좋은 시각적 self-calibration)
            if conf_final >= min_conf:
                # 0.6 + 0.4 * norm
                norm = (conf_final - min_conf) / max(1e-6, 1.0 - min_conf)
                conf_display = round(0.6 + 0.4 * norm, 3)
            else:
                conf_display = round(conf_final, 3)

            if conf_final >= min_conf:
                cards.append({
                    "situation_id": sid,
                    "primary_emotion": card.get("primary_emotion") or mp.get("primary_emotion"),
                    "sub_emotion_name": card.get("sub_emotion_name") or mp.get("sub_emotion_name"),
                    "situation_name": card.get("situation_name") or mp.get("situation_name"),
                    "confidence_raw": round(conf_final, 3),  # 원시 점수
                    "confidence": conf_display,              # 보정된 디스플레이 점수
                })

        cards.sort(key=lambda x: x.get("confidence", 0.0), reverse=True)
        if top_k > 0:
            cards = cards[:top_k]
        return cards

    def _compose_emotional_context(self, e: Dict[str, Any]) -> Dict[str, Any]:
        """
        EPSA 산출물을 오케스트레이터 표준 스키마로 변환.
        - progression.transitions + emotional_context.emotional_transitions 병합
        - 둘 다 없으면 flows 기반 전이 추론(flow_inferred)
        - 기본값으로 보조/마이크로 전이 숨김 및 동일 from/to 전이 제외(환경변수로 on/off 가능)
          * SIT_HIDE_FALLBACK_TRANS=1   (기본 1: 숨김)
          * SIT_ONLY_MEANINGFUL_TRANS=1 (기본 1: 의미 있는 전이만)
        """
        emotional_context = {
            "situation_emotions": {},
            "emotion_flows": [],
            "emotional_transitions": [],
            "intensity_progression": [],
            "dominant_emotions": [],
        }

        ec = (e.get("emotional_context") or {})
        ec_list = ec.get("emotional_contexts", []) or []
        flows = []
        dom_counts = defaultdict(int)

        # flows 생성
        for i, row in enumerate(ec_list):
            dom = row.get("dominant_sub_emotion") or "none"
            score = row.get("dominant_score", 0.0)
            flows.append({
                "sentence": row.get("text"),
                "dominant_emotion": {"emotion": dom, "score": round(float(score or 0.0), 3)},
                "position": row.get("index", i),
            })
            if dom and dom != "none":
                dom_counts[dom] += 1
        emotional_context["emotion_flows"] = flows

        # 1) progression.transitions 병합
        trans_merged: List[Dict[str, Any]] = []
        prog = e.get("progression") or {}
        t_prog = prog.get("transitions", []) or []
        if t_prog:
            for tr in t_prog:
                item = {
                    "from_emotion": tr.get("from_stage"),
                    "to_emotion": tr.get("to_stage"),
                    "position": tr.get("position"),
                    "confidence": tr.get("confidence"),
                    "type": tr.get("type", "shift"),
                }
                if tr.get("trigger"): item["trigger_matched"] = [tr.get("trigger")]
                trans_merged.append(item)

        # 2) EC 내부 전이 병합
        t_ec = (e.get("emotional_context") or {}).get("emotional_transitions", []) or []
        for tr in t_ec:
            item = {
                "from_emotion": tr.get("from_emotion"),
                "to_emotion": tr.get("to_emotion"),
                "position": tr.get("position"),
                "confidence": tr.get("confidence"),
                "type": tr.get("type", None),
            }
            if tr.get("trigger"): item["trigger_matched"] = [tr.get("trigger")]
            trans_merged.append(item)

        # 3) flows 기반 추론(없을 때만)
        if not trans_merged and flows:
            for i in range(1, len(flows)):
                a, b = flows[i - 1]["dominant_emotion"], flows[i]["dominant_emotion"]
                if (a and b) and (a.get("emotion") not in (None, "none")) and (b.get("emotion") not in (None, "none")):
                    if a["emotion"] != b["emotion"]:
                        conf = round(min(float(a.get("score", 0.0)), float(b.get("score", 0.0))), 3)
                        if conf > 0.0:
                            trans_merged.append({
                                "from_emotion": a["emotion"],
                                "to_emotion": b["emotion"],
                                "position": i,
                                "confidence": conf,
                                "type": "flow_inferred"
                            })

        # 4) 전이 표시 정책(ENV 제어)
        hide_fallback = os.environ.get("SIT_HIDE_FALLBACK_TRANS", "1") != "0"
        only_meaningful = os.environ.get("SIT_ONLY_MEANINGFUL_TRANS", "1") != "0"

        filtered = []
        seen = set()
        for t in trans_merged:
            key = (t.get("position"), t.get("from_emotion"), t.get("to_emotion"))
            if key in seen:
                continue
            seen.add(key)

            # 보조/마이크로/동일 from→to 숨김
            if hide_fallback:
                if t.get("type") in ("micro_shift", "fallback"):
                    continue
                if t.get("from_emotion") == t.get("to_emotion"):
                    continue

            if only_meaningful:
                if t.get("from_emotion") == t.get("to_emotion"):
                    continue

            filtered.append(t)

        emotional_context["emotional_transitions"] = filtered

        # 5) intensity_progression
        if ec.get("intensity_progression"):
            emotional_context["intensity_progression"] = ec["intensity_progression"]
        else:
            tmp = []
            for i, f in enumerate(flows):
                sc = float((f["dominant_emotion"] or {}).get("score") or 0.0)
                lv = "high" if sc >= 0.7 else ("medium" if sc >= 0.4 else "low")
                tmp.append({"position": i, "intensity": lv, "score": sc})
            emotional_context["intensity_progression"] = tmp

        # 6) dominant_emotions
        total = max(1, len(flows))
        dom_sorted = sorted(dom_counts.items(), key=lambda x: x[1], reverse=True)
        emotional_context["dominant_emotions"] = [
            {"emotion": k, "frequency": v, "ratio": round(v / total, 3)}
            for k, v in dom_sorted[:3]
        ]
        return emotional_context

    def _compose_spatiotemporal(self, e: Dict[str, Any], text: str) -> Dict[str, Any]:
        """
        EPSA 결과 + 전체 텍스트 재스캔으로 시간/위치를 보강:
        - 시간: '오전/오후 + hh시(+분)' 우선
        - 위치: 같은 문장에서 city + (district 1~2개) + place를 결합
          · 짧은 구명(중구/서구/남구/북구/동구)은 단독 토큰으로만 허용
          · '친구' 등 사람/대명사류는 deny 리스트로 제외
        """
        st = (e.get("spatiotemporal") or {}).copy()
        st.setdefault("location_details", {})
        txt = text or ""

        # ---- 시간 보강 ----
        full_time_rx = re.compile(r"(오전|오후)\s*\d{1,2}시(?:\s*\d{1,2}분)?")
        abs_time_rx = re.compile(r"\d{1,2}시(?:\s*\d{1,2}분)?")
        rel_time_rx = re.compile(r"(아침|점심|저녁|밤|새벽|정오|자정)")
        if not st.get("time") or not st.get("specific_time"):
            m_full = full_time_rx.search(txt)
            if m_full:
                t = m_full.group().strip()
                st["time"] = t;
                st["specific_time"] = t
            else:
                if not st.get("time"):
                    a = abs_time_rx.search(txt);
                    b = rel_time_rx.search(txt)
                    if a and b:
                        st["time"] = f"{b.group().strip()} {a.group().strip()}";
                        st["specific_time"] = st["time"]
                    elif a:
                        st["time"] = a.group().strip();
                        st["specific_time"] = st["time"]
                    elif b:
                        st["time"] = b.group().strip()

        # ---- 위치 보강 ----
        city_rx = re.compile(r"(서울|부산|대구|인천|광주|대전|울산|세종)")
        district_rx = re.compile(
            r"((?:[가-힣A-Za-z]{3,}(?:시|군|구|동|읍|면))|(?<![가-힣A-Za-z])(?:중구|동구|서구|남구|북구)(?![가-힣A-Za-z]))")
        place_rx = re.compile(r"(카페|회사|학교|도서관|식당|공원|병원|백화점|마트|공항|역|영화관|공연장|PC방|독서실|미술관|박물관)")
        deny_tokens = {"친구", "가족", "사람", "우리", "나"}

        def _assemble_from_sentence(s: str) -> Optional[str]:
            cities = [m.group(1) for m in city_rx.finditer(s)]
            dists = [m.group(1) for m in district_rx.finditer(s)]
            places = [m.group(1) for m in place_rx.finditer(s)]
            dists = [d for d in dists if d not in deny_tokens]
            places = [p for p in places if p not in deny_tokens]
            if not (cities or dists or places):
                return None
            parts, seen = [], set()
            if cities:
                c = cities[0]
                if c not in seen: parts.append(c); seen.add(c)
            for d in dists:
                if d not in seen:
                    parts.append(d);
                    seen.add(d)
                # 행정단위 2개 이상은 과다 → stop
                if len([x for x in parts if x.endswith(("시", "군", "구", "동", "읍", "면"))]) >= 2:
                    break
            if places:
                p = places[0]
                if p not in seen: parts.append(p); seen.add(p)
            return " ".join(parts).strip() if parts else None

        def _has_district(loc: str) -> bool:
            s = loc.strip()
            if re.search(r"(중구|동구|서구|남구|북구)", s): return True
            return bool(re.search(r"(시|군|구|동|읍|면)$", s[-2:] if len(s) >= 2 else s))

        if st.get("location"):
            loc = str(st["location"])
            if not _has_district(loc):
                sents = re.split(r'(?<=[.!?…])["”’)\]]*\s+|[\r\n]+', txt)
                for s in sents:
                    if loc in s:
                        rebuilt = _assemble_from_sentence(s)
                        if rebuilt:
                            st["location"] = rebuilt
                        break
        else:
            sents = re.split(r'(?<=[.!?…])["”’)\]]*\s+|[\r\n]+', txt)
            for s in sents:
                rebuilt = _assemble_from_sentence(s)
                if rebuilt:
                    st["location"] = rebuilt
                    break

        # 실내/실외 추정
        indoor_tokens = ("카페", "회사", "학교", "도서관", "식당", "영화관", "공연장", "병원", "백화점", "마트", "PC방", "독서실", "미술관", "박물관")
        if st.get("location") and any(tok in st["location"] for tok in indoor_tokens):
            st["location_details"]["type"] = "indoor"
        else:
            st["location_details"].setdefault("type", "outdoor")

        return st

    def _compose_situational_triggers(
        self,
        context_mapping: Dict[str, Dict[str, Any]],
        identified_situations: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        매핑에서 상황별 matched_triggers를 꺼내 간단 카드로 집계.
        """
        # ★★★ 안전하게 situation_id 추출 (없으면 빈 문자열) ★★★
        sid_set = {c.get("situation_id", "") for c in identified_situations if c.get("situation_id")}
        out = []
        for sid in sid_set:
            mp = context_mapping.get(sid) or {}
            trigs = mp.get("matched_triggers") or []
            if trigs:
                out.append({
                    "primary_emotion": mp.get("primary_emotion"),
                    "sub_emotion": mp.get("sub_emotion_name"),
                    "situation_key": mp.get("situation_name"),
                    "matched_triggers": trigs,
                })
        return out

    # ------------------------------ Metrics/Stats ------------------------------
    def _compute_situation_metrics(self, cards, emotional_context, spatiotemporal_context):
        """
        표시 정책과 동일하게 전이 집계도 '의미 있는 전이'만 카운트.
        """
        total = len(cards)
        avg_conf = np.mean([c.get("confidence", 0.0) for c in cards]) if cards else 0.0

        dist = defaultdict(int)
        for c in cards:
            dist[c.get("primary_emotion")] += 1

        trans = emotional_context.get("emotional_transitions", []) or []
        meaningful = [
            t for t in trans
            if not (
                    t.get("type") in ("micro_shift", "fallback") or
                    t.get("from_emotion") == t.get("to_emotion")
            )
        ]

        doms = emotional_context.get("dominant_emotions", []) or []
        return {
            "situation_analysis": {
                "total_situations": total,
                "emotion_distribution": dict(dist),
                "average_confidence": round(float(avg_conf), 3),
            },
            "emotional_analysis": {
                "dominant_emotions": doms,
                "transition_count": len(meaningful),
            },
            "contextual_coverage": {
                "spatial_context": bool(spatiotemporal_context.get("location")),
                "temporal_context": bool(spatiotemporal_context.get("time")) or
                                    bool(spatiotemporal_context.get("specific_time")),
                "emotional_context": bool(emotional_context.get("emotion_flows")),
            },
        }

    def _compute_metrics(self, t0: float, n_situations: int) -> Dict[str, Any]:
        dt = round(time.time() - t0, 3)
        mem_kb = self._get_memory_kb()
        return {
            "processing_time": dt,
            "memory_usage_kb": mem_kb,
            "situations_identified": n_situations,
            "success_rate": n_situations > 0,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

    # ------------------------------ Utilities ------------------------------
    @staticmethod
    def _get_memory_kb() -> int:
        try:
            import psutil
            return psutil.Process().memory_info().rss // 1024
        except Exception:
            return 0


# =============================================================================
# SituationAnalyzer (refined) — adaptive time-series + robust similarity
# =============================================================================
class SituationAnalyzer:
    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        emotions_data: Optional[Dict[str, Any]] = None,
        device: str = "cuda" if (torch and torch.cuda.is_available()) else "cpu",
        lazy_embeddings: bool = True,
    ):
        """
        - 임베딩 모델은 기본적으로 '지연 로딩(lazy)'하여 필요 시에만 메모리에 올립니다.
        - 감정 라벨(emotions_data)은 컨텍스트/흐름/유사도 보정에 활용(필수 아님).
        """
        try:
            os.environ["KSS_SUPPRESS_WARNINGS"] = "1"
            warnings.filterwarnings("ignore", category=UserWarning, module="kss")
            logging.getLogger("kss").setLevel(logging.ERROR)
            self.kss = kss
        except Exception:
            self.kss = None

        self.emotions_data = emotions_data or {}
        self.model_name = model_name
        self.device = device
        self._emb_model = None
        self._lazy = bool(lazy_embeddings)

        # 전이 가중(기존 값 유지)
        self.transition_weights = {"rapid": 1.2, "gradual": 0.8, "neutral": 1.0}

        # 강도 구간(기존 값 유지)
        self.intensity_levels = {"high": (0.7, 1.0), "medium": (0.4, 0.7), "low": (0.0, 0.4)}

        # 적응형 파라미터(노브)
        self.params = {
            "mad_k": 1.4826,              # MAD scaling
            "thr_floor": 0.08,            # 변화 감지 최소 임계 하한
            "thr_default": 0.20,          # 샘플 적을 때 기본 임계
            "micro_ratio": 0.5,           # micro 전이 임계 = full 임계 × ratio
            "ewma_alpha_min": 0.35,       # EWMA 최소 알파
            "ewma_alpha_max": 0.85,       # EWMA 최대 알파
            "hold_min": 1,                # 최소 유지 구간
        }

        # 즉시 로딩이 필요한 경우만 로딩
        if not self._lazy:
            self._lazy_load_model()

    # ------------------------------- Embeddings -------------------------------
    def _lazy_load_model(self) -> None:
        if self._emb_model is None:
            try:
                self._emb_model = SentenceTransformer(self.model_name, device=self.device)
                logger.info(f"임베딩 모델 '{self.model_name}' 로드 완료 (device={self.device})")
            except Exception as e:
                logger.warning(f"임베딩 모델 로드 실패(폴백 모드로 동작): {e}")
                self._emb_model = None

    # ------------------------------- Time-series -------------------------------
    def analyze_emotion_time_series(
        self,
        sentences: List[str],
        confidences: List[float],
        window_size: int = 3,
    ) -> Dict[str, Any]:
        """
        입력 문장별 confidence(0~1) 시계열을 받아:
        - EWMA 평활 + MAD 기반 적응 임계로 전이 감지(shift/micro_shift)
        - 이동평균, 전이 통계, 변동성·추세 등 요약 메트릭 산출
        반환 스키마는 기존과 호환
        """
        try:
            if not sentences or not confidences:
                return {}
            n = min(len(sentences), len(confidences))
            if n == 0:
                return {}

            # 길이 정합
            sents = sentences[:n]
            confs = [float(max(0.0, min(1.0, c))) for c in confidences[:n]]

            # 이동평균
            moving_avg, ts = [], []
            prev_smooth = confs[0]

            # 초기 변동성 기반 EWMA 알파를 점진 조정
            def _alpha(i: int, recent: List[float]) -> float:
                if len(recent) < 3:
                    return 0.5
                diffs = [abs(recent[k] - recent[k - 1]) for k in range(1, len(recent))]
                vol = float(np.median(diffs)) if diffs else 0.0
                # 0~1 스케일로 압축
                vol = max(0.0, min(1.0, vol / 0.5))
                a = self.params["ewma_alpha_min"] + (self.params["ewma_alpha_max"] - self.params["ewma_alpha_min"]) * vol
                return float(max(self.params["ewma_alpha_min"], min(self.params["ewma_alpha_max"], a)))

            # 평활 & 전이 감지 준비
            smooth = []
            for i, c in enumerate(confs):
                wstart = max(0, i - max(1, window_size - 1))
                w = confs[wstart:i + 1]
                ma = float(sum(w) / len(w))
                moving_avg.append(round(ma, 3))

                a = _alpha(i, w)
                prev_smooth = (1 - a) * prev_smooth + a * c if i > 0 else c
                smooth.append(prev_smooth)

            # 적응 임계 (MAD 기반)
            diffs = [smooth[i] - smooth[i - 1] for i in range(1, n)]
            adiffs = [abs(d) for d in diffs]
            if len(adiffs) >= 3:
                med = float(np.median(adiffs))
                mad = float(np.median([abs(x - med) for x in adiffs])) * self.params["mad_k"]
                thr = max(self.params["thr_floor"], med + 1.0 * mad)
            else:
                thr = self.params["thr_default"]
            micro_thr = max(self.params["thr_floor"], thr * self.params["micro_ratio"])

            # 전이 탐지(히스테리시스/hold)
            transitions = []
            pattern_seq = []
            last_idx = -999
            hold = int(self.params["hold_min"])

            for i in range(n):
                rec = {
                    "index": i,
                    "text": sents[i],
                    "confidence": round(confs[i], 4),
                    "intensity": self._determine_intensity(confs[i]),
                    "moving_average": moving_avg[i],
                    "transitions": [],
                    "patterns": [],
                }

                if i > 0:
                    delta = smooth[i] - smooth[i - 1]
                    ad = abs(delta)

                    if (i - last_idx) >= hold:
                        if ad >= thr:
                            ttype = "rapid" if ad >= (thr * 1.5) else "gradual"
                            tr = {
                                "from_intensity": self._determine_intensity(confs[i - 1]),
                                "to_intensity": self._determine_intensity(confs[i]),
                                "position": i,
                                "type": ttype,
                                "confidence": round(float(confs[i] * self.transition_weights.get(ttype, 1.0)), 4),
                                "delta": round(float(delta), 4),
                                "is_micro": False,
                            }
                            transitions.append(tr)
                            rec["transitions"].append(tr)
                            pattern_seq.append(ttype)
                            last_idx = i
                        elif ad >= micro_thr:
                            tr = {
                                "from_intensity": self._determine_intensity(confs[i - 1]),
                                "to_intensity": self._determine_intensity(confs[i]),
                                "position": i,
                                "type": "micro_shift",
                                "confidence": round(float(confs[i] * 0.9), 4),
                                "delta": round(float(delta), 4),
                                "is_micro": True,
                            }
                            transitions.append(tr)
                            rec["transitions"].append(tr)
                            pattern_seq.append("micro_shift")

                    # 최근 3개 전이 유형 패턴 태깅
                    if len(pattern_seq) >= 3:
                        tri = pattern_seq[-3:]
                        if all(p == "gradual" for p in tri):
                            rec["patterns"].append("consistent_gradual")
                        elif all(p == "rapid" for p in tri):
                            rec["patterns"].append("volatile")
                        elif tri.count("micro_shift") >= 2:
                            rec["patterns"].append("micro_oscillation")

                ts.append(rec)

            # 패턴 통계/요약 메트릭
            pat_stats = self._analyze_progression_patterns(ts, thr, micro_thr)
            diffs_full = [confs[i] - confs[i - 1] for i in range(1, n)]
            volatility = float(np.std(diffs_full)) if len(diffs_full) > 0 else 0.0
            try:
                trend = float(np.polyfit(np.arange(n, dtype=float), np.asarray(confs, dtype=float), 1)[0])
            except Exception:
                trend = 0.0

            return {
                "time_series_data": ts,
                "moving_average": moving_avg,
                "emotion_transitions": transitions,
                "pattern_statistics": pat_stats,
                "aggregated_metrics": {
                    "mean_confidence": float(np.mean(confs)),
                    "peak_confidence": float(np.max(confs)),
                    "transition_count": int(sum(1 for t in transitions if not t.get("is_micro"))),
                    "micro_transition_count": int(sum(1 for t in transitions if t.get("is_micro"))),
                    "volatility": float(volatility),
                    "trend": float(trend),
                    "pattern_coverage": float(sum(1 for d in ts if d["patterns"]) / max(1, len(ts))),
                    "threshold_used": round(float(thr), 4),
                    "micro_threshold_used": round(float(micro_thr), 4),
                },
            }
        except Exception as e:
            logger.error(f"시계열 분석 중 오류: {str(e)}")
            return {}

    def _analyze_progression_patterns(
        self,
        time_series_data: List[Dict[str, Any]],
        thr: float,
        micro_thr: float,
    ) -> Dict[str, Any]:
        """
        변화 크기 기준으로 증가/감소/안정/요동 구간을 세그먼트화하여 통계 산출.
        """
        try:
            if not time_series_data:
                return {
                    "detected_patterns": [],
                    "pattern_sequences": [],
                    "dominant_pattern": None,
                    "pattern_statistics": {},
                }

            smooth_conf = [float(d["confidence"]) for d in time_series_data]
            diffs = [smooth_conf[i] - smooth_conf[i - 1] for i in range(1, len(smooth_conf))]
            # 라벨링
            lab = []
            for d in diffs:
                ad = abs(d)
                if ad < micro_thr:
                    lab.append("stable")
                elif d > 0:
                    lab.append("increasing" if ad >= thr else "micro_up")
                else:
                    lab.append("decreasing" if ad >= thr else "micro_down")

            # 세그먼트화
            segs, cur, start = [], None, 0
            for i, tag in enumerate(lab):
                if cur is None:
                    cur, start = tag, 0
                elif tag != cur:
                    segs.append({"pattern": cur, "start_index": start, "end_index": i, "duration": i - start + 1})
                    cur, start = tag, i
            if cur is not None:
                segs.append({"pattern": cur, "start_index": start, "end_index": len(lab), "duration": len(lab) - start + 1})

            counts = defaultdict(int)
            for s in segs:
                counts[s["pattern"]] += s["duration"]

            total = sum(counts.values()) or 1
            pat_stats = {
                "pattern_distribution": {k: float(v) / total for k, v in counts.items()},
                "average_pattern_duration": float(np.mean([s["duration"] for s in segs])) if segs else 0.0,
                "pattern_count": int(len(segs)),
            }
            dom = max(counts.items(), key=lambda x: x[1])[0] if counts else None
            return {
                "detected_patterns": segs,
                "pattern_sequences": [s["pattern"] for s in segs],
                "dominant_pattern": dom,
                "pattern_statistics": pat_stats,
            }
        except Exception as e:
            logger.error(f"진행 패턴 분석 중 오류: {str(e)}")
            return {
                "detected_patterns": [],
                "pattern_sequences": [],
                "dominant_pattern": None,
                "pattern_statistics": {},
            }

    def _determine_intensity(self, confidence: float) -> str:
        for level, (lo, hi) in self.intensity_levels.items():
            if lo <= confidence < hi:
                return level
        return "medium"

    def _determine_transition_type(self, prev_intensity: str, curr_intensity: str) -> str:
        vals = {"high": 3, "medium": 2, "low": 1}
        diff = abs(vals.get(prev_intensity, 2) - vals.get(curr_intensity, 2))
        if diff >= 2:
            return "rapid"
        elif diff == 1:
            return "gradual"
        return "neutral"

    # ------------------------------- Similarity -------------------------------
    def get_similarity(self, text1: str, text2: str) -> Optional[float]:
        """
        문맥/흐름/의미를 혼합한 유사도:
          final = 0.4*semantic + 0.3*context + 0.3*flow  (임베딩 부재시 semantic 가중은 나머지로 재분배)
        """
        try:
            # 문장 분리
            if self.kss:
                s1 = self.kss.split_sentences(text1 or "")
                s2 = self.kss.split_sentences(text2 or "")
            else:
                s1 = [t.strip() for t in (text1 or "").split(".") if t.strip()]
                s2 = [t.strip() for t in (text2 or "").split(".") if t.strip()]

            # 문맥 피처
            f1 = self._extract_context_features(s1)
            f2 = self._extract_context_features(s2)

            # 의미 임베딩(지연 로딩)
            sem_sim = 0.0
            if not self._lazy and self._emb_model is None:
                self._lazy_load_model()
            if self._emb_model is None and self._lazy:
                self._lazy_load_model()

            if self._emb_model is not None:
                v1 = self._emb_model.encode(text1 or "", show_progress_bar=False)
                v2 = self._emb_model.encode(text2 or "", show_progress_bar=False)
                denom = (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12)
                sem_sim = float(np.dot(v1, v2) / denom)

            # 컨텍스트 유사도(라벨 기반 점수)
            ctx_sim = self._calculate_context_similarity(f1, f2, self.emotions_data)

            # 흐름 유사도(DTW 기반)
            flow_sim = self._calculate_emotion_flow_similarity(s1, s2)

            w_sem, w_ctx, w_flow = 0.4, 0.3, 0.3
            if self._emb_model is None:
                # 임베딩 부재 → 의미 가중을 나머지에 균등 재분배
                w_sem, w_ctx, w_flow = 0.0, 0.5, 0.5

            final = float(w_sem * sem_sim + w_ctx * ctx_sim + w_flow * flow_sim)
            return max(0.0, min(1.0, final))
        except Exception as e:
            logger.error(f"유사도 계산 중 오류: {str(e)}")
            return None

    # ------------------------------- Context feats -------------------------------
    def _extract_context_features(self, sentences: List[str]) -> Dict[str, Any]:
        feats = {
            "temporal_markers": [],
            "causal_markers": [],
            "emotion_words": [],
            "intensity_markers": [],
            "transition_markers": [],
            "situation_markers": [],
        }
        for s in sentences or []:
            if re.search(r"(먼저|그다음|이후|마지막|(\d+시)|오전|오후|아침|저녁)", s):
                feats["temporal_markers"].append(s)
            if re.search(r"(때문에|그래서|따라서|결과|원인)", s):
                feats["causal_markers"].append(s)
            if re.search(r"(매우|너무|정말|아주|가장|조금|약간)", s):
                feats["intensity_markers"].append(s)
            if re.search(r"(행복|기쁨|슬픔|분노|불안|만족|환희)", s):
                feats["emotion_words"].append(s)
            if re.search(r"(하지만|그런데|그러나|반면|오히려)", s):
                feats["transition_markers"].append(s)
            if re.search(r"(상황|일|사건|경우|순간)", s):
                feats["situation_markers"].append(s)
        return feats

    def analyze(self, text: str) -> Dict[str, Any]:
        """
        상황 분석 메인 메서드 - EMOTIONS.JSON 뼈대 기준으로 작동
        """
        try:
            # 기본 응답 구조
            result = {
                "identified_situations": [],
                "context_mapping": {},
                "situation_metrics": {},
                "emotional_context": {},
                "spatiotemporal_context": {},
                "situational_triggers": [],
                "metrics": {
                    "processing_time": 0.0,
                    "memory_usage_kb": 0,
                    "situations_identified": 0,
                    "success_rate": True,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "warnings": [],
                },
                "error": None,
            }
            
            # 텍스트 검증
            if not text or not text.strip():
                result["error"] = "Empty input text"
                result["metrics"]["success_rate"] = False
                return result
            
            # 문장 분리
            if self.kss:
                sentences = self.kss.split_sentences(text)
            else:
                sentences = [s.strip() for s in text.split(".") if s.strip()]
            
            if not sentences:
                result["error"] = "No sentences found"
                result["metrics"]["success_rate"] = False
                return result
            
            # 라벨링 순회: EMOTIONS.json의 상황 패턴 매칭
            identified = []
            
            if self.emotions_data:
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
                            description = situation_data.get('description', '') or ''
                            
                            matches = []
                            
                            # 키워드 매칭
                            for keyword in keywords:
                                if isinstance(keyword, str):
                                    for sentence in sentences:
                                        if keyword.lower() in sentence.lower():
                                            matches.append({
                                                "sentence": sentence,
                                                "keyword": keyword,
                                                "type": "keyword",
                                                "confidence": 0.8
                                            })
                            
                            # 예시 매칭
                            for example in examples:
                                if isinstance(example, str):
                                    for sentence in sentences:
                                        if example.lower() in sentence.lower():
                                            matches.append({
                                                "sentence": sentence,
                                                "keyword": example,
                                                "type": "example",
                                                "confidence": 0.7
                                            })
                            
                            # 변형 매칭
                            for variation in variations:
                                if isinstance(variation, str):
                                    for sentence in sentences:
                                        if variation.lower() in sentence.lower():
                                            matches.append({
                                                "sentence": sentence,
                                                "keyword": variation,
                                                "type": "variation",
                                                "confidence": 0.6
                                            })
                            
                            # 설명 매칭
                            if description:
                                for sentence in sentences:
                                    if description.lower() in sentence.lower():
                                        matches.append({
                                            "sentence": sentence,
                                            "keyword": description,
                                            "type": "description",
                                            "confidence": 0.5
                                        })
                            
                            # 매칭이 있으면 결과에 추가
                            if matches:
                                intensity_level = situation_data.get('intensity', 'medium')
                                intensity_weight = {'low': 0.3, 'medium': 0.6, 'high': 0.9}.get(intensity_level, 0.5)
                                
                                # 신뢰도 계산 (매칭 타입별 가중치 적용)
                                confidence = min(0.9, len(matches) * 0.15 * intensity_weight)
                                
                                identified.append({
                                    "situation": f"{primary_emotion}-{sub_emotion}-{situation_key}",
                                    "primary_emotion": primary_emotion,
                                    "sub_emotion": sub_emotion,
                                    "situation_key": situation_key,
                                    "matches": matches,
                                    "confidence": confidence,
                                    "evidence_count": len(matches),
                                    "intensity_level": intensity_level,
                                    "description": description
                                })
            else:
                # 폴백: 기본 상황 패턴 매칭 (한국어 + 영어)
                situation_patterns = {
                    "갈등": ["갈등", "충돌", "분쟁", "언쟁", "대립", "싸움", "conflict", "fight", "argument", "dispute"],
                    "축하": ["축하", "경축", "파티", "치하", "기념", "celebration", "party", "congratulation", "festival"],
                    "상실": ["상실", "이별", "상심", "슬픔", "잃다", "loss", "sadness", "grief", "mourning", "departure"],
                    "협상": ["협상", "타협", "조율", "교섭", "합의", "negotiation", "compromise", "agreement", "deal"],
                    "일상": ["일상", "평범", "보통", "일반", "daily", "normal", "ordinary", "routine", "regular"],
                    "긴급": ["긴급", "급함", "빨리", "서둘러", "응급", "urgent", "emergency", "hurry", "quick", "fast"],
                    "행복": ["행복", "기쁨", "만족", "환희", "happy", "joy", "pleasure", "satisfaction", "delight"],
                    "슬픔": ["슬픔", "우울", "절망", "sad", "depressed", "melancholy", "gloomy", "sorrow"],
                    "분노": ["분노", "짜증", "화남", "angry", "mad", "furious", "rage", "irritated"],
                    "불안": ["불안", "걱정", "초조", "anxiety", "worry", "nervous", "uneasy", "concern"],
                }
                
                for situation, keywords in situation_patterns.items():
                    matches = []
                    for sentence in sentences:
                        for keyword in keywords:
                            if keyword in sentence:
                                matches.append({
                                    "sentence": sentence,
                                    "keyword": keyword,
                                    "type": "fallback",
                                    "confidence": 0.8
                                })
                    
                    if matches:
                        identified.append({
                            "situation": situation,
                            "matches": matches,
                            "confidence": min(0.9, len(matches) * 0.2),
                            "evidence_count": len(matches)
                        })
            
            # 결과 구성
            result["identified_situations"] = identified
            result["situation_metrics"] = {
                "total_situations": len(identified),
                "high_confidence": len([s for s in identified if s["confidence"] > 0.7]),
                "medium_confidence": len([s for s in identified if 0.4 <= s["confidence"] <= 0.7]),
                "low_confidence": len([s for s in identified if s["confidence"] < 0.4]),
            }
            
            # 컨텍스트 매핑
            result["context_mapping"] = {
                "sentence_count": len(sentences),
                "text_length": len(text),
                "has_temporal_markers": any("시" in s or "오전" in s or "오후" in s for s in sentences),
                "has_emotional_markers": any("기분" in s or "감정" in s or "마음" in s for s in sentences),
            }
            
            # 감정적 컨텍스트: EMOTIONS.json 기반 분석
            found_emotions = []
            emotion_intensities = {}
            
            if self.emotions_data:
                # EMOTIONS.json의 감정 키워드 수집
                for primary_emotion, emotion_info in self.emotions_data.items():
                    emotion_profile = emotion_info.get('emotion_profile', {}) or {}
                    core_keywords = emotion_profile.get('core_keywords', []) or []
                    
                    # 세부 감정의 키워드도 수집
                    sub_emotions = emotion_info.get('sub_emotions', {}) or {}
                    for sub_emotion, sub_info in sub_emotions.items():
                        sub_profile = sub_info.get('emotion_profile', {}) or {}
                        sub_keywords = sub_profile.get('core_keywords', []) or []
                        core_keywords.extend(sub_keywords)
                    
                    # 감정 키워드 매칭
                    for keyword in core_keywords:
                        if isinstance(keyword, str):
                            for sentence in sentences:
                                if keyword.lower() in sentence.lower():
                                    found_emotions.append(keyword)
                                    if primary_emotion not in emotion_intensities:
                                        emotion_intensities[primary_emotion] = 0
                                    emotion_intensities[primary_emotion] += 1
            else:
                # 폴백: 기본 감정 키워드
                emotion_keywords = ["행복", "기쁨", "슬픔", "분노", "불안", "만족", "환희", "우울", "짜증", "피곤", 
                                   "happy", "joy", "sad", "angry", "anxiety", "satisfaction", "depressed", "tired", "excited"]
                for sentence in sentences:
                    for emotion in emotion_keywords:
                        if emotion in sentence:
                            found_emotions.append(emotion)
            
            result["emotional_context"] = {
                "detected_emotions": list(set(found_emotions)),
                "emotion_density": len(found_emotions) / len(sentences) if sentences else 0,
                "emotion_intensities": emotion_intensities,
                "primary_emotions_detected": list(emotion_intensities.keys())
            }
            
            # 시공간 컨텍스트 (한국어 + 영어)
            spatial_patterns = ["서울", "부산", "카페", "집", "회사", "학교", "병원", "공원", 
                              "Seoul", "Busan", "cafe", "home", "office", "school", "hospital", "park"]
            temporal_patterns = ["오전", "오후", "아침", "저녁", "밤", "주말", "평일", 
                               "morning", "afternoon", "evening", "night", "weekend", "weekday"]
            
            spatial_found = [p for p in spatial_patterns if any(p in s for s in sentences)]
            temporal_found = [p for p in temporal_patterns if any(p in s for s in sentences)]
            
            result["spatiotemporal_context"] = {
                "spatial_markers": spatial_found,
                "temporal_markers": temporal_found,
                "has_location": len(spatial_found) > 0,
                "has_time": len(temporal_found) > 0,
            }
            
            # 상황 트리거
            triggers = []
            for situation in identified:
                for match in situation["matches"]:
                    triggers.append({
                        "trigger": match["keyword"],
                        "situation": situation["situation"],
                        "sentence": match["sentence"],
                        "confidence": match["confidence"]
                    })
            
            result["situational_triggers"] = triggers
            
            # 메트릭 업데이트
            result["metrics"]["situations_identified"] = len(identified)
            result["metrics"]["processing_time"] = 0.1  # 간단한 처리 시간
            
            return result
            
        except Exception as e:
            logger.error(f"SituationAnalyzer.analyze 오류: {str(e)}")
            return {
                "identified_situations": [],
                "context_mapping": {},
                "situation_metrics": {},
                "emotional_context": {},
                "spatiotemporal_context": {},
                "situational_triggers": [],
                "metrics": {
                    "processing_time": 0.0,
                    "memory_usage_kb": 0,
                    "situations_identified": 0,
                    "success_rate": False,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "warnings": [f"Error: {str(e)}"],
                },
                "error": str(e),
            }

    def _calculate_context_similarity(
        self,
        features1: Dict[str, Any],
        features2: Dict[str, Any],
        emotions_data: Dict[str, Any],
    ) -> float:
        """
        라벨링 뼈대 상의 상황/변형/예시를 가볍게 대조한 컨텍스트 점수.
        (현 구현은 기존 파일 로직과 호환되는 방식으로 요약 유지)
        """
        try:
            weights = {
                "temporal_markers": 0.2,
                "causal_markers": 0.2,
                "emotion_words": 0.3,
                "intensity_markers": 0.15,
                "transition_markers": 0.15,
                "situation_markers": 0.2,
            }
            sims: List[float] = []
            for _, emo in (emotions_data or {}).items():
                subs = (emo.get("emotion_profile", {}) or {}).get("sub_emotions", {})
                for _, sub in (subs or {}).items():
                    ctxp = sub.get("context_patterns", {}) or (sub.get("emotion_profile", {}) or {}).get("context_patterns", {})
                    sits = ctxp.get("situations", {})
                    for _, sit in (sits or {}).items():
                        core = str(sit.get("core_concept", "")).strip()
                        if core:
                            in1 = any(core in " ".join(v if isinstance(v, list) else [v]) for v in features1.values())
                            in2 = any(core in " ".join(v if isinstance(v, list) else [v]) for v in features2.values())
                            if in1 and in2:
                                sims.append(1.0)
                                continue
                        # 마커 Jaccard × 가중
                        acc = 0.0
                        for m, w in weights.items():
                            a, b = set(features1.get(m, [])), set(features2.get(m, []))
                            if a or b:
                                inter = len(a & b)
                                union = len(a | b) or 1
                                acc += w * (inter / union)
                        sims.append(acc)
            return float(max(sims) if sims else 0.0)
        except Exception as e:
            logger.error(f"문맥 유사도 계산 중 오류: {str(e)}")
            return 0.0

    def _calculate_emotion_flow_similarity(self, sentences1: List[str], sentences2: List[str]) -> float:
        try:
            p1 = self._extract_emotion_flow_pattern(sentences1)
            p2 = self._extract_emotion_flow_pattern(sentences2)
            if not p1 or not p2:
                return 0.0
            dist = self._calculate_dtw_distance(p1, p2)
            maxd = float(max(len(p1), len(p2)))
            sim = 1.0 - (dist / (maxd + 1e-8))
            return float(max(0.0, min(1.0, sim)))
        except Exception as e:
            logger.error(f"감정 흐름 유사도 계산 중 오류: {str(e)}")
            return 0.0

    def _extract_emotion_flow_pattern(self, sentences: List[str]) -> List[float]:
        """
        간단한 감정 스코어링(라벨 사전 없이도 동작).
        실제 파이프라인에서는 EPSA/컨텍스트 점수를 넘겨 쓰는 것이 바람직.
        """
        out: List[float] = []
        for s in sentences or []:
            if re.search(r"(행복|기쁨)", s):
                out.append(0.95)
            elif re.search(r"(슬픔|우울)", s):
                out.append(0.2)
            elif re.search(r"(분노|화)", s):
                out.append(0.75)
            else:
                out.append(0.5)
        return out

    def _calculate_dtw_distance(
        self,
        pattern1: Union[List[float], np.ndarray],
        pattern2: Union[List[float], np.ndarray],
    ) -> float:
        try:
            a1 = np.asarray(pattern1, dtype=float)
            a2 = np.asarray(pattern2, dtype=float)
            n, m = len(a1), len(a2)
            D = np.full((n + 1, m + 1), float("inf"), dtype=float)
            D[0, 0] = 0.0
            for i in range(1, n + 1):
                for j in range(1, m + 1):
                    cost = abs(a1[i - 1] - a2[j - 1])
                    D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
            return float(D[n, m])
        except Exception as e:
            logger.error(f"DTW 거리 계산 중 오류: {str(e)}")
            return float("inf")


# =============================================================================
# Public Independent Functions (safe to import from __init__.py)
# =============================================================================
@contextmanager
def _env_overrides(overrides: Dict[str, Any]):
    """
    os.environ 일시 오버라이드. with 블록 종료 시 원복.
    값이 None인 키는 건너뜀.
    """
    backup = {}
    try:
        for k, v in (overrides or {}).items():
            if v is None:
                continue
            k = str(k)
            backup[k] = os.environ.get(k)
            os.environ[k] = str(v)
        yield
    finally:
        for k, old in backup.items():
            if old is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old


def run_situation_analysis(
    text: str,
    emotions_data: Dict[str, Any],
    *,
    # 표시/필터링 노브(권장 기본값)
    min_conf: float = 0.20,                 # SIT_ORCH_MIN_CONF / SIT_MIN_CONF
    topk: int = 8,                          # SIT_ORCH_TOPK
    emit_min_evidence: int = 2,             # SIT_EMIT_MIN_EVID(키워드/변형/예시/트리거/단계 중 최소 2가지)
    # 전이 감지 노브
    trans_conf_thr: float = 0.50,           # SIT_TRANS_CONF_THR (기본 0.5; 필요시 0.6~0.7로 상향)
    allow_fallback_triggers: bool = True,   # SIT_ALLOW_FALLBACK_TRIGGERS
    hide_fallback_transitions: bool = True, # SIT_HIDE_FALLBACK_TRANS
    only_meaningful_transitions: bool = True,# SIT_ONLY_MEANINGFUL_TRANS
    # 임베딩 모델 주입(선택)
    embedding_model: Any = None,
) -> Dict[str, Any]:
    """
    독립 함수 #1: 한 번 호출로 상황 매핑 + 감정 전이/흐름 + 시공간 지표까지 리포트 생성.
    """
    if not isinstance(text, str) or not text.strip():
        return _empty_situation_report("Input text is empty")
    if not isinstance(emotions_data, dict) or not emotions_data:
        return _empty_situation_report("Invalid emotions_data")

    env = {
        "SIT_MIN_CONF": min_conf,
        "SIT_ORCH_MIN_CONF": min_conf,
        "SIT_ORCH_TOPK": topk,
        "SIT_EMIT_MIN_EVID": emit_min_evidence,
        "SIT_TRANS_CONF_THR": trans_conf_thr,
        "SIT_ALLOW_FALLBACK_TRIGGERS": "1" if allow_fallback_triggers else "0",
        "SIT_HIDE_FALLBACK_TRANS": "1" if hide_fallback_transitions else "0",
        "SIT_ONLY_MEANINGFUL_TRANS": "1" if only_meaningful_transitions else "0",
    }

    with _env_overrides(env):
        orch = SituationContextOrchestrator(emotions_data, embedding_model=embedding_model)
        return orch.analyze(text)


def run_spatiotemporal_analysis(
    text: str,
    emotions_data: Dict[str, Any],
    *,
    trans_conf_thr: float = 0.50,
    allow_fallback_triggers: bool = True,
    embedding_model: Any = None,
) -> Dict[str, Any]:
    """
    독립 함수 #2: 시공간(시간/위치/시퀀스)만 빠르게 뽑고 싶은 경우.
    내부적으로 EPSA를 한 번 돌려 시공간 요소를 보강해 반환합니다.
    """
    if not isinstance(text, str) or not text.strip():
        return {"spatiotemporal_context": {}, "error": "Input text is empty"}
    if not isinstance(emotions_data, dict) or not emotions_data:
        return {"spatiotemporal_context": {}, "error": "Invalid emotions_data"}

    env = {
        "SIT_TRANS_CONF_THR": trans_conf_thr,
        "SIT_ALLOW_FALLBACK_TRIGGERS": "1" if allow_fallback_triggers else "0",
    }
    with _env_overrides(env):
        orch = SituationContextOrchestrator(emotions_data, embedding_model=embedding_model)
        # EPSA만 돌려서 시공간 재합성
        epsa_out = orch.epsa.analyze(text, emotions_data)
        st = orch._compose_spatiotemporal(epsa_out, text)
        return {"spatiotemporal_context": st}


def run_full_situation_analysis(
    text: str,
    emotions_data: Dict[str, Any],
    *,
    # 카드/전이 정책
    min_conf: float = 0.20,
    topk: int = 8,
    emit_min_evidence: int = 2,
    trans_conf_thr: float = 0.50,
    allow_fallback_triggers: bool = True,
    hide_fallback_transitions: bool = True,
    only_meaningful_transitions: bool = True,
    # 시계열 보조 분석
    do_time_series: bool = True,
    # 임베딩 모델 주입(선택)
    embedding_model: Any = None,
) -> Dict[str, Any]:
    """
    독립 함수 #3: 상황 리포트 + (선택) 시계열 보조 분석(전이/변동성/추세)까지 종합 반환.
    """
    report = run_situation_analysis(
        text,
        emotions_data,
        min_conf=min_conf,
        topk=topk,
        emit_min_evidence=emit_min_evidence,
        trans_conf_thr=trans_conf_thr,
        allow_fallback_triggers=allow_fallback_triggers,
        hide_fallback_transitions=hide_fallback_transitions,
        only_meaningful_transitions=only_meaningful_transitions,
        embedding_model=embedding_model,
    )
    if do_time_series and isinstance(report, dict):
        flows = (report.get("emotional_context") or {}).get("emotion_flows", []) or []
        if flows:
            sentences = [row.get("sentence", "") for row in flows]
            confidences = [float((row.get("dominant_emotion") or {}).get("score", 0.0)) for row in flows]
            ts = SituationAnalyzer(emotions_data=emotions_data, lazy_embeddings=True)
            ts_res = ts.analyze_emotion_time_series(sentences, confidences)
            report.setdefault("diagnostics", {})["time_series"] = ts_res.get("aggregated_metrics", {})
    return report


# 중복 함수 제거: analyze_situation_context는 run_situation_analysis와 동일한 기능
# 필요시 run_situation_analysis를 직접 사용하거나 __init__.py에서 별칭으로 제공



# =============================================================================
# Main Function (drop-in) — 독립 오케스트레이션 스모크 테스트
# =============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _setup_main_logger(log_dir: Path) -> Tuple[logging.Logger, Path, Path]:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = log_dir / "situation_analyzer.log"
    json_file_path = log_dir / "situation_analyzer.json"

    logger_root = logging.getLogger()
    for h in list(logger_root.handlers):
        logger_root.removeHandler(h)

    logging.basicConfig(
        level=logging.INFO,  # 상세 확인시 DEBUG
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file_path, encoding="utf-8", mode="w"),
        ],
    )
    try:
        os.environ["KSS_SUPPRESS_WARNINGS"] = "1"
        warnings.filterwarnings("ignore", category=UserWarning, module="kss")
        logging.getLogger("kss").setLevel(logging.ERROR)
    except Exception:
        pass
    return logging.getLogger(__name__), log_file_path, json_file_path


def _find_emotions_json(cwd: Path) -> Optional[Path]:
    # 우선순위: ENV → 현재/부모 경로의 표준 파일명 → /mnt/data
    cand = os.environ.get("EMOTIONS_JSON")
    if cand and Path(cand).exists():
        return Path(cand)

    names = [
        "EMOTIONS.json", "EMOTIONS.JSON", "emotions.json",
        "EMOTIONS.ndjson", "emotions.ndjson",
        "EMOTIONS.txt", "EMOTIONS.JSON(일부).txt",
    ]
    search_dirs = [cwd, cwd.parent, Path(BASE_DIR), Path("/mnt/data")]
    for d in search_dirs:
        for n in names:
            p = d / n
            if p.exists():
                return p
    return None


def _pretty_print_section(title: str, payload: Any) -> None:
    print(f"\n{title}")
    if isinstance(payload, (dict, list)):
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(str(payload))


def main():
    current_dir = Path(__file__).resolve().parent
    base_dir = current_dir.parent
    # 통합 로그 관리자 사용 (날짜별 폴더)
    try:
        from log_manager import get_log_manager
        log_manager = get_log_manager()
        log_dir = log_manager.get_log_dir("emotion_analysis", use_date_folder=True)
    except ImportError:
        # 폴백: 기존 방식 (날짜별 폴더 추가)
        from datetime import datetime
        base_log_dir = base_dir / "emotion_analysis" / "logs"
        today = datetime.now().strftime("%Y%m%d")
        log_dir = base_log_dir / today
    logger, log_file_path, json_file_path = _setup_main_logger(log_dir)

    logger.info("===== Situation Analyzer 시작 =====")
    try:
        emotions_path = _find_emotions_json(base_dir)
        if not emotions_path:
            logger.error("감정 데이터 파일(EMOTIONS.json)을 찾을 수 없습니다. Soft-fail 모드로 진행합니다.")
            emotions_data = {}  # 빈 라벨로도 안전 동작 (카드 0건)
        else:
            with emotions_path.open("r", encoding="utf-8") as f:
                emotions_data = json.load(f)
            logger.info(f"감정 데이터 로드 완료: {emotions_path}")
        # 라벨링 뼈대 검증 + 역할 적합도 확인
        vd = validate_emotion_data(emotions_data)
        if not vd.get("ok", False):
            logger.warning("라벨링 뼈대 검증 경고/에러가 있습니다. (동작은 느슨 모드로 계속)")
        readiness = vd.get("readiness", {}).get("percent", 0)
        logger.info(f"라벨링 역할 적합도(readiness): {readiness}%")
        if vd.get("hints"):
            logger.info(f"라벨 개선 힌트: {vd['hints']}")

        # 오케스트레이터(독립): 상황 매핑 + EPSA(흐름/전이/시공간) 합성
        orchestrator = SituationContextOrchestrator(emotions_data)

        # 샘플 텍스트 (원하시면 CLI/ENV로 주입 가능)
        test_text = (
            "오늘 오후 2시에 서울 강남구 카페에서 친구를 만났습니다.\n"
            "정말 행복하고 기분이 좋아서 아주 만족감이 최고조였습니다.\n"
            "그런데 갑자기 친구에게서 안 좋은 소식이 들려와서 기분이 확 무거워졌습니다.\n"
            "그래도 함께 이야기하면서 서서히 위로받고 기분이 점점 나아졌습니다."
        )
        logger.info("테스트 텍스트 분석을 시작합니다.")
        logger.debug("분석할 텍스트:\n%s", test_text)

        # 최종 리포트 생성
        report = orchestrator.analyze(test_text)

        # (선택) 시계열 보조 분석: EPSA 흐름에서 confidence 추출하여 타임라인 품질 디버깅
        try:
            flows = report.get("emotional_context", {}).get("emotion_flows", []) or []
            sentences = [row.get("sentence", "") for row in flows]
            confidences = [float((row.get("dominant_emotion") or {}).get("score", 0.0)) for row in flows]
            if sentences and confidences:
                ts_an = SituationAnalyzer(emotions_data=emotions_data, lazy_embeddings=True)
                ts_res = ts_an.analyze_emotion_time_series(sentences, confidences)
                # 디버깅 메트릭만 메인 리포트에 삽입(선택)
                report.setdefault("diagnostics", {})["time_series"] = ts_res.get("aggregated_metrics", {})
        except Exception as e:
            logger.debug(f"보조 시계열 분석 생략: {e}")

        # JSON 저장
        with open(json_file_path, "w", encoding="utf-8") as jf:
            json.dump(report, jf, ensure_ascii=False, indent=4)
        logger.info(f"결과 JSON 저장 완료: {json_file_path}")

        # 콘솔 요약
        sections = [
            ("1) 식별된 상황:", "identified_situations"),
            ("2) 시공간 컨텍스트:", "spatiotemporal_context"),
            ("3) 감정 컨텍스트:", "emotional_context"),
            ("4) 상황 메트릭:", "situation_metrics"),
            ("5) 처리 메트릭:", "metrics"),
        ]
        print("\n=== 상황 컨텍스트 분석 결과 (요약) ===")
        for title, key in sections:
            _pretty_print_section(title, report.get(key, {}))

        # 역할 적합도 안내 (테스트 시 가이드)
        if readiness < 60:
            logger.warning("라벨링 역할 적합도가 낮습니다(<60). 상황/전이/강도/시공간 축을 보강하면 정확도가 향상됩니다.")

    except FileNotFoundError as e:
        logger.error(f"파일을 찾을 수 없습니다: {e}")
    except json.JSONDecodeError as e:
        logger.error(f"JSON 파싱 오류: {e}")
    except Exception as e:
        logger.exception("메인 실행 중 오류 발생")
    finally:
        logger.info("프로그램 종료")


if __name__ == "__main__":
    main()
