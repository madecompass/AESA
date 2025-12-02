# time_series_analyzer.py
# -*- coding: utf-8 -*-
from __future__ import annotations


import json
import logging
import importlib.util, argparse, os
import time
from pathlib import Path
import unicodedata
from enum import Enum
from pprint import pprint
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Union, Dict, Any, List
from threading import RLock

try:
    import psutil
except ImportError:
    psutil = None

# =============================================================================
# Logger 설정
# =============================================================================
# 중복 초기화를 피하기 위해 여기서는 로거만 참조하고, 실제 핸들러 구성은 하단 _setup_logger()에서 수행합니다.
logger = logging.getLogger("time_series_analyzer")

# =============================================================================
# TimeFlow Mode (standardized, enum-like)
# =============================================================================
class TimeFlowMode(str, Enum):
    STATIC = "static"            # 문장 1개 또는 시간흐름 판단 불가(정지)
    LINEAR = "linear"            # 정상적인 전진 흐름(기본)
    LINEAR_CAPPED = "linear_capped"  # 데이터 희소/신뢰도 낮아 보수적 캡
    UNDETERMINED = "undetermined"    # 판단 유보

def _choose_timeflow_mode(sent_count: int, detected_events: int) -> tuple[str, list[str], float]:
    """문장 수/이벤트 커버리지 기반 모드/사유/커버리지 산출"""
    if sent_count <= 1:
        return TimeFlowMode.STATIC.value, ["min_sequence"], 0.0
    coverage = (detected_events / max(sent_count, 1))
    if coverage < 0.10:
        return TimeFlowMode.LINEAR_CAPPED.value, ["low_coverage"], coverage
    return TimeFlowMode.LINEAR.value, [], coverage


# =============================================================================
# Dataclass
# =============================================================================
@dataclass
class AnalysisResult:
    """
    최종 분석 결과 데이터 구조 (하위호환 보장)
    - 기존 필드 명세는 유지
    - 추가 필드는 모두 Optional/기본값 처리
    - 직렬화/병합/편의 메서드 제공
    """
    # 기존 스펙
    emotion_sequence: List[Dict[str, Any]]
    summary: Dict[str, Any]
    cause_effect: List[Dict[str, Any]]
    time_flow: Dict[str, Any]
    complex_emotions: List[str]
    refined_sub_emotions: Dict[str, Any]
    # (신규) 감정 변화량 필드: 기본값 비어있는 리스트
    emotion_changes: List[Dict[str, Any]] = field(default_factory=list)

    # ---------- 확장(선택) 필드들: 하위호환 영향 없음 ----------
    # 분석 시각 (ISO 문자열 또는 datetime 모두 허용; 직렬화 시 ISO로 변환)
    analysis_timestamp: Optional[Union[str, datetime]] = None
    # 진단용/디버깅 메타 (예: 내부 스코어/가중치 스냅샷, 규칙 히트 로그 등)
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    # 실행 시 사용된 설정 스냅샷 (읽기 전용 추천)
    config_snapshot: Optional[Dict[str, Any]] = None

    # ---------------------------------------------------------------------
    # 직렬화/역직렬화 유틸
    # ---------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """JSON 직렬화 친화적 dict 변환 (datetime -> ISO 문자열)"""
        data = asdict(self)
        ts = data.get("analysis_timestamp")
        if isinstance(ts, datetime):
            data["analysis_timestamp"] = ts.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnalysisResult":
        """dict로부터 안전 복원 (analysis_timestamp가 ISO면 datetime으로 복원 시도 X: 그대로 유지)"""
        return cls(
            emotion_sequence=list(data.get("emotion_sequence", [])),
            summary=dict(data.get("summary", {})),
            cause_effect=list(data.get("cause_effect", [])),
            time_flow=dict(data.get("time_flow", {})),
            complex_emotions=list(data.get("complex_emotions", [])),
            refined_sub_emotions=dict(data.get("refined_sub_emotions", {})),
            emotion_changes=list(data.get("emotion_changes", [])),
            analysis_timestamp=data.get("analysis_timestamp"),
            diagnostics=dict(data.get("diagnostics", {})),
            config_snapshot=data.get("config_snapshot"),
        )

    @classmethod
    def empty(cls) -> "AnalysisResult":
        """빈 결과 생성기 (파이프라인 초기화용)"""
        return cls(
            emotion_sequence=[],
            summary={},
            cause_effect=[],
            time_flow={},
            complex_emotions=[],
            refined_sub_emotions={},
            emotion_changes=[],
            analysis_timestamp=None,
            diagnostics={},
            config_snapshot=None,
        )

    # ---------------------------------------------------------------------
    # 편의 메서드 (분석 파이프라인 단계별 축적에 최적화)
    # ---------------------------------------------------------------------
    def append_sequence_item(self, sentence_index: int, sentence: str, emotions: Dict[str, float]) -> None:
        self.emotion_sequence.append({
            "sentence_index": sentence_index,
            "sentence": sentence,
            "emotions": dict(emotions),
        })

    def add_cause_effect(
        self,
        from_sentence_index: int,
        to_sentence_index: int,
        cause_patterns: List[Dict[str, Any]],
        effect_patterns: List[Dict[str, Any]],
        transition_patterns: List[Dict[str, Any]],
        combined_score: float,
        from_emotions: Dict[str, float],
        to_emotions: Dict[str, float],
    ) -> None:
        self.cause_effect.append({
            "from_sentence_index": from_sentence_index,
            "to_sentence_index": to_sentence_index,
            "cause_patterns": list(cause_patterns),
            "effect_patterns": list(effect_patterns),
            "transition_patterns": list(transition_patterns),
            "combined_score": float(combined_score),
            "from_emotions": dict(from_emotions),
            "to_emotions": dict(to_emotions),
        })

    def add_emotion_change(self, sentence_index: int, time: Any, emotion_deltas: Dict[str, float]) -> None:
        self.emotion_changes.append({
            "sentence_index": sentence_index,
            "time": time,
            "emotion_deltas": dict(emotion_deltas),
        })

    def set_summary(self, emotion_count: int, most_frequent_emotions: List[List[Any]]) -> None:
        self.summary = {
            "emotion_count": int(emotion_count),
            "most_frequent_emotions": list(most_frequent_emotions),
        }

    def set_time_flow(self, timestamps: Dict[str, Any], detected_events: List[Dict[str, Any]], note: str) -> None:
        self.time_flow = {
            "timestamps": dict(timestamps),
            "detected_events": list(detected_events),
            "note": note,
        }

    def add_complex_emotion(self, description: str) -> None:
        self.complex_emotions.append(description)

    def update_refined_sub_emotions(self, updates: Dict[str, float], mode: str = "max") -> None:
        """
        세부감정 스코어 병합.
        - mode="max": 기존/신규 중 큰 값 선택 (기본)
        - mode="sum": 합산
        - mode="avg": 평균 (새 값만 들어올 경우는 신규 그대로)
        ※ 4대 대표감정 × N(가변) 구조를 전제로, 감정 수 증가에도 로직 변경 불요.
        """
        for k, v in updates.items():
            if k not in self.refined_sub_emotions:
                self.refined_sub_emotions[k] = float(v)
                continue
            if mode == "sum":
                self.refined_sub_emotions[k] = float(self.refined_sub_emotions[k]) + float(v)
            elif mode == "avg":
                prev = float(self.refined_sub_emotions[k])
                self.refined_sub_emotions[k] = (prev + float(v)) / 2.0
            else:  # "max"
                self.refined_sub_emotions[k] = max(float(self.refined_sub_emotions[k]), float(v))

    # ---------------------------------------------------------------------
    # 병합 유틸 (멀티 세그먼트/배치 분석 결과를 하나로 합치고자 할 때)
    # ---------------------------------------------------------------------
    def merge(self, other: "AnalysisResult", merge_strategy: str = "append", refined_mode: str = "max") -> "AnalysisResult":
        """
        두 AnalysisResult를 병합.
        - merge_strategy="append": 시퀀스/인과/변화/이벤트를 그대로 이어붙임(기본)
        - summary는 호출자가 재계산하는 것을 권장(케이스마다 정의가 다름)
        - refined_sub_emotions는 `refined_mode`에 따라 병합
        - diagnostics/config_snapshot은 첫 번째 것을 유지(필요 시 외부에서 관리)
        """
        if merge_strategy != "append":
            raise ValueError("merge_strategy currently supports only 'append'.")

        merged = AnalysisResult(
            emotion_sequence=[*self.emotion_sequence, *other.emotion_sequence],
            summary=dict(self.summary),  # 호출자 재계산 권장
            cause_effect=[*self.cause_effect, *other.cause_effect],
            time_flow=self._merge_time_flow(self.time_flow, other.time_flow),
            complex_emotions=[*self.complex_emotions, *other.complex_emotions],
            refined_sub_emotions=dict(self.refined_sub_emotions),
            emotion_changes=[*self.emotion_changes, *other.emotion_changes],
            analysis_timestamp=self.analysis_timestamp or other.analysis_timestamp,
            diagnostics=dict(self.diagnostics),
            config_snapshot=self.config_snapshot or other.config_snapshot,
        )
        merged.update_refined_sub_emotions(other.refined_sub_emotions, mode=refined_mode)
        return merged

    @staticmethod
    def _merge_time_flow(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
        """
        time_flow 병합 규칙(보수적):
        - timestamps: 키 충돌 시 a 우선, b의 신규 키만 추가
        - detected_events: 단순 이어붙임
        - note: a의 note 유지 (필요 시 상위에서 메시지 재작성)
        """
        if not a:
            return dict(b)
        if not b:
            return dict(a)

        merged_ts = {}
        merged_ts.update(a.get("timestamps", {}))
        for k, v in b.get("timestamps", {}).items():
            if k not in merged_ts:
                merged_ts[k] = v

        merged_events = list(a.get("detected_events", [])) + list(b.get("detected_events", []))
        merged_note = a.get("note", "") or b.get("note", "")

        return {
            "timestamps": merged_ts,
            "detected_events": merged_events,
            "note": merged_note,
        }



# =============================================================================
# EmotionDataManager (Improved, JSON-Driven, No fragile hardcoding)
# =============================================================================
# 안전 가드: logger 미정의 시 로컬 로거 사용
try:
    logger  # type: ignore
except NameError:  # pragma: no cover
    import logging
    logger = logging.getLogger(__name__)

import re, copy
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import defaultdict, Counter

try:
    logger  # type: ignore
except NameError:  # pragma: no cover
    import logging
    logger = logging.getLogger(__name__)


class EmotionDataManager:
    def __init__(self, emotions_data: Dict[str, Any]):
        self.emotions_data = emotions_data
        # (선택) 캐시 접근 동시성 보강용 락
        self._lock: RLock = RLock()

        self.flat_emotions: Dict[str, Dict[str, Any]] = {}
        self.emotion_transitions_cache: Dict[str, Dict[str, Any]] = {}

        self._polarity_cache: Dict[str, str] = {}
        self._polarity_strength_cache: Dict[str, float] = {}
        self._conflict_with_cache: Dict[str, Set[str]] = {}
        self._conditional_conflicts_cache: Dict[str, Dict[str, Any]] = {}
        self._synergy_with_cache: Dict[str, Set[str]] = {}
        self._conditional_synergy_cache: Dict[str, Dict[str, Any]] = {}
        self._emotion_ids_cache: List[str] = []
        self._profile_cache: Dict[str, Dict[str, Any]] = {}
        self._metadata_cache: Dict[str, Dict[str, Any]] = {}
        self._transition_cache: Dict[str, Dict[str, Any]] = {}
        self._complexity_cache: Dict[str, str] = {}
        self._subcategory_cache: Dict[str, str] = {}
        self._intensity_cache: Dict[str, Dict[str, Any]] = {}
        self._linguistic_cache: Dict[str, Dict[str, Any]] = {}
        self._related_cache: Dict[str, Dict[str, List[str]]] = {}
        self._situation_cache: Dict[str, Dict[str, Any]] = {}

        self._compiled_regex_cache: Dict[str, re.Pattern] = {}
        self._global_marker_cache: Dict[str, Set[str]] = {"cause": set(), "contrast": set()}
        self._global_marker_counter: Dict[str, Counter] = {"cause": Counter(), "contrast": Counter()}
        self._CAUSE_HINT_RE = re.compile(r"(그래서|그러므로|때문에|하여|따라서|그\s*결과|그로\s*인해|그러자|덕분)", re.I)
        self._CONTRAST_HINT_RE = re.compile(r"(하지만|그러나|반면|반대로|대신)", re.I)

        self._flatten_emotions()
        self._process_transitions()
        self._build_conflict_polarity_info()
        self._build_synergy_info()
        self._rebuild_marker_caches()

    def _compile_boundary_regex(self, text: str) -> Optional[re.Pattern]:
        if not isinstance(text, str):
            return None
        key = f"b::{text}"
        if key in self._compiled_regex_cache:
            return self._compiled_regex_cache[key]
        txt = text.strip()
        if not txt:
            return None
        pat = r"(?<!\w)" + re.escape(txt) + r"(?!\w)"
        try:
            cp = re.compile(pat, re.IGNORECASE)
        except re.error:
            cp = re.compile(re.escape(txt), re.IGNORECASE)
        self._compiled_regex_cache[key] = cp
        return cp

    def _compile_plain_regex(self, text: str) -> Optional[re.Pattern]:
        if not isinstance(text, str):
            return None
        key = f"p::{text}"
        if key in self._compiled_regex_cache:
            return self._compiled_regex_cache[key]
        txt = text.strip()
        if not txt:
            return None
        try:
            cp = re.compile(re.escape(txt), re.IGNORECASE)
        except re.error:
            return None
        self._compiled_regex_cache[key] = cp
        return cp

    def _add_marker(self, kind: str, phrase: str) -> None:
        p = phrase.strip()
        if not p:
            return
        kind = "contrast" if kind == "contrast" else "cause"
        self._global_marker_cache[kind].add(p)
        self._global_marker_counter[kind][p] += 1

    def _rebuild_marker_caches(self) -> None:
        self._global_marker_cache["cause"].clear()
        self._global_marker_cache["contrast"].clear()
        self._global_marker_counter["cause"].clear()
        self._global_marker_counter["contrast"].clear()

        for e_id in self.get_emotion_ids():
            trans = self.emotion_transitions_cache.get(e_id, {}) or {}
            for pat in trans.get("patterns", []) or []:
                if not isinstance(pat, dict):
                    continue
                t = str(pat.get("type", "")).lower()
                s = str(pat.get("pattern", "")).strip()
                if not s:
                    continue
                if t == "cause":
                    self._add_marker("cause", s)
                elif t == "contrast":
                    self._add_marker("contrast", s)

            ling = self.flat_emotions.get(e_id, {}).get("linguistic_patterns", {}) or {}
            for kp in ling.get("key_phrases", []) or []:
                if isinstance(kp, dict):
                    ptn = str(kp.get("pattern", "")).strip()
                    ctx = str(kp.get("context_requirement", "")).strip()
                    if not ptn:
                        continue
                    if self._CAUSE_HINT_RE.search(ptn) or (ctx and self._CAUSE_HINT_RE.search(ctx)):
                        self._add_marker("cause", ptn)
                    if self._CONTRAST_HINT_RE.search(ptn) or (ctx and self._CONTRAST_HINT_RE.search(ctx)):
                        self._add_marker("contrast", ptn)
                elif isinstance(kp, str):
                    ptn = kp.strip()
                    if not ptn:
                        continue
                    if self._CAUSE_HINT_RE.search(ptn):
                        self._add_marker("cause", ptn)
                    if self._CONTRAST_HINT_RE.search(ptn):
                        self._add_marker("contrast", ptn)

    def _get_expected_per_category(self, default: int = 30) -> int:
        required_categories = ['희', '노', '애', '락']
        counts = []
        for cat in required_categories:
            meta = self.emotions_data.get(cat, {}).get('metadata', {}) if isinstance(self.emotions_data.get(cat), dict) else {}
            if isinstance(meta, dict) and isinstance(meta.get('expected_count', None), int):
                counts.append(int(meta['expected_count']))
        if counts and all(isinstance(c, int) and c > 0 for c in counts):
            return int(sum(counts) / len(counts))
        return int(default)

    def _derive_category_polarities(self) -> Dict[str, str]:
        category_polarity: Dict[str, str] = {}
        required_categories = ['희', '노', '애', '락']
        for cat in required_categories:
            meta = self.emotions_data.get(cat, {}).get('metadata', {})
            if isinstance(meta, dict):
                pol = meta.get('default_polarity')
                if isinstance(pol, str) and pol.lower() in {'positive', 'negative', 'neutral'}:
                    category_polarity[cat] = pol.lower()
        for cat in required_categories:
            if cat in category_polarity:
                continue
            votes = defaultdict(int)
            for e_id, e_data in self.flat_emotions.items():
                md = e_data.get('metadata', {})
                if md.get('primary_category') != cat:
                    continue
                pol = e_data.get('ml_training_metadata', {}).get('polarity')
                if isinstance(pol, str) and pol.lower() in {'positive', 'negative', 'neutral'}:
                    votes[pol.lower()] += 1
            if votes:
                category_polarity[cat] = max(votes.items(), key=lambda x: x[1])[0]
        for cat in required_categories:
            if cat not in category_polarity:
                if cat in ('희', '락'):
                    category_polarity[cat] = 'positive'
                elif cat in ('노', '애'):
                    category_polarity[cat] = 'negative'
                else:
                    category_polarity[cat] = 'neutral'
        return category_polarity

    def _category_of(self, emotion_id: str) -> Optional[str]:
        return self.flat_emotions.get(emotion_id, {}).get('metadata', {}).get('primary_category')

    def _build_conflict_polarity_info(self):
        self._polarity_cache.clear()
        self._polarity_strength_cache.clear()
        self._conflict_with_cache.clear()
        self._conditional_conflicts_cache.clear()

        category_polarity = self._derive_category_polarities()

        for e_id, e_data in self.flat_emotions.items():
            ml_data = e_data.get('ml_training_metadata', {}) or {}
            metadata = e_data.get('metadata', {}) or {}

            explicit = ml_data.get('polarity')
            if isinstance(explicit, str) and explicit.lower() in {'positive', 'negative', 'neutral'}:
                self._polarity_cache[e_id] = explicit.lower()
            else:
                cat = metadata.get('primary_category', '')
                self._polarity_cache[e_id] = category_polarity.get(cat, 'neutral')

            strength = ml_data.get('polarity_strength', 1.0)
            try:
                strength_f = float(strength)
            except (TypeError, ValueError):
                strength_f = 1.0
            self._polarity_strength_cache[e_id] = max(0.1, min(strength_f, 2.0))

            conflicts: Set[str] = set()
            conditional_conflicts: Dict[str, Dict[str, Any]] = {}

            direct_conflicts = ml_data.get('conflict_with', [])
            if isinstance(direct_conflicts, list):
                for c in direct_conflicts:
                    if isinstance(c, str):
                        conflicts.add(c)

            emotion_profile = e_data.get('emotion_profile', {})
            if isinstance(emotion_profile, dict):
                related = emotion_profile.get('related_emotions', {})
                if isinstance(related, dict):
                    neg = related.get('negative', [])
                    if isinstance(neg, list):
                        for c in neg:
                            if isinstance(c, str):
                                conflicts.add(c)

            transitions = e_data.get('transitions', {}) or {}
            patterns = transitions.get('patterns', [])
            if isinstance(patterns, list):
                for p in patterns:
                    if isinstance(p, dict) and p.get('type') == 'conflict':
                        tgt = p.get('target_emotion')
                        if isinstance(tgt, str):
                            conflicts.add(tgt)
                            conds = p.get('conditions', {})
                            if isinstance(conds, dict) and conds:
                                conditional_conflicts.setdefault(tgt, {}).update(conds)

            if isinstance(ml_data, dict):
                intensity_conflicts = ml_data.get('intensity_based_conflicts', {})
                if isinstance(intensity_conflicts, dict):
                    for tgt, thr in intensity_conflicts.items():
                        if isinstance(tgt, str) and isinstance(thr, (int, float)):
                            cc = conditional_conflicts.setdefault(tgt, {})
                            cc.setdefault('intensity_threshold', [])
                            if isinstance(cc['intensity_threshold'], list):
                                cc['intensity_threshold'].append(float(thr))
                            else:
                                cc['intensity_threshold'] = [float(thr)]
                ctx_conflicts = ml_data.get('context_based_conflicts', [])
                if isinstance(ctx_conflicts, list):
                    for cc_item in ctx_conflicts:
                        if not isinstance(cc_item, dict):
                            continue
                        tgt = cc_item.get('target_emotion')
                        ctx = cc_item.get('context')
                        if isinstance(tgt, str) and ctx:
                            cc = conditional_conflicts.setdefault(tgt, {})
                            ctx_list = cc.setdefault('contexts', [])
                            if isinstance(ctx_list, list):
                                ctx_list.append(ctx)

            cat_me = metadata.get('primary_category')
            if cat_me and conflicts:
                opp_candidates = set()
                for cid in conflicts:
                    oc = self._category_of(cid)
                    if oc:
                        opp_candidates.add(oc)
                if opp_candidates:
                    freq = defaultdict(int)
                    for cid in conflicts:
                        oc = self._category_of(cid)
                        if oc:
                            freq[oc] += 1
                    if freq:
                        max_count = max(freq.values())
                        dominant_cats = {k for k, v in freq.items() if v == max_count}
                        for dcat in dominant_cats:
                            for other_id, other_data in self.flat_emotions.items():
                                if other_data.get('metadata', {}).get('primary_category') == dcat:
                                    conflicts.add(other_id)

            self._conflict_with_cache[e_id] = conflicts
            if conditional_conflicts:
                self._conditional_conflicts_cache[e_id] = conditional_conflicts

        self._validate_conflict_relationships()

    def _build_synergy_info(self):
        if getattr(self, "_synergy_with_cache", None) is None:
            self._synergy_with_cache = {}
        if getattr(self, "_conditional_synergy_cache", None) is None:
            self._conditional_synergy_cache = {}

        self._synergy_with_cache.clear()
        self._conditional_synergy_cache.clear()

        for e_id, e_data in self.flat_emotions.items():
            ml_data = e_data.get('ml_training_metadata', {}) or {}

            sset: Set[str] = set()
            s_list = ml_data.get('synergy_with', [])
            if isinstance(s_list, list):
                for sid in s_list:
                    if isinstance(sid, str):
                        sset.add(sid)
            self._synergy_with_cache[e_id] = sset

            csy = {}
            cond = ml_data.get('conditional_synergy', {})
            if isinstance(cond, dict):
                for tgt, cval in cond.items():
                    if isinstance(tgt, str) and isinstance(cval, dict):
                        csy[tgt] = cval
            if csy:
                self._conditional_synergy_cache[e_id] = csy

    def _validate_conflict_relationships(self):
        for e_id, conflicts in list(self._conflict_with_cache.items()):
            for cid in list(conflicts):
                self._conflict_with_cache.setdefault(cid, set()).add(e_id)
                e_pol = self._polarity_cache.get(e_id, 'neutral')
                c_pol = self._polarity_cache.get(cid, 'neutral')
                if e_pol == c_pol and e_pol != 'neutral':
                    logger.warning(
                        f"[conflict-check] Potentially incorrect conflict between {e_id} and {cid}: same polarity={e_pol}"
                    )

    def get_polarity(self, emotion_id: str) -> str:
        return self._polarity_cache.get(emotion_id, 'neutral')

    def get_conflict_with_list(self, emotion_id: str) -> Set[str]:
        return self._conflict_with_cache.get(emotion_id, set())

    def get_conditional_conflicts(self, emotion_id: str) -> Dict[str, Dict[str, Any]]:
        return self._conditional_conflicts_cache.get(emotion_id, {})

    def get_synergy_with_list(self, emotion_id: str) -> Set[str]:
        return self._synergy_with_cache.get(emotion_id, set())

    def get_conditional_synergy(self, emotion_id: str) -> Dict[str, Dict[str, Any]]:
        return self._conditional_synergy_cache.get(emotion_id, {})

    def _flatten_emotions(self):
        required_categories = {'희', '노', '애', '락'}
        if not all(cat in self.emotions_data for cat in required_categories):
            logger.error("필수 감정 카테고리가 누락되었습니다: (희, 노, 애, 락)")
            raise ValueError("Required emotion categories missing")

        processing_queue: List[Tuple[str, Dict[str, Any], Optional[Dict[str, Any]], List[str], int]] = []
        emotion_counts = {cat: 0 for cat in required_categories}
        for k, v in self.emotions_data.items():
            if isinstance(v, dict) and k in required_categories:
                processing_queue.append((k, v, None, [], 0))

        while processing_queue:
            e_key, data, parent, path, depth = processing_queue.pop(0)
            meta = (data.get('metadata') or {})
            meta_copy = meta if isinstance(meta, dict) else {}
            meta = dict(meta_copy)
            if not meta.get('primary_category') and parent:
                pmeta = parent.get('metadata', {})
                if isinstance(pmeta, dict):
                    meta['primary_category'] = pmeta.get('primary_category', '')

            emotion_id = meta.get('emotion_id')
            if not isinstance(emotion_id, str) or not emotion_id.strip():
                emotion_id = '-'.join(path + [e_key]) if path else e_key
                meta['emotion_id'] = emotion_id

            cat = meta.get('primary_category')
            if cat in emotion_counts:
                emotion_counts[cat] += 1

            emotion_profile = self._process_emotion_profile(data.get('emotion_profile', {}) or {})
            context_patterns = self._process_context_patterns(data.get('context_patterns', {}) or {})
            linguistic_patterns = self._process_linguistic_patterns(data.get('linguistic_patterns', {}) or {})

            current = {
                'emotion_id': emotion_id,
                'full_path': path + [e_key],
                'depth': depth,
                'parent_data': parent,
                'metadata': meta,
                'emotion_profile': emotion_profile,
                'transitions': data.get('emotion_transitions', {}) or {},
                'context_patterns': context_patterns,
                'linguistic_patterns': linguistic_patterns,
                'ml_training_metadata': data.get('ml_training_metadata', {}) or {}
            }
            self.flat_emotions[emotion_id] = current

            sub = data.get('sub_emotions', {})
            if isinstance(sub, dict):
                new_path = path + [e_key]
                for sk, sv in sub.items():
                    if isinstance(sv, dict):
                        processing_queue.append((sk, sv, current, new_path, depth + 1))

            for sub_cat, sub_data in data.items():
                if sub_cat in {'metadata', 'emotion_profile', 'emotion_transitions', 'context_patterns', 'linguistic_patterns', 'ml_training_metadata', 'sub_emotions'}:
                    continue
                if isinstance(sub_data, dict):
                    new_path = path + [e_key]
                    processing_queue.append((sub_cat, sub_data, current, new_path, depth + 1))

        counts = list(emotion_counts.values())
        expected = self._get_expected_per_category(default=30)
        if len(counts) >= 4:
            sc = sorted(counts)
            n = len(sc)
            median = (sc[n//2] if n % 2 else (sc[n//2 - 1] + sc[n//2]) / 2)
            q1 = sc[n//4]
            q3 = sc[(3*n)//4 - (1 if (3*n)//4 == n else 0)]
            iqr = max(1.0, float(q3) - float(q1))
            upper = int(round(median + iqr))
            lower = int(round(max(5.0, median - iqr/2)))
        else:
            upper = expected
            lower = max(5, expected // 2)

        for category, count in emotion_counts.items():
            if count > upper:
                logger.warning(f"{category} 카테고리의 감정이 기대치({upper})를 초과합니다: {count}개")
            elif count < lower:
                logger.warning(f"{category} 카테고리의 감정이 기대치({lower})에 비해 적습니다: {count}개")

        self._validate_emotion_relationships()
        logger.info(f"총 {len(self.flat_emotions)}개의 감정이 처리되었습니다.")

    def _process_linguistic_patterns(self, lp_data: Dict[str, Any]) -> Dict[str, Any]:
        processed = {"key_phrases": [], "sentiment_combinations": [], "sentiment_modifiers": {"amplifiers": [], "diminishers": []}}
        if not isinstance(lp_data, dict):
            return processed

        seen = set()
        for item in lp_data.get("key_phrases", []) or []:
            if isinstance(item, str):
                pat = item.strip()
                if pat and pat not in seen:
                    processed["key_phrases"].append({"pattern": pat, "weight": 1.0, "context_requirement": "", "compiled": self._compile_boundary_regex(pat)})
                    seen.add(pat)
            elif isinstance(item, dict):
                pat = str(item.get("pattern", "")).strip()
                if pat and pat not in seen:
                    try:
                        w = float(item.get("weight", 1.0))
                    except (TypeError, ValueError):
                        w = 1.0
                    processed["key_phrases"].append({"pattern": pat, "weight": w, "context_requirement": str(item.get("context_requirement", "")).strip(), "compiled": self._compile_boundary_regex(pat)})
                    seen.add(pat)

        seen_combo = set()
        for combo in lp_data.get("sentiment_combinations", []) or []:
            if not isinstance(combo, dict):
                continue
            words = [w.strip() for w in combo.get("words", []) if isinstance(w, str) and w.strip()]
            if not words:
                continue
            key = tuple(sorted(words))
            if key in seen_combo:
                continue
            try:
                w = float(combo.get("weight", 1.0))
            except (TypeError, ValueError):
                w = 1.0
            processed["sentiment_combinations"].append({"words": words, "weight": w})
            seen_combo.add(key)

        mods = lp_data.get("sentiment_modifiers", {}) or {}
        if isinstance(mods, dict):
            amps = set()
            for a in mods.get("amplifiers", []) or []:
                if isinstance(a, str) and a.strip():
                    amps.add(a.strip())
            dims = set()
            for d in mods.get("diminishers", []) or []:
                if isinstance(d, str) and d.strip():
                    dims.add(d.strip())
            processed["sentiment_modifiers"]["amplifiers"] = sorted(amps)
            processed["sentiment_modifiers"]["diminishers"] = sorted(dims)

        return processed

    def _process_emotion_profile(self, profile_data: Dict[str, Any]) -> Dict[str, Any]:
        processed = copy.deepcopy(profile_data) if isinstance(profile_data, dict) else {}
        rel = processed.get('related_emotions', {})
        if isinstance(rel, dict):
            processed['related_emotions'] = {k: sorted({e for e in (v or []) if isinstance(e, str) and e.strip()})
                                             for k, v in rel.items() if isinstance(v, list)}
        if 'core_keywords' in processed and isinstance(processed['core_keywords'], list):
            processed['core_keywords'] = sorted({kw for kw in processed['core_keywords'] if isinstance(kw, str) and kw.strip()})
        if 'intensity_levels' in processed and isinstance(processed['intensity_levels'], dict):
            processed['intensity_levels'] = self._normalize_intensity_levels(processed['intensity_levels'])
        return processed

    def _process_context_patterns(self, patterns_data: Dict[str, Any]) -> Dict[str, Any]:
        processed = {}
        if not isinstance(patterns_data, dict):
            return processed
        sits = patterns_data.get('situations', {}) or {}
        if isinstance(sits, dict):
            out = {}
            for k, v in sits.items():
                if not isinstance(v, dict):
                    continue
                out[k] = {
                    'description': v.get('description', ''),
                    'intensity': v.get('intensity', 'medium'),
                    'variations': sorted({x for x in v.get('variations', []) if isinstance(x, str) and x.strip()}),
                    'keywords': sorted({x for x in v.get('keywords', []) if isinstance(x, str) and x.strip()}),
                }
            processed['situations'] = out
        return processed

    def _normalize_intensity_levels(self, levels: Dict[str, Any]) -> Dict[str, Any]:
        normalized = {}
        for lvl in ('low', 'medium', 'high'):
            data = levels.get(lvl, {})
            if not isinstance(data, dict):
                data = {}
            try:
                weight = float(data.get('weight', 1.0))
            except (TypeError, ValueError):
                weight = 1.0
            ex = data.get('intensity_examples', {}) or {}
            if isinstance(ex, dict):
                ex = {k: [s for s in (v or []) if isinstance(s, str) and s.strip()] for k, v in ex.items() if isinstance(v, list)}
            else:
                ex = {}
            normalized[lvl] = {'description': data.get('description', f'{lvl} intensity level'), 'weight': weight, 'intensity_examples': ex}
        return normalized

    def _validate_emotion_relationships(self):
        for eid, ed in self.flat_emotions.items():
            rel = ed.get('emotion_profile', {}).get('related_emotions', {})
            if not isinstance(rel, dict):
                continue
            for cat in list(rel.keys()):
                if not isinstance(rel[cat], list):
                    rel[cat] = []
                rel[cat] = [x for x in rel[cat] if x in self.flat_emotions]

    def _process_transitions(self):
        for e_id, e_data in self.flat_emotions.items():
            transitions = e_data.get('transitions', {}) or {}
            processed = {'patterns': [], 'multi': [], 'rules': [], 'dependencies': []}

            raw_patterns = transitions.get('patterns')
            if isinstance(raw_patterns, list):
                seen = set()
                for pat in raw_patterns:
                    if isinstance(pat, str):
                        key = pat.strip()
                        if key and key not in seen:
                            processed['patterns'].append({'pattern': key, 'type': 'basic', 'weight': 1.0, 'conditions': {}, 'advanced_conditions': {}, 'compiled': self._compile_boundary_regex(key)})
                            seen.add(key)
                    elif isinstance(pat, dict):
                        pstr = str(pat.get('pattern', '')).strip()
                        if pstr and pstr not in seen:
                            entry = {
                                'pattern': pstr,
                                'type': pat.get('type', 'basic'),
                                'weight': float(pat.get('weight', 1.0)) if isinstance(pat.get('weight', 1.0), (int, float, str)) else 1.0,
                                'conditions': pat.get('conditions', {}) or {},
                                'advanced_conditions': pat.get('advanced_conditions', {}) or {},
                                'compiled': self._compile_boundary_regex(pstr)
                            }
                            processed['patterns'].append(entry)
                            seen.add(pstr)

            raw_multi = transitions.get('multi_emotion_transitions')
            if isinstance(raw_multi, list):
                for item in raw_multi:
                    if not isinstance(item, dict):
                        continue
                    fe = [x for x in (item.get('from_emotions', []) or []) if isinstance(x, str) and x.strip()]
                    te = [x for x in (item.get('to_emotions', []) or []) if isinstance(x, str) and x.strip()]
                    if not fe or not te:
                        continue
                    try:
                        strength = float(item.get('strength', 1.0))
                    except (TypeError, ValueError):
                        strength = 1.0
                    processed['multi'].append({
                        'from_emotions': fe,
                        'to_emotions': te,
                        'triggers': [t for t in (item.get('triggers', []) or []) if isinstance(t, str) and t.strip()],
                        'strength': strength,
                        'conditions': item.get('conditions', {}) or {},
                        'advanced_conditions': item.get('advanced_conditions', {}) or {}
                    })

            raw_rules = transitions.get('temporal_rules')
            if isinstance(raw_rules, list):
                for rule in raw_rules:
                    if not isinstance(rule, dict):
                        continue
                    try:
                        weight = float(rule.get('weight', 1.0))
                    except (TypeError, ValueError):
                        weight = 1.0
                    try:
                        trp = float(rule.get('transition_probability', 0.5))
                    except (TypeError, ValueError):
                        trp = 0.5
                    processed['rules'].append({
                        'time_window': rule.get('time_window', {}) or {},
                        'condition': rule.get('condition', '') or '',
                        'weight': weight,
                        'transition_probability': trp,
                        'advanced_conditions': rule.get('advanced_conditions', {}) or {}
                    })

            raw_deps = transitions.get('emotion_dependencies')
            if isinstance(raw_deps, list):
                for dep in raw_deps:
                    if not isinstance(dep, dict):
                        continue
                    try:
                        strength = float(dep.get('strength', 1.0))
                    except (TypeError, ValueError):
                        strength = 1.0
                    processed['dependencies'].append({
                        'dependent_emotion': dep.get('dependent_emotion', '') or '',
                        'relationship_type': dep.get('relationship_type', '') or '',
                        'strength': strength,
                        'conditions': dep.get('conditions', {}) or {},
                        'advanced_conditions': dep.get('advanced_conditions', {}) or {}
                    })

            self.emotion_transitions_cache[e_id] = processed

    def get_transitions(self, emotion_id: str) -> Dict[str, Any]:
        if emotion_id in self._transition_cache:
            return self._transition_cache[emotion_id]

        metadata = self.get_metadata(emotion_id)
        primary_category = metadata.get('primary_category', '')

        child = self.emotion_transitions_cache.get(emotion_id, {})
        merged = {
            'patterns': [],
            'multi': [],
            'rules': [],
            'dependencies': [],
            'intensity_rules': [],
            'context_transitions': [],
            'category_transitions': {}
        }

        for p in child.get('patterns', []):
            if isinstance(p, dict):
                merged['patterns'].append(self._normalize_transition_pattern(p, primary_category))
        for m in child.get('multi', []):
            if isinstance(m, dict):
                nm = self._normalize_multi_transition(m, primary_category)
                if nm:
                    merged['multi'].append(nm)
        for r in child.get('rules', []):
            if isinstance(r, dict):
                nr = self._normalize_time_rule(r)
                if nr:
                    merged['rules'].append(nr)
        for d in child.get('dependencies', []):
            if isinstance(d, dict):
                nd = self._normalize_dependency(d, primary_category)
                if nd:
                    merged['dependencies'].append(nd)

        if primary_category:
            merged['category_transitions'] = self._setup_category_transitions(primary_category)

        e_data = self.flat_emotions.get(emotion_id, {})
        parent = e_data.get('parent_data')
        if isinstance(parent, dict):
            pid = parent.get('emotion_id')
            if isinstance(pid, str):
                parent_transitions = self.get_transitions(pid)
                merged = self._merge_transition_info(parent_transitions, merged)

        self._validate_transitions(merged, emotion_id)
        opt = self._optimize_transitions(merged)
        self._transition_cache[emotion_id] = opt
        return opt

    def _normalize_transition_pattern(self, pattern: Dict[str, Any], primary_category: str) -> Dict[str, Any]:
        return {
            'pattern': pattern.get('pattern', ''),
            'type': pattern.get('type', 'basic'),
            'weight': float(pattern.get('weight', 1.0)) if isinstance(pattern.get('weight', 1.0), (int, float, str)) else 1.0,
            'primary_category': primary_category,
            'conditions': pattern.get('conditions', {}) or {},
            'probability': float(pattern.get('probability', 0.5)) if isinstance(pattern.get('probability', 0.5), (int, float, str)) else 0.5,
            'advanced_conditions': pattern.get('advanced_conditions', {}) or {},
            'compiled': self._compile_boundary_regex(pattern.get('pattern', '') or '')
        }

    def _normalize_multi_transition(self, transition: Dict[str, Any], primary_category: str) -> Optional[Dict[str, Any]]:
        fe = [x for x in (transition.get('from_emotions', []) or []) if isinstance(x, str) and x.strip()]
        te = [x for x in (transition.get('to_emotions', []) or []) if isinstance(x, str) and x.strip()]
        if not fe or not te:
            return None
        try:
            strength = float(transition.get('strength', 1.0))
        except (TypeError, ValueError):
            strength = 1.0
        return {
            'from_emotions': fe,
            'to_emotions': te,
            'triggers': [t for t in (transition.get('triggers', []) or []) if isinstance(t, str) and t.strip()],
            'strength': strength,
            'primary_category': primary_category,
            'conditions': transition.get('conditions', {}) or {},
            'advanced_conditions': transition.get('advanced_conditions', {}) or {}
        }

    def _normalize_time_rule(self, rule: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            weight = float(rule.get('weight', 1.0))
        except (TypeError, ValueError):
            weight = 1.0
        try:
            prob = float(rule.get('transition_probability', 0.5))
        except (TypeError, ValueError):
            prob = 0.5
        return {
            'time_window': rule.get('time_window', {}) or {},
            'condition': rule.get('condition', '') or '',
            'weight': weight,
            'transition_probability': prob,
            'advanced_conditions': rule.get('advanced_conditions', {}) or {}
        }

    def _normalize_dependency(self, dependency: Dict[str, Any], primary_category: str) -> Optional[Dict[str, Any]]:
        try:
            strength = float(dependency.get('strength', 1.0))
        except (TypeError, ValueError):
            strength = 1.0
        return {
            'dependent_emotion': dependency.get('dependent_emotion', '') or '',
            'relationship_type': dependency.get('relationship_type', '') or '',
            'strength': strength,
            'primary_category': primary_category,
            'conditions': dependency.get('conditions', {}) or {},
            'advanced_conditions': dependency.get('advanced_conditions', {}) or {}
        }

    def _setup_category_transitions(self, primary_category: str) -> Dict[str, Any]:
        cat_pol = self._derive_category_polarities()
        target_freq = defaultdict(float)
        total = 0.0
        for e_id, e_data in self.flat_emotions.items():
            if e_data.get('metadata', {}).get('primary_category') != primary_category:
                continue
            tr = self.emotion_transitions_cache.get(e_id, {})
            for m in tr.get('multi', []):
                if not isinstance(m, dict):
                    continue
                strength = float(m.get('strength', 1.0)) if isinstance(m.get('strength', 1.0), (int, float, str)) else 1.0
                for to_e in (m.get('to_emotions', []) or []):
                    oc = self._category_of(to_e)
                    if oc:
                        target_freq[oc] += max(0.0, strength)
                        total += max(0.0, strength)
        target_probs = {k: (v / total) for k, v in target_freq.items()} if total > 0 else {}
        pol_sums = defaultdict(float)
        for cat, p in target_probs.items():
            pol = cat_pol.get(cat, 'neutral')
            pol_sums[pol] += p
        return {
            'category': primary_category,
            'target_category_probabilities': target_probs,
            'rules': {
                'positive': pol_sums.get('positive', 0.0),
                'negative': pol_sums.get('negative', 0.0),
                'neutral': pol_sums.get('neutral', 0.0)
            }
        }

    def _validate_transitions(self, transitions: Dict[str, Any], emotion_id: str) -> None:
        for p in transitions.get('patterns', []):
            if not isinstance(p.get('weight'), (int, float)):
                p['weight'] = 1.0
        for m in transitions.get('multi', []):
            if not all(isinstance(e, str) for e in m.get('from_emotions', [])):
                m['from_emotions'] = [x for x in m.get('from_emotions', []) if isinstance(x, str)]
            if not all(isinstance(e, str) for e in m.get('to_emotions', [])):
                m['to_emotions'] = [x for x in m.get('to_emotions', []) if isinstance(x, str)]
        for r in transitions.get('rules', []):
            if not isinstance(r.get('time_window'), dict):
                r['time_window'] = {}

    def _optimize_transitions(self, transitions: Dict[str, Any]) -> Dict[str, Any]:
        optimized = copy.deepcopy(transitions)
        seen = set()
        uniq = []
        for p in optimized.get('patterns', []):
            key = f"{p.get('pattern','')}_{p.get('type','')}"
            if key not in seen:
                uniq.append(p); seen.add(key)
        optimized['patterns'] = sorted(uniq, key=lambda x: x.get('weight', 0.0), reverse=True)
        optimized['multi'] = sorted(optimized.get('multi', []), key=lambda x: x.get('strength', 0.0), reverse=True)
        return optimized

    def _merge_transition_info(self, base: Dict[str, Any], child: Dict[str, Any]) -> Dict[str, Any]:
        result = {}
        for key in ['patterns', 'multi', 'rules', 'dependencies']:
            base_list = base.get(key, [])
            child_list = child.get(key, [])
            merged, seen = [], set()
            for item in base_list + child_list:
                ukey = self._make_transition_unique_key(item)
                if ukey not in seen:
                    merged.append(item)
                    seen.add(ukey)
            result[key] = merged
        ct = dict(base.get('category_transitions', {}))
        ct.update(child.get('category_transitions', {}))
        result['category_transitions'] = ct
        return result

    def _make_transition_unique_key(self, item: Any) -> str:
        if isinstance(item, str):
            return f"s::{item}"
        if isinstance(item, dict):
            return "d::" + "|".join([
                str(item.get('pattern', '')),
                str(item.get('type', '')),
                ",".join(item.get('from_emotions', []) or []),
                ",".join(item.get('to_emotions', []) or []),
                str(item.get('condition', '')),
            ])
        return f"o::{repr(item)}"

    def gather_causal_and_contrast_markers(self) -> Dict[str, Set[str]]:
        if self._global_marker_cache["cause"] or self._global_marker_cache["contrast"]:
            return {
                "cause": set(self._global_marker_cache["cause"]),
                "contrast": set(self._global_marker_cache["contrast"]),
            }

        cause_set: Set[str] = set()
        contrast_set: Set[str] = set()

        for e_id in self.get_emotion_ids():
            ml = self.flat_emotions[e_id].get('ml_training_metadata', {}) or {}
            if isinstance(ml, dict):
                for c in ml.get('cause_markers', []) or []:
                    if isinstance(c, str) and c.strip():
                        cause_set.add(c.strip())
                for c in ml.get('contrast_markers', []) or []:
                    if isinstance(c, str) and c.strip():
                        contrast_set.add(c.strip())
            ling = self.flat_emotions[e_id].get('linguistic_patterns', {}) or {}
            if isinstance(ling, dict):
                for c in ling.get('cause_markers', []) or []:
                    if isinstance(c, str) and c.strip():
                        cause_set.add(c.strip())
                for c in ling.get('contrast_markers', []) or []:
                    if isinstance(c, str) and c.strip():
                        contrast_set.add(c.strip())
            trans = self.emotion_transitions_cache.get(e_id, {}) or {}
            for pat in trans.get('patterns', []) or []:
                if isinstance(pat, dict):
                    if pat.get('type') == 'cause':
                        s = str(pat.get('pattern', '')).strip()
                        if s:
                            cause_set.add(s)
                    elif pat.get('type') == 'contrast':
                        s = str(pat.get('pattern', '')).strip()
                        if s:
                            contrast_set.add(s)

        if cause_set:
            for s in cause_set:
                self._add_marker("cause", s)
        if contrast_set:
            for s in contrast_set:
                self._add_marker("contrast", s)

        return {
            "cause": set(self._global_marker_cache["cause"]),
            "contrast": set(self._global_marker_cache["contrast"]),
        }

    def get_emotion_ids(self) -> List[str]:
        if self._emotion_ids_cache:
            return self._emotion_ids_cache
        id_with_depth = [(eid, ed.get('depth', 0)) for eid, ed in self.flat_emotions.items()]
        id_with_depth.sort(key=lambda x: (x[1], x[0]))
        out, seen = [], set()
        for eid, _ in id_with_depth:
            if eid not in seen:
                out.append(eid)
                seen.add(eid)
        self._emotion_ids_cache = out
        return out

    def get_emotion_profile(self, emotion_id: str) -> Dict[str, Any]:
        if emotion_id in self._profile_cache:
            return copy.deepcopy(self._profile_cache[emotion_id])

        def merge_dicts(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
            r = copy.deepcopy(a)
            for k, v in b.items():
                if k not in r:
                    r[k] = copy.deepcopy(v)
                else:
                    if isinstance(r[k], dict) and isinstance(v, dict):
                        r[k] = merge_dicts(r[k], v)
                    elif isinstance(r[k], list) and isinstance(v, list):
                        merged = r[k] + [x for x in v if x not in r[k]]
                        r[k] = merged
                    else:
                        r[k] = copy.deepcopy(v)
            return r

        ed = self.flat_emotions.get(emotion_id, {})
        cur = copy.deepcopy(ed.get('emotion_profile', {}) or {})
        if 'intensity_levels' in cur and isinstance(cur['intensity_levels'], dict):
            cur['intensity_levels'] = self._normalize_intensity_levels(cur['intensity_levels'])
        rel = cur.get('related_emotions', {})
        if isinstance(rel, dict):
            cur['related_emotions'] = {k: [x for x in v if isinstance(x, str) and x.strip()] for k, v in rel.items() if isinstance(v, list)}
        if 'core_keywords' in cur and isinstance(cur['core_keywords'], list):
            cur['core_keywords'] = [x for x in cur['core_keywords'] if isinstance(x, str) and x.strip()]

        parent = ed.get('parent_data')
        if isinstance(parent, dict):
            pid = parent.get('emotion_id')
            if isinstance(pid, str):
                pprof = self.get_emotion_profile(pid)
                cur = merge_dicts(pprof, cur)

        md = ed.get('metadata', {}) or {}
        if isinstance(md, dict):
            if md.get('primary_category'):
                cur['category'] = md['primary_category']
            if md.get('emotion_complexity'):
                cur['complexity'] = md['emotion_complexity']

        self._profile_cache[emotion_id] = cur
        return copy.deepcopy(cur)

    def get_emotion_complexity(self, emotion_id: str) -> str:
        """Return metadata.emotion_complexity or 'basic' when missing."""
        try:
            metadata = self.get_metadata(emotion_id)
            value = metadata.get("emotion_complexity")
            return value if isinstance(value, str) and value else "basic"
        except Exception:
            return "basic"

    def get_metadata(self, emotion_id: str) -> Dict[str, Any]:
        if emotion_id in self._metadata_cache:
            return self._metadata_cache[emotion_id]
        ed = self.flat_emotions.get(emotion_id, {})
        cur = ed.get('metadata', {}) or {}
        parent = ed.get('parent_data')
        if isinstance(parent, dict):
            pid = parent.get('emotion_id')
            if isinstance(pid, str):
                pmd = self.get_metadata(pid)
                tmp = dict(pmd)
                tmp.update(cur)
                cur = tmp
        self._metadata_cache[emotion_id] = cur
        return cur

    def get_intensity_levels(self, emotion_id: str) -> Optional[Dict[str, Any]]:
        if emotion_id in self._intensity_cache:
            return self._intensity_cache[emotion_id]
        prof = self.get_emotion_profile(emotion_id)
        levels = prof.get('intensity_levels')
        if not isinstance(levels, dict):
            self._intensity_cache[emotion_id] = None
            return None
        cleaned = {}
        for k, v in levels.items():
            if not isinstance(v, dict):
                continue
            cleaned[k] = {
                'weight': v.get('weight', 1.0),
                'intensity_examples': {ek: [x for x in ev if isinstance(x, str) and x] for ek, ev in (v.get('intensity_examples', {}) or {}).items() if isinstance(ev, list)}
            }
        self._intensity_cache[emotion_id] = cleaned
        return cleaned

    def get_linguistic_patterns(self, emotion_id: str) -> Optional[Dict[str, Any]]:
        if emotion_id in self._linguistic_cache:
            return self._linguistic_cache[emotion_id]

        child = self.flat_emotions.get(emotion_id, {}).get('linguistic_patterns') or {}
        merged = {'key_phrases': [], 'sentiment_combinations': [], 'sentiment_modifiers': {}}
        if isinstance(child, dict):
            if isinstance(child.get('key_phrases'), list):
                merged['key_phrases'] = child['key_phrases'][:]
            if isinstance(child.get('sentiment_combinations'), list):
                merged['sentiment_combinations'] = child['sentiment_combinations'][:]
            if isinstance(child.get('sentiment_modifiers'), dict):
                merged['sentiment_modifiers'] = dict(child['sentiment_modifiers'])

        ed = self.flat_emotions.get(emotion_id, {})
        parent = ed.get('parent_data')
        if isinstance(parent, dict):
            pid = parent.get('emotion_id')
            if isinstance(pid, str):
                plp = self.get_linguistic_patterns(pid)
                if isinstance(plp, dict):
                    merged = self._merge_linguistic_patterns(plp, merged)

        self._linguistic_cache[emotion_id] = merged
        return merged

    def _merge_linguistic_patterns(self, base_lp: Dict[str, Any], child_lp: Dict[str, Any]) -> Dict[str, Any]:
        result = {'key_phrases': [], 'sentiment_combinations': [], 'sentiment_modifiers': {}}
        seen = set()
        for src in (base_lp.get('key_phrases', []), child_lp.get('key_phrases', [])):
            for kp in src:
                if isinstance(kp, dict):
                    pat = kp.get('pattern', '')
                    if pat and pat not in seen:
                        result['key_phrases'].append(kp)
                        seen.add(pat)
                elif isinstance(kp, str) and kp not in seen:
                    result['key_phrases'].append({'pattern': kp})
                    seen.add(kp)
        seen_combo = set()
        for src in (base_lp.get('sentiment_combinations', []), child_lp.get('sentiment_combinations', [])):
            for sc in src:
                if isinstance(sc, dict):
                    key = tuple(sc.get('words', []) or [])
                    if key and key not in seen_combo:
                        result['sentiment_combinations'].append(sc)
                        seen_combo.add(key)
        merged_sm = dict(base_lp.get('sentiment_modifiers', {}))
        merged_sm.update(child_lp.get('sentiment_modifiers', {}))
        result['sentiment_modifiers'] = merged_sm
        return result

    def get_related_emotions(self, emotion_id: str) -> Dict[str, List[str]]:
        if emotion_id in self._related_cache:
            return self._related_cache[emotion_id]
        prof = self.get_emotion_profile(emotion_id)
        raw = prof.get('related_emotions', {}) or {}
        out = {'positive': [], 'negative': [], 'neutral': []}
        for k in out.keys():
            if isinstance(raw.get(k), list):
                out[k] = list(raw[k])
        ed = self.flat_emotions.get(emotion_id, {})
        parent = ed.get('parent_data')
        if isinstance(parent, dict):
            pid = parent.get('emotion_id')
            if isinstance(pid, str):
                prel = self.get_related_emotions(pid)
                for k in out.keys():
                    for x in prel.get(k, []):
                        if x not in out[k]:
                            out[k].append(x)
        self._related_cache[emotion_id] = out
        return out

    def get_progression_situations(self, emotion_id: str) -> Dict[str, Any]:
        if emotion_id in self._situation_cache:
            return self._situation_cache[emotion_id]
        cp = self.flat_emotions.get(emotion_id, {}).get('context_patterns', {}) or {}
        child = cp.get('situations', {}) or {}
        merged = {}
        for k, v in child.items():
            if isinstance(v, dict):
                merged[k] = dict(v)
        ed = self.flat_emotions.get(emotion_id, {})
        parent = ed.get('parent_data')
        if isinstance(parent, dict):
            pid = parent.get('emotion_id')
            if isinstance(pid, str):
                ps = self.get_progression_situations(pid)
                for pk, pv in ps.items():
                    if pk not in merged:
                        merged[pk] = dict(pv)
        self._situation_cache[emotion_id] = merged
        return merged



# =============================================================================
# Class A: EmotionSequenceAnalyzer
# =============================================================================
class EmotionSequenceAnalyzer:
    def __init__(self, data_manager: "EmotionDataManager" = None, config: Optional[Dict[str, Any]] = None):
        self.dm = data_manager
        self.config = config or {}
        # (선택) 내부 캐시/인덱스 생성의 잠금 보호
        self._lock: RLock = RLock()
        self.max_emotions_per_sentence = int(self.config.get("max_emotions_per_sentence", 3))
        self.max_memory_usage = float(self.config.get("max_memory_usage", 80.0))
        
        # 독립 모듈을 위한 기본 데이터 매니저 생성
        if self.dm is None:
            self.dm = self._create_default_data_manager()

        global_rules = {}
        if isinstance(getattr(self.dm, "emotions_data", None), dict):
            global_rules = self.dm.emotions_data.get("global_rules", {}) or {}

        self.negative_context_map: Dict[str, List[str]] = (
            self.config.get("negative_context_map")
            or global_rules.get("negative_context_map")
            or {"희": [], "노": [], "애": [], "락": []}
        )

        # feature weights (tuned; overridable via config["feature_weights"])
        self.feature_weights = {
            'pattern': 0.22,
            'context': 0.22,
            'intensity': 0.16,
            'transition': 0.12,
            'complexity': 0.10,
            'related': 0.06,
            'ml': 0.08,
            'temporal': 0.04
        }

    def _create_default_data_manager(self):
        """독립 모듈을 위한 기본 데이터 매니저 생성"""
        class DefaultDataManager:
            def __init__(self):
                self.emotions_data = {
                    "희": {"patterns": ["happy", "joy", "행복", "기쁨"]},
                    "노": {"patterns": ["angry", "mad", "분노", "화남"]},
                    "애": {"patterns": ["sad", "depressed", "슬픔", "우울"]},
                    "락": {"patterns": ["pleasure", "satisfaction", "만족", "환희"]},
                }
            
            def get_emotion_ids(self):
                return list(self.emotions_data.keys())
            
            def get_transitions(self, emotion_id):
                return {"patterns": []}
        
        return DefaultDataManager()

    def analyze(self, text: str) -> Dict[str, Any]:
        """독립 모듈을 위한 간단한 시간 시퀀스 분석"""
        try:
            # 문장 분리
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            if not sentences:
                sentences = [text.strip()]
            
            # ★★★ 확장된 감정 패턴 매칭 ★★★
            emotion_patterns = {
                "희": [
                    "happy", "joy", "행복", "기쁨", "기쁘", "즐겁", "즐거", "좋아", "좋았",
                    "신나", "신났", "설레", "기대", "희망", "감사", "고맙", "축하", "반가",
                    "웃음", "미소", "환하", "뿌듯", "보람", "감동"
                ],
                "노": [
                    "angry", "mad", "분노", "화남", "화가", "화났", "화나", "짜증", "불만",
                    "불쾌", "싫어", "싫었", "싫다", "열받", "빡치", "빡쳤", "짜증", "답답",
                    "억울", "분함", "분하", "원망", "증오", "미움", "거부", "거부감"
                ],
                "애": [
                    "sad", "depressed", "슬픔", "슬프", "슬퍼", "우울", "우울하", "울었",
                    "눈물", "힘들", "힘듦", "아프", "아팠", "고통", "고독", "외로", "외롭",
                    "허무", "허탈", "실망", "좌절", "상실", "그리", "그립", "서럽", "서러"
                ],
                "락": [
                    "pleasure", "satisfaction", "만족", "환희", "content", "cheerful",
                    "편안", "평화", "안정", "여유", "느긋", "홀가분", "후련", "시원",
                    "감탄", "경탄", "놀라", "놀랐", "대단", "멋지", "훌륭"
                ],
            }
            
            sequence_results = []
            for i, sentence in enumerate(sentences):
                sentence_lower = sentence.lower()
                detected_emotions = []
                
                for emotion, patterns in emotion_patterns.items():
                    for pattern in patterns:
                        if pattern in sentence_lower:
                            detected_emotions.append({
                                "emotion": emotion,
                                "pattern": pattern,
                                "confidence": 0.8
                            })
                            break
                
                sequence_results.append({
                    "sentence_index": i,
                    "sentence": sentence,
                    "detected_emotions": detected_emotions,
                    "emotion_count": len(detected_emotions)
                })
            
            # 시퀀스 분석
            total_emotions = sum(len(r["detected_emotions"]) for r in sequence_results)
            emotion_transitions = []
            
            for i in range(1, len(sequence_results)):
                prev_emotions = [e["emotion"] for e in sequence_results[i-1]["detected_emotions"]]
                curr_emotions = [e["emotion"] for e in sequence_results[i]["detected_emotions"]]
                
                if prev_emotions != curr_emotions:
                    emotion_transitions.append({
                        "from_sentence": i-1,
                        "to_sentence": i,
                        "transition_type": "emotion_change",
                        "previous_emotions": prev_emotions,
                        "current_emotions": curr_emotions
                    })
            
            return {
                "sequence_analysis": sequence_results,
                "emotion_transitions": emotion_transitions,
                "summary": {
                    "total_sentences": len(sentences),
                    "total_emotions_detected": total_emotions,
                    "transition_count": len(emotion_transitions),
                    "average_emotions_per_sentence": total_emotions / len(sentences) if sentences else 0
                },
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "success": True
            }
            
        except Exception as e:
            return {
                "sequence_analysis": [],
                "emotion_transitions": [],
                "summary": {
                    "total_sentences": 0,
                    "total_emotions_detected": 0,
                    "transition_count": 0,
                    "average_emotions_per_sentence": 0
                },
                "error": str(e),
                "success": False
            }

    def _initialize_advanced_features(self):
        """고급 기능들을 위한 초기화"""
        global_rules = {}
        if isinstance(getattr(self.dm, "emotions_data", None), dict):
            global_rules = self.dm.emotions_data.get("global_rules", {}) or {}

        self.negative_context_map: Dict[str, List[str]] = (
            self.config.get("negative_context_map")
            or global_rules.get("negative_context_map")
            or {"희": [], "노": [], "애": [], "락": []}
        )

        # feature weights (tuned; overridable via config["feature_weights"])
        self.feature_weights = {
            'pattern': 0.22,
            'context': 0.22,
            'intensity': 0.16,
            'transition': 0.12,
            'complexity': 0.10,
            'related': 0.06,
            'ml': 0.08,
            'linguistic': 0.04,
        }
        self.feature_weights.update(self.config.get("feature_weights", {}))

        # paragraph boost parameters
        self.paragraph_top_k = int(self.config.get("paragraph_top_k", 3))
        self.paragraph_boost_alpha = float(self.config.get("paragraph_boost_alpha", 0.10))

        # generic negation regexes (boundary-light for Korean)
        self._negation_res: List[re.Pattern] = [
            re.compile(r"지\s*않", re.I),
            re.compile(r"지\s*못", re.I),
            re.compile(r"않았", re.I),
            re.compile(r"않고", re.I),
            re.compile(r"못하", re.I),
            re.compile(r"별로", re.I),
            re.compile(r"그다지", re.I),
            re.compile(r"전혀", re.I),
            re.compile(r"아니(야|다)", re.I),
            re.compile(r"없(다|어)", re.I),
        ]

        # local regex cache for boundary compiles (paragraph-level & intensity examples)
        self._re_cache: Dict[str, re.Pattern] = {}

        logger.debug(
            f"[SeqAnalyzer] init: max_emotions_per_sentence={self.max_emotions_per_sentence} "
            f"max_memory_usage={self.max_memory_usage} weights={self.feature_weights}"
        )

        # 후보 프리필터 설정 및 인덱스 보관(샤드 정규식)
        self._regex_shard_size = int(self.config.get("seq_regex_shard", 256))
        self._cand_topk = int(self.config.get("seq_cand_topk", 30))
        self._phrase2eid: Dict[str, Set[str]] = {}
        self._rx_phrase_shards: Tuple[re.Pattern, ...] = tuple()

        self._cat_pattern_weights: Dict[str, Dict[str, float]] = {}
        self._cat_intensity_weights: Dict[str, Dict[str, float]] = {}
        self._cat_rx_basic: Dict[str, Tuple[re.Pattern, ...]] = {}
        self._cat_rx_intensity: Dict[str, Tuple[re.Pattern, ...]] = {}
        self._build_category_regex_cache()
        
        # 고급 기능 초기화
        try:
            self._initialize_advanced_features()
        except Exception as e:
            logger.warning(f"고급 기능 초기화 실패: {e}")
            # 기본값으로 설정
            self.paragraph_top_k = int(self.config.get("paragraph_top_k", 3))
            self.paragraph_boost_alpha = float(self.config.get("paragraph_boost_alpha", 0.10))
            self.feature_weights = {
                'pattern': 0.22, 'context': 0.22, 'intensity': 0.16, 'transition': 0.12,
                'complexity': 0.10, 'related': 0.06, 'ml': 0.08, 'linguistic': 0.04
            }
        
        # 초기화 시점에 phrase index 구축
        try:
            self._build_phrase_index()
        except Exception as e:
            logger.warning(f"Phrase index 구축 실패: {e}")
            # 기본값으로 설정
            self._rx_phrase_shards = tuple()
            self._phrase2eid = {}

        # config 기반 부정어 주입(있으면 우선 적용)
        neg_from_cfg = (self.config.get("negations") or [])
        if isinstance(neg_from_cfg, list) and neg_from_cfg:
            try:
                self._negation_res = [re.compile(re.escape(n), re.I) for n in neg_from_cfg if isinstance(n, str) and n.strip()]
            except Exception:
                pass
    def _build_category_regex_cache(self) -> None:
        categories = ("희", "노", "애", "락")
        regex_shard_size = getattr(self, '_regex_shard_size', 256)
        shard_config = int(self.config.get("category_regex_shard", regex_shard_size))
        shard = max(32, min(2048, shard_config))
        self._cat_pattern_weights = {}
        self._cat_intensity_weights = {}
        self._cat_rx_basic = {}
        self._cat_rx_intensity = {}

        def store_target(store: Dict[str, Tuple[float, str]], text: Any, weight: float) -> None:
            if not isinstance(text, str):
                return
            token = text.strip()
            if not token:
                return
            key = token.lower()
            existing = store.get(key)
            if existing is None or weight > existing[0]:
                store[key] = (weight, token)

        for cat in categories:
            pattern_store: Dict[str, Tuple[float, str]] = {}
            intensity_store: Dict[str, Tuple[float, str]] = {}
            for e_data in self.dm.flat_emotions.values():
                if e_data.get("metadata", {}).get("primary_category") != cat:
                    continue
                profile = e_data.get("emotion_profile", {}) or {}
                for kw in profile.get("core_keywords", []) or []:
                    store_target(pattern_store, kw, 1.0)
                transitions = (e_data.get("transitions", {}) or {})
                for pat in transitions.get("patterns", []) or []:
                    if not isinstance(pat, dict):
                        continue
                    token = pat.get("pattern", "")
                    weight = pat.get("weight", 1.0)
                    try:
                        weight_f = float(weight)
                    except (TypeError, ValueError):
                        weight_f = 1.0
                    store_target(pattern_store, token, weight_f)
                intensity_levels = profile.get("intensity_levels", {}) or {}
                for lv_data in intensity_levels.values():
                    if not isinstance(lv_data, dict):
                        continue
                    weight = lv_data.get("weight", 1.0)
                    try:
                        weight_f = float(weight)
                    except (TypeError, ValueError):
                        weight_f = 1.0
                    examples = lv_data.get("intensity_examples", {}) or {}
                    for ex_list in examples.values():
                        if not isinstance(ex_list, list):
                            continue
                        for ex in ex_list:
                            store_target(intensity_store, ex, weight_f)
            self._cat_pattern_weights[cat] = {k: v[0] for k, v in pattern_store.items()}
            self._cat_intensity_weights[cat] = {k: v[0] for k, v in intensity_store.items()}
            self._cat_rx_basic[cat] = self._compile_regex_shards([v[1] for v in pattern_store.values()], shard)
            self._cat_rx_intensity[cat] = self._compile_regex_shards([v[1] for v in intensity_store.values()], shard)

    def _compile_regex_shards(self, words: List[str], shard: int) -> Tuple[re.Pattern, ...]:
        unique: List[str] = []
        seen: Set[str] = set()
        for word in words:
            token = word.strip() if isinstance(word, str) else ""
            if not token or token in seen:
                continue
            seen.add(token)
            unique.append(token)
        if not unique:
            return tuple()
        shard_size = max(1, shard)
        shards: List[re.Pattern] = []
        for i in range(0, len(unique), shard_size):
            block = unique[i:i + shard_size]
            if not block:
                continue
            alt = "|".join(sorted((re.escape(w) for w in block), key=len, reverse=True))
            if not alt:
                continue
            pattern = fr"(?<!\w)(?:{alt})(?!\w)"
            shards.append(re.compile(pattern, re.I))
        return tuple(shards)


    def _build_phrase_index(self):
        from collections import defaultdict
        bank: List[str] = []
        p2e = defaultdict(set)

        # 라벨 전체를 1패스로 수집
        for e_id in self.dm.get_emotion_ids():
            # ① key_phrases
            ling = self.dm.get_linguistic_patterns(e_id) or {}
            for kp in ling.get("key_phrases", []) or []:
                if isinstance(kp, dict):
                    p = (kp.get("pattern") or "").strip()
                    if p:
                        bank.append(p); p2e[p.lower()].add(e_id)
            # ② intensity_examples
            levels = self.dm.get_intensity_levels(e_id) or {}
            for lv in levels.values():
                exd = (lv.get("intensity_examples") or {}) if isinstance(lv, dict) else {}
                for ex_list in (exd or {}).values():
                    for ex in (ex_list or []):
                        ex = (ex or "").strip()
                        if ex:
                            bank.append(ex); p2e[ex.lower()].add(e_id)
            # ③ transition.patterns & multi.triggers
            tr = self.dm.get_transitions(e_id) or {}
            for p in tr.get("patterns", []) or []:
                if isinstance(p, dict):
                    s = (p.get("pattern") or "").strip()
                    if s:
                        bank.append(s); p2e[s.lower()].add(e_id)
            for m in tr.get("multi", []) or []:
                if isinstance(m, dict):
                    for t in (m.get("triggers") or []):
                        t = (t or "").strip()
                        if t:
                            bank.append(t); p2e[t.lower()].add(e_id)

        # 샤딩 컴파일(대소문자 무시)
        words = sorted({w for w in bank if w})
        regex_shard_size = getattr(self, '_regex_shard_size', 256)
        shard = max(32, min(2048, regex_shard_size))
        shards: List[re.Pattern] = []
        for i in range(0, len(words), shard):
            alt = "|".join(sorted(map(re.escape, words[i:i + shard]), key=len, reverse=True))
            shards.append(re.compile(alt, re.I))

        self._phrase2eid = {k: set(v) for k, v in p2e.items()}
        self._rx_phrase_shards = tuple(shards)

    # ------------------------------ Public API ------------------------------
    def split_into_paragraphs(self, text: str) -> List[str]:
        paragraphs: List[str] = []
        raw_paras = re.split(r'\n\s*\n+', text)
        for rp in raw_paras:
            candidate = rp.strip()
            if not candidate:
                continue
            if len(candidate) > 1000:
                chunk_size = 500
                for start in range(0, len(candidate), chunk_size):
                    chunk = candidate[start:start + chunk_size].strip()
                    if chunk:
                        paragraphs.append(chunk)
            else:
                paragraphs.append(candidate)
        return paragraphs

    def _split_sentences(self, text: str) -> List[str]:
        """
        한국어 문장 분할 개선:
        - config['sentence_splitter']가 callable이면 우선 사용
        - kss가 설치되어 있으면 kss.split_sentences 사용
        - 폴백: 한국어/영문 문장부호 + 개행 기반 정규식 분할
        """
        try:
            splitter = self.config.get("sentence_splitter")
            if callable(splitter):
                out = splitter(text)
                if isinstance(out, list) and all(isinstance(s, str) for s in out):
                    return [s.strip() for s in out if s and str(s).strip()]
        except Exception:
            pass
        try:
            import kss  # type: ignore
            return [s.strip() for s in kss.split_sentences(text or "") if s and s.strip()]
        except Exception:
            pass
        # 폴백(간단 정규식)
        return [s.strip() for s in re.split(r'(?<=[.!?…])\s+|[\r\n]+', text or "") if s and s.strip()]

    def _apply_intensity_levels(self, e_id: str, base: float, sentence: str) -> float:
        """
        intensity_levels 예시 문구 매칭으로 단일 감정 강도를 미세 보정.
        - 경계 인식(bregex)으로 과포착 완화
        - 매칭된 레벨 weight에 부분 수렴(과도 증폭 방지)
        - 예시 하나도 안 맞으면 원본(base) 유지
        """
        levels = self.dm.get_intensity_levels(e_id)
        if not isinstance(levels, dict) or not levels:
            return base

        s = sentence
        score = base
        hits = 0

        for lvl_key, lvl_data in levels.items():
            if not isinstance(lvl_data, dict):
                continue
            w = float(lvl_data.get("weight", 1.0))
            exd = lvl_data.get("intensity_examples", {}) or {}
            for ex_list in exd.values():
                if not isinstance(ex_list, list):
                    continue
                for ex in ex_list:
                    if not isinstance(ex, str) or not ex.strip():
                        continue
                    # 경계 기반 정규식으로 안정 매칭
                    creg = self._bregex(ex)
                    if creg.search(s):
                        # weight로 절반만 수렴해 과도 증폭 방지 (예: w=1.3 → +15%)
                        score *= (1.0 + (w - 1.0) * 0.5)
                        hits += 1

        # 매칭 없으면 원본 유지, 있으면 소수점 3자리로 클램프
        return round(score if hits else base, 3)

    def analyze_paragraphs(self, text: str) -> List[Dict[str, Any]]:
        paragraphs = self.split_into_paragraphs(text)
        results: List[Dict[str, Any]] = []
        for idx, para in enumerate(paragraphs):
            para_l = para.lower()
            scores_by_emotion: Dict[str, float] = {}

            # 후보 감정 선별(문단 단위, 전역 1패스)
            if not hasattr(self, '_rx_phrase_shards') or not self._rx_phrase_shards:
                try:
                    self._build_phrase_index()
                except Exception as e:
                    logger.warning(f"Phrase index 구축 실패: {e}")
                    self._rx_phrase_shards = tuple()
                    self._phrase2eid = {}
            cand_eids = set()
            for rx in self._rx_phrase_shards:
                for m in rx.finditer(para):
                    cand_eids |= self._phrase2eid.get(m.group(0).lower(), set())

            # scan transitions.patterns (use compiled boundary regex where available)
            for e_id in (cand_eids or self.dm.get_emotion_ids()):
                s = 0.0
                trans = self.dm.get_transitions(e_id)
                for pat in trans.get("patterns", []):
                    if not isinstance(pat, dict):
                        continue
                    ptn = pat.get("pattern", "")
                    if not ptn:
                        continue
                    creg = pat.get("compiled")
                    hit = False
                    if creg is not None:
                        if creg.search(para):
                            hit = True
                    else:
                        if ptn.lower() in para_l:
                            hit = True
                    if hit:
                        s += float(pat.get("weight", 1.0)) * 0.12

                # scan linguistic key phrases (compiled boundary)
                ling = self.dm.get_linguistic_patterns(e_id) or {}
                for kp in ling.get("key_phrases", []):
                    if not isinstance(kp, dict):
                        continue
                    ptn = kp.get("pattern", "")
                    if not ptn:
                        continue
                    creg = kp.get("compiled")
                    hit = False
                    if creg is not None:
                        if creg.search(para):
                            hit = True
                    else:
                        if ptn.lower() in para_l:
                            hit = True
                    if hit:
                        s += float(kp.get("weight", 1.0)) * 0.10

                # scan intensity example phrases (compile locally with boundary)
                levels = self.dm.get_intensity_levels(e_id) or {}
                for lv_data in levels.values():
                    for ex_list in (lv_data.get("intensity_examples", {}) or {}).values():
                        for ex in ex_list:
                            if not isinstance(ex, str) or not ex.strip():
                                continue
                            creg = self._bregex(ex)
                            if creg.search(para):
                                s += float(lv_data.get("weight", 1.0)) * 0.08

                if s > 0:
                    scores_by_emotion[e_id] = scores_by_emotion.get(e_id, 0.0) + s

            # length proxy
            total_score = min(len(para) / 400.0, 1.0) + sum(scores_by_emotion.values())
            paragraph_top_k = getattr(self, 'paragraph_top_k', 3)
            top_emotions = [k for k, _ in sorted(scores_by_emotion.items(), key=lambda x: x[1], reverse=True)[:paragraph_top_k]]
            results.append({
                "paragraph_index": idx,
                "paragraph_preview": para[:80],
                "matched_emotions": top_emotions,  # backward-compatible name
                "scores_by_emotion": scores_by_emotion,
                "score": round(total_score, 3),
            })
        return results

    def build_emotion_sequence(self, text: str) -> Tuple[List[str], List[Dict[str, Any]], List[Dict[str, Any]]]:
        if not self._check_memory():
            logger.warning("[SeqAnalyzer] Memory cap exceeded. Skipping sequence build.")
            return [], [], []

        paragraph_info = self.analyze_paragraphs(text)
        sentences = self._split_sentences(text)
        emotion_seq: List[Dict[str, Any]] = []

        boundaries = self._compute_paragraph_boundaries(text)
        cursor = 0
        for i, sent in enumerate(sentences):
            matched = self._analyze_sentence(sent)
            end = cursor + len(sent)
            p_idx = self._find_paragraph_index(boundaries, cursor, end)
            if 0 <= p_idx < len(paragraph_info):
                p_info = paragraph_info[p_idx]
                p_matched = set(p_info.get("matched_emotions", []))
                for e_id in list(matched.keys()):
                    if e_id in p_matched:
                        matched[e_id] = round(matched[e_id] + self.paragraph_boost_alpha, 3)
            emotion_seq.append({"sentence_index": i, "sentence": sent, "emotions": matched})
            cursor = end + 1

        # 사후 풍부화: 문장 결과로 paragraph matched_emotions/scores_by_emotion 채우기
        self._enrich_paragraph_from_sentences(text, emotion_seq, paragraph_info)
        return sentences, emotion_seq, paragraph_info

    def _enrich_paragraph_from_sentences(self, text: str, emotion_seq: List[Dict[str, Any]],
                                         paragraph_info: List[Dict[str, Any]]) -> None:
        if not paragraph_info or not emotion_seq:
            return
        bounds = self._compute_paragraph_boundaries(text)
        agg: List[Dict[str, float]] = [dict() for _ in range(len(paragraph_info))]

        cursor = 0
        for entry in emotion_seq:
            s = entry.get("sentence", "")
            if not s:
                continue
            # 원문에서 이 문장을 현재 cursor 이후로 탐색
            start = text.find(s, cursor)
            if start < 0:
                # 못 찾으면 바로 이어붙이기 전략(폴백)
                start = cursor
            end = start + len(s)
            cursor = end  # 다음 탐색 출발점 업데이트

            # paragraph index 결정
            p_idx = self._find_paragraph_index(bounds, start, end)
            if 0 <= p_idx < len(agg):
                for e_id, val in entry.get("emotions", {}).items():
                    agg[p_idx][e_id] = agg[p_idx].get(e_id, 0.0) + float(val)

        for p_idx, acc in enumerate(agg):
            if not acc:
                continue
            paragraph_top_k = getattr(self, 'paragraph_top_k', 3)
            top = sorted(acc.items(), key=lambda x: x[1], reverse=True)[: paragraph_top_k]
            paragraph_info[p_idx]["scores_by_emotion"] = {k: round(v, 3) for k, v in acc.items()}
            if not paragraph_info[p_idx].get("matched_emotions"):
                paragraph_info[p_idx]["matched_emotions"] = [k for k, _ in top]

    def analyze_time_flow(self, sentences: List[str]) -> Dict[str, Any]:
        time_map: Dict[str, Any] = {}
        extracted_events: List[Dict[str, Any]] = []
        base_time_rules = self._collect_temporal_rules()
        for idx, sent in enumerate(sentences):
            ts = self._extract_time_from_sentence(sent, base_time_rules)
            if ts is None:
                time_map[str(idx)] = 0 if idx == 0 else int(time_map[str(idx - 1)]) + 10
            else:
                time_map[str(idx)] = ts
                extracted_events.append({"sentence_index": idx, "sentence": sent, "timestamp": ts})
        mode, reasons, coverage = _choose_timeflow_mode(len(sentences), len(extracted_events))
        return {
            "timestamps": time_map,
            "detected_events": extracted_events,
            "mode": mode,
            "coverage": round(coverage, 3),
            "reason_codes": reasons,
            "note": "temporal_rules 기반 추정(기존 출력과 호환)",
        }

    def refine_sub_emotions(self, emotion_sequence: List[Dict[str, Any]]) -> Dict[str, float]:
        refined: Dict[str, float] = {}
        prev: Dict[str, float] = {}
        for entry in emotion_sequence:
            sent = entry.get("sentence", "")
            for e_id, base in entry.get("emotions", {}).items():
                adj = self._apply_intensity_levels(e_id, base, sent)
                prev_val = prev.get(e_id, base)
                if abs(adj - prev_val) > 0.5:
                    adj *= 0.95 if adj > prev_val else 1.05
                refined[e_id] = min(max(adj, 0.0), 1.0)
                prev[e_id] = refined[e_id]
        mx = max(refined.values()) if refined else 1.0
        if mx > 0:
            refined = {k: round(v / mx, 3) for k, v in refined.items()}
        return refined

    def calculate_emotion_changes(self, emotion_sequence: List[Dict[str, Any]], time_flow: Dict[str, Any]) -> List[Dict[str, Any]]:
        timestamps = time_flow.get("timestamps", {})
        changes: List[Dict[str, Any]] = []
        prev_emotions: Dict[str, float] = {}
        for entry in emotion_sequence:
            idx = entry["sentence_index"]
            cur = entry["emotions"]
            deltas: Dict[str, float] = {}
            ids = set(prev_emotions) | set(cur)
            for e_id in ids:
                diff = round(cur.get(e_id, 0.0) - prev_emotions.get(e_id, 0.0), 3)
                if abs(diff) > 0.01:
                    deltas[e_id] = diff
            if deltas:
                changes.append({"sentence_index": idx, "time": timestamps.get(str(idx), 0), "emotion_deltas": deltas})
            prev_emotions = cur
        return changes

    # --------------------------- Sentence-level core ---------------------------
    def _analyze_sentence(self, sentence: str) -> Dict[str, float]:
        s = self._preprocess_sentence(sentence)
        if not s:
            return {}
        emotions: Dict[str, float] = {}

        # 후보 감정 선별(문장 단위 Top-K)
        if not self._rx_phrase_shards:
            self._build_phrase_index()
        cand_eids = set()
        for rx in self._rx_phrase_shards:
            for m in rx.finditer(s):
                cand_eids |= self._phrase2eid.get(m.group(0).lower(), set())
        if cand_eids:
            from collections import Counter
            cnt = Counter()
            for rx in self._rx_phrase_shards:
                for m in rx.finditer(s):
                    for eid in self._phrase2eid.get(m.group(0).lower(), ()):
                        cnt[eid] += 1
            eid_iter = [eid for eid, _ in cnt.most_common(max(1, self._cand_topk))]
        else:
            eid_iter = self.dm.get_emotion_ids()

        categories = {'희', '노', '애', '락'}
        cat_w = {c: self._category_weight(s, c) for c in categories}

        for e_id in eid_iter:
            meta = self.dm.get_metadata(e_id)
            cat = meta.get("primary_category", "")
            if cat not in categories:
                continue

            pat_score = self._pattern_feature_boundary(s, e_id)
            ctx_score = self._context_feature(s, e_id)
            cpx_w     = self._complexity_weight(self.dm.get_emotion_complexity(e_id))
            lvl_score = self._intensity_feature_boundary(s, e_id)
            trn_score = self._transition_feature_boundary(s, e_id)
            rel_score = self._related_feature(s, e_id)
            ml_score  = self._ml_feature(s, e_id)
            ling_score= self._ling_feature(s, e_id)

            w = self.feature_weights
            score = (
                pat_score * w['pattern'] +
                ctx_score * w['context'] +
                lvl_score * w['intensity'] +
                trn_score * w['transition'] +
                cpx_w     * w['complexity'] +
                rel_score * w['related'] +
                ml_score  * w['ml'] +
                ling_score* w.get('linguistic', 0.04)
            )
            score *= cat_w.get(cat, 1.0)
            if score > 0:
                emotions[e_id] = score

        emotions = self._suppress_conflicts(emotions)
        emotions = self._boost_same_positive_synergy(emotions)  # new boost before top-K/normalize

        if len(emotions) > self.max_emotions_per_sentence:
            emotions = dict(sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:self.max_emotions_per_sentence])

        mx = max(emotions.values()) if emotions else 0
        if mx > 0:
            emotions = {k: round(v / mx, 3) for k, v in emotions.items()}
        return emotions

    # --------------------------- Boundary-aware features ---------------------------
    def _pattern_feature_boundary(self, s: str, e_id: str) -> float:
        lp = self.dm.get_linguistic_patterns(e_id) or {}
        tot = 0.0
        for kp in lp.get("key_phrases", []):
            if not isinstance(kp, dict):
                continue
            ptn = kp.get("pattern", "")
            if not ptn:
                continue
            creg = kp.get("compiled")
            hit = False
            if creg is not None:
                if creg.search(s):
                    hit = True
            else:
                if ptn.lower() in s.lower():
                    hit = True
            if hit:
                w = float(kp.get("weight", 1.0))
                ctx = kp.get("context_requirement", "")
                if ctx and (ctx.lower() not in s.lower()):
                    continue
                tot += w * 0.1
        for combo in lp.get("sentiment_combinations", []):
            if not isinstance(combo, dict): continue
            words = [w.lower() for w in combo.get("words", []) if isinstance(w, str)]
            if words and all(w in s.lower() for w in words):
                tot += float(combo.get("weight", 1.0)) * 0.1
        mods = lp.get("sentiment_modifiers", {}) or {}
        factor = 1.0
        for a in mods.get("amplifiers", []):
            if isinstance(a, str) and a.lower() in s.lower():
                factor *= 1.2
        for d in mods.get("diminishers", []):
            if isinstance(d, str) and d.lower() in s.lower():
                factor *= 0.8
        return min(1.0, tot * factor)

    def _intensity_feature_boundary(self, s: str, e_id: str) -> float:
        prof = self.dm.get_emotion_profile(e_id) or {}
        levels = prof.get("intensity_levels", {})
        if not isinstance(levels, dict) or not levels:
            return 0.0
        tot = 0.0; matches = 0
        for lvl, lvdata in levels.items():
            if not isinstance(lvdata, dict): continue
            w = float(lvdata.get("weight", 1.0))
            exd = lvdata.get("intensity_examples", {}) or {}
            for ex_list in exd.values():
                for ex in ex_list:
                    if not isinstance(ex, str) or not ex.strip():
                        continue
                    creg = self._bregex(ex)
                    if creg.search(s):
                        tot += w * 0.3
                        matches += 1
        return min(1.0, tot if matches else 0.0)

    def _transition_feature_boundary(self, s: str, e_id: str) -> float:
        tr = self.dm.get_transitions(e_id) or {}
        score = 0.0
        for p in tr.get("patterns", []):
            if not isinstance(p, dict): continue
            ptn = p.get("pattern", "")
            if not ptn:
                continue
            creg = p.get("compiled")
            hit = False
            if creg is not None:
                if creg.search(s):
                    hit = True
            else:
                if ptn.lower() in s.lower():
                    hit = True
            if hit:
                score += float(p.get("weight", 1.0)) * 0.2
        for m in tr.get("multi", []):
            if not isinstance(m, dict): continue
            strength = float(m.get("strength", 1.0))
            for trg in (m.get("triggers", []) or []):
                if isinstance(trg, str):
                    creg = self._bregex(trg)
                    if creg.search(s):
                        score += strength * 0.1
        return min(1.0, score)

    # ------------------------------ Other features ------------------------------
    def _context_feature(self, s: str, e_id: str) -> float:
        score = 0.0
        sits = self.dm.get_progression_situations(e_id)
        s_l = s.lower()
        for sd in sits.values():
            kws = sd.get("keywords", [])
            if kws:
                hits = sum(1 for k in kws if isinstance(k, str) and k.lower() in s_l)
                if hits:
                    inten = sd.get("intensity", "medium")
                    w = 1.2 if inten == "high" else 0.8 if inten == "low" else 1.0
                    score += (hits / len(kws)) * w
        return min(1.0, score)

    def _related_feature(self, s: str, e_id: str) -> float:
        rel = self.dm.get_related_emotions(e_id) or {}
        s_l = s.lower()
        pos = sum(1 for x in rel.get("positive", []) if isinstance(x, str) and x.lower() in s_l)
        neg = sum(1 for x in rel.get("negative", []) if isinstance(x, str) and x.lower() in s_l)
        score = max(0.0, (pos * 0.1) - (neg * 0.05))
        return min(1.0, score)

    def _ml_feature(self, s: str, e_id: str) -> float:
        ml = self.dm.flat_emotions.get(e_id, {}).get("ml_training_metadata", {})
        if not isinstance(ml, dict): return 0.0
        s_l = s.lower()
        req = ml.get("context_requirements", {})
        if isinstance(req, dict):
            min_len = int(req.get("minimum_length", 0))
            if len(s.split()) < min_len:
                return 0.0
            rk = req.get("required_keywords", {})
            need = int(rk.get("basic", 0)) if isinstance(rk, dict) else 0
            if need > 0:
                have = 0
                for kw in ml.get("basic_keywords", []):
                    if isinstance(kw, str) and kw.lower() in s_l:
                        have += 1
                if have < need:
                    return 0.0
        sc = 0.0
        th = ml.get("confidence_thresholds", {})
        if isinstance(th, dict) and th.get("basic", 0.7) <= 0.7: sc += 0.1
        pm = ml.get("pattern_matching", {})
        if isinstance(pm, dict) and pm.get("basic", 0.65) < 0.7: sc += 0.05
        mods = ml.get("analysis_modules", {})
        if isinstance(mods, dict):
            sc += sum(0.02 for v in mods.values() if isinstance(v, dict) and v.get("enabled"))
        return min(1.0, sc)

    def _ling_feature(self, s: str, e_id: str) -> float:
        return self._pattern_feature_boundary(s, e_id)

    # ------------------------------ Helpers ------------------------------
    def _category_weight(self, s: str, category: str) -> float:
        weight = 1.0
        s_l = s.lower()

        if not hasattr(self, '_cat_rx_basic') or not self._cat_rx_basic:
            self._build_category_regex_cache()

        pattern_weights = self._cat_pattern_weights.get(category, {})
        intensity_weights = self._cat_intensity_weights.get(category, {})

        matched_patterns: Set[str] = set()
        for rx in self._cat_rx_basic.get(category, ()):
            for match in rx.finditer(s):
                key = match.group(0).lower()
                if key in pattern_weights:
                    matched_patterns.add(key)
        matches = sum(pattern_weights.get(key, 0.0) for key in matched_patterns)

        matched_intensity: Set[str] = set()
        for rx in self._cat_rx_intensity.get(category, ()):
            for match in rx.finditer(s):
                key = match.group(0).lower()
                if key in intensity_weights:
                    matched_intensity.add(key)
        intens_hits = [intensity_weights[key] for key in matched_intensity]

        if matches:
            weight *= min(2.0, 1.0 + 0.1 * matches)
        if intens_hits:
            weight *= min(2.0, (sum(intens_hits) / len(intens_hits)))

        # 문장 맥락 prior 적용
        prior, pos_hit, neg_hit = self._sentence_context_prior(s)
        weight *= (1.0 + prior.get(category, 0.0))

        # 긍정 prior일 때 부정 카테고리 완만 감쇠, 반대로도 적용
        if pos_hit and category in {"희", "락"}:
            weight *= 0.92
        if neg_hit and category in {"노", "애"}:
            weight *= 0.92

        # 기존 부정 키워드 맵 감쇠
        for neg_kw in self.negative_context_map.get(category, []):
            if isinstance(neg_kw, str) and neg_kw.lower() in s_l:
                weight *= 0.95

        return min(3.0, max(0.1, weight))

    def _generic_neg_penalty(self, s: str) -> float:
        hits = 0
        for rg in self._negation_res:
            if rg.search(s):
                hits += 1
        if hits == 0:
            return 1.0
        # each hit reduces by ~6%, capped at 25% reduction
        return max(0.75, 1.0 - 0.06 * hits)

    def _boost_same_positive_synergy(self, emotions: Dict[str, float]) -> Dict[str, float]:
        if not emotions:
            return emotions
        items = list(emotions.items())
        boost_pairs: List[Tuple[str, str]] = []
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                e1, v1 = items[i]
                e2, v2 = items[j]
                cat1 = self.dm.get_metadata(e1).get("primary_category", "")
                cat2 = self.dm.get_metadata(e2).get("primary_category", "")
                if cat1 == cat2 and cat1 in {"희", "락"} and min(v1, v2) >= 0.6:
                    boost_pairs.append((e1, e2))
        if not boost_pairs:
            return emotions
        boosted = dict(emotions)
        for e1, e2 in boost_pairs:
            boosted[e1] *= 1.05
            boosted[e2] *= 1.05
        return boosted

    def _suppress_conflicts(self, emotions: Dict[str, float]) -> Dict[str, float]:
        adjusted = emotions.copy()
        if not adjusted:
            return adjusted

        min_th = float(self.config.get("conflict_min_intensity", 0.35))
        weaken = float(self.config.get("conflict_weaken_factor", 0.85))

        ids = list(adjusted.keys())
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                e1, e2 = ids[i], ids[j]
                v1, v2 = adjusted[e1], adjusted[e2]
                if min(v1, v2) < min_th:
                    continue
                cat1 = self.dm.get_metadata(e1).get("primary_category", "")
                cat2 = self.dm.get_metadata(e2).get("primary_category", "")
                if cat1 == cat2 and cat1 in {"희", "락"}:
                    continue
                if e2 in self.dm.get_conflict_with_list(e1):
                    if v1 > v2:
                        adjusted[e2] *= weaken
                    else:
                        adjusted[e1] *= weaken
        return adjusted

    def _sentence_context_prior(self, s: str) -> Tuple[Dict[str, float], bool, bool]:
        pos_terms = set(self.config.get("context_bias_positive_terms", ["칭찬", "축하", "기뻤", "기쁜", "행복", "만족", "성취"]))
        neg_terms = set(self.config.get("context_bias_negative_terms", ["피곤", "아프", "아픈", "힘들", "불안", "걱정", "우울"]))
        prior = {"희": 0.0, "락": 0.0, "노": 0.0, "애": 0.0}
        pos_hit, neg_hit = False, False
        for t in pos_terms:
            if self._bregex(t).search(s):
                prior["희"] += 0.15
                prior["락"] += 0.10
                pos_hit = True
        for t in neg_terms:
            if self._bregex(t).search(s):
                prior["애"] += 0.15
                prior["노"] += 0.10
                neg_hit = True
        return prior, pos_hit, neg_hit

    # ------------------------------ Time helpers ------------------------------
    def _collect_temporal_rules(self) -> List[Dict[str, Any]]:
        rules: List[Dict[str, Any]] = []
        for e_id in self.dm.get_emotion_ids():
            tr = self.dm.get_transitions(e_id)
            for r in tr.get("rules", []):
                if isinstance(r, dict):
                    rules.append(r)
        return rules

    def _extract_time_from_sentence(self, sentence: str, base_rules: List[Dict[str, Any]]) -> Optional[int]:
        m = re.search(r'(\d{4}-\d{1,2}-\d{1,2})', sentence)
        if m:
            return self._yyyy_mm_dd_to_int(m.group(1))
        s_l = sentence.lower()
        for r in base_rules:
            cond = str(r.get("condition", "")).lower()
            if cond and cond in s_l:
                return int((r.get("time_window", {}) or {}).get("offset_seconds", 0))
        return None

    @staticmethod
    def _yyyy_mm_dd_to_int(date_str: str) -> int:
        try:
            y, m, d = [int(x) for x in date_str.split("-")]
            return y * 10000 + m * 100 + d
        except Exception:
            return 0

    # ------------------------------ Paragraph map ------------------------------
    def _compute_paragraph_boundaries(self, text: str) -> List[Tuple[int, int]]:
        paragraphs = self.split_into_paragraphs(text)
        bounds: List[Tuple[int, int]] = []
        idx = 0
        for p in paragraphs:
            start = text.find(p, idx)
            start = idx if start < 0 else start
            end = start + len(p)
            bounds.append((start, end))
            idx = end
        return bounds

    @staticmethod
    def _find_paragraph_index(boundaries: List[Tuple[int, int]], s_start: int, s_end: int) -> int:
        for i, (b0, b1) in enumerate(boundaries):
            if b0 <= s_start < b1:
                return i
        return -1

    # ------------------------------ Text utils ------------------------------
    @staticmethod
    def _preprocess_sentence(s: str) -> str:
        if not isinstance(s, str): return ""
        s = re.sub(r'\s+', ' ', s).strip()
        s = re.sub(r'[^\w\s.]', ' ', s)
        return s

    @staticmethod
    def _complexity_weight(complexity: str) -> float:
        return {'complex': 1.2, 'subtle': 1.1, 'basic': 1.0}.get(complexity, 1.0)

    def _check_memory(self) -> bool:
        if psutil is None: return True
        return psutil.virtual_memory().percent < self.max_memory_usage

    # ------------------------------ Local boundary regex ------------------------------
    def _bregex(self, text: str) -> re.Pattern:
        key = f"b::{text}"
        if not hasattr(self, '_re_cache'):
            self._re_cache = {}
        cp = self._re_cache.get(key)
        if cp is not None:
            return cp
        txt = text.strip()
        if not txt:
            pat = r"$^"  # never match
        else:
            pat = r"(?<!\w)" + re.escape(txt) + r"(?!\w)"
        try:
            cp = re.compile(pat, re.IGNORECASE)
        except re.error:
            cp = re.compile(re.escape(text), re.IGNORECASE)
        self._re_cache[key] = cp
        return cp



# =============================================================================
# Class B: CausalityTransitionAnalyzer
# =============================================================================
class CausalityTransitionAnalyzer:
    def __init__(self, data_manager: "EmotionDataManager", config: Optional[Dict[str, Any]] = None):
        self.dm = data_manager
        self.config = config or {}
        self.min_causality_strength = float(self.config.get("min_causality_strength", 0.3))
        self.min_transition_score   = float(self.config.get("min_transition_score", 0.4))
        self.conflict_threshold     = float(self.config.get("conflict_threshold", 0.8))
        # Connective dictionary: merge defaults + user config + domain(dep) connectives
        default_conn = {
            "cause":    ["때문에", "따라서", "그래서", "그러므로", "그 결과", "그로 인해", "하여", "왜냐하면"],
            "contrast": ["하지만", "그러나", "반면에", "그럼에도", "다만"],
            "sequence": ["그리고", "이후에", "그러자", "먼저", "다음으로", "마침내", "결국", "즉"],
        }
        user_conn = (self.config.get("connectives") or {})
        dep_cfg = self._collect_dependency_rules()
        dep_conns = dep_cfg.get("dep_connectives", []) if dep_cfg else []

        merged: Dict[str, set] = {k: set(v) for k, v in default_conn.items()}
        if isinstance(user_conn, dict):
            for k, arr in user_conn.items():
                if isinstance(arr, list):
                    merged.setdefault(k, set()).update(x for x in arr if isinstance(x, str) and x.strip())
        for k in list(merged.keys()):
            merged[k].update(x for x in dep_conns if isinstance(x, str) and x.strip())

        self._conn_re_by_type = {
            k: re.compile("|".join(sorted(map(re.escape, list(v)))), re.I)
            for k, v in merged.items() if v
        }
        all_terms = sorted({t for s in merged.values() for t in s})
        self._connective_re = re.compile("|".join(map(re.escape, all_terms)), re.I) if all_terms else re.compile(r"$^")
        self._conn_weights = self.config.get("connective_scores", {"cause": 0.45, "contrast": 0.40, "sequence": 0.35})

    def analyze_cause_effect(
            self,
            sentences: List[str],
            emotion_sequence: List[Dict[str, Any]],
            context_window: int = 1
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        alpha = float(self.config.get("causality_weight_alpha", 0.6))
        beta = float(self.config.get("transition_weight_beta", 0.4))
        N = len(sentences)

        def clip01(x: float) -> float:
            try:
                return max(0.0, min(1.0, float(x)))
            except Exception:
                return 0.0

        trans_cache = self._build_transition_pattern_cache()
        rule_cache = self._build_emotion_rules_cache()

        for i in range(N):
            for j in range(i + 1, min(i + 1 + context_window, N)):
                s_from, s_to = sentences[i], sentences[j]
                e_from = emotion_sequence[i]["emotions"]
                e_to = emotion_sequence[j]["emotions"]

                base_caus = self._sentence_causality(s_from, e_from, e_to, trans_cache)
                tr_score = self._detailed_transition_score(e_from, e_to, trans_cache, rule_cache)

                extra = self._additional_markers(s_from, s_to)
                base_caus["strength"] += extra["score"]

                b_bonus, b_conns = self._boundary_connective_bonus(s_from, s_to)
                base_caus["strength"] += b_bonus
                for conn in b_conns:
                    base_caus["transition_patterns"].append({
                        "from_emotion": "NLP_inferred",
                        "to_emotions": list(e_to.keys()),
                        "pattern": conn,
                        "type": "nlp_cause",
                        "weight": b_bonus
                    })

                gamma, delta_ev = self._delta_based_gamma(e_from, e_to)
                base_caus["strength"] += gamma

                ct_bonus, ct_patterns = self._contrast_turn_bridge(s_from, s_to, delta_ev)
                if ct_bonus > 0:
                    base_caus["strength"] += ct_bonus
                    base_caus["transition_patterns"].extend(ct_patterns)

                nlp_cands = []
                if (self.config.get("enable_advanced_nlp") or
                        (self.config.get("nlp_parser_settings", {}) or {}).get("enable_dependency_parse")):
                    nlp_cands = self._nlp_causality_candidates(f"{s_from} {s_to}")
                merged = self._merge_nlp_causality(base_caus, nlp_cands)

                merged["strength"] = clip01(merged.get("strength", 0.0))
                tr_score = clip01(tr_score)

                inten = self._intensity_factor(e_from, e_to)
                ctx = self._context_factor(s_from, s_to, e_from, e_to)
                compx = self._complex_factor(emotion_sequence, i, j)
                rel, harmony, h_factors = self._relationship_factor_with_harmony(e_from, e_to)

                base = clip01(alpha * merged["strength"] + beta * tr_score)
                combined = clip01(base * inten * ctx * compx * rel)

                cause_markers = self._count_cause_markers(extra["markers"])
                contrast_as_cause = 1 if (ct_bonus > 0) else 0
                marker_count = cause_markers + len(b_conns) + contrast_as_cause
                thr = max(self.min_causality_strength, self.min_transition_score)
                if marker_count >= 3:
                    thr *= 0.5
                elif marker_count == 2:
                    thr *= 0.65
                elif marker_count == 1:
                    thr *= 0.8
                threshold = thr

                if combined >= threshold:
                    merged["transition_patterns"] = self._dedupe_patterns(merged["transition_patterns"])
                    merged["cause_patterns"] = self._dedupe_patterns(merged.get("cause_patterns", []))
                    merged["effect_patterns"] = self._dedupe_patterns(merged.get("effect_patterns", []))

                    de_pos = clip01(delta_ev.get("pos_gain", 0.0))
                    de_neg = clip01(delta_ev.get("neg_loss", 0.0))
                    evidence = extra["markers"] + b_conns + (["contrast_turn"] if ct_bonus > 0 else [])
                    evidence = list(dict.fromkeys(evidence))

                    R = {
                        "from_sentence_index": i,
                        "to_sentence_index": j,
                        "from_sentence": s_from,
                        "to_sentence": s_to,
                        "cause_patterns": merged["cause_patterns"],
                        "effect_patterns": merged["effect_patterns"],
                        "transition_patterns": merged["transition_patterns"],
                        "causality_strength": round(merged["strength"], 3),
                        "transition_score": round(tr_score, 3),
                        "combined_score": round(combined, 3),
                        "extra_causal_evidence": evidence,
                        "from_emotions": e_from,
                        "to_emotions": e_to,
                        "delta_evidence": {"pos_gain": de_pos, "neg_loss": de_neg},
                        "overall_harmony": round(clip01(harmony), 3),
                        "harmony_factors": h_factors
                    }
                    R["confidence_score"] = round(self._confidence_score(R), 3)
                    results.append(R)

        return self._post_process_results(results)

    def _contrast_turn_bridge(self, s_from: str, s_to: str, delta_ev: Dict[str, float]) -> Tuple[
        float, List[Dict[str, Any]]]:
        cfg = self.config.get("contrast_turn_cfg", {
            "both_base": 0.22,  # pos_gain & neg_loss 동시 존재 기본 보너스
            "both_scale": 0.12,  # (pos+neg)/2 스케일 팩터
            "both_max": 0.40,  # 동시 존재 상한
            "single_base": 0.14,  # 단일 존재 기본 보너스
            "single_scale": 0.10,  # max(pos,neg) 스케일 팩터
            "single_max": 0.30  # 단일 존재 상한
        })
        txt = (s_from + " " + s_to).lower()
        markers = self.dm.gather_causal_and_contrast_markers()
        contrast = markers.get("contrast", set()) or set()
        has_contrast = any(m.lower() in txt for m in contrast) or bool(re.search(r"(하지만|그러나|반면|반대로)", txt))
        if not has_contrast:
            return 0.0, []
        pos_gain = float(delta_ev.get("pos_gain", 0.0))
        neg_loss = float(delta_ev.get("neg_loss", 0.0))
        if pos_gain <= 0 and neg_loss <= 0:
            return 0.0, []
        if pos_gain > 0 and neg_loss > 0:
            bonus = min(cfg["both_max"], cfg["both_base"] + cfg["both_scale"] * min(1.0, (pos_gain + neg_loss) / 2.0))
        else:
            bonus = min(cfg["single_max"], cfg["single_base"] + cfg["single_scale"] * min(1.0, max(pos_gain, neg_loss)))
        patterns = [{
            "from_emotion": "NLP_inferred",
            "to_emotions": [],
            "pattern": "contrast_turn",
            "type": "nlp_cause",
            "weight": round(bonus, 3)
        }]
        return round(bonus, 3), patterns

    def handle_complex_emotions(self, emotion_sequence: List[Dict[str, Any]]) -> List[str]:
        out: List[str] = []
        for entry in emotion_sequence:
            emos = entry.get("emotions", {})
            if len(emos) < 2: continue
            ids = list(emos.keys())
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    e1, e2 = ids[i], ids[j]
                    v1, v2 = emos[e1], emos[e2]
                    rel = self._pair_relation(e1, e2, v1, v2)
                    if rel:
                        out.append(f"{e1} & {e2} -> {rel}")
        return out

    def _additional_markers(self, s_from: str, s_to: str) -> Dict[str, Any]:
        markers = []
        score = 0.0
        sets = self.dm.gather_causal_and_contrast_markers()
        cause = sets.get("cause", set()) or set()
        contrast = sets.get("contrast", set()) or set()
        l_from = s_from.lower(); l_to = s_to.lower()
        for ck in cause:
            ck_l = str(ck).lower()
            if ck_l and (ck_l in l_from or ck_l in l_to):
                markers.append(ck); score += 0.15
        for ck in contrast:
            ck_l = str(ck).lower()
            if ck_l and (ck_l in l_from or ck_l in l_to):
                markers.append(ck); score += 0.05
        return {"markers": markers, "score": round(score, 3)}

    def _sentence_causality(self, sentence: str, e_from: Dict[str, float], e_to: Dict[str, float],
                            trans_cache: Dict[str, Any]) -> Dict[str, Any]:
        info = {"cause_patterns": [], "effect_patterns": [], "transition_patterns": [], "strength": 0.0}
        s_l = sentence.lower()
        sum_w = 0.0; cnt = 0
        for e_id, tr_full in (trans_cache or {}).items():
            # 캐시된 전이 객체에서 간단 패턴 뽑기
            tr = {"cause": [], "effect": [], "transition": []}
            for p in (tr_full.get("patterns", []) or []):
                if not isinstance(p, dict):
                    continue
                pat_str = str(p.get("pattern", "")).strip()
                if not pat_str:
                    continue
                t = str(p.get("type", "")).lower()
                item = {"pattern": pat_str, "weight": float(p.get("weight", 1.0))}
                if t == "cause":
                    tr["cause"].append(item)
                elif t == "effect":
                    tr["effect"].append(item)
            for m in (tr_full.get("multi", []) or []):
                if not isinstance(m, dict):
                    continue
                trig = (m.get("triggers", []) or [None])[0] or ""
                tr["transition"].append({
                    "pattern": trig,
                    "from": m.get("from_emotions", []) or [],
                    "to":   m.get("to_emotions", []) or [],
                    "weight": float(m.get("strength", 1.0))
                })

            # 버그 픽스: 존재하지 않는 감정 강도를 1.0으로 올리지 않음(0.0 유지)
            cur_int = float(e_from.get(e_id, 0.0))

            for cp in tr["cause"]:
                p = cp["pattern"].lower()
                if p and p in s_l:
                    w = float(cp["weight"]) * cur_int
                    info["cause_patterns"].append({"emotion": e_id, "pattern": cp["pattern"], "weight": w})
                    sum_w += w; cnt += 1
            for ep in tr["effect"]:
                p = ep["pattern"].lower()
                if p and p in s_l:
                    w = float(ep["weight"]) * cur_int
                    info["effect_patterns"].append({"emotion": e_id, "pattern": ep["pattern"], "weight": w})
                    sum_w += w; cnt += 1
            for tp in tr["transition"]:
                p = str(tp.get("pattern", "")).lower()
                if p and p in s_l:
                    from_ok = e_id in (tp.get("from") or [])
                    to_ok = any(x in (tp.get("to") or []) for x in e_to.keys())
                    if from_ok and to_ok:
                        w = float(tp.get("weight", 1.0)) * cur_int
                        info["transition_patterns"].append({
                            "from_emotion": e_id,
                            "to_emotions": [x for x in e_to.keys() if x in (tp.get("to") or [])],
                            "pattern": tp.get("pattern", ""),
                            "weight": w
                        })
                        sum_w += w; cnt += 1
        if cnt:
            info["strength"] = round(sum_w / cnt, 3)
        return info

    def _detailed_transition_score(
        self,
        e_from: Dict[str, float],
        e_to: Dict[str, float],
        trans_cache: Dict[str, Any],
        rule_cache: Dict[str, List[Dict[str, Any]]]
    ) -> float:
        score = 0.0; total = 0
        for a, va in e_from.items():
            for b, vb in e_to.items():
                if a in trans_cache:
                    score += self._eval_transition_patterns(trans_cache[a], a, b, va, vb); total += 1
                if a in rule_cache:
                    score += self._eval_emotion_rules(rule_cache[a], a, b, va, vb); total += 1
        return (score / total) if total else 0.0

    def _dedupe_patterns(self, arr: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not isinstance(arr, list):
            return arr
        bucket: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        for it in arr:
            if not isinstance(it, dict):
                continue
            pat = str(it.get("pattern", ""))
            typ = str(it.get("type", ""))
            frm = str(it.get("from_emotion", ""))
            key = (typ, pat, frm)
            w = float(it.get("weight", 0.0))
            if key not in bucket or w > float(bucket[key].get("weight", 0.0)):
                bucket[key] = it
        return list(bucket.values())

    def _eval_transition_patterns(self, curr: Dict[str, Any], a: str, b: str, va: float, vb: float) -> float:
        sc = 0.0
        for p in curr.get("patterns", []):
            if not isinstance(p, dict): continue
            conds = p.get("conditions", {}) or {}
            if self._transition_conditions_ok(conds, a, b, va, vb):
                if va >= 0.3 and vb >= 0.3:
                    sc += float(p.get("weight", 1.0)) * float(p.get("probability", 0.5))
        for m in curr.get("multi", []):
            if not isinstance(m, dict): continue
            if a in (m.get("from_emotions", []) or []) and b in (m.get("to_emotions", []) or []):
                if va >= 0.3 and vb >= 0.3 and self._transition_conditions_ok(m.get("conditions", {}) or {}, a, b, va, vb):
                    sc += 0.9 * float(m.get("strength", 1.0))
        for r in curr.get("rules", []):
            if not isinstance(r, dict): continue
            cond = str(r.get("condition", "")).lower()
            if "fast" in cond and abs(vb - va) >= 0.3:
                sc += float(r.get("weight", 1.0)) * float(r.get("transition_probability", 0.5))
        for d in curr.get("dependencies", []):
            if not isinstance(d, dict): continue
            if d.get("dependent_emotion") == b and self._transition_conditions_ok(d.get("conditions", {}) or {}, a, b, va, vb):
                rel = d.get("relationship_type", "")
                if rel == "enhances":
                    sc += 0.25 * ((va + vb) / 2)
                elif rel == "conflicts":
                    sc -= 0.25 * ((va + vb) / 2)
        cat = curr.get("category_transitions", {})
        if isinstance(cat, dict):
            probs = cat.get("target_category_probabilities", {})
            to_cat = self.dm.get_metadata(b).get("primary_category", "")
            if to_cat and to_cat in probs:
                sc += float(probs[to_cat]) * ((va + vb) / 4)
        return min(1.0, max(0.0, sc))

    def _eval_emotion_rules(self, rules: List[Dict[str, Any]], a: str, b: str, va: float, vb: float) -> float:
        tot = 0.0
        for r in rules:
            if not isinstance(r, dict): continue
            need = r.get("required_category_match", [])
            if need:
                ac = self.dm.get_metadata(a).get("primary_category", "")
                bc = self.dm.get_metadata(b).get("primary_category", "")
                if ac not in need and bc not in need:
                    continue
            dt = r.get("delta_threshold")
            if dt is not None and abs(vb - va) < float(dt): continue
            min_s = r.get("min_strength")
            if min_s is not None and (va < min_s and vb < min_s): continue
            max_s = r.get("max_strength")
            if max_s is not None and (va > max_s or vb > max_s): continue
            req_cpx = r.get("required_complexity")
            if req_cpx:
                if self.dm.get_emotion_complexity(a) != req_cpx or self.dm.get_emotion_complexity(b) != req_cpx:
                    continue
            tot += ((va + vb) / 2) * float(r.get("multiplier", 1.0))
        return min(1.0, tot)

    @staticmethod
    def _transition_conditions_ok(conds: Dict[str, Any], a: str, b: str, va: float, vb: float) -> bool:
        mi = conds.get("min_intensity")
        if mi is not None and (va < mi or vb < mi): return False
        rd = conds.get("required_delta")
        if rd is not None and abs(vb - va) < rd: return False
        return True

    def _nlp_causality_candidates(self, combined_text: str) -> List[Dict[str, Any]]:
        cands: List[Dict[str, Any]] = []
        text_l = combined_text.lower()
        for e_id in self.dm.get_emotion_ids():
            ml = self.dm.flat_emotions.get(e_id, {}).get("ml_training_metadata", {})
            for rule in (ml.get("nlp_causality_rules", []) or []):
                if not isinstance(rule, dict): continue
                pat = str(rule.get("pattern", "")).lower()
                if pat and pat in text_l:
                    sc = float(rule.get("score", 0.3))
                    cands.append({"cause_event": rule.get("cause_event", "CAUSE"), "effect_event": rule.get("effect_event", "EFFECT"), "nlp_score": sc})
        if self._connective_re.search(combined_text):
            cands.append({"cause_event": "CONNECTIVE", "effect_event": "CONNECTIVE", "nlp_score": 0.4})
        dep_cfg = self._collect_dependency_rules()
        if dep_cfg:
            conns = dep_cfg.get("dep_connectives", [])
            if any(isinstance(c, str) and c.lower() in text_l for c in conns):
                cands.append({"cause_event": "DEP_SUBJ", "effect_event": "DEP_VERB", "nlp_score": float(dep_cfg.get("connective_weight", 0.4))})
        return cands

    def _collect_dependency_rules(self) -> Dict[str, Any]:
        rules: Dict[str, Any] = {}
        for e_id in self.dm.get_emotion_ids():
            ml = self.dm.flat_emotions.get(e_id, {}).get("ml_training_metadata", {})
            dep = ml.get("dependency_rules", {})
            if isinstance(dep, dict):
                for k, v in dep.items():
                    if k not in rules:
                        rules[k] = v
                    else:
                        if isinstance(v, list) and isinstance(rules[k], list):
                            rules[k] = list(set(rules[k]) | set(v))
                        elif isinstance(v, (int, float)) and isinstance(rules[k], (int, float)):
                            rules[k] = max(float(v), float(rules[k]))
        return rules

    @staticmethod
    def _merge_nlp_causality(base: Dict[str, Any], cands: List[Dict[str, Any]]) -> Dict[str, Any]:
        merged = copy.deepcopy(base)
        if not cands:
            return merged
        extra = 0.0
        for it in cands:
            sc = float(it.get("nlp_score", 0.0))
            if sc <= 0: continue
            merged["cause_patterns"].append(
                {"emotion": "NLP_inferred", "pattern": it.get("cause_event", ""), "weight": round(sc, 2)})
            merged["effect_patterns"].append(
                {"emotion": "NLP_inferred", "pattern": it.get("effect_event", ""), "weight": round(sc, 2)})
            extra += sc
        merged["strength"] = max(0.0, min(1.0, round(merged.get("strength", 0.0) + extra, 3)))
        return merged

    def _intensity_factor(self, e_from: Dict[str, float], e_to: Dict[str, float]) -> float:
        base = 1.0
        changes = []
        for eid in set(e_from) | set(e_to):
            delta = e_to.get(eid, 0.0) - e_from.get(eid, 0.0)
            if abs(delta) >= 0.1:
                md = self.dm.flat_emotions.get(eid, {}).get("metadata", {})
                changes.append((eid, delta, md.get("primary_category", ""), md.get("emotion_complexity", "basic")))
        if not changes:
            return base
        score = 0.0
        for _, delta, _, cpx in changes:
            w_cpx = {'complex': 1.2, 'subtle': 1.1}.get(cpx, 1.0)
            score += abs(delta) * w_cpx
        adj = (score / max(1, len(changes))) * 0.3
        return max(0.1, min(2.0, base + adj))

    def _context_factor(self, s_from: str, s_to: str, e_from: Dict[str, float], e_to: Dict[str, float]) -> float:
        base = 1.0
        adj = 0.0
        hits = 0.0; cnt = 0.0
        for eid, _ in e_to.items():
            tr = self.dm.get_transitions(eid)
            for p in tr.get("patterns", []):
                if not isinstance(p, dict): continue
                pat = p.get("pattern", "")
                if pat:
                    if pat.lower() in s_from.lower() or pat.lower() in s_to.lower():
                        hits += 0.1 * float(p.get("weight", 1.0)); cnt += 0.1
        if cnt > 0:
            adj += min(0.3, hits / cnt)
        return max(0.1, min(2.0, base + adj))

    def _complex_factor(self, emotion_sequence: List[Dict[str, Any]], i: int, j: int) -> float:
        base = 1.0
        window = emotion_sequence[max(0, i-2):min(len(emotion_sequence), j+3)]
        if len(window) < 3:
            return base
        sizes = [sum(1 for v in ent.get("emotions", {}).values() if v >= 0.3) for ent in window]
        if all(sizes[k] >= sizes[k+1] for k in range(len(sizes)-1)):
            return min(2.0, base + 0.3)
        if all(sizes[k] <= sizes[k+1] for k in range(len(sizes)-1)):
            return min(2.0, base + 0.3)
        return base

    def _relationship_factor_with_harmony(self, e_from: Dict[str, float], e_to: Dict[str, float]) -> Tuple[float, float, Dict[str, float]]:
        base = 1.0
        conf, syn = 0.0, 0.0
        for a, va in e_from.items():
            for b, vb in e_to.items():
                if b in self.dm.get_conflict_with_list(a):
                    conf += (va + vb) / 2
                if b in self.dm.get_synergy_with_list(a):
                    syn  += (va + vb) / 2
                pa = self.dm.get_polarity(a); pb = self.dm.get_polarity(b)
                if pa != pb and {'positive','negative'} <= {pa, pb}:
                    conf += 0.1
        adj = syn * 0.2 - conf * 0.3
        factor = max(0.1, min(2.0, base + adj))
        harmony = max(0.0, min(2.0, 1.0 + syn*0.4 - conf*0.3))
        return factor, harmony, {"synergy_sum": syn, "conflict_sum": conf}

    def _confidence_score(self, result: Dict[str, Any]) -> float:
        base = 0.7
        factors: List[float] = []

        cs = result.get("causality_strength", 0.0)
        factors.append(1.2 if cs >= 0.8 else 1.0 if cs >= 0.5 else 0.8)

        pe = (len(result.get("cause_patterns", []))
              + len(result.get("effect_patterns", []))
              + len(result.get("transition_patterns", [])))
        if pe >= 4:
            factors.append(1.15)
        elif pe >= 2:
            factors.append(1.05)

        # overall_harmony 가중 (1.0 기준, 0~2 범위)
        harmony = float(result.get("overall_harmony", 1.0))
        if harmony >= 1.1:
            factors.append(1.05)
        elif harmony <= 0.9:
            factors.append(0.95)

        # 원인 마커 개수(강한 텍스트 증거)
        markers = result.get("extra_causal_evidence", [])
        if isinstance(markers, list):
            mc = len(markers)
            if mc >= 3:
                factors.append(1.08)
            elif mc >= 1:
                factors.append(1.03)

        # Δ 증거(부정 하락 + 긍정 상승 동시)
        de = result.get("delta_evidence", {}) or {}
        if (de.get("pos_gain", 0.0) > 0) and (de.get("neg_loss", 0.0) > 0):
            factors.append(1.07)

        e_from, e_to = result.get("from_emotions", {}), result.get("to_emotions", {})
        penalty = 0.0
        for a, va in e_from.items():
            if va < 0.8: continue
            cset = self.dm.get_conflict_with_list(a)
            for b, vb in e_to.items():
                if vb < 0.8: continue
                if b in cset:
                    penalty += float(self.config.get("conflict_strong_penalty", 0.05))

        conf = base * (sum(factors) / len(factors) if factors else 1.0)
        conf = max(0.0, min(1.0, conf - penalty))
        return conf

    def _pair_relation(self, e1: str, e2: str, v1: float, v2: float) -> Optional[str]:
        if e2 in self.dm.get_conflict_with_list(e1) or e1 in self.dm.get_conflict_with_list(e2):
            if ((v1 + v2) / 2) * 0.7 > self.conflict_threshold:
                return "conflict"
        if e2 in self.dm.get_synergy_with_list(e1) or e1 in self.dm.get_synergy_with_list(e2):
            if ((v1 + v2) / 2) * 0.7 > (self.config.get("synergy_threshold", 0.7)):
                return "synergy"
        cat1 = self.dm.get_metadata(e1).get("primary_category", "")
        cat2 = self.dm.get_metadata(e2).get("primary_category", "")
        if cat1 == cat2 and cat1 in {"희", "락"} and min(v1, v2) >= 0.6:
            return "synergy"
        p1 = self.dm.get_polarity(e1); p2 = self.dm.get_polarity(e2)
        if p1 != p2 and {'positive','negative'} <= {p1, p2} and min(v1, v2) >= 0.6:
            return "conflict"
        return None

    def _extract_transition_patterns(self, e_id: str) -> Dict[str, List[Dict[str, Any]]]:
        tr = self.dm.get_transitions(e_id)
        out = {"cause": [], "effect": [], "transition": []}
        for p in tr.get("patterns", []):
            if not isinstance(p, dict): continue
            t = p.get("type", "")
            if t == "cause":
                out["cause"].append({"pattern": p.get("pattern", ""), "weight": float(p.get("weight", 1.0))})
            elif t == "effect":
                out["effect"].append({"pattern": p.get("pattern", ""), "weight": float(p.get("weight", 1.0))})
        for m in tr.get("multi", []):
            if not isinstance(m, dict): continue
            out["transition"].append({
                "pattern": (m.get("triggers", []) or [None])[0] or "",
                "from": m.get("from_emotions", []) or [],
                "to":   m.get("to_emotions", []) or [],
                "weight": float(m.get("strength", 1.0))
            })
        return out

    def _build_transition_pattern_cache(self) -> Dict[str, Any]:
        cache: Dict[str, Any] = {}
        for e_id in self.dm.get_emotion_ids():
            cache[e_id] = self.dm.get_transitions(e_id)
        return cache

    def _build_emotion_rules_cache(self) -> Dict[str, List[Dict[str, Any]]]:
        rules: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for e_id in self.dm.get_emotion_ids():
            ml = self.dm.flat_emotions.get(e_id, {}).get("ml_training_metadata", {})
            if isinstance(ml, dict) and isinstance(ml.get("emotion_rules"), list):
                rules[e_id].extend(ml["emotion_rules"])
        return rules

    @staticmethod
    def _post_process_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set(); out: List[Dict[str, Any]] = []
        for r in sorted(results, key=lambda x: x.get("combined_score", 0.0), reverse=True):
            key = (r["from_sentence_index"], r["to_sentence_index"])
            if key in seen: continue
            seen.add(key); out.append(r)
        return out

    def _boundary_connective_bonus(self, s_from: str, s_to: str) -> Tuple[float, List[str]]:
        conns: List[str] = []
        text = (s_from or "") + " " + (s_to or "")
        type_hits: Dict[str, List[str]] = {}
        for t, creg in getattr(self, "_conn_re_by_type", {}).items():
            try:
                hits = [m.group(0) for m in creg.finditer(text)]
            except Exception:
                hits = []
            if hits:
                type_hits[t] = hits
                conns.extend(hits)

        strong = 0
        for c in conns:
            c_l = c.lower()
            if (s_to or "").strip().lower().startswith(c_l) or (s_from or "").strip().lower().endswith(c_l):
                strong += 1
        pos_bonus = (0.3 if strong >= 1 else 0.2)

        type_bonus = 0.0
        cw = getattr(self, "_conn_weights", {"cause": 0.45, "contrast": 0.40, "sequence": 0.35})
        for t, hits in type_hits.items():
            try:
                w = float(cw.get(t, 0.35))
            except Exception:
                w = 0.35
            type_bonus += w * min(1, len(hits)) * 0.5

        bonus = min(0.5, pos_bonus + type_bonus) if conns else 0.0
        return bonus, sorted(set(conns))

    def _delta_based_gamma(self, e_from: Dict[str, float], e_to: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
        pos_cats = {"희", "락"}
        neg_cats = {"노", "애"}

        def sum_cat(emotions: Dict[str, float], cats: Set[str]) -> float:
            s = 0.0
            for eid, v in emotions.items():
                if self.dm.get_metadata(eid).get("primary_category", "") in cats:
                    s += float(v)
            return s

        raw_pos_gain = sum_cat(e_to, pos_cats) - sum_cat(e_from, pos_cats)
        raw_neg_loss = sum_cat(e_from, neg_cats) - sum_cat(e_to, neg_cats)
        pos_gain = max(0.0, min(1.0, raw_pos_gain))
        neg_loss = max(0.0, min(1.0, raw_neg_loss))
        gamma = 0.0
        if pos_gain > 0 and neg_loss > 0:
            gamma = 0.2 * min(1.0, (pos_gain + neg_loss) / 2.0)
        elif pos_gain > 0:
            gamma = 0.12 * pos_gain
        elif neg_loss > 0:
            gamma = 0.12 * neg_loss
        return round(gamma, 3), {"pos_gain": round(pos_gain, 3), "neg_loss": round(neg_loss, 3)}

    def _count_cause_markers(self, markers: List[str]) -> int:
        if not markers:
            return 0
        sets = self.dm.gather_causal_and_contrast_markers()
        cause = sets.get("cause", set()) or set()
        return sum(1 for m in markers if m in cause)

# =============================================================================
# Independent Functions (public API)
# =============================================================================
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime

try:
    logger  # type: ignore
except NameError:  # pragma: no cover
    import logging
    logger = logging.getLogger(__name__)


def _clip01(x: float) -> float:
    try:
        return max(0.0, min(1.0, float(x)))
    except Exception:
        return 0.0


def build_time_series_components(
    emotions_data: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None,
) -> Tuple["EmotionDataManager", "EmotionSequenceAnalyzer", "CausalityTransitionAnalyzer"]:
    """
    EMOTIONS.json(dict), config(dict)로부터 DataManager/Analyzers를 구성해서 반환.
    외부에서 구성/재사용하고 싶을 때 사용.
    """
    if not isinstance(emotions_data, dict) or not emotions_data:
        raise ValueError("Invalid emotions_data")
    dm = EmotionDataManager(emotions_data)
    cfg = dict(config or {})  # 방어적 복사
    seq = EmotionSequenceAnalyzer(dm, cfg)
    caus = CausalityTransitionAnalyzer(dm, cfg)
    return dm, seq, caus


def build_emotion_sequence(
    text: str,
    dm: "EmotionDataManager",
    seq: "EmotionSequenceAnalyzer",
) -> Tuple[List[str], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    문장을 분할하고, 문장별 감정 시퀀스를 생성(문단 분석/보정 포함)
    """
    if not text or not text.strip():
        raise ValueError("Input text is empty")
    sentences, emotion_seq, paragraph_info = seq.build_emotion_sequence(text)
    return sentences, emotion_seq, paragraph_info


def analyze_causality_only(
    sentences: List[str],
    emotion_sequence: List[Dict[str, Any]],
    caus: "CausalityTransitionAnalyzer",
) -> List[Dict[str, Any]]:
    """
    이미 계산된 문장/감정 시퀀스를 받아 인과 관계만 단독 분석.
    """
    return caus.analyze_cause_effect(sentences, emotion_sequence)


def _make_summary(
    emotion_seq: List[Dict[str, Any]],
    paragraph_info: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    안전 요약 생성:
    - 구조 보장: 필드 누락/타입 불일치에도 항상 동일한 키 반환
    - 핵심 지표: 총 문장수, 감정 총개수, 유니크 감정수, 상위 k 감정, 단락 분석 원본
    - 밀도(density): 문장당 감정 수
    """
    from collections import Counter

    if not isinstance(emotion_seq, list):
        emotion_seq = []
    if not isinstance(paragraph_info, list):
        paragraph_info = []

    # 감정 카운트
    counts = Counter()
    total_emotions = 0
    for entry in emotion_seq:
        emos = entry.get("emotions") or {}
        if isinstance(emos, dict):
            total_emotions += len(emos)
            counts.update(list(emos.keys()))

    sentence_count = len(emotion_seq)
    unique_emotions = len(counts)
    density = round(total_emotions / max(1, sentence_count), 3)
    top_emotions = counts.most_common(5)

    return {
        "paragraph_analysis": paragraph_info,
        "sentence_count": sentence_count,
        "emotion_count": total_emotions,
        "unique_emotions": unique_emotions,
        "density_per_sentence": density,
        "most_frequent_emotions": top_emotions,
    }


# =============================================================================
# Timeseries Drop-in Reinforcement (no external deps)
# =============================================================================
# 하단 run_time_series_analysis()에서 가드/휴리스틱/스키마 보장을 위해 사용
MIN_SEQ_LEN = 2
LOW_DATA_CONF_CAP = 0.60
CONNECTIVE_WINDOW = 1           # 신호 전/후 1문장 윈도우
MAX_SENT_LEN_FOR_STATIC = 6     # 토큰 매우 짧으면 static 유도
REPEAT_CHAR_MAX = 3             # 과도 반복 축약 상한

CONNECTIVES = {
    "cause":  ["때문에", "따라서", "그래서", "그러므로", "그 결과", "결과적으로", "이에 따라"],
    "contrast":["하지만", "그러나", "반면에", "그럼에도", "다만"],
    "sequence":["그리고", "이후에", "그러자", "먼저", "다음으로", "마침내", "결국", "즉"],
}
CONNECTIVE_RE = {k: re.compile("|".join(map(re.escape, v))) for k, v in CONNECTIVES.items()}

def _ts_normalize_text(s: str) -> str:
    if not s:
        return ""
    try:
        s = unicodedata.normalize("NFKC", s)
    except Exception:
        pass
    # 제어문자 제거
    s = re.sub(r"[\u0000-\u001F\u007F]", " ", s)
    # 과도 반복 문자 축약 (ㅋㅋㅋㅋ, !!!! 등)
    try:
        s = re.sub(r"(.)\\1{" + str(REPEAT_CHAR_MAX) + r",}", lambda m: m.group(1) * REPEAT_CHAR_MAX, s)
    except Exception:
        pass
    # 공백 정리
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _ts_is_junk_text(s: str) -> bool:
    if not s:
        return True
    total = max(len(s), 1)
    letters = len(re.findall(r"[A-Za-z가-힣]", s))
    digits  = len(re.findall(r"\d", s))
    symbols = total - (letters + digits)
    if total == 0:
        return True
    # 글자 거의 없음/기호 과다/의미토큰 희소 -> 잡음
    if letters/total < 0.20 and symbols/total > 0.50:
        return True
    return False

def _ts_segment_sentences(s: str) -> List[str]:
    # 외부 분기 없이 안전한 폴백 세그멘터
    if not s:
        return []
    parts = re.split(r"[\.\!?\n]+", s)
    out: List[str] = []
    for p in parts:
        p = p.strip(" \"'“”‘’()[]{}")
        if p:
            out.append(p)
    return out

def _ts_low_data_flag(sentences: List[str]) -> bool:
    if len(sentences) < MIN_SEQ_LEN:
        return True
    short = sum(1 for x in sentences if len((x or "").split()) <= MAX_SENT_LEN_FOR_STATIC)
    return (short / max(len(sentences), 1)) >= 0.5

def _ts_make_structured_output(
    *,
    cause_effect: Optional[List[Dict[str, Any]]] = None,
    time_flow: Optional[str] = None,
    emotion_changes: Optional[List[Dict[str, Any]]] = None,
    transitions: Optional[List[Dict[str, Any]]] = None,
    confidence: Optional[float] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "cause_effect": list(cause_effect or []),
        "time_flow": str(time_flow or "unknown"),
        "emotion_changes": list(emotion_changes or []),
        "transitions": list(transitions or []),
        "confidence": float(confidence) if isinstance(confidence, (int, float)) else 0.0,
        "meta": meta if isinstance(meta, dict) else {"warnings": [], "fallbacks_used": [], "coverage": 0.0},
    }

def _ts_static_timeline_stub(sentences: List[str], reason: str) -> Dict[str, Any]:
    meta = {
        "warnings": [f"static_timeline:{reason}"],
        "fallbacks_used": ["static_timeline"],
        "coverage": 0.0,
    }
    return _ts_make_structured_output(
        cause_effect=[],
        time_flow="static",
        emotion_changes=[],
        transitions=[],
        confidence=0.2,
        meta=meta,
    )

def _ts_extract_connective_edges(sentences: List[str]) -> Tuple[List[Dict[str, Any]], List[str], float]:
    edges: List[Dict[str, Any]] = []
    fallbacks: List[str] = []
    hits = 0
    for i, sent in enumerate(sentences):
        for label, creg in CONNECTIVE_RE.items():
            try:
                if creg.search(sent):
                    hits += 1
                    src_idx = max(0, i - CONNECTIVE_WINDOW)
                    dst_idx = min(len(sentences) - 1, i + CONNECTIVE_WINDOW)
                    if src_idx == dst_idx:
                        continue
                    edges.append({
                        "type": ("cause" if label == "cause" else ("contrast" if label == "contrast" else "sequence")),
                        "from": src_idx,
                        "to": dst_idx,
                        "connector": label,
                        "evidence": sentences[i],
                        "confidence": (0.45 if label == "cause" else (0.40 if label == "contrast" else 0.35)),
                        "heuristic": "connective",
                    })
            except Exception:
                continue
    coverage = hits / float(max(1, len(sentences)))
    if hits > 0:
        fallbacks.append("connective_heuristic")
    return edges, fallbacks, float(round(coverage, 3))

def _ts_cap_confidence(base_conf: float, is_low_data: bool) -> float:
    return min(base_conf, LOW_DATA_CONF_CAP) if is_low_data else base_conf

# === [PATCH A] add below existing helpers in Timeseries Drop-in section ===
def _ts_dynamic_conf_cap(coverage: float, sent_count: int, edge_count: int, low_data: bool) -> float:
    """
    연결사 기반 히트 커버리지·문장수·엣지 개수에 따라 신뢰도 상한을 동적으로 산출.
    - 저데이터(half 이상이 초단문 등)면 기본적으로 낮게 캡.
    """
    # 기본선
    cap = 0.60
    # 매우 낮은 증거
    if sent_count <= 2 or coverage < 0.08 or edge_count == 0:
        cap = 0.55
    # 보통 증거
    if coverage >= 0.15 and edge_count >= 1 and sent_count >= 3:
        cap = 0.70
    # 충분 증거
    if coverage >= 0.25 and edge_count >= 2 and sent_count >= 4:
        cap = 0.80
    # 매우 충분
    if coverage >= 0.35 and edge_count >= 3 and sent_count >= 5:
        cap = 0.85
    # 저데이터 플래그가 켜져 있으면 상한을 보수적으로
    if low_data:
        cap = min(cap, 0.60)
    return float(cap)


def _ts_blend_external_conf(base_conf: float,
                            sentences: list,
                            edges: list,
                            external_signals: dict | None) -> tuple[float, dict]:
    """
    외부 신호(transitions_count, coherence[, reliability])를 r(신뢰도)로 가중해
    자동 튜닝한 신뢰도(base_conf)를 반환. 디버그 메타도 함께 리턴.
    """
    if not isinstance(external_signals, dict) or not external_signals:
        return float(base_conf), {"used": False}

    # 입력 값 안전 파싱
    try:
        trn = int(external_signals.get("transitions_count", 0) or 0)
    except Exception:
        trn = 0
    try:
        coh = float(external_signals.get("coherence", 0.0) or 0.0)
    except Exception:
        coh = 0.0
    coh = max(0.0, min(1.0, coh))

    # 신뢰도 r: 명시가 있으면 우선, 없으면 내부 증거로 근사
    if isinstance(external_signals.get("reliability", None), (int, float)):
        r = max(0.0, min(1.0, float(external_signals["reliability"])))
    else:
        e_density = (len(edges) / max(1, len(sentences)))  # 문장당 엣지
        r = max(0.0, min(1.0, 0.5 * coh + 0.5 * e_density))

    # 자동 가중: r이 높을수록 외부신호 영향 확장
    base_offset = 0.30 + 0.10 * r                  # 0.30 ~ 0.40
    w_trn       = 0.05 + 0.10 * r                  # 0.05 ~ 0.15
    w_coh       = 0.15 + 0.20 * r                  # 0.15 ~ 0.35

    tuned = base_offset + w_trn * (min(trn, 3) / 3.0) + w_coh * coh
    tuned = min(0.90, max(base_conf, tuned))       # 상한 0.90, 기존보다 낮아지지 않게

    dbg = {"used": True, "r": r, "trn": trn, "coh": coh,
           "base_offset": base_offset, "w_trn": w_trn, "w_coh": w_coh, "tuned": tuned}
    return float(tuned), dbg

def analyze_time_series_reinforced(raw_text: str, external_signals: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    text = _ts_normalize_text(raw_text)
    if _ts_is_junk_text(text):
        return _ts_static_timeline_stub([], "junk_text")
    sents = _ts_segment_sentences(text)
    if len(sents) < MIN_SEQ_LEN:
        return _ts_static_timeline_stub(sents, "min_sequence")

    edges, fallbacks, coverage = _ts_extract_connective_edges(sents)
    # [PATCH B] 외부신호 자동 튜닝 + 동적 캡 적용
    base_conf = 0.50
    base_conf, ext_dbg = _ts_blend_external_conf(base_conf, sents, edges, external_signals)
    is_low = _ts_low_data_flag(sents) or (coverage < 0.10 and len(sents) <= 3)
    dyn_cap = _ts_dynamic_conf_cap(coverage, len(sents), len(edges), low_data=is_low)
    conf = min(base_conf, dyn_cap)
    time_flow = "linear" if edges else "undetermined"
    meta = {
        "warnings": (["low_data_caps_applied"] if is_low else []),
        "fallbacks_used": fallbacks,
        "coverage": coverage,
        "sentences": len(sents),
        "low_data": bool(is_low),
        # 디버그 가시성(옵션)
        "conf_debug": {
            "base_conf": round(base_conf, 3),
            "dynamic_cap": round(dyn_cap, 3),
            "final_confidence": round(conf, 3),
            "ext_tuning": ext_dbg,
        },
    }
    return _ts_make_structured_output(
        cause_effect=edges,
        time_flow=("linear_capped" if is_low else time_flow),
        emotion_changes=[],
        transitions=[],
        confidence=conf,
        meta=meta,
    )

def _ts_reinforced_to_core_result(guard_out: Dict[str, Any]) -> Dict[str, Any]:
    """Re-map guard output -> core schema (run_time_series_analysis result)."""
    edges = []
    for e in guard_out.get("cause_effect", []) or []:
        try:
            edges.append({
                "from_sentence_index": int(e.get("from", 0)),
                "to_sentence_index": int(e.get("to", 0)),
                "cause_patterns": ([{"pattern": str(e.get("evidence", "")), "weight": 0.5}] if str(e.get("type", "")) == "cause" else []),
                "effect_patterns": [],
                "transition_patterns": [],
                "combined_score": float(e.get("confidence", 0.35)),
                "from_emotions": {},
                "to_emotions": {},
                "extra_causal_evidence": [str(e.get("connector", "")) or str(e.get("heuristic", ""))],
                "overall_harmony": 1.0,
                "harmony_factors": {},
            })
        except Exception:
            continue
    tf_note = str(guard_out.get("time_flow", "unknown"))
    meta = guard_out.get("meta", {}) if isinstance(guard_out.get("meta"), dict) else {}
    try:
        coverage = float(meta.get("coverage", 0.0))
    except Exception:
        coverage = 0.0
    reason_codes: List[str] = []
    try:
        warns = meta.get("warnings", []) or []
        if tf_note == TimeFlowMode.STATIC.value:
            if any("min_sequence" in str(w) for w in warns):
                reason_codes.append("min_sequence")
            if any("junk_text" in str(w) for w in warns):
                reason_codes.append("junk_text")
        elif tf_note == TimeFlowMode.LINEAR_CAPPED.value:
            if coverage < 0.10:
                reason_codes.append("low_coverage")
            if bool(meta.get("low_data")):
                reason_codes.append("low_data")
    except Exception:
        reason_codes = []
    result = {
        "emotion_sequence": [],
        "cause_effect": edges,
        "time_flow": {
            "timestamps": {},
            "detected_events": [],
            "mode": tf_note if tf_note in {m.value for m in TimeFlowMode} else TimeFlowMode.UNDETERMINED.value,
            "coverage": round(coverage, 3),
            "reason_codes": reason_codes,
            "note": tf_note,
        },
        "refined_sub_emotions": {},
        "emotion_changes": list(guard_out.get("emotion_changes", []) or []),
        "summary": {"paragraph_analysis": [], "emotion_count": 0, "most_frequent_emotions": []},
        "analysis_timestamp": datetime.now().isoformat(),
        "meta": dict(meta),
        "diagnostics": {
            "time_flow_mode": tf_note if tf_note in {m.value for m in TimeFlowMode} else TimeFlowMode.UNDETERMINED.value,
            "time_flow_coverage": round(coverage, 3),
            "reason_codes": reason_codes,
        },
    }
    return result


def run_emotion_analysis(
    text: str,
    emotions_data: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    가장 단순한 고수준 API: 전체 분석 파이프라인을 실행하고, AnalysisResult를 dict로 반환.
    """
    # Pre-guard: run reinforcement to guarantee schema if core is insufficient or fails
    guard_out = analyze_time_series_reinforced(text, (config or {}).get("external_signals"))
    try:
        dm, seq, caus = build_time_series_components(emotions_data, config)
        sentences, emotion_seq, paragraph_info = build_emotion_sequence(text, dm, seq)
        cause_effect = caus.analyze_cause_effect(sentences, emotion_seq)
        time_flow = seq.analyze_time_flow(sentences)
        complex_list = caus.handle_complex_emotions(emotion_seq)
        refined_sub = seq.refine_sub_emotions(emotion_seq)
        emotion_changes = seq.calculate_emotion_changes(emotion_seq, time_flow)
        summary = _make_summary(emotion_seq, paragraph_info)
        result = {
            "emotion_sequence": emotion_seq,
            "summary": summary,
            "cause_effect": cause_effect,
            "time_flow": time_flow,
            "complex_emotions": complex_list,
            "refined_sub_emotions": refined_sub,
            "emotion_changes": emotion_changes,
            "analysis_timestamp": datetime.now().isoformat(),
        }
        # Diagnostics exposure for ops/tuning visibility
        try:
            result["diagnostics"] = {
                "time_flow_mode": time_flow.get("mode", "undetermined"),
                "time_flow_coverage": time_flow.get("coverage", 0.0),
                "reason_codes": time_flow.get("reason_codes", []),
            }
        except Exception:
            result["diagnostics"] = {
                "time_flow_mode": "undetermined",
                "time_flow_coverage": 0.0,
                "reason_codes": [],
            }
        # Sufficiency check: minimal signal (sequence length or cause edges or non-unknown time_flow)
        try:
            seq_len_ok = isinstance(result.get("emotion_sequence"), list) and len(result["emotion_sequence"]) >= MIN_SEQ_LEN
            ce_ok = isinstance(result.get("cause_effect"), list) and len(result["cause_effect"]) > 0
            tf_ok = bool(result.get("time_flow", {}).get("note")) if isinstance(result.get("time_flow"), dict) else False
            sufficient = bool(seq_len_ok or ce_ok or tf_ok)
        except Exception:
            sufficient = False
        if not sufficient:
            go = _ts_reinforced_to_core_result(guard_out)
            m = go.setdefault("meta", {})
            if isinstance(m, dict):
                m.setdefault("warnings", []).append("core_insufficient_use_guard")
                m.setdefault("fallbacks_used", []).append("core_insufficient_use_guard")
            return go
        # Merge guard meta for observability
        meta = dict(guard_out.get("meta", {})) if isinstance(guard_out.get("meta"), dict) else {}
        result["meta"] = meta
        return result
    except Exception as e:
        logger.exception("[run_emotion_analysis] 오류")
        # On failure, return guard output mapped to core schema
        go = _ts_reinforced_to_core_result(guard_out)
        m = go.setdefault("meta", {})
        if isinstance(m, dict):
            m.setdefault("warnings", []).append(f"core_exception:{type(e).__name__}")
            m.setdefault("fallbacks_used", []).append("core_failed_use_guard")
        return go


def run_time_series_analysis(
    text: str,
    emotions_data: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    타임 시리즈 관점의 핵심 산출물을 반환(문장 감정, 인과, 시간 흐름, Δ, 요약).
    """
    logger.info("[run_time_series_analysis] 시작")
    # Pre-guard for schema/observability
    guard_out = analyze_time_series_reinforced(text, (config or {}).get("external_signals"))
    try:
        dm, seq, caus = build_time_series_components(emotions_data, config)
        sentences, emotion_seq, paragraph_info = build_emotion_sequence(text, dm, seq)

        cause_effect = caus.analyze_cause_effect(sentences, emotion_seq)
        time_flow = seq.analyze_time_flow(sentences)
        refined_sub = seq.refine_sub_emotions(emotion_seq)
        emotion_changes = seq.calculate_emotion_changes(emotion_seq, time_flow)
        summary = _make_summary(emotion_seq, paragraph_info)

        result = {
            "emotion_sequence": emotion_seq,
            "cause_effect": cause_effect,
            "time_flow": time_flow,
            "refined_sub_emotions": refined_sub,
            "emotion_changes": emotion_changes,
            "summary": summary,
            "analysis_timestamp": datetime.now().isoformat(),
        }
        # Diagnostics exposure
        try:
            result["diagnostics"] = {
                "time_flow_mode": time_flow.get("mode", "undetermined"),
                "time_flow_coverage": time_flow.get("coverage", 0.0),
                "reason_codes": time_flow.get("reason_codes", []),
            }
        except Exception:
            result["diagnostics"] = {
                "time_flow_mode": "undetermined",
                "time_flow_coverage": 0.0,
                "reason_codes": [],
            }
        # Sufficiency check
        try:
            seq_len_ok = isinstance(result.get("emotion_sequence"), list) and len(result["emotion_sequence"]) >= MIN_SEQ_LEN
            ce_ok = isinstance(result.get("cause_effect"), list) and len(result["cause_effect"]) > 0
            tf_ok = bool(result.get("time_flow", {}).get("note")) if isinstance(result.get("time_flow"), dict) else False
            sufficient = bool(seq_len_ok or ce_ok or tf_ok)
        except Exception:
            sufficient = False
        if not sufficient:
            go = _ts_reinforced_to_core_result(guard_out)
            m = go.setdefault("meta", {})
            if isinstance(m, dict):
                m.setdefault("warnings", []).append("core_insufficient_use_guard")
                m.setdefault("fallbacks_used", []).append("core_insufficient_use_guard")
            logger.info("[run_time_series_analysis] guard used due to insufficient core output")
            return go
        # Merge observability meta
        meta = dict(guard_out.get("meta", {})) if isinstance(guard_out.get("meta"), dict) else {}
        result["meta"] = meta
        logger.info("[run_time_series_analysis] 완료")
        return result
    except Exception as e:
        logger.exception("[run_time_series_analysis] 분석 오류")
        go = _ts_reinforced_to_core_result(guard_out)
        m = go.setdefault("meta", {})
        if isinstance(m, dict):
            m.setdefault("warnings", []).append(f"core_exception:{type(e).__name__}")
            m.setdefault("fallbacks_used", []).append("core_failed_use_guard")
        return go


def run_causality_analysis(
    text: str,
    emotions_data: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    인과 분석만 필요한 경우: 내부적으로 문장/감정 시퀀스를 만들고 인과만 계산.
    """
    logger.info("[run_causality_analysis] 시작")
    try:
        dm, seq, caus = build_time_series_components(emotions_data, config)
        sentences, emotion_seq, _ = build_emotion_sequence(text, dm, seq)
        cause_effect = caus.analyze_cause_effect(sentences, emotion_seq)
        result = {
            "cause_effect": cause_effect,
            "details": {
                "sentence_count": len(sentences),
                "emotion_sequence_preview": emotion_seq[:3]
            },
            "analysis_timestamp": datetime.now().isoformat(),
        }
        logger.info("[run_causality_analysis] 완료")
        return result
    except Exception as e:
        logger.exception("[run_causality_analysis] 오류")
        return {"error": str(e), "analysis_timestamp": datetime.now().isoformat()}


def run_full_time_series_analysis(
    text: str,
    emotions_data: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    전체 파이프라인 실행 + complex_emotions 포함(완전판).
    """
    logger.info("[run_full_time_series_analysis] 시작")
    guard_out = analyze_time_series_reinforced(text, (config or {}).get("external_signals"))
    try:
        dm, seq, caus = build_time_series_components(emotions_data, config)

        sentences, emotion_seq, paragraph_info = build_emotion_sequence(text, dm, seq)
        cause_effect = caus.analyze_cause_effect(sentences, emotion_seq)
        time_flow = seq.analyze_time_flow(sentences)
        complex_list = caus.handle_complex_emotions(emotion_seq)
        refined_sub = seq.refine_sub_emotions(emotion_seq)
        emotion_changes = seq.calculate_emotion_changes(emotion_seq, time_flow)
        summary = _make_summary(emotion_seq, paragraph_info)

        result = {
            "emotion_sequence": emotion_seq,
            "cause_effect": cause_effect,
            "time_flow": time_flow,
            "complex_emotions": complex_list,
            "refined_sub_emotions": refined_sub,
            "emotion_changes": emotion_changes,
            "summary": summary,
            "analysis_timestamp": datetime.now().isoformat(),
        }
        # Diagnostics exposure
        try:
            result["diagnostics"] = {
                "time_flow_mode": time_flow.get("mode", "undetermined"),
                "time_flow_coverage": time_flow.get("coverage", 0.0),
                "reason_codes": time_flow.get("reason_codes", []),
            }
        except Exception:
            result["diagnostics"] = {
                "time_flow_mode": "undetermined",
                "time_flow_coverage": 0.0,
                "reason_codes": [],
            }
        # Sufficiency check same as above
        try:
            seq_len_ok = isinstance(result.get("emotion_sequence"), list) and len(result["emotion_sequence"]) >= MIN_SEQ_LEN
            ce_ok = isinstance(result.get("cause_effect"), list) and len(result["cause_effect"]) > 0
            tf_ok = bool(result.get("time_flow", {}).get("note")) if isinstance(result.get("time_flow"), dict) else False
            sufficient = bool(seq_len_ok or ce_ok or tf_ok)
        except Exception:
            sufficient = False
        if not sufficient:
            go = _ts_reinforced_to_core_result(guard_out)
            m = go.setdefault("meta", {})
            if isinstance(m, dict):
                m.setdefault("warnings", []).append("core_insufficient_use_guard")
                m.setdefault("fallbacks_used", []).append("core_insufficient_use_guard")
            logger.info("[run_full_time_series_analysis] guard used due to insufficient core output")
            return go
        meta = dict(guard_out.get("meta", {})) if isinstance(guard_out.get("meta"), dict) else {}
        result["meta"] = meta
        logger.info("[run_full_time_series_analysis] 완료")
        return result
    except Exception as e:
        logger.exception("[run_full_time_series_analysis] 오류")
        go = _ts_reinforced_to_core_result(guard_out)
        m = go.setdefault("meta", {})
        if isinstance(m, dict):
            m.setdefault("warnings", []).append(f"core_exception:{type(e).__name__}")
            m.setdefault("fallbacks_used", []).append("core_failed_use_guard")
        return go


# =============================================================================
# Main (Refactored for your project layout)
# =============================================================================
try:
    import psutil
except Exception:
    psutil = None

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR.parent
# 통합 로그 관리자 사용 (날짜별 폴더)
try:
    from log_manager import get_log_manager
    log_manager = get_log_manager()
    LOGS_DIR = log_manager.get_log_dir("emotion_analysis", use_date_folder=True)
except ImportError:
    # 폴백: 기존 방식 (날짜별 폴더 추가)
    base_logs_dir = BASE_DIR / "logs"
    today = datetime.now().strftime("%Y%m%d")
    LOGS_DIR = base_logs_dir / today
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

def _setup_logger() -> logging.Logger:
    logger = logging.getLogger("time_series_analyzer")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    # 기본: 비간섭 (NullHandler만)
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    # Opt-in: 파일 로깅
    if os.environ.get("TSA_FILE_LOG", "0").lower() in ("1", "true", "yes"):
        try:
            from logging.handlers import RotatingFileHandler
            LOGS_DIR.mkdir(parents=True, exist_ok=True)
            fh = RotatingFileHandler(str(LOGS_DIR / "time_series_analyzer.log"),
                                     maxBytes=10*1024*1024, backupCount=5, encoding="utf-8")
            fh.setLevel(logging.INFO)
            fh.setFormatter(fmt)
            logger.addHandler(fh)
        except Exception:
            pass
    # Opt-in: 콘솔 로깅
    if os.environ.get("TSA_CONSOLE_LOG", "0").lower() in ("1", "true", "yes"):
        try:
            sh = logging.StreamHandler()
            sh.setLevel(logging.INFO)
            sh.setFormatter(fmt)
            logger.addHandler(sh)
        except Exception:
            pass
    return logger

logger = _setup_logger()

def _top_k_emotions(emotion_seq, k=5):
    from collections import defaultdict
    counts = defaultdict(int)
    for entry in emotion_seq:
        for e in entry.get("emotions", {}).keys():
            counts[e] += 1
    return sorted(counts.items(), key=lambda x: x[1], reverse=True)[:k]

def _maybe_load_src_config() -> dict:
    cfg_path = SRC_DIR / "config.py"
    if not cfg_path.exists():
        return {}
    spec = importlib.util.spec_from_file_location("project_config", str(cfg_path))
    if not spec or not spec.loader:
        return {}
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)  # type: ignore
        cfg = getattr(module, "ANALYZER_CONFIG", {})
        return cfg if isinstance(cfg, dict) else {}
    except Exception as e:
        logging.getLogger("project_config").warning(f"config.py load warning: {e}")
        return {}

def run_emotion_analysis_with_components(text, seq_analyzer, caus_analyzer, config_snapshot=None) -> dict:
    sentences, emotion_seq, paragraph_info = seq_analyzer.build_emotion_sequence(text)
    time_flow = seq_analyzer.analyze_time_flow(sentences)
    cause_effect = caus_analyzer.analyze_cause_effect(sentences, emotion_seq, context_window=1)
    complex_list = caus_analyzer.handle_complex_emotions(emotion_seq)
    refined = seq_analyzer.refine_sub_emotions(emotion_seq)
    deltas = seq_analyzer.calculate_emotion_changes(emotion_seq, time_flow)
    summary = {
        "paragraph_analysis": paragraph_info,
        "emotion_count": sum(len(e["emotions"]) for e in emotion_seq),
        "most_frequent_emotions": _top_k_emotions(emotion_seq, k=5),
    }
    result = AnalysisResult(
        emotion_sequence=emotion_seq,
        summary=summary,
        cause_effect=cause_effect,
        time_flow=time_flow,
        complex_emotions=complex_list,
        refined_sub_emotions=refined,
        emotion_changes=deltas,
        analysis_timestamp=datetime.now().isoformat(),
        config_snapshot=config_snapshot or {},
    )
    # [NEW] diagnostics exposure
    try:
        result.diagnostics.update({
            "time_flow_mode": time_flow.get("mode", "undetermined"),
            "time_flow_coverage": time_flow.get("coverage", 0.0),
            "reason_codes": time_flow.get("reason_codes", []),
        })
    except Exception:
        result.diagnostics.update({
            "time_flow_mode": "undetermined",
            "time_flow_coverage": 0.0,
            "reason_codes": [],
        })
    return result.to_dict()

if __name__ == "__main__":
    logger.info("[Main] TimeSeriesAnalyzer 시작")

    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--emotions", type=str, default=None)
    args, _ = ap.parse_known_args()

    candidates = []
    if args.emotions:
        candidates.append(Path(args.emotions).expanduser())
    env_path = os.getenv("EMOTIONS_JSON")
    if env_path:
        candidates.append(Path(env_path).expanduser())
    candidates += [
        BASE_DIR / "EMOTIONS.json",
        SRC_DIR / "EMOTIONS.json",
    ]
    json_path = None
    for c in candidates:
        if c and c.exists():
            json_path = c
            break
    if not json_path:
        tried = " | ".join(str(c) for c in candidates if c)
        raise FileNotFoundError("EMOTIONS.json not found. Tried: " + tried)

    with open(json_path, "r", encoding="utf-8") as f:
        emotions_json = json.load(f)
    required_roots = {"희", "노", "애", "락"}
    if not all(root in emotions_json for root in required_roots):
        raise ValueError("EMOTIONS.json 필수 감정 카테고리 누락")
    logger.info(f"EMOTIONS.json 로드 완료: {json_path}")

    global_rules = emotions_json.get("global_rules", {}) if isinstance(emotions_json, dict) else {}
    sample_config = {
        "max_emotions_per_sentence": 3,
        "conflict_threshold": 0.8,
        "confidence_conflict_threshold": 0.6,
        "confidence_conflict_penalty": 0.1,
        "confidence_synergy_threshold": 0.7,
        "confidence_synergy_bonus": 0.05,
        "conflict_strong_penalty": 0.05,
        "max_memory_usage": 85.0,
        "min_causality_strength": 0.3,
        "min_transition_score": 0.4,
        "intensity_change_mode": "scaled",
        "absolute_intensity_threshold": 0.1,
        "relative_intensity_threshold": 0.15,
        "negative_context_map": global_rules.get("negative_context_map", {
            "희": ["아니야", "불행", "슬프", "꺼려져"],
            "노": ["해소", "괜찮아", "용서", "가라앉"],
            "애": ["회복", "극복", "견뎌", "탈출"],
            "락": ["불쾌", "부정", "불편", "문제"],
        }),
        "enable_detailed_logging": True,
        "enable_advanced_nlp": True,
        "nlp_parser_settings": {"enable_dependency_parse": True, "dep_min_confidence": 0.5},
    }
    external_cfg = _maybe_load_src_config()
    if external_cfg:
        sample_config.update(external_cfg)
        logger.info("src/config.py ANALYZER_CONFIG applied.")

    try:
        has_method = hasattr(EmotionDataManager, "get_emotion_complexity")
    except NameError:
        has_method = False
    if not has_method:
        def _fallback_get_emotion_complexity(self, emotion_id: str) -> str:
            try:
                md = self.get_metadata(emotion_id)
                val = md.get("emotion_complexity")
                return val if isinstance(val, str) and val else "basic"
            except Exception:
                return "basic"
        EmotionDataManager.get_emotion_complexity = _fallback_get_emotion_complexity  # type: ignore
        logger.info("[Compat] get_emotion_complexity polyfilled.")
    logger.info(f"[Compat] Using EmotionDataManager from: {EmotionDataManager.__module__}.{EmotionDataManager.__name__}")

    dm = EmotionDataManager(emotions_json)
    seq = EmotionSequenceAnalyzer(dm, sample_config)
    caus = CausalityTransitionAnalyzer(dm, sample_config)

    test_cases = [
        {
            "id": "시간_흐름_1",
            "text": (
                "2024-01-01 아침에 일어나자마자 피곤함을 느꼈습니다.\n\n"
                "그래서 커피를 마셨더니 기분이 좋아졌고, 활력이 생겼습니다.\n"
                "오후가 되자 업무에 집중하면서 만족감을 느꼈고,\n"
                "저녁에는 성과를 이루어서 성취감으로 가득했습니다."
            ),
        },
        {
            "id": "감정_전이_1",
            "text": (
                "처음에는 발표가 너무 걱정되어 불안했습니다.\n"
                "하지만 준비를 하면서 점점 자신감이 생겼고,\n"
                "발표를 성공적으로 마치고 나서는 큰 성취감을 느꼈습니다.\n"
                "이후에는 동료들의 칭찬을 받으며 더욱 기뻤습니다."
            ),
        },
        {
            "id": "인과_관계_1",
            "text": (
                "갑작스러운 프로젝트 변경 때문에 처음에는 당황했습니다.\n"
                "하지만 팀원들과 함께 열심히 준비했기 때문에 자신감이 생겼고,\n"
                "결국 성공적으로 마무리하여 큰 보람을 느꼈습니다.\n"
                "그래서 지금은 어떤 변화가 와도 긍정적으로 받아들일 수 있습니다."
            ),
        },
        {
            "id": "부정문맥_테스트",
            "text": "나는 사실 별로 기쁘지 않다. 아니야, 오히려 좀 불행해. 하지만 조금은 희망이 생기는 것 같기도 해.",
        },
        {
            "id": "의존구문_테스트",
            "text": "내가 너무 피곤해서 약을 먹었다. 그러자 컨디션이 좋아졌고 하여 오후에는 무리 없이 일할 수 있었다.",
        },
    ]

    all_results = {}
    for case in test_cases:
        logger.info(f"\n[{case['id']}] 분석 시작")
        logger.info(f"입력 텍스트:\n{case['text']}")
        result_dict = run_emotion_analysis_with_components(
            text=case["text"],
            seq_analyzer=seq,
            caus_analyzer=caus,
            config_snapshot=sample_config,
        )
        emotion_changes = result_dict.get("emotion_changes", [])
        if emotion_changes:
            logger.info(f"  -> 감정 변화 {len(emotion_changes)}건 발생: {emotion_changes}")
        tf = result_dict.get("time_flow", {})
        if tf:
            ts_len = len(tf.get("timestamps", {}))
            logger.info(f"  -> 시간 흐름: {ts_len} 문장에 대해 타임스탬프 할당")
        ce = result_dict.get("cause_effect", [])
        if ce:
            logger.info(f"  -> 인과 관계: {len(ce)} 건 발견")
        seq_len = len(result_dict.get("emotion_sequence", []))
        logger.info(f"  -> 감정 시퀀스: 총 {seq_len} 문장 분석")
        if seq_len:
            first_e = result_dict["emotion_sequence"][0].get("emotions", {})
            logger.info(f"    첫 문장 감정: {first_e}")
        print(f"\n[TestCase: {case['id']}] 분석 완료. 결과 요약:")
        pprint(result_dict.get("summary", {}), width=100, compact=True)
        all_results[case["id"]] = {"input_text": case["text"], "analysis_result": result_dict}

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = LOGS_DIR / f"time_series_analyzer_{timestamp}.json"
    try:
        with open(out_path, "w", encoding="utf-8") as jf:
            json.dump(
                {"config": sample_config, "analysis_timestamp": datetime.now().isoformat(), "results": all_results},
                jf,
                ensure_ascii=False,
                indent=4,
            )
        logger.info(f"상세 분석 결과가 저장되었습니다: {out_path}")
    except Exception as e:
        logger.error(f"결과 저장 중 오류 발생: {e}")

    if psutil is not None:
        process = psutil.Process()
        mem_mb = process.memory_info().rss / 1024 / 1024
        cpu_pct = process.cpu_percent(interval=0.3)
        logger.info(f"\n[성능 메트릭]\n- 메모리 사용량: {mem_mb:.2f} MB\n- CPU 사용률: {cpu_pct}%\n- 처리된 테스트 케이스: {len(test_cases)}개\n")

    logger.info("시간적 흐름 + 인과/전이 + 감정 변화 Δ 분석 완료\n")
    print("===== 프로그램 실행이 완료되었습니다. =====")
