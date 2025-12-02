# emotion_relationship_analyzer.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import os, re, json, time, argparse, logging, heapq
from functools import lru_cache
import unicodedata
import threading

from collections import defaultdict, Counter
from logging.handlers import RotatingFileHandler
from typing import Dict, Any, List, Optional, Tuple, Iterable, Set
from copy import deepcopy

TOKEN_BOUNDARY_PATTERN = r"(?<![\uac00-\ud7a3A-Za-z0-9]){needle}(?![\uac00-\ud7a3A-Za-z0-9])"

@lru_cache(maxsize=4096)
def _get_token_regex(needle: str) -> re.Pattern:
    return re.compile(TOKEN_BOUNDARY_PATTERN.format(needle=re.escape(needle)))

def deep_update(base_dict: Dict[str, Any], update_dict: Dict[str, Any]) -> Dict[str, Any]:
    """딕셔너리 깊은 업데이트"""
    result = deepcopy(base_dict)
    for key, value in update_dict.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result

class TextAnalyzer:
    """간단한 텍스트 분석기"""
    def __init__(self, emotions_data: Dict[str, Any]):
        self.emotions_data = emotions_data

class _StageTimerContext:
    def __init__(self, timer: 'StageTimer', name: str):
        self.timer = timer
        self.name = name
        self.start = 0.0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        duration = time.perf_counter() - self.start
        self.timer.add(self.name, duration)


class StageTimer:
    def __init__(self, max_samples: int = 100):
        self.max_samples = max_samples
        self.timings: Dict[str, List[float]] = defaultdict(list)

    def timing(self, name: str) -> _StageTimerContext:
        return _StageTimerContext(self, name)

    def add(self, name: str, duration: float) -> None:
        lst = self.timings[name]
        lst.append(duration)
        if len(lst) > self.max_samples:
            del lst[: len(lst) - self.max_samples]

    def summary(self) -> Dict[str, Dict[str, float]]:
        summary: Dict[str, Dict[str, float]] = {}
        for stage, values in self.timings.items():
            if not values:
                continue
            sorted_vals = sorted(values)
            summary[stage] = {
                'p50': _percentile(sorted_vals, 0.5),
                'p95': _percentile(sorted_vals, 0.95),
            }
        return summary

    def log_summary(self, logger: logging.Logger) -> None:
        summary = self.summary()
        if not summary:
            return
        parts = [f"{stage}:p50={vals['p50']:.4f}s p95={vals['p95']:.4f}s" for stage, vals in summary.items()]
        logger.info("[timing] " + " | ".join(parts))


def _percentile(sorted_values: List[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if q <= 0:
        return sorted_values[0]
    if q >= 1:
        return sorted_values[-1]
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = q * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


_STAGE_TIMER = StageTimer(max_samples=64)

# =============================================================================
# Logger 설정
# =============================================================================
# 모듈 전역 로거(있어도 무방)
logger = logging.getLogger(__name__)

def setup_logger(log_dir: str, log_filename: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("emotion_main")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # 기본 비간섭: 핸들러 없으면 NullHandler만
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # opt-in: 환경변수로 파일/콘솔 로깅 허용
    if os.environ.get("ERA_FILE_LOG", "0") == "1":
        try:
            log_file_path = os.path.join(log_dir, log_filename)
            fh = RotatingFileHandler(log_file_path, mode="w", encoding="utf-8", maxBytes=5 * 1024 * 1024, backupCount=2)
            fh.setLevel(logging.INFO)
            fh.setFormatter(fmt)
            logger.addHandler(fh)
        except Exception:
            pass
    if os.environ.get("ERA_CONSOLE_LOG", "0") == "1":
        try:
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            ch.setFormatter(fmt)
            logger.addHandler(ch)
        except Exception:
            pass

    return logger


# =============================================================================
# Helper
# =============================================================================
def deep_merge_dicts(source: Dict, overrides: Dict) -> Dict:
    """ 중첩된 딕셔너리를 재귀적으로 업데이트하는 헬퍼 함수. """
    for key, value in overrides.items():
        if isinstance(value, dict) and key in source and isinstance(source[key], dict):
            source[key] = deep_merge_dicts(source[key], value)
        else:
            source[key] = value
    return source

def analyze_social_emotion_graph(
    text: str,
    emotions_data: Dict[str, Any],
    entities: Optional[List[Dict[str, Any]]] = None,
    max_depth: int = 2,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    builder = SocialEmotionGraph(emotions_data, config=config)
    return builder.analyze(text, entities=entities, max_depth=max_depth)


# =============================================================================
# SocialEmotionGraph
# =============================================================================
class SocialEmotionGraph:
    """라벨링 뼈대 기반 소셜 감정 그래프 빌더(클래스화, 하위호환용 래퍼 제공)."""

    DEFAULT_CONFIG = {
        "particles": ["", "가", "이", "을", "를", "은", "는"],
        "stop_connectors": {"함께", "같이", "그리고", "그러나", "하지만", "동시에"},
        "emotion_frames": ("느끼", "느꼈", "느낀", "나다", "났", "들었", "되었", "됐다"),
        "min_token_length": 2,
        "local_seq_threshold": 0.18,
        "dynlex_min_df": 1,  # 저빈도 토큰 수용
        "dynlex_min_score": 0.03,  # 텍스트 매칭 임계치 완화
        "dynlex_top_k": 5,
    }

    def __init__(
        self,
        emotions_data: Dict[str, Any],
        text_analyzer: Optional["TextAnalyzer"] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        self.emotions_data = emotions_data if isinstance(emotions_data, dict) else {}
        self.text_analyzer = text_analyzer or TextAnalyzer(self.emotions_data)
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}

        self.validator = EmotionValidatorExtended()
        self._labeling_index: Optional[Dict[str, Set[str]]] = None
        self._emotionish: Optional[Set[str]] = None

        # Memory monitor 초기화 (Python 3.11.13 호환성을 위해 비활성화)
        try:
            self.memory_monitor = MemoryMonitor()
        except Exception:
            class _DummyMem:
                def check_memory(self): return True
            self.memory_monitor = _DummyMem()

        # 캐시
        self._dynamic_lexicon: Optional[Dict[str, Dict[str, Any]]] = None
        meta_stop_raw = (self.emotions_data.get('metadata', {}) or {}).get('stop_tokens', [])
        if isinstance(meta_stop_raw, (list, set, tuple)):
            meta_stop_set = {str(t).strip() for t in meta_stop_raw if str(t).strip()}
        else:
            meta_stop_set = set()
        cfg_stop_raw = self.config.get('stop_tokens', [])
        if isinstance(cfg_stop_raw, (list, set, tuple)):
            cfg_stop_set = {str(t).strip() for t in cfg_stop_raw if str(t).strip()}
        else:
            cfg_stop_set = set()
        self._stop_tokens = set(meta_stop_set or cfg_stop_set)

    # ─────────────────────────────────────────────────────────────
    # 기본 유틸
    # ─────────────────────────────────────────────────────────────
    def _norm_key(self, s: str) -> str:
        """영문은 소문자, 한글은 조사/어미 제거한 베이스폼으로 정규화."""
        s = (s or "").strip()
        if not s:
            return ""
        if re.search(r"[가-힣]", s):
            return self._normalize_korean_token(s)
        return s.lower()

    def _build_or_load_dynamic_lexicon(
            self,
            emotions_data: Dict[str, Any],
            *,
            cache_path: Optional[str] = None,
            force_rebuild: bool = False,
            min_df: Optional[int] = None,
            max_terms: Optional[int] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """대용량 EMOTIONS.json 대비: 렉시콘을 파일로 캐시."""
        try:
            import pickle
            if (not force_rebuild) and cache_path and os.path.exists(cache_path):
                with open(cache_path, "rb") as f:
                    payload = pickle.load(f)
                lex = payload.get("lexicon") or {}
                if lex:
                    self._dynamic_lexicon = lex
                    return lex
        except Exception:
            pass  # 캐시 읽기 실패 시 재생성

        lex = self._build_dynamic_social_lexicon(
            emotions_data,
            min_df=min_df if min_df is not None else self.config.get("dynlex_min_df", 1),
            max_terms=max_terms if max_terms is not None else self.config.get("dynlex_max_terms", 50000),
        )

        try:
            if cache_path:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                import pickle  # ensure availability in this scope
                with open(cache_path, "wb") as f:
                    data = pickle.dumps({"lexicon": lex})
                    f.write(data)
        except Exception:
            pass  # 캐시 저장 실패는 무지

        self._dynamic_lexicon = lex
        return lex

    def _normalize_korean_token(self, tok: str) -> str:
        """
        조사/어미/구두점 제거 + 한글만 남김.
        예) '가족과' -> '가족', '오늘은' -> '오늘' (최종 매칭은 렉시콘 교차검증)
        """
        if not isinstance(tok, str):
            return ""
        t = re.sub(r"[^가-힣]", "", tok)
        if not t:
            return ""
        josa = ("들과", "과의", "와의", "으로", "에게", "에서", "라고", "이라", "와", "과",
                "은", "는", "이", "가", "을", "를", "도", "에", "로")
        for j in josa:
            if t.endswith(j) and len(t) > len(j) + 1:
                t = t[: -len(j)]
                break
        return t

    def _is_noise_token(self, tok: str) -> bool:
        '''Return True when the token is a stop item or too short to be meaningful.'''
        if not tok or len(tok) < 2:
            return True
        if self._stop_tokens:
            return tok in self._stop_tokens
        fallback_stop = {
            '오늘', '동시', '동시에', '함께', '그리고', '그러나', '하지만',
            '혹은', '평온', '평온함', '약간', '조금', '매우', '아주', '너무'
        }
        return tok in fallback_stop

    def _guess_entity_type_kor(self, term: str) -> str:
        """간단 타입 추정(동적 렉시콘에 타입 정보가 없을 때 보조)."""
        fam = {"가족", "부모", "부모님", "엄마", "아빠", "형", "누나", "오빠", "언니", "동생", "자녀", "아이", "아들", "딸", "배우자"}
        person = {"친구", "동료", "직장동료", "상사", "부하", "팀장", "팀원", "고객", "선생님", "선생", "학생"}
        org = {"회사", "팀", "학교", "부서", "병원", "클럽", "동아리"}
        if term in fam: return "family"
        if term in person: return "person"
        if term in org: return "org"
        return "unknown"

    def _compute_dominant_relationships(
            self,
            edges: List[Dict[str, Any]],
            *,
            top_n: int = 3,
            min_weight: float = 0.15,
            min_count: int = 1
    ) -> List[Dict[str, Any]]:
        """
        그래프 엣지들에서 상위 관계 Top-N을 뽑아 요약합니다.
        - 가중치 우선, 동률이면 count로 정렬
        - 너무 약한 엣지는(min_weight/min_count) 제외
        """
        if not edges:
            return []

        total_w = sum(float(e.get("weight", 0.0) or 0.0) for e in edges) or 1.0

        ranked = sorted(
            (
                e for e in edges
                if float(e.get("weight", 0.0) or 0.0) >= min_weight
                   and int(e.get("count", 0) or 0) >= min_count
            ),
            key=lambda e: (float(e.get("weight", 0.0) or 0.0), int(e.get("count", 0) or 0)),
            reverse=True,
        )

        out: List[Dict[str, Any]] = []
        for e in ranked[:max(1, top_n)]:
            w = float(e.get("weight", 0.0) or 0.0)
            c = int(e.get("count", 0) or 0)
            out.append({
                "source": e.get("source"),
                "target": e.get("target"),
                "emotion_id": e.get("emotion_id"),
                "weight": round(w, 3),
                "count": c,
                "weight_share": round(w / total_w, 3),
            })
        return out

    # ─────────────────────────────────────────────────────────────
    # 동적 소셜 렉시콘 (EMOTIONS.json → 용어/가중치/유형)
    # ─────────────────────────────────────────────────────────────
    def _build_dynamic_social_lexicon(
            self,
            emotions_data: Dict[str, Any],
            *,
            min_df: int = 2,
            max_terms: int = 50000
    ) -> Dict[str, Dict[str, Any]]:
        import re, math
        from collections import Counter, defaultdict

        # 설정 반영(최소 DF)
        min_df = self.config.get("dynlex_min_df", min_df)

        def harvest_strings(obj, path=""):
            if obj is None:
                return
            if isinstance(obj, str):
                yield (path, obj)
            elif isinstance(obj, dict):
                for k, v in obj.items():
                    subp = f"{path}.{k}" if path else str(k)
                    if isinstance(v, (dict, list, str)):
                        yield from harvest_strings(v, subp)
            elif isinstance(obj, list):
                for i, v in enumerate(obj):
                    subp = f"{path}[{i}]"
                    yield from harvest_strings(v, subp)

        counter = Counter()
        sources_map: Dict[str, set] = defaultdict(set)
        token_re = re.compile(r"[가-힣A-Za-z]{2,}")

        STOP = {
            "그리고", "그러나", "하지만", "혹은", "어떤", "어느", "그", "저", "것",
            "하다", "되다", "이다", "에서", "으로", "에게", "동시에", "함께", "정말", "매우",
            "아주", "너무", "조금", "오늘", "평온", "평온함", "약간"
        }
        FIELD_BONUS = [
            ("entities", 2.0), ("social_entities", 2.0), ("targets", 2.0), ("actors", 1.6),
            ("keywords", 1.3), ("key_phrases", 1.3), ("examples", 1.1), ("variations", 1.1),
            ("social_context", 1.4), ("context", 1.0), ("situation", 1.0), ("situations", 1.0)
        ]

        for path, s in harvest_strings(emotions_data):
            if not s or not isinstance(s, str):
                continue
            raw_tokens = [t for t in token_re.findall(s) if t not in STOP]
            if not raw_tokens:
                continue

            bonus = 1.0
            p_lower = path.lower()
            for key, w in FIELD_BONUS:
                if key in p_lower:
                    bonus = max(bonus, w)

            # 정규화 카운트(예: '가족과' -> '가족')
            for t in raw_tokens:
                base = self._normalize_korean_token(t)
                if not base or len(base) < 2:
                    continue
                counter[base] += bonus
                sources_map[base].add(path)

        if not counter:
            return {}

        most_common = counter.most_common(max_terms)
        max_val = float(most_common[0][1]) if most_common else 1.0

        lexicon: Dict[str, Dict[str, Any]] = {}
        for tok, val in most_common:
            if val < min_df:
                continue
            weight = math.log1p(val) / math.log1p(max_val)
            lexicon[tok] = {
                "type": "unknown",
                "weight": float(round(weight, 4)),
                "sources": list(sources_map.get(tok, []))[:5],
                "df": int(val if isinstance(val, int) else round(val))
            }

        # 명시적 엔티티 타입 반영
        try:
            ents = emotions_data.get("entities") or emotions_data.get("social_entities")
            if isinstance(ents, list):
                for e in ents:
                    if isinstance(e, dict):
                        eid = str(e.get("id") or e.get("name") or "").strip()
                        ety = str(e.get("type") or e.get("category") or "unknown").strip()
                        if eid and eid in lexicon and ety:
                            lexicon[eid]["type"] = ety
        except Exception:
            pass

        # 휴리스틱 타입 보정 (라벨링 품질 점검 병행)
        unknown_count = 0
        for tok, info in lexicon.items():
            if info.get("type") in (None, "", "unknown"):
                guessed_type = self._guess_entity_type_kor(tok)
                info["type"] = guessed_type
                if guessed_type == "unknown":
                    unknown_count += 1
        
        # 운영 환경에서는 unknown 타입이 많으면 경고
        if unknown_count > len(lexicon) * 0.3:  # 30% 이상이 unknown
            from config import EA_PROFILE, RENDER_DEPLOYMENT
            is_production = EA_PROFILE == "prod" or RENDER_DEPLOYMENT or os.getenv("PRODUCTION_MODE", "0") == "1"
            if is_production:
                logger.warning(f"[동적 렉시콘] unknown 타입 비율이 높음 ({unknown_count}/{len(lexicon)}). EMOTIONS.json 품질 점검 권장.")

        return lexicon

    # 텍스트 → 동적 렉시콘 기반 소셜 타깃 후보
    def _extract_social_targets_dynamic(
            self,
            text: str,
            lexicon: Dict[str, Dict[str, Any]],
            *,
            top_k: int = 5,
            min_score: float = 0.03
    ) -> List[Dict[str, Any]]:
        import re
        from collections import defaultdict

        if not text or not isinstance(lexicon, dict) or not lexicon:
            return []

        # 설정 반영
        top_k = self.config.get("dynlex_top_k", top_k)
        min_score = self.config.get("dynlex_min_score", min_score)

        token_re = re.compile(r"[가-힣A-Za-z]{2,}")
        raw_tokens = token_re.findall(text)
        if not raw_tokens:
            return []

        scores: Dict[str, float] = defaultdict(float)

        for t in raw_tokens:
            base = self._normalize_korean_token(t)
            if not base or len(base) < 2:
                continue
            if self._is_noise_token(base) or self._is_timeish(base) or self._emotion_frame_hit(base, text):
                continue
            if base in lexicon:
                scores[base] += float(lexicon[base].get("weight", 0.0))

        cand = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)

        out: List[Dict[str, Any]] = []
        kept: Set[str] = set()
        for tok, sc in cand:
            if sc < min_score:
                continue
            info = lexicon.get(tok, {}) or {}
            typ = info.get("type") or self._guess_entity_type_kor(tok) or "unknown"
            if typ == "unknown":
                # 재시도(휴리스틱 재평가)
                try:
                    typ = self._guess_entity_type_kor(tok)
                except Exception:
                    typ = "unknown"
            if typ == "unknown":
                continue
            if tok in kept:
                continue
            kept.add(tok)
            out.append({
                "id": tok,
                "type": typ,
                "label": tok,
                "score": round(float(sc), 3),
                "df": int(info.get("df") or 0),
                "source": "social",  # ★ 메트릭에서 집계되도록 명시
            })
            if len(out) >= max(1, top_k):
                break

        return out

    # ─────────────────────────────────────────────────────────────
    # 라벨 인덱스/엔티티 추출
    # ─────────────────────────────────────────────────────────────
    @staticmethod
    def _flatten_tokens(x: Any) -> Iterable[str]:
        if isinstance(x, str):
            yield x.strip()
        elif isinstance(x, (list, tuple, set)):
            for it in x:
                for t in SocialEmotionGraph._flatten_tokens(it):
                    if t:
                        yield t
        elif isinstance(x, dict):
            for v in x.values():
                for t in SocialEmotionGraph._flatten_tokens(v):
                    if t:
                        yield t

    @staticmethod
    def _dedup(seq: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen: Set[str] = set()
        out: List[Dict[str, Any]] = []
        for e in seq:
            if not isinstance(e, dict):
                continue
            k = str(e.get("id", "")) + "|" + str(e.get("type", ""))
            if k not in seen:
                seen.add(k)
                out.append(e)
        return out

    def _collect_emotionish_tokens(self, emotions_data: Dict[str, Any]) -> Set[str]:
        emo: Set[str] = set()
        particles = self.config["particles"]

        for cat, cdata in emotions_data.items():
            if not isinstance(cdata, dict):
                continue
            cat_s = str(cat)
            for p in particles:
                emo.add(cat_s + p)

            sub = (cdata.get("emotion_profile", {}) or {}).get("sub_emotions", {}) \
                  or cdata.get("sub_emotions", {}) or {}
            if isinstance(sub, dict):
                for sid, sdata in sub.items():
                    sname = str(sid)
                    for p in particles:
                        emo.add(sname + p)
                    prof = sdata.get("emotion_profile", {}) if isinstance(sdata, dict) else {}
                    for field in ("core_keywords", "keywords"):
                        for kw in (prof.get(field) or sdata.get(field) or []):
                            k = str(kw)
                            for p in particles:
                                emo.add(k + p)

        emo.update({"화", "화가", "분노", "짜증", "불안", "걱정", "슬픔", "기쁨", "우울"})
        return emo

    @staticmethod
    def _strip_particles(s: str) -> str:
        return re.sub(r"[이가을를은는의]$", "", s or "")

    @staticmethod
    def _is_timeish(label: str) -> bool:
        if re.search(r"(순간|때|동안)$", label):
            return True
        if re.search(r"[0-9]+(시|분|일|월|년|주)", label):
            return True
        return False

    def _is_connector(self, label: str) -> bool:
        return label in self.config["stop_connectors"]

    def _emotionish_like(self, base: str, emo: Set[str]) -> bool:
        if base in emo:
            return True
        for e in emo:
            if e in base and (len(base) - len(e) <= 1):
                return True
        return False

    def _emotion_frame_hit(self, token: str, text: str) -> bool:
        base = self._strip_particles(token)
        if not base or not text:
            return False
        try:
            pat = re.compile(rf"{re.escape(base)}.{0,6}({'|'.join(self.config['emotion_frames'])})")
            return bool(pat.search(text))
        except re.error:
            return False

    def _build_labeling_entity_index(self, emotions_data: Dict[str, Any]) -> Dict[str, Set[str]]:
        idx: Dict[str, Set[str]] = {"social": set(), "location": set(), "triggers": set(), "targets": set()}

        for _cat_name, cat in emotions_data.items():
            if not isinstance(cat, dict):
                continue

            cp = cat.get("context_patterns", {}) or {}
            inds = cp.get("indicators", {}) or {}

            for key in ("social", "location", "triggers"):
                toks = set(t.strip() for t in self._flatten_tokens(inds.get(key, [])) if isinstance(t, str))
                idx[key].update(toks)

            situations = cp.get("situations", {}) or {}
            for _sname, sdata in situations.items():
                for f in ("keywords", "examples", "variations"):
                    toks = set(t.strip() for t in self._flatten_tokens(sdata.get(f, [])) if isinstance(t, str))
                    idx["targets"].update(toks)
                trig = (sdata.get("emotion_progression", {}) or {}).get("trigger")
                if isinstance(trig, str) and trig.strip():
                    idx["triggers"].add(trig.strip())

            sub = cat.get("sub_emotions", {}) or cat.get("emotion_profile", {}).get("sub_emotions", {}) or {}
            for _sub_name, sub_data in sub.items():
                scp = sub_data.get("context_patterns", {}) or {}
                sinds = scp.get("indicators", {}) or {}
                for key in ("social", "location", "triggers"):
                    toks = set(t.strip() for t in self._flatten_tokens(sinds.get(key, [])) if isinstance(t, str))
                    idx[key].update(toks)

                ssit = scp.get("situations", {}) or {}
                for _sn, sdata in ssit.items():
                    for f in ("keywords", "examples", "variations"):
                        toks = set(t.strip() for t in self._flatten_tokens(sdata.get(f, [])) if isinstance(t, str))
                        idx["targets"].update(toks)
                    trig2 = (sdata.get("emotion_progression", {}) or {}).get("trigger")
                    if isinstance(trig2, str) and trig2.strip():
                        idx["triggers"].add(trig2.strip())

        def _clean(s: str) -> str:
            return re.sub(r"\s+", "", s)

        for k in list(idx.keys()):
            idx[k] = set(_clean(t) for t in idx[k] if isinstance(t, str) and t.strip())
        return idx

    def _extract_entities_from_text_via_index(
        self,
        text: str,
        idx: Dict[str, Set[str]],
        emotionish: Optional[Set[str]] = None
    ) -> List[Dict[str, Any]]:
        emo = emotionish or set()
        bucket_hits: Dict[str, Set[str]] = {k: set() for k in ("social", "location", "triggers", "targets")}

        for bucket in bucket_hits.keys():
            for tok in idx.get(bucket, set()):
                if not tok or len(tok) < self.config["min_token_length"]:
                    continue
                pat = _get_token_regex(tok)
                if pat.search(text):
                    bucket_hits[bucket].add(tok)

        def _longest_first_keep(tokens: Set[str]) -> List[str]:
            kept: List[str] = []
            for t in sorted(tokens, key=lambda x: len(x), reverse=True):
                if not any(t in k and t != k for k in kept):
                    kept.append(t)
            return kept

        for b in list(bucket_hits.keys()):
            bucket_hits[b] = set(_longest_first_keep(bucket_hits[b]))

        def should_keep(tok: str, bucket: str) -> bool:
            base = self._strip_particles(tok)
            if bucket == "targets":
                if self._is_timeish(tok) or self._is_connector(tok) or self._emotion_frame_hit(tok, text):
                    return False
                if self._emotionish_like(base, emo):
                    return False
            return True

        filtered: Dict[str, Set[str]] = {k: set() for k in bucket_hits.keys()}
        for b, toks in bucket_hits.items():
            for t in toks:
                if should_keep(t, b):
                    filtered[b].add(t)

        token_best_type: Dict[str, str] = {}
        for t in filtered["social"]:
            token_best_type[t] = "social"
        for t in filtered["targets"]:
            token_best_type.setdefault(t, "target")
        for t in filtered["location"]:
            token_best_type.setdefault(t, "location")
        for t in filtered["triggers"]:
            token_best_type.setdefault(t, "trigger")

        ents = [
            {"id": f"{token_best_type[t]}:{t}", "type": token_best_type[t], "label": t, "source": "label"}
            for t in sorted(token_best_type.keys(), key=lambda x: len(x), reverse=True)
        ]
        return self._dedup(ents)

    @staticmethod
    def _naive_entities_from_text(text: str) -> List[Dict[str, Any]]:
        ents: List[Dict[str, Any]] = []
        if any(k in text for k in ["나", "저", "우린", "우리"]):
            ents.append({"id": "speaker", "mentions": ["나", "저", "우리"], "type": "person", "source": "heuristic"})
        if any(k in text for k in ["너", "당신", "그대"]):
            ents.append({"id": "addressee", "mentions": ["너", "당신"], "type": "person", "source": "heuristic"})
        return ents

    # ─────────────────────────────────────────────────────────────
    # 시퀀스/강도 헬퍼
    # ─────────────────────────────────────────────────────────────
    def _local_emotion_sequence(
        self,
        text: str,
        emotions_data: Dict[str, Any],
        threshold: float
    ) -> List[Dict[str, Any]]:
        sentences = re.split(r'(?<=[.!?])\s+|[\n\r]+', (text or "").strip())
        seq: List[Dict[str, Any]] = []
        for s in sentences:
            s = s.strip()
            if not s:
                continue
            try:
                scores = self.text_analyzer.analyze_emotion(s)
            except Exception as e:
                logger.warning(f"[SocialGraph] 문장 점수화 실패: {e}")
                continue
            if not isinstance(scores, dict):
                continue
            dom: Dict[str, float] = {}
            for k, v in scores.items():
                if k == "_intrasentence_candidates":
                    continue
                try:
                    fv = float(v)
                except Exception:
                    continue
                if fv > threshold:
                    dom[k] = fv
            if dom:
                seq.append({"text": s, "emotions": dom, "intensity": {"score": max(dom.values())}, "stage": "unknown"})
        return seq

    @staticmethod
    def _pick_peak_emotion(em_field: Any) -> Tuple[Optional[str], float]:
        emo_id: Optional[str] = None
        peak_val: float = 0.0
        if isinstance(em_field, dict) and em_field:
            try:
                emo_id, peak_val = max(((str(k), float(v)) for k, v in em_field.items()), key=lambda kv: kv[1])
            except Exception:
                emo_id = next(iter(em_field.keys()))
                try:
                    peak_val = float(em_field.get(emo_id, 0.0) or 0.0)
                except Exception:
                    peak_val = 0.0
        elif isinstance(em_field, list) and em_field and isinstance(em_field[0], dict):
            emo_id = em_field[0].get("emotion_id")
            try:
                peak_val = float(em_field[0].get("score", 0.0))
            except Exception:
                peak_val = 0.0
        return emo_id, peak_val

    @staticmethod
    def _extract_intensity_score(entry: Dict[str, Any]) -> float:
        inten = entry.get("intensity", {})
        if isinstance(inten, dict) and "score" in inten:
            try:
                return float(inten.get("score", 0.0))
            except Exception:
                return 0.0
        return 0.0

    def _derive_top_emotion_from_sequence(self, seq: List[Dict[str, Any]]) -> Tuple[Optional[str], float]:
        """시퀀스 전체에서 가장 대표적인 감정 1개(엣지 가중치/레이블용)."""
        if not isinstance(seq, list) or not seq:
            return (None, 0.0)
        best_eid, best_score = None, 0.0
        for seg in seq:
            if not isinstance(seg, dict):
                continue
            try:
                sc = float((seg.get("intensity") or {}).get("score") or 0.0)
            except Exception:
                sc = 0.0
            eid, peak = self._pick_peak_emotion(seg.get("emotions"))
            score = sc or peak or 0.0
            if score > best_score and isinstance(eid, str):
                best_eid, best_score = eid, float(score)
        return (best_eid, float(best_score))

    # ─────────────────────────────────────────────────────────────
    # 타깃 선택/엣지 확장
    # ─────────────────────────────────────────────────────────────
    def _select_best_target(
        self,
        entities: List[Dict[str, Any]],
        *,
        emotionish: Optional[Set[str]] = None,
        text: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        소셜 추출된 후보를 최우선으로, 타입 가중치 + 텍스트 빈도 스코어로 선택.
        부정편향(노/애↑)일 때 environment에 보너스.
        """
        entities = entities or []
        seq = (self.emotions_data or {}).get("emotion_sequence") or []
        neg_bias = False
        try:
            if seq and isinstance(seq[0], dict) and isinstance(seq[0].get("emotions"), dict):
                e = seq[0]["emotions"]
                pos = max(float(e.get("희", 0) or 0), float(e.get("락", 0) or 0))
                neg = max(float(e.get("노", 0) or 0), float(e.get("애", 0) or 0))
                neg_bias = (neg >= pos + 0.05)
        except Exception:
            neg_bias = False

        def _type_rank(tp: str) -> float:
            return {"family": 0.25, "person": 0.2, "org": 0.1, "environment": 0.15}.get(tp, 0.0)

        def _score(e: Dict[str, Any]) -> float:
            s = 0.0
            if e.get("source") == "social":
                s += 0.6
            t = e.get("type", "unknown")
            s += _type_rank(t)
            label = (e.get("label") or e.get("id") or "")
            if isinstance(label, str) and text:
                s += min(0.15, 0.05 * str(text).count(label))
            s += float(e.get("confidence", 0.0)) * 0.2
            if neg_bias and t == "environment":
                s += 0.3
            return s

        ranked = sorted([e for e in entities if isinstance(e, dict)], key=_score, reverse=True)
        if ranked:
            best = dict(ranked[0])
            best["source"] = best.get("source") or "heuristic"
            best.setdefault("label", best.get("id"))
            return best
        return None

    def _extend_edges_with_social_entities(
        self,
        edges_acc: Dict[Tuple[str, str, str], Dict[str, Any]],
        *,
        src_id: str,
        dst_id: str,
        top_eid: Optional[str],
        top_score: float,
        social_entities: Optional[List[Dict[str, Any]]]
    ) -> None:
        """소셜 엔티티로 확장된 엣지 추가. unknown/저신뢰/중복은 제외."""
        if not social_entities or not isinstance(social_entities, list) or not top_eid:
            return
        for ent in social_entities:
            try:
                tid = str(ent.get("id", "") or "")
                if not tid or tid in (src_id, dst_id):
                    continue
                if ent.get("type") == "unknown":
                    continue
                conf = float(ent.get("confidence", ent.get("score", 0.0)) or 0.0)
                if conf < 0.5:
                    continue
                key = (src_id, tid, str(top_eid))
                edges_acc[key]["weight"] += max(0.0, float(top_score or 0.0))
                edges_acc[key]["count"] += 1
            except Exception:
                continue

    # ─────────────────────────────────────────────────────────────
    # 메인: 그래프 생성
    # ─────────────────────────────────────────────────────────────
    def analyze(
            self,
            text: str,
            entities: Optional[List[Dict[str, Any]]] = None,
            max_depth: int = 2  # (호환용, 내부 미사용)
    ) -> Dict[str, Any]:
        """
        텍스트 → 감정 시퀀스 → 엔티티 병합(라벨/휴리스틱/동적 소셜) →
        타깃 선택 → 엣지 누적 → 노이즈 억제 노드 구성 → 메트릭/요약 반환
        (구버전 _compute_dominant_relationships 시그니처와의 호환 처리 포함)
        """
        try:
            # 0) 유효성
            is_valid, issues = self.validator.validate_emotion_structure(self.emotions_data)
            if not is_valid:
                return {"nodes": [], "edges": [], "metrics": {"validation_errors": issues}}

            # 1) 감정 시퀀스(외부 주입 우선)
            seq = self.emotions_data.get("emotion_sequence")
            if not isinstance(seq, list) or not seq:
                seq = self._local_emotion_sequence(
                    text, self.emotions_data, self.config.get("local_seq_threshold", 0.18)
                )

            # 2) 라벨 인덱스 / 감정표면형 캐시
            if self._labeling_index is None:
                self._labeling_index = self._build_labeling_entity_index(self.emotions_data)
            if self._emotionish is None:
                self._emotionish = self._collect_emotionish_tokens(self.emotions_data)

            # 3) 동적 렉시콘 준비(캐시 지원)
            if getattr(self, "_dynamic_lexicon", None) is None:
                if hasattr(self, "_build_or_load_dynamic_lexicon"):
                    self._build_or_load_dynamic_lexicon(
                        self.emotions_data,
                        cache_path=self.config.get("dynlex_cache_path"),
                        force_rebuild=self.config.get("dynlex_rebuild", False),
                        min_df=self.config.get("dynlex_min_df", 1),
                        max_terms=self.config.get("dynlex_max_terms", 50000),
                    )
                else:
                    self._dynamic_lexicon = self._build_dynamic_social_lexicon(
                        self.emotions_data,
                        min_df=self.config.get("dynlex_min_df", 1),
                        max_terms=self.config.get("dynlex_max_terms", 50000),
                    )

            # 4) 엔티티 결합: 외부→라벨→휴리스틱→소셜(동적)
            label_entities = self._extract_entities_from_text_via_index(
                text, self._labeling_index, self._emotionish
            ) or []
            base_entities = self._naive_entities_from_text(text) or []

            scan_text = (text or "") + " " + " ".join(
                str((seg.get("text") or "")) for seg in (seq or []) if isinstance(seg, dict)
            )
            social_entities = self._extract_social_targets_dynamic(
                scan_text,
                getattr(self, "_dynamic_lexicon", {}) or {},
                top_k=self.config.get("dynlex_top_k", 5),
                min_score=self.config.get("dynlex_min_score", 0.03),
            ) or []
            for se in social_entities:
                if isinstance(se, dict):
                    se.setdefault("source", "social")

            entities_all = self._dedup((entities or []) + label_entities + base_entities + social_entities)

            # 5) 주체/타깃
            subject = next((e for e in entities_all if e.get("id") == "speaker"), None)
            target = self._select_best_target(entities_all, emotionish=self._emotionish, text=text)
            if not target and social_entities:
                target = social_entities[0]

            src_id = (subject or {"id": "narrator", "type": "person"}).get("id", "narrator")
            dst_id = (target or {"id": "context", "type": "unknown"}).get("id", "context")

            # 타깃 출처
            target_source = "fallback"
            if target:
                tid = target.get("id")
                if any((e.get("id") == tid) for e in (social_entities or [])):
                    target_source = "social"
                elif any((e.get("id") == tid) for e in (label_entities or [])):
                    target_source = "label"
                elif any((e.get("id") == tid) for e in (base_entities or [])):
                    target_source = "heuristic"

            # 6) 대표 감정
            top_eid, top_score = self._derive_top_emotion_from_sequence(seq)

            # 7) 엣지 누적
            from collections import defaultdict
            edges_acc: Dict[Tuple[str, str, str], Dict[str, Any]] = defaultdict(lambda: {"weight": 0.0, "count": 0})

            # 소셜 후보 확장(선택된 dst 제외)
            self._extend_edges_with_social_entities(
                edges_acc,
                src_id=src_id,
                dst_id=dst_id,
                top_eid=top_eid,
                top_score=top_score,
                social_entities=social_entities,
            )

            # 본문 시퀀스 → 주 타깃(dst)
            for item in seq:
                if not isinstance(item, dict):
                    continue
                emo_id, peak_val = self._pick_peak_emotion(item.get("emotions"))
                if not emo_id:
                    continue
                score = self._extract_intensity_score(item) or peak_val
                key = (src_id, dst_id, str(emo_id))
                edges_acc[key]["weight"] += max(0.0, float(score or 0.0))
                edges_acc[key]["count"] += 1

            edges: List[Dict[str, Any]] = sorted(
                (
                    {"source": s, "target": t, "emotion_id": e, "weight": round(v["weight"], 3),
                     "count": int(v["count"])}
                    for (s, t, e), v in edges_acc.items()
                ),
                key=lambda x: (x["weight"], x["count"]),
                reverse=True,
            )

            # 8) 노드(unknown 소셜 제외)
            nodes_map: Dict[str, Dict[str, Any]] = {}
            nodes_map[src_id] = {"id": src_id, "type": (subject or {}).get("type", "person")}
            nodes_map[dst_id] = {"id": dst_id, "type": (target or {}).get("type", "unknown")}
            for ent in (social_entities or []):
                if not isinstance(ent, dict):
                    continue
                tid2 = str(ent.get("id", "") or "")
                if not tid2 or tid2 in (src_id, dst_id):
                    continue
                if ent.get("type") == "unknown":
                    continue
                nodes_map.setdefault(tid2, {"id": tid2, "type": ent.get("type", "unknown")})
            nodes = list(nodes_map.values())

            # 9) 메트릭
            avg_w = (sum(x["weight"] for x in edges) / len(edges)) if edges else 0.0
            metrics = {
                "unique_nodes": len(nodes),
                "unique_edges": len(edges),
                "avg_edge_weight": avg_w,
                "matched_entities": len(label_entities),
                "social_entities_detected": len([e for e in (social_entities or []) if e.get("source") == "social"]),
                "selected_target": (target or {}).get("id", "context"),
                "selected_target_type": (target or {}).get("type", "unknown"),
                "selected_target_reason": (target or {}).get("label") or (target or {}).get("id"),
                "selected_target_source": target_source,
                "lexicon_size": len(getattr(self, "_dynamic_lexicon", {}) or {}),
                "lexicon_hits": len(social_entities or []),
                "graph_density_approx": (
                        (2 * len(edges)) / (len(nodes) * (len(nodes) - 1))
                ) if len(nodes) > 1 else 0.0,
            }

            # 10) 우세 관계 요약(구/신 시그니처 모두 지원)
            topk = int(self.config.get("dominant_top_k", 3) or 3)
            dominant: List[Dict[str, Any]] = []
            try:
                # 신버전( top_k 지원 )
                dominant = self._compute_dominant_relationships(edges, top_k=topk)
            except TypeError:
                # 구버전( positional only )
                try:
                    dom_any = self._compute_dominant_relationships(edges)
                    if isinstance(dom_any, list):
                        dominant = dom_any[:topk]
                    elif isinstance(dom_any, dict) and "dominant_relationships" in dom_any:
                        v = dom_any.get("dominant_relationships") or []
                        dominant = v[:topk] if isinstance(v, list) else []
                    else:
                        # 형식 미정이면 최대한 비슷하게 맞춤
                        dominant = []
                except Exception:
                    dominant = []

            return {"nodes": nodes, "edges": edges, "metrics": metrics, "dominant_relationships": dominant}

        except Exception as e:
            logger.exception(f"[SocialEmotionGraph.analyze] 오류: {e}")
            return {"nodes": [], "edges": [], "metrics": {}, "error": str(e)}


# =============================================================================
# TextAnalyzer
# =============================================================================
class TextAnalyzer:

    def __init__(self, emotions_data: Dict[str, Dict[str, Any]]):
        self.emotions_data = emotions_data or {}
        self._validate_labeling_schema()
        self.emotion_keywords = self._build_emotion_keywords()             # {emotion_id: {'kw_weights','combos','synonyms'}}
        self.direct_emotion_patterns = self._build_direct_emotion_patterns()
        self.intensity_modifiers = self._build_intensity_modifiers()       # 유지: 외부 호환
        self._compiled_context = self._build_dynamic_context_data()        # {id: {'compiled': [...], 'sudden': [...], 'gradual': [...]}}
        self._global_intensity = self._build_global_intensity_lexicon()    # {'high': [...], 'medium': [...], 'low': [...]}
        self._per_emotion_intensity = self._build_per_emotion_intensity()  # {emotion_id: {'high': set(), ...}}
        self._primary_by_emotion = self._build_primary_category_map()      # {emotion_or_sub_id: '희'|'노'|'애'|'락'}
        self._special_tokens = self._build_dynamic_special_tokens()        # {'explosive': [...], 'gradual': [...]}
        self._combo_regex_cache: Dict[Tuple[str, ...], re.Pattern] = {}

    # -------- 정규화/토큰화 유틸 --------
    def _normalize(self, text: str) -> str:
        if not isinstance(text, str): return ""
        t = unicodedata.normalize("NFKC", text).strip().lower()
        punct = r"""'"\[\]\(\)\{\}<>,\\.!?;:~^|/\\\-_=+*&@#%`"""
        t = re.sub(f"([{punct}])", r" \1 ", t)
        t = re.sub(r"\s+", " ", t)
        return t.strip()

    def tokenize(self, text: str) -> List[str]:
        return self._normalize(text).split()

    def _make_ngrams(self, tokens: List[str], n: int) -> Set[Tuple[str, ...]]:
        return {tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)} if n > 0 else set()

    # -------- 스키마 검증 --------
    def _validate_labeling_schema(self):
        def _check(node: Dict[str, Any], path: str, top_level: bool):
            if not isinstance(node, dict):
                logger.warning(f"[스키마] dict 아님: {path}")
                return
            if 'metadata' not in node:
                logger.warning(f"[스키마] '{path}'에 'metadata' 누락")
            if not top_level and 'emotion_profile' not in node:
                logger.warning(f"[스키마] '{path}'에 'emotion_profile' 누락")
            subs = node.get('sub_emotions', {})
            if isinstance(subs, dict):
                for sub_name, sub_data in subs.items():
                    _check(sub_data, f"{path}.{sub_name}", False)
        for emotion_id, emotion_info in self.emotions_data.items():
            if emotion_id in ('emotion_sequence', 'metadata'):
                continue
            if isinstance(emotion_info, dict):
                _check(emotion_info, emotion_id, True)

    # -------- 키워드/콤보/동의어 구축 --------
    def _build_emotion_keywords(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for emotion_id, emotion_info in self.emotions_data.items():
            if emotion_id in ('emotion_sequence', 'metadata') or not isinstance(emotion_info, dict):
                continue
            kw_weights: Dict[str, float] = {}
            combos: List[Tuple[Tuple[str, ...], float]] = []
            synonyms_map: Dict[str, List[str]] = defaultdict(list)
            # 메타/임계
            emotion_profile = emotion_info.get('emotion_profile', {}) or {}
            direct_sub_emotions = emotion_info.get('sub_emotions')
            if direct_sub_emotions is None:
                direct_sub_emotions = emotion_profile.get('sub_emotions', {})
            sub_emotions = direct_sub_emotions if isinstance(direct_sub_emotions, dict) else {}
            sentiment_data = emotion_info.get('sentiment_analysis', {}) or {}
            intensity_modifiers = sentiment_data.get('intensity_modifiers', {}) or {}
            amplifiers = set(intensity_modifiers.get('amplifiers', []) or [])
            diminishers = set(intensity_modifiers.get('diminishers', []) or [])
            ml_metadata = emotion_info.get('ml_training_metadata', {}) or {}
            context_req = ml_metadata.get('context_requirements', {}) or {}
            complexity = (emotion_info.get('metadata', {}) or {}).get('emotion_complexity', 'basic')
            min_required_keywords = (context_req.get('required_keywords', {}) or {}).get(complexity, 0)

            def _calc_weight(word: str, base: float) -> float:
                if word in amplifiers:   return base * 1.5
                if word in diminishers:  return base * 0.7
                return base
            # 상위 프로필 키워드
            for kw in (emotion_profile.get('core_keywords', []) or []):
                if not isinstance(kw, str): continue
                norm = self._normalize(kw)
                if not norm: continue
                kw_weights[norm] = max(kw_weights.get(norm, 0.0), _calc_weight(norm, 1.0))
            # 동의어(상위)
            for base_kw, syns in (emotion_profile.get('keyword_synonyms', {}) or {}).items():
                if not isinstance(base_kw, str) or not isinstance(syns, list): continue
                b = self._normalize(base_kw)
                if not b: continue
                for s in syns:
                    if isinstance(s, str) and s.strip():
                        synonyms_map[b].append(self._normalize(s))
            # 서브 감정들 순회
            for sub_id, sub_info in (sub_emotions or {}).items():
                if not isinstance(sub_info, dict): continue
                sub_prof = sub_info.get('emotion_profile', {}) or {}
                # 키워드
                for keyword in (sub_prof.get('core_keywords', []) or []):
                    if not isinstance(keyword, str): continue
                    norm = self._normalize(keyword)
                    if not norm: continue
                    kw_weights[norm] = max(kw_weights.get(norm, 0.0), _calc_weight(norm, 1.2))
                # 강도 예시에서 토큰 추출
                intensity_levels = sub_prof.get('intensity_levels', {}) or {}
                intensity_examples = intensity_levels.get('intensity_examples', {}) or {}
                base_map = {'high': 1.2, 'medium': 1.0, 'low': 0.8}
                if isinstance(intensity_examples, dict):
                    for level_key, examples in intensity_examples.items():
                        base_val = base_map.get(level_key, 1.0)
                        if not isinstance(examples, list): continue
                        for ex in examples:
                            for tk in self.tokenize(ex):
                                kw_weights[tk] = max(kw_weights.get(tk, 0.0), _calc_weight(tk, base_val))
                # 관련 감정명도 보조 키워드로 사용
                related = sub_prof.get('related_emotions', {}) or {}
                base_map_re = {'positive': 1.2, 'negative': 0.8, 'neutral': 1.0}
                for e_type, emotions_list in related.items():
                    base_val = base_map_re.get(e_type, 1.0)
                    if not isinstance(emotions_list, list): continue
                    for rel in emotions_list:
                        if not isinstance(rel, str): continue
                        r = self._normalize(rel)
                        if r:
                            kw_weights[r] = max(kw_weights.get(r, 0.0), _calc_weight(r, base_val))
                # linguistic_patterns
                ling = sub_info.get('linguistic_patterns', {}) or {}
                # 1) 키프레이즈(문구)
                for kp in (ling.get('key_phrases', []) or []):
                    if not isinstance(kp, dict): continue
                    p_txt = kp.get('pattern')
                    if not isinstance(p_txt, str) or not p_txt.strip(): continue
                    w_val = float(kp.get('weight', 1.0))
                    tokens = self.tokenize(p_txt)
                    if len(tokens) > 1:
                        combos.append((tuple(tokens), min(_calc_weight(p_txt, w_val), 2.0)))
                    else:
                        k = tokens[0]
                        kw_weights[k] = max(kw_weights.get(k, 0.0), _calc_weight(k, w_val))
                    # 2) 동의어
                    for base_kw, syns in (kp.get('synonyms', {}) or {}).items():
                        if not isinstance(base_kw, str) or not isinstance(syns, list): continue
                        b = self._normalize(base_kw)
                        if not b: continue
                        for s in syns:
                            if isinstance(s, str) and s.strip():
                                synonyms_map[b].append(self._normalize(s))
            kw_weights = {k: w for k, w in kw_weights.items() if w >= 0.2}
            # 최소 요구 키워드 미만이면 보수적으로 비우기(학습 메타 정책 준수)
            if len(kw_weights) < (min_required_keywords or 0):
                kw_weights, combos, synonyms_map = {}, [], {}
            result[emotion_id] = {
                'kw_weights': kw_weights,                           # {token: weight}
                'combos': combos,                                   # [((t1,t2,...), weight), ...]
                'synonyms': {k: sorted(set(v)) for k, v in synonyms_map.items()}
            }
        return result

    # -------- 직접 패턴(참조용) --------
    def _build_direct_emotion_patterns(self) -> Dict[str, List[str]]:
        direct_patterns: Dict[str, List[str]] = {}
        for emotion_id, emotion_info in self.emotions_data.items():
            if emotion_id in ('emotion_sequence', 'metadata') or not isinstance(emotion_info, dict):
                continue
            patterns: List[str] = []
            emotion_profile = emotion_info.get('emotion_profile', {}) or {}
            direct_sub_emotions = emotion_info.get('sub_emotions')
            if direct_sub_emotions is None:
                direct_sub_emotions = emotion_profile.get('sub_emotions', {})
            sub_emotions = direct_sub_emotions if isinstance(direct_sub_emotions, dict) else {}
            core_keywords = emotion_profile.get('core_keywords', []) or []
            patterns.extend([k for k in core_keywords if isinstance(k, str)])
            intensity_levels = emotion_profile.get('intensity_levels', {}) or {}
            intensity_examples = intensity_levels.get('intensity_examples', {}) or {}
            if isinstance(intensity_examples, dict):
                for ex_list in intensity_examples.values():
                    if isinstance(ex_list, list):
                        patterns.extend([ex for ex in ex_list if isinstance(ex, str)])
            for sub_info in (sub_emotions or {}).values():
                if not isinstance(sub_info, dict): continue
                sub_prof = sub_info.get('emotion_profile', {}) or {}
                sub_core = sub_prof.get('core_keywords', []) or []
                patterns.extend([x for x in sub_core if isinstance(x, str)])
                sub_intensity = sub_prof.get('intensity_levels', {}) or {}
                sub_examples = sub_intensity.get('intensity_examples', {}) or {}
                if isinstance(sub_examples, dict):
                    for ex_list2 in sub_examples.values():
                        if isinstance(ex_list2, list):
                            patterns.extend([ex for ex in ex_list2 if isinstance(ex, str)])
            uniq = list({self._normalize(p) for p in patterns if isinstance(p, str) and p.strip()})
            direct_patterns[emotion_id] = uniq
        return direct_patterns

    # -------- 강도 수식어(기존 API 유지용) --------
    def _build_intensity_modifiers(self) -> Dict[str, List[str]]:
        modifiers = {'high': set(), 'medium': set(), 'low': set()}
        try:
            for _, emotion_info in self.emotions_data.items():
                if not isinstance(emotion_info, dict): continue
                self._collect_intensity_from_emotion_info(emotion_info, modifiers)
                for sub_info in (emotion_info.get('sub_emotions', {}) or {}).values():
                    if isinstance(sub_info, dict):
                        self._collect_intensity_from_emotion_info(sub_info, modifiers)
            return {k: sorted(v) for k, v in modifiers.items()}
        except Exception as e:
            logger.exception(f"[강도 수식어 구축] 오류: {e}")
            return {'high': [], 'medium': [], 'low': []}

    def _collect_intensity_from_emotion_info(self, emotion_info: Dict[str, Any], modifiers: Dict[str, Set[str]]) -> None:
        emotion_profile = emotion_info.get('emotion_profile', {}) or {}
        intensity_levels = emotion_profile.get('intensity_levels', {}) or {}
        if not isinstance(intensity_levels, dict): return
        intensity_mods = intensity_levels.get('intensity_modifiers', {}) or {}
        if not isinstance(intensity_mods, dict): return
        for level_key, words_list in intensity_mods.items():
            if level_key not in modifiers or not isinstance(words_list, list): continue
            for w in words_list:
                if isinstance(w, str) and w.strip():
                    modifiers[level_key].add(self._normalize(w))

    # -------- 추가 강도 사전(전역/감정별) --------
    def _build_global_intensity_lexicon(self) -> Dict[str, List[str]]:
        intensity_map = {'high': set(), 'medium': set(), 'low': set()}
        for _, emotion_info in self.emotions_data.items():
            if not isinstance(emotion_info, dict): continue
            prof = emotion_info.get('emotion_profile', {}) or {}
            i_data = prof.get('intensity_data', {}) or {}
            for level in ('high', 'medium', 'low'):
                for w in (i_data.get(level, []) or []):
                    if isinstance(w, str) and w.strip():
                        intensity_map[level].add(self._normalize(w))
            for sub_info in (emotion_info.get('sub_emotions', {}) or {}).values():
                if not isinstance(sub_info, dict): continue
                sprof = sub_info.get('emotion_profile', {}) or {}
                si = sprof.get('intensity_data', {}) or {}
                for level in ('high', 'medium', 'low'):
                    for w in (si.get(level, []) or []):
                        if isinstance(w, str) and w.strip():
                            intensity_map[level].add(self._normalize(w))
        return {k: sorted(v) for k, v in intensity_map.items()}

    def _build_per_emotion_intensity(self) -> Dict[str, Dict[str, Set[str]]]:
        per_map: Dict[str, Dict[str, Set[str]]] = {}
        for emotion_id, emotion_info in self.emotions_data.items():
            if emotion_id in ('emotion_sequence', 'metadata') or not isinstance(emotion_info, dict):
                continue
            per_map[emotion_id] = {'high': set(), 'medium': set(), 'low': set()}
            prof = emotion_info.get('emotion_profile', {}) or {}
            i_data = prof.get('intensity_data', {}) or {}
            for level in ('high', 'medium', 'low'):
                for w in (i_data.get(level, []) or []):
                    if isinstance(w, str) and w.strip():
                        per_map[emotion_id][level].add(self._normalize(w))
            for sub_id, sub_info in (emotion_info.get('sub_emotions', {}) or {}).items():
                if not isinstance(sub_info, dict): continue
                sprof = sub_info.get('emotion_profile', {}) or {}
                si = sprof.get('intensity_data', {}) or {}
                # 서브도 독립 키로 보유(정밀도 ↑)
                per_map[sub_id] = {'high': set(), 'medium': set(), 'low': set()}
                for level in ('high', 'medium', 'low'):
                    for w in (si.get(level, []) or []):
                        if isinstance(w, str) and w.strip():
                            per_map[sub_id][level].add(self._normalize(w))
        return per_map

    def _build_primary_category_map(self) -> Dict[str, str]:
        m: Dict[str, str] = {}
        for eid, einfo in self.emotions_data.items():
            if not isinstance(einfo, dict): continue
            primary = (einfo.get('metadata', {}) or {}).get('primary_category')
            if isinstance(primary, str) and primary in {'희', '노', '애', '락'}:
                m[eid] = primary
            for sid, sinfo in (einfo.get('sub_emotions', {}) or {}).items():
                if not isinstance(sinfo, dict): continue
                p = (sinfo.get('metadata', {}) or {}).get('primary_category', primary)
                if isinstance(p, str) and p in {'희', '노', '애', '락'}:
                    m[sid] = p
        return m

    # -------- 컨텍스트 데이터(정규식 컴파일) --------
    def _build_dynamic_context_data(self) -> Dict[str, Dict[str, Any]]:
        results: Dict[str, Dict[str, Any]] = {}
        def _safe_compile(p: str) -> Optional[re.Pattern]:
            if not isinstance(p, str) or not p.strip(): return None
            try:
                return re.compile(p, re.IGNORECASE)
            except re.error:
                # 패턴이 평문일 경우 escape하여 부분일치 허용
                return re.compile(re.escape(p), re.IGNORECASE)

        for emotion_id, emotion_info in self.emotions_data.items():
            if emotion_id == 'emotion_sequence' or not isinstance(emotion_info, dict):
                continue
            ctx = emotion_info.get('context_patterns', {}) or {}
            expanded = ctx.get('expanded_patterns', []) or []
            sudden = ctx.get('sudden_synonyms', []) or []
            gradual = ctx.get('gradual_synonyms', []) or []
            compiled = [c for p in expanded if (c := _safe_compile(p))]

            results[emotion_id] = {
                'compiled': compiled,
                'sudden': [self._normalize(s) for s in sudden if isinstance(s, str)],
                'gradual': [self._normalize(g) for g in gradual if isinstance(g, str)]
            }

            for sub_id, sub_info in (emotion_info.get('sub_emotions', {}) or {}).items():
                if not isinstance(sub_info, dict): continue
                sctx = sub_info.get('context_patterns', {}) or {}
                expanded = sctx.get('expanded_patterns', []) or []
                sudden = sctx.get('sudden_synonyms', []) or []
                gradual = sctx.get('gradual_synonyms', []) or []
                compiled = [c for p in expanded if (c := _safe_compile(p))]
                results[sub_id] = {
                    'compiled': compiled,
                    'sudden': [self._normalize(s) for s in sudden if isinstance(s, str)],
                    'gradual': [self._normalize(g) for g in gradual if isinstance(g, str)]
                }
        return results

    # -------- 특수 강도 토큰 --------
    def _build_dynamic_special_tokens(self) -> Dict[str, List[str]]:
        result = {"explosive": [], "gradual": []}
        global_meta = self.emotions_data.get("metadata", {}) or {}
        special_data = global_meta.get("special_intensity", {}) or {}
        if isinstance(special_data, dict):
            result["explosive"].extend([self._normalize(x) for x in (special_data.get("explosive_keywords", []) or []) if isinstance(x, str)])
            result["gradual"].extend([self._normalize(x) for x in (special_data.get("gradual_keywords", []) or []) if isinstance(x, str)])
        for _, emotion_info in self.emotions_data.items():
            if not isinstance(emotion_info, dict): continue
            meta = emotion_info.get("metadata", {}) or {}
            sp_data = meta.get("special_intensity", {}) or {}
            result["explosive"].extend([self._normalize(x) for x in (sp_data.get("explosive_keywords", []) or []) if isinstance(x, str)])
            result["gradual"].extend([self._normalize(x) for x in (sp_data.get("gradual_keywords", []) or []) if isinstance(x, str)])
            for sub_info in (emotion_info.get("sub_emotions", {}) or {}).values():
                if not isinstance(sub_info, dict): continue
                sub_meta = sub_info.get("metadata", {}) or {}
                ss = sub_meta.get("special_intensity", {}) or {}
                result["explosive"].extend([self._normalize(x) for x in (ss.get("explosive_keywords", []) or []) if isinstance(x, str)])
                result["gradual"].extend([self._normalize(x) for x in (ss.get("gradual_keywords", []) or []) if isinstance(x, str)])
        result["explosive"] = sorted(set(result["explosive"]))
        result["gradual"] = sorted(set(result["gradual"]))
        return result

    # -------- 컨텍스트 분석 --------
    def _analyze_context(self, text: str) -> dict:
        context_scores = {cat: 0.0 for cat in ['희', '노', '애', '락']}
        norm = self._normalize(text)

        for emotion_id, ctx_info in self._compiled_context.items():
            compiled = ctx_info.get('compiled', []) or []
            sudden_list = ctx_info.get('sudden', []) or []
            gradual_list = ctx_info.get('gradual', []) or []
            matched = any(p.search(norm) for p in compiled)

            if matched:
                base = 1.5
                if any(s in norm for s in sudden_list):
                    base += 0.3
                elif any(g in norm for g in gradual_list):
                    base += 0.1

                # emotion_id가 1차 카테고리면 바로 가산, 아니면 매핑
                target = emotion_id if emotion_id in {'희', '노', '애', '락'} else self._primary_by_emotion.get(emotion_id)
                if target in context_scores:
                    context_scores[target] += base
        return context_scores

    # -------- 강도 분석 --------
    def _analyze_intensity(self, text: str) -> dict:
        intensity_scores = {cat: 0.0 for cat in ['희', '노', '애', '락']}
        tokens = self.tokenize(text)
        token_set = set(tokens)
        weights = {'high': 0.3, 'medium': 0.2, 'low': 0.1}

        # (1) 감정별 강도어 → 해당 1차 카테고리에만 가산 (정밀)
        for eid, levels in self._per_emotion_intensity.items():
            target = self._primary_by_emotion.get(eid)
            if target not in intensity_scores: continue
            for level, words in levels.items():
                if not words: continue
                cnt = sum(1 for w in words if w in token_set)
                if cnt:
                    intensity_scores[target] += weights.get(level, 0.1) * cnt

        # (2) 전역 강도어 → 보정치(약하게) 전체 가산 (기존 동작 유지)
        for level, words in self._global_intensity.items():
            cnt = sum(1 for w in words if w in token_set)
            if cnt:
                intensity_scores = {k: v + weights.get(level, 0.1) * cnt * 0.3 for k, v in intensity_scores.items()}

        # (3) 특수 토큰
        if any(t in token_set for t in (self._special_tokens.get("explosive") or [])):
            for em in intensity_scores: intensity_scores[em] += 0.1
        if any(t in token_set for t in (self._special_tokens.get("gradual") or [])):
            for em in intensity_scores: intensity_scores[em] += 0.05

        return {em: round(val, 3) for em, val in intensity_scores.items()}

    # -------- 메인: 감정 점수화 --------
    def analyze_emotion(self, text: str) -> Dict[str, Any]:
        tokens: List[str] = self.tokenize(text or "")
        if not tokens: return {'희': 0.0, '노': 0.0, '애': 0.0, '락': 0.0}
        token_set = set(tokens)
        bigrams = self._make_ngrams(tokens, 2)
        trigrams = self._make_ngrams(tokens, 3)
        text_norm = " ".join(tokens)
        # 모든 라벨 0.0으로 초기화
        sub_emotion_scores: Dict[str, float] = {}
        for emotion_id, info in (self.emotions_data or {}).items():
            if emotion_id == "emotion_sequence" or not isinstance(info, dict):
                continue
            sub_emotion_scores[emotion_id] = 0.0
        # 키워드/동의어/콤보 스코어링
        for emotion_id, kw_data in (self.emotion_keywords or {}).items():
            if emotion_id not in sub_emotion_scores or not isinstance(kw_data, dict):
                continue
            kw_weights: Dict[str, float] = kw_data.get("kw_weights", {}) or {}
            combos: List[Tuple[Tuple[str, ...], float]] = kw_data.get("combos", []) or []
            synonyms_map: Dict[str, List[str]] = kw_data.get("synonyms", {}) or {}
            # 1) 단일 키워드
            for kw, w in kw_weights.items():
                if kw in token_set:
                    sub_emotion_scores[emotion_id] += w
                else:
                    # 한국어 활용형 등 부분일치 보조(가중 0.6)
                    if re.search(rf"(?:(?<=\s)|^){re.escape(kw)}(?=\s|$|[.,!?;:])", text_norm):
                        sub_emotion_scores[emotion_id] += w * 0.6
                # 2) 동의어(약하게 0.8배)
                syns = synonyms_map.get(kw, [])
                if syns:
                    if any(s in token_set for s in syns):
                        sub_emotion_scores[emotion_id] += w * 0.8
                    else:
                        for s in syns:
                            if re.search(rf"(?:(?<=\s)|^){re.escape(s)}(?=\s|$|[.,!?;:])", text_norm):
                                sub_emotion_scores[emotion_id] += w * 0.5
                                break
            # 3) 콤보(모든 토큰 등장 시 가산)
            for words_tuple, combo_weight in combos:
                n = len(words_tuple)
                if n == 2 and tuple(words_tuple) in bigrams:
                    sub_emotion_scores[emotion_id] += combo_weight
                elif n == 3 and tuple(words_tuple) in trigrams:
                    sub_emotion_scores[emotion_id] += combo_weight
                elif all(w in token_set for w in words_tuple):
                    sub_emotion_scores[emotion_id] += combo_weight
                else:
                    # 정규식으로 느슨하게: 단어 경계 기준 순서 매칭
                    pat = self._combo_regex_cache.get(words_tuple)
                    if pat is None:
                        esc = [re.escape(w) for w in words_tuple]
                        pat = re.compile(r"(?:^|\s)" + r"\s+".join(esc) + r"(?:\s|$|[.,!?;:])")
                        self._combo_regex_cache[words_tuple] = pat
                    if pat.search(text_norm):
                        sub_emotion_scores[emotion_id] += combo_weight * 0.6
        # 문맥 가산
        context_scores = self._analyze_context(text) or {}
        for e_id, c_score in (context_scores.items() if isinstance(context_scores, dict) else []):
            if e_id in sub_emotion_scores:
                sub_emotion_scores[e_id] += float(c_score)
            else:
                # 맵핑
                target = self._primary_by_emotion.get(e_id)
                if target and target in sub_emotion_scores:
                    sub_emotion_scores[target] += float(c_score)
        # 강도 스케일링
        intensity_scores = self._analyze_intensity(text) or {}
        for e_id, i_score in (intensity_scores.items() if isinstance(intensity_scores, dict) else []):
            if e_id in sub_emotion_scores:
                i = float(i_score)
                if i > 0.0:
                    sub_emotion_scores[e_id] *= (1.0 + i)
        # 1차/하위 분리
        primary_ids = {'희', '노', '애', '락'}
        primary_emotion_temp: Dict[str, float] = {}
        sub_emotion_temp: Dict[str, float] = {}
        for e_id, sc in sub_emotion_scores.items():
            if e_id in primary_ids:
                primary_emotion_temp[e_id] = float(sc)
            else:
                sub_emotion_temp[e_id] = float(sc)
        base_min_score = 0.15

        def get_min_score_for_emotion(e_id: str) -> float:
            e_info = self.emotions_data.get(e_id, {}) if isinstance(self.emotions_data, dict) else {}
            ml_meta = e_info.get('ml_training_metadata', {}) if isinstance(e_info, dict) else {}
            conf_thresholds = ml_meta.get('confidence_thresholds', {}) if isinstance(ml_meta, dict) else {}
            complexity = (e_info.get('metadata', {}) or {}).get('emotion_complexity', 'basic')
            try:
                dynamic_thresh = float(conf_thresholds.get(complexity, base_min_score))
            except Exception:
                dynamic_thresh = base_min_score
            # [버그 픽스] 기본값보다 낮아지지 않도록
            return max(dynamic_thresh, base_min_score)
        # 임계치 필터
        filtered_primary = {pid: v for pid, v in primary_emotion_temp.items() if v >= get_min_score_for_emotion(pid)}
        filtered_sub = {sid: v for sid, v in sub_emotion_temp.items() if v >= get_min_score_for_emotion(sid)}
        # Top-N
        top_n = 3
        def topn(d: Dict[str, float]) -> Dict[str, float]:
            return dict(sorted(d.items(), key=lambda kv: kv[1], reverse=True)[:top_n])
        filtered_primary = topn(filtered_primary)
        filtered_sub = topn(filtered_sub)
        # 최종 점수 (항상 float)
        final_scores: Dict[str, Any] = {'희': 0.0, '노': 0.0, '애': 0.0, '락': 0.0}
        for pid, sc in filtered_primary.items():
            final_scores[pid] = float(sc)
        for sid, sc in filtered_sub.items():
            primary_cat = self._primary_by_emotion.get(sid)
            if primary_cat in final_scores:
                final_scores[primary_cat] = float(final_scores[primary_cat]) + float(sc)
        # 정규화
        total = sum(v for k, v in final_scores.items() if k in primary_ids and isinstance(v, (int, float)))
        if total > 0:
            for k in list(primary_ids):
                final_scores[k] = round(float(final_scores[k]) / total, 3)
        # 후보 감정(메타)
        intras = list(filtered_primary.keys()) + list(filtered_sub.keys())
        if len(intras) > 1:
            final_scores['_intrasentence_candidates'] = intras
        return final_scores


# =============================================================================
# MemoryMonitor
# =============================================================================
class MemoryMonitor:
    def check_memory(self) -> bool:
        return True


# =============================================================================
# EmotionValidatorExtended
# =============================================================================
class EmotionValidatorExtended:
    """
    - validate_emotion_structure(emotions_data, autofix=False) -> (is_valid, issues)
      * 기존 반환값 유지. autofix=True이면 가능한 범위에서 제자리 보정(in-place) 수행.
    - take_schema_snapshot(emotions_data) -> Dict[str, List[str]]
      * 스냅샷 테스트: 노드별 핵심 키 집합을 path 기반으로 요약.
    """

    def __init__(self):
        self.primary_categories = ['희', '노', '애', '락']
        self.allowed_complexity = {'basic', 'complex', 'subtle'}
        self.allowed_intensity_levels = {'low', 'medium', 'high'}

    # ---------------- public ----------------
    def validate_emotion_structure(
        self,
        emotions_data: Dict[str, Any],
        autofix: bool = False
    ) -> Tuple[bool, List[Dict[str, str]]]:
        issues: List[Dict[str, str]] = []

        if not isinstance(emotions_data, dict):
            self._add_issue(issues, 'ERROR', 'root', 'emotions_data가 dict 형식이 아닙니다.')
            return False, issues

        # 최상위 메타 컨테이너 허용(선택)
        if 'metadata' in emotions_data and not isinstance(emotions_data['metadata'], dict):
            self._add_issue(issues, 'WARNING', 'metadata', "'metadata'는 dict 이어야 합니다. 무시합니다.")
            if autofix:
                emotions_data['metadata'] = {}

        # 4대 대표 감정 확인 (+필요 시 생성)
        for cat in self.primary_categories:
            if cat not in emotions_data:
                self._add_issue(issues, 'ERROR', 'root', f"필수 대표감정 '{cat}'가 누락되었습니다.")
                if autofix:
                    emotions_data[cat] = self._build_default_node(primary_cat=cat)
                    self._add_issue(issues, 'INFO', cat, f"누락된 '{cat}' 노드를 기본 스켈레톤으로 생성했습니다.")
            node = emotions_data.get(cat)
            self._validate_node(node, cat, issues, top_level=True, autofix=autofix, inherited_primary=cat)

        is_valid = not any(i['level'] == 'ERROR' for i in issues)
        return is_valid, issues

    def take_schema_snapshot(self, emotions_data: Dict[str, Any]) -> Dict[str, List[str]]:
        """노드별 핵심 키 목록 스냅샷(스키마 회귀 테스트에 사용)."""
        snapshot: Dict[str, List[str]] = {}

        def walk(node: Any, path: str):
            if not isinstance(node, dict):
                snapshot[path] = ['<non-dict>']
                return
            keys = []
            for k in ('metadata', 'emotion_profile', 'sub_emotions',
                      'sentiment_analysis', 'context_patterns',
                      'linguistic_patterns', 'ml_training_metadata'):
                if k in node:
                    keys.append(k)
            snapshot[path] = sorted(keys)
            subs = node.get('sub_emotions', {})
            if isinstance(subs, dict):
                for sname, sdata in subs.items():
                    walk(sdata, f"{path}.sub_emotions.{sname}")

        for cat in self.primary_categories:
            if isinstance(emotions_data.get(cat), dict):
                walk(emotions_data[cat], cat)
        return snapshot

    # ---------------- core validation ----------------
    def _validate_node(
        self,
        node_data: Any,
        path: str,
        issues: List[Dict[str, str]],
        *,
        top_level: bool,
        autofix: bool,
        inherited_primary: Optional[str]
    ) -> None:
        if not isinstance(node_data, dict):
            self._add_issue(issues, 'ERROR', path, '감정 데이터가 dict 형식이 아닙니다.')
            if autofix:
                # 비정상 노드를 기본 스켈레톤으로 치환
                new_node = self._build_default_node(primary_cat=inherited_primary)
                # 상위 dict에 반영할 수 없으므로 경고만 남김(호출측에서 이미 dict 참조로 전달됨을 가정)
                try:
                    # 부모 dict가 수정 가능한 경우를 대비하여 최소한의 복구 시도
                    node_data = new_node  # 지역치환(참조 미연결 시 상위에서 이미 dict로 존재)
                except Exception:
                    pass
            return

        # --- metadata ---
        if 'metadata' not in node_data or not isinstance(node_data.get('metadata'), dict):
            self._add_issue(issues, 'ERROR', path, "'metadata' 키가 없거나 dict가 아닙니다.")
            if autofix:
                node_data['metadata'] = {}
        meta = node_data.get('metadata', {})

        # primary_category 보정
        if 'primary_category' not in meta or meta.get('primary_category') not in self.primary_categories:
            if inherited_primary in self.primary_categories:
                if autofix:
                    meta['primary_category'] = inherited_primary
                    self._add_issue(issues, 'INFO', path, f"'primary_category'가 없거나 잘못됨 -> '{inherited_primary}'로 보정.")
            else:
                self._add_issue(issues, 'WARNING', path, "'primary_category'가 유효하지 않습니다(희/노/애/락).")

        # emotion_complexity 보정
        ec = meta.get('emotion_complexity')
        if ec is not None and ec not in self.allowed_complexity:
            self._add_issue(issues, 'WARNING', path, f"'emotion_complexity' 값이 비표준입니다: {ec}.")
        if ec is None and autofix:
            meta['emotion_complexity'] = 'basic'

        # --- emotion_profile ---
        if not top_level and 'emotion_profile' not in node_data:
            self._add_issue(issues, 'WARNING', path, "'emotion_profile' 키가 없습니다.")
            if autofix:
                node_data['emotion_profile'] = {}
        self._ensure_emotion_profile_defaults(node_data)

        # core_keywords 타입 보정
        prof = node_data.get('emotion_profile', {})
        if not isinstance(prof.get('core_keywords'), list):
            self._add_issue(issues, 'WARNING', f"{path}.emotion_profile.core_keywords",
                            "list[str] 이어야 합니다.")
            if autofix:
                prof['core_keywords'] = []

        # intensity_levels 구조/타입 보정
        ilv = prof.get('intensity_levels', {})
        if not isinstance(ilv, dict):
            self._add_issue(issues, 'WARNING', f"{path}.emotion_profile.intensity_levels",
                            "dict 이어야 합니다.")
            if autofix:
                prof['intensity_levels'] = {'low': [], 'medium': [], 'high': []}
        else:
            for lv in self.allowed_intensity_levels:
                if lv not in ilv or not isinstance(ilv[lv], list):
                    self._add_issue(issues, 'WARNING', f"{path}.emotion_profile.intensity_levels.{lv}",
                                    "list[str] 이어야 합니다.")
                    if autofix:
                        ilv[lv] = []

            # 선택 필드들(예시/수식어) 타입 보정
            iex = ilv.get('intensity_examples', {})
            if iex is not None and not isinstance(iex, dict):
                self._add_issue(issues, 'WARNING', f"{path}.emotion_profile.intensity_levels.intensity_examples",
                                "dict 이어야 합니다.")
                if autofix:
                    ilv['intensity_examples'] = {}
            imods = ilv.get('intensity_modifiers', {})
            if imods is not None and not isinstance(imods, dict):
                self._add_issue(issues, 'WARNING', f"{path}.emotion_profile.intensity_levels.intensity_modifiers",
                                "dict 이어야 합니다.")
                if autofix:
                    ilv['intensity_modifiers'] = {}

        # 선택: intensity_data 전역/서브 감정에서 사용
        if 'intensity_data' in prof and not isinstance(prof['intensity_data'], dict):
            self._add_issue(issues, 'WARNING', f"{path}.emotion_profile.intensity_data", "dict 이어야 합니다.")
            if autofix:
                prof['intensity_data'] = {}

        # --- sentiment_analysis / intensity_modifiers ---
        sa = node_data.get('sentiment_analysis')
        if sa is None:
            if autofix:
                node_data['sentiment_analysis'] = {'intensity_modifiers': {'amplifiers': [], 'diminishers': []}}
                self._add_issue(issues, 'INFO', path, "'sentiment_analysis'가 없어 기본 구조를 생성했습니다.")
        else:
            if not isinstance(sa, dict):
                self._add_issue(issues, 'WARNING', f"{path}.sentiment_analysis", "dict 이어야 합니다.")
                if autofix:
                    node_data['sentiment_analysis'] = {'intensity_modifiers': {'amplifiers': [], 'diminishers': []}}
            else:
                im = sa.get('intensity_modifiers')
                if im is None or not isinstance(im, dict):
                    self._add_issue(issues, 'WARNING', f"{path}.sentiment_analysis.intensity_modifiers", "dict 이어야 합니다.")
                    if autofix:
                        sa['intensity_modifiers'] = {'amplifiers': [], 'diminishers': []}
                else:
                    for k in ('amplifiers', 'diminishers'):
                        if k not in im or not isinstance(im[k], list):
                            self._add_issue(issues, 'WARNING', f"{path}.sentiment_analysis.intensity_modifiers.{k}",
                                            "list[str] 이어야 합니다.")
                            if autofix:
                                im[k] = []

        # --- context_patterns ---
        cp = node_data.get('context_patterns')
        if cp is not None and not isinstance(cp, dict):
            self._add_issue(issues, 'WARNING', f"{path}.context_patterns", "dict 이어야 합니다.")
            if autofix:
                node_data['context_patterns'] = {}
                cp = node_data['context_patterns']
        if isinstance(cp, dict):
            for k in ('expanded_patterns', 'sudden_synonyms', 'gradual_synonyms'):
                if k in cp and not isinstance(cp[k], list):
                    self._add_issue(issues, 'WARNING', f"{path}.context_patterns.{k}", "list[str] 이어야 합니다.")
                    if autofix:
                        cp[k] = []
            # situations 구조
            sits = cp.get('situations')
            if sits is not None and not isinstance(sits, dict):
                self._add_issue(issues, 'WARNING', f"{path}.context_patterns.situations", "dict 이어야 합니다.")
                if autofix:
                    cp['situations'] = {}
            elif isinstance(sits, dict):
                for sname, s in sits.items():
                    if not isinstance(s, dict):
                        self._add_issue(issues, 'WARNING', f"{path}.context_patterns.situations.{sname}", "dict 이어야 합니다.")
                        if autofix:
                            sits[sname] = {'keywords': [], 'examples': [], 'intensity': 'medium'}
                        continue
                    if 'keywords' in s and not isinstance(s['keywords'], list):
                        self._add_issue(issues, 'WARNING', f"{path}.context_patterns.situations.{sname}.keywords", "list[str] 필요.")
                        if autofix: s['keywords'] = []
                    if 'examples' in s and not isinstance(s['examples'], list):
                        self._add_issue(issues, 'WARNING', f"{path}.context_patterns.situations.{sname}.examples", "list[str] 필요.")
                        if autofix: s['examples'] = []
                    if 'intensity' in s and s['intensity'] not in self.allowed_intensity_levels:
                        self._add_issue(issues, 'WARNING', f"{path}.context_patterns.situations.{sname}.intensity",
                                        "low/medium/high 중 하나여야 합니다.")
                        if autofix: s['intensity'] = 'medium'

        # --- linguistic_patterns ---
        lp = node_data.get('linguistic_patterns')
        if lp is not None and not isinstance(lp, dict):
            self._add_issue(issues, 'WARNING', f"{path}.linguistic_patterns", "dict 이어야 합니다.")
            if autofix:
                node_data['linguistic_patterns'] = {'key_phrases': []}
                lp = node_data['linguistic_patterns']
        if isinstance(lp, dict):
            kps = lp.get('key_phrases')
            if kps is not None and not isinstance(kps, list):
                self._add_issue(issues, 'WARNING', f"{path}.linguistic_patterns.key_phrases", "list[dict] 이어야 합니다.")
                if autofix:
                    lp['key_phrases'] = []
            elif isinstance(kps, list):
                for i, kp in enumerate(kps):
                    if not isinstance(kp, dict):
                        self._add_issue(issues, 'WARNING', f"{path}.linguistic_patterns.key_phrases[{i}]", "dict 필요.")
                        if autofix: kps[i] = {'pattern': '', 'weight': 1.0, 'synonyms': {}}
                        continue
                    # pattern
                    if 'pattern' not in kp or not isinstance(kp['pattern'], str):
                        self._add_issue(issues, 'WARNING', f"{path}.linguistic_patterns.key_phrases[{i}].pattern",
                                        "str 필요.")
                        if autofix: kp['pattern'] = ''
                    # weight
                    if 'weight' in kp and not isinstance(kp['weight'], (int, float)):
                        self._add_issue(issues, 'WARNING', f"{path}.linguistic_patterns.key_phrases[{i}].weight",
                                        "float 필요.")
                        if autofix: kp['weight'] = 1.0
                    # synonyms
                    if 'synonyms' in kp and not isinstance(kp['synonyms'], dict):
                        self._add_issue(issues, 'WARNING', f"{path}.linguistic_patterns.key_phrases[{i}].synonyms",
                                        "dict[str, list[str]] 필요.")
                        if autofix: kp['synonyms'] = {}

        # --- ML training metadata(+표준 모듈/임계) ---
        self._ensure_ml_training_metadata(node_data)
        self._clamp_ml_meta(node_data, issues, path, autofix=autofix)

        # --- sub_emotions ---
        if 'sub_emotions' in node_data:
            subs = node_data['sub_emotions']
            if not isinstance(subs, dict):
                self._add_issue(issues, 'ERROR', f"{path}.sub_emotions", "'sub_emotions'가 dict 형식이 아닙니다.")
            else:
                if len(subs) != 30:
                    self._add_issue(issues, 'WARNING', f"{path}.sub_emotions",
                                    f"세부감정이 30개가 아닙니다. (현재: {len(subs)})")
                for sub_name, sub_data in subs.items():
                    # 서브에도 primary_category 상속 보정
                    sub_primary = (node_data.get('metadata') or {}).get('primary_category', inherited_primary)
                    self._validate_node(sub_data, f"{path}.sub_emotions.{sub_name}",
                                        issues, top_level=False, autofix=autofix, inherited_primary=sub_primary)

    # ---------------- defaults / fixes ----------------
    def _ensure_ml_training_metadata(self, emotion_info: Dict[str, Any]) -> None:
        if 'ml_training_metadata' not in emotion_info or not isinstance(emotion_info['ml_training_metadata'], dict):
            emotion_info['ml_training_metadata'] = {}
        ml_meta = emotion_info['ml_training_metadata']

        if 'analysis_modules' not in ml_meta or not isinstance(ml_meta.get('analysis_modules'), dict):
            ml_meta['analysis_modules'] = {}
        analysis_modules = ml_meta['analysis_modules']

        analysis_modules.setdefault('progression_analyzer', {
            'enabled': True,
            'progression_trigger_intensity': 0.7,
            'progression_stages': ['trigger', 'development', 'peak', 'aftermath'],
            'stage_weights': {'trigger': 0.7, 'development': 0.8, 'peak': 1.0, 'aftermath': 0.6},
            'stage_transitions': {'trigger_to_development': 0.3, 'development_to_peak': 0.4, 'peak_to_aftermath': 0.3}
        })
        analysis_modules.setdefault('transition_analyzer', {
            'enabled': True,
            'min_transition_score': 0.6,
            'smoothing_factor': 0.3,
            'transition_types': ['direct', 'gradual', 'sudden', 'cyclic'],
            'transition_weights': {'direct': 1.0, 'gradual': 0.7, 'sudden': 0.5, 'cyclic': 0.3},
            'transition_thresholds': {'high': 0.8, 'medium': 0.5, 'low': 0.3}
        })
        ml_meta.setdefault('context_requirements', {'minimum_length': 0, 'maximum_length': 999999, 'required_keywords': {}})
        ml_meta.setdefault('confidence_thresholds', {'basic': 0.7, 'complex': 0.8, 'subtle': 0.9})
        ml_meta.setdefault('pattern_matching', {'basic': 0.65, 'complex': 0.75, 'subtle': 0.85})
        ml_meta.setdefault('emotion_classification', {'primary': 0.7, 'intensity': 0.6, 'transition': 0.5})

        standard_modules = {
            'pattern_extractor': {'enabled': True, 'threshold': 0.75},
            'context_extractor': {'enabled': True, 'cache_size': 1000, 'memory_limit': 512},
            'intensity_analyzer': {'enabled': True, 'minimum_confidence': 0.6, 'pattern_match_threshold': 0.7},
            'linguistic_matcher': {'enabled': True, 'lang_detection': True, 'match_strategy': 'exact'},
            'intrasentence_analyzer': {'enabled': True, 'min_intrasentence_score': 0.25, 'max_intrasentence_relations': 5}
        }
        for module_name, default_config in standard_modules.items():
            analysis_modules.setdefault(module_name, default_config)

    def _ensure_emotion_profile_defaults(self, emotion_info: Dict[str, Any]) -> None:
        if 'emotion_profile' not in emotion_info or not isinstance(emotion_info.get('emotion_profile'), dict):
            emotion_info['emotion_profile'] = {}
        profile = emotion_info['emotion_profile']
        profile.setdefault('core_keywords', [])
        profile.setdefault('related_emotions', {})
        profile.setdefault('intensity_levels', {'low': [], 'medium': [], 'high': []})
        # 확장 필드가 존재한다면 최소 구조 보장
        if 'intensity_levels' in profile and isinstance(profile['intensity_levels'], dict):
            profile['intensity_levels'].setdefault('intensity_examples', {})
            profile['intensity_levels'].setdefault('intensity_modifiers', {})
        # 선택 필드
        profile.setdefault('intensity_data', {})  # 전역/서브 강도 어휘를 사용할 때

    def _clamp_ml_meta(self, node_data: Dict[str, Any], issues: List[Dict[str, str]], path: str, *, autofix: bool):
        """0~1 범위의 수치들 클램프 및 stage_weights 정규화."""
        ml = node_data.get('ml_training_metadata', {})
        if not isinstance(ml, dict):
            return
        def clamp01(x):
            try:
                v = float(x)
            except Exception:
                return None
            return max(0.0, min(1.0, v))

        # confidence_thresholds / pattern_matching / emotion_classification
        for key in ('confidence_thresholds', 'pattern_matching', 'emotion_classification'):
            sub = ml.get(key)
            if isinstance(sub, dict):
                for k, v in list(sub.items()):
                    nv = clamp01(v)
                    if nv is None:
                        self._add_issue(issues, 'WARNING', f"{path}.ml_training_metadata.{key}.{k}",
                                        "숫자여야 합니다.")
                        if autofix: sub[k] = 0.0
                    elif nv != v:
                        self._add_issue(issues, 'INFO', f"{path}.ml_training_metadata.{key}.{k}",
                                        f"값을 0~1 범위로 보정했습니다({v}→{nv}).")
                        if autofix: sub[k] = nv

        # analysis_modules 내부
        am = (ml.get('analysis_modules') or {})
        if isinstance(am, dict):
            pa = am.get('progression_analyzer')
            if isinstance(pa, dict):
                if 'progression_trigger_intensity' in pa:
                    nv = clamp01(pa.get('progression_trigger_intensity'))
                    if nv is not None and nv != pa.get('progression_trigger_intensity'):
                        self._add_issue(issues, 'INFO', f"{path}.ml_training_metadata.analysis_modules.progression_analyzer.progression_trigger_intensity",
                                        "0~1로 보정했습니다.")
                        if autofix: pa['progression_trigger_intensity'] = nv
                # stage_weights normalize(합이 0이 아니면 1로 정규화)
                sw = pa.get('stage_weights')
                if isinstance(sw, dict) and sw:
                    total = 0.0
                    ok = True
                    for k, v in sw.items():
                        try: total += float(v)
                        except Exception:
                            ok = False
                            break
                    if ok and total > 0:
                        if abs(total - 1.0) > 1e-6:
                            if autofix:
                                for k in sw:
                                    sw[k] = float(sw[k]) / total
                            self._add_issue(issues, 'INFO', f"{path}.ml_training_metadata.analysis_modules.progression_analyzer.stage_weights",
                                            f"가중치 합({round(total,3)})을 1로 정규화했습니다.")
            ta = am.get('transition_analyzer')
            if isinstance(ta, dict):
                for key in ('min_transition_score', 'smoothing_factor'):
                    if key in ta:
                        nv = clamp01(ta.get(key))
                        if nv is not None and nv != ta.get(key):
                            self._add_issue(issues, 'INFO', f"{path}.ml_training_metadata.analysis_modules.transition_analyzer.{key}",
                                            "0~1로 보정했습니다.")
                            if autofix: ta[key] = nv
                tw = ta.get('transition_weights')
                if isinstance(tw, dict):
                    for k, v in list(tw.items()):
                        nv = clamp01(v)
                        if nv is None:
                            self._add_issue(issues, 'WARNING', f"{path}.ml_training_metadata.analysis_modules.transition_analyzer.transition_weights.{k}",
                                            "숫자여야 합니다.")
                            if autofix: tw[k] = 0.0
                        elif nv != v:
                            self._add_issue(issues, 'INFO', f"{path}.ml_training_metadata.analysis_modules.transition_analyzer.transition_weights.{k}",
                                            "0~1로 보정했습니다.")
                            if autofix: tw[k] = nv

    # ---------------- utils ----------------
    def _add_issue(self, issues: List[Dict[str, str]], level: str, path: str, message: str):
        issues.append({'level': level, 'path': path, 'message': message})

    def _build_default_node(self, *, primary_cat: Optional[str]) -> Dict[str, Any]:
        return {
            'metadata': {
                'primary_category': primary_cat if primary_cat in self.primary_categories else None,
                'emotion_complexity': 'basic'
            },
            'emotion_profile': {
                'core_keywords': [],
                'related_emotions': {},
                'intensity_levels': {
                    'low': [], 'medium': [], 'high': [],
                    'intensity_examples': {},
                    'intensity_modifiers': {}
                },
                'intensity_data': {}
            },
            'sentiment_analysis': {
                'intensity_modifiers': {'amplifiers': [], 'diminishers': []}
            },
            'context_patterns': {
                'expanded_patterns': [],
                'sudden_synonyms': [],
                'gradual_synonyms': [],
                'situations': {}
            },
            'linguistic_patterns': {'key_phrases': []},
            'ml_training_metadata': {}
        }


# =============================================================================
# EmotionProgressionRelationshipAnalyzer
# =============================================================================
class EmotionProgressionRelationshipAnalyzer:
    """ 감정 진행과 관계 분석을 담당하는 클래스 """

    def __init__(self, emotions_data: Dict[str, Any]):
        self.emotions_data = emotions_data or {}
        self.text_analyzer = TextAnalyzer(self.emotions_data)

        # (1) 스테이지 셋업 (라벨링 우선)
        self.progression_stages = self._load_progression_stages_from_labeling(self.emotions_data)

        # (2) 세부감정 복합도 가중치(유지)
        self.sub_emotion_weights = {'subtle': 1.2, 'basic': 1.0, 'complex': 0.8}

        # (3) 라벨링 기반 모듈 설정 캐시(스코어 가중치/임계 등)
        self._progression_cfg_cache: Dict[str, Dict[str, Any]] = {}
        self._stage_pattern_cache: Dict[str, List[re.Pattern]] = {}
        self._context_pattern_cache: Dict[str, Dict[str, Any]] = {}

        logger.info("[EmotionProgressionRelationshipAnalyzer] 초기화")

    # ---------- 라벨링/메타 로드 ----------
    def _load_progression_stages_from_labeling(self, emotions_data: Dict[str, Any]) -> List[str]:
        """ 라벨링 뼈대에서 스테이지를 동적으로 로드. 없으면 기본값 사용. """
        try:
            global_meta = emotions_data.get('metadata', {}) if isinstance(emotions_data, dict) else {}
            # 글로벌에 정의된 경우 최우선
            available = global_meta.get('available_stages')
            if isinstance(available, list) and available:
                return [str(s) for s in available]
        except Exception as e:
            logger.warning(f"[스테이지 로드] 라벨링 global metadata 파싱 오류: {e}")

        # 개별 emotion 노드의 progression_analyzer에 정의가 있을 수 있음 → 합집합
        stages: List[str] = []
        try:
            for _, info in (emotions_data or {}).items():
                if not isinstance(info, dict):
                    continue
                ml = info.get('ml_training_metadata', {})
                am = (ml.get('analysis_modules') or {})
                pa = am.get('progression_analyzer') or {}
                st = pa.get('progression_stages')
                if isinstance(st, list) and st:
                    for s in st:
                        s = str(s)
                        if s not in stages:
                            stages.append(s)
        except Exception:
            pass

        # 최종 fallback
        return stages if stages else ['trigger', 'development', 'peak', 'aftermath']

    def _get_progression_cfg(self, emotion_info: Dict[str, Any]) -> Dict[str, Any]:
        """ emotion별 progression/transition 설정(라벨링 메타) 불러오기 + 기본값 보정 """
        key = id(emotion_info)
        if key in self._progression_cfg_cache:
            return self._progression_cfg_cache[key]

        ml = emotion_info.get('ml_training_metadata', {}) if isinstance(emotion_info, dict) else {}
        am = ml.get('analysis_modules', {}) if isinstance(ml, dict) else {}
        pa = am.get('progression_analyzer', {}) if isinstance(am, dict) else {}
        ta = am.get('transition_analyzer', {}) if isinstance(am, dict) else {}

        # progression 가중치(스테이지별 기본 점수)
        stage_weights = pa.get('stage_weights', {
            'trigger': 0.7, 'development': 0.8, 'peak': 1.0, 'aftermath': 0.6
        })
        # 최종 aggregation 가중치(전체 진행 점수에서의 비중)
        agg_weights = {
            'trigger': 0.2, 'development': 0.3, 'peak': 0.3, 'aftermath': 0.2
        }

        # transition 설정
        transition_weights = ta.get('transition_weights', {
            'direct': 1.0, 'gradual': 0.7, 'sudden': 0.5, 'cyclic': 0.3
        })
        thresholds = ta.get('transition_thresholds', {'high': 0.8, 'medium': 0.5, 'low': 0.3})
        smoothing = float(ta.get('smoothing_factor', 0.3))

        cfg = {
            'stage_base_scores': stage_weights,
            'agg_weights': agg_weights,
            'transition_weights': transition_weights,
            'transition_thresholds': thresholds,
            'smoothing_factor': smoothing
        }
        self._progression_cfg_cache[key] = cfg
        return cfg

    # ---------- 최상위 분석 ----------
    def analyze_stage_relationships(self, text: str, emotions_data: Dict[str, Any]) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        sentences = self._split_text_into_sentences(text or "")
        if not sentences:
            return results

        emotion_stages = list(self.progression_stages)  # 라벨링/기본 혼합 결과
        for emotion_id, emotion_info in (emotions_data or {}).items():
            if emotion_id in ('emotion_sequence', 'metadata') or not isinstance(emotion_info, dict):
                continue

            # 이 감정에서 사용할 라벨링 기반 진행 패턴/설정 로드
            progression_patterns = self._extract_progression_patterns(emotion_info)
            if not progression_patterns:
                continue
            cfg = self._get_progression_cfg(emotion_info)
            stage_relationships: Dict[str, Any] = {}

            # 상황 셋업
            situations = emotion_info.get('context_patterns', {}).get('situations', {})
            if not isinstance(situations, dict) or not situations:
                continue

            for situation_key, situation in situations.items():
                if not isinstance(situation, dict):
                    continue

                context_key = f"{emotion_id}::{situation_key}"
                if context_key not in self._context_pattern_cache:
                    self._context_pattern_cache[context_key] = self._analyze_context_patterns(situation)
                context_patterns = self._context_pattern_cache[context_key]

                # 상황 강도 가중
                intensity = situation.get('intensity', 'medium')
                intensity_multiplier = {'high': 1.2, 'medium': 1.0, 'low': 0.8}.get(intensity, 1.0)

                # 각 스테이지 매칭
                for stage_name in emotion_stages:
                    matched_info = self._analyze_stage_pattern(
                        sentences=sentences,
                        stage_name=stage_name,
                        situation=situation,
                        emotion_info=emotion_info,
                        context_patterns=context_patterns,
                        intensity_multiplier=intensity_multiplier,
                        progression_patterns=progression_patterns.get(stage_name, [])
                    )
                    if matched_info:
                        base_score = float(cfg['stage_base_scores'].get(stage_name, 0.5))
                        matched_info['stage_score'] = float(matched_info['stage_score']) * base_score * intensity_multiplier

                        # 전이 스코어(직전 스테이지가 있으면)
                        if stage_name in self.progression_stages:
                            idx = self.progression_stages.index(stage_name)
                            if idx > 0:
                                prev_stage = self.progression_stages[idx - 1]
                                if prev_stage in stage_relationships:
                                    transition_score = self._analyze_stage_transition(
                                        prev_stage_info=stage_relationships[prev_stage],
                                        current_stage_info=matched_info,
                                        emotion_info=emotion_info,
                                        cfg=cfg
                                    )
                                    matched_info['transition_score'] = transition_score
                        stage_relationships[stage_name] = matched_info

                if stage_relationships:
                    progression_score = self._calculate_progression_score(
                        stage_relationships, emotion_stages, intensity_multiplier, cfg
                    )
                    results[f"{emotion_id}_{situation_key}"] = {
                        "emotion_id": emotion_id,
                        "situation": situation_key,
                        "stage_relationships": stage_relationships,
                        "progression_score": progression_score,
                        "intensity": intensity,
                        "context_patterns": context_patterns
                    }
        return results

    def _extract_progression_patterns(self, emotion_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        라벨링 뼈대(context_patterns.situations.*.emotion_progression)에서
        단계별(trigger/development/peak/aftermath) 패턴을 뽑아 옵니다.
        반환 형식 예:
        {
            "trigger":     [{"pattern": "...", "weight": 1.0, "situation": "S1"}, ...],
            "development": [{"pattern": "...", "weight": 1.2, "situation": "S2"}, ...],
            "peak":        [...],
            "aftermath":   [...]
        }
        """
        patterns: Dict[str, List[Dict[str, Any]]] = {}

        def _add(stage: str, pat: str, w: float, situation_id: Optional[str]):
            if not isinstance(pat, str) or not pat.strip():
                return
            try:
                weight = float(w)
            except Exception:
                weight = 1.0
            patterns.setdefault(stage, []).append({
                "pattern": pat.strip(),
                "weight": weight,
                "situation": situation_id
            })

        # 1) 상위 감정의 situations
        ctx = emotion_info.get("context_patterns", {}) or {}
        situations = ctx.get("situations", {}) or {}
        if isinstance(situations, dict):
            for s_id, s_data in situations.items():
                if not isinstance(s_data, dict):
                    continue
                prog = s_data.get("emotion_progression", {}) or {}
                if isinstance(prog, dict):
                    for stage, info in prog.items():
                        if isinstance(info, str):
                            _add(stage, info, 1.0, s_id)
                        elif isinstance(info, dict):
                            pat = info.get("pattern") or info.get("description") or ""
                            w = info.get("weight", 1.0)
                            _add(stage, pat, w, s_id)

        # 2) 세부감정(sub_emotions)의 situations (있으면 가중치 살짝 보너스)
        subs = emotion_info.get("sub_emotions", {}) \
               or (emotion_info.get("emotion_profile", {}) or {}).get("sub_emotions", {}) \
               or {}
        if isinstance(subs, dict):
            for _sub_name, sub_info in subs.items():
                if not isinstance(sub_info, dict):
                    continue
                sctx = sub_info.get("context_patterns", {}) or {}
                ssits = sctx.get("situations", {}) or {}
                if not isinstance(ssits, dict):
                    continue
                for s_id, s_data in ssits.items():
                    if not isinstance(s_data, dict):
                        continue
                    prog = s_data.get("emotion_progression", {}) or {}
                    if isinstance(prog, dict):
                        for stage, info in prog.items():
                            if isinstance(info, str):
                                _add(stage, info, 1.2, s_id)  # sub은 +20% 가중
                            elif isinstance(info, dict):
                                pat = info.get("pattern") or info.get("description") or ""
                                w = info.get("weight", 1.0)
                                _add(stage, pat, float(w) * 1.2, s_id)

        return patterns

    # ---------- 상황/패턴/전이 분석 ----------
    def _analyze_context_patterns(self, situation: Dict[str, Any]) -> Dict[str, Any]:
        """상황별 문맥 패턴 분석(키워드/변형/예시 토큰 캐시)"""
        patterns = {
            'keywords': set(map(str, situation.get('keywords', []) or [])),
            'variations': set(map(str, situation.get('variations', []) or [])),
            'examples': set()
        }
        for example in situation.get('examples', []) or []:
            if isinstance(example, str):
                tokens = self.text_analyzer.tokenize(example)
                patterns['examples'].update(tokens)
        return patterns

    def _compile_pattern_list(self, key: str, patterns: List[str]) -> List[re.Pattern]:
        """문자열/평문 패턴을 안전한 정규식으로 컴파일 + 캐시"""
        if key in self._stage_pattern_cache:
            return self._stage_pattern_cache[key]
        compiled: List[re.Pattern] = []
        for p in patterns:
            if not isinstance(p, str) or not p.strip():
                continue
            try:
                compiled.append(re.compile(p, re.IGNORECASE))
            except re.error:
                compiled.append(re.compile(re.escape(p), re.IGNORECASE))
        self._stage_pattern_cache[key] = compiled
        return compiled

    def _analyze_stage_transition(
        self,
        prev_stage_info: Dict[str, Any],
        current_stage_info: Dict[str, Any],
        emotion_info: Dict[str, Any],
        cfg: Dict[str, Any]
    ) -> float:
        """단계 간 전이 분석(문장 위치/강도 변화/관련 감정 연속성 + 라벨링 전이 설정 반영)"""
        try:
            base = 0.5
            prev_index = int(prev_stage_info.get('sentence_index', -1))
            curr_index = int(current_stage_info.get('sentence_index', -1))
            if 0 <= prev_index < curr_index:
                base += 0.2  # 순서 일치 보너스
            elif curr_index <= prev_index and curr_index >= 0:
                base -= 0.05  # 역행 약감

            prev_score = float(prev_stage_info.get('stage_score', 0.0))
            curr_score = float(current_stage_info.get('stage_score', 0.0))
            delta = abs(curr_score - prev_score)

            # 전이 타입 추정(간이 휴리스틱)
            th = cfg['transition_thresholds']
            if delta >= float(th.get('high', 0.8)):
                ttype = 'sudden'
            elif delta <= float(th.get('low', 0.3)):
                ttype = 'gradual'
            else:
                ttype = 'direct'

            # 연속성(관련 감정 교집합)
            prev_emotions = set(prev_stage_info.get('related_emotions', []) or [])
            curr_emotions = set(current_stage_info.get('related_emotions', []) or [])
            continuity = (len(prev_emotions & curr_emotions) / max(len(prev_emotions | curr_emotions), 1)) if (prev_emotions or curr_emotions) else 0.0
            base += continuity * 0.2

            # 스무딩
            sm = float(cfg.get('smoothing_factor', 0.3))
            base = (1.0 - sm) * base + sm * (1.0 - min(1.0, delta))

            # 전이 타입 가중
            base *= float(cfg['transition_weights'].get(ttype, 1.0))

            # 클램프
            return max(0.0, min(1.0, round(base, 3)))
        except Exception as e:
            logger.exception(f"단계 전이 분석 중 오류: {e}")
            return 0.5

    def _calculate_progression_score(
        self,
        stage_relationships: Dict[str, Any],
        emotion_stages: List[str],
        intensity_multiplier: float,
        cfg: Dict[str, Any]
    ) -> float:
        """전체 진행 패턴의 점수 계산(라벨링 agg 가중치 반영)"""
        try:
            if not stage_relationships:
                return 0.0
            total = 0.0
            wsum = 0.0
            agg_w = cfg['agg_weights']
            for st in emotion_stages:
                if st not in stage_relationships:
                    continue
                info = stage_relationships[st]
                stage_score = float(info.get('stage_score', 0.0))
                transition_score = float(info.get('transition_score', 1.0))
                w = float(agg_w.get(st, 0.25))
                total += stage_score * transition_score * w
                wsum += w
            if wsum <= 0:
                return 0.0
            final = (total / wsum) * float(intensity_multiplier)
            return max(0.0, min(1.0, round(final, 3)))
        except Exception as e:
            logger.exception(f"진행 패턴 점수 계산 중 오류: {e}")
            return 0.0

    def _analyze_stage_pattern(
        self,
        sentences: List[str],
        stage_name: str,
        situation: Dict[str, Any],
        emotion_info: Dict[str, Any],
        context_patterns: Dict[str, Any],
        intensity_multiplier: float,
        progression_patterns: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """스테이지별 패턴/문맥 매칭 및 스코어 산출"""
        try:
            # 패턴 풀 만들기(라벨이 str/dict/list 혼재 가능)
            patterns: List[str] = []
            weights: List[float] = []
            for item in progression_patterns or []:
                if isinstance(item, dict):
                    p = item.get('pattern') or item.get('description') or ""
                    w = float(item.get('weight', 1.0))
                    if isinstance(p, str) and p.strip():
                        patterns.append(p)
                        weights.append(w)
                elif isinstance(item, str) and item.strip():
                    patterns.append(item)
                    weights.append(1.0)

            if not patterns:
                return None

            # 정규식 컴파일(캐시)
            cache_key = f"{id(emotion_info)}::{stage_name}"
            compiled = self._compile_pattern_list(cache_key, patterns)

            # 문장 스캔
            best = {
                "idx": -1, "pat_idx": -1, "score": 0.0,
                "matched_sentence": None
            }
            for idx, sent in enumerate(sentences):
                # 패턴 매칭 강도 = 매칭된 패턴 수 + 길이 보정
                hit = 0
                pat_idx = -1
                for pi, rgx in enumerate(compiled):
                    if rgx.search(sent):
                        hit += 1
                        pat_idx = pi
                if hit == 0:
                    continue

                # 문맥 점수(상황 키워드/예시/변형 최대값)
                ctx_score = self._calculate_context_score(sent, situation, emotion_info)

                # 인트라 보너스: TextAnalyzer가 후보 감정을 복수 인지하면 보너스
                ta_scores = self.text_analyzer.analyze_emotion(sent) or {}
                intra_bonus = 0.0
                if isinstance(ta_scores, dict) and '_intrasentence_candidates' in ta_scores:
                    intra_bonus = 0.1

                # 단계 점수(기본 0.7/0.3 혼합 유지)
                st_weight, ctx_weight = 0.7, 0.3
                base = st_weight * min(1.0, 0.5 + 0.25 * hit) + ctx_weight * ctx_score
                # 감정 복잡도 가중
                complexity = (emotion_info.get('metadata', {}) or {}).get('emotion_complexity', 'basic')
                c_map = {'subtle': 1.2, 'basic': 1.0, 'complex': 0.8}
                base *= float(c_map.get(complexity, 1.0))
                base += intra_bonus

                if base > best["score"]:
                    best.update({"idx": idx, "pat_idx": pat_idx, "score": base, "matched_sentence": sent})

            if best["idx"] < 0:
                return None

            # 선택된 패턴의 가중치 적용
            w = weights[best["pat_idx"]] if 0 <= best["pat_idx"] < len(weights) else 1.0
            stage_score = min(1.0, best["score"] * float(w))

            # 관련 감정 식별(기존 로직 유지)
            related = self._identify_related_emotions(best["matched_sentence"], emotion_info, include_sub_emotions=True)

            return {
                "description": patterns[best["pat_idx"]] if 0 <= best["pat_idx"] < len(patterns) else "",
                "matched_sentence": best["matched_sentence"],
                "stage_score": stage_score,
                "weight": float(w),
                "related_emotions": related,
                "intrasentence_impact": 0.1 if '_intrasentence_candidates' in (self.text_analyzer.analyze_emotion(best["matched_sentence"]) or {}) else 0.0,
                "sentence_index": best["idx"]
            }
        except Exception as e:
            logger.exception(f"[스테이지 패턴 분석] 오류: {e}")
            return None

    # ---------- 후보/트리거/세부감정/유틸 (기존 시그니처 유지, 내부 안정화만) ----------
    def _get_top_emotion_candidates(self, sentence: str, emotions_data: Dict[str, Any], top_k: int = 3) -> List[Tuple[str, str]]:
        # 기존 동작 유지(토큰 경계 매칭/문맥 점수/트리거 보너스), 내부 가드 강화
        sent = str(sentence or "").strip()
        if not sent:
            return []
        def safe_items(d: Any):
            return d.items() if isinstance(d, dict) else []
        scores: List[Tuple[str, float]] = []

        for emotion_id, info in safe_items(emotions_data):
            if emotion_id in ('emotion_sequence', 'metadata') or not isinstance(info, dict):
                continue
            lex: Set[str] = set()

            # 최상위/서브의 핵심 키워드/상황 키워드/예시/변형 수집
            prof = info.get("emotion_profile", {}) if isinstance(info.get("emotion_profile"), dict) else {}
            subs = prof.get("sub_emotions") or info.get("sub_emotions") or {}
            for _sid, sub in safe_items(subs):
                for kw in (sub.get("core_keywords") or []): lex.add(str(kw))
                sits = ((sub.get("context_patterns") or {}).get("situations") or {})
                for s in (sits.values() if isinstance(sits, dict) else []):
                    if not isinstance(s, dict): continue
                    for k in (s.get("keywords") or []): lex.add(str(k))
                    for k in (s.get("examples") or []): lex.add(str(k))
                    for k in (s.get("variations") or []): lex.add(str(k))
            top_cp = info.get("context_patterns", {}) if isinstance(info.get("context_patterns"), dict) else {}
            top_sits = top_cp.get("situations", {}) if isinstance(top_cp.get("situations"), dict) else {}
            for s in (top_sits.values() if isinstance(top_sits, dict) else []):
                if not isinstance(s, dict): continue
                for k in (s.get("keywords") or []): lex.add(str(k))
                for k in (s.get("examples") or []): lex.add(str(k))
                for k in (s.get("variations") or []): lex.add(str(k))

            if not lex:
                continue

            hits = sum(1 for token in lex if self._contains_token(sent, token))
            kw_overlap = hits / max(len(lex), 1)

            ctx_scores: List[float] = []
            for _sid, sub in safe_items(subs):
                sits = ((sub.get("context_patterns") or {}).get("situations") or {})
                if isinstance(sits, dict):
                    for s in sits.values():
                        if isinstance(s, dict):
                            ctx_scores.append(float(self._calculate_context_score(sent, s, info)))
            if isinstance(top_sits, dict):
                for s in top_sits.values():
                    if isinstance(s, dict):
                        ctx_scores.append(float(self._calculate_context_score(sent, s, info)))
            context_max = max(ctx_scores) if ctx_scores else 0.0

            trig_bonus = 0.05 if self._extract_triggers(sent, info) else 0.0
            raw = 0.7 * kw_overlap + 0.25 * context_max + trig_bonus
            scores.append((emotion_id, float(raw)))

        scores.sort(key=lambda x: x[1], reverse=True)
        top = scores[:max(1, top_k)]

        out: List[Tuple[str, str]] = []
        boost = 1 if re.search(r"(매우|아주|정말|엄청|굉장히|너무|완전|진짜|!+)", sent) else 0
        for eid, sc in top:
            if sc >= 0.45 or boost: lvl = "high"
            elif sc >= 0.18:       lvl = "medium"
            else:                  lvl = "low"
            out.append((eid, lvl))
        return out

    def _extract_triggers(self, sentence: str, emotion_info: Dict[str, Any]) -> List[str]:
        sent = str(sentence or "")
        found: set = set()

        trans = (emotion_info.get("emotion_transitions") or {})
        for p in (trans.get("patterns") or []):
            if not isinstance(p, dict): continue
            for t in (p.get("triggers") or []):
                ts = str(t)
                if ts and self._contains_token(sent, ts):
                    found.add(ts)

        top_cp = emotion_info.get("context_patterns", {}) if isinstance(emotion_info.get("context_patterns"), dict) else {}
        top_sits = top_cp.get("situations", {}) if isinstance(top_cp.get("situations"), dict) else {}
        for s in (top_sits.values() if isinstance(top_sits, dict) else []):
            if not isinstance(s, dict): continue
            trig = (s.get("emotion_progression") or {}).get("trigger")
            if isinstance(trig, str):
                if trig and self._contains_token(sent, trig):
                    found.add(trig)
            elif isinstance(trig, list):
                for ts in trig:
                    ts = str(ts)
                    if ts and self._contains_token(sent, ts):
                        found.add(ts)

        prof = emotion_info.get("emotion_profile", {}) if isinstance(emotion_info.get("emotion_profile"), dict) else {}
        subs = prof.get("sub_emotions") or emotion_info.get("sub_emotions") or {}
        for _sid, sub in (subs.items() if isinstance(subs, dict) else []):
            if not isinstance(sub, dict): continue
            sub_trans = (sub.get("emotion_transitions") or {})
            for p in (sub_trans.get("patterns") or []):
                if not isinstance(p, dict): continue
                for t in (p.get("triggers") or []):
                    ts = str(t)
                    if ts and self._contains_token(sent, ts):
                        found.add(ts)
            sits = ((sub.get("context_patterns") or {}).get("situations") or {})
            for s in (sits.values() if isinstance(sits, dict) else []):
                if not isinstance(s, dict): continue
                trig = (s.get("emotion_progression") or {}).get("trigger")
                if isinstance(trig, str):
                    if trig and self._contains_token(sent, trig):
                        found.add(trig)
                elif isinstance(trig, list):
                    for ts in trig:
                        ts = str(ts)
                        if ts and self._contains_token(sent, ts):
                            found.add(ts)

        return sorted(found)

    def _intrasentence_bonus(self, sentence: str, emotion_id: str) -> float:
        """관계/전환 단서, 라벨링 modifiers, 강도 강조 표지에 따른 보너스(0~0.25)"""
        sent = str(sentence or "")
        bonus = 0.0
        if any(c in sent for c in ["동시에", "하지만", "그러나", "그리고", "한편", "그런데", ";", ","]):
            bonus += 0.05
        # 라벨 기반 강/약 화자표
        info = (self.emotions_data or {}).get(emotion_id, {}) or {}
        mods = info.get("emotion_profile", {}).get("modifiers", {}) if isinstance(info.get("emotion_profile"), dict) else {}
        strong = set()
        weak = set()
        for k in ("intensifiers", "boosters", "amplifiers", "strong_modifiers"):
            for w in (mods.get(k) or []): strong.add(str(w))
        for k in ("downtoners", "attenuators", "weak_modifiers"):
            for w in (mods.get(k) or []): weak.add(str(w))
        if not strong: strong.update(["매우", "아주", "정말", "엄청", "굉장히", "너무", "완전", "진짜"])
        if not weak:   weak.update(["약간", "조금", "살짝", "다소"])
        if any(self._contains_token(sent, w) for w in strong) or "!" in sent: bonus += 0.10
        if any(self._contains_token(sent, w) for w in weak):                 bonus += 0.03
        return float(min(0.25, round(bonus, 3)))

    def analyze_relationship_expansion(self, text: str, emotions_data: Dict[str, Any], threshold: float = 0.2) -> Dict[str, List[Dict[str, Any]]]:
        """문장 내 후보 감정에 대해 라벨링 기반 하위감정/문맥 점수 확장(기존 시그니처 유지)"""
        expansions: Dict[str, List[Dict[str, Any]]] = {}
        sentences = self._split_text_into_sentences(text.strip()) if text else []
        if not sentences:
            return expansions

        for idx, sentence in enumerate(sentences):
            try:
                candidates = self._get_top_emotion_candidates(sentence, emotions_data, top_k=3)
            except Exception:
                candidates = []
            if not candidates:
                continue

            for emotion_id, intensity_level in candidates:
                emotion_info = emotions_data.get(emotion_id, {})
                if not isinstance(emotion_info, dict):
                    continue

                subs_src = ((emotion_info.get("emotion_profile") or {}).get("sub_emotions")
                            or emotion_info.get("sub_emotions") or {})
                try:
                    triggers = self._extract_triggers(sentence, emotion_info)
                except Exception:
                    triggers = []
                try:
                    sub_emotions = self._extract_sub_emotions(sentence, emotion_info, subs_src)
                except Exception:
                    sub_emotions = []
                try:
                    bonus = self._intrasentence_bonus(sentence, emotion_id)
                except Exception:
                    bonus = 0.0

                expansion_info: Dict[str, Any] = {
                    "sentence": sentence,
                    "position": idx,
                    "triggers": triggers,
                    "sub_emotions": sub_emotions,
                    "intensity": intensity_level,
                    "context_score": 0.0,
                    "intrasentence_expansion_bonus": float(bonus) if isinstance(bonus, (int, float)) else 0.0,
                }

                sits = ((emotion_info.get("context_patterns") or {}).get("situations") or {})
                if isinstance(sits, dict) and sits:
                    scores: List[float] = []
                    for s in sits.values():
                        if isinstance(s, dict):
                            try:
                                sc = self._calculate_context_score(sentence, s, emotion_info)
                                scores.append(float(sc))
                            except Exception:
                                continue
                    expansion_info["context_score"] = max(scores) if scores else 0.0
                else:
                    expansion_info["context_score"] = 0.0

                pass_intensity = (str(intensity_level) != "low")
                if expansion_info["context_score"] >= threshold or pass_intensity:
                    expansions.setdefault(emotion_id, []).append(expansion_info)

        return expansions

    # ---------- 세부 계산/유틸 ----------
    def _analyze_sentence_expansion(self, sentence: str, emotion_id: str, emotion_info: Dict[str, Any], sub_emotions: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            expansion_info: Dict[str, Any] = {'triggers': [], 'sub_emotions': [], 'intensity': 'medium', 'context_score': 0.0}
            triggers = self._extract_emotion_triggers(emotion_info)
            matched_triggers = [t for t in triggers if t and self._contains_token(sentence, t)]
            if matched_triggers:
                expansion_info['triggers'].extend(matched_triggers)
            for sub_id, sub_info in (sub_emotions or {}).items():
                if not isinstance(sub_info, dict):
                    continue
                sub_triggers = self._extract_emotion_triggers(sub_info)
                sub_matched = [t for t in sub_triggers if t and self._contains_token(sentence, t)]
                if sub_matched:
                    expansion_info['sub_emotions'].append({'id': sub_id, 'triggers': sub_matched})
            expansion_info['intensity'] = self._analyze_intensity(sentence, emotion_info)
            expansion_info['context_score'] = self._calculate_context_score(
                sentence, emotion_info.get('context_patterns', {}).get('situations', {}), emotion_info
            )
            return expansion_info if (expansion_info['triggers'] or expansion_info['sub_emotions']) else None
        except Exception as e:
            logger.exception(f"[문장 확장 분석] 오류: {e}")
            return None

    def _extract_emotion_triggers(self, emotion_info: Dict[str, Any]) -> List[str]:
        """감정 트리거 추출(기존 시グ 유지, 가드 강화)"""
        triggers: Set[str] = set()
        try:
            et = emotion_info.get('emotion_triggers', {})
            if isinstance(et, dict):
                triggers.update([str(x) for x in (et.get('expansion_triggers') or [])])
            trans = emotion_info.get('emotion_transitions', {})
            if isinstance(trans, dict):
                for pattern in (trans.get('patterns') or []):
                    if isinstance(pattern, dict):
                        triggers.update([str(x) for x in (pattern.get('triggers') or [])])
            return sorted(t for t in triggers if t)
        except Exception as e:
            logger.exception(f"[트리거 추출] 오류: {e}")
            return []

    def _analyze_intensity(self, sentence: str, emotion_info: Dict[str, Any]) -> str:
        """문장의 감정 강도 분석(라벨 강도 예시 + TextAnalyzer 강도 보조) → 'low'|'medium'|'high'"""
        try:
            # 1) 규칙 기반 기본값
            base = 'medium'
            tokens = set(self.text_analyzer.tokenize(sentence or ""))
            gi = getattr(self.text_analyzer, '_global_intensity', {}) or {}
            mods = getattr(self.text_analyzer, 'intensity_modifiers', {}) or {}
            if not mods:
                try:
                    mods = self.text_analyzer._build_intensity_modifiers()
                except Exception:
                    mods = {}
            intensity_terms = {
                'high': set(gi.get('high', [])) | set(mods.get('high', [])),
                'medium': set(gi.get('medium', [])) | set(mods.get('medium', [])),
                'low': set(gi.get('low', [])) | set(mods.get('low', [])),
            }
            for lvl in ('high', 'medium', 'low'):
                terms = intensity_terms.get(lvl) or set()
                if any(term in tokens for term in terms):
                    base = lvl
                    break

            # 2) 라벨 예시 유사도
            prof = emotion_info.get('emotion_profile', {}) if isinstance(emotion_info.get('emotion_profile'), dict) else {}
            ilv = prof.get('intensity_levels', {}) if isinstance(prof.get('intensity_levels'), dict) else {}
            iex = ilv.get('intensity_examples', {}) if isinstance(ilv.get('intensity_examples'), dict) else {}
            max_sim, best = 0.0, base
            for level, examples in (iex.items() if isinstance(iex, dict) else []):
                if not isinstance(examples, list): continue
                for ex in examples:
                    sim = self._calculate_text_similarity(sentence, str(ex))
                    if sim > max_sim:
                        max_sim, best = sim, str(level)
            if max_sim > 0.5:
                base = best

            # 3) TextAnalyzer 강도(숫자) 보조 → 레벨 맵핑
            try:
                i_scores = self.text_analyzer._analyze_intensity(sentence) or {}
                if isinstance(i_scores, dict):
                    v = max(i_scores.values()) if i_scores else 0.0
                    # 간단 맵핑
                    if v >= 0.45: base = 'high'
                    elif v >= 0.2: base = 'medium'
                    else: base = 'low'
            except Exception:
                pass

            return base
        except Exception as e:
            logger.exception(f"[강도 분석] 오류: {e}")
            return 'medium'

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        try:
            t1 = set(self.text_analyzer.tokenize(text1))
            t2 = set(self.text_analyzer.tokenize(text2))
            inter = len(t1 & t2)
            uni = len(t1 | t2)
            return inter / uni if uni > 0 else 0.0
        except Exception as e:
            logger.exception(f"[텍스트 유사도] 오류: {e}")
            return 0.0

    def _analyze_emotion_influence(self, from_emotion: Dict[str, Any], to_emotion: Dict[str, Any], sentence: str) -> float:
        """ 한 감정이 다른 감정에 미치는 영향력(전이 패턴 + 트리거 + 강도 변화) """
        try:
            base = 0.5
            transitions = from_emotion.get('emotion_transitions', {})
            if isinstance(transitions, dict):
                for pattern in (transitions.get('patterns') or []):
                    if not isinstance(pattern, dict):
                        continue
                    to_id = (to_emotion.get('metadata', {}) or {}).get('emotion_id')
                    if pattern.get('to_emotion') and to_id and pattern.get('to_emotion') == to_id:
                        if any((t and self._contains_token(sentence, str(t))) for t in (pattern.get('triggers') or [])):
                            ta = pattern.get('transition_analysis', {}) or {}
                            ic = str(ta.get('intensity_change', 'medium'))
                            return {'high': 0.8, 'medium': 0.5, 'low': 0.3}.get(ic, 0.5)
            return base
        except Exception as e:
            logger.exception(f"[감정 영향력] 오류: {e}")
            return 0.5

    def _identify_related_emotions(self, text: str, emotion_info: Dict[str, Any], include_sub_emotions: bool = False) -> List[str]:
        related: List[str] = []
        try:
            prof = emotion_info.get('emotion_profile', {}) if isinstance(emotion_info.get('emotion_profile'), dict) else {}
            core = prof.get('core_keywords', []) if isinstance(prof.get('core_keywords'), list) else []
            ilv = prof.get('intensity_levels', {}) if isinstance(prof.get('intensity_levels'), dict) else {}
            iex = ilv.get('intensity_examples', {}) if isinstance(ilv.get('intensity_examples'), dict) else {}
            all_kw: List[str] = [kw for kw in core if isinstance(kw, str)]
            if isinstance(iex, dict):
                for exs in iex.values():
                    if isinstance(exs, list):
                        all_kw.extend([ex for ex in exs if isinstance(ex, str)])

            if include_sub_emotions:
                subs = emotion_info.get('sub_emotions', {})
                if isinstance(subs, dict):
                    for _, sub in subs.items():
                        if not isinstance(sub, dict): continue
                        sprof = sub.get('emotion_profile', {}) if isinstance(sub.get('emotion_profile'), dict) else {}
                        all_kw.extend([kw for kw in (sprof.get('core_keywords') or []) if isinstance(kw, str)])
                        s_ilv = sprof.get('intensity_levels', {}) if isinstance(sprof.get('intensity_levels'), dict) else {}
                        s_iex = s_ilv.get('intensity_examples', {}) if isinstance(s_ilv.get('intensity_examples'), dict) else {}
                        for exs in (s_iex.values() if isinstance(s_iex, dict) else []):
                            if isinstance(exs, list):
                                all_kw.extend([ex for ex in exs if isinstance(ex, str)])

            if any((kw and self._contains_token(text, kw)) for kw in all_kw):
                meta = emotion_info.get('metadata', {}) or {}
                eid = meta.get('emotion_id') or meta.get('name') or meta.get('label') or 'unknown'
                related.append(str(eid))
            return related
        except Exception as e:
            logger.exception(f"[감정 매칭] 에러: {e}")
            return related

    def _calculate_context_score(self, sentence: str, situation_or_situations: Any, emotion_info: Dict[str, Any]) -> float:
        """단일/복수 situation 모두 처리, overlap 기반 0~1 점수"""
        if isinstance(situation_or_situations, dict) and not any(
            k in situation_or_situations for k in ("keywords", "examples", "variations")
        ):
            scores = []
            for v in situation_or_situations.values():
                if isinstance(v, dict):
                    scores.append(self._calculate_context_score(sentence, v, emotion_info))
            return max(scores) if scores else 0.0

        situation = situation_or_situations if isinstance(situation_or_situations, dict) else {}
        if not situation:
            return 0.0

        sent = str(sentence)
        kw = set(map(str, situation.get("keywords", []) or []))
        ex = set(map(str, situation.get("examples", []) or []))
        var = set(map(str, situation.get("variations", []) or []))

        def overlap_score(candidates: Set[str]) -> float:
            if not candidates: return 0.0
            hit = sum(1 for c in candidates if self._contains_token(sent, c))
            return hit / max(len(candidates), 1)

        score = 0.5 * overlap_score(kw) + 0.3 * overlap_score(ex) + 0.2 * overlap_score(var)
        return float(max(0.0, min(1.0, score)))

    # ---------- 토큰/매칭/문장 분리 ----------

    def _contains_token(self, text: str, needle: str) -> bool:
        if not needle:
            return False
        return bool(_get_token_regex(needle).search(text or ""))


    def _extract_sub_emotions(self, text: str, emotion_info: Dict[str, Any], sub_emotions: Dict[str, Any]) -> List[Dict[str, Any]]:
        """ 세부감정 추출(기존 로직 유지, 경계 매칭/가중치 보정) """
        try:
            if sub_emotions is None:
                sub_emotions = ((emotion_info.get("emotion_profile") or {}).get("sub_emotions")
                                or emotion_info.get("sub_emotions") or {})
            detected: List[Dict[str, Any]] = []
            if isinstance(sub_emotions, dict):
                for sub_id, sub in sub_emotions.items():
                    if not isinstance(sub, dict): continue
                    profile = sub.get('emotion_profile', {}) if isinstance(sub.get('emotion_profile'), dict) else {}
                    keywords = list(profile.get('core_keywords', []) or [])
                    # 상황 키워드/예시
                    ctx = sub.get('context_patterns', {}) if isinstance(sub.get('context_patterns'), dict) else {}
                    sits = ctx.get('situations', {}) if isinstance(ctx.get('situations'), dict) else {}
                    for st in (sits.values() if isinstance(sits, dict) else []):
                        if isinstance(st, dict):
                            keywords.extend(st.get('keywords', []) or [])
                    # 강도 예시 토큰화
                    ilv = profile.get('intensity_levels', {}) if isinstance(profile.get('intensity_levels'), dict) else {}
                    iex = ilv.get('intensity_examples', {}) if isinstance(ilv.get('intensity_examples'), dict) else {}
                    for exs in (iex.values() if isinstance(iex, dict) else []):
                        if isinstance(exs, list):
                            for ex in exs:
                                keywords.extend(self.text_analyzer.tokenize(str(ex)))

                    matched_kw = [k for k in keywords if isinstance(k, str) and self._contains_token(text, k)]
                    if matched_kw:
                        complexity = (sub.get('metadata', {}) or {}).get('emotion_complexity', 'basic')
                        weight = self.sub_emotion_weights.get(complexity, 1.0)
                        detected.append({'sub_emotion_id': sub_id, 'weight': weight, 'matched_keywords': matched_kw})
            return detected
        except Exception as e:
            logger.exception(f"[세부감정 추출] 에러: {e}")
            return []

    def _split_text_into_sentences(self, text: str) -> List[str]:
        """한글 구두점/개행/따옴표/말줄임표 대응 문장 분리"""
        if not text:
            return []
        # Sentence end: ., !, ? + closing quotes/parentheses allowed
        pat = r'(?<=[\.!\?])[""\')\]\}]*\s+'
        parts = re.split(pat, text.strip())
        # 라인브레이크도 문장 경계로 취급
        out: List[str] = []
        for p in parts:
            out.extend([s for s in re.split(r'\s*[\r\n]+\s*', p) if s])
        return out

    def _find_matching_sentence(self, sentences: List[str], pattern: str) -> Optional[str]:
        """패턴과 일치하는 문장 찾기(정규식 안전 컴파일)"""
        try:
            try:
                rgx = re.compile(pattern)
            except re.error:
                rgx = re.compile(re.escape(pattern))
            for s in sentences:
                if rgx.search(s):
                    return s
            return None
        except Exception as e:
            logger.exception(f"[문장 매칭] 오류 발생: {e}")
            return None


# =============================================================================
# EmotionRelationshipAnalyzer (improved)
# =============================================================================
class EmotionRelationshipAnalyzer:
    def __init__(
            self,
            config: Optional[Dict[str, Any]] = None,
            emotions_data: Optional[Dict[str, Any]] = None,
            text_analyzer: Optional["TextAnalyzer"] = None
    ):
        # 1) 기본 설정 로드 후 외부 config로 덮어씀
        default_config = self._get_default_config()
        self.config = deep_update(default_config, config or {})

        # 2) EMOTIONS.json 로드 (하드코딩 대신)
        if emotions_data is None:
            self.emotions_data = self._load_emotions_data()
        else:
            self.emotions_data = emotions_data
            
        self.text_analyzer = text_analyzer if text_analyzer else TextAnalyzer(self.emotions_data)

        # 분석 가중치/임계/환경
        self.weights = self.config['analysis_weights']
        self.relationship_base_scores = self.config['relationship_scores']
        self.default_emotion_relations = self.config['default_relations']
        
        # EMOTIONS.json 기반 패턴 캐시 초기화
        self._emotion_patterns_cache = {}
        self._relationship_cache = {}
        self._load_emotion_patterns_recursive()
        self.emotion_complexity = self.config['complexity_multipliers']
        self.intensity_weights = self.config['intensity_weights']
        self.transition_weights = self.config['transition_weights']
        self.context_weights = self.config['context_weights']
        
        # 메모리 모니터 초기화
        self.memory_monitor = None
        try:
            from src.emotion_analysis.psychological_analyzer import MemoryGuard
            self.memory_monitor = MemoryGuard(limit_mb=512)
        except Exception:
            pass  # 메모리 모니터 없어도 계속 진행

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

    def _load_emotion_patterns_recursive(self, emotion_data: Dict[str, Any] = None, path: str = "") -> None:
        """EMOTIONS.json에서 감정 패턴을 재귀적으로 로드"""
        if emotion_data is None:
            emotion_data = self.emotions_data
            
        for emotion_key, emotion_info in emotion_data.items():
            current_path = f"{path}.{emotion_key}" if path else emotion_key
            
            # 하위 감정 재귀 처리
            if isinstance(emotion_info, dict):
                if "sub_emotions" in emotion_info:
                    self._load_emotion_patterns_recursive(emotion_info["sub_emotions"], current_path)
                
                # 패턴 추출
                patterns = self._extract_patterns_from_emotion(emotion_info, current_path)
                if patterns:
                    self._emotion_patterns_cache[current_path] = patterns
                    
                # 관계 정보 추출
                relationships = self._extract_relationships_from_emotion(emotion_info, current_path)
                if relationships:
                    self._relationship_cache[current_path] = relationships

    def _extract_patterns_from_emotion(self, emotion_info: Dict[str, Any], emotion_path: str) -> List[str]:
        """감정 정보에서 패턴 추출"""
        patterns = []
        
        # intensity_examples에서 패턴 추출
        if "emotion_profile" in emotion_info:
            profile = emotion_info["emotion_profile"]
            if "intensity_levels" in profile:
                intensity_levels = profile["intensity_levels"]
                if "intensity_examples" in intensity_levels:
                    examples = intensity_levels["intensity_examples"]
                    for level, example_list in examples.items():
                        if isinstance(example_list, list):
                            patterns.extend(example_list)
        
        # keywords에서 패턴 추출
        if "keywords" in emotion_info:
            patterns.extend(emotion_info["keywords"])
            
        # triggers에서 패턴 추출
        if "triggers" in emotion_info:
            patterns.extend(emotion_info["triggers"])
            
        return patterns

    def _extract_relationships_from_emotion(self, emotion_info: Dict[str, Any], emotion_path: str) -> Dict[str, Any]:
        """감정 정보에서 관계 정보 추출"""
        relationships = {}
        
        if "emotion_profile" in emotion_info:
            profile = emotion_info["emotion_profile"]
            if "related_emotions" in profile:
                related = profile["related_emotions"]
                relationships.update(related)
                
        return relationships

    def analyze_relationships(self, text: str) -> Dict[str, Any]:
        """EMOTIONS.json 기반 감정 관계 분석"""
        try:
            # 문장 분리
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            if not sentences:
                sentences = [text.strip()]
            
            # 각 문장에서 감정 패턴 매칭
            sentence_emotions = []
            for i, sentence in enumerate(sentences):
                detected_emotions = self._detect_emotions_in_sentence(sentence)
                sentence_emotions.append({
                    "sentence_index": i,
                    "sentence": sentence,
                    "detected_emotions": detected_emotions
                })
            
            # 감정 관계 분석
            relationships = self._analyze_emotion_relationships(sentence_emotions)
            
            # 전이 분석
            transitions = self._analyze_emotion_transitions(sentence_emotions)
            
            return {
                "sentence_emotions": sentence_emotions,
                "relationships": relationships,
                "transitions": transitions,
                "summary": {
                    "total_sentences": len(sentences),
                    "total_emotions": sum(len(se["detected_emotions"]) for se in sentence_emotions),
                    "relationship_count": len(relationships),
                    "transition_count": len(transitions)
                },
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "success": True
            }
            
        except Exception as e:
            return {
                "sentence_emotions": [],
                "relationships": [],
                "transitions": [],
                "summary": {
                    "total_sentences": 0,
                    "total_emotions": 0,
                    "relationship_count": 0,
                    "transition_count": 0
                },
                "error": str(e),
                "success": False
            }

    def _detect_emotions_in_sentence(self, sentence: str) -> List[Dict[str, Any]]:
        """문장에서 감정 패턴 매칭"""
        detected = []
        sentence_lower = sentence.lower()
        
        # 기본 영어 감정 키워드 매핑 (하드코딩된 트리거)
        english_emotion_mapping = {
            "happy": "희",
            "joy": "희", 
            "pleasure": "희",
            "delight": "희",
            "angry": "노",
            "mad": "노",
            "furious": "노",
            "rage": "노",
            "sad": "애",
            "depressed": "애",
            "gloomy": "애",
            "sorrow": "애",
            "satisfaction": "락",
            "content": "락",
            "cheerful": "락"
        }
        
        # 영어 키워드 매칭
        for english_word, emotion in english_emotion_mapping.items():
            if english_word in sentence_lower:
                detected.append({
                    "emotion": emotion,
                    "pattern": english_word,
                    "confidence": 0.8
                })
                break
        
        # EMOTIONS.json 패턴 매칭 (한국어)
        for emotion_path, patterns in self._emotion_patterns_cache.items():
            for pattern in patterns:
                if isinstance(pattern, str) and pattern.lower() in sentence_lower:
                    detected.append({
                        "emotion": emotion_path,
                        "pattern": pattern,
                        "confidence": 0.8
                    })
                    break  # 한 감정당 하나의 패턴만 매칭
                    
        return detected

    def _analyze_emotion_relationships(self, sentence_emotions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """감정 관계 분석"""
        relationships = []
        
        for i, se in enumerate(sentence_emotions):
            for emotion in se["detected_emotions"]:
                emotion_path = emotion["emotion"]
                
                # 관계 정보 확인
                if emotion_path in self._relationship_cache:
                    related_emotions = self._relationship_cache[emotion_path]
                    
                    # 같은 문장 내 다른 감정과의 관계 확인
                    for other_emotion in se["detected_emotions"]:
                        if other_emotion["emotion"] != emotion_path:
                            relationships.append({
                                "emotion1": emotion_path,
                                "emotion2": other_emotion["emotion"],
                                "relationship_type": "co_occurrence",
                                "sentence_index": i,
                                "confidence": 0.7
                            })
        
        return relationships

    def _analyze_emotion_transitions(self, sentence_emotions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """감정 전이 분석"""
        transitions = []
        
        for i in range(1, len(sentence_emotions)):
            prev_emotions = [e["emotion"] for e in sentence_emotions[i-1]["detected_emotions"]]
            curr_emotions = [e["emotion"] for e in sentence_emotions[i]["detected_emotions"]]
            
            if prev_emotions != curr_emotions:
                transitions.append({
                    "from_sentence": i-1,
                    "to_sentence": i,
                    "previous_emotions": prev_emotions,
                    "current_emotions": curr_emotions,
                    "transition_type": "emotion_change"
                })
        
        return transitions

    def _get_default_config(self) -> Dict[str, Any]:
        """기본 설정 반환"""
        return {
            'analysis_weights': {
                'pattern': 0.3,
                'context': 0.2,
                'intensity': 0.2,
                'transition': 0.15,
                'complexity': 0.15
            },
            'relationship_scores': {
                'positive': 0.8,
                'negative': 0.6,
                'neutral': 0.4
            },
            'default_relations': {
                '희': ['락'],
                '노': ['애'],
                '애': ['노'],
                '락': ['희']
            },
            'complexity_multipliers': {
                'simple': 1.0,
                'complex': 1.2,
                'very_complex': 1.5
            },
            'intensity_weights': {
                'low': 0.5,
                'medium': 1.0,
                'high': 1.5
            },
            'transition_weights': {
                'smooth': 0.8,
                'abrupt': 0.6,
                'conflict': 0.4
            },
            'context_weights': {
                'temporal': 0.3,
                'spatial': 0.2,
                'social': 0.3,
                'situational': 0.2
            },
            'thresholds': {
                'compatibility': 0.5,
                'relationship_strength': 0.6
            }
        }
        self.keyword_similarity_threshold = self.config['thresholds']['keyword_similarity']
        self.context_gate_factor = float(self.config['thresholds'].get('context_gate_factor', 0.85))
        self.keyword_gate_factor = float(self.config['thresholds'].get('keyword_gate_factor', 0.90))

        self.cache_config = self.config['cache']
        self.analysis_metadata = self.config['metadata']
        self.logging_config = self.config['logging']
        self._last_relationship_metrics: Dict[str, float] = {}

        # 3) 의존 객체/캐시
        # memory_monitor는 이미 __init__에서 초기화됨

        self.relationship_cache: Dict[str, Any] = {}
        self.current_emotion_state: Dict[str, Any] = {}
        self.validator = EmotionValidatorExtended()
        self.progression_analyzer = None

        # 컨텍스트/키워드 인덱스(호환성 계산 가속)
        self._context_index = self._build_context_index(self.emotions_data)

        logger.info(f"[EmotionRelationshipAnalyzer] 초기화 완료 (버전: {self.analysis_metadata['version']})")

    # ---------------------------- 기본 설정 ---------------------------- #
    def _get_default_config(self) -> Dict[str, Any]:
        return {
            'analysis_weights': {
                'core_keywords_match': 0.4,
                'intensity_match': 0.3,
                'context_match': 0.2,
                'sentiment_match': 0.1,
                'transition_match': 0.3,
                'situation_match': 0.2,
                'progression_match': 0.2
            },
            'relationship_scores': {
                'synergistic': 0.8,
                'complementary': 0.6,
                'reinforcing': 0.5,
                'neutral': 0.0,
                'contrasting': -0.4,
                'conflicting': -0.7
            },
            'default_relations': {
                '희': {
                    '노': {'relationship_type': 'contrasting', 'strength': 0.3, 'compatibility': -0.5},
                    '애': {'relationship_type': 'complementary', 'strength': 0.6, 'compatibility': 0.7},
                    '락': {'relationship_type': 'synergistic', 'strength': 0.8, 'compatibility': 0.9}
                },
                '노': {
                    '희': {'relationship_type': 'contrasting', 'strength': 0.3, 'compatibility': -0.5},
                    '애': {'relationship_type': 'reinforcing', 'strength': 0.7, 'compatibility': 0.6},
                    '락': {'relationship_type': 'contrasting', 'strength': 0.2, 'compatibility': -0.3}
                },
                '애': {
                    '희': {'relationship_type': 'complementary', 'strength': 0.6, 'compatibility': 0.7},
                    '노': {'relationship_type': 'reinforcing', 'strength': 0.7, 'compatibility': 0.6},
                    '락': {'relationship_type': 'contrasting', 'strength': 0.4, 'compatibility': -0.4}
                },
                '락': {
                    '희': {'relationship_type': 'synergistic', 'strength': 0.8, 'compatibility': 0.9},
                    '노': {'relationship_type': 'contrasting', 'strength': 0.2, 'compatibility': -0.3},
                    '애': {'relationship_type': 'contrasting', 'strength': 0.4, 'compatibility': -0.4}
                }
            },
            'complexity_multipliers': {
                'low': {'threshold_multiplier': 1.0},
                'medium': {'threshold_multiplier': 0.8},
                'high': {'threshold_multiplier': 0.6}
            },
            'intensity_weights': {'high': 0.9, 'medium': 0.6, 'low': 0.3},
            'transition_weights': {'direct': 1.0, 'gradual': 0.7, 'sudden': 0.5, 'cyclic': 0.3},
            'context_weights': {'exact_match': 1.0, 'partial_match': 0.7, 'similar_match': 0.5, 'related_match': 0.3},
            'thresholds': {
                'compatibility': 0.5,
                'relationship_strength': 0.5,
                'sentiment': 0.3,
                'context_similarity': 0.4,
                'keyword_similarity': 0.3,
                'context_gate_factor': 0.85,
                'keyword_gate_factor': 0.90
            },
            'cache': {'max_size': 1000, 'ttl': 3600, 'cleanup_interval': 300},
            'metadata': {
                'version': '1.2',  # <== bumped
                'supported_emotions': ['희', '노', '애', '락'],
                'supported_relationship_types': ['synergistic', 'complementary', 'reinforcing', 'neutral',
                                                 'contrasting', 'conflicting'],
                'analysis_components': ['core_keywords_match', 'intensity_match', 'context_match', 'sentiment_match',
                                        'transition_match', 'situation_match', 'progression_match']
            },
            'logging': {
                'log_level': 'INFO',
                'log_format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'log_file': 'emotion_analysis.log'
            }
        }

    # ---------------------------- 유틸/헬퍼 ---------------------------- #
    @staticmethod
    def _clamp01(x: float) -> float:
        try:
            return 0.0 if x != x else 1.0 if x > 1.0 else (0.0 if x < 0.0 else float(x))
        except Exception:
            return 0.0

    def _contains_token(self, text: str, needle: str) -> bool:
        if not needle:
            return False
        return bool(_get_token_regex(needle).search(text or ""))


    def _build_context_index(self, emotions_data: Dict[str, Any]) -> Dict[str, Dict[str, Set[str]]]:
        """문맥/예시/변형·키워드를 미리 set으로 축약해 유사도 계산을 가속."""
        index: Dict[str, Dict[str, Set[str]]] = {}
        for eid, info in (emotions_data or {}).items():
            if eid in ('emotion_sequence', 'metadata') or not isinstance(info, dict):
                continue
            kw_set, var_set, ex_tok = set(), set(), set()
            contexts = (info.get('context_patterns') or {}).get('situations', {}) or {}
            if isinstance(contexts, dict):
                for s in contexts.values():
                    if not isinstance(s, dict):
                        continue
                    for k in (s.get('keywords') or []): kw_set.add(str(k))
                    for v in (s.get('variations') or []): var_set.add(str(v))
                    ex = s.get('examples', [])
                    if isinstance(ex, list):
                        for e in ex:
                            if isinstance(e, str):
                                ex_tok.update(self.tokenize(e))
            index[eid] = {'keywords': kw_set, 'variations': var_set, 'examples': ex_tok}
        return index

    def _adjust_threshold_by_complexity(self, base_threshold: float) -> float:
        emo_cplx = self._get_current_emotion_complexity()
        try:
            multiplier = float(self.emotion_complexity[emo_cplx]['threshold_multiplier'])
        except Exception:
            multiplier = 1.0
        return base_threshold * multiplier

    def _get_current_emotion_complexity(self) -> str:
        try:
            seq = (self.emotions_data or {}).get('emotion_sequence', []) if isinstance(getattr(self, 'emotions_data', {}), dict) else []
            active: Set[str] = set()
            for entry in seq or []:
                if not isinstance(entry, dict):
                    continue
                emotions = entry.get('emotions')
                if isinstance(emotions, dict):
                    for eid, val in emotions.items():
                        try:
                            if float(val) >= 0.2:
                                active.add(str(eid))
                        except Exception:
                            continue
                elif isinstance(emotions, list):
                    for item in emotions:
                        if not isinstance(item, dict):
                            continue
                        eid = item.get('emotion_id') or item.get('id') or item.get('emotion')
                        val = item.get('confidence', item.get('score'))
                        try:
                            if eid and float(val) >= 0.2:
                                active.add(str(eid))
                        except Exception:
                            continue
                sub_emotions = entry.get('sub_emotions')
                if isinstance(sub_emotions, dict):
                    for sid, sval in sub_emotions.items():
                        try:
                            if float(sval) >= 0.2:
                                active.add(str(sid))
                        except Exception:
                            continue
                elif isinstance(sub_emotions, list):
                    for sub in sub_emotions:
                        if not isinstance(sub, dict):
                            continue
                        sid = sub.get('name') or sub.get('emotion_id') or sub.get('id')
                        val = sub.get('score', sub.get('confidence'))
                        try:
                            if sid and float(val) >= 0.2:
                                active.add(str(sid))
                        except Exception:
                            continue
            n = len(active)
            if n > 5:
                return 'high'
            if n > 2:
                return 'medium'
            return 'low'
        except Exception:
            return 'low'

    # ---------------------------- 핵심: 관계 산출 ---------------------------- #
    def _apply_safety_gates(
            self,
            total_score: float,
            context_similarity: float,
            keyword_similarity: float
    ) -> float:
        ctx_th = float(getattr(self, "context_similarity_threshold", 0.35))
        kwd_th = float(getattr(self, "keyword_similarity_threshold", 0.35))
        ctx_gate = float(getattr(self, "context_gate_factor", 0.85))
        kwd_gate = float(getattr(self, "keyword_gate_factor", 0.90))

        if context_similarity < ctx_th:
            gap = min(1.0, (ctx_th - context_similarity) / max(1e-6, ctx_th))
            total_score *= (1.0 - gap * (1.0 - ctx_gate))
        if keyword_similarity < kwd_th:
            gap = min(1.0, (kwd_th - keyword_similarity) / max(1e-6, kwd_th))
            total_score *= (1.0 - gap * (1.0 - kwd_gate))
        return total_score


    def _calculate_emotion_relationship(self, from_emotion: Dict, to_emotion: Dict) -> Dict[str, Any]:
        """라벨링 메타·키워드·강도·문맥·극성·서브감정 요소를 혼합한 관계 요약."""
        try:
            from_meta = from_emotion.get('metadata', {}) if isinstance(from_emotion, dict) else {}
            to_meta = to_emotion.get('metadata', {}) if isinstance(to_emotion, dict) else {}

            # (A) 라벨링 전이 패턴 기반 베이스 스코어
            relationship_base_scores: Dict[str, float] = {}
            transitions = (from_emotion.get('emotion_transitions') or {}).get('patterns', []) if isinstance(from_emotion, dict) else []
            for p in transitions:
                if not isinstance(p, dict): continue
                if p.get('to_emotion') == to_meta.get('emotion_id'):
                    rtype = p.get('relationship_type', 'neutral')
                    t_analysis = p.get('transition_analysis', {}) or {}
                    intensity_change = str(t_analysis.get('intensity_change', 'medium'))
                    weight = {'high': 0.9, 'medium': 0.6, 'low': 0.3}.get(intensity_change, 0.5)
                    relationship_base_scores[rtype] = weight

            pattern_signal = max(relationship_base_scores.values()) if relationship_base_scores else 0.0

            # (B) 구성 요소별 유사도/호환성
            core_similarity = self._calculate_keyword_similarity(
                (from_emotion.get('emotion_profile') or {}).get('core_keywords', []),
                (to_emotion.get('emotion_profile') or {}).get('core_keywords', [])
            )
            intensity_compatibility = self._calculate_intensity_compatibility(
                (from_emotion.get('emotion_profile') or {}).get('intensity_levels', {}),
                (to_emotion.get('emotion_profile') or {}).get('intensity_levels', {})
            )
            context_similarity = self._calculate_context_similarity(
                from_emotion.get('context_patterns', {}),
                to_emotion.get('context_patterns', {})
            )
            sentiment_compatibility = self._calculate_sentiment_compatibility(
                (from_emotion.get('emotion_profile') or {}).get('related_emotions', {}),
                (to_emotion.get('emotion_profile') or {}).get('related_emotions', {})
            )
            sub_emotion_factor = self._calculate_sub_emotion_factor(from_emotion, to_emotion)

            ml_meta = (from_emotion.get('ml_training_metadata') or {}) if isinstance(from_emotion, dict) else {}
            conf = ml_meta.get('confidence_thresholds', {'basic': 0.7, 'complex': 0.8, 'subtle': 0.9})
            emo_cplx = from_meta.get('emotion_complexity', 'basic')
            threshold = float(conf.get(emo_cplx, 0.7))

            keyword_similarity = core_similarity
            total_score = (
                core_similarity * self.weights['core_keywords_match'] +
                intensity_compatibility * self.weights['intensity_match'] +
                context_similarity * self.weights['context_match'] +
                sentiment_compatibility * self.weights['sentiment_match'] +
                sub_emotion_factor * 0.2 +
                pattern_signal * self.weights.get('transition_match', 0.3)
            )
            pre_gate_score = total_score
            total_score = self._apply_safety_gates(total_score, context_similarity, keyword_similarity)
            total_score = self._clamp01(total_score)
            self._last_relationship_metrics = {
                'context_similarity': context_similarity,
                'keyword_similarity': keyword_similarity,
                'pattern_signal': pattern_signal,
                'pre_gate_score': pre_gate_score,
                'total_score': total_score
            }

            relationship_type = self._determine_relationship_type(total_score)
            return {
                'relationship_type': relationship_type,
                'strength': abs(total_score),
                'compatibility': total_score,
                'details': {
                    'core_similarity': round(core_similarity, 3),
                    'keyword_similarity': round(keyword_similarity, 3),
                    'intensity_compatibility': round(intensity_compatibility, 3),
                    'context_similarity': round(context_similarity, 3),
                    'sentiment_compatibility': round(sentiment_compatibility, 3),
                    'sub_emotion_factor': round(sub_emotion_factor, 3),
                    'pattern_signal': round(pattern_signal, 3),
                    'base_patterns': relationship_base_scores
                },
                'threshold_applied': threshold,
                'complexity_level': emo_cplx
            }
        except Exception as e:
            logger.exception(f"감정 관계 계산 중 오류: {e}")
            return {'relationship_type': 'unknown', 'strength': 0.0, 'compatibility': 0.0, 'details': {}, 'error': str(e)}

    # ---------------------------- 구성 요소 스코어 ---------------------------- #
    def _calculate_sub_emotion_factor(self, from_emotion: Dict[str, Any], to_emotion: Dict[str, Any]) -> float:
        try:
            from_sub = (from_emotion.get('emotion_profile') or {}).get('sub_emotions', {})
            to_sub = (to_emotion.get('emotion_profile') or {}).get('sub_emotions', {})
            if not isinstance(from_sub, dict) or not isinstance(to_sub, dict): return 0.0
            fk, tk = set(from_sub.keys()), set(to_sub.keys())
            if not fk or not tk: return 0.0
            ratio = len(fk & tk) / len(fk | tk)
            return round(min(max(ratio * 0.5, 0.0), 1.0), 3)
        except Exception as e:
            logger.exception(f"[서브감정 Factor] 계산 오류: {e}")
            return 0.0

    def _calculate_keyword_similarity(self, keywords1: List[str], keywords2: List[str]) -> float:
        if not isinstance(keywords1, list) or not isinstance(keywords2, list) or not keywords1 or not keywords2:
            return 0.0
        a, b = set(map(str, keywords1)), set(map(str, keywords2))
        return self._clamp01(len(a & b) / len(a | b) if (a or b) else 0.0)

    def _calculate_intensity_compatibility(self, intensity1: Dict, intensity2: Dict) -> float:
        try:
            levels = {'low': 0.3, 'medium': 0.6, 'high': 0.9}
            def avg_i(d: Dict) -> float:
                if not isinstance(d, dict): return 0.5
                ex = d.get('intensity_examples', {}) if isinstance(d.get('intensity_examples'), dict) else {}
                tot_w, tot_c = 0.0, 0
                for lvl, w in levels.items():
                    lst = ex.get(lvl, [])
                    if isinstance(lst, list):
                        tot_w += w * len(lst); tot_c += len(lst)
                return tot_w / tot_c if tot_c > 0 else 0.5
            diff = abs(avg_i(intensity1) - avg_i(intensity2))
            return self._clamp01(1.0 - diff)
        except Exception as e:
            logger.exception(f"강도 호환성 계산 중 오류: {e}")
            return 0.5

    def _calculate_context_similarity(self, context1: Dict, context2: Dict) -> float:
        try:
            s1 = (context1 or {}).get('situations', {}) or {}
            s2 = (context2 or {}).get('situations', {}) or {}
            if not isinstance(s1, dict) or not isinstance(s2, dict) or not s1 or not s2:
                return 0.0
            def collect(d: Dict) -> Set[str]:
                out = set()
                for v in d.values():
                    if not isinstance(v, dict): continue
                    out.update(map(str, v.get('keywords', []) or []))
                    out.update(map(str, v.get('variations', []) or []))
                    ex = v.get('examples', [])
                    if isinstance(ex, list):
                        for e in ex:
                            if isinstance(e, str):
                                try:
                                    out.update(self.tokenize(e))
                                except Exception:
                                    out.add(e)
                return out
            A, B = collect(s1), collect(s2)
            return self._clamp01(len(A & B) / len(A | B) if (A or B) else 0.0)
        except Exception as e:
            logger.exception(f"컨텍스트 유사도 계산 중 오류: {e}")
            return 0.0

    def _calculate_sentiment_compatibility(self, related1: Dict, related2: Dict) -> float:
        try:
            w = {'positive': 1.0, 'negative': -1.0, 'neutral': 0.0}
            def score(rel: Dict) -> float:
                tot = 0.0; cnt = 0
                for k, lst in (rel or {}).items():
                    if not isinstance(lst, list): continue
                    tot += w.get(k, 0.0) * len(lst); cnt += len(lst)
                return (tot / cnt) if cnt > 0 else 0.0
            return self._clamp01(1.0 - abs(score(related1) - score(related2)))
        except Exception as e:
            logger.exception(f"감정 극성 호환성 계산 중 오류: {e}")
            return 0.0

    def _determine_relationship_type(self, score: float) -> str:
        s = float(score)
        if s >= 0.7: return 'synergistic'
        elif s >= 0.4: return 'complementary'
        elif s >= 0.2: return 'reinforcing'
        elif s >= -0.2: return 'neutral'
        elif s >= -0.5: return 'contrasting'
        else: return 'conflicting'

    # ---------------------------- 최상위 엔트리 ---------------------------- #
    def analyze_emotion_relationships(
            self,
            text: str,
            emotions_data: Dict[str, Any],
            complex_emotions: Optional[Dict[str, Any]] = None,
            situation_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        try:
            if not self.validate_input_format(text) or not isinstance(emotions_data, dict):
                logger.warning("[입력 데이터] 유효성 실패")
                return {"relationship_analysis": False}

            if self.memory_monitor and not self.memory_monitor.check().get("ok", True):
                self._cleanup_resources()

            if isinstance(complex_emotions, dict):
                emotions_data.update(complex_emotions)
            if isinstance(situation_context, dict):
                emotions_data["situation_context"] = situation_context

            # validator가 없으면 기본 검증
            is_valid = isinstance(emotions_data, dict) and bool(emotions_data)
            errors = [] if is_valid else ["Invalid emotions_data structure"]
            if not is_valid:
                for err in errors:
                    path = err.get("path", "UNKNOWN_PATH")
                    msg = err.get("message", err.get("msg", "UNKNOWN_ERROR"))  # <== message 키로 수정
                    logger.error(f"[구조 오류] path='{path}' → {msg}")
                return {"relationship_analysis": False, "validation_errors": errors}

            if not hasattr(self, 'progression_analyzer') or self.progression_analyzer is None:
                self.progression_analyzer = EmotionProgressionRelationshipAnalyzer(emotions_data)

            emotion_sequence = self._validate_emotion_sequence(emotions_data)
            if not emotion_sequence:
                logger.warning("[EmotionSequence] 유효 시퀀스 없음")
                return {"relationship_analysis": False}

            # 진행/확장(스테이지·문맥 기반)
            dynamic_relationships = self.progression_analyzer.analyze_stage_relationships(text, emotions_data)
            relationship_expansion = self.progression_analyzer.analyze_relationship_expansion(text, emotions_data)

            # 페어/호환성/충돌/영향/연결/의존/지배적 관계/강도
            emotion_pairs = self._identify_emotion_pairs(emotion_sequence, emotions_data)
            compatibility = self._analyze_emotion_compatibility(emotion_pairs, emotions_data)
            conflicts = self._detect_emotion_conflicts(emotion_sequence)
            influence = self._calculate_emotion_influence(emotion_pairs, emotions_data)
            connections = self._map_emotion_connections(emotion_sequence)
            dependencies = self._analyze_emotion_dependencies(emotion_pairs, emotions_data)
            dominant_rels = self.dominant_relationships(emotion_sequence, emotions_data)
            strength = self._evaluate_relationship_strength(emotion_pairs, emotions_data)

            return {
                "relationship_analysis": True,
                "dynamic_relationships": dynamic_relationships,
                "relationship_expansion": relationship_expansion,
                "emotion_pairs": emotion_pairs,
                "compatibility": compatibility,
                "conflicts": conflicts,
                "influence": influence,
                "connections": connections,
                "dependencies": dependencies,
                "dominant_relationships": dominant_rels,
                "relationship_strength": strength,
            }
        except Exception as e:
            logger.exception(f"[감정 관계 분석] 예외: {e}")
            return {"relationship_analysis": False, "error": str(e)}

    # ---------------------------- 부가 제공/래퍼 ---------------------------- #
    def validate_input_format(self, text: str) -> bool:
        """한국어 특화 입력 유효성 검사 (한국어 문자가 포함된 텍스트만 허용)"""
        if not isinstance(text, str) or not text.strip():
            return False
        
        # 한국어 문자가 포함되어 있는지 확인
        # 한글, 공백, 구두점, 숫자만 허용 (영문은 제외)
        korean_pattern = r'[가-힣]'
        if not re.search(korean_pattern, text):
            return False
            
        # 최소 길이 체크 (너무 짧은 텍스트는 제외)
        if len(text.strip()) < 2:
            return False
            
        return True

    def analyze_dynamic_relationships(self, text: str, emotions_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            dynamic_results = {}
            for emotion_id, emotion_info in (emotions_data or {}).items():
                if not isinstance(emotion_info, dict): continue
                contexts = (emotion_info.get('context_patterns') or {})
                situations = contexts.get('situations', {}) if isinstance(contexts.get('situations'), dict) else {}
                for sid, s in situations.items():
                    if not isinstance(s, dict): continue
                    prog = s.get('emotion_progression', {}) if isinstance(s.get('emotion_progression'), dict) else {}
                    if prog:
                        dynamic_results[f"{emotion_id}_{sid}"] = {
                            'progression': prog,
                            'intensity': s.get('intensity'),
                            'keywords': s.get('keywords', [])
                        }
            return dynamic_results
        except Exception as e:
            logger.exception("[동적 관계 분석] 오류")
            return {}

    def analyze_relationship_expansion(self, text: str, emotions_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if not hasattr(self, 'progression_analyzer') or self.progression_analyzer is None:
                self.progression_analyzer = EmotionProgressionRelationshipAnalyzer(emotions_data)
            return self.progression_analyzer.analyze_relationship_expansion(text, emotions_data)
        except Exception as e:
            logger.exception(f"[관계 확장 분석] 에러: {e}")
            return {}

    def analyze_emotion_compatibility(
            self,
            emotion_pairs: List[Dict[str, Any]],
            emotions_data: Dict[str, Any],
            include_sub_emotions: bool = True
    ) -> Dict[str, float]:
        """공개 API: 감정쌍 호환성(내부 안정 메서드 위임)."""
        try:
            return self._analyze_emotion_compatibility(
                emotion_pairs, emotions_data, include_sub_emotions=include_sub_emotions
            )
        except Exception as e:
            logger.exception("analyze_emotion_compatibility(public) 오류: %s", e)
            return {}

    # ---------------------------- 보조 루틴(원 시그니처 유지) ---------------------------- #
    def relationship_expansion(self, text: str, emotions_data: Dict[str, Any], sub_emotion_threshold: float = 0.2) -> Dict[str, Any]:
        results = {}
        sentences = self._split_text_into_sentences(text or "")
        for emotion_id, emotion_info in (emotions_data or {}).items():
            if emotion_id == "emotion_sequence" or not isinstance(emotion_info, dict):
                continue
            sub_emotions = emotion_info.get("sub_emotions", {})
            if not isinstance(sub_emotions, dict):
                continue
            expansions = []
            for idx, sentence in enumerate(sentences):
                token_set = set(self.text_analyzer.tokenize(sentence))
                found_subs = []
                for sub_id, sub_info in sub_emotions.items():
                    if not isinstance(sub_info, dict): continue
                    sub_triggers = self._extract_emotion_triggers(sub_info)
                    # 경계 매칭으로 false positive 감소
                    matched_triggers = [t for t in sub_triggers if any(self._contains_token(sentence, t) for _ in [0])]
                    if matched_triggers:
                        found_subs.append({"sub_id": sub_id, "triggers": matched_triggers})
                if found_subs:
                    expansions.append({"sentence": sentence, "position": idx, "sub_emotions": found_subs})
            if expansions:
                results[emotion_id] = expansions
        return results

    def _split_text_into_sentences(self, text: str) -> List[str]:
        if not text: return []
        pat = r'(?<=[\.!\?])[""\')\]\}]*\s+'
        parts = re.split(pat, text.strip())
        out: List[str] = []
        for p in parts:
            out.extend([s for s in re.split(r'\s*[\r\n]+\s*', p) if s])
        return out

    def _extract_emotion_triggers(self, emotion_info: Dict[str, Any]) -> List[str]:
        triggers: Set[str] = set()
        if not isinstance(emotion_info, dict):
            return []
        et = emotion_info.get("emotion_triggers", {})
        if isinstance(et, dict):
            triggers.update(map(str, et.get("expansion_triggers", []) or []))
        transitions = emotion_info.get("emotion_transitions", {})
        if isinstance(transitions, dict):
            for p in transitions.get("patterns", []) or []:
                if isinstance(p, dict):
                    triggers.update(map(str, p.get("triggers", []) or []))
        return sorted(t for t in triggers if t)

    # ---------------------------- Emotion Sequence ---------------------------- #
    def _validate_emotion_sequence(self, emotions_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        sequence = emotions_data.get('emotion_sequence', [])
        if not isinstance(sequence, list) or not sequence:
            logger.warning("[감정 시퀀스] 유효하지 않음")
            return []
        validated = []
        for entry in sequence:
            if not isinstance(entry, dict) or 'emotions' not in entry:
                continue
            stage = entry.get('stage') or self._infer_emotion_stage(entry, emotions_data)
            entry['stage'] = stage
            validated.append(entry)
        return validated

    def _infer_emotion_stage(self, entry: Dict[str, Any], emotions_data: Dict[str, Any]) -> str:
        try:
            text_segment = str(entry.get('text', '') or '')
            if not text_segment: return 'unknown'
            emotions_scores = entry.get('emotions', {}) if isinstance(entry.get('emotions'), dict) else {}
            if not emotions_scores: return 'unknown'

            # 1) 점수 기반 초기 추정
            m = max(emotions_scores.values())
            if m >= 0.8: est = 'peak'
            elif m >= 0.6: est = 'development'
            elif m >= 0.4: est = 'trigger'
            else: est = 'aftermath'

            # 2) 라벨링 progression 패턴/지표 반영
            stage_points = {'trigger': 0, 'development': 0, 'peak': 0, 'aftermath': 0}
            for emotion_id, emotion_info in (emotions_data or {}).items():
                if emotion_id == 'emotion_sequence' or not isinstance(emotion_info, dict): continue
                situations = (emotion_info.get('context_patterns') or {}).get('situations', {}) or {}
                if not isinstance(situations, dict): continue
                for s in situations.values():
                    if not isinstance(s, dict): continue
                    prog = s.get('emotion_progression', {}) or {}
                    if not isinstance(prog, dict): continue
                    for stage, info in prog.items():
                        if stage not in stage_points: continue
                        if isinstance(info, str):
                            if self._contains_token(text_segment, info): stage_points[stage] += 1
                        elif isinstance(info, dict):
                            desc = info.get('description', '')
                            kws = info.get('keywords', []) or []
                            trigs = info.get('triggers', []) or []
                            if desc and self._contains_token(text_segment, desc): stage_points[stage] += 2
                            for k in kws:
                                if self._contains_token(text_segment, k): stage_points[stage] += 1
                            for t in trigs:
                                if self._contains_token(text_segment, t): stage_points[stage] += 1

            temporal = {
                'trigger': ['갑자기', '처음', '시작', '막', '이제'],
                'development': ['점점', '서서히', '차츰', '계속'],
                'peak': ['매우', '너무', '가장', '완전히', '극도로'],
                'aftermath': ['결국', '마침내', '드디어', '끝내']
            }
            for st, terms in temporal.items():
                for t in terms:
                    if self._contains_token(text_segment, t):
                        stage_points[st] += 2

            max_points = max(stage_points.values())
            inferred = max(stage_points, key=lambda k: stage_points[k]) if max_points > 0 else est

            # 라벨 meta의 stage 제약과 합치
            for _, info in (emotions_data or {}).items():
                if not isinstance(info, dict): continue
                ml = (info.get('ml_training_metadata') or {}).get('analysis_modules', {}) or {}
                pa = ml.get('progression_analyzer', {}) or {}
                if pa.get('enabled', True):
                    stg = pa.get('progression_stages', []) or []
                    if stg and inferred not in stg:
                        standard = ['trigger', 'development', 'peak', 'aftermath']
                        inferred = 'development' if inferred not in standard else inferred
            return inferred
        except Exception as e:
            logger.exception(f"[감정 단계 추론] 에러 발생: {e}")
            return 'unknown'

    def _cleanup_resources(self):
        try:
            if len(self.relationship_cache) > self.cache_config.get('max_size', 1000):
                # 오래된 항목부터 제거 (insertion-ordered dict 가정)
                while len(self.relationship_cache) > self.cache_config.get('max_size', 1000):
                    self.relationship_cache.pop(next(iter(self.relationship_cache)))
            logger.info("[자원 정리] 완료")
        except Exception:
            pass

    # ---------------------------- 페어/호환성/충돌/영향 ---------------------------- #
    def _identify_emotion_pairs(
            self,
            emotion_sequence: List[Dict[str, Any]],
            emotions_data: Dict[str, Any],
            include_sub_emotions: bool = True,
            score_threshold: float = 0.2
    ) -> List[Dict[str, Any]]:
        pairs: List[Dict[str, Any]] = []
        if not emotion_sequence:
            return pairs

        # 같은 문장 내 페어
        for idx, entry in enumerate(emotion_sequence):
            if not isinstance(entry, dict): continue
            cur = entry.get("emotions", {}) if isinstance(entry.get("emotions"), dict) else {}
            valid = [eid for eid, sc in cur.items() if isinstance(sc, (int, float)) and sc >= score_threshold]
            if include_sub_emotions:
                sub = entry.get("sub_emotions", {}) if isinstance(entry.get("sub_emotions"), dict) else {}
                for sid, sval in sub.items():
                    if isinstance(sval, (int, float)) and sval >= score_threshold:
                        valid.append(sid)
            valid = list(dict.fromkeys(valid))  # unique order-preserving
            for i in range(len(valid)):
                for j in range(i + 1, len(valid)):
                    pairs.append({"from": valid[i], "to": valid[j], "transition_type": "intrasentence", "sentence_index": idx})

        # 인접 문장 페어
        for i in range(len(emotion_sequence) - 1):
            cur = emotion_sequence[i]; nxt = emotion_sequence[i + 1]
            cur_e = cur.get("emotions", {}) if isinstance(cur.get("emotions"), dict) else {}
            nxt_e = nxt.get("emotions", {}) if isinstance(nxt.get("emotions"), dict) else {}
            cur_valid = [k for k, v in cur_e.items() if isinstance(v, (int, float)) and v >= score_threshold]
            nxt_valid = [k for k, v in nxt_e.items() if isinstance(v, (int, float)) and v >= score_threshold]
            if include_sub_emotions:
                for sid, sval in (cur.get("sub_emotions", {}) or {}).items():
                    if isinstance(sval, (int, float)) and sval >= score_threshold:
                        cur_valid.append(sid)
                for sid, sval in (nxt.get("sub_emotions", {}) or {}).items():
                    if isinstance(sval, (int, float)) and sval >= score_threshold:
                        nxt_valid.append(sid)
            cur_valid, nxt_valid = list(dict.fromkeys(cur_valid)), list(dict.fromkeys(nxt_valid))
            using_default_primaries = False
            if not nxt_valid:
                nxt_valid = ["희", "노", "애", "락"]  # 백업
                using_default_primaries = True

            for fe in cur_valid:
                from_info = emotions_data.get(fe, {}) if isinstance(emotions_data.get(fe), dict) else {}
                transition_type = "unknown"
                for te in nxt_valid:
                    # 라벨 전이 패턴 우선
                    patterns = (from_info.get("emotion_transitions") or {}).get("patterns", []) if isinstance(from_info, dict) else []
                    for p in patterns:
                        if not isinstance(p, dict): continue
                        if (p.get("from_emotion") == fe and p.get("to_emotion") == te):
                            trigs = p.get("triggers", []) or []
                            if not trigs:
                                transition_type = (p.get("transition_analysis", {}) or {}).get("intensity_change", "unknown")
                            else:
                                next_text = str(nxt.get("text", "") or "")
                                if any(self._contains_token(next_text, tg) for tg in trigs):
                                    transition_type = (p.get("transition_analysis", {}) or {}).get("intensity_change", "unknown")
                            if transition_type != "unknown": break
                    # 기본 관계 백업
                    if transition_type == "unknown":
                        default_rel = self.default_emotion_relations.get(fe, {}).get(te, {})
                        if default_rel:
                            transition_type = default_rel.get("relationship_type", "unknown")
                    pair = {"from": fe, "to": te, "transition_type": transition_type, "sentence_index": (i, i + 1)}
                    if using_default_primaries:
                        pair["is_fallback"] = True
                    pairs.append(pair)
        return pairs

    def tokenize(self, text: str) -> List[str]:
        try:
            return [t for t in self.text_analyzer.tokenize(text)]
        except Exception:
            text = re.sub(r'([.,!?])', r' \1 ', text or "")
            return [t for t in text.split() if t.strip()]

    def _analyze_emotion_compatibility(
            self,
            emotion_pairs: List[Dict[str, Any]],
            emotions_data: Dict[str, Any],
            include_sub_emotions: bool = True
    ) -> Dict[str, float]:
        scores: Dict[str, float] = {}
        if not emotion_pairs:
            return scores

        complexity_weights = {'subtle': 1.2, 'basic': 1.0, 'complex': 0.8}
        min_floor = 0.2

        for pair in emotion_pairs:
            fe, te = pair.get('from'), pair.get('to')
            if not fe or not te:
                continue
            key = f"{fe}_to_{te}"
            if key in scores:
                continue
            try:
                fi = emotions_data.get(fe, {}) if isinstance(emotions_data.get(fe), dict) else {}
                ti = emotions_data.get(te, {}) if isinstance(emotions_data.get(te), dict) else {}

                base = self._calculate_base_compatibility(fe, te)
                kw = self._calculate_advanced_keyword_similarity(fi, ti)
                iv = self._calculate_detailed_intensity_compatibility(
                    (fi.get('emotion_profile') or {}).get('intensity_levels', {}),
                    (ti.get('emotion_profile') or {}).get('intensity_levels', {})
                )
                ctx = self._calculate_enhanced_context_compatibility(
                    fi.get('context_patterns', {}), ti.get('context_patterns', {})
                )
                sent = self._calculate_sentiment_compatibility(
                    (fi.get('emotion_profile') or {}).get('related_emotions', {}),
                    (ti.get('emotion_profile') or {}).get('related_emotions', {})
                )
                sub = self._calculate_detailed_sub_emotion_factor(fi, ti) if include_sub_emotions else 0.0
                sit = self._calculate_situation_based_adjustment(fi, ti, pair.get('transition_type', ''))

                fcx = (fi.get('metadata', {}) or {}).get('emotion_complexity', 'basic')
                tcx = (ti.get('metadata', {}) or {}).get('emotion_complexity', 'basic')
                cfac = (complexity_weights.get(fcx, 1.0) + complexity_weights.get(tcx, 1.0)) / 2

                w = {'base': 0.25, 'keyword': 0.15, 'intensity': 0.15, 'context': 0.15, 'sentiment': 0.10, 'sub_emotion': 0.15, 'situation': 0.10}
                final = (base * w['base'] + kw * w['keyword'] + iv * w['intensity'] +
                         ctx * w['context'] + sent * w['sentiment'] + sub * w['sub_emotion'] + sit * w['situation']) * cfac
                if final < min_floor: final = min_floor
                scores[key] = round(self._clamp01(final), 3)
            except Exception as e:
                logger.exception(f"감정 호환성 분석 중 오류 발생: {e}")
                scores[key] = 0.5
        return scores

    # --- 세부 요인(상세판) --- #
    def _calculate_detailed_sub_emotion_factor(self, from_info: Dict[str, Any], to_info: Dict[str, Any]) -> float:
        try:
            fs, ts = from_info.get('sub_emotions', {}), to_info.get('sub_emotions', {})
            if not isinstance(fs, dict) or not isinstance(ts, dict) or not fs or not ts:
                return 0.0
            total, cnt = 0.0, 0
            cwx = {'subtle': 1.2, 'basic': 1.0, 'complex': 0.8}
            for fid, f in fs.items():
                if not isinstance(f, dict): continue
                fprof = f.get('emotion_profile', {}) if isinstance(f.get('emotion_profile'), dict) else {}
                fk = set(fprof.get('core_keywords', []) or [])
                fc = (f.get('metadata', {}) or {}).get('emotion_complexity', 'basic')
                for tid, t in ts.items():
                    if not isinstance(t, dict): continue
                    tprof = t.get('emotion_profile', {}) if isinstance(t.get('emotion_profile'), dict) else {}
                    tk = set(tprof.get('core_keywords', []) or [])
                    tc = (t.get('metadata', {}) or {}).get('emotion_complexity', 'basic')
                    kw_sim = len(fk & tk) / len(fk | tk) if (fk or tk) else 0.0

                    fr = fprof.get('related_emotions', {}) if isinstance(fprof.get('related_emotions'), dict) else {}
                    tr = tprof.get('related_emotions', {}) if isinstance(tprof.get('related_emotions'), dict) else {}
                    def flat(rel: Dict) -> Set[str]:
                        out = set()
                        for lst in (rel or {}).values():
                            if isinstance(lst, list): out.update(map(str, lst))
                        return out
                    rel_sim = 0.0
                    FA, TA = flat(fr), flat(tr)
                    if FA or TA:
                        rel_sim = len(FA & TA) / len(FA | TA) if (FA or TA) else 0.0

                    def ctx_kw(s: Dict) -> Set[str]:
                        out = set()
                        sits = (s.get('context_patterns') or {}).get('situations', {}) or {}
                        if isinstance(sits, dict):
                            for si in sits.values():
                                if not isinstance(si, dict): continue
                                out.update(map(str, si.get('keywords', []) or []))
                        return out
                    fctx, tctx = ctx_kw(f), ctx_kw(t)
                    ctx_sim = len(fctx & tctx) / len(fctx | tctx) if (fctx or tctx) else 0.0

                    cfac = (cwx.get(fc, 1.0) + cwx.get(tc, 1.0)) / 2
                    pair_score = (kw_sim * 0.4 + rel_sim * 0.3 + ctx_sim * 0.3) * cfac
                    total += pair_score; cnt += 1
            return self._clamp01((total / cnt) if cnt > 0 else 0.0)
        except Exception as e:
            logger.exception(f"세부감정 factor 계산 중 오류: {e}")
            return 0.0

    def _calculate_base_compatibility(self, from_emotion: str, to_emotion: str) -> float:
        rel = self.default_emotion_relations.get(from_emotion, {}).get(to_emotion)
        if isinstance(rel, dict):
            rel_type = rel.get('relationship_type', 'neutral')
            candidate = rel.get('compatibility', self.relationship_base_scores.get(rel_type, 0.5))
            try:
                return float(candidate)
            except (TypeError, ValueError):
                pass
        # 기본 폴백 매트릭스(대표 4감정). default_relations가 없을 때 사용됩니다.
        # 값은 상단 default_relations의 경향을 반영합니다.
        fallback = {
            '희': {'희': 1.0, '노': -0.5, '애': 0.7, '락': 0.9},
            '노': {'희': -0.5, '노': 1.0, '애': 0.6, '락': -0.3},
            '애': {'희': 0.7, '노': 0.6, '애': 1.0, '락': -0.4},
            '락': {'희': 0.9, '노': -0.3, '애': -0.4, '락': 1.0},
        }
        return float(fallback.get(from_emotion, {}).get(to_emotion, 0.5))

    def _calculate_advanced_keyword_similarity(self, from_info: Dict[str, Any], to_info: Dict[str, Any]) -> float:
        fk = set(((from_info.get('emotion_profile') or {}).get('core_keywords', []) or []))
        tk = set(((to_info.get('emotion_profile') or {}).get('core_keywords', []) or []))
        if not fk or not tk: return 0.0
        inter = sum(1.2 for kw in fk if kw in tk)  # 교집합만 가중
        return self._clamp01(inter / max(1, len(fk | tk)))

    def _calculate_detailed_intensity_compatibility(self, i1: Dict[str, Any], i2: Dict[str, Any]) -> float:
        if not i1 or not i2: return 0.5
        try:
            w = {'high': 0.9, 'medium': 0.6, 'low': 0.3}
            def weighted(d: Dict) -> float:
                ex = d.get('intensity_examples', {}) if isinstance(d.get('intensity_examples'), dict) else {}
                tot, cnt = 0.0, 0
                for lvl, ww in w.items():
                    lst = ex.get(lvl, [])
                    if isinstance(lst, list):
                        tot += ww * len(lst); cnt += len(lst)
                return tot / cnt if cnt else 0.5
            return self._clamp01(1.0 - abs(weighted(i1) - weighted(i2)))
        except Exception as e:
            logger.exception(f"강도 호환성 계산 중 오류: {e}")
            return 0.5

    def _calculate_enhanced_context_compatibility(self, c1: Dict[str, Any], c2: Dict[str, Any]) -> float:
        try:
            idx = self._context_index  # 미리 구축된 인덱스 활용
            # c1/c2는 개별 감정 노드용 설계였으나, 여기선 union 유사도 관점으로 다시 수집
            def collect(ctx: Dict[str, Any]) -> Set[str]:
                out = set()
                sits = (ctx or {}).get('situations', {}) or {}
                if not isinstance(sits, dict): return out
                for s in sits.values():
                    if not isinstance(s, dict): continue
                    out.update(map(str, s.get('keywords', []) or []))
                    out.update(map(str, s.get('variations', []) or []))
                    ex = s.get('examples', [])
                    if isinstance(ex, list):
                        for e in ex:
                            if isinstance(e, str):
                                out.update(self.tokenize(e))
                return out
            A, B = collect(c1), collect(c2)
            if not A or not B: return 0.0
            inter = len(A & B) * 1.2
            return self._clamp01(inter / len(A | B))
        except Exception as e:
            logger.exception(f"문맥 호환성 계산 중 오류: {e}")
            return 0.0

    def _calculate_situation_based_adjustment(self, fi: Dict[str, Any], ti: Dict[str, Any], transition_type: str) -> float:
        try:
            adj = 0.0
            if transition_type == 'intrasentence':
                adj += 0.1
            fs = (fi.get('context_patterns') or {}).get('situations', {}) or {}
            ts = (ti.get('context_patterns') or {}).get('situations', {}) or {}
            matches, total = 0, (len(fs) + len(ts))
            for s1 in (fs.values() if isinstance(fs, dict) else []):
                if not isinstance(s1, dict): continue
                k1 = set(map(str, s1.get('keywords', []) or []))
                for s2 in (ts.values() if isinstance(ts, dict) else []):
                    if not isinstance(s2, dict): continue
                    k2 = set(map(str, s2.get('keywords', []) or []))
                    if k1 & k2: matches += 1
            if total > 0:
                adj += (matches / total) * 0.2
            return max(-0.3, min(0.3, adj))
        except Exception as e:
            logger.exception(f"상황 기반 보정값 계산 중 오류: {e}")
            return 0.0

    def _calculate_context_compatibility(self, from_contexts: Dict, to_contexts: Dict) -> float:
        try:
            if not from_contexts or not to_contexts: return 0.0
            total_keywords = 0; common = 0
            def kw_total(ctx: Dict) -> int:
                tot = 0
                for s in (ctx.values() if isinstance(ctx, dict) else []):
                    if isinstance(s, dict):
                        ks = s.get('keywords', [])
                        if isinstance(ks, list): tot += len(ks)
                return tot
            fc = from_contexts.get('situations', {}) if isinstance(from_contexts.get('situations'), dict) else {}
            tc = to_contexts.get('situations', {}) if isinstance(to_contexts.get('situations'), dict) else {}
            total_keywords = kw_total(fc) + kw_total(tc)
            for s1 in fc.values():
                if not isinstance(s1, dict): continue
                k1 = set(map(str, s1.get('keywords', []) or []))
                for s2 in tc.values():
                    if not isinstance(s2, dict): continue
                    k2 = set(map(str, s2.get('keywords', []) or []))
                    common += len(k1 & k2)
            return self._clamp01(common / total_keywords if total_keywords else 0.0)
        except Exception as e:
            logger.exception(f"[문맥 호환성 계산] 에러: {e}")
            return 0.0

    def _detect_emotion_conflicts(self, emotion_sequence: List[Dict[str, Any]], score_threshold: float = 0.2, include_sub_emotions: bool = True) -> List[Dict[str, Any]]:
        conflicts: List[Dict[str, Any]] = []
        conflicting_pairs = [('노', '희')]
        for idx, entry in enumerate(emotion_sequence):
            if not isinstance(entry, dict): continue
            em = entry.get('emotions', {}) if isinstance(entry.get('emotions'), dict) else {}
            valid = [eid for eid, val in em.items() if isinstance(val, (int, float)) and val >= score_threshold]
            if include_sub_emotions:
                sub = entry.get('sub_emotions', {}) if isinstance(entry.get('sub_emotions'), dict) else {}
                for sid, sval in sub.items():
                    if isinstance(sval, (int, float)) and sval >= score_threshold:
                        valid.append(sid)
            valid = set(valid)
            for a, b in conflicting_pairs:
                if a in valid and b in valid:
                    conflicts.append({"sentence_index": idx, "conflicting_emotions": [a, b], "type": "intrasentence_conflict"})
            # 세부감정 polarity 충돌(옵셔널)
            details = entry.get('sub_emotion_details', {}) if isinstance(entry.get('sub_emotion_details'), dict) else {}
            for sid, d1 in details.items():
                pol1 = (d1 or {}).get('polarity', '')
                for sid2, d2 in details.items():
                    if sid == sid2: continue
                    pol2 = (d2 or {}).get('polarity', '')
                    if pol1 == 'negative' and pol2 == 'positive':
                        conflicts.append({"sentence_index": idx, "conflicting_sub_emotions": [sid, sid2], "type": "sub_emotion_polarity_conflict"})
        return conflicts

    def _calculate_emotion_influence(self, emotion_pairs: List[Dict[str, Any]], emotions_data: Dict[str, Any], include_sub_emotions: bool = True) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for pair in emotion_pairs:
            fe, te = pair.get("from"), pair.get("to")
            if not fe or not te: continue
            base = 0.0
            fi = emotions_data.get(fe, {}) if isinstance(emotions_data.get(fe), dict) else {}
            ti = emotions_data.get(te, {}) if isinstance(emotions_data.get(te), dict) else {}

            # 기본 관계 강도
            if fe in self.default_emotion_relations:
                rel = self.default_emotion_relations[fe].get(te, {})
                base = float(rel.get("strength", 0.0))

            # 전이 패턴 우선
            patterns = (fi.get('emotion_transitions') or {}).get('patterns', []) if isinstance(fi, dict) else []
            for p in patterns:
                if not isinstance(p, dict): continue
                if p.get('from_emotion') == fe and p.get('to_emotion') == te:
                    ic = (p.get('transition_analysis', {}) or {}).get('intensity_change', 'medium')
                    tscore = {'high': 0.8, 'medium': 0.5, 'low': 0.3}.get(ic, 0.5)
                    base = max(base, tscore); break

            # 서브감정 보너스
            if include_sub_emotions:
                fm = (fi.get('metadata', {}) or {}); tm = (ti.get('metadata', {}) or {})
                fsub = fm.get("primary_category") not in ["희", "노", "애", "락"]
                tsub = tm.get("primary_category") not in ["희", "노", "애", "락"]
                if fsub or tsub:
                    cwx = {'subtle': 1.2, 'basic': 1.0, 'complex': 0.8}
                    base += 0.05 * cwx.get(fm.get('emotion_complexity', 'basic'), 1.0) * cwx.get(tm.get('emotion_complexity', 'basic'), 1.0)

            if base <= 0.0: base = 0.05
            out[f"{fe}_to_{te}"] = round(self._clamp01(base), 2)
        return out

    # ---------------------------- 연결/의존/지배 관계 ---------------------------- #
    def _map_emotion_connections(self, emotion_sequence: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        connections: Dict[str, List[str]] = {}
        try:
            for entry in (emotion_sequence or []):
                ems = list((entry.get('emotions') or {}).keys())
                for i in range(len(ems)):
                    for j in range(i + 1, len(ems)):
                        a, b = ems[i], ems[j]
                        connections.setdefault(a, []).append(b)
                        connections.setdefault(b, []).append(a)
            return {k: sorted(set(v)) for k, v in connections.items()}
        except Exception as e:
            logger.exception(f"[감정 연결 매핑] 에러: {e}")
            return {}

    def _analyze_emotion_dependencies(self, emotion_pairs: List[Dict[str, Any]], emotions_data: Dict[str, Any]) -> Dict[str, Any]:
        deps: Dict[str, Any] = {}
        try:
            for p in (emotion_pairs or []):
                rels = (emotions_data.get(p['from'], {}) or {}).get('emotion_relations', []) or []
                for r in rels:
                    if not isinstance(r, dict): continue
                    if r.get('target_emotion') == p['to']:
                        deps[f"{p['from']}_to_{p['to']}"] = {
                            'relationship_type': r.get('relationship_type'),
                            'strength': r.get('strength'),
                            'compatibility': r.get('compatibility')
                        }
                        break
            return deps
        except Exception as e:
            logger.exception(f"[감정 의존성 분석] 에러: {e}")
            return {}

    def dominant_relationships(
            self,
            emotion_sequence: List[Dict[str, Any]],
            emotions_data: Dict[str, Any],
            score_threshold: float = 0.2,
            include_sub_emotions: bool = True,
            top_n: int = 5
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        counts: Dict[Tuple[str, str], int] = {}
        scores: Dict[Tuple[str, str], float] = {}
        ctxs: Dict[Tuple[str, str], float] = {}
        intens: Dict[Tuple[str, str], float] = {}
        try:
            for i in range(len(emotion_sequence) - 1):
                cur, nxt = emotion_sequence[i], emotion_sequence[i + 1]
                ce, ne = dict(cur.get("emotions", {}) or {}), dict(nxt.get("emotions", {}) or {})

                if include_sub_emotions:
                    for sid, sval in (cur.get("sub_emotions", {}) or {}).items():
                        if isinstance(sval, (int, float)) and sval >= score_threshold:
                            ce[sid] = float(sval)
                    for sid, sval in (nxt.get("sub_emotions", {}) or {}).items():
                        if isinstance(sval, (int, float)) and sval >= score_threshold:
                            ne[sid] = float(sval)

                cur_valid = [(k, float(v)) for k, v in ce.items() if isinstance(v, (int, float)) and v >= score_threshold]
                nxt_valid = [(k, float(v)) for k, v in ne.items() if isinstance(v, (int, float)) and v >= score_threshold]

                for (fe, fs) in cur_valid:
                    for (te, ts) in nxt_valid:
                        if fe == te: continue
                        key = (fe, te)
                        counts[key] = counts.get(key, 0) + 1
                        scores[key] = scores.get(key, 0.0) + ((fs + ts) / 2.0)
                        ctxs[key] = ctxs.get(key, 0.0) + self._calculate_context_based_score(fe, te, str(cur.get('text', '') or ''), str(nxt.get('text', '') or ''))
                        intens[key] = max(intens.get(key, 0.0), abs(ts - fs))

            for key, cnt in counts.items():
                fe, te = key
                avg = scores.get(key, 0.0) / cnt
                ctx = ctxs.get(key, 0.0) / cnt
                inten = intens.get(key, 0.0)
                final = (avg * 0.4 + ctx * 0.3 + inten * 0.3)
                rtype = self._determine_relationship_type(final)
                cfac = self._calculate_complexity_factor(fe, te, emotions_data)
                final *= cfac
                if final >= score_threshold:
                    results.append({
                        "from": fe, "to": te, "strength": cnt,
                        "score": round(self._clamp01(final), 3),
                        "relationship_type": rtype,
                        "context_influence": round(self._clamp01(ctx), 3),
                        "intensity_change": round(self._clamp01(inten), 3)
                    })
            return sorted(results, key=lambda x: (x["strength"], x["score"]), reverse=True)[:top_n]
        except Exception as e:
            logger.error(f"[지배적 관계 분석] 오류 발생: {e}")
            return []

    def _calculate_complexity_factor(self, fe: str, te: str, emotions_data: Dict[str, Any]) -> float:
        try:
            fi, ti = emotions_data.get(fe, {}) or {}, emotions_data.get(te, {}) or {}
            fc, tc = (fi.get('metadata', {}) or {}).get('emotion_complexity', 'basic'), (ti.get('metadata', {}) or {}).get('emotion_complexity', 'basic')
            w = {'subtle': 1.2, 'basic': 1.0, 'complex': 0.8}
            return (w.get(fc, 1.0) + w.get(tc, 1.0)) / 2.0
        except Exception:
            return 1.0

    def _calculate_context_based_score(self, fe: str, te: str, cur_txt: str, nxt_txt: str) -> float:
        try:
            fi, ti = self.emotions_data.get(fe, {}) or {}, self.emotions_data.get(te, {}) or {}
            fk, tk = set(), set()
            for s in ((fi.get('context_patterns') or {}).get('situations', {}) or {}).values():
                if isinstance(s, dict): fk.update(map(str, s.get('keywords', []) or []))
            for s in ((ti.get('context_patterns') or {}).get('situations', {}) or {}).values():
                if isinstance(s, dict): tk.update(map(str, s.get('keywords', []) or []))
            if not fk or not tk: return 0.0
            cur_toks, nxt_toks = set(self.tokenize(cur_txt)), set(self.tokenize(nxt_txt))
            fm = len(fk & cur_toks); tm = len(tk & nxt_toks)
            return self._clamp01((fm + tm) / (len(fk) + len(tk)))
        except Exception as e:
            logger.exception("[문맥 점수 계산] 오류")
            return 0.0

    def _evaluate_relationship_strength(self, emotion_pairs: List[Dict[str, Any]], emotions_data: Dict[str, Any]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        try:
            for p in (emotion_pairs or []):
                fe, te = p['from'], p['to']
                fi, ti = emotions_data.get(fe, {}) or {}, emotions_data.get(te, {}) or {}
                ml = fi.get('ml_training_metadata', {}) if isinstance(fi, dict) else {}
                fc = (fi.get('metadata', {}) or {}).get('emotion_complexity', 'basic')
                conf = ml.get('confidence_thresholds', {'basic': 0.7, 'complex': 0.8, 'subtle': 0.9})
                threshold = float(conf.get(fc, 0.7))

                base = self._calculate_base_relationship_strength(fe, te, emotions_data)
                prof = self._analyze_profile_based_strength(fi, ti)
                ctx = self._calculate_context_based_strength(fi, ti, p.get('transition_type', ''))
                pers = self._calculate_persistence_factor(fi, ti, p.get('sentence_index', None))
                trans = self._analyze_transition_strength(fi, ti)
                sub = self._analyze_sub_emotion_impact(fi, ti)
                sit = self._calculate_situation_strength_modifier(fi, ti, emotions_data)

                w = {'base': 0.25, 'profile': 0.15, 'context': 0.15, 'persistence': 0.15, 'transition': 0.1, 'sub_emotion': 0.1, 'situation': 0.1}
                final = (base * w['base'] + prof * w['profile'] + ctx * w['context'] + pers * w['persistence'] +
                         trans * w['transition'] + sub * w['sub_emotion'] + sit * w['situation'])
                if final < 0.05: final = 0.05
                out[f"{fe}_to_{te}"] = round(self._clamp01(final), 3)
            return out
        except Exception as e:
            logger.exception(f"[_evaluate_relationship_strength] 오류 발생: {e}")
            return {}

    def _analyze_transition_strength(self, fi: Dict[str, Any], ti: Dict[str, Any]) -> float:
        try:
            base = 0.5
            pats = (fi.get('emotion_transitions') or {}).get('patterns', []) if isinstance(fi, dict) else []
            if not pats: return base
            w = {'direct': 1.0, 'gradual': 0.7, 'sudden': 0.5, 'cyclic': 0.3}
            mx = base
            for p in pats:
                if not isinstance(p, dict): continue
                ttype = p.get('type', 'gradual')
                weight = w.get(ttype, 0.5)
                ic = (p.get('transition_analysis', {}) or {}).get('intensity_change')
                if ic:
                    if ic == 'high': factor = 1.2
                    elif ic == 'low': factor = 0.8
                    else: factor = 1.0
                    mx = max(mx, weight * factor)
            return self._clamp01(mx)
        except Exception as e:
            logger.exception(f"전이 강도 분석 중 오류: {e}")
            return 0.5

    def _analyze_sub_emotion_impact(self, fi: Dict[str, Any], ti: Dict[str, Any]) -> float:
        try:
            fs, ts = fi.get('sub_emotions', {}), ti.get('sub_emotions', {})
            if not isinstance(fs, dict) or not isinstance(ts, dict) or not fs or not ts:
                return 0.0
            total, cnt = 0.0, 0
            cwx = {'subtle': 1.2, 'basic': 1.0, 'complex': 0.8}
            for _, f in fs.items():
                if not isinstance(f, dict): continue
                fprof = f.get('emotion_profile', {}) if isinstance(f.get('emotion_profile'), dict) else {}
                fk = set(fprof.get('core_keywords', []) or [])
                fc = (f.get('metadata', {}) or {}).get('emotion_complexity', 'basic')
                for _, t in ts.items():
                    if not isinstance(t, dict): continue
                    tprof = t.get('emotion_profile', {}) if isinstance(t.get('emotion_profile'), dict) else {}
                    tk = set(tprof.get('core_keywords', []) or [])
                    tc = (t.get('metadata', {}) or {}).get('emotion_complexity', 'basic')
                    kw_sim = len(fk & tk) / len(fk | tk) if (fk or tk) else 0.0
                    # related + situation 유사도
                    def flat(rel: Dict) -> Set[str]:
                        o = set()
                        for lst in (rel or {}).values():
                            if isinstance(lst, list): o.update(map(str, lst))
                        return o
                    fr, tr = flat(fprof.get('related_emotions', {}) or {}), flat(tprof.get('related_emotions', {}) or {})
                    rel_sim = len(fr & tr) / len(fr | tr) if (fr or tr) else 0.0

                    def ctx_kw(s: Dict) -> Set[str]:
                        o = set()
                        sits = (s.get('context_patterns') or {}).get('situations', {}) or {}
                        if isinstance(sits, dict):
                            for si in sits.values():
                                if isinstance(si, dict):
                                    o.update(map(str, si.get('keywords', []) or []))
                        return o
                    fctx, tctx = ctx_kw(f), ctx_kw(t)
                    ctx_sim = len(fctx & tctx) / len(fctx | tctx) if (fctx or tctx) else 0.0

                    cfac = (cwx.get(fc, 1.0) + cwx.get(tc, 1.0)) / 2
                    total += (kw_sim * 0.4 + rel_sim * 0.3 + ctx_sim * 0.3) * cfac; cnt += 1
            return self._clamp01((total / cnt) if cnt else 0.0)
        except Exception as e:
            logger.exception(f"세부 감정 영향도 분석 중 오류: {e}")
            return 0.0

    def _calculate_base_relationship_strength(self, fe: str, te: str, emotions_data: Dict[str, Any]) -> float:
        try:
            base = float(self.default_emotion_relations.get(fe, {}).get(te, {}).get('strength', 0.5))
            fi = emotions_data.get(fe, {}) or {}
            for p in ((fi.get('emotion_transitions') or {}).get('patterns', []) or []):
                if not isinstance(p, dict): continue
                if p.get('from_emotion') == fe and p.get('to_emotion') == te:
                    ic = (p.get('transition_analysis', {}) or {}).get('intensity_change', 'medium')
                    w = {'high': 1.2, 'medium': 1.0, 'low': 0.8}.get(ic, 1.0)
                    base *= w; break
            return self._clamp01(base)
        except Exception as e:
            logger.exception(f"기본 관계 강도 계산 중 오류: {e}")
            return 0.5

    def _analyze_profile_based_strength(self, fi: Dict[str, Any], ti: Dict[str, Any]) -> float:
        try:
            fk = set(((fi.get('emotion_profile') or {}).get('core_keywords', []) or []))
            tk = set(((ti.get('emotion_profile') or {}).get('core_keywords', []) or []))
            kw = (len(fk & tk) / len(fk | tk)) if (fk or tk) else 0.0
            def rel_strength(fr: Dict, tr: Dict) -> float:
                FA, TA = set(), set()
                for v in (fr or {}).values():
                    if isinstance(v, list): FA.update(map(str, v))
                for v in (tr or {}).values():
                    if isinstance(v, list): TA.update(map(str, v))
                if not FA and not TA: return 0.0
                return (len(FA & TA) / len(FA | TA)) if (FA or TA) else 0.0
            rel = rel_strength((fi.get('emotion_profile') or {}).get('related_emotions', {}),
                               (ti.get('emotion_profile') or {}).get('related_emotions', {}))
            return self._clamp01(kw * 0.6 + rel * 0.4)
        except Exception as e:
            logger.exception(f"프로필 기반 강도 분석 중 오류: {e}")
            return 0.0

    def _calculate_related_emotions_strength(self, from_related: Dict[str, List[str]], to_related: Dict[str, List[str]]) -> float:
        try:
            FA, TA = set(), set()
            for lst in (from_related or {}).values():
                if isinstance(lst, list): FA.update(map(str, lst))
            for lst in (to_related or {}).values():
                if isinstance(lst, list): TA.update(map(str, lst))
            if not FA or not TA: return 0.0
            return self._clamp01(len(FA & TA) / len(FA | TA))
        except Exception as e:
            logger.exception(f"관련 감정 강도 계산 중 오류: {e}")
            return 0.0

    def _calculate_persistence_factor(self, fi: Dict[str, Any], ti: Dict[str, Any], idx: Optional[Any]) -> float:
        try:
            base = 0.5
            if isinstance(idx, tuple) and len(idx) == 2:
                if (idx[1] - idx[0]) <= 1: base += 0.2
            elif isinstance(idx, int):
                base += 0.1
            fp, tp = fi.get('emotion_progression', {}), ti.get('emotion_progression', {})
            if isinstance(fp, dict) and isinstance(tp, dict) and fp and tp:
                base += self._calculate_progression_similarity(fp, tp) * 0.3
            return self._clamp01(base)
        except Exception as e:
            logger.exception(f"지속성 요인 계산 중 오류: {e}")
            return 0.5

    def _calculate_progression_similarity(self, fp: Dict[str, Any], tp: Dict[str, Any]) -> float:
        try:
            fs, ts = set((fp or {}).keys()), set((tp or {}).keys())
            if not fs or not ts: return 0.0
            return round(self._clamp01(len(fs & ts) / len(fs | ts)), 3)
        except Exception as e:
            logger.exception(f"[진행 유사도 계산] 에러: {e}")
            return 0.0

    def _calculate_situation_strength_modifier(self, fi: Dict[str, Any], ti: Dict[str, Any], emotions_data: Dict[str, Any]) -> float:
        try:
            mod = 0.0
            fs = (fi.get('context_patterns') or {}).get('situations', {}) or {}
            ts = (ti.get('context_patterns') or {}).get('situations', {}) or {}
            if isinstance(fs, dict) and isinstance(ts, dict) and fs and ts:
                mod += self._calculate_situation_similarity(fs, ts) * 0.2
            seq = emotions_data.get('emotion_sequence', []) if isinstance(emotions_data, dict) else []
            if isinstance(seq, list) and seq:
                mod += self._analyze_sequence_impact(fi, ti, seq) * 0.3
            return max(-0.3, min(0.3, mod))
        except Exception as e:
            logger.exception(f"상황 기반 수정 요인 계산 중 오류: {e}")
            return 0.0

    def _calculate_situation_similarity(self, fs: Dict[str, Any], ts: Dict[str, Any]) -> float:
        try:
            if not isinstance(fs, dict) or not isinstance(ts, dict): return 0.0
            def collect(d: Dict) -> Set[str]:
                out = set()
                for s in d.values():
                    if not isinstance(s, dict): continue
                    out.update(map(str, s.get('keywords', []) or []))
                    out.update(map(str, s.get('variations', []) or []))
                    ex = s.get('examples', [])
                    if isinstance(ex, list):
                        for e in ex:
                            if isinstance(e, str): out.update(self.tokenize(e))
                return out
            A, B = collect(fs), collect(ts)
            if not A or not B: return 0.0
            return round(self._clamp01(len(A & B) / len(A | B)), 3)
        except Exception as e:
            logger.exception(f"[상황 유사도 계산] 오류: {e}")
            return 0.0

    def _analyze_sequence_impact(self, fi: Dict[str, Any], ti: Dict[str, Any], seq: List[Dict[str, Any]]) -> float:
        try:
            fid = (fi.get('metadata', {}) or {}).get('emotion_id', '')
            tid = (ti.get('metadata', {}) or {}).get('emotion_id', '')
            if not fid or not tid or fid == tid: return 0.0
            tot = adj = same = 0
            for idx, entry in enumerate(seq):
                if not isinstance(entry, dict): continue
                em = entry.get('emotions', {}) if isinstance(entry.get('emotions'), dict) else {}
                if fid in em and tid in em: same += 1
                if idx < len(seq) - 1:
                    nxt = seq[idx + 1]
                    nx = nxt.get('emotions', {}) if isinstance(nxt.get('emotions'), dict) else {}
                    if fid in em and tid in nx: adj += 1
                    if tid in em and fid in nx: adj += 1
                tot += 1
            if tot == 0: return 0.0
            return round(self._clamp01((same * 0.15 + adj * 0.1) / tot), 3)
        except Exception as e:
            logger.exception(f"[시퀀스 영향도 분석] 오류: {e}")
            return 0.0

    def _calculate_context_based_strength(self, fi: Dict[str, Any], ti: Dict[str, Any], transition_type: str) -> float:
        try:
            if not isinstance(fi, dict) or not isinstance(ti, dict): return 0.0
            fk, tk = set(), set()
            for s in ((fi.get('context_patterns') or {}).get('situations', {}) or {}).values():
                if isinstance(s, dict): fk.update(map(str, s.get('keywords', []) or []))
            for s in ((ti.get('context_patterns') or {}).get('situations', {}) or {}).values():
                if isinstance(s, dict): tk.update(map(str, s.get('keywords', []) or []))
            if not fk or not tk: return 0.0
            base = len(fk & tk) / len(fk | tk)
            if transition_type == 'intrasentence': base *= 1.2
            return self._clamp01(base)
        except Exception as e:
            logger.exception(f"문맥 기반 강도 계산 중 오류: {e}")
            return 0.0

# ================================
# Analyzer Registry (캐시/공유)
# ================================
class _AnalyzerRegistry:
    """(emotions_data, config) 조합으로 분석기 번들을 캐시/공유합니다."""
    def __init__(self):
        self._cache = {}
        self._lock = threading.RLock()

    @staticmethod
    def _cfg_sig(cfg):
        try:
            return json.dumps(cfg or {}, sort_keys=True, ensure_ascii=False)
        except Exception:
            return str(type(cfg))

    @staticmethod
    def _data_sig(emotions_data: Dict[str, Any], use_content_hash: bool = False) -> str:
        if use_content_hash:
            try:
                payload = json.dumps(emotions_data or {}, sort_keys=True, ensure_ascii=False, separators=(',', ':'))
                import hashlib
                return hashlib.sha1(payload.encode('utf-8')).hexdigest()
            except Exception:
                # 해시 실패 시 id 기반으로 폴백
                pass
        return f"id:{id(emotions_data)}"

    def get(self, emotions_data: Dict[str, Any], config: Optional[Dict[str, Any]] = None, force_rebuild: bool = False):
        # 캐시 키 구성: (emotions_data 시그니처, config 시그니처)
        try:
            use_hash = bool(((config or {}).get('cache') or {}).get('use_content_hash', False))
        except Exception:
            use_hash = False
        data_sig = self._data_sig(emotions_data, use_content_hash=use_hash)
        cfg_sig = self._cfg_sig(config)
        key = (data_sig, cfg_sig)

        with self._lock:
            if not force_rebuild and key in self._cache:
                return self._cache[key]

        # 1) 공통 구성요소
        validator = EmotionValidatorExtended()
        try:
            is_valid, issues = validator.validate_emotion_structure(emotions_data)
            if not is_valid:
                logger.warning("[Pipeline] 라벨링 구조 경고(진행은 계속): %s", issues)
        except Exception as e:
            logger.warning("[Pipeline] 라벨링 검증 중 오류: %s", e)

        text_analyzer = TextAnalyzer(emotions_data)

        # 2) 분석기들 - 동일 emotions_data / text_analyzer 공유
        progression = EmotionProgressionRelationshipAnalyzer(emotions_data)
        relationship = EmotionRelationshipAnalyzer(
            config=config, emotions_data=emotions_data, text_analyzer=text_analyzer
        )

        # 3) 소셜 그래프: 클래스가 있으면 인스턴스, 없으면 함수 폴백 (config 전달)
        social_graph = None
        if 'SocialEmotionGraph' in globals():
            try:
                social_graph = SocialEmotionGraph(emotions_data, text_analyzer=text_analyzer, config=config)  # noqa
            except Exception:
                social_graph = None

        bundle = {
            "validator": validator,
            "text_analyzer": text_analyzer,
            "progression": progression,
            "relationship": relationship,
            "social_graph": social_graph,  # 없을 수 있음(None)
        }

        with self._lock:
            self._cache[key] = bundle
        return bundle

    def clear(self):
        with self._lock:
            n = len(self._cache)
            self._cache.clear()
            return n

# 전역 레지스트리
_REGISTRY = _AnalyzerRegistry()


# =============================================================================
# Independent Functions (공개 API) — 안전/일관 버전
#  - 레지스트리에서 공유 인스턴스를 받아 "바운드 메서드"로 호출
#  - 실패 시 일관된 형태로 에러 반환
# =============================================================================
def prewarm_pipeline(emotions_data: Dict[str, Any],
                     config: Optional[Dict[str, Any]] = None,
                     *, force_rebuild: bool = False) -> Dict[str, Any]:
    """1회 사전 초기화(캐시/토크나이저/분석기 준비). 성공/메타 반환."""
    try:
        _REGISTRY.get(emotions_data, config=config, force_rebuild=force_rebuild)
        return {"ok": True, "cache_size": len(getattr(_REGISTRY, "_cache", {}))}
    except Exception as e:
        logger.error("[prewarm_pipeline] %s", e, exc_info=True)
        return {"ok": False, "error": str(e)}

def reset_pipeline_cache() -> Dict[str, Any]:
    """레지스트리 캐시 비우기(실험/테스트 간 교체 시). 성공/메타 반환."""
    try:
        cleared = _REGISTRY.clear()
        return {"ok": True, "cleared": int(cleared)}
    except Exception as e:
        logger.error("[reset_pipeline_cache] %s", e, exc_info=True)
        return {"ok": False, "error": str(e)}

def analyze_emotion_relationships(text: str,
                                  emotions_data: Dict[str, Any],
                                  config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    문장→단계→감정쌍까지 고려한 관계 분석의 메인 엔트리.
    반환: {"relationship_analysis": bool, ...} (기존 키 유지)
    """
    try:
        bundle = _REGISTRY.get(emotions_data, config=config)
        rel = bundle["relationship"]  # EmotionRelationshipAnalyzer (bound)
        return rel.analyze_emotion_relationships(text, emotions_data)
    except Exception as e:
        logger.error("[analyze_emotion_relationships] %s", e, exc_info=True)
        return {"relationship_analysis": False, "error": str(e)}

def analyze_emotion_progression(text: str,
                                emotions_data: Dict[str, Any],
                                config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    발단→전개→절정→여파 스테이지 매칭/스코어.
    반환: stage별 매칭 정보 딕셔너리(기존 형식 유지).
    """
    if not isinstance(text, str) or not text.strip():
        return {}
    try:
        bundle = _REGISTRY.get(emotions_data, config=config)
        prog = bundle["progression"]  # EmotionProgressionRelationshipAnalyzer
        return prog.analyze_stage_relationships(text, emotions_data)
    except Exception as e:
        logger.error("[analyze_emotion_progression] %s", e, exc_info=True)
        return {}

def analyze_relationship_expansion(text: str,
                                   emotions_data: Dict[str, Any],
                                   threshold: float = 0.2,
                                   config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    문장 단위 후보 감정에 대해 트리거/세부감정/문맥 점수 확장.
    threshold: 상황 점수 하한(기본 0.2).
    """
    if not isinstance(text, str) or not text.strip():
        return {}
    try:
        bundle = _REGISTRY.get(emotions_data, config=config)
        prog = bundle["progression"]
        return prog.analyze_relationship_expansion(text, emotions_data, threshold)
    except Exception as e:
        logger.error("[analyze_relationship_expansion] %s", e, exc_info=True)
        return {}

def analyze_emotion_compatibility(emotion_pairs: List[Dict[str, Any]],
                                  emotions_data: Dict[str, Any],
                                  config: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    """
    감정쌍 호환성 스코어 맵 산출.
    입력: emotion_pairs = [{"from": ..., "to": ..., ...}, ...]
    반환: {"희_to_락": 0.83, ...}
    """
    try:
        bundle = _REGISTRY.get(emotions_data, config=config)
        rel = bundle["relationship"]
        return rel.analyze_emotion_compatibility(emotion_pairs, emotions_data)
    except Exception as e:
        logger.error("[analyze_emotion_compatibility] %s", e, exc_info=True)
        return {}

def identify_emotion_pairs(emotion_sequence: List[Dict[str, Any]],
                           emotions_data: Dict[str, Any],
                           config: Optional[Dict[str, Any]] = None,
                           *,
                           include_sub_emotions: bool = True,
                           score_threshold: float = 0.2) -> List[Dict[str, Any]]:
    """
    시퀀스→감정쌍 생성(동문장/인접문장).
    옵션:
      - include_sub_emotions: 서브 감정 포함 여부
      - score_threshold: 감정 점수 하한
    """
    try:
        bundle = _REGISTRY.get(emotions_data, config=config)
        rel = bundle["relationship"]
        return rel._identify_emotion_pairs(  # noqa
            emotion_sequence, emotions_data,
            include_sub_emotions=include_sub_emotions,
            score_threshold=score_threshold
        )
    except Exception as e:
        logger.error("[identify_emotion_pairs] %s", e, exc_info=True)
        return []

# ───────── 추가 유틸(선택) — 소셜 그래프 & 일괄 요약 ─────────

def build_social_emotion_graph(text: str,
                               emotions_data: Dict[str, Any],
                               entities: Optional[List[Dict[str, Any]]] = None,
                               max_depth: int = 2,
                               config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    대상 간 소셜 감정 그래프 구축(주체↔타깃↔기타 노드, 엣지 가중 포함).
    """
    try:
        bundle = _REGISTRY.get(emotions_data, config=config)
        sg = bundle.get("social_graph")
        if sg is None:
            # 클래스 사용 불가 시 안전 폴백
            return analyze_social_emotion_graph(text, emotions_data, entities=entities, max_depth=max_depth)
        return sg.analyze(text, entities=entities, max_depth=max_depth)
    except Exception as e:
        logger.error("[build_social_emotion_graph] %s", e, exc_info=True)
        return {"nodes": [], "edges": [], "metrics": {}, "error": str(e)}

def compute_pairs_compatibility_strength(emotion_sequence: List[Dict[str, Any]],
                                         emotions_data: Dict[str, Any],
                                         config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    감정쌍 생성 → 호환성/관계강도 일괄 계산(파이프라인 한 번 호출).
    """
    try:
        bundle = _REGISTRY.get(emotions_data, config=config)
        rel = bundle["relationship"]
        pairs = rel._identify_emotion_pairs(emotion_sequence, emotions_data)  # noqa
        compatibility = rel._analyze_emotion_compatibility(pairs, emotions_data)  # noqa
        strength = rel._evaluate_relationship_strength(pairs, emotions_data)
        return {"pairs": pairs, "compatibility": compatibility, "relationship_strength": strength}
    except Exception as e:
        logger.error("[compute_pairs_compatibility_strength] %s", e, exc_info=True)
        return {"pairs": [], "compatibility": {}, "relationship_strength": {}}



# =============================================================================
# Main Function
# =============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _setup_console_logger(name: str = "emotion_main", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # 중복 핸들러 방지
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(ch)
    return logger
_mainlog = _setup_console_logger()

def split_text_into_sentences(text: str) -> List[str]:
    if not text:
        return []
    s = re.split(r'(?<=[.!?])\s+|[\n\r]+', text.strip())
    return [x for x in s if x]

def _ensure_file_logging(out_dir: str,
                         filename: str = "emotion_relationship_analyzer.log",
                         level: int = logging.INFO,
                         rotate: bool = True,
                         max_bytes: int = 2 * 1024 * 1024,
                         backups: int = 3) -> str:
    """
    루트 로거에 파일 핸들러를 1회만 장착.
    rotate=True면 크기 기준 로테이팅.
    """
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, filename)
    root = logging.getLogger()  # 루트에 달아야 하위 모듈 logger도 함께 파일로 기록됨
    # 중복 장착 방지
    for h in root.handlers:
        if isinstance(h, logging.FileHandler):
            try:
                if os.path.abspath(getattr(h, 'baseFilename', '')) == os.path.abspath(log_path):
                    return log_path
            except Exception:
                continue
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    if rotate:
        fh = RotatingFileHandler(log_path, maxBytes=max_bytes, backupCount=backups, encoding="utf-8")
    else:
        fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt)
    root.addHandler(fh)
    return log_path



def _ensure_min_labeling_skeleton(data: Dict[str, Any]) -> Dict[str, Any]:
    """ EMOTIONS.json이 비어있거나 핵심 키가 없을 때 최소 동작을 위한 스켈레톤을 채워줍니다.
    실제 프로젝트 라벨링이 있으면 그대로 사용되고, 없을 때만 보강합니다. """
    if not isinstance(data, dict):
        data = {}
    for cat in ['희', '노', '애', '락']:
        if cat not in data or not isinstance(data.get(cat), dict):
            data[cat] = {
                "metadata": {"emotion_id": cat, "primary_category": cat, "emotion_complexity": "basic"},
                "emotion_profile": {
                    "core_keywords": [],
                    "intensity_levels": {"intensity_examples": {}}
                },
                "context_patterns": {"situations": {}}
            }
    return data

def load_emotions_data(path: Optional[str] = None, *, allow_fallback: bool = True) -> Dict[str, Any]:
    """ EMOTIONS.json 로드. 없으면 스켈레톤 폴백(allow_fallback=True). """
    emotions_path = path or os.path.join(BASE_DIR, 'EMOTIONS.json')
    if os.path.isfile(emotions_path):
        _mainlog.info(f"[load] EMOTIONS.json: {emotions_path}")
        with open(emotions_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return _ensure_min_labeling_skeleton(data)
    if allow_fallback:
        _mainlog.warning(f"[load] EMOTIONS.json 미발견. 최소 스켈레톤으로 폴백합니다. (경로 시도: {emotions_path})")
        return _ensure_min_labeling_skeleton({})
    raise FileNotFoundError(f"EMOTIONS.json not found at: {emotions_path}")

def create_emotion_sequence(
    text: str,
    analyzer: "TextAnalyzer",
    *,
    base_threshold: float = 0.20
) -> List[Dict[str, Any]]:
    """ 문장→대표감정 스코어→임계치 상회 항목만 고르고 시퀀스 구성.
    강도(intensity.score)는 최대 대표감정 점수로 간단 폴백합니다. """
    PRIMARY = {'희', '노', '애', '락'}
    seq: List[Dict[str, Any]] = []
    for sent in split_text_into_sentences(text):
        try:
            scores = analyzer.analyze_emotion(sent)
            if not isinstance(scores, dict):
                continue
            dom: Dict[str, float] = {}
            for k, v in scores.items():
                if k == "_intrasentence_candidates":
                    continue
                try:
                    fv = float(v)
                except Exception:
                    continue
                if k in PRIMARY and fv > base_threshold:
                    dom[k] = fv
            if dom:
                seq.append({
                    "text": sent,
                    "emotions": dom,
                    "stage": "unknown",
                    "intensity": {"score": max(dom.values())}
                })
        except Exception as e:
            _mainlog.warning(f"[sequence] 문장 처리 실패: {e}")
    return seq

def _annotate_sequence_with_stages(
    seq: list,
    stage_obj: dict | list,
    *,
    override_unknown_only: bool = True
) -> None:
    """ progression 결과를 받아 각 문장 seg['stage']를 표준 라벨로 주석.
    stage_obj는 dict({stage_by_sentence:[...]}) 또는 list([...]) 모두 허용. """
    def _canon(s: str) -> str:
        return _STAGE_CANON_MAP.get(str(s or "").lower(), "beginning")
    # stage 리스트 뽑기
    if isinstance(stage_obj, dict) and isinstance(stage_obj.get("stage_by_sentence"), list):
        stages = list(stage_obj["stage_by_sentence"])
    elif isinstance(stage_obj, list):
        stages = list(stage_obj)
    else:
        stages = []
    # 길이 보정
    n = len(seq) if isinstance(seq, list) else 0
    if len(stages) < n:
        stages += ["beginning"] * (n - len(stages))
    # 반영
    for i, seg in enumerate(seq or []):
        if not isinstance(seg, dict):
            continue
        new_stage = _canon(stages[i]) if i < len(stages) else "beginning"
        cur = seg.get("stage")
        if not override_unknown_only or (not cur or str(cur).lower() in ("unknown", "none")):
            seg["stage"] = new_stage

def _ensure_stage_consistency_in_result(result: dict) -> None:
    """ result['progression']['stage_by_sentence'] → result['emotion_sequence'][i]['stage'] 동기화. """
    try:
        seq = result.get("emotion_sequence") or []
        prog = result.get("progression") or {}
        stages = prog.get("stage_by_sentence") or []
        if not seq or not stages:
            return
        if len(stages) < len(seq):
            stages += ["beginning"] * (len(seq) - len(stages))
        for i, seg in enumerate(seq):
            if not isinstance(seg, dict):
                continue
            if not seg.get("stage") or str(seg.get("stage")).lower() in ("unknown", "none"):
                seg["stage"] = stages[i]
    except Exception:
        pass

# === Stage fallback heuristics ==============================================
def _fallback_stage_annotation(seq: list[dict], text: str) -> None:
    """
    stage_result가 비었거나 'unknown'인 세그먼트에 대해
    (문장 위치/강도/표지어) 기반으로 전개 단계를 보정합니다.
    seq를 제자리에서 수정합니다.
    """
    if not isinstance(seq, list) or not seq:
        return
    n = len(seq)
    # 절정 후보: 강도 최대인 문장
    peak_idx, max_int = None, -1.0
    for i, seg in enumerate(seq):
        try:
            s = float((seg.get("intensity") or {}).get("score", 0.0))
        except Exception:
            s = 0.0
        if s >= max_int:
            max_int = s
            peak_idx = i
    # 한국어 전개 표지어(가벼운 규칙)
    TRIG = ("처음", "시작", "문득", "오늘", "갑자기")
    DEV  = ("점점", "점차", "차츰", "서서히", "더욱", "계속")
    PEAK = ("특히", "가장", "정말", "너무", "매우", "절정", "무엇보다")
    AFTER= ("결국", "그래서", "이후", "그 뒤", "다음", "마침내", "돌이켜")
    for i, seg in enumerate(seq):
        st = seg.get("stage")
        if st and st != "unknown":
            continue
        sent = seg.get("sentence") or seg.get("text") or ""
        score = {"beginning": 0.0, "development": 0.0, "climax": 0.0, "aftermath": 0.0}
        if any(t in sent for t in TRIG):  score["beginning"]   += 0.5
        if any(t in sent for t in DEV):   score["development"] += 0.4
        if any(t in sent for t in PEAK):  score["climax"]      += 0.6
        if any(t in sent for t in AFTER): score["aftermath"]   += 0.5
        # 위치 보정: 첫 문장/마지막 문장 가중
        if i == 0:     score["beginning"] += 0.25
        if i == n - 1: score["aftermath"] += 0.25
        # 강도 보정: 강도 최댓값 문장은 절정에 가산
        if peak_idx is not None and i == peak_idx: score["climax"] += 0.35
        seg["stage"] = max(score, key=score.get)


def _build_fallback_progression(seq: list[dict]) -> dict:
    """ progression 결과가 비었을 때 최소 요약을 만들어 채웁니다. """
    if not isinstance(seq, list) or not seq:
        return {"stage_by_sentence": [], "signals": {}}

    stages = [(seg.get("stage") or "unknown") for seg in seq]
    total  = len(stages)
    counts = {k: stages.count(k) for k in ("beginning", "development", "climax", "aftermath")}
    signals = {k: round((counts.get(k, 0) / total), 3) for k in counts}
    return {
        "stage_by_sentence": stages,
        "signals": signals,
        "note": "fallback_inferred"
    }


def convert_keys_to_str(data: Any) -> Any:
    """ dict의 tuple key 등을 문자열로 안전 변환 """
    if isinstance(data, dict):
        out = {}
        for k, v in data.items():
            nk = f"{k[0]}->{k[1]}" if isinstance(k, tuple) else k
            out[nk] = convert_keys_to_str(v)
        return out
    if isinstance(data, list):
        return [convert_keys_to_str(x) for x in data]
    return data

# ── Stage label normalizer ─────────────────────────────────────────────────────
_STAGE_CANON_MAP = {
    "trigger": "beginning",
    "development": "development",
    "peak": "climax",
    "aftermath": "aftermath",
    "beginning": "beginning",
    "climax": "climax",
}
def _normalize_stage_output(stage_obj: dict | list | None, *, n_sentences: int | None = None) -> dict:
    """ progression 결과(혹은 유사 구조)를 받아 표준 키로 정규화합니다. """
    # 기본 틀
    out = {
        "stage_by_sentence": [],
        "signals": {"beginning": 0.0, "development": 0.0, "climax": 0.0, "aftermath": 0.0}
    }
    if stage_obj is None:
        # 문장 수가 있으면 모두 beginning으로 채워 가시성 확보
        if isinstance(n_sentences, int) and n_sentences > 0:
            out["stage_by_sentence"] = ["beginning"] * n_sentences
            out["signals"]["beginning"] = 1.0
        return out

    # 1) stage_by_sentence (list 또는 dict 내부에서 추출)
    stage_list: list[str] = []
    if isinstance(stage_obj, dict):
        if isinstance(stage_obj.get("stage_by_sentence"), list):
            stage_list = stage_obj["stage_by_sentence"]
        elif isinstance(stage_obj.get("stages"), list):  # 혹시 다른 키를 쓰는 경우
            stage_list = stage_obj["stages"]
    elif isinstance(stage_obj, list):
        stage_list = stage_obj

    # 표준화 + 길이 보정
    def _canon(s: str) -> str:
        return _STAGE_CANON_MAP.get((s or "").lower(), "beginning")

    if stage_list:
        stage_list = [_canon(s) for s in stage_list]
    elif isinstance(n_sentences, int) and n_sentences > 0:
        stage_list = ["beginning"] * n_sentences
    out["stage_by_sentence"] = stage_list

    # 2) signals 표준화 (키 합산)
    raw_signals = {}
    if isinstance(stage_obj, dict) and isinstance(stage_obj.get("signals"), dict):
        raw_signals = dict(stage_obj["signals"])

    for k, v in (raw_signals.items() if raw_signals else []):
        try:
            out["signals"][_canon(str(k))] += float(v)
        except Exception:
            continue

    # 신호가 전부 0이면, stage_by_sentence 기반으로 beginning 1.0로 기본 표기
    if sum(out["signals"].values()) == 0.0:
        if out["stage_by_sentence"]:
            out["signals"]["beginning"] = 1.0

    # 3) 정규화(선택): 합이 0이 아니면 1.0로 스케일
    total = sum(out["signals"].values())
    if total > 0:
        for k in out["signals"]:
            out["signals"][k] = round(out["signals"][k] / total, 3)

    return out

def run_main_once(
    text: str,
    emotions_data: Dict[str, Any],
    *,
    include_graph: bool = True,
    json_out_dir: Optional[str] = None,
    fast_mode: bool = False,
    max_topk: int = 5,
    max_nodes: int = 20,
    max_edges: int = 80
) -> Dict[str, Any]:
    """
    문장→단계→감정쌍→호환/충돌/영향/강도 + 소셜 그래프까지 한 번에.
    - 스테이지(발단/전개/절정/여파) 표준 라벨로 일원화
    - 콘솔 로그/JSON 모두 동일한 스테이지 표기 사용
    """
    t0 = time.perf_counter()

    # 0) Analyzer 준비(텍스트 분석기/관계 분석기 초기화)
    ta = TextAnalyzer(emotions_data)
    rel = EmotionRelationshipAnalyzer(emotions_data=emotions_data, text_analyzer=ta)
    prog = EmotionProgressionRelationshipAnalyzer(emotions_data)

    with _STAGE_TIMER.timing("tokenize"):
        seq = create_emotion_sequence(text, ta)
    emotions_data = dict(emotions_data)  # shallow copy
    emotions_data["emotion_sequence"] = seq

    token_count = sum(len((entry.get("text") or "").split()) for entry in seq)
    sentence_count = len(seq)
    short_text = token_count < 50 and sentence_count < 3
    fast_trigger = bool(fast_mode or short_text)

    prev_log_levels = []
    log_messages: List[str] = []

    def _emit(msg: str) -> None:
        if fast_trigger:
            log_messages.append(msg)
        else:
            _mainlog.info(msg)

    if fast_trigger and os.environ.get("ERA_FAST_SILENT", "1") == "1":
        try:
            prev_log_levels.append((_mainlog, _mainlog.level))
            _mainlog.setLevel(logging.WARNING)
        except Exception:
            pass
        for handler in getattr(_mainlog, "handlers", []) or []:
            try:
                prev_log_levels.append((handler, handler.level))
                handler.setLevel(logging.WARNING)
            except Exception:
                pass

    with _STAGE_TIMER.timing("progression"):
        stage_raw = prog.analyze_stage_relationships(text, emotions_data)
        stage_norm = _normalize_stage_output(stage_raw, n_sentences=len(seq))

    try:
        _annotate_sequence_with_stages(seq, stage_norm)  # in-place
    except Exception:
        if isinstance(stage_norm, dict):
            _annotate_sequence_with_stages(seq, stage_norm.get("stage_by_sentence", []))

    with _STAGE_TIMER.timing("relationships"):
        main = rel.analyze_emotion_relationships(text, emotions_data)

    with _STAGE_TIMER.timing("pair_analysis"):
        pairs = rel._identify_emotion_pairs(seq, emotions_data)
        compatibility = rel._analyze_emotion_compatibility(pairs, emotions_data)
        if fast_trigger and isinstance(compatibility, dict) and len(compatibility) > max_topk:
            compatibility = dict(heapq.nlargest(max_topk, compatibility.items(), key=lambda kv: kv[1]))
        strength = rel._evaluate_relationship_strength(pairs, emotions_data)
        if fast_trigger and isinstance(strength, dict) and len(strength) > max_topk:
            strength = dict(heapq.nlargest(max_topk, strength.items(), key=lambda kv: kv[1]))
        if fast_trigger and isinstance(pairs, list) and len(pairs) > max_topk and isinstance(compatibility, dict):
            keep_keys = set(compatibility.keys())
            if keep_keys:
                filtered_pairs = [p for p in pairs if f"{p.get('from')}_to_{p.get('to')}" in keep_keys]
                if filtered_pairs:
                    pairs = filtered_pairs

    run_graph = include_graph and not fast_trigger
    with _STAGE_TIMER.timing("graph"):
        graph = analyze_social_emotion_graph(text, emotions_data) if run_graph else None
        if graph and fast_trigger and isinstance(graph, dict):
            graph['nodes'] = graph.get('nodes', [])[:max_nodes]
            graph['edges'] = graph.get('edges', [])[:max_edges]

    result = {
        "ok": True,
        "emotion_sequence": seq,
        "analysis": main,
        "progression": {
            "stage_by_sentence": stage_norm["stage_by_sentence"],
            "signals": stage_norm["signals"],
        },
        "pairs": pairs,
        "compatibility": compatibility,
        "relationship_strength": strength,
        "social_graph": graph,
        "meta": {
            "elapsed_sec": round(time.perf_counter() - t0, 4),
            "fast_mode": fast_trigger,
        },
    }
    rel_metrics = getattr(rel, '_last_relationship_metrics', {})
    if isinstance(rel_metrics, dict) and rel_metrics:
        result['meta'].update({
            'context_similarity': round(rel_metrics.get('context_similarity', 0.0), 3),
            'keyword_similarity': round(rel_metrics.get('keyword_similarity', 0.0), 3),
            'pattern_signal': round(rel_metrics.get('pattern_signal', 0.0), 3),
            'pre_gate_score': round(rel_metrics.get('pre_gate_score', 0.0), 3),
            'gated_score': round(rel_metrics.get('total_score', 0.0), 3),
        })

    stage_summary = _STAGE_TIMER.summary()
    if stage_summary:
        result['meta']['stage_summary'] = {
            stage: {
                'p50': round(vals['p50'], 4),
                'p95': round(vals['p95'], 4),
            }
            for stage, vals in stage_summary.items()
        }
        timing_parts = [f"{stage}:p50={vals['p50']:.4f}s p95={vals['p95']:.4f}s" for stage, vals in stage_summary.items()]
        _emit('[timing] ' + ' | '.join(timing_parts))

    _ensure_stage_consistency_in_result(result)
    graph_edges = len(graph.get('edges', [])) if isinstance(graph, dict) else 0
    _emit(f"[done] 문장 수={len(seq)}, 후보 수={len(pairs)}, 그래프간선={graph_edges}")
    if compatibility:
        top_count = min(len(compatibility), max_topk)
        topc = heapq.nlargest(top_count, compatibility.items(), key=lambda kv: kv[1])
        _emit(f"[top compatibility] {topc}")
    if isinstance(main, dict) and main.get("conflicts"):
        _emit(f"[conflicts] {len(main['conflicts'])}건 감지")
    _emit(f"[progression signals] {stage_norm['signals']}")

    # JSON 덤프(선택)
    if json_out_dir:
        os.makedirs(json_out_dir, exist_ok=True)
        outpath = os.path.join(json_out_dir, "emotion_relationship_analyzer.json")
        with open(outpath, "w", encoding="utf-8") as jf:
            json.dump(convert_keys_to_str(result), jf, ensure_ascii=False, indent=2)
        _emit(f"[saved] 결과 JSON: {outpath}")

    if fast_trigger:
        for obj, level in prev_log_levels:
            try:
                obj.setLevel(level)
            except Exception:
                pass
        for msg in log_messages:
            _mainlog.info(msg)

    return result



def cli_main():
    parser = argparse.ArgumentParser(description="관계/전개 분석기(문장→단계→감정쌍→소셜 그래프) 실행")
    parser.add_argument("--emotions", type=str, default=None, help="EMOTIONS.json 경로(없으면 스켈레톤 폴백)")
    parser.add_argument("--text", type=str, default=None, help="직접 텍스트 입력")
    parser.add_argument("--text-file", type=str, default=None, help="텍스트 파일 경로")
    parser.add_argument("--no-graph", action="store_true", help="소셜 그래프 비활성화")
    # 통합 로그 관리자 사용 (날짜별 폴더)
    try:
        from log_manager import get_log_manager
        log_manager = get_log_manager()
        default_out_dir = str(log_manager.get_log_dir("emotion_analysis", use_date_folder=True))
    except ImportError:
        # 폴백: 기존 방식 (날짜별 폴더 추가)
        from datetime import datetime
        base_log_dir = os.path.join(BASE_DIR, "emotion_analysis", "logs")
        today = datetime.now().strftime("%Y%m%d")
        default_out_dir = os.path.join(base_log_dir, today)
    
    parser.add_argument("--out", type=str, default=default_out_dir, help="결과 JSON 저장 디렉토리")
    parser.add_argument("--demo", action="store_true", help="데모 문장 3건 실행")
    parser.add_argument("--logfile", type=str, default="emotion_relationship_analyzer.log", help="로그 파일명")
    args = parser.parse_args()
    log_path = _ensure_file_logging(args.out, filename=args.logfile)
    _mainlog.info(f"[log] 파일 로깅 활성화: {log_path}")

    try:
        emotions = load_emotions_data(args.emotions, allow_fallback=True)
    except Exception as e:
        _mainlog.error(f"[fatal] 라벨링 로드 실패: {e}")
        return

    tests: List[str] = []
    if args.demo or (not args.text and not args.text_file):
        tests = [
            "오늘 프로젝트가 성공해서 기쁘고 뿌듯하다.",
            "회의에서 화가 났지만, 금방 진정하고 해결책을 찾았다.",
            "불꽃놀이를 보며 감탄했다.",
            "이 회의는 무의미하고 비효율적이라 짜증났다.",
            "서비스 latency가 줄어 들어서 안도했다.",
            "대기 시간이 30분 넘어서 지쳤다.",
            "그냥 그런 하루였다.",
            "성과는 좋았지만 팀원이 떠나서 서운했다.",
            "고객과의 대화가 즐거웠고 팀원들이 고마웠다.",
            "버그를 발견하고 불안했지만 즉시 패치되어 안심했다.",
            "드디어 끝! ☺️",
            "짜증."
        ]
        _mainlog.info("[demo] 12개 회귀 문장으로 실행합니다. --text / --text-file 로 교체 가능.")
    else:
        if args.text_file:
            if not os.path.isfile(args.text_file):
                _mainlog.error(f"[input] 텍스트 파일을 찾을 수 없습니다: {args.text_file}")
                return
            with open(args.text_file, "r", encoding="utf-8") as f:
                tests = [f.read()]
        elif args.text:
            tests = [args.text]

    for i, t in enumerate(tests, 1):
        _mainlog.info(f"────────── Test #{i} ──────────")
        run_main_once(t, emotions, include_graph=not args.no_graph, json_out_dir=args.out)

if __name__ == "__main__":
    cli_main()
