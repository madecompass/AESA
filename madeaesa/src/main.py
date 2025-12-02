# -*- coding: utf-8 -*-

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
Unified meta-model pipeline
- Input  : embeddings (run.jsonl), labels (labels.*.jsonl), 4 sub models (sub_희/노/애/락.pt)
- Feature: concat(embedding, sub-model scores & stats)
- Model  : MLP meta classifier => predict main/sub
- Modes  : train / eval / predict
"""

# Standard library imports
import argparse
import gzip
import inspect
import json
import math
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

# Third-party library imports
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

try:
    from src.models.reasoner_heads import ReasonerHead
except Exception:
    from models.reasoner_heads import ReasonerHead

# ============================
# 0. Utilities (robust I/O, seed, device, discovery)
# ============================
# --------- small logger ---------
def log(msg: str) -> None:
    """Unified timestamped log."""
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {msg}")

def now_tag(fmt: str = "%Y%m%d_%H%M%S") -> str:
    return time.strftime(fmt)

# --------- safe torch.load (PyTorch 2.6+ weights_only default) ---------
def _trusted_torch_load(path: str | Path, map_location: str = "cpu"):
    """
    안전 로더: 로컬에서 저장한 신뢰 가능한 체크포인트를 읽기 위해
    - 가능한 경우 weights_only=False로 로드
    - UnpicklingError 시 numpy reconstruct allowlist 추가 후 재시도
    """
    import inspect as _inspect
    import pickle as _pickle
    p = str(path)
    kwargs = {"map_location": map_location}
    try:
        if "weights_only" in _inspect.signature(torch.load).parameters:
            kwargs["weights_only"] = False
    except Exception:
        pass
    try:
        return torch.load(p, **kwargs)
    except _pickle.UnpicklingError:
        try:
            # allowlist numpy reconstruct (신뢰 파일에 한함)
            from torch.serialization import add_safe_globals as _add_safe
            import numpy as _np
            try:
                # numpy 2.x 경로
                _add_safe([_np._core.multiarray._reconstruct])  # type: ignore[attr-defined]
            except Exception:
                # 일부 환경에서 다른 네임스페이스
                _add_safe([_np.core.multiarray._reconstruct])   # type: ignore[attr-defined]
        except Exception:
            pass
        kwargs["weights_only"] = False
        return torch.load(p, **kwargs)

# --------- seed / determinism ---------
def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Set RNG seeds and (optionally) deterministic switches.
    - deterministic=True 에서 CuBLAS/CuDNN 권장 설정 적용(가능한 범위).
    """
    import numpy as _np
    random.seed(seed)
    _np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        # CuBLAS 워크스페이스 권장 설정(없으면 설정)
        if torch.cuda.is_available() and "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        # 결정론 알고리즘 시도 (버전/환경에 따라 일부 연산은 비결정론)
        try:
            torch.use_deterministic_algorithms(True)
        except Exception as e:
            log(f"[warn] deterministic algorithms not fully enabled: {e} -> soft fallback")
            try:
                torch.use_deterministic_algorithms(False)
            except Exception:
                pass
        # CuDNN 플래그 정리
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

# --------- device resolver ---------
def resolve_device(pref: str = "auto") -> str:
    """
    Return device string: 'cuda'|'mps'|'cpu'.
    - 환경변수 FORCE_CPU='1' 이면 강제 cpu.
    - CUDA 가능 시 matmul precision을 high 로 시도(2.x).
    """
    if os.getenv("FORCE_CPU", "0") == "1":
        return "cpu"

    d = (pref or "auto").lower()
    if d != "auto":
        return d

    if torch.cuda.is_available():
        try:
            # PyTorch 2.x: TF32/FP32 매트멀 정밀도 제어
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        return "cuda"

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"

    return "cpu"

# --------- robust JSON utilities ---------
_RE_TCOMMA = re.compile(r",\s*([}\]])")     # trailing comma
_RE_LINECOM = re.compile(r"//.*?$", re.MULTILINE)
_RE_BLKCOM  = re.compile(r"/\*.*?\*/", re.DOTALL)

def _strip_json_noise(s: str) -> str:
    """Remove comments / trailing commas; keep BOM-safe."""
    t = s.lstrip("\ufeff").strip()
    t = _RE_LINECOM.sub("", t)
    t = _RE_BLKCOM.sub("", t)
    t = _RE_TCOMMA.sub(r"\1", t)
    return t

def _loads_relaxed(s: str) -> Optional[dict]:
    """Relaxed JSON loads: comments/trailing commas tolerance + line fallback."""
    if not isinstance(s, str) or not s.strip():
        return None
    t = _strip_json_noise(s)
    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    # try largest {...} chunk
    beg, end = t.find("{"), t.rfind("}")
    if beg != -1 and end != -1 and end > beg:
        chunk = _strip_json_noise(t[beg:end+1])
        try:
            obj = json.loads(chunk)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    # line by line salvage
    for line in t.splitlines():
        line = _strip_json_noise(line)
        if not line or "{" not in line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    return None

def read_jsonl(path: str) -> Iterator[dict]:
    """
    Robust JSONL reader.
    - .jsonl 또는 .jsonl.gz 지원
    - 주석/트레일링 콤마/잘린 라인 복구 시도
    - dict만 yield
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    opener = gzip.open if p.suffix == ".gz" else open
    with opener(p, "rt", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            s = ln.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
                if isinstance(obj, dict):
                    yield obj
                    continue
            except Exception:
                pass
            obj = _loads_relaxed(s)
            if isinstance(obj, dict):
                yield obj

def write_jsonl(path: str, rows: List[dict]) -> None:
    """Safe JSONL writer (atomic)."""
    out = Path(path); out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fo:
        for o in rows:
            fo.write(json.dumps(o, ensure_ascii=False) + "\n")
    tmp.replace(out)

# --------- id helpers ---------
def pick_id(o: dict) -> Optional[str]:
    """
    Try multiple id fields commonly used in our pipes.
    Order: id > meta.trace_id > trace_id > sample_id > record.id > uid/guid.
    """
    for cand in (
        o.get("id"),
        (o.get("meta") or {}).get("trace_id"),
        o.get("trace_id"),
        o.get("sample_id"),
        (o.get("record") or {}).get("id"),
        o.get("uid"),
        o.get("guid"),
        (o.get("metadata") or {}).get("trace_id"),
    ):
        if isinstance(cand, str) and cand.strip():
            return cand.strip()
    return None

# --------- small fs helpers ---------
def ensure_dir(p: str | Path) -> Path:
    pth = Path(p); pth.mkdir(parents=True, exist_ok=True); return pth



def _resolve_pointer(p: str) -> str:
    q = Path(p)
    if q.suffix == ".path" and q.exists():
        try:
            tgt = q.read_text(encoding="utf-8").strip()
            if tgt:
                print(f"[auto] labels-jsonl pointer -> {tgt}")
                return tgt
        except Exception:
            pass
    return p
def find_latest_run_dir(models_root: str = "src/models") -> Optional[Path]:
    """
    Auto-discover latest model run dir under src/models (e.g., quick_YYYY..., smoke_..., full_...).
    가장 최근 mtime 기준으로 하나 반환.
    """
    root = Path(models_root)
    if not root.exists(): return None
    dirs = [d for d in root.iterdir() if d.is_dir()]
    if not dirs: return None
    dirs.sort(key=lambda d: (d.stat().st_mtime, d.name), reverse=True)
    return dirs[0]

def find_latest_sub_run_dir(models_root: str = "src/models") -> Optional[Path]:
    """
    서브 체크포인트 4개(sub_희/노/애/락)가 모두 들어있는 디렉터리 중 최신을 고른다.
    각 메인에 대해 sub_{m}_best.pt 또는 sub_{m}.pt 또는 sub_{m}_last.pt 중 하나라도 있으면 해당 메인은 충족으로 본다.
    """
    root = Path(models_root)
    if not root.exists():
        return None
    candidates: list[Path] = []
    for d in root.iterdir():
        if not d.is_dir():
            continue
        has_all = True
        for m in ("희", "노", "애", "락"):
            if not ((d / f"sub_{m}_best.pt").exists() or (d / f"sub_{m}.pt").exists() or (d / f"sub_{m}_last.pt").exists()):
                has_all = False
                break
        if has_all:
            candidates.append(d)
    if not candidates:
        return None
    candidates.sort(key=lambda dd: (dd.stat().st_mtime, dd.name), reverse=True)
    return candidates[0]

def _load_cause_vocab_size(path: str, default: int = 128) -> int:
    try:
        p = Path(path)
        if not p.exists(): return default
        obj = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(obj, dict):
            return max(1, len(obj))
        return default
    except Exception:
        return default


def find_sub_models(run_dir: str | Path) -> Dict[str, Path]:
    """
    run_dir 아래에서 sub_희/노/애/락 체크포인트 자동 탐색.
    우선순위: sub_{m}_best.pt > sub_{m}.pt > sub_{m}_last.pt
    """
    run = Path(run_dir)
    found: Dict[str, Path] = {}
    for m in ("희","노","애","락"):
        for name in (f"sub_{m}_best.pt", f"sub_{m}.pt", f"sub_{m}_last.pt"):
            p = run / name
            if p.exists():
                found[m] = p
                break
    return found


# --------- 분리 저장 sub/unified 자동 페어링 --------
PROJECT_ROOT = Path(__file__).parent.parent  # src 상위 디렉터리

def _models_cfg() -> Dict[str, Any]:
    """모델 탐색 설정 반환 (환경변수로 튜닝 가능)"""
    return {
        "search_bases": os.getenv("MODEL_SEARCH_BASES", "").split(",") if os.getenv("MODEL_SEARCH_BASES") else [],
        "ignore_globs": os.getenv("MODEL_IGNORE_GLOBS", "temp,tmp,backup,old").split(","),
        "unified_filenames": ["unified_model.pt", "unified.pt", "meta_model.pt", "main_model.pt"]
    }

def _iter_ignore(base: Path, ignore_globs: List[str]) -> Iterator[Path]:
    """무시할 디렉터리들의 절대경로 생성"""
    for ig in ignore_globs:
        ig = ig.strip()
        if ig:
            yield (base / ig).resolve()

def _find_best_separate_pair(models_root: Optional[str]) -> Tuple[Optional[Tuple[Path, Path]], List[str]]:
    """
    분리 저장된 최신 sub/unified를 각각 찾아서 페어링
    Returns: ((sub_dir, unified_file), logs) or (None, logs)
    """
    cfg = _models_cfg()
    bases: List[Path] = []
    if models_root: bases.append(Path(models_root))
    if cfg["search_bases"]:
        bases += [Path(b) for b in cfg["search_bases"]]
    bases += [PROJECT_ROOT / "src" / "models", PROJECT_ROOT / "models", PROJECT_ROOT / "runs"]

    # 중복 제거
    seen, uniq = set(), []
    for b in bases:
        r = b.resolve() if b.exists() else b
        if str(r) not in seen:
            uniq.append(r); seen.add(str(r))

    logs: List[str] = ["[fallback] separate sub/unified scan"]
    best_sub_dir, best_sub_mtime = None, 0.0
    best_uni,     best_uni_mtime = None, 0.0

    # ① 최신 sub 디렉터리
    for base in uniq:
        if not base.exists(): continue
        for p in base.rglob("sub_*.pt"):
            try:
                rd = p.parent.resolve()
                mt = float(p.stat().st_mtime)
                # 무시 목록 제외
                if any(str(rd).startswith(str(ig)) for ig in _iter_ignore(base, cfg["ignore_globs"])):
                    continue
                # sub 파일이 많은(또는 최신 mtime) 디렉터리 우선
                if mt > best_sub_mtime:
                    best_sub_mtime = mt; best_sub_dir = rd
            except Exception:
                continue

    # ② 최신 unified 파일(어디에 있어도 허용)
    UNI_NAMES = tuple(cfg["unified_filenames"]) + ("model.pt","model.safetensors")
    for base in uniq:
        if not base.exists(): continue
        for patt in ["**/unified*/**/*"] + [f"**/{name}" for name in UNI_NAMES]:
            for u in base.glob(patt):
                try:
                    if u.is_file() and any(u.name.endswith(n) for n in UNI_NAMES):
                        mt = float(u.stat().st_mtime)
                        if mt > best_uni_mtime:
                            best_uni_mtime = mt; best_uni = u.resolve()
                except Exception:
                    continue

    if best_sub_dir and best_uni:
        logs += [f"  - pick sub_dir={best_sub_dir}", f"  - pick unified={best_uni}"]
        return ((best_sub_dir, best_uni), logs)
    logs += ["  - no separate pair found"]
    return (None, logs)


# ----------------------------
# 1. Data joining
# ----------------------------
def load_embeddings_jsonl(path: str, id_field: str = "id", emb_field: str = "embedding") -> Tuple[np.ndarray, List[str]]:
    """
    Robust embedding loader from JSONL.
    - Accepts list/np.ndarray 1D embeddings
    - Filters NaN/Inf/zero-norm/empty
    - Resolves dim mixing by majority dim (mode)
    - Deduplicates by id (last-wins within the chosen mode)
    - Optional L2 normalization via ENV: EMB_L2=1
    Returns
      X  : [N, D] float32
      ids: [N]    list[str] (sorted)
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))

    rows: List[Tuple[str, np.ndarray]] = []
    n_total = n_bad = 0
    dim_count: Dict[int, int] = {}

    for o in read_jsonl(str(p)):
        _id = pick_id(o) or o.get(id_field)
        emb = o.get(emb_field)
        n_total += 1
        if not (isinstance(_id, str) and _id.strip() and isinstance(emb, (list, np.ndarray))):
            n_bad += 1
            continue
        v = np.asarray(emb, dtype=np.float32)
        if v.ndim != 1 or v.size == 0 or not np.isfinite(v).all() or float(np.linalg.norm(v)) == 0.0:
            n_bad += 1
            continue
        rows.append((_id.strip(), v))
        dim_count[v.size] = dim_count.get(v.size, 0) + 1

    if not rows:
        raise ValueError(f"no embeddings in {path}")

    # majority dim only
    dim_mode = max(dim_count.items(), key=lambda kv: kv[1])[0]
    uniq: Dict[str, np.ndarray] = {}
    for _id, v in rows:
        if v.size == dim_mode:
            # last-wins for duplicated id
            uniq[_id] = v

    if not uniq:
        raise ValueError(f"all embeddings filtered by dim; expected dim={dim_mode}")

    # sort by id for reproducibility
    ids = sorted(uniq.keys())
    X = np.vstack([uniq[i] for i in ids]).astype(np.float32, copy=False)

    # optional L2 normalize (env switch)
    if os.getenv("EMB_L2", "0") == "1":
        nrm = np.linalg.norm(X, axis=1, keepdims=True); nrm = np.maximum(nrm, 1e-9)
        X = (X / nrm).astype(np.float32, copy=False)

    log(f"[emb] loaded {len(ids)} rows (mode_dim={dim_mode}) | skipped={n_bad}/{n_total} from {path}")
    return X, ids  # [N,D], [N]


def load_labels_jsonl(path: str, *, target_main: Optional[str] = None) -> Dict[str, Tuple[str, Optional[str]]]:
    """
    Load labels (id -> (main, sub)) from JSONL.
    - Accepts fields: id | meta.trace_id | record.id ...
    - Chooses the highest-confidence occurrence when duplicates exist (reasons.confidence -> score_main -> last-wins)
    - If target_main is given, keeps only that main
    - sub is kept as None if absent (downstream can ignore_index=-1)
    """
    path = _resolve_pointer(path)
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))

    y: Dict[str, Tuple[str, Optional[str]]] = {}
    best_conf: Dict[str, float] = {}

    def _confidence(o: dict) -> float:
        try:
            # 1) top-level confidence
            if "confidence" in o and o["confidence"] is not None:
                return float(o["confidence"])
            # 2) reasons.confidence
            r = o.get("reasons") or {}
            if "confidence" in r and r["confidence"] is not None:
                return float(r["confidence"])
            # 3) fallback: score_main
            return float(o.get("score_main", 0.0) or 0.0)
        except Exception:
            return 0.0

    n = 0
    for o in read_jsonl(str(p)):
        n += 1
        _id = pick_id(o) or o.get("id")
        m = (
            o.get("main")
            or o.get("label_main")
            or (o.get("reasons") or {}).get("label_main")
            or (o.get("debug_meta") or {}).get("main")
            or o.get("pred_main")
        )
        s = (
            o.get("sub")
            or o.get("label_sub")
            or (o.get("reasons") or {}).get("label_sub")
            or o.get("label")
            or o.get("pred_sub")
        )
        if not (isinstance(_id, str) and _id.strip() and isinstance(m, str) and m.strip()):
            continue
        if target_main and m != target_main:
            continue

        conf = _confidence(o)
        prev = best_conf.get(_id, -1.0)
        min_conf_thresh = float(os.getenv("MIN_LABEL_CONF", "0.0"))
        if conf >= prev and conf >= min_conf_thresh:
            y[_id.strip()] = (m.strip(), s.strip() if isinstance(s, str) and s.strip() else None)
            best_conf[_id.strip()] = conf

    log(f"[lab] loaded {len(y)} labeled ids from {path} (filtered main={target_main or 'ALL'})")
    return y

def load_labels_jsonl_full(path: str) -> Dict[str, dict]:
    """
    labels JSONL에서 id -> full object 맵을 만든다.
    (Reasoner 보조 라벨: target_unc, cause_ids, target_pivot_idx 등을 꺼내기 위함)
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    tbl: Dict[str, dict] = {}
    for o in read_jsonl(str(p)):
        _id = pick_id(o) or o.get("id")
        if isinstance(_id, str) and _id.strip():
            tbl[_id.strip()] = o
    log(f"[lab-full] loaded {len(tbl)} rows from {path}")
    return tbl


# --- NEW: compat submodel for checkpoints with 'backbone.*' & '_logit_scale' (from sub_classifier.py) ---
class SubModelCompat(nn.Module):
    def __init__(self, in_dim: int, n_cls: int, hidden: int = 512, drop: float = 0.2, act: str = "gelu"):
        super().__init__()
        Act = nn.GELU if act.lower() == "gelu" else (nn.ReLU if act.lower() == "relu" else nn.SiLU)
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
        # absorb '_logit_scale' from checkpoint
        self.register_buffer("_logit_scale", torch.tensor(1.0, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.float32:
            x = x.float()
        logits = self.backbone(x)
        if self._logit_scale is not None:
            logits = logits * self._logit_scale
        return logits

# ----------------------------
# 2. Sub-model wrapper & feature builder
# ----------------------------
from dataclasses import dataclass

@dataclass
class SubModelInfo:
    main: str
    path: Path
    sub_list: List[str]
    in_dim: int

class SubModel(torch.nn.Module):
    """
    단순 MLP 헤드(서브모델 저장 포맷과 동일 구조가 아닐 수 있어, state_dict로 치환).
    주의: 실제 서브모델은 저장된 state_dict를 그대로 로드합니다.
    """
    def __init__(self, in_dim: int, n_cls: int, hidden: int = 512, drop: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(hidden, hidden//2), nn.ReLU(),
            nn.Linear(hidden//2, n_cls)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.float32: x = x.float()
        return self.net(x)

def _softmax_np(x: np.ndarray, T: float = 1.0, eps: float = 1e-9) -> np.ndarray:
    z = x.astype(np.float32) / max(T, eps)
    z -= z.max()
    e = np.exp(z)
    p = e / max(e.sum(), eps)
    return p

class SubEnsemble:
    """
    4개 서브모델(sub_희/노/애/락.pt) 래퍼 (호환 로더 포함).
    기대 저장 포맷 (둘 중 하나):
      1) torch.save({"state_dict": SD, "meta": {"sub_list":[...], "in_dim":384, "temperature":..., "logit_scale":...}}, path)
      2) torch.save(SD, path)  # state_dict만 저장된 경우도 처리

    - run_dir가 None이거나 파일이 없으면 src/models에서 최신 폴더 자동 탐색.
    - 각 모델의 temperature(또는 logit_scale, 또는 state_dict의 '_logit_scale' 버퍼)로 소프트맥스 보정.
    - SubModel(기존 net.* 키)과 SubModelCompat(backbone.* 키) 모두 자동 감지 로드.
    """
    def __init__(self, run_dir: Optional[str] = None, device: str = "cpu"):
        # device="auto" 처리
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.mains = ["희", "노", "애", "락"]
        # model, sub_list, temperature
        self.models: Dict[str, Tuple[nn.Module, List[str], float]] = {}
        self.in_dim: Optional[int] = None

        # --- 자동 탐색 ---
        rd: Optional[Path] = None
        if run_dir:
            rd = Path(run_dir)
            if not rd.exists() or not rd.is_dir():
                raise FileNotFoundError(f"run_dir not found: {run_dir}")
        else:
            try:
                rd = find_latest_run_dir("src/models")
            except Exception:
                rd = None
        if rd is None:
            raise FileNotFoundError("No run_dir given and no latest run under src/models.")

        # 실제 로딩된 실행 디렉터리 보관(캐시 키 안정화용)
        try:
            self.run_dir = rd.resolve()
        except Exception:
            self.run_dir = rd

        # --- sub_*.pt 탐색 ---
        try:
            found = find_sub_models(rd)  # 유틸이 있으면 사용
        except Exception:
            # 유틸이 없다면 로컬 탐색
            found = {}
            for m in self.mains:
                cands = list(rd.glob(f"sub_{m}.pt"))
                if cands:
                    found[m] = cands[0]

        missing = [m for m in self.mains if m not in found]
        # [개선사항 2] 결측 메인을 더미로 보완
        if missing:
            log(f"[warn] sub models missing for mains: {missing} in {rd} -> using DUMMY uniform heads")
            # in_dim 추정(이미 로드된 모델이 있으면 그 in_dim, 없으면 384)
            in_dim_use = self.in_dim or 384
            for m in missing:
                # 서브 리스트(없으면 30개 기본) — EMOTIONS.json의 서브를 끼워도 좋습니다.
                subs = [f"{m}_sub_{i+1}" for i in range(30)]
                # 더미 확률(균일분포)을 만들기 위해 "항상 0 로짓"을 내는 얕은 모델 준비
                dummy = nn.Linear(in_dim_use, len(subs), bias=False)
                with torch.no_grad():
                    dummy.weight.zero_()
                dummy.to(self.device).eval()
                # mask(전부 True) / T=1.0
                if not hasattr(self, "present_mask_by_main"): self.present_mask_by_main = {}
                self.present_mask_by_main[m] = np.ones((len(subs),), dtype=bool)
                self.models[m] = (dummy, subs, 1.0)
            # in_dim 통일
            if self.in_dim is None: self.in_dim = in_dim_use

        # --- 로드 ---
        for m in self.mains:
            # 더미로 이미 처리된 메인은 건너뛰기
            if m in missing:
                continue
            # 1) 체크포인트 로드 (dict or state_dict) - 안전 로더 사용 (PyTorch 2.6+ 호환)
            raw = _trusted_torch_load(str(found[m]), map_location="cpu")
            if isinstance(raw, dict) and "state_dict" in raw:
                ckpt = raw
                sd = ckpt["state_dict"]
                meta = ckpt.get("meta") or {}
            elif isinstance(raw, dict):
                # state_dict만 저장된 경우
                sd = raw
                meta = {}
            else:
                # 알 수 없는 포맷
                raise RuntimeError(f"Unsupported checkpoint format for {found[m].name}: type={type(raw)}")

            # 2) sub_list 확보 (없으면 out_features로 추정)
            sub_list: List[str] = meta.get("sub_list") or []
            if not sub_list:
                n_cls = self._infer_num_classes_from_state_dict(sd)
                if n_cls is None:
                    n_cls = 30  # 보수적 fallback
                    log(f"[warn] meta.sub_list missing for {m}; fallback n_cls={n_cls}")
                else:
                    log(f"[info] meta.sub_list missing for {m}; inferred n_cls={n_cls}")
                sub_list = [f"{m}_sub_{i+1}" for i in range(n_cls)]

            # 3) in_dim 확보 (없으면 첫 Linear in_features로 추정)
            in_dim = int(meta.get("in_dim", 0)) if meta.get("in_dim") is not None else 0
            if not in_dim:
                in_dim = self._infer_in_dim_from_state_dict(sd) or 384  # 최종 fallback
                if "in_dim" not in meta:
                    log(f"[info] meta.in_dim missing for {m}; inferred in_dim={in_dim}")

            n_cls = len(sub_list)

            # [PATCH 1.1] present_idx 읽어서 마스크 생성
            present_idx = None
            if isinstance(meta, dict):
                pi = meta.get("present_idx")
                if isinstance(pi, (list, tuple)) and len(pi) > 0:
                    present_idx = [int(i) for i in pi if isinstance(i, (int, np.integer))]

            # 마스크: 기본은 전체 True, present_idx가 있으면 해당 인덱스만 True
            mask = np.ones((n_cls,), dtype=bool)
            if present_idx is not None:
                mask[:] = False
                mask[np.clip(present_idx, 0, n_cls-1)] = True

            # models 사전은 그대로 두고, 별도 dict로 마스크를 보관
            if not hasattr(self, "present_mask_by_main"):
                self.present_mask_by_main = {}
            self.present_mask_by_main[m] = mask

            # 4) 온도 스케일 T 결정 (temperature or logit_scale or _logit_scale)
            T = self._resolve_temperature(meta, sd)
            # [IMP] per-class T가 있으면 우선 적용 (검증 NLL로 적합된 클래스별 온도)
            try:
                per_class_T = meta.get("per_class_T", None)
                if isinstance(per_class_T, (list, tuple)) and len(per_class_T) == n_cls:
                    T = np.asarray([max(1e-6, float(t)) for t in per_class_T], dtype=np.float32)
            except Exception:
                pass

            # 5) 어떤 래퍼로 로드할지 결정 (키 패턴으로 감지)
            uses_backbone = any(k.startswith("backbone.") for k in sd.keys()) or ("_logit_scale" in sd)
            if uses_backbone:
                # sub_classifier.py 포맷과 호환: SubModelCompat로 로드
                mdl = SubModelCompat(in_dim, n_cls, hidden=512, drop=0.2, act="gelu")
                mdl.load_state_dict(sd, strict=True)
                tag = "compat"
            else:
                # 기존 포맷(net.*) 대응: SubModel로 로드
                mdl = SubModel(in_dim, n_cls)
                # 혹시 여분/결측 키가 있을 수 있어 strict=False가 더 안전
                mdl.load_state_dict(sd, strict=False)
                tag = "default"

            mdl.to(self.device).eval()
            self.models[m] = (mdl, sub_list, max(T, 1e-6))

            # 6) in_dim 일관성 체크
            if self.in_dim is None:
                self.in_dim = in_dim
            elif self.in_dim != in_dim:
                raise ValueError(f"in_dim mismatch across sub models: prev={self.in_dim}, {m}={in_dim}")

            log(f"[sub] loaded {found[m].name} ({tag}) | in_dim={in_dim} | n_cls={n_cls} | T={T:.4f}")

        # 요약 로그
        cls_summ = ", ".join([f"{m}:{len(self.models[m][1])}" for m in self.mains])
        log(f"[sub] loaded models from {rd} | in_dim={self.in_dim} | classes={{" + cls_summ + "}}")

    def set_temperature_per_main(self, tmap: dict[str, float]) -> None:
        """온도 보정 주입: {'희':1.2,'노':1.1,'애':1.0,'락':0.95}"""
        try:
            for m, t in (tmap or {}).items():
                if m in self.models:
                    mdl, subs, _T = self.models[m]
                    self.models[m] = (mdl, subs, float(t))
        except Exception:
            pass

    # -------------------------
    # 내부 유틸 (state_dict 해석)
    # -------------------------
    @staticmethod
    def _infer_num_classes_from_state_dict(sd: Dict[str, torch.Tensor]) -> Optional[int]:
        """
        마지막 Linear의 out_features로 n_cls 추정.
        backbone.* 또는 net.* 경로 모두 지원.
        """
        # 우선순위: backbone의 가장 마지막 Linear
        last_w = None
        last_idx = -1
        for k, w in sd.items():
            if k.endswith(".weight") and w.ndim == 2:
                # 인덱스 파싱 (예: backbone.7.weight -> 7)
                idx = -1
                try:
                    # 'backbone.X.weight' or 'net.X.weight'
                    parts = k.split(".")
                    if len(parts) >= 3 and parts[1].isdigit():
                        idx = int(parts[1])
                except Exception:
                    pass
                # 더 뒤에 오는 선형층을 후보로
                if idx > last_idx:
                    last_idx = idx
                    last_w = w
        if last_w is not None:
            return int(last_w.shape[0])
        # 못 찾으면 None
        return None

    @staticmethod
    def _infer_in_dim_from_state_dict(sd: Dict[str, torch.Tensor]) -> Optional[int]:
        """
        첫 Linear의 in_features로 in_dim 추정.
        backbone.0.weight 또는 net.0.weight 우선 탐색.
        """
        for first_key in ("backbone.0.weight", "net.0.weight"):
            if first_key in sd and sd[first_key].ndim == 2:
                return int(sd[first_key].shape[1])
        # 백업: 가장 작은 index의 *.weight에서 in_features 사용
        best = None
        best_idx = 10**9
        for k, w in sd.items():
            if k.endswith(".weight") and w.ndim == 2:
                # 작은 인덱스가 앞단일 확률이 높음
                idx = 10**9
                try:
                    parts = k.split(".")
                    if len(parts) >= 3 and parts[1].isdigit():
                        idx = int(parts[1])
                except Exception:
                    pass
                if idx < best_idx:
                    best_idx = idx
                    best = w
        if best is not None:
            return int(best.shape[1])
        return None

    @staticmethod
    def _resolve_temperature(meta: Dict[str, Any], sd: Dict[str, torch.Tensor]) -> float:
        """
        temperature 우선, 없으면 logit_scale, 없으면 state_dict의 '_logit_scale' 버퍼 반영.
        환경변수 SUB_T가 유효하면 최종 오버라이드.
        """
        T = 1.0
        # meta 기반
        if "temperature" in meta and meta["temperature"]:
            try:
                T = float(meta["temperature"])
            except Exception:
                T = 1.0
        elif "logit_scale" in meta and meta["logit_scale"]:
            try:
                ls = float(meta["logit_scale"])
                T = 1.0 / max(ls, 1e-6)
            except Exception:
                T = 1.0
        else:
            # state_dict 버퍼 기반
            try:
                if "_logit_scale" in sd and torch.is_tensor(sd["_logit_scale"]):
                    ls_val = float(sd["_logit_scale"].item())
                    T = 1.0 / max(ls_val, 1e-6)
            except Exception:
                pass

        # 환경변수 오버라이드 (테스트/튜닝용)
        try:
            T_env = float(os.getenv("SUB_T", "nan"))
            if not math.isnan(T_env):
                T = T_env
        except Exception:
            pass

        return T

    @torch.no_grad()
    def predict_proba(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        """
        x: [D] numpy -> returns {main: [n_sub] probabilities}
        """
        if not isinstance(x, np.ndarray): x = np.asarray(x, dtype=np.float32)
        if x.ndim != 1 or x.size != int(self.in_dim):
            raise ValueError(f"embedding dim mismatch: expected {self.in_dim}, got {x.size}")
        x_t = torch.from_numpy(x.astype(np.float32)).to(self.device).unsqueeze(0)
        out: Dict[str, np.ndarray] = {}
        
        # CuBLAS deterministic 오류 방지: 일시적으로 deterministic 모드 비활성화
        deterministic_was_enabled = torch.are_deterministic_algorithms_enabled()
        if deterministic_was_enabled:
            torch.use_deterministic_algorithms(False)
        
        try:
            for m in self.mains:
                mdl, subs, T = self.models[m]
                logits = mdl(x_t).cpu().numpy().squeeze(0)
                # scalar T 또는 per-class T(vector) 모두 지원
                if isinstance(T, np.ndarray) and T.shape[0] == logits.shape[0]:
                    z = logits.astype(np.float32) / np.maximum(T, 1e-6)
                    z = z - float(z.max())
                    e = np.exp(z)
                    s = float(e.sum())
                    p = (e / max(s, 1e-9)).astype(np.float32, copy=False)
                else:
                    p = _softmax_np(logits, T=float(T) if not isinstance(T, (int, float)) else T)
                # [PATCH 1.2] 마스크 적용
                mask = getattr(self, "present_mask_by_main", {}).get(m, None)
                if mask is not None and mask.size == p.size and not mask.all():
                    p = p * mask.astype(np.float32)
                    s = float(p.sum())
                    p = (p / max(s, 1e-9)).astype(np.float32, copy=False)
                out[m] = p
        finally:
            # 원래 deterministic 모드 상태로 복원
            if deterministic_was_enabled:
                try:
                    torch.use_deterministic_algorithms(True)
                except Exception:
                    pass
        
        return out

    @torch.no_grad()
    def predict_proba_batch(self, X: np.ndarray, batch: int = 1024) -> Dict[str, np.ndarray]:
        """
        X: [N, D] -> returns {main: [N, n_sub]}
        빠른 배치 추론(완전 벡터화).
        """
        if not isinstance(X, np.ndarray): X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2 or X.shape[1] != int(self.in_dim):
            raise ValueError(f"embedding dim mismatch: expected {self.in_dim}, got {X.shape[1]}")
        N = X.shape[0]
        out: Dict[str, List[np.ndarray]] = {m: [] for m in self.mains}

        # CuBLAS deterministic 오류 방지: 일시적으로 deterministic 모드 비활성화
        deterministic_was_enabled = torch.are_deterministic_algorithms_enabled()
        if deterministic_was_enabled:
            torch.use_deterministic_algorithms(False)
        
        try:
            for s in range(0, N, batch):
                xb = torch.from_numpy(X[s:s + batch].astype(np.float32)).to(self.device)
                for m in self.mains:
                    mdl, subs, T = self.models[m]
                    logits = mdl(xb).cpu().numpy()  # [b, C]
                    # scalar T 또는 per-class T(vector) 모두 지원
                    if isinstance(T, np.ndarray) and T.shape[0] == logits.shape[1]:
                        z = logits.astype(np.float32) / np.maximum(T[None, :], 1e-6)
                    else:
                        z = logits / max(float(T), 1e-6)
                    z -= z.max(axis=1, keepdims=True)
                    e = np.exp(z, dtype=np.float32)
                    p = e / np.clip(e.sum(axis=1, keepdims=True), 1e-9, None)  # [b,C]
                    # [PATCH 1.3] 배치 마스크 적용
                    mask = getattr(self, "present_mask_by_main", {}).get(m, None)
                    if mask is not None and mask.size == p.shape[1] and not mask.all():
                        pm = mask.astype(np.float32)[None, :]                    # [1,C]
                        p = p * pm
                        s = p.sum(axis=1, keepdims=True)
                        p = p / np.clip(s, 1e-9, None)
                    out[m].append(p.astype(np.float32, copy=False))

            return {m: np.vstack(chunks) for m, chunks in out.items()}
        finally:
            # 원래 deterministic 모드 상태로 복원
            if deterministic_was_enabled:
                try:
                    torch.use_deterministic_algorithms(True)
                except Exception:
                    pass


def _entropy(p: np.ndarray, eps: float = 1e-9) -> float:
    q = np.clip(p, eps, 1.0)
    return float(-(q * np.log(q)).sum())

def _gini(p: np.ndarray, eps: float = 1e-9) -> float:
    q = np.clip(p, eps, 1.0)
    return float(1.0 - (q*q).sum())

def _margin(p: np.ndarray) -> float:
    if p.size < 2:
        return float(p.max() if p.size else 0.0)
    top2 = np.sort(p)[-2:]
    return float(top2[-1] - top2[-2])

def build_feature_for_id(emb: np.ndarray, sub_out: Dict[str, np.ndarray], *, stats_only: bool = False) -> np.ndarray:
    """
    단일 샘플의 메타 특징 벡터 생성.
    - per-main 통계 5종: [max, top3_mean, entropy, margin, gini]
    - (옵션) 원시 확률 전부 concat
    - (옵션) 임베딩 concat
    Feature dim = 4*5 (+ sum(C_m) if not stats_only) (+ emb_dim if not stats_only)
    """
    feats: List[float] = []
    mains = ["희", "노", "애", "락"]
    for m in mains:
        p = np.asarray(sub_out[m], dtype=np.float32)
        top3 = np.sort(p)[-3:] if p.size >= 3 else np.sort(p)
        feats.extend([
            float(p.max()),
            float(top3.mean()),
            _entropy(p),
            _margin(p),
            _gini(p),
        ])

    if not stats_only:
        # 원시 확률
        for m in mains:
            feats.extend(list(np.asarray(sub_out[m], dtype=np.float32)))
        # 임베딩
        feats.extend(list(np.asarray(emb, dtype=np.float32)))

    return np.asarray(feats, dtype=np.float32)

def build_feature_for_batch(X: np.ndarray, sub_probs: Dict[str, np.ndarray], *, stats_only: bool = False) -> np.ndarray:
    """
    배치 버전 메타 특징 생성.
    - X: [N, D]
    - sub_probs: {main: [N, C_m]}
    """
    N = X.shape[0]
    mains = ["희", "노", "애", "락"]
    stats = []
    for m in mains:
        P = sub_probs[m].astype(np.float32)         # [N, C]
        # max
        maxv = P.max(axis=1)
        # top3_mean
        if P.shape[1] >= 3:
            # 상위 3개를 안정적으로 얻기 위해 partial sort 후 평균
            t3 = np.partition(P, kth=P.shape[1]-3, axis=1)[:, -3:]
            t3m = t3.mean(axis=1)
        else:
            t3m = P.mean(axis=1)
        # entropy
        Q = np.clip(P, 1e-9, 1.0)
        ent = -(Q * np.log(Q)).sum(axis=1)
        # margin = top1 - top2 (항상 비음수)
        if P.shape[1] >= 2:
            top2 = np.partition(P, kth=P.shape[1]-2, axis=1)[:, -2:]   # 두 개는 정렬 보장 X
            second = top2.min(axis=1)                                   # 두 값 중 작은 게 2등
            mar = maxv - second
        else:
            mar = maxv
        # gini
        gini = 1.0 - (Q * Q).sum(axis=1)

        stats.append(np.stack([maxv, t3m, ent, mar, gini], axis=1))  # [N,5]

    stats_cat = np.concatenate(stats, axis=1)  # [N, 4*5]
    if stats_only:
        return stats_cat.astype(np.float32, copy=False)

    # raw probs concat
    raw = [sub_probs[m].astype(np.float32, copy=False) for m in mains]
    raw_cat = np.concatenate(raw, axis=1)  # [N, sum C_m]
    feats = np.concatenate([stats_cat, raw_cat, X.astype(np.float32, copy=False)], axis=1)
    return feats



# === add: calibrate temperatures on validation ===
def fit_temperature_on_val(sub_probs_val: Dict[str, np.ndarray],
                           y_main_val: np.ndarray,
                           y_sub_name_val: List[Tuple[str, Optional[str]]],
                           sub_list_by_main: Dict[str, list[str]],
                           init_T: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """
    per-main 서브확률에 softmax(z/T)를 적용하는 T를 NLL 최소화로 추정.
    y_sub_name_val은 (main, sub) 튜플. sub가 None이면 해당 로우는 건너뜀.
    """
    import numpy as _np
    Ts = {m: (init_T[m] if (init_T and m in init_T) else 1.0) for m in sub_list_by_main.keys()}
    for m in Ts.keys():
        P = sub_probs_val[m].astype(_np.float64)  # [N, C_m]
        subs = sub_list_by_main[m]
        idx = [i for i, (mm, ss) in enumerate(y_sub_name_val) if (mm == m and isinstance(ss, str) and ss in subs)]
        if not idx:
            continue
        y = _np.array([subs.index(y_sub_name_val[i][1]) for i in idx], dtype=_np.int64)
        Z = _np.log(_np.clip(P[idx], 1e-12, 1.0))  # logits ~= log prob, 근사

        grid = _np.linspace(0.5, 5.0, 91)
        best_T, best_nll = Ts[m], 1e9
        for T in grid:
            z = Z / max(T, 1e-6)
            z = z - z.max(axis=1, keepdims=True)
            logp = z - _np.log(_np.exp(z).sum(axis=1, keepdims=True))
            nll = -float(logp[_np.arange(len(y)), y].mean())
            if nll < best_nll:
                best_nll, best_T = nll, float(T)
        Ts[m] = best_T
    return Ts


def _hash_ckpts(run_dir: str) -> str:
    import hashlib as _hl
    h = _hl.sha256()
    rd = Path(run_dir)
    for m in ("희", "노", "애", "락"):
        for name in (f"sub_{m}_best.pt", f"sub_{m}.pt", f"sub_{m}_last.pt"):
            p = rd / name
            if p.exists():
                try:
                    h.update(str(p.stat().st_mtime_ns).encode())
                except Exception:
                    h.update(str(p.stat().st_mtime).encode())
                break  # 이 메인에 대해 가장 우선순위 높은 파일 하나만 반영
    return h.hexdigest()[:16]



def cached_sub_probs(X: np.ndarray, sub_ens: "SubEnsemble", outdir: str) -> Dict[str, np.ndarray]:
    """
    서브 배치 확률 캐시(NPZ). ckpt 해시·입력차원으로 키 생성.
    """
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    try:
        any_m = next(iter(sub_ens.models))
        mdl_cls = sub_ens.models[any_m][0].__class__.__name__
    except Exception:
        mdl_cls = "SubModel"
    # 실제 로딩된 run_dir로 해시 생성(불일치/ENV 의존 제거)
    rd = getattr(sub_ens, "run_dir", None)
    key_hash = _hash_ckpts(str(rd)) if isinstance(rd, Path) and rd.exists() else mdl_cls
    key = f"sub_probs_{int(sub_ens.in_dim)}_{key_hash}.npz"
    path = out / key
    if path.exists():
        data = np.load(path, allow_pickle=True)
        return {m: data[m] for m in data.files}
    probs = sub_ens.predict_proba_batch(X, batch=2048)
    np.savez_compressed(path, **probs)
    return probs


# === feature scaler ===
class FeatureScaler:
    def __init__(self):
        self.mu = None
        self.sigma = None

    def fit(self, X: torch.Tensor):
        mu = X.mean(dim=0)
        sigma = X.std(dim=0).clamp_min(1e-6)
        self.mu = mu
        self.sigma = sigma

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        if self.mu is None or self.sigma is None:
            return X
        return (X - self.mu) / self.sigma


# === util: load sample confidence map ===
def load_label_confidence(path: str) -> dict[str, float]:
    conf: dict[str, float] = {}
    for o in read_jsonl(path):
        _id = pick_id(o) or o.get("id")
        if not isinstance(_id, str):
            continue
        # 1) top-level → 2) reasons → 3) score_main
        r = o.get("reasons") or {}
        raw = o.get("confidence", r.get("confidence", o.get("score_main", 0.0)))
        try:
            v = max(0.0, min(1.0, float(raw)))
        except Exception:
            v = 0.0
        if v >= conf.get(_id, -1.0):
            conf[_id] = v
    return conf


def _ece(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 15) -> float:
    """
    Expected Calibration Error (ECE)
    """
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (conf >= lo) & (conf < hi)
        if not m.any():
            continue
        acc = (pred[m] == y_true[m]).mean()
        ece += float(abs(acc - conf[m].mean()) * (m.mean()))
    return float(ece)


def fit_meta_temp(model: nn.Module, dl_va: DataLoader, device: torch.device, grid: Optional[np.ndarray] = None) -> float:
    """
    검증셋에서 메타 로짓 Temperature를 NLL 최소화로 적합.
    """
    import numpy as _np
    model.eval()
    grid = grid if grid is not None else _np.linspace(0.5, 5.0, 91)
    ce = nn.CrossEntropyLoss(reduction="mean")
    logits, ys = [], []
    with torch.no_grad():
        for batch in dl_va:
            xb, yb_m = batch[0].to(device), batch[1].to(device)
            log_m, _ = model(xb)
            logits.append(log_m.cpu())
            ys.append(yb_m.cpu())
    if not logits:
        return 1.0
    Z = torch.cat(logits, 0).numpy()
    y = torch.cat(ys, 0).numpy()
    bestT, bestNLL = 1.0, 1e9
    for T in grid:
        z = Z / max(T, 1e-6); z = z - z.max(1, keepdims=True)
        logp = z - _np.log(_np.exp(z).sum(1, keepdims=True))
        nll = -float(logp[_np.arange(y.size), y].mean())
        if nll < bestNLL:
            bestNLL, bestT = nll, float(T)
    return float(bestT)


# ----------------------------
# 3. Dataset / Meta-model / Trainer
# ----------------------------
class MetaDataset(Dataset):
    """
    메타모델 학습용 데이터셋
    - 입력: 임베딩 X[N,D], ids[N], 라벨맵(id -> (main, sub)), 서브엔SEMBLE
    - 특징: per-main 통계 5종 + (옵션) 원시 확률 + (옵션) 임베딩
    - 출력: (feats, y_main, y_sub_global)
      * y_main: 0..3 (희/노/애/락)
      * y_sub:  글로벌 sub 인덱스(희 서브부터 이어붙임), 없는 경우 -1 (loss ignore_index)
    """
    def __init__(self,
                 X: np.ndarray,
                 ids: List[str],
                 y_map: Dict[str, Tuple[str, Optional[str]]],
                 sub_ens: SubEnsemble,
                 sub_list_by_main: Dict[str, List[str]],
                 use_embedding: bool = True,
                 cache_dir: Optional[str] = None):
        self.labels_main = {"희": 0, "노": 1, "애": 2, "락": 3}
        mains = ["희", "노", "애", "락"]

        # 1) join
        keep_idx = []
        y_main_idx = []
        y_sub_name = []
        for i, _id in enumerate(ids):
            if _id not in y_map:
                continue
            m, s = y_map[_id]
            m_idx = self.labels_main.get(m, -1)
            if m_idx < 0:
                continue
            keep_idx.append(i)
            y_main_idx.append(m_idx)
            y_sub_name.append((m, s if isinstance(s, str) and s.strip() else None))

        if not keep_idx:
            raise ValueError("no joined samples for meta dataset")

        Xj = X[np.asarray(keep_idx)]

        self.ids_joined = [ids[i] for i in keep_idx]  # ← join 후 id 보관

        # 2) 서브모델 배치 추론 → per-main 확률 (NPZ 캐시 사용)
        cache_root = cache_dir or os.getenv("SUB_CACHE_DIR", str(Path("src/models/unified") / "cache"))
        sub_probs = cached_sub_probs(Xj, sub_ens, cache_root)  # {main: [N, C_m]}

        # 3) 글로벌 sub 인덱스 사전 (희→노→애→락 순으로 이어붙임)
        self.sub_index: Dict[str, Dict[str, int]] = {}
        offset = 0
        for m in mains:
            subs = sub_list_by_main[m]
            self.sub_index[m] = {s: (offset + i) for i, s in enumerate(subs)}
            offset += len(subs)
        n_sub_total = offset

        # 4) 특징 구성
        feats = build_feature_for_batch(Xj, sub_probs, stats_only=not use_embedding)  # [N, feat_dim]
        # 5) 라벨 배열
        y_main_arr = np.asarray(y_main_idx, dtype=np.int64)
        y_sub_arr = []
        for (m, s) in y_sub_name:
            if s is None or s not in self.sub_index[m]:
                y_sub_arr.append(-1)
            else:
                y_sub_arr.append(self.sub_index[m][s])
        y_sub_arr = np.asarray(y_sub_arr, dtype=np.int64)

        # 6) 텐서화
        self.X = torch.tensor(feats, dtype=torch.float32)
        self.y_main = torch.tensor(y_main_arr, dtype=torch.long)
        self.y_sub = torch.tensor(y_sub_arr, dtype=torch.long)
        self.n_sub_total = int(n_sub_total)

        # 샘플 가중(기본 1.0)
        self.sample_weight = torch.ones(len(self.ids_joined), dtype=torch.float32)

        # 간단 통계
        self._stats = {
            "N": int(self.X.shape[0]),
            "feat_dim": int(self.X.shape[1]),
            "main_counts": {k: int((self.y_main.numpy() == v).sum()) for k, v in self.labels_main.items()},
            "sub_labeled": int((self.y_sub.numpy() >= 0).sum()),
        }
        log(f"[meta-ds] N={self._stats['N']} | feat_dim={self._stats['feat_dim']} | "
            f"sub_labeled={self._stats['sub_labeled']}/{self._stats['N']} | "
            f"main_counts={self._stats['main_counts']}")

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, i: int):
        # (feats, y_main, y_sub, sample_weight)
        return self.X[i], self.y_main[i], self.y_sub[i], self.sample_weight[i]


class MetaMLP(nn.Module):
    """
    메타 분류기
    - backbone(shared) → head_main(4-class), head_sub(global-sub-class)
    """
    def __init__(self, in_dim: int, n_main: int, n_sub_total: int, hidden: int = 512, drop: float = 0.2):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(drop),
        )
        self.head_main = nn.Linear(hidden // 2, n_main)
        self.head_sub = nn.Linear(hidden // 2, n_sub_total)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0.0
                    nn.init.uniform_(m.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.dtype != torch.float32:
            x = x.float()
        z = self.backbone(x)
        return self.head_main(z), self.head_sub(z)


class MetaDatasetR(MetaDataset):
    """
    MetaDataset 확장: Reasoner 보조 라벨을 텐서로 제공
    - cause_vocab_size: 보카브 크기
    - aux_by_id: id -> {"target_unc": float, "cause_ids": [int], "target_pivot_idx": int}
    """
    def __init__(self,
                 X: np.ndarray,
                 ids: List[str],
                 y_map: Dict[str, Tuple[str, Optional[str]]],
                 sub_ens: SubEnsemble,
                 sub_list_by_main: Dict[str, List[str]],
                 aux_by_id: Dict[str, dict],
                 cause_vocab_size: int,
                 use_embedding: bool = True,
                 cache_dir: Optional[str] = None):
        super().__init__(X, ids, y_map, sub_ens, sub_list_by_main, use_embedding=use_embedding, cache_dir=cache_dir)
        N = self.X.shape[0]
        self.cause_vocab_size = int(cause_vocab_size)

        # 보조 라벨 정렬
        unc   = np.zeros((N,), dtype=np.float32)
        piv   = np.full((N,), -1, dtype=np.int64)
        cmhot = np.zeros((N, self.cause_vocab_size), dtype=np.float32)

        for i, _id in enumerate(self.ids_joined):
            rec = aux_by_id.get(_id, {})
            u = rec.get("target_unc", None)
            if isinstance(u, (int, float)): unc[i] = float(u)
            pi = rec.get("target_pivot_idx", None)
            if isinstance(pi, int): piv[i] = int(pi)
            for cid in rec.get("cause_ids", []) or []:
                if isinstance(cid, int) and 0 <= cid < self.cause_vocab_size:
                    cmhot[i, cid] = 1.0

        self.target_unc = torch.tensor(unc, dtype=torch.float32)
        self.target_pivot_idx = torch.tensor(piv, dtype=torch.long)
        self.cause_multi_hot = torch.tensor(cmhot, dtype=torch.float32)

    def __getitem__(self, i: int):
        # (feat, y_main, y_sub, target_unc, cause_mh, target_pivot, sample_weight)
        return (self.X[i],
                self.y_main[i],
                self.y_sub[i],
                self.target_unc[i],
                self.cause_multi_hot[i],
                self.target_pivot_idx[i],
                self.sample_weight[i])

class UnifiedWithReasoner(nn.Module):
    def __init__(self, in_dim: int, n_main: int, n_sub_total: int, cause_vocab: int = 128, use_pivot: bool = True):
        super().__init__()
        self.backbone = MetaMLP(in_dim, n_main, n_sub_total)  # 기존 메타모델 재사용
        self.reasoner = ReasonerHead(in_dim, cause_vocab=cause_vocab, use_pivot=use_pivot)

    def forward(self, x: torch.Tensor, sent_feats: torch.Tensor = None, sent_mask: torch.Tensor = None):
        log_main, log_sub = self.backbone(x)
        unc, cause_logits, pivot_logits = self.reasoner(x, sent_feats, sent_mask)
        return log_main, log_sub, unc, cause_logits, pivot_logits


def compute_reasoner_losses(outputs, batch, lam_unc=0.2, lam_cause=0.1, lam_pivot=0.2):
    log_main, log_sub, unc, cause_logits, pivot_logits = outputs
    xb, yb_m, yb_s, t_unc, cause_mh, t_piv = batch

    ce_main = nn.CrossEntropyLoss()
    ce_sub  = nn.CrossEntropyLoss(ignore_index=-1)

    loss = ce_main(log_main, yb_m) + 0.5 * ce_sub(log_sub, yb_s)

    # 불확실성 회귀
    if unc is not None and t_unc is not None:
        loss += lam_unc * nn.functional.l1_loss(unc.squeeze(1), t_unc)

    # 원인 보카브 멀티라벨
    if cause_logits is not None and cause_mh is not None and cause_mh.numel() > 0:
        loss += lam_cause * nn.functional.binary_cross_entropy_with_logits(cause_logits, cause_mh)

    # 피벗 포인터
    if pivot_logits is not None and t_piv is not None:
        valid = (t_piv >= 0)
        if valid.any():
            loss += lam_pivot * nn.functional.cross_entropy(pivot_logits[valid], t_piv[valid])

    return loss


def compute_reasoner_losses_weighted(outputs, batch, lam_unc: float = 0.2, lam_cause: float = 0.1, lam_pivot: float = 0.2, sw: Optional[torch.Tensor] = None):
    """
    샘플별 가중치를 적용한 Reasoner 손실.
    batch: (xb, yb_m, yb_s, t_unc, cause_mh, t_piv, sw_b)
    """
    log_main, log_sub, unc, cause_logits, pivot_logits = outputs
    xb, yb_m, yb_s, t_unc, cause_mh, t_piv, sw_b = batch
    if sw is None:
        sw = sw_b
    sw = sw.clamp_min(1e-6)

    # main/sub
    loss_m = nn.functional.cross_entropy(log_main, yb_m, reduction='none')  # [B]
    loss_s = nn.functional.cross_entropy(log_sub, yb_s, ignore_index=-1, reduction='none')  # [B]
    mask_s = (yb_s >= 0).float()
    loss = (loss_m * sw).mean() + 0.5 * ((loss_s * mask_s * sw).sum() / mask_s.sum().clamp_min(1.0))

    # unc (L1)
    if unc is not None and t_unc is not None:
        l_unc = (unc.squeeze(1) - t_unc).abs()  # [B]
        loss += lam_unc * (l_unc * sw).mean()

    # cause (BCE) — per-sample avg → 가중
    if cause_logits is not None and cause_mh is not None and cause_mh.numel() > 0:
        bce = nn.functional.binary_cross_entropy_with_logits(cause_logits, cause_mh, reduction='none')  # [B, V]
        loss += lam_cause * ((bce.mean(dim=1) * sw).mean())

    # pivot
    if pivot_logits is not None and t_piv is not None:
        valid = (t_piv >= 0)
        if valid.any():
            ce_p = nn.functional.cross_entropy(pivot_logits[valid], t_piv[valid], reduction='none')  # [Bv]
            loss += lam_pivot * (ce_p * sw[valid]).mean()

    return loss

def train_unified_with_reasoner(X: np.ndarray,
                                ids: List[str],
                                labels_jsonl: str,
                                y_map: Dict[str, Tuple[str, Optional[str]]],
                                sub_ens: SubEnsemble,
                                sub_list_by_main: Dict[str, List[str]],
                                outdir: str,
                                cause_vocab_path: str = "data/cause_vocab.json",
                                device: str = "cuda",
                                epochs: int = 10,
                                batch: int = 128,
                                seed: int = 42,
                                freeze_backbone_epochs: int = 5,
                                lam_unc: float = 0.2,
                                lam_cause: float = 0.1,
                                lam_pivot: float = 0.2,
                                use_embedding: bool = True) -> str:

    set_seed(seed)
    dev = torch.device(device)

    # 1) 기본 메타 데이터셋(특징/조인, 캐시 사용)
    cache_root = str(Path(outdir) / "cache")
    base_ds = MetaDataset(X, ids, y_map, sub_ens, sub_list_by_main, use_embedding=use_embedding, cache_dir=cache_root)
    N, feat_dim = base_ds.X.shape
    n_main = 4; n_sub_total = base_ds.n_sub_total

    # 2) 풀 라벨 테이블에서 보조 라벨 맵 구성
    aux_tbl = load_labels_jsonl_full(labels_jsonl)  # id -> full
    cause_vocab_size = _load_cause_vocab_size(cause_vocab_path, default=128)

    # Reasoner DS
    dsR = MetaDatasetR(X, ids, y_map, sub_ens, sub_list_by_main,
                       aux_by_id=aux_tbl, cause_vocab_size=cause_vocab_size,
                       use_embedding=use_embedding, cache_dir=cache_root)

    # 샘플별 신뢰도 가중치 세팅(Reasoner)
    try:
        conf_map_R = load_label_confidence(labels_jsonl)
        wR = [conf_map_R.get(_id, 1.0) for _id in dsR.ids_joined]
        dsR.sample_weight = torch.tensor(wR, dtype=torch.float32)
    except Exception:
        pass

    # split(메인 라벨 기반 계층 분할) — 이후 TS/스케일러에서 사용
    tr_idx, va_idx = _split_stratified(dsR.y_main.numpy(), ratio=0.2, seed=seed)
    tr_ds = torch.utils.data.Subset(dsR, tr_idx)
    va_ds = torch.utils.data.Subset(dsR, va_idx)

    # --- Validation 기반 서브 Temperature 보정 ---
    try:
        id2pos = {s_id: i for i, s_id in enumerate(ids)}
        Xj_all = np.vstack([X[id2pos[_id]] for _id in dsR.ids_joined]).astype(np.float32, copy=False)
        va_X = Xj_all[va_idx]
        sub_probs_val = sub_ens.predict_proba_batch(va_X, batch=2048)
        y_main_val = dsR.y_main[va_idx].cpu().numpy()
        y_sub_name_val = [y_map[_id] for _id in [dsR.ids_joined[i] for i in va_idx]]
        init_T = {m: sub_ens.models[m][2] for m in sub_list_by_main.keys()}
        T_meta = fit_temperature_on_val(sub_probs_val, y_main_val, y_sub_name_val, sub_list_by_main, init_T=init_T)
        for m in sub_list_by_main.keys():
            mdl, subs, _T = sub_ens.models[m]
            sub_ens.models[m] = (mdl, subs, float(T_meta.get(m, _T)))
        # 보정된 T로 전체 특징 재계산
        sub_probs_full = sub_ens.predict_proba_batch(Xj_all, batch=2048)
        feats_full = build_feature_for_batch(Xj_all, sub_probs_full, stats_only=not use_embedding)
        dsR.X = torch.tensor(feats_full, dtype=torch.float32)
        feat_dim = dsR.X.shape[1]
    except Exception as _e:
        log(f"[reasoner] temperature fit skipped: {type(_e).__name__}: {_e}")

    # --- Feature 표준화 (train 기준) ---
    feat_scaler = FeatureScaler()
    try:
        feat_scaler.fit(dsR.X[tr_idx])
        dsR.X = feat_scaler.transform(dsR.X)
    except Exception as _e:
        log(f"[reasoner] feature scaling skipped: {type(_e).__name__}: {_e}")
        feat_scaler = FeatureScaler()

    # Loader (학습 로더는 셔플하여 일반화 향상; 배치별 가중치는 배치에서 직접 제공됨)
    num_workers = max(0, min(32, (os.cpu_count() or 1) - 1))  # 라이젠 AI 9/HX 370에 맞춰 워커 수 대폭 증가
    dl_tr = DataLoader(tr_ds, batch_size=min(batch, max(1, len(tr_ds))), shuffle=True,
                       num_workers=num_workers, pin_memory=(dev.type == "cuda"),
                       persistent_workers=(num_workers > 0))
    dl_va = DataLoader(va_ds, batch_size=min(batch, max(1, len(va_ds))), shuffle=False,
                       num_workers=num_workers, pin_memory=(dev.type == "cuda"))

    # 3) 모델
    model = UnifiedWithReasoner(feat_dim, n_main, n_sub_total,
                                cause_vocab=cause_vocab_size, use_pivot=True).to(dev)

    # Stage-1: backbone freeze → reasoner 먼저 학습
    for p in model.backbone.parameters(): p.requires_grad = False
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-4)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=(dev.type=="cuda"))
    amp_ctx = torch.cuda.amp.autocast(enabled=(dev.type=="cuda"), dtype=torch.float16)

    def _run_epoch(dl, is_train: bool):
        if is_train:
            model.train()
        else:
            model.eval()
        tot, seen = 0.0, 0
        for batch in dl:
            # 모든 텐서를 dev로 (샘플별 가중 sw 포함)
            xb, yb_m, yb_s, t_unc, cause_mh, t_piv, sw = [
                t.to(dev) if torch.is_tensor(t) else t for t in batch
            ]
            batch_dev = (xb, yb_m, yb_s, t_unc, cause_mh, t_piv, sw)

            opt.zero_grad(set_to_none=True)
            try:
                with amp_ctx:
                    outputs = model(xb, None, None)  # sent_feats/mask 없으면 None
                    loss = compute_reasoner_losses_weighted(outputs, batch_dev, lam_unc, lam_cause, lam_pivot, sw=sw)
                if is_train:
                    grad_scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    grad_scaler.step(opt); grad_scaler.update()
            except torch.cuda.OutOfMemoryError:
                # AMP OOM → FP32로 재시도
                torch.cuda.empty_cache()
                fp32_ctx = torch.cuda.amp.autocast(enabled=False, dtype=torch.float16)
                with fp32_ctx:
                    outputs = model(xb, None, None)
                    loss = compute_reasoner_losses_weighted(outputs, batch_dev, lam_unc, lam_cause, lam_pivot, sw=sw)
                if is_train:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step(); opt.zero_grad(set_to_none=True)

            tot += float(loss.detach().item()) * xb.size(0)
            seen += xb.size(0)
        return tot / max(1, seen)
    for _ in range(max(0, freeze_backbone_epochs)):
        _ = _run_epoch(dl_tr, True)

    # Stage-2: joint finetune
    for p in model.parameters(): p.requires_grad = True
    opt = torch.optim.AdamW(model.parameters(), lr=1.5e-4, weight_decay=1e-4)
    best_f1 = -1.0; best_state = None

    # 간단한 val 루프로 macro-F1 점검 (메인만)
    def _eval_meta(dl, meta_temp_eval: float = 1.0):
        model.eval()
        with torch.no_grad():
            tot, acc, f1, extra = _eval_epoch(model.backbone, dl, dev, 4, meta_temp=meta_temp_eval) if isinstance(model, UnifiedWithReasoner) else _eval_epoch(model, dl, dev, 4, meta_temp=meta_temp_eval)
        return tot, f1

    for ep in range(1, epochs+1):
        tr_loss = _run_epoch(dl_tr, True)
        va_loss, va_f1 = _eval_meta(dl_va)
        # ===== 개선사항 5: train_loss 가시성 개선 =====
        log(f"[reasoner] ep {ep:02d}/{epochs} | tr_loss={tr_loss:.6f} | va_loss={va_loss:.6f} | va_macro_f1={va_f1:.4f}")
        if va_f1 > best_f1 + 1e-6:
            best_f1 = va_f1
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    # save
    out = ensure_dir(outdir)
    path = out / "unified_model.pt"
    if best_state is None:
        best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    # 메타 로짓 Temperature 적합(Reasoner 백본 기준)
    try:
        base_mdl = model.backbone if isinstance(model, UnifiedWithReasoner) else model
        meta_T = fit_meta_temp(base_mdl, dl_va, dev)
    except Exception:
        meta_T = 1.0

    # 개선사항 3: 서브 리스트 및 ckpt 해시 저장
    sub_lists = {m: sub_ens.models[m][1] for m in ("희", "노", "애", "락")}
    ck_hash = _hash_ckpts(getattr(sub_ens, "run_dir", ""))
    
    torch.save({
        "state_dict": best_state,
        "meta": {
            "feat_dim": int(feat_dim),
            "n_main": 4,
            "n_sub_total": int(n_sub_total),
            "reasoner": True,
            "cause_vocab_size": int(cause_vocab_size),
            "best_macro_f1": float(best_f1),
            "feat_mu": (feat_scaler.mu.cpu().numpy().tolist() if feat_scaler.mu is not None else None),
            "feat_sigma": (feat_scaler.sigma.cpu().numpy().tolist() if feat_scaler.sigma is not None else None),
            "sub_temp_fitted": {m: float(sub_ens.models[m][2]) for m in sub_list_by_main.keys()},
            "meta_temp_fitted": float(meta_T),
            "sub_lists": sub_lists,           # ★ 추가
            "sub_ckpt_hash": ck_hash          # ★ 추가
        },
        "rng": {
            "py": random.getstate(),
            "np": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "torch_cuda": (torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
        }
    }, path)
    log(f"[saved:reasoner] {path} | best_macro_f1={best_f1:.6f}")

    # 일관 리포팅: 메타 평가 리포트 저장(ECE 포함)
    try:
        ds_eval = MetaDataset(X, ids, y_map, sub_ens, sub_list_by_main, use_embedding=use_embedding, cache_dir=cache_root)
        mdl_eval = MetaMLP(feat_dim, 4, n_sub_total).to(dev)
        # Unified→MetaMLP 가중치 매핑(backbone. 프리픽스 제거)
        try:
            remap = {k[len("backbone."):]: v for k, v in best_state.items() if k.startswith("backbone.")}
            mdl_eval.load_state_dict(remap, strict=False)
        except Exception:
            mdl_eval.load_state_dict(best_state, strict=False)
        mdl_eval.eval()
        meta_T = fit_meta_temp(mdl_eval, DataLoader(va_ds, batch_size=256, shuffle=False,
                                                    num_workers=0, pin_memory=(dev.type=="cuda")), dev)
        ck = _trusted_torch_load(path, map_location="cpu")
        fm = (ck.get("meta") or {}).get("feat_mu"); fs = (ck.get("meta") or {}).get("feat_sigma")
        _evaluate_unified(mdl_eval, ds_eval, dev, Path(outdir),
                          meta_temp=meta_T, feat_mu=fm, feat_sigma=fs)
    except Exception as _e:
        log(f"[reasoner] eval report skipped: {type(_e).__name__}: {_e}")

    return str(path)


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_cls: int) -> float:
    f1s = []
    for c in range(n_cls):
        tp = int(((y_true == c) & (y_pred == c)).sum())
        fp = int(((y_true != c) & (y_pred == c)).sum())
        fn = int(((y_true == c) & (y_pred != c)).sum())
        p = tp / max(1, tp + fp)
        r = tp / max(1, tp + fn)
        f = 2 * p * r / max(1e-9, p + r)
        f1s.append(f)
    return float(np.mean(f1s) if f1s else 0.0)


def _split_stratified(y: np.ndarray, ratio: float = 0.2, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    메인 라벨로 계층 분할. 각 클래스의 비율을 유지.
    """
    rng = np.random.RandomState(seed)
    idx = np.arange(len(y))
    va_idx = []
    for c in np.unique(y):
        ci = idx[y == c]
        rng.shuffle(ci)
        k = max(1, int(len(ci) * ratio))
        va_idx.append(ci[:k])
    va_idx = np.concatenate(va_idx) if va_idx else np.array([], dtype=int)
    va_idx = np.unique(va_idx)
    tr_mask = np.ones(len(y), dtype=bool)
    tr_mask[va_idx] = False
    tr_idx = np.where(tr_mask)[0]
    rng.shuffle(tr_idx)
    rng.shuffle(va_idx)
    if va_idx.size == 0:
        va_idx = np.array([tr_idx[0]])
        tr_idx = tr_idx[1:]
    return tr_idx, va_idx


@torch.no_grad()
def _eval_epoch(model: MetaMLP, dl: DataLoader, device: torch.device, n_main: int, meta_temp: float = 1.0) -> Tuple[float, float, float, Dict[str, Any]]:
    """
    검증 루프: (loss_mean, acc, macro_f1, extra) 반환
    - sub loss는 무시(평가 단순화)
    """
    model.eval()
    ce = nn.CrossEntropyLoss(reduction="mean")
    tot_loss = 0.0
    tot = 0
    correct = 0
    all_t, all_p = [], []
    all_pm = []
    for batch in dl:
        # MetaDataset: (X, y_main, y_sub, [sample_weight])
        xb = batch[0].to(device); yb_m = batch[1].to(device)
        log_m, _ = model(xb)
        loss = ce(log_m, yb_m)
        tot_loss += float(loss) * xb.size(0)
        pm = torch.softmax(log_m / max(float(meta_temp), 1e-6), dim=1).cpu().numpy()
        pred = pm.argmax(1)
        correct += int((pred == yb_m.cpu().numpy()).sum())
        tot += int(yb_m.size(0))
        all_t.append(yb_m.cpu().numpy()); all_p.append(pred)
        all_pm.append(pm)
    if tot == 0:
        return 0.0, 0.0, 0.0, {}
    targets = np.concatenate(all_t, axis=0)
    preds = np.concatenate(all_p, axis=0)
    f1 = macro_f1(targets, preds, n_cls=n_main)
    pm_all = np.vstack(all_pm)
    try:
        from sklearn.metrics import balanced_accuracy_score  # type: ignore
        bal_acc = float(balanced_accuracy_score(targets, preds))
    except Exception:
        bal_acc = 0.0
    ece_val = _ece(pm_all, targets, n_bins=15)
    extra = {"ece": float(ece_val), "balanced_acc": bal_acc}
    return tot_loss / tot, correct / tot, f1, extra


def train_meta(X: np.ndarray,
               ids: List[str],
               y_map: Dict[str, Tuple[str, Optional[str]]],
               sub_ens: SubEnsemble,
               sub_list_by_main: Dict[str, List[str]],
               outdir: str,
               device: str = "cuda",
               epochs: int = 12,
               batch: int = 128,
               use_embedding: bool = True,
               seed: int = 42,
               patience: int = 3,           # ✅ CLI에서 넘어온 patience 사용 (0이면 조기종료 끔)
               min_epochs: int = 6,         # ✅ 최소 에폭 가드
               labels_jsonl: Optional[str] = None,
               grad_accum: int = 1
               ) -> str:
    """
    서브 4모델의 출력을 특징으로 메타모델을 학습.
    - 조기종료: macro_f1 기준, patience 사용 (0이면 비활성화)
    - AMP/clip/scheduler 포함
    - 소량 데이터 안정화를 위해 main 계층에 간단한 클래스 가중치 적용
    """
    set_seed(seed)
    dev = torch.device(device)

    # Dataset (서브확률 캐시 사용)
    cache_root = str(Path(outdir) / "cache")
    ds = MetaDataset(X, ids, y_map, sub_ens, sub_list_by_main, use_embedding=use_embedding, cache_dir=cache_root)
    N, feat_dim = ds.X.shape
    n_main = 4
    n_sub_total = ds.n_sub_total

    # split (메인 계층)
    tr_idx, va_idx = _split_stratified(ds.y_main.numpy(), ratio=0.2, seed=seed)
    tr_ds = torch.utils.data.Subset(ds, tr_idx)
    va_ds = torch.utils.data.Subset(ds, va_idx)

    # --- Validation 기반 서브 Temperature 보정 ---
    try:
        # ds.ids_joined 순서로 Xj를 복원한 뒤, 검증 부분을 추출
        id2pos = {s_id: i for i, s_id in enumerate(ids)}
        Xj_all = np.vstack([X[id2pos[_id]] for _id in ds.ids_joined]).astype(np.float32, copy=False)
        va_X = Xj_all[va_idx]
        sub_probs_val = sub_ens.predict_proba_batch(va_X, batch=2048)
        y_main_val = ds.y_main[va_idx].cpu().numpy()
        y_sub_name_val = [y_map[_id] for _id in [ds.ids_joined[i] for i in va_idx]]
        init_T = {m: sub_ens.models[m][2] for m in sub_list_by_main.keys()}
        T_meta = fit_temperature_on_val(sub_probs_val, y_main_val, y_sub_name_val, sub_list_by_main, init_T=init_T)
        # 적용
        for m in sub_list_by_main.keys():
            mdl, subs, _T = sub_ens.models[m]
            sub_ens.models[m] = (mdl, subs, float(T_meta.get(m, _T)))
        # 보정된 T로 전체 특징 재계산
        sub_probs_full = sub_ens.predict_proba_batch(Xj_all, batch=2048)
        feats_full = build_feature_for_batch(Xj_all, sub_probs_full, stats_only=not use_embedding)
        ds.X = torch.tensor(feats_full, dtype=torch.float32)
        feat_dim = ds.X.shape[1]
    except Exception as _e:
        T_meta = {m: sub_ens.models[m][2] for m in sub_list_by_main.keys()}
        log(f"[meta] temperature fit skipped: {type(_e).__name__}: {_e}")

    # --- Feature 표준화 (train 기준) ---
    feat_scaler = FeatureScaler()
    try:
        feat_scaler.fit(ds.X[tr_idx])
        ds.X = feat_scaler.transform(ds.X)
    except Exception as _e:
        log(f"[meta] feature scaling skipped: {type(_e).__name__}: {_e}")
        feat_scaler = FeatureScaler()  # 빈 상태 저장

    # DataLoader (shuffle=False: 샘플 가중 정합을 위해 순차 인출)
    num_workers = max(0, min(32, (os.cpu_count() or 1) - 1))  # 라이젠 AI 9/HX 370에 맞춰 워커 수 대폭 증가
    dl_tr = DataLoader(tr_ds, batch_size=min(batch, max(1, len(tr_ds))), shuffle=True,
                       num_workers=num_workers, pin_memory=(dev.type == "cuda"),
                       persistent_workers=(num_workers > 0))
    dl_va = DataLoader(va_ds, batch_size=min(batch, max(1, len(va_ds))), shuffle=False,
                       num_workers=num_workers, pin_memory=(dev.type == "cuda"),
                       persistent_workers=False)

    # --- 라벨 신뢰도 샘플 가중 로드 ---
    if isinstance(labels_jsonl, str) and Path(labels_jsonl).exists():
        try:
            conf_map = load_label_confidence(labels_jsonl)
            w = [conf_map.get(_id, 1.0) for _id in ds.ids_joined]
            ds.sample_weight = torch.tensor(w, dtype=torch.float32)
        except Exception as _e:
            log(f"[meta] confidence weights skipped: {type(_e).__name__}: {_e}")

    # ===== Model / loss / opt / sched =====
    model = MetaMLP(feat_dim, n_main, n_sub_total).to(dev)

    # ✅ (가벼운) 클래스 가중치: 불균형 완화 (sum-normalized inverse frequency)
    with torch.no_grad():
        counts = torch.bincount(ds.y_main, minlength=n_main).float()
        counts = torch.clamp(counts, min=1.0)
        inv = counts.sum() / counts          # inversely proportional
        weight_main = (inv / inv.mean()).to(dev)  # normalize for stability

    ce_main = nn.CrossEntropyLoss(weight=weight_main)  # 가중치 적용
    ce_sub  = nn.CrossEntropyLoss(ignore_index=-1)

    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2)

    # AMP
    use_amp = (dev.type == "cuda")
    try:
        autocast = torch.amp.autocast
        GradScaler = torch.amp.GradScaler
        amp_ctx = autocast(device_type="cuda", enabled=use_amp, dtype=torch.float16)
        grad_scaler = GradScaler(enabled=use_amp)
    except Exception:
        amp_ctx = torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16)
        grad_scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # ===== Train loop =====
    best_state = None
    best_f1 = -1.0
    bad = 0
    alpha_sub = 0.5  # 서브 손실 비중

    # 간단 VAL 헬퍼(메타 온도 사용 가능)
    def _eval_meta(dl, meta_temp_eval: float = 1.0):
        model.eval()
        with torch.no_grad():
            tot, acc, f1, extra = _eval_epoch(model, dl, dev, 4, meta_temp=meta_temp_eval)
        return tot, f1

    acc = max(1, int(grad_accum))

    for ep in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        seen = 0
        step = 0

        for batch in dl_tr:
            # (X, y_main, y_sub, sample_weight)
            xb, yb_m, yb_s, sw = batch[0].to(dev), batch[1].to(dev), batch[2].to(dev), batch[3].to(dev)
            opt.zero_grad(set_to_none=True)

            try:
                with amp_ctx:
                    log_m, log_s = model(xb)
                    loss_m = nn.functional.cross_entropy(log_m, yb_m, weight=weight_main, reduction='none')
                    loss_s = nn.functional.cross_entropy(log_s, yb_s, ignore_index=-1, reduction='none')
                    mask_s = (yb_s >= 0).float()
                    loss = ( (loss_m * sw).mean() + alpha_sub * (loss_s * sw * mask_s).sum() / (mask_s.sum().clamp_min(1.0)) )
                    loss = loss / acc
                grad_scaler.scale(loss).backward()
                step += 1
                if step % acc == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    grad_scaler.step(opt); grad_scaler.update()
                    opt.zero_grad(set_to_none=True)
            except torch.cuda.OutOfMemoryError as _oom:
                if use_amp and (dev.type == "cuda"):
                    log("[meta] AMP OOM → fallback FP32")
                    use_amp = False
                    try:
                        autocast = torch.amp.autocast
                        GradScaler = torch.amp.GradScaler
                        amp_ctx = autocast(device_type="cuda", enabled=False, dtype=torch.float16)
                        grad_scaler = GradScaler(enabled=False)
                    except Exception:
                        amp_ctx = torch.cuda.amp.autocast(enabled=False, dtype=torch.float16)
                        grad_scaler = torch.cuda.amp.GradScaler(enabled=False)
                    torch.cuda.empty_cache()
                    # 재시도(FP32)
                    log_m, log_s = model(xb)
                    loss_m = nn.functional.cross_entropy(log_m, yb_m, weight=weight_main, reduction='none')
                    loss_s = nn.functional.cross_entropy(log_s, yb_s, ignore_index=-1, reduction='none')
                    mask_s = (yb_s >= 0).float()
                    loss = ( (loss_m * sw).mean() + alpha_sub * (loss_s * sw * mask_s).sum() / (mask_s.sum().clamp_min(1.0)) )
                    loss = loss / acc
                    loss.backward()
                    step += 1
                    if step % acc == 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        opt.step(); opt.zero_grad(set_to_none=True)
                else:
                    torch.cuda.empty_cache()
                    log("[meta] OOM: batch skipped")
                    continue

            if not torch.isfinite(loss):
                log("[meta] non-finite loss; skip step")
                continue

            tr_loss += float(loss.detach().item()) * xb.size(0); seen += xb.size(0)

        # [PATCH 2] 마지막 누적 스텝이 남았다면 한 번 더 step
        if (step % acc) != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            grad_scaler.step(opt); grad_scaler.update()
            opt.zero_grad(set_to_none=True)

        # VAL
        va_loss, va_f1 = _eval_meta(dl_va, meta_temp_eval=1.0)
        try:
            sched.step(va_f1)
        except (ValueError, RuntimeError) as e:
            # ReduceLROnPlateau 스케줄러 에러 처리
            if "step" in str(e).lower() or "scheduler" in str(e).lower():
                log(f"[warn] Scheduler step skipped: {e}")
            else:
                raise

        # 현재 LR 로깅(디버깅용)
        try:
            cur_lr = opt.param_groups[0]["lr"]
        except Exception:
            cur_lr = float("nan")

        # ===== 개선사항 5: train_loss 가시성 개선 =====
        log(f"[meta] ep {ep:02d}/{epochs} | tr_loss={tr_loss/max(1,seen):.6f} "
            f"| va_loss={va_loss:.6f} | va_macro_f1={va_f1:.4f} | lr={cur_lr:.2e}")

        # Early stopping on macro-F1
        improved = (va_f1 > best_f1 + 1e-6)
        if improved:
            best_f1 = va_f1
            bad = 0
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        else:
            bad += 1
            # ✅ patience==0 이면 early-stop 비활성화, min_epochs 가드 유지
            if patience > 0 and ep >= max(min_epochs, patience) and bad >= patience:
                log(f"[early-stop] no improvement {patience} epochs (best_f1={best_f1:.4f})")
                break

    # ===== save best =====
    out = ensure_dir(outdir)
    path = out / "unified_model.pt"
    if best_state is None:
        best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    # 메타 로짓 Temperature 적합
    try:
        meta_T = fit_meta_temp(model, dl_va, dev)
    except Exception:
        meta_T = 1.0

    # 개선사항 3: 서브 리스트 및 ckpt 해시 저장 (기존 코드 통합)
    sub_lists = {m: sub_list_by_main[m] for m in ["희", "노", "애", "락"]}
    ck_hash = _hash_ckpts(getattr(sub_ens, "run_dir", ""))
    
    # 최종 모델 저장 (원자적 저장)
    try:
        tmp_path = path.with_suffix(".tmp")
        torch.save({
            "state_dict": best_state,
            "meta": {
                "feat_dim": int(feat_dim),
                "n_main": int(n_main),
                "n_sub_total": int(n_sub_total),
                "use_embedding": bool(use_embedding),
                "sub_lists": sub_lists,                # ★ 개선사항 3
                "sub_ckpt_hash": ck_hash,               # ★ 개선사항 3
                "sub_run_dir": str(getattr(sub_ens, "run_dir", "")),  # 개선사항 6
                "best_macro_f1": float(best_f1),       # ✅ 기록
                "patience": int(patience),             # ✅ 기록
                "min_epochs": int(min_epochs),         # ✅ 기록
                "main_counts": {k: int((ds.y_main.numpy() == v).sum()) for k, v in ds.labels_main.items()},
                "N": int(N),
                "feat_mu": (feat_scaler.mu.cpu().numpy().tolist() if feat_scaler.mu is not None else None),
                "feat_sigma": (feat_scaler.sigma.cpu().numpy().tolist() if feat_scaler.sigma is not None else None),
                "sub_temp_fitted": {m: float(sub_ens.models[m][2]) for m in sub_list_by_main.keys()},
                "meta_temp_fitted": float(meta_T)
            },
            "rng": {
                "py": random.getstate(),
                "np": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "torch_cuda": (torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
            }
        }, tmp_path)
        # 원자적 교체
        tmp_path.replace(path)
        file_size_mb = path.stat().st_size / (1024 * 1024)
        log(f"[✓ 저장 완료] {path.name} ({file_size_mb:.2f} MB) | best_macro_f1={best_f1:.6f}")
    except Exception as e:
        log(f"[✗ 저장 실패] {path}: {e}")
        raise
    return str(path)



# ----------------------------
# 4. CLI
# ----------------------------
def _per_class_f1(y_true: np.ndarray, y_pred: np.ndarray, n_cls: int) -> Dict[int, float]:
    out = {}
    for c in range(n_cls):
        tp = int(((y_true == c) & (y_pred == c)).sum())
        fp = int(((y_true != c) & (y_pred == c)).sum())
        fn = int(((y_true == c) & (y_pred != c)).sum())
        p = tp / max(1, tp + fp)
        r = tp / max(1, tp + fn)
        f = 2 * p * r / max(1e-9, p + r)
        out[c] = float(f)
    return out

def _confusion(y_true: np.ndarray, y_pred: np.ndarray, n_cls: int) -> List[List[int]]:
    M = np.zeros((n_cls, n_cls), dtype=int)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < n_cls and 0 <= p < n_cls:
            M[t, p] += 1
    return M.tolist()

def predict_with_bundle(bundle_path: str, emb_jsonl: Optional[str] = None, vec: Optional[np.ndarray] = None,
                        id: Optional[str] = None, topk: int = 3, device: str = "auto"):
    """번들 파일만으로 예측 수행"""
    dev = torch.device(resolve_device(device))
    B = _trusted_torch_load(bundle_path, map_location="cpu")
    ck = B["unified"]
    meta = ck["meta"]
    mains = ["희", "노", "애", "락"]
    
    # 1) sub ensemble 재구성(메모리상)
    class _MemSub(SubEnsemble): 
        pass
    mem = _MemSub.__new__(_MemSub)
    mem.device = dev
    mem.mains = mains
    mem.in_dim = None
    mem.models = {}
    mem.present_mask_by_main = {}
    
    for m, info in B["sub_bundle"].items():
        mdl = SubModelCompat(info["in_dim"], len(info["sub_list"]))
        mdl.load_state_dict(info["state_dict"], strict=False)
        mdl.to(dev).eval()
        mem.models[m] = (mdl, info["sub_list"], float(info["T"]))
        mem.present_mask_by_main[m] = np.asarray(info.get("present_mask"), dtype=bool) if info.get("present_mask") is not None else np.ones((len(info["sub_list"]),), bool)
        mem.in_dim = mem.in_dim or int(info["in_dim"])
    
    # 2) 입력 벡터 확보
    if vec is None:
        X, ids = load_embeddings_jsonl(emb_jsonl)
        i = ids.index(id) if id else 0
        vec = X[i]
    
    # 3) 서브 확률 → 메타 특징 → 예측
    sub_out = mem.predict_proba(vec)
    feats = build_feature_for_id(vec, sub_out, stats_only=False)
    mu, sg = meta.get("feat_mu"), meta.get("feat_sigma")
    if isinstance(mu, list) and isinstance(sg, list):
        mu = np.asarray(mu, np.float32)
        sg = np.asarray(sg, np.float32)
        if mu.shape == feats.shape == sg.shape:
            feats = (feats - mu) / np.maximum(sg, 1e-8)
    
    mdl = MetaMLP(meta["feat_dim"], 4, meta["n_sub_total"]).to(dev)
    mdl.load_state_dict(ck["state_dict"], strict=False)
    mdl.eval()
    
    with torch.no_grad():
        y = torch.from_numpy(feats).unsqueeze(0).to(dev)
        log_m, log_s = mdl(y)
        Tm = float(meta.get("meta_temp_fitted", 1.0))
        pm = torch.softmax(log_m / max(Tm, 1e-6), 1).cpu().numpy().squeeze(0)
        ps = torch.softmax(log_s, 1).cpu().numpy().squeeze(0)
    
    # 4) top-k
    main_idx = int(pm.argmax())
    main = mains[main_idx]
    # saved sub_list 우선
    subs_saved = (meta.get("sub_lists") or {})
    order = []
    for mm in mains:
        order += (subs_saved.get(mm) or mem.models[mm][1])
    # 슬라이스 추출
    offs = 0
    spans = []
    for mm in mains:
        cnt = len(subs_saved.get(mm) or mem.models[mm][1])
        spans.append((offs, offs + cnt))
        offs += cnt
    lo, hi = spans[main_idx]
    part = ps[lo:hi]
    names = (subs_saved.get(main) or mem.models[main][1])
    k = min(topk, len(names))
    top_idx = np.argsort(part)[-k:][::-1]
    return main, float(pm[main_idx]), [(names[j], float(part[j])) for j in top_idx]


def pack_bundle(unified_ckpt_path: str, sub_ens: SubEnsemble, out_path: str) -> str:
    """unified + sub 모델들을 하나의 번들 파일로 패킹"""
    ck = _trusted_torch_load(unified_ckpt_path, map_location="cpu")
    bundle = {
        "schema_version": "1.0",
        "unified": ck,
        "sub_bundle": {}
    }
    for m in ("희", "노", "애", "락"):
        mdl, subs, T = sub_ens.models[m]
        bundle["sub_bundle"][m] = {
            "state_dict": {k: v.cpu() for k, v in mdl.state_dict().items()},
            "sub_list": subs,
            "present_mask": getattr(sub_ens, "present_mask_by_main", {}).get(m, None),
            "T": float(T),
            "in_dim": int(sub_ens.in_dim)
        }
    out = Path(out_path or (Path(unified_ckpt_path).with_name("unified_bundle.pt")))
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, out)
    log(f"[bundle] saved -> {out}")
    return str(out)


def _save_eval_report(outdir: Path,
                      y_true: np.ndarray, y_pred: np.ndarray,
                      n_cls: int, mains: List[str],
                      extra: Dict[str, Any]) -> None:
    acc = float((y_true == y_pred).mean()) if y_true.size else 0.0
    mf1 = macro_f1(y_true, y_pred, n_cls=n_cls) if y_true.size else 0.0
    report = {
        "N": int(y_true.size),
        "acc": acc,
        "macro_f1": float(mf1),
        "per_class_f1": {mains[i]: f for i, f in _per_class_f1(y_true, y_pred, n_cls).items()},
        "confusion": {"axes": mains, "matrix": _confusion(y_true, y_pred, n_cls)},
        **(extra or {})
    }
    ensure_dir(outdir)
    (outdir / "unified_eval.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"[report] eval -> {outdir/'unified_eval.json'}")

def _coverage_report(labels_jsonl: str, mains: List[str]) -> Dict[str, Any]:
    """라벨 파일에서 메인별 서브 커버리지·카운트를 요약."""
    by_main: Dict[str, Dict[str, int]] = {m: {} for m in mains}
    total = 0
    for o in read_jsonl(labels_jsonl):
        m = (
            o.get("main")
            or o.get("label_main")
            or (o.get("reasons") or {}).get("label_main")
            or (o.get("debug_meta") or {}).get("main")
            or o.get("pred_main")
        )
        s = (
            o.get("sub")
            or o.get("label_sub")
            or (o.get("reasons") or {}).get("label_sub")
            or o.get("label")
            or o.get("pred_sub")
        )
        if isinstance(m, str) and m in by_main:
            total += 1
            if isinstance(s, str) and s.strip():
                by_main[m][s] = by_main[m].get(s, 0) + 1
    cov = {
        m: {
            "n": sum(by_main[m].values()),
            "subs": len(by_main[m]),
            "per_sub_min": (min(by_main[m].values()) if by_main[m] else 0),
            "per_sub_max": (max(by_main[m].values()) if by_main[m] else 0),
        } for m in mains
    }
    return {"total": total, "by_main": cov}

def _evaluate_unified(
    mdl: MetaMLP, ds: MetaDataset, device: torch.device, outdir: Path,
    meta_temp: float = 1.0,
    feat_mu: Optional[np.ndarray] = None,
    feat_sigma: Optional[np.ndarray] = None,
) -> None:
    dl = DataLoader(ds, batch_size=256, shuffle=False,
                    num_workers=max(0, min(4, (os.cpu_count() or 1) - 1)),
                    pin_memory=(device.type == "cuda"))
    mdl.eval()
    ce = nn.CrossEntropyLoss(reduction="mean")
    tot_loss = 0.0; tot = 0; correct = 0
    ys, ps = [], []
    all_pm = []
    with torch.no_grad():
        for batch in dl:
            xb, yb_m = batch[0].to(device), batch[1].to(device)
            # --- APPLY SCALER (if present) ---
            if feat_mu is not None and feat_sigma is not None:
                try:
                    mu_t = torch.as_tensor(feat_mu, dtype=xb.dtype, device=device)
                    sg_t = torch.as_tensor(feat_sigma, dtype=xb.dtype, device=device)
                    if mu_t.shape == xb.shape[1:] == sg_t.shape:
                        xb = (xb - mu_t) / torch.clamp(sg_t, min=1e-6)
                except Exception:
                    pass
            log_m, _ = mdl(xb)
            loss = ce(log_m, yb_m)
            tot_loss += float(loss) * xb.size(0); tot += int(yb_m.size(0))
            pm = torch.softmax(log_m / max(float(meta_temp), 1e-6), dim=1).cpu().numpy()
            pred = pm.argmax(1)
            correct += int((pred == yb_m.cpu().numpy()).sum())
            ys.append(yb_m.cpu().numpy()); ps.append(pred); all_pm.append(pm)
    if tot == 0:
        log("[eval] empty dataset"); return
    y_true = np.concatenate(ys); y_pred = np.concatenate(ps)
    acc = correct / tot; mf1 = macro_f1(y_true, y_pred, n_cls=4)
    extra = {"ece": _ece(np.vstack(all_pm), y_true, n_bins=15)}
    log(f"[eval] N={tot} | acc={acc:.4f} | macro_f1={mf1:.4f} | ece={extra['ece']:.4f}")
    _save_eval_report(outdir, y_true, y_pred, 4, ["희","노","애","락"], extra=extra)

def main():
    ap = argparse.ArgumentParser("Unified meta-model (train/eval/predict/report)")
    ap.add_argument("--stage", choices=["train", "eval", "predict", "report", "export", "bundle"], default="train")
    ap.add_argument("--bundle-out", default=None, help="(bundle) unified+sub 묶음 ckpt 경로. 예: src/models/unified/unified_bundle.pt")
    ap.add_argument("--stage-bundle", action="store_true", help="현재 unified_model.pt + sub_들을 하나의 번들로 패킹")
    ap.add_argument("--bundle", default=None, help="unified_bundle.pt 경로(있으면 번들 서빙 경로 사용)")
    ap.add_argument("--emb-jsonl", dest="emb_jsonl", required=False, default=None, help="embeddings run.jsonl")
    ap.add_argument("--labels-jsonl", dest="labels_jsonl", required=False, default=None, help="labels *.jsonl (id,main,sub)")
    ap.add_argument("--sub-run-dir", dest="sub_run_dir", required=False, default=None,
                    help="folder with sub_희/노/애/락.pt (자동 탐색 지원)")
    ap.add_argument("--outdir", default="src/models/unified", help="unified model & reports output dir")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--no-embed", action="store_true", help="meta features without raw embedding")
    ap.add_argument("--seed", type=int, default=42)

    # ✅ 개선사항 1: 조기종료 제어 옵션
    ap.add_argument("--patience", type=int, default=3,
                    help="Early stopping patience (epochs). 0이면 조기종료 비활성화")
    ap.add_argument("--no-early-stop", action="store_true",
                    help="조기종료를 완전히 끕니다(=모든 에폭 수행)")
    ap.add_argument("--min-epochs", type=int, default=6,
                    help="조기종료를 보더라도 최소 보장 학습 에폭 수")

    # predict options
    ap.add_argument("--id", type=str, default=None, help="predict by embedding id (from --emb-jsonl)")
    ap.add_argument("--text-emb", nargs="+", type=float, help="embedding vector(384 floats) for single prediction")
    ap.add_argument("--topk", type=int, default=3, help="top-k sub labels under predicted main")

    ap.add_argument("--use-reasoner", action="store_true", help="Unified+Reasoner 학습/추론 사용")
    ap.add_argument("--cause-vocab", default="data/cause_vocab.json", help="원인 어휘 보카브 JSON 경로")
    ap.add_argument("--freeze-backbone-epochs", type=int, default=5)
    ap.add_argument("--lam-unc", type=float, default=0.2)
    ap.add_argument("--lam-cause", type=float, default=0.1)
    ap.add_argument("--lam-pivot", type=float, default=0.2)

    # 추가: 그래드 누적/메타 로짓 온도
    ap.add_argument("--grad-accum", type=int, default=1, help="gradient accumulation steps")
    ap.add_argument("--meta-temp", type=float, default=1.0, help="meta logits temperature at inference")
    ap.add_argument("--min-label-conf", type=float, default=0.0,
                    help="이 값 미만 신뢰도의 라벨은 메타 학습에서 제외(0이면 비활성)")

    args = ap.parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    mains = ["희","노","애","락"]
    
    # 개선사항 1: 입력 경로 자동화 - 최신 파일 자동 선택
    def _latest(globpat):
        from pathlib import Path
        p = sorted(Path(".").glob(globpat), key=lambda x: x.stat().st_mtime, reverse=True)
        return str(p[0]) if p else None
    
    if args.stage in ["train", "eval"]:
        if not args.emb_jsonl:
            args.emb_jsonl = _latest("src/embeddings/run*.jsonl")
            if not args.emb_jsonl:
                print("[error] no embeddings found under src/embeddings")
                return
            log(f"[auto] emb = {args.emb_jsonl}")
        
        if not args.labels_jsonl:
            args.labels_jsonl = _latest("data/labels*.bal.jsonl") or _latest("data/labels.quick*.jsonl")
            if not args.labels_jsonl:
                print("[error] no labels found under data/")
                return
            log(f"[auto] labels = {args.labels_jsonl}")

    # --- sub ensemble autodiscovery ---
    sub_dir = args.sub_run_dir
    if not sub_dir:
        latest = find_latest_sub_run_dir("src/models")
        if latest is None:
            # ★ 부분결측 허용: 아무거나 최신 폴더라도 선택
            any_latest = find_latest_run_dir("src/models")
            if any_latest is None:
                raise FileNotFoundError("no sub-run-dir and no latest under src/models")
            sub_dir = str(any_latest)
            log(f"[auto] sub-run-dir (partial) = {sub_dir}")
        else:
            sub_dir = str(latest)
            log(f"[auto] sub-run-dir = {sub_dir}")

    # bundle stage 처리
    if args.stage == "bundle":
        ck_path = str(Path(args.outdir) / "unified_model.pt")
        if not Path(ck_path).exists():
            print(f"[error] unified_model.pt not found: {ck_path}")
            return
        sub_ens = SubEnsemble(sub_dir, device=device)
        pack_bundle(ck_path, sub_ens, args.bundle_out)
        return

    # predict-only fast path
    if args.stage == "predict":
        if args.bundle:
            main, p, top = predict_with_bundle(args.bundle, emb_jsonl=args.emb_jsonl,
                                               vec=(np.asarray(args.text_emb, np.float32) if args.text_emb else None),
                                               id=args.id, topk=args.topk, device=device)
            print(f"[predict:bundle] main={main}({p:.3f}) | top{args.topk}_subs={top}")
            return
        if args.text_emb is None and (args.id is None or args.emb_jsonl is None):
            print("[predict] provide either --text-emb or (--id and --emb-jsonl)"); return
        sub_ens = SubEnsemble(sub_dir, device=device)

        # unified ck를 먼저 로드하여 서브 온도/스케일러 메타를 확보
        ck = _trusted_torch_load(Path(args.outdir) / "unified_model.pt", map_location="cpu")

        # --- INJECT SUB TEMPERATURE (if checkpoint provides it) ---
        try:
            subT = (ck.get("meta") or {}).get("sub_temp_fitted")
            if isinstance(subT, dict):
                if hasattr(sub_ens, "set_temperature_per_main"):
                    sub_ens.set_temperature_per_main({k: float(v) for k, v in subT.items()})
                else:
                    setattr(sub_ens, "temp_per_main", {k: float(v) for k, v in subT.items()})
        except Exception:
            pass

        if args.text_emb is not None:
            vec = np.asarray(args.text_emb, dtype=np.float32)
        else:
            # lookup by id
            X, ids = load_embeddings_jsonl(args.emb_jsonl)
            try:
                i = ids.index(args.id)
            except ValueError:
                print(f"[predict] id not found: {args.id}"); return
            vec = X[i]

        o = sub_ens.predict_proba(vec)
        feats = build_feature_for_id(vec, o, stats_only=False)

        # [PATCH 3.1] 피처 스케일러 적용 보강 - 형상확인 후 적용
        mu = (ck.get("meta") or {}).get("feat_mu")
        sigma = (ck.get("meta") or {}).get("feat_sigma")
        if isinstance(mu, list) and isinstance(sigma, list):
            mu = np.asarray(mu, dtype=np.float32)
            sigma = np.asarray(sigma, dtype=np.float32)
            if mu.shape == feats.shape == sigma.shape:
                feats = (feats - mu) / np.maximum(sigma, 1e-8)

        # unified model 준비
        meta = ck["meta"]; feat_dim = meta["feat_dim"]
        n_sub_total = meta.get("n_sub_total", sum(len(v[1]) for v in sub_ens.models.values()))
        mdl = MetaMLP(feat_dim, 4, n_sub_total).to(device)
        mdl.load_state_dict(ck["state_dict"]); mdl.eval()

        with torch.no_grad():
            y = torch.from_numpy(feats).unsqueeze(0).to(device)
            log_m, log_s = mdl(y)
            fitted_T = float((ck.get("meta") or {}).get("meta_temp_fitted", 1.0))
            use_T = float(args.meta_temp if (args.meta_temp != 1.0 or fitted_T == 1.0) else fitted_T)
            pm = torch.softmax(log_m / max(use_T, 1e-6), 1).cpu().numpy().squeeze(0)
            ps = torch.softmax(log_s, 1).cpu().numpy().squeeze(0)

        # predicted main + top-k subs of that main
        m_idx = int(pm.argmax())
        main = mains[m_idx]
        # build global sub index order (saved sub_lists 우선)
        # 개선사항 3: 저장된 sub_lists가 있으면 우선 사용(불일치 예방)
        subs_saved = (ck.get("meta") or {}).get("sub_lists") or {}
        offset = 0; global2name = []
        for m in mains:
            subs = subs_saved.get(m) or sub_ens.models[m][1]  # ★ 저장된 순서 우선
            for s in subs:
                global2name.append((m, s))
            offset += len(subs)
        # pick top-k subs under predicted main (슬라이스 폭도 saved 우선)
        sub_slices = []
        off = 0
        for m in mains:
            cnt = len(subs_saved.get(m) or sub_ens.models[m][1])
            sub_slices.append((off, off+cnt))
            off += cnt
        lo, hi = sub_slices[m_idx]
        part = ps[lo:hi]
        names_for_main = (subs_saved.get(main) or sub_ens.models[main][1])
        topk = min(args.topk, len(names_for_main))
        top_idx = part.argsort()[-topk:][::-1]
        top_subs = [(names_for_main[j], float(part[j])) for j in top_idx]

        print("[predict] main probs:", {mains[i]: float(pm[i]) for i in range(len(mains))})
        print(f"[predict] main = {main} | top{topk}_subs:", top_subs)
        return

    # report-only (라벨 커버리지/분포 점검)
    if args.stage == "report":
        if not args.labels_jsonl:
            print("[report] --labels-jsonl required"); return
        cov = _coverage_report(args.labels_jsonl, mains)
        ensure_dir(args.outdir)
        (Path(args.outdir) / "unified_label_coverage.json").write_text(
            json.dumps(cov, ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"[report] label coverage -> {Path(args.outdir)/'unified_label_coverage.json'}")
        return
    
    # 개선사항 4: 배치 예측(export) 모드 추가
    if args.stage == "export":
        if not args.emb_jsonl:
            print("[export] --emb-jsonl required")
            return
        
        # ck & sub ensemble
        ck = _trusted_torch_load(Path(args.outdir) / "unified_model.pt", map_location="cpu")
        sub_ens = SubEnsemble(sub_dir, device=device)
        
        # 서브 T 주입
        try:
            subT = (ck.get("meta") or {}).get("sub_temp_fitted")
            if isinstance(subT, dict):
                if hasattr(sub_ens, "set_temperature_per_main"):
                    sub_ens.set_temperature_per_main({k: float(v) for k, v in subT.items()})
                else:
                    for m in subT:
                        if m in sub_ens.models:
                            mdl, subs, _T = sub_ens.models[m]
                            sub_ens.models[m] = (mdl, subs, float(subT[m]))
        except Exception:
            pass
        
        # 데이터 로드/피처화
        X, ids = load_embeddings_jsonl(args.emb_jsonl)
        # 개선사항 3: 저장된 sub_lists 우선 사용
        subs_saved = (ck.get("meta") or {}).get("sub_lists") or {}
        sub_lists = {m: (subs_saved.get(m) or sub_ens.models[m][1]) for m in mains}  # ★ 저장된 순서 우선
        
        # 더미 y_map 생성 (모든 id에 대해 임시로 "희" 할당)
        y_map_dummy = {id_: ("희", None) for id_ in ids}
        ds = MetaDataset(X, ids, y_map_dummy, sub_ens, sub_lists, 
                        use_embedding=(not args.no_embed),
                        cache_dir=str(Path(args.outdir)/"cache_export"))
        
        # 피처 스케일링 적용
        mu = (ck.get("meta") or {}).get("feat_mu")
        sigma = (ck.get("meta") or {}).get("feat_sigma")
        if mu is not None and sigma is not None:
            mu = torch.tensor(mu, dtype=torch.float32)
            sigma = torch.tensor(sigma, dtype=torch.float32)
            if ds.X.shape[1] == len(mu) == len(sigma):
                ds.X = (ds.X - mu) / torch.clamp(sigma, min=1e-8)
        
        # unified model 준비
        mdl = MetaMLP(ck["meta"]["feat_dim"], 4, ck["meta"]["n_sub_total"]).to(device)
        mdl.load_state_dict(ck["state_dict"], strict=False)
        mdl.eval()
        Tm = float((ck.get("meta") or {}).get("meta_temp_fitted", 1.0))
        
        out = []
        with torch.no_grad():
            for i in range(0, ds.X.shape[0], 256):
                xb = ds.X[i:i+256].to(device)
                log_m, log_s = mdl(xb)
                pm = torch.softmax(log_m / max(Tm, 1e-6), 1).cpu().numpy()
                ps = torch.softmax(log_s, 1).cpu().numpy()
                
                for j, pid in enumerate(ds.ids_joined[i:i+256]):
                    m_idx = int(pm[j].argmax())
                    main_pred = mains[m_idx]
                    
                    # 서브 예측을 위한 global sub index
                    sub_probs = {}
                    offset = 0
                    for m in mains:
                        slist = sub_lists[m]
                        if m == main_pred:
                            for k, sub_name in enumerate(slist):
                                sub_probs[sub_name] = float(ps[j, offset + k])
                        offset += len(slist)
                    
                    # top-3 서브
                    top_subs = sorted(sub_probs.items(), key=lambda x: x[1], reverse=True)[:3]
                    
                    out.append({
                        "id": pid,
                        "main": main_pred,
                        "main_probs": {mains[k]: float(pm[j, k]) for k in range(4)},
                        "top_subs": [(s, p) for s, p in top_subs]
                    })
        
        # JSONL로 저장
        export_path = Path(args.outdir) / "unified_preds.jsonl"
        with open(export_path, "w", encoding="utf-8") as f:
            for rec in out:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        
        log(f"[export] {len(out)} predictions -> {export_path}")
        return

    # train/eval require emb & labels (이미 위에서 처리됨)
    if args.stage in ["train", "eval"]:
        if not args.emb_jsonl or not args.labels_jsonl:
            print("[error] --emb-jsonl and --labels-jsonl are required for train/eval")
            return

    # load embeddings & labels
    X, ids = load_embeddings_jsonl(args.emb_jsonl)
    y_map = load_labels_jsonl(args.labels_jsonl)  # id -> (main, sub)
    sub_ens = SubEnsemble(sub_dir, device=device)
    sub_lists = {m: sub_ens.models[m][1] for m in mains}

    if args.stage == "train":
        use_emb = not args.no_embed
        # ✅ no-early-stop이면 patience=0 으로 전달하여 조기종료 완전 비활성화
        patience_val = 0 if args.no_early_stop else max(args.patience, 0)

        if args.use_reasoner:
            use_emb = not args.no_embed
            model_path = train_unified_with_reasoner(
                X, ids,
                labels_jsonl=args.labels_jsonl,
                y_map=y_map,
                sub_ens=sub_ens,
                sub_list_by_main=sub_lists,
                outdir=args.outdir,
                cause_vocab_path=args.cause_vocab,
                device=device,
                epochs=args.epochs,
                batch=args.batch,
                seed=args.seed,
                freeze_backbone_epochs=args.freeze_backbone_epochs,
                lam_unc=args.lam_unc, lam_cause=args.lam_cause, lam_pivot=args.lam_pivot,
                use_embedding=use_emb
            )
            # Reasoner 학습 후 간단 평가
            ck = _trusted_torch_load(model_path, map_location="cpu")
            feat_dim = ck["meta"]["feat_dim"];
            n_sub_total = ck["meta"]["n_sub_total"]
            ds = MetaDataset(X, ids, y_map, sub_ens, sub_lists, use_embedding=use_emb, cache_dir=str(Path(args.outdir)/"cache"))
            mdl = MetaMLP(feat_dim, 4, n_sub_total).to(device)
            mdl.load_state_dict(ck["state_dict"], strict=False);
            mdl.eval()
            # [PATCH 3.2] 학습 시 적합한 meta_temp 사용
            fitted_T = float((ck.get("meta") or {}).get("meta_temp_fitted", 1.0))
            use_T = float(args.meta_temp if (hasattr(args,'meta_temp') and args.meta_temp != 1.0 and fitted_T == 1.0) else fitted_T)
            fm = (ck.get("meta") or {}).get("feat_mu"); fs = (ck.get("meta") or {}).get("feat_sigma")
            _evaluate_unified(mdl, ds, torch.device(device), Path(args.outdir),
                              meta_temp=use_T, feat_mu=fm, feat_sigma=fs)
            return

        model_path = train_meta(X, ids, y_map, sub_ens, sub_lists,
                                outdir=args.outdir, device=device,
                                epochs=args.epochs, batch=args.batch,
                                use_embedding=use_emb, seed=args.seed,
                                patience=patience_val, min_epochs=args.min_epochs,
                                labels_jsonl=args.labels_jsonl,
                                grad_accum=args.grad_accum)  # <<< 확장 전달

        # post-train eval on full set (간단 점검)
        ck = _trusted_torch_load(model_path, map_location="cpu")
        feat_dim = ck["meta"]["feat_dim"]; n_sub_total = ck["meta"]["n_sub_total"]
        ds = MetaDataset(X, ids, y_map, sub_ens, sub_lists, use_embedding=use_emb, cache_dir=str(Path(args.outdir)/"cache"))
        mdl = MetaMLP(feat_dim, 4, n_sub_total).to(device)
        mdl.load_state_dict(ck["state_dict"]); mdl.eval()
        fitted_T = float((ck.get("meta") or {}).get("meta_temp_fitted", 1.0))
        use_T = float(args.meta_temp if (args.meta_temp != 1.0 or fitted_T == 1.0) else fitted_T)
        fm = (ck.get("meta") or {}).get("feat_mu"); fs = (ck.get("meta") or {}).get("feat_sigma")
        _evaluate_unified(mdl, ds, torch.device(device), Path(args.outdir),
                          meta_temp=use_T, feat_mu=fm, feat_sigma=fs)
        return

    # eval
    use_emb = not args.no_embed
    ck = _trusted_torch_load(Path(args.outdir) / "unified_model.pt", map_location="cpu")
    feat_dim = ck["meta"]["feat_dim"]; n_sub_total = ck["meta"]["n_sub_total"]
    # 개선사항 3: 저장된 sub_lists 우선 사용
    subs_saved = (ck.get("meta") or {}).get("sub_lists") or {}
    sub_lists_for_eval = {m: (subs_saved.get(m) or sub_lists[m]) for m in mains}  # ★ 저장된 순서 우선
    ds = MetaDataset(X, ids, y_map, sub_ens, sub_lists_for_eval, use_embedding=use_emb)
    mdl = MetaMLP(feat_dim, 4, n_sub_total).to(device)
    mdl.load_state_dict(ck["state_dict"]); mdl.eval()
    fitted_T = float((ck.get("meta") or {}).get("meta_temp_fitted", 1.0))
    use_T = float(args.meta_temp if (args.meta_temp != 1.0 or fitted_T == 1.0) else fitted_T)
    fm = (ck.get("meta") or {}).get("feat_mu"); fs = (ck.get("meta") or {}).get("feat_sigma")
    _evaluate_unified(mdl, ds, torch.device(device), Path(args.outdir),
                      meta_temp=use_T, feat_mu=fm, feat_sigma=fs)

if __name__ == "__main__":
    main()


