"""Stage 2: Multilingual safety circuit discovery and surgical locality export.

Discovers contrastive safety refusal circuits using CircuitKIT EAP-IG, evaluates
them with a Calibrated Hybrid 4-pillar + standalone KL protocol, guards against
layer saturation in SafeTune apply_ctheta, and chains Stage 1 provenance.
"""
from __future__ import annotations

# CircuitKIT MUST be imported before TransformerLens so _tl_compat patches cohere2.
import circuitkit as ck  # noqa: E402,F401

import argparse
import copy
import gc
import hashlib
import json
import logging
import math
import os
import re
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from dotenv import load_dotenv
from huggingface_hub import HfApi, snapshot_download

from circuitkit.api import (
    _eap_kl_divergence,
    _reconstruct_circuit_graph,
    discover_circuit,
    evaluate_circuit,
)
from circuitkit.artifacts.scores import interop_fields
from circuitkit.data.eap_dataset import EAPDiscoveryDataset
from circuitkit.data.normalized_task import NormalizedTaskSpec
from circuitkit.data.template import template_normalize
from circuitkit.evaluation.evaluate import evaluate_baseline, evaluate_graph
from circuitkit.provenance import (
    FILENAME as PROVENANCE_FILENAME,
    model_input,
    read_provenance,
    write_provenance,
)
from circuitkit.tasks._chat import VALID_MODES, resolve_chat_template, wrap_prompt
from circuitkit.tasks import registry as _task_registry
from circuitkit.tasks.registry import is_task_registered, register_task

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "CohereLabs/tiny-aya-fire"
DEFAULT_DATASET = Path("artifacts/stage1_dataset/calibration_100_pairs.jsonl")
DEFAULT_OUTPUT_DIR = Path("artifacts/stage2_circuits")
DEFAULT_HF_REPO = "soapyyapps/circuit-kavach-cyber-codemixed"
DEFAULT_HF_SUBFOLDER = "stage2_circuits"
DEFAULT_SATURATION_TIERS: Tuple[float, ...] = (0.05, 0.10, 0.15, 0.20)
DEFAULT_EXPORT_TIERS: Tuple[float, ...] = (0.05, 0.10, 0.20)
DEFAULT_MAX_SURGICAL_SUBMODULES = 4
DEFAULT_SURGICAL_SUBMODULE_TIERS: Tuple[int, ...] = (3, 4, 6)
EVAL_PILLARS: Tuple[str, ...] = ("patching", "ablation", "baselines", "stability")

_HEAD_RE = re.compile(r"^A(\d+)\.\d+$")
_MLP_RE = re.compile(r"^MLP\s+(\d+)$")
_COHERE_ATTN = ("q_proj", "k_proj", "v_proj", "o_proj")
_COHERE_MLP = ("gate_proj", "up_proj", "down_proj")
_GPT2_ATTN = ("c_attn", "c_proj")
_GPT2_MLP = ("c_fc", "c_proj")


@dataclass(frozen=True)
class RuntimeProfile:
    model_name: str
    device: str
    dtype: str
    chat_template_mode: str
    templated: bool
    prepend_bos: bool
    max_seq_len: int
    batch_size: int
    low_vram_t4: bool = False


@dataclass(frozen=True)
class CalibrationResult:
    clean_token: str
    clean_token_id: int
    corrupt_token: str
    corrupt_token_id: int
    mean_clean_margin: float
    mean_corrupt_margin: float
    logit_diff_gap: float
    n_pairs_used: int
    templated: bool
    prepend_bos: bool
    pair_ids: List[str]


def _resolve_device(device: Optional[str]) -> str:
    if device is not None and device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_auto_dtype(model_name: str, dtype: str, resolved_device: str) -> str:
    if dtype != "auto":
        return dtype
    if resolved_device == "cpu" or "gpt2" in model_name.lower():
        return "float32"
    if resolved_device.startswith("cuda"):
        if torch.cuda.is_available() and not torch.cuda.is_bf16_supported(
            including_emulation=False
        ):
            return "float16"
        return "bfloat16"
    return "bfloat16"


def resolve_runtime_profile(
    model: Optional[Any] = None,
    *,
    model_name: str,
    chat_template_mode: str = "auto",
    batch_size: Optional[int] = None,
    max_seq_len: Optional[int] = None,
    dtype: str = "auto",
    device: Optional[str] = None,
    low_vram_t4: bool = False,
) -> RuntimeProfile:
    if low_vram_t4:
        chat_template_mode = "off"
        max_seq_len = 128
        batch_size = 1

    if chat_template_mode not in VALID_MODES:
        raise ValueError(
            f"chat_template_mode must be one of {VALID_MODES}, got {chat_template_mode!r}"
        )

    resolved_device = _resolve_device(device)
    templated = resolve_chat_template(chat_template_mode, model)
    if chat_template_mode == "on":
        tok = getattr(model, "tokenizer", None)
        if tok is None or getattr(tok, "chat_template", None) is None:
            raise ValueError(
                f"chat_template_mode='on' requested for {model_name!r}, "
                "but model.tokenizer.chat_template is None."
            )
    prepend_bos = not templated

    lower_name = model_name.lower()
    is_instruct_family = templated or ("aya" in lower_name) or ("cohere" in lower_name)
    resolved_max_seq_len = max_seq_len if max_seq_len is not None else (512 if is_instruct_family else 128)
    if resolved_max_seq_len <= 0:
        raise ValueError(f"max_seq_len must be positive, got {resolved_max_seq_len}")

    resolved_batch_size = batch_size if batch_size is not None else 1
    if resolved_batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {resolved_batch_size}")

    resolved_dtype = _resolve_auto_dtype(model_name, dtype, resolved_device)

    return RuntimeProfile(
        model_name=model_name,
        device=resolved_device,
        dtype=resolved_dtype,
        chat_template_mode=chat_template_mode,
        templated=templated,
        prepend_bos=prepend_bos,
        max_seq_len=resolved_max_seq_len,
        batch_size=resolved_batch_size,
        low_vram_t4=bool(low_vram_t4),
    )


def load_contrast_pairs(path: Path | str) -> List[Dict[str, Any]]:
    src = Path(path)
    if not src.exists():
        raise FileNotFoundError(f"Contrast pair dataset not found: {src}")

    required = ("pair_id", "harmful", "benign")
    pairs: List[Dict[str, Any]] = []
    with src.open("r", encoding="utf-8") as fh:
        for line_no, raw_line in enumerate(fh, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSON at {src}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object at {src}:{line_no}, got {type(row).__name__}")
            for key in required:
                if key not in row or not str(row[key]).strip():
                    raise ValueError(
                        f"Missing or empty required field {key!r} at {src}:{line_no}"
                    )
            pairs.append(row)

    if not pairs:
        raise ValueError(f"Dataset file is empty (no pairs loaded): {src}")
    return pairs


def select_preflight_pairs(pairs: Sequence[Dict[str, Any]], *, limit: int = 16) -> List[Dict[str, Any]]:
    if not pairs:
        raise ValueError("Cannot select preflight pairs from an empty pair sequence")
    if limit <= 0:
        raise ValueError(f"limit must be positive, got {limit}")
    if len(pairs) <= limit:
        return list(pairs)

    buckets: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for row in pairs:
        key = (str(row.get("language", "")), str(row.get("threat_category", "")))
        buckets.setdefault(key, []).append(row)

    selected: List[Dict[str, Any]] = []
    seen_ids = set()
    ordered_keys = sorted(buckets.keys())
    idx = 0
    while len(selected) < limit:
        progressed = False
        for key in ordered_keys:
            bucket = buckets[key]
            if idx < len(bucket):
                item = bucket[idx]
                pid = str(item.get("pair_id", id(item)))
                if pid not in seen_ids:
                    selected.append(item)
                    seen_ids.add(pid)
                    progressed = True
                    if len(selected) >= limit:
                        break
        if not progressed:
            break
        idx += 1
    return selected


def _truncate_prompt_text(model: Any, text: str, max_user_tokens: int) -> str:
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None or max_user_tokens <= 0:
        return text
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= max_user_tokens:
        return text
    return tokenizer.decode(ids[:max_user_tokens], skip_special_tokens=False)


def _tokenize_single_prompt(
    model: Any,
    wrapped_text: str,
    *,
    templated: bool,
    max_seq_len: int,
) -> torch.Tensor:
    tokens = model.to_tokens(wrapped_text, prepend_bos=not templated)
    if tokens.shape[1] > max_seq_len:
        if max_seq_len > 1:
            tokens = torch.cat([tokens[:, :1], tokens[:, -(max_seq_len - 1) :]], dim=1)
        else:
            tokens = tokens[:, -1:]
    device = getattr(model.cfg, "device", None)
    if device is not None:
        tokens = tokens.to(device)
    return tokens


def _template_overhead(model: Any, templated: bool) -> int:
    empty_wrapped = wrap_prompt(model, "", apply=templated)
    return int(model.to_tokens(empty_wrapped, prepend_bos=not templated).shape[1])


def _wrapped_token_ids(model: Any, text: str, *, templated: bool) -> List[int]:
    wrapped = wrap_prompt(model, text, apply=templated)
    return model.to_tokens(wrapped, prepend_bos=not templated)[0].tolist()


def _pad_piece_candidates(tokenizer: Any, *, templated: bool) -> List[str]:
    pad_tok = getattr(tokenizer, "pad_token", None)
    pad_id = getattr(tokenizer, "pad_token_id", None)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    raw: List[str] = []
    if pad_tok and (templated or (pad_id is not None and pad_id != eos_id)):
        # Include both space-separated and direct forms so Whitespace and BPE/SentencePiece both match +1/step
        raw.extend([str(pad_tok), f" {pad_tok}"])
    raw.extend([" .", " padding", "."])
    dedup: List[str] = []
    for item in raw:
        if item and item not in dedup:
            dedup.append(item)
    return dedup


def _common_prefix_len(a: Sequence[int], b: Sequence[int]) -> int:
    n = min(len(a), len(b))
    idx = 0
    while idx < n and a[idx] == b[idx]:
        idx += 1
    return idx


def equalize_pair_texts(
    model: Any,
    harmful_raw: str,
    benign_raw: str,
    *,
    templated: bool,
    max_seq_len: int,
    max_user_tokens: Optional[int] = None,
) -> Tuple[str, str, Dict[str, Any]]:
    """Equalize wrapped token lengths within a (harmful, benign) contrast pair.

    Right-pads the shorter user text before the closing assistant turn header so
    that system preamble, shared user prefix, and final generation token [-1]
    stay 100% index-aligned under ``pair_padding_side='left'``.
    """
    tokenizer = getattr(model, "tokenizer", None)
    if max_user_tokens is None:
        overhead = _template_overhead(model, templated)
        max_user_tokens = max(8, int(max_seq_len) - overhead - 2)

    h_text = _truncate_prompt_text(model, harmful_raw, max_user_tokens)
    b_text = _truncate_prompt_text(model, benign_raw, max_user_tokens)

    # Ensure neither wrapped sequence alone exceeds max_seq_len even after subword decode/re-encode
    budget = max_user_tokens
    while budget > 4:
        ids_h = _wrapped_token_ids(model, h_text, templated=templated)
        ids_b = _wrapped_token_ids(model, b_text, templated=templated)
        if max(len(ids_h), len(ids_b)) <= max_seq_len:
            break
        budget -= max(1, max(len(ids_h), len(ids_b)) - max_seq_len)
        h_text = _truncate_prompt_text(model, h_text, budget)
        b_text = _truncate_prompt_text(model, b_text, budget)

    ids_h = _wrapped_token_ids(model, h_text, templated=templated)
    ids_b = _wrapped_token_ids(model, b_text, templated=templated)
    raw_len_h, raw_len_b = len(ids_h), len(ids_b)
    if raw_len_h == raw_len_b:
        return h_text, b_text, {"equalized": False, "len_before_h": raw_len_h, "len_before_b": raw_len_b, "len_after": raw_len_h}

    orig_prefix_len = _common_prefix_len(ids_h, ids_b)
    shorter_is_h = raw_len_h < raw_len_b
    short_text = h_text if shorter_is_h else b_text
    long_text = b_text if shorter_is_h else h_text
    long_ids = ids_b if shorter_is_h else ids_h
    target_len = len(long_ids)

    candidates = _pad_piece_candidates(tokenizer, templated=templated)
    matched_texts: Optional[Tuple[str, str]] = None

    for piece in candidates:
        diff = target_len - (raw_len_h if shorter_is_h else raw_len_b)
        k = max(1, diff)
        cand_short = short_text + (piece * k)
        cand_ids = _wrapped_token_ids(model, cand_short, templated=templated)

        steps = 0
        while len(cand_ids) < target_len and steps < diff + 8:
            k += 1
            cand_short = short_text + (piece * k)
            cand_ids = _wrapped_token_ids(model, cand_short, templated=templated)
            steps += 1
        while len(cand_ids) > target_len and k > 1:
            k -= 1
            cand_short = short_text + (piece * k)
            cand_ids = _wrapped_token_ids(model, cand_short, templated=templated)

        if len(cand_ids) == target_len and _common_prefix_len(cand_ids, long_ids) >= orig_prefix_len:
            matched_texts = (cand_short, long_text) if shorter_is_h else (long_text, cand_short)
            break

        # If a word boundary added +1/+2 on first insertion, symmetrically match at target_len + delta
        if len(cand_ids) > target_len and len(cand_ids) <= max_seq_len:
            cand_long = long_text + piece
            cand_long_ids = _wrapped_token_ids(model, cand_long, templated=templated)
            steps_l = 0
            while len(cand_long_ids) < len(cand_ids) and steps_l < 6:
                steps_l += 1
                cand_long = long_text + (piece * (steps_l + 1))
                cand_long_ids = _wrapped_token_ids(model, cand_long, templated=templated)
            if (
                len(cand_long_ids) == len(cand_ids)
                and _common_prefix_len(cand_ids, cand_long_ids) >= orig_prefix_len
            ):
                matched_texts = (cand_short, cand_long) if shorter_is_h else (cand_long, cand_short)
                break

    if matched_texts is None and tokenizer is not None:
        # Final fallback: match by trimming trailing tokens from the longer user text without collapsing prompts
        l_toks = tokenizer.encode(long_text, add_special_tokens=False)
        cur_long = long_text
        while len(_wrapped_token_ids(model, cur_long, templated=templated)) > len(_wrapped_token_ids(model, short_text, templated=templated)) and len(l_toks) > 4:
            l_toks = l_toks[:-1]
            cur_long = tokenizer.decode(l_toks, skip_special_tokens=False)
        if len(_wrapped_token_ids(model, cur_long, templated=templated)) == len(_wrapped_token_ids(model, short_text, templated=templated)):
            matched_texts = (short_text, cur_long) if shorter_is_h else (cur_long, short_text)

    if matched_texts is None:
        matched_texts = (h_text, b_text)

    out_h, out_b = matched_texts
    out_len_h = len(_wrapped_token_ids(model, out_h, templated=templated))
    out_len_b = len(_wrapped_token_ids(model, out_b, templated=templated))
    return out_h, out_b, {
        "equalized": True,
        "len_before_h": raw_len_h,
        "len_before_b": raw_len_b,
        "len_after": max(out_len_h, out_len_b),
        "aligned": out_len_h == out_len_b,
    }


def equalize_contrast_pairs(
    model: Any,
    pairs: Sequence[Dict[str, Any]],
    *,
    templated: bool,
    max_seq_len: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Return length-equalized copies of contrast pairs and alignment statistics."""
    overhead = _template_overhead(model, templated)
    max_user_tokens = max(8, int(max_seq_len) - overhead - 2)
    out_pairs: List[Dict[str, Any]] = []
    n_modified = 0
    max_diff_before = 0
    misaligned = 0

    for row in pairs:
        eq_h, eq_b, info = equalize_pair_texts(
            model,
            str(row["harmful"]),
            str(row["benign"]),
            templated=templated,
            max_seq_len=max_seq_len,
            max_user_tokens=max_user_tokens,
        )
        max_diff_before = max(max_diff_before, abs(info["len_before_h"] - info["len_before_b"]))
        if info["equalized"]:
            n_modified += 1
            if not info.get("aligned", True):
                misaligned += 1
        updated = dict(row)
        updated["harmful"] = eq_h
        updated["benign"] = eq_b
        out_pairs.append(updated)

    stats = {
        "total_pairs": len(out_pairs),
        "pairs_equalized": n_modified,
        "pairs_already_aligned": len(out_pairs) - n_modified,
        "max_len_diff_before": max_diff_before,
        "remaining_misaligned_pairs": misaligned,
        "max_seq_len": int(max_seq_len),
    }
    return out_pairs, stats


def _valid_single_token_roundtrip(model: Any, tid: int, tok_str: str) -> bool:
    # Reject empty/whitespace-only, replacement char, pure punctuation, or newlines (break CSV rows).
    # Keep leading spaces valid so BPE/SentencePiece first tokens like " Sorry" round-trip cleanly.
    if not tok_str or not tok_str.strip() or "\ufffd" in tok_str or "\n" in tok_str or "\r" in tok_str:
        return False
    if all(not ch.isalnum() for ch in tok_str.strip()):
        return False
    tokenizer = model.tokenizer
    try:
        enc = tokenizer.encode(tok_str, add_special_tokens=False)
        if len(enc) != 1 or int(enc[0]) != tid:
            return False
    except Exception:
        return False
    return True


def calibrate_answer_tokens(
    model: Any,
    pairs: Sequence[Dict[str, Any]],
    *,
    templated: bool,
    max_seq_len: int = 512,
    preflight_limit: int = 16,
) -> CalibrationResult:
    preflight_raw = select_preflight_pairs(pairs, limit=preflight_limit)
    preflight, _ = equalize_contrast_pairs(
        model, preflight_raw, templated=templated, max_seq_len=max_seq_len
    )
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        raise ValueError("Model must expose a tokenizer for preflight token calibration")

    vocab_size = int(getattr(model.cfg, "d_vocab", None) or len(tokenizer))
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])

    sum_logp_clean = torch.zeros(vocab_size, dtype=torch.float64)
    sum_logp_corr = torch.zeros(vocab_size, dtype=torch.float64)
    sum_logits_clean = torch.zeros(vocab_size, dtype=torch.float64)
    sum_logits_corr = torch.zeros(vocab_size, dtype=torch.float64)

    n_used = 0
    pair_ids: List[str] = []

    with torch.inference_mode():
        for row in preflight:
            clean_prompt = wrap_prompt(model, str(row["harmful"]), apply=templated)
            corrupt_prompt = wrap_prompt(model, str(row["benign"]), apply=templated)

            clean_ids = _tokenize_single_prompt(
                model, clean_prompt, templated=templated, max_seq_len=max_seq_len
            )
            corr_ids = _tokenize_single_prompt(
                model, corrupt_prompt, templated=templated, max_seq_len=max_seq_len
            )

            c_logits = model(clean_ids)[0, -1, :vocab_size].detach().to("cpu", dtype=torch.float64)
            r_logits = model(corr_ids)[0, -1, :vocab_size].detach().to("cpu", dtype=torch.float64)

            sum_logits_clean += c_logits
            sum_logits_corr += r_logits
            sum_logp_clean += torch.log_softmax(c_logits, dim=-1)
            sum_logp_corr += torch.log_softmax(r_logits, dim=-1)
            n_used += 1
            pair_ids.append(str(row["pair_id"]))

    mean_logp_clean = sum_logp_clean / n_used
    mean_logp_corr = sum_logp_corr / n_used
    mean_logits_clean = sum_logits_clean / n_used
    mean_logits_corr = sum_logits_corr / n_used

    diff_clean = mean_logp_clean - mean_logp_corr
    diff_corr = mean_logp_corr - mean_logp_clean

    for sid in special_ids:
        if 0 <= int(sid) < vocab_size:
            diff_clean[int(sid)] = float("-inf")
            diff_corr[int(sid)] = float("-inf")

    clean_token_id: Optional[int] = None
    clean_token_str: str = ""
    for tid_tensor in torch.argsort(diff_clean, descending=True)[:512]:
        tid = int(tid_tensor.item())
        if diff_clean[tid] == float("-inf"):
            break
        tok_str = tokenizer.decode([tid])
        if _valid_single_token_roundtrip(model, tid, tok_str):
            clean_token_id = tid
            clean_token_str = tok_str
            break

    corrupt_token_id: Optional[int] = None
    corrupt_token_str: str = ""
    for tid_tensor in torch.argsort(diff_corr, descending=True)[:512]:
        tid = int(tid_tensor.item())
        if tid == clean_token_id or diff_corr[tid] == float("-inf"):
            continue
        tok_str = tokenizer.decode([tid])
        if _valid_single_token_roundtrip(model, tid, tok_str):
            corrupt_token_id = tid
            corrupt_token_str = tok_str
            break

    if clean_token_id is None or corrupt_token_id is None:
        raise RuntimeError("Preflight calibration could not find two valid single-token answer IDs")

    mean_clean_margin = float((mean_logits_clean[clean_token_id] - mean_logits_clean[corrupt_token_id]).item())
    mean_corrupt_margin = float((mean_logits_corr[clean_token_id] - mean_logits_corr[corrupt_token_id]).item())
    gap = mean_clean_margin - mean_corrupt_margin
    if gap <= 1e-6:
        raise RuntimeError(
            f"Calibrated logit_diff gap is non-positive/degenerate ({gap:.6e}); "
            "cannot satisfy positive faithfulness denominator."
        )

    return CalibrationResult(
        clean_token=clean_token_str,
        clean_token_id=clean_token_id,
        corrupt_token=corrupt_token_str,
        corrupt_token_id=corrupt_token_id,
        mean_clean_margin=mean_clean_margin,
        mean_corrupt_margin=mean_corrupt_margin,
        logit_diff_gap=gap,
        n_pairs_used=n_used,
        templated=templated,
        prepend_bos=not templated,
        pair_ids=pair_ids,
    )


def filter_pairs_by_baseline_margin(
    model: Any,
    pairs: Sequence[Dict[str, Any]],
    calibration: CalibrationResult,
    *,
    templated: bool,
    max_seq_len: int = 512,
    margin_threshold: float = 0.5,
    fallback_top_n: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Filter pairs to positive baseline refusal separation delta_m > margin_threshold.

    When fewer than min(2, len(pairs)) pairs pass (e.g. unaligned CPU gpt2 runs),
    deterministically falls back to top-N highest-margin pairs stratified across languages.
    """
    if not pairs:
        raise ValueError("Cannot filter baseline margins on an empty pair sequence")
    threshold = float(margin_threshold)
    if not math.isfinite(threshold) or threshold < 0.0:
        raise ValueError(f"margin_threshold must be finite and >= 0.0, got {margin_threshold}")

    eq_pairs, _ = equalize_contrast_pairs(model, pairs, templated=templated, max_seq_len=max_seq_len)
    c_id = int(calibration.clean_token_id)
    r_id = int(calibration.corrupt_token_id)

    scored: List[Dict[str, Any]] = []
    pos_counts_by_lang: Dict[str, int] = {}
    eval_counts_by_lang: Dict[str, int] = {}

    with torch.inference_mode():
        for row in eq_pairs:
            lang = str(row.get("language", "unknown"))
            eval_counts_by_lang[lang] = eval_counts_by_lang.get(lang, 0) + 1
            pos_counts_by_lang.setdefault(lang, 0)

            clean_prompt = wrap_prompt(model, str(row["harmful"]), apply=templated)
            corrupt_prompt = wrap_prompt(model, str(row["benign"]), apply=templated)
            clean_ids = _tokenize_single_prompt(
                model, clean_prompt, templated=templated, max_seq_len=max_seq_len
            )
            corr_ids = _tokenize_single_prompt(
                model, corrupt_prompt, templated=templated, max_seq_len=max_seq_len
            )
            c_logits = model(clean_ids)[0, -1].detach().to("cpu", dtype=torch.float64)
            r_logits = model(corr_ids)[0, -1].detach().to("cpu", dtype=torch.float64)

            m_clean = float((c_logits[c_id] - c_logits[r_id]).item())
            m_corrupt = float((r_logits[c_id] - r_logits[r_id]).item())
            delta_m = m_clean - m_corrupt
            if delta_m > threshold:
                pos_counts_by_lang[lang] += 1

            scored.append(
                {
                    "pair": row,
                    "pair_id": str(row["pair_id"]),
                    "language": lang,
                    "threat_category": str(row.get("threat_category", "")),
                    "m_clean": m_clean,
                    "m_corrupt": m_corrupt,
                    "delta_m": delta_m,
                }
            )

    passing = [entry for entry in scored if entry["delta_m"] > threshold]
    min_required = min(2, len(eq_pairs))
    fallback_used = False

    if len(passing) >= min_required:
        selected_entries = passing
    else:
        fallback_used = True
        pos_only = [entry for entry in scored if entry["delta_m"] > 1e-6]
        min_floor = min(len(eq_pairs), max(4, math.ceil(0.5 * len(eq_pairs))))

        if fallback_top_n is None and len(pos_only) >= min_floor:
            selected_entries = sorted(pos_only, key=lambda e: (-e["delta_m"], e["pair_id"]))
        else:
            target_n = (
                min(len(eq_pairs), max(1, int(fallback_top_n)))
                if fallback_top_n is not None
                else min_floor
            )
            by_lang: Dict[str, List[Dict[str, Any]]] = {}
            for entry in scored:
                by_lang.setdefault(entry["language"], []).append(entry)
            for bucket in by_lang.values():
                bucket.sort(key=lambda e: (-e["delta_m"], e["pair_id"]))
            # Start round-robin from the language with the highest top margin
            langs_sorted = sorted(
                by_lang.keys(),
                key=lambda lg: (-by_lang[lg][0]["delta_m"], lg),
            )
            selected_entries = []
            seen_pids = set()
            idx = 0
            while len(selected_entries) < target_n:
                progressed = False
                for lg in langs_sorted:
                    bucket = by_lang[lg]
                    if idx < len(bucket):
                        item = bucket[idx]
                        if item["pair_id"] not in seen_pids:
                            selected_entries.append(item)
                            seen_pids.add(item["pair_id"])
                            progressed = True
                            if len(selected_entries) >= target_n:
                                break
                if not progressed:
                    break
                idx += 1

    selected_ids = {e["pair_id"] for e in selected_entries}
    selected_pairs = [e["pair"] for e in selected_entries]
    failed_refusal_pairs = [
        {
            "pair_id": e["pair_id"],
            "language": e["language"],
            "threat_category": e["threat_category"],
            "m_clean": e["m_clean"],
            "m_corrupt": e["m_corrupt"],
            "delta_m": e["delta_m"],
            "held_out": e["pair_id"] not in selected_ids,
        }
        for e in scored
        if e["delta_m"] <= threshold
    ]
    held_out_pairs = [e["pair_id"] for e in scored if e["pair_id"] not in selected_ids]

    mean_selected_delta = sum(e["delta_m"] for e in selected_entries) / max(1, len(selected_entries))
    stats = {
        "margin_threshold": threshold,
        "total_evaluated": len(scored),
        "selected_count": len(selected_pairs),
        "positive_margin_count": sum(pos_counts_by_lang.values()),
        "positive_margin_count_by_language": {k: pos_counts_by_lang[k] for k in sorted(pos_counts_by_lang)},
        "evaluated_count_by_language": {k: eval_counts_by_lang[k] for k in sorted(eval_counts_by_lang)},
        "fallback_used": fallback_used,
        "selected_mean_delta_m": mean_selected_delta,
        "held_out_count": len(held_out_pairs),
        "held_out_pair_ids": held_out_pairs,
        "failed_refusal_count": len(failed_refusal_pairs),
        "failed_refusal_pairs": failed_refusal_pairs,
        "selected_pair_ids": [e["pair_id"] for e in selected_entries],
    }
    return selected_pairs, stats


class CalibratedSafetyTaskSpec(NormalizedTaskSpec):
    """NormalizedTaskSpec enforcing align_strategy='none', left-padding, and templated flag."""

    pair_padding_side = "left"

    def __init__(
        self,
        ds,
        *,
        name: str,
        cache_dir: str,
        chat_template_mode: str = "auto",
        templated: bool = False,
        max_seq_len: int = 512,
    ) -> None:
        super().__init__(ds, name=name, cache_dir=cache_dir)
        self.chat_template_mode = chat_template_mode
        self.default_templated = bool(templated)
        self.max_seq_len = int(max_seq_len)
        self.pair_padding_side = "left"

    def _build_eap_dataloader(self, model, batch_size: int, seed: int = 42, *, templated: Optional[bool] = None):
        cache_dir = Path(self.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        tok_name = getattr(getattr(model, "tokenizer", None), "name_or_path", None) or getattr(
            model.cfg, "model_name", "unknown"
        )
        resolved_templated = self.default_templated if templated is None else bool(templated)
        align_tag = self.ds.meta.get("_alignment", {}).get("align_strategy", "unk")
        records = getattr(self.ds, "records", ()) or ()
        row_digest_src = "\n".join(
            f"{r.clean_prompt}\t{r.clean_answer}\t{r.corrupt_prompt}\t{r.corrupt_answer}"
            for r in records
        )
        rows_hash = hashlib.sha256(row_digest_src.encode("utf-8")).hexdigest()[:12]
        key = hashlib.sha256(
            f"{self.name}|{tok_name}|{align_tag}|tmpl={resolved_templated}|{seed}|n={len(self.ds)}|{rows_hash}".encode()
        ).hexdigest()[:12]
        cache_path = cache_dir / f"{self.name}_{key}.csv"
        if not cache_path.exists():
            self._materialise_eap_csv(model, cache_path, seed=seed)

        ds = EAPDiscoveryDataset(str(cache_path), templated=resolved_templated)
        dl = ds.to_dataloader(
            batch_size=batch_size,
            pair_padding_side="left",
            templated=resolved_templated,
        )
        dl.templated = resolved_templated
        dl.pair_padding_side = "left"
        dl.max_length = self.max_seq_len
        return dl

    def build_dataloader(self, model, discovery_cfg: Dict[str, Any], device: str):
        if model is None:
            raise ValueError("CalibratedSafetyTaskSpec.build_dataloader requires a loaded model.")
        batch_size = int(discovery_cfg.get("batch_size", 1))
        seed = int(discovery_cfg.get("data_params", {}).get("seed", discovery_cfg.get("seed", 42)))
        mode = discovery_cfg.get("chat_template_mode", self.chat_template_mode)
        resolved_templated = resolve_chat_template(mode, model) if mode in VALID_MODES else self.default_templated
        return self._build_eap_dataloader(model, batch_size=batch_size, seed=seed, templated=resolved_templated)


def register_calibrated_task(
    *,
    pairs: Sequence[Dict[str, Any]],
    calibration: CalibrationResult,
    model: Any,
    templated: bool,
    max_seq_len: int,
    task_name: str,
    cache_dir: str,
    chat_template_mode: str = "auto",
) -> CalibratedSafetyTaskSpec:
    eq_pairs, _ = equalize_contrast_pairs(model, pairs, templated=templated, max_seq_len=max_seq_len)
    raw_rows: List[Dict[str, str]] = []
    for row in eq_pairs:
        raw_rows.append(
            {
                "clean_prompt": wrap_prompt(model, str(row["harmful"]), apply=templated),
                "clean_answer": calibration.clean_token,
                "corrupt_prompt": wrap_prompt(model, str(row["benign"]), apply=templated),
                "corrupt_answer": calibration.corrupt_token,
            }
        )

    ds = template_normalize(
        raw_rows,
        template_spec={
            "clean_prompt": "{clean_prompt}",
            "clean_answer": "{clean_answer}",
            "corrupt_prompt": "{corrupt_prompt}",
            "corrupt_answer": "{corrupt_answer}",
        },
        align_strategy="none",
        name=task_name,
        source=task_name,
    )
    spec = CalibratedSafetyTaskSpec(
        ds,
        name=task_name,
        cache_dir=cache_dir,
        chat_template_mode=chat_template_mode,
        templated=templated,
        max_seq_len=max_seq_len,
    )
    if is_task_registered(task_name):
        _task_registry._TASKS[task_name.lower()] = spec
    else:
        register_task(spec)
    return spec


def _parse_node_layer(node_id: str) -> Optional[int]:
    m = _HEAD_RE.match(node_id) or _MLP_RE.match(node_id)
    return int(m.group(1)) if m else None


def _arch_module_sets(model_name: str, target_modules: Sequence[str]) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    lower = model_name.lower()
    if "gpt2" in lower or any(m in {"c_attn", "c_fc"} for m in target_modules):
        return _GPT2_ATTN, _GPT2_MLP
    return _COHERE_ATTN, _COHERE_MLP


def _touched_component_counts(
    layer_subset: Sequence[int],
    target_modules: Sequence[str],
    *,
    model_name: str,
    n_layers: int,
) -> Tuple[int, int, float, float]:
    attn_mods, mlp_mods = _arch_module_sets(model_name, target_modules)
    tm_set = set(target_modules)
    has_attn = bool(tm_set & set(attn_mods))
    has_mlp = bool(tm_set & set(mlp_mods))
    total_matrices = max(1, 2 * int(n_layers))
    touched_matrices = len(layer_subset) * (int(has_attn) + int(has_mlp))
    touched_pct = 100.0 * touched_matrices / total_matrices
    frozen_pct = 100.0 - touched_pct
    return touched_matrices, total_matrices, touched_pct, frozen_pct


def apply_surgical_layer_cap(
    node_scores: Dict[str, float],
    *,
    model_name: str,
    n_layers: int,
    top_fraction: float,
    max_surgical_layers: Optional[int],
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    raw = interop_fields(node_scores, model=model_name, method="discover", top_fraction=top_fraction)
    raw_su = copy.deepcopy(raw["safety_units"])
    raw_ls = copy.deepcopy(raw["layer_suggestions"])

    raw_layers = list(raw_ls.get("layer_subset", []))
    raw_unit_ids = list(raw_su.get("unit_ids", []))
    raw_modules = list(raw_ls.get("target_modules", []))

    layer_selected_mass: Dict[int, float] = {}
    for uid in raw_unit_ids:
        layer = _parse_node_layer(uid)
        if layer is None:
            continue
        layer_selected_mass[layer] = layer_selected_mass.get(layer, 0.0) + abs(float(node_scores.get(uid, 0.0)))

    layer_total_mass: Dict[int, float] = {}
    for uid, val in node_scores.items():
        layer = _parse_node_layer(uid)
        if layer is None:
            continue
        layer_total_mass[layer] = layer_total_mass.get(layer, 0.0) + abs(float(val))

    ranked_layers = sorted(
        raw_layers,
        key=lambda lyr: (
            -layer_selected_mass.get(lyr, 0.0),
            -layer_total_mass.get(lyr, 0.0),
            lyr,
        ),
    )

    if max_surgical_layers is not None and int(max_surgical_layers) < 1:
        raise ValueError(f"max_surgical_layers must be >= 1 or None, got {max_surgical_layers}")

    if max_surgical_layers is not None and len(ranked_layers) > max_surgical_layers:
        kept_layers = sorted(ranked_layers[:max_surgical_layers])
    else:
        kept_layers = sorted(raw_layers)
    kept_set = set(kept_layers)

    capped_units = [u for u in raw_unit_ids if _parse_node_layer(u) in kept_set]
    has_head = any(_HEAD_RE.match(u) for u in capped_units)
    has_mlp = any(_MLP_RE.match(u) for u in capped_units)

    attn_mods, mlp_mods = _arch_module_sets(model_name, raw_modules)
    capped_target_modules: List[str] = []
    if has_head:
        for m in attn_mods:
            if m in raw_modules and m not in capped_target_modules:
                capped_target_modules.append(m)
    if has_mlp:
        for m in mlp_mods:
            if m in raw_modules and m not in capped_target_modules:
                capped_target_modules.append(m)
    if not capped_target_modules:
        capped_target_modules = list(raw_modules)

    raw_corr = dict(raw_su.get("activation_correlation", {}))
    capped_corr = {k: v for k, v in raw_corr.items() if _parse_node_layer(k) in kept_set}

    # Build SafeTune module_names matching kept layers + projections without leaving stale layers
    is_gpt2 = "gpt2" in model_name.lower() or "c_attn" in capped_target_modules
    prefix = "transformer.h" if is_gpt2 else "model.layers"
    capped_module_names: List[str] = []
    for lyr in kept_layers:
        for proj in capped_target_modules:
            if proj in attn_mods:
                sub = "attn" if is_gpt2 else "self_attn"
            else:
                sub = "mlp"
            capped_module_names.append(f"{prefix}.{lyr}.{sub}.{proj}")

    raw_priority = dict(raw_ls.get("priority", {}))
    capped_priority = {m: raw_priority.get(m, 1.0) for m in capped_target_modules}

    touched_raw, total_mat, raw_pct, raw_frozen = _touched_component_counts(
        raw_layers, raw_modules, model_name=model_name, n_layers=n_layers
    )
    touched_cap, _, cap_pct, cap_frozen = _touched_component_counts(
        kept_layers, capped_target_modules, model_name=model_name, n_layers=n_layers
    )

    cap_meta = {
        "top_fraction": top_fraction,
        "max_surgical_layers": max_surgical_layers,
        "uncapped_layer_subset": raw_layers,
        "capped_layer_subset": kept_layers,
        "cumulative_layer_scores": {str(lyr): layer_selected_mass.get(lyr, 0.0) for lyr in kept_layers},
        "total_layers": n_layers,
        "total_matrices": total_mat,
        "capped_matrices_touched": touched_cap,
        "frozen_matrix_pct": cap_frozen,
    }

    capped_su = {
        "layer_indices": kept_layers,
        "module_names": capped_module_names,
        "unit_ids": capped_units,
        "activation_correlation": capped_corr,
        "metadata": {**(raw_su.get("metadata") or {}), **cap_meta},
    }
    capped_ls = {
        "target_modules": capped_target_modules,
        "layer_subset": kept_layers,
        "priority": capped_priority,
        "metadata": {**(raw_ls.get("metadata") or {}), **cap_meta},
    }
    diag = {
        "top_fraction": float(top_fraction),
        "raw_node_count": len(raw_unit_ids),
        "raw_layers_touched": len(raw_layers),
        "total_layers": int(n_layers),
        "raw_matrices_touched": touched_raw,
        "total_matrices": total_mat,
        "raw_matrices_pct": raw_pct,
        "raw_frozen_pct": raw_frozen,
        "capped_node_count": len(capped_units),
        "capped_layers_touched": len(kept_layers),
        "capped_matrices_touched": touched_cap,
        "capped_matrices_pct": cap_pct,
        "capped_frozen_pct": cap_frozen,
    }
    return capped_su, capped_ls, diag


def compute_layer_saturation_table(
    *,
    node_scores: Dict[str, float],
    model_name: str,
    n_layers: int,
    tiers: Sequence[float] = DEFAULT_SATURATION_TIERS,
    max_surgical_layers: Optional[int] = 8,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for tier in tiers:
        _, _, diag = apply_surgical_layer_cap(
            node_scores,
            model_name=model_name,
            n_layers=n_layers,
            top_fraction=float(tier),
            max_surgical_layers=max_surgical_layers,
        )
        rows.append(diag)
    return rows


def format_layer_saturation_table(rows: Sequence[Dict[str, Any]]) -> str:
    header = (
        "Tier   | Nodes | Raw Layers | Raw Matrices (72=36a+36m) | "
        "Capped Layers | Capped Matrices | Frozen %"
    )
    sep = "-" * len(header)
    lines = [header, sep]
    for r in rows:
        lines.append(
            f"{r['top_fraction']:>5.2f}  | {r['raw_node_count']:>5d} | "
            f"{r['raw_layers_touched']:>2d}/{r['total_layers']:<2d}      | "
            f"{r['raw_matrices_touched']:>2d}/{r['total_matrices']:<2d} ({r['raw_matrices_pct']:>5.1f}%)           | "
            f"{r['capped_layers_touched']:>2d}/{r['total_layers']:<2d}         | "
            f"{r['capped_matrices_touched']:>2d}/{r['total_matrices']:<2d} ({r['capped_matrices_pct']:>5.1f}%)    | "
            f"{r['capped_frozen_pct']:>6.2f}%"
        )
    return "\n".join(lines)


def export_surgical_tiers(
    *,
    node_scores: Dict[str, float],
    model_name: str,
    n_layers: int,
    output_dir: Path | str,
    tiers: Sequence[float] = DEFAULT_EXPORT_TIERS,
    max_surgical_layers: Optional[int] = 8,
    provenance: Optional[Dict[str, Any]] = None,
) -> Dict[float, Path]:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    exported: Dict[float, Path] = {}
    for tier in tiers:
        su, ls, diag = apply_surgical_layer_cap(
            node_scores,
            model_name=model_name,
            n_layers=n_layers,
            top_fraction=float(tier),
            max_surgical_layers=max_surgical_layers,
        )
        pct_label = f"{int(round(float(tier) * 100)):02d}"
        file_path = out_path / f"safety_units_top{pct_label}.json"
        payload: Dict[str, Any] = {
            "version": "1.0",
            "model": model_name,
            "top_fraction": float(tier),
            "node_scores": dict(node_scores),
            "safety_units": su,
            "layer_suggestions": ls,
            "saturation_diagnostic": diag,
        }
        if provenance is not None:
            payload["provenance"] = provenance
        file_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        exported[float(tier)] = file_path
    return exported


def _submodule_target_names(
    layer: int,
    submodule: str,
    model_name: str,
    existing_targets: Sequence[str] = (),
) -> Tuple[List[str], List[str]]:
    """Return (layer_suggestions.target_modules patterns, safety_units.module_names)."""
    is_gpt2 = "gpt2" in model_name.lower() or any(m in {"c_attn", "c_fc"} for m in existing_targets)
    if is_gpt2:
        if submodule == "self_attn":
            return (
                [f"h.{layer}.attn.{p}" for p in _GPT2_ATTN],
                [f"transformer.h.{layer}.attn.{p}" for p in _GPT2_ATTN],
            )
        return (
            [f"h.{layer}.mlp.{p}" for p in _GPT2_MLP],
            [f"transformer.h.{layer}.mlp.{p}" for p in _GPT2_MLP],
        )

    if submodule == "self_attn":
        return (
            [f"layers.{layer}.self_attn.{p}" for p in _COHERE_ATTN],
            [f"model.layers.{layer}.self_attn.{p}" for p in _COHERE_ATTN],
        )
    return (
        [f"layers.{layer}.mlp.{p}" for p in _COHERE_MLP],
        [f"model.layers.{layer}.mlp.{p}" for p in _COHERE_MLP],
    )


def compute_submodule_surgical_mask(
    node_scores: Dict[str, float],
    *,
    model_name: str,
    n_layers: int,
    max_surgical_submodules: int = DEFAULT_MAX_SURGICAL_SUBMODULES,
    attn_aggregation: str = "sum",
    existing_targets: Sequence[str] = (),
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Rank all 2 * n_layers submodules (self_attn + mlp) and keep top-K surgical mask."""
    if int(max_surgical_submodules) < 1:
        raise ValueError(f"max_surgical_submodules must be >= 1, got {max_surgical_submodules}")
    if attn_aggregation not in {"sum", "mean"}:
        raise ValueError(f"attn_aggregation must be 'sum' or 'mean', got {attn_aggregation!r}")

    attn_heads_by_layer: Dict[int, List[str]] = {}
    mlp_key_by_layer: Dict[int, str] = {}
    max_layer_seen = max(-1, int(n_layers) - 1)
    for uid in node_scores:
        hm = _HEAD_RE.match(uid)
        if hm:
            lyr = int(hm.group(1))
            max_layer_seen = max(max_layer_seen, lyr)
            attn_heads_by_layer.setdefault(lyr, []).append(uid)
            continue
        mm = _MLP_RE.match(uid)
        if mm:
            lyr = int(mm.group(1))
            max_layer_seen = max(max_layer_seen, lyr)
            mlp_key_by_layer[lyr] = uid

    effective_layers = max(int(n_layers), max_layer_seen + 1)
    ranked_records: List[Dict[str, Any]] = []
    for lyr in range(effective_layers):
        heads = sorted(attn_heads_by_layer.get(lyr, []))
        head_abs = [abs(float(node_scores[h])) for h in heads]
        attn_sum = sum(head_abs)
        attn_mean = (attn_sum / len(head_abs)) if head_abs else 0.0
        attn_rank_score = attn_mean if attn_aggregation == "mean" else attn_sum
        ranked_records.append(
            {
                "layer": lyr,
                "submodule": "self_attn",
                "rank_score": attn_rank_score,
                "sum_abs_score": attn_sum,
                "mean_abs_score": attn_mean,
                "unit_ids": heads,
            }
        )

        mlp_uid = mlp_key_by_layer.get(lyr)
        mlp_abs = abs(float(node_scores[mlp_uid])) if mlp_uid is not None else 0.0
        ranked_records.append(
            {
                "layer": lyr,
                "submodule": "mlp",
                "rank_score": mlp_abs,
                "sum_abs_score": mlp_abs,
                "mean_abs_score": mlp_abs,
                "unit_ids": [mlp_uid] if mlp_uid is not None else [],
            }
        )

    ranked_records.sort(
        key=lambda r: (
            -r["rank_score"],
            -r["sum_abs_score"],
            r["layer"],
            0 if r["submodule"] == "self_attn" else 1,
        )
    )

    total_submodules = len(ranked_records)
    kept_k = min(int(max_surgical_submodules), total_submodules)
    selected = ranked_records[:kept_k]

    selected_unit_ids: List[str] = []
    selected_layers = sorted({int(r["layer"]) for r in selected})
    target_modules: List[str] = []
    module_names: List[str] = []
    priority_map: Dict[str, float] = {}
    max_sub_score = max((r["rank_score"] for r in selected), default=1.0) or 1.0

    # Emit targets ordered deterministically by (layer, submodule) so _matched_param_keys matches ONLY selected submodules
    for rec in sorted(selected, key=lambda r: (r["layer"], 0 if r["submodule"] == "self_attn" else 1)):
        selected_unit_ids.extend(rec["unit_ids"])
        tms, mns = _submodule_target_names(
            rec["layer"],
            rec["submodule"],
            model_name,
            existing_targets=existing_targets,
        )
        sub_norm = round(rec["rank_score"] / max_sub_score, 6)
        for tm in tms:
            if tm not in target_modules:
                target_modules.append(tm)
            priority_map[tm] = sub_norm
        for mn in mns:
            if mn not in module_names:
                module_names.append(mn)

    selected_unit_ids.sort(key=lambda u: (-abs(float(node_scores.get(u, 0.0))), u))
    max_unit_abs = max((abs(float(node_scores.get(u, 0.0))) for u in selected_unit_ids), default=1.0) or 1.0
    act_corr = {
        u: round(abs(float(node_scores.get(u, 0.0))) / max_unit_abs, 6)
        for u in selected_unit_ids
    }

    touched_pct = 100.0 * kept_k / max(1, total_submodules)
    frozen_pct = 100.0 - touched_pct

    mask_meta = {
        "selection_method": "submodule_ranked_surgical_mask",
        "attn_aggregation": attn_aggregation,
        "max_surgical_submodules": int(max_surgical_submodules),
        "selected_submodules": [
            {
                "layer": r["layer"],
                "submodule": r["submodule"],
                "rank_score": r["rank_score"],
                "sum_abs_score": r["sum_abs_score"],
                "mean_abs_score": r["mean_abs_score"],
            }
            for r in selected
        ],
        "layers_touched": len(selected_layers),
        "total_layers": effective_layers,
        "projection_blocks_touched": kept_k,
        "total_projection_blocks": total_submodules,
        "projection_blocks_pct": touched_pct,
        "frozen_pct": frozen_pct,
    }

    su = {
        "layer_indices": selected_layers,
        "module_names": module_names,
        "unit_ids": selected_unit_ids,
        "activation_correlation": act_corr,
        "metadata": mask_meta,
    }
    ls = {
        "target_modules": target_modules,
        "layer_subset": selected_layers,
        "priority": priority_map,
        "metadata": mask_meta,
    }
    diag = {
        "label": f"submodule-top-{int(max_surgical_submodules)}",
        "node_count": len(selected_unit_ids),
        "layers_touched": len(selected_layers),
        "total_layers": effective_layers,
        "projection_blocks_touched": kept_k,
        "total_projection_blocks": total_submodules,
        "projection_blocks_pct": touched_pct,
        "frozen_pct": frozen_pct,
    }
    return su, ls, diag


def compute_submodule_diagnostic_table(
    *,
    node_scores: Dict[str, float],
    model_name: str,
    n_layers: int,
    tiers: Sequence[float] = DEFAULT_EXPORT_TIERS,
    max_surgical_submodules: int = DEFAULT_MAX_SURGICAL_SUBMODULES,
    submodule_budgets: Sequence[int] = DEFAULT_SURGICAL_SUBMODULE_TIERS,
    attn_aggregation: str = "sum",
    existing_targets: Sequence[str] = (),
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    inferred_targets: List[str] = list(existing_targets)
    for tier in tiers:
        raw = interop_fields(node_scores, model=model_name, method="discover", top_fraction=float(tier))
        raw_ls = raw["layer_suggestions"]
        raw_su = raw["safety_units"]
        layers = list(raw_ls.get("layer_subset", []))
        modules = list(raw_ls.get("target_modules", []))
        for m in modules:
            if m not in inferred_targets:
                inferred_targets.append(m)
        touched, total_blocks, touched_pct, frozen_pct = _touched_component_counts(
            layers, modules, model_name=model_name, n_layers=n_layers
        )
        rows.append(
            {
                "label": f"{float(tier):.2f}",
                "node_count": len(raw_su.get("unit_ids", [])),
                "layers_touched": len(layers),
                "total_layers": int(n_layers),
                "projection_blocks_touched": touched,
                "total_projection_blocks": total_blocks,
                "projection_blocks_pct": touched_pct,
                "frozen_pct": frozen_pct,
            }
        )
    ordered_budgets: List[int] = [int(b) for b in submodule_budgets if int(b) >= 1]
    if int(max_surgical_submodules) not in ordered_budgets:
        ordered_budgets.append(int(max_surgical_submodules))
    for k_budget in ordered_budgets:
        _, _, sub_diag = compute_submodule_surgical_mask(
            node_scores=node_scores,
            model_name=model_name,
            n_layers=n_layers,
            max_surgical_submodules=k_budget,
            attn_aggregation=attn_aggregation,
            existing_targets=inferred_targets,
        )
        rows.append(sub_diag)
    return rows


def export_submodule_surgical_tiers(
    *,
    node_scores: Dict[str, float],
    model_name: str,
    n_layers: int,
    output_dir: Path | str,
    stem: str,
    budgets: Sequence[int] = DEFAULT_SURGICAL_SUBMODULE_TIERS,
    existing_targets: Sequence[str] = (),
    provenance: Optional[Dict[str, Any]] = None,
    common_fields: Optional[Dict[str, Any]] = None,
) -> Dict[int, Path]:
    """Export top-K submodule masks (3, 4, 6) in SafeTune core/circuit_kit/adapter.py schema."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    exported: Dict[int, Path] = {}
    for k_budget in budgets:
        su, ls, diag = compute_submodule_surgical_mask(
            node_scores=node_scores,
            model_name=model_name,
            n_layers=n_layers,
            max_surgical_submodules=int(k_budget),
            existing_targets=existing_targets,
        )
        file_path = out_path / f"{stem}_surgical_submodules_top{int(k_budget)}_scores.json"
        payload: Dict[str, Any] = {
            "version": "1.0",
            "model": model_name,
            "algorithm": "eap-ig",
            "level": "node",
            "selection_method": "submodule_ranked_surgical_mask",
            "max_surgical_submodules": int(k_budget),
            "node_scores": dict(node_scores),
            "safety_units": su,
            "layer_suggestions": ls,
            "surgical_submodules_summary": diag,
        }
        if provenance is not None:
            payload["provenance"] = provenance
        if common_fields:
            payload.update(common_fields)
        file_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        exported[int(k_budget)] = file_path
    return exported


def format_submodule_diagnostic_table(rows: Sequence[Dict[str, Any]]) -> str:
    header = "Mask Regime      | Nodes | Unique Layers | Projection Blocks | Frozen %"
    sep = "-" * len(header)
    lines = [header, sep]
    for r in rows:
        lines.append(
            f"{r['label']:<16} | {r['node_count']:>5d} | "
            f"{r['layers_touched']:>2d}/{r['total_layers']:<2d}         | "
            f"{r['projection_blocks_touched']:>2d}/{r['total_projection_blocks']:<2d} ({r['projection_blocks_pct']:>5.1f}%)     | "
            f"{r['frozen_pct']:>6.2f}%"
        )
    return "\n".join(lines)


def compute_kl_diagnostic(
    *,
    model: Any,
    scores_pt_path: Path | str,
    task_spec: CalibratedSafetyTaskSpec,
    discovery_cfg: Dict[str, Any],
    pruning_cfg: Dict[str, Any],
    device: str,
) -> Dict[str, Any]:
    scores_blob = torch.load(str(scores_pt_path), map_location="cpu", weights_only=True)
    scores_data = (
        scores_blob
        if isinstance(scores_blob, dict) and "node_scores" in scores_blob
        else {"node_scores": scores_blob}
    )

    graph = _reconstruct_circuit_graph(model, scores_data, discovery_cfg, pruning_cfg, device)

    dataloader = task_spec.build_dataloader(model, discovery_cfg, device)

    def _kl_per_sample(logits, clean_logits, input_lengths, labels):
        return _eap_kl_divergence(logits, clean_logits, input_lengths, labels, mean=False)

    circuit_kl_tensor = evaluate_graph(
        model,
        graph,
        dataloader,
        metrics=_kl_per_sample,
        intervention="patching",
        skip_clean=False,
        quiet=True,
    )
    corrupt_kl_tensor = evaluate_baseline(
        model,
        dataloader,
        metrics=_kl_per_sample,
        run_corrupted=True,
        quiet=True,
    )
    if isinstance(circuit_kl_tensor, list):
        circuit_kl_tensor = circuit_kl_tensor[0]
    if isinstance(corrupt_kl_tensor, list):
        corrupt_kl_tensor = corrupt_kl_tensor[0]

    kl_circuit = float(circuit_kl_tensor.float().mean().item())
    kl_corrupt = float(corrupt_kl_tensor.float().mean().item())
    kl_reduction_ratio = (1.0 - (kl_circuit / kl_corrupt)) if kl_corrupt > 1e-9 else None

    return {
        "kl_circuit_vs_clean": kl_circuit,
        "kl_corrupt_vs_clean": kl_corrupt,
        "kl_reduction_ratio": kl_reduction_ratio,
        "n_samples": int(circuit_kl_tensor.numel()),
    }


def _release_vram() -> None:
    """Release Python cyclic references and CUDA allocator cache between heavy passes."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def resolve_model_revision(model: Any, model_name: str) -> Optional[str]:
    """Best-effort commit revision SHA for reproducibility (offline-safe for local dirs)."""
    for owner in (model, getattr(model, "cfg", None), getattr(model, "config", None)):
        commit = getattr(owner, "_commit_hash", None)
        if isinstance(commit, str) and commit.strip():
            return commit.strip()
    if Path(str(model_name)).exists():
        return None
    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(model_name, local_files_only=True)
        commit = getattr(cfg, "_commit_hash", None)
        if isinstance(commit, str) and commit.strip():
            return commit.strip()
    except Exception:
        pass
    try:
        token = os.getenv("HF_TOKEN") or None
        info = HfApi(token=token).model_info(model_name)
        sha = getattr(info, "sha", None)
        if isinstance(sha, str) and sha.strip():
            return sha.strip()
    except Exception:
        pass
    return None


def verify_hf_auth(token: Optional[str] = None) -> str:
    """Verify Hugging Face Hub authentication before allocating GPU/model resources."""
    load_dotenv()
    resolved = token if token is not None else os.getenv("HF_TOKEN")
    if not resolved:
        raise RuntimeError(
            "HF_TOKEN environment variable is not set; cannot authenticate for --upload-hf."
        )
    api = HfApi(token=resolved)
    try:
        info = api.whoami()
    except Exception as exc:
        raise RuntimeError(f"Hugging Face authentication check failed before Stage 2 compute: {exc}") from exc
    if isinstance(info, dict):
        return str(info.get("name") or info.get("fullname") or "authenticated")
    return "authenticated"


def upload_stage2_to_hf(
    output_dir: Path | str,
    *,
    repo_id: str = DEFAULT_HF_REPO,
    path_in_repo: str = DEFAULT_HF_SUBFOLDER,
    repo_type: str = "dataset",
    commit_message: str = "Upload Stage 2 safety circuit artifacts and provenance",
) -> str:
    load_dotenv()
    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN environment variable is not set; cannot upload Stage 2 artifacts to Hugging Face."
        )
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type=repo_type, exist_ok=True)
    res = api.upload_folder(
        folder_path=str(output_dir),
        repo_id=repo_id,
        path_in_repo=path_in_repo,
        repo_type=repo_type,
        ignore_patterns=["_cache/*", "_cache/**"],
        commit_message=commit_message,
    )
    commit_url = getattr(res, "commit_url", None) or str(res)
    return str(commit_url)


def _serialize_faithfulness_report(
    report: Any,
    *,
    out_dir: Path,
    eval_pillars: Sequence[str],
) -> Dict[str, Any]:
    if hasattr(report, "to_json"):
        raw_path = out_dir / "_circuitkit_faithfulness_raw.json"
        try:
            report.to_json(raw_path)
            data = json.loads(raw_path.read_text(encoding="utf-8"))
        finally:
            raw_path.unlink(missing_ok=True)
    elif is_dataclass(report):
        data = asdict(report)
    elif hasattr(report, "to_dict"):
        data = report.to_dict()
    else:
        data = dict(report)

    meta = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
    meta_pillars = meta.get("pillars")
    if isinstance(meta_pillars, dict):
        data["pillars"] = dict(meta_pillars)
        return data

    requested = set(meta_pillars) if isinstance(meta_pillars, list) else set(eval_pillars)
    pillar_map = {
        "patching": data.get("patching_score"),
        "ablation": data.get("ablation_score"),
        "baselines": data.get("baseline_comparison"),
        "stability": data.get("stability"),
        "robustness": data.get("robustness"),
        "generalization": data.get("generalization"),
        "intervention_reliability": data.get("intervention_reliability"),
    }
    data["pillars"] = {
        pname: pval
        for pname, pval in pillar_map.items()
        if pname in requested or pval is not None
    }
    return data


def run_stage2(
    *,
    model_name: str = DEFAULT_MODEL,
    dataset_path: Path | str = DEFAULT_DATASET,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    chat_template_mode: str = "auto",
    batch_size: Optional[int] = None,
    max_seq_len: Optional[int] = None,
    dtype: str = "auto",
    device: Optional[str] = None,
    top_fraction: float = 0.10,
    max_surgical_layers: Optional[int] = 8,
    max_surgical_submodules: int = DEFAULT_MAX_SURGICAL_SUBMODULES,
    margin_threshold: float = 0.5,
    n_examples: Optional[int] = None,
    n_stability_runs: int = 3,
    ig_steps: int = 3,
    low_vram_t4: bool = False,
    hydrate_hf: bool = False,
    upload_hf: bool = False,
    hf_repo: str = DEFAULT_HF_REPO,
    stem: str = "safety_circuit",
) -> Dict[str, Any]:
    if not (0.0 < float(top_fraction) <= 1.0):
        raise ValueError(f"top_fraction must be in (0.0, 1.0], got {top_fraction}")
    if max_surgical_layers is not None and int(max_surgical_layers) < 0:
        raise ValueError(f"max_surgical_layers must be >= 0 or None, got {max_surgical_layers}")
    if int(max_surgical_submodules) < 1:
        raise ValueError(f"max_surgical_submodules must be >= 1, got {max_surgical_submodules}")
    if not math.isfinite(float(margin_threshold)) or float(margin_threshold) < 0.0:
        raise ValueError(f"margin_threshold must be finite and >= 0.0, got {margin_threshold}")
    if n_examples is not None and int(n_examples) < 1:
        raise ValueError(f"n_examples must be >= 1 or None, got {n_examples}")
    if int(n_stability_runs) < 1:
        raise ValueError(f"n_stability_runs must be >= 1, got {n_stability_runs}")
    if int(ig_steps) < 1:
        raise ValueError(f"ig_steps must be >= 1, got {ig_steps}")

    load_dotenv()
    if upload_hf:
        hf_user = verify_hf_auth()
        logger.info("Verified Hugging Face Hub auth before Stage 2 compute (user=%s).", hf_user)

    if low_vram_t4:
        chat_template_mode = "off"
        max_seq_len = 128
        batch_size = 1

    ds_path_obj = Path(dataset_path)
    if hydrate_hf and not ds_path_obj.exists():
        hydrate_root = (
            ds_path_obj.parent.parent
            if ds_path_obj.parent.name == "stage1_dataset"
            else ds_path_obj.parent
        )
        hydrate_root.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=hf_repo,
            repo_type="dataset",
            local_dir=str(hydrate_root),
            allow_patterns=["stage1_dataset/**"],
            token=os.getenv("HF_TOKEN") or None,
        )

    pairs = load_contrast_pairs(dataset_path)
    n_examples_eff = min(len(pairs), n_examples) if n_examples is not None else len(pairs)

    resolved_device = _resolve_device(device)
    init_dtype = _resolve_auto_dtype(model_name, dtype, resolved_device)
    model = ck.load_model(model_name, dtype=init_dtype, device=resolved_device)
    model_revision = resolve_model_revision(model, model_name)

    profile = resolve_runtime_profile(
        model,
        model_name=model_name,
        chat_template_mode=chat_template_mode,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        dtype=init_dtype,
        device=resolved_device,
        low_vram_t4=low_vram_t4,
    )

    equalized_pairs, equalization_stats = equalize_contrast_pairs(
        model,
        pairs[:n_examples_eff],
        templated=profile.templated,
        max_seq_len=profile.max_seq_len,
    )
    calibration = calibrate_answer_tokens(
        model,
        equalized_pairs,
        templated=profile.templated,
        max_seq_len=profile.max_seq_len,
    )
    filtered_pairs, margin_stats = filter_pairs_by_baseline_margin(
        model,
        equalized_pairs,
        calibration,
        templated=profile.templated,
        max_seq_len=profile.max_seq_len,
        margin_threshold=margin_threshold,
    )
    n_active = len(filtered_pairs)
    effective_batch_size = max(1, min(profile.batch_size, n_active))

    logger.info(
        "Preflight Baseline Margin Filter (threshold=%.3f): selected=%d/%d fallback=%s pos_by_lang=%s failed_refusal=%d",
        margin_stats["margin_threshold"],
        margin_stats["selected_count"],
        margin_stats["total_evaluated"],
        margin_stats["fallback_used"],
        margin_stats["positive_margin_count_by_language"],
        margin_stats["failed_refusal_count"],
    )
    print(
        f"\nPreflight Baseline Margin Filter (threshold={margin_stats['margin_threshold']:.3f}): "
        f"selected={margin_stats['selected_count']}/{margin_stats['total_evaluated']} "
        f"(fallback={margin_stats['fallback_used']}), "
        f"positive_margin_count_by_language={margin_stats['positive_margin_count_by_language']}, "
        f"failed_refusal_count={margin_stats['failed_refusal_count']}"
    )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_pt = out_dir / f"{stem}.pt"
    scores_pt = out_dir / f"{stem}_scores.pt"
    scores_json = out_dir / f"{stem}_scores.json"
    surgical_submodules_json = out_dir / f"{stem}_surgical_submodules_scores.json"

    safe_tag = re.sub(r"[^A-Za-z0-9._-]+", "_", f"stage2_{Path(dataset_path).stem}_{stem}").strip("_")
    task_spec = register_calibrated_task(
        pairs=filtered_pairs,
        calibration=calibration,
        model=model,
        templated=profile.templated,
        max_seq_len=profile.max_seq_len,
        task_name=safe_tag,
        cache_dir=str(out_dir / "_cache"),
        chat_template_mode=profile.chat_template_mode,
    )

    target_sparsity = round(max(0.0, min(0.99, 1.0 - float(top_fraction))), 6)
    discovery_cfg: Dict[str, Any] = {
        "algorithm": "eap-ig",
        "task": task_spec.name,
        "level": "node",
        "ig_steps": int(ig_steps),
        "batch_size": effective_batch_size,
        "max_seq_len": profile.max_seq_len,
        "chat_template_mode": profile.chat_template_mode,
        "model_name": model_name,
        "data_params": {"num_examples": n_active, "batch_size": effective_batch_size, "seed": 42},
    }
    pruning_cfg: Dict[str, Any] = {"target_sparsity": target_sparsity, "scope": "both"}
    config: Dict[str, Any] = {
        "model": {"name": model_name, "precision": profile.dtype, "device": profile.device},
        "discovery": discovery_cfg,
        "pruning": pruning_cfg,
        "output_path": str(artifact_pt),
    }

    _release_vram()
    discover_circuit(config, _model=model)
    if scores_json.exists():
        raw_blob = json.loads(scores_json.read_text(encoding="utf-8"))
        raw_ns = raw_blob.get("node_scores", {})
    else:
        raw_blob = torch.load(str(scores_pt), map_location="cpu", weights_only=True)
        raw_ns = raw_blob["node_scores"] if isinstance(raw_blob, dict) and "node_scores" in raw_blob else raw_blob
    node_scores = {str(k): float(v) for k, v in raw_ns.items()}
    non_finite = [k for k, v in node_scores.items() if not math.isfinite(v)]
    if non_finite:
        raise ValueError(f"Non-finite node attribution scores detected: {non_finite[:5]}")
    n_layers = int(getattr(model.cfg, "n_layers", 36))

    # Phase A: compute surgical masks/tables and persist + upload before heavy evaluation passes
    std_interop = interop_fields(node_scores, model=model_name, method="discover", top_fraction=float(top_fraction))
    existing_targets = tuple(std_interop.get("layer_suggestions", {}).get("target_modules", ()) or ())
    saturation_table = compute_layer_saturation_table(
        node_scores=node_scores,
        model_name=model_name,
        n_layers=n_layers,
        tiers=DEFAULT_SATURATION_TIERS,
        max_surgical_layers=max_surgical_layers,
    )
    submodule_diag_table = compute_submodule_diagnostic_table(
        node_scores=node_scores,
        model_name=model_name,
        n_layers=n_layers,
        tiers=DEFAULT_EXPORT_TIERS,
        max_surgical_submodules=max_surgical_submodules,
        existing_targets=existing_targets,
    )
    logger.info("\nLayer Saturation Diagnostic Table:\n%s", format_layer_saturation_table(saturation_table))
    logger.info("\nSubmodule Mask Diagnostic Table:\n%s", format_submodule_diagnostic_table(submodule_diag_table))
    print("\nLayer Saturation Diagnostic Table:\n" + format_layer_saturation_table(saturation_table))
    print("\nSubmodule Mask Diagnostic Table:\n" + format_submodule_diagnostic_table(submodule_diag_table))

    _, _, surgical_diag = apply_surgical_layer_cap(
        node_scores,
        model_name=model_name,
        n_layers=n_layers,
        top_fraction=float(top_fraction),
        max_surgical_layers=max_surgical_layers,
    )
    sub_su, sub_ls, sub_diag = compute_submodule_surgical_mask(
        node_scores,
        model_name=model_name,
        n_layers=n_layers,
        max_surgical_submodules=max_surgical_submodules,
        existing_targets=existing_targets,
    )

    ds_path_obj = Path(dataset_path)
    ds_prov_dir = ds_path_obj if ds_path_obj.is_dir() else ds_path_obj.parent
    ds_provenance = read_provenance(ds_prov_dir)
    prov_inputs = [
        model_input(model_name),
        {
            "kind": "dataset",
            "ref": str(dataset_path),
            "config": ds_path_obj.stem,
            "provenance": ds_provenance,
        },
    ]
    prov_params: Dict[str, Any] = {
        "algorithm": "eap-ig",
        "model_revision": model_revision,
        "ig_steps": int(ig_steps),
        "metric": "logit_diff",
        "align_strategy": "none",
        "pair_padding_side": "left",
        "runtime_profile": asdict(profile),
        "equalization_stats": equalization_stats,
        "calibration": asdict(calibration),
        "margin_filter": {k: v for k, v in margin_stats.items() if k != "selected_pair_ids"},
        "top_fraction": float(top_fraction),
        "target_sparsity": target_sparsity,
        "max_surgical_layers": max_surgical_layers,
        "max_surgical_submodules": int(max_surgical_submodules),
        "margin_threshold": float(margin_threshold),
        "eval_pillars": list(EVAL_PILLARS),
        "kl_diagnostic": None,
    }

    existing_scores_blob: Dict[str, Any] = {}
    if scores_json.exists():
        existing_scores_blob = json.loads(scores_json.read_text(encoding="utf-8"))
    ts_now = existing_scores_blob.get("timestamp", datetime.now(timezone.utc).isoformat())

    def _persist_phase_outputs(
        kl_value: Optional[Dict[str, Any]],
    ) -> tuple[Dict[str, Any], Dict[Any, Path], Dict[int, Path]]:
        prov_params["kl_diagnostic"] = kl_value
        prov_rec = write_provenance(
            out_dir,
            method="discover.eap-ig",
            inputs=prov_inputs,
            base_model=model_name,
            params=prov_params,
        )
        assert (out_dir / PROVENANCE_FILENAME).exists()
        existing_scores_blob.update(
            {
                "version": existing_scores_blob.get("version", "1.0"),
                "timestamp": ts_now,
                "task": task_spec.name,
                "model": model_name,
                "model_revision": model_revision,
                "algorithm": "eap-ig",
                "level": "node",
                "node_scores": node_scores,
                "safety_units": std_interop["safety_units"],
                "layer_suggestions": std_interop["layer_suggestions"],
                "provenance": prov_rec,
                "layer_saturation_table": saturation_table,
                "submodule_diagnostic_table": submodule_diag_table,
                "surgical_cap_summary": surgical_diag,
                "surgical_submodules_summary": sub_diag,
                "equalization_stats": equalization_stats,
                "margin_filter": margin_stats,
                "kl_diagnostic": kl_value,
                "calibration": asdict(calibration),
            }
        )
        scores_json.write_text(json.dumps(existing_scores_blob, indent=2) + "\n", encoding="utf-8")

        surgical_submodules_blob: Dict[str, Any] = {
            "version": existing_scores_blob.get("version", "1.0"),
            "timestamp": ts_now,
            "task": task_spec.name,
            "model": model_name,
            "model_revision": model_revision,
            "algorithm": "eap-ig",
            "level": "node",
            "selection_method": "submodule_ranked_surgical_mask",
            "max_surgical_submodules": int(max_surgical_submodules),
            "node_scores": node_scores,
            "safety_units": sub_su,
            "layer_suggestions": sub_ls,
            "surgical_submodules_summary": sub_diag,
            "submodule_diagnostic_table": submodule_diag_table,
            "provenance": prov_rec,
            "equalization_stats": equalization_stats,
            "margin_filter": margin_stats,
            "kl_diagnostic": kl_value,
            "calibration": asdict(calibration),
        }
        surgical_submodules_json.write_text(
            json.dumps(surgical_submodules_blob, indent=2) + "\n", encoding="utf-8"
        )

        tiers_map = export_surgical_tiers(
            node_scores=node_scores,
            model_name=model_name,
            n_layers=n_layers,
            output_dir=out_dir,
            tiers=DEFAULT_EXPORT_TIERS,
            max_surgical_layers=max_surgical_layers,
            provenance=prov_rec,
        )
        submodule_tiers_map = export_submodule_surgical_tiers(
            node_scores=node_scores,
            model_name=model_name,
            n_layers=n_layers,
            output_dir=out_dir,
            stem=stem,
            existing_targets=existing_targets,
            provenance=prov_rec,
            common_fields={
                "model_revision": model_revision,
                "task": task_spec.name,
                "timestamp": ts_now,
            },
        )
        return prov_rec, tiers_map, submodule_tiers_map

    _, exported_tiers, exported_submodule_tiers = _persist_phase_outputs(kl_value=None)
    hf_phase_a_url: Optional[str] = None
    if upload_hf:
        hf_phase_a_url = upload_stage2_to_hf(
            out_dir,
            repo_id=hf_repo,
            path_in_repo=DEFAULT_HF_SUBFOLDER,
            commit_message="Upload Stage 2 Phase A circuit and surgical masks",
        )

    _release_vram()

    # Phase B: full 4-pillar evaluation + KL diagnostic + final report/Hub sync
    eval_config = copy.deepcopy(config)
    eval_config["eval"] = {
        "num_examples": n_active,
        "full_faithfulness_eval": True,
        "n_stability_runs": int(n_stability_runs),
        "pillars": list(EVAL_PILLARS),
    }
    report = evaluate_circuit(
        eval_config,
        pruned_artifact_path=str(artifact_pt),
        scores_path=str(scores_pt),
        _model=model,
    )
    report_dict = _serialize_faithfulness_report(
        report,
        out_dir=out_dir,
        eval_pillars=EVAL_PILLARS,
    )

    _release_vram()

    kl_diag = compute_kl_diagnostic(
        model=model,
        scores_pt_path=scores_pt,
        task_spec=task_spec,
        discovery_cfg=discovery_cfg,
        pruning_cfg=pruning_cfg,
        device=profile.device,
    )

    _, exported_tiers, exported_submodule_tiers = _persist_phase_outputs(kl_value=kl_diag)

    faithfulness_out = out_dir / "faithfulness_report.json"
    summary: Dict[str, Any] = {
        "model": model_name,
        "model_revision": model_revision,
        "dataset": str(dataset_path),
        "runtime_profile": asdict(profile),
        "equalization_stats": equalization_stats,
        "calibration": asdict(calibration),
        "margin_filter": margin_stats,
        "eval_pillars": list(EVAL_PILLARS),
        "faithfulness_report": report_dict,
        "kl_diagnostic": kl_diag,
        "layer_saturation_table": saturation_table,
        "submodule_diagnostic_table": submodule_diag_table,
        "surgical_cap_summary": surgical_diag,
        "surgical_submodules_summary": sub_diag,
        "hf_phase_a_upload_url": hf_phase_a_url,
        "artifacts": {
            "circuit_pt": str(artifact_pt),
            "scores_pt": str(scores_pt),
            "scores_json": str(scores_json),
            "surgical_submodules_scores_json": str(surgical_submodules_json),
            "provenance_json": str(out_dir / PROVENANCE_FILENAME),
            "tier_exports": {str(k): str(v) for k, v in exported_tiers.items()},
            "submodule_tier_exports": {str(k): str(v) for k, v in exported_submodule_tiers.items()},
        },
    }
    faithfulness_out.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")

    if upload_hf:
        hf_url = upload_stage2_to_hf(
            out_dir,
            repo_id=hf_repo,
            path_in_repo=DEFAULT_HF_SUBFOLDER,
            commit_message="Upload Stage 2 Phase B faithfulness report and provenance",
        )
        summary["hf_upload_url"] = hf_url
        faithfulness_out.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")

    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stage 2: Multilingual safety circuit discovery and surgical locality export."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HF/TransformerLens model ID.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="Stage 1 JSONL dataset path.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    parser.add_argument(
        "--chat-template-mode",
        choices=list(VALID_MODES),
        default="auto",
        help="Chat template policy: auto, on, or off.",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size override.")
    parser.add_argument("--max-seq-len", type=int, default=None, help="Max sequence length override.")
    parser.add_argument(
        "--dtype",
        choices=["auto", "bfloat16", "float16", "float32"],
        default="auto",
        help="Model precision override.",
    )
    parser.add_argument("--device", default=None, help="Target device override (cuda/cpu/auto).")
    parser.add_argument("--top-fraction", type=float, default=0.10, help="Fraction of top nodes to keep.")
    parser.add_argument(
        "--max-surgical-layers",
        type=int,
        default=8,
        help="Cap unique layers in tier exports (ranked by cumulative node score).",
    )
    parser.add_argument(
        "--max-surgical-submodules",
        type=int,
        default=DEFAULT_MAX_SURGICAL_SUBMODULES,
        help="Cap layer-submodules (attn/mlp out of 2*n_layers) in Stage 3 surgical mask export.",
    )
    parser.add_argument(
        "--margin-threshold",
        type=float,
        default=0.5,
        help="Minimum positive baseline refusal separation delta_m = m(clean) - m(corrupt).",
    )
    parser.add_argument("--n-examples", type=int, default=None, help="Cap number of calibration examples.")
    parser.add_argument("--n-stability-runs", type=int, default=3, help="Stability pillar rediscovery runs.")
    parser.add_argument("--ig-steps", type=int, default=3, help="Integrated gradients steps for EAP-IG.")
    parser.add_argument("--stem", default="safety_circuit", help="Output filename stem.")
    parser.add_argument(
        "--low-vram-t4",
        action="store_true",
        default=False,
        help="Force 16 GB T4 compatibility mode (chat_template_mode=off, max_seq_len=128, batch_size=1, aggressive cache clearing).",
    )
    parser.add_argument(
        "--hydrate-hf",
        action="store_true",
        default=False,
        help="Download missing Stage 1 dataset from HF Hub before running discovery.",
    )
    parser.add_argument("--upload-hf", action="store_true", default=False, help="Upload artifacts to HF Hub.")
    parser.add_argument("--hf-repo", default=DEFAULT_HF_REPO, help="Target Hugging Face repository.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    run_stage2(
        model_name=args.model,
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        chat_template_mode=args.chat_template_mode,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        dtype=args.dtype,
        device=args.device,
        top_fraction=args.top_fraction,
        max_surgical_layers=args.max_surgical_layers,
        max_surgical_submodules=args.max_surgical_submodules,
        margin_threshold=args.margin_threshold,
        n_examples=args.n_examples,
        n_stability_runs=args.n_stability_runs,
        ig_steps=args.ig_steps,
        low_vram_t4=args.low_vram_t4,
        hydrate_hf=args.hydrate_hf,
        upload_hf=args.upload_hf,
        hf_repo=args.hf_repo,
        stem=args.stem,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
