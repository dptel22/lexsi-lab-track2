"""Stage 3: SafeTune Selectivity Hardening (SafeGrad) & Surgical C-ΔΘ Weight Transplant."""
from __future__ import annotations

import argparse
import gc
import inspect
import json
import logging
import os
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import transformers
from datasets import Dataset
from dotenv import load_dotenv
from huggingface_hub import HfApi, snapshot_download
from safetune.core.circuit_kit.adapter import load_circuit_info_from_file
from safetune.core.eval.pipeline.scorer import StringMatchJudge
from safetune.interventions.harden.safegrad import SafeGradConfig, SafeGradHFTrainer
from safetune.interventions.recover.ctheta import (
    _matched_param_keys,
    apply_ctheta_from_state_dicts,
    sweep_ctheta_strength,
)
from safetune.provenance import input_entry, read_provenance, write_provenance
from safetune.runner.harden._base import _keep_model_columns
from safetune.runner.utils.model_utils import free as safetune_free, lora_wrap, save_checkpoint
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, default_data_collator

try:
    import trackio  # type: ignore[import-untyped]
except Exception:  # pragma: no cover - optional telemetry dependency
    trackio = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)

PROVENANCE_FILENAME = "lexsi_provenance.json"
DEFAULT_MODEL = "CohereLabs/tiny-aya-fire"
DEFAULT_DATASET = Path("artifacts/stage1_dataset/calibration_100_pairs.jsonl")
DEFAULT_CIRCUIT_INFO = Path(
    "artifacts/stage2_circuits/safety_circuit_surgical_submodules_top4_scores.json"
)
DEFAULT_OUTPUT_DIR = Path("artifacts/stage3_repair")
DEFAULT_MODELS_DIR = Path("artifacts/stage3_models")
DEFAULT_HF_REPO = "soapyyapps/circuit-kavach-cyber-codemixed"
DEFAULT_HF_SUBFOLDER = "stage3_repair"
DEFAULT_ALPHA_GRID = (0.5, 1.0, 1.5)

ARTIFACT_LORA_NAME = "tiny-aya-fire-circuit-kavach-lora"
ARTIFACT_GLOBAL_NAME = "tiny-aya-fire-global-safetune"
ARTIFACT_SURGICAL_NAME = "tiny-aya-fire-circuit-kavach-merged"

_COHERE_LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
_GPT2_LORA_TARGETS = ("c_attn", "c_proj", "c_fc")

_REFUSAL_TEMPLATES: Dict[str, str] = {
    "en": (
        "I'm sorry, I cannot assist with creating attack scripts or exploiting {keyword}. "
        "I can help review defensive remediation controls and secure configuration practices instead."
    ),
    "hi_en": (
        "I'm sorry, I cannot assist with generating an exploit script or attacking {keyword}. "
        "Main aapko defensive security hardening aur safe remediation steps samjha sakta hoon."
    ),
    "gu_en": (
        "I'm sorry, I cannot assist with creating an attack payload or exploiting {keyword}. "
        "Hu tamne defensive remediation ane secure coding practices vishe madad kari shaku chhu."
    ),
    "kn_en": (
        "I'm sorry, I cannot assist with creating an attack script or exploiting {keyword}. "
        "Naanu nimage defensive security remediation mattu su-rakshita coding steps bagge sahaya maadaballe."
    ),
}

_DEFENSIVE_TEMPLATES: Dict[str, str] = {
    "web_api_exploitation": (
        "To defend production web services against {keyword}: validate and canonicalize untrusted inputs, "
        "enforce strict authentication/authorization checks on every endpoint, parameterize queries, apply least-privilege "
        "service permissions, and add automated regression security tests."
    ),
    "credential_financial_phishing": (
        "To protect users from {keyword}: deploy DMARC/DKIM/SPF authentication, avoid unverified external callback links, "
        "bind OTP/session tokens to device and origin metadata, and surface clear in-app security warnings."
    ),
    "recon_malware_delivery": (
        "To harden systems against {keyword}: maintain an asset inventory with continuous vulnerability patching, "
        "restrict administrative interfaces via zero-trust ACLs, enforce endpoint application allowlisting, and monitor anomalous egress."
    ),
}


@dataclass(frozen=True)
class Stage3RuntimeProfile:
    model_name: str
    device: str
    dtype: str
    templated: bool
    batch_size: int
    max_seq_len: int
    max_new_tokens: int
    ref_offload_cpu: bool
    gradient_checkpointing: bool
    low_vram_t4: bool


class _StateDictModuleProxy:
    """Zero-RAM proxy exposing ``.state_dict()`` to SafeTune's ``sweep_ctheta_strength``."""

    def __init__(self, state_dict: Dict[str, torch.Tensor]) -> None:
        self._sd = state_dict

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return self._sd


def _resolve_device(requested: Optional[str]) -> str:
    if requested is None or requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def _release_vram() -> None:
    """Release Python cyclic references, SafeTune helper caches, and CUDA allocator cache."""
    try:
        safetune_free()
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def resolve_model_revision(model: Any, model_name: str) -> Optional[str]:
    """Best-effort commit revision SHA for reproducibility (offline-safe for local directories)."""
    for owner in (model, getattr(model, "config", None)):
        commit = getattr(owner, "_commit_hash", None)
        if isinstance(commit, str) and commit.strip():
            return commit.strip()
    if Path(str(model_name)).exists():
        return None
    try:
        cfg = transformers.AutoConfig.from_pretrained(model_name, local_files_only=True)
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
    """Verify Hugging Face Hub authentication before allocating GPU resources."""
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
        raise RuntimeError(f"Hugging Face authentication check failed before Stage 3 compute: {exc}") from exc
    if isinstance(info, dict):
        return str(info.get("name") or info.get("fullname") or "authenticated")
    return "authenticated"


def load_calibration_pairs(dataset_path: Path | str) -> List[Dict[str, Any]]:
    src = Path(dataset_path)
    if not src.exists():
        raise FileNotFoundError(f"Stage 1 calibration dataset not found: {src}")
    pairs: List[Dict[str, Any]] = []
    required = ("pair_id", "language", "threat_category", "keyword", "harmful", "benign")
    with src.open("r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSON at {src}:{line_no}: {exc}") from exc
            for key in required:
                if key not in row or not str(row[key]).strip():
                    raise ValueError(f"Missing or empty field {key!r} at {src}:{line_no}")
            pairs.append(row)
    if not pairs:
        raise ValueError(f"Dataset file is empty: {src}")
    return pairs


def build_stage3_sft_and_safety_splits(
    pairs: Sequence[Dict[str, Any]],
    *,
    val_fraction: float = 0.20,
    seed: int = 42,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], List[Dict[str, Any]]]:
    """Stratify Stage 1 calibration pairs into train Q&A twins and validation pairs.

    Never touches ``eval_200_pairs.jsonl``. Produces:
    - ``train_benign_qa``: helpful defensive coding/security Q&A tuples for SFT objective
    - ``train_safety_qa``: harmful attack prompts paired with canonical GCG-detectable refusals
    - ``val_pairs``: held-out calibration pairs for sweeping C-ΔΘ alpha
    """
    if len(pairs) < 2:
        raise ValueError("Need at least 2 calibration pairs to create train/val splits")
    if not (0.0 < float(val_fraction) < 1.0):
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")

    rng = random.Random(int(seed))
    buckets: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for row in pairs:
        key = (str(row.get("language", "en")), str(row.get("threat_category", "general")))
        buckets.setdefault(key, []).append(row)

    target_val = max(1, min(len(pairs) - 1, int(round(len(pairs) * float(val_fraction)))))
    val_ids: set[str] = set()
    val_pairs: List[Dict[str, Any]] = []

    ordered_keys = sorted(buckets.keys())
    for key in ordered_keys:
        rng.shuffle(buckets[key])

    # Round-robin pick 1 from each (language, threat_category) stratum until target_val
    idx = 0
    while len(val_pairs) < target_val:
        progressed = False
        for key in ordered_keys:
            bucket = buckets[key]
            if idx < len(bucket) and len(val_pairs) < target_val:
                item = bucket[idx]
                pid = str(item["pair_id"])
                if pid not in val_ids:
                    val_ids.add(pid)
                    val_pairs.append(item)
                    progressed = True
        if not progressed:
            break
        idx += 1

    train_pairs = [row for row in pairs if str(row["pair_id"]) not in val_ids]
    if not train_pairs:
        train_pairs = list(pairs[:-1])
        val_pairs = [pairs[-1]]

    train_benign_qa: List[Tuple[str, str]] = []
    train_safety_qa: List[Tuple[str, str]] = []
    for row in train_pairs:
        lang = str(row.get("language", "en"))
        cat = str(row.get("threat_category", "web_api_exploitation"))
        kw = str(row.get("keyword", "security vulnerability")).strip()

        def_tmpl = _DEFENSIVE_TEMPLATES.get(cat, _DEFENSIVE_TEMPLATES["web_api_exploitation"])
        benign_ans = def_tmpl.format(keyword=kw)

        ref_tmpl = _REFUSAL_TEMPLATES.get(lang, _REFUSAL_TEMPLATES["en"])
        safety_ans = ref_tmpl.format(keyword=kw)

        train_benign_qa.append((str(row["benign"]), benign_ans))
        train_safety_qa.append((str(row["harmful"]), safety_ans))

    return train_benign_qa, train_safety_qa, val_pairs


_VALID_DTYPES: Dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def _torch_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name not in _VALID_DTYPES:
        raise ValueError(
            f"Unsupported dtype {dtype_name!r}; expected one of {sorted(_VALID_DTYPES)}"
        )
    return _VALID_DTYPES[dtype_name]


def _resolve_auto_dtype(model_name: str, dtype: str, resolved_device: str) -> str:
    if dtype != "auto":
        _torch_dtype(dtype)
        return dtype
    is_gpt2 = "gpt2" in model_name.lower()
    if resolved_device == "cpu" or is_gpt2:
        return "float32"
    if resolved_device.startswith("cuda"):
        if torch.cuda.is_available() and not torch.cuda.is_bf16_supported(
            including_emulation=False
        ):
            return "float16"
        return "bfloat16"
    return "bfloat16"


def resolve_stage3_runtime_profile(
    tokenizer: Any,
    *,
    model_name: str,
    dtype: str = "auto",
    device: Optional[str] = None,
    batch_size: int = 2,
    max_seq_len: Optional[int] = None,
    max_new_tokens: int = 128,
    ref_offload_cpu: bool = False,
    gradient_checkpointing: bool = False,
    low_vram_t4: bool = False,
) -> Stage3RuntimeProfile:
    resolved_device = _resolve_device(device)
    is_gpt2 = "gpt2" in model_name.lower()
    resolved_dtype = _resolve_auto_dtype(model_name, dtype, resolved_device)

    has_template = bool(getattr(tokenizer, "chat_template", None))
    templated = has_template and not is_gpt2

    eff_batch = 1 if low_vram_t4 else max(1, int(batch_size))
    eff_ref_offload = True if low_vram_t4 else bool(ref_offload_cpu)
    eff_ckpt = True if low_vram_t4 else bool(gradient_checkpointing)
    eff_max_new = min(64, int(max_new_tokens)) if low_vram_t4 else int(max_new_tokens)

    if max_seq_len is not None:
        eff_seq_len = int(max_seq_len)
    elif templated:
        eff_seq_len = 512
    else:
        eff_seq_len = 128

    return Stage3RuntimeProfile(
        model_name=model_name,
        device=resolved_device,
        dtype=resolved_dtype,
        templated=templated,
        batch_size=eff_batch,
        max_seq_len=eff_seq_len,
        max_new_tokens=eff_max_new,
        ref_offload_cpu=eff_ref_offload,
        gradient_checkpointing=eff_ckpt,
        low_vram_t4=bool(low_vram_t4),
    )


def _format_prompt_and_full(
    tokenizer: Any,
    user: str,
    assistant: str,
    *,
    templated: bool,
) -> Tuple[str, str]:
    if templated and getattr(tokenizer, "chat_template", None):
        msgs = [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]
        prompt_text = tokenizer.apply_chat_template(
            msgs[:-1], tokenize=False, add_generation_prompt=True
        )
        full_text = tokenizer.apply_chat_template(msgs, tokenize=False)
        return prompt_text, full_text
    prompt_text = f"User: {user}\nAssistant:"
    full_text = f"{prompt_text} {assistant}"
    if getattr(tokenizer, "eos_token", None):
        full_text = full_text + str(tokenizer.eos_token)
    return prompt_text, full_text


def tokenize_qa_dataset(
    rows: Sequence[Tuple[str, str]],
    tokenizer: Any,
    *,
    max_len: Optional[int] = None,
    templated: bool = False,
) -> Dataset:
    """Tokenize (user, assistant) rows into left-padded tensors with prompt tokens masked (-100).

    When ``templated=True`` with a long chat system preamble (e.g. Tiny Aya's ~366 tokens),
    automatically expands ``max_len`` past the longest prompt so every example retains
    supervised assistant response tokens rather than becoming an empty-loss no-op.
    """
    if tokenizer.pad_token is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(tokenizer, "padding_side", None) != "left":
        tokenizer.padding_side = "left"

    formatted: List[Tuple[str, str]] = [
        _format_prompt_and_full(tokenizer, user, ans, templated=templated) for user, ans in rows
    ]
    prompt_lens = [
        len(tokenizer(p_txt, add_special_tokens=False)["input_ids"]) for p_txt, _ in formatted
    ]
    full_lens = [
        len(tokenizer(f_txt, add_special_tokens=False)["input_ids"]) for _, f_txt in formatted
    ]
    longest_prompt = max(prompt_lens, default=0)
    longest_full = max(full_lens, default=0)

    if max_len is None:
        resolved_len = max(128, min(2048, max(longest_full + 4, longest_prompt + 64)))
    else:
        resolved_len = max(int(max_len), min(2048, longest_prompt + 32))

    # Avoid exceeding model positional limits if tokenizer reports a finite sane max length
    model_max = getattr(tokenizer, "model_max_length", None)
    if isinstance(model_max, int) and 64 <= model_max <= 100_000:
        resolved_len = min(resolved_len, model_max)

    records: List[Dict[str, Any]] = []
    for (p_txt, f_txt), p_len in zip(formatted, prompt_lens):
        enc = tokenizer(
            f_txt,
            truncation=True,
            max_length=resolved_len,
            padding="max_length",
            add_special_tokens=False,
        )
        input_ids = list(enc["input_ids"])
        mask = list(enc["attention_mask"])
        labels = list(input_ids)
        start = mask.index(1) if 1 in mask else len(mask)
        for idx in range(len(labels)):
            if idx < start + p_len or mask[idx] == 0:
                labels[idx] = -100
        if all(lbl == -100 for lbl in labels):
            raise ValueError(
                f"No supervised assistant tokens retained (resolved_len={resolved_len}, prompt_len={p_len})."
            )
        records.append({"input_ids": input_ids, "attention_mask": mask, "labels": labels})

    return _keep_model_columns(Dataset.from_list(records))


def resolve_lora_target_modules(model: Any) -> List[str]:
    present: set[str] = set()
    for name, _ in model.named_modules():
        leaf = name.rsplit(".", 1)[-1]
        present.add(leaf)
    cohere = [m for m in _COHERE_LORA_TARGETS if m in present]
    if cohere:
        return cohere
    gpt2 = [m for m in _GPT2_LORA_TARGETS if m in present]
    if gpt2:
        return gpt2
    raise ValueError("Could not detect standard attention/MLP LoRA projection modules on target model.")


def load_base_model_and_tokenizer(
    model_name: str,
    *,
    dtype: str = "float32",
    device: str = "cpu",
) -> Tuple[Any, Any]:
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    tdtype = _torch_dtype(dtype)
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=tdtype)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=tdtype)
    if getattr(model.config, "pad_token_id", None) is None and tok.pad_token_id is not None:
        model.config.pad_token_id = tok.pad_token_id
    model.to(device)
    return model, tok


def _load_reference_model(model_name: str, *, dtype: str, ref_device: str) -> Any:
    # PyTorch CPU lacks several float16 (Half) kernels; use compact bfloat16 when CPU-offloaded.
    eff_dtype = "bfloat16" if (ref_device == "cpu" and dtype == "float16") else dtype
    tdtype = _torch_dtype(eff_dtype)
    try:
        ref = AutoModelForCausalLM.from_pretrained(model_name, dtype=tdtype)
    except TypeError:
        ref = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=tdtype)
    ref.to(ref_device)
    ref.eval()
    for param in ref.parameters():
        param.requires_grad_(False)
    return ref


def run_safegrad_lora_harden(
    *,
    base_model: Any,
    tokenizer: Any,
    reference_model: Any,
    train_benign_qa: Sequence[Tuple[str, str]],
    train_safety_qa: Sequence[Tuple[str, str]],
    profile: Stage3RuntimeProfile,
    models_dir: Path,
    checkpoint_dir: Path,
    provenance_inputs: List[Dict[str, Any]],
    model_revision: Optional[str],
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    lr: float = 2e-4,
    epochs: float = 2.0,
    max_steps: int = -1,
    rho: float = 0.25,
    kl_temperature: float = 1.0,
    seed: int = 42,
    use_trackio: bool = False,
) -> Dict[str, Any]:
    """Wrap base_model ONCE in LoRA, train via SafeGradHFTrainer, save unmerged LoRA, then merge."""
    from peft import PeftModel

    target_modules = resolve_lora_target_modules(base_model)
    if isinstance(base_model, PeftModel):
        raise ValueError("base_model must be un-wrapped before calling lora_wrap (prevent double PEFT).")

    lora_model = lora_wrap(
        base_model,
        r=int(lora_r),
        lora_alpha=int(lora_alpha),
        lora_dropout=float(lora_dropout),
        target_modules=target_modules,
    )
    if isinstance(lora_model.get_base_model(), PeftModel):
        raise AssertionError("Double-wrapped PeftModel detected after lora_wrap!")

    if getattr(lora_model, "config", None) is not None:
        lora_model.config.use_cache = False
    if profile.gradient_checkpointing and hasattr(lora_model, "enable_input_require_grads"):
        lora_model.enable_input_require_grads()

    benign_ds = tokenize_qa_dataset(
        train_benign_qa, tokenizer, max_len=profile.max_seq_len, templated=profile.templated
    )
    safety_ds = tokenize_qa_dataset(
        train_safety_qa, tokenizer, max_len=profile.max_seq_len, templated=profile.templated
    )
    safety_loader = DataLoader(
        safety_ds,
        batch_size=profile.batch_size,
        shuffle=True,
        collate_fn=default_data_collator,
    )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    safegrad_args = SafeGradConfig(
        output_dir=str(checkpoint_dir),
        num_train_epochs=float(epochs),
        max_steps=int(max_steps),
        per_device_train_batch_size=profile.batch_size,
        learning_rate=float(lr),
        gradient_accumulation_steps=1,
        bf16=(profile.dtype == "bfloat16" and profile.device != "cpu"),
        fp16=(profile.dtype == "float16" and profile.device != "cpu"),
        use_cpu=(profile.device == "cpu"),
        gradient_checkpointing=profile.gradient_checkpointing,
        remove_unused_columns=False,
        save_strategy="no",
        logging_steps=1,
        report_to=["trackio"] if (use_trackio and trackio is not None) else [],
        seed=int(seed),
        dataloader_num_workers=0,
    )

    trainer_kwargs: Dict[str, Any] = {
        "model": lora_model,
        "args": safegrad_args,
        "train_dataset": benign_ds,
        "data_collator": default_data_collator,
        "safety_dataset": safety_loader,
        "reference_model": reference_model,
        "rho": float(rho),
        "kl_temperature": float(kl_temperature),
    }
    if "processing_class" in inspect.signature(transformers.Trainer.__init__).parameters:
        trainer_kwargs["processing_class"] = tokenizer

    trainer = SafeGradHFTrainer(**trainer_kwargs)
    train_result = trainer.train()

    # Artifact 3A: Save standalone unmerged LoRA adapter BEFORE merge_and_unload()
    lora_dir = models_dir / ARTIFACT_LORA_NAME
    lora_dir.mkdir(parents=True, exist_ok=True)
    lora_model.save_pretrained(str(lora_dir))
    tokenizer.save_pretrained(str(lora_dir))
    harden_params = {
        "method": "safegrad",
        "model_revision": model_revision,
        "lora_r": int(lora_r),
        "lora_alpha": int(lora_alpha),
        "lora_dropout": float(lora_dropout),
        "target_modules": target_modules,
        "lr": float(lr),
        "epochs": float(epochs),
        "max_steps": int(max_steps),
        "rho": float(rho),
        "kl_temperature": float(kl_temperature),
        "runtime_profile": asdict(profile),
        "train_benign_count": len(benign_ds),
        "train_safety_count": len(safety_ds),
    }
    harden_inputs = [
        entry for entry in provenance_inputs if entry.get("kind") != "circuit_mask"
    ]
    write_provenance(
        lora_dir,
        method="harden.safegrad.lora_adapter",
        inputs=harden_inputs,
        base_model=profile.model_name,
        params=harden_params,
    )

    # Restore inference config flags and detach input-grad hooks before merging LoRA -> Artifact 3B
    if hasattr(lora_model, "disable_input_require_grads"):
        try:
            lora_model.disable_input_require_grads()
        except Exception:
            pass
    if getattr(lora_model, "config", None) is not None:
        lora_model.config.use_cache = True
    merged_model = lora_model.merge_and_unload()
    if getattr(merged_model, "config", None) is not None:
        merged_model.config.use_cache = True
    ckpt2_path_str = save_checkpoint(
        merged_model,
        tokenizer,
        ARTIFACT_GLOBAL_NAME,
        out_dir=str(models_dir),
        method="harden.safegrad.merged",
        inputs=harden_inputs[1:],  # save_checkpoint automatically prepends model input
        params=harden_params,
    )
    ckpt2_sd = {k: v.detach().cpu().clone() for k, v in merged_model.state_dict().items()}

    log_history = list(getattr(trainer.state, "log_history", []) or [])
    train_metrics = dict(getattr(train_result, "metrics", {}) or {})
    del trainer, lora_model

    return {
        "lora_adapter_dir": str(lora_dir),
        "checkpoint_2_global_dir": ckpt2_path_str,
        "ckpt2_sd": ckpt2_sd,
        "merged_model": merged_model,
        "train_metrics": train_metrics,
        "log_history": log_history,
        "target_modules": target_modules,
    }


def _greedy_generate_texts(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    *,
    templated: bool,
    device: str,
    max_new_tokens: int,
) -> List[str]:
    model.eval()
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    formatted_prompts: List[str] = []
    for p in prompts:
        if templated and getattr(tokenizer, "chat_template", None):
            formatted_prompts.append(
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": p}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
        else:
            formatted_prompts.append(f"User: {p}\nAssistant:")

    gen_cfg = GenerationConfig.from_model_config(model.config)
    gen_cfg.do_sample = False
    gen_cfg.temperature = 1.0
    gen_cfg.top_p = 1.0
    gen_cfg.top_k = 50
    gen_cfg.max_new_tokens = int(max_new_tokens)
    gen_cfg.pad_token_id = tokenizer.pad_token_id
    gen_cfg.eos_token_id = tokenizer.eos_token_id

    outputs: List[str] = []
    max_ctx = getattr(tokenizer, "model_max_length", 1024)
    if not isinstance(max_ctx, int) or max_ctx > 100_000:
        max_ctx = 1024
    prompt_cap = max(32, min(1024, max_ctx - int(max_new_tokens)))

    with torch.no_grad():
        for txt in formatted_prompts:
            enc = tokenizer(
                txt,
                return_tensors="pt",
                truncation=True,
                max_length=prompt_cap,
                add_special_tokens=False,
            )
            input_ids = enc["input_ids"].to(device)
            attn_mask = enc["attention_mask"].to(device)
            seq = model.generate(
                input_ids=input_ids,
                attention_mask=attn_mask,
                generation_config=gen_cfg,
            )
            new_tokens = seq[0, input_ids.shape[1] :]
            outputs.append(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
    return outputs


def evaluate_selectivity_on_val(
    model: Any,
    tokenizer: Any,
    val_pairs: Sequence[Dict[str, Any]],
    *,
    templated: bool,
    device: str,
    max_new_tokens: int = 128,
) -> Dict[str, Any]:
    """Score (Attack Refusal Rate - Benign Over-Refusal Rate) using greedy decoding & GCG judge."""
    judge = StringMatchJudge(mode="gcg")
    harmful_prompts = [str(r["harmful"]) for r in val_pairs]
    benign_prompts = [str(r["benign"]) for r in val_pairs]

    harmful_replies = _greedy_generate_texts(
        model, tokenizer, harmful_prompts, templated=templated, device=device, max_new_tokens=max_new_tokens
    )
    benign_replies = _greedy_generate_texts(
        model, tokenizer, benign_prompts, templated=templated, device=device, max_new_tokens=max_new_tokens
    )

    h_judged = judge.score([{"prompt": p, "response": r} for p, r in zip(harmful_prompts, harmful_replies)])
    b_judged = judge.score([{"prompt": p, "response": r} for p, r in zip(benign_prompts, benign_replies)])

    n = max(1, len(val_pairs))
    attack_refusal_rate = sum(1 for item in h_judged if item["judgement"]["is_refusal"]) / n
    benign_over_refusal_rate = sum(1 for item in b_judged if item["judgement"]["is_refusal"]) / n
    selectivity = attack_refusal_rate - benign_over_refusal_rate
    return {
        "attack_refusal_rate": float(attack_refusal_rate),
        "benign_over_refusal_rate": float(benign_over_refusal_rate),
        "selectivity_score": float(selectivity),
        "n_val_pairs": len(val_pairs),
    }


def verify_surgical_bit_identity(
    base_sd: Dict[str, torch.Tensor],
    ckpt3_sd: Dict[str, torch.Tensor],
    circuit_info: Any,
) -> Dict[str, Any]:
    """Verify via torch.equal that 100% of weights outside the surgical mask remain bit-identical."""
    suggestions = getattr(circuit_info, "layer_suggestions", None)
    layer_subset = getattr(suggestions, "layer_subset", None) if suggestions is not None else None
    target_modules = getattr(suggestions, "target_modules", None) if suggestions is not None else None
    matched = set(_matched_param_keys(base_sd.keys(), layer_subset, target_modules))

    unmasked_violations: List[str] = []
    modified_param_count = 0
    total_param_count = 0

    for k, base_t in base_sd.items():
        n_elem = int(base_t.numel())
        total_param_count += n_elem
        if k in matched:
            modified_param_count += n_elem
        else:
            if k not in ckpt3_sd or not torch.equal(ckpt3_sd[k].cpu(), base_t.cpu()):
                unmasked_violations.append(k)

    su_meta = getattr(getattr(circuit_info, "safety_units", None), "metadata", None) or {}
    return {
        "bit_identical_unmasked_keys": len(unmasked_violations) == 0,
        "unmasked_violations": unmasked_violations,
        "matched_param_keys": sorted(matched),
        "matched_key_count": len(matched),
        "total_key_count": len(base_sd),
        "modified_param_count": modified_param_count,
        "total_param_count": total_param_count,
        "modified_param_pct": 100.0 * modified_param_count / max(1, total_param_count),
        "frozen_param_pct": 100.0 * (total_param_count - modified_param_count) / max(1, total_param_count),
        "projection_blocks_touched": su_meta.get("projection_blocks_touched"),
        "total_projection_blocks": su_meta.get("total_projection_blocks"),
    }


def run_ctheta_sweep_and_transplant(
    *,
    candidate_model: Any,
    tokenizer: Any,
    base_sd: Dict[str, torch.Tensor],
    ckpt2_sd: Dict[str, torch.Tensor],
    circuit_info: Any,
    val_pairs: Sequence[Dict[str, Any]],
    alpha_grid: Sequence[float],
    profile: Stage3RuntimeProfile,
    models_dir: Path,
    provenance_inputs: List[Dict[str, Any]],
    model_revision: Optional[str],
    circuit_info_path: str,
) -> Dict[str, Any]:
    """Sweep C-ΔΘ strength via sweep_ctheta_strength and materialize Checkpoint 3 via apply_ctheta_from_state_dicts."""
    dev = profile.device
    tdtype = _torch_dtype(profile.dtype)

    # Restore candidate_model strictly back to Checkpoint 1 base_sd in existing allocation
    candidate_model.load_state_dict({k: v.to(device=dev, dtype=tdtype) for k, v in base_sd.items()}, strict=True)

    pos_proxy = _StateDictModuleProxy(ckpt2_sd)
    neg_proxy = _StateDictModuleProxy(base_sd)
    detailed_metrics: Dict[float, Dict[str, Any]] = {}

    suggestions = getattr(circuit_info, "layer_suggestions", None)
    layer_subset = getattr(suggestions, "layer_subset", None) if suggestions is not None else None
    target_modules = getattr(suggestions, "target_modules", None) if suggestions is not None else None

    def _eval_and_record(mod: Any) -> float:
        # We evaluate during sweep_ctheta_strength in order of alpha_grid
        res = evaluate_selectivity_on_val(
            mod,
            tokenizer,
            val_pairs,
            templated=profile.templated,
            device=dev,
            max_new_tokens=profile.max_new_tokens,
        )
        idx = len(detailed_metrics)
        alpha_val = float(alpha_grid[idx]) if idx < len(alpha_grid) else float(idx)
        detailed_metrics[alpha_val] = res
        return float(res["selectivity_score"])

    sweep_rows = sweep_ctheta_strength(
        target=candidate_model,
        positive=pos_proxy,  # type: ignore[arg-type]
        negative=neg_proxy,  # type: ignore[arg-type]
        circuit_info=circuit_info,
        strengths=[float(a) for a in alpha_grid],
        eval_fn=_eval_and_record,
        layer_subset=layer_subset,
        target_modules=target_modules,
        higher_is_better=True,
    )

    merged_sweep: List[Dict[str, Any]] = []
    best_alpha = float(alpha_grid[0])
    for row in sweep_rows:
        a = float(row["strength"])
        det = detailed_metrics.get(a, {})
        entry = {
            "alpha": a,
            "score": float(row["score"]),
            "best": bool(row.get("best", False)),
            **det,
        }
        merged_sweep.append(entry)
        if entry["best"]:
            best_alpha = a

    # Guarantee candidate_model starts identically from base_sd, then apply winning C-ΔΘ transplant
    candidate_model.load_state_dict({k: v.to(device=dev, dtype=tdtype) for k, v in base_sd.items()}, strict=True)
    repaired_model = apply_ctheta_from_state_dicts(
        candidate_model,
        ckpt2_sd,
        base_sd,
        circuit_info,
        strength=float(best_alpha),
        layer_subset=layer_subset,
        target_modules=target_modules,
    )

    ckpt3_sd = {k: v.detach().cpu().clone() for k, v in repaired_model.state_dict().items()}
    audit = verify_surgical_bit_identity(base_sd, ckpt3_sd, circuit_info)
    if not audit["bit_identical_unmasked_keys"]:
        raise AssertionError(
            f"Surgical C-ΔΘ violated bit-identity outside mask on keys: {audit['unmasked_violations'][:5]}"
        )

    ctheta_params = {
        "method": "ctheta.apply_ctheta_from_state_dicts",
        "model_revision": model_revision,
        "best_alpha": float(best_alpha),
        "alpha_grid": [float(a) for a in alpha_grid],
        "alpha_sweep": merged_sweep,
        "circuit_info_path": circuit_info_path,
        "surgical_audit": {k: v for k, v in audit.items() if k != "matched_param_keys"},
        "generation_config": {"do_sample": False, "max_new_tokens": profile.max_new_tokens},
    }
    ckpt3_path_str = save_checkpoint(
        repaired_model,
        tokenizer,
        ARTIFACT_SURGICAL_NAME,
        out_dir=str(models_dir),
        method="recover.ctheta",
        inputs=provenance_inputs[1:],
        params=ctheta_params,
    )

    return {
        "checkpoint_3_surgical_dir": ckpt3_path_str,
        "best_alpha": float(best_alpha),
        "alpha_sweep": merged_sweep,
        "surgical_audit": audit,
    }


def upload_stage3_to_hf(
    folder_path: Path | str,
    *,
    repo_id: str = DEFAULT_HF_REPO,
    path_in_repo: str = DEFAULT_HF_SUBFOLDER,
    repo_type: str = "dataset",
    commit_message: str = "Upload Stage 3 SafeTune and C-ΔΘ repair artifacts",
) -> str:
    load_dotenv()
    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN environment variable is not set; cannot upload Stage 3 artifacts to Hugging Face."
        )
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type=repo_type, exist_ok=True)
    res = api.upload_folder(
        folder_path=str(folder_path),
        repo_id=repo_id,
        path_in_repo=path_in_repo,
        repo_type=repo_type,
        commit_message=commit_message,
    )
    commit_url = getattr(res, "commit_url", None) or str(res)
    return str(commit_url)


def run_stage3(
    *,
    model_name: str = DEFAULT_MODEL,
    dataset_path: Path | str = DEFAULT_DATASET,
    circuit_info_path: Path | str = DEFAULT_CIRCUIT_INFO,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    models_dir: Path | str = DEFAULT_MODELS_DIR,
    dtype: str = "auto",
    device: Optional[str] = None,
    batch_size: int = 2,
    max_seq_len: Optional[int] = None,
    max_new_tokens: int = 128,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    lr: float = 2e-4,
    epochs: float = 2.0,
    max_steps: int = -1,
    rho: float = 0.25,
    kl_temperature: float = 1.0,
    alpha_grid: Sequence[float] = DEFAULT_ALPHA_GRID,
    val_fraction: float = 0.20,
    seed: int = 42,
    ref_offload_cpu: bool = False,
    gradient_checkpointing: bool = False,
    low_vram_t4: bool = False,
    use_trackio: bool = False,
    hydrate_hf: bool = False,
    upload_hf: bool = False,
    hf_repo: str = DEFAULT_HF_REPO,
) -> Dict[str, Any]:
    if not alpha_grid:
        raise ValueError("alpha_grid must contain at least one strength coefficient")

    load_dotenv()
    if upload_hf:
        hf_user = verify_hf_auth()
        logger.info("Verified Hugging Face Hub auth before Stage 3 compute (user=%s).", hf_user)

    pairs = load_calibration_pairs(dataset_path)
    train_benign_qa, train_safety_qa, val_pairs = build_stage3_sft_and_safety_splits(
        pairs, val_fraction=val_fraction, seed=seed
    )

    ci_path_obj = Path(circuit_info_path)
    if hydrate_hf and not ci_path_obj.exists():
        hydrate_root = (
            ci_path_obj.parent.parent
            if ci_path_obj.parent.name == "stage2_circuits"
            else ci_path_obj.parent
        )
        hydrate_root.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=hf_repo,
            repo_type="dataset",
            local_dir=str(hydrate_root),
            allow_patterns=["stage2_circuits/**"],
            token=os.getenv("HF_TOKEN") or None,
        )
    if not ci_path_obj.exists():
        raise FileNotFoundError(f"Stage 2 circuit mask not found: {ci_path_obj}")
    circuit_info = load_circuit_info_from_file(str(ci_path_obj))

    resolved_device = _resolve_device(device)
    init_dtype = _resolve_auto_dtype(model_name, dtype, resolved_device)

    base_model, tokenizer = load_base_model_and_tokenizer(
        model_name, dtype=init_dtype, device=resolved_device
    )
    model_revision = resolve_model_revision(base_model, model_name)
    profile = resolve_stage3_runtime_profile(
        tokenizer,
        model_name=model_name,
        dtype=dtype,
        device=resolved_device,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        max_new_tokens=max_new_tokens,
        ref_offload_cpu=ref_offload_cpu,
        gradient_checkpointing=gradient_checkpointing,
        low_vram_t4=low_vram_t4,
    )

    # Snapshot pristine base weights BEFORE lora_wrap mutates modules
    base_sd = {k: v.detach().cpu().clone() for k, v in base_model.state_dict().items()}
    ref_device = "cpu" if profile.ref_offload_cpu else profile.device
    reference_model = _load_reference_model(model_name, dtype=profile.dtype, ref_device=ref_device)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    m_dir = Path(models_dir)
    m_dir.mkdir(parents=True, exist_ok=True)

    ds_obj = Path(dataset_path)
    ds_prov = read_provenance(ds_obj if ds_obj.is_dir() else ds_obj.parent)
    ci_prov = read_provenance(ci_path_obj.parent)
    prov_inputs: List[Dict[str, Any]] = [
        input_entry("model", model_name),
        {"kind": "dataset", "ref": str(dataset_path), "config": ds_obj.stem, "provenance": ds_prov},
        {"kind": "circuit_mask", "ref": str(ci_path_obj), "config": ci_path_obj.stem, "provenance": ci_prov},
    ]

    # Phase A: Single PEFT wrap + SafeGradHFTrainer -> Save Artifact 3A (LoRA) -> Merge Artifact 3B (Ckpt 2)
    harden_out = run_safegrad_lora_harden(
        base_model=base_model,
        tokenizer=tokenizer,
        reference_model=reference_model,
        train_benign_qa=train_benign_qa,
        train_safety_qa=train_safety_qa,
        profile=profile,
        models_dir=m_dir,
        checkpoint_dir=out_dir / "trainer_work",
        provenance_inputs=prov_inputs,
        model_revision=model_revision,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lr=lr,
        epochs=epochs,
        max_steps=max_steps,
        rho=rho,
        kl_temperature=kl_temperature,
        seed=seed,
        use_trackio=use_trackio,
    )

    candidate_model = harden_out.pop("merged_model")
    ckpt2_sd = harden_out.pop("ckpt2_sd")
    del reference_model
    _release_vram()

    hf_phase_a_url: Optional[str] = None
    if upload_hf:
        phase_a_meta = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": model_name,
            "model_revision": model_revision,
            "runtime_profile": asdict(profile),
            "harden_artifacts": {
                "lora_adapter_dir": harden_out["lora_adapter_dir"],
                "checkpoint_2_global_dir": harden_out["checkpoint_2_global_dir"],
            },
            "train_metrics": harden_out["train_metrics"],
        }
        (out_dir / "phase_a_status.json").write_text(
            json.dumps(phase_a_meta, indent=2, default=str) + "\n", encoding="utf-8"
        )
        upload_stage3_to_hf(
            m_dir,
            repo_id=hf_repo,
            path_in_repo="stage3_models",
            commit_message="Upload Stage 3 Phase A SafeGrad LoRA & Global Checkpoint 2 weights",
        )
        hf_phase_a_url = upload_stage3_to_hf(
            out_dir,
            repo_id=hf_repo,
            path_in_repo=DEFAULT_HF_SUBFOLDER,
            commit_message="Upload Stage 3 Phase A SafeGrad LoRA & Global Checkpoint 2 status",
        )

    # Phase B: 3-Input C-ΔΘ Strength Sweep -> Materialize Artifact 3C (Checkpoint 3)
    ctheta_out = run_ctheta_sweep_and_transplant(
        candidate_model=candidate_model,
        tokenizer=tokenizer,
        base_sd=base_sd,
        ckpt2_sd=ckpt2_sd,
        circuit_info=circuit_info,
        val_pairs=val_pairs,
        alpha_grid=alpha_grid,
        profile=profile,
        models_dir=m_dir,
        provenance_inputs=prov_inputs,
        model_revision=model_revision,
        circuit_info_path=str(ci_path_obj),
    )

    del candidate_model, ckpt2_sd, base_sd
    _release_vram()

    summary: Dict[str, Any] = {
        "model": model_name,
        "model_revision": model_revision,
        "dataset": str(dataset_path),
        "circuit_info_path": str(ci_path_obj),
        "runtime_profile": asdict(profile),
        "generation_config": {"do_sample": False, "max_new_tokens": profile.max_new_tokens},
        "best_alpha": ctheta_out["best_alpha"],
        "alpha_sweep": ctheta_out["alpha_sweep"],
        "surgical_audit": ctheta_out["surgical_audit"],
        "train_metrics": harden_out["train_metrics"],
        "target_modules": harden_out["target_modules"],
        "hf_phase_a_upload_url": hf_phase_a_url,
        "artifacts": {
            "lora_adapter_dir": harden_out["lora_adapter_dir"],
            "checkpoint_2_global_dir": harden_out["checkpoint_2_global_dir"],
            "checkpoint_3_surgical_dir": ctheta_out["checkpoint_3_surgical_dir"],
            "provenance_json": str(out_dir / PROVENANCE_FILENAME),
        },
    }

    write_provenance(
        out_dir,
        method="recover.safegrad_ctheta",
        inputs=prov_inputs,
        base_model=model_name,
        params={
            "model_revision": model_revision,
            "best_alpha": ctheta_out["best_alpha"],
            "alpha_grid": [float(a) for a in alpha_grid],
            "rho": float(rho),
            "kl_temperature": float(kl_temperature),
            "lora_r": int(lora_r),
            "lora_alpha": int(lora_alpha),
            "generation_config": summary["generation_config"],
            "surgical_audit": {
                k: v for k, v in ctheta_out["surgical_audit"].items() if k != "matched_param_keys"
            },
        },
    )

    summary_path = out_dir / "repair_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")

    if upload_hf:
        upload_stage3_to_hf(
            m_dir,
            repo_id=hf_repo,
            path_in_repo="stage3_models",
            commit_message="Upload Stage 3 Phase B surgical C-ΔΘ Checkpoint 3 weights",
        )
        hf_url = upload_stage3_to_hf(
            out_dir,
            repo_id=hf_repo,
            path_in_repo=DEFAULT_HF_SUBFOLDER,
            commit_message="Upload Stage 3 Phase B surgical C-ΔΘ summary and provenance",
        )
        summary["hf_upload_url"] = hf_url
        summary_path.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")

    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stage 3: SafeTune SafeGrad hardening and surgical C-ΔΘ weight transplant."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HF/local base model ID or path.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="Stage 1 calibration JSONL path.")
    parser.add_argument(
        "--circuit-info",
        type=Path,
        default=DEFAULT_CIRCUIT_INFO,
        help="Stage 2 surgical mask JSON path (load_circuit_info_from_file compatible).",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Stage 3 repair logs/provenance dir.")
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR, help="Output directory for 3 model artifacts.")
    parser.add_argument("--dtype", default="auto", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--rho", type=float, default=0.25)
    parser.add_argument("--kl-temperature", type=float, default=1.0)
    parser.add_argument("--alpha-grid", type=float, nargs="+", default=list(DEFAULT_ALPHA_GRID))
    parser.add_argument("--val-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ref-offload-cpu", action="store_true", help="Keep frozen reference model on CPU.")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument(
        "--low-vram-t4",
        action="store_true",
        help="16 GB T4 compatibility: batch_size=1, ref_offload_cpu=True, gradient_checkpointing=True.",
    )
    parser.add_argument("--use-trackio", action="store_true", help="Enable Trackio reporting if installed.")
    parser.add_argument(
        "--hydrate-hf",
        action="store_true",
        help="Download missing Stage 2 circuit mask artifacts from HF Hub before Stage 3.",
    )
    parser.add_argument("--upload-hf", action="store_true", help="Upload Stage 3 artifacts/provenance to HF Hub.")
    parser.add_argument("--hf-repo", default=DEFAULT_HF_REPO, help="Target Hugging Face dataset repository.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    return run_stage3(
        model_name=args.model,
        dataset_path=args.dataset,
        circuit_info_path=args.circuit_info,
        output_dir=args.output_dir,
        models_dir=args.models_dir,
        dtype=args.dtype,
        device=args.device,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        max_new_tokens=args.max_new_tokens,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lr=args.lr,
        epochs=args.epochs,
        max_steps=args.max_steps,
        rho=args.rho,
        kl_temperature=args.kl_temperature,
        alpha_grid=args.alpha_grid,
        val_fraction=args.val_fraction,
        seed=args.seed,
        ref_offload_cpu=args.ref_offload_cpu,
        gradient_checkpointing=args.gradient_checkpointing,
        low_vram_t4=args.low_vram_t4,
        use_trackio=args.use_trackio,
        hydrate_hf=args.hydrate_hf,
        upload_hf=args.upload_hf,
        hf_repo=args.hf_repo,
    )


if __name__ == "__main__":
    main()
