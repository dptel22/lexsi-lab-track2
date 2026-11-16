"""Tests for Stage 3 SafeTune hardening & Surgical C-ΔΘ repair (src/stage3_repair.py)."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import transformers


ROOT = Path(__file__).resolve().parent.parent
STAGE1_CALIB = ROOT / "artifacts" / "stage1_dataset" / "calibration_100_pairs.jsonl"
STAGE1_EVAL = ROOT / "artifacts" / "stage1_dataset" / "eval_200_pairs.jsonl"


def _create_tiny_cohere2_checkpoint(
    ckpt_dir: Path,
    *,
    with_chat_template: bool = True,
    long_system_preamble: bool = False,
    n_layers: int = 4,
) -> Path:
    """Create a tiny offline Cohere2ForCausalLM + tokenizer directory for fast CPU tests."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    words = [
        "<pad>", "<bos>", "<eos>", "<unk>",
        "User", "Assistant", ":", "I'm", "sorry", "sorry,", "cannot", "assist",
        "with", "Here", "are", "defensive", "steps", "for", "SQL", "injection",
        "payload", "UPI", "OTP", "bypass", "audit", "and", "protect", "systems",
        "safe", "remediation", "code", "patch", "attack", "script", "system",
        "preamble", "token", "check", "security", "review",
    ]
    # Include extra filler tokens when simulating a long system preamble
    for idx in range(40):
        words.append(f"sys_{idx}")
    vocab = {w: i for i, w in enumerate(words)}
    tok_obj = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    tok_obj.pre_tokenizer = Whitespace()
    tok = transformers.PreTrainedTokenizerFast(
        tokenizer_object=tok_obj,
        bos_token="<bos>",
        eos_token="<eos>",
        pad_token="<pad>",
        unk_token="<unk>",
    )
    tok.padding_side = "left"
    if with_chat_template:
        preamble = " ".join(f"sys_{i % 40}" for i in range(360)) if long_system_preamble else "system preamble"
        tok.chat_template = (
            "{{ bos_token }}"
            + preamble
            + " {% for m in messages %}{{ m['role'] }}: {{ m['content'] }} {% endfor %}"
            "{% if add_generation_prompt %}Assistant:{% endif %}"
        )
    tok.save_pretrained(ckpt_dir)

    torch.manual_seed(7)
    cfg = transformers.Cohere2Config(
        vocab_size=len(vocab),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=n_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=512,
        sliding_window=64,
        pad_token_id=vocab["<pad>"],
        bos_token_id=vocab["<bos>"],
        eos_token_id=vocab["<eos>"],
    )
    model = transformers.Cohere2ForCausalLM(cfg)
    model.save_pretrained(ckpt_dir)
    return ckpt_dir


def _write_stage2_submodule_mask(
    out_json: Path,
    *,
    model_name: str,
    selected_submodules: list[tuple[int, str]],
    n_layers: int = 4,
) -> Path:
    """Create a valid SafeTune load_circuit_info_from_file surgical mask."""
    is_gpt2 = "gpt2" in model_name.lower()
    target_modules: list[str] = []
    module_names: list[str] = []
    unit_ids: list[str] = []
    layer_subset = sorted({int(lyr) for lyr, _ in selected_submodules})
    for lyr, sub in selected_submodules:
        if sub == "self_attn":
            unit_ids.append(f"A{lyr}.0")
            if is_gpt2:
                module_names.append(f"blocks.{lyr}.attn")
                target_modules.extend([f"h.{lyr}.attn.c_attn", f"h.{lyr}.attn.c_proj"])
            else:
                module_names.append(f"blocks.{lyr}.self_attn")
                target_modules.extend(
                    [
                        f"layers.{lyr}.self_attn.q_proj",
                        f"layers.{lyr}.self_attn.k_proj",
                        f"layers.{lyr}.self_attn.v_proj",
                        f"layers.{lyr}.self_attn.o_proj",
                    ]
                )
        else:
            unit_ids.append(f"MLP {lyr}")
            module_names.append(f"blocks.{lyr}.mlp")
            if is_gpt2:
                target_modules.extend([f"h.{lyr}.mlp.c_fc", f"h.{lyr}.mlp.c_proj"])
            else:
                target_modules.extend(
                    [
                        f"layers.{lyr}.mlp.gate_proj",
                        f"layers.{lyr}.mlp.up_proj",
                        f"layers.{lyr}.mlp.down_proj",
                    ]
                )
    payload = {
        "version": "1.0",
        "model": model_name,
        "algorithm": "eap-ig",
        "level": "node",
        "selection_method": "submodule_ranked_surgical_mask",
        "max_surgical_submodules": len(selected_submodules),
        "node_scores": {u: 1.0 for u in unit_ids},
        "safety_units": {
            "layer_indices": layer_subset,
            "module_names": module_names,
            "unit_ids": unit_ids,
            "activation_correlation": {u: 1.0 for u in unit_ids},
            "metadata": {
                "projection_blocks_touched": len(selected_submodules),
                "total_projection_blocks": 2 * n_layers,
            },
        },
        "layer_suggestions": {
            "target_modules": target_modules,
            "layer_subset": layer_subset,
            "priority": {tm: 1.0 for tm in target_modules},
        },
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_json


def _sample_pairs(n: int = 12) -> list[dict]:
    langs = ["en", "hi_en", "gu_en", "kn_en"]
    cats = ["web_api_exploitation", "credential_financial_phishing", "recon_malware_delivery"]
    kws = ["SQL injection payload", "UPI OTP bypass", "audit systems"]
    rows = []
    for i in range(n):
        lang = langs[i % len(langs)]
        cat = cats[i % len(cats)]
        kw = kws[i % len(kws)]
        rows.append(
            {
                "pair_id": f"{lang}_{i:03d}",
                "language": lang,
                "threat_category": cat,
                "keyword": kw,
                "harmful": f"For {kw} create attack script to compromise target systems.",
                "benign": f"For {kw} provide defensive steps and safe code patch to protect systems.",
            }
        )
    return rows


def test_cli_defaults_and_args():
    import stage3_repair as s3

    parser = s3.build_arg_parser()
    defaults = parser.parse_args([])
    assert defaults.model == "CohereLabs/tiny-aya-fire"
    assert str(defaults.dataset).endswith("calibration_100_pairs.jsonl")
    assert str(defaults.circuit_info).endswith("safety_circuit_surgical_submodules_top4_scores.json")
    assert str(defaults.output_dir).endswith("stage3_repair")
    assert str(defaults.models_dir).endswith("stage3_models")
    assert defaults.alpha_grid == [0.5, 1.0, 1.5]
    assert defaults.low_vram_t4 is False
    assert defaults.upload_hf is False
    assert defaults.hf_repo == "soapyyapps/circuit-kavach-cyber-codemixed"


def test_calibration_split_stratified_and_disjoint_from_eval_200():
    import stage3_repair as s3
    from safetune.core.eval.pipeline.scorer import StringMatchJudge

    assert STAGE1_CALIB.exists() and STAGE1_EVAL.exists()
    calib_pairs = s3.load_calibration_pairs(STAGE1_CALIB)
    eval_ids = {json.loads(line)["pair_id"] for line in STAGE1_EVAL.read_text(encoding="utf-8").splitlines() if line.strip()}
    assert {p["pair_id"] for p in calib_pairs}.isdisjoint(eval_ids)

    benign_qa, safety_qa, val_pairs = s3.build_stage3_sft_and_safety_splits(
        calib_pairs, val_fraction=0.20, seed=42
    )
    assert len(val_pairs) == 20
    assert len(benign_qa) == 80
    assert len(safety_qa) == 80
    assert {p["language"] for p in val_pairs} == {"en", "hi_en", "gu_en", "kn_en"}

    # Safety answers must trigger GCG refusal detector while benign defensive answers must NOT falsely refuse
    judge = StringMatchJudge(mode="gcg")
    for prompt, ref_ans in safety_qa[:8]:
        assert judge.score_one(ref_ans) == 0.0, f"Safety answer not recognized as refusal: {ref_ans}"
    for prompt, ben_ans in benign_qa[:8]:
        assert judge.score_one(ben_ans) == 1.0, f"Benign defensive answer falsely matches refusal prefix: {ben_ans}"


def test_tokenized_dataset_preserves_supervised_tokens_under_long_system_preamble(tmp_path):
    import stage3_repair as s3

    ckpt = _create_tiny_cohere2_checkpoint(
        tmp_path / "long_tmpl", with_chat_template=True, long_system_preamble=True, n_layers=2
    )
    tok = transformers.AutoTokenizer.from_pretrained(ckpt)
    rows = [("How to protect SQL injection payload?", "Use parameterized SQL queries and input validation.")]

    # Even if low-memory caller passes max_len=128, templated tokenization must auto-expand past preamble
    ds_tmpl = s3.tokenize_qa_dataset(rows, tok, max_len=128, templated=True)
    assert len(ds_tmpl) == 1
    labels = ds_tmpl[0]["labels"]
    assert sum(1 for x in labels if x != -100) >= 3

    # Raw mode fallback when chat_template is None (e.g. gpt2)
    raw_ckpt = _create_tiny_cohere2_checkpoint(
        tmp_path / "raw_tok", with_chat_template=False, n_layers=2
    )
    raw_tok = transformers.AutoTokenizer.from_pretrained(raw_ckpt)
    ds_raw = s3.tokenize_qa_dataset(rows, raw_tok, max_len=64, templated=False)
    raw_labels = ds_raw[0]["labels"]
    assert sum(1 for x in raw_labels if x == -100) >= 1
    assert sum(1 for x in raw_labels if x != -100) >= 3


def test_single_peft_wrap_unmerged_lora_saved_before_merge_and_ctheta_bit_identity(tmp_path):
    import stage3_repair as s3
    from peft import PeftModel
    from safetune.core.circuit_kit.adapter import load_circuit_info_from_file
    from safetune.interventions.recover.ctheta import _matched_param_keys
    from safetune.runner.utils import model_utils as mu

    ckpt = _create_tiny_cohere2_checkpoint(tmp_path / "tiny_aya", with_chat_template=True, n_layers=4)
    mask_path = _write_stage2_submodule_mask(
        tmp_path / "stage2_circuits" / "safety_circuit_surgical_submodules_top2_scores.json",
        model_name=str(ckpt),
        selected_submodules=[(1, "mlp"), (3, "self_attn")],
        n_layers=4,
    )
    ds_path = tmp_path / "stage1_dataset" / "calibration_100_pairs.jsonl"
    ds_path.parent.mkdir(parents=True, exist_ok=True)
    ds_path.write_text("\n".join(json.dumps(r) for r in _sample_pairs(12)) + "\n", encoding="utf-8")
    (ds_path.parent / "lexsi_provenance.json").write_text(
        json.dumps({"schema": "lexsi.provenance/1", "method": "stage1_test", "params": {}}),
        encoding="utf-8",
    )
    (mask_path.parent / "lexsi_provenance.json").write_text(
        json.dumps({"schema": "lexsi.provenance/1", "method": "discover.eap-ig", "params": {}}),
        encoding="utf-8",
    )

    out_dir = tmp_path / "stage3_repair"
    models_dir = tmp_path / "stage3_models"

    wrap_calls = []
    orig_lora_wrap = mu.lora_wrap

    def _spy_lora_wrap(model, **kwargs):
        wrapped = orig_lora_wrap(model, **kwargs)
        assert isinstance(wrapped, PeftModel)
        assert not isinstance(wrapped.get_base_model(), PeftModel), "Double-wrapped PEFT model detected!"
        wrap_calls.append(kwargs)
        return wrapped

    with patch.object(s3, "lora_wrap", side_effect=_spy_lora_wrap):
        summary = s3.run_stage3(
            model_name=str(ckpt),
            dataset_path=ds_path,
            circuit_info_path=mask_path,
            output_dir=out_dir,
            models_dir=models_dir,
            dtype="float32",
            device="cpu",
            max_steps=2,
            batch_size=2,
            lora_r=4,
            lora_alpha=8,
            alpha_grid=[0.5, 1.0, 1.5],
            max_new_tokens=8,
            upload_hf=False,
        )

    assert len(wrap_calls) == 1
    assert summary["best_alpha"] in [0.5, 1.0, 1.5]
    assert len(summary["alpha_sweep"]) == 3
    assert summary["generation_config"]["do_sample"] is False
    assert "model_revision" in summary

    # Artifact 3A: Unmerged LoRA adapter exists BEFORE merge
    lora_dir = Path(summary["artifacts"]["lora_adapter_dir"])
    assert (lora_dir / "adapter_config.json").exists()
    assert (lora_dir / "adapter_model.safetensors").exists() or (lora_dir / "adapter_model.bin").exists()
    assert (lora_dir / "lexsi_provenance.json").exists()

    # Artifact 3B & 3C: Full merged checkpoints exist and load cleanly
    ckpt2_dir = Path(summary["artifacts"]["checkpoint_2_global_dir"])
    ckpt3_dir = Path(summary["artifacts"]["checkpoint_3_surgical_dir"])
    assert (ckpt2_dir / "lexsi_provenance.json").exists()
    assert (ckpt3_dir / "lexsi_provenance.json").exists()

    base_model = transformers.Cohere2ForCausalLM.from_pretrained(ckpt)
    ckpt2_model = transformers.Cohere2ForCausalLM.from_pretrained(ckpt2_dir)
    ckpt3_model = transformers.Cohere2ForCausalLM.from_pretrained(ckpt3_dir)

    base_sd = base_model.state_dict()
    ckpt2_sd = ckpt2_model.state_dict()
    ckpt3_sd = ckpt3_model.state_dict()

    ci = load_circuit_info_from_file(str(mask_path))
    matched = set(
        _matched_param_keys(
            base_sd.keys(),
            ci.layer_suggestions.layer_subset,
            ci.layer_suggestions.target_modules,
        )
    )
    assert len(matched) > 0
    for key in base_sd:
        if key in matched:
            expected = base_sd[key] + float(summary["best_alpha"]) * (ckpt2_sd[key] - base_sd[key])
            assert torch.allclose(ckpt3_sd[key], expected, atol=1e-5), f"Masked key {key} C-ΔΘ mismatch"
        else:
            assert torch.equal(ckpt3_sd[key], base_sd[key]), f"Unmasked key {key} was modified!"

    assert summary["surgical_audit"]["bit_identical_unmasked_keys"] is True
    assert summary["surgical_audit"]["unmasked_violations"] == []
    assert 0.0 < summary["surgical_audit"]["modified_param_pct"] < 50.0

    prov = json.loads((out_dir / "lexsi_provenance.json").read_text(encoding="utf-8"))
    assert prov["params"]["model_revision"] == summary["model_revision"]
    assert any(inp.get("kind") == "circuit_mask" for inp in prov.get("inputs", []))


def test_real_gpt2_cpu_stage3_end_to_end_with_stage1_calibration(tmp_path):
    """Run Stage 3 on real CPU gpt2 + real Stage 1 calibration subset + 2/24 submodule mask."""
    import stage3_repair as s3
    from safetune.interventions.recover.ctheta import _matched_param_keys
    from safetune.core.circuit_kit.adapter import load_circuit_info_from_file

    mask_path = _write_stage2_submodule_mask(
        tmp_path / "gpt2_top2_scores.json",
        model_name="gpt2",
        selected_submodules=[(2, "mlp"), (5, "self_attn")],
        n_layers=12,
    )
    calib_rows = [json.loads(line) for line in STAGE1_CALIB.read_text(encoding="utf-8").splitlines() if line.strip()][:8]
    ds_path = tmp_path / "calib_8.jsonl"
    ds_path.write_text("\n".join(json.dumps(r) for r in calib_rows) + "\n", encoding="utf-8")

    summary = s3.run_stage3(
        model_name="gpt2",
        dataset_path=ds_path,
        circuit_info_path=mask_path,
        output_dir=tmp_path / "gpt2_stage3_out",
        models_dir=tmp_path / "gpt2_stage3_models",
        dtype="float32",
        device="cpu",
        max_steps=1,
        batch_size=1,
        lora_r=4,
        lora_alpha=8,
        alpha_grid=[0.5, 1.0, 1.5],
        max_new_tokens=6,
        upload_hf=False,
    )

    assert summary["surgical_audit"]["bit_identical_unmasked_keys"] is True
    assert summary["model_revision"] is not None
    ckpt3_dir = Path(summary["artifacts"]["checkpoint_3_surgical_dir"])
    base_sd = transformers.AutoModelForCausalLM.from_pretrained("gpt2").state_dict()
    ckpt3_sd = transformers.AutoModelForCausalLM.from_pretrained(ckpt3_dir).state_dict()
    ci = load_circuit_info_from_file(str(mask_path))
    matched = set(
        _matched_param_keys(base_sd.keys(), ci.layer_suggestions.layer_subset, ci.layer_suggestions.target_modules)
    )
    assert len(matched) == 4  # 2 GPT-2 submodules * 2 projection matrices
    for k in base_sd:
        if k not in matched:
            assert torch.equal(ckpt3_sd[k], base_sd[k])


def test_low_vram_t4_offloads_ref_model_to_cpu_and_releases_vram(tmp_path):
    import stage3_repair as s3

    ckpt = _create_tiny_cohere2_checkpoint(tmp_path / "t4_aya", with_chat_template=False, n_layers=2)
    mask_path = _write_stage2_submodule_mask(
        tmp_path / "mask.json", model_name=str(ckpt), selected_submodules=[(0, "mlp")], n_layers=2
    )
    ds_path = tmp_path / "calib.jsonl"
    ds_path.write_text("\n".join(json.dumps(r) for r in _sample_pairs(8)) + "\n", encoding="utf-8")

    seen_ref_devices = []
    release_calls = []
    orig_trainer_init = s3.SafeGradHFTrainer.__init__
    orig_release = s3._release_vram

    def _spy_trainer_init(self, *args, **kwargs):
        ref = kwargs.get("reference_model")
        if ref is not None:
            seen_ref_devices.append(str(next(ref.parameters()).device))
        return orig_trainer_init(self, *args, **kwargs)

    def _spy_release():
        release_calls.append("release")
        return orig_release()

    with patch.object(s3.SafeGradHFTrainer, "__init__", _spy_trainer_init), \
         patch.object(s3, "_release_vram", side_effect=_spy_release):
        summary = s3.run_stage3(
            model_name=str(ckpt),
            dataset_path=ds_path,
            circuit_info_path=mask_path,
            output_dir=tmp_path / "t4_out",
            models_dir=tmp_path / "t4_models",
            dtype="float32",
            device="cpu",
            max_steps=1,
            batch_size=4,
            low_vram_t4=True,
            upload_hf=False,
        )

    assert seen_ref_devices == ["cpu"]
    assert len(release_calls) >= 2
    assert summary["runtime_profile"]["low_vram_t4"] is True
    assert summary["runtime_profile"]["ref_offload_cpu"] is True
    assert summary["runtime_profile"]["batch_size"] == 1
    assert summary["runtime_profile"]["gradient_checkpointing"] is True


def test_upload_hf_fails_fast_and_uploads_phase_a_before_ctheta(tmp_path, monkeypatch):
    import stage3_repair as s3

    ds_path = tmp_path / "calib.jsonl"
    ds_path.write_text("\n".join(json.dumps(r) for r in _sample_pairs(8)) + "\n", encoding="utf-8")
    mask_path = _write_stage2_submodule_mask(
        tmp_path / "mask.json", model_name="unused", selected_submodules=[(0, "mlp")], n_layers=2
    )

    # 1) Missing token aborts BEFORE loading base model
    monkeypatch.delenv("HF_TOKEN", raising=False)
    load_spy = MagicMock()
    with patch("stage3_repair.load_dotenv"), patch.object(s3, "load_base_model_and_tokenizer", load_spy):
        with pytest.raises(RuntimeError, match="HF_TOKEN"):
            s3.run_stage3(
                model_name="unused",
                dataset_path=ds_path,
                circuit_info_path=mask_path,
                output_dir=tmp_path / "fail_fast_out",
                models_dir=tmp_path / "fail_fast_models",
                upload_hf=True,
            )
    load_spy.assert_not_called()

    # 2) Valid token runs whoami -> SafeGrad -> Phase A upload -> C-ΔΘ -> Phase B upload
    ckpt = _create_tiny_cohere2_checkpoint(tmp_path / "hf_aya", with_chat_template=False, n_layers=2)
    mask_path = _write_stage2_submodule_mask(
        tmp_path / "mask2.json", model_name=str(ckpt), selected_submodules=[(0, "mlp")], n_layers=2
    )
    monkeypatch.setenv("HF_TOKEN", "hf_dummy_token_do_not_leak")

    events: list[str] = []
    commits: list[str] = []
    upload_paths: list[str] = []
    mock_api = MagicMock()
    mock_api.whoami.side_effect = lambda: events.append("whoami") or {"name": "tester"}

    def _fake_upload_folder(**kwargs):
        events.append("upload")
        commits.append(str(kwargs.get("commit_message", "")))
        upload_paths.append(str(kwargs.get("path_in_repo", "")))
        return SimpleNamespace(commit_url=f"https://huggingface.co/datasets/test/commit/{len(commits)}")

    mock_api.upload_folder.side_effect = _fake_upload_folder

    orig_harden = s3.run_safegrad_lora_harden
    orig_ctheta = s3.run_ctheta_sweep_and_transplant

    with patch("stage3_repair.HfApi", return_value=mock_api), \
         patch.object(s3, "run_safegrad_lora_harden", side_effect=lambda *a, **kw: events.append("safegrad") or orig_harden(*a, **kw)), \
         patch.object(s3, "run_ctheta_sweep_and_transplant", side_effect=lambda *a, **kw: events.append("ctheta") or orig_ctheta(*a, **kw)):
        summary = s3.run_stage3(
            model_name=str(ckpt),
            dataset_path=ds_path,
            circuit_info_path=mask_path,
            output_dir=tmp_path / "ok_out",
            models_dir=tmp_path / "ok_models",
            dtype="float32",
            device="cpu",
            max_steps=1,
            batch_size=1,
            alpha_grid=[1.0],
            max_new_tokens=6,
            upload_hf=True,
        )

    assert events[0] == "whoami"
    assert "safegrad" in events and "ctheta" in events
    safe_idx = events.index("safegrad")
    ctheta_idx = events.index("ctheta")
    assert events.count("upload") >= 4
    phase_a_paths = upload_paths[: ctheta_idx - safe_idx - 1]
    phase_b_paths = upload_paths[ctheta_idx - safe_idx - 1 :]
    assert {"stage3_models", "stage3_repair"}.issubset(set(phase_a_paths))
    assert {"stage3_models", "stage3_repair"}.issubset(set(phase_b_paths))
    assert mock_api.create_repo.call_count >= 1
    assert summary["hf_phase_a_upload_url"]
    assert summary["hf_upload_url"]


def test_stage3_t4_float16_ref_offload_use_cache_and_hydrate_hf(tmp_path, monkeypatch):
    import stage3_repair as s3

    args = s3.build_arg_parser().parse_args([])
    assert args.hydrate_hf is False

    # T4 CUDA with including_emulation=False -> float16, invalid dtype -> ValueError
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda including_emulation=True: False)

    ckpt = _create_tiny_cohere2_checkpoint(tmp_path / "tiny_aya_t4", with_chat_template=True, n_layers=2)
    tok = transformers.AutoTokenizer.from_pretrained(ckpt)

    prof_t4 = s3.resolve_stage3_runtime_profile(
        tok, model_name="CohereLabs/tiny-aya-fire", dtype="auto", device="cuda", low_vram_t4=True
    )
    assert prof_t4.dtype == "float16"
    with pytest.raises(ValueError, match="dtype"):
        s3._torch_dtype("int8")

    # CPU-offloaded reference model must never load float16 on CPU
    seen_ref_dtypes = []
    orig_from_pretrained = transformers.AutoModelForCausalLM.from_pretrained

    def _spy_ref_load(*args, **kwargs):
        seen_ref_dtypes.append(kwargs.get("dtype") or kwargs.get("torch_dtype"))
        return orig_from_pretrained(*args, **kwargs)

    with patch.object(s3.AutoModelForCausalLM, "from_pretrained", side_effect=_spy_ref_load):
        ref_mod = s3._load_reference_model(str(ckpt), dtype="float16", ref_device="cpu")
    assert seen_ref_dtypes and seen_ref_dtypes[-1] != torch.float16
    assert ref_mod.device.type == "cpu"

    # Merged ckpt2 & ckpt3 must restore use_cache=True and support --hydrate-hf
    ds_path = tmp_path / "calib.jsonl"
    ds_path.write_text("\n".join(json.dumps(r) for r in _sample_pairs(8)) + "\n", encoding="utf-8")
    hydrated_mask_rel = tmp_path / "stage2_circuits" / "safety_circuit_surgical_submodules_top2_scores.json"

    def _fake_snapshot_download(**kwargs):
        _write_stage2_submodule_mask(
            hydrated_mask_rel,
            model_name=str(ckpt),
            selected_submodules=[(0, "mlp")],
            n_layers=2,
        )
        return str(hydrated_mask_rel.parent.parent)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with patch.object(s3, "snapshot_download", side_effect=_fake_snapshot_download, create=True) as dl_spy:
        summary = s3.run_stage3(
            model_name=str(ckpt),
            dataset_path=ds_path,
            circuit_info_path=hydrated_mask_rel,
            output_dir=tmp_path / "out_hyd",
            models_dir=tmp_path / "models_hyd",
            dtype="float32",
            device="cpu",
            max_steps=1,
            batch_size=1,
            alpha_grid=[1.0],
            max_new_tokens=6,
            hydrate_hf=True,
            upload_hf=False,
        )
    dl_spy.assert_called_once()
    ckpt2_cfg = transformers.AutoConfig.from_pretrained(summary["artifacts"]["checkpoint_2_global_dir"])
    ckpt3_cfg = transformers.AutoConfig.from_pretrained(summary["artifacts"]["checkpoint_3_surgical_dir"])
    assert ckpt2_cfg.use_cache is True
    assert ckpt3_cfg.use_cache is True


def test_stage3_hydrates_missing_dataset_before_load(tmp_path, monkeypatch) -> None:
    import sys
    from unittest.mock import patch

    src_dir = str(Path(__file__).resolve().parents[1] / "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    import stage3_repair as s3

    ckpt = _create_tiny_cohere2_checkpoint(tmp_path / "ckpt_ds", n_layers=2)
    missing_ds = tmp_path / "stage1_dataset" / "calibration_50_pairs.jsonl"
    missing_mask = tmp_path / "stage2_circuits" / "safety_circuit_surgical_submodules_top2_scores.json"

    def _fake_dl(**kwargs):
        assert set(kwargs["allow_patterns"]) == {"stage1_dataset/**", "stage2_circuits/**"}
        missing_ds.parent.mkdir(parents=True, exist_ok=True)
        missing_ds.write_text("\n".join(json.dumps(r) for r in _sample_pairs(8)) + "\n", encoding="utf-8")
        _write_stage2_submodule_mask(
            missing_mask,
            model_name=str(ckpt),
            selected_submodules=[(0, "mlp")],
            n_layers=2,
        )
        return str(tmp_path)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with patch.object(s3, "snapshot_download", side_effect=_fake_dl, create=True) as dl_spy:
        summary = s3.run_stage3(
            model_name=str(ckpt),
            dataset_path=missing_ds,
            circuit_info_path=missing_mask,
            output_dir=tmp_path / "out_ds",
            models_dir=tmp_path / "models_ds",
            dtype="float32",
            device="cpu",
            max_steps=1,
            batch_size=1,
            alpha_grid=[1.0],
            max_new_tokens=6,
            hydrate_hf=True,
            upload_hf=False,
        )
    dl_spy.assert_called_once()
    assert Path(summary["artifacts"]["checkpoint_3_surgical_dir"]).is_dir()


def test_safegrad_sft_alignment_loss_mode_uses_supervised_refusal_labels(tmp_path) -> None:
    """SafeGradHFTrainer alignment_loss_mode='sft' computes supervised refusal loss and skips reference_model; 'auto' keeps KL."""
    import stage3_repair as s3
    from torch.utils.data import DataLoader
    from transformers import default_data_collator

    ckpt = _create_tiny_cohere2_checkpoint(tmp_path / "sft_mode", with_chat_template=False, n_layers=2)
    tok = transformers.AutoTokenizer.from_pretrained(ckpt)
    student = transformers.Cohere2ForCausalLM.from_pretrained(ckpt)
    student = s3.lora_wrap(
        student, r=4, lora_alpha=8, lora_dropout=0.0, target_modules=s3.resolve_lora_target_modules(student)
    )
    ref_model = transformers.Cohere2ForCausalLM.from_pretrained(ckpt)

    ds = s3.tokenize_qa_dataset(
        [("For SQL injection create attack script.", "I'm sorry, I cannot assist with attack scripts.")],
        tok,
        max_len=64,
        templated=False,
    )
    loader = DataLoader(ds, batch_size=1, collate_fn=default_data_collator)
    batch = next(iter(loader))

    args = s3.SafeGradConfig(
        output_dir=str(tmp_path / "work"),
        per_device_train_batch_size=1,
        max_steps=1,
        use_cpu=True,
        save_strategy="no",
        report_to=[],
    )

    ref_fwd_spy = MagicMock(side_effect=ref_model.forward)
    with patch.object(ref_model, "forward", ref_fwd_spy):
        sft_trainer = s3.SafeGradHFTrainer(
            model=student,
            args=args,
            train_dataset=ds,
            data_collator=default_data_collator,
            safety_dataset=loader,
            reference_model=ref_model,
            alignment_loss_mode="sft",
        )
        grads_sft = sft_trainer._alignment_gradient(student, batch)

    assert ref_fwd_spy.call_count == 0
    assert grads_sft is not None
    expected_loss = student(**{k: v for k, v in batch.items()}).loss
    trainable = [p for p in student.parameters() if p.requires_grad]
    expected_grads = torch.autograd.grad(expected_loss, trainable, allow_unused=True)
    for g_act, g_exp in zip(grads_sft, expected_grads):
        if g_exp is None:
            assert torch.count_nonzero(g_act) == 0
        else:
            assert torch.allclose(g_act, g_exp, atol=1e-5)

    # Default alignment_loss_mode='auto' preserves SafeTune KL behavior when reference_model is present
    ref_fwd_auto = MagicMock(side_effect=ref_model.forward)
    with patch.object(ref_model, "forward", ref_fwd_auto):
        auto_trainer = s3.SafeGradHFTrainer(
            model=student,
            args=args,
            train_dataset=ds,
            data_collator=default_data_collator,
            safety_dataset=loader,
            reference_model=ref_model,
        )
        grads_auto = auto_trainer._alignment_gradient(student, batch)
    assert ref_fwd_auto.call_count >= 1
    assert grads_auto is not None


def test_surgical_ctheta_adamw_param_groups_concentrate_updates_in_circuit(tmp_path) -> None:
    """AdamW param-group LR scaling concentrates updates in circuit submodules while keeping C-ΔΘ exact."""
    import stage3_repair as s3
    from safetune.core.circuit_kit.adapter import load_circuit_info_from_file
    from safetune.interventions.recover.ctheta import _matched_param_keys

    ckpt = _create_tiny_cohere2_checkpoint(tmp_path / "conc_aya", with_chat_template=True, n_layers=4)
    mask_path = _write_stage2_submodule_mask(
        tmp_path / "mask.json",
        model_name=str(ckpt),
        selected_submodules=[(1, "mlp")],
        n_layers=4,
    )
    ds_path = tmp_path / "calib.jsonl"
    ds_path.write_text("\n".join(json.dumps(r) for r in _sample_pairs(12)) + "\n", encoding="utf-8")

    summary = s3.run_stage3(
        model_name=str(ckpt),
        dataset_path=ds_path,
        circuit_info_path=mask_path,
        output_dir=tmp_path / "out",
        models_dir=tmp_path / "models",
        dtype="float32",
        device="cpu",
        max_steps=4,
        batch_size=2,
        lora_r=4,
        lora_alpha=8,
        lr=5e-4,
        alpha_grid=[1.0],
        max_new_tokens=6,
        non_circuit_lr_scale=0.10,
        upload_hf=False,
    )

    base_sd = transformers.Cohere2ForCausalLM.from_pretrained(ckpt).state_dict()
    ckpt2_sd = transformers.Cohere2ForCausalLM.from_pretrained(summary["artifacts"]["checkpoint_2_global_dir"]).state_dict()
    ckpt3_sd = transformers.Cohere2ForCausalLM.from_pretrained(summary["artifacts"]["checkpoint_3_surgical_dir"]).state_dict()

    ci = load_circuit_info_from_file(str(mask_path))
    matched = set(
        _matched_param_keys(
            base_sd.keys(),
            ci.layer_suggestions.layer_subset,
            ci.layer_suggestions.target_modules,
        )
    )
    lora_projs = tuple(summary["target_modules"])
    circuit_deltas = [
        (ckpt2_sd[k] - base_sd[k]).abs().mean().item()
        for k in matched
    ]
    non_circuit_deltas = [
        (ckpt2_sd[k] - base_sd[k]).abs().mean().item()
        for k in base_sd
        if k not in matched and k.endswith(".weight") and any(f".{p}." in k for p in lora_projs)
    ]
    assert circuit_deltas and non_circuit_deltas
    circuit_mean = sum(circuit_deltas) / len(circuit_deltas)
    non_circuit_mean = sum(non_circuit_deltas) / len(non_circuit_deltas)
    assert circuit_mean > 3.0 * non_circuit_mean, (
        f"Expected circuit weight delta ({circuit_mean:.6e}) > 3x non-circuit ({non_circuit_mean:.6e})"
    )

    for k in base_sd:
        if k in matched:
            assert torch.allclose(ckpt3_sd[k], ckpt2_sd[k], atol=1e-5)
        else:
            assert torch.equal(ckpt3_sd[k], base_sd[k])


def test_stage3_model_slug_naming_mask_guard_ref_release_and_scoped_upload(tmp_path, monkeypatch) -> None:
    """Stage 3 isolates model slugs, rejects cross-model Stage 2 masks, frees ref_model before ckpt2_sd, and scopes HF uploads."""
    import stage3_repair as s3

    assert s3.resolve_artifact_names("CohereLabs/tiny-aya-fire") == (
        "tiny-aya-fire-circuit-kavach-lora",
        "tiny-aya-fire-global-safetune",
        "tiny-aya-fire-circuit-kavach-merged",
    )
    assert s3.resolve_artifact_names("CohereLabs/tiny-aya-global") == (
        "tiny-aya-global-circuit-kavach-lora",
        "tiny-aya-global-global-safetune",
        "tiny-aya-global-circuit-kavach-merged",
    )
    assert s3.resolve_artifact_names("hf:CohereLabs/tiny-aya-water@main/") == (
        "tiny-aya-water-circuit-kavach-lora",
        "tiny-aya-water-global-safetune",
        "tiny-aya-water-circuit-kavach-merged",
    )
    assert s3.resolve_artifact_names("CohereLabs/tiny-aya-earth") == (
        "tiny-aya-earth-circuit-kavach-lora",
        "tiny-aya-earth-global-safetune",
        "tiny-aya-earth-circuit-kavach-merged",
    )
    assert s3.resolve_artifact_names("gpt2") == (
        "tiny-aya-fire-circuit-kavach-lora",
        "tiny-aya-fire-global-safetune",
        "tiny-aya-fire-circuit-kavach-merged",
    )

    # Cross-model mask mismatch guard must reject loading tiny-aya-fire mask for tiny-aya-global
    ds_path = tmp_path / "calib.jsonl"
    ds_path.write_text("\n".join(json.dumps(r) for r in _sample_pairs(8)) + "\n", encoding="utf-8")
    fire_mask = _write_stage2_submodule_mask(
        tmp_path / "fire_mask.json",
        model_name="CohereLabs/tiny-aya-fire",
        selected_submodules=[(0, "mlp")],
        n_layers=2,
    )
    with pytest.raises(ValueError, match="mismatch"):
        s3.run_stage3(
            model_name="CohereLabs/tiny-aya-global",
            dataset_path=ds_path,
            circuit_info_path=fire_mask,
            output_dir=tmp_path / "mismatch_out",
            models_dir=tmp_path / "mismatch_models",
            upload_hf=False,
        )

    # End-to-end slugged run + scoped Phase A / Phase B allow_patterns + ref_model cleared before ckpt2_sd save
    ckpt = _create_tiny_cohere2_checkpoint(tmp_path / "tiny-aya-global", with_chat_template=False, n_layers=2)
    global_mask = _write_stage2_submodule_mask(
        tmp_path / "global_mask.json",
        model_name=str(ckpt),
        selected_submodules=[(0, "mlp")],
        n_layers=2,
    )
    monkeypatch.setenv("HF_TOKEN", "hf_dummy_token_do_not_leak")
    upload_calls: list[dict] = []

    class _SpyApi:
        def whoami(self):
            return {"name": "tester"}

        def create_repo(self, **kwargs):
            return None

        def upload_folder(self, **kwargs):
            upload_calls.append(dict(kwargs))
            return SimpleNamespace(commit_url=f"https://huggingface.co/datasets/test/commit/{len(upload_calls)}")

    with patch("stage3_repair.HfApi", return_value=_SpyApi()):
        summary = s3.run_stage3(
            model_name=str(ckpt),
            dataset_path=ds_path,
            circuit_info_path=global_mask,
            output_dir=tmp_path / "g_out",
            models_dir=tmp_path / "g_models",
            dtype="float32",
            device="cpu",
            max_steps=1,
            batch_size=1,
            alpha_grid=[1.0],
            max_new_tokens=6,
            upload_hf=True,
        )

    assert Path(summary["artifacts"]["lora_adapter_dir"]).name == "tiny-aya-global-circuit-kavach-lora"
    assert Path(summary["artifacts"]["checkpoint_2_global_dir"]).name == "tiny-aya-global-global-safetune"
    assert Path(summary["artifacts"]["checkpoint_3_surgical_dir"]).name == "tiny-aya-global-circuit-kavach-merged"

    model_uploads = [c for c in upload_calls if c.get("path_in_repo") == "stage3_models"]
    assert len(model_uploads) == 2
    assert set(model_uploads[0].get("allow_patterns") or []) == {
        "tiny-aya-global-circuit-kavach-lora/**",
        "tiny-aya-global-global-safetune/**",
    }
    assert set(model_uploads[1].get("allow_patterns") or []) == {
        "tiny-aya-global-circuit-kavach-merged/**",
    }


