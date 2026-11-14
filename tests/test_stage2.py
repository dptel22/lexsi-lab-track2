"""Tests for Stage 2 multilingual safety circuit discovery (src/stage2_discover.py)."""
from __future__ import annotations

import ast
import json
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import transformers
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace


ROOT = Path(__file__).resolve().parent.parent
STAGE2_PATH = ROOT / "src" / "stage2_discover.py"
STAGE1_DIR = ROOT / "artifacts" / "stage1_dataset"


# ---------------------------------------------------------------------------
# Helpers: tiny deterministic offline TL models (no network, <0.2s on CPU)
# ---------------------------------------------------------------------------

_VOCAB_WORDS = [
    "regarding", "our", "production", "web", "app", "sql", "injection",
    "generate", "attack", "script", "provide", "defensive", "patch",
    "sorry", "cannot", "assist", "here", "safe", "guide", "otp", "bypass",
    "draft", "phishing", "explain", "security", "rules", "user", "assistant",
]
_SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>", "<|USER|>", "<|ASSISTANT|>"]
_VOCAB = {w: i for i, w in enumerate(_SPECIALS + _VOCAB_WORDS)}
_CHAT_JINJA = (
    "{{ bos_token }}"
    "{% for m in messages %}<|USER|>{{ m['content'] }}{% endfor %}"
    "<|ASSISTANT|>"
)


def _build_word_tokenizer() -> Tokenizer:
    tok = Tokenizer(WordLevel(_VOCAB, unk_token="<unk>"))
    tok.pre_tokenizer = Whitespace()
    return tok


def _create_tiny_hf_checkpoint(
    dest: Path,
    *,
    arch: str = "cohere2",
    with_chat_template: bool = True,
    n_layers: int = 4,
    seed: int = 7,
) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    shape = dict(
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        vocab_size=len(_VOCAB),
    )
    torch.manual_seed(seed)
    if arch == "gpt2":
        cfg = transformers.GPT2Config(
            n_embd=32, n_layer=n_layers, n_head=4, n_positions=128, n_ctx=128, **shape
        )
        cls = transformers.GPT2LMHeadModel
    elif arch == "cohere2":
        cfg = transformers.Cohere2Config(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=n_layers,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            sliding_window=8,
            max_position_embeddings=128,
            logit_scale=1.0,
            **shape,
        )
        cls = transformers.Cohere2ForCausalLM
    else:
        raise ValueError(f"Unsupported synthetic arch: {arch}")

    cls(cfg).eval().save_pretrained(dest)
    tok = transformers.PreTrainedTokenizerFast(
        tokenizer_object=_build_word_tokenizer(),
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
    )
    if with_chat_template:
        tok.chat_template = _CHAT_JINJA
    tok.save_pretrained(dest)
    return dest


def _sample_pairs(n: int = 20) -> list[dict]:
    langs = ["en", "hi_latn", "hinglish", "bn_latn"]
    cats = ["web_api_exploitation", "credential_financial_phishing"]
    rows = []
    for i in range(n):
        lang = langs[i % len(langs)]
        cat = cats[(i // len(langs)) % len(cats)]
        rows.append(
            {
                "pair_id": f"{lang}_{i:03d}",
                "language": lang,
                "threat_category": cat,
                "keyword": "sql injection",
                "harmful": "regarding our production web app sql injection generate attack script",
                "benign": "regarding our production web app sql injection provide defensive patch",
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Cycle 1: Import order, CLI flags & runtime profile resolution
# ---------------------------------------------------------------------------


def test_import_order_circuitkit_before_transformer_lens():
    assert STAGE2_PATH.exists(), "src/stage2_discover.py must exist"
    tree = ast.parse(STAGE2_PATH.read_text(encoding="utf-8"))
    import_order = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                import_order.append(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            import_order.append(node.module.split(".")[0])

    assert "circuitkit" in import_order, "stage2_discover.py must import circuitkit"
    if "transformer_lens" in import_order:
        assert import_order.index("circuitkit") < import_order.index("transformer_lens")


def test_cli_defaults_and_escape_hatches():
    import stage2_discover as s2

    parser = s2.build_arg_parser()
    defaults = parser.parse_args([])
    assert defaults.model == "CohereLabs/tiny-aya-fire"
    assert defaults.chat_template_mode == "auto"
    assert defaults.top_fraction == pytest.approx(0.10)
    assert defaults.max_surgical_layers == 8
    assert defaults.max_surgical_submodules == 4
    assert defaults.margin_threshold == pytest.approx(0.5)
    assert defaults.upload_hf is False
    assert defaults.hf_repo == "soapyyapps/circuit-kavach-cyber-codemixed"
    assert defaults.batch_size is None
    assert defaults.max_seq_len is None
    assert str(defaults.dataset).endswith("calibration_100_pairs.jsonl")
    assert str(defaults.output_dir).endswith("stage2_circuits")

    custom = parser.parse_args(
        [
            "--model", "gpt2",
            "--chat-template-mode", "off",
            "--batch-size", "2",
            "--max-seq-len", "128",
            "--top-fraction", "0.15",
            "--max-surgical-layers", "6",
            "--max-surgical-submodules", "9",
            "--margin-threshold", "0.25",
            "--upload-hf",
            "--hf-repo", "custom/repo",
        ]
    )
    assert custom.model == "gpt2"
    assert custom.chat_template_mode == "off"
    assert custom.batch_size == 2
    assert custom.max_seq_len == 128
    assert custom.top_fraction == pytest.approx(0.15)
    assert custom.max_surgical_layers == 6
    assert custom.max_surgical_submodules == 9
    assert custom.margin_threshold == pytest.approx(0.25)
    assert custom.upload_hf is True
    assert custom.hf_repo == "custom/repo"


def test_runtime_profile_resolution_auto_instruct_vs_raw():
    import stage2_discover as s2

    instruct_model = SimpleNamespace(
        tokenizer=SimpleNamespace(chat_template=_CHAT_JINJA),
        cfg=SimpleNamespace(model_name="CohereLabs/tiny-aya-fire", n_ctx=8192, device="cuda"),
    )
    raw_model = SimpleNamespace(
        tokenizer=SimpleNamespace(chat_template=None),
        cfg=SimpleNamespace(model_name="gpt2", n_ctx=1024, device="cpu"),
    )

    p_aya = s2.resolve_runtime_profile(
        instruct_model,
        model_name="CohereLabs/tiny-aya-fire",
        chat_template_mode="auto",
        batch_size=None,
        max_seq_len=None,
        dtype="auto",
        device="cuda",
    )
    assert p_aya.templated is True
    assert p_aya.prepend_bos is False
    assert p_aya.max_seq_len == 512
    assert p_aya.batch_size == 1
    assert p_aya.dtype == "bfloat16"

    p_gpt = s2.resolve_runtime_profile(
        raw_model,
        model_name="gpt2",
        chat_template_mode="auto",
        batch_size=None,
        max_seq_len=None,
        dtype="auto",
        device="cpu",
    )
    assert p_gpt.templated is False
    assert p_gpt.prepend_bos is True
    assert p_gpt.max_seq_len == 128
    assert p_gpt.batch_size == 1
    assert p_gpt.dtype == "float32"

    # Escape hatch: force off on instruct model
    p_forced_off = s2.resolve_runtime_profile(
        instruct_model,
        model_name="CohereLabs/tiny-aya-fire",
        chat_template_mode="off",
        batch_size=4,
        max_seq_len=256,
        dtype="float16",
        device="cuda",
    )
    assert p_forced_off.templated is False
    assert p_forced_off.prepend_bos is True
    assert p_forced_off.batch_size == 4
    assert p_forced_off.max_seq_len == 256
    assert p_forced_off.dtype == "float16"

    # Escape hatch: forcing "on" without a tokenizer chat_template fails loudly
    with pytest.raises(ValueError):
        s2.resolve_runtime_profile(
            raw_model,
            model_name="gpt2",
            chat_template_mode="on",
            batch_size=None,
            max_seq_len=None,
            dtype="auto",
            device="cpu",
        )


# ---------------------------------------------------------------------------
# Cycle 2: Data loading, stratified 16-pair selection & preflight calibration
# ---------------------------------------------------------------------------


def test_load_contrast_pairs_validation_edge_cases(tmp_path):
    import stage2_discover as s2

    with pytest.raises(FileNotFoundError):
        s2.load_contrast_pairs(tmp_path / "missing.jsonl")

    empty_file = tmp_path / "empty.jsonl"
    empty_file.write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError, match="empty|no pairs"):
        s2.load_contrast_pairs(empty_file)

    bad_row = tmp_path / "bad.jsonl"
    bad_row.write_text(json.dumps({"pair_id": "1", "harmful": "x"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="benign"):
        s2.load_contrast_pairs(bad_row)

    rows = _sample_pairs(5)
    ok_file = tmp_path / "ok.jsonl"
    ok_file.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    loaded = s2.load_contrast_pairs(ok_file)
    assert len(loaded) == 5
    assert loaded[0]["harmful"] == rows[0]["harmful"]


def test_select_preflight_pairs_stratified_and_bounded():
    import stage2_discover as s2

    pairs = _sample_pairs(24)
    chosen = s2.select_preflight_pairs(pairs, limit=16)
    assert len(chosen) == 16
    assert len({p["language"] for p in chosen}) == 4
    assert len({p["threat_category"] for p in chosen}) == 2
    assert len({p["pair_id"] for p in chosen}) == 16

    small = _sample_pairs(4)
    assert len(s2.select_preflight_pairs(small, limit=16)) == 4

    with pytest.raises(ValueError):
        s2.select_preflight_pairs([], limit=16)


def test_preflight_token_calibration_and_no_double_bos(tmp_path):
    import circuitkit as ck
    import stage2_discover as s2

    ckpt = _create_tiny_hf_checkpoint(tmp_path / "tiny_cohere2", arch="cohere2", with_chat_template=True)
    model = ck.load_model(str(ckpt), dtype="float32", device="cpu")
    pairs = _sample_pairs(16)

    cal = s2.calibrate_answer_tokens(model, pairs, templated=True, max_seq_len=128)
    assert cal.clean_token_id != cal.corrupt_token_id
    assert cal.clean_token_id not in set(model.tokenizer.all_special_ids)
    assert cal.corrupt_token_id not in set(model.tokenizer.all_special_ids)
    assert cal.clean_token.strip() != ""
    assert cal.corrupt_token.strip() != ""
    assert cal.logit_diff_gap > 1e-6
    assert cal.n_pairs_used == 16
    assert cal.templated is True
    assert cal.prepend_bos is False

    # Verify wrapped clean/corrupt prompts tokenized with prepend_bos=False have AT MOST 1 BOS
    bos_id = model.tokenizer.bos_token_id
    wrapped = ck.tasks._chat.wrap_prompt(model, pairs[0]["harmful"], apply=True)
    ids = model.to_tokens(wrapped, prepend_bos=False)[0].tolist()
    assert ids.count(bos_id) == 1, f"Expected single BOS token, got {ids.count(bos_id)} in {ids}"


# ---------------------------------------------------------------------------
# Cycle 3: Task alignment/templating, 4-pillar evaluation & KL diagnostic
# ---------------------------------------------------------------------------


def test_calibrated_task_spec_preserves_align_none_left_pad_and_templated(tmp_path):
    import circuitkit as ck
    import stage2_discover as s2
    from circuitkit.tasks.registry import get_task

    ckpt = _create_tiny_hf_checkpoint(tmp_path / "tiny_cohere2", arch="cohere2", with_chat_template=True)
    model = ck.load_model(str(ckpt), dtype="float32", device="cpu")
    pairs = _sample_pairs(8)
    cal = s2.calibrate_answer_tokens(model, pairs, templated=True, max_seq_len=128)

    task_name = "test_stage2_calibrated_cohere2"
    task_spec = s2.register_calibrated_task(
        pairs=pairs,
        calibration=cal,
        model=model,
        templated=True,
        max_seq_len=128,
        task_name=task_name,
        cache_dir=str(tmp_path / "norm_cache"),
    )
    registered = get_task(task_name)
    assert registered is task_spec
    assert task_spec.pair_padding_side == "left"
    assert task_spec.ds.meta["_alignment"]["align_strategy"] == "none"

    dl = task_spec.build_dataloader(model, {"batch_size": 2, "num_examples": 8, "max_seq_len": 128}, "cpu")
    assert getattr(dl, "templated", None) is True
    assert getattr(dl, "pair_padding_side", None) == "left"
    clean_batch, corrupt_batch, labels = next(iter(dl))
    assert "<|USER|>" in clean_batch[0] and "<|ASSISTANT|>" in clean_batch[0]
    assert "<|USER|>" in corrupt_batch[0] and "<|ASSISTANT|>" in corrupt_batch[0]
    bos_id = model.tokenizer.bos_token_id
    toks = model.to_tokens(clean_batch[0], prepend_bos=not dl.templated)[0].tolist()
    assert toks.count(bos_id) == 1
    assert labels.tolist()[0] == [cal.clean_token_id, cal.corrupt_token_id]


def test_end_to_end_stage2_on_tiny_cohere2_model(tmp_path):
    import stage2_discover as s2
    from safetune.core.circuit_kit.adapter import load_circuit_info_from_file

    ckpt = _create_tiny_hf_checkpoint(tmp_path / "tiny_cohere2", arch="cohere2", with_chat_template=True, n_layers=4)
    stage1_dir = tmp_path / "stage1_dataset"
    stage1_dir.mkdir(parents=True)
    data_path = stage1_dir / "calibration_100_pairs.jsonl"
    data_path.write_text("\n".join(json.dumps(r) for r in _sample_pairs(8)) + "\n", encoding="utf-8")
    stage1_prov = {
        "schema": "lexsi.provenance/1",
        "library": "curatorkit",
        "version": "0.1.21",
        "base_model": "CohereLabs/tiny-aya-fire",
        "method": "contrastive_lexical_twin_curation",
        "inputs": [],
        "params": {"n_pairs": 100},
    }
    (stage1_dir / "lexsi_provenance.json").write_text(json.dumps(stage1_prov), encoding="utf-8")

    out_dir = tmp_path / "stage2_circuits"
    summary = s2.run_stage2(
        model_name=str(ckpt),
        dataset_path=data_path,
        output_dir=out_dir,
        chat_template_mode="auto",
        batch_size=2,
        max_seq_len=128,
        dtype="float32",
        device="cpu",
        top_fraction=0.10,
        max_surgical_layers=2,
        n_examples=8,
        n_stability_runs=2,
        ig_steps=2,
        upload_hf=False,
        stem="test_circuit",
    )

    assert summary["runtime_profile"]["templated"] is True
    assert summary["runtime_profile"]["prepend_bos"] is False
    assert summary["eval_pillars"] == ["patching", "ablation", "baselines", "stability"]
    assert "model_revision" in summary
    pillars = summary["faithfulness_report"]["pillars"]
    assert set(pillars.keys()) == {"patching", "ablation", "baselines", "stability"}
    assert "robustness" not in pillars
    assert isinstance(pillars["patching"], (int, float)) and math.isfinite(float(pillars["patching"]))
    assert isinstance(pillars["ablation"], (int, float)) and math.isfinite(float(pillars["ablation"]))
    assert isinstance(pillars["baselines"], dict) and len(pillars["baselines"]) > 0
    assert isinstance(pillars["stability"], dict) and len(pillars["stability"]) > 0
    assert summary["kl_diagnostic"]["kl_circuit_vs_clean"] >= 0.0
    assert summary["kl_diagnostic"]["kl_corrupt_vs_clean"] >= 0.0

    scores_json = out_dir / "test_circuit_scores.json"
    assert scores_json.exists()
    blob = json.loads(scores_json.read_text(encoding="utf-8"))
    # 4 layers * (4 attn heads + 1 MLP) = 20 nodes total
    assert len(blob["node_scores"]) == 20
    assert len(blob["layer_saturation_table"]) == 4  # tiers 0.05, 0.10, 0.15, 0.20
    assert [r["label"] for r in blob["submodule_diagnostic_table"]] == [
        "0.05", "0.10", "0.20", "submodule-top-3", "submodule-top-4", "submodule-top-6"
    ]
    # Standard <stem>_scores.json remains SafeTune-compatible standard interop_fields
    ci_std = load_circuit_info_from_file(str(scores_json))
    assert ci_std.safety_units is not None
    assert ci_std.layer_suggestions is not None

    # Sibling submodule surgical mask artifact exists and passes SafeTune adapter
    sub_scores_json = out_dir / "test_circuit_surgical_submodules_scores.json"
    assert sub_scores_json.exists()
    ci_sub = load_circuit_info_from_file(str(sub_scores_json))
    assert ci_sub.layer_suggestions is not None
    assert len(ci_sub.layer_suggestions.target_modules) > 0
    assert "submodule_tier_exports" in summary["artifacts"]
    for budget in (3, 4, 6):
        sub_tier_path = Path(summary["artifacts"]["submodule_tier_exports"][str(budget)])
        assert sub_tier_path.name == f"test_circuit_surgical_submodules_top{budget}_scores.json"
        assert sub_tier_path.exists()
        ci_budget = load_circuit_info_from_file(str(sub_tier_path))
        assert ci_budget.safety_units is not None
        assert ci_budget.layer_suggestions is not None

    # Tier exports + SafeTune strict validation
    for suffix in ("top05", "top10", "top20"):
        tier_path = out_dir / f"safety_units_{suffix}.json"
        assert tier_path.exists()
        ci = load_circuit_info_from_file(str(tier_path))
        assert ci.safety_units is not None
        assert ci.layer_suggestions is not None
        assert len(ci.layer_suggestions.layer_subset) <= 2

    # Provenance chain links Stage 1 record
    prov_path = out_dir / "lexsi_provenance.json"
    assert prov_path.exists()
    prov = json.loads(prov_path.read_text(encoding="utf-8"))
    assert prov["schema"] == "lexsi.provenance/1"
    ds_inputs = [x for x in prov["inputs"] if x["kind"] == "dataset"]
    assert len(ds_inputs) == 1
    assert ds_inputs[0]["provenance"] == stage1_prov


# ---------------------------------------------------------------------------
# Cycle 4: Independent Layer Saturation Arithmetic & Surgical Cap Validation
# ---------------------------------------------------------------------------


def test_layer_saturation_table_and_surgical_cap_math_for_36_layer_aya(tmp_path):
    import stage2_discover as s2
    from safetune.core.circuit_kit.adapter import load_circuit_info_from_file
    from safetune.interventions.recover.ctheta import _matched_param_keys

    # Construct an exact 612-node Aya score dictionary (36 layers x 16 heads + 36 MLPs)
    # Give layers 10..27 (18 layers) at least 1 strong node so raw top-10% (61 nodes) hits 18 layers,
    # while layers 0..7 have highest cumulative head+MLP attribution mass!
    node_scores: dict[str, float] = {}
    for layer in range(36):
        for head in range(16):
            node_scores[f"A{layer}.{head}"] = 0.01
        node_scores[f"MLP {layer}"] = 0.01

    # Top 8 layers (0..7): give each layer 4 heads @ 1.0 + MLP @ 1.0 => cumulative = 5.0/layer (40 nodes total)
    for layer in range(8):
        for head in range(4):
            node_scores[f"A{layer}.{head}"] = 1.0
        node_scores[f"MLP {layer}"] = 1.0

    # Spread remaining 21 top-10% slots (61 - 40 = 21) across layers 10..30 with lower score 0.5
    for layer in range(10, 31):
        node_scores[f"A{layer}.0"] = 0.5

    assert len(node_scores) == 612

    table = s2.compute_layer_saturation_table(
        node_scores=node_scores,
        model_name="CohereLabs/tiny-aya-fire",
        n_layers=36,
        tiers=(0.05, 0.10, 0.15, 0.20),
        max_surgical_layers=8,
    )
    assert [row["top_fraction"] for row in table] == [0.05, 0.10, 0.15, 0.20]

    row10 = next(r for r in table if r["top_fraction"] == 0.10)
    assert row10["raw_node_count"] == 61  # round/int(612 * 0.10) == 61
    # Uncapped touches layers 0..7 + 10..30 = 29 unique layers -> 58 / 72 matrices
    assert row10["raw_layers_touched"] == 29
    assert row10["total_layers"] == 36
    assert row10["raw_matrices_touched"] == 58
    assert row10["total_matrices"] == 72
    assert row10["raw_matrices_pct"] == pytest.approx(100.0 * 58 / 72)

    # Capped preserves exact top 8 layers (0..7) by cumulative score -> 16/72 = 22.22% -> 77.78% frozen
    assert row10["capped_layers_touched"] == 8
    assert row10["capped_matrices_touched"] == 16
    assert row10["capped_matrices_pct"] == pytest.approx(100.0 * 16 / 72)
    assert row10["capped_frozen_pct"] == pytest.approx(100.0 - 100.0 * 16 / 72)
    assert row10["capped_frozen_pct"] >= 75.0

    # Export capped tier JSONs and verify SafeTune apply_ctheta mask stays >= 75% frozen
    exported = s2.export_surgical_tiers(
        node_scores=node_scores,
        model_name="CohereLabs/tiny-aya-fire",
        n_layers=36,
        output_dir=tmp_path,
        tiers=(0.05, 0.10, 0.20),
        max_surgical_layers=8,
        provenance={"schema": "lexsi.provenance/1", "method": "discover.eap-ig"},
    )
    assert set(exported.keys()) == {0.05, 0.10, 0.20}

    # Build a synthetic 36-layer Aya state_dict key list (36 * 7 = 252 projection weights)
    aya_sd_keys = []
    for layer in range(36):
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            aya_sd_keys.append(f"model.layers.{layer}.self_attn.{proj}.weight")
        for proj in ("gate_proj", "up_proj", "down_proj"):
            aya_sd_keys.append(f"model.layers.{layer}.mlp.{proj}.weight")

    ci_10 = load_circuit_info_from_file(str(exported[0.10]))
    assert ci_10.layer_suggestions.layer_subset == list(range(8))
    matched = _matched_param_keys(
        aya_sd_keys,
        ci_10.layer_suggestions.layer_subset,
        ci_10.layer_suggestions.target_modules,
    )
    frozen_ratio = 1.0 - (len(matched) / len(aya_sd_keys))
    assert frozen_ratio >= 0.75, f"Expected >=75% matrices frozen, got {frozen_ratio:.2%}"


# ---------------------------------------------------------------------------
# Cycle 5: Hugging Face Upload Contract (mocked HfApi)
# ---------------------------------------------------------------------------


def test_upload_stage2_to_hf_contract(tmp_path, monkeypatch):
    import stage2_discover as s2

    (tmp_path / "safety_circuit_scores.json").write_text("{}", encoding="utf-8")
    (tmp_path / "lexsi_provenance.json").write_text("{}", encoding="utf-8")

    monkeypatch.delenv("HF_TOKEN", raising=False)
    with patch("stage2_discover.load_dotenv"):
        with pytest.raises(RuntimeError, match="HF_TOKEN"):
            s2.upload_stage2_to_hf(tmp_path, repo_id="soapyyapps/circuit-kavach-cyber-codemixed")

    monkeypatch.setenv("HF_TOKEN", "hf_dummy_token_do_not_leak")
    mock_api = MagicMock()
    mock_api.upload_folder.return_value = SimpleNamespace(commit_url="https://huggingface.co/datasets/x/commit/1")

    with patch("stage2_discover.HfApi", return_value=mock_api):
        url = s2.upload_stage2_to_hf(tmp_path, repo_id="soapyyapps/circuit-kavach-cyber-codemixed")

    mock_api.upload_folder.assert_called_once()
    call_kw = mock_api.upload_folder.call_args.kwargs
    assert call_kw["repo_id"] == "soapyyapps/circuit-kavach-cyber-codemixed"
    assert call_kw["path_in_repo"] == "stage2_circuits"
    assert call_kw["repo_type"] == "dataset"
    assert "stage2_circuits" in url or "huggingface.co" in url


# ---------------------------------------------------------------------------
# Cycle 6: Real GPT-2 CPU Integration Test on Stage 1 calibration_100_pairs.jsonl
# ---------------------------------------------------------------------------


def test_real_gpt2_cpu_stage2_with_stage1_pairs(tmp_path):
    """Real CPU end-to-end verification using cached 'gpt2' (dtype=float32) & Stage 1 subset."""
    import stage2_discover as s2
    from safetune.core.circuit_kit.adapter import load_circuit_info_from_file
    from safetune.interventions.recover.ctheta import _matched_param_keys

    stage1_cal = STAGE1_DIR / "calibration_100_pairs.jsonl"
    assert stage1_cal.exists(), "Stage 1 calibration_100_pairs.jsonl must exist"

    # Use first 8 Stage 1 pairs so real GPT-2 CPU EAP-IG + 4-pillar runs fast (~15s)
    subset_path = tmp_path / "stage1_subset_8.jsonl"
    lines = stage1_cal.read_text(encoding="utf-8").splitlines()[:8]
    subset_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Copy Stage 1 provenance alongside subset to verify lineage chaining
    stage1_prov_path = STAGE1_DIR / "lexsi_provenance.json"
    if stage1_prov_path.exists():
        (tmp_path / "lexsi_provenance.json").write_text(
            stage1_prov_path.read_text(encoding="utf-8"), encoding="utf-8"
        )

    out_dir = tmp_path / "gpt2_stage2_out"
    summary = s2.run_stage2(
        model_name="gpt2",
        dataset_path=subset_path,
        output_dir=out_dir,
        chat_template_mode="auto",
        batch_size=2,
        max_seq_len=128,
        dtype="float32",
        device="cpu",
        top_fraction=0.10,
        max_surgical_layers=3,
        max_surgical_submodules=4,
        n_examples=8,
        n_stability_runs=2,
        ig_steps=2,
        upload_hf=False,
        stem="gpt2_circuit",
    )

    assert summary["runtime_profile"]["templated"] is False
    assert summary["runtime_profile"]["prepend_bos"] is True
    assert summary["runtime_profile"]["max_seq_len"] == 128
    assert summary["calibration"]["logit_diff_gap"] > 0.0
    assert summary["eval_pillars"] == ["patching", "ablation", "baselines", "stability"]
    pillars = summary["faithfulness_report"]["pillars"]
    assert set(pillars.keys()) == {"patching", "ablation", "baselines", "stability"}
    assert "robustness" not in pillars
    assert isinstance(pillars["patching"], (int, float)) and math.isfinite(float(pillars["patching"]))
    assert isinstance(pillars["ablation"], (int, float)) and math.isfinite(float(pillars["ablation"]))
    assert isinstance(pillars["baselines"], dict) and len(pillars["baselines"]) > 0
    assert isinstance(pillars["stability"], dict) and len(pillars["stability"]) > 0
    assert "margin_filter" in summary
    assert "positive_margin_count_by_language" in summary["margin_filter"]

    blob = json.loads((out_dir / "gpt2_circuit_scores.json").read_text(encoding="utf-8"))
    # GPT-2 has 12 layers x (12 heads + 1 MLP) = 156 nodes
    assert len(blob["node_scores"]) == 156
    assert any(abs(float(v)) > 0.0 for v in blob["node_scores"].values())

    ci = load_circuit_info_from_file(str(out_dir / "safety_units_top10.json"))
    assert len(ci.layer_suggestions.layer_subset) <= 3

    # Verify GPT-2 <stem>_surgical_submodules_scores.json masks only <=4/24 submodule blocks
    sub_blob_path = out_dir / "gpt2_circuit_surgical_submodules_scores.json"
    assert sub_blob_path.exists()
    ci_sub = load_circuit_info_from_file(str(sub_blob_path))
    gpt2_sd_keys = []
    for layer in range(12):
        gpt2_sd_keys.append(f"transformer.h.{layer}.attn.c_attn.weight")
        gpt2_sd_keys.append(f"transformer.h.{layer}.attn.c_proj.weight")
        gpt2_sd_keys.append(f"transformer.h.{layer}.mlp.c_fc.weight")
        gpt2_sd_keys.append(f"transformer.h.{layer}.mlp.c_proj.weight")
    matched = _matched_param_keys(
        gpt2_sd_keys,
        ci_sub.layer_suggestions.layer_subset,
        ci_sub.layer_suggestions.target_modules,
    )
    assert len(matched) == 8  # 4 submodules x 2 weight matrices each out of 48 total


def test_cli_main_entrypoint_and_edge_case_rejections(tmp_path):
    import stage2_discover as s2

    # Malformed JSONL raises ValueError with location context
    malformed = tmp_path / "malformed.jsonl"
    malformed.write_text("{not valid json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Malformed JSON"):
        s2.load_contrast_pairs(malformed)

    # Boundary rejections in run_stage2
    valid_ds = tmp_path / "pairs.jsonl"
    valid_ds.write_text("\n".join(json.dumps(r) for r in _sample_pairs(8)) + "\n", encoding="utf-8")
    for bad_kw in (
        {"top_fraction": 0.0},
        {"top_fraction": 1.2},
        {"max_surgical_layers": -1},
        {"max_surgical_submodules": 0},
        {"margin_threshold": -0.1},
        {"margin_threshold": float("nan")},
        {"n_examples": 0},
        {"n_stability_runs": 0},
        {"ig_steps": 0},
    ):
        with pytest.raises(ValueError):
            s2.run_stage2(dataset_path=valid_ds, output_dir=tmp_path / "err_out", **bad_kw)

    # --chat-template-mode on without chat_template raises ValueError
    raw_model = SimpleNamespace(
        tokenizer=SimpleNamespace(chat_template=None),
        cfg=SimpleNamespace(model_name="gpt2", n_ctx=1024, device="cpu"),
    )
    with pytest.raises(ValueError, match="chat_template"):
        s2.resolve_runtime_profile(
            raw_model,
            model_name="gpt2",
            chat_template_mode="on",
            batch_size=None,
            max_seq_len=None,
            dtype="float32",
            device="cpu",
        )

    # End-to-end CLI main() execution on tiny cohere2 with --chat-template-mode off escape hatch
    ckpt = _create_tiny_hf_checkpoint(tmp_path / "tiny_cohere2_cli", arch="cohere2", with_chat_template=True, n_layers=4)
    out_dir = tmp_path / "cli_out"
    rc = s2.main(
        [
            "--model", str(ckpt),
            "--dataset", str(valid_ds),
            "--output-dir", str(out_dir),
            "--chat-template-mode", "off",
            "--batch-size", "2",
            "--max-seq-len", "64",
            "--dtype", "float32",
            "--device", "cpu",
            "--top-fraction", "0.10",
            "--max-surgical-layers", "2",
            "--n-examples", "6",
            "--n-stability-runs", "2",
            "--ig-steps", "2",
            "--stem", "cli_circuit",
        ]
    )
    assert rc == 0
    report = json.loads((out_dir / "faithfulness_report.json").read_text(encoding="utf-8"))
    assert report["runtime_profile"]["templated"] is False
    assert report["runtime_profile"]["prepend_bos"] is True
    assert (out_dir / "cli_circuit_scores.json").exists()
    assert (out_dir / "cli_circuit_surgical_submodules_scores.json").exists()
    assert (out_dir / "lexsi_provenance.json").exists()
    assert (out_dir / "safety_units_top10.json").exists()


# ---------------------------------------------------------------------------
# Adversarial Fixes #1–#4: Index Alignment, Margin Filter, Submodules, BOS
# ---------------------------------------------------------------------------


def test_within_pair_length_equalization_prevents_left_pad_preamble_shift(tmp_path):
    import circuitkit as ck
    import stage2_discover as s2
    from circuitkit.backends.eap.eap_utils import tokenize_batch_pair

    ckpt = _create_tiny_hf_checkpoint(tmp_path / "tiny_cohere2_long_sys", arch="cohere2", with_chat_template=True)
    # Give chat_template an explicit multi-token system preamble so left-pad shifts are immediately visible
    tok = transformers.AutoTokenizer.from_pretrained(ckpt)
    tok.chat_template = (
        "{{ bos_token }}"
        "<|USER|> security rules explain production web app "
        "{% for m in messages %}{{ m['content'] }}{% endfor %}"
        "<|ASSISTANT|>"
    )
    tok.save_pretrained(ckpt)

    model = ck.load_model(str(ckpt), dtype="float32", device="cpu")
    # Unequal suffixes after shared prefix "regarding our production web app sql injection"
    unequal_pairs = [
        {
            "pair_id": f"p_{i}",
            "language": "en" if i % 2 == 0 else "hi_latn",
            "threat_category": "web_api_exploitation",
            "keyword": "sql injection",
            "harmful": "regarding our production web app sql injection generate attack",
            "benign": "regarding our production web app sql injection provide defensive safe guide rules explain patch",
        }
        for i in range(4)
    ]

    eq_pairs, eq_stats = s2.equalize_contrast_pairs(model, unequal_pairs, templated=True, max_seq_len=128)
    assert eq_stats["pairs_equalized"] == 4
    cal = s2.calibrate_answer_tokens(model, eq_pairs, templated=True, max_seq_len=128)

    task_spec = s2.register_calibrated_task(
        pairs=eq_pairs,
        calibration=cal,
        model=model,
        templated=True,
        max_seq_len=128,
        task_name="test_fix1_equalized_cohere2",
        cache_dir=str(tmp_path / "fix1_cache"),
    )
    dl = task_spec.build_dataloader(model, {"batch_size": 2, "num_examples": 4, "max_seq_len": 128}, "cpu")
    clean_batch, corrupt_batch, _labels = next(iter(dl))
    clean_tokens, corr_tokens, clean_mask, corr_mask, clean_lengths, _ = tokenize_batch_pair(
        model,
        clean_batch,
        corrupt_batch,
        pair_padding_side=dl.pair_padding_side,
        templated=dl.templated,
    )
    corr_lengths = corr_mask.sum(dim=1)
    real_clean_lengths = clean_mask.sum(dim=1)

    # Every pair within the batch has identical non-pad token length and aligned system+shared prefix
    assert torch.equal(real_clean_lengths, corr_lengths)
    assert int(clean_lengths[0].item()) == int(corr_lengths[0].item())
    # System preamble + shared user prefix spans at least 12 tokens from index 0
    prefix_probe = ck.tasks._chat.wrap_prompt(
        model, "regarding our production web app sql injection", apply=True
    )
    # Strip closing <|ASSISTANT|> tail from probe to measure exact shared prefix length
    prefix_ids = model.to_tokens(prefix_probe.split("<|ASSISTANT|>")[0], prepend_bos=False)[0]
    prefix_len = len(prefix_ids)
    assert prefix_len >= 10
    for i in range(len(clean_batch)):
        assert torch.equal(clean_tokens[i, :prefix_len], corr_tokens[i, :prefix_len]), (
            f"Left-pad shifted shared preamble/prefix for sample {i}: "
            f"{clean_tokens[i, :prefix_len].tolist()} != {corr_tokens[i, :prefix_len].tolist()}"
        )


def test_within_pair_length_equalization_on_real_gpt2_stage1_unequal_pairs(tmp_path):
    import circuitkit as ck
    import stage2_discover as s2
    from circuitkit.backends.eap.eap_utils import tokenize_batch_pair

    stage1_cal = STAGE1_DIR / "calibration_100_pairs.jsonl"
    all_pairs = s2.load_contrast_pairs(stage1_cal)
    model = ck.load_model("gpt2", dtype="float32", device="cpu")
    # Pick pairs with confirmed unequal GPT-2 lengths (e.g. hi_en / gu_en pairs after index 8)
    unequal_subset = [
        p for p in all_pairs
        if len(model.tokenizer.encode(p["harmful"], add_special_tokens=False))
        != len(model.tokenizer.encode(p["benign"], add_special_tokens=False))
    ][:4]
    assert len(unequal_subset) == 4

    eq_pairs, _ = s2.equalize_contrast_pairs(model, unequal_subset, templated=False, max_seq_len=128)
    cal = s2.calibrate_answer_tokens(model, eq_pairs, templated=False, max_seq_len=128)
    task_spec = s2.register_calibrated_task(
        pairs=eq_pairs,
        calibration=cal,
        model=model,
        templated=False,
        max_seq_len=128,
        task_name="test_fix1_equalized_gpt2",
        cache_dir=str(tmp_path / "fix1_gpt2_cache"),
    )
    dl = task_spec.build_dataloader(model, {"batch_size": 1, "num_examples": 4, "max_seq_len": 128}, "cpu")
    for clean_batch, corrupt_batch, _labels in dl:
        clean_tokens, corr_tokens, clean_mask, corr_mask, _, _ = tokenize_batch_pair(
            model, clean_batch, corrupt_batch, pair_padding_side=dl.pair_padding_side, templated=dl.templated
        )
        assert torch.equal(clean_mask.sum(dim=1), corr_mask.sum(dim=1))
        # Determine common prefix length directly from the yielded batch strings (dataset shuffles order)
        h, b = clean_batch[0], corrupt_batch[0]
        common_chars = 0
        for ca, cb in zip(h, b):
            if ca != cb:
                break
            common_chars += 1
        shared_text = h[:common_chars].rstrip()
        shared_ids = model.to_tokens(shared_text, prepend_bos=True)[0]
        # Exclude boundary final token if GPT-2 BPE merges with trailing whitespace/first divergent char
        safe_prefix_len = max(2, len(shared_ids) - 1)
        assert torch.equal(clean_tokens[0, :safe_prefix_len], corr_tokens[0, :safe_prefix_len])


def test_normalized_task_spec_no_double_bos_on_tokenized_batches(tmp_path):
    import circuitkit as ck
    import stage2_discover as s2
    from circuitkit.backends.eap.eap_utils import tokenize_batch_pair

    ckpt = _create_tiny_hf_checkpoint(tmp_path / "tiny_cohere2_bos", arch="cohere2", with_chat_template=True)
    model = ck.load_model(str(ckpt), dtype="float32", device="cpu")
    pairs = _sample_pairs(4)
    pairs[0]["benign"] = pairs[0]["benign"] + " extra safe rules"
    eq_pairs, _ = s2.equalize_contrast_pairs(model, pairs, templated=True, max_seq_len=128)
    cal = s2.calibrate_answer_tokens(model, eq_pairs, templated=True, max_seq_len=128)
    task_spec = s2.register_calibrated_task(
        pairs=eq_pairs,
        calibration=cal,
        model=model,
        templated=True,
        max_seq_len=128,
        task_name="test_fix4_no_double_bos",
        cache_dir=str(tmp_path / "fix4_cache"),
    )
    bos_id = model.tokenizer.bos_token_id
    assert bos_id is not None

    for bs in (1, 2):
        dl = task_spec.build_dataloader(model, {"batch_size": bs, "num_examples": 4, "max_seq_len": 128}, "cpu")
        for clean_batch, corrupt_batch, _ in dl:
            clean_tokens, corr_tokens, _, _, _, _ = tokenize_batch_pair(
                model, clean_batch, corrupt_batch, pair_padding_side=dl.pair_padding_side, templated=dl.templated
            )
            for tensor in (clean_tokens, corr_tokens):
                for row in tensor:
                    ids = row.tolist()
                    if ids[0] == bos_id:
                        assert ids[1] != bos_id, f"Double BOS detected at start of sequence: {ids}"
                    assert ids.count(bos_id) <= 1, f"Multiple BOS tokens detected: {ids}"


def test_positive_baseline_margin_filter_and_fallback(tmp_path):
    import stage2_discover as s2

    # Construct deterministic mock model + tokenizer where first 3 pairs pass (>0.5) and 3 pairs fail (<=0.5)
    vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "Sorry": 3, "Here": 4, "good": 5, "bad": 6, "x": 7}

    class _FakeTok:
        pad_token_id = 0
        bos_token_id = 1
        eos_token_id = 2
        pad_token = "<pad>"
        chat_template = None
        all_special_ids = [0, 1, 2]
        name_or_path = "fake"

        def encode(self, text, add_special_tokens=False):
            return [vocab.get(t, 7) for t in str(text).split()]

        def decode(self, ids, skip_special_tokens=False):
            inv = {v: k for k, v in vocab.items()}
            return " ".join(inv.get(int(i), "x") for i in ids)

        def __len__(self):
            return len(vocab)

    class _FakeModel:
        tokenizer = _FakeTok()
        cfg = SimpleNamespace(d_vocab=len(vocab), device="cpu", model_name="gpt2", n_layers=2)

        def to_tokens(self, text, prepend_bos=True):
            ids = self.tokenizer.encode(text)
            if prepend_bos:
                ids = [1] + ids
            return torch.tensor([ids], dtype=torch.long)

        def __call__(self, tokens):
            # Token id 5 ("good") yields strong clean refusal margin (+2.0), token 6 ("bad") negative (-1.0)
            seq = tokens[0].tolist()
            logits = torch.zeros((1, len(seq), len(vocab)), dtype=torch.float32)
            if 5 in seq:
                logits[0, -1, 3] = 2.0  # Sorry
                logits[0, -1, 4] = -1.0  # Here -> m_clean = +3.0
            elif 6 in seq:
                logits[0, -1, 3] = -1.0
                logits[0, -1, 4] = 0.0  # -> m_clean = -1.0
            else:
                # corrupt baseline
                logits[0, -1, 3] = 0.0
                logits[0, -1, 4] = 0.5  # m_corrupt = -0.5 -> good delta_m=3.5, bad delta_m=-0.5
            return logits

    pairs = [
        {"pair_id": "en_0", "language": "en", "threat_category": "t", "keyword": "k", "harmful": "good en a", "benign": "x en a"},
        {"pair_id": "hi_0", "language": "hi_en", "threat_category": "t", "keyword": "k", "harmful": "good hi a", "benign": "x hi a"},
        {"pair_id": "gu_0", "language": "gu_en", "threat_category": "t", "keyword": "k", "harmful": "good gu a", "benign": "x gu a"},
        {"pair_id": "en_fail", "language": "en", "threat_category": "t", "keyword": "k", "harmful": "bad en b", "benign": "x en b"},
        {"pair_id": "kn_fail", "language": "kn_en", "threat_category": "t", "keyword": "k", "harmful": "bad kn b", "benign": "x kn b"},
        {"pair_id": "hi_fail", "language": "hi_en", "threat_category": "t", "keyword": "k", "harmful": "bad hi b", "benign": "x hi b"},
    ]
    model = _FakeModel()
    cal = s2.CalibrationResult(
        clean_token="Sorry",
        clean_token_id=3,
        corrupt_token="Here",
        corrupt_token_id=4,
        mean_clean_margin=1.0,
        mean_corrupt_margin=-0.5,
        logit_diff_gap=1.5,
        n_pairs_used=6,
        templated=False,
        prepend_bos=True,
        pair_ids=[p["pair_id"] for p in pairs],
    )

    kept, stats = s2.filter_pairs_by_baseline_margin(
        model, pairs, cal, templated=False, max_seq_len=64, margin_threshold=0.5
    )
    assert [p["pair_id"] for p in kept] == ["en_0", "hi_0", "gu_0"]
    assert stats["fallback_used"] is False
    assert stats["positive_margin_count_by_language"] == {"en": 1, "gu_en": 1, "hi_en": 1, "kn_en": 0}
    assert {r["pair_id"] for r in stats["failed_refusal_pairs"]} == {"en_fail", "kn_fail", "hi_fail"}

    # Fallback mode: strict threshold too high (10.0) so 0 pairs pass -> deterministic stratified top-N highest margin
    kept_fb, stats_fb = s2.filter_pairs_by_baseline_margin(
        model, pairs, cal, templated=False, max_seq_len=64, margin_threshold=10.0, fallback_top_n=3
    )
    assert stats_fb["fallback_used"] is True
    assert len(kept_fb) == 3
    assert {p["pair_id"] for p in kept_fb} == {"en_0", "hi_0", "gu_0"}
    assert stats_fb["positive_margin_count_by_language"] == {"en": 0, "gu_en": 0, "hi_en": 0, "kn_en": 0}


def test_submodule_surgical_mask_exports_and_ctheta_locality(tmp_path):
    import stage2_discover as s2
    from safetune.core.circuit_kit.adapter import load_circuit_info_from_file
    from safetune.interventions.recover.ctheta import _matched_param_keys

    # Construct 36-layer Aya score dict where 24 scattered layers have 1 strong head each,
    # while only 12 distinct submodules hold concentrated safety mass:
    # 6 self_attn submodules (layers 2, 5, 8, 11, 14, 17) + 6 mlp submodules (layers 3, 6, 9, 12, 15, 18)
    # Note these 12 submodules span 12 DIFFERENT layers!
    node_scores: dict[str, float] = {}
    for layer in range(36):
        for head in range(16):
            node_scores[f"A{layer}.{head}"] = 0.01
        node_scores[f"MLP {layer}"] = 0.01

    attn_layers = [2, 5, 8, 11, 14, 17]
    mlp_layers = [3, 6, 9, 12, 15, 18]
    for lyr in attn_layers:
        for h in range(4):
            node_scores[f"A{lyr}.{h}"] = 1.0  # self_attn sum = 4.0 + 12*0.01 = 4.12
    for lyr in mlp_layers:
        node_scores[f"MLP {lyr}"] = 4.5  # mlp score = 4.5
    # Add 12 distracting single heads across layers 20..31 so standard top-10% saturates 24 layers
    for lyr in range(20, 32):
        node_scores[f"A{lyr}.0"] = 0.9

    # Diagnostic table compares 0.05, 0.10, 0.20 vs submodule-top-3, 4, 6, and custom 12
    diag_table = s2.compute_submodule_diagnostic_table(
        node_scores=node_scores,
        model_name="CohereLabs/tiny-aya-fire",
        n_layers=36,
        max_surgical_submodules=12,
    )
    assert [r["label"] for r in diag_table] == [
        "0.05",
        "0.10",
        "0.20",
        "submodule-top-3",
        "submodule-top-4",
        "submodule-top-6",
        "submodule-top-12",
    ]
    row_10 = next(r for r in diag_table if r["label"] == "0.10")
    row_sub4 = next(r for r in diag_table if r["label"] == "submodule-top-4")
    row_sub = next(r for r in diag_table if r["label"] == "submodule-top-12")
    assert row_10["projection_blocks_touched"] > row_sub["projection_blocks_touched"]
    assert row_sub4["projection_blocks_touched"] == 4
    assert row_sub4["projection_blocks_pct"] == pytest.approx(100.0 * 4 / 72)
    assert row_sub4["projection_blocks_pct"] <= 8.0
    assert row_sub4["frozen_pct"] == pytest.approx(100.0 * 68 / 72)
    assert row_sub["projection_blocks_touched"] == 12
    assert row_sub["total_projection_blocks"] == 72
    assert row_sub["projection_blocks_pct"] == pytest.approx(100.0 * 12 / 72)
    assert row_sub["frozen_pct"] == pytest.approx(100.0 * 60 / 72)

    su, ls, sub_diag = s2.compute_submodule_surgical_mask(
        node_scores=node_scores,
        model_name="CohereLabs/tiny-aya-fire",
        n_layers=36,
        max_surgical_submodules=12,
    )
    out_json = tmp_path / "aya_surgical_submodules_scores.json"
    out_json.write_text(
        json.dumps(
            {
                "version": "1.0",
                "model": "CohereLabs/tiny-aya-fire",
                "node_scores": node_scores,
                "safety_units": su,
                "layer_suggestions": ls,
            }
        ),
        encoding="utf-8",
    )
    ci = load_circuit_info_from_file(str(out_json))
    assert ci.layer_suggestions.layer_subset == sorted(attn_layers + mlp_layers)

    # Verify apply_ctheta _matched_param_keys matches ONLY those 12/72 submodules (NO sibling leakage!)
    aya_sd_keys = []
    for layer in range(36):
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            aya_sd_keys.append(f"model.layers.{layer}.self_attn.{proj}.weight")
        for proj in ("gate_proj", "up_proj", "down_proj"):
            aya_sd_keys.append(f"model.layers.{layer}.mlp.{proj}.weight")

    matched = _matched_param_keys(
        aya_sd_keys,
        ci.layer_suggestions.layer_subset,
        ci.layer_suggestions.target_modules,
    )
    # 6 attn * 4 weights + 6 mlp * 3 weights = 42 / 252 weights (83.33% frozen)
    assert len(matched) == 42
    frozen_ratio = 1.0 - (len(matched) / len(aya_sd_keys))
    assert frozen_ratio == pytest.approx(1.0 - 42 / 252)
    assert sub_diag["frozen_pct"] >= 83.3




def test_apply_ctheta_torch_equal_on_cohere2_and_custom_path_gpt2(tmp_path):
    """End-to-end SafeTune apply_ctheta preserves unmasked weights strictly (torch.equal) on Cohere2 and custom-path GPT-2."""
    import stage2_discover as s2
    from safetune.core.circuit_kit.adapter import load_circuit_info_from_file
    from safetune.interventions.recover.ctheta import _matched_param_keys, apply_ctheta

    stage1_dir = tmp_path / "stage1"
    stage1_dir.mkdir(parents=True)
    ds_path = stage1_dir / "pairs.jsonl"
    ds_path.write_text("\n".join(json.dumps(r) for r in _sample_pairs(8)) + "\n", encoding="utf-8")

    # 1) Tiny 4-layer Cohere2ForCausalLM
    cohere_ckpt = _create_tiny_hf_checkpoint(tmp_path / "local_aya_ckpt", arch="cohere2", with_chat_template=True, n_layers=4)
    cohere_out = tmp_path / "out_cohere"
    s2.run_stage2(
        model_name=str(cohere_ckpt),
        dataset_path=ds_path,
        output_dir=cohere_out,
        dtype="float32",
        device="cpu",
        max_surgical_layers=2,
        max_surgical_submodules=3,
        n_examples=6,
        n_stability_runs=2,
        ig_steps=2,
        stem="aya_tiny",
    )
    ci_aya = load_circuit_info_from_file(str(cohere_out / "aya_tiny_surgical_submodules_scores.json"))
    base_aya = transformers.Cohere2ForCausalLM.from_pretrained(cohere_ckpt)
    pos_aya = transformers.Cohere2ForCausalLM.from_pretrained(cohere_ckpt)
    neg_aya = transformers.Cohere2ForCausalLM.from_pretrained(cohere_ckpt)
    target_aya = transformers.Cohere2ForCausalLM.from_pretrained(cohere_ckpt)
    with torch.no_grad():
        for p in pos_aya.parameters():
            p.add_(0.25)

    repaired_aya = apply_ctheta(
        target_aya,
        pos_aya,
        neg_aya,
        ci_aya,
        strength=1.0,
    )
    base_sd = base_aya.state_dict()
    repaired_sd = repaired_aya.state_dict()
    matched_aya = set(
        _matched_param_keys(
            base_sd.keys(),
            ci_aya.layer_suggestions.layer_subset,
            ci_aya.layer_suggestions.target_modules,
        )
    )
    assert 0 < len(matched_aya) < len(base_sd)
    for k in base_sd:
        if k in matched_aya:
            assert not torch.equal(repaired_sd[k], base_sd[k]), f"Expected masked key {k} to change"
        else:
            assert torch.equal(repaired_sd[k], base_sd[k]), f"Unmasked weight {k} mutated unexpectedly"

    # 2) Custom local directory GPT2LMHeadModel (path deliberately omits 'gpt2' substring)
    custom_gpt2 = _create_tiny_hf_checkpoint(tmp_path / "custom_local_arch_dir", arch="gpt2", with_chat_template=False, n_layers=4)
    gpt2_out = tmp_path / "out_custom_gpt2"
    s2.run_stage2(
        model_name=str(custom_gpt2),
        dataset_path=ds_path,
        output_dir=gpt2_out,
        dtype="float32",
        device="cpu",
        max_surgical_layers=2,
        max_surgical_submodules=3,
        n_examples=6,
        n_stability_runs=2,
        ig_steps=2,
        stem="custom_gpt2",
    )
    ci_gpt2 = load_circuit_info_from_file(str(gpt2_out / "custom_gpt2_surgical_submodules_scores.json"))
    base_g = transformers.GPT2LMHeadModel.from_pretrained(custom_gpt2)
    pos_g = transformers.GPT2LMHeadModel.from_pretrained(custom_gpt2)
    neg_g = transformers.GPT2LMHeadModel.from_pretrained(custom_gpt2)
    target_g = transformers.GPT2LMHeadModel.from_pretrained(custom_gpt2)
    with torch.no_grad():
        for p in pos_g.parameters():
            p.add_(0.25)

    repaired_g = apply_ctheta(
        target_g,
        pos_g,
        neg_g,
        ci_gpt2,
        strength=1.0,
    )
    base_g_sd = base_g.state_dict()
    rep_g_sd = repaired_g.state_dict()
    matched_g = set(
        _matched_param_keys(
            base_g_sd.keys(),
            ci_gpt2.layer_suggestions.layer_subset,
            ci_gpt2.layer_suggestions.target_modules,
        )
    )
    assert 0 < len(matched_g) < len(base_g_sd)
    for k in base_g_sd:
        if k in matched_g:
            assert not torch.equal(rep_g_sd[k], base_g_sd[k]), f"Expected custom GPT2 key {k} to change"
        else:
            assert torch.equal(rep_g_sd[k], base_g_sd[k]), f"Unmasked custom GPT2 key {k} mutated unexpectedly"


def test_calibrated_task_spec_csv_cache_collision_resistance(tmp_path):
    """Different pair subsets sharing identical task_name must produce distinct hashed CSV caches and dataloader lengths."""
    import stage2_discover as s2

    ckpt = _create_tiny_hf_checkpoint(tmp_path / "ckpt", arch="cohere2", with_chat_template=False, n_layers=2)
    model = s2.ck.load_model(str(ckpt), dtype="float32", device="cpu")
    cal = s2.CalibrationResult(
        clean_token="sorry",
        clean_token_id=_VOCAB["sorry"],
        corrupt_token="here",
        corrupt_token_id=_VOCAB["here"],
        mean_clean_margin=0.5,
        mean_corrupt_margin=-0.5,
        logit_diff_gap=1.0,
        n_pairs_used=4,
        templated=False,
        prepend_bos=True,
        pair_ids=["p0"],
    )
    cache_dir = tmp_path / "csv_cache"
    all_pairs = _sample_pairs(8)
    spec_4 = s2.register_calibrated_task(
        pairs=all_pairs[:4],
        calibration=cal,
        model=model,
        templated=False,
        max_seq_len=64,
        task_name="shared_task_name",
        cache_dir=str(cache_dir),
    )
    spec_8 = s2.register_calibrated_task(
        pairs=all_pairs,
        calibration=cal,
        model=model,
        templated=False,
        max_seq_len=64,
        task_name="shared_task_name",
        cache_dir=str(cache_dir),
    )

    dl_4 = spec_4.build_dataloader(model, {"batch_size": 1, "num_examples": 8, "max_seq_len": 64}, "cpu")
    dl_8 = spec_8.build_dataloader(model, {"batch_size": 1, "num_examples": 8, "max_seq_len": 64}, "cpu")

    csv_paths = sorted(cache_dir.glob("shared_task_name_*.csv"))
    assert len(csv_paths) == 2
    assert len({p.name for p in csv_paths}) == 2
    assert len(list(dl_4)) == 4
    assert len(list(dl_8)) == 8


def test_leading_space_single_token_calibration_roundtrip():
    """Preflight calibration must accept leading-space BPE answer tokens like ' Sorry' and ' Here' while rejecting noise."""
    import stage2_discover as s2

    vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2, " Sorry": 3, " Here": 4, "!!!": 5, "\ufffd": 6, " prompt": 7}

    class _BpeSpaceTok:
        pad_token_id = 0
        bos_token_id = 1
        eos_token_id = 2
        pad_token = "<pad>"
        chat_template = None
        all_special_ids = [0, 1, 2]
        name_or_path = "space_bpe"

        def encode(self, text, add_special_tokens=False):
            if text in vocab:
                return [vocab[text]]
            return [vocab.get(w, 7) for w in str(text).split()]

        def decode(self, ids, skip_special_tokens=False):
            inv = {v: k for k, v in vocab.items()}
            return "".join(inv.get(int(i), "x") for i in ids)

        def __len__(self):
            return len(vocab)

    class _SpaceModel:
        tokenizer = _BpeSpaceTok()
        cfg = SimpleNamespace(d_vocab=len(vocab), device="cpu", model_name="gpt2")

        def to_tokens(self, text, prepend_bos=True):
            return torch.tensor([[1, 7]], dtype=torch.long)

        def __call__(self, tokens):
            # Clean ("harmful") prefers ' Sorry' and punctuation '!!!'; Corrupt prefers ' Here'
            logits = torch.zeros((tokens.shape[0], tokens.shape[1], len(vocab)), dtype=torch.float32)
            # Detect whether this batch represents harmful vs benign using caller mock count
            if getattr(self, "_call_idx", 0) % 2 == 0:
                logits[:, -1, 3] = 4.0   # ' Sorry'
                logits[:, -1, 5] = 9.0   # pure punct '!!!' must still be rejected
                logits[:, -1, 6] = 10.0  # replacement char must still be rejected
            else:
                logits[:, -1, 4] = 4.0   # ' Here'
            self._call_idx = getattr(self, "_call_idx", 0) + 1
            return logits

    pairs = _sample_pairs(4)
    cal = s2.calibrate_answer_tokens(_SpaceModel(), pairs, templated=False, max_seq_len=32)
    assert cal.clean_token == " Sorry"
    assert cal.clean_token_id == 3
    assert cal.corrupt_token == " Here"
    assert cal.corrupt_token_id == 4


def test_low_vram_t4_mode_overrides_profile_and_releases_vram(tmp_path):
    """--low-vram-t4 forces templated=False, max_seq_len=128, batch_size=1 and calls _release_vram between heavy stages."""
    import stage2_discover as s2

    ckpt = _create_tiny_hf_checkpoint(tmp_path / "t4_cohere2", arch="cohere2", with_chat_template=True, n_layers=2)
    tok = transformers.AutoTokenizer.from_pretrained(ckpt)
    prof = s2.resolve_runtime_profile(
        SimpleNamespace(tokenizer=tok),
        model_name=str(ckpt),
        chat_template_mode="on",
        batch_size=8,
        max_seq_len=512,
        dtype="auto",
        device="cpu",
        low_vram_t4=True,
    )
    assert prof.low_vram_t4 is True
    assert prof.templated is False
    assert prof.prepend_bos is True
    assert prof.max_seq_len == 128
    assert prof.batch_size == 1

    stage1_dir = tmp_path / "stage1"
    stage1_dir.mkdir(parents=True)
    ds_path = stage1_dir / "pairs.jsonl"
    ds_path.write_text("\n".join(json.dumps(r) for r in _sample_pairs(4)) + "\n", encoding="utf-8")

    events: list[str] = []
    orig_discover = s2.discover_circuit
    orig_eval = s2.evaluate_circuit
    orig_kl = s2.compute_kl_diagnostic
    orig_release = s2._release_vram

    def _wrap_discover(*a, **kw):
        events.append("discover")
        return orig_discover(*a, **kw)

    def _wrap_eval(*a, **kw):
        events.append("evaluate")
        return orig_eval(*a, **kw)

    def _wrap_kl(*a, **kw):
        events.append("kl")
        return orig_kl(*a, **kw)

    def _wrap_release():
        events.append("release")
        return orig_release()

    with patch.object(s2, "discover_circuit", side_effect=_wrap_discover), \
         patch.object(s2, "evaluate_circuit", side_effect=_wrap_eval), \
         patch.object(s2, "compute_kl_diagnostic", side_effect=_wrap_kl), \
         patch.object(s2, "_release_vram", side_effect=_wrap_release):
        s2.main(
            [
                "--model", str(ckpt),
                "--dataset", str(ds_path),
                "--output-dir", str(tmp_path / "t4_out"),
                "--chat-template-mode", "on",
                "--batch-size", "4",
                "--max-seq-len", "512",
                "--dtype", "float32",
                "--device", "cpu",
                "--n-examples", "4",
                "--n-stability-runs", "2",
                "--ig-steps", "2",
                "--low-vram-t4",
            ]
        )

    assert events == ["release", "discover", "release", "evaluate", "release", "kl"]
    report = json.loads((tmp_path / "t4_out" / "faithfulness_report.json").read_text(encoding="utf-8"))
    assert report["runtime_profile"]["low_vram_t4"] is True
    assert report["runtime_profile"]["templated"] is False
    assert report["runtime_profile"]["max_seq_len"] == 128
    assert report["runtime_profile"]["batch_size"] == 1


def test_upload_hf_fails_fast_and_uploads_phase_a_before_evaluate(tmp_path, monkeypatch):
    """--upload-hf must check HF auth before loading models and commit Phase A masks before evaluate_circuit starts."""
    import stage2_discover as s2

    stage1_dir = tmp_path / "stage1"
    stage1_dir.mkdir(parents=True)
    ds_path = stage1_dir / "pairs.jsonl"
    ds_path.write_text("\n".join(json.dumps(r) for r in _sample_pairs(4)) + "\n", encoding="utf-8")

    # 1) Missing token aborts BEFORE ck.load_model
    monkeypatch.delenv("HF_TOKEN", raising=False)
    load_spy = MagicMock()
    with patch("stage2_discover.load_dotenv"), patch.object(s2.ck, "load_model", load_spy):
        with pytest.raises(RuntimeError, match="HF_TOKEN"):
            s2.run_stage2(
                model_name="unused_model",
                dataset_path=ds_path,
                output_dir=tmp_path / "fail_fast_out",
                upload_hf=True,
            )
    load_spy.assert_not_called()

    # 2) Valid token executes verify_hf_auth -> model load -> discover -> Phase A upload -> evaluate -> KL -> Phase B upload
    ckpt = _create_tiny_hf_checkpoint(tmp_path / "hf_cohere2", arch="cohere2", with_chat_template=False, n_layers=2)
    monkeypatch.setenv("HF_TOKEN", "hf_dummy_token_do_not_leak")
    events: list[str] = []
    commit_messages: list[str] = []

    mock_api = MagicMock()
    mock_api.whoami.side_effect = lambda: events.append("whoami") or {"name": "tester"}

    def _fake_upload_folder(**kwargs):
        events.append("upload")
        commit_messages.append(str(kwargs.get("commit_message", "")))
        idx = len(commit_messages)
        return SimpleNamespace(commit_url=f"https://huggingface.co/datasets/test/commit/{idx}")

    mock_api.upload_folder.side_effect = _fake_upload_folder

    orig_load = s2.ck.load_model
    orig_discover = s2.discover_circuit
    orig_eval = s2.evaluate_circuit
    orig_kl = s2.compute_kl_diagnostic

    with patch("stage2_discover.HfApi", return_value=mock_api), \
         patch.object(s2.ck, "load_model", side_effect=lambda *a, **kw: events.append("load_model") or orig_load(*a, **kw)), \
         patch.object(s2, "discover_circuit", side_effect=lambda *a, **kw: events.append("discover") or orig_discover(*a, **kw)), \
         patch.object(s2, "evaluate_circuit", side_effect=lambda *a, **kw: events.append("evaluate") or orig_eval(*a, **kw)), \
         patch.object(s2, "compute_kl_diagnostic", side_effect=lambda *a, **kw: events.append("kl") or orig_kl(*a, **kw)):
        out_dir = tmp_path / "ok_upload_out"
        summary = s2.run_stage2(
            model_name=str(ckpt),
            dataset_path=ds_path,
            output_dir=out_dir,
            dtype="float32",
            device="cpu",
            n_examples=4,
            n_stability_runs=2,
            ig_steps=2,
            upload_hf=True,
            hf_repo="soapyyapps/circuit-kavach-cyber-codemixed",
        )

    assert events == ["whoami", "load_model", "discover", "upload", "evaluate", "kl", "upload"]
    assert "Phase A" in commit_messages[0]
    assert "Phase B" in commit_messages[1]
    assert summary["hf_phase_a_upload_url"] == "https://huggingface.co/datasets/test/commit/1"
    assert summary["hf_upload_url"] == "https://huggingface.co/datasets/test/commit/2"


def test_run_hf_jobs_script_contract_and_dry_run():
    """Validate scripts/run_hf_jobs.sh defaults (l4x1, 60m, HF_TOKEN, -v .:/workspace:ro) and bash dry-run."""
    import shutil
    import subprocess

    script_path = ROOT / "scripts" / "run_hf_jobs.sh"
    assert script_path.exists(), "scripts/run_hf_jobs.sh must exist"
    raw_bytes = script_path.read_bytes()
    assert b"\r\n" not in raw_bytes, "Shell script must use LF line endings"
    text = raw_bytes.decode("utf-8")
    for required in ("l4x1", "60m", "HF_TOKEN", "-v .:/workspace:ro", "stage2|stage3|stage4", "--dry-run", "--low-vram-t4"):
        assert required in text, f"Missing expected runner token: {required}"

    # Find real GNU/Git bash (avoid Windows System32 WSL stub if no distro installed)
    bash_candidates = [
        r"C:\Program Files\Git\bin\bash.exe",
        r"C:\Program Files\Git\usr\bin\bash.exe",
        shutil.which("bash"),
    ]
    bash_exe = None
    for cand in bash_candidates:
        if cand and Path(cand).exists():
            probe = subprocess.run([cand, "-c", "echo ok"], capture_output=True, text=True)
            if probe.returncode == 0 and probe.stdout.strip() == "ok":
                bash_exe = cand
                break
    assert bash_exe is not None, "Working bash executable not found to validate scripts/run_hf_jobs.sh"

    syntax = subprocess.run([bash_exe, "-n", str(script_path)], cwd=str(ROOT), capture_output=True, text=True)
    assert syntax.returncode == 0, f"bash -n failed: {syntax.stderr}"

    for stage, expected_py in (
        ("stage2", "src/stage2_discover.py"),
        ("stage3", "src/stage3_repair.py"),
        ("stage4", "src/stage4_eval.py"),
    ):
        res = subprocess.run(
            [bash_exe, str(script_path), stage, "--dry-run", "--low-vram-t4"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
        )
        assert res.returncode == 0, f"Dry-run failed for {stage}: {res.stderr}"
        out = res.stdout
        assert "DRY RUN:" in out
        assert "t4-small" in out
        assert "60m" in out
        assert "HF_TOKEN" in out
        assert ":/workspace:ro" in out
        assert expected_py in out
        assert "--low-vram-t4" in out
        assert "--upload-hf" in out


def test_stage2_t4_auto_dtype_and_upload_create_repo(tmp_path, monkeypatch) -> None:
    import sys
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    import torch

    src_dir = str(Path(__file__).resolve().parents[1] / "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    import stage2_discover as s2

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda including_emulation=True: False)

    prof = s2.resolve_runtime_profile(
        model_name="CohereLabs/tiny-aya-fire",
        dtype="auto",
        device="cuda",
        chat_template_mode="auto",
        batch_size=1,
        low_vram_t4=True,
    )
    assert prof.dtype == "float16"

    monkeypatch.setenv("HF_TOKEN", "hf_dummy_token_do_not_leak")
    mock_api = MagicMock()
    mock_api.upload_folder.return_value = SimpleNamespace(
        commit_url="https://huggingface.co/datasets/mock/stage2/commit/1"
    )
    monkeypatch.setattr(s2, "HfApi", lambda token=None: mock_api)
    out_dir = tmp_path / "stage2_out"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "faithfulness_report.json").write_text("{}", encoding="utf-8")

    s2.upload_stage2_to_hf(out_dir, repo_id="soapyyapps/test-repo")
    mock_api.create_repo.assert_called_once_with(
        repo_id="soapyyapps/test-repo", repo_type="dataset", exist_ok=True
    )


def test_stage2_hydrate_hf_cli_and_layer_cap_validation(tmp_path) -> None:
    import sys

    src_dir = str(Path(__file__).resolve().parents[1] / "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    import stage2_discover as s2

    parser = s2.build_arg_parser()
    parsed = parser.parse_args(["--hydrate-hf"])
    assert parsed.hydrate_hf is True

    try:
        s2.apply_surgical_layer_cap(
            {"blocks.0.hook_mlp_out": 1.0},
            model_name="gpt2",
            n_layers=2,
            top_fraction=0.5,
            max_surgical_layers=0,
        )
        raise AssertionError("Expected ValueError when max_surgical_layers=0")
    except ValueError as exc:
        assert ">= 1" in str(exc)
