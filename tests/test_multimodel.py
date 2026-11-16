"""Tests for cross-model universality and coherence aggregation (src/aggregate_multimodel.py)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _write_model_run_dir(
    root: Path,
    *,
    model_name: str,
    top_submodules: list[tuple[int, str]],
    scores_bias: float,
    base_attack: float,
    global_attack: float,
    surgical_attack: float,
    kn_surgical: float,
    en_only_warning: bool = False,
) -> Path:
    s2 = root / "stage2_circuits"
    s3 = root / "stage3_repair"
    s4 = root / "stage4_eval_logs"
    gen = s4 / "generations"
    s2.mkdir(parents=True, exist_ok=True)
    s3.mkdir(parents=True, exist_ok=True)
    gen.mkdir(parents=True, exist_ok=True)

    # 6 deterministic candidate submodules for Spearman correlation testing
    all_cands = [
        (0, "mlp"),
        (16, "self_attn"),
        (20, "self_attn"),
        (21, "self_attn"),
        (5, "mlp"),
        (12, "self_attn"),
    ]
    top_set = set(top_submodules)
    submodule_scores = {}
    for idx, (lyr, mod) in enumerate(all_cands):
        key = f"layers.{lyr}.{mod}"
        submodule_scores[key] = (10.0 - idx) + (5.0 if (lyr, mod) in top_set else 0.0) + scores_bias

    (s2 / "safety_circuit_surgical_submodules_top4_scores.json").write_text(
        json.dumps(
            {
                "model": model_name,
                "layer_suggestions": {
                    "layer_subset": sorted({lyr for lyr, _ in top_submodules}),
                    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                },
                "selected_submodules": [[lyr, mod] for lyr, mod in top_submodules],
                "submodule_scores": submodule_scores,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (s2 / "faithfulness_report.json").write_text(
        json.dumps(
            {
                "model": model_name,
                "calibration": {"clean_token": " sorry", "corrupt_token": " when", "logit_diff_gap": 12.5},
                "margin_filter": {
                    "selected_count": 90,
                    "total_evaluated": 100,
                    "positive_margin_count_by_language": {"en": 25, "hi_en": 24, "gu_en": 21, "kn_en": 20},
                    "selected_count_by_language": {"en": 25, "hi_en": 0 if en_only_warning else 24, "gu_en": 0 if en_only_warning else 21, "kn_en": 0 if en_only_warning else 20},
                    "english_only_circuit_warning": en_only_warning,
                    "fallback_used": False,
                },
                "metrics": {"ablation_score": 0.65, "mean_jaccard": 0.95},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (s3 / "repair_summary.json").write_text(
        json.dumps(
            {
                "model": model_name,
                "best_alpha": 1.0,
                "surgical_audit": {"modified_param_pct": 2.55, "frozen_param_pct": 97.45, "projection_blocks_touched": 4, "total_projection_blocks": 72},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (s4 / "eval_summary.json").write_text(
        json.dumps(
            {
                "model": model_name,
                "checkpoints": {
                    "checkpoint_1_base": {
                        "metrics": {
                            "attack_refusal_rate": base_attack,
                            "benign_over_refusal_rate": 0.0,
                            "benign_helpfulness_rate": 1.0,
                            "selectivity_gap": base_attack,
                            "by_language": {
                                "en": {"attack_refusal_rate": 1.0, "benign_over_refusal_rate": 0.0},
                                "hi_en": {"attack_refusal_rate": 0.4, "benign_over_refusal_rate": 0.0},
                                "gu_en": {"attack_refusal_rate": 0.1, "benign_over_refusal_rate": 0.0},
                                "kn_en": {"attack_refusal_rate": 0.1, "benign_over_refusal_rate": 0.0},
                            },
                        },
                        "redteam": {"attack_pass_rate": 0.2},
                    },
                    "checkpoint_2_global_safetune": {
                        "metrics": {
                            "attack_refusal_rate": global_attack,
                            "benign_over_refusal_rate": 0.0,
                            "benign_helpfulness_rate": 1.0,
                            "selectivity_gap": global_attack,
                            "by_language": {
                                "en": {"attack_refusal_rate": 1.0, "benign_over_refusal_rate": 0.0},
                                "hi_en": {"attack_refusal_rate": 1.0, "benign_over_refusal_rate": 0.0},
                                "gu_en": {"attack_refusal_rate": 1.0, "benign_over_refusal_rate": 0.0},
                                "kn_en": {"attack_refusal_rate": 0.96, "benign_over_refusal_rate": 0.0},
                            },
                        },
                        "redteam": {"attack_pass_rate": 0.48},
                    },
                    "checkpoint_3_surgical_ctheta": {
                        "metrics": {
                            "attack_refusal_rate": surgical_attack,
                            "benign_over_refusal_rate": 0.0,
                            "benign_helpfulness_rate": 1.0,
                            "selectivity_gap": surgical_attack,
                            "by_language": {
                                "en": {"attack_refusal_rate": 1.0, "benign_over_refusal_rate": 0.0},
                                "hi_en": {"attack_refusal_rate": 1.0, "benign_over_refusal_rate": 0.0},
                                "gu_en": {"attack_refusal_rate": 1.0, "benign_over_refusal_rate": 0.0},
                                "kn_en": {"attack_refusal_rate": kn_surgical, "benign_over_refusal_rate": 0.0},
                            },
                        },
                        "redteam": {"attack_pass_rate": 0.38},
                    },
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    benign_rows = [
        {"split": "benign", "language": "en", "response": "Use parameterized SQL queries to keep database inputs safely separated."},
        {"split": "benign", "language": "kn_en", "response": "Always verify UPI merchant requests using official banking apps before approving."},
    ]
    (gen / "checkpoint_3_surgical_ctheta.jsonl").write_text(
        "\n".join(json.dumps(r) for r in benign_rows) + "\n", encoding="utf-8"
    )
    return root


def test_aggregate_multimodel_computes_jaccard_spearman_safety_and_coherence(tmp_path: Path) -> None:
    import aggregate_multimodel as agg

    fire_dir = _write_model_run_dir(
        tmp_path / "fire",
        model_name="CohereLabs/tiny-aya-fire",
        top_submodules=[(0, "mlp"), (16, "self_attn"), (20, "self_attn"), (21, "self_attn")],
        scores_bias=0.0,
        base_attack=0.40,
        global_attack=0.99,
        surgical_attack=0.945,
        kn_surgical=0.78,
    )
    global_dir = _write_model_run_dir(
        tmp_path / "global",
        model_name="CohereLabs/tiny-aya-global",
        top_submodules=[(0, "mlp"), (16, "self_attn"), (5, "mlp"), (12, "self_attn")],
        scores_bias=1.0,
        base_attack=0.35,
        global_attack=0.96,
        surgical_attack=0.91,
        kn_surgical=0.72,
    )
    water_dir = _write_model_run_dir(
        tmp_path / "water",
        model_name="CohereLabs/tiny-aya-water",
        top_submodules=[(0, "mlp"), (16, "self_attn"), (20, "self_attn"), (21, "self_attn")],
        scores_bias=0.5,
        base_attack=0.30,
        global_attack=0.90,
        surgical_attack=0.85,
        kn_surgical=0.60,
        en_only_warning=True,
    )

    out_json = tmp_path / "out" / "multimodel_comparison.json"
    out_md = tmp_path / "out" / "multimodel_comparison.md"
    result = agg.aggregate_multimodel_runs(
        {"fire": fire_dir, "global": global_dir, "water": water_dir},
        output_json=out_json,
        output_md=out_md,
    )

    assert out_json.is_file()
    assert out_md.is_file()
    # fire vs water share 4/4 submodules -> Jaccard = 1.0; fire vs global share 2/6 -> Jaccard = 1/3
    assert result["circuit_overlap"]["fire__water"]["jaccard"] == pytest.approx(1.0)
    assert result["circuit_overlap"]["fire__global"]["jaccard"] == pytest.approx(2.0 / 6.0)
    assert result["circuit_overlap"]["fire__water"]["spearman"] == pytest.approx(1.0)
    assert result["models"]["water"]["english_only_circuit_warning"] is True
    assert result["models"]["fire"]["coherence"]["benign_count"] == 2
    assert result["models"]["fire"]["coherence"]["mean_benign_chars"] > 40.0
    md_text = out_md.read_text(encoding="utf-8")
    assert "tiny-aya-fire" in md_text
    assert "tiny-aya-global" in md_text
    assert "Jaccard" in md_text


def test_aggregate_multimodel_supports_nested_stage2_schema_and_node_scores_fallback(tmp_path: Path) -> None:
    import aggregate_multimodel as agg

    run_a = _write_model_run_dir(
        tmp_path / "run_a",
        model_name="CohereLabs/tiny-aya-fire",
        top_submodules=[(0, "mlp"), (16, "self_attn"), (20, "self_attn"), (21, "self_attn")],
        scores_bias=0.0,
        base_attack=0.40,
        global_attack=0.99,
        surgical_attack=0.945,
        kn_surgical=0.78,
    )
    run_b = _write_model_run_dir(
        tmp_path / "run_b",
        model_name="CohereLabs/tiny-aya-global",
        top_submodules=[(0, "mlp"), (16, "self_attn"), (20, "self_attn"), (21, "self_attn")],
        scores_bias=0.0,
        base_attack=0.38,
        global_attack=0.98,
        surgical_attack=0.92,
        kn_surgical=0.76,
    )

    # Replace synthetic flat schema with real Stage 2 nested schemas + node_scores fallback.
    (run_a / "stage2_circuits" / "safety_circuit_surgical_submodules_top4_scores.json").write_text(
        json.dumps(
            {
                "model": "CohereLabs/tiny-aya-fire",
                "node_scores": {
                    "MLP 0": 7.0,
                    "A16.0": 1.5,
                    "A16.1": 1.0,
                    "A20.0": 1.8,
                    "A21.0": 1.7,
                    "MLP 5": 0.4,
                },
                "layer_suggestions": {
                    "metadata": {
                        "selected_submodules": [
                            {"layer": 0, "submodule": "mlp", "rank_score": 7.0},
                            {"layer": 16, "submodule": "self_attn", "rank_score": 2.5},
                            {"layer": 20, "submodule": "self_attn", "rank_score": 1.8},
                            {"layer": 21, "submodule": "self_attn", "rank_score": 1.7},
                        ]
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (run_a / "stage2_circuits" / "faithfulness_report.json").write_text(
        json.dumps(
            {
                "model": "CohereLabs/tiny-aya-fire",
                "calibration": {"clean_token": " sorry", "corrupt_token": " when", "logit_diff_gap": 16.2},
                "margin_filter": {"selected_count": 99, "english_only_circuit_warning": False},
                "faithfulness_report": {
                    "patching_score": 0.023,
                    "ablation_score": 0.655,
                    "stability": {"mean_jaccard": 0.968, "circuits": [{"a0.h0": 0.1}]},
                },
            }
        ),
        encoding="utf-8",
    )

    (run_b / "stage2_circuits" / "safety_circuit_surgical_submodules_top4_scores.json").write_text(
        json.dumps(
            {
                "model": "CohereLabs/tiny-aya-global",
                "node_scores": {
                    "MLP 0": 4.5,
                    "A16.0": 1.6,
                    "A16.1": 0.9,
                    "A20.0": 1.7,
                    "A21.0": 1.6,
                    "MLP 5": 0.3,
                },
                "surgical_submodules": {
                    "selected_submodules": [
                        {"layer": 0, "submodule": "mlp", "rank_score": 4.5},
                        {"layer": 16, "submodule": "self_attn", "rank_score": 2.5},
                        {"layer": 20, "submodule": "self_attn", "rank_score": 1.7},
                        {"layer": 21, "submodule": "self_attn", "rank_score": 1.6},
                    ]
                },
            }
        ),
        encoding="utf-8",
    )

    res = agg.aggregate_multimodel_runs({"fire": run_a, "global": run_b})
    ov = res["circuit_overlap"]["fire__global"]
    assert res["models"]["fire"]["selected_submodules"] == [[0, "mlp"], [16, "self_attn"], [20, "self_attn"], [21, "self_attn"]]
    assert res["models"]["global"]["selected_submodules"] == [[0, "mlp"], [16, "self_attn"], [20, "self_attn"], [21, "self_attn"]]
    assert ov["jaccard"] == pytest.approx(1.0)
    assert ov["n_shared_score_keys"] == 5
    assert ov["spearman"] == pytest.approx(1.0)
    assert res["models"]["fire"]["faithfulness_metrics"]["ablation_score"] == pytest.approx(0.655)
    assert res["models"]["fire"]["faithfulness_metrics"]["mean_jaccard"] == pytest.approx(0.968)
    assert "circuits" not in res["models"]["fire"]["faithfulness_metrics"]
