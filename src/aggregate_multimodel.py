"""Cross-model circuit overlap, safety, and coherence aggregator for CircuitKavach-Aya."""
from __future__ import annotations

import argparse
import json
import math
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _average_ranks(values: Sequence[float]) -> List[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j - 1) / 2.0 + 1.0
        for k in range(i, j):
            ranks[indexed[k][0]] = avg_rank
        i = j
    return ranks


def compute_spearman_correlation(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Compute Spearman rank correlation between two equal-length numeric sequences."""
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    rx = _average_ranks(xs)
    ry = _average_ranks(ys)
    mean_x = sum(rx) / len(rx)
    mean_y = sum(ry) / len(ry)
    num = sum((a - mean_x) * (b - mean_y) for a, b in zip(rx, ry))
    den_x = sum((a - mean_x) ** 2 for a in rx)
    den_y = sum((b - mean_y) ** 2 for b in ry)
    denom = math.sqrt(den_x * den_y)
    return float(num / denom) if denom > 0.0 else 0.0


def compute_submodule_overlap(
    sub_a: Iterable[Tuple[int, str]],
    sub_b: Iterable[Tuple[int, str]],
    scores_a: Mapping[str, float],
    scores_b: Mapping[str, float],
) -> Dict[str, Any]:
    """Compute top-K submodule Jaccard similarity and full-grid Spearman score correlation."""
    set_a = {(int(lyr), str(mod)) for lyr, mod in sub_a}
    set_b = {(int(lyr), str(mod)) for lyr, mod in sub_b}
    inter = sorted(set_a & set_b)
    union = set_a | set_b
    jaccard = float(len(inter) / len(union)) if union else 0.0

    shared_keys = sorted(set(scores_a.keys()) & set(scores_b.keys()))
    spearman = compute_spearman_correlation(
        [float(scores_a[k]) for k in shared_keys],
        [float(scores_b[k]) for k in shared_keys],
    )
    return {
        "jaccard": jaccard,
        "shared_submodules": [[lyr, mod] for lyr, mod in inter],
        "spearman": spearman,
        "n_shared_score_keys": len(shared_keys),
    }


def compute_generation_coherence(jsonl_path: Path) -> Dict[str, Any]:
    """Audit benign responses for non-empty generation length and unique-token diversity."""
    if not jsonl_path.is_file():
        return {"benign_count": 0, "mean_benign_chars": 0.0, "non_empty_rate": 0.0, "mean_token_diversity": 0.0, "by_language": {}}

    lengths: List[int] = []
    diversities: List[float] = []
    non_empty = 0
    by_lang: Dict[str, Dict[str, Any]] = {}

    for raw in jsonl_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        row = json.loads(line)
        if str(row.get("split", "")) != "benign":
            continue
        resp = str(row.get("response") or "").strip()
        lang = str(row.get("language") or "unknown")
        toks = resp.split()
        div = (len(set(toks)) / len(toks)) if toks else 0.0
        lengths.append(len(resp))
        diversities.append(div)
        if len(resp) >= 16:
            non_empty += 1
        bucket = by_lang.setdefault(lang, {"count": 0, "chars": 0, "div_sum": 0.0})
        bucket["count"] += 1
        bucket["chars"] += len(resp)
        bucket["div_sum"] += div

    total = len(lengths)
    by_lang_summary = {
        lg: {
            "count": info["count"],
            "mean_chars": round(info["chars"] / max(info["count"], 1), 2),
            "mean_token_diversity": round(info["div_sum"] / max(info["count"], 1), 4),
        }
        for lg, info in sorted(by_lang.items())
    }
    return {
        "benign_count": total,
        "mean_benign_chars": round(sum(lengths) / max(total, 1), 2),
        "non_empty_rate": round(non_empty / max(total, 1), 4),
        "mean_token_diversity": round(sum(diversities) / max(total, 1), 4),
        "by_language": by_lang_summary,
    }


def _parse_submodule_item(item: Any) -> Optional[Tuple[int, str]]:
    if isinstance(item, Mapping):
        if "layer" in item and "submodule" in item:
            return (int(item["layer"]), str(item["submodule"]))
        return None
    if isinstance(item, Sequence) and not isinstance(item, (str, bytes)) and len(item) >= 2:
        return (int(item[0]), str(item[1]))
    return None


def _extract_selected_submodules(mask: Mapping[str, Any]) -> List[Tuple[int, str]]:
    """Extract ordered (layer, submodule) tuples from top-level or nested Stage 2 masks."""
    candidates: List[Any] = [
        mask.get("selected_submodules"),
        ((mask.get("layer_suggestions") or {}).get("metadata") or {}).get("selected_submodules"),
        (mask.get("surgical_submodules") or (mask.get("metadata") or {}).get("surgical_submodules") or {}).get(
            "selected_submodules"
        ),
    ]
    for raw_list in candidates:
        if not raw_list:
            continue
        parsed = [p for p in (_parse_submodule_item(item) for item in raw_list) if p is not None]
        if parsed:
            return parsed
    return []


def _scores_from_node_scores(node_scores: Mapping[str, Any]) -> Dict[str, float]:
    """Aggregate CircuitKIT node attribution scores into 72-block submodule scores."""
    agg: Dict[str, float] = {}
    has_structured_keys = False
    for raw_key, raw_val in node_scores.items():
        key = str(raw_key)
        val = abs(float(raw_val))
        if key.startswith("MLP "):
            suffix = key.split(None, 1)[1].strip()
            if suffix.isdigit():
                has_structured_keys = True
                blk = f"layers.{int(suffix)}.mlp"
                agg[blk] = agg.get(blk, 0.0) + val
                continue
        if key.startswith("A") and "." in key:
            head_layer = key[1:].split(".", 1)[0]
            if head_layer.isdigit():
                has_structured_keys = True
                blk = f"layers.{int(head_layer)}.self_attn"
                agg[blk] = agg.get(blk, 0.0) + val
                continue
    if has_structured_keys and agg:
        return agg
    return {str(k): float(v) for k, v in node_scores.items()}


def _extract_submodule_scores(mask: Mapping[str, Any], s2_dir: Path) -> Dict[str, float]:
    """Resolve full submodule score map from top-4 mask, full mask, or node_scores."""
    direct = mask.get("submodule_scores")
    if isinstance(direct, Mapping) and direct:
        return {str(k): float(v) for k, v in direct.items()}

    full_path = s2_dir / "safety_circuit_surgical_submodules_scores.json"
    full_mask: Dict[str, Any] = _read_json(full_path) if full_path.is_file() else {}
    if isinstance(full_mask.get("submodule_scores"), Mapping) and full_mask["submodule_scores"]:
        return {str(k): float(v) for k, v in full_mask["submodule_scores"].items()}

    node_scores = mask.get("node_scores") or full_mask.get("node_scores")
    if isinstance(node_scores, Mapping) and node_scores:
        return _scores_from_node_scores(node_scores)

    for src in (full_mask, mask):
        sel = _extract_selected_submodules(src)
        raw_list = ((src.get("layer_suggestions") or {}).get("metadata") or {}).get("selected_submodules") or (
            src.get("surgical_submodules") or (src.get("metadata") or {}).get("surgical_submodules") or {}
        ).get("selected_submodules")
        if sel and isinstance(raw_list, list):
            out: Dict[str, float] = {}
            for item in raw_list:
                if isinstance(item, Mapping) and "rank_score" in item and "layer" in item and "submodule" in item:
                    out[f"layers.{int(item['layer'])}.{item['submodule']}"] = float(item["rank_score"])
            if out:
                return out
    return {}


def _extract_faithfulness_metrics(faith: Mapping[str, Any]) -> Dict[str, Any]:
    """Extract scalar faithfulness metrics from either compact test schema or Stage 2 report."""
    metrics = faith.get("metrics")
    if isinstance(metrics, Mapping) and metrics:
        return dict(metrics)
    fr = faith.get("faithfulness_report")
    if not isinstance(fr, Mapping):
        return {}
    out: Dict[str, Any] = {}
    for key in ("patching_score", "ablation_score"):
        if key in fr:
            out[key] = fr[key]
    stability = fr.get("stability")
    if isinstance(stability, Mapping):
        for key in ("mean_jaccard", "std_jaccard", "mean_dice", "mean_spearman", "n_stable_nodes"):
            if key in stability:
                out[key] = stability[key]
    return out


def _extract_model_summary(slug: str, run_dir: Path) -> Dict[str, Any]:
    s2_dir = run_dir / "stage2_circuits"
    s3_dir = run_dir / "stage3_repair"
    s4_dir = run_dir / "stage4_eval_logs"

    faith = _read_json(s2_dir / "faithfulness_report.json")
    mask = _read_json(s2_dir / "safety_circuit_surgical_submodules_top4_scores.json")
    repair = _read_json(s3_dir / "repair_summary.json")
    eval_sum = _read_json(s4_dir / "eval_summary.json")
    coherence = compute_generation_coherence(s4_dir / "generations" / "checkpoint_3_surgical_ctheta.jsonl")

    margin = faith.get("margin_filter") or {}
    cal = faith.get("calibration") or {}
    submodules = _extract_selected_submodules(mask)
    scores = _extract_submodule_scores(mask, s2_dir)
    audit = repair.get("surgical_audit") or {}

    return {
        "slug": slug,
        "model": str(eval_sum.get("model") or repair.get("model") or mask.get("model") or slug),
        "calibration": cal,
        "margin_filter": margin,
        "english_only_circuit_warning": bool(margin.get("english_only_circuit_warning", False)),
        "selected_submodules": [[lyr, mod] for lyr, mod in submodules],
        "submodule_scores": scores,
        "faithfulness_metrics": _extract_faithfulness_metrics(faith),
        "best_alpha": repair.get("best_alpha"),
        "surgical_audit": audit,
        "checkpoints": eval_sum.get("checkpoints") or {},
        "coherence": coherence,
    }


def _render_markdown(result: Dict[str, Any]) -> str:
    lines = [
        "# CircuitKavach-Aya Multi-Model Universality Comparison",
        "",
        "## Safety, Locality & Selectivity Matrix",
        "",
        "| Model | Base Attack Refusal | Global Refusal (Over-Refusal) | Surgical Refusal (Over-Refusal) | Surgical Selectivity | Surgical Modified % | Best Alpha | `kn_en` Surgical Refusal | En-Only Circuit Warning |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |",
    ]
    for slug, info in result["models"].items():
        ckpts = info["checkpoints"]
        m_base = (ckpts.get("checkpoint_1_base") or {}).get("metrics") or {}
        m_glob = (ckpts.get("checkpoint_2_global_safetune") or {}).get("metrics") or {}
        m_surg = (ckpts.get("checkpoint_3_surgical_ctheta") or {}).get("metrics") or {}
        kn_rate = (((m_surg.get("by_language") or {}).get("kn_en") or {}).get("attack_refusal_rate"))
        kn_str = f"{float(kn_rate) * 100:.1f}%" if kn_rate is not None else "N/A"
        mod_pct = (info["surgical_audit"] or {}).get("modified_param_pct")
        mod_str = f"{float(mod_pct):.2f}%" if mod_pct is not None else "N/A"
        warn_str = "YES" if info["english_only_circuit_warning"] else "NO"
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{info['model']}` (`{slug}`)",
                    f"{float(m_base.get('attack_refusal_rate', 0.0)) * 100:.1f}%",
                    f"{float(m_glob.get('attack_refusal_rate', 0.0)) * 100:.1f}% ({float(m_glob.get('benign_over_refusal_rate', 0.0)) * 100:.1f}%)",
                    f"{float(m_surg.get('attack_refusal_rate', 0.0)) * 100:.1f}% ({float(m_surg.get('benign_over_refusal_rate', 0.0)) * 100:.1f}%)",
                    f"{float(m_surg.get('selectivity_gap', 0.0)):+.3f}",
                    mod_str,
                    str(info.get("best_alpha", "N/A")),
                    kn_str,
                    warn_str,
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Pairwise Circuit Overlap (Top-4 Submodule Jaccard & 72-Block Spearman Rank)",
            "",
            "| Model Pair | Top-4 Jaccard | Spearman Rho | Shared Submodules |",
            "| --- | ---: | ---: | --- |",
        ]
    )
    for pair_key, ov in result["circuit_overlap"].items():
        shared = ", ".join(f"`layers.{lyr}.{mod}`" for lyr, mod in ov["shared_submodules"]) or "None"
        lines.append(
            f"| `{pair_key}` | {ov['jaccard']:.3f} | {ov['spearman']:.4f} | {shared} |"
        )
    lines.append("")
    return "\n".join(lines)


def aggregate_multimodel_runs(
    run_dirs: Mapping[str, Path | str],
    *,
    output_json: Optional[Path | str] = None,
    output_md: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """Aggregate Stage 2-4 summaries across multiple models and compute circuit overlap + coherence."""
    models: Dict[str, Dict[str, Any]] = {}
    for slug, path in run_dirs.items():
        models[str(slug)] = _extract_model_summary(str(slug), Path(path))

    overlap: Dict[str, Dict[str, Any]] = {}
    for slug_a, slug_b in combinations(models.keys(), 2):
        ma = models[slug_a]
        mb = models[slug_b]
        overlap[f"{slug_a}__{slug_b}"] = compute_submodule_overlap(
            [(int(x[0]), str(x[1])) for x in ma["selected_submodules"]],
            [(int(x[0]), str(x[1])) for x in mb["selected_submodules"]],
            ma["submodule_scores"],
            mb["submodule_scores"],
        )

    result = {
        "models": {
            slug: {k: v for k, v in info.items() if k != "submodule_scores"}
            for slug, info in models.items()
        },
        "circuit_overlap": overlap,
    }
    md_text = _render_markdown(result)

    if output_json is not None:
        out_j = Path(output_json)
        out_j.parent.mkdir(parents=True, exist_ok=True)
        out_j.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if output_md is not None:
        out_m = Path(output_md)
        out_m.parent.mkdir(parents=True, exist_ok=True)
        out_m.write_text(md_text, encoding="utf-8")
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Aggregate multi-model CircuitKavach-Aya runs.")
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Model run mapping in the form slug=path/to/dir (repeatable).",
    )
    parser.add_argument("--output-json", default="artifacts/multimodel_comparison.json")
    parser.add_argument("--output-md", default="artifacts/multimodel_comparison.md")
    args = parser.parse_args(list(argv) if argv is not None else None)

    run_map: Dict[str, Path] = {}
    for spec in args.run:
        if "=" not in spec:
            raise ValueError(f"Expected --run slug=path, got {spec!r}")
        slug, path_str = spec.split("=", 1)
        run_map[slug.strip()] = Path(path_str.strip())

    aggregate_multimodel_runs(run_map, output_json=args.output_json, output_md=args.output_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
