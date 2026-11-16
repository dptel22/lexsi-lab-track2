---
title: CircuitKavach-Aya
emoji: 🛡️
colorFrom: indigo
colorTo: gray
sdk: gradio
sdk_version: 5.25.0
app_file: app.py
pinned: false
license: apache-2.0
---

# CircuitKavach-Aya: Surgical Multilingual Safety Repair

Interactive ZeroGPU Space for **Lexsi Lab Track 2**, demonstrating sparse **Circuit-Masked Parameter Transplantation (`C-ΔΘ`)** across all 4 regional variants of Cohere Labs' 3.35B Tiny Aya family (`fire`, `global`, `water`, `earth`) on English and romanized Indic code-mixed cybersecurity prompts (`en`, `hi_en`, `gu_en`, `kn_en`).

## Universal `4/72` Safety Circuit Finding

Across all four independently discovered variants (`CohereLabs/tiny-aya-{fire,global,water,earth}`), Stage 2 `CircuitKIT` EAP-IG converges on the **exact same 4 out of 72 projection submodules** (`layers.0.mlp`, `layers.16.self_attn`, `layers.20.self_attn`, `layers.21.self_attn` — **top-4 Jaccard = 1.000**, 72-block **Spearman $\rho$ = 0.9217–0.9704**). Stage 3 `SafeTune` `C-ΔΘ` (`best_alpha = 1.0`) modifies only **2.55%** of total parameters (`99,090,432 / 3,886,098,432`) while keeping **97.45% (`276/291` tensors)** bit-for-bit identical:

| Model Variant | Base Attack Refusal | Global SafeTune (`72/72`) | Surgical `C-ΔΘ` (`4/72`, `2.55%` params) | Benign Over-Refusal | Selectivity Gap |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `CohereLabs/tiny-aya-fire` | 40.0% | 99.0% | **94.5%** | 0.0% | **+0.945** |
| `CohereLabs/tiny-aya-global` | 45.5% | 99.0% | **96.5%** | 0.0% | **+0.965** |
| `CohereLabs/tiny-aya-water` | 53.5% | 100.0% | **96.5%** | 0.0% | **+0.965** |
| `CohereLabs/tiny-aya-earth` | 53.5% | 99.5% | **99.0%** | 0.0% | **+0.990** |

## Links & Artifact Collections

- **Model Collection (4 base + 12 trained/merged repos):** [CircuitKavach-Aya Models & Adapters](https://huggingface.co/collections/soapyyapps/circuitkavach-aya-surgical-safety-repair-models-and-adapters-6af9a5f97e17226835ac6f71)
- **Dataset & Audit Collection (4 datasets + Space):** [CircuitKavach-Aya Datasets, Circuits & AuditKIT Logs](https://huggingface.co/collections/soapyyapps/circuitkavach-aya-datasets-circuits-and-auditkit-logs-6af9a5fb037f3d7eaf674f49)
- **Variant Datasets:** [`circuit-kavach-cyber-codemixed` (`fire`)](https://huggingface.co/datasets/soapyyapps/circuit-kavach-cyber-codemixed) | [`circuit-kavach-global`](https://huggingface.co/datasets/soapyyapps/circuit-kavach-global) | [`circuit-kavach-water`](https://huggingface.co/datasets/soapyyapps/circuit-kavach-water) | [`circuit-kavach-earth`](https://huggingface.co/datasets/soapyyapps/circuit-kavach-earth)
- **GitHub Repository:** [`dptel22/lexsi-lab-track2`](https://github.com/dptel22/lexsi-lab-track2)

## Space Tabs

1. **Live ZeroGPU Inference:** Run `CohereLabs/tiny-aya-fire` (or a configured `CIRCUITKAVACH_MODEL` checkpoint) with Pydantic v2 validated output and live `tok/s` / latency telemetry.
2. **Stage 4 ROI & 3-Checkpoint Comparison:** Inspect the 3-checkpoint `AuditKIT` selectivity and `RedTeamRunner` ROI table alongside the full `eval_summary.json`.
3. **Stage 2 Surgical Circuit Mask (4/72 Submodules):** Inspect the `4/72` (`94.44%` blocks frozen) EAP-IG surgical submodule scores (`safety_circuit_surgical_submodules_top4_scores.json`).
4. **4-Model Universality & Cross-Regional Transfer:** Inspect the full 4-model comparison table, pairwise Jaccard/Spearman matrix, and `multimodel_comparison.json`.
