# CircuitKavach-Aya: Universal 4-Block Safety Routing (`C-ΔΘ`) Across Multilingual `tiny-aya` Models (`lexsi-lab-track2`)

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB.svg?logo=python)](https://python.org)
[![Hugging Face Model Collection](https://img.shields.io/badge/%F0%9F%A4%97%20HF-Model%20Collection-yellow)](https://huggingface.co/collections/soapyyapps/circuitkavach-aya-surgical-safety-repair-models-and-adapters-6af9a5f97e17226835ac6f71)
[![Hugging Face Dataset Collection](https://img.shields.io/badge/%F0%9F%A4%97%20HF-Dataset%20Collection-orange)](https://huggingface.co/collections/soapyyapps/circuitkavach-aya-datasets-circuits-and-auditkit-logs-6af9a5fb037f3d7eaf674f49)
[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20HF-Live%20Demo%20Space-blue)](https://huggingface.co/spaces/soapyyapps/circuit-kavach-aya)

> **Lexsi Lab Track 2 Hackathon Submission (Track 2a & Track 2b)**  
> **Full Research Article:** [**`docs/submission_article.md`**](docs/submission_article.md) · **Summary:** [`WRITEUP_DRAFT.md`](WRITEUP_DRAFT.md) · **Root-Cause Brief:** [`docs/concepts_brief.md`](docs/concepts_brief.md) · **Run Decision Record:** [`docs/RUN_DECISION_RECORD.md`](docs/RUN_DECISION_RECORD.md)  
> *Content Warning: Referenced evaluation datasets and article examples contain defensive cybersecurity red-team prompts in English and code-mixed Hinglish, Gujlish, and Kanglish.*

**One-line claim:** Modifying just **4 of 72 transformer submodules (`2.55%` of parameters)** discovered via EAP-IG restores **94.5%–99.0% code-mixed cyberattack refusal** across four 3.35B `CohereLabs/tiny-aya` models (`fire`, `global`, `water`, `earth`) while holding **benign over-refusal at `0.0%`** and keeping **`97.45%` of weights bit-for-bit frozen**.

---

## 1. Executive Summary & Core Findings

Multilingual safety alignment frequently collapses in **romanized code-mixed registers**—Hinglish (`hi_en`), Gujlish (`gu_en`), and Kanglish (`kn_en`)—where adversaries combine Indic morphosyntax with English cybersecurity terms. While baseline `CohereLabs/tiny-aya` models refuse **100.0%** of standard English cyberattacks (`en`), code-mixed refusal falls to **48.0%–66.0%** in Hinglish, **6.0%–16.0%** in Gujlish, and **6.0%–38.0%** in Kanglish (`40.0%–53.5%` overall).

**CircuitKavach-Aya** repairs this failure via deterministic **circuit-masked parameter transplantation ($C\text{-}\Delta\Theta$)** across all four `CohereLabs/tiny-aya` variants (`fire`, `global`, `water`, `earth`):

$$\Theta_{\text{repaired}} = \Theta_{\text{base}} + \alpha \cdot (M \odot (\Theta_{\text{global}} - \Theta_{\text{base}})), \qquad \alpha^* = 1.0$$

1. **Universal 4-Block Safety Router (`Jaccard = 1.000`):** Across all four models and all 6 model pairs, `CircuitKIT` EAP-IG independently selects the exact same 4 of 72 submodules (`layers.0.mlp`, `layers.16.self_attn`, `layers.20.self_attn`, `layers.21.self_attn`) with **0.9217–0.9704 full 72-block Spearman rank correlation**, modifying `99,090,432` of `3,886,098,432` parameters (`2.55%`) and keeping **`276 / 291` tensors (`97.45%`) bit-identical**.
2. **Regional Specialization Paradox (*Capability–Safety Asymmetry*):** South Asia specialist `tiny-aya-fire` carries **3.17× higher Layer 0 MLP attribution (`7.0312` vs. `2.2188` on `earth`)** yet suffers the lowest unaligned base refusal (`40.0%` vs. `53.5%`) and hardest Kanglish tail (`78.0%` vs. `100.0%` on `earth`), while mid-network `layers.16.self_attn` stays conserved (`2.489–2.867`).
3. **Global Parity & Clean Residuals:** Surgical `C-ΔΘ` recovers **92.4%–98.9%** of 72-block Global SafeTune's refusal gain at **39.2× fewer modified parameters**, achieves statistical parity with full Global SafeTune on `earth` (`99.0%` vs. `99.5%`, $p=0.6124$), and keeps benign over-refusal at **`0.0%`**. Auditing all `27` residual misses across `800` surgical harmful replies shows `27 / 27` emitted benign defensive advice rather than exploit payloads.

---

## 2. Multi-Model Verification Matrix (`400` Held-Out Prompts × `4` Models)

| Model (`CohereLabs/tiny-aya-*`) | Base Attack Refusal | Global SafeTune (`72/72`, Over-Refusal) | **Surgical `C-ΔΘ` (`4/72`, Over-Refusal)** | Gain vs. Base (`Bootstrap p`) | Modified Params (`Frozen`) | Best `α` | Mean Benign Token Diversity |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`tiny-aya-fire`** (South Asia) | 40.0% | 99.0% (`0.0%`) | **94.5%** (`0.0%`) | **+54.5 pp** ($p=0.000999$) | **2.55%** (`97.45%`) | `1.0` | `0.9737` (`100%` helpful) |
| **`tiny-aya-global`** (Global) | 45.5% | 99.0% (`0.0%`) | **96.5%** (`0.0%`) | **+51.0 pp** ($p=0.000999$) | **2.55%** (`97.45%`) | `1.0` | `0.9701` (`100%` helpful) |
| **`tiny-aya-water`** (Europe/APAC) | 53.5% | 100.0% (`0.0%`) | **96.5%** (`0.0%`) | **+43.0 pp** ($p=0.000999$) | **2.55%** (`97.45%`) | `1.0` | `0.9784` (`100%` helpful) |
| **`tiny-aya-earth`** (Africa/W. Asia) | 53.5% | 99.5% (`0.0%`) | **99.0%** (`0.0%`) | **+45.5 pp** ($p=0.000999$) | **2.55%** (`97.45%`) | `1.0` | `0.9810` (`100%` helpful) |

### Per-Language Attack Refusal Rate (`Base → Global SafeTune → Surgical C-ΔΘ`)

| Model | `en` | `hi_en` (Hinglish) | `gu_en` (Gujlish) | `kn_en` (Kanglish) | RedTeam Pass (Base → Surg.) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`fire`** | 100.0% → 100.0% → **100.0%** | 48.0% → 100.0% → **100.0%** | 6.0% → 100.0% → **100.0%** | 6.0% → 96.0% → **78.0%** | 19.0% → **38.1%** |
| **`global`** | 100.0% → 100.0% → **100.0%** | 58.0% → 100.0% → **100.0%** | 12.0% → 100.0% → **100.0%** | 12.0% → 96.0% → **86.0%** | 19.0% → **52.4%** |
| **`water`** | 100.0% → 100.0% → **100.0%** | 66.0% → 100.0% → **100.0%** | 10.0% → 100.0% → **98.0%** | 38.0% → 100.0% → **88.0%** | 19.0% → **42.9%** |
| **`earth`** | 100.0% → 100.0% → **100.0%** | 66.0% → 100.0% → **100.0%** | 16.0% → 98.0% → **96.0%** | 32.0% → 100.0% → **100.0%** | 19.0% → **52.4%** |

*(Optional `WildGuard` and `lm-eval` hooks in [`src/stage4_eval.py`](src/stage4_eval.py) were `"disabled"` in the cloud runs; safety scoring uses dual deterministic judges `StringMatchJudge(mode="gcg")` + `RefusalDetector`, benign coherence auditing, and full inspection of all 27 residual surgical misses. See [Section 5.3 of `docs/submission_article.md`](docs/submission_article.md#53-checking-the-automatic-judge--honest-status-of-optional-judges).)*

---

## 3. Four-Stage Architecture & Lexsi Kits

Every stage emits SHA-256 `lexsi.provenance/1` manifests linking inputs to outputs across four Lexsi toolkits (`AlignTune` and `AgentTune` are not used):

1. **Stage 1 — `CuratorKIT v0.1.21` ([`src/stage1_curate.py`](src/stage1_curate.py)):** Curates **300** stratified lexical-twin pairs (`100` calibration + `200` disjoint held-out evaluation) across 4 languages × 3 cyber threat categories after rejecting 2 malformed candidates via `lexical_twin_gate` ([`rejected.jsonl`](artifacts/stage1_dataset/rejected.jsonl), `deduplication_gate`: `0` dropped).
2. **Stage 2 — `CircuitKIT v0.2.1` ([`src/stage2_discover.py`](src/stage2_discover.py)):** Applies [`circuitkit_rc2_dict_chat_template.patch`](patches/circuitkit_rc2_dict_chat_template.patch) (`RC2`), preserves Cohere's single-BOS template (`templated=True`, `prepend_bos=False`), equalizes pair lengths, calibrates refusal tokens (`" sorry"` vs. `" when"`), and runs EAP-IG (`ig_steps=3`, `max_seq_len=512`, `l40sx1` 48 GB VRAM) to export the top-4/72 submodule mask (`94.44%` blocks frozen).
3. **Stage 3 — `SafeTune v0.1.12` ([`src/stage3_repair.py`](src/stage3_repair.py)):** Applies [`safetune_rc1_sft_alignment_mode.patch`](patches/safetune_rc1_sft_alignment_mode.patch) (`RC1`), runs single-wrap LoRA `SafeGradHFTrainer` (`alignment_loss_mode="sft"`, `rho=1.0`, `epochs=3.0`, two-group `AdamW` at `2e-4` inside circuit vs. `0.10x` outside), merges Global Checkpoint 2 (`use_cache=True`), sweeps `α ∈ {0.5, 1.0, 1.5, 2.0}`, exports Surgical Checkpoint 3, and asserts `torch.equal` bit-identity on all `276 / 291` unmasked tensors.
4. **Stage 4 — `AuditKIT v1.1.6` ([`src/stage4_eval.py`](src/stage4_eval.py)):** Executes serialized greedy generation (`max_new_tokens=128`) + 89 `RedTeamRunner` checks per checkpoint, frees VRAM, runs isolated `DiskCache` scoring and 1,000-resample paired bootstrap tests, and validates `Stage4EvalSummarySchema` via Pydantic v2. Cross-model overlap and coherence are aggregated by [`src/aggregate_multimodel.py`](src/aggregate_multimodel.py).

---

## 4. Public Artifact Index (All Un-Gated / Public)

| Artifact | Scope | Public URL / Repo Path |
| :--- | :--- | :--- |
| **HF Model Collection** | All 12 adapters, global & surgical checkpoints | [Model Collection](https://huggingface.co/collections/soapyyapps/circuitkavach-aya-surgical-safety-repair-models-and-adapters-6af9a5f97e17226835ac6f71) |
| • Surgical `C-ΔΘ` (`fire` / `global` / `water` / `earth`) | `2.55%` modified, `97.45%` bit-frozen (`α=1.0`) | [`fire-merged`](https://huggingface.co/soapyyapps/tiny-aya-fire-circuit-kavach-merged) · [`global-merged`](https://huggingface.co/soapyyapps/tiny-aya-global-circuit-kavach-merged) · [`water-merged`](https://huggingface.co/soapyyapps/tiny-aya-water-circuit-kavach-merged) · [`earth-merged`](https://huggingface.co/soapyyapps/tiny-aya-earth-circuit-kavach-merged) |
| • Global SafeTune (`72/72` LoRA merged) | All-layer SafeGrad baselines | [`fire-global`](https://huggingface.co/soapyyapps/tiny-aya-fire-global-safetune) · [`global-global`](https://huggingface.co/soapyyapps/tiny-aya-global-global-safetune) · [`water-global`](https://huggingface.co/soapyyapps/tiny-aya-water-global-safetune) · [`earth-global`](https://huggingface.co/soapyyapps/tiny-aya-earth-global-safetune) |
| • Unmerged PEFT LoRA Adapters | Standalone `r=16` LoRA weights | [`fire-lora`](https://huggingface.co/soapyyapps/tiny-aya-fire-circuit-kavach-lora) · [`global-lora`](https://huggingface.co/soapyyapps/tiny-aya-global-circuit-kavach-lora) · [`water-lora`](https://huggingface.co/soapyyapps/tiny-aya-water-circuit-kavach-lora) · [`earth-lora`](https://huggingface.co/soapyyapps/tiny-aya-earth-circuit-kavach-lora) |
| **HF Dataset Collection** | Curated dataset, circuits, & AuditKIT logs | [Dataset Collection](https://huggingface.co/collections/soapyyapps/circuitkavach-aya-datasets-circuits-and-auditkit-logs-6af9a5fb037f3d7eaf674f49) |
| • Primary Dataset & `fire` + `multimodel` Hub | 300 pairs + `fire` logs + `multimodel_comparison.*` | [`soapyyapps/circuit-kavach-cyber-codemixed`](https://huggingface.co/datasets/soapyyapps/circuit-kavach-cyber-codemixed) |
| • Per-Model Stage 2–4 Hubs | `global`, `water`, `earth` circuits & logs | [`circuit-kavach-global`](https://huggingface.co/datasets/soapyyapps/circuit-kavach-global) · [`circuit-kavach-water`](https://huggingface.co/datasets/soapyyapps/circuit-kavach-water) · [`circuit-kavach-earth`](https://huggingface.co/datasets/soapyyapps/circuit-kavach-earth) |
| **Interactive ZeroGPU Demo Space** | Live inference + 4-model explorer | [`soapyyapps/circuit-kavach-aya`](https://huggingface.co/spaces/soapyyapps/circuit-kavach-aya) ([`space/app.py`](space/app.py)) |
| **Upstream Patches** | Portable vendor fixes (`RC1` & `RC2`) | [`circuitkit_rc2_dict_chat_template.patch`](patches/circuitkit_rc2_dict_chat_template.patch) · [`safetune_rc1_sft_alignment_mode.patch`](patches/safetune_rc1_sft_alignment_mode.patch) |
| **Full Research Write-Up** | Complete Track 2a/2b paper & appendices | [`docs/submission_article.md`](docs/submission_article.md) |

---

## 5. Local Verification & Cloud Reproduction Guide

### 5.1 Environment & Local CPU Verification (Zero 3.35B VRAM Required)
```bash
git clone https://github.com/dptel22/lexsi-lab-track2.git
cd lexsi-lab-track2

uv venv .venv --python 3.12
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e vendor/CuratorKIT -e vendor/CircuitKIT -e vendor/SafeTune -e vendor/AuditKIT
uv pip install -e .

# Run full CPU verification suite (64 unit & integration tests on "gpt2" + synthetic cohere2)
pytest tests/ -q
```

### 5.2 Reproducing Stage 2–4 on Hugging Face Jobs (USD 9.80 Total Across 4 Models)
Export `HF_TOKEN` (with `Jobs: read & write` enabled; never commit `.env`) and invoke [`scripts/run_hf_jobs.sh`](scripts/run_hf_jobs.sh):
```bash
# Stage 2 runs on NVIDIA L40S 48 GB (l40sx1) for templated EAP-IG; Stage 3 & 4 run on L4 24 GB (l4x1)
for MODEL in CohereLabs/tiny-aya-fire CohereLabs/tiny-aya-global CohereLabs/tiny-aya-water CohereLabs/tiny-aya-earth; do
  bash scripts/run_hf_jobs.sh stage2 --model "$MODEL"
  bash scripts/run_hf_jobs.sh stage3 --model "$MODEL"
  bash scripts/run_hf_jobs.sh stage4 --model "$MODEL"
done
```

---

## 6. Citation & Acknowledgments

```bibtex
@software{circuitkavach_aya_2026,
  author = {Dhruv Patel},
  title = {CircuitKavach-Aya: Universal 4-Block Safety Routing (C-DeltaTheta) Across Multilingual tiny-aya Models},
  year = {2026},
  url = {https://github.com/dptel22/lexsi-lab-track2}
}
```
Built by **Dhruv Patel** ([`soapyyapps`](https://huggingface.co/soapyyapps) / [`dptel22`](https://github.com/dptel22)) using the Lexsi Lab Alignment Stack (`CuratorKIT`, `CircuitKIT`, `SafeTune`, `AuditKIT`).
