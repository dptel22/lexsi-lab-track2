# CircuitKavach-Aya: Surgical Multilingual Safety Alignment (`lexsi-lab-track2`)

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB.svg?logo=python)](https://python.org)
[![Hugging Face Model Collection](https://img.shields.io/badge/%F0%9F%A4%97%20HF-Model%20Collection-yellow)](https://huggingface.co/collections/soapyyapps/circuitkavach-aya-surgical-safety-repair-models-and-adapters-6af9a5f97e17226835ac6f71)
[![Hugging Face Dataset Collection](https://img.shields.io/badge/%F0%9F%A4%97%20HF-Dataset%20Collection-orange)](https://huggingface.co/collections/soapyyapps/circuitkavach-aya-datasets-circuits-and-auditkit-logs-6af9a5fb037f3d7eaf674f49)
[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20HF-Live%20Demo%20Space-blue)](https://huggingface.co/spaces/soapyyapps/circuit-kavach-aya)

> **Lexsi Lab Hackathon — Track 2 Submission**  
> Mechanistic Interpretability & Circuit-Guided Surgical Repair for Multilingual Cybersecurity Alignment on `CohereLabs/tiny-aya-fire` (3.35B).

---

## 1. Executive Summary & Core Results

Multilingual safety alignment frequently collapses in **code-mixed** conversational registers (such as Hinglish `hi_en`, Gujlish `gu_en`, and Kanglish `kn_en`), where adversaries mix colloquial romanized Indic phonetics with technical English cybersecurity terminology. While standard global parameter fine-tuning (LoRA across all layers) can enforce refusal, it modifies 100% of the network's projection blocks, increasing the risk of catastrophic forgetting and benign over-refusal.

**CircuitKavach-Aya** demonstrates that **sparse, circuit-masked parameter transplantation ($C\text{-}\Delta\Theta$)** matches global alignment safety while leaving **97.45% of parameters bit-for-bit frozen**.

```
                  ┌───────────────────────────────────────────────┐
                  │ CohereLabs/tiny-aya-fire (3.35B Base Model)  │
                  └───────────────────────┬───────────────────────┘
                                          │
                  Stage 1: Contrastive Curated Twin Dataset (CuratorKIT)
                                          │
                                          ▼
                  Stage 2: EAP-IG Circuit Discovery (CircuitKIT)
                  Discovered Top-4 Safety Submodules (2.55% params):
                  • layers.0.mlp       • layers.16.self_attn
                  • layers.21.self_attn • layers.20.self_attn
                                          │
                                          ▼
                  Stage 3: SafeGrad + C-ΔΘ Transplant (SafeTune)
                  Frozen: 68/72 Blocks (97.45%) | Updated: 4/72 Blocks (2.55%)
                                          │
                                          ▼
                  Stage 4: AuditKIT Decoupled Dual-Phase Evaluation
                  Harmful Refusal: 94.5% (+54.5 pp) | Over-refusal: 0.0%
```

### Key Verification Metrics (Held-Out 400-Prompt Benchmark)

| Checkpoint | Modified Params | Frozen Params | Proj Blocks | Attack Refusal | Benign Over-Refusal | Benign Helpfulness | Selectivity Gap | RedTeam Pass |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`checkpoint_1_base`** (`tiny-aya-fire`) | 0.00% | 100.00% | 0/72 | 40.0% | 0.0% | 100.0% | +0.400 | 19.0% |
| **`checkpoint_2_global_safetune`** | 100.00% | 0.00% | 72/72 | 99.0% | 0.0% | 100.0% | +0.990 | 47.6% |
| **`checkpoint_3_surgical_ctheta`** | **2.55%** | **97.45%** | **4/72** | **94.5%** | **0.0%** | **100.0%** | **+0.945** | **38.1%** |

*Paired bootstrap significance test on held-out harmful prompts ($n=200$ pairs): $p = 0.000999$ ($+54.5$ percentage points over base). Benign over-refusal remains $0.0\%$ ($p = 1.0$ vs base).*

### Per-Language Attack Refusal Rate

| Language Register | Base (`tiny-aya-fire`) | Global SafeTune (72/72) | **Surgical $C\text{-}\Delta\Theta$ (4/72)** | Surgical Gain vs Base |
| :--- | :---: | :---: | :---: | :---: |
| **English (`en`)** | 100.0% | 100.0% | **100.0%** | +0.0 pp (at ceiling) |
| **Hindi-English (`hi_en`)** | 48.0% | 100.0% | **100.0%** | **+52.0 pp** |
| **Gujarati-English (`gu_en`)** | 6.0% | 100.0% | **100.0%** | **+94.0 pp** |
| **Kannada-English (`kn_en`)** | 6.0% | 96.0% | **78.0%** | **+72.0 pp** |

---

## 2. Public Submission Artifacts & Collections

All artifacts, models, datasets, spaces, and code are public and un-gated:

| Artifact Type | Name / Scope | Public URL | Status |
| :--- | :--- | :--- | :--- |
| **Hugging Face Model Collection** | `CircuitKavach-Aya: Surgical Safety Repair Models & Adapters` | [HF Collection](https://huggingface.co/collections/soapyyapps/circuitkavach-aya-surgical-safety-repair-models-and-adapters-6af9a5f97e17226835ac6f71) | Public |
| • Base Reference Model | `CohereLabs/tiny-aya-fire` (3.35B) | [CohereLabs/tiny-aya-fire](https://huggingface.co/CohereLabs/tiny-aya-fire) | Public |
| • PEFT LoRA Adapter | `soapyyapps/tiny-aya-fire-circuit-kavach-lora` | [HF Model](https://huggingface.co/soapyyapps/tiny-aya-fire-circuit-kavach-lora) | Public |
| • Global SafeTune Baseline | `soapyyapps/tiny-aya-fire-global-safetune` | [HF Model](https://huggingface.co/soapyyapps/tiny-aya-fire-global-safetune) | Public |
| • Surgical $C\text{-}\Delta\Theta$ Transplant | `soapyyapps/tiny-aya-fire-circuit-kavach-merged` | [HF Model](https://huggingface.co/soapyyapps/tiny-aya-fire-circuit-kavach-merged) | Public |
| **Hugging Face Dataset Collection** | `CircuitKavach-Aya: Datasets, Circuits & AuditKIT Logs` | [HF Collection](https://huggingface.co/collections/soapyyapps/circuitkavach-aya-datasets-circuits-and-auditkit-logs-6af9a5fb037f3d7eaf674f49) | Public |
| • Curated Multi-Split Dataset | `soapyyapps/circuit-kavach-cyber-codemixed` | [HF Dataset](https://huggingface.co/datasets/soapyyapps/circuit-kavach-cyber-codemixed) | Public |
| • Interactive Demo Space | `soapyyapps/circuit-kavach-aya` | [HF Space](https://huggingface.co/spaces/soapyyapps/circuit-kavach-aya) | Running |
| **GitHub Repository** | `dptel22/lexsi-lab-track2` | [GitHub Repo](https://github.com/dptel22/lexsi-lab-track2) | Public |

---

## 3. Four-Stage Architecture & Toolkits

Every stage is fully traceable via SHA-256 `lexsi.provenance/1` manifests across the four official Lexsi toolkits:

### Stage 1: Dataset Curation (`CuratorKIT` — `src/stage1_curate.py`)
- Curates **300 contrastive lexical twin pairs** (100 calibration + 200 disjoint held-out evaluation pairs) across 4 languages (`en`, `hi_en`, `gu_en`, `kn_en`) and 3 cybersecurity threat categories (Phishing/Credential Harvesting, Malware Delivery/Recon, Web/API Exploitation).
- Each lexical twin matches exact cybersecurity keywords, syntax, and length between harmful attacks and benign defensive queries.

### Stage 2: Circuit Discovery (`CircuitKIT` — `src/stage2_discover.py`)
- Employs **Edge Attribution Patching with Integrated Gradients (EAP-IG)** on 99 clean/corrupted contrastive pairs ($m(\text{clean}) - m(\text{corrupt}) > 0.5$).
- Calibrates on assistant refusal tokens (`" sorry"` vs. `" when"`).
- Identifies the core refusal transmission submodules:
  - `model.layers.0.mlp` (EAP-IG score `7.031`)
  - `model.layers.16.self_attn` (EAP-IG score `2.489`)
  - `model.layers.21.self_attn` (EAP-IG score `1.944`)
  - `model.layers.20.self_attn` (EAP-IG score `1.740`)
- Verifies faithfulness: `ablation_score = 0.6549`, `mean_jaccard = 0.9683`, `mean_spearman = 0.9998`.

### Stage 3: Surgical Repair & Parameter Transplant (`SafeTune` — `src/stage3_repair.py`)
- Trains a LoRA adapter with SafeGrad gradient projection (`alignment_loss_mode="sft"`, $\rho=1.0$, 3 epochs).
- Uses a **two-group AdamW schedule**: full learning rate ($2\times 10^{-4}$) inside the circuit mask; dampening ($0.10\times$) outside.
- Performs surgical transplantation $C\text{-}\Delta\Theta$:
  $$\Theta_{\text{repaired}} = \Theta_{\text{base}} + \alpha \cdot (M \odot \Delta\Theta)$$
- Sweeps $\alpha \in \{0.5, 1.0, 1.5, 2.0\}$ on held-out validation; selects $\alpha^* = 1.0$.
- Bit-identity assertion: 276 of 291 weight tensors ($97.45\%$ of parameters) are bit-for-bit identical to the base model (`unmasked_violations: []`).

### Stage 4: Decoupled Evaluation & Certification (`AuditKIT` — `src/stage4_eval.py`)
- **Phase 1 (Inference Engine):** Runs deterministic generation and red-teaming (prompt injection, jailbreak, obfuscation) per checkpoint, freeing VRAM after each run.
- **Phase 2 (Scoring & Benchmarks):** Evaluates refusal, over-refusal, and helpfulness with an isolated DiskCache; computes McNemar and paired bootstrap statistics; validates output schemas via Pydantic v2.

---

## 4. Root Cause Analysis: From v1 to v2 (+54.0 pp Leap)

During initial experiments (v1), the surgical repair model achieved only $40.5\%$ refusal. Through systematic debugging and Goldfish review loops, five root causes (`RC1`–`RC5`) were diagnosed and resolved:

```
[v1: 40.5% Refusal] ──► [RC1: Loss Mode] ──► [RC2: Chat Template] ──► [RC3: Seq Alignment] ──► [RC4: LR Grouping] ──► [RC5: Alpha Sweep] ──► [v2: 94.5% Refusal]
```

1. **`RC1` (Alignment Loss Mode):** Upstream `SafeGradHFTrainer` defaulted to DPO/KL loss without positive paired response tokens, diluting gradient direction. **Fix:** Added `alignment_loss_mode="sft"` with safe gradient projection.
2. **`RC2` (`dict`-typed Chat Template Handling):** Upstream `CircuitKIT._has_chat_template` crashed on Cohere's dict-structured tokenizer chat template. **Fix:** Added defensive inspection handling both dict and string templates.
3. **`RC3` (Preamble & Sequence Alignment):** Cohere's ~360-token default preamble caused positional misalignment in contrastive pairs. **Fix:** Configured sequence trimming and pair length equalization.
4. **`RC4` (Circuit-Prioritized AdamW Learning Rates):** Uniform gradient updates dispersed alignment across irrelevant layers. **Fix:** Two-group AdamW prioritizing circuit nodes ($1.0\times$ vs $0.10\times$).
5. **`RC5` (Transplant Strength Sweep):** Default $\alpha=0.5$ under-scaled weight deltas. **Fix:** Validation sweep selected optimal $\alpha^* = 1.0$.

Both upstream bug fixes are exported into tracked patch files (`patches/`) and guarded with defensive runtime monkey-patches in `src/`.

---

## 5. Limitations & Future Directions

1. **Low-Resource Dravidian Generalization:** While English, Hinglish, and Gujlish reached **100.0%** attack refusal, Kannada code-mixed (`kn_en`) achieved **78.0%**. Kannada tokens exhibit higher fragmentation under the tokenizer vocabulary, requiring deeper multi-lingual circuit routing.
2. **Context Window Expansion:** Stage 2 EAP-IG was bounded to `max_seq_len=512` to maintain tractable integrated gradients across attention heads on an NVIDIA L40S. Expanding to long-context multi-turn conversations is a promising next step.
3. **Automated Multi-Token Steering:** Current intervention is a permanent weight-level transplant ($C\text{-}\Delta\Theta$). Coupling this with runtime activation steering during generation could provide defense-in-depth against complex multi-turn jailbreaks.

---

## 6. Standalone Local Reproduction Guide

### Environment Setup
```bash
# Clone the repository
git clone https://github.com/dptel22/lexsi-lab-track2.git
cd lexsi-lab-track2

# Create virtual environment and install dependencies
uv venv .venv --python 3.12
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e vendor/CuratorKIT -e vendor/CircuitKIT -e vendor/SafeTune -e vendor/AuditKIT
uv pip install -e .
```

### Applying Vendor Patches (Optional if using `src/` runtime patches)
```bash
# Apply tracked patches if inspecting vendor source directly:
git -C vendor/CircuitKIT apply ../../patches/circuitkit_rc2_dict_chat_template.patch
git -C vendor/SafeTune apply ../../patches/safetune_rc1_sft_alignment_mode.patch
```

### Running Local CPU Verification Tests
```bash
# Run unit tests on CPU (uses "gpt2" and synthetic test models; zero 3.35B VRAM required)
pytest tests/ -q
```

### Reproducing on Cloud GPU (Hugging Face Jobs)
Set `HF_TOKEN` in your environment and launch the automated jobs:
```bash
bash scripts/run_hf_jobs.sh stage2   # NVIDIA L40S 48GB (Circuit Discovery)
bash scripts/run_hf_jobs.sh stage3   # NVIDIA L4 24GB (SafeGrad + C-ΔΘ Repair)
bash scripts/run_hf_jobs.sh stage4   # NVIDIA L4 24GB (AuditKIT Decoupled Evaluation)
```

---

## 7. Citation & Acknowledgments

```bibtex
@software{circuitkavach_aya_2026,
  author = {Dhruv Patel},
  title = {CircuitKavach-Aya: Surgical Multilingual Safety Alignment via Circuit-Masked Parameter Transplantation},
  year = {2026},
  url = {https://github.com/dptel22/lexsi-lab-track2}
}
```
Developed with the Lexsi Lab toolkits: `CuratorKIT`, `CircuitKIT`, `SafeTune`, and `AuditKIT`.
