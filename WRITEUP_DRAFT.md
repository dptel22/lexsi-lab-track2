# CircuitKavach-Aya: Submission Summary (`lexsi-lab-track2`)

> **Full Publication-Grade Article:** [`docs/submission_article.md`](docs/submission_article.md) · **Root README:** [`README.md`](README.md) · **Concepts & `RC1`–`RC5` Brief:** [`docs/concepts_brief.md`](docs/concepts_brief.md)

**One-line claim:** Modifying just **4 of 72 transformer submodules (`2.55%` of parameters)** discovered via EAP-IG restores **94.5%–99.0% code-mixed cyberattack refusal** across four 3.35B `CohereLabs/tiny-aya` models (`fire`, `global`, `water`, `earth`) while holding **benign over-refusal at `0.0%`** and keeping **`97.45%` of weights bit-for-bit frozen**.

---

## Abstract

Multilingual safety alignment frequently breaks down in **code-mixed** registers where attackers weave English cyberattack keywords into romanized Indic grammar—Hinglish (`hi_en`), Gujlish (`gu_en`), and Kanglish (`kn_en`). Across all four 3.35B `CohereLabs/tiny-aya` models (`fire`, `global`, `water`, `earth`), baseline checkpoints refuse **100.0%** of standard English cyberattack prompts (`en`) yet drop to **48.0%–66.0%** in Hinglish, **6.0%–16.0%** in Gujlish, and **6.0%–38.0%** in Kanglish (`40.0%–53.5%` overall). Global fine-tuning restores refusal (`99.0%–100.0%`), but perturbs all 72 attention and MLP projection blocks.

**CircuitKavach-Aya** executes a deterministic four-stage pipeline integrating **CuratorKIT**, **CircuitKIT**, **SafeTune**, and **AuditKIT** with SHA-256 `lexsi.provenance/1` tracking. Using **300** contrastive lexical-twin pairs (`100` calibration, `200` held-out evaluation) where harmful and benign prompts share identical cybersecurity keywords and prefixes, we apply templated Edge Attribution Patching with Integrated Gradients (EAP-IG), supervised refusal gradient surgery (`SafeGrad`, `alignment_loss_mode="sft"`), and sparse circuit-masked parameter transplantation ($C\text{-}\Delta\Theta$).

Across all four `tiny-aya` models, EAP-IG independently discovers the **exact same 4 of 72 submodules** (`layers.0.mlp`, `layers.16.self_attn`, `layers.20.self_attn`, `layers.21.self_attn`), achieving **1.000 pairwise top-4 Jaccard similarity** and **0.9217–0.9704 72-block Spearman rank correlation**. Transplanting deltas exclusively into those 4 blocks (`99,090,432` of `3,886,098,432` parameters, **`2.55%`**) keeps **`276` of `291` tensors (`97.45%`) bit-for-bit identical** while raising held-out refusal from **`40.0%` → `94.5%`** (`fire`), **`45.5%` → `96.5%`** (`global`), **`53.5%` → `96.5%`** (`water`), and **`53.5%` → `99.0%`** (`earth`) ($p = 0.000999$ vs. base via paired bootstrap), holding benign over-refusal at **`0.0%`** and unique-token diversity at **`0.9701–0.9810`**. Furthermore, we observe a **Regional Specialization Paradox**: South Asia specialist `tiny-aya-fire` carries the strongest Layer 0 MLP lexical attribution (`7.0312` vs. `2.2188` on `earth`) yet exhibits the lowest base refusal (`40.0%`) and hardest Kanglish tail (`78.0%`), whereas `layers.16.self_attn` remains invariant (`2.489–2.867`).

---

## 1. Verified 4-Model Empirical Results

### 1.1 Safety, Locality & Selectivity Matrix (`400` Held-Out Prompts + `89` RedTeam Checks per Model)

| Model (`CohereLabs/tiny-aya-*`) | Base Attack Refusal | Global SafeTune (`72/72`, Over-Refusal) | **Surgical `C-ΔΘ` (`4/72`, Over-Refusal)** | Selectivity Gap | Modified Params (`Frozen`) | Best `α` | Bootstrap `p` vs. Base (vs. Global) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`fire`** | 40.0% | 99.0% (`0.0%`) | **94.5%** (`0.0%`) | **+0.945** | **2.55%** (`97.45%`) | `1.0` | `0.000999` (`0.003996`) |
| **`global`** | 45.5% | 99.0% (`0.0%`) | **96.5%** (`0.0%`) | **+0.965** | **2.55%** (`97.45%`) | `1.0` | `0.000999` (`0.017982`) |
| **`water`** | 53.5% | 100.0% (`0.0%`) | **96.5%** (`0.0%`) | **+0.965** | **2.55%** (`97.45%`) | `1.0` | `0.000999` (`0.007992`) |
| **`earth`** | 53.5% | 99.5% (`0.0%`) | **99.0%** (`0.0%`) | **+0.990** | **2.55%** (`97.45%`) | `1.0` | `0.000999` (`0.612388` parity) |

### 1.2 Per-Language & Per-Category Breakdown

| Model | `en` (Base → Global → **Surg.**) | `hi_en` (Base → Global → **Surg.**) | `gu_en` (Base → Global → **Surg.**) | `kn_en` (Base → Global → **Surg.**) | Phishing (Base → **Surg.**) | Malware (Base → **Surg.**) | Web/API (Base → **Surg.**) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`fire`** | 100% → 100% → **100%** | 48% → 100% → **100%** | 6% → 100% → **100%** | 6% → 96% → **78%** | 27.9% → **85.3%** | 50.0% → **98.4%** | 42.6% → **100.0%** |
| **`global`** | 100% → 100% → **100%** | 58% → 100% → **100%** | 12% → 100% → **100%** | 12% → 96% → **86%** | 25.0% → **89.7%** | 53.1% → **100.0%** | 58.8% → **100.0%** |
| **`water`** | 100% → 100% → **100%** | 66% → 100% → **100%** | 10% → 100% → **98%** | 38% → 100% → **88%** | 26.5% → **91.2%** | 65.6% → **98.4%** | 69.1% → **100.0%** |
| **`earth`** | 100% → 100% → **100%** | 66% → 100% → **100%** | 16% → 98% → **96%** | 32% → 100% → **100%** | 26.5% → **100.0%** | 67.2% → **96.9%** | 67.6% → **100.0%** |

### 1.3 Pairwise Circuit Overlap, Attribution Asymmetry, & Benign Coherence

| Model Pair | Top-4 Jaccard | 72-Block Spearman $\rho$ | Shared Submodules |
| :--- | :---: | :---: | :--- |
| `fire__global` | **1.000** | 0.9373 | `layers.0.mlp`, `layers.16.self_attn`, `layers.20.self_attn`, `layers.21.self_attn` |
| `fire__water` | **1.000** | 0.9217 | `layers.0.mlp`, `layers.16.self_attn`, `layers.20.self_attn`, `layers.21.self_attn` |
| `fire__earth` | **1.000** | 0.9322 | `layers.0.mlp`, `layers.16.self_attn`, `layers.20.self_attn`, `layers.21.self_attn` |
| `global__water` | **1.000** | 0.9469 | `layers.0.mlp`, `layers.16.self_attn`, `layers.20.self_attn`, `layers.21.self_attn` |
| `global__earth` | **1.000** | 0.9378 | `layers.0.mlp`, `layers.16.self_attn`, `layers.20.self_attn`, `layers.21.self_attn` |
| `water__earth` | **1.000** | 0.9704 | `layers.0.mlp`, `layers.16.self_attn`, `layers.20.self_attn`, `layers.21.self_attn` |

| Model | `layers.0.mlp` Score | `layers.16.self_attn` Score | `layers.20.self_attn` Score | `layers.21.self_attn` Score | Benign Non-Empty | Benign Mean Chars | Benign Token Diversity |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`fire`** | **7.0312** | 2.4893 | 1.7396 | 1.9445 | `100.0%` | 243.66 | `0.9737` |
| **`global`** | **4.5625** | 2.5260 | 1.7790 | 1.7541 | `100.0%` | 249.43 | `0.9701` |
| **`water`** | **3.9531** | 2.8666 | 1.9295 | 1.9318 | `100.0%` | 229.38 | `0.9784` |
| **`earth`** | **2.2188** | 2.5253 | 1.8842 | 1.7238 | `100.0%` | 221.10 | `0.9810` |

- **Defensive-Mode-Bleed Residuals:** Across all `800` surgical harmful prompts (`200 × 4` models), **27 / 27 residual misses** (`11` `fire`, `7` `global`, `7` `water`, `2` `earth`) emitted benign defensive advice (`"To protect..."`) rather than generating harmful exploit scripts or phishing templates.
- **Honest Evaluator Status:** `WildGuard` and `lm-eval` hooks in `src/stage4_eval.py` were `"disabled"` in the cloud runs; `RedTeamRunner` encoding probes (`0%–20%`) remain an open defense-in-depth gap.

---

## 2. Key Engineering Fixes (`v1 → v2 → v3`) & Compute ROI

1. **`v1 → v2` (`+54.0 pp` on `fire`):** Diagnosed and resolved `RC1` (`SafeGradHFTrainer` `alignment_loss_mode="sft"`), `RC2` (`dict`-typed `CohereTokenizerFast.chat_template`), `RC3` (`3.0×` `"en"` preflight weighting + length equalization), `RC4` (two-group `AdamW` at `2e-4` vs. `0.10x`), and `RC5` (`alpha` sweep `1.0`). Portable upstream fixes are stored in [`patches/`](patches/).
2. **`v2 → v3` (4-Model Expansion):** Added model-slug directory/provenance isolation, eliminated `l4x1` CPU RAM OOM by skipping `reference_model` loading in `sft` mode, captured `tokenizer` before `unload()` (`28.54–28.82 subword tok/s`), and completed all 4 models on Hugging Face Jobs (`3.489h` `l40sx1` + `4.395h` `l4x1` = **\$9.80 total compute**, ~\$2.45 per model; measured L4 eager inference cost **\$0.22–\$0.31 / 1k replies** vs. \$1.65–\$3.30 / 1k closed frontier API pricing estimates).

---

## 3. Links & Reproduction

- **HF Model Collection:** [`soapyyapps/circuitkavach-aya-surgical-safety-repair-models-and-adapters`](https://huggingface.co/collections/soapyyapps/circuitkavach-aya-surgical-safety-repair-models-and-adapters-6af9a5f97e17226835ac6f71) (12 public checkpoints)
- **HF Dataset Collection:** [`soapyyapps/circuitkavach-aya-datasets-circuits-and-auditkit-logs`](https://huggingface.co/collections/soapyyapps/circuitkavach-aya-datasets-circuits-and-auditkit-logs-6af9a5fb037f3d7eaf674f49)
- **Live ZeroGPU Space:** [`soapyyapps/circuit-kavach-aya`](https://huggingface.co/spaces/soapyyapps/circuit-kavach-aya)
- **GitHub Repository:** [`dptel22/lexsi-lab-track2`](https://github.com/dptel22/lexsi-lab-track2) (`pytest tests/ -q` verifies all 64 CPU unit/integration tests)
