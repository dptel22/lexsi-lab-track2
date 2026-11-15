# CircuitKavach-Aya: Targeted Multilingual Safety Repair (`lexsi-lab-track2`)

> **Status:** Executed and verified across Stages 1–4 on `CohereLabs/tiny-aya-fire` (3.35B) on Hugging Face Jobs (`l40sx1` for Stage 2 templated EAP-IG circuit discovery; `l4x1` for Stage 3 SafeGrad + `C-ΔΘ` repair and Stage 4 `AuditKIT` evaluation). All empirical metrics below reflect verified v2 artifacts (`6af96d1629dd05ff03686a08`, `6af97a2723985fb9cd8320bb`, `6af97fa323985fb9cd8325eb`).

## 1. Problem Statement & Motivation

Multilingual safety alignment frequently degrades in **code-mixed** registers (e.g., Hinglish `hi_en`, Gujlish `gu_en`, Kanglish `kn_en`) where users mix romanized Indic grammar with English cybersecurity terminology. Global fine-tuning (LoRA adapters merged across all layers) can restore refusal on harmful prompts, but often increases **over-refusal** on benign defensive queries and modifies all 72 attention/MLP projection blocks.

**CircuitKavach-Aya** tests whether sparse **circuit-masked parameter transplantation (`C-ΔΘ`)** on `CohereLabs/tiny-aya-fire` (`cohere2` 3.35B: 36 layers, 72 attention and MLP submodules) can match or exceed global fine-tuning safety while keeping **68 of 72 attention/MLP submodules (`94.44%`) bit-for-bit frozen**:

$$\Theta_{\text{repaired}} = \Theta_{\text{base}} + \alpha \cdot (M \odot \Delta\Theta), \qquad \Delta\Theta = \Theta_{\text{global}} - \Theta_{\text{base}}$$

where $M$ is the binary mask selecting the top 4 safety-relevant submodules and $\alpha$ is a scalar transplant strength tuned on a held-out validation split.

## 2. Four-Stage Architecture & Provenance Chain

Every stage writes deterministic artifacts linked via `lexsi.provenance/1` SHA-256 input/output manifests (`src/` uses the vendored Lexsi toolkits `CuratorKIT`, `CircuitKIT`, `SafeTune`, and `AuditKIT`):

1. **Stage 1 (`src/stage1_curate.py` — `CuratorKIT`, dataset curation):**
   - Generates **300** stratified **lexical-twin contrast pairs** (`100` calibration + `200` disjoint held-out evaluation) across `4` languages (`en`, `hi_en`, `gu_en`, `kn_en`) × `3` cyber threat categories. Each lexical twin pairs a harmful prompt and a benign-defensive prompt sharing the same cybersecurity keywords and sentence structure.
   - Runs deduplication and lexical-twin checks before publishing to [`soapyyapps/circuit-kavach-cyber-codemixed`](https://huggingface.co/datasets/soapyyapps/circuit-kavach-cyber-codemixed).
2. **Stage 2 (`src/stage2_discover.py` — `CircuitKIT`, circuit discovery):**
   - Accepts `dict`-typed `CohereTokenizerFast.chat_template`, preserves single-BOS chat-template formatting (`templated=True`, `prepend_bos=False`, `max_seq_len=512`), equalizes within-pair token lengths so left-padding does not misalign positions during attribution, weights `"en"` preflight pairs `3.0x` inside `calibrate_answer_tokens` to anchor on assistant refusal tokens (`" sorry"` vs `" when"`), and filters pairs with baseline logit-difference margin $\Delta m = m(\text{clean}) - m(\text{corrupt}) > 0.5$ (`99/100` selected: `en=25`, `gu_en=25`, `hi_en=25`, `kn_en=24`).
   - Runs **EAP-IG** (Edge Attribution Patching with Integrated Gradients, aggregated to node/submodule scores) attribution on `l40sx1` (48 GB VRAM to accommodate Cohere's ~375-token contextual safety chat preamble), evaluates 4 faithfulness metrics (`ablation_score=0.6549`, `mean_jaccard=0.9683`, `mean_spearman=0.9998`, `61` stable nodes across 3 runs) plus a KL divergence diagnostic, and exports the top-4/72 submodule mask (`artifacts/stage2_circuits/safety_circuit_surgical_submodules_top4_scores.json`).
3. **Stage 3 (`src/stage3_repair.py` — `SafeTune`, hardening and transplant):**
   - Starts from **Checkpoint 1** (`checkpoint_1_base`, unmodified `CohereLabs/tiny-aya-fire`), trains a single-wrapped LoRA adapter via `SafeGradHFTrainer` with supervised refusal gradient surgery (`alignment_loss_mode="sft"`, `rho=1.0`, `epochs=3.0`, CPU-offloaded reference model consuming `0 GB` GPU VRAM, `--gradient-checkpointing`), and two-group `AdamW` learning rates (`2e-4` inside circuit submodules vs. `0.10x` outside the circuit), saves the unmerged LoRA adapter, restores `use_cache=True`, and merges **Checkpoint 2** (`checkpoint_2_global_safetune`, deltas baked into all 72 projection blocks).
   - Sweeps transplant strength $\alpha \in \{0.5, 1.0, 1.5, 2.0\}$ on the held-out validation split (`best_alpha=1.0`: `95.0%` validation attack refusal, `0.0%` benign over-refusal), exports **Checkpoint 3** (`checkpoint_3_surgical_ctheta`), and asserts bit-for-bit equality (`torch.equal`) across all `276` unmasked weight tensors.
4. **Stage 4 (`src/stage4_eval.py` — `AuditKIT`, decoupled evaluation):**
   - **Phase 1** loads one checkpoint at a time, records deterministic greedy generations, `RedTeamRunner` adversarial probes (prompt injection, jailbreak, encoding), and latency/throughput (`wall_time_s`, `mean_latency_ms`, `tokens_per_sec`), then unloads VRAM.
   - **Phase 2** scores cached generations with `AuditKIT` (`DiskCache` isolated by sample digest, model revision, dtype, device, `max_new_tokens`, and `n_examples`), runs paired bootstrap significance tests, supports optional `WildGuard` safety classification and `lm-eval` capability benchmarks, and validates `eval_summary.json` via Pydantic v2 (`Stage4EvalSummarySchema`).

## 3. Hardware & Reproducibility Strategy

| Environment | Hardware | Purpose |
| :--- | :--- | :--- |
| **Local Laptop** | RTX 4050 6 GB / CPU-only PyTorch (`6.44 GB` free RAM) | CPU verification with `"gpt2"` (`float32`) and tiny synthetic `cohere2` checkpoints; host memory checked via `ml-hardware-preflight`. |
| **HF Jobs (`l40sx1` / `l4x1`)** | NVIDIA L40S 48 GB (Stage 2) & L4 24 GB (Stages 3–4) | Cloud compute (`bfloat16`, `templated=True`, `max_seq_len=512`) for `CohereLabs/tiny-aya-fire` via `scripts/run_hf_jobs.sh` (`--hydrate-hf --upload-hf`). Stage 2 uses `l40sx1` because Cohere's ~360-token contextual safety chat preamble expands `HookedTransformer` `use_attn_result` EAP-IG peak to `21.8–27.1 GiB`. |
| **HF ZeroGPU (`zero-a10g`)** | Gradio Space (`space/app.py`) | Public interactive demo loading one base model at module scope and displaying Stage 4 trade-off tables and live `tok/s` throughput. |

## 4. Empirical Results (`CohereLabs/tiny-aya-fire`, v2 Verified Run)

**Selectivity Gap** is defined as `Attack Refusal - Benign Over-Refusal` (higher is better).

| Checkpoint | Modified Params | Frozen Params | Proj Blocks | Attack Refusal | Benign Over-Refusal | Benign Helpfulness | Selectivity Gap | RedTeam Attack Pass | Tokens/s |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `checkpoint_1_base` | 0.00% | 100.00% | 0/72 | 40.0% | 0.0% | 100.0% | +0.400 | 19.0% | 20.92 |
| `checkpoint_2_global_safetune` | 100.00% (LoRA all-layer) | 0.00% | 72/72 | 99.0% | 0.0% | 100.0% | +0.990 | 47.6% | 20.23 |
| `checkpoint_3_surgical_ctheta` | **2.55%** | **97.45%** | **4/72** | **94.5%** | **0.0%** | **100.0%** | **+0.945** | **38.1%** | 20.79 |

Paired bootstrap comparison on held-out harmful refusal (`n=200` pairs) confirms that `checkpoint_3_surgical_ctheta` outperforms `checkpoint_1_base` by **+54.5 percentage points** (`p = 0.000999`, statistically significant), recovering **92.4%** of the global 72-block refusal gain while keeping **97.45%** of weights bit-identical and maintaining **0.0%** benign over-refusal across all `200` benign twins (`p = 1.0` vs. base).

### Per-Register Breakdown (`Attack Refusal Rate`)

| Language Register | Checkpoint 1 (Base) | Checkpoint 2 (Global LoRA, 72/72) | Checkpoint 3 (Surgical C-ΔΘ, 4/72) | Surgical Gain vs. Base |
| :--- | ---: | ---: | ---: | ---: |
| **English (`en`)** | 100.0% | 100.0% | 100.0% | +0.0% (at ceiling) |
| **Hindi-English (`hi_en`)** | 48.0% | 100.0% | **100.0%** | **+52.0%** |
| **Gujarati-English (`gu_en`)** | 6.0% | 100.0% | **100.0%** | **+94.0%** |
| **Kannada-English (`kn_en`)** | 6.0% | 96.0% | **78.0%** | **+72.0%** |

### Per-Threat-Category Breakdown (`Attack Refusal Rate`)

| Threat Category | Pairs | Checkpoint 1 (Base) | Checkpoint 2 (Global LoRA) | Checkpoint 3 (Surgical C-ΔΘ) |
| :--- | ---: | ---: | ---: | ---: |
| **Credential / Financial Phishing** | 68 | 27.9% | 97.1% | **85.3%** |
| **Recon / Malware Delivery** | 64 | 50.0% | 100.0% | **98.4%** |
| **Web / API Exploitation** | 68 | 42.6% | 100.0% | **100.0%** |

### Locality & Bit-Identity Verification
- **Verified Unmasked Keys:** 100% bit-identical (`276/291` tensors untouched; `unmasked_violations: []`).
- **Modified Submodules (Top 4, `best_alpha = 1.0`):**
  - `model.layers.0.mlp` (`gate_proj`, `up_proj`, `down_proj` — EAP-IG score `7.03125`)
  - `model.layers.16.self_attn` (`q_proj`, `k_proj`, `v_proj`, `o_proj` — EAP-IG score `2.48931`)
  - `model.layers.21.self_attn` (`q_proj`, `k_proj`, `v_proj`, `o_proj` — EAP-IG score `1.94449`)
  - `model.layers.20.self_attn` (`q_proj`, `k_proj`, `v_proj`, `o_proj` — EAP-IG score `1.73963`)
- **Total Modified Parameters:** `99,090,432` of `3,886,098,432` (**2.55%**), keeping **97.45%** of all model parameters strictly frozen.

## 5. Limitations & Execution Sequence

- Local `pytest` verifies all Stage 1–4 invariants on `"gpt2"` and tiny synthetic `cohere2`, not 3.35B weights.
- Stage 1 data is already published at [`soapyyapps/circuit-kavach-cyber-codemixed`](https://huggingface.co/datasets/soapyyapps/circuit-kavach-cyber-codemixed) (`src/stage1_curate.py` regenerates/re-uploads it if needed).
- `scripts/run_hf_jobs.sh` requires an `HF_TOKEN` with `Jobs: read & write` permissions enabled, and passes `--hydrate-hf --upload-hf` so each container fetches upstream artifacts and persists outputs to the Hub:
  ```bash
  bash scripts/run_hf_jobs.sh stage2
  bash scripts/run_hf_jobs.sh stage3
  bash scripts/run_hf_jobs.sh stage4
  ```
