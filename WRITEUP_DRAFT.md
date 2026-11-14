# CircuitKavach-Aya: Targeted Multilingual Safety Repair (`lexsi-lab-track2`)

> **Status:** Verified on CPU with `"gpt2"` (`float32`) and tiny synthetic `cohere2` checkpoints. Full `CohereLabs/tiny-aya-fire` (3.35B) Stage 2–4 empirical metrics are marked `[PENDING L4 GPU RUN]` until executed on NVIDIA L4 (`l4x1`) or Colab Pro.

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
   - Preserves single-BOS (Beginning-of-Sequence) chat-template formatting, equalizes within-pair token lengths so left-padding does not misalign token positions during attribution, calibrates refusal/compliance answer tokens, and filters pairs with baseline logit-difference margin $\Delta m = m(\text{clean}) - m(\text{corrupt}) > 0.5$ where $m(x) = \text{logit}(\text{refusal}) - \text{logit}(\text{compliance})$ on harmful (`clean`) vs. benign (`corrupt`) prompts.
   - Runs **EAP-IG** (Edge Attribution Patching with Integrated Gradients, aggregated to node/submodule scores) attribution, evaluates 4 faithfulness metrics (sufficiency, completeness, minimality/sparsity, stability) plus a KL (Kullback–Leibler) divergence diagnostic against the base model, and exports the top-4/72 submodule mask (`artifacts/stage2_circuits/safety_circuit_surgical_submodules_top4_scores.json`).
3. **Stage 3 (`src/stage3_repair.py` — `SafeTune`, hardening and transplant):**
   - Starts from **Checkpoint 1** (`checkpoint_1_base`, unmodified `CohereLabs/tiny-aya-fire`), trains a single-wrapped LoRA adapter via `SafeGradHFTrainer` (gradient-projected safety + utility fine-tuning, with `--ref-offload-cpu` and `--gradient-checkpointing` for 16–24 GB GPUs), saves the unmerged LoRA adapter, restores `use_cache=True`, and merges **Checkpoint 2** (`checkpoint_2_global_safetune`, deltas baked into all 72 projection blocks).
   - Sweeps transplant strength $\alpha$ on the validation split, exports **Checkpoint 3** (`checkpoint_3_surgical_ctheta`), and asserts bit-for-bit equality (`torch.equal`) across all unmasked parameters.
4. **Stage 4 (`src/stage4_eval.py` — `AuditKIT`, decoupled evaluation):**
   - **Phase 1** loads one checkpoint at a time, records deterministic greedy generations, `RedTeamRunner` adversarial probes (prompt injection, jailbreak, encoding), and latency/throughput (`wall_time_s`, `mean_latency_ms`, `tokens_per_sec`), then unloads VRAM.
   - **Phase 2** scores cached generations with `AuditKIT` (`DiskCache` isolated by sample digest, model revision, dtype, device, `max_new_tokens`, and `n_examples`), runs paired bootstrap significance tests, supports optional `WildGuard` safety classification and `lm-eval` capability benchmarks, and validates `eval_summary.json` via Pydantic v2 (`Stage4EvalSummarySchema`).

## 3. Hardware & Reproducibility Strategy

| Environment | Hardware | Purpose |
| :--- | :--- | :--- |
| **Local Laptop** | RTX 4050 6 GB / CPU-only PyTorch (`6.44 GB` free RAM) | CPU verification with `"gpt2"` (`float32`) and tiny synthetic `cohere2` checkpoints; host memory checked via `ml-hardware-preflight`. |
| **HF Jobs (`l4x1`)** | NVIDIA L4 24 GB (`bfloat16`, `templated=True`, `max_seq_len=512`) | Primary cloud compute for `CohereLabs/tiny-aya-fire` Stages 2 → 3 → 4 via `scripts/run_hf_jobs.sh` (`--hydrate-hf --upload-hf`). |
| **HF ZeroGPU (`zero-a10g`)** | Gradio Space (`space/app.py`) | Public interactive demo loading one base model at module scope and displaying Stage 4 trade-off tables and live `tok/s` throughput. |

## 4. Empirical Results (`CohereLabs/tiny-aya-fire`)

**Selectivity Gap** is defined as `Attack Refusal - Benign Over-Refusal` (higher is better).

| Checkpoint | Modified Params | Frozen Params | Proj Blocks | Attack Refusal | Benign Over-Refusal | Benign Helpfulness | Selectivity Gap | RedTeam Attack Pass | Tokens/s |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `checkpoint_1_base` | `0.00%` | `100.00%` | `0/72` | `[PENDING L4 GPU RUN]` | `[PENDING L4 GPU RUN]` | `[PENDING L4 GPU RUN]` | `[PENDING L4 GPU RUN]` | `[PENDING L4 GPU RUN]` | `[PENDING L4 GPU RUN]` |
| `checkpoint_2_global_safetune` | `[PENDING L4 GPU RUN]` | `[PENDING L4 GPU RUN]` | `72/72` | `[PENDING L4 GPU RUN]` | `[PENDING L4 GPU RUN]` | `[PENDING L4 GPU RUN]` | `[PENDING L4 GPU RUN]` | `[PENDING L4 GPU RUN]` | `[PENDING L4 GPU RUN]` |
| `checkpoint_3_surgical_ctheta` | `[PENDING L4 GPU RUN]` | `[PENDING L4 GPU RUN]` | `4/72` | `[PENDING L4 GPU RUN]` | `[PENDING L4 GPU RUN]` | `[PENDING L4 GPU RUN]` | `[PENDING L4 GPU RUN]` | `[PENDING L4 GPU RUN]` | `[PENDING L4 GPU RUN]` |

## 5. Limitations & Execution Sequence

- Local `pytest` verifies all Stage 1–4 invariants on `"gpt2"` and tiny synthetic `cohere2`, not 3.35B weights.
- Stage 1 data is already published at [`soapyyapps/circuit-kavach-cyber-codemixed`](https://huggingface.co/datasets/soapyyapps/circuit-kavach-cyber-codemixed) (`src/stage1_curate.py` regenerates/re-uploads it if needed).
- `scripts/run_hf_jobs.sh` requires an `HF_TOKEN` with `Jobs: read & write` permissions enabled, and passes `--hydrate-hf --upload-hf` so each container fetches upstream artifacts and persists outputs to the Hub:
  ```bash
  bash scripts/run_hf_jobs.sh stage2
  bash scripts/run_hf_jobs.sh stage3
  bash scripts/run_hf_jobs.sh stage4
  ```
