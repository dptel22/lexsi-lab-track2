# Stage 2 Design Specification: Multilingual Safety Circuit Discovery (`src/stage2_discover.py`)

**Date:** 2026-11-11
**Status:** Locked & Approved (Revised with 4 Adversarial Audit Fixes)
**Upstream Input:** `artifacts/stage1_dataset/calibration_100_pairs.jsonl` + `artifacts/stage1_dataset/lexsi_provenance.json`
**Downstream Consumers:** Stage 3 `SafeTune` (`apply_ctheta` / `CThetaTrainer` via `<stem>_surgical_submodules_scores.json`) and Stage 4 `AuditKIT`

---

## 1. Purpose & Architectural Position

Stage 2 discovers, validates, and exports the mechanistic safety refusal circuit of the target LLM (`CohereLabs/tiny-aya-fire` on Colab Pro GPU; `"gpt2"` / synthetic local checkpoints for CPU verification) over the code-mixed cybersecurity contrastive twin pairs curated in Stage 1.

Because downstream Stage 3 (`SafeTune.apply_ctheta`) applies dense matrix deltas (`target[k] += strength * (positive[k] - negative[k])`) across the Cartesian product `(layer_subset x target_modules)`, unfiltered circuit discovery in `circuitkit.artifacts.scores.interop_fields` can saturate 65–85% of a 36-layer model even from scattered single heads. Stage 2 therefore combines **Within-Pair Length Equalization**, **Positive Baseline Margin Preflight Filtering**, **Calibrated Hybrid attribution & evaluation**, and a **Submodule-Ranked Surgical Mask Guard (`--max-surgical-submodules 4`)** so Stage 3 Checkpoint 3 leaves **94.44% of projection blocks strictly frozen** (`4/72`).

---

## 2. Target Hardware & Runtime Profiles

| Environment | Model | `chat_template_mode` | Resolved `templated` | `prepend_bos` (EAP) | `max_seq_len` | `batch_size` | `dtype` |
|---|---|---|---|---|---|---|---|
| HF Jobs Primary (`l4x1` 24 GB VRAM, `--timeout 35m`) / Colab Pro L4 & A100 | `CohereLabs/tiny-aya-fire` (36 layers, GQA `16Q/4KV`, 612 nodes) | `"auto"` (default) | `True` (tokenizer has `chat_template`) | `False` (prevent double-BOS) | `512` | `1` | `bfloat16` (`float16` escape hatch if needed) |
| Free T4 16 GB Backup (`--low-vram-t4` CLI flag) | `CohereLabs/tiny-aya-fire` | Forced `"off"` | `False` (4x lower activation memory) | `True` | Forced `128` | Forced `1` | `bfloat16` (`float16` escape hatch) + `_release_vram()` |
| Local Laptop CPU Verification | `"gpt2"` (12 layers, 156 nodes) | `"auto"` (default) | `False` (no `chat_template`) | `True` | `128` | `1` (overridable via `--batch-size`) | `float32` |

### Runtime Auto-Resolution Rules (`resolve_runtime_profile`)
1. `chat_template_mode` defaults to `"auto"` and is resolved via `circuitkit.tasks._chat.resolve_chat_template(mode, model)`.
2. CLI exposes `--chat-template-mode {auto,on,off}`, `--batch-size`, `--max-seq-len`, `--dtype`, and `--low-vram-t4`.
3. When `--low-vram-t4` is passed, Stage 2 forces `chat_template_mode="off"`, `max_seq_len=128`, `batch_size=1`, and invokes `_release_vram()` (`gc.collect()` + `torch.cuda.empty_cache()`) between Phase A discovery, Phase B `evaluate_circuit()`, and `compute_kl_diagnostic()`.
4. `prepend_bos` in `EAPDiscoveryDataset` and `tokenize_batch_pair` is governed strictly by `not templated` (`templated=True => prepend_bos=False`), preventing double-BOS on Cohere chat templates that already emit `<BOS_TOKEN>`.
5. `import circuitkit` MUST execute before `import transformer_lens` so `circuitkit.backends._tl_compat` registers `cohere2` before `HookedTransformer` initialization.
6. Never load `CohereLabs/tiny-aya-fire` locally (6 GB VRAM / CPU boundary); local tests and smoke verification run strictly on `"gpt2"` (`dtype="float32"`) and tiny local `cohere2` / `gpt2` test fixtures.

---

## 3. Data Preparation, Preflight Token Calibration & Discovery

### 3.1 Contrast Pair Contract, Within-Pair Length Equalization & Alignment (Fix #1 & Fix #4)
- Load `artifacts/stage1_dataset/calibration_100_pairs.jsonl` (fields: `pair_id`, `language`, `threat_category`, `keyword`, `harmful`, `benign`).
- Contrastive direction:
  - `clean = harmful` (safety-refusal trigger)
  - `corrupted = benign` (lexical twin sharing template/prefix with benign intent)
- Normalization strategy: **`align_strategy="none"`** and **`pair_padding_side="left"`** via `circuitkit.data.template.template_normalize`, preserving Stage 1's shared-prefix structure without destructive alignment insertion.
- **Within-Pair Token Length Equalization (Fix #1):** Under `pair_padding_side="left"`, any token-length mismatch between `harmful_i` and `benign_i` causes `tokenize_batch_pair` to prepend `<pad>` tokens only to the shorter side, shifting the entire 366-token system preamble and shared `{keyword}` relative to its twin and poisoning `eap-ig` `(z_corrupt - z_clean) * grad` with positional shift noise. Before preflight calibration, discovery, and evaluation, `equalize_pair_token_lengths` computes template overhead and equalizes wrapped token lengths *within each `(harmful_i, benign_i)` pair* by right-padding the shorter user suffix before the closing assistant turn header (or matching/truncating within budget) with verified tokenizer round-trip checks so that:
  - `clean_lengths[i] == corrupt_lengths[i] <= max_seq_len`
  - `clean_tokens[i, :prefix_len] == corrupt_tokens[i, :prefix_len]` (system preamble + shared user prefix 100% index-aligned)
  - Final assistant generation position `[-1]` (and closing chat turn header) stays 100% index-aligned.
- When `templated=True`, prompts are wrapped identically via `circuitkit.tasks._chat.wrap_prompt(model, text, apply=True)` before tokenization, and `CalibratedSafetyTaskSpec` ensures both `dataloader.templated = templated` and `dataloader.pair_padding_side = "left"` propagate identically through `discover_circuit`, `evaluate_circuit`, and `Pillar3_Stability`. Furthermore, `CalibratedSafetyTaskSpec._build_eap_dataloader` hashes all `(clean_prompt, clean_answer, corrupt_prompt, corrupt_answer)` rows plus `n={len(self.ds)}` into its CSV filename so repeated runs or varied pair subsets under the same `task_name` never collide on stale cached CSVs.
- **No-Double-BOS Invariant (Fix #4):** Verified explicitly on tokenized `NormalizedTaskSpec` batches: whenever `tokens[i, 0] == bos_token_id`, `tokens[i, 1] != bos_token_id` (no consecutive BOS tokens).

### 3.2 16-Pair Preflight Token Calibration & Positive Baseline Margin Filter (Fix #2)
CircuitKIT `Pillar1_CausalPatching` and `Pillar2_Ablation` require a positive, non-degenerate denominator (`y_clean - y_corrupt > 1e-6`), otherwise `_faithfulness_ratio` returns `status="invalid"` (`inverted_denominator=True` or `degenerate_denominator=True`). Moreover, unaligned/code-mixed prompts often bypass base `CohereLabs/tiny-aya-fire`, producing `m(clean) - m(corrupt) <= 0`:
1. Deterministically select up to **16 stratified pairs** across `(language, threat_category)` from the equalized dataset, tokenize via `wrap_prompt` + `prepend_bos=not templated`, run `torch.no_grad()` forward passes, and pick round-trip-valid single non-special tokens `(clean_token_id, corrupt_token_id)` maximizing contrastive log-probability separation. `_valid_single_token_roundtrip` permits meaningful leading-space BPE answer tokens (e.g., `" Sorry"`, `" Here"`) while rejecting empty, replacement-char (`\ufffd`), newline/CR, pure-punctuation, or multi-token re-encodings.
2. **Per-Pair Baseline Margin Filtering (Fix #2):** Using the calibrated `(clean_token_id, corrupt_token_id)`, score **every** pair `i` in the calibration set: `m(clean_i) = logit_clean_i[c] - logit_clean_i[r]`, `m(corrupt_i) = logit_corr_i[c] - logit_corr_i[r]`, `delta_m_i = m(clean_i) - m(corrupt_i)`.
   - Keep pairs with positive refusal separation `delta_m_i > margin_threshold` (`--margin-threshold`, default `0.5`) for `discover_circuit` and 4-pillar `evaluate_circuit`.
   - Deterministic fallback: when fewer than `min(2, len(pairs))` pairs meet `margin_threshold` (e.g. untrained/raw local `"gpt2"` CPU runs), fall back to top-N highest-`delta_m` pairs stratified across languages (preferring `delta_m > 1e-6`).
   - Log and persist `positive_margin_count` by language alongside the held-out `failed_refusal_pairs` (`pair_id`, `language`, `threat_category`, `delta_m`) evidencing Track 2a #4 baseline vulnerability.
3. Verify aggregate `logit_diff` gap over the filtered discovery subset is `> 1e-6` and attach full calibration + margin-filter diagnostics to provenance and reports.

### 3.3 Discovery Algorithm & Sparsity Mapping
- Algorithm: **`eap-ig`** (`EAP-IG-inputs`, `ig_steps=3`, `level="node"`, `scope="both"`).
- Metric: calibrated **`logit_diff`**.
- Note on CircuitKIT polarity:
  - `CircuitScores.to_json(..., top_fraction=f)` / `interop_fields(..., top_fraction=f)` **keeps** the top fraction `f` (default `0.10`, ~`61/612` nodes on `tiny-aya-fire`).
  - `pruning.target_sparsity` in `discover_circuit` / `evaluate_circuit` / `Pillar3_Stability` specifies the fraction **discarded**, so Stage 2 sets `target_sparsity = round(1.0 - top_fraction, 6)` (default `0.90`).

---

## 4. Calibrated Hybrid Faithfulness Evaluation & KL Diagnostic

Stage 2 evaluates the discovered circuit via:
1. **4-Pillar `evaluate_circuit`** (`circuitkit.api.evaluate_circuit`):
   - `pillars = ["patching", "ablation", "baselines", "stability"]`
   - Explicitly **excludes** `"robustness"` (and `"generalization"`).
   - Evaluates the pruned `.pt` artifact and `_scores.pt` under the exact calibrated `NormalizedTaskSpec` (`align_strategy="none"`, `pair_padding_side="left"`, `templated`, `logit_diff`).
2. **Standalone KL Divergence Diagnostic** (`compute_kl_diagnostic`):
   - Runs `circuitkit.evaluation.evaluate.evaluate_graph(model, graph, dataloader, _eap_kl_divergence, intervention="patching", skip_clean=False)` alongside unmodified corrupted baseline KL (`evaluate_baseline(..., run_corrupted=True)`), ensuring `clean_logits` is populated (`skip_clean=False`).
   - Records `kl_circuit_vs_clean`, `kl_corrupt_vs_clean`, and `kl_reduction_ratio = 1.0 - (kl_circuit_vs_clean / kl_corrupt_vs_clean)`.
   - Also inspects full-vocabulary vs calibrated-pair divergence cleanly if `kl_corrupt_vs_clean > 1e-9`.

---

## 5. Surgical Locality Target & Submodule-Aggregated Mask (Fix #3)

In `circuitkit.artifacts.scores.interop_fields`:
- Selecting even 1 of the 16 attention heads `A{L}.{H}` in Layer `L` emits all 4 attention projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`) globally into `target_modules` and adds `L` to `layer_subset`.
- Selecting any `MLP {L}` adds MLP projections (`gate_proj`, `up_proj`, `down_proj`) globally into `target_modules`.
- Because `safetune.interventions.recover.ctheta.apply_ctheta` takes the Cartesian product `layer_subset x target_modules` via `_matched_param_keys`, scattered heads across 36 layers (`72` submodule projection blocks = `36 self_attn + 36 mlp`) rewrite 65–85% of model weights.

### 5.1 Submodule-Ranked Surgical Mask (`--max-surgical-submodules`, default `4`, `< 5%–8%` C-ΔΘ Target)
In addition to exporting standard `interop_fields` at `--top-fraction` (default `0.10`) into `<stem>_scores.json`, Stage 2 builds a **Submodule-Ranked Surgical Mask** exported into `<stem>_surgical_submodules_scores.json` and pre-computed budget tiers `top3`, `top4`, `top6` for Stage 3 Checkpoint 3:
1. Partition the transformer into `2 * n_layers` submodules: `(L, "self_attn")` and `(L, "mlp")` for `L in 0 .. n_layers - 1` (`72` submodules on 36-layer Aya).
2. Score each submodule:
   - `(L, "self_attn")`: ranked by `sum(|node_scores[A{L}.H]|)` (also recording `mean(|node_scores[A{L}.H]|)` in metadata).
   - `(L, "mlp")`: ranked by `|node_scores[MLP {L}]|`.
3. Select only the top `K = max_surgical_submodules` submodules (default **`4` out of `72` = `5.56%` of projection blocks**, leaving **`94.44%` frozen** and aligning with the `< 5%–8%` C-ΔΘ paper budget). Also export `<stem>_surgical_submodules_top{3,4,6}_scores.json` (`4.17%`, `5.56%`, `8.33%` of blocks) in exact `safetune.core.circuit_kit.adapter` (`safety_units` / `layer_suggestions`) schema.
4. Filter `safety_units` to units inside the selected submodules, and emit submodule-qualified `layer_suggestions.target_modules` entries (e.g. `layers.{L}.self_attn.{q,k,v,o}_proj` or `layers.{L}.mlp.{gate,up,down}_proj` on Cohere2/Llama; `h.{L}.attn.{c_attn,c_proj}` / `h.{L}.mlp.{c_fc,c_proj}` on GPT-2) paired with `layer_subset`. Because `_matched_param_keys` matches substring patterns within `layer_subset`, unselected sibling submodules in the same layer are never rewritten.
5. Verify `<stem>_scores.json`, `<stem>_surgical_submodules_scores.json`, and all `top{3,4,6}` tier exports via `safetune.core.circuit_kit.adapter.load_circuit_info_from_file`.

### 5.2 Submodule Saturation Diagnostic Table (`0.05`, `0.10`, `0.20`, and `submodule-top-{3,4,6}`)
Stage 2 prints/logs and persists both:
- `submodule_diagnostic_table`: compares **unique layers touched** and **projection blocks touched** (`N / 72` and `%`) across standard `interop_fields` at `0.05`, `0.10`, `0.20`, and `submodule-top-{3,4,6}` (plus custom `--max-surgical-submodules` if non-default).
- `layer_saturation_table` & `safety_units_top{05,10,20}.json`: legacy layer-cap summary (`--max-surgical-layers`, default `8`) over `[0.05, 0.10, 0.15, 0.20]`.

---

## 6. Artifact Layout, Provenance Chaining & Hugging Face Upload

### 6.1 Output Directory (`--output-dir`, default `artifacts/stage2_circuits`)
- `<stem>.pt` (default stem `safety_circuit`): CircuitKIT pruned circuit checkpoint
- `<stem>_scores.pt`: torch binary node scores (`evaluate_circuit` compatible)
- `<stem>_scores.json`: **full** discovered `node_scores` (`612` on `tiny-aya-fire`) + **standard `interop_fields`** at `--top-fraction` (`0.10`) + `model_revision` SHA + `provenance` + `submodule_diagnostic_table` + `layer_saturation_table` + `margin_filter` + `kl_diagnostic`
- `<stem>_surgical_submodules_scores.json` & `<stem>_surgical_submodules_top{3,4,6}_scores.json`: full `node_scores` + Submodule-Ranked Surgical Masks (`safety_units` + submodule-qualified `layer_suggestions`, default `4/72 = 5.56%` blocks touched) + `model_revision` SHA strictly validated for Stage 3 `apply_ctheta`
- `safety_units_top05.json`, `safety_units_top10.json`, `safety_units_top20.json`: pre-computed layer-capped SafeTune-compatible tier exports
- `faithfulness_report.json`: 4-pillar `FaithfulnessReport` + standalone KL diagnostic + `model_revision` SHA + calibration, margin filter (`positive_margin_count`, `failed_refusal_pairs`), & saturation diagnostics
- `lexsi_provenance.json`: `lexsi.provenance/1` record written via `circuitkit.provenance.write_provenance` capturing `model_revision` SHA and chaining Stage 1's `artifacts/stage1_dataset/lexsi_provenance.json` under `inputs[].provenance`.

### 6.2 Hugging Face Upload & Ephemeral Container Fail-Safe
- `--upload-hf` (default `False` for local CPU safety)
- `--hf-repo` (default `"soapyyapps/circuit-kavach-cyber-codemixed"`, uploading folder to `path_in_repo="stage2_circuits"`, `repo_type="dataset"`, loading token silently via `dotenv.load_dotenv()` + `HF_TOKEN` without ever inspecting `.env`).
- **Fail-Fast Pre-GPU Auth Guard (`verify_hf_auth()`):** Whenever `--upload-hf` is enabled, Stage 2 validates `HF_TOKEN` and executes `huggingface_hub.HfApi(token=...).whoami()` at the very start of `run_stage2()` **before** loading the model or consuming GPU compute units. Missing/invalid credentials abort immediately with a clear error.
- **Incremental Two-Phase Hub Sync:** Because Hugging Face Jobs containers wipe ephemeral disk on exit, Stage 2 uploads artifacts incrementally:
  - **Phase A Upload** (immediately after `discover_circuit` + surgical mask generation + initial `lexsi_provenance.json`, *before* `evaluate_circuit`): commits `<stem>.pt`, `<stem>_scores.pt`, `<stem>_scores.json`, `<stem>_surgical_submodules_scores.json`, `<stem>_surgical_submodules_top{3,4,6}_scores.json`, `safety_units_top{05,10,20}.json`, and `lexsi_provenance.json`.
  - **Phase B Upload** (after 4-pillar `evaluate_circuit` + `compute_kl_diagnostic` + provenance/KL updates): commits `faithfulness_report.json` (recording `hf_phase_a_upload_url`) alongside updated `scores` JSONs and `lexsi_provenance.json`.
- **Cloud Runner (`scripts/run_hf_jobs.sh`):** One-command launcher `scripts/run_hf_jobs.sh <stage2|stage3|stage4>` targeting `hf jobs run --flavor l4x1 --timeout 35m --secrets HF_TOKEN` with `--dry-run` and `--low-vram-t4` support (never embedding raw secrets on CLI arguments).
