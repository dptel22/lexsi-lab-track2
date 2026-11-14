# Stages 2–4 Technical Audit & Pre-Execution Safety Review

**Project**: `lexsi-lab` (Track 2: Lexical-Syntactic Interventions for LLM Safety)  
**Date**: 2026-11-13  
**Auditor**: Teamwork Orchestrator & Explorers (Stages 2, 3, 4, and Cloud Infrastructure)  
**Integrity Mode**: Development  
**Target Architecture**: Cohere2 (`CohereLabs/tiny-aya-fire`, 3.35B parameters, 36 layers, GQA 16Q/4KV, periodic NoPE global attention)  
**Target Hardware**: Hugging Face Jobs `l4x1` (24 GB VRAM) & `t4-small` / `t4-medium` (16 GB VRAM fallback)  

---

## Executive Summary

Before committing cloud GPU credits and launching Hugging Face Jobs, an exhaustive audit of `src/stage2_discover.py`, `src/stage3_repair.py`, `src/stage4_eval.py`, `scripts/run_hf_jobs.sh`, and their corresponding test suites (`tests/test_stage2.py`, `tests/test_stage3.py`, `tests/test_stage4.py`) was conducted.

The codebase exhibits strong architectural foundations:
1. **Cohere2 NoPE Attention Patching**: Proper integration of CircuitKIT's `_tl_compat` patches TransformerLens rotary embeddings on local sliding-window layers while disabling them on global layers.
2. **SafeTune PEFT & Locality Discipline**: Strict single-wrap discipline, unmerged LoRA persistence before merging, state restoration (`use_cache=True`, `disable_input_require_grads()`), and surgical bit-identity verification ensuring >94% frozen parameters.
3. **Hardware T4 Offload Strategy**: Reference model offload to CPU in `bfloat16` protects 16 GB T4 instances during SafeGrad KL gradient surgery.

However, seven concrete issues spanning cloud GPU memory safety, ephemeral container hydration, AuditKIT cache isolation, and credential protection were identified. All identified issues have direct, focused, and minimal in-place solutions.

---

## Detailed Audit Findings & Risk Analysis

### 1. Stage 2 (`src/stage2_discover.py`)

| ID | Severity | Finding | Root Cause | Cloud GPU Impact |
|---|---|---|---|---|
| **S2-1** | **High** | Missing `--hydrate-hf` Support | Unlike Stages 3 and 4, `stage2_discover.py` lacked `--hydrate-hf` CLI flag and `snapshot_download` logic for `stage1_dataset/**`. | In stateless cloud containers, Stage 2 fails immediately with `FileNotFoundError` if calibration dataset is not pre-baked into container image. |
| **S2-2** | **Medium** | Missing VRAM Release before Circuit Discovery | Preflight calibration (16 pairs) and margin filtering (100 forward passes) fragment PyTorch allocator prior to EAP-IG integrated gradients. | On 16 GB T4 GPUs, memory fragmentation before heavy gradient attribution risks unnecessary CUDA OOM. |
| **S2-3** | **Low** | `max_surgical_layers` Boundary Validation | Parser and validator did not assert `max_surgical_layers >= 1`. | Passing 0 could unintentionally result in uncapped layer selection. |

### 2. Stage 3 (`src/stage3_repair.py`)

| ID | Severity | Finding | Root Cause | Cloud GPU Impact |
|---|---|---|---|---|
| **S3-1** | **Critical** | Acute Peak VRAM Spike in C-ΔΘ State-Dict Loading | Lines 816 and 869 instantiated `{k: v.to(device=dev, dtype=tdtype) for k, v in base_sd.items()}` on GPU. | Allocating duplicate 6.7 GB weights on GPU while `candidate_model` (6.7 GB) is already loaded peaks VRAM at ~13.4 GB, risking immediate OOM on 16 GB T4 instances. |
| **S3-2** | **High** | Inverted Container Hydration Ordering | `load_calibration_pairs(dataset_path)` executed at line 981 *before* `--hydrate-hf` `snapshot_download` at line 987. | In stateless cloud containers, `load_calibration_pairs` crashes with `FileNotFoundError` before hydration can fetch upstream data from Hugging Face. |
| **S3-3** | **Low** | Defensive `use_cache` Assignment before Checkpoint 3 | Checkpoint 2 restored `use_cache=True`, but Checkpoint 3 did not defensively guarantee `use_cache=True`. | Downstream evaluation generation could suffer performance degradation if `use_cache` remained unset. |

### 3. Stage 4 (`src/stage4_eval.py`)

| ID | Severity | Finding | Root Cause | Cloud GPU Impact |
|---|---|---|---|---|
| **S4-1** | **High** | Missing Mandatory `RunConfig(extra=...)` Keys for DiskCache Isolation | `evaluate_precomputed_split` omitted `weights_revision`, `weights_mtime`, `dtype`, `device`, `max_new_tokens`, `n_examples`, and `prompt_set_digest`. | AuditKIT's `DiskCache` relies on `RunConfig.to_dict()` for hash key generation with precomputed models; missing fields cause cache collisions across different checkpoints or precision modes. |
| **S4-2** | **High** | CircuitKIT Import Order Rule Violation | `src/stage4_eval.py` lacked top-level `import circuitkit`. | If `stage4_eval.py` loads `transformer_lens` before `circuitkit`, Cohere2 NoPE rotary position embeddings remain unpatched. |
| **S4-3** | **Medium** | Missing `Metric` Base Class Import | Did not import `from auditkit.metric import Metric`. | AuditKIT custom judges should explicitly subclass `Metric` to adhere to evaluation API standards. |
| **S4-4** | **Medium** | Missing Post-Hydration Artifact Verification | `snapshot_download` completed without asserting that `repair_summary.json` or model directories exist. | Missing upstream Stage 3 artifacts produce confusing AutoModel loading errors instead of clear diagnostic fail-fast errors. |
| **S4-5** | **Low** | Subscript Access in `build_roi_comparison_table` | Direct dictionary indexing on optional red-team output keys. | Potential `KeyError` if red-team evaluation dictionary is partially populated. |

### 4. Cloud Jobs Runner Script (`scripts/run_hf_jobs.sh`)

| ID | Severity | Finding | Root Cause | Cloud GPU Impact |
|---|---|---|---|---|
| **SH-1** | **High** | Volume Syncing Overhead & Credential Leak Hazard | Script used `-v .:/workspace:ro` and inside container copied `/workspace /tmp/lexsi`. | HF CLI volume sync does not respect `.gitignore`, uploading the ~3 GB Windows `.venv` and sensitive `.env` file to cloud storage. |
| **SH-2** | **Medium** | Hardware Flavor & Low-VRAM Mismatch | Passing `--low-vram-t4` did not switch `FLAVOR="l4x1"` to `t4-small`, and `--flavor t4-small` did not enable `LOW_VRAM_T4=true`. | Users could pay for expensive L4 GPUs while throttled to 128 tokens, or suffer OOM on 16 GB T4 instances by omitting `--low-vram-t4`. |
| **SH-3** | **Low** | Default Timeout Cushion | Default timeout was set to `35m`. | Long evaluation runs with 3 checkpoints and multiple RedTeam attack models can hit timeout before completing. |

---

## Actionable Remediation Plan

### Remediation Step 1: `src/stage2_discover.py`
1. Add `--hydrate-hf` CLI argument and implement dataset hydration via `snapshot_download` in `run_stage2()`.
2. Insert `_release_vram()` immediately prior to `discover_circuit()` (line 1721).
3. Validate `max_surgical_layers >= 1`.

### Remediation Step 2: `src/stage3_repair.py`
1. In lines 816 and 869, change `candidate_model.load_state_dict({k: v.to(device=dev, dtype=tdtype) for k, v in base_sd.items()}, strict=True)` to `candidate_model.load_state_dict(base_sd, strict=True)`.
2. Move `--hydrate-hf` logic before `load_calibration_pairs(dataset_path)` in `run_stage3()`, hydrating both `stage1_dataset/**` and `stage2_circuits/**`.
3. Defensively assign `repaired_model.config.use_cache = True` before saving Checkpoint 3.

### Remediation Step 3: `src/stage4_eval.py`
1. Add top-level `import circuitkit  # noqa: F401` and `from auditkit.metric import Metric`.
2. Update `evaluate_precomputed_split` signature to accept `profile`, `model_revision`, `weights_fingerprint`, and populate all mandatory `RunConfig(extra=...)` fields.
3. Update callers in `run_stage4` to supply runtime profile and checkpoint fingerprints.
4. Add post-hydration existence validation for Stage 3 artifacts.
5. Use `.get("attack_pass_rate", 0.0)` in `build_roi_comparison_table`.

### Remediation Step 4: `scripts/run_hf_jobs.sh`
1. Modify `CONTAINER_CMD` to copy only `src`, `vendor`, `artifacts`, `scripts`, `pyproject.toml`, strictly excluding `.venv` and `.env`.
2. Implement bidirectional sync between `--low-vram-t4` and `FLAVOR=t4*`.
3. Increase default timeout from `35m` to `60m`.

### Remediation Step 5: Test Verification
1. Add unit tests for new functionality (e.g. `stage4` RunConfig extra keys assertion, Stage 2 `--hydrate-hf` parser check).
2. Execute syntax compilation (`py_compile`) across all audited files.
3. Execute CLI `--help` checks on all stage scripts.
4. Run all pytest suites (`tests/test_stage1.py`, `tests/test_stage2.py`, `tests/test_stage3.py`, `tests/test_stage4.py`, `tests/test_env.py`) on CPU with `gpt2` and `echo` models.

---

## Verification & Guardrail Attestation
- **Local VRAM Limit**: No execution or download of `CohereLabs/tiny-aya-fire` weights was or will be performed locally (~6 GB VRAM / CPU-only environment).
- **Environment Privacy**: No command or code accesses or exposes `.env`.
- **Import Ordering**: `circuitkit` precedes `transformer_lens` across all stages.
- **CPU Verification**: All tests execute strictly in CPU mode with `float32` and offline mocks.
