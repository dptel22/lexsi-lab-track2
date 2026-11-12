# Stage 3 Design Specification: SafeTune Selectivity Hardening & Surgical `C-ΔΘ` Weight Transplant (`src/stage3_repair.py`)

**Date:** 2026-11-12  
**Status:** Approved & Locked (incorporating Adversarial Patches 2, 3, and 6)  
**Target Models:** `CohereLabs/tiny-aya-fire` (`cohere2`, `bfloat16` on L4 24 GB / T4 16 GB fallback), local `"gpt2"` / tiny synthetic `Cohere2` (`float32` on CPU tests)

---

## 1. Objectives & Three Output Checkpoints

Stage 3 repairs over-refusal and strengthens code-mixed cyber attack refusal using `SafeTune`, producing three persisted artifacts ready for Stage 4 comparison and public release:

1. **Artifact 3A (`tiny-aya-fire-circuit-kavach-lora`)** — Standalone unmerged PEFT LoRA adapter (`adapter_config.json` + `adapter_model.safetensors`).
2. **Artifact 3B (`tiny-aya-fire-global-safetune`)** — **Checkpoint 2 (Global SafeTune)** full merged model (`lora_model.merge_and_unload()`).
3. **Artifact 3C (`tiny-aya-fire-circuit-kavach-merged`)** — **Checkpoint 3 (Surgical `C-ΔΘ`)** sparse model transplanting only Stage 2's `< 5%–8%` surgical mask (`top4`: `4/72 = 5.56%` of projection blocks) via three-input `apply_ctheta_from_state_dicts` and `alpha` strength sweep, leaving all other weights **bit-for-bit identical** (`torch.equal`) to Checkpoint 1 (`base`).

---

## 2. Adversarial Patch 2: Single PEFT Wrap & `SafeGradHFTrainer` (No Auto-Merge Trap)

In `vendor/SafeTune/src/safetune/runner/harden/_gradient_surgery.py`, `SafeGradTrainer.train()` calls `self._lora_base()` internally (wrapping in PEFT) and automatically calls `merge_and_unload()` without returning or saving the standalone adapter. Passing a manually wrapped `PeftModel` double-wraps PEFT layers.

**Stage 3 Implementation:**
1. Load the base model **once** (`bfloat16` on CUDA, `float32` on CPU/`gpt2`), capture `model_revision = resolve_model_revision(...)` (**Patch 6**), and retain a CPU `state_dict` snapshot `base_sd` of $\theta_0$ before training.
2. Wrap the base model **once** using `safetune.runner.utils.model_utils.lora_wrap(base_model, r=lora_r, alpha=lora_alpha, target_modules=...)` (auto-detecting `cohere2` projections `["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]` vs `gpt2` projections `["c_attn","c_proj","c_fc"]`). Verify no double `base_model.model.base_model` PEFT wrapping.
3. Instantiate a frozen reference copy `reference_model` ($\theta_0$, `eval()`, `requires_grad_(False)`):
   - Placed on GPU for L4 (`24 GB`) runs, or automatically kept on **CPU** under `--low-vram-t4` (supported natively by `SafeGradHFTrainer._alignment_gradient`, which moves `ref_inputs` to `self._reference_model.device`).
4. Train using **`safetune.interventions.harden.safegrad.SafeGradHFTrainer`** with `SafeGradConfig(rho=rho, kl_temperature=kl_temperature)` and `TrainingArguments(..., gradient_accumulation_steps=1, bf16=...)`:
   - `train_dataset`: benign defensive coding/remediation answers generated from `calibration_100_pairs.jsonl` (80/20 stratified train/val split by `language` and `threat_category`; **never touches `eval_200_pairs.jsonl`**).
   - `safety_dataset`: harmful prompts from the calibration split paired with explicit safe refusals (using canonical refusal phrasing compatible with `StringMatchJudge(mode="gcg")` and `AuditKIT.RefusalDetector`).
   - Supervised length resolution ensures prompt prefixes never consume `max_len` (fixing `_tokenize_qa_rows` empty-label errors on Tiny Aya's ~366-token system preamble, with raw prompt-completion fallback when `tokenizer.chat_template is None`).
5. **Save Artifact 3A BEFORE merge:** call `lora_model.save_pretrained(...)` + `tokenizer.save_pretrained(...)`.
6. **Merge into Artifact 3B:** call `ckpt2 = lora_model.merge_and_unload()`, save to `models_dir / "tiny-aya-fire-global-safetune"`, and snapshot `ckpt2_sd` on CPU.

---

## 3. Adversarial Patch 3: Three-Input `apply_ctheta` & `sweep_ctheta_strength`

1. Load Stage 2's surgical mask via `safetune.core.circuit_kit.adapter.load_circuit_info_from_file(circuit_info_path)` (default `artifacts/stage2_circuits/safety_circuit_surgical_submodules_top4_scores.json`, `4/72 = 5.56%` blocks).
2. Sweep `alpha` over `--alpha-grid` (default `[0.5, 1.0, 1.5]`) using `safetune.interventions.recover.ctheta.sweep_ctheta_strength` / `apply_ctheta_from_state_dicts` with the strict **3-model contract**:
   - `target=base_sd`
   - `positive=ckpt2_sd` (global SafeTune)
   - `negative=base_sd`
3. Evaluate each candidate on the calibration validation split (`val_pairs`, greedy decoding `do_sample=False`, **Patch 6**) maximizing:
   $$\text{Selectivity}(M_\alpha) = \text{AttackRefusalRate}(M_\alpha) - \text{BenignOverRefusalRate}(M_\alpha)$$
   via `StringMatchJudge(mode="gcg")`.
4. Materialize and save **Artifact 3C (`tiny-aya-fire-circuit-kavach-merged`)** using the winning `alpha*`, and assert **bit-for-bit weight preservation** (`torch.equal(ckpt3_sd[k], base_sd[k])`) across **100% of parameter keys outside the surgical mask**.
5. Persist exact parameter audit metrics (`modified_param_count`, `total_param_count`, `modified_param_pct`, `frozen_param_pct`, `projection_blocks_touched`, `total_projection_blocks`).

---

## 4. Runtime Safety, Provenance & Two-Phase HF Sync

- **`--low-vram-t4`:** Sets `batch_size=1`, `ref_offload_cpu=True`, enables gradient checkpointing during LoRA hardening, caps `max_new_tokens=64` during sweep, and calls `_release_vram()` between hardening, sweep, and checkpoint export.
- **Patch 6 Determinism & Provenance:** Greedy decoding (`do_sample=False`), explicit `model_revision` SHA tracking, and `lexsi_provenance.json` written via `safetune.provenance.write_provenance` chaining Stage 1 and Stage 2 provenance records. Optional `--use-trackio` logging degrades gracefully offline.
- **`--upload-hf` Fail-Fast + Two-Phase Upload:** Runs `verify_hf_auth()` before model loading; uploads Phase A (Artifacts 3A & 3B + Stage 3 intermediate provenance) right after SafeGrad, and Phase B (Artifact 3C + `repair_summary.json` + final `lexsi_provenance.json`) after `C-ΔΘ`.
