# Concepts Brief: Stage 2–4 Root-Cause Remediation (`RC1`–`RC5`, Post-Goldfish Revision)

## Goals (In-Scope)
- Enable supervised refusal gradient surgery (`alignment_loss_mode="sft"`) in Stage 3 `SafeGradHFTrainer` so `Checkpoint 2 (Global)` and `Checkpoint 3 (Surgical C-ΔΘ)` raise code-mixed refusal (`hi_en`, `gu_en`, `kn_en`) on `CohereLabs/tiny-aya-fire` while holding benign over-refusal near `0%`.
- Fix CircuitKIT chat-template detection for `dict`-typed `CohereTokenizerFast.chat_template` without breaking mock-tokenizer unit tests, and anchor preflight token calibration on English refusal pairs (`3.0x` pair weight) while preserving `n_pairs_used`.
- Replace ineffective backward tensor hooks with native `AdamW` per-parameter-group learning rates so Stage 3 concentrates plasticity in the surgical circuit submodules cleanly and respects SafeGrad Eq. 3–4 global surgery geometry.
- Isolate `scripts/run_hf_jobs.sh` artifact staging to `artifacts/stage1_dataset` so `--hydrate-hf` never reads stale local `stage2_circuits/` or `stage3_repair/` artifacts, then rerun Stage 2 → Stage 3 → Stage 4 on HF Jobs (`l4x1`).

## Non-Goals (Out-of-Scope)
- Modifying Stage 1 JSONL datasets, Stage 4 `AuditKIT` scorers, or breaking CLI/test defaults (`DEFAULT_ALPHA_GRID = (0.5, 1.0, 1.5)`, `cal.n_pairs_used`, `seen_ref_devices == ["cpu"]`).
- Running `CohereLabs/tiny-aya-fire` (3.35B) locally on Windows CPU/6 GB VRAM.

---

## Five Revised Root Causes & Resolutions

### RC1 — Opt-In `alignment_loss_mode="sft"` in `SafeGradHFTrainer`
- **Evidence:** [vendor/SafeTune/src/safetune/interventions/harden/safegrad.py:226-245](file:///c:/Users/dhruv/PycharmProjects/lexsi-lab/vendor/SafeTune/src/safetune/interventions/harden/safegrad.py#L226-L245) & [src/stage3_repair.py:578](file:///c:/Users/dhruv/PycharmProjects/lexsi-lab/src/stage3_repair.py#L578).
- **Goldfish Constraint:** Mutating `SafeGradHFTrainer` implicitly via `labels != -100` would break `SafeGradTrainer` (`_gradient_surgery.py:60-97`), whose `build_safety_dataset` always emits non-`-100` labels alongside a frozen aligned `reference_model`. Conversely, omitting `reference_model` breaks `tests/test_stage3.py:441`.
- **Fix:** Add `alignment_loss_mode: str = "auto"` (`"auto" | "sft" | "kl"`) to `SafeGradHFTrainer.__init__`. `"auto"` preserves existing KL-vs-SFT dispatch; `"sft"` runs supervised refusal cross-entropy `model(**prepared).loss` without querying the unaligned `reference_model`. Stage 3 passes `alignment_loss_mode="sft"` and defaults `ref_offload_cpu: bool = True` so CPU-offloaded `reference_model` consumes `0 GB` GPU VRAM.

### RC2 — Accept `dict`-Typed `chat_template` Without `apply_chat_template` Gate
- **Evidence:** [vendor/CircuitKIT/src/circuitkit/tasks/_chat.py:60-68](file:///c:/Users/dhruv/PycharmProjects/lexsi-lab/vendor/CircuitKIT/src/circuitkit/tasks/_chat.py#L60-L68), [src/stage2_discover.py:143-149](file:///c:/Users/dhruv/PycharmProjects/lexsi-lab/src/stage2_discover.py#L143-L149), and Layer 0 dominance (`"MLP 0": 15.3125`) in [artifacts/stage2_circuits/safety_circuit_surgical_submodules_top4_scores.json:25,689-717](file:///c:/Users/dhruv/PycharmProjects/lexsi-lab/artifacts/stage2_circuits/safety_circuit_surgical_submodules_top4_scores.json#L25).
- **Goldfish Constraint:** Checking `hasattr(obj, "apply_chat_template")` breaks `tests/test_stage2.py:189-208` (`SimpleNamespace(chat_template=_CHAT_JINJA)`).
- **Fix:** Update `_has_chat_template` to return `True` for non-empty `str` or `dict` (checking `template.get("default")` first, then any non-empty string value), without checking `hasattr(..., "apply_chat_template")`. Update `stage2_discover.py:145` (`chat_template_mode == "on"`) to validate `not _has_chat_template(tok)` rather than `is None`.

### RC3 — Weighted `"en"` Refusal Anchoring in `calibrate_answer_tokens`
- **Evidence:** [src/stage2_discover.py:480-590](file:///c:/Users/dhruv/PycharmProjects/lexsi-lab/src/stage2_discover.py#L480-L590) & [artifacts/stage4_eval_logs/eval_summary.json:17-44](file:///c:/Users/dhruv/PycharmProjects/lexsi-lab/artifacts/stage4_eval_logs/eval_summary.json#L17-L44).
- **Goldfish Constraint:** Hard-filtering non-`"en"` rows before `select_preflight_pairs` drops `cal.n_pairs_used` from `16` to `4` and breaks `tests/test_stage2.py:318`.
- **Fix:** Keep all preflight pairs (`n_used == 16`), but accumulate `sum_logits_*` and `sum_logp_*` using `w = 3.0 if str(row.get("language", "")).strip() == "en" else 1.0` and divide by `total_weight`. Single-language or uniform datasets behave identically while English refusal anchors `(clean_token, corrupt_token)` when `gu_en`/`kn_en` base prompts comply (`94%`).

### RC4 — `AdamW` Parameter-Group LR Scaling Instead of Backward Hooks
- **Evidence:** [src/stage3_repair.py:520-611](file:///c:/Users/dhruv/PycharmProjects/lexsi-lab/src/stage3_repair.py#L520-L611), [vendor/SafeTune/src/safetune/interventions/harden/safegrad.py:253-327](file:///c:/Users/dhruv/PycharmProjects/lexsi-lab/vendor/SafeTune/src/safetune/interventions/harden/safegrad.py#L253-L327), and [vendor/SafeTune/src/safetune/interventions/recover/ctheta.py:13-43](file:///c:/Users/dhruv/PycharmProjects/lexsi-lab/vendor/SafeTune/src/safetune/interventions/recover/ctheta.py#L13-L43).
- **Goldfish Constraint:** Backward tensor hooks cancel out under `AdamW`'s $\hat{m}_t / \sqrt{\hat{v}_t}$ update and distort `torch.autograd.grad` / SafeGrad Eq. 3–4 projection geometry.
- **Fix:** Leave gradients and SafeGrad projection unscaled. Accept `circuit_info: Optional[Any] = None` and `non_circuit_lr_scale: float = 0.10` in `run_safegrad_lora_harden`, partition **trainable** LoRA parameter names (`[n for n, p in lora_model.named_parameters() if p.requires_grad]`) via `_matched_param_keys`, construct a two-group `torch.optim.AdamW` (`lr` vs `lr * non_circuit_lr_scale`), pass `optimizers=(optimizer, None)` to `SafeGradHFTrainer`, and preserve `circuit_mask` in `harden_inputs` provenance when `circuit_info is not None`.

### RC5 — Stage 3 SafeGrad Strength (`rho=1.0`, `epochs=3.0`) & Cloud Hydration Isolation
- **Evidence:** [src/stage3_repair.py:509-515,962-964,1222-1224](file:///c:/Users/dhruv/PycharmProjects/lexsi-lab/src/stage3_repair.py#L509-L515) & [scripts/run_hf_jobs.sh:146](file:///c:/Users/dhruv/PycharmProjects/lexsi-lab/scripts/run_hf_jobs.sh#L146).
- **Fix:** Update `rho` default from `0.25` to `1.0` and `epochs` default from `2.0` to `3.0` (retaining `lr=2e-4` and `DEFAULT_ALPHA_GRID = (0.5, 1.0, 1.5)`). In `scripts/run_hf_jobs.sh:146`, stage only `artifacts/stage1_dataset` (keeping LF line endings) so stale local Stage 2/3 artifacts never suppress `--hydrate-hf`.


---

## Empirical Verification (HF Jobs Rerun v2)
- **Cloud Runs Executed:**
  - Stage 2 (`6af96d1629dd05ff03686a08`, `l40sx1`, `51m49s`): `templated=true`, `prepend_bos=false`, calibrated `" sorry"` (`40449`) vs `" when"` (`1434`, `logit_diff_gap=16.21`), `99/100` positive-margin pairs, `ablation_score=0.6549`, `mean_jaccard=0.9683`, `mean_spearman=0.9998`, top-4 submodules `layers.0.mlp`, `layers.16.self_attn`, `layers.21.self_attn`, `layers.20.self_attn` (`94.44%` blocks frozen). Note: `l40sx1` (`48 GB` VRAM) was required because Cohere's ~360-token contextual safety chat preamble inflated `HookedTransformer` `use_attn_result` EAP-IG peak beyond L4 (`22.03 GiB`).
  - Stage 3 (`6af97a2723985fb9cd8320bb`, `l4x1`, `15m50s`): `best_alpha=1.0`, validation refusal `0.95` (`0.45` in v1), `benign_over_refusal=0.0`, `modified_param_pct=2.55%` (`97.45%` frozen), `bit_identical_unmasked_keys=true`.
  - Stage 4 (`6af97fa323985fb9cd8325eb`, `l4x1`, `51m53s`):
    - `checkpoint_1_base`: `40.0%` attack refusal (`en=100%`, `hi_en=48%`, `gu_en=6%`, `kn_en=6%`), `0.0%` benign over-refusal.
    - `checkpoint_2_global_safetune` (`72/72` blocks): `99.0%` attack refusal (`en=100%`, `hi_en=100%`, `gu_en=100%`, `kn_en=96%`), `0.0%` benign over-refusal (`38.5%` in v1).
    - `checkpoint_3_surgical_ctheta` (`4/72` blocks, `2.55%` params modified): **`94.5%`** attack refusal (`en=100%`, `hi_en=100%`, `gu_en=100%`, `kn_en=78%`; `+54.5 pp` vs base, `p=0.000999`), **`0.0%`** benign over-refusal (`40.5%` in v1), `38.1%` RedTeam pass rate.
