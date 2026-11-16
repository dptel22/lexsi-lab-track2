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

---

## Multi-Model Universality Expansion (Brainstorm & Goldfish Synthesis)

### 1. Overview & Motivation
Following the verified v2 turnaround on `CohereLabs/tiny-aya-fire` (3.35B), the user proposed expanding `CircuitKavach-Aya` to additional gated-accessible `CohereLabs/tiny-aya-*` models (`tiny-aya-global`, `tiny-aya-water`, and `tiny-aya-earth`, all verified `ACCESS GRANTED` via `hf_hub_download` while `tiny-aya-base` and `aya-expanse-{8b,32b}` returned `GatedRepoError`) to test whether safety-circuit localization and surgical `C-ΔΘ` repair generalize across regional Cohere checkpoints before the deadline (`Nov 15, 11:59 PM AoE` / `2026-11-16 11:59 UTC`).

### 2. Goldfish Analysis (`egm-zoom-out`, `egm-design`, `egm-codebase-details`)
Three fresh-eyes Goldfish reviewers audited the codebase and proposed 4-model expansion plan independently and uncovered the following critical Blockers and Major risks:

1. **🔴 Blocker — Construct Validity & Regional Mismatch on `water` and `earth` (`egm-zoom-out`, `egm-design`, `egm-codebase-details`):**
   - Our fixed 300-pair benchmark tests India-specific code-mixed prompts (`UPI OTP`, `SBI KYC`, `Aadhaar`, `IRCTC`, `EPFO` in `en`, `hi_en`, `gu_en`, `kn_en`). `tiny-aya-water` (Europe + APAC) and `tiny-aya-earth` (Africa + West Asia) are regional specialists merged with `tiny-aya-global`, and Kannada (`kn`) is absent from Cohere's model-card language metadata across these models.
   - `src/stage2_discover.py:546` weights `en` at `3.0x` vs `1.0x` for non-English in `calibrate_answer_tokens`, and `filter_pairs_by_baseline_margin` (`L688-731`) only sets `fallback_used=True` when `< 2` total pairs exceed `delta_m > 0.5`. Thus a model passing `25/25` `en` pairs and `~0` Indic pairs silently produces an **English-only circuit** while reporting multilingual discovery.
   - `src/stage3_repair.py:119-135` (`_REFUSAL_TEMPLATES`), `StringMatchJudge(mode="gcg")`, and Stage 4 refusal detectors score English prefixes; `benign_helpfulness` checks `not refused and len >= 16` chars (`wildguard` and `lmeval` are `"disabled"` in `eval_summary.json`), so incoherent Indic generations could score 100% helpfulness.

2. **🔴 Blocker — Stage 3 Host System RAM OOM Risk on `l4x1` (`egm-design`):**
   - On `l4x1` (24 GB VRAM, 30 GB System RAM), `run_stage3` (`src/stage3_repair.py:1112-1132`) and `run_safegrad_lora_harden` (`L728`) simultaneously keep `base_sd` (~6.7–7.2 GB), CPU-offloaded `reference_model` (~6.7–7.2 GB, active by default), and `ckpt2_sd` (~6.7–7.2 GB) live in CPU RAM **before** `del reference_model` at `L1176` — peaking near ~21+ GB of tensor copies alone inside a 30 GB container — even though `alignment_loss_mode="sft"` never invokes `reference_model.forward()`.

3. **🔴 Blocker — Cross-Stage & Multi-Model Artifact Contamination (`egm-design`, `egm-codebase-details`):**
   - `src/stage3_repair.py:111-113` and `src/stage4_eval.py:450-451` hardcode `tiny-aya-fire-*` folder names. Even with a slug resolver, `src/stage3_repair.py:1075-1107` skips `--hydrate-hf` if `artifacts/stage2_circuits/safety_circuit_surgical_submodules_top4_scores.json` already exists locally and `load_circuit_info_from_file` drops top-level `"model"` identity checks.
   - `_resolve_checkpoint_specs_from_stage3` (`src/stage4_eval.py:476-496`) reads literal paths from `repair_summary.json` without verifying `repair_summary["model"] == model_name`.
   - `upload_stage3_to_hf(m_dir, ...)` (`src/stage3_repair.py:1195-1200`) uploads all contents of `m_dir`, and `artifacts/stage1_dataset/{README.md,lexsi_provenance.json,manifest.json}` hardcode `"CohereLabs/tiny-aya-fire"` (`scripts/run_hf_jobs.sh:146` copies local `/workspace/artifacts/stage1_dataset` into containers, suppressing per-repo Stage 1 hydration, while Stage 2 flags omit `--hydrate-hf`).
   - Furthermore, `src/stage4_eval.py:1212` loads `eval_dataset_path` *before* `if hydrate_hf:` (`L1223`) and `L1244` only hydrates `stage3_models/**` and `stage3_repair/**` (not `stage1_dataset/**`).

4. **🔴 Blocker — `scripts/run_hf_jobs.sh` Launch, Quoting & Observability Gaps (`egm-design`, `egm-codebase-details`):**
   - `run_hf_jobs.sh:29` defaults `FLAVOR="l4x1"` even for Stage 2 (which requires `l40sx1`), hardcodes `--model CohereLabs/tiny-aya-fire` (`L100,104,108`), lacks `--model`, uses unguarded `shift 2` under `set -euo pipefail` (`L49-65`), passes `pydantic` differently from inspected working jobs, and lacks a gated-model `hf_hub_download(..., "config.json")` preflight or correlation `--run-id`.

5. **🟡 Major — Container Round-Trip Tax, Hub Repo Sprawl & Evaluation Blind Spots (`egm-zoom-out`, `egm-design`, `egm-codebase-details`):**
   - Running Stage 3 and Stage 4 in separate containers pays a ~15–20 GB Hub upload+download tax (`~13.5 GB` Stage 3 checkpoints uploaded twice at Phase A & Phase B, then re-downloaded for Stage 4) plus ~2–4 min `uv` bootstrap per job; `src/stage4_eval.py:1097,1134-1137` also reads `model_obj._tok` after `unload()` (`_pipeline.tokenizer` is cleared), falling back to whitespace word counts.
   - Creating 12 new HF repos (3 dataset + 9 model repos) and duplicating `15+ GB` checkpoints into dataset repos is heavy compared to scoping `fire` vs `global` (with optional `water`/`earth` preflight/namespaced runs under `runs/<slug>/...`) and adding a lightweight cross-model aggregation/comparison script (`Jaccard`/Spearman circuit overlap, per-language passing quotas, McNemar/bootstrap CIs).

### 3. Goals vs. Non-Goals
- **Goals:**
  - Isolate model-specific naming, directories, provenance, and Stage 2->3->4 identity checks safely in code with strict CPU TDD (`gpt2` / synthetic models).
  - Run a fast preflight + high-signal cross-model comparison (`tiny-aya-fire` vs `tiny-aya-global` at minimum, and `water`/`earth` if preflight validates cleanly and budget/time permit) on the held-out Indic code-mixed benchmark without risking any overwrite of verified `tiny-aya-fire` artifacts.
  - Eliminate Stage 3 `l4x1` CPU RAM waste (`reference_model` when `alignment_loss_mode=="sft"`), fix Stage 4 standalone dataset hydration and tokenizer throughput reporting, and unify cross-model statistical comparison & circuit-overlap reporting in our HF Collections, `README.md`, and `docs/submission_article.md`.
- **Non-Goals:**
  - Curating brand-new human-audited multilingual datasets for Europe/APAC (`water`) or Africa/West Asia (`earth`) before submission unless explicitly scoped.
  - Modifying the verified `tiny-aya-fire` v2 baseline artifacts already sealed in `soapyyapps/circuit-kavach-cyber-codemixed` and its standalone HF model repos.

### 4. Alternatives Considered
1. **Option 1 — Original Plan (Blind 4-Model Standalone Parallel Expansion across `{fire, global, water, earth}` into 12 New HF Repos):** High OOM risk in Stage 3 (`~21 GB` CPU RAM on `l4x1`), high timeout/bandwidth waste (`~81 GB` uploads), and high scientific risk (`water`/`earth` silent English-only Stage 2 collapse or meaningless low scores on Indic prompts).
2. **Option 2 — Goldfish-Hardened Comparative Expansion (Recommended):**
   - **Phase 0 (Code + TDD):** Fix `model_slug` isolation + provenance identity validation, skip `reference_model` CPU copy in `sft` mode, add per-language Stage 2 margin reporting/quotas, fix `run_hf_jobs.sh` (`--model`, safe `shift 2`, `stage2 -> l40sx1`, gated-model preflight, `--run-id`, optional Stage 3+4 single-container chain), and fix Stage 4 dataset hydration + tokenizer count.
   - **Phase 1 (Fast Preflight Gate):** Evaluate base Stage 4 refusal + Stage 2 calibration/margin distribution on `tiny-aya-global` (and `water`/`earth` if included) (~10-20 min) before burning full `~2h` repair pipelines.
   - **Phase 2 (Focused Full Run & Unified Publication):** Run full Stage 2 (`l40sx1`) + chained Stage 3->4 (`l4x1`) cleanly (prioritizing `tiny-aya-global` first for clean Regional vs Global generalization, plus `water`/`earth` only if preflight passes language margin quotas), using clean Hub namespacing (`runs/<slug>/...` or per-model model repos for LoRA/merged adapters as decided in plan review), then aggregate circuit overlap + paired statistics into the existing public HF Collections and writeup.

### 5. Key Decisions Needed in Re-Planning
1. **Target Model Scope:** Prioritize `tiny-aya-fire` vs `tiny-aya-global` first (strictly valid South Asia vs Global comparison on the Indic benchmark), OR run full `{global, water, earth}` subject to a Stage 2 per-language margin preflight gate (`water`/`earth` framed explicitly as zero-shot cross-regional transfer)?
2. **HF Hub Storage Architecture:** Keep `soapyyapps/circuit-kavach-cyber-codemixed` as the canonical dataset repo with namespaced subfolders (`runs/<slug>/stage{2,3,4}`) + standalone HF **Model** repos for LoRA/merged adapters added to the existing Model Collection, OR separate per-model dataset repos (`soapyyapps/circuit-kavach-<slug>`)?
3. **Cloud Execution Topology:** Chain Stage 3 + Stage 4 in one `l4x1` container (saving ~15 GB download/upload + cold boot per model, keeping Phase A HF backup) vs separate `stage3` and `stage4` jobs?
4. **Stage 2/3 Statistical Safeguards:** Enforce per-language margin logging + `sft`-mode `reference_model` skip (fixing the `l4x1` 30 GB RAM risk) and add automated cross-model aggregation (`circuit_overlap` + McNemar/bootstrap stats)?

---

## Empirical Verification (HF Jobs v3 Multi-Model Universality Expansion)

- **Hardened Code & CPU Verification:** Resolved all Goldfish blockers (`resolve_model_slug` / `resolve_artifact_names` isolation across Stages 2–4, skipping CPU `reference_model` load in `alignment_loss_mode="sft"`, `english_only_circuit_warning` & per-language margin tracking, Stage 4 pre-`unload()` tokenizer token counting, and `src/aggregate_multimodel.py`). All **64 CPU verification tests** pass (`pytest tests/ -q` on `"gpt2"` + synthetic `cohere2`).
- **Cloud Runs Executed Across `{global, water, earth}` + Hub Sync (\$9.80 total across all 4 models):**
  - `tiny-aya-global`: Stage 2 (`6af9d4bf23985fb9cd837e02`, `l40sx1`, `51m54s`) + chained Stage 3–4 (`6af9e2a029dd05ff0368ef20`, `l4x1`, `64m55s`): `97/100` positive-margin pairs, `ablation_score=0.8659`, `mean_jaccard=0.9788`, `mean_spearman=0.9998`, `best_alpha=1.0` (`100%` val refusal); held-out attack refusal **`45.5%` → `96.5%`** (`+51.0 pp`, $p=0.000999$), **`0.0%` benign over-refusal**, **`28.82 tok/s`**.
  - `tiny-aya-water`: Stage 2 (`6af9d4bf23985fb9cd837e04`, `l40sx1`, `52m42s`) + chained Stage 3–4 (`6af9e2a223985fb9cd83897f`, `l4x1`, `64m60s`): `99/100` positive-margin pairs, `mean_jaccard=0.9788`, `mean_spearman=0.9998`, `ablation_score=0.0` (CircuitKIT `[0,1]`-clipped zero-ablation ratio artifact at 90% node sparsity), `best_alpha=1.0` (`95%` val refusal); held-out attack refusal **`53.5%` → `96.5%`** (`+43.0 pp`, $p=0.000999$), **`0.0%` benign over-refusal**, **`28.76 tok/s`**.
  - `tiny-aya-earth`: Stage 2 (`6af9d4c029dd05ff0368e1f7`, `l40sx1`, `52m56s`) + chained Stage 3–4 (`6af9e2a323985fb9cd838985`, `l4x1`, `66m03s`): `99/100` positive-margin pairs, `ablation_score=0.8052`, `mean_jaccard=1.0000`, `mean_spearman=0.9998`, `best_alpha=1.0` (`100%` val refusal); held-out attack refusal **`53.5%` → `99.0%`** (`+45.5 pp`, $p=0.000999$; parity with 72-block Global SafeTune `99.5%`, $p=0.612388$), **`0.0%` benign over-refusal**, **`100.0%` `kn_en` refusal**, **`28.54 tok/s`**.
  - Model Hub Sync (`6af9ea4329dd05ff0368f74d`, `cpu-basic`, `2m39s`): Published all 12 adapters/global/surgical models cleanly to HF Hub.
- **Key Cross-Model Findings Confirmed (`artifacts/multimodel_comparison.json`):**
  1. **Universal 4-Block Safety Router (`Jaccard = 1.000`):** Every model selected `{layers.0.mlp, layers.16.self_attn, layers.20.self_attn, layers.21.self_attn}` (`99,090,432 / 3,886,098,432 = 2.55%` modified, `97.45%` bit-identical across `276/291` tensors), with `0.9217–0.9704` full 72-block Spearman correlation across all 6 model pairs.
  2. **Regional Specialization Paradox:** Early lexical attribution `layers.0.mlp` scales inversely with base/surgical robustness (`fire=7.0312` → `global=4.5625` → `water=3.9531` → `earth=2.2188`), while `layers.16.self_attn` remains conserved (`2.489–2.867`).
  3. **Benign Coherence & Clean Residual Verification:** `non_empty_rate = 100.0%` and `mean_token_diversity = 0.9701–0.9810` across all `800` benign generations; **27 / 27** residual surgical misses across all `800` harmful generations emitted defensive remediation guidance (`"To protect..."`) rather than exploit payloads.


