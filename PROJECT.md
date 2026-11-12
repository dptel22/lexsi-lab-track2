# Project: CircuitKavach-Aya

## Architecture
CircuitKavach-Aya implements an in-weights, surgical safety repair pipeline for `CohereLabs/tiny-aya-fire` (3.35B, `cohere2` architecture: 36 layers, GQA `16Q/4KV`, parallel attention/MLP, NoPE on every 4th layer `[3, 7, 11, 15, 19, 23, 27, 31, 35]`) targeting multilingual and code-mixed cybersecurity & phishing safety across English and three Indic code-mixed registers (`en`, `hi_en`, `gu_en`, `kn_en`).

The pipeline integrates the 4-tier Lexsi-Labs Alignment Stack (`vendor/`):
1. **CuratorKIT (`src/stage1_curate.py`)**: Stage 1 contrastive safety dataset curation producing exact lexical twins partitioned into disjoint `calibration_100_pairs.jsonl` (100 pairs) and `eval_200_pairs.jsonl` (200 pairs) under `artifacts/stage1_dataset/`, with `manifest.json`, `checksums.txt`, `dataset_card.md`, `README.md`, `rejected.jsonl`, and `lexsi_provenance.json`.
2. **CircuitKIT (`src/stage2_discover.py`)**: Stage 2 mechanistic refusal circuit discovery using `eap-ig` with within-pair length equalization, positive baseline margin preflight filtering, 4-pillar faithfulness evaluation, layer-saturation diagnostics, and submodule-ranked surgical mask guard exporting `<stem>_surgical_submodules_*.json` + `lexsi_provenance.json`.
3. **SafeTune (`src/stage3_repair.py`)**: Stage 3 two-phase selectivity repair — Phase A runs `SafeGradHFTrainer` with single LoRA wrapping and exports the unmerged LoRA adapter + merged Global Checkpoint 2 (`use_cache=True`), and Phase B sweeps C-ΔΘ transplant strength `alpha_grid` over the Stage 2 submodule mask to materialize Surgical Checkpoint 3 while auditing bit-identity outside the mask.
4. **AuditKIT (`src/stage4_eval.py`)**: Stage 4 3-checkpoint decoupled evaluation (`checkpoint_1_base`, `checkpoint_2_global_safetune`, `checkpoint_3_surgical_ctheta`) running serialized greedy generation (`do_sample=False`), `RedTeamRunner` probes with non-refusal compliance detection, offline refusal/helpfulness scorers with `DiskCache` key isolation, optional WildGuard/`lm-eval` hooks, paired bootstrap significance comparison, and Markdown/JSON ROI tables.

```
[Stage 1: CuratorKIT] (artifacts/stage1_dataset/: 100 calib + 200 eval pairs)
         │
         ▼
[Stage 2: CircuitKIT] (src/stage2_discover.py -> artifacts/stage2_circuits/)
         │
         ▼ (submodule-ranked surgical mask + --hydrate-hf support)
[Stage 3: SafeTune]   (src/stage3_repair.py -> artifacts/stage3_models/ & artifacts/stage3_repair/)
         │
         ▼ (Checkpoint 1 Base, Checkpoint 2 Global SafeGrad, Checkpoint 3 Surgical C-ΔΘ)
[Stage 4: AuditKIT]   (src/stage4_eval.py -> artifacts/stage4_eval_logs/)
```

## Cloud & Local Execution
- **Hugging Face Jobs Runner (`scripts/run_hf_jobs.sh`)**: Submits `stage2`, `stage3` (`--hydrate-hf`), or `stage4` (`--hydrate-hf`) to `l4x1` (`--timeout 35m`, optional `--low-vram-t4` 16 GB T4 fallback) with fail-fast `HF_TOKEN` verification and incremental `--upload-hf`.
- **Local Verification Boundary (`AGENTS.md`)**: Never load 3.35B `CohereLabs/tiny-aya-fire` on the local 6 GB laptop; all local unit and smoke tests run on CPU using `"gpt2"` (`dtype="float32"`), synthetic tiny `Cohere2ForCausalLM` checkpoints, `"echo"`, or `CallableModel`.

## Repository Layout
```
lexsi-lab/
├── AGENTS.md
├── PROJECT.md
├── pyproject.toml
├── .agents/skills/hf-dataset-splits/
├── artifacts/
│   └── stage1_dataset/              # Tracked Stage 1 curated dataset & provenance
│       ├── calibration_100_pairs.jsonl
│       ├── eval_200_pairs.jsonl
│       ├── manifest.json
│       ├── checksums.txt
│       ├── dataset_card.md
│       ├── README.md
│       ├── rejected.jsonl
│       └── lexsi_provenance.json
├── docs/
│   ├── api_surface_verified.md
│   └── superpowers/specs/
│       ├── 2026-11-10-stage1-dataset-design.md
│       ├── 2026-11-11-stage2-discover-design.md
│       ├── 2026-11-11-stage3-repair-design.md
│       └── 2026-11-11-stage4-eval-design.md
├── scripts/
│   └── run_hf_jobs.sh               # One-command HF Jobs launcher (Stages 2–4)
├── src/
│   ├── __init__.py
│   ├── stage1_curate.py
│   ├── stage2_discover.py
│   ├── stage3_repair.py
│   └── stage4_eval.py
├── tests/
│   ├── __init__.py
│   ├── test_env.py
│   ├── test_stage1.py
│   ├── test_stage2.py
│   ├── test_stage3.py
│   └── test_stage4.py
└── vendor/                          # Vendored Lexsi-Labs libraries (gitignored)
    ├── CuratorKIT/
    ├── CircuitKIT/
    ├── SafeTune/
    └── AuditKIT/
```
