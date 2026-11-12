# Graph Report - lexsi-lab  (2026-11-12)

## Corpus Check
- Corpus is ~40,387 words - fits in a single context window. You may not need a graph.

## Summary
- 532 nodes · 941 edges · 36 communities (17 shown, 19 thin omitted)
- Extraction: 94% EXTRACTED · 6% INFERRED · 0% AMBIGUOUS · INFERRED: 58 edges (avg confidence: 0.87)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Stage 2: CircuitKIT EAP-IG & Discovery Pipeline
- Stage 3: SafeTune & C-DeltaTheta Repair Core
- Stage 1: CuratorKIT Twin Curation & Config
- Shared Pytest & Test Fixtures
- Stage 1 Provenance & Manifest Schema
- Stage 3 Unit & Cohere2 Parity Tests
- Stage 4: AuditKIT Core & Evaluation Orchestration
- Stage 4 Reporting & ROI Synthesis
- Stage 2 Unit & Hugging Face Runner Tests
- Project Rules, Lexsi API Verification & Stack Docs
- Stage 4 Decoupled Two-Phase VRAM Generation
- Stage 4 Precomputed Scorer Harness
- Stage 4 Offline Refusal & Helpfulness Scorers
- Stage 2 Length Equalization & Calibration Tests
- Stage 4 Attack Non-Refusal Detector
- Stage 1 SHA-256 Checksum Integrity
- Stage 1 Dataset Card & HF Split YAML Spec
- Stage 4 Per-Sample Metric Aggregation
- Stage 4 Optional WildGuard & lm-eval Hooks
- Stage 2 BPE Roundtrip Calibration Tests
- Stage 2 Low-VRAM T4 & Fail-Fast Auth Tests
- Stage 2 Baseline Margin Filter Tests
- Surgical Submodule Mask & Bit-Identity Guarantees
- Hardware-Aware T4/L4 SafeGrad Profiles
- Cross-Lingual Safety Decay & Lexical Twin Design
- 3-Checkpoint Selectivity & Greedy Lock Design
- Stage 2 Phase A Incremental HF Sync Tests
- HF Dataset Repo & Programmatic Split Check
- Cohere2 Periodic NoPE & Single-BOS Spec
- Word Tokenizer Test Helper
- GitNexus Graph Index Workflow
- Python Package Definition (circuit-kavach-aya)

## God Nodes (most connected - your core abstractions)
1. `run_stage2()` - 32 edges
2. `run_stage4()` - 27 edges
3. `run_stage3()` - 18 edges
4. `_phase1_generate_for_checkpoint()` - 14 edges
5. `_create_tiny_hf_checkpoint()` - 13 edges
6. `_sample_pairs()` - 13 edges
7. `export_provenance_bundle()` - 12 edges
8. `ContrastPair` - 11 edges
9. `run_ctheta_sweep_and_transplant()` - 11 edges
10. `_classify_refusal_signals()` - 11 edges

## Surprising Connections (you probably didn't know these)
- `Laptop CPU & Stdlib-First Constraint (Ponytail Rule)` --rationale_for--> `generate_300_pairs()`  [INFERRED]
  C:/Users/dhruv/PycharmProjects/lexsi-lab/docs/superpowers/specs/2026-11-10-stage1-dataset-design.md → src/stage1_curate.py
- `Stage 2 L4 (24GB), T4 (16GB) & Local CPU (6GB) Profiles` --implements--> `resolve_runtime_profile()`  [EXTRACTED]
  C:/Users/dhruv/PycharmProjects/lexsi-lab/docs/superpowers/specs/2026-11-11-stage2-discover-design.md → src/stage2_discover.py
- `Stage 3 Repair Design Specification` --implements--> `run_stage3()`  [INFERRED]
  C:/Users/dhruv/PycharmProjects/lexsi-lab/docs/superpowers/specs/2026-11-11-stage3-repair-design.md → src/stage3_repair.py
- `Multilingual Cyber Seed Matrix` --implements--> `generate_300_pairs()`  [EXTRACTED]
  C:/Users/dhruv/PycharmProjects/lexsi-lab/docs/superpowers/plans/2026-11-10-stage1-curate-dataset.md → src/stage1_curate.py
- `CuratorKIT Quality Gate & Provenance Engine` --implements--> `run_quality_gate()`  [EXTRACTED]
  C:/Users/dhruv/PycharmProjects/lexsi-lab/docs/superpowers/plans/2026-11-10-stage1-curate-dataset.md → src/stage1_curate.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **4-Tier Lexsi Alignment Stack Pipeline** — src_stage1_curate, src_stage2_discover, src_stage3_repair, src_stage4_eval [EXTRACTED 1.00]
- **Lexsi Ecosystem Repositories Chained** — docs_api_surface_verified_curatorkit, docs_api_surface_verified_circuitkit, docs_api_surface_verified_safetune, docs_api_surface_verified_auditkit, docs_api_surface_verified_lexsi_provenance_schema [EXTRACTED 1.00]
- **Stage 2 Surgical Circuit Discovery Pipeline** — docs_superpowers_specs_2026_11_11_stage2_discover_design_within_pair_length_equalization, docs_superpowers_specs_2026_11_11_stage2_discover_design_positive_baseline_margin_filter, docs_superpowers_specs_2026_11_11_stage2_discover_design_eap_ig, docs_superpowers_specs_2026_11_11_stage2_discover_design_submodule_ranked_surgical_mask [EXTRACTED 1.00]
- **Stage 3 SafeGrad + Surgical C-DeltaTheta Repair Flow** — docs_superpowers_specs_2026_11_11_stage3_repair_design_single_peft_wrap_safegrad, docs_superpowers_specs_2026_11_11_stage3_repair_design_three_input_apply_ctheta, docs_superpowers_specs_2026_11_11_stage3_repair_design_bit_for_bit_weight_preservation [EXTRACTED 1.00]
- **Circuit-Kavach Disjoint Calibration-to-Evaluation Chain** — docs_superpowers_specs_2026_11_10_stage1_dataset_design_disjoint_split, docs_superpowers_specs_2026_11_11_stage2_discover_design_stage2_spec, docs_superpowers_specs_2026_11_11_stage3_repair_design_stage3_spec, docs_superpowers_specs_2026_11_11_stage4_eval_design_auditkit_eval [INFERRED 0.95]
- **Stage 1 Multi-Split HF Dataset Publishing Contract** — artifacts_stage1_dataset_readme_circuit_kavach_dataset, artifacts_stage1_dataset_readme_disjoint_splits, agents_skills_hf_dataset_splits_skill_explicit_yaml_splits, artifacts_stage1_dataset_dataset_card_circuit_kavach_card [INFERRED 0.95]

## Communities (36 total, 19 thin omitted)

### Community 0 - "Stage 2: CircuitKIT EAP-IG & Discovery Pipeline"
Cohesion: 0.05
Nodes (44): SafeTune Circuit Adapter Regex Contract, 4-Pillar Circuit Evaluation & KL Diagnostic, EAP-IG (EAP-IG-inputs Discovery), apply_surgical_layer_cap(), _arch_module_sets(), calibrate_answer_tokens(), CalibratedSafetyTaskSpec, CalibrationResult (+36 more)

### Community 1 - "Stage 3: SafeTune & C-DeltaTheta Repair Core"
Cohesion: 0.06
Nodes (28): SafeTune apply_ctheta (C-Delta-Theta Surgical Transplant), SafeTune SafeGradTrainer, build_arg_parser(), build_stage3_sft_and_safety_splits(), evaluate_selectivity_on_val(), _format_prompt_and_full(), _greedy_generate_texts(), load_base_model_and_tokenizer() (+20 more)

### Community 2 - "Stage 1: CuratorKIT Twin Curation & Config"
Cohesion: 0.07
Nodes (19): Multilingual Cyber Seed Matrix, Stage 1 Dataset Curation & HF Upload Plan, Stage 1 Dataset Design Specification, build_arg_parser(), ContrastPair, export_provenance_bundle(), generate_300_pairs(), load_jsonl() (+11 more)

### Community 3 - "Shared Pytest & Test Fixtures"
Cohesion: 0.06
Nodes (13): test_auditkit_import_and_contract(), test_curatorkit_import_and_contract(), test_python_version_is_3_12(), test_src_scaffold_and_pythonpath(), test_vendor_repositories_accessible(), _isolate_auditkit_cache(), test_decoupled_two_phase_vram_unload_greedy_and_provenance(), test_optional_wildguard_and_lmeval_graceful_fallback() (+5 more)

### Community 4 - "Stage 1 Provenance & Manifest Schema"
Cohesion: 0.06
Nodes (34): checksums, calibration_100_pairs.jsonl, dataset_card.md, eval_200_pairs.jsonl, lexsi_provenance.json, README.md, rejected.jsonl, created_at (+26 more)

### Community 5 - "Stage 3 Unit & Cohere2 Parity Tests"
Cohesion: 0.12
Nodes (10): _create_tiny_cohere2_checkpoint(), _sample_pairs(), test_low_vram_t4_offloads_ref_model_to_cpu_and_releases_vram(), test_real_gpt2_cpu_stage3_end_to_end_with_stage1_calibration(), test_single_peft_wrap_unmerged_lora_saved_before_merge_and_ctheta_bit_identity(), test_stage3_t4_float16_ref_offload_use_cache_and_hydrate_hf(), _fake_snapshot_download(), test_tokenized_dataset_preserves_supervised_tokens_under_long_system_preamble() (+2 more)

### Community 7 - "Stage 4 Reporting & ROI Synthesis"
Cohesion: 0.18
Nodes (12): Stage 4 Evaluation Design Specification, build_roi_comparison_table(), _default_checkpoint_specs(), load_eval_pairs(), _load_generations_jsonl(), main(), _read_stage3_audit(), _resolve_checkpoint_specs_from_stage3() (+4 more)

### Community 9 - "Project Rules, Lexsi API Verification & Stack Docs"
Cohesion: 0.12
Nodes (12): Lexsi Gemma-4-31B-it Safety Repair Reference Model Cards, AuditKIT (v1.1.6), CircuitKIT (v0.2.1), Cohere2 Periodic NoPE and SWA Architecture, CuratorKIT (v0.1.21), SafeTune (v0.1.12 / LSAL v1.2), Lexsi Hackathon Track 2 Quantitative Rubric, CircuitKavach-Aya (+4 more)

### Community 10 - "Stage 4 Decoupled Two-Phase VRAM Generation"
Cohesion: 0.14
Nodes (8): Stage 4 L4 (24GB) & T4 (16GB) VRAM OOM Mitigation, _ensure_left_padding(), _phase1_generate_for_checkpoint(), _resolve_checkpoint_model(), resolve_stage4_runtime_profile(), _spec_cache_key(), Stage4RuntimeProfile, _torch_dtype()

### Community 11 - "Stage 4 Precomputed Scorer Harness"
Cohesion: 0.15
Nodes (6): build_eval_samples(), evaluate_precomputed_split(), _make_precomputed_callable(), _rehydrate_samples_with_tags(), _safe_compare_section(), _samples_digest()

### Community 12 - "Stage 4 Offline Refusal & Helpfulness Scorers"
Cohesion: 0.20
Nodes (9): build_benign_scorers(), benign_helpfulness(), benign_over_refusal(), build_harmful_scorers(), attack_refusal_combined(), attack_refusal_detector(), attack_refusal_gcg(), _classify_refusal_signals() (+1 more)

### Community 13 - "Stage 2 Length Equalization & Calibration Tests"
Cohesion: 0.19
Nodes (12): _create_tiny_hf_checkpoint(), _sample_pairs(), test_apply_ctheta_torch_equal_on_cohere2_and_custom_path_gpt2(), test_calibrated_task_spec_csv_cache_collision_resistance(), test_calibrated_task_spec_preserves_align_none_left_pad_and_templated(), test_cli_main_entrypoint_and_edge_case_rejections(), test_end_to_end_stage2_on_tiny_cohere2_model(), test_load_contrast_pairs_validation_edge_cases() (+4 more)

### Community 14 - "Stage 4 Attack Non-Refusal Detector"
Cohesion: 0.18
Nodes (3): _AttackNonRefusalDetector, _bound_redteam_model(), run_redteam_suite()

### Community 15 - "Stage 1 SHA-256 Checksum Integrity"
Cohesion: 0.29
Nodes (8): calibration_100_pairs.jsonl (SHA-256 e9aabcf3), dataset_card.md & README.md Identical SHA-256 (87e5ad98), eval_200_pairs.jsonl (SHA-256 7f5497db), lexsi_provenance.json (SHA-256 781ceeeb), Stage 1 SHA-256 Checksums Manifest, CuratorKIT Provenance Compliance, Stage 2 Circuit Discovery Design Specification, Stage 3 Repair Design Specification

### Community 16 - "Stage 1 Dataset Card & HF Split YAML Spec"
Cohesion: 0.33
Nodes (3): CircuitKavach Dataset Card, CircuitKavach Multilingual Cybersecurity and Phishing Selectivity Dataset, Universal Lineage Schema (lexsi.provenance/1)

### Community 18 - "Stage 4 Optional WildGuard & lm-eval Hooks"
Cohesion: 0.29
Nodes (4): _infer_wildguard_split(), maybe_run_lmeval(), maybe_run_wildguard(), _release_vram()

### Community 23 - "Hardware-Aware T4/L4 SafeGrad Profiles"
Cohesion: 0.50
Nodes (3): Stage 2 L4 (24GB), T4 (16GB) & Local CPU (6GB) Profiles, SafeGrad Gradient Surgery Algorithm, Stage 3 L4 (24GB) & T4 (16GB) Low-VRAM Offload

## Knowledge Gaps
- **20 isolated node(s):** `circuit-kavach-aya`, `GitNexus Code Intelligence Index (lexsi-lab-track2)`, `Lexsi Gemma-4-31B-it Safety Repair Reference Model Cards`, `Multilingual and Indic Code-Mixed Registers (en, hi_en, gu_en, kn_en)`, `Universal Lineage Schema (lexsi.provenance/1)` (+15 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 260 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **19 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `run_stage2()` connect `Stage 2: CircuitKIT EAP-IG & Discovery Pipeline` to `Stage 1 SHA-256 Checksum Integrity`?**
  _High betweenness centrality (0.031) - this node is a cross-community bridge._
- **What connects `circuit-kavach-aya`, `GitNexus Code Intelligence Index (lexsi-lab-track2)`, `Lexsi Gemma-4-31B-it Safety Repair Reference Model Cards` to the rest of the system?**
  _20 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Stage 2: CircuitKIT EAP-IG & Discovery Pipeline` be split into smaller, more focused modules?**
  _Cohesion score 0.052659716653301256 - nodes in this community are weakly interconnected._
- **Why does `test_leading_space_single_token_calibration_roundtrip()` connect `Stage 2 BPE Roundtrip Calibration Tests` to `Stage 2 Unit & Hugging Face Runner Tests`, `Stage 2 Length Equalization & Calibration Tests`?**
  _High betweenness centrality (0.020) - this node is a cross-community bridge._
- **Should `Stage 3: SafeTune & C-DeltaTheta Repair Core` be split into smaller, more focused modules?**
  _Cohesion score 0.06327683615819209 - nodes in this community are weakly interconnected._
- **Why does `_classify_refusal_signals()` connect `Stage 4 Offline Refusal & Helpfulness Scorers` to `Stage 4: AuditKIT Core & Evaluation Orchestration`, `Stage 4 Attack Non-Refusal Detector`?**
  _High betweenness centrality (0.019) - this node is a cross-community bridge._
- **Should `Stage 1: CuratorKIT Twin Curation & Config` be split into smaller, more focused modules?**
  _Cohesion score 0.0726950354609929 - nodes in this community are weakly interconnected._