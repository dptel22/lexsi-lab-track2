# Run Decision Record: CircuitKavach-Aya (`lexsi-lab-track2`)

| Field | Value |
| :--- | :--- |
| **Document Status** | Approved / Operational |
| **Target Model** | `CohereLabs/tiny-aya-fire` (3.35B parameters, `cohere2` architecture) |
| **Execution Hardware** | Cloud: Hugging Face Jobs (`l40sx1` NVIDIA L40S 48 GB VRAM for Stage 2; `l4x1` NVIDIA L4 24 GB VRAM for Stages 3–4) & ZeroGPU; Local: Windows Laptop (RTX 4050 6 GB / CPU verification) |
| **Target Repositories** | Dataset: [`soapyyapps/circuit-kavach-cyber-codemixed`](https://huggingface.co/datasets/soapyyapps/circuit-kavach-cyber-codemixed) <br> Space: [`soapyyapps/circuit-kavach-aya`](https://huggingface.co/spaces/soapyyapps/circuit-kavach-aya) |
| **Pipeline Scope** | Stages 1–4 (Curation, EAP-IG Discovery, SafeTune Surgical Repair, AuditKIT Evaluation) |

---

## 1. Compute & Hardware Partitioning

### Decision 1.1: Local CPU Testing vs. Cloud GPU Execution
- **Context / Initial Setup:** The local development workstation is a Windows laptop with an NVIDIA RTX 4050 Laptop GPU (6,141 MB total VRAM) and a CPU-only PyTorch `.venv` installation.
- **Problem / Failure Encountered:** Loading 3.35B parameter weights in `bfloat16` requires ~6.7 GB static memory before activations, gradients, or `CircuitKIT` hooks are attached. Local execution causes immediate out-of-memory (OOM) allocation failures.
- **Decision / Rationale:** Enforce a strict hardware boundary:
  1. *Local Verification:* Execute all 58 pytest unit and integration tests strictly on CPU using `"gpt2"` (`float32`) and synthetic 2-layer `cohere2` checkpoints.
  2. *Cloud Execution:* Offload Stage 2 (`l40sx1` 48 GB VRAM for templated `HookedTransformer` EAP-IG) and Stages 3–4 (`l4x1` 24 GB VRAM, `bfloat16`) to Hugging Face Jobs.
- **Outcome:** Zero OOM crashes on the local host; 100% test pass rate across all 58 tests on CPU.

---

## 2. Containerization & HF Jobs Staging

### Decision 2.1: Sanitized Workspace Sync & Defense-in-Depth Isolation
- **Context / Initial Setup:** `hf jobs run` mounts local files by synchronizing the working directory to an HF Bucket (`hf://buckets/soapyyapps/jobs-artifacts`).
- **Problem / Failure Encountered:** Directly syncing the workspace directory uploads the local `.venv` (~4 GB of Windows binaries) and risks exposing local `.env` configuration files to the bucket.
- **Decision / Rationale:** A two-stage file exclusion mechanism was implemented:
  1. *Transient Staging:* A transient staging directory (`lexsi_hf_stage`) is created via `tempfile.gettempdir()`, copying only `src/`, `vendor/`, `artifacts/`, `scripts/`, and `pyproject.toml`, explicitly omitting `.venv`, `.env`, and `__pycache__`.
  2. *Repository-level `.hfignore`:* A dedicated `.hfignore` file was established in the repository root to ensure direct `hf` CLI sync commands ignore `.venv/`, `.env*`, `.git/`, and caches by default.
- **Outcome:** A 2,381-file payload (~120 MB total) uploaded without `.venv` binaries or `.env` files.

### Decision 2.2: Dynamic Python 3.12 Bootstrap via `uv`
- **Context / Initial Setup:** The official container `pytorch/pytorch:2.7.0-cuda12.6-cudnn9-runtime` bundles Python 3.11.
- **Problem / Failure Encountered:** `SafeTune` enforces `python >= 3.12` in its build metadata. Stage 2 failed during `pip install ./vendor/SafeTune` with `RuntimeError: SafeTune requires Python 3.12 or higher`.
- **Decision / Rationale:** Injected a runtime Python 3.12 virtual environment bootstrap directly inside the container entrypoint:
  ```bash
  python -m pip install --no-cache-dir --quiet uv \
    && uv venv /tmp/py312 --python 3.12 --seed \
    && uv pip install --python /tmp/py312/bin/python ...
  ```
- **Outcome:** Container provisions an isolated CPython 3.12.16 environment in under 15 seconds, satisfying all toolchain requirements.

---

## 3. Stage 2 Circuit Discovery (CircuitKIT)

### Decision 3.1: Token Length Equalization & Margin Preflight
- **Context / Initial Setup:** Stage 2 computes Edge Attribution Patching via Integrated Gradients (EAP-IG) over 100 clean/corrupted contrast pairs across English, Hindi-English, Gujarati-English, and Kannada-English code-mixed registers.
- **Problem / Failure Encountered:** Unpadded contrast pairs with differing token counts cause token positional drift in causal attention masks. In addition, v1 `tiny-aya-fire` discovery failed to recognize `dict`-typed `CohereTokenizerFast.chat_template` (`templated=False`), calibrated on un-templated tokens (`"ploit"`/`" pare"`), and then hit L4 (`22.03 GiB`) OOM once `templated=True` added Cohere's ~360-token contextual safety preamble (`n_pos ≈ 375`, pushing `use_attn_result` `(z * w)` + autograd graph to `21.8–27.1 GiB`).
- **Decision / Rationale:** Right-pad contrast pairs within each sample to identical token lengths, accept `dict`-typed `chat_template` in `_has_chat_template` without gating mock tokenizers, weight `"en"` preflight rows `3.0x` inside `calibrate_answer_tokens`, filter pairs with $\Delta m > 0.5$, and allocate `l40sx1` (`48 GB` VRAM) for Stage 2 (`6af96d1629dd05ff03686a08`).
- **Outcome:** Stage 2 (`6af96d1629dd05ff03686a08`, `51m49s` on `l40sx1`) ran with `templated=true`, calibrated on `" sorry"` (`40449`) vs. `" when"` (`1434`, `logit_diff_gap=16.21`), retained `99/100` pairs (`en=25`, `gu_en=25`, `hi_en=25`, `kn_en=24`), achieved `mean_jaccard=0.9683`, `mean_spearman=0.9998`, `ablation_score=0.6549`, and isolated `layers.0.mlp`, `layers.16.self_attn`, `layers.21.self_attn`, and `layers.20.self_attn` (`94.44%` projection blocks frozen).

---

## 4. Stage 3 Circuit Repair (SafeTune)

### Decision 4.1: Supervised Refusal Surgery (`alignment_loss_mode="sft"`), CPU Reference Offload & Two-Group `AdamW`
- **Context / Initial Setup:** Stage 3 trains a LoRA adapter via `SafeGradHFTrainer` and transplants circuit-masked deltas $\Delta\Theta$ via `C-ΔΘ`.
- **Problem / Failure Encountered:** In v1, passing a CPU-offloaded unaligned `tiny-aya-fire` `reference_model` caused `SafeGradHFTrainer` to run KL-divergence toward the unaligned base model (ignoring `_REFUSAL_TEMPLATES` labels and worsening `gu_en`/`kn_en` refusal from `6%` to `2%`) and took `147.6 s/step` when queried on every step. Uniform `AdamW` updates across all 72 projection blocks also diluted deltas inside the 4 selected circuit submodules before masking.
- **Decision / Rationale:** Add explicit `alignment_loss_mode: str = "auto"` (`"auto" | "sft" | "kl"`) to `SafeGradHFTrainer`, opt Stage 3 into `"sft"` (`_REFUSAL_TEMPLATES` cross-entropy loss with `rho=1.0`, `epochs=3.0`) while defaulting `ref_offload_cpu=True` (`0 GB` GPU VRAM), and partition trainable LoRA parameters via `_matched_param_keys` into a two-group `torch.optim.AdamW` (`2e-4` inside circuit vs. `0.10x` outside).
- **Outcome:** Stage 3 (`6af97a2723985fb9cd8320bb`, `15m50s` on `l4x1`) converged in 3 epochs (`train_loss=0.7112`), raised validation `attack_refusal_rate` from `45.0%` to `95.0%` (`alpha=1.0`, `0.0%` benign over-refusal), kept `97.45%` of parameters bit-identical (`2.55%` modified across `15/291` tensors), and Stage 4 (`6af97fa323985fb9cd8325eb`, `l4x1`) verified `94.5%` held-out refusal (`+54.5 pp` over base, `p=0.000999`) with `0.0%` benign over-refusal.

---

## 5. Stage 4 Circuit Evaluation (AuditKIT)

### Decision 5.1: Token Generation Cap (`max_new_tokens`)
- **Context / Initial Setup:** Stage 4 evaluates 3 model checkpoints (`checkpoint_1_base`, `checkpoint_2_global_safetune`, `checkpoint_3_surgical_c_delta_theta`) across 400 test prompts (200 harmful + 200 benign twins) plus red-team suites.
- **Problem / Failure Encountered:** Generating up to 512 tokens across 1,200 generations timed out after 90 minutes.
- **Decision / Rationale:** Capped `--max-new-tokens` at 128 and increased job timeout to 120 minutes. Safety refusal prefixes and benign explanations resolve in under 100 tokens; generating 512 tokens increases execution time 4× with negligible safety signal gain.
- **Outcome:** Generation volume reduced by 4×, ensuring completion inside the GPU allocation window.

### Decision 5.2: Decoupled Evaluation & Pydantic v2 Schema Telemetry
- **Context / Initial Setup:** Stage 4 measures both behavioral safety metrics (refusal rate, over-refusal, helpfulness, selectivity gap) and operational inference latency.
- **Problem / Failure Encountered:** Running scoring judges, regex detectors, and text generation concurrently inside the same execution loop risks GPU memory fragmentation and complicates failure isolation.
- **Decision / Rationale:** Partition evaluation into two sequential phases:
  1. *Phase 1 (GPU Generation):* Serialized greedy generation on GPU, strictly unloading each model from VRAM and calling `torch.cuda.empty_cache()` before loading the next.
  2. *Phase 2 (CPU Scoring):* Offline scoring using `AuditKIT` judges (`StringMatchJudge`, `RefusalDetector`), validating summary outputs against `Stage4EvalSummarySchema` (Pydantic v2).
- **Outcome:** Zero GPU memory collisions during scoring; outputs are deterministically validated and saved to `roi_table.md` and `eval_summary.json` with embedded latency metrics.

---

## 6. Interactive Space Deployment (ZeroGPU)

### Decision 6.1: Single Base Model Architecture on ZeroGPU
- **Context / Initial Setup:** The production demo is deployed on Hugging Face Spaces using the ZeroGPU hardware tier.
- **Problem / Failure Encountered:** Loading three distinct 3.35B model checkpoints simultaneously exceeds the 24 GB ZeroGPU memory allocation and exhausts ephemeral quota pools.
- **Decision / Rationale:** Load a single base `tiny-aya-fire` model at module scope. Route interactive user prompts through `@spaces.GPU(duration=15)`, while serving precomputed Stage 2 circuit heatmaps and Stage 4 ROI comparative metrics via static CPU tabs.
- **Outcome:** The Space ([`soapyyapps/circuit-kavach-aya`](https://huggingface.co/spaces/soapyyapps/circuit-kavach-aya)) is operational in production with zero idle GPU consumption.

---

## 7. Future Protocol Improvements

### Observation 7.2: Hardware Tiering for Templated Attribution vs. LoRA Training
- **Observation:** Enabling `tiny-aya-fire`'s `dict` chat template injects Cohere's ~360-token contextual safety preamble (~375 tokens per prompt), driving Stage 2 `HookedTransformer` `use_attn_result` autograd peak VRAM to `21.8–27.1 GiB` (OOM on `22.03 GiB` L4), whereas Stage 3 (`gradient_checkpointing=True`, `batch_size=1`, `ref_offload_cpu=True`, `alignment_loss_mode="sft"`) and Stage 4 serialized evaluation stay under `~12.5 GiB` VRAM.
- **Recommendation:** Allocate `l40sx1` (`48 GB` VRAM) for Stage 2 circuit discovery while retaining cost-efficient `l4x1` (`24 GB` VRAM) for Stages 3 and 4.
