# Run Decision Record: CircuitKavach-Aya (`lexsi-lab-track2`)

| Field | Value |
| :--- | :--- |
| **Document Status** | Approved / Operational |
| **Target Model** | `CohereLabs/tiny-aya-fire` (3.35B parameters, `cohere2` architecture) |
| **Execution Hardware** | Cloud: Hugging Face Jobs (`l4x1`, NVIDIA L4 24 GB VRAM) & ZeroGPU; Local: Windows Laptop (RTX 4050 6 GB / CPU verification) |
| **Target Repositories** | Dataset: [`soapyyapps/circuit-kavach-cyber-codemixed`](https://huggingface.co/datasets/soapyyapps/circuit-kavach-cyber-codemixed) <br> Space: [`soapyyapps/circuit-kavach-aya`](https://huggingface.co/spaces/soapyyapps/circuit-kavach-aya) |
| **Pipeline Scope** | Stages 1–4 (Curation, EAP-IG Discovery, SafeTune Surgical Repair, AuditKIT Evaluation) |

---

## 1. Compute & Hardware Partitioning

### Decision 1.1: Local CPU Testing vs. Cloud GPU Execution
- **Context / Initial Setup:** The local development workstation is a Windows laptop with an NVIDIA RTX 4050 Laptop GPU (6,141 MB total VRAM) and a CPU-only PyTorch `.venv` installation.
- **Problem / Failure Encountered:** Loading 3.35B parameter weights in `bfloat16` requires ~6.7 GB static memory before activations, gradients, or `CircuitKIT` hooks are attached. Local execution causes immediate out-of-memory (OOM) allocation failures.
- **Decision / Rationale:** Enforce a strict hardware boundary:
  1. *Local Verification:* Execute all 54 pytest unit and integration tests strictly on CPU using `"gpt2"` (`float32`) and synthetic 2-layer `cohere2` checkpoints.
  2. *Cloud Execution:* Offload all 3.35B parameter training, attribution, and evaluation runs to Hugging Face Jobs using the `l4x1` flavor (NVIDIA L4, 24 GB VRAM, `bfloat16`).
- **Outcome:** Zero OOM crashes on the local host; 100% test pass rate across all 54 test suites on CPU.

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
- **Problem / Failure Encountered:** Unpadded contrast pairs with differing token counts cause token positional drift in causal attention masks, distorting downstream edge attribution matrices.
- **Decision / Rationale:** Right-pad all contrast pairs within each sample to identical token lengths before running EAP-IG attribution, and filter out pairs where the pre-attribution logit margin $\Delta m = m(\text{clean}) - m(\text{corrupt})$ falls below $0.5$.
- **Outcome:** Attribution stability achieved on `l4x1` (`6af80c1423985fb9cd817385`), yielding a Mean Jaccard similarity of 0.9788, Spearman $\rho$ of 0.9997, and isolating 4 of 72 submodules (94.44% frozen parameter retention).

---

## 4. Stage 3 Circuit Repair (SafeTune)

### Decision 4.1: Reference Model GPU Allocation vs. CPU Offload
- **Context / Initial Setup:** Stage 3 trains a LoRA adapter on identified circuit submodules using KL-divergence distillation against a frozen reference model.
- **Problem / Failure Encountered:** Enabling `--ref-offload-cpu` caused the reference forward pass to take 147.6 seconds per step, triggering a 60-minute timeout after only 23 of 80 steps.
- **Decision / Rationale:** Analyzed static memory allocations on the 24 GB NVIDIA L4:
  - Base student model (`tiny-aya-fire` 3.35B in `bf16` with LoRA): ~7.2 GB VRAM.
  - Frozen reference model (`bf16`, `eval()`, `no_grad`): ~6.7 GB VRAM.
  - Combined static allocation: ~13.9 GB, leaving >10 GB headroom for activations.
  - Removed `--ref-offload-cpu`, enabled `--gradient-checkpointing`, and reduced `--batch-size` from 2 to 1 (doubling gradient steps from 80 to 160).
- **Outcome:** Step latency dropped from 147.6 seconds to 0.99 seconds (150× speedup). Stage 3 completed 160 steps in 159.1 seconds, achieving steady convergence to a final `train_loss` of 0.8305 without gradient overflow, verified bit-identity across all 68 frozen submodules, and uploaded Checkpoints 2 and 3.

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

### Observation 7.1: Asymmetric Generation Caps
- **Observation:** In evaluation runs containing both malicious prompts and benign defensive tasks, refusals occur within the first 30 generated tokens, whereas comprehensive benign code patches require larger token allowances.
- **Recommendation:** Future evaluation harnesses should adopt an asymmetric cap: 96 tokens for malicious prompts and 256 tokens for benign prompts, optimizing both runtime efficiency and explanatory completeness.
