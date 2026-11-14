#!/usr/bin/env bash
set -euo pipefail

# One-command launcher for CircuitKavach-Aya Stages 2-4 on Hugging Face Jobs.
# Uses per-second billing on l4x1 by default, passes HF_TOKEN securely via --secrets HF_TOKEN,
# and fails fast on missing credentials before submitting GPU jobs.

usage() {
  cat <<'EOF'
Usage: scripts/run_hf_jobs.sh <stage2|stage3|stage4> [options] [-- extra stage args...]

Options:
  --flavor <name>       HF Jobs GPU flavor (default: l4x1; e.g. t4-small, t4-medium, l4x1, a10g-small)
  --timeout <duration>  HF Jobs timeout (default: 60m)
  --hf-repo <repo_id>   Target HF dataset repo (default: soapyyapps/circuit-kavach-cyber-codemixed)
  --image <docker_img>  Container image (default: pytorch/pytorch:2.7.0-cuda12.6-cudnn9-runtime)
  --low-vram-t4         Pass --low-vram-t4 (16 GB T4 compatibility mode) to the stage script
  --dry-run             Print the exact `hf jobs run` command without validating secrets or submitting
  -h, --help            Show this help message
EOF
}

if [[ $# -lt 1 ]]; then
  usage >&2
  exit 2
fi

STAGE=""
FLAVOR="l4x1"
FLAVOR_SET=false
TIMEOUT="60m"
HF_REPO="soapyyapps/circuit-kavach-cyber-codemixed"
IMAGE="pytorch/pytorch:2.7.0-cuda12.6-cudnn9-runtime"
LOW_VRAM_T4=false
DRY_RUN=false
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    stage2|stage3|stage4)
      if [[ -n "$STAGE" ]]; then
        echo "Error: multiple stages specified ('$STAGE' and '$1')." >&2
        exit 2
      fi
      STAGE="$1"
      shift
      ;;
    --flavor)
      FLAVOR="${2:-}"
      FLAVOR_SET=true
      shift 2
      ;;
    --timeout)
      TIMEOUT="${2:-}"
      shift 2
      ;;
    --hf-repo)
      HF_REPO="${2:-}"
      shift 2
      ;;
    --image)
      IMAGE="${2:-}"
      shift 2
      ;;
    --low-vram-t4)
      LOW_VRAM_T4=true
      shift
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -z "$STAGE" ]]; then
  echo "Error: stage must be one of stage2, stage3, stage4." >&2
  usage >&2
  exit 2
fi

case "$STAGE" in
  stage2)
    STAGE_SCRIPT="src/stage2_discover.py"
    STAGE_FLAGS="--model CohereLabs/tiny-aya-fire --dataset artifacts/stage1_dataset/calibration_100_pairs.jsonl --output-dir artifacts/stage2_circuits"
    ;;
  stage3)
    STAGE_SCRIPT="src/stage3_repair.py"
    STAGE_FLAGS="--model CohereLabs/tiny-aya-fire --output-dir artifacts/stage3_repair --gradient-checkpointing --batch-size 1 --hydrate-hf"
    ;;
  stage4)
    STAGE_SCRIPT="src/stage4_eval.py"
    STAGE_FLAGS="--model CohereLabs/tiny-aya-fire --output-dir artifacts/stage4_eval_logs --hydrate-hf"
    ;;
esac

if [[ "$FLAVOR" == t4* ]]; then
  LOW_VRAM_T4=true
elif [[ "$LOW_VRAM_T4" == "true" && "$FLAVOR_SET" == "false" ]]; then
  FLAVOR="t4-small"
fi

if [[ "$LOW_VRAM_T4" == "true" ]]; then
  STAGE_FLAGS="$STAGE_FLAGS --low-vram-t4"
fi

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  for arg in "${EXTRA_ARGS[@]}"; do
    STAGE_FLAGS="$STAGE_FLAGS $(printf '%q' "$arg")"
  done
fi

HF_CLI="${HF_BIN:-}"
if [[ -z "$HF_CLI" ]]; then
  if command -v hf >/dev/null 2>&1; then
    HF_CLI="hf"
  elif [[ -x ".venv/Scripts/hf.exe" ]]; then
    HF_CLI=".venv/Scripts/hf.exe"
  elif [[ -x ".venv/bin/hf" ]]; then
    HF_CLI=".venv/bin/hf"
  else
    HF_CLI="hf"
  fi
fi

# Mount /workspace read-only cleanly, copy only runtime code/artifacts (excluding .venv/.env) to /tmp/lexsi, install vendored kits, and execute stage.
PY312_SETUP="python -m pip install --no-cache-dir --quiet uv && uv venv /tmp/py312 --python 3.12 --seed && uv pip install --python /tmp/py312/bin/python --quiet torch ./vendor/CircuitKIT ./vendor/SafeTune ./vendor/AuditKIT python-dotenv huggingface_hub datasets accelerate peft trl"
CONTAINER_CMD="set -euo pipefail && rm -rf /tmp/lexsi && mkdir -p /tmp/lexsi && cp -a /workspace/src /workspace/vendor /workspace/scripts /workspace/pyproject.toml /tmp/lexsi/ && if [ -d /workspace/artifacts ]; then cp -a /workspace/artifacts /tmp/lexsi/; fi && cd /tmp/lexsi && ${PY312_SETUP} && /tmp/py312/bin/python ${STAGE_SCRIPT} ${STAGE_FLAGS} --upload-hf --hf-repo $(printf '%q' "$HF_REPO")"

HF_CMD=(
  "$HF_CLI" jobs run
  --flavor "$FLAVOR"
  --timeout "$TIMEOUT"
  --secrets HF_TOKEN
  -v .:/workspace:ro
  "$IMAGE"
  --
  bash -lc "$CONTAINER_CMD"
)

if [[ "$DRY_RUN" == "true" ]]; then
  printf 'DRY RUN:'
  printf ' %q' "${HF_CMD[@]}"
  printf '\n'
  exit 0
fi

# Fail-fast HF authentication validation before incurring cloud GPU charges.
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "Error: HF_TOKEN must be exported in your environment so --secrets HF_TOKEN can pass it to HF Jobs." >&2
  exit 1
fi

if ! "$HF_CLI" auth whoami >/dev/null 2>&1; then
  PY_CHECK="${PYTHON_BIN:-python}"
  if ! command -v "$PY_CHECK" >/dev/null 2>&1; then
    [[ -x ".venv/Scripts/python.exe" ]] && PY_CHECK=".venv/Scripts/python.exe"
    [[ -x ".venv/bin/python" ]] && PY_CHECK=".venv/bin/python"
  fi
  if command -v "$PY_CHECK" >/dev/null 2>&1 || [[ -x "$PY_CHECK" ]]; then
    "$PY_CHECK" -c "import os; from huggingface_hub import HfApi; HfApi(token=os.environ['HF_TOKEN']).whoami()" >/dev/null 2>&1 || {
      echo "Error: HF_TOKEN authentication check failed." >&2
      exit 1
    }
  else
    echo "Error: HF CLI authentication check failed for HF_TOKEN." >&2
    exit 1
  fi
fi

exec "${HF_CMD[@]}"
