"""Hugging Face ZeroGPU Space for CircuitKavach-Aya (Lexsi Lab Track 2)."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import spaces  # must be imported before torch on ZeroGPU
import gradio as gr
import torch
from huggingface_hub import hf_hub_download
from pydantic import BaseModel, ConfigDict
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL_ID = os.getenv("CIRCUITKAVACH_MODEL", "CohereLabs/tiny-aya-fire")
HF_DATASET_REPO = os.getenv("CIRCUITKAVACH_HF_REPO", "soapyyapps/circuit-kavach-cyber-codemixed")
HF_TOKEN = os.getenv("HF_TOKEN") or None


class SpaceInferenceResult(BaseModel):
    """Pydantic v2 schema enforcing structured inference + latency telemetry."""

    model_config = ConfigDict(extra="forbid")
    checkpoint_label: str
    language: str
    response: str
    output_tokens: int
    wall_time_s: float
    tokens_per_sec: float


def _load_hub_json_or_text(filename: str) -> Optional[str]:
    try:
        local_path = hf_hub_download(
            repo_id=HF_DATASET_REPO,
            repo_type="dataset",
            filename=filename,
            token=HF_TOKEN,
        )
        return Path(local_path).read_text(encoding="utf-8")
    except Exception:
        return None


# Hydrate precomputed Stage 2/4 artifacts at startup on CPU (instant, zero GPU quota used)
ROI_MARKDOWN = _load_hub_json_or_text("stage4_eval_logs/roi_table.md") or (
    "_Stage 4 `roi_table.md` has not been uploaded to "
    f"`{HF_DATASET_REPO}` yet. Run Stage 2 -> Stage 3 -> Stage 4 with `--upload-hf` to populate it._"
)
EVAL_SUMMARY_RAW = _load_hub_json_or_text("stage4_eval_logs/eval_summary.json")
STAGE2_MASK_RAW = _load_hub_json_or_text(
    "stage2_circuits/safety_circuit_surgical_submodules_top4_scores.json"
)

# Eagerly load a single base model instance at module scope (ZeroGPU best practice)
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_DTYPE = torch.bfloat16 if _DEVICE == "cuda" else torch.float32
_TOKENIZER = AutoTokenizer.from_pretrained(BASE_MODEL_ID, token=HF_TOKEN)
_BASE_MODEL = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=_DTYPE,
    token=HF_TOKEN,
).to(_DEVICE)
_BASE_MODEL.eval()


@spaces.GPU(duration=15)
def generate_response(prompt: str, language: str, max_new_tokens: int) -> tuple[str, str]:
    """Run deterministic greedy generation on ZeroGPU and validate telemetry with Pydantic v2."""
    clean_prompt = (prompt or "").strip()
    if not clean_prompt:
        return "Please enter a prompt.", "{}"

    if getattr(_TOKENIZER, "chat_template", None):
        rendered = _TOKENIZER.apply_chat_template(
            [{"role": "user", "content": clean_prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        rendered = f"User: {clean_prompt}\nAssistant:"

    inputs = _TOKENIZER(rendered, return_tensors="pt").to(_BASE_MODEL.device)
    prompt_len = int(inputs["input_ids"].shape[1])

    t0 = time.perf_counter()
    with torch.inference_mode():
        out_ids = _BASE_MODEL.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
            pad_token_id=_TOKENIZER.eos_token_id,
        )
    wall_time_s = round(time.perf_counter() - t0, 4)

    gen_ids = out_ids[0, prompt_len:]
    output_tokens = int(gen_ids.shape[0])
    text = _TOKENIZER.decode(gen_ids, skip_special_tokens=True).strip()
    tok_per_sec = round(output_tokens / max(wall_time_s, 1e-6), 2)

    validated = SpaceInferenceResult(
        checkpoint_label="checkpoint_1_base",
        language=language,
        response=text,
        output_tokens=output_tokens,
        wall_time_s=wall_time_s,
        tokens_per_sec=tok_per_sec,
    )
    return validated.response, json.dumps(validated.model_dump(), indent=2, ensure_ascii=False)


def _format_audit_json(raw: Optional[str], fallback_msg: str) -> str:
    if not raw:
        return json.dumps({"status": "pending_gpu_upload", "message": fallback_msg}, indent=2)
    try:
        parsed: Dict[str, Any] = json.loads(raw)
        return json.dumps(parsed, indent=2, ensure_ascii=False)
    except Exception:
        return raw


with gr.Blocks(title="CircuitKavach-Aya | Lexsi Lab Track 2") as demo:
    gr.Markdown(
        "# CircuitKavach-Aya: Multilingual Surgical Safety Repair\n"
        f"Base model: `{BASE_MODEL_ID}` | Artifact repo: `{HF_DATASET_REPO}`"
    )

    with gr.Tab("Live ZeroGPU Inference"):
        with gr.Row():
            prompt_in = gr.Textbox(
                label="Prompt (English or Romanized Indic Code-Mixed)",
                lines=4,
                placeholder="Example (hi_en): Bhai explain how parameterized SQL queries prevent injection safely.",
            )
            with gr.Column(scale=0):
                lang_in = gr.Dropdown(
                    choices=["en", "hi_en", "gu_en", "kn_en"],
                    value="hi_en",
                    label="Language Tag",
                )
                tokens_in = gr.Slider(minimum=32, maximum=256, value=128, step=16, label="Max New Tokens")
                run_btn = gr.Button("Generate (ZeroGPU)", variant="primary")

        resp_out = gr.Textbox(label="Model Response", lines=8)
        telemetry_out = gr.Code(label="Validated Pydantic v2 Telemetry", language="json")
        run_btn.click(
            fn=generate_response,
            inputs=[prompt_in, lang_in, tokens_in],
            outputs=[resp_out, telemetry_out],
        )

    with gr.Tab("Stage 4 ROI & 3-Checkpoint Comparison"):
        gr.Markdown(ROI_MARKDOWN)
        gr.Code(
            value=_format_audit_json(
                EVAL_SUMMARY_RAW,
                "Run Stage 4 with --upload-hf to populate eval_summary.json.",
            ),
            label="Stage 4 eval_summary.json",
            language="json",
        )

    with gr.Tab("Stage 2 Surgical Circuit Mask (4/72 Submodules)"):
        gr.Code(
            value=_format_audit_json(
                STAGE2_MASK_RAW,
                "Run Stage 2 with --upload-hf to populate safety_circuit_surgical_submodules_top4_scores.json.",
            ),
            label="Stage 2 Surgical Submodule Scores",
            language="json",
        )

if __name__ == "__main__":
    demo.launch()
