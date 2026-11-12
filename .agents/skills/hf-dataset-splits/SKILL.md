---
name: hf-dataset-splits
description: >-
  Publish and verify multi-split JSONL datasets on Hugging Face Hub with explicit
  README.md YAML split configs so the Dataset Viewer detects every split cleanly.
  Use when uploading datasets or troubleshooting missing/merged HF splits.
---

# Hugging Face Multi-Split JSONL Publishing

## Checklist

1. **Declare splits explicitly in `README.md` YAML frontmatter** before uploading (never rely on HF filename auto-detection for custom split names such as `calibration` + `eval`):
   ```yaml
   ---
   configs:
   - config_name: default
     data_files:
     - split: calibration
       path: calibration_100_pairs.jsonl
     - split: eval
       path: eval_200_pairs.jsonl
   ---
   ```
2. **Upload the folder** via `HfApi.upload_folder` (preserving exact filenames referenced under `data_files`).
3. **Verify splits programmatically** after upload rather than assuming Hub auto-conversion succeeded:
   ```python
   from datasets import load_dataset

   for split in ("calibration", "eval"):
       ds = load_dataset("<user>/<repo>", split=split)
       assert len(ds) > 0
   ```
