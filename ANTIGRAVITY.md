# ANTIGRAVITY MASTER ENGINEERING DIRECTIVE: CircuitKavach-Aya

You are the lead autonomous implementation agent operating under the `@superpowers` (Brainstorming Locked -> Git Worktrees -> TDD RED-GREEN-REFACTOR -> Verification Before Completion) and `@hackathon-engineer` workflows.

## 0. IRON LAW OF SOURCE VERIFICATION (DO NOT ASSUME APIS)

The four mandatory hackathon libraries (`CuratorKIT`, `CircuitKIT`, `SafeTune`, `AuditKIT`) and the target model (`CohereLabs/tiny-aya-fire`) are actively developed in late 2026. **Your pre-cutoff training weights DO NOT contain their exact function signatures, class names, or CLI flags.**

**MANDATORY RULE BEFORE WRITING CODE FOR ANY STAGE:**
1. Clone or install the official repository into a local `vendor/` directory or virtual environment first.
2. Inspect the real source code (`README.md`, `pyproject.toml`, `examples/`, `src/`, and CLI `--help` or `python -c "import pkg; dir(pkg)"`) using your file-reading and terminal tools.
3. Verify every import path, class constructor, dataset column name, and method signature against the local cloned repo in `vendor/` before writing any test or implementation file.
4. Never guess or hallucinate a library call. If an API differs from expectations, follow the actual implementation inside `vendor/<repo>/`.

### Official Source Repositories to Clone & Inspect:
- `CuratorKIT`: https://github.com/Lexsi-Labs/CuratorKIT (Docs: https://curatorkit.lexsi.ai/ | https://lexsi-labs.github.io/CuratorKIT/)
- `CircuitKIT`: https://github.com/Lexsi-Labs/CircuitKIT (Docs: https://circuitkit.lexsi.ai/ | https://lexsi-labs.github.io/CircuitKIT/)
- `SafeTune`: https://github.com/Lexsi-Labs/SafeTune (Docs: https://safetune.lexsi.ai/)
- `AuditKIT`: https://github.com/Lexsi-Labs/AuditKIT (Docs: https://auditkit.lexsi.ai/)
- Target Base Model: `CohereLabs/tiny-aya-fire` (https://huggingface.co/CohereLabs/tiny-aya-fire)
- Reference Model Cards (for schema/reporting style):
  - https://huggingface.co/Lexsi/Gemma-4-31B-it-Multilingual-Safety-Repair
  - https://huggingface.co/Lexsi/Gemma-4-31B-it-Cybersecurity-Safety-Repair
