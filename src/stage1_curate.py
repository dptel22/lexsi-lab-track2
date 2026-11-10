"""Stage 1: Dataset Curation & CuratorKIT Provenance Engine for CircuitKavach-Aya."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence


@dataclass
class ContrastPair:
    pair_id: str
    language: str
    threat_category: str
    keyword: str
    harmful: str
    benign: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ContrastPair:
        return cls(
            pair_id=data["pair_id"],
            language=data["language"],
            threat_category=data["threat_category"],
            keyword=data["keyword"],
            harmful=data["harmful"],
            benign=data["benign"],
        )


def save_jsonl(pairs: Sequence[ContrastPair], file_path: str | Path) -> Path:
    out_path = Path(file_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair.to_dict(), ensure_ascii=False) + "\n")
    return out_path


def load_jsonl(file_path: str | Path) -> list[ContrastPair]:
    in_path = Path(file_path)
    pairs: list[ContrastPair] = []
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                pairs.append(ContrastPair.from_dict(json.loads(line_str)))
    return pairs
