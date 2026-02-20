from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class Artifact:
    kind: str
    payload: Dict[str, Any]
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class ArtifactStore:
    """
    Writes audit-grade artifacts to disk:
    - inputs
    - framing
    - workplan
    - pod outputs
    - qa reports
    - synthesis
    - deliverables
    """

    def __init__(self, run_dir: str):
        self.run_dir = run_dir
        os.makedirs(self.run_dir, exist_ok=True)
        self._index: List[Artifact] = []

    def add(self, kind: str, payload: Dict[str, Any]) -> None:
        self._index.append(Artifact(kind=kind, payload=payload))

    def flush(self) -> None:
        idx = [asdict(a) for a in self._index]
        with open(os.path.join(self.run_dir, "index.json"), "w", encoding="utf-8") as f:
            json.dump(idx, f, ensure_ascii=False, indent=2)

    def write_json(self, filename: str, payload: Dict[str, Any]) -> None:
        with open(os.path.join(self.run_dir, filename), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def write_text(self, filename: str, text: str) -> None:
        with open(os.path.join(self.run_dir, filename), "w", encoding="utf-8") as f:
            f.write(text)