from __future__ import annotations

import json
import re
from typing import Any, Dict

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def extract_json(text: str) -> Dict[str, Any]:
    m = _JSON_RE.search((text or "").strip())
    if not m:
        raise ValueError("No JSON found in model output.")
    return json.loads(m.group(0))


def safe_str(x: Any) -> str:
    return "" if x is None else str(x).strip()