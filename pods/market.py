from __future__ import annotations

from typing import Any, Dict

from pods.base import Pod
from schema import extract_json


class MarketPod(Pod):
    name = "market"

    def run(self, case) -> Dict[str, Any]:
        system = """
You are a market analyst. Produce a crisp market view.
Output JSON ONLY:
{
  "icp":"string",
  "buyer":"string",
  "demand_signals":["string","string"],
  "tamtoms":"string",
  "channels":["string","string"],
  "risks":["string","string"]
}
""".strip()

        user = f"BRIEF:\n{case.state.brief}\n\nFRAMING:\n{case.state.framing}"
        raw = self.llm.chat(system=system, user=user, temperature=0.5)
        return extract_json(raw)