from __future__ import annotations

from typing import Any, Dict

from pods.base import Pod
from schema import extract_json


class CompetitionPod(Pod):
    name = "competition"

    def run(self, case) -> Dict[str, Any]:
        system = """
You are a competitive strategist. Map competitors and differentiation.
Output JSON ONLY:
{
  "direct_competitors":["string","string"],
  "alternatives":["string","string"],
  "differentiation":"string",
  "moat_wedge":"string",
  "risks":["string","string"]
}
""".strip()

        user = f"BRIEF:\n{case.state.brief}\n\nFRAMING:\n{case.state.framing}"
        raw = self.llm.chat(system=system, user=user, temperature=0.5)
        return extract_json(raw)