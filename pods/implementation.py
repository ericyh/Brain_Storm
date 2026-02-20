from __future__ import annotations

from typing import Any, Dict

from pods.base import Pod
from schema import extract_json


class ImplementationPod(Pod):
    name = "implementation"

    def run(self, case) -> Dict[str, Any]:
        system = """
You are an implementation lead. Produce a 30/60/90 day plan.
Output JSON ONLY:
{
  "milestones":{
    "day_30":["string","string"],
    "day_60":["string","string"],
    "day_90":["string","string"]
  },
  "metrics":["string","string"],
  "risks":["string","string"]
}
""".strip()

        user = f"BRIEF:\n{case.state.brief}\n\nPODS:\n{case.state.pod_outputs}"
        raw = self.llm.chat(system=system, user=user, temperature=0.4)
        return extract_json(raw)