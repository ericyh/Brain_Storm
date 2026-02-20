from __future__ import annotations

from typing import Any, Dict

from pods.base import Pod
from schema import extract_json


class OpsPod(Pod):
    name = "ops"

    def run(self, case) -> Dict[str, Any]:
        system = """
You are an ops lead. Define an MVP operating model.
Output JSON ONLY:
{
  "mvp_scope":"string",
  "process_steps":["string","string","string"],
  "tools_stack":["string","string"],
  "headcount_plan":"string",
  "failure_modes":["string","string"]
}
""".strip()

        user = f"BRIEF:\n{case.state.brief}\n\nWORKPLAN:\n{case.state.workplan}"
        raw = self.llm.chat(system=system, user=user, temperature=0.5)
        return extract_json(raw)