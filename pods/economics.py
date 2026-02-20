from __future__ import annotations

from typing import Any, Dict

from pods.base import Pod
from schema import extract_json


class EconomicsPod(Pod):
    name = "economics"

    def run(self, case) -> Dict[str, Any]:
        system = """
You are a unit economics operator. Propose a realistic pricing + cost stack.
Output JSON ONLY:
{
  "pricing_model":"string",
  "price_points":["string","string"],
  "unit":"string",
  "revenue_unit_calc":"string",
  "cost_drivers":["string","string","string"],
  "margin_notes":"string",
  "cashflow_risks":["string","string"]
}
""".strip()

        user = f"BRIEF:\n{case.state.brief}\n\nWORKPLAN:\n{case.state.workplan}"
        raw = self.llm.chat(system=system, user=user, temperature=0.4)
        return extract_json(raw)