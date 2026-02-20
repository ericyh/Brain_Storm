from __future__ import annotations

from qa.base import QACheck
from prompts import qa_risk_system
from schema import extract_json


class RiskQACheck(QACheck):
    name = "risk"

    def run(self, case):
        system = qa_risk_system()
        user = f"BRIEF:\n{case.state.brief}\n\nPODS:\n{case.state.pod_outputs}\n\nSYNTHESIS:\n{case.state.synthesis}"
        raw = self.llm.chat(system=system, user=user, temperature=0.2)
        out = extract_json(raw)
        out["check"] = self.name
        return out