from __future__ import annotations

from qa.base import QACheck
from prompts import qa_evidence_system
from schema import extract_json


class EvidenceQACheck(QACheck):
    name = "evidence"

    def run(self, case):
        system = qa_evidence_system()
        user = f"CLAIMS:\n{case.state.synthesis.get('claims')}\n\nASSUMPTIONS:\n{case.state.synthesis.get('assumptions')}"
        raw = self.llm.chat(system=system, user=user, temperature=0.2)
        out = extract_json(raw)
        out["check"] = self.name
        return out