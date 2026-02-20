from __future__ import annotations

from qa.base import QACheck
from prompts import qa_logic_system
from schema import extract_json


class LogicQACheck(QACheck):
    name = "logic"

    def run(self, case):
        system = qa_logic_system()
        user = f"FRAMING:\n{case.state.framing}\n\nSYNTHESIS_DRAFT:\n{case.state.synthesis}"
        raw = self.llm.chat(system=system, user=user, temperature=0.2)
        out = extract_json(raw)
        out["check"] = self.name
        return out