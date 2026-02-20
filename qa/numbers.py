from __future__ import annotations

from qa.base import QACheck
from prompts import qa_numbers_system
from schema import extract_json


class NumbersQACheck(QACheck):
    name = "numbers"

    def run(self, case):
        system = qa_numbers_system()
        user = f"ECONOMICS:\n{case.state.pod_outputs.get('economics')}\n\nSYNTHESIS:\n{case.state.synthesis}"
        raw = self.llm.chat(system=system, user=user, temperature=0.2)
        out = extract_json(raw)
        out["check"] = self.name
        return out