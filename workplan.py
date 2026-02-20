from __future__ import annotations

from llm import LLMClient
from prompts import workplan_system
from schema import extract_json
from case import Case


class Workplanner:
    """
    Turns framing into an execution plan: workstreams, tasks, critical path, risks.
    """

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def run(self, case: Case) -> None:
        system = workplan_system()
        user = (
            "BRIEF:\n"
            f"{case.state.brief}\n\n"
            "FRAMING_JSON:\n"
            f"{case.state.framing}\n\n"
            "Generate a workplan."
        )
        raw = self.llm.chat(system=system, user=user, temperature=0.4)
        case.state.workplan = extract_json(raw)