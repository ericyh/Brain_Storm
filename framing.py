from __future__ import annotations

from typing import Any, Dict

from llm import LLMClient
from prompts import framing_system
from schema import extract_json
from case import Case


class Framer:
    """
    Produces: key question, success metrics, constraints, issue tree, hypotheses, data needed.
    """

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def run(self, case: Case) -> None:
        system = framing_system()
        user = f"BRIEF:\n{case.state.brief}\n\nFrame the case."
        raw = self.llm.chat(system=system, user=user, temperature=0.4)
        case.state.framing = extract_json(raw)