from __future__ import annotations

from llm import LLMClient
from prompts import synthesis_system
from schema import extract_json
from case import Case


class Synthesizer:
    """
    Partner-style synthesis using brief + framing + pods.
    Produces: executive summary, recommendations, assumptions, claims.
    """

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def run(self, case: Case) -> None:
        system = synthesis_system()
        user = (
            "BRIEF:\n"
            f"{case.state.brief}\n\n"
            "FRAMING:\n"
            f"{case.state.framing}\n\n"
            "POD_OUTPUTS:\n"
            f"{case.state.pod_outputs}\n\n"
            "Produce synthesis."
        )
        raw = self.llm.chat(system=system, user=user, temperature=0.35)
        case.state.synthesis = extract_json(raw)