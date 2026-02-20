from __future__ import annotations

import json
from typing import Any, Dict

from case import Case


class DeliverableBuilder:
    """
    Produces:
    - a deck outline (slide-by-slide JSON)
    - mermaid diagrams for the run
    """

    def build_deck_outline(self, case: Case) -> Dict[str, Any]:
        syn = case.state.synthesis or {}
        recs = syn.get("recommendations", [])
        return {
            "title": "Brain Storm — Consulting Pack",
            "slides": [
                {"title": "Executive Summary", "bullets": [syn.get("executive_summary", "")]},
                {"title": "Problem Framing", "bullets": [syn.get("executive_summary", ""), "Key question + issue tree"]},
                {"title": "Market & Customer", "bullets": [json.dumps(case.state.pod_outputs.get("market", {}), ensure_ascii=False)]},
                {"title": "Economics", "bullets": [json.dumps(case.state.pod_outputs.get("economics", {}), ensure_ascii=False)]},
                {"title": "Competition", "bullets": [json.dumps(case.state.pod_outputs.get("competition", {}), ensure_ascii=False)]},
                {"title": "Operating Model", "bullets": [json.dumps(case.state.pod_outputs.get("ops", {}), ensure_ascii=False)]},
                {"title": "Implementation Plan", "bullets": [json.dumps(case.state.pod_outputs.get("implementation", {}), ensure_ascii=False)]},
                {"title": "Recommendations", "bullets": [json.dumps(recs, ensure_ascii=False)]},
                {"title": "Risks & Mitigations", "bullets": [json.dumps(case.state.qa_reports, ensure_ascii=False)]},
                {"title": "Appendix: Claims & Assumptions", "bullets": [json.dumps({"claims": syn.get("claims", []), "assumptions": syn.get("assumptions", [])}, ensure_ascii=False)]},
            ],
        }

    def build_mermaid_run_flow(self, case: Case) -> str:
        return f"""
flowchart TD
    A[Inputs: Query + Skills + Profile] --> B[Intake: Brief]
    B --> C[Framing: Issue Tree + Hypotheses]
    C --> D[Workplan]
    D --> E[Pods: Market/Econ/Comp/Ops/Impl]
    E --> F[Synthesis: Partner Output]
    F --> G[QA: Logic/Numbers/Evidence/Risk]
    G --> H[Deliverables: Deck Outline + Diagrams]
""".strip()

    def run(self, case: Case) -> None:
        case.state.deliverables = {
            "deck_outline": self.build_deck_outline(case),
            "mermaid_run_flow": self.build_mermaid_run_flow(case),
        }