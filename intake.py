from __future__ import annotations

import json
from typing import Any, Dict

from case import Case


class Intake:
    """
    Intake stage: turns query + skills doc + profile into a single brief string.
    Stored in case.state.brief.
    """

    def build_brief(self, case: Case) -> str:
        profile: Dict[str, Any] = case.inp.profile or {}
        parts = []

        q = (case.inp.query or "").strip()
        if q:
            parts.append("USER_QUERY:\n" + q)

        skills = (case.inp.skills_text or "").strip()
        if skills:
            parts.append("SKILLS_DOC:\n" + skills)

        parts.append("PROFILE_JSON:\n" + json.dumps(profile, ensure_ascii=False, indent=2))

        extra = (case.inp.extra or "").strip()
        if extra:
            parts.append("EXTRA_CONTEXT:\n" + extra)

        parts.append(
            "BRIEFING_RULES:\n"
            "- Prefer feasible, cash-flow-positive ideas with identifiable buyers.\n"
            "- Avoid heavy liability and regulated advice unless explicitly requested.\n"
            "- Be concrete: buyer, pricing, operations, time-to-revenue.\n"
            "- If uncertain, narrow the niche and cut scope.\n"
        )

        return "\n\n".join(parts).strip()

    def run(self, case: Case) -> None:
        case.state.brief = self.build_brief(case)