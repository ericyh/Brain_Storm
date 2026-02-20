from __future__ import annotations

import argparse
import uuid

from case import CaseInput
from llm import LLMClient
from orchestrator import ConsultingOrchestrator


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", type=str, default="Generate 3 boring but profitable B2B business ideas I can build in 90 days.")
    ap.add_argument("--skills_file", type=str, default="")
    ap.add_argument("--extra", type=str, default="")
    ap.add_argument("--case_id", type=str, default="")
    args = ap.parse_args()

    skills_text = ""
    if args.skills_file:
        with open(args.skills_file, "r", encoding="utf-8") as f:
            skills_text = f.read()

    profile = {
        "location": "UK",
        "capital_available_gbp": 15000,
        "risk_tolerance": "moderate",
        "time_available_hours_per_week": 20,
        "preferences": ["boring-but-profitable", "B2B", "fast-to-revenue"],
    }

    inp = CaseInput(profile=profile, query=args.query, skills_text=skills_text, extra=args.extra)
    case_id = args.case_id.strip() or f"case_{uuid.uuid4().hex[:8]}"

    llm = LLMClient(models=["gemini/gemini-2.5-flash"])
    orch = ConsultingOrchestrator(llm=llm)

    out = orch.run(case_id=case_id, inp=inp)
    print("Wrote run artifacts to:", out["run_dir"])
    print("Executive summary:\n", out["synthesis"].get("executive_summary", ""))


if __name__ == "__main__":
    main()