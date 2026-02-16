from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Optional

from diagrams import write_diagrams

# imports from your existing file
from agents import GeneratorAgent, CriticAgent, critic_system_prompts


def read_text_file(path: Optional[str]) -> str:
    if not path:
        return ""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"skills file not found: {path}")
    # Keep it simple: md/txt. (If you want PDF support, we can add it properly.)
    return p.read_text(encoding="utf-8", errors="ignore").strip()


def build_brief(query: str, skills_text: str) -> str:
    parts = []
    if query.strip():
        parts.append("User query:\n" + query.strip())
    if skills_text.strip():
        parts.append("User skills/constraints:\n" + skills_text.strip())
    if not parts:
        parts.append("User query:\n(general)")

    parts.append(
        "\nInstructions:\n"
        "- Generate ideas the user can actually execute given the skills/constraints.\n"
        "- Keep ideas specific and operational.\n"
    )
    return "\n\n".join(parts).strip()


def mock_generator_output() -> str:
    # For testing without any API/auth
    return """1. Name: OpsBackoffice
2. What it is: A small service + toolkit that automates reconciliation and document chasing for tiny accounting firms.
3. How we extract money: Monthly retainer + one-time onboarding fee.
4. Step-by-step explanation of how it would actually operate:
   - Interview 10 small accounting firms; map the top 2 repetitive workflows
   - Build a simple upload + rules engine + notifications
   - Start as a done-for-you service, then productise the repeatable pieces

1. Name: ComplianceChase
2. What it is: A niche workflow tracker for SMEs to manage recurring compliance tasks (filings, renewals, certificates).
3. How we extract money: Per-seat subscription + templates pack.
4. Step-by-step explanation of how it would actually operate:
   - Pick one vertical with recurring paperwork
   - Build a minimal checklist + reminders + audit log
   - Sell via industry associations and accountants

1. Name: VendorProof
2. What it is: A lightweight vendor onboarding portal for small operators (collect docs, validate, store, remind).
3. How we extract money: Subscription tiers + paid setup for custom templates.
4. Step-by-step explanation of how it would actually operate:
   - Start with one niche (construction subcontractors / facilities vendors)
   - Offer a prebuilt “doc pack” and email chase automation
   - Expand with integrations after first 10 paying customers
"""


def ensure_run_dir(base_dir: str) -> Path:
    run_id = time.strftime("%Y%m%d_%H%M%S")
    out = Path(base_dir) / run_id
    out.mkdir(parents=True, exist_ok=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", default="", help="Optional query to steer generation")
    ap.add_argument("--skills", default=None, help="Optional path to a skills doc (.md/.txt)")
    ap.add_argument("--outdir", default="runs", help="Where to write run artifacts")
    ap.add_argument("--mock", action="store_true", help="Run without any model calls")
    args = ap.parse_args()

    skills_text = read_text_file(args.skills)
    brief = build_brief(args.query, skills_text)

    run_dir = ensure_run_dir(args.outdir)

    # Save the brief so your later diagrams / audits have traceability
    (run_dir / "brief.txt").write_text(brief + "\n", encoding="utf-8")

    # 1) Generate ideas (your existing agent)
    if args.mock:
        gen_output = mock_generator_output()
        (run_dir / "generator_output.txt").write_text(gen_output + "\n", encoding="utf-8")
        ideas_text = gen_output
    else:
        gen = GeneratorAgent()
        gen_output = gen.generate(brief)
        (run_dir / "generator_output.txt").write_text(gen_output + "\n", encoding="utf-8")
        ideas_text = gen_output

    (run_dir / "ideas_raw.txt").write_text(ideas_text + "\n", encoding="utf-8")

    # 2) Critique panel (your existing critic prompts)
    critiques_path = run_dir / "critiques.jsonl"
    with critiques_path.open("w", encoding="utf-8") as f:
        for c in critic_system_prompts:
            name = c.get("name", "critic")
            system_prompt = c["system_prompt"]

            if args.mock:
                critique_text = f"[mock critique] {name}: looks plausible, check numbers, validate ICP, reduce scope."
            else:
                critic = CriticAgent(system_prompt=system_prompt)
                critique_text = critic.generate(ideas_text)

            row = {
                "critic_name": name,
                "critique": critique_text,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # 3) Diagrams AFTER critiques (as requested)
    critic_names = [c.get("name", "critic") for c in critic_system_prompts]
    write_diagrams(run_dir, run_id=run_dir.name, critic_names=critic_names)


    print(f"Run complete: {run_dir}")
    print("Artifacts:")
    for fn in [
        "brief.txt",
        "generator_output.txt",
        "ideas_raw.txt",
        "critiques.jsonl",
        "pipeline.mmd",
        "run_flow.mmd",
        "pipeline.dot",
    ]:
        print(" -", run_dir / fn)


if __name__ == "__main__":
    main()
