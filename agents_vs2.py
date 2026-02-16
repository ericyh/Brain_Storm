from __future__ import annotations

import json
import random
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from litellm import completion

try:
    from datasets import load_dataset  # type: ignore
except Exception:
    load_dataset = None  # type: ignore


# ============================
# Config + retry settings
# ============================

MODELS = [
    # "openai/gpt-4o",
    "vertex_ai/gemini-2.5-pro",
]
DEFAULT_MODEL = MODELS[0] if MODELS else "vertex_ai/gemini-2.5-pro"
MAX_RETRIES = 3
BACKOFF_BASE_S = 1.4


# ============================
# PersonaSource: streams NVIDIA Nemotron personas safely
# ============================

class PersonaSource:
    def __init__(self, seed: int = 7, buffer_size: int = 10_000):
        self.seed = seed
        self.buffer_size = buffer_size
        self._rng = random.Random(seed)
        self._iter = None

    def _init_iter(self) -> None:
        if load_dataset is None:
            self._iter = None
            return
        ds = load_dataset("nvidia/Nemotron-Personas-USA", split="train", streaming=True)
        ds = ds.shuffle(seed=self._rng.randint(1, 10_000), buffer_size=self.buffer_size)
        self._iter = iter(ds)

    def next(self) -> Optional[Dict[str, Any]]:
        if self._iter is None:
            try:
                self._init_iter()
            except Exception:
                self._iter = None
                return None

        if self._iter is None:
            return None

        try:
            row = next(self._iter)
            return dict(row)
        except Exception:
            try:
                self._init_iter()
                row = next(self._iter)  # type: ignore
                return dict(row)
            except Exception:
                return None


# ============================
# LLM call wrapper with retry/backoff
# ============================

def _sleep_backoff(attempt: int) -> None:
    time.sleep((BACKOFF_BASE_S**attempt) + random.random() * 0.25)


def _call_llm(
    model: str,
    system: str,
    user: str,
    temperature: float = 0.8,
    reasoning_effort: Optional[str] = None,
    max_retries: int = MAX_RETRIES,
) -> str:
    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            kwargs: Dict[str, Any] = {}
            if reasoning_effort is not None:
                kwargs["reasoning_effort"] = reasoning_effort

            resp = completion(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
                **kwargs,
            )
            return resp.choices[0].message.content
        except Exception as e:
            last_err = e
            _sleep_backoff(attempt)

    raise RuntimeError(f"LLM call failed after {max_retries} retries: {last_err}") from last_err


# ============================
# JSON extraction + repair pass
# ============================

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text: str) -> Dict[str, Any]:
    m = _JSON_RE.search((text or "").strip())
    if not m:
        raise ValueError("No JSON object found in output.")
    return json.loads(m.group(0))


def _repair_json(model: str, broken: str) -> Dict[str, Any]:
    system = "You fix JSON. Return valid JSON ONLY. No markdown. No commentary."
    user = "Fix the following so it is valid JSON and matches the requested schema:\n\n" + broken
    fixed = _call_llm(model=model, system=system, user=user, temperature=0.0, max_retries=2)
    return _extract_json(fixed)


def _json_or_repair(model: str, text: str) -> Dict[str, Any]:
    try:
        return _extract_json(text)
    except Exception:
        return _repair_json(model, text)


def _safe_list(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(i).strip() for i in x if str(i).strip()]
    return [str(x).strip()]


# ============================
# Data structures for ideas + critiques
# ============================

@dataclass
class Idea:
    idea_id: str
    name: str
    what_it_is: str
    how_it_makes_money: str
    operating_steps: List[str]
    target_customer: str = ""
    why_it_works: str = ""
    demand_signal: str = ""
    competitive_landscape: str = ""
    feasibility_notes: str = ""
    unit_econ_sketch: str = ""
    risks: List[str] = None
    tags: List[str] = None
    persona: Optional[Dict[str, Any]] = None
    worker_id: Optional[str] = None
    model: Optional[str] = None
    raw: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "idea_id": self.idea_id,
            "name": self.name,
            "what_it_is": self.what_it_is,
            "how_it_makes_money": self.how_it_makes_money,
            "operating_steps": self.operating_steps,
            "target_customer": self.target_customer,
            "why_it_works": self.why_it_works,
            "demand_signal": self.demand_signal,
            "competitive_landscape": self.competitive_landscape,
            "feasibility_notes": self.feasibility_notes,
            "unit_econ_sketch": self.unit_econ_sketch,
            "risks": self.risks or [],
            "tags": self.tags or [],
            "worker_id": self.worker_id,
            "model": self.model,
        }


@dataclass
class Critique:
    critique_id: str
    idea_id: str
    critic_name: str
    score: float
    verdict: str
    summary: str
    fatal_flags: List[str]
    improvements: List[str]
    assumptions_to_validate: List[str]
    model: Optional[str] = None
    raw: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "critique_id": self.critique_id,
            "idea_id": self.idea_id,
            "critic_name": self.critic_name,
            "score": self.score,
            "verdict": self.verdict,
            "summary": self.summary,
            "fatal_flags": self.fatal_flags,
            "improvements": self.improvements,
            "assumptions_to_validate": self.assumptions_to_validate,
            "model": self.model,
        }


# ============================
# Worker + critic prompts (your style, structured outputs)
# ============================

WORKER_SYSTEM_PROMPT = """
You are a pragmatic, highly analytical entrepreneur.

Generate ONE highly specialised, innovative but realistic business idea for the user.

Process deeply (but output only JSON):
- Why this would realistically work
- Whether demand is strong and identifiable
- Operational feasibility
- Competitive landscape
- Basic economic viability
- What problem it solves, and for whom

Focus on:
- Boring but profitable businesses (import/export, niche manufacturing, supply chain arbitrage, B2B services, regulatory gaps, product adaptation).
- Cross-industry transfer of an existing solution to solve a specific problem.
- Underserved niches.
- Buildable within 12–36 months.
- Extremely specific ideas.

Avoid:
- Generic SaaS dashboards
- Vague AI platforms
- Overly speculative moonshots

Output STRICT JSON ONLY (no markdown, no extra keys):

{
  "name": "string",
  "target_customer": "string",
  "what_it_is": "string",
  "how_it_makes_money": "string",
  "operating_steps": ["string", "string", "string"],
  "why_it_works": "string",
  "demand_signal": "string",
  "competitive_landscape": "string",
  "feasibility_notes": "string",
  "unit_econ_sketch": "string",
  "risks": ["string", "string"],
  "tags": ["string", "string"]
}
""".strip()


CRITIC_JSON_SCHEMA = """
Return STRICT JSON ONLY (no markdown, no extra keys):

{
  "score": 0-10,
  "verdict": "advance" | "revise" | "archive",
  "summary": "string",
  "fatal_flags": ["string"],
  "improvements": ["string", "string", "string"],
  "assumptions_to_validate": ["string", "string", "string"]
}

Rules:
- fatal_flags are issues that must be fixed, otherwise archive.
- Be conservative and grounded.
""".strip()


critic_system_prompts: List[Dict[str, str]] = [
    {"name": "Market Sizing Researcher", "system_prompt": """You are a highly analytical unit economics expert with deep operational experience in traditional, cash-flow-oriented businesses.

Define the core economic unit, estimate revenue/unit, costs/unit, gross margin, contribution margin, CAC/LTV/payback, working capital, and stress-test assumptions.
Be conservative and explicit about assumptions. Use realistic ranges.
"""},
    {"name": "Unit Economics Researcher", "system_prompt": """You are a unit economics critic. Evaluate CAC, servicing costs, churn risk, margin resilience, and payback.
Be numerical where possible and list key sensitivities.
"""},
    {"name": "Product Feasibility Critic", "system_prompt": """You are a product feasibility analyst. Assess build scope, dependencies, time-to-MVP, operational constraints, and key risks.
Suggest changes to reduce complexity and increase feasibility.
"""},
    {"name": "Law and Compliance Skeptic", "system_prompt": """You are a legal/compliance skeptic. Identify applicable regulations, privacy risks, liabilities, and operational compliance requirements.
Provide mitigations and severity priorities.
"""},
    {"name": "Competitive Strategist", "system_prompt": """You are a competitive strategist. Identify competitors/substitutes, barriers, differentiation, and realistic defensibility.
Give actionable competitive moves.
"""},
]

_extra_critics = [
    ("Distribution Realist", "You are a go-to-market operator. Pressure test acquisition channels, sales motion, and buyer reachability. Kill hand-wavy distribution."),
    ("Pricing & Willingness-To-Pay", "You are a pricing strategist. Evaluate pricing power, who pays, realistic price points, and packaging."),
    ("Retention & Stickiness", "You are a retention critic. Identify stickiness, churn drivers, and switching costs."),
    ("Operational Load", "You are an operations lead. Estimate support burden, manual work, and firefighting risks."),
    ("Scope Cutter", "You are a ruthless PM. Cut scope to a narrow MVP and identify bloat."),
    ("Time-to-Revenue", "You are revenue-first. Estimate fastest path to first £/$ and what must be true."),
    ("Founder Fit", "You evaluate founder-fit. Check match to skills/constraints and propose modifications."),
    ("Data & Input Risk", "You challenge data assumptions. Identify data availability/quality risks and mitigations."),
    ("Trust & Abuse", "You review trust/safety. Identify abuse modes and necessary guardrails."),
    ("Moat & Wedge", "You evaluate defensibility. Check wedge strategy and sustainable advantage."),
    ("Partnership Leverage", "You identify realistic channel partners and whether partnerships are plausible."),
    ("Implementation Complexity", "You are a senior engineer. Assess integration complexity, edge cases, maintenance and reliability."),
    ("Working Capital Risk", "You identify cash traps: payment terms, inventory, receivables, financing risk."),
    ("Regulatory Practicality", "You focus on operational compliance: policies, logs, audits, procedures."),
    ("Competitive Entry Response", "Assume an incumbent responds. How do they kill it and what counter-move prevents it?"),
]
for name, sys in _extra_critics:
    critic_system_prompts.append({"name": name, "system_prompt": sys})
critic_system_prompts = critic_system_prompts[:20]


SUPERVISOR_SYSTEM_PROMPT = """
You are the Supervisor Agent coordinating a multi-agent idea generation system.

Filter ideas to fit the user's profile, skills, constraints, and query. Use critic feedback to rank and decide.
Prefer feasible, cash-flow-positive ideas with identifiable buyers.

Return STRICT JSON ONLY:

{
  "shortlist": [
    {
      "idea_id": "string",
      "decision": "advance" | "revise" | "archive",
      "overall_score": 0-10,
      "rationale": "string",
      "next_actions": ["string", "string", "string"]
    }
  ],
  "notes": "string"
}

No markdown. No extra keys.
""".strip()


# ============================
# Legacy agents (keep your current notebooks working)
# ============================

class GeneratorAgent:
    def __init__(self, seed: int = 7):
        self._rng = random.Random(seed)
        self.persona_source = PersonaSource(seed=seed)
        self.persona = self.persona_source.next()
        self.model = self._rng.choice(MODELS) if MODELS else DEFAULT_MODEL

        self.system_prompt = f"""
You are a pragmatic, highly analytical entrepreneur with the following persona:
{json.dumps(self.persona)}

Your task:
Generate 3 new, highly specialised, innovative but realistic business ideas for the user.

Important:
Process the following requirements deeply, but only provide the final structured output for the 3 ideas.
- Why this would realistically work
- Whether demand is strong and identifiable
- Operational feasibility
- Competitive landscape
- Basic economic viability
- What problem does it specifically solve, and for whom?

Focus on:
- “Boring” but profitable businesses (import/export, niche manufacturing, supply chain arbitrage, B2B services, regulatory gaps, product adaptation across industries).
- Taking an existing idea / solution from one industry and applying it to another to solve a specific problem.
- Underserved niche markets.
- Businesses that could realistically be built within 12–36 months.
- Ideas that are extremely specific

Avoid:
- Generic SaaS dashboards
- Vague AI platforms
- Overly speculative moonshots

For each idea, provide:
1. Name
2. What it is (clear and concrete)
3. How we extract money
4. Step-by-step explanation of how it would actually operate

Be detailed, practical, and grounded in reality.
Prefer operational clarity over flashy creativity.
""".strip()

        self.reasoning_content = None
        self.content = None

    def generate(self, prompt: str) -> str:
        resp = completion(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            reasoning_effort="high",
            temperature=1.2,
        )
        try:
            self.reasoning_content = resp.choices[0].message.reasoning_content
        except Exception:
            self.reasoning_content = None
        self.content = resp.choices[0].message.content
        return self.content


class CriticAgent:
    def __init__(self, system_prompt: str, seed: int = 7):
        self.system_prompt = system_prompt
        self._rng = random.Random(seed)

    def generate(self, prompt: str) -> str:
        model = self._rng.choice(MODELS) if MODELS else DEFAULT_MODEL
        return _call_llm(
            model=model,
            system=self.system_prompt,
            user=f"Business idea to analyse:\n{prompt}",
            temperature=0.8,
            reasoning_effort=None,
        )


# ============================
# WorkerAgent + PanelCritic: structured generation/critique
# ============================

@dataclass
class WorkerAgent:
    worker_id: str
    persona: Optional[Dict[str, Any]]
    model: str

    def generate_one(self, brief: str) -> Idea:
        user = f"""
USER_PROFILE_AND_BRIEF:
{brief}

PERSONA (sampled from NVIDIA Nemotron personas):
{json.dumps(self.persona) if self.persona else "(persona unavailable)"}

Generate ONE idea. Output STRICT JSON only.
""".strip()

        raw = _call_llm(
            model=self.model,
            system=WORKER_SYSTEM_PROMPT,
            user=user,
            temperature=1.0,
            reasoning_effort="high",
            max_retries=MAX_RETRIES,
        )
        data = _json_or_repair(self.model, raw)

        idea_id = f"idea_{uuid.uuid4().hex[:10]}"
        return Idea(
            idea_id=idea_id,
            name=str(data.get("name", "")).strip() or f"Idea {idea_id}",
            target_customer=str(data.get("target_customer", "")).strip(),
            what_it_is=str(data.get("what_it_is", "")).strip(),
            how_it_makes_money=str(data.get("how_it_makes_money", "")).strip(),
            operating_steps=_safe_list(data.get("operating_steps")),
            why_it_works=str(data.get("why_it_works", "")).strip(),
            demand_signal=str(data.get("demand_signal", "")).strip(),
            competitive_landscape=str(data.get("competitive_landscape", "")).strip(),
            feasibility_notes=str(data.get("feasibility_notes", "")).strip(),
            unit_econ_sketch=str(data.get("unit_econ_sketch", "")).strip(),
            risks=_safe_list(data.get("risks")),
            tags=_safe_list(data.get("tags")),
            persona=self.persona,
            worker_id=self.worker_id,
            model=self.model,
            raw=raw,
        )


@dataclass
class PanelCritic:
    critic_name: str
    system_prompt: str
    model: str

    def critique(self, brief: str, idea: Idea) -> Critique:
        user = f"""
USER_PROFILE_AND_BRIEF:
{brief}

IDEA (JSON):
{json.dumps(idea.to_dict(), ensure_ascii=False)}

{CRITIC_JSON_SCHEMA}
""".strip()

        raw = _call_llm(
            model=self.model,
            system=self.system_prompt,
            user=user,
            temperature=0.6,
            reasoning_effort=None,
            max_retries=MAX_RETRIES,
        )
        data = _json_or_repair(self.model, raw)

        score = data.get("score", 0)
        try:
            score_f = float(score)
        except Exception:
            score_f = 0.0
        score_f = max(0.0, min(10.0, score_f))

        verdict = str(data.get("verdict", "revise")).strip().lower()
        if verdict not in ("advance", "revise", "archive"):
            verdict = "revise"

        return Critique(
            critique_id=f"crit_{uuid.uuid4().hex[:10]}",
            idea_id=idea.idea_id,
            critic_name=self.critic_name,
            score=score_f,
            verdict=verdict,
            summary=str(data.get("summary", "")).strip(),
            fatal_flags=_safe_list(data.get("fatal_flags")),
            improvements=_safe_list(data.get("improvements")),
            assumptions_to_validate=_safe_list(data.get("assumptions_to_validate")),
            model=self.model,
            raw=raw,
        )


# ============================
# Dedupe: removes near-duplicates cheaply without embeddings
# ============================

def _normalize(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def dedupe_ideas(ideas: List[Idea]) -> List[Idea]:
    kept: List[Idea] = []
    seen = set()

    for idea in ideas:
        key = (_normalize(idea.name), _normalize(idea.target_customer))
        if key in seen:
            continue

        too_close = False
        for k in kept:
            a = set(_normalize(idea.what_it_is).split())
            b = set(_normalize(k.what_it_is).split())
            if a and b:
                j = len(a & b) / max(1, len(a | b))
                if j >= 0.72 and _normalize(idea.target_customer) == _normalize(k.target_customer):
                    too_close = True
                    break

        if too_close:
            continue

        seen.add(key)
        kept.append(idea)

    return kept


# ============================
# SupervisorAgent: orchestrates workers -> critics -> shortlist decisions
# ============================

class SupervisorAgent:
    def __init__(
        self,
        worker_count: int = 100,
        critic_count: int = 20,
        seed: int = 7,
        model: Optional[str] = None,
        persona_seed: int = 7,
    ):
        self.worker_count = int(worker_count)
        self.critic_count = int(critic_count)
        self.seed = seed
        self._rng = random.Random(seed)
        self.model = model or (self._rng.choice(MODELS) if MODELS else DEFAULT_MODEL)
        self.personas = PersonaSource(seed=persona_seed)
        self.critic_defs = critic_system_prompts[: self.critic_count]

    def build_brief(
        self,
        profile: Dict[str, Any],
        query: str = "",
        skills_text: str = "",
        extra: str = "",
    ) -> str:
        parts = []
        if query.strip():
            parts.append("USER_QUERY:\n" + query.strip())
        if skills_text.strip():
            parts.append("SKILLS_DOC:\n" + skills_text.strip())
        parts.append("PROFILE_JSON:\n" + json.dumps(profile or {}, ensure_ascii=False, indent=2))
        if extra.strip():
            parts.append("EXTRA_CONTEXT:\n" + extra.strip())
        parts.append(
            "BRIEFING_RULES:\n"
            "- Prefer feasible, cash-flow-positive ideas with identifiable buyers.\n"
            "- Respect constraints (time/budget/risk/compliance).\n"
            "- If uncertain, narrow the niche and simplify ops.\n"
        )
        return "\n\n".join(parts).strip()

    def run(
        self,
        profile: Dict[str, Any],
        query: str = "",
        skills_text: str = "",
        extra: str = "",
        top_k: int = 10,
        max_workers: Optional[int] = None,
        max_critics: Optional[int] = None,
    ) -> Dict[str, Any]:
        n_workers = min(self.worker_count, int(max_workers)) if max_workers else self.worker_count
        n_critics = min(self.critic_count, int(max_critics)) if max_critics else self.critic_count

        brief = self.build_brief(profile=profile, query=query, skills_text=skills_text, extra=extra)

        workers: List[WorkerAgent] = []
        for i in range(n_workers):
            workers.append(
                WorkerAgent(
                    worker_id=f"worker_{i+1:03d}",
                    persona=self.personas.next(),
                    model=self.model,
                )
            )

        ideas: List[Idea] = []
        for w in workers:
            try:
                ideas.append(w.generate_one(brief))
            except Exception:
                continue

        ideas = dedupe_ideas(ideas)

        critics: List[PanelCritic] = []
        for c in self.critic_defs[:n_critics]:
            critics.append(
                PanelCritic(
                    critic_name=c["name"],
                    system_prompt=c["system_prompt"],
                    model=self.model,
                )
            )

        critiques: List[Critique] = []
        for idea in ideas:
            for critic in critics:
                try:
                    critiques.append(critic.critique(brief, idea))
                except Exception:
                    continue

        aggregate = self._aggregate(ideas, critiques)
        shortlist = self._final_shortlist(brief, aggregate, top_k=top_k)

        return {
            "brief": brief,
            "ideas": [i.to_dict() for i in ideas],
            "critiques": [c.to_dict() for c in critiques],
            "aggregate": aggregate,
            "shortlist": shortlist,
        }

    def _aggregate(self, ideas: List[Idea], critiques: List[Critique]) -> List[Dict[str, Any]]:
        by_idea: Dict[str, List[Critique]] = {}
        for c in critiques:
            by_idea.setdefault(c.idea_id, []).append(c)

        rows: List[Dict[str, Any]] = []
        for idea in ideas:
            cs = by_idea.get(idea.idea_id, [])
            if cs:
                avg = sum(x.score for x in cs) / max(1, len(cs))
                fatals = sorted({f for x in cs for f in (x.fatal_flags or [])})
                archive_votes = sum(1 for x in cs if x.verdict == "archive")
            else:
                avg, fatals, archive_votes = 0.0, [], 0

            rows.append(
                {
                    "idea": idea.to_dict(),
                    "avg_score": round(avg, 2),
                    "critic_count": len(cs),
                    "fatal_flags": fatals,
                    "archive_votes": archive_votes,
                }
            )

        rows.sort(key=lambda r: (len(r["fatal_flags"]), r["archive_votes"], -r["avg_score"]))
        return rows

    def _final_shortlist(self, brief: str, aggregate: List[Dict[str, Any]], top_k: int) -> Dict[str, Any]:
        candidates = aggregate[: min(30, len(aggregate))]
        user = (
            "USER_PROFILE_AND_BRIEF:\n"
            f"{brief}\n\n"
            "CANDIDATES (top aggregated):\n"
            f"{json.dumps(candidates, ensure_ascii=False)}\n\n"
            f"Pick up to {top_k} ideas.\n"
            "Return STRICT JSON only (schema in system prompt)."
        )
        raw = _call_llm(
            model=self.model,
            system=SUPERVISOR_SYSTEM_PROMPT,
            user=user,
            temperature=0.5,
            reasoning_effort="high",
            max_retries=MAX_RETRIES,
        )
        return _json_or_repair(self.model, raw)


# ============================
# Convenience helper for a one-shot supervised run
# ============================

def run_supervised_generation(
    profile: Dict[str, Any],
    query: str = "",
    skills_text: str = "",
    extra: str = "",
    worker_count: int = 100,
    critic_count: int = 20,
    top_k: int = 10,
    seed: int = 7,
) -> Dict[str, Any]:
    sup = SupervisorAgent(
        worker_count=worker_count,
        critic_count=critic_count,
        seed=seed,
        persona_seed=seed,
    )
    return sup.run(profile=profile, query=query, skills_text=skills_text, extra=extra, top_k=top_k)
