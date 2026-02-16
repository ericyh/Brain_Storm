from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional


def _mmd_header(title: str) -> str:
    # Mermaid init lets you set theme + spacing.
    # "neutral" is clean; you can switch to "base" if you want more contrast.
    return f"""%%{{init: {{
  "theme": "neutral",
  "flowchart": {{
    "curve": "basis",
    "nodeSpacing": 40,
    "rankSpacing": 55
  }},
  "themeVariables": {{
    "fontFamily": "Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial",
    "fontSize": "14px",
    "primaryTextColor": "#111827",
    "lineColor": "#6B7280"
  }}
}}}}%%
%% {title}
"""


def build_pipeline_mmd() -> str:
    return _mmd_header("Idea Generator Pipeline") + r"""
flowchart LR

%% ---------- Inputs ----------
subgraph S0["Inputs"]
direction TB
Q["Query (optional)"]:::input
U["Skill docs upload (optional)"]:::input
end

%% ---------- Layer 0 ----------
subgraph S1["Layer 0 — Brief compiler"]
direction TB
B["Brief\n(query + skills + constraints)"]:::process
end

%% ---------- Layer 1 ----------
subgraph S2["Layer 1 — Idea generation"]
direction TB
G["GeneratorAgent x N\n(persona-conditioned)"]:::process
I["Ideas (raw text)"]:::artifact
end

%% ---------- Layer 2 ----------
subgraph S3["Layer 2 — Critique panel"]
direction TB
R["Critic router\n(assign lenses)"]:::process
C["CriticAgent x M\n(market, unit econ, feasibility, legal, competition)"]:::process
X["Critiques (structured-ish)"]:::artifact
end

%% ---------- Outputs ----------
subgraph S4["Outputs"]
direction TB
A["Run artifacts\nbrief.txt, ideas_raw.txt, critiques.jsonl"]:::artifact
D["Diagrams\npipeline.mmd, run_flow.mmd, pipeline.dot"]:::artifact
end

%% ---------- Edges ----------
Q --> B
U --> B

B --> G --> I --> R --> C --> X --> A --> D

%% ---------- Styles ----------
classDef input fill:#EEF2FF,stroke:#6366F1,stroke-width:1px,color:#111827;
classDef process fill:#ECFDF5,stroke:#10B981,stroke-width:1px,color:#111827;
classDef artifact fill:#F9FAFB,stroke:#6B7280,stroke-width:1px,color:#111827;

%% ---------- Nice labels ----------
linkStyle default stroke-width:1.2px;
"""


def build_run_flow_mmd(run_id: str, critic_names: Iterable[str]) -> str:
    # This diagram emphasizes traceability and shows per-critic fan-out.
    critic_nodes = []
    critic_edges = []
    i = 0
    for name in critic_names:
        i += 1
        safe = _slug(name)
        critic_nodes.append(f'  C_{safe}["{name} critic"]:::process')
        critic_edges.append(f"  IDEAS --> C_{safe} --> CRITS")

    critics_block = "\n".join(critic_nodes + critic_edges)

    return _mmd_header(f"Run Flow — {run_id}") + f"""
flowchart LR

subgraph RUN["Run: {run_id}"]
direction LR

BRIEF["brief.txt"]:::artifact
GEN["generator_output.txt"]:::artifact
IDEAS["ideas_raw.txt"]:::artifact
CRITS["critiques.jsonl"]:::artifact

BRIEF --> GEN --> IDEAS

subgraph PANEL["Critique panel (fan-out)"]
direction TB
{critics_block}
end

CRITS --> DIAGS["pipeline.mmd\\nrun_flow.mmd\\npipeline.dot"]:::artifact
end

classDef process fill:#ECFDF5,stroke:#10B981,stroke-width:1px,color:#111827;
classDef artifact fill:#F9FAFB,stroke:#6B7280,stroke-width:1px,color:#111827;

linkStyle default stroke-width:1.2px;
"""


def build_pipeline_dot() -> str:
    # Graphviz DOT with clusters + mild styling.
    return r"""
digraph Pipeline {
  rankdir=LR;
  splines=true;
  nodesep=0.35;
  ranksep=0.55;
  fontname="Inter";
  fontsize=12;

  node [shape=box, style="rounded", fontname="Inter", fontsize=11];
  edge [color="#6B7280", penwidth=1.2];

  subgraph cluster_inputs {
    label="Inputs";
    color="#6366F1";
    style="rounded";
    Q [label="Query (optional)"];
    U [label="Skill docs upload (optional)"];
  }

  subgraph cluster_brief {
    label="Layer 0 — Brief compiler";
    color="#10B981";
    style="rounded";
    B [label="Brief"];
  }

  subgraph cluster_gen {
    label="Layer 1 — Idea generation";
    color="#10B981";
    style="rounded";
    G [label="GeneratorAgent x N"];
    I [label="Ideas (raw text)", shape=note];
  }

  subgraph cluster_crit {
    label="Layer 2 — Critique panel";
    color="#10B981";
    style="rounded";
    R [label="Critic router"];
    C [label="CriticAgent x M"];
    X [label="Critiques", shape=note];
  }

  subgraph cluster_out {
    label="Outputs";
    color="#6B7280";
    style="rounded";
    A [label="Run artifacts", shape=note];
    D [label="Diagrams", shape=note];
  }

  Q -> B;
  U -> B;
  B -> G -> I -> R -> C -> X -> A -> D;
}
"""


def write_diagrams(run_dir: str | Path, run_id: str, critic_names: Optional[Iterable[str]] = None) -> None:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    critic_names = list(critic_names or [])

    (run_dir / "pipeline.mmd").write_text(build_pipeline_mmd(), encoding="utf-8")
    (run_dir / "run_flow.mmd").write_text(build_run_flow_mmd(run_id, critic_names), encoding="utf-8")
    (run_dir / "pipeline.dot").write_text(build_pipeline_dot(), encoding="utf-8")


def _slug(s: str) -> str:
    out = []
    for ch in s.lower():
        if ch.isalnum():
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_")
