from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class CaseInput:
    profile: Dict[str, Any]
    query: str = ""
    skills_text: str = ""
    extra: str = ""


@dataclass
class CaseState:
    brief: str = ""
    framing: Dict[str, Any] = field(default_factory=dict)
    workplan: Dict[str, Any] = field(default_factory=dict)
    pod_outputs: Dict[str, Any] = field(default_factory=dict)
    qa_reports: List[Dict[str, Any]] = field(default_factory=list)
    synthesis: Dict[str, Any] = field(default_factory=dict)
    deliverables: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Case:
    case_id: str
    inp: CaseInput
    state: CaseState = field(default_factory=CaseState)