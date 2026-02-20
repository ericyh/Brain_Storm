from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, List, Type

from artifacts import ArtifactStore
from case import Case, CaseInput
from intake import Intake
from framing import Framer
from workplan import Workplanner
from synthesis import Synthesizer
from deliverables import DeliverableBuilder
from llm import LLMClient
from pods import DEFAULT_PODS
from qa import DEFAULT_QA


class ConsultingOrchestrator:
    """
    Full consulting-style lifecycle:
    Intake -> Framing -> Workplan -> Pods -> Synthesis -> QA -> Deliverables
    """

    def __init__(
        self,
        llm: LLMClient,
        pods: List[Type] = None,
        qa_checks: List[Type] = None,
        out_root: str = "runs",
    ):
        self.llm = llm
        self.pod_types = pods or DEFAULT_PODS
        self.qa_types = qa_checks or DEFAULT_QA
        self.out_root = out_root

    def run(self, case_id: str, inp: CaseInput) -> Dict[str, Any]:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(self.out_root, f"{case_id}_{ts}")
        store = ArtifactStore(run_dir=run_dir)

        case = Case(case_id=case_id, inp=inp)

        Intake().run(case)
        store.add("brief", {"brief": case.state.brief})

        Framer(self.llm).run(case)
        store.add("framing", case.state.framing)

        Workplanner(self.llm).run(case)
        store.add("workplan", case.state.workplan)

        for PodType in self.pod_types:
            pod = PodType(self.llm)
            out = pod.run(case)
            case.state.pod_outputs[pod.name] = out
            store.add(f"pod.{pod.name}", out)

        Synthesizer(self.llm).run(case)
        store.add("synthesis", case.state.synthesis)

        case.state.qa_reports = []
        for QType in self.qa_types:
            qc = QType(self.llm)
            rep = qc.run(case)
            case.state.qa_reports.append(rep)
            store.add(f"qa.{qc.name}", rep)

        DeliverableBuilder().run(case)
        store.add("deliverables", case.state.deliverables)

        # Write key artifacts as first-class files
        store.write_json("brief.json", {"brief": case.state.brief})
        store.write_json("framing.json", case.state.framing)
        store.write_json("workplan.json", case.state.workplan)
        store.write_json("pods.json", case.state.pod_outputs)
        store.write_json("synthesis.json", case.state.synthesis)
        store.write_json("qa.json", {"qa": case.state.qa_reports})
        store.write_json("deck_outline.json", case.state.deliverables.get("deck_outline", {}))
        store.write_text("run_flow.mmd", case.state.deliverables.get("mermaid_run_flow", ""))

        store.flush()

        return {
            "run_dir": run_dir,
            "brief": case.state.brief,
            "framing": case.state.framing,
            "workplan": case.state.workplan,
            "pods": case.state.pod_outputs,
            "synthesis": case.state.synthesis,
            "qa": case.state.qa_reports,
            "deliverables": case.state.deliverables,
        }