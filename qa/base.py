from __future__ import annotations

from typing import Any, Dict

from case import Case
from llm import LLMClient


class QACheck:
    name = "base"

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def run(self, case: Case) -> Dict[str, Any]:
        raise NotImplementedError