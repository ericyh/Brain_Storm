from __future__ import annotations

import os
import random
import time
from typing import Any, Dict, List, Optional

from litellm import completion

try:
    from dotenv import load_dotenv  # type: ignore
except Exception:
    load_dotenv = None  # type: ignore


class LLMClient:
    """
    Minimal LLM wrapper for LiteLLM + Gemini API key.
    - Reads GOOGLE_API_KEY / GEMINI_API_KEY from env (.env supported)
    - Retries with backoff
    """

    def __init__(
        self,
        models: Optional[List[str]] = None,
        max_retries: int = 3,
        backoff_base_s: float = 1.4,
        seed: int = 7,
    ):
        if load_dotenv is not None:
            load_dotenv()

        self.models = models or ["gemini/gemini-2.5-flash"]
        self.max_retries = int(max_retries)
        self.backoff_base_s = float(backoff_base_s)
        self.rng = random.Random(seed)

        # Hard fail early with a helpful message.
        if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
            raise RuntimeError(
                "Missing Gemini API key. Set GOOGLE_API_KEY (recommended) in your environment or .env."
            )

    def _sleep(self, attempt: int) -> None:
        time.sleep((self.backoff_base_s**attempt) + self.rng.random() * 0.25)

    def chat(
        self,
        system: str,
        user: str,
        temperature: float = 0.6,
        model: Optional[str] = None,
    ) -> str:
        last_err: Optional[Exception] = None
        chosen = model or self.rng.choice(self.models)

        for attempt in range(1, self.max_retries + 1):
            try:
                resp = completion(
                    model=chosen,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=temperature,
                )
                return resp.choices[0].message.content
            except Exception as e:
                last_err = e
                self._sleep(attempt)

        raise RuntimeError(f"LLM call failed after {self.max_retries} retries: {last_err}") from last_err