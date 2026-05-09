from __future__ import annotations

import json
from pathlib import Path

import instructor
from litellm import completion
from pydantic import BaseModel

from loom.models import DAG, Artifact


# ── Instructor client ─────────────────────────────────────────────────────────

_client = instructor.from_litellm(completion)

# ── Prompt ────────────────────────────────────────────────────────────────────

_PROMPT_PATH = Path(__file__).parent.parent.parent / "prompts" / "compiler.txt"


def _load_prompt() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8")


# ── Response schema ───────────────────────────────────────────────────────────

class OutputSection(BaseModel):
    title: str
    content: str
    source_artifacts: list[str]


class CompilerResponse(BaseModel):
    title: str
    summary: str
    sections: list[OutputSection]
    recommendations: list[str]
    metadata: dict


# ── Agent ─────────────────────────────────────────────────────────────────────

class CompilerAgent:
    """
    Stage 5 — Takes all completed artifacts from the executed DAG
    and compiles them into a single coherent final output.

    Responsibilities:
    - Synthesize all artifacts into a unified response
    - Organize content into logical sections
    - Surface synthesis decisions and metadata
    - Directly address the user's original task
    """

    def __init__(self, model: str = "gpt-4o"):
        self.model = model
        self.system_prompt = _load_prompt()

    def _build_user_message(self, dag: DAG) -> str:
        """
        Build compiler context from the completed DAG.

        Includes the original task and all produced artifact bodies.
        Filters out partial artifacts (synthesizer intermediates)
        since their content is already merged into final artifacts.
        """
        final_artifacts = {
            name: artifact
            for name, artifact in dag.artifacts.items()
            if "__partial__" not in name
        }

        payload = {
            "task": dag.task,
            "artifacts": [
                {
                    "name": artifact.name,
                    "description": artifact.description,
                    "body": artifact.body,
                }
                for artifact in final_artifacts.values()
            ],
            "total_agents": len(dag.nodes),
            "total_artifacts": len(final_artifacts),
        }

        return json.dumps(payload, indent=2)

    def run(self, dag: DAG) -> CompilerResponse:
        """
        Synchronous compilation call.

        Args:
            dag: Fully executed DAG with all artifact bodies populated.

        Returns:
            CompilerResponse — the final structured output.
        """
        response: CompilerResponse = _client.chat.completions.create(
            model=self.model,
            response_model=CompilerResponse,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user",   "content": self._build_user_message(dag)},
            ],
        )
        return response

    async def arun(self, dag: DAG) -> CompilerResponse:
        """
        Async compilation call.

        Args:
            dag: Fully executed DAG with all artifact bodies populated.

        Returns:
            CompilerResponse — the final structured output.
        """
        response: CompilerResponse = await _client.chat.completions.acreate(
            model=self.model,
            response_model=CompilerResponse,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user",   "content": self._build_user_message(dag)},
            ],
        )
        return response