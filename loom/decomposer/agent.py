from __future__ import annotations

import json
from pathlib import Path

import instructor
import litellm
from litellm import completion
from pydantic import BaseModel

from loom.models import Artifact, Node, NodeType


# ── Instructor client wrapping litellm ────────────────────────────────────────

_client = instructor.from_litellm(completion)

# ── Prompt ────────────────────────────────────────────────────────────────────

_PROMPT_PATH = Path(__file__).parent.parent.parent / "prompts" / "decomposer.txt"


def _load_prompt() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8")


# ── Response schema ───────────────────────────────────────────────────────────

class DecomposedArtifact(BaseModel):
    name: str
    description: str
    contributors: list[str]
    users: list[str]
    body: dict = {}


class DecomposedNode(BaseModel):
    id: str
    name: str
    node_type: NodeType
    system_prompt: str
    query_tool: dict = {}
    input_artifacts: list[str]
    output_artifacts: list[str]


class DecomposerResponse(BaseModel):
    nodes: list[DecomposedNode]
    artifacts: list[DecomposedArtifact]


# ── Agent ─────────────────────────────────────────────────────────────────────

class DecomposerAgent:
    """
    Stage 1 — Takes a raw complex task and returns a list of
    domain agent nodes and artifacts via a single LLM call.

    Does not build the DAG — that is the graph builder's job.
    Only decomposes the task into its constituent parts.
    """

    def __init__(self, model: str = "gpt-4o"):
        self.model = model
        self.system_prompt = _load_prompt()

    def run(self, task: str) -> DecomposerResponse:
        """
        Synchronous decomposition call.

        Args:
            task: The raw complex task string from the user.

        Returns:
            DecomposerResponse containing nodes and artifacts.
        """
        response: DecomposerResponse = _client.chat.completions.create(
            model=self.model,
            response_model=DecomposerResponse,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user",   "content": f"Task: {task}"},
            ],
        )
        return response

    async def arun(self, task: str) -> DecomposerResponse:
        """
        Async decomposition call for use within the pipeline executor.

        Args:
            task: The raw complex task string from the user.

        Returns:
            DecomposerResponse containing nodes and artifacts.
        """
        response: DecomposerResponse = await _client.chat.completions.acreate(
            model=self.model,
            response_model=DecomposerResponse,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user",   "content": f"Task: {task}"},
            ],
        )
        return response