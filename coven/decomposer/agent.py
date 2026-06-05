from __future__ import annotations

from pathlib import Path

import instructor
from litellm import completion
from pydantic import BaseModel

from coven.models.node import ToolQuery


# ── Instructor client ─────────────────────────────────────────────────────────

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
    node_type: str
    system_prompt: str
    # FIX: list[ToolQuery] not dict — must match Node.query_tool type
    query_tool: list[ToolQuery] = []
    input_artifacts: list[str] = []
    output_artifacts: list[str] = []


class DecomposerResponse(BaseModel):
    nodes: list[DecomposedNode]
    artifacts: list[DecomposedArtifact]


# ── Agent ─────────────────────────────────────────────────────────────────────

class DecomposerAgent:
    """
    Stage 1 — Takes a raw complex task and returns a list of
    domain agent nodes and artifacts via a single LLM call.
    """

    def __init__(self, model: str = "gpt-4o"):
        self.model = model
        self.system_prompt = _load_prompt()

    def run(self, task: str) -> DecomposerResponse:
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
        response: DecomposerResponse = await _client.chat.completions.acreate(
            model=self.model,
            response_model=DecomposerResponse,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user",   "content": f"Task: {task}"},
            ],
        )
        return response