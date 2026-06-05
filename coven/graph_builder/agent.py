from __future__ import annotations

from pathlib import Path

import instructor
from litellm import completion
from pydantic import BaseModel

from coven.models import Artifact, Node


# ── Instructor client ─────────────────────────────────────────────────────────

_client = instructor.from_litellm(completion)

# ── Prompt ────────────────────────────────────────────────────────────────────

_PROMPT_PATH = Path(__file__).parent.parent.parent / "prompts" / "graph_builder.txt"


def _load_prompt() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8")


# ── Response schema ───────────────────────────────────────────────────────────

class Edge(BaseModel):
    """A directed edge in the DAG from one node to another via an artifact."""
    from_node: str
    to_node: str
    artifact: str


class GraphBuilderResponse(BaseModel):
    nodes: list[dict]       # corrected node definitions
    artifacts: list[dict]   # corrected artifact definitions
    edges: list[Edge]       # explicit edge list derived from artifact wiring
    issues: list[str]       # repair log and synthesizer injection hints


# ── Agent ─────────────────────────────────────────────────────────────────────

class GraphBuilderAgent:
    """
    Stage 2 — Takes decomposed nodes and artifacts and constructs
    a validated DAG with explicit edges.

    Responsibilities:
    - Verify artifact contributor/user wiring consistency
    - Derive and formalize all edges
    - Detect and repair broken references
    - Flag artifacts needing synthesizer injection
    - Detect cycles
    """

    def __init__(self, model: str = "gpt-4o"):
        self.model = model
        self.system_prompt = _load_prompt()

    def _build_user_message(
        self,
        nodes: dict[str, Node],
        artifacts: dict[str, Artifact],
    ) -> str:
        import json

        payload = {
            "nodes": [node.model_dump() for node in nodes.values()],
            "artifacts": [artifact.model_dump() for artifact in artifacts.values()],
        }
        return json.dumps(payload, indent=2)

    def run(
        self,
        nodes: dict[str, Node],
        artifacts: dict[str, Artifact],
    ) -> GraphBuilderResponse:
        """
        Synchronous graph building call.

        Args:
            nodes: Node dict from DecomposerParser.
            artifacts: Artifact dict from DecomposerParser.

        Returns:
            GraphBuilderResponse with edges and any repair notes.
        """
        response: GraphBuilderResponse = _client.chat.completions.create(
            model=self.model,
            response_model=GraphBuilderResponse,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user",   "content": self._build_user_message(nodes, artifacts)},
            ],
        )
        return response

    async def arun(
        self,
        nodes: dict[str, Node],
        artifacts: dict[str, Artifact],
    ) -> GraphBuilderResponse:
        """
        Async graph building call.

        Args:
            nodes: Node dict from DecomposerParser.
            artifacts: Artifact dict from DecomposerParser.

        Returns:
            GraphBuilderResponse with edges and any repair notes.
        """
        response: GraphBuilderResponse = await _client.chat.completions.acreate(
            model=self.model,
            response_model=GraphBuilderResponse,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user",   "content": self._build_user_message(nodes, artifacts)},
            ],
        )
        return response