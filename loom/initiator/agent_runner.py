from __future__ import annotations

import json
from pathlib import Path

import instructor
from litellm import completion
from pydantic import BaseModel

from loom.models import Node, NodeType, NodeStatus, Artifact
from .artifact_store import ArtifactStore
from loom.synthesizer import SynthesizerAgent, SynthesizerParser


# ── Instructor client ─────────────────────────────────────────────────────────

_client = instructor.from_litellm(completion)


# ── Domain agent response schema ──────────────────────────────────────────────

class DomainAgentResponse(BaseModel):
    """
    Structured output for a domain agent.
    Each output artifact gets its own body keyed by artifact name.
    """
    outputs: dict[str, dict]   # artifact_name → artifact body


# ── Agent Runner ──────────────────────────────────────────────────────────────

class AgentRunner:
    """
    Executes a single node in the DAG.

    Handles two node types:
    - DOMAIN: standard LLM call with input artifacts as context
    - SYNTHESIZER: delegates to SynthesizerAgent with partial artifacts

    After execution:
    - Writes all output artifact bodies to the ArtifactStore
    - Updates the node status to COMPLETED or FAILED
    """

    def __init__(self, model: str = "gpt-4o"):
        self.model = model
        self._synth_agent = SynthesizerAgent(model=model)
        self._synth_parser = SynthesizerParser()

    async def run(
        self,
        node: Node,
        store: ArtifactStore,
        nodes: dict[str, Node],
    ) -> Node:
        """
        Execute a single node asynchronously.

        Args:
            node: The node to execute.
            store: Shared artifact store.
            nodes: Full node registry (for synthesizer context).

        Returns:
            Updated Node with status set to COMPLETED or FAILED.
        """
        try:
            if node.node_type == NodeType.SYNTHESIZER:
                await self._run_synthesizer(node, store)
            else:
                await self._run_domain(node, store)

            return node.model_copy(update={"status": NodeStatus.COMPLETED})

        except Exception as e:
            return node.model_copy(
                update={
                    "status": NodeStatus.FAILED,
                    "result": {"error": str(e)},
                }
            )

    async def _run_domain(self, node: Node, store: ArtifactStore) -> None:
        """
        Execute a domain agent node.

        Builds context from input artifacts, calls the LLM,
        and writes each output artifact body to the store.
        """
        input_artifacts = await store.get_many(node.input_artifacts)
        user_message    = self._build_domain_message(node, input_artifacts)

        response: DomainAgentResponse = await _client.chat.completions.acreate(
            model=self.model,
            response_model=DomainAgentResponse,
            messages=[
                {"role": "system", "content": node.system_prompt},
                {"role": "user",   "content": user_message},
            ],
        )

        for artifact_name, body in response.outputs.items():
            if artifact_name in node.output_artifacts:
                await store.put(artifact_name, body)

    async def _run_synthesizer(self, node: Node, store: ArtifactStore) -> None:
        """
        Execute a synthesizer node.

        Retrieves all partial artifacts, delegates to SynthesizerAgent,
        and writes the merged artifact body to the store.
        """
        partial_artifacts  = await store.get_many(node.input_artifacts)
        target_artifact    = await store.get(node.output_artifacts[0])

        response = await self._synth_agent.arun(
            target_artifact=target_artifact,
            partial_artifacts=partial_artifacts,
            synth_node=node,
        )

        merged_artifact = self._synth_parser.parse(response, target_artifact)
        await store.put(target_artifact.name, merged_artifact.body)

    def _build_domain_message(
        self,
        node: Node,
        input_artifacts: list[Artifact],
    ) -> str:
        """
        Build the user message for a domain agent.

        Includes:
        - The node's task (derived from output artifact descriptions)
        - All input artifact bodies as context
        - Expected output artifact names and descriptions
        """
        payload = {
            "input_artifacts": [
                {
                    "name": artifact.name,
                    "description": artifact.description,
                    "body": artifact.body,
                }
                for artifact in input_artifacts
            ],
            "output_artifacts": node.output_artifacts,
            "instructions": (
                "Using the input artifacts provided as context, "
                "produce the required output artifacts. "
                "Return each output artifact body keyed by its artifact name "
                "inside the 'outputs' field."
            ),
        }
        return json.dumps(payload, indent=2)