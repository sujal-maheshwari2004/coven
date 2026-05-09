from __future__ import annotations

import json
from pathlib import Path

import instructor
from litellm import completion
from pydantic import BaseModel

from loom.models import Artifact, Node


# ── Instructor client ─────────────────────────────────────────────────────────

_client = instructor.from_litellm(completion)


# ── Response schema ───────────────────────────────────────────────────────────

class SynthesizerResponse(BaseModel):
    body: dict                  # merged artifact body
    qc_notes: list[str]         # conflict resolutions, gap fills, decisions made


# ── Agent ─────────────────────────────────────────────────────────────────────

class SynthesizerAgent:
    """
    Merges partial artifact contributions from multiple domain agents
    into a single coherent, quality-controlled artifact body.

    Receives:
    - The target artifact definition (name + description)
    - System prompts of all contributing agents
    - Partial artifact bodies from each contributor

    Produces:
    - A single merged artifact body
    - QC notes explaining synthesis decisions
    """

    def __init__(self, model: str = "gpt-4o"):
        self.model = model

    def _build_user_message(
        self,
        target_artifact: Artifact,
        partial_artifacts: list[Artifact],
        synth_node: Node,
    ) -> str:
        """
        Construct the full context message for the synthesizer LLM.
        Includes artifact definition, contributor intents, and all partial bodies.
        """
        payload = {
            "target_artifact": {
                "name": target_artifact.name,
                "description": target_artifact.description,
            },
            "contributor_perspectives": [
                {
                    "contributor_id": partial.contributors[0] if partial.contributors else "unknown",
                    "system_prompt": prompt,
                    "contribution": partial.body,
                }
                for partial, prompt in zip(
                    partial_artifacts,
                    synth_node.contributor_system_prompts
                )
            ],
        }
        return json.dumps(payload, indent=2)

    def run(
        self,
        target_artifact: Artifact,
        partial_artifacts: list[Artifact],
        synth_node: Node,
    ) -> SynthesizerResponse:
        """
        Synchronous synthesis call.

        Args:
            target_artifact: The artifact definition to produce.
            partial_artifacts: Partial bodies from each contributor.
            synth_node: The synthesizer Node carrying contributor system prompts.

        Returns:
            SynthesizerResponse with merged body and QC notes.
        """
        response: SynthesizerResponse = _client.chat.completions.create(
            model=self.model,
            response_model=SynthesizerResponse,
            messages=[
                {"role": "system", "content": synth_node.system_prompt},
                {"role": "user",   "content": self._build_user_message(
                    target_artifact, partial_artifacts, synth_node
                )},
            ],
        )
        return response

    async def arun(
        self,
        target_artifact: Artifact,
        partial_artifacts: list[Artifact],
        synth_node: Node,
    ) -> SynthesizerResponse:
        """
        Async synthesis call for use within the pipeline executor.

        Args:
            target_artifact: The artifact definition to produce.
            partial_artifacts: Partial bodies from each contributor.
            synth_node: The synthesizer Node carrying contributor system prompts.

        Returns:
            SynthesizerResponse with merged body and QC notes.
        """
        response: SynthesizerResponse = await _client.chat.completions.acreate(
            model=self.model,
            response_model=SynthesizerResponse,
            messages=[
                {"role": "system", "content": synth_node.system_prompt},
                {"role": "user",   "content": self._build_user_message(
                    target_artifact, partial_artifacts, synth_node
                )},
            ],
        )
        return response