from __future__ import annotations

from coven.models import Artifact
from .agent import SynthesizerResponse


class SynthesizerParser:
    """
    Applies the SynthesizerResponse back onto the target artifact.

    Writes the merged body into the artifact and attaches QC notes
    as metadata so downstream agents and the compiler can reference
    synthesis decisions if needed.
    """

    def parse(
        self,
        response: SynthesizerResponse,
        target_artifact: Artifact,
    ) -> Artifact:
        """
        Apply synthesized body and QC notes to the target artifact.

        Args:
            response: SynthesizerResponse from SynthesizerAgent.
            target_artifact: The artifact to update.

        Returns:
            Updated Artifact with merged body and qc_notes in body metadata.
        """
        merged_body = {
            **response.body,
            "__qc_notes__": response.qc_notes,
        }

        return target_artifact.model_copy(update={"body": merged_body})