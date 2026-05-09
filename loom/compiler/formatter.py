from __future__ import annotations

from loom.models import DAG
from .agent import CompilerResponse


class CompilerFormatter:
    """
    Applies the CompilerResponse back onto the DAG as the final output.

    Converts the structured CompilerResponse into a clean dict
    stored in dag.final_output, ready for the caller to consume,
    serialize, or display.

    Also produces a plain text rendering for quick human reading.
    """

    def format(
        self,
        response: CompilerResponse,
        dag: DAG,
    ) -> DAG:
        """
        Write the compiled output into dag.final_output.

        Args:
            response: CompilerResponse from CompilerAgent.
            dag: The completed DAG.

        Returns:
            DAG with final_output populated.
        """
        final_output = {
            "title":           response.title,
            "summary":         response.summary,
            "sections":        [s.model_dump() for s in response.sections],
            "recommendations": response.recommendations,
            "metadata": {
                **response.metadata,
                "dag_id":            dag.id,
                "task":              dag.task,
                "total_agents":      len(dag.nodes),
                "artifacts_produced": len([
                    a for a in dag.artifacts
                    if "__partial__" not in a
                ]),
            },
        }

        return dag.model_copy(update={"final_output": final_output})

    def to_text(self, response: CompilerResponse) -> str:
        """
        Render the CompilerResponse as a plain text string.
        Useful for CLI output, logging, or simple display.

        Args:
            response: CompilerResponse from CompilerAgent.

        Returns:
            Formatted plain text string.
        """
        lines: list[str] = []

        lines.append("=" * 60)
        lines.append(response.title.upper())
        lines.append("=" * 60)
        lines.append("")
        lines.append(response.summary)
        lines.append("")

        for section in response.sections:
            lines.append(f"── {section.title} {'─' * (50 - len(section.title))}")
            lines.append(section.content)
            lines.append(
                f"  [sources: {', '.join(section.source_artifacts)}]"
            )
            lines.append("")

        if response.recommendations:
            lines.append("── Recommendations " + "─" * 41)
            for i, rec in enumerate(response.recommendations, 1):
                lines.append(f"  {i}. {rec}")
            lines.append("")

        meta = response.metadata
        lines.append("── Metadata " + "─" * 48)
        lines.append(f"  Agents: {meta.get('total_agents', '—')}")
        lines.append(f"  Artifacts: {meta.get('artifacts_produced', '—')}")

        if meta.get("synthesis_decisions"):
            lines.append("  Synthesis decisions:")
            for decision in meta["synthesis_decisions"]:
                lines.append(f"    • {decision}")

        lines.append("=" * 60)

        return "\n".join(lines)