"""
tests/compiler/test_compiler.py

Unit tests for CompilerFormatter.
CompilerAgent LLM calls are not tested here (covered by integration tests).
"""

import pytest
from loom.compiler.agent import CompilerResponse, OutputSection
from loom.compiler.formatter import CompilerFormatter
from loom.models import DAG, DAGStatus, Node, NodeType, Artifact


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def _make_response(
    title="Test Report",
    summary="This is a summary.",
    sections=None,
    recommendations=None,
    metadata=None,
):
    return CompilerResponse(
        title=title,
        summary=summary,
        sections=sections or [
            OutputSection(
                title="Section One",
                content="Content of section one.",
                source_artifacts=["artifact_a"],
            )
        ],
        recommendations=recommendations or [],
        metadata=metadata or {},
    )


def _make_dag(node_ids=None, artifact_names=None):
    nodes = {
        id: Node(
            id=id,
            name=id.title(),
            node_type=NodeType.DOMAIN,
            system_prompt="...",
        )
        for id in (node_ids or ["agent_a"])
    }
    artifacts = {
        name: Artifact(name=name, description=f"Artifact {name}", body={"x": 1})
        for name in (artifact_names or ["artifact_a"])
    }
    return DAG(
        id="run_test",
        task="test task",
        nodes=nodes,
        artifacts=artifacts,
    )


# ─────────────────────────────────────────────────────────────
# FORMATTER TESTS
# ─────────────────────────────────────────────────────────────

class TestCompilerFormatter:

    def setup_method(self):
        self.formatter = CompilerFormatter()

    def test_format_writes_title(self):
        response = _make_response(title="My Report")
        dag = _make_dag()
        result = self.formatter.format(response, dag)
        assert result.final_output["title"] == "My Report"

    def test_format_writes_summary(self):
        response = _make_response(summary="Executive summary here.")
        dag = _make_dag()
        result = self.formatter.format(response, dag)
        assert result.final_output["summary"] == "Executive summary here."

    def test_format_writes_sections(self):
        response = _make_response(
            sections=[
                OutputSection(title="A", content="Content A", source_artifacts=["art_a"]),
                OutputSection(title="B", content="Content B", source_artifacts=["art_b"]),
            ]
        )
        dag = _make_dag()
        result = self.formatter.format(response, dag)
        assert len(result.final_output["sections"]) == 2
        assert result.final_output["sections"][0]["title"] == "A"

    def test_format_writes_recommendations(self):
        response = _make_response(recommendations=["Do X", "Do Y"])
        dag = _make_dag()
        result = self.formatter.format(response, dag)
        assert result.final_output["recommendations"] == ["Do X", "Do Y"]

    def test_format_metadata_includes_dag_id(self):
        response = _make_response()
        dag = _make_dag()
        result = self.formatter.format(response, dag)
        assert result.final_output["metadata"]["dag_id"] == "run_test"

    def test_format_metadata_includes_task(self):
        response = _make_response()
        dag = _make_dag()
        result = self.formatter.format(response, dag)
        assert result.final_output["metadata"]["task"] == "test task"

    def test_format_metadata_total_agents(self):
        response = _make_response()
        dag = _make_dag(node_ids=["a", "b", "c"])
        result = self.formatter.format(response, dag)
        assert result.final_output["metadata"]["total_agents"] == 3

    def test_format_excludes_partial_artifacts_from_count(self):
        response = _make_response()
        dag = _make_dag(artifact_names=["real_art", "real_art__partial__agent_a"])
        result = self.formatter.format(response, dag)
        # Only real_art should count — partial should be excluded
        assert result.final_output["metadata"]["artifacts_produced"] == 1

    def test_format_does_not_mutate_input_dag(self):
        response = _make_response()
        dag = _make_dag()
        original_final_output = dict(dag.final_output)
        self.formatter.format(response, dag)
        assert dag.final_output == original_final_output

    def test_format_returns_dag_with_final_output(self):
        response = _make_response()
        dag = _make_dag()
        result = self.formatter.format(response, dag)
        assert result.final_output != {}

    # ── to_text tests ─────────────────────────────────────────

    def test_to_text_contains_title(self):
        response = _make_response(title="My Great Report")
        text = self.formatter.to_text(response)
        assert "MY GREAT REPORT" in text

    def test_to_text_contains_summary(self):
        response = _make_response(summary="The summary text.")
        text = self.formatter.to_text(response)
        assert "The summary text." in text

    def test_to_text_contains_section_title(self):
        response = _make_response(
            sections=[OutputSection(title="Key Findings", content="...", source_artifacts=[])]
        )
        text = self.formatter.to_text(response)
        assert "Key Findings" in text

    def test_to_text_contains_section_content(self):
        response = _make_response(
            sections=[OutputSection(title="S", content="Important content here.", source_artifacts=[])]
        )
        text = self.formatter.to_text(response)
        assert "Important content here." in text

    def test_to_text_contains_recommendations(self):
        response = _make_response(recommendations=["Implement feature X", "Reduce cost Y"])
        text = self.formatter.to_text(response)
        assert "Implement feature X" in text
        assert "Reduce cost Y" in text

    def test_to_text_contains_metadata(self):
        response = _make_response(metadata={"total_agents": 5, "artifacts_produced": 3})
        text = self.formatter.to_text(response)
        assert "Agents:" in text

    def test_to_text_no_recommendations_skips_section(self):
        response = _make_response(recommendations=[])
        text = self.formatter.to_text(response)
        assert "Recommendations" not in text

    def test_to_text_returns_string(self):
        response = _make_response()
        text = self.formatter.to_text(response)
        assert isinstance(text, str)
        assert len(text) > 0