"""
tests/synthesizer/test_synthesizer.py

Unit tests for SynthesizerInjector and SynthesizerParser.
No LLM calls.
"""

import pytest
import networkx as nx

from coven.synthesizer.injector import SynthesizerInjector
from coven.synthesizer.parser import SynthesizerParser
from coven.synthesizer.agent import SynthesizerResponse
from coven.models import Node, NodeType, NodeStatus, Artifact


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def _make_node(id, outputs=None, inputs=None):
    return Node(
        id=id,
        name=id.title(),
        node_type=NodeType.DOMAIN,
        system_prompt=f"System prompt for {id}.",
        input_artifacts=inputs or [],
        output_artifacts=outputs or [],
    )


def _make_artifact(name, contributors=None, users=None):
    return Artifact(
        name=name,
        description=f"Artifact {name}",
        contributors=contributors or [],
        users=users or [],
    )


def _build_graph(nodes, edges):
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    return G


# ─────────────────────────────────────────────────────────────
# SYNTHESIZER INJECTOR TESTS
# ─────────────────────────────────────────────────────────────

class TestSynthesizerInjector:

    def setup_method(self):
        self.injector = SynthesizerInjector()

    def _make_two_contributor_setup(self):
        nodes = {
            "agent_a": _make_node("agent_a", outputs=["shared_art"]),
            "agent_b": _make_node("agent_b", outputs=["shared_art"]),
            "agent_c": _make_node("agent_c", inputs=["shared_art"]),
        }
        artifacts = {
            "shared_art": _make_artifact(
                "shared_art",
                contributors=["agent_a", "agent_b"],
                users=["agent_c"],
            )
        }
        G = _build_graph(
            nodes=list(nodes.keys()),
            edges=[("agent_a", "agent_c"), ("agent_b", "agent_c")],
        )
        return nodes, artifacts, G

    def test_inject_creates_synthesizer_node(self):
        nodes, artifacts, G = self._make_two_contributor_setup()
        nodes, artifacts, G = self.injector.inject(
            nodes, artifacts, ["shared_art"], G
        )
        assert "synthesizer_shared_art" in nodes

    def test_synthesizer_node_type(self):
        nodes, artifacts, G = self._make_two_contributor_setup()
        nodes, artifacts, G = self.injector.inject(
            nodes, artifacts, ["shared_art"], G
        )
        synth = nodes["synthesizer_shared_art"]
        assert synth.node_type == NodeType.SYNTHESIZER

    def test_synthesizer_has_contributor_system_prompts(self):
        nodes, artifacts, G = self._make_two_contributor_setup()
        nodes, artifacts, G = self.injector.inject(
            nodes, artifacts, ["shared_art"], G
        )
        synth = nodes["synthesizer_shared_art"]
        assert len(synth.contributor_system_prompts) == 2
        assert "System prompt for agent_a." in synth.contributor_system_prompts
        assert "System prompt for agent_b." in synth.contributor_system_prompts

    def test_partial_artifacts_created(self):
        nodes, artifacts, G = self._make_two_contributor_setup()
        nodes, artifacts, G = self.injector.inject(
            nodes, artifacts, ["shared_art"], G
        )
        assert "shared_art__partial__agent_a" in artifacts
        assert "shared_art__partial__agent_b" in artifacts

    def test_partial_artifacts_have_correct_contributor(self):
        nodes, artifacts, G = self._make_two_contributor_setup()
        nodes, artifacts, G = self.injector.inject(
            nodes, artifacts, ["shared_art"], G
        )
        partial_a = artifacts["shared_art__partial__agent_a"]
        assert partial_a.contributors == ["agent_a"]
        assert partial_a.users == ["synthesizer_shared_art"]

    def test_original_artifact_contributor_updated_to_synthesizer(self):
        nodes, artifacts, G = self._make_two_contributor_setup()
        nodes, artifacts, G = self.injector.inject(
            nodes, artifacts, ["shared_art"], G
        )
        assert artifacts["shared_art"].contributors == ["synthesizer_shared_art"]

    def test_contributor_outputs_updated_to_partial(self):
        nodes, artifacts, G = self._make_two_contributor_setup()
        nodes, artifacts, G = self.injector.inject(
            nodes, artifacts, ["shared_art"], G
        )
        # agent_a's output should now be the partial, not the original
        assert "shared_art__partial__agent_a" in nodes["agent_a"].output_artifacts
        assert "shared_art" not in nodes["agent_a"].output_artifacts

    def test_synthesizer_in_graph(self):
        nodes, artifacts, G = self._make_two_contributor_setup()
        nodes, artifacts, G = self.injector.inject(
            nodes, artifacts, ["shared_art"], G
        )
        assert "synthesizer_shared_art" in G.nodes

    def test_synthesizer_input_output_artifacts(self):
        nodes, artifacts, G = self._make_two_contributor_setup()
        nodes, artifacts, G = self.injector.inject(
            nodes, artifacts, ["shared_art"], G
        )
        synth = nodes["synthesizer_shared_art"]
        assert "shared_art__partial__agent_a" in synth.input_artifacts
        assert "shared_art__partial__agent_b" in synth.input_artifacts
        assert synth.output_artifacts == ["shared_art"]

    def test_skip_single_contributor_artifact(self):
        nodes = {
            "agent_a": _make_node("agent_a", outputs=["art_1"]),
            "agent_b": _make_node("agent_b", inputs=["art_1"]),
        }
        artifacts = {
            "art_1": _make_artifact("art_1", contributors=["agent_a"], users=["agent_b"]),
        }
        G = _build_graph(["agent_a", "agent_b"], [("agent_a", "agent_b")])
        original_node_count = len(nodes)

        nodes, artifacts, G = self.injector.inject(nodes, artifacts, ["art_1"], G)

        # No synthesizer should be injected — only one contributor
        assert len(nodes) == original_node_count

    def test_skip_unknown_artifact_name(self):
        nodes = {"agent_a": _make_node("agent_a", outputs=["art_1"])}
        artifacts = {"art_1": _make_artifact("art_1", contributors=["agent_a"])}
        G = _build_graph(["agent_a"], [])
        original_node_count = len(nodes)

        nodes, artifacts, G = self.injector.inject(
            nodes, artifacts, ["nonexistent_artifact"], G
        )
        assert len(nodes) == original_node_count


# ─────────────────────────────────────────────────────────────
# SYNTHESIZER PARSER TESTS
# ─────────────────────────────────────────────────────────────

class TestSynthesizerParser:

    def setup_method(self):
        self.parser = SynthesizerParser()

    def _make_target_artifact(self):
        return Artifact(
            name="merged_artifact",
            description="Merged output from multiple contributors",
            contributors=["synthesizer_merged_artifact"],
            users=["compiler"],
        )

    def test_parse_applies_body(self):
        response = SynthesizerResponse(
            body={"key": "value", "score": 42},
            qc_notes=[],
        )
        target = self._make_target_artifact()
        result = self.parser.parse(response, target)
        assert result.body["key"] == "value"
        assert result.body["score"] == 42

    def test_parse_injects_qc_notes(self):
        response = SynthesizerResponse(
            body={"key": "value"},
            qc_notes=["Resolved conflict between A and B", "Filled gap in section 2"],
        )
        target = self._make_target_artifact()
        result = self.parser.parse(response, target)
        assert "__qc_notes__" in result.body
        assert len(result.body["__qc_notes__"]) == 2

    def test_parse_preserves_artifact_metadata(self):
        response = SynthesizerResponse(body={"x": 1}, qc_notes=[])
        target = self._make_target_artifact()
        result = self.parser.parse(response, target)
        assert result.name == "merged_artifact"
        assert result.description == "Merged output from multiple contributors"

    def test_parse_empty_body(self):
        response = SynthesizerResponse(body={}, qc_notes=["Note 1"])
        target = self._make_target_artifact()
        result = self.parser.parse(response, target)
        assert result.body["__qc_notes__"] == ["Note 1"]

    def test_parse_does_not_mutate_original(self):
        response = SynthesizerResponse(body={"x": 1}, qc_notes=[])
        target = self._make_target_artifact()
        original_body = dict(target.body)
        self.parser.parse(response, target)
        assert target.body == original_body  # original unchanged