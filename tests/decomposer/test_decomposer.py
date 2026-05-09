"""
tests/decomposer/test_parser.py

Unit tests for DecomposerParser.
No LLM calls — all inputs are hand-crafted DecomposerResponse objects.
"""

import pytest
from loom.decomposer.agent import DecomposerResponse, DecomposedNode, DecomposedArtifact
from loom.decomposer.parser import DecomposerParser
from loom.models import NodeType, NodeStatus


# ─────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────

def _make_response(nodes=None, artifacts=None):
    return DecomposerResponse(
        nodes=nodes or [],
        artifacts=artifacts or [],
    )


def _node(id, inputs=None, outputs=None, node_type="domain"):
    return DecomposedNode(
        id=id,
        name=id.replace("_", " ").title(),
        node_type=node_type,
        system_prompt=f"You are {id}.",
        query_tool=[],
        input_artifacts=inputs or [],
        output_artifacts=outputs or [],
    )


def _artifact(name, contributors=None, users=None):
    return DecomposedArtifact(
        name=name,
        description=f"Artifact {name}",
        contributors=contributors or [],
        users=users or [],
        body={},
    )


# ─────────────────────────────────────────────────────────────
# TESTS
# ─────────────────────────────────────────────────────────────

class TestDecomposerParser:

    def setup_method(self):
        self.parser = DecomposerParser()

    def test_parse_single_node_single_artifact(self):
        response = _make_response(
            nodes=[_node("agent_a", outputs=["art_1"])],
            artifacts=[_artifact("art_1", contributors=["agent_a"])],
        )
        nodes, artifacts = self.parser.parse(response)
        assert "agent_a" in nodes
        assert "art_1" in artifacts

    def test_parse_sets_node_status_pending(self):
        response = _make_response(
            nodes=[_node("agent_a", outputs=["art_1"])],
            artifacts=[_artifact("art_1", contributors=["agent_a"])],
        )
        nodes, _ = self.parser.parse(response)
        assert nodes["agent_a"].status == NodeStatus.PENDING

    def test_parse_node_type_domain(self):
        response = _make_response(
            nodes=[_node("agent_a", outputs=["art_1"])],
            artifacts=[_artifact("art_1", contributors=["agent_a"])],
        )
        nodes, _ = self.parser.parse(response)
        assert nodes["agent_a"].node_type == NodeType.DOMAIN

    def test_parse_multiple_nodes(self):
        response = _make_response(
            nodes=[
                _node("agent_a", outputs=["art_1"]),
                _node("agent_b", inputs=["art_1"], outputs=["art_2"]),
            ],
            artifacts=[
                _artifact("art_1", contributors=["agent_a"], users=["agent_b"]),
                _artifact("art_2", contributors=["agent_b"]),
            ],
        )
        nodes, artifacts = self.parser.parse(response)
        assert len(nodes) == 2
        assert len(artifacts) == 2

    def test_parse_artifacts_keyed_by_name(self):
        response = _make_response(
            nodes=[_node("agent_a", outputs=["my_artifact"])],
            artifacts=[_artifact("my_artifact", contributors=["agent_a"])],
        )
        _, artifacts = self.parser.parse(response)
        assert "my_artifact" in artifacts

    def test_validate_rejects_node_with_unknown_input_artifact(self):
        response = _make_response(
            nodes=[_node("agent_a", inputs=["nonexistent"], outputs=["art_1"])],
            artifacts=[_artifact("art_1", contributors=["agent_a"])],
        )
        with pytest.raises(ValueError, match="nonexistent"):
            self.parser.parse(response)

    def test_validate_rejects_node_with_unknown_output_artifact(self):
        response = _make_response(
            nodes=[_node("agent_a", outputs=["ghost_art"])],
            artifacts=[],
        )
        with pytest.raises(ValueError, match="ghost_art"):
            self.parser.parse(response)

    def test_validate_rejects_artifact_with_unknown_contributor(self):
        response = _make_response(
            nodes=[_node("agent_a", outputs=["art_1"])],
            artifacts=[_artifact("art_1", contributors=["ghost_node"])],
        )
        with pytest.raises(ValueError, match="ghost_node"):
            self.parser.parse(response)

    def test_validate_rejects_artifact_with_unknown_user(self):
        response = _make_response(
            nodes=[_node("agent_a", outputs=["art_1"])],
            artifacts=[_artifact("art_1", contributors=["agent_a"], users=["ghost_node"])],
        )
        with pytest.raises(ValueError, match="ghost_node"):
            self.parser.parse(response)

    def test_parse_artifacts_separately(self):
        raw = [_artifact("x", contributors=["a"])]
        # patch a fake node list to bypass validation
        response = _make_response(
            nodes=[_node("a", outputs=["x"])],
            artifacts=raw,
        )
        _, artifacts = self.parser.parse(response)
        assert artifacts["x"].description == "Artifact x"

    def test_parse_nodes_separately(self):
        response = _make_response(
            nodes=[_node("agent_z", outputs=["z_out"])],
            artifacts=[_artifact("z_out", contributors=["agent_z"])],
        )
        nodes, _ = self.parser.parse(response)
        assert nodes["agent_z"].system_prompt == "You are agent_z."

    def test_empty_response_parses_successfully(self):
        response = _make_response()
        nodes, artifacts = self.parser.parse(response)
        assert nodes == {}
        assert artifacts == {}